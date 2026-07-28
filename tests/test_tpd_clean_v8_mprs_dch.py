from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from model.tpd import PhaseKeepBlock, SPDPatchEmbedding
from model.tpd_clean_v7_dch import TPDCleanV7DCHBlock


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MODEL_PATH = REPO_ROOT / "model" / "tpd_clean_v8_mprs_dch.py"
PROTOTYPE_MODEL_PATH = REPO_ROOT / "tpd_clean_v8_mprs_dch.py"


def _load_v8_module():
    """Prefer the production module and temporarily support the root prototype."""

    if FORMAL_MODEL_PATH.is_file():
        return importlib.import_module("model.tpd_clean_v8_mprs_dch")
    if not PROTOTYPE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Neither the formal V8 module nor the root V8 prototype exists"
        )
    module_name = "_tpd_clean_v8_mprs_dch_test_prototype"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROTOTYPE_MODEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V8 prototype from {PROTOTYPE_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


V8 = _load_v8_module()


def _require(condition: bool, message: str) -> None:
    """Assertion helper that remains active when Python is run with ``-O``."""

    if not condition:
        raise AssertionError(message)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    message: str,
) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=message,
    )


def _assert_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    message: str,
) -> None:
    _require(
        torch.equal(actual, expected),
        (
            f"{message}: tensors are not bitwise equal; "
            f"max_abs={(actual - expected).abs().max().item():.9g}"
        ),
    )


def _manual_phases(x: torch.Tensor) -> torch.Tensor:
    """Return TL/TR/BL/BR phase tensors without using PixelUnshuffle."""

    return torch.stack(
        (
            x[..., 0::2, 0::2],
            x[..., 0::2, 1::2],
            x[..., 1::2, 0::2],
            x[..., 1::2, 1::2],
        ),
        dim=2,
    )


def _image_from_manual_phases(phases: torch.Tensor) -> torch.Tensor:
    batch, channels, phase_count, height, width = phases.shape
    _require(phase_count == 4, f"expected four phases, got {phase_count}")
    output = phases.new_empty(batch, channels, 2 * height, 2 * width)
    output[..., 0::2, 0::2] = phases[:, :, 0]
    output[..., 0::2, 1::2] = phases[:, :, 1]
    output[..., 1::2, 0::2] = phases[:, :, 2]
    output[..., 1::2, 1::2] = phases[:, :, 3]
    return output


def _manual_phase_saliency(x: torch.Tensor) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    phases = _manual_phases(x).float()
    context = phases.mean(dim=2)
    scalar_saliency = phases.amax(dim=2) - context
    phase_saliency = (
        scalar_saliency.unsqueeze(2)
        + (phases - context.unsqueeze(2)) / 3.0
    )
    return phases, context, phase_saliency


def _independent_forward(
    block: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute V8 directly from four image slices, not V8 helper methods."""

    phases, context, phase_saliency = _manual_phase_saliency(x)
    batch, channels, _, height, width = phases.shape
    weight = block.phase_compress.weight.float()
    bias = block.phase_compress.bias
    if bias is None:
        raise AssertionError("the V7-compatible Keep projection requires bias")

    rearranged = phases.reshape(batch, 4 * channels, height, width)
    keep = F.conv2d(rearranged, weight, bias.float())
    phase_weight = weight.reshape(
        weight.shape[0],
        channels,
        4,
        1,
        1,
    )
    tied_weight = phase_weight.sum(dim=2)
    saliency_aligned = F.conv2d(
        phase_saliency.reshape(batch, 4 * channels, height, width),
        weight,
        bias=None,
    )
    context_aligned = F.conv2d(
        context,
        tied_weight,
        bias=None,
    )

    scale = torch.tanh(block.saliency_scale.float()).view(1, -1, 1, 1)
    if block.context_gate == 0.0:
        modulation = torch.zeros_like(saliency_aligned)
        headroom = torch.ones_like(saliency_aligned)
    else:
        centered = context_aligned - context_aligned.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True)
            + float(block.eps)
        )
        context_code = torch.tanh(centered * inverse_rms)
        modulation = 0.5 * (
            context_code
            - context_code.mean(dim=(-2, -1), keepdim=True)
        )
        magnitude = scale.abs()
        headroom = 1.0 + magnitude * (1.0 - magnitude) * modulation

    residual = (saliency_aligned * scale * headroom).to(keep.dtype)
    pre_activation = keep + residual
    if isinstance(block.activation, nn.Identity):
        output = pre_activation
    elif isinstance(block.activation, nn.ReLU):
        output = torch.relu(pre_activation)
    else:
        raise AssertionError(
            f"unsupported V8 activation {type(block.activation).__name__}"
        )
    return output, {
        "keep": keep,
        "context_aligned": context_aligned,
        "saliency_aligned": saliency_aligned,
        "modulation": modulation,
        "headroom": headroom,
    }


def _flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_tensors(item)


class _RankFiveOutputRecorder(TorchDispatchMode):
    """Record rank-five operation outputs during one ordinary forward."""

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        output = func(*args, **(kwargs or {}))
        for tensor in _flatten_tensors(output):
            if tensor.ndim == 5:
                self.shapes.append(tuple(tensor.shape))
        return output


def _assert_nested_exact(actual: Any, expected: Any, path: str = "root") -> None:
    if torch.is_tensor(actual):
        _assert_exact(actual, expected, message=path)
        return
    _require(
        type(actual) is type(expected),
        f"{path}: type mismatch {type(actual)} != {type(expected)}",
    )
    if isinstance(actual, dict):
        _require(
            tuple(actual) == tuple(expected),
            f"{path}: key mismatch {tuple(actual)} != {tuple(expected)}",
        )
        for key in actual:
            _assert_nested_exact(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, (tuple, list)):
        _require(
            len(actual) == len(expected),
            f"{path}: length mismatch {len(actual)} != {len(expected)}",
        )
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_nested_exact(left, right, f"{path}[{index}]")
        return
    _require(actual == expected, f"{path}: {actual!r} != {expected!r}")


@pytest.mark.parametrize("context_gate", [0.0, 1.0])
def test_state_layout_is_strictly_compatible_with_real_v7(
    context_gate: float,
) -> None:
    torch.manual_seed(11)
    legacy = TPDCleanV7DCHBlock(
        5,
        activate=True,
        context_gate=context_gate,
    )
    block = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=True,
        context_gate=context_gate,
    )

    expected_keys = (
        "saliency_scale",
        "phase_compress.weight",
        "phase_compress.bias",
    )
    _require(
        tuple(legacy.state_dict()) == expected_keys,
        f"unexpected real V7 state layout: {tuple(legacy.state_dict())}",
    )
    _require(
        tuple(block.state_dict()) == expected_keys,
        f"unexpected V8 state layout: {tuple(block.state_dict())}",
    )
    block.load_state_dict(legacy.state_dict(), strict=True)
    for key in expected_keys:
        _assert_exact(
            block.state_dict()[key],
            legacy.state_dict()[key],
            message=f"V7-to-V8 state {key}",
        )

    round_trip = TPDCleanV7DCHBlock(
        5,
        activate=True,
        context_gate=context_gate,
    )
    round_trip.load_state_dict(block.state_dict(), strict=True)
    _require(
        tuple(block.named_buffers()) == (),
        f"V8 unexpectedly registered buffers: {tuple(block.named_buffers())}",
    )
    _require(
        tuple(name for name, _ in block.named_children())
        == ("phase_compress", "activation"),
        f"unexpected V8 child layout: {tuple(block.named_children())}",
    )
    _require(
        tuple(name for name, _ in block.named_parameters()) == expected_keys,
        f"unexpected V8 parameter layout: {tuple(block.named_parameters())}",
    )


def test_tl_tr_bl_br_impulses_lock_pixel_unshuffle_phase_order() -> None:
    block = V8.TPDCleanV8MPRSDCHBlock(
        1,
        activate=False,
        context_gate=1.0,
    )
    offsets = ((0, 0), (0, 1), (1, 0), (1, 1))

    for expected_phase, (row, column) in enumerate(offsets):
        x = torch.zeros(1, 1, 4, 4)
        x[0, 0, row, column] = 1.0
        manual_phases = _manual_phases(x)
        rearranged, _, _, explicit_saliency = block.phase_sources(x)
        reshaped = rearranged.reshape(1, 1, 4, 2, 2)

        _assert_exact(
            reshaped,
            manual_phases,
            message=f"phase order for impulse {expected_phase}",
        )
        active = torch.nonzero(
            reshaped[0, 0, :, 0, 0],
            as_tuple=False,
        ).flatten()
        _require(
            active.tolist() == [expected_phase],
            (
                f"impulse {(row, column)} mapped to phases "
                f"{active.tolist()}, expected {[expected_phase]}"
            ),
        )
        _, _, manual_saliency = _manual_phase_saliency(x)
        _assert_close(
            explicit_saliency,
            manual_saliency,
            atol=1e-7,
            rtol=0.0,
            message=f"phase saliency for impulse {expected_phase}",
        )


@pytest.mark.parametrize("context_gate", [0.0, 1.0])
def test_joint_phase_and_weight_permutation_preserves_projection(
    context_gate: float,
) -> None:
    torch.manual_seed(23)
    original = V8.TPDCleanV8MPRSDCHBlock(
        4,
        activate=False,
        context_gate=context_gate,
    )
    permuted = V8.TPDCleanV8MPRSDCHBlock(
        4,
        activate=False,
        context_gate=context_gate,
    )
    with torch.no_grad():
        original.saliency_scale.uniform_(-0.5, 0.5)
    permuted.load_state_dict(original.state_dict(), strict=True)

    phase_permutation = [2, 0, 3, 1]
    with torch.no_grad():
        weight = original.phase_compress.weight.reshape(4, 4, 4, 1, 1)
        permuted.phase_compress.weight.copy_(
            weight[:, :, phase_permutation].reshape(4, 16, 1, 1)
        )

    x = torch.randn(2, 4, 16, 20)
    x_permuted = _image_from_manual_phases(
        _manual_phases(x)[:, :, phase_permutation]
    )
    _assert_close(
        permuted(x_permuted),
        original(x),
        atol=1e-5,
        rtol=1e-5,
        message=f"joint phase/weight permutation, gate={context_gate}",
    )


@pytest.mark.parametrize("context_gate", [0.0, 1.0])
def test_optimized_algebra_and_all_gradients_match_independent_reference(
    context_gate: float,
) -> None:
    torch.manual_seed(37 + int(context_gate))
    optimized = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=context_gate,
    )
    with torch.no_grad():
        optimized.saliency_scale.uniform_(-0.6, 0.6)
    reference = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=context_gate,
    )
    reference.load_state_dict(optimized.state_dict(), strict=True)

    optimized_x = torch.randn(2, 5, 16, 20, requires_grad=True)
    reference_x = optimized_x.detach().clone().requires_grad_(True)
    optimized_output = optimized(optimized_x)
    reference_output, reference_terms = _independent_forward(
        reference,
        reference_x,
    )
    _assert_close(
        optimized_output,
        reference_output,
        atol=5e-5,
        rtol=1e-5,
        message=f"optimized/direct output, gate={context_gate}",
    )
    _assert_close(
        optimized.aligned_mprs_terms(optimized_x)[-1],
        reference_terms["saliency_aligned"],
        atol=5e-5,
        rtol=1e-5,
        message=f"optimized/direct saliency, gate={context_gate}",
    )

    probe = torch.randn_like(optimized_output)
    optimized_gradients = torch.autograd.grad(
        (optimized_output * probe).mean(),
        (
            optimized_x,
            optimized.phase_compress.weight,
            optimized.phase_compress.bias,
            optimized.saliency_scale,
        ),
    )
    reference_gradients = torch.autograd.grad(
        (reference_output * probe).mean(),
        (
            reference_x,
            reference.phase_compress.weight,
            reference.phase_compress.bias,
            reference.saliency_scale,
        ),
    )
    gradient_names = ("input", "weight", "bias", "saliency_scale")
    for name, actual, expected in zip(
        gradient_names,
        optimized_gradients,
        reference_gradients,
    ):
        _assert_close(
            actual,
            expected,
            atol=2e-5,
            rtol=1e-4,
            message=f"optimized/direct {name} gradient, gate={context_gate}",
        )


def test_saliency_shortcut_cancels_keep_bias_in_value_and_gradient() -> None:
    torch.manual_seed(49)
    block = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=1.0,
    )
    x = torch.randn(2, 5, 16, 20)
    before = block.aligned_mprs_terms(x)[-1].detach()
    with torch.no_grad():
        block.phase_compress.bias.add_(
            torch.randn_like(block.phase_compress.bias)
        )
    after = block.aligned_mprs_terms(x)[-1]
    _assert_close(
        after,
        before,
        atol=5e-5,
        rtol=1e-5,
        message="MPRS saliency must not depend on Keep bias",
    )

    probe = torch.randn_like(after)
    bias_gradient = torch.autograd.grad(
        (after * probe).mean(),
        block.phase_compress.bias,
    )[0]
    _require(
        bias_gradient.abs().max().item() <= 1e-6,
        (
            "normalized MPRS-only bias gradient is too large: "
            f"{bias_gradient.abs().max().item():.9g}"
        ),
    )


@pytest.mark.parametrize("context_gate", [0.0, 1.0])
def test_standard_forward_has_three_expected_convolutions_and_no_5d_phase(
    context_gate: float,
) -> None:
    torch.manual_seed(61 + int(context_gate))
    block = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=context_gate,
    )
    x = torch.randn(2, 5, 16, 20)
    convolution_calls: list[
        tuple[tuple[int, ...], tuple[int, ...], bool]
    ] = []
    real_conv2d = V8.F.conv2d

    def traced_conv2d(input_tensor, weight, bias=None, *args, **kwargs):
        convolution_calls.append(
            (
                tuple(input_tensor.shape),
                tuple(weight.shape),
                bias is not None,
            )
        )
        return real_conv2d(input_tensor, weight, bias, *args, **kwargs)

    rank_five = _RankFiveOutputRecorder()
    with (
        mock.patch.object(
            block,
            "phase_sources",
            side_effect=AssertionError(
                "ordinary forward called the explicit phase-source path"
            ),
        ),
        mock.patch.object(V8.F, "conv2d", side_effect=traced_conv2d),
        rank_five,
    ):
        output = block(x)

    _require(
        tuple(output.shape) == (2, 5, 8, 10),
        f"unexpected block output shape: {tuple(output.shape)}",
    )
    expected_calls = [
        ((2, 20, 8, 10), (5, 20, 1, 1), True),
        ((2, 5, 8, 10), (5, 5, 1, 1), False),
        ((2, 5, 8, 10), (5, 5, 1, 1), False),
    ]
    _require(
        convolution_calls == expected_calls,
        (
            f"gate={context_gate} convolution contract mismatch: "
            f"{convolution_calls} != {expected_calls}"
        ),
    )

    expected_weight_view = (5, 5, 4, 1, 1)
    _require(
        rank_five.shapes,
        "rank-five observer did not see the expected phase-weight view",
    )
    _require(
        all(shape == expected_weight_view for shape in rank_five.shapes),
        (
            "ordinary forward materialized a rank-five tensor other than the "
            f"small phase-weight view: {rank_five.shapes}"
        ),
    )


@pytest.mark.parametrize("context_gate", [0.0, 1.0])
def test_diagnostics_are_single_pass_and_keep_three_conv_contract(
    context_gate: float,
) -> None:
    torch.manual_seed(73 + int(context_gate))
    block = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=context_gate,
    )
    x = torch.randn(2, 5, 16, 20)
    real_conv2d = V8.F.conv2d

    with (
        mock.patch.object(
            block,
            "aligned_mprs_terms",
            wraps=block.aligned_mprs_terms,
        ) as aligned_calls,
        mock.patch.object(
            block,
            "phase_sources",
            side_effect=AssertionError(
                "diagnostics called the explicit 5D phase path"
            ),
        ),
        mock.patch.object(
            V8.F,
            "conv2d",
            wraps=real_conv2d,
        ) as convolution_calls,
    ):
        diagnostic_output, diagnostics = (
            block.forward_with_mprs_diagnostics(x)
        )

    _require(
        aligned_calls.call_count == 1,
        f"diagnostics computed aligned terms {aligned_calls.call_count} times",
    )
    _require(
        convolution_calls.call_count == 3,
        f"diagnostics used {convolution_calls.call_count} convolutions",
    )
    _require(
        set(diagnostics)
        == {
            "context_aligned",
            "saliency_v7",
            "phase_correction",
            "saliency_v8",
            "scale",
            "modulation",
            "headroom",
        },
        f"unexpected diagnostic keys: {set(diagnostics)}",
    )
    _assert_close(
        diagnostic_output,
        block(x),
        atol=1e-6,
        rtol=1e-6,
        message=f"diagnostic/ordinary output, gate={context_gate}",
    )


@pytest.mark.parametrize("activate", [False, True])
@pytest.mark.parametrize("context_gate", [0.0, 1.0])
def test_zero_scale_matches_real_spd_output_and_shared_gradients(
    activate: bool,
    context_gate: float,
) -> None:
    torch.manual_seed(89 + 10 * int(activate) + int(context_gate))
    block = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=activate,
        context_gate=context_gate,
    )
    spd = PhaseKeepBlock(5, activate=activate)
    spd.phase_compress.load_state_dict(
        block.phase_compress.state_dict(),
        strict=True,
    )

    block_x = torch.randn(2, 5, 16, 20, requires_grad=True)
    spd_x = block_x.detach().clone().requires_grad_(True)
    block_output = block(block_x)
    spd_output = spd(spd_x)
    _assert_exact(
        block_output,
        spd_output,
        message=(
            f"zero-scale SPD output, activate={activate}, "
            f"gate={context_gate}"
        ),
    )

    probe = torch.randn_like(block_output)
    block_gradients = torch.autograd.grad(
        (block_output * probe).sum(),
        (
            block_x,
            block.phase_compress.weight,
            block.phase_compress.bias,
        ),
    )
    spd_gradients = torch.autograd.grad(
        (spd_output * probe).sum(),
        (
            spd_x,
            spd.phase_compress.weight,
            spd.phase_compress.bias,
        ),
    )
    for name, actual, expected in zip(
        ("input", "weight", "bias"),
        block_gradients,
        spd_gradients,
    ):
        _assert_exact(
            actual,
            expected,
            message=(
                f"zero-scale SPD {name} gradient, activate={activate}, "
                f"gate={context_gate}"
            ),
        )


@pytest.mark.parametrize("context_gate", [0.0, 1.0])
def test_zero_scale_multilevel_embedding_is_exactly_real_spd(
    context_gate: float,
) -> None:
    torch.manual_seed(101 + int(context_gate))
    embedding = V8.TPDCleanV8MPRSDCHPatchEmbedding(
        8,
        8,
        context_gate=context_gate,
    )
    spd = SPDPatchEmbedding(8, 8)
    _require(
        len(embedding.blocks) == len(spd.blocks) == 3,
        "stride-8 embeddings must contain three 2x blocks",
    )
    for block, spd_block in zip(embedding.blocks, spd.blocks):
        spd_block.phase_compress.load_state_dict(
            block.phase_compress.state_dict(),
            strict=True,
        )

    x = torch.randn(1, 8, 64, 64)
    _assert_exact(
        embedding(x),
        spd(x),
        message=f"zero-scale stride-8 embedding, gate={context_gate}",
    )

    emb1 = V8.build_clean_v8_mprs_dch_patch_embedding(
        "tpd_clean_v8_mprs_dch_full",
        32,
        16,
    )
    emb2 = V8.build_clean_v8_mprs_dch_patch_embedding(
        "tpd_clean_v8_mprs_dch_full",
        64,
        8,
    )
    shallow_parameters = V8.parameter_count(emb1) + V8.parameter_count(emb2)
    _require(
        shallow_parameters == 66_176,
        f"unexpected shallow parameter count: {shallow_parameters}",
    )


def test_zero_scale_full_capacity_gradients_and_first_adam_step_are_exact() -> None:
    torch.manual_seed(113)
    full = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=1.0,
    )
    capacity = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=0.0,
    )
    capacity.load_state_dict(full.state_dict(), strict=True)

    full_x = torch.randn(2, 5, 16, 20, requires_grad=True)
    capacity_x = full_x.detach().clone().requires_grad_(True)
    target = torch.randn(2, 5, 8, 10)
    full_optimizer = torch.optim.Adam(full.parameters(), lr=1e-3)
    capacity_optimizer = torch.optim.Adam(capacity.parameters(), lr=1e-3)

    ((full(full_x) - target) ** 2).mean().backward()
    ((capacity(capacity_x) - target) ** 2).mean().backward()
    _assert_exact(
        full_x.grad,
        capacity_x.grad,
        message="zero-scale Full/Capacity input gradient",
    )
    capacity_parameters = dict(capacity.named_parameters())
    for name, full_parameter in full.named_parameters():
        capacity_parameter = capacity_parameters[name]
        _require(
            full_parameter.grad is not None
            and capacity_parameter.grad is not None,
            f"missing zero-scale gradient for {name}",
        )
        _assert_exact(
            full_parameter.grad,
            capacity_parameter.grad,
            message=f"zero-scale Full/Capacity gradient {name}",
        )

    full_optimizer.step()
    capacity_optimizer.step()
    for name, full_parameter in full.named_parameters():
        _assert_exact(
            full_parameter,
            capacity_parameters[name],
            message=f"first Adam parameter {name}",
        )
    _assert_nested_exact(
        full_optimizer.state_dict(),
        capacity_optimizer.state_dict(),
        path="first_adam_state",
    )
