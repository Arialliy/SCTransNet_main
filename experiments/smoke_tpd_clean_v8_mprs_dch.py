#!/usr/bin/env python3
"""Two-step whole-model smoke for the V8-MPRS-DCH candidate pair.

The smoke exercises the real six-output SCTransNet objective on CPU or one
externally assigned physical GPU.  It verifies the dense-SPD zero-scale
anchor, paired Full/Capacity initialization and first Adam step, finite
gradients and updates, strict state/output reload, and the optimized MPRS
diagnostic/three-convolution contract.

CUDA reports are accepted only when exactly physical GPU 2 or 3 is exposed as
logical ``cuda:0`` and its registered UUID is verified.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence
from unittest import mock

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import smoke_tpd_clean_v3 as v3_smoke  # noqa: E402
from experiments import smoke_tpd_clean_v6 as v6_smoke  # noqa: E402
from experiments import smoke_tpd_clean_v7_dch as v7_smoke  # noqa: E402
from experiments.train_tpd_clean_v8_mprs_dch import (  # noqa: E402
    CONTEXT_CODE_FORMULA,
    FULL_HEADROOM_FORMULA,
    FUSION_FORMULA,
    MPRS_MASS_FORMULA,
    MPRS_REUSE_FORMULA,
    MPRS_SOURCE_FORMULA,
    PHASE_TIED_PROJECTION_FORMULA,
    build_clean_v8_mprs_dch_model,
)
from model import tpd_clean_v8_mprs_dch as v8_model  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    CONTEXT_HEADROOM_CEILING,
    CONTEXT_HEADROOM_FLOOR,
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
    TPDCleanV8MPRSDCHBlock,
)


SCHEMA = "sctransnet_tpd_clean_v8_mprs_dch_smoke_v1"
UINT32_MAX = 4_294_967_295
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FORMAL_EPS = v6_smoke.FORMAL_EPS
FORMAL_DTYPE = v6_smoke.FORMAL_DTYPE
EXPECTED_DEVICE_NAME = v6_smoke.EXPECTED_DEVICE_NAME
PHYSICAL_GPU_UUIDS = v6_smoke.PHYSICAL_GPU_UUIDS
EXPECTED_SCALE_PARAMETER_NAMES = v6_smoke.EXPECTED_SCALE_PARAMETER_NAMES
EXPECTED_KEEP_PARAMETER_NAMES = v6_smoke.EXPECTED_KEEP_PARAMETER_NAMES
EXPECTED_BLOCK_EPS_NAMES = v6_smoke.EXPECTED_BLOCK_EPS_NAMES
EXPECTED_MPRS_DIAGNOSTIC_KEYS = frozenset(
    {
        "context_aligned",
        "saliency_v7",
        "phase_correction",
        "saliency_v8",
        "scale",
        "modulation",
        "headroom",
    }
)
SOURCE_RELATIVES = (
    "experiments/smoke_tpd_clean_v8_mprs_dch.py",
    "experiments/train_tpd_clean_v8_mprs_dch.py",
    "model/tpd_clean_v8_mprs_dch.py",
    "model/SCTransNet.py",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
)

# The V7 implementation already owns the canonical nested optimizer-state
# serializer.  Reusing it keeps first-step evidence comparable across versions.
_state_sha256 = v7_smoke._state_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V8-MPRS-DCH exact two-step CPU/CUDA smoke"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "atomically write the completed report as a pure JSON document; "
            "when omitted, JSON is written to stdout"
        ),
    )
    parser.add_argument("--expected-device-name", default=None)
    parser.add_argument(
        "--expected-cuda-visible-devices",
        default=None,
        help=(
            "required for CUDA; must equal the single external physical "
            "GPU index 2 or 3"
        ),
    )
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2")
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error("--patch-size must be >= 32 and divisible by 32")
    if args.steps != 2:
        parser.error("--steps must equal 2")
    if not 0 <= args.seed <= UINT32_MAX:
        parser.error("--seed must lie in [0, 4294967295]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    return args


@contextlib.contextmanager
def _deterministic_smoke_execution(
    device_text: str,
) -> Iterator[Dict[str, Any]]:
    """Make byte-exact paired CUDA replay an explicit smoke condition.

    Full and Capacity have the same zero-scale first-order computation, so
    their first Adam states must match exactly.  CUDA convolution backward can
    otherwise select execution paths whose reduction order is not byte-stable
    across the two sequential candidate runs.  This scoped contract fixes
    those execution conditions and restores every process-wide setting on
    exit.
    """

    is_cuda = torch.device(device_text).type == "cuda"
    previous_deterministic = (
        torch.are_deterministic_algorithms_enabled()
    )
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cuda_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cuda_initialized_before_contract = (
        torch.cuda.is_initialized() if is_cuda else False
    )

    if is_cuda:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    evidence = {
        "applicable": is_cuda,
        "enabled": is_cuda,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if is_cuda
            else None
        ),
        "cuda_initialized_before_contract": (
            cuda_initialized_before_contract if is_cuda else None
        ),
        "purpose": "byte_exact_sequential_first_adam_pair",
    }
    try:
        yield evidence
    finally:
        if is_cuda:
            torch.use_deterministic_algorithms(
                previous_deterministic,
                warn_only=previous_warn_only,
            )
            torch.backends.cudnn.deterministic = (
                previous_cudnn_deterministic
            )
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                previous_cuda_matmul_tf32
            )
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
            if previous_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


def _validate_mprs_block(
    block: nn.Module,
    block_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Validate one optimized block without consuming the training RNG."""

    if not isinstance(block, TPDCleanV8MPRSDCHBlock):
        raise TypeError(f"{block_name} is not a V8-MPRS-DCH block")
    value_count = block.channels * 8 * 8
    diagnostic_input = torch.linspace(
        -1.0,
        1.0,
        steps=value_count,
        dtype=FORMAL_DTYPE,
        device=device,
    ).reshape(1, block.channels, 8, 8)

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
        mock.patch.object(
            block,
            "phase_sources",
            side_effect=AssertionError(
                f"{block_name} ordinary forward used explicit phase sources"
            ),
        ),
        mock.patch.object(
            v8_model.F,
            "conv2d",
            wraps=v8_model.F.conv2d,
        ) as standard_calls,
    ):
        standard_output = block(diagnostic_input)
    if standard_calls.call_count != 3:
        raise RuntimeError(
            f"{block_name} ordinary forward used "
            f"{standard_calls.call_count} convolutions instead of 3"
        )

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
        mock.patch.object(
            block,
            "phase_sources",
            side_effect=AssertionError(
                f"{block_name} diagnostics used explicit phase sources"
            ),
        ),
        mock.patch.object(
            v8_model.F,
            "conv2d",
            wraps=v8_model.F.conv2d,
        ) as diagnostic_calls,
    ):
        diagnostic_output, diagnostics = (
            block.forward_with_mprs_diagnostics(diagnostic_input)
        )
    if diagnostic_calls.call_count != 3:
        raise RuntimeError(
            f"{block_name} diagnostic forward used "
            f"{diagnostic_calls.call_count} convolutions instead of 3"
        )
    if not torch.equal(standard_output, diagnostic_output):
        difference = float(
            (standard_output - diagnostic_output).abs().max().item()
        )
        raise RuntimeError(
            f"{block_name} diagnostic output differs from ordinary forward: "
            f"max_abs_difference={difference}"
        )
    if set(diagnostics) != set(EXPECTED_MPRS_DIAGNOSTIC_KEYS):
        raise RuntimeError(
            f"{block_name} MPRS diagnostic keys differ from the contract"
        )
    for name, tensor in diagnostics.items():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(
            tensor
        ).all():
            raise FloatingPointError(
                f"{block_name}.{name} is missing or non-finite"
            )

    reconstructed = (
        diagnostics["saliency_v7"] + diagnostics["phase_correction"]
    )
    identity_max_abs = float(
        (diagnostics["saliency_v8"] - reconstructed).abs().max().item()
    )
    if identity_max_abs != 0.0:
        raise RuntimeError(
            f"{block_name} violates Sa8=Sa7+phase_correction: "
            f"max_abs_difference={identity_max_abs}"
        )
    return {
        "status": "complete",
        "channels": block.channels,
        "context_gate": block.context_gate,
        "ordinary_forward_conv2d_calls": standard_calls.call_count,
        "diagnostic_forward_conv2d_calls": diagnostic_calls.call_count,
        "explicit_phase_sources_in_production": False,
        "diagnostic_keys": sorted(diagnostics),
        "diagnostic_output_exact": True,
        "reuse_identity_max_abs_difference": identity_max_abs,
    }


def _validate_mprs_model(
    model: nn.Module,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Validate all seven formal MPRS blocks."""

    evidence = {
        name: _validate_mprs_block(block, name, device)
        for name, block in v6_smoke._v6_blocks(model).items()
    }
    if len(evidence) != 7:
        raise RuntimeError(f"expected 7 MPRS block reports, got {len(evidence)}")
    return evidence


def _validate_finite_gradients(model: nn.Module) -> int:
    """Require every observed whole-model gradient tensor to be finite."""

    observed = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        observed += 1
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"{name} gradient is not finite")
    if observed == 0:
        raise RuntimeError("whole-model backward produced no gradients")
    return observed


def _validate_finite_parameters(model: nn.Module) -> int:
    """Require every updated model parameter tensor to remain finite."""

    observed = 0
    for name, parameter in model.named_parameters():
        observed += 1
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"{name} parameter is not finite")
    if observed == 0:
        raise RuntimeError("whole model has no parameters")
    return observed


def _run_variant(
    variant: str,
    *,
    inputs_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    spd_outputs_cpu: tuple[torch.Tensor, ...],
    device: torch.device,
    steps: int,
    seed: int,
    learning_rate: float,
) -> Dict[str, Any]:
    """Exercise one real V8 candidate through two Adam steps and reload."""

    with contextlib.redirect_stdout(sys.stderr):
        model, metadata = build_clean_v8_mprs_dch_model(variant, seed)
    model.to(device)
    formal_contract = v6_smoke._formal_model_contract(model)
    inputs = inputs_cpu.to(device, dtype=FORMAL_DTYPE)
    targets = targets_cpu.to(device, dtype=FORMAL_DTYPE)

    model.eval()
    mprs_blocks = _validate_mprs_model(model, device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        initial_outputs = v6_smoke._validate_v6_outputs(
            model(inputs),
            batch_size=inputs.shape[0],
            patch_size=inputs.shape[-1],
        )
        zero_scale_max_abs = max(
            float(
                (
                    output.detach().cpu() - reference
                ).abs().max().item()
            )
            for output, reference in zip(initial_outputs, spd_outputs_cpu)
        )
    if zero_scale_max_abs != 0.0:
        raise RuntimeError(
            f"{variant} is not exact dense-SPD at zero scale: "
            f"max_abs_difference={zero_scale_max_abs}"
        )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="mean")
    scales = v6_smoke._v6_scale_parameters(model)
    keep_parameters = v6_smoke._v6_keep_parameters(model)
    scale_before = v3_smoke._snapshot(scales)
    keep_before = v3_smoke._snapshot(keep_parameters)
    losses: list[float] = []
    per_head_losses: list[list[float]] = []
    scale_gradient_l1: Dict[str, float] = {}
    keep_gradient_l1: Dict[str, float] = {}
    first_step_model_checksum: str | None = None
    first_step_optimizer_checksum: str | None = None
    first_step_optimizer_state: Dict[str, Any] | None = None
    finite_gradient_tensor_count: int | None = None
    finite_parameter_tensor_count: int | None = None

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=False):
            outputs = v6_smoke._validate_v6_outputs(
                model(inputs),
                batch_size=inputs.shape[0],
                patch_size=inputs.shape[-1],
            )
            loss, head_losses = v6_smoke.six_output_bce_loss(
                outputs,
                targets,
                criterion,
            )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(
                f"{variant} step {step_index + 1} produced invalid loss"
            )
        loss.backward()
        observed_gradients = _validate_finite_gradients(model)
        losses.append(float(loss.detach().item()))
        per_head_losses.append(
            [float(value.detach().item()) for value in head_losses]
        )
        if step_index == 0:
            finite_gradient_tensor_count = observed_gradients
            scale_gradient_l1 = v3_smoke._gradient_l1(scales, "scale")
            keep_gradient_l1 = v3_smoke._gradient_l1(
                keep_parameters,
                "keep",
            )
        optimizer.step()
        observed_parameters = _validate_finite_parameters(model)
        if step_index == 0:
            finite_parameter_tensor_count = observed_parameters
            first_step_model_checksum = v3_smoke.model_checksum(model)
            optimizer_state_dict = optimizer.state_dict()
            first_step_optimizer_checksum = _state_sha256(
                optimizer_state_dict
            )
            state = optimizer_state_dict["state"]
            first_step_optimizer_state = {
                "parameter_state_count": len(state),
                "step_values": sorted(
                    {
                        float(item["step"].detach().cpu().item())
                        for item in state.values()
                        if "step" in item
                    }
                ),
                "exp_avg_l1": float(
                    sum(
                        item["exp_avg"].detach().abs().sum().cpu().item()
                        for item in state.values()
                        if "exp_avg" in item
                    )
                ),
                "exp_avg_sq_l1": float(
                    sum(
                        item["exp_avg_sq"].detach().abs().sum().cpu().item()
                        for item in state.values()
                        if "exp_avg_sq" in item
                    )
                ),
            }

    scale_update_l1 = v3_smoke._update_l1(
        scale_before,
        scales,
        "scale",
    )
    keep_update_l1 = v3_smoke._update_l1(
        keep_before,
        keep_parameters,
        "keep",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    trained_checksum = v3_smoke.model_checksum(model)
    model.eval()
    reload_inputs = inputs[: min(2, inputs.shape[0])]
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        source_outputs = tuple(
            output.detach().clone()
            for output in v6_smoke._validate_v6_outputs(
                model(reload_inputs),
                batch_size=reload_inputs.shape[0],
                patch_size=reload_inputs.shape[-1],
            )
        )

    with contextlib.redirect_stdout(sys.stderr):
        rebuilt, _ = build_clean_v8_mprs_dch_model(
            variant,
            v3_smoke._next_seed(seed),
        )
    incompatible = rebuilt.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{variant} strict state load is incompatible")
    if v3_smoke.model_checksum(rebuilt) != trained_checksum:
        raise RuntimeError(f"{variant} checksum changed after strict reload")
    rebuilt.to(device)
    v6_smoke._formal_model_contract(rebuilt)
    rebuilt.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        rebuilt_outputs = v6_smoke._validate_v6_outputs(
            rebuilt(reload_inputs),
            batch_size=reload_inputs.shape[0],
            patch_size=reload_inputs.shape[-1],
        )
        reload_max_abs = max(
            float((left - right).abs().max().item())
            for left, right in zip(source_outputs, rebuilt_outputs)
        )
    if reload_max_abs != 0.0:
        raise RuntimeError(
            f"{variant} strict reload changed outputs: "
            f"max_abs_difference={reload_max_abs}"
        )
    if (
        first_step_model_checksum is None
        or first_step_optimizer_checksum is None
        or first_step_optimizer_state is None
        or finite_gradient_tensor_count is None
        or finite_parameter_tensor_count is None
    ):
        raise RuntimeError(
            "first Adam-step or finite-state evidence was not recorded"
        )

    report = {
        "variant": variant,
        "status": "complete",
        "output_count": 6,
        "loss_head_count": 6,
        "loss_definition": "sum_of_six_mean_bce_outputs",
        "loss_sum_verified": True,
        "losses": losses,
        "per_head_losses": per_head_losses,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "optimizer_steps_completed": steps,
        "step_zero_exact_spd": True,
        "step_zero_max_abs_difference": zero_scale_max_abs,
        "first_step_model_checksum": first_step_model_checksum,
        "first_step_optimizer_checksum": first_step_optimizer_checksum,
        "first_step_optimizer_state": first_step_optimizer_state,
        "all_observed_gradients_finite": True,
        "finite_gradient_tensor_count": finite_gradient_tensor_count,
        "all_updated_parameters_finite": True,
        "finite_parameter_tensor_count": finite_parameter_tensor_count,
        "scale_parameter_names": sorted(scales),
        "keep_parameter_names": sorted(keep_parameters),
        "scale_gradient_l1": scale_gradient_l1,
        "phase_gradient_l1": keep_gradient_l1,
        "scale_update_l1": scale_update_l1,
        "phase_update_l1": keep_update_l1,
        "strict_rebuild_load": True,
        "strict_reload_max_abs_difference": reload_max_abs,
        "trained_model_checksum": trained_checksum,
        "initial_model_checksum": metadata["full_initialization_sha256"],
        "total_parameters": metadata["total_parameters"],
        "shallow_embedding_parameters": metadata[
            "shallow_embedding_parameters"
        ],
        "context_gate": metadata["context_gate"],
        "context_reference": metadata["context_reference"],
        "context_headroom": metadata["context_headroom"],
        "phase_tied_projection": metadata["phase_tied_projection"],
        "derived_projection_parameters": metadata[
            "derived_projection_parameters"
        ],
        "saliency_representation": metadata["saliency_representation"],
        "saliency_formula": metadata["saliency_formula"],
        "saliency_mass_invariant": metadata["saliency_mass_invariant"],
        "saliency_nonnegative": metadata["saliency_nonnegative"],
        "saliency_forward_implementation": metadata[
            "saliency_forward_implementation"
        ],
        "state_compatible_with": metadata["state_compatible_with"],
        "cross_version_exact_resume_supported": metadata[
            "cross_version_exact_resume_supported"
        ],
        "fusion_formula": metadata["fusion_formula"],
        "standard_forward_conv2d_calls_per_block": 3,
        "mprs_diagnostics_verified": True,
        "mprs_block_count": len(mprs_blocks),
        "mprs_blocks": mprs_blocks,
        "formal_eps": formal_contract["formal_eps"],
        "block_eps": formal_contract["block_eps"],
        "amp_enabled": False,
        "autocast_forced_disabled": True,
        "input_dtype": str(inputs.dtype),
        "target_dtype": str(targets.dtype),
        "output_dtypes": sorted({str(item.dtype) for item in outputs}),
        "model_parameter_dtypes": formal_contract[
            "model_parameter_dtypes"
        ],
        "model_floating_buffer_dtypes": formal_contract[
            "model_floating_buffer_dtypes"
        ],
        "projection_precision": metadata["projection_precision"],
        "context_precision": metadata["context_precision"],
        "coefficient_precision": metadata["coefficient_precision"],
        "residual_output_dtype": metadata["residual_output_dtype"],
    }
    del rebuilt_outputs
    del rebuilt
    del source_outputs
    del state_dict
    del outputs
    del initial_outputs
    del optimizer
    del targets
    del inputs
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _pair_evidence(
    reports: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
) -> tuple[bool | None, str, str | None, bool | None]:
    if tuple(variants) != tuple(SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS):
        return None, "not_checked_single_variant", None, None
    initial = {str(item["initial_model_checksum"]) for item in reports}
    if len(initial) != 1:
        raise RuntimeError(
            "V8-MPRS-DCH candidates do not share one initial state"
        )
    first_step_models = {
        str(item["first_step_model_checksum"]) for item in reports
    }
    first_step_optimizers = {
        str(item["first_step_optimizer_checksum"]) for item in reports
    }
    first_step_equal = (
        len(first_step_models) == 1 and len(first_step_optimizers) == 1
    )
    if not first_step_equal:
        raise RuntimeError(
            "V8-MPRS-DCH Full/Capacity differ at their zero-scale "
            "first Adam step"
        )
    return True, "verified", next(iter(initial)), True


def _run_smoke_impl(
    *,
    variant: str,
    device_text: str,
    batch_size: int,
    patch_size: int,
    steps: int,
    seed: int,
    learning_rate: float = 1e-3,
    expected_device_name: str | None = None,
    expected_cuda_visible_devices: str | None = None,
) -> Dict[str, Any]:
    """Run the V8 pair on CPU or one UUID-verified physical GPU 2/3."""

    if variant not in ("all",) + SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")
    if patch_size < 32 or patch_size % 32:
        raise ValueError("patch_size must be >= 32 and divisible by 32")
    if steps != 2:
        raise ValueError("steps must equal 2")
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError("seed must lie in [0, 4294967295]")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    environment_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    device, device_name, cuda_contract = v6_smoke._resolve_device_contract(
        device_text,
        expected_device_name,
        expected_cuda_visible_devices,
    )
    variants = (
        SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS
        if variant == "all"
        else (variant,)
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    inputs_cpu, targets_cpu = v3_smoke._paired_inputs(
        batch_size,
        patch_size,
        seed,
    )
    spd_outputs_cpu = v6_smoke._spd_reference_v6(
        inputs_cpu,
        device,
        seed,
    )
    reports = [
        _run_variant(
            current,
            inputs_cpu=inputs_cpu,
            targets_cpu=targets_cpu,
            spd_outputs_cpu=spd_outputs_cpu,
            device=device,
            steps=steps,
            seed=seed,
            learning_rate=learning_rate,
        )
        for current in variants
    ]
    paired, pair_status, pair_checksum, first_step_equal = _pair_evidence(
        reports,
        variants,
    )

    cuda_memory: Dict[str, float] | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_memory = {
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
            ),
        }
    physical_gpu_index = (
        cuda_contract["declared_physical_index"]
        if cuda_contract["validated"]
        else None
    )
    physical_gpu_uuid = (
        cuda_contract["device_uuid"]
        if cuda_contract["validated"]
        else None
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "source_sha256": {
            relative: file_sha256(REPO_ROOT / relative)
            for relative in SOURCE_RELATIVES
        },
        "variants": reports,
        "paired_initialization": paired,
        "paired_initialization_status": pair_status,
        "paired_initialization_sha256": pair_checksum,
        "paired_first_adam_step_exact": first_step_equal,
        "device": str(device),
        "device_name": device_name,
        "environment_cuda_visible_devices": environment_mask,
        "cuda_visible_devices": physical_gpu_index,
        "physical_gpu_index": physical_gpu_index,
        "physical_gpu_uuid": physical_gpu_uuid,
        "cuda_device_contract": cuda_contract,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "steps": steps,
        "seed": seed,
        "cuda_memory": cuda_memory,
        "formal_eps": FORMAL_EPS,
        "formal_amp_enabled": False,
        "autocast_forced_disabled": True,
        "input_dtype": str(inputs_cpu.dtype),
        "target_dtype": str(targets_cpu.dtype),
        "loss_head_count": 6,
        "loss_definition": "sum_of_six_mean_bce_outputs",
        "scale_parameter_names": sorted(EXPECTED_SCALE_PARAMETER_NAMES),
        "keep_parameter_names": sorted(EXPECTED_KEEP_PARAMETER_NAMES),
        "phase_tied_projection_formula": PHASE_TIED_PROJECTION_FORMULA,
        "mprs_source_formula": MPRS_SOURCE_FORMULA,
        "mprs_mass_formula": MPRS_MASS_FORMULA,
        "mprs_reuse_formula": MPRS_REUSE_FORMULA,
        "context_code_formula": CONTEXT_CODE_FORMULA,
        "deferred_context_headroom_formula": FULL_HEADROOM_FORMULA,
        "fusion_equation": FUSION_FORMULA,
        "learned_scales_per_block": 1,
        "derived_projection_parameters": 0,
        "phase_contrast_parameters": 0,
        "standard_forward_conv2d_calls_per_block": 3,
        "mprs_diagnostics_required": True,
        "headroom_bound": [
            CONTEXT_HEADROOM_FLOOR,
            CONTEXT_HEADROOM_CEILING,
        ],
        "coefficient_bound": "abs(a*H)<=1",
    }


def run_smoke(
    *,
    variant: str,
    device_text: str,
    batch_size: int,
    patch_size: int,
    steps: int,
    seed: int,
    learning_rate: float = 1e-3,
    expected_device_name: str | None = None,
    expected_cuda_visible_devices: str | None = None,
) -> Dict[str, Any]:
    """Run one smoke inside the byte-exact deterministic replay contract."""

    with _deterministic_smoke_execution(device_text) as execution:
        report = _run_smoke_impl(
            variant=variant,
            device_text=device_text,
            batch_size=batch_size,
            patch_size=patch_size,
            steps=steps,
            seed=seed,
            learning_rate=learning_rate,
            expected_device_name=expected_device_name,
            expected_cuda_visible_devices=expected_cuda_visible_devices,
        )
        report["deterministic_execution"] = dict(execution)
    return report


def _json_payload(report: Mapping[str, Any]) -> str:
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json_report(path: Path, payload: str) -> None:
    """Atomically publish exactly one JSON document."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(
        f".{resolved.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_smoke(
        variant=args.variant,
        device_text=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        expected_device_name=args.expected_device_name,
        expected_cuda_visible_devices=args.expected_cuda_visible_devices,
    )
    payload = _json_payload(report)
    if args.output is None:
        print(payload, flush=True)
    else:
        _write_json_report(args.output, payload)
        print(
            f"pure JSON report written: {args.output.expanduser().resolve()}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
