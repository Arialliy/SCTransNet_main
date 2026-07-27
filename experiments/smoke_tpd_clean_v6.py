#!/usr/bin/env python3
"""Source-neutral two-step preflight for isolated TPD-Clean-v6 candidates.

CUDA selection is process-external.  A CUDA report is protocol-valid only
when exactly one of the preregistered physical GPUs is exposed as ``cuda:0``::

    CUDA_VISIBLE_DEVICES=2 python experiments/smoke_tpd_clean_v6.py \
        --device cuda:0 --expected-cuda-visible-devices 2

The smoke checks the six-output objective, two optimizer steps, all seven
Saliency scales, all seven dense-Keep projections, step-zero dense-SPD
equivalence, paired initialization, and strict state/output reload.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import smoke_tpd_clean_v3 as v3_smoke  # noqa: E402
from experiments.train_tpd_clean_v6 import (  # noqa: E402
    CONTEXT_CODE_FORMULA,
    FUSION_FORMULA,
    PHASE_TIED_PROJECTION_FORMULA,
    build_clean_v6_model,
)
from model.tpd_clean_v6 import SUPPORTED_CLEAN_V6_VARIANTS  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v6_smoke_v1"
UINT32_MAX = 4_294_967_295
FORMAL_EPS = 1e-6
FORMAL_DTYPE = torch.float32
EXPECTED_DEVICE_NAME = "NVIDIA GeForce RTX 5090"

# These UUIDs are already frozen in the repository's physical-GPU 2/3
# completion validators and exact-run launch contracts.
PHYSICAL_GPU_UUIDS: Mapping[str, str] = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}

EXPECTED_SCALE_PARAMETER_NAMES = frozenset(
    {
        *{
            f"embeddings_1.blocks.{index}.saliency_scale"
            for index in range(4)
        },
        *{
            f"embeddings_2.blocks.{index}.saliency_scale"
            for index in range(3)
        },
    }
)
EXPECTED_KEEP_PARAMETER_NAMES = frozenset(
    {
        *{
            f"embeddings_1.blocks.{index}.phase_compress.{parameter}"
            for index in range(4)
            for parameter in ("weight", "bias")
        },
        *{
            f"embeddings_2.blocks.{index}.phase_compress.{parameter}"
            for index in range(3)
            for parameter in ("weight", "bias")
        },
    }
)
EXPECTED_BLOCK_EPS_NAMES = frozenset(
    {
        f"{name.rsplit('.', 1)[0]}.eps"
        for name in EXPECTED_SCALE_PARAMETER_NAMES
    }
)


def normalized_gpu_uuid(value: Any) -> str:
    """Normalize PyTorch/NVML UUID text to the ``GPU-...`` form."""

    text = str(value).strip()
    if not text:
        raise ValueError("GPU UUID is empty")
    return text if text.startswith("GPU-") else f"GPU-{text}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TPD-Clean-v6 exact two-step CPU/CUDA validation"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_CLEAN_V6_VARIANTS,
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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


def _external_cuda_mapping(
    device_text: str,
    expected_cuda_visible_devices: str | None,
) -> str | None:
    """Validate the external ordinal mask without touching CUDA runtime state."""

    requested = torch.device(device_text)
    if requested.type == "cpu":
        if str(requested) != "cpu":
            raise ValueError("CPU smoke must use device=cpu")
        if expected_cuda_visible_devices is not None:
            raise ValueError(
                "--expected-cuda-visible-devices is valid only for CUDA"
            )
        return None
    if requested.type != "cuda" or str(requested) != "cuda:0":
        raise ValueError("CUDA smoke must use the single logical device cuda:0")
    if expected_cuda_visible_devices is None:
        raise ValueError(
            "--expected-cuda-visible-devices is required for CUDA"
        )
    if expected_cuda_visible_devices not in PHYSICAL_GPU_UUIDS:
        raise ValueError(
            "--expected-cuda-visible-devices must be exactly 2 or 3"
        )
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual != expected_cuda_visible_devices:
        raise RuntimeError(
            "unexpected CUDA_VISIBLE_DEVICES mapping: "
            f"expected={expected_cuda_visible_devices!r}, actual={actual!r}"
        )
    return actual


def _resolve_device_contract(
    device_text: str,
    expected_device_name: str | None,
    expected_cuda_visible_devices: str | None,
) -> tuple[torch.device, str, Dict[str, Any]]:
    """Resolve CPU or one UUID-verified preregistered RTX 5090."""

    declared = _external_cuda_mapping(
        device_text,
        expected_cuda_visible_devices,
    )
    requested = torch.device(device_text)
    if requested.type == "cpu":
        if expected_device_name is not None:
            raise ValueError("--expected-device-name is valid only for CUDA")
        return (
            torch.device("cpu"),
            "cpu",
            {
                "applicable": False,
                "validated": False,
                "declared_physical_index": None,
                "expected_physical_index": None,
                "logical_device": None,
                "visible_device_count": None,
                "device_name": None,
                "expected_device_name": None,
                "device_uuid": None,
                "expected_device_uuid": None,
            },
        )

    if (
        expected_device_name is not None
        and expected_device_name != EXPECTED_DEVICE_NAME
    ):
        raise ValueError(
            f"--expected-device-name must equal {EXPECTED_DEVICE_NAME!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    visible_count = torch.cuda.device_count()
    if visible_count != 1:
        raise RuntimeError(
            "expected exactly one visible CUDA device, "
            f"found {visible_count}"
        )
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if device_name != EXPECTED_DEVICE_NAME:
        raise RuntimeError(
            f"unexpected device: expected={EXPECTED_DEVICE_NAME!r}, "
            f"actual={device_name!r}"
        )
    properties = torch.cuda.get_device_properties(device)
    device_uuid_value = getattr(properties, "uuid", None)
    if device_uuid_value is None:
        raise RuntimeError("torch CUDA device properties do not expose a UUID")
    device_uuid = normalized_gpu_uuid(device_uuid_value)
    expected_uuid = PHYSICAL_GPU_UUIDS[str(declared)]
    if device_uuid != expected_uuid:
        raise RuntimeError(
            "CUDA physical-device UUID differs from the preregistered "
            f"GPU {declared}: expected={expected_uuid!r}, "
            f"actual={device_uuid!r}"
        )
    return (
        device,
        device_name,
        {
            "applicable": True,
            "validated": True,
            "declared_physical_index": declared,
            "expected_physical_index": expected_cuda_visible_devices,
            "logical_device": "cuda:0",
            "visible_device_count": visible_count,
            "device_name": device_name,
            "expected_device_name": EXPECTED_DEVICE_NAME,
            "device_uuid": device_uuid,
            "expected_device_uuid": expected_uuid,
        },
    )


def _v6_blocks(model: nn.Module) -> Dict[str, nn.Module]:
    blocks_by_name: Dict[str, nn.Module] = {}
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        blocks: Iterable[nn.Module] = embedding.blocks
        for index, block in enumerate(blocks):
            blocks_by_name[f"{embedding_name}.blocks.{index}"] = block
    if len(blocks_by_name) != 7:
        raise RuntimeError(f"expected 7 v6 KCS blocks, got {len(blocks_by_name)}")
    return blocks_by_name


def _v6_scale_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    parameters: Dict[str, nn.Parameter] = {}
    for block_name, block in _v6_blocks(model).items():
        scale = getattr(block, "saliency_scale", None)
        if not isinstance(scale, nn.Parameter):
            raise TypeError(f"{block_name}.saliency_scale is missing")
        if hasattr(block, "context_scale"):
            raise RuntimeError(
                f"{block_name} unexpectedly has a second learned Context scale"
            )
        parameters[f"{block_name}.saliency_scale"] = scale
    if set(parameters) != set(EXPECTED_SCALE_PARAMETER_NAMES):
        raise RuntimeError(
            "v6 scale parameter names differ from the frozen seven-name set"
        )
    return parameters


def _v6_keep_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    parameters = v3_smoke._phase_parameters(model)
    if set(parameters) != set(EXPECTED_KEEP_PARAMETER_NAMES):
        raise RuntimeError(
            "v6 Keep parameter names differ from the frozen fourteen-name set"
        )
    return parameters


def _formal_model_contract(model: nn.Module) -> Dict[str, Any]:
    block_eps = {
        f"{name}.eps": float(getattr(block, "eps"))
        for name, block in _v6_blocks(model).items()
    }
    if (
        set(block_eps) != set(EXPECTED_BLOCK_EPS_NAMES)
        or any(value != FORMAL_EPS for value in block_eps.values())
    ):
        raise RuntimeError("all seven V6 blocks must use eps=1e-6")
    parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters()}
    )
    floating_buffer_dtypes = sorted(
        {
            str(buffer.dtype)
            for buffer in model.buffers()
            if buffer.is_floating_point()
        }
    )
    if parameter_dtypes != [str(FORMAL_DTYPE)]:
        raise RuntimeError(
            f"model parameter dtypes are not formal FP32: {parameter_dtypes}"
        )
    if floating_buffer_dtypes not in ([], [str(FORMAL_DTYPE)]):
        raise RuntimeError(
            "model floating-buffer dtypes are not formal FP32: "
            f"{floating_buffer_dtypes}"
        )
    return {
        "block_eps": block_eps,
        "formal_eps": FORMAL_EPS,
        "model_parameter_dtypes": parameter_dtypes,
        "model_floating_buffer_dtypes": floating_buffer_dtypes,
    }


def _validate_v6_outputs(
    outputs: Any,
    *,
    batch_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, ...]:
    normalized = v3_smoke._validate_outputs(
        outputs,
        batch_size=batch_size,
        patch_size=patch_size,
    )
    for index, output in enumerate(normalized):
        if output.dtype != FORMAL_DTYPE:
            raise RuntimeError(
                f"output {index} dtype={output.dtype}, "
                f"expected={FORMAL_DTYPE}"
            )
    return normalized


def six_output_bce_loss(
    outputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return the explicit six-head BCE sum and verify baseline equivalence."""

    normalized = tuple(outputs)
    if len(normalized) != 6:
        raise RuntimeError(
            f"the formal objective requires exactly six outputs, "
            f"got {len(normalized)}"
        )
    head_losses = tuple(criterion(output, target) for output in normalized)
    total = sum(head_losses)
    with torch.no_grad():
        baseline_total = v3_smoke.deep_supervision_loss(
            normalized,
            target,
            criterion,
        )
    if not torch.equal(total.detach(), baseline_total.detach()):
        difference = float(
            (total.detach() - baseline_total.detach()).abs().item()
        )
        raise RuntimeError(
            "explicit six-head BCE differs from the baseline objective: "
            f"absolute_difference={difference}"
        )
    return total, head_losses


def _spd_reference_v6(
    inputs_cpu: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Build the dense-SPD reference without rebinding shared harness state."""

    with contextlib.redirect_stdout(sys.stderr):
        model, _ = v3_smoke.build_model("spd", seed)
    model.to(device)
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        outputs = _validate_v6_outputs(
            model(inputs_cpu.to(device, dtype=FORMAL_DTYPE)),
            batch_size=inputs_cpu.shape[0],
            patch_size=inputs_cpu.shape[-1],
        )
        reference = tuple(
            output.detach().cpu().clone() for output in outputs
        )
    del outputs
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reference


def _run_v6_variant(
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
    """Run one V6 candidate through forward/backward/update/reload checks."""

    with contextlib.redirect_stdout(sys.stderr):
        model, metadata = build_clean_v6_model(variant, seed)
    model.to(device)
    formal_contract = _formal_model_contract(model)
    model.eval()
    inputs = inputs_cpu.to(device, dtype=FORMAL_DTYPE)
    targets = targets_cpu.to(device, dtype=FORMAL_DTYPE)
    if inputs.dtype != FORMAL_DTYPE or targets.dtype != FORMAL_DTYPE:
        raise RuntimeError("formal smoke inputs and targets must be FP32")

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        initial_outputs = _validate_v6_outputs(
            model(inputs),
            batch_size=inputs.shape[0],
            patch_size=inputs.shape[-1],
        )
        for index, (output, reference) in enumerate(
            zip(initial_outputs, spd_outputs_cpu)
        ):
            output_cpu = output.detach().cpu()
            if not torch.equal(output_cpu, reference):
                difference = float((output_cpu - reference).abs().max().item())
                raise RuntimeError(
                    f"{variant} output {index} is not exact SPD at step zero; "
                    f"max_abs_difference={difference}"
                )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="mean")
    scales = _v6_scale_parameters(model)
    keep_parameters = _v6_keep_parameters(model)
    scale_before = v3_smoke._snapshot(scales)
    keep_before = v3_smoke._snapshot(keep_parameters)
    losses: list[float] = []
    per_head_losses: list[list[float]] = []
    scale_gradient_l1: Dict[str, float] = {}
    keep_gradient_l1: Dict[str, float] = {}

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=False):
            outputs = _validate_v6_outputs(
                model(inputs),
                batch_size=inputs.shape[0],
                patch_size=inputs.shape[-1],
            )
            loss, head_losses = six_output_bce_loss(
                outputs,
                targets,
                criterion,
            )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(
                f"{variant} step {step_index + 1} produced invalid loss"
            )
        loss.backward()
        losses.append(float(loss.detach().item()))
        per_head_losses.append(
            [float(value.detach().item()) for value in head_losses]
        )
        if step_index == 0:
            scale_gradient_l1 = v3_smoke._gradient_l1(scales, "scale")
            keep_gradient_l1 = v3_smoke._gradient_l1(
                keep_parameters,
                "keep",
            )
        optimizer.step()

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
            for output in _validate_v6_outputs(
                model(reload_inputs),
                batch_size=reload_inputs.shape[0],
                patch_size=reload_inputs.shape[-1],
            )
        )

    with contextlib.redirect_stdout(sys.stderr):
        rebuilt, _ = build_clean_v6_model(
            variant,
            v3_smoke._next_seed(seed),
        )
    incompatible = rebuilt.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{variant} strict state load reported incompatibility")
    if v3_smoke.model_checksum(rebuilt) != trained_checksum:
        raise RuntimeError(f"{variant} checksum changed after strict reload")
    rebuilt.to(device)
    _formal_model_contract(rebuilt)
    rebuilt.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        rebuilt_outputs = _validate_v6_outputs(
            rebuilt(reload_inputs),
            batch_size=reload_inputs.shape[0],
            patch_size=reload_inputs.shape[-1],
        )
        reload_max_abs = max(
            float((source - rebuilt_output).abs().max().item())
            for source, rebuilt_output in zip(source_outputs, rebuilt_outputs)
        )
    if reload_max_abs != 0.0:
        raise RuntimeError(
            f"{variant} strict reload changed outputs: "
            f"max_abs_difference={reload_max_abs}"
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
        "context_code": metadata["context_code"],
        "context_modulation": metadata["context_modulation"],
        "context_headroom": metadata["context_headroom"],
        "phase_tied_projection": metadata["phase_tied_projection"],
        "derived_projection_parameters": metadata[
            "derived_projection_parameters"
        ],
        "fusion_formula": metadata["fusion_formula"],
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


def _paired_initialization_fields(
    reports: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
) -> tuple[bool | None, str, str | None]:
    if tuple(variants) != tuple(SUPPORTED_CLEAN_V6_VARIANTS):
        return None, "not_checked_single_variant", None
    initial_checksums = {
        str(report["initial_model_checksum"]) for report in reports
    }
    if len(initial_checksums) != 1:
        raise RuntimeError("candidate initial states are not exactly paired")
    checksum = next(iter(initial_checksums))
    return True, "verified", checksum


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
    """Run the isolated V6 exact two-step smoke on CPU or physical GPU 2/3."""

    if variant not in ("all",) + SUPPORTED_CLEAN_V6_VARIANTS:
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

    environment_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    device, device_name, cuda_contract = _resolve_device_contract(
        device_text,
        expected_device_name,
        expected_cuda_visible_devices,
    )
    variants = (
        SUPPORTED_CLEAN_V6_VARIANTS if variant == "all" else (variant,)
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    inputs_cpu, targets_cpu = v3_smoke._paired_inputs(
        batch_size,
        patch_size,
        seed,
    )
    if inputs_cpu.dtype != FORMAL_DTYPE or targets_cpu.dtype != FORMAL_DTYPE:
        raise RuntimeError("paired smoke inputs and targets must be FP32")
    spd_outputs_cpu = _spd_reference_v6(inputs_cpu, device, seed)
    reports = [
        _run_v6_variant(
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
    paired, paired_status, paired_checksum = _paired_initialization_fields(
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
    return {
        "schema": SCHEMA,
        "status": "complete",
        "variants": reports,
        "paired_initialization": paired,
        "paired_initialization_status": paired_status,
        "paired_initialization_sha256": paired_checksum,
        "device": str(device),
        "device_name": device_name,
        "environment_cuda_visible_devices": environment_cuda_visible_devices,
        "cuda_visible_devices": (
            cuda_contract["declared_physical_index"]
            if cuda_contract["validated"]
            else None
        ),
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
        "context_code_formula": CONTEXT_CODE_FORMULA,
        "fusion_equation": FUSION_FORMULA,
        "learned_scales_per_block": 1,
        "derived_projection_parameters": 0,
        "headroom_bound": [0.5, 1.5],
        "coefficient_bound": "abs(a*H)<=1",
        "residual_bound": "abs(R)<=abs(Sa)",
    }


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
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
