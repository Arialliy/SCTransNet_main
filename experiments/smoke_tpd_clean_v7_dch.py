#!/usr/bin/env python3
"""Two-step whole-model smoke test for the isolated V7-DCH pair.

The check is deliberately small, but it exercises the real six-output
SCTransNet objective on CPU or one externally selected physical GPU.  It
verifies the dense-SPD zero-scale anchor, the paired Full/Capacity
initialization and first Adam step, finite gradients/updates, and exact
state/output reload.  CUDA reports are accepted only for physical GPU 2 or 3.
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
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import smoke_tpd_clean_v6 as v6_smoke  # noqa: E402
from experiments import smoke_tpd_clean_v3 as v3_smoke  # noqa: E402
from experiments.train_tpd_clean_v7_dch import (  # noqa: E402
    CONTEXT_CODE_FORMULA,
    FULL_HEADROOM_FORMULA,
    FUSION_FORMULA,
    PHASE_TIED_PROJECTION_FORMULA,
    build_clean_v7_dch_model,
)
from model.tpd_clean_v7_dch import (  # noqa: E402
    CONTEXT_HEADROOM_CEILING,
    CONTEXT_HEADROOM_FLOOR,
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
)


SCHEMA = "sctransnet_tpd_clean_v7_dch_smoke_v1"
UINT32_MAX = 4_294_967_295
FORMAL_EPS = v6_smoke.FORMAL_EPS
FORMAL_DTYPE = v6_smoke.FORMAL_DTYPE
EXPECTED_DEVICE_NAME = v6_smoke.EXPECTED_DEVICE_NAME
PHYSICAL_GPU_UUIDS = v6_smoke.PHYSICAL_GPU_UUIDS
EXPECTED_SCALE_PARAMETER_NAMES = v6_smoke.EXPECTED_SCALE_PARAMETER_NAMES
EXPECTED_KEEP_PARAMETER_NAMES = v6_smoke.EXPECTED_KEEP_PARAMETER_NAMES
EXPECTED_BLOCK_EPS_NAMES = v6_smoke.EXPECTED_BLOCK_EPS_NAMES


def _state_sha256(value: Any) -> str:
    """Hash a nested optimizer state with explicit type/shape boundaries."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda current: repr(current)):
                update(key)
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(
                b"list\0" if isinstance(item, list) else b"tuple\0"
            )
            for child in item:
                update(child)
            return
        digest.update(type(item).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(item).encode("utf-8"))
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V7-DCH exact two-step CPU/CUDA smoke test"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_CLEAN_V7_DCH_VARIANTS,
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expected-device-name", default=None)
    parser.add_argument("--expected-cuda-visible-devices", default=None)
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
    """Exercise one real DCH candidate through two Adam steps and reload."""

    with contextlib.redirect_stdout(sys.stderr):
        model, metadata = build_clean_v7_dch_model(variant, seed)
    model.to(device)
    formal_contract = v6_smoke._formal_model_contract(model)
    inputs = inputs_cpu.to(device, dtype=FORMAL_DTYPE)
    targets = targets_cpu.to(device, dtype=FORMAL_DTYPE)

    model.eval()
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
        if step_index == 0:
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
        rebuilt, _ = build_clean_v7_dch_model(
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
    ):
        raise RuntimeError("first Adam-step evidence was not recorded")

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


def _pair_evidence(
    reports: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
) -> tuple[bool | None, str, str | None, bool | None]:
    if tuple(variants) != tuple(SUPPORTED_CLEAN_V7_DCH_VARIANTS):
        return None, "not_checked_single_variant", None, None
    initial = {str(item["initial_model_checksum"]) for item in reports}
    if len(initial) != 1:
        raise RuntimeError("DCH candidates do not share one initial state")
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
            "DCH Full/Capacity differ at their zero-scale first Adam step"
        )
    return True, "verified", next(iter(initial)), True


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
    """Run the isolated DCH pair on CPU or physical GPU 2/3."""

    if variant not in ("all",) + SUPPORTED_CLEAN_V7_DCH_VARIANTS:
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
        SUPPORTED_CLEAN_V7_DCH_VARIANTS
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
    return {
        "schema": SCHEMA,
        "status": "complete",
        "variants": reports,
        "paired_initialization": paired,
        "paired_initialization_status": pair_status,
        "paired_initialization_sha256": pair_checksum,
        "paired_first_adam_step_exact": first_step_equal,
        "device": str(device),
        "device_name": device_name,
        "environment_cuda_visible_devices": environment_mask,
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
        "deferred_context_headroom_formula": FULL_HEADROOM_FORMULA,
        "fusion_equation": FUSION_FORMULA,
        "learned_scales_per_block": 1,
        "derived_projection_parameters": 0,
        "headroom_bound": [
            CONTEXT_HEADROOM_FLOOR,
            CONTEXT_HEADROOM_CEILING,
        ],
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
