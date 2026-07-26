#!/usr/bin/env python3
"""Forward/backward/reload preflight for TPD-Clean-v5 candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import smoke_tpd_clean_v3 as v3_smoke  # noqa: E402
from experiments.train_tpd_clean_v5 import build_clean_v5_model  # noqa: E402
from model.tpd_clean_v5 import SUPPORTED_CLEAN_V5_VARIANTS  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v5_smoke_v1"
UINT32_MAX = 4_294_967_295


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TPD-Clean-v5 forward/backward/reload validation"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_CLEAN_V5_VARIANTS,
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expected-device-name", default=None)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2")
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error("--patch-size must be >= 32 and divisible by 32")
    if args.steps < 2:
        parser.error("--steps must be >= 2")
    if not 0 <= args.seed <= UINT32_MAX:
        parser.error("--seed must lie in [0, 4294967295]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    return args


def _v5_scale_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    parameters: Dict[str, nn.Parameter] = {}
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        blocks: Iterable[nn.Module] = embedding.blocks
        for index, block in enumerate(blocks):
            scale = getattr(block, "saliency_scale", None)
            if not isinstance(scale, nn.Parameter):
                raise TypeError(
                    f"{embedding_name}.blocks.{index}.saliency_scale "
                    "is missing"
                )
            if hasattr(block, "context_scale"):
                raise RuntimeError(
                    f"{embedding_name}.blocks.{index} unexpectedly has "
                    "a second learned Context scale"
                )
            parameters[
                f"{embedding_name}.blocks.{index}.saliency_scale"
            ] = scale
    if len(parameters) != 7:
        raise RuntimeError(f"expected 7 v5 KCS scales, got {len(parameters)}")
    return parameters


def _spd_reference_v5(
    inputs_cpu: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Build the dense-SPD reference without process-wide stream rebinding."""

    model, _ = v3_smoke.build_model("spd", seed)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        outputs = v3_smoke._validate_outputs(
            model(inputs_cpu.to(device)),
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


def _run_v5_variant(
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
    """Run one v5 candidate without rebinding any v3 module globals."""

    model, metadata = build_clean_v5_model(variant, seed)
    model.to(device)
    model.eval()
    inputs = inputs_cpu.to(device)
    targets = targets_cpu.to(device)

    with torch.inference_mode():
        initial_outputs = v3_smoke._validate_outputs(
            model(inputs),
            batch_size=inputs.shape[0],
            patch_size=inputs.shape[-1],
        )
        for index, (output, reference) in enumerate(
            zip(initial_outputs, spd_outputs_cpu)
        ):
            if not torch.equal(output.detach().cpu(), reference):
                difference = float(
                    (
                        output.detach().cpu() - reference
                    ).abs().max().item()
                )
                raise RuntimeError(
                    f"{variant} output {index} is not exact SPD at step zero; "
                    f"max_abs_difference={difference}"
                )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="mean")
    scales = _v5_scale_parameters(model)
    phase = v3_smoke._phase_parameters(model)
    scale_before = v3_smoke._snapshot(scales)
    phase_before = v3_smoke._snapshot(phase)
    losses: list[float] = []
    scale_gradient_l1: Dict[str, float] = {}
    phase_gradient_l1: Dict[str, float] = {}

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = v3_smoke._validate_outputs(
            model(inputs),
            batch_size=inputs.shape[0],
            patch_size=inputs.shape[-1],
        )
        loss = v3_smoke.deep_supervision_loss(outputs, targets, criterion)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(
                f"{variant} step {step_index + 1} produced invalid loss"
            )
        loss.backward()
        losses.append(float(loss.detach().item()))
        if step_index == 0:
            scale_gradient_l1 = v3_smoke._gradient_l1(scales, "scale")
            phase_gradient_l1 = v3_smoke._gradient_l1(phase, "phase")
        optimizer.step()

    scale_update_l1 = v3_smoke._update_l1(
        scale_before,
        scales,
        "scale",
    )
    phase_update_l1 = v3_smoke._update_l1(
        phase_before,
        phase,
        "phase",
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
    with torch.inference_mode():
        source_outputs = tuple(
            output.detach().clone()
            for output in v3_smoke._validate_outputs(
                model(reload_inputs),
                batch_size=reload_inputs.shape[0],
                patch_size=reload_inputs.shape[-1],
            )
        )

    rebuilt, _ = build_clean_v5_model(
        variant,
        v3_smoke._next_seed(seed),
    )
    incompatible = rebuilt.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{variant} strict state load reported incompatibility")
    if v3_smoke.model_checksum(rebuilt) != trained_checksum:
        raise RuntimeError(f"{variant} checksum changed after strict reload")
    rebuilt.to(device)
    rebuilt.eval()
    with torch.inference_mode():
        rebuilt_outputs = v3_smoke._validate_outputs(
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
        "losses": losses,
        "step_zero_exact_spd": True,
        "scale_gradient_l1": scale_gradient_l1,
        "phase_gradient_l1": phase_gradient_l1,
        "scale_update_l1": scale_update_l1,
        "phase_update_l1": phase_update_l1,
        "strict_rebuild_load": True,
        "strict_reload_max_abs_difference": reload_max_abs,
        "trained_model_checksum": trained_checksum,
        "initial_model_checksum": metadata["full_initialization_sha256"],
        "total_parameters": metadata["total_parameters"],
        "shallow_embedding_parameters": metadata[
            "shallow_embedding_parameters"
        ],
        "context_code": metadata["context_code"],
    }
    del rebuilt_outputs
    del rebuilt
    del source_outputs
    del state_dict
    del outputs
    del initial_outputs
    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


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
) -> Dict[str, Any]:
    """Run the established checks through a v5-parameterized harness."""

    if variant not in ("all",) + SUPPORTED_CLEAN_V5_VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")
    if patch_size < 32 or patch_size % 32:
        raise ValueError("patch_size must be >= 32 and divisible by 32")
    if steps < 2:
        raise ValueError("steps must be >= 2")
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError("seed must lie in [0, 4294967295]")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    device, device_name = v3_smoke._resolve_device(
        device_text,
        expected_device_name,
    )
    variants = (
        SUPPORTED_CLEAN_V5_VARIANTS if variant == "all" else (variant,)
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    inputs_cpu, targets_cpu = v3_smoke._paired_inputs(
        batch_size,
        patch_size,
        seed,
    )
    spd_outputs_cpu = _spd_reference_v5(inputs_cpu, device, seed)
    reports = [
        _run_v5_variant(
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
    initial_checksums = {
        report["initial_model_checksum"] for report in reports
    }
    if len(initial_checksums) != 1:
        raise RuntimeError("candidate initial states are not exactly paired")
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
        "paired_initialization": True,
        "device": str(device),
        "device_name": device_name,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "steps": steps,
        "seed": seed,
        "cuda_memory": cuda_memory,
        "fusion_formula": (
            "K+S*tanh(saliency_scale*(1+0.5*context_code))"
        ),
        "context_selector_range": [0.5, 1.5],
        "learned_scales_per_block": 1,
        "residual_bound": (
            "absolute_residual_at_most_absolute_saliency"
        ),
    }


def main() -> None:
    args = parse_args()
    report = run_smoke(
        variant=args.variant,
        device_text=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        expected_device_name=args.expected_device_name,
    )
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
