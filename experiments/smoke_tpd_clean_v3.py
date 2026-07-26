#!/usr/bin/env python3
"""Forward/backward/reload preflight for TPD-Clean-v3 candidates."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_tpd_clean_v3 import build_clean_v3_model  # noqa: E402
from experiments.train_tpd_pilot import (  # noqa: E402
    build_model,
    deep_supervision_loss,
    model_checksum,
)
from model.tpd_clean_v3 import SUPPORTED_CLEAN_V3_VARIANTS  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v3_smoke_v1"
UINT32_MAX = 4_294_967_295


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TPD-Clean-v3 forward/backward/reload validation"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_CLEAN_V3_VARIANTS,
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


def _resolve_device(
    device_text: str, expected_device_name: str | None
) -> tuple[torch.device, str]:
    device = torch.device(device_text)
    if device.type != "cuda":
        if expected_device_name is not None:
            raise ValueError("--expected-device-name is valid only for CUDA")
        return device, "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expected exactly one visible CUDA device, "
            f"found {torch.cuda.device_count()}"
        )
    if device.index not in (None, 0):
        raise ValueError("with one visible CUDA device, use cuda or cuda:0")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if expected_device_name is not None and device_name != expected_device_name:
        raise RuntimeError(
            f"unexpected device: expected={expected_device_name!r}, "
            f"actual={device_name!r}"
        )
    return device, device_name


def _validate_outputs(
    outputs: Any, *, batch_size: int, patch_size: int
) -> tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)):
        raise TypeError("expected six deep-supervision outputs")
    normalized = tuple(outputs)
    if len(normalized) != 6:
        raise RuntimeError(f"expected six outputs, got {len(normalized)}")
    expected_shape = (batch_size, 1, patch_size, patch_size)
    for index, output in enumerate(normalized):
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"output {index} shape={tuple(output.shape)}, "
                f"expected={expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise FloatingPointError(f"output {index} is not finite")
    return normalized


def _kcs_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    parameters: Dict[str, nn.Parameter] = {}
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        blocks: Iterable[nn.Module] = embedding.blocks
        for index, block in enumerate(blocks):
            for parameter_name in (
                "context_scale",
                "saliency_scale",
            ):
                parameter = getattr(block, parameter_name, None)
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(
                        f"{embedding_name}.blocks.{index}.{parameter_name} "
                        "is missing"
                    )
                parameters[
                    f"{embedding_name}.blocks.{index}.{parameter_name}"
                ] = parameter
    if len(parameters) != 14:
        raise RuntimeError(f"expected 14 KCS scales, got {len(parameters)}")
    return parameters


def _phase_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    parameters: Dict[str, nn.Parameter] = {}
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        for index, block in enumerate(embedding.blocks):
            for parameter_name, parameter in block.phase_compress.named_parameters():
                parameters[
                    f"{embedding_name}.blocks.{index}."
                    f"phase_compress.{parameter_name}"
                ] = parameter
    if len(parameters) != 14:
        raise RuntimeError(f"expected 14 dense-Keep tensors, got {len(parameters)}")
    return parameters


def _snapshot(
    parameters: Mapping[str, nn.Parameter],
) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in parameters.items()
    }


def _gradient_l1(
    parameters: Mapping[str, nn.Parameter], label: str
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"{label}.{name} has no gradient")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"{label}.{name} gradient is not finite")
        value = float(gradient.detach().abs().sum().item())
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{label}.{name} gradient is zero")
        values[name] = value
    return values


def _update_l1(
    before: Mapping[str, torch.Tensor],
    parameters: Mapping[str, nn.Parameter],
    label: str,
) -> Dict[str, float]:
    if set(before) != set(parameters):
        raise RuntimeError(f"{label} parameter names changed")
    values = {
        name: float(
            (parameters[name].detach() - before[name]).abs().sum().item()
        )
        for name in before
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{label}.{name} did not update")
    return values


def _next_seed(seed: int) -> int:
    return (seed + 1) & UINT32_MAX


def _paired_inputs(
    batch_size: int, patch_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10_001)
    inputs = torch.randn(
        batch_size,
        1,
        patch_size,
        patch_size,
        generator=generator,
    )
    targets = (
        torch.rand(
            batch_size,
            1,
            patch_size,
            patch_size,
            generator=generator,
        )
        < 0.01
    ).to(torch.float32)
    return inputs, targets


def _spd_reference(
    inputs_cpu: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    with contextlib.redirect_stdout(sys.stderr):
        model, _ = build_model("spd", seed)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        outputs = _validate_outputs(
            model(inputs_cpu.to(device)),
            batch_size=inputs_cpu.shape[0],
            patch_size=inputs_cpu.shape[-1],
        )
        reference = tuple(output.detach().cpu().clone() for output in outputs)
    del outputs
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reference


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
    with contextlib.redirect_stdout(sys.stderr):
        model, metadata = build_clean_v3_model(variant, seed)
    model.to(device)
    model.eval()
    inputs = inputs_cpu.to(device)
    targets = targets_cpu.to(device)

    with torch.inference_mode():
        initial_outputs = _validate_outputs(
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
    scales = _kcs_parameters(model)
    phase = _phase_parameters(model)
    scale_before = _snapshot(scales)
    phase_before = _snapshot(phase)
    losses: list[float] = []
    scale_gradient_l1: Dict[str, float] = {}
    phase_gradient_l1: Dict[str, float] = {}

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = _validate_outputs(
            model(inputs),
            batch_size=inputs.shape[0],
            patch_size=inputs.shape[-1],
        )
        loss = deep_supervision_loss(outputs, targets, criterion)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(
                f"{variant} step {step_index + 1} produced invalid loss"
            )
        loss.backward()
        losses.append(float(loss.detach().item()))
        if step_index == 0:
            scale_gradient_l1 = _gradient_l1(scales, "scale")
            phase_gradient_l1 = _gradient_l1(phase, "phase")
        optimizer.step()

    scale_update_l1 = _update_l1(scale_before, scales, "scale")
    phase_update_l1 = _update_l1(phase_before, phase, "phase")
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    trained_checksum = model_checksum(model)
    model.eval()
    reload_inputs = inputs[: min(2, inputs.shape[0])]
    with torch.inference_mode():
        source_outputs = tuple(
            output.detach().clone()
            for output in _validate_outputs(
                model(reload_inputs),
                batch_size=reload_inputs.shape[0],
                patch_size=reload_inputs.shape[-1],
            )
        )

    with contextlib.redirect_stdout(sys.stderr):
        rebuilt, _ = build_clean_v3_model(variant, _next_seed(seed))
    incompatible = rebuilt.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{variant} strict state load reported incompatibility")
    if model_checksum(rebuilt) != trained_checksum:
        raise RuntimeError(f"{variant} checksum changed after strict reload")
    rebuilt.to(device)
    rebuilt.eval()
    with torch.inference_mode():
        rebuilt_outputs = _validate_outputs(
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
    if variant not in ("all",) + SUPPORTED_CLEAN_V3_VARIANTS:
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
    device, device_name = _resolve_device(device_text, expected_device_name)
    variants = (
        SUPPORTED_CLEAN_V3_VARIANTS if variant == "all" else (variant,)
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    inputs_cpu, targets_cpu = _paired_inputs(batch_size, patch_size, seed)
    spd_outputs_cpu = _spd_reference(inputs_cpu, device, seed)
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
    print(json.dumps(report, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
