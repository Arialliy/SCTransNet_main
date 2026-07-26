#!/usr/bin/env python3
"""Two-step trainability preflight for the isolated TPD-NER v1 model."""

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

from experiments.train_tpd_ner_v1 import build_tpd_ner_model  # noqa: E402
from experiments.train_tpd_pilot import (  # noqa: E402
    deep_supervision_loss,
    model_checksum,
)


SCHEMA = "sctransnet_tpd_ner_v1_two_step_smoke_v1"
UINT32_MAX = 4_294_967_295


def _next_rebuild_seed(seed: int) -> int:
    """Choose a distinct NumPy-compatible seed without leaving uint32."""

    if not 0 <= seed <= UINT32_MAX:
        raise ValueError(f"seed must lie in [0, {UINT32_MAX}]")
    return (seed + 1) & UINT32_MAX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-step TPD-NER trainability and memory preflight"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expected-device-name", default=None)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2")
    if args.patch_size < 32 or args.patch_size % 16:
        parser.error("--patch-size must be >= 32 and divisible by 16")
    if args.steps < 2:
        parser.error("--steps must be >= 2")
    if not 0 <= args.seed <= UINT32_MAX:
        parser.error("--seed must lie in [0, 4294967295]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    return args


def _snapshot(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _update_l1(
    before: Mapping[str, torch.Tensor],
    module: nn.Module,
) -> float:
    current = dict(module.named_parameters())
    if set(before) != set(current):
        raise RuntimeError("parameter names changed during the smoke run")
    return sum(
        float((current[name].detach() - value).abs().sum().item())
        for name, value in before.items()
    )


def _gradient_l1(module: nn.Module, label: str) -> float:
    total = 0.0
    parameter_count = 0
    for name, parameter in module.named_parameters():
        parameter_count += 1
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"{label}.{name} has no gradient")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"{label}.{name} has a non-finite gradient")
        total += float(gradient.detach().abs().sum().item())
    if parameter_count == 0:
        raise RuntimeError(f"{label} has no trainable parameters")
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"{label} has no non-zero finite gradient")
    return total


def _tpd_scales(model: nn.Module) -> Dict[str, nn.Parameter]:
    scales: Dict[str, nn.Parameter] = {}
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        blocks: Iterable[nn.Module] = getattr(embedding, "blocks")
        for index, block in enumerate(blocks):
            for scale_name in ("context_scale", "saliency_scale"):
                scale = getattr(block, scale_name, None)
                if not isinstance(scale, nn.Parameter):
                    raise TypeError(
                        f"{embedding_name}.blocks[{index}].{scale_name} is missing"
                    )
                scales[f"{embedding_name}.blocks.{index}.{scale_name}"] = scale
    if len(scales) != 14:
        raise RuntimeError(f"expected 14 K-C-S residual scales, got {len(scales)}")
    return scales


def _validate_outputs(
    outputs: Any,
    *,
    batch_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)):
        raise TypeError("TPD-NER smoke requires six deep-supervision outputs")
    normalized = tuple(outputs)
    if len(normalized) != 6:
        raise RuntimeError(f"expected six model outputs, got {len(normalized)}")
    expected_shape = (batch_size, 1, patch_size, patch_size)
    for index, output in enumerate(normalized):
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"output {index} has shape={tuple(output.shape)}, "
                f"expected={expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise FloatingPointError(f"output {index} contains non-finite values")
    return normalized


def _resolve_device(
    device_text: str,
    expected_device_name: str | None,
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
            "TPD-NER requires exactly one visible CUDA device; "
            f"found {torch.cuda.device_count()}"
        )
    if device.index not in (None, 0):
        raise ValueError("with one visible CUDA device, --device must be cuda or cuda:0")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if expected_device_name is not None and device_name != expected_device_name:
        raise RuntimeError(
            f"unexpected CUDA device name: expected={expected_device_name!r}, "
            f"actual={device_name!r}"
        )
    return device, device_name


def run_smoke(
    *,
    device_text: str,
    batch_size: int,
    patch_size: int,
    steps: int,
    seed: int,
    learning_rate: float = 1e-3,
    expected_device_name: str | None = None,
) -> Dict[str, Any]:
    """Run real forward/backward/update steps and return a JSON-ready report."""

    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")
    if patch_size < 32 or patch_size % 16:
        raise ValueError("patch_size must be >= 32 and divisible by 16")
    if steps < 2:
        raise ValueError("steps must be >= 2")
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError("seed must lie in [0, 4294967295]")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    device, device_name = _resolve_device(device_text, expected_device_name)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # SCTransNet reports its deep-supervision mode during construction. Keep
    # stdout machine-readable by redirecting that informational line to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        model, metadata = build_tpd_ner_model("tpd_clean_full_ner", seed)
    model.to(device)
    model.train()
    relay = model.tpd_ner
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="mean")

    input_generator = torch.Generator(device="cpu")
    input_generator.manual_seed(seed + 10_001)
    inputs = torch.randn(
        batch_size,
        1,
        patch_size,
        patch_size,
        generator=input_generator,
    ).to(device)
    targets = (
        torch.rand(
            batch_size,
            1,
            patch_size,
            patch_size,
            generator=input_generator,
        )
        < 0.01
    ).to(dtype=torch.float32, device=device)

    scales = _tpd_scales(model)
    scale_before = {
        name: parameter.detach().clone()
        for name, parameter in scales.items()
    }
    gate_before = {
        stage: _snapshot(gate)
        for stage, gate in relay.gates.items()
    }
    losses: list[float] = []
    gate_gradient_l1: Dict[str, float] = {}
    gate_update_l1: Dict[str, float] = {}
    fusion_gradient_l1: Dict[str, float] = {}
    fusion_update_l1: Dict[str, float] = {}

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        fusion_before_step = (
            {
                stage: _snapshot(fusion)
                for stage, fusion in relay.fusions.items()
            }
            if step_index == 1
            else {}
        )
        outputs = _validate_outputs(
            model(inputs),
            batch_size=batch_size,
            patch_size=patch_size,
        )
        loss = deep_supervision_loss(outputs, targets, criterion)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(f"step {step_index + 1} produced invalid loss")
        loss.backward()
        losses.append(float(loss.detach().item()))

        if step_index == 0:
            gate_gradient_l1 = {
                stage: _gradient_l1(gate, f"gate[{stage}]")
                for stage, gate in relay.gates.items()
            }
            scale_gradient_l1 = sum(
                float(parameter.grad.detach().abs().sum().item())
                for parameter in scales.values()
                if parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
            )
            if not math.isfinite(scale_gradient_l1) or scale_gradient_l1 <= 0.0:
                raise RuntimeError("K-C-S residual scales have no finite gradient")
        elif step_index == 1:
            fusion_gradient_l1 = {
                stage: _gradient_l1(fusion, f"fusion[{stage}]")
                for stage, fusion in relay.fusions.items()
            }

        optimizer.step()

        if step_index == 0:
            gate_update_l1 = {
                stage: _update_l1(gate_before[stage], gate)
                for stage, gate in relay.gates.items()
            }
            if any(value <= 0.0 for value in gate_update_l1.values()):
                raise RuntimeError("at least one relay gate did not update on step 1")
            scale_update_l1 = sum(
                float((scales[name].detach() - value).abs().sum().item())
                for name, value in scale_before.items()
            )
            if not math.isfinite(scale_update_l1) or scale_update_l1 <= 0.0:
                raise RuntimeError("no K-C-S residual scale updated on step 1")
        elif step_index == 1:
            fusion_update_l1 = {
                stage: _update_l1(fusion_before_step[stage], fusion)
                for stage, fusion in relay.fusions.items()
            }
            if any(value <= 0.0 for value in fusion_update_l1.values()):
                raise RuntimeError("at least one relay fusion did not update on step 2")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    source_checksum = model_checksum(model)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    with contextlib.redirect_stdout(sys.stderr):
        rebuilt, _ = build_tpd_ner_model(
            "tpd_clean_full_ner",
            _next_rebuild_seed(seed),
        )
    rebuilt.load_state_dict(state_dict, strict=True)
    rebuilt_checksum = model_checksum(rebuilt)
    if rebuilt_checksum != source_checksum:
        raise RuntimeError(
            "strictly rebuilt model checksum differs after state_dict loading"
        )

    cuda_memory: Dict[str, float] | None = None
    if device.type == "cuda":
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
        "variant": "tpd_clean_full_ner",
        "device": str(device),
        "device_name": device_name,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "steps": steps,
        "output_count": 6,
        "losses": losses,
        "gate_gradient_l1": gate_gradient_l1,
        "gate_update_l1": gate_update_l1,
        "fusion_gradient_l1": fusion_gradient_l1,
        "fusion_update_l1": fusion_update_l1,
        "tpd_scale_update_l1": scale_update_l1,
        "strict_rebuild_load": True,
        "model_checksum": source_checksum,
        "relay_parameters": metadata["relay_parameters"],
        "total_parameters": metadata["total_parameters"],
        "cuda_memory": cuda_memory,
    }


def main() -> None:
    args = parse_args()
    report = run_smoke(
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
