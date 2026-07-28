#!/usr/bin/env python3
"""Two-step smoke test for the V8-MPRS-DCH five-node NER extension."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_tpd_clean_v8_mprs_dch import (  # noqa: E402
    build_clean_v8_mprs_dch_model,
)
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
)
from model.tpd_ner_v8_mprs_dch import (  # noqa: E402
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_RELAY_ON_PARAMETERS,
    PRODUCTION_RELAY_PARAMETERS,
    adapt_v8_mprs_dch_parent,
    relay_parameter_count,
)


SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_two_step_smoke_v1"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
RELAY_WIDTH = 8
INPUT_GENERATOR_OFFSET = 10_001


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_state_sha256(model: nn.Module) -> str:
    return _state_sha256(model.state_dict())


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _resolve_device(
    device_text: str,
    expected_device_name: str | None,
) -> tuple[torch.device, Dict[str, Any]]:
    device = torch.device(device_text)
    if device.type != "cuda":
        if device.type != "cpu":
            raise ValueError(
                f"smoke supports only CPU or CUDA, got {device.type!r}"
            )
        if expected_device_name is not None:
            raise ValueError("--expected-device-name is valid only for CUDA")
        return torch.device("cpu"), {
            "type": "cpu",
            "visible_cuda_devices": torch.cuda.device_count(),
            "name": "cpu",
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.index is None:
        device = torch.device("cuda:0")
    if device.index < 0 or device.index >= torch.cuda.device_count():
        raise ValueError(
            f"requested {device}, visible CUDA count={torch.cuda.device_count()}"
        )
    name = torch.cuda.get_device_name(device)
    if expected_device_name is not None and name != expected_device_name:
        raise RuntimeError(
            f"unexpected CUDA device: expected={expected_device_name!r}, "
            f"actual={name!r}"
        )
    return device, {
        "type": "cuda",
        "visible_cuda_devices": torch.cuda.device_count(),
        "index": int(device.index),
        "name": name,
    }


def _build_parent(variant: str):
    # SCTransNet reports deep supervision during construction.  Keep stdout
    # available for the final machine-readable smoke record.
    with contextlib.redirect_stdout(sys.stderr):
        return build_clean_v8_mprs_dch_model(variant, TRAINING_SEED)


def _build_pair(parent: nn.Module, variant: str):
    off = adapt_v8_mprs_dch_parent(
        parent,
        variant=variant,
        relay_enabled=False,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    on = adapt_v8_mprs_dch_parent(
        parent,
        variant=variant,
        relay_enabled=True,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    return off, on


def _paired_inputs(
    batch_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRAINING_SEED + INPUT_GENERATOR_OFFSET)
    inputs = torch.randn(
        batch_size,
        1,
        patch_size,
        patch_size,
        generator=generator,
    )
    targets = torch.rand(
        batch_size,
        1,
        patch_size,
        patch_size,
        generator=generator,
    )
    return inputs, targets


def _normalize_outputs(
    outputs: object,
    *,
    batch_size: int,
    patch_size: int,
    label: str,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(f"{label} did not return deep-supervision outputs")
    normalized = tuple(outputs)
    if len(normalized) != 6:
        raise RuntimeError(f"{label} returned {len(normalized)} outputs")
    expected_shape = (batch_size, 1, patch_size, patch_size)
    for index, output in enumerate(normalized):
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"{label}[{index}] shape={tuple(output.shape)}, "
                f"expected={expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise FloatingPointError(f"{label}[{index}] is not finite")
    return normalized


def _loss(
    outputs: Sequence[torch.Tensor],
    targets: torch.Tensor,
) -> torch.Tensor:
    criterion = nn.BCELoss(reduction="mean")
    value = sum(criterion(output, targets) for output in outputs)
    if not torch.isfinite(value):
        raise FloatingPointError("smoke loss is not finite")
    return value


def _gradient_l1(parameters: Sequence[nn.Parameter], label: str) -> float:
    value = 0.0
    count = 0
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError(f"{label} parameter has no gradient tensor")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"{label} gradient is not finite")
        value += float(parameter.grad.detach().abs().sum())
        count += 1
    if count == 0 or not math.isfinite(value):
        raise RuntimeError(f"{label} gradient summary is invalid")
    return value


def _all_parameters_finite(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )


def _state_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _state_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _state_values_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _batch_norm_update_counts(model: nn.Module) -> list[int]:
    return [
        int(module.num_batches_tracked.detach().cpu())
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.num_batches_tracked is not None
    ]


@contextlib.contextmanager
def _preserved_global_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if python_hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = python_hash_seed


def _validate_pair_state(off: nn.Module, on: nn.Module) -> Dict[str, Any]:
    off_state = off.state_dict()
    on_state = on.state_dict()
    extra = set(on_state) - set(off_state)
    expected_extra = {
        name for name in on_state if name.startswith("tpd_ner.")
    }
    if extra != expected_extra or set(off_state) - set(on_state):
        raise RuntimeError("relay on/off state-key relationship differs")
    for name, value in off_state.items():
        if not torch.equal(value, on_state[name]):
            raise RuntimeError(f"relay on/off common state differs at {name}")
    return {
        "common_state_exact": True,
        "extra_state_prefix": "tpd_ner.",
        "extra_state_key_count": len(extra),
    }


def _strict_checkpoint_roundtrip(
    *,
    parent: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    variant: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    checkpoint_output: Path | None,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        path = (
            checkpoint_output
            if checkpoint_output is not None
            else Path(temporary) / "tpd_ner_v8_smoke.pth.tar"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "variant": variant,
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "relay_width": RELAY_WIDTH,
            "relay_enabled": True,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(payload, path)
        checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = torch.load(path, map_location="cpu", weights_only=False)

        rebuilt = adapt_v8_mprs_dch_parent(
            parent,
            variant=variant,
            relay_enabled=True,
            relay_width=RELAY_WIDTH,
            relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        )
        incompatible = rebuilt.load_state_dict(
            loaded["state_dict"],
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict checkpoint reload returned mismatches")
        rebuilt.to(inputs.device)
        rebuilt_optimizer = torch.optim.Adam(rebuilt.parameters(), lr=1e-3)
        rebuilt_optimizer.load_state_dict(loaded["optimizer"])
        optimizer_state_exact = _state_values_equal(
            optimizer.state_dict(),
            rebuilt_optimizer.state_dict(),
        )
        if not optimizer_state_exact:
            raise RuntimeError("strict checkpoint optimizer state differs")

        model.eval()
        rebuilt.eval()
        with torch.no_grad():
            expected = tuple(model(inputs))
            actual = tuple(rebuilt(inputs))
        max_abs_difference = max(
            float((left - right).abs().max())
            for left, right in zip(expected, actual)
        )
        if max_abs_difference != 0.0:
            raise RuntimeError(
                "strict checkpoint reload output differs: "
                f"{max_abs_difference}"
            )

        model.train()
        rebuilt.train()
        optimizer.zero_grad(set_to_none=True)
        rebuilt_optimizer.zero_grad(set_to_none=True)
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(inputs.device)
            if inputs.device.type == "cuda"
            else None
        )
        expected_loss = _loss(
            _normalize_outputs(
                model(inputs),
                batch_size=int(inputs.shape[0]),
                patch_size=int(inputs.shape[-1]),
                label="paired_resume_original",
            ),
            targets,
        )
        expected_loss.backward()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, inputs.device)
        actual_loss = _loss(
            _normalize_outputs(
                rebuilt(inputs),
                batch_size=int(inputs.shape[0]),
                patch_size=int(inputs.shape[-1]),
                label="paired_resume_reloaded",
            ),
            targets,
        )
        actual_loss.backward()
        optimizer.step()
        rebuilt_optimizer.step()
        paired_loss_difference = abs(
            float(expected_loss.detach()) - float(actual_loss.detach())
        )
        paired_model_state_exact = _state_values_equal(
            model.state_dict(),
            rebuilt.state_dict(),
        )
        paired_optimizer_state_exact = _state_values_equal(
            optimizer.state_dict(),
            rebuilt_optimizer.state_dict(),
        )
        if (
            paired_loss_difference != 0.0
            or not paired_model_state_exact
            or not paired_optimizer_state_exact
        ):
            raise RuntimeError(
                "paired checkpoint continuation differs after one step"
            )
        return {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_preserved": checkpoint_output is not None,
            "strict_model_reload": True,
            "optimizer_reload": True,
            "optimizer_state_exact": optimizer_state_exact,
            "reload_output_max_abs_difference": max_abs_difference,
            "paired_resume_step_exact": True,
            "paired_resume_loss_difference": paired_loss_difference,
            "paired_resume_model_state_exact": paired_model_state_exact,
            "paired_resume_optimizer_state_exact": (
                paired_optimizer_state_exact
            ),
        }


@_preserved_global_rng()
def run_smoke(
    *,
    variant: str,
    device_text: str = "cpu",
    expected_device_name: str | None = None,
    batch_size: int = 2,
    patch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_output: Path | None = None,
) -> Dict[str, Any]:
    variant = variant.lower()
    if variant not in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
        raise ValueError(f"unsupported V8-MPRS-DCH variant: {variant}")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if patch_size < 32 or patch_size % 32:
        raise ValueError("patch_size must be a multiple of 32")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if RELAY_WIDTH != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("smoke relay width differs from the frozen width")
    if DEFAULT_RELAY_INITIALIZATION_SEED != TRAINING_SEED:
        raise RuntimeError("model and relay initialization seeds differ")

    torch.manual_seed(TRAINING_SEED)
    device, device_identity = _resolve_device(
        device_text,
        expected_device_name,
    )
    parent, parent_metadata = _build_parent(variant)
    parent_state_before = _model_state_sha256(parent)
    off, on = _build_pair(parent, variant)
    parent_state_after_adaptation = _model_state_sha256(parent)
    if parent_state_after_adaptation != parent_state_before:
        raise RuntimeError("adapting relay on/off changed the V8 parent state")

    pair_state = _validate_pair_state(off, on)
    if _parameter_count(off) != PRODUCTION_PARENT_PARAMETERS:
        raise RuntimeError("relay-off parameter count differs")
    if relay_parameter_count(on) != PRODUCTION_RELAY_PARAMETERS:
        raise RuntimeError("relay parameter count differs")
    if _parameter_count(on) != PRODUCTION_RELAY_ON_PARAMETERS:
        raise RuntimeError("relay-on total parameter count differs")

    parent.to(device)
    off.to(device)
    on.to(device)
    inputs_cpu, targets_cpu = _paired_inputs(batch_size, patch_size)
    inputs = inputs_cpu.to(device)
    targets = targets_cpu.to(device)

    parent.eval()
    off.eval()
    on.eval()
    with torch.no_grad():
        parent_outputs = _normalize_outputs(
            parent(inputs),
            batch_size=batch_size,
            patch_size=patch_size,
            label="v8_parent",
        )
        off_outputs = _normalize_outputs(
            off(inputs),
            batch_size=batch_size,
            patch_size=patch_size,
            label="relay_off",
        )
        on_outputs = _normalize_outputs(
            on(inputs),
            batch_size=batch_size,
            patch_size=patch_size,
            label="relay_on",
        )
    parent_off_differences = [
        float((left - right).abs().max())
        for left, right in zip(parent_outputs, off_outputs)
    ]
    if any(value != 0.0 for value in parent_off_differences):
        raise RuntimeError(
            "relay-off adapter differs from its V8 parent: "
            f"{parent_off_differences}"
        )
    step_zero_differences = [
        float((left - right).abs().max())
        for left, right in zip(off_outputs, on_outputs)
    ]
    if any(value != 0.0 for value in step_zero_differences):
        raise RuntimeError(
            f"relay step-zero outputs differ: {step_zero_differences}"
        )

    on.train()
    optimizer = torch.optim.Adam(on.parameters(), lr=learning_rate)
    losses = []
    gate_gradient_l1 = []
    fusion_gradient_l1 = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = _normalize_outputs(
            on(inputs),
            batch_size=batch_size,
            patch_size=patch_size,
            label=f"relay_on_step_{step + 1}",
        )
        loss = _loss(outputs, targets)
        loss.backward()
        gate_gradient_l1.append(
            _gradient_l1(
                tuple(on.tpd_ner.gates.parameters()),
                f"step-{step + 1} relay gate",
            )
        )
        fusion_gradient_l1.append(
            _gradient_l1(
                tuple(on.tpd_ner.fusions.parameters()),
                f"step-{step + 1} relay fusion",
            )
        )
        losses.append(float(loss.detach()))
        optimizer.step()
        if not _all_parameters_finite(on):
            raise FloatingPointError(
                f"relay-on parameters are not finite after step {step + 1}"
            )
    if gate_gradient_l1[0] <= 0.0:
        raise RuntimeError("zero gates did not receive a first-step gradient")
    if fusion_gradient_l1[0] != 0.0:
        raise RuntimeError("relay fusion activated before its zero gates opened")
    if fusion_gradient_l1[1] <= 0.0:
        raise RuntimeError("relay fusion did not activate on the second step")
    batch_norm_update_counts = _batch_norm_update_counts(on)
    if (
        not batch_norm_update_counts
        or set(batch_norm_update_counts) != {2}
    ):
        raise RuntimeError(
            "two-step checkpoint has an unexpected BatchNorm update count: "
            f"{batch_norm_update_counts}"
        )

    roundtrip = _strict_checkpoint_roundtrip(
        parent=parent,
        model=on,
        optimizer=optimizer,
        variant=variant,
        inputs=inputs,
        targets=targets,
        checkpoint_output=checkpoint_output,
    )
    parent_state_after_smoke = _model_state_sha256(parent)
    if parent_state_after_smoke != parent_state_before:
        raise RuntimeError("smoke training changed the original V8 parent state")
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "schema": SCHEMA,
        "status": "complete",
        "variant": variant,
        "training_seed": TRAINING_SEED,
        "relay_initialization_seed": DEFAULT_RELAY_INITIALIZATION_SEED,
        "split_seed": SPLIT_SEED,
        "seed_contract": (
            "model_and_relay_initialization_42;"
            "data_split_only_20260722"
        ),
        "relay_width": RELAY_WIDTH,
        "device": device_identity,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "learning_rate": learning_rate,
        "parent_metadata_variant": parent_metadata["variant"],
        "parent_state_sha256": parent_state_before,
        "parent_state_unchanged_after_adaptation": True,
        "parent_state_unchanged_after_smoke": True,
        "relay_off_parameters": _parameter_count(off),
        "relay_on_parameters": _parameter_count(on),
        "relay_parameters": relay_parameter_count(on),
        "output_count": 6,
        "outputs_finite": True,
        "step_zero_output_exact": True,
        "step_zero_max_abs_differences": step_zero_differences,
        "parent_relay_off_output_exact": True,
        "parent_relay_off_max_abs_differences": (
            parent_off_differences
        ),
        "losses": losses,
        "gate_gradient_l1": gate_gradient_l1,
        "fusion_gradient_l1": fusion_gradient_l1,
        "first_step_gate_active": gate_gradient_l1[0] > 0.0,
        "first_step_fusion_blocked": fusion_gradient_l1[0] == 0.0,
        "second_step_fusion_active": fusion_gradient_l1[1] > 0.0,
        "batch_norm_update_counts": batch_norm_update_counts,
        "train_forward_count": 2,
        "parameters_finite": True,
        "formal_training_started": False,
        **pair_state,
        **roundtrip,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V8-MPRS-DCH five-node NER smoke test"
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
        default="tpd_clean_v8_mprs_dch_full",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-device-name", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error("--patch-size must be a multiple of 32")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("--learning-rate must be finite and positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_smoke(
        variant=args.variant,
        device_text=args.device,
        expected_device_name=args.expected_device_name,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        learning_rate=args.learning_rate,
        checkpoint_output=args.checkpoint_output,
    )
    print(json.dumps(report, sort_keys=True))


__all__ = [
    "RELAY_WIDTH",
    "SCHEMA",
    "SPLIT_SEED",
    "TRAINING_SEED",
    "main",
    "parse_args",
    "run_smoke",
]


if __name__ == "__main__":
    main()
