#!/usr/bin/env python3
"""Two-step preflight for the isolated four-variant V5-NER matrix."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_tpd_ner_v5 import (  # noqa: E402
    SUPPORTED_TPD_NER_V5_VARIANTS,
    build_tpd_ner_v5_model,
)
from experiments.train_tpd_pilot import (  # noqa: E402
    deep_supervision_loss,
    model_checksum,
)
from model.Config import get_SCTrans_config  # noqa: E402
from model.tpd_clean_v5 import PRIMARY_CLEAN_V5_VARIANT  # noqa: E402
from model.tpd_ner_v5 import PROGRESSIVE_TOKENIZER  # noqa: E402


SCHEMA = "sctransnet_tpd_ner_v5_two_step_smoke_v1"
UINT32_MAX = 4_294_967_295
PAIR_VARIANTS = {
    "tpd_clean_v5_full": (
        "tpd_clean_v5_full_relay_off",
        "tpd_clean_v5_full_relay_on",
    ),
    "progressive": (
        "progressive_relay_off",
        "progressive_relay_on",
    ),
}
VARIANT_TO_PAIR = {
    variant: pair
    for pair, variants in PAIR_VARIANTS.items()
    for variant in variants
}
PRODUCTION_PARAMETER_CONTRACT = {
    "shallow_embedding_parameters": 66_176,
    "relay_off_total_parameters": 10_843_155,
    "relay_on_total_parameters": 10_854_446,
    "relay_parameters": 11_291,
    "relay_gate_parameters": 27,
}


def _validate_production_parameter_contract(
    metadata: Mapping[str, Any],
) -> None:
    """Require a production-dimension build to match every frozen count."""

    relay_enabled = bool(metadata["relay_enabled"])
    expected_total_key = (
        "relay_on_total_parameters"
        if relay_enabled
        else "relay_off_total_parameters"
    )
    actual = {
        "shallow_embedding_parameters": metadata[
            "shallow_embedding_parameters"
        ],
        expected_total_key: metadata["total_parameters"],
        "relay_parameters": metadata["relay_parameters"],
        "relay_gate_parameters": metadata["relay_gate_parameters"],
    }
    expected = {
        "shallow_embedding_parameters": PRODUCTION_PARAMETER_CONTRACT[
            "shallow_embedding_parameters"
        ],
        expected_total_key: PRODUCTION_PARAMETER_CONTRACT[
            expected_total_key
        ],
        "relay_parameters": (
            PRODUCTION_PARAMETER_CONTRACT["relay_parameters"]
            if relay_enabled
            else 0
        ),
        "relay_gate_parameters": (
            PRODUCTION_PARAMETER_CONTRACT["relay_gate_parameters"]
            if relay_enabled
            else 0
        ),
    }
    if actual != expected:
        raise RuntimeError(
            "production parameter contract mismatch: "
            f"variant={metadata['variant']} expected={expected} actual={actual}"
        )


def _next_seed(seed: int) -> int:
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError(f"seed must lie in [0, {UINT32_MAX}]")
    return (seed + 1) & UINT32_MAX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated V5-NER two-step preflight"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_TPD_NER_V5_VARIANTS,
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expected-device-name", default=None)
    parser.add_argument(
        "--full-dimensions",
        action="store_true",
        help="Use production channels/relay width on CPU instead of the light profile",
    )
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
            "V5-NER smoke requires exactly one visible CUDA device; "
            f"found {torch.cuda.device_count()}"
        )
    if device.index not in (None, 0):
        raise ValueError("with one visible CUDA device, use cuda or cuda:0")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if expected_device_name is not None and device_name != expected_device_name:
        raise RuntimeError(
            f"unexpected CUDA device: expected={expected_device_name!r}, "
            f"actual={device_name!r}"
        )
    return device, device_name


def _light_config():
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    return config


def _build_variant(
    variant: str,
    seed: int,
    *,
    patch_size: int,
    lightweight: bool,
):
    kwargs: Dict[str, Any] = {}
    if lightweight:
        kwargs = {
            "config": _light_config(),
            "img_size": patch_size,
            "relay_width": 2,
        }
    # SCTransNet prints its deep-supervision setting during construction.
    # Keep stdout reserved for the final machine-readable JSON line.
    with contextlib.redirect_stdout(sys.stderr):
        return build_tpd_ner_v5_model(variant, seed, **kwargs)


def _validate_outputs(
    outputs: Any,
    *,
    batch_size: int,
    patch_size: int,
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
            raise FloatingPointError(f"output {index} contains non-finite values")
    return normalized


def _paired_inputs(
    batch_size: int,
    patch_size: int,
    seed: int,
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
    targets = torch.rand(
        batch_size,
        1,
        patch_size,
        patch_size,
        generator=generator,
    )
    return inputs, targets


def _snapshot(
    parameters: Mapping[str, nn.Parameter],
) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in parameters.items()
    }


def _gradient_l1(
    parameters: Mapping[str, nn.Parameter],
    label: str,
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
    if not values:
        raise RuntimeError(f"{label} has no parameters")
    return values


def _zero_gradient_check(
    parameters: Mapping[str, nn.Parameter],
    label: str,
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"{label}.{name} has no gradient tensor")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"{label}.{name} gradient is not finite")
        value = float(gradient.detach().abs().sum().item())
        if value != 0.0:
            raise RuntimeError(f"{label}.{name} activated before its zero gate")
        values[name] = value
    if not values:
        raise RuntimeError(f"{label} has no parameters")
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


def _zero_update_check(
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
    if any(value != 0.0 for value in values.values()):
        raise RuntimeError(f"{label} updated before the gates opened")
    return values


def _tokenizer_controls(
    model: nn.Module,
    tokenizer_variant: str,
) -> tuple[str, Dict[str, nn.Parameter]]:
    if tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT:
        attribute = "saliency_scale"
        label = "v5_saliency_scale"
    elif tokenizer_variant == PROGRESSIVE_TOKENIZER:
        attribute = "channel_gain"
        label = "progressive_channel_gain"
    else:
        raise ValueError(f"unsupported tokenizer metadata: {tokenizer_variant}")
    parameters: Dict[str, nn.Parameter] = {}
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        for index, block in enumerate(embedding.blocks):
            parameter = getattr(block, attribute, None)
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(
                    f"{embedding_name}.blocks.{index}.{attribute} is missing"
                )
            parameters[
                f"{embedding_name}.blocks.{index}.{attribute}"
            ] = parameter
    if len(parameters) != 7:
        raise RuntimeError(f"expected seven {label} tensors, got {len(parameters)}")
    return label, parameters


def _relay_parameters(
    model: nn.Module,
) -> tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
    relay = model.tpd_ner
    gates = {
        f"{stage}.{name}": parameter
        for stage, gate in relay.gates.items()
        for name, parameter in gate.named_parameters()
    }
    fusions = {
        f"{stage}.{name}": parameter
        for stage, fusion in relay.fusions.items()
        for name, parameter in fusion.named_parameters()
    }
    if len(gates) != 6 or len(fusions) != 13:
        raise RuntimeError(
            f"unexpected relay tensors: gates={len(gates)} fusions={len(fusions)}"
        )
    return gates, fusions


def _pair_check(
    pair: str,
    *,
    inputs_cpu: torch.Tensor,
    device: torch.device,
    seed: int,
    patch_size: int,
    lightweight: bool,
) -> Dict[str, Any]:
    off_variant, on_variant = PAIR_VARIANTS[pair]
    off, off_metadata = _build_variant(
        off_variant,
        seed,
        patch_size=patch_size,
        lightweight=lightweight,
    )
    on, on_metadata = _build_variant(
        on_variant,
        seed,
        patch_size=patch_size,
        lightweight=lightweight,
    )
    off_state = off.state_dict()
    on_state = on.state_dict()
    missing_from_on = set(off_state) - set(on_state)
    extra_keys = set(on_state) - set(off_state)
    if missing_from_on:
        raise RuntimeError(f"{pair} relay-on lost common state keys")
    if not extra_keys or not all(key.startswith("tpd_ner.") for key in extra_keys):
        raise RuntimeError(f"{pair} relay-on added non-relay state")
    for name, tensor in off_state.items():
        if not torch.equal(tensor, on_state[name]):
            raise RuntimeError(f"{pair} common state differs at {name}")
    if (
        on_metadata["total_parameters"] - off_metadata["total_parameters"]
        != on_metadata["relay_parameters"]
    ):
        raise RuntimeError(f"{pair} parameter delta is not exactly the relay")
    if (
        off_metadata["shallow_embedding_parameters"]
        != on_metadata["shallow_embedding_parameters"]
    ):
        raise RuntimeError(f"{pair} shallow parameter count changed")
    if not lightweight:
        _validate_production_parameter_contract(off_metadata)
        _validate_production_parameter_contract(on_metadata)

    off.to(device)
    on.to(device)
    off.eval()
    on.eval()
    inputs = inputs_cpu.to(device)
    with torch.inference_mode():
        off_outputs = _validate_outputs(
            off(inputs),
            batch_size=inputs.shape[0],
            patch_size=patch_size,
        )
        on_outputs = _validate_outputs(
            on(inputs),
            batch_size=inputs.shape[0],
            patch_size=patch_size,
        )
        differences = [
            float((expected - actual).abs().max().item())
            for expected, actual in zip(off_outputs, on_outputs)
        ]
    if any(difference != 0.0 for difference in differences):
        raise RuntimeError(
            f"{pair} relay-on is not exact at step zero: {differences}"
        )
    report = {
        "pair": pair,
        "off_variant": off_variant,
        "on_variant": on_variant,
        "common_state_exact": True,
        "step_zero_output_exact": True,
        "step_zero_max_abs_difference": max(differences),
        "extra_state_key_count": len(extra_keys),
        "extra_state_prefix": "tpd_ner.",
        "shallow_embedding_parameters": off_metadata[
            "shallow_embedding_parameters"
        ],
        "relay_parameter_delta": on_metadata["relay_parameters"],
    }
    del off_outputs
    del on_outputs
    del off
    del on
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _run_variant(
    variant: str,
    *,
    inputs_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    device: torch.device,
    steps: int,
    seed: int,
    learning_rate: float,
    patch_size: int,
    lightweight: bool,
) -> Dict[str, Any]:
    model, metadata = _build_variant(
        variant,
        seed,
        patch_size=patch_size,
        lightweight=lightweight,
    )
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != metadata["total_parameters"]:
        raise RuntimeError(f"{variant} metadata total parameter mismatch")
    if metadata["fourth_parallel_branch_added"] is not False:
        raise RuntimeError(f"{variant} changed the frozen branch contract")
    if metadata["shallow_parameter_match_verified"] is not True:
        raise RuntimeError(f"{variant} lacks the shallow capacity check")
    if not lightweight:
        _validate_production_parameter_contract(metadata)
    relay_enabled = bool(metadata["relay_enabled"])
    if relay_enabled != hasattr(model, "tpd_ner"):
        raise RuntimeError(f"{variant} relay metadata/model mismatch")
    initial_checksum = model_checksum(model)
    if initial_checksum != metadata["full_initialization_sha256"]:
        raise RuntimeError(f"{variant} initialization checksum mismatch")

    model.to(device)
    inputs = inputs_cpu.to(device)
    targets = targets_cpu.to(device)
    model.eval()
    with torch.inference_mode():
        initial_outputs = _validate_outputs(
            model(inputs),
            batch_size=inputs.shape[0],
            patch_size=patch_size,
        )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="mean")
    control_kind, controls = _tokenizer_controls(
        model,
        str(metadata["tokenizer_variant"]),
    )
    controls_before = _snapshot(controls)
    gates: Dict[str, nn.Parameter] = {}
    fusions: Dict[str, nn.Parameter] = {}
    gates_before: Dict[str, torch.Tensor] = {}
    fusions_before_step1: Dict[str, torch.Tensor] = {}
    if relay_enabled:
        gates, fusions = _relay_parameters(model)
        gates_before = _snapshot(gates)
        fusions_before_step1 = _snapshot(fusions)

    losses: list[float] = []
    control_gradient_l1: Dict[str, float] = {}
    control_update_l1: Dict[str, float] = {}
    gate_gradient_l1: Dict[str, float] = {}
    gate_update_l1: Dict[str, float] = {}
    fusion_step1_zero_gradient: Dict[str, float] = {}
    fusion_step1_zero_update: Dict[str, float] = {}
    fusion_gradient_l1: Dict[str, float] = {}
    fusion_update_l1: Dict[str, float] = {}

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        fusions_before_step2 = (
            _snapshot(fusions)
            if relay_enabled and step_index == 1
            else {}
        )
        outputs = _validate_outputs(
            model(inputs),
            batch_size=inputs.shape[0],
            patch_size=patch_size,
        )
        loss = deep_supervision_loss(outputs, targets, criterion)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError(
                f"{variant} step {step_index + 1} produced invalid loss"
            )
        loss.backward()
        losses.append(float(loss.detach().item()))

        if step_index == 0:
            control_gradient_l1 = _gradient_l1(controls, control_kind)
            if relay_enabled:
                gate_gradient_l1 = _gradient_l1(gates, "relay_gate")
                fusion_step1_zero_gradient = _zero_gradient_check(
                    fusions,
                    "relay_fusion_step1",
                )
        elif step_index == 1 and relay_enabled:
            fusion_gradient_l1 = _gradient_l1(
                fusions,
                "relay_fusion_step2",
            )

        optimizer.step()
        if step_index == 0:
            control_update_l1 = _update_l1(
                controls_before,
                controls,
                control_kind,
            )
            if relay_enabled:
                gate_update_l1 = _update_l1(
                    gates_before,
                    gates,
                    "relay_gate",
                )
                fusion_step1_zero_update = _zero_update_check(
                    fusions_before_step1,
                    fusions,
                    "relay_fusion_step1",
                )
        elif step_index == 1 and relay_enabled:
            fusion_update_l1 = _update_l1(
                fusions_before_step2,
                fusions,
                "relay_fusion_step2",
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trained_checksum = model_checksum(model)
    if trained_checksum == initial_checksum:
        raise RuntimeError(f"{variant} model checksum did not change")
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    model.eval()
    reload_inputs = inputs[: min(2, inputs.shape[0])]
    with torch.inference_mode():
        source_outputs = tuple(
            output.detach().clone()
            for output in _validate_outputs(
                model(reload_inputs),
                batch_size=reload_inputs.shape[0],
                patch_size=patch_size,
            )
        )

    rebuilt, _ = _build_variant(
        variant,
        _next_seed(seed),
        patch_size=patch_size,
        lightweight=lightweight,
    )
    incompatible = rebuilt.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{variant} strict reload reported incompatibility")
    if model_checksum(rebuilt) != trained_checksum:
        raise RuntimeError(f"{variant} checksum changed after strict reload")
    rebuilt.to(device)
    rebuilt.eval()
    with torch.inference_mode():
        rebuilt_outputs = _validate_outputs(
            rebuilt(reload_inputs),
            batch_size=reload_inputs.shape[0],
            patch_size=patch_size,
        )
        reload_differences = [
            float((source - actual).abs().max().item())
            for source, actual in zip(source_outputs, rebuilt_outputs)
        ]
    if any(difference != 0.0 for difference in reload_differences):
        raise RuntimeError(
            f"{variant} strict reload changed outputs: {reload_differences}"
        )

    report = {
        "variant": variant,
        "status": "complete",
        "output_count": len(initial_outputs),
        "losses": losses,
        "control_kind": control_kind,
        "control_gradient_l1": control_gradient_l1,
        "control_update_l1": control_update_l1,
        "relay_enabled": relay_enabled,
        "gate_gradient_l1": gate_gradient_l1,
        "gate_update_l1": gate_update_l1,
        "fusion_step1_zero_gradient": fusion_step1_zero_gradient,
        "fusion_step1_zero_update": fusion_step1_zero_update,
        "fusion_gradient_l1": fusion_gradient_l1,
        "fusion_update_l1": fusion_update_l1,
        "strict_rebuild_load": True,
        "strict_reload_max_abs_difference": max(reload_differences),
        "initial_model_checksum": initial_checksum,
        "trained_model_checksum": trained_checksum,
        "parameter_counts": {
            "total": metadata["total_parameters"],
            "trainable": metadata["trainable_parameters"],
            "common": metadata["common_parameters"],
            "shallow_embedding": metadata["shallow_embedding_parameters"],
            "relay": metadata["relay_parameters"],
            "relay_gate": metadata["relay_gate_parameters"],
        },
        "metadata_contract": {
            "candidate_family": metadata["candidate_family"],
            "construction_schema": metadata["construction_schema"],
            "tokenizer_variant": metadata["tokenizer_variant"],
            "relay_topology": metadata["relay_topology"],
            "evidence_nodes": metadata["relay_taps"]["evidence_nodes"],
            "fourth_parallel_branch_added": metadata[
                "fourth_parallel_branch_added"
            ],
            "shallow_parameter_match_verified": metadata[
                "shallow_parameter_match_verified"
            ],
            "capacity_contract": metadata["capacity_contract"],
            "production_parameter_contract_verified": not lightweight,
        },
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
    variant: str = "all",
    device_text: str = "cpu",
    batch_size: int = 2,
    patch_size: int = 32,
    steps: int = 2,
    seed: int = 42,
    learning_rate: float = 1e-3,
    expected_device_name: str | None = None,
    lightweight_cpu: bool = True,
) -> Dict[str, Any]:
    if variant not in ("all",) + SUPPORTED_TPD_NER_V5_VARIANTS:
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
    lightweight = bool(lightweight_cpu and device.type == "cpu")
    dimension_profile = (
        "cpu_light_base4_relay2" if lightweight else "production_base32_relay8"
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    inputs_cpu, targets_cpu = _paired_inputs(batch_size, patch_size, seed)
    variants = (
        SUPPORTED_TPD_NER_V5_VARIANTS if variant == "all" else (variant,)
    )
    pairs = tuple(
        pair
        for pair in PAIR_VARIANTS
        if any(VARIANT_TO_PAIR[current] == pair for current in variants)
    )
    pair_checks = [
        _pair_check(
            pair,
            inputs_cpu=inputs_cpu,
            device=device,
            seed=seed,
            patch_size=patch_size,
            lightweight=lightweight,
        )
        for pair in pairs
    ]
    reports = [
        _run_variant(
            current,
            inputs_cpu=inputs_cpu,
            targets_cpu=targets_cpu,
            device=device,
            steps=steps,
            seed=seed,
            learning_rate=learning_rate,
            patch_size=patch_size,
            lightweight=lightweight,
        )
        for current in variants
    ]

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
        "requested_variant": variant,
        "variants": reports,
        "pair_checks": pair_checks,
        "off_on_step_zero_exact": all(
            check["common_state_exact"] and check["step_zero_output_exact"]
            for check in pair_checks
        ),
        "device": str(device),
        "device_name": device_name,
        "dimension_profile": dimension_profile,
        "lightweight_cpu": lightweight,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "steps": steps,
        "seed": seed,
        "cuda_memory": cuda_memory,
        "production_parameter_contract": PRODUCTION_PARAMETER_CONTRACT,
        "production_parameter_contract_verified_in_this_run": not lightweight,
        "formal_training_started": False,
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
        lightweight_cpu=not args.full_dimensions,
    )
    print(
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


__all__ = [
    "PAIR_VARIANTS",
    "PRODUCTION_PARAMETER_CONTRACT",
    "SCHEMA",
    "SUPPORTED_TPD_NER_V5_VARIANTS",
    "run_smoke",
    "main",
]


if __name__ == "__main__":
    main()
