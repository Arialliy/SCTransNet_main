#!/usr/bin/env python3
"""Two-step CPU/CUDA smoke for the sole formal V3 NER candidate."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import smoke_tpd_ner_v8_mprs_dch as v1_smoke  # noqa: E402
from experiments import smoke_tpd_ner_v8_mprs_dch_v2 as v2_smoke  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
)
from model.tpd_ner_v8_mprs_dch import (  # noqa: E402
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
)
from model.tpd_ner_v8_mprs_dch_v3 import (  # noqa: E402
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_V3_RELAY_ON_PARAMETERS,
    PRODUCTION_V3_RELAY_PARAMETERS,
    RELAY_RMS_EPS,
    RELAY_STAGE_ORDER,
    TPDNERV8MPRSDCHV3SCTransNet,
    V2_MASK_LIMIT,
    V3_RELAY_VERSION,
    adapt_v8_mprs_dch_parent_v3,
    v3_relay_parameter_count,
)


SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v3_two_step_smoke_v1"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
RELAY_WIDTH = 8
CUBLAS_WORKSPACE_CONFIG = v2_smoke.CUBLAS_WORKSPACE_CONFIG
TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON = (
    "tpd_ner_v8_mprs_dch_v3_full_relay_on"
)
V1_RELAY_OFF_REFERENCE = "tpd_ner_v8_mprs_dch_full_relay_off"
V2_RELAY_ON_REFERENCE = v2_smoke.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON
OFFSET_STAGES = tuple(str(stage) for stage in RELAY_STAGE_ORDER)
OFFSET_STATE_KEYS = tuple(
    f"tpd_ner.dc_offsets.{stage}" for stage in OFFSET_STAGES
)
EXPECTED_RELAY_STATE_KEY_COUNT = 19

_resolve_device = v1_smoke._resolve_device
_build_parent = v1_smoke._build_parent
_paired_inputs = v1_smoke._paired_inputs
_normalize_outputs = v1_smoke._normalize_outputs
_loss = v1_smoke._loss
_gradient_l1 = v1_smoke._gradient_l1
_all_parameters_finite = v1_smoke._all_parameters_finite
_state_values_equal = v1_smoke._state_values_equal
_batch_norm_update_counts = v1_smoke._batch_norm_update_counts
_preserved_global_rng = v1_smoke._preserved_global_rng
_clone_function = v2_smoke._clone_function
_deterministic_smoke_execution = v2_smoke._deterministic_smoke_execution


def _capture_global_rng_state(
    *,
    include_cuda: bool = False,
) -> Dict[str, Any]:
    """Capture global RNGs without initializing CUDA for a CPU smoke."""

    cuda_states = None
    if include_cuda or torch.cuda.is_initialized():
        cuda_states = tuple(
            state.detach().cpu().clone()
            for state in torch.cuda.get_rng_state_all()
        )
    numpy_state = np.random.get_state()
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state().detach().cpu().clone(),
        "cuda": cuda_states,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def _global_rng_states_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_numpy = left["numpy"]
    right_numpy = right["numpy"]
    numpy_equal = (
        left_numpy[0] == right_numpy[0]
        and np.array_equal(left_numpy[1], right_numpy[1])
        and left_numpy[2:] == right_numpy[2:]
    )
    left_cuda = left["cuda"]
    right_cuda = right["cuda"]
    if left_cuda is None or right_cuda is None:
        cuda_equal = left_cuda is None and right_cuda is None
    else:
        cuda_equal = len(left_cuda) == len(right_cuda) and all(
            torch.equal(a, b) for a, b in zip(left_cuda, right_cuda)
        )
    return bool(
        left["python"] == right["python"]
        and numpy_equal
        and torch.equal(left["torch"], right["torch"])
        and cuda_equal
        and left["python_hash_seed"] == right["python_hash_seed"]
    )


def _offset_parameters(
    model: nn.Module,
) -> Dict[str, nn.Parameter]:
    relay = getattr(model, "tpd_ner", None)
    offsets = getattr(relay, "dc_offsets", None)
    if not isinstance(offsets, nn.ParameterDict):
        raise TypeError("V3 smoke requires a relay ParameterDict of DC offsets")
    if set(offsets) != set(OFFSET_STAGES):
        raise RuntimeError(
            "V3 smoke DC offset stages differ: "
            f"actual={sorted(offsets)}, expected={sorted(OFFSET_STAGES)}"
        )
    result = {stage: offsets[stage] for stage in OFFSET_STAGES}
    for stage, parameter in result.items():
        if tuple(parameter.shape) != (1,):
            raise RuntimeError(
                f"V3 stage {stage} DC offset has shape "
                f"{tuple(parameter.shape)}"
            )
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(
                f"V3 stage {stage} DC offset is not finite"
            )
    return result


def _offset_values(model: nn.Module) -> Dict[str, float]:
    return {
        stage: float(parameter.detach().cpu())
        for stage, parameter in _offset_parameters(model).items()
    }


def _build_pair(
    parent: nn.Module,
    parent_variant: str,
) -> tuple[nn.Module, TPDNERV8MPRSDCHV3SCTransNet]:
    if parent_variant != PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT:
        raise ValueError("V3 smoke supports only the Full parent")
    off = adapt_v8_mprs_dch_parent_v3(
        parent,
        variant=parent_variant,
        relay_enabled=False,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    on = adapt_v8_mprs_dch_parent_v3(
        parent,
        variant=parent_variant,
        relay_enabled=True,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    if type(on) is not TPDNERV8MPRSDCHV3SCTransNet:
        raise TypeError("V3 smoke did not build the exact V3 model class")
    return off, on


def _strict_checkpoint_roundtrip_v3(
    *,
    parent: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    variant: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    checkpoint_output: Path | None,
) -> Dict[str, Any]:
    """Run the V2-template strict continuation and verify V3 offsets."""

    del variant
    offsets_before = {
        stage: parameter.detach().cpu().clone()
        for stage, parameter in _offset_parameters(model).items()
    }
    if any(
        not bool(torch.isfinite(value).all())
        or int(torch.count_nonzero(value)) != 1
        for value in offsets_before.values()
    ):
        raise RuntimeError(
            "all three V3 DC offsets must be finite and nonzero "
            "after two updates"
        )

    def rebuild_v3(
        parent_model: nn.Module,
        *,
        variant: str,
        relay_enabled: bool,
        relay_width: int,
        relay_initialization_seed: int,
    ) -> nn.Module:
        if variant != TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON:
            raise ValueError("V3 smoke checkpoint variant differs")
        if relay_enabled is not True:
            raise ValueError("V3 smoke checkpoint must be relay-on")
        rebuilt = adapt_v8_mprs_dch_parent_v3(
            parent_model,
            variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            relay_enabled=True,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
        )
        if type(rebuilt) is not TPDNERV8MPRSDCHV3SCTransNet:
            raise TypeError(
                "V3 smoke checkpoint did not rebuild the exact V3 class"
            )
        return rebuilt

    strict = _clone_function(
        v1_smoke._strict_checkpoint_roundtrip,
        {
            "SCHEMA": SCHEMA,
            "TRAINING_SEED": TRAINING_SEED,
            "SPLIT_SEED": SPLIT_SEED,
            "RELAY_WIDTH": RELAY_WIDTH,
            "adapt_v8_mprs_dch_parent": rebuild_v3,
            "_state_values_equal": _state_values_equal,
            "_normalize_outputs": _normalize_outputs,
            "_loss": _loss,
        },
    )

    checkpoint_requested = checkpoint_output is not None
    with tempfile.TemporaryDirectory() as temporary:
        path = (
            checkpoint_output
            if checkpoint_output is not None
            else Path(temporary) / "tpd_ner_v8_mprs_dch_v3_smoke.pth.tar"
        )
        report = strict(
            parent=parent,
            model=model,
            optimizer=optimizer,
            variant=TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
            inputs=inputs,
            targets=targets,
            checkpoint_output=path,
        )
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        state = loaded.get("state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("V3 smoke checkpoint state_dict is not a mapping")
        checkpoint_offset_keys = {
            name for name in state if name.startswith("tpd_ner.dc_offsets.")
        }
        if checkpoint_offset_keys != set(OFFSET_STATE_KEYS):
            raise RuntimeError(
                "V3 smoke checkpoint DC offset keys differ: "
                f"{sorted(checkpoint_offset_keys)}"
            )
        for stage, expected in offsets_before.items():
            actual = state[f"tpd_ner.dc_offsets.{stage}"].detach().cpu()
            if not torch.equal(actual, expected):
                raise RuntimeError(
                    f"V3 stage {stage} DC offset changed in checkpoint"
                )

        rebuilt = rebuild_v3(
            parent,
            variant=TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
            relay_enabled=True,
            relay_width=RELAY_WIDTH,
            relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        )
        incompatible = rebuilt.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "independent V3 strict reload returned incompatibilities"
            )
        reloaded_offsets = _offset_parameters(rebuilt)
        for stage, expected in offsets_before.items():
            actual = reloaded_offsets[stage].detach().cpu()
            if not torch.equal(actual, expected):
                raise RuntimeError(
                    f"strict reload changed V3 stage {stage} DC offset"
                )

    report.update(
        {
            "checkpoint_preserved": checkpoint_requested,
            "checkpoint_dc_offset_keys": list(OFFSET_STATE_KEYS),
            "checkpoint_dc_offset_values": {
                stage: float(value)
                for stage, value in offsets_before.items()
            },
            "checkpoint_dc_offsets_nonzero": True,
            "checkpoint_dc_offsets_exact": True,
            "strict_reload_dc_offsets_exact": True,
        }
    )
    return report


def run_smoke(
    *,
    variant: str = TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
    device_text: str = "cpu",
    expected_device_name: str | None = None,
    batch_size: int = 2,
    patch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_output: Path | None = None,
) -> Dict[str, Any]:
    """Run step-zero identity, two updates, and strict V3 reload."""

    if variant != TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON:
        raise ValueError("V3 smoke accepts only the formal V3 relay-on variant")

    observations: list[dict[str, Any]] = []
    instrumented_relays: list[nn.Module] = []
    gradient_handles: list[Any] = []
    offset_gradients: Dict[str, list[float]] = {
        stage: [] for stage in OFFSET_STAGES
    }
    initial_offsets: Dict[str, float] | None = None

    def instrumented_pair(
        parent: nn.Module,
        parent_variant: str,
    ) -> tuple[nn.Module, TPDNERV8MPRSDCHV3SCTransNet]:
        nonlocal initial_offsets
        off, on = _build_pair(parent, parent_variant)
        relay = on.tpd_ner
        offsets = _offset_parameters(on)
        initial_offsets = _offset_values(on)
        if any(value != 0.0 for value in initial_offsets.values()):
            raise RuntimeError("V3 smoke DC offsets are not zero initialized")

        for stage, parameter in offsets.items():

            def observe_gradient(
                gradient: torch.Tensor,
                *,
                current_stage: str = stage,
            ) -> torch.Tensor:
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(
                        f"V3 stage {current_stage} DC offset gradient "
                        "is not finite"
                    )
                value = float(gradient.detach().abs().sum().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"V3 stage {current_stage} gradient summary "
                        "is not finite"
                    )
                offset_gradients[current_stage].append(value)
                return gradient

            gradient_handles.append(
                parameter.register_hook(observe_gradient)
            )

        original_forward_stage = relay.forward_stage

        def observed_forward_stage(
            self: nn.Module,
            stage: int,
            sources: Sequence[torch.Tensor],
            output_size: tuple[int, int],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del self
            value, mask = original_forward_stage(
                stage,
                sources,
                output_size,
            )
            working = value.detach().float()
            per_sample_rms = (
                working.square()
                .mean(dim=(1, 2, 3))
                .sqrt()
                .cpu()
                .tolist()
            )
            offset_value = float(
                offsets[str(stage)].detach().cpu()
            )
            observations.append(
                {
                    "stage": int(stage),
                    "relay_value_finite": bool(torch.isfinite(value).all()),
                    "mask_finite": bool(torch.isfinite(mask).all()),
                    "mask_abs_max": float(
                        mask.detach().abs().max().cpu()
                    ),
                    "dc_offset": offset_value,
                    "dc_offset_finite": math.isfinite(offset_value),
                    "per_sample_rms": [
                        float(item) for item in per_sample_rms
                    ],
                }
            )
            return value, mask

        relay.forward_stage = types.MethodType(  # type: ignore[method-assign]
            observed_forward_stage,
            relay,
        )
        instrumented_relays.append(relay)
        return off, on

    raw_run = _clone_function(
        v1_smoke.run_smoke.__wrapped__,
        {
            "SCHEMA": SCHEMA,
            "TRAINING_SEED": TRAINING_SEED,
            "SPLIT_SEED": SPLIT_SEED,
            "RELAY_WIDTH": RELAY_WIDTH,
            "SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS": (
                PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            ),
            "PRODUCTION_PARENT_PARAMETERS": PRODUCTION_PARENT_PARAMETERS,
            "PRODUCTION_RELAY_PARAMETERS": PRODUCTION_V3_RELAY_PARAMETERS,
            "PRODUCTION_RELAY_ON_PARAMETERS": (
                PRODUCTION_V3_RELAY_ON_PARAMETERS
            ),
            "relay_parameter_count": v3_relay_parameter_count,
            "_resolve_device": _resolve_device,
            "_build_parent": _build_parent,
            "_build_pair": instrumented_pair,
            "_paired_inputs": _paired_inputs,
            "_normalize_outputs": _normalize_outputs,
            "_loss": _loss,
            "_gradient_l1": _gradient_l1,
            "_all_parameters_finite": _all_parameters_finite,
            "_state_values_equal": _state_values_equal,
            "_batch_norm_update_counts": _batch_norm_update_counts,
            "_strict_checkpoint_roundtrip": (
                _strict_checkpoint_roundtrip_v3
            ),
        },
    )

    try:
        with _deterministic_smoke_execution(
            device_text
        ) as deterministic_runtime:
            include_cuda_rng = torch.device(device_text).type == "cuda"
            rng_before = _capture_global_rng_state(
                include_cuda=include_cuda_rng,
            )
            with _preserved_global_rng():
                report = raw_run(
                    variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
                    device_text=device_text,
                    expected_device_name=expected_device_name,
                    batch_size=batch_size,
                    patch_size=patch_size,
                    learning_rate=learning_rate,
                    checkpoint_output=checkpoint_output,
                )
            rng_after = _capture_global_rng_state(
                include_cuda=include_cuda_rng,
            )
            if not _global_rng_states_equal(rng_before, rng_after):
                raise RuntimeError("V3 smoke changed global RNG state")
    finally:
        for handle in gradient_handles:
            handle.remove()
        for relay in instrumented_relays:
            if "forward_stage" in relay.__dict__:
                delattr(relay, "forward_stage")

    if initial_offsets is None:
        raise RuntimeError("V3 smoke did not capture initial DC offsets")
    if not observations:
        raise RuntimeError("V3 smoke did not observe relay stages")
    rms_values = [
        value
        for observation in observations
        for value in observation["per_sample_rms"]
    ]
    if (
        not all(
            observation["relay_value_finite"]
            for observation in observations
        )
        or not all(observation["mask_finite"] for observation in observations)
        or not all(
            observation["dc_offset_finite"]
            for observation in observations
        )
        or not all(math.isfinite(value) for value in rms_values)
    ):
        raise FloatingPointError(
            "V3 relay/RMS/DC-offset smoke observations are non-finite"
        )
    mask_abs_max = max(
        float(observation["mask_abs_max"])
        for observation in observations
    )
    if mask_abs_max >= V2_MASK_LIMIT:
        raise RuntimeError("V3 arctangent mask left its strict open bounds")

    first_two_gradients: Dict[str, list[float]] = {}
    for stage in OFFSET_STAGES:
        values = offset_gradients[stage]
        if len(values) < 2:
            raise RuntimeError(
                f"V3 stage {stage} observed only {len(values)} gradients"
            )
        first_two_gradients[stage] = values[:2]
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in values[:2]
        ):
            raise RuntimeError(
                f"V3 stage {stage} DC offset did not receive finite "
                "nonzero gradients in both updates"
            )
    offset_gradient_l1 = [
        sum(first_two_gradients[stage][step] for stage in OFFSET_STAGES)
        for step in range(2)
    ]

    trained_offsets = report.get("checkpoint_dc_offset_values")
    if not isinstance(trained_offsets, Mapping):
        raise RuntimeError("V3 smoke did not report checkpoint DC offsets")
    offset_update_l1 = {
        stage: abs(float(trained_offsets[stage]) - initial_offsets[stage])
        for stage in OFFSET_STAGES
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in offset_update_l1.values()
    ):
        raise RuntimeError("each V3 DC offset must update after two steps")
    if report.get("extra_state_key_count") != EXPECTED_RELAY_STATE_KEY_COUNT:
        raise RuntimeError(
            "V3 relay state-key count differs: "
            f"{report.get('extra_state_key_count')}"
        )

    report.update(
        {
            "schema": SCHEMA,
            "variant": TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "control_variant": V1_RELAY_OFF_REFERENCE,
            "structural_predecessor": V2_RELAY_ON_REFERENCE,
            "relay_off_retrained": False,
            "relay_version": V3_RELAY_VERSION,
            "relay_rms_eps": RELAY_RMS_EPS,
            "relay_state_key_count": EXPECTED_RELAY_STATE_KEY_COUNT,
            "dc_offset_count": len(OFFSET_STAGES),
            "dc_offset_stages": list(OFFSET_STAGES),
            "dc_offset_state_keys": list(OFFSET_STATE_KEYS),
            "dc_offset_initial_values": dict(initial_offsets),
            "dc_offset_gradient_l1": first_two_gradients,
            "offset_gradient_l1_by_stage": first_two_gradients,
            "offset_gradient_l1": offset_gradient_l1,
            "dc_offset_gradients_finite": True,
            "dc_offset_gradients_nonzero": True,
            "dc_offset_update_l1": offset_update_l1,
            "offset_update_l1": offset_update_l1,
            "dc_offsets_updated": True,
            "dc_offsets_nonzero_after_two_steps": True,
            "relay_value_observation_count": len(observations),
            "relay_values_finite": True,
            "relay_rms_finite": True,
            "relay_rms_min": min(rms_values),
            "relay_rms_max": max(rms_values),
            "relay_masks_finite": True,
            "relay_mask_abs_max": mask_abs_max,
            "relay_masks_within_open_bounds": True,
            "global_rng_preserved": True,
            "deterministic_runtime": deterministic_runtime,
        }
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V3 post-centering-DC RMS-balanced NER smoke"
        )
    )
    parser.add_argument(
        "--variant",
        choices=(TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,),
        default=TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
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
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
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
    "CUBLAS_WORKSPACE_CONFIG",
    "EXPECTED_RELAY_STATE_KEY_COUNT",
    "OFFSET_STAGES",
    "OFFSET_STATE_KEYS",
    "RELAY_WIDTH",
    "SCHEMA",
    "SPLIT_SEED",
    "TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON",
    "TRAINING_SEED",
    "V1_RELAY_OFF_REFERENCE",
    "V2_RELAY_ON_REFERENCE",
    "_deterministic_smoke_execution",
    "main",
    "parse_args",
    "run_smoke",
]


if __name__ == "__main__":
    main()
