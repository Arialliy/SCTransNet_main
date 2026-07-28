#!/usr/bin/env python3
"""Two-step CPU/CUDA smoke for the sole formal V2 NER candidate."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import smoke_tpd_ner_v8_mprs_dch as v1_smoke  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
)
from model.tpd_ner_v8_mprs_dch import (  # noqa: E402
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    adapt_v8_mprs_dch_parent,
)
from model.tpd_ner_v8_mprs_dch_v2 import (  # noqa: E402
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_V2_RELAY_ON_PARAMETERS,
    PRODUCTION_V2_RELAY_PARAMETERS,
    RELAY_RMS_EPS,
    TPDNERV8MPRSDCHV2SCTransNet,
    V2_MASK_LIMIT,
    adapt_v8_mprs_dch_parent_v2,
    v2_relay_parameter_count,
)


SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_two_step_smoke_v1"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
RELAY_WIDTH = 8
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON = (
    "tpd_ner_v8_mprs_dch_v2_full_relay_on"
)
V1_RELAY_OFF_REFERENCE = "tpd_ner_v8_mprs_dch_full_relay_off"

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


@contextlib.contextmanager
def _deterministic_smoke_execution(
    device_text: str,
) -> Iterator[Dict[str, Any]]:
    """Scope the formal exact CUDA execution contract to this smoke run.

    The checkpoint comparison executes the original and reloaded models
    sequentially.  CUDA convolution backward may otherwise use reduction
    orders that are numerically valid but not byte-identical, which would make
    a correct checkpoint look non-resumable.  The formal trainer already uses
    these settings; the smoke must establish them itself as well.
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
    previous_float32_precision = torch.get_float32_matmul_precision()
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cuda_initialized_before_contract = (
        torch.cuda.is_initialized() if is_cuda else False
    )
    if (
        is_cuda
        and cuda_initialized_before_contract
        and previous_workspace != CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUDA was initialized before the deterministic smoke contract; "
            "start a fresh process or pre-set "
            f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}"
        )

    if is_cuda:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
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
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if is_cuda
            else None
        ),
        "cuda_initialized_before_contract": (
            cuda_initialized_before_contract if is_cuda else None
        ),
        "purpose": "byte_exact_checkpoint_continuation",
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
            torch.set_float32_matmul_precision(
                previous_float32_precision
            )
            if previous_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


def _clone_function(
    function: Any,
    bindings: Mapping[str, Any],
) -> Any:
    namespace = dict(function.__globals__)
    namespace.update(bindings)
    cloned = types.FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
    return cloned


def _build_pair(
    parent: nn.Module,
    parent_variant: str,
) -> tuple[nn.Module, TPDNERV8MPRSDCHV2SCTransNet]:
    if parent_variant != PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT:
        raise ValueError("V2 smoke supports only the Full parent")
    off = adapt_v8_mprs_dch_parent(
        parent,
        variant=parent_variant,
        relay_enabled=False,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    on = adapt_v8_mprs_dch_parent_v2(
        parent,
        variant=parent_variant,
        relay_enabled=True,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    if type(on) is not TPDNERV8MPRSDCHV2SCTransNet:
        raise TypeError("V2 smoke did not build the exact V2 model class")
    return off, on


def _strict_checkpoint_roundtrip_v2(
    *,
    parent: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    variant: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    checkpoint_output: Path | None,
) -> Dict[str, Any]:
    del variant

    def rebuild_v2(
        parent_model: nn.Module,
        *,
        variant: str,
        relay_enabled: bool,
        relay_width: int,
        relay_initialization_seed: int,
    ) -> nn.Module:
        if variant != TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON:
            raise ValueError("V2 smoke checkpoint variant differs")
        if relay_enabled is not True:
            raise ValueError("V2 smoke checkpoint must be relay-on")
        return adapt_v8_mprs_dch_parent_v2(
            parent_model,
            variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            relay_enabled=True,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
        )

    strict = _clone_function(
        v1_smoke._strict_checkpoint_roundtrip,
        {
            "SCHEMA": SCHEMA,
            "TRAINING_SEED": TRAINING_SEED,
            "SPLIT_SEED": SPLIT_SEED,
            "RELAY_WIDTH": RELAY_WIDTH,
            "adapt_v8_mprs_dch_parent": rebuild_v2,
            "_state_values_equal": _state_values_equal,
            "_normalize_outputs": _normalize_outputs,
            "_loss": _loss,
        },
    )
    return strict(
        parent=parent,
        model=model,
        optimizer=optimizer,
        variant=TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
        inputs=inputs,
        targets=targets,
        checkpoint_output=checkpoint_output,
    )


def run_smoke(
    *,
    variant: str = TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
    device_text: str = "cpu",
    expected_device_name: str | None = None,
    batch_size: int = 2,
    patch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_output: Path | None = None,
) -> Dict[str, Any]:
    """Run step-zero identity, two updates, and strict V2 reload."""

    if variant != TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON:
        raise ValueError("V2 smoke accepts only the formal V2 relay-on variant")
    observations: list[dict[str, Any]] = []
    instrumented_relays: list[nn.Module] = []

    def instrumented_pair(
        parent: nn.Module,
        parent_variant: str,
    ) -> tuple[nn.Module, TPDNERV8MPRSDCHV2SCTransNet]:
        off, on = _build_pair(parent, parent_variant)
        relay = on.tpd_ner
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
            observations.append(
                {
                    "stage": int(stage),
                    "relay_value_finite": bool(torch.isfinite(value).all()),
                    "mask_finite": bool(torch.isfinite(mask).all()),
                    "mask_abs_max": float(mask.detach().abs().max().cpu()),
                    "per_sample_rms": [float(item) for item in per_sample_rms],
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
            "PRODUCTION_RELAY_PARAMETERS": PRODUCTION_V2_RELAY_PARAMETERS,
            "PRODUCTION_RELAY_ON_PARAMETERS": (
                PRODUCTION_V2_RELAY_ON_PARAMETERS
            ),
            "relay_parameter_count": v2_relay_parameter_count,
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
                _strict_checkpoint_roundtrip_v2
            ),
        },
    )
    try:
        with _deterministic_smoke_execution(
            device_text
        ) as deterministic_runtime:
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
    finally:
        for relay in instrumented_relays:
            if "forward_stage" in relay.__dict__:
                delattr(relay, "forward_stage")

    if not observations:
        raise RuntimeError("V2 smoke did not observe relay stages")
    rms_values = [
        value
        for observation in observations
        for value in observation["per_sample_rms"]
    ]
    if (
        not all(observation["relay_value_finite"] for observation in observations)
        or not all(observation["mask_finite"] for observation in observations)
        or not all(math.isfinite(value) for value in rms_values)
    ):
        raise FloatingPointError("V2 relay/RMS smoke observations are non-finite")
    mask_abs_max = max(
        float(observation["mask_abs_max"])
        for observation in observations
    )
    if mask_abs_max >= V2_MASK_LIMIT:
        raise RuntimeError("V2 arctangent mask left its strict open bounds")

    report.update(
        {
            "schema": SCHEMA,
            "variant": TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "control_variant": V1_RELAY_OFF_REFERENCE,
            "relay_off_retrained": False,
            "relay_version": "v2_rms_centered_arctangent",
            "relay_rms_eps": RELAY_RMS_EPS,
            "relay_value_observation_count": len(observations),
            "relay_values_finite": True,
            "relay_rms_finite": True,
            "relay_rms_min": min(rms_values),
            "relay_rms_max": max(rms_values),
            "relay_masks_finite": True,
            "relay_mask_abs_max": mask_abs_max,
            "relay_masks_within_open_bounds": True,
            "deterministic_runtime": deterministic_runtime,
        }
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V2 RMS-balanced centered-gate NER smoke"
    )
    parser.add_argument(
        "--variant",
        choices=(TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,),
        default=TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
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
    "RELAY_WIDTH",
    "SCHEMA",
    "SPLIT_SEED",
    "TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON",
    "TRAINING_SEED",
    "V1_RELAY_OFF_REFERENCE",
    "_deterministic_smoke_execution",
    "main",
    "parse_args",
    "run_smoke",
]


if __name__ == "__main__":
    main()
