#!/usr/bin/env python3
"""Exact seed-42 trainer for the sole V4 tail-aware NER candidate.

The numerical loop, data split, loss, optimizer, schedule, validation metrics,
and best/best-mIoU selectors are inherited from the already verified V3 exact
trainer.  V4 owns a distinct trajectory identity, schemas, source-lock key,
run directory, checkpoint contract, and resume guard.

Only the parameter-free ``complement_tail`` formula is trainable here.
Its stage thresholds are immutable architecture constants.  Fresh training
always constructs a new V8-MPRS-DCH parent and V4 relay; V3 warm-start is not
supported.  Exact resume accepts only the same V4 epoch-boundary trajectory.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v3_exact as v3_exact,
)
from experiments.train_tpd_clean_v8_mprs_dch import (  # noqa: E402
    build_clean_v8_mprs_dch_model,
)
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (  # noqa: E402
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_TAIL_Z_THRESHOLDS,
    PRODUCTION_V4_RELAY_ON_PARAMETERS,
    PRODUCTION_V4_RELAY_PARAMETERS,
    RELAY_RMS_EPS,
    TPDNERV8MPRSDCHV4SCTransNet,
    V4_RELAY_VERSION,
    adapt_v8_mprs_dch_parent_v4,
    v4_relay_parameter_count,
)


ENTRY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_exact_entry_v1"
)
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock_v1"
)
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "exact_architecture_manifest_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "exact_checkpoint_identity_v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_exact_checkpoint_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "completion_summary_v1"
)
SOURCE_LOCK_KEY = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock"
)
RUN_ID_PREFIX = "tpd-ner-v8-mprs-dch-v4-tail-aware-exact:"
FORMAL_RUN_TAG = "formal800_exact_v4_tail_aware_seed42"
CANDIDATE_FAMILY = (
    "tpd_clean_v8_mprs_dch_explicit_five_node_ner_"
    "v4_tail_aware_complement"
)
CONSTRUCTION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "five_node_complement_tail_v1"
)

TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
)
V3_RELAY_ON_VARIANT = (
    v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
)
REQUIRED_CONTROL = v3_exact.V8_PARENT_RELAY_OFF_REFERENCE
PAIRED_GATE_PREDECESSOR = v3_exact.V2_RELAY_ON_VARIANT
STRUCTURAL_PREDECESSOR = V3_RELAY_ON_VARIANT
FALLBACK_CANDIDATE_VARIANTS = (
    TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON,
)

DC_SUPPORT_MODE = DEFAULT_DC_SUPPORT_MODE
DC_SUPPORT_FORMULA_STAGE4 = "1"
DC_SUPPORT_FORMULA_STAGE3_2 = "1-P"
DC_SUPPORT_SCOPE = (
    "post_centering_ner_gate_offset_not_tokenizer_mprs_dch"
)
TAIL_Z_THRESHOLDS = {
    str(stage): float(value)
    for stage, value in DEFAULT_TAIL_Z_THRESHOLDS.items()
}

EVALUATOR_CHECKPOINT_REQUIRED_FIELDS = (
    *v3_exact.EVALUATOR_CHECKPOINT_REQUIRED_FIELDS,
    "ner_dc_offset_support_scope",
    "dc_support_mode",
    "dc_support_formula_stage4",
    "dc_support_formula_stage3_2",
    "tail_z_thresholds",
    "tail_z_thresholds_frozen",
    "target_protective_complement",
    "formula_selection_decision",
    "formula_selection_aggregate_sha256",
    "formula_selection_marker_sha256",
    "paired_gate_predecessor",
    "fresh_training",
    "warm_start_applied",
)

DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1"
)
PROTOCOL_DRAFT_PATH = (
    REPO_ROOT
    / "experiments/TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PROTOCOL.md"
)
FORMULA_SELECTION_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual_v1/"
    "NUDT-SIRST"
)
FORMULA_SELECTION_AGGREGATE_PATH = (
    FORMULA_SELECTION_ROOT
    / "tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual_aggregate.json"
)
FORMULA_SELECTION_MARKER_PATH = (
    FORMULA_SELECTION_ROOT / "POSTPROCESS_COMPLETE.json"
)
FORMULA_SELECTION_AGGREGATE_SHA256 = (
    "07f6d9b5bdabcc5df1a323485bbf590fb8e297a8361998816cfb256b803ae3d7"
)
FORMULA_SELECTION_MARKER_SHA256 = (
    "2cea9183f2ce6197df5d6425b0aac306121ad96366231b9f5453e9caf24d2d76"
)

TRAINING_SEED = v3_exact.TRAINING_SEED
SPLIT_SEED = v3_exact.SPLIT_SEED
RELAY_WIDTH = v3_exact.RELAY_WIDTH
RELAY_INITIALIZATION_SEED = v3_exact.RELAY_INITIALIZATION_SEED
FORMAL_EPOCHS = v3_exact.FORMAL_EPOCHS
FORMAL_BATCH_SIZE = v3_exact.FORMAL_BATCH_SIZE
FORMAL_PATCH_SIZE = v3_exact.FORMAL_PATCH_SIZE
FORMAL_WORKERS = v3_exact.FORMAL_WORKERS
FORMAL_VAL_FRACTION = v3_exact.FORMAL_VAL_FRACTION
FORMAL_EVAL_EVERY = v3_exact.FORMAL_EVAL_EVERY
FORMAL_BASE_LR = v3_exact.FORMAL_BASE_LR
FORMAL_MIN_LR = v3_exact.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = v3_exact.FORMAL_WARMUP_EPOCHS
FORMAL_THRESHOLD = v3_exact.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = v3_exact.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = v3_exact.FORMAL_TINY_AREA
FORMAL_AMP = v3_exact.FORMAL_AMP
FORMAL_EPS = v3_exact.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = (
    v3_exact.FORMAL_CUBLAS_WORKSPACE_CONFIG
)
FORMAL_INITIALIZATION_MODES = v3_exact.FORMAL_INITIALIZATION_MODES
PHYSICAL_GPU_UUIDS = dict(v3_exact.PHYSICAL_GPU_UUIDS)
SELECTION_METRICS = v3_exact.SELECTION_METRICS
STORED_VALIDATION_METRICS = v3_exact.STORED_VALIDATION_METRICS

v8_kernel = v3_exact.v8_kernel
file_sha256 = v3_exact.file_sha256
canonical_sha256 = v3_exact.canonical_sha256
load_json_mapping = v3_exact.load_json_mapping
PreparedData = v3_exact.PreparedData
prepare_data = v3_exact.prepare_data
split_fingerprints = v3_exact.split_fingerprints
data_fingerprints = v3_exact.data_fingerprints
configure_determinism = v3_exact.configure_determinism
write_or_verify_json = v3_exact.write_or_verify_json
base = v3_exact.base
shared_exact = v3_exact.shared_exact
InitializationPlan = v3_exact.InitializationPlan


def _ordered_unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


# The source lock is intentionally not generated by this module.  Once these
# sources are frozen, a separate sealing step must create the V4-owned lock.
RUNTIME_SOURCE_PATHS = _ordered_unique_paths(
    (
        Path(__file__).resolve(),
        PROTOCOL_DRAFT_PATH,
        REPO_ROOT / "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py",
        *v3_exact.RUNTIME_SOURCE_PATHS,
    )
)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _normalized_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("V4 tail thresholds must be a mapping")
    try:
        normalized = {
            str(stage): float(threshold)
            for stage, threshold in value.items()
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("V4 tail thresholds are invalid") from exc
    if normalized != TAIL_Z_THRESHOLDS:
        raise ValueError(
            "V4 tail thresholds differ from the frozen architecture"
        )
    return normalized


def supported_candidate_variants() -> tuple[str, ...]:
    return FALLBACK_CANDIDATE_VARIANTS


def candidate_contract(candidate_variant: str) -> dict[str, Any]:
    if not isinstance(candidate_variant, str):
        raise TypeError("V4 candidate variant must be a string")
    if candidate_variant != (
        TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON
    ):
        raise ValueError(
            f"unsupported V4 exact variant {candidate_variant!r}; "
            f"choices={FALLBACK_CANDIDATE_VARIANTS}"
        )
    return {
        "candidate_variant": candidate_variant,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V4_RELAY_VERSION,
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "required_control": REQUIRED_CONTROL,
        "relay_off_retrained": False,
        "dc_support_mode": DC_SUPPORT_MODE,
        "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
    }


def formula_selection_contract() -> dict[str, Any]:
    for label, path, expected_sha in (
        (
            "aggregate",
            FORMULA_SELECTION_AGGREGATE_PATH,
            FORMULA_SELECTION_AGGREGATE_SHA256,
        ),
        (
            "completion marker",
            FORMULA_SELECTION_MARKER_PATH,
            FORMULA_SELECTION_MARKER_SHA256,
        ),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"V4 formula-selection {label} is not an immutable file"
            )
        _require_equal(
            f"V4 formula-selection {label} SHA",
            file_sha256(path),
            expected_sha,
        )
    aggregate = load_json_mapping(
        FORMULA_SELECTION_AGGREGATE_PATH,
        "V4 formula-selection aggregate",
    )
    marker = load_json_mapping(
        FORMULA_SELECTION_MARKER_PATH,
        "V4 formula-selection completion marker",
    )
    for label, payload in (("aggregate", aggregate), ("marker", marker)):
        _require_equal(
            f"V4 formula-selection {label} status",
            payload.get("status"),
            "complete",
        )
        _require_equal(
            f"V4 formula-selection {label} decision",
            payload.get("decision"),
            "COMPLEMENT_TAIL_SELECTED",
        )
        _require_equal(
            f"V4 formula-selection {label} selected mode",
            payload.get("selected_formula_mode"),
            DC_SUPPORT_MODE,
        )
        _require_equal(
            f"V4 formula-selection {label} selection flag",
            payload.get("formal_v4_formula_selected"),
            True,
        )
        _require_equal(
            f"V4 formula-selection {label} authorization scope",
            payload.get("formal_training_authorized_by_this_artifact"),
            False,
        )
    _require_equal(
        "V4 qualifying local formula modes",
        aggregate.get("qualifying_local_formula_modes"),
        [DC_SUPPORT_MODE],
    )
    aggregate_record = marker.get("aggregate_json")
    if not isinstance(aggregate_record, Mapping):
        raise ValueError("V4 selection marker lacks aggregate identity")
    _require_equal(
        "V4 marker aggregate SHA",
        aggregate_record.get("sha256"),
        FORMULA_SELECTION_AGGREGATE_SHA256,
    )
    return {
        "decision": "COMPLEMENT_TAIL_SELECTED",
        "selected_formula_mode": DC_SUPPORT_MODE,
        "qualifying_local_formula_modes": [DC_SUPPORT_MODE],
        "aggregate_path": str(
            FORMULA_SELECTION_AGGREGATE_PATH.relative_to(REPO_ROOT)
        ),
        "aggregate_sha256": FORMULA_SELECTION_AGGREGATE_SHA256,
        "completion_marker_path": str(
            FORMULA_SELECTION_MARKER_PATH.relative_to(REPO_ROOT)
        ),
        "completion_marker_sha256": FORMULA_SELECTION_MARKER_SHA256,
        "selection_artifact_role": (
            "formula_selection_only_not_training_authorization"
        ),
        "formal_training_authorized_by_selection_artifact": False,
    }


def formal_contract() -> dict[str, Any]:
    inherited = copy.deepcopy(v3_exact.formal_contract())
    inherited.update(
        {
            "candidate_variants": list(supported_candidate_variants()),
            "relay_version": V4_RELAY_VERSION,
            "zero_gate_reference": "v3_v2_and_relay_off_exact",
            "structural_predecessor": STRUCTURAL_PREDECESSOR,
            "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
            "required_control": REQUIRED_CONTROL,
            "mask_mapping": (
                "atan(pi*(centered+dc*selected_support))/pi"
            ),
            "dc_support_mode": DC_SUPPORT_MODE,
            "dc_support_formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
            "dc_support_formula_stage3_2": (
                DC_SUPPORT_FORMULA_STAGE3_2
            ),
            "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
            "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
            "tail_z_thresholds_frozen": True,
            "tail_support_parameters": 0,
            "tail_support_buffers": 0,
            "target_protective_complement": True,
            "fresh_training": True,
            "v3_warm_start": False,
            "formula_selection": formula_selection_contract(),
            "physical_gpu_index_environment": (
                "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_INDEX"
            ),
            "physical_gpu_uuid_environment": (
                "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_UUID"
            ),
        }
    )
    return inherited


def _validate_formal_args(args: argparse.Namespace) -> None:
    candidate_contract(getattr(args, "variant", None))
    expected = {
        "dataset": "NUDT-SIRST",
        "variant": TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON,
        "epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "patch_size": FORMAL_PATCH_SIZE,
        "eval_every": FORMAL_EVAL_EVERY,
        "workers": FORMAL_WORKERS,
        "val_fraction": FORMAL_VAL_FRACTION,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "threshold": FORMAL_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "run_tag": FORMAL_RUN_TAG,
        "max_train_images": None,
        "max_val_images": None,
        "dc_support_mode": DC_SUPPORT_MODE,
        "tail_z_thresholds": TAIL_Z_THRESHOLDS,
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            "formal V4 tail-aware exact arguments differ: "
            f"expected={expected}, observed={observed}"
        )
    device = getattr(args, "device", None)
    allow_cpu_smoke = getattr(args, "allow_cpu_smoke", False)
    if allow_cpu_smoke:
        if device != "cpu":
            raise ValueError("CPU smoke permission requires --device=cpu")
    elif device != "cuda:0":
        raise ValueError(
            "formal V4 tail-aware exact training requires --device=cuda:0"
        )


def _option_present(arguments: Sequence[str], option: str) -> bool:
    return any(
        value == option or value.startswith(f"{option}=")
        for value in arguments
    )


def _replace_option_value(
    arguments: list[str],
    option: str,
    value: str,
) -> None:
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            arguments[index + 1] = value
            return
        if argument.startswith(f"{option}="):
            arguments[index] = f"{option}={value}"
            return
    raise ValueError(f"V4 exact entry requires {option}")


def _selected_option(
    arguments: Sequence[str],
    option: str,
) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            return (
                None
                if index + 1 >= len(arguments)
                else arguments[index + 1]
            )
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return None


def _emit_v4_help() -> None:
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            v3_exact.parse_args(["--help"])
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    help_text = output.getvalue()
    help_text = help_text.replace(
        Path(v3_exact.__file__).name,
        Path(__file__).name,
    )
    help_text = help_text.replace(
        v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON,
    )
    help_text = help_text.replace(
        "V3 post-centering DC-calibrated",
        "V4 complement-tail-aware",
    )
    sys.stdout.write(help_text)
    raise SystemExit(0)


def _mapped_v3_arguments(arguments: Sequence[str]) -> list[str]:
    forbidden = (
        "--relay-enabled",
        "--relay-off",
        "--no-relay-enabled",
        "--relay-width",
        "--relay-initialization-seed",
        "--dc-support-mode",
        "--tail-z-thresholds",
        "--tail-z-threshold-stage4",
        "--tail-z-threshold-stage3",
        "--tail-z-threshold-stage2",
    )
    for argument in arguments:
        if any(
            argument == option or argument.startswith(f"{option}=")
            for option in forbidden
        ):
            raise ValueError(
                "V4 relay formula and tail thresholds are fixed "
                "architecture constants"
            )
    selected = _selected_option(arguments, "--variant")
    if selected != (
        TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON
    ):
        raise ValueError(
            "V4 exact entry accepts only "
            "--variant="
            f"{TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON}"
        )
    selected_tag = _selected_option(arguments, "--run-tag")
    if selected_tag not in (None, FORMAL_RUN_TAG):
        raise ValueError(f"V4 run tag is fixed to {FORMAL_RUN_TAG}")
    mapped = list(arguments)
    _replace_option_value(mapped, "--variant", V3_RELAY_ON_VARIANT)
    if selected_tag is None:
        mapped.extend(("--run-tag", v3_exact.FORMAL_RUN_TAG))
    else:
        _replace_option_value(
            mapped,
            "--run-tag",
            v3_exact.FORMAL_RUN_TAG,
        )
    return mapped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value in ("-h", "--help") for value in arguments):
        _emit_v4_help()
    mapped = _mapped_v3_arguments(arguments)
    args = v3_exact.parse_args(mapped)
    args.variant = TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON
    args.parent_variant = PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT
    args.relay_enabled = True
    args.relay_width = RELAY_WIDTH
    args.relay_initialization_seed = RELAY_INITIALIZATION_SEED
    args.run_tag = FORMAL_RUN_TAG
    args.dc_support_mode = DC_SUPPORT_MODE
    args.tail_z_thresholds = dict(TAIL_Z_THRESHOLDS)
    if not _option_present(arguments, "--output-root"):
        args.output_root = DEFAULT_OUTPUT_ROOT
    if not _option_present(arguments, "--exact-source-lock"):
        args.exact_source_lock = DEFAULT_EXACT_SOURCE_LOCK_PATH
    _validate_formal_args(args)
    return args


def run_directory(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    return (
        args.output_root.resolve()
        / args.dataset
        / args.variant
        / f"seed_{args.seed}_{args.run_tag}"
    )


def _required_determinism() -> dict[str, Any]:
    return {
        "entry_schema": ENTRY_SCHEMA,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V4_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "relay_rms_eps": RELAY_RMS_EPS,
        "gate_bias": False,
        "gate_spatial_centering": "per_sample_mean_hw",
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": (
            "atan(pi*(centered+dc*selected_support))/pi"
        ),
        "zero_gate_reference": "v3_v2_and_relay_off_exact",
        "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
        "dc_support_mode": DC_SUPPORT_MODE,
        "dc_support_formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
        "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
        "formula_selection_aggregate_sha256": (
            FORMULA_SELECTION_AGGREGATE_SHA256
        ),
        "formula_selection_marker_sha256": (
            FORMULA_SELECTION_MARKER_SHA256
        ),
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "required_control": REQUIRED_CONTROL,
        "fresh_training": True,
        "v3_warm_start": False,
        "scheduler_restore_mode": (
            "identity_bound_manual_schedule_from_completed_epoch"
        ),
    }


def require_v4_run_identity(
    identity: Any,
    *,
    label: str,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no V4 exact run identity")
    value = copy.deepcopy(dict(identity))
    variant = value.get("variant")
    run_id = value.get("run_id")
    source_locks = value.get("source_locks")
    training = value.get("training_contract")
    determinism = (
        training.get("determinism")
        if isinstance(training, Mapping)
        else None
    )
    entry_schema = (
        determinism.get("entry_schema")
        if isinstance(determinism, Mapping)
        else None
    )
    if entry_schema != ENTRY_SCHEMA:
        raise ValueError(f"{label} entry schema is not V4 tail-aware")
    if variant != (
        TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON
    ):
        raise ValueError(f"{label} variant is not the sole V4 candidate")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"{label} variant {variant!r} differs from {expected_variant!r}"
        )
    if not isinstance(run_id, str) or not run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"{label} run_id is not V4 tail-aware")
    forbidden_lock_keys = {
        v3_exact.SOURCE_LOCK_KEY,
        v3_exact.v1_exact.SOURCE_LOCK_KEY,
        v3_exact.V2_EXACT_SOURCE_LOCK_KEY,
        v8_kernel.SOURCE_LOCK_KEY,
    }
    if (
        not isinstance(source_locks, Mapping)
        or set(source_locks) != {SOURCE_LOCK_KEY, "training_data"}
        or forbidden_lock_keys.intersection(source_locks)
    ):
        raise ValueError(f"{label} source-lock identity is not V4")
    if not isinstance(determinism, Mapping):
        raise ValueError(f"{label} determinism identity is missing")
    for name, expected in _required_determinism().items():
        observed = determinism.get(name)
        if name == "tail_z_thresholds":
            _normalized_thresholds(observed)
        elif observed != expected:
            raise ValueError(
                f"{label} V4 determinism field {name} differs"
            )
    _require_equal(f"{label} training seed", value.get("seed"), TRAINING_SEED)
    _require_equal(f"{label} split seed", value.get("split_seed"), SPLIT_SEED)
    return value


def _existing_training_contract(
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, Any]:
    protocol = load_json_mapping(
        directory / "protocol.json",
        "existing V4 tail-aware exact protocol",
    )
    _require_equal("existing protocol schema", protocol.get("schema"), ENTRY_SCHEMA)
    identity = require_v4_run_identity(
        protocol.get("run_identity"),
        label="existing V4 protocol",
        expected_variant=args.variant,
    )
    training = identity.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError("existing V4 protocol has no training contract")
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(
            f"existing V4 training contract lacks fields: {missing}"
        )
    return copy.deepcopy(dict(training))


def initialization_plan(
    args: argparse.Namespace,
    directory: Path,
    model: nn.Module,
) -> InitializationPlan:
    _validate_formal_args(args)
    if args.fresh:
        return InitializationPlan(
            request=exact_runner.InitializationRequest.fresh(),
            contract=exact_runner.fresh_initialization_contract(),
            initial_model_state_sha256=(
                exact_runner.initial_model_state_sha256(model)
            ),
        )
    if args.exact_resume:
        training = _existing_training_contract(args, directory)
        initial_rng = training["initial_rng"]
        selection_policy = training["selection_policy"]
        if not isinstance(initial_rng, Mapping):
            raise ValueError("existing V4 initial_rng is not a mapping")
        if not isinstance(selection_policy, Mapping):
            raise ValueError(
                "existing V4 selection_policy is not a mapping"
            )
        return InitializationPlan(
            request=exact_runner.InitializationRequest.exact(),
            contract=copy.deepcopy(training["initialization_contract"]),
            initial_model_state_sha256=str(
                training["initial_model_state_sha256"]
            ),
            initial_rng=copy.deepcopy(dict(initial_rng)),
            selection_policy=copy.deepcopy(dict(selection_policy)),
        )
    raise RuntimeError("V4 exact entry requires --fresh or --exact-resume")


def _require_v4_manifest(
    manifest: Any,
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("V4 architecture manifest is missing")
    value = copy.deepcopy(dict(manifest))
    expected = {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "relay_enabled": True,
        "relay_version": V4_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
        "dc_support_mode": DC_SUPPORT_MODE,
        "dc_support_formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
        "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "tail_support_parameters": 0,
        "tail_support_buffers": 0,
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
        "formula_selection_aggregate_sha256": (
            FORMULA_SELECTION_AGGREGATE_SHA256
        ),
        "formula_selection_marker_sha256": (
            FORMULA_SELECTION_MARKER_SHA256
        ),
        "zero_gate_reference": "v3_v2_and_relay_off_exact",
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "required_control": REQUIRED_CONTROL,
        "exact_resume_scope": "same_v4_tail_aware_variant_only",
        "cross_version_exact_resume_supported": False,
    }
    for name, required in expected.items():
        _require_equal(f"V4 manifest {name}", value.get(name), required)
    _normalized_thresholds(value.get("tail_z_thresholds"))
    _require_equal(
        "V4 manifest stage-4 support",
        value.get("ner_dc_offset_support_stage4"),
        "global_v3_exact",
    )
    _require_equal(
        "V4 manifest stage-3 support",
        value.get("ner_dc_offset_support_stage3"),
        "stopgrad_one_minus_geomean_tail_q3_q4",
    )
    _require_equal(
        "V4 manifest stage-2 support",
        value.get("ner_dc_offset_support_stage2"),
        "stopgrad_one_minus_geomean_tail_q2_q3",
    )
    return value


def _architecture_manifest(
    variant: str,
    model: TPDNERV8MPRSDCHV4SCTransNet,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    relay_manifest = model.architecture_manifest()
    relay_keys = tuple(
        name for name in model.state_dict() if name.startswith("tpd_ner.")
    )
    manifest = {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "model": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware."
            "TPDNERV8MPRSDCHV4SCTransNet"
        ),
        "parent_model": "model.SCTransNet.SCTransNet",
        "candidate_family": metadata.get("candidate_family"),
        "mainline_contract": relay_manifest["mainline_contract"],
        "semantic_sources": relay_manifest["semantic_sources"],
        "embedding_replacements": ("mtc.embeddings_1", "mtc.embeddings_2"),
        "evidence_nodes": relay_manifest["evidence_nodes"],
        "evidence_layout": relay_manifest["evidence_layout"],
        "relay_stage_order": relay_manifest["relay_stage_order"],
        "relay_enabled": True,
        "relay_version": relay_manifest["relay_version"],
        "relay_width": relay_manifest["relay_width"],
        "relay_initialization_seed": model.relay_initialization_seed,
        "relay_state_prefix": "tpd_ner.",
        "relay_state_key_count": len(relay_keys),
        "relay_parameters": v4_relay_parameter_count(model),
        "relay_rms_scope": relay_manifest["relay_rms_scope"],
        "relay_rms_eps": relay_manifest["relay_rms_eps"],
        "gate_bias": relay_manifest["gate_bias"],
        "gate_spatial_centering": relay_manifest["gate_spatial_centering"],
        "gate_dc_offset": relay_manifest["gate_dc_offset"],
        "gate_dc_offset_count": relay_manifest["gate_dc_offset_count"],
        "gate_dc_offset_initialization": relay_manifest[
            "gate_dc_offset_initialization"
        ],
        "gate_dc_offset_state_prefix": relay_manifest[
            "gate_dc_offset_state_prefix"
        ],
        "mask_mapping": relay_manifest["mask_mapping"],
        "mask_bounds": relay_manifest["mask_bounds"],
        "skip_factor_bounds": relay_manifest["skip_factor_bounds"],
        "ner_dc_offset_support_scope": relay_manifest[
            "ner_dc_offset_support_scope"
        ],
        "ner_dc_offset_support_stage4": relay_manifest[
            "ner_dc_offset_support_stage4"
        ],
        "ner_dc_offset_support_stage3": relay_manifest[
            "ner_dc_offset_support_stage3"
        ],
        "ner_dc_offset_support_stage2": relay_manifest[
            "ner_dc_offset_support_stage2"
        ],
        "dc_support_mode": relay_manifest[
            "ner_dc_offset_support_mode"
        ],
        "dc_support_formula_stage4": relay_manifest[
            "ner_dc_offset_support_formula_stage4"
        ],
        "dc_support_formula_stage3_2": relay_manifest[
            "ner_dc_offset_support_formula_stage3_2"
        ],
        "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
        "tail_z_thresholds_frozen": relay_manifest[
            "tail_z_thresholds_frozen"
        ],
        "tail_support_parameters": relay_manifest[
            "tail_support_parameters"
        ],
        "tail_support_buffers": relay_manifest["tail_support_buffers"],
        "tail_support_gradient": relay_manifest["tail_support_gradient"],
        "target_protective_complement": relay_manifest[
            "target_protective_complement"
        ],
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
        "formula_selection_aggregate_sha256": (
            FORMULA_SELECTION_AGGREGATE_SHA256
        ),
        "formula_selection_marker_sha256": (
            FORMULA_SELECTION_MARKER_SHA256
        ),
        "state_compatible_with": relay_manifest[
            "state_compatible_with"
        ],
        "zero_gate_reference": relay_manifest["zero_gate_reference"],
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "required_control": REQUIRED_CONTROL,
        "relay_off_retrained": False,
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "deep_supervision_outputs": 6,
        "loss": "unweighted sum of BCE over six post-sigmoid outputs",
        "exact_resume_scope": "same_v4_tail_aware_variant_only",
        "cross_version_exact_resume_supported": False,
        "fresh_training": True,
        "v3_warm_start": False,
        "eps": FORMAL_EPS,
        "formal_amp": FORMAL_AMP,
    }
    return _require_v4_manifest(manifest, variant=variant)


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    candidate_contract(variant)
    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError("V4 exact builder requires seed=42")
    if eps != FORMAL_EPS:
        raise ValueError(f"V4 exact builder requires eps={FORMAL_EPS}")
    parent, parent_metadata = build_clean_v8_mprs_dch_model(
        PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        seed,
    )
    if not isinstance(parent_metadata, Mapping):
        raise TypeError("V8 parent builder metadata is not a mapping")
    model = adapt_v8_mprs_dch_parent_v4(
        parent,
        variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_enabled=True,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=RELAY_INITIALIZATION_SEED,
        dc_support_mode=DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    if type(model) is not TPDNERV8MPRSDCHV4SCTransNet:
        raise TypeError("V4 exact builder requires the exact V4 model class")
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
        or model.tokenizer_variant != PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT
        or model.relay_width != RELAY_WIDTH
        or model.relay_initialization_seed != RELAY_INITIALIZATION_SEED
        or model.tpd_ner.dc_support_mode != DC_SUPPORT_MODE
        or _normalized_thresholds(
            model.tpd_ner.tail_z_thresholds
        )
        != TAIL_Z_THRESHOLDS
    ):
        raise ValueError("V4 exact model identity differs")
    relay_keys = [
        name for name in model.state_dict() if name.startswith("tpd_ner.")
    ]
    _require_equal("V4 relay state-key count", len(relay_keys), 19)
    _require_equal(
        "V4 relay parameter count",
        v4_relay_parameter_count(model),
        PRODUCTION_V4_RELAY_PARAMETERS,
    )
    _require_equal(
        "V4 total parameter count",
        sum(parameter.numel() for parameter in model.parameters()),
        PRODUCTION_V4_RELAY_ON_PARAMETERS,
    )
    relay_manifest = model.architecture_manifest()
    _require_equal(
        "V4 selected support mode",
        relay_manifest.get("ner_dc_offset_support_mode"),
        DC_SUPPORT_MODE,
    )
    _require_equal(
        "V4 selected support formula",
        relay_manifest.get("ner_dc_offset_support_formula_stage3_2"),
        DC_SUPPORT_FORMULA_STAGE3_2,
    )
    _normalized_thresholds(relay_manifest.get("tail_z_thresholds"))
    metadata: dict[str, Any] = {
        "variant": variant,
        "candidate_family": CANDIDATE_FAMILY,
        "construction_schema": CONSTRUCTION_SCHEMA,
        "comparison_role": "tpd_plus_ner_v4_complement_tail",
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "required_control": REQUIRED_CONTROL,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "parent_candidate_family": parent_metadata.get("candidate_family"),
        "parent_model_metadata": copy.deepcopy(dict(parent_metadata)),
        "relay_enabled": True,
        "relay_version": V4_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "relay_parameters": PRODUCTION_V4_RELAY_PARAMETERS,
        "relay_state_prefix": "tpd_ner.",
        "relay_state_key_count": len(relay_keys),
        "relay_rms_eps": relay_manifest["relay_rms_eps"],
        "gate_bias": False,
        "gate_spatial_centering": "per_sample_mean_hw",
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": relay_manifest["mask_mapping"],
        "zero_gate_reference": "v3_v2_and_relay_off_exact",
        "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
        "dc_support_mode": DC_SUPPORT_MODE,
        "dc_support_formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
        "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "tail_support_parameters": 0,
        "tail_support_buffers": 0,
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
        "formula_selection_aggregate_sha256": (
            FORMULA_SELECTION_AGGREGATE_SHA256
        ),
        "formula_selection_marker_sha256": (
            FORMULA_SELECTION_MARKER_SHA256
        ),
        "initialization_mode": (
            "fresh_full_v8_parent_plus_v4_tail_aware_relay"
        ),
        "fresh_training": True,
        "warm_start_applied": False,
        "v3_warm_start": False,
        "loss": "sum of BCE over six post-sigmoid outputs",
        "six_output_training_semantics": True,
        "relay_off_retrained": False,
        "total_parameters": PRODUCTION_V4_RELAY_ON_PARAMETERS,
    }
    exact_manifest = _architecture_manifest(variant, model, metadata)
    metadata.update(
        {
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "cross_version_exact_resume_supported": False,
            "architecture_manifest": exact_manifest,
            "architecture_id": canonical_sha256(exact_manifest),
        }
    )
    return model, metadata


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    return v8_kernel.resolve_device(args)


def environment_contract(device: torch.device) -> dict[str, Any]:
    payload = shared_exact.environment_contract(device)
    if device.type != "cuda":
        payload.update(
            {
                "physical_gpu_index": None,
                "physical_gpu_uuid": None,
                "physical_gpu_assignment_source": None,
            }
        )
        return payload
    physical_index = os.environ.get(
        "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_INDEX"
    )
    physical_uuid = os.environ.get(
        "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_UUID"
    )
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "V4 physical GPU index must identify registered GPU 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            f"V4 physical GPU UUID differs for GPU {physical_index}"
        )
    if payload.get("device_uuid") != expected_uuid:
        raise RuntimeError("visible cuda:0 UUID differs from V4 assignment")
    if payload.get("cuda_visible_devices") != expected_uuid:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must use the assigned GPU UUID"
        )
    payload.update(
        {
            "physical_gpu_index": int(physical_index),
            "physical_gpu_uuid": expected_uuid,
            "physical_gpu_assignment_source": (
                "verified_v4_tail_aware_worker_environment"
            ),
        }
    )
    return payload


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
) -> dict[str, str]:
    path = Path(exact_source_lock_path).resolve()
    payload = load_json_mapping(path, "V4 exact source lock")
    _require_equal(
        "V4 exact source-lock schema",
        payload.get("schema"),
        EXACT_SOURCE_LOCK_SCHEMA,
    )
    _require_equal(
        "V4 exact source-lock variants",
        tuple(payload.get("variants", ())),
        supported_candidate_variants(),
    )
    _require_equal(
        "V4 exact source-lock formal contract",
        payload.get("formal_contract"),
        formal_contract(),
    )
    _require_equal(
        "V4 exact source-lock training data",
        payload.get("training_data_sha256"),
        training_data_sha256,
    )
    locked = payload.get("source_sha256")
    if not isinstance(locked, Mapping):
        raise ValueError("V4 exact source lock has no source mapping")
    required = {
        str(runtime.relative_to(REPO_ROOT))
        for runtime in RUNTIME_SOURCE_PATHS
    }
    missing = sorted(required - set(locked))
    if missing:
        raise ValueError(
            f"V4 exact source lock omits runtime sources: {missing}"
        )
    for relative, expected in locked.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("V4 source lock has an invalid path")
        runtime = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(runtime.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError("V4 source path escapes repository") from exc
        _require_equal("V4 source canonical path", canonical, relative)
        _require_equal(
            f"V4 source digest for {relative}",
            file_sha256(runtime),
            expected,
        )
    return {
        SOURCE_LOCK_KEY: file_sha256(path),
        "training_data": training_data_sha256,
    }


def _require_v4_metadata(
    metadata: Any,
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("V4 model metadata is missing")
    value = copy.deepcopy(dict(metadata))
    expected = {
        "variant": variant,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V4_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "zero_gate_reference": "v3_v2_and_relay_off_exact",
        "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
        "dc_support_mode": DC_SUPPORT_MODE,
        "dc_support_formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
        "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
        "formula_selection_aggregate_sha256": (
            FORMULA_SELECTION_AGGREGATE_SHA256
        ),
        "formula_selection_marker_sha256": (
            FORMULA_SELECTION_MARKER_SHA256
        ),
        "fresh_training": True,
        "warm_start_applied": False,
        "v3_warm_start": False,
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "required_control": REQUIRED_CONTROL,
        "relay_off_retrained": False,
    }
    for name, required in expected.items():
        _require_equal(f"V4 metadata {name}", value.get(name), required)
    _normalized_thresholds(value.get("tail_z_thresholds"))
    manifest = _require_v4_manifest(
        value.get("architecture_manifest"),
        variant=variant,
    )
    _require_equal(
        "V4 metadata architecture digest",
        value.get("architecture_id"),
        canonical_sha256(manifest),
    )
    return value


def make_exact_run_spec(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_metadata: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    initialization_contract: Mapping[str, Any],
    initial_model_state_sha256: str,
    initial_rng: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    source_locks: Mapping[str, str],
    split_records: Mapping[str, exact_runner.OrderedFingerprint],
    data_records: Mapping[str, exact_runner.OrderedFingerprint],
    environment: Mapping[str, Any],
) -> exact_runner.ExactRunSpec:
    _validate_formal_args(args)
    metadata = _require_v4_metadata(
        model_metadata,
        variant=args.variant,
    )
    _require_equal(
        "V4 run source-lock keys",
        set(source_locks),
        {SOURCE_LOCK_KEY, "training_data"},
    )
    adapted = v8_kernel._clone_kernel_function(
        v8_kernel.exact_kernel.make_exact_run_spec,
        {
            "_validate_formal_args": _validate_formal_args,
            "ENTRY_SCHEMA": ENTRY_SCHEMA,
            "formal_contract": formal_contract,
            "FORMAL_EPOCHS": FORMAL_EPOCHS,
            "FORMAL_EVAL_EVERY": FORMAL_EVAL_EVERY,
            "FORMAL_WORKERS": FORMAL_WORKERS,
            "FORMAL_AMP": FORMAL_AMP,
            "FORMAL_EPS": FORMAL_EPS,
            "FORMAL_CUBLAS_WORKSPACE_CONFIG": (
                FORMAL_CUBLAS_WORKSPACE_CONFIG
            ),
        },
    )
    spec = adapted(
        args,
        model=model,
        model_metadata=metadata,
        optimizer=optimizer,
        scaler=scaler,
        initialization_contract=initialization_contract,
        initial_model_state_sha256=initial_model_state_sha256,
        initial_rng=initial_rng,
        selection_policy=selection_policy,
        source_locks=source_locks,
        split_records=split_records,
        data_records=data_records,
        environment=environment,
    )
    determinism = dict(spec.determinism)
    determinism.update(_required_determinism())
    return replace(
        spec,
        run_id=(
            f"{RUN_ID_PREFIX}{args.dataset}:{args.variant}:"
            f"seed-{args.seed}:split-{args.split_seed}:{args.run_tag}"
        ),
        determinism=determinism,
    )


def _require_complete_validation_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        name for name in STORED_VALIDATION_METRICS if name not in metrics
    ]
    if missing:
        raise ValueError(f"V4 validation metrics lack fields: {missing}")
    return {
        name: copy.deepcopy(metrics[name])
        for name in STORED_VALIDATION_METRICS
    }


def require_evaluator_checkpoint_payload(
    payload: Any,
    *,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("V4 evaluator checkpoint is not a mapping")
    value = copy.deepcopy(dict(payload))
    missing = [
        field
        for field in EVALUATOR_CHECKPOINT_REQUIRED_FIELDS
        if field not in value
    ]
    if missing:
        raise ValueError(f"V4 evaluator checkpoint lacks fields: {missing}")
    _require_equal("V4 checkpoint schema", value["schema"], CHECKPOINT_SCHEMA)
    identity = require_v4_run_identity(
        value["run_identity"],
        label="evaluator checkpoint",
        expected_variant=expected_variant,
    )
    expected_top_level = {
        "variant": identity["variant"],
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V4_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": (
            "atan(pi*(centered+dc*selected_support))/pi"
        ),
        "zero_gate_reference": "v3_v2_and_relay_off_exact",
        "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
        "dc_support_mode": DC_SUPPORT_MODE,
        "dc_support_formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
        "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
        "formula_selection_aggregate_sha256": (
            FORMULA_SELECTION_AGGREGATE_SHA256
        ),
        "formula_selection_marker_sha256": (
            FORMULA_SELECTION_MARKER_SHA256
        ),
        "fresh_training": True,
        "warm_start_applied": False,
        "dataset": identity["dataset"],
        "seed": identity["seed"],
        "split_seed": identity["split_seed"],
        "required_control": REQUIRED_CONTROL,
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "relay_off_retrained": False,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }
    for name, expected in expected_top_level.items():
        _require_equal(f"V4 checkpoint {name}", value.get(name), expected)
    _normalized_thresholds(value.get("tail_z_thresholds"))
    epoch = value["epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or epoch > FORMAL_EPOCHS
    ):
        raise ValueError("V4 evaluator checkpoint epoch is invalid")
    if value["checkpoint_role"] not in {
        "last_evaluated_epoch",
        "best_validation_pd_primary",
        "best_validation_miou_secondary",
    }:
        raise ValueError("V4 evaluator checkpoint role is invalid")
    for name in ("state_dict", "optimizer", "scaler"):
        if not isinstance(value[name], Mapping) or (
            name == "state_dict" and not value[name]
        ):
            raise ValueError(f"V4 evaluator checkpoint {name} is invalid")
    _require_equal(
        "V4 checkpoint scheduler",
        value["scheduler"],
        {
            "kind": "identity_bound_manual_schedule",
            "completed_epoch": epoch,
        },
    )
    metrics = _require_complete_validation_metrics(
        value["validation_metrics"]
    )
    for name, metric in metrics.items():
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float, np.number))
            or not math.isfinite(float(metric))
        ):
            raise ValueError(f"V4 evaluator metric {name} is invalid")
    metadata = _require_v4_metadata(
        value["model_metadata"],
        variant=identity["variant"],
    )
    manifest = metadata["architecture_manifest"]
    manifest_sha256 = canonical_sha256(manifest)
    _require_equal(
        "V4 identity architecture digest",
        identity.get("builder_manifest_sha256"),
        manifest_sha256,
    )
    split_hashes = value["split_hashes"]
    if not isinstance(split_hashes, Mapping) or not split_hashes:
        raise ValueError("V4 evaluator split hashes are invalid")
    for name, digest in split_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("V4 evaluator split hash is invalid")
        int(digest, 16)
    expected_checkpoint_identity = {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "variant": identity["variant"],
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V4_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
        "dc_support_mode": DC_SUPPORT_MODE,
        "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
        "formula_selection_aggregate_sha256": (
            FORMULA_SELECTION_AGGREGATE_SHA256
        ),
        "formula_selection_marker_sha256": (
            FORMULA_SELECTION_MARKER_SHA256
        ),
        "run_id": identity["run_id"],
        "architecture_id": identity["architecture_id"],
        "builder_manifest_sha256": identity[
            "builder_manifest_sha256"
        ],
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "required_control": REQUIRED_CONTROL,
    }
    _require_equal(
        "V4 evaluator checkpoint identity",
        value["checkpoint_identity"],
        expected_checkpoint_identity,
    )
    return value


@dataclass(frozen=True)
class EvaluatorCheckpointAdapter:
    model_metadata: Mapping[str, Any]
    split_hashes: Mapping[str, str]

    def __call__(
        self,
        context: exact_runner.CompatibilityPayloadContext,
    ) -> Mapping[str, Any]:
        identity = require_v4_run_identity(
            context.run_identity,
            label="checkpoint context",
        )
        metadata = _require_v4_metadata(
            self.model_metadata,
            variant=identity["variant"],
        )
        exact_payload = context.exact_payload
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": identity["variant"],
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "relay_enabled": True,
            "relay_version": V4_RELAY_VERSION,
            "relay_width": RELAY_WIDTH,
            "gate_dc_offset": "learned_per_stage_post_centering",
            "gate_dc_offset_count": 3,
            "gate_dc_offset_initialization": "zero",
            "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
            "mask_mapping": (
                "atan(pi*(centered+dc*selected_support))/pi"
            ),
            "zero_gate_reference": "v3_v2_and_relay_off_exact",
            "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
            "dc_support_mode": DC_SUPPORT_MODE,
            "dc_support_formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
            "dc_support_formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
            "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
            "tail_z_thresholds_frozen": True,
            "target_protective_complement": True,
            "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
            "formula_selection_aggregate_sha256": (
                FORMULA_SELECTION_AGGREGATE_SHA256
            ),
            "formula_selection_marker_sha256": (
                FORMULA_SELECTION_MARKER_SHA256
            ),
            "fresh_training": True,
            "warm_start_applied": False,
            "dataset": identity["dataset"],
            "seed": identity["seed"],
            "split_seed": identity["split_seed"],
            "state_dict": copy.deepcopy(
                exact_payload["model"]["state_dict"]
            ),
            "optimizer": copy.deepcopy(
                exact_payload["optimizer"]["state_dict"]
            ),
            "scaler": copy.deepcopy(
                exact_payload["scaler"]["state_dict"]
            ),
            "scheduler": {
                "kind": "identity_bound_manual_schedule",
                "completed_epoch": context.epoch,
            },
            "validation_metrics": _require_complete_validation_metrics(
                context.metrics
            ),
            "model_metadata": metadata,
            "split_hashes": copy.deepcopy(dict(self.split_hashes)),
            "run_identity": identity,
            "checkpoint_identity": {
                "schema": CHECKPOINT_IDENTITY_SCHEMA,
                "variant": identity["variant"],
                "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
                "relay_enabled": True,
                "relay_version": V4_RELAY_VERSION,
                "relay_width": RELAY_WIDTH,
                "gate_dc_offset": "learned_per_stage_post_centering",
                "gate_dc_offset_count": 3,
                "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
                "dc_support_mode": DC_SUPPORT_MODE,
                "dc_support_formula_stage3_2": (
                    DC_SUPPORT_FORMULA_STAGE3_2
                ),
                "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
                "formula_selection_decision": (
                    "COMPLEMENT_TAIL_SELECTED"
                ),
                "formula_selection_aggregate_sha256": (
                    FORMULA_SELECTION_AGGREGATE_SHA256
                ),
                "formula_selection_marker_sha256": (
                    FORMULA_SELECTION_MARKER_SHA256
                ),
                "run_id": identity["run_id"],
                "architecture_id": identity["architecture_id"],
                "builder_manifest_sha256": identity[
                    "builder_manifest_sha256"
                ],
                "structural_predecessor": STRUCTURAL_PREDECESSOR,
                "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
                "required_control": REQUIRED_CONTROL,
            },
            "required_control": REQUIRED_CONTROL,
            "structural_predecessor": STRUCTURAL_PREDECESSOR,
            "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
            "relay_off_retrained": False,
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }
        return require_evaluator_checkpoint_payload(
            payload,
            expected_variant=identity["variant"],
        )


class TPDNERV8V4TailAwareExactRunner(
    v3_exact.v1_exact.TPDNERV8ExactRunner
):
    """Exact runner that rejects non-V4 journals before state restoration."""

    def _require_v8_active_journal(self) -> None:
        active = self.journal.load_active()
        if active is None:
            return
        payload, _ = self._load_exact_payload(active.checkpoint_path)
        require_v4_run_identity(
            payload.get("run_identity"),
            label="active exact journal",
            expected_variant=self.spec.variant,
        )
        if not isinstance(payload.get("optimizer"), Mapping):
            raise ValueError("active V4 journal has no optimizer state")


DCHExactRunner = TPDNERV8V4TailAwareExactRunner


def six_output_bce_loss(
    outputs: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    return v3_exact.six_output_bce_loss(outputs, target, criterion)


def training_arguments(args: argparse.Namespace) -> dict[str, Any]:
    _validate_formal_args(args)
    payload = v8_kernel.training_arguments(args)
    payload.update(_required_determinism())
    payload.update(
        {
            "structural_predecessor": STRUCTURAL_PREDECESSOR,
            "required_control": REQUIRED_CONTROL,
            "relay_off_retrained": False,
        }
    )
    return payload


def protocol_payload(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    normalization: Mapping[str, float],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = require_v4_run_identity(
        run_identity,
        label="protocol",
        expected_variant=args.variant,
    )
    adapted = v8_kernel._clone_kernel_function(
        v8_kernel.exact_kernel.protocol_payload,
        {
            "ENTRY_SCHEMA": ENTRY_SCHEMA,
            "formal_contract": formal_contract,
            "training_arguments": training_arguments,
            "SELECTION_METRICS": SELECTION_METRICS,
            "STORED_VALIDATION_METRICS": STORED_VALIDATION_METRICS,
        },
    )
    payload = adapted(
        args,
        directory=directory,
        model_metadata=model_metadata,
        normalization=normalization,
        run_identity=identity,
    )
    payload["comparison_design"] = {
        "primary": [
            "baseline_sctransnet",
            REQUIRED_CONTROL,
            PAIRED_GATE_PREDECESSOR,
            V3_RELAY_ON_VARIANT,
            TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON,
        ],
        "required_control": REQUIRED_CONTROL,
        "required_control_role": "v1_relay_off_paired_control",
        "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
        "paired_gate_predecessor_role": (
            "v2_relay_on_formal_six_component_role_gate"
        ),
        "structural_predecessor": STRUCTURAL_PREDECESSOR,
        "structural_predecessor_role": (
            "v3_global_dc_support_additional_delta"
        ),
        "postprocess_requirements": {
            "v1_role_gate": True,
            "v2_role_gate": True,
            "v3_additional_delta_report": True,
        },
        "relay_off_retrained": False,
    }
    payload["tail_aware_identity"] = {
        "scope": DC_SUPPORT_SCOPE,
        "mode": DC_SUPPORT_MODE,
        "formula_stage4": DC_SUPPORT_FORMULA_STAGE4,
        "formula_stage3_2": DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
        "thresholds_frozen": True,
        "target_protective_complement": True,
        "parameters_added_over_v3": 0,
        "buffers_added_over_v3": 0,
    }
    payload["formula_selection"] = formula_selection_contract()
    payload["checkpoint_selection"] = {
        "best": "best_validation_pd_primary",
        "best_miou": "best_validation_miou_secondary",
        "source": "internal_validation_only",
        "same_as_v3_exact": True,
    }
    payload["exact_resume_policy"] = {
        "fresh": "fresh_v8_parent_plus_v4_only",
        "v3_warm_start": "forbidden",
        "same_version": "same_v4_epoch_boundary_only",
        "cross_version": "forbidden",
        "wrong_formula": "forbidden",
        "wrong_thresholds": "forbidden",
        "scheduler_restore": (
            "manual_schedule_reconstructed_from_identity_and_epoch"
        ),
    }
    return payload


def _load_complete_events(
    path: Path,
    epochs: int,
) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(events) != epochs or [
        event.get("epoch") for event in events
    ] != list(range(1, epochs + 1)):
        raise RuntimeError("V4 exact metrics are not a contiguous history")
    for event in events:
        _require_complete_validation_metrics(event)
    return events


def completion_summary(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    adapted = v8_kernel._clone_kernel_function(
        v8_kernel.exact_kernel.completion_summary,
        {
            "FORMAL_EPOCHS": FORMAL_EPOCHS,
            "formal_contract": formal_contract,
            "SELECTION_METRICS": SELECTION_METRICS,
            "STORED_VALIDATION_METRICS": STORED_VALIDATION_METRICS,
            "_load_complete_events": _load_complete_events,
            "_require_complete_validation_metrics": (
                _require_complete_validation_metrics
            ),
        },
    )
    payload = adapted(
        args,
        directory=directory,
        model_metadata=model_metadata,
        split_hashes=split_hashes,
        selection=selection,
    )
    payload.update(
        {
            "schema": COMPLETION_SUMMARY_SCHEMA,
            "candidate_family": CANDIDATE_FAMILY,
            "split_seed": args.split_seed,
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "relay_enabled": True,
            "relay_version": V4_RELAY_VERSION,
            "relay_width": RELAY_WIDTH,
            "gate_dc_offset": "learned_per_stage_post_centering",
            "gate_dc_offset_count": 3,
            "ner_dc_offset_support_scope": DC_SUPPORT_SCOPE,
            "dc_support_mode": DC_SUPPORT_MODE,
            "dc_support_formula_stage3_2": (
                DC_SUPPORT_FORMULA_STAGE3_2
            ),
            "tail_z_thresholds": dict(TAIL_Z_THRESHOLDS),
            "tail_z_thresholds_frozen": True,
            "target_protective_complement": True,
            "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
            "formula_selection_aggregate_sha256": (
                FORMULA_SELECTION_AGGREGATE_SHA256
            ),
            "formula_selection_marker_sha256": (
                FORMULA_SELECTION_MARKER_SHA256
            ),
            "fresh_training": True,
            "warm_start_applied": False,
            "structural_predecessor": STRUCTURAL_PREDECESSOR,
            "paired_gate_predecessor": PAIRED_GATE_PREDECESSOR,
            "required_control": REQUIRED_CONTROL,
            "relay_off_retrained": False,
        }
    )
    protocol = load_json_mapping(
        directory / "protocol.json",
        "completed V4 exact protocol",
    )
    _require_equal(
        "completed V4 protocol schema",
        protocol.get("schema"),
        ENTRY_SCHEMA,
    )
    payload["run_identity"] = require_v4_run_identity(
        protocol.get("run_identity"),
        label="completion protocol",
        expected_variant=args.variant,
    )
    return payload


def _check_metrics(metrics: Mapping[str, Any]) -> None:
    _require_complete_validation_metrics(metrics)
    for name, value in metrics.items():
        if (
            isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            raise FloatingPointError(
                f"non-finite V4 validation metric {name!r}: {value!r}"
            )


def _reused_training_kernel() -> Any:
    base_loop = v3_exact.v1_exact._reused_training_kernel()
    return v8_kernel._clone_kernel_function(
        base_loop,
        {
            "_validate_formal_args": _validate_formal_args,
            "resolve_device": resolve_device,
            "run_directory": run_directory,
            "source_lock_contract": source_lock_contract,
            "build_selected_model": build_selected_model,
            "initialization_plan": initialization_plan,
            "make_exact_run_spec": make_exact_run_spec,
            "environment_contract": environment_contract,
            "EvaluatorCheckpointAdapter": EvaluatorCheckpointAdapter,
            "DCHExactRunner": TPDNERV8V4TailAwareExactRunner,
            "six_output_bce_loss": six_output_bce_loss,
            "_check_metrics": _check_metrics,
            "protocol_payload": protocol_payload,
            "completion_summary": completion_summary,
            "FORMAL_EPOCHS": FORMAL_EPOCHS,
            "FORMAL_EVAL_EVERY": FORMAL_EVAL_EVERY,
            "FORMAL_WORKERS": FORMAL_WORKERS,
            "FORMAL_AMP": FORMAL_AMP,
            "FORMAL_EPS": FORMAL_EPS,
            "STORED_VALIDATION_METRICS": STORED_VALIDATION_METRICS,
        },
    )


def run_training(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    return _reused_training_kernel()(args)


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


__all__ = [
    "ARCHITECTURE_MANIFEST_SCHEMA",
    "CHECKPOINT_IDENTITY_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "COMPLETION_SUMMARY_SCHEMA",
    "DC_SUPPORT_FORMULA_STAGE3_2",
    "DC_SUPPORT_FORMULA_STAGE4",
    "DC_SUPPORT_MODE",
    "DC_SUPPORT_SCOPE",
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DCHExactRunner",
    "ENTRY_SCHEMA",
    "EVALUATOR_CHECKPOINT_REQUIRED_FIELDS",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EvaluatorCheckpointAdapter",
    "FALLBACK_CANDIDATE_VARIANTS",
    "FORMAL_AMP",
    "FORMAL_BATCH_SIZE",
    "FORMAL_CUBLAS_WORKSPACE_CONFIG",
    "FORMAL_EPOCHS",
    "FORMAL_EPS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_RUN_TAG",
    "FORMAL_WORKERS",
    "FORMULA_SELECTION_AGGREGATE_PATH",
    "FORMULA_SELECTION_AGGREGATE_SHA256",
    "FORMULA_SELECTION_MARKER_PATH",
    "FORMULA_SELECTION_MARKER_SHA256",
    "PAIRED_GATE_PREDECESSOR",
    "PHYSICAL_GPU_UUIDS",
    "PROTOCOL_DRAFT_PATH",
    "RELAY_INITIALIZATION_SEED",
    "RELAY_WIDTH",
    "RUNTIME_SOURCE_PATHS",
    "RUN_ID_PREFIX",
    "SELECTION_METRICS",
    "SOURCE_LOCK_KEY",
    "SPLIT_SEED",
    "STORED_VALIDATION_METRICS",
    "STRUCTURAL_PREDECESSOR",
    "TAIL_Z_THRESHOLDS",
    "TPDNERV8V4TailAwareExactRunner",
    "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON",
    "TRAINING_SEED",
    "V3_RELAY_ON_VARIANT",
    "V4_RELAY_VERSION",
    "build_selected_model",
    "candidate_contract",
    "completion_summary",
    "environment_contract",
    "formal_contract",
    "formula_selection_contract",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "parse_args",
    "protocol_payload",
    "require_evaluator_checkpoint_payload",
    "require_v4_run_identity",
    "run_directory",
    "run_training",
    "six_output_bce_loss",
    "source_lock_contract",
    "supported_candidate_variants",
]


if __name__ == "__main__":
    main()
