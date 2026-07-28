#!/usr/bin/env python3
"""Exact epoch-boundary trainer for the sole V3 five-node NER candidate.

The numerical loop is the verified V8 exact loop, cloned with V3-owned
builders, identities, source locks, checkpoint schemas, and pre-restore
guards.  Exact resume is accepted only for the same V3 relay-on trajectory.
V1/V2 checkpoints and every other variant are rejected before any model,
optimizer, scaler, RNG, or DataLoader-generator state is restored.

V3 deliberately schedules no relay-off retraining.  Its required control is
the existing V8-parent relay-off contract, whose public variant identity is
``tpd_ner_v8_mprs_dch_full_relay_off``.
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
from experiments import train_tpd_ner_v8_mprs_dch_exact as v1_exact  # noqa: E402
from experiments.train_tpd_clean_v8_mprs_dch import (  # noqa: E402
    build_clean_v8_mprs_dch_model,
)
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
)
from model.tpd_ner_v8_mprs_dch_v3 import (  # noqa: E402
    PRODUCTION_V3_RELAY_ON_PARAMETERS,
    PRODUCTION_V3_RELAY_PARAMETERS,
    RELAY_RMS_EPS,
    TPDNERV8MPRSDCHV3SCTransNet,
    V3_RELAY_VERSION,
    adapt_v8_mprs_dch_parent_v3,
    v3_relay_parameter_count,
)


ENTRY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v3_exact_entry_v1"
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_exact_source_lock_v1"
)
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_exact_architecture_manifest_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_exact_checkpoint_identity_v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_exact_checkpoint_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_completion_summary_v1"
)
SOURCE_LOCK_KEY = "tpd_ner_v8_mprs_dch_v3_exact_source_lock"
RUN_ID_PREFIX = "tpd-ner-v8-mprs-dch-v3-exact:"
FORMAL_RUN_TAG = "formal800_exact_v3_seed42"
CANDIDATE_FAMILY = (
    "tpd_clean_v8_mprs_dch_explicit_five_node_ner_"
    "v3_post_center_dc"
)
CONSTRUCTION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_"
    "five_node_post_center_dc_v1"
)

TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON = (
    "tpd_ner_v8_mprs_dch_v3_full_relay_on"
)
V8_PARENT_RELAY_OFF_REFERENCE = "tpd_ner_v8_mprs_dch_full_relay_off"
INHERITED_V8_RELAY_ON_PARSER_VARIANT = (
    "tpd_ner_v8_mprs_dch_full_relay_on"
)
V2_RELAY_ON_VARIANT = "tpd_ner_v8_mprs_dch_v2_full_relay_on"
V2_EXACT_ENTRY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_exact_entry_v1"
)
V2_EXACT_SOURCE_LOCK_KEY = "tpd_ner_v8_mprs_dch_v2_exact_source_lock"
FALLBACK_CANDIDATE_VARIANTS = (
    TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
)

EVALUATOR_CHECKPOINT_REQUIRED_FIELDS = (
    "schema",
    "epoch",
    "checkpoint_role",
    "variant",
    "parent_variant",
    "relay_enabled",
    "relay_version",
    "relay_width",
    "gate_dc_offset",
    "gate_dc_offset_count",
    "gate_dc_offset_initialization",
    "gate_dc_offset_state_prefix",
    "mask_mapping",
    "zero_gate_reference",
    "dataset",
    "seed",
    "split_seed",
    "state_dict",
    "optimizer",
    "scaler",
    "scheduler",
    "validation_metrics",
    "model_metadata",
    "split_hashes",
    "run_identity",
    "checkpoint_identity",
    "required_control",
    "structural_predecessor",
    "relay_off_retrained",
    "selection_source",
    "official_test_accessed",
)

DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT / "experiments/tpd_ner_v8_mprs_dch_v3_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_ner_v8_mprs_dch_v3_exact_v1"
)

TRAINING_SEED = 42
SPLIT_SEED = 20260722
RELAY_WIDTH = 8
RELAY_INITIALIZATION_SEED = 42
FORMAL_EPOCHS = 800
FORMAL_BATCH_SIZE = 16
FORMAL_PATCH_SIZE = 256
FORMAL_WORKERS = 0
FORMAL_VAL_FRACTION = 0.20
FORMAL_EVAL_EVERY = 1
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_THRESHOLD = 0.5
FORMAL_MATCH_RADIUS = 3.0
FORMAL_TINY_AREA = 9
FORMAL_AMP = False
FORMAL_EPS = v1_exact.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = v1_exact.FORMAL_CUBLAS_WORKSPACE_CONFIG
FORMAL_INITIALIZATION_MODES = v1_exact.FORMAL_INITIALIZATION_MODES
PHYSICAL_GPU_UUIDS = dict(v1_exact.PHYSICAL_GPU_UUIDS)
SELECTION_METRICS = v1_exact.SELECTION_METRICS
STORED_VALIDATION_METRICS = v1_exact.STORED_VALIDATION_METRICS


def _ordered_unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


# No lock file is created here.  The future lock must cover the V3-owned
# sources plus every inherited source whose code executes in the cloned loop.
RUNTIME_SOURCE_PATHS = _ordered_unique_paths(
    (
        REPO_ROOT / "experiments/train_tpd_ner_v8_mprs_dch_v3_exact.py",
        REPO_ROOT / "experiments/TPD_NER_V8_MPRS_DCH_V3_PROTOCOL.md",
        REPO_ROOT / "model/tpd_ner_v8_mprs_dch_v3.py",
        REPO_ROOT / "model/tpd_ner_v8_mprs_dch_v2.py",
        *v1_exact.RUNTIME_SOURCE_PATHS,
    )
)

v8_kernel = v1_exact.v8_kernel
file_sha256 = v1_exact.file_sha256
canonical_sha256 = v1_exact.canonical_sha256
load_json_mapping = v1_exact.load_json_mapping
PreparedData = v1_exact.PreparedData
prepare_data = v1_exact.prepare_data
split_fingerprints = v1_exact.split_fingerprints
data_fingerprints = v1_exact.data_fingerprints
configure_determinism = v1_exact.configure_determinism
write_or_verify_json = v1_exact.write_or_verify_json
base = v1_exact.base
shared_exact = v1_exact.shared_exact
InitializationPlan = v1_exact.InitializationPlan


def supported_candidate_variants() -> tuple[str, ...]:
    """Return the exact entry's closed, single-candidate V3 matrix."""

    return FALLBACK_CANDIDATE_VARIANTS


def candidate_contract(candidate_variant: str) -> dict[str, Any]:
    if not isinstance(candidate_variant, str):
        raise TypeError("V3 candidate variant must be a string")
    if candidate_variant != TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON:
        raise ValueError(
            f"unsupported V3 exact variant {candidate_variant!r}; "
            f"choices={FALLBACK_CANDIDATE_VARIANTS}"
        )
    return {
        "candidate_variant": candidate_variant,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V3_RELAY_VERSION,
        "structural_predecessor": V2_RELAY_ON_VARIANT,
        "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
    }


def formal_contract() -> dict[str, Any]:
    return {
        "dataset": "NUDT-SIRST",
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
        "precision": "FP32",
        "eps": FORMAL_EPS,
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        "initialization_modes": list(FORMAL_INITIALIZATION_MODES),
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "candidate_variants": list(supported_candidate_variants()),
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V3_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "relay_rms_eps": RELAY_RMS_EPS,
        "gate_bias": False,
        "gate_spatial_centering": "per_sample_mean_hw",
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": "atan(pi*(centered+dc))/pi",
        "zero_gate_reference": "v2_and_relay_off_exact",
        "structural_predecessor": V2_RELAY_ON_VARIANT,
        "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
        "multi_seed_scheduled": False,
        "logical_device": "cuda:0",
        "physical_gpu_choices": [2, 3],
        "physical_gpu_model": "NVIDIA GeForce RTX 5090",
        "physical_gpu_index_environment": (
            "TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_INDEX"
        ),
        "physical_gpu_uuid_environment": (
            "TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_UUID"
        ),
        "cuda_visible_devices_semantics": "single_registered_gpu_uuid",
        "scheduler_restore": (
            "identity_bound_manual_schedule_from_completed_epoch"
        ),
    }


def _validate_formal_args(args: argparse.Namespace) -> None:
    candidate_contract(getattr(args, "variant", None))
    expected = {
        "dataset": "NUDT-SIRST",
        "variant": TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
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
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            f"formal V3 exact arguments differ: "
            f"expected={expected}, observed={observed}"
        )
    device = getattr(args, "device", None)
    allow_cpu_smoke = getattr(args, "allow_cpu_smoke", False)
    if allow_cpu_smoke:
        if device != "cpu":
            raise ValueError("CPU smoke permission requires --device=cpu")
    elif device != "cuda:0":
        raise ValueError("formal V3 exact training requires --device=cuda:0")


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
    raise ValueError(f"V3 exact entry requires {option}")


def _emit_v3_help() -> None:
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            v1_exact.parse_args(
                [
                    "--variant",
                    INHERITED_V8_RELAY_ON_PARSER_VARIANT,
                    "--help",
                ]
            )
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    help_text = output.getvalue()
    help_text = help_text.replace(
        "train_tpd_ner_v8_mprs_dch_exact.py",
        Path(__file__).name,
    )
    help_text = help_text.replace(
        "Exact-resume V8-MPRS-DCH five-node NER training",
        (
            "Exact-resume V3 post-centering DC-calibrated "
            "five-node NER training"
        ),
    )
    v1_choices = (
        "{tpd_ner_v8_mprs_dch_full_relay_off,"
        "tpd_ner_v8_mprs_dch_full_relay_on}"
    )
    help_text = help_text.replace(
        v1_choices,
        "{" + TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON + "}",
    )
    sys.stdout.write(help_text)
    raise SystemExit(0)


def _mapped_inherited_arguments(
    arguments: Sequence[str],
) -> list[str]:
    forbidden = (
        "--relay-enabled",
        "--relay-off",
        "--no-relay-enabled",
        "--relay-width",
        "--relay-initialization-seed",
    )
    for argument in arguments:
        if any(
            argument == option or argument.startswith(f"{option}=")
            for option in forbidden
        ):
            raise ValueError("V3 relay identity is fixed by the sole variant")
    mapped = list(arguments)
    selected: str | None = None
    for index, argument in enumerate(mapped):
        if argument == "--variant":
            if index + 1 < len(mapped):
                selected = mapped[index + 1]
            break
        if argument.startswith("--variant="):
            selected = argument.split("=", 1)[1]
            break
    if selected != TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON:
        raise ValueError(
            "V3 exact entry accepts only "
            f"--variant={TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON}"
        )
    _replace_option_value(
        mapped,
        "--variant",
        INHERITED_V8_RELAY_ON_PARSER_VARIANT,
    )
    if not _option_present(mapped, "--run-tag"):
        mapped.extend(("--run-tag", FORMAL_RUN_TAG))
    return mapped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value in ("-h", "--help") for value in arguments):
        _emit_v3_help()
    mapped = _mapped_inherited_arguments(arguments)
    args = v1_exact.parse_args(mapped)
    args.variant = TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
    args.parent_variant = PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT
    args.relay_enabled = True
    args.relay_width = RELAY_WIDTH
    args.relay_initialization_seed = RELAY_INITIALIZATION_SEED
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


def require_v3_run_identity(
    identity: Any,
    *,
    label: str,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    """Require a same-version, same-variant V3 exact trajectory identity."""

    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no V3 exact run identity")
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
    if entry_schema in {
        v1_exact.ENTRY_SCHEMA,
        v8_kernel.ENTRY_SCHEMA,
        v8_kernel.exact_kernel.ENTRY_SCHEMA,
        V2_EXACT_ENTRY_SCHEMA,
    }:
        raise ValueError(f"{label} is a V1/V2/V8 trajectory, not V3")
    if variant in {
        V8_PARENT_RELAY_OFF_REFERENCE,
        INHERITED_V8_RELAY_ON_PARSER_VARIANT,
        V2_RELAY_ON_VARIANT,
    }:
        raise ValueError(f"{label} uses a non-V3 relay variant")
    if entry_schema != ENTRY_SCHEMA:
        raise ValueError(f"{label} entry schema is not V3")
    if variant != TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON:
        raise ValueError(f"{label} variant is not the sole V3 candidate")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"{label} variant {variant!r} differs from {expected_variant!r}"
        )
    if not isinstance(run_id, str) or not run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"{label} run_id is not V3")
    forbidden_lock_keys = {
        v1_exact.SOURCE_LOCK_KEY,
        v8_kernel.SOURCE_LOCK_KEY,
        "tpd_clean_v7_dch_exact_source_lock",
        V2_EXACT_SOURCE_LOCK_KEY,
    }
    if (
        not isinstance(source_locks, Mapping)
        or SOURCE_LOCK_KEY not in source_locks
        or forbidden_lock_keys.intersection(source_locks)
    ):
        raise ValueError(f"{label} source-lock identity is not V3")
    required_determinism = {
        "entry_schema": ENTRY_SCHEMA,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V3_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "relay_rms_eps": RELAY_RMS_EPS,
        "gate_bias": False,
        "gate_spatial_centering": "per_sample_mean_hw",
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": "atan(pi*(centered+dc))/pi",
        "zero_gate_reference": "v2_and_relay_off_exact",
        "scheduler_restore_mode": (
            "identity_bound_manual_schedule_from_completed_epoch"
        ),
    }
    if not isinstance(determinism, Mapping):
        raise ValueError(f"{label} determinism identity is missing")
    for name, expected in required_determinism.items():
        if determinism.get(name) != expected:
            raise ValueError(f"{label} V3 determinism field {name} differs")
    if value.get("seed") != TRAINING_SEED:
        raise ValueError(f"{label} training seed is not 42")
    if value.get("split_seed") != SPLIT_SEED:
        raise ValueError(f"{label} split seed is not 20260722")
    return value


_require_v3_run_identity = require_v3_run_identity


def _existing_training_contract(
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, Any]:
    protocol = load_json_mapping(
        directory / "protocol.json",
        "existing V3 exact protocol",
    )
    if protocol.get("schema") != ENTRY_SCHEMA:
        raise ValueError("existing protocol is not a V3 exact run")
    identity = require_v3_run_identity(
        protocol.get("run_identity"),
        label="existing protocol",
        expected_variant=args.variant,
    )
    training = identity.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError("existing V3 protocol has no training contract")
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(f"existing V3 training contract lacks fields: {missing}")
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
            raise ValueError("existing V3 initial_rng is not a mapping")
        if not isinstance(selection_policy, Mapping):
            raise ValueError("existing V3 selection_policy is not a mapping")
        return InitializationPlan(
            request=exact_runner.InitializationRequest.exact(),
            contract=copy.deepcopy(training["initialization_contract"]),
            initial_model_state_sha256=str(
                training["initial_model_state_sha256"]
            ),
            initial_rng=copy.deepcopy(dict(initial_rng)),
            selection_policy=copy.deepcopy(dict(selection_policy)),
        )
    raise RuntimeError("V3 exact entry requires --fresh or --exact-resume")


def _architecture_manifest(
    variant: str,
    model: TPDNERV8MPRSDCHV3SCTransNet,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    relay_manifest = model.architecture_manifest()
    relay_keys = tuple(
        name for name in model.state_dict() if name.startswith("tpd_ner.")
    )
    return {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "model": (
            "model.tpd_ner_v8_mprs_dch_v3."
            "TPDNERV8MPRSDCHV3SCTransNet"
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
        "relay_parameters": v3_relay_parameter_count(model),
        "relay_rms_scope": relay_manifest["relay_rms_scope"],
        "relay_rms_eps": relay_manifest["relay_rms_eps"],
        "source_projection_rms_normalized": relay_manifest[
            "source_projection_rms_normalized"
        ],
        "fusion_relu_output_rms_normalized": relay_manifest[
            "fusion_relu_output_rms_normalized"
        ],
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
        "zero_gate_reference": relay_manifest["zero_gate_reference"],
        "structural_predecessor": V2_RELAY_ON_VARIANT,
        "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "deep_supervision_outputs": 6,
        "loss": "unweighted sum of BCE over six post-sigmoid outputs",
        "exact_resume_scope": "same_v3_relay_on_variant_only",
        "cross_version_exact_resume_supported": False,
        "eps": FORMAL_EPS,
        "formal_amp": FORMAL_AMP,
    }


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    candidate_contract(variant)
    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError("V3 exact builder requires seed=42")
    if eps != FORMAL_EPS:
        raise ValueError(f"V3 exact builder requires eps={FORMAL_EPS}")
    parent, parent_metadata = build_clean_v8_mprs_dch_model(
        PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        seed,
    )
    if not isinstance(parent_metadata, Mapping):
        raise TypeError("V8 parent builder metadata is not a mapping")
    model = adapt_v8_mprs_dch_parent_v3(
        parent,
        variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_enabled=True,
        relay_width=RELAY_WIDTH,
        relay_initialization_seed=RELAY_INITIALIZATION_SEED,
    )
    if type(model) is not TPDNERV8MPRSDCHV3SCTransNet:
        raise TypeError("V3 exact builder requires the exact V3 model class")
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
        or model.tokenizer_variant != PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT
        or model.relay_width != RELAY_WIDTH
        or model.relay_initialization_seed != RELAY_INITIALIZATION_SEED
    ):
        raise ValueError("V3 exact model identity differs")
    relay_keys = [
        name for name in model.state_dict() if name.startswith("tpd_ner.")
    ]
    if len(relay_keys) != 19:
        raise ValueError(f"V3 relay state-key count differs: {len(relay_keys)}")
    if v3_relay_parameter_count(model) != PRODUCTION_V3_RELAY_PARAMETERS:
        raise ValueError("V3 exact relay parameter count differs")
    if sum(parameter.numel() for parameter in model.parameters()) != (
        PRODUCTION_V3_RELAY_ON_PARAMETERS
    ):
        raise ValueError("V3 exact total parameter count differs")
    relay_manifest = model.architecture_manifest()
    if (
        relay_manifest.get("relay_version") != V3_RELAY_VERSION
        or relay_manifest.get("gate_dc_offset")
        != "learned_per_stage_post_centering"
        or relay_manifest.get("gate_dc_offset_count") != 3
        or relay_manifest.get("gate_dc_offset_initialization") != "zero"
        or relay_manifest.get("gate_dc_offset_state_prefix")
        != "tpd_ner.dc_offsets."
        or relay_manifest.get("zero_gate_reference")
        != "v2_and_relay_off_exact"
    ):
        raise ValueError("V3 model architecture manifest differs")
    metadata: dict[str, Any] = {
        "variant": variant,
        "candidate_family": CANDIDATE_FAMILY,
        "construction_schema": CONSTRUCTION_SCHEMA,
        "comparison_role": "tpd_plus_ner_v3_post_center_dc",
        "structural_predecessor": V2_RELAY_ON_VARIANT,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "parent_candidate_family": parent_metadata.get("candidate_family"),
        "parent_model_metadata": copy.deepcopy(dict(parent_metadata)),
        "relay_enabled": True,
        "relay_version": V3_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "relay_parameters": PRODUCTION_V3_RELAY_PARAMETERS,
        "relay_state_prefix": "tpd_ner.",
        "relay_state_key_count": len(relay_keys),
        "relay_rms_scope": relay_manifest["relay_rms_scope"],
        "relay_rms_eps": relay_manifest["relay_rms_eps"],
        "source_projection_rms_normalized": relay_manifest[
            "source_projection_rms_normalized"
        ],
        "fusion_relu_output_rms_normalized": relay_manifest[
            "fusion_relu_output_rms_normalized"
        ],
        "gate_bias": relay_manifest["gate_bias"],
        "gate_spatial_centering": relay_manifest[
            "gate_spatial_centering"
        ],
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
        "zero_gate_reference": relay_manifest["zero_gate_reference"],
        "initialization_mode": "fresh_full_v8_parent_plus_v3_relay",
        "warm_start_applied": False,
        "loss": "sum of BCE over six post-sigmoid outputs",
        "six_output_training_semantics": True,
        "total_parameters": PRODUCTION_V3_RELAY_ON_PARAMETERS,
    }
    exact_manifest = _architecture_manifest(variant, model, metadata)
    metadata.update(
        {
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
            "relay_off_retrained": False,
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
        "TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_INDEX"
    )
    physical_uuid = os.environ.get(
        "TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_UUID"
    )
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "V3 physical GPU index must identify registered GPU 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            f"V3 physical GPU UUID differs for GPU {physical_index}"
        )
    if payload.get("device_uuid") != expected_uuid:
        raise RuntimeError("visible cuda:0 UUID differs from V3 assignment")
    if payload.get("cuda_visible_devices") != expected_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must use the assigned GPU UUID")
    payload.update(
        {
            "physical_gpu_index": int(physical_index),
            "physical_gpu_uuid": expected_uuid,
            "physical_gpu_assignment_source": (
                "verified_v3_ner_worker_environment"
            ),
        }
    )
    return payload


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
) -> dict[str, str]:
    path = Path(exact_source_lock_path).resolve()
    payload = load_json_mapping(path, "V3 exact source lock")
    if payload.get("schema") != EXACT_SOURCE_LOCK_SCHEMA:
        raise ValueError("V3 exact source-lock schema mismatch")
    if tuple(payload.get("variants", ())) != supported_candidate_variants():
        raise ValueError("V3 exact source-lock variant matrix mismatch")
    if payload.get("formal_contract") != formal_contract():
        raise ValueError("V3 exact source-lock formal contract mismatch")
    if payload.get("training_data_sha256") != training_data_sha256:
        raise ValueError("training data differs from the V3 source lock")
    locked = payload.get("source_sha256")
    if not isinstance(locked, Mapping):
        raise ValueError("V3 exact source lock has no source mapping")
    required = {
        str(runtime.relative_to(REPO_ROOT))
        for runtime in RUNTIME_SOURCE_PATHS
    }
    missing = sorted(required - set(locked))
    if missing:
        raise ValueError(f"V3 exact source lock omits runtime sources: {missing}")
    for relative, expected in locked.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("V3 source lock has an invalid path")
        runtime = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(runtime.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError("V3 source path escapes repository") from exc
        if canonical != relative:
            raise ValueError("V3 source path is not canonical")
        if file_sha256(runtime) != expected:
            raise ValueError(f"V3 source lock differs for {relative}")
    return {
        SOURCE_LOCK_KEY: file_sha256(path),
        "training_data": training_data_sha256,
    }


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
    manifest = model_metadata.get("architecture_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != ARCHITECTURE_MANIFEST_SCHEMA
        or manifest.get("variant") != args.variant
        or manifest.get("relay_version") != V3_RELAY_VERSION
        or manifest.get("relay_enabled") is not True
        or manifest.get("relay_width") != RELAY_WIDTH
        or manifest.get("gate_dc_offset")
        != "learned_per_stage_post_centering"
        or manifest.get("gate_dc_offset_count") != 3
        or manifest.get("gate_dc_offset_state_prefix")
        != "tpd_ner.dc_offsets."
        or manifest.get("zero_gate_reference")
        != "v2_and_relay_off_exact"
    ):
        raise ValueError("V3 exact architecture manifest differs")
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
            "FORMAL_CUBLAS_WORKSPACE_CONFIG": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        },
    )
    spec = adapted(
        args,
        model=model,
        model_metadata=model_metadata,
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
    determinism.update(
        {
            "entry_schema": ENTRY_SCHEMA,
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "relay_enabled": True,
            "relay_version": V3_RELAY_VERSION,
            "relay_width": RELAY_WIDTH,
            "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
            "relay_rms_eps": RELAY_RMS_EPS,
            "gate_bias": False,
            "gate_spatial_centering": "per_sample_mean_hw",
            "gate_dc_offset": "learned_per_stage_post_centering",
            "gate_dc_offset_count": 3,
            "gate_dc_offset_initialization": "zero",
            "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
            "mask_mapping": "atan(pi*(centered+dc))/pi",
            "zero_gate_reference": "v2_and_relay_off_exact",
            "scheduler_restore_mode": (
                "identity_bound_manual_schedule_from_completed_epoch"
            ),
        }
    )
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
        raise ValueError(f"V3 validation metrics lack fields: {missing}")
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
        raise ValueError("V3 exact evaluator checkpoint is not a mapping")
    value = copy.deepcopy(dict(payload))
    missing = [
        field
        for field in EVALUATOR_CHECKPOINT_REQUIRED_FIELDS
        if field not in value
    ]
    if missing:
        raise ValueError(f"V3 evaluator checkpoint lacks fields: {missing}")
    if value["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("V3 evaluator checkpoint schema differs")
    identity = require_v3_run_identity(
        value["run_identity"],
        label="evaluator checkpoint",
        expected_variant=expected_variant,
    )
    expected_top_level = {
        "variant": identity["variant"],
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V3_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": "atan(pi*(centered+dc))/pi",
        "zero_gate_reference": "v2_and_relay_off_exact",
        "dataset": identity["dataset"],
        "seed": identity["seed"],
        "split_seed": identity["split_seed"],
        "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
        "structural_predecessor": V2_RELAY_ON_VARIANT,
        "relay_off_retrained": False,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }
    for name, expected in expected_top_level.items():
        if value.get(name) != expected:
            raise ValueError(f"V3 evaluator checkpoint {name} differs")
    epoch = value["epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or epoch > FORMAL_EPOCHS
    ):
        raise ValueError("V3 evaluator checkpoint epoch is invalid")
    checkpoint_role = value["checkpoint_role"]
    if (
        not isinstance(checkpoint_role, str)
        or checkpoint_role
        not in {
            "last_evaluated_epoch",
            "best_validation_pd_primary",
            "best_validation_miou_secondary",
        }
    ):
        raise ValueError("V3 evaluator checkpoint role is invalid")
    for name in ("state_dict", "optimizer", "scaler"):
        if not isinstance(value[name], Mapping) or (
            name == "state_dict" and not value[name]
        ):
            raise ValueError(f"V3 evaluator checkpoint {name} is invalid")
    scheduler = value["scheduler"]
    if (
        not isinstance(scheduler, Mapping)
        or dict(scheduler)
        != {
            "kind": "identity_bound_manual_schedule",
            "completed_epoch": epoch,
        }
    ):
        raise ValueError("V3 evaluator checkpoint scheduler differs")
    metrics = _require_complete_validation_metrics(
        value["validation_metrics"]
    )
    for name, metric in metrics.items():
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float, np.number))
            or not math.isfinite(float(metric))
        ):
            raise ValueError(f"V3 evaluator metric {name} is invalid")
    metadata = value["model_metadata"]
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("variant") != identity["variant"]
        or metadata.get("relay_enabled") is not True
        or metadata.get("relay_version") != V3_RELAY_VERSION
        or metadata.get("relay_width") != RELAY_WIDTH
        or metadata.get("relay_initialization_seed")
        != RELAY_INITIALIZATION_SEED
        or metadata.get("gate_dc_offset")
        != "learned_per_stage_post_centering"
        or metadata.get("gate_dc_offset_count") != 3
        or metadata.get("gate_dc_offset_initialization") != "zero"
        or metadata.get("gate_dc_offset_state_prefix")
        != "tpd_ner.dc_offsets."
        or metadata.get("mask_mapping")
        != "atan(pi*(centered+dc))/pi"
        or metadata.get("zero_gate_reference")
        != "v2_and_relay_off_exact"
        or metadata.get("required_control")
        != V8_PARENT_RELAY_OFF_REFERENCE
        or metadata.get("structural_predecessor")
        != V2_RELAY_ON_VARIANT
        or metadata.get("relay_off_retrained") is not False
    ):
        raise ValueError("V3 evaluator model metadata differs")
    manifest = metadata.get("architecture_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != ARCHITECTURE_MANIFEST_SCHEMA
        or manifest.get("variant") != identity["variant"]
        or manifest.get("relay_version") != V3_RELAY_VERSION
        or manifest.get("gate_dc_offset")
        != "learned_per_stage_post_centering"
        or manifest.get("gate_dc_offset_count") != 3
        or manifest.get("gate_dc_offset_state_prefix")
        != "tpd_ner.dc_offsets."
        or manifest.get("zero_gate_reference")
        != "v2_and_relay_off_exact"
    ):
        raise ValueError("V3 evaluator architecture manifest differs")
    manifest_sha256 = canonical_sha256(manifest)
    if (
        metadata.get("architecture_id") != manifest_sha256
        or identity.get("builder_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("V3 evaluator architecture digest differs")
    split_hashes = value["split_hashes"]
    if not isinstance(split_hashes, Mapping) or not split_hashes:
        raise ValueError("V3 evaluator split hashes are invalid")
    for name, digest in split_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("V3 evaluator split hash is invalid")
        int(digest, 16)
    expected_checkpoint_identity = {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "variant": identity["variant"],
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": V3_RELAY_VERSION,
        "relay_width": RELAY_WIDTH,
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "run_id": identity["run_id"],
        "architecture_id": identity["architecture_id"],
        "builder_manifest_sha256": identity["builder_manifest_sha256"],
        "structural_predecessor": V2_RELAY_ON_VARIANT,
    }
    checkpoint_identity = value["checkpoint_identity"]
    if (
        not isinstance(checkpoint_identity, Mapping)
        or dict(checkpoint_identity) != expected_checkpoint_identity
    ):
        raise ValueError("V3 evaluator checkpoint identity differs")
    return value


@dataclass(frozen=True)
class EvaluatorCheckpointAdapter:
    model_metadata: Mapping[str, Any]
    split_hashes: Mapping[str, str]

    def __call__(
        self,
        context: exact_runner.CompatibilityPayloadContext,
    ) -> Mapping[str, Any]:
        identity = require_v3_run_identity(
            context.run_identity,
            label="checkpoint context",
        )
        exact_payload = context.exact_payload
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": identity["variant"],
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "relay_enabled": True,
            "relay_version": V3_RELAY_VERSION,
            "relay_width": RELAY_WIDTH,
            "gate_dc_offset": "learned_per_stage_post_centering",
            "gate_dc_offset_count": 3,
            "gate_dc_offset_initialization": "zero",
            "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
            "mask_mapping": "atan(pi*(centered+dc))/pi",
            "zero_gate_reference": "v2_and_relay_off_exact",
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
            "model_metadata": copy.deepcopy(dict(self.model_metadata)),
            "split_hashes": copy.deepcopy(dict(self.split_hashes)),
            "run_identity": identity,
            "checkpoint_identity": {
                "schema": CHECKPOINT_IDENTITY_SCHEMA,
                "variant": identity["variant"],
                "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
                "relay_enabled": True,
                "relay_version": V3_RELAY_VERSION,
                "relay_width": RELAY_WIDTH,
                "gate_dc_offset": "learned_per_stage_post_centering",
                "gate_dc_offset_count": 3,
                "run_id": identity["run_id"],
                "architecture_id": identity["architecture_id"],
                "builder_manifest_sha256": identity[
                    "builder_manifest_sha256"
                ],
                "structural_predecessor": V2_RELAY_ON_VARIANT,
            },
            "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
            "structural_predecessor": V2_RELAY_ON_VARIANT,
            "relay_off_retrained": False,
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }
        return require_evaluator_checkpoint_payload(
            payload,
            expected_variant=identity["variant"],
        )


class TPDNERV8V3ExactRunner(v1_exact.TPDNERV8ExactRunner):
    """Exact runner that validates V3 identity before restoring any state."""

    def _require_v8_active_journal(self) -> None:
        active = self.journal.load_active()
        if active is None:
            return
        payload, _ = self._load_exact_payload(active.checkpoint_path)
        require_v3_run_identity(
            payload.get("run_identity"),
            label="active exact journal",
            expected_variant=self.spec.variant,
        )
        if not isinstance(payload.get("optimizer"), Mapping):
            raise ValueError("active V3 journal has no optimizer state")


DCHExactRunner = TPDNERV8V3ExactRunner


def six_output_bce_loss(
    outputs: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    return v1_exact.six_output_bce_loss(outputs, target, criterion)


def training_arguments(args: argparse.Namespace) -> dict[str, Any]:
    _validate_formal_args(args)
    payload = v8_kernel.training_arguments(args)
    payload.update(
        {
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "relay_enabled": True,
            "relay_version": V3_RELAY_VERSION,
            "relay_width": RELAY_WIDTH,
            "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
            "relay_rms_eps": RELAY_RMS_EPS,
            "gate_bias": False,
            "gate_spatial_centering": "per_sample_mean_hw",
            "gate_dc_offset": "learned_per_stage_post_centering",
            "gate_dc_offset_count": 3,
            "gate_dc_offset_initialization": "zero",
            "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
            "mask_mapping": "atan(pi*(centered+dc))/pi",
            "zero_gate_reference": "v2_and_relay_off_exact",
            "structural_predecessor": V2_RELAY_ON_VARIANT,
            "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
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
    identity = require_v3_run_identity(
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
            V8_PARENT_RELAY_OFF_REFERENCE,
            V2_RELAY_ON_VARIANT,
            TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        ],
        "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
        "structural_predecessor": V2_RELAY_ON_VARIANT,
        "structural_predecessor_role": "direct_v3_structural_predecessor",
        "relay_off_source": "existing_v8_parent_formal_run",
        "relay_off_retrained": False,
    }
    payload["relay_identity"] = {
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "enabled": True,
        "version": V3_RELAY_VERSION,
        "width": RELAY_WIDTH,
        "initialization_seed": RELAY_INITIALIZATION_SEED,
        "rms_eps": RELAY_RMS_EPS,
        "gate_bias": False,
        "spatial_centering": "per_sample_mean_hw",
        "dc_offset": "learned_per_stage_post_centering",
        "dc_offset_count": 3,
        "dc_offset_initialization": "zero",
        "dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": "atan(pi*(centered+dc))/pi",
        "zero_gate_reference": "v2_and_relay_off_exact",
    }
    payload["exact_resume_policy"] = {
        "same_version": "same_v3_relay_on_epoch_boundary_only",
        "v8_parent_relay_off": "reference_only_not_resumable",
        "inherited_v8_relay_on": "forbidden",
        "v2_relay_on": "forbidden",
        "cross_version": "forbidden",
        "wrong_variant": "forbidden",
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
        raise RuntimeError("V3 exact metrics are not a contiguous history")
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
            "candidate_family": "tpd_ner_v8_mprs_dch_v3",
            "split_seed": args.split_seed,
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "relay_enabled": True,
            "relay_version": V3_RELAY_VERSION,
            "relay_width": RELAY_WIDTH,
            "gate_dc_offset": "learned_per_stage_post_centering",
            "gate_dc_offset_count": 3,
            "gate_dc_offset_initialization": "zero",
            "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
            "zero_gate_reference": "v2_and_relay_off_exact",
            "structural_predecessor": V2_RELAY_ON_VARIANT,
            "required_control": V8_PARENT_RELAY_OFF_REFERENCE,
            "relay_off_retrained": False,
        }
    )
    protocol = load_json_mapping(
        directory / "protocol.json",
        "completed V3 exact protocol",
    )
    if protocol.get("schema") != ENTRY_SCHEMA:
        raise ValueError("completed protocol is not V3")
    payload["run_identity"] = require_v3_run_identity(
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
                f"non-finite V3 validation metric {name!r}: {value!r}"
            )


def _reused_training_kernel() -> Any:
    base_loop = v1_exact._reused_training_kernel()
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
            "DCHExactRunner": TPDNERV8V3ExactRunner,
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
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DCHExactRunner",
    "ENTRY_SCHEMA",
    "EVALUATOR_CHECKPOINT_REQUIRED_FIELDS",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EvaluatorCheckpointAdapter",
    "FORMAL_AMP",
    "FORMAL_BATCH_SIZE",
    "FORMAL_CUBLAS_WORKSPACE_CONFIG",
    "FORMAL_EPOCHS",
    "FORMAL_EPS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_RUN_TAG",
    "FORMAL_WORKERS",
    "FALLBACK_CANDIDATE_VARIANTS",
    "PHYSICAL_GPU_UUIDS",
    "RELAY_INITIALIZATION_SEED",
    "RELAY_WIDTH",
    "RUNTIME_SOURCE_PATHS",
    "RUN_ID_PREFIX",
    "SELECTION_METRICS",
    "SOURCE_LOCK_KEY",
    "SPLIT_SEED",
    "STORED_VALIDATION_METRICS",
    "TPDNERV8V3ExactRunner",
    "TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON",
    "TRAINING_SEED",
    "V3_RELAY_VERSION",
    "V8_PARENT_RELAY_OFF_REFERENCE",
    "build_selected_model",
    "candidate_contract",
    "completion_summary",
    "environment_contract",
    "formal_contract",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "parse_args",
    "protocol_payload",
    "require_evaluator_checkpoint_payload",
    "require_v3_run_identity",
    "run_directory",
    "run_training",
    "six_output_bce_loss",
    "source_lock_contract",
    "supported_candidate_variants",
]


if __name__ == "__main__":
    main()
