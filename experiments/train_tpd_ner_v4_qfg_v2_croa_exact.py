#!/usr/bin/env python3
"""Exact800 C/D training for the V4 QFG-V2-CROA integration.

Both formal arms start a new child trajectory from the same immutable V4
best-mIoU checkpoint.  They use the same integrated model, optimizer, data,
seed, and schedule:

* ``qfg_only`` (C) registers both QFG and target-survival modules but disables
  the target-survival objective with an exact weight of zero.
* ``tss_qfg`` (D) changes only the target-survival objective weight to 0.005.

The strict extension warm-start copies all 544 parent state entries and
preserves exactly 24 builder-owned extension entries under
``target_survival`` and ``tpd_qfg``.  Same-arm exact resume is the only resume
mode.  Importing this module is read-only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import train_tpd_pilot as base  # noqa: E402
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact as v4_exact,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact
    as survival_exact,
)
from experiments.tpd_extension_warm_start import (  # noqa: E402
    PROVENANCE_SCHEMA as EXTENSION_WARM_START_SCHEMA,
    load_parent_into_extension,
)
from model import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival as qfg_model,
)
from model import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v4_tail_aware_survival as survival_model,
)


ENTRY_SCHEMA = "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_entry_v1"
RUN_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_run_identity_v1"
)
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_source_lock_v1"
)
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_"
    "architecture_manifest_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_"
    "checkpoint_identity_v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_checkpoint_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_completion_summary_v1"
)
SOURCE_LOCK_KEY = "tpd_ner_v4_qfg_v2_croa_exact_source_lock"
RUN_ID_PREFIX = "tpd-ner-v4-qfg-v2-croa-exact:"

QFG_ONLY_VARIANT = "qfg_only"
TSS_QFG_VARIANT = "tss_qfg"
C_VARIANT = QFG_ONLY_VARIANT
D_VARIANT = TSS_QFG_VARIANT
SUPPORTED_CANDIDATE_VARIANTS = (
    QFG_ONLY_VARIANT,
    TSS_QFG_VARIANT,
)
FALLBACK_CANDIDATE_VARIANTS = SUPPORTED_CANDIDATE_VARIANTS

QFG_VARIANT = "qfg_v2_croa"
TSS_CONTROL_VARIANT = survival_exact.TSS_CONTROL_VARIANT
TSS_ON_VARIANT = survival_exact.TSS_ON_VARIANT
FORMAL_TSS_VARIANTS = {
    QFG_ONLY_VARIANT: TSS_CONTROL_VARIANT,
    TSS_QFG_VARIANT: TSS_ON_VARIANT,
}
FORMAL_SURVIVAL_WEIGHTS = {
    QFG_ONLY_VARIANT: 0.0,
    TSS_QFG_VARIANT: 0.005,
}
FORMAL_QFG_ONLY_RUN_TAG = "formal800_qfg_only"
FORMAL_TSS_QFG_RUN_TAG = "formal800_tss_qfg"
FORMAL_RUN_TAGS = {
    QFG_ONLY_VARIANT: FORMAL_QFG_ONLY_RUN_TAG,
    TSS_QFG_VARIANT: FORMAL_TSS_QFG_RUN_TAG,
}

TRAINING_SEED = survival_exact.TRAINING_SEED
SPLIT_SEED = survival_exact.SPLIT_SEED
FORMAL_EPOCHS = survival_exact.FORMAL_EPOCHS
FORMAL_BATCH_SIZE = survival_exact.FORMAL_BATCH_SIZE
FORMAL_PATCH_SIZE = survival_exact.FORMAL_PATCH_SIZE
FORMAL_WORKERS = survival_exact.FORMAL_WORKERS
FORMAL_VAL_FRACTION = survival_exact.FORMAL_VAL_FRACTION
FORMAL_EVAL_EVERY = survival_exact.FORMAL_EVAL_EVERY
FORMAL_BASE_LR = survival_exact.FORMAL_BASE_LR
FORMAL_MIN_LR = survival_exact.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = survival_exact.FORMAL_WARMUP_EPOCHS
FORMAL_THRESHOLD = survival_exact.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = survival_exact.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = survival_exact.FORMAL_TINY_AREA
FORMAL_AMP = survival_exact.FORMAL_AMP
FORMAL_EPS = survival_exact.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = (
    survival_exact.FORMAL_CUBLAS_WORKSPACE_CONFIG
)
FORMAL_INITIALIZATION_MODES = (
    exact_runner.EXTENSION_PARENT_MODE,
    "exact_resume",
)

PARENT_VARIANT = survival_exact.PARENT_VARIANT
PARENT_CHECKPOINT_EPOCH = survival_exact.PARENT_CHECKPOINT_EPOCH
PARENT_CHECKPOINT_ROLE = survival_exact.PARENT_CHECKPOINT_ROLE
PARENT_CHECKPOINT_ROLE_SHORT = (
    survival_exact.PARENT_CHECKPOINT_ROLE_SHORT
)
PARENT_CHECKPOINT_SHA256 = survival_exact.PARENT_CHECKPOINT_SHA256
PARENT_STATE_DICT_SHA256 = survival_exact.PARENT_STATE_DICT_SHA256
PARENT_STATE_DICT_PATH = survival_exact.PARENT_STATE_DICT_PATH
PARENT_CHECKPOINT_PATH = survival_exact.PARENT_CHECKPOINT_PATH

SURVIVAL_VERSION = survival_exact.SURVIVAL_VERSION
SURVIVAL_STATE_PREFIX = survival_exact.SURVIVAL_STATE_PREFIX
SURVIVAL_STATE_KEYS = tuple(survival_exact.SURVIVAL_STATE_KEYS)
PRODUCTION_SURVIVAL_PARAMETERS = (
    survival_exact.PRODUCTION_SURVIVAL_PARAMETERS
)
FORMAL_PARENT_STATE_KEY_COUNT = (
    survival_exact.FORMAL_PARENT_STATE_KEY_COUNT
)
TARGET_STATISTICS_SCHEMA = survival_exact.TARGET_STATISTICS_SCHEMA

QFG_VERSION = qfg_model.QFG_V2_CROA_INTEGRATION_VERSION
QFG_STATE_PREFIX = qfg_model.QFG_STATE_PREFIX
QFG_STATE_KEYS = tuple(qfg_model.QFG_STATE_KEYS)
QFG_TERMINAL_STATE_KEYS = tuple(qfg_model.QFG_TERMINAL_STATE_KEYS)
QFG_ALPHA_EFFECTIVE_INIT = qfg_model.FORMAL_QFG_ALPHA_EFFECTIVE_INIT
PRODUCTION_QFG_PARAMETERS = (
    qfg_model.PRODUCTION_QFG_V2_CROA_PARAMETERS
)
PRODUCTION_TOTAL_PARAMETERS = (
    qfg_model.PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
)
FORMAL_STATE_KEY_COUNT = (
    qfg_model.FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
)
QFG_NEW_MODULE_PREFIXES = ("target_survival", "tpd_qfg")
QFG_ZERO_INIT_PREFIXES = (
    "target_survival.heads.emb1.classifier",
    "target_survival.heads.emb2.classifier",
    *QFG_TERMINAL_STATE_KEYS,
)

DEFAULT_TARGET_STATISTICS_PATH = (
    survival_exact.DEFAULT_TARGET_STATISTICS_PATH
)
DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT / "experiments/tpd_ner_v4_qfg_v2_croa_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v1"
)

SELECTION_METRICS = survival_exact.SELECTION_METRICS
STORED_VALIDATION_METRICS = survival_exact.STORED_VALIDATION_METRICS

file_sha256 = survival_exact.file_sha256
canonical_sha256 = survival_exact.canonical_sha256
load_json_mapping = survival_exact.load_json_mapping
PreparedData = survival_exact.PreparedData
prepare_data = survival_exact.prepare_data
split_fingerprints = survival_exact.split_fingerprints
data_fingerprints = survival_exact.data_fingerprints
configure_determinism = survival_exact.configure_determinism
write_or_verify_json = survival_exact.write_or_verify_json
shared_exact = survival_exact.shared_exact
load_survival_target_statistics = (
    survival_exact.load_survival_target_statistics
)
compute_stage_loss = survival_exact.compute_stage_loss
EpochLossAccumulator = survival_exact.EpochLossAccumulator
PHYSICAL_GPU_UUIDS = dict(survival_exact.PHYSICAL_GPU_UUIDS)


def _ordered_unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return tuple(ordered)


RUNTIME_SOURCE_PATHS = _ordered_unique_paths(
    (
        Path(__file__).resolve(),
        REPO_ROOT
        / "model/"
        "tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
        REPO_ROOT / "model/tpd_frequency_gate_v2_croa.py",
        REPO_ROOT / "model/tpd_query_frequency_bridge.py",
        *survival_exact.RUNTIME_SOURCE_PATHS,
    )
)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            base.json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def supported_candidate_variants() -> tuple[str, ...]:
    return SUPPORTED_CANDIDATE_VARIANTS


def candidate_contract(candidate_variant: str) -> dict[str, Any]:
    if candidate_variant not in SUPPORTED_CANDIDATE_VARIANTS:
        raise ValueError(
            f"unsupported QFG exact variant {candidate_variant!r}; "
            f"choices={SUPPORTED_CANDIDATE_VARIANTS}"
        )
    weight = FORMAL_SURVIVAL_WEIGHTS[candidate_variant]
    return {
        "candidate_variant": candidate_variant,
        "qfg_variant": QFG_VARIANT,
        "tss_variant": FORMAL_TSS_VARIANTS[candidate_variant],
        "parent_variant": PARENT_VARIANT,
        "qfg_enabled": True,
        "survival_weight": weight,
        "tss_control": candidate_variant == QFG_ONLY_VARIANT,
        "formal_run_tag": FORMAL_RUN_TAGS[candidate_variant],
    }


def formal_contract() -> dict[str, Any]:
    statistics = load_survival_target_statistics()
    return {
        "candidate_variants": list(SUPPORTED_CANDIDATE_VARIANTS),
        "qfg_variant": QFG_VARIANT,
        "tss_variants": dict(FORMAL_TSS_VARIANTS),
        "dataset": "NUDT-SIRST",
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "patch_size": FORMAL_PATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "val_fraction": FORMAL_VAL_FRACTION,
        "eval_every": FORMAL_EVAL_EVERY,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "threshold": FORMAL_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        "initialization_modes": list(FORMAL_INITIALIZATION_MODES),
        "parent_checkpoint_path": str(
            PARENT_CHECKPOINT_PATH.relative_to(REPO_ROOT)
        ),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            PARENT_STATE_DICT_SHA256
        ),
        "new_module_prefixes": list(QFG_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(QFG_ZERO_INIT_PREFIXES),
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
        "qfg_version": QFG_VERSION,
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "qfg_state_key_count": len(QFG_STATE_KEYS),
        "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
        "qfg_terminal_initialization": "exact_zero",
        "qfg_alpha_effective_initialization": (
            QFG_ALPHA_EFFECTIVE_INIT
        ),
        "qfg_inference_required": True,
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_state_key_count": len(SURVIVAL_STATE_KEYS),
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "survival_target": "max_pool_16_binary_presence",
        "survival_downsample": survival_exact.SURVIVAL_DOWNSAMPLE,
        "survival_weights": dict(FORMAL_SURVIVAL_WEIGHTS),
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_path": str(
            DEFAULT_TARGET_STATISTICS_PATH.relative_to(REPO_ROOT)
        ),
        "survival_target_statistics_sha256": statistics["sha256"],
        "segmentation_objective_unchanged": True,
        "official_test_accessed": False,
    }


def _validate_formal_args(args: argparse.Namespace) -> None:
    candidate = candidate_contract(getattr(args, "variant", None))
    statistics_path = Path(
        getattr(args, "survival_target_statistics", "")
    ).resolve()
    _require_equal(
        "formal QFG target-statistics path",
        statistics_path,
        DEFAULT_TARGET_STATISTICS_PATH.resolve(),
    )
    statistics = load_survival_target_statistics(statistics_path)
    expected = {
        "dataset": "NUDT-SIRST",
        "epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "patch_size": FORMAL_PATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "val_fraction": FORMAL_VAL_FRACTION,
        "eval_every": FORMAL_EVAL_EVERY,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "threshold": FORMAL_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "eps": FORMAL_EPS,
        "amp": FORMAL_AMP,
        "run_tag": candidate["formal_run_tag"],
        "survival_weight": candidate["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "max_train_images": None,
        "max_val_images": None,
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            "formal QFG exact arguments differ: "
            f"expected={expected}, observed={observed}"
        )
    _require_equal(
        "formal QFG parent checkpoint path",
        Path(getattr(args, "parent_checkpoint", "")).resolve(),
        PARENT_CHECKPOINT_PATH.resolve(),
    )
    if bool(getattr(args, "parent_warm_start", False)) == bool(
        getattr(args, "exact_resume", False)
    ):
        raise ValueError(
            "QFG exact entry requires exactly one initialization mode"
        )
    device = getattr(args, "device", None)
    allow_cpu_smoke = getattr(args, "allow_cpu_smoke", False)
    if allow_cpu_smoke:
        if device != "cpu":
            raise ValueError("CPU smoke permission requires --device=cpu")
    elif device != "cuda:0":
        raise ValueError("formal QFG exact training requires --device=cuda:0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact800 paired QFG-only/TSS+QFG continued training from "
            "the immutable V4 best-mIoU checkpoint"
        )
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_CANDIDATE_VARIANTS,
        required=True,
    )
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=FORMAL_PATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--val-fraction", type=float, default=FORMAL_VAL_FRACTION)
    parser.add_argument("--eval-every", type=int, default=FORMAL_EVAL_EVERY)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=FORMAL_WARMUP_EPOCHS,
    )
    parser.add_argument("--threshold", type=float, default=FORMAL_THRESHOLD)
    parser.add_argument("--match-radius", type=float, default=FORMAL_MATCH_RADIUS)
    parser.add_argument("--tiny-area", type=int, default=FORMAL_TINY_AREA)
    parser.add_argument("--eps", type=float, default=FORMAL_EPS)
    parser.set_defaults(amp=FORMAL_AMP)
    parser.add_argument("--survival-weight", type=float, default=None)
    parser.add_argument("--survival-pos-weight", type=float, default=None)
    parser.add_argument(
        "--survival-target-statistics",
        type=Path,
        default=DEFAULT_TARGET_STATISTICS_PATH,
    )
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        default=PARENT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--exact-source-lock",
        type=Path,
        default=DEFAULT_EXACT_SOURCE_LOCK_PATH,
    )
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument(
        "--parent-warm-start",
        "--fresh",
        dest="parent_warm_start",
        action="store_true",
        help=(
            "start a new child trajectory by strict extension warm-start "
            "(--fresh is only a compatibility spelling)"
        ),
    )
    initialization.add_argument("--exact-resume", action="store_true")

    args = parser.parse_args(argv)
    candidate = candidate_contract(args.variant)
    if args.run_tag is None:
        args.run_tag = candidate["formal_run_tag"]
    if args.survival_weight is None:
        args.survival_weight = candidate["survival_weight"]
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    if args.survival_pos_weight is None:
        args.survival_pos_weight = statistics["survival_pos_weight"]
    args.fresh = bool(args.parent_warm_start)
    args.parent_variant = PARENT_VARIANT
    args.relay_enabled = True
    args.relay_width = v4_exact.RELAY_WIDTH
    args.relay_initialization_seed = (
        v4_exact.RELAY_INITIALIZATION_SEED
    )
    args.dc_support_mode = v4_exact.DC_SUPPORT_MODE
    args.tail_z_thresholds = dict(v4_exact.TAIL_Z_THRESHOLDS)
    args.qfg_variant = candidate["qfg_variant"]
    args.tss_variant = candidate["tss_variant"]
    _validate_formal_args(args)
    return args


def run_directory(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    return (
        Path(args.output_root).resolve()
        / args.dataset
        / args.variant
        / f"seed_{args.seed}_{args.run_tag}"
    )


def validate_parent_checkpoint(
    parent_checkpoint: Path = PARENT_CHECKPOINT_PATH,
) -> dict[str, Any]:
    return survival_exact.validate_parent_checkpoint(parent_checkpoint)


def _architecture_manifest(
    variant: str,
    model: nn.Module,
    validated_model: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(variant)
    manifest = _json_value(model.architecture_manifest())
    manifest.update(
        {
            "schema": ARCHITECTURE_MANIFEST_SCHEMA,
            "variant": variant,
            "qfg_variant": candidate["qfg_variant"],
            "tss_variant": candidate["tss_variant"],
            "model": (
                "model.tpd_ner_v8_mprs_dch_v4_tail_aware_"
                "qfg_v2_croa_survival."
                "TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet"
            ),
            "parent_checkpoint_variant": PARENT_VARIANT,
            "state_key_count": FORMAL_STATE_KEY_COUNT,
            "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
            "extension_state_key_count": (
                len(SURVIVAL_STATE_KEYS) + len(QFG_STATE_KEYS)
            ),
            "survival_state_key_count": len(SURVIVAL_STATE_KEYS),
            "survival_state_keys": list(SURVIVAL_STATE_KEYS),
            "qfg_state_key_count": len(QFG_STATE_KEYS),
            "qfg_state_keys": list(QFG_STATE_KEYS),
            "qfg_terminal_state_keys": list(QFG_TERMINAL_STATE_KEYS),
            "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
            "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
            "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
            "survival_version": SURVIVAL_VERSION,
            "survival_state_prefix": SURVIVAL_STATE_PREFIX,
            "qfg_version": QFG_VERSION,
            "qfg_state_prefix": QFG_STATE_PREFIX,
            "survival_head_initialization": "exact_zero",
            "qfg_terminal_initialization": "exact_zero",
            "qfg_alpha_effective_initialization": (
                QFG_ALPHA_EFFECTIVE_INIT
            ),
            "survival_training_only": True,
            "qfg_inference_required": True,
            "segmentation_path_modified": True,
            "segmentation_objective_unchanged": True,
            "training_output": "TPDForwardOutput",
            "evaluation_output": "legacy_six_segmentation_maps",
            "exact_resume_scope": "same_qfg_tss_variant_only",
            "cross_version_exact_resume_supported": False,
            "formal_amp": FORMAL_AMP,
            "eps": FORMAL_EPS,
        }
    )
    _require_equal(
        "QFG validated total parameters",
        validated_model.get("total_parameters"),
        PRODUCTION_TOTAL_PARAMETERS,
    )
    _require_equal(
        "QFG validated state-key count",
        validated_model.get("state_key_count"),
        FORMAL_STATE_KEY_COUNT,
    )
    return manifest


def _require_qfg_manifest(
    manifest: Any,
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("QFG architecture manifest is missing")
    value = _json_value(dict(manifest))
    candidate = candidate_contract(variant)
    expected = {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "parent_checkpoint_variant": PARENT_VARIANT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "extension_state_key_count": (
            len(SURVIVAL_STATE_KEYS) + len(QFG_STATE_KEYS)
        ),
        "survival_state_key_count": len(SURVIVAL_STATE_KEYS),
        "survival_state_keys": list(SURVIVAL_STATE_KEYS),
        "qfg_state_key_count": len(QFG_STATE_KEYS),
        "qfg_state_keys": list(QFG_STATE_KEYS),
        "qfg_terminal_state_keys": list(QFG_TERMINAL_STATE_KEYS),
        "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "qfg_version": QFG_VERSION,
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "survival_head_initialization": "exact_zero",
        "qfg_terminal_initialization": "exact_zero",
        "qfg_alpha_effective_initialization": QFG_ALPHA_EFFECTIVE_INIT,
        "survival_training_only": True,
        "qfg_inference_required": True,
        "segmentation_path_modified": True,
        "segmentation_objective_unchanged": True,
        "training_output": "TPDForwardOutput",
        "evaluation_output": "legacy_six_segmentation_maps",
        "exact_resume_scope": "same_qfg_tss_variant_only",
        "cross_version_exact_resume_supported": False,
        "formal_amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
    }
    for name, required in expected.items():
        _require_equal(f"QFG manifest {name}", value.get(name), required)
    _require_equal(
        "QFG manifest DC support",
        value.get("ner_dc_offset_support_mode"),
        v4_exact.DC_SUPPORT_MODE,
    )
    _require_equal(
        "QFG manifest tail thresholds",
        value.get("tail_z_thresholds"),
        dict(v4_exact.TAIL_Z_THRESHOLDS),
    )
    _require_equal(
        "QFG manifest attention tensors",
        value.get("qfg_modified_attention_tensors"),
        ["Q"],
    )
    return value


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    candidate = candidate_contract(variant)
    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError("formal QFG exact builder requires seed=42")
    if eps != FORMAL_EPS:
        raise ValueError(
            f"formal QFG exact builder requires eps={FORMAL_EPS}"
        )
    model, builder_metadata = (
        qfg_model.build_formal_v4_qfg_v2_croa_survival_model(seed=seed)
    )
    validated = qfg_model.validate_formal_qfg_v2_croa_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
    )
    if not isinstance(builder_metadata, Mapping):
        raise TypeError("formal QFG builder metadata is not a mapping")
    _require_equal(
        "formal QFG state-key count",
        len(model.state_dict()),
        FORMAL_STATE_KEY_COUNT,
    )
    _require_equal(
        "formal QFG total parameter count",
        sum(parameter.numel() for parameter in model.parameters()),
        PRODUCTION_TOTAL_PARAMETERS,
    )
    survival_keys = {
        name
        for name in model.state_dict()
        if name.startswith(SURVIVAL_STATE_PREFIX)
    }
    qfg_keys = {
        name
        for name in model.state_dict()
        if name.startswith(QFG_STATE_PREFIX)
    }
    _require_equal(
        "formal QFG Survival extension state keys",
        survival_keys,
        set(SURVIVAL_STATE_KEYS),
    )
    _require_equal(
        "formal QFG state keys",
        qfg_keys,
        set(QFG_STATE_KEYS),
    )
    manifest = _architecture_manifest(variant, model, validated)
    metadata: dict[str, Any] = {
        "variant": variant,
        "candidate_family": "v4_tail_aware_qfg_v2_croa_tss_pair",
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "parent_variant": PARENT_VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "qfg_enabled": True,
        "survival_weight": candidate["survival_weight"],
        "tss_control": candidate["tss_control"],
        "qfg_version": QFG_VERSION,
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "qfg_state_keys": list(QFG_STATE_KEYS),
        "qfg_terminal_state_keys": list(QFG_TERMINAL_STATE_KEYS),
        "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
        "qfg_terminal_initialization": "exact_zero",
        "qfg_alpha_effective_initialization": QFG_ALPHA_EFFECTIVE_INIT,
        "qfg_inference_required": True,
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_state_keys": list(SURVIVAL_STATE_KEYS),
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "segmentation_path_modified": True,
        "segmentation_objective_unchanged": True,
        "fresh_training": False,
        "warm_start_applied": True,
        "warm_start_schema": EXTENSION_WARM_START_SCHEMA,
        "initialization_mode": exact_runner.EXTENSION_PARENT_MODE,
        "parent_checkpoint_path": str(PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            PARENT_STATE_DICT_SHA256
        ),
        "formal_builder_contract": {
            "model": validated.get("model"),
            "parent_model": validated.get("parent_model"),
            "dc_support_mode": validated.get("dc_support_mode"),
            "tail_z_thresholds": _json_value(
                validated.get("tail_z_thresholds")
            ),
            "qfg_core_manifest": _json_value(
                validated.get("qfg_core_manifest")
            ),
        },
        "architecture_manifest": manifest,
        "architecture_id": canonical_sha256(manifest),
    }
    return model, metadata


def _require_qfg_metadata(
    metadata: Any,
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("QFG model metadata is missing")
    value = _json_value(dict(metadata))
    candidate = candidate_contract(variant)
    expected = {
        "variant": variant,
        "candidate_family": "v4_tail_aware_qfg_v2_croa_tss_pair",
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "parent_variant": PARENT_VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "qfg_enabled": True,
        "survival_weight": candidate["survival_weight"],
        "tss_control": candidate["tss_control"],
        "qfg_version": QFG_VERSION,
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "qfg_state_keys": list(QFG_STATE_KEYS),
        "qfg_terminal_state_keys": list(QFG_TERMINAL_STATE_KEYS),
        "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
        "qfg_terminal_initialization": "exact_zero",
        "qfg_alpha_effective_initialization": QFG_ALPHA_EFFECTIVE_INIT,
        "qfg_inference_required": True,
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_state_keys": list(SURVIVAL_STATE_KEYS),
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "segmentation_path_modified": True,
        "segmentation_objective_unchanged": True,
        "fresh_training": False,
        "warm_start_applied": True,
        "warm_start_schema": EXTENSION_WARM_START_SCHEMA,
        "initialization_mode": exact_runner.EXTENSION_PARENT_MODE,
        "parent_checkpoint_path": str(PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            PARENT_STATE_DICT_SHA256
        ),
    }
    for name, required in expected.items():
        _require_equal(f"QFG metadata {name}", value.get(name), required)
    manifest = _require_qfg_manifest(
        value.get("architecture_manifest"),
        variant=variant,
    )
    _require_equal(
        "QFG metadata architecture digest",
        value.get("architecture_id"),
        canonical_sha256(manifest),
    )
    return value


@dataclass(frozen=True)
class InitializationPlan:
    request: exact_runner.InitializationRequest
    contract: Mapping[str, Any]
    initial_model_state_sha256: str
    initial_rng: Mapping[str, Any] | None = None
    selection_policy: Mapping[str, Any] | None = None


def _existing_training_contract(
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, Any]:
    protocol = load_json_mapping(
        directory / "protocol.json",
        "existing QFG exact protocol",
    )
    _require_equal(
        "existing QFG protocol schema",
        protocol.get("schema"),
        ENTRY_SCHEMA,
    )
    identity = require_qfg_run_identity(
        protocol.get("run_identity"),
        label="existing QFG protocol",
        expected_variant=args.variant,
    )
    training = identity.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError("existing QFG protocol has no training contract")
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(
            f"existing QFG training contract lacks fields: {missing}"
        )
    return copy.deepcopy(dict(training))


def initialization_plan(
    args: argparse.Namespace,
    directory: Path,
    model: nn.Module,
) -> InitializationPlan:
    _validate_formal_args(args)
    if args.parent_warm_start:
        validate_parent_checkpoint(args.parent_checkpoint)
        parent_model, _ = survival_model.build_formal_v4_reference(
            seed=TRAINING_SEED
        )
        result = load_parent_into_extension(
            parent_checkpoint=args.parent_checkpoint,
            parent_model=parent_model,
            extension_model=model,
            new_module_prefixes=QFG_NEW_MODULE_PREFIXES,
            zero_init_prefixes=QFG_ZERO_INIT_PREFIXES,
            parent_state_dict_path=PARENT_STATE_DICT_PATH,
            expected_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            map_location="cpu",
        )
        qfg_model.validate_formal_qfg_v2_croa_survival_model(
            model,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
        )
        loaded_sha256 = exact_runner.initial_model_state_sha256(model)
        request = exact_runner.InitializationRequest.extension_parent(
            result.provenance(),
            loaded_child_model_state_sha256=loaded_sha256,
        )
        contract = request.initialization_contract()
        if contract is None:
            raise RuntimeError(
                "extension parent request produced no initialization contract"
            )
        return InitializationPlan(
            request=request,
            contract=contract,
            initial_model_state_sha256=loaded_sha256,
        )
    if args.exact_resume:
        training = _existing_training_contract(args, directory)
        initialization = training["initialization_contract"]
        if (
            not isinstance(initialization, Mapping)
            or initialization.get("mode")
            != exact_runner.EXTENSION_PARENT_MODE
        ):
            raise ValueError(
                "QFG exact resume lacks extension-parent initialization"
            )
        initial_rng = training["initial_rng"]
        selection_policy = training["selection_policy"]
        if not isinstance(initial_rng, Mapping):
            raise ValueError("existing QFG initial_rng is invalid")
        if not isinstance(selection_policy, Mapping):
            raise ValueError("existing QFG selection_policy is invalid")
        return InitializationPlan(
            request=exact_runner.InitializationRequest.exact(),
            contract=copy.deepcopy(dict(initialization)),
            initial_model_state_sha256=str(
                training["initial_model_state_sha256"]
            ),
            initial_rng=copy.deepcopy(dict(initial_rng)),
            selection_policy=copy.deepcopy(dict(selection_policy)),
        )
    raise RuntimeError("QFG exact entry requires warm-start or exact resume")


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if str(device) != "cuda:0":
            raise ValueError("each QFG exact process must use cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "QFG exact training requires one process-visible GPU"
            )
        shared_exact.visible_gpu_identity()
        if os.environ.get("PYTHONHASHSEED") != str(args.seed):
            raise RuntimeError(
                "the process must start with PYTHONHASHSEED equal to --seed"
            )
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != FORMAL_CUBLAS_WORKSPACE_CONFIG
        ):
            raise RuntimeError(
                "the process must start with CUBLAS_WORKSPACE_CONFIG="
                f"{FORMAL_CUBLAS_WORKSPACE_CONFIG}"
            )
    elif device.type != "cpu":
        raise ValueError("QFG exact entry supports only cpu or cuda:0")
    elif not args.allow_cpu_smoke:
        raise ValueError("CPU execution requires --allow-cpu-smoke")
    return device


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
        "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX"
    )
    physical_uuid = os.environ.get("TPD_NER_V4_QFG_PHYSICAL_GPU_UUID")
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "QFG physical GPU index must identify registered GPU 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            f"QFG physical GPU UUID differs for GPU {physical_index}"
        )
    if payload.get("device_uuid") != expected_uuid:
        raise RuntimeError(
            "visible cuda:0 UUID differs from QFG assignment"
        )
    if payload.get("cuda_visible_devices") != expected_uuid:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must use the assigned GPU UUID"
        )
    payload.update(
        {
            "physical_gpu_index": int(physical_index),
            "physical_gpu_uuid": expected_uuid,
            "physical_gpu_assignment_source": (
                "verified_v4_qfg_worker_environment"
            ),
        }
    )
    return payload


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
    target_statistics_path: Path = DEFAULT_TARGET_STATISTICS_PATH,
) -> dict[str, str]:
    path = Path(exact_source_lock_path).resolve()
    statistics = load_survival_target_statistics(target_statistics_path)
    payload = load_json_mapping(path, "QFG exact source lock")
    _require_equal(
        "QFG exact source-lock schema",
        payload.get("schema"),
        EXACT_SOURCE_LOCK_SCHEMA,
    )
    _require_equal(
        "QFG exact source-lock variants",
        tuple(payload.get("variants", ())),
        SUPPORTED_CANDIDATE_VARIANTS,
    )
    _require_equal(
        "QFG exact source-lock formal contract",
        payload.get("formal_contract"),
        formal_contract(),
    )
    _require_equal(
        "QFG exact source-lock training data",
        payload.get("training_data_sha256"),
        training_data_sha256,
    )
    _require_equal(
        "QFG exact source-lock target statistics",
        payload.get("survival_target_statistics_sha256"),
        statistics["sha256"],
    )
    _require_equal(
        "QFG exact source-lock parent checkpoint",
        payload.get("parent_checkpoint_sha256"),
        PARENT_CHECKPOINT_SHA256,
    )
    locked = payload.get("source_sha256")
    if not isinstance(locked, Mapping):
        raise ValueError("QFG exact source lock has no source mapping")
    required = {
        str(runtime.relative_to(REPO_ROOT))
        for runtime in RUNTIME_SOURCE_PATHS
    }
    _require_equal(
        "QFG exact source-lock source count",
        payload.get("source_count"),
        len(locked),
    )
    _require_equal(
        "QFG exact locked runtime source set",
        set(locked),
        required,
    )
    for relative, expected in locked.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("QFG source lock has an invalid path")
        runtime = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(runtime.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError("QFG source path escapes repository") from exc
        _require_equal("QFG source canonical path", canonical, relative)
        _require_equal(
            f"QFG source digest for {relative}",
            file_sha256(runtime),
            expected,
        )
    return {
        SOURCE_LOCK_KEY: file_sha256(path),
        "training_data": _validate_sha256(
            training_data_sha256,
            "training data SHA-256",
        ),
        "survival_target_statistics": statistics["sha256"],
        "parent_checkpoint": PARENT_CHECKPOINT_SHA256,
    }


def _loss_contract(
    variant: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(variant)
    return {
        "segmentation": {
            "class": "torch.nn.modules.loss.BCELoss",
            "reduction": "mean",
            "input": "post_sigmoid_probability",
            "outputs": 6,
            "aggregate": "python_ordered_sum",
            "compute_dtype": "float32",
            "unchanged_from_v4": True,
        },
        "survival": {
            "class": (
                "torch.nn.functional.binary_cross_entropy_with_logits"
            ),
            "reduction": "mean",
            "input": "raw_logits",
            "heads": ["emb1", "emb2"],
            "aggregate": "python_ordered_sum",
            "survival_weight": candidate["survival_weight"],
            "survival_pos_weight": statistics["survival_pos_weight"],
            "target": "max_pool_16_binary_presence",
            "target_pool_kernel": survival_exact.SURVIVAL_DOWNSAMPLE,
            "target_pool_stride": survival_exact.SURVIVAL_DOWNSAMPLE,
            "target_statistics_schema": TARGET_STATISTICS_SCHEMA,
            "target_statistics_sha256": statistics["sha256"],
            "target_statistics_used_train_ids_sha256": statistics[
                "used_train_ids_sha256"
            ],
            "disabled_path_builds_target": False,
            "disabled_path_reads_logits": False,
        },
        "variant": variant,
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "tss_control": candidate["tss_control"],
        "total": "segmentation + survival_weight * survival",
        "compute_dtype": "float32",
    }


def _required_determinism(
    variant: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(variant)
    return {
        "entry_schema": ENTRY_SCHEMA,
        "qfg_run_identity_schema": RUN_IDENTITY_SCHEMA,
        "candidate_variant": variant,
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "parent_variant": PARENT_VARIANT,
        "parent_checkpoint_path": str(PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            PARENT_STATE_DICT_SHA256
        ),
        "warm_start_applied": True,
        "warm_start_schema": EXTENSION_WARM_START_SCHEMA,
        "initialization_mode": exact_runner.EXTENSION_PARENT_MODE,
        "new_module_prefixes": list(QFG_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(QFG_ZERO_INIT_PREFIXES),
        "qfg_version": QFG_VERSION,
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
        "qfg_terminal_initialization": "exact_zero",
        "qfg_alpha_effective_initialization": QFG_ALPHA_EFFECTIVE_INIT,
        "qfg_inference_required": True,
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "survival_weight": candidate["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_sha256": statistics["sha256"],
        "survival_target_downsample": survival_exact.SURVIVAL_DOWNSAMPLE,
        "tss_control": candidate["tss_control"],
        "segmentation_path_modified": True,
        "segmentation_objective_unchanged": True,
        "relay_version": v4_exact.V4_RELAY_VERSION,
        "dc_support_mode": v4_exact.DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(v4_exact.TAIL_Z_THRESHOLDS),
        "workers": FORMAL_WORKERS,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        "explicit_cpu_loader_generator": True,
        "loader_generator_seed": TRAINING_SEED,
        "manual_lr_schedule": True,
        "scheduler": None,
        "scheduler_restore_mode": (
            "identity_bound_manual_schedule_from_completed_epoch"
        ),
        "drop_last": False,
        "skip_singleton_batches": True,
        "eval_every_epoch": True,
        "validation_uses_legacy_segmentation_output": True,
        "selection_uses_segmentation_metrics_only": True,
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
    metadata = _require_qfg_metadata(
        model_metadata,
        variant=args.variant,
    )
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    expected_source_locks = {
        SOURCE_LOCK_KEY,
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    }
    _require_equal(
        "QFG run source-lock keys",
        set(source_locks),
        expected_source_locks,
    )
    _require_equal(
        "QFG run target-statistics source lock",
        source_locks["survival_target_statistics"],
        statistics["sha256"],
    )
    _require_equal(
        "QFG run parent-checkpoint source lock",
        source_locks["parent_checkpoint"],
        PARENT_CHECKPOINT_SHA256,
    )
    return exact_runner.ExactRunSpec(
        run_id=(
            f"{RUN_ID_PREFIX}{args.dataset}:{args.variant}:"
            f"seed-{args.seed}:split-{args.split_seed}:{args.run_tag}"
        ),
        variant=args.variant,
        dataset=args.dataset,
        seed=args.seed,
        split_seed=args.split_seed,
        builder_metadata=metadata,
        builder_manifest_sha256=canonical_sha256(
            metadata["architecture_manifest"]
        ),
        source_locks=dict(source_locks),
        split_fingerprints=dict(split_records),
        data_fingerprints=dict(data_records),
        optimizer=exact_runner.optimizer_contract(model, optimizer),
        scaler=exact_runner.scaler_contract(scaler, amp=FORMAL_AMP),
        initialization_contract=copy.deepcopy(
            dict(initialization_contract)
        ),
        lr_schedule=exact_runner.ManualCosineSchedule(
            total_epochs=FORMAL_EPOCHS,
            base_lr=FORMAL_BASE_LR,
            min_lr=FORMAL_MIN_LR,
            warmup_epochs=FORMAL_WARMUP_EPOCHS,
        ),
        loss=_loss_contract(args.variant, statistics),
        deep_supervision={
            "enabled": True,
            "expected_segmentation_outputs": 6,
            "training_uses_all_segmentation_outputs": True,
            "training_output_contract": "TPDForwardOutput",
            "validation_uses_final_segmentation_output": True,
            "validation_output_contract": "legacy_six_tuple",
        },
        batch_size=FORMAL_BATCH_SIZE,
        patch_size=FORMAL_PATCH_SIZE,
        workers=FORMAL_WORKERS,
        amp=FORMAL_AMP,
        total_epochs=FORMAL_EPOCHS,
        eval_interval=FORMAL_EVAL_EVERY,
        metric_config={
            "threshold": FORMAL_THRESHOLD,
            "match_radius": FORMAL_MATCH_RADIUS,
            "tiny_area": FORMAL_TINY_AREA,
            "validation_batch_size": 1,
            "checkpoint_selection_uses_survival_loss": False,
            "official_test_accessed": False,
        },
        environment=dict(environment),
        determinism=_required_determinism(args.variant, statistics),
        initial_model_state_sha256=initial_model_state_sha256,
        initial_rng=copy.deepcopy(dict(initial_rng)),
        selection_policy=copy.deepcopy(dict(selection_policy)),
    )


def require_qfg_run_identity(
    identity: Any,
    *,
    label: str,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no QFG exact run identity")
    value = copy.deepcopy(dict(identity))
    variant = value.get("variant")
    candidate_contract(variant)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"{label} variant {variant!r} differs from "
            f"{expected_variant!r}"
        )
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"{label} run_id is not QFG-owned")
    _require_equal(
        f"{label} exact-runner identity schema",
        value.get("schema"),
        exact_runner.RUN_IDENTITY_SCHEMA,
    )
    _require_equal(f"{label} training seed", value.get("seed"), TRAINING_SEED)
    _require_equal(f"{label} split seed", value.get("split_seed"), SPLIT_SEED)
    source_locks = value.get("source_locks")
    expected_lock_keys = {
        SOURCE_LOCK_KEY,
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    }
    if (
        not isinstance(source_locks, Mapping)
        or set(source_locks) != expected_lock_keys
    ):
        raise ValueError(f"{label} source-lock identity is not QFG-owned")
    statistics = load_survival_target_statistics()
    _require_equal(
        f"{label} target-statistics digest",
        source_locks.get("survival_target_statistics"),
        statistics["sha256"],
    )
    _require_equal(
        f"{label} parent checkpoint digest",
        source_locks.get("parent_checkpoint"),
        PARENT_CHECKPOINT_SHA256,
    )
    training = value.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError(f"{label} training contract is missing")
    determinism = training.get("determinism")
    if not isinstance(determinism, Mapping):
        raise ValueError(f"{label} determinism identity is missing")
    for name, expected in _required_determinism(
        variant,
        statistics,
    ).items():
        _require_equal(
            f"{label} QFG determinism field {name}",
            determinism.get(name),
            expected,
        )
    _require_equal(
        f"{label} loss contract",
        training.get("loss"),
        _loss_contract(variant, statistics),
    )
    initialization = training.get("initialization_contract")
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("mode")
        != exact_runner.EXTENSION_PARENT_MODE
    ):
        raise ValueError(
            f"{label} initialization is not extension-parent warm-start"
        )
    provenance = initialization.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} warm-start provenance is missing")
    expected_provenance = {
        "schema": EXTENSION_WARM_START_SCHEMA,
        "parent_checkpoint_path": str(PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_path": list(PARENT_STATE_DICT_PATH),
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "preserved_new_state_key_count": (
            len(SURVIVAL_STATE_KEYS) + len(QFG_STATE_KEYS)
        ),
        "new_module_prefixes": list(QFG_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(QFG_ZERO_INIT_PREFIXES),
    }
    _require_equal(
        f"{label} extension-parent provenance",
        provenance,
        expected_provenance,
    )
    return value


def _require_complete_validation_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        name for name in STORED_VALIDATION_METRICS if name not in metrics
    ]
    if missing:
        raise ValueError(f"QFG validation metrics lack fields: {missing}")
    return {
        name: copy.deepcopy(metrics[name])
        for name in STORED_VALIDATION_METRICS
    }


EVALUATOR_CHECKPOINT_REQUIRED_FIELDS = (
    "schema",
    "epoch",
    "checkpoint_role",
    "variant",
    "candidate_variant",
    "qfg_variant",
    "tss_variant",
    "parent_variant",
    "dataset",
    "seed",
    "split_seed",
    "state_dict",
    "optimizer",
    "scaler",
    "scheduler",
    "validation_metrics",
    "model_metadata",
    "architecture_manifest",
    "architecture_id",
    "split_hashes",
    "run_identity",
    "checkpoint_identity",
    "source_locks",
    "exact_source_lock_sha256",
    "parent_checkpoint_path",
    "parent_checkpoint_sha256",
    "parent_checkpoint_role",
    "parent_checkpoint_epoch",
    "parent_checkpoint_state_dict_sha256",
    "warm_start_applied",
    "warm_start_schema",
    "initialization_mode",
    "qfg_version",
    "qfg_state_prefix",
    "qfg_parameters",
    "qfg_terminal_initialization",
    "qfg_alpha_effective_initialization",
    "qfg_inference_required",
    "survival_version",
    "survival_state_prefix",
    "survival_parameters",
    "survival_head_initialization",
    "survival_training_only",
    "survival_weight",
    "survival_pos_weight",
    "survival_target_statistics_sha256",
    "segmentation_path_modified",
    "segmentation_objective_unchanged",
    "tss_control",
    "selection_source",
    "official_test_accessed",
)


def _checkpoint_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    variant = str(identity["variant"])
    statistics = load_survival_target_statistics()
    candidate = candidate_contract(variant)
    source_locks = copy.deepcopy(dict(identity["source_locks"]))
    return {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "variant": variant,
        "candidate_variant": variant,
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "parent_variant": PARENT_VARIANT,
        "run_id": identity["run_id"],
        "architecture_id": identity["architecture_id"],
        "builder_manifest_sha256": identity["builder_manifest_sha256"],
        "source_locks": source_locks,
        "exact_source_lock_sha256": source_locks[SOURCE_LOCK_KEY],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            PARENT_STATE_DICT_SHA256
        ),
        "qfg_version": QFG_VERSION,
        "survival_version": SURVIVAL_VERSION,
        "survival_weight": candidate["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_sha256": statistics["sha256"],
        "tss_control": candidate["tss_control"],
    }


def require_evaluator_checkpoint_payload(
    payload: Any,
    *,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("QFG evaluator checkpoint is not a mapping")
    value = copy.deepcopy(dict(payload))
    missing = [
        field
        for field in EVALUATOR_CHECKPOINT_REQUIRED_FIELDS
        if field not in value
    ]
    if missing:
        raise ValueError(f"QFG evaluator checkpoint lacks fields: {missing}")
    _require_equal(
        "QFG checkpoint schema",
        value["schema"],
        CHECKPOINT_SCHEMA,
    )
    identity = require_qfg_run_identity(
        value["run_identity"],
        label="QFG evaluator checkpoint",
        expected_variant=expected_variant,
    )
    statistics = load_survival_target_statistics()
    candidate = candidate_contract(identity["variant"])
    source_locks = copy.deepcopy(dict(identity["source_locks"]))
    expected_top_level = {
        "variant": identity["variant"],
        "candidate_variant": identity["variant"],
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "parent_variant": PARENT_VARIANT,
        "dataset": identity["dataset"],
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "architecture_id": identity["architecture_id"],
        "source_locks": source_locks,
        "exact_source_lock_sha256": source_locks[SOURCE_LOCK_KEY],
        "parent_checkpoint_path": str(PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            PARENT_STATE_DICT_SHA256
        ),
        "warm_start_applied": True,
        "warm_start_schema": EXTENSION_WARM_START_SCHEMA,
        "initialization_mode": exact_runner.EXTENSION_PARENT_MODE,
        "qfg_version": QFG_VERSION,
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
        "qfg_terminal_initialization": "exact_zero",
        "qfg_alpha_effective_initialization": QFG_ALPHA_EFFECTIVE_INIT,
        "qfg_inference_required": True,
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "survival_weight": candidate["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_sha256": statistics["sha256"],
        "segmentation_path_modified": True,
        "segmentation_objective_unchanged": True,
        "tss_control": candidate["tss_control"],
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }
    for name, expected in expected_top_level.items():
        _require_equal(f"QFG checkpoint {name}", value.get(name), expected)
    epoch = value["epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or epoch > FORMAL_EPOCHS
    ):
        raise ValueError("QFG evaluator checkpoint epoch is invalid")
    if value["checkpoint_role"] not in {
        "last_evaluated_epoch",
        "best_validation_pd_primary",
        "best_validation_miou_secondary",
    }:
        raise ValueError("QFG evaluator checkpoint role is invalid")
    state = value["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("QFG evaluator state_dict is invalid")
    _require_equal(
        "QFG evaluator state-key count",
        len(state),
        FORMAL_STATE_KEY_COUNT,
    )
    _require_equal(
        "QFG evaluator Survival state keys",
        {
            name
            for name in state
            if isinstance(name, str)
            and name.startswith(SURVIVAL_STATE_PREFIX)
        },
        set(SURVIVAL_STATE_KEYS),
    )
    _require_equal(
        "QFG evaluator QFG state keys",
        {
            name
            for name in state
            if isinstance(name, str) and name.startswith(QFG_STATE_PREFIX)
        },
        set(QFG_STATE_KEYS),
    )
    if "state_dict_sha256" in value:
        _require_equal(
            "QFG evaluator state_dict SHA-256",
            value["state_dict_sha256"],
            exact_runner._state_content_sha256(
                state,
                "QFG evaluator state_dict",
            ),
        )
    for name in ("optimizer", "scaler"):
        if not isinstance(value[name], Mapping):
            raise ValueError(f"QFG evaluator checkpoint {name} is invalid")
    _require_equal(
        "QFG checkpoint scheduler",
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
            raise ValueError(f"QFG evaluator metric {name} is invalid")
    metadata = _require_qfg_metadata(
        value["model_metadata"],
        variant=identity["variant"],
    )
    manifest = _require_qfg_manifest(
        value["architecture_manifest"],
        variant=identity["variant"],
    )
    _require_equal(
        "QFG checkpoint manifest copy",
        manifest,
        metadata["architecture_manifest"],
    )
    _require_equal(
        "QFG identity architecture digest",
        identity.get("builder_manifest_sha256"),
        canonical_sha256(manifest),
    )
    split_hashes = value["split_hashes"]
    if not isinstance(split_hashes, Mapping) or not split_hashes:
        raise ValueError("QFG evaluator split hashes are invalid")
    for name, digest in split_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("QFG evaluator split-hash name is invalid")
        _validate_sha256(digest, f"QFG split hash {name}")
    _require_equal(
        "QFG evaluator checkpoint identity",
        value["checkpoint_identity"],
        _checkpoint_identity(identity),
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
        identity = require_qfg_run_identity(
            context.run_identity,
            label="QFG checkpoint context",
        )
        metadata = _require_qfg_metadata(
            self.model_metadata,
            variant=identity["variant"],
        )
        statistics = load_survival_target_statistics()
        candidate = candidate_contract(identity["variant"])
        source_locks = copy.deepcopy(dict(identity["source_locks"]))
        exact_payload = context.exact_payload
        state_dict = copy.deepcopy(
            exact_payload["model"]["state_dict"]
        )
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": identity["variant"],
            "candidate_variant": identity["variant"],
            "qfg_variant": candidate["qfg_variant"],
            "tss_variant": candidate["tss_variant"],
            "parent_variant": PARENT_VARIANT,
            "dataset": identity["dataset"],
            "seed": identity["seed"],
            "split_seed": identity["split_seed"],
            "state_dict": state_dict,
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
            "architecture_manifest": copy.deepcopy(
                metadata["architecture_manifest"]
            ),
            "architecture_id": identity["architecture_id"],
            "split_hashes": copy.deepcopy(dict(self.split_hashes)),
            "run_identity": identity,
            "checkpoint_identity": _checkpoint_identity(identity),
            "source_locks": source_locks,
            "exact_source_lock_sha256": source_locks[SOURCE_LOCK_KEY],
            "parent_checkpoint_path": str(
                PARENT_CHECKPOINT_PATH.resolve()
            ),
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
            "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
            "parent_checkpoint_state_dict_sha256": (
                PARENT_STATE_DICT_SHA256
            ),
            "warm_start_applied": True,
            "warm_start_schema": EXTENSION_WARM_START_SCHEMA,
            "initialization_mode": exact_runner.EXTENSION_PARENT_MODE,
            "qfg_version": QFG_VERSION,
            "qfg_state_prefix": QFG_STATE_PREFIX,
            "qfg_parameters": PRODUCTION_QFG_PARAMETERS,
            "qfg_terminal_initialization": "exact_zero",
            "qfg_alpha_effective_initialization": (
                QFG_ALPHA_EFFECTIVE_INIT
            ),
            "qfg_inference_required": True,
            "survival_version": SURVIVAL_VERSION,
            "survival_state_prefix": SURVIVAL_STATE_PREFIX,
            "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
            "survival_head_initialization": "exact_zero",
            "survival_training_only": True,
            "survival_weight": candidate["survival_weight"],
            "survival_pos_weight": statistics["survival_pos_weight"],
            "survival_target_statistics_sha256": statistics["sha256"],
            "segmentation_path_modified": True,
            "segmentation_objective_unchanged": True,
            "tss_control": candidate["tss_control"],
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }
        return require_evaluator_checkpoint_payload(
            payload,
            expected_variant=identity["variant"],
        )


class TPDNERV4QFGV2CROAExactRunner(
    v4_exact.TPDNERV8V4TailAwareExactRunner
):
    """Reject every non-QFG C/D journal before exact state restoration."""

    def _require_v8_active_journal(self) -> None:
        active = self.journal.load_active()
        if active is None:
            return
        payload, _ = self._load_exact_payload(active.checkpoint_path)
        require_qfg_run_identity(
            payload.get("run_identity"),
            label="active QFG exact journal",
            expected_variant=self.spec.variant,
        )
        if not isinstance(payload.get("optimizer"), Mapping):
            raise ValueError("active QFG journal has no optimizer state")


DCHExactRunner = TPDNERV4QFGV2CROAExactRunner


def training_arguments(args: argparse.Namespace) -> dict[str, Any]:
    _validate_formal_args(args)
    names = (
        "variant",
        "dataset",
        "dataset_dir",
        "output_root",
        "run_tag",
        "device",
        "epochs",
        "batch_size",
        "patch_size",
        "workers",
        "seed",
        "split_seed",
        "val_fraction",
        "eval_every",
        "base_lr",
        "min_lr",
        "warmup_epochs",
        "threshold",
        "match_radius",
        "tiny_area",
        "eps",
        "amp",
        "allow_cpu_smoke",
        "exact_source_lock",
        "parent_checkpoint",
        "survival_target_statistics",
        "survival_weight",
        "survival_pos_weight",
        "max_train_images",
        "max_val_images",
        "qfg_variant",
        "tss_variant",
    )
    return {name: getattr(args, name) for name in names}


def protocol_payload(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    normalization: Mapping[str, float],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = require_qfg_run_identity(
        run_identity,
        label="QFG protocol",
        expected_variant=args.variant,
    )
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    candidate = candidate_contract(args.variant)
    return {
        "schema": ENTRY_SCHEMA,
        "formal_contract": formal_contract(),
        "arguments": training_arguments(args),
        "run_directory": directory,
        "model": _require_qfg_metadata(
            model_metadata,
            variant=args.variant,
        ),
        "architecture_manifest": _require_qfg_manifest(
            model_metadata["architecture_manifest"],
            variant=args.variant,
        ),
        "normalization": dict(normalization),
        "run_identity": identity,
        "candidate_variant": args.variant,
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "source_locks": copy.deepcopy(dict(identity["source_locks"])),
        "parent_checkpoint": {
            "path": str(PARENT_CHECKPOINT_PATH.resolve()),
            "sha256": PARENT_CHECKPOINT_SHA256,
            "role": PARENT_CHECKPOINT_ROLE_SHORT,
            "serialized_role": PARENT_CHECKPOINT_ROLE,
            "epoch": PARENT_CHECKPOINT_EPOCH,
            "state_dict_sha256": PARENT_STATE_DICT_SHA256,
        },
        "survival_target_statistics": {
            "path": str(DEFAULT_TARGET_STATISTICS_PATH.resolve()),
            "sha256": statistics["sha256"],
            "survival_pos_weight": statistics["survival_pos_weight"],
        },
        "loss": _loss_contract(args.variant, statistics),
        "selection_order_metrics": list(SELECTION_METRICS),
        "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
        "checkpoint_selection": {
            "best": "best_validation_pd_primary",
            "best_miou": "best_validation_miou_secondary",
            "uses_survival_loss": False,
            "source": "internal_validation_only",
        },
        "exact_resume_policy": {
            "new_trajectory": exact_runner.EXTENSION_PARENT_MODE,
            "parent_epoch_is_child_completed_epoch": False,
            "initial_child_completed_epoch": 0,
            "same_variant_epoch_boundary_only": True,
            "cross_variant": "forbidden",
            "cross_version": "forbidden",
            "optimizer_inherited_from_parent": False,
            "scheduler_restore": (
                "manual_schedule_reconstructed_from_identity_and_epoch"
            ),
        },
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }


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
        raise RuntimeError("QFG exact metrics are not a contiguous history")
    for event in events:
        _require_complete_validation_metrics(event)
        for name in (
            "train_total_loss",
            "train_segmentation_loss",
            "train_survival_loss",
            "train_survival_emb1_loss",
            "train_survival_emb2_loss",
        ):
            if name not in event:
                raise RuntimeError(f"QFG exact event lacks {name!r}")
    return events


def completion_summary(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    events = _load_complete_events(
        directory / exact_runner.METRICS_FILENAME,
        FORMAL_EPOCHS,
    )
    pd_epoch = int(selection["primary"]["epoch"])
    miou_epoch = int(selection["secondary"]["epoch"])
    protocol = load_json_mapping(
        directory / "protocol.json",
        "completed QFG exact protocol",
    )
    identity = require_qfg_run_identity(
        protocol.get("run_identity"),
        label="completed QFG protocol",
        expected_variant=args.variant,
    )
    candidate = candidate_contract(args.variant)
    return {
        "schema": COMPLETION_SUMMARY_SCHEMA,
        "status": "complete",
        "variant": args.variant,
        "candidate_variant": args.variant,
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "formal_contract": formal_contract(),
        "run_identity": identity,
        "source_locks": copy.deepcopy(dict(identity["source_locks"])),
        "best_epoch": pd_epoch,
        "best_validation_metrics": (
            _require_complete_validation_metrics(events[pd_epoch - 1])
        ),
        "best_pd_epoch": pd_epoch,
        "best_pd_validation_metrics": (
            _require_complete_validation_metrics(events[pd_epoch - 1])
        ),
        "best_miou_epoch": miou_epoch,
        "best_miou_validation_metrics": (
            _require_complete_validation_metrics(events[miou_epoch - 1])
        ),
        "model": _require_qfg_metadata(
            model_metadata,
            variant=args.variant,
        ),
        "split_hashes": dict(split_hashes),
        "survival_weight": args.survival_weight,
        "tss_control": candidate["tss_control"],
        "selection_order_metrics": list(SELECTION_METRICS),
        "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "skipped_singleton_batches": sum(
            int(event.get("skipped_singleton_batches", 0))
            for event in events
        ),
        "elapsed_seconds": sum(
            float(event["epoch_seconds"]) for event in events
        ),
        "best_checkpoint": directory / exact_runner.BEST_FILENAME,
        "best_miou_checkpoint": (
            directory / exact_runner.BEST_MIOU_FILENAME
        ),
        "last_checkpoint": directory / exact_runner.LAST_FILENAME,
    }


def _check_metrics(metrics: Mapping[str, Any]) -> None:
    _require_complete_validation_metrics(metrics)
    for name, value in metrics.items():
        if (
            isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            raise FloatingPointError(
                f"non-finite QFG validation metric {name!r}: {value!r}"
            )


def _require_prepared_statistics_match(
    statistics: Mapping[str, Any],
    prepared: PreparedData,
) -> None:
    _require_equal(
        "QFG statistics used-train IDs",
        statistics["used_train_ids_sha256"],
        prepared.split_hashes["used_train_sha256"],
    )
    _require_equal(
        "QFG statistics train image count",
        statistics["train_image_count"],
        len(prepared.train_ids),
    )


def run_training(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    configure_determinism()
    device = resolve_device(args)
    file_sha256(Path(args.exact_source_lock).resolve())
    directory = run_directory(args)
    prepared = prepare_data(args)
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    _require_prepared_statistics_match(statistics, prepared)
    sources = source_lock_contract(
        prepared.training_data_sha256,
        args.exact_source_lock,
        args.survival_target_statistics,
    )

    model, model_metadata = build_selected_model(
        args.variant,
        args.seed,
        eps=args.eps,
    )
    plan = initialization_plan(args, directory, model)
    model.to(device)
    train_set = base.TrainingSubset(
        prepared.dataset_dir,
        args.dataset,
        args.patch_size,
        prepared.train_ids,
        prepared.normalization,
    )
    val_set = base.ValidationSubset(
        prepared.dataset_root,
        prepared.val_ids,
        prepared.normalization,
    )

    base.seed_everything(args.seed)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=FORMAL_WORKERS,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=FORMAL_WORKERS,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=FORMAL_BASE_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=FORMAL_AMP)
    selection_policy = exact_runner.pd_miou_selection_policy(
        stored_metrics=STORED_VALIDATION_METRICS
    )
    actual_selection_policy = selection_policy.normalized()
    spec_selection_policy = (
        copy.deepcopy(dict(plan.selection_policy))
        if plan.selection_policy is not None
        else actual_selection_policy
    )
    initial_rng = (
        copy.deepcopy(dict(plan.initial_rng))
        if plan.initial_rng is not None
        else exact_runner.initial_rng_contract()
    )
    spec = make_exact_run_spec(
        args,
        model=model,
        model_metadata=model_metadata,
        optimizer=optimizer,
        scaler=scaler,
        initialization_contract=plan.contract,
        initial_model_state_sha256=plan.initial_model_state_sha256,
        initial_rng=initial_rng,
        selection_policy=spec_selection_policy,
        source_locks=sources,
        split_records=split_fingerprints(prepared),
        data_records=data_fingerprints(prepared),
        environment=environment_contract(device),
    )
    adapter = EvaluatorCheckpointAdapter(
        model_metadata=model_metadata,
        split_hashes=prepared.split_hashes,
    )
    runner = DCHExactRunner(
        directory,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=loader_generator,
        spec=spec,
        selection_policy=selection_policy,
        compatibility_payload_factory=adapter,
    )
    snapshot = runner.startup(plan.request)
    write_or_verify_json(directory / "split.json", prepared.split_manifest)
    write_or_verify_json(
        directory / "protocol.json",
        protocol_payload(
            args,
            directory=directory,
            model_metadata=model_metadata,
            normalization=prepared.normalization,
            run_identity=snapshot.run_identity,
        ),
    )

    print(
        f"START variant={args.variant} "
        f"qfg={args.qfg_variant} tss={args.tss_variant} "
        f"mode={snapshot.initialization_mode.value} "
        f"completed={snapshot.completed_epoch} "
        f"next={snapshot.next_epoch} device={device}",
        flush=True,
    )
    survival_enabled = args.survival_weight > 0.0
    while snapshot.next_epoch is not None:
        control = runner.next_epoch_control()
        if not control.should_evaluate:
            raise RuntimeError(
                "formal QFG exact training must evaluate each epoch"
            )
        epoch_started = time.time()
        model.train()
        accumulator = EpochLossAccumulator(
            survival_enabled=survival_enabled
        )
        skipped_singletons = 0
        for images, masks in train_loader:
            if images.shape[0] == 1:
                skipped_singletons += 1
                continue
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=False,
            ):
                outputs = model(images)
                losses = compute_stage_loss(
                    outputs,
                    masks,
                    criterion,
                    survival_weight=args.survival_weight,
                    survival_pos_weight=args.survival_pos_weight,
                )
            scaler.scale(losses.total).backward()
            scaler.step(optimizer)
            scaler.update()
            accumulator.update(losses, int(images.shape[0]))
        loss_fields = accumulator.fields()

        metrics = base.validate(
            model,
            val_loader,
            device,
            criterion,
            args.threshold,
            args.match_radius,
            args.tiny_area,
            FORMAL_AMP,
        )
        _check_metrics(metrics)
        snapshot = runner.commit_epoch(
            {
                "variant": args.variant,
                "qfg_variant": args.qfg_variant,
                "tss_variant": args.tss_variant,
                **loss_fields,
                "train_loss": loss_fields["train_total_loss"],
                "survival_weight": args.survival_weight,
                "survival_pos_weight": args.survival_pos_weight,
                "processed_train_samples": accumulator.sample_count,
                "epoch_seconds": time.time() - epoch_started,
                "skipped_singleton_batches": skipped_singletons,
                **metrics,
            },
            extra_state={
                "variant": args.variant,
                "qfg_variant": args.qfg_variant,
                "tss_variant": args.tss_variant,
                "formal_eps": FORMAL_EPS,
                "survival_weight": args.survival_weight,
                "survival_pos_weight": args.survival_pos_weight,
                "survival_target_statistics_sha256": statistics["sha256"],
                "processed_train_samples": accumulator.sample_count,
                "skipped_singleton_batches": skipped_singletons,
            },
        )
        print(
            f"EPOCH {control.epoch:03d}/{FORMAL_EPOCHS} "
            f"total={float(loss_fields['train_total_loss']):.6f} "
            f"seg={float(loss_fields['train_segmentation_loss']):.6f} "
            f"surv={float(loss_fields['train_survival_loss']):.6f} "
            f"mIoU={float(metrics['miou']):.6f} "
            f"Pd={float(metrics['pd']):.6f} "
            f"Fa={float(metrics['fa']):.8f}",
            flush=True,
        )

    if snapshot.best_selection is None:
        raise RuntimeError("completed QFG exact run has no best selection")
    summary = completion_summary(
        args,
        directory=directory,
        model_metadata=model_metadata,
        split_hashes=prepared.split_hashes,
        selection=snapshot.best_selection,
    )
    write_or_verify_json(directory / "summary.json", summary)
    print(
        f"COMPLETE variant={args.variant} "
        f"bestPdEpoch={summary['best_pd_epoch']} "
        f"bestMiouEpoch={summary['best_miou_epoch']}",
        flush=True,
    )
    return directory


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


__all__ = [
    "ARCHITECTURE_MANIFEST_SCHEMA",
    "CHECKPOINT_IDENTITY_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "COMPLETION_SUMMARY_SCHEMA",
    "C_VARIANT",
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TARGET_STATISTICS_PATH",
    "DCHExactRunner",
    "D_VARIANT",
    "ENTRY_SCHEMA",
    "EVALUATOR_CHECKPOINT_REQUIRED_FIELDS",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EXTENSION_WARM_START_SCHEMA",
    "EpochLossAccumulator",
    "EvaluatorCheckpointAdapter",
    "FALLBACK_CANDIDATE_VARIANTS",
    "FORMAL_AMP",
    "FORMAL_BASE_LR",
    "FORMAL_BATCH_SIZE",
    "FORMAL_EPOCHS",
    "FORMAL_EPS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_MATCH_RADIUS",
    "FORMAL_MIN_LR",
    "FORMAL_PATCH_SIZE",
    "FORMAL_QFG_ONLY_RUN_TAG",
    "FORMAL_RUN_TAGS",
    "FORMAL_STATE_KEY_COUNT",
    "FORMAL_SURVIVAL_WEIGHTS",
    "FORMAL_THRESHOLD",
    "FORMAL_TINY_AREA",
    "FORMAL_TSS_QFG_RUN_TAG",
    "FORMAL_TSS_VARIANTS",
    "FORMAL_VAL_FRACTION",
    "FORMAL_WARMUP_EPOCHS",
    "FORMAL_WORKERS",
    "InitializationPlan",
    "PARENT_CHECKPOINT_EPOCH",
    "PARENT_CHECKPOINT_PATH",
    "PARENT_CHECKPOINT_ROLE",
    "PARENT_CHECKPOINT_ROLE_SHORT",
    "PARENT_CHECKPOINT_SHA256",
    "PARENT_STATE_DICT_SHA256",
    "PARENT_VARIANT",
    "PRODUCTION_QFG_PARAMETERS",
    "PRODUCTION_SURVIVAL_PARAMETERS",
    "PRODUCTION_TOTAL_PARAMETERS",
    "QFG_ALPHA_EFFECTIVE_INIT",
    "QFG_NEW_MODULE_PREFIXES",
    "QFG_ONLY_VARIANT",
    "QFG_STATE_KEYS",
    "QFG_STATE_PREFIX",
    "QFG_TERMINAL_STATE_KEYS",
    "QFG_VARIANT",
    "QFG_VERSION",
    "QFG_ZERO_INIT_PREFIXES",
    "RUN_IDENTITY_SCHEMA",
    "RUN_ID_PREFIX",
    "RUNTIME_SOURCE_PATHS",
    "SOURCE_LOCK_KEY",
    "SPLIT_SEED",
    "STORED_VALIDATION_METRICS",
    "SUPPORTED_CANDIDATE_VARIANTS",
    "SURVIVAL_STATE_KEYS",
    "SURVIVAL_STATE_PREFIX",
    "SURVIVAL_VERSION",
    "TARGET_STATISTICS_SCHEMA",
    "TPDNERV4QFGV2CROAExactRunner",
    "TRAINING_SEED",
    "TSS_CONTROL_VARIANT",
    "TSS_ON_VARIANT",
    "TSS_QFG_VARIANT",
    "build_selected_model",
    "candidate_contract",
    "completion_summary",
    "compute_stage_loss",
    "environment_contract",
    "formal_contract",
    "initialization_plan",
    "load_survival_target_statistics",
    "main",
    "make_exact_run_spec",
    "parse_args",
    "protocol_payload",
    "require_evaluator_checkpoint_payload",
    "require_qfg_run_identity",
    "run_directory",
    "run_training",
    "source_lock_contract",
    "supported_candidate_variants",
    "training_arguments",
    "validate_parent_checkpoint",
]


if __name__ == "__main__":
    main()
