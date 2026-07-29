#!/usr/bin/env python3
"""Paired exact800 DLR experiment with a preselected 100-epoch TSS ramp.

The two arms share the same initialized network, data order, optimizer groups,
manual LR schedule and BatchNorm policy.  Their only numerical difference is
the epoch-controlled target-survival weight:

``qfg_dlr``
    ``w(e) = 0`` for every epoch.

``tss_qfg_dlr``
    ``w(e) = 0.005 * clamp((e - 1) / 99, 0, 1)``.

At epoch one both arms therefore use the exact zero-weight path: no survival
target is built, survival logits are not read, and the TSS parameters receive
no optimizer state.  The public variant, source lock, run identity, checkpoint
and completion schemas are owned by this paired recipe.  Importing the module
does not create a run directory or touch CUDA.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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

from experiments import tpd_exact_resume as exact_resume  # noqa: E402
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import tpd_group_scaled_exact_runner as scaled_runner  # noqa: E402
from experiments import (  # noqa: E402
    train_tpd_ner_v4_qfg_v2_croa_dlr_exact as dlr,
)


v2 = dlr.v2

ENTRY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_entry_v1"
)
PAIRED_RUN_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_run_identity_v1"
)
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_source_lock_v1"
)
MODEL_METADATA_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_model_metadata_v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_checkpoint_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "checkpoint_identity_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "completion_summary_v1"
)
SOURCE_LOCK_KEY = (
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock"
)
UPSTREAM_SOURCE_LOCK_KEY = "upstream_qfg_v2_exact_source_lock"
RUN_ID_PREFIX = "tpd-ner-v4-qfg-v2-croa-dlr-ramp100-exact:"

QFG_DLR_VARIANT = "qfg_dlr"
TSS_QFG_DLR_VARIANT = "tss_qfg_dlr"
SUPPORTED_CANDIDATE_VARIANTS = (
    QFG_DLR_VARIANT,
    TSS_QFG_DLR_VARIANT,
)
FAMILY_RECIPE = "qfg_tss_dlr_bn_ramp100_paired_v1"
CONTROL_RECIPE = "qfg_dlr_zero_tss_v1"
TREATMENT_RECIPE = "tss_qfg_dlr_ramp100_v1"
CANDIDATE_RECIPES = {
    QFG_DLR_VARIANT: CONTROL_RECIPE,
    TSS_QFG_DLR_VARIANT: TREATMENT_RECIPE,
}
BASE_MODEL_VARIANTS = {
    QFG_DLR_VARIANT: v2.QFG_ONLY_VARIANT,
    TSS_QFG_DLR_VARIANT: v2.TSS_QFG_VARIANT,
}
TSS_VARIANTS = {
    QFG_DLR_VARIANT: v2.TSS_CONTROL_VARIANT,
    TSS_QFG_DLR_VARIANT: v2.TSS_ON_VARIANT,
}
SURVIVAL_WEIGHT_MAXIMA = {
    QFG_DLR_VARIANT: 0.0,
    TSS_QFG_DLR_VARIANT: 0.005,
}
FORMAL_RUN_TAGS = {
    QFG_DLR_VARIANT: "formal800_qfg_dlr_control",
    TSS_QFG_DLR_VARIANT: "formal800_tss_qfg_dlr_ramp100",
}

TSS_WEIGHT_SCHEDULE_ID = "inclusive_epoch_linear_ramp100_v1"
CONTROL_WEIGHT_SCHEDULE_ID = "exact_zero_tss_control_v1"
TSS_RAMP_START_EPOCH = 1
TSS_RAMP_END_EPOCH = 100
TSS_RAMP_DENOMINATOR = 99
TSS_MAX_WEIGHT = 0.005
TSS_WEIGHT_FORMULA = (
    "0.005*(min(max(epoch-1,0),99)/99)"
)

BATCHNORM_EVENT_FIELD = dlr.BATCHNORM_EVENT_FIELD
SURVIVAL_WEIGHT_FIELD = "survival_weight_effective"
SURVIVAL_WEIGHT_MAX_FIELD = "survival_weight_max"
TSS_RAMP_FRACTION_FIELD = "tss_ramp_fraction"
TSS_WEIGHT_SCHEDULE_FIELD = "tss_weight_schedule_id"
WEIGHTED_SURVIVAL_LOSS_FIELD = "train_weighted_survival_loss"
SURVIVAL_ENABLED_FIELD = "survival_objective_enabled"
_RAMP_RUNNER_OWNED_FIELDS = frozenset(
    {
        "survival_weight",
        SURVIVAL_WEIGHT_FIELD,
        SURVIVAL_WEIGHT_MAX_FIELD,
        TSS_RAMP_FRACTION_FIELD,
        TSS_WEIGHT_SCHEDULE_FIELD,
        WEIGHTED_SURVIVAL_LOSS_FIELD,
        SURVIVAL_ENABLED_FIELD,
    }
)

TRAINING_SEED = v2.TRAINING_SEED
SPLIT_SEED = v2.SPLIT_SEED
FORMAL_EPOCHS = v2.FORMAL_EPOCHS
FORMAL_BATCH_SIZE = v2.FORMAL_BATCH_SIZE
FORMAL_PATCH_SIZE = v2.FORMAL_PATCH_SIZE
FORMAL_WORKERS = v2.FORMAL_WORKERS
FORMAL_VAL_FRACTION = v2.FORMAL_VAL_FRACTION
FORMAL_EVAL_EVERY = v2.FORMAL_EVAL_EVERY
FORMAL_BASE_LR = v2.FORMAL_BASE_LR
FORMAL_MIN_LR = v2.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = v2.FORMAL_WARMUP_EPOCHS
FORMAL_THRESHOLD = v2.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = v2.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = v2.FORMAL_TINY_AREA
FORMAL_AMP = v2.FORMAL_AMP
FORMAL_EPS = v2.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = v2.FORMAL_CUBLAS_WORKSPACE_CONFIG

PARENT_VARIANT = v2.PARENT_VARIANT
PARENT_CHECKPOINT_EPOCH = v2.PARENT_CHECKPOINT_EPOCH
PARENT_CHECKPOINT_ROLE = v2.PARENT_CHECKPOINT_ROLE
PARENT_CHECKPOINT_ROLE_SHORT = v2.PARENT_CHECKPOINT_ROLE_SHORT
PARENT_CHECKPOINT_SHA256 = v2.PARENT_CHECKPOINT_SHA256
PARENT_STATE_DICT_SHA256 = v2.PARENT_STATE_DICT_SHA256
PARENT_CHECKPOINT_PATH = v2.PARENT_CHECKPOINT_PATH
DEFAULT_TARGET_STATISTICS_PATH = v2.DEFAULT_TARGET_STATISTICS_PATH
STORED_VALIDATION_METRICS = v2.STORED_VALIDATION_METRICS
SELECTION_METRICS = v2.SELECTION_METRICS

UPSTREAM_SOURCE_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
)
UPSTREAM_SOURCE_LOCK_SHA256 = (
    "8d55464851db9441383854189eff64c05"
    "daf25e7ff3502c6c67cf06401996478"
)
DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_v1"
)

file_sha256 = v2.file_sha256
canonical_sha256 = v2.canonical_sha256
load_json_mapping = v2.load_json_mapping
configure_determinism = v2.configure_determinism
prepare_data = v2.prepare_data
split_fingerprints = v2.split_fingerprints
data_fingerprints = v2.data_fingerprints
load_survival_target_statistics = v2.load_survival_target_statistics
compute_stage_loss = v2.compute_stage_loss
EpochLossAccumulator = v2.EpochLossAccumulator


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
    (Path(__file__).resolve(), *dlr.RUNTIME_SOURCE_PATHS)
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
            v2.base.json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def supported_candidate_variants() -> tuple[str, ...]:
    return SUPPORTED_CANDIDATE_VARIANTS


def candidate_contract(candidate_variant: str) -> dict[str, Any]:
    if candidate_variant not in SUPPORTED_CANDIDATE_VARIANTS:
        raise ValueError(
            f"unsupported paired DLR variant {candidate_variant!r}; "
            f"choices={SUPPORTED_CANDIDATE_VARIANTS}"
        )
    return {
        "candidate_variant": candidate_variant,
        "base_model_variant": BASE_MODEL_VARIANTS[candidate_variant],
        "qfg_variant": v2.QFG_VARIANT,
        "tss_variant": TSS_VARIANTS[candidate_variant],
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": CANDIDATE_RECIPES[candidate_variant],
        "survival_weight_max": SURVIVAL_WEIGHT_MAXIMA[candidate_variant],
        "tss_control": candidate_variant == QFG_DLR_VARIANT,
        "formal_run_tag": FORMAL_RUN_TAGS[candidate_variant],
        "weight_schedule_id": (
            CONTROL_WEIGHT_SCHEDULE_ID
            if candidate_variant == QFG_DLR_VARIANT
            else TSS_WEIGHT_SCHEDULE_ID
        ),
    }


def tss_ramp_fraction(epoch: int) -> float:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError("TSS ramp epoch must be an integer")
    if epoch < 1 or epoch > FORMAL_EPOCHS:
        raise ValueError(
            f"TSS ramp epoch must be within [1, {FORMAL_EPOCHS}]"
        )
    return min(max(epoch - 1, 0), TSS_RAMP_DENOMINATOR) / (
        TSS_RAMP_DENOMINATOR
    )


def survival_weight_for_epoch(variant: str, epoch: int) -> float:
    candidate = candidate_contract(variant)
    if candidate["tss_control"]:
        tss_ramp_fraction(epoch)
        return 0.0
    return TSS_MAX_WEIGHT * tss_ramp_fraction(epoch)


def survival_schedule_contract(variant: str) -> dict[str, Any]:
    candidate = candidate_contract(variant)
    if candidate["tss_control"]:
        return {
            "schedule_id": CONTROL_WEIGHT_SCHEDULE_ID,
            "kind": "exact_zero",
            "formula": "0",
            "start_epoch": 1,
            "end_epoch": FORMAL_EPOCHS,
            "denominator": None,
            "maximum": 0.0,
            "ramp_fraction": "0",
        }
    return {
        "schedule_id": TSS_WEIGHT_SCHEDULE_ID,
        "kind": "inclusive_epoch_linear_ramp_then_hold",
        "formula": TSS_WEIGHT_FORMULA,
        "start_epoch": TSS_RAMP_START_EPOCH,
        "end_epoch": TSS_RAMP_END_EPOCH,
        "denominator": TSS_RAMP_DENOMINATOR,
        "maximum": TSS_MAX_WEIGHT,
        "ramp_fraction": "min(max(epoch-1,0),99)/99",
        "boundary_examples": {
            str(epoch): survival_weight_for_epoch(variant, epoch)
            for epoch in (1, 10, 50, 100, 800)
        },
    }


def optimizer_recipe_contract() -> dict[str, Any]:
    value = copy.deepcopy(dlr.optimizer_recipe_contract())
    value["training_recipe"] = FAMILY_RECIPE
    return value


def batchnorm_recipe_contract() -> dict[str, Any]:
    return copy.deepcopy(dlr.batchnorm_recipe_contract())


def formal_contract() -> dict[str, Any]:
    value = copy.deepcopy(v2.formal_contract())
    value.update(
        {
            "candidate_family": "v4_tail_aware_qfg_v2_croa_dlr_ramp100",
            "candidate_variants": list(SUPPORTED_CANDIDATE_VARIANTS),
            "base_model_variants": dict(BASE_MODEL_VARIANTS),
            "tss_variants": dict(TSS_VARIANTS),
            "survival_weight_maxima": dict(SURVIVAL_WEIGHT_MAXIMA),
            "family_recipe": FAMILY_RECIPE,
            "candidate_recipes": dict(CANDIDATE_RECIPES),
            "formal_run_tags": dict(FORMAL_RUN_TAGS),
            "tss_weight_schedules": {
                variant: survival_schedule_contract(variant)
                for variant in SUPPORTED_CANDIDATE_VARIANTS
            },
            "optimizer_recipe": optimizer_recipe_contract(),
            "batchnorm_recipe": batchnorm_recipe_contract(),
            "paired_epoch1_exact_zero_tss": True,
            "ramp_candidates_considered": [50, 100],
            "preselected_ramp_epochs": 100,
            "upstream_source_lock_path": str(
                UPSTREAM_SOURCE_LOCK_PATH.relative_to(REPO_ROOT)
            ),
            "upstream_source_lock_sha256": UPSTREAM_SOURCE_LOCK_SHA256,
        }
    )
    value.pop("survival_weights", None)
    return value


def _as_v2_args(args: argparse.Namespace) -> argparse.Namespace:
    candidate = candidate_contract(args.variant)
    base_variant = candidate["base_model_variant"]
    proxy = argparse.Namespace(**vars(args))
    proxy.variant = base_variant
    proxy.run_tag = v2.FORMAL_RUN_TAGS[base_variant]
    proxy.survival_weight = candidate["survival_weight_max"]
    proxy.qfg_variant = v2.QFG_VARIANT
    proxy.tss_variant = candidate["tss_variant"]
    return proxy


def _validate_formal_args(args: argparse.Namespace) -> None:
    candidate = candidate_contract(getattr(args, "variant", None))
    v2._validate_formal_args(_as_v2_args(args))
    expected = {
        "run_tag": candidate["formal_run_tag"],
        "survival_weight_max": candidate["survival_weight_max"],
        "qfg_variant": v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            "formal paired DLR arguments differ: "
            f"expected={expected}, observed={observed}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired exact800 QFG-DLR control/TSS-ramp100 training"
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
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=FORMAL_VAL_FRACTION,
    )
    parser.add_argument("--eval-every", type=int, default=FORMAL_EVAL_EVERY)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=FORMAL_WARMUP_EPOCHS,
    )
    parser.add_argument("--threshold", type=float, default=FORMAL_THRESHOLD)
    parser.add_argument(
        "--match-radius",
        type=float,
        default=FORMAL_MATCH_RADIUS,
    )
    parser.add_argument("--tiny-area", type=int, default=FORMAL_TINY_AREA)
    parser.add_argument("--eps", type=float, default=FORMAL_EPS)
    parser.set_defaults(amp=FORMAL_AMP)
    parser.add_argument("--survival-weight-max", type=float, default=None)
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
    )
    initialization.add_argument("--exact-resume", action="store_true")

    args = parser.parse_args(argv)
    candidate = candidate_contract(args.variant)
    if args.run_tag is None:
        args.run_tag = candidate["formal_run_tag"]
    if args.survival_weight_max is None:
        args.survival_weight_max = candidate["survival_weight_max"]
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    if args.survival_pos_weight is None:
        args.survival_pos_weight = statistics["survival_pos_weight"]
    args.survival_weight = candidate["survival_weight_max"]
    args.fresh = bool(args.parent_warm_start)
    args.parent_variant = PARENT_VARIANT
    args.relay_enabled = True
    args.relay_width = v2.v4_exact.RELAY_WIDTH
    args.relay_initialization_seed = (
        v2.v4_exact.RELAY_INITIALIZATION_SEED
    )
    args.dc_support_mode = v2.v4_exact.DC_SUPPORT_MODE
    args.tail_z_thresholds = dict(v2.v4_exact.TAIL_Z_THRESHOLDS)
    args.qfg_variant = v2.QFG_VARIANT
    args.tss_variant = candidate["tss_variant"]
    args.family_recipe = FAMILY_RECIPE
    args.candidate_recipe = candidate["candidate_recipe"]
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


def _adapt_model_metadata(
    variant: str,
    base_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(variant)
    validated = v2._require_qfg_metadata(
        base_metadata,
        variant=candidate["base_model_variant"],
    )
    manifest = copy.deepcopy(validated["architecture_manifest"])
    return {
        "schema": MODEL_METADATA_SCHEMA,
        "variant": variant,
        "candidate_variant": variant,
        "base_model_variant": candidate["base_model_variant"],
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "qfg_variant": v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "survival_weight_schedule": survival_schedule_contract(variant),
        "architecture_manifest": manifest,
        "architecture_id": canonical_sha256(manifest),
        "base_model_metadata": copy.deepcopy(validated),
        "optimizer_recipe": optimizer_recipe_contract(),
        "batchnorm_recipe": batchnorm_recipe_contract(),
    }


def _require_model_metadata(
    value: Any,
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("paired DLR model metadata is missing")
    metadata = _json_value(dict(value))
    candidate = candidate_contract(variant)
    expected = {
        "schema": MODEL_METADATA_SCHEMA,
        "variant": variant,
        "candidate_variant": variant,
        "base_model_variant": candidate["base_model_variant"],
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "qfg_variant": v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "survival_weight_schedule": survival_schedule_contract(variant),
        "optimizer_recipe": optimizer_recipe_contract(),
        "batchnorm_recipe": batchnorm_recipe_contract(),
    }
    for name, required in expected.items():
        _require_equal(
            f"paired DLR metadata {name}",
            metadata.get(name),
            required,
        )
    base_metadata = v2._require_qfg_metadata(
        metadata.get("base_model_metadata"),
        variant=candidate["base_model_variant"],
    )
    manifest = copy.deepcopy(base_metadata["architecture_manifest"])
    _require_equal(
        "paired DLR architecture manifest",
        metadata.get("architecture_manifest"),
        manifest,
    )
    _require_equal(
        "paired DLR architecture digest",
        metadata.get("architecture_id"),
        canonical_sha256(manifest),
    )
    return metadata


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    candidate = candidate_contract(variant)
    model, base_metadata = v2.build_selected_model(
        candidate["base_model_variant"],
        seed,
        eps=eps,
    )
    metadata = _adapt_model_metadata(variant, base_metadata)
    return model, _require_model_metadata(metadata, variant=variant)


def build_optimizer(model: nn.Module) -> torch.optim.Adam:
    return dlr.build_optimizer(model)


def optimizer_group_manifest(model: nn.Module) -> dict[str, Any]:
    return dlr.optimizer_group_manifest(model)


def count_batchnorm_modules(model: nn.Module) -> int:
    return dlr.count_batchnorm_modules(model)


def freeze_formal_batchnorm_running_stats(model: nn.Module) -> int:
    return dlr.freeze_formal_batchnorm_running_stats(model)


def _loss_contract(
    variant: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(variant)
    value = copy.deepcopy(
        v2._loss_contract(candidate["base_model_variant"], statistics)
    )
    survival = value["survival"]
    survival["survival_weight"] = "epoch_controlled"
    survival["survival_weight_max"] = candidate["survival_weight_max"]
    survival["survival_weight_schedule"] = (
        survival_schedule_contract(variant)
    )
    value.update(
        {
            "variant": variant,
            "candidate_variant": variant,
            "base_model_variant": candidate["base_model_variant"],
            "family_recipe": FAMILY_RECIPE,
            "candidate_recipe": candidate["candidate_recipe"],
            "total": (
                "segmentation + survival_weight_effective(epoch) "
                "* survival"
            ),
        }
    )
    return value


def _required_determinism(
    variant: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(variant)
    value = copy.deepcopy(
        v2._required_determinism(
            candidate["base_model_variant"],
            statistics,
        )
    )
    value.pop("qfg_run_identity_schema", None)
    value.update(
        {
            "entry_schema": ENTRY_SCHEMA,
            "paired_run_identity_schema": PAIRED_RUN_IDENTITY_SCHEMA,
            "candidate_variant": variant,
            "ordered_candidate_variants": list(
                SUPPORTED_CANDIDATE_VARIANTS
            ),
            "base_model_variant": candidate["base_model_variant"],
            "base_model_variants": dict(BASE_MODEL_VARIANTS),
            "tss_variant": candidate["tss_variant"],
            "tss_variants": dict(TSS_VARIANTS),
            "family_recipe": FAMILY_RECIPE,
            "candidate_recipe": candidate["candidate_recipe"],
            "candidate_recipes": dict(CANDIDATE_RECIPES),
            "source_lock_schema": EXACT_SOURCE_LOCK_SCHEMA,
            "upstream_source_lock_sha256": (
                UPSTREAM_SOURCE_LOCK_SHA256
            ),
            "survival_weight": "epoch_controlled",
            "survival_weight_max": candidate["survival_weight_max"],
            "survival_weight_schedule": (
                survival_schedule_contract(variant)
            ),
            "epoch1_exact_zero_tss": True,
            "effective_weight_recomputed_from_completed_epoch": True,
            "mutable_ramp_scheduler_state": False,
            "optimizer_recipe": optimizer_recipe_contract(),
            "batchnorm_recipe": batchnorm_recipe_contract(),
        }
    )
    value.update(scaled_runner.group_scaled_determinism_contract())
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
    metadata = _require_model_metadata(
        model_metadata,
        variant=args.variant,
    )
    optimizer_contract = dlr._require_optimizer_contract(model, optimizer)
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    expected_source_locks = {
        SOURCE_LOCK_KEY,
        UPSTREAM_SOURCE_LOCK_KEY,
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    }
    _require_equal(
        "paired DLR source-lock keys",
        set(source_locks),
        expected_source_locks,
    )
    _require_equal(
        "paired DLR upstream source lock",
        source_locks[UPSTREAM_SOURCE_LOCK_KEY],
        UPSTREAM_SOURCE_LOCK_SHA256,
    )
    _require_equal(
        "paired DLR target-statistics lock",
        source_locks["survival_target_statistics"],
        statistics["sha256"],
    )
    _require_equal(
        "paired DLR parent-checkpoint lock",
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
        optimizer=optimizer_contract,
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


def _require_optimizer_identity(
    optimizer: Any,
    *,
    label: str,
) -> None:
    if not isinstance(optimizer, Mapping):
        raise ValueError(f"{label} optimizer contract is missing")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list):
        raise ValueError(f"{label} optimizer groups are missing")
    _require_equal(
        f"{label} optimizer group count",
        len(groups),
        len(dlr.OPTIMIZER_GROUP_ORDER),
    )
    for index, (group, group_name) in enumerate(
        zip(groups, dlr.OPTIMIZER_GROUP_ORDER)
    ):
        if not isinstance(group, Mapping):
            raise ValueError(f"{label} optimizer group {index} is invalid")
        options = group.get("options")
        names = group.get("parameter_names")
        if not isinstance(options, Mapping) or not isinstance(names, list):
            raise ValueError(
                f"{label} optimizer group {index} contract is invalid"
            )
        _require_equal(
            f"{label} optimizer group {index} name",
            options.get(scaled_runner.GROUP_NAME_OPTION),
            group_name,
        )
        _require_equal(
            f"{label} optimizer group {index} multiplier",
            options.get(scaled_runner.SCHEDULE_MULTIPLIER_OPTION),
            dlr.OPTIMIZER_GROUP_MULTIPLIERS[group_name],
        )
        _require_equal(
            f"{label} optimizer group {index} initial LR",
            options.get("lr"),
            FORMAL_BASE_LR,
        )
        _require_equal(
            f"{label} optimizer group {index} tensor count",
            len(names),
            dlr.OPTIMIZER_GROUP_PARAMETER_TENSORS[group_name],
        )
        for parameter_name in names:
            _require_equal(
                f"{label} optimizer parameter {parameter_name}",
                dlr.optimizer_group_name(parameter_name),
                group_name,
            )


def require_paired_run_identity(
    identity: Any,
    *,
    label: str,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no paired DLR run identity")
    value = copy.deepcopy(dict(identity))
    variant = value.get("variant")
    candidate_contract(variant)
    if expected_variant is not None:
        _require_equal(f"{label} variant", variant, expected_variant)
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"{label} run_id is not ramp100-owned")
    _require_equal(
        f"{label} exact-runner schema",
        value.get("schema"),
        exact_runner.RUN_IDENTITY_SCHEMA,
    )
    _require_equal(f"{label} dataset", value.get("dataset"), "NUDT-SIRST")
    _require_equal(f"{label} seed", value.get("seed"), TRAINING_SEED)
    _require_equal(f"{label} split seed", value.get("split_seed"), SPLIT_SEED)
    v2._validate_sha256(
        value.get("builder_manifest_sha256"),
        f"{label} builder manifest SHA-256",
    )
    v2._validate_sha256(
        value.get("architecture_id"),
        f"{label} architecture SHA-256",
    )
    source_locks = value.get("source_locks")
    expected_lock_keys = {
        SOURCE_LOCK_KEY,
        UPSTREAM_SOURCE_LOCK_KEY,
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    }
    if not isinstance(source_locks, Mapping):
        raise ValueError(f"{label} source-lock identity is missing")
    _require_equal(
        f"{label} source-lock keys",
        set(source_locks),
        expected_lock_keys,
    )
    for name, digest in source_locks.items():
        v2._validate_sha256(digest, f"{label} source lock {name}")
    _require_equal(
        f"{label} upstream source lock",
        source_locks[UPSTREAM_SOURCE_LOCK_KEY],
        UPSTREAM_SOURCE_LOCK_SHA256,
    )
    statistics = load_survival_target_statistics()
    _require_equal(
        f"{label} target-statistics lock",
        source_locks["survival_target_statistics"],
        statistics["sha256"],
    )
    _require_equal(
        f"{label} parent-checkpoint lock",
        source_locks["parent_checkpoint"],
        PARENT_CHECKPOINT_SHA256,
    )
    training = value.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError(f"{label} training contract is missing")
    _require_equal(
        f"{label} loss contract",
        training.get("loss"),
        _loss_contract(variant, statistics),
    )
    _require_equal(
        f"{label} determinism contract",
        training.get("determinism"),
        _required_determinism(variant, statistics),
    )
    _require_equal(
        f"{label} LR schedule",
        training.get("manual_lr_schedule"),
        exact_runner.ManualCosineSchedule(
            total_epochs=FORMAL_EPOCHS,
            base_lr=FORMAL_BASE_LR,
            min_lr=FORMAL_MIN_LR,
            warmup_epochs=FORMAL_WARMUP_EPOCHS,
        ).normalized(),
    )
    _require_optimizer_identity(
        training.get("optimizer"),
        label=label,
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
    expected_provenance = {
        "schema": v2.EXTENSION_WARM_START_SCHEMA,
        "parent_checkpoint_path": str(PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_path": list(v2.PARENT_STATE_DICT_PATH),
        "parent_state_key_count": v2.FORMAL_PARENT_STATE_KEY_COUNT,
        "preserved_new_state_key_count": (
            len(v2.SURVIVAL_STATE_KEYS) + len(v2.QFG_STATE_KEYS)
        ),
        "new_module_prefixes": list(v2.QFG_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(v2.QFG_ZERO_INIT_PREFIXES),
    }
    _require_equal(
        f"{label} extension-parent provenance",
        provenance,
        expected_provenance,
    )
    return value


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
    target_statistics_path: Path = DEFAULT_TARGET_STATISTICS_PATH,
) -> dict[str, str]:
    path = Path(exact_source_lock_path).resolve()
    statistics = load_survival_target_statistics(target_statistics_path)
    payload = load_json_mapping(path, "paired DLR source lock")
    _require_equal(
        "paired DLR source-lock schema",
        payload.get("schema"),
        EXACT_SOURCE_LOCK_SCHEMA,
    )
    _require_equal(
        "paired DLR source-lock variants",
        tuple(payload.get("variants", ())),
        SUPPORTED_CANDIDATE_VARIANTS,
    )
    _require_equal(
        "paired DLR source-lock family recipe",
        payload.get("family_recipe"),
        FAMILY_RECIPE,
    )
    contract = formal_contract()
    _require_equal(
        "paired DLR source-lock formal contract",
        payload.get("formal_contract"),
        contract,
    )
    _require_equal(
        "paired DLR source-lock formal digest",
        payload.get("formal_contract_sha256"),
        canonical_sha256(contract),
    )
    _require_equal(
        "paired DLR source-lock training data",
        payload.get("training_data_sha256"),
        training_data_sha256,
    )
    _require_equal(
        "paired DLR source-lock target statistics",
        payload.get("survival_target_statistics_sha256"),
        statistics["sha256"],
    )
    _require_equal(
        "paired DLR source-lock parent checkpoint",
        payload.get("parent_checkpoint_sha256"),
        PARENT_CHECKPOINT_SHA256,
    )
    _require_equal(
        "paired DLR source-lock upstream path",
        payload.get("upstream_source_lock_path"),
        str(UPSTREAM_SOURCE_LOCK_PATH.relative_to(REPO_ROOT)),
    )
    _require_equal(
        "paired DLR source-lock upstream digest",
        payload.get("upstream_source_lock_sha256"),
        UPSTREAM_SOURCE_LOCK_SHA256,
    )
    _require_equal(
        "live upstream QFG source-lock digest",
        file_sha256(UPSTREAM_SOURCE_LOCK_PATH),
        UPSTREAM_SOURCE_LOCK_SHA256,
    )
    v2.source_lock_contract(
        training_data_sha256,
        UPSTREAM_SOURCE_LOCK_PATH,
        target_statistics_path,
    )
    locked = payload.get("source_sha256")
    if not isinstance(locked, Mapping):
        raise ValueError("paired DLR source lock has no source mapping")
    required = {
        str(runtime.relative_to(REPO_ROOT))
        for runtime in RUNTIME_SOURCE_PATHS
    }
    _require_equal(
        "paired DLR source-lock source count",
        payload.get("source_count"),
        len(locked),
    )
    _require_equal(
        "paired DLR locked runtime source set",
        set(locked),
        required,
    )
    for relative, expected in locked.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("paired DLR source lock has an invalid path")
        runtime = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(runtime.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError(
                "paired DLR source path escapes repository"
            ) from exc
        _require_equal(
            "paired DLR source canonical path",
            canonical,
            relative,
        )
        _require_equal(
            f"paired DLR source digest for {relative}",
            file_sha256(runtime),
            expected,
        )
    return {
        SOURCE_LOCK_KEY: file_sha256(path),
        UPSTREAM_SOURCE_LOCK_KEY: UPSTREAM_SOURCE_LOCK_SHA256,
        "training_data": v2._validate_sha256(
            training_data_sha256,
            "paired DLR training data SHA-256",
        ),
        "survival_target_statistics": statistics["sha256"],
        "parent_checkpoint": PARENT_CHECKPOINT_SHA256,
    }


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
        "existing paired DLR protocol",
    )
    _require_equal(
        "existing paired DLR protocol schema",
        protocol.get("schema"),
        ENTRY_SCHEMA,
    )
    identity = require_paired_run_identity(
        protocol.get("run_identity"),
        label="existing paired DLR protocol",
        expected_variant=args.variant,
    )
    training = identity.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError(
            "existing paired DLR protocol has no training contract"
        )
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(
            f"existing paired DLR training contract lacks fields: {missing}"
        )
    return copy.deepcopy(dict(training))


def initialization_plan(
    args: argparse.Namespace,
    directory: Path,
    model: nn.Module,
) -> InitializationPlan:
    _validate_formal_args(args)
    if args.parent_warm_start:
        plan = v2.initialization_plan(
            _as_v2_args(args),
            directory,
            model,
        )
        return InitializationPlan(
            request=plan.request,
            contract=copy.deepcopy(dict(plan.contract)),
            initial_model_state_sha256=plan.initial_model_state_sha256,
            initial_rng=copy.deepcopy(plan.initial_rng),
            selection_policy=copy.deepcopy(plan.selection_policy),
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
                "paired DLR exact resume lacks extension-parent "
                "initialization"
            )
        initial_rng = training["initial_rng"]
        selection_policy = training["selection_policy"]
        if not isinstance(initial_rng, Mapping):
            raise ValueError("existing paired DLR initial_rng is invalid")
        if not isinstance(selection_policy, Mapping):
            raise ValueError(
                "existing paired DLR selection_policy is invalid"
            )
        return InitializationPlan(
            request=exact_runner.InitializationRequest.exact(),
            contract=copy.deepcopy(dict(initialization)),
            initial_model_state_sha256=str(
                training["initial_model_state_sha256"]
            ),
            initial_rng=copy.deepcopy(dict(initial_rng)),
            selection_policy=copy.deepcopy(dict(selection_policy)),
        )
    raise RuntimeError(
        "paired DLR exact entry requires warm-start or exact resume"
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    return v2.resolve_device(_as_v2_args(args))


def environment_contract(device: torch.device) -> dict[str, Any]:
    payload = v2.environment_contract(device)
    payload["family_recipe"] = FAMILY_RECIPE
    return payload


def candidate_ramp_fraction(variant: str, epoch: int) -> float:
    candidate = candidate_contract(variant)
    if candidate["tss_control"]:
        tss_ramp_fraction(epoch)
        return 0.0
    return tss_ramp_fraction(epoch)


def _finite_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_epoch_loss_fields(
    variant: str,
    epoch: int,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(fields, Mapping):
        raise ValueError("paired DLR epoch fields must be a mapping")
    weight = survival_weight_for_epoch(variant, epoch)
    fraction = candidate_ramp_fraction(variant, epoch)
    segmentation = _finite_float(
        fields.get("train_segmentation_loss"),
        "train_segmentation_loss",
    )
    total = _finite_float(
        fields.get("train_total_loss"),
        "train_total_loss",
    )
    survival = _finite_float(
        fields.get("train_survival_loss"),
        "train_survival_loss",
    )
    emb1 = fields.get("train_survival_emb1_loss")
    emb2 = fields.get("train_survival_emb2_loss")
    if weight == 0.0:
        _require_equal("zero-path survival loss", survival, 0.0)
        _require_equal("zero-path emb1 survival loss", emb1, None)
        _require_equal("zero-path emb2 survival loss", emb2, None)
    else:
        emb1_value = _finite_float(emb1, "train_survival_emb1_loss")
        emb2_value = _finite_float(emb2, "train_survival_emb2_loss")
        if not math.isclose(
            survival,
            emb1_value + emb2_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "survival loss differs from emb1+emb2 losses"
            )
    weighted = weight * survival
    if not math.isclose(
        total,
        segmentation + weighted,
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise ValueError(
            "total loss differs from segmentation+weighted survival"
        )
    candidate = candidate_contract(variant)
    return {
        "survival_weight": weight,
        SURVIVAL_WEIGHT_FIELD: weight,
        SURVIVAL_WEIGHT_MAX_FIELD: candidate["survival_weight_max"],
        TSS_RAMP_FRACTION_FIELD: fraction,
        TSS_WEIGHT_SCHEDULE_FIELD: candidate["weight_schedule_id"],
        WEIGHTED_SURVIVAL_LOSS_FIELD: weighted,
        SURVIVAL_ENABLED_FIELD: weight > 0.0,
    }


@dataclass(frozen=True)
class RampEpochControl:
    epoch: int
    learning_rate: float
    should_evaluate: bool
    survival_weight_effective: float
    tss_ramp_fraction: float


class PairedRamp100ExactRunner(scaled_runner.GroupScaledExactRunner):
    """Own both group-LR and epoch-derived survival-weight controls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        candidate_contract(self.spec.variant)
        self._open_ramp_control: RampEpochControl | None = None

    def startup(
        self,
        request: exact_runner.InitializationRequest,
    ) -> exact_runner.RunnerSnapshot:
        if not isinstance(request, exact_runner.InitializationRequest):
            return super().startup(request)
        request.validate()
        if (
            exact_resume.InitializationMode(request.mode)
            is exact_resume.InitializationMode.EXACT_RESUME
        ):
            active = self.journal.load_active()
            if active is None:
                raise ValueError(
                    "paired DLR exact resume requires an active journal"
                )
            payload, _ = self._load_exact_payload(active.checkpoint_path)
            require_paired_run_identity(
                payload.get("run_identity"),
                label="active paired DLR journal",
                expected_variant=self.spec.variant,
            )
            if not isinstance(payload.get("optimizer"), Mapping):
                raise ValueError(
                    "active paired DLR journal has no optimizer state"
                )
        return super().startup(request)

    def next_epoch_control(self) -> RampEpochControl:
        if self._open_ramp_control is not None:
            raise exact_runner.ExactRunnerError(
                "paired DLR epoch is already open"
            )
        base_control = super().next_epoch_control()
        control = RampEpochControl(
            epoch=base_control.epoch,
            learning_rate=base_control.learning_rate,
            should_evaluate=base_control.should_evaluate,
            survival_weight_effective=survival_weight_for_epoch(
                self.spec.variant,
                base_control.epoch,
            ),
            tss_ramp_fraction=candidate_ramp_fraction(
                self.spec.variant,
                base_control.epoch,
            ),
        )
        self._open_ramp_control = control
        return control

    def commit_epoch(
        self,
        fields: Mapping[str, Any],
        *,
        extra_state: Mapping[str, Any] | None = None,
    ) -> exact_runner.RunnerSnapshot:
        if self._open_ramp_control is None:
            raise exact_runner.ExactRunnerError(
                "next_epoch_control must be called before commit_epoch"
            )
        forged = sorted(_RAMP_RUNNER_OWNED_FIELDS & set(fields))
        if forged:
            raise exact_runner.ExactRunnerError(
                "epoch fields contain ramp-runner-owned keys: "
                f"{forged}"
            )
        control = self._open_ramp_control
        evidence = validate_epoch_loss_fields(
            self.spec.variant,
            control.epoch,
            fields,
        )
        annotated = dict(fields)
        annotated.update(evidence)
        try:
            snapshot = super().commit_epoch(
                annotated,
                extra_state=extra_state,
            )
        except BaseException:
            if self._open_control is None:
                self._open_ramp_control = None
            raise
        self._open_ramp_control = None
        return snapshot


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
        "survival_weight_max",
        "survival_pos_weight",
        "max_train_images",
        "max_val_images",
        "qfg_variant",
        "tss_variant",
        "family_recipe",
        "candidate_recipe",
    )
    payload = {name: getattr(args, name) for name in names}
    payload["survival_weight_schedule"] = survival_schedule_contract(
        args.variant
    )
    payload["optimizer_recipe"] = optimizer_recipe_contract()
    payload["batchnorm_recipe"] = batchnorm_recipe_contract()
    return payload


def protocol_payload(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    normalization: Mapping[str, float],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = require_paired_run_identity(
        run_identity,
        label="paired DLR protocol",
        expected_variant=args.variant,
    )
    candidate = candidate_contract(args.variant)
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    return {
        "schema": ENTRY_SCHEMA,
        "formal_contract": formal_contract(),
        "arguments": training_arguments(args),
        "run_directory": directory,
        "model": _require_model_metadata(
            model_metadata,
            variant=args.variant,
        ),
        "architecture_manifest": copy.deepcopy(
            model_metadata["architecture_manifest"]
        ),
        "normalization": dict(normalization),
        "run_identity": identity,
        "candidate_variant": args.variant,
        "base_model_variant": candidate["base_model_variant"],
        "qfg_variant": v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "survival_weight_schedule": survival_schedule_contract(
            args.variant
        ),
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
        "optimizer_recipe": optimizer_recipe_contract(),
        "batchnorm_recipe": batchnorm_recipe_contract(),
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
            "initial_child_completed_epoch": 0,
            "same_candidate_epoch_boundary_only": True,
            "cross_candidate": "forbidden",
            "cross_recipe": "forbidden",
            "cross_ramp": "forbidden",
            "cross_version": "forbidden",
            "optimizer_inherited_from_parent": False,
            "scheduler_restore": (
                "manual_base_lr_then_group_multipliers_and_"
                "epoch_derived_tss_weight"
            ),
            "mutable_ramp_scheduler_state": False,
        },
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }


def _checkpoint_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    variant = str(identity["variant"])
    candidate = candidate_contract(variant)
    return {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "variant": variant,
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "run_id": identity["run_id"],
        "architecture_id": identity["architecture_id"],
        "builder_manifest_sha256": identity[
            "builder_manifest_sha256"
        ],
    }


@dataclass(frozen=True)
class EvaluatorCheckpointAdapter:
    model_metadata: Mapping[str, Any]
    split_hashes: Mapping[str, str]

    def __call__(
        self,
        context: exact_runner.CompatibilityPayloadContext,
    ) -> Mapping[str, Any]:
        identity = require_paired_run_identity(
            context.run_identity,
            label="paired DLR checkpoint context",
        )
        variant = str(identity["variant"])
        candidate = candidate_contract(variant)
        metadata = _require_model_metadata(
            self.model_metadata,
            variant=variant,
        )
        epoch = int(context.epoch)
        weight = survival_weight_for_epoch(variant, epoch)
        fraction = candidate_ramp_fraction(variant, epoch)
        exact_payload = context.exact_payload
        source_locks = copy.deepcopy(dict(identity["source_locks"]))
        return {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "checkpoint_role": context.role,
            "variant": variant,
            "candidate_variant": variant,
            "base_model_variant": candidate["base_model_variant"],
            "qfg_variant": v2.QFG_VARIANT,
            "tss_variant": candidate["tss_variant"],
            "family_recipe": FAMILY_RECIPE,
            "candidate_recipe": candidate["candidate_recipe"],
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
                "kind": "identity_bound_manual_group_scaled_schedule",
                "completed_epoch": epoch,
                "checkpoint_group_lr": (
                    "manual_cosine_lr(completed_epoch)"
                ),
                "next_epoch_reapplies_multipliers": True,
                "tss_weight_state": "derived_from_epoch_not_serialized",
            },
            "validation_metrics": copy.deepcopy(dict(context.metrics)),
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
            "upstream_source_lock_sha256": source_locks[
                UPSTREAM_SOURCE_LOCK_KEY
            ],
            "parent_checkpoint_path": str(
                PARENT_CHECKPOINT_PATH.resolve()
            ),
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
            "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
            "parent_checkpoint_state_dict_sha256": (
                PARENT_STATE_DICT_SHA256
            ),
            "initial_model_state_sha256": identity[
                "training_contract"
            ]["initial_model_state_sha256"],
            "survival_weight_schedule": (
                survival_schedule_contract(variant)
            ),
            TSS_WEIGHT_SCHEDULE_FIELD: candidate["weight_schedule_id"],
            SURVIVAL_WEIGHT_MAX_FIELD: candidate[
                "survival_weight_max"
            ],
            SURVIVAL_WEIGHT_FIELD: weight,
            TSS_RAMP_FRACTION_FIELD: fraction,
            "optimizer_recipe": optimizer_recipe_contract(),
            "batchnorm_recipe": batchnorm_recipe_contract(),
            "selection_uses_survival_loss": False,
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }


def validate_epoch_event(
    event: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise ValueError("paired DLR event must be a mapping")
    value = copy.deepcopy(dict(event))
    candidate = candidate_contract(variant)
    epoch = value.get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or epoch > FORMAL_EPOCHS
    ):
        raise ValueError("paired DLR event epoch is invalid")
    required_identity = {
        "variant": variant,
        "candidate_variant": variant,
        "base_model_variant": candidate["base_model_variant"],
        "qfg_variant": v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
    }
    for name, expected in required_identity.items():
        _require_equal(
            f"paired DLR epoch {epoch} {name}",
            value.get(name),
            expected,
        )
    v2._require_complete_validation_metrics(value)
    base_lr = exact_runner.ManualCosineSchedule(
        total_epochs=FORMAL_EPOCHS,
        base_lr=FORMAL_BASE_LR,
        min_lr=FORMAL_MIN_LR,
        warmup_epochs=FORMAL_WARMUP_EPOCHS,
    ).learning_rate(epoch)
    _require_equal(
        f"paired DLR epoch {epoch} base LR",
        value.get("learning_rate"),
        base_lr,
    )
    _require_equal(
        f"paired DLR epoch {epoch} optimizer group names",
        value.get(scaled_runner.GROUP_NAMES_EVENT_FIELD),
        list(dlr.OPTIMIZER_GROUP_ORDER),
    )
    multipliers = [
        dlr.OPTIMIZER_GROUP_MULTIPLIERS[name]
        for name in dlr.OPTIMIZER_GROUP_ORDER
    ]
    _require_equal(
        f"paired DLR epoch {epoch} multipliers",
        value.get(scaled_runner.SCHEDULE_MULTIPLIERS_EVENT_FIELD),
        multipliers,
    )
    _require_equal(
        f"paired DLR epoch {epoch} group LRs",
        value.get(scaled_runner.GROUP_LEARNING_RATES_EVENT_FIELD),
        [base_lr * multiplier for multiplier in multipliers],
    )
    _require_equal(
        f"paired DLR epoch {epoch} BatchNorm count",
        value.get(dlr.BATCHNORM_EVENT_FIELD),
        dlr.FORMAL_BATCHNORM_MODULE_COUNT,
    )
    expected_loss_evidence = validate_epoch_loss_fields(
        variant,
        epoch,
        value,
    )
    for name, expected in expected_loss_evidence.items():
        _require_equal(
            f"paired DLR epoch {epoch} {name}",
            value.get(name),
            expected,
        )
    return value


def _load_complete_events(
    path: Path,
    epochs: int,
    *,
    variant: str,
) -> list[dict[str, Any]]:
    events = v2._load_complete_events(path, epochs)
    return [
        validate_epoch_event(event, variant=variant)
        for event in events
    ]


def _load_checkpoint(
    path: Path,
    *,
    role: str,
    epoch: int,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"paired DLR checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"paired DLR checkpoint is invalid: {path}")
    value = dict(payload)
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "checkpoint_role": role,
        "variant": identity["variant"],
        "run_identity": dict(identity),
    }
    for name, required in expected.items():
        _require_equal(
            f"paired DLR checkpoint {path.name} {name}",
            value.get(name),
            required,
        )
    state = value.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(
            f"paired DLR checkpoint has no state_dict: {path}"
        )
    _require_equal(
        f"paired DLR checkpoint {path.name} state digest",
        value.get("state_dict_sha256"),
        exact_runner._state_content_sha256(
            state,
            "paired DLR checkpoint state_dict",
        ),
    )
    return value, file_sha256(path)


def _qfg_state_audit(state: Mapping[str, Any]) -> dict[str, Any]:
    terminal_nonzero = {}
    for name in v2.QFG_TERMINAL_STATE_KEYS:
        tensor = state[name]
        terminal_nonzero[name] = int(torch.count_nonzero(tensor))
    effective_alpha = {}
    for level in range(4):
        name = f"tpd_qfg.levels.{level}.alpha"
        effective_alpha[str(level)] = float(
            torch.tanh(state[name].detach().cpu())
        )
    return {
        "terminal_nonzero_counts": terminal_nonzero,
        "any_terminal_nonzero": any(terminal_nonzero.values()),
        "effective_alpha": effective_alpha,
    }


def _require_zero_tss_state(
    state: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for name in v2.SURVIVAL_STATE_KEYS:
        tensor = state.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"{label} lacks TSS tensor {name}")
        if int(torch.count_nonzero(tensor)) != 0:
            raise RuntimeError(f"{label} TSS tensor {name} is nonzero")


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
        variant=args.variant,
    )
    selection_policy = exact_runner.pd_miou_selection_policy(
        stored_metrics=STORED_VALIDATION_METRICS
    )
    recomputed_selection = selection_policy.recompute(events)
    _require_equal(
        "paired DLR recomputed checkpoint selection",
        selection,
        recomputed_selection,
    )
    pd_epoch = int(selection["primary"]["epoch"])
    miou_epoch = int(selection["secondary"]["epoch"])
    protocol = load_json_mapping(
        directory / "protocol.json",
        "completed paired DLR protocol",
    )
    identity = require_paired_run_identity(
        protocol.get("run_identity"),
        label="completed paired DLR protocol",
        expected_variant=args.variant,
    )
    live_sources = source_lock_contract(
        identity["source_locks"]["training_data"],
        args.exact_source_lock,
        args.survival_target_statistics,
    )
    _require_equal(
        "paired DLR completion source locks",
        live_sources,
        identity["source_locks"],
    )
    checkpoints = {}
    for name, filename, role, epoch in (
        (
            "best",
            exact_runner.BEST_FILENAME,
            "best_validation_pd_primary",
            pd_epoch,
        ),
        (
            "best_miou",
            exact_runner.BEST_MIOU_FILENAME,
            "best_validation_miou_secondary",
            miou_epoch,
        ),
        (
            "last",
            exact_runner.LAST_FILENAME,
            "last_evaluated_epoch",
            FORMAL_EPOCHS,
        ),
    ):
        payload, digest = _load_checkpoint(
            directory / filename,
            role=role,
            epoch=epoch,
            identity=identity,
        )
        checkpoints[name] = {
            "path": directory / filename,
            "sha256": digest,
            "epoch": epoch,
            "role": role,
        }
        if args.variant == QFG_DLR_VARIANT:
            _require_zero_tss_state(
                payload["state_dict"],
                label=f"control {name} checkpoint",
            )
    best_payload = torch.load(
        directory / exact_runner.BEST_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    candidate = candidate_contract(args.variant)
    return {
        "schema": COMPLETION_SUMMARY_SCHEMA,
        "status": "complete",
        "variant": args.variant,
        "candidate_variant": args.variant,
        "base_model_variant": candidate["base_model_variant"],
        "qfg_variant": v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "family_recipe": FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "formal_contract": formal_contract(),
        "run_identity": identity,
        "source_locks": copy.deepcopy(dict(identity["source_locks"])),
        "best_epoch": pd_epoch,
        "best_validation_metrics": v2._require_complete_validation_metrics(
            events[pd_epoch - 1]
        ),
        "best_pd_epoch": pd_epoch,
        "best_pd_validation_metrics": (
            v2._require_complete_validation_metrics(events[pd_epoch - 1])
        ),
        "best_pd_survival_weight_effective": (
            survival_weight_for_epoch(args.variant, pd_epoch)
        ),
        "best_pd_tss_ramp_fraction": candidate_ramp_fraction(
            args.variant,
            pd_epoch,
        ),
        "best_miou_epoch": miou_epoch,
        "best_miou_validation_metrics": (
            v2._require_complete_validation_metrics(events[miou_epoch - 1])
        ),
        "best_miou_survival_weight_effective": (
            survival_weight_for_epoch(args.variant, miou_epoch)
        ),
        "best_miou_tss_ramp_fraction": candidate_ramp_fraction(
            args.variant,
            miou_epoch,
        ),
        "model": _require_model_metadata(
            model_metadata,
            variant=args.variant,
        ),
        "split_hashes": dict(split_hashes),
        "survival_weight_schedule": survival_schedule_contract(
            args.variant
        ),
        "optimizer_recipe": optimizer_recipe_contract(),
        "batchnorm_recipe": batchnorm_recipe_contract(),
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
        "checkpoints": checkpoints,
        "qfg_gate_audit": _qfg_state_audit(
            best_payload["state_dict"]
        ),
        "control_tss_all_zero_verified": (
            args.variant == QFG_DLR_VARIANT
        ),
    }


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
    v2._require_prepared_statistics_match(statistics, prepared)
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
    _require_equal(
        "formal paired DLR BatchNorm topology",
        count_batchnorm_modules(model),
        dlr.FORMAL_BATCHNORM_MODULE_COUNT,
    )
    train_set = v2.base.TrainingSubset(
        prepared.dataset_dir,
        args.dataset,
        args.patch_size,
        prepared.train_ids,
        prepared.normalization,
    )
    val_set = v2.base.ValidationSubset(
        prepared.dataset_root,
        prepared.val_ids,
        prepared.normalization,
    )

    v2.base.seed_everything(args.seed)
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
    optimizer = build_optimizer(model)
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
    runner = PairedRamp100ExactRunner(
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
    v2.write_or_verify_json(directory / "split.json", prepared.split_manifest)
    v2.write_or_verify_json(
        directory / "protocol.json",
        protocol_payload(
            args,
            directory=directory,
            model_metadata=model_metadata,
            normalization=prepared.normalization,
            run_identity=snapshot.run_identity,
        ),
    )

    candidate = candidate_contract(args.variant)
    print(
        f"START variant={args.variant} "
        f"base={candidate['base_model_variant']} "
        f"recipe={candidate['candidate_recipe']} "
        f"mode={snapshot.initialization_mode.value} "
        f"completed={snapshot.completed_epoch} "
        f"next={snapshot.next_epoch} device={device}",
        flush=True,
    )
    while snapshot.next_epoch is not None:
        control = runner.next_epoch_control()
        if not control.should_evaluate:
            raise RuntimeError(
                "formal paired DLR training must evaluate each epoch"
            )
        epoch_started = time.time()
        effective_weight = control.survival_weight_effective
        model.train()
        batchnorm_count = freeze_formal_batchnorm_running_stats(model)
        accumulator = EpochLossAccumulator(
            survival_enabled=effective_weight > 0.0
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
                    survival_weight=effective_weight,
                    survival_pos_weight=args.survival_pos_weight,
                )
            scaler.scale(losses.total).backward()
            scaler.step(optimizer)
            scaler.update()
            accumulator.update(losses, int(images.shape[0]))
        loss_fields = accumulator.fields()

        metrics = v2.base.validate(
            model,
            val_loader,
            device,
            criterion,
            args.threshold,
            args.match_radius,
            args.tiny_area,
            FORMAL_AMP,
        )
        v2._check_metrics(metrics)
        snapshot = runner.commit_epoch(
            {
                "variant": args.variant,
                "candidate_variant": args.variant,
                "base_model_variant": candidate["base_model_variant"],
                "qfg_variant": v2.QFG_VARIANT,
                "tss_variant": candidate["tss_variant"],
                "family_recipe": FAMILY_RECIPE,
                "candidate_recipe": candidate["candidate_recipe"],
                **loss_fields,
                "train_loss": loss_fields["train_total_loss"],
                "survival_pos_weight": args.survival_pos_weight,
                dlr.BATCHNORM_EVENT_FIELD: batchnorm_count,
                "processed_train_samples": accumulator.sample_count,
                "epoch_seconds": time.time() - epoch_started,
                "skipped_singleton_batches": skipped_singletons,
                **metrics,
            },
            extra_state={
                "variant": args.variant,
                "base_model_variant": candidate["base_model_variant"],
                "qfg_variant": v2.QFG_VARIANT,
                "tss_variant": candidate["tss_variant"],
                "family_recipe": FAMILY_RECIPE,
                "candidate_recipe": candidate["candidate_recipe"],
                "formal_eps": FORMAL_EPS,
                TSS_WEIGHT_SCHEDULE_FIELD: candidate[
                    "weight_schedule_id"
                ],
                SURVIVAL_WEIGHT_MAX_FIELD: candidate[
                    "survival_weight_max"
                ],
                SURVIVAL_WEIGHT_FIELD: effective_weight,
                TSS_RAMP_FRACTION_FIELD: control.tss_ramp_fraction,
                "survival_pos_weight": args.survival_pos_weight,
                "survival_target_statistics_sha256": statistics["sha256"],
                dlr.BATCHNORM_EVENT_FIELD: batchnorm_count,
                "processed_train_samples": accumulator.sample_count,
                "skipped_singleton_batches": skipped_singletons,
            },
        )
        print(
            f"EPOCH {control.epoch:03d}/{FORMAL_EPOCHS} "
            f"wTSS={effective_weight:.8f} "
            f"total={float(loss_fields['train_total_loss']):.6f} "
            f"seg={float(loss_fields['train_segmentation_loss']):.6f} "
            f"surv={float(loss_fields['train_survival_loss']):.6f} "
            f"mIoU={float(metrics['miou']):.6f} "
            f"Pd={float(metrics['pd']):.6f} "
            f"Fa={float(metrics['fa']):.8f}",
            flush=True,
        )

    if snapshot.best_selection is None:
        raise RuntimeError(
            "completed paired DLR run has no best selection"
        )
    summary = completion_summary(
        args,
        directory=directory,
        model_metadata=model_metadata,
        split_hashes=prepared.split_hashes,
        selection=snapshot.best_selection,
    )
    v2.write_or_verify_json(directory / "summary.json", summary)
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
    "BATCHNORM_EVENT_FIELD",
    "CHECKPOINT_IDENTITY_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "COMPLETION_SUMMARY_SCHEMA",
    "CONTROL_RECIPE",
    "CONTROL_WEIGHT_SCHEDULE_ID",
    "CANDIDATE_RECIPES",
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "ENTRY_SCHEMA",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EvaluatorCheckpointAdapter",
    "FAMILY_RECIPE",
    "FORMAL_BASE_LR",
    "FORMAL_EPOCHS",
    "FORMAL_RUN_TAGS",
    "MODEL_METADATA_SCHEMA",
    "PAIRED_RUN_IDENTITY_SCHEMA",
    "PairedRamp100ExactRunner",
    "QFG_DLR_VARIANT",
    "RUNTIME_SOURCE_PATHS",
    "RampEpochControl",
    "SOURCE_LOCK_KEY",
    "SUPPORTED_CANDIDATE_VARIANTS",
    "SURVIVAL_ENABLED_FIELD",
    "SURVIVAL_WEIGHT_FIELD",
    "SURVIVAL_WEIGHT_MAX_FIELD",
    "TREATMENT_RECIPE",
    "TSS_MAX_WEIGHT",
    "TSS_QFG_DLR_VARIANT",
    "TSS_RAMP_DENOMINATOR",
    "TSS_RAMP_END_EPOCH",
    "TSS_RAMP_FRACTION_FIELD",
    "TSS_RAMP_START_EPOCH",
    "TSS_WEIGHT_FORMULA",
    "TSS_WEIGHT_SCHEDULE_FIELD",
    "TSS_WEIGHT_SCHEDULE_ID",
    "UPSTREAM_SOURCE_LOCK_KEY",
    "UPSTREAM_SOURCE_LOCK_PATH",
    "UPSTREAM_SOURCE_LOCK_SHA256",
    "WEIGHTED_SURVIVAL_LOSS_FIELD",
    "batchnorm_recipe_contract",
    "build_optimizer",
    "build_selected_model",
    "candidate_contract",
    "candidate_ramp_fraction",
    "completion_summary",
    "environment_contract",
    "formal_contract",
    "freeze_formal_batchnorm_running_stats",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "optimizer_group_manifest",
    "optimizer_recipe_contract",
    "parse_args",
    "protocol_payload",
    "require_paired_run_identity",
    "resolve_device",
    "run_directory",
    "run_training",
    "source_lock_contract",
    "supported_candidate_variants",
    "survival_schedule_contract",
    "survival_weight_for_epoch",
    "training_arguments",
    "tss_ramp_fraction",
    "validate_epoch_event",
    "validate_epoch_loss_fields",
]


if __name__ == "__main__":
    main()
