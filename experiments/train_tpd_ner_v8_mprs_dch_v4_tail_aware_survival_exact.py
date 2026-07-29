#!/usr/bin/env python3
"""Exact V4 target-survival continued-training entry.

Both formal variants start a new 800-epoch trajectory from the same frozen
V4 ``best_miou`` checkpoint.  ``tss_control`` keeps the survival objective
disabled while retaining the identical extension model and optimizer layout;
``tss_on`` changes only the survival loss weight from 0 to 0.005.  Exact
resume is accepted only for the same variant and immutable loss/provenance
identity.

Importing this module is read-only.  Training starts only through
``run_training``/``main``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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
from experiments.tpd_extension_warm_start import (  # noqa: E402
    PROVENANCE_SCHEMA as EXTENSION_WARM_START_SCHEMA,
    load_parent_into_extension,
)
from experiments.tpd_training_loss import (  # noqa: E402
    SURVIVAL_DOWNSAMPLE,
    TPDTrainingLoss,
    compute_tpd_training_loss,
)
from model import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v4_tail_aware_survival as survival_model,
)


ENTRY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_exact_entry_v1"
)
RUN_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_exact_run_identity_v1"
)
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_survival_exact_source_lock_v1"
)
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_"
    "exact_architecture_manifest_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_"
    "exact_checkpoint_identity_v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_exact_checkpoint_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_"
    "completion_summary_v1"
)
TARGET_STATISTICS_SCHEMA = (
    "sctransnet_tpd_survival_target_statistics_v1"
)
SOURCE_LOCK_KEY = "tpd_ner_v4_survival_exact_source_lock"
RUN_ID_PREFIX = "tpd-ner-v8-mprs-dch-v4-survival-exact:"

TSS_CONTROL_VARIANT = "tss_control"
TSS_ON_VARIANT = "tss_on"
SURVIVAL_CONTROL_VARIANT = TSS_CONTROL_VARIANT
SURVIVAL_ON_VARIANT = TSS_ON_VARIANT
SUPPORTED_CANDIDATE_VARIANTS = (
    TSS_CONTROL_VARIANT,
    TSS_ON_VARIANT,
)
FALLBACK_CANDIDATE_VARIANTS = SUPPORTED_CANDIDATE_VARIANTS

FORMAL_CONTROL_RUN_TAG = "formal800_control"
FORMAL_TSS_RUN_TAG = "formal800_tss"
FORMAL_RUN_TAGS = {
    TSS_CONTROL_VARIANT: FORMAL_CONTROL_RUN_TAG,
    TSS_ON_VARIANT: FORMAL_TSS_RUN_TAG,
}
FORMAL_SURVIVAL_WEIGHTS = {
    TSS_CONTROL_VARIANT: 0.0,
    TSS_ON_VARIANT: 0.005,
}

TRAINING_SEED = 42
SPLIT_SEED = 20260722
FORMAL_EPOCHS = 800
FORMAL_BATCH_SIZE = 16
FORMAL_PATCH_SIZE = 256
FORMAL_WORKERS = 0
FORMAL_VAL_FRACTION = 0.20
FORMAL_EVAL_EVERY = 1
FORMAL_BASE_LR = 1e-4
FORMAL_MIN_LR = 1e-6
FORMAL_WARMUP_EPOCHS = 10
FORMAL_THRESHOLD = 0.5
FORMAL_MATCH_RADIUS = 3.0
FORMAL_TINY_AREA = 9
FORMAL_AMP = False
FORMAL_EPS = v4_exact.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = v4_exact.FORMAL_CUBLAS_WORKSPACE_CONFIG
FORMAL_INITIALIZATION_MODES = (
    exact_runner.EXTENSION_PARENT_MODE,
    "exact_resume",
)

PARENT_VARIANT = (
    v4_exact.TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON
)
PARENT_CHECKPOINT_EPOCH = 489
PARENT_CHECKPOINT_ROLE = "best_validation_miou_secondary"
PARENT_CHECKPOINT_ROLE_SHORT = "best_miou"
PARENT_CHECKPOINT_SHA256 = (
    "0ae6c0e034952e18333d8fa6ccd3bbf635cae5efa8017b06df5e00ccc4ed14ab"
)
PARENT_STATE_DICT_SHA256 = (
    "2b8249ffd86866597f376c80839395a3cbdbb72a68301cd8a5a6eb36595c7e75"
)
PARENT_STATE_DICT_PATH = ("state_dict",)
PARENT_CHECKPOINT_PATH = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/"
    "NUDT-SIRST/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/"
    "seed_42_formal800_exact_v4_tail_aware_seed42/"
    "best_miou.pth.tar"
)

SURVIVAL_VERSION = survival_model.SURVIVAL_VERSION
SURVIVAL_STATE_PREFIX = survival_model.SURVIVAL_STATE_PREFIX
SURVIVAL_STATE_KEYS = tuple(survival_model.SURVIVAL_STATE_KEYS)
PRODUCTION_SURVIVAL_PARAMETERS = (
    survival_model.PRODUCTION_SURVIVAL_PARAMETERS
)
PRODUCTION_TOTAL_PARAMETERS = (
    survival_model.PRODUCTION_V4_SURVIVAL_PARAMETERS
)
FORMAL_PARENT_STATE_KEY_COUNT = (
    survival_model.FORMAL_V4_PARENT_STATE_KEY_COUNT
)
FORMAL_STATE_KEY_COUNT = (
    survival_model.FORMAL_V4_SURVIVAL_STATE_KEY_COUNT
)
SURVIVAL_NEW_MODULE_PREFIXES = ("target_survival",)
SURVIVAL_ZERO_INIT_PREFIXES = (
    "target_survival.heads.emb1.classifier",
    "target_survival.heads.emb2.classifier",
)

DEFAULT_TARGET_STATISTICS_PATH = (
    REPO_ROOT
    / "experiments/tpd_survival_target_statistics_nudt_sirst_v1.json"
)
DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT / "experiments/tpd_ner_v4_survival_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v4_survival_exact_v1"
)
PROTOCOL_DRAFT_PATH = (
    REPO_ROOT
    / "experiments/TPD_NER_V8_MPRS_DCH_V4_SURVIVAL_PROTOCOL.md"
)

SELECTION_METRICS = v4_exact.SELECTION_METRICS
STORED_VALIDATION_METRICS = v4_exact.STORED_VALIDATION_METRICS

file_sha256 = v4_exact.file_sha256
canonical_sha256 = v4_exact.canonical_sha256
load_json_mapping = v4_exact.load_json_mapping
PreparedData = v4_exact.PreparedData
prepare_data = v4_exact.prepare_data
split_fingerprints = v4_exact.split_fingerprints
data_fingerprints = v4_exact.data_fingerprints
configure_determinism = v4_exact.configure_determinism
write_or_verify_json = v4_exact.write_or_verify_json
shared_exact = v4_exact.shared_exact

PHYSICAL_GPU_UUIDS = dict(v4_exact.PHYSICAL_GPU_UUIDS)


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
        / "model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py",
        REPO_ROOT / "model/tpd_survival.py",
        REPO_ROOT / "model/tpd_forward_contract.py",
        REPO_ROOT / "experiments/tpd_training_loss.py",
        REPO_ROOT / "experiments/tpd_extension_warm_start.py",
        *v4_exact.RUNTIME_SOURCE_PATHS,
    )
)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _json_value(value: Any) -> Any:
    """Return a detached JSON-safe value using the training stack's rules."""

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
            f"unsupported Survival exact variant {candidate_variant!r}; "
            f"choices={SUPPORTED_CANDIDATE_VARIANTS}"
        )
    weight = FORMAL_SURVIVAL_WEIGHTS[candidate_variant]
    return {
        "candidate_variant": candidate_variant,
        "parent_variant": PARENT_VARIANT,
        "survival_weight": weight,
        "continued_training_control": candidate_variant
        == TSS_CONTROL_VARIANT,
        "formal_run_tag": FORMAL_RUN_TAGS[candidate_variant],
    }


def load_survival_target_statistics(
    path: Path = DEFAULT_TARGET_STATISTICS_PATH,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = load_json_mapping(
        resolved,
        "Survival target statistics",
    )
    expected = {
        "schema": TARGET_STATISTICS_SCHEMA,
        "dataset": "NUDT-SIRST",
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "used_train_ids_sha256": (
            "9565f584a5429fd1e5f0451b2d9496877f6f887493dd4d9954b4e976989f245b"
        ),
        "train_image_count": 530,
        "image_sizes": [[256, 256]],
        "patch_size": FORMAL_PATCH_SIZE,
        "mask_binarization": "float(mask)/255 > 0.5",
        "pool_kernel": SURVIVAL_DOWNSAMPLE,
        "pool_stride": SURVIVAL_DOWNSAMPLE,
        "full_image_equals_training_crop": True,
        "transform_preserves_positive_cell_count": True,
        "positive_cells": 1313,
        "negative_cells": 134367,
        "total_cells": 135680,
    }
    for name, required in expected.items():
        _require_equal(
            f"Survival target statistics {name}",
            payload.get(name),
            required,
        )
    if payload["positive_cells"] + payload["negative_cells"] != payload[
        "total_cells"
    ]:
        raise ValueError("Survival target statistics cell totals differ")
    expected_weight = (
        float(payload["negative_cells"]) / float(payload["positive_cells"])
    )
    observed_weight = payload.get("survival_pos_weight")
    if (
        isinstance(observed_weight, bool)
        or not isinstance(observed_weight, (int, float))
        or not math.isfinite(float(observed_weight))
        or float(observed_weight) != expected_weight
    ):
        raise ValueError(
            "Survival target statistics positive-class weight differs"
        )
    validation = payload.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("official_test_accessed") is not False
    ):
        raise ValueError(
            "Survival target statistics must exclude the official test set"
        )
    payload["path"] = str(resolved)
    payload["sha256"] = file_sha256(resolved)
    return payload


def formal_contract() -> dict[str, Any]:
    statistics = load_survival_target_statistics()
    return {
        "candidate_variants": list(SUPPORTED_CANDIDATE_VARIANTS),
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
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "survival_target": "max_pool_16_binary_presence",
        "survival_downsample": SURVIVAL_DOWNSAMPLE,
        "survival_weights": dict(FORMAL_SURVIVAL_WEIGHTS),
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_path": str(
            DEFAULT_TARGET_STATISTICS_PATH.relative_to(REPO_ROOT)
        ),
        "survival_target_statistics_sha256": statistics["sha256"],
        "segmentation_objective_unchanged": True,
        "inference_heads_required": False,
    }


def _validate_formal_args(args: argparse.Namespace) -> None:
    contract = candidate_contract(getattr(args, "variant", None))
    statistics_path = Path(
        getattr(args, "survival_target_statistics", "")
    ).resolve()
    _require_equal(
        "formal Survival target-statistics path",
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
        "run_tag": contract["formal_run_tag"],
        "survival_weight": contract["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "max_train_images": None,
        "max_val_images": None,
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            "formal Survival exact arguments differ: "
            f"expected={expected}, observed={observed}"
        )
    _require_equal(
        "formal Survival parent checkpoint path",
        Path(getattr(args, "parent_checkpoint", "")).resolve(),
        PARENT_CHECKPOINT_PATH.resolve(),
    )
    if bool(getattr(args, "parent_warm_start", False)) == bool(
        getattr(args, "exact_resume", False)
    ):
        raise ValueError(
            "Survival exact entry requires exactly one initialization mode"
        )
    device = getattr(args, "device", None)
    allow_cpu_smoke = getattr(args, "allow_cpu_smoke", False)
    if allow_cpu_smoke:
        if device != "cpu":
            raise ValueError("CPU smoke permission requires --device=cpu")
    elif device != "cuda:0":
        raise ValueError(
            "formal Survival exact training requires --device=cuda:0"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact V4 best-mIoU continued training with paired "
            "target-survival control/on variants"
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
    parser.add_argument(
        "--match-radius",
        type=float,
        default=FORMAL_MATCH_RADIUS,
    )
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
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="explicitly permit CPU-only contract tests",
    )
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
            "(--fresh is a compatibility spelling, not fresh weights)"
        ),
    )
    initialization.add_argument("--exact-resume", action="store_true")

    args = parser.parse_args(argv)
    contract = candidate_contract(args.variant)
    if args.run_tag is None:
        args.run_tag = contract["formal_run_tag"]
    if args.survival_weight is None:
        args.survival_weight = contract["survival_weight"]
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    if args.survival_pos_weight is None:
        args.survival_pos_weight = statistics["survival_pos_weight"]
    args.fresh = bool(args.parent_warm_start)
    args.parent_variant = PARENT_VARIANT
    args.relay_enabled = True
    args.relay_width = v4_exact.RELAY_WIDTH
    args.relay_initialization_seed = v4_exact.RELAY_INITIALIZATION_SEED
    args.dc_support_mode = v4_exact.DC_SUPPORT_MODE
    args.tail_z_thresholds = dict(v4_exact.TAIL_Z_THRESHOLDS)
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
    path = Path(parent_checkpoint).resolve()
    _require_equal(
        "Survival parent checkpoint path",
        path,
        PARENT_CHECKPOINT_PATH.resolve(),
    )
    actual_file_sha256 = file_sha256(path)
    _require_equal(
        "Survival parent checkpoint SHA-256",
        actual_file_sha256,
        PARENT_CHECKPOINT_SHA256,
    )
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(
            f"cannot load Survival parent checkpoint {path}: {exc}"
        ) from exc
    validated = v4_exact.require_evaluator_checkpoint_payload(
        payload,
        expected_variant=PARENT_VARIANT,
    )
    expected = {
        "epoch": PARENT_CHECKPOINT_EPOCH,
        "checkpoint_role": PARENT_CHECKPOINT_ROLE,
        "variant": PARENT_VARIANT,
        "dataset": "NUDT-SIRST",
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "state_dict_sha256": PARENT_STATE_DICT_SHA256,
    }
    for name, required in expected.items():
        _require_equal(
            f"Survival parent checkpoint {name}",
            validated.get(name),
            required,
        )
    state_dict = validated["state_dict"]
    _require_equal(
        "Survival parent checkpoint state-key count",
        len(state_dict),
        FORMAL_PARENT_STATE_KEY_COUNT,
    )
    actual_state_sha256 = exact_runner._state_content_sha256(
        state_dict,
        "Survival parent state_dict",
    )
    _require_equal(
        "Survival parent state_dict content SHA-256",
        actual_state_sha256,
        PARENT_STATE_DICT_SHA256,
    )
    return {
        "path": str(path),
        "checkpoint_sha256": actual_file_sha256,
        "checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "checkpoint_role_serialized": PARENT_CHECKPOINT_ROLE,
        "checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "state_dict_sha256": actual_state_sha256,
        "state_key_count": len(state_dict),
        "checkpoint_schema": validated["schema"],
    }


def _architecture_manifest(
    variant: str,
    model: nn.Module,
    validated_model: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _json_value(model.architecture_manifest())
    manifest.update(
        {
            "schema": ARCHITECTURE_MANIFEST_SCHEMA,
            "variant": variant,
            "model": (
                "model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival."
                "TPDNERV8MPRSDCHV4SurvivalSCTransNet"
            ),
            "parent_checkpoint_variant": PARENT_VARIANT,
            "state_key_count": FORMAL_STATE_KEY_COUNT,
            "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
            "survival_state_key_count": len(SURVIVAL_STATE_KEYS),
            "survival_state_keys": list(SURVIVAL_STATE_KEYS),
            "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
            "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
            "survival_version": SURVIVAL_VERSION,
            "survival_state_prefix": SURVIVAL_STATE_PREFIX,
            "survival_head_initialization": "exact_zero",
            "survival_training_only": True,
            "segmentation_path_modified": False,
            "segmentation_objective_unchanged": True,
            "inference_heads_required": False,
            "training_output": "TPDForwardOutput",
            "evaluation_output": "legacy_six_segmentation_maps",
            "exact_resume_scope": "same_tss_variant_only",
            "cross_version_exact_resume_supported": False,
            "formal_amp": FORMAL_AMP,
            "eps": FORMAL_EPS,
        }
    )
    _require_equal(
        "Survival validated total parameters",
        validated_model.get("total_parameters"),
        PRODUCTION_TOTAL_PARAMETERS,
    )
    return manifest


def _require_tss_manifest(
    manifest: Any,
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("Survival architecture manifest is missing")
    value = _json_value(dict(manifest))
    expected = {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "parent_checkpoint_variant": PARENT_VARIANT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "survival_state_key_count": len(SURVIVAL_STATE_KEYS),
        "survival_state_keys": list(SURVIVAL_STATE_KEYS),
        "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "segmentation_path_modified": False,
        "segmentation_objective_unchanged": True,
        "inference_heads_required": False,
        "training_output": "TPDForwardOutput",
        "evaluation_output": "legacy_six_segmentation_maps",
        "exact_resume_scope": "same_tss_variant_only",
        "cross_version_exact_resume_supported": False,
        "formal_amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
    }
    for name, required in expected.items():
        _require_equal(f"Survival manifest {name}", value.get(name), required)
    _require_equal(
        "Survival manifest DC support",
        value.get("ner_dc_offset_support_mode"),
        v4_exact.DC_SUPPORT_MODE,
    )
    _require_equal(
        "Survival manifest tail thresholds",
        value.get("tail_z_thresholds"),
        dict(v4_exact.TAIL_Z_THRESHOLDS),
    )
    return value


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    contract = candidate_contract(variant)
    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError("formal Survival exact builder requires seed=42")
    if eps != FORMAL_EPS:
        raise ValueError(
            f"formal Survival exact builder requires eps={FORMAL_EPS}"
        )
    model, builder_metadata = (
        survival_model.build_formal_v4_survival_model(seed=seed)
    )
    validated = survival_model.validate_formal_survival_model(
        model,
        require_zero_initialized_heads=True,
    )
    if not isinstance(builder_metadata, Mapping):
        raise TypeError("formal Survival builder metadata is not a mapping")
    _require_equal(
        "formal Survival state-key count",
        len(model.state_dict()),
        FORMAL_STATE_KEY_COUNT,
    )
    _require_equal(
        "formal Survival total parameter count",
        sum(parameter.numel() for parameter in model.parameters()),
        PRODUCTION_TOTAL_PARAMETERS,
    )
    extension_keys = tuple(
        name
        for name in model.state_dict()
        if name.startswith(SURVIVAL_STATE_PREFIX)
    )
    _require_equal(
        "formal Survival extension state keys",
        set(extension_keys),
        set(SURVIVAL_STATE_KEYS),
    )
    for name in extension_keys:
        if torch.count_nonzero(model.state_dict()[name]).item() != 0:
            raise ValueError(
                f"formal Survival initial state {name!r} is not zero"
            )
    manifest = _architecture_manifest(variant, model, validated)
    metadata: dict[str, Any] = {
        "variant": variant,
        "candidate_family": "v4_tail_aware_target_survival",
        "parent_variant": PARENT_VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "survival_weight": contract["survival_weight"],
        "continued_training_control": contract[
            "continued_training_control"
        ],
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_state_keys": list(SURVIVAL_STATE_KEYS),
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "segmentation_objective_unchanged": True,
        "inference_heads_required": False,
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
        },
        "architecture_manifest": manifest,
        "architecture_id": canonical_sha256(manifest),
    }
    return model, metadata


def _require_tss_metadata(
    metadata: Any,
    *,
    variant: str,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("Survival model metadata is missing")
    value = _json_value(dict(metadata))
    contract = candidate_contract(variant)
    expected = {
        "variant": variant,
        "parent_variant": PARENT_VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "survival_weight": contract["survival_weight"],
        "continued_training_control": contract[
            "continued_training_control"
        ],
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_state_keys": list(SURVIVAL_STATE_KEYS),
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "total_parameters": PRODUCTION_TOTAL_PARAMETERS,
        "parent_state_key_count": FORMAL_PARENT_STATE_KEY_COUNT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "segmentation_objective_unchanged": True,
        "inference_heads_required": False,
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
        _require_equal(f"Survival metadata {name}", value.get(name), required)
    manifest = _require_tss_manifest(
        value.get("architecture_manifest"),
        variant=variant,
    )
    _require_equal(
        "Survival metadata architecture digest",
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
        "existing Survival exact protocol",
    )
    _require_equal(
        "existing Survival protocol schema",
        protocol.get("schema"),
        ENTRY_SCHEMA,
    )
    identity = require_tss_run_identity(
        protocol.get("run_identity"),
        label="existing Survival protocol",
        expected_variant=args.variant,
    )
    training = identity.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError(
            "existing Survival protocol has no training contract"
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
            "existing Survival training contract lacks fields: "
            f"{missing}"
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
            new_module_prefixes=SURVIVAL_NEW_MODULE_PREFIXES,
            zero_init_prefixes=SURVIVAL_ZERO_INIT_PREFIXES,
            parent_state_dict_path=PARENT_STATE_DICT_PATH,
            expected_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            map_location="cpu",
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
                "Survival exact resume lacks extension-parent initialization"
            )
        initial_rng = training["initial_rng"]
        selection_policy = training["selection_policy"]
        if not isinstance(initial_rng, Mapping):
            raise ValueError("existing Survival initial_rng is invalid")
        if not isinstance(selection_policy, Mapping):
            raise ValueError(
                "existing Survival selection_policy is invalid"
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
        "Survival exact entry requires parent warm-start or exact resume"
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if str(device) != "cuda:0":
            raise ValueError(
                "each Survival exact process must use cuda:0"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Survival exact training requires one process-visible GPU"
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
        raise ValueError(
            "Survival exact entry supports only cpu or cuda:0"
        )
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
        "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_INDEX"
    )
    physical_uuid = os.environ.get(
        "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_UUID"
    )
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "Survival physical GPU index must identify registered GPU 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            f"Survival physical GPU UUID differs for GPU {physical_index}"
        )
    if payload.get("device_uuid") != expected_uuid:
        raise RuntimeError(
            "visible cuda:0 UUID differs from Survival assignment"
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
                "verified_v4_survival_worker_environment"
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
    payload = load_json_mapping(path, "Survival exact source lock")
    _require_equal(
        "Survival exact source-lock schema",
        payload.get("schema"),
        EXACT_SOURCE_LOCK_SCHEMA,
    )
    _require_equal(
        "Survival exact source-lock variants",
        tuple(payload.get("variants", ())),
        SUPPORTED_CANDIDATE_VARIANTS,
    )
    _require_equal(
        "Survival exact source-lock formal contract",
        payload.get("formal_contract"),
        formal_contract(),
    )
    _require_equal(
        "Survival exact source-lock training data",
        payload.get("training_data_sha256"),
        training_data_sha256,
    )
    _require_equal(
        "Survival exact source-lock target statistics",
        payload.get("survival_target_statistics_sha256"),
        statistics["sha256"],
    )
    _require_equal(
        "Survival exact source-lock parent checkpoint",
        payload.get("parent_checkpoint_sha256"),
        PARENT_CHECKPOINT_SHA256,
    )
    locked = payload.get("source_sha256")
    if not isinstance(locked, Mapping):
        raise ValueError(
            "Survival exact source lock has no source mapping"
        )
    required = {
        str(runtime.relative_to(REPO_ROOT))
        for runtime in RUNTIME_SOURCE_PATHS
    }
    missing = sorted(required - set(locked))
    if missing:
        raise ValueError(
            f"Survival exact source lock omits runtime sources: {missing}"
        )
    for relative, expected in locked.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("Survival source lock has an invalid path")
        runtime = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(runtime.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError(
                "Survival source path escapes repository"
            ) from exc
        _require_equal(
            "Survival source canonical path",
            canonical,
            relative,
        )
        _require_equal(
            f"Survival source digest for {relative}",
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
            "class": "torch.nn.functional.binary_cross_entropy_with_logits",
            "reduction": "mean",
            "input": "raw_logits",
            "heads": ["emb1", "emb2"],
            "aggregate": "python_ordered_sum",
            "survival_weight": candidate["survival_weight"],
            "survival_pos_weight": statistics["survival_pos_weight"],
            "target": "max_pool_16_binary_presence",
            "target_pool_kernel": SURVIVAL_DOWNSAMPLE,
            "target_pool_stride": SURVIVAL_DOWNSAMPLE,
            "target_statistics_schema": TARGET_STATISTICS_SCHEMA,
            "target_statistics_sha256": statistics["sha256"],
            "target_statistics_used_train_ids_sha256": statistics[
                "used_train_ids_sha256"
            ],
            "disabled_path_builds_target": False,
            "disabled_path_reads_logits": False,
        },
        "variant": variant,
        "continued_training_control": candidate[
            "continued_training_control"
        ],
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
        "tss_run_identity_schema": RUN_IDENTITY_SCHEMA,
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
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "survival_weight": candidate["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_sha256": statistics["sha256"],
        "survival_target_downsample": SURVIVAL_DOWNSAMPLE,
        "continued_training_control": candidate[
            "continued_training_control"
        ],
        "segmentation_objective_unchanged": True,
        "inference_heads_required": False,
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
    metadata = _require_tss_metadata(
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
        "Survival run source-lock keys",
        set(source_locks),
        expected_source_locks,
    )
    _require_equal(
        "Survival run target-statistics source lock",
        source_locks["survival_target_statistics"],
        statistics["sha256"],
    )
    _require_equal(
        "Survival run parent-checkpoint source lock",
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
        determinism=_required_determinism(
            args.variant,
            statistics,
        ),
        initial_model_state_sha256=initial_model_state_sha256,
        initial_rng=copy.deepcopy(dict(initial_rng)),
        selection_policy=copy.deepcopy(dict(selection_policy)),
    )


def require_tss_run_identity(
    identity: Any,
    *,
    label: str,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no Survival exact run identity")
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
        raise ValueError(f"{label} run_id is not Survival-owned")
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
        raise ValueError(f"{label} source-lock identity is not Survival-owned")
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
            f"{label} Survival determinism field {name}",
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
        "preserved_new_state_key_count": len(SURVIVAL_STATE_KEYS),
        "new_module_prefixes": list(SURVIVAL_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(SURVIVAL_ZERO_INIT_PREFIXES),
    }
    _require_equal(
        f"{label} extension-parent provenance",
        provenance,
        expected_provenance,
    )
    # Builder metadata is intentionally not duplicated at the top-level by
    # ExactRunIdentity.  The architecture digest still binds it through the
    # exact runner contract.
    return value


def _require_complete_validation_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        name for name in STORED_VALIDATION_METRICS if name not in metrics
    ]
    if missing:
        raise ValueError(
            f"Survival validation metrics lack fields: {missing}"
        )
    return {
        name: copy.deepcopy(metrics[name])
        for name in STORED_VALIDATION_METRICS
    }


EVALUATOR_CHECKPOINT_REQUIRED_FIELDS = (
    "schema",
    "epoch",
    "checkpoint_role",
    "variant",
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
    "split_hashes",
    "run_identity",
    "checkpoint_identity",
    "parent_checkpoint_path",
    "parent_checkpoint_sha256",
    "parent_checkpoint_role",
    "parent_checkpoint_epoch",
    "parent_checkpoint_state_dict_sha256",
    "warm_start_applied",
    "warm_start_schema",
    "initialization_mode",
    "survival_version",
    "survival_state_prefix",
    "survival_parameters",
    "survival_head_initialization",
    "survival_training_only",
    "survival_weight",
    "survival_pos_weight",
    "survival_target_statistics_sha256",
    "segmentation_objective_unchanged",
    "inference_heads_required",
    "continued_training_control",
    "selection_source",
    "official_test_accessed",
)


def _checkpoint_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    variant = str(identity["variant"])
    statistics = load_survival_target_statistics()
    candidate = candidate_contract(variant)
    return {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "variant": variant,
        "parent_variant": PARENT_VARIANT,
        "run_id": identity["run_id"],
        "architecture_id": identity["architecture_id"],
        "builder_manifest_sha256": identity["builder_manifest_sha256"],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            PARENT_STATE_DICT_SHA256
        ),
        "survival_version": SURVIVAL_VERSION,
        "survival_weight": candidate["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_sha256": statistics["sha256"],
        "continued_training_control": candidate[
            "continued_training_control"
        ],
    }


def require_evaluator_checkpoint_payload(
    payload: Any,
    *,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Survival evaluator checkpoint is not a mapping")
    value = copy.deepcopy(dict(payload))
    missing = [
        field
        for field in EVALUATOR_CHECKPOINT_REQUIRED_FIELDS
        if field not in value
    ]
    if missing:
        raise ValueError(
            f"Survival evaluator checkpoint lacks fields: {missing}"
        )
    _require_equal(
        "Survival checkpoint schema",
        value["schema"],
        CHECKPOINT_SCHEMA,
    )
    identity = require_tss_run_identity(
        value["run_identity"],
        label="Survival evaluator checkpoint",
        expected_variant=expected_variant,
    )
    statistics = load_survival_target_statistics()
    candidate = candidate_contract(identity["variant"])
    expected_top_level = {
        "variant": identity["variant"],
        "parent_variant": PARENT_VARIANT,
        "dataset": identity["dataset"],
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
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
        "survival_version": SURVIVAL_VERSION,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "survival_training_only": True,
        "survival_weight": candidate["survival_weight"],
        "survival_pos_weight": statistics["survival_pos_weight"],
        "survival_target_statistics_sha256": statistics["sha256"],
        "segmentation_objective_unchanged": True,
        "inference_heads_required": False,
        "continued_training_control": candidate[
            "continued_training_control"
        ],
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }
    for name, expected in expected_top_level.items():
        _require_equal(
            f"Survival checkpoint {name}",
            value.get(name),
            expected,
        )
    epoch = value["epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or epoch > FORMAL_EPOCHS
    ):
        raise ValueError("Survival evaluator checkpoint epoch is invalid")
    if value["checkpoint_role"] not in {
        "last_evaluated_epoch",
        "best_validation_pd_primary",
        "best_validation_miou_secondary",
    }:
        raise ValueError(
            "Survival evaluator checkpoint role is invalid"
        )
    state = value["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Survival evaluator state_dict is invalid")
    _require_equal(
        "Survival evaluator state-key count",
        len(state),
        FORMAL_STATE_KEY_COUNT,
    )
    _require_equal(
        "Survival evaluator extension state keys",
        {
            name
            for name in state
            if isinstance(name, str)
            and name.startswith(SURVIVAL_STATE_PREFIX)
        },
        set(SURVIVAL_STATE_KEYS),
    )
    for name in ("optimizer", "scaler"):
        if not isinstance(value[name], Mapping):
            raise ValueError(
                f"Survival evaluator checkpoint {name} is invalid"
            )
    _require_equal(
        "Survival checkpoint scheduler",
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
            raise ValueError(
                f"Survival evaluator metric {name} is invalid"
            )
    metadata = _require_tss_metadata(
        value["model_metadata"],
        variant=identity["variant"],
    )
    _require_equal(
        "Survival identity architecture digest",
        identity.get("builder_manifest_sha256"),
        canonical_sha256(metadata["architecture_manifest"]),
    )
    split_hashes = value["split_hashes"]
    if not isinstance(split_hashes, Mapping) or not split_hashes:
        raise ValueError("Survival evaluator split hashes are invalid")
    for name, digest in split_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Survival evaluator split-hash name is invalid")
        _validate_sha256(digest, f"Survival split hash {name}")
    _require_equal(
        "Survival evaluator checkpoint identity",
        value["checkpoint_identity"],
        _checkpoint_identity(identity),
    )
    if "state_dict_sha256" in value:
        _require_equal(
            "Survival evaluator state_dict SHA-256",
            value["state_dict_sha256"],
            exact_runner._state_content_sha256(
                state,
                "Survival evaluator state_dict",
            ),
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
        identity = require_tss_run_identity(
            context.run_identity,
            label="Survival checkpoint context",
        )
        metadata = _require_tss_metadata(
            self.model_metadata,
            variant=identity["variant"],
        )
        statistics = load_survival_target_statistics()
        candidate = candidate_contract(identity["variant"])
        exact_payload = context.exact_payload
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": identity["variant"],
            "parent_variant": PARENT_VARIANT,
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
            "checkpoint_identity": _checkpoint_identity(identity),
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
            "survival_version": SURVIVAL_VERSION,
            "survival_state_prefix": SURVIVAL_STATE_PREFIX,
            "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
            "survival_head_initialization": "exact_zero",
            "survival_training_only": True,
            "survival_weight": candidate["survival_weight"],
            "survival_pos_weight": statistics["survival_pos_weight"],
            "survival_target_statistics_sha256": statistics["sha256"],
            "segmentation_objective_unchanged": True,
            "inference_heads_required": False,
            "continued_training_control": candidate[
                "continued_training_control"
            ],
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }
        return require_evaluator_checkpoint_payload(
            payload,
            expected_variant=identity["variant"],
        )


class TPDNERV8V4SurvivalExactRunner(
    v4_exact.TPDNERV8V4TailAwareExactRunner
):
    """Reject every non-TSS journal before exact state restoration."""

    def _require_v8_active_journal(self) -> None:
        active = self.journal.load_active()
        if active is None:
            return
        payload, _ = self._load_exact_payload(active.checkpoint_path)
        require_tss_run_identity(
            payload.get("run_identity"),
            label="active Survival exact journal",
            expected_variant=self.spec.variant,
        )
        if not isinstance(payload.get("optimizer"), Mapping):
            raise ValueError(
                "active Survival journal has no optimizer state"
            )


DCHExactRunner = TPDNERV8V4SurvivalExactRunner


def compute_stage_loss(
    outputs: Any,
    target: torch.Tensor,
    criterion: nn.Module,
    *,
    survival_weight: float,
    survival_pos_weight: float,
) -> TPDTrainingLoss:
    return compute_tpd_training_loss(
        outputs,
        target,
        criterion,
        survival_weight=survival_weight,
        survival_pos_weight=survival_pos_weight,
    )


@dataclass
class EpochLossAccumulator:
    """Sample-weighted formal loss logging without auxiliary recomputation."""

    survival_enabled: bool
    total_sum: float = 0.0
    segmentation_sum: float = 0.0
    survival_sum: float = 0.0
    emb1_sum: float = 0.0
    emb2_sum: float = 0.0
    sample_count: int = 0

    def update(self, losses: TPDTrainingLoss, batch_count: int) -> None:
        if (
            isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count < 1
        ):
            raise ValueError("loss accumulator batch_count is invalid")
        terms = losses.survival_terms
        if self.survival_enabled:
            if len(terms) != 2:
                raise ValueError(
                    "enabled Survival loss must expose exactly two terms"
                )
        elif terms:
            raise ValueError(
                "control loss must not construct/read survival terms"
            )
        self.total_sum += float(losses.total.detach().item()) * batch_count
        self.segmentation_sum += (
            float(losses.segmentation.detach().item()) * batch_count
        )
        self.survival_sum += (
            float(losses.survival.detach().item()) * batch_count
        )
        if terms:
            self.emb1_sum += float(terms[0].detach().item()) * batch_count
            self.emb2_sum += float(terms[1].detach().item()) * batch_count
        self.sample_count += batch_count

    def fields(self) -> dict[str, float | None]:
        if not self.sample_count:
            raise RuntimeError("no training samples were accumulated")
        denominator = float(self.sample_count)
        return {
            "train_total_loss": self.total_sum / denominator,
            "train_segmentation_loss": (
                self.segmentation_sum / denominator
            ),
            "train_survival_loss": self.survival_sum / denominator,
            "train_survival_emb1_loss": (
                self.emb1_sum / denominator
                if self.survival_enabled
                else None
            ),
            "train_survival_emb2_loss": (
                self.emb2_sum / denominator
                if self.survival_enabled
                else None
            ),
        }


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
    )
    payload = {name: getattr(args, name) for name in names}
    return payload


def protocol_payload(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    normalization: Mapping[str, float],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = require_tss_run_identity(
        run_identity,
        label="Survival protocol",
        expected_variant=args.variant,
    )
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    return {
        "schema": ENTRY_SCHEMA,
        "formal_contract": formal_contract(),
        "arguments": training_arguments(args),
        "run_directory": directory,
        "model": _require_tss_metadata(
            model_metadata,
            variant=args.variant,
        ),
        "normalization": dict(normalization),
        "run_identity": identity,
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
        raise RuntimeError(
            "Survival exact metrics are not a complete contiguous run"
        )
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
                raise RuntimeError(
                    f"Survival exact event lacks {name!r}"
                )
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
        "completed Survival exact protocol",
    )
    identity = require_tss_run_identity(
        protocol.get("run_identity"),
        label="completed Survival protocol",
        expected_variant=args.variant,
    )
    return {
        "schema": COMPLETION_SUMMARY_SCHEMA,
        "status": "complete",
        "variant": args.variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "formal_contract": formal_contract(),
        "run_identity": identity,
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
        "model": _require_tss_metadata(
            model_metadata,
            variant=args.variant,
        ),
        "split_hashes": dict(split_hashes),
        "survival_weight": args.survival_weight,
        "continued_training_control": (
            args.variant == TSS_CONTROL_VARIANT
        ),
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
        "best_miou_checkpoint": directory
        / exact_runner.BEST_MIOU_FILENAME,
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
                f"non-finite Survival validation metric {name!r}: "
                f"{value!r}"
            )


def _require_prepared_statistics_match(
    statistics: Mapping[str, Any],
    prepared: PreparedData,
) -> None:
    _require_equal(
        "Survival statistics used-train IDs",
        statistics["used_train_ids_sha256"],
        prepared.split_hashes["used_train_sha256"],
    )
    _require_equal(
        "Survival statistics train image count",
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
    write_or_verify_json(
        directory / "split.json",
        prepared.split_manifest,
    )
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
                "formal Survival exact training must evaluate each epoch"
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
                **loss_fields,
                # Compatibility alias: it is exactly the total objective.
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
        raise RuntimeError(
            "completed Survival exact run has no best selection"
        )
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
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TARGET_STATISTICS_PATH",
    "DCHExactRunner",
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
    "FORMAL_CONTROL_RUN_TAG",
    "FORMAL_EPOCHS",
    "FORMAL_EPS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_MIN_LR",
    "FORMAL_PATCH_SIZE",
    "FORMAL_RUN_TAGS",
    "FORMAL_SURVIVAL_WEIGHTS",
    "FORMAL_TSS_RUN_TAG",
    "FORMAL_WARMUP_EPOCHS",
    "FORMAL_WORKERS",
    "PARENT_CHECKPOINT_EPOCH",
    "PARENT_CHECKPOINT_PATH",
    "PARENT_CHECKPOINT_ROLE",
    "PARENT_CHECKPOINT_ROLE_SHORT",
    "PARENT_CHECKPOINT_SHA256",
    "PARENT_STATE_DICT_SHA256",
    "PRODUCTION_SURVIVAL_PARAMETERS",
    "PRODUCTION_TOTAL_PARAMETERS",
    "RUN_IDENTITY_SCHEMA",
    "RUN_ID_PREFIX",
    "RUNTIME_SOURCE_PATHS",
    "SOURCE_LOCK_KEY",
    "SPLIT_SEED",
    "STORED_VALIDATION_METRICS",
    "SUPPORTED_CANDIDATE_VARIANTS",
    "SURVIVAL_CONTROL_VARIANT",
    "SURVIVAL_ON_VARIANT",
    "SURVIVAL_STATE_KEYS",
    "SURVIVAL_STATE_PREFIX",
    "SURVIVAL_VERSION",
    "TARGET_STATISTICS_SCHEMA",
    "TPDNERV8V4SurvivalExactRunner",
    "TRAINING_SEED",
    "TSS_CONTROL_VARIANT",
    "TSS_ON_VARIANT",
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
    "require_tss_run_identity",
    "run_directory",
    "run_training",
    "source_lock_contract",
    "supported_candidate_variants",
    "validate_parent_checkpoint",
]


if __name__ == "__main__":
    main()
