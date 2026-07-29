#!/usr/bin/env python3
"""Exact formal800 entry for the C-style QFG discriminative-LR recipe.

This entry deliberately does not alter the QFG-V2 C/D source lock or either
active trajectory.  It reuses the exact same ``qfg_only`` model, parent
checkpoint, data split, segmentation loss, checkpoint selection policy and
extension warm-start, but gives the training recipe its own identity:

* the public/run/checkpoint variant is only ``qfg_dlr``;
* target-survival loss is exactly zero (the C-style semantic arm);
* optimizer groups are ordered ``parent``, ``qfg``, ``tss``;
* every group is constructed at the canonical base LR, while the exact runner
  applies schedule multipliers ``0.1``, ``1.0``, ``1.0`` each epoch;
* BatchNorm running statistics are frozen immediately after every
  ``model.train()`` call; affine tensors remain trainable.

The underlying architecture manifest remains the original ``qfg_only``
manifest because the network graph and its initialized tensors are unchanged.
The outer model metadata, exact-run identity, source lock and compatibility
checkpoints are DLR-owned, preventing cross-recipe exact resume.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tpd_exact_resume as exact_resume  # noqa: E402
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import tpd_group_scaled_exact_runner as scaled_runner  # noqa: E402
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as v2  # noqa: E402


ENTRY_SCHEMA = "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_exact_entry_v1"
DLR_RUN_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_run_identity_v1"
)
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_source_lock_v1"
)
MODEL_METADATA_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_model_metadata_v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_checkpoint_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_checkpoint_identity_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_completion_summary_v1"
)
SOURCE_LOCK_KEY = "tpd_ner_v4_qfg_v2_croa_dlr_exact_source_lock"
RUN_ID_PREFIX = "tpd-ner-v4-qfg-v2-croa-dlr-exact:"

QFG_DLR_VARIANT = "qfg_dlr"
SUPPORTED_CANDIDATE_VARIANTS = (QFG_DLR_VARIANT,)
MODEL_VARIANT = v2.QFG_ONLY_VARIANT
QFG_VARIANT = v2.QFG_VARIANT
TSS_VARIANT = v2.TSS_CONTROL_VARIANT
TRAINING_RECIPE = "qfg_v2_croa_dlr_v1"
FORMAL_RUN_TAG = "formal800_qfg_dlr_parent01_bn_frozen"
FORMAL_SURVIVAL_WEIGHT = 0.0

PARENT_GROUP = "parent"
QFG_GROUP = "qfg"
TSS_GROUP = "tss"
OPTIMIZER_GROUP_ORDER = (PARENT_GROUP, QFG_GROUP, TSS_GROUP)
OPTIMIZER_GROUP_PREFIX_RULES = {
    PARENT_GROUP: "all names not matched by qfg or tss",
    QFG_GROUP: "tpd_qfg.",
    TSS_GROUP: "target_survival.",
}
OPTIMIZER_GROUP_MULTIPLIERS = {
    PARENT_GROUP: 0.1,
    QFG_GROUP: 1.0,
    TSS_GROUP: 1.0,
}
OPTIMIZER_GROUP_PARAMETER_TENSORS = {
    PARENT_GROUP: 466,
    QFG_GROUP: 16,
    TSS_GROUP: 4,
}
OPTIMIZER_GROUP_PARAMETER_NUMEL = {
    PARENT_GROUP: 10_854_446,
    QFG_GROUP: 15_684,
    TSS_GROUP: 98,
}
FORMAL_BATCHNORM_MODULE_COUNT = 26
BATCHNORM_EVENT_FIELD = "batchnorm_running_stats_frozen_count"

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

DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT
    / "experiments/tpd_ner_v4_qfg_v2_croa_dlr_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v4_qfg_v2_croa_dlr_exact_v1"
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
    (
        Path(__file__).resolve(),
        Path(scaled_runner.__file__).resolve(),
        *v2.RUNTIME_SOURCE_PATHS,
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
            v2.base.json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def supported_candidate_variants() -> tuple[str, ...]:
    return SUPPORTED_CANDIDATE_VARIANTS


def candidate_contract(candidate_variant: str) -> dict[str, Any]:
    if candidate_variant != QFG_DLR_VARIANT:
        raise ValueError(
            f"unsupported DLR variant {candidate_variant!r}; "
            f"choices={SUPPORTED_CANDIDATE_VARIANTS}"
        )
    return {
        "candidate_variant": QFG_DLR_VARIANT,
        "model_variant": MODEL_VARIANT,
        "qfg_variant": QFG_VARIANT,
        "tss_variant": TSS_VARIANT,
        "training_recipe": TRAINING_RECIPE,
        "survival_weight": FORMAL_SURVIVAL_WEIGHT,
        "tss_control": True,
        "formal_run_tag": FORMAL_RUN_TAG,
    }


def optimizer_recipe_contract() -> dict[str, Any]:
    return {
        "training_recipe": TRAINING_RECIPE,
        "optimizer": "torch.optim.Adam",
        "group_order": list(OPTIMIZER_GROUP_ORDER),
        "group_prefix_rules": dict(OPTIMIZER_GROUP_PREFIX_RULES),
        "group_parameter_tensors": dict(
            OPTIMIZER_GROUP_PARAMETER_TENSORS
        ),
        "group_parameter_numel": dict(OPTIMIZER_GROUP_PARAMETER_NUMEL),
        "group_initial_lr": {
            name: FORMAL_BASE_LR for name in OPTIMIZER_GROUP_ORDER
        },
        "schedule_multipliers": dict(OPTIMIZER_GROUP_MULTIPLIERS),
        "group_lr_formula": scaled_runner.GROUP_LR_FORMULA,
    }


def batchnorm_recipe_contract() -> dict[str, Any]:
    return {
        "policy": "freeze_running_stats_after_every_model_train",
        "module_types": "torch.nn.modules.batchnorm._BatchNorm",
        "expected_module_count": FORMAL_BATCHNORM_MODULE_COUNT,
        "affine_parameters_trainable": True,
        "model_train_reapplication_required": True,
    }


def formal_contract() -> dict[str, Any]:
    base_contract = copy.deepcopy(v2.formal_contract())
    base_contract.update(
        {
            "candidate_family": "v4_tail_aware_qfg_v2_croa_dlr",
            "candidate_variants": [QFG_DLR_VARIANT],
            "model_variant": MODEL_VARIANT,
            "tss_variants": {QFG_DLR_VARIANT: TSS_VARIANT},
            "survival_weights": {
                QFG_DLR_VARIANT: FORMAL_SURVIVAL_WEIGHT
            },
            "training_recipe": TRAINING_RECIPE,
            "formal_run_tag": FORMAL_RUN_TAG,
            "optimizer_recipe": optimizer_recipe_contract(),
            "batchnorm_recipe": batchnorm_recipe_contract(),
        }
    )
    return base_contract


def _as_v2_args(args: argparse.Namespace) -> argparse.Namespace:
    proxy = argparse.Namespace(**vars(args))
    proxy.variant = MODEL_VARIANT
    proxy.run_tag = v2.FORMAL_QFG_ONLY_RUN_TAG
    proxy.survival_weight = FORMAL_SURVIVAL_WEIGHT
    proxy.qfg_variant = QFG_VARIANT
    proxy.tss_variant = TSS_VARIANT
    return proxy


def _validate_formal_args(args: argparse.Namespace) -> None:
    candidate_contract(getattr(args, "variant", None))
    proxy = _as_v2_args(args)
    v2._validate_formal_args(proxy)
    expected = {
        "run_tag": FORMAL_RUN_TAG,
        "survival_weight": FORMAL_SURVIVAL_WEIGHT,
        "qfg_variant": QFG_VARIANT,
        "tss_variant": TSS_VARIANT,
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            "formal DLR arguments differ: "
            f"expected={expected}, observed={observed}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact800 C-style QFG training with parent 0.1x LR, "
            "QFG 1x LR and frozen BatchNorm running statistics"
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
    parser.add_argument("--run-tag", default=FORMAL_RUN_TAG)
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
    parser.add_argument(
        "--survival-weight",
        type=float,
        default=FORMAL_SURVIVAL_WEIGHT,
    )
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
    statistics = load_survival_target_statistics(
        args.survival_target_statistics
    )
    if args.survival_pos_weight is None:
        args.survival_pos_weight = statistics["survival_pos_weight"]
    args.fresh = bool(args.parent_warm_start)
    args.parent_variant = PARENT_VARIANT
    args.relay_enabled = True
    args.relay_width = v2.v4_exact.RELAY_WIDTH
    args.relay_initialization_seed = (
        v2.v4_exact.RELAY_INITIALIZATION_SEED
    )
    args.dc_support_mode = v2.v4_exact.DC_SUPPORT_MODE
    args.tail_z_thresholds = dict(v2.v4_exact.TAIL_Z_THRESHOLDS)
    args.qfg_variant = QFG_VARIANT
    args.tss_variant = TSS_VARIANT
    args.training_recipe = TRAINING_RECIPE
    _validate_formal_args(args)
    return args


def run_directory(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    return (
        Path(args.output_root).resolve()
        / args.dataset
        / QFG_DLR_VARIANT
        / f"seed_{args.seed}_{args.run_tag}"
    )


def _adapt_model_metadata(
    base_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    validated = v2._require_qfg_metadata(
        base_metadata,
        variant=MODEL_VARIANT,
    )
    manifest = copy.deepcopy(validated["architecture_manifest"])
    return {
        "schema": MODEL_METADATA_SCHEMA,
        "variant": QFG_DLR_VARIANT,
        "candidate_variant": QFG_DLR_VARIANT,
        "model_variant": MODEL_VARIANT,
        "training_recipe": TRAINING_RECIPE,
        "qfg_variant": QFG_VARIANT,
        "tss_variant": TSS_VARIANT,
        "survival_weight": FORMAL_SURVIVAL_WEIGHT,
        "architecture_manifest": manifest,
        "architecture_id": canonical_sha256(manifest),
        "base_model_metadata": copy.deepcopy(validated),
        "optimizer_recipe": optimizer_recipe_contract(),
        "batchnorm_recipe": batchnorm_recipe_contract(),
    }


def _require_dlr_model_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("DLR model metadata is missing")
    metadata = _json_value(dict(value))
    expected = {
        "schema": MODEL_METADATA_SCHEMA,
        "variant": QFG_DLR_VARIANT,
        "candidate_variant": QFG_DLR_VARIANT,
        "model_variant": MODEL_VARIANT,
        "training_recipe": TRAINING_RECIPE,
        "qfg_variant": QFG_VARIANT,
        "tss_variant": TSS_VARIANT,
        "survival_weight": FORMAL_SURVIVAL_WEIGHT,
        "optimizer_recipe": optimizer_recipe_contract(),
        "batchnorm_recipe": batchnorm_recipe_contract(),
    }
    for name, required in expected.items():
        _require_equal(f"DLR metadata {name}", metadata.get(name), required)
    base_metadata = v2._require_qfg_metadata(
        metadata.get("base_model_metadata"),
        variant=MODEL_VARIANT,
    )
    manifest = copy.deepcopy(base_metadata["architecture_manifest"])
    _require_equal(
        "DLR architecture manifest",
        metadata.get("architecture_manifest"),
        manifest,
    )
    _require_equal(
        "DLR architecture digest",
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
    candidate_contract(variant)
    model, base_metadata = v2.build_selected_model(
        MODEL_VARIANT,
        seed,
        eps=eps,
    )
    metadata = _adapt_model_metadata(base_metadata)
    return model, _require_dlr_model_metadata(metadata)


def optimizer_group_name(parameter_name: str) -> str:
    if not isinstance(parameter_name, str) or not parameter_name:
        raise ValueError("optimizer parameter name must be non-empty")
    if parameter_name.startswith("tpd_qfg."):
        return QFG_GROUP
    if parameter_name.startswith("target_survival."):
        return TSS_GROUP
    return PARENT_GROUP


def optimizer_group_manifest(model: nn.Module) -> dict[str, Any]:
    if not isinstance(model, nn.Module):
        raise TypeError("optimizer group target must be an nn.Module")
    grouped_names = {name: [] for name in OPTIMIZER_GROUP_ORDER}
    grouped_parameters = {name: [] for name in OPTIMIZER_GROUP_ORDER}
    seen_parameter_ids: set[int] = set()
    for parameter_name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            raise ValueError(
                f"model parameter {parameter_name!r} is aliased"
            )
        seen_parameter_ids.add(parameter_id)
        group_name = optimizer_group_name(parameter_name)
        grouped_names[group_name].append(parameter_name)
        grouped_parameters[group_name].append(parameter)
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    if seen_parameter_ids != model_parameter_ids:
        raise ValueError("optimizer grouping is not complete")

    records: list[dict[str, Any]] = []
    for group_name in OPTIMIZER_GROUP_ORDER:
        names = grouped_names[group_name]
        parameters = grouped_parameters[group_name]
        tensor_count = len(parameters)
        numel = sum(parameter.numel() for parameter in parameters)
        _require_equal(
            f"DLR {group_name} parameter tensor count",
            tensor_count,
            OPTIMIZER_GROUP_PARAMETER_TENSORS[group_name],
        )
        _require_equal(
            f"DLR {group_name} parameter numel",
            numel,
            OPTIMIZER_GROUP_PARAMETER_NUMEL[group_name],
        )
        records.append(
            {
                "group_name": group_name,
                "parameter_names": list(names),
                "parameter_tensor_count": tensor_count,
                "parameter_numel": numel,
                "initial_lr": FORMAL_BASE_LR,
                "schedule_multiplier": (
                    OPTIMIZER_GROUP_MULTIPLIERS[group_name]
                ),
            }
        )
    return {
        "group_order": list(OPTIMIZER_GROUP_ORDER),
        "groups": records,
        "parameter_tensor_count": sum(
            record["parameter_tensor_count"] for record in records
        ),
        "parameter_numel": sum(
            record["parameter_numel"] for record in records
        ),
    }


def build_optimizer(model: nn.Module) -> torch.optim.Adam:
    manifest = optimizer_group_manifest(model)
    parameter_by_name = dict(model.named_parameters())
    groups = []
    for record in manifest["groups"]:
        group_name = record["group_name"]
        groups.append(
            {
                "params": [
                    parameter_by_name[name]
                    for name in record["parameter_names"]
                ],
                "lr": FORMAL_BASE_LR,
                scaled_runner.GROUP_NAME_OPTION: group_name,
                scaled_runner.SCHEDULE_MULTIPLIER_OPTION: (
                    record["schedule_multiplier"]
                ),
            }
        )
    optimizer = torch.optim.Adam(groups, lr=FORMAL_BASE_LR)
    _require_optimizer_contract(model, optimizer)
    return optimizer


def _require_optimizer_contract(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    expected_manifest = optimizer_group_manifest(model)
    contract = exact_runner.optimizer_contract(model, optimizer)
    groups = contract["param_groups"]
    _require_equal(
        "DLR optimizer group count",
        len(groups),
        len(OPTIMIZER_GROUP_ORDER),
    )
    for index, (record, expected) in enumerate(
        zip(groups, expected_manifest["groups"])
    ):
        options = record["options"]
        _require_equal(
            f"DLR optimizer group {index} name",
            options.get(scaled_runner.GROUP_NAME_OPTION),
            expected["group_name"],
        )
        _require_equal(
            f"DLR optimizer group {index} multiplier",
            options.get(scaled_runner.SCHEDULE_MULTIPLIER_OPTION),
            expected["schedule_multiplier"],
        )
        _require_equal(
            f"DLR optimizer group {index} initial LR",
            options.get("lr"),
            FORMAL_BASE_LR,
        )
        _require_equal(
            f"DLR optimizer group {index} parameter names",
            record["parameter_names"],
            expected["parameter_names"],
        )
    return contract


def count_batchnorm_modules(model: nn.Module) -> int:
    return sum(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )


def freeze_formal_batchnorm_running_stats(model: nn.Module) -> int:
    count = scaled_runner.freeze_batchnorm_running_stats(model)
    _require_equal(
        "formal DLR BatchNorm module count",
        count,
        FORMAL_BATCHNORM_MODULE_COUNT,
    )
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.training:
                raise RuntimeError("formal DLR BatchNorm remained in train mode")
            for parameter in (module.weight, module.bias):
                if parameter is not None and not parameter.requires_grad:
                    raise RuntimeError(
                        "formal DLR BatchNorm affine parameter was frozen"
                    )
    return count


def _loss_contract(statistics: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(v2._loss_contract(MODEL_VARIANT, statistics))
    value.update(
        {
            "variant": QFG_DLR_VARIANT,
            "candidate_variant": QFG_DLR_VARIANT,
            "model_variant": MODEL_VARIANT,
            "training_recipe": TRAINING_RECIPE,
        }
    )
    return value


def _required_determinism(
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(
        v2._required_determinism(MODEL_VARIANT, statistics)
    )
    value.pop("qfg_run_identity_schema", None)
    value.update(
        {
            "entry_schema": ENTRY_SCHEMA,
            "dlr_run_identity_schema": DLR_RUN_IDENTITY_SCHEMA,
            "candidate_variant": QFG_DLR_VARIANT,
            "model_variant": MODEL_VARIANT,
            "training_recipe": TRAINING_RECIPE,
            "source_lock_schema": EXACT_SOURCE_LOCK_SCHEMA,
            "survival_weight": FORMAL_SURVIVAL_WEIGHT,
            "tss_control": True,
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
    metadata = _require_dlr_model_metadata(model_metadata)
    optimizer_contract = _require_optimizer_contract(model, optimizer)
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
        "DLR run source-lock keys",
        set(source_locks),
        expected_source_locks,
    )
    _require_equal(
        "DLR run target-statistics source lock",
        source_locks["survival_target_statistics"],
        statistics["sha256"],
    )
    _require_equal(
        "DLR run parent-checkpoint source lock",
        source_locks["parent_checkpoint"],
        PARENT_CHECKPOINT_SHA256,
    )
    return exact_runner.ExactRunSpec(
        run_id=(
            f"{RUN_ID_PREFIX}{args.dataset}:{QFG_DLR_VARIANT}:"
            f"seed-{args.seed}:split-{args.split_seed}:{args.run_tag}"
        ),
        variant=QFG_DLR_VARIANT,
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
        loss=_loss_contract(statistics),
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
        determinism=_required_determinism(statistics),
        initial_model_state_sha256=initial_model_state_sha256,
        initial_rng=copy.deepcopy(dict(initial_rng)),
        selection_policy=copy.deepcopy(dict(selection_policy)),
    )


def require_dlr_run_identity(
    identity: Any,
    *,
    label: str,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no DLR exact run identity")
    value = copy.deepcopy(dict(identity))
    expected = expected_variant or QFG_DLR_VARIANT
    candidate_contract(expected)
    _require_equal(f"{label} schema", value.get("schema"), exact_runner.RUN_IDENTITY_SCHEMA)
    _require_equal(f"{label} variant", value.get("variant"), expected)
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"{label} run_id is not DLR-owned")
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
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    }
    if not isinstance(source_locks, Mapping):
        raise ValueError(f"{label} source locks are missing")
    _require_equal(
        f"{label} source-lock keys",
        set(source_locks),
        expected_lock_keys,
    )
    for name, digest in source_locks.items():
        v2._validate_sha256(digest, f"{label} source lock {name}")
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
        raise ValueError(f"{label} has no training contract")
    _require_equal(
        f"{label} loss contract",
        training.get("loss"),
        _loss_contract(statistics),
    )
    _require_equal(
        f"{label} determinism contract",
        training.get("determinism"),
        _required_determinism(statistics),
    )
    schedule = training.get("manual_lr_schedule")
    expected_schedule = exact_runner.ManualCosineSchedule(
        total_epochs=FORMAL_EPOCHS,
        base_lr=FORMAL_BASE_LR,
        min_lr=FORMAL_MIN_LR,
        warmup_epochs=FORMAL_WARMUP_EPOCHS,
    ).normalized()
    _require_equal(f"{label} LR schedule", schedule, expected_schedule)
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError(f"{label} optimizer contract is missing")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list):
        raise ValueError(f"{label} optimizer groups are missing")
    _require_equal(
        f"{label} optimizer group count",
        len(groups),
        len(OPTIMIZER_GROUP_ORDER),
    )
    for index, (group, group_name) in enumerate(
        zip(groups, OPTIMIZER_GROUP_ORDER)
    ):
        if not isinstance(group, Mapping):
            raise ValueError(f"{label} optimizer group {index} is invalid")
        options = group.get("options")
        if not isinstance(options, Mapping):
            raise ValueError(
                f"{label} optimizer group {index} options are invalid"
            )
        _require_equal(
            f"{label} optimizer group {index} name",
            options.get(scaled_runner.GROUP_NAME_OPTION),
            group_name,
        )
        _require_equal(
            f"{label} optimizer group {index} multiplier",
            options.get(scaled_runner.SCHEDULE_MULTIPLIER_OPTION),
            OPTIMIZER_GROUP_MULTIPLIERS[group_name],
        )
        _require_equal(
            f"{label} optimizer group {index} initial LR",
            options.get("lr"),
            FORMAL_BASE_LR,
        )
        names = group.get("parameter_names")
        if not isinstance(names, list):
            raise ValueError(
                f"{label} optimizer group {index} names are invalid"
            )
        _require_equal(
            f"{label} optimizer group {index} tensor count",
            len(names),
            OPTIMIZER_GROUP_PARAMETER_TENSORS[group_name],
        )
        for parameter_name in names:
            _require_equal(
                f"{label} optimizer parameter {parameter_name}",
                optimizer_group_name(parameter_name),
                group_name,
            )
    return value


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
    target_statistics_path: Path = DEFAULT_TARGET_STATISTICS_PATH,
) -> dict[str, str]:
    path = Path(exact_source_lock_path).resolve()
    statistics = load_survival_target_statistics(target_statistics_path)
    payload = load_json_mapping(path, "DLR exact source lock")
    _require_equal(
        "DLR source-lock schema",
        payload.get("schema"),
        EXACT_SOURCE_LOCK_SCHEMA,
    )
    _require_equal(
        "DLR source-lock variants",
        tuple(payload.get("variants", ())),
        SUPPORTED_CANDIDATE_VARIANTS,
    )
    _require_equal(
        "DLR source-lock training recipe",
        payload.get("training_recipe"),
        TRAINING_RECIPE,
    )
    _require_equal(
        "DLR source-lock formal contract",
        payload.get("formal_contract"),
        formal_contract(),
    )
    _require_equal(
        "DLR source-lock training data",
        payload.get("training_data_sha256"),
        training_data_sha256,
    )
    _require_equal(
        "DLR source-lock target statistics",
        payload.get("survival_target_statistics_sha256"),
        statistics["sha256"],
    )
    _require_equal(
        "DLR source-lock parent checkpoint",
        payload.get("parent_checkpoint_sha256"),
        PARENT_CHECKPOINT_SHA256,
    )
    locked = payload.get("source_sha256")
    if not isinstance(locked, Mapping):
        raise ValueError("DLR exact source lock has no source mapping")
    required = {
        str(runtime.relative_to(REPO_ROOT))
        for runtime in RUNTIME_SOURCE_PATHS
    }
    _require_equal(
        "DLR source-lock source count",
        payload.get("source_count"),
        len(locked),
    )
    _require_equal(
        "DLR locked runtime source set",
        set(locked),
        required,
    )
    for relative, expected in locked.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("DLR source lock has an invalid path")
        runtime = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(runtime.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError("DLR source path escapes repository") from exc
        _require_equal("DLR source canonical path", canonical, relative)
        _require_equal(
            f"DLR source digest for {relative}",
            file_sha256(runtime),
            expected,
        )
    return {
        SOURCE_LOCK_KEY: file_sha256(path),
        "training_data": v2._validate_sha256(
            training_data_sha256,
            "DLR training data SHA-256",
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
        "existing DLR exact protocol",
    )
    _require_equal(
        "existing DLR protocol schema",
        protocol.get("schema"),
        ENTRY_SCHEMA,
    )
    identity = require_dlr_run_identity(
        protocol.get("run_identity"),
        label="existing DLR protocol",
        expected_variant=args.variant,
    )
    training = identity.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError("existing DLR protocol has no training contract")
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(
            f"existing DLR training contract lacks fields: {missing}"
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
                "DLR exact resume lacks extension-parent initialization"
            )
        initial_rng = training["initial_rng"]
        selection_policy = training["selection_policy"]
        if not isinstance(initial_rng, Mapping):
            raise ValueError("existing DLR initial_rng is invalid")
        if not isinstance(selection_policy, Mapping):
            raise ValueError("existing DLR selection_policy is invalid")
        return InitializationPlan(
            request=exact_runner.InitializationRequest.exact(),
            contract=copy.deepcopy(dict(initialization)),
            initial_model_state_sha256=str(
                training["initial_model_state_sha256"]
            ),
            initial_rng=copy.deepcopy(dict(initial_rng)),
            selection_policy=copy.deepcopy(dict(selection_policy)),
        )
    raise RuntimeError("DLR exact entry requires warm-start or exact resume")


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    return v2.resolve_device(_as_v2_args(args))


def environment_contract(device: torch.device) -> dict[str, Any]:
    payload = v2.environment_contract(device)
    payload["training_recipe"] = TRAINING_RECIPE
    return payload


class DLRExactRunner(scaled_runner.GroupScaledExactRunner):
    """Group-scaled runner that rejects every non-DLR active journal."""

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
                    "DLR exact resume requires an active journal"
                )
            payload, _ = self._load_exact_payload(active.checkpoint_path)
            require_dlr_run_identity(
                payload.get("run_identity"),
                label="active DLR exact journal",
                expected_variant=self.spec.variant,
            )
            if not isinstance(payload.get("optimizer"), Mapping):
                raise ValueError(
                    "active DLR exact journal has no optimizer state"
                )
        return super().startup(request)


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
        "training_recipe",
    )
    payload = {name: getattr(args, name) for name in names}
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
    identity = require_dlr_run_identity(
        run_identity,
        label="DLR protocol",
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
        "model": _require_dlr_model_metadata(model_metadata),
        "architecture_manifest": copy.deepcopy(
            model_metadata["architecture_manifest"]
        ),
        "normalization": dict(normalization),
        "run_identity": identity,
        "candidate_variant": QFG_DLR_VARIANT,
        "model_variant": MODEL_VARIANT,
        "qfg_variant": QFG_VARIANT,
        "tss_variant": TSS_VARIANT,
        "training_recipe": TRAINING_RECIPE,
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
        "loss": _loss_contract(statistics),
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
            "same_variant_epoch_boundary_only": True,
            "cross_recipe": "forbidden",
            "cross_version": "forbidden",
            "optimizer_inherited_from_parent": False,
            "scheduler_restore": (
                "manual_base_schedule_then_group_multipliers"
            ),
        },
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }


def _checkpoint_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "variant": QFG_DLR_VARIANT,
        "training_recipe": TRAINING_RECIPE,
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
        identity = require_dlr_run_identity(
            context.run_identity,
            label="DLR checkpoint context",
        )
        metadata = _require_dlr_model_metadata(self.model_metadata)
        exact_payload = context.exact_payload
        source_locks = copy.deepcopy(dict(identity["source_locks"]))
        return {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": QFG_DLR_VARIANT,
            "candidate_variant": QFG_DLR_VARIANT,
            "model_variant": MODEL_VARIANT,
            "qfg_variant": QFG_VARIANT,
            "tss_variant": TSS_VARIANT,
            "training_recipe": TRAINING_RECIPE,
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
                "completed_epoch": context.epoch,
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
            "parent_checkpoint_path": str(
                PARENT_CHECKPOINT_PATH.resolve()
            ),
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_checkpoint_role": PARENT_CHECKPOINT_ROLE_SHORT,
            "parent_checkpoint_epoch": PARENT_CHECKPOINT_EPOCH,
            "parent_checkpoint_state_dict_sha256": (
                PARENT_STATE_DICT_SHA256
            ),
            "survival_weight": FORMAL_SURVIVAL_WEIGHT,
            "tss_control": True,
            "optimizer_recipe": optimizer_recipe_contract(),
            "batchnorm_recipe": batchnorm_recipe_contract(),
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }


def _load_complete_events(
    path: Path,
    epochs: int,
) -> list[dict[str, Any]]:
    events = v2._load_complete_events(path, epochs)
    schedule = exact_runner.ManualCosineSchedule(
        total_epochs=FORMAL_EPOCHS,
        base_lr=FORMAL_BASE_LR,
        min_lr=FORMAL_MIN_LR,
        warmup_epochs=FORMAL_WARMUP_EPOCHS,
    )
    for event in events:
        epoch = int(event["epoch"])
        base_lr = schedule.learning_rate(epoch)
        _require_equal(
            f"DLR epoch {epoch} optimizer group names",
            event.get(scaled_runner.GROUP_NAMES_EVENT_FIELD),
            list(OPTIMIZER_GROUP_ORDER),
        )
        _require_equal(
            f"DLR epoch {epoch} schedule multipliers",
            event.get(scaled_runner.SCHEDULE_MULTIPLIERS_EVENT_FIELD),
            [
                OPTIMIZER_GROUP_MULTIPLIERS[name]
                for name in OPTIMIZER_GROUP_ORDER
            ],
        )
        _require_equal(
            f"DLR epoch {epoch} group learning rates",
            event.get(scaled_runner.GROUP_LEARNING_RATES_EVENT_FIELD),
            [
                base_lr * OPTIMIZER_GROUP_MULTIPLIERS[name]
                for name in OPTIMIZER_GROUP_ORDER
            ],
        )
        _require_equal(
            f"DLR epoch {epoch} BatchNorm count",
            event.get(BATCHNORM_EVENT_FIELD),
            FORMAL_BATCHNORM_MODULE_COUNT,
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
        "completed DLR exact protocol",
    )
    identity = require_dlr_run_identity(
        protocol.get("run_identity"),
        label="completed DLR protocol",
        expected_variant=args.variant,
    )
    return {
        "schema": COMPLETION_SUMMARY_SCHEMA,
        "status": "complete",
        "variant": QFG_DLR_VARIANT,
        "candidate_variant": QFG_DLR_VARIANT,
        "model_variant": MODEL_VARIANT,
        "qfg_variant": QFG_VARIANT,
        "tss_variant": TSS_VARIANT,
        "training_recipe": TRAINING_RECIPE,
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
        "best_miou_epoch": miou_epoch,
        "best_miou_validation_metrics": (
            v2._require_complete_validation_metrics(events[miou_epoch - 1])
        ),
        "model": _require_dlr_model_metadata(model_metadata),
        "split_hashes": dict(split_hashes),
        "survival_weight": FORMAL_SURVIVAL_WEIGHT,
        "tss_control": True,
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
        "best_checkpoint": directory / exact_runner.BEST_FILENAME,
        "best_miou_checkpoint": (
            directory / exact_runner.BEST_MIOU_FILENAME
        ),
        "last_checkpoint": directory / exact_runner.LAST_FILENAME,
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
        "formal DLR BatchNorm topology",
        count_batchnorm_modules(model),
        FORMAL_BATCHNORM_MODULE_COUNT,
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
    runner = DLRExactRunner(
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

    print(
        f"START variant={QFG_DLR_VARIANT} model={MODEL_VARIANT} "
        f"recipe={TRAINING_RECIPE} "
        f"mode={snapshot.initialization_mode.value} "
        f"completed={snapshot.completed_epoch} "
        f"next={snapshot.next_epoch} device={device}",
        flush=True,
    )
    while snapshot.next_epoch is not None:
        control = runner.next_epoch_control()
        if not control.should_evaluate:
            raise RuntimeError(
                "formal DLR exact training must evaluate each epoch"
            )
        epoch_started = time.time()
        model.train()
        batchnorm_count = freeze_formal_batchnorm_running_stats(model)
        accumulator = EpochLossAccumulator(survival_enabled=False)
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
                    survival_weight=FORMAL_SURVIVAL_WEIGHT,
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
                "variant": QFG_DLR_VARIANT,
                "candidate_variant": QFG_DLR_VARIANT,
                "model_variant": MODEL_VARIANT,
                "qfg_variant": QFG_VARIANT,
                "tss_variant": TSS_VARIANT,
                "training_recipe": TRAINING_RECIPE,
                **loss_fields,
                "train_loss": loss_fields["train_total_loss"],
                "survival_weight": FORMAL_SURVIVAL_WEIGHT,
                "survival_pos_weight": args.survival_pos_weight,
                BATCHNORM_EVENT_FIELD: batchnorm_count,
                "processed_train_samples": accumulator.sample_count,
                "epoch_seconds": time.time() - epoch_started,
                "skipped_singleton_batches": skipped_singletons,
                **metrics,
            },
            extra_state={
                "variant": QFG_DLR_VARIANT,
                "model_variant": MODEL_VARIANT,
                "training_recipe": TRAINING_RECIPE,
                "qfg_variant": QFG_VARIANT,
                "tss_variant": TSS_VARIANT,
                "formal_eps": FORMAL_EPS,
                "survival_weight": FORMAL_SURVIVAL_WEIGHT,
                "survival_pos_weight": args.survival_pos_weight,
                "survival_target_statistics_sha256": statistics["sha256"],
                BATCHNORM_EVENT_FIELD: batchnorm_count,
                "processed_train_samples": accumulator.sample_count,
                "skipped_singleton_batches": skipped_singletons,
            },
        )
        print(
            f"EPOCH {control.epoch:03d}/{FORMAL_EPOCHS} "
            f"total={float(loss_fields['train_total_loss']):.6f} "
            f"seg={float(loss_fields['train_segmentation_loss']):.6f} "
            f"mIoU={float(metrics['miou']):.6f} "
            f"Pd={float(metrics['pd']):.6f} "
            f"Fa={float(metrics['fa']):.8f}",
            flush=True,
        )

    if snapshot.best_selection is None:
        raise RuntimeError("completed DLR exact run has no best selection")
    summary = completion_summary(
        args,
        directory=directory,
        model_metadata=model_metadata,
        split_hashes=prepared.split_hashes,
        selection=snapshot.best_selection,
    )
    v2.write_or_verify_json(directory / "summary.json", summary)
    print(
        f"COMPLETE variant={QFG_DLR_VARIANT} "
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
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DLRExactRunner",
    "DLR_RUN_IDENTITY_SCHEMA",
    "ENTRY_SCHEMA",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EvaluatorCheckpointAdapter",
    "FORMAL_BASE_LR",
    "FORMAL_BATCHNORM_MODULE_COUNT",
    "FORMAL_EPOCHS",
    "FORMAL_RUN_TAG",
    "FORMAL_SURVIVAL_WEIGHT",
    "MODEL_METADATA_SCHEMA",
    "MODEL_VARIANT",
    "OPTIMIZER_GROUP_MULTIPLIERS",
    "OPTIMIZER_GROUP_ORDER",
    "OPTIMIZER_GROUP_PARAMETER_NUMEL",
    "OPTIMIZER_GROUP_PARAMETER_TENSORS",
    "QFG_DLR_VARIANT",
    "QFG_VARIANT",
    "RUNTIME_SOURCE_PATHS",
    "RUN_ID_PREFIX",
    "SOURCE_LOCK_KEY",
    "SUPPORTED_CANDIDATE_VARIANTS",
    "TRAINING_RECIPE",
    "TSS_VARIANT",
    "batchnorm_recipe_contract",
    "build_optimizer",
    "build_selected_model",
    "candidate_contract",
    "completion_summary",
    "count_batchnorm_modules",
    "environment_contract",
    "formal_contract",
    "freeze_formal_batchnorm_running_stats",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "optimizer_group_manifest",
    "optimizer_group_name",
    "optimizer_recipe_contract",
    "parse_args",
    "protocol_payload",
    "require_dlr_run_identity",
    "resolve_device",
    "run_directory",
    "run_training",
    "source_lock_contract",
    "supported_candidate_variants",
    "training_arguments",
]


if __name__ == "__main__":
    main()
