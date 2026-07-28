#!/usr/bin/env python3
"""Frozen, diagnostic-only specification for the V3 DC knockout matrix.

This module is intentionally free of evaluation and publication side effects.
It is the single registry shared by the future knockout evaluator, source-lock
tool, and diagnostic aggregate.  Nothing in this contract may authorize or
modify the six-component formal V3 decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v3_exact as exact,
)


SPEC_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_spec_v2"
)
EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_evaluation_v2"
)
AGGREGATE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_aggregate_v2"
)
SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock_v2"
)
COMPLETE_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_complete_v2"
)
ARTIFACT_KIND = "dc_knockout_diagnostic"
DATASET = "NUDT-SIRST"
VARIANT = exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
VALIDATION_COUNT = 133
TARGET_COUNT = 189
TINY_TARGET_COUNT = 39
FORMAL_RUN_TAG = exact.FORMAL_RUN_TAG
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
CHECKPOINTS = tuple(CHECKPOINT_ROLES)
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
CUBLAS_WORKSPACE_CONFIG_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PYTHONHASHSEED_ENV = "PYTHONHASHSEED"
PYTHONHASHSEED = str(TRAINING_SEED)
PHYSICAL_GPU_INDEX_ENV = (
    "TPD_NER_V8_MPRS_DCH_V3_DC_KNOCKOUT_PHYSICAL_GPU_INDEX"
)
PHYSICAL_GPU_UUID_ENV = (
    "TPD_NER_V8_MPRS_DCH_V3_DC_KNOCKOUT_PHYSICAL_GPU_UUID"
)
CHECKPOINT_GPU_LANES = {
    "best.pth.tar": {
        "physical_gpu_index": 2,
        "physical_gpu_uuid": (
            "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
        ),
    },
    "best_miou.pth.tar": {
        "physical_gpu_index": 3,
        "physical_gpu_uuid": (
            "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
        ),
    },
}
DC_OFFSET_KEYS = (
    "tpd_ner.dc_offsets.4",
    "tpd_ner.dc_offsets.3",
    "tpd_ner.dc_offsets.2",
)
KNOCKOUT_ZERO_KEYS = {
    "zero_all_dc": DC_OFFSET_KEYS,
    "zero_dc_stage4": (DC_OFFSET_KEYS[0],),
    "zero_dc_stage3": (DC_OFFSET_KEYS[1],),
    "zero_dc_stage2": (DC_OFFSET_KEYS[2],),
}
KNOCKOUT_MODES = tuple(KNOCKOUT_ZERO_KEYS)
EXPECTED_ROW_COUNT = len(CHECKPOINTS) * len(KNOCKOUT_MODES)
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
FIXED_THRESHOLD_FIELDS = (
    "threshold",
    "pd",
    "fa",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "tiny_pd",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
)
RAW_POINT_FIELDS = frozenset(
    {
        "val_loss",
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
        "fa",
        "false_objects_per_image",
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
        "threshold",
    }
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v8_mprs_dch_v3_dc_knockout_v2"
)
FORMAL_RESULT_ROOT = exact.DEFAULT_OUTPUT_ROOT
FORMAL_RUN_DIR = (
    FORMAL_RESULT_ROOT
    / DATASET
    / VARIANT
    / f"seed_{TRAINING_SEED}_{FORMAL_RUN_TAG}"
)
DEFAULT_RUN_DIR = (
    DEFAULT_OUTPUT_ROOT
    / DATASET
    / VARIANT
    / f"seed_{TRAINING_SEED}_{FORMAL_RUN_TAG}"
)
DEFAULT_COMPARISON_DIR = DEFAULT_OUTPUT_ROOT / DATASET / "comparison"
AGGREGATE_JSON_NAME = (
    "tpd_ner_v8_mprs_dch_v3_dc_knockout_comparison.json"
)
AGGREGATE_MARKDOWN_NAME = (
    "tpd_ner_v8_mprs_dch_v3_dc_knockout_comparison.md"
)
COMPLETE_MARKER_NAME = "DC_KNOCKOUT_COMPLETE.json"
SWEEP_FILENAMES = {
    "best.pth.tar": "dc_knockout_pd_fa_sweep_best.pth.json",
    "best_miou.pth.tar": (
        "dc_knockout_pd_fa_sweep_best_miou.pth.json"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical byte encoding used for identities and locks."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def threshold_contract() -> dict[str, Any]:
    return {
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "threshold_min": 0.01,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "extra_thresholds": list(EXTRA_THRESHOLDS),
        "tail_logit_step": 0.1,
        "closed_interval_required": True,
        # Match the locked formal evaluator exactly: the inherited threshold
        # set supplies its positive lower endpoint; the closed-interval helper
        # adds the upper float32 endpoint and 1.0, not an extra 0.0 point.
        "include_zero": False,
        "include_one": True,
        "include_last_float32_below_one": True,
        "fixed_threshold": 0.5,
        "fa_budgets": list(FA_BUDGETS),
    }


def matrix_rows() -> list[dict[str, Any]]:
    """Return the fixed checkpoint-major 2 x 4 diagnostic row registry."""

    return [
        {
            "row_index": index,
            "row_id": f"{checkpoint}:{mode}",
            "checkpoint": checkpoint,
            "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
            "knockout_mode": mode,
            "zeroed_state_keys": list(KNOCKOUT_ZERO_KEYS[mode]),
        }
        for index, (checkpoint, mode) in enumerate(
            (
                (checkpoint, mode)
                for checkpoint in CHECKPOINTS
                for mode in KNOCKOUT_MODES
            ),
            start=1,
        )
    ]


def fixed_specification() -> dict[str, Any]:
    """Return the immutable, JSON-native knockout evidence contract."""

    return {
        "schema": SPEC_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "scope": "evaluation_only_same_checkpoint_counterfactual",
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_components": [],
        "dataset": DATASET,
        "variant": VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "multi_seed_scheduled": False,
        "official_test_accessed": False,
        "source_run_directory": str(FORMAL_RUN_DIR.resolve()),
        "diagnostic_output_root": str(DEFAULT_OUTPUT_ROOT.resolve()),
        "checkpoint_roles": dict(CHECKPOINT_ROLES),
        "knockout_modes": [
            {
                "mode": mode,
                "zeroed_state_keys": list(KNOCKOUT_ZERO_KEYS[mode]),
                "in_memory_only": True,
                "derived_checkpoint_written": False,
                "non_zeroed_dc_offsets_keep_learned_values": True,
            }
            for mode in KNOCKOUT_MODES
        ],
        "matrix": matrix_rows(),
        "row_count": EXPECTED_ROW_COUNT,
        "original_learned_v3_rows_counted": False,
        "original_learned_v3_is_read_only_reference": True,
        "zero_all_dc_is_v2_training_trajectory": False,
        "execution_contract": {
            "logical_device": "cuda:0",
            "cuda_device_order": CUDA_DEVICE_ORDER,
            "cublas_workspace_config_env": CUBLAS_WORKSPACE_CONFIG_ENV,
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "cublas_workspace_config_required_before_evaluator_start": True,
            "pythonhashseed_env": PYTHONHASHSEED_ENV,
            "pythonhashseed": PYTHONHASHSEED,
            "pythonhashseed_required_before_evaluator_start": True,
            "physical_gpu_index_env": PHYSICAL_GPU_INDEX_ENV,
            "physical_gpu_uuid_env": PHYSICAL_GPU_UUID_ENV,
            "checkpoint_gpu_lanes": {
                checkpoint: dict(CHECKPOINT_GPU_LANES[checkpoint])
                for checkpoint in CHECKPOINTS
            },
            "checkpoint_scheduling": "parallel_fixed_gpu2_gpu3",
            "knockout_mode_scheduling": (
                "sequential_pristine_state_per_checkpoint"
            ),
        },
        "threshold_contract": threshold_contract(),
        "metric_contract": {
            "validation_count": VALIDATION_COUNT,
            "target_count": TARGET_COUNT,
            "tiny_target_count": TINY_TARGET_COUNT,
            "fixed_threshold_fields": list(FIXED_THRESHOLD_FIELDS),
            "raw_point_fields": sorted(RAW_POINT_FIELDS),
            "fa_budget_keys": list(BUDGET_KEYS),
            "full_raw_points_required": True,
            "signed_delta_reference": "same_role_learned_v3_formal_row",
        },
        "publication_contract": {
            "aggregate_schema": AGGREGATE_SCHEMA,
            "complete_marker_schema": COMPLETE_MARKER_SCHEMA,
            "aggregate_may_contain_decision": False,
            "aggregate_may_contain_performance_gate_assessment": False,
            "formal_outputs_read_only": True,
            "completion_means_package_complete_not_model_success": True,
        },
    }


def specification_sha256() -> str:
    return canonical_sha256(fixed_specification())


def validate_specification(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping),
        "DC knockout specification must be a mapping",
    )
    observed = dict(value)
    expected = fixed_specification()
    _require(observed == expected, "DC knockout specification differs")
    _require(
        len(observed["matrix"]) == EXPECTED_ROW_COUNT,
        "DC knockout matrix is not eight rows",
    )
    _require(
        len({row["row_id"] for row in observed["matrix"]})
        == EXPECTED_ROW_COUNT,
        "DC knockout matrix row identities are not unique",
    )
    return observed


def validated_output_root(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Return a real, non-symlinked diagnostic root outside the formal tree."""

    unresolved = Path(os.path.abspath(os.fspath(output_root)))
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        _require(
            not current.is_symlink(),
            f"diagnostic output root has a symbolic-link component: {current}",
        )
    root = unresolved.resolve(strict=False)
    formal_root = FORMAL_RESULT_ROOT.resolve()
    _require(
        root != formal_root and not root.is_relative_to(formal_root),
        "diagnostic output root may not equal or descend from the formal V3 "
        "result root",
    )
    return root


def run_directory(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    root = validated_output_root(output_root)
    return (
        root
        / DATASET
        / VARIANT
        / f"seed_{TRAINING_SEED}_{FORMAL_RUN_TAG}"
    )


def sweep_path(
    checkpoint: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    _require(checkpoint in CHECKPOINTS, f"unsupported checkpoint: {checkpoint}")
    return run_directory(output_root) / SWEEP_FILENAMES[checkpoint]


def comparison_directory(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    root = validated_output_root(output_root)
    return root / DATASET / "comparison"


def aggregate_paths(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, Path, Path]:
    directory = comparison_directory(output_root)
    return (
        directory / AGGREGATE_JSON_NAME,
        directory / AGGREGATE_MARKDOWN_NAME,
        directory / COMPLETE_MARKER_NAME,
    )


def assert_static_contract() -> None:
    _require(EXPECTED_ROW_COUNT == 8, "knockout matrix must contain eight rows")
    _require(len(CHECKPOINTS) == 2, "knockout matrix must use two checkpoints")
    _require(len(KNOCKOUT_MODES) == 4, "knockout matrix must use four modes")
    _require(
        set(CHECKPOINT_GPU_LANES) == set(CHECKPOINTS)
        and {
            int(lane["physical_gpu_index"])
            for lane in CHECKPOINT_GPU_LANES.values()
        }
        == {2, 3},
        "knockout checkpoints must map exactly to physical GPU2/3",
    )
    _require(
        tuple(KNOCKOUT_ZERO_KEYS["zero_all_dc"]) == DC_OFFSET_KEYS,
        "all-DC knockout key set differs",
    )
    _require(
        validated_output_root(DEFAULT_OUTPUT_ROOT)
        != FORMAL_RESULT_ROOT.resolve(),
        "diagnostic/formal result roots collide",
    )
    validate_specification(fixed_specification())


assert_static_contract()


__all__ = [
    "AGGREGATE_JSON_NAME",
    "AGGREGATE_MARKDOWN_NAME",
    "AGGREGATE_SCHEMA",
    "ARTIFACT_KIND",
    "BUDGET_KEYS",
    "CHECKPOINTS",
    "CHECKPOINT_GPU_LANES",
    "CHECKPOINT_ROLES",
    "COMPLETE_MARKER_NAME",
    "COMPLETE_MARKER_SCHEMA",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLAS_WORKSPACE_CONFIG_ENV",
    "DATASET",
    "CUDA_DEVICE_ORDER",
    "DC_OFFSET_KEYS",
    "DEFAULT_COMPARISON_DIR",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RUN_DIR",
    "EVALUATION_SCHEMA",
    "EXPECTED_EPOCHS",
    "EXPECTED_ROW_COUNT",
    "EXTRA_THRESHOLDS",
    "FA_BUDGETS",
    "FIXED_THRESHOLD_FIELDS",
    "FORMAL_RESULT_ROOT",
    "FORMAL_RUN_DIR",
    "KNOCKOUT_MODES",
    "KNOCKOUT_ZERO_KEYS",
    "PHYSICAL_GPU_INDEX_ENV",
    "PHYSICAL_GPU_UUID_ENV",
    "PYTHONHASHSEED",
    "PYTHONHASHSEED_ENV",
    "RAW_POINT_FIELDS",
    "REPO_ROOT",
    "SOURCE_LOCK_SCHEMA",
    "SPEC_SCHEMA",
    "SPLIT_SEED",
    "SWEEP_FILENAMES",
    "TARGET_COUNT",
    "TINY_TARGET_COUNT",
    "TRAINING_SEED",
    "VALIDATION_COUNT",
    "VARIANT",
    "aggregate_paths",
    "canonical_json_bytes",
    "canonical_sha256",
    "comparison_directory",
    "fixed_specification",
    "matrix_rows",
    "run_directory",
    "specification_sha256",
    "sweep_path",
    "threshold_contract",
    "validated_output_root",
    "validate_specification",
]
