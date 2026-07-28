#!/usr/bin/env python3
"""Post-training evidence closure for formal V4 Tail-Aware NER.

This module is deliberately separate from the frozen training runtime.  It
does not evaluate a model and never derives final Pd@Fa values from
``metrics.jsonl``.  Its only V4 metric inputs are the two completed,
checkpoint-bound Pd/Fa sweep JSON files produced for the candidate's own
``best.pth.tar`` and ``best_miou.pth.tar`` checkpoints.

The baseline/V1/V2/V3 rows are read from the versioned V3 selection-contract
repair aggregate and are cross-checked against every bound raw sweep.  The V4
six-component gate is then recomputed independently from normalized raw
metrics.  V3 is reported as an additional structural-predecessor delta and
does not replace the preregistered V1/V2 gate roles.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "posttraining_aggregate_v1"
)
COMPLETE_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "posttraining_complete_v1"
)
V4_SWEEP_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa_v1"
)
V4_FINAL_METRIC_COVERAGE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "final_metric_coverage_v1"
)

DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
VALIDATION_COUNT = 133
TARGET_COUNT = 189
TINY_TARGET_COUNT = 39
VALIDATION_SPLIT_SHA256 = (
    "86247e5970f93224c64005e1ac7f3a933bafb37baf279ab71fce5670ae925e06"
)
TRAINING_DATA_SHA256 = (
    "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
)

BASELINE_VARIANT = "baseline_sctransnet"
V1_OFF_VARIANT = "tpd_ner_v8_mprs_dch_full_relay_off"
V2_ON_VARIANT = "tpd_ner_v8_mprs_dch_v2_full_relay_on"
V3_ON_VARIANT = "tpd_ner_v8_mprs_dch_v3_full_relay_on"
V4_ON_VARIANT = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
)
HISTORICAL_VARIANTS = (
    BASELINE_VARIANT,
    V1_OFF_VARIANT,
    V2_ON_VARIANT,
    V3_ON_VARIANT,
)
ALL_VARIANTS = (*HISTORICAL_VARIANTS, V4_ON_VARIANT)

CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
CHECKPOINTS = tuple(CHECKPOINT_ROLES)
ROLE_NAMES = {
    "best_validation_pd_primary": "pd_primary",
    "best_validation_miou_secondary": "miou_secondary",
}

BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_BY_KEY = dict(zip(BUDGET_KEYS, FA_BUDGETS))
LAST_FLOAT32_BELOW_ONE = float.fromhex("0x1.fffffep-1")
UPPER_BOUNDARY_THRESHOLD = 1.0
VALID_PIXEL_COUNT = VALIDATION_COUNT * 256 * 256
FINAL_COVERAGE_FIELDS = (
    "threshold",
    "pd",
    "fa",
    "miou",
    "false_objects_per_image",
    "tiny_pd",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
)
V4_SWEEP_POINT_FIELDS = (
    *FINAL_COVERAGE_FIELDS,
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
BUDGET_MINIMUM_MATCHED = {
    "1e-06": 187,
    "5e-06": 188,
    "1e-05": 188,
    "5e-05": 188,
    "0.0001": 188,
}

FIXED_GATE = {
    "pd_primary": {
        "minimum_matched_targets": 188,
        "minimum_pd": 188 / TARGET_COUNT,
        "maximum_fa": 1e-6,
        "minimum_miou": 0.933647,
    },
    "miou_secondary": {
        "minimum_matched_targets": 187,
        "minimum_pd": 187 / TARGET_COUNT,
        "maximum_fa": 1e-6,
        "minimum_miou": 0.946542,
    },
}
PAIRED_GATE = {
    "minimum_non_inferior_budget_count": 4,
    "minimum_strictly_better_budget_count": 1,
    "budget_count": 5,
}
SIX_COMPONENT_NAMES = (
    "pd_primary_absolute",
    "miou_secondary_absolute",
    "pd_primary_v4_vs_v1",
    "miou_secondary_v4_vs_v1",
    "pd_primary_v4_vs_v2",
    "miou_secondary_v4_vs_v2",
)

V4_SOURCE_LOCK_KEY = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock"
)
V4_RUN_ID_PREFIX = "tpd-ner-v8-mprs-dch-v4-tail-aware-exact:"
V4_RELAY_VERSION = "v4_tail_aware_persistent_post_center_dch"
V4_DC_SCOPE = (
    "post_centering_ner_gate_offset_not_tokenizer_mprs_dch"
)
V4_TAIL_THRESHOLDS = {"4": 1.5, "3": 2.0, "2": 2.5}
V4_SOURCE_LOCK = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock.json"
)
V4_SOURCE_LOCK_SHA256 = (
    "90dd24dfeef2d46c820fb5c89a899cec1961a7e718053f16395e256b3c27ccf3"
)
V4_RESULT_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1"
)
V4_RUN_DIR = (
    V4_RESULT_ROOT
    / DATASET
    / V4_ON_VARIANT
    / "seed_42_formal800_exact_v4_tail_aware_seed42"
)
DEFAULT_V4_SWEEPS = {
    "best.pth.tar": V4_RUN_DIR / "pd_fa_sweep_best.pth.json",
    "best_miou.pth.tar": (
        V4_RUN_DIR / "pd_fa_sweep_best_miou.pth.json"
    ),
}
COMPARISON_DIR = V4_RESULT_ROOT / DATASET / "comparison"
JSON_OUTPUT = (
    COMPARISON_DIR
    / "tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_comparison.json"
)
MARKDOWN_OUTPUT = (
    COMPARISON_DIR
    / "tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_comparison.md"
)
COMPLETE_MARKER = COMPARISON_DIR / "POSTPROCESS_COMPLETE.json"

HISTORICAL_AGGREGATE = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v8_mprs_dch_v3_exact_v1/"
    "NUDT-SIRST/comparison_selection_contract_repair_v1/"
    "tpd_ner_v8_mprs_dch_v3_formal800_comparison_"
    "selection_contract_repair_v1.json"
)
HISTORICAL_AGGREGATE_SHA256 = (
    "68e36d8afc1620821b61cae76138d90b9ddeb8f2143e76a307c76164c63cb711"
)
HISTORICAL_MARKER = (
    HISTORICAL_AGGREGATE.parent
    / "POSTPROCESS_COMPLETE_SELECTION_CONTRACT_REPAIR_V1.json"
)
HISTORICAL_MARKER_SHA256 = (
    "f94ab111f5129ea9a566a21eabfc7097de0d4958e35e97c5bbec32283dacb0ed"
)
HISTORICAL_AGGREGATE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_posttraining_aggregate_v1"
)
HISTORICAL_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_postprocess_complete_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".json" or path.name == "metrics.jsonl":
        raise ValueError(f"final evidence input is not a JSON sweep: {path}")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} is not an integer")
    return int(value)


def _close(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    return math.isclose(
        _finite_number(left, "left comparison value"),
        _finite_number(right, "right comparison value"),
        rel_tol=0.0,
        abs_tol=atol,
    )


def _normalize_thresholds(value: Any, label: str) -> dict[str, float]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    normalized = {str(key): _finite_number(item, label) for key, item in value.items()}
    _require(normalized == V4_TAIL_THRESHOLDS, f"{label} differs")
    return normalized


def normalize_fixed(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    target_count = _integer(value.get("target_count"), f"{label} target_count")
    matched = _integer(
        value.get("matched_target_count"),
        f"{label} matched_target_count",
    )
    tiny_total = _integer(
        value.get("tiny_target_count"),
        f"{label} tiny_target_count",
    )
    tiny_matched = _integer(
        value.get("matched_tiny_target_count"),
        f"{label} matched_tiny_target_count",
    )
    _require(target_count == TARGET_COUNT, f"{label} target_count differs")
    _require(
        0 <= matched <= target_count,
        f"{label} matched target count is invalid",
    )
    _require(tiny_total == TINY_TARGET_COUNT, f"{label} tiny count differs")
    _require(
        0 <= tiny_matched <= tiny_total,
        f"{label} tiny matched count is invalid",
    )
    threshold = _finite_number(value.get("threshold"), f"{label} threshold")
    pd = _finite_number(value.get("pd"), f"{label} pd")
    tiny_pd = _finite_number(value.get("tiny_pd"), f"{label} tiny_pd")
    fa = _finite_number(value.get("fa"), f"{label} fa")
    miou = _finite_number(value.get("miou"), f"{label} miou")
    false_objects = _finite_number(
        value.get("false_objects_per_image"),
        f"{label} false_objects_per_image",
    )
    _require(_close(threshold, 0.5), f"{label} is not threshold 0.5")
    _require(_close(pd, matched / target_count), f"{label} Pd/count differs")
    _require(
        _close(tiny_pd, tiny_matched / tiny_total),
        f"{label} tiny-Pd/count differs",
    )
    _require(0.0 <= pd <= 1.0, f"{label} Pd is out of range")
    _require(0.0 <= tiny_pd <= 1.0, f"{label} tiny-Pd is out of range")
    _require(0.0 <= miou <= 1.0, f"{label} mIoU is out of range")
    _require(fa >= 0.0, f"{label} Fa is negative")
    _require(false_objects >= 0.0, f"{label} false objects is negative")
    result: dict[str, Any] = {
        "threshold": threshold,
        "pd": pd,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": false_objects,
        "target_count": target_count,
        "matched_target_count": matched,
        "tiny_pd": tiny_pd,
        "tiny_target_count": tiny_total,
        "matched_tiny_target_count": tiny_matched,
    }
    for field in ("niou", "pixel_precision", "pixel_recall", "pixel_f1"):
        if field in value:
            metric = _finite_number(value[field], f"{label} {field}")
            _require(0.0 <= metric <= 1.0, f"{label} {field} out of range")
            result[field] = metric
    return result


def normalize_budgets(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    _require(set(value) == set(BUDGET_KEYS), f"{label} budget keys differ")
    result: dict[str, dict[str, Any]] = {}
    for key in BUDGET_KEYS:
        point = value[key]
        _require(isinstance(point, Mapping), f"{label}:{key} is missing")
        target_count = _integer(
            point.get("target_count"),
            f"{label}:{key} target_count",
        )
        matched = _integer(
            point.get("matched_target_count"),
            f"{label}:{key} matched_target_count",
        )
        pd = _finite_number(point.get("pd"), f"{label}:{key} pd")
        fa = _finite_number(point.get("fa"), f"{label}:{key} fa")
        threshold = _finite_number(
            point.get("threshold"),
            f"{label}:{key} threshold",
        )
        _require(target_count == TARGET_COUNT, f"{label}:{key} total differs")
        _require(
            0 <= matched <= target_count,
            f"{label}:{key} matched count is invalid",
        )
        _require(
            _close(pd, matched / target_count),
            f"{label}:{key} Pd/count differs",
        )
        _require(0.0 <= fa <= BUDGET_BY_KEY[key] + 1e-15, f"{label}:{key} exceeds Fa")
        _require(0.0 <= threshold <= 1.0, f"{label}:{key} threshold differs")
        normalized: dict[str, Any] = {
            "threshold": threshold,
            "pd": pd,
            "fa": fa,
            "target_count": target_count,
            "matched_target_count": matched,
        }
        for field in (
            "miou",
            "false_objects_per_image",
            "tiny_pd",
            "tiny_target_count",
            "matched_tiny_target_count",
        ):
            if field in point:
                normalized[field] = (
                    _integer(point[field], f"{label}:{key} {field}")
                    if field.endswith("_count")
                    else _finite_number(point[field], f"{label}:{key} {field}")
                )
        if "miou" in normalized:
            _require(
                0.0 <= normalized["miou"] <= 1.0,
                f"{label}:{key} mIoU is out of range",
            )
        if "false_objects_per_image" in normalized:
            _require(
                normalized["false_objects_per_image"] >= 0.0,
                f"{label}:{key} false objects is negative",
            )
        tiny_fields = {
            "tiny_pd",
            "tiny_target_count",
            "matched_tiny_target_count",
        }
        present_tiny = tiny_fields & set(normalized)
        _require(
            not present_tiny or present_tiny == tiny_fields,
            f"{label}:{key} tiny metrics are incomplete",
        )
        if present_tiny:
            _require(
                normalized["tiny_target_count"] == TINY_TARGET_COUNT,
                f"{label}:{key} tiny count differs",
            )
            _require(
                _close(
                    normalized["tiny_pd"],
                    normalized["matched_tiny_target_count"] / TINY_TARGET_COUNT,
                ),
                f"{label}:{key} tiny-Pd/count differs",
            )
        result[key] = normalized
    return result


def _normalize_v4_sweep_point(value: Any, *, label: str) -> dict[str, Any]:
    """Normalize one evaluator point and re-derive every count-backed metric."""

    _require(isinstance(value, Mapping), f"{label} is missing")
    missing = [field for field in V4_SWEEP_POINT_FIELDS if field not in value]
    _require(not missing, f"{label} lacks sweep fields: {missing}")
    target_count = _integer(value["target_count"], f"{label} target_count")
    matched = _integer(
        value["matched_target_count"], f"{label} matched_target_count"
    )
    tiny_count = _integer(
        value["tiny_target_count"], f"{label} tiny_target_count"
    )
    tiny_matched = _integer(
        value["matched_tiny_target_count"],
        f"{label} matched_tiny_target_count",
    )
    predicted = _integer(
        value["predicted_object_count"],
        f"{label} predicted_object_count",
    )
    unmatched = _integer(
        value["unmatched_predicted_object_count"],
        f"{label} unmatched_predicted_object_count",
    )
    valid_pixels = _integer(
        value["valid_pixel_count"], f"{label} valid_pixel_count"
    )
    _require(target_count == TARGET_COUNT, f"{label} target count differs")
    _require(tiny_count == TINY_TARGET_COUNT, f"{label} tiny count differs")
    _require(valid_pixels == VALID_PIXEL_COUNT, f"{label} pixel count differs")
    _require(0 <= matched <= target_count, f"{label} matched count invalid")
    _require(
        0 <= tiny_matched <= tiny_count,
        f"{label} tiny matched count invalid",
    )
    _require(
        0 <= unmatched <= predicted,
        f"{label} unmatched prediction count invalid",
    )
    result = {
        field: (
            _integer(value[field], f"{label} {field}")
            if field in {
                "target_count",
                "matched_target_count",
                "tiny_target_count",
                "matched_tiny_target_count",
                "predicted_object_count",
                "unmatched_predicted_object_count",
                "valid_pixel_count",
            }
            else _finite_number(value[field], f"{label} {field}")
        )
        for field in V4_SWEEP_POINT_FIELDS
    }
    for field in ("threshold", "pd", "fa", "miou", "tiny_pd"):
        _require(
            0.0 <= result[field] <= 1.0,
            f"{label} {field} is out of range",
        )
    _require(
        result["false_objects_per_image"] >= 0.0,
        f"{label} false objects is negative",
    )
    _require(
        _close(result["pd"], matched / target_count),
        f"{label} Pd/count differs",
    )
    _require(
        _close(result["tiny_pd"], tiny_matched / tiny_count),
        f"{label} tiny-Pd/count differs",
    )
    _require(
        _close(
            result["false_objects_per_image"],
            unmatched / VALIDATION_COUNT,
        ),
        f"{label} false-objects/count differs",
    )
    return result


def _validate_v4_raw_sweep(
    payload: Mapping[str, Any],
    fixed: Mapping[str, Any],
    budgets: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    """Recompute fixed and Pd@Fa selections from raw evaluator points."""

    raw_points = payload.get("points")
    _require(isinstance(raw_points, list) and raw_points, f"{label} points missing")
    points = [
        _normalize_v4_sweep_point(point, label=f"{label} points[{index}]")
        for index, point in enumerate(raw_points)
    ]
    thresholds = [point["threshold"] for point in points]
    _require(
        thresholds == sorted(thresholds) and len(thresholds) == len(set(thresholds)),
        f"{label} thresholds are not sorted and unique",
    )
    fixed_points = [point for point in points if point["threshold"] == 0.5]
    _require(len(fixed_points) == 1, f"{label} fixed point count differs")
    for field in FINAL_COVERAGE_FIELDS:
        _require(
            fixed_points[0][field] == fixed[field],
            f"{label} fixed/raw point differs: {field}",
        )
    for key, budget in zip(BUDGET_KEYS, FA_BUDGETS):
        eligible = [point for point in points if point["fa"] <= budget]
        _require(bool(eligible), f"{label}:{key} has no eligible raw point")
        expected = max(
            eligible,
            key=lambda point: (
                point["pd"],
                -point["fa"],
                point["tiny_pd"],
                point["miou"],
                -abs(point["threshold"] - 0.5),
            ),
        )
        for field in FINAL_COVERAGE_FIELDS:
            _require(
                expected[field] == budgets[key][field],
                f"{label}:{key} is not the best raw sweep point: {field}",
            )
    provenance = payload.get("threshold_provenance")
    _require(isinstance(provenance, Mapping), f"{label} provenance missing")
    _require(
        provenance.get("total_unique_threshold_count") == len(points),
        f"{label} provenance point count differs",
    )
    _require(
        provenance.get("score_count") == VALID_PIXEL_COUNT,
        f"{label} provenance score count differs",
    )
    by_threshold = {point["threshold"]: point for point in points}
    _require(
        LAST_FLOAT32_BELOW_ONE in by_threshold
        and UPPER_BOUNDARY_THRESHOLD in by_threshold,
        f"{label} closed-interval endpoints are missing",
    )
    upper = by_threshold[UPPER_BOUNDARY_THRESHOLD]
    for field, expected in {
        "pd": 0.0,
        "fa": 0.0,
        "matched_target_count": 0,
        "predicted_object_count": 0,
        "unmatched_predicted_object_count": 0,
    }.items():
        _require(upper[field] == expected, f"{label} upper endpoint differs: {field}")


def _metric_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fixed_threshold_0_5": normalize_fixed(
            row.get("fixed_threshold_0_5"),
            label="row fixed_threshold_0_5",
        ),
        "pd_at_fa_budget": normalize_budgets(
            row.get("pd_at_fa_budget"),
            label="row pd_at_fa_budget",
        ),
    }


def _same_metric_projection(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_value = _metric_projection(left)
    right_value = _metric_projection(right)
    if left_value["fixed_threshold_0_5"] != right_value[
        "fixed_threshold_0_5"
    ]:
        return False
    required_budget_fields = (
        "threshold",
        "pd",
        "fa",
        "target_count",
        "matched_target_count",
    )
    for key in BUDGET_KEYS:
        for field in required_budget_fields:
            if (
                left_value["pd_at_fa_budget"][key][field]
                != right_value["pd_at_fa_budget"][key][field]
            ):
                return False
    return True


def _checkpoint_name(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} is missing")
    return Path(value).name


def _validate_bound_file_entry(
    entry: Any,
    *,
    label: str,
) -> dict[str, str]:
    _require(isinstance(entry, Mapping), f"{label} binding is missing")
    path_value = entry.get("path")
    digest = entry.get("sha256")
    _require(isinstance(path_value, str) and path_value, f"{label} path missing")
    _require(_is_hex_sha256(digest), f"{label} SHA-256 is invalid")
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    observed = sha256_file(path)
    _require(observed == digest, f"{label} SHA-256 differs")
    relative = entry.get("relative_path")
    if relative is not None:
        _require(
            (REPO_ROOT / str(relative)).resolve() == path,
            f"{label} relative path differs",
        )
    return {"path": str(path), "sha256": observed}


def _validate_history_row_against_sweep(
    row: Mapping[str, Any],
    sweep_path: Path,
    *,
    variant: str,
    checkpoint: str,
) -> dict[str, Any]:
    payload = load_json(sweep_path)
    role = CHECKPOINT_ROLES[checkpoint]
    for field, expected in {
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "validation_count": VALIDATION_COUNT,
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "official_test_accessed": False,
        "checkpoint_role": role,
    }.items():
        _require(
            payload.get(field) == expected,
            f"historical {variant}:{checkpoint} differs: {field}",
        )
    expected_raw_variant = "original" if variant == BASELINE_VARIANT else variant
    _require(
        payload.get("variant") == expected_raw_variant,
        f"historical {variant}:{checkpoint} raw variant differs",
    )
    _require(
        _checkpoint_name(payload.get("checkpoint"), "historical checkpoint")
        == checkpoint,
        f"historical {variant}:{checkpoint} checkpoint differs",
    )
    checkpoint_sha = payload.get("checkpoint_sha256")
    _require(
        _is_hex_sha256(checkpoint_sha),
        f"historical {variant}:{checkpoint} checkpoint SHA invalid",
    )
    checkpoint_path = Path(str(payload["checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(str(payload["run_directory"])) / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    _require(
        sha256_file(checkpoint_path) == checkpoint_sha,
        f"historical {variant}:{checkpoint} checkpoint changed",
    )
    raw_row = {
        "fixed_threshold_0_5": payload.get("fixed_threshold_0_5"),
        "pd_at_fa_budget": payload.get("best_points_under_fa_budget"),
    }
    _require(
        _same_metric_projection(row, raw_row),
        f"historical {variant}:{checkpoint} aggregate/raw metrics differ",
    )
    for field, expected in {
        "variant": variant,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "checkpoint_sha256": checkpoint_sha,
    }.items():
        _require(
            row.get(field) == expected,
            f"historical {variant}:{checkpoint} row differs: {field}",
        )
    return {
        "path": str(Path(sweep_path).resolve()),
        "sha256": sha256_file(sweep_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
    }


def validate_historical_authority(
    *,
    aggregate_path: Path = HISTORICAL_AGGREGATE,
    marker_path: Path = HISTORICAL_MARKER,
    aggregate_sha256: str = HISTORICAL_AGGREGATE_SHA256,
    marker_sha256: str = HISTORICAL_MARKER_SHA256,
) -> dict[str, Any]:
    """Validate the frozen baseline/V1/V2/V3 authority and its raw sweeps."""

    aggregate_path = Path(aggregate_path).resolve()
    marker_path = Path(marker_path).resolve()
    _require(
        sha256_file(aggregate_path) == aggregate_sha256,
        "historical authority aggregate SHA-256 differs",
    )
    _require(
        sha256_file(marker_path) == marker_sha256,
        "historical authority marker SHA-256 differs",
    )
    report = load_json(aggregate_path)
    marker = load_json(marker_path)
    for field, expected in {
        "schema": HISTORICAL_AGGREGATE_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
        "row_count": 8,
    }.items():
        _require(report.get(field) == expected, f"historical report differs: {field}")
    _require(
        marker.get("schema") == HISTORICAL_MARKER_SCHEMA,
        "historical marker schema differs",
    )
    _require(marker.get("status") == "complete", "historical marker incomplete")
    outputs = marker.get("outputs")
    _require(isinstance(outputs, Mapping), "historical marker outputs missing")
    _require(
        outputs.get(aggregate_path.name) == aggregate_sha256,
        "historical marker does not bind aggregate",
    )
    comparison = report.get("comparison_contract")
    _require(isinstance(comparison, Mapping), "historical comparison contract missing")
    selection_repair = comparison.get("selection_contract_repair")
    _require(
        isinstance(selection_repair, Mapping)
        and selection_repair.get(
            "each_variant_uses_own_selected_checkpoints"
        )
        is True,
        "historical checkpoint ownership is not proven",
    )
    bindings = report.get("bindings")
    _require(isinstance(bindings, Mapping), "historical bindings missing")
    sweep_bindings = bindings.get("sweeps")
    _require(isinstance(sweep_bindings, Mapping), "historical sweep bindings missing")
    rows_value = report.get("rows")
    _require(isinstance(rows_value, list), "historical rows missing")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in rows_value:
        _require(isinstance(raw_row, Mapping), "historical row is not an object")
        variant = raw_row.get("variant")
        checkpoint = raw_row.get("checkpoint")
        _require(
            variant in HISTORICAL_VARIANTS and checkpoint in CHECKPOINTS,
            "historical row identity differs",
        )
        key = (str(variant), str(checkpoint))
        _require(key not in rows, f"duplicate historical row: {key}")
        rows[key] = copy.deepcopy(dict(raw_row))
    expected_rows = {
        (variant, checkpoint)
        for variant in HISTORICAL_VARIANTS
        for checkpoint in CHECKPOINTS
    }
    _require(set(rows) == expected_rows, "historical eight-row matrix differs")

    verified_sweeps: dict[str, Any] = {}
    for variant, checkpoint in sorted(expected_rows):
        binding_key = f"{variant}:{checkpoint}"
        entry = sweep_bindings.get(binding_key)
        verified_entry = _validate_bound_file_entry(
            entry,
            label=f"historical sweep {binding_key}",
        )
        details = _validate_history_row_against_sweep(
            rows[(variant, checkpoint)],
            Path(verified_entry["path"]),
            variant=variant,
            checkpoint=checkpoint,
        )
        _require(
            details["sha256"] == verified_entry["sha256"],
            f"historical sweep binding differs: {binding_key}",
        )
        verified_sweeps[binding_key] = details

    for name in (
        "v3_evaluator",
        "v2_evaluator",
        "v1_off_evaluator",
        "baseline_evaluator",
        "postprocess",
        "v3_training_source_lock",
        "v3_acceptance_source_lock",
        "v2_training_source_lock",
        "v2_acceptance_source_lock",
    ):
        path_value = bindings.get(name)
        digest = bindings.get(f"{name}_sha256")
        if path_value is None and name.startswith("v2_"):
            upstream = bindings.get("upstream_v2_and_v1_source_bindings")
            if isinstance(upstream, Mapping):
                path_value = upstream.get(name)
                digest = upstream.get(f"{name}_sha256")
        _require(
            isinstance(path_value, str) and _is_hex_sha256(digest),
            f"historical source binding missing: {name}",
        )
        _require(
            sha256_file(Path(path_value)) == digest,
            f"historical source binding changed: {name}",
        )

    return {
        "report": report,
        "rows": rows,
        "binding": {
            "aggregate": {
                "path": str(aggregate_path),
                "sha256": aggregate_sha256,
            },
            "completion_marker": {
                "path": str(marker_path),
                "sha256": marker_sha256,
            },
            "sweeps": verified_sweeps,
        },
    }


def _validate_final_coverage(
    payload: Mapping[str, Any],
    fixed: Mapping[str, Any],
    budgets: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    coverage = payload.get("final_metric_coverage")
    _require(isinstance(coverage, Mapping), f"{label} final metric coverage missing")
    _require(
        coverage.get("schema") == V4_FINAL_METRIC_COVERAGE_SCHEMA,
        f"{label} final metric coverage schema differs",
    )
    _require(
        coverage.get("required_metrics")
        == ["pd", "fa", "miou", "false_objects_per_image", "tiny_pd"],
        f"{label} required final metrics differ",
    )
    coverage_budgets = coverage.get("fa_budgets")
    _require(
        isinstance(coverage_budgets, list)
        and len(coverage_budgets) == len(FA_BUDGETS)
        and all(
            _close(observed, expected)
            for observed, expected in zip(coverage_budgets, FA_BUDGETS)
        ),
        f"{label} final coverage budgets differ",
    )
    for field, expected in {
        "fixed_threshold_complete": True,
        "fa_budget_curve_complete": True,
        "official_test_accessed": False,
    }.items():
        _require(
            coverage.get(field) == expected,
            f"{label} final coverage differs: {field}",
        )
    fixed_coverage = coverage.get("fixed_threshold_0_5")
    _require(
        isinstance(fixed_coverage, Mapping)
        and set(fixed_coverage) == set(FINAL_COVERAGE_FIELDS),
        f"{label} fixed coverage fields differ",
    )
    for field in FINAL_COVERAGE_FIELDS:
        _require(
            fixed_coverage[field] == fixed[field],
            f"{label} fixed coverage differs: {field}",
        )
    budget_coverage = coverage.get("fa_budget_points")
    _require(
        isinstance(budget_coverage, Mapping)
        and set(budget_coverage) == set(BUDGET_KEYS),
        f"{label} budget coverage differs",
    )
    for key in BUDGET_KEYS:
        point = budget_coverage[key]
        _require(
            isinstance(point, Mapping)
            and set(point) == set(FINAL_COVERAGE_FIELDS),
            f"{label}:{key} coverage fields differ",
        )
        expected = budgets[key]
        for field in FINAL_COVERAGE_FIELDS:
            _require(
                point[field] == expected[field],
                f"{label}:{key} coverage differs: {field}",
            )


def _validate_v4_identity(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    source_lock_path: Path,
    source_lock_sha256: str,
    label: str,
) -> dict[str, Any]:
    identity = payload.get("run_identity")
    _require(isinstance(identity, Mapping), f"{label} run identity missing")
    run_id = identity.get("run_id")
    _require(
        isinstance(run_id, str) and run_id.startswith(V4_RUN_ID_PREFIX),
        f"{label} run id is not V4",
    )
    for field, expected in {
        "variant": V4_ON_VARIANT,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
    }.items():
        _require(identity.get(field) == expected, f"{label} identity differs: {field}")
    source_locks = identity.get("source_locks")
    _require(
        isinstance(source_locks, Mapping)
        and set(source_locks) == {V4_SOURCE_LOCK_KEY, "training_data"},
        f"{label} source-lock identity differs",
    )
    _require(
        source_locks.get(V4_SOURCE_LOCK_KEY) == source_lock_sha256,
        f"{label} source-lock SHA differs",
    )
    _require(
        source_locks.get("training_data") == TRAINING_DATA_SHA256,
        f"{label} training-data SHA differs",
    )
    training = identity.get("training_contract")
    determinism = training.get("determinism") if isinstance(training, Mapping) else None
    _require(isinstance(determinism, Mapping), f"{label} determinism identity missing")
    for field, expected in {
        "relay_version": V4_RELAY_VERSION,
        "required_control": V1_OFF_VARIANT,
        "paired_gate_predecessor": V2_ON_VARIANT,
        "structural_predecessor": V3_ON_VARIANT,
        "ner_dc_offset_support_scope": V4_DC_SCOPE,
        "dc_support_mode": "complement_tail",
        "dc_support_formula_stage4": "1",
        "dc_support_formula_stage3_2": "1-P",
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "fresh_training": True,
        "v3_warm_start": False,
    }.items():
        _require(
            determinism.get(field) == expected,
            f"{label} determinism differs: {field}",
        )
    _normalize_thresholds(
        determinism.get("tail_z_thresholds"),
        f"{label} tail thresholds",
    )

    source_identity = payload.get("source_checkpoint_identity")
    _require(
        isinstance(source_identity, Mapping),
        f"{label} source checkpoint identity missing",
    )
    for field, expected in {
        "variant": V4_ON_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "required_control": V1_OFF_VARIANT,
        "paired_gate_predecessor": V2_ON_VARIANT,
        "structural_predecessor": V3_ON_VARIANT,
        "ner_dc_offset_support_scope": V4_DC_SCOPE,
        "dc_support_mode": "complement_tail",
        "dc_support_formula_stage3_2": "1-P",
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
    }.items():
        _require(
            source_identity.get(field) == expected,
            f"{label} checkpoint identity differs: {field}",
        )
    _normalize_thresholds(
        source_identity.get("tail_z_thresholds"),
        f"{label} checkpoint thresholds",
    )
    evaluated = payload.get("evaluated_checkpoint_identity")
    _require(
        isinstance(evaluated, Mapping),
        f"{label} evaluated checkpoint identity missing",
    )
    for field, expected in {
        "filename": checkpoint,
        "role": CHECKPOINT_ROLES[checkpoint],
        "sha256": checkpoint_sha256,
    }.items():
        _require(
            evaluated.get(field) == expected,
            f"{label} evaluated identity differs: {field}",
        )

    binding = payload.get("evaluation_source_binding")
    _require(
        isinstance(binding, Mapping),
        f"{label} evaluation source binding missing",
    )
    required_bindings = (
        "training_source_lock",
        "evaluator",
        "shared_metric_core",
        "closed_interval_core",
        "determinism_core",
    )
    verified_bindings: dict[str, Any] = {}
    for name in required_bindings:
        verified_bindings[name] = _validate_bound_file_entry(
            binding.get(name),
            label=f"{label} source {name}",
        )
    _require(
        Path(verified_bindings["training_source_lock"]["path"]).resolve()
        == Path(source_lock_path).resolve(),
        f"{label} training source-lock path differs",
    )
    _require(
        verified_bindings["training_source_lock"]["sha256"]
        == source_lock_sha256,
        f"{label} training source-lock binding differs",
    )
    return {
        "run_id": run_id,
        "source_bindings": verified_bindings,
        "run_identity": copy.deepcopy(dict(identity)),
    }


def validate_v4_sweep(
    path: Path,
    *,
    checkpoint: str,
    expected_run_dir: Path = V4_RUN_DIR,
    source_lock_path: Path = V4_SOURCE_LOCK,
    source_lock_sha256: str = V4_SOURCE_LOCK_SHA256,
) -> dict[str, Any]:
    """Validate one V4 checkpoint-owned evaluation and normalize its row."""

    _require(checkpoint in CHECKPOINTS, f"unknown checkpoint: {checkpoint}")
    path = Path(path).resolve()
    _require(
        path.name
        == (
            "pd_fa_sweep_best.pth.json"
            if checkpoint == "best.pth.tar"
            else "pd_fa_sweep_best_miou.pth.json"
        ),
        f"V4 {checkpoint} sweep filename differs",
    )
    _require(
        sha256_file(Path(source_lock_path)) == source_lock_sha256,
        "V4 training source-lock SHA differs",
    )
    payload = load_json(path)
    label = f"V4 {checkpoint}"
    for field, expected in {
        "schema": V4_SWEEP_SCHEMA,
        "variant": V4_ON_VARIANT,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "validation_count": VALIDATION_COUNT,
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "official_test_accessed": False,
        "match_radius": 3.0,
        "tiny_area": 9,
        "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
        "artifact_identity_preflight_passed": True,
    }.items():
        _require(payload.get(field) == expected, f"{label} differs: {field}")
    _require(
        _checkpoint_name(payload.get("checkpoint"), f"{label} checkpoint")
        == checkpoint,
        f"{label} checkpoint differs",
    )
    checkpoint_epoch = _integer(
        payload.get("checkpoint_epoch"),
        f"{label} checkpoint epoch",
    )
    _require(
        1 <= checkpoint_epoch <= EXPECTED_EPOCHS,
        f"{label} checkpoint epoch out of range",
    )
    run_dir = Path(str(payload.get("run_directory"))).resolve()
    _require(
        run_dir == Path(expected_run_dir).resolve(),
        f"{label} run directory differs",
    )
    _require(path.parent == run_dir, f"{label} is outside its run directory")
    checkpoint_path = run_dir / checkpoint
    checkpoint_sha = payload.get("checkpoint_sha256")
    _require(_is_hex_sha256(checkpoint_sha), f"{label} checkpoint SHA invalid")
    _require(
        sha256_file(checkpoint_path) == checkpoint_sha,
        f"{label} checkpoint SHA differs",
    )

    fixed = normalize_fixed(
        payload.get("fixed_threshold_0_5"),
        label=f"{label} fixed_threshold_0_5",
    )
    budgets = normalize_budgets(
        payload.get("best_points_under_fa_budget"),
        label=f"{label} best_points_under_fa_budget",
    )
    _validate_v4_raw_sweep(payload, fixed, budgets, label=label)
    _validate_final_coverage(payload, fixed, budgets, label=label)
    threshold_configuration = payload.get("threshold_configuration")
    _require(
        isinstance(threshold_configuration, Mapping),
        f"{label} threshold configuration missing",
    )
    observed_budgets = threshold_configuration.get("fa_budgets")
    _require(
        isinstance(observed_budgets, list)
        and len(observed_budgets) == len(FA_BUDGETS)
        and all(
            _close(observed, expected)
            for observed, expected in zip(observed_budgets, FA_BUDGETS)
        ),
        f"{label} threshold Fa budgets differ",
    )
    audit = payload.get("audit")
    _require(isinstance(audit, Mapping), f"{label} audit missing")
    for field, expected in {
        "expected_epochs": EXPECTED_EPOCHS,
        "metrics_event_count": EXPECTED_EPOCHS,
        "metrics_epoch_range": [1, EXPECTED_EPOCHS],
        "summary_status": "complete",
        "selection_source": "internal_validation_only",
    }.items():
        _require(audit.get(field) == expected, f"{label} audit differs: {field}")
    integrity = audit.get("integrity_checks_passed")
    _require(
        isinstance(integrity, Mapping)
        and integrity
        and all(value is True for value in integrity.values()),
        f"{label} evaluator integrity checks are incomplete",
    )
    identity = _validate_v4_identity(
        payload,
        checkpoint=checkpoint,
        checkpoint_sha256=str(checkpoint_sha),
        source_lock_path=Path(source_lock_path),
        source_lock_sha256=source_lock_sha256,
        label=label,
    )
    evaluator_contract = payload.get("evaluator_contract")
    _require(
        isinstance(evaluator_contract, Mapping),
        f"{label} evaluator contract missing",
    )
    for field, expected in {
        "dataset": DATASET,
        "formal_variant": V4_ON_VARIANT,
        "required_control": V1_OFF_VARIANT,
        "paired_gate_predecessor": V2_ON_VARIANT,
        "structural_predecessor": V3_ON_VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "fixed_threshold": 0.5,
        "official_test_accessed": False,
        "dc_support_mode": "complement_tail",
        "dc_support_formula_stage4": "1",
        "dc_support_formula_stage3_2": "1-P",
    }.items():
        _require(
            evaluator_contract.get(field) == expected,
            f"{label} evaluator contract differs: {field}",
        )
    contract_budgets = evaluator_contract.get("fa_budgets")
    _require(
        isinstance(contract_budgets, list)
        and len(contract_budgets) == len(FA_BUDGETS)
        and all(
            _close(observed, expected)
            for observed, expected in zip(contract_budgets, FA_BUDGETS)
        ),
        f"{label} evaluator contract budgets differ",
    )
    return {
        "source": "v4_formal_sweep",
        "variant": V4_ON_VARIANT,
        "checkpoint": checkpoint,
        "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "run_directory": str(run_dir),
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "absolute_gate": None,
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "sweep_binding": {
            "path": str(path),
            "sha256": sha256_file(path),
        },
        "checkpoint_binding": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
        },
        "evaluation_source_binding": identity["source_bindings"],
        "run_id": identity["run_id"],
    }


def absolute_gate_assessment(
    row: Mapping[str, Any],
    *,
    role_name: str,
) -> dict[str, Any]:
    _require(role_name in FIXED_GATE, f"unknown absolute role: {role_name}")
    fixed = normalize_fixed(
        row.get("fixed_threshold_0_5"),
        label=f"{role_name} fixed",
    )
    budgets = normalize_budgets(
        row.get("pd_at_fa_budget"),
        label=f"{role_name} budgets",
    )
    contract = FIXED_GATE[role_name]
    fixed_checks = {
        "matched_targets": (
            fixed["matched_target_count"]
            >= contract["minimum_matched_targets"]
        ),
        "pd": fixed["pd"] >= contract["minimum_pd"],
        "fa": fixed["fa"] <= contract["maximum_fa"],
        "miou": fixed["miou"] >= contract["minimum_miou"],
    }
    budget_checks: dict[str, Any] = {}
    for key in BUDGET_KEYS:
        point = budgets[key]
        minimum_matched = BUDGET_MINIMUM_MATCHED[key]
        minimum_pd = minimum_matched / TARGET_COUNT
        checks = {
            "matched_targets": (
                point["matched_target_count"] >= minimum_matched
            ),
            "pd": point["pd"] >= minimum_pd,
        }
        budget_checks[key] = {
            "observed_matched_target_count": point[
                "matched_target_count"
            ],
            "observed_target_count": point["target_count"],
            "observed_pd": point["pd"],
            "observed_fa": point["fa"],
            "required_matched_target_count": minimum_matched,
            "required_pd": minimum_pd,
            "checks": checks,
            "passed": all(checks.values()),
        }
    fixed_passed = all(fixed_checks.values())
    budgets_passed = all(
        budget_checks[key]["passed"] for key in BUDGET_KEYS
    )
    return {
        "role": role_name,
        "fixed_threshold_contract": copy.deepcopy(contract),
        "fixed_threshold_observed": {
            "matched_target_count": fixed["matched_target_count"],
            "target_count": fixed["target_count"],
            "pd": fixed["pd"],
            "fa": fixed["fa"],
            "miou": fixed["miou"],
        },
        "fixed_threshold_checks": fixed_checks,
        "fixed_threshold_passed": fixed_passed,
        "budget_checks": budget_checks,
        "all_fa_budgets_passed": budgets_passed,
        "absolute_checkpoint_gate_passed": (
            fixed_passed and budgets_passed
        ),
    }


def paired_gate_assessment(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_variant: str,
) -> dict[str, Any]:
    _require(
        reference_variant in (V1_OFF_VARIANT, V2_ON_VARIANT),
        "paired reference is not V1 or V2",
    )
    _require(
        reference.get("variant") == reference_variant,
        "paired reference variant differs",
    )
    _require(
        candidate.get("variant") == V4_ON_VARIANT,
        "paired candidate variant differs",
    )
    _require(
        reference.get("checkpoint_role") == candidate.get("checkpoint_role"),
        "paired checkpoint roles differ",
    )
    reference_budgets = normalize_budgets(
        reference.get("pd_at_fa_budget"),
        label="paired reference budgets",
    )
    candidate_budgets = normalize_budgets(
        candidate.get("pd_at_fa_budget"),
        label="paired candidate budgets",
    )
    comparisons: dict[str, Any] = {}
    non_inferior = 0
    strictly_better = 0
    for key in BUDGET_KEYS:
        reference_point = reference_budgets[key]
        candidate_point = candidate_budgets[key]
        reference_count = reference_point["matched_target_count"]
        candidate_count = candidate_point["matched_target_count"]
        no_worse = candidate_count >= reference_count
        better = candidate_count > reference_count
        non_inferior += int(no_worse)
        strictly_better += int(better)
        comparisons[key] = {
            "reference_matched_target_count": reference_count,
            "candidate_matched_target_count": candidate_count,
            "reference_pd": reference_point["pd"],
            "candidate_pd": candidate_point["pd"],
            "reference_achieved_fa": reference_point["fa"],
            "candidate_achieved_fa": candidate_point["fa"],
            "candidate_non_inferior": no_worse,
            "candidate_strictly_better": better,
        }
    passed = (
        non_inferior
        >= PAIRED_GATE["minimum_non_inferior_budget_count"]
        and strictly_better
        >= PAIRED_GATE["minimum_strictly_better_budget_count"]
    )
    return {
        "checkpoint_role": candidate["checkpoint_role"],
        "reference_variant": reference_variant,
        "candidate_variant": V4_ON_VARIANT,
        "comparisons": comparisons,
        "non_inferior_budget_count": non_inferior,
        "strictly_better_budget_count": strictly_better,
        "required_non_inferior_budget_count": PAIRED_GATE[
            "minimum_non_inferior_budget_count"
        ],
        "required_strictly_better_budget_count": PAIRED_GATE[
            "minimum_strictly_better_budget_count"
        ],
        "budget_count": PAIRED_GATE["budget_count"],
        "passed": passed,
    }


def comparison_delta(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        reference.get("checkpoint_role") == candidate.get("checkpoint_role"),
        "delta checkpoint roles differ",
    )
    fixed_reference = normalize_fixed(
        reference.get("fixed_threshold_0_5"),
        label="delta reference fixed",
    )
    fixed_candidate = normalize_fixed(
        candidate.get("fixed_threshold_0_5"),
        label="delta candidate fixed",
    )
    budgets_reference = normalize_budgets(
        reference.get("pd_at_fa_budget"),
        label="delta reference budgets",
    )
    budgets_candidate = normalize_budgets(
        candidate.get("pd_at_fa_budget"),
        label="delta candidate budgets",
    )
    budget_deltas = {}
    for key in BUDGET_KEYS:
        reference_point = budgets_reference[key]
        candidate_point = budgets_candidate[key]
        budget_deltas[key] = {
            "delta_matched_targets_candidate_minus_reference": (
                candidate_point["matched_target_count"]
                - reference_point["matched_target_count"]
            ),
            "delta_pd_candidate_minus_reference": (
                candidate_point["pd"] - reference_point["pd"]
            ),
            "candidate_achieved_fa": candidate_point["fa"],
            "reference_achieved_fa": reference_point["fa"],
        }
    return {
        "reference_variant": reference["variant"],
        "candidate_variant": candidate["variant"],
        "checkpoint_role": candidate["checkpoint_role"],
        "affects_six_component_gate": False,
        "fixed_threshold_0_5_delta_candidate_minus_reference": {
            "matched_targets": (
                fixed_candidate["matched_target_count"]
                - fixed_reference["matched_target_count"]
            ),
            "pd": fixed_candidate["pd"] - fixed_reference["pd"],
            "fa": fixed_candidate["fa"] - fixed_reference["fa"],
            "miou": fixed_candidate["miou"] - fixed_reference["miou"],
            "false_objects_per_image": (
                fixed_candidate["false_objects_per_image"]
                - fixed_reference["false_objects_per_image"]
            ),
        },
        "pd_at_fa_budget": budget_deltas,
    }


def build_report(
    historical: Mapping[str, Any],
    v4_rows: Mapping[str, Mapping[str, Any]],
    *,
    input_snapshot_before: Mapping[str, str],
    input_snapshot_after: Mapping[str, str],
) -> dict[str, Any]:
    _require(
        dict(input_snapshot_before) == dict(input_snapshot_after),
        "formal inputs changed during aggregation",
    )
    historical_rows = historical.get("rows")
    _require(
        isinstance(historical_rows, Mapping),
        "historical row authority missing",
    )
    _require(
        set(v4_rows) == set(CHECKPOINTS),
        "V4 two-role row matrix differs",
    )
    run_ids = {v4_rows[name].get("run_id") for name in CHECKPOINTS}
    _require(len(run_ids) == 1, "V4 sweeps refer to different run identities")
    source_bindings = {
        json.dumps(
            v4_rows[name].get("evaluation_source_binding"),
            sort_keys=True,
        )
        for name in CHECKPOINTS
    }
    _require(
        len(source_bindings) == 1,
        "V4 sweeps have different evaluation source bindings",
    )

    ordered_rows: list[dict[str, Any]] = []
    absolute_by_role: dict[str, Any] = {}
    paired_v1_by_role: dict[str, Any] = {}
    paired_v2_by_role: dict[str, Any] = {}
    delta_v3_by_role: dict[str, Any] = {}
    delta_baseline_by_role: dict[str, Any] = {}
    tiny_audit_by_role: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        role = CHECKPOINT_ROLES[checkpoint]
        role_name = ROLE_NAMES[role]
        role_rows: dict[str, dict[str, Any]] = {}
        for variant in HISTORICAL_VARIANTS:
            row = historical_rows[(variant, checkpoint)]
            role_rows[variant] = copy.deepcopy(dict(row))
        v4 = copy.deepcopy(dict(v4_rows[checkpoint]))
        role_rows[V4_ON_VARIANT] = v4
        absolute = absolute_gate_assessment(v4, role_name=role_name)
        v4["absolute_gate"] = absolute
        absolute_by_role[role_name] = absolute
        paired_v1_by_role[role_name] = paired_gate_assessment(
            role_rows[V1_OFF_VARIANT],
            v4,
            reference_variant=V1_OFF_VARIANT,
        )
        paired_v2_by_role[role_name] = paired_gate_assessment(
            role_rows[V2_ON_VARIANT],
            v4,
            reference_variant=V2_ON_VARIANT,
        )
        delta_v3_by_role[role_name] = comparison_delta(
            role_rows[V3_ON_VARIANT],
            v4,
        )
        delta_baseline_by_role[role_name] = comparison_delta(
            role_rows[BASELINE_VARIANT],
            v4,
        )
        fixed = v4["fixed_threshold_0_5"]
        tiny_regressed = (
            fixed["matched_tiny_target_count"] < TINY_TARGET_COUNT
        )
        tiny_audit_by_role[role_name] = {
            "matched_tiny_target_count": fixed[
                "matched_tiny_target_count"
            ],
            "tiny_target_count": fixed["tiny_target_count"],
            "tiny_pd": fixed["tiny_pd"],
            "reference_matched_tiny_target_count": TINY_TARGET_COUNT,
            "tiny_pd_regressed": tiny_regressed,
            "affects_six_component_gate": False,
        }
        for variant in ALL_VARIANTS:
            ordered_rows.append(
                v4 if variant == V4_ON_VARIANT else role_rows[variant]
            )

    components = {
        "pd_primary_absolute": absolute_by_role["pd_primary"][
            "absolute_checkpoint_gate_passed"
        ],
        "miou_secondary_absolute": absolute_by_role["miou_secondary"][
            "absolute_checkpoint_gate_passed"
        ],
        "pd_primary_v4_vs_v1": paired_v1_by_role["pd_primary"]["passed"],
        "miou_secondary_v4_vs_v1": paired_v1_by_role["miou_secondary"][
            "passed"
        ],
        "pd_primary_v4_vs_v2": paired_v2_by_role["pd_primary"]["passed"],
        "miou_secondary_v4_vs_v2": paired_v2_by_role["miou_secondary"][
            "passed"
        ],
    }
    _require(
        tuple(components) == SIX_COMPONENT_NAMES,
        "six-component gate registry differs",
    )
    gate_passed = all(value is True for value in components.values())
    aggregate_tiny_regressed = any(
        value["tiny_pd_regressed"] for value in tiny_audit_by_role.values()
    )
    historical_binding = historical.get("binding")
    _require(
        isinstance(historical_binding, Mapping),
        "historical authority binding missing",
    )
    v4_sweep_bindings = {
        f"{V4_ON_VARIANT}:{checkpoint}": copy.deepcopy(
            dict(v4_rows[checkpoint]["sweep_binding"])
        )
        for checkpoint in CHECKPOINTS
    }
    v4_checkpoint_bindings = {
        f"{V4_ON_VARIANT}:{checkpoint}": copy.deepcopy(
            dict(v4_rows[checkpoint]["checkpoint_binding"])
        )
        for checkpoint in CHECKPOINTS
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "scope": "single_seed_internal_validation",
        "multi_seed_scheduled": False,
        "official_test_accessed": False,
        "row_count": len(ordered_rows),
        "rows": ordered_rows,
        "performance_gate_contract": {
            "fixed_threshold_0_5": copy.deepcopy(FIXED_GATE),
            "pd_at_fa_budget_minimum_matched_targets": copy.deepcopy(
                BUDGET_MINIMUM_MATCHED
            ),
            "paired_each_checkpoint_role": copy.deepcopy(PAIRED_GATE),
            "all_required_components": list(SIX_COMPONENT_NAMES),
            "required_control": V1_OFF_VARIANT,
            "paired_gate_predecessor": V2_ON_VARIANT,
            "structural_predecessor_additional_delta_only": V3_ON_VARIANT,
            "baseline_affects_decision": False,
            "tiny_pd_reported_not_independent_gate": True,
        },
        "v4_candidate_absolute_gate_by_role": absolute_by_role,
        "paired_v4_vs_v1_gate_by_role": paired_v1_by_role,
        "paired_v4_vs_v2_gate_by_role": paired_v2_by_role,
        "v4_vs_v3_additional_delta_by_role": delta_v3_by_role,
        "v4_vs_baseline_delta_by_role": delta_baseline_by_role,
        "v4_tiny_pd_regression_by_role": tiny_audit_by_role,
        "aggregate_tiny_pd_regressed": aggregate_tiny_regressed,
        "tiny_pd_regression_affects_decision": False,
        "six_component_gate": components,
        "aggregate_full_model_gate_passed": gate_passed,
        "decision": (
            "NER_V4_GATE_PASS"
            if gate_passed
            else "RETURN_TO_MODEL_OPTIMIZATION"
        ),
        "v4_tail_aware_accepted": gate_passed,
        "next_model_stage_authorized": gate_passed,
        "bindings": {
            "historical_authority": copy.deepcopy(dict(historical_binding)),
            "v4_sweeps": v4_sweep_bindings,
            "v4_checkpoints": v4_checkpoint_bindings,
            "v4_evaluation_sources": copy.deepcopy(
                dict(v4_rows[CHECKPOINTS[0]][
                    "evaluation_source_binding"
                ])
            ),
            "v4_training_source_lock": {
                "path": str(V4_SOURCE_LOCK.resolve()),
                "sha256": V4_SOURCE_LOCK_SHA256,
            },
            "postprocess": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "input_snapshot_before": dict(input_snapshot_before),
            "input_snapshot_after": dict(input_snapshot_after),
        },
        "metric_provenance": {
            "v4_fixed_and_budget_source": (
                "two_checkpoint_bound_pd_fa_sweep_json_files"
            ),
            "historical_fixed_and_budget_source": (
                "versioned_v3_repair_authority_cross_checked_to_raw_sweeps"
            ),
            "training_metrics_jsonl_used_as_final_metric_source": False,
            "evaluator_audit_metrics_event_count_used_for_completion_only": True,
            "checkpoint_roles": copy.deepcopy(CHECKPOINT_ROLES),
            "each_model_uses_own_selected_checkpoints": True,
        },
        "claim_boundary": {
            "single_seed_only": True,
            "cross_seed_stability_claim": False,
            "cross_dataset_claim": False,
            "official_test_claim": False,
            "tiny_pd_is_independent_pass_gate": False,
            "v3_delta_affects_six_component_gate": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V8-MPRS-DCH + five-node NER V4 Tail-Aware formal800",
        "",
        f"- Decision: `{report['decision']}`",
        (
            "- Six-component gate passed: "
            f"`{str(report['aggregate_full_model_gate_passed']).lower()}`"
        ),
        "- Scope: seed 42, NUDT-SIRST internal 530/133 validation",
        "- Official test accessed: `false`",
        "- `metrics.jsonl` used as final Pd@Fa source: `false`",
        "",
        "## Fixed threshold 0.5 and Pd@Fa",
        "",
        "| Variant | Role | Pd@0.5 | Fa@0.5 | mIoU@0.5 | False objects/image | Tiny-Pd | Pd@1e-6 | Pd@5e-6 | Pd@1e-5 | Pd@5e-5 | Pd@1e-4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        fixed = row["fixed_threshold_0_5"]
        budget_cells = []
        for key in BUDGET_KEYS:
            point = row["pd_at_fa_budget"][key]
            budget_cells.append(
                f"{point['matched_target_count']}/{point['target_count']} "
                f"({point['pd']:.9f}; Fa={point['fa']:.9g})"
            )
        lines.append(
            f"| {row['variant']} | "
            f"{ROLE_NAMES[row['checkpoint_role']]} | "
            f"{fixed['matched_target_count']}/{fixed['target_count']} "
            f"({fixed['pd']:.9f}) | {fixed['fa']:.9g} | "
            f"{fixed['miou']:.9f} | "
            f"{fixed['false_objects_per_image']:.9f} | "
            f"{fixed['matched_tiny_target_count']}/"
            f"{fixed['tiny_target_count']} ({fixed['tiny_pd']:.9f}) | "
            + " | ".join(budget_cells)
            + " |"
        )
    lines.extend(["", "## Six-component gate", ""])
    for name in SIX_COMPONENT_NAMES:
        lines.append(
            f"- `{name}`: "
            f"`{str(report['six_component_gate'][name]).lower()}`"
        )
    lines.extend(["", "## V4 relative to V3 (additional delta only)", ""])
    for role, delta in report["v4_vs_v3_additional_delta_by_role"].items():
        fixed = delta[
            "fixed_threshold_0_5_delta_candidate_minus_reference"
        ]
        lines.append(
            f"- `{role}`: matched {fixed['matched_targets']:+d}, "
            f"Pd {fixed['pd']:+.9f}, Fa {fixed['fa']:+.9g}, "
            f"mIoU {fixed['miou']:+.9f}."
        )
    lines.extend(
        [
            "",
            "V3 delta is descriptive and does not replace V1/V2 in the "
            "six-component decision.",
            "",
            "Tiny-Pd is reported but is not a seventh independent gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _input_paths(
    historical: Mapping[str, Any],
    v4_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "historical_aggregate": HISTORICAL_AGGREGATE,
        "historical_marker": HISTORICAL_MARKER,
        "v4_source_lock": V4_SOURCE_LOCK,
        "postprocess": Path(__file__),
    }
    binding = historical["binding"]
    for name, value in binding["sweeps"].items():
        paths[f"historical_sweep:{name}"] = Path(value["path"])
        paths[f"historical_checkpoint:{name}"] = Path(
            value["checkpoint_path"]
        )
    for checkpoint in CHECKPOINTS:
        row = v4_rows[checkpoint]
        paths[f"v4_sweep:{checkpoint}"] = Path(
            row["sweep_binding"]["path"]
        )
        paths[f"v4_checkpoint:{checkpoint}"] = Path(
            row["checkpoint_binding"]["path"]
        )
        for name, entry in row["evaluation_source_binding"].items():
            paths[f"v4_source:{name}"] = Path(entry["path"])
    return paths


def snapshot_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        name: sha256_file(Path(path))
        for name, path in sorted(paths.items())
    }


def _atomic_write_new(path: Path, text: str) -> None:
    path = Path(path)
    _require(not path.exists(), f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _require(not path.exists(), f"refusing to overwrite output: {path}")
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_report(
    report: Mapping[str, Any],
    *,
    output_dir: Path = COMPARISON_DIR,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir).resolve()
    json_path = output_dir / JSON_OUTPUT.name
    markdown_path = output_dir / MARKDOWN_OUTPUT.name
    marker_path = output_dir / COMPLETE_MARKER.name
    for path in (json_path, markdown_path, marker_path):
        _require(not path.exists(), f"refusing to overwrite output: {path}")
    json_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    markdown_text = render_markdown(report)
    _atomic_write_new(json_path, json_text)
    try:
        _atomic_write_new(markdown_path, markdown_text)
        marker = {
            "schema": COMPLETE_MARKER_SCHEMA,
            "status": "complete",
            "decision": report["decision"],
            "aggregate_full_model_gate_passed": report[
                "aggregate_full_model_gate_passed"
            ],
            "outputs": {
                json_path.name: sha256_file(json_path),
                markdown_path.name: sha256_file(markdown_path),
            },
        }
        _atomic_write_new(
            marker_path,
            json.dumps(
                marker,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
    except Exception:
        # Publication is write-once.  Leave any already-created evidence in
        # place for diagnosis instead of silently replacing it.
        raise
    return json_path, markdown_path, marker_path


def aggregate(
    *,
    best_sweep: Path = DEFAULT_V4_SWEEPS["best.pth.tar"],
    best_miou_sweep: Path = DEFAULT_V4_SWEEPS["best_miou.pth.tar"],
    expected_run_dir: Path = V4_RUN_DIR,
    source_lock_path: Path = V4_SOURCE_LOCK,
    source_lock_sha256: str = V4_SOURCE_LOCK_SHA256,
) -> dict[str, Any]:
    historical = validate_historical_authority()
    v4_rows = {
        "best.pth.tar": validate_v4_sweep(
            best_sweep,
            checkpoint="best.pth.tar",
            expected_run_dir=expected_run_dir,
            source_lock_path=source_lock_path,
            source_lock_sha256=source_lock_sha256,
        ),
        "best_miou.pth.tar": validate_v4_sweep(
            best_miou_sweep,
            checkpoint="best_miou.pth.tar",
            expected_run_dir=expected_run_dir,
            source_lock_path=source_lock_path,
            source_lock_sha256=source_lock_sha256,
        ),
    }
    paths = _input_paths(historical, v4_rows)
    before = snapshot_paths(paths)
    report = build_report(
        historical,
        v4_rows,
        input_snapshot_before=before,
        input_snapshot_after=before,
    )
    after = snapshot_paths(paths)
    _require(before == after, "formal inputs changed during aggregation")
    report["bindings"]["input_snapshot_after"] = after
    return report


def execution_plan() -> dict[str, Any]:
    return {
        "schema": (
            "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
            "posttraining_plan_v1"
        ),
        "status": "ready" if all(
            path.is_file() for path in DEFAULT_V4_SWEEPS.values()
        ) else "waiting_for_two_v4_formal_sweeps",
        "metric_inputs": {
            checkpoint: {
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
            }
            for checkpoint, path in DEFAULT_V4_SWEEPS.items()
        },
        "historical_authority": {
            "aggregate": str(HISTORICAL_AGGREGATE.resolve()),
            "aggregate_sha256": HISTORICAL_AGGREGATE_SHA256,
            "completion_marker": str(HISTORICAL_MARKER.resolve()),
            "completion_marker_sha256": HISTORICAL_MARKER_SHA256,
        },
        "gate_components": list(SIX_COMPONENT_NAMES),
        "v3_role": "additional_delta_only_not_six_component_gate",
        "metrics_jsonl_final_metric_source": False,
        "outputs": [
            str(JSON_OUTPUT.resolve()),
            str(MARKDOWN_OUTPUT.resolve()),
            str(COMPLETE_MARKER.resolve()),
        ],
        "writes_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--best-sweep",
        type=Path,
        default=DEFAULT_V4_SWEEPS["best.pth.tar"],
    )
    parser.add_argument(
        "--best-miou-sweep",
        type=Path,
        default=DEFAULT_V4_SWEEPS["best_miou.pth.tar"],
    )
    parser.add_argument("--output-dir", type=Path, default=COMPARISON_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan:
        print(
            json.dumps(
                execution_plan(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = aggregate(
        best_sweep=args.best_sweep,
        best_miou_sweep=args.best_miou_sweep,
    )
    paths = publish_report(report, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "aggregate_full_model_gate_passed": report[
                    "aggregate_full_model_gate_passed"
                ],
                "outputs": [str(path) for path in paths],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_VARIANTS",
    "BASELINE_VARIANT",
    "BUDGET_KEYS",
    "CHECKPOINT_ROLES",
    "CHECKPOINTS",
    "COMPLETE_MARKER",
    "JSON_OUTPUT",
    "MARKDOWN_OUTPUT",
    "SIX_COMPONENT_NAMES",
    "V1_OFF_VARIANT",
    "V2_ON_VARIANT",
    "V3_ON_VARIANT",
    "V4_ON_VARIANT",
    "absolute_gate_assessment",
    "aggregate",
    "build_report",
    "comparison_delta",
    "execution_plan",
    "normalize_budgets",
    "normalize_fixed",
    "paired_gate_assessment",
    "publish_report",
    "validate_historical_authority",
    "validate_v4_sweep",
]
