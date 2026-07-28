#!/usr/bin/env python3
"""Additive relative-performance decision for formal V4 Tail-Aware NER.

This script consumes the completed, write-once V4 formal comparison.  It
does not evaluate checkpoints, alter the original six-component decision,
or overwrite any existing evidence.  The original absolute gate remains a
diagnostic input; this supplemental decision asks whether V4 demonstrates
reproducible *relative* model improvement while preserving all observed
tradeoffs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "comparative_performance_decision_v1"
)
COMPLETE_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "comparative_performance_complete_v1"
)
SOURCE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "posttraining_aggregate_v1"
)
SOURCE_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "posttraining_complete_v1"
)

DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
TARGET_COUNT = 189
TINY_TARGET_COUNT = 39
VALIDATION_COUNT = 133
FLOAT_ATOL = 1e-12

BASELINE_VARIANT = "baseline_sctransnet"
V1_VARIANT = "tpd_ner_v8_mprs_dch_full_relay_off"
V2_VARIANT = "tpd_ner_v8_mprs_dch_v2_full_relay_on"
V3_VARIANT = "tpd_ner_v8_mprs_dch_v3_full_relay_on"
V4_VARIANT = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
)
REFERENCE_VARIANTS = (
    BASELINE_VARIANT,
    V1_VARIANT,
    V2_VARIANT,
    V3_VARIANT,
)
ALL_VARIANTS = (*REFERENCE_VARIANTS, V4_VARIANT)

CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
ROLE_NAMES = {
    "best_validation_pd_primary": "pd_primary",
    "best_validation_miou_secondary": "miou_secondary",
}
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_BY_KEY = dict(zip(BUDGET_KEYS, FA_BUDGETS))

SOURCE_COMPARISON_DIR = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/"
    "NUDT-SIRST/comparison"
)
SOURCE_COMPARISON = (
    SOURCE_COMPARISON_DIR
    / "tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_comparison.json"
)
SOURCE_MARKER = SOURCE_COMPARISON_DIR / "POSTPROCESS_COMPLETE.json"
SOURCE_COMPARISON_SHA256 = (
    "fdcb7dd0a1f591fcd6446a806d007ed8f07b1fd9e217549318dbd1ee1a69e968"
)
SOURCE_MARKER_SHA256 = (
    "be222b88b22a558ec2c8588d5e863da630f5c6af6d660de2c5b86b55547edc95"
)

OUTPUT_DIR = (
    SOURCE_COMPARISON_DIR.parent / "comparison_relative_policy_v1"
)
JSON_OUTPUT_NAME = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "formal800_comparative_performance_decision_v1.json"
)
MARKDOWN_OUTPUT_NAME = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "formal800_comparative_performance_decision_v1.md"
)
COMPLETE_MARKER_NAME = "COMPARATIVE_PERFORMANCE_COMPLETE_V1.json"


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
    _require(
        path.is_file() and not path.is_symlink(),
        f"required regular file is missing: {path}",
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(
        path.is_file() and not path.is_symlink(),
        f"required JSON file is missing: {path}",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is not an integer",
    )
    return int(value)


def _close(left: Any, right: Any, *, atol: float = FLOAT_ATOL) -> bool:
    return math.isclose(
        _finite_number(left, "left comparison value"),
        _finite_number(right, "right comparison value"),
        rel_tol=0.0,
        abs_tol=atol,
    )


def _validate_fixed(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} fixed metrics missing")
    target_count = _integer(value.get("target_count"), f"{label} targets")
    matched = _integer(
        value.get("matched_target_count"),
        f"{label} matched targets",
    )
    tiny_count = _integer(
        value.get("tiny_target_count"),
        f"{label} tiny targets",
    )
    tiny_matched = _integer(
        value.get("matched_tiny_target_count"),
        f"{label} matched tiny targets",
    )
    threshold = _finite_number(value.get("threshold"), f"{label} threshold")
    pd = _finite_number(value.get("pd"), f"{label} Pd")
    fa = _finite_number(value.get("fa"), f"{label} Fa")
    miou = _finite_number(value.get("miou"), f"{label} mIoU")
    false_per_image = _finite_number(
        value.get("false_objects_per_image"),
        f"{label} false objects per image",
    )
    tiny_pd = _finite_number(value.get("tiny_pd"), f"{label} tiny-Pd")

    _require(_close(threshold, 0.5), f"{label} is not threshold 0.5")
    _require(target_count == TARGET_COUNT, f"{label} target count differs")
    _require(
        0 <= matched <= target_count,
        f"{label} matched target count invalid",
    )
    _require(
        _close(pd, matched / target_count),
        f"{label} Pd/count differs",
    )
    _require(tiny_count == TINY_TARGET_COUNT, f"{label} tiny count differs")
    _require(
        0 <= tiny_matched <= tiny_count,
        f"{label} matched tiny count invalid",
    )
    _require(
        _close(tiny_pd, tiny_matched / tiny_count),
        f"{label} tiny-Pd/count differs",
    )
    _require(0.0 <= fa, f"{label} Fa is negative")
    _require(0.0 <= miou <= 1.0, f"{label} mIoU is out of range")
    _require(
        false_per_image >= 0.0,
        f"{label} false objects per image is negative",
    )
    derived_false_count = round(false_per_image * VALIDATION_COUNT)
    _require(
        _close(
            false_per_image,
            derived_false_count / VALIDATION_COUNT,
        ),
        f"{label} false objects per image/count differs",
    )
    return {
        "threshold": threshold,
        "target_count": target_count,
        "matched_target_count": matched,
        "pd": pd,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": false_per_image,
        "derived_false_object_count": derived_false_count,
        "tiny_target_count": tiny_count,
        "matched_tiny_target_count": tiny_matched,
        "tiny_pd": tiny_pd,
    }


def _validate_budgets(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} budget metrics missing")
    _require(
        set(value) == set(BUDGET_KEYS),
        f"{label} budget keys differ",
    )
    normalized: dict[str, Any] = {}
    for key in BUDGET_KEYS:
        point = value[key]
        _require(isinstance(point, Mapping), f"{label}:{key} is invalid")
        target_count = _integer(
            point.get("target_count"),
            f"{label}:{key} target count",
        )
        matched = _integer(
            point.get("matched_target_count"),
            f"{label}:{key} matched targets",
        )
        pd = _finite_number(point.get("pd"), f"{label}:{key} Pd")
        fa = _finite_number(point.get("fa"), f"{label}:{key} Fa")
        threshold = _finite_number(
            point.get("threshold"),
            f"{label}:{key} threshold",
        )
        _require(
            target_count == TARGET_COUNT,
            f"{label}:{key} target count differs",
        )
        _require(
            0 <= matched <= target_count,
            f"{label}:{key} matched target count invalid",
        )
        _require(
            _close(pd, matched / target_count),
            f"{label}:{key} Pd/count differs",
        )
        _require(fa >= 0.0, f"{label}:{key} Fa is negative")
        _require(
            fa <= BUDGET_BY_KEY[key] + FLOAT_ATOL,
            f"{label}:{key} exceeds Fa budget",
        )
        _require(
            0.0 <= threshold <= 1.0,
            f"{label}:{key} threshold is out of range",
        )
        normalized[key] = {
            "threshold": threshold,
            "target_count": target_count,
            "matched_target_count": matched,
            "pd": pd,
            "fa": fa,
        }
    return normalized


def _validate_row(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} row is invalid")
    variant = value.get("variant")
    checkpoint = value.get("checkpoint")
    role = value.get("checkpoint_role")
    _require(variant in ALL_VARIANTS, f"{label} variant differs")
    _require(
        checkpoint in CHECKPOINT_ROLES,
        f"{label} checkpoint differs",
    )
    _require(
        role == CHECKPOINT_ROLES[checkpoint],
        f"{label} checkpoint role differs",
    )
    checkpoint_epoch = _integer(
        value.get("checkpoint_epoch"),
        f"{label} checkpoint epoch",
    )
    checkpoint_sha = value.get("checkpoint_sha256")
    _require(checkpoint_epoch >= 1, f"{label} checkpoint epoch invalid")
    _require(
        _is_hex_sha256(checkpoint_sha),
        f"{label} checkpoint SHA-256 invalid",
    )
    return {
        "variant": variant,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "role_name": ROLE_NAMES[role],
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "fixed_threshold_0_5": _validate_fixed(
            value.get("fixed_threshold_0_5"),
            label=label,
        ),
        "pd_at_fa_budget": _validate_budgets(
            value.get("pd_at_fa_budget"),
            label=label,
        ),
    }


def _resolve_marker_outputs(
    marker: Mapping[str, Any],
    *,
    marker_path: Path,
) -> dict[str, Path]:
    outputs = marker.get("outputs")
    _require(isinstance(outputs, Mapping), "source marker outputs missing")
    _require(len(outputs) >= 1, "source marker outputs are empty")
    result: dict[str, Path] = {}
    for name, expected_sha in outputs.items():
        _require(
            isinstance(name, str)
            and name
            and Path(name).name == name,
            "source marker output name invalid",
        )
        _require(
            _is_hex_sha256(expected_sha),
            f"source marker output SHA-256 invalid: {name}",
        )
        result[name] = marker_path.parent / name
    return result


def validate_source_authority(
    *,
    comparison_path: Path = SOURCE_COMPARISON,
    marker_path: Path = SOURCE_MARKER,
    expected_comparison_sha256: str = SOURCE_COMPARISON_SHA256,
    expected_marker_sha256: str = SOURCE_MARKER_SHA256,
) -> dict[str, Any]:
    comparison_path = Path(comparison_path).resolve()
    marker_path = Path(marker_path).resolve()
    _require(
        _is_hex_sha256(expected_comparison_sha256),
        "expected comparison SHA-256 invalid",
    )
    _require(
        _is_hex_sha256(expected_marker_sha256),
        "expected marker SHA-256 invalid",
    )
    comparison_sha = sha256_file(comparison_path)
    marker_sha = sha256_file(marker_path)
    _require(
        comparison_sha == expected_comparison_sha256,
        "source comparison SHA-256 differs",
    )
    _require(
        marker_sha == expected_marker_sha256,
        "source marker SHA-256 differs",
    )
    marker = load_json(marker_path)
    _require(
        marker.get("schema") == SOURCE_MARKER_SCHEMA,
        "source marker schema differs",
    )
    _require(marker.get("status") == "complete", "source marker incomplete")
    marker_outputs = _resolve_marker_outputs(
        marker,
        marker_path=marker_path,
    )
    _require(
        comparison_path.name in marker_outputs,
        "source marker does not bind comparison",
    )
    observed_output_hashes: dict[str, str] = {}
    for name, path in sorted(marker_outputs.items()):
        observed_sha = sha256_file(path)
        expected_sha = marker["outputs"][name]
        _require(
            observed_sha == expected_sha,
            f"source marker output SHA-256 differs: {name}",
        )
        observed_output_hashes[name] = observed_sha
    _require(
        observed_output_hashes[comparison_path.name] == comparison_sha,
        "source marker comparison binding differs",
    )

    source = load_json(comparison_path)
    expected_header = {
        "schema": SOURCE_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "scope": "single_seed_internal_validation",
        "row_count": len(ALL_VARIANTS) * len(CHECKPOINT_ROLES),
    }
    for key, expected in expected_header.items():
        _require(
            source.get(key) == expected,
            f"source comparison {key} differs",
        )
    _require(
        source.get("official_test_accessed") is False,
        "source official-test state differs",
    )
    provenance = source.get("metric_provenance")
    _require(
        isinstance(provenance, Mapping),
        "source metric provenance missing",
    )
    _require(
        provenance.get("each_model_uses_own_selected_checkpoints") is True,
        "source does not use each model's own checkpoints",
    )
    _require(
        provenance.get("checkpoint_roles") == CHECKPOINT_ROLES,
        "source checkpoint role registry differs",
    )
    bindings = source.get("bindings")
    _require(isinstance(bindings, Mapping), "source bindings missing")
    before = bindings.get("input_snapshot_before")
    after = bindings.get("input_snapshot_after")
    _require(
        isinstance(before, Mapping) and dict(before) == dict(after),
        "source input snapshots differ",
    )

    rows = source.get("rows")
    _require(isinstance(rows, list), "source rows missing")
    _require(
        len(rows) == expected_header["row_count"],
        "source row count differs",
    )
    normalized_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        normalized = _validate_row(row, label=f"row[{index}]")
        key = (normalized["variant"], normalized["checkpoint_role"])
        _require(key not in normalized_rows, f"duplicate source row: {key}")
        normalized_rows[key] = normalized
    expected_keys = {
        (variant, role)
        for variant in ALL_VARIANTS
        for role in CHECKPOINT_ROLES.values()
    }
    _require(
        set(normalized_rows) == expected_keys,
        "source model/checkpoint matrix differs",
    )
    _require(
        marker.get("decision") == source.get("decision"),
        "source marker decision differs",
    )
    _require(
        marker.get("aggregate_full_model_gate_passed")
        == source.get("aggregate_full_model_gate_passed"),
        "source marker gate result differs",
    )
    return {
        "source": source,
        "rows": normalized_rows,
        "binding": {
            "comparison": {
                "path": str(comparison_path),
                "sha256": comparison_sha,
            },
            "completion_marker": {
                "path": str(marker_path),
                "sha256": marker_sha,
            },
            "marker_outputs": observed_output_hashes,
        },
        "bound_paths": {
            "comparison": comparison_path,
            "completion_marker": marker_path,
            **{
                f"marker_output:{name}": path
                for name, path in marker_outputs.items()
            },
        },
    }


def _compare_higher(candidate: float, reference: float) -> str:
    if _close(candidate, reference):
        return "equal"
    return "better" if candidate > reference else "worse"


def _compare_lower(candidate: float, reference: float) -> str:
    if _close(candidate, reference):
        return "equal"
    return "better" if candidate < reference else "worse"


def fixed_pareto_assessment(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    reference_fixed = reference["fixed_threshold_0_5"]
    candidate_fixed = candidate["fixed_threshold_0_5"]
    relations = {
        "detection": _compare_higher(
            candidate_fixed["matched_target_count"],
            reference_fixed["matched_target_count"],
        ),
        "fa": _compare_lower(
            candidate_fixed["fa"],
            reference_fixed["fa"],
        ),
        "miou": _compare_higher(
            candidate_fixed["miou"],
            reference_fixed["miou"],
        ),
        "false_objects": _compare_lower(
            candidate_fixed["false_objects_per_image"],
            reference_fixed["false_objects_per_image"],
        ),
        "tiny_detection": _compare_higher(
            candidate_fixed["matched_tiny_target_count"],
            reference_fixed["matched_tiny_target_count"],
        ),
    }
    relation_values = tuple(relations.values())
    candidate_dominates = (
        "worse" not in relation_values and "better" in relation_values
    )
    reference_dominates = (
        "better" not in relation_values and "worse" in relation_values
    )
    if candidate_dominates:
        relation = "candidate_dominates"
    elif reference_dominates:
        relation = "reference_dominates"
    elif all(value == "equal" for value in relation_values):
        relation = "equal"
    else:
        relation = "tradeoff"
    return {
        "relation": relation,
        "candidate_dominates": candidate_dominates,
        "reference_dominates": reference_dominates,
        "objective_group_relations": relations,
        "candidate": dict(candidate_fixed),
        "reference": dict(reference_fixed),
        "delta_candidate_minus_reference": {
            "matched_target_count": (
                candidate_fixed["matched_target_count"]
                - reference_fixed["matched_target_count"]
            ),
            "pd": candidate_fixed["pd"] - reference_fixed["pd"],
            "fa": candidate_fixed["fa"] - reference_fixed["fa"],
            "miou": candidate_fixed["miou"] - reference_fixed["miou"],
            "false_objects_per_image": (
                candidate_fixed["false_objects_per_image"]
                - reference_fixed["false_objects_per_image"]
            ),
            "derived_false_object_count": (
                candidate_fixed["derived_false_object_count"]
                - reference_fixed["derived_false_object_count"]
            ),
            "matched_tiny_target_count": (
                candidate_fixed["matched_tiny_target_count"]
                - reference_fixed["matched_tiny_target_count"]
            ),
            "tiny_pd": (
                candidate_fixed["tiny_pd"] - reference_fixed["tiny_pd"]
            ),
        },
    }


def budget_assessment(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    reference_budgets = reference["pd_at_fa_budget"]
    candidate_budgets = candidate["pd_at_fa_budget"]
    points: dict[str, Any] = {}
    noninferior_count = 0
    strict_better_count = 0
    strict_worse_count = 0
    equal_detection_lower_fa_count = 0
    for key in BUDGET_KEYS:
        reference_point = reference_budgets[key]
        candidate_point = candidate_budgets[key]
        reference_count = reference_point["matched_target_count"]
        candidate_count = candidate_point["matched_target_count"]
        noninferior = candidate_count >= reference_count
        strict_better = candidate_count > reference_count
        strict_worse = candidate_count < reference_count
        noninferior_count += int(noninferior)
        strict_better_count += int(strict_better)
        strict_worse_count += int(strict_worse)
        if strict_better:
            relation = "candidate_more_detections"
        elif strict_worse:
            relation = "reference_more_detections"
        else:
            fa_relation = _compare_lower(
                candidate_point["fa"],
                reference_point["fa"],
            )
            if fa_relation == "better":
                relation = "equal_detections_candidate_lower_fa"
                equal_detection_lower_fa_count += 1
            elif fa_relation == "worse":
                relation = "equal_detections_reference_lower_fa"
            else:
                relation = "equal_detection_and_fa"
        points[key] = {
            "fa_budget": BUDGET_BY_KEY[key],
            "relation": relation,
            "candidate_matched_target_count": candidate_count,
            "reference_matched_target_count": reference_count,
            "delta_matched_targets_candidate_minus_reference": (
                candidate_count - reference_count
            ),
            "candidate_pd": candidate_point["pd"],
            "reference_pd": reference_point["pd"],
            "candidate_achieved_fa": candidate_point["fa"],
            "reference_achieved_fa": reference_point["fa"],
            "candidate_matched_noninferior": noninferior,
            "candidate_matched_strictly_better": strict_better,
            "candidate_matched_strictly_worse": strict_worse,
        }
    if noninferior_count == len(BUDGET_KEYS) and strict_better_count > 0:
        profile_relation = "candidate_pd_budget_dominates"
    elif strict_better_count == 0 and strict_worse_count > 0:
        profile_relation = "reference_pd_budget_dominates"
    elif strict_better_count == 0 and strict_worse_count == 0:
        profile_relation = "equal_pd_budget_profile"
    else:
        profile_relation = "pd_budget_tradeoff"
    return {
        "comparison_semantics": (
            "matched_targets_primary_achieved_fa_reported_"
            "and_tiebreak_only_when_matched_equal"
        ),
        "global_point_pareto_claimed": False,
        "points": points,
        "noninferior_budget_count": noninferior_count,
        "strictly_better_budget_count": strict_better_count,
        "strictly_worse_budget_count": strict_worse_count,
        "equal_detection_lower_fa_count": (
            equal_detection_lower_fa_count
        ),
        "budget_count": len(BUDGET_KEYS),
        "profile_relation": profile_relation,
    }


def pairwise_assessment(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        reference["checkpoint_role"] == candidate["checkpoint_role"],
        "pairwise checkpoint roles differ",
    )
    fixed = fixed_pareto_assessment(reference, candidate)
    budgets = budget_assessment(reference, candidate)
    return {
        "reference_variant": reference["variant"],
        "candidate_variant": candidate["variant"],
        "checkpoint": candidate["checkpoint"],
        "checkpoint_role": candidate["checkpoint_role"],
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "tiny_pd_no_regression": (
            fixed["delta_candidate_minus_reference"][
                "matched_tiny_target_count"
            ]
            >= 0
        ),
    }


def _point_key(value: Mapping[str, Any]) -> str:
    return f"{value['variant']}:{value['role_name']}"


def _ordered_rows(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        rows[(variant, role)]
        for variant in ALL_VARIANTS
        for role in CHECKPOINT_ROLES.values()
    ]


def global_fixed_pareto_frontier(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = _ordered_rows(rows)
    point_assessments: dict[str, Any] = {}
    for point in ordered:
        point_key = _point_key(point)
        dominated_by: list[str] = []
        dominates: list[str] = []
        for other in ordered:
            other_key = _point_key(other)
            if other_key == point_key:
                continue
            other_vs_point = fixed_pareto_assessment(point, other)
            if other_vs_point["candidate_dominates"]:
                dominated_by.append(other_key)
            point_vs_other = fixed_pareto_assessment(other, point)
            if point_vs_other["candidate_dominates"]:
                dominates.append(other_key)
        point_assessments[point_key] = {
            "point_key": point_key,
            "variant": point["variant"],
            "checkpoint": point["checkpoint"],
            "checkpoint_role": point["checkpoint_role"],
            "role_name": point["role_name"],
            "fixed_threshold_0_5": dict(
                point["fixed_threshold_0_5"]
            ),
            "is_global_pareto": not dominated_by,
            "dominated_by": dominated_by,
            "dominates": dominates,
        }
    frontier_keys = [
        _point_key(point)
        for point in ordered
        if point_assessments[_point_key(point)]["is_global_pareto"]
    ]
    return {
        "comparison_scope": "all_five_models_both_checkpoint_roles",
        "point_count": len(ordered),
        "objective_groups": [
            "detection_maximize",
            "fa_minimize",
            "miou_maximize",
            "false_objects_minimize",
            "tiny_detection_maximize",
        ],
        "frontier_point_count": len(frontier_keys),
        "frontier_point_keys": frontier_keys,
        "frontier_points": [
            point_assessments[key] for key in frontier_keys
        ],
        "points": point_assessments,
    }


def model_fa_budget_envelopes(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    role_order = list(CHECKPOINT_ROLES.values())
    envelopes: dict[str, Any] = {}
    for variant in ALL_VARIANTS:
        points: dict[str, Any] = {}
        matched_profile: list[int] = []
        achieved_fa_profile: list[float] = []
        selected_role_profile: list[str] = []
        for key in BUDGET_KEYS:
            role_candidates = []
            for role in role_order:
                row = rows[(variant, role)]
                point = row["pd_at_fa_budget"][key]
                role_candidates.append(
                    {
                        "checkpoint": row["checkpoint"],
                        "checkpoint_role": role,
                        "role_name": row["role_name"],
                        "matched_target_count": point[
                            "matched_target_count"
                        ],
                        "pd": point["pd"],
                        "achieved_fa": point["fa"],
                        "threshold": point["threshold"],
                    }
                )
            maximum_matched = max(
                value["matched_target_count"]
                for value in role_candidates
            )
            minimum_fa = min(
                value["achieved_fa"]
                for value in role_candidates
                if value["matched_target_count"] == maximum_matched
            )
            co_leaders = [
                value
                for value in role_candidates
                if value["matched_target_count"] == maximum_matched
                and value["achieved_fa"] == minimum_fa
            ]
            selected = co_leaders[0]
            points[key] = {
                "fa_budget": BUDGET_BY_KEY[key],
                "selection_rule": (
                    "maximize_matched_then_minimize_achieved_fa_"
                    "then_checkpoint_role_order_for_exact_tie"
                ),
                "selected_checkpoint": selected["checkpoint"],
                "selected_checkpoint_role": selected[
                    "checkpoint_role"
                ],
                "selected_role_name": selected["role_name"],
                "selected_matched_target_count": selected[
                    "matched_target_count"
                ],
                "selected_pd": selected["pd"],
                "selected_achieved_fa": selected["achieved_fa"],
                "selected_threshold": selected["threshold"],
                "co_leader_role_names": [
                    value["role_name"] for value in co_leaders
                ],
                "role_candidates": role_candidates,
            }
            matched_profile.append(selected["matched_target_count"])
            achieved_fa_profile.append(selected["achieved_fa"])
            selected_role_profile.append(selected["role_name"])
        envelopes[variant] = {
            "variant": variant,
            "checkpoint_roles_considered": [
                ROLE_NAMES[role] for role in role_order
            ],
            "points": points,
            "matched_target_profile_in_budget_order": matched_profile,
            "achieved_fa_profile_in_budget_order": achieved_fa_profile,
            "selected_role_profile_in_budget_order": (
                selected_role_profile
            ),
        }
    return envelopes


def global_fa_budget_leaders(
    envelopes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    points: dict[str, Any] = {}
    v4_leader_keys: list[str] = []
    v4_strict_new_detection_keys: list[str] = []
    for key in BUDGET_KEYS:
        model_points = [
            {
                "variant": variant,
                **dict(envelopes[variant]["points"][key]),
            }
            for variant in ALL_VARIANTS
        ]
        maximum_matched = max(
            value["selected_matched_target_count"]
            for value in model_points
        )
        minimum_fa = min(
            value["selected_achieved_fa"]
            for value in model_points
            if value["selected_matched_target_count"]
            == maximum_matched
        )
        leaders = [
            value
            for value in model_points
            if value["selected_matched_target_count"] == maximum_matched
            and value["selected_achieved_fa"] == minimum_fa
        ]
        leader_variants = [value["variant"] for value in leaders]
        historical_maximum = max(
            value["selected_matched_target_count"]
            for value in model_points
            if value["variant"] != V4_VARIANT
        )
        v4_point = envelopes[V4_VARIANT]["points"][key]
        v4_strict_new = (
            v4_point["selected_matched_target_count"]
            > historical_maximum
        )
        v4_is_leader = V4_VARIANT in leader_variants
        if v4_is_leader:
            v4_leader_keys.append(key)
        if v4_strict_new:
            v4_strict_new_detection_keys.append(key)
        points[key] = {
            "fa_budget": BUDGET_BY_KEY[key],
            "selection_rule": (
                "maximize_model_envelope_matched_then_minimize_"
                "model_envelope_achieved_fa_with_all_exact_co_leaders"
            ),
            "maximum_matched_target_count": maximum_matched,
            "minimum_achieved_fa_at_maximum_matched": minimum_fa,
            "leader_variants": leader_variants,
            "selected_leader_variant_for_display": leader_variants[0],
            "model_envelope_points": model_points,
            "v4_is_global_leader": v4_is_leader,
            "historical_maximum_matched_target_count": (
                historical_maximum
            ),
            "v4_matched_target_count": v4_point[
                "selected_matched_target_count"
            ],
            "v4_strictly_exceeds_all_historical_matched": v4_strict_new,
        }
    return {
        "points": points,
        "v4_global_leader_budget_keys": v4_leader_keys,
        "v4_global_leader_budget_count": len(v4_leader_keys),
        "v4_strict_new_detection_budget_keys": (
            v4_strict_new_detection_keys
        ),
    }


def build_report(authority: Mapping[str, Any]) -> dict[str, Any]:
    source = authority["source"]
    rows = authority["rows"]
    pairwise: dict[str, Any] = {}
    for reference_variant in REFERENCE_VARIANTS:
        by_role: dict[str, Any] = {}
        for role in CHECKPOINT_ROLES.values():
            by_role[ROLE_NAMES[role]] = pairwise_assessment(
                rows[(reference_variant, role)],
                rows[(V4_VARIANT, role)],
            )
        pairwise[reference_variant] = by_role

    fixed_frontier = global_fixed_pareto_frontier(rows)
    envelopes = model_fa_budget_envelopes(rows)
    budget_leaders = global_fa_budget_leaders(envelopes)
    v4_fixed_frontier_keys = [
        key
        for key in fixed_frontier["frontier_point_keys"]
        if key.startswith(f"{V4_VARIANT}:")
    ]
    tiny_no_regression_by_role = {
        ROLE_NAMES[role]: all(
            rows[(V4_VARIANT, role)]["fixed_threshold_0_5"][
                "matched_tiny_target_count"
            ]
            >= rows[(reference, role)]["fixed_threshold_0_5"][
                "matched_tiny_target_count"
            ]
            for reference in REFERENCE_VARIANTS
        )
        for role in CHECKPOINT_ROLES.values()
    }
    tiny_no_regression = all(
        value["tiny_pd_no_regression"] is True
        for by_role in pairwise.values()
        for value in by_role.values()
    )
    _require(
        tiny_no_regression
        == all(tiny_no_regression_by_role.values()),
        "tiny-Pd symmetric and pairwise audits differ",
    )
    components = {
        "v4_contributes_global_fixed_pareto_point": bool(
            v4_fixed_frontier_keys
        ),
        "v4_strictly_improves_historical_envelope_at_any_fa_budget": bool(
            budget_leaders["v4_strict_new_detection_budget_keys"]
        ),
        "v4_two_fixed_checkpoints_tiny_pd_no_regression": (
            tiny_no_regression
        ),
    }
    gate_passed = all(value is True for value in components.values())
    fixed_relations = [
        value["fixed_threshold_0_5"]["relation"]
        for by_role in pairwise.values()
        for value in by_role.values()
    ]
    historical_fixed_keys = [
        _point_key(row)
        for row in _ordered_rows(rows)
        if row["variant"] != V4_VARIANT
    ]
    universal_fixed_dominance = any(
        all(
            historical_key
            in fixed_frontier["points"][v4_key]["dominates"]
            for historical_key in historical_fixed_keys
        )
        for v4_key in (
            _point_key(rows[(V4_VARIANT, role)])
            for role in CHECKPOINT_ROLES.values()
        )
    )
    universal_budget_noninferiority = all(
        envelopes[V4_VARIANT]["points"][key][
            "selected_matched_target_count"
        ]
        >= max(
            envelopes[reference]["points"][key][
                "selected_matched_target_count"
            ]
            for reference in REFERENCE_VARIANTS
        )
        for key in BUDGET_KEYS
    )
    tradeoff_present = (
        "tradeoff" in fixed_relations
        or any(
            not point["v4_is_global_leader"]
            for point in budget_leaders["points"].values()
        )
        or any(
            not key.startswith(f"{V4_VARIANT}:")
            for key in fixed_frontier["frontier_point_keys"]
        )
    )
    strictest_fa_budget_advantage = budget_leaders["points"]["1e-06"][
        "v4_strictly_exceeds_all_historical_matched"
    ]
    if gate_passed and tradeoff_present:
        decision = (
            "RELATIVE_MODEL_IMPROVEMENT_CONFIRMED_WITH_TRADEOFF"
        )
    elif gate_passed:
        decision = "RELATIVE_MODEL_IMPROVEMENT_CONFIRMED"
    else:
        decision = "RELATIVE_MODEL_IMPROVEMENT_NOT_CONFIRMED"
    original_six_component_gate = source.get("six_component_gate")
    _require(
        isinstance(original_six_component_gate, Mapping),
        "source six-component gate missing",
    )
    universal_dominance = (
        universal_fixed_dominance
        and universal_budget_noninferiority
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "scope": source["scope"],
        "candidate_variant": V4_VARIANT,
        "reference_variants": list(REFERENCE_VARIANTS),
        "checkpoint_roles": dict(CHECKPOINT_ROLES),
        "policy": {
            "name": "symmetric_comparative_engineering_policy_v1",
            "timing": "post_training_user_confirmed_engineering_decision",
            "decision_basis": (
                "symmetric_direct_relative_model_performance"
            ),
            "confirmatory_paper_conclusion": False,
            "absolute_numeric_gate_role": "diagnostic_only_non_veto",
            "each_model_uses_own_selected_checkpoints": True,
            "fixed_objective_groups": {
                "detection": {
                    "reported": ["matched_target_count", "pd"],
                    "direction": "maximize",
                    "counted_once": True,
                },
                "fa": {"reported": ["fa"], "direction": "minimize"},
                "overlap": {
                    "reported": ["miou"],
                    "direction": "maximize",
                },
                "false_objects": {
                    "reported": [
                        "false_objects_per_image",
                        "derived_false_object_count",
                    ],
                    "direction": "minimize",
                    "counted_once": True,
                },
                "tiny_detection": {
                    "reported": [
                        "matched_tiny_target_count",
                        "tiny_pd",
                    ],
                    "direction": "maximize",
                    "counted_once": True,
                },
            },
            "fixed_pareto_definition": (
                "all_five_objective_groups_nonworse_and_at_least_"
                "one_strictly_better"
            ),
            "global_fixed_pareto_scope": (
                "all_five_models_times_both_checkpoint_roles"
            ),
            "fa_budget_keys": list(BUDGET_KEYS),
            "model_fa_budget_envelope_rule": (
                "between_own_best_and_best_miou_maximize_matched_"
                "then_minimize_achieved_fa"
            ),
            "global_fa_budget_leader_rule": (
                "across_five_model_envelopes_maximize_matched_"
                "then_minimize_achieved_fa_and_retain_exact_co_leaders"
            ),
            "pairwise_same_role_matrix_role": (
                "descriptive_only_not_a_pass_gate"
            ),
            "four_of_five_rule_used_as_veto": False,
            "float_absolute_tolerance": FLOAT_ATOL,
        },
        "source_original_decision": {
            "decision": source.get("decision"),
            "aggregate_full_model_gate_passed": source.get(
                "aggregate_full_model_gate_passed"
            ),
            "six_component_gate": dict(original_six_component_gate),
            "role": "retained_unchanged_diagnostic_only",
            "veto_applied_to_relative_decision": False,
        },
        "pairwise_by_reference_and_role": pairwise,
        "global_fixed_pareto_frontier": fixed_frontier,
        "model_fa_budget_envelopes": envelopes,
        "global_fa_budget_leaders": budget_leaders,
        "relative_success_evidence": {
            "v4_global_fixed_pareto_point_keys": (
                v4_fixed_frontier_keys
            ),
            "v4_global_leader_budget_keys": budget_leaders[
                "v4_global_leader_budget_keys"
            ],
            "v4_strict_new_detection_budget_keys": budget_leaders[
                "v4_strict_new_detection_budget_keys"
            ],
            "v4_fixed_tiny_pd_no_regression_by_role": (
                tiny_no_regression_by_role
            ),
        },
        "relative_success_components": components,
        "comprehensive_relative_gate_passed": gate_passed,
        "decision": decision,
        "model_iteration_success": gate_passed,
        "next_model_stage_authorized": gate_passed,
        "tradeoff_present": tradeoff_present,
        "universal_fixed_dominance": universal_fixed_dominance,
        "universal_pd_budget_noninferiority": (
            universal_budget_noninferiority
        ),
        "universal_dominance": universal_dominance,
        "strictest_fa_budget_advantage": strictest_fa_budget_advantage,
        "claim_boundary": {
            "single_seed_internal_validation_only": True,
            "official_test_accessed": False,
            "global_dominance_claim_supported": universal_dominance,
            "strictest_fa_budget_advantage_claim_supported": (
                strictest_fa_budget_advantage
            ),
            "cross_seed_stability_claim_supported": False,
            "cross_dataset_claim_supported": False,
            "model_iteration_decision_is_not_paper_level_stability_claim": (
                True
            ),
        },
        "bindings": {
            **dict(authority["binding"]),
            "decision_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }


def _snapshot_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def _validated_binding_snapshot(
    authority: Mapping[str, Any],
) -> dict[str, str]:
    binding = authority["binding"]
    return {
        "comparison": binding["comparison"]["sha256"],
        "completion_marker": binding["completion_marker"]["sha256"],
        **{
            f"marker_output:{name}": sha
            for name, sha in binding["marker_outputs"].items()
        },
    }


def aggregate(
    *,
    comparison_path: Path = SOURCE_COMPARISON,
    marker_path: Path = SOURCE_MARKER,
    expected_comparison_sha256: str = SOURCE_COMPARISON_SHA256,
    expected_marker_sha256: str = SOURCE_MARKER_SHA256,
) -> dict[str, Any]:
    authority = validate_source_authority(
        comparison_path=comparison_path,
        marker_path=marker_path,
        expected_comparison_sha256=expected_comparison_sha256,
        expected_marker_sha256=expected_marker_sha256,
    )
    snapshot_before = _snapshot_paths(authority["bound_paths"])
    _require(
        snapshot_before == _validated_binding_snapshot(authority),
        "source authority changed after validation before aggregation",
    )
    report = build_report(authority)
    snapshot_after = _snapshot_paths(authority["bound_paths"])
    _require(
        snapshot_before == snapshot_after,
        "source authority changed during comparative aggregation",
    )
    report["bindings"]["input_snapshot_before"] = snapshot_before
    report["bindings"]["input_snapshot_after"] = snapshot_after
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    gate_passed = (
        report["comprehensive_relative_gate_passed"] is True
    )
    if gate_passed:
        if report["tradeoff_present"]:
            conclusion = (
                "Within this post-training engineering policy, the result "
                "confirms a relative model iteration with explicit "
                "tradeoffs."
            )
        else:
            conclusion = (
                "Within this post-training engineering policy, the result "
                "confirms a relative model iteration."
            )
    else:
        conclusion = (
            "Within this post-training engineering policy, the result does "
            "not confirm a relative model iteration."
        )
    lines = [
        "# V4 Tail-Aware Formal800 Comparative Performance Decision",
        "",
        f"- Decision: `{report['decision']}`",
        "- Relative model iteration success: "
        f"`{str(report['model_iteration_success']).lower()}`",
        "- Next model stage authorized: "
        f"`{str(report['next_model_stage_authorized']).lower()}`",
        "- Original absolute gate retained as diagnostic only: `true`",
        "- Universal dominance: "
        f"`{str(report['universal_dominance']).lower()}`",
        "- Tradeoff present: "
        f"`{str(report['tradeoff_present']).lower()}`",
        "- Policy timing: "
        "`post_training_user_confirmed_engineering_decision`",
        "- Scope: one internal validation split, seed 42",
        "",
        "## Global fixed-threshold Pareto frontier",
        "",
        "| Model | Checkpoint role | Matched | Fa | mIoU | "
        "False objects/image | Tiny matched |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in report["global_fixed_pareto_frontier"][
        "frontier_points"
    ]:
        fixed = point["fixed_threshold_0_5"]
        lines.append(
            f"| {point['variant']} | {point['role_name']} | "
            f"{fixed['matched_target_count']}/{fixed['target_count']} | "
            f"{fixed['fa']:.12g} | {fixed['miou']:.9f} | "
            f"{fixed['false_objects_per_image']:.9f} | "
            f"{fixed['matched_tiny_target_count']}/"
            f"{fixed['tiny_target_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-model Fa-budget envelopes",
            "",
            "| Model | Matched profile at "
            "1e-6/5e-6/1e-5/5e-5/1e-4 | Selected roles |",
            "| --- | --- | --- |",
        ]
    )
    for variant in ALL_VARIANTS:
        envelope = report["model_fa_budget_envelopes"][variant]
        matched = "/".join(
            str(value)
            for value in envelope[
                "matched_target_profile_in_budget_order"
            ]
        )
        roles = "/".join(
            envelope["selected_role_profile_in_budget_order"]
        )
        lines.append(f"| {variant} | {matched} | {roles} |")
    lines.extend(
        [
            "",
            "## Global Fa-budget leaders",
            "",
            "| Fa budget | Leader model(s) | Matched | "
            "Achieved Fa | V4 strict new detection |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for key in BUDGET_KEYS:
        point = report["global_fa_budget_leaders"]["points"][key]
        lines.append(
            f"| {key} | {', '.join(point['leader_variants'])} | "
            f"{point['maximum_matched_target_count']} | "
            f"{point['minimum_achieved_fa_at_maximum_matched']:.12g} | "
            f"`{str(point['v4_strictly_exceeds_all_historical_matched']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## Same-role pairwise evidence (descriptive only)",
        "",
        "| Reference | Checkpoint role | Fixed relation | "
        "Pd@Fa noninferior | Strictly better | Strictly worse |",
        "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for reference in REFERENCE_VARIANTS:
        for role_name in ("pd_primary", "miou_secondary"):
            value = report["pairwise_by_reference_and_role"][reference][
                role_name
            ]
            fixed = value["fixed_threshold_0_5"]
            budgets = value["pd_at_fa_budget"]
            lines.append(
                f"| {reference} | {role_name} | "
                f"{fixed['relation']} | "
                f"{budgets['noninferior_budget_count']}/5 | "
                f"{budgets['strictly_better_budget_count']}/5 | "
                f"{budgets['strictly_worse_budget_count']}/5 |"
            )
    lines.extend(
        [
            "",
            "## Relative success components",
            "",
            "| Component | Passed |",
            "| --- | --- |",
        ]
    )
    for name, passed in report["relative_success_components"].items():
        lines.append(f"| {name} | `{str(passed).lower()}` |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            conclusion,
            "",
            "This is a post-training engineering decision, not a "
            "preregistered or confirmatory paper conclusion. It does not "
            "establish universal dominance, cross-seed stability, "
            "cross-dataset generalization, or an official-test result.",
            "",
            "The original absolute six-component decision remains unchanged "
            "in its frozen source artifact and is not used as a veto here.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


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


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir).resolve()
    return (
        output_dir / JSON_OUTPUT_NAME,
        output_dir / MARKDOWN_OUTPUT_NAME,
        output_dir / COMPLETE_MARKER_NAME,
    )


def _completion_marker(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    bindings = report["bindings"]
    return {
        "schema": COMPLETE_MARKER_SCHEMA,
        "status": "complete",
        "decision": report["decision"],
        "comprehensive_relative_gate_passed": report[
            "comprehensive_relative_gate_passed"
        ],
        "inputs": {
            "source_comparison": dict(bindings["comparison"]),
            "source_completion_marker": dict(
                bindings["completion_marker"]
            ),
        },
        "outputs": {
            json_path.name: sha256_file(json_path),
            markdown_path.name: sha256_file(markdown_path),
        },
    }


def publish_report(
    report: Mapping[str, Any],
    *,
    output_dir: Path = OUTPUT_DIR,
    comparison_path: Path = SOURCE_COMPARISON,
    marker_path: Path = SOURCE_MARKER,
    expected_comparison_sha256: str = SOURCE_COMPARISON_SHA256,
    expected_marker_sha256: str = SOURCE_MARKER_SHA256,
) -> tuple[Path, Path, Path]:
    authoritative_report = aggregate(
        comparison_path=comparison_path,
        marker_path=marker_path,
        expected_comparison_sha256=expected_comparison_sha256,
        expected_marker_sha256=expected_marker_sha256,
    )
    _require(
        _json_text(report) == _json_text(authoritative_report),
        "report differs from default frozen authority",
    )
    json_path, markdown_path, marker_path = _output_paths(output_dir)
    for path in (json_path, markdown_path, marker_path):
        _require(not path.exists(), f"refusing to overwrite output: {path}")
    _atomic_write_new(json_path, _json_text(authoritative_report))
    _atomic_write_new(
        markdown_path,
        render_markdown(authoritative_report),
    )
    marker = _completion_marker(
        authoritative_report,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    _atomic_write_new(marker_path, _json_text(marker))
    return json_path, markdown_path, marker_path


def verify_published(
    *,
    comparison_path: Path = SOURCE_COMPARISON,
    marker_path: Path = SOURCE_MARKER,
    output_dir: Path = OUTPUT_DIR,
    expected_comparison_sha256: str = SOURCE_COMPARISON_SHA256,
    expected_marker_sha256: str = SOURCE_MARKER_SHA256,
) -> tuple[Path, Path, Path]:
    expected_report = aggregate(
        comparison_path=comparison_path,
        marker_path=marker_path,
        expected_comparison_sha256=expected_comparison_sha256,
        expected_marker_sha256=expected_marker_sha256,
    )
    json_path, markdown_path, complete_path = _output_paths(output_dir)
    _require(
        json_path.is_file() and not json_path.is_symlink(),
        f"published JSON missing: {json_path}",
    )
    _require(
        markdown_path.is_file() and not markdown_path.is_symlink(),
        f"published Markdown missing: {markdown_path}",
    )
    _require(
        complete_path.is_file() and not complete_path.is_symlink(),
        f"published completion marker missing: {complete_path}",
    )
    _require(
        json_path.read_text(encoding="utf-8")
        == _json_text(expected_report),
        "published comparative JSON differs on recomputation",
    )
    _require(
        markdown_path.read_text(encoding="utf-8")
        == render_markdown(expected_report),
        "published comparative Markdown differs on recomputation",
    )
    expected_marker = _completion_marker(
        expected_report,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    _require(
        load_json(complete_path) == expected_marker,
        "published comparative completion marker differs",
    )
    return json_path, markdown_path, complete_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=SOURCE_COMPARISON,
    )
    parser.add_argument(
        "--source-marker",
        type=Path,
        default=SOURCE_MARKER,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.publish:
        report = aggregate(
            comparison_path=args.comparison,
            marker_path=args.source_marker,
        )
        paths = publish_report(
            report,
            output_dir=args.output_dir,
            comparison_path=args.comparison,
            marker_path=args.source_marker,
        )
    else:
        paths = verify_published(
            comparison_path=args.comparison,
            marker_path=args.source_marker,
            output_dir=args.output_dir,
        )
        report = load_json(paths[0])
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "comprehensive_relative_gate_passed": report[
                    "comprehensive_relative_gate_passed"
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
    "COMPLETE_MARKER_NAME",
    "JSON_OUTPUT_NAME",
    "MARKDOWN_OUTPUT_NAME",
    "REFERENCE_VARIANTS",
    "V1_VARIANT",
    "V2_VARIANT",
    "V3_VARIANT",
    "V4_VARIANT",
    "aggregate",
    "budget_assessment",
    "build_report",
    "fixed_pareto_assessment",
    "global_fa_budget_leaders",
    "global_fixed_pareto_frontier",
    "model_fa_budget_envelopes",
    "pairwise_assessment",
    "publish_report",
    "render_markdown",
    "sha256_file",
    "validate_source_authority",
    "verify_published",
]
