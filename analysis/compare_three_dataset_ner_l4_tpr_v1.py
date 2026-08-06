#!/usr/bin/env python3
"""Compare the six-role NER-L4-TPR V1 zero-training screen.

The comparison has two explicit reference surfaces:

* ``gpos025_l4_only`` measures the target loss of the unprotected L4
  reallocation.  A protected mode is checked for recovery relative to it.
* ``current_g0`` is the production output.  A protected mode is checked for
  retention of the component/background false-positive reduction relative to
  it.

All requested quality metrics are reported together.  The module deliberately
does not turn any one metric into a training-authorization threshold and does
not scalarize the metrics into a weighted score.  Its only categorical output
describes whether a *joint directional signal* occurs in one or both
checkpoint roles.  ``tpr_g025`` is retained as a boundary-limit
counterfactual; it is never labelled a finite-logit trainable point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_three_dataset_ner_l4_tpr_v1 as analyzer  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA = "sctransnet_three_dataset_ner_l4_tpr_comparison_v1/v1"
ANALYZER_SCHEMA = analyzer.SCHEMA
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLES = tuple(analyzer.CHECKPOINT_ROLES)
SEED = analyzer.SEED
FIXED_THRESHOLD = analyzer.FIXED_THRESHOLD
MODES = tuple(analyzer.PUBLIC_MODES)
CURRENT_MODE = analyzer.CURRENT_MODE
UNPROTECTED_MODE = analyzer.UNPROTECTED_MODE
TPR_MODES = tuple(mode for mode in MODES if mode.startswith("tpr_g"))
BOUNDARY_LIMIT_MODE = "tpr_g025"
REPRESENTABLE_TPR_MODES = tuple(
    mode for mode in TPR_MODES if mode != BOUNDARY_LIMIT_MODE
)

ASSESSMENT_CROSS_ROLE = "REPRESENTABLE_CROSS_ROLE_JOINT_SIGNAL"
ASSESSMENT_PARTIAL = "REPRESENTABLE_PARTIAL_JOINT_SIGNAL"
ASSESSMENT_NONE = "NO_REPRESENTABLE_JOINT_SIGNAL"

DEFAULT_INPUT_ROOT = analyzer.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison" / "seed42_six_role"

SCALAR_METRICS = (
    "pd",
    "tiny_pd",
    "fa",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
)
QUALITY_METRICS = (
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
)
COUNT_METRICS = (
    "matched_target_count",
    "matched_tiny_target_count",
    "component_false_positive_pixels",
    "background_false_positive_pixels",
)


class NERL4TPRComparisonError(ValueError):
    """An analyzer artifact differs from the comparison contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NERL4TPRComparisonError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NERL4TPRComparisonError(f"{label} must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NERL4TPRComparisonError(f"{label} must be numeric")
    ready = float(value)
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NERL4TPRComparisonError(f"{label} must be an integer")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def file_sha256(path: Path) -> str:
    ready = Path(path)
    if not ready.is_file() or ready.is_symlink():
        raise FileNotFoundError(ready)
    digest = hashlib.sha256()
    with ready.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    ready = Path(path)
    if not ready.is_file() or ready.is_symlink():
        raise FileNotFoundError(ready)
    raw = ready.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise NERL4TPRComparisonError(f"expected JSON object: {ready}")
    return value, _sha256_bytes(raw)


def _extract_point(raw_mode: Mapping[str, Any], label: str) -> dict[str, Any]:
    raw = _mapping(raw_mode.get("fixed_threshold_0_5"), f"{label}.fixed_threshold_0_5")
    threshold = _finite(raw.get("threshold"), f"{label}.threshold")
    _require(threshold == FIXED_THRESHOLD, f"{label} threshold differs")

    target_count = _nonnegative_int(raw.get("target_count"), f"{label}.target_count")
    tiny_count = _nonnegative_int(
        raw.get("tiny_target_count"), f"{label}.tiny_target_count"
    )
    matched = _nonnegative_int(
        raw.get("matched_target_count"), f"{label}.matched_target_count"
    )
    matched_tiny = _nonnegative_int(
        raw.get("matched_tiny_target_count"), f"{label}.matched_tiny_target_count"
    )
    valid = _nonnegative_int(
        raw.get("valid_pixel_count"), f"{label}.valid_pixel_count"
    )
    _require(valid > 0, f"{label} valid pixels must be positive")
    _require(matched <= target_count, f"{label} matched targets exceed total")
    _require(matched_tiny <= tiny_count, f"{label} matched tiny targets exceed total")

    component_raw = _nonnegative_int(
        raw.get("unmatched_predicted_pixels"),
        f"{label}.unmatched_predicted_pixels",
    )
    component_alias = _nonnegative_int(
        raw.get("component_false_positive_pixels"),
        f"{label}.component_false_positive_pixels",
    )
    background_raw = _nonnegative_int(
        raw.get("false_positive_pixels"), f"{label}.false_positive_pixels"
    )
    background_alias = _nonnegative_int(
        raw.get("background_false_positive_pixels"),
        f"{label}.background_false_positive_pixels",
    )
    _require(
        component_alias == component_raw,
        f"{label} component false-positive alias differs",
    )
    _require(
        background_alias == background_raw,
        f"{label} background false-positive alias differs",
    )
    _require(component_raw <= valid, f"{label} component FP exceeds valid pixels")
    _require(background_raw <= valid, f"{label} background FP exceeds valid pixels")

    point = {
        "threshold": threshold,
        "target_count": target_count,
        "tiny_target_count": tiny_count,
        "matched_target_count": matched,
        "matched_tiny_target_count": matched_tiny,
        "component_false_positive_pixels": component_raw,
        "background_false_positive_pixels": background_raw,
        "valid_pixel_count": valid,
        "pd": _finite(raw.get("pd"), f"{label}.pd"),
        "tiny_pd": _finite(raw.get("tiny_pd"), f"{label}.tiny_pd"),
        "fa": _finite(raw.get("fa"), f"{label}.fa"),
        "miou": _finite(raw.get("miou"), f"{label}.miou"),
        "niou": _finite(raw.get("niou"), f"{label}.niou"),
        "pixel_precision": _finite(
            raw.get("pixel_precision"), f"{label}.pixel_precision"
        ),
        "pixel_recall": _finite(raw.get("pixel_recall"), f"{label}.pixel_recall"),
        "pixel_f1": _finite(raw.get("pixel_f1"), f"{label}.pixel_f1"),
    }
    for metric in SCALAR_METRICS:
        _require(0.0 <= point[metric] <= 1.0, f"{label}.{metric} outside [0,1]")
    _require(
        _close(point["pd"], _ratio(matched, target_count)),
        f"{label}.pd differs from matched/total counts",
    )
    _require(
        _close(point["tiny_pd"], _ratio(matched_tiny, tiny_count)),
        f"{label}.tiny_pd differs from matched/total tiny counts",
    )
    _require(
        _close(point["fa"], component_raw / valid),
        f"{label}.fa differs from unmatched component pixels / valid pixels",
    )
    return point


def _mode_representation(raw_mode: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in TPR_MODES:
        return {
            "finite_logit_representable": None,
            "boundary_limit_counterfactual": False,
            "required_logit": None,
            "eligible_as_finite_logit_candidate": False,
        }
    finite = raw_mode.get("finite_logit_representable")
    boundary = raw_mode.get("boundary_limit_counterfactual")
    _require(isinstance(finite, bool), f"modes.{mode}.finite_logit_representable differs")
    _require(isinstance(boundary, bool), f"modes.{mode}.boundary_limit_counterfactual differs")
    expected_finite = mode != BOUNDARY_LIMIT_MODE
    _require(finite is expected_finite, f"modes.{mode} finite-logit status differs")
    _require(boundary is (not expected_finite), f"modes.{mode} boundary status differs")
    raw_logit = raw_mode.get("required_logit")
    if expected_finite:
        required_logit = _finite(raw_logit, f"modes.{mode}.required_logit")
    else:
        _require(raw_logit is None, f"modes.{mode}.required_logit must be null at boundary")
        required_logit = None
    return {
        "finite_logit_representable": finite,
        "boundary_limit_counterfactual": boundary,
        "required_logit": required_logit,
        "eligible_as_finite_logit_candidate": expected_finite,
    }


def validate_analyzer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    analyzer.validate_output_payload(payload)
    _require(payload.get("schema") == ANALYZER_SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer result is incomplete")
    dataset = payload.get("dataset")
    role = payload.get("checkpoint_role")
    _require(dataset in DATASETS, "analyzer dataset differs")
    _require(role in CHECKPOINT_ROLES, "analyzer checkpoint role differs")
    _require(payload.get("seed") == SEED, "analyzer seed differs")
    _require(payload.get("test_selected") is True, "test-selected contract differs")
    _require(payload.get("mode_order") == list(MODES), "analyzer mode order differs")

    raw_modes = _mapping(payload.get("modes"), "modes")
    _require(set(raw_modes) == set(MODES), "analyzer mode set differs")
    normalized_modes: dict[str, Any] = {}
    invariant: tuple[int, int, int] | None = None
    for mode in MODES:
        raw_mode = _mapping(raw_modes[mode], f"modes.{mode}")
        point = _extract_point(raw_mode, f"modes.{mode}")
        totals = (
            point["target_count"],
            point["tiny_target_count"],
            point["valid_pixel_count"],
        )
        invariant = totals if invariant is None else invariant
        _require(totals == invariant, f"modes.{mode} dataset totals differ")
        normalized_modes[mode] = {
            "fixed_threshold_0_5": point,
            "representation": _mode_representation(raw_mode, mode),
        }

    checkpoint_sha = None
    checkpoint_binding = payload.get("checkpoint_binding")
    if isinstance(checkpoint_binding, Mapping):
        if _is_sha256(checkpoint_binding.get("sha256")):
            checkpoint_sha = checkpoint_binding.get("sha256")
        checkpoint = checkpoint_binding.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            checkpoint_sha = checkpoint.get("sha256")
    if checkpoint_sha is None:
        checkpoint_sha = payload.get("checkpoint_sha256")
    _require(_is_sha256(checkpoint_sha), "checkpoint SHA differs")

    return {
        "dataset": str(dataset),
        "checkpoint_role": str(role),
        "checkpoint_sha256": str(checkpoint_sha),
        "modes": normalized_modes,
    }


def _fp_change(candidate: int, reference: int) -> dict[str, Any]:
    _require(candidate >= 0 and reference >= 0, "FP counts must be non-negative")
    if candidate < reference:
        direction = "decrease"
    elif candidate > reference:
        direction = "increase"
    else:
        direction = "equal"
    relative_reduction: float | None
    if reference == 0:
        relative_reduction = 0.0 if candidate == 0 else None
    else:
        relative_reduction = (reference - candidate) / reference
    return {
        "candidate_count": candidate,
        "reference_count": reference,
        "candidate_minus_reference": candidate - reference,
        "reference_minus_candidate": reference - candidate,
        "relative_reduction": relative_reduction,
        "reference_denominator_zero": reference == 0,
        "direction": direction,
        "decreased": candidate < reference,
        "nonincreased": candidate <= reference,
    }


def compare_points(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    for total in ("target_count", "tiny_target_count", "valid_pixel_count"):
        _require(candidate[total] == reference[total], f"comparison {total} differs")
    deltas = {
        metric: candidate[metric] - reference[metric]
        for metric in (*COUNT_METRICS, *SCALAR_METRICS)
    }
    return {
        "candidate_minus_reference": deltas,
        "component_false_positive_change": _fp_change(
            int(candidate["component_false_positive_pixels"]),
            int(reference["component_false_positive_pixels"]),
        ),
        "background_false_positive_change": _fp_change(
            int(candidate["background_false_positive_pixels"]),
            int(reference["background_false_positive_pixels"]),
        ),
    }


def _joint_evidence(
    candidate: Mapping[str, Any],
    unprotected: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    versus_unprotected = compare_points(candidate, unprotected)
    versus_current = compare_points(candidate, current)
    target_recovered = (
        candidate["matched_target_count"] > unprotected["matched_target_count"]
    )
    tiny_recovered = (
        candidate["matched_tiny_target_count"]
        > unprotected["matched_tiny_target_count"]
    )
    component_decreased = (
        candidate["component_false_positive_pixels"]
        < current["component_false_positive_pixels"]
    )
    background_decreased = (
        candidate["background_false_positive_pixels"]
        < current["background_false_positive_pixels"]
    )
    fa_decreased = candidate["fa"] < current["fa"]
    any_target_recovery = target_recovered or tiny_recovered
    both_fp_decreased = component_decreased and background_decreased
    all_fp_fa_decreased = both_fp_decreased and fa_decreased
    return {
        "candidate_vs_unprotected": versus_unprotected,
        "candidate_vs_current": versus_current,
        "target_recovery_vs_unprotected": {
            "matched_target_count_recovered": target_recovered,
            "pd_value_improved": candidate["pd"] > unprotected["pd"],
            "matched_tiny_target_count_recovered": tiny_recovered,
            "tiny_pd_value_improved": candidate["tiny_pd"] > unprotected["tiny_pd"],
            "any_total_or_tiny_target_recovery": any_target_recovery,
        },
        "false_positive_benefit_vs_current": {
            "component_pixels_decreased": component_decreased,
            "component_pixels_nonincreased": (
                candidate["component_false_positive_pixels"]
                <= current["component_false_positive_pixels"]
            ),
            "background_pixels_decreased": background_decreased,
            "background_pixels_nonincreased": (
                candidate["background_false_positive_pixels"]
                <= current["background_false_positive_pixels"]
            ),
            "fa_decreased": fa_decreased,
            "fa_nonincreased": candidate["fa"] <= current["fa"],
            "both_component_and_background_decreased": both_fp_decreased,
            "component_background_and_fa_decreased": all_fp_fa_decreased,
        },
        "restoration_to_current": {
            "matched_target_count_fully_restored": (
                candidate["matched_target_count"] >= current["matched_target_count"]
            ),
            "matched_tiny_target_count_fully_restored": (
                candidate["matched_tiny_target_count"]
                >= current["matched_tiny_target_count"]
            ),
        },
        "quality_direction_vs_current": {
            metric: (
                "increase"
                if candidate[metric] > current[metric]
                else "decrease"
                if candidate[metric] < current[metric]
                else "equal"
            )
            for metric in QUALITY_METRICS
        },
        "joint_target_recovery_and_both_fp_decrease": (
            any_target_recovery and both_fp_decreased
        ),
        "joint_target_recovery_and_component_background_fa_decrease": (
            any_target_recovery and all_fp_fa_decreased
        ),
    }


def _binding_key(dataset: str, role: str) -> str:
    return f"{dataset}::{role}"


def _expected_keys() -> tuple[str, ...]:
    return tuple(
        _binding_key(dataset, role)
        for dataset in DATASETS
        for role in CHECKPOINT_ROLES
    )


def _validate_input_bindings(
    input_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    _require(set(input_bindings) == set(_expected_keys()), "input bindings require six roles")
    ready: dict[str, dict[str, str]] = {}
    for key in _expected_keys():
        binding = _mapping(input_bindings[key], f"input_bindings.{key}")
        path = binding.get("path")
        sha = binding.get("sha256")
        _require(isinstance(path, str) and bool(path), f"input path differs: {key}")
        _require(_is_sha256(sha), f"input SHA differs: {key}")
        ready[key] = {"path": path, "sha256": str(sha)}
    return ready


def _mean(values: Iterable[float]) -> float:
    ready = list(values)
    _require(bool(ready), "cannot average an empty sequence")
    return math.fsum(ready) / len(ready)


def _aggregate_mode(
    mode: str,
    per_unit: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [per_unit[key]["tpr_modes"][mode] for key in _expected_keys()]
    role_rows: dict[str, Any] = {}
    for role in CHECKPOINT_ROLES:
        selected = [
            per_unit[_binding_key(dataset, role)]["tpr_modes"][mode]
            for dataset in DATASETS
        ]
        joint_units = [
            row["binding_key"]
            for row in selected
            if row["evidence"]["joint_target_recovery_and_both_fp_decrease"]
        ]
        role_rows[role] = {
            "joint_signal_units": joint_units,
            "joint_signal_unit_count": len(joint_units),
            "target_recovery_unit_count": sum(
                row["evidence"]["target_recovery_vs_unprotected"][
                    "any_total_or_tiny_target_recovery"
                ]
                for row in selected
            ),
            "both_fp_decrease_unit_count": sum(
                row["evidence"]["false_positive_benefit_vs_current"][
                    "both_component_and_background_decreased"
                ]
                for row in selected
            ),
        }

    same_dataset_cross_role = [
        dataset
        for dataset in DATASETS
        if all(
            per_unit[_binding_key(dataset, role)]["tpr_modes"][mode]["evidence"][
                "joint_target_recovery_and_both_fp_decrease"
            ]
            for role in CHECKPOINT_ROLES
        )
    ]
    current_points = [per_unit[key]["current_g0"] for key in _expected_keys()]
    unprotected_points = [
        per_unit[key]["unprotected_gpos025_l4_only"] for key in _expected_keys()
    ]
    candidate_points = [row["point"] for row in rows]

    sum_counts: dict[str, Any] = {}
    for metric in COUNT_METRICS:
        current_sum = sum(int(point[metric]) for point in current_points)
        unprotected_sum = sum(int(point[metric]) for point in unprotected_points)
        candidate_sum = sum(int(point[metric]) for point in candidate_points)
        sum_counts[metric] = {
            "current_g0": current_sum,
            "gpos025_l4_only": unprotected_sum,
            "candidate": candidate_sum,
            "candidate_minus_unprotected": candidate_sum - unprotected_sum,
            "candidate_minus_current": candidate_sum - current_sum,
        }
    macro_metrics: dict[str, Any] = {}
    for metric in SCALAR_METRICS:
        current_mean = _mean(float(point[metric]) for point in current_points)
        unprotected_mean = _mean(float(point[metric]) for point in unprotected_points)
        candidate_mean = _mean(float(point[metric]) for point in candidate_points)
        macro_metrics[metric] = {
            "current_g0": current_mean,
            "gpos025_l4_only": unprotected_mean,
            "candidate": candidate_mean,
            "candidate_minus_unprotected": candidate_mean - unprotected_mean,
            "candidate_minus_current": candidate_mean - current_mean,
        }

    joint_units = [
        row["binding_key"]
        for row in rows
        if row["evidence"]["joint_target_recovery_and_both_fp_decrease"]
    ]
    return {
        "mode": mode,
        "representation": rows[0]["representation"],
        "unit_count": len(rows),
        "target_recovery_unit_count": sum(
            row["evidence"]["target_recovery_vs_unprotected"][
                "any_total_or_tiny_target_recovery"
            ]
            for row in rows
        ),
        "total_target_recovery_unit_count": sum(
            row["evidence"]["target_recovery_vs_unprotected"][
                "matched_target_count_recovered"
            ]
            for row in rows
        ),
        "tiny_target_recovery_unit_count": sum(
            row["evidence"]["target_recovery_vs_unprotected"][
                "matched_tiny_target_count_recovered"
            ]
            for row in rows
        ),
        "both_fp_decrease_unit_count": sum(
            row["evidence"]["false_positive_benefit_vs_current"][
                "both_component_and_background_decreased"
            ]
            for row in rows
        ),
        "joint_signal_units": joint_units,
        "joint_signal_unit_count": len(joint_units),
        "joint_signal_present_in_both_checkpoint_roles": all(
            role_rows[role]["joint_signal_unit_count"] > 0 for role in CHECKPOINT_ROLES
        ),
        "same_dataset_joint_signal_in_both_roles": same_dataset_cross_role,
        "sum_counts": sum_counts,
        "macro_metrics": macro_metrics,
        "by_checkpoint_role": role_rows,
    }


def _pareto_objectives(row: Mapping[str, Any]) -> dict[str, float]:
    counts = row["sum_counts"]
    metrics = row["macro_metrics"]
    return {
        "target_recovery_count_vs_unprotected": float(
            counts["matched_target_count"]["candidate_minus_unprotected"]
        ),
        "tiny_target_recovery_count_vs_unprotected": float(
            counts["matched_tiny_target_count"]["candidate_minus_unprotected"]
        ),
        "component_fp_benefit_vs_current": float(
            -counts["component_false_positive_pixels"]["candidate_minus_current"]
        ),
        "background_fp_benefit_vs_current": float(
            -counts["background_false_positive_pixels"]["candidate_minus_current"]
        ),
        "fa_benefit_vs_current": float(-metrics["fa"]["candidate_minus_current"]),
        "miou_delta_vs_current": float(metrics["miou"]["candidate_minus_current"]),
        "niou_delta_vs_current": float(metrics["niou"]["candidate_minus_current"]),
        "precision_delta_vs_current": float(
            metrics["pixel_precision"]["candidate_minus_current"]
        ),
        "recall_delta_vs_current": float(
            metrics["pixel_recall"]["candidate_minus_current"]
        ),
        "f1_delta_vs_current": float(metrics["pixel_f1"]["candidate_minus_current"]),
    }


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = tuple(left)
    _require(set(keys) == set(right), "Pareto objective sets differ")
    no_worse = all(left[key] >= right[key] for key in keys)
    strictly_better = any(left[key] > right[key] for key in keys)
    return no_worse and strictly_better


def _pareto_summary(mode_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    objectives = {mode: _pareto_objectives(mode_rows[mode]) for mode in TPR_MODES}
    non_dominated = [
        mode
        for mode in TPR_MODES
        if not any(
            other != mode and _dominates(objectives[other], objectives[mode])
            for other in TPR_MODES
        )
    ]
    representable = [
        mode
        for mode in REPRESENTABLE_TPR_MODES
        if not any(
            other != mode and _dominates(objectives[other], objectives[mode])
            for other in REPRESENTABLE_TPR_MODES
        )
    ]
    return {
        "method": "non_dominated_joint_objectives_without_weighted_scalarization",
        "larger_is_better_for_every_objective": True,
        "objectives": objectives,
        "all_tpr_non_dominated_modes": non_dominated,
        "finite_logit_representable_non_dominated_modes": representable,
        "boundary_limit_mode_reported_descriptively": BOUNDARY_LIMIT_MODE,
        "boundary_limit_mode_eligible_as_finite_logit_candidate": False,
    }


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    input_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    _require(set(payloads) == set(_expected_keys()), "comparison requires six payloads")
    bindings = _validate_input_bindings(input_bindings)
    normalized: dict[str, Any] = {}
    for key in _expected_keys():
        one = validate_analyzer_payload(payloads[key])
        dataset, role = key.split("::", 1)
        _require(one["dataset"] == dataset, f"dataset binding differs: {key}")
        _require(one["checkpoint_role"] == role, f"role binding differs: {key}")
        normalized[key] = one

    per_unit: dict[str, Any] = {}
    for key in _expected_keys():
        one = normalized[key]
        current = one["modes"][CURRENT_MODE]["fixed_threshold_0_5"]
        unprotected = one["modes"][UNPROTECTED_MODE]["fixed_threshold_0_5"]
        unprotected_vs_current = compare_points(unprotected, current)
        candidates: dict[str, Any] = {}
        for mode in TPR_MODES:
            candidate = one["modes"][mode]["fixed_threshold_0_5"]
            candidates[mode] = {
                "binding_key": key,
                "point": candidate,
                "representation": one["modes"][mode]["representation"],
                "evidence": _joint_evidence(candidate, unprotected, current),
            }
        per_unit[key] = {
            "dataset": one["dataset"],
            "checkpoint_role": one["checkpoint_role"],
            "checkpoint_sha256": one["checkpoint_sha256"],
            "current_g0": current,
            "unprotected_gpos025_l4_only": unprotected,
            "unprotected_vs_current": unprotected_vs_current,
            "tpr_modes": candidates,
        }

    mode_rows = {mode: _aggregate_mode(mode, per_unit) for mode in TPR_MODES}
    cross_role_modes = [
        mode
        for mode in REPRESENTABLE_TPR_MODES
        if mode_rows[mode]["joint_signal_present_in_both_checkpoint_roles"]
    ]
    partial_modes = [
        mode
        for mode in REPRESENTABLE_TPR_MODES
        if mode_rows[mode]["joint_signal_unit_count"] > 0
    ]
    if cross_role_modes:
        assessment = ASSESSMENT_CROSS_ROLE
    elif partial_modes:
        assessment = ASSESSMENT_PARTIAL
    else:
        assessment = ASSESSMENT_NONE

    return {
        "schema": SCHEMA,
        "status": "complete",
        "assessment": assessment,
        "seed": SEED,
        "test_selected": True,
        "fixed_threshold": FIXED_THRESHOLD,
        "datasets": list(DATASETS),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "mode_order": list(MODES),
        "current_reference_mode": CURRENT_MODE,
        "unprotected_reference_mode": UNPROTECTED_MODE,
        "tpr_modes": list(TPR_MODES),
        "finite_logit_representable_tpr_modes": list(REPRESENTABLE_TPR_MODES),
        "boundary_limit_counterfactual_mode": BOUNDARY_LIMIT_MODE,
        "joint_directional_assessment": {
            "rule": (
                "within_the_same_dataset_role_unit_any_total_or_tiny_target_recovery_"
                "vs_gpos025_l4_only_and_both_component_and_background_fp_decrease_"
                "vs_current_g0"
            ),
            "uses_single_metric_hard_threshold": False,
            "uses_weighted_metric_sum": False,
            "training_authorization_made": False,
            "representable_cross_role_joint_modes": cross_role_modes,
            "representable_any_joint_modes": partial_modes,
            "boundary_limit_mode_excluded_from_finite_logit_signal": True,
        },
        "per_mode": mode_rows,
        "per_unit": per_unit,
        "pareto": _pareto_summary(mode_rows),
        "input_bindings": bindings,
        "scope": {
            "zero_training_test_selected_screen": True,
            "formal_training_or_analysis_executed_by_comparator": False,
            "single_seed": True,
            "fixed_threshold_only_for_joint_comparison": True,
            "threshold_1_0_empty_endpoint_not_used_for_comparison": True,
            "stability_claim_supported": False,
            "paper_claim_established": False,
        },
        "source_sha256": {
            "analysis/compare_three_dataset_ner_l4_tpr_v1.py": file_sha256(Path(__file__)),
            "analysis/analyze_three_dataset_ner_l4_tpr_v1.py": file_sha256(
                Path(analyzer.__file__)
            ),
        },
        "no_fabricated_results": True,
    }


def validate_comparison_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "comparison schema differs")
    _require(payload.get("status") == "complete", "comparison is incomplete")
    assessment = payload.get("assessment")
    _require(
        assessment in (ASSESSMENT_CROSS_ROLE, ASSESSMENT_PARTIAL, ASSESSMENT_NONE),
        "comparison assessment differs",
    )
    joint = _mapping(payload.get("joint_directional_assessment"), "joint assessment")
    _require(joint.get("uses_single_metric_hard_threshold") is False, "single-metric gate used")
    _require(joint.get("uses_weighted_metric_sum") is False, "weighted score used")
    _require(joint.get("training_authorization_made") is False, "training was authorized")
    cross_role = joint.get("representable_cross_role_joint_modes")
    partial = joint.get("representable_any_joint_modes")
    _require(isinstance(cross_role, list) and isinstance(partial, list), "joint mode lists differ")
    expected_assessment = (
        ASSESSMENT_CROSS_ROLE
        if cross_role
        else ASSESSMENT_PARTIAL
        if partial
        else ASSESSMENT_NONE
    )
    _require(assessment == expected_assessment, "assessment differs from joint evidence")
    _require(set(payload.get("per_unit", {})) == set(_expected_keys()), "per-unit matrix differs")
    _require(set(payload.get("per_mode", {})) == set(TPR_MODES), "per-mode matrix differs")
    _require(set(payload.get("input_bindings", {})) == set(_expected_keys()), "bindings differ")
    pareto = _mapping(payload.get("pareto"), "pareto")
    _require(
        pareto.get("boundary_limit_mode_eligible_as_finite_logit_candidate") is False,
        "boundary mode was labelled finite-logit",
    )
    expected_sources = {
        "analysis/compare_three_dataset_ner_l4_tpr_v1.py": file_sha256(Path(__file__)),
        "analysis/analyze_three_dataset_ner_l4_tpr_v1.py": file_sha256(
            Path(analyzer.__file__)
        ),
    }
    _require(payload.get("source_sha256") == expected_sources, "comparison source lock differs")


def _format_float(value: Any, *, signed: bool = False) -> str:
    return f"{float(value):+.6f}" if signed else f"{float(value):.6f}"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# NER-L4-TPR V1 零训练联合比较",
        "",
        f"- 联合方向性结果：`{result['assessment']}`",
        "- 目标恢复参照：`gpos025_l4_only`；FP/Fa 保留参照：`current_g0`。",
        "- 没有使用单指标硬门，也没有把异质量纲指标加权求和。",
        "- 本比较器不授权训练，只报告固定 seed42、测试集选择式筛选信号。",
        "- `tpr_g025` 是 `G=0.25*tanh(a)` 的开区间上界极限，不是有限 logit 可训练点。",
        "",
        "## 模式联合汇总",
        "",
        "| 模式 | 有限logit | 目标恢复单元 | tiny恢复单元 | 两类FP下降单元 | 联合单元 | best_mIoU联合 | best_Pd联合 | ΣΔ目标 vs无保护 | ΣΔcomponent FP vs当前 | ΣΔbackground FP vs当前 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in TPR_MODES:
        row = result["per_mode"][mode]
        representation = row["representation"]
        lines.append(
            f"| `{mode}` | {str(bool(representation['finite_logit_representable'])).lower()} | "
            f"{row['total_target_recovery_unit_count']}/6 | "
            f"{row['tiny_target_recovery_unit_count']}/6 | "
            f"{row['both_fp_decrease_unit_count']}/6 | {row['joint_signal_unit_count']}/6 | "
            f"{row['by_checkpoint_role']['best_miou']['joint_signal_unit_count']}/3 | "
            f"{row['by_checkpoint_role']['best_pd']['joint_signal_unit_count']}/3 | "
            f"{row['sum_counts']['matched_target_count']['candidate_minus_unprotected']:+d} | "
            f"{row['sum_counts']['component_false_positive_pixels']['candidate_minus_current']:+d} | "
            f"{row['sum_counts']['background_false_positive_pixels']['candidate_minus_current']:+d} |"
        )
    lines.extend(
        [
            "",
            "## 宏平均数值与差值",
            "",
            "| 模式 | Pd | ΔPd vs无保护 | tiny-Pd | Δtiny-Pd vs无保护 | Fa | ΔFa vs当前 | mIoU | ΔmIoU vs当前 | nIoU | ΔnIoU vs当前 | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in TPR_MODES:
        metrics = result["per_mode"][mode]["macro_metrics"]
        lines.append(
            f"| `{mode}` | {_format_float(metrics['pd']['candidate'])} | "
            f"{_format_float(metrics['pd']['candidate_minus_unprotected'], signed=True)} | "
            f"{_format_float(metrics['tiny_pd']['candidate'])} | "
            f"{_format_float(metrics['tiny_pd']['candidate_minus_unprotected'], signed=True)} | "
            f"{_format_float(metrics['fa']['candidate'])} | "
            f"{_format_float(metrics['fa']['candidate_minus_current'], signed=True)} | "
            f"{_format_float(metrics['miou']['candidate'])} | "
            f"{_format_float(metrics['miou']['candidate_minus_current'], signed=True)} | "
            f"{_format_float(metrics['niou']['candidate'])} | "
            f"{_format_float(metrics['niou']['candidate_minus_current'], signed=True)} | "
            f"{_format_float(metrics['pixel_precision']['candidate'])} | "
            f"{_format_float(metrics['pixel_recall']['candidate'])} | "
            f"{_format_float(metrics['pixel_f1']['candidate'])} |"
        )
    lines.extend(["", "## 六个 dataset×role 原始工作点", ""])
    for key in _expected_keys():
        unit = result["per_unit"][key]
        lines.extend(
            [
                f"### {unit['dataset']} — {unit['checkpoint_role']}",
                "",
                "| 模式 | Pd(计数) | Pd | tiny-Pd(计数) | tiny-Pd | component FP | background FP | Fa | mIoU | nIoU | Precision | Recall | F1 | 联合信号 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        display_rows = [
            (CURRENT_MODE, unit["current_g0"], False),
            (UNPROTECTED_MODE, unit["unprotected_gpos025_l4_only"], False),
        ] + [
            (
                mode,
                unit["tpr_modes"][mode]["point"],
                unit["tpr_modes"][mode]["evidence"][
                    "joint_target_recovery_and_both_fp_decrease"
                ],
            )
            for mode in TPR_MODES
        ]
        for mode, point, joint in display_rows:
            lines.append(
                f"| `{mode}` | {point['matched_target_count']}/{point['target_count']} | "
                f"{_format_float(point['pd'])} | "
                f"{point['matched_tiny_target_count']}/{point['tiny_target_count']} | "
                f"{_format_float(point['tiny_pd'])} | "
                f"{point['component_false_positive_pixels']} | "
                f"{point['background_false_positive_pixels']} | {_format_float(point['fa'])} | "
                f"{_format_float(point['miou'])} | {_format_float(point['niou'])} | "
                f"{_format_float(point['pixel_precision'])} | "
                f"{_format_float(point['pixel_recall'])} | "
                f"{_format_float(point['pixel_f1'])} | {str(bool(joint)).lower()} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Pareto 说明",
            "",
            "联合 Pareto 只做非支配筛选，不进行原始指标求和或加权评分。有限-logit "
            f"Pareto 模式为：`{', '.join(result['pareto']['finite_logit_representable_non_dominated_modes']) or 'none'}`。",
            f"边界模式 `{BOUNDARY_LIMIT_MODE}` 始终只作极限反事实描述。",
            "",
        ]
    )
    return "\n".join(lines)


def _default_input(dataset: str, role: str) -> Path:
    return (
        DEFAULT_INPUT_ROOT
        / "runs"
        / dataset
        / f"final_tss_off_{role}_seed42"
        / "evaluation.json"
    )


def _parse_bindings(values: Sequence[str]) -> dict[str, Path]:
    if not values:
        return {
            _binding_key(dataset, role): _default_input(dataset, role)
            for dataset in DATASETS
            for role in CHECKPOINT_ROLES
        }
    ready: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise NERL4TPRComparisonError("--input must use DATASET::ROLE=PATH")
        key, raw_path = value.split("=", 1)
        _require(key in _expected_keys(), f"unknown input key: {key}")
        _require(key not in ready, f"duplicate input key: {key}")
        ready[key] = Path(raw_path)
    _require(set(ready) == set(_expected_keys()), "--input must provide all six bindings")
    return ready


def _atomic_write_once(path: Path, text: str) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(json_path: Path, markdown_path: Path, result: Mapping[str, Any]) -> None:
    validate_comparison_payload(result)
    if json_path.exists() or json_path.is_symlink():
        raise FileExistsError(f"refusing existing output: {json_path}")
    if markdown_path.exists() or markdown_path.is_symlink():
        raise FileExistsError(f"refusing existing output: {markdown_path}")
    json_text = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    _atomic_write_once(json_path, json_text)
    _atomic_write_once(markdown_path, render_markdown(result))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="DATASET::ROLE=PATH; omit all six to use formal defaults",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "decision.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "decision.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = _parse_bindings(args.input)
    payloads: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, Mapping[str, str]] = {}
    for key in _expected_keys():
        payload, sha = _load_json_with_sha(paths[key])
        payloads[key] = payload
        bindings[key] = {"path": str(paths[key].resolve()), "sha256": sha}
    result = compare_payloads(payloads, input_bindings=bindings)
    write_outputs(args.output_json, args.output_markdown, result)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete",
                "assessment": result["assessment"],
                "output_json": str(args.output_json.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
