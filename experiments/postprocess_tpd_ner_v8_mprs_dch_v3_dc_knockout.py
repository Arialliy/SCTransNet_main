#!/usr/bin/env python3
"""Validate and publish the diagnostic-only V3 DC-knockout eight-row matrix.

This module never launches an evaluator.  It consumes the two independently
produced checkpoint JSON files (four fixed knockout evaluations each), binds
them to the diagnostic source lock and immutable formal V3 references, and
publishes only beneath the separate diagnostic result root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock as freezer,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout as knockout_eval,
)
from experiments import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)


SCHEMA = spec.AGGREGATE_SCHEMA
COMPLETE_MARKER_SCHEMA = spec.COMPLETE_MARKER_SCHEMA
DEFAULT_SOURCE_LOCK = freezer.DEFAULT_SOURCE_LOCK
DEFAULT_OUTPUT_ROOT = spec.DEFAULT_OUTPUT_ROOT
DEFAULT_FORMAL_REPORT = freezer.DEFAULT_FORMAL_REPORT
LAST_FLOAT32_BELOW_ONE = 0.9999999403953552
FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "decision",
        "performance_gate_assessment",
        "aggregate_full_model_gate_passed",
        "required_decision_components",
        "success_components",
        "preregistered_required_components",
    }
)


class IncompleteDiagnostic(RuntimeError):
    """The fixed knockout package is absent or only partially complete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    _require(
        value.is_file() and not value.is_symlink(),
        f"{label} must be a regular non-symlink file: {value}",
    )
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    content = _regular_file(path, label).read_text(encoding="utf-8")
    try:
        value = json.loads(
            content,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, "hashed artifact").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    if minimum is not None:
        _require(value >= minimum, f"{label} is below {minimum}")
    if maximum is not None:
        _require(value <= maximum, f"{label} exceeds {maximum}")
    return value


def _normalize_raw_point(
    value: Any,
    *,
    location: str,
) -> dict[str, Any]:
    point = dict(_mapping(value, location))
    _require(
        set(point) == set(spec.RAW_POINT_FIELDS),
        f"{location} raw metric field set differs",
    )
    target_count = _integer(
        point.get("target_count"),
        f"{location}.target_count",
        minimum=1,
    )
    tiny_target_count = _integer(
        point.get("tiny_target_count"),
        f"{location}.tiny_target_count",
        minimum=1,
    )
    _require(
        target_count == spec.TARGET_COUNT,
        f"{location}.target_count differs",
    )
    _require(
        tiny_target_count == spec.TINY_TARGET_COUNT,
        f"{location}.tiny_target_count differs",
    )
    matched = _integer(
        point.get("matched_target_count"),
        f"{location}.matched_target_count",
        minimum=0,
        maximum=target_count,
    )
    matched_tiny = _integer(
        point.get("matched_tiny_target_count"),
        f"{location}.matched_tiny_target_count",
        minimum=0,
        maximum=tiny_target_count,
    )
    for name in (
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    ):
        _integer(point.get(name), f"{location}.{name}", minimum=0)
    normalized = {
        name: _finite(point.get(name), f"{location}.{name}")
        for name in (
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
            "threshold",
        )
    }
    for name in (
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
        "fa",
        "threshold",
    ):
        _require(
            0.0 <= normalized[name] <= 1.0,
            f"{location}.{name} lies outside [0, 1]",
        )
    _require(
        normalized["val_loss"] >= 0.0
        and normalized["false_objects_per_image"] >= 0.0,
        f"{location} loss/false-object metric is negative",
    )
    _require(
        normalized["pd"] == matched / target_count,
        f"{location}.pd differs from counts",
    )
    _require(
        normalized["tiny_pd"] == matched_tiny / tiny_target_count,
        f"{location}.tiny_pd differs from counts",
    )
    return copy.deepcopy(point)


def _normalize_fixed(
    value: Any,
    *,
    location: str,
) -> dict[str, Any]:
    point = dict(_mapping(value, location))
    _require(
        set(spec.FIXED_THRESHOLD_FIELDS).issubset(point),
        f"{location} fixed metric coverage is incomplete",
    )
    _require(point.get("threshold") == 0.5, f"{location} threshold is not 0.5")
    target_count = _integer(
        point.get("target_count"),
        f"{location}.target_count",
        minimum=1,
    )
    tiny_count = _integer(
        point.get("tiny_target_count"),
        f"{location}.tiny_target_count",
        minimum=1,
    )
    _require(
        target_count == spec.TARGET_COUNT
        and tiny_count == spec.TINY_TARGET_COUNT,
        f"{location} target counts differ",
    )
    matched = _integer(
        point.get("matched_target_count"),
        f"{location}.matched_target_count",
        minimum=0,
        maximum=target_count,
    )
    matched_tiny = _integer(
        point.get("matched_tiny_target_count"),
        f"{location}.matched_tiny_target_count",
        minimum=0,
        maximum=tiny_count,
    )
    result = {
        name: copy.deepcopy(point[name])
        for name in spec.FIXED_THRESHOLD_FIELDS
    }
    for name in (
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
    ):
        result[name] = _finite(result[name], f"{location}.{name}")
    _require(
        result["pd"] == matched / target_count,
        f"{location}.pd differs from counts",
    )
    _require(
        result["tiny_pd"] == matched_tiny / tiny_count,
        f"{location}.tiny_pd differs from counts",
    )
    for name in (
        "threshold",
        "pd",
        "fa",
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "tiny_pd",
    ):
        _require(
            0.0 <= result[name] <= 1.0,
            f"{location}.{name} lies outside [0, 1]",
        )
    _require(
        result["false_objects_per_image"] >= 0.0,
        f"{location}.false_objects_per_image is negative",
    )
    return result


def _best_point_under_budget(
    points: Sequence[Mapping[str, Any]],
    budget: float,
) -> Mapping[str, Any]:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    _require(bool(feasible), f"no raw point lies under Fa budget {budget}")
    return max(
        feasible,
        key=lambda point: (
            float(point["pd"]),
            -float(point["fa"]),
            float(point["tiny_pd"]),
            float(point["miou"]),
            -abs(float(point["threshold"]) - 0.5),
        ),
    )


def _normalize_budgets(
    value: Any,
    *,
    points: Sequence[Mapping[str, Any]],
    location: str,
) -> dict[str, Any]:
    raw = _mapping(value, location)
    _require(
        set(raw) == set(spec.BUDGET_KEYS),
        f"{location} Fa-budget keys differ",
    )
    normalized: dict[str, Any] = {}
    for budget, key in zip(spec.FA_BUDGETS, spec.BUDGET_KEYS):
        point = dict(_mapping(raw[key], f"{location}.{key}"))
        expected = _best_point_under_budget(points, budget)
        _require(
            point == expected,
            f"{location}.{key} is not the optimum raw point",
        )
        normalized[key] = {
            "budget": budget,
            "pd": _finite(point.get("pd"), f"{location}.{key}.pd"),
            "achieved_fa": _finite(
                point.get("fa"),
                f"{location}.{key}.fa",
            ),
            "threshold": _finite(
                point.get("threshold"),
                f"{location}.{key}.threshold",
            ),
            "matched_target_count": _integer(
                point.get("matched_target_count"),
                f"{location}.{key}.matched_target_count",
                minimum=0,
                maximum=spec.TARGET_COUNT,
            ),
            "target_count": _integer(
                point.get("target_count"),
                f"{location}.{key}.target_count",
                minimum=spec.TARGET_COUNT,
                maximum=spec.TARGET_COUNT,
            ),
        }
        _require(
            0.0 <= normalized[key]["achieved_fa"] <= budget,
            f"{location}.{key} exceeds the Fa budget",
        )
        _require(
            normalized[key]["pd"]
            == (
                normalized[key]["matched_target_count"]
                / normalized[key]["target_count"]
            ),
            f"{location}.{key}.pd differs from counts",
        )
    return normalized


def _validate_metric_coverage(
    value: Any,
    *,
    fixed: Mapping[str, Any],
    budgets: Mapping[str, Any],
    location: str,
) -> None:
    coverage = _mapping(value, location)
    expected = {
        "schema": (
            "sctransnet_tpd_ner_v8_mprs_dch_v3_"
            "dc_knockout_final_metric_coverage_v1"
        ),
        "fixed_threshold": 0.5,
        "required_fixed_threshold_fields": list(
            spec.FIXED_THRESHOLD_FIELDS
        ),
        "fixed_threshold_0_5": dict(fixed),
        "required_fa_budget_keys": list(spec.BUDGET_KEYS),
        "pd_at_fa_budget": copy.deepcopy(dict(budgets)),
        "all_required_metrics_present": True,
    }
    # Evaluators may reuse the locked formal coverage schema; every semantic
    # field must still match exactly, while the schema name may be either the
    # dedicated diagnostic version or the locked formal V3 version.
    observed = dict(coverage)
    schema_name = observed.pop("schema", None)
    expected.pop("schema")
    _require(
        schema_name
        in {
            (
                "sctransnet_tpd_ner_v8_mprs_dch_v3_"
                "dc_knockout_final_metric_coverage_v1"
            ),
            "sctransnet_tpd_ner_v8_mprs_dch_v3_final_metric_coverage_v1",
        },
        f"{location} schema differs",
    )
    _require(observed == expected, f"{location} content differs")


def _normalize_offset_records(
    value: Any,
    *,
    location: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    records = dict(_mapping(value, location))
    _require(
        set(records) == set(spec.DC_OFFSET_KEYS),
        f"{location} DC-offset key set differs",
    )
    normalized: dict[str, Any] = {}
    scalar_values: dict[str, float] = {}
    for key in spec.DC_OFFSET_KEYS:
        record = dict(_mapping(records[key], f"{location}.{key}"))
        _require(
            record.get("shape") == [1]
            and record.get("dtype") == "float32",
            f"{location}.{key} tensor identity differs",
        )
        scalar = _finite(record.get("value"), f"{location}.{key}.value")
        digest = _sha256_value(
            record.get("tensor_sha256"),
            f"{location}.{key}.tensor_sha256",
        )
        normalized[key] = {
            "shape": [1],
            "dtype": "float32",
            "value": scalar,
            "tensor_sha256": digest,
        }
        scalar_values[key] = scalar
    return normalized, scalar_values


def _deep_validate_evaluator_output(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    expected_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-run the evaluator's CPU-only identity validator before aggregation."""

    artifact_audit = knockout_eval.formal_evaluator.validate_run_artifacts(
        spec.FORMAL_RUN_DIR,
        checkpoint,
    )
    checkpoint_path = spec.FORMAL_RUN_DIR / checkpoint
    checkpoint_payload = torch.load(
        _regular_file(
            checkpoint_path,
            f"formal source checkpoint {checkpoint}",
        ),
        map_location="cpu",
        weights_only=False,
    )
    _require(
        isinstance(checkpoint_payload, Mapping),
        f"formal source checkpoint is not a mapping: {checkpoint}",
    )
    ready = knockout_eval.validate_evaluation_payload(
        payload,
        checkpoint_payload,
        artifact_audit,
        expected_source_binding,
    )
    expected_artifact_hashes = knockout_eval._artifact_hashes(
        checkpoint_path=checkpoint_path,
        artifact_audit=artifact_audit,
        source_binding=expected_source_binding,
    )
    _require(
        ready.get("artifact_sha256") == expected_artifact_hashes,
        f"{checkpoint} evaluator artifact SHA registry differs",
    )
    return ready


def _validate_evaluation(
    value: Any,
    *,
    checkpoint: str,
    source: Mapping[str, Any],
    original_offsets: Mapping[str, Any],
    expected_source_binding: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    evaluation = dict(_mapping(value, location))
    mode = evaluation.get("knockout_mode")
    _require(mode in spec.KNOCKOUT_MODES, f"{location}.mode differs")
    if "status" in evaluation:
        _require(
            evaluation.get("status") == "complete",
            f"{location}.status is not complete",
        )
    zeroed = evaluation.get("zeroed_state_keys")
    _require(
        zeroed == list(spec.KNOCKOUT_ZERO_KEYS[mode]),
        f"{location}.zeroed_state_keys differs",
    )
    changed = evaluation.get("effective_changed_state_keys")
    _require(
        isinstance(changed, list)
        and set(changed).issubset(spec.KNOCKOUT_ZERO_KEYS[mode]),
        f"{location}.effective_changed_state_keys escapes requested keys",
    )
    offset_records, offsets = _normalize_offset_records(
        evaluation.get("evaluated_dc_offsets"),
        location=f"{location}.evaluated_dc_offsets",
    )
    for key, original in original_offsets.items():
        observed = offsets[key]
        expected = (
            0.0
            if key in spec.KNOCKOUT_ZERO_KEYS[mode]
            else float(original["value"])
        )
        _require(observed == expected, f"{location}.{key} intervention differs")
    effective_state_sha = _sha256_value(
        evaluation.get("evaluated_state_dict_sha256"),
        f"{location}.evaluated_state_dict_sha256",
    )
    points_value = evaluation.get("points")
    _require(
        isinstance(points_value, list) and bool(points_value),
        f"{location}.points is empty",
    )
    points = [
        _normalize_raw_point(
            point,
            location=f"{location}.points[{index}]",
        )
        for index, point in enumerate(points_value)
    ]
    thresholds = [float(point["threshold"]) for point in points]
    _require(
        thresholds == sorted(thresholds)
        and len(thresholds) == len(set(thresholds)),
        f"{location} thresholds are not sorted and unique",
    )
    _require(
        0.0 not in thresholds,
        f"{location} inserted a non-formal threshold 0",
    )
    _require(
        LAST_FLOAT32_BELOW_ONE in thresholds and 1.0 in thresholds,
        f"{location} closed upper endpoints are incomplete",
    )
    fixed_matches = [
        point for point in points if float(point["threshold"]) == 0.5
    ]
    _require(
        len(fixed_matches) == 1,
        f"{location} must contain exactly one threshold 0.5 point",
    )
    declared_fixed = evaluation.get("fixed_threshold_0_5")
    _require(
        declared_fixed == fixed_matches[0],
        f"{location} fixed point differs from raw points",
    )
    fixed = _normalize_fixed(
        declared_fixed,
        location=f"{location}.fixed_threshold_0_5",
    )
    budgets = _normalize_budgets(
        evaluation.get("best_points_under_fa_budget"),
        points=points,
        location=f"{location}.best_points_under_fa_budget",
    )
    if "final_metric_coverage" in evaluation:
        _validate_metric_coverage(
            evaluation["final_metric_coverage"],
            fixed=fixed,
            budgets=budgets,
            location=f"{location}.final_metric_coverage",
        )
    provenance = _mapping(
        evaluation.get("threshold_provenance"),
        f"{location}.threshold_provenance",
    )
    _require(
        provenance.get("total_unique_threshold_count") == len(points),
        f"{location} threshold provenance count differs",
    )
    audit = _mapping(evaluation.get("audit"), f"{location}.audit")
    required_audit_true = (
        "source_state_strict_load",
        "only_requested_dc_state_keys_changed",
        "requested_dc_state_keys_zero",
        "non_zeroed_dc_offsets_preserved",
        "source_state_unchanged",
        "transformed_state_stable_during_inference",
        "non_dc_state_unchanged",
        "closed_interval_validated",
    )
    for field in required_audit_true:
        _require(audit.get(field) is True, f"{location}.audit.{field} failed")
    _require(
        audit.get("derived_checkpoint_written") is False,
        f"{location} wrote a derived checkpoint",
    )
    _require(
        evaluation.get("source_checkpoint_sha256_before")
        == source["checkpoint_sha256"]
        == evaluation.get("source_checkpoint_sha256_after"),
        f"{location} source checkpoint changed",
    )
    _require(
        evaluation.get("source_state_dict_sha256")
        == source["state_dict_sha256"],
        f"{location} source state SHA differs",
    )
    _require(
        evaluation.get("non_dc_state_sha256_before")
        == evaluation.get("non_dc_state_sha256_after"),
        f"{location} non-DC state changed",
    )
    _require(
        not FORBIDDEN_DECISION_FIELDS.intersection(evaluation),
        f"{location} contains a formal decision field",
    )
    return {
        "checkpoint": checkpoint,
        "checkpoint_role": spec.CHECKPOINT_ROLES[checkpoint],
        "knockout_mode": mode,
        "zeroed_state_keys": list(zeroed),
        "effective_changed_state_keys": list(changed),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_state_dict_sha256": source["state_dict_sha256"],
        "original_dc_offsets": copy.deepcopy(dict(original_offsets)),
        "evaluated_dc_offsets": copy.deepcopy(offset_records),
        "evaluated_state_dict_sha256": effective_state_sha,
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "raw_point_count": len(points),
        "threshold_provenance_sha256": spec.canonical_sha256(provenance),
        "diagnostic_source_binding": copy.deepcopy(
            dict(expected_source_binding)
        ),
    }


def validate_checkpoint_sweep(
    path: Path,
    *,
    checkpoint: str,
    expected_source_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate one checkpoint file and return its four normalized rows."""

    _require(checkpoint in spec.CHECKPOINTS, "checkpoint is outside the matrix")
    payload = _load_json(path, f"knockout sweep {checkpoint}")
    payload = _deep_validate_evaluator_output(
        payload,
        checkpoint=checkpoint,
        expected_source_binding=expected_source_binding,
    )
    _require(
        payload.get("schema") == spec.EVALUATION_SCHEMA
        and payload.get("status") == "complete"
        and payload.get("artifact_kind") == spec.ARTIFACT_KIND
        and payload.get("scope")
        == "evaluation_only_same_checkpoint_counterfactual"
        and payload.get("diagnostic_only") is True
        and payload.get("affects_formal_gate") is False
        and payload.get("formal_decision_authority") is False
        and payload.get("formal_gate_eligible") is False,
        f"knockout sweep identity differs: {checkpoint}",
    )
    for field, expected in {
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "expected_epochs": spec.EXPECTED_EPOCHS,
        "validation_count": spec.VALIDATION_COUNT,
        "checkpoint_filename": checkpoint,
        "checkpoint_role": spec.CHECKPOINT_ROLES[checkpoint],
        "official_test_accessed": False,
    }.items():
        _require(
            payload.get(field) == expected,
            f"knockout sweep field differs: {checkpoint}.{field}",
        )
    _integer(
        payload.get("checkpoint_epoch"),
        f"{checkpoint}.checkpoint_epoch",
        minimum=1,
        maximum=spec.EXPECTED_EPOCHS,
    )
    _require(
        not FORBIDDEN_DECISION_FIELDS.intersection(payload),
        f"knockout sweep contains a formal decision: {checkpoint}",
    )
    _require(
        payload.get("knockout_spec_sha256")
        == spec.specification_sha256(),
        f"knockout sweep spec SHA differs: {checkpoint}",
    )
    _require(
        payload.get("diagnostic_source_lock_sha256")
        == expected_source_binding["diagnostic_source_lock"]["sha256"],
        f"knockout sweep source-lock SHA differs: {checkpoint}",
    )
    _require(
        payload.get("source_binding") == dict(expected_source_binding),
        f"knockout sweep source binding differs: {checkpoint}",
    )
    _require(
        payload.get("run_directory") == str(spec.FORMAL_RUN_DIR.resolve()),
        f"{checkpoint} formal run directory differs",
    )
    run_identity = _mapping(
        payload.get("run_identity"),
        f"{checkpoint}.run_identity",
    )
    for field, expected in {
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
    }.items():
        _require(
            run_identity.get(field) == expected,
            f"{checkpoint}.run_identity.{field} differs",
        )
    source_checkpoint = dict(
        _mapping(
            payload.get("source_checkpoint"),
            f"{checkpoint}.source_checkpoint",
        )
    )
    for field, expected in {
        "filename": checkpoint,
        "role": spec.CHECKPOINT_ROLES[checkpoint],
        "epoch": payload["checkpoint_epoch"],
    }.items():
        _require(
            source_checkpoint.get(field) == expected,
            f"{checkpoint}.source_checkpoint.{field} differs",
        )
    checkpoint_sha = _sha256_value(
        source_checkpoint.get("sha256"),
        f"{checkpoint}.source_checkpoint.sha256",
    )
    state_dict_sha = _sha256_value(
        source_checkpoint.get("state_dict_sha256"),
        f"{checkpoint}.source_checkpoint.state_dict_sha256",
    )
    _require(
        payload.get("source_state_dict_sha256") == state_dict_sha,
        f"{checkpoint} source state SHA differs",
    )
    _sha256_value(
        payload.get("validation_split_sha256"),
        f"{checkpoint}.validation_split_sha256",
    )
    _mapping(
        source_checkpoint.get("checkpoint_identity"),
        f"{checkpoint}.checkpoint_identity",
    )
    _require(
        payload.get("knockout_modes") == list(spec.KNOCKOUT_MODES)
        and payload.get("knockout_specification")
        == spec.fixed_specification()
        and payload.get("threshold_contract") == spec.threshold_contract(),
        f"{checkpoint} knockout specification differs",
    )
    original_offset_records, _ = _normalize_offset_records(
        payload.get("original_dc_offsets"),
        location=f"{checkpoint}.original_dc_offsets",
    )
    source = {
        "checkpoint_sha256": checkpoint_sha,
        "state_dict_sha256": state_dict_sha,
    }
    evaluations = payload.get("evaluations")
    _require(
        isinstance(evaluations, list)
        and len(evaluations) == len(spec.KNOCKOUT_MODES),
        f"{checkpoint} must contain four evaluations",
    )
    _require(
        [evaluation.get("knockout_mode") for evaluation in evaluations]
        == list(spec.KNOCKOUT_MODES),
        f"{checkpoint} evaluation order differs",
    )
    artifact_hashes = _mapping(
        payload.get("artifact_sha256"),
        f"{checkpoint}.artifact_sha256",
    )
    _require(
        bool(artifact_hashes)
        and all(
            isinstance(name, str)
            and isinstance(value, str)
            and len(value) == 64
            for name, value in artifact_hashes.items()
        ),
        f"{checkpoint} artifact SHA registry is invalid",
    )
    top_audit = _mapping(payload.get("audit"), f"{checkpoint}.audit")
    for field in (
        "formal_artifacts_read_only",
        "formal_artifacts_unchanged",
        "all_modes_from_pristine_source_state",
        "modes_evaluated_sequentially",
    ):
        _require(
            top_audit.get(field) is True,
            f"{checkpoint}.audit.{field} failed",
        )
    _require(
        top_audit.get("derived_checkpoint_written") is False,
        f"{checkpoint} wrote a derived checkpoint",
    )
    _require(
        top_audit.get("source_checkpoint_sha256_before")
        == checkpoint_sha
        == top_audit.get("source_checkpoint_sha256_after"),
        f"{checkpoint} source checkpoint changed",
    )
    return [
        _validate_evaluation(
            evaluation,
            checkpoint=checkpoint,
            source=source,
            original_offsets=original_offset_records,
            expected_source_binding=expected_source_binding,
            location=f"{checkpoint}.evaluations[{index}]",
        )
        for index, evaluation in enumerate(evaluations)
    ]


def load_formal_reference_rows(
    path: Path = DEFAULT_FORMAL_REPORT,
) -> dict[str, dict[str, Any]]:
    report = _load_json(path, "repaired formal V3 aggregate report")
    _require(
        report.get("status") == "complete"
        and report.get("row_count") == 8
        and report.get("dataset") == spec.DATASET
        and report.get("training_seed") == spec.TRAINING_SEED
        and report.get("split_seed") == spec.SPLIT_SEED,
        "repaired formal V3 aggregate identity differs",
    )
    repair = _mapping(
        _mapping(
            report.get("comparison_contract"),
            "repaired formal comparison contract",
        ).get("selection_contract_repair"),
        "repaired formal selection contract",
    )
    _require(
        report.get("decision") == freezer.EXPECTED_FORMAL_DECISION
        and report.get("aggregate_full_model_gate_passed") is False
        and repair.get("repair_id") == freezer.FORMAL_REPAIR_ID
        and repair.get("each_variant_uses_own_selected_checkpoints")
        is True,
        "repaired formal aggregate authority differs",
    )
    rows = report.get("rows")
    _require(isinstance(rows, list), "formal V3 aggregate rows are missing")
    references: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            isinstance(row, Mapping)
            and row.get("variant") == spec.VARIANT
            and row.get("checkpoint_role") in spec.CHECKPOINT_ROLES.values()
        ):
            role = str(row["checkpoint_role"])
            _require(role not in references, f"duplicate formal V3 role: {role}")
            fixed = _normalize_fixed(
                row.get("fixed_threshold_0_5"),
                location=f"formal reference {role}.fixed",
            )
            raw_budgets = _mapping(
                row.get("pd_at_fa_budget"),
                f"formal reference {role}.budgets",
            )
            _require(
                set(raw_budgets) == set(spec.BUDGET_KEYS),
                f"formal reference {role} budget keys differ",
            )
            budgets: dict[str, Any] = {}
            for budget, key in zip(spec.FA_BUDGETS, spec.BUDGET_KEYS):
                point = _mapping(
                    raw_budgets[key],
                    f"formal reference {role}.{key}",
                )
                budgets[key] = {
                    "budget": budget,
                    "pd": _finite(point.get("pd"), f"formal {role}.{key}.pd"),
                    "achieved_fa": _finite(
                        point.get("achieved_fa"),
                        f"formal {role}.{key}.achieved_fa",
                    ),
                    "threshold": _finite(
                        point.get("threshold"),
                        f"formal {role}.{key}.threshold",
                    ),
                    "matched_target_count": _integer(
                        point.get("matched_target_count"),
                        f"formal {role}.{key}.matched",
                        minimum=0,
                        maximum=spec.TARGET_COUNT,
                    ),
                    "target_count": _integer(
                        point.get("target_count"),
                        f"formal {role}.{key}.target_count",
                        minimum=spec.TARGET_COUNT,
                        maximum=spec.TARGET_COUNT,
                    ),
                }
            references[role] = {
                "fixed_threshold_0_5": fixed,
                "pd_at_fa_budget": budgets,
                "formal_row_sha256": spec.canonical_sha256(dict(row)),
            }
    _require(
        set(references) == set(spec.CHECKPOINT_ROLES.values()),
        "formal V3 learned reference roles differ",
    )
    return references


def _validate_repaired_formal_report_input(
    path: Path,
    *,
    formal_binding: Mapping[str, Any],
) -> Path:
    """Require the one repaired aggregate frozen into the source lock."""

    value = _regular_file(path, "repaired formal V3 aggregate report")
    locked_report = _mapping(
        formal_binding.get("formal_aggregate_json"),
        "source-lock repaired formal aggregate binding",
    )
    repair_binding = _mapping(
        formal_binding.get("formal_selection_contract_repair"),
        "source-lock formal selection-contract repair binding",
    )
    _require(
        value.resolve() == DEFAULT_FORMAL_REPORT.resolve()
        and locked_report.get("path") == str(value.resolve()),
        "only the canonical repaired formal aggregate may be referenced",
    )
    _require(
        locked_report.get("sha256") == sha256_file(value),
        "repaired formal aggregate SHA differs from the source lock",
    )
    _require(
        repair_binding.get("authority")
        == "versioned_selection_contract_repair_v1_only"
        and repair_binding.get(
            "each_variant_uses_own_selected_checkpoints"
        )
        is True
        and repair_binding.get("formal_aggregate_decision")
        == freezer.EXPECTED_FORMAL_DECISION
        and repair_binding.get("aggregate_full_model_gate_passed")
        is False,
        "source-lock selection-contract repair authority differs",
    )
    return value


def _signed_deltas(
    row: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = row["fixed_threshold_0_5"]
    fixed_reference = reference["fixed_threshold_0_5"]
    fixed_fields = (
        "matched_target_count",
        "matched_tiny_target_count",
        "pd",
        "fa",
        "miou",
        "false_objects_per_image",
        "tiny_pd",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    )
    fixed_delta = {
        name: fixed[name] - fixed_reference[name]
        for name in fixed_fields
    }
    budget_deltas: dict[str, Any] = {}
    for key in spec.BUDGET_KEYS:
        point = row["pd_at_fa_budget"][key]
        learned = reference["pd_at_fa_budget"][key]
        budget_deltas[key] = {
            "matched_target_count": (
                point["matched_target_count"]
                - learned["matched_target_count"]
            ),
            "pd": point["pd"] - learned["pd"],
            "achieved_fa": point["achieved_fa"] - learned["achieved_fa"],
            "threshold": point["threshold"] - learned["threshold"],
        }
    return {
        "direction": "knockout_minus_same_role_learned_v3",
        "fixed_threshold_0_5": fixed_delta,
        "pd_at_fa_budget": budget_deltas,
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    references: Mapping[str, Mapping[str, Any]],
    source_lock_payload: Mapping[str, Any],
    source_lock_path: Path,
    sweep_bindings: Mapping[str, Mapping[str, Any]],
    formal_report_path: Path,
) -> dict[str, Any]:
    normalized_rows = [copy.deepcopy(dict(row)) for row in rows]
    _require(
        len(normalized_rows) == spec.EXPECTED_ROW_COUNT,
        "DC knockout aggregate does not contain eight rows",
    )
    expected_ids = [
        (entry["checkpoint"], entry["knockout_mode"])
        for entry in spec.matrix_rows()
    ]
    actual_ids = [
        (row.get("checkpoint"), row.get("knockout_mode"))
        for row in normalized_rows
    ]
    _require(actual_ids == expected_ids, "DC knockout aggregate matrix differs")
    for index, row in enumerate(normalized_rows, start=1):
        _require(
            not FORBIDDEN_DECISION_FIELDS.intersection(row),
            "diagnostic row contains a formal decision field",
        )
        role = str(row["checkpoint_role"])
        reference = references[role]
        row["row_index"] = index
        row["row_id"] = f"{row['checkpoint']}:{row['knockout_mode']}"
        row["formal_learned_reference"] = {
            "checkpoint_role": role,
            "formal_row_sha256": reference["formal_row_sha256"],
        }
        row["signed_delta_knockout_minus_learned"] = _signed_deltas(
            row,
            reference,
        )
    formal_binding = source_lock_payload["formal_artifact_binding"]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "artifact_kind": spec.ARTIFACT_KIND,
        "scope": "evaluation_only_same_checkpoint_counterfactual",
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_components": [],
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "multi_seed_scheduled": False,
        "official_test_accessed": False,
        "row_count": len(normalized_rows),
        "matrix": spec.matrix_rows(),
        "matrix_identity_sha256": spec.canonical_sha256(spec.matrix_rows()),
        "knockout_spec_sha256": spec.specification_sha256(),
        "rows": normalized_rows,
        "formal_learned_rows_counted": False,
        "zero_all_dc_is_v2_training_trajectory": False,
        "source_binding": {
            "diagnostic_source_lock": {
                "path": str(Path(source_lock_path).resolve()),
                "sha256": sha256_file(source_lock_path),
            },
            "formal_artifact_snapshot_sha256": formal_binding[
                "snapshot_sha256"
            ],
            "formal_completion_marker_sha256": formal_binding[
                "formal_completion_marker"
            ]["sha256"],
            "formal_aggregate_json": {
                "path": str(Path(formal_report_path).resolve()),
                "sha256": sha256_file(formal_report_path),
            },
            "formal_selection_contract_repair": copy.deepcopy(
                formal_binding["formal_selection_contract_repair"]
            ),
            "sweeps": copy.deepcopy(dict(sweep_bindings)),
        },
        "formal_artifacts_unchanged": True,
        "claim_boundary": {
            "diagnostic_completion_is_not_model_success": True,
            "formal_six_component_gate_unchanged": True,
            "single_seed_only": True,
            "cross_seed_stability_claim": False,
            "cross_dataset_claim": False,
            "official_test_claim": False,
            "causal_training_trajectory_claim": False,
        },
    }
    _require(
        not FORBIDDEN_DECISION_FIELDS.intersection(report),
        "diagnostic aggregate contains a formal decision field",
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V3 DC-offset knockout diagnostic",
        "",
        "- Scope: diagnostic-only same-checkpoint counterfactual",
        "- Formal six-component gate affected: `false`",
        "- Completion denotes model success: `false`",
        "- Seed: `42`; split seed: `20260722`; official test accessed: `false`",
        "",
        "| Checkpoint role | Knockout | Pd@0.5 | Fa@0.5 | mIoU@0.5 | Tiny-Pd@0.5 | Δmatched@0.5 | ΔFa@0.5 | Pd@1e-6 | Pd@5e-6 | Pd@1e-5 | Pd@5e-5 | Pd@1e-4 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        fixed = row["fixed_threshold_0_5"]
        delta = row["signed_delta_knockout_minus_learned"][
            "fixed_threshold_0_5"
        ]
        budgets = [
            row["pd_at_fa_budget"][key]["pd"]
            for key in spec.BUDGET_KEYS
        ]
        lines.append(
            f"| {row['checkpoint_role']} | {row['knockout_mode']} | "
            f"{fixed['pd']:.9f} | {fixed['fa']:.9g} | "
            f"{fixed['miou']:.9f} | {fixed['tiny_pd']:.9f} | "
            f"{delta['matched_target_count']:+d} | {delta['fa']:+.9g} | "
            + " | ".join(f"{value:.9f}" for value in budgets)
            + " |"
        )
    lines.extend(
        [
            "",
            "All deltas are `knockout - same-role learned V3`. "
            "No row is eligible to alter the formal decision.",
            "",
        ]
    )
    return "\n".join(lines)


def _report_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(report),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _marker_payload(
    *,
    report: Mapping[str, Any],
    json_bytes: bytes,
    markdown_bytes: bytes,
    json_path: Path,
    markdown_path: Path,
    source_lock_path: Path,
    sweep_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": COMPLETE_MARKER_SCHEMA,
        "status": "complete",
        "artifact_kind": spec.ARTIFACT_KIND,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "row_count": spec.EXPECTED_ROW_COUNT,
        "matrix_identity_sha256": report["matrix_identity_sha256"],
        "knockout_spec_sha256": spec.specification_sha256(),
        "diagnostic_source_lock_sha256": sha256_file(source_lock_path),
        "formal_completion_marker_sha256": report["source_binding"][
            "formal_completion_marker_sha256"
        ],
        "formal_selection_contract_repair_sha256": spec.canonical_sha256(
            report["source_binding"]["formal_selection_contract_repair"]
        ),
        "formal_artifacts_unchanged": True,
        "official_test_accessed": False,
        "sweep_sha256": {
            name: binding["sha256"]
            for name, binding in sorted(sweep_bindings.items())
        },
        "outputs": {
            json_path.name: hashlib.sha256(json_bytes).hexdigest(),
            markdown_path.name: hashlib.sha256(markdown_bytes).hexdigest(),
        },
    }


def _assert_owned_path(path: Path, output_root: Path) -> None:
    root = spec.validated_output_root(output_root)
    value = Path(path).resolve()
    _require(
        value.is_relative_to(root),
        f"diagnostic output escapes its root: {value}",
    )


def _atomic_publish_new(path: Path, content: bytes) -> None:
    _require(
        not path.exists() and not path.is_symlink(),
        f"refusing to overwrite diagnostic artifact: {path}",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(path) from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(
    report: Mapping[str, Any],
    *,
    output_root: Path,
    source_lock_path: Path,
    sweep_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    json_path, markdown_path, marker_path = spec.aggregate_paths(output_root)
    for path in (json_path, markdown_path, marker_path):
        _assert_owned_path(path, output_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        json_path.parent.is_dir() and not json_path.parent.is_symlink(),
        "diagnostic comparison directory is unsafe",
    )
    json_bytes = _report_bytes(report)
    markdown_bytes = render_markdown(report).encode("utf-8")
    marker = _marker_payload(
        report=report,
        json_bytes=json_bytes,
        markdown_bytes=markdown_bytes,
        json_path=json_path,
        markdown_path=markdown_path,
        source_lock_path=source_lock_path,
        sweep_bindings=sweep_bindings,
    )
    marker_bytes = _report_bytes(marker)
    expected = (
        (json_path, json_bytes),
        (markdown_path, markdown_bytes),
        (marker_path, marker_bytes),
    )
    for path, content in expected:
        if path.exists() or path.is_symlink():
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == content,
                f"existing diagnostic artifact conflicts: {path}",
            )
        else:
            _atomic_publish_new(path, content)
    return json_path, markdown_path, marker_path


def inspect_complete(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_lock_path: Path = DEFAULT_SOURCE_LOCK,
) -> dict[str, Any] | None:
    json_path, markdown_path, marker_path = spec.aggregate_paths(output_root)
    if not marker_path.exists():
        _require(
            not marker_path.is_symlink(),
            "diagnostic completion marker may not be a symlink",
        )
        return None
    marker = _load_json(marker_path, "DC knockout completion marker")
    _require(
        marker.get("schema") == COMPLETE_MARKER_SCHEMA
        and marker.get("status") == "complete"
        and marker.get("artifact_kind") == spec.ARTIFACT_KIND
        and marker.get("diagnostic_only") is True
        and marker.get("affects_formal_gate") is False
        and marker.get("formal_decision_authority") is False
        and marker.get("row_count") == spec.EXPECTED_ROW_COUNT
        and marker.get("formal_artifacts_unchanged") is True
        and marker.get("official_test_accessed") is False,
        "DC knockout completion marker identity differs",
    )
    _require(
        not FORBIDDEN_DECISION_FIELDS.intersection(marker),
        "DC knockout completion marker contains a formal decision",
    )
    _require(
        marker.get("matrix_identity_sha256")
        == spec.canonical_sha256(spec.matrix_rows())
        and marker.get("knockout_spec_sha256")
        == spec.specification_sha256(),
        "DC knockout completion marker fixed-contract binding differs",
    )
    outputs = _mapping(marker.get("outputs"), "diagnostic marker outputs")
    expected = {
        json_path.name: sha256_file(json_path),
        markdown_path.name: sha256_file(markdown_path),
    }
    _require(dict(outputs) == expected, "diagnostic marker output hashes differ")
    report = _load_json(json_path, "DC knockout aggregate JSON")
    _require(
        report.get("schema") == SCHEMA
        and report.get("status") == "complete"
        and report.get("row_count") == spec.EXPECTED_ROW_COUNT
        and report.get("matrix_identity_sha256")
        == marker.get("matrix_identity_sha256")
        and report.get("knockout_spec_sha256")
        == marker.get("knockout_spec_sha256"),
        "diagnostic aggregate/marker binding differs",
    )
    _require(
        report.get("diagnostic_only") is True
        and report.get("affects_formal_gate") is False
        and report.get("formal_decision_authority") is False
        and report.get("formal_artifacts_unchanged") is True
        and report.get("official_test_accessed") is False
        and not FORBIDDEN_DECISION_FIELDS.intersection(report),
        "diagnostic aggregate authority boundary differs",
    )
    source_lock_payload = freezer.verify_source_lock(source_lock_path)
    source_lock_sha256 = sha256_file(source_lock_path)
    _require(
        marker.get("diagnostic_source_lock_sha256")
        == source_lock_sha256,
        "diagnostic marker source-lock SHA differs",
    )
    report_binding = _mapping(
        report.get("source_binding"),
        "diagnostic aggregate source binding",
    )
    _require(
        report_binding.get("diagnostic_source_lock")
        == {
            "path": str(Path(source_lock_path).resolve()),
            "sha256": source_lock_sha256,
        },
        "diagnostic aggregate source-lock binding differs",
    )
    formal_binding = _mapping(
        source_lock_payload.get("formal_artifact_binding"),
        "diagnostic source lock formal binding",
    )
    formal_marker = _mapping(
        formal_binding.get("formal_completion_marker"),
        "formal completion marker binding",
    )
    _require(
        marker.get("formal_completion_marker_sha256")
        == formal_marker.get("sha256")
        == report_binding.get("formal_completion_marker_sha256"),
        "diagnostic package formal marker binding differs",
    )
    _require(
        report_binding.get("formal_artifact_snapshot_sha256")
        == formal_binding.get("snapshot_sha256"),
        "diagnostic package formal snapshot binding differs",
    )
    report_repair = _mapping(
        report_binding.get("formal_selection_contract_repair"),
        "diagnostic aggregate formal repair binding",
    )
    locked_repair = _mapping(
        formal_binding.get("formal_selection_contract_repair"),
        "diagnostic source lock formal repair binding",
    )
    _require(
        report_repair == locked_repair
        and marker.get("formal_selection_contract_repair_sha256")
        == spec.canonical_sha256(locked_repair),
        "diagnostic package selection-contract repair binding differs",
    )
    formal_report = _mapping(
        report_binding.get("formal_aggregate_json"),
        "diagnostic aggregate formal report binding",
    )
    locked_formal_report = _mapping(
        formal_binding.get("formal_aggregate_json"),
        "diagnostic source lock formal report binding",
    )
    formal_report_path = Path(str(formal_report.get("path")))
    _require(
        formal_report
        == {
            "path": str(formal_report_path.resolve()),
            "sha256": sha256_file(formal_report_path),
        },
        "diagnostic aggregate formal report hash differs",
    )
    _require(
        formal_report["path"] == locked_formal_report.get("path")
        and formal_report["sha256"] == locked_formal_report.get("sha256"),
        "diagnostic aggregate formal report lock binding differs",
    )
    sweep_hashes = _mapping(
        marker.get("sweep_sha256"),
        "diagnostic marker sweep hashes",
    )
    report_sweeps = _mapping(
        report_binding.get("sweeps"),
        "diagnostic aggregate sweep bindings",
    )
    _require(
        set(sweep_hashes) == set(spec.CHECKPOINTS)
        and set(report_sweeps) == set(spec.CHECKPOINTS),
        "diagnostic package sweep set differs",
    )
    for checkpoint in spec.CHECKPOINTS:
        sweep_path = spec.sweep_path(checkpoint, output_root)
        expected_sweep_binding = {
            "path": str(sweep_path.resolve()),
            "sha256": sha256_file(sweep_path),
        }
        _require(
            report_sweeps.get(checkpoint) == expected_sweep_binding
            and sweep_hashes.get(checkpoint)
            == expected_sweep_binding["sha256"],
            f"diagnostic package sweep binding differs: {checkpoint}",
        )
    return marker


def aggregate_and_write(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_lock_path: Path = DEFAULT_SOURCE_LOCK,
    formal_report_path: Path = DEFAULT_FORMAL_REPORT,
) -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    """Validate all inputs twice and atomically publish the diagnostic package."""

    before = freezer.verify_source_lock(source_lock_path)
    expected_source_binding = freezer.current_source_binding(source_lock_path)
    formal_binding = _mapping(
        before.get("formal_artifact_binding"),
        "diagnostic source lock formal binding",
    )
    repaired_report = _validate_repaired_formal_report_input(
        formal_report_path,
        formal_binding=formal_binding,
    )
    rows: list[dict[str, Any]] = []
    sweep_bindings: dict[str, dict[str, Any]] = {}
    for checkpoint in spec.CHECKPOINTS:
        path = spec.sweep_path(checkpoint, output_root)
        rows.extend(
            validate_checkpoint_sweep(
                path,
                checkpoint=checkpoint,
                expected_source_binding=expected_source_binding,
            )
        )
        sweep_bindings[checkpoint] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    references = load_formal_reference_rows(repaired_report)
    report = build_report(
        rows,
        references=references,
        source_lock_payload=before,
        source_lock_path=source_lock_path,
        sweep_bindings=sweep_bindings,
        formal_report_path=repaired_report,
    )
    after_validation = freezer.verify_source_lock(source_lock_path)
    _require(
        before == after_validation,
        "formal or diagnostic source binding changed during aggregation",
    )
    paths = write_report(
        report,
        output_root=output_root,
        source_lock_path=source_lock_path,
        sweep_bindings=sweep_bindings,
    )
    after_publish = freezer.verify_source_lock(source_lock_path)
    _require(
        before == after_publish,
        "formal inputs changed while publishing diagnostic outputs",
    )
    inspect_complete(
        output_root=output_root,
        source_lock_path=source_lock_path,
    )
    return report, paths


def execution_plan(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    paths = {
        checkpoint: spec.sweep_path(checkpoint, output_root)
        for checkpoint in spec.CHECKPOINTS
    }
    return {
        "artifact_kind": spec.ARTIFACT_KIND,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "evaluator": str(
            (
                REPO_ROOT
                / "experiments/"
                "evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout.py"
            ).resolve()
        ),
        "checkpoint_input_count": len(paths),
        "knockout_row_count": spec.EXPECTED_ROW_COUNT,
        "inputs": {
            checkpoint: {
                "path": str(path.resolve()),
                "exists": path.is_file() and not path.is_symlink(),
            }
            for checkpoint, path in paths.items()
        },
        "aggregate_outputs": [
            str(path.resolve())
            for path in spec.aggregate_paths(output_root)
        ],
        "invokes_evaluator": False,
        "invokes_gpu": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the diagnostic-only V3 DC knockout matrix"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        marker = inspect_complete()
        result = {
            "artifact_kind": spec.ARTIFACT_KIND,
            "diagnostic_only": True,
            "affects_formal_gate": False,
            "status": "complete" if marker is not None else "incomplete",
            "marker": marker,
        }
    elif args.plan:
        freezer.verify_source_lock(DEFAULT_SOURCE_LOCK)
        result = execution_plan()
    else:
        report, paths = aggregate_and_write()
        result = {
            "artifact_kind": spec.ARTIFACT_KIND,
            "diagnostic_only": True,
            "affects_formal_gate": False,
            "status": "complete",
            "row_count": report["row_count"],
            "outputs": [str(path) for path in paths],
        }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLETE_MARKER_SCHEMA",
    "DEFAULT_FORMAL_REPORT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SOURCE_LOCK",
    "FORBIDDEN_DECISION_FIELDS",
    "IncompleteDiagnostic",
    "SCHEMA",
    "aggregate_and_write",
    "build_report",
    "execution_plan",
    "inspect_complete",
    "load_formal_reference_rows",
    "render_markdown",
    "sha256_file",
    "validate_checkpoint_sweep",
    "write_report",
]
