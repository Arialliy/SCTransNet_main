#!/usr/bin/env python3
"""Finalize the three-dataset EC-TSS V3.1 experiment from real artifacts.

The frozen V3 gates continue to use the five predeclared metrics.  An additive
joint-quality audit also reports pixel precision/F1 and false-object counts so
that Pd, component Fa, and region quality are never interpreted in isolation.
No checkpoint is reselected and no training/evaluation artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

POSITIVE_ROOT = RESULTS_ROOT / "three_dataset_seed42_global_tss_v2"
OFF_ROOT = RESULTS_ROOT / "three_dataset_tss_off_seed42_v1"
EC_ROOT = RESULTS_ROOT / "three_dataset_ec_tss_v3_1_seed42"
DEFAULT_OUTPUT_DIR = EC_ROOT / "comparison"

DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
ROLES = ("best_miou", "best_pd")
RECIPES = (
    "original",
    "tss_off",
    "lambda_0p0025",
    "lambda_0p005",
    "lambda_0p01",
    "ec_tss_v3_1",
)
REFERENCE_RECIPES = ("original", "tss_off", "lambda_0p005")

EVALUATION_SCHEMA = "sctransnet_three_dataset_v2_evaluation_v1"
OUTPUT_SCHEMA = "sctransnet_ec_tss_v3_1_final_comparison/v1"
FIXED_THRESHOLD = 0.5
SEED = 42
IOU_QUANTUM = 0.0001
SERIOUS_IOU_DROP_QUANTA = 50

# The five metrics below are exactly the frozen V3-B--V3-E vector.
FROZEN_METRICS = (
    "miou",
    "niou",
    "matched_target_count",
    "unmatched_predicted_pixels",
    "matched_tiny_target_count",
)
JOINT_METRICS = FROZEN_METRICS + (
    "pixel_precision",
    "pixel_f1",
    "unmatched_predicted_object_count",
)
HIGHER_IS_BETTER = {
    "miou": True,
    "niou": True,
    "matched_target_count": True,
    "unmatched_predicted_pixels": False,
    "matched_tiny_target_count": True,
    "pixel_precision": True,
    "pixel_f1": True,
    "unmatched_predicted_object_count": False,
}
QUANTIZED_METRICS = {"miou", "niou", "pixel_precision", "pixel_f1"}

PASS_DECISION = "EC_TSS_V3_1_SEED42_TEST_SELECTED_PASS"
PENDING_DECISION = "EC_TSS_V3_1_PERFORMANCE_PASS_DIAGNOSTICS_PENDING"
FAIL_DECISION = "EC_TSS_V3_1_PERFORMANCE_FAIL_STOP_TSS_OPTIMIZATION"
JOINT_MIXED_VERDICT = "MIXED_TRADEOFF_WITH_JOINT_PARETO_VALUE"


class ComparisonError(ValueError):
    """Raised when a supplied artifact violates the comparison contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in output, f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def load_json(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    _require(candidate.is_file() and not candidate.is_symlink(), f"missing file: {candidate}")
    payload = json.loads(
        candidate.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ComparisonError(f"non-finite JSON token: {token}")
        ),
    )
    _require(isinstance(payload, dict), f"JSON root is not an object: {candidate}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ComparisonError(f"{label} must be numeric") from error
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _unit(value: Any, label: str) -> float:
    number = _finite(value, label)
    _require(0.0 <= number <= 1.0, f"{label} must be in [0, 1]")
    return number


def _nonnegative(value: Any, label: str) -> float:
    number = _finite(value, label)
    _require(number >= 0.0, f"{label} must be non-negative")
    return number


def _count(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be int")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def quantize(value: float) -> int:
    """Apply the frozen q(x)=floor(x/0.0001+0.5) rule deterministically."""

    finite = _finite(value, "quantized metric")
    scaled = Decimal(str(finite)) / Decimal("0.0001") + Decimal("0.5")
    return int(scaled.to_integral_value(rounding=ROUND_FLOOR))


def recipe_evaluation_path(
    recipe: str,
    dataset: str,
    role: str,
    *,
    positive_root: Path = POSITIVE_ROOT,
    off_root: Path = OFF_ROOT,
    ec_root: Path = EC_ROOT,
) -> Path:
    if recipe == "original":
        return positive_root / "runs" / dataset / "original" / "seed_42" / "evaluations" / f"{role}.json"
    if recipe.startswith("lambda_"):
        return positive_root / "runs" / dataset / "final" / recipe / "seed_42" / "evaluations" / f"{role}.json"
    if recipe == "tss_off":
        return off_root / "runs" / dataset / "final_tss_off" / "seed_42" / "evaluations" / f"{role}.json"
    if recipe == "ec_tss_v3_1":
        return ec_root / "runs" / dataset / "final_ec_tss_v3_1" / "seed_42" / "evaluations" / f"{role}.json"
    raise ComparisonError(f"unknown recipe: {recipe}")


def _expected_method(recipe: str) -> str:
    if recipe == "original":
        return "original"
    if recipe == "tss_off":
        return "final_tss_off"
    if recipe == "ec_tss_v3_1":
        return "final_ec_tss_v3_1"
    return "final"


def normalize_point(point: Mapping[str, Any], label: str) -> dict[str, Any]:
    required_rates = (
        "miou",
        "niou",
        "pd",
        "tiny_pd",
        "fa",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    )
    required_counts = (
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "unmatched_predicted_pixels",
        "valid_pixel_count",
    )
    normalized: dict[str, Any] = {
        key: _unit(point.get(key), f"{label}.{key}") for key in required_rates
    }
    normalized["false_objects_per_image"] = _nonnegative(
        point.get("false_objects_per_image"),
        f"{label}.false_objects_per_image",
    )
    normalized.update(
        {key: _count(point.get(key), f"{label}.{key}") for key in required_counts}
    )
    normalized["threshold"] = _finite(point.get("threshold"), f"{label}.threshold")
    normalized["test_loss"] = _nonnegative(
        point.get("test_loss"), f"{label}.test_loss"
    )

    _require(normalized["target_count"] > 0, f"{label} target_count must be positive")
    _require(normalized["valid_pixel_count"] > 0, f"{label} valid_pixel_count must be positive")
    _require(
        normalized["matched_target_count"] <= normalized["target_count"],
        f"{label} matched targets exceed target count",
    )
    _require(
        normalized["matched_tiny_target_count"] <= normalized["tiny_target_count"],
        f"{label} matched tiny targets exceed tiny target count",
    )
    _require(
        normalized["predicted_object_count"]
        == normalized["matched_target_count"] + normalized["unmatched_predicted_object_count"],
        f"{label} predicted-object identity is inconsistent",
    )
    expected_pd = normalized["matched_target_count"] / normalized["target_count"]
    expected_tiny = (
        normalized["matched_tiny_target_count"] / normalized["tiny_target_count"]
        if normalized["tiny_target_count"]
        else 0.0
    )
    expected_fa = normalized["unmatched_predicted_pixels"] / normalized["valid_pixel_count"]
    _require(math.isclose(normalized["pd"], expected_pd, abs_tol=1e-12), f"{label} Pd/count mismatch")
    _require(math.isclose(normalized["tiny_pd"], expected_tiny, abs_tol=1e-12), f"{label} tiny-Pd/count mismatch")
    _require(math.isclose(normalized["fa"], expected_fa, abs_tol=1e-15), f"{label} Fa/count mismatch")
    _require(math.isclose(normalized["threshold"], FIXED_THRESHOLD, abs_tol=0.0), f"{label} threshold is not 0.5")

    for metric in QUANTIZED_METRICS:
        normalized[f"{metric}_q"] = quantize(normalized[metric])
    predicted = normalized["predicted_object_count"]
    normalized["object_precision"] = (
        normalized["matched_target_count"] / predicted if predicted else 0.0
    )
    pd = normalized["pd"]
    object_precision = normalized["object_precision"]
    normalized["object_f1"] = (
        2.0 * pd * object_precision / (pd + object_precision)
        if pd + object_precision
        else 0.0
    )
    return normalized


def validate_evaluation(
    payload: Mapping[str, Any], *, recipe: str, dataset: str, role: str
) -> dict[str, Any]:
    for field, expected in (
        ("schema", EVALUATION_SCHEMA),
        ("status", "complete"),
        ("dataset", dataset),
        ("checkpoint_role", role),
        ("method", _expected_method(recipe)),
        ("seed", SEED),
        ("test_selected", True),
        ("selection_is_optimistic", True),
        ("no_fabricated_results", True),
    ):
        _require(payload.get(field) == expected, f"{recipe}/{dataset}/{role} invalid {field}")
    fixed = payload.get("fixed_threshold_0_5")
    _require(isinstance(fixed, Mapping), f"{recipe}/{dataset}/{role} lacks fixed point")
    return normalize_point(fixed, f"{recipe}/{dataset}/{role}")


def load_records(
    *,
    positive_root: Path = POSITIVE_ROOT,
    off_root: Path = OFF_ROOT,
    ec_root: Path = EC_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records: dict[str, Any] = {dataset: {} for dataset in DATASETS}
    bindings: dict[str, Any] = {}
    for dataset in DATASETS:
        for role in ROLES:
            records[dataset][role] = {}
            for recipe in RECIPES:
                path = recipe_evaluation_path(
                    recipe,
                    dataset,
                    role,
                    positive_root=positive_root,
                    off_root=off_root,
                    ec_root=ec_root,
                )
                payload = load_json(path)
                records[dataset][role][recipe] = validate_evaluation(
                    payload, recipe=recipe, dataset=dataset, role=role
                )
                key = f"{dataset}/{role}/{recipe}"
                bindings[key] = {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                if recipe == "ec_tss_v3_1":
                    model = payload.get("model")
                    metric_audit = payload.get("checkpoint_metric_audit")
                    checkpoint_binding = payload.get("checkpoint_binding")
                    _require(isinstance(model, Mapping), f"{key} lacks model evidence")
                    _require(
                        isinstance(metric_audit, Mapping),
                        f"{key} lacks checkpoint metric audit",
                    )
                    _require(
                        isinstance(checkpoint_binding, Mapping),
                        f"{key} lacks checkpoint binding",
                    )
                    training_sources = checkpoint_binding.get(
                        "training_runtime_sources"
                    )
                    _require(
                        isinstance(training_sources, Mapping),
                        f"{key} lacks training source evidence",
                    )
                    bindings[key]["engineering_evidence"] = {
                        "strict_inference_state_load": model.get("strict_load") is True,
                        "training_only_tss_heads_removed": (
                            model.get("target_survival_registered") is False
                        ),
                        "checkpoint_metric_replay_passed": (
                            metric_audit.get("passed") is True
                        ),
                        "training_runtime_sources_validated": (
                            training_sources.get("validated") is True
                        ),
                        "inference_state_sha256": model.get(
                            "inference_state_sha256"
                        ),
                    }
    return records, bindings


def metric_value(point: Mapping[str, Any], metric: str) -> int:
    if metric in QUANTIZED_METRICS:
        return int(point[f"{metric}_q"])
    return int(point[metric])


def compare_value(candidate: int, reference: int, *, higher_is_better: bool) -> int:
    if candidate == reference:
        return 0
    if higher_is_better:
        return 1 if candidate > reference else -1
    return 1 if candidate < reference else -1


def relation(signs: Iterable[int]) -> str:
    values = tuple(signs)
    _require(bool(values), "comparison vector is empty")
    if all(value == 0 for value in values):
        return "equal"
    if all(value >= 0 for value in values):
        return "dominates"
    if all(value <= 0 for value in values):
        return "dominated"
    return "incomparable"


def point_dominates(
    left: Mapping[str, Any], right: Mapping[str, Any], metrics: Sequence[str]
) -> bool:
    signs = [
        compare_value(
            metric_value(left, metric),
            metric_value(right, metric),
            higher_is_better=HIGHER_IS_BETTER[metric],
        )
        for metric in metrics
    ]
    return all(sign >= 0 for sign in signs) and any(sign > 0 for sign in signs)


def pairwise_summary(
    records: Mapping[str, Any], reference_recipe: str, metrics: Sequence[str]
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    signs: list[int] = []
    for dataset in DATASETS:
        for role in ROLES:
            candidate = records[dataset][role]["ec_tss_v3_1"]
            reference = records[dataset][role][reference_recipe]
            for metric in metrics:
                candidate_value = metric_value(candidate, metric)
                reference_value = metric_value(reference, metric)
                sign = compare_value(
                    candidate_value,
                    reference_value,
                    higher_is_better=HIGHER_IS_BETTER[metric],
                )
                cells[f"{dataset}/{role}/{metric}"] = {
                    "ec_value": candidate_value,
                    "reference_value": reference_value,
                    "comparison_from_ec_perspective": sign,
                }
                signs.append(sign)
    return {
        "reference_recipe": reference_recipe,
        "metrics": list(metrics),
        "vector_dimension": len(signs),
        "relation": relation(signs),
        "better": sum(sign > 0 for sign in signs),
        "equal": sum(sign == 0 for sign in signs),
        "worse": sum(sign < 0 for sign in signs),
        "better_than_worse": sum(sign > 0 for sign in signs) > sum(sign < 0 for sign in signs),
        "cells": cells,
    }


def severe_violations(
    dataset: str,
    role: str,
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prefix = {"dataset": dataset, "checkpoint_role": role}
    violations: list[dict[str, Any]] = []
    if int(original["matched_target_count"]) - int(candidate["matched_target_count"]) >= 2:
        violations.append({**prefix, "rule": "matched_target_drop_at_least_2"})
    if int(original["matched_tiny_target_count"]) - int(candidate["matched_tiny_target_count"]) >= 2:
        violations.append({**prefix, "rule": "matched_tiny_target_drop_at_least_2"})
    for metric in ("miou", "niou"):
        drop = int(original[f"{metric}_q"]) - int(candidate[f"{metric}_q"])
        if drop >= SERIOUS_IOU_DROP_QUANTA:
            violations.append({**prefix, "rule": f"{metric}_drop_at_least_0.005", "drop_quanta": drop})
    original_fa = int(original["unmatched_predicted_pixels"])
    candidate_fa = int(candidate["unmatched_predicted_pixels"])
    fa_trigger = (original_fa == 0 and candidate_fa > 0) or (
        original_fa > 0 and candidate_fa * 4 > original_fa * 5
    )
    target_gain = int(candidate["matched_target_count"]) - int(original["matched_target_count"])
    if fa_trigger and target_gain < 2:
        violations.append(
            {
                **prefix,
                "rule": "fa_increase_over_25_percent_without_2_matched_gain",
                "original_unmatched_predicted_pixels": original_fa,
                "ec_unmatched_predicted_pixels": candidate_fa,
                "matched_target_gain": target_gain,
            }
        )
    return violations


def gate_b(records: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    dominance: dict[str, Any] = {}
    for dataset in DATASETS:
        dominance[dataset] = {}
        for role in ROLES:
            original = records[dataset][role]["original"]
            ec = records[dataset][role]["ec_tss_v3_1"]
            violations.extend(severe_violations(dataset, role, original, ec))
            dominance[dataset][role] = point_dominates(original, ec, FROZEN_METRICS)
    dual_role = [dataset for dataset in DATASETS if all(dominance[dataset].values())]
    return {
        "severe_degradation_violation_count": len(violations),
        "severe_degradation_violations": violations,
        "stage_progression_pass": len(violations) < 5 and not dual_role,
        "formal_zero_violation_pass": not violations and not dual_role,
        "original_strict_dominance_by_dataset_role": dominance,
        "original_dual_role_dominated_datasets": dual_role,
    }


def gate_c(records: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "NUDT-SIRST": ("matched_target_count", "miou", "niou"),
        "IRSTD-1K": ("matched_target_count", "miou"),
    }
    violations: list[dict[str, Any]] = []
    for dataset, metrics in checks.items():
        for role in ROLES:
            anchor = records[dataset][role]["lambda_0p005"]
            ec = records[dataset][role]["ec_tss_v3_1"]
            for metric in metrics:
                drop = metric_value(anchor, metric) - metric_value(ec, metric)
                limit = 2 if metric == "matched_target_count" else SERIOUS_IOU_DROP_QUANTA
                if drop >= limit:
                    violations.append(
                        {
                            "dataset": dataset,
                            "checkpoint_role": role,
                            "metric": metric,
                            "drop": drop,
                            "failure_limit": limit,
                        }
                    )
    return {"passed": not violations, "violations": violations}


def gate_e(records: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    unique_ec_cells: list[str] = []
    for dataset in DATASETS:
        for role in ROLES:
            cell_key = f"{dataset}/{role}"
            points = records[dataset][role]
            dominated_by: dict[str, list[str]] = {}
            vectors = {
                recipe: tuple(metric_value(points[recipe], metric) for metric in FROZEN_METRICS)
                for recipe in RECIPES
            }
            for recipe in RECIPES:
                dominated_by[recipe] = [
                    other
                    for other in RECIPES
                    if other != recipe and point_dominates(points[other], points[recipe], FROZEN_METRICS)
                ]
            ec_unique = not dominated_by["ec_tss_v3_1"] and all(
                vectors[other] != vectors["ec_tss_v3_1"] for other in RECIPES if other != "ec_tss_v3_1"
            )
            if ec_unique:
                unique_ec_cells.append(cell_key)
            cells[cell_key] = {
                "ec_unique_non_dominated": ec_unique,
                "dominated_by": dominated_by,
                "vectors": {recipe: list(vector) for recipe, vector in vectors.items()},
            }

    dual_role_dominators: list[dict[str, str]] = []
    for dataset in DATASETS:
        for reference in REFERENCE_RECIPES:
            if all(
                point_dominates(
                    records[dataset][role][reference],
                    records[dataset][role]["ec_tss_v3_1"],
                    FROZEN_METRICS,
                )
                for role in ROLES
            ):
                dual_role_dominators.append({"dataset": dataset, "reference": reference})
    passed = len(unique_ec_cells) >= 2 and not dual_role_dominators
    return {
        "passed": passed,
        "ec_unique_non_dominated_cell_count": len(unique_ec_cells),
        "required_count": 2,
        "ec_unique_non_dominated_cells": unique_ec_cells,
        "dual_role_dominators": dual_role_dominators,
        "cells": cells,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(line.strip() != "", f"blank JSONL line: {path}:{line_number}")
        value = json.loads(line, object_pairs_hook=_reject_duplicates)
        _require(isinstance(value, dict), f"JSONL row is not object: {path}:{line_number}")
        rows.append(value)
    return rows


def diagnostics_gate(ec_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    all_epoch_logs_complete = True
    both_risks_active = True
    for dataset in DATASETS:
        path = ec_root / "runs" / dataset / "final_ec_tss_v3_1" / "seed_42" / "metrics.jsonl"
        rows = _read_jsonl(path)
        epochs = [row.get("epoch") for row in rows]
        epoch_complete = epochs == list(range(1, 1001))
        all_epoch_logs_complete &= epoch_complete
        first20 = rows[:20]
        positive_seen = any(_finite(row.get("train_ec_tss_positive_risk_mass_mean"), "positive risk") > 0 for row in first20)
        negative_seen = any(_finite(row.get("train_ec_tss_negative_risk_mass_mean"), "negative risk") > 0 for row in first20)
        both_risks_active &= positive_seen and negative_seen

        def window_mean(field: str, start: int, stop: int) -> float:
            values = [_finite(row.get(field), field) for row in rows[start - 1 : stop]]
            return sum(values) / len(values)

        datasets[dataset] = {
            "metrics_jsonl": {"path": str(path.resolve()), "sha256": file_sha256(path)},
            "epoch_1_1000_complete": epoch_complete,
            "first20_positive_risk_nonzero": positive_seen,
            "first20_negative_risk_nonzero": negative_seen,
            "window_means": {
                "epochs_1_200": {
                    "positive_risk_mass": window_mean("train_ec_tss_positive_risk_mass_mean", 1, 200),
                    "negative_risk_mass": window_mean("train_ec_tss_negative_risk_mass_mean", 1, 200),
                    "weighted_survival": window_mean("train_ec_tss_weighted_survival_mean", 1, 200),
                },
                "epochs_801_1000": {
                    "positive_risk_mass": window_mean("train_ec_tss_positive_risk_mass_mean", 801, 1000),
                    "negative_risk_mass": window_mean("train_ec_tss_negative_risk_mass_mean", 801, 1000),
                    "weighted_survival": window_mean("train_ec_tss_weighted_survival_mean", 801, 1000),
                },
            },
        }
    return {
        "epoch_logs_complete": all_epoch_logs_complete,
        "both_risk_branches_active_in_first20": both_risks_active,
        "fixed_checkpoint_cell_separation_completed": False,
        "complete": False,
        "performance_failure_makes_deep_cell_diagnostics_non_blocking": True,
        "datasets": datasets,
    }


def engineering_gate(ec_root: Path, bindings: Mapping[str, Any]) -> dict[str, Any]:
    progress_checks: dict[str, Any] = {}
    for dataset in DATASETS:
        path = ec_root / "runs" / dataset / "final_ec_tss_v3_1" / "seed_42" / "progress.json"
        progress = load_json(path)
        progress_checks[dataset] = {
            "status": progress.get("status"),
            "completed_epoch": progress.get("completed_epoch"),
            "planned_total_epochs": progress.get("planned_total_epochs"),
            "passed": progress.get("status") == "complete"
            and progress.get("completed_epoch") == 1000
            and progress.get("planned_total_epochs") == 1000,
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    pilot_path = ec_root / "pilot_gate" / "pilot200_runtime_gate.json"
    pilot = load_json(pilot_path)
    status_path = ec_root / "launch" / "formal" / "supervisor_status.json"
    status = load_json(status_path)
    ec_evaluation_evidence = {
        key: value["engineering_evidence"]
        for key, value in bindings.items()
        if key.endswith("/ec_tss_v3_1")
    }
    evaluation_engineering_pass = (
        len(ec_evaluation_evidence) == len(DATASETS) * len(ROLES)
        and all(
            all(
                evidence[field]
                for field in (
                    "strict_inference_state_load",
                    "training_only_tss_heads_removed",
                    "checkpoint_metric_replay_passed",
                    "training_runtime_sources_validated",
                )
            )
            and isinstance(evidence["inference_state_sha256"], str)
            and len(evidence["inference_state_sha256"]) == 64
            for evidence in ec_evaluation_evidence.values()
        )
    )
    passed = (
        all(record["passed"] for record in progress_checks.values())
        and len(bindings) == len(DATASETS) * len(ROLES) * len(RECIPES)
        and evaluation_engineering_pass
        and pilot.get("status") == "passed"
        and pilot.get("gate_passed") is True
        and status.get("status") == "evaluation_complete_comparison_pending"
    )
    return {
        "passed": passed,
        "formal1000": progress_checks,
        "evaluation_artifact_count": len(bindings),
        "expected_evaluation_artifact_count": len(DATASETS) * len(ROLES) * len(RECIPES),
        "ec_evaluation_engineering_passed": evaluation_engineering_pass,
        "ec_evaluation_engineering_evidence": ec_evaluation_evidence,
        "pilot_gate": {"path": str(pilot_path.resolve()), "sha256": file_sha256(pilot_path), "passed": pilot.get("gate_passed") is True},
        "launcher_status": {"path": str(status_path.resolve()), "sha256": file_sha256(status_path), "status": status.get("status")},
    }


def display_point(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "miou": point["miou"],
        "niou": point["niou"],
        "pd": point["pd"],
        "matched_target_count": point["matched_target_count"],
        "target_count": point["target_count"],
        "fa": point["fa"],
        "unmatched_predicted_pixels": point["unmatched_predicted_pixels"],
        "pixel_precision": point["pixel_precision"],
        "pixel_f1": point["pixel_f1"],
        "false_objects_per_image": point["false_objects_per_image"],
        "matched_tiny_target_count": point["matched_tiny_target_count"],
        "tiny_target_count": point["tiny_target_count"],
        "object_precision": point["object_precision"],
        "object_f1": point["object_f1"],
    }


def build_comparison(
    records: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    ec_root: Path = EC_ROOT,
) -> dict[str, Any]:
    gate_a = engineering_gate(ec_root, bindings)
    gate_b_record = gate_b(records)
    gate_c_record = gate_c(records)
    frozen_pairwise = {
        reference: pairwise_summary(records, reference, FROZEN_METRICS)
        for reference in REFERENCE_RECIPES
    }
    gate_d_pass = (
        frozen_pairwise["tss_off"]["better_than_worse"]
        and frozen_pairwise["lambda_0p005"]["better_than_worse"]
    )
    gate_e_record = gate_e(records)
    gate_f_record = diagnostics_gate(ec_root)
    joint_pairwise = {
        reference: pairwise_summary(records, reference, JOINT_METRICS)
        for reference in REFERENCE_RECIPES
    }
    joint_pass = (
        joint_pairwise["tss_off"]["better_than_worse"]
        and joint_pairwise["lambda_0p005"]["better_than_worse"]
    )
    # The post-result joint audit is additive.  It may narrow claims, but it
    # must not silently rewrite the predeclared V3-A--V3-E decision contract.
    performance_pass = all(
        (
            gate_a["passed"],
            gate_b_record["formal_zero_violation_pass"],
            gate_c_record["passed"],
            gate_d_pass,
            gate_e_record["passed"],
        )
    )
    if not performance_pass:
        decision = FAIL_DECISION
    elif not gate_f_record["complete"]:
        decision = PENDING_DECISION
    else:
        decision = PASS_DECISION

    results_table = {
        dataset: {
            role: {
                recipe: display_point(records[dataset][role][recipe])
                for recipe in RECIPES
            }
            for role in ROLES
        }
        for dataset in DATASETS
    }
    output = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "decision": decision,
        "tss_optimization_closed": decision == FAIL_DECISION,
        "unified_tss_recipe_optimization_closed": decision == FAIL_DECISION,
        "tss_role_after_failure": (
            "optional_training_auxiliary_not_unified_default"
            if decision == FAIL_DECISION
            else None
        ),
        "tss_training_innovation_supported": decision == PASS_DECISION,
        "seed42_test_selected_operational_candidate": (
            "EC_TSS_V3_1" if decision == PASS_DECISION else None
        ),
        "seed42_operational_recipe_admissible": decision == PASS_DECISION,
        "global_operational_default": None,
        "paper_core_established": False,
        "training_recipe_finalized": False,
        "mainline_changed": False,
        "inference_architecture_changed": False,
        "next_model_focus_if_failed": "NER_then_QFG_then_TPD_single_component_diagnostics",
        "scope": "fixed_seed42_img_idx_test_selected",
        "selection_is_optimistic": True,
        "independent_test_confirmation": False,
        "stability_claim_supported": False,
        "fixed_threshold": FIXED_THRESHOLD,
        "datasets": list(DATASETS),
        "checkpoint_roles": list(ROLES),
        "recipe_population": list(RECIPES),
        "search_budget_disclosure": {
            "original_runs": 3,
            "positive_lambda_runs": 9,
            "tss_off_runs": 3,
            "ec_tss_v3_1_runs": 3,
            "total_runs": 18,
            "final_family_to_original_ratio": 5.0,
            "per_run_protocol_matched": True,
            "global_search_budget_equal": False,
        },
        "gates": {
            "V3_A_engineering": gate_a,
            "V3_B_original_floor": gate_b_record,
            "V3_C_anchor_strengths": gate_c_record,
            "V3_D_pairwise_performance": {"passed": gate_d_pass, "comparisons": frozen_pairwise},
            "V3_E_joint_pareto": gate_e_record,
            "V3_F_diagnostics": gate_f_record,
        },
        "additive_joint_quality_audit": {
            "status": "post_result_additive_metric_guard",
            "changes_frozen_v3_gate": False,
            "metrics": list(JOINT_METRICS),
            "positive_vote_gate_passed": joint_pass,
            "verdict": JOINT_MIXED_VERDICT,
            "comparisons": joint_pairwise,
            "reason": "Pd and component Fa are reported jointly with overlap and pixel-quality metrics.",
        },
        "results": results_table,
        "artifact_bindings": dict(bindings),
        "source_sha256": {
            "experiments/compare_finalize_ec_tss_v3_1.py": file_sha256(Path(__file__)),
        },
        "no_fabricated_results": True,
    }
    output["artifact_sha256"] = canonical_sha256(output)
    return output


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# EC-TSS V3.1 三数据集最终比较",
        "",
        f"- 决策：`{result['decision']}`",
        f"- 统一 TSS 配方优化关闭：`{str(result['unified_tss_recipe_optimization_closed']).lower()}`",
        f"- 失败后 TSS 定位：`{result['tss_role_after_failure']}`",
        f"- 固定阈值：`{result['fixed_threshold']}`",
        "- 口径：seed 42、各数据集 img_idx/test、test-selected；不支持跨随机性外推。",
        "",
        "## EC-TSS 正式固定点",
        "",
        "| 数据集 | checkpoint | mIoU | nIoU | Pd | component-Fa | Pixel Precision | Pixel F1 | 错误目标/图 | tiny-Pd |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for role in ROLES:
            point = result["results"][dataset][role]["ec_tss_v3_1"]
            tiny = f"{point['matched_tiny_target_count']}/{point['tiny_target_count']}"
            lines.append(
                f"| {dataset} | {role} | {point['miou']:.6f} | {point['niou']:.6f} | "
                f"{point['matched_target_count']}/{point['target_count']} ({point['pd']:.6f}) | "
                f"{point['fa']:.6e} | {point['pixel_precision']:.6f} | "
                f"{point['pixel_f1']:.6f} | {point['false_objects_per_image']:.6f} | {tiny} |"
            )
    lines.extend(["", "## 裁决门", "", "| 门 | 结果 |", "|---|---:|"])
    gates = result["gates"]
    lines.extend(
        [
            f"| V3-A 工程闭环 | {gates['V3_A_engineering']['passed']} |",
            f"| V3-B 严重退化为零 | {gates['V3_B_original_floor']['formal_zero_violation_pass']}（{gates['V3_B_original_floor']['severe_degradation_violation_count']}项） |",
            f"| V3-C 保留旧强项 | {gates['V3_C_anchor_strengths']['passed']} |",
            f"| V3-D 相对 off/.005 正向票更多 | {gates['V3_D_pairwise_performance']['passed']} |",
            f"| V3-E 至少两个独有非支配点 | {gates['V3_E_joint_pareto']['passed']}（{gates['V3_E_joint_pareto']['ec_unique_non_dominated_cell_count']}个） |",
            f"| 联合像素质量趋势（不改写冻结 Gate） | {result['additive_joint_quality_audit']['verdict']} |",
        ]
    )
    lines.extend(["", "## 成对比较票数", "", "| 口径 | 参考 | 更好 | 相同 | 更差 |", "|---|---|---:|---:|---:|"])
    for label, container in (
        ("冻结五指标", gates["V3_D_pairwise_performance"]["comparisons"]),
        ("联合八指标", result["additive_joint_quality_audit"]["comparisons"]),
    ):
        for reference in REFERENCE_RECIPES:
            row = container[reference]
            lines.append(f"| {label} | {reference} | {row['better']} | {row['equal']} | {row['worse']} |")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- `best_pd` 是 Pd 优先的高召回工作点，不单独代表整体性能。",
            "- 正式结论同时查看 Pd、component-Fa、mIoU、nIoU、pixel precision/F1 和错误目标数。",
            "- 本结果来自固定 seed 42 的 test-selected 开发实验。",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, content: str, *, overwrite: bool) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if destination.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(destination)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, default=POSITIVE_ROOT)
    parser.add_argument("--off-root", type=Path, default=OFF_ROOT)
    parser.add_argument("--ec-root", type=Path, default=EC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records, bindings = load_records(
        positive_root=args.positive_root,
        off_root=args.off_root,
        ec_root=args.ec_root,
    )
    result = build_comparison(records, bindings, ec_root=args.ec_root)
    json_path = args.output_dir / "ec_tss_v3_1_final_comparison.json"
    markdown_path = args.output_dir / "ec_tss_v3_1_final_comparison.md"
    atomic_write(
        json_path,
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        overwrite=args.overwrite,
    )
    atomic_write(markdown_path, render_markdown(result), overwrite=args.overwrite)
    print(f"EC_TSS_V3_1_FINAL decision={result['decision']} json={json_path} markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
