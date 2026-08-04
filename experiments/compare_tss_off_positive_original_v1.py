#!/usr/bin/env python3
"""Compare the frozen TSS-off diagnostic without reopening positive selection.

Axis 1 applies the V2 selector's *unchanged* severe-degradation and
Original-dual-role-dominance gates to TSS-off versus Original.  Axis 2 reports
the Pareto relation of TSS-off to each preregistered positive TSS request in
the same 30 quantized dataset/role/metric cells.  Only threshold-0.5 points
are read; descriptive Pd--Fa sweeps never enter either axis.

The command also has a result-free ``--preflight`` mode.  It validates the
already frozen positive-selection artifacts and the comparator contract, but
does not invent placeholder TSS-off metrics or a diagnostic decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import select_three_dataset_global_tss_recipe_v2 as selector  # noqa: E402


COMPARISON_INPUT_SCHEMA = "sctransnet_three_dataset_tss_off_comparison_input/v1"
COMPARISON_SCHEMA = "sctransnet_three_dataset_tss_off_comparison/v1"
PREFLIGHT_SCHEMA = "sctransnet_three_dataset_tss_off_comparison_preflight/v1"
EVALUATION_SCHEMA = "sctransnet_three_dataset_v2_evaluation_v1"

DATASETS = selector.DATASETS
CHECKPOINT_ROLES = selector.CHECKPOINT_ROLES
CANDIDATE_LAMBDAS = selector.CANDIDATE_LAMBDAS
METRICS = selector.METRICS
SELECTION_SPLIT = selector.SELECTION_SPLIT
TRAINING_SEED = selector.TRAINING_SEED
FIXED_THRESHOLD = selector.FIXED_THRESHOLD
OFF_METHOD = "final_tss_off"
OFF_WEIGHT = Decimal("0")

DEFAULT_POSITIVE_ROOT = REPO_ROOT / "results" / "three_dataset_seed42_global_tss_v2"
DEFAULT_TSS_OFF_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_COMPARISON_DIR = DEFAULT_TSS_OFF_ROOT / "comparison"
COMPARISON_FILENAME = "tss_off_comparison_v1.json"
PREFLIGHT_FILENAME = "tss_off_comparison_preflight_v1.json"

COMPARATOR_SOURCE_PATH = Path(__file__).resolve()
SOURCE_PATHS = {
    "experiments/compare_tss_off_positive_original_v1.py": COMPARATOR_SOURCE_PATH,
    **selector.SOURCE_PATHS,
}

POSITIVE_DECISION = "NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED"
ELIGIBLE_DECISION = "TSS_OFF_OPERATIONALLY_ADMISSIBLE_SEED42_TEST_SELECTED"
INELIGIBLE_DECISION = "TSS_OFF_NOT_GLOBALLY_ADMISSIBLE_SEED42_TEST_SELECTED"

RELATION_BY_SIGNS = {
    "dominates": "off is no worse in every cell and better in at least one",
    "dominated": "positive TSS is no worse in every cell and better in at least one",
    "equal": "all quantized/count cells are equal",
    "incomparable": "each side is better in at least one cell",
}


class TssOffComparisonError(ValueError):
    """Raised when an artifact violates the frozen TSS-off contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TssOffComparisonError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    observed = set(mapping)
    required = set(expected)
    _require(
        observed == required,
        f"{label} keys differ: missing={sorted(required - observed)}, "
        f"extra={sorted(observed - required)}",
    )


def _decimal(value: Any, label: str) -> Decimal:
    _require(not isinstance(value, bool), f"{label} must be numeric, not bool")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise TssOffComparisonError(f"{label} must be numeric") from error
    _require(result.is_finite(), f"{label} must be finite")
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> dict[str, str]:
    return {
        relative: _file_sha256(path)
        for relative, path in sorted(SOURCE_PATHS.items())
    }


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return _canonical_sha256(unsigned)


def seal_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["artifact_sha256"] = _artifact_sha256(output)
    return output


def validate_artifact_sha256(payload: Mapping[str, Any], label: str) -> None:
    declared = payload.get("artifact_sha256")
    _require(
        isinstance(declared, str) and len(declared) == 64,
        f"{label}.artifact_sha256 must be a SHA-256",
    )
    _require(
        declared == _artifact_sha256(payload),
        f"{label}.artifact_sha256 differs from canonical content",
    )


def _lambda_key(value: Decimal) -> str:
    return format(value, "f").rstrip("0").rstrip(".")


def _lambda_field(value: Decimal) -> str:
    return "off_vs_" + _lambda_key(value).replace(".", "p")


def _validate_sha_map(raw: Any, label: str) -> dict[str, str]:
    mapping = _mapping(raw, label)
    _require(bool(mapping), f"{label} must not be empty")
    ready: dict[str, str] = {}
    for key, value in mapping.items():
        _require(isinstance(key, str) and bool(key), f"{label} has an invalid key")
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{label}.{key} must be a lowercase SHA-256",
        )
        ready[key] = value
    return dict(sorted(ready.items()))


def validate_positive_selection(
    selection_payload: Mapping[str, Any],
    selector_input_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the stored negative decision without rerunning the selector."""

    selection = _mapping(selection_payload, "positive_selection")
    _require(
        selection.get("schema") == selector.OUTPUT_SCHEMA,
        f"positive selection schema must be {selector.OUTPUT_SCHEMA!r}",
    )
    for field, expected in (
        ("status", "complete"),
        ("decision", POSITIVE_DECISION),
        ("global_tss_recipe_established", False),
        ("global_tss_lambda", None),
        ("selection_split", SELECTION_SPLIT),
        ("test_selected", True),
        ("threshold", float(FIXED_THRESHOLD)),
        ("training_seed", TRAINING_SEED),
        ("datasets", list(DATASETS)),
        ("checkpoint_roles", list(CHECKPOINT_ROLES)),
        ("candidate_lambdas", [float(value) for value in CANDIDATE_LAMBDAS]),
        ("metrics_used", list(METRICS)),
        ("candidate_ranking", []),
        ("no_fabricated_results", True),
    ):
        _require(
            selection.get(field) == expected,
            f"positive selection {field} differs from the frozen negative decision",
        )
    _require(
        selection.get("input_sha256") == selector._input_sha256(selector_input_payload),
        "positive selection is not bound to selector_input_v2.json",
    )
    candidates = _mapping(selection.get("candidates"), "positive_selection.candidates")
    lambda_keys = tuple(_lambda_key(value) for value in CANDIDATE_LAMBDAS)
    _exact_keys(candidates, lambda_keys, "positive_selection.candidates")
    for lambda_key in lambda_keys:
        candidate = _mapping(candidates[lambda_key], f"positive candidate {lambda_key}")
        _require(
            candidate.get("gate_eligible") is False,
            f"positive candidate {lambda_key} unexpectedly passed its frozen gate",
        )
    stored_violation_counts: dict[str, int] = {}
    for lambda_key in lambda_keys:
        candidate = _mapping(candidates[lambda_key], f"positive candidate {lambda_key}")
        violations = candidate.get("severe_degradation_violations")
        _require(
            isinstance(violations, list),
            f"positive candidate {lambda_key} severe violations must be a list",
        )
        stored_violation_counts[lambda_key] = len(violations)
    minimum_count = min(stored_violation_counts.values())
    minimum_keys = [
        key for key, count in stored_violation_counts.items() if count == minimum_count
    ]
    _require(
        minimum_keys == ["0.005"],
        "stored positive result no longer has 0.005 as its unique fewest-violation anchor",
    )
    current_sources = selector.selector_source_sha256()
    stored_sources = _validate_sha_map(
        selection.get("source_sha256"), "positive_selection.source_sha256"
    )
    _require(
        stored_sources == current_sources,
        "frozen positive selector/data-protocol sources changed",
    )
    launch_binding = _mapping(
        selection.get("launch_plan_binding"),
        "positive_selection.launch_plan_binding",
    )
    _require(
        launch_binding.get("provided") is True
        and launch_binding.get("validated") is True,
        "positive selection lacks a validated launch-plan binding",
    )
    return {
        "decision": POSITIVE_DECISION,
        "global_tss_recipe_established": False,
        "global_tss_lambda": None,
        "candidate_ranking": [],
        "descriptive_fewest_violation_anchor": 0.005,
        "descriptive_anchor_is_selected_candidate": False,
        "stored_severe_violation_counts": stored_violation_counts,
        "positive_selection_recomputed": False,
        "prior_positive_conclusion_authoritative": True,
        "source_sha256": current_sources,
    }


def validate_tss_off_input(
    payload: Mapping[str, Any],
    *,
    normalized_positive: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize the exact three-run, two-role TSS-off input."""

    root = _mapping(payload, "tss_off_input")
    _exact_keys(
        root,
        (
            "schema",
            "selection_split",
            "test_selected",
            "selection_is_optimistic",
            "independent_test_confirmation",
            "training_seed",
            "threshold",
            "checkpoint_roles",
            "requested_tss_weight",
            "datasets",
        ),
        "tss_off_input",
    )
    for field, expected in (
        ("schema", COMPARISON_INPUT_SCHEMA),
        ("selection_split", SELECTION_SPLIT),
        ("test_selected", True),
        ("selection_is_optimistic", True),
        ("independent_test_confirmation", False),
        ("training_seed", TRAINING_SEED),
        ("checkpoint_roles", list(CHECKPOINT_ROLES)),
    ):
        _require(root.get(field) == expected, f"tss_off_input.{field} must be {expected!r}")
    _require(
        _decimal(root.get("threshold"), "tss_off_input.threshold") == FIXED_THRESHOLD,
        "tss_off_input.threshold must be exactly 0.5",
    )
    _require(
        _decimal(root.get("requested_tss_weight"), "requested_tss_weight")
        == OFF_WEIGHT,
        "requested_tss_weight must be exactly 0.0",
    )
    datasets = _mapping(root.get("datasets"), "tss_off_input.datasets")
    _exact_keys(datasets, DATASETS, "tss_off_input.datasets")
    normalized_datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        raw_dataset = _mapping(datasets[dataset], f"tss_off_input.datasets.{dataset}")
        _exact_keys(
            raw_dataset,
            (
                "selection_split",
                "img_idx_test_sha256",
                "img_idx_test_ordered_ids_sha256",
                "tss_off",
            ),
            f"tss_off_input.datasets.{dataset}",
        )
        _require(
            raw_dataset.get("selection_split") == SELECTION_SPLIT,
            f"{dataset} selection_split must be {SELECTION_SPLIT!r}",
        )
        roles = _mapping(raw_dataset.get("tss_off"), f"{dataset}.tss_off")
        _exact_keys(roles, CHECKPOINT_ROLES, f"{dataset}.tss_off")
        normalized_roles = {
            role: selector._normalize_point(roles[role], f"{dataset}.tss_off.{role}")
            for role in CHECKPOINT_ROLES
        }
        normalized_dataset = {
            "selection_split": SELECTION_SPLIT,
            "img_idx_test_sha256": raw_dataset.get("img_idx_test_sha256"),
            "img_idx_test_ordered_ids_sha256": raw_dataset.get(
                "img_idx_test_ordered_ids_sha256"
            ),
            "tss_off": normalized_roles,
        }
        if normalized_positive is not None:
            positive_dataset = normalized_positive["datasets"][dataset]
            for field in (
                "img_idx_test_sha256",
                "img_idx_test_ordered_ids_sha256",
            ):
                _require(
                    normalized_dataset[field] == positive_dataset[field],
                    f"{dataset} {field} differs from the frozen positive input",
                )
            selector._validate_point_invariants(
                dataset,
                list(positive_dataset["original"].values())
                + [
                    point
                    for candidate in positive_dataset["final_candidates"].values()
                    for point in candidate.values()
                ]
                + list(normalized_roles.values()),
            )
        normalized_datasets[dataset] = normalized_dataset
    return {
        "schema": COMPARISON_INPUT_SCHEMA,
        "selection_split": SELECTION_SPLIT,
        "test_selected": True,
        "selection_is_optimistic": True,
        "independent_test_confirmation": False,
        "training_seed": TRAINING_SEED,
        "threshold": float(FIXED_THRESHOLD),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "requested_tss_weight": 0.0,
        "datasets": normalized_datasets,
    }


def _relation_from_comparisons(comparisons: Sequence[int]) -> str:
    _require(bool(comparisons), "pairwise comparison vector must not be empty")
    if all(value == 0 for value in comparisons):
        return "equal"
    if all(value >= 0 for value in comparisons):
        return "dominates"
    if all(value <= 0 for value in comparisons):
        return "dominated"
    return "incomparable"


def compare_inputs(
    positive_input: Mapping[str, Any],
    tss_off_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Return both frozen diagnostic axes from already supplied metric inputs."""

    normalized_positive = selector.validate_input(positive_input)
    normalized_off = validate_tss_off_input(
        tss_off_input, normalized_positive=normalized_positive
    )

    severe: list[dict[str, Any]] = []
    original_dominance: dict[str, dict[str, bool]] = {}
    for dataset in DATASETS:
        original_dominance[dataset] = {}
        for role in CHECKPOINT_ROLES:
            original = normalized_positive["datasets"][dataset]["original"][role]
            off = normalized_off["datasets"][dataset]["tss_off"][role]
            severe.extend(
                selector._severe_degradation_violations(dataset, role, original, off)
            )
            original_dominance[dataset][role] = selector._original_strictly_dominates(
                original, off
            )
    dual_role_dominated = [
        dataset
        for dataset in DATASETS
        if all(original_dominance[dataset][role] for role in CHECKPOINT_ROLES)
    ]
    eligible = not severe and not dual_role_dominated

    pairwise: dict[str, Any] = {}
    relation_fields: dict[str, str] = {}
    for lambda_value in CANDIDATE_LAMBDAS:
        lambda_key = _lambda_key(lambda_value)
        cells: dict[str, dict[str, Any]] = {}
        signs: list[int] = []
        for dataset in DATASETS:
            for role in CHECKPOINT_ROLES:
                off = normalized_off["datasets"][dataset]["tss_off"][role]
                positive = normalized_positive["datasets"][dataset][
                    "final_candidates"
                ][lambda_key][role]
                for metric in METRICS:
                    off_value = selector._metric_value(off, metric)
                    positive_value = selector._metric_value(positive, metric)
                    if positive_value is None:
                        _require(
                            off_value is None,
                            f"tiny metric availability differs for {dataset}/{role}/{metric}",
                        )
                        continue
                    _require(
                        off_value is not None,
                        f"TSS-off lacks comparable {dataset}/{role}/{metric}",
                    )
                    sign = selector._compare(
                        off_value,
                        positive_value,
                        higher_is_better=selector.HIGHER_IS_BETTER[metric],
                    )
                    cell_key = f"{dataset}/{role}/{metric}"
                    cells[cell_key] = {
                        "off_value": off_value,
                        "positive_value": positive_value,
                        "comparison_from_off_perspective": sign,
                    }
                    signs.append(sign)
        relation = _relation_from_comparisons(signs)
        pairwise[lambda_key] = {
            "requested_tss_weight": float(lambda_value),
            "relation": relation,
            "relation_definition": RELATION_BY_SIGNS[relation],
            "vector_dimension": len(cells),
            "cells": cells,
            "off_better_cell_count": sum(value > 0 for value in signs),
            "equal_cell_count": sum(value == 0 for value in signs),
            "off_worse_cell_count": sum(value < 0 for value in signs),
        }
        relation_fields[_lambda_field(lambda_value)] = relation

    nominal_dimension = len(DATASETS) * len(CHECKPOINT_ROLES) * len(METRICS)
    effective_dimensions = {
        int(record["vector_dimension"]) for record in pairwise.values()
    }
    _require(
        len(effective_dimensions) == 1,
        "pairwise vectors have different dimensions across positive lambdas",
    )
    effective_dimension = effective_dimensions.pop()
    _require(
        0 < effective_dimension <= nominal_dimension,
        "pairwise vector dimension is outside the frozen nominal space",
    )
    for lambda_key, record in pairwise.items():
        _require(
            record["vector_dimension"] == effective_dimension,
            f"off versus {lambda_key} has a different effective cell set",
        )

    decision = ELIGIBLE_DECISION if eligible else INELIGIBLE_DECISION
    return {
        "decision": decision,
        "axis_1_tss_off_vs_original": {
            "off_gate_eligible": eligible,
            "severe_degradation_passed": not severe,
            "severe_degradation_violation_count": len(severe),
            "severe_degradation_violations": severe,
            "original_strict_dominance_by_dataset_role": original_dominance,
            "original_dual_role_dominated_datasets": dual_role_dominated,
            "eligibility_definition": (
                "zero severe-degradation violations and no dataset where "
                "Original strictly dominates both checkpoint roles"
            ),
        },
        "axis_2_tss_off_vs_positive": {
            "population": [float(value) for value in CANDIDATE_LAMBDAS],
            "nominal_pairwise_vector_dimension": nominal_dimension,
            "effective_pairwise_vector_dimension": effective_dimension,
            "unavailable_tiny_metric_policy": (
                "omit the tiny cell for all compared recipes on that dataset/role"
            ),
            "pairwise_relations": pairwise,
            **relation_fields,
            "all_relations_equal": all(
                record["relation"] == "equal" for record in pairwise.values()
            ),
            "secondary_axis_changes_axis_1_decision": False,
        },
    }


def _validate_evaluation(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    role: str,
    positive_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = _mapping(payload, f"evaluation[{dataset}/{role}]")
    for field, expected in (
        ("schema", EVALUATION_SCHEMA),
        ("status", "complete"),
        ("dataset", dataset),
        ("method", OFF_METHOD),
        ("checkpoint_role", role),
        ("seed", TRAINING_SEED),
        ("test_selected", True),
        ("selection_is_optimistic", True),
        ("no_fabricated_results", True),
        ("stability_claim_supported", False),
    ):
        _require(
            evaluation.get(field) == expected,
            f"evaluation[{dataset}/{role}].{field} must be {expected!r}",
        )
    _require(
        _decimal(evaluation.get("requested_tss_weight"), "evaluation weight")
        == OFF_WEIGHT,
        f"evaluation[{dataset}/{role}] requested_tss_weight must be 0.0",
    )
    threshold_roles = _mapping(
        evaluation.get("threshold_roles"), f"evaluation[{dataset}/{role}].threshold_roles"
    )
    for field in (
        "checkpoint_selection_threshold",
        "global_lambda_selection_threshold",
        "main_table_threshold",
    ):
        _require(
            _decimal(threshold_roles.get(field), f"threshold_roles.{field}")
            == FIXED_THRESHOLD,
            f"evaluation[{dataset}/{role}] {field} must be 0.5",
        )
    _require(
        threshold_roles.get("descriptive_sweep_only") is True,
        f"evaluation[{dataset}/{role}] must mark its sweep descriptive-only",
    )
    data = _mapping(evaluation.get("data"), f"evaluation[{dataset}/{role}].data")
    for field, expected in (
        ("split", SELECTION_SPLIT),
        ("img_idx_test_sha256", positive_dataset["img_idx_test_sha256"]),
        (
            "img_idx_test_ordered_ids_sha256",
            positive_dataset["img_idx_test_ordered_ids_sha256"],
        ),
        ("sirst3_in_formal_matrix", False),
    ):
        _require(
            data.get(field) == expected,
            f"evaluation[{dataset}/{role}].data.{field} differs",
        )
    _validate_sha_map(
        evaluation.get("source_sha256"),
        f"evaluation[{dataset}/{role}].source_sha256",
    )
    # Deliberately read only the fixed point.  The descriptive sweep is not
    # even copied into the comparison input or its artifact binding.
    return selector._normalize_point(
        evaluation.get("fixed_threshold_0_5"),
        f"evaluation[{dataset}/{role}].fixed_threshold_0_5",
    )


def _expected_run_directory(tss_off_root: Path, dataset: str) -> Path:
    return (
        Path(tss_off_root)
        / "runs"
        / dataset
        / OFF_METHOD
        / f"seed_{TRAINING_SEED}"
    ).resolve()


def validate_tss_off_launch_plan(path: Path, *, tss_off_root: Path) -> dict[str, Any]:
    candidate = Path(path)
    digest = _file_sha256(candidate)
    plan = selector.load_json(candidate)
    _require(
        isinstance(plan.get("schema"), str) and "tss_off" in plan["schema"],
        "TSS-off launch-plan schema must identify the tss_off stage",
    )
    if "dataset_order" in plan:
        _require(plan["dataset_order"] == list(DATASETS), "launch dataset order differs")
    _require(plan.get("worker_count") == len(DATASETS), "TSS-off plan must have 3 workers")
    workers = plan.get("workers")
    _require(isinstance(workers, list) and len(workers) == len(DATASETS), "workers differ")
    by_dataset: dict[str, Mapping[str, Any]] = {}
    for index, worker_raw in enumerate(workers):
        worker = _mapping(worker_raw, f"launch_plan.workers[{index}]")
        dataset = worker.get("dataset")
        _require(dataset in DATASETS and dataset not in by_dataset, "worker dataset differs")
        for field, expected in (
            ("method", OFF_METHOD),
            ("requested_tss_weight", 0.0),
            ("seed", TRAINING_SEED),
            ("threshold", float(FIXED_THRESHOLD)),
            ("checkpoint_roles", list(CHECKPOINT_ROLES)),
        ):
            _require(worker.get(field) == expected, f"worker {dataset} {field} differs")
        observed_run = worker.get("run_directory")
        _require(isinstance(observed_run, str), f"worker {dataset} lacks run_directory")
        _require(
            Path(observed_run).resolve() == _expected_run_directory(tss_off_root, dataset),
            f"worker {dataset} run_directory differs from the isolated off path",
        )
        by_dataset[str(dataset)] = worker
    _exact_keys(by_dataset, DATASETS, "launch_plan.worker datasets")
    static_inputs = _mapping(plan.get("static_inputs"), "launch_plan.static_inputs")
    _require(
        static_inputs.get("results_root") == str(Path(tss_off_root).resolve()),
        "launch-plan static results_root differs from the requested TSS-off root",
    )
    planned_sources = _mapping(
        static_inputs.get("sources"), "launch_plan.static_inputs.sources"
    )
    required_sources = {
        "comparator": COMPARATOR_SOURCE_PATH,
        "finalizer": REPO_ROOT / "experiments" / "finalize_tss_off_diagnostic_v1.py",
        "data_protocol": selector.DATA_PROTOCOL_SOURCE_PATH,
    }
    frozen_sources: dict[str, str] = {}
    for source_name, expected_path in required_sources.items():
        record = _mapping(
            planned_sources.get(source_name),
            f"launch_plan.static_inputs.sources.{source_name}",
        )
        observed_path = record.get("path")
        _require(
            isinstance(observed_path, str)
            and Path(observed_path).resolve(strict=True) == expected_path.resolve(strict=True),
            f"launch-plan {source_name} path differs",
        )
        observed_sha256 = _file_sha256(expected_path)
        _require(
            record.get("sha256") == observed_sha256,
            f"launch-plan frozen {source_name} SHA-256 differs from current source",
        )
        frozen_sources[source_name] = observed_sha256
    _require(
        _file_sha256(candidate) == digest,
        "TSS-off launch plan changed while it was validated",
    )
    return {
        "provided": True,
        "validated": True,
        "path": str(candidate.resolve()),
        "sha256": digest,
        "schema": plan["schema"],
        "worker_count": len(workers),
        "frozen_posttraining_source_sha256": frozen_sources,
        "run_directories": {
            dataset: str(_expected_run_directory(tss_off_root, dataset))
            for dataset in DATASETS
        },
    }


def assemble_tss_off_input(
    *,
    tss_off_root: Path,
    normalized_positive: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets: dict[str, Any] = {}
    evaluation_bindings: dict[str, Any] = {}
    for dataset in DATASETS:
        run_dir = _expected_run_directory(tss_off_root, dataset)
        positive_dataset = normalized_positive["datasets"][dataset]
        roles: dict[str, Any] = {}
        evaluation_bindings[dataset] = {}
        for role in CHECKPOINT_ROLES:
            path = run_dir / "evaluations" / f"{role}.json"
            before = _file_sha256(path)
            evaluation = selector.load_json(path)
            roles[role] = _validate_evaluation(
                evaluation,
                dataset=dataset,
                role=role,
                positive_dataset=positive_dataset,
            )
            _require(
                _file_sha256(path) == before,
                f"evaluation changed while it was validated: {path}",
            )
            evaluation_bindings[dataset][role] = {
                "path": str(path.resolve()),
                "sha256": before,
                "schema": evaluation["schema"],
                "fixed_threshold_field": "fixed_threshold_0_5",
                "descriptive_sweep_used_for_decision": False,
            }
        datasets[dataset] = {
            "selection_split": SELECTION_SPLIT,
            "img_idx_test_sha256": positive_dataset["img_idx_test_sha256"],
            "img_idx_test_ordered_ids_sha256": positive_dataset[
                "img_idx_test_ordered_ids_sha256"
            ],
            "tss_off": roles,
        }
    raw_input = {
        "schema": COMPARISON_INPUT_SCHEMA,
        "selection_split": SELECTION_SPLIT,
        "test_selected": True,
        "selection_is_optimistic": True,
        "independent_test_confirmation": False,
        "training_seed": TRAINING_SEED,
        "threshold": float(FIXED_THRESHOLD),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "requested_tss_weight": 0.0,
        "datasets": datasets,
    }
    validate_tss_off_input(raw_input, normalized_positive=normalized_positive)
    return raw_input, evaluation_bindings


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _file_sha256(path)}


def load_positive_bundle(
    *,
    positive_root: Path,
    selector_input_path: Path | None = None,
    selection_path: Path | None = None,
    launch_plan_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(positive_root).resolve()
    input_path = selector_input_path or root / "selection" / "selector_input_v2.json"
    stored_selection_path = (
        selection_path or root / "selection" / "global_tss_recipe_selection_v2.json"
    )
    stored_launch_path = launch_plan_path or root / "launch" / "formal" / "launch_plan.json"
    raw_input = selector.load_json(input_path)
    normalized_input = selector.validate_input(raw_input)
    stored_selection = selector.load_json(stored_selection_path)
    closure = validate_positive_selection(stored_selection, raw_input)
    computed_launch_binding = selector.validate_launch_plan_binding(
        stored_launch_path,
        current_sources=closure["source_sha256"],
    )
    stored_binding = _mapping(
        stored_selection.get("launch_plan_binding"), "positive launch binding"
    )
    for field in (
        "launch_plan_sha256",
        "frozen_selector_sha256",
        "frozen_data_protocol_sha256",
    ):
        _require(
            stored_binding.get(field) == computed_launch_binding[field],
            f"stored positive launch binding differs for {field}",
        )
    bindings = {
        "positive_root": str(root),
        "selector_input": _artifact_record(input_path),
        "selection": _artifact_record(stored_selection_path),
        "launch_plan": computed_launch_binding,
    }
    return raw_input, normalized_input, stored_selection, bindings


def fairness_and_search_budget() -> dict[str, Any]:
    original_runs = len(DATASETS)
    positive_runs = len(DATASETS) * len(CANDIDATE_LAMBDAS)
    off_runs = len(DATASETS)
    final_family_runs = positive_runs + off_runs
    return {
        "per_run_protocol_matched": True,
        "total_recipe_search_budget_equal": False,
        "original_training_runs": original_runs,
        "positive_tss_training_runs": positive_runs,
        "tss_off_training_runs": off_runs,
        "final_family_training_runs": final_family_runs,
        "total_training_runs": original_runs + final_family_runs,
        "final_to_original_recipe_search_ratio": final_family_runs / original_runs,
        "tss_off_added_after_positive_test_results": True,
        "permitted_claim": (
            "Original and Final-family runs use matched per-run protocols."
        ),
        "prohibited_claim": (
            "Original and the Final family have equal total recipe-search budgets."
        ),
    }


def build_preflight(
    *,
    positive_root: Path,
    positive_binding: Mapping[str, Any],
    tss_off_root: Path,
    launch_plan_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return seal_artifact(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": "preflight_complete",
            "decision": "NOT_EVALUATED",
            "comparison_executed": False,
            "tss_off_result_metrics_loaded": False,
            "tss_off_results_fabricated": False,
            "positive_selection_recomputed": False,
            "positive_selection_validation": {
                "decision": POSITIVE_DECISION,
                "authoritative": True,
            },
            "frozen_contract": {
                "datasets": list(DATASETS),
                "checkpoint_roles": list(CHECKPOINT_ROLES),
                "selection_split": SELECTION_SPLIT,
                "training_seed": TRAINING_SEED,
                "threshold": float(FIXED_THRESHOLD),
                "tss_off_requested_weight": 0.0,
                "primary_gate": (
                    "zero old-selector severe violations and zero Original "
                    "dual-role-dominated datasets"
                ),
                "secondary_relations": list(RELATION_BY_SIGNS),
                "nominal_pairwise_vector_dimension": (
                    len(DATASETS) * len(CHECKPOINT_ROLES) * len(METRICS)
                ),
                "miou_niou_quantization": "q=floor(x/1e-4+0.5)",
                "count_metrics_use_exact_integers": True,
                "descriptive_pd_fa_sweep_used_for_decision": False,
            },
            "roots": {
                "positive_root": str(Path(positive_root).resolve()),
                "tss_off_root": str(Path(tss_off_root).resolve()),
            },
            "positive_artifact_binding": dict(positive_binding),
            "tss_off_launch_plan_binding": (
                dict(launch_plan_binding)
                if launch_plan_binding is not None
                else {"provided": False, "validated": False}
            ),
            "fairness_and_search_budget": fairness_and_search_budget(),
            "source_sha256": source_sha256(),
            "test_selected": True,
            "selection_is_optimistic": True,
            "independent_test_confirmation": False,
            "causal_confirmation": False,
            "stability_claim_supported": False,
            "no_fabricated_results": True,
            "claim_scope": "fixed_seed42_img_idx_test_selected_paired_diagnostic",
        }
    )


def build_comparison(
    *,
    positive_input: Mapping[str, Any],
    positive_selection: Mapping[str, Any],
    tss_off_input: Mapping[str, Any],
    positive_binding: Mapping[str, Any],
    tss_off_root: Path,
    launch_plan_binding: Mapping[str, Any],
    evaluation_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    positive_closure = validate_positive_selection(positive_selection, positive_input)
    axes = compare_inputs(positive_input, tss_off_input)
    return seal_artifact(
        {
            "schema": COMPARISON_SCHEMA,
            "status": "complete",
            "decision": axes["decision"],
            "selection_split": SELECTION_SPLIT,
            "test_selected": True,
            "selection_is_optimistic": True,
            "independent_test_confirmation": False,
            "threshold": float(FIXED_THRESHOLD),
            "training_seed": TRAINING_SEED,
            "datasets": list(DATASETS),
            "checkpoint_roles": list(CHECKPOINT_ROLES),
            "metrics_used": list(METRICS),
            "requested_tss_weight": 0.0,
            "prior_positive_selection": positive_closure,
            "axis_1_tss_off_vs_original": axes["axis_1_tss_off_vs_original"],
            "axis_2_tss_off_vs_positive": axes["axis_2_tss_off_vs_positive"],
            "quantization_and_count_contract": {
                "miou_niou": "q=floor(x/1e-4+0.5)",
                "quantization_step": 0.0001,
                "count_metrics": [
                    "matched_target_count",
                    "unmatched_predicted_pixels",
                    "matched_tiny_target_count",
                ],
                "count_comparison": "exact_integer",
                "nominal_pairwise_vector_dimension": (
                    len(DATASETS) * len(CHECKPOINT_ROLES) * len(METRICS)
                ),
                "effective_pairwise_vector_dimension": axes[
                    "axis_2_tss_off_vs_positive"
                ]["effective_pairwise_vector_dimension"],
                "unavailable_tiny_metric_policy": (
                    "omit the tiny cell for all compared recipes on that dataset/role"
                ),
            },
            "frozen_gate_source": (
                "experiments/select_three_dataset_global_tss_recipe_v2.py"
            ),
            "old_positive_rank_population_modified": False,
            "positive_selection_recomputed": False,
            "descriptive_pd_fa_sweep_used_for_decision": False,
            "fairness_and_search_budget": fairness_and_search_budget(),
            "artifact_bindings": {
                "positive": dict(positive_binding),
                "tss_off_root": str(Path(tss_off_root).resolve()),
                "tss_off_launch_plan": dict(launch_plan_binding),
                "tss_off_evaluations": dict(evaluation_bindings),
                "tss_off_input_sha256": _canonical_sha256(tss_off_input),
            },
            "source_sha256": source_sha256(),
            "causal_confirmation": False,
            "stability_claim_supported": False,
            "final_training_recipe_established": False,
            "no_fabricated_results": True,
            "claim_scope": "fixed_seed42_img_idx_test_selected_paired_diagnostic",
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, default=DEFAULT_POSITIVE_ROOT)
    parser.add_argument("--tss-off-root", type=Path, default=DEFAULT_TSS_OFF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--positive-selector-input", type=Path)
    parser.add_argument("--positive-selection", type=Path)
    parser.add_argument("--positive-launch-plan", type=Path)
    parser.add_argument("--tss-off-launch-plan", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    (
        positive_input,
        normalized_positive,
        positive_selection,
        positive_binding,
    ) = load_positive_bundle(
        positive_root=args.positive_root,
        selector_input_path=args.positive_selector_input,
        selection_path=args.positive_selection,
        launch_plan_path=args.positive_launch_plan,
    )
    off_plan_path = (
        args.tss_off_launch_plan
        or args.tss_off_root / "launch" / "formal" / "launch_plan.json"
    )
    if args.preflight:
        plan_binding = None
        if off_plan_path.exists():
            plan_binding = validate_tss_off_launch_plan(
                off_plan_path, tss_off_root=args.tss_off_root
            )
        result = build_preflight(
            positive_root=args.positive_root,
            positive_binding=positive_binding,
            tss_off_root=args.tss_off_root,
            launch_plan_binding=plan_binding,
        )
        output_path = args.output_dir / PREFLIGHT_FILENAME
    else:
        plan_binding = validate_tss_off_launch_plan(
            off_plan_path, tss_off_root=args.tss_off_root
        )
        tss_off_input, evaluation_bindings = assemble_tss_off_input(
            tss_off_root=args.tss_off_root,
            normalized_positive=normalized_positive,
        )
        result = build_comparison(
            positive_input=positive_input,
            positive_selection=positive_selection,
            tss_off_input=tss_off_input,
            positive_binding=positive_binding,
            tss_off_root=args.tss_off_root,
            launch_plan_binding=plan_binding,
            evaluation_bindings=evaluation_bindings,
        )
        output_path = args.output_dir / COMPARISON_FILENAME
    selector.atomic_write_json(output_path, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
