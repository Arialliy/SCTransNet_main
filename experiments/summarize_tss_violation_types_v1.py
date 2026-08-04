#!/usr/bin/env python3
"""Seal positive-TSS violation types and the dataset/checkpoint-role matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tss_off_diagnostic_common_v1 as common


SCHEMA = "sctransnet_positive_tss_violation_matrix_v1"
LAMBDA_KEYS = ("0.0025", "0.005", "0.01")
CATEGORY_ORDER = (
    "pd",
    "tiny",
    "fa",
    "miou",
    "niou",
    "strict_domination",
)
RULE_CATEGORY = {
    "matched_target_drop_at_least_2": "pd",
    "matched_tiny_target_drop_at_least_2": "tiny",
    "fa_increase_over_25_percent_without_2_matched_gain": "fa",
    "miou_drop_at_least_0.005": "miou",
    "niou_drop_at_least_0.005": "niou",
    "original_dual_role_strict_dominance": "strict_domination",
}


def _monotonic(values: Sequence[int]) -> dict[str, Any]:
    nondecreasing = all(left <= right for left, right in zip(values, values[1:]))
    nonincreasing = all(left >= right for left, right in zip(values, values[1:]))
    return {
        "values_in_ascending_lambda_order": list(values),
        "nondecreasing": nondecreasing,
        "nonincreasing": nonincreasing,
        "monotonic_either_direction": nondecreasing or nonincreasing,
    }


def build_artifact(
    positive_root: Path = common.POSITIVE_RESULTS_ROOT,
    *,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    selection_path = (
        Path(selection_path)
        if selection_path is not None
        else Path(positive_root) / "selection" / "global_tss_recipe_selection_v2.json"
    )
    selection = common.load_json(selection_path)
    common.require(selection.get("status") == "complete", "positive selector is incomplete")
    common.require(
        selection.get("decision") == "NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED",
        "positive selector decision differs",
    )
    common.require(selection.get("global_tss_recipe_established") is False, "positive selector gate differs")
    candidates = selection.get("candidates")
    common.require(isinstance(candidates, dict) and set(candidates) == set(LAMBDA_KEYS), "selector candidates differ")

    type_matrix: dict[str, Any] = {}
    role_matrix: dict[str, Any] = {
        dataset: {
            role: {
                lambda_key: {"count": 0, "violations": []}
                for lambda_key in LAMBDA_KEYS
            }
            for role in common.CHECKPOINT_ROLES
        }
        for dataset in common.DATASETS
    }
    for lambda_key in LAMBDA_KEYS:
        candidate = candidates[lambda_key]
        common.require(isinstance(candidate, dict), f"candidate {lambda_key} is malformed")
        violations = candidate.get("severe_degradation_violations")
        common.require(isinstance(violations, list), f"candidate {lambda_key} lacks violations")
        counts = {category: 0 for category in CATEGORY_ORDER}
        normalized: list[dict[str, Any]] = []
        for violation in violations:
            common.require(isinstance(violation, dict), "violation record is malformed")
            dataset = violation.get("dataset")
            role = violation.get("checkpoint_role")
            rule = violation.get("rule")
            common.require(dataset in common.DATASETS, f"violation dataset differs: {dataset}")
            common.require(role in common.CHECKPOINT_ROLES, f"violation role differs: {role}")
            common.require(rule in RULE_CATEGORY, f"unknown severe-degradation rule: {rule}")
            category = RULE_CATEGORY[rule]
            counts[category] += 1
            record = dict(violation)
            record["category"] = category
            normalized.append(record)
            cell = role_matrix[dataset][role][lambda_key]
            cell["count"] += 1
            cell["violations"].append(record)
        dominated = candidate.get("original_dual_role_dominated_datasets")
        common.require(isinstance(dominated, list), "strict-domination list is malformed")
        for dataset in dominated:
            common.require(dataset in common.DATASETS, "strict-domination dataset differs")
            counts["strict_domination"] += 1
        common.require(sum(counts.values()) == len(violations) + len(dominated), "violation type counts differ")
        type_matrix[lambda_key] = {
            "total_violation_count": len(violations),
            "counts": counts,
            "strictly_dominated_datasets": list(dominated),
            "violations": normalized,
        }

    totals = [type_matrix[key]["total_violation_count"] for key in LAMBDA_KEYS]
    category_monotonicity = {
        category: _monotonic(
            [type_matrix[key]["counts"][category] for key in LAMBDA_KEYS]
        )
        for category in CATEGORY_ORDER
    }
    role_monotonicity = {
        dataset: {
            role: _monotonic(
                [role_matrix[dataset][role][key]["count"] for key in LAMBDA_KEYS]
            )
            for role in common.CHECKPOINT_ROLES
        }
        for dataset in common.DATASETS
    }
    total_monotonicity = _monotonic(totals)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "positive_root": str(Path(positive_root).resolve()),
        "selection": common.artifact_record(selection_path),
        "lambda_order": list(LAMBDA_KEYS),
        "violation_type_order": list(CATEGORY_ORDER),
        "violation_type_matrix": type_matrix,
        "dataset_checkpoint_role_matrix": role_matrix,
        "monotonicity": {
            "total": total_monotonicity,
            "by_type": category_monotonicity,
            "by_dataset_checkpoint_role": role_monotonicity,
            "simple_tss_strength_explanation_supported": total_monotonicity[
                "nondecreasing"
            ],
        },
        "gate_o1": {
            "violation_type_matrix_complete": True,
            "dataset_checkpoint_role_matrix_complete": True,
        },
        "source_sha256": {
            "experiments/summarize_tss_violation_types_v1.py": common.file_sha256(Path(__file__)),
            "experiments/tss_off_diagnostic_common_v1.py": common.file_sha256(Path(common.__file__)),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, default=common.POSITIVE_RESULTS_ROOT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            common.POSITIVE_RESULTS_ROOT
            / "pre_tss_off_gate_o1"
            / "violation_matrix_v1.json"
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = build_artifact(args.positive_root, selection_path=args.selection)
    action = "checked_only"
    if not args.check_only:
        action = common.write_once_or_identical(args.output, artifact)
    totals = {
        key: value["total_violation_count"]
        for key, value in artifact["violation_type_matrix"].items()
    }
    print(
        json.dumps(
            {
                "status": "complete",
                "action": action,
                "total_violations": totals,
                "output": None if args.check_only else str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
