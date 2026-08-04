#!/usr/bin/env python3
"""Seal the two-axis TSS-off comparison into the diagnostic terminal state.

This stage does not recompute metrics, rerun the old positive selector, or
change either comparison axis.  It verifies the comparison's canonical hash
and source/artifact bindings, then emits the conservative operational fields
and next action required by the frozen TSS-off protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import compare_tss_off_positive_original_v1 as comparator  # noqa: E402
from experiments import select_three_dataset_global_tss_recipe_v2 as selector  # noqa: E402


FINAL_SCHEMA = "sctransnet_three_dataset_tss_off_diagnostic/v1"
PREFLIGHT_SCHEMA = "sctransnet_three_dataset_tss_off_diagnostic_preflight/v1"
FINAL_FILENAME = "tss_off_diagnostic_v1.json"
PREFLIGHT_FILENAME = "tss_off_diagnostic_preflight_v1.json"
FINALIZER_SOURCE_PATH = Path(__file__).resolve()


class TssOffFinalizationError(ValueError):
    """Raised when comparison finalization would violate the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TssOffFinalizationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _finalizer_source_sha256() -> dict[str, str]:
    return {
        **comparator.source_sha256(),
        "experiments/finalize_tss_off_diagnostic_v1.py": comparator._file_sha256(
            FINALIZER_SOURCE_PATH
        ),
    }


def _validate_bound_file(record_raw: Any, label: str) -> None:
    record = _mapping(record_raw, label)
    path = record.get("path")
    digest = record.get("sha256")
    _require(isinstance(path, str) and bool(path), f"{label}.path is invalid")
    _require(
        isinstance(digest, str) and len(digest) == 64,
        f"{label}.sha256 is invalid",
    )
    _require(
        comparator._file_sha256(Path(path)) == digest,
        f"{label} changed after comparison",
    )


def validate_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    comparison = _mapping(payload, "comparison")
    comparator.validate_artifact_sha256(comparison, "comparison")
    for field, expected in (
        ("schema", comparator.COMPARISON_SCHEMA),
        ("status", "complete"),
        ("selection_split", comparator.SELECTION_SPLIT),
        ("test_selected", True),
        ("selection_is_optimistic", True),
        ("independent_test_confirmation", False),
        ("threshold", float(comparator.FIXED_THRESHOLD)),
        ("training_seed", comparator.TRAINING_SEED),
        ("datasets", list(comparator.DATASETS)),
        ("checkpoint_roles", list(comparator.CHECKPOINT_ROLES)),
        ("metrics_used", list(comparator.METRICS)),
        ("requested_tss_weight", 0.0),
        ("old_positive_rank_population_modified", False),
        ("positive_selection_recomputed", False),
        ("descriptive_pd_fa_sweep_used_for_decision", False),
        ("causal_confirmation", False),
        ("stability_claim_supported", False),
        ("final_training_recipe_established", False),
        ("no_fabricated_results", True),
        (
            "claim_scope",
            "fixed_seed42_img_idx_test_selected_paired_diagnostic",
        ),
    ):
        _require(comparison.get(field) == expected, f"comparison {field} differs")

    prior = _mapping(comparison.get("prior_positive_selection"), "prior positive")
    for field, expected in (
        ("decision", comparator.POSITIVE_DECISION),
        ("global_tss_recipe_established", False),
        ("global_tss_lambda", None),
        ("candidate_ranking", []),
        ("descriptive_fewest_violation_anchor", 0.005),
        ("descriptive_anchor_is_selected_candidate", False),
        ("positive_selection_recomputed", False),
        ("prior_positive_conclusion_authoritative", True),
    ):
        _require(prior.get(field) == expected, f"prior positive {field} differs")

    axis_1 = _mapping(
        comparison.get("axis_1_tss_off_vs_original"), "comparison.axis_1"
    )
    eligible = axis_1.get("off_gate_eligible")
    _require(isinstance(eligible, bool), "axis_1.off_gate_eligible must be bool")
    severe = axis_1.get("severe_degradation_violations")
    _require(isinstance(severe, list), "axis_1 severe violations must be a list")
    _require(
        axis_1.get("severe_degradation_violation_count") == len(severe),
        "axis_1 severe violation count differs",
    )
    _require(
        axis_1.get("severe_degradation_passed") is (not severe),
        "axis_1 severe pass flag differs",
    )
    dual = axis_1.get("original_dual_role_dominated_datasets")
    _require(isinstance(dual, list), "axis_1 dual-role dataset list is invalid")
    _require(
        eligible is (not severe and not dual),
        "axis_1 eligibility is not the frozen zero-violation/zero-dual-role gate",
    )
    expected_decision = (
        comparator.ELIGIBLE_DECISION
        if eligible
        else comparator.INELIGIBLE_DECISION
    )
    _require(comparison.get("decision") == expected_decision, "decision differs from axis 1")

    axis_2 = _mapping(
        comparison.get("axis_2_tss_off_vs_positive"), "comparison.axis_2"
    )
    pairwise = _mapping(axis_2.get("pairwise_relations"), "axis_2 pairwise")
    expected_lambda_keys = tuple(
        comparator._lambda_key(value) for value in comparator.CANDIDATE_LAMBDAS
    )
    _require(set(pairwise) == set(expected_lambda_keys), "axis_2 lambdas differ")
    nominal_dimension = (
        len(comparator.DATASETS)
        * len(comparator.CHECKPOINT_ROLES)
        * len(comparator.METRICS)
    )
    _require(
        axis_2.get("nominal_pairwise_vector_dimension") == nominal_dimension,
        "axis_2 nominal dimension differs",
    )
    expected_dimension = axis_2.get("effective_pairwise_vector_dimension")
    _require(
        isinstance(expected_dimension, int)
        and not isinstance(expected_dimension, bool)
        and 0 < expected_dimension <= nominal_dimension,
        "axis_2 effective dimension differs",
    )
    relations: dict[str, str] = {}
    for lambda_value in comparator.CANDIDATE_LAMBDAS:
        lambda_key = comparator._lambda_key(lambda_value)
        relation_record = _mapping(pairwise[lambda_key], f"axis_2[{lambda_key}]")
        relation = relation_record.get("relation")
        _require(relation in comparator.RELATION_BY_SIGNS, "axis_2 relation differs")
        cells = _mapping(relation_record.get("cells"), f"axis_2[{lambda_key}].cells")
        _require(
            relation_record.get("vector_dimension") == expected_dimension
            and len(cells) == expected_dimension,
            f"axis_2[{lambda_key}] must contain exactly {expected_dimension} cells",
        )
        counts = sum(
            int(relation_record.get(field, -expected_dimension))
            for field in (
                "off_better_cell_count",
                "equal_cell_count",
                "off_worse_cell_count",
            )
        )
        _require(counts == expected_dimension, f"axis_2[{lambda_key}] cell counts differ")
        alias = comparator._lambda_field(lambda_value)
        _require(axis_2.get(alias) == relation, f"axis_2 alias {alias} differs")
        relations[lambda_key] = str(relation)
    _require(
        axis_2.get("secondary_axis_changes_axis_1_decision") is False,
        "secondary axis must not change the primary decision",
    )
    _require(
        axis_2.get("all_relations_equal")
        is all(relation == "equal" for relation in relations.values()),
        "axis_2 all-equal flag differs",
    )

    _require(
        comparison.get("fairness_and_search_budget")
        == comparator.fairness_and_search_budget(),
        "comparison search-budget disclosure differs",
    )
    _require(
        comparison.get("source_sha256") == comparator.source_sha256(),
        "comparison source files changed; rerun comparison before finalizing",
    )
    return {
        "decision": expected_decision,
        "eligible": eligible,
        "relations": relations,
    }


def validate_comparison_bindings(
    comparison: Mapping[str, Any],
    *,
    positive_root: Path,
    tss_off_root: Path,
) -> None:
    bindings = _mapping(comparison.get("artifact_bindings"), "artifact_bindings")
    positive = _mapping(bindings.get("positive"), "artifact_bindings.positive")
    _require(
        positive.get("positive_root") == str(Path(positive_root).resolve()),
        "comparison positive root differs from finalizer request",
    )
    _validate_bound_file(positive.get("selector_input"), "bound selector input")
    _validate_bound_file(positive.get("selection"), "bound positive selection")
    launch = _mapping(positive.get("launch_plan"), "bound positive launch plan")
    _validate_bound_file(
        {"path": launch.get("launch_plan_path"), "sha256": launch.get("launch_plan_sha256")},
        "bound positive launch plan",
    )
    _require(
        bindings.get("tss_off_root") == str(Path(tss_off_root).resolve()),
        "comparison TSS-off root differs from finalizer request",
    )
    off_launch = _mapping(
        bindings.get("tss_off_launch_plan"), "bound TSS-off launch plan"
    )
    _validate_bound_file(off_launch, "bound TSS-off launch plan")
    evaluations = _mapping(
        bindings.get("tss_off_evaluations"), "bound TSS-off evaluations"
    )
    _require(set(evaluations) == set(comparator.DATASETS), "bound datasets differ")
    for dataset in comparator.DATASETS:
        roles = _mapping(evaluations[dataset], f"bound evaluations {dataset}")
        _require(set(roles) == set(comparator.CHECKPOINT_ROLES), "bound roles differ")
        for role in comparator.CHECKPOINT_ROLES:
            record = _mapping(roles[role], f"bound evaluation {dataset}/{role}")
            _require(
                record.get("fixed_threshold_field") == "fixed_threshold_0_5"
                and record.get("descriptive_sweep_used_for_decision") is False,
                f"bound evaluation {dataset}/{role} threshold role differs",
            )
            _validate_bound_file(record, f"bound evaluation {dataset}/{role}")


def _diagnostic_labels(relations: Mapping[str, str]) -> dict[str, Any]:
    values = tuple(relations.values())
    labels: list[str] = []
    if any(value == "dominates" for value in values):
        labels.append("TSS_OFF_IMPROVED_PREDECLARED_DIAGNOSTIC_CONTRAST")
    if any(value == "dominated" for value in values):
        labels.append("POSITIVE_TSS_OUTPERFORMED_OFF_IN_PREDECLARED_VECTOR")
    if any(value == "incomparable" for value in values):
        labels.append("MIXED_TRADE_OFF_RETAINED")
    all_equal = all(value == "equal" for value in values)
    if all_equal:
        labels.append("TSS_EFFECT_NOT_IDENTIFIABLE_UNDER_QUANTIZED_ENDPOINTS")
    return {
        "labels": labels,
        "tss_effect_not_identifiable_under_quantized_endpoints": all_equal,
        "tss_harm_confirmed": False,
        "tss_effectiveness_confirmed": False,
    }


def build_final(
    comparison: Mapping[str, Any],
    *,
    comparison_path: Path,
) -> dict[str, Any]:
    validated = validate_comparison(comparison)
    eligible = validated["eligible"]
    if eligible:
        operational = {
            "tss_default_enabled": False,
            "seed42_operational_recipe_admissible": True,
            "component_level_architecture_diagnosis": False,
            "next_action": (
                "seek independent non-test-selected and multi-seed confirmation "
                "before establishing a final training recipe"
            ),
            "next_component_diagnosis_order": [],
        }
    else:
        operational = {
            "tss_default_enabled": None,
            "seed42_operational_recipe_admissible": False,
            "component_level_architecture_diagnosis": True,
            "next_action": "enter frozen single-component architecture diagnosis",
            "next_component_diagnosis_order": [
                "NER relay target/background modulation",
                "QFG per-level utilization and knockout",
                "TPD tiny-target and component-connectivity effects",
            ],
        }
    output = {
        "schema": FINAL_SCHEMA,
        "status": "complete",
        "decision": validated["decision"],
        "positive_global_tss_recipe_established": False,
        "global_tss_lambda": None,
        "positive_tss_search_closed": True,
        "selected_positive_candidate": None,
        "descriptive_fewest_violation_anchor": 0.005,
        "positive_selection_recomputed": False,
        **operational,
        "off_gate_eligible": eligible,
        "axis_1_tss_off_vs_original": comparison["axis_1_tss_off_vs_original"],
        "axis_2_tss_off_vs_positive": comparison["axis_2_tss_off_vs_positive"],
        "diagnostic_interpretation": _diagnostic_labels(validated["relations"]),
        "architecture_global_advantage_not_established": True,
        "architecture_failure_supported": False,
        "final_training_recipe_established": False,
        "causal_confirmation": False,
        "stability_claim_supported": False,
        "test_selected": True,
        "selection_is_optimistic": True,
        "independent_test_confirmation": False,
        "training_seed": comparator.TRAINING_SEED,
        "threshold": float(comparator.FIXED_THRESHOLD),
        "descriptive_pd_fa_sweep_used_for_decision": False,
        "fairness_and_search_budget": comparison["fairness_and_search_budget"],
        "comparison_binding": {
            "path": str(Path(comparison_path).resolve()),
            "file_sha256": comparator._file_sha256(comparison_path),
            "artifact_sha256": comparison["artifact_sha256"],
            "validated": True,
            "metrics_recomputed_during_finalization": False,
        },
        "source_sha256": _finalizer_source_sha256(),
        "no_fabricated_results": True,
        "claim_scope": "fixed_seed42_img_idx_test_selected_paired_diagnostic",
    }
    return comparator.seal_artifact(output)


def build_preflight(
    *,
    positive_root: Path,
    tss_off_root: Path,
    positive_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return comparator.seal_artifact(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": "preflight_complete",
            "decision": "NOT_EVALUATED",
            "finalization_executed": False,
            "comparison_result_loaded": False,
            "result_metrics_loaded": False,
            "results_fabricated": False,
            "positive_selection_recomputed": False,
            "required_comparison_schema": comparator.COMPARISON_SCHEMA,
            "required_comparison_filename": comparator.COMPARISON_FILENAME,
            "roots": {
                "positive_root": str(Path(positive_root).resolve()),
                "tss_off_root": str(Path(tss_off_root).resolve()),
            },
            "positive_artifact_binding": dict(positive_binding),
            "source_sha256": _finalizer_source_sha256(),
            "test_selected": True,
            "selection_is_optimistic": True,
            "independent_test_confirmation": False,
            "causal_confirmation": False,
            "stability_claim_supported": False,
            "no_fabricated_results": True,
            "claim_scope": "fixed_seed42_img_idx_test_selected_paired_diagnostic",
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-root", type=Path, default=comparator.DEFAULT_POSITIVE_ROOT
    )
    parser.add_argument(
        "--tss-off-root", type=Path, default=comparator.DEFAULT_TSS_OFF_ROOT
    )
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--positive-selector-input", type=Path)
    parser.add_argument("--positive-selection", type=Path)
    parser.add_argument("--positive-launch-plan", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    comparison_dir = args.comparison_dir or args.tss_off_root / "comparison"
    output_dir = args.output_dir or args.tss_off_root / "selection"
    (
        _positive_input,
        _normalized_positive,
        _positive_selection,
        positive_binding,
    ) = comparator.load_positive_bundle(
        positive_root=args.positive_root,
        selector_input_path=args.positive_selector_input,
        selection_path=args.positive_selection,
        launch_plan_path=args.positive_launch_plan,
    )
    if args.preflight:
        result = build_preflight(
            positive_root=args.positive_root,
            tss_off_root=args.tss_off_root,
            positive_binding=positive_binding,
        )
        output_path = output_dir / PREFLIGHT_FILENAME
    else:
        comparison_path = comparison_dir / comparator.COMPARISON_FILENAME
        comparison = selector.load_json(comparison_path)
        validate_comparison_bindings(
            comparison,
            positive_root=args.positive_root,
            tss_off_root=args.tss_off_root,
        )
        result = build_final(comparison, comparison_path=comparison_path)
        output_path = output_dir / FINAL_FILENAME
    selector.atomic_write_json(output_path, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
