from __future__ import annotations

import ast
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from experiments import compare_tss_off_positive_original_v1 as comparator
from experiments import finalize_tss_off_diagnostic_v1 as finalizer
from experiments import select_three_dataset_global_tss_recipe_v2 as selector


def _point(
    *,
    miou: float = 0.7,
    niou: float = 0.7,
    matched: int = 100,
    unmatched_pixels: int = 100,
    matched_tiny: int | None = 20,
    tiny_count: int = 30,
) -> dict[str, object]:
    return {
        "threshold": 0.5,
        "miou": miou,
        "niou": niou,
        "target_count": 110,
        "matched_target_count": matched,
        "tiny_target_count": tiny_count,
        "matched_tiny_target_count": matched_tiny,
        "unmatched_predicted_pixels": unmatched_pixels,
        "valid_pixel_count": 100_000,
        # Display rates and sweeps must never be decision inputs.
        "pd": matched / 110,
        "fa": unmatched_pixels / 100_000,
    }


def _positive_payload() -> dict[str, object]:
    original = {role: _point() for role in selector.CHECKPOINT_ROLES}
    candidates = {
        "0.0025": _point(
            miou=0.6999,
            niou=0.6999,
            matched=99,
            unmatched_pixels=101,
            matched_tiny=19,
        ),
        "0.005": _point(),
        "0.01": _point(
            miou=0.7001,
            niou=0.7001,
            matched=101,
            unmatched_pixels=99,
            matched_tiny=21,
        ),
    }
    datasets: dict[str, object] = {}
    for dataset in selector.DATASETS:
        frozen = selector.data_protocol.EXPECTED_SPLITS[dataset]["test"]
        datasets[dataset] = {
            "selection_split": selector.SELECTION_SPLIT,
            "img_idx_test_sha256": frozen["file_sha256"],
            "img_idx_test_ordered_ids_sha256": frozen["ordered_ids_sha256"],
            "original": copy.deepcopy(original),
            "final_candidates": {
                key: {
                    role: copy.deepcopy(point)
                    for role in selector.CHECKPOINT_ROLES
                }
                for key, point in candidates.items()
            },
        }
    return {
        "schema": selector.INPUT_SCHEMA,
        "selection_split": selector.SELECTION_SPLIT,
        "test_selected": True,
        "training_seed": selector.TRAINING_SEED,
        "threshold": 0.5,
        "checkpoint_roles": list(selector.CHECKPOINT_ROLES),
        "candidate_lambdas": [0.0025, 0.005, 0.01],
        "datasets": datasets,
    }


def _off_payload(positive: dict[str, object]) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for dataset in selector.DATASETS:
        positive_dataset = positive["datasets"][dataset]
        datasets[dataset] = {
            "selection_split": selector.SELECTION_SPLIT,
            "img_idx_test_sha256": positive_dataset["img_idx_test_sha256"],
            "img_idx_test_ordered_ids_sha256": positive_dataset[
                "img_idx_test_ordered_ids_sha256"
            ],
            "tss_off": copy.deepcopy(positive_dataset["original"]),
        }
    return {
        "schema": comparator.COMPARISON_INPUT_SCHEMA,
        "selection_split": selector.SELECTION_SPLIT,
        "test_selected": True,
        "selection_is_optimistic": True,
        "independent_test_confirmation": False,
        "training_seed": selector.TRAINING_SEED,
        "threshold": 0.5,
        "checkpoint_roles": list(selector.CHECKPOINT_ROLES),
        "requested_tss_weight": 0.0,
        "datasets": datasets,
    }


def _official_bundle() -> tuple[dict[str, object], dict[str, object]]:
    root = comparator.DEFAULT_POSITIVE_ROOT / "selection"
    return (
        selector.load_json(root / "selector_input_v2.json"),
        selector.load_json(root / "global_tss_recipe_selection_v2.json"),
    )


def _evaluation(
    *,
    dataset: str,
    role: str,
    positive_dataset: dict[str, object],
    fixed_point: dict[str, object],
    sweep_sentinel: int,
) -> dict[str, object]:
    return {
        "schema": comparator.EVALUATION_SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": comparator.OFF_METHOD,
        "requested_tss_weight": 0.0,
        "checkpoint_role": role,
        "seed": 42,
        "test_selected": True,
        "selection_is_optimistic": True,
        "threshold_roles": {
            "checkpoint_selection_threshold": 0.5,
            "global_lambda_selection_threshold": 0.5,
            "main_table_threshold": 0.5,
            "descriptive_sweep_only": True,
        },
        "fixed_threshold_0_5": fixed_point,
        "descriptive_pd_fa": {
            "selection_effect": "none",
            "points": [{"threshold": 1.0, "sentinel": sweep_sentinel}],
        },
        "data": {
            "split": "img_idx/test",
            "img_idx_test_sha256": positive_dataset["img_idx_test_sha256"],
            "img_idx_test_ordered_ids_sha256": positive_dataset[
                "img_idx_test_ordered_ids_sha256"
            ],
            "sirst3_in_formal_matrix": False,
        },
        "source_sha256": {"test/evaluator.py": "0" * 64},
        "no_fabricated_results": True,
        "stability_claim_supported": False,
    }


def _off_launch_plan(tss_off_root: Path) -> dict[str, object]:
    source_paths = {
        "comparator": comparator.COMPARATOR_SOURCE_PATH,
        "finalizer": finalizer.FINALIZER_SOURCE_PATH,
        "data_protocol": selector.DATA_PROTOCOL_SOURCE_PATH,
    }
    return {
        "schema": "sctransnet_three_dataset_tss_off_launcher_v1/v1",
        "status": "prepared_not_started",
        "dataset_order": list(selector.DATASETS),
        "worker_count": 3,
        "workers": [
            {
                "dataset": dataset,
                "method": "final_tss_off",
                "requested_tss_weight": 0.0,
                "seed": 42,
                "threshold": 0.5,
                "checkpoint_roles": list(selector.CHECKPOINT_ROLES),
                "run_directory": str(
                    comparator._expected_run_directory(tss_off_root, dataset)
                ),
            }
            for dataset in selector.DATASETS
        ],
        "static_inputs": {
            "results_root": str(tss_off_root.resolve()),
            "sources": {
                key: {
                    "path": str(path.resolve()),
                    "sha256": comparator._file_sha256(path),
                }
                for key, path in source_paths.items()
            },
        },
    }


class TssOffComparatorTests(unittest.TestCase):
    def test_identical_to_original_is_gate_eligible_and_has_30_cells(self) -> None:
        positive = _positive_payload()
        result = comparator.compare_inputs(positive, _off_payload(positive))
        axis_1 = result["axis_1_tss_off_vs_original"]
        axis_2 = result["axis_2_tss_off_vs_positive"]
        self.assertTrue(axis_1["off_gate_eligible"])
        self.assertEqual(axis_1["severe_degradation_violations"], [])
        self.assertEqual(axis_1["original_dual_role_dominated_datasets"], [])
        self.assertEqual(result["decision"], comparator.ELIGIBLE_DECISION)
        self.assertEqual(axis_2["nominal_pairwise_vector_dimension"], 30)
        self.assertEqual(axis_2["effective_pairwise_vector_dimension"], 30)
        for record in axis_2["pairwise_relations"].values():
            self.assertEqual(record["vector_dimension"], 30)
            self.assertEqual(len(record["cells"]), 30)

    def test_axis_2_reports_dominates_equal_and_dominated(self) -> None:
        positive = _positive_payload()
        axis_2 = comparator.compare_inputs(positive, _off_payload(positive))[
            "axis_2_tss_off_vs_positive"
        ]
        self.assertEqual(axis_2["off_vs_0p0025"], "dominates")
        self.assertEqual(axis_2["off_vs_0p005"], "equal")
        self.assertEqual(axis_2["off_vs_0p01"], "dominated")
        # The secondary result cannot promote or reject Axis 1.
        self.assertFalse(axis_2["secondary_axis_changes_axis_1_decision"])

    def test_axis_2_reports_incomparable_without_forced_average(self) -> None:
        positive = _positive_payload()
        for dataset in selector.DATASETS:
            for role in selector.CHECKPOINT_ROLES:
                point = positive["datasets"][dataset]["final_candidates"]["0.01"][role]
                point.update(_point(miou=0.7001, unmatched_pixels=101))
        axis_2 = comparator.compare_inputs(positive, _off_payload(positive))[
            "axis_2_tss_off_vs_positive"
        ]
        relation = axis_2["pairwise_relations"]["0.01"]
        self.assertEqual(relation["relation"], "incomparable")
        self.assertGreater(relation["off_better_cell_count"], 0)
        self.assertGreater(relation["off_worse_cell_count"], 0)
        self.assertNotIn("score", relation)
        self.assertNotIn("mean", relation)

    def test_matched_drop_of_two_is_a_severe_rejection(self) -> None:
        positive = _positive_payload()
        off = _off_payload(positive)
        off["datasets"]["NUAA-SIRST"]["tss_off"]["best_miou"][
            "matched_target_count"
        ] = 98
        result = comparator.compare_inputs(positive, off)
        axis_1 = result["axis_1_tss_off_vs_original"]
        self.assertFalse(axis_1["off_gate_eligible"])
        self.assertIn(
            "matched_target_drop_at_least_2",
            {item["rule"] for item in axis_1["severe_degradation_violations"]},
        )
        self.assertEqual(result["decision"], comparator.INELIGIBLE_DECISION)

    def test_original_dual_role_strict_dominance_rejects_without_severe_drop(self) -> None:
        positive = _positive_payload()
        off = _off_payload(positive)
        for role in selector.CHECKPOINT_ROLES:
            off["datasets"]["NUDT-SIRST"]["tss_off"][role]["miou"] = 0.6999
        result = comparator.compare_inputs(positive, off)
        axis_1 = result["axis_1_tss_off_vs_original"]
        self.assertEqual(axis_1["severe_degradation_violations"], [])
        self.assertEqual(
            axis_1["original_dual_role_dominated_datasets"], ["NUDT-SIRST"]
        )
        self.assertFalse(axis_1["off_gate_eligible"])

    def test_iou_quantization_and_fa_integer_boundary_match_old_selector(self) -> None:
        positive = _positive_payload()
        off = _off_payload(positive)
        # Both values quantize to 7000, so the raw display difference is ignored.
        off["datasets"]["NUAA-SIRST"]["tss_off"]["best_miou"]["miou"] = 0.70004
        positive["datasets"]["NUAA-SIRST"]["final_candidates"]["0.005"][
            "best_miou"
        ]["miou"] = 0.70000
        cells = comparator.compare_inputs(positive, off)[
            "axis_2_tss_off_vs_positive"
        ]["pairwise_relations"]["0.005"]["cells"]
        self.assertEqual(
            cells["NUAA-SIRST/best_miou/miou"][
                "comparison_from_off_perspective"
            ],
            0,
        )

        original = selector._normalize_point(_point(), "original")
        boundary = selector._normalize_point(
            _point(unmatched_pixels=125), "boundary"
        )
        above = selector._normalize_point(
            _point(unmatched_pixels=126, matched=101), "above"
        )
        self.assertFalse(
            any(
                item["rule"].startswith("fa_increase")
                for item in selector._severe_degradation_violations(
                    "NUAA-SIRST", "best_miou", original, boundary
                )
            )
        )
        self.assertTrue(
            any(
                item["rule"].startswith("fa_increase")
                for item in selector._severe_degradation_violations(
                    "NUAA-SIRST", "best_miou", original, above
                )
            )
        )

    def test_old_unavailable_tiny_policy_yields_28_effective_cells(self) -> None:
        positive = _positive_payload()
        off = _off_payload(positive)
        dataset = positive["datasets"]["IRSTD-1K"]
        points = list(dataset["original"].values())
        for candidate in dataset["final_candidates"].values():
            points.extend(candidate.values())
        points.extend(off["datasets"]["IRSTD-1K"]["tss_off"].values())
        for point in points:
            point["tiny_target_count"] = 0
            point["matched_tiny_target_count"] = None
        axis_2 = comparator.compare_inputs(positive, off)[
            "axis_2_tss_off_vs_positive"
        ]
        self.assertEqual(axis_2["nominal_pairwise_vector_dimension"], 30)
        self.assertEqual(axis_2["effective_pairwise_vector_dimension"], 28)
        for record in axis_2["pairwise_relations"].values():
            self.assertEqual(record["vector_dimension"], 28)
            self.assertFalse(
                any(
                    key.startswith("IRSTD-1K/")
                    and key.endswith("/matched_tiny_target_count")
                    for key in record["cells"]
                )
            )

    def test_schema_rejects_extra_dataset_wrong_threshold_weight_and_float_count(self) -> None:
        positive = _positive_payload()
        cases: list[dict[str, object]] = []
        extra_dataset = _off_payload(positive)
        extra_dataset["datasets"]["SIRST3"] = copy.deepcopy(
            extra_dataset["datasets"]["NUAA-SIRST"]
        )
        cases.append(extra_dataset)
        wrong_threshold = _off_payload(positive)
        wrong_threshold["threshold"] = 0.6
        cases.append(wrong_threshold)
        wrong_weight = _off_payload(positive)
        wrong_weight["requested_tss_weight"] = 0.0025
        cases.append(wrong_weight)
        float_count = _off_payload(positive)
        float_count["datasets"]["NUAA-SIRST"]["tss_off"]["best_pd"][
            "matched_target_count"
        ] = 100.0
        cases.append(float_count)
        for case in cases:
            with self.subTest(case=cases.index(case)):
                with self.assertRaises(ValueError):
                    comparator.compare_inputs(positive, case)

    def test_descriptive_sweep_is_ignored_by_evaluation_extraction(self) -> None:
        positive = _positive_payload()
        normalized = selector.validate_input(positive)
        dataset = "NUAA-SIRST"
        role = "best_miou"
        positive_dataset = normalized["datasets"][dataset]
        first = _evaluation(
            dataset=dataset,
            role=role,
            positive_dataset=positive_dataset,
            fixed_point=_point(),
            sweep_sentinel=1,
        )
        second = copy.deepcopy(first)
        second["descriptive_pd_fa"]["points"][0]["sentinel"] = 999999
        self.assertEqual(
            comparator._validate_evaluation(
                first,
                dataset=dataset,
                role=role,
                positive_dataset=positive_dataset,
            ),
            comparator._validate_evaluation(
                second,
                dataset=dataset,
                role=role,
                positive_dataset=positive_dataset,
            ),
        )

    def test_budget_discloses_four_to_one_recipe_search(self) -> None:
        budget = comparator.fairness_and_search_budget()
        self.assertEqual(budget["original_training_runs"], 3)
        self.assertEqual(budget["positive_tss_training_runs"], 9)
        self.assertEqual(budget["tss_off_training_runs"], 3)
        self.assertEqual(budget["final_family_training_runs"], 12)
        self.assertEqual(budget["total_training_runs"], 15)
        self.assertEqual(budget["final_to_original_recipe_search_ratio"], 4.0)
        self.assertFalse(budget["total_recipe_search_budget_equal"])
        self.assertTrue(budget["tss_off_added_after_positive_test_results"])

    def test_launch_plan_binds_isolated_runs_and_posttraining_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "off"
            path = Path(directory) / "launch_plan.json"
            path.write_text(json.dumps(_off_launch_plan(root)), encoding="utf-8")
            binding = comparator.validate_tss_off_launch_plan(
                path, tss_off_root=root
            )
            self.assertTrue(binding["validated"])
            self.assertEqual(binding["worker_count"], 3)
            self.assertEqual(
                set(binding["frozen_posttraining_source_sha256"]),
                {"comparator", "finalizer", "data_protocol"},
            )

            tampered = _off_launch_plan(root)
            tampered["static_inputs"]["sources"]["comparator"]["sha256"] = "0" * 64
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comparator SHA-256"):
                comparator.validate_tss_off_launch_plan(path, tss_off_root=root)

    def test_official_positive_conclusion_is_validated_but_not_recomputed(self) -> None:
        positive, stored_selection = _official_bundle()
        closure = comparator.validate_positive_selection(stored_selection, positive)
        self.assertEqual(closure["decision"], comparator.POSITIVE_DECISION)
        self.assertFalse(closure["global_tss_recipe_established"])
        self.assertIsNone(closure["global_tss_lambda"])
        self.assertEqual(closure["candidate_ranking"], [])
        self.assertEqual(closure["descriptive_fewest_violation_anchor"], 0.005)
        self.assertFalse(closure["descriptive_anchor_is_selected_candidate"])
        self.assertFalse(closure["positive_selection_recomputed"])
        self.assertTrue(closure["prior_positive_conclusion_authoritative"])

    def test_preflight_has_no_result_metrics_or_placeholder_decision(self) -> None:
        result = comparator.build_preflight(
            positive_root=comparator.DEFAULT_POSITIVE_ROOT,
            positive_binding={"validated": True},
            tss_off_root=comparator.DEFAULT_TSS_OFF_ROOT,
            launch_plan_binding=None,
        )
        self.assertEqual(result["decision"], "NOT_EVALUATED")
        self.assertFalse(result["comparison_executed"])
        self.assertFalse(result["tss_off_result_metrics_loaded"])
        self.assertNotIn("axis_1_tss_off_vs_original", result)
        self.assertNotIn("axis_2_tss_off_vs_positive", result)
        self.assertNotIn("off_gate_eligible", result)
        self.assertTrue(result["no_fabricated_results"])
        comparator.validate_artifact_sha256(result, "preflight")

    def test_real_cli_preflight_writes_atomically_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    comparator.main(
                        ["--preflight", "--output-dir", str(output_dir)]
                    ),
                    0,
                )
            path = output_dir / comparator.PREFLIGHT_FILENAME
            result = selector.load_json(path)
            self.assertEqual(result["decision"], "NOT_EVALUATED")
            self.assertFalse(result["positive_selection_recomputed"])
            with self.assertRaises(FileExistsError):
                with redirect_stdout(io.StringIO()):
                    comparator.main(
                        ["--preflight", "--output-dir", str(output_dir)]
                    )

    def test_comparator_source_has_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(comparator.COMPARATOR_SOURCE_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            self.assertEqual(
                len(keys),
                len(set(keys)),
                f"duplicate literal dictionary key near line {node.lineno}",
            )


class TssOffFinalizerTests(unittest.TestCase):
    def _comparison(self, *, eligible: bool) -> dict[str, object]:
        positive, stored_selection = _official_bundle()
        off = _off_payload(positive)
        if not eligible:
            dataset = "NUAA-SIRST"
            role = "best_miou"
            original_matched = positive["datasets"][dataset]["original"][role][
                "matched_target_count"
            ]
            off["datasets"][dataset]["tss_off"][role][
                "matched_target_count"
            ] = original_matched - 2
        return comparator.build_comparison(
            positive_input=positive,
            positive_selection=stored_selection,
            tss_off_input=off,
            positive_binding={"test": True},
            tss_off_root=comparator.DEFAULT_TSS_OFF_ROOT,
            launch_plan_binding={"test": True},
            evaluation_bindings={"test": True},
        )

    def test_finalizer_preserves_axis_1_and_never_claims_causality(self) -> None:
        for eligible in (True, False):
            with self.subTest(eligible=eligible), tempfile.TemporaryDirectory() as directory:
                comparison = self._comparison(eligible=eligible)
                path = Path(directory) / comparator.COMPARISON_FILENAME
                path.write_text(json.dumps(comparison), encoding="utf-8")
                result = finalizer.build_final(comparison, comparison_path=path)
                self.assertEqual(result["off_gate_eligible"], eligible)
                self.assertEqual(
                    result["decision"],
                    comparator.ELIGIBLE_DECISION
                    if eligible
                    else comparator.INELIGIBLE_DECISION,
                )
                self.assertFalse(result["final_training_recipe_established"])
                self.assertFalse(result["causal_confirmation"])
                self.assertFalse(result["stability_claim_supported"])
                self.assertFalse(result["positive_selection_recomputed"])
                self.assertFalse(
                    result["diagnostic_interpretation"]["tss_harm_confirmed"]
                )
                comparator.validate_artifact_sha256(result, "final")

    def test_finalizer_rejects_tampered_comparison(self) -> None:
        comparison = self._comparison(eligible=True)
        comparison["decision"] = comparator.INELIGIBLE_DECISION
        with self.assertRaisesRegex(ValueError, "artifact_sha256"):
            finalizer.validate_comparison(comparison)


if __name__ == "__main__":
    unittest.main()
