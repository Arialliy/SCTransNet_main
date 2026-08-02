from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

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
        # These display-only rates must not enter selection.
        "pd": matched / 110,
        "fa": unmatched_pixels / 100_000,
    }


def _payload() -> dict[str, object]:
    original = {
        role: _point()
        for role in selector.CHECKPOINT_ROLES
    }
    candidate_points = {
        "0.0025": _point(
            miou=0.701,
            niou=0.701,
            matched=101,
            unmatched_pixels=90,
            matched_tiny=20,
        ),
        "0.005": _point(
            miou=0.702,
            niou=0.7005,
            matched=102,
            unmatched_pixels=95,
            matched_tiny=21,
        ),
        "0.01": _point(
            miou=0.7005,
            niou=0.7005,
            matched=100,
            unmatched_pixels=99,
            matched_tiny=20,
        ),
    }
    datasets = {}
    for dataset in selector.DATASETS:
        datasets[dataset] = {
            "selection_split": "img_idx/test",
            "img_idx_test_sha256": selector.data_protocol.EXPECTED_SPLITS[
                dataset
            ]["test"]["file_sha256"],
            "img_idx_test_ordered_ids_sha256": (
                selector.data_protocol.EXPECTED_SPLITS[dataset]["test"][
                    "ordered_ids_sha256"
                ]
            ),
            "original": copy.deepcopy(original),
            "final_candidates": {
                lambda_key: {
                    role: copy.deepcopy(point)
                    for role in selector.CHECKPOINT_ROLES
                }
                for lambda_key, point in candidate_points.items()
            },
        }
    return {
        "schema": selector.INPUT_SCHEMA,
        "selection_split": "img_idx/test",
        "test_selected": True,
        "training_seed": 42,
        "threshold": 0.5,
        "checkpoint_roles": ["best_miou", "best_pd"],
        "candidate_lambdas": [0.0025, 0.005, 0.01],
        "datasets": datasets,
    }


def _launch_plan() -> dict[str, object]:
    sources = selector.selector_source_sha256()
    return {
        "schema": selector.LAUNCH_PLAN_SCHEMA,
        "status": "prepared_not_started",
        "dataset_order": list(selector.DATASETS),
        "worker_count": 12,
        "original_run_count": 3,
        "final_run_count": 9,
        "total_search_budget_equal": False,
        "final_to_original_run_budget_ratio": 3.0,
        "static_inputs": {
            "training_sources": {
                "global_recipe_selector": {
                    "path": str(selector.SELECTOR_SOURCE_PATH),
                    "sha256": sources[
                        "experiments/select_three_dataset_global_tss_recipe_v2.py"
                    ],
                },
                "data_protocol": {
                    "path": str(selector.DATA_PROTOCOL_SOURCE_PATH),
                    "sha256": sources[
                        "experiments/three_dataset_v2_protocol.py"
                    ],
                },
            }
        },
    }


class QuantizationTests(unittest.TestCase):
    def test_decimal_half_step_rounds_up_deterministically(self) -> None:
        self.assertEqual(selector.quantize_iou(0.50004), 5000)
        self.assertEqual(selector.quantize_iou(0.50005), 5001)
        self.assertEqual(selector.quantize_iou("0.99995"), 10000)


class SelectorProtocolTests(unittest.TestCase):
    def test_rank_pareto_selects_expected_candidate(self) -> None:
        result = selector.select_global_recipe(_payload())
        self.assertTrue(result["global_tss_recipe_established"])
        self.assertEqual(result["global_tss_lambda"], 0.005)
        self.assertFalse(result["aggregation"]["raw_metric_sum_used"])
        self.assertEqual(
            result["aggregation"]["rank_population"],
            "all_three_preregistered_candidates_before_eligibility_gates",
        )

        weak = result["candidates"]["0.01"]
        self.assertTrue(weak["gate_eligible"])
        self.assertTrue(weak["pareto_dominated"])
        self.assertEqual(
            weak["pareto_dominated_by"],
            ["0.0025", "0.005"],
        )
        self.assertEqual(weak["rank_vector_dimension"], 30)
        self.assertEqual(
            result["aggregation"]["nominal_rank_vector_dimension"], 30
        )
        self.assertEqual(
            len(result["candidates"]["0.005"]["rank_vector"]),
            3 * 2 * 5,
        )

    def test_exact_tie_uses_smaller_positive_lambda(self) -> None:
        payload = _payload()
        for dataset in selector.DATASETS:
            candidates = payload["datasets"][dataset]["final_candidates"]
            common = copy.deepcopy(candidates["0.0025"])
            for lambda_key in ("0.005", "0.01"):
                candidates[lambda_key] = copy.deepcopy(common)
        result = selector.select_global_recipe(payload)
        self.assertEqual(result["global_tss_lambda"], 0.0025)
        self.assertEqual(
            [row["lambda_req"] for row in result["candidate_ranking"]],
            [0.0025, 0.005, 0.01],
        )
        for record in result["candidates"].values():
            self.assertFalse(record["pareto_dominated"])
            self.assertEqual(record["pareto_dominated_by"], [])

    def test_search_budget_disclosure_forbids_total_budget_fairness_claim(self) -> None:
        disclosure = selector.select_global_recipe(_payload())[
            "fairness_and_search_budget"
        ]
        self.assertTrue(disclosure["per_run_protocol_matched"])
        self.assertFalse(disclosure["total_search_budget_matched"])
        self.assertFalse(disclosure["total_search_budget_equal"])
        self.assertEqual(disclosure["original_training_runs"], 3)
        self.assertEqual(disclosure["final_training_runs"], 9)
        self.assertEqual(disclosure["final_to_original_run_budget_ratio"], 3.0)
        self.assertEqual(disclosure["original_run_count"], 3)
        self.assertEqual(disclosure["final_search_run_count"], 9)
        self.assertEqual(disclosure["total_run_count"], 12)
        self.assertIn("equal total", disclosure["prohibited_claim"])

    def test_result_hashes_selector_and_data_protocol_sources(self) -> None:
        result = selector.select_global_recipe(_payload())
        self.assertEqual(result["source_sha256"], selector.selector_source_sha256())
        self.assertEqual(
            set(result["source_sha256"]),
            {
                "experiments/select_three_dataset_global_tss_recipe_v2.py",
                "experiments/three_dataset_v2_protocol.py",
            },
        )
        self.assertFalse(result["launch_plan_binding"]["provided"])
        self.assertFalse(result["launch_plan_binding"]["validated"])

    def test_test_index_identities_are_bound_to_frozen_protocol(self) -> None:
        payload = _payload()
        result = selector.select_global_recipe(payload)
        for dataset in selector.DATASETS:
            expected = selector.data_protocol.EXPECTED_SPLITS[dataset]["test"]
            observed = result["dataset_test_identities"][dataset]
            self.assertEqual(
                observed["img_idx_test_sha256"], expected["file_sha256"]
            )
            self.assertEqual(
                observed["img_idx_test_ordered_ids_sha256"],
                expected["ordered_ids_sha256"],
            )

        for field in (
            "img_idx_test_sha256",
            "img_idx_test_ordered_ids_sha256",
        ):
            tampered = _payload()
            tampered["datasets"]["NUDT-SIRST"][field] = "0" * 64
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    selector.SelectorInputError, "does not match the frozen"
                ):
                    selector.select_global_recipe(tampered)

    def test_iou_severe_drop_uses_frozen_quantized_boundary(self) -> None:
        payload = _payload()
        payload["datasets"]["NUAA-SIRST"]["final_candidates"]["0.0025"][
            "best_miou"
        ]["miou"] = 0.695
        result = selector.select_global_recipe(payload)
        record = result["candidates"]["0.0025"]
        self.assertFalse(record["severe_degradation_passed"])
        self.assertTrue(
            any(
                violation["rule"] == "miou_drop_at_least_0.005"
                for violation in record["severe_degradation_violations"]
            )
        )
        self.assertFalse(record["gate_eligible"])

    def test_fa_guard_uses_integer_counts_and_strict_over_25_percent(self) -> None:
        original = selector._normalize_point(_point(), "original")
        at_boundary = selector._normalize_point(
            _point(unmatched_pixels=125, matched=100), "boundary"
        )
        above_boundary = selector._normalize_point(
            _point(unmatched_pixels=126, matched=101), "above"
        )
        boundary_rules = {
            item["rule"]
            for item in selector._severe_degradation_violations(
                "NUAA-SIRST", "best_miou", original, at_boundary
            )
        }
        above_rules = {
            item["rule"]
            for item in selector._severe_degradation_violations(
                "NUAA-SIRST", "best_miou", original, above_boundary
            )
        }
        self.assertNotIn(
            "fa_increase_over_25_percent_without_2_matched_gain",
            boundary_rules,
        )
        self.assertIn(
            "fa_increase_over_25_percent_without_2_matched_gain",
            above_rules,
        )

    def test_original_dominating_both_roles_on_one_dataset_rejects_candidate(self) -> None:
        payload = _payload()
        for role in selector.CHECKPOINT_ROLES:
            payload["datasets"]["NUAA-SIRST"]["final_candidates"]["0.0025"][
                role
            ] = _point(miou=0.6999)
        result = selector.select_global_recipe(payload)
        record = result["candidates"]["0.0025"]
        self.assertEqual(
            record["original_dual_role_dominated_datasets"],
            ["NUAA-SIRST"],
        )
        self.assertFalse(record["gate_eligible"])

    def test_no_candidate_passes_does_not_force_recipe(self) -> None:
        payload = _payload()
        for lambda_key in ("0.0025", "0.005", "0.01"):
            for role in selector.CHECKPOINT_ROLES:
                payload["datasets"]["NUAA-SIRST"]["final_candidates"][lambda_key][
                    role
                ] = _point(miou=0.6999)
        result = selector.select_global_recipe(payload)
        self.assertFalse(result["global_tss_recipe_established"])
        self.assertIsNone(result["global_tss_lambda"])
        self.assertEqual(
            result["decision"],
            "NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED",
        )
        self.assertEqual(result["candidate_ranking"], [])

    def test_tiny_na_is_omitted_from_every_candidate_rank_vector(self) -> None:
        payload = _payload()
        dataset = payload["datasets"]["IRSTD-1K"]
        points = list(dataset["original"].values())
        for candidate in dataset["final_candidates"].values():
            points.extend(candidate.values())
        for point in points:
            point["tiny_target_count"] = 0
            point["matched_tiny_target_count"] = None
        result = selector.select_global_recipe(payload)
        for candidate in result["candidates"].values():
            self.assertEqual(len(candidate["rank_vector"]), 28)
            self.assertEqual(candidate["rank_vector_dimension"], 28)
            self.assertFalse(
                any(
                    key.startswith("IRSTD-1K/")
                    and key.endswith("/matched_tiny_target_count")
                    for key in candidate["rank_vector"]
                )
            )

    def test_rejects_sirst3_wrong_split_extra_role_and_extra_lambda(self) -> None:
        cases = []

        with_sirst3 = _payload()
        with_sirst3["datasets"]["SIRST3"] = copy.deepcopy(
            with_sirst3["datasets"]["NUAA-SIRST"]
        )
        cases.append(with_sirst3)

        wrong_split = _payload()
        wrong_split["datasets"]["NUDT-SIRST"]["selection_split"] = "custom/test"
        cases.append(wrong_split)

        extra_role = _payload()
        extra_role["checkpoint_roles"].append("best_joint")
        cases.append(extra_role)

        extra_lambda = _payload()
        extra_lambda["candidate_lambdas"].append(0.02)
        cases.append(extra_lambda)

        for payload in cases:
            with self.subTest(case=cases.index(payload)):
                with self.assertRaises(selector.SelectorInputError):
                    selector.select_global_recipe(payload)

    def test_rejects_sweep_threshold_and_zero_tiny_encoded_as_count(self) -> None:
        sweep_point = _payload()
        sweep_point["datasets"]["NUAA-SIRST"]["original"]["best_miou"][
            "threshold"
        ] = 1.0
        with self.assertRaisesRegex(
            selector.SelectorInputError, "Pd--Fa sweep points"
        ):
            selector.select_global_recipe(sweep_point)

        invalid_na = _payload()
        point = invalid_na["datasets"]["NUAA-SIRST"]["original"]["best_miou"]
        point["tiny_target_count"] = 0
        point["matched_tiny_target_count"] = 0
        with self.assertRaisesRegex(selector.SelectorInputError, "must be null"):
            selector.select_global_recipe(invalid_na)

    def test_cli_writes_auditable_json_and_refuses_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            output = root / "selection.json"
            source.write_text(json.dumps(_payload()), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    selector.main(
                        ["--input", str(source), "--output", str(output)]
                    ),
                    0,
                )
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], selector.OUTPUT_SCHEMA)
            self.assertEqual(len(written["input_sha256"]), 64)
            self.assertFalse(written["launch_plan_binding"]["provided"])
            with self.assertRaises(FileExistsError):
                with redirect_stdout(io.StringIO()):
                    selector.main(
                        ["--input", str(source), "--output", str(output)]
                    )

    def test_cli_strictly_binds_prepared_launch_plan_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            plan_path = root / "launch_plan.json"
            output = root / "selection.json"
            source.write_text(json.dumps(_payload()), encoding="utf-8")
            plan_path.write_text(json.dumps(_launch_plan()), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    selector.main(
                        [
                            "--input",
                            str(source),
                            "--output",
                            str(output),
                            "--launch-plan",
                            str(plan_path),
                        ]
                    ),
                    0,
                )
            result = json.loads(output.read_text(encoding="utf-8"))
            binding = result["launch_plan_binding"]
            self.assertTrue(binding["provided"])
            self.assertTrue(binding["validated"])
            self.assertTrue(binding["matches_current_source_sha256"])
            self.assertEqual(
                binding["frozen_selector_sha256"],
                result["source_sha256"][
                    "experiments/select_three_dataset_global_tss_recipe_v2.py"
                ],
            )
            self.assertEqual(
                binding["launch_plan_sha256"],
                selector._file_sha256(plan_path),
            )

            tampered = _launch_plan()
            tampered["static_inputs"]["training_sources"][
                "global_recipe_selector"
            ]["sha256"] = "0" * 64
            plan_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(
                selector.SelectorInputError, "differs from current source"
            ):
                selector.validate_launch_plan_binding(plan_path)


if __name__ == "__main__":
    unittest.main()
