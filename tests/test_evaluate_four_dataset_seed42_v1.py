from __future__ import annotations

import unittest

import numpy as np

from experiments import evaluate_four_dataset_seed42_v1 as evaluator
from experiments import four_dataset_evaluation_protocol_v1 as protocol


class FourDatasetEvaluatorCoreTests(unittest.TestCase):
    @staticmethod
    def _audit_metrics() -> dict[str, float | int]:
        return {
            "test_loss": 0.1,
            "miou": 0.5,
            "niou": 0.6,
            "pixel_f1": 0.7,
            "pixel_precision": 0.8,
            "pixel_recall": 0.65,
            "pd": 1.0,
            "tiny_pd": 1.0,
            "fa": 0.0,
            "false_objects_per_image": 0.0,
            "target_count": 1,
            "matched_target_count": 1,
            "tiny_target_count": 1,
            "matched_tiny_target_count": 1,
            "predicted_object_count": 1,
            "unmatched_predicted_object_count": 0,
            "valid_pixel_count": 256,
        }

    def test_checkpoint_audit_allows_only_small_continuous_pixel_drift(
        self,
    ) -> None:
        expected = self._audit_metrics()
        observed = dict(expected)
        for key in (
            "miou",
            "niou",
            "pixel_f1",
            "pixel_precision",
            "pixel_recall",
        ):
            observed[key] = float(observed[key]) + 5e-5
        audit = evaluator._checkpoint_metric_audit(
            {"epoch": 10, "test_metrics": expected},
            observed,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["numeric_absolute_tolerances"]["miou"], 1e-4)
        self.assertEqual(audit["numeric_absolute_tolerances"]["pd"], 1e-15)

    def test_checkpoint_audit_still_rejects_raw_count_drift(self) -> None:
        expected = self._audit_metrics()
        observed = dict(expected)
        observed["matched_target_count"] = 0
        with self.assertRaisesRegex(ValueError, "raw count differs"):
            evaluator._checkpoint_metric_audit(
                {"epoch": 10, "test_metrics": expected},
                observed,
            )

    def test_output_paths_separate_dataset_and_sirst3_source_results(self) -> None:
        dataset_request = evaluator.EvaluationRequest(
            "NUAA-SIRST",
            "NUAA-SIRST",
            "original",
            "best_miou",
        )
        source_request = evaluator.EvaluationRequest(
            "SIRST3",
            "NUAA-SIRST",
            "original",
            "best_miou",
        )
        dataset_path = evaluator.output_path_for_request(
            dataset_request,
            results_root=protocol.EXPERIMENT_ROOT,
            sweep=False,
        )
        source_path = evaluator.output_path_for_request(
            source_request,
            results_root=protocol.EXPERIMENT_ROOT,
            sweep=False,
        )
        self.assertIn("fixed_0_5", dataset_path.parts)
        self.assertIn("sirst3_three_sources", source_path.parts)
        self.assertNotEqual(dataset_path, source_path)

    def test_strict_metrics_report_raw_counts_f1_and_tiny_pd(self) -> None:
        target = np.zeros((16, 16), dtype=np.float32)
        target[2:4, 2:4] = 1.0
        target[10:14, 10:14] = 1.0
        probability = np.zeros_like(target)
        probability[2:4, 2:4] = 0.9
        probability[10:14, 10:14] = 0.9
        point = protocol.strict_metric_points(
            [probability],
            [target],
            [0.1],
            [0.5],
        )[0]
        self.assertEqual(point["target_count"], 2)
        self.assertEqual(point["matched_target_count"], 2)
        self.assertEqual(point["tiny_target_count"], 1)
        self.assertEqual(point["matched_tiny_target_count"], 1)
        self.assertEqual(point["pd"], 1.0)
        self.assertEqual(point["tiny_pd"], 1.0)
        self.assertEqual(point["pixel_f1"], 1.0)

    def test_one_prediction_cannot_match_two_targets(self) -> None:
        target = np.zeros((16, 16), dtype=np.float32)
        target[5, 5] = 1.0
        target[5, 8] = 1.0
        probability = np.zeros_like(target)
        probability[5, 6:8] = 0.9
        point = protocol.strict_metric_points(
            [probability],
            [target],
            [0.1],
            [0.5],
        )[0]
        self.assertEqual(point["target_count"], 2)
        self.assertEqual(point["predicted_object_count"], 1)
        self.assertEqual(point["matched_target_count"], 1)
        self.assertEqual(point["pd"], 0.5)

    def test_first_preregistered_fa_budget_is_present(self) -> None:
        self.assertEqual(protocol.FA_BUDGETS[0], 0.5e-6)
        points = [
            {
                "threshold": 0.5,
                "fa": 0.4e-6,
                "pd": 0.5,
                "tiny_pd": 0.5,
                "miou": 0.5,
                "niou": 0.5,
            }
        ]
        selected = protocol.fa_budget_points(points)
        self.assertIn("5e-07", selected)
        self.assertEqual(selected["5e-07"]["pd"], 0.5)


if __name__ == "__main__":
    unittest.main()
