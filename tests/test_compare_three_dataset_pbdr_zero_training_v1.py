from __future__ import annotations

import copy
import unittest

from analysis import analyze_three_dataset_pbdr_zero_training_v1 as analyzer
from analysis import compare_three_dataset_pbdr_zero_training_v1 as subject


def _metrics() -> dict[str, int | float]:
    return {
        "matched_target_count": 100,
        "matched_tiny_target_count": 20,
        "miou": 0.8,
        "niou": 0.75,
        "unmatched_predicted_pixels": 100,
        "component_false_positive_pixels": 100,
        "background_false_positive_pixels": 200,
    }


def _payload() -> dict[str, object]:
    modes = {}
    for g_eighths in analyzer.G_EIGHTHS_ORDER:
        modes[analyzer.MODE_BY_G_EIGHTHS[g_eighths]] = {
            "g_eighths": g_eighths,
            "fixed_threshold_0_5": copy.deepcopy(_metrics()),
        }
    return {
        "modes": modes,
        "signals": {
            "target_rescue": {
                "missed_gt_objects_with_protected_rescue_pixels": 1,
            },
            "background_suppression": {
                "unmatched_fp_pixels_with_unprotected_suppression": 1,
            },
        },
        "protection": {
            "protected_background_pixel_count": 49,
            "background_pixel_count": 100,
            "protected_background_fraction": 0.49,
        },
    }


def _payloads() -> dict[str, dict[str, object]]:
    return {
        subject.role_key(dataset, role): _payload()
        for dataset in subject.DATASETS
        for role in subject.CHECKPOINT_ROLES
    }


class PBDRZeroTrainingComparatorTests(unittest.TestCase):
    def test_t1_requires_two_strict_core_improvements(self) -> None:
        current = _metrics()
        candidate = copy.deepcopy(current)
        candidate["unmatched_predicted_pixels"] = 99
        candidate["miou"] = 0.8 + 2e-12
        result = subject.evaluate_t1_dataset(candidate, current)
        self.assertTrue(result["pass"])
        self.assertEqual(result["strict_core_improvement_count"], 2)

        one_only = copy.deepcopy(current)
        one_only["unmatched_predicted_pixels"] = 99
        one_only["miou"] = 0.8 + 1e-12
        result = subject.evaluate_t1_dataset(one_only, current)
        self.assertFalse(result["pass"])
        self.assertEqual(result["strict_core_improvement_count"], 1)

    def test_t1_nondegradation_accepts_minus_atol(self) -> None:
        current = _metrics()
        candidate = copy.deepcopy(current)
        candidate["matched_target_count"] = 101
        candidate["unmatched_predicted_pixels"] = 99
        candidate["miou"] = 0.8 - subject.FLOAT_EQ_ATOL
        candidate["niou"] = 0.75 - subject.FLOAT_EQ_ATOL
        self.assertTrue(subject.evaluate_t1_dataset(candidate, current)["pass"])

        candidate["miou"] = 0.8 - 1.01 * subject.FLOAT_EQ_ATOL
        self.assertFalse(subject.evaluate_t1_dataset(candidate, current)["pass"])

    def test_t2_exact_severe_boundaries_and_zero_reference(self) -> None:
        payloads = _payloads()
        first = payloads[subject.role_key(subject.DATASETS[0], "best_miou")]
        candidate = first["modes"][analyzer.MODE_BY_G_EIGHTHS[1]][
            "fixed_threshold_0_5"
        ]
        candidate["matched_target_count"] = 98
        result = subject.evaluate_t2(payloads, 1)
        self.assertEqual(result["severe_role_count"], 1)
        self.assertIn(
            "delta_target_le_minus_2",
            result["role_rows"][0]["true_severe_conditions"],
        )

        candidate["matched_target_count"] = 100
        candidate["component_false_positive_pixels"] = 125
        result = subject.evaluate_t2(payloads, 1)
        self.assertEqual(result["severe_role_count"], 1)
        self.assertIn(
            "component_fp_increase_ge_25pct",
            result["role_rows"][0]["true_severe_conditions"],
        )

        current = first["modes"][analyzer.CURRENT_MODE]["fixed_threshold_0_5"]
        current["component_false_positive_pixels"] = 0
        candidate["component_false_positive_pixels"] = 1
        result = subject.evaluate_t2(payloads, 1)
        self.assertEqual(result["severe_role_count"], 1)

    def test_t5_is_strictly_below_one_half(self) -> None:
        payloads = _payloads()
        key = subject.role_key(subject.DATASETS[0], "best_miou")
        payloads[key]["protection"].update(
            {
                "protected_background_pixel_count": 50,
                "protected_background_fraction": 0.5,
            }
        )
        result = subject.evaluate_t5(payloads)
        self.assertFalse(result["pass"])
        self.assertEqual(result["passed_role_count"], 5)

    def test_oracle_is_reported_but_never_authorization_eligible(self) -> None:
        payloads = _payloads()
        t3 = subject.evaluate_t3(payloads)
        t4 = subject.evaluate_t4(payloads)
        t5 = subject.evaluate_t5(payloads)
        oracle = subject.evaluate_gate(
            payloads,
            analyzer.ORACLE_G_EIGHTHS,
            t3=t3,
            t4=t4,
            t5=t5,
        )
        self.assertFalse(oracle["authorization_eligible"])


if __name__ == "__main__":
    unittest.main()
