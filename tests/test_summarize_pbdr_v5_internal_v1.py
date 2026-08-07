from __future__ import annotations

from pathlib import Path
import unittest

from experiments.pbdr_v5_internal_selector import FROZEN_FAMILY_ORDER
from experiments.summarize_pbdr_v5_internal_v1 import (
    RUNS,
    V5_RUN_RELATIVE_DIRS,
    _independent_baseline_comparison,
    _v3_selected_metrics,
    _v5_summary_path,
)


class PBDRV5InternalSummaryTests(unittest.TestCase):
    def test_mixed_v5_run_roots_are_explicit_and_complete(self) -> None:
        root = Path("/tmp/repository")
        self.assertEqual(set(V5_RUN_RELATIVE_DIRS), set(RUNS))
        self.assertEqual(
            _v5_summary_path(root, "NUAA-SIRST", "best_miou"),
            root / "results/pbdr_v5_v1/training/NUAA-SIRST/best_miou/summary.json",
        )
        for dataset, role in (
            ("NUDT-SIRST", "best_pd"),
            ("IRSTD-1K", "best_miou"),
        ):
            self.assertEqual(
                _v5_summary_path(root, dataset, role),
                root
                / "results/pbdr_v5_v1/training_idle_gpu"
                / dataset
                / role
                / "summary.json",
            )

    def test_v3_uses_named_selected_candidate_not_anchor(self) -> None:
        selected_metrics = {"intersection_pixels": 101}
        sweep = {
            "selected": {"name": "selected-grid"},
            "candidates": [
                {"name": "anchor", "metrics": {"intersection_pixels": 999}},
                {"name": "selected-grid", "metrics": selected_metrics},
            ],
        }
        self.assertEqual(_v3_selected_metrics(sweep), selected_metrics)

    def test_frozen_order_and_baseline_comparison_boundary(self) -> None:
        self.assertEqual(
            FROZEN_FAMILY_ORDER,
            (
                "Original",
                "Current",
                "V3-calibrated",
                "V4-Stage1",
                "V4-Stage2",
                "V5",
            ),
        )
        diagnosis = {
            "user_supplied_independent_baseline": {
                "datasets": {dataset: {} for dataset, _role in RUNS}
            },
            "designed_current_vs_user_baseline_best_miou": {
                "comparison_scope": "best_miou_checkpoint_to_best_miou_checkpoint",
                **{
                    dataset: {
                        "designed_current": {},
                        "delta_designed_minus_baseline": {},
                    }
                    for dataset, _role in RUNS
                },
            },
        }
        baseline, comparison = _independent_baseline_comparison(diagnosis)
        self.assertIn("datasets", baseline)
        self.assertEqual(
            comparison["comparison_scope"],
            "best_miou_checkpoint_to_best_miou_checkpoint",
        )
        self.assertNotIn("V5", comparison)


if __name__ == "__main__":
    unittest.main()
