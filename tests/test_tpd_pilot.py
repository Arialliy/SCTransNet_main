from __future__ import annotations

import unittest

import numpy as np

from experiments.evaluate_pd_fa_sweep import (
    adaptive_thresholds,
    best_point_under_fa,
    threshold_grid,
)
from experiments.train_tpd_pilot import (
    MaskStats,
    ValidationMetrics,
    learning_rate_for_epoch,
    pd_selection_key,
    stratified_split,
)


def synthetic_stats(identifier: str, stratum: str) -> MaskStats:
    return MaskStats(identifier, 256, 256, 1, 4, 1, 4, stratum)


class TPDPilotTests(unittest.TestCase):
    def test_stratified_split_is_exact_disjoint_and_deterministic(self) -> None:
        stats = [synthetic_stats(f"tiny_{index}", "tiny_single") for index in range(20)]
        stats += [synthetic_stats(f"large_{index}", "larger") for index in range(10)]
        train_a, val_a = stratified_split(stats, val_fraction=0.2, split_seed=7)
        train_b, val_b = stratified_split(stats, val_fraction=0.2, split_seed=7)
        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual(len(train_a), 24)
        self.assertEqual(len(val_a), 6)
        self.assertFalse(set(train_a) & set(val_a))
        self.assertEqual(set(train_a) | set(val_a), {item.identifier for item in stats})

    def test_matching_maximizes_cardinality_before_distance(self) -> None:
        # Greedy distance sorting can match (10,10)->(10,12) first and leave
        # (10,14) unmatched. Maximum-cardinality matching finds both pairs.
        target = np.zeros((24, 24), dtype=np.float32)
        probability = np.zeros_like(target)
        target[10, 10] = 1.0
        target[10, 14] = 1.0
        probability[10, 8] = 1.0
        probability[10, 12] = 1.0
        metrics = ValidationMetrics(threshold=0.5, match_radius=3.0, tiny_area=9)
        metrics.update(probability, target, loss=0.0)
        result = metrics.compute()
        self.assertEqual(result["matched_target_count"], 2)
        self.assertEqual(result["matched_tiny_target_count"], 2)
        self.assertEqual(result["pd"], 1.0)
        self.assertEqual(result["tiny_pd"], 1.0)
        self.assertEqual(result["fa"], 0.0)

    def test_learning_rate_schedule_reaches_both_endpoints(self) -> None:
        self.assertAlmostEqual(learning_rate_for_epoch(1, 100, 1e-3, 1e-5, 10), 1e-4)
        self.assertAlmostEqual(learning_rate_for_epoch(10, 100, 1e-3, 1e-5, 10), 1e-3)
        self.assertAlmostEqual(learning_rate_for_epoch(100, 100, 1e-3, 1e-5, 10), 1e-5)

    def test_primary_checkpoint_selection_is_pd_then_fa(self) -> None:
        base = {"pd": 0.95, "fa": 2e-5, "tiny_pd": 0.9, "miou": 0.7, "val_loss": 0.01}
        higher_pd = dict(base, pd=0.96, fa=1e-3, miou=0.5)
        lower_fa_tie = dict(base, fa=1e-5, miou=0.6)
        self.assertGreater(pd_selection_key(higher_pd), pd_selection_key(base))
        self.assertGreater(pd_selection_key(lower_fa_tie), pd_selection_key(base))

    def test_pd_fa_sweep_uses_fa_constraint(self) -> None:
        points = [
            {"threshold": 0.4, "pd": 0.99, "tiny_pd": 1.0, "miou": 0.6, "fa": 2e-4},
            {"threshold": 0.5, "pd": 0.95, "tiny_pd": 0.9, "miou": 0.7, "fa": 8e-6},
            {"threshold": 0.6, "pd": 0.90, "tiny_pd": 0.85, "miou": 0.65, "fa": 2e-6},
        ]
        self.assertEqual(best_point_under_fa(points, 1e-5)["threshold"], 0.5)
        self.assertEqual(best_point_under_fa(points, 3e-6)["threshold"], 0.6)
        self.assertIsNone(best_point_under_fa(points, 1e-7))
        self.assertIn(0.5, threshold_grid(0.05, 0.95, 0.03))

    def test_adaptive_thresholds_cover_empirical_and_logit_tails(self) -> None:
        probabilities = [np.asarray([[0.1, 0.9, 0.99, 0.999]], dtype=np.float32)]
        thresholds, provenance = adaptive_thresholds(probabilities, [0.5], 0.2)
        self.assertIn(0.5, thresholds)
        self.assertGreater(max(thresholds), 0.999)
        self.assertGreater(provenance["tail_logit_threshold_count"], 1)
        self.assertTrue(provenance["empirical_score_quantiles"])


if __name__ == "__main__":
    unittest.main()
