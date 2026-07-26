from __future__ import annotations

import unittest

import numpy as np

from experiments.evaluate_tpd_clean_v3_pd_fa_closed_interval import (
    LAST_FLOAT32_BELOW_ONE,
    adaptive_thresholds_closed_interval,
)


class ClosedIntervalSweepTests(unittest.TestCase):
    def test_appends_exact_one_plateau_and_empty_boundary(self) -> None:
        probabilities = [
            np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
        ]
        thresholds, provenance = adaptive_thresholds_closed_interval(
            probabilities,
            [0.5, 0.9999],
            0.1,
        )

        self.assertEqual(thresholds[-2], LAST_FLOAT32_BELOW_ONE)
        self.assertEqual(thresholds[-1], 1.0)
        self.assertEqual(thresholds.count(LAST_FLOAT32_BELOW_ONE), 1)
        self.assertEqual(thresholds.count(1.0), 1)
        self.assertEqual(thresholds, sorted(set(thresholds)))
        self.assertTrue(provenance["closed_probability_interval"])
        self.assertTrue(provenance["posthoc_endpoint_completion"])
        self.assertEqual(provenance["score_dtype"], "float32")
        self.assertEqual(provenance["score_count"], 3)
        self.assertEqual(provenance["exact_one_score_count"], 1)
        self.assertEqual(
            provenance["added_thresholds"],
            [LAST_FLOAT32_BELOW_ONE, 1.0],
        )
        self.assertEqual(
            provenance["last_float32_below_one"],
            LAST_FLOAT32_BELOW_ONE,
        )
        self.assertEqual(
            provenance["last_float32_semantics"],
            "exact_one_score_plateau",
        )
        self.assertEqual(provenance["upper_boundary_threshold"], 1.0)
        self.assertEqual(
            provenance["upper_boundary_comparison"],
            "prediction > threshold",
        )
        self.assertEqual(
            provenance["upper_boundary_semantics"],
            "empty_prediction_pd0_fa0",
        )
        self.assertEqual(
            provenance["total_unique_threshold_count"],
            len(thresholds),
        )
        plateau = probabilities[0] > thresholds[-2]
        self.assertEqual(plateau.tolist(), [[False, False, True]])
        self.assertFalse((probabilities[0] > thresholds[-1]).any())


if __name__ == "__main__":
    unittest.main()
