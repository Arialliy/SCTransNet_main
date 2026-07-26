from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from experiments import evaluate_tpd_clean_v5_pd_fa as evaluator


class TPDCleanV5ClosedIntervalSweepTests(unittest.TestCase):
    def test_appends_exact_one_plateau_and_empty_boundary(self) -> None:
        probabilities = [
            np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
            np.asarray([[1.0, 0.75]], dtype=np.float32),
        ]
        thresholds, provenance = evaluator.adaptive_thresholds_closed_interval(
            probabilities,
            [0.5, 0.9999],
            0.1,
        )

        self.assertEqual(thresholds[-2], evaluator.LAST_FLOAT32_BELOW_ONE)
        self.assertEqual(thresholds[-1], evaluator.UPPER_BOUNDARY_THRESHOLD)
        self.assertEqual(
            thresholds.count(evaluator.LAST_FLOAT32_BELOW_ONE),
            1,
        )
        self.assertEqual(
            thresholds.count(evaluator.UPPER_BOUNDARY_THRESHOLD),
            1,
        )
        self.assertEqual(thresholds, sorted(set(thresholds)))
        self.assertTrue(provenance["closed_probability_interval"])
        self.assertFalse(provenance["posthoc_endpoint_completion"])
        self.assertTrue(provenance["preregistered_endpoint_completion"])
        self.assertEqual(
            provenance["endpoint_protocol_stage"],
            "before_formal_training",
        )
        self.assertEqual(provenance["score_dtype"], "float32")
        self.assertEqual(provenance["score_count"], 5)
        self.assertEqual(provenance["exact_one_score_count"], 2)
        self.assertEqual(
            provenance["added_thresholds"],
            [
                evaluator.LAST_FLOAT32_BELOW_ONE,
                evaluator.UPPER_BOUNDARY_THRESHOLD,
            ],
        )
        self.assertEqual(
            provenance["last_float32_semantics"],
            "exact_one_score_plateau",
        )
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
        self.assertFalse(
            any(
                (probability > thresholds[-1]).any()
                for probability in probabilities
            )
        )

    def test_main_binds_v5_builder_variants_and_wrapper_source(self) -> None:
        self.assertEqual(
            evaluator.SUPPORTED_CLEAN_V5_VARIANTS,
            ("tpd_clean_v5_full", "tpd_clean_v5_sal_capacity"),
        )
        original_adaptive = evaluator.base.adaptive_thresholds
        original_builder = evaluator.base.build_model
        original_file = evaluator.base.__file__
        observed: dict[str, object] = {}

        def capture_bindings() -> None:
            observed["adaptive"] = evaluator.base.adaptive_thresholds
            observed["builder"] = evaluator.base.build_model
            observed["file"] = evaluator.base.__file__

        try:
            with mock.patch.object(
                evaluator.base,
                "main",
                side_effect=capture_bindings,
            ) as base_main:
                evaluator.main()
            base_main.assert_called_once_with()
            self.assertIs(
                observed["adaptive"],
                evaluator.adaptive_thresholds_closed_interval,
            )
            self.assertIs(observed["builder"], evaluator.build_clean_v5_model)
            self.assertEqual(observed["file"], evaluator.__file__)
        finally:
            evaluator.base.adaptive_thresholds = original_adaptive
            evaluator.base.build_model = original_builder
            evaluator.base.__file__ = original_file


if __name__ == "__main__":
    unittest.main()
