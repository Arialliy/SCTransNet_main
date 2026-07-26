from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from experiments import evaluate_pd_fa_sweep as canonical_base
from experiments import evaluate_tpd_ner_v5_pd_fa as evaluator
from experiments.train_tpd_ner_v5 import build_tpd_ner_v5_model


class TPDNERV5ClosedIntervalEvaluatorTests(unittest.TestCase):
    def test_reuses_v5_preregistered_closed_interval_semantics(self) -> None:
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
        self.assertEqual(thresholds, sorted(set(thresholds)))
        self.assertTrue(provenance["closed_probability_interval"])
        self.assertTrue(provenance["preregistered_endpoint_completion"])
        self.assertFalse(provenance["posthoc_endpoint_completion"])
        self.assertEqual(
            provenance["endpoint_protocol_stage"],
            "before_formal_training",
        )
        self.assertEqual(provenance["exact_one_score_count"], 2)
        self.assertEqual(
            provenance["upper_boundary_comparison"],
            "prediction > threshold",
        )
        self.assertEqual(
            provenance["upper_boundary_semantics"],
            "empty_prediction_pd0_fa0",
        )
        self.assertFalse(
            any(
                (probability > thresholds[-1]).any()
                for probability in probabilities
            )
        )

    def test_private_evaluator_binds_builder_and_wrapper_provenance(self) -> None:
        original = (
            canonical_base.adaptive_thresholds,
            canonical_base.build_model,
            canonical_base.__file__,
        )
        isolated = evaluator._load_isolated_base_evaluator()
        self.assertIsNot(isolated, canonical_base)
        self.assertIs(
            isolated.adaptive_thresholds,
            evaluator.adaptive_thresholds_closed_interval,
        )
        self.assertIs(isolated.build_model, build_tpd_ner_v5_model)
        self.assertEqual(isolated.__file__, evaluator.__file__)
        self.assertEqual(
            (
                canonical_base.adaptive_thresholds,
                canonical_base.build_model,
                canonical_base.__file__,
            ),
            original,
        )

    def test_main_calls_only_the_private_module(self) -> None:
        observed: list[bool] = []
        private = types.SimpleNamespace(main=lambda: observed.append(True))
        original = (
            canonical_base.adaptive_thresholds,
            canonical_base.build_model,
            canonical_base.__file__,
        )
        with mock.patch.object(
            evaluator,
            "_load_isolated_base_evaluator",
            return_value=private,
        ) as loader:
            evaluator.main()
        loader.assert_called_once_with()
        self.assertEqual(observed, [True])
        self.assertEqual(
            (
                canonical_base.adaptive_thresholds,
                canonical_base.build_model,
                canonical_base.__file__,
            ),
            original,
        )

    def test_private_failure_cannot_pollute_canonical_globals(self) -> None:
        class InjectedFailure(Exception):
            pass

        def fail() -> None:
            raise InjectedFailure

        original = (
            canonical_base.adaptive_thresholds,
            canonical_base.build_model,
            canonical_base.__file__,
        )
        with mock.patch.object(
            evaluator,
            "_load_isolated_base_evaluator",
            return_value=types.SimpleNamespace(main=fail),
        ):
            with self.assertRaises(InjectedFailure):
                evaluator.main()
        self.assertEqual(
            (
                canonical_base.adaptive_thresholds,
                canonical_base.build_model,
                canonical_base.__file__,
            ),
            original,
        )

    def test_complete_four_variant_builder_is_registered(self) -> None:
        self.assertEqual(
            evaluator.SUPPORTED_TPD_NER_V5_VARIANTS,
            (
                "tpd_clean_v5_full_relay_off",
                "tpd_clean_v5_full_relay_on",
                "progressive_relay_off",
                "progressive_relay_on",
            ),
        )


if __name__ == "__main__":
    unittest.main()
