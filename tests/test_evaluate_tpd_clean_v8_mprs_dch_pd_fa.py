from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from experiments import evaluate_tpd_clean_v8_mprs_dch_pd_fa as evaluator


class TPDCleanV8MPRSDCHClosedIntervalEvaluatorTests(unittest.TestCase):
    def _torch_settings(self) -> dict[str, object]:
        return {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32":
            torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms":
            torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision":
            torch.get_float32_matmul_precision(),
        }

    def _restore_torch_settings(
        self,
        settings: dict[str, object],
    ) -> None:
        torch.backends.cudnn.benchmark = bool(
            settings["cudnn_benchmark"]
        )
        torch.backends.cudnn.deterministic = bool(
            settings["cudnn_deterministic"]
        )
        torch.backends.cuda.matmul.allow_tf32 = bool(
            settings["cuda_matmul_allow_tf32"]
        )
        torch.backends.cudnn.allow_tf32 = bool(
            settings["cudnn_allow_tf32"]
        )
        torch.use_deterministic_algorithms(
            bool(settings["deterministic_algorithms"])
        )
        torch.set_float32_matmul_precision(
            str(settings["float32_matmul_precision"])
        )

    def test_reuses_closed_interval_core_without_metric_override(self) -> None:
        probabilities = [
            np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
            np.asarray([[1.0, 0.75]], dtype=np.float32),
        ]
        thresholds, provenance = (
            evaluator.adaptive_thresholds_closed_interval(
                probabilities,
                [0.5, 0.9999],
                0.1,
            )
        )
        self.assertEqual(
            thresholds[-2],
            evaluator.LAST_FLOAT32_BELOW_ONE,
        )
        self.assertEqual(
            thresholds[-1],
            evaluator.UPPER_BOUNDARY_THRESHOLD,
        )
        self.assertEqual(thresholds, sorted(set(thresholds)))
        self.assertTrue(provenance["closed_probability_interval"])
        self.assertTrue(provenance["preregistered_endpoint_completion"])
        self.assertFalse(provenance["posthoc_endpoint_completion"])
        self.assertEqual(provenance["exact_one_score_count"], 2)

        contract = evaluator.evaluator_contract()
        self.assertFalse(contract["matching_or_metric_override"])
        self.assertEqual(
            contract["variants"],
            [
                "tpd_clean_v8_mprs_dch_full",
                "tpd_clean_v8_mprs_dch_capacity",
            ],
        )
        self.assertEqual(
            contract["fa_budgets"],
            [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        )
        self.assertEqual(
            contract["determinism"],
            evaluator.DETERMINISM_SETTINGS,
        )

    def test_configure_cpu_inference_applies_exact_contract(self) -> None:
        original = self._torch_settings()
        try:
            observed = evaluator.configure_v8_inference("cpu")
            self.assertEqual(
                {
                    key: observed[key]
                    for key in evaluator.DETERMINISM_SETTINGS
                },
                evaluator.DETERMINISM_SETTINGS,
            )
            self.assertIsNone(observed["cublas_workspace_config"])
        finally:
            self._restore_torch_settings(original)

    def test_cuda_configuration_requires_registered_cublas_value(
        self,
    ) -> None:
        original = self._torch_settings()
        try:
            with mock.patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": ""},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "CUBLAS_WORKSPACE_CONFIG",
                ):
                    evaluator.configure_v8_inference("cuda:0")
            with mock.patch.dict(
                os.environ,
                {
                    "CUBLAS_WORKSPACE_CONFIG":
                    evaluator.CUBLAS_WORKSPACE_CONFIG
                },
                clear=False,
            ):
                observed = evaluator.configure_v8_inference("cuda:0")
            self.assertEqual(
                observed["cublas_workspace_config"],
                evaluator.CUBLAS_WORKSPACE_CONFIG,
            )
        finally:
            self._restore_torch_settings(original)

    def test_main_binds_only_v8_builder_and_closed_core(self) -> None:
        original_adaptive = evaluator.base.adaptive_thresholds
        original_builder = evaluator.base.build_model
        original_file = evaluator.base.__file__
        observed: dict[str, object] = {}

        def capture_bindings() -> None:
            observed["adaptive"] = evaluator.base.adaptive_thresholds
            observed["builder"] = evaluator.base.build_model
            observed["file"] = evaluator.base.__file__

        try:
            with (
                mock.patch.object(
                    evaluator.base,
                    "main",
                    side_effect=capture_bindings,
                ) as base_main,
                mock.patch.object(
                    sys,
                    "argv",
                    [evaluator.__file__, "--device", "cpu"],
                ),
            ):
                original_settings = self._torch_settings()
                try:
                    evaluator.main()
                finally:
                    self._restore_torch_settings(original_settings)
            base_main.assert_called_once_with()
            self.assertIs(
                observed["adaptive"],
                evaluator.adaptive_thresholds_closed_interval,
            )
            self.assertIs(
                observed["builder"],
                evaluator.build_clean_v8_mprs_dch_model,
            )
            self.assertEqual(observed["file"], evaluator.__file__)
        finally:
            evaluator.base.adaptive_thresholds = original_adaptive
            evaluator.base.build_model = original_builder
            evaluator.base.__file__ = original_file


if __name__ == "__main__":
    unittest.main()
