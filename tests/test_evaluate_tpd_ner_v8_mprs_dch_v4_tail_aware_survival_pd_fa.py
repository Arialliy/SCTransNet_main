#!/usr/bin/env python3
"""CPU-only contract tests for the formal TSS Pd/Fa evaluator."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa as evaluation,
)


def synthetic_point(
    threshold: float,
    *,
    matched: int,
    matched_tiny: int,
    unmatched: int,
    fa: float,
    miou: float,
) -> dict[str, object]:
    return {
        "val_loss": 0.001,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.9,
        "pixel_recall": 0.9,
        "pixel_f1": 0.9,
        "pd": matched / evaluation.EXPECTED_TARGET_COUNT,
        "tiny_pd": matched_tiny / evaluation.EXPECTED_TINY_TARGET_COUNT,
        "fa": fa,
        "false_objects_per_image": (
            unmatched / evaluation.EXPECTED_VALIDATION_COUNT
        ),
        "target_count": evaluation.EXPECTED_TARGET_COUNT,
        "matched_target_count": matched,
        "tiny_target_count": evaluation.EXPECTED_TINY_TARGET_COUNT,
        "matched_tiny_target_count": matched_tiny,
        "predicted_object_count": matched + unmatched,
        "unmatched_predicted_object_count": unmatched,
        "valid_pixel_count": 8_716_288,
        "threshold": threshold,
    }


def synthetic_sweep() -> tuple[dict[str, object], dict[str, object]]:
    fixed = synthetic_point(
        0.5,
        matched=188,
        matched_tiny=39,
        unmatched=6,
        fa=4.0e-6,
        miou=0.91,
    )
    strict = synthetic_point(
        0.9,
        matched=187,
        matched_tiny=39,
        unmatched=0,
        fa=0.0,
        miou=0.90,
    )
    last = synthetic_point(
        evaluation.LAST_FLOAT32_BELOW_ONE,
        matched=0,
        matched_tiny=0,
        unmatched=0,
        fa=0.0,
        miou=0.0,
    )
    upper = synthetic_point(
        evaluation.UPPER_BOUNDARY_THRESHOLD,
        matched=0,
        matched_tiny=0,
        unmatched=0,
        fa=0.0,
        miou=0.0,
    )
    points = [fixed, strict, last, upper]
    checkpoint_metrics = {
        name: copy.deepcopy(fixed[name])
        for name in evaluation.exact.STORED_VALIDATION_METRICS
    }
    payload: dict[str, object] = {
        "run_directory": "/tmp/tss",
        "checkpoint": "/tmp/tss/best.pth.tar",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_epoch": 37,
        "checkpoint_role": "best_validation_pd_primary",
        "checkpoint_validation_metrics": checkpoint_metrics,
        "variant": evaluation.exact.TSS_CONTROL_VARIANT,
        "dataset": evaluation.DATASET,
        "seed": evaluation.TRAINING_SEED,
        "split_seed": evaluation.SPLIT_SEED,
        "validation_count": evaluation.EXPECTED_VALIDATION_COUNT,
        "validation_split_sha256": "b" * 64,
        "official_test_accessed": False,
        "match_radius": 3.0,
        "tiny_area": 9,
        "threshold_configuration": {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": list(evaluation.EXTRA_THRESHOLDS),
            "tail_logit_step": 0.1,
            "fa_budgets": list(evaluation.FA_BUDGETS),
        },
        "threshold_provenance": {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "score_count": 8_716_288,
            "added_thresholds": [
                evaluation.LAST_FLOAT32_BELOW_ONE,
                evaluation.UPPER_BOUNDARY_THRESHOLD,
            ],
            "last_float32_below_one": (
                evaluation.LAST_FLOAT32_BELOW_ONE
            ),
            "upper_boundary_threshold": (
                evaluation.UPPER_BOUNDARY_THRESHOLD
            ),
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
            "total_unique_threshold_count": len(points),
        },
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "fixed_threshold_0_5_checkpoint_audit": (
            evaluation.v4_evaluator._fixed_threshold_checkpoint_audit(
                fixed,
                checkpoint_metrics,
            )
        ),
        "best_points_under_fa_budget": {
            "1e-06": copy.deepcopy(strict),
            "5e-06": copy.deepcopy(fixed),
            "1e-05": copy.deepcopy(fixed),
            "5e-05": copy.deepcopy(fixed),
            "0.0001": copy.deepcopy(fixed),
        },
        "points": points,
        "audit": {
            "integrity_checks_passed": {
                name: True
                for name in evaluation.v4_evaluator.REQUIRED_INTEGRITY_CHECKS
            }
        },
    }
    return payload, checkpoint_metrics


class ArgumentTests(unittest.TestCase):
    def test_single_checkpoint_defaults_to_best(self) -> None:
        run_dir = evaluation.DEFAULT_RUN_DIRS[
            evaluation.exact.TSS_CONTROL_VARIANT
        ]
        args = evaluation.validate_formal_arguments(
            ["--run-dir", str(run_dir), "--device", "cpu"]
        )
        requests = evaluation.evaluation_requests(args)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].checkpoint, "best.pth.tar")
        self.assertEqual(
            requests[0].variant,
            evaluation.exact.TSS_CONTROL_VARIANT,
        )

    def test_all_four_are_four_checkpoint_local_requests(self) -> None:
        args = evaluation.validate_formal_arguments(
            ["--all-four", "--device", "cpu"]
        )
        requests = evaluation.evaluation_requests(args)
        self.assertEqual(len(requests), 4)
        self.assertEqual(
            {(request.variant, request.checkpoint) for request in requests},
            {
                (variant, checkpoint)
                for variant in evaluation.SUPPORTED_VARIANTS
                for checkpoint in evaluation.CHECKPOINT_ROLES
            },
        )

    def test_all_four_rejects_single_checkpoint_and_overwrite(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            evaluation.validate_formal_arguments(
                ["--all-four", "--checkpoint", "best.pth.tar"]
            )
        with self.assertRaisesRegex(ValueError, "forbids --overwrite"):
            evaluation.validate_formal_arguments(
                [
                    "--run-dir",
                    str(
                        evaluation.DEFAULT_RUN_DIRS[
                            evaluation.exact.TSS_ON_VARIANT
                        ]
                    ),
                    "--overwrite",
                ]
            )

    def test_frozen_budgets_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "fa_budgets"):
            evaluation.validate_formal_arguments(
                [
                    "--run-dir",
                    str(
                        evaluation.DEFAULT_RUN_DIRS[
                            evaluation.exact.TSS_ON_VARIANT
                        ]
                    ),
                    "--fa-budgets",
                    "2e-6",
                ]
            )


class CoreReuseTests(unittest.TestCase):
    def test_closed_interval_and_budget_helpers_are_v4_core(self) -> None:
        self.assertIs(
            evaluation._normalize_budgets,
            evaluation.v4_evaluator._normalize_budgets,
        )
        self.assertIs(
            evaluation._validate_closed_interval,
            evaluation.v4_evaluator._validate_closed_interval,
        )
        self.assertEqual(
            evaluation.FA_BUDGETS,
            (1e-6, 5e-6, 1e-5, 5e-5, 1e-4),
        )

    def test_legacy_guard_accepts_only_six_tensor_tuple(self) -> None:
        tensors = tuple(torch.zeros(1) for _ in range(6))
        self.assertEqual(evaluation._require_legacy_eval_output(tensors), tensors)
        with self.assertRaisesRegex(RuntimeError, "six-tensor"):
            evaluation._require_legacy_eval_output(tensors[:5])
        with self.assertRaisesRegex(RuntimeError, "non-tensor"):
            evaluation._require_legacy_eval_output((*tensors[:5], object()))

    def test_source_binding_verifies_frozen_training_sources(self) -> None:
        binding = evaluation.verify_frozen_training_sources()
        self.assertEqual(binding["schema"], evaluation.SOURCE_BINDING_SCHEMA)
        self.assertEqual(
            binding["training_source_lock"]["sha256"],
            evaluation._sha256_file(
                evaluation.exact.DEFAULT_EXACT_SOURCE_LOCK_PATH
            ),
        )


class OutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload, checkpoint_metrics = synthetic_sweep()
        self.audit = {
            "run_directory": "/tmp/tss",
            "variant": evaluation.exact.TSS_CONTROL_VARIANT,
            "checkpoint_filename": "best.pth.tar",
            "checkpoint_path": "/tmp/tss/best.pth.tar",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_epoch": 37,
            "checkpoint_role": "best_validation_pd_primary",
            "checkpoint_validation_metrics": checkpoint_metrics,
            "checkpoint_identity": {"schema": "checkpoint"},
            "run_identity": {"schema": "run"},
            "source_binding": {"schema": "binding"},
            "state_dict_strict_load": True,
            "legacy_eval_output_verified": True,
        }

    def test_final_output_is_explicitly_checkpoint_local(self) -> None:
        with mock.patch.object(
            evaluation,
            "evaluator_contract",
            return_value={"schema": "contract"},
        ):
            ready = evaluation.finalize_evaluation_output(
                self.payload,
                self.audit,
                device_assignment=evaluation._device_assignment("cpu"),
            )
        self.assertEqual(ready["evaluated_checkpoint_count"], 1)
        self.assertEqual(
            ready["threshold_selection_scope"],
            "single_checkpoint_only",
        )
        self.assertFalse(ready["cross_checkpoint_point_pooling"])
        self.assertEqual(
            tuple(ready["best_points_under_fa_budget"]),
            evaluation.BUDGET_KEYS,
        )

    def test_budget_point_must_come_from_this_checkpoint_raw_points(self) -> None:
        self.payload["best_points_under_fa_budget"]["1e-06"] = copy.deepcopy(
            self.payload["points"][-1]
        )
        with mock.patch.object(
            evaluation,
            "evaluator_contract",
            return_value={"schema": "contract"},
        ):
            with self.assertRaisesRegex(ValueError, "best point"):
                evaluation.finalize_evaluation_output(
                    self.payload,
                    self.audit,
                    device_assignment=evaluation._device_assignment("cpu"),
                )

    def test_atomic_output_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pd_fa_sweep_best.pth.json"
            with mock.patch.object(
                evaluation,
                "finalize_evaluation_output",
                return_value={"status": "complete"},
            ):
                evaluation._atomic_write_output(
                    path,
                    {},
                    False,
                    artifact_audit={},
                    device_assignment={},
                    json_ready=lambda value: value,
                )
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8")),
                    {"status": "complete"},
                )
                with self.assertRaisesRegex(
                    FileExistsError,
                    "refusing to replace",
                ):
                    evaluation._atomic_write_output(
                        path,
                        {},
                        False,
                        artifact_audit={},
                        device_assignment={},
                        json_ready=lambda value: value,
                    )


if __name__ == "__main__":
    unittest.main()
