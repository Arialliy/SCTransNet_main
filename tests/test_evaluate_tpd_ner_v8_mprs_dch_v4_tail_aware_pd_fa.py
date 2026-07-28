#!/usr/bin/env python3
"""CPU-only tests for the formal V4 tail-aware Pd/Fa evaluator."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]

from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as evaluation,
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
    target_count = evaluation.EXPECTED_TARGET_COUNT
    tiny_count = evaluation.EXPECTED_TINY_TARGET_COUNT
    return {
        "val_loss": 0.001,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.9,
        "pixel_recall": 0.9,
        "pixel_f1": 0.9,
        "pd": matched / target_count,
        "tiny_pd": matched_tiny / tiny_count,
        "fa": fa,
        "false_objects_per_image": (
            unmatched / evaluation.EXPECTED_VALIDATION_COUNT
        ),
        "target_count": target_count,
        "matched_target_count": matched,
        "tiny_target_count": tiny_count,
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
        "points": points,
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "fixed_threshold_0_5_checkpoint_audit": (
            evaluation._fixed_threshold_checkpoint_audit(
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
    }
    return payload, checkpoint_metrics


class EvaluatorContractTests(unittest.TestCase):
    def test_checkpoint_roles_and_budgets_are_frozen(self) -> None:
        self.assertEqual(
            evaluation.CHECKPOINT_ROLES,
            {
                "best.pth.tar": "best_validation_pd_primary",
                "best_miou.pth.tar": (
                    "best_validation_miou_secondary"
                ),
            },
        )
        self.assertEqual(
            evaluation.FA_BUDGETS,
            (1e-6, 5e-6, 1e-5, 5e-5, 1e-4),
        )

    def test_source_binding_verifies_current_training_lock(self) -> None:
        binding = evaluation.verify_frozen_training_sources()
        self.assertEqual(
            binding["schema"],
            evaluation.EVALUATION_SOURCE_BINDING_SCHEMA,
        )
        self.assertEqual(
            binding["training_source_lock"]["schema"],
            evaluation.TRAINING_SCHEMA,
        )
        self.assertEqual(
            binding["training_source_lock"]["sha256"],
            evaluation._sha256_file(evaluation.DEFAULT_TRAINING_LOCK),
        )
        self.assertEqual(
            binding["evaluator"]["sha256"],
            evaluation._sha256_file(Path(evaluation.__file__)),
        )

    def test_contract_exposes_v4_formula_identity(self) -> None:
        with mock.patch.object(
            evaluation,
            "_current_evaluation_source_binding",
            return_value={
                "training_source_lock": {"sha256": "a" * 64},
                "shared_metric_core": {"sha256": "b" * 64},
                "closed_interval_core": {"sha256": "c" * 64},
                "determinism_core": {"sha256": "d" * 64},
            },
        ):
            contract = evaluation.evaluator_contract()
        self.assertEqual(contract["dc_support_mode"], "complement_tail")
        self.assertEqual(contract["dc_support_formula_stage4"], "1")
        self.assertEqual(contract["dc_support_formula_stage3_2"], "1-P")
        self.assertEqual(
            contract["tail_z_thresholds"],
            {"4": 1.5, "3": 2.0, "2": 2.5},
        )
        self.assertEqual(
            contract["required_control"],
            evaluation.exact.REQUIRED_CONTROL,
        )
        self.assertEqual(
            contract["paired_gate_predecessor"],
            evaluation.exact.PAIRED_GATE_PREDECESSOR,
        )
        self.assertEqual(
            contract["structural_predecessor"],
            evaluation.exact.STRUCTURAL_PREDECESSOR,
        )

    def test_source_checkpoint_identity_is_formula_complete(self) -> None:
        identity = evaluation._source_checkpoint_identity(
            {"schema": "training-checkpoint-identity"}
        )
        self.assertEqual(identity["dc_support_mode"], "complement_tail")
        self.assertEqual(identity["dc_support_formula_stage4"], "1")
        self.assertEqual(identity["dc_support_formula_stage3_2"], "1-P")
        self.assertEqual(
            identity["tail_z_thresholds"],
            {"4": 1.5, "3": 2.0, "2": 2.5},
        )
        self.assertTrue(identity["tail_z_thresholds_frozen"])
        self.assertTrue(identity["target_protective_complement"])


class ArgumentAndDeviceTests(unittest.TestCase):
    def test_formal_arguments_accept_each_owned_checkpoint(self) -> None:
        for checkpoint in evaluation.CHECKPOINT_ROLES:
            args = evaluation.validate_formal_arguments(
                [
                    "--run-dir",
                    "/tmp/v4-evaluator-test",
                    "--checkpoint",
                    checkpoint,
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(args.checkpoint, checkpoint)
            self.assertEqual(args.device, "cpu")

    def test_formal_arguments_reject_non_owned_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "only best"):
            evaluation.validate_formal_arguments(
                [
                    "--run-dir",
                    "/tmp/v4-evaluator-test",
                    "--checkpoint",
                    "last.pth.tar",
                    "--device",
                    "cpu",
                ]
            )

    def test_formal_arguments_reject_metric_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "fa_budgets"):
            evaluation.validate_formal_arguments(
                [
                    "--run-dir",
                    "/tmp/v4-evaluator-test",
                    "--device",
                    "cpu",
                    "--fa-budgets",
                    "2e-6",
                ]
            )
        with self.assertRaisesRegex(ValueError, "forbids --overwrite"):
            evaluation.validate_formal_arguments(
                [
                    "--run-dir",
                    "/tmp/v4-evaluator-test",
                    "--device",
                    "cpu",
                    "--overwrite",
                ]
            )

    def test_cpu_device_assignment_is_explicit(self) -> None:
        self.assertEqual(
            evaluation._device_assignment("cpu"),
            {
                "device": "cpu",
                "physical_gpu_index": None,
                "physical_gpu_uuid": None,
                "cuda_visible_devices": None,
                "device_name": "cpu",
            },
        )

    def test_cuda_assignment_accepts_only_registered_gpu_2_or_3(self) -> None:
        uuid = evaluation.PHYSICAL_GPU_UUIDS["2"]
        environment = {
            "visible_cuda_device_count": 1,
            "device_uuid": uuid,
            "device_name": "NVIDIA GeForce RTX 5090",
        }
        variables = {
            evaluation.PHYSICAL_GPU_INDEX_ENV: "2",
            evaluation.PHYSICAL_GPU_UUID_ENV: uuid,
            "CUDA_VISIBLE_DEVICES": uuid,
        }
        with mock.patch.dict(os.environ, variables, clear=False), mock.patch.object(
            evaluation.exact.shared_exact,
            "environment_contract",
            return_value=environment,
        ):
            assignment = evaluation._device_assignment("cuda:0")
        self.assertEqual(assignment["physical_gpu_index"], 2)
        self.assertEqual(assignment["physical_gpu_uuid"], uuid)

    def test_cuda_assignment_rejects_gpu_0(self) -> None:
        with mock.patch.dict(
            os.environ,
            {evaluation.PHYSICAL_GPU_INDEX_ENV: "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "must be 2 or 3"):
                evaluation._device_assignment("cuda:0")


class SyntheticMetricTests(unittest.TestCase):
    def test_fixed_point_and_budget_curve_validate(self) -> None:
        payload, checkpoint_metrics = synthetic_sweep()
        fixed = evaluation._validate_point_collection(
            payload,
            checkpoint_metrics,
        )
        budgets = evaluation._normalize_budgets(payload)
        evaluation._validate_closed_interval(payload)
        self.assertEqual(fixed["matched_target_count"], 188)
        self.assertEqual(
            budgets["1e-06"]["matched_target_count"],
            187,
        )
        self.assertEqual(
            budgets["5e-06"]["matched_target_count"],
            188,
        )

    def test_budget_point_is_recomputed_from_raw_points(self) -> None:
        payload, _ = synthetic_sweep()
        payload["best_points_under_fa_budget"]["1e-06"] = copy.deepcopy(
            payload["points"][-1]
        )
        with self.assertRaisesRegex(ValueError, "best point"):
            evaluation._normalize_budgets(payload)

    def test_fixed_point_must_equal_raw_threshold_point(self) -> None:
        payload, checkpoint_metrics = synthetic_sweep()
        payload["fixed_threshold_0_5"]["miou"] = 0.5
        with self.assertRaisesRegex(ValueError, "fixed threshold/raw point"):
            evaluation._validate_point_collection(
                payload,
                checkpoint_metrics,
            )

    def test_false_objects_per_image_is_count_derived(self) -> None:
        payload, checkpoint_metrics = synthetic_sweep()
        payload["points"][0]["false_objects_per_image"] = 0.0
        with self.assertRaisesRegex(
            ValueError,
            "false_objects_per_image differs",
        ):
            evaluation._validate_point_collection(
                payload,
                checkpoint_metrics,
            )

    def test_tiny_pd_is_count_derived(self) -> None:
        payload, checkpoint_metrics = synthetic_sweep()
        payload["points"][0]["tiny_pd"] = 0.5
        with self.assertRaisesRegex(ValueError, "tiny_pd differs"):
            evaluation._validate_point_collection(
                payload,
                checkpoint_metrics,
            )

    def test_closed_interval_requires_empty_upper_endpoint(self) -> None:
        payload, _ = synthetic_sweep()
        payload["points"][-1]["matched_target_count"] = 1
        payload["points"][-1]["pd"] = 1 / evaluation.EXPECTED_TARGET_COUNT
        with self.assertRaisesRegex(ValueError, "upper endpoint pd"):
            evaluation._validate_closed_interval(payload)

    def test_final_coverage_contains_all_requested_metrics(self) -> None:
        payload, checkpoint_metrics = synthetic_sweep()
        fixed = evaluation._validate_point_collection(
            payload,
            checkpoint_metrics,
        )
        budgets = evaluation._normalize_budgets(payload)
        coverage = evaluation._final_metric_coverage(fixed, budgets)
        self.assertEqual(
            coverage["required_metrics"],
            [
                "pd",
                "fa",
                "miou",
                "false_objects_per_image",
                "tiny_pd",
            ],
        )
        self.assertEqual(
            tuple(coverage["fa_budget_points"]),
            evaluation.BUDGET_KEYS,
        )


class ModelAndOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = evaluation.build_model(
            evaluation.VARIANT,
            evaluation.TRAINING_SEED,
        )

    def test_synthetic_v4_state_strict_loads(self) -> None:
        evaluation._validate_model_state(
            {
                "state_dict": self.model.state_dict(),
                "model_metadata": self.metadata,
            }
        )

    def test_wrong_variant_or_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sole formal V4"):
            evaluation.build_model("baseline_sctransnet", 42)
        with self.assertRaisesRegex(ValueError, "seed 42"):
            evaluation.build_model(evaluation.VARIANT, 7)

    def test_output_write_is_atomic_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pd_fa_sweep_best.pth.json"
            with mock.patch.object(
                evaluation,
                "finalize_evaluation_output",
                return_value={"status": "complete"},
            ):
                evaluation._atomic_write_output(
                    output,
                    {},
                    False,
                    artifact_audit={},
                    device_assignment={},
                    json_ready=lambda value: value,
                )
                self.assertEqual(
                    json.loads(output.read_text(encoding="utf-8")),
                    {"status": "complete"},
                )
                with self.assertRaisesRegex(
                    FileExistsError,
                    "refusing to replace",
                ):
                    evaluation._atomic_write_output(
                        output,
                        {},
                        False,
                        artifact_audit={},
                        device_assignment={},
                        json_ready=lambda value: value,
                    )


if __name__ == "__main__":
    unittest.main()
