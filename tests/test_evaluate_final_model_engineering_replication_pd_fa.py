from __future__ import annotations

import contextlib
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments import (
    evaluate_final_model_engineering_replication_pd_fa as subject,
)
from experiments import final_model_replication_exact_core as core
from experiments import final_model_replication_seed_contract as seeds
from experiments import watch_final_model_engineering_replication as watcher


SHA = "a" * 64
OTHER_SHA = "b" * 64
VALID_PIXELS = 133 * 32 * 32
TARGET_PIXELS = 39 + 150 * 16
MATCHED_TARGET_PIXELS = TARGET_PIXELS - 16
FIXED_MIOU = MATCHED_TARGET_PIXELS / (TARGET_PIXELS + 2)
FIXED_FA = 2 / VALID_PIXELS
VALIDATION_IDS = tuple(f"image_{index:03d}" for index in range(133))
VALIDATION_IDS_SHA = (
    subject.statistics_cache.validation_identifier_sha256(VALIDATION_IDS)
)


def checkpoint_metrics() -> dict[str, float | int]:
    return {
        "pd": 188 / 189,
        "fa": FIXED_FA,
        "miou": FIXED_MIOU,
        "tiny_pd": 1.0,
        "false_objects_per_image": 2 / 133,
        "target_count": 189,
        "matched_target_count": 188,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 39,
        "unmatched_predicted_object_count": 2,
        "valid_pixel_count": VALID_PIXELS,
    }


def make_request(
    *,
    output_root: Path,
    trajectory_seed: int,
    arm: str,
    checkpoint_filename: str,
    checkpoint_sha256: str = SHA,
) -> subject.CheckpointEvaluationRequest:
    definition = core.arm_definition(arm)
    run_directory = watcher.run_directory(
        output_root,
        trajectory_seed,
        arm,
    ).resolve()
    selection_role, checkpoint_role = {
        name: (selection, role)
        for name, selection, role in subject.CHECKPOINT_SPECS
    }[checkpoint_filename]
    run_identity = {
        "schema": core.exact_runner.RUN_IDENTITY_SCHEMA,
        "run_id": (
            f"{definition.trainer.RUN_ID_PREFIX}NUDT-SIRST:"
            f"{definition.variant}:seed-{trajectory_seed}:"
            f"split-{seeds.SPLIT_SEED}:"
            f"{core.ENGINEERING_RUN_TAGS[arm]}"
        ),
        "variant": definition.variant,
        "dataset": subject.DATASET,
        "seed": trajectory_seed,
        "split_seed": seeds.SPLIT_SEED,
        "source_locks": {
            core.SOURCE_LOCK_KEY: SHA,
            "training_data": OTHER_SHA,
            "survival_target_statistics": OTHER_SHA,
            "parent_checkpoint": OTHER_SHA,
        },
        "training_contract": {},
    }
    return subject.CheckpointEvaluationRequest(
        arm=arm,
        variant=definition.variant,
        trajectory_seed=trajectory_seed,
        run_directory=run_directory,
        run_identity=run_identity,
        seed_contract_path=Path("/tmp/seed-contract.json"),
        seed_contract_sha256=SHA,
        child_manifest_path=Path(
            f"/tmp/child-{trajectory_seed}-{arm}.json"
        ),
        child_manifest_sha256=SHA,
        source_lock_path=Path("/tmp/source-lock.json"),
        source_lock_sha256=SHA,
        protocol_sha256=SHA,
        split_sha256=SHA,
        summary_sha256=SHA,
        metrics_sha256=SHA,
        training_data_sha256=OTHER_SHA,
        normalization_sha256=SHA,
        validation_split_sha256=VALIDATION_IDS_SHA,
        validation_ids=VALIDATION_IDS,
        checkpoint_filename=checkpoint_filename,
        checkpoint_path=(run_directory / checkpoint_filename).resolve(),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=700,
        checkpoint_role=checkpoint_role,
        selection_role=selection_role,
        checkpoint_validation_metrics=checkpoint_metrics(),
    )


def recorded_assignment(
    request: subject.CheckpointEvaluationRequest,
) -> dict[str, object]:
    physical_gpu_index = subject.ARM_PHYSICAL_GPU_INDICES[request.arm]
    physical_gpu_uuid = core.arm_definition(
        request.arm
    ).trainer.PHYSICAL_GPU_UUIDS[str(physical_gpu_index)]
    return {
        "device": "cuda:0",
        "physical_gpu_index": physical_gpu_index,
        "physical_gpu_uuid": physical_gpu_uuid,
        "cuda_visible_devices": physical_gpu_uuid,
        "visible_cuda_device_count": 1,
        "device_name": "NVIDIA GeForce RTX 5090",
    }


def matrix_requests(
    output_root: Path,
) -> list[subject.CheckpointEvaluationRequest]:
    return [
        make_request(
            output_root=output_root,
            trajectory_seed=trajectory_seed,
            arm=arm,
            checkpoint_filename=checkpoint_filename,
            checkpoint_sha256=f"{index + 1:064x}",
        )
        for index, (trajectory_seed, arm, checkpoint_filename) in enumerate(
            (
                (trajectory_seed, arm, checkpoint_filename)
                for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
                for arm in core.SUPPORTED_ARMS
                for checkpoint_filename, _, _ in subject.CHECKPOINT_SPECS
            )
        )
    ]


def point(
    *,
    threshold: float,
    matched: int,
    matched_tiny: int,
    fa: float,
    miou: float,
    predicted: int,
    unmatched: int,
) -> dict[str, float | int]:
    return {
        "threshold": threshold,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": unmatched / 133,
        "tiny_pd": matched_tiny / 39,
        "target_count": 189,
        "matched_target_count": matched,
        "tiny_target_count": 39,
        "matched_tiny_target_count": matched_tiny,
        "predicted_object_count": predicted,
        "unmatched_predicted_object_count": unmatched,
        "valid_pixel_count": VALID_PIXELS,
    }


def prediction_arrays():
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    target_index = 0
    for image_index in range(133):
        probability = np.zeros((32, 32), dtype=np.float32)
        target = np.zeros((32, 32), dtype=np.float32)
        target_count = 2 if image_index < 56 else 1
        for slot in range(target_count):
            tiny = target_index < 39
            missed = target_index == 188
            if slot == 0:
                row, column = 2, 2
            else:
                row, column = 20, 20
            if tiny:
                target[row, column] = 1.0
                if not missed:
                    probability[row, column] = 0.9
            else:
                target[row : row + 4, column : column + 4] = 1.0
                if not missed:
                    probability[
                        row : row + 4,
                        column : column + 4,
                    ] = 0.9
            target_index += 1
        if image_index == 0:
            probability[2, 28] = 0.9
        elif image_index == 1:
            probability[28, 2] = 0.9
        probabilities.append(probability)
        targets.append(target)
        losses.append(0.1)
    assert target_index == 189
    return probabilities, targets, losses


def shared_result(
    request: subject.CheckpointEvaluationRequest,
) -> dict[str, object]:
    source_binding = subject.frozen_evaluation_core_binding()
    fixed = point(
        threshold=0.5,
        matched=188,
        matched_tiny=39,
        fa=FIXED_FA,
        miou=FIXED_MIOU,
        predicted=190,
        unmatched=2,
    )
    strict = point(
        threshold=subject.closed_interval_core.LAST_FLOAT32_BELOW_ONE,
        matched=187,
        matched_tiny=39,
        fa=0.0,
        miou=0.94,
        predicted=187,
        unmatched=0,
    )
    upper = point(
        threshold=subject.closed_interval_core.UPPER_BOUNDARY_THRESHOLD,
        matched=0,
        matched_tiny=0,
        fa=0.0,
        miou=0.0,
        predicted=0,
        unmatched=0,
    )
    points = [fixed, strict, upper]
    budget_points = {
        key: copy.deepcopy(
            subject.sweep_core.best_point_under_fa(points, budget)
        )
        for key, budget in zip(subject.BUDGET_KEYS, subject.FA_BUDGETS)
    }
    return {
        "run_directory": str(request.run_directory),
        "checkpoint": str(request.checkpoint_path),
        "checkpoint_sha256": request.checkpoint_sha256,
        "checkpoint_epoch": request.checkpoint_epoch,
        "checkpoint_role": request.checkpoint_role,
        "checkpoint_validation_metrics": copy.deepcopy(
            dict(request.checkpoint_validation_metrics)
        ),
        "variant": request.variant,
        "dataset": subject.DATASET,
        "seed": request.trajectory_seed,
        "split_seed": seeds.SPLIT_SEED,
        "validation_count": subject.EXPECTED_VALIDATION_COUNT,
        "validation_split_sha256": request.validation_split_sha256,
        "official_test_accessed": False,
        "match_radius": subject.FORMAL_MATCH_RADIUS,
        "tiny_area": subject.FORMAL_TINY_AREA,
        "threshold_configuration": {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": list(subject.EXTRA_THRESHOLDS),
            "tail_logit_step": 0.1,
            "fa_budgets": list(subject.FA_BUDGETS),
        },
        "threshold_provenance": {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "score_count": VALID_PIXELS,
            "exact_one_score_count": 0,
            "added_thresholds": [
                subject.closed_interval_core.LAST_FLOAT32_BELOW_ONE,
                subject.closed_interval_core.UPPER_BOUNDARY_THRESHOLD,
            ],
            "last_float32_below_one": (
                subject.closed_interval_core.LAST_FLOAT32_BELOW_ONE
            ),
            "last_float32_semantics": "exact_one_score_plateau",
            "upper_boundary_threshold": (
                subject.closed_interval_core.UPPER_BOUNDARY_THRESHOLD
            ),
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
            "total_unique_threshold_count": len(points),
        },
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "fixed_threshold_0_5_checkpoint_audit": (
            subject.point_validator._fixed_threshold_checkpoint_audit(
                fixed,
                request.checkpoint_validation_metrics,
            )
        ),
        "best_points_under_fa_budget": budget_points,
        "points": points,
        "audit": {
            "expected_epochs": subject.EXPECTED_EPOCHS,
            "metrics_event_count": subject.EXPECTED_EPOCHS,
            "metrics_epoch_range": [1, subject.EXPECTED_EPOCHS],
            "summary_status": "complete",
            "selection_source": "internal_validation_only",
            "integrity_checks_passed": {
                name: True
                for name in subject.point_validator.REQUIRED_INTEGRITY_CHECKS
            },
            "artifact_sha256": {
                "protocol.json": request.protocol_sha256,
                "split.json": request.split_sha256,
                "summary.json": request.summary_sha256,
                "metrics.jsonl": request.metrics_sha256,
                "checkpoint": request.checkpoint_sha256,
                "evaluator": source_binding[
                    "checkpoint_local_adapter"
                ]["sha256"],
            }
        },
    }


class EngineeringCheckpointLocalPlanTests(unittest.TestCase):
    def test_exact_matrix_has_eight_unique_threshold_domains(self) -> None:
        output_root = Path("/tmp/engineering-evaluation-unit")
        requests = [
            make_request(
                output_root=output_root,
                trajectory_seed=trajectory_seed,
                arm=arm,
                checkpoint_filename=checkpoint_filename,
                checkpoint_sha256=(
                    f"{index + 1:064x}"
                ),
            )
            for index, (trajectory_seed, arm, checkpoint_filename) in enumerate(
                (
                    (trajectory_seed, arm, checkpoint_filename)
                    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
                    for arm in core.SUPPORTED_ARMS
                    for checkpoint_filename, _, _ in subject.CHECKPOINT_SPECS
                )
            )
        ]
        plan = subject.assemble_evaluation_plan(requests)
        self.assertEqual(plan["request_count"], 8)
        self.assertEqual(plan["threshold_domain_count"], 8)
        self.assertEqual(plan["fixed_threshold"], 0.5)
        self.assertEqual(plan["fa_budgets"], list(subject.FA_BUDGETS))
        self.assertFalse(plan["cross_checkpoint_point_pooling"])
        self.assertFalse(plan["gpu_work_started"])
        self.assertFalse(plan["persistent_artifact_written"])
        self.assertTrue(plan["execution_available_in_this_stage"])
        self.assertEqual(
            {
                (
                    record["trajectory_seed"],
                    record["arm"],
                    record["checkpoint_filename"],
                )
                for record in plan["requests"]
            },
            {
                (trajectory_seed, arm, checkpoint_filename)
                for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
                for arm in core.SUPPORTED_ARMS
                for checkpoint_filename, _, _ in subject.CHECKPOINT_SPECS
            },
        )

    def test_matrix_rejects_missing_duplicate_and_noncanonical_order(self) -> None:
        requests = matrix_requests(Path("/tmp/engineering-evaluation-order"))
        invalid_matrices = (
            requests[:-1],
            [requests[0], *requests[:-1]],
            list(reversed(requests)),
        )
        for invalid in invalid_matrices:
            with self.subTest(
                request_count=len(invalid),
                first=invalid[0].threshold_domain_id,
            ):
                with self.assertRaises(subject.EngineeringEvaluationError):
                    subject.assemble_evaluation_plan(invalid)

    def test_checkpoint_epoch_must_lie_in_closed_1_to_800_range(self) -> None:
        requests = matrix_requests(Path("/tmp/engineering-evaluation-epoch"))
        for epoch in (0, 801, True, 1.5):
            bad = subject.CheckpointEvaluationRequest(
                **{
                    **requests[0].__dict__,
                    "checkpoint_epoch": epoch,
                }
            )
            with self.subTest(epoch=epoch):
                with self.assertRaisesRegex(
                    subject.EngineeringEvaluationError,
                    "outside 1..800",
                ):
                    subject._validate_request_shape(bad)

    def test_builder_seed_42_and_unregistered_seed_are_rejected(self) -> None:
        output_root = Path("/tmp/engineering-evaluation-unit")
        for invalid in (42, 123456):
            request = make_request(
                output_root=output_root,
                trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
                arm=core.ARM_B,
                checkpoint_filename="best_miou.pth.tar",
            )
            tampered = subject.CheckpointEvaluationRequest(
                **{
                    **request.__dict__,
                    "trajectory_seed": invalid,
                }
            )
            with self.assertRaises(subject.EngineeringEvaluationError):
                subject.assemble_evaluation_plan([tampered] * 8)

    def test_build_plan_calls_only_four_run_preflights(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            output_root = Path(directory_text) / "uncreated-results"
            calls: list[tuple[int, str]] = []

            def fake_preflight(**kwargs):
                trajectory_seed = kwargs["trajectory_seed"]
                arm = kwargs["arm"]
                calls.append((trajectory_seed, arm))
                return tuple(
                    make_request(
                        output_root=output_root,
                        trajectory_seed=trajectory_seed,
                        arm=arm,
                        checkpoint_filename=filename,
                        checkpoint_sha256=(
                            f"{len(calls):02x}" * 32
                            if filename == "best_miou.pth.tar"
                            else f"{len(calls) + 4:02x}" * 32
                        ),
                    )
                    for filename, _, _ in subject.CHECKPOINT_SPECS
                )

            plan = subject.build_evaluation_plan(
                output_root=output_root,
                source_lock_path=Path("/tmp/source-lock.json"),
                run_preflight=fake_preflight,
            )
            self.assertEqual(
                calls,
                [
                    (trajectory_seed, arm)
                    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
                    for arm in core.SUPPORTED_ARMS
                ],
            )
            self.assertEqual(plan["request_count"], 8)
            self.assertFalse(output_root.exists())

    def test_subset_selection_supports_one_arm_seed_and_checkpoint(self) -> None:
        output_root = Path("/tmp/engineering-evaluation-unit")
        requests = [
            make_request(
                output_root=output_root,
                trajectory_seed=trajectory_seed,
                arm=arm,
                checkpoint_filename=filename,
                checkpoint_sha256=f"{index + 1:064x}",
            )
            for index, (trajectory_seed, arm, filename) in enumerate(
                (
                    (trajectory_seed, arm, filename)
                    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
                    for arm in core.SUPPORTED_ARMS
                    for filename, _, _ in subject.CHECKPOINT_SPECS
                )
            )
        ]
        selected = subject.select_requests(
            requests,
            arms=[core.ARM_D],
            trajectory_seeds=[seeds.ENGINEERING_TRAJECTORY_SEEDS[1]],
            checkpoints=["best_miou.pth.tar"],
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].arm, core.ARM_D)
        self.assertEqual(
            selected[0].trajectory_seed,
            seeds.ENGINEERING_TRAJECTORY_SEEDS[1],
        )


class EngineeringTrajectoryBuilderTests(unittest.TestCase):
    def test_protocol_must_bind_seed_manifest_and_source_lock(self) -> None:
        replication = {
            "arm": core.ARM_B,
            "variant": core.arm_definition(core.ARM_B).variant,
            "trajectory_seed": seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
            "seed_contract_sha256": SHA,
            "child_initialization_manifest_sha256": SHA,
            "certification_source_lock_sha256": SHA,
        }
        inputs = SimpleNamespace(
            metadata=lambda: copy.deepcopy(replication),
            source_lock_sha256=SHA,
            initialization_sha256=SHA,
            schedule_sha256=SHA,
        )
        protocol = {
            "model": {
                "replication_contract": copy.deepcopy(replication),
            }
        }
        identity = {
            "source_locks": {
                core.SOURCE_LOCK_KEY: SHA,
            }
        }
        subject._validate_protocol_replication_binding(
            protocol,
            inputs=inputs,
            run_identity=identity,
        )
        protocol["model"]["replication_contract"][
            "child_initialization_manifest_sha256"
        ] = OTHER_SHA
        with self.assertRaises(subject.EngineeringEvaluationError):
            subject._validate_protocol_replication_binding(
                protocol,
                inputs=inputs,
                run_identity=identity,
            )

    def test_non42_builder_is_bound_to_validated_inputs(self) -> None:
        output_root = Path("/tmp/engineering-evaluation-unit")
        request = make_request(
            output_root=output_root,
            trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
            arm=core.ARM_B,
            checkpoint_filename="best_miou.pth.tar",
        )
        definition = SimpleNamespace(
            arm=request.arm,
            variant=request.variant,
        )
        inputs = SimpleNamespace(
            definition=definition,
            trajectory_seed=request.trajectory_seed,
            schedule_path=request.seed_contract_path,
            schedule_sha256=request.seed_contract_sha256,
            initialization_path=request.child_manifest_path,
            initialization_sha256=request.child_manifest_sha256,
            source_lock_path=request.source_lock_path,
            source_lock_sha256=request.source_lock_sha256,
        )
        trainer = SimpleNamespace(
            FORMAL_EPS=1e-6,
            build_selected_model=mock.Mock(return_value=("model", {})),
        )
        with mock.patch.object(
            subject.core,
            "replication_trainer_overlay",
            return_value=contextlib.nullcontext(trainer),
        ):
            with subject.trajectory_model_builder(request, inputs) as builder:
                self.assertEqual(
                    builder(request.variant, request.trajectory_seed),
                    ("model", {}),
                )
                with self.assertRaises(subject.EngineeringEvaluationError):
                    builder(request.variant, 42)
        trainer.build_selected_model.assert_called_once_with(
            request.variant,
            request.trajectory_seed,
            eps=trainer.FORMAL_EPS,
        )

    def test_manifest_digest_mismatch_blocks_builder(self) -> None:
        output_root = Path("/tmp/engineering-evaluation-unit")
        request = make_request(
            output_root=output_root,
            trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
            arm=core.ARM_D,
            checkpoint_filename="best.pth.tar",
        )
        inputs = SimpleNamespace(
            definition=SimpleNamespace(
                arm=request.arm,
                variant=request.variant,
            ),
            trajectory_seed=request.trajectory_seed,
            schedule_path=request.seed_contract_path,
            schedule_sha256=request.seed_contract_sha256,
            initialization_path=request.child_manifest_path,
            initialization_sha256=OTHER_SHA,
            source_lock_path=request.source_lock_path,
            source_lock_sha256=request.source_lock_sha256,
        )
        with self.assertRaisesRegex(
            subject.EngineeringEvaluationError,
            "child-manifest SHA-256",
        ):
            with subject.trajectory_model_builder(request, inputs):
                self.fail("mismatched input binding must not yield a builder")


class EngineeringFormalDeviceAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.b_request = make_request(
            output_root=Path("/tmp/engineering-evaluation-device"),
            trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
            arm=core.ARM_B,
            checkpoint_filename="best_miou.pth.tar",
        )
        self.d_request = make_request(
            output_root=Path("/tmp/engineering-evaluation-device"),
            trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
            arm=core.ARM_D,
            checkpoint_filename="best_miou.pth.tar",
        )

    def test_recorded_assignment_requires_b_gpu2_and_d_gpu3(self) -> None:
        self.assertEqual(
            subject._validate_recorded_device_assignment(
                recorded_assignment(self.b_request),
                request=self.b_request,
            )["physical_gpu_index"],
            2,
        )
        self.assertEqual(
            subject._validate_recorded_device_assignment(
                recorded_assignment(self.d_request),
                request=self.d_request,
            )["physical_gpu_index"],
            3,
        )
        with self.assertRaises(subject.EngineeringEvaluationError):
            subject._validate_recorded_device_assignment(
                recorded_assignment(self.d_request),
                request=self.b_request,
            )
        with self.assertRaises(subject.EngineeringEvaluationError):
            subject._validate_recorded_device_assignment(
                {
                    "device": "cpu",
                    "physical_gpu_index": None,
                    "physical_gpu_uuid": None,
                    "cuda_visible_devices": None,
                    "visible_cuda_device_count": 0,
                    "device_name": "cpu",
                },
                request=self.b_request,
            )
        non_integer_count = recorded_assignment(self.b_request)
        non_integer_count["visible_cuda_device_count"] = True
        with self.assertRaisesRegex(
            subject.EngineeringEvaluationError,
            "count must be an integer",
        ):
            subject._validate_recorded_device_assignment(
                non_integer_count,
                request=self.b_request,
            )

    def test_live_assignment_rejects_cpu_and_wrong_arm_gpu(self) -> None:
        with self.assertRaisesRegex(
            subject.EngineeringEvaluationError,
            "cannot use CPU",
        ):
            subject.device_assignment("cpu", arm=core.ARM_B)
        assignment = recorded_assignment(self.b_request)
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: assignment["device_name"],
        )
        environment = {
            "CUDA_VISIBLE_DEVICES": assignment["physical_gpu_uuid"],
            subject.EVALUATION_PHYSICAL_GPU_INDEX_ENV: "2",
            subject.EVALUATION_PHYSICAL_GPU_UUID_ENV: (
                assignment["physical_gpu_uuid"]
            ),
        }
        with mock.patch.object(
            subject.sweep_core,
            "torch",
            SimpleNamespace(cuda=fake_cuda),
        ), mock.patch.dict("os.environ", environment, clear=False):
            self.assertEqual(
                subject.device_assignment(
                    "cuda:0",
                    arm=core.ARM_B,
                ),
                assignment,
            )
            with self.assertRaises(subject.EngineeringEvaluationError):
                subject.device_assignment(
                    "cuda:0",
                    arm=core.ARM_D,
                    physical_gpu_index=2,
                    physical_gpu_uuid=assignment["physical_gpu_uuid"],
                )


class EngineeringCheckpointLocalResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = make_request(
            output_root=Path("/tmp/engineering-evaluation-unit"),
            trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[1],
            arm=core.ARM_D,
            checkpoint_filename="best_miou.pth.tar",
        )

    def test_fixed_metrics_and_five_local_budgets_are_exposed(self) -> None:
        finalized = subject.validate_checkpoint_local_result(
            shared_result(self.request),
            self.request,
        )
        fixed = finalized["fixed_threshold_0_5"]
        self.assertEqual(
            tuple(subject.METRIC_OUTPUTS),
            (
                "pd",
                "fa",
                "miou",
                "tiny_pd",
                "false_objects_per_image",
                "unmatched_predicted_object_count",
            ),
        )
        for metric in subject.METRIC_OUTPUTS:
            self.assertIn(metric, fixed)
        self.assertEqual(
            tuple(finalized["best_points_under_fa_budget"]),
            subject.BUDGET_KEYS,
        )
        self.assertEqual(
            finalized["source_checkpoint_identity"][
                "threshold_domain_id"
            ],
            self.request.threshold_domain_id,
        )
        self.assertFalse(finalized["cross_checkpoint_point_pooling"])

    def test_foreign_budget_point_is_rejected(self) -> None:
        payload = shared_result(self.request)
        foreign = point(
            threshold=0.7,
            matched=189,
            matched_tiny=39,
            fa=0.0,
            miou=0.99,
            predicted=189,
            unmatched=0,
        )
        payload["best_points_under_fa_budget"]["5e-06"] = foreign
        with self.assertRaisesRegex(
            subject.EngineeringEvaluationError,
            "Fa budget",
        ):
            subject.validate_checkpoint_local_result(payload, self.request)

    def test_checkpoint_and_run_identity_mismatch_are_rejected(self) -> None:
        for field, value in (
            ("seed", seeds.ENGINEERING_TRAJECTORY_SEEDS[0]),
            ("checkpoint_sha256", OTHER_SHA),
        ):
            payload = shared_result(self.request)
            payload[field] = value
            with self.assertRaises(subject.EngineeringEvaluationError):
                subject.validate_checkpoint_local_result(
                    payload,
                    self.request,
                )

    def test_cache_identity_directly_binds_full_engineering_request(self) -> None:
        source_binding = subject.frozen_evaluation_core_binding()
        engineering = subject._engineering_cache_request_identity(
            self.request,
            source_binding=source_binding,
        )
        for name, expected in (
            ("arm", self.request.arm),
            ("variant", self.request.variant),
            ("trajectory_seed", self.request.trajectory_seed),
            ("run_id", self.request.run_identity["run_id"]),
            ("run_directory", str(self.request.run_directory)),
            ("seed_contract_sha256", self.request.seed_contract_sha256),
            ("child_manifest_sha256", self.request.child_manifest_sha256),
            ("source_lock_sha256", self.request.source_lock_sha256),
            ("checkpoint_sha256", self.request.checkpoint_sha256),
        ):
            self.assertEqual(engineering[name], expected)
        derivation = engineering[
            "collector_evaluator_sha256_derivation"
        ]
        self.assertEqual(
            derivation["adapter_source_sha256"],
            source_binding["checkpoint_local_adapter"]["sha256"],
        )
        self.assertEqual(
            engineering["collector_evaluator_sha256"],
            subject._canonical_digest(derivation),
        )
        baseline = subject._prediction_cache_identity(
            self.request,
            source_binding=source_binding,
        )
        for field in (
            "seed_contract_sha256",
            "child_manifest_sha256",
            "source_lock_sha256",
        ):
            changed = subject.CheckpointEvaluationRequest(
                **{
                    **self.request.__dict__,
                    field: OTHER_SHA
                    if self.request.__dict__[field] != OTHER_SHA
                    else "c" * 64,
                }
            )
            self.assertNotEqual(
                subject._prediction_cache_identity(
                    changed,
                    source_binding=source_binding,
                )["cache_key_sha256"],
                baseline["cache_key_sha256"],
            )

    def test_one_file_partial_cache_is_preserved_then_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            request = make_request(
                output_root=Path(directory_text),
                trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
                arm=core.ARM_B,
                checkpoint_filename="best_miou.pth.tar",
            )
            request.run_directory.mkdir(parents=True)
            source_binding = subject.frozen_evaluation_core_binding()
            identity = subject._prediction_cache_identity(
                request,
                source_binding=source_binding,
            )
            metadata_path, arrays_path = subject.statistics_cache.cache_paths(
                request.prediction_cache_directory,
                identity,
            )
            request.prediction_cache_directory.mkdir()
            orphan = b"recognized-one-file-partial"
            arrays_path.write_bytes(orphan)
            probabilities, targets, losses = prediction_arrays()
            binding = subject.seal_or_validate_prediction_cache(
                request,
                probabilities=probabilities,
                targets=targets,
                losses=losses,
                source_binding=source_binding,
            )
            self.assertTrue(metadata_path.is_file())
            self.assertTrue(Path(binding["arrays_path"]).is_file())
            quarantines = list(
                request.run_directory.glob(
                    f"{subject.CACHE_QUARANTINE_PREFIX}"
                    f"{request.prediction_cache_directory.name}.*"
                )
            )
            self.assertEqual(len(quarantines), 1)
            quarantined_arrays = quarantines[0] / arrays_path.name
            self.assertEqual(quarantined_arrays.read_bytes(), orphan)
            self.assertEqual(
                binding["engineering_request_identity"][
                    "seed_contract_sha256"
                ],
                request.seed_contract_sha256,
            )

    def test_invalid_partial_cache_directory_is_not_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            request = make_request(
                output_root=Path(directory_text),
                trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
                arm=core.ARM_B,
                checkpoint_filename="best_miou.pth.tar",
            )
            request.run_directory.mkdir(parents=True)
            source_binding = subject.frozen_evaluation_core_binding()
            identity = subject._prediction_cache_identity(
                request,
                source_binding=source_binding,
            )
            request.prediction_cache_directory.mkdir()
            unexpected = request.prediction_cache_directory / "foreign.bin"
            unexpected.write_bytes(b"preserve")
            probabilities, targets, losses = prediction_arrays()
            with self.assertRaisesRegex(
                subject.EngineeringEvaluationError,
                "unexpected entries",
            ):
                subject.seal_or_validate_prediction_cache(
                    request,
                    probabilities=probabilities,
                    targets=targets,
                    losses=losses,
                    source_binding=source_binding,
                )
            self.assertEqual(unexpected.read_bytes(), b"preserve")
            self.assertFalse(
                list(
                    request.run_directory.glob(
                        f"{subject.CACHE_QUARANTINE_PREFIX}*"
                    )
                )
            )

    def test_lossless_cache_executed_result_write_once_and_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            request = make_request(
                output_root=Path(directory_text),
                trajectory_seed=seeds.ENGINEERING_TRAJECTORY_SEEDS[0],
                arm=core.ARM_B,
                checkpoint_filename="best_miou.pth.tar",
            )
            request.run_directory.mkdir(parents=True)
            probabilities, targets, losses = prediction_arrays()
            source_binding = subject.frozen_evaluation_core_binding()
            cache_binding = subject.seal_or_validate_prediction_cache(
                request,
                probabilities=probabilities,
                targets=targets,
                losses=losses,
                source_binding=source_binding,
            )
            # An interrupted run may leave the complete cache before its JSON.
            self.assertEqual(
                subject.seal_or_validate_prediction_cache(
                    request,
                    probabilities=probabilities,
                    targets=targets,
                    losses=losses,
                    source_binding=source_binding,
                ),
                cache_binding,
            )
            assignment = recorded_assignment(request)
            finalized = subject.validate_checkpoint_local_result(
                shared_result(request),
                request,
                execution_context={
                    "shared_evaluator_completed": True,
                    "legacy_six_tensor_eval_output": True,
                    "device_assignment": assignment,
                    "evaluation_source_binding": source_binding,
                    "prediction_cache": cache_binding,
                },
            )
            self.assertTrue(finalized["execution_complete"])
            self.assertTrue(finalized["paired_image_statistics_available"])
            self.assertEqual(
                finalized["prediction_cache"]["image_count"],
                133,
            )
            def fake_executor(bound_request, *, assignment):
                self.assertEqual(assignment, recorded_assignment(bound_request))
                return subject.write_result_once(
                    bound_request.planned_output_path,
                    finalized,
                    bound_request,
                )

            with mock.patch.object(
                subject,
                "_verify_request_files_unchanged",
            ):
                created = subject.evaluate_or_skip_checkpoint(
                    request,
                    assignment=assignment,
                    executor=fake_executor,
                )
            self.assertEqual(created["status"], "created")
            with self.assertRaises(FileExistsError):
                subject.write_result_once(
                    request.planned_output_path,
                    finalized,
                    request,
                )
            executor = mock.Mock()
            with mock.patch.object(
                subject,
                "_verify_request_files_unchanged",
            ):
                record = subject.evaluate_or_skip_checkpoint(
                    request,
                    assignment=assignment,
                    executor=executor,
                )
            self.assertEqual(record["status"], "skipped_valid_complete")
            executor.assert_not_called()


class EngineeringResultManifestTests(unittest.TestCase):
    def test_manifest_requires_and_pairs_all_eight_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            output_root = Path(directory_text)
            requests = [
                make_request(
                    output_root=output_root,
                    trajectory_seed=trajectory_seed,
                    arm=arm,
                    checkpoint_filename=filename,
                    checkpoint_sha256=f"{index + 1:064x}",
                )
                for index, (trajectory_seed, arm, filename) in enumerate(
                    (
                        (trajectory_seed, arm, filename)
                        for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
                        for arm in core.SUPPORTED_ARMS
                        for filename, _, _ in subject.CHECKPOINT_SPECS
                    )
                )
            ]

            def fake_load(request):
                payload = shared_result(request)
                payload["execution_device_assignment"] = (
                    recorded_assignment(request)
                )
                source_binding = subject.frozen_evaluation_core_binding()
                payload["prediction_cache"] = {
                    "metadata_path": (
                        f"/tmp/{request.threshold_domain_id}.cache.json"
                    ),
                    "metadata_sha256": request.checkpoint_sha256,
                    "arrays_path": (
                        f"/tmp/{request.threshold_domain_id}.arrays.npz"
                    ),
                    "arrays_sha256": request.checkpoint_sha256,
                    "prediction_content_sha256": (
                        request.checkpoint_sha256
                    ),
                    "identity": subject._prediction_cache_identity(
                        request,
                        source_binding=source_binding,
                    ),
                    "engineering_request_identity": (
                        subject._engineering_cache_request_identity(
                            request,
                            source_binding=source_binding,
                        )
                    ),
                    "image_count": 133,
                    "image_ids_sha256": VALIDATION_IDS_SHA,
                    "paired_image_statistics_available": True,
                }
                return payload, request.checkpoint_sha256

            with mock.patch.object(
                subject,
                "load_completed_result",
                side_effect=fake_load,
            ):
                manifest = subject.build_results_manifest(requests)
            self.assertEqual(manifest["result_count"], 8)
            self.assertEqual(manifest["paired_checkpoint_group_count"], 4)
            self.assertTrue(
                manifest["gate_m_train_image_level_inputs_ready"]
            )
            self.assertTrue(
                manifest["all_results_expected_physical_gpu_bound"]
            )
            self.assertEqual(
                manifest["formal_gpu_binding_policy"],
                {
                    "cpu_results_accepted": False,
                    "arm_assignments": {
                        core.ARM_B: {
                            "physical_gpu_index": 2,
                            "physical_gpu_uuid": core.arm_definition(
                                core.ARM_B
                            ).trainer.PHYSICAL_GPU_UUIDS["2"],
                            "logical_device": "cuda:0",
                        },
                        core.ARM_D: {
                            "physical_gpu_index": 3,
                            "physical_gpu_uuid": core.arm_definition(
                                core.ARM_D
                            ).trainer.PHYSICAL_GPU_UUIDS["3"],
                            "logical_device": "cuda:0",
                        },
                    },
                },
            )
            for record in manifest["results"]:
                self.assertEqual(
                    record["execution_device_assignment"][
                        "physical_gpu_index"
                    ],
                    subject.ARM_PHYSICAL_GPU_INDICES[record["arm"]],
                )
                self.assertIn(
                    "engineering_request_identity",
                    record["prediction_cache"],
                )
            self.assertFalse(
                manifest["paired_confidence_intervals_computed"]
            )
            path = subject.default_manifest_path(output_root)
            _, action = subject.write_or_validate_manifest(path, manifest)
            self.assertEqual(action, "created")
            _, action = subject.write_or_validate_manifest(path, manifest)
            self.assertEqual(action, "skipped_identical_complete")
            mutations = {
                "top_level_count": lambda value: value.__setitem__(
                    "result_count",
                    7,
                ),
                "missing_result": lambda value: value["results"].pop(),
                "duplicate_result": lambda value: value["results"].__setitem__(
                    1,
                    copy.deepcopy(value["results"][0]),
                ),
                "wrong_result_order": lambda value: value["results"].reverse(),
                "result_hash": lambda value: value["results"][0].__setitem__(
                    "result_sha256",
                    OTHER_SHA,
                ),
                "cache_hash": lambda value: value["results"][0][
                    "prediction_cache"
                ].__setitem__("arrays_sha256", OTHER_SHA),
                "gpu_assignment": lambda value: value["results"][0][
                    "execution_device_assignment"
                ].__setitem__("physical_gpu_index", 3),
            }
            for name, mutation in mutations.items():
                path.write_bytes(subject._manifest_json_bytes(manifest))
                tampered = copy.deepcopy(manifest)
                mutation(tampered)
                path.write_bytes(subject._manifest_json_bytes(tampered))
                with self.subTest(name=name):
                    with self.assertRaises(
                        subject.EngineeringEvaluationError
                    ):
                        subject.write_or_validate_manifest(path, manifest)


if __name__ == "__main__":
    unittest.main()
