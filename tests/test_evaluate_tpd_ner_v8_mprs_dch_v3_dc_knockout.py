from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments import (
    evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout as subject,
)
from experiments import (
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)


def valid_state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            ("encoder.weight", torch.tensor([[1.0, -2.0]], dtype=torch.float32)),
            ("tpd_ner.dc_offsets.4", torch.tensor([0.4], dtype=torch.float32)),
            ("tpd_ner.dc_offsets.3", torch.tensor([-0.3], dtype=torch.float32)),
            ("tpd_ner.dc_offsets.2", torch.tensor([0.2], dtype=torch.float32)),
            ("decoder.bias", torch.tensor([0.125], dtype=torch.float32)),
        )
    )


def changed_keys(
    before: OrderedDict[str, torch.Tensor],
    after: OrderedDict[str, torch.Tensor],
) -> set[str]:
    return {
        key
        for key in before
        if not torch.equal(before[key], after[key])
    }


_SYNTHETIC_SWEEP: dict | None = None


def synthetic_complete_sweep() -> dict:
    """Build one small, real 133-image sweep with the formal metric core."""

    global _SYNTHETIC_SWEEP
    if _SYNTHETIC_SWEEP is None:
        shape = (45, 90)
        target = np.zeros(shape, dtype=np.float32)
        # 150 separated area-10 objects plus 39 separated one-pixel objects
        # exactly exercise the frozen target/tiny-target count contract.
        for row in range(10):
            for column in range(15):
                target[
                    3 * row : 3 * row + 2,
                    6 * column : 6 * column + 5,
                ] = 1.0
        for index in range(39):
            target[
                32 + 2 * (index // 20),
                2 * (index % 20),
            ] = 1.0
        probabilities = [
            np.full(shape, 0.25, dtype=np.float32),
            *[
                np.full((1, 1), 0.25, dtype=np.float32)
                for _ in range(spec.VALIDATION_COUNT - 1)
            ],
        ]
        targets = [
            target,
            *[
                np.zeros((1, 1), dtype=np.float32)
                for _ in range(spec.VALIDATION_COUNT - 1)
            ],
        ]
        _SYNTHETIC_SWEEP = subject.sweep_predictions(
            probabilities,
            targets,
            [0.1] * spec.VALIDATION_COUNT,
            validation_count=spec.VALIDATION_COUNT,
        )
    return copy.deepcopy(_SYNTHETIC_SWEEP)


def complete_payload_fixture(
    checkpoint_name: str,
) -> tuple[dict, dict, dict, dict]:
    source_state = valid_state()
    checkpoint = {"state_dict": source_state}
    checkpoint_sha256 = (
        "a" * 64 if checkpoint_name == "best.pth.tar" else "c" * 64
    )
    artifact_audit = {
        "run_directory": str(spec.FORMAL_RUN_DIR),
        "checkpoint_filename": checkpoint_name,
        "checkpoint_role": spec.CHECKPOINT_ROLES[checkpoint_name],
        "checkpoint_epoch": spec.EXPECTED_EPOCHS,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity": {
            "fixture": "checkpoint",
            "checkpoint": checkpoint_name,
        },
        "run_identity": {"fixture": "run"},
    }
    source_binding = {
        "diagnostic_source_lock": {"sha256": "b" * 64},
        "knockout_spec_sha256": spec.specification_sha256(),
    }
    source_sha256 = subject.state_content_sha256(source_state)
    non_dc_sha256 = subject.non_dc_state_sha256(source_state)
    sweep = synthetic_complete_sweep()
    evaluations = []
    for mode in spec.KNOCKOUT_MODES:
        transformed = subject.transform_state_dict(source_state, mode)
        evaluations.append(
            {
                "schema": subject.STATE_TRANSFORM_SCHEMA,
                "status": "complete",
                "knockout_mode": mode,
                "zeroed_state_keys": list(spec.KNOCKOUT_ZERO_KEYS[mode]),
                "effective_changed_state_keys": sorted(
                    changed_keys(source_state, transformed)
                ),
                "source_dc_offsets": subject.dc_offset_records(source_state),
                "evaluated_dc_offsets": subject.dc_offset_records(transformed),
                "source_state_dict_sha256": source_sha256,
                "evaluated_state_dict_sha256": (
                    subject.state_content_sha256(transformed)
                ),
                "source_checkpoint_sha256_before": checkpoint_sha256,
                "source_checkpoint_sha256_after": checkpoint_sha256,
                "non_dc_state_sha256_before": non_dc_sha256,
                "non_dc_state_sha256_after": non_dc_sha256,
                "validation_count": spec.VALIDATION_COUNT,
                "diagnostic_only": True,
                "affects_formal_gate": False,
                "formal_decision_authority": False,
                "formal_gate_eligible": False,
                **copy.deepcopy(sweep),
                "audit": {
                    "source_state_strict_load": True,
                    "only_requested_dc_state_keys_changed": True,
                    "requested_dc_state_keys_zero": True,
                    "non_zeroed_dc_offsets_preserved": True,
                    "source_state_unchanged": True,
                    "transformed_state_stable_during_inference": True,
                    "non_dc_state_unchanged": True,
                    "closed_interval_validated": True,
                    "derived_checkpoint_written": False,
                    "gpu_memory": {
                        "device_type": "cpu",
                        "max_memory_allocated_bytes": None,
                        "max_memory_reserved_bytes": None,
                    },
                },
            }
        )
    payload = {
        "schema": spec.EVALUATION_SCHEMA,
        "status": "complete",
        "artifact_kind": spec.ARTIFACT_KIND,
        "scope": "evaluation_only_same_checkpoint_counterfactual",
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_eligible": False,
        "formal_gate_components": [],
        "official_test_accessed": False,
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "expected_epochs": spec.EXPECTED_EPOCHS,
        "validation_count": spec.VALIDATION_COUNT,
        "run_directory": str(spec.FORMAL_RUN_DIR),
        "run_identity": artifact_audit["run_identity"],
        "checkpoint_filename": checkpoint_name,
        "checkpoint_role": spec.CHECKPOINT_ROLES[checkpoint_name],
        "checkpoint_epoch": spec.EXPECTED_EPOCHS,
        "source_checkpoint_identity": artifact_audit["checkpoint_identity"],
        "source_state_dict_sha256": source_sha256,
        "source_non_dc_state_sha256": non_dc_sha256,
        "original_dc_offsets": subject.dc_offset_records(source_state),
        "knockout_modes": list(spec.KNOCKOUT_MODES),
        "knockout_specification": spec.fixed_specification(),
        "diagnostic_source_lock_sha256": "b" * 64,
        "knockout_spec_sha256": spec.specification_sha256(),
        "source_binding": source_binding,
        "threshold_contract": spec.threshold_contract(),
        "device_lane": {
            "logical_device": "cuda:0",
            "physical_gpu_index": subject.CHECKPOINT_GPU_LANES[
                checkpoint_name
            ]["physical_gpu_index"],
            "physical_gpu_uuid": subject.CHECKPOINT_GPU_LANES[
                checkpoint_name
            ]["physical_gpu_uuid"],
            "cuda_device_order": subject.CUDA_DEVICE_ORDER,
            "cuda_visible_devices": subject.CHECKPOINT_GPU_LANES[
                checkpoint_name
            ]["physical_gpu_uuid"],
            "cublas_workspace_config": (
                subject.CUBLAS_WORKSPACE_CONFIG_VALUE
            ),
            "pythonhashseed": subject.PYTHONHASHSEED_VALUE,
            "checkpoint": checkpoint_name,
        },
        "artifact_sha256": {"fixture": "c" * 64},
        "evaluations": evaluations,
    }
    return payload, checkpoint, artifact_audit, source_binding


class KnockoutStateTransformTests(unittest.TestCase):
    def test_all_four_modes_zero_exact_requested_keys_only(self) -> None:
        for mode in spec.KNOCKOUT_MODES:
            with self.subTest(mode=mode):
                source = valid_state()
                source_snapshot = {
                    key: value.detach().clone()
                    for key, value in source.items()
                }
                source_sha256 = subject.state_content_sha256(source)
                transformed = subject.transform_state_dict(source, mode)

                self.assertEqual(
                    changed_keys(source, transformed),
                    set(spec.KNOCKOUT_ZERO_KEYS[mode]),
                )
                for key in spec.DC_OFFSET_KEYS:
                    expected = (
                        torch.zeros_like(source[key])
                        if key in spec.KNOCKOUT_ZERO_KEYS[mode]
                        else source[key]
                    )
                    self.assertTrue(
                        torch.equal(transformed[key], expected),
                        msg=f"{mode} produced an unexpected value for {key}",
                    )
                for key in ("encoder.weight", "decoder.bias"):
                    self.assertTrue(torch.equal(transformed[key], source[key]))
                self.assertEqual(
                    subject.non_dc_state_sha256(transformed),
                    subject.non_dc_state_sha256(source),
                )

                # The counterfactual is an in-memory copy.  Building it must
                # never mutate the checkpoint state mapping supplied by the
                # caller.
                self.assertEqual(
                    subject.state_content_sha256(source),
                    source_sha256,
                )
                for key, expected in source_snapshot.items():
                    self.assertTrue(torch.equal(source[key], expected))

    def test_state_and_non_dc_hashes_have_expected_scope(self) -> None:
        source = valid_state()
        transformed = subject.transform_state_dict(source, "zero_dc_stage4")
        self.assertNotEqual(
            subject.state_content_sha256(source),
            subject.state_content_sha256(transformed),
        )
        self.assertEqual(
            subject.non_dc_state_sha256(source),
            subject.non_dc_state_sha256(transformed),
        )

        changed_non_dc = copy.deepcopy(source)
        changed_non_dc["encoder.weight"][0, 0] += 1.0
        self.assertNotEqual(
            subject.non_dc_state_sha256(source),
            subject.non_dc_state_sha256(changed_non_dc),
        )

    def test_dc_offset_records_are_complete_and_json_native(self) -> None:
        records = subject.dc_offset_records(valid_state())
        self.assertEqual(tuple(records), spec.DC_OFFSET_KEYS)
        json.dumps(records, allow_nan=False)
        for key, expected_value in zip(
            spec.DC_OFFSET_KEYS,
            (0.4, -0.3, 0.2),
        ):
            self.assertEqual(records[key]["shape"], [1])
            self.assertEqual(records[key]["dtype"], "float32")
            self.assertAlmostEqual(records[key]["value"], expected_value)
            self.assertRegex(records[key]["tensor_sha256"], r"^[0-9a-f]{64}$")

    def test_missing_extra_wrong_shape_and_wrong_dtype_are_rejected(self) -> None:
        invalid_states: dict[str, OrderedDict[str, torch.Tensor]] = {}

        missing = valid_state()
        del missing["tpd_ner.dc_offsets.3"]
        invalid_states["missing"] = missing

        extra = valid_state()
        extra["tpd_ner.dc_offsets.1"] = torch.tensor(
            [0.1],
            dtype=torch.float32,
        )
        invalid_states["extra"] = extra

        wrong_shape = valid_state()
        wrong_shape["tpd_ner.dc_offsets.4"] = torch.tensor(
            [0.4, 0.5],
            dtype=torch.float32,
        )
        invalid_states["shape"] = wrong_shape

        wrong_dtype = valid_state()
        wrong_dtype["tpd_ner.dc_offsets.2"] = torch.tensor(
            [0.2],
            dtype=torch.float64,
        )
        invalid_states["dtype"] = wrong_dtype

        for label, state in invalid_states.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    subject.transform_state_dict(state, "zero_all_dc")
                with self.assertRaises(ValueError):
                    subject.dc_offset_records(state)

    def test_non_finite_offset_and_unknown_mode_are_rejected(self) -> None:
        non_finite = valid_state()
        non_finite["tpd_ner.dc_offsets.3"] = torch.tensor(
            [float("nan")],
            dtype=torch.float32,
        )
        with self.assertRaises(ValueError):
            subject.transform_state_dict(non_finite, "zero_all_dc")
        with self.assertRaises(ValueError):
            subject.transform_state_dict(valid_state(), "not_a_mode")


class KnockoutPublicationAndCliTests(unittest.TestCase):
    def test_atomic_publication_is_new_file_only(self) -> None:
        payload = {
            "schema": spec.EVALUATION_SCHEMA,
            "status": "complete",
            "diagnostic_only": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "evaluation.json"
            subject.atomic_publish_new(output, payload)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                payload,
            )

            original_bytes = output.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.atomic_publish_new(
                    output,
                    {**payload, "status": "replacement"},
                )
            self.assertEqual(output.read_bytes(), original_bytes)
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
            )

    def test_plan_and_run_are_required_and_mutually_exclusive(self) -> None:
        plan = subject.parse_args(
            ["--plan", "--checkpoint", "best.pth.tar"]
        )
        self.assertTrue(plan.plan)
        self.assertFalse(plan.run)
        self.assertEqual(plan.checkpoint, "best.pth.tar")

        run = subject.parse_args(
            [
                "--run",
                "--checkpoint",
                "best_miou.pth.tar",
                "--device",
                "cuda:0",
            ]
        )
        self.assertFalse(run.plan)
        self.assertTrue(run.run)
        self.assertEqual(run.device, "cuda:0")

        for argv in (
            ["--checkpoint", "best.pth.tar"],
            [
                "--plan",
                "--run",
                "--checkpoint",
                "best.pth.tar",
            ],
            ["--plan"],
            ["--run", "--checkpoint", "unsupported.pth.tar"],
            [
                "--plan",
                "--checkpoint",
                "best.pth.tar",
                "--output",
                "/tmp/forbidden.json",
            ],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    subject.parse_args(argv)

    def test_plan_fails_closed_before_formal_or_device_work(self) -> None:
        failure = ValueError("diagnostic source lock is unavailable")
        with (
            mock.patch.object(
                subject.source_freezer,
                "current_source_binding",
                side_effect=failure,
            ),
            mock.patch.object(
                subject.formal_evaluator,
                "validate_run_artifacts",
            ) as validate_formal,
            mock.patch.object(subject.torch.cuda, "is_available") as cuda_probe,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "diagnostic source lock is unavailable",
            ):
                subject.build_plan("best.pth.tar")
        validate_formal.assert_not_called()
        cuda_probe.assert_not_called()

    def test_complete_two_checkpoint_by_four_mode_output_contract(self) -> None:
        row_ids = []
        for checkpoint_name in spec.CHECKPOINTS:
            with self.subTest(checkpoint=checkpoint_name):
                (
                    payload,
                    checkpoint,
                    artifact_audit,
                    source_binding,
                ) = complete_payload_fixture(checkpoint_name)
                with mock.patch.object(
                    subject,
                    "_artifact_hashes",
                    return_value={"fixture": "c" * 64},
                ):
                    ready = subject.validate_evaluation_payload(
                        payload,
                        checkpoint,
                        artifact_audit,
                        source_binding,
                    )
                self.assertEqual(ready["schema"], spec.EVALUATION_SCHEMA)
                self.assertEqual(
                    [
                        evaluation["knockout_mode"]
                        for evaluation in ready["evaluations"]
                    ],
                    list(spec.KNOCKOUT_MODES),
                )
                self.assertTrue(ready["diagnostic_only"])
                self.assertFalse(ready["affects_formal_gate"])
                self.assertFalse(ready["formal_decision_authority"])
                self.assertFalse(ready["formal_gate_eligible"])
                for evaluation in ready["evaluations"]:
                    self.assertEqual(
                        evaluation["validation_count"],
                        spec.VALIDATION_COUNT,
                    )
                    self.assertEqual(
                        set(
                            evaluation[
                                "best_points_under_fa_budget"
                            ]
                        ),
                        set(spec.BUDGET_KEYS),
                    )
                    self.assertTrue(
                        evaluation["final_metric_coverage"][
                            "all_required_metrics_present"
                        ]
                    )
                    row_ids.append(
                        (
                            checkpoint_name,
                            evaluation["knockout_mode"],
                        )
                    )
        self.assertEqual(len(row_ids), spec.EXPECTED_ROW_COUNT)
        self.assertEqual(len(set(row_ids)), spec.EXPECTED_ROW_COUNT)

    def test_payload_validator_rejects_incomplete_or_formal_authority(self) -> None:
        (
            payload,
            checkpoint,
            artifact_audit,
            source_binding,
        ) = complete_payload_fixture("best.pth.tar")
        with self.assertRaises(ValueError):
            subject.validate_evaluation_payload(
                {},
                checkpoint,
                artifact_audit,
                source_binding,
            )

        payload["decision"] = "ACCEPT"
        with (
            mock.patch.object(
                subject,
                "_artifact_hashes",
                return_value={"fixture": "c" * 64},
            ),
            self.assertRaisesRegex(ValueError, "decision fields forbidden"),
        ):
            subject.validate_evaluation_payload(
                payload,
                checkpoint,
                artifact_audit,
                source_binding,
            )

    def test_payload_validator_recomputes_artifact_hash_registry(self) -> None:
        (
            payload,
            checkpoint,
            artifact_audit,
            source_binding,
        ) = complete_payload_fixture("best.pth.tar")
        with (
            mock.patch.object(
                subject,
                "_artifact_hashes",
                return_value={"fixture": "d" * 64},
            ),
            self.assertRaisesRegex(ValueError, "artifact SHA registry"),
        ):
            subject.validate_evaluation_payload(
                payload,
                checkpoint,
                artifact_audit,
                source_binding,
            )

    def test_cuda_lane_is_fixed_to_physical_gpu2_or_gpu3(self) -> None:
        for checkpoint in spec.CHECKPOINTS:
            lane = subject.CHECKPOINT_GPU_LANES[checkpoint]
            environment = {
                "CUDA_DEVICE_ORDER": subject.CUDA_DEVICE_ORDER,
                "CUDA_VISIBLE_DEVICES": lane["physical_gpu_uuid"],
                subject.PHYSICAL_GPU_INDEX_ENV: str(
                    lane["physical_gpu_index"]
                ),
                subject.PHYSICAL_GPU_UUID_ENV: lane["physical_gpu_uuid"],
                subject.CUBLAS_WORKSPACE_CONFIG_ENV: (
                    subject.CUBLAS_WORKSPACE_CONFIG_VALUE
                ),
                subject.PYTHONHASHSEED_ENV: (
                    subject.PYTHONHASHSEED_VALUE
                ),
            }
            with self.subTest(checkpoint=checkpoint):
                with mock.patch.dict(
                    subject.os.environ,
                    environment,
                    clear=True,
                ):
                    observed = subject._validated_cuda_lane(
                        checkpoint,
                        "cuda:0",
                    )
                self.assertEqual(
                    observed["physical_gpu_index"],
                    lane["physical_gpu_index"],
                )
                self.assertEqual(
                    observed["physical_gpu_uuid"],
                    lane["physical_gpu_uuid"],
                )
                self.assertEqual(
                    observed["cublas_workspace_config"],
                    ":4096:8",
                )
                self.assertEqual(observed["pythonhashseed"], "42")

        with mock.patch.dict(subject.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "CUDA_DEVICE_ORDER"):
                subject._validated_cuda_lane(
                    "best.pth.tar",
                    "cuda:0",
                )

    def test_cuda_lane_rejects_missing_or_wrong_determinism_environment(
        self,
    ) -> None:
        checkpoint = "best.pth.tar"
        lane = subject.CHECKPOINT_GPU_LANES[checkpoint]
        base = {
            "CUDA_DEVICE_ORDER": subject.CUDA_DEVICE_ORDER,
            "CUDA_VISIBLE_DEVICES": lane["physical_gpu_uuid"],
            subject.PHYSICAL_GPU_INDEX_ENV: str(
                lane["physical_gpu_index"]
            ),
            subject.PHYSICAL_GPU_UUID_ENV: lane["physical_gpu_uuid"],
            subject.CUBLAS_WORKSPACE_CONFIG_ENV: (
                subject.CUBLAS_WORKSPACE_CONFIG_VALUE
            ),
            subject.PYTHONHASHSEED_ENV: subject.PYTHONHASHSEED_VALUE,
        }
        for name, observed, pattern in (
            (
                subject.CUBLAS_WORKSPACE_CONFIG_ENV,
                None,
                "CUBLAS_WORKSPACE_CONFIG",
            ),
            (
                subject.CUBLAS_WORKSPACE_CONFIG_ENV,
                ":16:8",
                "CUBLAS_WORKSPACE_CONFIG",
            ),
            (subject.PYTHONHASHSEED_ENV, None, "PYTHONHASHSEED"),
            (subject.PYTHONHASHSEED_ENV, "7", "PYTHONHASHSEED"),
        ):
            with self.subTest(name=name, observed=observed):
                environment = dict(base)
                if observed is None:
                    environment.pop(name)
                else:
                    environment[name] = observed
                with (
                    mock.patch.dict(
                        subject.os.environ,
                        environment,
                        clear=True,
                    ),
                    self.assertRaisesRegex(ValueError, pattern),
                ):
                    subject._validated_cuda_lane(
                        checkpoint,
                        "cuda:0",
                    )


if __name__ == "__main__":
    unittest.main()
