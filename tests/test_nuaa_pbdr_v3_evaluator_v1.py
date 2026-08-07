from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments import evaluate_nuaa_pbdr_v3_stage1_v1 as evaluator
from experiments import pbdr_v3_non_regression_gate as gate
from experiments import train_nuaa_pbdr_v3_stage1_v1 as trainer


def _metrics(
    *,
    matched: int = 5,
    total: int = 5,
    fa: float = 0.01,
    miou: float = 0.50,
    niou: float = 0.50,
) -> dict[str, float | int]:
    return {
        "matched_target_count": matched,
        "target_count": total,
        "pd": matched / total if total else 0.0,
        "fa": fa,
        "miou": miou,
        "niou": niou,
    }


def _decision_payload(
    current: dict[str, float | int],
    candidate: dict[str, float | int],
) -> tuple[gate.CertificationDecision, dict[str, object]]:
    decision = gate.certify(
        gate.CertificationMetrics.from_mapping(current),
        gate.CertificationMetrics.from_mapping(candidate),
    )
    return decision, {
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "current": asdict(decision.current),
        "candidate": asdict(decision.candidate),
        "scope": "frozen_internal_validation_split",
    }


def _validation_sweep() -> dict[str, object]:
    current = _metrics()
    fixed_candidate = _metrics(miou=0.503)
    sweep = {
        f"{threshold:.2f}": dict(fixed_candidate)
        for threshold in trainer.THRESHOLDS
    }
    sweep["0.51"] = _metrics(miou=0.60, niou=0.60)
    return {
        "fixed_0_5": {
            "current": current,
            "candidate": fixed_candidate,
        },
        "candidate_threshold_sweep": sweep,
    }


class _FakeRunArtifacts:
    """Small trainer-shaped artifact set for validator-only tests."""

    def __init__(
        self,
        root: Path,
        *,
        selected_threshold: float = 0.51,
    ) -> None:
        self.run_dir = root / "formal" / "best_miou" / "core"
        self.run_dir.mkdir(parents=True)
        self.parent_path = root / "current.pth.tar"
        self.parent_path.write_bytes(b"immutable-current-placeholder")
        self.base_state = {"base.weight": torch.tensor([1.0])}
        self.base_sha = evaluator.models.tensor_mapping_sha256(self.base_state)
        self.parent_record = {
            "path": str(self.parent_path.resolve()),
            "sha256": "1" * 64,
            "bytes": self.parent_path.stat().st_size,
            "state_key_count": 1,
            "state_sha256": self.base_sha,
            "checkpoint_role": "best_miou",
            "epoch": 850,
            "schema": "fake-current/v1",
            "protocol_sha256": "2" * 64,
        }
        self.runtime_sources = {
            "fake/source.py": {
                "path": str((root / "fake_source.py").resolve()),
                "sha256": "3" * 64,
                "bytes": 7,
            }
        }

        official_ids = [f"train-{index:03d}" for index in range(213)]
        self.official_ids = official_ids
        mask_stats = [
            asdict(
                trainer.MaskStats(
                    identifier=identifier,
                    height=32,
                    width=32,
                    target_count=1,
                    target_pixels=64,
                    tiny_target_count=0,
                    minimum_target_area=64,
                    stratum="larger",
                )
            )
            for identifier in official_ids
        ]
        development_ids, validation_ids = trainer.stratified_split(
            [trainer.MaskStats(**record) for record in mask_stats],
            trainer.VAL_FRACTION,
            trainer.SPLIT_SEED,
        )
        self.data_manifest_path = root / "data_protocol_manifest.json"
        self.data_manifest_path.write_text(
            '{"schema": "fake-data-protocol/v1"}\n',
            encoding="utf-8",
        )
        self.data_manifest_binding = {
            "path": str(self.data_manifest_path.resolve()),
            "sha256": evaluator.models.file_sha256(self.data_manifest_path),
        }
        split_unsigned = {
            "schema": "sctransnet_nuaa_pbdr_v3_internal_split/v1",
            "dataset": evaluator.models.DATASET,
            "source_split": "official_train_only",
            "official_test_index_opened": False,
            "official_train_ids": official_ids,
            "development_train_ids": development_ids,
            "internal_validation_ids": validation_ids,
            "split_seed": trainer.SPLIT_SEED,
            "val_fraction": trainer.VAL_FRACTION,
            "official_train_index_sha256": evaluator.data_protocol.EXPECTED_SPLITS[
                evaluator.models.DATASET
            ]["train"]["file_sha256"],
            "mask_stats": mask_stats,
            "data_protocol_manifest": self.data_manifest_binding,
        }
        self.split_sha = evaluator.models.canonical_sha256(split_unsigned)
        self.split = dict(split_unsigned, split_sha256=self.split_sha)
        self.split_path = self.run_dir / "split_manifest.json"
        self.split_path.write_text(
            json.dumps(self.split, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        source_locks = {
            "parent_checkpoint": self.parent_record["sha256"],
            "split_manifest": self.split_sha,
            "protocol_document": evaluator.models.file_sha256(
                evaluator.PROTOCOL_DOCUMENT
            ),
            "runtime_sources": self.runtime_sources,
        }
        protocol = {
            "schema": trainer.SCHEMA,
            "mode": "formal",
            "dataset": evaluator.models.DATASET,
            "training_seed": trainer.TRAINING_SEED,
            "parent_role": "best_miou",
            "recipe": "core",
            "epochs": trainer.FORMAL_EPOCHS,
            "eval_every": trainer.FORMAL_EVAL_EVERY,
            "batch_size": trainer.FORMAL_BATCH_SIZE,
            "workers": trainer.FORMAL_WORKERS,
            "device": "cuda:0",
            "expected_gpu_uuid": trainer.GPU0_UUID,
            "precision": "fp32",
            "fixed_threshold": trainer.FORMAL_THRESHOLD,
            "threshold_grid": list(trainer.THRESHOLDS),
            "split_seed": trainer.SPLIT_SEED,
            "val_fraction": trainer.VAL_FRACTION,
            "data_root": str(root.resolve()),
            "data_protocol_manifest": self.data_manifest_binding,
            "smoke_limits": {
                "max_train_images": None,
                "max_val_images": None,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": trainer.FORMAL_LR,
                "weight_decay": trainer.FORMAL_WEIGHT_DECAY,
            },
            "official_test_accessed": False,
            "model": {"parent_checkpoint": self.parent_record},
            "source_locks": source_locks,
            "freeze_before": {
                "trainable_parameter_count": 6018,
                "base_state_sha256": self.base_sha,
                "batchnorm_buffer_sha256": "4" * 64,
            },
        }
        protocol["protocol_sha256"] = evaluator.models.canonical_sha256(protocol)
        self.protocol = protocol
        self.protocol_path = self.run_dir / "protocol.json"
        self.protocol_path.write_text(
            json.dumps(protocol, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        validation = _validation_sweep()
        current = validation["fixed_0_5"]["current"]
        fixed_candidate = validation["fixed_0_5"]["candidate"]
        decision, decision_payload = _decision_payload(current, fixed_candidate)
        recomputed_threshold, threshold_selection = (
            trainer.select_validation_threshold("best_miou", validation)
        )
        if selected_threshold != recomputed_threshold:
            # Model a jointly edited summary/checkpoint that remains internally
            # self-consistent but is not derived from the validation sweep.
            threshold_selection = {
                "selection_source": "internal_validation_only",
                "threshold": selected_threshold,
                "forged_for_contract_test": True,
            }

        self.candidate_path = self.run_dir / "selected_candidate.pth.tar"
        candidate = {
            "schema": trainer.SCHEMA,
            "epoch": 5,
            "parent_role": "best_miou",
            "recipe": "core",
            "state_dict": {
                "base.weight": self.base_state["base.weight"].clone(),
                "pbdr_v3.weight": torch.tensor([0.0]),
            },
            "validation": validation,
            "internal_certification_fixed_0_5": decision_payload,
            "selected_threshold": selected_threshold,
            "threshold_selection": threshold_selection,
            "protocol_sha256": protocol["protocol_sha256"],
            "source_locks": source_locks,
            "parent_checkpoint": self.parent_record,
            "base_state_sha256": self.base_sha,
            "batchnorm_buffer_sha256_before": "4" * 64,
            "batchnorm_buffer_sha256_after": "4" * 64,
        }
        torch.save(candidate, self.candidate_path)
        self.candidate = candidate

        summary = {
            "schema": trainer.SCHEMA,
            "status": "complete",
            "dataset": evaluator.models.DATASET,
            "seed": trainer.TRAINING_SEED,
            "parent_role": "best_miou",
            "recipe": "core",
            "selected_epoch": 5,
            "internal_gate_passed": decision.passed,
            "internal_certification_fixed_0_5": decision_payload,
            "selected_threshold": selected_threshold,
            "threshold_selection": threshold_selection,
            "official_test_accessed": False,
            "selected_checkpoint": {
                "path": str(self.candidate_path.resolve()),
                "sha256": evaluator.models.file_sha256(self.candidate_path),
            },
            "protocol_sha256": protocol["protocol_sha256"],
            "base_state_sha256_before_after": [self.base_sha, self.base_sha],
            "batchnorm_buffer_sha256_before_after": ["4" * 64, "4" * 64],
        }
        self.summary_path = self.run_dir / "summary.json"
        self.summary_path.write_text(
            json.dumps(summary, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        gate.write_decision(self.run_dir / "internal_certification.json", decision)

        self.expected_splits = copy.deepcopy(
            evaluator.data_protocol.EXPECTED_SPLITS
        )
        self.expected_splits[evaluator.models.DATASET]["train"][
            "ordered_ids_sha256"
        ] = evaluator.data_protocol.ordered_ids_sha256(official_ids)

    @contextmanager
    def validator_patches(self):
        with (
            mock.patch.object(
                evaluator.models,
                "runtime_source_records",
                return_value=self.runtime_sources,
            ),
            mock.patch.object(
                evaluator.models,
                "load_current_checkpoint",
                return_value=({}, self.base_state, self.parent_record),
            ),
            mock.patch.object(
                evaluator.models,
                "TRAINING_STATE_KEY_COUNT",
                2,
            ),
            mock.patch.object(
                evaluator.data_protocol,
                "EXPECTED_SPLITS",
                self.expected_splits,
            ),
        ):
            yield


def _lightweight_validated(root: Path, *, passed: bool) -> evaluator.ValidatedRun:
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    protocol_path = root / "protocol.json"
    candidate_path = root / "selected_candidate.pth.tar"
    summary_path.write_text("{}\n", encoding="utf-8")
    protocol_path.write_text("{}\n", encoding="utf-8")
    candidate_path.write_bytes(b"candidate")
    split_path = root / "split_manifest.json"
    split_path.write_text("{}\n", encoding="utf-8")
    current = _metrics()
    candidate = _metrics(miou=0.503 if passed else 0.50)
    _, decision = _decision_payload(current, candidate)
    parent_path = root / "current.pth.tar"
    parent_path.write_bytes(b"current")
    return evaluator.ValidatedRun(
        run_dir=root,
        summary_path=summary_path,
        protocol_path=protocol_path,
        split_path=split_path,
        candidate_path=candidate_path,
        summary={},
        protocol={},
        split_manifest={},
        candidate={"epoch": 5},
        candidate_state={},
        candidate_sha256=evaluator.models.file_sha256(candidate_path),
        protocol_sha256="a" * 64,
        parent_role="best_miou",
        recipe="core",
        selected_threshold=0.51,
        internal_decision=decision,
        parent_checkpoint={
            "path": str(parent_path.resolve()),
            "sha256": evaluator.models.file_sha256(parent_path),
            "bytes": parent_path.stat().st_size,
            "checkpoint_role": "best_miou",
        },
        runtime_sources={},
    )


@contextmanager
def _mock_authorized_cuda():
    properties = SimpleNamespace(uuid=trainer.GPU0_UUID)
    with (
        mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": trainer.GPU0_UUID},
            clear=False,
        ),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "device_count", return_value=1),
        mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=properties,
        ),
    ):
        yield


class TestCompletedRunValidation(unittest.TestCase):
    def test_threshold_artifact_accepts_json_equivalent_numpy_scalars(self) -> None:
        """Checkpoint NumPy scalars must equal their JSON-native forms."""

        checkpoint_artifact = {
            "metrics": {
                "matched_tiny_target_count": np.int64(8),
                "tiny_pd": np.float64(1.0),
                "tiny_target_count": np.int64(8),
            }
        }
        summary_artifact = {
            "metrics": {
                "matched_tiny_target_count": 8,
                "tiny_pd": 1.0,
                "tiny_target_count": 8,
            }
        }

        self.assertTrue(
            evaluator._canonical_equal(summary_artifact, checkpoint_artifact)
        )
        forged_artifact = copy.deepcopy(summary_artifact)
        forged_artifact["metrics"]["matched_tiny_target_count"] = "8"
        self.assertFalse(
            evaluator._canonical_equal(summary_artifact, forged_artifact)
        )

    def test_trainer_summary_and_checkpoint_field_shapes_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _FakeRunArtifacts(Path(temporary))
            with artifacts.validator_patches():
                validated = evaluator.validate_completed_run(artifacts.run_dir)
            self.assertEqual(validated.parent_role, "best_miou")
            self.assertEqual(validated.selected_threshold, 0.51)
            self.assertTrue(validated.internal_decision["passed"])

    def test_missing_split_manifest_fails_closed(self) -> None:
        """Protocol stop condition requires a missing split artifact to fail."""

        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _FakeRunArtifacts(Path(temporary))
            artifacts.split_path.unlink()
            with artifacts.validator_patches():
                with self.assertRaises(
                    (FileNotFoundError, evaluator.PBDRV3EvaluationProtocolError)
                ):
                    evaluator.validate_completed_run(artifacts.run_dir)

    def test_selected_threshold_is_recomputed_from_internal_validation(self) -> None:
        """Summary/checkpoint agreement alone must not establish provenance."""

        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _FakeRunArtifacts(
                Path(temporary),
                selected_threshold=0.50,
            )
            with artifacts.validator_patches():
                with self.assertRaises(evaluator.PBDRV3EvaluationProtocolError):
                    evaluator.validate_completed_run(artifacts.run_dir)

    def test_official_train_id_substitution_fails_frozen_ordered_hash(self) -> None:
        """A self-consistent split still cannot substitute an official ID."""

        with tempfile.TemporaryDirectory() as temporary:
            artifacts = _FakeRunArtifacts(Path(temporary))
            split = copy.deepcopy(artifacts.split)
            old_identifier = split["official_train_ids"][0]
            new_identifier = "attacker-substituted-train-id"
            split["official_train_ids"][0] = new_identifier
            for record in split["mask_stats"]:
                if record["identifier"] == old_identifier:
                    record["identifier"] = new_identifier
            for field in (
                "development_train_ids",
                "internal_validation_ids",
            ):
                split[field] = [
                    new_identifier if value == old_identifier else value
                    for value in split[field]
                ]
            split.pop("split_sha256")
            split_sha = evaluator.models.canonical_sha256(split)
            split["split_sha256"] = split_sha
            artifacts.split_path.write_text(
                json.dumps(split, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with artifacts.validator_patches():
                with self.assertRaises(evaluator.PBDRV3EvaluationProtocolError):
                    evaluator._validate_split_manifest(
                        artifacts.run_dir,
                        {"split_manifest": split_sha},
                    )


class TestFallbackAndOfficialAccessBoundary(unittest.TestCase):
    def test_failed_internal_gate_returns_before_device_or_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _lightweight_validated(root, passed=False)
            deployment = root / "deployment.json"
            with (
                mock.patch.object(
                    evaluator,
                    "validate_completed_run",
                    return_value=validated,
                ),
                mock.patch.object(
                    evaluator.torch,
                    "device",
                    side_effect=AssertionError("device path was entered"),
                ),
                mock.patch.object(
                    evaluator,
                    "_evaluate_official_test",
                    side_effect=AssertionError("official test was entered"),
                ),
            ):
                result = evaluator.run(
                    run_dir=root,
                    data_root=root / "does-not-exist",
                    protocol_manifest=root / "does-not-exist.json",
                    device_name="cuda:0",
                    deployment_output=deployment,
                )
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["selected"], "current")
            self.assertFalse(payload["official_test_accessed"])
            self.assertFalse(
                payload["candidate_artifact"]["evaluated_on_official_test"]
            )
            self.assertFalse((root / "evaluation.json").exists())

    def test_existing_outputs_are_rejected_before_official_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _lightweight_validated(root, passed=True)
            evaluation_path = root / "already-present-evaluation.json"
            deployment_path = root / "deployment.json"
            evaluation_path.write_text("reserved\n", encoding="utf-8")
            with (
                mock.patch.object(
                    evaluator,
                    "validate_completed_run",
                    return_value=validated,
                ),
                mock.patch.object(
                    evaluator,
                    "_evaluate_official_test",
                    side_effect=AssertionError("official test was entered"),
                ) as official,
                _mock_authorized_cuda(),
            ):
                with self.assertRaises(FileExistsError):
                    evaluator.run(
                        run_dir=root,
                        device_name="cuda:0",
                        evaluation_output=evaluation_path,
                        deployment_output=deployment_path,
                    )
            official.assert_not_called()

    def test_same_run_cannot_reenter_official_test_before_publication(self) -> None:
        """A run/output lock must cover test access, not just final writes."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _lightweight_validated(root, passed=True)
            evaluation_path = root / "evaluation.json"
            deployment_path = root / "deployment.json"
            entries = 0

            def official_side_effect(*args, **kwargs):
                nonlocal entries
                del args, kwargs
                entries += 1
                if entries > 1:
                    raise AssertionError("official test was entered twice")
                with self.assertRaises(
                    (
                        FileExistsError,
                        BlockingIOError,
                        RuntimeError,
                        evaluator.PBDRV3EvaluationProtocolError,
                    )
                ):
                    evaluator.run(
                        run_dir=root,
                        device_name="cuda:0",
                        evaluation_output=evaluation_path,
                        deployment_output=deployment_path,
                    )
                return ({"schema": evaluator.SCHEMA}, {"selected": "candidate"})

            with (
                mock.patch.object(
                    evaluator,
                    "validate_completed_run",
                    return_value=validated,
                ),
                mock.patch.object(
                    evaluator,
                    "_evaluate_official_test",
                    side_effect=official_side_effect,
                ),
                _mock_authorized_cuda(),
            ):
                evaluator.run(
                    run_dir=root,
                    device_name="cuda:0",
                    evaluation_output=evaluation_path,
                    deployment_output=deployment_path,
                )
            self.assertEqual(entries, 1)

    def test_pass_path_opens_only_nuaa_test_index_once(self) -> None:
        """The NUAA gate must not live-validate other datasets' test indexes."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "protocol-manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            validated = _lightweight_validated(root, passed=True)
            validated = replace(
                validated,
                protocol={"data_root": str(root.resolve())},
                split_manifest={
                    "data_protocol_manifest": {
                        "path": str(manifest_path.resolve()),
                        "sha256": evaluator.models.file_sha256(manifest_path),
                    }
                },
            )
            identifiers = ["test-a", "test-b", "test-c"]
            index_calls: list[tuple[str, str]] = []

            def load_one_index(data_root, dataset, split):
                self.assertEqual(Path(data_root).resolve(), root.resolve())
                index_calls.append((dataset, split))
                return list(identifiers)

            expected_splits = copy.deepcopy(
                evaluator.data_protocol.EXPECTED_SPLITS
            )
            expected_splits[evaluator.models.DATASET]["test"].update(
                {
                    "count": len(identifiers),
                    "ordered_ids_sha256": (
                        evaluator.data_protocol.ordered_ids_sha256(identifiers)
                    ),
                }
            )
            gate_point = {
                "threshold": 0.5,
                "target_count": evaluator.TARGET_COUNT,
                "matched_target_count": 256,
                "fa": 1.0e-5,
                "miou": 0.80,
                "niou": 0.80,
            }

            class EmptyModel(torch.nn.Module):
                pass

            def metric_points(*args, **kwargs):
                del args, kwargs
                return {
                    "0.50": dict(gate_point),
                    "0.51": dict(gate_point, threshold=0.51),
                }

            cache = {
                "candidate_probabilities": [],
                "current_probabilities": [],
                "candidate_losses": [],
                "current_losses": [],
                "targets": [],
                "identifiers": identifiers,
            }
            with (
                mock.patch.object(
                    evaluator.models,
                    "build_inference_model_from_candidate_state",
                    return_value=(
                        EmptyModel(),
                        {
                            "strict_load": True,
                            "base_bitwise_equal_to_parent": True,
                            "inference_state_key_count": (
                                evaluator.models.INFERENCE_STATE_KEY_COUNT
                            ),
                        },
                    ),
                ),
                mock.patch.object(
                    evaluator.data_protocol,
                    "load_frozen_index",
                    side_effect=AssertionError(
                        "whole three-dataset manifest validation was entered"
                    ),
                ) as frozen_loader,
                mock.patch.object(
                    evaluator.data_protocol,
                    "load_index",
                    side_effect=load_one_index,
                ),
                mock.patch.object(
                    evaluator.data_protocol,
                    "EXPECTED_SPLITS",
                    expected_splits,
                ),
                mock.patch.object(
                    evaluator,
                    "_collect_candidate_and_current",
                    return_value=cache,
                ),
                mock.patch.object(
                    evaluator,
                    "_metric_points",
                    side_effect=metric_points,
                ),
            ):
                evaluation, deployment = evaluator._evaluate_official_test(
                    validated,
                    data_root=root,
                    protocol_manifest=manifest_path,
                    device=torch.device("cpu"),
                    workers=0,
                    access_claim_path=root / "official-access-claim.json",
                )
            frozen_loader.assert_not_called()
            self.assertEqual(index_calls, [(evaluator.models.DATASET, "test")])
            self.assertTrue(evaluation["official_test_accessed"])
            self.assertEqual(deployment["selected"], "candidate")


class TestDeviceAndPrecisionContract(unittest.TestCase):
    def test_passed_gate_rejects_cpu_before_official_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _lightweight_validated(root, passed=True)
            with (
                mock.patch.object(
                    evaluator,
                    "validate_completed_run",
                    return_value=validated,
                ),
                mock.patch.object(
                    evaluator,
                    "_evaluate_official_test",
                    side_effect=AssertionError("official test was entered"),
                ) as official,
            ):
                with self.assertRaises(evaluator.PBDRV3EvaluationProtocolError):
                    evaluator.run(run_dir=root, device_name="cpu")
            official.assert_not_called()

    def test_passed_gate_rejects_wrong_visible_gpu_uuid_before_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _lightweight_validated(root, passed=True)
            properties = SimpleNamespace(uuid="GPU-wrong")
            with (
                mock.patch.object(
                    evaluator,
                    "validate_completed_run",
                    return_value=validated,
                ),
                mock.patch.object(
                    evaluator,
                    "_evaluate_official_test",
                    side_effect=AssertionError("official test was entered"),
                ) as official,
                mock.patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "GPU-wrong"},
                    clear=False,
                ),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(torch.cuda, "device_count", return_value=1),
                mock.patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=properties,
                ),
            ):
                with self.assertRaises(evaluator.PBDRV3EvaluationProtocolError):
                    evaluator.run(run_dir=root, device_name="cuda:0")
            official.assert_not_called()

    def test_tf32_is_disabled_before_model_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _lightweight_validated(root, passed=True)
            previous_cudnn = torch.backends.cudnn.allow_tf32
            previous_matmul = torch.backends.cuda.matmul.allow_tf32
            previous_deterministic = torch.are_deterministic_algorithms_enabled()

            class StopAfterPrecisionAudit(Exception):
                pass

            def builder(*args, **kwargs):
                del args, kwargs
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                raise StopAfterPrecisionAudit

            try:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                with mock.patch.object(
                    evaluator.models,
                    "build_inference_model_from_candidate_state",
                    side_effect=builder,
                ):
                    with self.assertRaises(StopAfterPrecisionAudit):
                        evaluator._evaluate_official_test(
                            validated,
                            data_root=root,
                            protocol_manifest=root / "not-reached.json",
                            device=torch.device("cpu"),
                            workers=0,
                            access_claim_path=root / "not-reached-claim.json",
                        )
            finally:
                torch.backends.cudnn.allow_tf32 = previous_cudnn
                torch.backends.cuda.matmul.allow_tf32 = previous_matmul
                torch.use_deterministic_algorithms(previous_deterministic)


class _TwoSampleDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = torch.zeros(1, 2, 2)
        mask = torch.zeros(1, 2, 2)
        return image, mask, (2, 2), f"sample-{index}"


class _SameForwardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.mode = "unset"

    def forward(self, images: torch.Tensor):
        del images
        raise AssertionError("ordinary/separate forward must not be used")

    def forward_for_pbdr_v3_training(self, images: torch.Tensor):
        self.calls += 1
        shape = (images.shape[0], 1, images.shape[-2], images.shape[-1])
        base = torch.full(shape, -2.0, device=images.device)
        routed = torch.full(shape, 2.0, device=images.device)
        return (), SimpleNamespace(base_logits=base, routed_logits=routed)


class TestSameForwardAndOfficialGate(unittest.TestCase):
    def test_candidate_and_exact_current_are_collected_in_one_forward(self) -> None:
        dataset = _TwoSampleDataset()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = _SameForwardModel()
        cache = evaluator._collect_candidate_and_current(
            model,
            loader,
            torch.device("cpu"),
        )
        self.assertEqual(model.calls, len(dataset))
        self.assertEqual(model.mode, "test")
        expected_current = float(torch.sigmoid(torch.tensor(-2.0)))
        expected_candidate = float(torch.sigmoid(torch.tensor(2.0)))
        for current, candidate in zip(
            cache["current_probabilities"],
            cache["candidate_probabilities"],
        ):
            np.testing.assert_allclose(current, expected_current)
            np.testing.assert_allclose(candidate, expected_candidate)

    def test_role_specific_fixed_gate_boundaries_are_exact(self) -> None:
        for role, thresholds in evaluator.OFFICIAL_ROLE_THRESHOLDS.items():
            with self.subTest(role=role):
                at_boundary = {
                    "target_count": thresholds["target_count"],
                    "matched_target_count": thresholds[
                        "minimum_matched_target_count"
                    ],
                    "fa": thresholds["maximum_fa"],
                    "miou": thresholds["minimum_miou"],
                    "niou": thresholds["minimum_niou"],
                }
                decision = evaluator._official_decision(role, at_boundary)
                self.assertTrue(decision["passed"])
                self.assertEqual(decision["selected"], "candidate")
                self.assertEqual(decision["threshold"], 0.5)
                below_count = dict(at_boundary)
                below_count["matched_target_count"] -= 1
                self.assertFalse(
                    evaluator._official_decision(role, below_count)["passed"]
                )

    def test_best_miou_and_best_pd_do_not_share_target_count_gate(self) -> None:
        candidate = {
            "target_count": evaluator.TARGET_COUNT,
            "matched_target_count": 256,
            "fa": 1.0e-5,
            "miou": 0.80,
            "niou": 0.80,
        }
        self.assertTrue(
            evaluator._official_decision("best_miou", candidate)["passed"]
        )
        self.assertFalse(
            evaluator._official_decision("best_pd", candidate)["passed"]
        )


class TestAtomicPublication(unittest.TestCase):
    def test_failed_replacement_keeps_previous_json_and_removes_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "deployment.json"
            destination.write_text('{"old": true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluator._atomic_write_json(
                    destination,
                    {"non_finite": float("nan")},
                    overwrite=True,
                )
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"old": True},
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_atomic_writer_refuses_existing_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "deployment.json"
            evaluator._atomic_write_json(
                destination,
                {"version": 1},
                overwrite=False,
            )
            with self.assertRaises(FileExistsError):
                evaluator._atomic_write_json(
                    destination,
                    {"version": 2},
                    overwrite=False,
                )
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"version": 1},
            )


if __name__ == "__main__":
    unittest.main()
