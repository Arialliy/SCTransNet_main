from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from experiments import tpd_exact_runner as exact_runner
from experiments import evaluate_tpd_ner_v8_mprs_dch_v2_pd_fa as subject
from experiments import train_tpd_ner_v8_mprs_dch_v2_exact as exact


def source_binding(training_sha256: str = "1" * 64) -> dict:
    return {
        "schema": subject.EVALUATION_SOURCE_BINDING_SCHEMA,
        "training_source_lock": {
            "path": "/fixture/training-lock.json",
            "sha256": training_sha256,
        },
        "acceptance_source_lock": {
            "path": "/fixture/acceptance-lock.json",
            "sha256": "2" * 64,
            "training_source_lock_sha256": training_sha256,
        },
        "evaluator": {
            "path": "/fixture/evaluator.py",
            "relative_path": "experiments/evaluator.py",
            "sha256": "3" * 64,
        },
        "shared_metric_core": {
            "path": "/fixture/metric-core.py",
            "relative_path": "experiments/metric-core.py",
            "sha256": "4" * 64,
        },
        "closed_interval_core": {
            "path": "/fixture/closed-core.py",
            "relative_path": "experiments/closed-core.py",
            "sha256": "5" * 64,
        },
    }


def identifier_sha256(identifiers: list[str]) -> str:
    content = "\n".join(sorted(identifiers)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def fixed_point(matched: int, *, fa: float, miou: float) -> dict:
    return {
        "threshold": 0.5,
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": 0.0,
    }


def budget_points(strict_count: int, other_count: int) -> dict:
    result = {}
    for index, (budget, key) in enumerate(
        zip(subject.FA_BUDGETS, subject.BUDGET_KEYS)
    ):
        matched = strict_count if index == 0 else other_count
        result[key] = {
            "budget": budget,
            "matched_target_count": matched,
            "target_count": 189,
            "pd": matched / 189,
            "achieved_fa": min(budget, 5e-7),
            "threshold": 0.5,
        }
    return result


def raw_point(
    threshold: float,
    *,
    matched: int = 187,
    miou: float = 0.95,
    fa: float = 0.0,
) -> dict:
    matched_tiny = 39 if matched else 0
    return {
        "val_loss": 0.001,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 1.0 if matched else 0.0,
        "pixel_recall": matched / 189,
        "pixel_f1": matched / 189,
        "pd": matched / 189,
        "tiny_pd": matched_tiny / 39,
        "fa": fa,
        "false_objects_per_image": 0.0,
        "target_count": 189,
        "matched_target_count": matched,
        "tiny_target_count": 39,
        "matched_tiny_target_count": matched_tiny,
        "predicted_object_count": matched,
        "unmatched_predicted_object_count": 0,
        "valid_pixel_count": 8716288,
        "threshold": threshold,
    }


def raw_sweep_fixture(
    quantile_keys: tuple[str, ...] = subject.EMPIRICAL_QUANTILE_KEYS,
) -> tuple[dict, dict]:
    base = {
        round(0.01 + index * 0.01, 10)
        for index in range(99)
    }
    base.update(subject.EXTRA_THRESHOLDS)
    tail_start = math.log(0.95 / (1.0 - 0.95))
    tail_end = math.log(0.9999 / (1.0 - 0.9999))
    tail = {
        1.0 / (1.0 + math.exp(-(tail_start + index * 0.1)))
        for index in range(64)
    }
    quantiles = {
        key: (index + 1) * 1e-5
        for index, key in enumerate(quantile_keys)
    }
    thresholds = sorted(
        {
            *base,
            *tail,
            *quantiles.values(),
            subject.LAST_FLOAT32_BELOW_ONE,
            subject.UPPER_BOUNDARY_THRESHOLD,
        }
    )
    points = [raw_point(threshold) for threshold in thresholds]
    points[-1] = raw_point(
        subject.UPPER_BOUNDARY_THRESHOLD,
        matched=0,
        miou=0.0,
        fa=0.0,
    )
    fixed = next(point for point in points if point["threshold"] == 0.5)
    checkpoint_metrics = {
        key: value for key, value in fixed.items() if key != "threshold"
    }
    provenance = {
        "uniform_probability_grid_count": 105,
        "tail_logit_range": [tail_start, tail_end],
        "tail_logit_step": 0.1,
        "tail_logit_threshold_count": 64,
        "empirical_score_quantiles": quantiles,
        "total_unique_threshold_count": len(points),
        "posthoc_endpoint_completion": False,
        "preregistered_endpoint_completion": True,
        "endpoint_protocol_stage": "before_formal_training",
        "closed_probability_interval": True,
        "score_dtype": "float32",
        "score_count": 8716288,
        "added_thresholds": [
            subject.LAST_FLOAT32_BELOW_ONE,
            subject.UPPER_BOUNDARY_THRESHOLD,
        ],
        "last_float32_below_one": subject.LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": subject.UPPER_BOUNDARY_THRESHOLD,
        "upper_boundary_comparison": "prediction > threshold",
        "upper_boundary_semantics": "empty_prediction_pd0_fa0",
    }
    payload = {
        "validation_count": 133,
        "checkpoint_validation_metrics": checkpoint_metrics,
        "threshold_provenance": provenance,
        "points": points,
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "fixed_threshold_0_5_checkpoint_audit": (
            subject._fixed_threshold_checkpoint_audit(
                fixed,
                checkpoint_metrics,
            )
        ),
        "best_points_under_fa_budget": {
            key: copy.deepcopy(fixed) for key in subject.BUDGET_KEYS
        },
    }
    artifact = {
        "validation_count": 133,
        "checkpoint_validation_metrics": checkpoint_metrics,
    }
    return payload, artifact


class V2EvaluatorContractTests(unittest.TestCase):
    def test_pd_primary_absolute_gate_boundary_passes(self) -> None:
        gate = subject._absolute_gate(
            "best_validation_pd_primary",
            fixed_point(188, fa=1e-6, miou=0.933647),
            budget_points(187, 188),
        )
        self.assertTrue(gate["absolute_checkpoint_gate_passed"])
        self.assertFalse(gate["formal_success_claim_authorized"])

    def test_miou_secondary_absolute_gate_boundary_passes(self) -> None:
        gate = subject._absolute_gate(
            "best_validation_miou_secondary",
            fixed_point(187, fa=1e-6, miou=0.946542),
            budget_points(187, 188),
        )
        self.assertTrue(gate["absolute_checkpoint_gate_passed"])

    def test_each_absolute_axis_is_required(self) -> None:
        cases = {
            "matched": fixed_point(187, fa=1e-6, miou=0.933647),
            "fa": fixed_point(188, fa=1.000001e-6, miou=0.933647),
            "miou": fixed_point(188, fa=1e-6, miou=0.933646),
        }
        for name, fixed in cases.items():
            with self.subTest(name=name):
                gate = subject._absolute_gate(
                    "best_validation_pd_primary",
                    fixed,
                    budget_points(187, 188),
                )
                self.assertFalse(
                    gate["absolute_checkpoint_gate_passed"]
                )
        budgets = budget_points(186, 188)
        self.assertFalse(
            subject._absolute_gate(
                "best_validation_pd_primary",
                fixed_point(188, fa=1e-6, miou=0.933647),
                budgets,
            )["absolute_checkpoint_gate_passed"]
        )

    def test_formal_arguments_fix_single_seed_artifact_contract(self) -> None:
        args = subject.validate_formal_arguments(
            [
                "--run-dir",
                "/tmp/v2-run",
                "--checkpoint",
                "best.pth.tar",
                "--device",
                "cpu",
                "--expected-epochs",
                "800",
            ]
        )
        self.assertEqual(args.expected_epochs, 800)
        with self.assertRaises(ValueError):
            subject.validate_formal_arguments(
                [
                    "--run-dir",
                    "/tmp/v2-run",
                    "--checkpoint",
                    "last.pth.tar",
                ]
            )
        with self.assertRaises(ValueError):
            subject.validate_formal_arguments(
                [
                    "--run-dir",
                    "/tmp/v2-run",
                    "--checkpoint",
                    "best.pth.tar",
                    "--overwrite",
                ]
            )

    def test_builder_rejects_v1_control_identity(self) -> None:
        with self.assertRaises(ValueError):
            subject.build_model(subject.V1_CONTROL, 42)

    def test_gate_contract_is_stable_under_copy(self) -> None:
        observed = subject.performance_gate_contract()
        self.assertEqual(observed, copy.deepcopy(observed))
        self.assertFalse(observed["v1_off_absolute_gate_required"])

    def test_raw_points_recompute_fixed_and_all_budget_summaries(self) -> None:
        payload, artifact = raw_sweep_fixture()
        fixed, budgets = subject._validate_raw_points_and_summaries(
            payload,
            artifact_audit=artifact,
        )
        self.assertEqual(fixed["matched_target_count"], 187)
        self.assertEqual(set(budgets), set(subject.BUDGET_KEYS))

        missing = copy.deepcopy(payload)
        missing["points"].pop(1)
        missing["threshold_provenance"]["total_unique_threshold_count"] -= 1
        with self.assertRaisesRegex(ValueError, "raw/expected threshold count"):
            subject._validate_raw_points_and_summaries(
                missing,
                artifact_audit=artifact,
            )

        negative_fa = copy.deepcopy(payload)
        negative_fa["points"][0]["fa"] = -1e-9
        with self.assertRaisesRegex(ValueError, "fa lies outside"):
            subject._validate_raw_points_and_summaries(
                negative_fa,
                artifact_audit=artifact,
            )

        invalid_miou = copy.deepcopy(payload)
        invalid_miou["points"][0]["miou"] = 1.01
        with self.assertRaisesRegex(ValueError, "miou lies outside"):
            subject._validate_raw_points_and_summaries(
                invalid_miou,
                artifact_audit=artifact,
            )

        false_fixed = copy.deepcopy(payload)
        false_fixed["fixed_threshold_0_5"]["miou"] = 0.99
        with self.assertRaisesRegex(ValueError, "fixed_threshold_0_5/raw point"):
            subject._validate_raw_points_and_summaries(
                false_fixed,
                artifact_audit=artifact,
            )

        false_budget = copy.deepcopy(payload)
        false_budget["best_points_under_fa_budget"][
            subject.BUDGET_KEYS[0]
        ] = copy.deepcopy(payload["points"][-1])
        with self.assertRaisesRegex(ValueError, "raw optimum"):
            subject._validate_raw_points_and_summaries(
                false_budget,
                artifact_audit=artifact,
            )

    def test_raw_points_accept_saturated_empirical_quantile_subsequence(
        self,
    ) -> None:
        payload, artifact = raw_sweep_fixture(
            subject.EMPIRICAL_QUANTILE_KEYS[:7]
        )
        self.assertEqual(len(payload["points"]), 178)
        subject._validate_raw_points_and_summaries(
            payload,
            artifact_audit=artifact,
        )

        noncontiguous, artifact = raw_sweep_fixture(
            (
                subject.EMPIRICAL_QUANTILE_KEYS[0],
                subject.EMPIRICAL_QUANTILE_KEYS[2],
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "empirical quantile key order",
        ):
            subject._validate_raw_points_and_summaries(
                noncontiguous,
                artifact_audit=artifact,
            )

    def test_standard_integrity_keys_are_exact_and_all_true(self) -> None:
        audit = {
            "integrity_checks_passed": {
                name: True for name in subject.REQUIRED_INTEGRITY_CHECKS
            }
        }
        subject._validate_standard_integrity(audit)
        missing = copy.deepcopy(audit)
        missing["integrity_checks_passed"].pop(
            next(iter(subject.REQUIRED_INTEGRITY_CHECKS))
        )
        with self.assertRaisesRegex(ValueError, "integrity check keys"):
            subject._validate_standard_integrity(missing)
        false_value = copy.deepcopy(audit)
        false_value["integrity_checks_passed"][
            next(iter(subject.REQUIRED_INTEGRITY_CHECKS))
        ] = False
        with self.assertRaisesRegex(ValueError, "integrity checks are incomplete"):
            subject._validate_standard_integrity(false_value)

    def test_source_binding_rejects_stale_shared_metric_core(self) -> None:
        training = Path("/fixture/v2-training-lock.json")
        acceptance = Path("/fixture/v2-acceptance-lock.json")
        evaluator = Path(subject.__file__).resolve()
        source_hashes = {
            evaluator: "3" * 64,
            subject.BASE_EVALUATOR_PATH.resolve(): "4" * 64,
            subject.CLOSED_INTERVAL_CORE_PATH.resolve(): "5" * 64,
        }
        acceptance_sources = {
            str(path.relative_to(subject.REPO_ROOT)): digest
            for path, digest in source_hashes.items()
        }
        payloads = {
            training: {
                "schema": exact.EXACT_SOURCE_LOCK_SCHEMA,
                "lock_kind": "training",
            },
            acceptance: {
                "schema": subject.v2_freeze.ACCEPTANCE_SCHEMA,
                "lock_kind": "acceptance",
                "training_source_lock_sha256": "1" * 64,
                "source_sha256": acceptance_sources,
            },
        }

        def load(path: Path) -> dict:
            return copy.deepcopy(payloads[Path(path)])

        def digest(path: Path) -> str:
            path = Path(path)
            if path == training:
                return "1" * 64
            if path == acceptance:
                return "2" * 64
            return source_hashes[path.resolve()]

        with (
            mock.patch.object(
                subject.v2_freeze,
                "DEFAULT_TRAINING_LOCK",
                training,
            ),
            mock.patch.object(
                subject.v2_freeze,
                "DEFAULT_ACCEPTANCE_LOCK",
                acceptance,
            ),
            mock.patch.object(subject, "_load_json", side_effect=load),
            mock.patch.object(subject, "_sha256_file", side_effect=digest),
        ):
            binding = subject._current_evaluation_source_binding()
            self.assertEqual(
                binding["shared_metric_core"]["sha256"],
                "4" * 64,
            )
            payloads[acceptance]["source_sha256"][
                "experiments/evaluate_pd_fa_sweep.py"
            ] = "9" * 64
            with self.assertRaisesRegex(
                ValueError,
                "acceptance source binding",
            ):
                subject._current_evaluation_source_binding()

    def test_checkpoint_sha_guard_rejects_preflight_to_evaluator_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / "best.pth.tar"
            checkpoint.write_bytes(b"before")
            audit = {
                "run_directory": str(run_dir),
                "checkpoint_filename": checkpoint.name,
                "checkpoint_sha256": subject._sha256_file(checkpoint),
            }
            subject._require_preflight_checkpoint_unchanged(
                audit,
                stage="before base evaluator",
            )
            checkpoint.write_bytes(b"after")
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint SHA after base evaluator differs",
            ):
                subject._require_preflight_checkpoint_unchanged(
                    audit,
                    stage="after base evaluator",
                )

    def test_exact_checkpoint_adapter_is_accepted_end_to_end(self) -> None:
        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([1.0]))

        class TinyScaler:
            def state_dict(self) -> dict:
                return {"scale": 1.0}

        args = exact.parse_args(
            [
                "--variant",
                subject.VARIANT,
                "--device",
                "cuda:0",
                "--fresh",
            ]
        )
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        manifest = {
            "schema": exact.ARCHITECTURE_MANIFEST_SCHEMA,
            "variant": subject.VARIANT,
            "relay_enabled": True,
            "relay_version": "v2_rms_centered_arctangent",
            "relay_width": 8,
            "eps": exact.FORMAL_EPS,
        }
        metadata = {
            "variant": subject.VARIANT,
            "parent_variant": "tpd_clean_v8_mprs_dch_full",
            "relay_enabled": True,
            "relay_version": "v2_rms_centered_arctangent",
            "relay_width": 8,
            "relay_initialization_seed": 42,
            "required_control": subject.V1_CONTROL,
            "relay_off_retrained": False,
            "architecture_manifest": manifest,
            "architecture_id": exact.canonical_sha256(manifest),
            "sequence_contract": {
                "evidence_nodes": ("h11", "h12", "h13", "h21", "h22"),
                "nested": (("keep", "context"), ("saliency",)),
            },
        }
        train_ids = [f"train-{index:03d}" for index in range(530)]
        val_ids = [f"val-{index:03d}" for index in range(133)]
        split_records = {
            name: exact_runner.OrderedFingerprint.from_values(name, values)
            for name, values in {
                "full_train": train_ids,
                "full_validation": val_ids,
                "train": train_ids,
                "validation": val_ids,
            }.items()
        }
        data_records = {
            name: exact_runner.OrderedFingerprint.from_values(
                name,
                (f"{name}:fixture",),
            )
            for name in (
                "official_training_data",
                "train_samples",
                "validation_samples",
                "normalization",
            )
        }
        selection_policy = exact_runner.pd_miou_selection_policy(
            stored_metrics=exact.STORED_VALIDATION_METRICS
        ).normalized()
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.cuda, "device_count", return_value=0),
            mock.patch.object(torch.cuda, "get_rng_state_all"),
        ):
            spec = exact.make_exact_run_spec(
                args,
                model=model,
                model_metadata=metadata,
                optimizer=optimizer,
                scaler=TinyScaler(),
                initialization_contract=(
                    exact_runner.fresh_initialization_contract()
                ),
                initial_model_state_sha256=(
                    exact_runner.initial_model_state_sha256(model)
                ),
                initial_rng=exact_runner.initial_rng_contract(),
                selection_policy=selection_policy,
                source_locks={exact.SOURCE_LOCK_KEY: "1" * 64},
                split_records=split_records,
                data_records=data_records,
                environment={
                    "device_type": "cuda",
                    "logical_device": "cuda:0",
                    "visible_cuda_device_count": 1,
                    "device_name": "NVIDIA GeForce RTX 5090",
                    "device_uuid": exact.PHYSICAL_GPU_UUIDS["2"],
                    "cuda_visible_devices": exact.PHYSICAL_GPU_UUIDS["2"],
                    "physical_gpu_index": 2,
                    "physical_gpu_uuid": exact.PHYSICAL_GPU_UUIDS["2"],
                    "physical_gpu_assignment_source": (
                        "verified_v2_ner_worker_environment"
                    ),
                    "pythonhashseed": "42",
                    "cublas_workspace_config": (
                        exact.FORMAL_CUBLAS_WORKSPACE_CONFIG
                    ),
                },
            )
        identity = exact_runner.build_run_identity(model, spec)
        split_hashes = {
            "full_internal_train_sha256": identifier_sha256(train_ids),
            "full_internal_val_sha256": identifier_sha256(val_ids),
            "used_train_sha256": identifier_sha256(train_ids),
            "used_val_sha256": identifier_sha256(val_ids),
        }
        metrics = {
            name: 0 if name.endswith("_count") else 0.0
            for name in exact.STORED_VALIDATION_METRICS
        }
        metrics.update(
            {
                "target_count": 189,
                "matched_target_count": 188,
                "tiny_target_count": 39,
                "matched_tiny_target_count": 39,
                "predicted_object_count": 188,
                "unmatched_predicted_object_count": 0,
                "valid_pixel_count": 8716288,
                "val_loss": 0.001,
                "miou": 0.94,
                "niou": 0.93,
                "pixel_precision": 0.95,
                "pixel_recall": 0.96,
                "pixel_f1": 2 * 0.95 * 0.96 / (0.95 + 0.96),
                "pd": 188 / 189,
                "tiny_pd": 1.0,
                "fa": 5e-7,
                "false_objects_per_image": 0.0,
            }
        )
        checkpoint = dict(
            exact.EvaluatorCheckpointAdapter(
                model_metadata=metadata,
                split_hashes=split_hashes,
            )(
                exact_runner.CompatibilityPayloadContext(
                    role="best_validation_pd_primary",
                    epoch=800,
                    metrics=metrics,
                    event={"epoch": 800, **metrics},
                    exact_payload={
                        "model": {"state_dict": model.state_dict()},
                        "optimizer": {"state_dict": optimizer.state_dict()},
                        "scaler": {"state_dict": {"scale": 1.0}},
                    },
                    run_identity=identity,
                    normalized_spec=spec.normalized(),
                )
            )
        )
        checkpoint.update(
            {
                "derived_schema": exact_runner.DERIVED_CHECKPOINT_SCHEMA,
                "source_exact_checkpoint_sha256": "4" * 64,
                "state_dict_sha256": exact_runner._state_content_sha256(
                    checkpoint["state_dict"],
                    "fixture state",
                ),
                "optimizer_state_sha256": (
                    exact_runner._state_content_sha256(
                        checkpoint["optimizer"],
                        "fixture optimizer",
                    )
                ),
                "scaler_state_sha256": exact_runner._state_content_sha256(
                    checkpoint["scaler"],
                    "fixture scaler",
                ),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = (
                Path(directory)
                / subject.DATASET
                / subject.VARIANT
                / f"seed_42_{exact.FORMAL_RUN_TAG}"
            )
            run_dir.mkdir(parents=True)
            protocol = exact.protocol_payload(
                args,
                directory=run_dir,
                model_metadata=metadata,
                normalization={"mean": 0.0, "std": 1.0},
                run_identity=identity,
            )
            split = {
                "dataset": subject.DATASET,
                "split_seed": 20260722,
                "full_official_train_count": 663,
                "full_internal_train_count": 530,
                "full_internal_val_count": 133,
                "used_train_count": 530,
                "used_val_count": 133,
                "official_test_accessed": False,
                "full_internal_train_ids": train_ids,
                "full_internal_val_ids": val_ids,
                "used_train_ids": train_ids,
                "used_val_ids": val_ids,
                "hashes": split_hashes,
            }
            earlier_metrics = copy.deepcopy(metrics)
            earlier_metrics.update(
                {
                    "matched_target_count": 187,
                    "predicted_object_count": 188,
                    "unmatched_predicted_object_count": 1,
                    "pd": 187 / 189,
                    "fa": 1e-5,
                    "false_objects_per_image": 1 / 133,
                    "miou": 0.90,
                    "niou": 0.89,
                    "val_loss": 0.01,
                }
            )
            metric_events = []
            for epoch in range(1, 801):
                event_metrics = metrics if epoch == 800 else earlier_metrics
                metric_events.append(
                    {
                        "epoch": epoch,
                        "variant": subject.VARIANT,
                        "train_loss": 1.0,
                        "learning_rate": 1e-3,
                        "processed_train_samples": 530,
                        "epoch_seconds": 1.0,
                        **copy.deepcopy(event_metrics),
                        "new_best_pd": epoch in (1, 800),
                        "new_best_miou": epoch in (1, 800),
                    }
                )
            summary = {
                "schema": exact.COMPLETION_SUMMARY_SCHEMA,
                "status": "complete",
                "variant": subject.VARIANT,
                "dataset": subject.DATASET,
                "seed": 42,
                "split_seed": 20260722,
                "parent_variant": "tpd_clean_v8_mprs_dch_full",
                "relay_enabled": True,
                "relay_version": "v2_rms_centered_arctangent",
                "relay_width": 8,
                "required_control": subject.V1_CONTROL,
                "relay_off_retrained": False,
                "selection_source": "internal_validation_only",
                "official_test_accessed": False,
                "stored_validation_metrics": list(
                    exact.STORED_VALIDATION_METRICS
                ),
                "run_identity": identity,
                "split_hashes": split_hashes,
                "model": metadata,
                "best_epoch": 800,
                "best_pd_epoch": 800,
                "best_miou_epoch": 800,
                "best_validation_metrics": metrics,
                "best_pd_validation_metrics": metrics,
                "best_miou_validation_metrics": metrics,
            }
            for name, payload in (
                ("protocol.json", protocol),
                ("split.json", split),
                ("summary.json", summary),
            ):
                (run_dir / name).write_text(
                    json.dumps(exact.base.json_ready(payload)),
                    encoding="utf-8",
                )
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in metric_events),
                encoding="utf-8",
            )
            torch.save(checkpoint, run_dir / "best.pth.tar")
            with mock.patch.object(
                subject,
                "_current_evaluation_source_binding",
                return_value=source_binding(),
            ):
                audit = subject.validate_run_artifacts(
                    run_dir,
                    "best.pth.tar",
                )
            self.assertEqual(
                audit["checkpoint_sha256"],
                subject._sha256_file(run_dir / "best.pth.tar"),
            )
            wrong_selection = copy.deepcopy(metric_events)
            wrong_selection[798].update(
                {
                    **copy.deepcopy(metrics),
                    "matched_target_count": 189,
                    "predicted_object_count": 189,
                    "pd": 1.0,
                    "fa": 0.0,
                    "miou": 0.99,
                    "val_loss": 0.0001,
                    "new_best_pd": True,
                    "new_best_miou": True,
                }
            )
            wrong_selection[799]["new_best_pd"] = False
            wrong_selection[799]["new_best_miou"] = False
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in wrong_selection),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    subject,
                    "_current_evaluation_source_binding",
                    return_value=source_binding(),
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "checkpoint epoch/global selection differs",
                ),
            ):
                subject.validate_run_artifacts(run_dir, "best.pth.tar")
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in metric_events),
                encoding="utf-8",
            )
            checkpoint["model_metadata"]["sequence_contract"]["nested"] = (
                ("keep", "context"),
                ("changed",),
            )
            torch.save(checkpoint, run_dir / "best.pth.tar")
            with (
                mock.patch.object(
                    subject,
                    "_current_evaluation_source_binding",
                    return_value=source_binding(),
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "checkpoint.model differs",
                ),
            ):
                subject.validate_run_artifacts(run_dir, "best.pth.tar")
            cpu_identity = copy.deepcopy(identity)
            cpu_identity["training_contract"]["environment"].update(
                {
                    "device_type": "cpu",
                    "physical_gpu_index": None,
                    "physical_gpu_uuid": None,
                }
            )
            with self.assertRaisesRegex(
                ValueError,
                "physical GPU 2 or 3",
            ):
                subject._validate_run_identity(
                    cpu_identity,
                    split,
                    source_binding=source_binding(),
                )
            with self.assertRaisesRegex(
                ValueError,
                "run identity/current training source lock differs",
            ):
                subject._validate_run_identity(
                    identity,
                    split,
                    source_binding=source_binding("9" * 64),
                )
        self.assertEqual(audit["variant"], subject.VARIANT)
        self.assertEqual(
            audit["checkpoint_role"],
            "best_validation_pd_primary",
        )


if __name__ == "__main__":
    unittest.main()
