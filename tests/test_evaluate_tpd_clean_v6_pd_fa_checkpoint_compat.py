from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments import evaluate_tpd_clean_v6_pd_fa_checkpoint_compat as subject


class CheckpointMetricCompatibilityEvaluatorTests(unittest.TestCase):
    def _formal_args(self) -> SimpleNamespace:
        expected = subject.strict.EXPECTED_THRESHOLD_CONFIGURATION
        run_dir = (
            subject.summary.DEFAULT_CANDIDATE_ROOT
            / subject.summary.DATASET
            / subject.summary.PRIMARY_VARIANT
            / f"seed_42_{subject.summary.RUN_TAG}"
        )
        return SimpleNamespace(
            run_dir=run_dir,
            checkpoint="best.pth.tar",
            device="cuda:0",
            expected_epochs=subject.summary.EXPECTED_EPOCHS,
            threshold_min=expected["threshold_min"],
            threshold_max=expected["threshold_max"],
            threshold_step=expected["threshold_step"],
            extra_thresholds=list(expected["extra_thresholds"]),
            tail_logit_step=expected["tail_logit_step"],
            fa_budgets=list(expected["fa_budgets"]),
            match_radius=None,
            tiny_area=None,
            overwrite=False,
        )

    def _formal_argv(self, args: SimpleNamespace) -> list[str]:
        return [
            "--run-dir",
            str(args.run_dir),
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.device,
            "--expected-epochs",
            str(args.expected_epochs),
        ]

    def _make_run(
        self,
        root: Path,
        *,
        checkpoint_epoch: int = 1,
        checkpoint_role: str = "best_validation_pd_primary",
        checkpoint_metrics: dict[str, int | float] | None = None,
        event_changes: dict[str, object] | None = None,
    ) -> tuple[Path, dict[str, int | float], dict[str, object]]:
        run_dir = root / "run"
        run_dir.mkdir()
        selection = {
            "pd": 188 / 189,
            "fa": 1.0e-6,
            "tiny_pd": 1.0,
            "miou": 0.93,
            "val_loss": 0.001,
        }
        audit = {
            "false_objects_per_image": 3 / 133,
            "matched_target_count": 188,
            "matched_tiny_target_count": 39,
            "predicted_object_count": 191,
            "target_count": 189,
            "tiny_target_count": 39,
            "unmatched_predicted_object_count": 3,
            "valid_pixel_count": 8_716_288,
        }
        event: dict[str, object] = {
            "epoch": 1,
            **selection,
            **audit,
            "new_best_pd": True,
            "new_best_miou": True,
        }
        if event_changes:
            event.update(event_changes)
        environment = {
            "pythonhashseed": "42",
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "torch_num_threads": 1,
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cuda_visible_devices": "GPU-test-fixture",
            "device_uuid": "GPU-test-fixture",
        }
        (run_dir / "protocol.json").write_text(
            json.dumps(
                {
                    "arguments": {"epochs": 1, "seed": 42},
                    "run_identity": {
                        "training_contract": {
                            "environment": environment,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "seed": 42,
                    "best_pd_epoch": 1,
                    "best_miou_epoch": 1,
                    "best_pd_validation_metrics": selection,
                    "best_miou_validation_metrics": selection,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics.jsonl").write_text(
            json.dumps(event) + "\n",
            encoding="utf-8",
        )
        metrics = (
            dict(selection)
            if checkpoint_metrics is None
            else dict(checkpoint_metrics)
        )
        torch.save(
            {
                "epoch": checkpoint_epoch,
                "seed": 42,
                "checkpoint_role": checkpoint_role,
                "validation_metrics": metrics,
            },
            run_dir / "best.pth.tar",
        )
        return run_dir, metrics, event

    def _fake_lock(self, _: Path) -> tuple[dict[str, object], str]:
        wrapper_relative = str(
            Path(subject.__file__)
            .resolve()
            .relative_to(subject.REPO_ROOT)
        )
        return (
            {
                "source_sha256": {
                    wrapper_relative: subject.sha256_file(
                        Path(subject.__file__).resolve()
                    )
                }
            },
            "a" * 64,
        )

    def _context(
        self,
        run_dir: Path,
    ) -> dict[str, object]:
        argv = [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best.pth.tar",
            "--expected-epochs",
            "1",
            "--device",
            "cpu",
        ]
        return subject.build_compatibility_context(
            run_dir,
            "best.pth.tar",
            1,
            argv,
            source_lock_validator=self._fake_lock,
        )

    def test_missing_audit_fields_are_added_only_to_temporary_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, original, event = self._make_run(Path(directory))
            context = self._context(run_dir)
            before = copy.deepcopy(original)
            temporary = subject.temporary_checkpoint_metrics_for_audit(
                original, context
            )
            self.assertEqual(original, before)
            self.assertEqual(
                context["supplemented_fields"],
                sorted(context["audit_only_fields"]),
            )
            for key in context["supplemented_fields"]:
                self.assertEqual(temporary[key], event[key])

    def test_selection_value_mismatch_is_rejected_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, _, _ = self._make_run(
                Path(directory),
                event_changes={"pd": 0.5},
            )
            with self.assertRaisesRegex(ValueError, "selection metric mismatch"):
                self._context(run_dir)

    def test_invalid_checkpoint_epoch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, _, _ = self._make_run(
                Path(directory),
                checkpoint_epoch=2,
            )
            with self.assertRaisesRegex(ValueError, "checkpoint epoch"):
                self._context(run_dir)

    def test_invalid_checkpoint_role_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, _, _ = self._make_run(
                Path(directory),
                checkpoint_role="best_validation_miou_secondary",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_role"):
                self._context(run_dir)

    def test_full_legacy_schema_passes_through_without_supplement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, selection, event = self._make_run(root)
            full_metrics = {
                **selection,
                **{
                    key: event[key]
                    for key in subject.REQUIRED_AUDIT_ONLY_FIELDS
                },
            }
            torch.save(
                {
                    "epoch": 1,
                    "seed": 42,
                    "checkpoint_role": "best_validation_pd_primary",
                    "validation_metrics": full_metrics,
                },
                run_dir / "best.pth.tar",
            )
            context = self._context(run_dir)
            temporary = subject.temporary_checkpoint_metrics_for_audit(
                full_metrics, context
            )
            self.assertEqual(context["supplemented_fields"], [])
            self.assertEqual(
                context["preexisting_audit_fields"],
                sorted(subject.REQUIRED_AUDIT_ONLY_FIELDS),
            )
            self.assertEqual(temporary, full_metrics)

    def test_output_provenance_preserves_points_and_checkpoint_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, original, _ = self._make_run(Path(directory))
            context = self._context(run_dir)
            temporary = subject.temporary_checkpoint_metrics_for_audit(
                original, context
            )
            fixed = {
                "threshold": 0.5,
                **temporary,
            }
            fixed_audit = subject._FROZEN_AUDIT(fixed, temporary)
            payload = {
                "checkpoint": Path(context["checkpoint"]["path"]),
                "checkpoint_sha256": context["checkpoint"]["sha256"],
                "checkpoint_epoch": context["checkpoint"]["epoch"],
                "checkpoint_role": context["checkpoint"]["role"],
                "checkpoint_validation_metrics": copy.deepcopy(original),
                "points": [copy.deepcopy(fixed)],
                "fixed_threshold_0_5": copy.deepcopy(fixed),
                "best_points_under_fa_budget": {
                    "1e-06": copy.deepcopy(fixed),
                },
                "fixed_threshold_0_5_checkpoint_audit": fixed_audit,
                "threshold_provenance": {},
                "audit": {
                    "artifact_sha256": {
                        "evaluator": context["base_evaluator_sha256"],
                    }
                },
            }
            points_before = copy.deepcopy(payload["points"])
            metrics_before = copy.deepcopy(
                payload["checkpoint_validation_metrics"]
            )
            subject.enrich_output_payload(
                payload,
                context,
                source_lock_validator=self._fake_lock,
            )
            self.assertEqual(payload["points"], points_before)
            self.assertEqual(
                payload["checkpoint_validation_metrics"], metrics_before
            )
            records = [
                payload["threshold_provenance"][subject.COMPATIBILITY_KEY],
                payload["fixed_threshold_0_5_checkpoint_audit"][
                    subject.COMPATIBILITY_KEY
                ],
                payload["audit"][subject.COMPATIBILITY_KEY],
            ]
            self.assertEqual(records[0], records[1])
            self.assertEqual(records[1], records[2])
            self.assertTrue(records[0]["temporary_audit_copy_only"])
            self.assertEqual(
                records[0]["raw_fixed_threshold_checkpoint_audit"][
                    "max_abs_non_strict_numeric_delta"
                ],
                0.0,
            )
            self.assertEqual(
                payload["fixed_threshold_0_5_checkpoint_audit"][
                    "max_abs_non_strict_numeric_delta"
                ],
                0.0,
            )
            self.assertEqual(
                records[0]["actual_runtime_argv"][1],
                str(Path(subject.__file__).resolve()),
            )

    def test_non_strict_delta_bounds_allow_gpu_loss_roundoff(self) -> None:
        audit = {
            "non_strict_numeric_deltas_sweep_minus_checkpoint": {
                "miou": 0.0,
                "val_loss": 1.5819799565572638e-8,
            },
            "max_abs_non_strict_numeric_delta": 1.5819799565572638e-8,
        }
        subject.validate_non_strict_numeric_deltas(audit)

    def test_non_strict_delta_bounds_reject_excessive_loss_delta(self) -> None:
        audit = {
            "non_strict_numeric_deltas_sweep_minus_checkpoint": {
                "miou": 0.0,
                "val_loss": 1.1e-7,
            },
            "max_abs_non_strict_numeric_delta": 1.1e-7,
        }
        with self.assertRaisesRegex(ValueError, "exceeds bound: val_loss"):
            subject.validate_non_strict_numeric_deltas(audit)

    def test_threshold_invariant_loss_is_normalized_without_task_change(
        self,
    ) -> None:
        checkpoint_loss = 0.001
        raw_loss = checkpoint_loss + 1.0e-8
        delta = raw_loss - checkpoint_loss
        first = {
            "threshold": 0.5,
            "pd": 188 / 189,
            "fa": 1.0e-6,
            "tiny_pd": 1.0,
            "miou": 0.93,
            "val_loss": raw_loss,
        }
        second = {**first, "threshold": 0.6, "pd": 187 / 189}
        payload = {
            "points": [copy.deepcopy(first), copy.deepcopy(second)],
            "fixed_threshold_0_5": copy.deepcopy(first),
            "best_points_under_fa_budget": {
                "1e-06": copy.deepcopy(first),
            },
        }
        audit = {
            "non_strict_numeric_deltas_sweep_minus_checkpoint": {
                "miou": 0.0,
                "val_loss": delta,
            },
            "max_abs_non_strict_numeric_delta": abs(delta),
        }
        before = copy.deepcopy(payload["points"])
        record = subject.normalize_threshold_invariant_val_loss(
            payload,
            {"val_loss": checkpoint_loss},
            audit,
        )
        self.assertEqual(
            [point["val_loss"] for point in payload["points"]],
            [checkpoint_loss, checkpoint_loss],
        )
        self.assertEqual(
            [
                {key: value for key, value in point.items() if key != "val_loss"}
                for point in payload["points"]
            ],
            [
                {key: value for key, value in point.items() if key != "val_loss"}
                for point in before
            ],
        )
        self.assertEqual(record["raw_recomputed_value"], raw_loss)

    def test_direct_wrapper_rejects_overwrite_and_nonfrozen_thresholds(
        self,
    ) -> None:
        args = self._formal_args()
        argv = self._formal_argv(args)
        subject.validate_formal_invocation(args, argv)

        args.overwrite = True
        with self.assertRaisesRegex(ValueError, "overwrite is forbidden"):
            subject.validate_formal_invocation(args, [*argv, "--overwrite"])

        args = self._formal_args()
        args.threshold_step = 0.02
        with self.assertRaisesRegex(ValueError, "threshold argument differs"):
            subject.validate_formal_invocation(args, self._formal_argv(args))


if __name__ == "__main__":
    unittest.main()
