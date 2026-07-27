from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import accept_tpd_clean_v6_formal800_checkpoint_compat_results as acceptance
from experiments import evaluate_pd_fa_sweep as base
from experiments import evaluate_tpd_clean_v6_pd_fa_checkpoint_compat as compat
from experiments import summarize_tpd_clean_v6_formal800 as summary
from experiments import validate_tpd_clean_v6_checkpoint_compatibility as subject


class CompatibilityValidatorTests(unittest.TestCase):
    def _fake_lock(self, _: Path) -> tuple[dict[str, object], str]:
        relative = str(
            Path(compat.__file__).resolve().relative_to(compat.REPO_ROOT)
        )
        return (
            {
                "source_sha256": {
                    relative: compat.sha256_file(Path(compat.__file__))
                }
            },
            "a" * 64,
        )

    def _fixture(
        self, root: Path
    ) -> tuple[Path, Path, dict[str, object]]:
        run_dir = root / "run"
        run_dir.mkdir()
        selection = {
            "pd": 188 / 189,
            "fa": 1.0e-6,
            "tiny_pd": 1.0,
            "miou": 0.93,
            "val_loss": 0.001,
        }
        event: dict[str, object] = {
            "epoch": 1,
            **selection,
            "false_objects_per_image": 3 / 133,
            "matched_target_count": 188,
            "matched_tiny_target_count": 39,
            "predicted_object_count": 191,
            "target_count": 189,
            "tiny_target_count": 39,
            "unmatched_predicted_object_count": 3,
            "valid_pixel_count": 8_716_288,
            "new_best_pd": True,
            "new_best_miou": True,
        }
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
        checkpoint_path = run_dir / "best.pth.tar"
        torch.save(
            {
                "epoch": 1,
                "seed": 42,
                "checkpoint_role": "best_validation_pd_primary",
                "validation_metrics": selection,
            },
            checkpoint_path,
        )
        argv = [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best.pth.tar",
            "--device",
            "cpu",
            "--expected-epochs",
            str(summary.EXPECTED_EPOCHS),
        ]
        context = compat.build_compatibility_context(
            run_dir,
            "best.pth.tar",
            1,
            argv,
            source_lock_validator=self._fake_lock,
        )
        context["formal_inference_determinism"] = dict(environment)
        temporary = compat.temporary_checkpoint_metrics_for_audit(
            selection, context
        )
        fixed = {"threshold": 0.5, **temporary}
        fixed_audit = compat._FROZEN_AUDIT(fixed, temporary)
        payload: dict[str, object] = {
            "variant": summary.PRIMARY_VARIANT,
            "seed": 42,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": context["checkpoint"]["sha256"],
            "checkpoint_epoch": 1,
            "checkpoint_role": "best_validation_pd_primary",
            "checkpoint_validation_metrics": selection,
            "points": [fixed],
            "fixed_threshold_0_5": fixed,
            "best_points_under_fa_budget": {
                "1e-06": dict(fixed),
            },
            "fixed_threshold_0_5_checkpoint_audit": fixed_audit,
            "threshold_provenance": {},
            "audit": {
                "artifact_sha256": {
                    "protocol.json": "0" * 64,
                    "split.json": "1" * 64,
                    "summary.json": "2" * 64,
                    "metrics.jsonl": compat.sha256_file(
                        run_dir / "metrics.jsonl"
                    ),
                    "checkpoint": context["checkpoint"]["sha256"],
                    "evaluator": compat.sha256_file(
                        compat.FROZEN_EVALUATOR
                    ),
                }
            },
        }
        compat.enrich_output_payload(
            payload,
            context,
            source_lock_validator=self._fake_lock,
        )
        sweep = run_dir / "pd_fa_sweep_best.pth.json"
        sweep.write_text(
            json.dumps(base.json_ready(payload)),
            encoding="utf-8",
        )
        return run_dir, sweep, event

    def test_read_only_validator_recomputes_all_field_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, sweep, event = self._fixture(Path(directory))
            events = [dict(event) for _ in range(summary.EXPECTED_EPOCHS)]
            with (
                mock.patch.object(
                    compat,
                    "validate_compatibility_source_lock",
                    side_effect=self._fake_lock,
                ),
                mock.patch.object(
                    subject.base,
                    "load_complete_metrics",
                    return_value=events,
                ),
            ):
                result = subject.validate_compatibility_sweep(
                    sweep,
                    run_dir=run_dir,
                    variant=summary.PRIMARY_VARIANT,
                    seed=42,
                    role_name="pd_primary",
                )
            self.assertTrue(result["valid"])
            self.assertEqual(
                result["supplemented_fields"],
                sorted(
                    [
                        "false_objects_per_image",
                        *[
                            key
                            for key in event
                            if key.endswith("_count")
                        ],
                    ]
                ),
            )

    def test_validator_rejects_a_tampered_field_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, sweep, event = self._fixture(Path(directory))
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            for location in (
                payload["threshold_provenance"],
                payload["fixed_threshold_0_5_checkpoint_audit"],
                payload["audit"],
            ):
                location[compat.COMPATIBILITY_KEY][
                    "audit_only_field_sources"
                ]["target_count"]["value"] = 999
            sweep.write_text(json.dumps(payload), encoding="utf-8")
            events = [dict(event) for _ in range(summary.EXPECTED_EPOCHS)]
            with (
                mock.patch.object(
                    compat,
                    "validate_compatibility_source_lock",
                    side_effect=self._fake_lock,
                ),
                mock.patch.object(
                    subject.base,
                    "load_complete_metrics",
                    return_value=events,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "field source differs"
                ):
                    subject.validate_compatibility_sweep(
                        sweep,
                        run_dir=run_dir,
                        variant=summary.PRIMARY_VARIANT,
                        seed=42,
                        role_name="pd_primary",
                    )

    def test_final_acceptance_runs_frozen_acceptance_first(self) -> None:
        order: list[str] = []
        frozen_result = {
            "authoritative_result_accepted": True,
            "decision": "KEEP",
            "engineering_gate_passed": True,
            "ner_stage_authorized": True,
            "strict_valid_sweeps": 8,
            "supplemental_source_lock_sha256": "1" * 64,
            "completion_manifest_sha256": "2" * 64,
            "completion_marker_sha256": "3" * 64,
        }
        compatibility_result = {
            "complete_and_compatibility_valid": True,
            "valid_sweeps": 8,
            "compatibility_source_lock_sha256": "4" * 64,
        }

        def frozen() -> dict[str, object]:
            order.append("frozen")
            return frozen_result

        def compatibility() -> dict[str, object]:
            order.append("compatibility")
            return compatibility_result

        with (
            mock.patch.object(
                acceptance.frozen_acceptance,
                "verify_and_accept",
                side_effect=frozen,
            ),
            mock.patch.object(
                acceptance.compat_validation,
                "validate_all_compatibility_sweeps",
                side_effect=compatibility,
            ),
        ):
            result = acceptance.verify_and_accept()
        self.assertEqual(order, ["frozen", "compatibility"])
        self.assertTrue(result["authoritative_result_accepted"])
        self.assertEqual(result["compatibility_valid_sweeps"], 8)

    def test_failed_frozen_acceptance_stops_before_compatibility(self) -> None:
        with (
            mock.patch.object(
                acceptance.frozen_acceptance,
                "verify_and_accept",
                side_effect=RuntimeError("old rejection"),
            ),
            mock.patch.object(
                acceptance.compat_validation,
                "validate_all_compatibility_sweeps",
            ) as compatibility,
        ):
            with self.assertRaisesRegex(RuntimeError, "old rejection"):
                acceptance.verify_and_accept()
        compatibility.assert_not_called()


if __name__ == "__main__":
    unittest.main()
