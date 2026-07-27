from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    validate_tpd_clean_v7_dch_formal800_completion as subject,
)


def validation_fields(epoch: int = 1) -> dict[str, float | int]:
    return {
        "val_loss": 0.1,
        "miou": 0.9,
        "niou": 0.89,
        "pixel_precision": 0.95,
        "pixel_recall": 0.94,
        "pixel_f1": 0.945,
        "pd": 188 / 189,
        "tiny_pd": 1.0,
        "fa": 1e-6,
        "false_objects_per_image": 0.01,
        "target_count": 189,
        "matched_target_count": 188,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 39,
        "predicted_object_count": 190,
        "unmatched_predicted_object_count": 2,
        "valid_pixel_count": 1_000_000,
    }


class V7DCHFormal800CompletionTests(unittest.TestCase):
    def test_default_acceptance_lock_is_v3_and_v1_v2_are_rejected(self) -> None:
        expected = (
            subject.REPO_ROOT
            / "experiments/tpd_clean_v7_dch_acceptance_source_lock_v3.json"
        )
        self.assertEqual(subject.DEFAULT_ACCEPTANCE_SOURCE_LOCK, expected)
        for relative in (
            subject.locks.SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE,
            subject.locks.SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
        ):
            superseded = subject.REPO_ROOT / relative
            self.assertTrue(superseded.is_file())
            with self.assertRaisesRegex(
                subject.IncompleteArtifact,
                "superseded.*current v3",
            ):
                subject.validate_acceptance_source_lock(superseded)

    def test_gpu_artifacts_require_canonical_nested_environment(self) -> None:
        variant = subject.PRIMARY_VARIANT
        seed = 42
        physical_index, gpu_uuid = subject.GPU_ASSIGNMENTS[(variant, seed)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignment_path = (
                root
                / "lane_assignments"
                / f"{variant}_seed{seed}.json"
            )
            assignment_path.parent.mkdir(parents=True)
            assignment_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "sctransnet_tpd_clean_v7_dch_lane_assignment_v1"
                        ),
                        "variant": variant,
                        "seed": seed,
                        "physical_gpu_index": physical_index,
                        "physical_gpu_uuid": gpu_uuid,
                        "logical_device": "cuda:0",
                        "run_directory": str(
                            subject._run_directory(
                                root,
                                variant,
                                seed,
                            ).resolve()
                        ),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            log_path = root / "logs" / f"{variant}_seed{seed}.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                (
                    "TPDCLEANV7DCH_2X_COMPLETE "
                    f"variant={variant} seed={seed} "
                    f"physical_gpu={physical_index} gpu_uuid={gpu_uuid} "
                    "epochs=800 stored_validation_metrics=17\n"
                ),
                encoding="utf-8",
            )
            environment = {
                "physical_gpu_index": physical_index,
                "physical_gpu_uuid": gpu_uuid,
                "device_uuid": gpu_uuid,
                "cuda_visible_devices": gpu_uuid,
                "logical_device": "cuda:0",
                "physical_gpu_assignment_source": (
                    "verified_worker_environment"
                ),
            }
            canonical = {
                "run_identity": {
                    "training_contract": {"environment": environment}
                }
            }
            result = subject._validate_gpu_run_artifacts(
                root,
                canonical,
                variant=variant,
                seed=seed,
            )
            self.assertEqual(result["physical_gpu_index"], physical_index)
            self.assertEqual(result["physical_gpu_uuid"], gpu_uuid)

            noncanonical = {
                "run_identity": {"environment": environment}
            }
            with self.assertRaisesRegex(
                subject.IncompleteArtifact,
                "run environment is missing",
            ):
                subject._validate_gpu_run_artifacts(
                    root,
                    noncanonical,
                    variant=variant,
                    seed=seed,
                )

    def test_native_validation_schema_requires_all_17_fields(self) -> None:
        metrics = validation_fields()
        self.assertEqual(len(subject._validation_metrics(metrics, "fixture")), 17)
        metrics.pop("pixel_f1")
        with self.assertRaisesRegex(subject.IncompleteArtifact, "17-field"):
            subject._validation_metrics(metrics, "fixture")

    def test_training_inspector_requires_four_native_800_epoch_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in subject.SEEDS:
                for variant in subject.VARIANTS:
                    run = (
                        root
                        / subject.DATASET
                        / variant
                        / f"seed_{seed}_{subject.RUN_TAG}"
                    )
                    run.mkdir(parents=True)
                    rows = [
                        {
                            "epoch": epoch,
                            "variant": variant,
                            **validation_fields(epoch),
                        }
                        for epoch in range(1, 801)
                    ]
                    (run / "metrics.jsonl").write_text(
                        "".join(
                            json.dumps(row, sort_keys=True) + "\n"
                            for row in rows
                        ),
                        encoding="utf-8",
                    )
                    (run / "summary.json").write_text(
                        json.dumps(
                            {
                                "status": "complete",
                                "stored_validation_metrics": list(
                                    subject.VALIDATION_FIELDS
                                ),
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    for name in subject.CHECKPOINT_SPECS:
                        (run / name).write_bytes(b"checkpoint")
            readiness = subject.inspect_training_readiness(root)
            self.assertTrue(readiness["formal_matrix_complete"])
            self.assertEqual(readiness["observed_checkpoints"], 12)
            self.assertEqual(len(readiness["runs"]), 4)
            for run in readiness["runs"].values():
                self.assertTrue(run["native_17_fields_present"])
                self.assertEqual(run["metrics_rows"], 800)

    def test_matrix_api_returns_required_counts_integrity_and_run_schema(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluator.py"
            evaluator.write_text("# fixture\n", encoding="utf-8")

            def fake_run(
                candidate_root: Path,
                *,
                variant: str,
                seed: int,
                evaluator_path: Path,
            ) -> dict:
                return {
                    "variant": variant,
                    "seed": seed,
                    "run_directory": str(
                        subject._run_directory(candidate_root, variant, seed)
                    ),
                    "validation_split_sha256": "a" * 64,
                    "checkpoints": {
                        name: {
                            "path": str(root / f"{variant}-{seed}-{name}"),
                            "sha256": "b" * 64,
                            "role": spec["checkpoint_role"],
                        }
                        for name, spec in subject.CHECKPOINT_SPECS.items()
                    },
                    "roles": {
                        role: {
                            "fixed_threshold_0_5": validation_fields(),
                            "budgets": {
                                key: validation_fields()
                                for key in subject.BUDGET_KEYS
                            },
                            "sweep_sha256": "c" * 64,
                        }
                        for role in subject.ROLE_SPECS
                    },
                }

            with mock.patch.object(subject, "_validate_run", side_effect=fake_run):
                matrix = subject.validate_completion_matrix(
                    root,
                    evaluator,
                    acceptance_lock_path=None,
                )
            self.assertTrue(matrix["ready"])
            self.assertEqual(matrix["run_count"], 4)
            self.assertEqual(matrix["checkpoint_count"], 12)
            self.assertEqual(matrix["sweep_count"], 8)
            self.assertEqual(matrix["validation_field_count"], 17)
            self.assertEqual(len(matrix["runs"]), 4)
            self.assertEqual(
                set(matrix["integrity"]),
                {
                    "four_runs_contiguous_800_epochs",
                    "twelve_checkpoints_present_and_strict_load",
                    "eight_closed_interval_sweeps",
                    "model_split_protocol_evaluator_hashes_consistent",
                    "fixed_threshold_reproduction_exact",
                    "all_five_budgets_available",
                    "preregistered_endpoint_provenance",
                    "exact_epoch_journals_complete",
                    "worker_logs_complete_gpu_mapped",
                    "native_17_fields_complete",
                },
            )
            self.assertTrue(all(matrix["integrity"].values()))
            for run in matrix["runs"].values():
                self.assertEqual(run["validation_split_sha256"], "a" * 64)

    def test_manifest_binds_each_input_by_digest_size_and_matrix_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=subject.REPO_ROOT) as directory:
            root = Path(directory)
            first = root / "first.bin"
            second = root / "second.bin"
            lock = root / "lock.json"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            lock.write_bytes(b"lock")
            matrix = {
                "schema": "fixture",
                "candidate_root": str(root),
                "run_count": 4,
                "checkpoint_count": 12,
                "sweep_count": 8,
                "validation_field_count": 17,
                "runs": {},
            }
            records = [
                ("first", "candidate_training", first),
                ("second", "candidate_sweep", second),
            ]
            with mock.patch.object(subject, "_input_paths", return_value=records):
                manifest = subject.build_manifest(
                    matrix,
                    training_source_lock=lock,
                    acceptance_source_lock=lock,
                )
            self.assertEqual(manifest["schema"], subject.MANIFEST_SCHEMA)
            self.assertEqual(
                manifest["matrix_counts"],
                {
                    "runs": 4,
                    "checkpoints": 12,
                    "sweeps": 8,
                    "validation_fields": 17,
                },
            )
            self.assertEqual(manifest["input_count"], 2)
            self.assertEqual(
                manifest["category_counts"],
                {"candidate_training": 1, "candidate_sweep": 1},
            )
            self.assertEqual(
                manifest["inputs"][0]["sha256"],
                subject.sha256_file(first),
            )
            self.assertEqual(manifest["inputs"][0]["size_bytes"], 5)

    def test_publish_is_exclusive_and_marker_binds_three_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / subject.JSON_OUTPUT_NAME
            markdown_path = root / subject.MARKDOWN_OUTPUT_NAME
            manifest_path = root / subject.MANIFEST_NAME
            marker_path = root / subject.MARKER_NAME
            json_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_text("# report\n", encoding="utf-8")
            report = {"decision": "ENGINEERING_GATE_FAIL"}
            matrix = {"runs": {}}
            manifest = {
                "schema": subject.MANIFEST_SCHEMA,
                "input_count": 0,
                "inputs": [],
            }
            with (
                mock.patch.object(
                    subject,
                    "_report_paths",
                    return_value=(
                        json_path,
                        markdown_path,
                        manifest_path,
                        marker_path,
                    ),
                ),
                mock.patch.object(
                    subject,
                    "validate_published_report",
                    return_value=(report, matrix),
                ),
                mock.patch.object(
                    subject,
                    "build_manifest",
                    return_value=manifest,
                ),
            ):
                result = subject.publish_completion(root)
                with self.assertRaises(FileExistsError):
                    subject.publish_completion(root)
            self.assertEqual(result["status"], "complete")
            rows = marker_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[0].endswith(subject.JSON_OUTPUT_NAME))
            self.assertTrue(rows[1].endswith(subject.MARKDOWN_OUTPUT_NAME))
            self.assertTrue(rows[2].endswith(subject.MANIFEST_NAME))

    def test_verify_rejects_tampered_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / subject.JSON_OUTPUT_NAME
            markdown_path = root / subject.MARKDOWN_OUTPUT_NAME
            manifest_path = root / subject.MANIFEST_NAME
            marker_path = root / subject.MARKER_NAME
            json_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_text("# report\n", encoding="utf-8")
            manifest = {
                "schema": subject.MANIFEST_SCHEMA,
                "input_count": 0,
                "inputs": [],
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            marker_path.write_text("0" * 64 + "  bad\n", encoding="utf-8")
            report = {
                "decision": "ENGINEERING_GATE_FAIL",
                "engineering_gate_passed": False,
                "ner_stage_authorized": False,
            }
            with (
                mock.patch.object(
                    subject,
                    "_report_paths",
                    return_value=(
                        json_path,
                        markdown_path,
                        manifest_path,
                        marker_path,
                    ),
                ),
                mock.patch.object(
                    subject,
                    "validate_published_report",
                    return_value=(report, {"runs": {}}),
                ),
                mock.patch.object(
                    subject,
                    "build_manifest",
                    return_value=manifest,
                ),
            ):
                with self.assertRaisesRegex(
                    subject.IncompleteArtifact,
                    "marker",
                ):
                    subject.verify_completion(root)

    def test_cli_requires_explicit_mode(self) -> None:
        self.assertEqual(subject.parse_args(["preflight"]).mode, "preflight")
        self.assertEqual(subject.parse_args(["publish"]).mode, "publish")
        self.assertEqual(subject.parse_args(["verify"]).mode, "verify")
        with self.assertRaises(SystemExit):
            subject.parse_args([])


if __name__ == "__main__":
    unittest.main()
