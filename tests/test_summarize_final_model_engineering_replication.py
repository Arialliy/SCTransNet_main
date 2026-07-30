from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import final_model_replication_exact_core as core
from experiments import final_model_replication_seed_contract as seeds
from experiments import freeze_final_model_certification_parent_lock as parent_lock
from experiments import summarize_final_model_engineering_replication as summary
from experiments import watch_final_model_engineering_replication as watcher


def metrics(*, pd: float, fa: float, miou: float) -> dict[str, float | int]:
    return {
        "pd": pd,
        "fa": fa,
        "miou": miou,
        "tiny_pd": 1.0,
        "false_objects_per_image": 0.1,
        "target_count": 189,
        "matched_target_count": 188,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 39,
        "unmatched_predicted_object_count": 2,
        "valid_pixel_count": 1000,
    }


class FinalModelEngineeringSummaryTests(unittest.TestCase):
    def test_delta_uses_d_minus_b_for_all_primary_metrics(self) -> None:
        b = metrics(pd=0.90, fa=0.00002, miou=0.91)
        d = metrics(pd=0.92, fa=0.00001, miou=0.93)
        delta = summary.metric_delta(d, b)
        self.assertAlmostEqual(delta["pd"], 0.02)
        self.assertAlmostEqual(delta["fa"], -0.00001)
        self.assertAlmostEqual(delta["miou"], 0.02)

    def test_metric_projection_requires_pd_fa_miou_and_counts(self) -> None:
        incomplete = metrics(pd=0.9, fa=0.0, miou=0.9)
        incomplete.pop("tiny_target_count")
        with self.assertRaises(summary.EngineeringSummaryError):
            summary.metric_projection(incomplete)

    def test_write_once_verifies_equal_and_rejects_different(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "summary.json"
            payload = {"schema": summary.SCHEMA, "status": "complete"}
            summary.write_once(path, payload)
            summary.write_once(path, payload)
            with self.assertRaises(FileExistsError):
                summary.write_once(
                    path,
                    {"schema": summary.SCHEMA, "status": "changed"},
                )


class FinalModelEngineeringRunStateTests(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(watcher._canonical_pretty_json_bytes(value))

    def _run_directory(
        self,
        root: Path,
        *,
        seed: int = seeds.HISTORICAL_PRESSURE_SEED,
        arm: str = core.ARM_B,
    ) -> Path:
        return watcher.run_directory(root, seed, arm)

    def _protocol(
        self,
        directory: Path,
        *,
        seed: int = seeds.HISTORICAL_PRESSURE_SEED,
        arm: str = core.ARM_B,
    ) -> dict[str, object]:
        definition = core.arm_definition(arm)
        trainer = definition.trainer
        gpu_index = 2 if arm == core.ARM_B else 3
        gpu_uuid = trainer.PHYSICAL_GPU_UUIDS[str(gpu_index)]
        identity = {
            "schema": core.exact_runner.RUN_IDENTITY_SCHEMA,
            "run_id": (
                f"{trainer.RUN_ID_PREFIX}NUDT-SIRST:"
                f"{definition.variant}:seed-{seed}:"
                f"split-{seeds.SPLIT_SEED}:"
                f"{core.ENGINEERING_RUN_TAGS[arm]}"
            ),
            "variant": definition.variant,
            "dataset": "NUDT-SIRST",
            "seed": seed,
            "split_seed": seeds.SPLIT_SEED,
            "source_locks": {
                core.SOURCE_LOCK_KEY: "1" * 64,
                "training_data": parent_lock.TRAINING_DATA_SHA256,
                "survival_target_statistics": "2" * 64,
                "parent_checkpoint": trainer.PARENT_CHECKPOINT_SHA256,
            },
            "training_contract": {
                "initialization_contract": {
                    "mode": core.exact_runner.EXTENSION_PARENT_MODE,
                    "provenance": {
                        "parent_checkpoint_sha256": (
                            trainer.PARENT_CHECKPOINT_SHA256
                        )
                    },
                },
                "manual_lr_schedule": {"total_epochs": 800},
                "determinism": {"loader_generator_seed": seed},
                "environment": {
                    "physical_gpu_index": gpu_index,
                    "physical_gpu_uuid": gpu_uuid,
                },
            },
        }
        replication = {
            "arm": arm,
            "variant": definition.variant,
            "trajectory_seed": seed,
            "split_seed": seeds.SPLIT_SEED,
            "parent_training_seed": seeds.BUILDER_COMPATIBILITY_SEED,
            "parent_checkpoint_sha256": trainer.PARENT_CHECKPOINT_SHA256,
            "parent_state_dict_sha256": trainer.PARENT_STATE_DICT_SHA256,
            "parent_load_count": 1,
            "optimizer_inherited": False,
            "scheduler_inherited": False,
            "all_child_parameters_trainable": True,
        }
        return {
            "schema": trainer.ENTRY_SCHEMA,
            "run_directory": str(directory.resolve()),
            "run_identity": identity,
            "model": {
                "variant": definition.variant,
                "training_seed": seed,
                "replication_contract": replication,
            },
        }

    def _split(self) -> dict[str, object]:
        path = (
            parent_lock.REPO_ROOT
            / parent_lock.D_RUN_ROOT
            / "split.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def _prepare_started_run(
        self,
        root: Path,
        *,
        seed: int = seeds.HISTORICAL_PRESSURE_SEED,
        arm: str = core.ARM_B,
    ) -> tuple[Path, dict[str, object]]:
        directory = self._run_directory(root, seed=seed, arm=arm)
        directory.mkdir(parents=True)
        protocol = self._protocol(directory, seed=seed, arm=arm)
        self._write_json(directory / "protocol.json", protocol)
        self._write_json(directory / "split.json", self._split())
        (directory / "exact_journal").mkdir()
        return directory, protocol

    def test_resolver_returns_parent_warm_start_only_for_absent_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            self.assertEqual(
                watcher.resolve_initialization_mode(
                    root,
                    seeds.HISTORICAL_PRESSURE_SEED,
                    core.ARM_B,
                ),
                "--parent-warm-start",
            )
            run = self._run_directory(root)
            run.parent.mkdir(parents=True)
            run.symlink_to(root / "missing-run")
            with self.assertRaises(watcher.ReplicationRunStateError):
                watcher.resolve_initialization_mode(
                    root,
                    seeds.HISTORICAL_PRESSURE_SEED,
                    core.ARM_B,
                )

    def test_resolver_requires_valid_active_journal_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            self._prepare_started_run(root)
            journal = mock.Mock()
            journal.load_active.return_value = SimpleNamespace(epoch=17)
            with mock.patch.object(
                watcher,
                "ExactEpochJournal",
                return_value=journal,
            ):
                self.assertEqual(
                    watcher.resolve_initialization_mode(
                        root,
                        seeds.HISTORICAL_PRESSURE_SEED,
                        core.ARM_B,
                    ),
                    "--exact-resume",
                )
            journal.load_active.return_value = None
            with (
                mock.patch.object(
                    watcher,
                    "ExactEpochJournal",
                    return_value=journal,
                ),
                self.assertRaisesRegex(
                    watcher.ReplicationRunStateError,
                    "no committed active epoch",
                ),
            ):
                watcher.resolve_initialization_mode(
                    root,
                    seeds.HISTORICAL_PRESSURE_SEED,
                    core.ARM_B,
                )

    def test_resolver_rejects_protocol_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            directory, protocol = self._prepare_started_run(root)
            protocol["run_identity"]["seed"] = 99
            self._write_json(directory / "protocol.json", protocol)
            with self.assertRaisesRegex(
                watcher.ReplicationRunStateError,
                "run identity seed",
            ):
                watcher.resolve_initialization_mode(
                    root,
                    seeds.HISTORICAL_PRESSURE_SEED,
                    core.ARM_B,
                )

    def test_resolver_rejects_noncanonical_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            directory, protocol = self._prepare_started_run(root)
            (directory / "protocol.json").write_text(
                json.dumps(protocol),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                watcher.ReplicationRunStateError,
                "not canonical pretty JSON",
            ):
                watcher.resolve_initialization_mode(
                    root,
                    seeds.HISTORICAL_PRESSURE_SEED,
                    core.ARM_B,
                )

    def test_split_validation_recomputes_locked_identifier_hashes(self) -> None:
        split = self._split()
        watcher._validate_split(split)
        train_ids = split["full_internal_train_ids"]
        self.assertIsInstance(train_ids, list)
        train_ids[0] = "mutated-but-declared-hash-left-unchanged"
        with self.assertRaisesRegex(
            watcher.ReplicationRunStateError,
            "recomputed hash full_internal_train_sha256",
        ):
            watcher._validate_split(split)

    def test_split_validation_rejects_duplicate_identifiers(self) -> None:
        split = self._split()
        train_ids = split["full_internal_train_ids"]
        self.assertIsInstance(train_ids, list)
        train_ids[0] = train_ids[1]
        with self.assertRaisesRegex(
            watcher.ReplicationRunStateError,
            "duplicate identifiers",
        ):
            watcher._validate_split(split)

    def test_resolver_validates_complete_summary_and_800_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            directory, protocol = self._prepare_started_run(root)
            model = protocol["model"]
            complete = {
                "schema": core.arm_definition(
                    core.ARM_B
                ).trainer.COMPLETION_SUMMARY_SCHEMA,
                "status": "complete",
                "variant": core.arm_definition(core.ARM_B).variant,
                "dataset": "NUDT-SIRST",
                "seed": seeds.HISTORICAL_PRESSURE_SEED,
                "split_seed": seeds.SPLIT_SEED,
                "run_identity": protocol["run_identity"],
                "model": model,
                "best_epoch": 2,
                "best_pd_epoch": 2,
                "best_miou_epoch": 3,
                "best_validation_metrics": {},
                "best_pd_validation_metrics": {},
                "best_miou_validation_metrics": {},
            }
            self._write_json(directory / "summary.json", complete)
            with (directory / "metrics.jsonl").open(
                "w",
                encoding="utf-8",
            ) as handle:
                for epoch in range(1, 801):
                    handle.write(json.dumps({"epoch": epoch}) + "\n")
            for name in ("last.pth.tar", "best.pth.tar", "best_miou.pth.tar"):
                (directory / name).write_bytes(b"fixture")
            journal = mock.Mock()
            journal.load_active.return_value = SimpleNamespace(epoch=800)
            with mock.patch.object(
                watcher,
                "ExactEpochJournal",
                return_value=journal,
            ):
                self.assertEqual(
                    watcher.resolve_initialization_mode(
                        root,
                        seeds.HISTORICAL_PRESSURE_SEED,
                        core.ARM_B,
                    ),
                    "--complete",
                )
            complete["status"] = "running"
            self._write_json(directory / "summary.json", complete)
            with (
                mock.patch.object(
                    watcher,
                    "ExactEpochJournal",
                    return_value=journal,
                ),
                self.assertRaisesRegex(
                    watcher.ReplicationRunStateError,
                    "completion summary status",
                ),
            ):
                watcher.resolve_initialization_mode(
                    root,
                    seeds.HISTORICAL_PRESSURE_SEED,
                    core.ARM_B,
                )


if __name__ == "__main__":
    unittest.main()
