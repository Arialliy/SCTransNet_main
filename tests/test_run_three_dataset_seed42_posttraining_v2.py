from __future__ import annotations

import copy
import fcntl
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import run_three_dataset_seed42_posttraining_v2 as subject


def _point(*, matched: int = 100, matched_tiny: int = 20) -> dict[str, object]:
    return {
        "threshold": 0.5,
        "miou": 0.7,
        "niou": 0.7,
        "pd": matched / 110,
        "fa": 0.001,
        "target_count": 110,
        "matched_target_count": matched,
        "tiny_target_count": 30,
        "matched_tiny_target_count": matched_tiny,
        "unmatched_predicted_pixels": 100,
        "valid_pixel_count": 100_000,
    }


class TaskMatrixTests(unittest.TestCase):
    def test_real_completed_plan_builds_exact_24_role_partition(self) -> None:
        plan = subject.load_completed_plan(subject.LAUNCH_PLAN)
        tasks = subject.build_tasks(plan)
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len({task.key for task in tasks}), 24)
        self.assertEqual(
            sum(task.checkpoint_role == "best_miou" for task in tasks), 12
        )
        self.assertEqual(
            sum(task.checkpoint_role == "best_pd" for task in tasks), 12
        )
        self.assertEqual(
            {task.dataset for task in tasks}, set(subject.data_protocol.DATASETS)
        )
        self.assertNotIn("SIRST3", {task.dataset for task in tasks})

    def test_original_command_omits_weight_and_final_command_includes_it(self) -> None:
        run_dir = Path("/tmp/example")
        original = subject.EvaluationTask(
            "NUAA-SIRST",
            "original",
            None,
            "best_miou",
            run_dir,
            run_dir / "evaluations" / "best_miou.json",
        )
        final = subject.EvaluationTask(
            "NUAA-SIRST",
            "final",
            0.005,
            "best_pd",
            run_dir,
            run_dir / "evaluations" / "best_pd.json",
        )
        original_command = subject._task_command(
            original,
            python=subject.PYTHON,
            dataset_root=subject.DATASET_ROOT,
            protocol_manifest=subject.PROTOCOL_MANIFEST,
            workers=0,
        )
        final_command = subject._task_command(
            final,
            python=subject.PYTHON,
            dataset_root=subject.DATASET_ROOT,
            protocol_manifest=subject.PROTOCOL_MANIFEST,
            workers=0,
        )
        self.assertNotIn("--requested-tss-weight", original_command)
        self.assertIn("--requested-tss-weight", final_command)
        self.assertEqual(final_command[final_command.index("--device") + 1], "cuda:0")

    def test_method_specific_canonical_run_directories(self) -> None:
        root = Path("/tmp/results")
        original = subject.EvaluationTask(
            "NUAA-SIRST", "original", None, "best_miou", Path("/tmp/x"), Path("/tmp/y")
        )
        final = subject.EvaluationTask(
            "IRSTD-1K", "final", 0.0025, "best_pd", Path("/tmp/x"), Path("/tmp/y")
        )
        self.assertEqual(
            subject._expected_run_dir(root, original),
            root / "runs" / "NUAA-SIRST" / "original" / "seed_42",
        )
        self.assertEqual(
            subject._expected_run_dir(root, final),
            root
            / "runs"
            / "IRSTD-1K"
            / "final"
            / "lambda_0p0025"
            / "seed_42",
        )


class WriteOnceTests(unittest.TestCase):
    def test_write_once_reuses_identical_and_rejects_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            self.assertEqual(subject._write_once_or_identical(path, {"x": 1}), "written")
            self.assertEqual(
                subject._write_once_or_identical(path, {"x": 1}),
                "reused_identical",
            )
            with self.assertRaisesRegex(subject.PosttrainingError, "conflicts"):
                subject._write_once_or_identical(path, {"x": 2})

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(subject.PosttrainingError, "duplicate"):
                subject._load_json(path)

    def test_status_archive_and_attempt_logs_preserve_prior_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status.json"
            subject._atomic_json(status, {"status": "failed"})
            archived = subject._archive_existing_status(status, root / "history")
            self.assertIsNotNone(archived)
            self.assertFalse(status.exists())
            self.assertEqual(subject._load_json(archived)["status"], "failed")
            logs = root / "logs"
            logs.mkdir()
            first = subject._next_log_path(logs, "task")
            first.write_text("first\n", encoding="utf-8")
            second = subject._next_log_path(logs, "task")
            self.assertEqual(second.name, "task.attempt_2.log")

    def test_failed_second_lock_does_not_move_live_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status.json"
            subject._atomic_json(status, {"status": "evaluating"})
            before = status.read_bytes()
            held = (root / "posttraining.lock").open("a+", encoding="utf-8")
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaisesRegex(
                    subject.PosttrainingError, "another post-training"
                ):
                    subject._acquire_posttraining_lock(root, status)
                self.assertTrue(status.is_file())
                self.assertEqual(status.read_bytes(), before)
                self.assertFalse((root / "status_history").exists())
            finally:
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)
                held.close()


class EvaluationReuseTests(unittest.TestCase):
    def test_tampered_fixed_metric_is_rejected_against_checkpoint_payload(self) -> None:
        plan = subject.load_completed_plan(subject.LAUNCH_PLAN)
        source_task = next(
            task
            for task in subject.build_tasks(plan)
            if task.dataset == "NUAA-SIRST"
            and task.method == "original"
            and task.checkpoint_role == "best_miou"
        )
        payload = subject._load_json(source_task.output_path)
        payload["fixed_threshold_0_5"]["miou"] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tampered.json"
            subject._atomic_json(output, payload)
            task = subject.EvaluationTask(
                dataset=source_task.dataset,
                method=source_task.method,
                requested_tss_weight=source_task.requested_tss_weight,
                checkpoint_role=source_task.checkpoint_role,
                run_dir=source_task.run_dir,
                output_path=output,
            )
            with self.assertRaisesRegex(ValueError, "checkpoint metric differs"):
                subject.validate_evaluation(task)


class SelectorAssemblyTests(unittest.TestCase):
    def test_assembly_uses_only_fixed_points_and_exact_frozen_matrix(self) -> None:
        plan = subject.load_completed_plan(subject.LAUNCH_PLAN)
        tasks = subject.build_tasks(plan)
        payload_by_key = {}
        for task in tasks:
            expected = subject.data_protocol.EXPECTED_SPLITS[task.dataset]["test"]
            payload_by_key[task.key] = {
                "fixed_threshold_0_5": copy.deepcopy(_point()),
                "data": {
                    "img_idx_test_sha256": expected["file_sha256"],
                    "img_idx_test_ordered_ids_sha256": expected[
                        "ordered_ids_sha256"
                    ],
                },
                "descriptive_pd_fa": {"points": [{"threshold": 1.0}]},
            }
        with mock.patch.object(
            subject,
            "validate_evaluation",
            side_effect=lambda task: payload_by_key[task.key],
        ):
            assembled = subject._selector_input(tasks)
        normalized = subject.selector.validate_input(assembled)
        self.assertEqual(normalized["threshold"], 0.5)
        self.assertEqual(
            tuple(normalized["datasets"]), subject.data_protocol.DATASETS
        )
        self.assertNotIn("descriptive_pd_fa", str(assembled))
        for dataset in subject.data_protocol.DATASETS:
            self.assertEqual(
                tuple(assembled["datasets"][dataset]["final_candidates"]),
                ("0.0025", "0.005", "0.01"),
            )


class GpuBindingTests(unittest.TestCase):
    def test_exact_physical_gpu_uuid_mapping_is_required(self) -> None:
        valid = {
            2: subject.GPU_BINDINGS["best_miou"]["uuid"],
            3: subject.GPU_BINDINGS["best_pd"]["uuid"],
        }
        with mock.patch.object(subject, "_gpu_inventory", return_value=valid):
            observed = subject.verify_gpu_bindings()
        self.assertEqual(observed["best_miou"]["physical_index"], 2)
        self.assertEqual(observed["best_pd"]["physical_index"], 3)
        invalid = dict(valid)
        invalid[3] = "GPU-wrong"
        with mock.patch.object(subject, "_gpu_inventory", return_value=invalid):
            with self.assertRaisesRegex(subject.PosttrainingError, "UUID differs"):
                subject.verify_gpu_bindings()


if __name__ == "__main__":
    unittest.main()
