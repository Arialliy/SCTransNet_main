from __future__ import annotations

import re
import stat
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
WORKER = EXPERIMENTS / "run_tpd_clean_v3_resume_2x5090_worker.sh"
LAUNCHER = EXPERIMENTS / "launch_tpd_clean_v3_resume_2x5090.sh"
STATUS = EXPERIMENTS / "status_tpd_clean_v3_resume_2x5090.sh"
PROTOCOL = EXPERIMENTS / "TPD_CLEAN_V3_RESUME_2GPU_PROTOCOL.md"
RESUME_PROGRAM = EXPERIMENTS / "resume_tpd_clean_v3.py"

GPU2 = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3 = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
GPU0 = "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70"
GPU1 = "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640"
RESULT_ROOT = "tpd_clean_v3_screen800_4x5090_v1"
RUN_TAG = "screen800_pd_fp32_shared4x5090_v1"
UNIT_PREFIX = "sctransnet-tpd-clean-v3-resume-2x-"
SOURCE_LOCK = "tpd_clean_v3_resume_2x_source_lock.json"
SOURCE_LOCK_SCHEMA = "sctransnet_tpd_clean_v3_resume_2x_source_lock_v1"
MANIFEST_SCHEMA = "sctransnet_tpd_clean_v3_resume_2x5090_launch_v1"

EXPECTED_MAPPING = (
    ("tpd_clean_v3_full", 42, GPU3, "full-s42"),
    ("tpd_clean_v3_sal_capacity", 42, GPU2, "cap-s42"),
    ("tpd_clean_v3_full", 3407, GPU2, "full-s3407"),
    ("tpd_clean_v3_sal_capacity", 3407, GPU3, "cap-s3407"),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TpdCleanV3ResumeTwoGpuRuntimeTests(unittest.TestCase):
    def test_shell_scripts_are_executable_and_parse(self) -> None:
        for script in (WORKER, LAUNCHER, STATUS):
            self.assertTrue(script.is_file(), script)
            self.assertTrue(
                script.stat().st_mode & stat.S_IXUSR,
                f"{script} must be executable",
            )
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{script}: {completed.stderr}",
            )

    def test_worker_rejects_missing_arguments_without_side_effects(self) -> None:
        completed = subprocess.run(
            ["bash", str(WORKER)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("VARIANT SEED GPU_UUID RUN_DIR", completed.stderr)

    def test_only_gpu2_and_gpu3_are_bound(self) -> None:
        for path in (WORKER, LAUNCHER, STATUS, PROTOCOL):
            text = _read(path)
            self.assertIn(GPU2, text)
            self.assertIn(GPU3, text)
            self.assertNotIn(GPU0, text)
            self.assertNotIn(GPU1, text)

    def test_counterbalanced_mapping_is_exact(self) -> None:
        worker = _read(WORKER)
        launcher = _read(LAUNCHER)
        status = _read(STATUS)
        for variant, seed, gpu_uuid, unit_tag in EXPECTED_MAPPING:
            mapping = f"{variant}:{seed}:{gpu_uuid}"
            expected_job = f"{mapping}:{unit_tag}"
            self.assertIn(mapping, worker)
            self.assertIn(expected_job, launcher)
            self.assertIn(unit_tag, status)
        self.assertEqual(
            tuple(item[2] for item in EXPECTED_MAPPING).count(GPU2), 2
        )
        self.assertEqual(
            tuple(item[2] for item in EXPECTED_MAPPING).count(GPU3), 2
        )
        self.assertIn("counterbalanced_mapping_mismatch", launcher)
        self.assertIn("invalid_gpu_multiplicity", launcher)

    def test_original_run_identity_and_resume_isolation(self) -> None:
        combined = "\n".join(
            _read(path) for path in (WORKER, LAUNCHER, STATUS, PROTOCOL)
        )
        for expected in (
            RESULT_ROOT,
            RUN_TAG,
            "resume_2x5090_v1",
            UNIT_PREFIX,
            SOURCE_LOCK,
            SOURCE_LOCK_SCHEMA,
            MANIFEST_SCHEMA,
        ):
            self.assertIn(expected, combined)
        self.assertNotIn("tpd_clean_v3_screen800_2x5090_v1", combined)
        self.assertNotIn("screen800_pd_fp32_shared2x5090_v1", combined)

    def test_fixed_resume_cli_matches_engine(self) -> None:
        worker = _read(WORKER)
        for expected in (
            'experiments/resume_tpd_clean_v3.py \\',
            '--run-dir "$v3_run_dir"',
            "--device cuda:0",
            "--target-epoch 800",
            '--expected-resume-epoch "$v3_boundary_epoch"',
            '--resume-gpu-uuid "$v3_gpu_uuid"',
        ):
            self.assertIn(expected, worker)
        self.assertNotIn("--expected-start-epoch", worker)
        self.assertNotIn("--epochs 800", worker)

        help_result = subprocess.run(
            [sys.executable, str(RESUME_PROGRAM), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for option in (
            "--run-dir",
            "--device",
            "--target-epoch",
            "--expected-resume-epoch",
            "--resume-gpu-uuid",
        ):
            self.assertIn(option, help_result.stdout)

    def test_cpu_replay_thread_cap_precedes_python_and_is_manifested(self) -> None:
        worker = _read(WORKER)
        first_python = worker.index('"$v3_python"')
        cap_assignment = worker.index("v3_cpu_threads=1")
        self.assertLess(cap_assignment, first_python)
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            export = f'export {variable}="$v3_cpu_threads"'
            self.assertIn(export, worker)
            self.assertLess(worker.index(export), first_python)
            self.assertIn(
                f'"{variable}": os.environ["{variable}"]', worker
            )
        self.assertIn('"cpu_replay_thread_cap": 1', worker)
        self.assertIn("cpu_threads=$v3_cpu_threads", worker)

    def test_boundary_validation_and_snapshot_contract(self) -> None:
        worker = _read(WORKER)
        for expected in (
            "metrics epochs are not contiguous from 1",
            "1 <= epoch < 800",
            "last checkpoint epoch does not match metrics boundary",
            "checkpoint/metrics numeric mismatch",
            "split hashes do not match last checkpoint",
            "resume boundary already exists",
            "immutable_no_overwrite",
            "sctransnet_tpd_clean_v3_resume_boundary_v1",
            "original_launch_manifest.json",
            "original_worker.log",
            '"metrics.jsonl"',
            '"last.pth.tar"',
            '"best.pth.tar"',
            '"best_miou.pth.tar"',
        ):
            self.assertIn(expected, worker)
        self.assertIn("source_sha256", worker)
        self.assertIn("snapshot_sha256", worker)
        self.assertIn("path.chmod(0o444)", worker)
        self.assertIn("target.chmod(0o555)", worker)

    def test_manifest_policy_and_completion_contract(self) -> None:
        worker = _read(WORKER)
        for expected in (
            '"in_place_resume": True',
            '"fresh_run": False',
            '"original_results_preserved_by_boundary": True',
            '"immutable_resume_boundary": True',
            '"allowed_gpu_indices": [2, 3]',
            '"concurrent_jobs_per_gpu": 2',
            '"counterbalanced_mapping": True',
            '"efficiency_comparison_allowed": False',
            '"official_test_accessed": False',
            '"cpu_replay_thread_cap": 1',
            "pd_fa_sweep_best.pth.json",
            "pd_fa_sweep_best_miou.pth.json",
            "TPDCLEANV3_RESUME_2X_COMPLETE",
        ):
            self.assertIn(expected, worker)

    def test_launcher_requires_old_inactive_and_new_absent(self) -> None:
        launcher = _read(LAUNCHER)
        for old_unit in (
            "sctransnet-tpd-clean-v3-full-s42.service",
            "sctransnet-tpd-clean-v3-cap-s42.service",
            "sctransnet-tpd-clean-v3-full-s3407.service",
            "sctransnet-tpd-clean-v3-cap-s3407.service",
        ):
            self.assertIn(old_unit, launcher)
        self.assertIn('[[ "$v3_old_state" != "inactive" ]]', launcher)
        self.assertIn("old_unit_not_inactive", launcher)
        self.assertIn("new_unit_already_exists", launcher)
        self.assertIn("required_mib=17000", launcher)
        self.assertIn('if [[ "$v3_mode" == "--preflight" ]]', launcher)

    def test_wrapper_markers_are_namespaced(self) -> None:
        for path in (WORKER, LAUNCHER):
            markers = re.findall(r"TPDCLEANV3_[A-Z0-9_]+", _read(path))
            self.assertTrue(markers)
            for marker in markers:
                self.assertTrue(
                    marker.startswith("TPDCLEANV3_RESUME_2X"),
                    f"unscoped wrapper marker in {path}: {marker}",
                )

    def test_status_reports_boundary_sweeps_and_completion(self) -> None:
        status = _read(STATUS)
        for expected in (
            "old_unit=",
            "resume_unit=",
            "resume_boundary=",
            "epochs=",
            "sweeps=",
            "completion=",
            "TPDCLEANV3_RESUME_2X_COMPLETE",
        ):
            self.assertIn(expected, status)


if __name__ == "__main__":
    unittest.main()
