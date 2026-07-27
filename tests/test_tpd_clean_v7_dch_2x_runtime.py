from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = (
    REPO_ROOT
    / "experiments/run_tpd_clean_v7_dch_formal800_2x5090_worker.sh"
)
LANE = (
    REPO_ROOT
    / "experiments/run_tpd_clean_v7_dch_formal800_2x5090_lane.sh"
)
LAUNCHER = (
    REPO_ROOT
    / "experiments/launch_tpd_clean_v7_dch_formal800_2x5090.sh"
)
STATUS = (
    REPO_ROOT
    / "experiments/status_tpd_clean_v7_dch_formal800_2x5090.sh"
)

GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
EXPECTED_JOBS = {
    f"tpd_clean_v7_dch_full:42:{GPU2_UUID}",
    f"tpd_clean_v7_dch_capacity:3407:{GPU2_UUID}",
    f"tpd_clean_v7_dch_capacity:42:{GPU3_UUID}",
    f"tpd_clean_v7_dch_full:3407:{GPU3_UUID}",
}


class TPDCleanV7DCHTwoGpuRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worker = WORKER.read_text(encoding="utf-8")
        cls.lane = LANE.read_text(encoding="utf-8")
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.status = STATUS.read_text(encoding="utf-8")

    def test_shell_entrypoints_are_executable_and_syntax_valid(self) -> None:
        for path in (WORKER, LANE, LAUNCHER, STATUS):
            with self.subTest(path=path.name):
                self.assertTrue(path.stat().st_mode & 0o111)
                subprocess.run(
                    ["/usr/bin/bash", "-n", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

    def test_worker_freezes_the_counterbalanced_four_jobs(self) -> None:
        cases = set(
            re.findall(
                r"^\s+(tpd_clean_v7_dch_(?:full|capacity):"
                r"(?:42|3407):GPU-[^)]+)\)",
                self.worker,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(cases, EXPECTED_JOBS)
        self.assertIn("reason=invalid_job_mapping", self.worker)

    def test_only_physical_gpu2_and_gpu3_are_used(self) -> None:
        self.assertIn(
            'export CUDA_VISIBLE_DEVICES="$dch_gpu_uuid"',
            self.worker,
        )
        self.assertIn(
            'export TPD_DCH_PHYSICAL_GPU_INDEX="$dch_physical_index"',
            self.worker,
        )
        self.assertIn(
            'export TPD_DCH_PHYSICAL_GPU_UUID="$dch_gpu_uuid"',
            self.worker,
        )
        self.assertNotRegex(
            self.worker,
            r"CUDA_VISIBLE_DEVICES\s*=\s*[\"']?[01](?:[\"']|\s|$)",
        )
        self.assertIn("torch.cuda.device_count() != 1", self.worker)
        self.assertIn("actual_uuid != expected_uuid", self.worker)
        self.assertIn("NVIDIA GeForce RTX 5090", self.worker)
        self.assertIn(
            '"physical_gpu_index": int(sys.argv[4])',
            self.worker,
        )
        self.assertIn('"physical_gpu_uuid": sys.argv[5]', self.worker)

    def test_lanes_are_serial_and_launcher_starts_two_units(self) -> None:
        self.assertIn("for dch_index in 0 1; do", self.lane)
        self.assertEqual(
            self.lane.count(
                '"$dch_worker" "$dch_variant" "$dch_seed" "$dch_gpu_uuid"'
            ),
            1,
        )
        self.assertNotRegex(self.lane, r"(?:^|[;\s])&(?:$|[;\s])")
        self.assertEqual(self.launcher.count("systemd-run --user"), 2)
        self.assertEqual(
            self.launcher.count("--property=Restart=on-failure"),
            2,
        )
        self.assertIn("concurrent_tasks_per_gpu=1", self.launcher)
        self.assertIn(
            "physical_gpu${dch_physical_index}.lock",
            self.worker,
        )
        self.assertIn("flock -n 8", self.worker)
        self.assertIn("flock -n 9", self.worker)

    def test_worker_has_fresh_resume_and_idempotent_complete_modes(
        self,
    ) -> None:
        command = self.worker.index(
            '"$dch_python" experiments/train_tpd_clean_v7_dch_exact.py'
        )
        self.assertLess(self.worker.index('print("fresh")'), command)
        self.assertLess(self.worker.index('print("exact-resume")'), command)
        self.assertLess(self.worker.index('print("complete")'), command)
        self.assertIn('dch_init_flag="--fresh"', self.worker)
        self.assertIn('dch_init_flag="--exact-resume"', self.worker)
        self.assertIn(
            "TPDCLEANV7DCH_2X_IDEMPOTENT_COMPLETE",
            self.worker,
        )

    def test_exact_command_freezes_formal_axes(self) -> None:
        required = (
            'export PYTHONHASHSEED="$dch_seed"',
            'export CUBLAS_WORKSPACE_CONFIG=":4096:8"',
            "--device cuda:0",
            "--epochs 800",
            "--batch-size 16",
            "--patch-size 256",
            "--workers 0",
            "--split-seed 20260722",
            "--val-fraction 0.20",
            "--eval-every 1",
            "--base-lr 0.001",
            "--min-lr 0.00001",
            "--warmup-epochs 10",
            "--threshold 0.5",
            "--match-radius 3",
            "--tiny-area 9",
            "--eps 0.000001",
            '--exact-source-lock "$dch_source_lock"',
            '"$dch_init_flag"',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.worker)
        command = self.worker[command_index(self.worker) :]
        self.assertNotIn("--amp", command)

    def test_worker_requires_source_data_smoke_and_17_metrics(self) -> None:
        self.assertIn("exact.source_lock_contract(", self.worker)
        self.assertIn(
            "shared_exact.official_training_data_sha256(",
            self.worker,
        )
        self.assertIn(
            "smoke_verifier.validate_smoke_reports(smoke_root)",
            self.worker,
        )
        self.assertIn("STORED_VALIDATION_METRICS", self.worker)
        self.assertIn("stored_validation_metrics=17", self.worker)

    def test_preflight_precedes_both_systemd_launches(self) -> None:
        first_systemd = self.launcher.index("systemd-run --user")
        self.assertLess(
            self.launcher.index('"$dch_lane_runner" --preflight gpu2'),
            first_systemd,
        )
        self.assertLess(
            self.launcher.index('"$dch_lane_runner" --preflight gpu3'),
            first_systemd,
        )

    def test_status_reports_four_tasks_metrics_and_assignment(self) -> None:
        self.assertEqual(
            re.findall(
                r"^\s+tpd_clean_v7_dch_(?:full|capacity)$",
                self.status,
                flags=re.MULTILINE,
            ),
            [
                "    tpd_clean_v7_dch_full",
                "    tpd_clean_v7_dch_capacity",
                "    tpd_clean_v7_dch_capacity",
                "    tpd_clean_v7_dch_full",
            ],
        )
        self.assertIn("latest=[$dch_latest]", self.status)
        self.assertIn("assignment=$dch_assignment_state", self.status)
        self.assertIn("gpu_uuid=$dch_gpu_uuid", self.status)

    def test_invalid_mapping_stops_before_missing_source_lock(self) -> None:
        completed = subprocess.run(
            [
                "/usr/bin/bash",
                str(WORKER),
                "tpd_clean_v7_dch_full",
                "42",
                GPU3_UUID,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("reason=invalid_job_mapping", completed.stderr)
        self.assertNotIn("missing_formal_source_lock", completed.stderr)


def command_index(worker: str) -> int:
    return worker.index(
        '"$dch_python" experiments/train_tpd_clean_v7_dch_exact.py'
    )


if __name__ == "__main__":
    unittest.main()
