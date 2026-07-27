from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = (
    REPO_ROOT
    / "experiments/run_tpd_clean_v6_formal800_2x5090_worker.sh"
)
LANE = (
    REPO_ROOT / "experiments/run_tpd_clean_v6_formal800_2x5090_lane.sh"
)
LAUNCHER = (
    REPO_ROOT / "experiments/launch_tpd_clean_v6_formal800_2x5090.sh"
)
STATUS = (
    REPO_ROOT / "experiments/status_tpd_clean_v6_formal800_2x5090.sh"
)

GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
EXPECTED_JOBS = {
    f"tpd_clean_v6_full:42:{GPU2_UUID}",
    f"tpd_clean_v6_phase_capacity:42:{GPU3_UUID}",
    f"tpd_clean_v6_full:3407:{GPU3_UUID}",
    f"tpd_clean_v6_phase_capacity:3407:{GPU2_UUID}",
}


class TPDCleanV6TwoGpuRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worker = WORKER.read_text(encoding="utf-8")
        cls.lane = LANE.read_text(encoding="utf-8")
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.status = STATUS.read_text(encoding="utf-8")

    def test_all_shell_entrypoints_are_syntax_valid(self) -> None:
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

    def test_worker_freezes_the_four_counterbalanced_jobs(self) -> None:
        cases = set(
            re.findall(
                r"^\s+(tpd_clean_v6_(?:full|phase_capacity):"
                r"(?:42|3407):GPU-[^)]+)\)",
                self.worker,
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(cases, EXPECTED_JOBS)
        self.assertIn('v6_physical_index="2"', self.worker)
        self.assertIn('v6_physical_index="3"', self.worker)
        self.assertIn("reason=invalid_job_mapping", self.worker)

    def test_only_physical_gpu2_and_gpu3_are_exposed_as_one_uuid(self) -> None:
        self.assertIn(
            'export CUDA_VISIBLE_DEVICES="$v6_gpu_uuid"',
            self.worker,
        )
        self.assertNotRegex(
            self.worker,
            r"CUDA_VISIBLE_DEVICES\s*=\s*[\"']?[01](?:[\"']|\s|$)",
        )
        self.assertIn("torch.cuda.device_count() != 1", self.worker)
        self.assertIn('logical_device=cuda:0', self.worker)
        self.assertIn("actual_uuid != expected_uuid", self.worker)
        self.assertIn("NVIDIA GeForce RTX 5090", self.worker)
        self.assertEqual(self.worker.count(GPU2_UUID), 4)
        self.assertEqual(self.worker.count(GPU3_UUID), 4)

    def test_each_lane_runs_two_tasks_serially_and_launcher_starts_two_lanes(
        self,
    ) -> None:
        self.assertIn("for v6_index in 0 1; do", self.lane)
        self.assertEqual(
            self.lane.count(
                '"$v6_worker" "$v6_variant" "$v6_seed" "$v6_gpu_uuid"'
            ),
            1,
        )
        self.assertNotRegex(
            self.lane,
            r"(?:^|[;\s])&(?:$|[;\s])",
        )
        self.assertEqual(self.launcher.count("systemd-run --user"), 2)
        self.assertEqual(self.launcher.count("--collect"), 2)
        self.assertEqual(
            self.launcher.count("--property=Restart=on-failure"),
            2,
        )
        self.assertEqual(
            self.launcher.count("--property=RestartSec=30"),
            2,
        )
        self.assertIn(
            'v6_gpu2_unit="sctransnet-tpd-clean-v6-gpu2-lane"',
            self.launcher,
        )
        self.assertIn(
            'v6_gpu3_unit="sctransnet-tpd-clean-v6-gpu3-lane"',
            self.launcher,
        )
        self.assertIn("concurrent_tasks_per_gpu=1", self.launcher)
        self.assertIn("physical_gpu${v6_physical_index}.lock", self.worker)
        self.assertIn("flock -n 8", self.worker)
        self.assertIn(
            "${v6_variant}_seed${v6_seed}.lock",
            self.worker,
        )
        self.assertIn("flock -n 9", self.worker)

    def test_worker_selects_fresh_resume_or_idempotent_complete(self) -> None:
        fresh = self.worker.index('print("fresh")')
        resume = self.worker.index('print("exact-resume")')
        complete = self.worker.index('print("complete")')
        command = self.worker.index(
            '"$v6_python" experiments/train_tpd_clean_v6_exact.py'
        )
        self.assertLess(fresh, command)
        self.assertLess(resume, command)
        self.assertLess(complete, command)
        self.assertIn('v6_init_flag="--fresh"', self.worker)
        self.assertIn('v6_init_flag="--exact-resume"', self.worker)
        self.assertIn(
            'if [[ "$v6_mode" == "run" && '
            '"$v6_initialization" == "complete" ]]',
            self.worker,
        )
        self.assertIn(
            'if [[ "$v6_mode" == "preflight" ]]',
            self.worker,
        )
        self.assertIn("TPDCLEANV6_2X_IDEMPOTENT_COMPLETE", self.worker)

    def test_exact_command_freezes_formal_arguments_and_environment(self) -> None:
        required_fragments = (
            'export PYTHONHASHSEED="$v6_seed"',
            'export CUBLAS_WORKSPACE_CONFIG=":4096:8"',
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
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
            '--exact-source-lock "$v6_source_lock"',
            '"$v6_init_flag"',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.worker)
        command = self.worker[
            self.worker.index(
                '"$v6_python" experiments/train_tpd_clean_v6_exact.py'
            ) :
        ]
        self.assertNotIn("--amp", command)

    def test_worker_verifies_source_data_and_three_smoke_reports(self) -> None:
        self.assertIn(
            "exact.source_lock_contract(",
            self.worker,
        )
        self.assertIn(
            "exact.official_training_data_sha256(",
            self.worker,
        )
        self.assertIn(
            "smoke_verifier.validate_smoke_reports(smoke_root)",
            self.worker,
        )
        self.assertIn(
            '"2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"',
            self.worker,
        )
        self.assertIn(
            '"3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"',
            self.worker,
        )

    def test_preflight_finishes_before_any_systemd_launch(self) -> None:
        gpu2_preflight = self.launcher.index(
            '"$v6_lane_runner" --preflight gpu2'
        )
        gpu3_preflight = self.launcher.index(
            '"$v6_lane_runner" --preflight gpu3'
        )
        preflight_exit = self.launcher.index(
            'if [[ "$v6_mode" == "preflight" ]]'
        )
        first_systemd = self.launcher.index("systemd-run --user")
        self.assertLess(gpu2_preflight, preflight_exit)
        self.assertLess(gpu3_preflight, preflight_exit)
        self.assertLess(preflight_exit, first_systemd)

    def test_status_covers_all_four_tasks_and_reports_latest_and_active(
        self,
    ) -> None:
        self.assertEqual(
            re.findall(
                r"^\s+tpd_clean_v6_(?:full|phase_capacity)$",
                self.status,
                flags=re.MULTILINE,
            ),
            [
                "    tpd_clean_v6_full",
                "    tpd_clean_v6_phase_capacity",
                "    tpd_clean_v6_full",
                "    tpd_clean_v6_phase_capacity",
            ],
        )
        self.assertIn("latest=[$v6_latest]", self.status)
        self.assertIn("active=$v6_task_active", self.status)
        self.assertIn("summary=[$v6_completion]", self.status)

    def test_invalid_mapping_is_rejected_before_any_runtime_check(self) -> None:
        completed = subprocess.run(
            [
                "/usr/bin/bash",
                str(WORKER),
                "tpd_clean_v6_full",
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


if __name__ == "__main__":
    unittest.main()
