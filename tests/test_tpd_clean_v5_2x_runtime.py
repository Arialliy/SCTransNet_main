from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v5_screen800_2x_source_lock.json"
)
LAUNCHER = (
    REPO_ROOT / "experiments/launch_tpd_clean_v5_screen800_2x5090.sh"
)
SMOKE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v5_preflight_v1"
)
REQUIRED_SOURCES = {
    "model/tpd_clean_v5.py",
    "experiments/train_tpd_clean_v5.py",
    "experiments/evaluate_tpd_clean_v5_pd_fa.py",
    "experiments/smoke_tpd_clean_v5.py",
    "experiments/capture_tpd_clean_v5_smoke_report.py",
    "experiments/run_tpd_clean_v5_screen800_2x5090_worker.sh",
    "experiments/launch_tpd_clean_v5_screen800_2x5090.sh",
    "experiments/status_tpd_clean_v5_screen800_2x5090.sh",
    "experiments/TPD_CLEAN_V5_PROTOCOL.md",
    "experiments/TPD_CLEAN_V5_2GPU_PROTOCOL.md",
    "tests/test_tpd_clean_v5.py",
    "tests/test_train_tpd_clean_v5.py",
    "tests/test_evaluate_tpd_clean_v5_pd_fa.py",
    "tests/test_smoke_tpd_clean_v5.py",
    "tests/test_tpd_clean_v5_runner.py",
    "tests/test_tpd_clean_v5_2x_runtime.py",
    "experiments/train_tpd_pilot.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/fingerprint_tpd_training_data.py",
    "dataset.py",
    "utils.py",
    "warmup_scheduler.py",
    "model/SCTransNet.py",
    "model/Config.py",
    "model/tpd.py",
    "experiments/smoke_tpd_clean_v3.py",
    "experiments/train_tpd_clean_v3.py",
    "model/tpd_clean_v3.py",
    "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
    "experiments/tpd_clean_v3_screen800_source_lock.json",
    "experiments/tpd_clean_screen800_source_lock.json",
    "experiments/tpd_ner_v1_source_lock.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TPDCleanV5TwoGpuRuntimeTests(unittest.TestCase):
    def test_source_lock_recomputes_and_binds_smoke_reports(self) -> None:
        payload = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["schema"],
            "sctransnet_tpd_clean_v5_screen800_2x_source_lock_v1",
        )
        self.assertEqual(
            payload["candidate_family"],
            "spd_anchored_tpd_clean_v5_positive_context_selector",
        )
        self.assertEqual(
            payload["variants"],
            ["tpd_clean_v5_full", "tpd_clean_v5_sal_capacity"],
        )
        self.assertEqual(payload["model_seeds"], [42, 3407])
        self.assertEqual(payload["allowed_gpu_indices"], [2, 3])
        self.assertEqual(payload["concurrent_jobs_per_gpu"], 2)
        self.assertFalse(payload["fourth_parallel_branch_added"])
        self.assertEqual(set(payload["source_sha256"]), REQUIRED_SOURCES)
        for relative, expected in payload["source_sha256"].items():
            path = REPO_ROOT / relative
            with self.subTest(relative=relative):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(sha256(path), expected)

        expected_smoke = {
            "cpu_all.json",
            "gpu2_full.json",
            "gpu3_capacity.json",
        }
        self.assertEqual(set(payload["smoke_sha256"]), expected_smoke)
        for name, expected in payload["smoke_sha256"].items():
            path = SMOKE_ROOT / name
            with self.subTest(smoke=name):
                self.assertEqual(sha256(path), expected)
                envelope = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(envelope["status"], "complete")
                self.assertEqual(
                    envelope["report"]["schema"],
                    "sctransnet_tpd_clean_v5_smoke_v1",
                )
                self.assertEqual(envelope["report"]["status"], "complete")

    def test_absolute_python_entrypoints_help_without_pythonpath(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        entrypoints = (
            "experiments/train_tpd_clean_v5.py",
            "experiments/smoke_tpd_clean_v5.py",
            "experiments/evaluate_tpd_clean_v5_pd_fa.py",
            "experiments/capture_tpd_clean_v5_smoke_report.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            for relative in entrypoints:
                with self.subTest(relative=relative):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(REPO_ROOT / relative),
                            "--help",
                        ],
                        cwd=temporary,
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertIn("usage:", completed.stdout.lower())

    def test_launcher_source_and_smoke_preflight_blocks_execute(self) -> None:
        blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY",
            LAUNCHER.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        self.assertEqual(len(blocks), 2)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        cases = (
            (
                blocks[0],
                [str(REPO_ROOT), str(SOURCE_LOCK)],
                "TPDCLEANV5_2X_PREFLIGHT_SOURCES_OK",
            ),
            (
                blocks[1],
                [str(REPO_ROOT), str(SOURCE_LOCK), str(SMOKE_ROOT)],
                "TPDCLEANV5_2X_PREFLIGHT_SMOKE_OK",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for source, arguments, marker in cases:
                with self.subTest(marker=marker):
                    completed = subprocess.run(
                        [sys.executable, "-", *arguments],
                        input=source + "\n",
                        cwd=temporary,
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertIn(marker, completed.stdout)


if __name__ == "__main__":
    unittest.main()
