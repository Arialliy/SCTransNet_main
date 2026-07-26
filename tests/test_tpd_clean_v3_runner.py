from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "experiments/tpd_clean_v3_screen800_source_lock.json"
OLD_LOCKS = (
    REPO_ROOT / "experiments/tpd_clean_screen800_source_lock.json",
    REPO_ROOT / "experiments/tpd_ner_v1_source_lock.json",
)
SHELL_FILES = (
    REPO_ROOT / "experiments/run_tpd_clean_v3_screen800_4x5090_worker.sh",
    REPO_ROOT / "experiments/launch_tpd_clean_v3_screen800_4x5090.sh",
    REPO_ROOT / "experiments/status_tpd_clean_v3_screen800_4x5090.sh",
)


def _verify_lock(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in payload["source_sha256"].items():
        source = REPO_ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise AssertionError(f"invalid source path: {relative}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(
                f"digest mismatch for {relative}: {actual} != {expected}"
            )


class TPDCleanV3RunnerTests(unittest.TestCase):
    def test_all_shell_entrypoints_parse(self) -> None:
        subprocess.run(
            ["bash", "-n", *(str(path) for path in SHELL_FILES)],
            cwd=REPO_ROOT,
            check=True,
        )

    def test_worker_rejects_unregistered_mapping_before_side_effects(self) -> None:
        completed = subprocess.run(
            [
                "bash",
                str(SHELL_FILES[0]),
                "tpd_clean_v3_full",
                "99",
                "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid_job_mapping", completed.stderr)

    def test_protocol_and_launch_freeze_two_models_two_seeds(self) -> None:
        protocol = (
            REPO_ROOT / "experiments/TPD_CLEAN_V3_PROTOCOL.md"
        ).read_text(encoding="utf-8")
        launcher = SHELL_FILES[1].read_text(encoding="utf-8")
        worker = SHELL_FILES[0].read_text(encoding="utf-8")
        for value in (
            "tpd_clean_v3_full",
            "tpd_clean_v3_sal_capacity",
            "42",
            "3407",
        ):
            with self.subTest(value=value):
                self.assertIn(value, protocol)
                self.assertIn(value, launcher)
                self.assertIn(value, worker)
        self.assertIn(
            "tpd_clean_v3_screen800_4x5090_v1", launcher
        )
        self.assertNotIn(
            "results/tpd_clean_screen800_4x5090_v1", launcher
        )

    def test_v3_source_lock_covers_training_surface(self) -> None:
        payload = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["schema"],
            "sctransnet_tpd_clean_v3_screen800_source_lock_v1",
        )
        required = {
            "model/tpd_clean_v3.py",
            "experiments/train_tpd_clean_v3.py",
            "experiments/evaluate_tpd_clean_v3_pd_fa.py",
            "experiments/smoke_tpd_clean_v3.py",
            "experiments/run_tpd_clean_v3_screen800_4x5090_worker.sh",
            "experiments/launch_tpd_clean_v3_screen800_4x5090.sh",
            "experiments/status_tpd_clean_v3_screen800_4x5090.sh",
            "experiments/TPD_CLEAN_V3_PROTOCOL.md",
            "tests/test_tpd_clean_v3.py",
            "tests/test_train_tpd_clean_v3.py",
            "tests/test_smoke_tpd_clean_v3.py",
            "tests/test_tpd_clean_v3_runner.py",
            "experiments/train_tpd_pilot.py",
            "experiments/evaluate_pd_fa_sweep.py",
            "experiments/fingerprint_tpd_training_data.py",
            "dataset.py",
            "utils.py",
            "model/SCTransNet.py",
            "model/Config.py",
            "model/tpd.py",
        }
        self.assertTrue(required.issubset(payload["source_sha256"]))
        _verify_lock(SOURCE_LOCK)

    def test_existing_clean_v2_and_ner_locks_still_match(self) -> None:
        for lock in OLD_LOCKS:
            with self.subTest(lock=lock.name):
                _verify_lock(lock)


if __name__ == "__main__":
    unittest.main()
