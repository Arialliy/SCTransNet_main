from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHER = (
    REPO_ROOT
    / "experiments"
    / "watch_tpd_ner_v4_qfg_v2_croa_formal800_training.sh"
)


class Formal800TrainingWatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.repo = root / "repo"
        self.result_root = root / "results"
        self.repo.mkdir()
        self.launch_record = root / "launch_record.txt"
        self.launcher = root / "launcher.sh"
        self.launcher.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"printf '%s\\n' \"$*\" > {self.launch_record}\n",
            encoding="utf-8",
        )
        self.launcher.chmod(
            self.launcher.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )
        self.environment = {
            **os.environ,
            "TPD_NER_V4_QFG_V2_CROA_REPO": str(self.repo),
            "TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT": str(self.result_root),
            "TPD_NER_V4_QFG_V2_CROA_TRAIN_LAUNCHER": str(self.launcher),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_watcher(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(WATCHER)],
            check=False,
            capture_output=True,
            text=True,
            env=self.environment,
        )

    def test_free_claim_delegates_to_authoritative_launcher(self) -> None:
        result = self.run_watcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("QFG2X_WATCH_RESTART", result.stdout)
        self.assertEqual(
            self.launch_record.read_text(encoding="utf-8"),
            "--freeze verify\n",
        )

    def test_busy_claim_returns_tempfail_without_cuda_preflight(self) -> None:
        lock_dir = self.result_root / ".launcher_locks"
        lock_dir.mkdir(parents=True)
        claim = lock_dir / "formal800_seed42_paired.lock"
        with claim.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_watcher()
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("QFG2X_WATCH_WAIT", result.stdout)
        self.assertFalse(self.launch_record.exists())

    def test_invalid_launcher_is_configuration_error(self) -> None:
        self.launcher.unlink()
        result = self.run_watcher()
        self.assertEqual(result.returncode, 64)
        self.assertIn("reason=invalid_launcher", result.stderr)


if __name__ == "__main__":
    unittest.main()
