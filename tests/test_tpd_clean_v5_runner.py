from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "experiments/run_tpd_clean_v5_screen800_2x5090_worker.sh"
LAUNCHER = REPO_ROOT / "experiments/launch_tpd_clean_v5_screen800_2x5090.sh"
STATUS = REPO_ROOT / "experiments/status_tpd_clean_v5_screen800_2x5090.sh"


class TPDCleanV5RunnerContractTests(unittest.TestCase):
    def test_shell_sources_parse_and_use_only_v5_entrypoints(self) -> None:
        for path in (WORKER, LAUNCHER, STATUS):
            with self.subTest(path=path.name):
                subprocess.run(
                    ["/usr/bin/bash", "-n", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

        worker = WORKER.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        combined = worker + launcher
        self.assertIn("experiments/train_tpd_clean_v5.py", worker)
        self.assertIn("experiments/evaluate_tpd_clean_v5_pd_fa.py", worker)
        self.assertNotIn("experiments/train_tpd_clean_v4.py", combined)
        self.assertNotIn("experiments/evaluate_tpd_clean_v4_pd_fa.py", combined)
        self.assertIn("--epochs 800", worker)
        self.assertIn("--batch-size 16", worker)
        self.assertIn("--patch-size 256", worker)
        self.assertIn("--workers 0", worker)
        self.assertIn("--split-seed 20260722", worker)
        self.assertNotIn("--amp", worker)
        self.assertIn("posthoc_endpoint_completion == false", worker)
        self.assertIn("preregistered_endpoint_completion == true", worker)

    def test_gpu_mapping_is_counterbalanced_and_excludes_gpu_zero_one(self) -> None:
        worker = WORKER.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        gpu2 = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
        gpu3 = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
        gpu0 = "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70"
        gpu1 = "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640"
        self.assertGreaterEqual(worker.count(gpu2), 2)
        self.assertGreaterEqual(worker.count(gpu3), 2)
        self.assertEqual(launcher.count(gpu2), 5)
        self.assertEqual(launcher.count(gpu3), 5)
        self.assertNotIn(gpu0, worker + launcher)
        self.assertNotIn(gpu1, worker + launcher)
        for mapping in (
            f"tpd_clean_v5_full:42:{gpu2}",
            f"tpd_clean_v5_sal_capacity:42:{gpu3}",
            f"tpd_clean_v5_full:3407:{gpu3}",
            f"tpd_clean_v5_sal_capacity:3407:{gpu2}",
        ):
            self.assertIn(mapping, worker)
            self.assertIn(mapping, launcher)

    def test_fresh_paths_sources_smoke_and_old_locks_are_enforced(self) -> None:
        worker = WORKER.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("run_path_not_fresh", worker)
        self.assertIn("run_path_not_fresh", launcher)
        self.assertIn("source_lock_changed_during_run", worker)
        self.assertIn("training_data_drift", worker)
        self.assertIn("TPDCLEANV5_2X_LAST_OK", worker)
        self.assertIn("last_evaluated_epoch", worker)
        self.assertIn("load_state_dict(payload[\"state_dict\"], strict=True)", worker)
        self.assertIn("expected_schemas", worker)
        self.assertIn("locks={len(seen)}", worker)
        self.assertIn("TPDCLEANV5_2X_PREFLIGHT_SMOKE_OK", launcher)
        self.assertIn("smoke_sha256", launcher)
        self.assertIn("Restart=no", launcher)
        self.assertIn("concurrent_jobs_per_gpu=2", launcher)


if __name__ == "__main__":
    unittest.main()
