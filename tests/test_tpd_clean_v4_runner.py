from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
TRAIN = EXPERIMENTS / "train_tpd_clean_v4.py"
EVALUATOR = EXPERIMENTS / "evaluate_tpd_clean_v4_pd_fa.py"
WORKER = EXPERIMENTS / "run_tpd_clean_v4_screen800_2x5090_worker.sh"


class TPDCleanV4RunnerTests(unittest.TestCase):
    def test_training_cli_exposes_only_v4_variants(self) -> None:
        completed = subprocess.run(
            [str(PYTHON), str(TRAIN), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("tpd_clean_v4_full", completed.stdout)
        self.assertIn("tpd_clean_v4_sal_capacity", completed.stdout)
        self.assertNotIn("tpd_clean_v3_full", completed.stdout)

    def test_worker_freezes_formal_training_arguments(self) -> None:
        text = WORKER.read_text(encoding="utf-8")
        start = text.index(
            '"$v4_python" experiments/train_tpd_clean_v4.py'
        )
        end = text.index('\n\n[[ "$(wc -l', start)
        command = text[start:end]
        required_fragments = (
            '--epochs 800',
            '--batch-size 16',
            '--patch-size 256',
            '--workers 0',
            '--split-seed 20260722',
            '--val-fraction 0.20',
            '--eval-every 1',
            '--base-lr 0.001',
            '--min-lr 0.00001',
            '--warmup-epochs 10',
            '--threshold 0.5',
            '--match-radius 3',
            '--tiny-area 9',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, command)
        self.assertNotIn("--resume", command)
        self.assertNotIn("--amp", command)
        self.assertNotIn("max-train-images", command)
        self.assertNotIn("max-val-images", command)

    def test_worker_uses_preregistered_closed_interval_evaluator(self) -> None:
        worker = WORKER.read_text(encoding="utf-8")
        evaluator = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn(
            "experiments/evaluate_tpd_clean_v4_pd_fa.py",
            worker,
        )
        self.assertIn(
            ".threshold_provenance.preregistered_endpoint_completion == true",
            worker,
        )
        self.assertIn(
            ".threshold_provenance.posthoc_endpoint_completion == false",
            worker,
        )
        self.assertIn("LAST_FLOAT32_BELOW_ONE", evaluator)
        self.assertIn("UPPER_BOUNDARY_THRESHOLD = 1.0", evaluator)
        self.assertIn(
            '"endpoint_protocol_stage": "before_formal_training"',
            evaluator,
        )


if __name__ == "__main__":
    unittest.main()
