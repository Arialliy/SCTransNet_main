from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from experiments import summarize_tpd_clean_v6_formal800 as summary


RUNNER = (
    summary.REPO_ROOT
    / "experiments/run_tpd_clean_v6_formal800_finalizer.sh"
)
LAUNCHER = (
    summary.REPO_ROOT
    / "experiments/launch_tpd_clean_v6_formal800_finalizer.sh"
)
STATUS = (
    summary.REPO_ROOT
    / "experiments/status_tpd_clean_v6_formal800_finalizer.sh"
)


class V6Formal800FinalizerTests(unittest.TestCase):
    def test_shell_scripts_are_syntactically_valid(self) -> None:
        for path in (RUNNER, LAUNCHER, STATUS):
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_finalizer_order_is_sweep_summary_publish_verify(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        positions = [
            text.index("run_tpd_clean_v6_formal800_sweeps.py"),
            text.index("summarize_tpd_clean_v6_formal800.py --write"),
            text.index(
                "validate_tpd_clean_v6_formal800_completion.py publish"
            ),
            text.index(
                "validate_tpd_clean_v6_formal800_completion.py verify"
            ),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("formal_matrix_incomplete", text)
        self.assertIn("exit 75", text)
        self.assertIn("flock -n", text)
        self.assertNotIn("--overwrite", text)

    def test_finalizer_is_pinned_to_physical_gpu2_uuid(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--physical-gpu 2", text)
        self.assertIn(
            "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
            text,
        )
        self.assertNotIn("--physical-gpu 0", text)
        self.assertNotIn("--physical-gpu 1", text)

    def test_launcher_uses_restart_on_failure_and_has_read_only_preflight(
        self,
    ) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--property=Restart=on-failure", text)
        self.assertIn("--property=RestartSec=60", text)
        self.assertIn("--preflight", text)
        self.assertIn("--physical-gpu 2", text)

    def test_status_reports_service_training_and_four_outputs(self) -> None:
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("NRestarts", text)
        self.assertIn("summarize_tpd_clean_v6_formal800.py", text)
        for name in (
            summary.JSON_OUTPUT_NAME,
            summary.MARKDOWN_OUTPUT_NAME,
            "completion_inputs.json",
            "COMPLETE.sha256",
        ):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
