from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "experiments/run_tpd_clean_v7_dch_formal800_finalizer.sh"
)
LAUNCHER = (
    REPO_ROOT
    / "experiments/launch_tpd_clean_v7_dch_formal800_finalizer.sh"
)
STATUS = (
    REPO_ROOT
    / "experiments/status_tpd_clean_v7_dch_formal800_finalizer.sh"
)
THIS_TEST = (
    REPO_ROOT / "tests/test_tpd_clean_v7_dch_formal800_finalizer.py"
)


class V7DCHFormal800FinalizerTests(unittest.TestCase):
    def test_only_expected_control_plane_files_exist(self) -> None:
        for path in (RUNNER, LAUNCHER, STATUS, THIS_TEST):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())

    def test_shell_scripts_are_syntactically_valid(self) -> None:
        for path in (RUNNER, LAUNCHER, STATUS):
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_embedded_python_is_syntactically_valid(self) -> None:
        pattern = re.compile(r"<<'PY'\n(.*?)\nPY", re.DOTALL)
        for path in (RUNNER, STATUS):
            snippets = pattern.findall(path.read_text(encoding="utf-8"))
            self.assertGreater(len(snippets), 0, path.name)
            for index, snippet in enumerate(snippets):
                with self.subTest(path=path.name, snippet=index):
                    compile(snippet, f"{path.name}:heredoc-{index}", "exec")

    def test_runner_has_fixed_idempotent_stage_order(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        positions = [
            text.index("run_tpd_clean_v7_dch_formal800_sweeps.py"),
            text.index("summarize_tpd_clean_v7_dch_formal800.py --write"),
            text.index(
                "validate_tpd_clean_v7_dch_formal800_completion.py"
                " \\\n        publish"
            ),
            text.index("Stage 4: resumable Mechanism Audit M"),
            text.index("finalize_tpd_clean_v7_dch.py --write"),
            text.index("finalizer_control_manifest.json"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "validate_tpd_clean_v7_dch_formal800_completion.py \\\n"
            "    verify",
            text,
        )

    def test_runner_waits_for_both_lanes_and_uses_one_global_flock(
        self,
    ) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "sctransnet-tpd-clean-v7-dch-gpu2-lane.service", text
        )
        self.assertIn(
            "sctransnet-tpd-clean-v7-dch-gpu3-lane.service", text
        )
        self.assertIn("training_lane_active", text)
        self.assertIn("formal_matrix_incomplete", text)
        self.assertIn("exit 75", text)
        self.assertIn("exit 64", text)
        self.assertIn("dch_map_unexpected_error", text)
        self.assertIn('dch_global_lock="$dch_root/.postprocess.lock"', text)
        self.assertIn("global_postprocess_lock_nonregular", text)
        self.assertIn('exec 9>>"$dch_global_lock"', text)
        self.assertNotIn('exec 9>"$dch_global_lock"', text)
        self.assertIn("flock -n 9", text)

    def test_sweeps_replay_training_gpus_and_audit_is_pinned_to_gpu2(
        self,
    ) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        sweep_call = text[
            text.index("run_tpd_clean_v7_dch_formal800_sweeps.py"):
            text.index("# Stage 2:")
        ]
        self.assertIn("--device cuda:0", sweep_call)
        self.assertNotIn("--physical-gpu", sweep_call)
        self.assertIn('bind_requested_device("cuda:0", "2")', text)
        self.assertIn(
            "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
            text,
        )
        self.assertNotIn("--physical-gpu 0", text)
        self.assertNotIn("--physical-gpu 1", text)
        self.assertNotIn("--physical-gpu 3", text)

    def test_three_frozen_locks_are_validated_without_expanding_them(
        self,
    ) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("for kind in locks.LOCK_KINDS", text)
        self.assertIn("locks.DEFAULT_LOCK_RELATIVES[kind]", text)
        self.assertIn("locks.validate_source_lock(", text)
        self.assertNotIn("freeze_source_lock(", text)

    def test_partial_outputs_are_rejected_and_audit_is_resumable(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'dch_abort "partial_or_nonregular_${dch_label}"', text
        )
        for label in (
            '"comparison_report"',
            '"completion_publish"',
            '"final_report"',
        ):
            self.assertIn(label, text)
        self.assertIn("validate_checkpoint_audit", text)
        self.assertIn("audit.write_json(output, payload, overwrite=False)", text)
        self.assertIn(
            "audit.write_json(report_path, report, overwrite=False)", text
        )
        self.assertNotIn("--overwrite", text)
        self.assertNotIn("overwrite=True", text)

    def test_existing_mechanism_report_is_exactly_rederived(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("def build_expected_report(", text)
        self.assertIn(
            "reference_after != reference_before",
            text,
        )
        self.assertIn(
            "expected_report = build_expected_report(payloads)",
            text,
        )
        self.assertIn("if report != expected_report:", text)
        self.assertIn(
            "Mechanism Audit M report differs from its 12 exact audits",
            text,
        )

    def test_final_unified_verify_precedes_control_manifest(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        final_verify = text.index(
            "# First perform one final, unified revalidation"
        )
        manifest = text.index(
            '"sctransnet_tpd_clean_v7_dch_finalizer_control_manifest_v1"'
        )
        block = text[final_verify:manifest]
        positions = [
            block.index("dch_validate_source_locks"),
            block.index(
                "validate_tpd_clean_v7_dch_formal800_completion.py"
            ),
            block.index("dch_mechanism_stage"),
            block.index("dch_verify_final_report"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "finalizer control manifest readback differs",
            text,
        )

    def test_launcher_is_restartable_and_preflight_does_not_launch(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        preflight = text.index('if [[ "${1:-}" == "--preflight" ]]')
        launch = text.index("systemd-run --user")
        self.assertLess(preflight, launch)
        self.assertIn("--property=Restart=on-failure", text)
        self.assertIn("--property=RestartPreventExitStatus=64", text)
        self.assertIn("--property=RestartSec=60", text)
        self.assertIn("--property=StartLimitIntervalSec=0", text)
        self.assertIn("/usr/bin/bash \"$dch_runner\" --preflight", text)

    def test_status_reports_every_artifact_stage_and_sha(self) -> None:
        text = STATUS.read_text(encoding="utf-8")
        self.assertIn("NRestarts", text)
        self.assertIn("TPDCLEANV7DCH_FINALIZER_ARTIFACT", text)
        self.assertIn("sha256=", text)
        for stage in (
            "sweeps",
            "summary_gates",
            "completion",
            "mechanism_checkpoint",
            "mechanism_report",
            "final_report",
            "control_manifest",
        ):
            self.assertIn(f'"{stage}"', text)
        self.assertIn("locks.DEFAULT_LOCK_RELATIVES.items()", text)

    def test_control_manifest_self_binds_all_new_sources(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        for relative in (
            "experiments/run_tpd_clean_v7_dch_formal800_finalizer.sh",
            "experiments/launch_tpd_clean_v7_dch_formal800_finalizer.sh",
            "experiments/status_tpd_clean_v7_dch_formal800_finalizer.sh",
            "tests/test_tpd_clean_v7_dch_formal800_finalizer.py",
        ):
            self.assertIn(relative, text)
        self.assertIn('"artifact_count": len(artifacts)', text)
        self.assertIn('"control_sources"', text)
        self.assertIn('"source_locks"', text)


if __name__ == "__main__":
    unittest.main()
