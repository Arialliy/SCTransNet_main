from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/run_tpd_clean_v3_resume_finalizer.sh"
LAUNCHER = REPO / "experiments/launch_tpd_clean_v3_resume_finalizer.sh"
VALIDATOR = REPO / "experiments/validate_tpd_clean_v3_resume_completion.py"
POSTPROCESS_LOCK = (
    REPO / "experiments/tpd_clean_v3_resume_postprocess_source_lock.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResumeFinalizerTests(unittest.TestCase):
    def test_postprocess_source_lock_binds_runtime_and_tests(self) -> None:
        payload = json.loads(POSTPROCESS_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(
            payload.get("schema"),
            "sctransnet_tpd_clean_v3_resume_postprocess_source_lock_v1",
        )
        expected_entries = {
            "experiments/summarize_tpd_clean_v3_screen800.py",
            "experiments/validate_tpd_clean_v3_completion.py",
            "experiments/validate_tpd_clean_v3_resume_completion.py",
            "experiments/run_tpd_clean_v3_resume_finalizer.sh",
            "experiments/launch_tpd_clean_v3_resume_finalizer.sh",
            "tests/test_summarize_tpd_clean_v3_screen800.py",
            "tests/test_validate_tpd_clean_v3_completion.py",
            "tests/test_validate_tpd_clean_v3_resume_completion.py",
            "tests/test_tpd_clean_v3_resume_finalizer.py",
            "experiments/TPD_CLEAN_V3_RESUME_2GPU_PROTOCOL.md",
            "experiments/tpd_clean_v3_screen800_source_lock.json",
            "experiments/tpd_clean_v3_resume_2x_source_lock.json",
            "experiments/TPD_CLEAN_V3_CLOSED_INTERVAL_SWEEP_RECOVERY.md",
            "experiments/evaluate_tpd_clean_v3_pd_fa_closed_interval.py",
            "experiments/audit_tpd_clean_v3_closed_interval_sweeps.py",
            "tests/test_evaluate_tpd_clean_v3_pd_fa_closed_interval.py",
        }
        entries = payload.get("source_sha256")
        self.assertIsInstance(entries, dict)
        self.assertEqual(set(entries), expected_entries)
        for relative, expected_sha in entries.items():
            source = REPO / relative
            self.assertTrue(source.is_file(), relative)
            self.assertFalse(source.is_symlink(), relative)
            self.assertEqual(_sha256(source), expected_sha, relative)

        training_lock = REPO / "experiments/tpd_clean_v3_screen800_source_lock.json"
        resume_lock = REPO / "experiments/tpd_clean_v3_resume_2x_source_lock.json"
        self.assertEqual(
            payload.get("training_source_lock_sha256"),
            _sha256(training_lock),
        )
        self.assertEqual(
            payload.get("resume_source_lock_sha256"),
            _sha256(resume_lock),
        )
        self.assertEqual(
            payload.get("policy"),
            {
                "separate_from_training_and_resume_source_locks": True,
                "does_not_modify_training_or_resume_results": True,
                "marker_written_last": True,
                "published_outputs_are_no_overwrite": True,
                "closed_interval_candidate_sweeps": True,
                "historical_reference_sweeps_unchanged": True,
                "automatic_mainline_replacement": False,
            },
        )

    def test_shell_syntax(self) -> None:
        for path in (RUNNER, LAUNCHER):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runner_binds_fixed_workers_boundaries_and_isolated_outputs(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        for unit in (
            "sctransnet-tpd-clean-v3-resume-2x-full-s42.service",
            "sctransnet-tpd-clean-v3-resume-2x-cap-s42.service",
            "sctransnet-tpd-clean-v3-resume-2x-full-s3407.service",
            "sctransnet-tpd-clean-v3-resume-2x-cap-s3407.service",
        ):
            self.assertIn(unit, text)
        self.assertIn("v3_boundaries=(279 331 323 372)", text)
        self.assertIn("v3_gpu_indices=(3 2 2 3)", text)
        self.assertIn("screen800_pd_fp32_shared4x5090_v1", text)
        self.assertIn("tpd_clean_v3_screen800_4x5090_v1", text)
        self.assertIn("RESUME_COMPLETE.sha256", text)
        self.assertIn(".resume-staging.XXXXXX", text)
        self.assertIn("tpd_clean_v3_screen800_comparison.json", text)
        self.assertIn("tpd_clean_v3_screen800_comparison.md", text)

    def test_runner_audits_before_summary_and_verifies_after_publish(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        audit = text.index("if ! v3_audit_completion")
        summarize = text.index('if ! "$v3_python" "$v3_summarizer"')
        publish = text.index('if ! v3_publish_completion "$v3_staging_dir"')
        verify = text.index("if ! v3_verify_completion", publish)
        cleanup = text.index("rmdir \"$v3_staging_dir\"")
        self.assertLess(audit, summarize)
        self.assertLess(summarize, publish)
        self.assertLess(publish, verify)
        self.assertLess(verify, cleanup)
        self.assertNotIn("COMPLETE.sha256", text.replace("RESUME_COMPLETE.sha256", ""))

    def test_launcher_uses_dedicated_service(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'v3_unit="sctransnet-tpd-clean-v3-resume-finalizer.service"',
            text,
        )
        self.assertIn("--collect", text)
        self.assertIn("--property=Restart=no", text)
        self.assertIn("RESUME_COMPLETE.sha256", text)
        self.assertIn("--preflight", text)

    def test_validator_uses_resume_prefixed_publication_names(self) -> None:
        text = VALIDATOR.read_text(encoding="utf-8")
        self.assertIn(
            'PUBLISHED_JSON_NAME = "resume_tpd_clean_v3_screen800_comparison.json"',
            text,
        )
        self.assertIn(
            'PUBLISHED_MARKDOWN_NAME = "resume_tpd_clean_v3_screen800_comparison.md"',
            text,
        )
        self.assertIn('MANIFEST_NAME = "resume_completion_inputs.json"', text)
        self.assertIn('MARKER_NAME = "RESUME_COMPLETE.sha256"', text)
        self.assertIn('with path.open("xb")', text)

    def test_launcher_preflight_does_not_invoke_service_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_repo = root / "repo"
            experiments = fake_repo / "experiments"
            experiments.mkdir(parents=True)
            relative_sources = (
                "experiments/summarize_tpd_clean_v3_screen800.py",
                "experiments/validate_tpd_clean_v3_completion.py",
                "experiments/validate_tpd_clean_v3_resume_completion.py",
                "experiments/run_tpd_clean_v3_resume_finalizer.sh",
                "experiments/launch_tpd_clean_v3_resume_finalizer.sh",
            )
            for relative in relative_sources:
                source = REPO / relative
                target = fake_repo / relative
                shutil.copyfile(source, target)
            fake_runner = experiments / "run_tpd_clean_v3_resume_finalizer.sh"
            fake_launcher = experiments / "launch_tpd_clean_v3_resume_finalizer.sh"
            fake_runner.chmod(0o755)
            fake_launcher.chmod(0o755)

            training_lock = experiments / "tpd_clean_v3_screen800_source_lock.json"
            resume_lock = experiments / "tpd_clean_v3_resume_2x_source_lock.json"
            training_lock.write_text('{"fixture":"training"}\n', encoding="utf-8")
            resume_lock.write_text('{"fixture":"resume"}\n', encoding="utf-8")
            lock = {
                "schema": "sctransnet_tpd_clean_v3_resume_postprocess_source_lock_v1",
                "training_source_lock_sha256": _sha256(training_lock),
                "resume_source_lock_sha256": _sha256(resume_lock),
                "source_sha256": {
                    relative: _sha256(fake_repo / relative)
                    for relative in relative_sources
                },
            }
            postprocess_lock = (
                experiments
                / "tpd_clean_v3_resume_postprocess_source_lock.json"
            )
            postprocess_lock.write_text(
                json.dumps(lock, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            forbidden_systemctl = root / "forbidden-systemctl"
            forbidden_systemd_run = root / "forbidden-systemd-run"
            for path in (forbidden_systemctl, forbidden_systemd_run):
                path.write_text("#!/usr/bin/env bash\nexit 91\n", encoding="utf-8")
                path.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "TPDCLEANV3_RESUME_FINALIZER_REPO": str(fake_repo),
                    "TPDCLEANV3_RESUME_FINALIZER_PYTHON": sys.executable,
                    "TPDCLEANV3_RESUME_FINALIZER_SYSTEMCTL": str(
                        forbidden_systemctl
                    ),
                    "TPDCLEANV3_RESUME_FINALIZER_SYSTEMD_RUN": str(
                        forbidden_systemd_run
                    ),
                }
            )
            completed = subprocess.run(
                [str(fake_launcher), "--preflight"],
                cwd=fake_repo,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("TPDCLEANV3_RESUME_FINALIZER_PREFLIGHT_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
