from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_4X = (
    REPO_ROOT / "experiments/run_tpd_clean_v3_screen800_finalizer.sh"
)
LAUNCHER_4X = (
    REPO_ROOT / "experiments/launch_tpd_clean_v3_screen800_finalizer.sh"
)
RUNNER_2X = (
    REPO_ROOT
    / "experiments/run_tpd_clean_v3_screen800_2x5090_finalizer.sh"
)
LAUNCHER_2X = (
    REPO_ROOT
    / "experiments/launch_tpd_clean_v3_screen800_2x5090_finalizer.sh"
)

REPLACEMENTS = (
    ("TPDCLEANV3_", "TPDCLEANV3_2X_"),
    (
        "tpd_clean_v3_screen800_4x5090_v1",
        "tpd_clean_v3_screen800_2x5090_v1",
    ),
    (
        "screen800_pd_fp32_shared4x5090_v1",
        "screen800_pd_fp32_shared2x5090_v1",
    ),
    (
        "summarize_tpd_clean_v3_screen800.py",
        "summarize_tpd_clean_v3_screen800_2x5090.py",
    ),
    (
        "validate_tpd_clean_v3_completion.py",
        "validate_tpd_clean_v3_2x_completion.py",
    ),
    (
        "tpd_clean_v3_postprocess_source_lock.json",
        "tpd_clean_v3_2x_postprocess_source_lock.json",
    ),
    (
        "run_tpd_clean_v3_screen800_finalizer.sh",
        "run_tpd_clean_v3_screen800_2x5090_finalizer.sh",
    ),
    (
        "launch_tpd_clean_v3_screen800_finalizer.sh",
        "launch_tpd_clean_v3_screen800_2x5090_finalizer.sh",
    ),
    ("sctransnet-tpd-clean-v3-", "sctransnet-tpd-clean-v3-2x-"),
    (
        "sctransnet_tpd_clean_v3_screen800_launch_v1",
        "sctransnet_tpd_clean_v3_screen800_2x5090_launch_v1",
    ),
    (
        "sctransnet_tpd_clean_v3_postprocess_source_lock_v1",
        "sctransnet_tpd_clean_v3_2x_postprocess_source_lock_v1",
    ),
    (
        "sctransnet_tpd_clean_v3_screen800_finalizer_state_v1",
        "sctransnet_tpd_clean_v3_2x_screen800_finalizer_state_v1",
    ),
)


def _to_2x(source: str) -> str:
    for old, new in REPLACEMENTS:
        source = source.replace(old, new)
    return source


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TPDCleanV32XFinalizerTests(unittest.TestCase):
    def test_shell_entrypoints_parse(self) -> None:
        subprocess.run(
            ["bash", "-n", str(RUNNER_2X), str(LAUNCHER_2X)],
            cwd=REPO_ROOT,
            check=True,
        )

    def test_two_card_files_are_exact_isolated_template_copies(self) -> None:
        self.assertEqual(
            RUNNER_2X.read_text(encoding="utf-8"),
            _to_2x(RUNNER_4X.read_text(encoding="utf-8")),
        )
        self.assertEqual(
            LAUNCHER_2X.read_text(encoding="utf-8"),
            _to_2x(LAUNCHER_4X.read_text(encoding="utf-8")),
        )

    def test_two_card_paths_units_schemas_and_outputs_are_bound(self) -> None:
        runner = RUNNER_2X.read_text(encoding="utf-8")
        launcher = LAUNCHER_2X.read_text(encoding="utf-8")
        combined = runner + launcher

        for required in (
            "tpd_clean_v3_screen800_2x5090_v1",
            "screen800_pd_fp32_shared2x5090_v1",
            "summarize_tpd_clean_v3_screen800_2x5090.py",
            "validate_tpd_clean_v3_2x_completion.py",
            "tpd_clean_v3_2x_postprocess_source_lock.json",
            "sctransnet_tpd_clean_v3_screen800_2x5090_launch_v1",
            "TPDCLEANV3_2X_COMPLETE",
            "TPDCLEANV3_2X_FINALIZER_",
        ):
            with self.subTest(required=required):
                self.assertIn(required, combined)

        for unit in (
            "sctransnet-tpd-clean-v3-2x-full-s42.service",
            "sctransnet-tpd-clean-v3-2x-cap-s42.service",
            "sctransnet-tpd-clean-v3-2x-full-s3407.service",
            "sctransnet-tpd-clean-v3-2x-cap-s3407.service",
        ):
            with self.subTest(unit=unit):
                self.assertIn(unit, runner)
        self.assertIn(
            'v3_unit="sctransnet-tpd-clean-v3-2x-screen800-finalizer.service"',
            launcher,
        )
        self.assertIn(
            '"$v3_staging_dir/tpd_clean_v3_screen800_comparison.json"',
            runner,
        )
        self.assertIn(
            '"$v3_staging_dir/tpd_clean_v3_screen800_comparison.md"',
            runner,
        )

        for forbidden in (
            "tpd_clean_v3_screen800_4x5090_v1",
            "screen800_pd_fp32_shared4x5090_v1",
            "sctransnet-tpd-clean-v3-full-s42.service",
            "sctransnet-tpd-clean-v3-cap-s42.service",
            'v3_unit="sctransnet-tpd-clean-v3-screen800-finalizer.service"',
            "sctransnet_tpd_clean_v3_screen800_launch_v1",
            "TPDCLEANV3_FINALIZER_",
            "TPDCLEANV3_COMPLETE",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, combined)

    def test_locked_staging_publish_verify_order_is_preserved(self) -> None:
        source = RUNNER_2X.read_text(encoding="utf-8")
        flock_index = source.index("flock -n 9")
        first_verify_index = source.index(
            "if v3_verify_complete_marker", flock_index
        )
        staging_index = source.index(
            'mktemp -d "$v3_comparison_dir/.staging.', first_verify_index
        )
        publish_index = source.index(
            'v3_publish_complete_bundle "$v3_staging_dir"', staging_index
        )
        final_verify_index = source.index(
            "if ! v3_verify_complete_marker", publish_index
        )
        self.assertLess(flock_index, first_verify_index)
        self.assertLess(first_verify_index, staging_index)
        self.assertLess(staging_index, publish_index)
        self.assertLess(publish_index, final_verify_index)
        self.assertIn('--output-dir "$v3_staging_dir"', source)
        self.assertIn('--staging-dir "$v3_staging_dir"', source)
        self.assertNotIn("--overwrite", source)

    def test_launcher_preflight_does_not_start_a_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            repo = temporary_root / "repo"
            experiments = repo / "experiments"
            experiments.mkdir(parents=True)

            runner = experiments / RUNNER_2X.name
            launcher = experiments / LAUNCHER_2X.name
            summarizer = (
                experiments / "summarize_tpd_clean_v3_screen800_2x5090.py"
            )
            validator = experiments / "validate_tpd_clean_v3_2x_completion.py"
            postprocess_lock = (
                experiments / "tpd_clean_v3_2x_postprocess_source_lock.json"
            )
            shutil.copy2(RUNNER_2X, runner)
            shutil.copy2(LAUNCHER_2X, launcher)
            summarizer.write_text("# test summarizer\n", encoding="utf-8")
            validator.write_text("# test validator\n", encoding="utf-8")
            _make_executable(runner)
            _make_executable(launcher)

            lock_entries = {
                f"experiments/{path.name}": _sha256(path)
                for path in (summarizer, validator, runner, launcher)
            }
            postprocess_lock.write_text(
                json.dumps(
                    {
                        "schema": (
                            "sctransnet_tpd_clean_v3_2x_"
                            "postprocess_source_lock_v1"
                        ),
                        "source_sha256": lock_entries,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            fake_systemctl = temporary_root / "systemctl"
            fake_systemd_run = temporary_root / "systemd-run"
            started = temporary_root / "service-started"
            fake_systemctl.write_text(
                "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
            )
            fake_systemd_run.write_text(
                "#!/usr/bin/env bash\n"
                ': > "$TPDCLEANV3_2X_TEST_STARTED"\n',
                encoding="utf-8",
            )
            _make_executable(fake_systemctl)
            _make_executable(fake_systemd_run)

            environment = os.environ.copy()
            environment.update(
                {
                    "TPDCLEANV3_2X_FINALIZER_REPO": str(repo),
                    "TPDCLEANV3_2X_FINALIZER_RUNNER": str(runner),
                    "TPDCLEANV3_2X_FINALIZER_SUMMARIZER": str(summarizer),
                    "TPDCLEANV3_2X_FINALIZER_COMPLETION_VALIDATOR": str(
                        validator
                    ),
                    "TPDCLEANV3_2X_FINALIZER_POSTPROCESS_LOCK": str(
                        postprocess_lock
                    ),
                    "TPDCLEANV3_2X_FINALIZER_PYTHON": sys.executable,
                    "TPDCLEANV3_2X_FINALIZER_RESULT_ROOT": str(
                        temporary_root / "candidate"
                    ),
                    "TPDCLEANV3_2X_FINALIZER_SYSTEMCTL": str(fake_systemctl),
                    "TPDCLEANV3_2X_FINALIZER_SYSTEMD_RUN": str(
                        fake_systemd_run
                    ),
                    "TPDCLEANV3_2X_TEST_STARTED": str(started),
                }
            )
            completed = subprocess.run(
                ["bash", str(launcher), "--preflight"],
                cwd=repo,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertIn(
                "TPDCLEANV3_2X_FINALIZER_PREFLIGHT_OK",
                completed.stdout,
            )
            self.assertFalse(started.exists())


if __name__ == "__main__":
    unittest.main()
