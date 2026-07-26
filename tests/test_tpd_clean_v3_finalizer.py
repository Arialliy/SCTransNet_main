from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "experiments/run_tpd_clean_v3_screen800_finalizer.sh"
LAUNCHER = REPO_ROOT / "experiments/launch_tpd_clean_v3_screen800_finalizer.sh"
VALIDATOR = REPO_ROOT / "experiments/validate_tpd_clean_v3_completion.py"
SUMMARIZER = REPO_ROOT / "experiments/summarize_tpd_clean_v3_screen800.py"
TRAINING_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v3_screen800_source_lock.json"
)
POSTPROCESS_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v3_postprocess_source_lock.json"
)
VARIANT_SEEDS = (
    ("tpd_clean_v3_full", 42),
    ("tpd_clean_v3_sal_capacity", 42),
    ("tpd_clean_v3_full", 3407),
    ("tpd_clean_v3_sal_capacity", 3407),
)
RUN_TAG = "screen800_pd_fp32_shared4x5090_v1"
INTEGRITY_CHECKS = (
    "summary_complete",
    "metrics_complete_contiguous_finite",
    "metadata_consistent",
    "official_test_isolated",
    "split_hashes_recomputed_consistent",
    "checkpoint_role_epoch_metrics_consistent",
    "global_selection_keys_recomputed",
    "state_dict_strict_load",
    "fixed_threshold_object_metrics_exact",
)
BUDGETS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_candidate_run(root: Path, variant: str, seed: int) -> Path:
    run_dir = (
        root
        / "NUDT-SIRST"
        / variant
        / f"seed_{seed}_{RUN_TAG}"
    )
    run_dir.mkdir(parents=True)
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps(
                {
                    "epoch": epoch,
                    "variant": variant,
                    "train_loss": 0.1,
                    "learning_rate": 0.001,
                    "epoch_seconds": 1.0,
                }
            )
            + "\n"
            for epoch in range(1, 801)
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "variant": variant,
                "dataset": "NUDT-SIRST",
                "seed": seed,
                "selection_source": "internal_validation_only",
                "official_test_accessed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for checkpoint_name, sweep_name, checkpoint_role in (
        (
            "best.pth.tar",
            "pd_fa_sweep_best.pth.json",
            "best_validation_pd_primary",
        ),
        (
            "best_miou.pth.tar",
            "pd_fa_sweep_best_miou.pth.json",
            "best_validation_miou_secondary",
        ),
    ):
        checkpoint = run_dir / checkpoint_name
        checkpoint.write_bytes(
            f"{variant}:{seed}:{checkpoint_name}".encode("utf-8")
        )
        checkpoint_sha = _sha256(checkpoint)
        sweep = {
            "variant": variant,
            "seed": seed,
            "dataset": "NUDT-SIRST",
            "checkpoint": str(checkpoint),
            "checkpoint_role": checkpoint_role,
            "checkpoint_sha256": checkpoint_sha,
            "official_test_accessed": False,
            "fixed_threshold_0_5_checkpoint_audit": {
                "max_abs_non_strict_numeric_delta": 0.0
            },
            "best_points_under_fa_budget": {
                budget: {"matched_target_count": 188}
                for budget in BUDGETS
            },
            "audit": {
                "expected_epochs": 800,
                "metrics_event_count": 800,
                "metrics_epoch_range": [1, 800],
                "integrity_checks_passed": {
                    name: True for name in INTEGRITY_CHECKS
                },
                "artifact_sha256": {"checkpoint": checkpoint_sha},
            },
        }
        (run_dir / sweep_name).write_text(
            json.dumps(sweep) + "\n", encoding="utf-8"
        )
    return run_dir


def _write_all_candidate_runs(root: Path) -> dict[tuple[str, int], Path]:
    return {
        (variant, seed): _write_candidate_run(root, variant, seed)
        for variant, seed in VARIANT_SEEDS
    }


def _write_worker_completion_evidence(
    root: Path,
    runs: dict[tuple[str, int], Path],
) -> None:
    launch = root / "launch"
    logs = root / "logs"
    launch.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    for (variant, seed), run_dir in runs.items():
        (launch / f"{variant}_seed{seed}.json").write_text(
            json.dumps(
                {
                    "schema": "sctransnet_tpd_clean_v3_screen800_launch_v1",
                    "variant": variant,
                    "seed": seed,
                    "run_directory": str(run_dir),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (logs / f"{variant}_seed{seed}.log").write_text(
            "TPDCLEANV3_COMPLETE"
            f" variant={variant}"
            f" seed={seed}"
            " gpu_uuid=GPU-test"
            " epochs=800\n",
            encoding="utf-8",
        )


def _write_fake_systemctl(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import os
import pathlib
import sys

mode = os.environ.get("FAKE_SYSTEMCTL_MODE", "success")
counter_path = pathlib.Path(os.environ["FAKE_SYSTEMCTL_COUNTER"])
count = int(counter_path.read_text() or "0") if counter_path.exists() else 0
count += 1
counter_path.write_text(str(count))
unit = next((value for value in sys.argv if value.endswith(".service")), "")

load = "loaded"
active = "inactive"
sub = "dead"
result = "success"
code = "exited"
status = "0"
if mode == "active_then_success" and count <= 4:
    active = "active"
    sub = "running"
elif mode == "active_then_collected":
    if count <= 4:
        active = "active"
        sub = "running"
    else:
        load = "not-found"
        active = "inactive"
        sub = "dead"
        result = ""
        code = ""
        status = ""
elif mode == "failed" and "cap-s42" in unit:
    active = "failed"
    sub = "failed"
    result = "exit-code"
    status = "7"
elif mode == "nonzero_active" and "full-s42" in unit:
    active = "active"
    sub = "running"
    status = "9"

print(f"LoadState={load}")
print(f"ActiveState={active}")
print(f"SubState={sub}")
print(f"Result={result}")
print(f"ExecMainCode={code}")
print(f"ExecMainStatus={status}")
""",
        encoding="utf-8",
    )
    _make_executable(path)


def _runner_environment(
    root: Path,
    *,
    systemctl_mode: str = "success",
) -> tuple[dict[str, str], Path, Path]:
    candidate = root / "candidate"
    formal = root / "formal"
    v2 = root / "v2"
    miou = root / "miou"
    for directory in (candidate, formal, v2, miou):
        directory.mkdir(parents=True, exist_ok=True)
    fake_systemctl = root / "fake-systemctl"
    systemctl_counter = root / "systemctl-count.txt"
    _write_fake_systemctl(fake_systemctl)
    environment = os.environ.copy()
    environment.update(
        {
            "TPDCLEANV3_FINALIZER_REPO": str(REPO_ROOT),
            "TPDCLEANV3_FINALIZER_PYTHON": sys.executable,
            "TPDCLEANV3_FINALIZER_RESULT_ROOT": str(candidate),
            "TPDCLEANV3_FINALIZER_FORMAL_ROOT": str(formal),
            "TPDCLEANV3_FINALIZER_V2_ROOT": str(v2),
            "TPDCLEANV3_FINALIZER_REFERENCE_MIOU_ROOT": str(miou),
            "TPDCLEANV3_FINALIZER_SUMMARIZER": str(SUMMARIZER),
            "TPDCLEANV3_FINALIZER_COMPLETION_VALIDATOR": str(VALIDATOR),
            "TPDCLEANV3_FINALIZER_POSTPROCESS_LOCK": str(POSTPROCESS_LOCK),
            "TPDCLEANV3_FINALIZER_SYSTEMCTL": str(fake_systemctl),
            "TPDCLEANV3_FINALIZER_POLL_SECONDS": "0.01",
            "FAKE_SYSTEMCTL_MODE": systemctl_mode,
            "FAKE_SYSTEMCTL_COUNTER": str(systemctl_counter),
        }
    )
    return environment, candidate, systemctl_counter


def _run_finalizer(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TPDCleanV3FinalizerTests(unittest.TestCase):
    def test_shell_entrypoints_parse_and_default_to_sixty_seconds(self) -> None:
        subprocess.run(
            ["bash", "-n", str(RUNNER), str(LAUNCHER)],
            cwd=REPO_ROOT,
            check=True,
        )
        source = RUNNER.read_text(encoding="utf-8")
        for unit in (
            "sctransnet-tpd-clean-v3-full-s42.service",
            "sctransnet-tpd-clean-v3-cap-s42.service",
            "sctransnet-tpd-clean-v3-full-s3407.service",
            "sctransnet-tpd-clean-v3-cap-s3407.service",
        ):
            with self.subTest(unit=unit):
                self.assertIn(unit, source)
        self.assertIn("FINALIZER_POLL_SECONDS:-60", source)
        self.assertIn("checkpoint_total != 8", source)
        self.assertIn("sweep_total != 8", source)

    def test_training_source_lock_remains_unchanged_and_excludes_finalizer(self) -> None:
        payload = json.loads(TRAINING_LOCK.read_text(encoding="utf-8"))
        self.assertNotIn(
            "experiments/run_tpd_clean_v3_screen800_finalizer.sh",
            payload["source_sha256"],
        )
        self.assertNotIn(
            "experiments/launch_tpd_clean_v3_screen800_finalizer.sh",
            payload["source_sha256"],
        )
        for relative, expected in payload["source_sha256"].items():
            with self.subTest(relative=relative):
                self.assertEqual(_sha256(REPO_ROOT / relative), expected)

    def test_postprocess_source_lock_covers_runtime_and_matches(self) -> None:
        payload = json.loads(POSTPROCESS_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["schema"],
            "sctransnet_tpd_clean_v3_postprocess_source_lock_v1",
        )
        entries = payload["source_sha256"]
        self.assertTrue(
            {
                "experiments/summarize_tpd_clean_v3_screen800.py",
                "experiments/validate_tpd_clean_v3_completion.py",
                "experiments/run_tpd_clean_v3_screen800_finalizer.sh",
                "experiments/launch_tpd_clean_v3_screen800_finalizer.sh",
                "tests/test_summarize_tpd_clean_v3_screen800.py",
                "tests/test_validate_tpd_clean_v3_completion.py",
                "tests/test_tpd_clean_v3_finalizer.py",
            }.issubset(entries)
        )
        for relative, expected in entries.items():
            with self.subTest(relative=relative):
                self.assertEqual(_sha256(REPO_ROOT / relative), expected)

    def test_locked_staging_publish_verify_pipeline_is_ordered(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        validator_source = VALIDATOR.read_text(encoding="utf-8")

        self.assertIn(
            'v3_completion_validator="${TPDCLEANV3_FINALIZER_COMPLETION_VALIDATOR'
            ':-$v3_repo/experiments/validate_tpd_clean_v3_completion.py}"',
            source,
        )
        self.assertIn(
            'v3_postprocess_lock="${TPDCLEANV3_FINALIZER_POSTPROCESS_LOCK'
            ':-$v3_repo/experiments/tpd_clean_v3_postprocess_source_lock.json}"',
            source,
        )

        flock_index = source.index("flock -n 9")
        verify_wrapper_index = source.index(
            'v3_verify_complete_marker() {'
        )
        publish_wrapper_index = source.index(
            'v3_publish_complete_bundle() {'
        )
        self.assertIn(
            '"$v3_completion_validator" verify',
            source[verify_wrapper_index:publish_wrapper_index],
        )
        self.assertIn(
            '"$v3_completion_validator" publish',
            source[publish_wrapper_index:flock_index],
        )

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

        summarizer_call_index = source.index(
            'if ! "$v3_python" "$v3_summarizer"', staging_index
        )
        summarizer_call_end = source.index("\nthen", summarizer_call_index)
        summarizer_call = source[
            summarizer_call_index:summarizer_call_end
        ]
        self.assertIn('--output-dir "$v3_staging_dir"', summarizer_call)
        self.assertNotIn('--output-dir "$v3_comparison_dir"', summarizer_call)
        self.assertNotIn("--overwrite", summarizer_call)
        self.assertIn('--staging-dir "$v3_staging_dir"', source)
        self.assertNotIn("--overwrite", source)
        self.assertNotIn("v3_create_complete_marker()", source)

        self.assertIn('MANIFEST_NAME = "completion_inputs.json"', validator_source)
        marker_names_index = validator_source.index("marker_names = (")
        marker_names = validator_source[
            marker_names_index : marker_names_index + 240
        ]
        self.assertIn("module.JSON_OUTPUT_NAME", marker_names)
        self.assertIn("module.MARKDOWN_OUTPUT_NAME", marker_names)
        self.assertIn("MANIFEST_NAME", marker_names)

    def test_active_units_are_polled_before_artifact_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, candidate, systemctl_counter = (
                _runner_environment(
                    root,
                    systemctl_mode="active_then_success",
                )
            )
            runs = _write_all_candidate_runs(candidate)
            (
                runs[("tpd_clean_v3_full", 3407)]
                / "pd_fa_sweep_best_miou.pth.json"
            ).unlink()

            completed = _run_finalizer(environment)
            self.assertNotEqual(completed.returncode, 0)
            combined = completed.stdout + completed.stderr
            self.assertIn("TPDCLEANV3_FINALIZER_WAITING", combined)
            self.assertIn("artifact_validation", combined)
            self.assertGreaterEqual(int(systemctl_counter.read_text()), 8)
            comparison = candidate / "NUDT-SIRST/comparison"
            self.assertFalse((comparison / "COMPLETE.sha256").exists())
            self.assertFalse(
                (comparison / "completion_inputs.json").exists()
            )
            self.assertEqual(list(comparison.glob(".staging.*")), [])

    def test_failed_or_nonzero_worker_is_recorded_before_aggregation(self) -> None:
        for mode in ("failed", "nonzero_active"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                environment, candidate, _ = (
                    _runner_environment(root, systemctl_mode=mode)
                )
                completed = _run_finalizer(environment)
                self.assertNotEqual(completed.returncode, 0)
                combined = completed.stdout + completed.stderr
                self.assertIn("TPDCLEANV3_FINALIZER_FAILED", combined)
                state = json.loads(
                    (
                        candidate / "launch/finalizer_state.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(state["state"], "failed")
                comparison = candidate / "NUDT-SIRST/comparison"
                self.assertFalse((comparison / "COMPLETE.sha256").exists())
                self.assertEqual(list(comparison.glob(".staging.*")), [])

    def test_successfully_unloaded_transient_units_use_worker_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, candidate, systemctl_counter = (
                _runner_environment(
                    root,
                    systemctl_mode="active_then_collected",
                )
            )
            runs = _write_all_candidate_runs(candidate)
            _write_worker_completion_evidence(candidate, runs)
            invalid_sweep = (
                runs[("tpd_clean_v3_full", 3407)]
                / "pd_fa_sweep_best_miou.pth.json"
            )
            payload = json.loads(invalid_sweep.read_text(encoding="utf-8"))
            payload["best_points_under_fa_budget"] = {}
            invalid_sweep.write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            completed = _run_finalizer(environment)
            self.assertNotEqual(completed.returncode, 0)
            combined = completed.stdout + completed.stderr
            self.assertIn("TPDCLEANV3_FINALIZER_WAITING", combined)
            self.assertIn("artifact_validation", combined)
            self.assertNotIn(
                "unloaded without complete worker evidence", combined
            )
            self.assertGreaterEqual(int(systemctl_counter.read_text()), 8)
            comparison = candidate / "NUDT-SIRST/comparison"
            self.assertFalse((comparison / "COMPLETE.sha256").exists())
            self.assertEqual(list(comparison.glob(".staging.*")), [])

    def test_missing_sweep_blocks_summarizer_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, candidate, _ = _runner_environment(root)
            runs = _write_all_candidate_runs(candidate)
            (
                runs[("tpd_clean_v3_full", 3407)]
                / "pd_fa_sweep_best_miou.pth.json"
            ).unlink()
            completed = _run_finalizer(environment)
            self.assertNotEqual(completed.returncode, 0)
            combined = completed.stdout + completed.stderr
            self.assertIn("artifact_validation", combined)
            comparison = candidate / "NUDT-SIRST/comparison"
            self.assertFalse((comparison / "COMPLETE.sha256").exists())
            self.assertFalse((comparison / "completion_inputs.json").exists())
            self.assertEqual(list(comparison.glob(".staging.*")), [])

    def test_launcher_preflight_never_starts_a_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_systemctl = root / "systemctl"
            fake_systemd_run = root / "systemd-run"
            _write_fake_systemctl(fake_systemctl)
            started = root / "systemd-run-called"
            fake_systemd_run.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {started!s}\n",
                encoding="utf-8",
            )
            _make_executable(fake_systemd_run)
            environment = os.environ.copy()
            environment.update(
                {
                    "TPDCLEANV3_FINALIZER_REPO": str(REPO_ROOT),
                    "TPDCLEANV3_FINALIZER_RUNNER": str(RUNNER),
                    "TPDCLEANV3_FINALIZER_SUMMARIZER": str(SUMMARIZER),
                    "TPDCLEANV3_FINALIZER_COMPLETION_VALIDATOR": str(
                        VALIDATOR
                    ),
                    "TPDCLEANV3_FINALIZER_POSTPROCESS_LOCK": str(
                        POSTPROCESS_LOCK
                    ),
                    "TPDCLEANV3_FINALIZER_PYTHON": sys.executable,
                    "TPDCLEANV3_FINALIZER_RESULT_ROOT": str(
                        root / "candidate"
                    ),
                    "TPDCLEANV3_FINALIZER_SYSTEMCTL": str(fake_systemctl),
                    "TPDCLEANV3_FINALIZER_SYSTEMD_RUN": str(fake_systemd_run),
                    "FAKE_SYSTEMCTL_COUNTER": str(
                        root / "systemctl-count.txt"
                    ),
                }
            )
            completed = subprocess.run(
                ["bash", str(LAUNCHER), "--preflight"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertIn("PREFLIGHT_OK", completed.stdout)
            self.assertFalse(started.exists())


if __name__ == "__main__":
    unittest.main()
