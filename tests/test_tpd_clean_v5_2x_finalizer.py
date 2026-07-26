from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

from experiments.validate_tpd_clean_v5_2x_completion import (
    POSTPROCESS_SOURCE_SET,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "experiments/run_tpd_clean_v5_screen800_2x5090_finalizer.sh"
)
LAUNCHER = (
    REPO_ROOT
    / "experiments/launch_tpd_clean_v5_screen800_2x5090_finalizer.sh"
)
SUMMARIZER = REPO_ROOT / "experiments/summarize_tpd_clean_v5_screen800.py"
VALIDATOR = (
    REPO_ROOT / "experiments/validate_tpd_clean_v5_2x_completion.py"
)
POSTPROCESS_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v5_2x_postprocess_source_lock.json"
)
TRAINING_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v5_screen800_2x_source_lock.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_finalizer_scripts_are_executable_and_parse() -> None:
    for script in (RUNNER, LAUNCHER):
        assert script.is_file()
        assert not script.is_symlink()
        assert script.stat().st_mode & stat.S_IXUSR
        subprocess.run(
            ["bash", "-n", str(script)],
            cwd=REPO_ROOT,
            check=True,
        )


def test_runner_waits_for_exact_four_worker_bundles() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    for tag in ("full-s42", "cap-s42", "full-s3407", "cap-s3407"):
        assert f"sctransnet-tpd-clean-v5-2x-$v5_tag.service" in text
        assert tag in text
    for filename in (
        "metrics.jsonl",
        "summary.json",
        "best.pth.tar",
        "best_miou.pth.tar",
        "last.pth.tar",
        "pd_fa_sweep_best.pth.json",
        "pd_fa_sweep_best_miou.pth.json",
    ):
        assert filename in text
    assert '[[ "$v5_metrics_count" -eq 800 ]]' in text
    assert "TPDCLEANV5_2X_COMPLETE variant=$v5_variant seed=$v5_seed" in text
    assert "TPDCLEANV5_2X_ABORT|Traceback|out of memory" in text
    assert "resume" in text
    assert "TPDCLEANV5_2X_FINALIZER_WAIT" in text


def test_runner_delegates_marker_last_publication_to_validator() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '"$v5_completion_validator" publish' in text
    assert '"$v5_completion_validator" verify' in text
    assert "--staging-dir" in text
    assert "--reference-miou-root" in text
    assert "--require-complete" in text
    assert "completion_inputs.json" not in text
    assert "os.replace(source_json" not in text
    assert "lines.append(f" not in text
    assert "published and verified JSON, Markdown, manifest, and three-row marker" in text
    assert "train_tpd_clean_v5.py" not in text
    assert "evaluate_tpd_clean_v5_pd_fa.py" not in text
    assert "systemd-run" not in text


def test_launcher_preflight_precedes_any_persistent_start() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    preflight = text.index('if [[ "$v5_mode" == "--preflight" ]]')
    start = text.index('"$v5_systemd_run" --user')
    assert preflight < start
    assert "v5_verify_postprocess_sources" not in text
    assert "TPDCLEANV5_2X_POSTPROCESS_SOURCES_OK" in text
    assert "sctransnet-tpd-clean-v5-2x-screen800-finalizer.service" in text
    assert "/usr/bin/bash \"$v5_runner\"" in text
    assert "train_tpd_clean_v5.py" not in text


def test_postprocess_lock_exactly_binds_finalization_sources() -> None:
    payload = json.loads(POSTPROCESS_LOCK.read_text(encoding="utf-8"))

    assert (
        payload["schema"]
        == "sctransnet_tpd_clean_v5_2x_postprocess_source_lock_v1"
    )
    assert set(payload["source_sha256"]) == set(POSTPROCESS_SOURCE_SET)
    for relative, expected in payload["source_sha256"].items():
        path = REPO_ROOT / relative
        assert path.is_file()
        assert not path.is_symlink()
        assert _sha256(path) == expected
    assert payload["training_source_lock_sha256"] == _sha256(TRAINING_LOCK)
    assert payload["policy"] == {
        "separate_from_training_source_lock": True,
        "does_not_modify_frozen_training_results": True,
        "candidate_null_budget_points_forbidden": True,
        "unused_frozen_reference_null_points_disclosed": True,
        "required_gate_reference_null_points_forbidden": True,
        "automatic_mainline_replacement": False,
    }


def test_launcher_preflight_does_not_start_finalizer() -> None:
    completed = subprocess.run(
        [str(LAUNCHER), "--preflight"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "TPDCLEANV5_2X_FINALIZER_PREFLIGHT_OK" in completed.stdout
    assert "FINALIZER_UNIT_STARTED" not in completed.stdout


def test_runner_wait_limit_is_nonpublishing_in_isolated_root(
    tmp_path: Path,
) -> None:
    fake_systemctl = tmp_path / "systemctl"
    fake_systemctl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "'LoadState=loaded' "
        "'ActiveState=active' "
        "'SubState=running' "
        "'Result=success' "
        "'ExecMainCode=exited' "
        "'ExecMainStatus=0'\n",
        encoding="utf-8",
    )
    fake_systemctl.chmod(0o755)
    candidate_root = tmp_path / "candidate"
    environment = {
        **os.environ,
        "TPDCLEANV5_2X_FINALIZER_RESULT_ROOT": str(candidate_root),
        "TPDCLEANV5_2X_FINALIZER_SYSTEMCTL": str(fake_systemctl),
        "TPDCLEANV5_2X_FINALIZER_SLEEP": "/bin/true",
        "TPDCLEANV5_2X_FINALIZER_POLL_SECONDS": "0",
        "TPDCLEANV5_2X_FINALIZER_MAX_POLLS": "1",
    }

    completed = subprocess.run(
        [str(RUNNER)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 3, completed.stderr
    assert "TPDCLEANV5_2X_FINALIZER_WAIT poll=1" in completed.stdout
    assert not (
        candidate_root / "NUDT-SIRST/comparison/COMPLETE.sha256"
    ).exists()
    assert not (
        candidate_root
        / "NUDT-SIRST/comparison/tpd_clean_v5_screen800_comparison.json"
    ).exists()
    state = json.loads(
        (candidate_root / "launch/finalizer_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["state"] == "waiting"
