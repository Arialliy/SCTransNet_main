from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from experiments import (
    freeze_final_model_seed42_certification_completion_envfix_source_lock
    as envfix_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    REPO_ROOT
    / "experiments/"
    "run_final_model_seed42_certification_completion_envfix_v2.sh"
)


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_successor_lock_live_verifies_v1_without_rewriting_it(
    tmp_path: Path,
) -> None:
    upstream = envfix_lock.completion_source_lock.DEFAULT_OUTPUT
    upstream_before = upstream.read_bytes()
    output = tmp_path / "envfix_source_lock_v2.json"
    created, action = envfix_lock.freeze_source_lock(output)
    assert action == "created"
    assert envfix_lock.completion_source_lock.DEFAULT_OUTPUT.read_bytes() == (
        upstream_before
    )
    verified = envfix_lock.verify_source_lock(created)
    assert verified["status"] == "locked"
    assert verified["source_count"] == 3
    assert verified["upstream_completion_source_lock_v1"]["sha256"] == (
        envfix_lock.EXPECTED_COMPLETION_SOURCE_LOCK_SHA256
    )
    same, action = envfix_lock.freeze_source_lock(output)
    assert same == created
    assert action == "skipped_identical_locked"

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(envfix_lock.CompletionEnvfixSourceLockError):
        envfix_lock.verify_source_lock(output)


def test_wrapper_forces_environment_and_forwards_arguments_to_stub(
    tmp_path: Path,
) -> None:
    fake_repo = tmp_path / "repo"
    capture = tmp_path / "capture.json"
    python_capture = tmp_path / "python_args.txt"
    python_stub = _executable(
        tmp_path / "python-stub",
        """#!/usr/bin/env bash
set -euo pipefail
[[ "${CUBLAS_WORKSPACE_CONFIG:-}" == ":4096:8" ]]
printf '%s\\n' "$@" >"$ENVFIX_PYTHON_CAPTURE"
""",
    )
    _executable(
        fake_repo
        / "experiments/"
        "run_final_model_seed42_certification_completion.sh",
        """#!/usr/bin/env bash
set -euo pipefail
python3 - "$ENVFIX_CAPTURE" "$CUBLAS_WORKSPACE_CONFIG" "$@" <<'PY'
import json
import sys
path, value, *arguments = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"cublas": value, "arguments": arguments}, handle)
PY
""",
    )
    fake_lock = fake_repo / "experiments/envfix.json"
    fake_lock.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_ENVFIX_REPO_ROOT": str(fake_repo),
            "FINAL_MODEL_SEED42_ENVFIX_PYTHON": str(python_stub),
            "FINAL_MODEL_SEED42_ENVFIX_SOURCE_LOCK": str(fake_lock),
            "ENVFIX_CAPTURE": str(capture),
            "ENVFIX_PYTHON_CAPTURE": str(python_capture),
        }
    )
    subprocess.run(
        [str(WRAPPER), "--dry-run", "--poll-seconds", "17"],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
    observed = json.loads(capture.read_text(encoding="utf-8"))
    assert observed == {
        "cublas": ":4096:8",
        "arguments": ["--dry-run", "--poll-seconds", "17"],
    }
    verifier_arguments = python_capture.read_text(encoding="utf-8").splitlines()
    assert verifier_arguments[:3] == [
        "-m",
        "experiments.freeze_final_model_seed42_certification_completion_envfix_source_lock",
        "--verify",
    ]
    assert "--require-runtime-env" in verifier_arguments
    assert verifier_arguments[-1] == str(fake_lock)


def test_real_wrapper_dry_run_queries_no_gpu_and_writes_nothing() -> None:
    before = {
        path: path.stat().st_mtime_ns
        for path in (
            envfix_lock.DEFAULT_OUTPUT,
            envfix_lock.completion_source_lock.DEFAULT_OUTPUT,
        )
    }
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_ENVFIX_REPO_ROOT": str(REPO_ROOT),
            "FINAL_MODEL_SEED42_ENVFIX_PYTHON": sys.executable,
        }
    )
    completed = subprocess.run(
        [str(WRAPPER), "--dry-run"],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert '"runtime_environment_verified": true' in completed.stdout
    assert '"cublas_workspace_config": ":4096:8"' in completed.stdout
    assert '"gpu_command_launched": false' in completed.stdout
    assert '"gpu_queried": false' in completed.stdout
    assert {
        path: path.stat().st_mtime_ns
        for path in before
    } == before


def test_wrapper_and_builder_are_shell_and_python_parseable() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = Path(envfix_lock.__file__).resolve()
    ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
