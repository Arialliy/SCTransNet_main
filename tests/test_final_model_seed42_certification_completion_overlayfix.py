from __future__ import annotations

import ast
import contextlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import (
    final_model_seed42_certification_completion_overlayfix_attestation_v3
    as overlayfix_attestation,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_overlayfix_v3
    as overlayfix,
)
from experiments import (
    freeze_final_model_seed42_certification_completion_overlayfix_source_lock
    as overlayfix_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REDIRECTOR = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_python_overlayfix_v3.sh"
)
WRAPPER = (
    REPO_ROOT
    / "experiments/"
    "run_final_model_seed42_certification_completion_overlayfix_v3.sh"
)


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_builder_overlay_is_active_only_during_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"overlay_active": False, "validation_calls": 0}
    request = SimpleNamespace(variant="variant-d")
    inputs = object()
    sentinel = object()

    def validate(observed_request: object) -> None:
        assert observed_request is request
        assert state["overlay_active"] is False
        state["validation_calls"] += 1

    class Trainer:
        FORMAL_EPS = 1e-8

        @staticmethod
        def build_selected_model(
            variant: str,
            seed: int,
            *,
            eps: float,
        ) -> object:
            assert state["overlay_active"] is True
            assert variant == "variant-d"
            assert seed == frozen_posttraining.TRAJECTORY_SEED
            assert eps == Trainer.FORMAL_EPS
            return sentinel

    @contextlib.contextmanager
    def fake_overlay(observed_inputs: object):
        assert observed_inputs is inputs
        assert state["overlay_active"] is False
        state["overlay_active"] = True
        try:
            yield Trainer
        finally:
            state["overlay_active"] = False

    monkeypatch.setattr(
        frozen_posttraining,
        "_validate_request_shape",
        validate,
    )
    monkeypatch.setattr(
        frozen_posttraining.replay_core,
        "replay_trainer_overlay",
        fake_overlay,
    )

    with overlayfix.build_local_trajectory_model_builder(
        request,
        inputs,
    ) as build_model:
        assert state["overlay_active"] is False
        assert build_model(
            "variant-d",
            frozen_posttraining.TRAJECTORY_SEED,
        ) is sentinel
        assert state["overlay_active"] is False
        validate(request)

    assert state == {"overlay_active": False, "validation_calls": 2}


def test_builder_restores_overlay_when_model_construction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"overlay_active": False}
    request = SimpleNamespace(variant="variant-b")
    inputs = object()

    class Trainer:
        FORMAL_EPS = 1e-8

        @staticmethod
        def build_selected_model(*_args, **_kwargs):
            assert state["overlay_active"] is True
            raise RuntimeError("expected model-build failure")

    @contextlib.contextmanager
    def fake_overlay(_inputs: object):
        state["overlay_active"] = True
        try:
            yield Trainer
        finally:
            state["overlay_active"] = False

    monkeypatch.setattr(
        frozen_posttraining,
        "_validate_request_shape",
        lambda _request: None,
    )
    monkeypatch.setattr(
        frozen_posttraining.replay_core,
        "replay_trainer_overlay",
        fake_overlay,
    )

    with overlayfix.build_local_trajectory_model_builder(
        request,
        inputs,
    ) as build_model:
        with pytest.raises(RuntimeError, match="model-build failure"):
            build_model(
                "variant-b",
                frozen_posttraining.TRAJECTORY_SEED,
            )
    assert state["overlay_active"] is False


def test_successor_main_installs_and_restores_only_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = frozen_posttraining._trajectory_model_builder
    original_loader = (
        frozen_posttraining.evaluator._load_bound_shared_evaluator
    )
    observed: list[object] = []

    def fake_main(argv):
        observed.append(argv)
        assert (
            frozen_posttraining._trajectory_model_builder
            is overlayfix.build_local_trajectory_model_builder
        )
        assert (
            frozen_posttraining.evaluator._load_bound_shared_evaluator
            is overlayfix.seed42_source_bound_shared_evaluator
        )

    monkeypatch.setattr(frozen_posttraining, "main", fake_main)
    overlayfix.main(["--dry-run"])
    assert observed == [["--dry-run"]]
    assert frozen_posttraining._trajectory_model_builder is original
    assert (
        frozen_posttraining.evaluator._load_bound_shared_evaluator
        is original_loader
    )


def test_dynamic_sweep_binds_seed42_adapter_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic = SimpleNamespace(__file__="/tmp/wrong-engineering-adapter.py")
    state = {"write_called": False}

    def fake_loader(*args, **kwargs):
        assert args == ("request",)
        assert kwargs == {"assignment": "gpu2"}
        return dynamic, state

    monkeypatch.setattr(
        overlayfix,
        "_FROZEN_BOUND_SHARED_EVALUATOR_LOADER",
        fake_loader,
    )
    observed, observed_state = (
        overlayfix.seed42_source_bound_shared_evaluator(
            "request",
            assignment="gpu2",
        )
    )
    assert observed is dynamic
    assert observed_state is state
    assert Path(observed.__file__).resolve() == Path(
        frozen_posttraining.__file__
    ).resolve()
    binding = frozen_posttraining._evaluation_source_binding()
    adapter = binding["checkpoint_local_adapter"]
    assert Path(adapter["path"]).resolve() == Path(
        observed.__file__
    ).resolve()
    assert (
        frozen_posttraining._sha256_file(
            Path(observed.__file__),
            "dynamic evaluator",
        )
        == adapter["sha256"]
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (
            [
                "-m",
                (
                    "experiments."
                    "final_model_seed42_certification_replay_posttraining"
                ),
                "--execute",
                "--arm",
                "b",
            ],
            [
                "-m",
                (
                    "experiments."
                    "final_model_seed42_certification_replay_"
                    "posttraining_overlayfix_v3"
                ),
                "--execute",
                "--arm",
                "b",
            ],
        ),
        (
            ["-m", "experiments.other_module", "--verify"],
            ["-m", "experiments.other_module", "--verify"],
        ),
        (
            [
                "-m",
                (
                    "experiments."
                    "final_model_seed42_certification_replay_posttraining_extra"
                ),
            ],
            [
                "-m",
                (
                    "experiments."
                    "final_model_seed42_certification_replay_posttraining_extra"
                ),
            ],
        ),
        (["-c", "print('ok')"], ["-c", "print('ok')"]),
        ([], []),
    ),
)
def test_redirector_rewrites_only_the_exact_module(
    tmp_path: Path,
    arguments: list[str],
    expected: list[str],
) -> None:
    capture = tmp_path / "arguments.txt"
    fake_python = _executable(
        tmp_path / "real-python",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" >"$OVERLAYFIX_CAPTURE"
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FINAL_MODEL_SEED42_OVERLAYFIX_REAL_PYTHON": str(fake_python),
            "OVERLAYFIX_CAPTURE": str(capture),
        }
    )
    subprocess.run(
        [str(REDIRECTOR), *arguments],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
    observed = capture.read_text(encoding="utf-8").splitlines()
    if not arguments:
        observed = [] if observed == [""] else observed
    assert observed == expected


def test_completion_wrapper_forces_env_and_forwards_to_envfix(
    tmp_path: Path,
) -> None:
    fake_repo = tmp_path / "repo"
    verifier_capture = tmp_path / "verifier.txt"
    completion_capture = tmp_path / "completion.txt"
    fake_python = _executable(
        tmp_path / "real-python",
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'CALL\\n'
  printf '%s\\n' "$@"
} >>"$OVERLAYFIX_VERIFIER_CAPTURE"
""",
    )
    fake_redirector = _executable(
        fake_repo
        / "experiments/"
        "final_model_seed42_certification_python_overlayfix_v3.sh",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _executable(
        fake_repo
        / "experiments/"
        "run_final_model_seed42_certification_completion_envfix_v2.sh",
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf '%s\\n' "$CUBLAS_WORKSPACE_CONFIG"
  printf '%s\\n' "$FINAL_MODEL_SEED42_ENVFIX_PYTHON"
  printf '%s\\n' "$@"
} >"$OVERLAYFIX_COMPLETION_CAPTURE"
""",
    )
    fake_lock = fake_repo / "experiments/overlayfix.json"
    fake_lock.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_OVERLAYFIX_REPO_ROOT": str(fake_repo),
            "FINAL_MODEL_SEED42_OVERLAYFIX_REAL_PYTHON": str(fake_python),
            "FINAL_MODEL_SEED42_OVERLAYFIX_SOURCE_LOCK": str(fake_lock),
            "OVERLAYFIX_VERIFIER_CAPTURE": str(verifier_capture),
            "OVERLAYFIX_COMPLETION_CAPTURE": str(completion_capture),
        }
    )
    subprocess.run(
        [str(WRAPPER), "--dry-run", "--poll-seconds", "19"],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
    invocations: list[list[str]] = []
    for line in verifier_capture.read_text(encoding="utf-8").splitlines():
        if line == "CALL":
            invocations.append([])
        else:
            invocations[-1].append(line)
    assert invocations[0][:3] == [
        "-m",
        (
            "experiments."
            "freeze_final_model_seed42_certification_completion_"
            "overlayfix_source_lock"
        ),
        "--verify",
    ]
    assert "--require-runtime-env" in invocations[0]
    assert invocations[0][-1] == str(fake_lock)
    assert invocations[1][:3] == [
        "-m",
        (
            "experiments."
            "final_model_seed42_certification_completion_"
            "overlayfix_attestation_v3"
        ),
        "--dry-run",
    ]
    assert invocations[1][-1].endswith(
        "final_model_seed42_certification_overlayfix_attestation_v3.json"
    )
    forwarded = completion_capture.read_text(encoding="utf-8").splitlines()
    assert forwarded == [
        ":4096:8",
        str(fake_redirector),
        "--dry-run",
        "--poll-seconds",
        "19",
    ]


def test_successor_lock_live_verifies_upstreams_without_rewriting(
    tmp_path: Path,
) -> None:
    upstream = overlayfix_lock.envfix_source_lock.DEFAULT_OUTPUT
    upstream_before = upstream.read_bytes()
    output = tmp_path / "overlayfix_source_lock_v3.json"
    created, action = overlayfix_lock.freeze_source_lock(output)
    assert action == "created"
    assert upstream.read_bytes() == upstream_before
    verified = overlayfix_lock.verify_source_lock(created)
    assert verified["status"] == "locked"
    assert verified["source_count"] == 6
    assert (
        verified["upstream_completion_envfix_source_lock_v2"]["sha256"]
        == overlayfix_lock.EXPECTED_ENVFIX_SOURCE_LOCK_SHA256
    )
    same, action = overlayfix_lock.freeze_source_lock(output)
    assert same == created
    assert action == "skipped_identical_locked"

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        overlayfix_lock.CompletionOverlayfixSourceLockError
    ):
        overlayfix_lock.verify_source_lock(output)


def test_successor_attestation_is_write_once_and_keeps_claims_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_lock = tmp_path / "overlayfix-source-lock.json"
    source_payload = {
        "schema": overlayfix_lock.SCHEMA,
        "source_count": 6,
        "upstream_completion_envfix_source_lock_v2": {
            "path": (
                "experiments/"
                "final_model_seed42_certification_completion_"
                "envfix_source_lock_v2.json"
            ),
            "sha256": overlayfix_lock.EXPECTED_ENVFIX_SOURCE_LOCK_SHA256,
            "schema": overlayfix_lock.envfix_source_lock.SCHEMA,
        },
    }
    source_lock.write_bytes(
        overlayfix_attestation.canonical_json_bytes(source_payload)
    )
    base = tmp_path / "base-attestation.json"
    base_payload = {
        "schema": frozen_posttraining.SCHEMA,
        "status": "complete",
        "decision": "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
        "model_contract": {
            "mainline": (
                "SCTransNet+TPD8+five-node-NER4+QFG2-CROA"
            ),
            "mainline_changed": False,
            "innovation_changed": False,
            "default_threshold": 0.5,
            "seed42_deployment_weight_changed": False,
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
        "official_test_accessed": False,
    }
    base.write_bytes(
        overlayfix_attestation.canonical_json_bytes(base_payload)
    )
    monkeypatch.setattr(
        overlayfix_attestation.overlayfix_source_lock,
        "verify_source_lock",
        lambda path: source_payload,
    )
    monkeypatch.setattr(
        overlayfix_attestation.completion,
        "verify_attestation",
        lambda **_kwargs: {"status": "verified_complete"},
    )
    monkeypatch.setattr(
        overlayfix_attestation.overlayfix_source_lock,
        "REPO_ROOT",
        REPO_ROOT,
    )
    output = tmp_path / "successor-attestation.json"
    first = overlayfix_attestation.finalize_attestation(
        source_lock_path=source_lock,
        base_attestation_path=base,
        output=output,
        require_runtime_env=False,
    )
    assert first["attestation_action"] == "created"
    second = overlayfix_attestation.finalize_attestation(
        source_lock_path=source_lock,
        base_attestation_path=base,
        output=output,
        require_runtime_env=False,
    )
    assert second["attestation_action"] == "skipped_identical_complete"
    verified = overlayfix_attestation.verify_attestation(
        source_lock_path=source_lock,
        base_attestation_path=base,
        output=output,
    )
    assert verified["status"] == "verified_complete"
    payload = overlayfix_attestation._canonical_object(
        output,
        "test successor attestation",
    )
    assert payload["model_contract"]["mainline_changed"] is False
    assert payload["model_contract"]["innovation_changed"] is False
    assert payload["paper_core_established"] is False
    assert payload["stability_claim_supported"] is False
    assert (
        payload["claim_boundary"]["multiseed_replication_supported"]
        is False
    )


def test_real_wrapper_dry_run_queries_no_gpu_and_writes_nothing() -> None:
    tracked = (
        overlayfix_lock.DEFAULT_OUTPUT,
        overlayfix_lock.envfix_source_lock.DEFAULT_OUTPUT,
        (
            overlayfix_lock.envfix_source_lock.completion_source_lock
            .DEFAULT_OUTPUT
        ),
    )
    before = {path: path.stat().st_mtime_ns for path in tracked}
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_OVERLAYFIX_REPO_ROOT": str(REPO_ROOT),
            "FINAL_MODEL_SEED42_OVERLAYFIX_REAL_PYTHON": sys.executable,
        }
    )
    completed = subprocess.run(
        [str(WRAPPER), "--dry-run"],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.count('"runtime_environment_verified": true') == 2
    assert '"cublas_workspace_config": ":4096:8"' in completed.stdout
    assert '"gpu_command_launched": false' in completed.stdout
    assert '"gpu_queried": false' in completed.stdout
    assert {path: path.stat().st_mtime_ns for path in tracked} == before


def test_new_sources_are_parseable() -> None:
    subprocess.run(["bash", "-n", str(REDIRECTOR)], check=True)
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    for source in (
        Path(overlayfix.__file__).resolve(),
        Path(overlayfix_attestation.__file__).resolve(),
        Path(overlayfix_lock.__file__).resolve(),
    ):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
