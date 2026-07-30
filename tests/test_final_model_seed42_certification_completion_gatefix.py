from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import (
    final_model_seed42_certification_completion as frozen_completion,
)
from experiments import (
    final_model_seed42_certification_completion_gatefix_attestation_v5
    as gatefix_attestation,
)
from experiments import (
    final_model_seed42_certification_completion_gatefix_v5
    as completion_gatefix,
)
from experiments import (
    final_model_seed42_certification_metricsfix_attestation_gatefix_v5
    as metricsfix_attestation_gatefix,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_gatefix_v5
    as gatefix,
)
from experiments import (
    freeze_final_model_seed42_certification_completion_gatefix_source_lock
    as gatefix_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REDIRECTOR = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_python_gatefix_v5.sh"
)
WRAPPER = (
    REPO_ROOT
    / "experiments/"
    "run_final_model_seed42_certification_completion_gatefix_v5.sh"
)
RESULT_ROOT = (
    REPO_ROOT
    / "experiments/results/final_model_seed42_certification_replay_v1"
)
MANIFEST = (
    RESULT_ROOT / "seed42_replay_checkpoint_local_pd_fa_manifest_v1.json"
)


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_manifest_validator_enters_and_restores_seed42_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_count = frozen_posttraining.evaluator.EXPECTED_SWEEP_COUNT
    expected_seeds = (
        frozen_posttraining.evaluator.seeds.ENGINEERING_TRAJECTORY_SEEDS
    )
    assert expected_count == 8
    assert expected_seeds == (3407, 426780603)
    observed: list[tuple[int, tuple[int, ...]]] = []

    def frozen_validator(_manifest):
        state = (
            frozen_posttraining.evaluator.EXPECTED_SWEEP_COUNT,
            frozen_posttraining.evaluator.seeds.ENGINEERING_TRAJECTORY_SEEDS,
        )
        observed.append(state)
        return ({"result": {}}, {"group": {}})

    monkeypatch.setattr(
        gatefix,
        "_FROZEN_REPLAY_MANIFEST_VALIDATOR",
        frozen_validator,
    )
    result = gatefix.seed42_overlay_bound_manifest_validator({})
    assert result == ({"result": {}}, {"group": {}})
    assert observed == [(4, (42,))]
    assert frozen_posttraining.evaluator.EXPECTED_SWEEP_COUNT == 8
    assert (
        frozen_posttraining.evaluator.seeds.ENGINEERING_TRAJECTORY_SEEDS
        == (3407, 426780603)
    )


def test_live_four_result_manifest_passes_strict_validator_only_in_overlay(
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["result_count"] == 4
    assert manifest["expected_result_count"] == 4
    assert manifest["paired_checkpoint_group_count"] == 2
    with pytest.raises(
        frozen_posttraining.paired_core.EngineeringPairedScreenError,
        match="manifest result count differs",
    ):
        gatefix._FROZEN_REPLAY_MANIFEST_VALIDATOR(manifest)
    results, groups = gatefix.seed42_overlay_bound_manifest_validator(
        manifest
    )
    assert len(results) == 4
    assert len(groups) == 2
    assert set(results) == {
        (42, arm, role)
        for arm in ("b", "d")
        for role in (
            frozen_posttraining.PRIMARY_SELECTION_ROLE,
            frozen_posttraining.SECONDARY_SELECTION_ROLE,
        )
    }
    assert frozen_posttraining.evaluator.EXPECTED_SWEEP_COUNT == 8


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "wrong-schema"),
        ("scope", "wrong-scope"),
        ("result_count", 8),
        ("expected_result_count", 8),
        ("paired_checkpoint_group_count", 4),
        ("fixed_threshold", 0.6),
    ),
)
def test_gatefix_does_not_relax_seed42_manifest_contract(
    field: str,
    value: object,
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest[field] = value
    with pytest.raises(Exception, match="differs"):
        gatefix.seed42_overlay_bound_manifest_validator(manifest)


def test_posttraining_successor_installs_and_restores_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = frozen_posttraining._validate_replay_manifest_for_paired
    observed: list[object] = []

    def fake_v4_main(argv):
        observed.append(argv)
        assert (
            frozen_posttraining._validate_replay_manifest_for_paired
            is gatefix.seed42_overlay_bound_manifest_validator
        )

    monkeypatch.setattr(gatefix.metricsfix_v4, "main", fake_v4_main)
    gatefix.main(["--gate"])
    assert observed == [["--gate"]]
    assert frozen_posttraining._validate_replay_manifest_for_paired is original


def test_successor_preserves_v4_metric_projection_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_manifest = (
        frozen_posttraining._validate_replay_manifest_for_paired
    )
    original_result_validator = (
        frozen_posttraining.evaluator.validate_checkpoint_local_result
    )
    observed: list[object] = []

    def fake_v3_main(argv):
        observed.append(argv)
        assert (
            frozen_posttraining._validate_replay_manifest_for_paired
            is gatefix.seed42_overlay_bound_manifest_validator
        )
        assert (
            frozen_posttraining.evaluator.validate_checkpoint_local_result
            is gatefix.metricsfix_v4.projected_checkpoint_metrics_validator
        )

    monkeypatch.setattr(
        gatefix.metricsfix_v4.overlayfix_v3,
        "main",
        fake_v3_main,
    )
    gatefix.main(["--gate"])
    assert observed == [["--gate"]]
    assert (
        frozen_posttraining._validate_replay_manifest_for_paired
        is original_manifest
    )
    assert (
        frozen_posttraining.evaluator.validate_checkpoint_local_result
        is original_result_validator
    )


def test_live_gate_path_uses_strict_manifest_and_two_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directories = {
        "b": (
            RESULT_ROOT
            / "NUDT-SIRST/tss_on/"
            "seed_42_seed42_certification_replay_b_formal800"
        ),
        "d": (
            RESULT_ROOT
            / "NUDT-SIRST/tss_qfg/"
            "seed_42_seed42_certification_replay_d_formal800"
        ),
    }
    requests = tuple(
        SimpleNamespace(arm=arm, checkpoint_filename=filename)
        for arm in ("b", "d")
        for filename in ("best_miou.pth.tar", "best.pth.tar")
    )

    def completed(request):
        path = (
            run_directories[request.arm]
            / (
                "pd_fa_sweep_"
                f"{request.checkpoint_filename.removesuffix('.tar')}.json"
            )
        )
        return json.loads(path.read_text(encoding="utf-8")), path

    monkeypatch.setattr(
        frozen_posttraining.evaluator,
        "assemble_evaluation_plan",
        lambda _requests: {"complete": True},
    )
    monkeypatch.setattr(
        frozen_posttraining.evaluator,
        "load_completed_result",
        completed,
    )
    with frozen_posttraining._temporary_attributes(
        frozen_posttraining,
        {
            "_validate_replay_manifest_for_paired": (
                gatefix.seed42_overlay_bound_manifest_validator
            ),
        },
    ):
        payload = frozen_posttraining.adjudicate_gate(
            requests,
            manifest_path=MANIFEST,
            paired_path=(
                REPO_ROOT
                / "analysis/results/"
                "final_model_seed42_replay_paired_screen_v1.json"
            ),
        )
    assert payload["status"] == "complete"
    assert payload["decision"] == (
        "SEED42_REPLAY_ENGINEERING_COMPLETE_MIOU_ROUTE_NOT_MET"
    )
    assert payload["trajectory_seeds"] == [42]
    assert len(payload["fixed_threshold_and_budget_comparisons"]) == 2
    assert payload["claim_boundary"]["paper_core_established"] is False
    assert payload["claim_boundary"]["stability_claim_supported"] is False
    assert frozen_posttraining.evaluator.EXPECTED_SWEEP_COUNT == 8


def test_completion_successor_covers_verify_and_attestation_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = frozen_posttraining._validate_replay_manifest_for_paired
    observed: list[object] = []

    def fake_completion_main(argv):
        observed.append(argv)
        assert (
            frozen_completion.posttraining
            ._validate_replay_manifest_for_paired
            is gatefix.seed42_overlay_bound_manifest_validator
        )

    monkeypatch.setattr(
        completion_gatefix.completion,
        "main",
        fake_completion_main,
    )
    for arguments in (
        ["--verify-posttraining"],
        ["--finalize-attestation"],
        ["--verify-attestation"],
    ):
        completion_gatefix.main(arguments)
    assert observed == [
        ["--verify-posttraining"],
        ["--finalize-attestation"],
        ["--verify-attestation"],
    ]
    assert frozen_posttraining._validate_replay_manifest_for_paired is original


def test_metricsfix_attestation_successor_installs_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = frozen_posttraining._validate_replay_manifest_for_paired
    observed: list[object] = []

    def fake_main(argv):
        observed.append(argv)
        assert (
            frozen_posttraining._validate_replay_manifest_for_paired
            is gatefix.seed42_overlay_bound_manifest_validator
        )

    monkeypatch.setattr(
        metricsfix_attestation_gatefix.metricsfix_attestation,
        "main",
        fake_main,
    )
    metricsfix_attestation_gatefix.main(["--verify"])
    assert observed == [["--verify"]]
    assert frozen_posttraining._validate_replay_manifest_for_paired is original


@pytest.mark.parametrize(
    ("arguments", "expected_module"),
    (
        (
            [
                "-m",
                (
                    "experiments."
                    "final_model_seed42_certification_replay_posttraining"
                ),
                "--gate",
            ],
            (
                "experiments."
                "final_model_seed42_certification_replay_"
                "posttraining_gatefix_v5"
            ),
        ),
        (
            [
                "-m",
                "experiments.final_model_seed42_certification_completion",
                "--verify-posttraining",
            ],
            (
                "experiments."
                "final_model_seed42_certification_completion_gatefix_v5"
            ),
        ),
        (
            ["-m", "experiments.other", "--verify"],
            "experiments.other",
        ),
        (
            [
                "-m",
                (
                    "experiments."
                    "final_model_seed42_certification_completion_extra"
                ),
            ],
            (
                "experiments."
                "final_model_seed42_certification_completion_extra"
            ),
        ),
    ),
)
def test_redirector_rewrites_only_two_exact_modules(
    tmp_path: Path,
    arguments: list[str],
    expected_module: str,
) -> None:
    capture = tmp_path / "args.txt"
    fake_python = _executable(
        tmp_path / "real-python",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$GATEFIX_CAPTURE"
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FINAL_MODEL_SEED42_GATEFIX_REAL_PYTHON": str(fake_python),
            "GATEFIX_CAPTURE": str(capture),
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
    assert observed[0] == "-m"
    assert observed[1] == expected_module
    assert observed[2:] == arguments[2:]


def test_completion_wrapper_forces_env_and_runs_both_attestations(
    tmp_path: Path,
) -> None:
    fake_repo = tmp_path / "repo"
    invocation_capture = tmp_path / "invocations.txt"
    completion_capture = tmp_path / "completion.txt"
    fake_python = _executable(
        tmp_path / "real-python",
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'CALL\n'
  printf '%s\n' "$@"
} >>"$GATEFIX_INVOCATION_CAPTURE"
""",
    )
    fake_redirector = _executable(
        fake_repo
        / "experiments/"
        "final_model_seed42_certification_python_gatefix_v5.sh",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _executable(
        fake_repo
        / "experiments/"
        "run_final_model_seed42_certification_completion_envfix_v2.sh",
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf '%s\n' "$CUBLAS_WORKSPACE_CONFIG"
  printf '%s\n' "$FINAL_MODEL_SEED42_ENVFIX_PYTHON"
  printf '%s\n' "$@"
} >"$GATEFIX_COMPLETION_CAPTURE"
""",
    )
    gate_lock = fake_repo / "experiments/gatefix.json"
    gate_lock.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_GATEFIX_REPO_ROOT": str(fake_repo),
            "FINAL_MODEL_SEED42_GATEFIX_REAL_PYTHON": str(fake_python),
            "FINAL_MODEL_SEED42_GATEFIX_SOURCE_LOCK": str(gate_lock),
            "GATEFIX_INVOCATION_CAPTURE": str(invocation_capture),
            "GATEFIX_COMPLETION_CAPTURE": str(completion_capture),
        }
    )
    subprocess.run(
        [str(WRAPPER), "--dry-run", "--poll-seconds", "23"],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
    invocations: list[list[str]] = []
    for line in invocation_capture.read_text(
        encoding="utf-8"
    ).splitlines():
        if line == "CALL":
            invocations.append([])
        else:
            invocations[-1].append(line)
    assert invocations[0][:3] == [
        "-m",
        (
            "experiments."
            "freeze_final_model_seed42_certification_completion_"
            "gatefix_source_lock"
        ),
        "--verify",
    ]
    assert "--require-runtime-env" in invocations[0]
    assert invocations[1][1] == (
        "experiments."
        "final_model_seed42_certification_metricsfix_"
        "attestation_gatefix_v5"
    )
    assert invocations[1][2] == "--dry-run"
    assert invocations[2][1] == (
        "experiments."
        "final_model_seed42_certification_completion_"
        "gatefix_attestation_v5"
    )
    assert invocations[2][2] == "--dry-run"
    assert completion_capture.read_text(
        encoding="utf-8"
    ).splitlines() == [
        ":4096:8",
        str(fake_redirector),
        "--dry-run",
        "--poll-seconds",
        "23",
    ]


def test_successor_lock_live_verifies_v4_without_rewriting(
    tmp_path: Path,
) -> None:
    upstream = gatefix_lock.metricsfix_source_lock.DEFAULT_OUTPUT
    upstream_before = upstream.read_bytes()
    output = tmp_path / "gatefix_source_lock_v5.json"
    created, action = gatefix_lock.freeze_source_lock(output)
    assert action == "created"
    assert upstream.read_bytes() == upstream_before
    verified = gatefix_lock.verify_source_lock(created)
    assert verified["status"] == "locked"
    assert verified["source_count"] == 8
    assert (
        verified["upstream_completion_metricsfix_source_lock_v4"][
            "sha256"
        ]
        == gatefix_lock.EXPECTED_METRICSFIX_SOURCE_LOCK_SHA256
    )
    same, action = gatefix_lock.freeze_source_lock(output)
    assert same == created
    assert action == "skipped_identical_locked"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(gatefix_lock.CompletionGatefixSourceLockError):
        gatefix_lock.verify_source_lock(output)


def test_successor_attestation_write_once_and_claim_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_lock = tmp_path / "gatefix-source-lock.json"
    upstream_lock_sha = "9" * 64
    source_payload = {
        "schema": gatefix_lock.SCHEMA,
        "source_count": 8,
        "upstream_completion_metricsfix_source_lock_v4": {
            "path": (
                "experiments/"
                "final_model_seed42_certification_completion_"
                "metricsfix_source_lock_v4.json"
            ),
            "sha256": upstream_lock_sha,
            "schema": gatefix_lock.metricsfix_source_lock.SCHEMA,
        },
    }
    source_lock.write_bytes(
        gatefix_attestation.canonical_json_bytes(source_payload)
    )
    upstream_attestation = tmp_path / "metricsfix-attestation.json"
    upstream_payload = {
        "status": "complete",
        "decision": "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
        "metricsfix_source_lock_v4": {"sha256": upstream_lock_sha},
        "model_contract": {
            "mainline": "SCTransNet+TPD8+five-node-NER4+QFG2-CROA",
            "mainline_changed": False,
            "innovation_changed": False,
            "default_threshold": 0.5,
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
        "official_test_accessed": False,
    }
    upstream_attestation.write_bytes(
        gatefix_attestation.canonical_json_bytes(upstream_payload)
    )
    monkeypatch.setattr(
        gatefix_attestation.gatefix_source_lock,
        "verify_source_lock",
        lambda _path: source_payload,
    )
    monkeypatch.setattr(
        gatefix_attestation.metricsfix_attestation,
        "verify_attestation",
        lambda **_kwargs: {"status": "verified_complete"},
    )
    output = tmp_path / "attestation.json"
    first = gatefix_attestation.finalize_attestation(
        source_lock_path=source_lock,
        metricsfix_attestation_path=upstream_attestation,
        output=output,
        require_runtime_env=False,
    )
    assert first["attestation_action"] == "created"
    second = gatefix_attestation.finalize_attestation(
        source_lock_path=source_lock,
        metricsfix_attestation_path=upstream_attestation,
        output=output,
        require_runtime_env=False,
    )
    assert second["attestation_action"] == "skipped_identical_complete"
    assert gatefix_attestation.verify_attestation(
        source_lock_path=source_lock,
        metricsfix_attestation_path=upstream_attestation,
        output=output,
    )["status"] == "verified_complete"
    payload = gatefix_attestation._canonical_object(
        output,
        "test Gate-fix attestation",
    )
    assert payload["paper_core_established"] is False
    assert payload["stability_claim_supported"] is False
    assert (
        payload["claim_boundary"]["multiseed_replication_supported"]
        is False
    )


def test_real_wrapper_dry_run_queries_no_gpu_and_writes_nothing() -> None:
    tracked = (
        gatefix_lock.DEFAULT_OUTPUT,
        gatefix_lock.metricsfix_source_lock.DEFAULT_OUTPUT,
    )
    before = {path: path.stat().st_mtime_ns for path in tracked}
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_GATEFIX_REPO_ROOT": str(REPO_ROOT),
            "FINAL_MODEL_SEED42_GATEFIX_REAL_PYTHON": sys.executable,
        }
    )
    completed = subprocess.run(
        [str(WRAPPER), "--dry-run"],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.count(
        '"runtime_environment_verified": true'
    ) == 2
    assert '"cublas_workspace_config": ":4096:8"' in completed.stdout
    assert '"gpu_command_launched": false' in completed.stdout
    assert '"gpu_queried": false' in completed.stdout
    assert {path: path.stat().st_mtime_ns for path in tracked} == before


def test_new_sources_are_parseable() -> None:
    assert os.access(REDIRECTOR, os.X_OK)
    assert os.access(WRAPPER, os.X_OK)
    subprocess.run(["bash", "-n", str(REDIRECTOR)], check=True)
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    for source in (
        Path(gatefix.__file__).resolve(),
        Path(completion_gatefix.__file__).resolve(),
        Path(metricsfix_attestation_gatefix.__file__).resolve(),
        Path(gatefix_attestation.__file__).resolve(),
        Path(gatefix_lock.__file__).resolve(),
    ):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
