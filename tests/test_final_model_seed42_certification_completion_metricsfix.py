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
import torch

from experiments import (
    final_model_seed42_certification_completion_metricsfix_attestation_v4
    as metricsfix_attestation,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_metricsfix_v4
    as metricsfix,
)
from experiments import (
    freeze_final_model_seed42_certification_completion_metricsfix_source_lock
    as metricsfix_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REDIRECTOR = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_python_metricsfix_v4.sh"
)
WRAPPER = (
    REPO_ROOT
    / "experiments/"
    "run_final_model_seed42_certification_completion_metricsfix_v4.sh"
)
RESULT_ROOT = (
    REPO_ROOT
    / "experiments/results/final_model_seed42_certification_replay_v1"
)
FULL_METRICS = {
    "fa": 4.1301985432330825e-06,
    "false_objects_per_image": 0.03759398496240601,
    "matched_target_count": 188,
    "matched_tiny_target_count": 39,
    "miou": 0.9368702770780857,
    "niou": 0.9363119882567429,
    "pd": 0.9947089947089947,
    "pixel_f1": 0.9674063236608957,
    "pixel_precision": 0.9739770867430442,
    "pixel_recall": 0.9609236234458259,
    "predicted_object_count": 193,
    "target_count": 189,
    "tiny_pd": 1.0,
    "tiny_target_count": 39,
    "unmatched_predicted_object_count": 5,
    "val_loss": 0.00021190011287395,
    "valid_pixel_count": 8716288,
}


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _request(
    *,
    projected: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=Path("/tmp/best_miou.pth.tar"),
        checkpoint_filename="best_miou.pth.tar",
        checkpoint_epoch=3,
        run_directory=Path("/tmp/run"),
        checkpoint_validation_metrics=(
            projected
            if projected is not None
            else frozen_posttraining.evaluator._validate_checkpoint_metrics(
                FULL_METRICS,
                label="test full metrics",
            )
        ),
    )


def test_full_metric_sources_require_exact_checkpoint_summary_and_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    events = [{} for _ in range(800)]
    events[2] = copy.deepcopy(FULL_METRICS)
    monkeypatch.setattr(
        frozen_posttraining.evaluator,
        "_verify_request_files_unchanged",
        lambda _request: None,
    )
    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core.torch,
        "load",
        lambda *_args, **_kwargs: {
            "validation_metrics": copy.deepcopy(FULL_METRICS)
        },
    )
    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core,
        "load_json_object",
        lambda _path: {
            "best_miou_validation_metrics": copy.deepcopy(FULL_METRICS),
            "best_validation_metrics": copy.deepcopy(FULL_METRICS),
            "best_pd_validation_metrics": copy.deepcopy(FULL_METRICS),
        },
    )
    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core,
        "load_complete_metrics",
        lambda *_args: events,
    )
    assert metricsfix._validate_full_metric_sources(
        FULL_METRICS,
        request,
    ) == FULL_METRICS

    bad_checkpoint = copy.deepcopy(FULL_METRICS)
    bad_checkpoint["niou"] += 1e-6
    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core.torch,
        "load",
        lambda *_args, **_kwargs: {
            "validation_metrics": bad_checkpoint
        },
    )
    with pytest.raises(
        frozen_posttraining.evaluator.EngineeringEvaluationError
    ):
        metricsfix._validate_full_metric_sources(FULL_METRICS, request)

    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core.torch,
        "load",
        lambda *_args, **_kwargs: {
            "validation_metrics": copy.deepcopy(FULL_METRICS)
        },
    )
    bad_summary = copy.deepcopy(FULL_METRICS)
    bad_summary["pixel_f1"] += 1e-6
    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core,
        "load_json_object",
        lambda _path: {
            "best_miou_validation_metrics": bad_summary,
        },
    )
    with pytest.raises(
        frozen_posttraining.evaluator.EngineeringEvaluationError
    ):
        metricsfix._validate_full_metric_sources(FULL_METRICS, request)

    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core,
        "load_json_object",
        lambda _path: {
            "best_miou_validation_metrics": copy.deepcopy(FULL_METRICS),
        },
    )
    bad_events = copy.deepcopy(events)
    bad_events[2]["val_loss"] += 1e-6
    monkeypatch.setattr(
        frozen_posttraining.evaluator.sweep_core,
        "load_complete_metrics",
        lambda *_args: bad_events,
    )
    with pytest.raises(
        frozen_posttraining.evaluator.EngineeringEvaluationError
    ):
        metricsfix._validate_full_metric_sources(FULL_METRICS, request)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "bool", "nan"),
)
def test_full_metric_source_shape_and_values_are_strict(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    observed = copy.deepcopy(FULL_METRICS)
    if mutation == "missing":
        observed.pop("niou")
    elif mutation == "extra":
        observed["unexpected"] = 1.0
    elif mutation == "bool":
        observed["pixel_f1"] = True
    else:
        observed["pixel_f1"] = float("nan")
    monkeypatch.setattr(
        frozen_posttraining.evaluator,
        "_verify_request_files_unchanged",
        lambda _request: None,
    )
    with pytest.raises(
        frozen_posttraining.evaluator.EngineeringEvaluationError
    ):
        metricsfix._validate_full_metric_sources(
            observed,
            _request(),
        )


def test_validator_revalidates_raw_audit_then_persists_projected_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    fixed = {**FULL_METRICS, "threshold": 0.5}
    raw_audit = metricsfix._FROZEN_FIXED_THRESHOLD_AUDIT(
        fixed,
        FULL_METRICS,
    )
    payload = {
        "checkpoint_validation_metrics": copy.deepcopy(FULL_METRICS),
        "fixed_threshold_0_5": fixed,
        "fixed_threshold_0_5_checkpoint_audit": raw_audit,
        "points": [{"threshold": 0.5, "pd": FULL_METRICS["pd"]}],
        "best_points_under_fa_budget": {"1e-06": None},
    }
    captured: dict[str, object] = {}

    def frozen_validator(
        normalized,
        observed_request,
        *,
        execution_context=None,
    ):
        captured["payload"] = normalized
        captured["request"] = observed_request
        captured["execution_context"] = execution_context
        return normalized

    monkeypatch.setattr(
        metricsfix,
        "_validate_full_metric_sources",
        lambda metrics, observed_request: copy.deepcopy(dict(metrics)),
    )
    monkeypatch.setattr(
        metricsfix,
        "_FROZEN_CHECKPOINT_LOCAL_VALIDATOR",
        frozen_validator,
    )
    result = metricsfix.projected_checkpoint_metrics_validator(
        payload,
        request,
        execution_context={"complete": True},
    )
    assert result is captured["payload"]
    assert captured["request"] is request
    assert captured["execution_context"] == {"complete": True}
    assert set(result["checkpoint_validation_metrics"]) == set(
        frozen_posttraining.summary_core.METRICS
    )
    assert result["checkpoint_validation_metrics"] == (
        request.checkpoint_validation_metrics
    )
    assert result["fixed_threshold_0_5_checkpoint_audit"] == (
        metricsfix._FROZEN_FIXED_THRESHOLD_AUDIT(
            fixed,
            request.checkpoint_validation_metrics,
        )
    )
    assert result["fixed_threshold_0_5_checkpoint_audit"] == (
        frozen_posttraining.evaluator.point_validator
        ._fixed_threshold_checkpoint_audit(
            fixed,
            request.checkpoint_validation_metrics,
        )
    )
    assert result["points"] == payload["points"]
    assert (
        result["best_points_under_fa_budget"]
        == payload["best_points_under_fa_budget"]
    )
    assert payload["checkpoint_validation_metrics"] == FULL_METRICS
    assert (
        payload["fixed_threshold_0_5_checkpoint_audit"]
        == raw_audit
    )


def test_projected_payload_passes_the_real_frozen_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import (
        test_evaluate_final_model_engineering_replication_pd_fa
        as engineering_fixture,
    )

    request = engineering_fixture.make_request(
        output_root=tmp_path / "results",
        trajectory_seed=(
            engineering_fixture.seeds.ENGINEERING_TRAJECTORY_SEEDS[0]
        ),
        arm=engineering_fixture.core.ARM_B,
        checkpoint_filename="best_miou.pth.tar",
    )
    payload = engineering_fixture.shared_result(request)
    full_metrics = {
        **request.checkpoint_validation_metrics,
        "niou": 0.93,
        "pixel_f1": 0.96,
        "pixel_precision": 0.97,
        "pixel_recall": 0.95,
        "predicted_object_count": (
            payload["fixed_threshold_0_5"]["predicted_object_count"]
        ),
        "val_loss": 0.001,
    }
    payload["checkpoint_validation_metrics"] = copy.deepcopy(full_metrics)
    payload["fixed_threshold_0_5_checkpoint_audit"] = (
        metricsfix._FROZEN_FIXED_THRESHOLD_AUDIT(
            payload["fixed_threshold_0_5"],
            full_metrics,
        )
    )
    monkeypatch.setattr(
        metricsfix,
        "_validate_full_metric_sources",
        lambda metrics, observed_request: copy.deepcopy(dict(metrics)),
    )
    finalized = metricsfix.projected_checkpoint_metrics_validator(
        payload,
        request,
    )
    assert finalized["execution_complete"] is False
    assert finalized["checkpoint_validation_metrics"] == (
        request.checkpoint_validation_metrics
    )
    frozen_posttraining.evaluator.validate_finalized_result(
        finalized,
        request,
    )


def test_validator_rejects_raw_audit_or_registered_value_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    fixed = {**FULL_METRICS, "threshold": 0.5}
    audit = metricsfix._FROZEN_FIXED_THRESHOLD_AUDIT(
        fixed,
        FULL_METRICS,
    )
    monkeypatch.setattr(
        metricsfix,
        "_validate_full_metric_sources",
        lambda metrics, observed_request: copy.deepcopy(dict(metrics)),
    )
    monkeypatch.setattr(
        metricsfix,
        "_FROZEN_CHECKPOINT_LOCAL_VALIDATOR",
        lambda *_args, **_kwargs: pytest.fail(
            "frozen validator must not be reached"
        ),
    )
    bad_audit = copy.deepcopy(audit)
    bad_audit["max_abs_non_strict_numeric_delta"] = 1.0
    with pytest.raises(
        frozen_posttraining.evaluator.EngineeringEvaluationError
    ):
        metricsfix.projected_checkpoint_metrics_validator(
            {
                "checkpoint_validation_metrics": FULL_METRICS,
                "fixed_threshold_0_5": fixed,
                "fixed_threshold_0_5_checkpoint_audit": bad_audit,
            },
            request,
        )

    changed = copy.deepcopy(FULL_METRICS)
    changed["miou"] += 1e-6
    changed_fixed = {**changed, "threshold": 0.5}
    with pytest.raises(
        frozen_posttraining.evaluator.EngineeringEvaluationError
    ):
        metricsfix.projected_checkpoint_metrics_validator(
            {
                "checkpoint_validation_metrics": changed,
                "fixed_threshold_0_5": changed_fixed,
                "fixed_threshold_0_5_checkpoint_audit": (
                    metricsfix._FROZEN_FIXED_THRESHOLD_AUDIT(
                        changed_fixed,
                        changed,
                    )
                ),
            },
            request,
        )


def test_successor_main_installs_and_restores_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = (
        frozen_posttraining.evaluator.validate_checkpoint_local_result
    )
    observed: list[object] = []

    def fake_v3_main(argv):
        observed.append(argv)
        assert (
            frozen_posttraining.evaluator.validate_checkpoint_local_result
            is metricsfix.projected_checkpoint_metrics_validator
        )

    monkeypatch.setattr(metricsfix.overlayfix_v3, "main", fake_v3_main)
    metricsfix.main(["--dry-run"])
    assert observed == [["--dry-run"]]
    assert (
        frozen_posttraining.evaluator.validate_checkpoint_local_result
        is original
    )


def test_actual_four_checkpoints_are_17_to_11_with_no_common_delta() -> None:
    summary = json.loads(
        (RESULT_ROOT / "seed42_replay_summary_v1.json").read_text(
            encoding="utf-8"
        )
    )
    observed = 0
    for run in summary["runs"]:
        for record in run["checkpoints"]:
            checkpoint = torch.load(
                record["path"],
                map_location="cpu",
                weights_only=False,
            )
            full = checkpoint["validation_metrics"]
            projected = record["metrics"]
            assert set(full) == metricsfix.EXPECTED_FULL_METRICS
            assert set(projected) == set(
                frozen_posttraining.summary_core.METRICS
            )
            assert set(full) - set(projected) == set(
                metricsfix.AUXILIARY_METRICS
            )
            assert {
                key: (full[key], projected[key])
                for key in projected
                if full[key] != projected[key]
                or type(full[key]) is not type(projected[key])
            } == {}
            observed += 1
    assert observed == 4


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
                "--execute",
            ],
            (
                "experiments."
                "final_model_seed42_certification_replay_"
                "posttraining_metricsfix_v4"
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
                    "final_model_seed42_certification_replay_posttraining_extra"
                ),
            ],
            (
                "experiments."
                "final_model_seed42_certification_replay_posttraining_extra"
            ),
        ),
    ),
)
def test_redirector_rewrites_only_exact_module(
    tmp_path: Path,
    arguments: list[str],
    expected_module: str,
) -> None:
    capture = tmp_path / "args.txt"
    fake_python = _executable(
        tmp_path / "real-python",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" >"$METRICSFIX_CAPTURE"
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FINAL_MODEL_SEED42_METRICSFIX_REAL_PYTHON": str(
                fake_python
            ),
            "METRICSFIX_CAPTURE": str(capture),
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


def test_completion_wrapper_forces_env_and_forwards_to_envfix(
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
  printf 'CALL\\n'
  printf '%s\\n' "$@"
} >>"$METRICSFIX_INVOCATION_CAPTURE"
""",
    )
    fake_redirector = _executable(
        fake_repo
        / "experiments/"
        "final_model_seed42_certification_python_metricsfix_v4.sh",
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
} >"$METRICSFIX_COMPLETION_CAPTURE"
""",
    )
    fake_lock = fake_repo / "experiments/metricsfix.json"
    fake_lock.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_METRICSFIX_REPO_ROOT": str(fake_repo),
            "FINAL_MODEL_SEED42_METRICSFIX_REAL_PYTHON": str(fake_python),
            "FINAL_MODEL_SEED42_METRICSFIX_SOURCE_LOCK": str(fake_lock),
            "METRICSFIX_INVOCATION_CAPTURE": str(invocation_capture),
            "METRICSFIX_COMPLETION_CAPTURE": str(completion_capture),
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
            "metricsfix_source_lock"
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
            "metricsfix_attestation_v4"
        ),
        "--dry-run",
    ]
    assert completion_capture.read_text(
        encoding="utf-8"
    ).splitlines() == [
        ":4096:8",
        str(fake_redirector),
        "--dry-run",
        "--poll-seconds",
        "23",
    ]


def test_successor_lock_live_verifies_v3_without_rewriting(
    tmp_path: Path,
) -> None:
    upstream = metricsfix_lock.overlayfix_source_lock.DEFAULT_OUTPUT
    upstream_before = upstream.read_bytes()
    output = tmp_path / "metricsfix_source_lock_v4.json"
    created, action = metricsfix_lock.freeze_source_lock(output)
    assert action == "created"
    assert upstream.read_bytes() == upstream_before
    verified = metricsfix_lock.verify_source_lock(created)
    assert verified["status"] == "locked"
    assert verified["source_count"] == 6
    assert (
        verified["upstream_completion_overlayfix_source_lock_v3"][
            "sha256"
        ]
        == metricsfix_lock.EXPECTED_OVERLAYFIX_SOURCE_LOCK_SHA256
    )
    same, action = metricsfix_lock.freeze_source_lock(output)
    assert same == created
    assert action == "skipped_identical_locked"

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        metricsfix_lock.CompletionMetricsfixSourceLockError
    ):
        metricsfix_lock.verify_source_lock(output)


def test_successor_attestation_write_once_and_claim_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_lock = tmp_path / "metricsfix-source-lock.json"
    source_payload = {
        "schema": metricsfix_lock.SCHEMA,
        "source_count": 6,
        "upstream_completion_overlayfix_source_lock_v3": {
            "path": (
                "experiments/"
                "final_model_seed42_certification_completion_"
                "overlayfix_source_lock_v3.json"
            ),
            "sha256": (
                metricsfix_lock.EXPECTED_OVERLAYFIX_SOURCE_LOCK_SHA256
            ),
            "schema": metricsfix_lock.overlayfix_source_lock.SCHEMA,
        },
    }
    source_lock.write_bytes(
        metricsfix_attestation.canonical_json_bytes(source_payload)
    )
    base = tmp_path / "base.json"
    base_payload = {
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
        metricsfix_attestation.canonical_json_bytes(base_payload)
    )
    monkeypatch.setattr(
        metricsfix_attestation.metricsfix_source_lock,
        "verify_source_lock",
        lambda _path: source_payload,
    )
    monkeypatch.setattr(
        metricsfix_attestation.completion,
        "verify_attestation",
        lambda **_kwargs: {"status": "verified_complete"},
    )
    output = tmp_path / "attestation.json"
    first = metricsfix_attestation.finalize_attestation(
        source_lock_path=source_lock,
        base_attestation_path=base,
        output=output,
        require_runtime_env=False,
    )
    assert first["attestation_action"] == "created"
    second = metricsfix_attestation.finalize_attestation(
        source_lock_path=source_lock,
        base_attestation_path=base,
        output=output,
        require_runtime_env=False,
    )
    assert second["attestation_action"] == "skipped_identical_complete"
    assert metricsfix_attestation.verify_attestation(
        source_lock_path=source_lock,
        base_attestation_path=base,
        output=output,
    )["status"] == "verified_complete"
    payload = metricsfix_attestation._canonical_object(
        output,
        "test metricsfix attestation",
    )
    assert payload["paper_core_established"] is False
    assert payload["stability_claim_supported"] is False
    assert (
        payload["claim_boundary"]["multiseed_replication_supported"]
        is False
    )


def test_real_wrapper_dry_run_queries_no_gpu_and_writes_nothing() -> None:
    tracked = (
        metricsfix_lock.DEFAULT_OUTPUT,
        metricsfix_lock.overlayfix_source_lock.DEFAULT_OUTPUT,
        (
            metricsfix_lock.overlayfix_source_lock.envfix_source_lock
            .DEFAULT_OUTPUT
        ),
    )
    before = {path: path.stat().st_mtime_ns for path in tracked}
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": "caller-wrong-value",
            "FINAL_MODEL_SEED42_METRICSFIX_REPO_ROOT": str(REPO_ROOT),
            "FINAL_MODEL_SEED42_METRICSFIX_REAL_PYTHON": sys.executable,
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
    subprocess.run(["bash", "-n", str(REDIRECTOR)], check=True)
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    for source in (
        Path(metricsfix.__file__).resolve(),
        Path(metricsfix_attestation.__file__).resolve(),
        Path(metricsfix_lock.__file__).resolve(),
    ):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
