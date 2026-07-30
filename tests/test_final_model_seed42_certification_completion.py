from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments import (
    final_model_seed42_certification_completion as completion,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining as posttraining,
)
from experiments import (
    freeze_final_model_seed42_certification_completion_source_lock
    as completion_source_lock,
)


def _run_record(arm: str, *, complete: bool) -> dict[str, object]:
    return {
        "arm": arm,
        "variant": "tss_on" if arm == "b" else "tss_qfg",
        "trajectory_seed": 42,
        "run_id": f"run-{arm}",
        "run_directory": str(
            completion.replay_contract.DEFAULT_OUTPUT_ROOT / arm
        ),
        "completed_epoch": 800 if complete else 799,
        "summary_present": complete,
        "strict_formal800_complete": complete,
        "state": (
            "strict_formal800_complete"
            if complete
            else "training_or_finalizing"
        ),
    }


def _write(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_canonical_json_is_finite_and_stable_under_optimized_python():
    payload = {"z": [2, 1], "a": False}
    assert completion.canonical_json_bytes(payload) == (
        b'{"a":false,"z":[2,1]}\n'
    )
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion.canonical_json_bytes({"bad": math.nan})


def test_new_replay_path_guard_rejects_external_legacy_and_other_seeds():
    root = completion.replay_contract.DEFAULT_OUTPUT_ROOT
    assert completion._inside_new_replay(root / "ok", "test") == (
        root / "ok"
    ).resolve()
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion._inside_new_replay(Path("/tmp/outside"), "test")
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion._inside_new_replay(
            root / "tpd_ner_v4_survival_exact_v1/result",
            "test",
        )
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion._inside_new_replay(root / "seed_3407_bad", "test")
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion._inside_new_replay(
            root / "seed_426780603_bad",
            "test",
        )


def test_training_pair_lock_reports_busy_released_and_inherited_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock = tmp_path / "pair.lock"
    monkeypatch.setattr(completion, "TRAINING_PAIR_LOCK", lock)
    assert completion.training_pair_lock_status()["state"] == "not_present"

    lock.touch()
    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert completion.training_pair_lock_status() == {
            "path": str(lock.resolve()),
            "state": "held_by_training_pair",
            "released": False,
        }
        monkeypatch.setenv(
            completion.TRAINING_LOCK_FD_ENV,
            str(descriptor),
        )
        assert completion.training_pair_lock_status() == {
            "path": str(lock.resolve()),
            "state": "held_exclusively_by_completion_runner",
            "released": True,
        }
        monkeypatch.delenv(completion.TRAINING_LOCK_FD_ENV)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        assert completion.training_pair_lock_status()["state"] == "released"
    finally:
        os.close(descriptor)


def test_training_readiness_requires_both_formal800_and_pair_lock_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    records = {
        "b": _run_record("b", complete=True),
        "d": _run_record("d", complete=True),
    }
    monkeypatch.setattr(
        completion,
        "_lightweight_run_progress",
        lambda arm: records[arm],
    )
    monkeypatch.setattr(
        completion,
        "training_pair_lock_status",
        lambda: {"path": "/lock", "state": "held", "released": False},
    )
    monkeypatch.setattr(
        posttraining,
        "collect_requests",
        lambda: (1, 2, 3, 4),
    )
    waiting = completion.training_completion_status()
    assert waiting["ready"] is False
    assert waiting["formal800_artifacts_ready"] is True

    monkeypatch.setattr(
        completion,
        "training_pair_lock_status",
        lambda: {"path": "/lock", "state": "released", "released": True},
    )
    ready = completion.training_completion_status()
    assert ready["ready"] is True
    assert ready["training_pair_process_exited"] is True

    records["d"] = _run_record("d", complete=False)
    waiting = completion.training_completion_status()
    assert waiting["ready"] is False
    assert waiting["formal800_artifacts_ready"] is False


def test_dry_run_is_read_only_and_separates_replay_from_deployment_f1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        completion,
        "verify_completion_source_lock",
        lambda: {
            "schema": completion_source_lock.SCHEMA,
            "source_count": len(completion_source_lock.SOURCE_RELATIVE_PATHS),
        },
    )
    lock = _write(tmp_path / "lock.json")
    monkeypatch.setattr(completion_source_lock, "DEFAULT_OUTPUT", lock)
    monkeypatch.setattr(
        completion,
        "training_completion_status",
        lambda: {
            "status": "waiting_for_new_seed42_formal800",
            "ready": False,
            "runs": [],
        },
    )
    monkeypatch.setattr(
        completion.f1_runner,
        "preflight",
        lambda *args: {
            "status": "ready",
            "gpu_used": False,
            "writes_performed": False,
        },
    )
    payload = completion.dry_run_plan(
        f1_report=tmp_path / "f1.json",
        deep_output=tmp_path / "deep.json",
        attestation=tmp_path / "final.json",
    )
    assert payload["status"] == "waiting_for_new_seed42_training"
    assert payload["f1"]["subject"] == "frozen_deployment_d_not_replay_d"
    assert payload["gpu_queried"] is False
    assert payload["gpu_command_launched"] is False
    assert payload["persistent_artifact_written"] is False
    assert list(tmp_path.iterdir()) == [lock]


def test_verify_posttraining_enforces_fixed_seed_claim_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    base = {
        "status": "verified_complete",
        "run_count": 2,
        "sweep_count": 4,
        "trajectory_seeds": [42],
        "excluded_seeds": [3407, 426780603],
        "fixed_threshold": 0.5,
        "closure": {
            "path": str(posttraining.DEFAULT_CLOSURE),
            "sha256": "0" * 64,
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
    }
    monkeypatch.setattr(
        posttraining,
        "verify_complete_closure",
        lambda: dict(base),
    )
    assert completion.verify_posttraining()["status"] == "verified_complete"

    changed = dict(base)
    changed["stability_claim_supported"] = True
    monkeypatch.setattr(
        posttraining,
        "verify_complete_closure",
        lambda: changed,
    )
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion.verify_posttraining()


def test_build_attestation_binds_gate_frozen_f1_and_deep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    closure = _write(tmp_path / "closure.json")
    gate_path = tmp_path / "gate.json"
    comparisons = [
        {
            "selection_role": role,
            "metrics": {
                "pd": 1.0,
                "fa": 0.0,
                "miou": 0.9,
                "tiny_pd": 1.0,
                "false_objects_per_image": 0.0,
            },
        }
        for role in ("primary_best_miou", "secondary_best_pd")
    ]
    gate = {
        "schema": posttraining.GATE_SCHEMA,
        "status": "complete",
        "decision": "TEST_GATE_DECISION",
        "trajectory_seeds": [42],
        "fixed_threshold": 0.5,
        "fixed_threshold_and_budget_comparisons": comparisons,
        "claim_boundary": {
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }
    gate_path.write_bytes(completion.canonical_json_bytes(gate))
    monkeypatch.setattr(posttraining, "DEFAULT_CLOSURE", closure)
    monkeypatch.setattr(posttraining, "DEFAULT_GATE", gate_path)
    monkeypatch.setattr(
        completion,
        "verify_completion_source_lock",
        lambda: {
            "schema": completion_source_lock.SCHEMA,
            "source_count": len(completion_source_lock.SOURCE_RELATIVE_PATHS),
        },
    )
    completion_lock = _write(tmp_path / "completion_source_lock.json")
    replay_lock = _write(tmp_path / "replay_source_lock.json")
    f0_lock = _write(tmp_path / "f0_source_lock.json")
    locked_parent = _write(tmp_path / "parent_lock.json")
    monkeypatch.setattr(completion_source_lock, "DEFAULT_OUTPUT", completion_lock)
    monkeypatch.setattr(posttraining, "DEFAULT_REPLAY_SOURCE_LOCK", replay_lock)
    monkeypatch.setattr(posttraining, "DEFAULT_CERTIFICATION_SOURCE_LOCK", f0_lock)
    monkeypatch.setattr(posttraining, "DEFAULT_PARENT_LOCK", locked_parent)
    monkeypatch.setattr(
        completion,
        "training_completion_status",
        lambda: {"ready": True},
    )
    monkeypatch.setattr(
        completion,
        "_strict_training_bindings",
        lambda status: [{"arm": "b"}, {"arm": "d"}],
    )
    monkeypatch.setattr(
        completion,
        "verify_posttraining",
        lambda: {
            "status": "verified_complete",
            "run_count": 2,
            "sweep_count": 4,
            "trajectory_seeds": [42],
            "excluded_seeds": [3407, 426780603],
            "fixed_threshold": 0.5,
            "closure": {
                "path": str(closure),
                "sha256": completion._sha256_file(closure, "closure"),
            },
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    )
    f1_report = _write(tmp_path / "f1/f1.json")
    cache_payloads = {}
    modes = {}
    for mode in completion.f1_runner.PUBLIC_MODES:
        cache = _write(tmp_path / f"f1/caches/{mode}.json")
        relative = cache.relative_to(f1_report.parent).as_posix()
        cache_payloads[mode] = cache
        modes[mode] = {
            "cache": {
                "path": relative,
                "sha256": completion._sha256_file(cache, mode),
            }
        }
    f1 = {
        "status": "complete",
        "official_test_accessed": False,
        "execution_contract": {
            "fixed_threshold": 0.5,
            "checkpoint_sha256": completion.f1_runner.EXPECTED_INFERENCE_SHA256,
            "source_checkpoint_sha256": (
                completion.f1_runner.EXPECTED_SOURCE_CHECKPOINT_SHA256
            ),
            "validation_count": 133,
            "validation_ids_sha256": "a" * 64,
        },
        "functional_gate": {
            "status": "complete",
            "qfg_functionally_active": True,
            "performance_causal_claim_established": False,
        },
        "modes": modes,
    }
    monkeypatch.setattr(completion, "_verify_f1", lambda path: f1)
    deep_output = _write(tmp_path / "deep.json")
    monkeypatch.setattr(
        completion,
        "_verify_deep",
        lambda deep, report: {
            "status": "verified",
            "no_invention_status": True,
            "limitations": [],
        },
    )
    monkeypatch.setattr(
        completion,
        "_source_bindings",
        lambda: {"completion_helper": {"path": "x", "sha256": "b" * 64}},
    )

    result = completion.build_attestation(
        f1_report=f1_report,
        deep_output=deep_output,
    )
    assert result["status"] == "complete"
    assert result["new_seed42_replay"][
        "old_seed42_stage_results_used_as_new_replay"
    ] is False
    assert result["frozen_deployment_d_qfg_audit"][
        "subject_is_replay_d"
    ] is False
    assert set(
        result["frozen_deployment_d_qfg_audit"]["cache_manifests"]
    ) == set(completion.f1_runner.PUBLIC_MODES)
    assert result["paper_core_established"] is False
    assert result["stability_claim_supported"] is False
    f2 = result["implementation_closure"][
        "f2_engineering_replication_tooling"
    ]
    assert f2["contract_runner_tests_implemented"] is True
    assert f2["multiseed_execution_complete"] is False
    assert f2["current_gate_uses_new_seed42_replay_only"] is True
    assert "supplementary_only" in f2["seed_3407_execution"]
    assert "cancelled" in f2["seed_426780603_execution"]


def test_write_once_is_idempotent_and_rejects_different_existing_bytes(
    tmp_path: Path,
):
    output = tmp_path / "attestation.json"
    payload = {"schema": "test", "status": "complete"}
    first, action = completion._write_or_validate(output, payload)
    assert action == "created"
    second, action = completion._write_or_validate(output, payload)
    assert action == "skipped_identical_complete"
    assert first == second
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion._write_or_validate(output, {"schema": "different"})


def test_training_ready_cli_uses_distinct_waiting_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        completion,
        "training_completion_status",
        lambda: {
            "status": "waiting",
            "ready": False,
            "training_pair_lock": {"released": False},
        },
    )
    with pytest.raises(SystemExit) as exc:
        completion.main(["--training-ready"])
    assert exc.value.code == completion.WAITING_EXIT_CODE
    assert json.loads(capsys.readouterr().out)["ready"] is False


def test_runtime_gpu2_assertion_checks_exact_visibility(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "Fake RTX 5090",
    )
    monkeypatch.setattr(completion.f1_runner, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", completion.GPU2_UUID)
    monkeypatch.setenv(completion.GPU_INDEX_ENV, "2")
    monkeypatch.setenv(completion.GPU_UUID_ENV, completion.GPU2_UUID)
    payload = completion.assert_runtime_gpu2()
    assert payload["physical_gpu_index"] == 2
    assert payload["gpu_queried"] is True

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong")
    with pytest.raises(completion.Seed42CertificationCompletionError):
        completion.assert_runtime_gpu2()


def test_completion_shell_is_ordered_fail_fast_and_dry_run_precedes_writes():
    shell = completion.SHELL_PATH
    subprocess.run(["bash", "-n", str(shell)], check=True)
    text = shell.read_text(encoding="utf-8")
    dry_run = text.index('if [[ "$mode" == "--dry-run" ]]')
    mkdir = text.index('mkdir -p "$output_root"')
    posttraining_call = text.index('"$post_launcher" --run')
    f1_call = text.index('-m "$f1_module"')
    deep_call = text.index('-m "$deep_module"')
    final_call = text.index("--finalize-attestation")
    assert dry_run < mkdir
    assert posttraining_call < f1_call < deep_call < final_call
    assert 'exec 8>"$training_pair_lock"' in text
    assert "FINAL_MODEL_SEED42_COMPLETION_TRAINING_LOCK_FD=8" in text
    assert "nvidia-smi" not in text
    assert "tpd_ner_v4_survival_exact_v1" not in text
    assert "final_model_engineering_replication_v1" not in text


def test_completion_successor_source_lock_covers_posttraining_and_orchestrator():
    expected = {
        "experiments/final_model_seed42_certification_replay_posttraining.py",
        "experiments/run_final_model_seed42_certification_replay_posttraining_2x5090.sh",
        "tests/test_final_model_seed42_certification_replay_posttraining.py",
        "experiments/final_model_seed42_certification_completion.py",
        "experiments/run_final_model_seed42_certification_completion.sh",
        "tests/test_final_model_seed42_certification_completion.py",
        "experiments/evaluate_final_model_engineering_replication_pd_fa.py",
        "experiments/analyze_final_model_engineering_paired_screen.py",
        "experiments/adjudicate_final_model_engineering_gate.py",
        "tests/test_evaluate_final_model_engineering_replication_pd_fa.py",
        "tests/test_analyze_final_model_engineering_paired_screen.py",
        "tests/test_adjudicate_final_model_engineering_gate.py",
    }
    assert expected.issubset(
        set(completion_source_lock.SOURCE_RELATIVE_PATHS)
    )
    payload = completion_source_lock.build_source_lock()
    assert payload["schema"] == completion_source_lock.SCHEMA
    assert payload["source_count"] == len(
        completion_source_lock.SOURCE_RELATIVE_PATHS
    )
    assert payload["readiness"][
        "f2_full_parameter_contract_runner_tests_complete"
    ] is True
    assert payload["execution_scope"][
        "supplementary_seed_3407_used_in_current_gate"
    ] is False
    assert payload["execution_scope"]["seed_426780603_scheduled"] is False


def test_completion_source_lock_is_write_once_and_live_verifiable(
    tmp_path: Path,
):
    output = tmp_path / "completion_source_lock.json"
    path, action = completion_source_lock.freeze_source_lock(output)
    assert action == "created"
    path2, action = completion_source_lock.freeze_source_lock(output)
    assert path2 == path
    assert action == "skipped_identical_locked"
    verified = completion_source_lock.verify_source_lock(output)
    assert verified["status"] == "locked"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(completion_source_lock.CompletionSourceLockError):
        completion_source_lock.verify_source_lock(output)
