from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import textwrap
import time
from types import SimpleNamespace

import pytest

from experiments import (
    final_model_post_training_closure_preflight as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL = (
    REPO_ROOT
    / "experiments/run_final_model_post_training_closure_2x5090.sh"
)
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"


FAKE_PYTHON = r"""#!/usr/bin/env python3
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


arguments = sys.argv[1:]
module = arguments[1] if len(arguments) >= 2 and arguments[0] == "-m" else ""
log_path = Path(os.environ["FAKE_CALL_LOG"])


def option(name):
    if name not in arguments:
        return None
    return arguments[arguments.index(name) + 1]


def record(event, **extra):
    payload = {
        "event": event,
        "module": module,
        "arguments": arguments,
        "arm": option("--arm"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": os.environ.get(
            "FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX"
        ),
        "physical_gpu_uuid": os.environ.get(
            "FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID"
        ),
        "monotonic_ns": time.monotonic_ns(),
        **extra,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def create_file(path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text("{}\n", encoding="utf-8")


record("start")
if (
    module
    == "experiments.evaluate_final_model_engineering_replication_pd_fa"
    and "--execute" in arguments
):
    arm = option("--arm")
    if os.environ.get("FAKE_FAIL_ARM") == arm:
        time.sleep(float(os.environ.get("FAKE_FAIL_DELAY", "0")))
        record("failure", returncode=int(os.environ.get("FAKE_FAIL_CODE", "17")))
        raise SystemExit(int(os.environ.get("FAKE_FAIL_CODE", "17")))
    sleep_arm = os.environ.get("FAKE_SLEEP_ARM")
    if sleep_arm == arm:
        marker = os.environ.get("FAKE_TERMINATION_MARKER")

        def terminate(_signum, _frame):
            if marker:
                create_file(marker)
            record("terminated", returncode=143)
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, terminate)
        time.sleep(float(os.environ.get("FAKE_SLEEP_SECONDS", "10")))
    else:
        time.sleep(float(os.environ.get("FAKE_EXECUTE_DELAY", "0.05")))
elif module == "experiments.summarize_final_model_engineering_replication":
    create_file(option("--output"))
elif module == "experiments.analyze_final_model_engineering_paired_screen":
    create_file(option("--output"))
elif module == "analysis.run_final_qfg_six_mode_audit" and "--run" in arguments:
    output_directory = Path(option("--output-dir"))
    create_file(
        output_directory / "final_model_qfg_six_mode_audit_v1.json"
    )
elif (
    module == "analysis.verify_final_qfg_six_mode_audit_deep"
    and "--write-once" in arguments
):
    create_file(option("--output"))
elif (
    module
    == "experiments.final_model_post_training_closure_preflight"
    and "--publish-f1-staging" in arguments
):
    staging = Path(option("--f1-staging-dir"))
    final = Path(option("--f1-report")).parent
    staging.rename(final)
elif module == "experiments.adjudicate_final_model_engineering_gate":
    create_file(option("--output"))
elif (
    module == "experiments.final_model_post_training_closure_preflight"
    and "--finalize-closure" in arguments
):
    create_file(option("--closure-output"))
record("end")
"""


def _closure_environment(
    temporary_root: Path,
    *,
    python: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "FINAL_MODEL_POST_REPO_ROOT": str(REPO_ROOT),
            "FINAL_MODEL_POST_PYTHON": str(python),
            "FINAL_MODEL_POST_OUTPUT_ROOT": str(temporary_root / "output"),
            "FINAL_MODEL_POST_SOURCE_LOCK": str(
                temporary_root / "source-lock.json"
            ),
            "FINAL_MODEL_POST_SEED_CONTRACT": str(
                temporary_root / "seed-contract.json"
            ),
            "FINAL_MODEL_POST_MANIFEST_DIRECTORY": str(
                temporary_root / "manifests"
            ),
            "FINAL_MODEL_POST_SUMMARY": str(temporary_root / "summary.json"),
            "FINAL_MODEL_POST_PAIRED_OUTPUT": str(
                temporary_root / "paired.json"
            ),
            "FINAL_MODEL_POST_F1_OUTPUT_DIR": str(temporary_root / "f1"),
            "FINAL_MODEL_POST_DEEP_OUTPUT": str(temporary_root / "deep.json"),
            "FINAL_MODEL_POST_GATE_OUTPUT": str(temporary_root / "gate.json"),
            "FINAL_MODEL_POST_CLOSURE_OUTPUT": str(
                temporary_root / "closure.json"
            ),
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return environment


def _write_fake_python(path: Path) -> None:
    path.write_text(textwrap.dedent(FAKE_PYTHON), encoding="utf-8")
    path.chmod(0o755)


def _events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _first_event(
    events: list[dict[str, object]],
    module: str,
    flag: str,
    *,
    event: str = "start",
) -> dict[str, object]:
    return next(
        item
        for item in events
        if item["event"] == event
        and item["module"] == module
        and flag in item["arguments"]
    )


def _functional_report(
    *,
    repeat_equivalent: bool = True,
    output_different: bool = True,
    nontrivial_factor: bool = True,
) -> dict[str, object]:
    active = (
        repeat_equivalent
        and output_different
        and nontrivial_factor
    )
    return {
        "status": "complete",
        "functional_gate": {
            "status": "complete",
            "repeat_inference_equivalent": repeat_equivalent,
            "full_vs_qfg_off_functionally_different": output_different,
            "nontrivial_factor_use": nontrivial_factor,
            "qfg_functionally_active": active,
            "performance_causal_claim_established": False,
        },
        "repeat_inference": {"equivalent": repeat_equivalent},
        "modes": {
            "qfg_off": {
                "comparison_to_full": {
                    "output_difference": {
                        "functionally_different": output_different,
                    }
                }
            },
            "full": {
                "factor_summary": {
                    "nontrivial_factor_use": nontrivial_factor,
                }
            },
        },
    }


def _engineering_gate_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "decision": "ENGINEERING_GATE_S_E_PASS",
        "gates": {
            "M-train": {
                "status": "insufficient_evidence",
                "passed": None,
            }
        },
        "claim_boundary": {
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }


def test_preflight_has_exact_matrix_and_does_not_query_gpu() -> None:
    payload = subject.preflight(f1_gpu_index=3)
    assert payload["status"] == "ready"
    assert payload["run_matrix"]["run_count"] == 4
    assert payload["run_matrix"]["sweep_count"] == 8
    assert payload["run_matrix"]["arm_b"] == {
        "physical_gpu_index": 2,
        "physical_gpu_uuid": GPU2_UUID,
        "sweep_count": 4,
    }
    assert payload["run_matrix"]["arm_d"] == {
        "physical_gpu_index": 3,
        "physical_gpu_uuid": GPU3_UUID,
        "sweep_count": 4,
    }
    assert payload["f1_assignment"]["physical_gpu_index"] == 3
    assert payload["source_lock"]["source_count"] == 31
    assert (
        payload["stage_order"][4]
        == "cpu_engineering_paired_screen"
    )
    assert (
        payload["source_bindings"]["paired_screen_analyzer"]["path"]
        == "experiments/analyze_final_model_engineering_paired_screen.py"
    )
    assert payload["gpu_queried"] is False
    assert payload["writes_performed"] is False
    assert (
        "atomic_directory_publish_noreplace"
        in payload["reentry_policy"]["f1"]
    )


def test_functional_activation_is_boolean_conjunction_not_causal_claim() -> None:
    verified = subject._verify_f1_functional_activation(
        _functional_report(),
    )
    assert verified == {
        "status": "verified",
        "qfg_functionally_active": True,
        "logical_inputs": {
            "repeat_inference_equivalent": True,
            "full_vs_qfg_off_functionally_different": True,
            "nontrivial_factor_use": True,
        },
        "logical_rule": "all_three_inputs_must_be_true",
        "performance_causal_claim_established": False,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("active_type", "functionally-active flag must be Boolean"),
        ("conjunction", "functional-activation conjunction differs"),
        ("repeat_source", "gate/repeat-inference equivalence differs"),
        ("causal", "performance-causal-claim boundary differs"),
    ),
)
def test_functional_activation_rejects_logic_or_type_drift(
    mutation: str,
    match: str,
) -> None:
    report = copy.deepcopy(_functional_report())
    if mutation == "active_type":
        report["functional_gate"]["qfg_functionally_active"] = 1
    elif mutation == "conjunction":
        report["functional_gate"]["qfg_functionally_active"] = False
    elif mutation == "repeat_source":
        report["repeat_inference"]["equivalent"] = False
    elif mutation == "causal":
        report["functional_gate"][
            "performance_causal_claim_established"
        ] = True
    else:
        raise AssertionError(mutation)
    with pytest.raises(subject.PostTrainingClosureError, match=match):
        subject._verify_f1_functional_activation(report)


def test_deep_summary_binds_no_invention_and_limitation_count() -> None:
    deep = {
        "no_invention_status": True,
        "limitations": [
            {"field": "repeat", "reason": "second cache unavailable"},
            {"field": "runtime", "reason": "runtime state not persisted"},
        ],
    }
    assert subject._deep_verification_summary(deep) == {
        "status": "verified",
        "no_invention_status": True,
        "limitations_count": 2,
    }
    deep["no_invention_status"] = False
    with pytest.raises(
        subject.PostTrainingClosureError,
        match="no-invention status differs",
    ):
        subject._deep_verification_summary(deep)


def test_attestation_binds_functional_and_deep_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f1_report = tmp_path / "f1.json"
    deep_output = tmp_path / "deep.json"
    f1_report.write_text("{}\n", encoding="utf-8")
    deep_output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        subject,
        "preflight",
        lambda **_kwargs: {
            "run_matrix": {
                "arm_b": {"physical_gpu_index": 2},
                "arm_d": {"physical_gpu_index": 3},
            },
            "f1_assignment": {"physical_gpu_index": 2},
            "source_lock": {"source_count": 31},
            "source_bindings": {},
            "reentry_policy": {},
            "parallel_failure_policy": "first_nonzero_terminates_peer",
        },
    )
    monkeypatch.setattr(
        subject,
        "_verify_summary",
        lambda **_kwargs: {"run_count": 4},
    )
    monkeypatch.setattr(
        subject,
        "_verify_eight_result_manifest",
        lambda **_kwargs: {"result_count": 8},
    )
    monkeypatch.setattr(
        subject,
        "verify_paired_screen_output",
        lambda **_kwargs: {
            "decision": "ENGINEERING_PAIRED_SCREEN_ROUTE_MET",
            "engineering_paired_route_met": True,
            "establishes_gate_m_train": False,
            "gate_m_train_status": "insufficient_evidence",
            "gate_m_train_passed": None,
        },
    )
    monkeypatch.setattr(
        subject.f1_runner,
        "verify_audit_report",
        lambda *_args, **_kwargs: _functional_report(),
    )
    monkeypatch.setattr(
        subject.deep_verifier,
        "verify_deep_verification",
        lambda *_args, **_kwargs: {
            "status": "verified",
            "no_invention_status": True,
            "limitations": [
                {"field": "repeat", "reason": "not persisted"},
                {"field": "runtime", "reason": "not persisted"},
            ],
        },
    )
    monkeypatch.setattr(
        subject,
        "verify_gate_output",
        lambda **_kwargs: {
            "decision": "ENGINEERING_GATE_S_E_PASS",
            "gate_m_train_status": "insufficient_evidence",
            "gate_m_train_passed": None,
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    )
    payload = subject.build_closure_attestation(
        repo_root=tmp_path,
        output_root=tmp_path,
        source_lock_path=tmp_path / "source-lock.json",
        seed_contract_path=tmp_path / "seed.json",
        manifest_directory=tmp_path / "manifests",
        summary_path=tmp_path / "summary.json",
        paired_output=tmp_path / "paired.json",
        f1_report=f1_report,
        deep_output=deep_output,
        gate_output=tmp_path / "gate.json",
        f1_gpu_index=2,
    )
    assert payload["qfg_functional_activation"][
        "qfg_functionally_active"
    ] is True
    assert payload["qfg_functional_activation"][
        "performance_causal_claim_established"
    ] is False
    assert payload["deep_verification_summary"] == {
        "status": "verified",
        "no_invention_status": True,
        "limitations_count": 2,
    }
    assert (
        payload["paired_screen_gate_m_train_status"]
        == "insufficient_evidence"
    )
    assert payload["paired_screen_gate_m_train_passed"] is None
    assert (
        payload["engineering_gate_m_train_status"]
        == "insufficient_evidence"
    )
    assert payload["engineering_gate_m_train_passed"] is None
    assert payload["paper_core_established"] is False
    assert payload["stability_claim_supported"] is False
    assert payload["artifacts"]["f1_six_mode_audit"][
        "functional_activation"
    ] == payload["qfg_functional_activation"]
    assert payload["artifacts"]["f1_deep_verification"][
        "limitations_count"
    ] == 2


def test_runtime_gpu_assertion_rejects_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda _index: "synthetic RTX 5090",
    )
    monkeypatch.setattr(
        subject,
        "_registered_gpu_bindings",
        lambda: dict(subject.GPU_BINDINGS),
    )
    monkeypatch.setattr(
        subject.evaluator.sweep_core,
        "torch",
        SimpleNamespace(cuda=fake_cuda),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU2_UUID)
    monkeypatch.setenv(
        subject.evaluator.EVALUATION_PHYSICAL_GPU_INDEX_ENV,
        "2",
    )
    monkeypatch.setenv(
        subject.evaluator.EVALUATION_PHYSICAL_GPU_UUID_ENV,
        GPU2_UUID,
    )
    assert subject.assert_runtime_gpu(2)["visible_cuda_device_count"] == 1

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU3_UUID)
    with pytest.raises(
        subject.PostTrainingClosureError,
        match="CUDA_VISIBLE_DEVICES differs",
    ):
        subject.assert_runtime_gpu(2)


def test_closure_attestation_is_create_once_then_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema": subject.ATTESTATION_SCHEMA,
        "status": "complete",
        "gate_decision": "ENGINEERING_GATE_S_E_PASS",
        "qfg_functional_activation": {
            "qfg_functionally_active": True,
            "logical_inputs": {
                "repeat_inference_equivalent": True,
                "full_vs_qfg_off_functionally_different": True,
                "nontrivial_factor_use": True,
            },
            "logical_rule": "all_three_inputs_must_be_true",
            "performance_causal_claim_established": False,
        },
        "deep_verification_summary": {
            "no_invention_status": True,
            "limitations_count": 4,
        },
        "engineering_gate_m_train_status": "insufficient_evidence",
        "engineering_gate_m_train_passed": None,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }
    monkeypatch.setattr(
        subject,
        "build_closure_attestation",
        lambda **_kwargs: payload,
    )
    closure = tmp_path / "closure.json"
    arguments = {
        "repo_root": tmp_path,
        "output_root": tmp_path,
        "source_lock_path": tmp_path / "lock.json",
        "seed_contract_path": tmp_path / "seed.json",
        "manifest_directory": tmp_path / "manifests",
        "summary_path": tmp_path / "summary.json",
        "paired_output": tmp_path / "paired.json",
        "f1_report": tmp_path / "f1.json",
        "deep_output": tmp_path / "deep.json",
        "gate_output": tmp_path / "gate.json",
        "closure_output": closure,
        "f1_gpu_index": 2,
    }
    first = subject.finalize_closure(**arguments)
    second = subject.finalize_closure(**arguments)
    assert first["attestation_action"] == "created"
    assert second["attestation_action"] == "skipped_identical_complete"
    assert first["qfg_functionally_active"] is True
    assert (
        first["functional_activation_logical_rule"]
        == "all_three_inputs_must_be_true"
    )
    assert first["performance_causal_claim_established"] is False
    assert first["deep_no_invention_status"] is True
    assert first["deep_limitations_count"] == 4
    assert (
        first["engineering_gate_m_train_status"]
        == "insufficient_evidence"
    )
    assert first["engineering_gate_m_train_passed"] is None
    assert first["paper_core_established"] is False
    assert first["stability_claim_supported"] is False
    closure.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        subject.PostTrainingClosureError,
        match="stored/rebuilt closure attestation differs",
    ):
        subject.finalize_closure(**arguments)


@pytest.mark.parametrize(
    ("collision_kind", "match"),
    (
        ("symlink", "must not be a symlink"),
        ("directory", "must be a regular file"),
    ),
)
def test_closure_concurrent_publish_rejects_nonregular_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
    match: str,
) -> None:
    destination = tmp_path / "closure.json"
    target = tmp_path / "foreign.json"
    target.write_text("{}\n", encoding="utf-8")

    def collide(
        _source: Path,
        requested: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        path = Path(requested)
        if collision_kind == "symlink":
            path.symlink_to(target)
        else:
            path.mkdir()
        raise FileExistsError

    monkeypatch.setattr(subject.os, "link", collide)
    with pytest.raises(subject.PostTrainingClosureError, match=match):
        subject._write_or_validate(
            destination,
            {"schema": subject.ATTESTATION_SCHEMA, "status": "complete"},
        )


def _f1_staging_fixture(
    root: Path,
) -> tuple[Path, Path, Path]:
    parent = root / "analysis-results"
    parent.mkdir()
    final = parent / "final_model_qfg_six_mode_audit_v1"
    container = (
        parent
        / f".{final.name}.closure-staging.ABC123"
    )
    staging = container / "payload"
    staging.mkdir(parents=True)
    report = staging / subject.f1_runner.REPORT_FILENAME
    report.write_text("{}\n", encoding="utf-8")
    return staging, final, report


def test_f1_staging_is_verified_then_atomically_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging, final, staged_report = _f1_staging_fixture(tmp_path)
    verified: list[Path] = []

    def verify(path: Path, **_kwargs: object) -> dict[str, object]:
        verified.append(Path(path))
        return {"status": "complete"}

    monkeypatch.setattr(
        subject.f1_runner,
        "verify_audit_report",
        verify,
    )
    result = subject.publish_f1_staging_directory(
        staging_output_dir=staging,
        final_output_dir=final,
        repo_root=tmp_path,
        source_lock_path=tmp_path / "source.json",
    )
    final_report = final / subject.f1_runner.REPORT_FILENAME
    assert result["atomic_directory_publish"] is True
    assert result["rename_noreplace"] is True
    assert final_report.is_file()
    assert not staging.exists()
    assert verified == [staged_report, final_report]


def test_f1_atomic_publish_never_replaces_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging, final, _ = _f1_staging_fixture(tmp_path)
    final.mkdir()
    sentinel = final / "sentinel.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        subject.f1_runner,
        "verify_audit_report",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        subject.publish_f1_staging_directory(
            staging_output_dir=staging,
            final_output_dir=final,
            repo_root=tmp_path,
            source_lock_path=tmp_path / "source.json",
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert staging.is_dir()


def test_f1_atomic_publish_rejects_target_created_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging, final, _ = _f1_staging_fixture(tmp_path)
    sentinel = final / "sentinel.txt"

    def verify(*_args: object, **_kwargs: object) -> dict[str, object]:
        final.mkdir()
        sentinel.write_text("concurrent\n", encoding="utf-8")
        return {"status": "complete"}

    monkeypatch.setattr(
        subject.f1_runner,
        "verify_audit_report",
        verify,
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        subject.publish_f1_staging_directory(
            staging_output_dir=staging,
            final_output_dir=final,
            repo_root=tmp_path,
            source_lock_path=tmp_path / "source.json",
        )
    assert sentinel.read_text(encoding="utf-8") == "concurrent\n"
    assert staging.is_dir()


def test_invalid_f1_staging_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging, final, _ = _f1_staging_fixture(tmp_path)

    def reject(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid staged F1 report")

    monkeypatch.setattr(
        subject.f1_runner,
        "verify_audit_report",
        reject,
    )
    with pytest.raises(ValueError, match="invalid staged F1 report"):
        subject.publish_f1_staging_directory(
            staging_output_dir=staging,
            final_output_dir=final,
            repo_root=tmp_path,
            source_lock_path=tmp_path / "source.json",
        )
    assert staging.is_dir()
    assert not final.exists()


def test_paired_screen_existing_output_is_recomputed_and_compared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema": subject.paired_screen.SCHEMA,
        "status": "complete",
        "decision": "ENGINEERING_PAIRED_SCREEN_ROUTE_MET",
        "manifest": {"result_count": 8},
        "engineering_paired_route_met": True,
        "establishes_gate_m_train": False,
        "gates": {
            "M-train": {
                "status": "insufficient_evidence",
                "passed": None,
                "establishes_gate_m_train": False,
            }
        },
    }
    monkeypatch.setattr(
        subject.paired_screen,
        "analyze",
        lambda *, manifest_path: payload,
    )
    output = tmp_path / "paired.json"
    output.write_bytes(subject.paired_screen.canonical_json_bytes(payload))
    verified = subject.verify_paired_screen_output(
        output_root=tmp_path,
        paired_output=output,
    )
    assert verified["status"] == "verified_complete"
    assert verified["engineering_paired_route_met"] is True
    assert verified["establishes_gate_m_train"] is False
    assert verified["gate_m_train_status"] == "insufficient_evidence"
    assert verified["gate_m_train_passed"] is None

    invalid_gate = copy.deepcopy(payload)
    invalid_gate["gates"]["M-train"]["status"] = "passed"
    monkeypatch.setattr(
        subject.paired_screen,
        "analyze",
        lambda *, manifest_path: invalid_gate,
    )
    output.write_bytes(
        subject.paired_screen.canonical_json_bytes(invalid_gate)
    )
    with pytest.raises(
        subject.PostTrainingClosureError,
        match="Gate M-train status differs",
    ):
        subject.verify_paired_screen_output(
            output_root=tmp_path,
            paired_output=output,
        )
    monkeypatch.setattr(
        subject.paired_screen,
        "analyze",
        lambda *, manifest_path: payload,
    )
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        subject.PostTrainingClosureError,
        match="stored/recomputed engineering paired screen differs",
    ):
        subject.verify_paired_screen_output(
            output_root=tmp_path,
            paired_output=output,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("m_train_status", "Gate M-train status differs"),
        ("m_train_passed", "Gate M-train passed differs"),
        ("paper_core", "Gate paper-core claim boundary differs"),
        ("stability", "Gate stability claim boundary differs"),
    ),
)
def test_gate_verification_rejects_claim_boundary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    payload = _engineering_gate_payload()
    if mutation == "m_train_status":
        payload["gates"]["M-train"]["status"] = "passed"
    elif mutation == "m_train_passed":
        payload["gates"]["M-train"]["passed"] = True
    elif mutation == "paper_core":
        payload["claim_boundary"]["paper_core_established"] = True
    elif mutation == "stability":
        payload["claim_boundary"]["stability_claim_supported"] = True
    else:
        raise AssertionError(mutation)
    monkeypatch.setattr(
        subject.gate_core,
        "adjudicate",
        lambda **_kwargs: payload,
    )
    output = tmp_path / "gate.json"
    output.write_bytes(subject.gate_core.canonical_json_bytes(payload))
    with pytest.raises(subject.PostTrainingClosureError, match=match):
        subject.verify_gate_output(
            output_root=tmp_path,
            source_lock_path=tmp_path / "source.json",
            seed_contract_path=tmp_path / "seed.json",
            manifest_directory=tmp_path / "manifests",
            summary_path=tmp_path / "summary.json",
            gate_output=output,
        )

def test_dry_run_is_cpu_only_and_creates_no_outputs(tmp_path: Path) -> None:
    environment = _closure_environment(tmp_path, python=PYTHON)
    environment["FINAL_MODEL_POST_SOURCE_LOCK"] = str(
        REPO_ROOT
        / "experiments/final_model_certification_source_lock_v1.json"
    )
    environment["FINAL_MODEL_POST_SEED_CONTRACT"] = str(
        REPO_ROOT / "experiments/final_model_replication_seed_contract.json"
    )
    environment["FINAL_MODEL_POST_MANIFEST_DIRECTORY"] = str(
        REPO_ROOT / "experiments/final_model_replication_manifests_v1"
    )
    result = subprocess.run(
        ["bash", str(SHELL), "--dry-run", "--f1-gpu", "3"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN ONLY; no GPU command was launched." in result.stdout
    assert f"GPU2 UUID {GPU2_UUID} -> arm B (4 sweeps)" in result.stdout
    assert f"GPU3 UUID {GPU3_UUID} -> arm D (4 sweeps)" in result.stdout
    assert "engineering B/D paired screen on CPU" in result.stdout
    assert "physical GPU3" in result.stdout
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "f1").exists()
    assert not (tmp_path / "paired.json").exists()
    assert not (tmp_path / "deep.json").exists()
    assert not (tmp_path / "gate.json").exists()
    assert not (tmp_path / "closure.json").exists()


def test_formal_orchestration_binds_arms_orders_stages_and_reenters(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "fake-python"
    _write_fake_python(fake_python)
    log = tmp_path / "calls.jsonl"
    environment = _closure_environment(tmp_path, python=fake_python)
    environment["FAKE_CALL_LOG"] = str(log)
    environment["FAKE_EXECUTE_DELAY"] = "0.15"
    orphan = tmp_path / ".f1.closure-staging.ORPHAN"
    orphan.mkdir()
    (orphan / "sentinel.txt").write_text(
        "do not delete\n",
        encoding="utf-8",
    )

    first = subprocess.run(
        ["bash", str(SHELL), "--run", "--f1-gpu", "3"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert first.returncode == 0, first.stderr
    assert (orphan / "sentinel.txt").read_text(
        encoding="utf-8"
    ) == "do not delete\n"
    events = _events(log)
    evaluator_module = (
        "experiments.evaluate_final_model_engineering_replication_pd_fa"
    )
    execute_starts = [
        item
        for item in events
        if item["event"] == "start"
        and item["module"] == evaluator_module
        and "--execute" in item["arguments"]
    ]
    assert len(execute_starts) == 2
    by_arm = {str(item["arm"]): item for item in execute_starts}
    assert by_arm["b"]["cuda_visible_devices"] == GPU2_UUID
    assert by_arm["b"]["physical_gpu_index"] == "2"
    assert by_arm["b"]["physical_gpu_uuid"] == GPU2_UUID
    assert by_arm["d"]["cuda_visible_devices"] == GPU3_UUID
    assert by_arm["d"]["physical_gpu_index"] == "3"
    assert by_arm["d"]["physical_gpu_uuid"] == GPU3_UUID

    execute_ends = [
        item
        for item in events
        if item["event"] == "end"
        and item["module"] == evaluator_module
        and "--execute" in item["arguments"]
    ]
    assert len(execute_ends) == 2
    assert max(
        int(item["monotonic_ns"]) for item in execute_starts
    ) < min(int(item["monotonic_ns"]) for item in execute_ends)
    summary_end = _first_event(
        events,
        "experiments.summarize_final_model_engineering_replication",
        "--output",
        event="end",
    )
    finalize_start = _first_event(
        events,
        evaluator_module,
        "--finalize-manifest",
    )
    verify_start = _first_event(
        events,
        evaluator_module,
        "--verify-results",
    )
    paired_write = _first_event(
        events,
        "experiments.analyze_final_model_engineering_paired_screen",
        "--output",
    )
    paired_verify = _first_event(
        events,
        "experiments.final_model_post_training_closure_preflight",
        "--verify-paired-screen-output",
    )
    f1_run = _first_event(
        events,
        "analysis.run_final_qfg_six_mode_audit",
        "--run",
    )
    f1_publish = _first_event(
        events,
        "experiments.final_model_post_training_closure_preflight",
        "--publish-f1-staging",
    )
    deep_write = _first_event(
        events,
        "analysis.verify_final_qfg_six_mode_audit_deep",
        "--write-once",
    )
    gate_write = _first_event(
        events,
        "experiments.adjudicate_final_model_engineering_gate",
        "--output",
    )
    closure = _first_event(
        events,
        "experiments.final_model_post_training_closure_preflight",
        "--finalize-closure",
    )
    assert int(summary_end["monotonic_ns"]) < min(
        int(item["monotonic_ns"]) for item in execute_starts
    )
    assert max(
        int(item["monotonic_ns"]) for item in execute_ends
    ) < int(finalize_start["monotonic_ns"])
    assert (
        int(finalize_start["monotonic_ns"])
        < int(verify_start["monotonic_ns"])
        < int(paired_write["monotonic_ns"])
        < int(paired_verify["monotonic_ns"])
        < int(f1_run["monotonic_ns"])
        < int(f1_publish["monotonic_ns"])
        < int(deep_write["monotonic_ns"])
        < int(gate_write["monotonic_ns"])
        < int(closure["monotonic_ns"])
    )
    assert f1_run["cuda_visible_devices"] == GPU3_UUID
    assert f1_run["physical_gpu_index"] == "3"
    assert f1_run["physical_gpu_uuid"] == GPU3_UUID
    staged_output = Path(
        f1_run["arguments"][
            f1_run["arguments"].index("--output-dir") + 1
        ]
    )
    assert staged_output.name == "payload"
    assert staged_output.parent.name.startswith(
        ".f1.closure-staging."
    )
    assert Path(
        f1_publish["arguments"][
            f1_publish["arguments"].index("--f1-staging-dir") + 1
        ]
    ) == staged_output

    first_event_count = len(events)
    second = subprocess.run(
        ["bash", str(SHELL), "--run", "--f1-gpu", "3"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert second.returncode == 0, second.stderr
    reentry_events = _events(log)[first_event_count:]
    assert any(
        item["module"] == evaluator_module
        and "--execute" in item["arguments"]
        for item in reentry_events
    )
    assert any(
        item["module"] == "analysis.run_final_qfg_six_mode_audit"
        and "--verify" in item["arguments"]
        for item in reentry_events
    )
    assert not any(
        item["module"] == "analysis.run_final_qfg_six_mode_audit"
        and "--run" in item["arguments"]
        for item in reentry_events
    )
    assert not any(
        item["module"]
        == "experiments.final_model_post_training_closure_preflight"
        and "--publish-f1-staging" in item["arguments"]
        for item in reentry_events
    )
    assert any(
        item["module"]
        == "analysis.verify_final_qfg_six_mode_audit_deep"
        and "--verify-attestation" in item["arguments"]
        for item in reentry_events
    )
    assert not any(
        item["module"]
        == "analysis.verify_final_qfg_six_mode_audit_deep"
        and "--write-once" in item["arguments"]
        for item in reentry_events
    )
    assert not any(
        item["module"]
        == "experiments.analyze_final_model_engineering_paired_screen"
        for item in reentry_events
    )
    assert any(
        item["module"]
        == "experiments.final_model_post_training_closure_preflight"
        and "--verify-paired-screen-output" in item["arguments"]
        for item in reentry_events
    )
    assert not any(
        item["module"] == "experiments.adjudicate_final_model_engineering_gate"
        for item in reentry_events
    )
    assert any(
        item["module"]
        == "experiments.final_model_post_training_closure_preflight"
        and "--verify-gate-output" in item["arguments"]
        for item in reentry_events
    )


def test_parallel_failure_terminates_peer_and_stops_later_stages(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "fake-python"
    _write_fake_python(fake_python)
    log = tmp_path / "calls.jsonl"
    terminated = tmp_path / "terminated-d.json"
    environment = _closure_environment(tmp_path, python=fake_python)
    environment.update(
        {
            "FAKE_CALL_LOG": str(log),
            "FAKE_FAIL_ARM": "b",
            "FAKE_FAIL_CODE": "17",
            "FAKE_FAIL_DELAY": "0.2",
            "FAKE_SLEEP_ARM": "d",
            "FAKE_SLEEP_SECONDS": "10",
            "FAKE_TERMINATION_MARKER": str(terminated),
        }
    )
    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(SHELL), "--run"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 17
    assert elapsed < 5
    assert "arm b sweep process failed" in result.stderr
    assert terminated.is_file()
    events = _events(log)
    assert any(
        item["event"] == "terminated" and item["arm"] == "d"
        for item in events
    )
    evaluator_module = (
        "experiments.evaluate_final_model_engineering_replication_pd_fa"
    )
    assert not any(
        item["module"] == evaluator_module
        and "--finalize-manifest" in item["arguments"]
        for item in events
    )
    assert not any(
        item["module"] == "analysis.run_final_qfg_six_mode_audit"
        for item in events
    )
