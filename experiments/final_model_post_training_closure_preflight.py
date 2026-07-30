#!/usr/bin/env python3
"""Preflight, runtime checks, and final attestation for GPU2/3 closure."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import run_final_qfg_six_mode_audit as f1_runner  # noqa: E402
from analysis import verify_final_qfg_six_mode_audit_deep as deep_verifier  # noqa: E402
from experiments import adjudicate_final_model_engineering_gate as gate_core  # noqa: E402
from experiments import analyze_final_model_engineering_paired_screen as paired_screen  # noqa: E402
from experiments import evaluate_final_model_engineering_replication_pd_fa as evaluator  # noqa: E402
from experiments import final_model_replication_exact_core as replication_core  # noqa: E402
from experiments import freeze_final_model_certification_source_lock as source_lock_core  # noqa: E402
from experiments import prepare_final_model_engineering_replication as prepare  # noqa: E402
from experiments import summarize_final_model_engineering_replication as summary_core  # noqa: E402
from experiments import watch_final_model_engineering_replication as watcher  # noqa: E402


SCHEMA = "sctransnet_final_model_post_training_closure_contract_v1"
ATTESTATION_SCHEMA = (
    "sctransnet_final_model_post_training_closure_attestation_v1"
)
ACTION_SCHEMA = "sctransnet_final_model_post_training_closure_action_v1"
SHELL_PATH = REPO_ROOT / "experiments/run_final_model_post_training_closure_2x5090.sh"
DEFAULT_OUTPUT_ROOT = watcher.DEFAULT_OUTPUT_ROOT
DEFAULT_SUMMARY = (
    DEFAULT_OUTPUT_ROOT / "engineering_replication_summary_v1.json"
)
DEFAULT_GATE_OUTPUT = (
    DEFAULT_OUTPUT_ROOT / "engineering_gate_adjudication_v1.json"
)
DEFAULT_PAIRED_SCREEN_OUTPUT = paired_screen.DEFAULT_OUTPUT
DEFAULT_CLOSURE_OUTPUT = (
    DEFAULT_OUTPUT_ROOT / "post_training_closure_attestation_v1.json"
)
GPU_BINDINGS = {
    2: "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    3: "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
STAGES = (
    "four_run_summary",
    "parallel_arm_b_gpu2_and_arm_d_gpu3_sweeps",
    "finalize_eight_result_manifest",
    "verify_eight_result_manifest",
    "cpu_engineering_paired_screen",
    "f1_six_mode_audit",
    "cpu_deep_verification",
    "engineering_gate_adjudication",
    "closure_attestation",
)


class PostTrainingClosureError(ValueError):
    """A post-training closure input or artifact differs from its contract."""


def _fail(message: str) -> None:
    raise PostTrainingClosureError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: expected={expected!r}, observed={observed!r}"
        )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink():
        _fail(f"{label} must not be a symlink: {value}")
    try:
        metadata = value.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {value}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file: {value}")
    return value.resolve()


def _sha256_file(path: Path, label: str) -> str:
    value = _regular_file(path, label)
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = _regular_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostTrainingClosureError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    _require(isinstance(value, dict), f"{label} must contain one object")
    return value, raw


def _source_bindings(repo_root: Path) -> dict[str, dict[str, str]]:
    root = Path(repo_root).resolve()
    sources = {
        "closure_shell": SHELL_PATH.resolve(),
        "closure_helper": Path(__file__).resolve(),
        "checkpoint_evaluator": Path(evaluator.__file__).resolve(),
        "f1_six_mode_runner": Path(f1_runner.__file__).resolve(),
        "deep_verifier": Path(deep_verifier.__file__).resolve(),
        "gate_adjudicator": Path(gate_core.__file__).resolve(),
        "paired_screen_analyzer": Path(paired_screen.__file__).resolve(),
        "four_run_summarizer": Path(summary_core.__file__).resolve(),
    }
    result: dict[str, dict[str, str]] = {}
    for role, path in sources.items():
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise PostTrainingClosureError(
                f"{role} source lies outside repository"
            ) from exc
        result[role] = {
            "path": relative,
            "sha256": _sha256_file(path, role),
        }
    return result


def _registered_gpu_bindings() -> dict[int, str]:
    observed: dict[int, str] = {}
    for index, expected in GPU_BINDINGS.items():
        values = {
            replication_core.arm_definition(arm)
            .trainer.PHYSICAL_GPU_UUIDS[str(index)]
            for arm in replication_core.SUPPORTED_ARMS
        }
        _equal(f"GPU{index} B/D UUID registry size", len(values), 1)
        value = next(iter(values))
        _equal(f"GPU{index} physical UUID", value, expected)
        observed[index] = value
    _require(
        len(set(observed.values())) == 2,
        "GPU2 and GPU3 must have distinct physical UUIDs",
    )
    return observed


def preflight(
    *,
    repo_root: Path = REPO_ROOT,
    source_lock_path: Path = source_lock_core.DEFAULT_OUTPUT,
    f1_gpu_index: int = 2,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    _require(f1_gpu_index in GPU_BINDINGS, "F1 GPU index must be 2 or 3")
    source_payload = source_lock_core.verify_source_lock(
        source_lock_path,
        repo_root=root,
    )
    contract = evaluator.evaluator_contract()
    _equal("evaluator sweep count", contract["expected_sweep_count"], 8)
    _equal("evaluator arms", contract["arms"], ["b", "d"])
    _equal(
        "evaluator trajectory-seed count",
        len(contract["trajectory_seeds"]),
        2,
    )
    _equal(
        "evaluator completed-result policy",
        contract["completed_result_policy"],
        "validate_then_skip",
    )
    _equal(
        "evaluator final-manifest requirement",
        contract["final_manifest_requires_all_eight_results"],
        True,
    )
    gpu_bindings = _registered_gpu_bindings()
    return {
        "schema": SCHEMA,
        "status": "ready",
        "action": "preflight",
        "stage_order": list(STAGES),
        "run_matrix": {
            "run_count": 4,
            "trajectory_seed_count": 2,
            "arms": ["b", "d"],
            "checkpoint_count_per_run": 2,
            "sweep_count": 8,
            "arm_b": {
                "physical_gpu_index": 2,
                "physical_gpu_uuid": gpu_bindings[2],
                "sweep_count": 4,
            },
            "arm_d": {
                "physical_gpu_index": 3,
                "physical_gpu_uuid": gpu_bindings[3],
                "sweep_count": 4,
            },
        },
        "f1_assignment": {
            "physical_gpu_index": f1_gpu_index,
            "physical_gpu_uuid": gpu_bindings[f1_gpu_index],
        },
        "source_lock": {
            "path": str(Path(source_lock_path).resolve()),
            "sha256": _sha256_file(
                Path(source_lock_path),
                "certification source lock",
            ),
            "schema": source_payload["schema"],
            "source_count": source_payload["source_count"],
        },
        "source_bindings": _source_bindings(root),
        "reentry_policy": {
            "sweeps": "validate_then_skip",
            "summary": "validate_identical_then_skip",
            "eight_result_manifest": "validate_identical_then_skip",
            "paired_screen": "recompute_validate_existing_else_create_once",
            "f1": (
                "verify_existing_else_stage_verify_and_atomic_"
                "directory_publish_noreplace"
            ),
            "deep_verification": "verify_existing_else_create_once",
            "gate": "recompute_validate_existing_else_create_once",
            "closure_attestation": "validate_identical_then_skip",
        },
        "parallel_failure_policy": "first_nonzero_terminates_peer",
        "official_test_accessed": False,
        "gpu_queried": False,
        "writes_performed": False,
    }


def assert_runtime_gpu(physical_gpu_index: int) -> dict[str, Any]:
    _require(
        physical_gpu_index in GPU_BINDINGS,
        "runtime physical GPU index must be 2 or 3",
    )
    expected_uuid = _registered_gpu_bindings()[physical_gpu_index]
    _equal(
        "CUDA_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        expected_uuid,
    )
    _equal(
        evaluator.EVALUATION_PHYSICAL_GPU_INDEX_ENV,
        os.environ.get(evaluator.EVALUATION_PHYSICAL_GPU_INDEX_ENV),
        str(physical_gpu_index),
    )
    _equal(
        evaluator.EVALUATION_PHYSICAL_GPU_UUID_ENV,
        os.environ.get(evaluator.EVALUATION_PHYSICAL_GPU_UUID_ENV),
        expected_uuid,
    )
    torch = evaluator.sweep_core.torch
    _require(torch.cuda.is_available(), "runtime CUDA is unavailable")
    _equal("visible CUDA device count", torch.cuda.device_count(), 1)
    return {
        "schema": SCHEMA,
        "status": "ready",
        "action": "assert-runtime-gpu",
        "device": "cuda:0",
        "physical_gpu_index": physical_gpu_index,
        "physical_gpu_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "visible_cuda_device_count": 1,
        "device_name": torch.cuda.get_device_name(0),
        "writes_performed": False,
    }


def _summary_path(output_root: Path, summary_path: Path | None) -> Path:
    return (
        Path(summary_path)
        if summary_path is not None
        else Path(output_root) / "engineering_replication_summary_v1.json"
    )


def _gate_output_path(output_root: Path, gate_output: Path | None) -> Path:
    return (
        Path(gate_output)
        if gate_output is not None
        else Path(output_root) / "engineering_gate_adjudication_v1.json"
    )


def _regular_directory(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink():
        _fail(f"{label} must not be a symlink: {value}")
    try:
        metadata = value.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {value}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} must be a directory: {value}")
    return value.resolve()


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without replacing any destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(
        renameat2 is not None,
        "atomic F1 directory publication requires renameat2",
    )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,  # AT_FDCWD
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            f"refusing to replace existing F1 output: {destination}"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(source),
        str(destination),
    )


def publish_f1_staging_directory(
    *,
    staging_output_dir: Path,
    final_output_dir: Path,
    repo_root: Path = REPO_ROOT,
    source_lock_path: Path = source_lock_core.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Verify and atomically publish a complete F1 directory once."""

    root = Path(repo_root).resolve()
    staging_requested = Path(staging_output_dir).expanduser()
    staging = _regular_directory(
        staging_requested,
        "F1 staging output",
    )
    container = _regular_directory(
        staging_requested.parent,
        "F1 staging container",
    )
    final_requested = Path(final_output_dir).expanduser()
    if final_requested.is_symlink() or os.path.lexists(final_requested):
        raise FileExistsError(
            f"refusing to replace existing F1 output: {final_requested}"
        )
    final_parent = _regular_directory(
        final_requested.parent,
        "F1 final-output parent",
    )
    final = final_parent / final_requested.name
    _equal("F1 staging payload name", staging.name, "payload")
    _equal(
        "F1 staging container parent",
        container.parent.resolve(),
        final_parent,
    )
    _require(
        container.name.startswith(
            f".{final.name}.closure-staging."
        ),
        "F1 staging container name is outside the closure namespace",
    )
    _equal(
        "F1 staging/final filesystem",
        staging.stat().st_dev,
        final_parent.stat().st_dev,
    )
    staged_report = staging / f1_runner.REPORT_FILENAME
    f1_runner.verify_audit_report(
        staged_report,
        repo_root=root,
        source_lock=source_lock_path,
    )
    _rename_directory_noreplace(staging, final)
    published = _regular_directory(final, "published F1 output")
    _equal("published F1 output path", published, final)
    final_report = published / f1_runner.REPORT_FILENAME
    f1_runner.verify_audit_report(
        final_report,
        repo_root=root,
        source_lock=source_lock_path,
    )
    _require(
        not os.path.lexists(staging),
        "F1 staging payload still exists after atomic publication",
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "publish-f1-staging",
        "atomic_directory_publish": True,
        "rename_noreplace": True,
        "overwrite_forbidden": True,
        "staging_container": str(container),
        "final_output_dir": str(published),
        "report": str(final_report),
        "report_sha256": _sha256_file(
            final_report,
            "published F1 report",
        ),
        "gpu_used": False,
    }


def verify_gate_output(
    *,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
    summary_path: Path,
    gate_output: Path,
) -> dict[str, Any]:
    expected = gate_core.adjudicate(
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
        summary_path=summary_path,
    )
    _equal("Gate status", expected.get("status"), "complete")
    _equal(
        "Gate decision",
        expected.get("decision"),
        "ENGINEERING_GATE_S_E_PASS",
    )
    observed, raw = _load_json(gate_output, "engineering Gate output")
    _equal(
        "engineering Gate canonical bytes",
        raw,
        gate_core.canonical_json_bytes(observed),
    )
    _equal(
        "stored/recomputed engineering Gate",
        gate_core.canonical_json_bytes(observed),
        gate_core.canonical_json_bytes(expected),
    )
    gates = observed.get("gates")
    _require(isinstance(gates, Mapping), "Gate gates must be an object")
    m_train = gates.get("M-train")
    _require(
        isinstance(m_train, Mapping),
        "Gate M-train must be an object",
    )
    _equal(
        "Gate M-train status",
        m_train.get("status"),
        "insufficient_evidence",
    )
    _equal("Gate M-train passed", m_train.get("passed"), None)
    claim_boundary = observed.get("claim_boundary")
    _require(
        isinstance(claim_boundary, Mapping),
        "Gate claim boundary must be an object",
    )
    _equal(
        "Gate paper-core claim boundary",
        claim_boundary.get("paper_core_established"),
        False,
    )
    _equal(
        "Gate stability claim boundary",
        claim_boundary.get("stability_claim_supported"),
        False,
    )
    return {
        "status": "verified_complete",
        "decision": observed["decision"],
        "gate_m_train_status": "insufficient_evidence",
        "gate_m_train_passed": None,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "path": str(Path(gate_output).resolve()),
        "sha256": _sha256_file(Path(gate_output), "engineering Gate output"),
    }


def _verify_summary(
    *,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
    summary_path: Path,
) -> dict[str, Any]:
    expected = summary_core.build_summary(
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
    )
    observed, raw = _load_json(summary_path, "four-run summary")
    _equal(
        "four-run summary canonical bytes",
        raw,
        summary_core.canonical_json_bytes(observed),
    )
    _equal(
        "stored/rebuilt four-run summary",
        summary_core.canonical_json_bytes(observed),
        summary_core.canonical_json_bytes(expected),
    )
    _equal("four-run summary status", observed.get("status"), "complete")
    _equal("four-run summary run count", observed.get("run_count"), 4)
    return {
        "path": str(Path(summary_path).resolve()),
        "sha256": _sha256_file(Path(summary_path), "four-run summary"),
        "run_count": 4,
    }


def _verify_eight_result_manifest(
    *,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
) -> dict[str, Any]:
    requests = evaluator.collect_evaluation_requests(
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
    )
    expected = evaluator.build_results_manifest(requests)
    manifest_path = evaluator.default_manifest_path(output_root)
    observed, raw = _load_json(manifest_path, "eight-result manifest")
    _equal(
        "eight-result canonical bytes",
        raw,
        evaluator._manifest_json_bytes(observed),
    )
    _equal(
        "stored/rebuilt eight-result manifest",
        evaluator._manifest_json_bytes(observed),
        evaluator._manifest_json_bytes(expected),
    )
    _equal("eight-result count", observed.get("result_count"), 8)
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256_file(manifest_path, "eight-result manifest"),
        "result_count": 8,
    }


def verify_paired_screen_output(
    *,
    output_root: Path,
    paired_output: Path,
) -> dict[str, Any]:
    manifest_path = evaluator.default_manifest_path(output_root)
    expected = paired_screen.analyze(manifest_path=manifest_path)
    _equal("paired-screen status", expected.get("status"), "complete")
    observed, raw = _load_json(
        paired_output,
        "engineering paired-screen output",
    )
    _equal(
        "engineering paired-screen canonical bytes",
        raw,
        paired_screen.canonical_json_bytes(observed),
    )
    _equal(
        "stored/recomputed engineering paired screen",
        paired_screen.canonical_json_bytes(observed),
        paired_screen.canonical_json_bytes(expected),
    )
    _equal(
        "paired-screen manifest result count",
        observed.get("manifest", {}).get("result_count"),
        8,
    )
    _equal(
        "paired-screen establishes Gate M-train",
        observed.get("establishes_gate_m_train"),
        False,
    )
    gates = observed.get("gates")
    _require(
        isinstance(gates, Mapping),
        "paired-screen gates must be an object",
    )
    m_train = gates.get("M-train")
    _require(
        isinstance(m_train, Mapping),
        "paired-screen Gate M-train must be an object",
    )
    _equal(
        "paired-screen Gate M-train status",
        m_train.get("status"),
        "insufficient_evidence",
    )
    _equal(
        "paired-screen Gate M-train passed",
        m_train.get("passed"),
        None,
    )
    _equal(
        "paired-screen Gate M-train establishment",
        m_train.get("establishes_gate_m_train"),
        False,
    )
    route_met = observed.get("engineering_paired_route_met")
    _require(
        isinstance(route_met, bool),
        "paired-screen engineering route must be Boolean",
    )
    decision = observed.get("decision")
    _equal(
        "paired-screen decision",
        decision,
        (
            "ENGINEERING_PAIRED_SCREEN_ROUTE_MET"
            if route_met
            else "ENGINEERING_PAIRED_SCREEN_ROUTE_NOT_MET"
        ),
    )
    return {
        "status": "verified_complete",
        "decision": decision,
        "engineering_paired_route_met": route_met,
        "establishes_gate_m_train": False,
        "gate_m_train_status": "insufficient_evidence",
        "gate_m_train_passed": None,
        "path": str(Path(paired_output).resolve()),
        "sha256": _sha256_file(
            Path(paired_output),
            "engineering paired-screen output",
        ),
    }


def _strict_boolean(value: Any, label: str) -> bool:
    _require(type(value) is bool, f"{label} must be Boolean")
    return value


def _verify_f1_functional_activation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    gate = report.get("functional_gate")
    _require(
        isinstance(gate, Mapping),
        "F1 functional gate must be an object",
    )
    _equal("F1 functional gate status", gate.get("status"), "complete")
    repeat_equivalent = _strict_boolean(
        gate.get("repeat_inference_equivalent"),
        "F1 repeat-inference equivalence",
    )
    output_different = _strict_boolean(
        gate.get("full_vs_qfg_off_functionally_different"),
        "F1 full-vs-QFG-off output-difference flag",
    )
    nontrivial_factor = _strict_boolean(
        gate.get("nontrivial_factor_use"),
        "F1 nontrivial-factor-use flag",
    )
    active = _strict_boolean(
        gate.get("qfg_functionally_active"),
        "F1 QFG functionally-active flag",
    )
    _equal(
        "F1 QFG functional-activation conjunction",
        active,
        repeat_equivalent and output_different and nontrivial_factor,
    )
    repeat = report.get("repeat_inference")
    _require(
        isinstance(repeat, Mapping),
        "F1 repeat-inference record must be an object",
    )
    _equal(
        "F1 gate/repeat-inference equivalence",
        repeat_equivalent,
        _strict_boolean(
            repeat.get("equivalent"),
            "F1 repeat-inference source equivalence",
        ),
    )
    modes = report.get("modes")
    _require(isinstance(modes, Mapping), "F1 modes must be an object")
    qfg_off = modes.get("qfg_off")
    full = modes.get("full")
    _require(
        isinstance(qfg_off, Mapping),
        "F1 qfg_off mode must be an object",
    )
    _require(
        isinstance(full, Mapping),
        "F1 full mode must be an object",
    )
    try:
        source_output_different = qfg_off["comparison_to_full"][
            "output_difference"
        ]["functionally_different"]
        source_nontrivial_factor = full["factor_summary"][
            "nontrivial_factor_use"
        ]
    except (KeyError, TypeError) as exc:
        raise PostTrainingClosureError(
            "F1 functional-activation source fields are incomplete"
        ) from exc
    _equal(
        "F1 gate/output-difference source flag",
        output_different,
        _strict_boolean(
            source_output_different,
            "F1 output-difference source flag",
        ),
    )
    _equal(
        "F1 gate/nontrivial-factor source flag",
        nontrivial_factor,
        _strict_boolean(
            source_nontrivial_factor,
            "F1 nontrivial-factor source flag",
        ),
    )
    causal = _strict_boolean(
        gate.get("performance_causal_claim_established"),
        "F1 performance-causal-claim flag",
    )
    _equal("F1 performance-causal-claim boundary", causal, False)
    return {
        "status": "verified",
        "qfg_functionally_active": active,
        "logical_inputs": {
            "repeat_inference_equivalent": repeat_equivalent,
            "full_vs_qfg_off_functionally_different": output_different,
            "nontrivial_factor_use": nontrivial_factor,
        },
        "logical_rule": "all_three_inputs_must_be_true",
        "performance_causal_claim_established": False,
    }


def _deep_verification_summary(
    deep: Mapping[str, Any],
) -> dict[str, Any]:
    no_invention = _strict_boolean(
        deep.get("no_invention_status"),
        "deep-verifier no-invention status",
    )
    _equal("deep-verifier no-invention status", no_invention, True)
    limitations = deep.get("limitations")
    _require(
        isinstance(limitations, list),
        "deep-verifier limitations must be an array",
    )
    for index, record in enumerate(limitations):
        _require(
            isinstance(record, Mapping),
            f"deep-verifier limitation {index} must be an object",
        )
        for key in ("field", "reason"):
            _require(
                isinstance(record.get(key), str) and bool(record[key]),
                f"deep-verifier limitation {index} {key} is invalid",
            )
    return {
        "status": "verified",
        "no_invention_status": True,
        "limitations_count": len(limitations),
    }


def _artifact_record(path: Path, label: str) -> dict[str, str]:
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256_file(Path(path), label),
    }


def build_closure_attestation(
    *,
    repo_root: Path,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
    summary_path: Path,
    paired_output: Path,
    f1_report: Path,
    deep_output: Path,
    gate_output: Path,
    f1_gpu_index: int,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    contract = preflight(
        repo_root=root,
        source_lock_path=source_lock_path,
        f1_gpu_index=f1_gpu_index,
    )
    summary = _verify_summary(
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
        summary_path=summary_path,
    )
    result_manifest = _verify_eight_result_manifest(
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
    )
    paired = verify_paired_screen_output(
        output_root=output_root,
        paired_output=paired_output,
    )
    f1 = f1_runner.verify_audit_report(
        f1_report,
        repo_root=root,
        source_lock=source_lock_path,
    )
    _equal("F1 audit status", f1.get("status"), "complete")
    functional_activation = _verify_f1_functional_activation(f1)
    deep = deep_verifier.verify_deep_verification(
        deep_output,
        f1_report,
        repo_root=root,
        source_lock=source_lock_path,
    )
    _equal("deep-verification status", deep.get("status"), "verified")
    deep_summary = _deep_verification_summary(deep)
    gate = verify_gate_output(
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
        summary_path=summary_path,
        gate_output=gate_output,
    )
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "complete",
        "scope": "post_training_internal_validation_closure",
        "stage_order": list(STAGES),
        "gpu_assignments": {
            "arm_b_sweeps": contract["run_matrix"]["arm_b"],
            "arm_d_sweeps": contract["run_matrix"]["arm_d"],
            "f1_six_mode_audit": contract["f1_assignment"],
        },
        "artifacts": {
            "four_run_summary": summary,
            "eight_result_manifest": result_manifest,
            "engineering_paired_screen": paired,
            "f1_six_mode_audit": {
                **_artifact_record(
                    f1_report,
                    "F1 six-mode audit",
                ),
                "functional_activation": functional_activation,
            },
            "f1_deep_verification": {
                **_artifact_record(
                    deep_output,
                    "F1 deep verification",
                ),
                **deep_summary,
            },
            "engineering_gate": gate,
        },
        "source_lock": contract["source_lock"],
        "source_bindings": contract["source_bindings"],
        "reentry_policy": contract["reentry_policy"],
        "parallel_failure_policy": contract["parallel_failure_policy"],
        "paired_screen_decision": paired["decision"],
        "engineering_paired_route_met": paired[
            "engineering_paired_route_met"
        ],
        "paired_screen_establishes_gate_m_train": False,
        "paired_screen_gate_m_train_status": paired[
            "gate_m_train_status"
        ],
        "paired_screen_gate_m_train_passed": paired[
            "gate_m_train_passed"
        ],
        "qfg_functional_activation": functional_activation,
        "deep_verification_summary": deep_summary,
        "gate_decision": gate["decision"],
        "engineering_gate_m_train_status": gate[
            "gate_m_train_status"
        ],
        "engineering_gate_m_train_passed": gate[
            "gate_m_train_passed"
        ],
        "paper_core_established": gate["paper_core_established"],
        "stability_claim_supported": gate[
            "stability_claim_supported"
        ],
        "official_test_accessed": False,
        "paper_core_established_changed": False,
        "stability_claim_supported_changed": False,
        "write_once": True,
        "overwrite_forbidden": True,
    }


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(f"closure attestation must not be a symlink: {destination}")
    content = canonical_json_bytes(payload)
    if destination.exists():
        stored = _regular_file(
            destination,
            "existing closure attestation",
        )
        _equal(
            "stored/rebuilt closure attestation",
            stored.read_bytes(),
            content,
        )
        return stored, "skipped_identical_complete"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            concurrent = _regular_file(
                destination,
                "concurrently created closure attestation",
            )
            _equal(
                "concurrent/rebuilt closure attestation",
                concurrent.read_bytes(),
                content,
            )
            return concurrent, "skipped_identical_complete"
    finally:
        temporary.unlink(missing_ok=True)
    written = _regular_file(destination, "written closure attestation")
    _equal("written closure attestation", written.read_bytes(), content)
    return written, "created"


def finalize_closure(
    *,
    repo_root: Path,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
    summary_path: Path,
    paired_output: Path,
    f1_report: Path,
    deep_output: Path,
    gate_output: Path,
    closure_output: Path,
    f1_gpu_index: int,
) -> dict[str, Any]:
    payload = build_closure_attestation(
        repo_root=repo_root,
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
        summary_path=summary_path,
        paired_output=paired_output,
        f1_report=f1_report,
        deep_output=deep_output,
        gate_output=gate_output,
        f1_gpu_index=f1_gpu_index,
    )
    path, action = _write_or_validate(closure_output, payload)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "finalize-closure",
        "attestation_action": action,
        "attestation_path": str(path),
        "attestation_sha256": _sha256_file(
            path,
            "closure attestation",
        ),
        "gate_decision": payload["gate_decision"],
        "qfg_functionally_active": payload[
            "qfg_functional_activation"
        ]["qfg_functionally_active"],
        "functional_activation_logical_inputs": payload[
            "qfg_functional_activation"
        ]["logical_inputs"],
        "functional_activation_logical_rule": payload[
            "qfg_functional_activation"
        ]["logical_rule"],
        "performance_causal_claim_established": payload[
            "qfg_functional_activation"
        ]["performance_causal_claim_established"],
        "deep_no_invention_status": payload[
            "deep_verification_summary"
        ]["no_invention_status"],
        "deep_limitations_count": payload[
            "deep_verification_summary"
        ]["limitations_count"],
        "engineering_gate_m_train_status": payload[
            "engineering_gate_m_train_status"
        ],
        "engineering_gate_m_train_passed": payload[
            "engineering_gate_m_train_passed"
        ],
        "paper_core_established": payload["paper_core_established"],
        "stability_claim_supported": payload[
            "stability_claim_supported"
        ],
        "gpu_queried": False,
        "official_test_accessed": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--assert-runtime-gpu", action="store_true")
    action.add_argument("--publish-f1-staging", action="store_true")
    action.add_argument("--verify-paired-screen-output", action="store_true")
    action.add_argument("--verify-gate-output", action="store_true")
    action.add_argument("--finalize-closure", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=source_lock_core.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=prepare.DEFAULT_SEED_CONTRACT,
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=prepare.DEFAULT_MANIFEST_DIRECTORY,
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=DEFAULT_PAIRED_SCREEN_OUTPUT,
    )
    parser.add_argument("--f1-report", type=Path, default=f1_runner.DEFAULT_OUTPUT_DIR / f1_runner.REPORT_FILENAME)
    parser.add_argument("--deep-output", type=Path, default=deep_verifier.DEFAULT_OUTPUT)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--closure-output", type=Path)
    parser.add_argument("--f1-gpu-index", type=int, choices=(2, 3), default=2)
    parser.add_argument("--physical-gpu-index", type=int, choices=(2, 3))
    parser.add_argument("--f1-staging-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary_path = _summary_path(args.output_root, args.summary)
    gate_output = _gate_output_path(args.output_root, args.gate_output)
    closure_output = (
        args.closure_output
        if args.closure_output is not None
        else args.output_root / "post_training_closure_attestation_v1.json"
    )
    if args.preflight:
        payload = preflight(
            repo_root=args.repo_root,
            source_lock_path=args.source_lock,
            f1_gpu_index=args.f1_gpu_index,
        )
    elif args.assert_runtime_gpu:
        _require(
            args.physical_gpu_index is not None,
            "--assert-runtime-gpu requires --physical-gpu-index",
        )
        payload = assert_runtime_gpu(args.physical_gpu_index)
    elif args.publish_f1_staging:
        _require(
            args.f1_staging_dir is not None,
            "--publish-f1-staging requires --f1-staging-dir",
        )
        _equal(
            "F1 final report filename",
            args.f1_report.name,
            f1_runner.REPORT_FILENAME,
        )
        payload = publish_f1_staging_directory(
            staging_output_dir=args.f1_staging_dir,
            final_output_dir=args.f1_report.parent,
            repo_root=args.repo_root,
            source_lock_path=args.source_lock,
        )
    elif args.verify_paired_screen_output:
        payload = verify_paired_screen_output(
            output_root=args.output_root,
            paired_output=args.paired_output,
        )
    elif args.verify_gate_output:
        payload = verify_gate_output(
            output_root=args.output_root,
            source_lock_path=args.source_lock,
            seed_contract_path=args.seed_contract,
            manifest_directory=args.manifest_directory,
            summary_path=summary_path,
            gate_output=gate_output,
        )
    else:
        payload = finalize_closure(
            repo_root=args.repo_root,
            output_root=args.output_root,
            source_lock_path=args.source_lock,
            seed_contract_path=args.seed_contract,
            manifest_directory=args.manifest_directory,
            summary_path=summary_path,
            paired_output=args.paired_output,
            f1_report=args.f1_report,
            deep_output=args.deep_output,
            gate_output=gate_output,
            closure_output=closure_output,
            f1_gpu_index=args.f1_gpu_index,
        )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


__all__ = [
    "ACTION_SCHEMA",
    "ATTESTATION_SCHEMA",
    "DEFAULT_CLOSURE_OUTPUT",
    "DEFAULT_GATE_OUTPUT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PAIRED_SCREEN_OUTPUT",
    "DEFAULT_SUMMARY",
    "GPU_BINDINGS",
    "PostTrainingClosureError",
    "SCHEMA",
    "STAGES",
    "assert_runtime_gpu",
    "build_closure_attestation",
    "canonical_json_bytes",
    "finalize_closure",
    "main",
    "preflight",
    "publish_f1_staging_directory",
    "verify_gate_output",
    "verify_paired_screen_output",
]


if __name__ == "__main__":
    main()
