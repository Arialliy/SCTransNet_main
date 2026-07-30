#!/usr/bin/env python3
"""Top-level completion contract for the fixed-seed-42 final-model closure.

This module joins three deliberately different evidence subjects without
conflating them:

1. the *new* seed-42 B/D formal-800 replay and its four checkpoint-local
   sweeps/Gate;
2. the already frozen deployment-D artifact and its F1 six-mode QFG audit;
3. the independent CPU deep verification of that F1 artifact set.

Read-only status and dry-run modes never initialize CUDA.  Formal execution is
owned by ``run_final_model_seed42_certification_completion.sh``.  The final
attestation is canonical, write-once, idempotently verifiable, and keeps the
single-seed claim boundary explicit.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import run_final_qfg_six_mode_audit as f1_runner  # noqa: E402
from analysis import (  # noqa: E402
    verify_final_qfg_six_mode_audit_deep as deep_verifier,
)
from experiments import (  # noqa: E402
    final_model_seed42_certification_replay_contract as replay_contract,
)
from experiments import (  # noqa: E402
    final_model_seed42_certification_replay_exact_core as replay_core,
)
from experiments import (  # noqa: E402
    final_model_seed42_certification_replay_posttraining as posttraining,
)
from experiments import (  # noqa: E402
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (  # noqa: E402
    freeze_final_model_certification_source_lock as certification_source_lock,
)
from experiments import (  # noqa: E402
    freeze_final_model_seed42_certification_completion_source_lock
    as completion_source_lock,
)


SCHEMA = "sctransnet_final_model_seed42_certification_completion_v1"
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_action_v1"
)
ATTESTATION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_attestation_v1"
)
SCOPE = "new_seed42_replay_plus_frozen_deployment_qfg_audit"
WAITING_EXIT_CODE = 3
RESUME_NEEDED_EXIT_CODE = 4
FIXED_THRESHOLD = 0.5
TRAJECTORY_SEED = 42
EXCLUDED_REPLAY_SEEDS = (3407, 426780603)

SHELL_PATH = (
    REPO_ROOT
    / "experiments/run_final_model_seed42_certification_completion.sh"
)
POSTTRAINING_SHELL_PATH = (
    REPO_ROOT
    / "experiments/"
    "run_final_model_seed42_certification_replay_posttraining_2x5090.sh"
)
DEFAULT_F1_REPORT = (
    f1_runner.DEFAULT_OUTPUT_DIR / f1_runner.REPORT_FILENAME
)
DEFAULT_DEEP_OUTPUT = deep_verifier.DEFAULT_OUTPUT
DEFAULT_ATTESTATION = (
    replay_contract.DEFAULT_OUTPUT_ROOT
    / "final_model_seed42_certification_completion_attestation_v1.json"
)
TRAINING_PAIR_LOCK = (
    replay_contract.DEFAULT_OUTPUT_ROOT
    / ".gpu23_seed42_certification_replay.lock"
)

GPU2_INDEX = 2
GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU_INDEX_ENV = "FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX"
GPU_UUID_ENV = "FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID"
TRAINING_LOCK_FD_ENV = "FINAL_MODEL_SEED42_COMPLETION_TRAINING_LOCK_FD"

STAGES = (
    "wait_for_new_seed42_b_d_formal800",
    "new_seed42_four_checkpoint_local_sweeps_and_gate",
    "frozen_deployment_d_f1_six_mode_on_gpu2",
    "f1_deep_verification_on_cpu",
    "final_completion_attestation",
)


class Seed42CertificationCompletionError(ValueError):
    """A completion input or artifact differs from the frozen contract."""


def _fail(message: str) -> None:
    raise Seed42CertificationCompletionError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: observed={observed!r}, expected={expected!r}"
        )


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be one finite number")
    return float(value)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        _fail(f"value is not canonical finite JSON: {exc}")


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
    source = _regular_file(path, label)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, label: str) -> dict[str, str]:
    source = _regular_file(path, label)
    return {
        "path": str(source),
        "sha256": _sha256_file(source, label),
    }


def _load_json(path: Path, label: str) -> tuple[Path, dict[str, Any], bytes]:
    source = _regular_file(path, label)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Seed42CertificationCompletionError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        _fail(f"{label} must contain one object")
    return source, payload, raw


def _canonical_object(path: Path, label: str) -> dict[str, Any]:
    _, payload, raw = _load_json(path, label)
    _equal(f"{label} canonical bytes", raw, canonical_json_bytes(payload))
    return payload


def _write_or_validate(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(f"final attestation must not be a symlink: {destination}")
    content = canonical_json_bytes(payload)
    if destination.exists():
        stored = _regular_file(destination, "existing final attestation")
        _equal("stored/live final attestation", stored.read_bytes(), content)
        return stored, "skipped_identical_complete"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
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
                "concurrently created final attestation",
            )
            _equal(
                "concurrent/live final attestation",
                concurrent.read_bytes(),
                content,
            )
            return concurrent, "skipped_identical_complete"
    finally:
        temporary.unlink(missing_ok=True)
    stored = _regular_file(destination, "written final attestation")
    _equal("written final attestation bytes", stored.read_bytes(), content)
    return stored, "created"


def _inside_new_replay(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    root = replay_contract.DEFAULT_OUTPUT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} lies outside the new replay root: {resolved}")
    text = resolved.as_posix()
    for marker in posttraining.LEGACY_RUN_DIRECTORY_MARKERS:
        if marker in text:
            _fail(f"{label} references a legacy stage directory: {marker}")
    for seed in EXCLUDED_REPLAY_SEEDS:
        if f"seed_{seed}_" in text or f"seed-{seed}:" in text:
            _fail(f"{label} references excluded replay seed {seed}")
    return resolved


def _portable_report_child(report: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} relative path is missing")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != relative
    ):
        _fail(f"{label} relative path is not canonical: {relative!r}")
    child = report.parent.joinpath(*pure.parts)
    source = _regular_file(child, label)
    if not source.is_relative_to(report.parent.resolve()):
        _fail(f"{label} escapes its report directory")
    return source


def _lightweight_run_progress(arm: str) -> dict[str, Any]:
    """Read progress without loading/hash-scanning the large exact checkpoint."""

    inputs = posttraining._inputs(  # noqa: SLF001 - contract adapter API
        arm,
        replay_contract_path=posttraining.DEFAULT_REPLAY_CONTRACT,
        manifest_directory=posttraining.DEFAULT_MANIFEST_DIRECTORY,
        replay_source_lock_path=posttraining.DEFAULT_REPLAY_SOURCE_LOCK,
        certification_source_lock_path=(
            posttraining.DEFAULT_CERTIFICATION_SOURCE_LOCK
        ),
        parent_lock_path=posttraining.DEFAULT_PARENT_LOCK,
    )
    run_directory = _inside_new_replay(
        replay_core.run_directory(inputs),
        f"arm {arm} run directory",
    )
    record: dict[str, Any] = {
        "arm": arm,
        "variant": inputs.definition.variant,
        "trajectory_seed": inputs.trajectory_seed,
        "run_id": replay_core.expected_run_id(inputs),
        "run_directory": str(run_directory),
        "completed_epoch": 0,
        "summary_present": False,
        "strict_formal800_complete": False,
    }
    if run_directory.is_symlink():
        _fail(f"arm {arm} run directory must not be a symlink")
    if not run_directory.exists():
        record["state"] = "not_started"
        return record
    if not run_directory.is_dir():
        _fail(f"arm {arm} run path is not a directory")

    # A newly created trainer may not yet have committed epoch one.  Protocol
    # and split are still required as soon as the run directory is visible.
    replay_core._validate_existing_protocol(  # noqa: SLF001
        inputs,
        run_directory,
    )
    replay_core._load_pretty_object(  # noqa: SLF001
        run_directory / "split.json",
        f"arm {arm} replay split",
    )
    journal_root = run_directory / "exact_journal"
    if journal_root.is_symlink():
        _fail(f"arm {arm} exact journal must not be a symlink")
    marker_path = journal_root / "active.json"
    if not marker_path.exists():
        if marker_path.is_symlink():
            _fail(f"arm {arm} exact marker must not be a symlink")
        record["state"] = "waiting_for_first_exact_epoch"
        return record
    _, marker, marker_raw = _load_json(
        marker_path,
        f"arm {arm} exact active marker",
    )
    if not marker_raw.endswith(b"\n"):
        _fail(f"arm {arm} exact marker is not newline terminated")
    epoch = marker.get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= replay_contract.FORMAL_EPOCHS
    ):
        _fail(f"arm {arm} exact marker epoch is outside 1..800")
    for key in ("checkpoint_file", "metrics_file"):
        filename = marker.get(key)
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            _fail(f"arm {arm} exact marker {key} is invalid")
        _regular_file(journal_root / filename, f"arm {arm} active {key}")
    record["completed_epoch"] = epoch
    record["exact_active_marker"] = {
        "path": str(marker_path.resolve()),
        "sha256": hashlib.sha256(marker_raw).hexdigest(),
    }
    summary_path = run_directory / "summary.json"
    if summary_path.is_symlink():
        _fail(f"arm {arm} completion summary must not be a symlink")
    record["summary_present"] = summary_path.is_file()
    if not record["summary_present"]:
        record["state"] = "training_or_finalizing"
        return record

    # A summary is accepted only after the full exact-journal, metrics,
    # checkpoint, protocol, split, and identity validation used by posttraining.
    requests = posttraining.preflight_completed_run(arm)
    _equal(f"arm {arm} request count", len(requests), 2)
    _equal(
        f"arm {arm} exact completion epoch",
        epoch,
        replay_contract.FORMAL_EPOCHS,
    )
    record["strict_formal800_complete"] = True
    record["state"] = "strict_formal800_complete"
    record["completion_summary"] = _artifact(
        summary_path,
        f"arm {arm} completion summary",
    )
    return record


def training_pair_lock_status() -> dict[str, Any]:
    """Report whether the replay pair launcher and all lock inheritors exited."""

    path = Path(TRAINING_PAIR_LOCK)
    if path.is_symlink():
        _fail(f"training pair lock must not be a symlink: {path}")
    if not path.exists():
        return {
            "path": str(path.resolve()),
            "state": "not_present",
            "released": True,
        }
    source = _regular_file(path, "training pair lock")
    inherited_fd = os.environ.get(TRAINING_LOCK_FD_ENV)
    if inherited_fd is not None:
        try:
            descriptor = int(inherited_fd)
        except ValueError:
            _fail(f"{TRAINING_LOCK_FD_ENV} must be an integer descriptor")
        if descriptor < 0:
            _fail(f"{TRAINING_LOCK_FD_ENV} must be non-negative")
        try:
            descriptor_stat = os.fstat(descriptor)
        except OSError as exc:
            _fail(f"{TRAINING_LOCK_FD_ENV} is not open: {exc}")
        source_stat = source.stat()
        _equal(
            "inherited training-lock device/inode",
            (descriptor_stat.st_dev, descriptor_stat.st_ino),
            (source_stat.st_dev, source_stat.st_ino),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("inherited completion descriptor does not own the pair lock")
        return {
            "path": str(source),
            "state": "held_exclusively_by_completion_runner",
            "released": True,
        }
    descriptor = os.open(source, os.O_RDWR)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        return {
            "path": str(source),
            "state": "released" if acquired else "held_by_training_pair",
            "released": acquired,
        }
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def training_completion_status() -> dict[str, Any]:
    runs = [
        _lightweight_run_progress(arm)
        for arm in replay_core.SUPPORTED_ARMS
    ]
    pair_lock = training_pair_lock_status()
    formal800_ready = all(
        record["strict_formal800_complete"] for record in runs
    )
    ready = formal800_ready and pair_lock["released"]
    if ready:
        # This cross-arm validation additionally rejects missing/duplicate
        # requests and confirms the four own-checkpoint evaluation plan.
        requests = posttraining.collect_requests()
        _equal("completed replay request count", len(requests), 4)
    return {
        "schema": ACTION_SCHEMA,
        "status": (
            "ready_for_posttraining"
            if ready
            else "waiting_for_new_seed42_formal800"
        ),
        "scope": SCOPE,
        "ready": ready,
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_replay_seeds": list(EXCLUDED_REPLAY_SEEDS),
        "old_seed42_stage_results_count_as_replay": False,
        "required_epoch": replay_contract.FORMAL_EPOCHS,
        "formal800_artifacts_ready": formal800_ready,
        "training_pair_lock": pair_lock,
        "training_pair_process_exited": pair_lock["released"],
        "run_count": len(runs),
        "runs": runs,
        "gpu_queried": False,
        "gpu_command_launched": False,
        "writes_performed": False,
    }


def _source_bindings() -> dict[str, dict[str, str]]:
    paths = {
        "completion_helper": Path(__file__).resolve(),
        "completion_shell": SHELL_PATH.resolve(),
        "seed42_posttraining_adapter": Path(posttraining.__file__).resolve(),
        "seed42_posttraining_shell": POSTTRAINING_SHELL_PATH.resolve(),
        "f1_six_mode_runner": Path(f1_runner.__file__).resolve(),
        "f1_deep_verifier": Path(deep_verifier.__file__).resolve(),
    }
    return {
        role: _artifact(path, f"{role} source")
        for role, path in paths.items()
    }


def verify_completion_source_lock() -> dict[str, Any]:
    payload = completion_source_lock.verify_source_lock(
        completion_source_lock.DEFAULT_OUTPUT,
        replay_source_lock_path=posttraining.DEFAULT_REPLAY_SOURCE_LOCK,
        certification_source_lock_path=(
            posttraining.DEFAULT_CERTIFICATION_SOURCE_LOCK
        ),
        parent_lock_path=posttraining.DEFAULT_PARENT_LOCK,
    )
    _equal("completion source-lock schema", payload.get("schema"), completion_source_lock.SCHEMA)
    _equal("completion source-lock status", payload.get("status"), "locked")
    frozen = payload.get("frozen_model")
    if not isinstance(frozen, Mapping):
        _fail("completion source-lock frozen model is missing")
    for label, observed, expected in (
        ("completion source-lock mainline changed", frozen.get("mainline_changed"), False),
        ("completion source-lock innovation changed", frozen.get("innovation_changed"), False),
        (
            "completion source-lock deployment weights changed",
            frozen.get("seed42_deployment_weights_changed"),
            False,
        ),
        ("completion source-lock threshold", frozen.get("default_threshold"), 0.5),
    ):
        _equal(label, observed, expected)
    return payload


def _artifact_presence(path: Path) -> str:
    value = Path(path)
    if value.is_symlink():
        return "invalid_symlink"
    if value.is_file():
        return "present_requires_live_verification"
    if value.exists():
        return "invalid_non_file"
    return "pending"


def dry_run_plan(
    *,
    f1_report: Path = DEFAULT_F1_REPORT,
    deep_output: Path = DEFAULT_DEEP_OUTPUT,
    attestation: Path = DEFAULT_ATTESTATION,
) -> dict[str, Any]:
    locked_sources = verify_completion_source_lock()
    training = training_completion_status()
    f1_preflight = f1_runner.preflight(
        REPO_ROOT,
        parent_lock.DEFAULT_OUTPUT,
        certification_source_lock.DEFAULT_OUTPUT,
    )
    _equal("F1 preflight GPU use", f1_preflight.get("gpu_used"), False)
    _equal("F1 preflight write status", f1_preflight.get("writes_performed"), False)
    return {
        "schema": SCHEMA,
        "status": (
            "ready_to_execute_completion"
            if training["ready"]
            else "waiting_for_new_seed42_training"
        ),
        "scope": SCOPE,
        "stage_order": list(STAGES),
        "completion_source_lock": {
            **_artifact(
                completion_source_lock.DEFAULT_OUTPUT,
                "completion source lock",
            ),
            "schema": locked_sources["schema"],
            "source_count": locked_sources["source_count"],
        },
        "training": training,
        "posttraining": {
            "launcher": str(POSTTRAINING_SHELL_PATH),
            "planned_sweep_count": 4,
            "checkpoint_policy": (
                "each_arm_own_best_miou_and_each_arm_own_best_pd"
            ),
            "physical_gpu_assignments": {"b": 2, "d": 3},
            "closure": {
                "path": str(posttraining.DEFAULT_CLOSURE.resolve()),
                "state": _artifact_presence(posttraining.DEFAULT_CLOSURE),
            },
        },
        "f1": {
            "subject": "frozen_deployment_d_not_replay_d",
            "physical_gpu_index": GPU2_INDEX,
            "physical_gpu_uuid": GPU2_UUID,
            "report": {
                "path": str(Path(f1_report).resolve()),
                "state": _artifact_presence(f1_report),
            },
            "preflight": f1_preflight,
        },
        "deep_verification": {
            "device": "cpu",
            "output": {
                "path": str(Path(deep_output).resolve()),
                "state": _artifact_presence(deep_output),
            },
        },
        "final_attestation": {
            "path": str(Path(attestation).resolve()),
            "state": _artifact_presence(attestation),
        },
        "fixed_threshold": FIXED_THRESHOLD,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "multiseed_replication_supported": False,
        "gpu_queried": False,
        "gpu_command_launched": False,
        "persistent_artifact_written": False,
    }


def verify_posttraining() -> dict[str, Any]:
    result = posttraining.verify_complete_closure()
    _equal("posttraining verification status", result.get("status"), "verified_complete")
    _equal("posttraining run count", result.get("run_count"), 2)
    _equal("posttraining sweep count", result.get("sweep_count"), 4)
    _equal("posttraining trajectory seed", result.get("trajectory_seeds"), [42])
    _equal("posttraining excluded seeds", result.get("excluded_seeds"), [3407, 426780603])
    _equal("posttraining fixed threshold", result.get("fixed_threshold"), 0.5)
    _equal("posttraining paper-core claim", result.get("paper_core_established"), False)
    _equal("posttraining stability claim", result.get("stability_claim_supported"), False)
    closure_path = Path(result["closure"]["path"])
    _inside_new_replay(closure_path, "posttraining closure")
    return result


def _verify_f1(report_path: Path) -> dict[str, Any]:
    report = f1_runner.verify_audit_report(
        report_path,
        repo_root=REPO_ROOT,
        parent_lock=parent_lock.DEFAULT_OUTPUT,
        source_lock=certification_source_lock.DEFAULT_OUTPUT,
    )
    contract = report.get("execution_contract")
    if not isinstance(contract, Mapping):
        _fail("F1 execution contract is missing")
    for label, observed, expected in (
        ("F1 status", report.get("status"), "complete"),
        ("F1 fixed threshold", contract.get("fixed_threshold"), FIXED_THRESHOLD),
        (
            "F1 deployment artifact SHA",
            contract.get("checkpoint_sha256"),
            f1_runner.EXPECTED_INFERENCE_SHA256,
        ),
        (
            "F1 source checkpoint SHA",
            contract.get("source_checkpoint_sha256"),
            f1_runner.EXPECTED_SOURCE_CHECKPOINT_SHA256,
        ),
        ("F1 official-test access", report.get("official_test_accessed"), False),
    ):
        _equal(label, observed, expected)
    gate = report.get("functional_gate")
    if not isinstance(gate, Mapping):
        _fail("F1 functional gate is missing")
    _equal("F1 functional gate status", gate.get("status"), "complete")
    return report


def _verify_deep(deep_output: Path, f1_report: Path) -> dict[str, Any]:
    payload = deep_verifier.verify_deep_verification(
        deep_output,
        f1_report,
        repo_root=REPO_ROOT,
        parent_lock=parent_lock.DEFAULT_OUTPUT,
        source_lock=certification_source_lock.DEFAULT_OUTPUT,
    )
    _equal("deep verification status", payload.get("status"), "verified")
    _equal("deep no-invention status", payload.get("no_invention_status"), True)
    return payload


def _strict_training_bindings(
    status: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _equal("training readiness", status.get("ready"), True)
    requests = posttraining.collect_requests()
    bindings: list[dict[str, Any]] = []
    for arm in replay_core.SUPPORTED_ARMS:
        selected = [request for request in requests if request.arm == arm]
        _equal(f"arm {arm} selected checkpoint count", len(selected), 2)
        first = selected[0]
        run_directory = _inside_new_replay(
            first.run_directory,
            f"arm {arm} completed run",
        )
        marker_path = run_directory / "exact_journal/active.json"
        marker = _canonical_object(
            marker_path,
            f"arm {arm} exact active marker",
        )
        _equal(
            f"arm {arm} exact epoch",
            marker.get("epoch"),
            replay_contract.FORMAL_EPOCHS,
        )
        active_checkpoint = (
            marker_path.parent / str(marker.get("checkpoint_file"))
        )
        _equal(
            f"arm {arm} active checkpoint SHA",
            _sha256_file(active_checkpoint, f"arm {arm} active exact checkpoint"),
            marker.get("checkpoint_sha256"),
        )
        bindings.append(
            {
                "arm": arm,
                "variant": first.variant,
                "trajectory_seed": first.trajectory_seed,
                "run_id": first.run_identity["run_id"],
                "run_directory": str(run_directory),
                "formal_epochs": replay_contract.FORMAL_EPOCHS,
                "completion_summary": _artifact(
                    run_directory / "summary.json",
                    f"arm {arm} completion summary",
                ),
                "protocol": _artifact(
                    run_directory / "protocol.json",
                    f"arm {arm} protocol",
                ),
                "split": _artifact(
                    run_directory / "split.json",
                    f"arm {arm} split",
                ),
                "metrics": _artifact(
                    run_directory / "metrics.jsonl",
                    f"arm {arm} metrics",
                ),
                "exact_active_marker": _artifact(
                    marker_path,
                    f"arm {arm} exact active marker",
                ),
                "exact_active_checkpoint": {
                    "path": str(active_checkpoint.resolve()),
                    "sha256": marker["checkpoint_sha256"],
                },
                "selected_checkpoints": [
                    {
                        "selection_role": request.selection_role,
                        "filename": request.checkpoint_filename,
                        "path": str(request.checkpoint_path),
                        "sha256": request.checkpoint_sha256,
                        "epoch": request.checkpoint_epoch,
                    }
                    for request in selected
                ],
            }
        )
    return bindings


def build_attestation(
    *,
    f1_report: Path = DEFAULT_F1_REPORT,
    deep_output: Path = DEFAULT_DEEP_OUTPUT,
) -> dict[str, Any]:
    locked_sources = verify_completion_source_lock()
    training = training_completion_status()
    training_bindings = _strict_training_bindings(training)
    post = verify_posttraining()
    gate = _canonical_object(
        posttraining.DEFAULT_GATE,
        "new seed42 replay Gate",
    )
    _equal("new seed42 Gate schema", gate.get("schema"), posttraining.GATE_SCHEMA)
    _equal("new seed42 Gate status", gate.get("status"), "complete")
    _equal("new seed42 Gate seed", gate.get("trajectory_seeds"), [42])
    _equal("new seed42 Gate threshold", gate.get("fixed_threshold"), 0.5)
    boundary = gate.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        _fail("new seed42 Gate claim boundary is missing")
    _equal("new seed42 Gate paper-core claim", boundary.get("paper_core_established"), False)
    _equal("new seed42 Gate stability claim", boundary.get("stability_claim_supported"), False)

    f1_path = _regular_file(f1_report, "F1 six-mode report")
    f1 = _verify_f1(f1_path)
    deep_path = _regular_file(deep_output, "F1 deep-verification attestation")
    deep = _verify_deep(deep_path, f1_path)
    f1_contract = f1["execution_contract"]
    functional = f1["functional_gate"]
    limitations = deep.get("limitations")
    if not isinstance(limitations, list):
        _fail("deep verification limitations are missing")

    # Bind all six cache manifests in addition to the report so the final
    # attestation makes its per-image evidence inventory explicit.
    cache_bindings: dict[str, dict[str, str]] = {}
    for mode in f1_runner.PUBLIC_MODES:
        mode_payload = f1["modes"].get(mode)
        if not isinstance(mode_payload, Mapping):
            _fail(f"F1 mode {mode} is missing")
        cache = mode_payload.get("cache")
        if not isinstance(cache, Mapping):
            _fail(f"F1 mode {mode} cache binding is missing")
        cache_path = _portable_report_child(
            f1_path,
            cache.get("path"),
            f"F1 {mode} cache manifest",
        )
        _equal(
            f"F1 {mode} cache SHA",
            _sha256_file(cache_path, f"F1 {mode} cache manifest"),
            cache.get("sha256"),
        )
        cache_bindings[mode] = _artifact(
            cache_path,
            f"F1 {mode} cache manifest",
        )

    comparisons = gate.get("fixed_threshold_and_budget_comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 2:
        _fail("new seed42 Gate must contain two checkpoint-policy comparisons")
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            _fail(f"new seed42 Gate comparison {index} is invalid")
        for metric in (
            "pd",
            "fa",
            "miou",
            "tiny_pd",
            "false_objects_per_image",
        ):
            # The exact nesting is owned by the Gate.  Recursively requiring
            # the metric token avoids silently dropping any registered metric
            # if the presentation shape changes.
            if metric not in json.dumps(comparison, sort_keys=True):
                _fail(
                    f"new seed42 Gate comparison {index} lacks {metric}"
                )

    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "complete",
        "decision": "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
        "scope": SCOPE,
        "stage_order": list(STAGES),
        "model_contract": {
            "mainline": "SCTransNet+TPD8+five-node-NER4+QFG2-CROA",
            "mainline_changed": False,
            "innovation_changed": False,
            "default_threshold": FIXED_THRESHOLD,
            "seed42_deployment_weight_changed": False,
        },
        "implementation_closure": {
            "f0_protocol_and_locks_complete": True,
            "f1_runner_and_deep_verifier_complete": True,
            "f2_full_parameter_contract_runner_tests_complete": True,
            "new_seed42_posttraining_runner_tests_complete": True,
            "top_level_completion_runner_tests_complete": True,
            "old_3407_426_execution_used_in_current_gate": False,
            "seed_3407_role": "supplementary_only",
            "seed_426780603_role": "cancelled_not_scheduled",
            "completion_source_lock": {
                **_artifact(
                    completion_source_lock.DEFAULT_OUTPUT,
                    "completion source lock",
                ),
                "schema": locked_sources["schema"],
                "source_count": locked_sources["source_count"],
            },
            "replay_source_lock_v4": _artifact(
                posttraining.DEFAULT_REPLAY_SOURCE_LOCK,
                "seed42 replay source lock v4",
            ),
            "f0_source_lock_v1": _artifact(
                posttraining.DEFAULT_CERTIFICATION_SOURCE_LOCK,
                "F0 certification source lock",
            ),
            "parent_lock_v1": _artifact(
                posttraining.DEFAULT_PARENT_LOCK,
                "certification parent lock",
            ),
            "f2_engineering_replication_tooling": {
                "status": "implementation_complete_execution_not_in_current_gate",
                "contract_runner_tests_implemented": True,
                "bound_by_f0_source_lock": True,
                "contract": _artifact(
                    REPO_ROOT
                    / "experiments/final_model_replication_seed_contract.py",
                    "F2 seed contract",
                ),
                "exact_core": _artifact(
                    REPO_ROOT
                    / "experiments/final_model_replication_exact_core.py",
                    "F2 exact core",
                ),
                "pair_runner": _artifact(
                    REPO_ROOT
                    / "experiments/run_final_model_replication_seed_pair_2x5090.sh",
                    "F2 pair runner",
                ),
                "runner_tests": [
                    _artifact(
                        REPO_ROOT
                        / "tests/test_final_model_replication_seed_contract.py",
                        "F2 seed-contract tests",
                    ),
                    _artifact(
                        REPO_ROOT
                        / "tests/test_final_model_replication_exact.py",
                        "F2 exact-runner tests",
                    ),
                ],
                "seed_3407_execution": (
                    "supplementary_only_not_closed_not_in_current_gate"
                ),
                "seed_426780603_execution": (
                    "cancelled_not_scheduled_not_in_current_gate"
                ),
                "multiseed_execution_complete": False,
                "current_gate_uses_new_seed42_replay_only": True,
            },
        },
        "new_seed42_replay": {
            "trajectory_seeds": [TRAJECTORY_SEED],
            "excluded_replay_seeds": list(EXCLUDED_REPLAY_SEEDS),
            "old_seed42_stage_results_used_as_new_replay": False,
            "run_count": 2,
            "sweep_count": 4,
            "runs": training_bindings,
            "posttraining_closure": _artifact(
                posttraining.DEFAULT_CLOSURE,
                "new seed42 posttraining closure",
            ),
            "gate": {
                **_artifact(
                    posttraining.DEFAULT_GATE,
                    "new seed42 replay Gate",
                ),
                "decision": gate["decision"],
                "comparisons": copy.deepcopy(comparisons),
            },
            "strict_posttraining_verification": copy.deepcopy(post),
        },
        "frozen_deployment_d_qfg_audit": {
            "subject_is_replay_d": False,
            "subject": "frozen_seed42_deployment_d",
            "physical_gpu_index": GPU2_INDEX,
            "physical_gpu_uuid": GPU2_UUID,
            "report": _artifact(f1_path, "F1 six-mode report"),
            "inference_artifact_sha256": f1_contract["checkpoint_sha256"],
            "source_checkpoint_sha256": f1_contract[
                "source_checkpoint_sha256"
            ],
            "validation_count": f1_contract["validation_count"],
            "validation_ids_sha256": f1_contract[
                "validation_ids_sha256"
            ],
            "fixed_threshold": f1_contract["fixed_threshold"],
            "functional_gate": copy.deepcopy(dict(functional)),
            "cache_manifests": cache_bindings,
        },
        "f1_deep_verification": {
            **_artifact(deep_path, "F1 deep-verification attestation"),
            "status": deep["status"],
            "no_invention_status": deep["no_invention_status"],
            "limitations_count": len(limitations),
        },
        "source_bindings": _source_bindings(),
        "claim_boundary": {
            "single_seed_internal_validation_only": True,
            "official_test_accessed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "multiseed_replication_supported": False,
            "qfg_performance_causal_claim_established": False,
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
        "official_test_accessed": False,
        "write_once": True,
        "overwrite_forbidden": True,
    }


def finalize_attestation(
    *,
    f1_report: Path = DEFAULT_F1_REPORT,
    deep_output: Path = DEFAULT_DEEP_OUTPUT,
    output: Path = DEFAULT_ATTESTATION,
) -> dict[str, Any]:
    destination = Path(output).expanduser()
    _inside_new_replay(destination, "final attestation")
    payload = build_attestation(
        f1_report=f1_report,
        deep_output=deep_output,
    )
    path, action = _write_or_validate(destination, payload)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "finalize-attestation",
        "attestation_action": action,
        "attestation": _artifact(path, "final completion attestation"),
        "decision": payload["decision"],
        "paper_core_established": False,
        "stability_claim_supported": False,
        "gpu_queried": False,
    }


def verify_attestation(
    *,
    f1_report: Path = DEFAULT_F1_REPORT,
    deep_output: Path = DEFAULT_DEEP_OUTPUT,
    output: Path = DEFAULT_ATTESTATION,
) -> dict[str, Any]:
    stored = _canonical_object(output, "final completion attestation")
    expected = build_attestation(
        f1_report=f1_report,
        deep_output=deep_output,
    )
    _equal(
        "stored/live final completion attestation",
        canonical_json_bytes(stored),
        canonical_json_bytes(expected),
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "verified_complete",
        "action": "verify-attestation",
        "attestation": _artifact(output, "final completion attestation"),
        "decision": stored["decision"],
        "paper_core_established": False,
        "stability_claim_supported": False,
        "gpu_queried": False,
    }


def assert_runtime_gpu2() -> dict[str, Any]:
    _equal("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), GPU2_UUID)
    _equal(GPU_INDEX_ENV, os.environ.get(GPU_INDEX_ENV), str(GPU2_INDEX))
    _equal(GPU_UUID_ENV, os.environ.get(GPU_UUID_ENV), GPU2_UUID)
    torch = f1_runner.torch
    _require(torch.cuda.is_available(), "runtime CUDA is unavailable")
    _equal("visible CUDA device count", torch.cuda.device_count(), 1)
    return {
        "schema": ACTION_SCHEMA,
        "status": "ready",
        "action": "assert-runtime-gpu2",
        "physical_gpu_index": GPU2_INDEX,
        "physical_gpu_uuid": GPU2_UUID,
        "visible_cuda_device_count": 1,
        "device_name": torch.cuda.get_device_name(0),
        "gpu_queried": True,
        "writes_performed": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--training-ready", action="store_true")
    action.add_argument("--verify-posttraining", action="store_true")
    action.add_argument("--verify-source-lock", action="store_true")
    action.add_argument("--assert-runtime-gpu2", action="store_true")
    action.add_argument("--finalize-attestation", action="store_true")
    action.add_argument("--verify-attestation", action="store_true")
    parser.add_argument("--f1-report", type=Path, default=DEFAULT_F1_REPORT)
    parser.add_argument("--deep-output", type=Path, default=DEFAULT_DEEP_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ATTESTATION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    exit_code = 0
    if args.dry_run:
        payload = dry_run_plan(
            f1_report=args.f1_report,
            deep_output=args.deep_output,
            attestation=args.output,
        )
    elif args.training_ready:
        payload = training_completion_status()
        if not payload["ready"]:
            exit_code = (
                RESUME_NEEDED_EXIT_CODE
                if payload["training_pair_lock"]["released"]
                else WAITING_EXIT_CODE
            )
    elif args.verify_posttraining:
        payload = verify_posttraining()
    elif args.verify_source_lock:
        stored = verify_completion_source_lock()
        payload = {
            "schema": ACTION_SCHEMA,
            "status": "verified_locked",
            "action": "verify-source-lock",
            "source_lock": _artifact(
                completion_source_lock.DEFAULT_OUTPUT,
                "completion source lock",
            ),
            "source_count": stored["source_count"],
            "gpu_queried": False,
            "writes_performed": False,
        }
    elif args.assert_runtime_gpu2:
        payload = assert_runtime_gpu2()
    elif args.finalize_attestation:
        payload = finalize_attestation(
            f1_report=args.f1_report,
            deep_output=args.deep_output,
            output=args.output,
        )
    else:
        payload = verify_attestation(
            f1_report=args.f1_report,
            deep_output=args.deep_output,
            output=args.output,
        )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    if exit_code:
        raise SystemExit(exit_code)


__all__ = [
    "ACTION_SCHEMA",
    "ATTESTATION_SCHEMA",
    "DEFAULT_ATTESTATION",
    "DEFAULT_DEEP_OUTPUT",
    "DEFAULT_F1_REPORT",
    "GPU2_INDEX",
    "GPU2_UUID",
    "RESUME_NEEDED_EXIT_CODE",
    "SCHEMA",
    "STAGES",
    "Seed42CertificationCompletionError",
    "WAITING_EXIT_CODE",
    "assert_runtime_gpu2",
    "build_attestation",
    "canonical_json_bytes",
    "dry_run_plan",
    "finalize_attestation",
    "main",
    "training_completion_status",
    "training_pair_lock_status",
    "verify_attestation",
    "verify_completion_source_lock",
    "verify_posttraining",
]


if __name__ == "__main__":
    main()
