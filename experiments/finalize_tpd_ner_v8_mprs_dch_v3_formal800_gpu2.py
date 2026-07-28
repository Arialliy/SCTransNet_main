#!/usr/bin/env python3
"""Finalize the canonical V3 formal800 GPU2 lane exactly once.

This watcher is deliberately separate from the frozen V3 training and
postprocessing sources.  It waits for the one canonical transient service to
stop, proves the complete exact-epoch training closure, asks the frozen
postprocessor for a read-only plan, and only then invokes its GPU2 ``--run-now``
path once.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import TracebackType
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tpd_exact_epoch_journal as epoch_journal  # noqa: E402
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v3_exact as exact,
)


SERVICE = "sctransnet-tpd-ner-v8-v3-relay-on-gpu2.service"
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
POSTPROCESS = (
    REPO_ROOT / "experiments/postprocess_tpd_ner_v8_mprs_dch_v3_formal800.py"
)
EVALUATOR = (
    REPO_ROOT / "experiments/evaluate_tpd_ner_v8_mprs_dch_v3_pd_fa.py"
)
DATASET = "NUDT-SIRST"
VARIANT = exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
CANONICAL_RUN_DIR = (
    exact.DEFAULT_OUTPUT_ROOT
    / DATASET
    / VARIANT
    / f"seed_{TRAINING_SEED}_{exact.FORMAL_RUN_TAG}"
)
COMPARISON_DIR = exact.DEFAULT_OUTPUT_ROOT / DATASET / "comparison"
JSON_OUTPUT = (
    COMPARISON_DIR / "tpd_ner_v8_mprs_dch_v3_formal800_comparison.json"
)
MARKDOWN_OUTPUT = (
    COMPARISON_DIR / "tpd_ner_v8_mprs_dch_v3_formal800_comparison.md"
)
COMPLETE_MARKER = COMPARISON_DIR / "POSTPROCESS_COMPLETE.json"
POSTPROCESS_REPORT_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_posttraining_aggregate_v1"
)
POSTPROCESS_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_postprocess_complete_v1"
)
POSTPROCESS_READINESS_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_posttraining_readiness_v1"
)
ACTIVE_STATES = frozenset(
    {"active", "activating", "deactivating", "reloading"}
)
CHECKPOINT_CONTRACTS = (
    ("best.pth.tar", "best_validation_pd_primary"),
    ("best_miou.pth.tar", "best_validation_miou_secondary"),
    ("last.pth.tar", "last_evaluated_epoch"),
)


class FinalizerError(RuntimeError):
    """The automatic V3 closure cannot safely proceed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizerError(message)


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular non-symlink file: {path}",
    )
    return path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    content = _regular_file(path, label).read_text(encoding="utf-8")
    try:
        value = json.loads(
            content,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FinalizerError(f"{label} is invalid JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, "hashed output").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_postprocess_complete() -> dict[str, Any] | None:
    """Validate and return the canonical completion marker, if it exists."""

    if not COMPLETE_MARKER.exists():
        _require(
            not COMPLETE_MARKER.is_symlink(),
            f"completion marker may not be a symlink: {COMPLETE_MARKER}",
        )
        return None
    marker = _load_json_object(COMPLETE_MARKER, "postprocess completion marker")
    expected_keys = {
        "schema",
        "status",
        "decision",
        "aggregate_full_model_gate_passed",
        "aggregate_tiny_pd_regressed",
        "tiny_pd_regression_affects_decision",
        "outputs",
    }
    _require(
        set(marker) == expected_keys,
        "POSTPROCESS_COMPLETE has an unexpected schema",
    )
    _require(
        marker.get("schema") == POSTPROCESS_MARKER_SCHEMA
        and marker.get("status") == "complete",
        "POSTPROCESS_COMPLETE is not a valid V3 complete marker",
    )
    _require(
        marker.get("decision")
        in {"FULL_MODEL_GATE_PASSED", "RETURN_TO_MODEL_OPTIMIZATION"},
        "POSTPROCESS_COMPLETE decision is invalid",
    )
    for field in (
        "aggregate_full_model_gate_passed",
        "aggregate_tiny_pd_regressed",
    ):
        _require(type(marker.get(field)) is bool, f"marker {field} is invalid")
    _require(
        marker.get("tiny_pd_regression_affects_decision") is False,
        "POSTPROCESS_COMPLETE changes the preregistered tiny-Pd decision",
    )
    outputs = marker.get("outputs")
    expected_outputs = {
        JSON_OUTPUT.name: JSON_OUTPUT,
        MARKDOWN_OUTPUT.name: MARKDOWN_OUTPUT,
    }
    _require(
        isinstance(outputs, dict) and set(outputs) == set(expected_outputs),
        "POSTPROCESS_COMPLETE output set differs",
    )
    for name, path in expected_outputs.items():
        digest = outputs.get(name)
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"POSTPROCESS_COMPLETE digest is invalid: {name}",
        )
        _require(
            _sha256_file(path) == digest,
            f"POSTPROCESS_COMPLETE digest differs: {name}",
        )
    report = _load_json_object(JSON_OUTPUT, "V3 postprocess JSON report")
    _require(
        report.get("schema") == POSTPROCESS_REPORT_SCHEMA
        and report.get("status") == "complete",
        "V3 postprocess report is not complete",
    )
    for field in (
        "decision",
        "aggregate_full_model_gate_passed",
        "aggregate_tiny_pd_regressed",
        "tiny_pd_regression_affects_decision",
    ):
        _require(
            report.get(field) == marker.get(field),
            f"POSTPROCESS_COMPLETE differs from report field: {field}",
        )
    return marker


def _expected_run_id() -> str:
    return (
        f"{exact.RUN_ID_PREFIX}{DATASET}:{VARIANT}:"
        f"seed-{TRAINING_SEED}:split-{SPLIT_SEED}:{exact.FORMAL_RUN_TAG}"
    )


def _validated_run_identity(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        identity = exact.require_v3_run_identity(
            value,
            label=label,
            expected_variant=VARIANT,
        )
    except (TypeError, ValueError) as exc:
        raise FinalizerError(f"{label} is invalid: {exc}") from exc
    _require(identity.get("dataset") == DATASET, f"{label} dataset differs")
    _require(
        identity.get("seed") == TRAINING_SEED,
        f"{label} training seed differs",
    )
    _require(
        identity.get("split_seed") == SPLIT_SEED,
        f"{label} split seed differs",
    )
    _require(
        identity.get("run_id") == _expected_run_id(),
        f"{label} run_id differs",
    )
    return identity


def inspect_strict_training_completion(
    run_dir: Path = CANONICAL_RUN_DIR,
) -> dict[str, Any]:
    """Prove the same strict epoch-800 closure enforced by the frozen lane."""

    directory = Path(run_dir)
    if not directory.exists():
        _require(
            not directory.is_symlink(),
            f"canonical V3 run directory may not be a symlink: {directory}",
        )
        return {
            "complete": False,
            "reason": "canonical V3 run directory is absent",
            "run_dir": str(directory),
        }
    _require(
        directory.is_dir() and not directory.is_symlink(),
        f"canonical V3 run path is not a real directory: {directory}",
    )
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        _require(
            not summary_path.is_symlink(),
            "V3 summary path may not be a symlink",
        )
        return {
            "complete": False,
            "reason": "completion summary is absent",
            "run_dir": str(directory),
        }
    summary = _load_json_object(summary_path, "V3 completion summary")
    if summary.get("status") != "complete":
        return {
            "complete": False,
            "reason": (
                "completion summary status is "
                f"{summary.get('status')!r}, not 'complete'"
            ),
            "run_dir": str(directory),
        }
    _require(
        summary.get("schema") == exact.COMPLETION_SUMMARY_SCHEMA,
        "V3 completion summary schema differs",
    )
    for field, expected in {
        "variant": VARIANT,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
    }.items():
        _require(
            summary.get(field) == expected,
            f"V3 completion summary {field} differs",
        )

    protocol = _load_json_object(directory / "protocol.json", "V3 protocol")
    _require(
        protocol.get("schema") == exact.ENTRY_SCHEMA,
        "V3 protocol schema differs",
    )
    protocol_identity = _validated_run_identity(
        protocol.get("run_identity"),
        label="V3 protocol run identity",
    )
    summary_identity = _validated_run_identity(
        summary.get("run_identity"),
        label="V3 completion summary run identity",
    )
    _require(
        summary_identity == protocol_identity,
        "V3 completion summary and protocol identities differ",
    )
    source_locks = protocol_identity.get("source_locks")
    _require(
        isinstance(source_locks, Mapping)
        and source_locks.get(exact.SOURCE_LOCK_KEY)
        == exact.file_sha256(exact.DEFAULT_EXACT_SOURCE_LOCK_PATH),
        "V3 protocol is not bound to the frozen training lock",
    )

    metrics_path = _regular_file(
        directory / "metrics.jsonl",
        "V3 metrics journal",
    )
    metrics_bytes = metrics_path.read_bytes()
    _require(
        metrics_bytes.endswith(b"\n"),
        "V3 metrics journal is truncated",
    )
    try:
        metrics_text = metrics_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinalizerError("V3 metrics journal is not UTF-8") from exc
    metric_lines = metrics_text.splitlines()
    _require(
        len(metric_lines) == EXPECTED_EPOCHS
        and all(line.strip() for line in metric_lines),
        "V3 metrics journal does not contain exactly 800 nonempty events",
    )
    events: list[dict[str, Any]] = []
    for epoch, line in enumerate(metric_lines, start=1):
        try:
            event = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise FinalizerError(
                f"V3 metrics event {epoch} is invalid JSON: {exc}"
            ) from exc
        _require(
            isinstance(event, dict) and event.get("epoch") == epoch,
            "V3 metrics epochs are not contiguous 1..800",
        )
        _require(
            event.get("variant") == VARIANT,
            f"V3 metrics event {epoch} variant differs",
        )
        _require(
            set(exact.STORED_VALIDATION_METRICS).issubset(event),
            f"V3 metrics event {epoch} lacks stored validation fields",
        )
        events.append(event)

    last_checkpoint_epoch: Any = None
    for checkpoint_name, expected_role in CHECKPOINT_CONTRACTS:
        checkpoint_path = _regular_file(
            directory / checkpoint_name,
            f"V3 checkpoint {checkpoint_name}",
        )
        try:
            raw_checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint = exact.require_evaluator_checkpoint_payload(
                raw_checkpoint,
                expected_variant=VARIANT,
            )
        except Exception as exc:
            raise FinalizerError(
                f"V3 checkpoint is invalid: {checkpoint_name}: {exc}"
            ) from exc
        _require(
            checkpoint.get("run_identity") == protocol_identity,
            f"V3 checkpoint run identity differs: {checkpoint_name}",
        )
        _require(
            checkpoint.get("checkpoint_role") == expected_role,
            f"V3 checkpoint role differs: {checkpoint_name}",
        )
        if checkpoint_name == "last.pth.tar":
            last_checkpoint_epoch = checkpoint.get("epoch")
        del raw_checkpoint, checkpoint
    _require(
        last_checkpoint_epoch == EXPECTED_EPOCHS,
        "V3 last checkpoint is not evaluated epoch 800",
    )

    journal_root = directory / "exact_journal"
    active_path = journal_root / epoch_journal.MARKER_FILENAME
    _require(
        journal_root.is_dir()
        and not journal_root.is_symlink()
        and active_path.is_file()
        and not active_path.is_symlink(),
        "V3 run has no regular active exact journal",
    )
    try:
        active = epoch_journal.ExactEpochJournal(journal_root).load_active()
    except Exception as exc:
        raise FinalizerError(f"V3 exact journal is invalid: {exc}") from exc
    _require(
        active is not None and active.epoch == EXPECTED_EPOCHS,
        "V3 active exact journal is not committed at epoch 800",
    )
    _require(
        active.metrics_path.read_bytes() == metrics_bytes,
        "V3 derived metrics differ from the active exact journal",
    )
    try:
        active_payload = torch.load(
            active.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise FinalizerError(
            f"V3 active exact-journal checkpoint cannot be loaded: {exc}"
        ) from exc
    _require(
        isinstance(active_payload, Mapping),
        "V3 active exact-journal checkpoint is not a mapping",
    )
    active_identity = _validated_run_identity(
        active_payload.get("run_identity"),
        label="V3 active exact-journal run identity",
    )
    _require(
        active_identity == protocol_identity,
        "V3 active exact-journal run identity differs",
    )
    return {
        "complete": True,
        "reason": "strict epoch-800 exact closure verified",
        "run_dir": str(directory.resolve()),
        "event_count": len(events),
        "last_epoch": events[-1]["epoch"],
        "checkpoint_roles": {
            name: role for name, role in CHECKPOINT_CONTRACTS
        },
        "active_journal_epoch": active.epoch,
        "run_id": protocol_identity["run_id"],
    }


def inspect_training_service() -> dict[str, str]:
    """Read only the one fixed GPU2 transient service."""

    process = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            SERVICE,
            (
                "--property="
                "LoadState,ActiveState,SubState,Result,ExecMainStatus"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    properties: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    collected = properties.get("LoadState") == "not-found"
    if process.returncode and not collected:
        detail = process.stderr.strip() or process.stdout.strip()
        raise FinalizerError(
            f"cannot inspect fixed training service {SERVICE}: {detail}"
        )
    _require(
        "ActiveState" in properties or collected,
        f"systemd returned no state for fixed training service {SERVICE}",
    )
    return properties


def _postprocess_command(mode: str) -> list[str]:
    _require(mode in {"--plan", "--run-now"}, "invalid postprocess mode")
    return [
        str(PYTHON),
        str(POSTPROCESS),
        mode,
        "--device-mode",
        "gpu23",
        "--physical-gpu",
        "2",
        "--python",
        str(PYTHON),
    ]


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def validate_locked_postprocess_plan(plan: Any) -> dict[str, Any]:
    """Require a GPU2 plan containing only the two new V3 evaluations."""

    _require(isinstance(plan, dict), "locked postprocess plan is not an object")
    readiness = plan.get("readiness")
    _require(
        isinstance(readiness, dict)
        and readiness.get("schema") == POSTPROCESS_READINESS_SCHEMA
        and readiness.get("required_runs_complete") is True,
        "locked postprocess --plan is not ready",
    )
    v3_readiness = readiness.get("v3_on")
    _require(
        isinstance(v3_readiness, dict)
        and v3_readiness.get("complete") is True
        and Path(v3_readiness.get("run_dir", "")).resolve()
        == CANONICAL_RUN_DIR.resolve(),
        "locked postprocess plan does not bind the canonical V3 run",
    )
    _require(
        plan.get("new_evaluation_count") == 2,
        "locked postprocess plan must contain exactly two evaluations",
    )
    evaluations = plan.get("new_evaluations")
    _require(
        isinstance(evaluations, list) and len(evaluations) == 2,
        "locked postprocess plan evaluation list differs",
    )
    _require(
        {
            evaluation.get("checkpoint")
            for evaluation in evaluations
            if isinstance(evaluation, dict)
        }
        == {"best.pth.tar", "best_miou.pth.tar"},
        "locked postprocess plan checkpoint set differs",
    )
    for evaluation in evaluations:
        _require(
            isinstance(evaluation, dict)
            and evaluation.get("variant") == VARIANT
            and evaluation.get("physical_gpu_index") == 2,
            "locked postprocess plan contains a non-canonical evaluation",
        )
        command = evaluation.get("command")
        checkpoint = evaluation.get("checkpoint")
        _require(
            isinstance(command, list)
            and len(command) >= 2
            and Path(command[0]).resolve() == PYTHON.resolve()
            and Path(command[1]).resolve() == EVALUATOR.resolve()
            and Path(
                _flag_value(command, "--run-dir") or ""
            ).resolve()
            == CANONICAL_RUN_DIR.resolve()
            and _flag_value(command, "--checkpoint") == checkpoint
            and _flag_value(command, "--device") == "cuda:0"
            and _flag_value(command, "--expected-epochs")
            == str(EXPECTED_EPOCHS)
            and "--device-mode" not in command
            and "--physical-gpu" not in command,
            "locked postprocess evaluation is not the canonical GPU2 "
            "evaluator command",
        )
    for field in (
        "v2_on_evaluations",
        "v1_off_evaluations",
        "baseline_evaluations",
    ):
        _require(
            plan.get(field) == 0,
            f"locked postprocess plan would regenerate upstream data: {field}",
        )
    _require(
        plan.get("aggregate_outputs")
        == [str(JSON_OUTPUT), str(MARKDOWN_OUTPUT), str(COMPLETE_MARKER)],
        "locked postprocess aggregate outputs differ",
    )
    _require(
        plan.get("training_seed") == TRAINING_SEED
        and plan.get("split_seed") == SPLIT_SEED
        and plan.get("multi_seed_scheduled") is False,
        "locked postprocess plan seed scope differs",
    )
    return plan


def locked_postprocess_plan() -> dict[str, Any]:
    """Run the frozen postprocessor's read-only source-lock/readiness gate."""

    command = _postprocess_command("--plan")
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise FinalizerError(
            f"locked postprocess --plan failed ({process.returncode}): {detail}"
        )
    try:
        plan = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise FinalizerError(
            "locked postprocess --plan did not emit one JSON object"
        ) from exc
    return validate_locked_postprocess_plan(plan)


def run_locked_postprocess_once() -> None:
    """Invoke exactly the approved locked GPU2 postprocess command once."""

    command = _postprocess_command("--run-now")
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
    )
    if process.returncode:
        raise FinalizerError(
            "locked postprocess --run-now failed with "
            f"exit status {process.returncode}"
        )


def watch_and_finalize(
    *,
    poll_seconds: float = 30.0,
) -> dict[str, Any]:
    """Poll the fixed service and perform at most one run-now invocation."""

    _require(
        isinstance(poll_seconds, (int, float))
        and not isinstance(poll_seconds, bool)
        and 0 < float(poll_seconds) <= 30,
        "poll interval must be greater than 0 and no more than 30 seconds",
    )
    while True:
        marker = inspect_postprocess_complete()
        if marker is not None:
            return {
                "status": "already_complete",
                "service": SERVICE,
                "run_dir": str(CANONICAL_RUN_DIR),
                "marker": str(COMPLETE_MARKER),
                "decision": marker["decision"],
                "postprocess_invocations": 0,
            }

        service = inspect_training_service()
        completion = inspect_strict_training_completion()
        active_state = service.get("ActiveState", "inactive")
        if active_state in ACTIVE_STATES:
            print(
                "TPDNERV8V3_FINALIZER_WAIT"
                f" service={SERVICE}"
                f" active_state={active_state}"
                f" strict_complete={str(completion['complete']).lower()}"
                f" poll_seconds={float(poll_seconds):g}",
                flush=True,
            )
            time.sleep(float(poll_seconds))
            continue
        if completion.get("complete") is not True:
            raise FinalizerError(
                "fixed V3 training service stopped before strict completion: "
                f"service={SERVICE} "
                f"LoadState={service.get('LoadState')!r} "
                f"ActiveState={active_state!r} "
                f"SubState={service.get('SubState')!r} "
                f"Result={service.get('Result')!r} "
                f"ExecMainStatus={service.get('ExecMainStatus')!r}; "
                f"completion={completion.get('reason')}"
            )

        locked_postprocess_plan()
        run_locked_postprocess_once()
        marker = inspect_postprocess_complete()
        _require(
            marker is not None,
            "locked postprocess returned success without POSTPROCESS_COMPLETE",
        )
        return {
            "status": "finalized",
            "service": SERVICE,
            "run_dir": str(CANONICAL_RUN_DIR),
            "marker": str(COMPLETE_MARKER),
            "decision": marker["decision"],
            "postprocess_invocations": 1,
        }


class _FinalizerLock:
    """Non-blocking per-user lock preventing concurrent run-now calls."""

    def __init__(self) -> None:
        self._descriptor: int | None = None
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(runtime) if runtime else Path("/tmp")
        _require(
            base.is_dir() and not base.is_symlink(),
            f"finalizer lock directory is unsafe: {base}",
        )
        self.path = (
            base
            / f"sctransnet-tpd-ner-v8-v3-finalizer-gpu2-{os.getuid()}.lock"
        )

    def __enter__(self) -> "_FinalizerLock":
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise FinalizerError(
                f"cannot open finalizer lock {self.path}: {exc}"
            ) from exc
        try:
            mode = os.fstat(descriptor).st_mode
            _require(
                stat.S_ISREG(mode),
                f"finalizer lock is not a regular file: {self.path}",
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FinalizerError(
                    "another canonical V3 GPU2 finalizer is already running"
                ) from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch and finalize only the canonical V3 formal800 GPU2 service"
        )
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not 0 < args.poll_seconds <= 30:
        parser.error("--poll-seconds must be > 0 and <= 30")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        marker = inspect_postprocess_complete()
        if marker is not None:
            result = {
                "status": "already_complete",
                "service": SERVICE,
                "run_dir": str(CANONICAL_RUN_DIR),
                "marker": str(COMPLETE_MARKER),
                "decision": marker["decision"],
                "postprocess_invocations": 0,
            }
        else:
            with _FinalizerLock():
                result = watch_and_finalize(
                    poll_seconds=args.poll_seconds,
                )
    except (FinalizerError, OSError) as exc:
        print(f"TPDNERV8V3_FINALIZER_FAILED {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
