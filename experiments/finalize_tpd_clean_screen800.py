#!/usr/bin/env python3
"""Wait for all TPD-Clean workers, then run the independent summarizer.

This process is deliberately narrow.  It observes the four candidate logs and
result directories, publishes an atomic state file, and invokes
``summarize_tpd_clean_screen800.py`` only after every worker has written one
and only one ``TPDCLEAN_COMPLETE`` record.

It never writes into the frozen formal800 result root and never edits model or
mainline files.  Scientific interpretation remains the summarizer's job.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_screen800_4x5090_v1"
)
DEFAULT_REFERENCE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_pe_formal800_4x5090_v1"
)
DEFAULT_AGGREGATOR = REPO_ROOT / "experiments/summarize_tpd_clean_screen800.py"
BASE_EVALUATOR = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
DATA_FINGERPRINT_SCRIPT = REPO_ROOT / "experiments/fingerprint_tpd_training_data.py"
NEXT_MODULE_GATE_JSON = REPO_ROOT / "experiments/tpd_clean_next_module_gate_v1.json"
NEXT_MODULE_GATE_MD = REPO_ROOT / "experiments/TPD_CLEAN_V2_NEXT_MODULE_GATE.md"
EXPECTED_TRAINING_DATA_SHA256 = (
    "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
)
DEFAULT_RUN_NAME = "seed_42_screen800_pd_fp32_shared4x5090_v1"
DEFAULT_REFERENCE_RUN_NAME = "seed_42_formal800_pd_fp32_4x5090_v1"
DATASET = "NUDT-SIRST"
VARIANTS = (
    "grouped_keep",
    "tpd_clean_ctx",
    "tpd_clean_sal",
    "tpd_clean_full",
)
EXPECTED_EPOCHS = 800
REFERENCE_VARIANTS = ("spd", "tpd")
FROZEN_MIOU_CHECKPOINT_SHA256 = {
    "spd": "f932198ffa33408c8faa8801580bc8db6a337afa8544770d8c972f1c8bde232a",
    "tpd": "ce75b45494ada10ed3c2f8915a5e9be6223548fbce5e131acbb184d8d67b2676",
}
REFERENCE_MIRROR_FILES = (
    "protocol.json",
    "split.json",
    "summary.json",
    "metrics.jsonl",
    "best_miou.pth.tar",
)
REFERENCE_SWEEP_NAME = "pd_fa_sweep_best_miou.pth.json"
STATE_SCHEMA = "sctransnet_tpd_clean_screen800_finalizer_state_v1"
OUTPUT_SCHEMA = "sctransnet_tpd_clean_screen800_comparison_v2"
OUTPUT_JSON_NAME = "tpd_clean_screen800_comparison_seed42.json"
OUTPUT_MD_NAME = "tpd_clean_screen800_comparison_seed42.md"
OUTPUT_MARKER_NAME = "tpd_clean_screen800_comparison_seed42.COMPLETE.sha256"
VALID_STATES = (
    "waiting",
    "validating_candidates",
    "generating_frozen_miou_reference_sweeps",
    "aggregating",
    "complete",
    "failed",
)
WAITING_EXIT_CODE = 3
COMPLETE_PATTERN = re.compile(
    r"^TPDCLEAN_COMPLETE"
    r" variant=(?P<variant>[a-z0-9_]+)"
    r" gpu_uuid=(?P<gpu_uuid>\S+)"
    r" epochs=(?P<epochs>\d+)$"
)
ABORT_PATTERN = re.compile(r"^TPDCLEAN(?:_[A-Z0-9]+)*_ABORT\b")
REQUIRED_RUN_FILES = (
    "protocol.json",
    "split.json",
    "metrics.jsonl",
    "summary.json",
    "best.pth.tar",
    "best_miou.pth.tar",
    "last.pth.tar",
    "pd_fa_sweep_best.pth.json",
    "pd_fa_sweep_best_miou.pth.json",
)
PROTECTED_MAINLINE_FILES = (
    REPO_ROOT / "model/tpd.py",
    REPO_ROOT / "TPD_SCTransNet_主线修订版.md",
)


class FinalizerError(RuntimeError):
    """Raised when completion evidence or output state is inconsistent."""


@dataclass(frozen=True)
class VariantObservation:
    """Read-only status for one worker and its result directory."""

    variant: str
    unit: str
    run_directory: str
    log_path: str
    log_exists: bool
    metric_rows: int
    completion_count: int
    completion_gpu_uuid: str | None
    completion_epochs: int | None
    abort_count: int
    required_files_present: Dict[str, bool]
    unit_state: Dict[str, Any]
    ready: bool
    problems: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FinalizerError(f"{label} is missing, linked, or not a regular file: {path}")


def count_metric_rows(path: Path) -> int:
    if not path.is_file() or path.is_symlink():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise FinalizerError(
                    f"Blank metrics row at {path}:{line_number}"
                )
            count += 1
    return count


def atomic_write_text(path: Path, content: str) -> None:
    """Write a text file by replacing it only after the new bytes are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_json_object(path: Path) -> Dict[str, Any]:
    require_regular_file(path, "JSON artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalizerError(f"Expected a JSON object: {path}")
    return payload


def systemd_unit_state(variant: str) -> Dict[str, Any]:
    unit = f"sctransnet-tpd-clean-screen800-{variant}.service"
    command = [
        "systemctl",
        "--user",
        "show",
        unit,
        "--no-pager",
        "--property=LoadState",
        "--property=ActiveState",
        "--property=SubState",
        "--property=Result",
        "--property=ExecMainStatus",
        "--property=NRestarts",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    if completed.returncode != 0:
        return {
            "available": False,
            "returncode": completed.returncode,
            "error": completed.stderr.strip(),
        }
    values: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {"available": True, **values}


def inspect_variant(
    candidate_root: Path,
    run_name: str,
    variant: str,
    *,
    include_unit_state: bool,
) -> VariantObservation:
    run_directory = candidate_root / DATASET / variant / run_name
    log_path = candidate_root / "logs" / f"{variant}.log"
    problems: list[str] = []
    completion_records: list[re.Match[str]] = []
    abort_count = 0
    log_exists = log_path.is_file() and not log_path.is_symlink()

    if log_path.exists() and (not log_path.is_file() or log_path.is_symlink()):
        problems.append(f"{variant}: log is not a regular file")
    elif log_exists:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.rstrip("\r\n")
                completion = COMPLETE_PATTERN.fullmatch(stripped)
                if completion is not None:
                    completion_records.append(completion)
                if ABORT_PATTERN.match(stripped):
                    abort_count += 1

    own_records = [
        record
        for record in completion_records
        if record.group("variant") == variant
    ]
    foreign_records = [
        record.group("variant")
        for record in completion_records
        if record.group("variant") != variant
    ]
    if foreign_records:
        problems.append(
            f"{variant}: log contains completion records for {sorted(foreign_records)}"
        )
    if len(own_records) > 1:
        problems.append(
            f"{variant}: expected one completion record, found {len(own_records)}"
        )
    if abort_count:
        problems.append(f"{variant}: log contains {abort_count} abort record(s)")

    completion_gpu_uuid = (
        own_records[0].group("gpu_uuid") if len(own_records) == 1 else None
    )
    completion_epochs = (
        int(own_records[0].group("epochs")) if len(own_records) == 1 else None
    )
    if completion_epochs is not None and completion_epochs != EXPECTED_EPOCHS:
        problems.append(
            f"{variant}: completion epochs={completion_epochs}, "
            f"expected {EXPECTED_EPOCHS}"
        )

    if run_directory.exists() and (
        not run_directory.is_dir() or run_directory.is_symlink()
    ):
        problems.append(f"{variant}: run directory is linked or not a directory")

    required_files_present = {
        name: (run_directory / name).is_file()
        and not (run_directory / name).is_symlink()
        for name in REQUIRED_RUN_FILES
    }
    metrics_path = run_directory / "metrics.jsonl"
    try:
        metric_rows = count_metric_rows(metrics_path)
    except FinalizerError as error:
        metric_rows = 0
        problems.append(str(error))

    if len(own_records) == 1:
        if metric_rows != EXPECTED_EPOCHS:
            problems.append(
                f"{variant}: completed worker has {metric_rows} metrics rows, "
                f"expected {EXPECTED_EPOCHS}"
            )
        missing = [
            name for name, present in required_files_present.items() if not present
        ]
        if missing:
            problems.append(
                f"{variant}: completion record exists but artifacts are missing: "
                + ", ".join(missing)
            )

    unit_state = systemd_unit_state(variant) if include_unit_state else {
        "available": False,
        "reason": "unit inspection disabled",
    }
    if unit_state.get("available"):
        active_state = unit_state.get("ActiveState")
        result = unit_state.get("Result")
        if active_state == "failed" or result in {
            "exit-code",
            "signal",
            "timeout",
            "core-dump",
            "resources",
        }:
            problems.append(
                f"{variant}: unit ended unsuccessfully "
                f"(ActiveState={active_state}, Result={result})"
            )

    ready = (
        len(own_records) == 1
        and completion_epochs == EXPECTED_EPOCHS
        and metric_rows == EXPECTED_EPOCHS
        and all(required_files_present.values())
        and not problems
    )
    return VariantObservation(
        variant=variant,
        unit=f"sctransnet-tpd-clean-screen800-{variant}.service",
        run_directory=str(run_directory),
        log_path=str(log_path),
        log_exists=log_exists,
        metric_rows=metric_rows,
        completion_count=len(own_records),
        completion_gpu_uuid=completion_gpu_uuid,
        completion_epochs=completion_epochs,
        abort_count=abort_count,
        required_files_present=required_files_present,
        unit_state=unit_state,
        ready=ready,
        problems=tuple(problems),
    )


def inspect_all(
    candidate_root: Path,
    run_name: str,
    *,
    include_unit_state: bool,
) -> Dict[str, VariantObservation]:
    return {
        variant: inspect_variant(
            candidate_root,
            run_name,
            variant,
            include_unit_state=include_unit_state,
        )
        for variant in VARIANTS
    }


def observations_payload(
    observations: Mapping[str, VariantObservation],
) -> Dict[str, Any]:
    return {variant: asdict(observations[variant]) for variant in VARIANTS}


def observations_ready(
    observations: Mapping[str, VariantObservation],
) -> bool:
    return all(observations[variant].ready for variant in VARIANTS)


def observation_problems(
    observations: Mapping[str, VariantObservation],
) -> list[str]:
    return [
        problem
        for variant in VARIANTS
        for problem in observations[variant].problems
    ]


def snapshot_finalizer_inputs(aggregator: Path) -> Dict[str, Dict[str, str]]:
    """Hash the programs and frozen gate definitions that control finalization."""

    inputs = {
        "finalizer": Path(__file__).resolve(),
        "summarizer": aggregator.resolve(),
        "next_module_gate_json": NEXT_MODULE_GATE_JSON,
        "next_module_gate_markdown": NEXT_MODULE_GATE_MD,
        "base_pd_fa_evaluator": BASE_EVALUATOR,
        "training_data_fingerprint": DATA_FINGERPRINT_SCRIPT,
    }
    snapshot: Dict[str, Dict[str, str]] = {}
    for name, path in inputs.items():
        require_regular_file(path, f"finalizer input {name}")
        snapshot[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return snapshot


def state_payload(
    *,
    run_id: str,
    state: str,
    candidate_root: Path,
    reference_root: Path,
    reference_miou_root: Path,
    aggregator: Path,
    output_dir: Path,
    poll_seconds: int,
    observations: Mapping[str, VariantObservation] | None,
    message: str,
    aggregator_result: Mapping[str, Any] | None = None,
    reference_sweep_result: Mapping[str, Any] | None = None,
    reused_existing_outputs: bool = False,
) -> Dict[str, Any]:
    if state not in VALID_STATES:
        raise ValueError(f"Unknown finalizer state: {state}")
    return {
        "schema": STATE_SCHEMA,
        "run_id": run_id,
        "updated_at_utc": utc_now(),
        "state": state,
        "message": message,
        "candidate_root": str(candidate_root),
        "reference_root": str(reference_root),
        "reference_miou_root": str(reference_miou_root),
        "aggregator": str(aggregator),
        "output_dir": str(output_dir),
        "poll_seconds": poll_seconds,
        "expected_variants": list(VARIANTS),
        "expected_epochs": EXPECTED_EPOCHS,
        "completion_evidence": (
            observations_payload(observations) if observations is not None else {}
        ),
        "all_unique_completions_observed": (
            observations_ready(observations)
            if observations is not None
            else False
        ),
        "aggregator_result": dict(aggregator_result or {}),
        "reference_sweep_result": dict(reference_sweep_result or {}),
        "input_snapshot": snapshot_finalizer_inputs(aggregator),
        "reused_existing_outputs": reused_existing_outputs,
        "mainline_changed": False,
        "formal800_written": False,
    }


def output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / OUTPUT_JSON_NAME,
        output_dir / OUTPUT_MD_NAME,
        output_dir / OUTPUT_MARKER_NAME,
    )


def marker_content(json_path: Path, md_path: Path) -> str:
    return (
        f"{file_sha256(json_path)}  {json_path.name}\n"
        f"{file_sha256(md_path)}  {md_path.name}\n"
    )


def verify_marker(marker_path: Path, json_path: Path, md_path: Path) -> None:
    require_regular_file(marker_path, "comparison completion marker")
    expected = marker_content(json_path, md_path)
    actual = marker_path.read_text(encoding="utf-8")
    if actual != expected:
        raise FinalizerError(
            f"Comparison completion marker does not match output files: {marker_path}"
        )


def validate_output_set(
    output_dir: Path,
    *,
    marker_required: bool,
) -> Dict[str, Any]:
    json_path, md_path, marker_path = output_paths(output_dir)
    require_regular_file(json_path, "comparison JSON")
    require_regular_file(md_path, "comparison Markdown")
    payload = load_json_object(json_path)
    if payload.get("schema") != OUTPUT_SCHEMA:
        raise FinalizerError(
            f"Unexpected comparison schema: {payload.get('schema')!r}"
        )
    if payload.get("status") != "complete":
        raise FinalizerError("Comparison output status is not complete")
    scope = payload.get("scope")
    boundary = payload.get("decision_boundary")
    mainline_scope = payload.get("mainline_scope")
    next_module_gate = payload.get("next_module_gate")
    frozen_snapshot = payload.get("frozen_reference_snapshot")
    if not isinstance(scope, dict) or scope.get("official_test_accessed") is not False:
        raise FinalizerError("Comparison scope does not keep official_test_accessed=false")
    if not isinstance(boundary, dict):
        raise FinalizerError("Comparison output is missing decision_boundary")
    if not isinstance(mainline_scope, dict):
        raise FinalizerError("Comparison output is missing mainline_scope")
    if mainline_scope.get("mainline_before") != "TPD-v1" or mainline_scope.get("mainline_after") != "TPD-v1":
        raise FinalizerError("Comparison output changed the TPD-v1 mainline")
    if not isinstance(frozen_snapshot, dict) or frozen_snapshot.get("formal_decision") != "INCONCLUSIVE_MIXED_TRADEOFF":
        raise FinalizerError("Comparison output does not preserve the frozen formal decision")
    if not isinstance(next_module_gate, dict):
        raise FinalizerError("Comparison output is missing next_module_gate")
    if next_module_gate.get("gate_file_sha256") != file_sha256(NEXT_MODULE_GATE_JSON):
        raise FinalizerError("Comparison next-module gate digest mismatch")
    if next_module_gate.get("mainline_changed") is not False or next_module_gate.get("innovation_changed") is not False:
        raise FinalizerError("Next-module gate changed the mainline or innovation")
    expected_false = (
        "mainline_decision_made",
        "paper_core_established",
        "stability_claim_supported",
        "three_branch_necessity_established",
        "causal_mechanism_established",
        "mainline_changed",
    )
    for key in expected_false:
        if boundary.get(key) is not False:
            raise FinalizerError(f"Comparison decision_boundary.{key} must be false")
    conclusions = payload.get("candidate_conclusions")
    if not isinstance(conclusions, dict) or set(conclusions) != set(VARIANTS):
        raise FinalizerError("Comparison output does not contain exactly four candidates")
    if not md_path.read_text(encoding="utf-8").strip():
        raise FinalizerError("Comparison Markdown is empty")
    if marker_required:
        verify_marker(marker_path, json_path, md_path)
    return {
        "json": str(json_path),
        "json_sha256": file_sha256(json_path),
        "markdown": str(md_path),
        "markdown_sha256": file_sha256(md_path),
        "marker": str(marker_path),
        "marker_present": marker_path.is_file() and not marker_path.is_symlink(),
    }


def existing_output_status(output_dir: Path) -> str:
    json_path, md_path, marker_path = output_paths(output_dir)
    present = [path.exists() for path in (json_path, md_path, marker_path)]
    if not any(present):
        return "absent"
    if json_path.is_file() and md_path.is_file():
        return "complete_with_marker" if marker_path.is_file() else "complete_without_marker"
    return "partial"


def snapshot_reference_tree(root: Path) -> Dict[str, tuple[int, int]]:
    """Record file size and mtime without reading large checkpoint contents."""

    if not root.is_dir() or root.is_symlink():
        raise FinalizerError(f"Frozen reference root is unavailable: {root}")
    snapshot: Dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FinalizerError(f"Frozen reference tree contains a linked path: {path}")
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def snapshot_mainline_files() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in PROTECTED_MAINLINE_FILES:
        require_regular_file(path, "protected mainline file")
        result[str(path)] = file_sha256(path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for and finalize the four TPD-Clean screen800 candidates"
    )
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--candidate-run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--reference-run-name", default=DEFAULT_REFERENCE_RUN_NAME)
    parser.add_argument("--reference-miou-root", type=Path, default=None)
    parser.add_argument("--reference-sweep-device", default="cuda:0")
    parser.add_argument("--reference-sweep-timeout-seconds", type=int, default=3600)
    parser.add_argument("--aggregator", type=Path, default=DEFAULT_AGGREGATOR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--aggregate-timeout-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or args.poll_seconds > 3600:
        parser.error("--poll-seconds must be in [1, 3600]")
    if args.aggregate_timeout_seconds < 1:
        parser.error("--aggregate-timeout-seconds must be positive")
    if args.reference_sweep_timeout_seconds < 1:
        parser.error("--reference-sweep-timeout-seconds must be positive")
    if args.dry_run and args.status:
        parser.error("--dry-run and --status are mutually exclusive")
    args.candidate_root = args.candidate_root.resolve()
    args.reference_root = args.reference_root.resolve()
    args.aggregator = args.aggregator.resolve()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (args.candidate_root / DATASET / "comparison").resolve()
    )
    args.reference_miou_root = (
        args.reference_miou_root.resolve()
        if args.reference_miou_root is not None
        else (args.candidate_root / "frozen_reference_miou_runs").resolve()
    )
    args.state_file = (
        args.state_file.resolve()
        if args.state_file is not None
        else (args.candidate_root / "launch/finalizer_state.json").resolve()
    )
    args.lock_file = (
        args.lock_file.resolve()
        if args.lock_file is not None
        else (args.candidate_root / ".locks/finalizer.lock").resolve()
    )
    return args


def validate_paths(args: argparse.Namespace) -> None:
    if not args.candidate_root.is_dir() or args.candidate_root.is_symlink():
        raise FinalizerError(
            f"Candidate result root is unavailable: {args.candidate_root}"
        )
    if not args.reference_root.is_dir() or args.reference_root.is_symlink():
        raise FinalizerError(
            f"Frozen formal800 root is unavailable: {args.reference_root}"
        )
    require_regular_file(args.aggregator, "TPD-Clean summarizer")
    require_regular_file(BASE_EVALUATOR, "base Pd-Fa evaluator")
    require_regular_file(DATA_FINGERPRINT_SCRIPT, "training-data fingerprint script")
    require_regular_file(NEXT_MODULE_GATE_JSON, "next-module gate JSON")
    require_regular_file(NEXT_MODULE_GATE_MD, "next-module gate Markdown")
    for path, label in (
        (args.output_dir, "output directory"),
        (args.reference_miou_root, "reference mIoU mirror root"),
        (args.state_file, "state file"),
        (args.lock_file, "lock file"),
    ):
        if not is_within(path, args.candidate_root):
            raise FinalizerError(
                f"{label} must stay under the candidate result root: {path}"
            )
    if is_within(args.output_dir, args.reference_root):
        raise FinalizerError("Output directory cannot be inside frozen formal800")
    if args.candidate_root == args.reference_root:
        raise FinalizerError("Candidate and frozen reference roots must differ")


def read_previous_state(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json_object(path)
    if payload.get("schema") != STATE_SCHEMA:
        raise FinalizerError(f"Unexpected finalizer state schema in {path}")
    if payload.get("state") not in VALID_STATES:
        raise FinalizerError(f"Unexpected finalizer state value in {path}")
    return payload


def link_or_copy_file(source: Path, destination: Path) -> str:
    """Create a regular mirror file, preferring a hard link."""

    require_regular_file(source, "frozen reference source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require_regular_file(destination, "existing reference mirror artifact")
        if file_sha256(destination) != file_sha256(source):
            raise FinalizerError(
                f"Existing reference mirror differs from source: {destination}"
            )
        return "reused"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.copy.", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            if file_sha256(temporary) != file_sha256(source):
                raise FinalizerError(f"Reference mirror copy differs: {source}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return "copy"


def verify_training_data_fingerprint() -> Dict[str, Any]:
    require_regular_file(DATA_FINGERPRINT_SCRIPT, "training-data fingerprint script")
    command = [
        sys.executable,
        str(DATA_FINGERPRINT_SCRIPT),
        "--dataset",
        DATASET,
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=300, cwd=REPO_ROOT
    )
    if completed.returncode != 0:
        raise FinalizerError(
            f"Training-data fingerprint failed: {completed.stderr[-2000:].strip()}"
        )
    actual = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    if actual != EXPECTED_TRAINING_DATA_SHA256:
        raise FinalizerError(
            f"Training-data fingerprint mismatch: expected={EXPECTED_TRAINING_DATA_SHA256} actual={actual}"
        )
    return {
        "command": command,
        "expected_sha256": EXPECTED_TRAINING_DATA_SHA256,
        "actual_sha256": actual,
        "script_sha256": file_sha256(DATA_FINGERPRINT_SCRIPT),
    }


def validate_reference_sweep(path: Path, variant: str, mirror_run: Path) -> Dict[str, Any]:
    payload = load_json_object(path)
    expected_checkpoint = mirror_run / "best_miou.pth.tar"
    expected = {
        "variant": variant,
        "dataset": DATASET,
        "checkpoint_role": "best_validation_miou_secondary",
        "checkpoint_sha256": FROZEN_MIOU_CHECKPOINT_SHA256[variant],
        "official_test_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FinalizerError(f"{variant}: invalid reference sweep field {key}")
    if Path(str(payload.get("checkpoint"))).resolve() != expected_checkpoint.resolve():
        raise FinalizerError(f"{variant}: reference sweep checkpoint path mismatch")
    audit = payload.get("audit")
    if not isinstance(audit, dict) or audit.get("expected_epochs") != EXPECTED_EPOCHS:
        raise FinalizerError(f"{variant}: reference sweep epoch audit mismatch")
    flags = audit.get("integrity_checks_passed")
    if not isinstance(flags, dict) or not flags or not all(flags.values()):
        raise FinalizerError(f"{variant}: reference sweep checks are incomplete")
    configuration = payload.get("threshold_configuration")
    if not isinstance(configuration, dict) or configuration.get("fa_budgets") != [
        1e-6,
        5e-6,
        1e-5,
        5e-5,
        1e-4,
    ]:
        raise FinalizerError(f"{variant}: reference sweep Fa budgets mismatch")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
    }


def mirror_reference_sources(args: argparse.Namespace, variant: str) -> Dict[str, Any]:
    source_run = args.reference_root / DATASET / variant / args.reference_run_name
    mirror_run = args.reference_miou_root / DATASET / variant / args.reference_run_name
    if not source_run.is_dir() or source_run.is_symlink():
        raise FinalizerError(f"{variant}: frozen reference run is unavailable")
    if mirror_run.exists() and (not mirror_run.is_dir() or mirror_run.is_symlink()):
        raise FinalizerError(f"{variant}: invalid reference mirror run directory")
    mirror_run.mkdir(parents=True, exist_ok=True)
    modes: Dict[str, str] = {}
    for name in REFERENCE_MIRROR_FILES:
        source = source_run / name
        destination = mirror_run / name
        if name == "best_miou.pth.tar":
            actual = file_sha256(source)
            if actual != FROZEN_MIOU_CHECKPOINT_SHA256[variant]:
                raise FinalizerError(f"{variant}: frozen best-mIoU checkpoint mismatch")
        modes[name] = link_or_copy_file(source, destination)
    return {
        "source_run": str(source_run),
        "mirror_run": str(mirror_run),
        "mirror_modes": modes,
    }


def generate_frozen_reference_sweeps(args: argparse.Namespace) -> Dict[str, Any]:
    require_regular_file(BASE_EVALUATOR, "base Pd-Fa evaluator")
    results: Dict[str, Any] = {}
    for variant in REFERENCE_VARIANTS:
        mirror = mirror_reference_sources(args, variant)
        mirror_run = Path(mirror["mirror_run"])
        sweep_path = mirror_run / REFERENCE_SWEEP_NAME
        if sweep_path.exists():
            evaluation = {"reused": True}
        else:
            command = [
                sys.executable,
                str(BASE_EVALUATOR),
                "--run-dir",
                str(mirror_run),
                "--checkpoint",
                "best_miou.pth.tar",
                "--device",
                args.reference_sweep_device,
                "--expected-epochs",
                str(EXPECTED_EPOCHS),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.reference_sweep_timeout_seconds,
                cwd=REPO_ROOT,
            )
            evaluation = {
                "reused": False,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-8000:],
                "stderr": completed.stderr[-8000:],
            }
            if completed.returncode != 0:
                raise FinalizerError(
                    f"{variant}: reference best-mIoU sweep failed: {completed.stderr[-2000:].strip()}"
                )
        results[variant] = {
            **mirror,
            "evaluation": evaluation,
            "sweep": validate_reference_sweep(sweep_path, variant, mirror_run),
        }
    return {
        "evaluator": str(BASE_EVALUATOR),
        "evaluator_sha256": file_sha256(BASE_EVALUATOR),
        "variants": results,
    }


def _aggregator_subprocess_environment(
    environ: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """Preserve PYTHONPATH while making the repository importable by scripts."""

    environment = dict(os.environ if environ is None else environ)
    existing = environment.get("PYTHONPATH", "")
    entries = existing.split(os.pathsep) if existing else []

    def is_repo_root(entry: str) -> bool:
        if not entry:
            return False
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve() == REPO_ROOT

    retained = [entry for entry in entries if not is_repo_root(entry)]
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), *retained))
    return environment


def run_aggregator(args: argparse.Namespace) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(args.aggregator),
        "--candidate-root",
        str(args.candidate_root),
        "--reference-root",
        str(args.reference_root),
        "--candidate-run-name",
        args.candidate_run_name,
        "--reference-run-name",
        args.reference_run_name,
        "--reference-miou-root",
        str(args.reference_miou_root),
        "--output-dir",
        str(args.output_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.aggregate_timeout_seconds,
            cwd=REPO_ROOT,
            env=_aggregator_subprocess_environment(),
        )
    except subprocess.TimeoutExpired as error:
        raise FinalizerError(
            f"Summarizer exceeded {args.aggregate_timeout_seconds} seconds"
        ) from error
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
        "aggregator_sha256": file_sha256(args.aggregator),
    }
    if completed.returncode != 0:
        raise FinalizerError(
            "Summarizer failed with return code "
            f"{completed.returncode}: {completed.stderr[-2000:].strip()}"
        )
    return result


def status_mode(args: argparse.Namespace) -> int:
    previous = read_previous_state(args.state_file)
    if previous is not None:
        print(json.dumps(previous, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    observations = inspect_all(
        args.candidate_root,
        args.candidate_run_name,
        include_unit_state=True,
    )
    payload = state_payload(
        run_id="status-only",
        state=("validating_candidates" if observations_ready(observations) else "waiting"),
        candidate_root=args.candidate_root,
        reference_root=args.reference_root,
        reference_miou_root=args.reference_miou_root,
        aggregator=args.aggregator,
        output_dir=args.output_dir,
        poll_seconds=args.poll_seconds,
        observations=observations,
        message="No persisted finalizer state; reporting current observations.",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def dry_run_mode(args: argparse.Namespace) -> int:
    observations = inspect_all(
        args.candidate_root,
        args.candidate_run_name,
        include_unit_state=True,
    )
    problems = observation_problems(observations)
    state = "failed" if problems else (
        "validating_candidates" if observations_ready(observations) else "waiting"
    )
    payload = state_payload(
        run_id="dry-run",
        state=state,
        candidate_root=args.candidate_root,
        reference_root=args.reference_root,
        reference_miou_root=args.reference_miou_root,
        aggregator=args.aggregator,
        output_dir=args.output_dir,
        poll_seconds=args.poll_seconds,
        observations=observations,
        message=(
            "; ".join(problems)
            if problems
            else "Dry run only; no state or comparison files were written."
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def finalize(args: argparse.Namespace) -> int:
    validate_paths(args)
    run_id = uuid.uuid4().hex
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FinalizerError(
                f"Another finalizer instance holds {args.lock_file}"
            ) from error

        previous = read_previous_state(args.state_file)
        pinned_inputs = snapshot_finalizer_inputs(args.aggregator)
        if previous is not None and previous.get("state") in {
            "waiting",
            "validating_candidates",
            "generating_frozen_miou_reference_sweeps",
            "aggregating",
        }:
            if previous.get("input_snapshot") != pinned_inputs:
                raise FinalizerError(
                    "Finalizer inputs changed since the persistent run first recorded them"
                )
        existing = existing_output_status(args.output_dir)
        if existing == "partial":
            raise FinalizerError(
                f"Partial comparison outputs already exist under {args.output_dir}"
            )
        if existing in {"complete_with_marker", "complete_without_marker"}:
            output_record = validate_output_set(
                args.output_dir,
                marker_required=existing == "complete_with_marker",
            )
            json_path, md_path, marker_path = output_paths(args.output_dir)
            if existing == "complete_without_marker":
                atomic_write_text(marker_path, marker_content(json_path, md_path))
                output_record = validate_output_set(
                    args.output_dir, marker_required=True
                )
            observations = inspect_all(
                args.candidate_root,
                args.candidate_run_name,
                include_unit_state=True,
            )
            atomic_write_json(
                args.state_file,
                state_payload(
                    run_id=run_id,
                    state="complete",
                    candidate_root=args.candidate_root,
                    reference_root=args.reference_root,
                    reference_miou_root=args.reference_miou_root,
                    aggregator=args.aggregator,
                    output_dir=args.output_dir,
                    poll_seconds=args.poll_seconds,
                    observations=observations,
                    message="Validated and reused an existing complete comparison.",
                    aggregator_result=output_record,
                    reused_existing_outputs=True,
                ),
            )
            print(
                f"TPDCLEAN_FINALIZER_COMPLETE reused=true output_dir={args.output_dir}",
                flush=True,
            )
            return 0

        if previous is not None and previous.get("state") == "failed":
            if not args.retry_failed:
                raise FinalizerError(
                    "Previous finalizer state is failed; pass --retry-failed "
                    "after resolving the recorded cause"
                )

        while True:
            if snapshot_finalizer_inputs(args.aggregator) != pinned_inputs:
                raise FinalizerError("A pinned finalizer input changed while waiting")
            observations = inspect_all(
                args.candidate_root,
                args.candidate_run_name,
                include_unit_state=True,
            )
            problems = observation_problems(observations)
            if problems:
                atomic_write_json(
                    args.state_file,
                    state_payload(
                        run_id=run_id,
                        state="failed",
                        candidate_root=args.candidate_root,
                        reference_root=args.reference_root,
                        reference_miou_root=args.reference_miou_root,
                        aggregator=args.aggregator,
                        output_dir=args.output_dir,
                        poll_seconds=args.poll_seconds,
                        observations=observations,
                        message="; ".join(problems),
                    ),
                )
                raise FinalizerError("; ".join(problems))
            if observations_ready(observations):
                break
            atomic_write_json(
                args.state_file,
                state_payload(
                    run_id=run_id,
                    state="waiting",
                    candidate_root=args.candidate_root,
                    reference_root=args.reference_root,
                    reference_miou_root=args.reference_miou_root,
                    aggregator=args.aggregator,
                    output_dir=args.output_dir,
                    poll_seconds=args.poll_seconds,
                    observations=observations,
                    message="Waiting for four unique TPDCLEAN_COMPLETE records.",
                ),
            )
            if args.once:
                print(
                    "TPDCLEAN_FINALIZER_WAITING"
                    f" ready={sum(item.ready for item in observations.values())}/4",
                    flush=True,
                )
                return WAITING_EXIT_CODE
            time.sleep(args.poll_seconds)

        atomic_write_json(
            args.state_file,
            state_payload(
                run_id=run_id,
                state="validating_candidates",
                candidate_root=args.candidate_root,
                reference_root=args.reference_root,
                reference_miou_root=args.reference_miou_root,
                aggregator=args.aggregator,
                output_dir=args.output_dir,
                poll_seconds=args.poll_seconds,
                observations=observations,
                message="Four unique completions observed; validating inputs.",
            ),
        )
        # Re-read once in the candidate-validation state so aggregation never relies
        # on an earlier poll snapshot.
        observations = inspect_all(
            args.candidate_root,
            args.candidate_run_name,
            include_unit_state=True,
        )
        problems = observation_problems(observations)
        if problems or not observations_ready(observations):
            message = "; ".join(problems) or "Completion evidence changed during validation"
            raise FinalizerError(message)

        if snapshot_finalizer_inputs(args.aggregator) != pinned_inputs:
            raise FinalizerError("A pinned finalizer input changed before validation")
        fingerprint_result = verify_training_data_fingerprint()
        protected_mainline_before = snapshot_mainline_files()
        reference_before = snapshot_reference_tree(args.reference_root)
        finalizer_inputs_before = pinned_inputs
        atomic_write_json(
            args.state_file,
            state_payload(
                run_id=run_id,
                state="generating_frozen_miou_reference_sweeps",
                candidate_root=args.candidate_root,
                reference_root=args.reference_root,
                reference_miou_root=args.reference_miou_root,
                aggregator=args.aggregator,
                output_dir=args.output_dir,
                poll_seconds=args.poll_seconds,
                observations=observations,
                message="Generating SPD and TPD-v1 best-mIoU reference sweeps.",
                reference_sweep_result={"training_data_fingerprint": fingerprint_result},
            ),
        )
        reference_sweep_result = {
            "training_data_fingerprint": fingerprint_result,
            **generate_frozen_reference_sweeps(args),
        }
        atomic_write_json(
            args.state_file,
            state_payload(
                run_id=run_id,
                state="aggregating",
                candidate_root=args.candidate_root,
                reference_root=args.reference_root,
                reference_miou_root=args.reference_miou_root,
                aggregator=args.aggregator,
                output_dir=args.output_dir,
                poll_seconds=args.poll_seconds,
                observations=observations,
                message="Invoking the independent TPD-Clean summarizer.",
                reference_sweep_result=reference_sweep_result,
            ),
        )
        aggregator_result = run_aggregator(args)
        if snapshot_finalizer_inputs(args.aggregator) != finalizer_inputs_before:
            raise FinalizerError("A finalizer source/input file changed during aggregation")
        if snapshot_mainline_files() != protected_mainline_before:
            raise FinalizerError("A protected mainline file changed during aggregation")
        if snapshot_reference_tree(args.reference_root) != reference_before:
            raise FinalizerError("Frozen formal800 files changed during aggregation")

        output_record = validate_output_set(args.output_dir, marker_required=False)
        json_path, md_path, marker_path = output_paths(args.output_dir)
        atomic_write_text(marker_path, marker_content(json_path, md_path))
        output_record = validate_output_set(args.output_dir, marker_required=True)
        aggregator_result = {**aggregator_result, **output_record}
        atomic_write_json(
            args.state_file,
            state_payload(
                run_id=run_id,
                state="complete",
                candidate_root=args.candidate_root,
                reference_root=args.reference_root,
                reference_miou_root=args.reference_miou_root,
                aggregator=args.aggregator,
                output_dir=args.output_dir,
                poll_seconds=args.poll_seconds,
                observations=observations,
                message="TPD-Clean single-seed comparison completed.",
                aggregator_result=aggregator_result,
                reference_sweep_result=reference_sweep_result,
            ),
        )
        print(
            f"TPDCLEAN_FINALIZER_COMPLETE reused=false output_dir={args.output_dir}",
            flush=True,
        )
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.status:
            return status_mode(args)
        if args.dry_run:
            return dry_run_mode(args)
        return finalize(args)
    except Exception as error:
        # Failure state is best-effort here.  The normal validation paths write
        # richer observations before raising.
        if not args.dry_run and not args.status:
            try:
                previous = read_previous_state(args.state_file)
                observations = inspect_all(
                    args.candidate_root,
                    args.candidate_run_name,
                    include_unit_state=False,
                )
                if previous is None or previous.get("state") != "failed":
                    atomic_write_json(
                        args.state_file,
                        state_payload(
                            run_id=(
                                str(previous.get("run_id"))
                                if previous is not None
                                else uuid.uuid4().hex
                            ),
                            state="failed",
                            candidate_root=args.candidate_root,
                            reference_root=args.reference_root,
                            reference_miou_root=args.reference_miou_root,
                            aggregator=args.aggregator,
                            output_dir=args.output_dir,
                            poll_seconds=args.poll_seconds,
                            observations=observations,
                            message=str(error),
                        ),
                    )
            except Exception:
                pass
        print(f"TPDCLEAN_FINALIZER_FAILED reason={error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
