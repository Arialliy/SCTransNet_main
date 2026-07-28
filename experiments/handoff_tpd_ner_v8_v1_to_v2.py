#!/usr/bin/env python3
"""Conditional, idempotent handoff from the fixed-seed V1 result to V2.

The V1 completion marker is the sole commit signal.  A failed V1 model gate
starts (or reuses) one V2 training lane and one V2 postprocess wait service.
A passed V1 gate starts nothing.  The internal wait worker invokes the existing
frozen V2 postprocessor only after both required trajectories are complete.
This program never stops or alters V1.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_formal800 as v1_post,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_v2_formal800 as v2_post,
)


SCHEMA = "sctransnet_tpd_ner_v8_v1_to_v2_handoff_v1"
EXPECTED_TRAINING_SEED = 42
EXPECTED_SPLIT_SEED = 20260722
FULL_MODEL_GATE_PASSED = "FULL_MODEL_GATE_PASSED"
RETURN_TO_MODEL_OPTIMIZATION = "RETURN_TO_MODEL_OPTIMIZATION"
ALLOWED_DECISIONS = {
    FULL_MODEL_GATE_PASSED,
    RETURN_TO_MODEL_OPTIMIZATION,
}
V1_JSON = v1_post.JSON_OUTPUT
V1_MARKDOWN = v1_post.MARKDOWN_OUTPUT
V1_MARKER = v1_post.COMPLETE_MARKER
V2_LAUNCHER = (
    REPO_ROOT
    / "experiments/launch_tpd_ner_v8_mprs_dch_v2_formal800_1x5090.sh"
)
V2_LANE = (
    REPO_ROOT
    / "experiments/run_tpd_ner_v8_mprs_dch_v2_formal800_1x5090_lane.sh"
)
V2_POSTPROCESS = (
    REPO_ROOT
    / "experiments/postprocess_tpd_ner_v8_mprs_dch_v2_formal800.py"
)
V2_TRAINING_UNITS = {
    2: "sctransnet-tpd-ner-v8-v2-relay-on-gpu2",
    3: "sctransnet-tpd-ner-v8-v2-relay-on-gpu3",
}
V2_POSTPROCESS_UNIT = "sctransnet-tpd-ner-v8-v2-postprocess"
V2_GPU_UUIDS = {
    int(physical_gpu): gpu_uuid
    for physical_gpu, gpu_uuid in v2_post.GPU_UUIDS.items()
}
_runtime_root = Path(
    os.environ.get(
        "XDG_RUNTIME_DIR",
        str(Path(tempfile.gettempdir()) / f"codex-runtime-{os.getuid()}"),
    )
)
DEFAULT_LOCK_PATH = _runtime_root / "sctransnet-tpd-ner-v8-v1-to-v2.lock"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _python_command_path(python: Path) -> Path:
    """Return an absolute command path without resolving virtualenv symlinks."""

    expanded = Path(python).expanduser()
    command_path = Path(os.path.abspath(os.fspath(expanded)))
    _require(
        command_path.is_file() and os.access(command_path, os.X_OK),
        f"Python command is not executable: {command_path}",
    )
    return command_path


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _regular_bytes(path: Path, label: str) -> bytes:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.read_bytes()


def _json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_report_identity(report: Mapping[str, Any]) -> str:
    for name, expected in {
        "schema": v1_post.SCHEMA,
        "status": "complete",
        "dataset": v1_post.DATASET,
        "training_seed": EXPECTED_TRAINING_SEED,
        "split_seed": EXPECTED_SPLIT_SEED,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
    }.items():
        _require(
            report.get(name) == expected,
            f"V1 report identity differs: {name}",
        )
    decision = report.get("decision")
    _require(
        decision in ALLOWED_DECISIONS,
        f"V1 report decision is unsupported: {decision!r}",
    )
    aggregate = report.get("aggregate_full_model_gate_passed")
    absolute = report.get("all_four_absolute_checkpoint_gates_passed")
    paired = report.get("both_role_paired_relay_on_gates_passed")
    for name, value in {
        "aggregate_full_model_gate_passed": aggregate,
        "all_four_absolute_checkpoint_gates_passed": absolute,
        "both_role_paired_relay_on_gates_passed": paired,
    }.items():
        _require(type(value) is bool, f"V1 report field is not boolean: {name}")
    _require(
        aggregate == bool(absolute and paired),
        "V1 report aggregate result differs from its component gates",
    )
    _require(
        (decision == FULL_MODEL_GATE_PASSED) == aggregate,
        "V1 report decision differs from aggregate result",
    )
    rows = report.get("rows")
    _require(
        isinstance(rows, list) and len(rows) == 6,
        "V1 report must contain the six-row comparison matrix",
    )
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"V1 report row {index} is invalid")
        _require(
            row.get("seed") == EXPECTED_TRAINING_SEED,
            f"V1 report row {index} seed differs",
        )
        _require(
            row.get("split_seed") == EXPECTED_SPLIT_SEED,
            f"V1 report row {index} split seed differs",
        )
    readiness = report.get("readiness_binding")
    _require(
        isinstance(readiness, Mapping)
        and readiness.get("training_seed") == EXPECTED_TRAINING_SEED
        and readiness.get("split_seed") == EXPECTED_SPLIT_SEED
        and readiness.get("both_runs_complete") is True,
        "V1 report readiness binding differs",
    )
    return str(decision)


def _require_markdown_identity(
    markdown: str,
    *,
    decision: str,
    aggregate: bool,
) -> None:
    expected_lines = {
        f"- Decision: `{decision}`",
        (
            "- Aggregate full-model gate passed: "
            f"`{str(aggregate).lower()}`"
        ),
        "- Scope: seed 42, NUDT-SIRST internal 530/133 validation",
        "- Official test accessed: `false`",
    }
    lines = markdown.splitlines()
    for expected in expected_lines:
        _require(
            lines.count(expected) == 1,
            f"V1 Markdown identity line differs: {expected}",
        )
    other_decision = (
        RETURN_TO_MODEL_OPTIMIZATION
        if decision == FULL_MODEL_GATE_PASSED
        else FULL_MODEL_GATE_PASSED
    )
    _require(
        f"- Decision: `{other_decision}`" not in lines,
        "V1 Markdown contains conflicting decisions",
    )


def _rebuild_v1_report() -> dict[str, Any]:
    """Recompute the complete V1 report from current read-only artifacts."""

    locks = v1_post.verify_frozen_manifests()
    baseline_contract = v1_post._same_split_and_training_contract()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for checkpoint in v1_post.CHECKPOINTS:
        for variant in v1_post.VARIANTS:
            binding = v1_post.current_sweep_binding(
                variant=variant,
                checkpoint=checkpoint,
            )
            rows[(variant, checkpoint)] = v1_post.validate_existing_sweep(
                v1_post.sweep_path(v1_post.RUN_DIRS[variant], checkpoint),
                variant=variant,
                checkpoint=checkpoint,
                binding=binding,
            )
        baseline_binding = v2_post.current_reference_binding(
            v2_post.BASELINE_VARIANT,
            checkpoint,
        )
        rows[(v2_post.BASELINE_VARIANT, checkpoint)] = (
            v1_post.validate_existing_sweep(
                v1_post.sweep_path(v1_post.BASELINE_VIEW_RUN, checkpoint),
                variant=v2_post.BASELINE_VARIANT,
                checkpoint=checkpoint,
                binding=baseline_binding,
            )
        )
    rebuilt = v1_post.build_report(
        rows,
        lock_bindings=locks,
        baseline_contract=baseline_contract,
    )
    rebuilt["readiness_binding"] = v1_post.inspect_training_readiness()
    return rebuilt


def validate_v1_triplet(
    json_path: Path = V1_JSON,
    markdown_path: Path = V1_MARKDOWN,
    marker_path: Path = V1_MARKER,
) -> dict[str, Any]:
    """Validate the committed V1 report without changing any artifact."""

    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    marker_path = Path(marker_path)
    json_bytes = _regular_bytes(json_path, "V1 JSON report")
    markdown_bytes = _regular_bytes(markdown_path, "V1 Markdown report")
    marker_bytes = _regular_bytes(marker_path, "V1 completion marker")
    report = _json_object(json_bytes, "V1 JSON report")
    marker = _json_object(marker_bytes, "V1 completion marker")
    rebuilt = _rebuild_v1_report()
    _require(report == rebuilt, "V1 JSON differs from rebuilt report")
    _require(
        json_bytes == v1_post._canonical_bytes(rebuilt),
        "V1 JSON is not the canonical postprocess output",
    )
    decision = _require_report_identity(report)
    aggregate = report["aggregate_full_model_gate_passed"]

    for name, expected in {
        "schema": v1_post.COMPLETE_MARKER_SCHEMA
        if hasattr(v1_post, "COMPLETE_MARKER_SCHEMA")
        else "sctransnet_tpd_ner_v8_mprs_dch_postprocess_complete_v1",
        "status": "complete",
        "decision": decision,
        "aggregate_full_model_gate_passed": aggregate,
    }.items():
        _require(
            marker.get(name) == expected,
            f"V1 marker identity differs: {name}",
        )
    outputs = marker.get("outputs")
    _require(isinstance(outputs, Mapping), "V1 marker output hashes are missing")
    expected_output_keys = {json_path.name, markdown_path.name}
    _require(
        set(outputs) == expected_output_keys,
        "V1 marker output filenames differ",
    )
    actual_hashes = {
        json_path.name: sha256_bytes(json_bytes),
        markdown_path.name: sha256_bytes(markdown_bytes),
    }
    _require(
        dict(outputs) == actual_hashes,
        "V1 marker output hashes differ from current report files",
    )
    try:
        markdown = markdown_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("V1 Markdown report is not UTF-8") from exc
    _require_markdown_identity(
        markdown,
        decision=decision,
        aggregate=aggregate,
    )
    _require(
        markdown_bytes == v1_post.render_markdown(rebuilt).encode("utf-8"),
        "V1 Markdown differs from the rendered JSON report",
    )
    _require(
        marker_bytes
        == v1_post._completion_marker_bytes(
            rebuilt,
            json_bytes,
            markdown_bytes,
        ),
        "V1 completion marker is not the canonical postprocess marker",
    )
    return {
        "status": "ready",
        "decision": decision,
        "training_seed": EXPECTED_TRAINING_SEED,
        "split_seed": EXPECTED_SPLIT_SEED,
        "multi_seed_scheduled": False,
        "aggregate_full_model_gate_passed": aggregate,
        "paths": {
            "json": str(json_path.absolute()),
            "markdown": str(markdown_path.absolute()),
            "marker": str(marker_path.absolute()),
        },
        "sha256": {
            "json": actual_hashes[json_path.name],
            "markdown": actual_hashes[markdown_path.name],
            "marker": sha256_bytes(marker_bytes),
        },
        "v1_artifacts_modified": False,
    }


def inspect_v1_triplet(
    json_path: Path = V1_JSON,
    markdown_path: Path = V1_MARKDOWN,
    marker_path: Path = V1_MARKER,
) -> dict[str, Any]:
    """Treat an absent marker as waiting; a present marker must be valid."""

    marker = Path(marker_path)
    if not marker.exists() and not marker.is_symlink():
        paths = {
            "json": Path(json_path),
            "markdown": Path(markdown_path),
            "marker": marker,
        }
        return {
            "status": "waiting_for_v1_commit",
            "decision": None,
            "training_seed": EXPECTED_TRAINING_SEED,
            "split_seed": EXPECTED_SPLIT_SEED,
            "multi_seed_scheduled": False,
            "exists": {
                name: path.is_file() and not path.is_symlink()
                for name, path in paths.items()
            },
        }
    return validate_v1_triplet(json_path, markdown_path, marker_path)


def _default_runner(
    command: Sequence[str],
    *,
    check: bool,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def unit_state(unit: str, *, runner: Runner = _default_runner) -> str:
    result = runner(
        [
            "systemctl",
            "--user",
            "show",
            f"{unit}.service",
            "--property=ActiveState",
            "--value",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return "not-found"
    return (result.stdout or "").strip() or "unknown"


def unit_exec_start(unit: str, *, runner: Runner = _default_runner) -> str:
    result = runner(
        [
            "systemctl",
            "--user",
            "show",
            f"{unit}.service",
            "--property=ExecStart",
            "--value",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _fixed_v2_lane_contract() -> dict[str, Any]:
    lock_path = Path(v2_post.ACCEPTANCE_LOCK)
    lock_bytes = _regular_bytes(lock_path, "V2 acceptance source lock")
    payload = _json_object(lock_bytes, "V2 acceptance source lock")
    policy = payload.get("policy")
    sources = payload.get("source_sha256")
    lane_relative = str(V2_LANE.resolve().relative_to(REPO_ROOT))
    handoff_path = Path(__file__).resolve()
    handoff_relative = str(handoff_path.relative_to(REPO_ROOT))
    _require(
        payload.get("schema") == v2_post.v2_freeze.ACCEPTANCE_SCHEMA
        and payload.get("lock_kind") == "acceptance"
        and payload.get("variants") == [v2_post.VARIANT_V2_ON],
        "V2 acceptance source-lock identity differs",
    )
    _require(
        isinstance(policy, Mapping)
        and policy.get("training_seed") == EXPECTED_TRAINING_SEED
        and policy.get("split_seed") == EXPECTED_SPLIT_SEED
        and policy.get("multi_seed_scheduled") is False
        and policy.get("official_test_accessed") is False,
        "V2 fixed-seed training policy differs",
    )
    _require(
        isinstance(sources, Mapping)
        and sources.get(lane_relative)
        == sha256_bytes(_regular_bytes(V2_LANE, "V2 training lane")),
        "V2 training lane differs from its source lock",
    )
    _require(
        sources.get(handoff_relative)
        == sha256_bytes(_regular_bytes(handoff_path, "V1-to-V2 handoff")),
        "V1-to-V2 handoff differs from its source lock",
    )
    return {
        "acceptance_source_lock": str(lock_path.resolve()),
        "acceptance_source_lock_sha256": sha256_bytes(lock_bytes),
        "lane_sha256": sources[lane_relative],
        "handoff_sha256": sources[handoff_relative],
        "variant": v2_post.VARIANT_V2_ON,
        "training_seed": EXPECTED_TRAINING_SEED,
        "split_seed": EXPECTED_SPLIT_SEED,
        "multi_seed_scheduled": False,
    }


def _training_unit_identity(
    unit: str,
    *,
    runner: Runner,
) -> dict[str, Any]:
    inverse = {name: gpu for gpu, name in V2_TRAINING_UNITS.items()}
    _require(unit in inverse, "active V2 training unit name differs")
    physical_gpu = inverse[unit]
    exec_start = unit_exec_start(unit, runner=runner)
    expected_argv = (
        f"argv[]={V2_LANE.resolve()} {physical_gpu} "
        f"{V2_GPU_UUIDS[physical_gpu]} ;"
    )
    _require(
        expected_argv in exec_start,
        "active V2 training unit ExecStart differs",
    )
    contract = _fixed_v2_lane_contract()
    return {
        "unit": unit,
        "physical_gpu": physical_gpu,
        "physical_gpu_uuid": V2_GPU_UUIDS[physical_gpu],
        "variant": v2_post.VARIANT_V2_ON,
        "training_seed": EXPECTED_TRAINING_SEED,
        "source_contract": contract,
        "exec_start": exec_start,
        "identity_verified": True,
    }


def _postprocess_unit_identity(
    *,
    physical_gpu: int,
    python: Path,
    runner: Runner,
) -> dict[str, Any]:
    exec_start = unit_exec_start(V2_POSTPROCESS_UNIT, runner=runner)
    python_command = _python_command_path(python)
    expected_prefix = (
        f"argv[]={python_command} {Path(__file__).resolve()} "
        "--v2-postprocess-worker "
    )
    _require(
        expected_prefix in exec_start
        and f"--physical-gpu {physical_gpu}" in exec_start
        and f"--python {python_command}" in exec_start,
        "active V2 postprocess wait unit ExecStart differs",
    )
    return {
        "unit": V2_POSTPROCESS_UNIT,
        "physical_gpu": physical_gpu,
        "training_seed": EXPECTED_TRAINING_SEED,
        "worker": str(Path(__file__).resolve()),
        "exec_start": exec_start,
        "identity_verified": True,
    }


def _active_training_unit(
    *,
    runner: Runner,
) -> tuple[str | None, dict[str, str]]:
    states = {
        name: unit_state(name, runner=runner)
        for name in V2_TRAINING_UNITS.values()
    }
    active = [
        name
        for name, state in states.items()
        if state in {"active", "activating"}
    ]
    _require(
        len(active) <= 1,
        "more than one V2 training unit is active",
    )
    return (active[0] if active else None), states


def v2_training_complete() -> bool:
    return bool(v2_post.inspect_v2_progress()["complete"])


def _rebuild_v2_report() -> dict[str, Any]:
    """Recompute the complete V2 report from current read-only artifacts."""

    readiness = v2_post.inspect_training_readiness()
    _require(
        readiness.get("required_runs_complete") is True,
        "V2 postprocess report cannot be complete before both runs complete",
    )
    locks = v2_post.verify_frozen_manifests()
    reference_before = v2_post.reference_snapshot()
    comparison_contract = v2_post.same_split_and_training_contract()
    rows = v2_post.load_all_rows()
    reference_after = v2_post.reference_snapshot()
    rebuilt = v2_post.build_report(
        rows,
        lock_bindings=locks,
        comparison_contract=comparison_contract,
        reference_before=reference_before,
        reference_after=reference_after,
    )
    rebuilt["readiness_binding"] = readiness
    return rebuilt


def _require_v2_report_identity(report: Mapping[str, Any]) -> tuple[str, bool]:
    for name, expected in {
        "schema": v2_post.SCHEMA,
        "status": "complete",
        "dataset": v2_post.DATASET,
        "training_seed": EXPECTED_TRAINING_SEED,
        "split_seed": EXPECTED_SPLIT_SEED,
        "multi_seed_scheduled": False,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
    }.items():
        _require(
            report.get(name) == expected,
            f"V2 report identity differs: {name}",
        )
    decision = report.get("decision")
    aggregate = report.get("aggregate_full_model_gate_passed")
    _require(
        decision in ALLOWED_DECISIONS,
        f"V2 report decision is unsupported: {decision!r}",
    )
    _require(
        type(aggregate) is bool,
        "V2 report aggregate gate result is not boolean",
    )
    _require(
        (decision == FULL_MODEL_GATE_PASSED) is aggregate,
        "V2 report decision differs from aggregate result",
    )
    components = report.get("success_components")
    _require(
        isinstance(components, Mapping),
        "V2 report success components are missing",
    )
    decisive_names = (
        "v2_on_pd_primary_absolute",
        "v2_on_miou_secondary_absolute",
        "pd_primary_paired_v2_on_vs_v1_off",
        "miou_secondary_paired_v2_on_vs_v1_off",
    )
    decisive: list[bool] = []
    for name in decisive_names:
        value = components.get(name)
        _require(
            type(value) is bool,
            f"V2 report success component is not boolean: {name}",
        )
        decisive.append(value)
    _require(
        aggregate is all(decisive),
        "V2 report aggregate result differs from its success components",
    )
    _require(
        components.get("v1_off_absolute_gate_required") is False
        and components.get("baseline_affects_decision") is False,
        "V2 report decision boundary differs",
    )
    readiness = report.get("readiness_binding")
    _require(
        isinstance(readiness, Mapping)
        and readiness.get("training_seed") == EXPECTED_TRAINING_SEED
        and readiness.get("split_seed") == EXPECTED_SPLIT_SEED
        and readiness.get("required_runs_complete") is True,
        "V2 report readiness binding differs",
    )
    bindings = report.get("bindings")
    _require(isinstance(bindings, Mapping), "V2 report bindings are missing")
    sweeps = bindings.get("sweeps")
    expected_sweeps = {
        f"{variant}:{checkpoint}"
        for variant in (
            v2_post.BASELINE_VARIANT,
            v2_post.VARIANT_V1_OFF,
            v2_post.VARIANT_V2_ON,
        )
        for checkpoint in v2_post.CHECKPOINTS
    }
    _require(
        isinstance(sweeps, Mapping) and set(sweeps) == expected_sweeps,
        "V2 report sweep bindings differ",
    )
    return str(decision), aggregate


def validate_v2_triplet(
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    marker_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and rebuild the committed V2 triplet without changing it."""

    json_path = (
        v2_post.JSON_OUTPUT if json_path is None else Path(json_path)
    )
    markdown_path = (
        v2_post.MARKDOWN_OUTPUT
        if markdown_path is None
        else Path(markdown_path)
    )
    marker_path = (
        v2_post.COMPLETE_MARKER
        if marker_path is None
        else Path(marker_path)
    )
    json_bytes = _regular_bytes(json_path, "V2 JSON report")
    markdown_bytes = _regular_bytes(markdown_path, "V2 Markdown report")
    marker_bytes = _regular_bytes(marker_path, "V2 completion marker")
    report = _json_object(json_bytes, "V2 JSON report")
    marker = _json_object(marker_bytes, "V2 completion marker")
    rebuilt = _rebuild_v2_report()
    _require(report == rebuilt, "V2 JSON differs from rebuilt report")
    _require(
        json_bytes == v2_post._canonical_bytes(rebuilt),
        "V2 JSON is not the canonical postprocess output",
    )
    decision, aggregate = _require_v2_report_identity(report)
    for name, expected in {
        "schema": v2_post.COMPLETE_MARKER_SCHEMA,
        "status": "complete",
        "decision": decision,
        "aggregate_full_model_gate_passed": aggregate,
    }.items():
        _require(
            marker.get(name) == expected,
            f"V2 completion marker identity differs: {name}",
        )
    outputs = marker.get("outputs")
    _require(isinstance(outputs, Mapping), "V2 completion output hashes missing")
    expected_output_keys = {json_path.name, markdown_path.name}
    _require(
        set(outputs) == expected_output_keys,
        "V2 completion output filenames differ",
    )
    actual_hashes = {
        json_path.name: sha256_bytes(json_bytes),
        markdown_path.name: sha256_bytes(markdown_bytes),
    }
    _require(
        dict(outputs) == actual_hashes,
        "V2 completion output hashes differ",
    )
    _require(
        markdown_bytes == v2_post.render_markdown(rebuilt).encode("utf-8"),
        "V2 Markdown differs from the rebuilt report",
    )
    _require(
        marker_bytes
        == v2_post._completion_marker_bytes(
            rebuilt,
            json_bytes,
            markdown_bytes,
        ),
        "V2 completion marker is not the canonical postprocess marker",
    )
    return {
        "status": "ready",
        "decision": decision,
        "training_seed": EXPECTED_TRAINING_SEED,
        "split_seed": EXPECTED_SPLIT_SEED,
        "multi_seed_scheduled": False,
        "aggregate_full_model_gate_passed": aggregate,
        "sha256": {
            "json": actual_hashes[json_path.name],
            "markdown": actual_hashes[markdown_path.name],
            "marker": sha256_bytes(marker_bytes),
        },
        "v2_artifacts_modified": False,
    }


def v2_postprocess_complete() -> bool:
    marker_path = v2_post.COMPLETE_MARKER
    if not marker_path.exists() and not marker_path.is_symlink():
        return False
    try:
        validate_v2_triplet()
    except (FileNotFoundError, ValueError):
        # The worker treats every stale, incomplete, or inconsistent triplet
        # as unfinished and delegates repair/quarantine to postprocess --run-now.
        return False
    return True


@contextlib.contextmanager
def handoff_lock(path: Path = DEFAULT_LOCK_PATH) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _launch_v2(
    physical_gpu: int,
    *,
    runner: Runner,
) -> list[str]:
    command = [
        str(V2_LAUNCHER.resolve()),
        "--physical-gpu",
        str(physical_gpu),
    ]
    runner(command, check=True, capture_output=False)
    return command


def _start_v2_postprocess_wait_service(
    physical_gpu: int,
    poll_seconds: float,
    python: Path,
    *,
    runner: Runner,
) -> list[str]:
    python_command = _python_command_path(python)
    command = [
        "systemd-run",
        "--user",
        "--unit",
        V2_POSTPROCESS_UNIT,
        "--collect",
        "--property=Type=exec",
        "--property=Restart=on-failure",
        "--property=RestartSec=10",
        str(python_command),
        str(Path(__file__).resolve()),
        "--v2-postprocess-worker",
        "--poll-seconds",
        f"{poll_seconds:g}",
        "--physical-gpu",
        str(physical_gpu),
        "--python",
        str(python_command),
    ]
    runner(command, check=True, capture_output=False)
    return command


def _run_v2_postprocess_once(
    physical_gpu: int,
    python: Path,
    *,
    runner: Runner,
) -> list[str]:
    python_command = _python_command_path(python)
    command = [
        str(python_command),
        str(V2_POSTPROCESS.resolve()),
        "--run-now",
        "--python",
        str(python_command),
        "--device-mode",
        "gpu23",
        "--physical-gpu",
        str(physical_gpu),
    ]
    runner(command, check=True, capture_output=False)
    return command


def wait_for_v2_and_postprocess(
    *,
    physical_gpu: int = 2,
    poll_seconds: float = 30.0,
    python: Path = Path(sys.executable),
    runner: Runner = _default_runner,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Internal service worker: wait read-only, then run V2 postprocess."""

    if physical_gpu not in V2_TRAINING_UNITS:
        raise ValueError("physical GPU must be 2 or 3")
    if not math.isfinite(poll_seconds) or not 1.0 <= poll_seconds <= 60.0:
        raise ValueError("poll_seconds must lie in [1, 60]")
    if v2_postprocess_complete():
        return {
            "schema": SCHEMA,
            "status": "complete",
            "action": "v2_postprocess_already_complete",
            "training_seed": EXPECTED_TRAINING_SEED,
            "split_seed": EXPECTED_SPLIT_SEED,
            "multi_seed_scheduled": False,
            "postprocess_command": None,
        }

    last_counts: tuple[int, int] | None = None
    while True:
        readiness = v2_post.inspect_training_readiness()
        _require(
            readiness.get("training_seed") == EXPECTED_TRAINING_SEED
            and readiness.get("split_seed") == EXPECTED_SPLIT_SEED,
            "V2 readiness identity differs",
        )
        counts = (
            int(
                readiness["v1_off_read_only_control"]["metrics"][
                    "event_count"
                ]
            ),
            int(readiness["v2_on"]["metrics"]["event_count"]),
        )
        if counts != last_counts:
            print(
                f"WAIT v1_off={counts[0]}/{v2_post.EXPECTED_EPOCHS} "
                f"v2_on={counts[1]}/{v2_post.EXPECTED_EPOCHS}",
                flush=True,
            )
            last_counts = counts
        if readiness.get("required_runs_complete") is True:
            break
        sleep_fn(poll_seconds)

    if v2_postprocess_complete():
        command = None
        action = "v2_postprocess_completed_while_waiting"
    else:
        command = _run_v2_postprocess_once(
            physical_gpu,
            python,
            runner=runner,
        )
        action = "v2_postprocess_called"
    return {
        "schema": SCHEMA,
        "status": "complete",
        "action": action,
        "training_seed": EXPECTED_TRAINING_SEED,
        "split_seed": EXPECTED_SPLIT_SEED,
        "multi_seed_scheduled": False,
        "physical_gpu": physical_gpu,
        "postprocess_command": command,
        "v1_tasks_modified": False,
    }


def execute_handoff(
    *,
    json_path: Path = V1_JSON,
    markdown_path: Path = V1_MARKDOWN,
    marker_path: Path = V1_MARKER,
    physical_gpu: int = 2,
    poll_seconds: float = 30.0,
    python: Path = Path(sys.executable),
    lock_path: Path = DEFAULT_LOCK_PATH,
    runner: Runner = _default_runner,
) -> dict[str, Any]:
    if physical_gpu not in V2_TRAINING_UNITS:
        raise ValueError("physical GPU must be 2 or 3")
    if not math.isfinite(poll_seconds) or not 1.0 <= poll_seconds <= 60.0:
        raise ValueError("poll_seconds must lie in [1, 60]")
    with handoff_lock(lock_path):
        evidence = validate_v1_triplet(
            json_path,
            markdown_path,
            marker_path,
        )
        decision = evidence["decision"]
        if decision == FULL_MODEL_GATE_PASSED:
            return {
                "schema": SCHEMA,
                "status": "complete",
                "decision": decision,
                "action": "v2_not_started_v1_gate_passed",
                "v1_evidence": evidence,
                "v2_launcher_called": False,
                "v2_postprocess_service_started": False,
                "v1_tasks_modified": False,
                "training_seed": EXPECTED_TRAINING_SEED,
                "split_seed": EXPECTED_SPLIT_SEED,
                "multi_seed_scheduled": False,
            }

        _require(
            decision == RETURN_TO_MODEL_OPTIMIZATION,
            "unsupported V1 handoff decision",
        )
        v2_source_contract = _fixed_v2_lane_contract()
        active_unit, training_states = _active_training_unit(runner=runner)
        active_training_identity = (
            None
            if active_unit is None
            else _training_unit_identity(active_unit, runner=runner)
        )
        training_complete = v2_training_complete()
        launcher_command: list[str] | None = None
        if active_unit is not None:
            training_action = "reused_active_v2_training"
        elif training_complete:
            training_action = "v2_training_already_complete"
        else:
            launcher_command = _launch_v2(physical_gpu, runner=runner)
            training_action = "v2_launcher_called"

        postprocess_complete = v2_postprocess_complete()
        postprocess_state = unit_state(V2_POSTPROCESS_UNIT, runner=runner)
        postprocess_identity: dict[str, Any] | None = None
        postprocess_command: list[str] | None = None
        if postprocess_complete:
            postprocess_action = "v2_postprocess_already_complete"
        elif postprocess_state in {"active", "activating"}:
            postprocess_identity = _postprocess_unit_identity(
                physical_gpu=physical_gpu,
                python=python,
                runner=runner,
            )
            postprocess_action = "reused_v2_postprocess_wait_service"
        else:
            postprocess_command = _start_v2_postprocess_wait_service(
                physical_gpu,
                poll_seconds,
                python,
                runner=runner,
            )
            postprocess_action = "v2_postprocess_wait_service_started"

        return {
            "schema": SCHEMA,
            "status": "complete",
            "decision": decision,
            "action": "v2_optimization_handoff_ready",
            "physical_gpu": physical_gpu,
            "training_seed": EXPECTED_TRAINING_SEED,
            "split_seed": EXPECTED_SPLIT_SEED,
            "multi_seed_scheduled": False,
            "v2_source_contract": v2_source_contract,
            "v1_evidence": evidence,
            "v2_training": {
                "action": training_action,
                "active_unit": active_unit,
                "active_unit_identity": active_training_identity,
                "unit_states": training_states,
                "training_complete": training_complete,
                "launcher_command": launcher_command,
            },
            "v2_postprocess": {
                "action": postprocess_action,
                "unit": V2_POSTPROCESS_UNIT,
                "unit_state_before": postprocess_state,
                "active_unit_identity": postprocess_identity,
                "postprocess_complete": postprocess_complete,
                "service_command": postprocess_command,
            },
            "v2_launcher_called": launcher_command is not None,
            "v2_postprocess_service_started": (
                postprocess_command is not None
            ),
            "v1_tasks_modified": False,
        }


def status_payload(
    *,
    json_path: Path = V1_JSON,
    markdown_path: Path = V1_MARKDOWN,
    marker_path: Path = V1_MARKER,
    runner: Runner = _default_runner,
) -> dict[str, Any]:
    evidence = inspect_v1_triplet(json_path, markdown_path, marker_path)
    active_unit, training_states = _active_training_unit(runner=runner)
    postprocess_state = unit_state(V2_POSTPROCESS_UNIT, runner=runner)
    return {
        "schema": SCHEMA,
        "status": "inspection_complete",
        "training_seed": EXPECTED_TRAINING_SEED,
        "split_seed": EXPECTED_SPLIT_SEED,
        "multi_seed_scheduled": False,
        "v1": evidence,
        "v2": {
            "active_training_unit": active_unit,
            "training_unit_states": training_states,
            "postprocess_unit": V2_POSTPROCESS_UNIT,
            "postprocess_unit_state": postprocess_state,
        },
        "mutations_performed": False,
    }


def wait_and_run(
    *,
    json_path: Path = V1_JSON,
    markdown_path: Path = V1_MARKDOWN,
    marker_path: Path = V1_MARKER,
    physical_gpu: int = 2,
    poll_seconds: float = 30.0,
    python: Path = Path(sys.executable),
    lock_path: Path = DEFAULT_LOCK_PATH,
    runner: Runner = _default_runner,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    last_state: tuple[tuple[str, bool], ...] | None = None
    while True:
        evidence = inspect_v1_triplet(
            json_path,
            markdown_path,
            marker_path,
        )
        if evidence["status"] == "ready":
            break
        exists = tuple(sorted(evidence["exists"].items()))
        if exists != last_state:
            print(
                "WAIT "
                + " ".join(
                    f"{name}={str(value).lower()}"
                    for name, value in exists
                ),
                flush=True,
            )
            last_state = exists
        sleep_fn(poll_seconds)
    return execute_handoff(
        json_path=json_path,
        markdown_path=markdown_path,
        marker_path=marker_path,
        physical_gpu=physical_gpu,
        poll_seconds=poll_seconds,
        python=python,
        lock_path=lock_path,
        runner=runner,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditional V1-to-V2 NER handoff"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--wait-and-run", action="store_true")
    mode.add_argument(
        "--v2-postprocess-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        choices=(2, 3),
        default=2,
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--v1-json", type=Path, default=V1_JSON)
    parser.add_argument("--v1-markdown", type=Path, default=V1_MARKDOWN)
    parser.add_argument("--v1-marker", type=Path, default=V1_MARKER)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.poll_seconds)
        or not 1.0 <= args.poll_seconds <= 60.0
    ):
        parser.error("--poll-seconds must lie in [1, 60]")
    return args


def _print_json(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    common = {
        "json_path": args.v1_json,
        "markdown_path": args.v1_markdown,
        "marker_path": args.v1_marker,
    }
    if args.status:
        _print_json(status_payload(**common))
        return
    if args.v2_postprocess_worker:
        _print_json(
            wait_for_v2_and_postprocess(
                physical_gpu=args.physical_gpu,
                poll_seconds=args.poll_seconds,
                python=args.python,
            )
        )
        return
    result = wait_and_run(
        **common,
        physical_gpu=args.physical_gpu,
        poll_seconds=args.poll_seconds,
        python=args.python,
    )
    _print_json(result)


__all__ = [
    "ALLOWED_DECISIONS",
    "DEFAULT_LOCK_PATH",
    "FULL_MODEL_GATE_PASSED",
    "RETURN_TO_MODEL_OPTIMIZATION",
    "SCHEMA",
    "V2_POSTPROCESS_UNIT",
    "V2_TRAINING_UNITS",
    "execute_handoff",
    "inspect_v1_triplet",
    "parse_args",
    "status_payload",
    "unit_state",
    "validate_v1_triplet",
    "validate_v2_triplet",
    "v2_postprocess_complete",
    "v2_training_complete",
    "wait_and_run",
    "wait_for_v2_and_postprocess",
]


if __name__ == "__main__":
    main()
