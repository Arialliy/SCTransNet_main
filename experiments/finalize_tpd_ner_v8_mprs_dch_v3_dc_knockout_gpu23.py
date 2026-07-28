#!/usr/bin/env python3
"""Run the independent V3 DC-knockout matrix on its fixed GPU2/GPU3 lanes.

This is deliberately an orchestration boundary, not a training or formal
postprocess component.  It waits for the immutable versioned repaired V3
aggregate closure, explicitly freezes (or verifies) the diagnostic-only source
lock, then launches at most one evaluator process per fixed checkpoint.  Each
evaluator owns the four sequential in-memory knockout modes for its checkpoint.
The two checkpoint processes may run concurrently on their separately pinned
physical GPUs.  Original formal V3 checkpoints and sweeps remain read-only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import TracebackType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout as evaluator,
)
from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock as freezer,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_v3_dc_knockout as diagnostic_post,
)
from experiments import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)


PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
EVALUATOR = REPO_ROOT / "experiments/evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout.py"
AGGREGATOR = REPO_ROOT / "experiments/postprocess_tpd_ner_v8_mprs_dch_v3_dc_knockout.py"
SOURCE_FREEZER = (
    REPO_ROOT
    / "experiments/freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock.py"
)
SOURCE_LOCK = freezer.DEFAULT_SOURCE_LOCK
FORMAL_COMPLETE_MARKER = freezer.DEFAULT_FORMAL_MARKER
FORMAL_REPAIR_ID = freezer.FORMAL_REPAIR_ID
COMPLETE_MARKER = spec.aggregate_paths()[2]

# These values are deliberately literal instead of discovering an ordinal at
# runtime: CUDA_VISIBLE_DEVICES is set to the UUID so evaluator cuda:0 means
# exactly the assigned physical device, independent of ambient ordering.
# The frozen diagnostic spec is the single authority for physical lanes.  Do
# not duplicate these UUID literals here: evaluator and orchestrator must bind
# the exact same lane contract.
CHECKPOINT_LANES: Mapping[str, Mapping[str, Any]] = spec.CHECKPOINT_GPU_LANES
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
CUDA_DEVICE_ORDER_ENV = "CUDA_DEVICE_ORDER"
CUBLAS_WORKSPACE_CONFIG_ENV = spec.CUBLAS_WORKSPACE_CONFIG_ENV
PYTHONHASHSEED_ENV = spec.PYTHONHASHSEED_ENV
KNOCKOUT_PHYSICAL_GPU_INDEX_ENV = spec.PHYSICAL_GPU_INDEX_ENV
KNOCKOUT_PHYSICAL_GPU_UUID_ENV = spec.PHYSICAL_GPU_UUID_ENV
CUDA_DEVICE_ORDER_VALUE = spec.CUDA_DEVICE_ORDER
CUBLAS_WORKSPACE_CONFIG_VALUE = spec.CUBLAS_WORKSPACE_CONFIG
PYTHONHASHSEED_VALUE = spec.PYTHONHASHSEED
POLL_MAX_SECONDS = 30.0


class KnockoutFinalizerError(RuntimeError):
    """The diagnostic-only GPU2/3 closure cannot safely proceed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KnockoutFinalizerError(message)


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    _require(
        value.is_file() and not value.is_symlink(),
        f"{label} must be a regular non-symlink file: {value}",
    )
    return value


def _validate_poll_seconds(value: float) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 < float(value) <= POLL_MAX_SECONDS,
        "poll interval must be greater than 0 and no more than 30 seconds",
    )
    return float(value)


def _lane(checkpoint: str) -> dict[str, Any]:
    _require(checkpoint in spec.CHECKPOINTS, f"unsupported checkpoint: {checkpoint}")
    _require(checkpoint in CHECKPOINT_LANES, f"checkpoint lane is absent: {checkpoint}")
    lane = dict(CHECKPOINT_LANES[checkpoint])
    _require(
        set(lane) == {"physical_gpu_index", "physical_gpu_uuid"},
        f"checkpoint lane schema differs: {checkpoint}",
    )
    _require(
        lane["physical_gpu_index"] in {2, 3}
        and isinstance(lane["physical_gpu_uuid"], str)
        and lane["physical_gpu_uuid"].startswith("GPU-"),
        f"checkpoint lane is invalid: {checkpoint}",
    )
    return lane


def evaluator_command(checkpoint: str) -> list[str]:
    """Return the only evaluator command permitted for one checkpoint."""

    _lane(checkpoint)
    return [
        str(PYTHON),
        str(EVALUATOR),
        "--run",
        "--checkpoint",
        checkpoint,
        "--device",
        "cuda:0",
    ]


def evaluator_environment(checkpoint: str) -> dict[str, str]:
    """Pin evaluator cuda:0 to the checkpoint's immutable physical GPU."""

    lane = _lane(checkpoint)
    environment = dict(os.environ)
    environment[CUDA_VISIBLE_DEVICES_ENV] = lane["physical_gpu_uuid"]
    environment[CUDA_DEVICE_ORDER_ENV] = CUDA_DEVICE_ORDER_VALUE
    environment[CUBLAS_WORKSPACE_CONFIG_ENV] = (
        CUBLAS_WORKSPACE_CONFIG_VALUE
    )
    environment[PYTHONHASHSEED_ENV] = PYTHONHASHSEED_VALUE
    environment[KNOCKOUT_PHYSICAL_GPU_INDEX_ENV] = str(
        lane["physical_gpu_index"]
    )
    environment[KNOCKOUT_PHYSICAL_GPU_UUID_ENV] = lane["physical_gpu_uuid"]
    return environment


def inspect_formal_postprocess_complete() -> dict[str, Any]:
    """Prove formal V3 closure before source locking or CUDA work.

    ``current_formal_artifact_binding`` validates the repaired marker, repaired
    report, repair attestation/wrapper/protocol, original formal locks,
    checkpoints and learned-V3 sweeps as one immutable input snapshot.  The
    repaired marker is checked explicitly first to preserve waiting semantics.
    """

    _regular_file(FORMAL_COMPLETE_MARKER, "formal V3 POSTPROCESS_COMPLETE")
    try:
        binding = freezer.current_formal_artifact_binding()
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"formal V3 postprocess closure is not valid: {exc}"
        ) from exc
    _require(
        isinstance(binding, Mapping)
        and binding.get("formal_completion_marker", {}).get("path")
        == str(FORMAL_COMPLETE_MARKER.resolve()),
        "repaired formal V3 completion-marker binding differs",
    )
    repair = binding.get("formal_selection_contract_repair")
    _require(
        isinstance(repair, Mapping)
        and repair.get("repair_id") == FORMAL_REPAIR_ID
        and repair.get("authority")
        == "versioned_selection_contract_repair_v1_only"
        and repair.get("each_variant_uses_own_selected_checkpoints")
        is True
        and repair.get("formal_aggregate_decision")
        == freezer.EXPECTED_FORMAL_DECISION,
        "repaired formal V3 selection-contract authority differs",
    )
    return dict(binding)


def freeze_or_verify_diagnostic_source_lock() -> dict[str, Any]:
    """Explicitly freeze once after formal closure, otherwise verify it.

    A raced no-overwrite freeze is harmless only when the winner's lock passes
    the same verification.  Existing invalid files are never repaired.
    """

    inspect_formal_postprocess_complete()
    if SOURCE_LOCK.exists() or SOURCE_LOCK.is_symlink():
        try:
            return freezer.verify_source_lock(SOURCE_LOCK)
        except Exception as exc:
            raise KnockoutFinalizerError(
                f"existing diagnostic source lock is invalid: {exc}"
            ) from exc
    try:
        freezer.publish_new_lock(SOURCE_LOCK, freezer.build_source_lock())
    except FileExistsError:
        # Another orchestrator may have won the no-overwrite publication race.
        pass
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"cannot freeze diagnostic source lock: {exc}"
        ) from exc
    try:
        return freezer.verify_source_lock(SOURCE_LOCK)
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"new diagnostic source lock did not verify: {exc}"
        ) from exc


def _source_binding() -> dict[str, Any]:
    try:
        return freezer.current_source_binding(SOURCE_LOCK)
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"diagnostic source binding is invalid: {exc}"
        ) from exc


def _validate_existing_sweep(
    checkpoint: str,
    *,
    source_binding: Mapping[str, Any],
) -> bool:
    """Return true only for a complete valid checkpoint artifact.

    Any existing partial, conflicting, symlinked, or stale-lock artifact is a
    hard error.  It is never overwritten or silently scheduled for repair.
    """

    path = spec.sweep_path(checkpoint)
    if not path.exists() and not path.is_symlink():
        return False
    try:
        _regular_file(path, f"diagnostic sweep {checkpoint}")
        rows = diagnostic_post.validate_checkpoint_sweep(
            path,
            checkpoint=checkpoint,
            expected_source_binding=source_binding,
        )
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"existing diagnostic sweep conflicts: {checkpoint}: {exc}"
        ) from exc
    _require(
        len(rows) == len(spec.KNOCKOUT_MODES),
        f"existing diagnostic sweep row count differs: {checkpoint}",
    )
    return True


def execution_plan() -> dict[str, Any]:
    """Read-only plan; it neither freezes a source lock nor touches CUDA."""

    formal_complete = FORMAL_COMPLETE_MARKER.is_file() and not FORMAL_COMPLETE_MARKER.is_symlink()
    sweeps = {
        checkpoint: {
            "path": str(spec.sweep_path(checkpoint).resolve()),
            "exists": spec.sweep_path(checkpoint).is_file()
            and not spec.sweep_path(checkpoint).is_symlink(),
            "lane": _lane(checkpoint),
            "command": evaluator_command(checkpoint),
            "environment": {
                CUDA_VISIBLE_DEVICES_ENV: _lane(checkpoint)["physical_gpu_uuid"],
                CUDA_DEVICE_ORDER_ENV: CUDA_DEVICE_ORDER_VALUE,
                CUBLAS_WORKSPACE_CONFIG_ENV: (
                    CUBLAS_WORKSPACE_CONFIG_VALUE
                ),
                PYTHONHASHSEED_ENV: PYTHONHASHSEED_VALUE,
                KNOCKOUT_PHYSICAL_GPU_INDEX_ENV: str(
                    _lane(checkpoint)["physical_gpu_index"]
                ),
                KNOCKOUT_PHYSICAL_GPU_UUID_ENV: _lane(checkpoint)["physical_gpu_uuid"],
            },
            "internal_knockout_modes": list(spec.KNOCKOUT_MODES),
            "modes_evaluated_sequentially": True,
        }
        for checkpoint in spec.CHECKPOINTS
    }
    return {
        "artifact_kind": spec.ARTIFACT_KIND,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_postprocess_complete": formal_complete,
        "formal_completion_marker": str(FORMAL_COMPLETE_MARKER.resolve()),
        "formal_aggregate_authority": (
            "versioned_selection_contract_repair_v1_only"
        ),
        "formal_selection_contract_repair_id": FORMAL_REPAIR_ID,
        "each_variant_uses_own_selected_checkpoints": True,
        "diagnostic_source_lock": {
            "path": str(SOURCE_LOCK.resolve()),
            "exists": SOURCE_LOCK.is_file() and not SOURCE_LOCK.is_symlink(),
        },
        "checkpoint_processes_parallel": True,
        "checkpoint_count": len(spec.CHECKPOINTS),
        "knockout_modes_per_checkpoint": len(spec.KNOCKOUT_MODES),
        "sweeps": sweeps,
        "aggregate_command": [str(PYTHON), str(AGGREGATOR), "--aggregate"],
        "completion_marker": str(COMPLETE_MARKER.resolve()),
        "invokes_gpu": False,
    }


def _launch_evaluators(checkpoints: Sequence[str]) -> None:
    """Launch all missing checkpoint evaluations, then require all success."""

    processes: dict[str, subprocess.Popen[str]] = {}
    try:
        for checkpoint in checkpoints:
            processes[checkpoint] = subprocess.Popen(
                evaluator_command(checkpoint),
                cwd=REPO_ROOT,
                env=evaluator_environment(checkpoint),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        failures: list[str] = []
        for checkpoint in checkpoints:
            stdout, stderr = processes[checkpoint].communicate()
            if processes[checkpoint].returncode:
                detail = (stderr or stdout or "").strip()
                failures.append(
                    f"{checkpoint} exit={processes[checkpoint].returncode}: {detail}"
                )
        if failures:
            raise KnockoutFinalizerError(
                "DC-knockout evaluator failure: " + "; ".join(failures)
            )
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            if process.poll() is None:
                process.communicate()
        raise


def _aggregate_and_verify_marker() -> dict[str, Any]:
    """Call only the diagnostic aggregator and require its valid marker."""

    process = subprocess.run(
        [str(PYTHON), str(AGGREGATOR), "--aggregate"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise KnockoutFinalizerError(
            f"DC-knockout diagnostic aggregate failed ({process.returncode}): {detail}"
        )
    try:
        marker = diagnostic_post.inspect_complete()
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"diagnostic aggregate marker is invalid: {exc}"
        ) from exc
    _require(marker is not None, "aggregate returned success without DC knockout marker")
    return marker


def run_now() -> dict[str, Any]:
    """Complete/reuse the fixed 2x4 diagnostic package exactly once."""

    try:
        marker = diagnostic_post.inspect_complete()
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"existing DC-knockout completion marker is invalid: {exc}"
        ) from exc
    if marker is not None:
        return {
            "status": "already_complete",
            "marker": str(COMPLETE_MARKER),
            "evaluator_invocations": 0,
            "aggregate_invocations": 0,
        }

    freeze_or_verify_diagnostic_source_lock()
    source_binding = _source_binding()
    pending = [
        checkpoint
        for checkpoint in spec.CHECKPOINTS
        if not _validate_existing_sweep(
            checkpoint,
            source_binding=source_binding,
        )
    ]
    if pending:
        _launch_evaluators(pending)
    # Validate both the reused and newly published artifacts before aggregate.
    source_binding_after = _source_binding()
    _require(
        source_binding_after == source_binding,
        "diagnostic source binding changed during checkpoint evaluation",
    )
    for checkpoint in spec.CHECKPOINTS:
        _require(
            _validate_existing_sweep(
                checkpoint,
                source_binding=source_binding,
            ),
            f"checkpoint evaluator did not publish a complete sweep: {checkpoint}",
        )
    marker = _aggregate_and_verify_marker()
    return {
        "status": "complete",
        "marker": str(COMPLETE_MARKER),
        "evaluator_invocations": len(pending),
        "aggregate_invocations": 1,
        "reused_checkpoints": [
            checkpoint for checkpoint in spec.CHECKPOINTS if checkpoint not in pending
        ],
        "launched_checkpoints": list(pending),
        "marker_schema": marker.get("schema"),
    }


def watch_and_run(*, poll_seconds: float = POLL_MAX_SECONDS) -> dict[str, Any]:
    """Wait only for formal completion, then execute the fixed closure."""

    interval = _validate_poll_seconds(poll_seconds)
    while True:
        try:
            marker = diagnostic_post.inspect_complete()
        except Exception as exc:
            raise KnockoutFinalizerError(
                f"existing DC-knockout completion marker is invalid: {exc}"
            ) from exc
        if marker is not None:
            return {
                "status": "already_complete",
                "marker": str(COMPLETE_MARKER),
                "evaluator_invocations": 0,
                "aggregate_invocations": 0,
            }
        if not FORMAL_COMPLETE_MARKER.exists() and not FORMAL_COMPLETE_MARKER.is_symlink():
            print(
                "TPDNERV8V3_DC_KNOCKOUT_WAIT "
                f"formal_marker={FORMAL_COMPLETE_MARKER} poll_seconds={interval:g}",
                flush=True,
            )
            time.sleep(interval)
            continue
        # A present marker must validate; it is not safe to spin past a conflict.
        inspect_formal_postprocess_complete()
        return run_now()


class _FinalizerLock:
    """Nonblocking user-local lock preventing duplicate GPU launches."""

    def __init__(self) -> None:
        self._descriptor: int | None = None
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        base = Path(runtime) if runtime else Path("/tmp")
        _require(
            base.is_dir() and not base.is_symlink(),
            f"DC-knockout finalizer lock directory is unsafe: {base}",
        )
        self.path = base / f"sctransnet-tpd-ner-v8-v3-dc-knockout-gpu23-{os.getuid()}.lock"

    def __enter__(self) -> "_FinalizerLock":
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise KnockoutFinalizerError(
                f"cannot open DC-knockout finalizer lock {self.path}: {exc}"
            ) from exc
        try:
            _require(
                stat.S_ISREG(os.fstat(descriptor).st_mode),
                f"DC-knockout finalizer lock is not regular: {self.path}",
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise KnockoutFinalizerError(
                    "another V3 DC-knockout GPU2/3 finalizer is already running"
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
        description="Finalize the V3 DC-knockout package on fixed GPU2/GPU3"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--status", action="store_true")
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run-now", action="store_true")
    action.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=POLL_MAX_SECONDS)
    args = parser.parse_args(argv)
    if not 0 < args.poll_seconds <= POLL_MAX_SECONDS:
        parser.error("--poll-seconds must be > 0 and <= 30")
    return args


def status() -> dict[str, Any]:
    try:
        marker = diagnostic_post.inspect_complete()
    except Exception as exc:
        raise KnockoutFinalizerError(
            f"existing DC-knockout completion marker is invalid: {exc}"
        ) from exc
    return {
        "artifact_kind": spec.ARTIFACT_KIND,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "status": "complete" if marker is not None else "incomplete",
        "marker": marker,
        "formal_postprocess_complete": FORMAL_COMPLETE_MARKER.is_file()
        and not FORMAL_COMPLETE_MARKER.is_symlink(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.status:
            result = status()
        elif args.plan:
            result = execution_plan()
        else:
            with _FinalizerLock():
                result = (
                    watch_and_run(poll_seconds=args.poll_seconds)
                    if args.watch
                    else run_now()
                )
    except (KnockoutFinalizerError, OSError) as exc:
        print(f"TPDNERV8V3_DC_KNOCKOUT_FINALIZER_FAILED {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATOR",
    "CHECKPOINT_LANES",
    "COMPLETE_MARKER",
    "CUBLAS_WORKSPACE_CONFIG_ENV",
    "CUBLAS_WORKSPACE_CONFIG_VALUE",
    "CUDA_DEVICE_ORDER_ENV",
    "CUDA_DEVICE_ORDER_VALUE",
    "CUDA_VISIBLE_DEVICES_ENV",
    "EVALUATOR",
    "FORMAL_COMPLETE_MARKER",
    "KNOCKOUT_PHYSICAL_GPU_INDEX_ENV",
    "KNOCKOUT_PHYSICAL_GPU_UUID_ENV",
    "KnockoutFinalizerError",
    "POLL_MAX_SECONDS",
    "PYTHONHASHSEED_ENV",
    "PYTHONHASHSEED_VALUE",
    "PYTHON",
    "SOURCE_FREEZER",
    "SOURCE_LOCK",
    "evaluator_command",
    "evaluator_environment",
    "execution_plan",
    "freeze_or_verify_diagnostic_source_lock",
    "inspect_formal_postprocess_complete",
    "main",
    "parse_args",
    "run_now",
    "status",
    "watch_and_run",
]
