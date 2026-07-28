#!/usr/bin/env python3
"""User-space GPU 2/3 memory reservation for SCTransNet experiments.

This helper is deliberately independent from every locked training source.
GPU2 uses a small adaptive reservation only after validating the fixed V4
training process.  GPU3 uses a larger adaptive reservation.  Both keep a
minimum free-memory floor for the real workload and release
their own buffers if that floor is crossed.

The helper cannot provide scheduler-level exclusivity.  It only makes the
selected device less attractive to unrelated jobs that size themselves from
currently free memory.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_ROOT = (
    REPO_ROOT / "experiments" / "runtime" / "gpu23_memory_reservation"
)
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"
MIB = 1024 * 1024
ALLOCATION_CHUNK_MIB = 256
RESERVATION_QUANTUM_MIB = 256
ALLOCATION_JITTER_HEADROOM_MIB = 256
GPU2_V4_UNIT = "sctransnet-tpd-ner-v8-v4-tail-aware-gpu2.service"
GPU2_V4_PROCESS_NAME = "/home/ly/BasicIRSTD/infrarenet/bin/python"

CUDA_CONTEXT_ALLOWANCE_MIB = 768

# GPU2 keeps a wide growth margin for the active batch-16 FP32 training job.
# GPU3 keeps enough unreserved memory for a normal evaluation process.
GPU_SPECS: dict[int, dict[str, Any]] = {
    2: {
        "uuid": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
        "default_reserve_mib": 1024,
        "default_min_free_mib": 4096,
        "role": "active_v4_training_small_direct_reservation",
    },
    3: {
        "uuid": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
        "default_reserve_mib": 8192,
        "default_min_free_mib": 6144,
        "role": "future_v4_evaluation",
    },
}

STOP_EVENT = Event()


class ReservationError(RuntimeError):
    """Raised when the reservation contract cannot be satisfied."""


@dataclass(frozen=True)
class GpuSnapshot:
    physical_gpu_index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    free_mib: int


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_mib: int


@dataclass(frozen=True)
class ReservationPlan:
    physical_gpu_index: int
    expected_uuid: str
    observed_uuid: str
    gpu_name: str
    gpu_role: str
    total_mib: int
    used_mib: int
    free_mib: int
    requested_reserve_mib: int
    reserve_mib: int
    adaptive_reduction_applied: bool
    min_free_mib: int
    cuda_context_allowance_mib: int
    maximum_safe_reserve_mib: int
    projected_free_after_mib: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed

def nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed



def parse_gpu_snapshot(line: str, *, expected_index: int) -> GpuSnapshot:
    fields = [field.strip() for field in line.strip().split(",")]
    if len(fields) != 6:
        raise ReservationError(
            f"unexpected nvidia-smi field count: expected 6, got {len(fields)}"
        )
    try:
        observed_index = int(fields[0])
        total_mib = int(fields[3])
        used_mib = int(fields[4])
        free_mib = int(fields[5])
    except ValueError as exc:
        raise ReservationError("nvidia-smi returned a non-integer field") from exc
    if observed_index != expected_index:
        raise ReservationError(
            f"physical GPU index mismatch: expected {expected_index}, "
            f"observed {observed_index}"
        )
    if min(total_mib, used_mib, free_mib) < 0:
        raise ReservationError("nvidia-smi returned negative memory")
    if used_mib + free_mib > total_mib + 1024:
        raise ReservationError("nvidia-smi memory fields are inconsistent")
    return GpuSnapshot(
        physical_gpu_index=observed_index,
        uuid=fields[1],
        name=fields[2],
        total_mib=total_mib,
        used_mib=used_mib,
        free_mib=free_mib,
    )


def query_gpu_snapshot(physical_gpu_index: int) -> GpuSnapshot:
    if physical_gpu_index not in GPU_SPECS:
        raise ReservationError("only physical GPU 2 or 3 is permitted")
    command = [
        "nvidia-smi",
        "-i",
        str(physical_gpu_index),
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReservationError(f"nvidia-smi probe failed: {exc}") from exc
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ReservationError(
            f"nvidia-smi returned {len(lines)} non-empty rows; expected one"
        )
    snapshot = parse_gpu_snapshot(lines[0], expected_index=physical_gpu_index)
    spec = GPU_SPECS[physical_gpu_index]
    if snapshot.uuid != spec["uuid"]:
        raise ReservationError(
            f"GPU UUID mismatch for physical GPU {physical_gpu_index}: "
            f"expected {spec['uuid']}, observed {snapshot.uuid}"
        )
    if snapshot.name != EXPECTED_GPU_NAME:
        raise ReservationError(
            f"GPU model mismatch: expected {EXPECTED_GPU_NAME}, "
            f"observed {snapshot.name}"
        )
    return snapshot

def parse_compute_processes(
    output: str,
    *,
    expected_uuid: str,
) -> list[ComputeProcess]:
    processes: list[ComputeProcess] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise ReservationError(
                "unexpected nvidia-smi compute-process field count"
            )
        if fields[0] != expected_uuid:
            continue
        try:
            pid = int(fields[1])
            used_mib = int(fields[3])
        except ValueError as exc:
            raise ReservationError(
                "nvidia-smi returned a non-integer process field"
            ) from exc
        if pid < 1 or used_mib < 0:
            raise ReservationError(
                "nvidia-smi returned an invalid compute-process field"
            )
        processes.append(
            ComputeProcess(
                gpu_uuid=fields[0],
                pid=pid,
                process_name=fields[2],
                used_mib=used_mib,
            )
        )
    return sorted(processes, key=lambda process: process.pid)


def query_compute_processes(expected_uuid: str) -> list[ComputeProcess]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReservationError(
            f"nvidia-smi compute-process probe failed: {exc}"
        ) from exc
    return parse_compute_processes(
        completed.stdout,
        expected_uuid=expected_uuid,
    )


def query_gpu2_v4_main_pid() -> int:
    command = [
        "systemctl",
        "--user",
        "show",
        GPU2_V4_UNIT,
        "--property=ActiveState,SubState,MainPID",
        "--no-pager",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReservationError(
            f"cannot inspect fixed V4 unit {GPU2_V4_UNIT}: {exc}"
        ) from exc
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            fields[key] = value
    if fields.get("ActiveState") != "active":
        raise ReservationError("fixed V4 GPU2 unit is not active")
    if fields.get("SubState") != "running":
        raise ReservationError("fixed V4 GPU2 unit is not running")
    try:
        main_pid = int(fields.get("MainPID", "0"))
    except ValueError as exc:
        raise ReservationError("fixed V4 GPU2 MainPID is not an integer") from exc
    if main_pid < 1:
        raise ReservationError("fixed V4 GPU2 MainPID must be positive")
    return main_pid


def inspect_existing_compute_processes(
    plan: ReservationPlan,
) -> list[ComputeProcess]:
    processes = query_compute_processes(plan.expected_uuid)
    if plan.physical_gpu_index == 2:
        main_pid = query_gpu2_v4_main_pid()
        matches = [
            process
            for process in processes
            if process.pid == main_pid
            and process.process_name == GPU2_V4_PROCESS_NAME
        ]
        if len(matches) != 1:
            raise ReservationError(
                "fixed V4 MainPID is not the expected compute process on GPU2"
            )
    return processes

def build_reservation_plan(
    snapshot: GpuSnapshot,
    *,
    reserve_mib: int,
    min_free_mib: int,
    context_allowance_mib: int = CUDA_CONTEXT_ALLOWANCE_MIB,
) -> ReservationPlan:
    if snapshot.physical_gpu_index not in GPU_SPECS:
        raise ReservationError("only physical GPU 2 or 3 is permitted")
    if reserve_mib < 1 or min_free_mib < 1:
        raise ReservationError("reserve and minimum-free must be positive")
    if context_allowance_mib < 0:
        raise ReservationError("context allowance cannot be negative")
    spec = GPU_SPECS[snapshot.physical_gpu_index]
    if snapshot.uuid != spec["uuid"]:
        raise ReservationError("snapshot UUID does not match the fixed GPU mapping")
    effective_context_allowance = context_allowance_mib
    maximum_safe = max(
        0,
        snapshot.free_mib - min_free_mib - effective_context_allowance,
    )
    effective_reserve = min(reserve_mib, maximum_safe)
    effective_reserve = (effective_reserve // RESERVATION_QUANTUM_MIB) * RESERVATION_QUANTUM_MIB
    if effective_reserve < RESERVATION_QUANTUM_MIB:
        raise ReservationError(
            "current safe limit cannot hold the minimum 256 MiB quantum: "
            f"maximum_safe={maximum_safe} MiB"
        )
    projected_free = (
        snapshot.free_mib - effective_reserve - effective_context_allowance
    )
    return ReservationPlan(
        physical_gpu_index=snapshot.physical_gpu_index,
        expected_uuid=spec["uuid"],
        observed_uuid=snapshot.uuid,
        gpu_name=snapshot.name,
        gpu_role=spec["role"],
        total_mib=snapshot.total_mib,
        used_mib=snapshot.used_mib,
        free_mib=snapshot.free_mib,
        requested_reserve_mib=reserve_mib,
        reserve_mib=effective_reserve,
        adaptive_reduction_applied=effective_reserve < reserve_mib,
        min_free_mib=min_free_mib,
        cuda_context_allowance_mib=effective_context_allowance,
        maximum_safe_reserve_mib=maximum_safe,
        projected_free_after_mib=projected_free,
    )


def next_allocation_chunk_mib(
    *,
    free_mib: int,
    remaining_mib: int,
    min_free_mib: int,
    jitter_headroom_mib: int = ALLOCATION_JITTER_HEADROOM_MIB,
) -> int:
    if remaining_mib < 1:
        return 0
    candidate = min(ALLOCATION_CHUNK_MIB, remaining_mib)
    required_free = candidate + min_free_mib + jitter_headroom_mib
    if free_mib < required_free:
        return 0
    return candidate


def actual_allocation_metadata(
    plan: ReservationPlan,
    allocated_mib: int,
) -> dict[str, Any]:
    if allocated_mib < RESERVATION_QUANTUM_MIB:
        raise ReservationError("no 256 MiB reservation chunk could be allocated")
    return {
        "requested_reserve_mib": plan.requested_reserve_mib,
        "planned_reserve_mib": plan.reserve_mib,
        "reserve_mib": allocated_mib,
        "allocated_mib": allocated_mib,
        "adaptive_reduction_applied": allocated_mib < plan.requested_reserve_mib,
    }


def build_final_reservation_state(
    *,
    base_state: dict[str, Any],
    plan: ReservationPlan,
    allocated_mib: int,
    exit_code: int,
    exit_reason: str,
    final_free_mib: int,
) -> dict[str, Any]:
    materialized = dict(base_state)
    if allocated_mib >= RESERVATION_QUANTUM_MIB:
        materialized.update(actual_allocation_metadata(plan, allocated_mib))
    return {
        **materialized,
        "status": "released" if exit_code == 0 else "self_released",
        "released_at": utc_now(),
        "release_reason": exit_reason,
        "runtime_free_after_release_mib": final_free_mib,
    }


def state_path(state_root: Path, physical_gpu_index: int) -> Path:
    return state_root / f"gpu{physical_gpu_index}.json"


def lock_path(state_root: Path, physical_gpu_index: int) -> Path:
    return state_root / f"gpu{physical_gpu_index}.lock"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ReservationError(f"invalid state path: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReservationError(f"cannot read state path {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReservationError("reservation state must be a JSON object")
    return payload


def process_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status_payload(
    *,
    snapshot: GpuSnapshot,
    recorded_state: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "sctransnet_gpu23_memory_reservation_status_v1",
        "observed_at": utc_now(),
        "gpu": asdict(snapshot),
        "recorded_state": recorded_state,
    }
    pid = recorded_state.get("pid") if recorded_state else None
    payload["holder_process_alive"] = process_is_alive(pid)
    payload["active"] = bool(
        recorded_state
        and recorded_state.get("status") == "active"
        and payload["holder_process_alive"]
    )
    return payload


def _install_signal_handlers() -> None:
    def request_stop(_signum: int, _frame: Any) -> None:
        STOP_EVENT.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def _release_cuda_buffers(torch_module: Any, buffers: list[Any]) -> None:
    buffers.clear()
    gc.collect()
    torch_module.cuda.empty_cache()
    torch_module.cuda.synchronize()


def hold_reservation(
    *,
    plan: ReservationPlan,
    state_root: Path,
    poll_seconds: int,
) -> int:
    """Allocate the requested buffers and hold them until stopped.

    Torch is imported only after the physical UUID is validated and
    CUDA_VISIBLE_DEVICES is restricted to that single UUID.
    """

    state_root.mkdir(parents=True, exist_ok=True)
    if state_root.is_symlink() or not state_root.is_dir():
        raise ReservationError(f"invalid state root: {state_root}")
    reservation_lock = lock_path(state_root, plan.physical_gpu_index)
    if reservation_lock.is_symlink():
        raise ReservationError(f"invalid lock path: {reservation_lock}")
    lock_handle = reservation_lock.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReservationError(
                f"reservation already active for GPU {plan.physical_gpu_index}"
            ) from exc

        inspect_existing_compute_processes(plan)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = plan.expected_uuid
        try:
            import torch
        except ImportError as exc:
            raise ReservationError("PyTorch is required for hold mode") from exc

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ReservationError(
                "hold mode requires exactly one visible CUDA device"
            )
        torch.cuda.set_device(0)
        visible_name = torch.cuda.get_device_name(0)
        if visible_name != EXPECTED_GPU_NAME:
            raise ReservationError(
                f"visible CUDA device model mismatch: {visible_name}"
            )
        inspect_existing_compute_processes(plan)

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_before_mib = free_bytes // MIB
        total_runtime_mib = total_bytes // MIB
        runtime_snapshot = GpuSnapshot(
            physical_gpu_index=plan.physical_gpu_index,
            uuid=plan.expected_uuid,
            name=visible_name,
            total_mib=total_runtime_mib,
            used_mib=max(0, total_runtime_mib - free_before_mib),
            free_mib=free_before_mib,
        )
        plan = build_reservation_plan(
            runtime_snapshot,
            reserve_mib=plan.requested_reserve_mib,
            min_free_mib=plan.min_free_mib,
            context_allowance_mib=ALLOCATION_JITTER_HEADROOM_MIB,
        )

        state_file = state_path(state_root, plan.physical_gpu_index)
        base_state: dict[str, Any] = {
            "schema": "sctransnet_gpu23_memory_reservation_state_v1",
            "pid": os.getpid(),
            "status": "allocating",
            "started_at": utc_now(),
            "physical_gpu_index": plan.physical_gpu_index,
            "expected_uuid": plan.expected_uuid,
            "gpu_name": visible_name,
            "gpu_role": plan.gpu_role,
            "requested_reserve_mib": plan.requested_reserve_mib,
            "planned_reserve_mib": plan.reserve_mib,
            "reserve_mib": 0,
            "allocated_mib": 0,
            "adaptive_reduction_applied": None,
            "min_free_mib": plan.min_free_mib,
            "poll_seconds": poll_seconds,
            "runtime_total_mib": total_runtime_mib,
            "runtime_free_before_mib": free_before_mib,
        }
        write_json_atomic(state_file, base_state)

        buffers: list[Any] = []
        remaining_mib = plan.reserve_mib
        allocated_mib = 0
        exit_reason = "release_requested"
        exit_code = 0
        try:
            while remaining_mib > 0:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                chunk_mib = next_allocation_chunk_mib(
                    free_mib=free_bytes // MIB,
                    remaining_mib=remaining_mib,
                    min_free_mib=plan.min_free_mib,
                )
                if chunk_mib == 0:
                    break
                buffer = torch.empty(
                    chunk_mib * MIB,
                    dtype=torch.uint8,
                    device="cuda:0",
                )
                buffers.append(buffer)
                buffer.zero_()
                remaining_mib -= chunk_mib
                allocated_mib += chunk_mib
                base_state.update(
                    actual_allocation_metadata(plan, allocated_mib)
                )
                base_state["allocation_updated_at"] = utc_now()
                write_json_atomic(state_file, base_state)

            allocation_metadata = actual_allocation_metadata(plan, allocated_mib)
            base_state.update(allocation_metadata)

            torch.cuda.synchronize()
            free_after_bytes, _ = torch.cuda.mem_get_info(0)
            free_after_mib = free_after_bytes // MIB
            if free_after_mib < plan.min_free_mib:
                raise ReservationError(
                    "free-memory floor was crossed after allocation"
                )
            active_state = {
                **base_state,
                "status": "active",
                "activated_at": utc_now(),
                "runtime_free_after_mib": free_after_mib,
                "allocated_chunks": len(buffers),
                "heartbeat_at": utc_now(),
            }
            write_json_atomic(state_file, active_state)
            print(
                json.dumps(
                    {
                        "event": "GPU23_MEMORY_RESERVATION_ACTIVE",
                        "physical_gpu_index": plan.physical_gpu_index,
                        "uuid": plan.expected_uuid,
                        "requested_reserve_mib": plan.requested_reserve_mib,
                        "reserve_mib": allocated_mib,
                        "adaptive_reduction_applied": allocated_mib < plan.requested_reserve_mib,
                        "min_free_mib": plan.min_free_mib,
                        "free_after_mib": free_after_mib,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            while not STOP_EVENT.wait(poll_seconds):
                current_free_bytes, _ = torch.cuda.mem_get_info(0)
                current_free_mib = current_free_bytes // MIB
                active_state["heartbeat_at"] = utc_now()
                active_state["runtime_free_current_mib"] = current_free_mib
                write_json_atomic(state_file, active_state)
                if current_free_mib < plan.min_free_mib:
                    exit_reason = "minimum_free_memory_crossed"
                    exit_code = 3
                    break
        except Exception:
            exit_reason = "allocation_or_monitor_failure"
            exit_code = 2
            raise
        finally:
            _release_cuda_buffers(torch, buffers)
            final_free_bytes, _ = torch.cuda.mem_get_info(0)
            final_state = build_final_reservation_state(
                base_state=base_state,
                plan=plan,
                allocated_mib=allocated_mib,
                exit_code=exit_code,
                exit_reason=exit_reason,
                final_free_mib=final_free_bytes // MIB,
            )
            write_json_atomic(state_file, final_state)
        return exit_code
    finally:
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reserve configurable memory on physical GPU 2 or 3."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "hold", "status"):
        child = subparsers.add_parser(action)
        child.add_argument(
            "--physical-gpu",
            type=int,
            choices=tuple(sorted(GPU_SPECS)),
            required=True,
        )
        child.add_argument(
            "--state-root",
            type=Path,
            default=DEFAULT_STATE_ROOT,
        )
        if action in {"preflight", "hold"}:
            child.add_argument("--reserve-mib", type=positive_integer)
            child.add_argument("--min-free-mib", type=positive_integer)
        if action == "hold":
            child.add_argument(
                "--poll-seconds",
                type=positive_integer,
                default=5,
            )
    return parser


def resolved_limits(args: argparse.Namespace) -> tuple[int, int]:
    spec = GPU_SPECS[args.physical_gpu]
    reserve_mib = (
        args.reserve_mib
        if args.reserve_mib is not None
        else int(spec["default_reserve_mib"])
    )
    min_free_mib = (
        args.min_free_mib
        if args.min_free_mib is not None
        else int(spec["default_min_free_mib"])
    )
    return reserve_mib, min_free_mib


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = query_gpu_snapshot(args.physical_gpu)
        if args.action == "status":
            payload = status_payload(
                snapshot=snapshot,
                recorded_state=read_state(
                    state_path(args.state_root, args.physical_gpu)
                ),
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        reserve_mib, min_free_mib = resolved_limits(args)
        plan = build_reservation_plan(
            snapshot,
            reserve_mib=reserve_mib,
            min_free_mib=min_free_mib,
        )
        compute_processes = inspect_existing_compute_processes(plan)
        if args.action == "preflight":
            print(
                json.dumps(
                    {
                        "schema": (
                            "sctransnet_gpu23_memory_reservation_preflight_v1"
                        ),
                        "status": "ready",
                        "plan": asdict(plan),
                        "compute_processes": [
                            asdict(process) for process in compute_processes
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        _install_signal_handlers()
        return hold_reservation(
            plan=plan,
            state_root=args.state_root,
            poll_seconds=args.poll_seconds,
        )
    except ReservationError as exc:
        print(
            json.dumps(
                {
                    "schema": "sctransnet_gpu23_memory_reservation_error_v1",
                    "status": "aborted",
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

