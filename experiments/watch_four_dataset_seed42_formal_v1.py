#!/usr/bin/env python3
"""Keep the four-wave formal supervisor recoverable across worker exits.

The currently running supervisor is observed first.  If it exits before all
eight runs complete, this watchdog relaunches the same resumable supervisor up
to a bounded number of times.  It never starts an additional supervisor while
the observed one is alive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
SUPERVISOR = (
    REPO_ROOT / "experiments" / "supervise_four_dataset_seed42_formal_v1.py"
)
ROOT = REPO_ROOT / "results" / "four_dataset_seed42_v1" / "launch" / "formal"
SUPERVISOR_STATUS = ROOT / "supervisor_status.json"
WATCHDOG_STATUS = ROOT / "watchdog_status.json"
CONSOLE_LOG = ROOT / "supervisor_console.log"
EXPECTED_COMMAND_FRAGMENTS = (
    b"supervise_four_dataset_seed42_formal_v1.py",
    b"experiments.supervise_four_dataset_seed42_formal_v1",
)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_matches(pid: int) -> bool:
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return any(fragment in command for fragment in EXPECTED_COMMAND_FRAGMENTS)


def supervisor_complete() -> bool:
    try:
        payload = json.loads(SUPERVISOR_STATUS.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return payload.get("mode") == "formal" and payload.get("status") == "complete"


def record(status: str, **detail: Any) -> None:
    write_json_atomic(
        WATCHDOG_STATUS,
        {
            "schema": "sctransnet_four_dataset_seed42_watchdog/v1",
            "status": status,
            "detail": detail,
            "updated_at_unix": time.time(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-supervisor-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    args = parser.parse_args()
    if args.initial_supervisor_pid < 1:
        parser.error("--initial-supervisor-pid must be positive")
    if args.poll_seconds <= 0 or args.retry_seconds <= 0:
        parser.error("poll/retry intervals must be positive")
    if args.max_restarts < 0:
        parser.error("--max-restarts must be non-negative")

    record(
        "observing",
        observed_pid=args.initial_supervisor_pid,
        max_restarts=args.max_restarts,
    )
    while process_matches(args.initial_supervisor_pid):
        time.sleep(args.poll_seconds)
    if supervisor_complete():
        record("complete", restarts_used=0)
        return 0

    ROOT.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.max_restarts + 1):
        record("restarting", attempt=attempt)
        with CONSOLE_LOG.open("ab", buffering=0) as console:
            process = subprocess.Popen(
                [str(PYTHON), str(SUPERVISOR), "--poll-seconds", "2"],
                cwd=REPO_ROOT,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            exit_code = process.wait()
        if exit_code == 0 and supervisor_complete():
            record("complete", restarts_used=attempt)
            return 0
        record("retry_wait", attempt=attempt, exit_code=exit_code)
        if attempt < args.max_restarts:
            time.sleep(args.retry_seconds)

    record("failed", restarts_used=args.max_restarts)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
