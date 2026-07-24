#!/usr/bin/env python3
"""Read-only runtime watchdog for the four formal800 training services."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "experiments/results/tpd_pe_formal800_4x5090_v1"
RUN_NAME = "seed_42_formal800_pd_fp32_4x5090_v1"
VARIANTS = ("original", "progressive", "tpd", "spd")
INVOCATIONS = {
    "original": "0e533dd2d1444feb9ba1ed1e3d42135c",
    "progressive": "0803e8ed5f974f909c54d6daaddc23bb",
    "tpd": "669e46b8bd2245db92788afa62066bd3",
    "spd": "75468708b771457daac1cf0586f7c333",
}
GPU_UUIDS = {
    "original": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "progressive": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "tpd": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "spd": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
FORBIDDEN_LOG_MARKERS = (
    "FORMAL4X5090_ABORT",
    "Traceback",
    "CUDA error",
    "out of memory",
    "OutOfMemory",
    "No space left",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch SCTransNet formal800 4x5090")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--stale-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("--poll-seconds must be >= 30")
    if args.stale_seconds < args.poll_seconds * 2:
        parser.error("--stale-seconds must be at least twice --poll-seconds")
    return args


def systemd_properties(unit: str) -> Dict[str, str]:
    process = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,InvocationID,NRestarts,MainPID",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(f"Cannot inspect {unit}: {process.stderr.strip()}")
    properties: Dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def invocation_journal(invocation: str) -> str:
    process = subprocess.run(
        [
            "journalctl",
            "--user",
            f"_SYSTEMD_INVOCATION_ID={invocation}",
            "--no-pager",
            "--output=cat",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout


def assert_finite(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number at {context}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{context}[{index}]")


def read_metrics_stably(path: Path) -> list[Dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            text = path.read_text(encoding="utf-8")
            events = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
            return events
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1)
    raise RuntimeError(f"Cannot read stable metrics from {path}") from last_error


def inspect_variant(
    variant: str, now: float, stale_seconds: int
) -> Dict[str, Any]:
    unit = f"sctransnet-formal800-4x5090-{variant}.service"
    expected_invocation = INVOCATIONS[variant]
    properties = systemd_properties(unit)
    load_state = properties.get("LoadState")
    current_invocation = properties.get("InvocationID")
    unit_collected = load_state == "not-found" and current_invocation == ""
    if not unit_collected:
        if load_state != "loaded":
            raise RuntimeError(f"{variant}: unexpected LoadState={load_state!r}")
        if current_invocation != expected_invocation:
            raise RuntimeError(
                f"{variant}: InvocationID changed from {expected_invocation} "
                f"to {current_invocation}"
            )
        if properties.get("NRestarts") != "0":
            raise RuntimeError(f"{variant}: unexpected restart count")

    journal = invocation_journal(expected_invocation)
    for marker in FORBIDDEN_LOG_MARKERS:
        if marker in journal:
            raise RuntimeError(f"{variant}: forbidden journal marker {marker!r}")
    completion_marker = (
        f"FORMAL4X5090_COMPLETE variant={variant} "
        f"gpu_uuid={GPU_UUIDS[variant]} epochs=800"
    )

    run_dir = RESULT_ROOT / "NUDT-SIRST" / variant / RUN_NAME
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise RuntimeError(f"{variant}: missing regular metrics stream")
    events = read_metrics_stably(metrics_path)
    if not events:
        raise RuntimeError(f"{variant}: empty metrics stream")
    if len(events) > 800:
        raise RuntimeError(f"{variant}: more than 800 metric events")
    expected_epochs = list(range(1, len(events) + 1))
    actual_epochs = [event.get("epoch") for event in events]
    if actual_epochs != expected_epochs:
        raise RuntimeError(f"{variant}: non-contiguous epoch stream")
    for index, event in enumerate(events, start=1):
        if event.get("variant") != variant:
            raise RuntimeError(f"{variant}: wrong variant at epoch {index}")
        assert_finite(event, f"{variant}.epoch_{index}")

    age_seconds = now - metrics_path.stat().st_mtime
    active = properties.get("ActiveState")
    summary_path = run_dir / "summary.json"
    complete = False
    if unit_collected:
        if not summary_path.is_file() or summary_path.is_symlink():
            raise RuntimeError(f"{variant}: collected unit without summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "complete"
            or summary.get("variant") != variant
            or summary.get("dataset") != "NUDT-SIRST"
            or summary.get("seed") != 42
            or summary.get("selection_source") != "internal_validation_only"
            or summary.get("official_test_accessed") is not False
            or len(events) != 800
            or journal.splitlines().count(completion_marker) != 1
        ):
            raise RuntimeError(
                f"{variant}: collected unit lacks complete artifact+journal proof"
            )
        active = "inactive_collected"
        complete = True
    elif active in {"active", "activating", "deactivating", "reloading"}:
        if len(events) < 800 and age_seconds > stale_seconds:
            raise RuntimeError(
                f"{variant}: metrics stalled for {age_seconds:.1f} seconds"
            )
    elif active == "inactive":
        if (
            properties.get("Result") != "success"
            or properties.get("ExecMainStatus") != "0"
        ):
            raise RuntimeError(f"{variant}: inactive without successful exit")
        if not summary_path.is_file() or summary_path.is_symlink():
            raise RuntimeError(f"{variant}: successful exit without summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "complete"
            or summary.get("variant") != variant
            or summary.get("dataset") != "NUDT-SIRST"
            or summary.get("seed") != 42
            or summary.get("selection_source") != "internal_validation_only"
            or len(events) != 800
            or summary.get("official_test_accessed") is not False
            or journal.splitlines().count(completion_marker) != 1
        ):
            raise RuntimeError(f"{variant}: invalid completion summary")
        complete = True
    elif active == "failed":
        raise RuntimeError(f"{variant}: systemd unit failed")
    else:
        raise RuntimeError(f"{variant}: unexpected ActiveState={active!r}")

    return {
        "variant": variant,
        "unit": unit,
        "invocation_id": expected_invocation,
        "active_state": active,
        "sub_state": properties.get("SubState"),
        "main_pid": int(properties.get("MainPID") or 0),
        "n_restarts": int(properties.get("NRestarts") or 0),
        "unit_collected_after_exit": unit_collected,
        "event_count": len(events),
        "last_epoch": int(events[-1]["epoch"]),
        "metrics_age_seconds": age_seconds,
        "last_train_loss": events[-1].get("train_loss"),
        "last_pd": events[-1].get("pd"),
        "last_tiny_pd": events[-1].get("tiny_pd"),
        "last_fa": events[-1].get("fa"),
        "complete": complete,
        "metrics_contiguous_and_finite": True,
        "journal_forbidden_markers_absent": True,
    }


def write_snapshot(payload: Dict[str, Any]) -> None:
    output = RESULT_ROOT / "launch/live_health.json"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def inspect_once(stale_seconds: int) -> bool:
    now = time.time()
    disk = shutil.disk_usage(RESULT_ROOT)
    if disk.free < 50 * 1024**3:
        raise RuntimeError(f"Less than 50 GiB free under {RESULT_ROOT}")
    variants = [
        inspect_variant(variant, now, stale_seconds) for variant in VARIANTS
    ]
    complete = all(item["complete"] for item in variants)
    payload = {
        "schema": "sctransnet_formal800_4x5090_live_health_v1",
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result_root": str(RESULT_ROOT),
        "disk_free_bytes": disk.free,
        "all_complete": complete,
        "variants": variants,
    }
    write_snapshot(payload)
    print(
        "FORMAL4X5090_WATCHDOG_OK "
        + " ".join(
            f"{item['variant']}={item['last_epoch']}/800:{item['active_state']}"
            for item in variants
        )
        + f" disk_free_gib={disk.free / 1024**3:.1f}",
        flush=True,
    )
    return complete


def main() -> None:
    args = parse_args()
    while True:
        if inspect_once(args.stale_seconds):
            print("FORMAL4X5090_WATCHDOG_COMPLETE training_units_complete=1", flush=True)
            return
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
