#!/usr/bin/env python3
"""Persist an exclusive, source-bound TPD-Clean-v3 smoke report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SOURCE = REPO_ROOT / "experiments/smoke_tpd_clean_v3.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.smoke_tpd_clean_v3 import run_smoke  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist a TPD-Clean-v3 CPU or CUDA smoke report"
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-device-name", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output
        if args.output.is_absolute()
        else (REPO_ROOT / args.output)
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke report: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FileNotFoundError(f"invalid smoke report parent: {output.parent}")
    report = run_smoke(
        variant="all",
        device_text=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        steps=args.steps,
        seed=args.seed,
        expected_device_name=args.expected_device_name,
    )
    envelope = {
        "schema": "sctransnet_tpd_clean_v3_persisted_smoke_v1",
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "capture_source": str(Path(__file__).resolve()),
        "capture_source_sha256": _sha256(Path(__file__).resolve()),
        "smoke_source": str(SMOKE_SOURCE.resolve()),
        "smoke_source_sha256": _sha256(SMOKE_SOURCE),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "report": report,
    }
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
    print(
        "TPDCLEANV3_PERSISTED_SMOKE_OK"
        f" device={report['device']}"
        f" variants={len(report['variants'])}"
        f" output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
