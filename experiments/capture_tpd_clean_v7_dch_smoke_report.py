#!/usr/bin/env python3
"""Persist one source-bound V7-DCH CPU or physical-GPU smoke report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.smoke_tpd_clean_v7_dch import run_smoke  # noqa: E402
from model.tpd_clean_v7_dch import (  # noqa: E402
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
)


SCHEMA = "sctransnet_tpd_clean_v7_dch_persisted_smoke_v1"
SOURCE_PATHS: Mapping[str, Path] = {
    "model": REPO_ROOT / "model/tpd_clean_v7_dch.py",
    "train": REPO_ROOT / "experiments/train_tpd_clean_v7_dch.py",
    "smoke": REPO_ROOT / "experiments/smoke_tpd_clean_v7_dch.py",
    "shared_v6_smoke": (
        REPO_ROOT / "experiments/smoke_tpd_clean_v6.py"
    ),
    "shared_v3_smoke": (
        REPO_ROOT / "experiments/smoke_tpd_clean_v3.py"
    ),
    "smoke_test": REPO_ROOT / "tests/test_smoke_tpd_clean_v7_dch.py",
    "capture": Path(__file__).resolve(),
    "verifier": (
        REPO_ROOT
        / "experiments/verify_tpd_clean_v7_dch_smoke_reports.py"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    records: dict[str, str] = {}
    for path in SOURCE_PATHS.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"invalid DCH smoke source file: {path}")
        records[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    return records


def build_envelope(
    report: Mapping[str, Any],
    *,
    source_sha256: Mapping[str, str],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    created = (
        created_at_utc
        if created_at_utc is not None
        else dt.datetime.now(dt.timezone.utc).isoformat()
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "created_at_utc": created,
        "source_sha256": dict(source_sha256),
        "environment_cuda_visible_devices": report.get(
            "environment_cuda_visible_devices"
        ),
        "cuda_visible_devices": report.get("cuda_visible_devices"),
        "cuda_device_contract": report.get("cuda_device_contract"),
        "report": dict(report),
    }


def exclusive_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite DCH smoke report: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FileNotFoundError(
            f"invalid DCH smoke report parent: {path.parent}"
        )
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist a source-bound V7-DCH smoke report"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_CLEAN_V7_DCH_VARIANTS,
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-device-name", default=None)
    parser.add_argument("--expected-cuda-visible-devices", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = (
        args.output if args.output.is_absolute() else REPO_ROOT / args.output
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite DCH smoke report: {output}"
        )
    sources_before = source_manifest()
    report = run_smoke(
        variant=args.variant,
        device_text=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        expected_device_name=args.expected_device_name,
        expected_cuda_visible_devices=args.expected_cuda_visible_devices,
    )
    sources_after = source_manifest()
    if sources_after != sources_before:
        raise RuntimeError(
            "V7-DCH smoke sources changed while the report was running"
        )
    exclusive_write_json(
        output,
        build_envelope(report, source_sha256=sources_before),
    )
    print(
        "TPDCLEANV7DCH_PERSISTED_SMOKE_OK"
        f" device={report['device']}"
        f" variants={len(report['variants'])}"
        f" output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
