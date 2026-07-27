#!/usr/bin/env python3
"""Create the independent V6 formal800 postprocess source lock once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v6_postprocess_source_lock_v1"
POSTPROCESS_SOURCE_PATHS = (
    REPO_ROOT / "experiments/summarize_tpd_clean_v6_formal800.py",
    REPO_ROOT / "experiments/run_tpd_clean_v6_formal800_sweeps.py",
    REPO_ROOT / "experiments/validate_tpd_clean_v6_formal800_completion.py",
    REPO_ROOT / "experiments/freeze_tpd_clean_v6_postprocess_source_lock.py",
    REPO_ROOT / "experiments/run_tpd_clean_v6_formal800_finalizer.sh",
    REPO_ROOT / "experiments/launch_tpd_clean_v6_formal800_finalizer.sh",
    REPO_ROOT / "experiments/status_tpd_clean_v6_formal800_finalizer.sh",
    REPO_ROOT / "tests/test_summarize_tpd_clean_v6_formal800.py",
    REPO_ROOT / "tests/test_run_tpd_clean_v6_formal800_sweeps.py",
    REPO_ROOT / "tests/test_validate_tpd_clean_v6_formal800_completion.py",
    REPO_ROOT / "tests/test_freeze_tpd_clean_v6_postprocess_source_lock.py",
    REPO_ROOT / "tests/test_tpd_clean_v6_formal800_finalizer.py",
)
FROZEN_REFERENCE_PATHS = (
    *summary.SPD_REFERENCE_FILES,
    summary.DEFAULT_SMOKE_ROOT / "cpu_all.json",
    summary.DEFAULT_SMOKE_ROOT / "gpu2_full.json",
    summary.DEFAULT_SMOKE_ROOT / "gpu3_capacity.json",
    summary.SMOKE_VERIFICATION,
)


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(f"path lies outside repository: {path}") from exc


def _hash_paths(paths: Sequence[Path]) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        relative = _relative(path)
        if relative in records:
            raise ValueError(f"duplicate lock input: {relative}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"lock input is not a regular file: {path}")
        records[relative] = summary.sha256_file(path)
    if not records:
        raise ValueError("source lock input set is empty")
    return records


def build_source_lock_payload() -> dict[str, Any]:
    training_lock, training_sha = summary._validate_current_training_contract()
    if training_sha != summary.EXPECTED_TRAINING_LOCK_SHA256:
        raise ValueError("unexpected V6 training source-lock digest")
    evaluator_sha = training_lock["source_sha256"][
        "experiments/evaluate_tpd_clean_v6_pd_fa.py"
    ]
    return {
        "schema": SCHEMA,
        "scope": (
            "TPD-Clean-v6 formal800 closed-interval sweeps, Gates A-E, "
            "and exact-input completion publication"
        ),
        "candidate_root": _relative(summary.DEFAULT_CANDIDATE_ROOT),
        "training_source_lock": _relative(
            summary.DEFAULT_TRAINING_SOURCE_LOCK
        ),
        "training_source_lock_sha256": training_sha,
        "training_data_sha256": summary.EXPECTED_TRAINING_DATA_SHA256,
        "evaluator_sha256": evaluator_sha,
        "source_sha256": _hash_paths(POSTPROCESS_SOURCE_PATHS),
        "frozen_reference_sha256": _hash_paths(FROZEN_REFERENCE_PATHS),
        "policy": {
            "separate_from_training_source_lock": True,
            "does_not_modify_frozen_training_sources": True,
            "does_not_modify_training_results": True,
            "formal_report_overwrite_forbidden": True,
            "gate_evaluation_before_four_complete_runs_forbidden": True,
            "candidate_null_budget_points_forbidden": True,
            "automatic_mainline_replacement": False,
        },
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_new_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NotADirectoryError(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    return path


def freeze_source_lock(
    output: Path = summary.DEFAULT_POSTPROCESS_SOURCE_LOCK,
) -> tuple[Path, dict[str, Any]]:
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    payload = build_source_lock_payload()
    return write_new_json(output, payload), payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the independent V6 postprocess source lock"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=summary.DEFAULT_POSTPROCESS_SOURCE_LOCK,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output, payload = freeze_source_lock(args.output)
    print(
        f"WROTE {output} "
        f"training_source_lock_sha256={payload['training_source_lock_sha256']} "
        f"sources={len(payload['source_sha256'])}",
        flush=True,
    )


__all__ = [
    "FROZEN_REFERENCE_PATHS",
    "POSTPROCESS_SOURCE_PATHS",
    "SCHEMA",
    "build_source_lock_payload",
    "freeze_source_lock",
    "main",
    "parse_args",
    "write_new_json",
]


if __name__ == "__main__":
    main()
