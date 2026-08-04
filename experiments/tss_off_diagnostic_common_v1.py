#!/usr/bin/env python3
"""Shared, side-effect-safe helpers for the three-dataset TSS-off stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
CHECKPOINT_ROLES = ("best_miou", "best_pd")
POSITIVE_LAMBDAS = (0.0025, 0.005, 0.01)
POSITIVE_TOKENS = {
    0.0025: "0p0025",
    0.005: "0p005",
    0.01: "0p01",
}
POSITIVE_RESULTS_ROOT = (
    REPO_ROOT / "results" / "three_dataset_seed42_global_tss_v2"
)
TSS_OFF_RESULTS_ROOT = (
    REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
)
DATASET_ROOT = REPO_ROOT / "datasets"
DATA_PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "results"
    / "three_dataset_v2"
    / "manifests"
    / "three_dataset_v2_protocol.json"
)


class TSSOffDiagnosticError(RuntimeError):
    """Raised when a frozen TSS-off input or artifact contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TSSOffDiagnosticError(message)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TSSOffDiagnosticError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise TSSOffDiagnosticError(f"non-finite JSON constant: {value}")


def parse_json(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TSSOffDiagnosticError(f"invalid JSON in {label}: {exc}") from exc


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"invalid JSON file: {path}")
    try:
        value = parse_json(path.read_text(encoding="utf-8"), label=str(path))
    except OSError as exc:
        raise TSSOffDiagnosticError(f"cannot read JSON file {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"invalid JSONL file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                require(bool(line.strip()), f"blank JSONL line: {path}:{line_number}")
                value = parse_json(line, label=f"{path}:{line_number}")
                require(
                    isinstance(value, dict),
                    f"JSONL row must be an object: {path}:{line_number}",
                )
                yield line_number, value
    except OSError as exc:
        raise TSSOffDiagnosticError(f"cannot read JSONL file {path}: {exc}") from exc


def file_sha256(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"invalid source file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_once_or_identical(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    expected = canonical_json_bytes(value)
    if path.exists():
        require(
            path.is_file() and not path.is_symlink(),
            f"write-once destination is invalid: {path}",
        )
        require(
            path.read_bytes() == expected,
            f"write-once artifact conflicts with current inputs: {path}",
        )
        return "reused_identical"
    atomic_write_json(path, value)
    return "written"


def artifact_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"invalid artifact: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def positive_run_directory(root: Path, dataset: str, value: float) -> Path:
    require(dataset in DATASETS, f"unsupported dataset: {dataset}")
    require(value in POSITIVE_TOKENS, f"unsupported positive lambda: {value}")
    return (
        Path(root)
        / "runs"
        / dataset
        / "final"
        / f"lambda_{POSITIVE_TOKENS[value]}"
        / "seed_42"
    )


def tss_off_run_directory(root: Path, dataset: str) -> Path:
    require(dataset in DATASETS, f"unsupported dataset: {dataset}")
    return Path(root) / "runs" / dataset / "final_tss_off" / "seed_42"
