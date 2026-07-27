#!/usr/bin/env python3
"""Create the reviewable source lock consumed by the V6 exact entry.

Importing this module only defines functions.  The lock is written exclusively
when :func:`freeze_source_lock` or the command-line ``main`` is called.
"""

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

from experiments import train_tpd_clean_v6_exact as exact_entry  # noqa: E402


DEFAULT_DATASET = "NUDT-SIRST"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the TPD-Clean-v6 exact source/data contract"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=exact_entry.DEFAULT_EXACT_SOURCE_LOCK_PATH,
    )
    args = parser.parse_args(argv)
    if (
        not args.dataset
        or Path(args.dataset).name != args.dataset
        or args.dataset in {".", ".."}
    ):
        parser.error("--dataset must be one directory name")
    return args


def runtime_source_sha256() -> dict[str, str]:
    """Hash every source declared by the exact entry in declaration order."""

    records: dict[str, str] = {}
    for source in exact_entry.RUNTIME_SOURCE_PATHS:
        source = Path(source)
        try:
            relative = str(source.relative_to(exact_entry.REPO_ROOT))
        except ValueError as exc:
            raise ValueError(
                f"runtime source lies outside the repository: {source}"
            ) from exc
        if relative in records:
            raise ValueError(f"duplicate runtime source: {relative}")
        records[relative] = exact_entry.file_sha256(source)
    if not records:
        raise ValueError("the V6 exact runtime source set is empty")
    return records


def build_source_lock_payload(
    *,
    dataset: str = DEFAULT_DATASET,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, Any]:
    """Build a deterministic lock payload without writing any file."""

    if (
        not dataset
        or Path(dataset).name != dataset
        or dataset in {".", ".."}
    ):
        raise ValueError("dataset must be one directory name")
    dataset_dir = Path(dataset_dir).resolve()
    dataset_root = dataset_dir / dataset
    index_bytes, identifiers = exact_entry.read_official_training_index(
        dataset_root,
        dataset,
    )
    training_data_sha256 = exact_entry.official_training_data_sha256(
        dataset_root,
        dataset,
        identifiers,
        index_bytes,
    )
    return {
        "schema": exact_entry.EXACT_SOURCE_LOCK_SCHEMA,
        "variants": list(exact_entry.SUPPORTED_CLEAN_V6_VARIANTS),
        "formal_contract": exact_entry.formal_contract(),
        "dataset": dataset,
        "official_training_index": f"img_idx/train_{dataset}.txt",
        "official_training_sample_count": len(identifiers),
        "training_data_sha256": training_data_sha256,
        "source_sha256": runtime_source_sha256(),
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
    """Write one new regular JSON file and never replace an existing path."""

    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise NotADirectoryError(
            f"output parent must be an existing regular directory: {parent}"
        )

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
    *,
    dataset: str = DEFAULT_DATASET,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output: Path = exact_entry.DEFAULT_EXACT_SOURCE_LOCK_PATH,
) -> tuple[Path, dict[str, Any]]:
    """Build and exclusively write one V6 exact source lock."""

    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    payload = build_source_lock_payload(
        dataset=dataset,
        dataset_dir=dataset_dir,
    )
    return write_new_json(output, payload), payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output, payload = freeze_source_lock(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        output=args.output,
    )
    print(
        f"WROTE {output} "
        f"training_data_sha256={payload['training_data_sha256']} "
        f"sources={len(payload['source_sha256'])}",
        flush=True,
    )


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_DATASET_DIR",
    "build_source_lock_payload",
    "freeze_source_lock",
    "main",
    "parse_args",
    "runtime_source_sha256",
    "write_new_json",
]


if __name__ == "__main__":
    main()
