#!/usr/bin/env python3
"""Read-only PBDR-V4 projection of the three frozen V3 train splits.

This module deliberately has no dataset or data-protocol imports.  It reads
only the three registered V3 ``split_manifest.json`` files, verifies their
bytes and semantic identities, and projects the bindings needed by later V4
model-selection stages.  It never reconstructs a split and never opens an
image, mask, ``img_idx`` file, or official-test index.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sctransnet_pbdr_v4_split_authority_v1/v1"
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")


class PBDRV4SplitAuthorityError(ValueError):
    """A registered V3 split manifest violated its frozen authority."""


@dataclass(frozen=True, slots=True)
class SourceManifestAuthority:
    relative_path: str
    schema: str
    bytes: int
    file_sha256: str
    canonical_split_sha256: str
    official_train_count: int
    development_train_count: int
    internal_validation_count: int
    official_train_ids_sha256: str
    development_train_ids_sha256: str
    internal_validation_ids_sha256: str


SOURCE_AUTHORITIES: Mapping[str, SourceManifestAuthority] = {
    "NUAA-SIRST": SourceManifestAuthority(
        relative_path=(
            "results/nuaa_pbdr_v3_stage1_v1/formal/best_miou/core/"
            "split_manifest.json"
        ),
        schema="sctransnet_nuaa_pbdr_v3_internal_split/v1",
        bytes=56_707,
        file_sha256=(
            "318a7bfb1c692f08f3dfbfc26de5e78f3ac7282e99e9e59e91fd71e7e308fde1"
        ),
        canonical_split_sha256=(
            "99b7d54a0b01a5cb02a12491ad1131cd28c42a9aa43b02728cb1899b0e76713f"
        ),
        official_train_count=213,
        development_train_count=170,
        internal_validation_count=43,
        official_train_ids_sha256=(
            "5cc9a267490af5f5230f34dd64fa872484761956fa730679406208b6ac7253fb"
        ),
        development_train_ids_sha256=(
            "49f67623cb89a2fecc1567e64f1f69209086fbd06afe700931c149afe662fadb"
        ),
        internal_validation_ids_sha256=(
            "c0d0f94075a22f016a1f92fb358d19cc1ef656e04579753cf5f73af8a0d742a9"
        ),
    ),
    "NUDT-SIRST": SourceManifestAuthority(
        relative_path=(
            "results/two_dataset_pbdr_v3_stage1_v1/runs/NUDT-SIRST/"
            "formal/best_miou/core/split_manifest.json"
        ),
        schema="sctransnet_two_dataset_pbdr_v3_internal_split_v1/v1",
        bytes=170_498,
        file_sha256=(
            "f492dc93a0e689786b0d9cfd6d695ff33f441e46f1f5eaab728b9be95bd57fe8"
        ),
        canonical_split_sha256=(
            "86a0637e9b62f1e25d44cbf8ab470e1f702fc3a836d6600b0b862c7d27c7e246"
        ),
        official_train_count=663,
        development_train_count=530,
        internal_validation_count=133,
        official_train_ids_sha256=(
            "4cf3882265e4f0a55e80d58e5e53e5f9a12ed721b6995a42a5e8320ad6f51c75"
        ),
        development_train_ids_sha256=(
            "7f9bb7f6a4d1b14801953ff00e1b426f50f4372b7c663a061b0e2cad06b6f829"
        ),
        internal_validation_ids_sha256=(
            "83bab5bfff68ed823b85055b9451d723518592fe73b78a4812e4ef4e6f419bef"
        ),
    ),
    "IRSTD-1K": SourceManifestAuthority(
        relative_path=(
            "results/two_dataset_pbdr_v3_stage1_v1/runs/IRSTD-1K/"
            "formal/best_miou/core/split_manifest.json"
        ),
        schema="sctransnet_two_dataset_pbdr_v3_internal_split_v1/v1",
        bytes=205_421,
        file_sha256=(
            "8bb2a0cb7cf7802c62ec54e1a43d8ff2524c1c2d45c5ffeb84cb88850f8bdeb4"
        ),
        canonical_split_sha256=(
            "9371a6be7a2671010a3eb014ef4763c97a6528757b2085e262e82237b2e14bac"
        ),
        official_train_count=800,
        development_train_count=640,
        internal_validation_count=160,
        official_train_ids_sha256=(
            "681e4d741fb857703471d6555faa0d86e931aa790567c28f4254331ea9ba3d95"
        ),
        development_train_ids_sha256=(
            "f4144772f01f7d373b450dd856c618f62d7aa0b3e1a4d14e47f7afff359ed589"
        ),
        internal_validation_ids_sha256=(
            "22ceba9e2af438a66470f2cdd73a27586fdf1bed760f9d3eb12fdb80e4133734"
        ),
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4SplitAuthorityError(message)


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4SplitAuthorityError(
            f"value cannot be encoded as canonical JSON: {error}"
        ) from error
    return encoded + (b"\n" if trailing_newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ordered_ids_sha256(identifiers: Sequence[str]) -> str:
    _require(
        not isinstance(identifiers, (str, bytes))
        and all(type(identifier) is str for identifier in identifiers),
        "ordered IDs must be a sequence of strings",
    )
    return canonical_sha256(list(identifiers))


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4SplitAuthorityError(
            f"source manifest must be a regular non-symlink file: {candidate}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_path(dataset_name: str) -> Path:
    try:
        authority = SOURCE_AUTHORITIES[dataset_name]
    except KeyError as error:
        raise PBDRV4SplitAuthorityError(
            f"unsupported dataset: {dataset_name!r}"
        ) from error
    path = REPO_ROOT / authority.relative_path
    _require(not path.is_symlink(), f"source manifest is a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(REPO_ROOT)
    except (FileNotFoundError, ValueError) as error:
        raise PBDRV4SplitAuthorityError(
            f"source manifest is missing or outside the repository: {path}"
        ) from error
    _require(
        relative.as_posix() == authority.relative_path,
        f"source manifest canonical path differs for {dataset_name}",
    )
    _require(resolved.is_file(), f"source manifest is not a file: {resolved}")
    return resolved


def _identifier_list(payload: Mapping[str, Any], field: str) -> list[str]:
    value = payload.get(field)
    _require(isinstance(value, list), f"{field} must be a list")
    _require(
        all(type(identifier) is str and identifier for identifier in value),
        f"{field} contains an invalid ID",
    )
    _require(len(value) == len(set(value)), f"{field} contains duplicate IDs")
    return list(value)


def validate_split_payload(
    dataset_name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one parsed V3 split against the hard-coded authority."""

    _require(dataset_name in SOURCE_AUTHORITIES, "unsupported dataset")
    _require(isinstance(payload, Mapping), "split manifest must be a mapping")
    authority = SOURCE_AUTHORITIES[dataset_name]
    _require(payload.get("schema") == authority.schema, "split schema differs")
    _require(payload.get("dataset") == dataset_name, "split dataset differs")
    _require(
        payload.get("source_split") == "official_train_only",
        "split is not sourced only from official train",
    )
    _require(
        payload.get("official_test_index_opened") is False,
        "source split claims official-test access",
    )
    _require(payload.get("split_seed") == 20260722, "split seed differs")
    _require(payload.get("val_fraction") == 0.20, "validation fraction differs")

    declared = payload.get("split_sha256")
    _require(
        type(declared) is str and len(declared) == 64,
        "declared split SHA-256 is malformed",
    )
    unsigned = dict(payload)
    del unsigned["split_sha256"]
    recomputed = canonical_sha256(unsigned)
    _require(declared == recomputed, "canonical split SHA-256 does not replay")
    _require(
        recomputed == authority.canonical_split_sha256,
        "canonical split SHA-256 differs from frozen authority",
    )

    official = _identifier_list(payload, "official_train_ids")
    development = _identifier_list(payload, "development_train_ids")
    validation = _identifier_list(payload, "internal_validation_ids")
    expected_counts = (
        authority.official_train_count,
        authority.development_train_count,
        authority.internal_validation_count,
    )
    _require(
        (len(official), len(development), len(validation)) == expected_counts,
        "train/development/validation counts differ",
    )
    _require(
        not set(development) & set(validation)
        and set(development) | set(validation) == set(official),
        "development/validation IDs do not partition official train",
    )

    observed_hashes = {
        "official_train_ids": ordered_ids_sha256(official),
        "development_train_ids": ordered_ids_sha256(development),
        "internal_validation_ids": ordered_ids_sha256(validation),
    }
    expected_hashes = {
        "official_train_ids": authority.official_train_ids_sha256,
        "development_train_ids": authority.development_train_ids_sha256,
        "internal_validation_ids": authority.internal_validation_ids_sha256,
    }
    _require(
        observed_hashes == expected_hashes,
        "ordered ID hashes differ from frozen authority",
    )
    return {
        "dataset": dataset_name,
        "canonical_split_sha256": recomputed,
        "counts": {
            "official_train": len(official),
            "development_train": len(development),
            "internal_validation": len(validation),
        },
        "ordered_id_sha256": observed_hashes,
    }


def validate_source_manifest(dataset_name: str) -> dict[str, Any]:
    """Validate one registered source file without touching any dataset data."""

    path = source_manifest_path(dataset_name)
    authority = SOURCE_AUTHORITIES[dataset_name]
    stat = path.stat()
    _require(stat.st_size == authority.bytes, "source manifest byte count differs")
    observed_file_sha = file_sha256(path)
    _require(
        observed_file_sha == authority.file_sha256,
        "source manifest file SHA-256 differs",
    )
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4SplitAuthorityError(
            f"cannot read source manifest {path}: {error}"
        ) from error
    _require(isinstance(payload, dict), "source manifest must contain one object")
    validated = validate_split_payload(dataset_name, payload)
    return {
        **validated,
        "source_path": str(path),
        "source_relative_path": authority.relative_path,
        "source_bytes": stat.st_size,
        "source_file_sha256": observed_file_sha,
    }


def build_projection() -> dict[str, Any]:
    """Build a deterministic read-only projection of all three authorities."""

    datasets: dict[str, Any] = {}
    for dataset_name in DATASETS:
        record = validate_source_manifest(dataset_name)
        datasets[dataset_name] = {
            **record,
            "model_selection_only": True,
            "parent_seen_official_train": True,
            "official_test_accessed": False,
        }
    projection: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_v3_split_authority_projection",
        "source_policy": "read_only_existing_v3_split_manifests",
        "dataset_order": list(DATASETS),
        "model_selection_only": True,
        "parent_seen_official_train": True,
        "official_test_accessed": False,
        "split_reconstruction_performed": False,
        "datasets": datasets,
    }
    projection["projection_sha256"] = canonical_sha256(projection)
    return projection


def write_projection_once(
    destination: Path,
    projection: Mapping[str, Any] | None = None,
) -> Path:
    """Create one canonical projection with O_EXCL; never overwrite or follow."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"projection destination already exists: {path}")
    expected = build_projection()
    ready = dict(expected if projection is None else projection)
    _require(
        ready == expected,
        "supplied projection differs from the live frozen authority",
    )
    declared = ready.get("projection_sha256")
    unsigned = dict(ready)
    unsigned.pop("projection_sha256", None)
    _require(
        type(declared) is str and declared == canonical_sha256(unsigned),
        "projection SHA-256 differs",
    )
    content = canonical_json_bytes(ready, trailing_newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return path.resolve(strict=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    projection = build_projection()
    if args.plan:
        print(canonical_json_bytes(projection, trailing_newline=False).decode("utf-8"))
        return
    assert args.output is not None
    print(write_projection_once(args.output, projection))


if __name__ == "__main__":
    main()


__all__ = [
    "DATASETS",
    "PBDRV4SplitAuthorityError",
    "REPO_ROOT",
    "SCHEMA",
    "SOURCE_AUTHORITIES",
    "SourceManifestAuthority",
    "build_projection",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_sha256",
    "ordered_ids_sha256",
    "source_manifest_path",
    "validate_source_manifest",
    "validate_split_payload",
    "write_projection_once",
]
