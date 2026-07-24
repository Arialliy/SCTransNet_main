#!/usr/bin/env python3
"""Fingerprint only the samples named by an official training index."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash an IRSTD official-training index and its referenced image/mask files"
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--dataset", default="NUDT-SIRST")
    return parser.parse_args()


def update_field(digest: "hashlib._Hash", label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def resolve_unique_file(directory: Path, identifier: str) -> Path:
    matches = [directory / f"{identifier}{extension}" for extension in IMAGE_EXTENSIONS]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file for {identifier!r} under {directory}, found {len(matches)}"
        )
    return matches[0]


def update_file(digest: "hashlib._Hash", label: str, path: Path) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    size = path.stat().st_size
    digest.update(size.to_bytes(8, "big"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def main() -> None:
    args = parse_args()
    dataset_root = (args.dataset_dir.resolve() / args.dataset)
    index_path = dataset_root / "img_idx" / f"train_{args.dataset}.txt"
    index_bytes = index_path.read_bytes()
    identifiers = [
        line.strip()
        for line in index_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(identifiers) < 2 or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Training index must contain at least two unique identifiers")

    digest = hashlib.sha256()
    update_field(digest, "schema", b"tpd-training-data-fingerprint-v1")
    update_field(digest, f"img_idx/{index_path.name}", index_bytes)
    for identifier in identifiers:
        for directory_name in ("images", "masks"):
            path = resolve_unique_file(dataset_root / directory_name, identifier)
            update_file(
                digest,
                f"{directory_name}/{path.name}",
                path,
            )

    print(digest.hexdigest())


if __name__ == "__main__":
    main()
