"""Materialize the NUAA-internal ``Misc_111`` corrected-mask overlay.

The corrected source is supplied explicitly.  It is accepted only when its
SHA-256 and dimensions match the frozen V2 contract.  The command never
overwrites the original NUAA mask; an existing corrected target is accepted
only when it is already byte-identical to the frozen artifact.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from experiments.three_dataset_v2_protocol import (
        DEFAULT_DATASET_ROOT,
        EXPECTED_NUAA_MISC111,
        NUAA_MISC111_PATHS,
        ThreeDatasetV2ProtocolError,
        build_protocol_manifest,
        sha256_file,
        validate_nuaa_misc111_overlay,
    )
except ModuleNotFoundError:
    from three_dataset_v2_protocol import (
        DEFAULT_DATASET_ROOT,
        EXPECTED_NUAA_MISC111,
        NUAA_MISC111_PATHS,
        ThreeDatasetV2ProtocolError,
        build_protocol_manifest,
        sha256_file,
        validate_nuaa_misc111_overlay,
    )


def _fail(message: str) -> None:
    raise ThreeDatasetV2ProtocolError(message)


def _size(path: Path, *, label: str) -> list[int]:
    try:
        with Image.open(path) as image:
            image.load()
            return list(image.size)
    except OSError as exc:
        _fail(f"{label} is not a readable image: {path}: {exc}")


def _verify_corrected_source(source: Path) -> bytes:
    if source.is_symlink() or not source.is_file():
        _fail(f"corrected source must be a regular, non-symlink file: {source}")
    observed_sha = sha256_file(source)
    expected_sha = EXPECTED_NUAA_MISC111["corrected_mask_sha256"]
    if observed_sha != expected_sha:
        _fail(
            f"corrected source SHA-256 mismatch: {observed_sha} != "
            f"{expected_sha}"
        )
    observed_size = _size(source, label="corrected source")
    expected_size = EXPECTED_NUAA_MISC111[
        "corrected_mask_size_width_height"
    ]
    if observed_size != expected_size:
        _fail(
            f"corrected source dimensions mismatch: {observed_size} != "
            f"{expected_size}"
        )
    return source.read_bytes()


def _install_without_overwrite(target: Path, content: bytes) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            _fail(f"corrected target is not a regular file: {target}")
        if sha256_file(target) != EXPECTED_NUAA_MISC111[
            "corrected_mask_sha256"
        ]:
            _fail(f"refusing to overwrite a different corrected target: {target}")
        return "verified_existing"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard link makes creation atomic and cannot overwrite a file
            # installed concurrently by another process.
            os.link(temporary, target)
            action = "created"
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            if target.is_symlink() or not target.is_file():
                _fail(f"concurrent corrected target is not regular: {target}")
            if sha256_file(target) != EXPECTED_NUAA_MISC111[
                "corrected_mask_sha256"
            ]:
                _fail(f"concurrent corrected target has unexpected content: {target}")
            action = "verified_concurrent"
    finally:
        if temporary.exists():
            temporary.unlink()
    return action


def prepare_overlay(
    *,
    dataset_root: str | os.PathLike[str],
    corrected_source: str | os.PathLike[str],
    protocol_manifest_output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve(strict=True)
    source_input = Path(corrected_source)
    if source_input.is_symlink() or not source_input.is_file():
        _fail(
            "corrected source must be a regular, non-symlink file: "
            f"{source_input}"
        )
    source = source_input.resolve(strict=True)
    raw_mask = root / NUAA_MISC111_PATHS["raw_mask_relpath"]
    if raw_mask.is_symlink() or not raw_mask.is_file():
        _fail(f"NUAA raw mask must be a regular, non-symlink file: {raw_mask}")
    raw_sha_before = sha256_file(raw_mask)
    if raw_sha_before != EXPECTED_NUAA_MISC111["raw_mask_sha256"]:
        _fail("NUAA raw mask differs from the frozen pre-install contract")

    content = _verify_corrected_source(source)
    target = root / NUAA_MISC111_PATHS["corrected_mask_relpath"]
    action = _install_without_overwrite(target, content)
    overlay = validate_nuaa_misc111_overlay(root)

    raw_sha_after = sha256_file(raw_mask)
    if raw_sha_after != raw_sha_before:
        _fail("NUAA raw mask changed while preparing the overlay")
    protocol_manifest = None
    if protocol_manifest_output is not None:
        protocol_manifest = build_protocol_manifest(
            dataset_root=root,
            output_path=protocol_manifest_output,
        )
    return {
        "action": action,
        "dataset_root": str(root),
        "corrected_mask_relpath": overlay["corrected_mask_relpath"],
        "image_sha256": overlay["image_sha256"],
        "raw_mask_sha256_before": raw_sha_before,
        "raw_mask_sha256_after": raw_sha_after,
        "corrected_mask_sha256": overlay["corrected_mask_sha256"],
        "raw_mask_preserved": True,
        "protocol_manifest_written": protocol_manifest is not None,
        "protocol_manifest_output": (
            str(Path(protocol_manifest_output))
            if protocol_manifest_output is not None
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--corrected-source",
        type=Path,
        required=True,
        help="verified 325x220 corrected Misc_111 mask to copy",
    )
    parser.add_argument(
        "--protocol-manifest-output",
        type=Path,
        default=None,
        help="optionally build the complete three-dataset protocol lock",
    )
    args = parser.parse_args(argv)
    result = prepare_overlay(
        dataset_root=args.dataset_root,
        corrected_source=args.corrected_source,
        protocol_manifest_output=args.protocol_manifest_output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
