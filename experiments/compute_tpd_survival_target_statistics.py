"""Freeze target-cell class statistics for dual-endpoint TSS.

Importing this module performs no argument parsing, file reads, writes, or
dataset traversal.  The command-line entry point requires an explicit output
or verification target.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


SCHEMA = "sctransnet_tpd_survival_target_statistics_v1"
DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
PATCH_SIZE = 256
DOWNSAMPLE = 16
MASK_THRESHOLD = 0.5
MASK_EXTENSION = ".png"

EXPECTED_TRAIN_IMAGE_COUNT = 530
EXPECTED_POSITIVE_CELLS = 1313
EXPECTED_NEGATIVE_CELLS = 134367
EXPECTED_TOTAL_CELLS = 135680
EXPECTED_SURVIVAL_POS_WEIGHT = 102.33587204874334
EXPECTED_USED_TRAIN_IDS_SHA256 = (
    "9565f584a5429fd1e5f0451b2d9496877f6f887493dd4d9954b4e976989f245b"
)
EXPECTED_SPLIT_JSON_SHA256 = (
    "391bca28848038d6386a6c70cbaeb902ba71a8dc73a4a134441ac2aa5b438828"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_RELATIVE = (
    "experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/"
    "NUDT-SIRST/tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/"
    "seed_42_formal800_exact_v4_tail_aware_seed42/split.json"
)
DEFAULT_MASKS_RELATIVE = "datasets/NUDT-SIRST/masks"
DEFAULT_OUTPUT_RELATIVE = (
    "experiments/tpd_survival_target_statistics_nudt_sirst_v1.json"
)
GENERATOR_RELATIVE = "experiments/compute_tpd_survival_target_statistics.py"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[0-9]{6}$")


class TargetStatisticsError(ValueError):
    """The requested statistics operation violates the frozen contract."""


def _fail(message: str) -> None:
    raise TargetStatisticsError(message)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot read {path}: {exc}")
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"statistics payload is not canonical JSON data: {exc}")
    return f"{text}\n".encode("utf-8")


def _compact_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"cannot encode fingerprint payload: {exc}")


def ordered_identifier_sha256(identifiers: Sequence[str]) -> str:
    """Reproduce the frozen split.json identifier-set digest."""

    return _sha256_bytes(
        "\n".join(
            sorted(str(identifier) for identifier in identifiers)
        ).encode("utf-8")
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"{label} is not UTF-8: {exc}")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TargetStatisticsError) as exc:
        _fail(f"{label} is invalid JSON: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    return value


def _resolve_input_file(
    path: str | os.PathLike[str],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"{label} must be a regular non-symlink file: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {label}: {exc}")
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        _fail(f"{label} must be inside repository root {repo_root}")
    return resolved


def _resolve_input_directory(
    path: str | os.PathLike[str],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if candidate.is_symlink() or not candidate.is_dir():
        _fail(f"{label} must be a directory and not a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {label}: {exc}")
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        _fail(f"{label} must be inside repository root {repo_root}")
    return resolved


def _relative_to_repo(path: Path, repo_root: Path, *, label: str) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        _fail(f"{label} is outside repository root {repo_root}")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_used_train_ids(
    split_payload: Mapping[str, Any],
    *,
    validate_formal: bool,
) -> tuple[list[str], str]:
    if split_payload.get("dataset") != DATASET:
        _fail(f"split dataset must be {DATASET!r}")
    split_seed = split_payload.get("split_seed")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        _fail("split_seed must be an integer")
    if validate_formal and split_seed != SPLIT_SEED:
        _fail(f"formal split_seed must be {SPLIT_SEED}")
    if split_payload.get("official_test_accessed") is not False:
        _fail("formal split must record official_test_accessed=false")

    raw_ids = split_payload.get("used_train_ids")
    if not isinstance(raw_ids, list):
        _fail("split.used_train_ids must be a list")
    identifiers: list[str] = []
    for index, value in enumerate(raw_ids):
        if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
            _fail(f"used_train_ids[{index}] must be a six-digit string")
        identifiers.append(value)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        _fail("used_train_ids must be non-empty and unique")
    if split_payload.get("used_train_count") != len(identifiers):
        _fail("split.used_train_count differs from used_train_ids")

    full_ids = split_payload.get("full_internal_train_ids")
    if not isinstance(full_ids, list) or full_ids != identifiers:
        _fail("formal used_train_ids must equal full_internal_train_ids")
    if split_payload.get("full_internal_train_count") != len(identifiers):
        _fail("full_internal_train_count differs from used_train_ids")

    observed_hash = ordered_identifier_sha256(identifiers)
    hashes = split_payload.get("hashes")
    if not isinstance(hashes, Mapping):
        _fail("split.hashes must be a JSON object")
    recorded_hash = _require_sha256(
        hashes.get("used_train_sha256"),
        label="split.hashes.used_train_sha256",
    )
    full_hash = _require_sha256(
        hashes.get("full_internal_train_sha256"),
        label="split.hashes.full_internal_train_sha256",
    )
    if observed_hash != recorded_hash or observed_hash != full_hash:
        _fail("ordered used-train ID SHA-256 differs from split.json")
    if validate_formal:
        if len(identifiers) != EXPECTED_TRAIN_IMAGE_COUNT:
            _fail(
                f"formal used-train count must be {EXPECTED_TRAIN_IMAGE_COUNT}"
            )
        if observed_hash != EXPECTED_USED_TRAIN_IDS_SHA256:
            _fail("formal used-train ID SHA-256 mismatch")
    return identifiers, observed_hash


def max_pool_presence_grid(
    luma: bytes,
    *,
    width: int,
    height: int,
    downsample: int = DOWNSAMPLE,
) -> tuple[tuple[bool, ...], ...]:
    """Return exact binary max-pool presence for row-major 8-bit L pixels."""

    for label, value in (
        ("width", width),
        ("height", height),
        ("downsample", downsample),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            _fail(f"{label} must be a positive integer")
    if len(luma) != width * height:
        _fail("luma byte count differs from width*height")
    if width % downsample or height % downsample:
        _fail("image dimensions must be divisible by downsample")

    rows: list[tuple[bool, ...]] = []
    for cell_y in range(height // downsample):
        pooled_row: list[bool] = []
        y_start = cell_y * downsample
        for cell_x in range(width // downsample):
            x_start = cell_x * downsample
            present = False
            for y_pos in range(y_start, y_start + downsample):
                start = y_pos * width + x_start
                stop = start + downsample
                if any(value >= 128 for value in luma[start:stop]):
                    present = True
                    break
            pooled_row.append(present)
        rows.append(tuple(pooled_row))
    return tuple(rows)


def _field_update(
    digest: "hashlib._Hash",
    label: str,
    content: bytes,
) -> None:
    label_bytes = label.encode("utf-8")
    for part in (label_bytes, content):
        digest.update(len(part).to_bytes(8, byteorder="big", signed=False))
        digest.update(part)


def _read_mask(
    path: Path,
    *,
    identifier: str,
) -> tuple[bytes, int, int, str, int]:
    if path.is_symlink() or not path.is_file():
        _fail(f"mask must be a regular non-symlink file: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read mask {path}: {exc}")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.format != "PNG":
                _fail(f"mask {identifier} must use PNG encoding")
            if image.mode != "L":
                _fail(f"mask {identifier} must use 8-bit grayscale mode L")
            width, height = image.size
            luma = image.tobytes()
    except (OSError, UnidentifiedImageError) as exc:
        _fail(f"cannot decode mask {identifier}: {exc}")
    return luma, width, height, _sha256_bytes(content), len(content)


def _require_formal_results(payload: Mapping[str, Any]) -> None:
    expected = {
        "train_image_count": EXPECTED_TRAIN_IMAGE_COUNT,
        "positive_cells": EXPECTED_POSITIVE_CELLS,
        "negative_cells": EXPECTED_NEGATIVE_CELLS,
        "total_cells": EXPECTED_TOTAL_CELLS,
        "used_train_ids_sha256": EXPECTED_USED_TRAIN_IDS_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            _fail(f"formal {key} mismatch: {payload.get(key)!r} != {value!r}")
    observed_weight = payload.get("survival_pos_weight")
    if (
        not isinstance(observed_weight, float)
        or not math.isfinite(observed_weight)
        or observed_weight != EXPECTED_SURVIVAL_POS_WEIGHT
    ):
        _fail(
            "formal survival_pos_weight mismatch: "
            f"{observed_weight!r} != {EXPECTED_SURVIVAL_POS_WEIGHT!r}"
        )


def compute_statistics(
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    split_json: str | os.PathLike[str] = DEFAULT_SPLIT_RELATIVE,
    masks_dir: str | os.PathLike[str] = DEFAULT_MASKS_RELATIVE,
    validate_formal: bool = True,
) -> dict[str, Any]:
    """Compute an auditable TSS class-statistics contract."""

    root = Path(repo_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        _fail(f"repository root must be a directory and not a symlink: {root}")
    root = root.resolve(strict=True)
    split_path = _resolve_input_file(
        split_json,
        repo_root=root,
        label="frozen V4 split.json",
    )
    masks_path = _resolve_input_directory(
        masks_dir,
        repo_root=root,
        label="NUDT-SIRST masks directory",
    )

    split_content = split_path.read_bytes()
    split_sha256 = _sha256_bytes(split_content)
    if validate_formal and split_sha256 != EXPECTED_SPLIT_JSON_SHA256:
        _fail("frozen V4 split.json SHA-256 mismatch")
    split_payload = _load_json_object(
        split_content,
        label="frozen V4 split.json",
    )
    identifiers, ids_sha256 = _load_used_train_ids(
        split_payload,
        validate_formal=validate_formal,
    )

    data_digest = hashlib.sha256()
    _field_update(
        data_digest,
        "schema",
        b"sctransnet-tpd-survival-mask-data-fingerprint-v1",
    )
    mask_records: list[dict[str, Any]] = []
    positive_cells = 0
    total_mask_bytes = 0
    observed_sizes: set[tuple[int, int]] = set()
    for identifier in identifiers:
        mask_path = masks_path / f"{identifier}{MASK_EXTENSION}"
        luma, width, height, mask_sha256, byte_count = _read_mask(
            mask_path,
            identifier=identifier,
        )
        if (width, height) != (PATCH_SIZE, PATCH_SIZE):
            _fail(
                f"mask {identifier} has size {(width, height)}, "
                f"expected {(PATCH_SIZE, PATCH_SIZE)}"
            )
        relative_mask = _relative_to_repo(
            mask_path,
            root,
            label=f"mask {identifier}",
        )
        raw_content = mask_path.read_bytes()
        if _sha256_bytes(raw_content) != mask_sha256 or len(raw_content) != byte_count:
            _fail(f"mask {identifier} changed while statistics were computed")
        _field_update(data_digest, relative_mask, raw_content)
        grid = max_pool_presence_grid(
            luma,
            width=width,
            height=height,
            downsample=DOWNSAMPLE,
        )
        positive_cells += sum(
            1 for row in grid for present in row if present
        )
        total_mask_bytes += byte_count
        observed_sizes.add((height, width))
        mask_records.append(
            {
                "bytes": byte_count,
                "identifier": identifier,
                "path": relative_mask,
                "sha256": mask_sha256,
            }
        )

    cells_per_image = (PATCH_SIZE // DOWNSAMPLE) ** 2
    total_cells = len(identifiers) * cells_per_image
    negative_cells = total_cells - positive_cells
    if positive_cells <= 0 or negative_cells <= 0:
        _fail("positive and negative cell counts must both be non-zero")
    survival_pos_weight = negative_cells / positive_cells
    mask_manifest_sha256 = _sha256_bytes(_compact_json_bytes(mask_records))
    mask_data_sha256 = data_digest.hexdigest()
    aggregate_data_sha256 = _sha256_bytes(
        _compact_json_bytes(
            {
                "mask_data_sha256": mask_data_sha256,
                "split_json_sha256": split_sha256,
                "used_train_ids_sha256": ids_sha256,
            }
        )
    )

    generator_path = REPO_ROOT / GENERATOR_RELATIVE
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": int(split_payload["split_seed"]),
        "train_image_count": len(identifiers),
        "used_train_ids_sha256": ids_sha256,
        "used_train_ids_sha256_contract": (
            "sha256(UTF-8 '\\n'.join(sorted(used_train_ids)))"
        ),
        "image_size_order": ["height", "width"],
        "image_sizes": [list(size) for size in sorted(observed_sizes)],
        "patch_size": PATCH_SIZE,
        "mask_binarization": "float(mask)/255 > 0.5",
        "mask_uint8_equivalent": "value >= 128",
        "pool_kernel": DOWNSAMPLE,
        "pool_stride": DOWNSAMPLE,
        "pool_padding": 0,
        "pool_dilation": 1,
        "pool_ceil_mode": False,
        "grid_size": [PATCH_SIZE // DOWNSAMPLE, PATCH_SIZE // DOWNSAMPLE],
        "cells_per_image": cells_per_image,
        "full_image_equals_training_crop": True,
        "transform_preserves_positive_cell_count": True,
        "positive_cells": positive_cells,
        "negative_cells": negative_cells,
        "total_cells": total_cells,
        "survival_pos_weight_formula": "negative_cells / positive_cells",
        "survival_pos_weight": survival_pos_weight,
        "inputs": {
            "split_json_path": _relative_to_repo(
                split_path,
                root,
                label="frozen V4 split.json",
            ),
            "split_json_sha256": split_sha256,
            "used_train_ids_field": "used_train_ids",
            "split_recorded_used_train_ids_sha256": (
                split_payload["hashes"]["used_train_sha256"]
            ),
            "masks_directory": _relative_to_repo(
                masks_path,
                root,
                label="masks directory",
            ),
            "mask_extension": MASK_EXTENSION,
            "mask_file_count": len(mask_records),
            "mask_total_file_bytes": total_mask_bytes,
            "ordered_mask_manifest_sha256": mask_manifest_sha256,
            "ordered_mask_data_sha256": mask_data_sha256,
            "ordered_mask_data_sha256_contract": (
                "sha256(length-prefixed UTF-8 field labels and raw values; "
                "used_train_ids order)"
            ),
            "aggregate_input_data_sha256": aggregate_data_sha256,
        },
        "generator": {
            "path": GENERATOR_RELATIVE,
            "sha256": sha256_file(generator_path),
        },
        "validation": {
            "all_masks_regular_png_l": True,
            "all_masks_256x256": observed_sizes == {(PATCH_SIZE, PATCH_SIZE)},
            "formal_v1_expectations_checked": bool(validate_formal),
            "official_test_accessed": False,
        },
    }
    if validate_formal:
        _require_formal_results(payload)
    canonical_json_bytes(payload)
    return payload


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(payload))


def _resolve_output_path(
    path: str | os.PathLike[str],
    *,
    repo_root: Path,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    parent = candidate.parent
    if parent.is_symlink() or not parent.is_dir():
        _fail(f"output parent must be an existing non-symlink directory: {parent}")
    resolved_parent = parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(repo_root)
    except ValueError:
        _fail("output must be inside repository root")
    return resolved_parent / candidate.name


def verify_artifact(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> str:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        _fail(f"verification target must be a regular non-symlink file: {artifact}")
    expected = canonical_json_bytes(payload)
    try:
        observed = artifact.read_bytes()
    except OSError as exc:
        _fail(f"cannot read verification target {artifact}: {exc}")
    if observed != expected:
        _fail(f"statistics artifact differs from recomputed canonical bytes: {artifact}")
    return artifact_sha256(payload)


def publish_or_verify(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> str:
    """Create one artifact once, or verify an existing byte-identical artifact."""

    artifact = Path(path)
    if artifact.is_symlink():
        _fail(f"refusing symlink output: {artifact}")
    if artifact.exists():
        verify_artifact(artifact, payload)
        return "verified_existing"

    content = canonical_json_bytes(payload)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            artifact,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        created = True
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(f"short write while publishing {artifact}")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError:
        verify_artifact(artifact, payload)
        return "verified_existing"
    except OSError as exc:
        if created:
            try:
                artifact.unlink()
            except OSError:
                pass
        _fail(f"cannot publish statistics artifact {artifact}: {exc}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return "created"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute frozen NUDT-SIRST max-pool16 target-cell statistics. "
            "An explicit --output or --verify target is required."
        )
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--split-json", default=DEFAULT_SPLIT_RELATIVE)
    parser.add_argument("--masks-dir", default=DEFAULT_MASKS_RELATIVE)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output",
        help=(
            "Create this canonical JSON once; if it exists, verify exact bytes "
            "without overwriting."
        ),
    )
    destination.add_argument(
        "--verify",
        help="Require this existing JSON to equal the recomputed canonical bytes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        root = Path(args.repo_root).expanduser().resolve(strict=True)
        payload = compute_statistics(
            repo_root=root,
            split_json=args.split_json,
            masks_dir=args.masks_dir,
            validate_formal=True,
        )
        if args.output is not None:
            output = _resolve_output_path(args.output, repo_root=root)
            mode = publish_or_verify(output, payload)
        else:
            output = _resolve_output_path(args.verify, repo_root=root)
            verify_artifact(output, payload)
            mode = "verified"
        status = {
            "artifact_sha256": artifact_sha256(payload),
            "negative_cells": payload["negative_cells"],
            "output": _relative_to_repo(output, root, label="output"),
            "positive_cells": payload["positive_cells"],
            "status": mode,
            "survival_pos_weight": payload["survival_pos_weight"],
            "total_cells": payload["total_cells"],
        }
        print(
            json.dumps(
                status,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    except (OSError, TargetStatisticsError) as exc:
        print(f"target-statistics error: {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "DATASET",
    "DEFAULT_MASKS_RELATIVE",
    "DEFAULT_OUTPUT_RELATIVE",
    "DEFAULT_SPLIT_RELATIVE",
    "DOWNSAMPLE",
    "EXPECTED_NEGATIVE_CELLS",
    "EXPECTED_POSITIVE_CELLS",
    "EXPECTED_SURVIVAL_POS_WEIGHT",
    "EXPECTED_TOTAL_CELLS",
    "EXPECTED_TRAIN_IMAGE_COUNT",
    "EXPECTED_USED_TRAIN_IDS_SHA256",
    "PATCH_SIZE",
    "SCHEMA",
    "TargetStatisticsError",
    "artifact_sha256",
    "canonical_json_bytes",
    "compute_statistics",
    "main",
    "max_pool_presence_grid",
    "ordered_identifier_sha256",
    "publish_or_verify",
    "sha256_file",
    "verify_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
