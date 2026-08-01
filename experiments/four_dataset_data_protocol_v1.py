"""Frozen four-regime data protocol for the seed-42 paper experiments.

This module is intentionally independent from the legacy ``dataset.py``.  It
provides the small, strict primitives shared by the new paper runner,
evaluator, manifest builders, and tests:

* exact existing ``img_idx`` loading and fingerprint checks;
* source-aware sample IDs for the SIRST3 concatenation;
* a non-destructive overlay for ``NUAA-SIRST::Misc_111``;
* frozen legacy normalization;
* stateless crop/augmentation plans derived from SHA-256; and
* correction-aware pair auditing and exact TSS crop statistics.

Importing the module does not traverse datasets or write files.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image


PROTOCOL_VERSION = 1
PROTOCOL_SEED = 42
PATCH_SIZE = 256
PAD_MULTIPLE = 32
TSS_DOWNSAMPLE = 16
TRAIN_POSITIVE_CROP_PROBABILITY = 0.5

TRAINING_REGIMES = (
    "SIRST3",
    "NUAA-SIRST",
    "NUDT-SIRST",
    "IRSTD-1K",
)
SOURCE_DATASETS = (
    "NUAA-SIRST",
    "NUDT-SIRST",
    "IRSTD-1K",
)
SPLITS = ("train", "test")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "four_dataset_seed42_v1"
DEFAULT_MANIFEST_DIR = DEFAULT_RESULTS_ROOT / "manifests"
DEFAULT_CORRECTION_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "nuaa_misc111_correction_v1.json"
)
DEFAULT_IMGIDX_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "four_dataset_imgidx_v1.json"
)
DEFAULT_PAIR_AUDIT = (
    DEFAULT_MANIFEST_DIR / "four_dataset_pair_audit_v1.json"
)
DEFAULT_PAIR_RECORDS = (
    DEFAULT_MANIFEST_DIR / "four_dataset_pair_records_v1.jsonl"
)
DEFAULT_NORMALIZATION_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "four_dataset_legacy_norm_v1.json"
)
DEFAULT_DATA_GATE = (
    DEFAULT_MANIFEST_DIR / "four_dataset_data_gate_v1.json"
)
DEFAULT_TSS_MANIFEST = (
    DEFAULT_MANIFEST_DIR / "four_dataset_tss_seed42_v1.json"
)

CORRECTION_KEY = "NUAA-SIRST::Misc_111"
CORRECTION_ID = "nuaa_misc111_alignment_v1"
EXPECTED_MISC111 = {
    "image_sha256": (
        "72561a22b2d1e09a167563f1f3dab7ee04153aabd87579df749ca15ecf3e60b1"
    ),
    "raw_mask_sha256": (
        "1bec16e5b0413d08f5b01c70faac97c72454586b03d10129fde778db4194a4aa"
    ),
    "corrected_mask_sha256": (
        "7e20ff7267737f367d2ea0545289152710225fe871d7c34c34b2d97c66b06fff"
    ),
    "image_size_width_height": [325, 220],
    "raw_mask_size_width_height": [592, 400],
    "corrected_mask_size_width_height": [325, 220],
}

EXPECTED_SPLITS: dict[str, dict[str, dict[str, Any]]] = {
    "SIRST3": {
        "train": {
            "count": 1676,
            "file_sha256": (
                "75c32b896b95e29b89edc1f5231f619f275c2b54da0264934e6e0df13d7e7d9a"
            ),
        },
        "test": {
            "count": 1079,
            "file_sha256": (
                "67a0f48b536ea6e2f8c895868c4bcd16c66c7c0a6280fd05ef7cd366d78b8922"
            ),
        },
    },
    "NUAA-SIRST": {
        "train": {
            "count": 213,
            "file_sha256": (
                "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4"
            ),
        },
        "test": {
            "count": 214,
            "file_sha256": (
                "e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e"
            ),
        },
    },
    "NUDT-SIRST": {
        "train": {
            "count": 663,
            "file_sha256": (
                "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3"
            ),
        },
        "test": {
            "count": 664,
            "file_sha256": (
                "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e"
            ),
        },
    },
    "IRSTD-1K": {
        "train": {
            "count": 800,
            "file_sha256": (
                "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744"
            ),
        },
        "test": {
            "count": 201,
            "file_sha256": (
                "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579"
            ),
        },
    },
}

LEGACY_NORMALIZATION: dict[str, dict[str, float]] = {
    "SIRST3": {
        "mean": 101.06385040283203,
        "std": 34.619606018066406,
    },
    "NUAA-SIRST": {
        "mean": 101.06385040283203,
        "std": 34.619606018066406,
    },
    "NUDT-SIRST": {
        "mean": 107.80905151367188,
        "std": 33.02274703979492,
    },
    "IRSTD-1K": {
        "mean": 87.4661865234375,
        "std": 39.71953201293945,
    },
}

_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_SUFFIXES = (".png", ".bmp")


class FourDatasetProtocolError(ValueError):
    """The requested operation violates the frozen data protocol."""


def _fail(message: str) -> None:
    raise FourDatasetProtocolError(message)


def _require_dataset(dataset_name: str) -> str:
    if dataset_name not in TRAINING_REGIMES:
        _fail(
            f"dataset must be one of {TRAINING_REGIMES}, got {dataset_name!r}"
        )
    return dataset_name


def _require_split(split: str) -> str:
    if split not in SPLITS:
        _fail(f"split must be one of {SPLITS}, got {split!r}")
    return split


def _require_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        _fail("seed must be an integer")
    if seed != PROTOCOL_SEED:
        _fail(f"formal protocol has one seed only: {PROTOCOL_SEED}")
    return seed


def _require_patch_size(patch_size: int) -> int:
    if isinstance(patch_size, bool) or not isinstance(patch_size, int):
        _fail("patch_size must be an integer")
    if patch_size != PATCH_SIZE:
        _fail(f"formal protocol patch_size must be {PATCH_SIZE}")
    return patch_size


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"expected a regular, non-symlink file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"value cannot be encoded as canonical JSON: {exc}")
    return f"{text}\n".encode("utf-8")


def compact_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"value cannot be encoded as compact JSON: {exc}")


def _atomic_write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def write_canonical_json(path: str | os.PathLike[str], payload: Any) -> Path:
    return _atomic_write_bytes(Path(path), canonical_json_bytes(payload))


def load_json_object(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular, non-symlink JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label} must contain one JSON object")
    return payload


def _safe_relative_path(root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} must be a non-empty relative path")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        _fail(f"{label} must not escape dataset_root: {relative!r}")
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / rel).resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        _fail(f"{label} escapes dataset_root: {relative!r}")
    return candidate


def _relative_to(path: Path, root: Path, *, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError):
        _fail(f"{label} is outside {root}: {path}")


def index_path(
    dataset_root: str | os.PathLike[str],
    dataset_name: str,
    split: str,
) -> Path:
    dataset_name = _require_dataset(dataset_name)
    split = _require_split(split)
    return (
        Path(dataset_root)
        / dataset_name
        / "img_idx"
        / f"{split}_{dataset_name}.txt"
    )


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    """Fingerprint the parsed order, independent of CRLF/LF file encoding."""

    return sha256_bytes(compact_json_bytes(list(ids)))


def load_index(
    dataset_root: str | os.PathLike[str],
    dataset_name: str,
    split: str,
    *,
    verify_expected: bool = True,
) -> list[str]:
    """Load one existing split without sorting, sampling, or rewriting it."""

    dataset_name = _require_dataset(dataset_name)
    split = _require_split(split)
    path = index_path(dataset_root, dataset_name, split)
    if path.is_symlink() or not path.is_file():
        _fail(f"index must be a regular, non-symlink file: {path}")
    content = path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"index is not UTF-8: {path}: {exc}")
    identifiers = text.splitlines()
    if not identifiers:
        _fail(f"index is empty: {path}")
    for position, identifier in enumerate(identifiers):
        if (
            identifier != identifier.strip()
            or _SAMPLE_ID_RE.fullmatch(identifier) is None
        ):
            _fail(
                f"unsafe or non-canonical sample ID at {path}:{position + 1}: "
                f"{identifier!r}"
            )
    if len(identifiers) != len(set(identifiers)):
        _fail(f"index contains duplicate IDs: {path}")
    if verify_expected:
        expected = EXPECTED_SPLITS[dataset_name][split]
        if len(identifiers) != expected["count"]:
            _fail(
                f"{dataset_name} {split} count mismatch: "
                f"{len(identifiers)} != {expected['count']}"
            )
        observed_sha = sha256_bytes(content)
        if observed_sha != expected["file_sha256"]:
            _fail(
                f"{dataset_name} {split} index SHA-256 mismatch: "
                f"{observed_sha} != {expected['file_sha256']}"
            )
    return identifiers


def load_frozen_index(
    dataset_root: str | os.PathLike[str],
    dataset_name: str,
    split: str,
    manifest: (
        str | os.PathLike[str] | Mapping[str, Any] | None
    ) = None,
) -> list[str]:
    """Load an index and, when available, bind it to the frozen manifest."""

    identifiers = load_index(dataset_root, dataset_name, split)
    if manifest is None:
        if DEFAULT_IMGIDX_MANIFEST.is_file():
            manifest = DEFAULT_IMGIDX_MANIFEST
        else:
            return identifiers
    payload = load_json_object(manifest, label="img_idx manifest")
    if payload.get("schema") != "sctransnet_four_dataset_imgidx/v1":
        _fail("unsupported img_idx manifest schema")
    regimes = payload.get("regimes")
    if not isinstance(regimes, Mapping):
        _fail("img_idx manifest regimes must be an object")
    regime = regimes.get(dataset_name)
    if not isinstance(regime, Mapping):
        _fail(f"img_idx manifest has no regime {dataset_name}")
    splits = regime.get("splits")
    if not isinstance(splits, Mapping):
        _fail(f"img_idx manifest {dataset_name}.splits must be an object")
    record = splits.get(split)
    if not isinstance(record, Mapping):
        _fail(f"img_idx manifest has no {dataset_name} {split} record")
    if record.get("ids") != identifiers:
        _fail(f"{dataset_name} {split} IDs differ from frozen manifest")
    path = index_path(dataset_root, dataset_name, split)
    if record.get("count") != len(identifiers):
        _fail(f"{dataset_name} {split} count differs from frozen manifest")
    if record.get("file_sha256") != sha256_file(path):
        _fail(
            f"{dataset_name} {split} file SHA-256 differs from frozen manifest"
        )
    if record.get("ordered_ids_sha256") != ordered_ids_sha256(identifiers):
        _fail(
            f"{dataset_name} {split} ID-order SHA-256 differs from manifest"
        )
    return identifiers


@lru_cache(maxsize=8)
def _source_membership_cached(dataset_root_resolved: str) -> dict[str, str]:
    root = Path(dataset_root_resolved)
    membership: dict[str, str] = {}
    for source_dataset in SOURCE_DATASETS:
        for split in SPLITS:
            for identifier in load_index(root, source_dataset, split):
                prior = membership.get(identifier)
                if prior is not None and prior != source_dataset:
                    _fail(
                        f"sample ID {identifier!r} is ambiguous between "
                        f"{prior} and {source_dataset}"
                    )
                membership[identifier] = source_dataset
    return membership


def source_dataset_for_sample(
    dataset_root: str | os.PathLike[str],
    dataset_name: str,
    sample_id: str,
) -> str:
    dataset_name = _require_dataset(dataset_name)
    if _SAMPLE_ID_RE.fullmatch(sample_id) is None:
        _fail(f"invalid sample ID: {sample_id!r}")
    if dataset_name != "SIRST3":
        return dataset_name
    root = Path(dataset_root).resolve(strict=True)
    source = _source_membership_cached(str(root)).get(sample_id)
    if source is None:
        _fail(f"SIRST3 sample is not in any source index: {sample_id!r}")
    return source


def namespaced_sample_id(source_dataset: str, sample_id: str) -> str:
    if source_dataset not in SOURCE_DATASETS:
        _fail(f"invalid source dataset: {source_dataset!r}")
    if _SAMPLE_ID_RE.fullmatch(sample_id) is None:
        _fail(f"invalid sample ID: {sample_id!r}")
    return f"{source_dataset}::{sample_id}"


def _find_unique_data_file(directory: Path, sample_id: str) -> Path:
    candidates = [
        directory / f"{sample_id}{suffix}"
        for suffix in _SUPPORTED_SUFFIXES
        if (directory / f"{sample_id}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        _fail(
            f"expected exactly one image file for {sample_id!r} in "
            f"{directory}, found {len(candidates)}"
        )
    if candidates[0].is_symlink():
        _fail(f"dataset files must not be symlinks: {candidates[0]}")
    return candidates[0].resolve(strict=True)


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def validate_correction_manifest(
    payload: Mapping[str, Any],
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    verify_files: bool = True,
) -> dict[str, Any]:
    result = dict(payload)
    if result.get("schema") != "sctransnet_four_dataset_corrections/v1":
        _fail("unsupported correction manifest schema")
    if result.get("manifest_id") != "nuaa_misc111_correction_v1":
        _fail("unexpected correction manifest_id")
    if result.get("raw_data_modified") is not False:
        _fail("correction manifest must state raw_data_modified=false")
    corrections = result.get("corrections")
    if not isinstance(corrections, Mapping) or set(corrections) != {
        CORRECTION_KEY
    }:
        _fail(f"correction manifest must contain only {CORRECTION_KEY}")
    entry = corrections[CORRECTION_KEY]
    if not isinstance(entry, Mapping):
        _fail(f"{CORRECTION_KEY} correction must be an object")
    if entry.get("correction_id") != CORRECTION_ID:
        _fail("unexpected Misc_111 correction_id")
    if entry.get("dataset") != "NUAA-SIRST":
        _fail("Misc_111 correction dataset must be NUAA-SIRST")
    if entry.get("sample_id") != "Misc_111":
        _fail("Misc_111 correction sample_id mismatch")
    if entry.get("applies_to_splits") != ["test"]:
        _fail("Misc_111 correction must apply only to the test split")
    expected_paths = {
        "image_relpath": "NUAA-SIRST/images/Misc_111.png",
        "raw_mask_relpath": "NUAA-SIRST/masks/Misc_111.png",
        "corrected_mask_relpath": "SIRST3/masks/Misc_111.png",
    }
    for field, expected_path in expected_paths.items():
        if entry.get(field) != expected_path:
            _fail(f"{field} must be {expected_path!r}")
    for field, expected_hash in (
        ("image_sha256", EXPECTED_MISC111["image_sha256"]),
        ("raw_mask_sha256", EXPECTED_MISC111["raw_mask_sha256"]),
        (
            "corrected_mask_sha256",
            EXPECTED_MISC111["corrected_mask_sha256"],
        ),
    ):
        observed = _validate_sha256(entry.get(field), label=field)
        if observed != expected_hash:
            _fail(f"{field} differs from the frozen correction contract")
    for field in (
        "image_size_width_height",
        "raw_mask_size_width_height",
        "corrected_mask_size_width_height",
    ):
        if entry.get(field) != EXPECTED_MISC111[field]:
            _fail(f"{field} differs from the frozen correction contract")
    if verify_files:
        root = Path(dataset_root)
        for path_field, hash_field in (
            ("image_relpath", "image_sha256"),
            ("raw_mask_relpath", "raw_mask_sha256"),
            ("corrected_mask_relpath", "corrected_mask_sha256"),
        ):
            path = _safe_relative_path(
                root,
                str(entry[path_field]),
                label=path_field,
            )
            if sha256_file(path) != entry[hash_field]:
                _fail(f"{path_field} content differs from correction manifest")
        image_path = _safe_relative_path(
            root, str(entry["image_relpath"]), label="image_relpath"
        )
        raw_mask_path = _safe_relative_path(
            root, str(entry["raw_mask_relpath"]), label="raw_mask_relpath"
        )
        corrected_path = _safe_relative_path(
            root,
            str(entry["corrected_mask_relpath"]),
            label="corrected_mask_relpath",
        )
        with (
            Image.open(image_path) as image,
            Image.open(raw_mask_path) as raw_mask,
            Image.open(corrected_path) as corrected,
        ):
            if list(image.size) != entry["image_size_width_height"]:
                _fail("Misc_111 image dimensions changed")
            if list(raw_mask.size) != entry["raw_mask_size_width_height"]:
                _fail("Misc_111 raw mask dimensions changed")
            if (
                list(corrected.size)
                != entry["corrected_mask_size_width_height"]
            ):
                _fail("Misc_111 corrected mask dimensions changed")
            if image.size != corrected.size:
                _fail("Misc_111 correction does not align with the image")
    return result


def load_correction_manifest(
    source: (
        str
        | os.PathLike[str]
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    required: bool = False,
    verify_files: bool = True,
) -> dict[str, Any] | None:
    if source is None:
        if DEFAULT_CORRECTION_MANIFEST.is_file():
            source = DEFAULT_CORRECTION_MANIFEST
        elif required:
            _fail(
                "correction manifest is required but has not been built: "
                f"{DEFAULT_CORRECTION_MANIFEST}"
            )
        else:
            return None
    payload = load_json_object(source, label="correction manifest")
    return validate_correction_manifest(
        payload,
        dataset_root=dataset_root,
        verify_files=verify_files,
    )


@dataclass(frozen=True)
class ResolvedSample:
    dataset_name: str
    source_dataset: str
    sample_id: str
    namespaced_sample_id: str
    image_path: Path
    raw_mask_path: Path
    mask_path: Path
    correction_id: str | None

    @property
    def correction_applied(self) -> bool:
        return self.correction_id is not None


def resolve_sample(
    dataset_root: str | os.PathLike[str],
    dataset_name: str,
    sample_id: str,
    correction_manifest: (
        str | os.PathLike[str] | Mapping[str, Any] | None
    ) = None,
    *,
    split: str | None = None,
) -> ResolvedSample:
    """Resolve one image/mask pair through the versioned correction overlay."""

    dataset_name = _require_dataset(dataset_name)
    if split is not None:
        _require_split(split)
    root = Path(dataset_root).resolve(strict=True)
    source_dataset = source_dataset_for_sample(root, dataset_name, sample_id)
    namespaced_id = namespaced_sample_id(source_dataset, sample_id)
    data_directory = root / dataset_name
    image_path = _find_unique_data_file(
        data_directory / "images", sample_id
    )
    raw_mask_path = _find_unique_data_file(
        data_directory / "masks", sample_id
    )
    mask_path = raw_mask_path
    correction_id: str | None = None

    manifest = load_correction_manifest(
        correction_manifest,
        dataset_root=root,
        required=(namespaced_id == CORRECTION_KEY and dataset_name != "SIRST3"),
        verify_files=(namespaced_id == CORRECTION_KEY),
    )
    if manifest is not None and dataset_name != "SIRST3":
        entry = manifest["corrections"].get(namespaced_id)
        if entry is not None:
            applies = entry["applies_to_splits"]
            if split is None or split in applies:
                expected_raw = _safe_relative_path(
                    root,
                    entry["raw_mask_relpath"],
                    label="raw_mask_relpath",
                )
                if raw_mask_path != expected_raw:
                    _fail("resolved raw mask differs from correction manifest")
                mask_path = _safe_relative_path(
                    root,
                    entry["corrected_mask_relpath"],
                    label="corrected_mask_relpath",
                )
                correction_id = str(entry["correction_id"])

    return ResolvedSample(
        dataset_name=dataset_name,
        source_dataset=source_dataset,
        sample_id=sample_id,
        namespaced_sample_id=namespaced_id,
        image_path=image_path,
        raw_mask_path=raw_mask_path,
        mask_path=mask_path,
        correction_id=correction_id,
    )


def validate_pair(
    sample: ResolvedSample,
    *,
    include_hashes: bool = False,
) -> dict[str, Any]:
    """Validate exact image/mask alignment; never crop or resize mismatches."""

    with Image.open(sample.image_path) as image:
        image_size = image.size
        image_mode = image.mode
    with Image.open(sample.raw_mask_path) as raw_mask:
        raw_mask_size = raw_mask.size
        raw_mask_mode = raw_mask.mode
    with Image.open(sample.mask_path) as mask:
        mask_size = mask.size
        mask_mode = mask.mode
    if image_size != mask_size:
        _fail(
            "image/mask dimensions differ after correction resolution for "
            f"{sample.dataset_name}::{sample.sample_id}: "
            f"image={image_size}, mask={mask_size}"
        )
    record: dict[str, Any] = {
        "image_size_width_height": list(image_size),
        "raw_mask_size_width_height": list(raw_mask_size),
        "effective_mask_size_width_height": list(mask_size),
        "image_mode": image_mode,
        "raw_mask_mode": raw_mask_mode,
        "effective_mask_mode": mask_mode,
        "correction_applied": sample.correction_applied,
        "correction_id": sample.correction_id,
    }
    if include_hashes:
        record.update(
            {
                "image_sha256": sha256_file(sample.image_path),
                "raw_mask_sha256": sha256_file(sample.raw_mask_path),
                "effective_mask_sha256": sha256_file(sample.mask_path),
            }
        )
    return record


def build_nuaa_misc111_correction_manifest(
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    output_path: str | os.PathLike[str] = DEFAULT_CORRECTION_MANIFEST,
) -> dict[str, Any]:
    """Freeze the existing corrected SIRST3 mask as a read-only overlay.

    No image or mask is copied, resized, deleted, or overwritten.
    """

    root = Path(dataset_root).resolve(strict=True)
    entry = {
        "correction_id": CORRECTION_ID,
        "dataset": "NUAA-SIRST",
        "sample_id": "Misc_111",
        "applies_to_splits": ["test"],
        "image_relpath": "NUAA-SIRST/images/Misc_111.png",
        "raw_mask_relpath": "NUAA-SIRST/masks/Misc_111.png",
        "corrected_mask_relpath": "SIRST3/masks/Misc_111.png",
        **EXPECTED_MISC111,
        "operation": "path_overlay_only",
        "raw_mask_preserved": True,
        "corrected_mask_provenance": (
            "existing verified SIRST3/Misc_111 mask"
        ),
    }
    payload = {
        "schema": "sctransnet_four_dataset_corrections/v1",
        "manifest_id": "nuaa_misc111_correction_v1",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_root": str(root),
        "raw_data_modified": False,
        "correction_count": 1,
        "corrections": {CORRECTION_KEY: entry},
    }
    validated = validate_correction_manifest(
        payload,
        dataset_root=root,
        verify_files=True,
    )
    write_canonical_json(output_path, validated)
    return validated


def build_legacy_normalization_manifest(
    *,
    output_path: str | os.PathLike[str] = DEFAULT_NORMALIZATION_MANIFEST,
) -> dict[str, Any]:
    payload = {
        "schema": "sctransnet_four_dataset_legacy_normalization/v1",
        "manifest_id": "four_dataset_legacy_norm_v1",
        "protocol_version": PROTOCOL_VERSION,
        "training_seed": PROTOCOL_SEED,
        "entries": {
            dataset_name: {
                **values,
                "source": "legacy hard-coded get_img_norm_cfg mapping",
                "recomputed_for_this_protocol": False,
            }
            for dataset_name, values in LEGACY_NORMALIZATION.items()
        },
        "sirst3_provenance_note": (
            "SIRST3 intentionally reuses the legacy NUAA-SIRST values; "
            "these values were not recomputed from 1676 SIRST3 train images."
        ),
        "evaluation_rule": (
            "normalization is selected by train_dataset_name; every official "
            "source evaluation of an SIRST3 checkpoint therefore uses SIRST3"
        ),
    }
    write_canonical_json(output_path, payload)
    return payload


def get_legacy_normalization(
    dataset_name: str,
    manifest: (
        str | os.PathLike[str] | Mapping[str, Any] | None
    ) = None,
) -> dict[str, float]:
    dataset_name = _require_dataset(dataset_name)
    if manifest is None:
        if DEFAULT_NORMALIZATION_MANIFEST.is_file():
            manifest = DEFAULT_NORMALIZATION_MANIFEST
        else:
            return dict(LEGACY_NORMALIZATION[dataset_name])
    payload = load_json_object(manifest, label="normalization manifest")
    if (
        payload.get("schema")
        != "sctransnet_four_dataset_legacy_normalization/v1"
    ):
        _fail("unsupported normalization manifest schema")
    entries = payload.get("entries")
    if not isinstance(entries, Mapping):
        _fail("normalization manifest entries must be an object")
    entry = entries.get(dataset_name)
    if not isinstance(entry, Mapping):
        _fail(f"normalization manifest has no entry for {dataset_name}")
    result: dict[str, float] = {}
    for field in ("mean", "std"):
        value = entry.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"{dataset_name} normalization {field} is not numeric")
        result[field] = float(value)
        if result[field] != LEGACY_NORMALIZATION[dataset_name][field]:
            _fail(
                f"{dataset_name} normalization {field} differs from frozen "
                "legacy value"
            )
    if result["std"] <= 0:
        _fail(f"{dataset_name} normalization std must be positive")
    return result


def build_imgidx_manifest(
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    output_path: str | os.PathLike[str] = DEFAULT_IMGIDX_MANIFEST,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve(strict=True)
    regimes: dict[str, Any] = {}
    loaded: dict[tuple[str, str], list[str]] = {}
    for dataset_name in TRAINING_REGIMES:
        split_records: dict[str, Any] = {}
        for split in SPLITS:
            ids = load_index(root, dataset_name, split)
            loaded[(dataset_name, split)] = ids
            path = index_path(root, dataset_name, split)
            split_records[split] = {
                "index_relpath": _relative_to(
                    path, root, label="index file"
                ),
                "count": len(ids),
                "file_sha256": sha256_file(path),
                "ordered_ids_sha256": ordered_ids_sha256(ids),
                "ordered_ids_sha256_contract": (
                    "sha256(canonical compact JSON array of parsed IDs)"
                ),
                "ids": ids,
            }
        train_namespaced = {
            namespaced_sample_id(
                source_dataset_for_sample(root, dataset_name, sample_id),
                sample_id,
            )
            for sample_id in loaded[(dataset_name, "train")]
        }
        test_namespaced = {
            namespaced_sample_id(
                source_dataset_for_sample(root, dataset_name, sample_id),
                sample_id,
            )
            for sample_id in loaded[(dataset_name, "test")]
        }
        overlap = sorted(train_namespaced & test_namespaced)
        if overlap:
            _fail(
                f"{dataset_name} train/test overlap contains "
                f"{len(overlap)} samples"
            )
        regimes[dataset_name] = {
            "splits": split_records,
            "train_test_disjoint": True,
            "train_test_overlap_count": 0,
            "sampling": "natural_frequency_concat_then_shuffle",
        }

    source_ranges: dict[str, dict[str, list[int]]] = {}
    for split in SPLITS:
        concatenated: list[str] = []
        offset = 0
        source_ranges[split] = {}
        for source_dataset in SOURCE_DATASETS:
            source_ids = loaded[(source_dataset, split)]
            concatenated.extend(source_ids)
            source_ranges[split][source_dataset] = [
                offset,
                offset + len(source_ids),
            ]
            offset += len(source_ids)
        if loaded[("SIRST3", split)] != concatenated:
            _fail(
                f"SIRST3 {split} is not the strict ordered concatenation of "
                f"{SOURCE_DATASETS}"
            )

    payload = {
        "schema": "sctransnet_four_dataset_imgidx/v1",
        "manifest_id": "four_dataset_imgidx_v1",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_root": str(root),
        "dataset_split_source": "existing_img_idx",
        "dataset_split_seed": "not_applicable",
        "training_seed": PROTOCOL_SEED,
        "training_regimes": list(TRAINING_REGIMES),
        "source_datasets": list(SOURCE_DATASETS),
        "regimes": regimes,
        "sirst3_strict_source_concatenation": True,
        "sirst3_source_ranges_start_inclusive_end_exclusive": source_ranges,
        "four_training_settings_are_not_four_independent_sources": True,
        "test_selected": True,
    }
    write_canonical_json(output_path, payload)
    return payload


def _mask_array(path: Path) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        _fail(f"NumPy is required for mask auditing: {exc}")
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim > 2:
        array = array[:, :, 0]
    if array.ndim != 2:
        _fail(f"mask must be two-dimensional after channel selection: {path}")
    return array


def _component_count_8(binary: Any) -> int:
    """Count 8-connected foreground components in a sparse binary mask."""

    try:
        import numpy as np
    except ImportError as exc:
        _fail(f"NumPy is required for target counting: {exc}")
    coordinates = np.argwhere(binary)
    remaining = {(int(row), int(column)) for row, column in coordinates}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            row, column = stack.pop()
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor = (
                        row + row_delta,
                        column + column_delta,
                    )
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return count


def _image_file_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "width": width,
        "height": height,
        "mode": mode,
        "pixel_count": width * height,
    }


def _mask_file_metadata(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        _fail(f"NumPy is required for mask auditing: {exc}")
    array = _mask_array(path)
    height, width = array.shape
    nonzero = array > 0
    foreground = array >= 128
    with Image.open(path) as image:
        mode = image.mode
    return {
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "pixel_count": int(width * height),
        "nonzero_pixel_count": int(np.count_nonzero(nonzero)),
        "foreground_pixel_count_ge_128": int(
            np.count_nonzero(foreground)
        ),
        "target_count_8_connected_ge_128": _component_count_8(foreground),
    }


def _aggregate_record_hash(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            compact_json_bytes({field: record[field] for field in fields})
        )
        digest.update(b"\n")
    return digest.hexdigest()


def audit_four_dataset_pairs(
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    correction_manifest: (
        str | os.PathLike[str] | Mapping[str, Any]
    ) = DEFAULT_CORRECTION_MANIFEST,
    output_path: str | os.PathLike[str] = DEFAULT_PAIR_AUDIT,
    records_path: str | os.PathLike[str] = DEFAULT_PAIR_RECORDS,
) -> dict[str, Any]:
    """Audit all four settings and freeze correction-aware pair fingerprints."""

    root = Path(dataset_root).resolve(strict=True)
    corrections = load_correction_manifest(
        correction_manifest,
        dataset_root=root,
        required=True,
        verify_files=True,
    )
    image_cache: dict[Path, dict[str, Any]] = {}
    mask_cache: dict[Path, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for dataset_name in TRAINING_REGIMES:
        summaries[dataset_name] = {}
        for split in SPLITS:
            split_records: list[dict[str, Any]] = []
            for position, sample_id in enumerate(
                load_index(root, dataset_name, split)
            ):
                sample = resolve_sample(
                    root,
                    dataset_name,
                    sample_id,
                    corrections,
                    split=split,
                )
                pair = validate_pair(sample)
                image_meta = image_cache.get(sample.image_path)
                if image_meta is None:
                    image_meta = _image_file_metadata(sample.image_path)
                    image_cache[sample.image_path] = image_meta
                raw_mask_meta = mask_cache.get(sample.raw_mask_path)
                if raw_mask_meta is None:
                    raw_mask_meta = _mask_file_metadata(
                        sample.raw_mask_path
                    )
                    mask_cache[sample.raw_mask_path] = raw_mask_meta
                effective_mask_meta = mask_cache.get(sample.mask_path)
                if effective_mask_meta is None:
                    effective_mask_meta = _mask_file_metadata(
                        sample.mask_path
                    )
                    mask_cache[sample.mask_path] = effective_mask_meta
                record = {
                    "dataset_name": dataset_name,
                    "split": split,
                    "position": position,
                    "sample_id": sample.sample_id,
                    "namespaced_sample_id": sample.namespaced_sample_id,
                    "source_dataset": sample.source_dataset,
                    "image_relpath": _relative_to(
                        sample.image_path, root, label="image"
                    ),
                    "raw_mask_relpath": _relative_to(
                        sample.raw_mask_path, root, label="raw mask"
                    ),
                    "effective_mask_relpath": _relative_to(
                        sample.mask_path, root, label="effective mask"
                    ),
                    "image_sha256": image_meta["sha256"],
                    "raw_mask_sha256": raw_mask_meta["sha256"],
                    "effective_mask_sha256": effective_mask_meta["sha256"],
                    "width": image_meta["width"],
                    "height": image_meta["height"],
                    "pixel_count": image_meta["pixel_count"],
                    "foreground_pixel_count_ge_128": effective_mask_meta[
                        "foreground_pixel_count_ge_128"
                    ],
                    "nonzero_pixel_count": effective_mask_meta[
                        "nonzero_pixel_count"
                    ],
                    "target_count_8_connected_ge_128": effective_mask_meta[
                        "target_count_8_connected_ge_128"
                    ],
                    "raw_mask_width": raw_mask_meta["width"],
                    "raw_mask_height": raw_mask_meta["height"],
                    "correction_applied": pair["correction_applied"],
                    "correction_id": pair["correction_id"],
                }
                records.append(record)
                split_records.append(record)
            summary = {
                "sample_count": len(split_records),
                "pixel_count": sum(
                    record["pixel_count"] for record in split_records
                ),
                "foreground_pixel_count_ge_128": sum(
                    record["foreground_pixel_count_ge_128"]
                    for record in split_records
                ),
                "nonzero_pixel_count": sum(
                    record["nonzero_pixel_count"]
                    for record in split_records
                ),
                "target_count_8_connected_ge_128": sum(
                    record["target_count_8_connected_ge_128"]
                    for record in split_records
                ),
                "correction_count": sum(
                    int(record["correction_applied"])
                    for record in split_records
                ),
                "image_sequence_sha256": _aggregate_record_hash(
                    split_records,
                    (
                        "namespaced_sample_id",
                        "image_sha256",
                        "width",
                        "height",
                    ),
                ),
                "raw_mask_sequence_sha256": _aggregate_record_hash(
                    split_records,
                    (
                        "namespaced_sample_id",
                        "raw_mask_sha256",
                        "raw_mask_width",
                        "raw_mask_height",
                    ),
                ),
                "effective_mask_sequence_sha256": _aggregate_record_hash(
                    split_records,
                    (
                        "namespaced_sample_id",
                        "effective_mask_sha256",
                        "width",
                        "height",
                    ),
                ),
                "all_effective_pairs_aligned": True,
            }
            expected_count = EXPECTED_SPLITS[dataset_name][split]["count"]
            if summary["sample_count"] != expected_count:
                _fail(
                    f"{dataset_name} {split} audit count differs from index"
                )
            summaries[dataset_name][split] = summary

    record_by_key = {
        (
            record["dataset_name"],
            record["split"],
            record["sample_id"],
        ): record
        for record in records
    }
    parity_mismatches: list[dict[str, str]] = []
    parity_count = 0
    for source_dataset in SOURCE_DATASETS:
        for split in SPLITS:
            for sample_id in load_index(root, source_dataset, split):
                source_record = record_by_key[
                    (source_dataset, split, sample_id)
                ]
                sirst3_record = record_by_key[
                    ("SIRST3", split, sample_id)
                ]
                parity_count += 1
                if (
                    source_record["image_sha256"]
                    != sirst3_record["image_sha256"]
                    or source_record["effective_mask_sha256"]
                    != sirst3_record["effective_mask_sha256"]
                ):
                    parity_mismatches.append(
                        {
                            "source_dataset": source_dataset,
                            "split": split,
                            "sample_id": sample_id,
                        }
                    )
    if parity_mismatches:
        _fail(
            "SIRST3/source effective data parity failed for "
            f"{len(parity_mismatches)} samples"
        )

    jsonl_content = b"".join(
        compact_json_bytes(record) + b"\n" for record in records
    )
    records_output = _atomic_write_bytes(Path(records_path), jsonl_content)
    correction_records = [
        record for record in records if record["correction_applied"]
    ]
    payload = {
        "schema": "sctransnet_four_dataset_pair_audit/v1",
        "manifest_id": "four_dataset_pair_audit_v1",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_root": str(root),
        "mask_foreground_definition": "uint8 value >= 128",
        "target_count_definition": (
            "8-connected components of uint8 value >= 128"
        ),
        "summaries": summaries,
        "total_regime_split_records": len(records),
        "records_relpath_from_manifest": os.path.relpath(
            records_output, Path(output_path).parent
        ),
        "records_sha256": sha256_bytes(jsonl_content),
        "all_effective_pairs_aligned": True,
        "sirst3_source_content_parity": True,
        "sirst3_source_parity_sample_count": parity_count,
        "sirst3_source_parity_mismatch_count": 0,
        "correction_application_records": [
            {
                "dataset_name": record["dataset_name"],
                "split": record["split"],
                "sample_id": record["sample_id"],
                "correction_id": record["correction_id"],
                "raw_mask_sha256": record["raw_mask_sha256"],
                "effective_mask_sha256": record[
                    "effective_mask_sha256"
                ],
                "effective_size_width_height": [
                    record["width"],
                    record["height"],
                ],
            }
            for record in correction_records
        ],
        "raw_data_modified": False,
    }
    write_canonical_json(output_path, payload)
    return payload


def build_manifests(
    output_root: str | os.PathLike[str] = DEFAULT_MANIFEST_DIR,
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    """Build every data-preparation artifact and a truthful ready gate."""

    output_directory = Path(output_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "correction": (
            output_directory / "nuaa_misc111_correction_v1.json"
        ),
        "imgidx": output_directory / "four_dataset_imgidx_v1.json",
        "normalization": (
            output_directory / "four_dataset_legacy_norm_v1.json"
        ),
        "pair_audit": (
            output_directory / "four_dataset_pair_audit_v1.json"
        ),
        "pair_records": (
            output_directory / "four_dataset_pair_records_v1.jsonl"
        ),
        "tss": output_directory / "four_dataset_tss_seed42_v1.json",
        "gate": output_directory / "four_dataset_data_gate_v1.json",
    }
    root = Path(dataset_root).resolve(strict=True)
    try:
        correction = build_nuaa_misc111_correction_manifest(
            dataset_root=root,
            output_path=paths["correction"],
        )
        imgidx = build_imgidx_manifest(
            dataset_root=root,
            output_path=paths["imgidx"],
        )
        normalization = build_legacy_normalization_manifest(
            output_path=paths["normalization"],
        )
        audit = audit_four_dataset_pairs(
            dataset_root=root,
            correction_manifest=correction,
            output_path=paths["pair_audit"],
            records_path=paths["pair_records"],
        )
    except Exception as exc:
        failure_gate = {
            "schema": "sctransnet_four_dataset_data_gate/v1",
            "manifest_id": "four_dataset_data_gate_v1",
            "protocol_version": PROTOCOL_VERSION,
            "training_seed": PROTOCOL_SEED,
            "dataset_root": str(root),
            "nuaa_dataset_ready": False,
            "four_dataset_suite_ready": False,
            "formal_training_and_evaluation_ready": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        write_canonical_json(paths["gate"], failure_gate)
        raise

    tss_ready = False
    if paths["tss"].is_file():
        validate_exact_tss_statistics_manifest(paths["tss"])
        tss_ready = True
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "gate" and path.is_file()
    }
    gate = {
        "schema": "sctransnet_four_dataset_data_gate/v1",
        "manifest_id": "four_dataset_data_gate_v1",
        "protocol_version": PROTOCOL_VERSION,
        "training_seed": PROTOCOL_SEED,
        "dataset_root": str(root),
        "checks": {
            "all_eight_index_counts_and_file_hashes_match": True,
            "all_four_train_test_splits_disjoint": all(
                regime["train_test_disjoint"]
                for regime in imgidx["regimes"].values()
            ),
            "sirst3_is_strict_source_concatenation": imgidx[
                "sirst3_strict_source_concatenation"
            ],
            "nuaa_misc111_overlay_verified": (
                correction["correction_count"] == 1
            ),
            "nuaa_raw_mask_preserved": (
                correction["raw_data_modified"] is False
            ),
            "all_effective_pairs_aligned": audit[
                "all_effective_pairs_aligned"
            ],
            "sirst3_source_content_parity": audit[
                "sirst3_source_content_parity"
            ],
            "legacy_normalization_frozen": (
                normalization["entries"]
                and set(normalization["entries"]) == set(TRAINING_REGIMES)
            ),
            "exact_tss_statistics_seed42_1000epochs_ready": tss_ready,
        },
        "nuaa_dataset_ready": True,
        "four_dataset_suite_ready": True,
        "formal_training_and_evaluation_ready": tss_ready,
        "artifact_sha256": artifact_hashes,
        "errors": [],
    }
    write_canonical_json(paths["gate"], gate)
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "gate": gate,
    }


def stable_sha256_uint64(*components: Any) -> int:
    """Derive a persistent unsigned 64-bit value without Python ``hash()``."""

    digest = hashlib.sha256()
    digest.update(b"sctransnet-four-dataset-seed-v1\0")
    for component in components:
        encoded = compact_json_bytes(component)
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class StatelessTransformPlan:
    augmentation_seed: int
    crop_top: int
    crop_left: int
    crop_size: int
    padded_height: int
    padded_width: int
    crop_attempts: int
    flip_axis0: bool
    flip_axis1: bool
    transpose: bool


def derive_stateless_transform_plan(
    *,
    protocol_seed: int,
    dataset_name: str,
    epoch: int,
    namespaced_id: str,
    image_height: int,
    image_width: int,
    has_positive_in_crop: Callable[[int, int, int], bool],
    patch_size: int = PATCH_SIZE,
    pos_prob: float = TRAIN_POSITIVE_CROP_PROBABILITY,
) -> StatelessTransformPlan:
    """Reproduce legacy crop/flip draws with a sample-local RNG."""

    _require_seed(protocol_seed)
    _require_dataset(dataset_name)
    _require_patch_size(patch_size)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        _fail("epoch must be a non-negative integer")
    if not isinstance(namespaced_id, str) or "::" not in namespaced_id:
        _fail("namespaced_id must have the form source_dataset::sample_id")
    if image_height < 1 or image_width < 1:
        _fail("image dimensions must be positive")
    if not isinstance(pos_prob, (int, float)) or not 0 <= pos_prob <= 1:
        _fail("pos_prob must be in [0, 1]")
    padded_height = max(image_height, patch_size)
    padded_width = max(image_width, patch_size)
    augmentation_seed = stable_sha256_uint64(
        protocol_seed,
        dataset_name,
        epoch,
        namespaced_id,
    )
    rng = random.Random(augmentation_seed)
    attempts = 0
    while True:
        attempts += 1
        crop_top = rng.randint(0, padded_height - patch_size)
        crop_left = rng.randint(0, padded_width - patch_size)
        unbiased_accept = rng.random() > pos_prob
        if unbiased_accept or has_positive_in_crop(
            crop_top, crop_left, patch_size
        ):
            break
        if attempts >= 1_000_000:
            _fail("stateless positive-biased crop exceeded safety limit")
    return StatelessTransformPlan(
        augmentation_seed=augmentation_seed,
        crop_top=crop_top,
        crop_left=crop_left,
        crop_size=patch_size,
        padded_height=padded_height,
        padded_width=padded_width,
        crop_attempts=attempts,
        flip_axis0=rng.random() < 0.5,
        flip_axis1=rng.random() < 0.5,
        transpose=rng.random() < 0.5,
    )


def dataloader_seed(dataset_name: str, seed: int = PROTOCOL_SEED) -> int:
    _require_dataset(dataset_name)
    _require_seed(seed)
    return stable_sha256_uint64(seed, dataset_name, "dataloader")


def make_dataloader_generator(
    dataset_name: str,
    seed: int = PROTOCOL_SEED,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        _fail(f"PyTorch is required for a DataLoader generator: {exc}")
    generator = torch.Generator()
    generator.manual_seed(dataloader_seed(dataset_name, seed))
    return generator


def _load_mask_coordinates(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        _fail(f"NumPy is required for TSS statistics: {exc}")
    array = _mask_array(path)
    any_rows, any_columns = np.nonzero(array > 0)
    tss_rows, tss_columns = np.nonzero(array >= 128)
    return {
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "any_rows": any_rows,
        "any_columns": any_columns,
        "tss_rows": tss_rows,
        "tss_columns": tss_columns,
        "sha256": sha256_file(path),
    }


def _coordinates_in_crop(
    rows: Any,
    columns: Any,
    top: int,
    left: int,
    size: int,
) -> Any:
    return (
        (rows >= top)
        & (rows < top + size)
        & (columns >= left)
        & (columns < left + size)
    )


def _positive_tss_cells_for_plan(
    mask_info: Mapping[str, Any],
    plan: StatelessTransformPlan,
) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        _fail(f"NumPy is required for TSS statistics: {exc}")
    rows = mask_info["tss_rows"]
    columns = mask_info["tss_columns"]
    selected = _coordinates_in_crop(
        rows,
        columns,
        plan.crop_top,
        plan.crop_left,
        plan.crop_size,
    )
    if not bool(np.any(selected)):
        return 0
    relative_rows = rows[selected] - plan.crop_top
    relative_columns = columns[selected] - plan.crop_left
    cells_per_side = plan.crop_size // TSS_DOWNSAMPLE
    flattened_cells = (
        (relative_rows // TSS_DOWNSAMPLE) * cells_per_side
        + (relative_columns // TSS_DOWNSAMPLE)
    )
    return int(np.unique(flattened_cells).size)


def compute_exact_tss_statistics(
    dataset: str,
    epochs: int = 1000,
    seed: int = PROTOCOL_SEED,
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    correction_manifest: (
        str | os.PathLike[str] | Mapping[str, Any] | None
    ) = None,
    start_epoch: int = 1,
    end_epoch: int | None = None,
    initial_state: (
        str | os.PathLike[str] | Mapping[str, Any] | None
    ) = None,
    progress_path: str | os.PathLike[str] | None = None,
    checkpoint_every: int = 10,
) -> dict[str, Any]:
    """Count exact stride-16 cells over the stateless crop schedule.

    ``initial_state`` plus ``end_epoch`` make the computation resumable.  A
    state is valid only for the same dataset, seed, epoch horizon, train IDs,
    and effective mask fingerprint.
    """

    dataset = _require_dataset(dataset)
    _require_seed(seed)
    _require_patch_size(PATCH_SIZE)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        _fail("epochs must be a positive integer")
    if (
        isinstance(start_epoch, bool)
        or not isinstance(start_epoch, int)
        or start_epoch < 1
    ):
        _fail("start_epoch must be a positive integer")
    if end_epoch is None:
        end_epoch = epochs
    if (
        isinstance(end_epoch, bool)
        or not isinstance(end_epoch, int)
        or end_epoch < start_epoch
        or end_epoch > epochs
    ):
        _fail("end_epoch must satisfy start_epoch <= end_epoch <= epochs")
    if (
        isinstance(checkpoint_every, bool)
        or not isinstance(checkpoint_every, int)
        or checkpoint_every < 1
    ):
        _fail("checkpoint_every must be a positive integer")

    root = Path(dataset_root).resolve(strict=True)
    identifiers = load_index(root, dataset, "train")
    correction_payload = load_correction_manifest(
        correction_manifest,
        dataset_root=root,
        required=False,
        verify_files=False,
    )
    mask_infos: list[tuple[str, str, dict[str, Any]]] = []
    mask_fingerprint_records: list[dict[str, str]] = []
    for sample_id in identifiers:
        sample = resolve_sample(
            root,
            dataset,
            sample_id,
            correction_payload,
            split="train",
        )
        pair = validate_pair(sample)
        if pair["correction_applied"]:
            _fail("formal train splits must not require correction overlays")
        info = _load_mask_coordinates(sample.mask_path)
        mask_infos.append((sample_id, sample.namespaced_sample_id, info))
        mask_fingerprint_records.append(
            {
                "namespaced_sample_id": sample.namespaced_sample_id,
                "effective_mask_sha256": info["sha256"],
            }
        )
    mask_sequence_sha256 = sha256_bytes(
        compact_json_bytes(mask_fingerprint_records)
    )
    train_ids_sha256 = ordered_ids_sha256(identifiers)

    positive_cells = 0
    completed_epoch_digests: list[str] = []
    if initial_state is not None:
        state = load_json_object(initial_state, label="TSS progress state")
        required_values = {
            "schema": "sctransnet_four_dataset_exact_tss_progress/v1",
            "dataset": dataset,
            "training_seed": seed,
            "epochs": epochs,
            "train_ids_sha256": train_ids_sha256,
            "effective_mask_sequence_sha256": mask_sequence_sha256,
        }
        for field, expected in required_values.items():
            if state.get(field) != expected:
                _fail(f"TSS progress state {field} mismatch")
        prior_end = state.get("completed_through_epoch")
        if (
            isinstance(prior_end, bool)
            or not isinstance(prior_end, int)
            or prior_end < 0
        ):
            _fail("TSS progress completed_through_epoch is invalid")
        if start_epoch != prior_end + 1:
            _fail(
                "resumed TSS start_epoch must equal "
                "completed_through_epoch + 1"
            )
        prior_positive = state.get("positive_cells")
        prior_digests = state.get("epoch_plan_sha256")
        if (
            isinstance(prior_positive, bool)
            or not isinstance(prior_positive, int)
            or prior_positive < 0
            or not isinstance(prior_digests, list)
            or len(prior_digests) != prior_end
        ):
            _fail("TSS progress cumulative fields are invalid")
        positive_cells = prior_positive
        completed_epoch_digests = list(prior_digests)
    elif start_epoch != 1:
        _fail("start_epoch > 1 requires initial_state")

    cells_per_sample = (PATCH_SIZE // TSS_DOWNSAMPLE) ** 2
    samples_per_epoch = len(mask_infos)
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_digest = hashlib.sha256()
        for sample_id, namespaced_id, info in mask_infos:
            any_rows = info["any_rows"]
            any_columns = info["any_columns"]

            def has_positive(top: int, left: int, size: int) -> bool:
                try:
                    import numpy as np
                except ImportError as exc:
                    _fail(f"NumPy is required for TSS statistics: {exc}")
                return bool(
                    np.any(
                        _coordinates_in_crop(
                            any_rows,
                            any_columns,
                            top,
                            left,
                            size,
                        )
                    )
                )

            plan = derive_stateless_transform_plan(
                protocol_seed=seed,
                dataset_name=dataset,
                epoch=epoch,
                namespaced_id=namespaced_id,
                image_height=info["height"],
                image_width=info["width"],
                has_positive_in_crop=has_positive,
            )
            positive_cells += _positive_tss_cells_for_plan(info, plan)
            epoch_digest.update(
                compact_json_bytes(
                    {
                        "sample_id": sample_id,
                        **asdict(plan),
                    }
                )
            )
            epoch_digest.update(b"\n")
        completed_epoch_digests.append(epoch_digest.hexdigest())
        if progress_path is not None and (
            epoch % checkpoint_every == 0 or epoch == end_epoch
        ):
            cumulative_total = (
                len(completed_epoch_digests)
                * samples_per_epoch
                * cells_per_sample
            )
            progress = {
                "schema": (
                    "sctransnet_four_dataset_exact_tss_progress/v1"
                ),
                "dataset": dataset,
                "training_seed": seed,
                "epochs": epochs,
                "completed_through_epoch": epoch,
                "train_image_count": samples_per_epoch,
                "train_ids_sha256": train_ids_sha256,
                "effective_mask_sequence_sha256": mask_sequence_sha256,
                "positive_cells": positive_cells,
                "negative_cells": cumulative_total - positive_cells,
                "total_cells": cumulative_total,
                "epoch_plan_sha256": completed_epoch_digests,
            }
            write_canonical_json(progress_path, progress)

    total_cells = (
        len(completed_epoch_digests)
        * samples_per_epoch
        * cells_per_sample
    )
    negative_cells = total_cells - positive_cells
    if positive_cells <= 0 or negative_cells <= 0:
        _fail("TSS positive and negative cells must both be non-zero")
    complete = end_epoch == epochs
    payload = {
        "schema": "sctransnet_four_dataset_exact_tss_progress/v1",
        "dataset": dataset,
        "training_seed": seed,
        "epochs": epochs,
        "completed_through_epoch": end_epoch,
        "complete": complete,
        "train_image_count": samples_per_epoch,
        "train_presentations_counted": (
            len(completed_epoch_digests) * samples_per_epoch
        ),
        "train_ids_sha256": train_ids_sha256,
        "effective_mask_sequence_sha256": mask_sequence_sha256,
        "patch_size": PATCH_SIZE,
        "positive_crop_probability": TRAIN_POSITIVE_CROP_PROBABILITY,
        "tss_mask_binarization": "float(mask)/255 > 0.5",
        "tss_uint8_equivalent": "value >= 128",
        "crop_positive_test": "float(mask)/255 sum > 0",
        "pool_kernel": TSS_DOWNSAMPLE,
        "pool_stride": TSS_DOWNSAMPLE,
        "grid_size": [
            PATCH_SIZE // TSS_DOWNSAMPLE,
            PATCH_SIZE // TSS_DOWNSAMPLE,
        ],
        "cells_per_sample": cells_per_sample,
        "positive_cells": positive_cells,
        "negative_cells": negative_cells,
        "total_cells": total_cells,
        "survival_pos_weight_formula": "negative_cells / positive_cells",
        "survival_pos_weight": negative_cells / positive_cells,
        "epoch_plan_sha256": completed_epoch_digests,
        "aggregate_plan_sha256": sha256_bytes(
            compact_json_bytes(completed_epoch_digests)
        ),
        "stateless_seed_contract": (
            "stable_sha256_uint64(42,dataset_name,epoch,"
            "namespaced_sample_id)"
        ),
        "transforms_preserve_positive_cell_count": True,
    }
    if progress_path is not None:
        write_canonical_json(progress_path, payload)
    return payload


def build_all_exact_tss_statistics(
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    correction_manifest: (
        str | os.PathLike[str] | Mapping[str, Any] | None
    ) = None,
    output_path: str | os.PathLike[str] = DEFAULT_TSS_MANIFEST,
    epochs: int = 1000,
    seed: int = PROTOCOL_SEED,
) -> dict[str, Any]:
    _require_seed(seed)
    datasets = {
        dataset: compute_exact_tss_statistics(
            dataset,
            epochs=epochs,
            seed=seed,
            dataset_root=dataset_root,
            correction_manifest=correction_manifest,
        )
        for dataset in TRAINING_REGIMES
    }
    payload = {
        "schema": "sctransnet_four_dataset_exact_tss_statistics/v1",
        "manifest_id": "four_dataset_tss_seed42_v1",
        "training_seed": seed,
        "epochs": epochs,
        "datasets": datasets,
    }
    write_canonical_json(output_path, payload)
    return payload


def validate_exact_tss_statistics_manifest(
    source: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    payload = load_json_object(source, label="exact TSS statistics manifest")
    if (
        payload.get("schema")
        != "sctransnet_four_dataset_exact_tss_statistics/v1"
    ):
        _fail("unsupported exact TSS statistics schema")
    if payload.get("manifest_id") != "four_dataset_tss_seed42_v1":
        _fail("unexpected exact TSS manifest_id")
    if payload.get("training_seed") != PROTOCOL_SEED:
        _fail("exact TSS statistics must use seed 42")
    if payload.get("epochs") != 1000:
        _fail("formal exact TSS statistics must cover 1000 epochs")
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(
        TRAINING_REGIMES
    ):
        _fail("exact TSS manifest must contain all four training regimes")
    cells_per_sample = (PATCH_SIZE // TSS_DOWNSAMPLE) ** 2
    for dataset_name in TRAINING_REGIMES:
        record = datasets[dataset_name]
        if not isinstance(record, Mapping):
            _fail(f"{dataset_name} exact TSS record must be an object")
        expected_count = EXPECTED_SPLITS[dataset_name]["train"]["count"]
        expected_total = expected_count * 1000 * cells_per_sample
        required = {
            "dataset": dataset_name,
            "training_seed": PROTOCOL_SEED,
            "epochs": 1000,
            "completed_through_epoch": 1000,
            "complete": True,
            "train_image_count": expected_count,
            "train_presentations_counted": expected_count * 1000,
            "cells_per_sample": cells_per_sample,
            "total_cells": expected_total,
        }
        for field, expected in required.items():
            if record.get(field) != expected:
                _fail(
                    f"{dataset_name} exact TSS {field} differs from "
                    f"{expected!r}"
                )
        positive = record.get("positive_cells")
        negative = record.get("negative_cells")
        weight = record.get("survival_pos_weight")
        if (
            isinstance(positive, bool)
            or not isinstance(positive, int)
            or positive <= 0
            or isinstance(negative, bool)
            or not isinstance(negative, int)
            or negative <= 0
            or positive + negative != expected_total
        ):
            _fail(f"{dataset_name} exact TSS cell counts are invalid")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or float(weight) != negative / positive
        ):
            _fail(f"{dataset_name} exact TSS pos_weight is invalid")
        epoch_hashes = record.get("epoch_plan_sha256")
        if (
            not isinstance(epoch_hashes, list)
            or len(epoch_hashes) != 1000
            or any(
                not isinstance(value, str)
                or _SHA256_RE.fullmatch(value) is None
                for value in epoch_hashes
            )
        ):
            _fail(f"{dataset_name} exact TSS epoch plan hashes are invalid")
    return payload


__all__ = [
    "CORRECTION_ID",
    "CORRECTION_KEY",
    "DEFAULT_CORRECTION_MANIFEST",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_IMGIDX_MANIFEST",
    "DEFAULT_MANIFEST_DIR",
    "DEFAULT_NORMALIZATION_MANIFEST",
    "DEFAULT_PAIR_AUDIT",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_TSS_MANIFEST",
    "EXPECTED_SPLITS",
    "FourDatasetProtocolError",
    "LEGACY_NORMALIZATION",
    "PAD_MULTIPLE",
    "PATCH_SIZE",
    "PROTOCOL_SEED",
    "ResolvedSample",
    "SOURCE_DATASETS",
    "StatelessTransformPlan",
    "TRAINING_REGIMES",
    "audit_four_dataset_pairs",
    "build_all_exact_tss_statistics",
    "build_imgidx_manifest",
    "build_legacy_normalization_manifest",
    "build_manifests",
    "build_nuaa_misc111_correction_manifest",
    "canonical_json_bytes",
    "compute_exact_tss_statistics",
    "dataloader_seed",
    "derive_stateless_transform_plan",
    "get_legacy_normalization",
    "load_correction_manifest",
    "load_frozen_index",
    "load_index",
    "make_dataloader_generator",
    "namespaced_sample_id",
    "ordered_ids_sha256",
    "resolve_sample",
    "sha256_file",
    "source_dataset_for_sample",
    "stable_sha256_uint64",
    "validate_pair",
    "validate_exact_tss_statistics_manifest",
    "write_canonical_json",
]
