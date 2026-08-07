#!/usr/bin/env python3
"""Build and commit a frozen PBDR-V4 component-atlas artifact.

This module has no dataset, split, index, checkpoint, or model-loading logic.
Its pure API accepts an explicitly ordered development-train projection and
already frozen Current probability/target arrays.  It thresholds both arrays
with strict ``> 0.5``, invokes the canonical component matcher, writes the
three categorical ID maps, reopens and validates the complete staged artifact,
and finally commits the directory without replacing an existing path.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from experiments import three_dataset_v2_protocol as data_protocol
from experiments.component_matching_v2 import match_components_v2
from experiments.pbdr_v4_atlas_dataset import (
    ATLAS_MANIFEST_SCHEMA,
    ATLAS_MAP_NAMES,
    ATLAS_NPZ_KEYS,
    ATLAS_SPLIT_SCOPE,
    array_semantic_sha256,
    file_sha256,
    matcher_source_sha256 as active_matcher_source_sha256,
    ordered_ids_sha256,
)
from experiments.pbdr_v4_component_atlas import atlas_maps_from_match


BUILDER_SCHEMA = "sctransnet_pbdr_v4_component_atlas_builder/v1"
MANIFEST_FILENAME = "manifest.json"
FIXED_THRESHOLD = 0.5
ROLES = ("best_miou", "best_pd")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = (
    "image_id",
    "parent_state_sha256",
    "matcher_source_sha256",
)
_COMPONENT_STATISTIC_KEYS = frozenset(
    {
        "target_component_count",
        "matched_target_component_count",
        "unmatched_target_component_count",
        "prediction_component_count",
        "matched_prediction_component_count",
        "unmatched_prediction_component_count",
        "unmatched_prediction_pixel_count",
        "target_positive_pixel_count",
        "prediction_positive_pixel_count",
        "rescue_component_count",
        "suppress_component_count",
        "preserve_component_count",
        "rescue_pixel_count",
        "suppress_pixel_count",
        "preserve_pixel_count",
    }
)


class PBDRV4AtlasBuildError(ValueError):
    """The build input, staged artifact, or commit contract is invalid."""


@dataclass(frozen=True, slots=True)
class FrozenCurrentSample:
    """One already frozen Current probability map and normalized target."""

    probability: np.ndarray = field(repr=False, compare=False)
    target: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AtlasBuildResult:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    sample_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4AtlasBuildError(message)


def _validated_sha256(value: Any, *, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 hex digest",
    )
    return value


def _validated_ids(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("development_train_ids must be an ordered sequence")
    identifiers = list(values)
    _require(bool(identifiers), "development_train_ids must not be empty")
    _require(
        all(
            isinstance(identifier, str)
            and _SAFE_ID.fullmatch(identifier) is not None
            for identifier in identifiers
        ),
        "development_train_ids contains an unsafe ID",
    )
    _require(
        len(identifiers) == len(set(identifiers)),
        "development_train_ids contains duplicates",
    )
    return identifiers


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4AtlasBuildError(
            f"value is not canonical-JSON encodable: {error}"
        ) from error


def _manifest_semantic_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _binary_semantic_sha256(value: np.ndarray) -> str:
    _require(
        isinstance(value, np.ndarray)
        and value.dtype == np.dtype(np.bool_)
        and value.ndim == 2,
        "binary semantic hash requires one two-dimensional bool array",
    )
    ready = np.ascontiguousarray(value, dtype=np.uint8)
    descriptor = _canonical_json_bytes(
        {"dtype": "bool", "shape": list(ready.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"sctransnet-pbdr-v4-binary-map-v1\0")
    digest.update(len(descriptor).to_bytes(8, byteorder="big"))
    digest.update(descriptor)
    digest.update(ready.tobytes(order="C"))
    return digest.hexdigest()


def _validated_sample_array(
    value: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    _require(isinstance(value, np.ndarray), f"{name} must be a numpy array")
    _require(value.ndim == 2, f"{name} must be two-dimensional")
    _require(
        value.shape[0] > 0 and value.shape[1] > 0,
        f"{name} dimensions must be positive",
    )
    _require(
        np.issubdtype(value.dtype, np.floating)
        or value.dtype == np.dtype(np.bool_),
        f"{name} must use a floating or bool dtype",
    )
    ready = np.asarray(value, dtype=np.float64, order="C")
    _require(bool(np.isfinite(ready).all()), f"{name} contains non-finite values")
    _require(
        bool(np.all((ready >= 0.0) & (ready <= 1.0))),
        f"{name} must lie in [0, 1]",
    )
    return ready


def _positive_component_count(value: np.ndarray) -> int:
    return sum(int(component_id) > 0 for component_id in np.unique(value))


def _component_statistics(
    *,
    result: Any,
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    maps: Mapping[str, np.ndarray],
) -> dict[str, int]:
    rescue_pixels = int(np.count_nonzero(maps["rescue_ids"] > 0))
    suppress_pixels = int(np.count_nonzero(maps["suppress_ids"] > 0))
    preserve_pixels = int(np.count_nonzero(maps["preserve_ids"] > 0))
    return {
        "target_component_count": len(result.targets),
        "matched_target_component_count": len(result.matched_target_ids),
        "unmatched_target_component_count": len(result.unmatched_target_ids),
        "prediction_component_count": len(result.predictions),
        "matched_prediction_component_count": len(
            result.matched_prediction_ids
        ),
        "unmatched_prediction_component_count": len(
            result.unmatched_prediction_ids
        ),
        "unmatched_prediction_pixel_count": int(
            result.unmatched_prediction_pixels
        ),
        "target_positive_pixel_count": int(np.count_nonzero(target_mask)),
        "prediction_positive_pixel_count": int(
            np.count_nonzero(prediction_mask)
        ),
        "rescue_component_count": _positive_component_count(
            maps["rescue_ids"]
        ),
        "suppress_component_count": _positive_component_count(
            maps["suppress_ids"]
        ),
        "preserve_component_count": _positive_component_count(
            maps["preserve_ids"]
        ),
        "rescue_pixel_count": rescue_pixels,
        "suppress_pixel_count": suppress_pixels,
        "preserve_pixel_count": preserve_pixels,
    }


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    try:
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise PBDRV4AtlasBuildError(f"cannot write atlas NPZ {path}: {error}") from error


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise PBDRV4AtlasBuildError(f"cannot write {path}: {error}") from error


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    ready = np.asarray(value)
    _require(ready.ndim == 0, f"{name} must be a scalar string")
    _require(ready.dtype.kind in ("U", "S"), f"{name} must use a string dtype")
    item = ready.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("ascii")
        except UnicodeDecodeError as error:
            raise PBDRV4AtlasBuildError(f"{name} is not ASCII") from error
    _require(isinstance(item, str) and bool(item), f"{name} must not be empty")
    return item


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in pairs:
        if key in ready:
            raise PBDRV4AtlasBuildError(
                f"manifest contains duplicate JSON key: {key!r}"
            )
        ready[key] = value
    return ready


def _load_manifest(path: Path) -> dict[str, Any]:
    _require(
        path.is_file() and not path.is_symlink(),
        "manifest must be a regular non-symlink file",
    )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except PBDRV4AtlasBuildError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4AtlasBuildError(f"cannot load manifest: {error}") from error
    _require(isinstance(value, dict), "manifest must contain one JSON object")
    return value


def _validate_component_statistics(
    statistics: Any,
    maps: Mapping[str, np.ndarray],
    *,
    image_id: str,
) -> None:
    _require(
        isinstance(statistics, Mapping)
        and set(statistics) == _COMPONENT_STATISTIC_KEYS,
        f"component statistic fields differ: {image_id}",
    )
    _require(
        all(type(value) is int and value >= 0 for value in statistics.values()),
        f"component statistics must be non-negative integers: {image_id}",
    )
    rescue_count = _positive_component_count(maps["rescue_ids"])
    suppress_count = _positive_component_count(maps["suppress_ids"])
    preserve_count = _positive_component_count(maps["preserve_ids"])
    rescue_pixels = int(np.count_nonzero(maps["rescue_ids"] > 0))
    suppress_pixels = int(np.count_nonzero(maps["suppress_ids"] > 0))
    preserve_pixels = int(np.count_nonzero(maps["preserve_ids"] > 0))
    expected = {
        "unmatched_target_component_count": rescue_count,
        "matched_target_component_count": preserve_count,
        "target_component_count": rescue_count + preserve_count,
        "unmatched_prediction_component_count": suppress_count,
        "unmatched_prediction_pixel_count": suppress_pixels,
        "target_positive_pixel_count": rescue_pixels + preserve_pixels,
        "rescue_component_count": rescue_count,
        "suppress_component_count": suppress_count,
        "preserve_component_count": preserve_count,
        "rescue_pixel_count": rescue_pixels,
        "suppress_pixel_count": suppress_pixels,
        "preserve_pixel_count": preserve_pixels,
    }
    for name, expected_value in expected.items():
        _require(
            statistics[name] == expected_value,
            f"component statistic {name} differs: {image_id}",
        )
    _require(
        statistics["matched_prediction_component_count"]
        == statistics["matched_target_component_count"],
        f"matched target/prediction counts differ: {image_id}",
    )
    _require(
        statistics["prediction_component_count"]
        == statistics["matched_prediction_component_count"]
        + statistics["unmatched_prediction_component_count"],
        f"prediction component partition differs: {image_id}",
    )
    _require(
        statistics["prediction_positive_pixel_count"] >= suppress_pixels,
        f"prediction positive pixels are smaller than suppress pixels: {image_id}",
    )


def validate_component_atlas_artifact(root: Path) -> dict[str, Any]:
    """Reopen and fully validate one staged or committed atlas directory."""

    candidate = Path(root)
    _require(
        candidate.is_dir() and not candidate.is_symlink(),
        "atlas root must be a regular non-symlink directory",
    )
    root_path = candidate.resolve(strict=True)
    entries = list(root_path.iterdir())
    _require(not any(entry.is_symlink() for entry in entries), "atlas contains a symlink")
    manifest = _load_manifest(root_path / MANIFEST_FILENAME)
    required = {
        "schema",
        "builder_schema",
        "dataset",
        "role",
        "split_scope",
        "official_test_accessed",
        "threshold_contract",
        "development_train_ids",
        "development_train_ids_sha256",
        "official_train_ids_sha256",
        "parent_checkpoint_sha256",
        "parent_state_sha256",
        "split_projection_sha256",
        "metric_source_sha256",
        "matcher_source_sha256",
        "source_lock_sha256",
        "samples",
        "manifest_sha256",
    }
    _require(required.issubset(manifest), "manifest fields are incomplete")
    _require(manifest["schema"] == ATLAS_MANIFEST_SCHEMA, "manifest schema differs")
    _require(manifest["builder_schema"] == BUILDER_SCHEMA, "builder schema differs")
    data_protocol.require_dataset(manifest["dataset"])
    _require(manifest["role"] in ROLES, "manifest role differs")
    _require(manifest["split_scope"] == ATLAS_SPLIT_SCOPE, "manifest split scope differs")
    _require(
        manifest["official_test_accessed"] is False,
        "manifest crossed the official-test boundary",
    )
    _require(
        manifest["threshold_contract"]
        == {
            "probability_threshold": FIXED_THRESHOLD,
            "probability_comparison": ">",
            "target_threshold": FIXED_THRESHOLD,
            "target_comparison": ">",
        },
        "threshold contract differs",
    )
    identifiers = _validated_ids(manifest["development_train_ids"])
    _require(
        manifest["development_train_ids_sha256"] == ordered_ids_sha256(identifiers),
        "development-ID SHA differs",
    )
    for name in (
        "official_train_ids_sha256",
        "parent_checkpoint_sha256",
        "parent_state_sha256",
        "split_projection_sha256",
        "metric_source_sha256",
        "matcher_source_sha256",
        "source_lock_sha256",
        "manifest_sha256",
    ):
        _validated_sha256(manifest[name], name=name)
    _require(
        manifest["matcher_source_sha256"] == active_matcher_source_sha256(),
        "manifest matcher source SHA differs from the active canonical matcher",
    )
    _require(
        manifest["manifest_sha256"] == _manifest_semantic_sha256(manifest),
        "manifest semantic SHA differs",
    )
    samples = manifest["samples"]
    _require(isinstance(samples, list), "manifest samples must be a list")
    _require(
        [item.get("image_id") if isinstance(item, Mapping) else None for item in samples]
        == identifiers,
        "manifest samples are missing, extra, or reordered",
    )
    filenames = [item.get("filename") for item in samples if isinstance(item, Mapping)]
    _require(
        all(
            isinstance(name, str)
            and Path(name).name == name
            and name.endswith(".npz")
            for name in filenames
        )
        and len(filenames) == len(set(filenames)),
        "manifest sample filenames are invalid",
    )
    _require(
        {entry.name for entry in entries}
        == {MANIFEST_FILENAME, *filenames},
        "atlas directory has missing or extra files",
    )

    for image_id, record in zip(identifiers, samples):
        _require(isinstance(record, Mapping), f"sample record is invalid: {image_id}")
        record_required = {
            "image_id",
            "filename",
            "file_sha256",
            "parent_state_sha256",
            "matcher_source_sha256",
            "target_binary_semantic_sha256",
            "prediction_binary_semantic_sha256",
            "maps",
            "component_statistics",
        }
        _require(
            record_required.issubset(record),
            f"sample record fields are incomplete: {image_id}",
        )
        for name in (
            "file_sha256",
            "parent_state_sha256",
            "matcher_source_sha256",
            "target_binary_semantic_sha256",
            "prediction_binary_semantic_sha256",
        ):
            _validated_sha256(record[name], name=f"{image_id}.{name}")
        _require(
            record["parent_state_sha256"] == manifest["parent_state_sha256"],
            f"sample parent state SHA differs: {image_id}",
        )
        _require(
            record["matcher_source_sha256"] == manifest["matcher_source_sha256"],
            f"sample matcher source SHA differs: {image_id}",
        )
        path = root_path / record["filename"]
        _require(
            file_sha256(path) == record["file_sha256"],
            f"sample file SHA differs: {image_id}",
        )
        try:
            with np.load(path, allow_pickle=False) as archive:
                _require(
                    len(archive.files) == len(set(archive.files))
                    and set(archive.files) == ATLAS_NPZ_KEYS,
                    f"sample NPZ keys differ: {image_id}",
                )
                observed_identity = {
                    name: _scalar_string(archive[name], name=name)
                    for name in _IDENTITY_KEYS
                }
                maps = {
                    name: np.asarray(archive[name]) for name in ATLAS_MAP_NAMES
                }
        except PBDRV4AtlasBuildError:
            raise
        except (OSError, ValueError, KeyError) as error:
            raise PBDRV4AtlasBuildError(
                f"cannot load sample NPZ {image_id}: {error}"
            ) from error
        _require(observed_identity["image_id"] == image_id, f"sample image_id differs: {image_id}")
        _require(
            observed_identity["parent_state_sha256"] == manifest["parent_state_sha256"],
            f"sample NPZ parent SHA differs: {image_id}",
        )
        _require(
            observed_identity["matcher_source_sha256"] == manifest["matcher_source_sha256"],
            f"sample NPZ matcher SHA differs: {image_id}",
        )
        metadata = record["maps"]
        _require(
            isinstance(metadata, Mapping) and set(metadata) == set(ATLAS_MAP_NAMES),
            f"sample map metadata differs: {image_id}",
        )
        shapes: set[tuple[int, int]] = set()
        for name, value in maps.items():
            details = metadata[name]
            _require(
                isinstance(details, Mapping)
                and set(details) == {"semantic_sha256", "shape", "dtype"},
                f"sample {name} metadata fields differ: {image_id}",
            )
            _require(value.dtype == np.dtype(np.int32), f"sample {name} dtype differs: {image_id}")
            _require(value.ndim == 2, f"sample {name} must be two-dimensional: {image_id}")
            _require(not bool(np.any(value < 0)), f"sample {name} has a negative ID: {image_id}")
            _require(details["dtype"] == "int32", f"sample {name} declared dtype differs: {image_id}")
            _require(details["shape"] == list(value.shape), f"sample {name} shape differs: {image_id}")
            _require(
                details["semantic_sha256"] == array_semantic_sha256(value),
                f"sample {name} semantic SHA differs: {image_id}",
            )
            shapes.add(value.shape)
        _require(len(shapes) == 1, f"sample map shapes differ: {image_id}")
        _require(
            not bool(np.any((maps["rescue_ids"] > 0) & (maps["preserve_ids"] > 0))),
            f"sample rescue/preserve maps overlap: {image_id}",
        )
        _validate_component_statistics(
            record["component_statistics"], maps, image_id=image_id
        )
    return manifest


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Commit a directory with Linux RENAME_NOREPLACE when available."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return
            observed_errno = ctypes.get_errno()
            if observed_errno == errno.EEXIST:
                raise FileExistsError(destination)
            if observed_errno not in (errno.ENOSYS, errno.EINVAL):
                raise OSError(
                    observed_errno,
                    os.strerror(observed_errno),
                    str(destination),
                )

    # Portable fallback: serialize cooperating builders with an O_EXCL claim,
    # recheck the destination, and use a same-directory rename.
    lock = destination.parent / f".{destination.name}.commit.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(source, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink(missing_ok=True)


def build_pbdr_v4_component_atlas(
    *,
    dataset_name: str,
    role: str,
    development_train_ids: Sequence[str],
    frozen_samples: Mapping[str, FrozenCurrentSample],
    parent_checkpoint_sha256: str,
    parent_state_sha256: str,
    split_projection_sha256: str,
    official_train_ids_sha256: str,
    metric_source_sha256: str,
    matcher_source_sha256: str,
    source_lock_sha256: str,
    output_root: Path,
) -> AtlasBuildResult:
    """Build, replay-validate, and no-replace commit one atlas directory."""

    dataset = data_protocol.require_dataset(dataset_name)
    _require(role in ROLES, f"role must be one of {ROLES}")
    identifiers = _validated_ids(development_train_ids)
    _require(isinstance(frozen_samples, Mapping), "frozen_samples must be a mapping")
    _require(
        set(frozen_samples) == set(identifiers),
        "frozen_samples has missing or extra development samples",
    )
    bindings = {
        "parent_checkpoint_sha256": _validated_sha256(
            parent_checkpoint_sha256, name="parent_checkpoint_sha256"
        ),
        "parent_state_sha256": _validated_sha256(
            parent_state_sha256, name="parent_state_sha256"
        ),
        "split_projection_sha256": _validated_sha256(
            split_projection_sha256, name="split_projection_sha256"
        ),
        "official_train_ids_sha256": _validated_sha256(
            official_train_ids_sha256, name="official_train_ids_sha256"
        ),
        "metric_source_sha256": _validated_sha256(
            metric_source_sha256, name="metric_source_sha256"
        ),
        "matcher_source_sha256": _validated_sha256(
            matcher_source_sha256, name="matcher_source_sha256"
        ),
        "source_lock_sha256": _validated_sha256(
            source_lock_sha256, name="source_lock_sha256"
        ),
    }
    _require(
        bindings["matcher_source_sha256"] == active_matcher_source_sha256(),
        "requested matcher source SHA differs from the active canonical matcher",
    )

    requested = Path(output_root)
    _require(
        requested.name not in ("", ".", ".."),
        "output_root must name one child directory",
    )
    parent = requested.parent
    _require(not parent.is_symlink(), "output parent must not be a symlink")
    parent = parent.resolve(strict=True)
    _require(parent.is_dir(), "output parent must be a directory")
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.stage.",
            dir=parent,
        )
    )
    committed = False
    try:
        records: list[dict[str, Any]] = []
        for image_id in identifiers:
            sample = frozen_samples[image_id]
            _require(
                isinstance(sample, FrozenCurrentSample),
                f"frozen sample has invalid type: {image_id}",
            )
            probability = _validated_sample_array(
                sample.probability, name=f"{image_id}.probability"
            )
            target = _validated_sample_array(
                sample.target, name=f"{image_id}.target"
            )
            _require(
                probability.shape == target.shape,
                f"probability/target shapes differ: {image_id}",
            )
            prediction_mask = np.ascontiguousarray(
                probability > FIXED_THRESHOLD
            )
            target_mask = np.ascontiguousarray(target > FIXED_THRESHOLD)
            match = match_components_v2(
                prediction_mask=prediction_mask,
                target_mask=target_mask,
            )
            atlas = atlas_maps_from_match(match)
            maps = {
                "rescue_ids": atlas.rescue_ids,
                "suppress_ids": atlas.suppress_ids,
                "preserve_ids": atlas.preserve_ids,
            }
            filename = f"{image_id}.npz"
            path = staging / filename
            archive = {
                **maps,
                "image_id": np.asarray(image_id),
                "parent_state_sha256": np.asarray(
                    bindings["parent_state_sha256"]
                ),
                "matcher_source_sha256": np.asarray(
                    bindings["matcher_source_sha256"]
                ),
            }
            _require(
                set(archive) == ATLAS_NPZ_KEYS
                and all(value.dtype.kind != "O" for value in archive.values()),
                f"NPZ payload would require pickle or has wrong keys: {image_id}",
            )
            _write_npz_exclusive(path, archive)
            records.append(
                {
                    "image_id": image_id,
                    "filename": filename,
                    "file_sha256": file_sha256(path),
                    "parent_state_sha256": bindings["parent_state_sha256"],
                    "matcher_source_sha256": bindings[
                        "matcher_source_sha256"
                    ],
                    "target_binary_semantic_sha256": _binary_semantic_sha256(
                        target_mask
                    ),
                    "prediction_binary_semantic_sha256": _binary_semantic_sha256(
                        prediction_mask
                    ),
                    "maps": {
                        name: {
                            "semantic_sha256": array_semantic_sha256(value),
                            "shape": list(value.shape),
                            "dtype": "int32",
                        }
                        for name, value in maps.items()
                    },
                    "component_statistics": _component_statistics(
                        result=match,
                        prediction_mask=prediction_mask,
                        target_mask=target_mask,
                        maps=maps,
                    ),
                }
            )

        manifest: dict[str, Any] = {
            "schema": ATLAS_MANIFEST_SCHEMA,
            "builder_schema": BUILDER_SCHEMA,
            "dataset": dataset,
            "role": role,
            "split_scope": ATLAS_SPLIT_SCOPE,
            "official_test_accessed": False,
            "threshold_contract": {
                "probability_threshold": FIXED_THRESHOLD,
                "probability_comparison": ">",
                "target_threshold": FIXED_THRESHOLD,
                "target_comparison": ">",
            },
            "development_train_ids": identifiers,
            "development_train_ids_sha256": ordered_ids_sha256(identifiers),
            "official_train_ids_sha256": bindings[
                "official_train_ids_sha256"
            ],
            "parent_checkpoint_sha256": bindings[
                "parent_checkpoint_sha256"
            ],
            "parent_state_sha256": bindings["parent_state_sha256"],
            "split_projection_sha256": bindings["split_projection_sha256"],
            "metric_source_sha256": bindings["metric_source_sha256"],
            "matcher_source_sha256": bindings["matcher_source_sha256"],
            "source_lock_sha256": bindings["source_lock_sha256"],
            "samples": records,
        }
        manifest["manifest_sha256"] = _manifest_semantic_sha256(manifest)
        _write_bytes_exclusive(
            staging / MANIFEST_FILENAME,
            _canonical_json_bytes(manifest),
        )
        replay = validate_component_atlas_artifact(staging)
        _require(
            _canonical_json_bytes(replay) == _canonical_json_bytes(manifest),
            "staged atlas replay differs from the generated manifest",
        )
        _rename_directory_noreplace(staging, destination)
        committed = True
        # Validate the exact committed directory once more.  No source arrays
        # or split information are consulted during either replay.
        committed_manifest = validate_component_atlas_artifact(destination)
        _require(
            committed_manifest["manifest_sha256"]
            == manifest["manifest_sha256"],
            "committed atlas manifest SHA differs",
        )
        return AtlasBuildResult(
            root=destination,
            manifest_path=destination / MANIFEST_FILENAME,
            manifest_sha256=manifest["manifest_sha256"],
            sample_count=len(identifiers),
        )
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "ATLAS_MANIFEST_SCHEMA",
    "AtlasBuildResult",
    "BUILDER_SCHEMA",
    "FIXED_THRESHOLD",
    "FrozenCurrentSample",
    "MANIFEST_FILENAME",
    "PBDRV4AtlasBuildError",
    "ROLES",
    "build_pbdr_v4_component_atlas",
    "validate_component_atlas_artifact",
]
