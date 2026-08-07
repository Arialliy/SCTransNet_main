#!/usr/bin/env python3
"""PBDR-V4 development-train dataset with component-atlas identities.

This module is intentionally independent of the repository-root ``dataset.py``
and never loads an index.  Its caller must supply both the development-train
IDs and the already verified official-train IDs.  Every sample resolution is
forced through ``split="train"``.

The geometric path exactly follows the PBDR-V3 stateless training path:

``bottom/right pad -> crop -> axis-0 flip -> axis-1 flip -> transpose``.

The image, target, and all three categorical component-ID maps share one plan.
There is no resize operation in this protocol; ID maps remain ``int32`` and
are never passed through an interpolation kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from experiments import component_matching_v2
from experiments import three_dataset_v2_protocol as data_protocol


ATLAS_MANIFEST_SCHEMA = "sctransnet_pbdr_v4_component_atlas/v1"
ATLAS_SPLIT_SCOPE = "development_train_ids_only"
ATLAS_MAP_NAMES = ("rescue_ids", "suppress_ids", "preserve_ids")
ATLAS_IDENTITY_NAMES = (
    "image_id",
    "parent_state_sha256",
    "matcher_source_sha256",
)
ATLAS_NPZ_KEYS = frozenset((*ATLAS_MAP_NAMES, *ATLAS_IDENTITY_NAMES))
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PBDRV4AtlasDatasetError(ValueError):
    """An atlas identity, artifact, sample, or transform contract is invalid."""


@dataclass(frozen=True, slots=True)
class TransformedAtlasArrays:
    image: np.ndarray = field(repr=False, compare=False)
    target: np.ndarray = field(repr=False, compare=False)
    rescue_ids: np.ndarray = field(repr=False, compare=False)
    suppress_ids: np.ndarray = field(repr=False, compare=False)
    preserve_ids: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AtlasSampleRecord:
    image_id: str
    path: Path
    file_sha256: str
    parent_state_sha256: str
    matcher_source_sha256: str
    maps: Mapping[str, Mapping[str, Any]] = field(repr=False, compare=False)


SampleResolver = Callable[..., Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4AtlasDatasetError(message)


def _validate_sha256(value: Any, *, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 hex digest",
    )
    return value


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4AtlasDatasetError(
            f"expected a regular non-symlink file: {candidate}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(identifiers: Sequence[str]) -> str:
    content = json.dumps(
        list(identifiers),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def array_semantic_sha256(value: np.ndarray) -> str:
    """Hash an int32 ID map by shape, semantic dtype, and C-order values."""

    if not isinstance(value, np.ndarray):
        raise TypeError("atlas map must be a numpy.ndarray")
    if value.dtype != np.dtype(np.int32):
        raise TypeError("atlas map must have int32 dtype")
    if value.ndim != 2:
        raise ValueError("atlas map must be two-dimensional")
    ready = np.ascontiguousarray(value.astype(np.dtype("<i4"), copy=False))
    descriptor = json.dumps(
        {"dtype": "int32", "shape": list(ready.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"sctransnet-pbdr-v4-atlas-map-v1\0")
    digest.update(len(descriptor).to_bytes(8, byteorder="big"))
    digest.update(descriptor)
    digest.update(ready.tobytes(order="C"))
    return digest.hexdigest()


def matcher_source_sha256() -> str:
    source = Path(component_matching_v2.__file__).resolve(strict=True)
    return file_sha256(source)


def _validated_ids(values: Sequence[str], *, name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of sample IDs")
    ready = list(values)
    _require(bool(ready), f"{name} must not be empty")
    _require(
        all(
            isinstance(identifier, str)
            and _SAFE_ID.fullmatch(identifier) is not None
            for identifier in ready
        ),
        f"{name} contains an unsafe sample ID",
    )
    _require(len(ready) == len(set(ready)), f"{name} contains duplicate IDs")
    return ready


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in pairs:
        if key in ready:
            raise PBDRV4AtlasDatasetError(
                f"atlas manifest contains duplicate key: {key!r}"
            )
        ready[key] = value
    return ready


def _load_manifest(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4AtlasDatasetError(
            f"atlas manifest must be a regular non-symlink file: {candidate}"
        )
    try:
        payload = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except PBDRV4AtlasDatasetError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4AtlasDatasetError(
            f"cannot read atlas manifest {candidate}: {error}"
        ) from error
    _require(isinstance(payload, dict), "atlas manifest must be a JSON object")
    return payload


def _validated_normalization(
    value: Mapping[str, Any],
) -> dict[str, float]:
    _require(isinstance(value, Mapping), "normalization must be a mapping")
    _require(set(value) == {"mean", "std"}, "normalization fields differ")
    mean, std = value["mean"], value["std"]
    _require(
        not isinstance(mean, bool)
        and isinstance(mean, (int, float))
        and math.isfinite(float(mean)),
        "normalization mean must be finite",
    )
    _require(
        not isinstance(std, bool)
        and isinstance(std, (int, float))
        and math.isfinite(float(std))
        and float(std) > 0.0,
        "normalization std must be finite and positive",
    )
    return {"mean": float(mean), "std": float(std)}


def _validate_plan_for_shape(
    plan: data_protocol.StatelessTransformPlan,
    shape: tuple[int, int],
) -> None:
    _require(
        isinstance(plan, data_protocol.StatelessTransformPlan),
        "plan must be a StatelessTransformPlan",
    )
    height, width = shape
    _require(
        plan.padded_height >= height and plan.padded_width >= width,
        "transform plan padded dimensions are too small",
    )
    _require(plan.crop_size > 0, "transform plan crop size must be positive")
    _require(
        0 <= plan.crop_top
        <= plan.padded_height - plan.crop_size,
        "transform plan crop_top is out of bounds",
    )
    _require(
        0 <= plan.crop_left
        <= plan.padded_width - plan.crop_size,
        "transform plan crop_left is out of bounds",
    )


def apply_stateless_geometry(
    plan: data_protocol.StatelessTransformPlan,
    *,
    image: np.ndarray,
    target: np.ndarray,
    rescue_ids: np.ndarray,
    suppress_ids: np.ndarray,
    preserve_ids: np.ndarray,
) -> TransformedAtlasArrays:
    """Apply one V3-compatible plan to all five spatial arrays."""

    arrays = {
        "image": image,
        "target": target,
        "rescue_ids": rescue_ids,
        "suppress_ids": suppress_ids,
        "preserve_ids": preserve_ids,
    }
    for name, value in arrays.items():
        _require(isinstance(value, np.ndarray), f"{name} must be a numpy array")
        _require(value.ndim == 2, f"{name} must be two-dimensional")
    reference_shape = image.shape
    _require(
        all(value.shape == reference_shape for value in arrays.values()),
        "image, target, and atlas maps must share shape",
    )
    for name in ATLAS_MAP_NAMES:
        value = arrays[name]
        _require(value.dtype == np.dtype(np.int32), f"{name} must be int32")
        _require(not bool(np.any(value < 0)), f"{name} contains a negative ID")

    _validate_plan_for_shape(plan, reference_shape)
    pad_width = (
        (0, plan.padded_height - reference_shape[0]),
        (0, plan.padded_width - reference_shape[1]),
    )
    transformed = {
        name: np.pad(value, pad_width, mode="constant", constant_values=0)
        for name, value in arrays.items()
    }
    top, left, size = plan.crop_top, plan.crop_left, plan.crop_size
    transformed = {
        name: value[top : top + size, left : left + size]
        for name, value in transformed.items()
    }
    if plan.flip_axis0:
        transformed = {name: value[::-1] for name, value in transformed.items()}
    if plan.flip_axis1:
        transformed = {
            name: value[:, ::-1] for name, value in transformed.items()
        }
    if plan.transpose:
        transformed = {name: value.T for name, value in transformed.items()}
    transformed = {
        name: np.ascontiguousarray(value)
        for name, value in transformed.items()
    }
    for name in ATLAS_MAP_NAMES:
        _require(
            transformed[name].dtype == np.dtype(np.int32),
            f"{name} dtype changed during geometry",
        )

    return TransformedAtlasArrays(**transformed)


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    ready = np.asarray(value)
    _require(ready.ndim == 0, f"atlas {name} must be a scalar string")
    _require(
        ready.dtype.kind in ("U", "S"),
        f"atlas {name} must use a string dtype",
    )
    item = ready.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("ascii")
        except UnicodeDecodeError as error:
            raise PBDRV4AtlasDatasetError(
                f"atlas {name} is not ASCII"
            ) from error
    _require(isinstance(item, str) and bool(item), f"atlas {name} is empty")
    return item


class PBDRV4AtlasTrainDataset(Dataset):
    """Development-train-only loader for image, target, and component IDs."""

    def __init__(
        self,
        development_train_ids: Sequence[str],
        known_official_train_ids: Sequence[str],
        *,
        dataset_name: str,
        data_root: Path,
        atlas_root: Path,
        atlas_manifest: Path,
        parent_state_sha256: str,
        seed: int = data_protocol.PROTOCOL_SEED,
        normalization: Mapping[str, Any] | None = None,
        sample_resolver: SampleResolver = data_protocol.resolve_sample,
    ) -> None:
        super().__init__()
        self.dataset_name = data_protocol.require_dataset(dataset_name)
        self.sample_ids = _validated_ids(
            development_train_ids, name="development_train_ids"
        )
        official_ids = _validated_ids(
            known_official_train_ids, name="known_official_train_ids"
        )
        _require(
            set(self.sample_ids).issubset(set(official_ids)),
            "development_train_ids are not a subset of official-train IDs",
        )
        self._known_official_train_ids = tuple(official_ids)
        self._known_ids = frozenset(official_ids)
        self.parent_state_sha256 = _validate_sha256(
            parent_state_sha256, name="parent_state_sha256"
        )
        self.matcher_source_sha256 = matcher_source_sha256()
        self.seed = data_protocol.require_seed(seed)
        self.epoch = 0
        self._resolver = sample_resolver
        _require(callable(self._resolver), "sample_resolver must be callable")

        root = Path(data_root)
        _require(not root.is_symlink(), "data_root must not be a symlink")
        self.data_root = root.resolve(strict=True)
        atlas_directory = Path(atlas_root)
        _require(not atlas_directory.is_symlink(), "atlas_root must not be a symlink")
        _require(atlas_directory.is_dir(), "atlas_root must be a directory")
        self.atlas_root = atlas_directory.resolve(strict=True)
        self.atlas_manifest_path = Path(atlas_manifest).resolve(strict=False)

        self.normalization = _validated_normalization(
            normalization
            if normalization is not None
            else data_protocol.get_legacy_normalization(self.dataset_name)
        )
        manifest = _load_manifest(Path(atlas_manifest))
        self._records = self._validate_manifest(manifest)
        self._validate_root_entries()
        # Fail closed before training begins: every declared NPZ is opened and
        # its identity and semantic map bindings are checked once here.
        for identifier in self.sample_ids:
            self._read_atlas_file(self._records[identifier])

    def _validate_manifest(
        self, manifest: Mapping[str, Any]
    ) -> dict[str, AtlasSampleRecord]:
        required = {
            "schema",
            "dataset",
            "split_scope",
            "official_test_accessed",
            "development_train_ids",
            "development_train_ids_sha256",
            "official_train_ids_sha256",
            "parent_state_sha256",
            "matcher_source_sha256",
            "samples",
        }
        _require(
            required.issubset(manifest),
            "atlas manifest fields are incomplete",
        )
        _require(
            manifest["schema"] == ATLAS_MANIFEST_SCHEMA,
            "atlas manifest schema differs",
        )
        _require(
            manifest["dataset"] == self.dataset_name,
            "atlas manifest dataset differs",
        )
        _require(
            manifest["split_scope"] == ATLAS_SPLIT_SCOPE,
            "atlas manifest split scope differs",
        )
        _require(
            manifest["official_test_accessed"] is False,
            "atlas manifest crossed the official-test boundary",
        )
        _require(
            manifest["development_train_ids"] == self.sample_ids,
            "atlas manifest development IDs differ",
        )
        _require(
            manifest["development_train_ids_sha256"]
            == ordered_ids_sha256(self.sample_ids),
            "atlas manifest development-ID SHA differs",
        )
        _require(
            manifest["official_train_ids_sha256"]
            == ordered_ids_sha256(self._known_official_train_ids),
            "atlas manifest official-train-ID SHA differs",
        )
        _require(
            manifest["parent_state_sha256"] == self.parent_state_sha256,
            "atlas manifest parent state SHA differs",
        )
        _require(
            manifest["matcher_source_sha256"]
            == self.matcher_source_sha256,
            "atlas manifest matcher source SHA differs",
        )

        samples = manifest["samples"]
        _require(isinstance(samples, list), "atlas manifest samples must be a list")
        _require(
            [
                item.get("image_id") if isinstance(item, Mapping) else None
                for item in samples
            ]
            == self.sample_ids,
            "atlas manifest has missing, extra, or reordered samples",
        )
        records: dict[str, AtlasSampleRecord] = {}
        filenames: set[str] = set()
        for expected_id, raw in zip(self.sample_ids, samples):
            _require(isinstance(raw, Mapping), "atlas sample record must be an object")
            sample_required = {
                "image_id",
                "filename",
                "file_sha256",
                "parent_state_sha256",
                "matcher_source_sha256",
                "maps",
            }
            _require(
                sample_required.issubset(raw),
                f"atlas sample record is incomplete: {expected_id}",
            )
            filename = raw["filename"]
            _require(
                isinstance(filename, str)
                and Path(filename).name == filename
                and filename.endswith(".npz"),
                f"atlas filename is unsafe: {expected_id}",
            )
            _require(filename not in filenames, "atlas filenames are not unique")
            filenames.add(filename)
            record_parent = _validate_sha256(
                raw["parent_state_sha256"],
                name=f"{expected_id}.parent_state_sha256",
            )
            record_matcher = _validate_sha256(
                raw["matcher_source_sha256"],
                name=f"{expected_id}.matcher_source_sha256",
            )
            _require(
                record_parent == self.parent_state_sha256,
                f"atlas sample parent state SHA differs: {expected_id}",
            )
            _require(
                record_matcher == self.matcher_source_sha256,
                f"atlas sample matcher source SHA differs: {expected_id}",
            )
            maps = raw["maps"]
            _require(
                isinstance(maps, Mapping) and set(maps) == set(ATLAS_MAP_NAMES),
                f"atlas map metadata differs: {expected_id}",
            )
            for map_name in ATLAS_MAP_NAMES:
                metadata = maps[map_name]
                _require(
                    isinstance(metadata, Mapping)
                    and set(metadata) == {"semantic_sha256", "shape", "dtype"},
                    f"atlas {map_name} metadata fields differ: {expected_id}",
                )
                _validate_sha256(
                    metadata["semantic_sha256"],
                    name=f"{expected_id}.{map_name}.semantic_sha256",
                )
                shape = metadata["shape"]
                _require(
                    isinstance(shape, list)
                    and len(shape) == 2
                    and all(type(value) is int and value > 0 for value in shape),
                    f"atlas {map_name} shape is invalid: {expected_id}",
                )
                _require(
                    metadata["dtype"] == "int32",
                    f"atlas {map_name} declared dtype differs: {expected_id}",
                )
            records[expected_id] = AtlasSampleRecord(
                image_id=expected_id,
                path=self.atlas_root / filename,
                file_sha256=_validate_sha256(
                    raw["file_sha256"],
                    name=f"{expected_id}.file_sha256",
                ),
                parent_state_sha256=record_parent,
                matcher_source_sha256=record_matcher,
                maps=dict(maps),
            )
        return records

    def _validate_root_entries(self) -> None:
        entries = list(self.atlas_root.iterdir())
        _require(
            not any(entry.is_symlink() for entry in entries),
            "atlas_root contains a symlink",
        )
        expected = {record.path.name for record in self._records.values()}
        observed = {entry.name for entry in entries if entry.suffix == ".npz"}
        _require(
            observed == expected,
            "atlas_root has missing or extra sample NPZ files",
        )
        for record in self._records.values():
            _require(
                record.path.is_file() and not record.path.is_symlink(),
                f"atlas sample is missing or not regular: {record.image_id}",
            )

    def _read_atlas_file(
        self, record: AtlasSampleRecord
    ) -> dict[str, np.ndarray]:
        _require(
            file_sha256(record.path) == record.file_sha256,
            f"atlas file SHA differs: {record.image_id}",
        )
        try:
            with np.load(record.path, allow_pickle=False) as archive:
                _require(
                    len(archive.files) == len(set(archive.files))
                    and set(archive.files) == ATLAS_NPZ_KEYS,
                    f"atlas NPZ keys differ: {record.image_id}",
                )
                observed_id = _scalar_string(
                    archive["image_id"], name="image_id"
                )
                observed_parent = _scalar_string(
                    archive["parent_state_sha256"],
                    name="parent_state_sha256",
                )
                observed_matcher = _scalar_string(
                    archive["matcher_source_sha256"],
                    name="matcher_source_sha256",
                )
                arrays = {
                    name: np.asarray(archive[name]) for name in ATLAS_MAP_NAMES
                }
        except PBDRV4AtlasDatasetError:
            raise
        except (OSError, ValueError, KeyError) as error:
            raise PBDRV4AtlasDatasetError(
                f"cannot load atlas sample {record.image_id}: {error}"
            ) from error

        _require(observed_id == record.image_id, f"atlas image_id differs: {record.image_id}")
        _require(
            observed_parent == record.parent_state_sha256,
            f"atlas parent_state_sha256 differs: {record.image_id}",
        )
        _require(
            observed_matcher == record.matcher_source_sha256,
            f"atlas matcher_source_sha256 differs: {record.image_id}",
        )
        reference_shape: tuple[int, int] | None = None
        ready: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            metadata = record.maps[name]
            _require(
                value.dtype == np.dtype(np.int32),
                f"atlas {name} dtype differs: {record.image_id}",
            )
            _require(
                value.ndim == 2 and list(value.shape) == metadata["shape"],
                f"atlas {name} shape differs: {record.image_id}",
            )
            _require(
                not bool(np.any(value < 0)),
                f"atlas {name} contains a negative ID: {record.image_id}",
            )
            _require(
                array_semantic_sha256(value) == metadata["semantic_sha256"],
                f"atlas {name} semantic SHA differs: {record.image_id}",
            )
            if reference_shape is None:
                reference_shape = value.shape
            _require(
                value.shape == reference_shape,
                f"atlas map shapes differ: {record.image_id}",
            )
            ready[name] = np.ascontiguousarray(value)
        _require(
            not bool(
                np.any(
                    (ready["rescue_ids"] > 0)
                    & (ready["preserve_ids"] > 0)
                )
            ),
            f"atlas rescue/preserve maps overlap: {record.image_id}",
        )
        return ready

    def __len__(self) -> int:
        return len(self.sample_ids)

    def set_epoch(self, epoch: int) -> None:
        _require(
            type(epoch) is int and epoch >= 0,
            "epoch must be a non-negative integer",
        )
        self.epoch = epoch

    def _resolve_training_sample(self, identifier: str) -> Any:
        sample = self._resolver(
            self.data_root,
            self.dataset_name,
            identifier,
            split="train",
            known_ids=self._known_ids,
        )
        _require(
            getattr(sample, "dataset_name", None) == self.dataset_name
            and getattr(sample, "split", None) == "train"
            and getattr(sample, "sample_id", None) == identifier,
            f"resolver returned a non-training or mismatched sample: {identifier}",
        )
        return sample

    @staticmethod
    def _regular_sample_path(value: Any, *, name: str) -> Path:
        path = Path(value)
        _require(
            path.is_file() and not path.is_symlink(),
            f"resolved {name} must be a regular non-symlink file",
        )
        return path.resolve(strict=True)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
    ]:
        identifier = self.sample_ids[index]
        sample = self._resolve_training_sample(identifier)
        image_path = self._regular_sample_path(
            getattr(sample, "image_path", None), name="image_path"
        )
        mask_path = self._regular_sample_path(
            getattr(sample, "mask_path", None), name="mask_path"
        )
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("I"), dtype=np.float32)
        with Image.open(mask_path) as handle:
            target = np.asarray(handle, dtype=np.float32)
        if target.ndim > 2:
            target = target[:, :, 0]
        _require(
            image.ndim == target.ndim == 2 and image.shape == target.shape,
            f"resolved image/mask dimensions differ: {identifier}",
        )
        _require(
            bool(np.isfinite(image).all())
            and bool(np.isfinite(target).all()),
            f"resolved image/mask contains non-finite pixels: {identifier}",
        )
        image = (
            image - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        target = target / np.float32(255.0)
        atlas = self._read_atlas_file(self._records[identifier])
        _require(
            all(value.shape == image.shape for value in atlas.values()),
            f"atlas and image shapes differ: {identifier}",
        )
        target_binary = target > np.float32(0.5)
        _require(
            np.array_equal(
                (atlas["rescue_ids"] > 0)
                | (atlas["preserve_ids"] > 0),
                target_binary,
            ),
            f"atlas rescue/preserve maps do not partition target: {identifier}",
        )

        positive = target > 0
        plan = data_protocol.derive_stateless_transform_plan(
            protocol_seed=self.seed,
            dataset_name=self.dataset_name,
            epoch=self.epoch,
            namespaced_id=f"{self.dataset_name}::{identifier}",
            image_height=image.shape[0],
            image_width=image.shape[1],
            has_positive_in_crop=lambda top, left, size: bool(
                positive[top : top + size, left : left + size].any()
            ),
        )
        transformed = apply_stateless_geometry(
            plan,
            image=image,
            target=target,
            rescue_ids=atlas["rescue_ids"],
            suppress_ids=atlas["suppress_ids"],
            preserve_ids=atlas["preserve_ids"],
        )
        return (
            torch.from_numpy(transformed.image[None].astype(np.float32, copy=False)),
            torch.from_numpy(transformed.target[None].astype(np.float32, copy=False)),
            torch.from_numpy(transformed.rescue_ids[None]),
            torch.from_numpy(transformed.suppress_ids[None]),
            torch.from_numpy(transformed.preserve_ids[None]),
            identifier,
        )


__all__ = [
    "ATLAS_IDENTITY_NAMES",
    "ATLAS_MANIFEST_SCHEMA",
    "ATLAS_MAP_NAMES",
    "ATLAS_NPZ_KEYS",
    "ATLAS_SPLIT_SCOPE",
    "AtlasSampleRecord",
    "PBDRV4AtlasDatasetError",
    "PBDRV4AtlasTrainDataset",
    "TransformedAtlasArrays",
    "apply_stateless_geometry",
    "array_semantic_sha256",
    "file_sha256",
    "matcher_source_sha256",
    "ordered_ids_sha256",
]
