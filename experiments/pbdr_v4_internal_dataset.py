#!/usr/bin/env python3
"""No-augmentation internal inference dataset for PBDR-V4.

The caller supplies one ordered projection and the complete, already verified
official-train ID sequence.  This module neither reads nor reconstructs an
index.  It is suitable for internal validation, frozen Current/V3 cache
generation, and frozen-parent atlas inference.

Every resolver call is forced to ``split="train"``.  Images use the frozen
legacy normalization, targets are divided by 255, and both arrays receive only
bottom/right zero padding to a multiple of 32.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from experiments import three_dataset_v2_protocol as data_protocol


PAD_MULTIPLE = 32
PROJECTION_SCOPES = (
    "development_train_ids",
    "internal_validation_ids",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PBDRV4InternalDatasetError(ValueError):
    """An internal projection, resolver, file, or array contract is invalid."""


SampleResolver = Callable[..., Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4InternalDatasetError(message)


def _validated_sha256(value: Any, *, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 hex digest",
    )
    return value


def _validated_ids(values: Sequence[str], *, name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an ordered sequence")
    ready = list(values)
    _require(bool(ready), f"{name} must not be empty")
    _require(
        all(
            isinstance(identifier, str)
            and _SAFE_ID.fullmatch(identifier) is not None
            for identifier in ready
        ),
        f"{name} contains an unsafe ID",
    )
    _require(len(ready) == len(set(ready)), f"{name} contains duplicate IDs")
    return ready


def _regular_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise PBDRV4InternalDatasetError(f"resolved {name} is not path-like")
    candidate = Path(value)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"resolved {name} must be a regular non-symlink file",
    )
    return candidate.resolve(strict=True)


def _pad_bottom_right_32(value: np.ndarray) -> np.ndarray:
    _require(value.ndim == 2, "only two-dimensional arrays can be padded")
    height, width = value.shape
    padded_height = ((height + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    padded_width = ((width + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    ready = np.pad(
        value,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="constant",
        constant_values=0,
    )
    return np.ascontiguousarray(ready)


class PBDRV4InternalInferenceDataset(Dataset):
    """Ordered official-train projection with no stochastic augmentation."""

    def __init__(
        self,
        selected_ids: Sequence[str],
        known_official_train_ids: Sequence[str],
        *,
        manifest_scope: str,
        selected_ids_ordered_sha256: str,
        official_train_count: int,
        official_train_ordered_ids_sha256: str,
        dataset_name: str,
        data_root: Path,
        sample_resolver: SampleResolver = data_protocol.resolve_sample,
    ) -> None:
        super().__init__()
        _require(
            manifest_scope in PROJECTION_SCOPES,
            f"manifest_scope must be one of {PROJECTION_SCOPES}",
        )
        self.manifest_scope = manifest_scope
        self.dataset_name = data_protocol.require_dataset(dataset_name)
        self.sample_ids = _validated_ids(selected_ids, name="selected_ids")
        official_ids = _validated_ids(
            known_official_train_ids,
            name="known_official_train_ids",
        )
        _require(
            type(official_train_count) is int and official_train_count > 0,
            "official_train_count must be a positive integer",
        )
        _require(
            len(official_ids) == official_train_count,
            "known official-train ID count differs",
        )
        expected_official_sha = _validated_sha256(
            official_train_ordered_ids_sha256,
            name="official_train_ordered_ids_sha256",
        )
        _require(
            data_protocol.ordered_ids_sha256(official_ids)
            == expected_official_sha,
            "known official-train ordered-ID SHA differs",
        )
        expected_selected_sha = _validated_sha256(
            selected_ids_ordered_sha256,
            name="selected_ids_ordered_sha256",
        )
        _require(
            data_protocol.ordered_ids_sha256(self.sample_ids)
            == expected_selected_sha,
            "selected ordered-ID SHA differs",
        )
        _require(
            set(self.sample_ids).issubset(set(official_ids)),
            "selected IDs are not a subset of official-train IDs",
        )
        self.official_train_count = official_train_count
        self.official_train_ordered_ids_sha256 = expected_official_sha
        self.selected_ids_ordered_sha256 = expected_selected_sha
        self._known_official_train_ids = tuple(official_ids)
        self._known_ids = frozenset(official_ids)

        root = Path(data_root)
        _require(not root.is_symlink(), "data_root must not be a symlink")
        _require(root.is_dir(), "data_root must be a directory")
        self.data_root = root.resolve(strict=True)
        _require(callable(sample_resolver), "sample_resolver must be callable")
        self._resolver = sample_resolver

        normalization = data_protocol.get_legacy_normalization(self.dataset_name)
        mean = float(normalization["mean"])
        std = float(normalization["std"])
        _require(
            math.isfinite(mean) and math.isfinite(std) and std > 0.0,
            "legacy normalization is invalid",
        )
        self.normalization = {"mean": mean, "std": std}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _resolve_training_sample(self, identifier: str) -> Any:
        sample = self._resolver(
            self.data_root,
            self.dataset_name,
            identifier,
            split="train",
            known_ids=self._known_ids,
        )
        _require(
            getattr(sample, "dataset_name", None) == self.dataset_name,
            f"resolver dataset differs: {identifier}",
        )
        _require(
            getattr(sample, "split", None) == "train",
            f"resolver returned a non-training sample: {identifier}",
        )
        _require(
            getattr(sample, "sample_id", None) == identifier,
            f"resolver sample ID differs: {identifier}",
        )
        _require(
            getattr(sample, "correction_id", None) is None,
            f"training sample unexpectedly uses a correction overlay: {identifier}",
        )
        return sample

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], str]:
        identifier = self.sample_ids[index]
        sample = self._resolve_training_sample(identifier)
        image_path = _regular_path(
            getattr(sample, "image_path", None), name="image_path"
        )
        mask_path = _regular_path(
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
        height, width = image.shape
        image = (
            image - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        target = target / np.float32(255.0)
        _require(
            bool(np.isfinite(image).all())
            and bool(np.isfinite(target).all()),
            f"normalized image/target contains non-finite pixels: {identifier}",
        )
        image = _pad_bottom_right_32(image)
        target = _pad_bottom_right_32(target)
        _require(image.shape == target.shape, "padded image/target shapes differ")
        return (
            torch.from_numpy(
                np.ascontiguousarray(image[None], dtype=np.float32)
            ),
            torch.from_numpy(
                np.ascontiguousarray(target[None], dtype=np.float32)
            ),
            (height, width),
            identifier,
        )


__all__ = [
    "PAD_MULTIPLE",
    "PROJECTION_SCOPES",
    "PBDRV4InternalDatasetError",
    "PBDRV4InternalInferenceDataset",
]
