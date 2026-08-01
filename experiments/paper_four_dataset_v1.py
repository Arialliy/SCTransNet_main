"""Correction-aware PyTorch datasets for the four-regime paper protocol.

The train dataset uses no global random state.  Every crop and augmentation is
derived from ``(42, dataset_name, epoch, namespaced_sample_id)``.  The test
dataset performs no augmentation, pads only on the right/bottom to a multiple
of 32, and raises on any image/mask size mismatch.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from experiments.four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        DEFAULT_IMGIDX_MANIFEST,
        DEFAULT_NORMALIZATION_MANIFEST,
        PAD_MULTIPLE,
        PATCH_SIZE,
        PROTOCOL_SEED,
        FourDatasetProtocolError,
        ResolvedSample,
        derive_stateless_transform_plan,
        get_legacy_normalization,
        load_correction_manifest,
        load_frozen_index,
        resolve_sample,
        source_dataset_for_sample,
        validate_pair,
    )
except ModuleNotFoundError:
    from four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        DEFAULT_IMGIDX_MANIFEST,
        DEFAULT_NORMALIZATION_MANIFEST,
        PAD_MULTIPLE,
        PATCH_SIZE,
        PROTOCOL_SEED,
        FourDatasetProtocolError,
        ResolvedSample,
        derive_stateless_transform_plan,
        get_legacy_normalization,
        load_correction_manifest,
        load_frozen_index,
        resolve_sample,
        source_dataset_for_sample,
        validate_pair,
    )


def _load_image_and_mask(
    sample: ResolvedSample,
) -> tuple[np.ndarray, np.ndarray]:
    """Load legacy-compatible luma and mask arrays, with a hard size check."""

    with Image.open(sample.image_path) as image:
        image_array = np.asarray(image.convert("I"), dtype=np.float32)
    with Image.open(sample.mask_path) as mask:
        mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim > 2:
        mask_array = mask_array[:, :, 0]
    if image_array.ndim != 2 or mask_array.ndim != 2:
        raise FourDatasetProtocolError(
            "image and mask must both be two-dimensional after conversion: "
            f"{sample.dataset_name}::{sample.sample_id}"
        )
    if image_array.shape != mask_array.shape:
        raise FourDatasetProtocolError(
            "image/mask dimensions differ after correction resolution for "
            f"{sample.dataset_name}::{sample.sample_id}: "
            f"image={image_array.shape}, mask={mask_array.shape}"
        )
    if not np.isfinite(image_array).all() or not np.isfinite(mask_array).all():
        raise FourDatasetProtocolError(
            f"non-finite pixels in {sample.dataset_name}::{sample.sample_id}"
        )
    return image_array, mask_array


def _pad_bottom_right(
    array: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    height, width = array.shape
    if height > target_height or width > target_width:
        raise FourDatasetProtocolError("padding target is smaller than input")
    return np.pad(
        array,
        ((0, target_height - height), (0, target_width - width)),
        mode="constant",
    )


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class FourDatasetTrainDataset(Dataset):
    """Formal train split with stateless seed-42 crop and augmentation."""

    def __init__(
        self,
        dataset: str,
        patch_size: int = PATCH_SIZE,
        seed: int = PROTOCOL_SEED,
        *,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        imgidx_manifest: (
            str | Path | Mapping[str, Any] | None
        ) = None,
        normalization_manifest: (
            str | Path | Mapping[str, Any] | None
        ) = None,
        correction_manifest: (
            str | Path | Mapping[str, Any] | None
        ) = None,
        return_metadata: bool = True,
    ) -> None:
        super().__init__()
        if patch_size != PATCH_SIZE:
            raise FourDatasetProtocolError(
                f"formal patch_size must be {PATCH_SIZE}"
            )
        if seed != PROTOCOL_SEED:
            raise FourDatasetProtocolError(
                f"formal training seed must be {PROTOCOL_SEED}"
            )
        self.dataset_name = dataset
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.patch_size = patch_size
        self.seed = seed
        self.return_metadata = bool(return_metadata)
        if imgidx_manifest is None and DEFAULT_IMGIDX_MANIFEST.is_file():
            imgidx_manifest = DEFAULT_IMGIDX_MANIFEST
        self.sample_ids = load_frozen_index(
            self.dataset_root,
            self.dataset_name,
            "train",
            imgidx_manifest,
        )
        self.normalization = get_legacy_normalization(
            self.dataset_name,
            normalization_manifest,
        )
        if (
            correction_manifest is None
            and DEFAULT_CORRECTION_MANIFEST.is_file()
        ):
            correction_manifest = DEFAULT_CORRECTION_MANIFEST
        self.correction_manifest = load_correction_manifest(
            correction_manifest,
            dataset_root=self.dataset_root,
            required=False,
            verify_files=False,
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise FourDatasetProtocolError(
                "epoch must be a non-negative integer"
            )
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _resolve(self, index: int) -> ResolvedSample:
        return resolve_sample(
            self.dataset_root,
            self.dataset_name,
            self.sample_ids[index],
            self.correction_manifest,
            split="train",
        )

    def __getitem__(self, index: int) -> Any:
        sample = self._resolve(index)
        image, raw_mask = _load_image_and_mask(sample)
        original_height, original_width = image.shape
        image = (
            image - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        mask = raw_mask / np.float32(255.0)
        mask_any = mask > 0

        def has_positive(top: int, left: int, size: int) -> bool:
            return bool(
                np.any(mask_any[top : top + size, left : left + size])
            )

        plan = derive_stateless_transform_plan(
            protocol_seed=self.seed,
            dataset_name=self.dataset_name,
            epoch=self.epoch,
            namespaced_id=sample.namespaced_sample_id,
            image_height=original_height,
            image_width=original_width,
            has_positive_in_crop=has_positive,
            patch_size=self.patch_size,
        )
        image = _pad_bottom_right(
            image, plan.padded_height, plan.padded_width
        )
        mask = _pad_bottom_right(
            mask, plan.padded_height, plan.padded_width
        )
        top = plan.crop_top
        left = plan.crop_left
        size = plan.crop_size
        image = image[top : top + size, left : left + size]
        mask = mask[top : top + size, left : left + size]
        if plan.flip_axis0:
            image = image[::-1, :]
            mask = mask[::-1, :]
        if plan.flip_axis1:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
        if plan.transpose:
            image = image.transpose(1, 0)
            mask = mask.transpose(1, 0)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[np.newaxis, :], dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask[np.newaxis, :], dtype=np.float32)
        )
        if not self.return_metadata:
            return image_tensor, mask_tensor
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset_name": self.dataset_name,
            "source_dataset": sample.source_dataset,
            "sample_id": sample.sample_id,
            "namespaced_sample_id": sample.namespaced_sample_id,
            "original_hw": (original_height, original_width),
            "epoch": self.epoch,
            "augmentation_seed": plan.augmentation_seed,
            "transform_plan": asdict(plan),
            "correction_applied": sample.correction_applied,
        }


class FourDatasetTestDataset(Dataset):
    """Full-image deterministic test split with strict correction handling."""

    def __init__(
        self,
        train_dataset_name: str,
        test_dataset_name: str,
        *,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        imgidx_manifest: (
            str | Path | Mapping[str, Any] | None
        ) = None,
        normalization_manifest: (
            str | Path | Mapping[str, Any] | None
        ) = None,
        correction_manifest: (
            str | Path | Mapping[str, Any] | None
        ) = None,
        return_metadata: bool = False,
        source_filter: str | None = None,
    ) -> None:
        super().__init__()
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.return_metadata = bool(return_metadata)
        if imgidx_manifest is None and DEFAULT_IMGIDX_MANIFEST.is_file():
            imgidx_manifest = DEFAULT_IMGIDX_MANIFEST
        sample_ids = load_frozen_index(
            self.dataset_root,
            self.test_dataset_name,
            "test",
            imgidx_manifest,
        )
        if source_filter is not None:
            sample_ids = [
                sample_id
                for sample_id in sample_ids
                if source_dataset_for_sample(
                    self.dataset_root,
                    self.test_dataset_name,
                    sample_id,
                )
                == source_filter
            ]
            if not sample_ids:
                raise FourDatasetProtocolError(
                    f"source_filter={source_filter!r} selected no samples"
                )
        self.sample_ids = sample_ids
        self.normalization = get_legacy_normalization(
            self.train_dataset_name,
            normalization_manifest,
        )
        if (
            correction_manifest is None
            and DEFAULT_CORRECTION_MANIFEST.is_file()
        ):
            correction_manifest = DEFAULT_CORRECTION_MANIFEST
        requires_correction = (
            self.test_dataset_name == "NUAA-SIRST"
            and "Misc_111" in self.sample_ids
        )
        self.correction_manifest = load_correction_manifest(
            correction_manifest,
            dataset_root=self.dataset_root,
            required=requires_correction,
            verify_files=requires_correction,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _resolve(self, index: int) -> ResolvedSample:
        return resolve_sample(
            self.dataset_root,
            self.test_dataset_name,
            self.sample_ids[index],
            self.correction_manifest,
            split="test",
        )

    def sample_record(self, index: int) -> dict[str, Any]:
        sample = self._resolve(index)
        pair = validate_pair(sample)
        return {
            "train_dataset_name": self.train_dataset_name,
            "test_dataset_name": self.test_dataset_name,
            "source_dataset": sample.source_dataset,
            "sample_id": sample.sample_id,
            "namespaced_sample_id": sample.namespaced_sample_id,
            "original_width_height": pair[
                "image_size_width_height"
            ],
            "correction_applied": sample.correction_applied,
            "correction_id": sample.correction_id,
        }

    def __getitem__(self, index: int) -> Any:
        sample = self._resolve(index)
        image, raw_mask = _load_image_and_mask(sample)
        original_height, original_width = image.shape
        image = (
            image - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        mask = raw_mask / np.float32(255.0)
        padded_height = _next_multiple(
            original_height, PAD_MULTIPLE
        )
        padded_width = _next_multiple(original_width, PAD_MULTIPLE)
        image = _pad_bottom_right(image, padded_height, padded_width)
        mask = _pad_bottom_right(mask, padded_height, padded_width)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[np.newaxis, :], dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask[np.newaxis, :], dtype=np.float32)
        )
        original_hw = (original_height, original_width)
        if not self.return_metadata:
            return image_tensor, mask_tensor, original_hw, sample.sample_id
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "train_dataset_name": self.train_dataset_name,
            "test_dataset_name": self.test_dataset_name,
            "source_dataset": sample.source_dataset,
            "sample_id": sample.sample_id,
            "namespaced_sample_id": sample.namespaced_sample_id,
            "original_hw": original_hw,
            "correction_applied": sample.correction_applied,
            "correction_id": sample.correction_id,
        }


def build_train_dataset(
    dataset_name: str,
    **kwargs: Any,
) -> FourDatasetTrainDataset:
    return FourDatasetTrainDataset(dataset_name, **kwargs)


def build_test_dataset(
    dataset_name: str | None = None,
    *,
    train_dataset_name: str | None = None,
    test_dataset_name: str | None = None,
    normalization_dataset: str | None = None,
    **kwargs: Any,
) -> FourDatasetTestDataset:
    """Flexible evaluator factory with train-selected normalization.

    Examples:

    ``build_test_dataset("NUAA-SIRST")`` uses NUAA normalization.

    ``build_test_dataset(test_dataset_name="NUAA-SIRST",
    normalization_dataset="SIRST3")`` evaluates an SIRST3 checkpoint on the
    official NUAA split while retaining SIRST3 normalization.
    """

    if test_dataset_name is None:
        if dataset_name is None:
            raise FourDatasetProtocolError("test dataset name is required")
        test_dataset_name = dataset_name
    elif dataset_name is not None and dataset_name != test_dataset_name:
        raise FourDatasetProtocolError(
            "dataset_name and test_dataset_name disagree"
        )
    normalization = normalization_dataset or train_dataset_name
    if normalization is None:
        normalization = test_dataset_name
    return FourDatasetTestDataset(
        normalization,
        test_dataset_name,
        **kwargs,
    )


def PaperFourDataset(
    dataset_name: str,
    *,
    split: str,
    train_dataset_name: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """Small mode-selecting facade for runner/evaluator integration."""

    if split == "train":
        if train_dataset_name not in (None, dataset_name):
            raise FourDatasetProtocolError(
                "train_dataset_name must equal dataset_name in train mode"
            )
        return FourDatasetTrainDataset(dataset_name, **kwargs)
    if split == "test":
        return FourDatasetTestDataset(
            train_dataset_name or dataset_name,
            dataset_name,
            **kwargs,
        )
    raise FourDatasetProtocolError("split must be 'train' or 'test'")


__all__ = [
    "FourDatasetTestDataset",
    "FourDatasetTrainDataset",
    "PaperFourDataset",
    "build_test_dataset",
    "build_train_dataset",
]
