"""PyTorch datasets bound to the frozen three-dataset V2 protocol.

The two datasets in this module have one data source only:
``experiments.three_dataset_v2_protocol``.  They retain the authoritative
``img_idx`` order, use the NUAA-internal ``masks_corrected`` overlay, and do
not import the historical four-regime data implementation.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from experiments import three_dataset_v2_protocol as protocol


def _load_pair(
    sample: protocol.ResolvedSample,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(sample.image_path) as image:
        image_array = np.asarray(image.convert("I"), dtype=np.float32)
    with Image.open(sample.mask_path) as mask:
        mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim > 2:
        mask_array = mask_array[:, :, 0]
    if image_array.ndim != 2 or mask_array.ndim != 2:
        raise protocol.ThreeDatasetV2ProtocolError(
            "image and mask must be two-dimensional after conversion: "
            f"{sample.dataset_name}::{sample.sample_id}"
        )
    if image_array.shape != mask_array.shape:
        raise protocol.ThreeDatasetV2ProtocolError(
            "effective image/mask dimensions differ: "
            f"{sample.dataset_name}::{sample.sample_id}: "
            f"{image_array.shape} != {mask_array.shape}"
        )
    if not np.isfinite(image_array).all() or not np.isfinite(mask_array).all():
        raise protocol.ThreeDatasetV2ProtocolError(
            f"non-finite pixels in {sample.dataset_name}::{sample.sample_id}"
        )
    return image_array, mask_array


def _pad_bottom_right(
    array: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    height, width = array.shape
    if height > target_height or width > target_width:
        raise protocol.ThreeDatasetV2ProtocolError(
            "padding target is smaller than input"
        )
    return np.pad(
        array,
        ((0, target_height - height), (0, target_width - width)),
        mode="constant",
    )


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class ThreeDatasetV2TrainDataset(Dataset):
    """Seed-42 stateless crops from one frozen ``img_idx/train`` split."""

    def __init__(
        self,
        dataset: str,
        *,
        dataset_root: str | Path = protocol.DEFAULT_DATASET_ROOT,
        protocol_manifest: str | Path | Mapping[str, Any] = (
            protocol.DEFAULT_MANIFEST_PATH
        ),
        patch_size: int = protocol.PATCH_SIZE,
        seed: int = protocol.PROTOCOL_SEED,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_name = protocol.require_dataset(dataset)
        protocol.require_seed(seed)
        if isinstance(patch_size, bool) or patch_size != protocol.PATCH_SIZE:
            raise protocol.ThreeDatasetV2ProtocolError(
                f"formal patch_size must be {protocol.PATCH_SIZE}"
            )
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.protocol_manifest = protocol_manifest
        self.patch_size = patch_size
        self.seed = seed
        self.return_metadata = bool(return_metadata)
        self.sample_ids = protocol.load_frozen_index(
            self.dataset_root,
            self.dataset_name,
            "train",
            self.protocol_manifest,
        )
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = protocol.get_legacy_normalization(
            self.dataset_name
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise protocol.ThreeDatasetV2ProtocolError(
                "epoch must be a non-negative integer"
            )
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _resolve(self, index: int) -> protocol.ResolvedSample:
        return protocol.resolve_sample(
            self.dataset_root,
            self.dataset_name,
            self.sample_ids[index],
            split="train",
            known_ids=self._known_ids,
        )

    def __getitem__(self, index: int) -> Any:
        sample = self._resolve(index)
        image, raw_mask = _load_pair(sample)
        original_height, original_width = image.shape
        image = (
            image - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        mask = raw_mask / np.float32(255.0)
        mask_any = mask > 0

        def has_positive(top: int, left: int, size: int) -> bool:
            return bool(np.any(mask_any[top : top + size, left : left + size]))

        namespaced_id = f"{self.dataset_name}::{sample.sample_id}"
        plan = protocol.derive_stateless_transform_plan(
            protocol_seed=self.seed,
            dataset_name=self.dataset_name,
            epoch=self.epoch,
            namespaced_id=namespaced_id,
            image_height=original_height,
            image_width=original_width,
            has_positive_in_crop=has_positive,
            patch_size=self.patch_size,
        )
        image = _pad_bottom_right(
            image, plan.padded_height, plan.padded_width
        )
        mask = _pad_bottom_right(mask, plan.padded_height, plan.padded_width)
        top, left, size = plan.crop_top, plan.crop_left, plan.crop_size
        image = image[top : top + size, left : left + size]
        mask = mask[top : top + size, left : left + size]
        if plan.flip_axis0:
            image, mask = image[::-1, :], mask[::-1, :]
        if plan.flip_axis1:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if plan.transpose:
            image, mask = image.transpose(1, 0), mask.transpose(1, 0)
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
            "sample_id": sample.sample_id,
            "namespaced_sample_id": namespaced_id,
            "original_hw": (original_height, original_width),
            "epoch": self.epoch,
            "augmentation_seed": plan.augmentation_seed,
            "transform_plan": asdict(plan),
            "correction_applied": sample.correction_applied,
            "correction_id": sample.correction_id,
        }


class ThreeDatasetV2TestDataset(Dataset):
    """Full-image ``img_idx/test`` data with frozen train normalization."""

    def __init__(
        self,
        train_dataset_name: str,
        test_dataset_name: str | None = None,
        *,
        dataset_root: str | Path = protocol.DEFAULT_DATASET_ROOT,
        protocol_manifest: str | Path | Mapping[str, Any] = (
            protocol.DEFAULT_MANIFEST_PATH
        ),
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset_name = protocol.require_dataset(train_dataset_name)
        self.test_dataset_name = protocol.require_dataset(
            test_dataset_name or train_dataset_name
        )
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.protocol_manifest = protocol_manifest
        self.return_metadata = bool(return_metadata)
        self.sample_ids = protocol.load_frozen_index(
            self.dataset_root,
            self.test_dataset_name,
            "test",
            self.protocol_manifest,
        )
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = protocol.get_legacy_normalization(
            self.train_dataset_name
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _resolve(self, index: int) -> protocol.ResolvedSample:
        return protocol.resolve_sample(
            self.dataset_root,
            self.test_dataset_name,
            self.sample_ids[index],
            split="test",
            known_ids=self._known_ids,
        )

    def sample_record(self, index: int) -> dict[str, Any]:
        sample = self._resolve(index)
        pair = protocol.validate_sample_pair(sample)
        return {
            "train_dataset_name": self.train_dataset_name,
            "test_dataset_name": self.test_dataset_name,
            **pair,
        }

    def __getitem__(self, index: int) -> Any:
        sample = self._resolve(index)
        image, raw_mask = _load_pair(sample)
        original_height, original_width = image.shape
        image = (
            image - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        mask = raw_mask / np.float32(255.0)
        padded_height = _next_multiple(original_height, protocol.PAD_MULTIPLE)
        padded_width = _next_multiple(original_width, protocol.PAD_MULTIPLE)
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
            "sample_id": sample.sample_id,
            "namespaced_sample_id": (
                f"{self.test_dataset_name}::{sample.sample_id}"
            ),
            "original_hw": original_hw,
            "correction_applied": sample.correction_applied,
            "correction_id": sample.correction_id,
        }


def build_train_dataset(
    dataset_name: str, **kwargs: Any
) -> ThreeDatasetV2TrainDataset:
    return ThreeDatasetV2TrainDataset(dataset_name, **kwargs)


def build_test_dataset(
    dataset_name: str, **kwargs: Any
) -> ThreeDatasetV2TestDataset:
    return ThreeDatasetV2TestDataset(dataset_name, dataset_name, **kwargs)


__all__ = [
    "ThreeDatasetV2TestDataset",
    "ThreeDatasetV2TrainDataset",
    "build_test_dataset",
    "build_train_dataset",
]
