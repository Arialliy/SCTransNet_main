"""Versioned fixed-0.5 metric core for PBDR-V4.

The object metric delegates to :mod:`component_matching_v2`, whose assignment
semantics are identical to the current formal V3 evaluator.  This module adds
the exact integer sufficient statistics required by the zero-margin selector
and performs online accumulation, so an official pass never needs to retain a
dataset-wide probability or logit cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from typing import Mapping

import numpy as np

from experiments.component_matching_v2 import (
    CANONICAL_CONNECTIVITY,
    CANONICAL_MATCH_RADIUS,
    match_components_v2,
)


METRIC_CORE_SCHEMA = "sctransnet_pbdr_v4_metric_core/v1"
FIXED_THRESHOLD = 0.5
TINY_AREA = 9


class PBDRV4MetricError(ValueError):
    """An input or accumulator state violates the frozen metric contract."""


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    ready = float(value)
    if not math.isfinite(ready):
        raise PBDRV4MetricError(f"{name} must be finite")
    return ready


def _float_map(value: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 2 or min(value.shape) < 1:
        raise PBDRV4MetricError(f"{name} must be a non-empty 2D array")
    if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must use float32 or float64")
    if not bool(np.isfinite(value).all()):
        raise PBDRV4MetricError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value)


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PBDRV4MetricError("identifier must be a non-empty string")
    if "\x00" in value:
        raise PBDRV4MetricError("identifier contains NUL")
    return value


@dataclass(frozen=True, slots=True)
class MetricCoreManifest:
    schema: str = METRIC_CORE_SCHEMA
    probability_comparison: str = "strict_greater_than"
    threshold: float = FIXED_THRESHOLD
    target_comparison: str = "strict_greater_than"
    target_threshold: float = FIXED_THRESHOLD
    connectivity: int = CANONICAL_CONNECTIVITY
    match_radius: float = CANONICAL_MATCH_RADIUS
    match_radius_comparison: str = "strict_less_than"
    tiny_area: int = TINY_AREA
    assignment: str = "hungarian_max_cardinality_then_min_centroid_distance"

    def as_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


class PBDRV4MetricAccumulator:
    """Online fixed-threshold metrics for one candidate on one frozen split."""

    def __init__(self) -> None:
        self.intersection_pixels = 0
        self.union_pixels = 0
        self.true_positive_pixels = 0
        self.false_positive_pixels = 0
        self.false_negative_pixels = 0
        self.target_count = 0
        self.matched_target_count = 0
        self.tiny_target_count = 0
        self.matched_tiny_target_count = 0
        self.predicted_object_count = 0
        self.unmatched_predicted_object_count = 0
        self.unmatched_component_pixels = 0
        self.valid_pixels = 0
        self._image_ious: list[float] = []
        self._losses: list[float] = []
        self._identifiers: list[str] = []
        self._sample_id_hasher = hashlib.sha256()
        self._target_hasher = hashlib.sha256()

    @property
    def sample_count(self) -> int:
        return len(self._identifiers)

    def update(
        self,
        *,
        probability: np.ndarray,
        target: np.ndarray,
        loss: float,
        identifier: str,
    ) -> None:
        probability_ready = _float_map(probability, name="probability")
        target_ready = _float_map(target, name="target")
        if probability_ready.shape != target_ready.shape:
            raise PBDRV4MetricError("probability and target shapes differ")
        loss_ready = _finite_real(loss, name="loss")
        if loss_ready < 0.0:
            raise PBDRV4MetricError("loss must be non-negative")
        identifier_ready = _identifier(identifier)
        if identifier_ready in set(self._identifiers):
            raise PBDRV4MetricError(f"duplicate sample identifier: {identifier_ready}")

        prediction = np.ascontiguousarray(
            probability_ready > FIXED_THRESHOLD,
            dtype=np.bool_,
        )
        target_binary = np.ascontiguousarray(
            target_ready > FIXED_THRESHOLD,
            dtype=np.bool_,
        )
        intersection = int(np.logical_and(prediction, target_binary).sum())
        union = int(np.logical_or(prediction, target_binary).sum())
        self.intersection_pixels += intersection
        self.union_pixels += union
        self.true_positive_pixels += intersection
        self.false_positive_pixels += int(
            np.logical_and(prediction, ~target_binary).sum()
        )
        self.false_negative_pixels += int(
            np.logical_and(~prediction, target_binary).sum()
        )
        self.valid_pixels += int(target_binary.size)
        self._image_ious.append(1.0 if union == 0 else intersection / union)
        self._losses.append(loss_ready)
        self._identifiers.append(identifier_ready)

        match = match_components_v2(
            prediction_mask=prediction,
            target_mask=target_binary,
            match_radius=CANONICAL_MATCH_RADIUS,
            connectivity=CANONICAL_CONNECTIVITY,
        )
        targets_by_id = {record.component_id: record for record in match.targets}
        self.target_count += len(match.targets)
        self.matched_target_count += len(match.matched_target_ids)
        tiny_ids = {
            component_id
            for component_id, record in targets_by_id.items()
            if record.area <= TINY_AREA
        }
        self.tiny_target_count += len(tiny_ids)
        self.matched_tiny_target_count += len(
            tiny_ids & set(match.matched_target_ids)
        )
        self.predicted_object_count += len(match.predictions)
        self.unmatched_predicted_object_count += len(
            match.unmatched_prediction_ids
        )
        self.unmatched_component_pixels += match.unmatched_prediction_pixels

        encoded_id = identifier_ready.encode("utf-8")
        shape = np.asarray(target_binary.shape, dtype="<i8").tobytes()
        packed = np.packbits(target_binary.reshape(-1), bitorder="little").tobytes()
        for hasher in (self._sample_id_hasher, self._target_hasher):
            hasher.update(len(encoded_id).to_bytes(8, "little"))
            hasher.update(encoded_id)
        self._target_hasher.update(shape)
        self._target_hasher.update(len(packed).to_bytes(8, "little"))
        self._target_hasher.update(packed)

    def compute(self) -> dict[str, object]:
        if self.sample_count == 0:
            raise PBDRV4MetricError("cannot compute an empty accumulator")
        if self.target_count <= 0 or self.union_pixels <= 0:
            raise PBDRV4MetricError(
                "formal selector requires positive target and union counts"
            )
        precision_denominator = self.true_positive_pixels + self.false_positive_pixels
        recall_denominator = self.true_positive_pixels + self.false_negative_pixels
        precision = self.true_positive_pixels / max(1, precision_denominator)
        recall = self.true_positive_pixels / max(1, recall_denominator)
        f1_denominator = precision + recall
        tiny_pd = (
            self.matched_tiny_target_count / self.tiny_target_count
            if self.tiny_target_count
            else None
        )
        return {
            "schema": METRIC_CORE_SCHEMA,
            "sample_count": self.sample_count,
            "intersection_pixels": self.intersection_pixels,
            "union_pixels": self.union_pixels,
            "miou": self.intersection_pixels / self.union_pixels,
            "niou": math.fsum(self._image_ious) / self.sample_count,
            "test_loss": math.fsum(self._losses) / self.sample_count,
            "true_positive_pixels": self.true_positive_pixels,
            "false_positive_pixels": self.false_positive_pixels,
            "false_negative_pixels": self.false_negative_pixels,
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_f1": (
                0.0 if f1_denominator == 0.0 else 2.0 * precision * recall / f1_denominator
            ),
            "target_count": self.target_count,
            "matched_target_count": self.matched_target_count,
            "pd": self.matched_target_count / self.target_count,
            "tiny_target_count": self.tiny_target_count,
            "matched_tiny_target_count": self.matched_tiny_target_count,
            "tiny_pd": tiny_pd,
            "predicted_object_count": self.predicted_object_count,
            "unmatched_predicted_object_count": self.unmatched_predicted_object_count,
            "unmatched_component_pixels": self.unmatched_component_pixels,
            "fa": self.unmatched_component_pixels / self.valid_pixels,
            "false_objects_per_image": (
                self.unmatched_predicted_object_count / self.sample_count
            ),
            "valid_pixel_count": self.valid_pixels,
            "sample_id_order_sha256": self._sample_id_hasher.hexdigest(),
            "target_sha256": self._target_hasher.hexdigest(),
            "metric_manifest": MetricCoreManifest().as_dict(),
        }


def exact_statistics(metrics: Mapping[str, object]) -> dict[str, int]:
    """Extract exact counts needed by the selector from a metric payload."""

    fields = (
        "intersection_pixels",
        "union_pixels",
        "matched_target_count",
        "target_count",
        "matched_tiny_target_count",
        "tiny_target_count",
        "unmatched_component_pixels",
        "valid_pixel_count",
    )
    result: dict[str, int] = {}
    for name in fields:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        result[name] = int(value)
    return result


__all__ = [
    "FIXED_THRESHOLD",
    "METRIC_CORE_SCHEMA",
    "MetricCoreManifest",
    "PBDRV4MetricAccumulator",
    "PBDRV4MetricError",
    "TINY_AREA",
    "exact_statistics",
]
