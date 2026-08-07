#!/usr/bin/env python3
"""Canonical component matching for new SCTransNet experiment protocols.

This module deliberately does not replace the source-locked historical metric
implementations.  It centralizes their current formal matching semantics for
new evaluators and component-atlas builders:

* two-dimensional 8-connected components;
* a strict Euclidean centroid-distance comparison;
* maximum-cardinality one-to-one matching first;
* minimum total centroid distance second.

Callers must perform probability/target thresholding explicitly and pass
boolean masks.  Keeping thresholding outside this module prevents a silent
``>`` versus ``>=`` protocol change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import measure


CANONICAL_CONNECTIVITY = 2
CANONICAL_MATCH_RADIUS = 3.0


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    """Stable scalar metadata for one positive connected component."""

    component_id: int
    area: int
    centroid_y: float
    centroid_x: float


@dataclass(frozen=True, slots=True)
class ComponentPair:
    """One target-to-prediction match under the canonical assignment."""

    target_id: int
    prediction_id: int
    centroid_distance: float


@dataclass(frozen=True, slots=True)
class ComponentMatchResult:
    """Complete ID-based result for evaluation and atlas construction."""

    target_id_map: np.ndarray = field(repr=False, compare=False)
    prediction_id_map: np.ndarray = field(repr=False, compare=False)
    targets: tuple[ComponentRecord, ...]
    predictions: tuple[ComponentRecord, ...]
    matches: tuple[ComponentPair, ...]
    matched_target_ids: tuple[int, ...]
    matched_prediction_ids: tuple[int, ...]
    unmatched_target_ids: tuple[int, ...]
    unmatched_prediction_ids: tuple[int, ...]
    unmatched_prediction_pixels: int


def _validated_binary_mask(value: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must have bool dtype")
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} dimensions must be positive")
    return np.ascontiguousarray(value)


def _validated_radius(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("match_radius must be a real number")
    ready = float(value)
    if not math.isfinite(ready) or ready <= 0.0:
        raise ValueError("match_radius must be finite and positive")
    return ready


def _validated_connectivity(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("connectivity must be an integer")
    ready = int(value)
    if ready != CANONICAL_CONNECTIVITY:
        raise ValueError(
            f"canonical two-dimensional connectivity must be "
            f"{CANONICAL_CONNECTIVITY} (8-connected)"
        )
    return ready


def _label_and_describe(
    mask: np.ndarray,
    *,
    connectivity: int,
) -> tuple[np.ndarray, tuple[ComponentRecord, ...]]:
    labeled = measure.label(mask, connectivity=connectivity, background=0)
    maximum = int(labeled.max(initial=0))
    if maximum > np.iinfo(np.int32).max:
        raise OverflowError("component IDs exceed int32 capacity")
    id_map = np.asarray(labeled, dtype=np.int32, order="C")
    regions = sorted(measure.regionprops(id_map), key=lambda region: region.label)
    records = tuple(
        ComponentRecord(
            component_id=int(region.label),
            area=int(region.area),
            centroid_y=float(region.centroid[0]),
            centroid_x=float(region.centroid[1]),
        )
        for region in regions
    )
    id_map.setflags(write=False)
    return id_map, records


def _distance_matrix(
    targets: tuple[ComponentRecord, ...],
    predictions: tuple[ComponentRecord, ...],
) -> np.ndarray:
    distances = np.empty((len(targets), len(predictions)), dtype=np.float64)
    for target_index, target in enumerate(targets):
        target_centroid = np.asarray(
            (target.centroid_y, target.centroid_x), dtype=np.float64
        )
        for prediction_index, prediction in enumerate(predictions):
            prediction_centroid = np.asarray(
                (prediction.centroid_y, prediction.centroid_x),
                dtype=np.float64,
            )
            distances[target_index, prediction_index] = np.linalg.norm(
                prediction_centroid - target_centroid
            )
    return distances


def match_components_v2(
    *,
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    match_radius: float = CANONICAL_MATCH_RADIUS,
    connectivity: int = CANONICAL_CONNECTIVITY,
) -> ComponentMatchResult:
    """Match binary prediction and target components by canonical IDs.

    The assignment cost is intentionally the same cardinality-reward
    construction used by the repository's current formal
    ``ValidationMetrics`` implementation.  One zero-cost dummy column per
    target permits every target to remain unmatched.  Every valid real edge
    receives a sufficiently large negative reward, so the solver maximizes
    match cardinality before minimizing total centroid distance.
    """

    prediction = _validated_binary_mask(
        prediction_mask, name="prediction_mask"
    )
    target = _validated_binary_mask(target_mask, name="target_mask")
    if prediction.shape != target.shape:
        raise ValueError("prediction_mask and target_mask must share shape")
    radius = _validated_radius(match_radius)
    ready_connectivity = _validated_connectivity(connectivity)

    prediction_id_map, predictions = _label_and_describe(
        prediction, connectivity=ready_connectivity
    )
    target_id_map, targets = _label_and_describe(
        target, connectivity=ready_connectivity
    )

    matched_target_indices: set[int] = set()
    matched_prediction_indices: set[int] = set()
    pairs: list[ComponentPair] = []
    if targets and predictions:
        distances = _distance_matrix(targets, predictions)

        # This is the source-locked formal cost semantics.  Do not add an
        # epsilon tie-breaker: doing so could alter the distance objective.
        cardinality_reward = (
            min(len(targets), len(predictions)) + 1
        ) * max(1.0, radius)
        real_cost = np.where(
            distances < radius,
            distances - cardinality_reward,
            cardinality_reward,
        )
        assignment_cost = np.concatenate(
            (
                real_cost,
                np.zeros((len(targets), len(targets)), dtype=np.float64),
            ),
            axis=1,
        )
        assigned_targets, assigned_columns = linear_sum_assignment(
            assignment_cost
        )
        for target_index, column_index in zip(
            assigned_targets.tolist(), assigned_columns.tolist()
        ):
            if (
                column_index < len(predictions)
                and distances[target_index, column_index] < radius
            ):
                matched_target_indices.add(int(target_index))
                matched_prediction_indices.add(int(column_index))
                pairs.append(
                    ComponentPair(
                        target_id=targets[target_index].component_id,
                        prediction_id=predictions[column_index].component_id,
                        centroid_distance=float(
                            distances[target_index, column_index]
                        ),
                    )
                )

    pairs.sort(key=lambda pair: (pair.target_id, pair.prediction_id))
    matched_target_ids = tuple(
        targets[index].component_id for index in sorted(matched_target_indices)
    )
    matched_prediction_ids = tuple(
        predictions[index].component_id
        for index in sorted(matched_prediction_indices)
    )
    unmatched_target_ids = tuple(
        target.component_id
        for index, target in enumerate(targets)
        if index not in matched_target_indices
    )
    unmatched_prediction_ids = tuple(
        prediction.component_id
        for index, prediction in enumerate(predictions)
        if index not in matched_prediction_indices
    )
    predictions_by_id = {
        prediction.component_id: prediction for prediction in predictions
    }
    unmatched_prediction_pixels = sum(
        predictions_by_id[component_id].area
        for component_id in unmatched_prediction_ids
    )

    return ComponentMatchResult(
        target_id_map=target_id_map,
        prediction_id_map=prediction_id_map,
        targets=targets,
        predictions=predictions,
        matches=tuple(pairs),
        matched_target_ids=matched_target_ids,
        matched_prediction_ids=matched_prediction_ids,
        unmatched_target_ids=unmatched_target_ids,
        unmatched_prediction_ids=unmatched_prediction_ids,
        unmatched_prediction_pixels=int(unmatched_prediction_pixels),
    )


__all__ = [
    "CANONICAL_CONNECTIVITY",
    "CANONICAL_MATCH_RADIUS",
    "ComponentMatchResult",
    "ComponentPair",
    "ComponentRecord",
    "match_components_v2",
]
