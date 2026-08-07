#!/usr/bin/env python3
"""Pure component-atlas construction for PBDR-V4.

The functions here perform no file access and no model inference.  They filter
the canonical matcher ID maps into three full-image identity maps:

* rescue: unmatched target components;
* suppress: unmatched predicted components;
* preserve: matched target components.

Suppress components are allowed to overlap target pixels.  Centroid matching
does not imply pixel overlap, and an unmatched prediction can legitimately
cross a target while its centroid remains outside the strict match radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from experiments.component_matching_v2 import (
    CANONICAL_CONNECTIVITY,
    CANONICAL_MATCH_RADIUS,
    ComponentMatchResult,
    ComponentRecord,
    match_components_v2,
)


@dataclass(frozen=True, slots=True)
class AtlasMaps:
    """Three categorical component-ID maps aligned to one source image."""

    rescue_ids: np.ndarray = field(repr=False, compare=False)
    suppress_ids: np.ndarray = field(repr=False, compare=False)
    preserve_ids: np.ndarray = field(repr=False, compare=False)


def _positive_ids(id_map: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.unique(id_map) if int(value) > 0)


def _validate_component_table(
    id_map: np.ndarray,
    records: tuple[ComponentRecord, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(id_map, np.ndarray):
        raise TypeError(f"{name}_id_map must be a numpy.ndarray")
    if id_map.dtype != np.dtype(np.int32):
        raise TypeError(f"{name}_id_map must have int32 dtype")
    if id_map.ndim != 2:
        raise ValueError(f"{name}_id_map must be two-dimensional")
    if bool(np.any(id_map < 0)):
        raise ValueError(f"{name}_id_map contains a negative ID")

    record_ids = tuple(record.component_id for record in records)
    if record_ids != tuple(sorted(record_ids)) or len(record_ids) != len(
        set(record_ids)
    ):
        raise ValueError(f"{name} component records are not uniquely sorted")
    if any(component_id <= 0 for component_id in record_ids):
        raise ValueError(f"{name} component records contain a non-positive ID")
    if _positive_ids(id_map) != record_ids:
        raise ValueError(f"{name} ID map and component records differ")
    for record in records:
        observed_area = int(np.count_nonzero(id_map == record.component_id))
        if observed_area != record.area:
            raise ValueError(f"{name} component area differs from its ID map")
    return record_ids


def _validate_partition(
    *,
    all_ids: tuple[int, ...],
    matched_ids: tuple[int, ...],
    unmatched_ids: tuple[int, ...],
    name: str,
) -> None:
    if matched_ids != tuple(sorted(matched_ids)) or len(matched_ids) != len(
        set(matched_ids)
    ):
        raise ValueError(f"matched {name} IDs are not uniquely sorted")
    if unmatched_ids != tuple(sorted(unmatched_ids)) or len(
        unmatched_ids
    ) != len(set(unmatched_ids)):
        raise ValueError(f"unmatched {name} IDs are not uniquely sorted")
    matched = set(matched_ids)
    unmatched = set(unmatched_ids)
    if matched & unmatched or matched | unmatched != set(all_ids):
        raise ValueError(f"matched/unmatched {name} IDs do not form a partition")


def _validate_match_result(result: ComponentMatchResult) -> None:
    if not isinstance(result, ComponentMatchResult):
        raise TypeError("result must be a ComponentMatchResult")
    if result.target_id_map.shape != result.prediction_id_map.shape:
        raise ValueError("target and prediction ID maps must share shape")

    target_ids = _validate_component_table(
        result.target_id_map, result.targets, name="target"
    )
    prediction_ids = _validate_component_table(
        result.prediction_id_map, result.predictions, name="prediction"
    )
    _validate_partition(
        all_ids=target_ids,
        matched_ids=result.matched_target_ids,
        unmatched_ids=result.unmatched_target_ids,
        name="target",
    )
    _validate_partition(
        all_ids=prediction_ids,
        matched_ids=result.matched_prediction_ids,
        unmatched_ids=result.unmatched_prediction_ids,
        name="prediction",
    )

    pair_target_ids = tuple(pair.target_id for pair in result.matches)
    pair_prediction_ids = tuple(pair.prediction_id for pair in result.matches)
    if pair_target_ids != result.matched_target_ids:
        raise ValueError("match pairs differ from matched target IDs")
    if tuple(sorted(pair_prediction_ids)) != result.matched_prediction_ids:
        raise ValueError("match pairs differ from matched prediction IDs")
    if len(pair_prediction_ids) != len(set(pair_prediction_ids)):
        raise ValueError("a prediction ID appears in more than one match")
    if any(
        not math.isfinite(pair.centroid_distance)
        or pair.centroid_distance < 0.0
        for pair in result.matches
    ):
        raise ValueError("match pairs contain an invalid centroid distance")

    predictions_by_id = {
        record.component_id: record for record in result.predictions
    }
    expected_unmatched_pixels = sum(
        predictions_by_id[component_id].area
        for component_id in result.unmatched_prediction_ids
    )
    if result.unmatched_prediction_pixels != expected_unmatched_pixels:
        raise ValueError("unmatched prediction pixel count differs from IDs")


def _filter_id_map(
    id_map: np.ndarray,
    selected_ids: tuple[int, ...],
) -> np.ndarray:
    maximum = int(id_map.max(initial=0))
    selected = np.zeros(maximum + 1, dtype=np.bool_)
    if selected_ids:
        selected[np.asarray(selected_ids, dtype=np.int64)] = True
    output = np.where(selected[id_map], id_map, 0).astype(
        np.int32, copy=False
    )
    output = np.ascontiguousarray(output)
    output.setflags(write=False)
    return output


def atlas_maps_from_match(result: ComponentMatchResult) -> AtlasMaps:
    """Filter one canonical match result into rescue/suppress/preserve IDs."""

    _validate_match_result(result)
    rescue_ids = _filter_id_map(
        result.target_id_map, result.unmatched_target_ids
    )
    preserve_ids = _filter_id_map(
        result.target_id_map, result.matched_target_ids
    )
    suppress_ids = _filter_id_map(
        result.prediction_id_map, result.unmatched_prediction_ids
    )

    if bool(np.any((rescue_ids > 0) & (preserve_ids > 0))):
        raise RuntimeError("rescue and preserve maps overlap")
    target_support = result.target_id_map > 0
    if not np.array_equal(
        (rescue_ids > 0) | (preserve_ids > 0), target_support
    ):
        raise RuntimeError("rescue/preserve maps do not partition the target")
    if bool(np.any((suppress_ids > 0) & (result.prediction_id_map == 0))):
        raise RuntimeError("suppress map escapes prediction support")
    if _positive_ids(rescue_ids) != result.unmatched_target_ids:
        raise RuntimeError("rescue map IDs differ from unmatched targets")
    if _positive_ids(preserve_ids) != result.matched_target_ids:
        raise RuntimeError("preserve map IDs differ from matched targets")
    if _positive_ids(suppress_ids) != result.unmatched_prediction_ids:
        raise RuntimeError("suppress map IDs differ from unmatched predictions")

    return AtlasMaps(
        rescue_ids=rescue_ids,
        suppress_ids=suppress_ids,
        preserve_ids=preserve_ids,
    )


def build_component_atlas(
    *,
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    match_radius: float = CANONICAL_MATCH_RADIUS,
    connectivity: int = CANONICAL_CONNECTIVITY,
) -> tuple[ComponentMatchResult, AtlasMaps]:
    """Match two boolean masks and return their pure in-memory atlas maps."""

    result = match_components_v2(
        prediction_mask=prediction_mask,
        target_mask=target_mask,
        match_radius=match_radius,
        connectivity=connectivity,
    )
    return result, atlas_maps_from_match(result)


__all__ = [
    "AtlasMaps",
    "atlas_maps_from_match",
    "build_component_atlas",
]
