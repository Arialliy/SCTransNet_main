"""Pure IRSTD target-core and attached-halo atlas construction.

This module performs no dataset, split, index, checkpoint, or model loading.
Callers provide aligned in-memory Current logits and a target mask.  Optional
Baseline logits are used only when an actual bound prediction is supplied;
missing Baseline evidence is represented by ``None``, never by an all-zero map
that could be mistaken for an available teacher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral

import numpy as np
from scipy import ndimage

from experiments.component_matching_v2 import match_components_v2


IRSTD_ERROR_ATLAS_VERSION = "irstd_error_atlas_v1"


@dataclass(frozen=True, slots=True)
class IRSTDErrorAtlas:
    """Aligned supervision maps for one IRSTD image."""

    target_component_ids: np.ndarray = field(repr=False, compare=False)
    rescue_component_ids: np.ndarray = field(repr=False, compare=False)
    core_target: np.ndarray = field(repr=False, compare=False)
    attached_halo: np.ndarray = field(repr=False, compare=False)
    detached_false_positive: np.ndarray = field(repr=False, compare=False)
    outer_ring: np.ndarray = field(repr=False, compare=False)
    halo_target: np.ndarray = field(repr=False, compare=False)
    far_background: np.ndarray = field(repr=False, compare=False)
    baseline_available: bool
    baseline_rescue: np.ndarray | None = field(
        repr=False,
        compare=False,
    )
    baseline_halo_advantage: np.ndarray | None = field(
        repr=False,
        compare=False,
    )


def _validated_radius(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    ready = int(value)
    if ready < 1:
        raise ValueError(f"{name} must be positive")
    return ready


def _validated_logits(value: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    ready = np.ascontiguousarray(value, dtype=np.float32)
    if not bool(np.isfinite(ready).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return ready


def _validated_binary_mask(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 2 or value.shape != shape:
        raise ValueError(f"{name} must match the logit shape")
    if value.dtype == np.dtype(np.bool_):
        return np.ascontiguousarray(value, dtype=np.bool_)
    if not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        raise TypeError(f"{name} must use bool or binary numeric values")
    if np.issubdtype(value.dtype, np.floating) and not bool(
        np.isfinite(value).all()
    ):
        raise FloatingPointError(f"{name} contains non-finite values")
    if not bool(np.all((value == 0) | (value == 1))):
        raise ValueError(f"{name} must contain only binary values")
    return np.ascontiguousarray(value.astype(np.bool_, copy=False))


def _component_core(component: np.ndarray) -> np.ndarray:
    """Return a non-empty morphology core for one non-empty component."""

    area = int(np.count_nonzero(component))
    if area < 1:
        raise ValueError("component core requires a non-empty component")
    if area <= 4:
        return np.ascontiguousarray(component.copy(), dtype=np.bool_)
    distance = ndimage.distance_transform_edt(component)
    maximum = float(np.max(distance))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise FloatingPointError("component distance transform is invalid")
    core = component & (distance >= max(1.0, 0.5 * maximum))
    if not bool(core.any()):
        core = np.zeros_like(component, dtype=np.bool_)
        core.flat[int(np.argmax(distance))] = True
    return np.ascontiguousarray(core, dtype=np.bool_)


def _selected_id_map(
    id_map: np.ndarray,
    selected_ids: tuple[int, ...],
) -> np.ndarray:
    if not selected_ids:
        return np.zeros_like(id_map, dtype=np.int32)
    selected = np.asarray(selected_ids, dtype=np.int32)
    return np.ascontiguousarray(
        np.where(np.isin(id_map, selected), id_map, 0),
        dtype=np.int32,
    )


def build_irstd_error_atlas(
    *,
    current_logits: np.ndarray,
    target_mask: np.ndarray,
    ring_radius: int = 3,
    far_background_radius: int = 7,
    baseline_logits: np.ndarray | None = None,
) -> IRSTDErrorAtlas:
    """Construct Current-only maps and optional bound-Baseline advantage maps.

    ``attached_halo`` is the background portion of every prediction component
    already matched to a target.  ``detached_false_positive`` is deliberately
    restricted to background pixels: an unmatched prediction component may
    overlap a target under centroid matching, but target pixels must never be
    turned into negative gate supervision.
    """

    logits = _validated_logits(current_logits, name="current_logits")
    target = _validated_binary_mask(
        target_mask,
        name="target_mask",
        shape=logits.shape,
    )
    ready_ring_radius = _validated_radius(ring_radius, name="ring_radius")
    ready_far_radius = _validated_radius(
        far_background_radius,
        name="far_background_radius",
    )
    if ready_far_radius <= ready_ring_radius:
        raise ValueError("far_background_radius must exceed ring_radius")

    baseline: np.ndarray | None
    if baseline_logits is None:
        baseline = None
    else:
        baseline = _validated_logits(baseline_logits, name="baseline_logits")
        if baseline.shape != logits.shape:
            raise ValueError("baseline_logits must match current_logits shape")

    prediction = np.ascontiguousarray(logits > 0.0)
    match = match_components_v2(
        prediction_mask=prediction,
        target_mask=target,
    )
    target_ids = np.ascontiguousarray(match.target_id_map, dtype=np.int32)
    prediction_ids = np.ascontiguousarray(
        match.prediction_id_map,
        dtype=np.int32,
    )

    core_target = np.zeros_like(target, dtype=np.bool_)
    for component_id in np.unique(target_ids):
        ready_id = int(component_id)
        if ready_id <= 0:
            continue
        core_target |= _component_core(target_ids == ready_id)

    rescue_ids = _selected_id_map(
        target_ids,
        match.unmatched_target_ids,
    )

    attached_halo = np.zeros_like(target, dtype=np.bool_)
    for pair in match.matches:
        prediction_component = prediction_ids == int(pair.prediction_id)
        attached_halo |= prediction_component & ~target

    unmatched_prediction_mask = (
        np.isin(
            prediction_ids,
            np.asarray(match.unmatched_prediction_ids, dtype=np.int32),
        )
        if match.unmatched_prediction_ids
        else np.zeros_like(target, dtype=np.bool_)
    )
    detached_false_positive = unmatched_prediction_mask & ~target

    ring_structure = ndimage.generate_binary_structure(2, 2)
    ring_dilation = ndimage.binary_dilation(
        target,
        structure=ring_structure,
        iterations=ready_ring_radius,
    )
    far_dilation = ndimage.binary_dilation(
        target,
        structure=ring_structure,
        iterations=ready_far_radius,
    )
    outer_ring = ring_dilation & ~target
    far_background = ~far_dilation

    baseline_rescue: np.ndarray | None
    baseline_halo_advantage: np.ndarray | None
    if baseline is None:
        baseline_rescue = None
        baseline_halo_advantage = None
    else:
        baseline_prediction = baseline > 0.0
        baseline_rescue = np.ascontiguousarray(
            target & baseline_prediction & ~prediction,
            dtype=np.bool_,
        )
        baseline_halo_advantage = np.ascontiguousarray(
            ~target & prediction & ~baseline_prediction,
            dtype=np.bool_,
        )

    halo_target = attached_halo | detached_false_positive
    if baseline_halo_advantage is not None:
        halo_target |= baseline_halo_advantage

    boolean_maps = {
        "core_target": core_target,
        "attached_halo": attached_halo,
        "detached_false_positive": detached_false_positive,
        "outer_ring": outer_ring,
        "halo_target": halo_target,
        "far_background": far_background,
    }
    if not np.array_equal(target_ids > 0, target):
        raise RuntimeError("target component IDs do not reproduce target_mask")
    if bool(np.any((rescue_ids > 0) & ~target)):
        raise RuntimeError("rescue component IDs escape target support")
    if not np.array_equal(core_target & target, core_target):
        raise RuntimeError("core_target escapes target support")
    for name in (
        "attached_halo",
        "detached_false_positive",
        "outer_ring",
        "halo_target",
        "far_background",
    ):
        if bool(np.any(boolean_maps[name] & target)):
            raise RuntimeError(f"{name} overlaps target support")
    if not np.array_equal(attached_halo & halo_target, attached_halo):
        raise RuntimeError("attached_halo is not included in halo_target")
    if baseline_rescue is not None and bool(np.any(baseline_rescue & ~target)):
        raise RuntimeError("baseline_rescue escapes target support")
    if baseline_halo_advantage is not None and bool(
        np.any(baseline_halo_advantage & target)
    ):
        raise RuntimeError("baseline_halo_advantage overlaps target support")

    return IRSTDErrorAtlas(
        target_component_ids=np.ascontiguousarray(target_ids, dtype=np.int32),
        rescue_component_ids=np.ascontiguousarray(rescue_ids, dtype=np.int32),
        core_target=np.ascontiguousarray(core_target, dtype=np.bool_),
        attached_halo=np.ascontiguousarray(attached_halo, dtype=np.bool_),
        detached_false_positive=np.ascontiguousarray(
            detached_false_positive,
            dtype=np.bool_,
        ),
        outer_ring=np.ascontiguousarray(outer_ring, dtype=np.bool_),
        halo_target=np.ascontiguousarray(halo_target, dtype=np.bool_),
        far_background=np.ascontiguousarray(far_background, dtype=np.bool_),
        baseline_available=baseline is not None,
        baseline_rescue=baseline_rescue,
        baseline_halo_advantage=baseline_halo_advantage,
    )


__all__ = [
    "IRSTD_ERROR_ATLAS_VERSION",
    "IRSTDErrorAtlas",
    "build_irstd_error_atlas",
]
