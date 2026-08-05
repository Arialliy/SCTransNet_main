#!/usr/bin/env python3
"""Compare seed-42 TPD8 block-residual knockouts on three datasets.

This comparator performs no inference.  It consumes the nine required modes
(``full``, seven single-block residual knockouts, and ``all7_off``), validates
their fixed-threshold and unpadded-probability contracts, then applies the
pre-registered performance gates.  A first-round harmful-block result may
suggest exactly one tenth-mode audit, but never authorizes that unmeasured
combination or a new training run by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import compare_three_dataset_qfg_level_knockout_v1 as gate_core  # noqa: E402
from analysis import (  # noqa: E402
    analyze_three_dataset_tpd8_block_residual_knockout_v1 as analyzer,
)
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA = "sctransnet_three_dataset_tpd8_block_residual_knockout_comparison_v1/v1"
ANALYZER_SCHEMA = analyzer.SCHEMA
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLE = "best_miou"
SEED = 42
FIXED_THRESHOLD = 0.5
SWEEP_THRESHOLDS = tuple(analyzer.SWEEP_THRESHOLDS)

# Short names are only report labels.  Every intervention binding is checked
# against the analyzer's fully qualified module paths.
BLOCK_IDS = tuple(analyzer.BLOCK_IDS)
BLOCK_PATHS = tuple(analyzer.BLOCK_PATHS)
SINGLE_MODE_TO_BLOCK = {
    "e1b0_off": "E1.B0",
    "e1b1_off": "E1.B1",
    "e1b2_off": "E1.B2",
    "e1b3_off": "E1.B3",
    "e2b0_off": "E2.B0",
    "e2b1_off": "E2.B1",
    "e2b2_off": "E2.B2",
}
BLOCK_TO_SINGLE_MODE = {
    block: mode for mode, block in SINGLE_MODE_TO_BLOCK.items()
}
SINGLE_MODES = tuple(SINGLE_MODE_TO_BLOCK)
MODES = tuple(analyzer.PUBLIC_MODES)
_require_mode_order = ("full", *SINGLE_MODES, "all7_off")
if MODES != _require_mode_order:  # fail closed at import if contracts drift
    raise RuntimeError("TPD8 analyzer mode order differs from the comparator")

REQUIRED_MPRS_TERMS = tuple(analyzer._TERM_NAMES)

EARLY_ONLY_OFF_SETS = {
    "early_0": (
        "E1.B1",
        "E1.B2",
        "E1.B3",
        "E2.B1",
        "E2.B2",
    ),
    "early_1": ("E1.B2", "E1.B3", "E2.B2"),
    "early_2": ("E1.B3",),
}

PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD = 1e-7
PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD = 1e-8

# Reuse the exact QFG directional thresholds and zero-denominator primitive.
SAFE_COUNT_DELTA_MIN_EXCLUSIVE = gate_core.SAFE_COUNT_DELTA_MIN_EXCLUSIVE
SAFE_IOU_DELTA_MIN_EXCLUSIVE = gate_core.SAFE_IOU_DELTA_MIN_EXCLUSIVE
SAFE_FP_REDUCTION_MIN_EXCLUSIVE = gate_core.SAFE_FP_REDUCTION_MIN_EXCLUSIVE
MATERIAL_COUNT_DELTA_MINIMUM = gate_core.MATERIAL_COUNT_DELTA_MINIMUM
MATERIAL_IOU_DELTA_MINIMUM = gate_core.MATERIAL_IOU_DELTA_MINIMUM
MATERIAL_FP_REDUCTION_MINIMUM = gate_core.MATERIAL_FP_REDUCTION_MINIMUM
SEVERE_COUNT_DELTA_MAXIMUM = gate_core.SEVERE_COUNT_DELTA_MAXIMUM
SEVERE_IOU_DELTA_MAXIMUM = gate_core.SEVERE_IOU_DELTA_MAXIMUM
SEVERE_FP_REDUCTION_MAXIMUM = gate_core.SEVERE_FP_REDUCTION_MAXIMUM

DECISION_LOCAL_AUDIT = "TPD_LOCAL_CANDIDATE_SUGGESTED_NOT_AUTHORIZED"
DECISION_BLOCK_SELECTIVE = "DESIGN_TPD_BLOCK_SELECTIVE_CANDIDATE"
DECISION_RESIDUAL_OFF = "DESIGN_TPD_RESIDUAL_OFF_CANDIDATE"
DECISION_KEEP = "FREEZE_TPD8_RESIDUAL_FULL"
DECISION_UNSUPPORTED = "TPD_RESIDUAL_CONTRIBUTION_UNSUPPORTED_CONSIDER_SIMPLIFY"
DECISION_INCONCLUSIVE = "TPD_INCONCLUSIVE_NO_FORMULA_CHANGE"

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "three_dataset_tpd8_block_residual_knockout_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison" / "best_miou_seed42"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    ready = float(value)
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _is_sha256(value: Any) -> bool:
    return gate_core._is_sha256(value)


def file_sha256(path: Path) -> str:
    return gate_core.file_sha256(path)


def _expected_selected_blocks(mode: str) -> list[str]:
    """Return short report labels, not analyzer intervention paths."""

    if mode == "full":
        return []
    if mode == "all7_off":
        return list(BLOCK_IDS)
    return [SINGLE_MODE_TO_BLOCK[mode]]


def _expected_selected_block_paths(mode: str) -> list[str]:
    return list(analyzer.MODE_TO_BLOCK_PATHS[mode])


def _extract_point(mode_payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    point = _mapping(mode_payload.get("fixed_threshold_0_5"), f"{label}.fixed")
    threshold = _finite_float(point.get("threshold"), f"{label}.threshold")
    _require(threshold == FIXED_THRESHOLD, f"{label} threshold differs")
    target_count = _nonnegative_int(point.get("target_count"), f"{label}.target_count")
    tiny_target_count = _nonnegative_int(
        point.get("tiny_target_count"), f"{label}.tiny_target_count"
    )
    matched_target_count = _nonnegative_int(
        point.get("matched_target_count"), f"{label}.matched_target_count"
    )
    matched_tiny_target_count = _nonnegative_int(
        point.get("matched_tiny_target_count"),
        f"{label}.matched_tiny_target_count",
    )
    _require(matched_target_count <= target_count, f"{label} matched targets exceed total")
    _require(
        matched_tiny_target_count <= tiny_target_count,
        f"{label} matched tiny targets exceed total",
    )
    valid_pixel_count = _nonnegative_int(
        point.get("valid_pixel_count"), f"{label}.valid_pixel_count"
    )
    _require(valid_pixel_count > 0, f"{label}.valid_pixel_count must be positive")
    ready = {
        "threshold": threshold,
        "target_count": target_count,
        "tiny_target_count": tiny_target_count,
        "matched_target_count": matched_target_count,
        "matched_tiny_target_count": matched_tiny_target_count,
        "miou": _finite_float(point.get("miou"), f"{label}.miou"),
        "niou": _finite_float(point.get("niou"), f"{label}.niou"),
        "component_false_positive_pixels": _nonnegative_int(
            point.get("unmatched_predicted_pixels"),
            f"{label}.unmatched_predicted_pixels",
        ),
        "background_false_positive_pixels": _nonnegative_int(
            point.get("false_positive_pixels"),
            f"{label}.false_positive_pixels",
        ),
        "valid_pixel_count": valid_pixel_count,
    }
    for metric in ("miou", "niou"):
        _require(0.0 <= ready[metric] <= 1.0, f"{label}.{metric} outside [0,1]")
    _require(
        ready["component_false_positive_pixels"] <= valid_pixel_count,
        f"{label} component FP exceeds valid pixels",
    )
    _require(
        ready["background_false_positive_pixels"] <= valid_pixel_count,
        f"{label} background FP exceeds valid pixels",
    )
    return ready


def _extract_probability_difference(
    mode_payload: Mapping[str, Any],
    label: str,
    *,
    valid_pixel_count: int,
) -> dict[str, Any]:
    raw = _mapping(
        mode_payload.get("probability_difference_to_full"),
        f"{label}.probability_difference_to_full",
    )
    max_abs = _finite_float(raw.get("max_abs"), f"{label}.max_abs")
    absolute_sum = _finite_float(
        raw.get("absolute_difference_sum"),
        f"{label}.absolute_difference_sum",
    )
    reported_mean = _finite_float(raw.get("mean_abs"), f"{label}.mean_abs")
    _require(
        max_abs >= 0.0 and absolute_sum >= 0.0 and reported_mean >= 0.0,
        f"{label} negative probability difference",
    )
    element_count = _nonnegative_int(raw.get("element_count"), f"{label}.element_count")
    _require(
        element_count == valid_pixel_count,
        f"{label}.element_count differs from unpadded valid_pixel_count",
    )
    _require(
        raw.get("scope") == "all_original_unpadded_test_pixels",
        f"{label} probability scope differs",
    )
    mean_abs = absolute_sum / element_count
    _require(
        reported_mean == mean_abs,
        f"{label}.mean_abs differs from absolute_difference_sum/element_count",
    )
    _require(mean_abs <= max_abs, f"{label}.mean_abs exceeds max_abs")
    _require(
        _finite_float(
            raw.get("equivalence_max_abs_threshold"),
            f"{label}.equivalence_max_abs_threshold",
        )
        == PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD,
        f"{label} max-abs threshold differs",
    )
    _require(
        _finite_float(
            raw.get("equivalence_mean_abs_threshold"),
            f"{label}.equivalence_mean_abs_threshold",
        )
        == PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD,
        f"{label} mean-abs threshold differs",
    )
    functionally_different = bool(
        max_abs > PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD
        or mean_abs > PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD
    )
    reported_different = _strict_bool(
        raw.get("functionally_different"), f"{label}.functionally_different"
    )
    reported_equivalent = _strict_bool(raw.get("equivalent"), f"{label}.equivalent")
    _require(
        reported_different == functionally_different,
        f"{label} functionally_different conflicts with frozen thresholds",
    )
    _require(
        reported_equivalent is (not functionally_different),
        f"{label} equivalent conflicts with frozen thresholds",
    )
    return {
        "max_abs": max_abs,
        "absolute_difference_sum": absolute_sum,
        "mean_abs": mean_abs,
        "element_count": element_count,
        "equivalent": not functionally_different,
        "functionally_different": functionally_different,
    }


def _validate_descriptive_sweep(
    mode_payload: Mapping[str, Any],
    label: str,
) -> None:
    """Lock the two registered points while keeping them out of decisions."""

    _require(
        mode_payload.get("sweep_thresholds") == list(SWEEP_THRESHOLDS),
        f"{label} sweep declaration differs",
    )
    threshold_roles = _mapping(
        mode_payload.get("threshold_roles"), f"{label}.threshold_roles"
    )
    expected_roles = {
        "checkpoint_selection_threshold": FIXED_THRESHOLD,
        "global_lambda_selection_threshold": FIXED_THRESHOLD,
        "main_table_threshold": FIXED_THRESHOLD,
        "descriptive_sweep_only": True,
        "descriptive_sweep_contains_threshold_1_0": True,
        "threshold_1_0_semantics": "empty_prediction_pd0_fa0",
        "sweep_reselects_checkpoint": False,
        "sweep_reselects_global_lambda": False,
    }
    _require(dict(threshold_roles) == expected_roles, f"{label} threshold roles differ")
    descriptive = _mapping(
        mode_payload.get("descriptive_pd_fa"), f"{label}.descriptive_pd_fa"
    )
    _require(
        descriptive.get("selection_effect") == "none",
        f"{label} descriptive sweep selection effect differs",
    )
    provenance = _mapping(
        descriptive.get("threshold_provenance"),
        f"{label}.descriptive_pd_fa.threshold_provenance",
    )
    _require(
        provenance.get("provided_thresholds") is True
        and provenance.get("closed_probability_interval") is True,
        f"{label} threshold provenance differs",
    )
    _mapping(
        descriptive.get("best_points_under_fa_budget"),
        f"{label}.descriptive_pd_fa.best_points_under_fa_budget",
    )
    frontier = descriptive.get("pareto_frontier")
    _require(isinstance(frontier, list), f"{label} Pareto frontier must be a list")
    raw_points = descriptive.get("points")
    _require(
        isinstance(raw_points, list) and len(raw_points) == len(SWEEP_THRESHOLDS),
        f"{label} descriptive sweep must contain exactly two points",
    )
    points = [
        _mapping(point, f"{label}.descriptive_pd_fa.points[{index}]")
        for index, point in enumerate(raw_points)
    ]
    thresholds = [
        _finite_float(point.get("threshold"), f"{label}.points[{index}].threshold")
        for index, point in enumerate(points)
    ]
    _require(
        thresholds == list(SWEEP_THRESHOLDS),
        f"{label} descriptive thresholds are not exactly [0.5, 1.0]",
    )

    fixed = _mapping(mode_payload.get("fixed_threshold_0_5"), f"{label}.fixed")
    computed_fixed_empty = analyzer.core._point_is_empty(points[0])
    if "selected_point_is_empty" in points[0]:
        fixed_empty_flag = _strict_bool(
            points[0].get("selected_point_is_empty"),
            f"{label} threshold-0.5 empty-point flag",
        )
        _require(
            fixed_empty_flag == computed_fixed_empty,
            f"{label} threshold-0.5 empty-point flag differs",
        )
    _require(
        points[1].get("selected_point_is_empty") is True,
        f"{label} threshold-1.0 empty-point flag differs",
    )
    fixed_without_background_fp = {
        key: value for key, value in fixed.items() if key != "false_positive_pixels"
    }
    sweep_fixed_without_empty_flag = {
        key: value
        for key, value in points[0].items()
        if key != "selected_point_is_empty"
    }
    _require(
        sweep_fixed_without_empty_flag == fixed_without_background_fp,
        f"{label} sweep-0.5 point differs from fixed-0.5",
    )
    empty = points[1]
    for field in ("pd", "fa"):
        _require(
            _finite_float(empty.get(field), f"{label}.empty.{field}") == 0.0,
            f"{label} threshold-1.0 {field} is not zero",
        )
    for field in (
        "matched_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "unmatched_predicted_pixels",
    ):
        _require(
            _nonnegative_int(empty.get(field), f"{label}.empty.{field}") == 0,
            f"{label} threshold-1.0 {field} is not zero",
        )
    for field in ("target_count", "tiny_target_count", "valid_pixel_count"):
        _require(
            empty.get(field) == fixed.get(field),
            f"{label} threshold-1.0 {field} differs from fixed",
        )


def _validate_source_scale_records(payload: Mapping[str, Any]) -> None:
    records = payload.get("source_saliency_scale_records")
    _require(
        isinstance(records, list) and len(records) == len(BLOCK_PATHS),
        "source saliency-scale records must contain all seven blocks",
    )
    for index, (record_value, path) in enumerate(zip(records, BLOCK_PATHS)):
        record = _mapping(record_value, f"source_saliency_scale_records[{index}]")
        expected_channels = 32 if index < 4 else 64
        _require(record.get("block_index_zero_based") == index, "scale record index differs")
        _require(record.get("block_path") == path, "scale record path differs")
        _require(record.get("channels") == expected_channels, "scale record channels differ")
        element_count = _nonnegative_int(
            record.get("element_count"), f"scale record {path}.element_count"
        )
        nonzero_count = _nonnegative_int(
            record.get("nonzero_count"), f"scale record {path}.nonzero_count"
        )
        _require(element_count == expected_channels, "scale vector length differs")
        _require(nonzero_count <= element_count, "scale nonzero count exceeds length")
        _require(isinstance(record.get("dtype"), str), "scale dtype is missing")
        _strict_bool(record.get("requires_grad"), f"scale record {path}.requires_grad")
        for field in (
            "parameter_minimum",
            "parameter_maximum",
            "parameter_mean",
            "parameter_rms",
            "effective_tanh_minimum",
            "effective_tanh_maximum",
            "effective_tanh_mean",
            "effective_tanh_rms",
        ):
            value = _finite_float(record.get(field), f"scale record {path}.{field}")
            if field.endswith("rms"):
                _require(value >= 0.0, f"scale record {path}.{field} is negative")
            if field.startswith("effective_tanh_"):
                _require(-1.0 <= value <= 1.0, f"scale record {path}.{field} outside tanh range")


def _validate_full_mprs_statistics(value: Any) -> None:
    mechanism = _mapping(value, "modes.full.full_mprs_statistics")
    _require(mechanism.get("schema") == analyzer.MECHANISM_SCHEMA, "MPRS schema differs")
    _require(
        mechanism.get("production_output_policy")
        == "return_original_forward_output_unchanged",
        "MPRS production-output policy differs",
    )
    _require(
        mechanism.get("diagnostic_execution")
        == "in_production_aligned_mprs_terms_capture_plus_no_grad_headroom",
        "MPRS diagnostic execution differs",
    )
    _require(
        mechanism.get("branch_projection_recomputed_for_statistics") is False,
        "MPRS branch projections were recomputed",
    )
    _require(mechanism.get("feature_cache_written") is False, "MPRS feature cache was written")
    batch_count = _nonnegative_int(mechanism.get("batch_count"), "MPRS batch_count")
    _require(batch_count > 0, "MPRS batch_count must be positive")
    _require(
        _nonnegative_int(mechanism.get("aborted_batch_count"), "MPRS aborted_batch_count")
        == 0,
        "MPRS captured an aborted batch",
    )
    _require(mechanism.get("block_count") == len(BLOCK_PATHS), "MPRS block_count differs")
    _require(mechanism.get("block_order") == list(BLOCK_PATHS), "MPRS block order differs")
    _require(
        mechanism.get("temporary_forward_wrappers_restored") is True,
        "MPRS temporary wrappers were not restored",
    )
    _require(
        mechanism.get("target_projection") == "adaptive_max_pool2d_binary_presence",
        "MPRS target projection differs",
    )
    _require(
        mechanism.get("valid_projection")
        == "adaptive_max_pool2d_any_original_support",
        "MPRS valid projection differs",
    )
    _require(
        mechanism.get("background_region")
        == "pooled_valid_and_not_pooled_target",
        "MPRS background-region definition differs",
    )
    blocks = mechanism.get("blocks")
    _require(
        isinstance(blocks, list) and len(blocks) == len(BLOCK_PATHS),
        "MPRS diagnostics must contain all seven block rows",
    )
    for index, (row_value, path) in enumerate(zip(blocks, BLOCK_PATHS)):
        row = _mapping(row_value, f"MPRS.blocks[{index}]")
        expected_embedding = "embeddings_1" if index < 4 else "embeddings_2"
        expected_local_index = index if index < 4 else index - 4
        expected_channels = 32 if index < 4 else 64
        _require(row.get("block_index_zero_based") == index, f"MPRS block index differs: {path}")
        _require(row.get("block_path") == path, f"MPRS block path differs: {path}")
        _require(row.get("embedding") == expected_embedding, f"MPRS embedding differs: {path}")
        _require(
            row.get("embedding_block_index_zero_based") == expected_local_index,
            f"MPRS local block index differs: {path}",
        )
        _require(row.get("channels") == expected_channels, f"MPRS channels differ: {path}")
        _require(isinstance(row.get("activation"), str) and bool(row.get("activation")), f"MPRS activation missing: {path}")
        _require(row.get("forward_call_count") == batch_count, f"MPRS call count differs: {path}")
        shapes = row.get("observed_output_shapes_chw")
        _require(isinstance(shapes, list) and bool(shapes), f"MPRS shapes missing: {path}")
        for shape in shapes:
            _require(
                isinstance(shape, list)
                and len(shape) == 3
                and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape),
                f"MPRS output shape is malformed: {path}",
            )
            _require(shape[0] == expected_channels, f"MPRS output channels differ: {path}")
        rms = _mapping(row.get("rms_statistics"), f"MPRS {path}.rms_statistics")
        _require(set(rms) == set(REQUIRED_MPRS_TERMS), f"MPRS term set differs: {path}")
        for term in REQUIRED_MPRS_TERMS:
            summary = _mapping(rms.get(term), f"MPRS {path}.{term}")
            element_count = _nonnegative_int(
                summary.get("element_count"), f"MPRS {path}.{term}.element_count"
            )
            _require(element_count > 0, f"MPRS {path}.{term} has no elements")
            square_sum = _finite_float(summary.get("square_sum"), f"MPRS {path}.{term}.square_sum")
            reported_rms = _finite_float(summary.get("rms"), f"MPRS {path}.{term}.rms")
            _require(square_sum >= 0.0 and reported_rms >= 0.0, f"MPRS {path}.{term} is negative")
            _require(
                reported_rms == math.sqrt(square_sum / element_count),
                f"MPRS {path}.{term} RMS identity differs",
            )
        target_rms = rms["target_residual_R"].get("rms")
        background_rms = rms["background_residual_R"].get("rms")
        expected_margin = (
            None
            if target_rms is None or background_rms is None
            else float(target_rms) - float(background_rms)
        )
        reported_margin = row.get("target_minus_background_residual_rms")
        if expected_margin is None:
            _require(reported_margin is None, f"MPRS residual margin differs: {path}")
        else:
            _require(
                _finite_float(reported_margin, f"MPRS residual margin {path}")
                == expected_margin,
                f"MPRS residual margin differs: {path}",
            )


def _validate_knockout_audit(
    raw_mode: Mapping[str, Any],
    mode: str,
    top_restoration: Mapping[str, Any],
) -> None:
    label = f"modes.{mode}.saliency_scale_knockout"
    knockout = _mapping(raw_mode.get("saliency_scale_knockout"), label)
    _require(knockout.get("schema") == analyzer.KNOCKOUT_SCHEMA, f"{label}.schema differs")
    _require(knockout.get("public_mode") == mode, f"{label}.public_mode differs")
    selected_paths = _expected_selected_block_paths(mode)
    selected_indices = [BLOCK_PATHS.index(path) for path in selected_paths]
    _require(knockout.get("selected_block_paths") == selected_paths, f"{label} selected paths differ")
    _require(
        knockout.get("selected_block_indices_zero_based") == selected_indices,
        f"{label} selected indices differ",
    )
    _require(knockout.get("derived_checkpoint_written") is False, f"{label} wrote a checkpoint")
    _require(knockout.get("diagnostic_only") is (mode != "full"), f"{label}.diagnostic_only differs")
    selected_vectors = knockout.get("selected_vectors")
    _require(
        isinstance(selected_vectors, list)
        and len(selected_vectors) == len(selected_paths),
        f"{label} selected-vector audit differs",
    )
    source_nonzero_total = 0
    active_nonzero_total = 0
    for vector_index, (record_value, block_path, block_index) in enumerate(
        zip(selected_vectors, selected_paths, selected_indices)
    ):
        record = _mapping(record_value, f"{label}.selected_vectors[{vector_index}]")
        expected_channels = 32 if block_index < 4 else 64
        _require(record.get("block_index_zero_based") == block_index, f"{label} vector index differs")
        _require(record.get("block_id") == BLOCK_IDS[block_index], f"{label} vector block ID differs")
        _require(record.get("block_path") == block_path, f"{label} vector block path differs")
        _require(record.get("element_count") == expected_channels, f"{label} vector length differs")
        source_nonzero = _nonnegative_int(
            record.get("source_nonzero_count"), f"{label} vector source_nonzero_count"
        )
        active_nonzero = _nonnegative_int(
            record.get("active_nonzero_count"), f"{label} vector active_nonzero_count"
        )
        _require(source_nonzero <= expected_channels, f"{label} source nonzero count exceeds length")
        _require(active_nonzero == 0, f"{label} selected vector is not exactly zero")
        source_nonzero_total += source_nonzero
        active_nonzero_total += active_nonzero
    _require(
        knockout.get("selected_source_nonzero_count") == source_nonzero_total,
        f"{label} source nonzero aggregate differs",
    )
    _require(
        knockout.get("selected_active_nonzero_count") == active_nonzero_total == 0,
        f"{label} active nonzero aggregate differs",
    )
    if mode != "full":
        _require(source_nonzero_total > 0, f"{label} selected source vectors are already zero")
    for family in ("saliency_scale", "model_state"):
        source = knockout.get(f"source_{family}_sha256")
        active = knockout.get(f"active_{family}_sha256")
        restored = knockout.get(f"restored_{family}_sha256")
        _require(_is_sha256(source), f"{label} source {family} SHA is invalid")
        _require(_is_sha256(active), f"{label} active {family} SHA is invalid")
        _require(_is_sha256(restored), f"{label} restored {family} SHA is invalid")
        _require(restored == source, f"{label} restored {family} SHA differs")
        _require(
            knockout.get(f"{family}_restored_exactly") is True,
            f"{label} {family} was not restored exactly",
        )
        top_prefix = "saliency_scale" if family == "saliency_scale" else "model_state"
        _require(
            source == top_restoration.get(f"{top_prefix}_sha256_before"),
            f"{label} source {family} differs from top restoration",
        )
        if mode == "full":
            _require(active == source, f"{label} full active {family} changed")
        else:
            _require(active != source, f"{label} off active {family} did not change")

    mode_restoration = _mapping(
        raw_mode.get("restoration_audit"), f"modes.{mode}.restoration_audit"
    )
    for family in ("saliency_scale", "model_state"):
        expected = mode_restoration.get(f"{family}_sha256_expected")
        after = mode_restoration.get(f"{family}_sha256_after_mode")
        _require(_is_sha256(expected) and _is_sha256(after), f"modes.{mode} {family} restoration SHA is invalid")
        _require(expected == after, f"modes.{mode} {family} was not restored")
        _require(mode_restoration.get(f"{family}_unchanged") is True, f"modes.{mode} {family} changed")
        _require(
            expected == top_restoration.get(f"{family}_sha256_before"),
            f"modes.{mode} expected {family} differs from top restoration",
        )


def validate_analyzer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate provenance, nine modes, metric fields, and restoration locks."""

    serialized_modes = _mapping(payload.get("modes"), "modes")
    _require(
        set(serialized_modes) == set(MODES),
        "serialized analyzer mode order differs from frozen nine modes",
    )
    _require(
        payload.get("mode_order") == list(MODES),
        "serialized analyzer mode_order differs",
    )
    # Analyzer artifacts are intentionally written with sort_keys=True, so
    # JSON round-tripping alphabetizes the mapping even though mode_order is
    # canonical.  Reconstruct only the mapping order for the analyzer's
    # in-memory validator; never alter the source payload or its values.
    analyzer_view = dict(payload)
    analyzer_view["modes"] = {
        mode: serialized_modes[mode] for mode in MODES
    }
    analyzer.validate_output_payload(analyzer_view)
    _require(payload.get("schema") == ANALYZER_SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer result is incomplete")
    dataset = payload.get("dataset")
    _require(dataset in DATASETS, f"unsupported dataset: {dataset!r}")
    _require(payload.get("method") == analyzer.REFERENCE_METHOD, "analyzer method differs")
    _require(
        payload.get("training_model_method") == analyzer.TRAINING_MODEL_METHOD,
        "training model method differs",
    )
    _require(payload.get("checkpoint_role") == CHECKPOINT_ROLE, "checkpoint role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("test_selected") is True, "result is not test-selected")
    _require(payload.get("selection_is_optimistic") is True, "selection policy differs")
    _require(
        payload.get("evaluation_protocol") == analyzer.EVALUATION_PROTOCOL,
        "evaluation protocol differs",
    )
    _require(payload.get("fixed_threshold") == FIXED_THRESHOLD, "top fixed threshold differs")
    _require(payload.get("sweep_thresholds") == list(SWEEP_THRESHOLDS), "top sweep differs")
    _require(payload.get("mode_order") == list(MODES), "canonical mode order differs")
    _require(payload.get("block_order") == list(BLOCK_PATHS), "canonical block order differs")
    replay = _mapping(payload.get("reference_replay_audit"), "reference_replay_audit")
    _require(replay.get("passed") is True, "reference replay did not pass")
    _require(
        replay.get("comparison")
        == "full_mode_fixed_threshold_0_5_vs_existing_best_miou",
        "reference replay comparison differs",
    )
    _require(
        isinstance(replay.get("compared"), Mapping) and bool(replay.get("compared")),
        "reference replay compared set is empty",
    )

    checkpoint = _mapping(payload.get("checkpoint_binding"), "checkpoint_binding")
    checkpoint_file = _mapping(checkpoint.get("checkpoint"), "checkpoint_binding.checkpoint")
    _require(_is_sha256(checkpoint_file.get("sha256")), "checkpoint SHA is invalid")
    _require(checkpoint_file.get("role") == CHECKPOINT_ROLE, "checkpoint role binding differs")
    checkpoint_protocol = _mapping(checkpoint.get("protocol"), "checkpoint_binding.protocol")
    _require(
        _is_sha256(checkpoint_protocol.get("payload_sha256")),
        "checkpoint protocol SHA is invalid",
    )
    data = _mapping(payload.get("data"), "data")
    manifest = _mapping(data.get("protocol_manifest"), "data.protocol_manifest")
    _require(_is_sha256(manifest.get("sha256")), "manifest SHA is invalid")
    _require(data.get("split") == "img_idx/test", "data split differs")
    _require(
        _is_sha256(data.get("inference_order_newline_sha256")),
        "inference-order SHA is invalid",
    )
    reference = _mapping(payload.get("reference_reuse"), "reference_reuse")
    _require(_is_sha256(reference.get("sha256")), "reference SHA is invalid")
    sources = _mapping(payload.get("source_sha256"), "source_sha256")
    _require(bool(sources), "analyzer source SHA map is empty")
    _require(all(_is_sha256(value) for value in sources.values()), "analyzer source SHA map is invalid")

    restoration = _mapping(payload.get("restoration_audit"), "restoration_audit")
    gate_core._validate_sha_pair(restoration, "model_state")
    gate_core._validate_sha_pair(restoration, "saliency_scale")
    _require(
        restoration.get("temporary_forward_wrappers_restored") is True,
        "temporary forward wrappers were not restored",
    )
    intervention = _mapping(payload.get("intervention_contract"), "intervention_contract")
    _require(
        intervention.get("parameter")
        == "selected TPD8 block saliency_scale vector"
        and intervention.get("active_value") == "exact_zero"
        and intervention.get("weights_saved_valuewise_and_restored") is True
        and intervention.get("phase_compress_modified") is False
        and intervention.get("other_model_state_modified") is False
        and intervention.get("derived_checkpoint_written") is False,
        "saliency-scale exact-zero intervention contract differs",
    )
    _validate_source_scale_records(payload)

    modes = _mapping(payload.get("modes"), "modes")
    _require(set(modes) == set(MODES), "analyzer mode set differs from frozen nine modes")
    extracted_points: dict[str, dict[str, Any]] = {}
    raw_modes: dict[str, Mapping[str, Any]] = {}
    for mode in MODES:
        raw_mode = _mapping(modes.get(mode), f"modes.{mode}")
        raw_modes[mode] = raw_mode
        _require(raw_mode.get("public_mode") == mode, f"modes.{mode}.public_mode differs")
        selected = raw_mode.get("knockout_block_paths")
        _require(
            isinstance(selected, list) and all(isinstance(value, str) for value in selected),
            f"modes.{mode} knockout paths are malformed",
        )
        _require(
            selected == _expected_selected_block_paths(mode),
            f"modes.{mode} selected wrong blocks",
        )
        _validate_descriptive_sweep(raw_mode, f"modes.{mode}")
        _validate_knockout_audit(raw_mode, mode, restoration)
        if mode == "full":
            _validate_full_mprs_statistics(raw_mode.get("full_mprs_statistics"))
        else:
            _require(
                raw_mode.get("full_mprs_statistics") is None,
                f"modes.{mode} mechanism diagnostics must be null",
            )
        extracted_points[mode] = _extract_point(raw_mode, f"modes.{mode}")

    full_point = extracted_points["full"]
    invariant_fields = ("target_count", "tiny_target_count", "valid_pixel_count")
    normalized_modes: dict[str, Any] = {}
    for mode in MODES:
        point = extracted_points[mode]
        for field in invariant_fields:
            _require(point[field] == full_point[field], f"modes.{mode}.{field} differs from full")
        probability = _extract_probability_difference(
            raw_modes[mode],
            f"modes.{mode}",
            valid_pixel_count=full_point["valid_pixel_count"],
        )
        if mode == "full":
            _require(
                probability["max_abs"] == 0.0
                and probability["absolute_difference_sum"] == 0.0
                and probability["mean_abs"] == 0.0,
                "full probability self-difference must be exactly zero",
            )
        normalized_modes[mode] = {
            "fixed_threshold_0_5": point,
            "probability_difference_to_full": probability,
        }
    return {
        "dataset": dataset,
        "checkpoint_sha256": checkpoint_file["sha256"],
        "modes": normalized_modes,
    }


def compare_direction(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the exact QFG count/IoU/FP directional gate."""

    return gate_core.compare_direction(candidate, reference)


def _aggregate_direction(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return gate_core._aggregate_direction(rows)


def _validate_input_bindings(
    input_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    _require(set(input_bindings) == set(DATASETS), "input SHA bindings require three datasets")
    ready: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        binding = _mapping(input_bindings[dataset], f"input_bindings.{dataset}")
        path = binding.get("path")
        sha256 = binding.get("sha256")
        _require(isinstance(path, str) and bool(path), f"invalid input path: {dataset}")
        _require(_is_sha256(sha256), f"invalid input SHA: {dataset}")
        ready[dataset] = {"path": path, "sha256": str(sha256)}
    return ready


def _suggest_local_candidate(persistent_modes: Sequence[str]) -> dict[str, Any]:
    harmful_blocks = tuple(
        block for block in BLOCK_IDS if BLOCK_TO_SINGLE_MODE[block] in persistent_modes
    )
    harmful_set = set(harmful_blocks)
    suggestion_type: str
    suggestion_name: str
    selected: tuple[str, ...]
    if len(harmful_blocks) == 1:
        suggestion_type = "single_block_residual_off"
        suggestion_name = BLOCK_TO_SINGLE_MODE[harmful_blocks[0]]
        selected = harmful_blocks
    else:
        early_match = next(
            (
                name
                for name, blocks in EARLY_ONLY_OFF_SETS.items()
                if set(blocks) == harmful_set
            ),
            None,
        )
        if early_match is not None:
            suggestion_type = "early_only_depth_mask"
            suggestion_name = early_match
            selected = EARLY_ONLY_OFF_SETS[early_match]
        else:
            suggestion_type = "harmful_block_union_off"
            suggestion_name = "harmful_union_off"
            selected = harmful_blocks
    existing_sets = {
        mode: tuple(_expected_selected_blocks(mode)) for mode in MODES
    }
    matching_existing = next(
        (mode for mode, blocks in existing_sets.items() if tuple(blocks) == tuple(selected)),
        None,
    )
    single_block_already_evaluated = len(harmful_blocks) == 1
    return {
        "triggered": True,
        "suggestion_type": suggestion_type,
        "suggested_mode_name": suggestion_name,
        "selected_off_block_ids": list(selected),
        "matching_required_mode_if_any": matching_existing,
        "requires_new_tenth_mode": matching_existing is None,
        "fixed_weight_combination_gate_evaluated": single_block_already_evaluated,
        "development_training_authorized": single_block_already_evaluated,
        "reason": (
            "single_block_mode_is_the_complete_measured_candidate"
            if single_block_already_evaluated
            else "multi_block_combination_not_measured_in_first_round"
        ),
    }


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    input_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Apply the frozen first-round nine-mode priority order."""

    _require(set(payloads) == set(DATASETS), "comparison requires exactly three datasets")
    bindings = _validate_input_bindings(input_bindings)
    normalized: dict[str, Any] = {}
    for dataset in DATASETS:
        one = validate_analyzer_payload(payloads[dataset])
        _require(one["dataset"] == dataset, f"input binding differs: {dataset}")
        normalized[dataset] = one

    per_dataset: dict[str, Any] = {}
    forward_rows: dict[str, dict[str, Any]] = {mode: {} for mode in MODES[1:]}
    reverse_rows: dict[str, dict[str, Any]] = {mode: {} for mode in MODES[1:]}
    for dataset in DATASETS:
        modes = normalized[dataset]["modes"]
        full = modes["full"]["fixed_threshold_0_5"]
        rows: dict[str, Any] = {}
        for mode in MODES[1:]:
            off = modes[mode]["fixed_threshold_0_5"]
            off_vs_full = compare_direction(off, full)
            full_vs_off = compare_direction(full, off)
            forward_rows[mode][dataset] = off_vs_full
            reverse_rows[mode][dataset] = full_vs_off
            rows[mode] = {
                "off_vs_full": off_vs_full,
                "full_vs_off": full_vs_off,
                "probability_difference_to_full": modes[mode]["probability_difference_to_full"],
            }
        per_dataset[dataset] = {
            "checkpoint_sha256": normalized[dataset]["checkpoint_sha256"],
            "modes": rows,
        }

    block_aggregates: dict[str, Any] = {}
    persistent_modes: list[str] = []
    for mode in SINGLE_MODES:
        forward = _aggregate_direction(forward_rows[mode])
        reverse = _aggregate_direction(reverse_rows[mode])
        harmful = bool(forward["cross_dataset_safe_material_improvement"])
        if harmful:
            persistent_modes.append(mode)
        block_aggregates[mode] = {
            "block_id": SINGLE_MODE_TO_BLOCK[mode],
            "off_vs_full": forward,
            "full_vs_off": reverse,
            "persistent_harmful_block": harmful,
            "functionally_different_datasets": [
                dataset
                for dataset in DATASETS
                if normalized[dataset]["modes"][mode]["probability_difference_to_full"]["functionally_different"]
            ],
        }

    all_off_forward = _aggregate_direction(forward_rows["all7_off"])
    full_vs_all_off = _aggregate_direction(reverse_rows["all7_off"])
    equivalent_datasets = [
        dataset
        for dataset in DATASETS
        if normalized[dataset]["modes"]["all7_off"]["probability_difference_to_full"]["equivalent"]
    ]
    equivalent_all_three = len(equivalent_datasets) == len(DATASETS)
    any_full_safe_material = any(
        reverse_rows["all7_off"][dataset]["safe_material_improvement"]
        for dataset in DATASETS
    )
    unsupported_gate = bool(
        equivalent_all_three
        and not persistent_modes
        and not any_full_safe_material
        and not all_off_forward["cross_dataset_safe_material_improvement"]
    )
    all_blocks_persistent = len(persistent_modes) == len(SINGLE_MODES)
    local_suggestion = (
        _suggest_local_candidate(persistent_modes)
        if persistent_modes and not all_blocks_persistent
        else {
            "triggered": False,
            "fixed_weight_combination_gate_evaluated": all_blocks_persistent,
            "development_training_authorized": False,
            "resolved_by_required_all7_off": all_blocks_persistent,
            "matching_required_mode_if_any": (
                "all7_off" if all_blocks_persistent else None
            ),
            "requires_new_tenth_mode": False,
        }
    )

    if len(persistent_modes) == 1:
        decision = DECISION_BLOCK_SELECTIVE
    elif persistent_modes and not all_blocks_persistent:
        decision = DECISION_LOCAL_AUDIT
    elif all_off_forward["cross_dataset_safe_material_improvement"]:
        decision = DECISION_RESIDUAL_OFF
    elif full_vs_all_off["cross_dataset_safe_material_improvement"]:
        decision = DECISION_KEEP
    elif unsupported_gate:
        decision = DECISION_UNSUPPORTED
    else:
        decision = DECISION_INCONCLUSIVE

    persistent_blocks = [SINGLE_MODE_TO_BLOCK[mode] for mode in persistent_modes]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": decision,
        "checkpoint_role": CHECKPOINT_ROLE,
        "seed": SEED,
        "test_selected": True,
        "fixed_threshold": FIXED_THRESHOLD,
        "datasets": list(DATASETS),
        "required_mode_count": len(MODES),
        "required_modes": list(MODES),
        "persistent_harmful_block_modes": persistent_modes,
        "persistent_harmful_block_ids": persistent_blocks,
        "local_candidate_suggestion": local_suggestion,
        "tpd_local_candidate_training_authorized": (
            decision == DECISION_BLOCK_SELECTIVE
        ),
        "tpd_residual_off_candidate_authorized": decision == DECISION_RESIDUAL_OFF,
        "tpd_residual_performance_contribution_supported": bool(
            full_vs_all_off["cross_dataset_safe_material_improvement"]
        ),
        "tpd_full_architecture_contribution_supported": False,
        "tpd_full_architecture_contribution_scope": (
            "not_evaluated_all7_off_retains_keep_spd_path"
        ),
        "tpd_residual_functional_contribution_unsupported": unsupported_gate,
        "threshold_contract": {
            "safe": {"count_delta": ">-2", "iou_delta": ">-0.005", "fp_reduction": ">-0.05"},
            "material_any": {"count_delta": ">=2", "iou_delta": ">=0.005", "fp_reduction": ">=0.05"},
            "severe_any": {
                "count_delta": "<=-2",
                "iou_delta": "<=-0.01",
                "fp_reduction": "<=-0.25",
                "fp_introduced_from_zero": True,
            },
            "cross_dataset": ">=2/3 safe-material and 0/3 severe",
            "functional_equivalence": "max_abs<=1e-7 AND mean_abs<=1e-8 on unpadded valid pixels",
        },
        "aggregates": {
            "blocks": block_aggregates,
            "all7_off_vs_full": all_off_forward,
            "full_vs_all7_off": full_vs_all_off,
            "all7_off_probability": {
                "equivalent_datasets": equivalent_datasets,
                "equivalent_dataset_count": len(equivalent_datasets),
                "equivalent_on_all_three_datasets": equivalent_all_three,
            },
            "any_full_vs_all7_off_safe_material_improvement": any_full_safe_material,
            "unsupported_contribution_gate": unsupported_gate,
        },
        "per_dataset": per_dataset,
        "input_bindings": bindings,
        "decision_priority": [
            {
                "priority": 1,
                "decisions": [DECISION_BLOCK_SELECTIVE, DECISION_LOCAL_AUDIT],
            },
            {"priority": 2, "decisions": [DECISION_RESIDUAL_OFF]},
            {"priority": 3, "decisions": [DECISION_KEEP]},
            {"priority": 4, "decisions": [DECISION_UNSUPPORTED]},
            {"priority": 5, "decisions": [DECISION_INCONCLUSIVE]},
        ],
        "scope": {
            "single_seed_test_selected_diagnostic": True,
            "fixed_weight_residual_sensitivity_not_retrained_candidate": True,
            "descriptive_sweep_used_for_decision": False,
            "mechanism_statistics_can_authorize_performance_change": False,
            "paper_mechanism_evidence": False,
            "stability_claim_supported": False,
        },
        "source_sha256": {
            "analysis/compare_three_dataset_tpd8_block_residual_knockout_v1.py": file_sha256(Path(__file__)),
            "analysis/analyze_three_dataset_tpd8_block_residual_knockout_v1.py": file_sha256(Path(analyzer.__file__)),
            "analysis/compare_three_dataset_qfg_level_knockout_v1.py": file_sha256(Path(gate_core.__file__)),
        },
        "no_fabricated_results": True,
    }


def _format_delta(value: Any) -> str:
    return f"{float(value):+.6f}"


def _format_reduction(value: Mapping[str, Any]) -> str:
    reduction = value["value"]
    if reduction is None:
        return "N/A (0→positive)"
    return f"{100.0 * float(reduction):+.2f}%"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# 三数据集 TPD8 Block Residual Knockout V1 裁决",
        "",
        f"- 决策：`{result['decision']}`",
        "- 固定条件：seed `42`、各自 `best_miou`、阈值 `0.5`。",
        "- knockout 仅将选中 block 的 `saliency_scale` 置零；Keep/SPD 路径保留。",
        "- 单个有害 block 的 single-off 已是完整候选；多 block 组合首轮只生成下一模式建议。",
        "",
        "| 数据集 | 模式 | Δtarget | Δtiny | ΔmIoU | ΔnIoU | component-Fa像素降幅 | background-FP降幅 | safe-material | severe |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for dataset in DATASETS:
        for mode in MODES[1:]:
            row = result["per_dataset"][dataset]["modes"][mode]["off_vs_full"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        dataset,
                        mode,
                        str(row["delta_target"]),
                        str(row["delta_tiny"]),
                        _format_delta(row["delta_miou"]),
                        _format_delta(row["delta_niou"]),
                        _format_reduction(row["component_fp_reduction"]),
                        _format_reduction(row["background_pixel_fp_reduction"]),
                        "YES" if row["safe_material_improvement"] else "NO",
                        "YES" if row["severe_degradation"] else "NO",
                    )
                )
                + " |"
            )
    lines.extend(("", "## 聚合", ""))
    for mode in SINGLE_MODES:
        aggregate = result["aggregates"]["blocks"][mode]
        lines.append(
            f"- `{mode}` / `{aggregate['block_id']}`：off改善 "
            f"{aggregate['off_vs_full']['safe_material_dataset_count']}/3，"
            f"severe {aggregate['off_vs_full']['severe_degradation_dataset_count']}/3，"
            f"persistent harmful=`{str(aggregate['persistent_harmful_block']).lower()}`。"
        )
    all_off = result["aggregates"]["all7_off_vs_full"]
    reverse = result["aggregates"]["full_vs_all7_off"]
    lines.extend(
        (
            f"- `all7_off` 相对 full：safe-material {all_off['safe_material_dataset_count']}/3，severe {all_off['severe_degradation_dataset_count']}/3。",
            f"- full 相对 `all7_off`：safe-material {reverse['safe_material_dataset_count']}/3，severe {reverse['severe_degradation_dataset_count']}/3。",
            f"- full/all7-off 功能等价：{result['aggregates']['all7_off_probability']['equivalent_dataset_count']}/3。",
            "",
            "该结果只裁决训练后 TPD residual 的固定权重敏感性；不等于完整 TPD 相对 Original 的训练对照。",
            "",
        )
    )
    return "\n".join(lines)


def _default_input(input_root: Path, dataset: str) -> Path:
    return (
        Path(input_root)
        / "runs"
        / dataset
        / "v4_tss_off_best_miou_seed42"
        / "evaluation.json"
    )


def _parse_bindings(values: Sequence[str], input_root: Path) -> dict[str, Path]:
    bindings = {dataset: _default_input(input_root, dataset) for dataset in DATASETS}
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--input must use DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        _require(dataset in DATASETS, f"unsupported --input dataset: {dataset}")
        _require(dataset not in seen, f"duplicate --input dataset: {dataset}")
        _require(bool(raw_path), f"empty --input path for {dataset}")
        bindings[dataset] = Path(raw_path)
        seen.add(dataset)
    return bindings


def write_outputs(json_path: Path, markdown_path: Path, result: Mapping[str, Any]) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    _require(json_path != markdown_path, "JSON and Markdown outputs must differ")
    for path in (json_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    json_text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(result)
    gate_core._atomic_write_once(json_path, json_text)
    try:
        gate_core._atomic_write_once(markdown_path, markdown_text)
    except BaseException:
        if json_path.is_file() and not json_path.is_symlink():
            json_path.unlink()
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--input", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = _parse_bindings(args.input, args.input_root)
    payloads: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        resolved = input_paths[dataset].resolve(strict=True)
        payload, sha256 = gate_core._load_json_with_sha(resolved)
        _require(file_sha256(resolved) == sha256, f"input changed while reading: {dataset}")
        payloads[dataset] = payload
        bindings[dataset] = {"path": str(resolved), "sha256": sha256}
    result = compare_payloads(payloads, input_bindings=bindings)
    write_outputs(args.output_json, args.output_md, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "decision": result["decision"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
