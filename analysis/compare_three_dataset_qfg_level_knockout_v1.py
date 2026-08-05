#!/usr/bin/env python3
"""Compare the frozen three-dataset QFG level-knockout diagnostics.

This program performs no inference.  It consumes exactly one analyzer JSON
for each frozen dataset and applies the pre-registered rules in section 11.2
of ``SCTransNet_TSS最终裁决与NER_QFG_TPD下一步优化方案.md``.  All primary
decisions use the seed-42, test-selected, best-mIoU, threshold-0.5 points.
The descriptive threshold sweeps are validated as present but never used for
the decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_three_dataset_qfg_level_knockout_v1 as analyzer  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA = "sctransnet_three_dataset_qfg_level_knockout_comparison_v1/v1"
ANALYZER_SCHEMA = analyzer.SCHEMA
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLE = "best_miou"
SEED = 42
FIXED_THRESHOLD = 0.5

MODES = tuple(analyzer.PUBLIC_MODES)
LEVEL_MODES = MODES[1:5]
MODE_TO_ZERO_BASED_LEVEL = {
    "level0_off": 0,
    "level1_off": 1,
    "level2_off": 2,
    "level3_off": 3,
}

SAFE_COUNT_DELTA_MIN_EXCLUSIVE = -2
SAFE_IOU_DELTA_MIN_EXCLUSIVE = -0.005
SAFE_FP_REDUCTION_MIN_EXCLUSIVE = -0.05

MATERIAL_COUNT_DELTA_MINIMUM = 2
MATERIAL_IOU_DELTA_MINIMUM = 0.005
MATERIAL_FP_REDUCTION_MINIMUM = 0.05

SEVERE_COUNT_DELTA_MAXIMUM = -2
SEVERE_IOU_DELTA_MAXIMUM = -0.01
SEVERE_FP_REDUCTION_MAXIMUM = -0.25

REQUIRED_DATASET_COUNT = 2
PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD = analyzer.OUTPUT_EQUIVALENCE_MAX_ABS
PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD = analyzer.OUTPUT_EQUIVALENCE_MEAN_ABS

DECISION_REMOVE_LEVELS = "DESIGN_QFG_V3_REMOVE_LEVELS"
DECISION_ALL_OFF = "DESIGN_QFG_OFF_CANDIDATE"
DECISION_KEEP = "FREEZE_QFG2_KEEP"
DECISION_UNSUPPORTED = "QFG_CONTRIBUTION_UNSUPPORTED_CONSIDER_REMOVE"
DECISION_INCONCLUSIVE = "QFG_INCONCLUSIVE_NO_FORMULA_CHANGE"

DEFAULT_INPUT_ROOT = analyzer.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison" / "best_miou_seed42"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    """Read and hash the same immutable byte snapshot."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value, _sha256_bytes(raw)


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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _validate_sha_pair(audit: Mapping[str, Any], prefix: str) -> None:
    before = audit.get(f"{prefix}_sha256_before")
    after = audit.get(f"{prefix}_sha256_after")
    unchanged = audit.get(f"{prefix}_unchanged")
    _require(_is_sha256(before), f"invalid {prefix} before SHA")
    _require(_is_sha256(after), f"invalid {prefix} after SHA")
    _require(unchanged is True, f"{prefix} was not restored")
    _require(before == after, f"{prefix} SHA changed")


def _extract_point(mode_payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    point = _mapping(mode_payload.get("fixed_threshold_0_5"), f"{label}.fixed")
    threshold = _finite_float(point.get("threshold"), f"{label}.threshold")
    _require(threshold == FIXED_THRESHOLD, f"{label} threshold differs")
    ready = {
        "threshold": threshold,
        "matched_target_count": _nonnegative_int(
            point.get("matched_target_count"), f"{label}.matched_target_count"
        ),
        "matched_tiny_target_count": _nonnegative_int(
            point.get("matched_tiny_target_count"),
            f"{label}.matched_tiny_target_count",
        ),
        "miou": _finite_float(point.get("miou"), f"{label}.miou"),
        "niou": _finite_float(point.get("niou"), f"{label}.niou"),
        "component_false_positive_pixels": _nonnegative_int(
            point.get("unmatched_predicted_pixels"),
            f"{label}.unmatched_predicted_pixels",
        ),
        "background_false_positive_pixels": _nonnegative_int(
            point.get("false_positive_pixels"), f"{label}.false_positive_pixels"
        ),
        "valid_pixel_count": _nonnegative_int(
            point.get("valid_pixel_count"), f"{label}.valid_pixel_count"
        ),
    }
    _require(ready["valid_pixel_count"] > 0, f"{label}.valid_pixel_count must be positive")
    for metric in ("miou", "niou"):
        _require(0.0 <= ready[metric] <= 1.0, f"{label}.{metric} is outside [0,1]")
    return ready


def _expected_selected_levels(mode: str) -> list[int]:
    if mode == "full":
        return []
    if mode == "all_off":
        return [0, 1, 2, 3]
    return [MODE_TO_ZERO_BASED_LEVEL[mode]]


def _extract_probability_difference(
    mode_payload: Mapping[str, Any], label: str
) -> dict[str, Any]:
    raw = _mapping(
        mode_payload.get("probability_difference_to_full"),
        f"{label}.probability_difference_to_full",
    )
    max_abs = _finite_float(raw.get("max_abs"), f"{label}.max_abs")
    mean_abs = _finite_float(raw.get("mean_abs"), f"{label}.mean_abs")
    _require(max_abs >= 0.0 and mean_abs >= 0.0, f"{label} has negative difference")
    element_count = _nonnegative_int(raw.get("element_count"), f"{label}.element_count")
    _require(element_count > 0, f"{label}.element_count must be positive")
    _require(
        raw.get("scope") == "all_original_unpadded_test_pixels",
        f"{label} probability scope differs",
    )
    absolute_sum = _finite_float(
        raw.get("absolute_difference_sum"), f"{label}.absolute_difference_sum"
    )
    _require(absolute_sum >= 0.0, f"{label} has a negative absolute sum")
    _require(
        mean_abs == absolute_sum / element_count,
        f"{label} probability mean is inconsistent",
    )
    _require(mean_abs <= max_abs, f"{label} probability mean exceeds maximum")
    reported_max_threshold = _finite_float(
        raw.get("equivalence_max_abs_threshold"),
        f"{label}.equivalence_max_abs_threshold",
    )
    reported_mean_threshold = _finite_float(
        raw.get("equivalence_mean_abs_threshold"),
        f"{label}.equivalence_mean_abs_threshold",
    )
    _require(
        reported_max_threshold == PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD,
        f"{label} max-abs threshold differs",
    )
    _require(
        reported_mean_threshold == PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD,
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
        "mean_abs": mean_abs,
        "absolute_difference_sum": absolute_sum,
        "element_count": element_count,
        "equivalent": not functionally_different,
        "functionally_different": functionally_different,
    }


def validate_analyzer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one analyzer payload and retain only decision inputs."""

    analyzer.validate_output_payload(payload)
    _require(payload.get("schema") == ANALYZER_SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer result is incomplete")
    dataset = payload.get("dataset")
    _require(dataset in DATASETS, f"unsupported dataset: {dataset!r}")
    _require(payload.get("checkpoint_role") == CHECKPOINT_ROLE, "checkpoint role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("test_selected") is True, "result is not test-selected")
    _require(
        payload.get("selection_is_optimistic") is True,
        "selection policy differs",
    )
    _require(
        payload.get("evaluation_protocol") == "img_idx_test_selected_development",
        "evaluation protocol differs",
    )
    _require(payload.get("fixed_threshold") == FIXED_THRESHOLD, "top fixed threshold differs")
    replay = _mapping(payload.get("reference_replay_audit"), "reference_replay_audit")
    _require(replay.get("passed") is True, "reference replay did not pass")

    checkpoint = _mapping(payload.get("checkpoint_binding"), "checkpoint_binding")
    checkpoint_file = _mapping(checkpoint.get("checkpoint"), "checkpoint_binding.checkpoint")
    _require(
        _is_sha256(checkpoint_file.get("sha256")),
        "checkpoint binding has an invalid SHA",
    )
    _require(
        checkpoint_file.get("role") == CHECKPOINT_ROLE,
        "checkpoint binding role differs",
    )
    checkpoint_protocol = _mapping(
        checkpoint.get("protocol"), "checkpoint_binding.protocol"
    )
    _require(
        _is_sha256(checkpoint_protocol.get("payload_sha256")),
        "checkpoint protocol has an invalid payload SHA",
    )
    data = _mapping(payload.get("data"), "data")
    manifest = _mapping(data.get("protocol_manifest"), "data.protocol_manifest")
    _require(_is_sha256(manifest.get("sha256")), "data manifest SHA is invalid")
    _require(
        _is_sha256(data.get("inference_order_newline_sha256")),
        "inference-order SHA is invalid",
    )
    reference_reuse = _mapping(payload.get("reference_reuse"), "reference_reuse")
    _require(
        _is_sha256(reference_reuse.get("sha256")),
        "reference input SHA is invalid",
    )
    source_sha256 = _mapping(payload.get("source_sha256"), "source_sha256")
    _require(bool(source_sha256), "analyzer source SHA map is empty")
    _require(
        all(_is_sha256(value) for value in source_sha256.values()),
        "analyzer source SHA map is invalid",
    )

    restoration = _mapping(payload.get("restoration_audit"), "restoration_audit")
    _validate_sha_pair(restoration, "model_state")
    _validate_sha_pair(restoration, "alpha_state")

    modes = _mapping(payload.get("modes"), "modes")
    _require(set(modes) == set(MODES), "analyzer modes differ from frozen six modes")
    _require(
        payload.get("mode_order") == list(MODES),
        "analyzer canonical mode_order differs",
    )
    normalized_modes: dict[str, Any] = {}
    full_valid_pixel_count: int | None = None
    for mode in MODES:
        raw_mode = _mapping(modes.get(mode), f"modes.{mode}")
        # These mappings are part of the analyzer contract even though only
        # the fixed point and probability delta enter the decision.
        for field in (
            "descriptive_pd_fa",
            "query_perturbation",
            "factor_summary",
            "spatial_gate_factor_statistics",
            "alpha_knockout",
        ):
            _mapping(raw_mode.get(field), f"modes.{mode}.{field}")
        alpha = _mapping(raw_mode.get("alpha_knockout"), f"modes.{mode}.alpha_knockout")
        selected = alpha.get("selected_level_indices_zero_based")
        _require(
            isinstance(selected, list)
            and all(isinstance(level, int) and not isinstance(level, bool) for level in selected),
            f"modes.{mode} selected levels are malformed",
        )
        _require(
            selected == _expected_selected_levels(mode),
            f"modes.{mode} selected the wrong QFG levels",
        )
        for sha_field in ("source_alpha_sha256", "active_alpha_sha256"):
            _require(
                _is_sha256(alpha.get(sha_field)),
                f"modes.{mode}.{sha_field} is invalid",
            )
        _require(
            alpha.get("source_alpha_sha256")
            == restoration.get("alpha_state_sha256_before"),
            f"modes.{mode} alpha source differs from restoration audit",
        )
        if mode == "full":
            _require(
                alpha.get("active_alpha_sha256") == alpha.get("source_alpha_sha256"),
                "full mode unexpectedly changed alpha",
            )
        mode_restoration = _mapping(
            raw_mode.get("restoration_audit"), f"modes.{mode}.restoration_audit"
        )
        _require(
            mode_restoration.get("alpha_state_unchanged") is True,
            f"modes.{mode} alpha was not restored",
        )
        point = _extract_point(raw_mode, f"modes.{mode}")
        probability = _extract_probability_difference(raw_mode, f"modes.{mode}")
        _require(
            probability["element_count"] == point["valid_pixel_count"],
            f"modes.{mode} probability count differs from valid pixels",
        )
        if full_valid_pixel_count is None:
            _require(mode == "full", "full mode must be first")
            full_valid_pixel_count = point["valid_pixel_count"]
        _require(
            point["valid_pixel_count"] == full_valid_pixel_count,
            f"modes.{mode} valid-pixel count differs from full",
        )
        if mode == "full":
            _require(
                probability["max_abs"] == 0.0 and probability["mean_abs"] == 0.0,
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


def _fp_reduction(candidate: int, reference: int, label: str) -> dict[str, Any]:
    _require(candidate >= 0 and reference >= 0, f"{label} counts must be non-negative")
    if reference == 0:
        if candidate == 0:
            return {
                "value": 0.0,
                "reference_denominator_zero": True,
                "introduced_from_zero": False,
                "safety_pass": True,
                "severe_degradation": False,
            }
        # A finite relative percentage does not exist.  JSON null preserves
        # that fact; introducing FP from a zero reference fails safety and is
        # conservatively treated as severe for the third-dataset veto.
        return {
            "value": None,
            "reference_denominator_zero": True,
            "introduced_from_zero": True,
            "safety_pass": False,
            "severe_degradation": True,
        }
    value = (reference - candidate) / reference
    return {
        "value": value,
        "reference_denominator_zero": False,
        "introduced_from_zero": False,
        "safety_pass": value > SAFE_FP_REDUCTION_MIN_EXCLUSIVE,
        "severe_degradation": value <= SEVERE_FP_REDUCTION_MAXIMUM,
    }


def compare_direction(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Score candidate relative to reference using the frozen strict gates."""

    delta_target = int(candidate["matched_target_count"]) - int(
        reference["matched_target_count"]
    )
    delta_tiny = int(candidate["matched_tiny_target_count"]) - int(
        reference["matched_tiny_target_count"]
    )
    delta_miou = float(candidate["miou"]) - float(reference["miou"])
    delta_niou = float(candidate["niou"]) - float(reference["niou"])
    component = _fp_reduction(
        int(candidate["component_false_positive_pixels"]),
        int(reference["component_false_positive_pixels"]),
        "component FP",
    )
    background = _fp_reduction(
        int(candidate["background_false_positive_pixels"]),
        int(reference["background_false_positive_pixels"]),
        "background pixel FP",
    )

    safe_conditions = {
        "delta_target_gt_minus_2": delta_target > SAFE_COUNT_DELTA_MIN_EXCLUSIVE,
        "delta_tiny_gt_minus_2": delta_tiny > SAFE_COUNT_DELTA_MIN_EXCLUSIVE,
        "delta_miou_gt_minus_0_005": delta_miou > SAFE_IOU_DELTA_MIN_EXCLUSIVE,
        "delta_niou_gt_minus_0_005": delta_niou > SAFE_IOU_DELTA_MIN_EXCLUSIVE,
        "component_fp_reduction_gt_minus_0_05": component["safety_pass"],
        "background_pixel_fp_reduction_gt_minus_0_05": background["safety_pass"],
    }
    material_conditions = {
        "delta_target_ge_2": delta_target >= MATERIAL_COUNT_DELTA_MINIMUM,
        "delta_tiny_ge_2": delta_tiny >= MATERIAL_COUNT_DELTA_MINIMUM,
        "delta_miou_ge_0_005": delta_miou >= MATERIAL_IOU_DELTA_MINIMUM,
        "delta_niou_ge_0_005": delta_niou >= MATERIAL_IOU_DELTA_MINIMUM,
        "component_fp_reduction_ge_0_05": (
            component["value"] is not None
            and component["value"] >= MATERIAL_FP_REDUCTION_MINIMUM
        ),
        "background_pixel_fp_reduction_ge_0_05": (
            background["value"] is not None
            and background["value"] >= MATERIAL_FP_REDUCTION_MINIMUM
        ),
    }
    severe_conditions = {
        "delta_target_le_minus_2": delta_target <= SEVERE_COUNT_DELTA_MAXIMUM,
        "delta_tiny_le_minus_2": delta_tiny <= SEVERE_COUNT_DELTA_MAXIMUM,
        "delta_miou_le_minus_0_01": delta_miou <= SEVERE_IOU_DELTA_MAXIMUM,
        "delta_niou_le_minus_0_01": delta_niou <= SEVERE_IOU_DELTA_MAXIMUM,
        "component_fp_increase_ge_25pct": component["severe_degradation"],
        "background_pixel_fp_increase_ge_25pct": background["severe_degradation"],
    }
    safe = all(safe_conditions.values())
    material = any(material_conditions.values())
    severe = any(severe_conditions.values())
    return {
        "delta_target": delta_target,
        "delta_tiny": delta_tiny,
        "delta_miou": delta_miou,
        "delta_niou": delta_niou,
        "component_fp_reduction": component,
        "background_pixel_fp_reduction": background,
        "safe_conditions": safe_conditions,
        "material_gain_conditions": material_conditions,
        "severe_degradation_conditions": severe_conditions,
        "safe": safe,
        "material_gain": material,
        "safe_material_improvement": safe and material,
        "severe_degradation": severe,
    }


def _aggregate_direction(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    safe_material = [
        dataset
        for dataset in DATASETS
        if bool(rows[dataset]["safe_material_improvement"])
    ]
    severe = [
        dataset for dataset in DATASETS if bool(rows[dataset]["severe_degradation"])
    ]
    cross_dataset = len(safe_material) >= REQUIRED_DATASET_COUNT and not severe
    return {
        "safe_material_datasets": safe_material,
        "safe_material_dataset_count": len(safe_material),
        "required_dataset_count": REQUIRED_DATASET_COUNT,
        "severe_degradation_datasets": severe,
        "severe_degradation_dataset_count": len(severe),
        "cross_dataset_safe_material_improvement": cross_dataset,
    }


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


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    input_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate three analyzer results and apply the frozen priority order."""

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
        mode_rows: dict[str, Any] = {}
        for mode in MODES[1:]:
            off = modes[mode]["fixed_threshold_0_5"]
            off_vs_full = compare_direction(off, full)
            full_vs_off = compare_direction(full, off)
            forward_rows[mode][dataset] = off_vs_full
            reverse_rows[mode][dataset] = full_vs_off
            mode_rows[mode] = {
                "off_vs_full": off_vs_full,
                "full_vs_off": full_vs_off,
                "probability_difference_to_full": modes[mode][
                    "probability_difference_to_full"
                ],
            }
        per_dataset[dataset] = {
            "checkpoint_sha256": normalized[dataset]["checkpoint_sha256"],
            "modes": mode_rows,
        }

    level_aggregates: dict[str, Any] = {}
    persistent_harmful_modes: list[str] = []
    stable_level_effect_modes: list[str] = []
    for mode in LEVEL_MODES:
        forward = _aggregate_direction(forward_rows[mode])
        reverse = _aggregate_direction(reverse_rows[mode])
        persistent_harmful = bool(forward["cross_dataset_safe_material_improvement"])
        stable_effect = bool(
            persistent_harmful or reverse["cross_dataset_safe_material_improvement"]
        )
        if persistent_harmful:
            persistent_harmful_modes.append(mode)
        if stable_effect:
            stable_level_effect_modes.append(mode)
        level_aggregates[mode] = {
            "zero_based_level": MODE_TO_ZERO_BASED_LEVEL[mode],
            "off_vs_full": forward,
            "full_vs_off": reverse,
            "persistent_harmful_level": persistent_harmful,
            "stable_cross_dataset_effect": stable_effect,
        }

    all_off_forward = _aggregate_direction(forward_rows["all_off"])
    full_vs_all_off = _aggregate_direction(reverse_rows["all_off"])
    equivalent_datasets = [
        dataset
        for dataset in DATASETS
        if not normalized[dataset]["modes"]["all_off"]
        ["probability_difference_to_full"]["functionally_different"]
    ]
    different_datasets = [dataset for dataset in DATASETS if dataset not in equivalent_datasets]
    no_functional_difference = len(equivalent_datasets) == len(DATASETS)
    functional_contribution_supported: bool | None
    if no_functional_difference:
        functional_contribution_supported = False
    elif len(different_datasets) >= REQUIRED_DATASET_COUNT:
        functional_contribution_supported = True
    else:
        functional_contribution_supported = None
    all_levels_no_stable_impact = not stable_level_effect_modes
    any_full_vs_all_off_safe_material = any(
        reverse_rows["all_off"][dataset]["safe_material_improvement"]
        for dataset in DATASETS
    )
    unsupported_gate = bool(
        no_functional_difference
        and not persistent_harmful_modes
        and not any_full_vs_all_off_safe_material
    )

    if persistent_harmful_modes:
        decision = DECISION_REMOVE_LEVELS
    elif all_off_forward["cross_dataset_safe_material_improvement"]:
        decision = DECISION_ALL_OFF
    elif full_vs_all_off["cross_dataset_safe_material_improvement"]:
        decision = DECISION_KEEP
    elif unsupported_gate:
        decision = DECISION_UNSUPPORTED
    else:
        decision = DECISION_INCONCLUSIVE

    harmful_levels = [MODE_TO_ZERO_BASED_LEVEL[mode] for mode in persistent_harmful_modes]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": decision,
        "checkpoint_role": CHECKPOINT_ROLE,
        "seed": SEED,
        "test_selected": True,
        "fixed_threshold": FIXED_THRESHOLD,
        "datasets": list(DATASETS),
        "persistent_harmful_level_modes": persistent_harmful_modes,
        "persistent_harmful_zero_based_levels": harmful_levels,
        "qfg_v3_remove_levels_authorized": decision == DECISION_REMOVE_LEVELS,
        "qfg_off_candidate_authorized": decision == DECISION_ALL_OFF,
        "qfg_performance_contribution_supported": bool(
            full_vs_all_off["cross_dataset_safe_material_improvement"]
        ),
        "qfg_functional_contribution_supported": functional_contribution_supported,
        "qfg_contribution_unsupported": unsupported_gate,
        "threshold_contract": {
            "safe": {
                "count_delta": ">-2",
                "iou_delta": ">-0.005",
                "fp_reduction": ">-0.05",
            },
            "material_any": {
                "count_delta": ">=2",
                "iou_delta": ">=0.005",
                "fp_reduction": ">=0.05",
            },
            "severe_any": {
                "count_delta": "<=-2",
                "iou_delta": "<=-0.01",
                "fp_reduction": "<=-0.25",
                "fp_introduced_from_zero": True,
            },
            "cross_dataset": ">=2/3 safe-material and 0/3 severe",
            "probability_functionally_different": (
                "max_abs>1e-7 OR mean_abs>1e-8"
            ),
        },
        "aggregates": {
            "levels": level_aggregates,
            "all_off_vs_full": all_off_forward,
            "full_vs_all_off": full_vs_all_off,
            "all_level_knockouts_no_stable_impact": all_levels_no_stable_impact,
            "stable_level_effect_modes": stable_level_effect_modes,
            "all_off_probability": {
                "equivalent_datasets": equivalent_datasets,
                "equivalent_dataset_count": len(equivalent_datasets),
                "functionally_different_datasets": different_datasets,
                "functionally_different_dataset_count": len(different_datasets),
                "equivalent_on_all_three_datasets": no_functional_difference,
            },
            "any_full_vs_all_off_safe_material_improvement": (
                any_full_vs_all_off_safe_material
            ),
            "unsupported_contribution_gate": unsupported_gate,
        },
        "per_dataset": per_dataset,
        "input_bindings": bindings,
        "decision_priority": [
            DECISION_REMOVE_LEVELS,
            DECISION_ALL_OFF,
            DECISION_KEEP,
            DECISION_UNSUPPORTED,
            DECISION_INCONCLUSIVE,
        ],
        "scope": {
            "single_seed_test_selected_diagnostic": True,
            "descriptive_sweep_used_for_decision": False,
            "authorizes_new_frequency_branches": False,
            "paper_mechanism_evidence": False,
            "stability_claim_supported": False,
        },
        "source_sha256": {
            "analysis/compare_three_dataset_qfg_level_knockout_v1.py": file_sha256(
                Path(__file__)
            ),
            "analysis/analyze_three_dataset_qfg_level_knockout_v1.py": file_sha256(
                Path(analyzer.__file__)
            ),
        },
        "no_fabricated_results": True,
    }


def _format_delta(value: Any) -> str:
    return f"{float(value):+.6f}"


def _format_reduction(reduction: Mapping[str, Any]) -> str:
    value = reduction["value"]
    if value is None:
        return "N/A (0→positive)"
    return f"{100.0 * float(value):+.2f}%"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# 三数据集 QFG Level Knockout V1 裁决",
        "",
        f"- 决策：`{result['decision']}`",
        f"- 固定条件：seed `{result['seed']}`、`best_miou`、阈值 `0.5`。",
        "- 扫描结果仅作描述，未参与裁决。",
        "",
        "| 数据集 | 模式 | Δtarget | Δtiny | ΔmIoU | ΔnIoU | component-FP降幅 | background-pixel-FP降幅 | 安全实质改善 | 严重退化 |",
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
    aggregates = result["aggregates"]
    lines.extend(("", "## 跨数据集裁决", ""))
    for mode in LEVEL_MODES:
        aggregate = aggregates["levels"][mode]
        lines.append(
            f"- `{mode}`：off改善 "
            f"{aggregate['off_vs_full']['safe_material_dataset_count']}/3，"
            f"严重退化 {aggregate['off_vs_full']['severe_degradation_dataset_count']}/3，"
            f"persistent harmful=`{str(aggregate['persistent_harmful_level']).lower()}`。"
        )
    lines.extend(
        (
            f"- `all_off` 相对 full：安全实质改善 "
            f"{aggregates['all_off_vs_full']['safe_material_dataset_count']}/3，"
            f"严重退化 {aggregates['all_off_vs_full']['severe_degradation_dataset_count']}/3。",
            f"- full 相对 `all_off`：安全实质改善 "
            f"{aggregates['full_vs_all_off']['safe_material_dataset_count']}/3，"
            f"严重退化 {aggregates['full_vs_all_off']['severe_degradation_dataset_count']}/3。",
            f"- full/all-off 概率功能等价："
            f"{aggregates['all_off_probability']['equivalent_dataset_count']}/3。",
            "",
            "该裁决仅用于当前 seed42、test-selected 开发流程，不构成多随机性或论文级稳定性结论。",
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


def _atomic_write_once(path: Path, text: str) -> None:
    """Publish a new file atomically and refuse every existing destination."""

    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        # Hard-link publication is exclusive: it cannot replace a destination
        # created after the preflight check.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_outputs(json_path: Path, markdown_path: Path, result: Mapping[str, Any]) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    _require(json_path != markdown_path, "JSON and Markdown outputs must differ")
    for path in (json_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    json_text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(result)
    _atomic_write_once(json_path, json_text)
    try:
        _atomic_write_once(markdown_path, markdown_text)
    except BaseException:
        # Roll back only the just-created JSON so the output pair is never
        # intentionally left half-published.
        if json_path.is_file() and not json_path.is_symlink():
            json_path.unlink()
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="override one discovered analyzer result",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = _parse_bindings(args.input, args.input_root)
    payloads: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        path = input_paths[dataset]
        resolved = path.resolve(strict=True)
        payload, sha256 = _load_json_with_sha(resolved)
        # Re-check the file after parsing to reject a concurrently replaced
        # input before the decision is published.
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
