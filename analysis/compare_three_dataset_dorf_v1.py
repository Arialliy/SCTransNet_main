#!/usr/bin/env python3
"""Compare the frozen twelve-input DORF V1 zero-training matrix.

The comparator is intentionally independent of the DORF analyzer module.  It
validates the literal analyzer schema, recomputes every fixed-threshold delta
and safe/material/severe predicate, applies the Final-only primary gate, and
uses Original only for the frozen shared-alpha competitiveness check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA = "sctransnet_three_dataset_dorf_v1_comparison/v1"
ANALYZER_SCHEMA = "sctransnet_three_dataset_dorf_v1/v1"
METHODS = ("final_tss_off", "original")
PRIMARY_METHOD = "final_tss_off"
BASELINE_METHOD = "original"
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLES = ("best_miou", "best_pd")
PRIMARY_ROLE = "best_miou"
VETO_ROLE = "best_pd"
SEED = 42
FIXED_THRESHOLD = 0.5
MODES = ("current_out", "dorf_a025", "dorf_a050", "dorf_a075", "d0_only")
ALPHA_BY_MODE = {
    "current_out": 0.0,
    "dorf_a025": 0.25,
    "dorf_a050": 0.50,
    "dorf_a075": 0.75,
    "d0_only": 1.0,
}
CURRENT_MODE = "current_out"
NONZERO_MODES = MODES[1:]
REQUIRED_PRIMARY_SAFE_MATERIAL_DATASETS = 2

SAFE_COUNT_DELTA_MIN_EXCLUSIVE = -2
SAFE_IOU_DELTA_MIN_EXCLUSIVE = -0.005
SAFE_FP_REDUCTION_MIN_EXCLUSIVE = -0.05
MATERIAL_COUNT_DELTA_MINIMUM = 2
MATERIAL_IOU_DELTA_MINIMUM = 0.005
MATERIAL_FP_REDUCTION_MINIMUM = 0.05
SEVERE_COUNT_DELTA_MAXIMUM = -2
SEVERE_IOU_DELTA_MAXIMUM = -0.01
SEVERE_FP_REDUCTION_MAXIMUM = -0.25

DECISION_AUTHORIZE = "AUTHORIZE_DORF_V1_PRODUCTION_IMPLEMENTATION"
DECISION_NO_AUTHORIZATION = "DORF_V1_ZERO_TRAINING_TRIGGER_FAILED"

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "three_dataset_dorf_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison" / "seed42_twelve_role"
INPUT_MANIFEST_SCHEMA = "sctransnet_three_dataset_dorf_v1_input_manifest/v1"
INPUT_MANIFEST_RELATIVE_PATH = Path(
    "results/three_dataset_dorf_v1/manifests/dorf_v1_input_manifest.json"
)
DEFAULT_INPUT_MANIFEST = REPO_ROOT / INPUT_MANIFEST_RELATIVE_PATH
FROZEN_INPUT_MANIFEST_SHA256 = (
    "38bb9a2e4ae5662ae32da6b346444e6d34f5aba57ca13c5ae1dc4516f4230359"
)

SEVERE_CONDITION_ORDER = (
    "delta_target_le_minus_2",
    "delta_tiny_le_minus_2",
    "delta_miou_le_minus_0_01",
    "delta_niou_le_minus_0_01",
    "component_fp_increase_ge_25pct",
    "background_pixel_fp_increase_ge_25pct",
)
EXPECTED_TRAINING_STATE_KEYS = {"final_tss_off": 568, "original": 510}
EXPECTED_INFERENCE_STATE_KEYS = {"final_tss_off": 564, "original": 510}
EXPECTED_REMOVED_TSS_KEYS = {"final_tss_off": 4, "original": 0}
EXPECTED_BUILDERS = {
    "final_tss_off": "build_final_inference_model_from_training_state_dict",
    "original": "build_paper_model_original_then_strict_load",
}


class DORFComparisonError(ValueError):
    """An input or result differs from the frozen DORF V1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DORFComparisonError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DORFComparisonError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise DORFComparisonError(f"{label} must be an array")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DORFComparisonError(f"{label} must be numeric")
    ready = float(value)
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DORFComparisonError(f"{label} must be an integer")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    ready = Path(path)
    if not ready.is_file() or ready.is_symlink():
        raise FileNotFoundError(ready)
    digest = hashlib.sha256()
    with ready.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _extract_point(raw_mode: Mapping[str, Any], label: str) -> dict[str, Any]:
    point = _mapping(raw_mode.get("fixed_threshold_0_5"), f"{label}.fixed_threshold_0_5")
    threshold = _finite(point.get("threshold"), f"{label}.threshold")
    _require(threshold == FIXED_THRESHOLD, f"{label} threshold differs")
    target_count = _nonnegative_int(point.get("target_count"), f"{label}.target_count")
    tiny_count = _nonnegative_int(
        point.get("tiny_target_count"), f"{label}.tiny_target_count"
    )
    matched = _nonnegative_int(
        point.get("matched_target_count"), f"{label}.matched_target_count"
    )
    matched_tiny = _nonnegative_int(
        point.get("matched_tiny_target_count"), f"{label}.matched_tiny_target_count"
    )
    _require(matched <= target_count, f"{label} matched targets exceed total")
    _require(matched_tiny <= tiny_count, f"{label} matched tiny targets exceed total")
    valid_pixels = _nonnegative_int(
        point.get("valid_pixel_count"), f"{label}.valid_pixel_count"
    )
    _require(valid_pixels > 0, f"{label} valid pixels must be positive")
    component_fp = _nonnegative_int(
        point.get("unmatched_predicted_pixels"), f"{label}.unmatched_predicted_pixels"
    )
    background_fp = _nonnegative_int(
        point.get("false_positive_pixels"), f"{label}.false_positive_pixels"
    )
    predicted_objects = _nonnegative_int(
        point.get("predicted_object_count"), f"{label}.predicted_object_count"
    )
    unmatched_objects = _nonnegative_int(
        point.get("unmatched_predicted_object_count"),
        f"{label}.unmatched_predicted_object_count",
    )
    _require(unmatched_objects <= predicted_objects, f"{label} unmatched objects differ")
    _require(component_fp <= valid_pixels, f"{label} component FP exceeds valid pixels")
    _require(background_fp <= valid_pixels, f"{label} background FP exceeds valid pixels")
    ready = {
        "threshold": threshold,
        "target_count": target_count,
        "tiny_target_count": tiny_count,
        "matched_target_count": matched,
        "matched_tiny_target_count": matched_tiny,
        "miou": _finite(point.get("miou"), f"{label}.miou"),
        "niou": _finite(point.get("niou"), f"{label}.niou"),
        "component_false_positive_pixels": component_fp,
        "background_false_positive_pixels": background_fp,
        "predicted_object_count": predicted_objects,
        "unmatched_predicted_object_count": unmatched_objects,
        "false_objects_per_image": _finite(
            point.get("false_objects_per_image"), f"{label}.false_objects_per_image"
        ),
        "valid_pixel_count": valid_pixels,
        "pd": _finite(point.get("pd"), f"{label}.pd"),
        "tiny_pd": _finite(point.get("tiny_pd"), f"{label}.tiny_pd"),
        "fa": _finite(point.get("fa"), f"{label}.fa"),
        "pixel_precision": _finite(
            point.get("pixel_precision"), f"{label}.pixel_precision"
        ),
        "pixel_recall": _finite(point.get("pixel_recall"), f"{label}.pixel_recall"),
        "pixel_f1": _finite(point.get("pixel_f1"), f"{label}.pixel_f1"),
        "test_loss": _finite(point.get("test_loss"), f"{label}.test_loss"),
    }
    for metric in (
        "miou",
        "niou",
        "pd",
        "tiny_pd",
        "fa",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    ):
        _require(0.0 <= ready[metric] <= 1.0, f"{label}.{metric} outside [0,1]")
    _require(ready["false_objects_per_image"] >= 0.0, f"{label} false objects negative")
    _require(ready["test_loss"] >= 0.0, f"{label} test loss negative")
    _require(_close(ready["pd"], _ratio(matched, target_count)), f"{label}.pd differs")
    _require(
        _close(ready["tiny_pd"], _ratio(matched_tiny, tiny_count)),
        f"{label}.tiny_pd differs",
    )
    _require(
        _close(ready["fa"], component_fp / valid_pixels),
        f"{label}.fa differs from unmatched predicted pixels / valid pixels",
    )
    if "image_count" in point:
        image_count = _nonnegative_int(point.get("image_count"), f"{label}.image_count")
        _require(image_count > 0, f"{label}.image_count must be positive")
        _require(
            _close(ready["false_objects_per_image"], unmatched_objects / image_count),
            f"{label}.false_objects_per_image differs",
        )
        ready["image_count"] = image_count
    return ready


def _extract_probability_difference(
    raw_mode: Mapping[str, Any], label: str, valid_pixel_count: int
) -> dict[str, Any]:
    raw = _mapping(
        raw_mode.get("probability_difference_to_current"),
        f"{label}.probability_difference_to_current",
    )
    count = _nonnegative_int(raw.get("element_count"), f"{label}.element_count")
    _require(count == valid_pixel_count, f"{label} probability element count differs")
    absolute_sum = _finite(
        raw.get("absolute_difference_sum"), f"{label}.absolute_difference_sum"
    )
    maximum = _finite(raw.get("max_abs"), f"{label}.max_abs")
    mean = _finite(raw.get("mean_abs"), f"{label}.mean_abs")
    _require(absolute_sum >= 0.0 and 0.0 <= maximum <= 1.0 and 0.0 <= mean <= 1.0,
             f"{label} probability difference differs")
    _require(_close(mean, absolute_sum / count), f"{label} probability mean differs")
    return {
        "element_count": count,
        "absolute_difference_sum": absolute_sum,
        "max_abs": maximum,
        "mean_abs": mean,
    }


def _validate_descriptive_sweep(raw_mode: Mapping[str, Any], label: str) -> None:
    sweep = _mapping(raw_mode.get("descriptive_pd_fa"), f"{label}.descriptive_pd_fa")
    points = list(_sequence(sweep.get("points"), f"{label}.descriptive_pd_fa.points"))
    _require(len(points) >= 2, f"{label} descriptive sweep is incomplete")
    endpoints = [
        _mapping(point, f"{label}.descriptive_pd_fa.points")
        for point in points
        if _finite(_mapping(point, "sweep point").get("threshold"), "sweep threshold") == 1.0
    ]
    _require(len(endpoints) == 1, f"{label} must contain one threshold-1 endpoint")
    endpoint = endpoints[0]
    _require(endpoint.get("selected_point_is_empty") is True, f"{label} endpoint is not empty")
    for field in (
        "matched_target_count",
        "matched_tiny_target_count",
        "unmatched_predicted_pixels",
        "predicted_object_count",
    ):
        _require(_nonnegative_int(endpoint.get(field), f"{label}.endpoint.{field}") == 0,
                 f"{label} endpoint {field} differs")
    _require(_finite(endpoint.get("pd"), f"{label}.endpoint.pd") == 0.0,
             f"{label} endpoint Pd differs")
    _require(_finite(endpoint.get("fa"), f"{label}.endpoint.fa") == 0.0,
             f"{label} endpoint Fa differs")
    roles = _mapping(raw_mode.get("threshold_roles"), f"{label}.threshold_roles")
    _require(_finite(roles.get("main_table_threshold"), f"{label}.main threshold") == 0.5,
             f"{label} main threshold differs")
    _require(roles.get("descriptive_sweep_only") is True, f"{label} sweep role differs")
    _require(roles.get("descriptive_sweep_contains_threshold_1_0") is True,
             f"{label} threshold-1 declaration differs")
    _require(roles.get("threshold_1_0_semantics") == "empty_prediction_pd0_fa0",
             f"{label} threshold-1 semantics differ")


def _validate_sha_map(raw: Any, label: str) -> dict[str, str]:
    source = _mapping(raw, label)
    _require(bool(source), f"{label} is empty")
    ready: dict[str, str] = {}
    for name, sha in source.items():
        _require(isinstance(name, str) and bool(name), f"{label} key differs")
        _require(_is_sha256(sha), f"{label}.{name} SHA differs")
        ready[str(name)] = str(sha)
    return ready


def _validate_analyzer_manifest_binding(
    raw: Any,
    *,
    method: str,
    dataset: str,
    role: str,
) -> dict[str, Any]:
    binding = _mapping(raw, "input_manifest_binding")
    _require(
        binding.get("path") == str(DEFAULT_INPUT_MANIFEST.resolve()),
        "analyzer input manifest path differs",
    )
    _require(
        binding.get("sha256") == FROZEN_INPUT_MANIFEST_SHA256,
        "analyzer input manifest SHA differs",
    )
    _require(binding.get("schema") == INPUT_MANIFEST_SCHEMA, "analyzer manifest schema differs")
    _require(
        binding.get("status") == "frozen_before_dorf_outputs",
        "analyzer manifest status differs",
    )
    key = _binding_key(method, dataset, role)
    _require(binding.get("entry_key") == key, "analyzer manifest entry key differs")
    _require(
        binding.get("verified_before_model_load") is True
        and binding.get("verified_after_inference") is True,
        "analyzer did not verify the input manifest before and after inference",
    )
    _require(
        binding.get("historical_metric_authority") == "bound_evaluation_json_only",
        "analyzer historical metric authority differs",
    )
    _require(
        binding.get("checkpoint_embedded_metrics_fallback_allowed") is False,
        "analyzer enabled checkpoint metric fallback",
    )
    entry = _mapping(binding.get("entry"), "input_manifest_binding.entry")
    _require(
        (entry.get("method"), entry.get("dataset"), entry.get("checkpoint_role"))
        == (method, dataset, role),
        "analyzer manifest entry identity differs",
    )
    epoch = _nonnegative_int(entry.get("epoch"), "input_manifest_binding.entry.epoch")
    _require(epoch > 0, "analyzer manifest epoch must be positive")
    run_dir = entry.get("run_dir")
    _require(
        isinstance(run_dir, str)
        and bool(run_dir)
        and not Path(run_dir).is_absolute()
        and ".." not in Path(run_dir).parts,
        "analyzer manifest run_dir differs",
    )
    for field in (
        "summary_sha256",
        "protocol_sha256",
        "checkpoint_sha256",
        "evaluation_sha256",
    ):
        _require(_is_sha256(entry.get(field)), f"analyzer manifest entry {field} differs")
    protocol = _mapping(
        binding.get("data_protocol_manifest"),
        "input_manifest_binding.data_protocol_manifest",
    )
    background = _mapping(
        binding.get("background_pixel_authority"),
        "input_manifest_binding.background_pixel_authority",
    )
    for item, label in ((protocol, "data protocol"), (background, "background authority")):
        _require(
            isinstance(item.get("path"), str) and Path(str(item["path"])).is_absolute(),
            f"analyzer {label} path differs",
        )
        _require(_is_sha256(item.get("sha256")), f"analyzer {label} SHA differs")
    return {
        "path": str(binding["path"]),
        "sha256": str(binding["sha256"]),
        "schema": str(binding["schema"]),
        "status": str(binding["status"]),
        "entry_key": key,
        "entry": dict(entry),
        "data_protocol_manifest": dict(protocol),
        "background_pixel_authority": dict(background),
        "historical_metric_authority": str(binding["historical_metric_authority"]),
        "checkpoint_embedded_metrics_fallback_allowed": False,
        "verified_before_model_load": True,
        "verified_after_inference": True,
    }


def validate_analyzer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema") == ANALYZER_SCHEMA, "DORF analyzer schema differs")
    _require(payload.get("status") == "complete", "DORF analyzer is incomplete")
    method = payload.get("method")
    dataset = payload.get("dataset")
    role = payload.get("checkpoint_role")
    _require(method in METHODS, "DORF method differs")
    _require(dataset in DATASETS, "DORF dataset differs")
    _require(role in CHECKPOINT_ROLES, "DORF checkpoint role differs")
    _require(payload.get("seed") == SEED, "DORF seed differs")
    _require(payload.get("test_selected") is True, "DORF test-selected scope differs")
    _require(
        payload.get("selection_is_optimistic") is True,
        "DORF selection-scope declaration differs",
    )
    _require(tuple(payload.get("mode_order", ())) == MODES, "DORF mode order differs")
    model_metadata = _mapping(payload.get("model_metadata"), "model_metadata")
    _require(bool(model_metadata), "model metadata is empty")
    loader_audit = _mapping(model_metadata.get("dorf_loader_audit"), "dorf_loader_audit")
    _require(
        loader_audit.get("passed") is True
        and loader_audit.get("builder") == EXPECTED_BUILDERS[str(method)]
        and loader_audit.get("training_state_key_count")
        == EXPECTED_TRAINING_STATE_KEYS[str(method)]
        and loader_audit.get("expected_training_state_key_count")
        == EXPECTED_TRAINING_STATE_KEYS[str(method)]
        and loader_audit.get("removed_training_only_tss_state_key_count")
        == EXPECTED_REMOVED_TSS_KEYS[str(method)]
        and loader_audit.get("inference_state_key_count")
        == EXPECTED_INFERENCE_STATE_KEYS[str(method)]
        and loader_audit.get("strict_load") is True
        and loader_audit.get("training_flag") is False
        and loader_audit.get("mode") == "test",
        "strict loader audit differs",
    )
    manifest_binding = _validate_analyzer_manifest_binding(
        payload.get("input_manifest_binding"),
        method=str(method),
        dataset=str(dataset),
        role=str(role),
    )

    checkpoint_binding = _mapping(payload.get("checkpoint_binding"), "checkpoint_binding")
    _require(
        checkpoint_binding.get("input_manifest_entry_key")
        == _binding_key(str(method), str(dataset), str(role)),
        "checkpoint manifest entry binding differs",
    )
    checkpoint = _mapping(checkpoint_binding.get("checkpoint"), "checkpoint_binding.checkpoint")
    _require(checkpoint.get("role") == role, "nested checkpoint role differs")
    checkpoint_epoch = _nonnegative_int(checkpoint.get("epoch"), "checkpoint epoch")
    _require(checkpoint_epoch > 0, "checkpoint epoch must be positive")
    checkpoint_path = checkpoint.get("path")
    _require(
        isinstance(checkpoint_path, str) and Path(checkpoint_path).is_absolute(),
        "checkpoint path differs",
    )
    checkpoint_sha = checkpoint.get("sha256")
    _require(_is_sha256(checkpoint_sha), "checkpoint SHA differs")
    _require(
        _is_sha256(checkpoint_binding.get("training_state_dict_sha256")),
        "training state-dict SHA differs",
    )
    run_dir = checkpoint_binding.get("run_dir")
    _require(isinstance(run_dir, str) and Path(run_dir).is_absolute(), "run_dir differs")
    summary_binding = _mapping(checkpoint_binding.get("summary"), "checkpoint_binding.summary")
    protocol_binding = _mapping(checkpoint_binding.get("protocol"), "checkpoint_binding.protocol")
    for artifact, label in (
        (summary_binding, "summary"),
        (protocol_binding, "protocol"),
    ):
        _require(
            isinstance(artifact.get("path"), str)
            and Path(str(artifact["path"])).is_absolute(),
            f"{label} binding path differs",
        )
        _require(_is_sha256(artifact.get("sha256")), f"{label} binding SHA differs")
    reference = _mapping(
        payload.get("reference_evaluation_binding"), "reference_evaluation_binding"
    )
    _require(reference.get("checkpoint_role") == role, "reference checkpoint role differs")
    _require(isinstance(reference.get("path"), str) and bool(reference.get("path")),
             "reference evaluation path differs")
    reference_sha = reference.get("sha256")
    _require(_is_sha256(reference_sha), "reference evaluation SHA differs")
    _require(
        reference.get("source") == "historical_evaluation_fixed_threshold_0_5",
        "reference evaluation is not the bound historical authority",
    )
    _require(
        reference.get("checkpoint_embedded_metrics_fallback_allowed") is False,
        "reference evaluation enabled checkpoint metric fallback",
    )
    source_sha = _validate_sha_map(payload.get("source_sha256"), "source_sha256")
    data = _mapping(payload.get("data"), "data")
    _require(data.get("split") == "img_idx/test", "DORF data split differs")
    protocol_manifest = _mapping(data.get("protocol_manifest"), "data.protocol_manifest")
    _require(_is_sha256(protocol_manifest.get("sha256")), "protocol manifest SHA differs")
    input_binding = _mapping(data.get("input_binding"), "data.input_binding")
    _require(_is_sha256(input_binding.get("sha256")), "ordered input SHA differs")
    _require(
        _is_sha256(input_binding.get("ordered_ids_newline_sha256")),
        "ordered-ID SHA differs",
    )

    intervention = _mapping(payload.get("intervention_contract"), "intervention_contract")
    _require(
        intervention.get("family")
        == "DORF_V1_existing_deep_supervision_readout_reuse",
        "DORF intervention family differs",
    )
    _require(
        intervention.get("formula") == "z_out + alpha * (z_d0 - z_out)"
        and intervention.get("fusion_space") == "raw_logits_before_sigmoid",
        "DORF fusion formula/space differs",
    )
    _require(
        tuple(intervention.get("alphas", ()))
        == tuple(ALPHA_BY_MODE[mode] for mode in MODES),
        "DORF intervention alpha grid differs",
    )
    for field in (
        "model_parameters_changed",
        "persistent_buffers_changed",
        "derived_checkpoint_written",
    ):
        _require(intervention.get(field) is False, f"DORF intervention changed {field}")
    _require(
        intervention.get("one_checkpoint_per_unit") is True,
        "DORF checkpoint reuse contract differs",
    )
    _require(
        payload.get("derived_checkpoint_written") is False
        and payload.get("probability_cache_written") is False,
        "DORF analyzer reports forbidden output artifacts",
    )

    replay = _mapping(
        payload.get("alpha0_historical_replay_audit"),
        "alpha0_historical_replay_audit",
    )
    _require(replay.get("mode") == CURRENT_MODE, "alpha0 replay mode differs")
    _require(replay.get("checkpoint_role") == role, "alpha0 replay role differs")
    _require(
        replay.get("reference_evaluation_sha256") == reference_sha,
        "alpha0 replay reference SHA differs",
    )
    _require(
        replay.get("background_pixel_authority_sha256")
        == manifest_binding["background_pixel_authority"]["sha256"],
        "alpha0 replay background authority SHA differs",
    )
    for field in (
        "passed",
        "counts_exact",
        "background_false_positive_pixels_exact",
        "within_frozen_float_tolerances",
        "exact",
    ):
        _require(isinstance(replay.get(field), bool), f"alpha0 replay {field} flag differs")
    alpha0_passed = bool(
        replay["counts_exact"]
        and replay["background_false_positive_pixels_exact"]
        and replay["within_frozen_float_tolerances"]
    )
    _require(replay["passed"] is alpha0_passed, "alpha0 replay aggregate flag differs")
    alpha0_bitwise_exact = bool(replay["exact"])

    background_record = _mapping(
        payload.get("background_pixel_authority_record"),
        "background_pixel_authority_record",
    )
    manifest_entry = manifest_binding["entry"]
    _require(
        background_record.get("dataset") == dataset
        and background_record.get("checkpoint_role") == role
        and background_record.get("checkpoint_epoch") == manifest_entry["epoch"]
        and background_record.get("checkpoint_sha256")
        == manifest_entry["checkpoint_sha256"]
        and background_record.get("evaluation_sha256")
        == manifest_entry["evaluation_sha256"],
        "background authority record identity differs",
    )
    authority_background_fp = _nonnegative_int(
        background_record.get("false_positive_pixels"),
        "background authority false_positive_pixels",
    )
    authority_valid_pixels = _nonnegative_int(
        background_record.get("valid_pixel_count"),
        "background authority valid_pixel_count",
    )

    engineering = _mapping(payload.get("engineering_audit"), "engineering_audit")
    expected_engineering = {
        "passed": True,
        "all_metrics_finite": True,
        "same_d0_out_logits_reused_for_all_modes": True,
        "one_model_forward_per_batch": True,
        "raw_logit_fusion": True,
        "model_state_unchanged": True,
        "model_training_flag_unchanged": True,
        "model_mode_unchanged": True,
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
    }
    engineering_checks: dict[str, bool] = {}
    for field, expected in expected_engineering.items():
        observed = engineering.get(field)
        _require(isinstance(observed, bool), f"engineering_audit.{field} must be bool")
        engineering_checks[field] = observed is expected
    engineering_valid = all(engineering_checks.values())
    batch_count = _nonnegative_int(
        engineering.get("batch_count"), "engineering_audit.batch_count"
    )
    _require(batch_count > 0, "engineering batch count must be positive")
    engineering_count_checks = {
        "model_forward_count": engineering.get("model_forward_count") == batch_count,
        "outc_hook_count": engineering.get("outc_hook_count") == batch_count,
        "outconv_hook_count": engineering.get("outconv_hook_count") == batch_count,
        "each_hook_exactly_once_per_batch": engineering.get(
            "each_hook_exactly_once_per_batch"
        )
        is True,
        "returned_probability_equals_sigmoid_raw_out_bitwise": engineering.get(
            "returned_probability_equals_sigmoid_raw_out_bitwise"
        )
        is True,
        "temporary_hooks_restored": engineering.get("temporary_hooks_restored") is True,
        "source_sha256_reverified_after_inference": engineering.get(
            "source_sha256_reverified_after_inference"
        )
        is True,
        "input_manifest_reverified_after_inference": engineering.get(
            "input_manifest_reverified_after_inference"
        )
        is True,
        "alpha0_historical_replay_passed": engineering.get(
            "alpha0_historical_replay_passed"
        )
        is True,
    }
    engineering_checks.update(engineering_count_checks)
    engineering_checks.update(
        {
            "model_training_flag_before_is_eval": engineering.get(
                "model_training_flag_before"
            )
            is False,
            "model_training_flag_after_is_eval": engineering.get(
                "model_training_flag_after"
            )
            is False,
            "model_mode_before_is_test": engineering.get("model_mode_before") == "test",
            "model_mode_after_is_test": engineering.get("model_mode_after") == "test",
        }
    )
    state_before = engineering.get("model_state_sha256_before")
    state_after = engineering.get("model_state_sha256_after")
    engineering_checks["model_state_sha256_restored"] = bool(
        _is_sha256(state_before) and state_before == state_after
    )
    _require(
        engineering.get("alpha0_historical_replay_passed") is alpha0_passed,
        "engineering/replay aggregate flags differ",
    )
    engineering_valid = all(engineering_checks.values())
    _require(
        input_binding.get("sample_count") == batch_count,
        "input sample count differs from engineering batch count",
    )

    serialized_modes = _mapping(payload.get("modes"), "modes")
    _require(set(serialized_modes) == set(MODES), "DORF mode set differs")
    modes: dict[str, Any] = {}
    invariant: tuple[int, int, int, int | None] | None = None
    for mode in MODES:
        raw_mode = _mapping(serialized_modes[mode], f"modes.{mode}")
        _require(raw_mode.get("mode") == mode, f"modes.{mode}.mode differs")
        alpha = _finite(raw_mode.get("alpha"), f"modes.{mode}.alpha")
        _require(alpha == ALPHA_BY_MODE[mode], f"modes.{mode}.alpha differs")
        point = _extract_point(raw_mode, f"modes.{mode}")
        point_invariant = (
            point["target_count"],
            point["tiny_target_count"],
            point["valid_pixel_count"],
            point.get("image_count"),
        )
        invariant = invariant or point_invariant
        _require(point_invariant == invariant, f"modes.{mode} evaluation totals differ")
        probability = _extract_probability_difference(
            raw_mode, f"modes.{mode}", point["valid_pixel_count"]
        )
        if mode == CURRENT_MODE:
            _require(
                probability["max_abs"] == 0.0
                and probability["mean_abs"] == 0.0
                and probability["absolute_difference_sum"] == 0.0,
                "current_out probability self-replay differs",
            )
        _validate_descriptive_sweep(raw_mode, f"modes.{mode}")
        modes[mode] = {
            "alpha": alpha,
            "fixed_threshold_0_5": point,
            "probability_difference_to_current": probability,
        }
    current_point = modes[CURRENT_MODE]["fixed_threshold_0_5"]
    _require(
        current_point["background_false_positive_pixels"] == authority_background_fp,
        "current_out background FP differs from bound authority",
    )
    _require(
        current_point["valid_pixel_count"] == authority_valid_pixels,
        "current_out valid pixels differ from bound authority",
    )
    return {
        "method": method,
        "dataset": dataset,
        "checkpoint_role": role,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "summary_binding": dict(summary_binding),
        "protocol_binding": dict(protocol_binding),
        "reference_evaluation_sha256": reference_sha,
        "reference_evaluation_path": str(reference["path"]),
        "input_manifest_binding": manifest_binding,
        "source_sha256": source_sha,
        "alpha0_historical_replay_passed": alpha0_passed,
        "alpha0_historical_replay_bitwise_exact": alpha0_bitwise_exact,
        "engineering_valid": engineering_valid,
        "engineering_checks": engineering_checks,
        "evaluation_invariant": invariant,
        "data_protocol_manifest": {
            "path": str(protocol_manifest.get("path", "")),
            "sha256": str(protocol_manifest["sha256"]),
        },
        "modes": modes,
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


def compare_direction(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the complete frozen safe/material/severe relation."""

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
        "component_fp_reduction_ge_0_05": component["value"] is not None
        and component["value"] >= MATERIAL_FP_REDUCTION_MINIMUM,
        "background_pixel_fp_reduction_ge_0_05": background["value"] is not None
        and background["value"] >= MATERIAL_FP_REDUCTION_MINIMUM,
    }
    severe_conditions = {
        "delta_target_le_minus_2": delta_target <= SEVERE_COUNT_DELTA_MAXIMUM,
        "delta_tiny_le_minus_2": delta_tiny <= SEVERE_COUNT_DELTA_MAXIMUM,
        "delta_miou_le_minus_0_01": delta_miou <= SEVERE_IOU_DELTA_MAXIMUM,
        "delta_niou_le_minus_0_01": delta_niou <= SEVERE_IOU_DELTA_MAXIMUM,
        "component_fp_increase_ge_25pct": component["severe_degradation"],
        "background_pixel_fp_increase_ge_25pct": background["severe_degradation"],
    }
    _require(
        tuple(severe_conditions) == SEVERE_CONDITION_ORDER,
        "internal severe condition order differs from the frozen contract",
    )
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


def _binding_key(method: str, dataset: str, role: str) -> str:
    return f"{method}::{dataset}::{role}"


def _expected_keys() -> tuple[str, ...]:
    return tuple(
        _binding_key(method, dataset, role)
        for method in METHODS
        for dataset in DATASETS
        for role in CHECKPOINT_ROLES
    )


def _validate_input_manifest(
    raw_manifest: Mapping[str, Any],
    raw_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one preregistered twelve-role artifact manifest.

    The CLI computes the binding SHA from the raw manifest bytes before this
    function is called.  Keeping payload validation separate makes the same
    literal contract testable without consulting mutable result directories.
    """

    manifest = _mapping(raw_manifest, "input_manifest")
    binding = _mapping(raw_binding, "input_manifest_binding")
    _require(
        binding.get("path") == str(DEFAULT_INPUT_MANIFEST.resolve()),
        "input manifest path differs from the frozen path",
    )
    _require(
        binding.get("sha256") == FROZEN_INPUT_MANIFEST_SHA256,
        "input manifest SHA differs from the frozen SHA",
    )
    _require(manifest.get("schema") == INPUT_MANIFEST_SCHEMA, "input manifest schema differs")
    _require(
        manifest.get("status") == "frozen_before_dorf_outputs",
        "input manifest status differs",
    )
    _require(manifest.get("seed") == SEED, "input manifest seed differs")
    _require(tuple(manifest.get("method_order", ())) == METHODS, "manifest method order differs")
    _require(tuple(manifest.get("dataset_order", ())) == DATASETS, "manifest dataset order differs")
    _require(
        tuple(manifest.get("checkpoint_role_order", ())) == CHECKPOINT_ROLES,
        "manifest checkpoint role order differs",
    )

    authority = _mapping(manifest.get("authority"), "input_manifest.authority")
    _require(
        authority.get("historical_fixed_threshold_source")
        == "bound_evaluation_json_only",
        "historical metric authority differs",
    )
    _require(
        authority.get("checkpoint_embedded_metrics_fallback_allowed") is False,
        "checkpoint metric fallback was enabled",
    )
    _require(authority.get("input_count") == 12, "input manifest count differs")
    _require(
        authority.get("outputs_exist_when_frozen") is False,
        "input manifest was not frozen before DORF outputs",
    )

    protocol = _mapping(
        manifest.get("data_protocol_manifest"),
        "input_manifest.data_protocol_manifest",
    )
    background = _mapping(
        manifest.get("background_pixel_authority"),
        "input_manifest.background_pixel_authority",
    )
    for item, label in ((protocol, "data protocol"), (background, "background authority")):
        _require(isinstance(item.get("path"), str) and bool(item.get("path")), f"{label} path differs")
        _require(_is_sha256(item.get("sha256")), f"{label} SHA differs")
    _require(
        background.get("required_record_count_for_bound_inputs") == 12,
        "background authority record count differs",
    )

    loaders = _mapping(manifest.get("loader_contract"), "input_manifest.loader_contract")
    _require(set(loaders) == set(METHODS), "input manifest loader set differs")
    final_loader = _mapping(loaders[PRIMARY_METHOD], "final loader contract")
    _require(
        final_loader.get("training_state_key_count") == 568
        and final_loader.get("removed_training_only_tss_state_keys") == 4
        and final_loader.get("inference_state_key_count") == 564
        and final_loader.get("builder")
        == "build_final_inference_model_from_training_state_dict"
        and final_loader.get("strict_load") is True,
        "Final loader contract differs",
    )
    original_loader = _mapping(loaders[BASELINE_METHOD], "Original loader contract")
    _require(
        original_loader.get("training_state_key_count") == 510
        and original_loader.get("inference_state_key_count") == 510
        and original_loader.get("builder") == "build_paper_model_original_then_strict_load"
        and original_loader.get("strict_load") is True,
        "Original loader contract differs",
    )

    entries = list(_sequence(manifest.get("entries"), "input_manifest.entries"))
    _require(len(entries) == 12, "input manifest must contain twelve entries")
    normalized_entries: dict[str, Any] = {}
    observed_order: list[str] = []
    for raw_entry in entries:
        entry = _mapping(raw_entry, "input_manifest entry")
        method = entry.get("method")
        dataset = entry.get("dataset")
        role = entry.get("checkpoint_role")
        _require(method in METHODS, "manifest entry method differs")
        _require(dataset in DATASETS, "manifest entry dataset differs")
        _require(role in CHECKPOINT_ROLES, "manifest entry checkpoint role differs")
        key = _binding_key(str(method), str(dataset), str(role))
        _require(key not in normalized_entries, f"duplicate manifest entry: {key}")
        observed_order.append(key)
        epoch = _nonnegative_int(entry.get("epoch"), f"manifest.{key}.epoch")
        _require(epoch > 0, f"manifest.{key}.epoch must be positive")
        run_dir = entry.get("run_dir")
        _require(
            isinstance(run_dir, str)
            and bool(run_dir)
            and not Path(run_dir).is_absolute()
            and ".." not in Path(run_dir).parts,
            f"manifest.{key}.run_dir differs",
        )
        for field in (
            "summary_sha256",
            "protocol_sha256",
            "checkpoint_sha256",
            "evaluation_sha256",
        ):
            _require(_is_sha256(entry.get(field)), f"manifest.{key}.{field} differs")
        normalized_entries[key] = {
            "method": str(method),
            "dataset": str(dataset),
            "checkpoint_role": str(role),
            "epoch": epoch,
            "run_dir": str(run_dir),
            "summary_sha256": str(entry["summary_sha256"]),
            "protocol_sha256": str(entry["protocol_sha256"]),
            "checkpoint_sha256": str(entry["checkpoint_sha256"]),
            "evaluation_sha256": str(entry["evaluation_sha256"]),
        }
    _require(tuple(observed_order) == _expected_keys(), "input manifest entry order differs")
    return {
        "path": str(binding["path"]),
        "sha256": str(binding["sha256"]),
        "schema": INPUT_MANIFEST_SCHEMA,
        "status": "frozen_before_dorf_outputs",
        "entries": normalized_entries,
        "data_protocol_manifest": {
            "path": str(protocol["path"]),
            "sha256": str(protocol["sha256"]),
        },
        "background_pixel_authority": {
            "path": str(background["path"]),
            "sha256": str(background["sha256"]),
        },
        "historical_metric_authority": "bound_evaluation_json_only",
        "checkpoint_embedded_metrics_fallback_allowed": False,
    }


def _validate_input_bindings(raw: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    _require(set(raw) == set(_expected_keys()), "input bindings require twelve roles")
    ready: dict[str, Any] = {}
    for key in _expected_keys():
        binding = _mapping(raw[key], f"input_bindings.{key}")
        path = binding.get("path")
        sha = binding.get("sha256")
        _require(isinstance(path, str) and bool(path), f"input path differs: {key}")
        _require(_is_sha256(sha), f"input SHA differs: {key}")
        ready[key] = {"path": str(path), "sha256": str(sha)}
    return ready


def _severe_condition_mask(units: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    mask: dict[str, bool] = {}
    for dataset in DATASETS:
        for role in CHECKPOINT_ROLES:
            unit_key = f"{dataset}::{role}"
            direction = _mapping(units.get(unit_key), f"competitiveness.{unit_key}")
            conditions = _mapping(
                direction.get("severe_degradation_conditions"),
                f"competitiveness.{unit_key}.severe_degradation_conditions",
            )
            _require(
                tuple(conditions) == SEVERE_CONDITION_ORDER,
                f"severe condition mask order differs: {unit_key}",
            )
            for condition in SEVERE_CONDITION_ORDER:
                value = conditions[condition]
                _require(isinstance(value, bool), f"severe condition is not bool: {unit_key}/{condition}")
                mask[f"{unit_key}::{condition}"] = value
    return mask


def _newly_true_conditions(
    candidate_mask: Mapping[str, bool],
    baseline_mask: Mapping[str, bool],
) -> list[str]:
    _require(set(candidate_mask) == set(baseline_mask), "severe mask dimensions differ")
    return [
        key
        for key in candidate_mask
        if candidate_mask[key] and not baseline_mask[key]
    ]


def _competitiveness_anchor(
    normalized: Mapping[str, Mapping[str, Any]],
    *,
    final_mode: str,
    original_mode: str,
) -> dict[str, Any]:
    _require(final_mode in MODES and original_mode in MODES, "unknown competitiveness mode")
    units: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for role in CHECKPOINT_ROLES:
            final_point = normalized[_binding_key(PRIMARY_METHOD, dataset, role)]["modes"][
                final_mode
            ]["fixed_threshold_0_5"]
            original_point = normalized[_binding_key(BASELINE_METHOD, dataset, role)]["modes"][
                original_mode
            ]["fixed_threshold_0_5"]
            direction = compare_direction(final_point, original_point)
            units[f"{dataset}::{role}"] = direction
            rows.append(direction)
    mask = _severe_condition_mask(units)
    true_keys = [key for key, value in mask.items() if value]
    primary_positive = [
        dataset
        for dataset in DATASETS
        if units[f"{dataset}::{PRIMARY_ROLE}"]["safe_material_improvement"]
    ]
    return {
        "final_mode": final_mode,
        "final_alpha": ALPHA_BY_MODE[final_mode],
        "original_mode": original_mode,
        "original_alpha": ALPHA_BY_MODE[original_mode],
        "final_vs_original_units": units,
        "severe_condition_mask": mask,
        "severe_condition_true_keys": true_keys,
        "severe_condition_true_count": len(true_keys),
        "severe_unit_count": sum(bool(row["severe_degradation"]) for row in rows),
        "primary_best_miou_safe_material_datasets": primary_positive,
    }


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    input_bindings: Mapping[str, Mapping[str, str]],
    input_manifest: Mapping[str, Any],
    input_manifest_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(payloads) == set(_expected_keys()), "comparison requires twelve payloads")
    bindings = _validate_input_bindings(input_bindings)
    manifest_contract = _validate_input_manifest(input_manifest, input_manifest_binding)
    normalized: dict[str, Any] = {}
    for key in _expected_keys():
        unit = validate_analyzer_payload(payloads[key])
        method, dataset, role = key.split("::", 2)
        _require(
            (unit["method"], unit["dataset"], unit["checkpoint_role"])
            == (method, dataset, role),
            f"input identity differs: {key}",
        )
        normalized[key] = unit

    for key in _expected_keys():
        unit = normalized[key]
        entry = manifest_contract["entries"][key]
        analyzer_manifest = unit["input_manifest_binding"]
        _require(analyzer_manifest["entry"] == entry, f"analyzer manifest entry differs: {key}")
        _require(
            analyzer_manifest["sha256"] == manifest_contract["sha256"]
            and analyzer_manifest["path"] == manifest_contract["path"],
            f"analyzer manifest identity differs: {key}",
        )
        expected_run_dir = (REPO_ROOT / entry["run_dir"]).resolve()
        expected_checkpoint = expected_run_dir / "checkpoints" / f"{entry['checkpoint_role']}.pth.tar"
        expected_evaluation = expected_run_dir / "evaluations" / f"{entry['checkpoint_role']}.json"
        expected_summary = expected_run_dir / "summary.json"
        expected_protocol = expected_run_dir / "protocol.json"
        _require(unit["checkpoint_epoch"] == entry["epoch"], f"checkpoint epoch differs: {key}")
        _require(
            unit["checkpoint_path"] == str(expected_checkpoint)
            and unit["checkpoint_sha256"] == entry["checkpoint_sha256"],
            f"checkpoint prebinding differs: {key}",
        )
        _require(unit["run_dir"] == str(expected_run_dir), f"run-dir prebinding differs: {key}")
        _require(
            unit["reference_evaluation_path"] == str(expected_evaluation)
            and unit["reference_evaluation_sha256"] == entry["evaluation_sha256"],
            f"historical evaluation prebinding differs: {key}",
        )
        _require(
            unit["summary_binding"].get("path") == str(expected_summary)
            and unit["summary_binding"].get("sha256") == entry["summary_sha256"],
            f"summary prebinding differs: {key}",
        )
        _require(
            unit["protocol_binding"].get("path") == str(expected_protocol)
            and unit["protocol_binding"].get("sha256") == entry["protocol_sha256"],
            f"run protocol prebinding differs: {key}",
        )
        expected_data_protocol = manifest_contract["data_protocol_manifest"]
        analyzer_data_protocol = analyzer_manifest["data_protocol_manifest"]
        _require(
            analyzer_data_protocol.get("path")
            == str((REPO_ROOT / expected_data_protocol["path"]).resolve())
            and analyzer_data_protocol.get("sha256") == expected_data_protocol["sha256"],
            f"analyzer data protocol prebinding differs: {key}",
        )
        _require(
            unit["data_protocol_manifest"]["path"]
            == str((REPO_ROOT / expected_data_protocol["path"]).resolve())
            and unit["data_protocol_manifest"]["sha256"]
            == expected_data_protocol["sha256"],
            f"evaluation data protocol prebinding differs: {key}",
        )
        expected_background = manifest_contract["background_pixel_authority"]
        analyzer_background = analyzer_manifest["background_pixel_authority"]
        _require(
            analyzer_background.get("path")
            == str((REPO_ROOT / expected_background["path"]).resolve())
            and analyzer_background.get("sha256") == expected_background["sha256"],
            f"background-pixel authority prebinding differs: {key}",
        )

    source_contracts = {canonical_sha256(unit["source_sha256"]) for unit in normalized.values()}
    _require(len(source_contracts) == 1, "twelve inputs do not share one analyzer source contract")
    for dataset in DATASETS:
        for role in CHECKPOINT_ROLES:
            invariants = {
                normalized[_binding_key(method, dataset, role)]["evaluation_invariant"]
                for method in METHODS
            }
            _require(len(invariants) == 1, f"Final/Original evaluation totals differ: {dataset}/{role}")

    all_engineering_valid = all(unit["engineering_valid"] for unit in normalized.values())
    all_alpha0_replay_passed = all(
        unit["alpha0_historical_replay_passed"] for unit in normalized.values()
    )
    all_alpha0_bitwise_exact = all(
        unit["alpha0_historical_replay_bitwise_exact"] for unit in normalized.values()
    )
    per_unit: dict[str, Any] = {}
    own_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        method: {
            mode: {role: {} for role in CHECKPOINT_ROLES}
            for mode in NONZERO_MODES
        }
        for method in METHODS
    }
    for key in _expected_keys():
        unit = normalized[key]
        current = unit["modes"][CURRENT_MODE]["fixed_threshold_0_5"]
        comparisons: dict[str, Any] = {}
        for mode in NONZERO_MODES:
            candidate = unit["modes"][mode]["fixed_threshold_0_5"]
            direction = compare_direction(candidate, current)
            own_rows[unit["method"]][mode][unit["checkpoint_role"]][unit["dataset"]] = direction
            comparisons[mode] = {
                "candidate_vs_current": direction,
                "probability_difference_to_current": unit["modes"][mode][
                    "probability_difference_to_current"
                ],
            }
        per_unit[key] = {
            "method": unit["method"],
            "dataset": unit["dataset"],
            "checkpoint_role": unit["checkpoint_role"],
            "checkpoint_sha256": unit["checkpoint_sha256"],
            "reference_evaluation_sha256": unit["reference_evaluation_sha256"],
            "engineering_valid": unit["engineering_valid"],
            "engineering_checks": unit["engineering_checks"],
            "alpha0_historical_replay_passed": unit["alpha0_historical_replay_passed"],
            "alpha0_historical_replay_bitwise_exact": unit[
                "alpha0_historical_replay_bitwise_exact"
            ],
            "current_fixed_threshold_0_5": current,
            "modes": comparisons,
        }

    baseline_competitiveness = _competitiveness_anchor(
        normalized,
        final_mode=CURRENT_MODE,
        original_mode=CURRENT_MODE,
    )
    baseline_severe_mask = baseline_competitiveness["severe_condition_mask"]
    baseline_primary_positive = baseline_competitiveness[
        "primary_best_miou_safe_material_datasets"
    ]
    candidate_anchors: dict[str, Any] = {}
    for mode in NONZERO_MODES:
        candidate_anchors[mode] = {
            "final_alpha_vs_original_zero": _competitiveness_anchor(
                normalized,
                final_mode=mode,
                original_mode=CURRENT_MODE,
            ),
            "final_alpha_vs_original_alpha": _competitiveness_anchor(
                normalized,
                final_mode=mode,
                original_mode=mode,
            ),
        }

    trigger_modes: dict[str, Any] = {}
    qualifying_modes: list[str] = []
    for mode in NONZERO_MODES:
        primary_rows = own_rows[PRIMARY_METHOD][mode][PRIMARY_ROLE]
        safe_material_datasets = [
            dataset
            for dataset in DATASETS
            if primary_rows[dataset]["safe_material_improvement"]
        ]
        severe_units = [
            f"{dataset}::{role}"
            for dataset in DATASETS
            for role in CHECKPOINT_ROLES
            if own_rows[PRIMARY_METHOD][mode][role][dataset]["severe_degradation"]
        ]
        fixed_original_anchor = candidate_anchors[mode][
            "final_alpha_vs_original_zero"
        ]
        shared_alpha_anchor = candidate_anchors[mode][
            "final_alpha_vs_original_alpha"
        ]
        fixed_original_new_severe = _newly_true_conditions(
            fixed_original_anchor["severe_condition_mask"], baseline_severe_mask
        )
        shared_alpha_new_severe = _newly_true_conditions(
            shared_alpha_anchor["severe_condition_mask"], baseline_severe_mask
        )
        fixed_original_missing_primary = [
            dataset
            for dataset in baseline_primary_positive
            if dataset
            not in fixed_original_anchor["primary_best_miou_safe_material_datasets"]
        ]
        shared_alpha_missing_primary = [
            dataset
            for dataset in baseline_primary_positive
            if dataset
            not in shared_alpha_anchor["primary_best_miou_safe_material_datasets"]
        ]
        fixed_original_mask_subset = not fixed_original_new_severe
        shared_alpha_mask_subset = not shared_alpha_new_severe
        fixed_original_primary_preserved = not fixed_original_missing_primary
        shared_alpha_primary_preserved = not shared_alpha_missing_primary
        competitiveness_non_degrading = bool(
            fixed_original_mask_subset
            and shared_alpha_mask_subset
            and fixed_original_primary_preserved
            and shared_alpha_primary_preserved
        )
        passed = bool(
            len(safe_material_datasets) >= REQUIRED_PRIMARY_SAFE_MATERIAL_DATASETS
            and not severe_units
            and competitiveness_non_degrading
            and all_alpha0_replay_passed
            and all_engineering_valid
        )
        if passed:
            qualifying_modes.append(mode)
        trigger_modes[mode] = {
            "alpha": ALPHA_BY_MODE[mode],
            "final_primary_safe_material_datasets": safe_material_datasets,
            "final_primary_safe_material_dataset_count": len(safe_material_datasets),
            "required_final_primary_safe_material_dataset_count": 2,
            "final_severe_units_across_six_roles": severe_units,
            "final_severe_unit_count": len(severe_units),
            "baseline_final_zero_vs_original_zero_severe_condition_true_count": (
                baseline_competitiveness["severe_condition_true_count"]
            ),
            "final_alpha_vs_original_zero_severe_condition_true_count": (
                fixed_original_anchor["severe_condition_true_count"]
            ),
            "final_alpha_vs_original_alpha_severe_condition_true_count": (
                shared_alpha_anchor["severe_condition_true_count"]
            ),
            "final_alpha_vs_original_zero_new_severe_conditions": fixed_original_new_severe,
            "final_alpha_vs_original_zero_new_severe_condition_count": len(
                fixed_original_new_severe
            ),
            "final_alpha_vs_original_alpha_new_severe_conditions": shared_alpha_new_severe,
            "final_alpha_vs_original_alpha_new_severe_condition_count": len(
                shared_alpha_new_severe
            ),
            "final_alpha_vs_original_zero_severe_mask_subset_of_baseline": (
                fixed_original_mask_subset
            ),
            "final_alpha_vs_original_alpha_severe_mask_subset_of_baseline": (
                shared_alpha_mask_subset
            ),
            "baseline_primary_best_miou_safe_material_datasets": list(
                baseline_primary_positive
            ),
            "final_alpha_vs_original_zero_missing_primary_safe_material_datasets": (
                fixed_original_missing_primary
            ),
            "final_alpha_vs_original_alpha_missing_primary_safe_material_datasets": (
                shared_alpha_missing_primary
            ),
            "final_alpha_vs_original_zero_primary_safe_material_preserved": (
                fixed_original_primary_preserved
            ),
            "final_alpha_vs_original_alpha_primary_safe_material_preserved": (
                shared_alpha_primary_preserved
            ),
            "final_vs_original_competitiveness_non_degrading": competitiveness_non_degrading,
            "all_twelve_alpha0_historical_replay_passed": all_alpha0_replay_passed,
            "all_twelve_engineering_valid": all_engineering_valid,
            "trigger_a_passed": passed,
        }
    selected_mode = min(
        qualifying_modes,
        key=lambda mode: (abs(ALPHA_BY_MODE[mode]), ALPHA_BY_MODE[mode]),
    ) if qualifying_modes else None
    selected_alpha = None if selected_mode is None else ALPHA_BY_MODE[selected_mode]
    trigger_passed = selected_mode is not None
    decision = DECISION_AUTHORIZE if trigger_passed else DECISION_NO_AUTHORIZATION
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": decision,
        "seed": SEED,
        "methods": list(METHODS),
        "datasets": list(DATASETS),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "mode_order": list(MODES),
        "alpha_by_mode": dict(ALPHA_BY_MODE),
        "fixed_threshold": FIXED_THRESHOLD,
        "trigger_a": {
            "implemented": True,
            "passed": trigger_passed,
            "same_alpha_applied_to_final_and_original": True,
            "best_miou_primary": True,
            "best_pd_severe_veto_only": True,
            "all_twelve_engineering_valid": all_engineering_valid,
            "all_twelve_alpha0_historical_replay_passed": all_alpha0_replay_passed,
            "all_twelve_alpha0_bitwise_exact_descriptive_only": all_alpha0_bitwise_exact,
            "qualifying_modes": qualifying_modes,
            "selected_mode": selected_mode,
            "selected_alpha": selected_alpha,
            "selection_rule": "minimum_absolute_nonzero_alpha_then_numeric_alpha",
            "modes": trigger_modes,
        },
        "competitiveness": {
            "baseline_final_zero_vs_original_zero": baseline_competitiveness,
            "candidate_anchors": candidate_anchors,
            "severe_mask_gate": "candidate_mask_subset_of_baseline_mask",
            "primary_positive_preservation_gate": (
                "baseline_best_miou_safe_material_datasets_preserved_under_both_anchors"
            ),
        },
        "original_own_dorf_gain_descriptive_only": {
            mode: own_rows[BASELINE_METHOD][mode] for mode in NONZERO_MODES
        },
        "dorf_v1_production_implementation_authorized": trigger_passed,
        "fresh_formal1000_after_production_engineering_gate": trigger_passed,
        "fresh_formal1000_launch_authorized_by_this_comparator": False,
        "training_loss_changed": False,
        "model_mainline_changed": False,
        "per_unit": per_unit,
        "input_bindings": bindings,
        "input_manifest_binding": {
            key: value
            for key, value in manifest_contract.items()
            if key != "entries"
        },
        "frozen_artifact_prebindings": manifest_contract["entries"],
        "threshold_contract": {
            "safe": {"count_delta": ">-2", "iou_delta": ">-0.005", "fp_reduction": ">-0.05"},
            "material_any": {"count_delta": ">=2", "iou_delta": ">=0.005", "fp_reduction": ">=0.05"},
            "severe_any": {"count_delta": "<=-2", "iou_delta": "<=-0.01", "fp_reduction": "<=-0.25", "fp_introduced_from_zero": True},
        },
        "scope": {
            "single_seed_test_selected_zero_training_diagnostic": True,
            "original_gain_used_for_final_primary_gate": False,
            "stability_claim_supported": False,
            "performance_claim_established": False,
        },
        "source_sha256": {
            "analysis/compare_three_dataset_dorf_v1.py": file_sha256(Path(__file__).resolve())
        },
    }


def validate_comparison_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "comparison schema differs")
    _require(payload.get("status") == "complete", "comparison is incomplete")
    trigger = _mapping(payload.get("trigger_a"), "trigger_a")
    passed = trigger.get("passed")
    _require(isinstance(passed, bool), "Trigger A result differs")
    expected_decision = DECISION_AUTHORIZE if passed else DECISION_NO_AUTHORIZATION
    _require(payload.get("decision") == expected_decision, "decision differs from Trigger A")
    _require(
        payload.get("dorf_v1_production_implementation_authorized") is passed,
        "production implementation authorization differs",
    )
    _require(
        payload.get("fresh_formal1000_launch_authorized_by_this_comparator") is False,
        "comparator directly authorized formal training launch",
    )
    _require(set(payload.get("input_bindings", {})) == set(_expected_keys()),
             "comparison input bindings differ")
    _require(set(payload.get("per_unit", {})) == set(_expected_keys()),
             "comparison unit matrix differs")
    _require(
        set(payload.get("frozen_artifact_prebindings", {})) == set(_expected_keys()),
        "comparison artifact prebindings differ",
    )
    manifest = _mapping(payload.get("input_manifest_binding"), "input_manifest_binding")
    _require(
        manifest.get("path") == str(DEFAULT_INPUT_MANIFEST.resolve())
        and manifest.get("sha256") == FROZEN_INPUT_MANIFEST_SHA256
        and manifest.get("schema") == INPUT_MANIFEST_SCHEMA
        and manifest.get("status") == "frozen_before_dorf_outputs",
        "comparison input manifest binding differs",
    )
    qualifying = list(_sequence(trigger.get("qualifying_modes"), "qualifying_modes"))
    trigger_modes = _mapping(trigger.get("modes"), "trigger_a.modes")
    _require(tuple(trigger_modes) == NONZERO_MODES, "Trigger A mode order differs")
    recomputed_qualifying: list[str] = []
    for mode in NONZERO_MODES:
        row = _mapping(trigger_modes[mode], f"trigger_a.modes.{mode}")
        row_passed = row.get("trigger_a_passed")
        _require(isinstance(row_passed, bool), f"Trigger A flag differs: {mode}")
        for anchor in (
            "final_alpha_vs_original_zero",
            "final_alpha_vs_original_alpha",
        ):
            new_items = list(
                _sequence(
                    row.get(f"{anchor}_new_severe_conditions"),
                    f"{mode}.{anchor}.new severe conditions",
                )
            )
            _require(
                row.get(f"{anchor}_new_severe_condition_count") == len(new_items),
                f"new severe condition count differs: {mode}/{anchor}",
            )
            _require(
                row.get(f"{anchor}_severe_mask_subset_of_baseline") is (not new_items),
                f"severe mask subset flag differs: {mode}/{anchor}",
            )
            missing = list(
                _sequence(
                    row.get(f"{anchor}_missing_primary_safe_material_datasets"),
                    f"{mode}.{anchor}.missing primary cells",
                )
            )
            _require(
                row.get(f"{anchor}_primary_safe_material_preserved") is (not missing),
                f"primary-cell preservation flag differs: {mode}/{anchor}",
            )
        expected_competitiveness = all(
            row.get(field) is True
            for field in (
                "final_alpha_vs_original_zero_severe_mask_subset_of_baseline",
                "final_alpha_vs_original_alpha_severe_mask_subset_of_baseline",
                "final_alpha_vs_original_zero_primary_safe_material_preserved",
                "final_alpha_vs_original_alpha_primary_safe_material_preserved",
            )
        )
        _require(
            row.get("final_vs_original_competitiveness_non_degrading")
            is expected_competitiveness,
            f"competitiveness aggregate differs: {mode}",
        )
        if row_passed:
            recomputed_qualifying.append(mode)
    _require(qualifying == recomputed_qualifying, "qualifying mode list differs")
    selected = trigger.get("selected_mode")
    expected_selected = min(
        qualifying,
        key=lambda mode: (abs(ALPHA_BY_MODE[mode]), ALPHA_BY_MODE[mode]),
    ) if qualifying else None
    _require(selected == expected_selected, "selected alpha is not the smallest passing alpha")
    _require(trigger.get("selected_alpha") == (
        None if selected is None else ALPHA_BY_MODE[selected]
    ), "selected alpha value differs")
    _require(passed is bool(qualifying), "Trigger A aggregate differs from qualifying modes")
    expected_source = {
        "analysis/compare_three_dataset_dorf_v1.py": file_sha256(Path(__file__).resolve())
    }
    _require(payload.get("source_sha256") == expected_source, "comparison source lock differs")


def _format_reduction(value: Mapping[str, Any]) -> str:
    return "introduced-from-zero" if value.get("value") is None else f"{float(value['value']):+.2%}"


def render_markdown(result: Mapping[str, Any]) -> str:
    trigger = result["trigger_a"]
    lines = [
        "# DORF V1 十二角色零训练裁决",
        "",
        f"- decision: `{result['decision']}`",
        f"- Trigger A: `{str(trigger['passed']).lower()}`",
        f"- selected mode: `{trigger['selected_mode']}`",
        f"- selected alpha: `{trigger['selected_alpha']}`",
        "- Final best_miou 为主门；Final best_pd 仅作 severe veto。",
        "- Original 与 Final 使用同一 alpha；Original 自身收益不进入 Final 2/3 主门。",
        "",
        "## Trigger A",
        "",
        "| mode | alpha | Final safe-material | Final severe | new severe Ma0/Maa | primary retained Ma0/Maa | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in NONZERO_MODES:
        row = trigger["modes"][mode]
        lines.append(
            f"| `{mode}` | {row['alpha']:.2f} | "
            f"{row['final_primary_safe_material_dataset_count']}/3 | "
            f"{row['final_severe_unit_count']}/6 | "
            f"{row['final_alpha_vs_original_zero_new_severe_condition_count']}/"
            f"{row['final_alpha_vs_original_alpha_new_severe_condition_count']} | "
            f"{str(row['final_alpha_vs_original_zero_primary_safe_material_preserved']).lower()}/"
            f"{str(row['final_alpha_vs_original_alpha_primary_safe_material_preserved']).lower()} | "
            f"{str(row['trigger_a_passed']).lower()} |"
        )
    lines.extend(["", "## Final 六角色逐模式差值", ""])
    for dataset in DATASETS:
        for role in CHECKPOINT_ROLES:
            key = _binding_key(PRIMARY_METHOD, dataset, role)
            unit = result["per_unit"][key]
            lines.extend([
                f"### {dataset} — {role}",
                "",
                "| mode | Δtarget | Δtiny | ΔmIoU | ΔnIoU | component FP reduction | background FP reduction | safe-material | severe |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for mode in NONZERO_MODES:
                row = unit["modes"][mode]["candidate_vs_current"]
                lines.append(
                    f"| `{mode}` | {row['delta_target']:+d} | {row['delta_tiny']:+d} | "
                    f"{row['delta_miou']:+.6f} | {row['delta_niou']:+.6f} | "
                    f"{_format_reduction(row['component_fp_reduction'])} | "
                    f"{_format_reduction(row['background_pixel_fp_reduction'])} | "
                    f"{str(row['safe_material_improvement']).lower()} | "
                    f"{str(row['severe_degradation']).lower()} |"
                )
            lines.append("")
    lines.extend([
        "## 边界",
        "",
        "本裁决只决定是否实现选定的固定-alpha DORF 生产图。",
        "fresh formal1000 仍需生产图工程门通过后另行启动。",
        "",
    ])
    return "\n".join(lines)


def _default_input(method: str, dataset: str, role: str) -> Path:
    return DEFAULT_INPUT_ROOT / "runs" / method / dataset / role / "evaluation.json"


def _parse_bindings(values: Sequence[str]) -> dict[str, Path]:
    if not values:
        return {
            _binding_key(method, dataset, role): _default_input(method, dataset, role)
            for method in METHODS
            for dataset in DATASETS
            for role in CHECKPOINT_ROLES
        }
    ready: dict[str, Path] = {}
    for value in values:
        _require("=" in value, "--input must use METHOD::DATASET::ROLE=PATH")
        key, raw_path = value.split("=", 1)
        _require(key in _expected_keys(), f"unknown input key: {key}")
        _require(key not in ready, f"duplicate input key: {key}")
        ready[key] = Path(raw_path)
    _require(set(ready) == set(_expected_keys()), "--input must provide all twelve bindings")
    return ready


def _load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    ready = Path(path).resolve(strict=True)
    _require(ready.is_file() and not ready.is_symlink(), f"invalid input file: {ready}")
    raw = ready.read_bytes()
    payload = json.loads(raw)
    _require(isinstance(payload, dict), f"input JSON root differs: {ready}")
    return payload, hashlib.sha256(raw).hexdigest()


def _atomic_write_once(path: Path, text: str) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(json_path: Path, markdown_path: Path, result: Mapping[str, Any]) -> None:
    validate_comparison_payload(result)
    if json_path.exists() or json_path.is_symlink():
        raise FileExistsError(json_path)
    if markdown_path.exists() or markdown_path.is_symlink():
        raise FileExistsError(markdown_path)
    json_text = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
        separators=(",", ": "),
    )
    _atomic_write_once(json_path, json_text)
    _atomic_write_once(markdown_path, render_markdown(result))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.json")
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = _parse_bindings(args.input)
    input_manifest, input_manifest_sha = _load_json_with_sha(DEFAULT_INPUT_MANIFEST)
    _require(
        input_manifest_sha == FROZEN_INPUT_MANIFEST_SHA256,
        "frozen input manifest file SHA differs",
    )
    input_manifest_binding = {
        "path": str(DEFAULT_INPUT_MANIFEST.resolve()),
        "sha256": input_manifest_sha,
    }
    payloads: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, Mapping[str, str]] = {}
    for key in _expected_keys():
        payload, sha = _load_json_with_sha(paths[key])
        payloads[key] = payload
        bindings[key] = {"path": str(paths[key].resolve()), "sha256": sha}
    result = compare_payloads(
        payloads,
        input_bindings=bindings,
        input_manifest=input_manifest,
        input_manifest_binding=input_manifest_binding,
    )
    write_outputs(args.output_json, args.output_markdown, result)
    print(json.dumps({
        "schema": SCHEMA,
        "status": "complete",
        "decision": result["decision"],
        "selected_mode": result["trigger_a"]["selected_mode"],
        "selected_alpha": result["trigger_a"]["selected_alpha"],
        "output_json": str(args.output_json.resolve()),
        "output_markdown": str(args.output_markdown.resolve()),
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
