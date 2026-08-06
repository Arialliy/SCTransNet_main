#!/usr/bin/env python3
"""Compare the six-role, zero-training GCSF branch-audit matrix.

Training authorization is controlled only by quantified Trigger A.  A single
nonzero, representable mode must be safe-material on at least two of three
``best_miou`` checkpoints, and it must have no severe degradation on any of
the six dataset/role units.  ``best_pd`` is therefore a severe-degradation
veto, not a second optimization objective.  Branch statistics are reported
descriptively and cannot authorize training; Triggers B/C are intentionally
not implemented.
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

from analysis import analyze_three_dataset_gcsf_branch_audit_v1 as analyzer  # noqa: E402
from analysis import compare_three_dataset_qfg_level_knockout_v1 as gate_core  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA = "sctransnet_three_dataset_gcsf_branch_audit_comparison_v1/v1"
ANALYZER_SCHEMA = analyzer.SCHEMA
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLES = tuple(analyzer.CHECKPOINT_ROLES)
PRIMARY_ROLE = "best_miou"
VETO_ROLE = "best_pd"
SEED = analyzer.SEED
FIXED_THRESHOLD = analyzer.FIXED_THRESHOLD
MODES = tuple(analyzer.PUBLIC_MODES)
CURRENT_MODE = analyzer.CURRENT_MODE
NONZERO_MODES = MODES[1:]
REQUIRED_PRIMARY_SAFE_MATERIAL_DATASETS = 2

DECISION_AUTHORIZE = "AUTHORIZE_GCSF_V1_IMPLEMENTATION_AND_PILOT"
DECISION_NO_AUTHORIZATION = "GCSF_BRANCH_AUDIT_NO_TRAINING_AUTHORIZATION"

DEFAULT_INPUT_ROOT = analyzer.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison" / "seed42_six_role"


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


def _is_sha256(value: Any) -> bool:
    return gate_core._is_sha256(value)


def file_sha256(path: Path) -> str:
    return gate_core.file_sha256(Path(path))


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    ready = Path(path)
    if not ready.is_file() or ready.is_symlink():
        raise FileNotFoundError(ready)
    raw = ready.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {ready}")
    return value, _sha256_bytes(raw)


def _extract_point(raw_mode: Mapping[str, Any], label: str) -> dict[str, Any]:
    point = _mapping(raw_mode.get("fixed_threshold_0_5"), f"{label}.fixed")
    _require(
        _finite_float(point.get("threshold"), f"{label}.threshold") == FIXED_THRESHOLD,
        f"{label} threshold differs",
    )
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
    _require(matched <= target_count and matched_tiny <= tiny_count, f"{label} count differs")
    valid = _nonnegative_int(point.get("valid_pixel_count"), f"{label}.valid_pixel_count")
    _require(valid > 0, f"{label} valid pixels must be positive")
    ready = {
        "target_count": target_count,
        "tiny_target_count": tiny_count,
        "matched_target_count": matched,
        "matched_tiny_target_count": matched_tiny,
        "miou": _finite_float(point.get("miou"), f"{label}.miou"),
        "niou": _finite_float(point.get("niou"), f"{label}.niou"),
        "component_false_positive_pixels": _nonnegative_int(
            point.get("unmatched_predicted_pixels"), f"{label}.component_fp"
        ),
        "background_false_positive_pixels": _nonnegative_int(
            point.get("false_positive_pixels"), f"{label}.background_fp"
        ),
        "predicted_object_count": _nonnegative_int(
            point.get("predicted_object_count"), f"{label}.predicted_object_count"
        ),
        "unmatched_predicted_object_count": _nonnegative_int(
            point.get("unmatched_predicted_object_count"),
            f"{label}.unmatched_predicted_object_count",
        ),
        "false_objects_per_image": _finite_float(
            point.get("false_objects_per_image"), f"{label}.false_objects_per_image"
        ),
        "valid_pixel_count": valid,
        "pd": _finite_float(point.get("pd"), f"{label}.pd"),
        "tiny_pd": _finite_float(point.get("tiny_pd"), f"{label}.tiny_pd"),
        "fa": _finite_float(point.get("fa"), f"{label}.fa"),
        "pixel_precision": _finite_float(
            point.get("pixel_precision"), f"{label}.pixel_precision"
        ),
        "pixel_recall": _finite_float(point.get("pixel_recall"), f"{label}.pixel_recall"),
        "pixel_f1": _finite_float(point.get("pixel_f1"), f"{label}.pixel_f1"),
        "test_loss": _finite_float(point.get("test_loss"), f"{label}.test_loss"),
    }
    for metric in (
        "miou",
        "niou",
        "pd",
        "tiny_pd",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    ):
        _require(0.0 <= ready[metric] <= 1.0, f"{label}.{metric} outside [0,1]")
    _require(ready["fa"] >= 0.0, f"{label}.fa is negative")
    _require(
        ready["unmatched_predicted_object_count"] <= ready["predicted_object_count"],
        f"{label} unmatched objects exceed predictions",
    )
    _require(ready["false_objects_per_image"] >= 0.0, f"{label} false objects is negative")
    return ready


def _extract_probability(raw_mode: Mapping[str, Any], label: str, valid: int) -> dict[str, Any]:
    raw = _mapping(
        raw_mode.get("probability_difference_to_current"),
        f"{label}.probability_difference_to_current",
    )
    count = _nonnegative_int(raw.get("element_count"), f"{label}.element_count")
    _require(count == valid, f"{label} probability count differs")
    maximum = _finite_float(raw.get("max_abs"), f"{label}.max_abs")
    absolute_sum = _finite_float(raw.get("absolute_difference_sum"), f"{label}.absolute_sum")
    mean = _finite_float(raw.get("mean_abs"), f"{label}.mean_abs")
    _require(maximum >= 0.0 and absolute_sum >= 0.0 and mean >= 0.0, f"{label} negative diff")
    _require(mean == absolute_sum / count, f"{label} probability mean differs")
    expected_different = bool(
        maximum > qfg_threshold_max() or mean > qfg_threshold_mean()
    )
    _require(raw.get("functionally_different") is expected_different, f"{label} diff flag differs")
    _require(raw.get("equivalent") is (not expected_different), f"{label} equivalent flag differs")
    return {
        "max_abs": maximum,
        "mean_abs": mean,
        "absolute_difference_sum": absolute_sum,
        "element_count": count,
        "functionally_different": expected_different,
        "equivalent": not expected_different,
    }


def qfg_threshold_max() -> float:
    return float(qfg_audit_constant("OUTPUT_EQUIVALENCE_MAX_ABS"))


def qfg_threshold_mean() -> float:
    return float(qfg_audit_constant("OUTPUT_EQUIVALENCE_MEAN_ABS"))


def qfg_audit_constant(name: str) -> Any:
    # Analyzer intentionally reuses the frozen QFG probability primitive.
    return getattr(analyzer.qfg_audit, name)


def validate_analyzer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    serialized_modes = _mapping(payload.get("modes"), "modes")
    _require(set(serialized_modes) == set(MODES), "analyzer mode set differs")
    analyzer.validate_output_payload(payload)
    _require(payload.get("schema") == ANALYZER_SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer is incomplete")
    dataset = payload.get("dataset")
    role = payload.get("checkpoint_role")
    _require(dataset in DATASETS, "analyzer dataset differs")
    _require(role in CHECKPOINT_ROLES, "analyzer role differs")
    _require(payload.get("method") == analyzer.REFERENCE_METHOD, "method differs")
    _require(payload.get("training_model_method") == analyzer.TRAINING_MODEL_METHOD, "training method differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("test_selected") is True, "test-selected differs")
    _require(payload.get("selection_is_optimistic") is True, "selection policy differs")

    checkpoint_binding = _mapping(payload.get("checkpoint_binding"), "checkpoint_binding")
    checkpoint = _mapping(checkpoint_binding.get("checkpoint"), "checkpoint_binding.checkpoint")
    _require(checkpoint.get("role") == role, "checkpoint role binding differs")
    _require(_is_sha256(checkpoint.get("sha256")), "checkpoint SHA differs")
    protocol = _mapping(checkpoint_binding.get("protocol"), "checkpoint_binding.protocol")
    _require(_is_sha256(protocol.get("payload_sha256")), "protocol payload SHA differs")
    data = _mapping(payload.get("data"), "data")
    _require(data.get("split") == "img_idx/test", "data split differs")
    _require(_is_sha256(data.get("inference_order_newline_sha256")), "order SHA differs")
    manifest = _mapping(data.get("protocol_manifest"), "data.protocol_manifest")
    _require(_is_sha256(manifest.get("sha256")), "manifest SHA differs")
    reference = _mapping(payload.get("reference_reuse"), "reference_reuse")
    _require(reference.get("checkpoint_role") == role, "reference role differs")
    _require(_is_sha256(reference.get("sha256")), "reference SHA differs")
    sources = _mapping(payload.get("source_sha256"), "source_sha256")
    _require(bool(sources) and all(_is_sha256(value) for value in sources.values()), "source SHA differs")

    modes: dict[str, Any] = {}
    invariant: tuple[int, int, int] | None = None
    for mode in MODES:
        raw_mode = _mapping(serialized_modes[mode], f"modes.{mode}")
        point = _extract_point(raw_mode, f"modes.{mode}")
        totals = (point["target_count"], point["tiny_target_count"], point["valid_pixel_count"])
        invariant = invariant or totals
        _require(totals == invariant, f"modes.{mode} totals differ")
        probability = _extract_probability(
            raw_mode, f"modes.{mode}", point["valid_pixel_count"]
        )
        if mode == CURRENT_MODE:
            _require(
                probability["max_abs"] == 0.0
                and probability["absolute_difference_sum"] == 0.0,
                "current probability self-difference differs",
            )
        modes[mode] = {
            "fixed_threshold_0_5": point,
            "probability_difference_to_current": probability,
        }
    statistics = _mapping(payload.get("branch_statistics"), "branch_statistics")
    rows = statistics.get("levels")
    _require(isinstance(rows, list) and len(rows) == 4, "branch statistics differ")
    descriptive_rows = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"branch_statistics.levels[{index}]")
        _require(row.get("level_index_zero_based") == index, "statistics index differs")
        descriptive_rows.append(
            {
                "level_index_zero_based": index,
                "level_name": row.get("level_name"),
                "transformed_rms": _finite_float(row.get("transformed_rms"), "transformed_rms"),
                "encoder_rms": _finite_float(row.get("encoder_rms"), "encoder_rms"),
                "transformed_encoder_cosine": (
                    None
                    if row.get("transformed_encoder_cosine") is None
                    else _finite_float(row.get("transformed_encoder_cosine"), "cosine")
                ),
                "transformed_target_to_background_rms_ratio": row.get(
                    "transformed_target_to_background_rms_ratio"
                ),
                "encoder_target_to_background_rms_ratio": row.get(
                    "encoder_target_to_background_rms_ratio"
                ),
                "current_transformed_amplitude_share_proxy": _finite_float(
                    row.get("current_transformed_amplitude_share_proxy"), "amplitude proxy"
                ),
            }
        )
    return {
        "dataset": dataset,
        "checkpoint_role": role,
        "checkpoint_sha256": checkpoint["sha256"],
        "modes": modes,
        "branch_statistics_descriptive_only": descriptive_rows,
    }


def compare_direction(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return gate_core.compare_direction(candidate, reference)


def _binding_key(dataset: str, role: str) -> str:
    return f"{dataset}::{role}"


def _expected_keys() -> tuple[str, ...]:
    return tuple(_binding_key(dataset, role) for dataset in DATASETS for role in CHECKPOINT_ROLES)


def _validate_input_bindings(
    input_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    _require(set(input_bindings) == set(_expected_keys()), "input bindings require six roles")
    ready: dict[str, dict[str, str]] = {}
    for key in _expected_keys():
        raw = _mapping(input_bindings[key], f"input_bindings.{key}")
        path = raw.get("path")
        sha = raw.get("sha256")
        _require(isinstance(path, str) and bool(path), f"input path differs: {key}")
        _require(_is_sha256(sha), f"input SHA differs: {key}")
        ready[key] = {"path": path, "sha256": str(sha)}
    return ready


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    input_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate six artifacts and apply only quantified Trigger A."""

    expected = set(_expected_keys())
    _require(set(payloads) == expected, "comparison requires six dataset-role payloads")
    bindings = _validate_input_bindings(input_bindings)
    normalized: dict[str, Any] = {}
    for key in _expected_keys():
        one = validate_analyzer_payload(payloads[key])
        dataset, role = key.split("::", 1)
        _require(one["dataset"] == dataset and one["checkpoint_role"] == role, f"binding differs: {key}")
        normalized[key] = one

    per_unit: dict[str, Any] = {}
    mode_rows: dict[str, dict[str, dict[str, Any]]] = {
        mode: {role: {} for role in CHECKPOINT_ROLES} for mode in NONZERO_MODES
    }
    for key in _expected_keys():
        one = normalized[key]
        dataset = one["dataset"]
        role = one["checkpoint_role"]
        current = one["modes"][CURRENT_MODE]["fixed_threshold_0_5"]
        comparisons: dict[str, Any] = {}
        for mode in NONZERO_MODES:
            candidate = one["modes"][mode]["fixed_threshold_0_5"]
            direction = compare_direction(candidate, current)
            mode_rows[mode][role][dataset] = direction
            comparisons[mode] = {
                "candidate_vs_current": direction,
                "probability_difference_to_current": one["modes"][mode][
                    "probability_difference_to_current"
                ],
            }
        per_unit[key] = {
            "dataset": dataset,
            "checkpoint_role": role,
            "checkpoint_sha256": one["checkpoint_sha256"],
            "current_fixed_threshold_0_5": current,
            "modes": comparisons,
            "branch_statistics_descriptive_only": one[
                "branch_statistics_descriptive_only"
            ],
        }

    trigger_mode_rows: dict[str, Any] = {}
    qualifying_modes: list[str] = []
    for mode in NONZERO_MODES:
        primary_rows = mode_rows[mode][PRIMARY_ROLE]
        safe_material = [
            dataset
            for dataset in DATASETS
            if primary_rows[dataset]["safe_material_improvement"]
        ]
        severe_units = [
            _binding_key(dataset, role)
            for dataset in DATASETS
            for role in CHECKPOINT_ROLES
            if mode_rows[mode][role][dataset]["severe_degradation"]
        ]
        passed = bool(
            len(safe_material) >= REQUIRED_PRIMARY_SAFE_MATERIAL_DATASETS
            and not severe_units
        )
        if passed:
            qualifying_modes.append(mode)
        trigger_mode_rows[mode] = {
            "primary_safe_material_datasets": safe_material,
            "primary_safe_material_dataset_count": len(safe_material),
            "required_primary_safe_material_dataset_count": (
                REQUIRED_PRIMARY_SAFE_MATERIAL_DATASETS
            ),
            "severe_degradation_units_across_six_roles": severe_units,
            "severe_degradation_unit_count": len(severe_units),
            "best_pd_used_as_severe_veto_only": True,
            "trigger_a_passed": passed,
        }

    trigger_a_passed = bool(qualifying_modes)
    decision = DECISION_AUTHORIZE if trigger_a_passed else DECISION_NO_AUTHORIZATION
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": decision,
        "seed": SEED,
        "test_selected": True,
        "fixed_threshold": FIXED_THRESHOLD,
        "datasets": list(DATASETS),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "primary_checkpoint_role": PRIMARY_ROLE,
        "severe_veto_checkpoint_role": VETO_ROLE,
        "trigger_a": {
            "implemented": True,
            "sole_training_authorization_trigger": True,
            "rule": (
                "same_nonzero_mode_best_miou_safe_material_on_at_least_2_of_3_"
                "and_zero_severe_across_all_6_dataset_role_units"
            ),
            "passed": trigger_a_passed,
            "qualifying_modes": qualifying_modes,
            "modes": trigger_mode_rows,
        },
        "trigger_b": {
            "implemented": False,
            "descriptive_statistics_only": True,
            "authorizes_training": False,
        },
        "trigger_c": {
            "implemented": False,
            "descriptive_statistics_only": True,
            "authorizes_training": False,
        },
        "gcsf_v1_implementation_and_pilot_authorized": trigger_a_passed,
        "unrepresentable_f1_f3_used_for_trigger": False,
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
        },
        "per_unit": per_unit,
        "input_bindings": bindings,
        "scope": {
            "single_seed_test_selected_zero_training_diagnostic": True,
            "best_miou_primary": True,
            "best_pd_severe_veto_only": True,
            "branch_statistics_used_for_training_authorization": False,
            "descriptive_sweep_used_for_decision": False,
            "paper_mechanism_evidence": False,
            "stability_claim_supported": False,
        },
        "source_sha256": {
            "analysis/compare_three_dataset_gcsf_branch_audit_v1.py": file_sha256(Path(__file__)),
            "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py": file_sha256(Path(analyzer.__file__)),
            "analysis/compare_three_dataset_qfg_level_knockout_v1.py": file_sha256(Path(gate_core.__file__)),
        },
        "no_fabricated_results": True,
    }


def validate_comparison_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "comparison schema differs")
    _require(payload.get("status") == "complete", "comparison incomplete")
    decision = payload.get("decision")
    _require(decision in (DECISION_AUTHORIZE, DECISION_NO_AUTHORIZATION), "decision differs")
    trigger = _mapping(payload.get("trigger_a"), "trigger_a")
    passed = trigger.get("passed")
    _require(isinstance(passed, bool), "Trigger A result differs")
    _require(
        payload.get("gcsf_v1_implementation_and_pilot_authorized") is passed,
        "authorization differs from Trigger A",
    )
    _require(
        decision == (DECISION_AUTHORIZE if passed else DECISION_NO_AUTHORIZATION),
        "decision differs from Trigger A",
    )
    _require(
        payload.get("trigger_b", {}).get("authorizes_training") is False
        and payload.get("trigger_c", {}).get("authorizes_training") is False,
        "unquantified trigger authorized training",
    )
    _require(set(payload.get("input_bindings", {})) == set(_expected_keys()), "bindings differ")
    _require(set(payload.get("per_unit", {})) == set(_expected_keys()), "per-unit matrix differs")
    expected_sources = {
        "analysis/compare_three_dataset_gcsf_branch_audit_v1.py": file_sha256(Path(__file__)),
        "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py": file_sha256(Path(analyzer.__file__)),
        "analysis/compare_three_dataset_qfg_level_knockout_v1.py": file_sha256(Path(gate_core.__file__)),
    }
    _require(payload.get("source_sha256") == expected_sources, "comparison source lock differs")


def _format_delta(value: Any) -> str:
    return f"{float(value):+.6f}"


def _format_reduction(value: Mapping[str, Any]) -> str:
    raw = value.get("value")
    return "introduced-from-zero" if raw is None else f"{float(raw):+.2%}"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# GCSF 零训练分支诊断裁决",
        "",
        f"- 决策：`{result['decision']}`",
        f"- Trigger A：`{str(result['trigger_a']['passed']).lower()}`",
        "- 主门：`best_miou`；`best_pd` 仅作六单元 severe veto。",
        "- Trigger B/C：未实现，仅保留分支统计作描述，不授权训练。",
        "- 正式阈值：`0.5`；阈值 `1.0` 仅为合法空预测端点。",
        "",
        "## Trigger A 模式汇总",
        "",
        "| 模式 | best_mIoU safe-material | 六角色 severe | 通过 |",
        "|---|---:|---:|---:|",
    ]
    for mode in NONZERO_MODES:
        row = result["trigger_a"]["modes"][mode]
        lines.append(
            f"| `{mode}` | {row['primary_safe_material_dataset_count']}/3 | "
            f"{row['severe_degradation_unit_count']}/6 | "
            f"{str(row['trigger_a_passed']).lower()} |"
        )
    lines.extend(["", "## 六角色逐模式差值", ""])
    for key in _expected_keys():
        unit = result["per_unit"][key]
        lines.extend(
            [
                f"### {unit['dataset']} — {unit['checkpoint_role']}",
                "",
                "| 模式 | Δ目标 | Δtiny | ΔmIoU | ΔnIoU | component FP reduction | background FP reduction | safe-material | severe |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in NONZERO_MODES:
            row = unit["modes"][mode]["candidate_vs_current"]
            lines.append(
                f"| `{mode}` | {row['delta_target']:+d} | {row['delta_tiny']:+d} | "
                f"{_format_delta(row['delta_miou'])} | {_format_delta(row['delta_niou'])} | "
                f"{_format_reduction(row['component_fp_reduction'])} | "
                f"{_format_reduction(row['background_pixel_fp_reduction'])} | "
                f"{str(row['safe_material_improvement']).lower()} | "
                f"{str(row['severe_degradation']).lower()} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "分支 RMS、余弦、目标/背景比和幅度份额只描述固定 checkpoint 的分支状态。",
            "它们不参与本次训练授权；授权结果完全由上面的 Trigger A 决定。",
            "",
        ]
    )
    return "\n".join(lines)


def _default_input(dataset: str, role: str) -> Path:
    return (
        DEFAULT_INPUT_ROOT
        / "runs"
        / dataset
        / f"v4_tss_off_{role}_seed42"
        / "evaluation.json"
    )


def _parse_bindings(values: Sequence[str]) -> dict[str, Path]:
    if not values:
        return {
            _binding_key(dataset, role): _default_input(dataset, role)
            for dataset in DATASETS
            for role in CHECKPOINT_ROLES
        }
    ready: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input must use DATASET::ROLE=PATH")
        key, raw_path = value.split("=", 1)
        _require(key in _expected_keys(), f"unknown input key: {key}")
        _require(key not in ready, f"duplicate input key: {key}")
        ready[key] = Path(raw_path)
    _require(set(ready) == set(_expected_keys()), "--input must provide all six bindings")
    return ready


def _atomic_write_once(path: Path, text: str) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
        raise FileExistsError(f"refusing existing output: {json_path}")
    if markdown_path.exists() or markdown_path.is_symlink():
        raise FileExistsError(f"refusing existing output: {markdown_path}")
    json_text = json.dumps(
        result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )
    _atomic_write_once(json_path, json_text)
    try:
        _atomic_write_once(markdown_path, render_markdown(result))
    except BaseException:
        # JSON is deliberately write-once.  Leave it as evidence if the second
        # publication fails rather than silently overwriting either artifact.
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="DATASET::ROLE=PATH; omit all six to use formal defaults",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.json")
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = _parse_bindings(args.input)
    payloads: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, Mapping[str, str]] = {}
    for key in _expected_keys():
        payload, sha = _load_json_with_sha(paths[key])
        payloads[key] = payload
        bindings[key] = {"path": str(paths[key].resolve()), "sha256": sha}
    result = compare_payloads(payloads, input_bindings=bindings)
    write_outputs(args.output_json, args.output_markdown, result)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete",
                "decision": result["decision"],
                "output_json": str(args.output_json.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
