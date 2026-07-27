#!/usr/bin/env python3
"""Merge and audit the two-lane frozen V6 failure atlas.

This program is deliberately a post-processing tool.  It does not import a
model, run inference, select a checkpoint, or modify any formal experiment
artifact.  It reads the two lane ``matrix_summary.json`` files and their eight
checkpoint diagnostics, verifies their provenance contracts, and writes one
compact JSON report plus one deterministic Markdown rendering.

The decision has a narrow scope:

* frozen counterfactuals may provide direct *forward-effect* support;
* the preregistered zero-scale gradient asymmetry is code/formula evidence;
* authorising a fresh DCH trajectory test is not a causal conclusion and does
  not replace Pd/Fa/mIoU Gates A--E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS_ROOT = (
    REPO_ROOT
    / "analysis/results/tpd_clean_v6_frozen_failure_atlas_v1"
)
SCHEMA = "sctransnet_tpd_clean_v6_failure_atlas_summary_v1"
MATRIX_SCHEMA = "sctransnet_tpd_clean_v6_frozen_failure_matrix_v1"
DIAGNOSTIC_SCHEMA = "sctransnet_tpd_clean_v6_frozen_failure_diagnostic_v1"

FULL_VARIANT = "tpd_clean_v6_full"
CAPACITY_VARIANT = "tpd_clean_v6_phase_capacity"
VARIANTS = (FULL_VARIANT, CAPACITY_VARIANT)
SEEDS = (42, 3407)
ROLES = ("pd_primary", "miou_primary")
MODES = (
    "as_trained",
    "same_weights_context_off",
    "same_weights_residual_off",
)
FIXED_THRESHOLDS = ("0.5", "0.58", "0.999")
FA_BUDGETS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
EXPECTED_LANES = {
    "gpu2_full": {
        "variant": FULL_VARIANT,
        "physical_gpu": "2",
    },
    "gpu3_capacity": {
        "variant": CAPACITY_VARIANT,
        "physical_gpu": "3",
    },
}
EXPECTED_MODE_IMPLEMENTATIONS = {
    "as_trained": "unchanged_forward",
    "same_weights_context_off": "temporary_use_context_headroom_false",
    "same_weights_residual_off": "temporary_keep_only_forward_hooks",
}
SALIENCY_SCALE_SATURATION_BOUND = 0.5

GATE_A_DEFICITS = {
    "pd_primary_miou": {
        "role": "pd_primary",
        "metric": "miou",
        "direction": "higher",
        "threshold": 0.9336470588,
    },
    "miou_primary_miou": {
        "role": "miou_primary",
        "metric": "miou",
        "direction": "higher",
        "threshold": 0.946542,
    },
    "miou_primary_fa": {
        "role": "miou_primary",
        "metric": "fa",
        "direction": "lower",
        "threshold": 1e-6,
    },
}

# This is an explicit preregistered code-evidence preset, not a value inferred
# from the frozen checkpoint atlas.  ``verify_code_evidence`` also checks and
# hashes the cited local sources so the report cannot silently promote a
# missing assertion.
ZERO_SCALE_GRADIENT_CODE_EVIDENCE = {
    "confirmed": True,
    "evidence_class": "static_formula_and_unit_level_code_review",
    "runtime_checkpoint_evidence": False,
    "claim": (
        "V6 Full and Capacity are output-identical at zero scale, while their "
        "saliency-scale first derivatives differ."
    ),
    "scope_limit": (
        "This establishes a formula/code asymmetry only; it does not establish "
        "that the asymmetry caused a trained checkpoint failure."
    ),
    "sources": (
        {
            "path": "model/tpd_clean_v6.py",
            "required_snippets": (
                '"fusion_formula": "K+Sa*tanh(saliency_scale)"',
                "(1.0 - scale.abs())",
            ),
        },
        {
            "path": "SCTransNet_TPD_V7_DCH_失败分析与不改主线修改计划.md",
            "required_snippets": (
                "step 0 前向输出完全一致",
                "`saliency_scale` 梯度不同",
            ),
        },
    ),
}

HOOK_RESTORATION_CODE_EVIDENCE = {
    "confirmed": True,
    "evidence_class": "implementation_and_unit_test",
    "runtime_checkpoint_evidence": False,
    "claim": (
        "The temporary residual-off forward hooks are removed on context exit."
    ),
    "sources": (
        {
            "path": "analysis/diagnose_tpd_clean_v6_fragmentation.py",
            "required_snippets": ("handle.remove()",),
        },
        {
            "path": "tests/test_diagnose_tpd_clean_v6_fragmentation.py",
            "required_snippets": (
                "test_residual_off_hook_is_keep_only_and_is_removed",
                "restored = block(sample)",
            ),
        },
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected regular JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        _require(math.isfinite(value), f"Non-finite value in summary: {value}")
    return value


def verify_code_evidence(
    preset: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    _require(preset.get("confirmed") is True, "Code evidence is not confirmed")
    sources = []
    for spec in preset["sources"]:
        relative = Path(spec["path"])
        path = (repo_root / relative).resolve()
        _require(
            path.is_relative_to(repo_root.resolve()),
            f"Code-evidence path escapes repository: {path}",
        )
        _require(
            path.is_file() and not path.is_symlink(),
            f"Code-evidence source is not a regular file: {path}",
        )
        text = path.read_text(encoding="utf-8")
        for snippet in spec["required_snippets"]:
            _require(
                snippet in text,
                f"Code-evidence snippet missing from {relative}: {snippet!r}",
            )
        sources.append(
            {
                "path": str(relative),
                "sha256": file_sha256(path),
                "required_snippets_verified": True,
            }
        )
    return {
        **{
            key: value
            for key, value in preset.items()
            if key != "sources"
        },
        "sources": sources,
    }


def checkpoint_key(
    variant: str,
    seed: int,
    role: str,
) -> str:
    return f"{variant}/seed_{seed}/{role}"


def _expected_diagnostic_path(
    atlas_root: Path,
    lane: str,
    variant: str,
    seed: int,
    role: str,
) -> Path:
    return (
        atlas_root
        / lane
        / variant
        / f"seed_{seed}"
        / f"{role}.json"
    ).resolve()


def _input_artifact_paths(payload: Mapping[str, Any]) -> Dict[str, Path]:
    run_dir = Path(payload["run_directory"]).resolve()
    return {
        "protocol": run_dir / "protocol.json",
        "split": run_dir / "split.json",
        "summary": run_dir / "summary.json",
        "metrics": run_dir / "metrics.jsonl",
        "checkpoint": Path(payload["checkpoint"]).resolve(),
        "formal_sweep": Path(payload["formal_sweep"]).resolve(),
    }


def _validate_matrix(
    matrix: Mapping[str, Any],
    matrix_path: Path,
    lane: str,
    lane_spec: Mapping[str, str],
    expected_outputs: Sequence[Path],
) -> None:
    _require(matrix.get("schema") == MATRIX_SCHEMA, f"Bad schema: {matrix_path}")
    _require(matrix.get("mode") == "run", f"Matrix is not a run: {matrix_path}")
    _require(
        matrix.get("training_performed") is False,
        f"Matrix reports training: {matrix_path}",
    )
    _require(
        matrix.get("official_test_accessed") is False,
        f"Matrix accessed official test: {matrix_path}",
    )
    _require(
        matrix.get("formal_gate_replacement") is False,
        f"Matrix replaces formal gates: {matrix_path}",
    )
    _require(
        matrix.get("complete_validation_split") is True,
        f"Matrix is not full-validation: {matrix_path}",
    )
    _require(
        matrix.get("output_count") == 4,
        f"Matrix must contain four diagnostics: {matrix_path}",
    )
    _require(
        matrix.get("requested_variants") == [lane_spec["variant"]],
        f"Unexpected variant in {matrix_path}",
    )
    _require(
        set(matrix.get("requested_seeds", [])) == set(SEEDS),
        f"Unexpected seeds in {matrix_path}",
    )
    _require(
        set(matrix.get("requested_checkpoint_roles", [])) == set(ROLES),
        f"Unexpected roles in {matrix_path}",
    )
    _require(
        tuple(matrix.get("requested_modes", [])) == MODES,
        f"Unexpected modes in {matrix_path}",
    )
    observed_thresholds = tuple(
        f"{float(value):.10g}" for value in matrix.get("fixed_thresholds", [])
    )
    _require(
        observed_thresholds == FIXED_THRESHOLDS,
        f"Unexpected fixed thresholds in {matrix_path}",
    )
    _require(
        tuple(f"{float(value):.10g}" for value in matrix.get("fa_budgets", []))
        == FA_BUDGETS,
        f"Unexpected Fa budgets in {matrix_path}",
    )
    _require(
        str(matrix.get("device", {}).get("physical_gpu"))
        == lane_spec["physical_gpu"],
        f"Wrong physical GPU provenance for lane {lane}",
    )
    observed_outputs = {Path(value).resolve() for value in matrix["outputs"]}
    _require(
        observed_outputs == set(expected_outputs),
        f"Matrix output manifest differs in lane {lane}",
    )


def _validate_diagnostic(
    payload: Mapping[str, Any],
    path: Path,
    variant: str,
    seed: int,
    role: str,
) -> Dict[str, Any]:
    label = checkpoint_key(variant, seed, role)
    _require(payload.get("schema") == DIAGNOSTIC_SCHEMA, f"Bad schema: {label}")
    _require(payload.get("variant") == variant, f"Variant mismatch: {label}")
    _require(payload.get("seed") == seed, f"Seed mismatch: {label}")
    _require(payload.get("checkpoint_role") == role, f"Role mismatch: {label}")
    _require(
        Path(payload.get("output", "")).resolve() == path,
        f"Output path mismatch: {label}",
    )
    for key in (
        "training_performed",
        "official_test_accessed",
        "formal_gate_replacement",
        "checkpoint_reselection_permitted",
    ):
        _require(payload.get(key) is False, f"{key} must be false: {label}")
    _require(
        payload.get("complete_validation_split") is True,
        f"Incomplete validation split: {label}",
    )
    validation = payload.get("validation", {})
    _require(
        validation.get("validation_count") == 133
        and validation.get("formal_validation_count") == 133,
        f"Expected complete 133-image validation: {label}",
    )
    _require(set(payload.get("modes", {})) == set(MODES), f"Bad modes: {label}")

    consistency = payload.get("as_trained_formal_sweep_consistency")
    _require(isinstance(consistency, dict), f"Missing formal consistency: {label}")
    _require(
        consistency.get("max_abs_numeric_delta") == 0.0,
        f"Non-zero as-trained/formal numeric delta: {label}",
    )
    _require(
        consistency.get("all_count_fields_match") is True,
        f"Formal count mismatch: {label}",
    )
    _require(
        all(
            value == 0.0
            for value in consistency.get(
                "fixed_threshold_0_5_numeric_deltas_diagnostic_minus_formal",
                {},
            ).values()
        ),
        f"At least one formal numeric field differs: {label}",
    )
    _require(
        all(
            value is True
            for value in consistency.get(
                "fixed_threshold_0_5_exact_count_matches",
                {},
            ).values()
        ),
        f"At least one formal count field differs: {label}",
    )

    before = payload.get("input_sha256_before")
    after = payload.get("input_sha256_after")
    _require(isinstance(before, dict) and before == after, f"Input SHA changed: {label}")
    _require(
        payload.get("formal_inputs_unchanged") is True,
        f"Input unchanged flag is false: {label}",
    )
    artifact_paths = _input_artifact_paths(payload)
    _require(set(artifact_paths) == set(before), f"Input manifest differs: {label}")
    for artifact_name, artifact_path in artifact_paths.items():
        _require(
            artifact_path.is_file() and not artifact_path.is_symlink(),
            f"Input artifact is not a regular file: {artifact_path}",
        )
        _require(
            file_sha256(artifact_path) == before[artifact_name],
            f"Current input SHA differs for {label}/{artifact_name}",
        )
    _require(
        consistency.get("formal_sweep_checkpoint_sha256")
        == before["checkpoint"],
        f"Formal sweep checkpoint SHA mismatch: {label}",
    )

    loaded_state = payload.get("loaded_model_state_sha256")
    _require(isinstance(loaded_state, str) and loaded_state, f"No state SHA: {label}")
    for mode in MODES:
        provenance = payload["modes"][mode].get("counterfactual_provenance", {})
        _require(
            provenance.get("mode") == mode,
            f"Mode provenance mismatch: {label}/{mode}",
        )
        _require(
            provenance.get("implementation")
            == EXPECTED_MODE_IMPLEMENTATIONS[mode],
            f"Mode implementation mismatch: {label}/{mode}",
        )
        _require(
            provenance.get("block_count") == 7,
            f"Mode block count mismatch: {label}/{mode}",
        )
        _require(
            provenance.get("state_sha256_before")
            == provenance.get("state_sha256_after_restore")
            == loaded_state,
            f"Model state was not restored: {label}/{mode}",
        )
        _require(
            provenance.get("state_restored_exactly") is True
            and provenance.get("zero_training") is True,
            f"Restoration/training provenance failed: {label}/{mode}",
        )

    static = payload.get("checkpoint_static_diagnostics", {})
    _require(static.get("block_count") == 7, f"Static block count mismatch: {label}")
    blocks = static.get("blocks", [])
    _require(len(blocks) == 7, f"Expected seven static rows: {label}")
    for block in blocks:
        scales = block.get("saliency_scale_effective_abs_tanh", {})
        for statistic in ("median", "p90", "max"):
            value = float(scales[statistic])
            _require(
                math.isfinite(value) and 0.0 <= value <= 1.0,
                f"Invalid scale statistic: {label}/{block.get('block')}",
            )
        for rho_name in ("rho_l1", "rho_l2"):
            value = float(block["phase_sum_cancellation"][rho_name])
            _require(
                math.isfinite(value) and 0.0 <= value <= 1.0 + 1e-7,
                f"Invalid {rho_name}: {label}/{block.get('block')}",
            )

    return {
        "as_trained_formal_max_abs_numeric_delta": 0.0,
        "formal_count_fields_match": True,
        "formal_inputs_sha_before_after_equal": True,
        "formal_inputs_current_sha_match": True,
        "model_state_restored_all_modes": True,
        "runtime_hook_mode_implementation_verified": True,
    }


def _compact_point(point: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if point is None:
        return None
    topology = point.get("gt_topology", {})
    taxonomy = point.get("component_taxonomy", {})
    return {
        "threshold": float(point["threshold"]),
        "matched_target_count": int(point["matched_target_count"]),
        "target_count": int(point["target_count"]),
        "pd": float(point["pd"]),
        "matched_tiny_target_count": int(point["matched_tiny_target_count"]),
        "tiny_target_count": int(point["tiny_target_count"]),
        "tiny_pd": float(point["tiny_pd"]),
        "fa": float(point["fa"]),
        "miou": float(point["miou"]),
        "niou": float(point["niou"]),
        "pixel_precision": float(point["pixel_precision"]),
        "pixel_recall": float(point["pixel_recall"]),
        "pixel_f1": float(point["pixel_f1"]),
        "predicted_object_count": int(point["predicted_object_count"]),
        "unmatched_predicted_object_count": int(
            point["unmatched_predicted_object_count"]
        ),
        "fragmented_gt_count": int(topology["fragmented_gt_count"]),
        "split_target_count": int(topology["split_target_count"]),
        "fragment_excess_total": int(topology["fragment_excess_total"]),
        "largest_fragment_fraction_mean": float(
            topology["largest_fragment_fraction_mean"]
        ),
        "largest_fragment_fraction_p10": float(
            topology["largest_fragment_fraction_p10"]
        ),
        "unmatched_component_count_by_class": dict(
            taxonomy["unmatched_component_count_by_class"]
        ),
        "unmatched_component_pixels_by_class": dict(
            taxonomy["unmatched_component_pixels_by_class"]
        ),
        "fragment_fa_fraction": float(taxonomy["fragment_fa_fraction"]),
        "background_fa_fraction": float(taxonomy["background_fa_fraction"]),
    }


def _compact_static(payload: Mapping[str, Any]) -> Dict[str, Any]:
    static = payload["checkpoint_static_diagnostics"]
    return {
        "block_count": 7,
        "blocks": [
            {
                "block": block["block"],
                "channels": int(block["channels"]),
                "use_context_headroom": bool(block["use_context_headroom"]),
                "saliency_scale_raw_abs": dict(block["saliency_scale_raw"]),
                "saliency_scale_effective_abs_tanh": dict(
                    block["saliency_scale_effective_abs_tanh"]
                ),
                "rho_l1": float(
                    block["phase_sum_cancellation"]["rho_l1"]
                ),
                "rho_l2": float(
                    block["phase_sum_cancellation"]["rho_l2"]
                ),
            }
            for block in static["blocks"]
        ],
        "aggregate": {
            "saliency_scale_effective_abs_tanh": dict(
                static["aggregate"]["saliency_scale_effective_abs_tanh"]
            ),
            "rho_l1": float(
                static["aggregate"]["phase_sum_cancellation"]["rho_l1"]
            ),
            "rho_l2": float(
                static["aggregate"]["phase_sum_cancellation"]["rho_l2"]
            ),
        },
        "definitions": dict(static["definitions"]),
        "scope_limit": static["scope_limit"],
    }


def _registered_operating_points(
    mode_payload: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    points = []
    for threshold in FIXED_THRESHOLDS:
        point = mode_payload["fixed_threshold_points"][threshold]
        points.append(
            {
                "source": f"fixed_threshold/{threshold}",
                "point": _compact_point(point),
            }
        )
    for budget in FA_BUDGETS:
        point = mode_payload["best_points_under_fa_budget"].get(budget)
        if point is not None:
            points.append(
                {
                    "source": f"fa_budget/{budget}",
                    "point": _compact_point(point),
                }
            )
    # One threshold can be selected by several budgets.  Preserve one stable
    # label because source labels must never influence the numeric selection.
    unique: Dict[tuple[float, int], Dict[str, Any]] = {}
    for item in points:
        point = item["point"]
        key = (float(point["threshold"]), int(point["matched_target_count"]))
        previous = unique.get(key)
        if previous is None or item["source"] < previous["source"]:
            unique[key] = item
    return [unique[key] for key in sorted(unique)]


def _matched_pd_point(
    mode_payload: Mapping[str, Any],
    matched_target_count: int,
) -> Dict[str, Any] | None:
    candidates = [
        item
        for item in _registered_operating_points(mode_payload)
        if item["point"]["matched_target_count"] == matched_target_count
    ]
    if not candidates:
        return None
    # Compare modes at an exact detection count.  Within a mode, choose the
    # lowest-Fa registered point, then higher mIoU, then a stable threshold and
    # source-label tie-break.  This rule is fixed in code and reported.
    return min(
        candidates,
        key=lambda item: (
            item["point"]["fa"],
            -item["point"]["miou"],
            item["point"]["threshold"],
            item["source"],
        ),
    )


def _point_delta(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "matched_target_count": (
            int(candidate["matched_target_count"])
            - int(reference["matched_target_count"])
        ),
        "pd": float(candidate["pd"]) - float(reference["pd"]),
        "fa": float(candidate["fa"]) - float(reference["fa"]),
        "miou": float(candidate["miou"]) - float(reference["miou"]),
        "fragmented_gt_count": (
            int(candidate["fragmented_gt_count"])
            - int(reference["fragmented_gt_count"])
        ),
        "fragment_excess_total": (
            int(candidate["fragment_excess_total"])
            - int(reference["fragment_excess_total"])
        ),
        "largest_fragment_fraction_mean": (
            float(candidate["largest_fragment_fraction_mean"])
            - float(reference["largest_fragment_fraction_mean"])
        ),
    }


def _registered_failure_improved(delta: Mapping[str, Any]) -> bool:
    return (
        int(delta["matched_target_count"]) >= 0
        and float(delta["pd"]) >= -1e-12
        and int(delta["fragment_excess_total"]) < 0
        and (
            float(delta["fa"]) < -1e-15
            or float(delta["miou"]) > 1e-12
        )
    )


def _build_matched_pd_comparisons(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[Dict[str, Any], bool]:
    comparisons: Dict[str, Any] = {}
    complete = True
    for role in ROLES:
        key = checkpoint_key(FULL_VARIANT, 3407, role)
        payload = payloads[key]
        anchor = payload["modes"]["as_trained"]["fixed_threshold_points"]["0.5"]
        target_count = int(anchor["matched_target_count"])
        selected = {
            mode: _matched_pd_point(payload["modes"][mode], target_count)
            for mode in MODES
        }
        available = all(item is not None for item in selected.values())
        complete = complete and available
        entry: Dict[str, Any] = {
            "anchor": (
                "as_trained fixed_threshold/0.5 matched_target_count"
            ),
            "target_matched_target_count": target_count,
            "all_modes_have_exact_matched_pd_point": available,
            "selection_rule": (
                "Among fixed-threshold and preregistered Fa-budget points with "
                "exactly the anchor matched-target count: lowest Fa, then "
                "higher mIoU, then lower threshold, then source label."
            ),
            "modes": selected,
            "comparisons_to_as_trained": {},
        }
        if available:
            reference = selected["as_trained"]["point"]
            for mode in MODES[1:]:
                candidate = selected[mode]["point"]
                delta = _point_delta(candidate, reference)
                entry["comparisons_to_as_trained"][mode] = {
                    "delta_candidate_minus_as_trained": delta,
                    "registered_failure_improved": (
                        _registered_failure_improved(delta)
                    ),
                }
        comparisons[role] = entry
    return comparisons, complete


def _gate_a_forward_effect(
    payloads: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    improved_names: list[str] = []
    worsened_names: list[str] = []
    for name, spec in GATE_A_DEFICITS.items():
        payload = payloads[checkpoint_key(FULL_VARIANT, 42, spec["role"])]
        reference = payload["modes"]["as_trained"]["fixed_threshold_points"]["0.5"]
        candidate = payload["modes"]["same_weights_context_off"][
            "fixed_threshold_points"
        ]["0.5"]
        before = float(reference[spec["metric"]])
        after = float(candidate[spec["metric"]])
        signed_toward_gate = (
            after - before
            if spec["direction"] == "higher"
            else before - after
        )
        tolerance = 1e-12 if spec["metric"] == "miou" else 1e-15
        status = (
            "improved"
            if signed_toward_gate > tolerance
            else "worsened"
            if signed_toward_gate < -tolerance
            else "unchanged"
        )
        if status == "improved":
            improved_names.append(name)
        elif status == "worsened":
            worsened_names.append(name)
        rows[name] = {
            **dict(spec),
            "as_trained": before,
            "same_weights_context_off": after,
            "signed_delta_toward_gate": signed_toward_gate,
            "status": status,
        }

    condition = False
    for improved in improved_names:
        other_names = [name for name in GATE_A_DEFICITS if name != improved]
        if sum(rows[name]["status"] == "worsened" for name in other_names) < 2:
            condition = True
            break
    return {
        "fixed_threshold": 0.5,
        "deficits": rows,
        "improved_deficits": improved_names,
        "worsened_deficits": worsened_names,
        "condition": condition,
        "rule": (
            "At least one of the three failed Gate-A quantities moves toward "
            "its frozen threshold, and the other two do not both worsen."
        ),
    }


def _decision(
    payloads: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    static_summaries: Mapping[str, Mapping[str, Any]],
    gradient_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    matched_pd, matched_pd_complete = _build_matched_pd_comparisons(payloads)
    primary_context = matched_pd["pd_primary"].get(
        "comparisons_to_as_trained", {}
    ).get("same_weights_context_off")
    context_primary_fragmentation = bool(
        primary_context
        and primary_context["delta_candidate_minus_as_trained"][
            "matched_target_count"
        ]
        >= 0
        and primary_context["delta_candidate_minus_as_trained"]["pd"]
        >= -1e-12
        and primary_context["delta_candidate_minus_as_trained"][
            "fragment_excess_total"
        ]
        < 0
    )
    gate_a = _gate_a_forward_effect(payloads)
    context_direct_support = (
        context_primary_fragmentation or gate_a["condition"]
    )

    residual_both = all(
        matched_pd[role].get("comparisons_to_as_trained", {})
        .get("same_weights_residual_off", {})
        .get("registered_failure_improved", False)
        for role in ROLES
    )
    context_neither = all(
        not matched_pd[role].get("comparisons_to_as_trained", {})
        .get("same_weights_context_off", {})
        .get("registered_failure_improved", False)
        for role in ROLES
    )
    residual_off_only = residual_both and context_neither

    scale_rows = [
        {
            "checkpoint": key,
            "block": block["block"],
            "max_abs_tanh_scale": float(
                block["saliency_scale_effective_abs_tanh"]["max"]
            ),
        }
        for key, static in static_summaries.items()
        for block in static["blocks"]
    ]
    observed_scale_max = max(
        (row["max_abs_tanh_scale"] for row in scale_rows),
        default=float("inf"),
    )
    scale_not_saturated = (
        len(scale_rows) == 8 * 7
        and all(
            row["max_abs_tanh_scale"] < SALIENCY_SCALE_SATURATION_BOUND
            for row in scale_rows
        )
    )
    diagnostics_complete = (
        validation["checkpoint_count"] == 8
        and validation["all_checks_passed"]
        and matched_pd_complete
    )
    zero_scale_gradient = gradient_evidence.get("confirmed") is True
    go = (
        diagnostics_complete
        and zero_scale_gradient
        and scale_not_saturated
        and not residual_off_only
    )
    status = (
        "CONTEXT_DIRECT_SUPPORT"
        if go and context_direct_support
        else "GO_DCH_TRAJECTORY_TEST"
        if go
        else "NO_GO_DCH"
    )
    return {
        "status": status,
        "registered_formula": (
            "diagnostics_complete && "
            "zero_scale_gradient_asymmetry_confirmed && "
            "saliency_scale_not_saturated && "
            "!residual_off_only_explains_registered_failure"
        ),
        "conditions": {
            "diagnostics_complete": diagnostics_complete,
            "zero_scale_gradient_asymmetry_confirmed": zero_scale_gradient,
            "saliency_scale_not_saturated": scale_not_saturated,
            "residual_off_only_explains_registered_failure": residual_off_only,
            "context_off_improves_seed3407_primary_fragmentation": (
                context_primary_fragmentation
            ),
            "context_off_improves_at_least_one_gate_A_deficit": (
                gate_a["condition"]
            ),
            "context_direct_support": context_direct_support,
        },
        "saliency_scale_saturation_contract": {
            "strict_upper_bound": SALIENCY_SCALE_SATURATION_BOUND,
            "observed_block_count": len(scale_rows),
            "expected_block_count": 56,
            "observed_global_max": observed_scale_max,
            "all_observed_block_maxima_strictly_below_bound": (
                scale_not_saturated
            ),
        },
        "matched_pd_comparisons_seed3407_full": matched_pd,
        "gate_A_context_off_forward_effect_seed42_full": gate_a,
        "implementation_state": {
            "v7_dch_formula_frozen": go,
            "v7_dch_implementation_authorized": go,
            "return_to_KCS_tokenizer_design": not go,
            "dch_causal_mechanism_established": False,
        },
        "interpretation_boundary": (
            "Frozen same-weight counterfactuals measure immediate forward "
            "effects only.  A GO result authorises one strictly paired fresh "
            "DCH trajectory test; it does not establish a causal mechanism, "
            "stability, paper core, or replacement for formal Gates A--E."
        ),
    }


def build_summary(
    atlas_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    atlas_root = atlas_root.resolve()
    _require(atlas_root.is_dir(), f"Atlas root does not exist: {atlas_root}")

    matrices: Dict[str, Dict[str, Any]] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    validation_rows: Dict[str, Dict[str, Any]] = {}
    input_files: list[Dict[str, Any]] = []
    for lane, lane_spec in EXPECTED_LANES.items():
        matrix_path = (atlas_root / lane / "matrix_summary.json").resolve()
        expected_paths = [
            _expected_diagnostic_path(
                atlas_root,
                lane,
                lane_spec["variant"],
                seed,
                role,
            )
            for seed in SEEDS
            for role in ROLES
        ]
        matrix = load_json_object(matrix_path)
        _validate_matrix(
            matrix,
            matrix_path,
            lane,
            lane_spec,
            expected_paths,
        )
        matrices[lane] = matrix
        input_files.append(
            {
                "kind": "lane_matrix",
                "path": str(matrix_path),
                "sha256": file_sha256(matrix_path),
            }
        )
        for seed in SEEDS:
            for role in ROLES:
                path = _expected_diagnostic_path(
                    atlas_root,
                    lane,
                    lane_spec["variant"],
                    seed,
                    role,
                )
                payload = load_json_object(path)
                key = checkpoint_key(lane_spec["variant"], seed, role)
                _require(key not in payloads, f"Duplicate diagnostic key: {key}")
                validation_rows[key] = _validate_diagnostic(
                    payload,
                    path,
                    lane_spec["variant"],
                    seed,
                    role,
                )
                payloads[key] = payload
                input_files.append(
                    {
                        "kind": "checkpoint_diagnostic",
                        "path": str(path),
                        "sha256": file_sha256(path),
                    }
                )

    expected_keys = {
        checkpoint_key(variant, seed, role)
        for variant in VARIANTS
        for seed in SEEDS
        for role in ROLES
    }
    _require(set(payloads) == expected_keys, "Eight-checkpoint matrix is incomplete")

    gradient_evidence = verify_code_evidence(
        ZERO_SCALE_GRADIENT_CODE_EVIDENCE,
        repo_root,
    )
    hook_evidence = verify_code_evidence(
        HOOK_RESTORATION_CODE_EVIDENCE,
        repo_root,
    )
    for row in validation_rows.values():
        row["hook_restoration_code_evidence_verified"] = True

    checkpoints: Dict[str, Any] = {}
    static_summaries: Dict[str, Any] = {}
    for key in sorted(payloads):
        payload = payloads[key]
        modes = {}
        for mode in MODES:
            mode_payload = payload["modes"][mode]
            modes[mode] = {
                "fixed_threshold_points": {
                    threshold: _compact_point(
                        mode_payload["fixed_threshold_points"][threshold]
                    )
                    for threshold in FIXED_THRESHOLDS
                },
                "best_points_under_fa_budget": {
                    budget: _compact_point(
                        mode_payload["best_points_under_fa_budget"].get(budget)
                    )
                    for budget in FA_BUDGETS
                },
            }
        static = _compact_static(payload)
        static_summaries[key] = static
        checkpoints[key] = {
            "variant": payload["variant"],
            "seed": payload["seed"],
            "checkpoint_role": payload["checkpoint_role"],
            "checkpoint_epoch": payload["checkpoint_epoch"],
            "checkpoint": payload["checkpoint"],
            "formal_sweep": payload["formal_sweep"],
            "validation_split_sha256": payload["validation"][
                "validation_split_sha256"
            ],
            "modes": modes,
            "static_checkpoint_diagnostics": static,
        }

    all_validation_checks = all(
        row["as_trained_formal_max_abs_numeric_delta"] == 0.0
        and row["formal_count_fields_match"] is True
        and row["formal_inputs_sha_before_after_equal"] is True
        and row["formal_inputs_current_sha_match"] is True
        and row["model_state_restored_all_modes"] is True
        and row["runtime_hook_mode_implementation_verified"] is True
        and row["hook_restoration_code_evidence_verified"] is True
        for row in validation_rows.values()
    )
    validation = {
        "checkpoint_count": len(payloads),
        "expected_checkpoint_count": 8,
        "as_trained_formal_sweep_exact_count": sum(
            row["as_trained_formal_max_abs_numeric_delta"] == 0.0
            and row["formal_count_fields_match"]
            for row in validation_rows.values()
        ),
        "input_sha_before_after_equal_count": sum(
            row["formal_inputs_sha_before_after_equal"]
            for row in validation_rows.values()
        ),
        "input_current_sha_match_count": sum(
            row["formal_inputs_current_sha_match"]
            for row in validation_rows.values()
        ),
        "model_state_restored_checkpoint_count": sum(
            row["model_state_restored_all_modes"]
            for row in validation_rows.values()
        ),
        "hook_restoration_code_evidence_verified": True,
        "all_checks_passed": all_validation_checks,
        "checkpoints": validation_rows,
    }
    decision = _decision(
        payloads,
        validation,
        static_summaries,
        gradient_evidence,
    )

    aggregate_rho_l1 = [
        static["aggregate"]["rho_l1"] for static in static_summaries.values()
    ]
    aggregate_rho_l2 = [
        static["aggregate"]["rho_l2"] for static in static_summaries.values()
    ]
    lane_screen_statuses = {
        lane: matrix.get("decision_inputs", {}).get("status")
        for lane, matrix in matrices.items()
    }
    report = {
        "schema": SCHEMA,
        "diagnostic_scope": "frozen_internal_validation_counterfactual_only",
        "training_performed": False,
        "official_test_accessed": False,
        "formal_gate_replacement": False,
        "checkpoint_reselection_permitted": False,
        "atlas_root": str(atlas_root),
        "input_provenance": {
            "files": sorted(input_files, key=lambda item: item["path"]),
            "summarizer_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "code_evidence": {
            "zero_scale_gradient_asymmetry": gradient_evidence,
            "hook_restoration": hook_evidence,
        },
        "validation": validation,
        "lane_frozen_screen_statuses": lane_screen_statuses,
        "checkpoint_summaries": checkpoints,
        "static_checkpoint_range_summary": {
            "checkpoint_count": len(static_summaries),
            "block_rows": len(static_summaries) * 7,
            "aggregate_rho_l1_min": min(aggregate_rho_l1),
            "aggregate_rho_l1_max": max(aggregate_rho_l1),
            "aggregate_rho_l2_min": min(aggregate_rho_l2),
            "aggregate_rho_l2_max": max(aggregate_rho_l2),
            "phase_cancellation_interpretation": (
                "These are static weight ratios; lower values mean stronger "
                "signed phase cancellation.  They are not activation or "
                "causal measurements."
            ),
        },
        "decision": decision,
        "claim_boundaries": {
            "dch_causal_mechanism_established": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "frozen_counterfactual_is_training_trajectory": False,
        },
    }
    return _json_ready(report)


def _fmt(value: float) -> str:
    return f"{float(value):.9g}"


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    conditions = decision["conditions"]
    lines = [
        "# TPD-Clean V6 Failure Atlas 汇总",
        "",
        f"最终判定：`{decision['status']}`。",
        "",
        (
            "该判定只决定是否实现并 fresh-train 一次严格配对的 DCH 轨迹"
            "实验；它不建立 DCH 因果机制、跨种子稳定性或论文核心。"
        ),
        "",
        "## 完整性与只读一致性",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
        (
            "| as-trained 与 formal sweep 完全一致 | "
            f"{report['validation']['as_trained_formal_sweep_exact_count']}/8 |"
        ),
        (
            "| 输入 SHA 诊断前后相同 | "
            f"{report['validation']['input_sha_before_after_equal_count']}/8 |"
        ),
        (
            "| 当前输入文件仍匹配记录 SHA | "
            f"{report['validation']['input_current_sha_match_count']}/8 |"
        ),
        (
            "| 三种模式 model state 完整恢复 | "
            f"{report['validation']['model_state_restored_checkpoint_count']}/8 |"
        ),
        (
            "| residual-off hook 移除代码/单测证据 | "
            f"{str(report['validation']['hook_restoration_code_evidence_verified']).lower()} |"
        ),
        "",
        "## Go/No-Go 条件",
        "",
        "| 条件 | 结果 |",
        "|---|---:|",
    ]
    condition_labels = (
        ("diagnostics_complete", "diagnostics_complete"),
        (
            "zero_scale_gradient_asymmetry_confirmed",
            "zero_scale_gradient_asymmetry_confirmed（代码证据）",
        ),
        ("saliency_scale_not_saturated", "saliency_scale_not_saturated"),
        (
            "residual_off_only_explains_registered_failure",
            "residual_off_only_explains_registered_failure",
        ),
        (
            "context_off_improves_seed3407_primary_fragmentation",
            "Context-off 改善 seed3407/Pd-primary 碎裂",
        ),
        (
            "context_off_improves_at_least_one_gate_A_deficit",
            "Context-off 改善至少一个 Gate-A 缺口",
        ),
    )
    for key, label in condition_labels:
        lines.append(f"| {label} | `{str(conditions[key]).lower()}` |")
    saturation = decision["saliency_scale_saturation_contract"]
    lines.extend(
        [
            "",
            (
                "观测到的 56 个 checkpoint/block 最大值中的全局最大值为 "
                f"`{_fmt(saturation['observed_global_max'])}`，预注册严格上界为 "
                f"`{_fmt(saturation['strict_upper_bound'])}`。"
            ),
            "",
            "## 固定阈值 0.5 主指标与碎裂构成",
            "",
            (
                "| checkpoint | mode | Pd | Fa | mIoU | frag excess | "
                "split GT | in-GT px | near/attached px | background px |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, checkpoint in report["checkpoint_summaries"].items():
        for mode in MODES:
            point = checkpoint["modes"][mode]["fixed_threshold_points"]["0.5"]
            pixels = point["unmatched_component_pixels_by_class"]
            near = (
                pixels["near_gt_duplicate"]
                + pixels["attached_or_near_gt"]
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        key,
                        mode,
                        (
                            f"{point['matched_target_count']}/"
                            f"{point['target_count']}"
                        ),
                        _fmt(point["fa"]),
                        _fmt(point["miou"]),
                        str(point["fragment_excess_total"]),
                        str(point["split_target_count"]),
                        str(pixels["in_gt_fragment"]),
                        str(near),
                        str(pixels["background_false_object"]),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## seed 3407 Full 的 matched-Pd 比较",
            "",
            (
                "| role | mode | source | Pd | Fa | mIoU | frag excess | "
                "registered failure improved |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    matched = decision["matched_pd_comparisons_seed3407_full"]
    for role in ROLES:
        role_data = matched[role]
        for mode in MODES:
            selected = role_data["modes"][mode]
            if selected is None:
                lines.append(
                    f"| {role} | {mode} | unavailable | — | — | — | — | false |"
                )
                continue
            point = selected["point"]
            improved = (
                False
                if mode == "as_trained"
                else role_data["comparisons_to_as_trained"][mode][
                    "registered_failure_improved"
                ]
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        role,
                        mode,
                        selected["source"],
                        (
                            f"{point['matched_target_count']}/"
                            f"{point['target_count']}"
                        ),
                        _fmt(point["fa"]),
                        _fmt(point["miou"]),
                        str(point["fragment_excess_total"]),
                        str(improved).lower(),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 七个 block 的 scale 与 phase-sum 统计",
            "",
            "| checkpoint | block | p90 | max | rho_L1 | rho_L2 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key, checkpoint in report["checkpoint_summaries"].items():
        for block in checkpoint["static_checkpoint_diagnostics"]["blocks"]:
            scales = block["saliency_scale_effective_abs_tanh"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        key,
                        block["block"],
                        _fmt(scales["p90"]),
                        _fmt(scales["max"]),
                        _fmt(block["rho_l1"]),
                        _fmt(block["rho_l2"]),
                    )
                )
                + " |"
            )

    implementation = decision["implementation_state"]
    lines.extend(
        [
            "",
            "## 实施状态与边界",
            "",
            f"- `v7_dch_formula_frozen={str(implementation['v7_dch_formula_frozen']).lower()}`",
            (
                "- `v7_dch_implementation_authorized="
                f"{str(implementation['v7_dch_implementation_authorized']).lower()}`"
            ),
            "- `dch_causal_mechanism_established=false`",
            "- `paper_core_established=false`",
            "- `stability_claim_supported=false`",
            "",
            (
                "Context-off/residual-off 是同权重冻结前向复算，不会重演早期优化"
                "轨迹。`GO_DCH_TRAJECTORY_TEST` 只表示满足了实现并训练一个严格"
                "配对候选的门槛；`CONTEXT_DIRECT_SUPPORT` 也仍不是因果结论。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    atlas_root: Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    json_path = atlas_root / "V6_FAILURE_ATLAS.json"
    markdown_path = atlas_root / "V6_FAILURE_ATLAS.md"
    for path in (json_path, markdown_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {path}; pass --overwrite"
            )
        if path.is_symlink():
            raise ValueError(f"Refusing to replace symlink: {path}")
    serialized = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    markdown = render_markdown(report)
    for path, content in (
        (json_path, serialized),
        (markdown_path, markdown),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"Temporary output exists: {temporary}")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and merge the frozen V6 failure atlas"
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=DEFAULT_ATLAS_ROOT,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.atlas_root = args.atlas_root.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_summary(args.atlas_root)
    json_path, markdown_path = write_outputs(
        args.atlas_root,
        report,
        overwrite=args.overwrite,
    )
    print(
        "ATLAS_SUMMARY_COMPLETE "
        f"decision={report['decision']['status']} "
        f"json={json_path} markdown={markdown_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
