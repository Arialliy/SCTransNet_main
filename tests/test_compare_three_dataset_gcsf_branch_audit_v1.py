from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from analysis import compare_three_dataset_gcsf_branch_audit_v1 as subject


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
VALID_PIXELS = 1000


def _point(
    *,
    matched: int = 100,
    matched_tiny: int = 20,
    miou: float = 0.8,
    niou: float = 0.75,
    component_fp: int = 100,
    background_fp: int = 200,
) -> dict[str, object]:
    return {
        "threshold": 0.5,
        "test_loss": 0.1,
        "target_count": 110,
        "tiny_target_count": 25,
        "matched_target_count": matched,
        "matched_tiny_target_count": matched_tiny,
        "miou": miou,
        "niou": niou,
        "pixel_precision": 0.8,
        "pixel_recall": 0.8,
        "pixel_f1": 0.8,
        "pd": matched / 110,
        "tiny_pd": matched_tiny / 25,
        "fa": component_fp / VALID_PIXELS,
        "unmatched_predicted_pixels": component_fp,
        "predicted_object_count": 110,
        "unmatched_predicted_object_count": 10,
        "false_positive_pixels": background_fp,
        "false_objects_per_image": 0.1,
        "valid_pixel_count": VALID_PIXELS,
    }


def _sweep(fixed: dict[str, object]) -> dict[str, object]:
    first = copy.deepcopy(fixed)
    first["selected_point_is_empty"] = False
    empty = copy.deepcopy(first)
    empty.update(
        {
            "threshold": 1.0,
            "miou": 0.0,
            "niou": 0.0,
            "pixel_precision": 0.0,
            "pixel_recall": 0.0,
            "pixel_f1": 0.0,
            "pd": 0.0,
            "tiny_pd": 0.0,
            "fa": 0.0,
            "matched_target_count": 0,
            "matched_tiny_target_count": 0,
            "predicted_object_count": 0,
            "unmatched_predicted_object_count": 0,
            "unmatched_predicted_pixels": 0,
            "false_positive_pixels": 0,
            "false_objects_per_image": 0.0,
            "selected_point_is_empty": True,
        }
    )
    return {
        "selection_effect": "none",
        "threshold_provenance": {
            "provided_thresholds": True,
            "closed_probability_interval": True,
        },
        "best_points_under_fa_budget": {},
        "pareto_frontier": [copy.deepcopy(first), copy.deepcopy(empty)],
        "points": [first, empty],
    }


def _threshold_roles() -> dict[str, object]:
    return {
        "checkpoint_selection_threshold": 0.5,
        "global_lambda_selection_threshold": 0.5,
        "main_table_threshold": 0.5,
        "descriptive_sweep_only": True,
        "descriptive_sweep_contains_threshold_1_0": True,
        "threshold_1_0_semantics": "empty_prediction_pd0_fa0",
        "sweep_reselects_checkpoint": False,
        "sweep_reselects_global_lambda": False,
    }


def _probability_difference(
    *, max_abs: float = 0.0, absolute_sum: float = 0.0
) -> dict[str, object]:
    mean = absolute_sum / VALID_PIXELS
    different = bool(
        max_abs > subject.qfg_threshold_max() or mean > subject.qfg_threshold_mean()
    )
    return {
        "scope": "all_original_unpadded_test_pixels",
        "element_count": VALID_PIXELS,
        "absolute_difference_sum": absolute_sum,
        "max_abs": max_abs,
        "mean_abs": mean,
        "equivalent": not different,
        "functionally_different": different,
        "equivalence_max_abs_threshold": subject.qfg_threshold_max(),
        "equivalence_mean_abs_threshold": subject.qfg_threshold_mean(),
    }


def _mode(mode: str) -> dict[str, object]:
    fixed = _point()
    return {
        **subject.analyzer.normalize_public_mode(mode),
        "fixed_threshold_0_5": fixed,
        "descriptive_pd_fa": _sweep(fixed),
        "threshold_roles": _threshold_roles(),
        "sweep_thresholds": [0.5, 1.0],
        "probability_difference_to_current": _probability_difference(),
    }


def _statistics() -> dict[str, object]:
    return {
        "schema": subject.analyzer.STATISTICS_SCHEMA,
        "batch_count": 2,
        "level_count": 4,
        "level_order": list(subject.analyzer.LEVEL_NAMES),
        "target_projection": "adaptive_max_pool2d_binary_presence",
        "valid_projection": "adaptive_max_pool2d_any_original_support",
        "background_region": "pooled_valid_and_not_pooled_target",
        "padding_policy": "exclude_bins_with_no_original_pixel_support",
        "cosine_aggregation": "global_masked_dot_over_global_masked_l2_product",
        "amplitude_share_proxy_formula": "RMS(T)/(RMS(T)+2*RMS(E)+1e-12)",
        "feature_tensors_retained_after_batch": False,
        "levels": [
            {
                "level_index_zero_based": index,
                "level_name": subject.analyzer.LEVEL_NAMES[index],
                "channels": subject.analyzer.LEVEL_CHANNELS[index],
                "observed_shapes_chw": [[subject.analyzer.LEVEL_CHANNELS[index], 8, 8]],
                "valid_spatial_location_count": 100,
                "target_spatial_location_count": 10,
                "background_spatial_location_count": 90,
                "transformed_rms": 0.5,
                "encoder_rms": 1.0,
                "transformed_to_encoder_rms_ratio": 0.5,
                "transformed_encoder_cosine": 0.2,
                "target_transformed_rms": 0.8,
                "target_encoder_rms": 1.2,
                "background_transformed_rms": 0.4,
                "background_encoder_rms": 0.9,
                "transformed_target_to_background_rms_ratio": 2.0,
                "encoder_target_to_background_rms_ratio": 4.0 / 3.0,
                "current_transformed_amplitude_share_proxy": 0.2,
            }
            for index in range(4)
        ],
    }


def _payload(dataset: str, role: str) -> dict[str, object]:
    return {
        "schema": subject.ANALYZER_SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": subject.analyzer.REFERENCE_METHOD,
        "training_model_method": subject.analyzer.TRAINING_MODEL_METHOD,
        "checkpoint_role": role,
        "seed": 42,
        "test_selected": True,
        "selection_is_optimistic": True,
        "evaluation_protocol": subject.analyzer.EVALUATION_PROTOCOL,
        "fixed_threshold": 0.5,
        "sweep_thresholds": [0.5, 1.0],
        "mode_order": list(subject.MODES),
        "modes": {mode: _mode(mode) for mode in subject.MODES},
        "branch_statistics": _statistics(),
        "execution_audit": {
            "batch_count": 2,
            "encoder_tpd_qfg_prepare_count": 2,
            "decoder_mode_count_per_batch": 11,
            "decoder_execution_count": 22,
            "encoder_tpd_qfg_recomputed_per_mode": False,
            "forward_local_feature_reuse_only": True,
        },
        "restoration_audit": {
            "model_state_sha256_before": SHA_C,
            "model_state_sha256_after": SHA_C,
            "model_state_unchanged": True,
        },
        "reference_replay_audit": {
            "passed": True,
            "comparison": f"current_g0_fixed_threshold_0_5_vs_existing_{role}",
            "compared": {"threshold": {"absolute_difference": 0.0}},
        },
        "checkpoint_binding": {
            "checkpoint": {"sha256": SHA_C, "role": role},
            "protocol": {"payload_sha256": SHA_D},
        },
        "data": {
            "protocol_manifest": {"sha256": SHA_A},
            "inference_order_newline_sha256": SHA_B,
            "split": "img_idx/test",
        },
        "reference_reuse": {"sha256": SHA_D, "checkpoint_role": role},
        "source_sha256": {
            "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py": SHA_A
        },
        "probability_arrays_persisted": False,
        "feature_tensors_persisted": False,
        "probability_cache_written": False,
        "feature_cache_written": False,
        "derived_checkpoint_written": False,
        "intervention_contract": {
            "family": "GCSF_constant_sum_representable_counterfactual",
            "current_formula_operation_order": "(T+E)+E",
            "selected_correction_operation_order": "baseline+(g*T-g*E)",
            "unrepresentable_f1_t_plus_e_used_for_trigger": False,
            "unrepresentable_f3_2t_plus_e_used_for_trigger": False,
            "model_state_modified": False,
            "derived_checkpoint_written": False,
        },
    }


def _payloads() -> dict[str, dict[str, object]]:
    return {
        subject._binding_key(dataset, role): _payload(dataset, role)
        for dataset in subject.DATASETS
        for role in subject.CHECKPOINT_ROLES
    }


def _bindings() -> dict[str, dict[str, str]]:
    return {
        key: {"path": f"/synthetic/{key}.json", "sha256": SHA_A}
        for key in subject._expected_keys()
    }


def _set_point(
    payloads: dict[str, dict[str, object]],
    dataset: str,
    role: str,
    mode: str,
    **updates: object,
) -> None:
    key = subject._binding_key(dataset, role)
    modes = payloads[key]["modes"]
    assert isinstance(modes, dict)
    raw_mode = modes[mode]
    assert isinstance(raw_mode, dict)
    point = raw_mode["fixed_threshold_0_5"]
    assert isinstance(point, dict)
    aliases = {
        "targets": "matched_target_count",
        "tiny": "matched_tiny_target_count",
        "component_fp": "unmatched_predicted_pixels",
        "background_fp": "false_positive_pixels",
    }
    point.update({aliases.get(name, name): value for name, value in updates.items()})
    point["pd"] = int(point["matched_target_count"]) / int(point["target_count"])
    point["tiny_pd"] = int(point["matched_tiny_target_count"]) / int(
        point["tiny_target_count"]
    )
    point["fa"] = int(point["unmatched_predicted_pixels"]) / int(
        point["valid_pixel_count"]
    )


def test_neutral_matrix_does_not_authorize_training() -> None:
    result = subject.compare_payloads(_payloads(), input_bindings=_bindings())
    assert result["decision"] == subject.DECISION_NO_AUTHORIZATION
    assert result["trigger_a"]["passed"] is False
    assert result["trigger_a"]["qualifying_modes"] == []
    assert result["gcsf_v1_implementation_and_pilot_authorized"] is False
    assert result["trigger_b"]["implemented"] is False
    assert result["trigger_b"]["authorizes_training"] is False
    assert result["trigger_c"]["implemented"] is False
    assert result["trigger_c"]["authorizes_training"] is False


def test_same_nonzero_mode_passes_two_primary_datasets_and_all_six_veto() -> None:
    payloads = _payloads()
    mode = "gpos025_all_levels"
    for dataset in subject.DATASETS[:2]:
        _set_point(payloads, dataset, "best_miou", mode, miou=0.806)
    result = subject.compare_payloads(payloads, input_bindings=_bindings())
    assert result["decision"] == subject.DECISION_AUTHORIZE
    assert result["trigger_a"]["passed"] is True
    assert result["trigger_a"]["qualifying_modes"] == [mode]
    row = result["trigger_a"]["modes"][mode]
    assert row["primary_safe_material_dataset_count"] == 2
    assert row["severe_degradation_unit_count"] == 0
    assert result["gcsf_v1_implementation_and_pilot_authorized"] is True


def test_best_pd_severe_veto_blocks_otherwise_qualifying_mode() -> None:
    payloads = _payloads()
    mode = "gneg025_l1_only"
    for dataset in subject.DATASETS[:2]:
        _set_point(payloads, dataset, "best_miou", mode, niou=0.756)
    _set_point(payloads, subject.DATASETS[2], "best_pd", mode, targets=98)
    result = subject.compare_payloads(payloads, input_bindings=_bindings())
    assert result["trigger_a"]["passed"] is False
    row = result["trigger_a"]["modes"][mode]
    assert row["primary_safe_material_dataset_count"] == 2
    assert row["severe_degradation_unit_count"] == 1
    assert row["severe_degradation_units_across_six_roles"] == [
        subject._binding_key(subject.DATASETS[2], "best_pd")
    ]


def test_improvements_in_different_modes_do_not_combine() -> None:
    payloads = _payloads()
    _set_point(payloads, subject.DATASETS[0], "best_miou", "gneg025_l1_only", miou=0.806)
    _set_point(payloads, subject.DATASETS[1], "best_miou", "gpos025_l1_only", miou=0.806)
    result = subject.compare_payloads(payloads, input_bindings=_bindings())
    assert result["trigger_a"]["passed"] is False
    assert all(
        row["primary_safe_material_dataset_count"] <= 1
        for row in result["trigger_a"]["modes"].values()
    )


def test_descriptive_branch_statistics_cannot_authorize() -> None:
    payloads = _payloads()
    for payload in payloads.values():
        statistics = payload["branch_statistics"]
        assert isinstance(statistics, dict)
        levels = statistics["levels"]
        assert isinstance(levels, list)
        for row in levels:
            assert isinstance(row, dict)
            row["transformed_target_to_background_rms_ratio"] = 100.0
            row["encoder_target_to_background_rms_ratio"] = 0.01
            row["current_transformed_amplitude_share_proxy"] = 0.001
    result = subject.compare_payloads(payloads, input_bindings=_bindings())
    assert result["trigger_a"]["passed"] is False
    assert result["gcsf_v1_implementation_and_pilot_authorized"] is False


def test_json_roundtrip_and_write_once(tmp_path: Path) -> None:
    payloads = json.loads(json.dumps(_payloads(), allow_nan=False))
    result = subject.compare_payloads(payloads, input_bindings=_bindings())
    subject.validate_comparison_payload(result)
    result = json.loads(json.dumps(result, allow_nan=False))
    subject.validate_comparison_payload(result)
    json_path = tmp_path / "decision.json"
    md_path = tmp_path / "decision.md"
    subject.write_outputs(json_path, md_path, result)
    assert json.loads(json_path.read_text(encoding="utf-8"))["decision"] == (
        subject.DECISION_NO_AUTHORIZATION
    )
    assert "Trigger A" in md_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        subject.write_outputs(json_path, md_path, result)


def test_input_parser_requires_all_six_or_uses_defaults() -> None:
    defaults = subject._parse_bindings([])
    assert set(defaults) == set(subject._expected_keys())
    with pytest.raises(ValueError, match="all six"):
        subject._parse_bindings(
            [f"{subject._expected_keys()[0]}=/tmp/one.json"]
        )


def test_source_hashes_are_complete_and_real() -> None:
    result = subject.compare_payloads(_payloads(), input_bindings=_bindings())
    sources = result["source_sha256"]
    assert set(sources) == {
        "analysis/compare_three_dataset_gcsf_branch_audit_v1.py",
        "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py",
        "analysis/compare_three_dataset_qfg_level_knockout_v1.py",
    }
    assert all(len(value) == 64 for value in sources.values())
