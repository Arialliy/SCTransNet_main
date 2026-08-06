from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from analysis import compare_three_dataset_dorf_v1 as subject


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
VALID_PIXELS = 1000
IMAGE_COUNT = 100
INPUT_MANIFEST = json.loads(subject.DEFAULT_INPUT_MANIFEST.read_text(encoding="utf-8"))
INPUT_MANIFEST_BINDING = {
    "path": str(subject.DEFAULT_INPUT_MANIFEST.resolve()),
    "sha256": subject.FROZEN_INPUT_MANIFEST_SHA256,
}
MANIFEST_ENTRIES = {
    subject._binding_key(
        entry["method"], entry["dataset"], entry["checkpoint_role"]
    ): entry
    for entry in INPUT_MANIFEST["entries"]
}


def _point(
    *,
    matched: int = 100,
    matched_tiny: int = 20,
    miou: float = 0.8,
    niou: float = 0.75,
    component_fp: int = 100,
    background_fp: int = 200,
) -> dict[str, object]:
    unmatched_objects = 10
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
        "unmatched_predicted_object_count": unmatched_objects,
        "false_positive_pixels": background_fp,
        "false_objects_per_image": unmatched_objects / IMAGE_COUNT,
        "valid_pixel_count": VALID_PIXELS,
        "image_count": IMAGE_COUNT,
    }


def _sweep(fixed: dict[str, object]) -> dict[str, object]:
    first = copy.deepcopy(fixed)
    first["selected_point_is_empty"] = False
    empty = copy.deepcopy(first)
    empty.update(
        {
            "threshold": 1.0,
            "matched_target_count": 0,
            "matched_tiny_target_count": 0,
            "unmatched_predicted_pixels": 0,
            "predicted_object_count": 0,
            "unmatched_predicted_object_count": 0,
            "false_positive_pixels": 0,
            "false_objects_per_image": 0.0,
            "pd": 0.0,
            "tiny_pd": 0.0,
            "fa": 0.0,
            "miou": 0.0,
            "niou": 0.0,
            "pixel_precision": 0.0,
            "pixel_recall": 0.0,
            "pixel_f1": 0.0,
            "selected_point_is_empty": True,
        }
    )
    return {"points": [first, empty], "pareto_frontier": [first, empty]}


def _threshold_roles() -> dict[str, object]:
    return {
        "checkpoint_selection_threshold": 0.5,
        "global_lambda_selection_threshold": 0.5,
        "main_table_threshold": 0.5,
        "descriptive_sweep_only": True,
        "descriptive_sweep_contains_threshold_1_0": True,
        "threshold_1_0_semantics": "empty_prediction_pd0_fa0",
    }


def _mode(mode: str) -> dict[str, object]:
    fixed = _point()
    if mode == subject.CURRENT_MODE:
        maximum = absolute_sum = 0.0
    else:
        maximum = 0.1
        absolute_sum = 10.0
    return {
        "mode": mode,
        "alpha": subject.ALPHA_BY_MODE[mode],
        "fixed_threshold_0_5": fixed,
        "descriptive_pd_fa": _sweep(fixed),
        "threshold_roles": _threshold_roles(),
        "probability_difference_to_current": {
            "element_count": VALID_PIXELS,
            "absolute_difference_sum": absolute_sum,
            "max_abs": maximum,
            "mean_abs": absolute_sum / VALID_PIXELS,
        },
    }


def _payload(method: str, dataset: str, role: str) -> dict[str, object]:
    key = subject._binding_key(method, dataset, role)
    entry = MANIFEST_ENTRIES[key]
    run_dir = (subject.REPO_ROOT / entry["run_dir"]).resolve()
    data_protocol = INPUT_MANIFEST["data_protocol_manifest"]
    background = INPUT_MANIFEST["background_pixel_authority"]
    return {
        "schema": subject.ANALYZER_SCHEMA,
        "status": "complete",
        "method": method,
        "training_model_method": "final" if method == "final_tss_off" else "original",
        "dataset": dataset,
        "checkpoint_role": role,
        "seed": 42,
        "test_selected": True,
        "selection_is_optimistic": True,
        "mode_order": list(subject.MODES),
        "modes": {mode: _mode(mode) for mode in subject.MODES},
        "checkpoint_binding": {
            "run_dir": str(run_dir),
            "input_manifest_entry_key": key,
            "checkpoint": {
                "role": role,
                "epoch": entry["epoch"],
                "path": str(run_dir / "checkpoints" / f"{role}.pth.tar"),
                "sha256": entry["checkpoint_sha256"],
            },
            "summary": {
                "path": str(run_dir / "summary.json"),
                "sha256": entry["summary_sha256"],
            },
            "protocol": {
                "path": str(run_dir / "protocol.json"),
                "sha256": entry["protocol_sha256"],
            },
            "training_state_dict_sha256": SHA_D,
        },
        "reference_evaluation_binding": {
            "checkpoint_role": role,
            "path": str(run_dir / "evaluations" / f"{role}.json"),
            "sha256": entry["evaluation_sha256"],
            "source": "historical_evaluation_fixed_threshold_0_5",
            "checkpoint_embedded_metrics_fallback_allowed": False,
        },
        "input_manifest_binding": {
            "path": str(subject.DEFAULT_INPUT_MANIFEST.resolve()),
            "sha256": subject.FROZEN_INPUT_MANIFEST_SHA256,
            "schema": subject.INPUT_MANIFEST_SCHEMA,
            "status": "frozen_before_dorf_outputs",
            "entry_key": key,
            "entry": copy.deepcopy(entry),
            "data_protocol_manifest": {
                "path": str((subject.REPO_ROOT / data_protocol["path"]).resolve()),
                "sha256": data_protocol["sha256"],
            },
            "background_pixel_authority": {
                "path": str((subject.REPO_ROOT / background["path"]).resolve()),
                "sha256": background["sha256"],
            },
            "historical_metric_authority": "bound_evaluation_json_only",
            "checkpoint_embedded_metrics_fallback_allowed": False,
            "verified_before_model_load": True,
            "verified_after_inference": True,
        },
        "source_sha256": {"analysis/analyze_three_dataset_dorf_v1.py": SHA_A},
        "model_metadata": {
            "method": method,
            "architecture": "SCTransNet",
            "dorf_loader_audit": {
                "passed": True,
                "builder": subject.EXPECTED_BUILDERS[method],
                "training_state_key_count": subject.EXPECTED_TRAINING_STATE_KEYS[method],
                "expected_training_state_key_count": subject.EXPECTED_TRAINING_STATE_KEYS[method],
                "removed_training_only_tss_state_key_count": subject.EXPECTED_REMOVED_TSS_KEYS[method],
                "inference_state_key_count": subject.EXPECTED_INFERENCE_STATE_KEYS[method],
                "strict_load": True,
                "training_flag": False,
                "mode": "test",
            },
        },
        "alpha0_historical_replay_audit": {
            "passed": True,
            "counts_exact": True,
            "background_false_positive_pixels_exact": True,
            "within_frozen_float_tolerances": True,
            "exact": True,
            "mode": subject.CURRENT_MODE,
            "checkpoint_role": role,
            "reference_evaluation_sha256": entry["evaluation_sha256"],
            "background_pixel_authority_sha256": background["sha256"],
        },
        "background_pixel_authority_record": {
            "dataset": dataset,
            "checkpoint_role": role,
            "checkpoint_epoch": entry["epoch"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "evaluation_sha256": entry["evaluation_sha256"],
            "false_positive_pixels": 200,
            "valid_pixel_count": VALID_PIXELS,
        },
        "engineering_audit": {
            "passed": True,
            "all_metrics_finite": True,
            "same_d0_out_logits_reused_for_all_modes": True,
            "one_model_forward_per_batch": True,
            "raw_logit_fusion": True,
            "model_state_unchanged": True,
            "model_training_flag_unchanged": True,
            "model_mode_unchanged": True,
            "model_training_flag_before": False,
            "model_training_flag_after": False,
            "model_mode_before": "test",
            "model_mode_after": "test",
            "derived_checkpoint_written": False,
            "probability_cache_written": False,
            "batch_count": IMAGE_COUNT,
            "model_forward_count": IMAGE_COUNT,
            "outc_hook_count": IMAGE_COUNT,
            "outconv_hook_count": IMAGE_COUNT,
            "each_hook_exactly_once_per_batch": True,
            "returned_probability_equals_sigmoid_raw_out_bitwise": True,
            "temporary_hooks_restored": True,
            "source_sha256_reverified_after_inference": True,
            "input_manifest_reverified_after_inference": True,
            "alpha0_historical_replay_passed": True,
            "model_state_sha256_before": SHA_D,
            "model_state_sha256_after": SHA_D,
        },
        "data": {
            "split": "img_idx/test",
            "protocol_manifest": {
                "path": str((subject.REPO_ROOT / data_protocol["path"]).resolve()),
                "sha256": data_protocol["sha256"],
            },
            "input_binding": {
                "sha256": SHA_C,
                "ordered_ids_newline_sha256": SHA_B,
                "sample_count": IMAGE_COUNT,
            },
        },
        "intervention_contract": {
            "family": "DORF_V1_existing_deep_supervision_readout_reuse",
            "formula": "z_out + alpha * (z_d0 - z_out)",
            "fusion_space": "raw_logits_before_sigmoid",
            "alphas": [subject.ALPHA_BY_MODE[mode] for mode in subject.MODES],
            "one_checkpoint_per_unit": True,
            "model_parameters_changed": False,
            "persistent_buffers_changed": False,
            "derived_checkpoint_written": False,
        },
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
    }


def _payloads() -> dict[str, dict[str, object]]:
    return {
        subject._binding_key(method, dataset, role): _payload(method, dataset, role)
        for method in subject.METHODS
        for dataset in subject.DATASETS
        for role in subject.CHECKPOINT_ROLES
    }


def _bindings() -> dict[str, dict[str, str]]:
    return {
        key: {"path": f"/synthetic/{key}.json", "sha256": SHA_A}
        for key in subject._expected_keys()
    }


def _compare(payloads: dict[str, dict[str, object]]) -> dict[str, object]:
    return subject.compare_payloads(
        payloads,
        input_bindings=_bindings(),
        input_manifest=copy.deepcopy(INPUT_MANIFEST),
        input_manifest_binding=dict(INPUT_MANIFEST_BINDING),
    )


def _set_point(
    payloads: dict[str, dict[str, object]],
    method: str,
    dataset: str,
    role: str,
    mode: str,
    **updates: object,
) -> None:
    payload = payloads[subject._binding_key(method, dataset, role)]
    modes = payload["modes"]
    assert isinstance(modes, dict)
    record = modes[mode]
    assert isinstance(record, dict)
    point = record["fixed_threshold_0_5"]
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


def _make_final_primary_gain(payloads, mode="dorf_a025"):
    for dataset in subject.DATASETS[:2]:
        _set_point(
            payloads,
            subject.PRIMARY_METHOD,
            dataset,
            subject.PRIMARY_ROLE,
            mode,
            miou=0.806,
        )


def test_neutral_matrix_does_not_authorize() -> None:
    result = _compare(_payloads())
    assert result["decision"] == subject.DECISION_NO_AUTHORIZATION
    assert result["trigger_a"]["passed"] is False
    assert result["trigger_a"]["selected_mode"] is None
    assert result["dorf_v1_production_implementation_authorized"] is False


def test_final_primary_gate_passes_and_smallest_qualifying_alpha_is_selected() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads, "dorf_a025")
    _make_final_primary_gain(payloads, "dorf_a050")
    result = _compare(payloads)
    assert result["trigger_a"]["qualifying_modes"] == ["dorf_a025", "dorf_a050"]
    assert result["trigger_a"]["selected_mode"] == "dorf_a025"
    assert result["trigger_a"]["selected_alpha"] == 0.25
    assert result["decision"] == subject.DECISION_AUTHORIZE
    assert result["fresh_formal1000_launch_authorized_by_this_comparator"] is False


def test_final_best_pd_severe_veto_blocks_an_otherwise_passing_alpha() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    _set_point(
        payloads,
        subject.PRIMARY_METHOD,
        subject.DATASETS[2],
        "best_pd",
        "dorf_a025",
        targets=98,
    )
    result = _compare(payloads)
    row = result["trigger_a"]["modes"]["dorf_a025"]
    assert row["final_primary_safe_material_dataset_count"] == 2
    assert row["final_severe_unit_count"] == 1
    assert row["trigger_a_passed"] is False


def test_shared_alpha_anchor_new_severe_condition_is_a_veto() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    _set_point(
        payloads,
        subject.BASELINE_METHOD,
        subject.DATASETS[2],
        "best_pd",
        "dorf_a025",
        targets=102,
    )
    result = _compare(payloads)
    row = result["trigger_a"]["modes"]["dorf_a025"]
    assert row["baseline_final_zero_vs_original_zero_severe_condition_true_count"] == 0
    assert row["final_alpha_vs_original_zero_new_severe_condition_count"] == 0
    assert row["final_alpha_vs_original_alpha_new_severe_condition_count"] == 1
    assert row["final_alpha_vs_original_alpha_severe_mask_subset_of_baseline"] is False
    assert row["final_vs_original_competitiveness_non_degrading"] is False
    assert row["trigger_a_passed"] is False


def test_equal_severe_counts_cannot_hide_condition_migration() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    # M0 contains one target-count severe cell in the third role.
    _set_point(
        payloads,
        subject.BASELINE_METHOD,
        subject.DATASETS[2],
        "best_pd",
        subject.CURRENT_MODE,
        targets=102,
    )
    # The shared-alpha anchor resolves that cell but creates the same condition
    # in another dataset.  A scalar-count gate would incorrectly accept it.
    _set_point(
        payloads,
        subject.BASELINE_METHOD,
        subject.DATASETS[1],
        "best_pd",
        "dorf_a025",
        targets=102,
    )
    result = _compare(payloads)
    row = result["trigger_a"]["modes"]["dorf_a025"]
    assert row["baseline_final_zero_vs_original_zero_severe_condition_true_count"] == 1
    assert row["final_alpha_vs_original_alpha_severe_condition_true_count"] == 1
    assert row["final_alpha_vs_original_alpha_new_severe_condition_count"] == 1
    assert row["final_alpha_vs_original_alpha_severe_mask_subset_of_baseline"] is False
    assert row["trigger_a_passed"] is False


def test_fixed_original_anchor_blocks_a_gain_caused_by_original_degradation() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    dataset = subject.DATASETS[2]
    _set_point(
        payloads,
        subject.BASELINE_METHOD,
        dataset,
        "best_pd",
        subject.CURRENT_MODE,
        targets=101,
    )
    _set_point(
        payloads,
        subject.PRIMARY_METHOD,
        dataset,
        "best_pd",
        "dorf_a025",
        targets=99,
    )
    _set_point(
        payloads,
        subject.BASELINE_METHOD,
        dataset,
        "best_pd",
        "dorf_a025",
        targets=99,
    )
    result = _compare(payloads)
    row = result["trigger_a"]["modes"]["dorf_a025"]
    assert row["final_severe_unit_count"] == 0
    assert row["final_alpha_vs_original_alpha_new_severe_condition_count"] == 0
    assert row["final_alpha_vs_original_zero_new_severe_condition_count"] == 1
    assert row["final_alpha_vs_original_zero_severe_mask_subset_of_baseline"] is False
    assert row["trigger_a_passed"] is False


def test_existing_alpha0_primary_positive_cell_must_survive_both_anchors() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    dataset = subject.DATASETS[2]
    _set_point(
        payloads,
        subject.PRIMARY_METHOD,
        dataset,
        subject.PRIMARY_ROLE,
        subject.CURRENT_MODE,
        miou=0.806,
    )
    _set_point(
        payloads,
        subject.PRIMARY_METHOD,
        dataset,
        subject.PRIMARY_ROLE,
        "dorf_a025",
        miou=0.8,
    )
    result = _compare(payloads)
    row = result["trigger_a"]["modes"]["dorf_a025"]
    assert row["baseline_primary_best_miou_safe_material_datasets"] == [dataset]
    assert row[
        "final_alpha_vs_original_zero_missing_primary_safe_material_datasets"
    ] == [dataset]
    assert row[
        "final_alpha_vs_original_alpha_missing_primary_safe_material_datasets"
    ] == [dataset]
    assert row["final_vs_original_competitiveness_non_degrading"] is False
    assert row["trigger_a_passed"] is False


def test_shared_alpha_anchor_also_preserves_alpha0_primary_positive_cells() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    dataset = subject.DATASETS[2]
    _set_point(
        payloads,
        subject.PRIMARY_METHOD,
        dataset,
        subject.PRIMARY_ROLE,
        subject.CURRENT_MODE,
        miou=0.806,
    )
    _set_point(
        payloads,
        subject.PRIMARY_METHOD,
        dataset,
        subject.PRIMARY_ROLE,
        "dorf_a025",
        miou=0.812,
    )
    _set_point(
        payloads,
        subject.BASELINE_METHOD,
        dataset,
        subject.PRIMARY_ROLE,
        "dorf_a025",
        miou=0.812,
    )
    result = _compare(payloads)
    row = result["trigger_a"]["modes"]["dorf_a025"]
    assert row["final_alpha_vs_original_zero_primary_safe_material_preserved"] is True
    assert row["final_alpha_vs_original_alpha_primary_safe_material_preserved"] is False
    assert row[
        "final_alpha_vs_original_alpha_missing_primary_safe_material_datasets"
    ] == [dataset]
    assert row["trigger_a_passed"] is False


def test_original_own_severe_is_reported_but_not_a_direct_final_veto() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    _set_point(
        payloads,
        subject.BASELINE_METHOD,
        subject.DATASETS[2],
        "best_pd",
        "dorf_a025",
        targets=98,
    )
    result = _compare(payloads)
    original = result["original_own_dorf_gain_descriptive_only"]["dorf_a025"][
        "best_pd"
    ][subject.DATASETS[2]]
    assert original["severe_degradation"] is True
    assert result["trigger_a"]["modes"]["dorf_a025"]["trigger_a_passed"] is True


def test_bitwise_replay_is_descriptive_but_tolerance_or_engineering_failure_blocks() -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    one = payloads[subject._expected_keys()[0]]
    replay = one["alpha0_historical_replay_audit"]
    assert isinstance(replay, dict)
    replay["exact"] = False
    result = _compare(payloads)
    assert result["trigger_a"]["all_twelve_alpha0_bitwise_exact_descriptive_only"] is False
    assert result["trigger_a"]["all_twelve_alpha0_historical_replay_passed"] is True
    assert result["trigger_a"]["passed"] is True

    payloads = _payloads()
    _make_final_primary_gain(payloads)
    one = payloads[subject._expected_keys()[0]]
    replay = one["alpha0_historical_replay_audit"]
    engineering = one["engineering_audit"]
    assert isinstance(replay, dict) and isinstance(engineering, dict)
    replay["within_frozen_float_tolerances"] = False
    replay["passed"] = False
    engineering["alpha0_historical_replay_passed"] = False
    result = _compare(payloads)
    assert result["trigger_a"]["all_twelve_alpha0_historical_replay_passed"] is False
    assert result["trigger_a"]["passed"] is False

    payloads = _payloads()
    _make_final_primary_gain(payloads)
    engineering = payloads[subject._expected_keys()[0]]["engineering_audit"]
    assert isinstance(engineering, dict)
    engineering["model_state_unchanged"] = False
    result = _compare(payloads)
    assert result["trigger_a"]["all_twelve_engineering_valid"] is False
    assert result["trigger_a"]["passed"] is False


def test_zero_reference_fp_and_all_frozen_boundaries_are_recomputed() -> None:
    reference = subject.validate_analyzer_payload(
        _payload(subject.PRIMARY_METHOD, subject.DATASETS[0], "best_miou")
    )["modes"][subject.CURRENT_MODE]["fixed_threshold_0_5"]
    candidate = dict(reference)
    reference = dict(reference)
    reference["component_false_positive_pixels"] = 0
    candidate["component_false_positive_pixels"] = 1
    direction = subject.compare_direction(candidate, reference)
    assert direction["component_fp_reduction"]["introduced_from_zero"] is True
    assert direction["severe_degradation"] is True

    candidate = dict(reference)
    candidate["matched_target_count"] -= 2
    direction = subject.compare_direction(candidate, reference)
    assert direction["safe"] is False
    assert direction["severe_degradation"] is True


def test_literal_schema_alpha_checkpoint_source_and_replay_bindings_are_strict() -> None:
    payload = _payload(subject.PRIMARY_METHOD, subject.DATASETS[0], "best_miou")
    subject.validate_analyzer_payload(payload)
    mutations = []
    wrong_schema = copy.deepcopy(payload)
    wrong_schema["schema"] = "wrong"
    mutations.append(wrong_schema)
    wrong_alpha = copy.deepcopy(payload)
    wrong_alpha["modes"]["dorf_a025"]["alpha"] = 0.3
    mutations.append(wrong_alpha)
    wrong_role = copy.deepcopy(payload)
    wrong_role["checkpoint_binding"]["checkpoint"]["role"] = "best_pd"
    mutations.append(wrong_role)
    wrong_source = copy.deepcopy(payload)
    wrong_source["source_sha256"] = {}
    mutations.append(wrong_source)
    wrong_replay = copy.deepcopy(payload)
    wrong_replay["alpha0_historical_replay_audit"]["reference_evaluation_sha256"] = SHA_A
    mutations.append(wrong_replay)
    for corrupted in mutations:
        with pytest.raises(subject.DORFComparisonError):
            subject.validate_analyzer_payload(corrupted)


def test_frozen_manifest_sha_order_and_all_twelve_artifact_prebindings_are_strict() -> None:
    payloads = _payloads()
    with pytest.raises(subject.DORFComparisonError, match="manifest SHA"):
        subject.compare_payloads(
            payloads,
            input_bindings=_bindings(),
            input_manifest=copy.deepcopy(INPUT_MANIFEST),
            input_manifest_binding={
                "path": str(subject.DEFAULT_INPUT_MANIFEST.resolve()),
                "sha256": SHA_A,
            },
        )

    manifest = copy.deepcopy(INPUT_MANIFEST)
    manifest["entries"][0], manifest["entries"][1] = (
        manifest["entries"][1],
        manifest["entries"][0],
    )
    with pytest.raises(subject.DORFComparisonError, match="entry order"):
        subject.compare_payloads(
            payloads,
            input_bindings=_bindings(),
            input_manifest=manifest,
            input_manifest_binding=dict(INPUT_MANIFEST_BINDING),
        )

    payloads = _payloads()
    first = payloads[subject._expected_keys()[0]]
    checkpoint = first["checkpoint_binding"]["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["sha256"] = SHA_A
    with pytest.raises(subject.DORFComparisonError, match="checkpoint prebinding"):
        _compare(payloads)


def test_input_parser_is_twelve_role_and_defaults_match_frozen_paths() -> None:
    defaults = subject._parse_bindings([])
    assert set(defaults) == set(subject._expected_keys())
    assert all("results/three_dataset_dorf_v1/runs" in str(path) for path in defaults.values())
    with pytest.raises(subject.DORFComparisonError, match="all twelve"):
        subject._parse_bindings([f"{subject._expected_keys()[0]}=/tmp/one.json"])


def test_cli_outputs_are_byte_identical_under_normal_and_optimized_python(tmp_path: Path) -> None:
    payloads = _payloads()
    _make_final_primary_gain(payloads)
    input_args: list[str] = []
    for index, key in enumerate(subject._expected_keys()):
        path = tmp_path / f"input_{index:02d}.json"
        path.write_text(
            json.dumps(payloads[key], ensure_ascii=False, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        input_args.extend(["--input", f"{key}={path}"])
    script = Path(subject.__file__).resolve()
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "random"
    outputs = []
    for label, optimized in (("normal", False), ("optimized", True)):
        json_path = tmp_path / f"{label}.json"
        md_path = tmp_path / f"{label}.md"
        command = [sys.executable]
        if optimized:
            command.append("-O")
        command.extend(
            [str(script), *input_args, "--output-json", str(json_path), "--output-markdown", str(md_path)]
        )
        subprocess.run(
            command,
            cwd=subject.REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append((json_path.read_bytes(), md_path.read_bytes()))
    assert outputs[0] == outputs[1]


def test_write_once_and_comparison_roundtrip(tmp_path: Path) -> None:
    result = _compare(_payloads())
    subject.validate_comparison_payload(result)
    roundtrip = json.loads(json.dumps(result, allow_nan=False))
    subject.validate_comparison_payload(roundtrip)
    json_path = tmp_path / "decision.json"
    markdown_path = tmp_path / "decision.md"
    subject.write_outputs(json_path, markdown_path, result)
    assert json_path.is_file() and markdown_path.is_file()
    with pytest.raises(FileExistsError):
        subject.write_outputs(json_path, markdown_path, result)
