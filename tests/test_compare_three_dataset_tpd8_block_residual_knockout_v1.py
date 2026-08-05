#!/usr/bin/env python3
"""CPU-only tests for the frozen TPD8 block-residual comparator."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from analysis import compare_three_dataset_tpd8_block_residual_knockout_v1 as subject


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
VALID_PIXELS = 1000


def _point(
    *,
    targets: int = 100,
    tiny: int = 20,
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
        "matched_target_count": targets,
        "matched_tiny_target_count": tiny,
        "miou": miou,
        "niou": niou,
        "pixel_precision": 0.8,
        "pixel_recall": 0.8,
        "pixel_f1": 0.8,
        "pd": targets / 110,
        "tiny_pd": tiny / 25,
        "fa": component_fp / VALID_PIXELS,
        "unmatched_predicted_pixels": component_fp,
        "predicted_object_count": 110,
        "unmatched_predicted_object_count": 10,
        "false_positive_pixels": background_fp,
        "false_objects_per_image": 0.1,
        "valid_pixel_count": VALID_PIXELS,
    }


def _sweep(fixed: dict[str, object]) -> dict[str, object]:
    fixed_point = {
        key: copy.deepcopy(value)
        for key, value in fixed.items()
        if key != "false_positive_pixels"
    }
    fixed_point["selected_point_is_empty"] = False
    empty = copy.deepcopy(fixed_point)
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
        "pareto_frontier": [copy.deepcopy(fixed_point), copy.deepcopy(empty)],
        "points": [fixed_point, empty],
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
    *,
    max_abs: float = 0.0,
    absolute_sum: float = 0.0,
    element_count: int = VALID_PIXELS,
) -> dict[str, object]:
    mean_abs = absolute_sum / element_count
    different = bool(
        max_abs > subject.PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD
        or mean_abs > subject.PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD
    )
    return {
        "scope": "all_original_unpadded_test_pixels",
        "max_abs": max_abs,
        "absolute_difference_sum": absolute_sum,
        "mean_abs": mean_abs,
        "element_count": element_count,
        "equivalent": not different,
        "functionally_different": different,
        "equivalence_max_abs_threshold": (
            subject.PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD
        ),
        "equivalence_mean_abs_threshold": (
            subject.PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD
        ),
    }


def _source_scale_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, path in enumerate(subject.BLOCK_PATHS):
        channels = 32 if index < 4 else 64
        rows.append(
            {
                "block_index_zero_based": index,
                "block_path": path,
                "channels": channels,
                "parameter_minimum": -0.1,
                "parameter_maximum": 0.1,
                "parameter_mean": 0.0,
                "parameter_rms": 0.05,
                "effective_tanh_minimum": -0.0996679946,
                "effective_tanh_maximum": 0.0996679946,
                "effective_tanh_mean": 0.0,
                "effective_tanh_rms": 0.0499,
                "nonzero_count": channels,
                "element_count": channels,
                "dtype": "float32",
                "requires_grad": True,
            }
        )
    return rows


def _mechanism() -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    for index, path in enumerate(subject.BLOCK_PATHS):
        channels = 32 if index < 4 else 64
        blocks.append(
            {
                "block_index_zero_based": index,
                "block_path": path,
                "embedding": "embeddings_1" if index < 4 else "embeddings_2",
                "embedding_block_index_zero_based": index if index < 4 else index - 4,
                "channels": channels,
                "activation": "ReLU",
                "forward_call_count": 2,
                "observed_output_shapes_chw": [[channels, 16, 16]],
                "rms_statistics": {
                    term: {"element_count": 4, "square_sum": 1.0, "rms": 0.5}
                    for term in subject.REQUIRED_MPRS_TERMS
                },
                "target_minus_background_residual_rms": 0.0,
            }
        )
    return {
        "schema": subject.analyzer.MECHANISM_SCHEMA,
        "production_output_policy": "return_original_forward_output_unchanged",
        "diagnostic_execution": (
            "in_production_aligned_mprs_terms_capture_plus_no_grad_headroom"
        ),
        "branch_projection_recomputed_for_statistics": False,
        "feature_cache_written": False,
        "batch_count": 2,
        "aborted_batch_count": 0,
        "block_count": 7,
        "block_order": list(subject.BLOCK_PATHS),
        "temporary_forward_wrappers_restored": True,
        "target_projection": "adaptive_max_pool2d_binary_presence",
        "valid_projection": "adaptive_max_pool2d_any_original_support",
        "background_region": "pooled_valid_and_not_pooled_target",
        "blocks": blocks,
    }


def _mode(mode: str) -> dict[str, object]:
    fixed = _point()
    selected_paths = subject._expected_selected_block_paths(mode)
    selected_indices = [subject.BLOCK_PATHS.index(path) for path in selected_paths]
    selected_vectors = []
    for block_index, path in zip(selected_indices, selected_paths):
        channels = 32 if block_index < 4 else 64
        selected_vectors.append(
            {
                "block_index_zero_based": block_index,
                "block_id": subject.BLOCK_IDS[block_index],
                "block_path": path,
                "element_count": channels,
                "source_nonzero_count": channels,
                "active_nonzero_count": 0,
            }
        )
    source_nonzero = sum(int(row["source_nonzero_count"]) for row in selected_vectors)
    active_scale_sha = SHA_A if mode == "full" else SHA_B
    active_model_sha = SHA_C if mode == "full" else SHA_D
    return {
        "public_mode": mode,
        "knockout_block_paths": selected_paths,
        "knockout_block_indices_zero_based": selected_indices,
        "diagnostic_only": mode != "full",
        "fixed_threshold_0_5": fixed,
        "descriptive_pd_fa": _sweep(fixed),
        "threshold_roles": _threshold_roles(),
        "sweep_thresholds": [0.5, 1.0],
        "saliency_scale_knockout": {
            "schema": subject.analyzer.KNOCKOUT_SCHEMA,
            "public_mode": mode,
            "selected_block_paths": selected_paths,
            "selected_block_indices_zero_based": selected_indices,
            "selected_vectors": selected_vectors,
            "selected_source_nonzero_count": source_nonzero,
            "selected_active_nonzero_count": 0,
            "source_saliency_scale_sha256": SHA_A,
            "active_saliency_scale_sha256": active_scale_sha,
            "restored_saliency_scale_sha256": SHA_A,
            "source_model_state_sha256": SHA_C,
            "active_model_state_sha256": active_model_sha,
            "restored_model_state_sha256": SHA_C,
            "saliency_scale_restored_exactly": True,
            "model_state_restored_exactly": True,
            "derived_checkpoint_written": False,
            "diagnostic_only": mode != "full",
        },
        "probability_difference_to_full": _probability_difference(),
        "full_mprs_statistics": _mechanism() if mode == "full" else None,
        "restoration_audit": {
            "saliency_scale_sha256_expected": SHA_A,
            "saliency_scale_sha256_after_mode": SHA_A,
            "saliency_scale_unchanged": True,
            "model_state_sha256_expected": SHA_C,
            "model_state_sha256_after_mode": SHA_C,
            "model_state_unchanged": True,
        },
    }


def _payload(dataset: str) -> dict[str, object]:
    return {
        "schema": subject.ANALYZER_SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": subject.analyzer.REFERENCE_METHOD,
        "training_model_method": subject.analyzer.TRAINING_MODEL_METHOD,
        "checkpoint_role": "best_miou",
        "seed": 42,
        "test_selected": True,
        "selection_is_optimistic": True,
        "evaluation_protocol": "img_idx_test_selected_development",
        "fixed_threshold": 0.5,
        "sweep_thresholds": [0.5, 1.0],
        "mode_order": list(subject.MODES),
        "block_order": list(subject.BLOCK_PATHS),
        "modes": {mode: _mode(mode) for mode in subject.MODES},
        "source_saliency_scale_records": _source_scale_records(),
        "restoration_audit": {
            "model_state_sha256_before": SHA_C,
            "model_state_sha256_after": SHA_C,
            "model_state_unchanged": True,
            "saliency_scale_sha256_before": SHA_A,
            "saliency_scale_sha256_after": SHA_A,
            "saliency_scale_unchanged": True,
            "temporary_forward_wrappers_restored": True,
        },
        "intervention_contract": {
            "parameter": "selected TPD8 block saliency_scale vector",
            "active_value": "exact_zero",
            "weights_saved_valuewise_and_restored": True,
            "phase_compress_modified": False,
            "other_model_state_modified": False,
            "derived_checkpoint_written": False,
        },
        "reference_replay_audit": {
            "passed": True,
            "comparison": "full_mode_fixed_threshold_0_5_vs_existing_best_miou",
            "compared": {
                "threshold": {"absolute_difference": 0.0, "absolute_tolerance": 0.0}
            },
        },
        "checkpoint_binding": {
            "checkpoint": {"sha256": SHA_C, "role": "best_miou"},
            "protocol": {"payload_sha256": SHA_D},
        },
        "data": {
            "protocol_manifest": {"sha256": SHA_A},
            "inference_order_newline_sha256": SHA_B,
            "split": "img_idx/test",
        },
        "reference_reuse": {"sha256": SHA_C},
        "source_sha256": {
            "analysis/analyze_three_dataset_tpd8_block_residual_knockout_v1.py": SHA_D
        },
        "probability_cache_written": False,
        "derived_checkpoint_written": False,
    }


def _payloads() -> dict[str, dict[str, object]]:
    return {dataset: _payload(dataset) for dataset in subject.DATASETS}


def _bindings() -> dict[str, dict[str, str]]:
    return {
        dataset: {"path": f"/synthetic/{dataset}.json", "sha256": SHA_A}
        for dataset in subject.DATASETS
    }


def _set_point(
    payloads: dict[str, dict[str, object]],
    dataset: str,
    mode: str,
    **updates: object,
) -> None:
    modes = payloads[dataset]["modes"]
    if not isinstance(modes, dict):
        raise TypeError("synthetic modes are malformed")
    mode_payload = modes[mode]
    if not isinstance(mode_payload, dict):
        raise TypeError("synthetic mode is malformed")
    point = mode_payload["fixed_threshold_0_5"]
    if not isinstance(point, dict):
        raise TypeError("synthetic point is malformed")
    aliases = {
        "targets": "matched_target_count",
        "tiny": "matched_tiny_target_count",
        "component_fp": "unmatched_predicted_pixels",
        "background_fp": "false_positive_pixels",
    }
    point.update({aliases.get(key, key): value for key, value in updates.items()})
    point["pd"] = int(point["matched_target_count"]) / int(point["target_count"])
    point["tiny_pd"] = int(point["matched_tiny_target_count"]) / int(
        point["tiny_target_count"]
    )
    point["fa"] = int(point["unmatched_predicted_pixels"]) / int(
        point["valid_pixel_count"]
    )
    descriptive = mode_payload["descriptive_pd_fa"]
    if not isinstance(descriptive, dict) or not isinstance(descriptive.get("points"), list):
        raise TypeError("synthetic descriptive sweep is malformed")
    sweep_fixed = descriptive["points"][0]
    if not isinstance(sweep_fixed, dict):
        raise TypeError("synthetic sweep fixed point is malformed")
    sweep_fixed.clear()
    sweep_fixed.update(
        {
            key: copy.deepcopy(value)
            for key, value in point.items()
            if key != "false_positive_pixels"
        }
    )
    sweep_fixed["selected_point_is_empty"] = (
        int(sweep_fixed["predicted_object_count"]) == 0
    )


def _set_probability(
    payloads: dict[str, dict[str, object]],
    dataset: str,
    mode: str,
    *,
    max_abs: float,
    absolute_sum: float,
    element_count: int = VALID_PIXELS,
) -> None:
    modes = payloads[dataset]["modes"]
    if not isinstance(modes, dict):
        raise TypeError("synthetic modes are malformed")
    mode_payload = modes[mode]
    if not isinstance(mode_payload, dict):
        raise TypeError("synthetic mode is malformed")
    mode_payload["probability_difference_to_full"] = _probability_difference(
        max_abs=max_abs,
        absolute_sum=absolute_sum,
        element_count=element_count,
    )


def _normalized_point(**updates: object) -> dict[str, object]:
    point: dict[str, object] = {
        "matched_target_count": 100,
        "matched_tiny_target_count": 20,
        "miou": 0.8,
        "niou": 0.75,
        "component_false_positive_pixels": 100,
        "background_false_positive_pixels": 200,
    }
    point.update(updates)
    return point


class DirectionAndBoundaryTests(unittest.TestCase):
    def test_zero_denominator_introduction_fails_and_is_severe(self) -> None:
        reference = _normalized_point(
            component_false_positive_pixels=0,
            background_false_positive_pixels=0,
        )
        equal = subject.compare_direction(copy.deepcopy(reference), reference)
        self.assertEqual(equal["component_fp_reduction"]["value"], 0.0)
        self.assertTrue(equal["component_fp_reduction"]["safety_pass"])
        self.assertFalse(equal["severe_degradation"])

        introduced = _normalized_point(
            component_false_positive_pixels=1,
            background_false_positive_pixels=1,
        )
        result = subject.compare_direction(introduced, reference)
        self.assertIsNone(result["component_fp_reduction"]["value"])
        self.assertFalse(result["safe"])
        self.assertTrue(result["severe_degradation"])
        json.dumps(result, allow_nan=False)

    def test_strict_safety_inclusive_material_and_severe_boundaries(self) -> None:
        count_drop = subject.compare_direction(
            _normalized_point(matched_target_count=98), _normalized_point()
        )
        self.assertFalse(count_drop["safe"])
        self.assertTrue(count_drop["severe_degradation"])

        iou_drop = subject.compare_direction(
            _normalized_point(miou=0.795), _normalized_point()
        )
        self.assertFalse(iou_drop["safe"])
        self.assertFalse(iou_drop["severe_degradation"])

        exact_material = subject.compare_direction(
            _normalized_point(component_false_positive_pixels=95),
            _normalized_point(),
        )
        self.assertTrue(exact_material["safe_material_improvement"])

        exact_severe = subject.compare_direction(
            _normalized_point(component_false_positive_pixels=125),
            _normalized_point(),
        )
        self.assertTrue(exact_severe["severe_degradation"])

    def test_reverse_uses_all7_off_as_reference_denominator(self) -> None:
        full = _normalized_point(component_false_positive_pixels=90)
        all_off = _normalized_point(component_false_positive_pixels=100)
        result = subject.compare_direction(full, all_off)
        self.assertAlmostEqual(result["component_fp_reduction"]["value"], 0.10)


class ContractTests(unittest.TestCase):
    def test_sorted_json_round_trip_preserves_declared_mode_order_contract(self) -> None:
        payload = _payload(subject.DATASETS[0])
        round_tripped = json.loads(json.dumps(payload, sort_keys=True))
        self.assertNotEqual(tuple(round_tripped["modes"]), subject.MODES)
        normalized = subject.validate_analyzer_payload(round_tripped)
        self.assertEqual(normalized["dataset"], subject.DATASETS[0])

    def test_fixed_sweep_empty_flag_is_recomputed_when_absent_and_checked_when_present(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        points = modes["full"]["descriptive_pd_fa"]["points"]
        del points[0]["selected_point_is_empty"]
        subject.validate_analyzer_payload(payload)

        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        modes["full"]["descriptive_pd_fa"]["points"][0][
            "selected_point_is_empty"
        ] = True
        with self.assertRaisesRegex(ValueError, "threshold-0.5 empty-point flag differs"):
            subject.validate_analyzer_payload(payload)

    def test_component_fa_uses_unmatched_pixels_not_object_count(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        point = modes["e1b0_off"]["fixed_threshold_0_5"]
        if not isinstance(point, dict):
            self.fail("synthetic point malformed")
        point["unmatched_predicted_object_count"] = 999999
        descriptive = modes["e1b0_off"]["descriptive_pd_fa"]
        if not isinstance(descriptive, dict) or not isinstance(descriptive.get("points"), list):
            self.fail("synthetic descriptive sweep malformed")
        descriptive["points"][0]["unmatched_predicted_object_count"] = 999999
        normalized = subject.validate_analyzer_payload(payload)
        observed = normalized["modes"]["e1b0_off"]["fixed_threshold_0_5"]
        self.assertEqual(observed["component_false_positive_pixels"], 100)

    def test_probability_count_and_sum_mean_identity_are_strict(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        difference = modes["all7_off"]["probability_difference_to_full"]
        if not isinstance(difference, dict):
            self.fail("synthetic difference malformed")
        difference["element_count"] = VALID_PIXELS + 1
        with self.assertRaisesRegex(ValueError, "difference count differs from valid pixels"):
            subject.validate_analyzer_payload(payload)

        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        difference = modes["all7_off"]["probability_difference_to_full"]
        if not isinstance(difference, dict):
            self.fail("synthetic difference malformed")
        difference["absolute_difference_sum"] = 1.0
        difference["mean_abs"] = 0.0
        with self.assertRaisesRegex(ValueError, "difference mean identity differs"):
            subject.validate_analyzer_payload(payload)

    def test_exact_functional_boundaries_are_equivalent(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        modes["all7_off"]["probability_difference_to_full"] = _probability_difference(
            max_abs=1e-7,
            absolute_sum=1e-8 * VALID_PIXELS,
        )
        normalized = subject.validate_analyzer_payload(payload)
        difference = normalized["modes"]["all7_off"]["probability_difference_to_full"]
        self.assertTrue(difference["equivalent"])

    def test_rejects_wrong_mode_selection_and_optional_tenth_mode(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        knockout = modes["e1b0_off"]["saliency_scale_knockout"]
        if not isinstance(knockout, dict):
            self.fail("synthetic knockout malformed")
        knockout["selected_block_paths"] = [subject.BLOCK_PATHS[1]]
        with self.assertRaisesRegex(ValueError, "selected paths differ"):
            subject.validate_analyzer_payload(payload)

        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        if not isinstance(modes, dict):
            self.fail("synthetic modes malformed")
        modes["early_1"] = copy.deepcopy(modes["full"])
        with self.assertRaisesRegex(ValueError, "mode order differs"):
            subject.validate_analyzer_payload(payload)

    def test_rejects_wrong_provenance_and_restoration(self) -> None:
        for field, value in (
            ("schema", "wrong"),
            ("seed", 7),
            ("checkpoint_role", "best_pd"),
            ("dataset", "SIRST3"),
        ):
            with self.subTest(field=field):
                payload = _payload(subject.DATASETS[0])
                payload[field] = value
                with self.assertRaises(ValueError):
                    subject.validate_analyzer_payload(payload)
        payload = _payload(subject.DATASETS[0])
        restoration = payload["restoration_audit"]
        if not isinstance(restoration, dict):
            self.fail("synthetic restoration malformed")
        restoration["saliency_scale_unchanged"] = False
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(payload)


class DecisionTests(unittest.TestCase):
    def _compare(self, payloads: dict[str, dict[str, object]]) -> dict[str, object]:
        return subject.compare_payloads(payloads, input_bindings=_bindings())

    def test_single_harmful_block_is_measured_candidate_and_authorizes(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "e1b3_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_BLOCK_SELECTIVE)
        self.assertEqual(result["persistent_harmful_block_ids"], ["E1.B3"])
        self.assertTrue(result["tpd_local_candidate_training_authorized"])
        suggestion = result["local_candidate_suggestion"]
        self.assertEqual(suggestion["suggestion_type"], "single_block_residual_off")
        self.assertTrue(suggestion["development_training_authorized"])
        self.assertTrue(suggestion["fixed_weight_combination_gate_evaluated"])

    def test_exact_harmful_suffix_suggests_one_early_only_mode(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            for mode in ("e1b2_off", "e1b3_off", "e2b2_off"):
                _set_point(payloads, dataset, mode, component_fp=90)
        result = self._compare(payloads)
        suggestion = result["local_candidate_suggestion"]
        self.assertEqual(result["decision"], subject.DECISION_LOCAL_AUDIT)
        self.assertEqual(suggestion["suggestion_type"], "early_only_depth_mask")
        self.assertEqual(suggestion["suggested_mode_name"], "early_1")
        self.assertTrue(suggestion["requires_new_tenth_mode"])
        self.assertFalse(suggestion["fixed_weight_combination_gate_evaluated"])

    def test_non_suffix_harmful_blocks_suggest_one_union(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            for mode in ("e1b0_off", "e2b2_off"):
                _set_point(payloads, dataset, mode, component_fp=90)
        result = self._compare(payloads)
        suggestion = result["local_candidate_suggestion"]
        self.assertEqual(suggestion["suggestion_type"], "harmful_block_union_off")
        self.assertEqual(suggestion["selected_off_block_ids"], ["E1.B0", "E2.B2"])

    def test_all_seven_harmful_reuses_all7_off_instead_of_tenth_mode(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            for mode in subject.SINGLE_MODES:
                _set_point(payloads, dataset, mode, component_fp=90)
            _set_point(payloads, dataset, "all7_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_RESIDUAL_OFF)
        self.assertFalse(result["local_candidate_suggestion"]["triggered"])
        self.assertTrue(
            result["local_candidate_suggestion"]["resolved_by_required_all7_off"]
        )
        self.assertTrue(
            result["local_candidate_suggestion"][
                "fixed_weight_combination_gate_evaluated"
            ]
        )
        self.assertEqual(
            result["local_candidate_suggestion"]["matching_required_mode_if_any"],
            "all7_off",
        )
        self.assertFalse(
            result["local_candidate_suggestion"]["requires_new_tenth_mode"]
        )

        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            for mode in subject.SINGLE_MODES:
                _set_point(payloads, dataset, mode, component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_INCONCLUSIVE)
        self.assertFalse(result["local_candidate_suggestion"]["triggered"])

    def test_different_blocks_on_different_datasets_do_not_combine(self) -> None:
        payloads = _payloads()
        _set_point(payloads, subject.DATASETS[0], "e1b0_off", component_fp=90)
        _set_point(payloads, subject.DATASETS[1], "e2b0_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["persistent_harmful_block_ids"], [])
        self.assertEqual(result["decision"], subject.DECISION_UNSUPPORTED)

    def test_third_dataset_severe_vetoes_single_block(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "e1b1_off", component_fp=90)
        _set_point(payloads, subject.DATASETS[2], "e1b1_off", targets=98)
        result = self._compare(payloads)
        self.assertNotIn("E1.B1", result["persistent_harmful_block_ids"])

    def test_local_signal_has_priority_over_all7_off_improvement(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "e2b1_off", component_fp=90)
            _set_point(payloads, dataset, "all7_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_BLOCK_SELECTIVE)
        self.assertFalse(result["tpd_residual_off_candidate_authorized"])

    def test_all7_off_improvement_is_second_priority(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "all7_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_RESIDUAL_OFF)
        self.assertTrue(result["tpd_residual_off_candidate_authorized"])
        self.assertFalse(result["tpd_full_architecture_contribution_supported"])

    def test_full_improvement_over_all7_off_freezes_residual(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "all7_off", component_fp=110)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_KEEP)
        self.assertTrue(result["tpd_residual_performance_contribution_supported"])

    def test_unsupported_requires_three_of_three_equivalence(self) -> None:
        result = self._compare(_payloads())
        self.assertEqual(result["decision"], subject.DECISION_UNSUPPORTED)
        self.assertTrue(result["tpd_residual_functional_contribution_unsupported"])

        payloads = _payloads()
        _set_probability(
            payloads,
            subject.DATASETS[2],
            "all7_off",
            max_abs=1.000001e-7,
            absolute_sum=0.0,
        )
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_INCONCLUSIVE)

    def test_one_dataset_full_safe_material_gain_blocks_unsupported(self) -> None:
        payloads = _payloads()
        _set_point(payloads, subject.DATASETS[0], "all7_off", component_fp=110)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_INCONCLUSIVE)
        self.assertTrue(
            result["aggregates"]["any_full_vs_all7_off_safe_material_improvement"]
        )

    def test_all7_off_cross_dataset_gain_blocks_unsupported_flag(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "all7_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_RESIDUAL_OFF)
        self.assertFalse(
            result["tpd_residual_functional_contribution_unsupported"]
        )


class InputAndOutputTests(unittest.TestCase):
    def test_default_inputs_are_exact_analyzer_outputs(self) -> None:
        for dataset in subject.DATASETS:
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    subject._default_input(subject.DEFAULT_INPUT_ROOT, dataset),
                    subject.analyzer._default_output(dataset),
                )

    def test_decision_priority_has_five_frozen_tiers(self) -> None:
        result = subject.compare_payloads(_payloads(), input_bindings=_bindings())
        self.assertEqual(
            [tier["priority"] for tier in result["decision_priority"]],
            [1, 2, 3, 4, 5],
        )

    def test_input_bindings_are_exact(self) -> None:
        with self.assertRaises(ValueError):
            subject.compare_payloads(_payloads(), input_bindings={})
        bindings = _bindings()
        bindings[subject.DATASETS[0]]["sha256"] = "not-a-sha"
        with self.assertRaises(ValueError):
            subject.compare_payloads(_payloads(), input_bindings=bindings)

    def test_outputs_are_atomic_and_write_once(self) -> None:
        result = subject.compare_payloads(_payloads(), input_bindings=_bindings())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "decision.json"
            markdown_path = root / "decision.md"
            subject.write_outputs(json_path, markdown_path, result)
            first_json = json_path.read_bytes()
            first_markdown = markdown_path.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.write_outputs(json_path, markdown_path, result)
            self.assertEqual(json_path.read_bytes(), first_json)
            self.assertEqual(markdown_path.read_bytes(), first_markdown)
            loaded = json.loads(first_json)
            self.assertEqual(loaded["decision"], subject.DECISION_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
