#!/usr/bin/env python3
"""CPU-only tests for the frozen three-dataset QFG comparator."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import compare_three_dataset_qfg_level_knockout_v1 as subject


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


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
        "matched_target_count": targets,
        "matched_tiny_target_count": tiny,
        "miou": miou,
        "niou": niou,
        "unmatched_predicted_pixels": component_fp,
        "unmatched_predicted_object_count": 10,
        "false_positive_pixels": background_fp,
        "false_objects_per_image": 0.1,
        "valid_pixel_count": 1234,
    }


def _probability_difference(
    max_abs: float = 0.0,
    mean_abs: float = 0.0,
) -> dict[str, object]:
    different = bool(
        max_abs > subject.PROBABILITY_MAX_ABS_FUNCTIONAL_THRESHOLD
        or mean_abs > subject.PROBABILITY_MEAN_ABS_FUNCTIONAL_THRESHOLD
    )
    element_count = 1234
    absolute_sum = mean_abs * element_count
    return {
        "scope": "all_original_unpadded_test_pixels",
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "absolute_difference_sum": absolute_sum,
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


def _selected(mode: str) -> list[int]:
    if mode == "full":
        return []
    if mode == "all_off":
        return [0, 1, 2, 3]
    return [subject.MODE_TO_ZERO_BASED_LEVEL[mode]]


def _mode(mode: str) -> dict[str, object]:
    source_sha = SHA_A
    return {
        "public_mode": mode,
        "fixed_threshold_0_5": _point(),
        "descriptive_pd_fa": {"used_for_decision": False},
        "query_perturbation": {"rms": 0.0},
        "factor_summary": {"rms_factor_minus_one": 0.0},
        "spatial_gate_factor_statistics": {"gate_rms": 0.0},
        "alpha_knockout": {
            "selected_level_indices_zero_based": _selected(mode),
            "source_alpha_sha256": source_sha,
            "active_alpha_sha256": source_sha if mode == "full" else SHA_B,
        },
        "probability_difference_to_full": _probability_difference(),
        "restoration_audit": {"alpha_state_unchanged": True},
    }


def _payload(dataset: str) -> dict[str, object]:
    return {
        "schema": subject.ANALYZER_SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "checkpoint_role": "best_miou",
        "seed": 42,
        "test_selected": True,
        "selection_is_optimistic": True,
        "evaluation_protocol": "img_idx_test_selected_development",
        "fixed_threshold": 0.5,
        "mode_order": list(subject.MODES),
        "modes": {mode: _mode(mode) for mode in subject.MODES},
        "restoration_audit": {
            "model_state_sha256_before": SHA_A,
            "model_state_sha256_after": SHA_A,
            "model_state_unchanged": True,
            "alpha_state_sha256_before": SHA_A,
            "alpha_state_sha256_after": SHA_A,
            "alpha_state_unchanged": True,
        },
        "reference_replay_audit": {"passed": True},
        "checkpoint_binding": {
            "checkpoint": {"sha256": SHA_C, "role": "best_miou"},
            "protocol": {"payload_sha256": SHA_D},
        },
        "data": {
            "protocol_manifest": {"sha256": SHA_A},
            "inference_order_newline_sha256": SHA_B,
        },
        "reference_reuse": {"sha256": SHA_C},
        "source_sha256": {
            "analysis/analyze_three_dataset_qfg_level_knockout_v1.py": SHA_D
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
    assert isinstance(modes, dict)
    mode_payload = modes[mode]
    assert isinstance(mode_payload, dict)
    point = mode_payload["fixed_threshold_0_5"]
    assert isinstance(point, dict)
    aliases = {
        "targets": "matched_target_count",
        "tiny": "matched_tiny_target_count",
        "component_fp": "unmatched_predicted_pixels",
        "background_fp": "false_positive_pixels",
    }
    point.update({aliases.get(key, key): value for key, value in updates.items()})


def _set_probability(
    payloads: dict[str, dict[str, object]],
    dataset: str,
    mode: str,
    *,
    max_abs: float,
    mean_abs: float,
) -> None:
    modes = payloads[dataset]["modes"]
    assert isinstance(modes, dict)
    mode_payload = modes[mode]
    assert isinstance(mode_payload, dict)
    mode_payload["probability_difference_to_full"] = _probability_difference(
        max_abs=max_abs, mean_abs=mean_abs
    )


def _normalized_point(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "matched_target_count": 100,
        "matched_tiny_target_count": 20,
        "miou": 0.8,
        "niou": 0.75,
        "component_false_positive_pixels": 100,
        "background_false_positive_pixels": 200,
    }
    value.update(updates)
    return value


class DirectionalBoundaryTests(unittest.TestCase):
    def test_zero_denominator_rules_are_finite_and_veto_introduction(self) -> None:
        reference = _normalized_point(
            component_false_positive_pixels=0,
            background_false_positive_pixels=0,
        )
        zero = subject.compare_direction(copy.deepcopy(reference), reference)
        self.assertEqual(zero["component_fp_reduction"]["value"], 0.0)
        self.assertTrue(zero["component_fp_reduction"]["safety_pass"])
        self.assertFalse(zero["component_fp_reduction"]["severe_degradation"])

        introduced = _normalized_point(
            component_false_positive_pixels=1,
            background_false_positive_pixels=1,
        )
        result = subject.compare_direction(introduced, reference)
        self.assertIsNone(result["component_fp_reduction"]["value"])
        self.assertIsNone(result["background_pixel_fp_reduction"]["value"])
        self.assertFalse(result["safe"])
        self.assertTrue(result["severe_degradation"])
        json.dumps(result, allow_nan=False)

    def test_strict_safety_and_inclusive_material_boundaries(self) -> None:
        reference = _normalized_point(miou=0.005, niou=0.005)
        exact_count_drop = subject.compare_direction(
            _normalized_point(
                matched_target_count=98,
                miou=0.005,
                niou=0.005,
            ),
            reference,
        )
        self.assertFalse(exact_count_drop["safe"])
        self.assertTrue(exact_count_drop["severe_degradation"])

        exact_iou_drop = subject.compare_direction(
            _normalized_point(miou=0.0, niou=0.005), reference
        )
        self.assertFalse(exact_iou_drop["safe"])
        self.assertFalse(exact_iou_drop["severe_degradation"])

        exact_fp_safety = subject.compare_direction(
            _normalized_point(component_false_positive_pixels=105),
            _normalized_point(),
        )
        self.assertFalse(exact_fp_safety["safe"])
        self.assertFalse(exact_fp_safety["severe_degradation"])

        exact_fp_severe = subject.compare_direction(
            _normalized_point(component_false_positive_pixels=125),
            _normalized_point(),
        )
        self.assertTrue(exact_fp_severe["severe_degradation"])

        exact_material = subject.compare_direction(
            _normalized_point(component_false_positive_pixels=95),
            _normalized_point(),
        )
        self.assertTrue(exact_material["safe_material_improvement"])

    def test_reverse_uses_all_off_as_denominator(self) -> None:
        full = _normalized_point(component_false_positive_pixels=90)
        all_off = _normalized_point(component_false_positive_pixels=100)
        reverse = subject.compare_direction(full, all_off)
        self.assertAlmostEqual(reverse["component_fp_reduction"]["value"], 0.10)


class DecisionTests(unittest.TestCase):
    def _compare(self, payloads: dict[str, dict[str, object]]) -> dict[str, object]:
        return subject.compare_payloads(payloads, input_bindings=_bindings())

    def test_same_level_two_of_three_is_persistent_harmful(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "level1_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_REMOVE_LEVELS)
        self.assertEqual(result["persistent_harmful_zero_based_levels"], [1])

    def test_different_levels_do_not_combine_across_datasets(self) -> None:
        payloads = _payloads()
        _set_point(payloads, subject.DATASETS[0], "level0_off", component_fp=90)
        _set_point(payloads, subject.DATASETS[1], "level1_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["persistent_harmful_zero_based_levels"], [])
        self.assertEqual(result["decision"], subject.DECISION_UNSUPPORTED)

    def test_third_dataset_severe_vetoes_persistent_level(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "level2_off", component_fp=90)
        _set_point(payloads, subject.DATASETS[2], "level2_off", targets=98)
        result = self._compare(payloads)
        self.assertEqual(result["persistent_harmful_zero_based_levels"], [])

    def test_all_off_safe_improvement_has_second_priority(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "all_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_ALL_OFF)
        self.assertTrue(result["qfg_off_candidate_authorized"])

    def test_full_safe_improvement_over_all_off_keeps_qfg(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "all_off", component_fp=110)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_KEEP)
        self.assertTrue(result["qfg_performance_contribution_supported"])

    def test_all_off_zero_is_the_reference_for_reverse_comparison(self) -> None:
        payloads = _payloads()
        dataset = subject.DATASETS[0]
        _set_point(payloads, dataset, "full", component_fp=1)
        _set_point(payloads, dataset, "all_off", component_fp=0)
        result = self._compare(payloads)
        reverse = result["per_dataset"][dataset]["modes"]["all_off"]["full_vs_off"]
        self.assertIsNone(reverse["component_fp_reduction"]["value"])
        self.assertFalse(reverse["safe"])
        self.assertTrue(reverse["severe_degradation"])

    def test_remove_level_has_priority_over_all_off(self) -> None:
        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_point(payloads, dataset, "level3_off", component_fp=90)
            _set_point(payloads, dataset, "all_off", component_fp=90)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_REMOVE_LEVELS)

    def test_unsupported_requires_three_of_three_probability_equivalence(self) -> None:
        payloads = _payloads()
        # Equality at the two thresholds is still equivalent: the functional
        # predicate is strict OR, exactly as frozen in the plan.
        _set_probability(
            payloads,
            subject.DATASETS[0],
            "all_off",
            max_abs=1e-7,
            mean_abs=1e-8,
        )
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_UNSUPPORTED)
        self.assertFalse(result["qfg_functional_contribution_supported"])

        payloads = _payloads()
        _set_probability(
            payloads,
            subject.DATASETS[2],
            "all_off",
            max_abs=1.0000001e-7,
            mean_abs=0.0,
        )
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_INCONCLUSIVE)
        self.assertIsNone(result["qfg_functional_contribution_supported"])

        payloads = _payloads()
        for dataset in subject.DATASETS[:2]:
            _set_probability(
                payloads,
                dataset,
                "all_off",
                max_abs=1.0000001e-7,
                mean_abs=0.0,
            )
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_INCONCLUSIVE)
        self.assertTrue(result["qfg_functional_contribution_supported"])

    def test_unsupported_rejected_by_one_dataset_full_safe_material_gain(self) -> None:
        payloads = _payloads()
        _set_point(payloads, subject.DATASETS[0], "all_off", component_fp=110)
        result = self._compare(payloads)
        self.assertEqual(result["decision"], subject.DECISION_INCONCLUSIVE)
        self.assertTrue(
            result["aggregates"]["any_full_vs_all_off_safe_material_improvement"]
        )
        self.assertFalse(result["qfg_functional_contribution_supported"])


class ContractAndIOTests(unittest.TestCase):
    def test_rejects_wrong_provenance_and_state_contracts(self) -> None:
        mutations = (
            ("schema", "wrong"),
            ("seed", 7),
            ("checkpoint_role", "best_pd"),
            ("test_selected", False),
            ("dataset", "SIRST3"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                payload = _payload(subject.DATASETS[0])
                payload[field] = value
                with self.assertRaises(ValueError):
                    subject.validate_analyzer_payload(payload)

        payload = _payload(subject.DATASETS[0])
        audit = payload["restoration_audit"]
        self.assertIsInstance(audit, dict)
        audit["alpha_state_unchanged"] = False
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(payload)

    def test_rejects_wrong_mode_and_probability_contracts(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        self.assertIsInstance(modes, dict)
        modes["level0_off"]["alpha_knockout"]["selected_level_indices_zero_based"] = [1]
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(payload)

    def test_probability_count_must_equal_every_mode_valid_pixel_count(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        self.assertIsInstance(modes, dict)
        modes["level2_off"]["fixed_threshold_0_5"]["valid_pixel_count"] = 1235
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(payload)

        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        self.assertIsInstance(modes, dict)
        difference = modes["all_off"]["probability_difference_to_full"]
        difference["element_count"] = 1235
        difference["mean_abs"] = difference["absolute_difference_sum"] / 1235
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(payload)

    def test_sorted_json_round_trip_uses_canonical_mode_order_field(self) -> None:
        payload = _payload(subject.DATASETS[0])
        round_tripped = json.loads(json.dumps(payload, sort_keys=True))
        self.assertNotEqual(tuple(round_tripped["modes"]), subject.MODES)
        validated = subject.validate_analyzer_payload(round_tripped)
        self.assertEqual(validated["dataset"], subject.DATASETS[0])

        round_tripped["mode_order"] = list(reversed(subject.MODES))
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(round_tripped)

    def test_rejects_probability_summary_inconsistent_with_raw_sum(self) -> None:
        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        self.assertIsInstance(modes, dict)
        difference = modes["all_off"]["probability_difference_to_full"]
        difference["absolute_difference_sum"] = 1e-6
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(payload)

        payload = _payload(subject.DATASETS[0])
        modes = payload["modes"]
        self.assertIsInstance(modes, dict)
        difference = modes["all_off"]["probability_difference_to_full"]
        difference["max_abs"] = 2e-7
        # Leave the analyzer booleans stale; strict validation must reject it.
        with self.assertRaises(ValueError):
            subject.validate_analyzer_payload(payload)

    def test_input_sha_bindings_are_mandatory_and_strict(self) -> None:
        with self.assertRaises(ValueError):
            subject.compare_payloads(_payloads(), input_bindings={})
        bindings = _bindings()
        bindings[subject.DATASETS[1]]["sha256"] = "not-a-sha"
        with self.assertRaises(ValueError):
            subject.compare_payloads(_payloads(), input_bindings=bindings)

    def test_output_pair_is_write_once(self) -> None:
        result = subject.compare_payloads(_payloads(), input_bindings=_bindings())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "decision.json"
            md_path = root / "decision.md"
            subject.write_outputs(json_path, md_path, result)
            first_json = json_path.read_bytes()
            first_md = md_path.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.write_outputs(json_path, md_path, result)
            self.assertEqual(json_path.read_bytes(), first_json)
            self.assertEqual(md_path.read_bytes(), first_md)
            loaded = json.loads(first_json)
            self.assertEqual(loaded["decision"], subject.DECISION_UNSUPPORTED)

    def test_module_tests_run_under_python_optimized_mode(self) -> None:
        if not __debug__:
            self.skipTest("already running with -O")
        completed = subprocess.run(
            [
                sys.executable,
                "-O",
                "-m",
                "unittest",
                "tests.test_compare_three_dataset_qfg_level_knockout_v1",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
