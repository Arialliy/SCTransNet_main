from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from analysis import compare_three_dataset_ner_l4_tpr_v1 as subject


SHA_A = "a" * 64
SHA_B = "b" * 64
VALID_PIXELS = 1000


def _point(
    *,
    matched: int,
    matched_tiny: int,
    component_fp: int,
    background_fp: int,
    miou: float = 0.80,
    niou: float = 0.75,
    precision: float = 0.82,
    recall: float = 0.78,
    f1: float = 0.80,
) -> dict[str, object]:
    target_count = 110
    tiny_count = 25
    return {
        "threshold": 0.5,
        "target_count": target_count,
        "tiny_target_count": tiny_count,
        "matched_target_count": matched,
        "matched_tiny_target_count": matched_tiny,
        "pd": matched / target_count,
        "tiny_pd": matched_tiny / tiny_count,
        "unmatched_predicted_pixels": component_fp,
        "component_false_positive_pixels": component_fp,
        "false_positive_pixels": background_fp,
        "background_false_positive_pixels": background_fp,
        "fa": component_fp / VALID_PIXELS,
        "miou": miou,
        "niou": niou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "valid_pixel_count": VALID_PIXELS,
    }


def _mode(mode: str) -> dict[str, object]:
    binding = subject.analyzer.normalize_public_mode(mode)
    if mode == subject.CURRENT_MODE:
        point = _point(
            matched=100,
            matched_tiny=20,
            component_fp=100,
            background_fp=200,
        )
    else:
        point = _point(
            matched=94,
            matched_tiny=18,
            component_fp=70,
            background_fp=150,
            miou=0.79,
            niou=0.74,
            precision=0.84,
            recall=0.74,
            f1=0.785,
        )
    return {**binding, "fixed_threshold_0_5": point}


def _payload(dataset: str, role: str) -> dict[str, object]:
    return {
        "schema": subject.ANALYZER_SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "checkpoint_role": role,
        "seed": subject.SEED,
        "test_selected": True,
        "mode_order": list(subject.MODES),
        "modes": {mode: _mode(mode) for mode in subject.MODES},
        "checkpoint_binding": {
            "sha256": SHA_B,
            "role": role,
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


def _replace_point(
    payloads: dict[str, dict[str, object]],
    dataset: str,
    role: str,
    mode: str,
    **updates: object,
) -> None:
    payload = payloads[subject._binding_key(dataset, role)]
    modes = payload["modes"]
    if not isinstance(modes, dict):
        raise TypeError("synthetic modes differ")
    raw_mode = modes[mode]
    if not isinstance(raw_mode, dict):
        raise TypeError("synthetic mode differs")
    point = raw_mode["fixed_threshold_0_5"]
    if not isinstance(point, dict):
        raise TypeError("synthetic point differs")
    aliases = {
        "target": "matched_target_count",
        "tiny": "matched_tiny_target_count",
        "component_fp": "component_false_positive_pixels",
        "background_fp": "background_false_positive_pixels",
    }
    for name, value in updates.items():
        point[aliases.get(name, name)] = value
    if "component_fp" in updates:
        point["unmatched_predicted_pixels"] = updates["component_fp"]
    if "background_fp" in updates:
        point["false_positive_pixels"] = updates["background_fp"]
    point["pd"] = int(point["matched_target_count"]) / int(point["target_count"])
    point["tiny_pd"] = int(point["matched_tiny_target_count"]) / int(
        point["tiny_target_count"]
    )
    point["fa"] = int(point["component_false_positive_pixels"]) / int(
        point["valid_pixel_count"]
    )


class NERL4TPRComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        """Unit-test comparator math without running the formal analyzer."""

        patcher = mock.patch.object(
            subject.analyzer,
            "validate_output_payload",
            side_effect=lambda payload: None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_neutral_protected_modes_have_no_joint_recovery_signal(self) -> None:
        result = subject.compare_payloads(_payloads(), input_bindings=_bindings())
        subject.validate_comparison_payload(result)
        self.assertEqual(result["assessment"], subject.ASSESSMENT_NONE)
        self.assertFalse(
            result["joint_directional_assessment"][
                "uses_single_metric_hard_threshold"
            ]
        )
        self.assertFalse(
            result["joint_directional_assessment"]["uses_weighted_metric_sum"]
        )
        self.assertFalse(
            result["joint_directional_assessment"]["training_authorization_made"]
        )

    def test_same_representable_mode_has_cross_role_joint_signal(self) -> None:
        payloads = _payloads()
        mode = "tpr_g0125"
        _replace_point(
            payloads,
            subject.DATASETS[0],
            "best_miou",
            mode,
            target=98,
            tiny=19,
            component_fp=80,
            background_fp=160,
            miou=0.805,
            niou=0.755,
            pixel_precision=0.835,
            pixel_recall=0.77,
            pixel_f1=0.801,
        )
        _replace_point(
            payloads,
            subject.DATASETS[1],
            "best_pd",
            mode,
            target=97,
            tiny=19,
            component_fp=85,
            background_fp=170,
        )
        result = subject.compare_payloads(payloads, input_bindings=_bindings())
        self.assertEqual(result["assessment"], subject.ASSESSMENT_CROSS_ROLE)
        self.assertEqual(
            result["joint_directional_assessment"][
                "representable_cross_role_joint_modes"
            ],
            [mode],
        )
        row = result["per_mode"][mode]
        self.assertEqual(row["joint_signal_unit_count"], 2)
        self.assertEqual(
            row["sum_counts"]["matched_target_count"][
                "candidate_minus_unprotected"
            ],
            7,
        )
        self.assertEqual(set(row["macro_metrics"]), set(subject.SCALAR_METRICS))

    def test_target_and_fp_signals_in_different_units_do_not_combine(self) -> None:
        payloads = _payloads()
        mode = "tpr_g01875"
        _replace_point(
            payloads,
            subject.DATASETS[0],
            "best_miou",
            mode,
            target=99,
            tiny=20,
            component_fp=110,
            background_fp=210,
        )
        _replace_point(
            payloads,
            subject.DATASETS[1],
            "best_pd",
            mode,
            target=94,
            tiny=18,
            component_fp=60,
            background_fp=140,
        )
        result = subject.compare_payloads(payloads, input_bindings=_bindings())
        self.assertEqual(result["assessment"], subject.ASSESSMENT_NONE)
        self.assertEqual(
            result["per_mode"][mode]["target_recovery_unit_count"], 1
        )
        self.assertEqual(
            result["per_mode"][mode]["both_fp_decrease_unit_count"], 5
        )
        self.assertEqual(result["per_mode"][mode]["joint_signal_unit_count"], 0)

    def test_boundary_limit_signal_is_not_a_finite_logit_signal(self) -> None:
        payloads = _payloads()
        mode = subject.BOUNDARY_LIMIT_MODE
        for role, dataset in zip(subject.CHECKPOINT_ROLES, subject.DATASETS):
            _replace_point(
                payloads,
                dataset,
                role,
                mode,
                target=100,
                tiny=20,
                component_fp=60,
                background_fp=140,
            )
        result = subject.compare_payloads(payloads, input_bindings=_bindings())
        self.assertEqual(result["per_mode"][mode]["joint_signal_unit_count"], 2)
        self.assertEqual(result["assessment"], subject.ASSESSMENT_NONE)
        self.assertNotIn(mode, result["finite_logit_representable_tpr_modes"])
        self.assertFalse(
            result["pareto"][
                "boundary_limit_mode_eligible_as_finite_logit_candidate"
            ]
        )

    def test_alias_and_pd_count_value_mismatches_are_rejected(self) -> None:
        payloads = _payloads()
        first = payloads[subject._expected_keys()[0]]
        modes = first["modes"]
        if not isinstance(modes, dict):
            raise TypeError("synthetic modes differ")
        fixed = modes["tpr_g00625"]["fixed_threshold_0_5"]
        if not isinstance(fixed, dict):
            raise TypeError("synthetic point differs")
        fixed["component_false_positive_pixels"] = 71
        with self.assertRaisesRegex(
            subject.NERL4TPRComparisonError, "alias differs"
        ):
            subject.compare_payloads(payloads, input_bindings=_bindings())

        payloads = _payloads()
        first = payloads[subject._expected_keys()[0]]
        modes = first["modes"]
        if not isinstance(modes, dict):
            raise TypeError("synthetic modes differ")
        fixed = modes["tpr_g00625"]["fixed_threshold_0_5"]
        if not isinstance(fixed, dict):
            raise TypeError("synthetic point differs")
        fixed["pd"] = 1.0
        with self.assertRaisesRegex(
            subject.NERL4TPRComparisonError, "matched/total counts"
        ):
            subject.compare_payloads(payloads, input_bindings=_bindings())

    def test_json_roundtrip_markdown_and_write_once(self) -> None:
        result = subject.compare_payloads(_payloads(), input_bindings=_bindings())
        roundtrip = json.loads(json.dumps(result, allow_nan=False))
        subject.validate_comparison_payload(roundtrip)
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            json_path = directory / "comparison.json"
            markdown_path = directory / "comparison.md"
            subject.write_outputs(json_path, markdown_path, roundtrip)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Pd(计数)", markdown)
            self.assertIn("Precision", markdown)
            self.assertIn("有限 logit 可训练点", markdown)
            with self.assertRaises(FileExistsError):
                subject.write_outputs(json_path, markdown_path, roundtrip)

    def test_binding_parser_requires_exactly_six_inputs(self) -> None:
        self.assertEqual(
            set(subject._parse_bindings([])), set(subject._expected_keys())
        )
        with self.assertRaisesRegex(
            subject.NERL4TPRComparisonError, "all six"
        ):
            subject._parse_bindings(
                [f"{subject._expected_keys()[0]}=/tmp/one.json"]
            )

    def test_compare_does_not_mutate_analyzer_payloads(self) -> None:
        payloads = _payloads()
        before = copy.deepcopy(payloads)
        subject.compare_payloads(payloads, input_bindings=_bindings())
        self.assertEqual(payloads, before)


if __name__ == "__main__":
    unittest.main()
