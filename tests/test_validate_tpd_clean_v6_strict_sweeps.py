from __future__ import annotations

import copy
import math

import pytest

from experiments import validate_tpd_clean_v6_strict_sweeps as strict


def _point(threshold: float) -> dict:
    matched = 188 if threshold < 1.0 else 0
    matched_tiny = 39 if threshold < 1.0 else 0
    predicted = matched
    unmatched = predicted - matched
    return {
        "threshold": threshold,
        "target_count": 189,
        "matched_target_count": matched,
        "pd": matched / 189,
        "tiny_target_count": 39,
        "matched_tiny_target_count": matched_tiny,
        "tiny_pd": matched_tiny / 39,
        "predicted_object_count": predicted,
        "unmatched_predicted_object_count": unmatched,
        "false_objects_per_image": unmatched / 133,
        "valid_pixel_count": 8_716_288,
        "fa": 0.0,
        "miou": 0.9 if threshold < 1.0 else 0.0,
        "niou": 0.9 if threshold < 1.0 else 0.0,
        "pixel_precision": 0.9 if threshold < 1.0 else 0.0,
        "pixel_recall": 0.9 if threshold < 1.0 else 0.0,
        "pixel_f1": 0.9 if threshold < 1.0 else 0.0,
        "val_loss": 0.001,
    }


def _payload() -> dict:
    base = strict._expected_base_thresholds()
    tail, tail_range = strict._expected_tail_thresholds()
    quantiles = {
        key: 0.000123 + index * 0.000001
        for index, key in enumerate(strict.EXPECTED_QUANTILE_KEYS)
    }
    thresholds = sorted(
        {
            *base,
            *tail,
            *quantiles.values(),
            strict.LAST_FLOAT32_BELOW_ONE,
            1.0,
        }
    )
    points = [_point(threshold) for threshold in thresholds]
    fixed = next(
        point for point in points if math.isclose(point["threshold"], 0.5)
    )
    return {
        "validation_count": 133,
        "match_radius": 3.0,
        "tiny_area": 9,
        "threshold_configuration": copy.deepcopy(
            strict.EXPECTED_THRESHOLD_CONFIGURATION
        ),
        "threshold_provenance": {
            "uniform_probability_grid_count": len(base),
            "tail_logit_range": tail_range,
            "tail_logit_step": 0.1,
            "tail_logit_threshold_count": len(tail),
            "empirical_score_quantiles": quantiles,
            "total_unique_threshold_count": len(points),
            "score_count": 8_716_288,
            "exact_one_score_count": 0,
            "added_thresholds": [
                strict.LAST_FLOAT32_BELOW_ONE,
                1.0,
            ],
        },
        "points": points,
        "fixed_threshold_0_5": fixed,
        "best_points_under_fa_budget": {
            f"{budget:.10g}": fixed
            for budget in strict.EXPECTED_THRESHOLD_CONFIGURATION["fa_budgets"]
        },
        "audit": {
            "invocation_argv": ["python", "evaluator.py"],
            "parsed_arguments": {
                **copy.deepcopy(strict.EXPECTED_THRESHOLD_CONFIGURATION),
                "expected_epochs": 800,
                "overwrite": False,
            },
        },
    }


def test_strict_payload_accepts_complete_contract() -> None:
    strict.validate_sweep_payload(_payload(), "synthetic")


def test_strict_payload_rejects_wrong_threshold_configuration() -> None:
    payload = _payload()
    payload["threshold_configuration"]["threshold_step"] = 0.02
    with pytest.raises(strict.StrictSweepValidationError, match="threshold_step"):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_missing_required_grid_point() -> None:
    payload = _payload()
    payload["points"] = [
        point for point in payload["points"] if point["threshold"] != 0.25
    ]
    payload["threshold_provenance"]["total_unique_threshold_count"] = len(
        payload["points"]
    )
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="threshold sequence differs from the registered union",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_tiny_pd_count_mismatch() -> None:
    payload = _payload()
    payload["points"][0]["tiny_pd"] = 0.0
    with pytest.raises(
        strict.StrictSweepValidationError, match="tiny-Pd/count identity"
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_matched_tiny_above_all_matches() -> None:
    payload = _payload()
    payload["points"][0]["matched_target_count"] = 1
    payload["points"][0]["pd"] = 1 / 189
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="matched tiny targets exceed all matched targets",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_point_count_provenance_mismatch() -> None:
    payload = _payload()
    payload["threshold_provenance"]["total_unique_threshold_count"] += 1
    with pytest.raises(
        strict.StrictSweepValidationError, match="total threshold count differs"
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_extra_threshold() -> None:
    payload = _payload()
    payload["points"].append(_point(0.123456789))
    payload["points"].sort(key=lambda point: point["threshold"])
    payload["threshold_provenance"]["total_unique_threshold_count"] = len(
        payload["points"]
    )
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="threshold sequence differs from the registered union",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_impossible_non_tiny_count() -> None:
    payload = _payload()
    point = payload["points"][0]
    point["matched_target_count"] = 151
    point["pd"] = 151 / 189
    point["matched_tiny_target_count"] = 0
    point["tiny_pd"] = 0.0
    point["predicted_object_count"] = 153
    point["unmatched_predicted_object_count"] = 2
    point["false_objects_per_image"] = 2 / 133
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="matched non-tiny targets exceed available non-tiny targets",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_empty_budget_mapping() -> None:
    payload = _payload()
    payload["best_points_under_fa_budget"] = {}
    with pytest.raises(
        strict.StrictSweepValidationError, match="budget point key set differs"
    ):
        strict.validate_sweep_payload(payload, "synthetic")


@pytest.mark.parametrize("key", ["fa", "miou"])
def test_strict_payload_rejects_nonfinite_point_metric(key: str) -> None:
    payload = _payload()
    payload["points"][0][key] = math.nan
    with pytest.raises(
        strict.StrictSweepValidationError, match=f"{key} must be finite"
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_wrong_fixed_threshold() -> None:
    payload = _payload()
    payload["fixed_threshold_0_5"] = _point(0.123456)
    with pytest.raises(
        strict.StrictSweepValidationError, match="fixed threshold is not 0.5"
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_fa_off_pixel_lattice() -> None:
    payload = _payload()
    payload["points"][0]["fa"] = 0.5 / 8_716_288
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="Fa is not on the valid-pixel lattice",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_unmatched_object_with_zero_fa() -> None:
    payload = _payload()
    point = payload["points"][0]
    point["predicted_object_count"] += 1
    point["unmatched_predicted_object_count"] = 1
    point["false_objects_per_image"] = 1 / 133
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="unmatched-object count exceeds unmatched pixels",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_nonmonotone_quantiles() -> None:
    payload = _payload()
    quantiles = payload["threshold_provenance"][
        "empirical_score_quantiles"
    ]
    second_value = quantiles["0.95"]
    quantiles["0.95"] = quantiles["0.9"] / 2
    payload["points"] = [
        point for point in payload["points"] if point["threshold"] != second_value
    ]
    payload["points"].append(_point(quantiles["0.95"]))
    payload["points"].sort(key=lambda point: point["threshold"])
    payload["threshold_provenance"]["total_unique_threshold_count"] = len(
        payload["points"]
    )
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="empirical quantiles are not monotone",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_missing_internal_quantile() -> None:
    payload = _payload()
    removed = payload["threshold_provenance"][
        "empirical_score_quantiles"
    ].pop("0.95")
    payload["points"] = [
        point for point in payload["points"] if point["threshold"] != removed
    ]
    payload["threshold_provenance"]["total_unique_threshold_count"] = len(
        payload["points"]
    )
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="quantile keys are not a contiguous registered slice",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_unjustified_quantile_suffix_omission() -> None:
    payload = _payload()
    removed = payload["threshold_provenance"][
        "empirical_score_quantiles"
    ].pop("0.999999")
    payload["points"] = [
        point for point in payload["points"] if point["threshold"] != removed
    ]
    payload["threshold_provenance"]["total_unique_threshold_count"] = len(
        payload["points"]
    )
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="quantile suffix omitted without exact-one scores",
    ):
        strict.validate_sweep_payload(payload, "synthetic")


def test_strict_payload_rejects_retained_unit_quantiles() -> None:
    payload = _payload()
    payload["threshold_provenance"][
        "exact_one_score_count"
    ] = 8_716_288
    with pytest.raises(
        strict.StrictSweepValidationError,
        match="unit quantile was incorrectly retained",
    ):
        strict.validate_sweep_payload(payload, "synthetic")
