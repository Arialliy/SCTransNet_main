from __future__ import annotations

from copy import deepcopy

import pytest

from experiments import compare_finalize_ec_tss_v3_1 as finalizer


def _raw_point() -> dict[str, int | float]:
    return {
        "miou": 0.5,
        "niou": 0.5,
        "pd": 2 / 3,
        "tiny_pd": 1.0,
        "fa": 0.02,
        "pixel_precision": 0.5,
        "pixel_recall": 0.5,
        "pixel_f1": 0.5,
        "false_objects_per_image": 0.25,
        "target_count": 3,
        "matched_target_count": 2,
        "tiny_target_count": 1,
        "matched_tiny_target_count": 1,
        "predicted_object_count": 3,
        "unmatched_predicted_object_count": 1,
        "unmatched_predicted_pixels": 2,
        "valid_pixel_count": 100,
        "threshold": 0.5,
        "test_loss": 0.1,
    }


def _frozen_point(
    *,
    miou_q: int = 10_000,
    niou_q: int = 10_000,
    matched: int = 10,
    unmatched_pixels: int = 100,
    matched_tiny: int = 5,
) -> dict[str, int]:
    return {
        "miou_q": miou_q,
        "niou_q": niou_q,
        "matched_target_count": matched,
        "unmatched_predicted_pixels": unmatched_pixels,
        "matched_tiny_target_count": matched_tiny,
    }


def _pairwise_records(
    reference: dict[str, int], candidate: dict[str, int]
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    return {
        dataset: {
            role: {
                "original": deepcopy(reference),
                "ec_tss_v3_1": deepcopy(candidate),
            }
            for role in finalizer.ROLES
        }
        for dataset in finalizer.DATASETS
    }


def test_normalize_point_validates_count_identities() -> None:
    normalized = finalizer.normalize_point(_raw_point(), "valid")
    assert normalized["pd"] == pytest.approx(2 / 3)
    assert normalized["fa"] == pytest.approx(0.02)
    assert normalized["object_precision"] == pytest.approx(2 / 3)

    bad_pd = _raw_point()
    bad_pd["pd"] = 0.5
    with pytest.raises(finalizer.ComparisonError, match="Pd/count mismatch"):
        finalizer.normalize_point(bad_pd, "bad-pd")

    bad_object_identity = _raw_point()
    bad_object_identity["predicted_object_count"] = 4
    with pytest.raises(
        finalizer.ComparisonError,
        match="predicted-object identity is inconsistent",
    ):
        finalizer.normalize_point(bad_object_identity, "bad-object-identity")


def test_quantize_uses_frozen_half_up_boundaries() -> None:
    assert finalizer.quantize(0.0) == 0
    assert finalizer.quantize(0.0000499999) == 0
    assert finalizer.quantize(0.00005) == 1
    assert finalizer.quantize(0.0001499999) == 1
    assert finalizer.quantize(0.00015) == 2
    assert finalizer.quantize(-0.00005) == 0
    assert finalizer.quantize(-0.0000500001) == -1


def test_severe_violations_respects_frozen_inclusive_boundaries() -> None:
    original = _frozen_point()
    just_below = _frozen_point(
        miou_q=9_951,
        niou_q=9_951,
        matched=9,
        matched_tiny=4,
        unmatched_pixels=125,
    )
    assert finalizer.severe_violations(
        "NUAA-SIRST", "best_miou", original, just_below
    ) == []

    at_or_over = _frozen_point(
        miou_q=9_950,
        niou_q=9_950,
        matched=8,
        matched_tiny=3,
        unmatched_pixels=126,
    )
    violations = finalizer.severe_violations(
        "NUAA-SIRST", "best_miou", original, at_or_over
    )
    assert {item["rule"] for item in violations} == {
        "matched_target_drop_at_least_2",
        "matched_tiny_target_drop_at_least_2",
        "miou_drop_at_least_0.005",
        "niou_drop_at_least_0.005",
        "fa_increase_over_25_percent_without_2_matched_gain",
    }

    compensated_fa = _frozen_point(matched=12, unmatched_pixels=126)
    assert finalizer.severe_violations(
        "NUAA-SIRST", "best_miou", original, compensated_fa
    ) == []

    zero_fa_original = _frozen_point(unmatched_pixels=0)
    zero_fa_candidate = _frozen_point(unmatched_pixels=1)
    assert {
        item["rule"]
        for item in finalizer.severe_violations(
            "NUDT-SIRST",
            "best_pd",
            zero_fa_original,
            zero_fa_candidate,
        )
    } == {"fa_increase_over_25_percent_without_2_matched_gain"}


def test_pairwise_and_pareto_follow_metric_directions() -> None:
    reference = _frozen_point()
    candidate = _frozen_point(
        miou_q=10_001,
        niou_q=10_001,
        matched=11,
        unmatched_pixels=99,
        matched_tiny=6,
    )
    records = _pairwise_records(reference, candidate)
    summary = finalizer.pairwise_summary(
        records, "original", finalizer.FROZEN_METRICS
    )

    assert summary["vector_dimension"] == 30
    assert (summary["better"], summary["equal"], summary["worse"]) == (30, 0, 0)
    assert summary["relation"] == "dominates"
    assert summary["better_than_worse"] is True
    assert finalizer.point_dominates(
        candidate, reference, finalizer.FROZEN_METRICS
    )
    assert not finalizer.point_dominates(
        reference, candidate, finalizer.FROZEN_METRICS
    )

    tradeoff = dict(candidate)
    tradeoff["unmatched_predicted_pixels"] = 101
    assert not finalizer.point_dominates(
        tradeoff, reference, finalizer.FROZEN_METRICS
    )
    assert not finalizer.point_dominates(
        reference, tradeoff, finalizer.FROZEN_METRICS
    )


def test_real_artifacts_produce_frozen_final_decision() -> None:
    records, bindings = finalizer.load_records()
    result = finalizer.build_comparison(records, bindings)
    gates = result["gates"]

    assert gates["V3_B_original_floor"][
        "severe_degradation_violation_count"
    ] == 5
    assert len(gates["V3_C_anchor_strengths"]["violations"]) == 4

    comparisons = gates["V3_D_pairwise_performance"]["comparisons"]
    assert (
        comparisons["original"]["better"],
        comparisons["original"]["equal"],
        comparisons["original"]["worse"],
    ) == (13, 2, 15)
    assert (
        comparisons["tss_off"]["better"],
        comparisons["tss_off"]["equal"],
        comparisons["tss_off"]["worse"],
    ) == (10, 2, 18)
    assert (
        comparisons["lambda_0p005"]["better"],
        comparisons["lambda_0p005"]["equal"],
        comparisons["lambda_0p005"]["worse"],
    ) == (13, 2, 15)

    assert gates["V3_E_joint_pareto"][
        "ec_unique_non_dominated_cell_count"
    ] == 4
    assert result["decision"] == finalizer.FAIL_DECISION
    assert result["tss_optimization_closed"] is True

