from __future__ import annotations

import math

import pytest

from experiments import pbdr_v3_zero_margin_role_gate as gate


def _metrics(**updates: float | int) -> gate.CertificationMetrics:
    values: dict[str, float | int] = {
        "matched_target_count": 8,
        "target_count": 10,
        "fa": 0.01,
        "miou": 0.70,
        "niou": 0.65,
        "matched_tiny_target_count": 2,
        "tiny_target_count": 4,
        "tiny_pd": 0.5,
        "test_loss": 0.1,
    }
    values.update(updates)
    return gate.CertificationMetrics(**values)


def test_any_positive_primary_gain_passes() -> None:
    current = _metrics()
    candidate = _metrics(miou=math.nextafter(current.miou, math.inf))
    decision = gate.certify("best_miou", current, candidate)
    assert decision.passed is True
    assert decision.decisive_index == 0
    assert decision.decisive_term == "higher_miou"


def test_exact_tie_does_not_pass() -> None:
    decision = gate.certify("best_miou", _metrics(), _metrics())
    assert decision.passed is False
    assert decision.selected == "current"
    assert decision.decisive_index is None


def test_best_pd_primary_loss_is_not_offset_by_other_metrics() -> None:
    current = _metrics()
    candidate = _metrics(
        matched_target_count=7,
        fa=0.0,
        miou=1.0,
        niou=1.0,
        matched_tiny_target_count=4,
        tiny_pd=1.0,
        test_loss=0.0,
    )
    decision = gate.certify("best_pd", current, candidate)
    assert decision.passed is False
    assert decision.decisive_term == "higher_pd"


def test_best_pd_equal_pd_then_any_lower_fa_passes() -> None:
    current = _metrics()
    candidate = _metrics(fa=math.nextafter(current.fa, 0.0))
    decision = gate.certify("best_pd", current, candidate)
    assert decision.passed is True
    assert decision.decisive_index == 1
    assert decision.decisive_term == "lower_fa"


def test_adapter_reconstructs_decisive_fields_from_legacy_payload() -> None:
    adapter = gate.RoleGateAdapter("best_miou")
    current = _metrics()
    candidate = _metrics(miou=math.nextafter(current.miou, math.inf))
    expected = adapter.certify(current, candidate)
    reconstructed = adapter.CertificationDecision(
        passed=expected.passed,
        selected=expected.selected,
        checks=dict(expected.checks),
        current=current,
        candidate=candidate,
    )
    assert reconstructed == expected
    assert reconstructed.decisive_term == "higher_miou"


def test_tiny_pd_must_equal_its_integer_ratio_with_zero_total_semantics() -> None:
    with pytest.raises(ValueError, match="tiny_pd differs"):
        _metrics(tiny_pd=0.75)
    zero = _metrics(
        matched_tiny_target_count=0,
        tiny_target_count=0,
        tiny_pd=0.0,
    )
    assert zero.tiny_pd == 0.0
    with pytest.raises(ValueError, match="tiny_pd differs"):
        _metrics(
            matched_tiny_target_count=0,
            tiny_target_count=0,
            tiny_pd=1.0,
        )


def test_tiny_denominator_mismatch_cannot_pass_on_a_primary_gain() -> None:
    current = _metrics()
    candidate = _metrics(
        miou=math.nextafter(current.miou, math.inf),
        tiny_target_count=5,
        tiny_pd=0.4,
    )
    decision = gate.certify("best_miou", current, candidate)
    assert decision.checks["strict_role_performance_gain"] is True
    assert decision.checks["tiny_target_count_equal"] is False
    assert decision.passed is False


def test_tiny_counts_must_be_subsets_of_all_target_counts() -> None:
    with pytest.raises(ValueError, match="tiny_target_count exceeds"):
        _metrics(matched_target_count=3, target_count=3, tiny_target_count=4)
    with pytest.raises(ValueError, match="matched_tiny_target_count exceeds"):
        _metrics(
            matched_target_count=1,
            matched_tiny_target_count=2,
        )
