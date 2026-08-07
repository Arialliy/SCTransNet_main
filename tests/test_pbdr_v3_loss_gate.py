from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from experiments.pbdr_v3_loss import (
    compute_pbdr_v3_loss,
    soft_iou_loss,
    topk_hard_negative_loss,
)
from experiments.pbdr_v3_non_regression_gate import (
    SCHEMA,
    CertificationDecision,
    CertificationMetrics,
    certify,
    write_decision,
)


def _metrics(**overrides: object) -> CertificationMetrics:
    values: dict[str, object] = {
        "matched_target_count": 256,
        "target_count": 263,
        "fa": 1.5435e-5,
        "miou": 0.796483,
        "niou": 0.795348,
    }
    values.update(overrides)
    return CertificationMetrics.from_mapping(values)


def test_loss_components_match_documented_formula_and_backward() -> None:
    routed = torch.tensor(
        [[[[0.4, -0.2], [0.3, -0.7]]]],
        requires_grad=True,
    )
    base = torch.tensor(
        [[[[0.0, 0.1], [0.6, -0.8]]]],
        requires_grad=True,
    )
    delta = (routed - base.detach()).clone()
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    auxiliary = torch.zeros_like(routed, requires_grad=True)

    result = compute_pbdr_v3_loss(
        routed_logits=routed,
        base_logits=base,
        delta_logits=delta,
        target=target,
        auxiliary_logits=(auxiliary,),
        soft_iou_weight=0.7,
        background_increase_weight=3.0,
        foreground_decrease_weight=4.0,
        trust_region_weight=0.2,
        residual_sparsity_weight=0.1,
        hard_negative_weight=0.5,
        deep_supervision_weight=0.25,
        hard_negative_candidate_floor=0.0,
        hard_negative_topk_fraction=0.5,
    )

    probability = routed.sigmoid()
    base_probability = base.detach().sigmoid()
    background = target < 0.5
    foreground = ~background
    expected_background = (
        F.relu(probability - base_probability).square()[background].mean()
    )
    expected_foreground = (
        F.relu(base_probability - probability).square()[foreground].mean()
    )
    expected_total = (
        F.binary_cross_entropy_with_logits(routed, target)
        + 0.7 * soft_iou_loss(probability, target)
        + 3.0 * expected_background
        + 4.0 * expected_foreground
        + 0.2 * (probability - base_probability).square().mean()
        + 0.1 * delta.abs().mean()
        + 0.5
        * topk_hard_negative_loss(
            routed,
            probability,
            base_probability,
            target,
            candidate_floor=0.0,
            topk_fraction=0.5,
        )
        + 0.25 * F.binary_cross_entropy_with_logits(auxiliary, target)
    )
    torch.testing.assert_close(result.background_increase, expected_background)
    torch.testing.assert_close(result.foreground_decrease, expected_foreground)
    torch.testing.assert_close(result.total, expected_total)

    result.total.backward()
    assert routed.grad is not None
    assert torch.isfinite(routed.grad).all()
    assert auxiliary.grad is not None
    assert base.grad is None


def test_one_way_constraints_penalize_only_unsafe_directions() -> None:
    base = torch.zeros(1, 1, 1, 2)
    target = torch.tensor([[[[0.0, 1.0]]]])

    unsafe = compute_pbdr_v3_loss(
        routed_logits=torch.tensor([[[[1.0, -1.0]]]]),
        base_logits=base,
        delta_logits=torch.tensor([[[[1.0, -1.0]]]]),
        target=target,
        hard_negative_weight=0.0,
    )
    safe = compute_pbdr_v3_loss(
        routed_logits=torch.tensor([[[[-1.0, 1.0]]]]),
        base_logits=base,
        delta_logits=torch.tensor([[[[-1.0, 1.0]]]]),
        target=target,
        hard_negative_weight=0.0,
    )
    assert unsafe.background_increase.item() > 0.0
    assert unsafe.foreground_decrease.item() > 0.0
    assert safe.background_increase.item() == 0.0
    assert safe.foreground_decrease.item() == 0.0


@pytest.mark.parametrize("target_value", [0.0, 1.0])
def test_empty_mask_terms_are_finite_zero(target_value: float) -> None:
    target = torch.full((2, 1, 3, 4), target_value)
    logits = torch.zeros_like(target, requires_grad=True)
    result = compute_pbdr_v3_loss(
        routed_logits=logits,
        base_logits=torch.zeros_like(target),
        delta_logits=torch.zeros_like(target),
        target=target,
        hard_negative_weight=0.0,
    )
    term = (
        result.foreground_decrease
        if target_value == 0.0
        else result.background_increase
    )
    assert term.item() == 0.0
    assert torch.isfinite(result.total)
    result.total.backward()
    assert torch.isfinite(logits.grad).all()


def test_hard_negative_selects_ceil_topk_and_handles_no_candidates() -> None:
    logits = torch.tensor([[[[-2.0, -1.0, 0.0, 1.0]]]])
    probabilities = logits.sigmoid()
    target = torch.zeros_like(logits)
    actual = topk_hard_negative_loss(
        logits,
        probabilities,
        probabilities,
        target,
        candidate_floor=0.0,
        topk_fraction=0.26,
    )
    expected = F.softplus(logits.flatten()).topk(2).values.mean()
    torch.testing.assert_close(actual, expected)

    none = topk_hard_negative_loss(
        logits,
        probabilities,
        probabilities,
        torch.ones_like(target),
    )
    assert none.item() == 0.0


def test_loss_rejects_bad_shapes_devices_and_hyperparameters() -> None:
    target = torch.zeros(1, 1, 2, 2)
    logits = torch.zeros_like(target)
    with pytest.raises(ValueError, match="base_logits shape"):
        compute_pbdr_v3_loss(
            routed_logits=logits,
            base_logits=torch.zeros(1, 1, 2, 3),
            delta_logits=logits,
            target=target,
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        compute_pbdr_v3_loss(
            routed_logits=logits,
            base_logits=logits,
            delta_logits=logits,
            target=target,
            trust_region_weight=-1.0,
        )
    with pytest.raises(ValueError, match="must be finite"):
        compute_pbdr_v3_loss(
            routed_logits=logits,
            base_logits=logits,
            delta_logits=logits,
            target=target,
            soft_iou_weight=float("nan"),
        )
    with pytest.raises(ValueError, match="topk_fraction"):
        compute_pbdr_v3_loss(
            routed_logits=logits,
            base_logits=logits,
            delta_logits=logits,
            target=target,
            hard_negative_topk_fraction=0.0,
        )


def test_zero_deep_supervision_weight_does_not_consume_auxiliary_values() -> None:
    target = torch.zeros(1, 1, 2, 2)
    logits = torch.zeros_like(target, requires_grad=True)
    malformed = torch.full((1,), float("nan"), requires_grad=True)
    result = compute_pbdr_v3_loss(
        routed_logits=logits,
        base_logits=torch.zeros_like(target),
        delta_logits=torch.zeros_like(target),
        target=target,
        auxiliary_logits=(malformed,),
        deep_supervision_weight=0.0,
        hard_negative_weight=0.0,
    )
    assert result.deep_supervision.item() == 0.0
    result.total.backward()
    assert malformed.grad is None


def test_non_regression_gate_falls_back_to_current() -> None:
    current = _metrics()
    candidate = _metrics(
        matched_target_count=257,
        fa=1.4e-5,
        miou=0.799,
        niou=0.79,
    )
    decision = certify(current, candidate)
    assert not decision.passed
    assert decision.selected == "current"
    assert decision.checks == {
        "pd_non_regression": True,
        "fa_non_regression": True,
        "miou_strict_gain": True,
        "niou_non_regression": False,
    }


def test_non_regression_gate_selects_only_full_dominance() -> None:
    current = _metrics()
    candidate = _metrics(
        matched_target_count=256,
        fa=1.4e-5,
        miou=current.miou + 0.002,
        niou=current.niou,
    )
    decision = certify(current, candidate, maximum_fa_ratio=0.95)
    assert decision.passed
    assert decision.selected == "candidate"
    assert all(decision.checks.values())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"matched_target_count": 264}, "matched_target_count"),
        ({"target_count": -1}, "target_count"),
        ({"fa": -1.0}, "fa"),
        ({"miou": float("nan")}, "miou must be finite"),
        ({"niou": 1.1}, "niou"),
    ],
)
def test_certification_metrics_validate_ranges(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _metrics(**overrides)


def test_certify_validates_contract() -> None:
    current = _metrics()
    with pytest.raises(ValueError, match="target counts differ"):
        certify(current, _metrics(target_count=264))
    with pytest.raises(ValueError, match="maximum_fa_ratio"):
        certify(current, current, maximum_fa_ratio=1.01)
    with pytest.raises(ValueError, match="minimum_miou_gain"):
        certify(current, current, minimum_miou_gain=-0.1)
    with pytest.raises(ValueError, match="selected conflicts"):
        CertificationDecision(
            passed=True,
            selected="current",
            checks={
                "pd_non_regression": True,
                "fa_non_regression": True,
                "miou_strict_gain": True,
                "niou_non_regression": True,
            },
            current=current,
            candidate=current,
        )


def test_write_decision_atomically_replaces_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deployment.json"
    path.parent.mkdir()
    path.write_text('{"stale": true}\n', encoding="utf-8")
    decision = certify(
        _metrics(),
        _metrics(fa=1.4e-5, miou=0.799, niou=0.796),
    )
    write_decision(path, decision)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["passed"] is True
    assert payload["selected"] == "candidate"
    assert payload["scope"] == "frozen_certification_split_only"
    assert payload["unseen_test_guarantee"] is False
    assert path.read_bytes().endswith(b"\n")
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deployment.json"
    previous = b'{"selected": "current"}\n'
    path.write_bytes(previous)
    decision = certify(_metrics(), _metrics(miou=0.7))

    with mock.patch(
        "experiments.pbdr_v3_non_regression_gate.os.replace",
        side_effect=OSError("injected replace failure"),
    ):
        with pytest.raises(OSError, match="injected replace failure"):
            write_decision(path, decision)

    assert path.read_bytes() == previous
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_write_decision_uses_same_directory_atomic_replace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deployment.json"
    decision = certify(_metrics(), _metrics(miou=0.7))
    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def recording_replace(source: os.PathLike[str], destination: os.PathLike[str]):
        calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    with mock.patch(
        "experiments.pbdr_v3_non_regression_gate.os.replace",
        side_effect=recording_replace,
    ):
        write_decision(path, decision)

    assert len(calls) == 1
    assert calls[0][0].parent == path.parent
    assert calls[0][1] == path
