from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from analysis import audit_final_qfg_functional_use as subject
from analysis import collect_final_model_validation_statistics as cache_core
from model.tpd_frequency_gate_v2_croa import QueryOnlyFrequencyGateV2CROA


class _DummyLevel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))


class _DummyQFG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [_DummyLevel(value) for value in (0.1, -0.2, 0.3, -0.4)]
        )

    def prepare(self, scale: float = 1.0) -> SimpleNamespace:
        prepared = []
        for index, level in enumerate(self.levels):
            factor = 1.0 + torch.tanh(level.alpha) * (index + 1) * scale
            prepared.append(
                SimpleNamespace(factor=factor.reshape(1, 1, 1, 1))
            )
        return SimpleNamespace(levels=tuple(prepared))


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tpd_qfg = _DummyQFG()
        self.eval()


def _alpha_values(model: _DummyModel) -> tuple[torch.Tensor, ...]:
    return tuple(
        level.alpha.detach().clone()
        for level in model.tpd_qfg.levels
    )


def _identity(mode: str, *, evaluator: str = "c") -> dict[str, object]:
    image_ids = tuple(
        f"synthetic/image_{index:03d}"
        for index in range(cache_core.EXPECTED_VALIDATION_COUNT)
    )
    return cache_core.build_cache_identity(
        checkpoint_sha256="a" * 64,
        dataset_sha256="b" * 64,
        evaluator_sha256=evaluator * 64,
        normalization_sha256="d" * 64,
        source_lock_sha256="e" * 64,
        validation_ids_sha256=(
            cache_core.validation_identifier_sha256(image_ids)
        ),
        validation_count=cache_core.EXPECTED_VALIDATION_COUNT,
        match_radius=3.0,
        tiny_area=9,
        mode=mode,
    )


def _cache(
    mode: str,
    *,
    delta: float = 0.0,
    evaluator: str = "c",
) -> cache_core.PredictionCache:
    probability = np.asarray(
        [[0.1, 0.9], [0.1, 0.1]],
        dtype=np.float32,
    )
    probability = np.clip(
        probability + np.float32(delta),
        0.0,
        1.0,
    ).astype(np.float32)
    target = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
    return cache_core.create_prediction_cache(
        [
            cache_core.PredictionRecord(
                f"synthetic/image_{index:03d}",
                probability,
                target,
                loss=0.25,
            )
            for index in range(cache_core.EXPECTED_VALIDATION_COUNT)
        ],
        identity=_identity(mode, evaluator=evaluator),
    )


@pytest.mark.parametrize(
    ("mode", "selected"),
    [
        ("full", ()),
        ("all_off", (0, 1, 2, 3)),
        ("level_1_off", (0,)),
        ("level_2_off", (1,)),
        ("level_3_off", (2,)),
        ("level_4_off", (3,)),
    ],
)
def test_knockout_changes_only_selected_levels_and_restores_exactly(
    mode: str,
    selected: tuple[int, ...],
) -> None:
    model = _DummyModel()
    before_values = _alpha_values(model)
    before_sha = subject.alpha_state_sha256(model)

    with subject.temporary_qfg_alpha_knockout(model, mode) as audit:
        inside_values = _alpha_values(model)
        assert audit["selected_level_indices_zero_based"] == list(selected)
        assert audit["derived_checkpoint_written"] is False
        for index, (before, inside) in enumerate(
            zip(before_values, inside_values)
        ):
            if index in selected:
                assert torch.count_nonzero(inside).item() == 0
            else:
                assert torch.equal(inside, before)

    assert subject.alpha_state_sha256(model) == before_sha
    for before, after in zip(before_values, _alpha_values(model)):
        assert torch.equal(after, before)


def test_knockout_restores_parameters_when_caller_raises() -> None:
    model = _DummyModel()
    before = _alpha_values(model)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with subject.temporary_qfg_alpha_knockout(model, "all_off"):
            raise RuntimeError("synthetic failure")
    for expected, actual in zip(before, _alpha_values(model)):
        assert torch.equal(actual, expected)


def test_factor_capture_is_nonpersistent_and_detects_active_or_off() -> None:
    model = _DummyModel()
    assert "prepare" not in model.tpd_qfg.__dict__
    with subject.capture_qfg_prepared_factors(model) as capture:
        model.tpd_qfg.prepare(scale=0.5)
        model.tpd_qfg.prepare(scale=1.0)
    assert "prepare" not in model.tpd_qfg.__dict__
    summary = capture.summary()
    assert summary["forward_count"] == 2
    assert summary["nontrivial_factor_use"] is True
    assert summary["maximum_level_mean_abs_factor_minus_one"] > 0.0

    with subject.temporary_qfg_alpha_knockout(model, "all_off"):
        with subject.capture_qfg_prepared_factors(model) as off_capture:
            model.tpd_qfg.prepare()
    off_summary = off_capture.summary()
    assert off_summary["nontrivial_factor_use"] is False
    assert off_summary["maximum_level_mean_abs_factor_minus_one"] == 0.0

    prepared = model.tpd_qfg.prepare()
    assert any(
        not torch.equal(level.factor, torch.ones_like(level.factor))
        for level in prepared.levels
    )


def test_probability_audit_detects_functional_difference_or_equivalence() -> None:
    full = _cache("full")
    changed = _cache("all_off", delta=-0.01)
    report = subject.audit_probability_caches(
        full,
        changed,
        threshold=0.5,
    )
    assert report["official_test_accessed"] is False
    assert report["output_difference"]["functionally_different"] is True
    assert report["output_difference"]["max_abs"] == pytest.approx(0.01)
    assert report["full_metrics"]["pd"] == 1.0
    assert report["counterfactual_metrics"]["pd"] == 1.0

    identical = _cache("level_1_off")
    equivalent = subject.audit_probability_caches(full, identical)
    assert equivalent["output_difference"]["equivalent"] is True
    assert equivalent["output_difference"]["max_abs"] == 0.0


def test_probability_audit_rejects_incompatible_identity() -> None:
    full = _cache("full")
    incompatible = _cache("all_off", evaluator="e")
    with pytest.raises(ValueError, match="compatibility differs"):
        subject.audit_probability_caches(full, incompatible)


def test_capture_and_knockout_match_real_qfg_v2_croa_api() -> None:
    torch.manual_seed(17)
    gate = QueryOnlyFrequencyGateV2CROA(
        (2, 3, 4, 5),
        hidden_channels=3,
    )
    gate.eval()
    with torch.no_grad():
        for level in gate.levels:
            level.gate_out.weight.fill_(1.0)
    features = (
        torch.randn(1, 2, 32, 48),
        torch.randn(1, 3, 16, 24),
        torch.randn(1, 4, 8, 12),
        torch.randn(1, 5, 4, 6),
    )
    source_sha = subject.alpha_state_sha256(gate)
    with subject.capture_qfg_prepared_factors(gate) as active_capture:
        gate.prepare(features, (2, 3))
    assert active_capture.summary()["nontrivial_factor_use"] is True

    with subject.temporary_qfg_alpha_knockout(gate, "all_off"):
        with subject.capture_qfg_prepared_factors(gate) as off_capture:
            prepared = gate.prepare(features, (2, 3))
        assert all(
            torch.equal(level.factor, torch.ones_like(level.factor))
            for level in prepared.levels
        )
    assert off_capture.summary()["nontrivial_factor_use"] is False
    assert subject.alpha_state_sha256(gate) == source_sha


def test_knockout_rejects_training_mode() -> None:
    model = _DummyModel()
    model.train()
    with pytest.raises(ValueError, match=r"model\.eval"):
        with subject.temporary_qfg_alpha_knockout(model, "all_off"):
            pass


def test_knockout_rejects_training_root_even_if_qfg_is_eval() -> None:
    model = _DummyModel()
    model.train()
    model.tpd_qfg.eval()
    assert model.training is True
    assert model.tpd_qfg.training is False
    with pytest.raises(ValueError, match=r"model\.eval"):
        with subject.temporary_qfg_alpha_knockout(model, "all_off"):
            pass
    with pytest.raises(ValueError, match=r"model\.eval"):
        with subject.capture_qfg_prepared_factors(model):
            pass
