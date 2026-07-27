from __future__ import annotations

import pytest

from experiments import accept_tpd_clean_v6_formal800_results as accept


def _completion_result(*, gate: bool, ner: bool) -> dict:
    return {
        "status": "complete",
        "decision": "PASS" if gate else "FAIL",
        "engineering_gate_passed": gate,
        "ner_stage_authorized": ner,
        "input_count": 72,
        "manifest_sha256": "a" * 64,
        "marker_sha256": "b" * 64,
    }


def _strict_result(valid: int = 8) -> dict:
    return {
        "complete_and_strict_valid": valid == 8,
        "strict_valid_sweeps": valid,
    }


def test_acceptance_forces_both_verifier_layers(monkeypatch) -> None:
    calls = []

    def fake_completion(*args, **kwargs):
        calls.append("completion")
        return _completion_result(gate=False, ner=False)

    def fake_strict(*args, **kwargs):
        calls.append("strict")
        return _strict_result()

    monkeypatch.setattr(accept.completion, "verify_completion", fake_completion)
    monkeypatch.setattr(
        accept.strict, "validate_all_strict_sweeps", fake_strict
    )
    result = accept.verify_and_accept()
    assert calls == ["completion", "strict"]
    assert result["authoritative_result_accepted"] is True
    assert result["strict_valid_sweeps"] == 8
    assert result["ner_stage_authorized"] is False


def test_acceptance_rejects_incomplete_strict_matrix(monkeypatch) -> None:
    monkeypatch.setattr(
        accept.completion,
        "verify_completion",
        lambda *args, **kwargs: _completion_result(gate=False, ner=False),
    )
    monkeypatch.setattr(
        accept.strict,
        "validate_all_strict_sweeps",
        lambda *args, **kwargs: _strict_result(7),
    )
    with pytest.raises(RuntimeError, match="did not pass 8/8"):
        accept.verify_and_accept()


def test_acceptance_rejects_ner_without_full_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        accept.completion,
        "verify_completion",
        lambda *args, **kwargs: _completion_result(gate=False, ner=True),
    )
    monkeypatch.setattr(
        accept.strict,
        "validate_all_strict_sweeps",
        lambda *args, **kwargs: _strict_result(),
    )
    with pytest.raises(RuntimeError, match="NER was authorized"):
        accept.verify_and_accept()
