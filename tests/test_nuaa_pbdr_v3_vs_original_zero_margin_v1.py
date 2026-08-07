from __future__ import annotations

import copy
import json
import math

import pytest

from analysis import adjudicate_nuaa_pbdr_v3_vs_original_zero_margin_v1 as adjudicator


def _metrics() -> dict[str, float | int]:
    return {
        "matched_target_count": 10,
        "pd": 0.5,
        "fa": 0.01,
        "miou": 0.6,
        "niou": 0.55,
        "matched_tiny_target_count": 2,
        "tiny_pd": 0.4,
        "tiny_target_count": 5,
        "unmatched_predicted_object_count": 4,
        "unmatched_predicted_pixel_count": 100,
        "test_loss": 0.2,
    }


def test_any_positive_primary_gain_wins_without_a_minimum_margin() -> None:
    original = _metrics()
    candidate = copy.deepcopy(original)
    candidate["miou"] = math.nextafter(float(original["miou"]), math.inf)

    result = adjudicator.compare_role(
        "best_miou",
        original,
        candidate,
        original_epoch=10,
        candidate_epoch=20,
    )

    assert result["minimum_gain"] == 0.0
    assert result["advisory_metric_winner"] == "candidate"
    assert result["decisive_index"] == 0
    assert result["decisive_term"] == "higher_miou"


def test_exact_tie_keeps_original() -> None:
    metrics = _metrics()
    result = adjudicator.compare_role(
        "best_miou",
        metrics,
        copy.deepcopy(metrics),
        original_epoch=10,
        candidate_epoch=10,
    )
    assert result["exact_tie"] is True
    assert result["advisory_metric_winner"] == "original"
    assert result["decisive_index"] is None


def test_best_pd_lower_primary_pd_loses_despite_other_improvements() -> None:
    original = _metrics()
    candidate = copy.deepcopy(original)
    candidate.update(
        pd=0.49,
        fa=0.0,
        tiny_pd=1.0,
        miou=1.0,
        niou=1.0,
        test_loss=0.0,
    )
    result = adjudicator.compare_role(
        "best_pd",
        original,
        candidate,
        original_epoch=10,
        candidate_epoch=1,
    )
    assert result["advisory_metric_winner"] == "original"
    assert result["decisive_term"] == "higher_pd"


def test_best_pd_equal_pd_then_lower_fa_wins() -> None:
    original = _metrics()
    candidate = copy.deepcopy(original)
    candidate["fa"] = math.nextafter(float(original["fa"]), 0.0)
    result = adjudicator.compare_role(
        "best_pd",
        original,
        candidate,
        original_epoch=10,
        candidate_epoch=20,
    )
    assert result["advisory_metric_winner"] == "candidate"
    assert result["decisive_index"] == 1
    assert result["decisive_term"] == "lower_fa"


def test_real_overlay_is_advisory_and_does_not_reaccess_test() -> None:
    payload = adjudicator.build_adjudication()
    assert payload["status"] == "advisory_complete_binding_blocked"
    assert payload["policy"]["official_test_reaccessed"] is False
    assert payload["policy"]["dataset_or_model_loaded"] is False
    assert payload["roles"]["best_miou"]["advisory_metric_winner"] == "candidate"
    assert payload["roles"]["best_pd"]["advisory_metric_winner"] == "original"
    for role in adjudicator.ROLES:
        binding = payload["roles"][role]["binding_decision"]
        assert binding["binding_eligible"] is False
        assert binding["binding_selected"] is None
        assert binding["effective_deployment"].endswith("unchanged")


def test_atomic_output_refuses_to_overwrite(tmp_path) -> None:
    destination = tmp_path / "overlay.json"
    adjudicator._atomic_json(destination, {"version": 1})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 1}
    with pytest.raises(FileExistsError):
        adjudicator._atomic_json(destination, {"version": 2})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 1}
