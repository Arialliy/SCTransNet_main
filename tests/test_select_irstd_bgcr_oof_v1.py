from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.irstd_bgcr_run_contract import (
    BGCR_CANDIDATE_KIND,
    BGCR_CANDIDATE_NAME,
    DATASET,
    FOLD_ASSIGNMENT_SHA256,
    FOLD_SIZES,
    OFFICIAL_FALSE_FLAGS,
    OOF_EVALUATION_EPOCHS,
    PERFORMANCE_ACCEPTANCE_MARGIN,
    PROBABILITY_COMPARISON,
    PROBABILITY_THRESHOLD,
    ROLE,
    SOURCE_SCOPE,
    SOURCE_SPLIT_MANIFEST_FILE_SHA256,
    build_frozen_fold_manifest,
    fold_metric_binding,
)
from experiments.select_irstd_bgcr_oof_v1 import (
    BASELINE_REFERENCE_NAME,
    IRSTDBGCRSelectorError,
    _commit_or_validate_json,
    select_from_fold_summaries,
)


FOLD_SUMMARY_SCHEMA = "sctransnet_irstd_bgcr_training_v1/fold_summary"


def _metric_row(
    fold_index: int,
    epoch: int,
    *,
    fold_manifest: dict[str, object],
    intersection_delta: int = 0,
) -> dict[str, object]:
    sample_count = FOLD_SIZES[fold_index]
    intersection = 1_000 + fold_index * 10 + intersection_delta
    union = 2_000 + fold_index * 100
    false_positive = 400 + fold_index
    row = fold_metric_binding(
        fold_index,
        epoch,
        fold_manifest=fold_manifest,
    )
    row.update(
        {
            "candidate_kind": BGCR_CANDIDATE_KIND,
            "candidate_name": BGCR_CANDIDATE_NAME,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "matched_target_count": 20 + fold_index,
            "target_count": 22 + fold_index,
            "unmatched_component_pixels": 100 + fold_index,
            "valid_pixel_count": sample_count * 100,
            "matched_tiny_target_count": 5 + fold_index,
            "tiny_target_count": 7 + fold_index,
            "true_positive_pixels": intersection,
            "false_positive_pixels": false_positive,
            "false_negative_pixels": union - intersection - false_positive,
            "niou_sum_numerator": sample_count * 3,
            "niou_sum_denominator": 4,
            "loss_sum_numerator": sample_count,
            "loss_sum_denominator": 4,
        }
    )
    return row


def _write_fold_summaries(
    root: Path,
    *,
    epochs: tuple[int, ...],
    improved_epoch: int | None = None,
) -> list[Path]:
    fold_manifest = build_frozen_fold_manifest()
    paths: list[Path] = []
    for fold_index in range(3):
        history = [
            _metric_row(
                fold_index,
                epoch,
                fold_manifest=fold_manifest,
                intersection_delta=(
                    1
                    if improved_epoch == epoch and fold_index == 0
                    else 0
                ),
            )
            for epoch in epochs
        ]
        payload: dict[str, object] = {
            "schema": FOLD_SUMMARY_SCHEMA,
            "dataset": DATASET,
            "role": ROLE,
            "source_scope": SOURCE_SCOPE,
            "fold_index": fold_index,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "fold_manifest_sha256": fold_manifest["manifest_sha256"],
            "source_split_manifest_file_sha256": (
                SOURCE_SPLIT_MANIFEST_FILE_SHA256
            ),
            "probability_threshold": PROBABILITY_THRESHOLD,
            "probability_comparison": PROBABILITY_COMPARISON,
            "performance_acceptance_margin": PERFORMANCE_ACCEPTANCE_MARGIN,
            "baseline1000_metric_row": {
                **_metric_row(
                    fold_index,
                    0,
                    fold_manifest=fold_manifest,
                ),
                "candidate_kind": "frozen_reference",
                "candidate_name": BASELINE_REFERENCE_NAME,
            },
            "evaluation_history": history,
            **OFFICIAL_FALSE_FLAGS,
        }
        path = root / f"fold_{fold_index}_summary.json"
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_synthetic_partial_history_is_explicitly_diagnostic(tmp_path: Path) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    selected = select_from_fold_summaries(paths, require_complete=False)
    assert selected["selected_epoch"] == 0
    assert selected["dataset"] == DATASET
    assert selected["role"] == ROLE
    assert selected["source_scope"] == SOURCE_SCOPE
    assert selected["selection"]["candidate_epochs"] == [0, 5]
    assert selected["performance_acceptance_margin"] is None
    baseline = selected["baseline1000_internal_oof_metrics"]
    assert baseline["candidate_kind"] == "frozen_reference"
    assert baseline["candidate_name"] == BASELINE_REFERENCE_NAME
    assert baseline["epoch"] == 0
    assert baseline["sample_count"] == sum(FOLD_SIZES)
    assert baseline["pixel_f1_exact"] == {
        "numerator": 202,
        "denominator": 311,
    }
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert selected[flag] is expected
        assert baseline[flag] is expected


def test_synthetic_partial_history_is_rejected_in_complete_mode(
    tmp_path: Path,
) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    with pytest.raises(IRSTDBGCRSelectorError, match="complete OOF history"):
        select_from_fold_summaries(paths)


def test_synthetic_complete_history_requires_all_frozen_epochs(tmp_path: Path) -> None:
    paths = _write_fold_summaries(
        tmp_path,
        epochs=OOF_EVALUATION_EPOCHS,
        improved_epoch=120,
    )
    selected = select_from_fold_summaries(paths)
    assert selected["selected_epoch"] == 120
    assert selected["selection"]["candidate_epochs"] == list(
        OOF_EVALUATION_EPOCHS
    )
    assert selected["strictly_improves_epoch0_miou"] is True


def test_epoch_zero_wins_an_exact_pooled_tie(tmp_path: Path) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5, 10))
    selected = select_from_fold_summaries(paths, require_complete=False)
    assert selected["selected_epoch"] == 0
    assert selected["strictly_improves_epoch0_miou"] is False
    assert selected["strictly_improves_epoch0_full_role_key"] is False


def test_any_strict_exact_miou_gain_is_selected_without_margin(tmp_path: Path) -> None:
    paths = _write_fold_summaries(
        tmp_path,
        epochs=(0, 5),
        improved_epoch=5,
    )
    selected = select_from_fold_summaries(paths, require_complete=False)
    assert selected["selected_epoch"] == 5
    assert selected["strictly_improves_epoch0_miou"] is True
    assert selected["performance_acceptance_margin"] is None


def test_official_flag_drift_fails_before_pooling(tmp_path: Path) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    changed = _read(paths[1])
    changed["official_test_accessed"] = True
    _rewrite(paths[1], changed)
    with pytest.raises(IRSTDBGCRSelectorError, match="official flags"):
        select_from_fold_summaries(paths, require_complete=False)


def test_row_level_official_flag_drift_fails_inside_exact_pool(tmp_path: Path) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    changed = _read(paths[2])
    history = changed["evaluation_history"]
    assert isinstance(history, list) and isinstance(history[0], dict)
    history[0]["official_test_index_parsed"] = True
    _rewrite(paths[2], changed)
    with pytest.raises(IRSTDBGCRSelectorError, match="official_test_index_parsed"):
        select_from_fold_summaries(paths, require_complete=False)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_kind", "frozen_reference"),
        ("candidate_name", "Current"),
    ),
)
def test_history_candidate_identity_drift_fails_before_pooling(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    changed = _read(paths[2])
    history = changed["evaluation_history"]
    assert isinstance(history, list) and isinstance(history[1], dict)
    history[1][field] = value
    _rewrite(paths[2], changed)
    with pytest.raises(IRSTDBGCRSelectorError, match="candidate identity"):
        select_from_fold_summaries(paths, require_complete=False)


def test_unknown_summary_schema_is_rejected(tmp_path: Path) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    changed = _read(paths[0])
    changed["schema"] = "arbitrary_string_was_previously_accepted"
    _rewrite(paths[0], changed)
    with pytest.raises(IRSTDBGCRSelectorError, match="schema is unsupported"):
        select_from_fold_summaries(paths, require_complete=False)


def test_noncanonical_fold_history_field_is_rejected(tmp_path: Path) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    changed = _read(paths[0])
    changed["fold_history"] = changed.pop("evaluation_history")
    _rewrite(paths[0], changed)
    with pytest.raises(IRSTDBGCRSelectorError, match="registered history field"):
        select_from_fold_summaries(paths, require_complete=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "baseline1000_metric_row is missing"),
        ("wrong_name", "reference identity differs"),
    ),
)
def test_all_three_baseline1000_reference_rows_are_required(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    changed = _read(paths[2])
    if mutation == "missing":
        del changed["baseline1000_metric_row"]
    else:
        baseline = changed["baseline1000_metric_row"]
        assert isinstance(baseline, dict)
        baseline["candidate_name"] = "not-the-frozen-reference"
    _rewrite(paths[2], changed)
    with pytest.raises(IRSTDBGCRSelectorError, match=message):
        select_from_fold_summaries(paths, require_complete=False)


def test_duplicate_fold_summary_is_rejected(tmp_path: Path) -> None:
    paths = _write_fold_summaries(tmp_path, epochs=(0, 5))
    changed = _read(paths[1])
    changed["fold_index"] = 0
    _rewrite(paths[1], changed)
    with pytest.raises(IRSTDBGCRSelectorError, match="duplicate fold summary"):
        select_from_fold_summaries(paths, require_complete=False)


def test_selector_output_is_append_only_and_idempotent(tmp_path: Path) -> None:
    paths = _write_fold_summaries(
        tmp_path,
        epochs=(0, 5),
        improved_epoch=5,
    )
    payload = select_from_fold_summaries(paths, require_complete=False)
    destination = tmp_path / "selection.json"
    first = _commit_or_validate_json(destination, payload)
    original_bytes = first.read_bytes()
    second = _commit_or_validate_json(destination, payload)
    assert second == first
    assert second.read_bytes() == original_bytes

    changed = copy.deepcopy(payload)
    changed["selected_epoch"] = 0
    with pytest.raises(IRSTDBGCRSelectorError, match="output differs"):
        _commit_or_validate_json(destination, changed)
    assert destination.read_bytes() == original_bytes
