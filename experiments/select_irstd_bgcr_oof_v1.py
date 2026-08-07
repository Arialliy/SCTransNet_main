#!/usr/bin/env python3
"""Pool the three train-only BGCR folds and select one zero-margin epoch.

This entry point is deliberately pure with respect to images, datasets and
checkpoints.  It reads only the three already-produced fold summaries, pools
their additive sufficient statistics over all 800 held-out predictions and
uses the frozen IRSTD ``best_miou`` role key.  Epoch 0 is mandatory and an
exact tie keeps that identity candidate.  The three fold-local frozen
Baseline-epoch1000 rows are pooled by the same exact protocol and reported as
an internal OOF reference; they never enter candidate selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.irstd_bgcr_run_contract import (
    BGCR_CANDIDATE_KIND,
    BGCR_CANDIDATE_NAME,
    DATASET,
    FOLD_ASSIGNMENT_SHA256,
    FOLD_MANIFEST_SCHEMA,
    FOLD_TIE_ORDER,
    IRSTDBGCRRunContractError,
    OFFICIAL_FALSE_FLAGS,
    OOF_EVALUATION_EPOCHS,
    PERFORMANCE_ACCEPTANCE_MARGIN,
    PROBABILITY_COMPARISON,
    PROBABILITY_THRESHOLD,
    ROLE,
    SOURCE_SCOPE,
    SOURCE_SPLIT_MANIFEST_FILE_SHA256,
    build_frozen_fold_manifest,
    canonical_json_sha256,
    pool_reference_sufficient_statistics,
    select_oof_epoch,
)
from experiments.pbdr_v4_run_artifacts import exclusive_json, file_sha256


SCHEMA = "sctransnet_irstd_bgcr_oof_selector/v1"
SUMMARY_SCHEMAS = {
    "sctransnet_irstd_bgcr_training_v1/fold_summary",
}
SUMMARY_HISTORY_FIELDS = ("evaluation_history",)
BASELINE_REFERENCE_NAME = "Baseline-epoch1000"


class IRSTDBGCRSelectorError(RuntimeError):
    """A fold summary or selection destination violates the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IRSTDBGCRSelectorError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IRSTDBGCRSelectorError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def _official_flags_are_false(payload: Mapping[str, object]) -> bool:
    return all(
        payload.get(name) is expected
        for name, expected in OFFICIAL_FALSE_FLAGS.items()
    )


def _history_from_summary(
    payload: Mapping[str, object],
    *,
    expected_fold: int,
    expected_fold_manifest_sha256: str,
) -> list[dict[str, object]]:
    schema = payload.get("schema")
    _require(
        isinstance(schema, str) and schema in SUMMARY_SCHEMAS,
        "fold summary schema is unsupported",
    )
    _require(payload.get("dataset") == DATASET, "fold summary dataset differs")
    _require(payload.get("role") == ROLE, "fold summary role differs")
    _require(
        payload.get("source_scope") == SOURCE_SCOPE,
        "fold summary is not train-only",
    )
    _require(
        payload.get("fold_index") == expected_fold,
        "fold summary index differs",
    )
    _require(
        payload.get("performance_acceptance_margin") is PERFORMANCE_ACCEPTANCE_MARGIN,
        "fold summary performance margin must be null",
    )
    _require(_official_flags_are_false(payload), "fold summary official flags differ")
    _require(
        payload.get("fold_assignment_sha256") == FOLD_ASSIGNMENT_SHA256,
        "fold summary assignment SHA differs",
    )
    _require(
        payload.get("fold_manifest_sha256") == expected_fold_manifest_sha256,
        "fold summary manifest SHA differs",
    )
    _require(
        payload.get("source_split_manifest_file_sha256")
        == SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "fold summary source split SHA differs",
    )
    _require(
        payload.get("probability_threshold") == PROBABILITY_THRESHOLD
        and payload.get("probability_comparison") == PROBABILITY_COMPARISON,
        "fold summary probability contract differs",
    )
    present_history_fields = [
        name for name in SUMMARY_HISTORY_FIELDS if payload.get(name) is not None
    ]
    _require(
        len(present_history_fields) == 1,
        "fold summary must contain exactly one registered history field",
    )
    history_field = present_history_fields[0]
    raw = payload.get(history_field)
    _require(isinstance(raw, list) and bool(raw), "fold summary history is empty")
    history: list[dict[str, object]] = []
    for item in raw:
        _require(isinstance(item, Mapping), "fold history item must be a mapping")
        row: object = item.get("metric_row", item)
        _require(isinstance(row, Mapping), "fold history metric row is missing")
        ready = dict(row)
        _require(ready.get("fold_index") == expected_fold, "history fold index differs")
        epoch = ready.get("epoch")
        _require(
            isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and epoch in OOF_EVALUATION_EPOCHS,
            "fold history epoch is outside the frozen schedule",
        )
        _require(
            ready.get("candidate_kind") == BGCR_CANDIDATE_KIND
            and ready.get("candidate_name") == BGCR_CANDIDATE_NAME,
            "fold history candidate identity is not the registered BGCR arm",
        )
        history.append(ready)
    epochs = [int(row["epoch"]) for row in history]
    _require(
        epochs == sorted(set(epochs)),
        "fold history epochs are not strictly ordered",
    )
    _require(epochs[0] == 0, "fold history lacks epoch-0 identity")
    return history


def _baseline1000_row_from_summary(
    payload: Mapping[str, object],
    *,
    expected_fold: int,
) -> dict[str, object]:
    value = payload.get("baseline1000_metric_row")
    _require(
        isinstance(value, Mapping),
        "fold summary baseline1000_metric_row is missing",
    )
    row = dict(value)
    _require(
        row.get("fold_index") == expected_fold,
        "Baseline-epoch1000 row fold index differs",
    )
    _require(row.get("epoch") == 0, "Baseline-epoch1000 row must use epoch 0")
    _require(
        row.get("candidate_kind") == "frozen_reference"
        and row.get("candidate_name") == BASELINE_REFERENCE_NAME,
        "Baseline-epoch1000 row reference identity differs",
    )
    return row


def select_from_fold_summaries(
    paths: Sequence[Path],
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Validate three fold summaries and return one append-only selection."""

    _require(
        len(paths) == len(FOLD_TIE_ORDER),
        "exactly three fold summaries are required",
    )
    fold_manifest = build_frozen_fold_manifest()
    _require(
        fold_manifest.get("schema") == FOLD_MANIFEST_SCHEMA,
        "frozen fold manifest schema differs",
    )
    fold_manifest_sha256 = fold_manifest.get("manifest_sha256")
    _require(
        isinstance(fold_manifest_sha256, str),
        "frozen fold manifest SHA is missing",
    )
    fold_history: list[dict[str, object]] = []
    baseline1000_rows: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    observed_folds: set[int] = set()
    for path in paths:
        payload = _read_json(Path(path), label="fold summary")
        fold = payload.get("fold_index")
        _require(
            isinstance(fold, int) and not isinstance(fold, bool) and fold in FOLD_TIE_ORDER,
            "fold summary index is unsupported",
        )
        _require(fold not in observed_folds, "duplicate fold summary")
        observed_folds.add(fold)
        fold_history.extend(
            _history_from_summary(
                payload,
                expected_fold=fold,
                expected_fold_manifest_sha256=fold_manifest_sha256,
            )
        )
        baseline1000_rows.append(
            _baseline1000_row_from_summary(
                payload,
                expected_fold=fold,
            )
        )
        resolved = Path(path).resolve(strict=True)
        bindings.append(
            {
                "fold_index": fold,
                "path": str(resolved),
                "file_sha256": file_sha256(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    _require(
        observed_folds == set(FOLD_TIE_ORDER),
        "fold summaries do not cover 0, 1, 2",
    )
    bindings.sort(key=lambda item: int(item["fold_index"]))
    try:
        selection = select_oof_epoch(
            fold_history,
            require_complete=require_complete,
        )
        baseline1000_metrics = pool_reference_sufficient_statistics(
            baseline1000_rows,
            reference_name=BASELINE_REFERENCE_NAME,
            epoch=0,
        )
    except IRSTDBGCRRunContractError as error:
        raise IRSTDBGCRSelectorError(
            f"fold histories violate the frozen OOF contract: {error}"
        ) from error
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "role": ROLE,
        "source_scope": SOURCE_SCOPE,
        "selection": selection,
        "selected_epoch": selection["selected_epoch"],
        "strictly_improves_epoch0_miou": selection[
            "strictly_improves_epoch0_miou"
        ],
        "strictly_improves_epoch0_full_role_key": selection[
            "strictly_improves_epoch0_full_role_key"
        ],
        "baseline1000_internal_oof_metrics": baseline1000_metrics,
        "fold_summaries": bindings,
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "fold_manifest_sha256": fold_manifest_sha256,
        "source_split_manifest_file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": PROBABILITY_COMPARISON,
        "performance_acceptance_margin": PERFORMANCE_ACCEPTANCE_MARGIN,
        **OFFICIAL_FALSE_FLAGS,
    }
    payload["selection_sha256"] = canonical_json_sha256(payload)
    return payload


def _commit_or_validate_json(path: Path, expected: Mapping[str, object]) -> Path:
    destination = Path(path)
    if not destination.exists() and not destination.is_symlink():
        return exclusive_json(destination, expected).resolve(strict=True)
    observed = _read_json(destination, label="existing selector output")
    _require(observed == dict(expected), "existing selector output differs")
    return destination.resolve(strict=True)


def write_frozen_fold_manifest(path: Path) -> Path:
    manifest = build_frozen_fold_manifest()
    _require(
        manifest.get("schema") == FOLD_MANIFEST_SCHEMA,
        "fold manifest schema differs",
    )
    return _commit_or_validate_json(path, manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-summary", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-fold-manifest", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="diagnostic only: select from common scheduled epochs already present",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.write_fold_manifest is not None:
        _require(
            not args.fold_summary and args.output is None,
            "fold-manifest mode is exclusive",
        )
        return write_frozen_fold_manifest(args.write_fold_manifest)
    _require(args.output is not None, "--output is required for OOF selection")
    _require(len(args.fold_summary) == 3, "three --fold-summary paths are required")
    result = select_from_fold_summaries(
        tuple(Path(value) for value in args.fold_summary),
        require_complete=not bool(args.allow_partial),
    )
    return _commit_or_validate_json(args.output, result)


def main(argv: Sequence[str] | None = None) -> None:
    path = run(parse_args(argv))
    print(
        json.dumps(
            {
                "event": "irstd_bgcr_oof_selection_committed",
                "path": str(path),
                "official_test_accessed": False,
                "performance_acceptance_margin": None,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "IRSTDBGCRSelectorError",
    "BASELINE_REFERENCE_NAME",
    "SCHEMA",
    "SUMMARY_SCHEMAS",
    "select_from_fold_summaries",
    "write_frozen_fold_manifest",
]
