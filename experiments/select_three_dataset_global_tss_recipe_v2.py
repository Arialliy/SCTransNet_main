#!/usr/bin/env python3
"""Select one positive TSS weight from the three fixed img_idx/test regimes.

This selector is intentionally narrow.  It consumes exactly three datasets,
two independently selected checkpoint roles, and three positive TSS weights.
All primary points are evaluated at threshold 0.5.  The selector never creates
a third checkpoint role and never sums raw metrics with incompatible units.

Selection order:

1. reject candidates that breach a predeclared severe-degradation guard;
2. reject a candidate when Original strictly dominates both checkpoint roles
   on any dataset;
3. rank candidates independently for every dataset/role/metric cell, then
   Pareto-filter the resulting direction-unified rank vectors;
4. break remaining ties by worst-dataset mean rank, macro mean rank, the
   signed metric vote versus Original, and finally the smaller requested
   weight.

The output explicitly records that per-run protocols are matched while the
total recipe-search budgets are not (nine Final runs versus three Original
runs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


INPUT_SCHEMA = "sctransnet_three_dataset_global_tss_recipe_input/v2"
OUTPUT_SCHEMA = "sctransnet_three_dataset_global_tss_recipe_selection/v2"
LAUNCH_PLAN_SCHEMA = (
    "sctransnet_three_dataset_seed42_global_tss_launcher_v2/v1"
)

DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLES = ("best_miou", "best_pd")
CANDIDATE_LAMBDAS = (
    Decimal("0.0025"),
    Decimal("0.005"),
    Decimal("0.01"),
)
SELECTION_SPLIT = "img_idx/test"
TRAINING_SEED = data_protocol.PROTOCOL_SEED
FIXED_THRESHOLD = Decimal("0.5")

ORIGINAL_RUN_COUNT = len(DATASETS)
FINAL_SEARCH_RUN_COUNT = len(DATASETS) * len(CANDIDATE_LAMBDAS)

METRICS = (
    "miou",
    "niou",
    "matched_target_count",
    "unmatched_predicted_pixels",
    "matched_tiny_target_count",
)
HIGHER_IS_BETTER = {
    "miou": True,
    "niou": True,
    "matched_target_count": True,
    "unmatched_predicted_pixels": False,
    "matched_tiny_target_count": True,
}
QUANTIZATION_STEP = Decimal("0.0001")
SERIOUS_IOU_DROP_QUANTA = 50  # 0.005 / 0.0001

SELECTOR_SOURCE_PATH = Path(__file__).resolve()
DATA_PROTOCOL_SOURCE_PATH = Path(data_protocol.__file__).resolve()
SOURCE_PATHS = {
    "experiments/select_three_dataset_global_tss_recipe_v2.py": (
        SELECTOR_SOURCE_PATH
    ),
    "experiments/three_dataset_v2_protocol.py": DATA_PROTOCOL_SOURCE_PATH,
}


class SelectorInputError(ValueError):
    """Raised when a selector input is not the frozen three-dataset protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorInputError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    observed = set(mapping)
    required = set(expected)
    _require(
        observed == required,
        f"{label} keys differ: missing={sorted(required - observed)}, "
        f"extra={sorted(observed - required)}",
    )


def _decimal(value: Any, label: str) -> Decimal:
    _require(not isinstance(value, bool), f"{label} must be numeric, not bool")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise SelectorInputError(f"{label} must be numeric") from error
    _require(result.is_finite(), f"{label} must be finite")
    return result


def _unit_float(value: Any, label: str) -> Decimal:
    result = _decimal(value, label)
    _require(Decimal(0) <= result <= Decimal(1), f"{label} must be in [0, 1]")
    return result


def _count(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer count",
    )
    _require(value >= 0, f"{label} must be non-negative")
    return value


def quantize_iou(value: Any) -> int:
    """Return q(x)=floor(x/1e-4+0.5) using deterministic decimal arithmetic."""

    number = _unit_float(value, "IoU value")
    return int(
        (number / QUANTIZATION_STEP + Decimal("0.5")).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )


def _lambda_key(value: Decimal) -> str:
    return format(value, "f").rstrip("0").rstrip(".")


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "value": float(value),
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _input_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise SelectorInputError(
            f"source must be a regular non-symlink file: {candidate}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selector_source_sha256() -> dict[str, str]:
    """Hash the selector and its frozen data protocol at selection time."""

    return {
        relative: _file_sha256(path)
        for relative, path in SOURCE_PATHS.items()
    }


def _normalize_point(raw: Any, label: str) -> dict[str, Any]:
    point = _mapping(raw, label)
    required = {
        "threshold",
        "miou",
        "niou",
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "unmatched_predicted_pixels",
        "valid_pixel_count",
    }
    missing = required - set(point)
    _require(not missing, f"{label} lacks fields: {sorted(missing)}")

    threshold = _decimal(point["threshold"], f"{label}.threshold")
    _require(
        threshold == FIXED_THRESHOLD,
        f"{label}.threshold must be 0.5; Pd--Fa sweep points are not selector inputs",
    )
    target_count = _count(point["target_count"], f"{label}.target_count")
    matched = _count(
        point["matched_target_count"], f"{label}.matched_target_count"
    )
    _require(matched <= target_count, f"{label} matched targets exceed target count")
    tiny_count = _count(point["tiny_target_count"], f"{label}.tiny_target_count")
    tiny_value = point["matched_tiny_target_count"]
    if tiny_count == 0:
        _require(
            tiny_value is None,
            f"{label}.matched_tiny_target_count must be null when tiny target count is zero",
        )
        matched_tiny: int | None = None
    else:
        matched_tiny = _count(
            tiny_value,
            f"{label}.matched_tiny_target_count",
        )
        _require(
            matched_tiny <= tiny_count,
            f"{label} matched tiny targets exceed tiny target count",
        )
    valid_pixels = _count(
        point["valid_pixel_count"], f"{label}.valid_pixel_count"
    )
    _require(valid_pixels > 0, f"{label}.valid_pixel_count must be positive")
    unmatched_pixels = _count(
        point["unmatched_predicted_pixels"],
        f"{label}.unmatched_predicted_pixels",
    )
    _require(
        unmatched_pixels <= valid_pixels,
        f"{label} unmatched predicted pixels exceed valid pixels",
    )
    miou = _unit_float(point["miou"], f"{label}.miou")
    niou = _unit_float(point["niou"], f"{label}.niou")
    return {
        "threshold": float(threshold),
        "miou": float(miou),
        "miou_q": quantize_iou(miou),
        "niou": float(niou),
        "niou_q": quantize_iou(niou),
        "target_count": target_count,
        "matched_target_count": matched,
        "tiny_target_count": tiny_count,
        "matched_tiny_target_count": matched_tiny,
        "unmatched_predicted_pixels": unmatched_pixels,
        "valid_pixel_count": valid_pixels,
    }


def _validate_point_invariants(
    dataset: str,
    points: Sequence[Mapping[str, Any]],
) -> None:
    for field in ("target_count", "tiny_target_count", "valid_pixel_count"):
        values = {point[field] for point in points}
        _require(
            len(values) == 1,
            f"{dataset} {field} differs across Original/Final role summaries: "
            f"{sorted(values)}",
        )


def validate_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the frozen selector input schema."""

    root = _mapping(payload, "input")
    _require(root.get("schema") == INPUT_SCHEMA, f"schema must be {INPUT_SCHEMA!r}")
    _require(
        root.get("selection_split") == SELECTION_SPLIT,
        "selection_split must be exactly 'img_idx/test'",
    )
    _require(root.get("test_selected") is True, "test_selected must be true")
    _require(root.get("training_seed") == TRAINING_SEED, "training_seed must be 42")
    _require(
        _decimal(root.get("threshold"), "threshold") == FIXED_THRESHOLD,
        "threshold must be exactly 0.5",
    )
    roles = root.get("checkpoint_roles")
    _require(
        isinstance(roles, list) and tuple(roles) == CHECKPOINT_ROLES,
        "checkpoint_roles must be exactly ['best_miou', 'best_pd']",
    )
    lambda_values = root.get("candidate_lambdas")
    _require(isinstance(lambda_values, list), "candidate_lambdas must be a list")
    normalized_lambdas = tuple(
        _decimal(value, f"candidate_lambdas[{index}]")
        for index, value in enumerate(lambda_values)
    )
    _require(
        normalized_lambdas == CANDIDATE_LAMBDAS,
        "candidate_lambdas must be exactly [0.0025, 0.005, 0.01]",
    )

    datasets = _mapping(root.get("datasets"), "datasets")
    _exact_keys(datasets, DATASETS, "datasets")
    normalized_datasets: dict[str, Any] = {}
    expected_lambda_keys = tuple(_lambda_key(value) for value in CANDIDATE_LAMBDAS)
    for dataset in DATASETS:
        dataset_raw = _mapping(datasets[dataset], f"datasets.{dataset}")
        _require(
            dataset_raw.get("selection_split") == SELECTION_SPLIT,
            f"datasets.{dataset}.selection_split must be 'img_idx/test'",
        )
        digest = dataset_raw.get("img_idx_test_sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"datasets.{dataset}.img_idx_test_sha256 must be lowercase SHA-256",
        )
        expected_split = data_protocol.EXPECTED_SPLITS[dataset]["test"]
        _require(
            digest == expected_split["file_sha256"],
            f"datasets.{dataset}.img_idx_test_sha256 does not match the frozen "
            "img_idx/test file byte SHA-256",
        )
        ordered_digest = dataset_raw.get("img_idx_test_ordered_ids_sha256")
        _require(
            isinstance(ordered_digest, str)
            and len(ordered_digest) == 64
            and all(
                character in "0123456789abcdef"
                for character in ordered_digest
            ),
            f"datasets.{dataset}.img_idx_test_ordered_ids_sha256 must be "
            "lowercase SHA-256",
        )
        _require(
            ordered_digest == expected_split["ordered_ids_sha256"],
            f"datasets.{dataset}.img_idx_test_ordered_ids_sha256 does not "
            "match the frozen ordered img_idx/test ID-sequence SHA-256",
        )
        original_raw = _mapping(
            dataset_raw.get("original"), f"datasets.{dataset}.original"
        )
        _exact_keys(original_raw, CHECKPOINT_ROLES, f"datasets.{dataset}.original")
        final_raw = _mapping(
            dataset_raw.get("final_candidates"),
            f"datasets.{dataset}.final_candidates",
        )
        _exact_keys(
            final_raw,
            expected_lambda_keys,
            f"datasets.{dataset}.final_candidates",
        )
        original = {
            role: _normalize_point(
                original_raw[role], f"datasets.{dataset}.original.{role}"
            )
            for role in CHECKPOINT_ROLES
        }
        final: dict[str, dict[str, dict[str, Any]]] = {}
        for lambda_key in expected_lambda_keys:
            role_raw = _mapping(
                final_raw[lambda_key],
                f"datasets.{dataset}.final_candidates.{lambda_key}",
            )
            _exact_keys(
                role_raw,
                CHECKPOINT_ROLES,
                f"datasets.{dataset}.final_candidates.{lambda_key}",
            )
            final[lambda_key] = {
                role: _normalize_point(
                    role_raw[role],
                    f"datasets.{dataset}.final_candidates.{lambda_key}.{role}",
                )
                for role in CHECKPOINT_ROLES
            }
        all_points = list(original.values()) + [
            point
            for role_points in final.values()
            for point in role_points.values()
        ]
        _validate_point_invariants(dataset, all_points)
        normalized_datasets[dataset] = {
            "selection_split": SELECTION_SPLIT,
            "img_idx_test_sha256": digest,
            "img_idx_test_ordered_ids_sha256": ordered_digest,
            "original": original,
            "final_candidates": final,
        }
    return {
        "schema": INPUT_SCHEMA,
        "selection_split": SELECTION_SPLIT,
        "test_selected": True,
        "training_seed": TRAINING_SEED,
        "threshold": float(FIXED_THRESHOLD),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "candidate_lambdas": [float(value) for value in CANDIDATE_LAMBDAS],
        "datasets": normalized_datasets,
    }


def _metric_value(point: Mapping[str, Any], metric: str) -> int | None:
    if metric == "miou":
        return int(point["miou_q"])
    if metric == "niou":
        return int(point["niou_q"])
    value = point[metric]
    return None if value is None else int(value)


def _average_ranks(
    values: Mapping[str, int],
    *,
    higher_is_better: bool,
) -> dict[str, Fraction]:
    """Return one-based average ranks with deterministic exact ties."""

    ordered = sorted(
        values.items(),
        key=lambda item: ((-item[1]) if higher_is_better else item[1], item[0]),
    )
    ranks: dict[str, Fraction] = {}
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[index][1]:
            stop += 1
        # Positions are one-based.  Average of [index+1, ..., stop].
        average = Fraction((index + 1) + stop, 2)
        for key, _ in ordered[index:stop]:
            ranks[key] = average
        index = stop
    return ranks


def _compare(final: int, original: int, *, higher_is_better: bool) -> int:
    if final == original:
        return 0
    if higher_is_better:
        return 1 if final > original else -1
    return 1 if final < original else -1


def _original_strictly_dominates(
    original: Mapping[str, Any],
    final: Mapping[str, Any],
) -> bool:
    comparisons: list[int] = []
    for metric in METRICS:
        original_value = _metric_value(original, metric)
        final_value = _metric_value(final, metric)
        if original_value is None:
            _require(
                final_value is None,
                f"tiny metric availability differs between Original and Final for {metric}",
            )
            continue
        _require(final_value is not None, f"Final lacks comparable metric {metric}")
        # From Final's perspective: -1 means Original is better.
        comparisons.append(
            _compare(
                final_value,
                original_value,
                higher_is_better=HIGHER_IS_BETTER[metric],
            )
        )
    return bool(comparisons) and all(value <= 0 for value in comparisons) and any(
        value < 0 for value in comparisons
    )


def _severe_degradation_violations(
    dataset: str,
    role: str,
    original: Mapping[str, Any],
    final: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prefix = {"dataset": dataset, "checkpoint_role": role}
    violations: list[dict[str, Any]] = []
    matched_drop = int(original["matched_target_count"]) - int(
        final["matched_target_count"]
    )
    if matched_drop >= 2:
        violations.append(
            {
                **prefix,
                "rule": "matched_target_drop_at_least_2",
                "original": original["matched_target_count"],
                "final": final["matched_target_count"],
            }
        )
    original_tiny = original["matched_tiny_target_count"]
    final_tiny = final["matched_tiny_target_count"]
    if original_tiny is not None:
        _require(final_tiny is not None, "tiny metric availability differs")
        if int(original_tiny) - int(final_tiny) >= 2:
            violations.append(
                {
                    **prefix,
                    "rule": "matched_tiny_target_drop_at_least_2",
                    "original": original_tiny,
                    "final": final_tiny,
                }
            )
    for metric in ("miou", "niou"):
        original_q = int(original[f"{metric}_q"])
        final_q = int(final[f"{metric}_q"])
        if original_q - final_q >= SERIOUS_IOU_DROP_QUANTA:
            violations.append(
                {
                    **prefix,
                    "rule": f"{metric}_drop_at_least_0.005",
                    "original_q": original_q,
                    "final_q": final_q,
                    "drop_quanta": original_q - final_q,
                }
            )
    original_fa = int(original["unmatched_predicted_pixels"])
    final_fa = int(final["unmatched_predicted_pixels"])
    fa_trigger = (original_fa == 0 and final_fa > 0) or (
        original_fa > 0 and final_fa * 4 > original_fa * 5
    )
    matched_gain = int(final["matched_target_count"]) - int(
        original["matched_target_count"]
    )
    if fa_trigger and matched_gain < 2:
        violations.append(
            {
                **prefix,
                "rule": "fa_increase_over_25_percent_without_2_matched_gain",
                "original_unmatched_predicted_pixels": original_fa,
                "final_unmatched_predicted_pixels": final_fa,
                "matched_target_gain": matched_gain,
            }
        )
    return violations


def _rank_vectors(normalized: Mapping[str, Any]) -> dict[str, Any]:
    lambda_keys = tuple(_lambda_key(value) for value in CANDIDATE_LAMBDAS)
    cells: dict[str, dict[str, Fraction]] = {key: {} for key in lambda_keys}
    nested: dict[str, dict[str, dict[str, dict[str, Fraction]]]] = {
        key: {} for key in lambda_keys
    }
    for dataset in DATASETS:
        dataset_record = normalized["datasets"][dataset]
        for role in CHECKPOINT_ROLES:
            for metric in METRICS:
                values: dict[str, int] = {}
                for lambda_key in lambda_keys:
                    value = _metric_value(
                        dataset_record["final_candidates"][lambda_key][role],
                        metric,
                    )
                    if value is not None:
                        values[lambda_key] = value
                if not values:
                    continue
                _require(
                    len(values) == len(lambda_keys),
                    f"{dataset}/{role}/{metric} is only partially available",
                )
                ranks = _average_ranks(
                    values,
                    higher_is_better=HIGHER_IS_BETTER[metric],
                )
                cell_key = f"{dataset}/{role}/{metric}"
                for lambda_key, rank in ranks.items():
                    cells[lambda_key][cell_key] = rank
                    nested[lambda_key].setdefault(dataset, {}).setdefault(role, {})[
                        metric
                    ] = rank
    return {"cells": cells, "nested": nested}


def _pareto_dominates(
    left: Mapping[str, Fraction],
    right: Mapping[str, Fraction],
) -> bool:
    _require(set(left) == set(right), "Pareto rank vectors have different cells")
    return all(left[key] <= right[key] for key in left) and any(
        left[key] < right[key] for key in left
    )


def select_global_recipe(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``payload`` and return the deterministic recipe decision."""

    normalized = validate_input(payload)
    lambda_keys = tuple(_lambda_key(value) for value in CANDIDATE_LAMBDAS)
    rank_data = _rank_vectors(normalized)
    candidate_records: dict[str, dict[str, Any]] = {}

    for lambda_key in lambda_keys:
        severe: list[dict[str, Any]] = []
        original_dominance: dict[str, dict[str, bool]] = {}
        votes: dict[str, dict[str, dict[str, int]]] = {}
        dataset_votes: dict[str, int] = {}
        role_rank_values: dict[str, dict[str, Fraction]] = {}
        dataset_rank_values: dict[str, Fraction] = {}

        for dataset in DATASETS:
            dataset_record = normalized["datasets"][dataset]
            original_dominance[dataset] = {}
            votes[dataset] = {}
            role_rank_values[dataset] = {}
            for role in CHECKPOINT_ROLES:
                original = dataset_record["original"][role]
                final = dataset_record["final_candidates"][lambda_key][role]
                severe.extend(
                    _severe_degradation_violations(
                        dataset, role, original, final
                    )
                )
                original_dominance[dataset][role] = _original_strictly_dominates(
                    original, final
                )
                role_votes: dict[str, int] = {}
                for metric in METRICS:
                    original_value = _metric_value(original, metric)
                    final_value = _metric_value(final, metric)
                    if original_value is None:
                        continue
                    _require(final_value is not None, f"Final lacks {metric}")
                    role_votes[metric] = _compare(
                        final_value,
                        original_value,
                        higher_is_better=HIGHER_IS_BETTER[metric],
                    )
                votes[dataset][role] = role_votes
                metric_ranks = rank_data["nested"][lambda_key][dataset][role]
                role_rank_values[dataset][role] = sum(
                    metric_ranks.values(), Fraction(0)
                ) / len(metric_ranks)
            dataset_votes[dataset] = sum(
                value
                for role_votes in votes[dataset].values()
                for value in role_votes.values()
            )
            dataset_rank_values[dataset] = sum(
                role_rank_values[dataset].values(), Fraction(0)
            ) / len(CHECKPOINT_ROLES)

        dual_role_dominated_datasets = [
            dataset
            for dataset in DATASETS
            if all(original_dominance[dataset].values())
        ]
        gate_eligible = not severe and not dual_role_dominated_datasets
        macro_rank = sum(dataset_rank_values.values(), Fraction(0)) / len(DATASETS)
        worst_rank = max(dataset_rank_values.values())
        vote_total = sum(dataset_votes.values())
        candidate_records[lambda_key] = {
            "lambda_req": float(Decimal(lambda_key)),
            "severe_degradation_passed": not severe,
            "severe_degradation_violations": severe,
            "original_strict_dominance_by_dataset_role": original_dominance,
            "original_dual_role_dominated_datasets": dual_role_dominated_datasets,
            "gate_eligible": gate_eligible,
            "rank_vector": {
                key: _fraction_record(value)
                for key, value in sorted(rank_data["cells"][lambda_key].items())
            },
            "rank_vector_dimension": len(rank_data["cells"][lambda_key]),
            "per_dataset_role_metric_rank": {
                dataset: {
                    role: {
                        metric: _fraction_record(rank)
                        for metric, rank in rank_data["nested"][lambda_key][dataset][
                            role
                        ].items()
                    }
                    for role in CHECKPOINT_ROLES
                }
                for dataset in DATASETS
            },
            "per_dataset_role_mean_rank": {
                dataset: {
                    role: _fraction_record(role_rank_values[dataset][role])
                    for role in CHECKPOINT_ROLES
                }
                for dataset in DATASETS
            },
            "per_dataset_mean_rank": {
                dataset: _fraction_record(dataset_rank_values[dataset])
                for dataset in DATASETS
            },
            "worst_dataset_mean_rank": _fraction_record(worst_rank),
            "macro_dataset_mean_rank": _fraction_record(macro_rank),
            "per_dataset_role_vote": votes,
            "per_dataset_total_vote": dataset_votes,
            "total_vote": vote_total,
            "pareto_dominated": False,
            "pareto_dominated_by": [],
            "pareto_eligible": False,
            "_worst_rank": worst_rank,
            "_macro_rank": macro_rank,
        }

    gate_eligible_keys = [
        key for key in lambda_keys if candidate_records[key]["gate_eligible"]
    ]
    for lambda_key in gate_eligible_keys:
        dominators = [
            other
            for other in gate_eligible_keys
            if other != lambda_key
            and _pareto_dominates(
                rank_data["cells"][other], rank_data["cells"][lambda_key]
            )
        ]
        candidate_records[lambda_key]["pareto_dominated"] = bool(dominators)
        candidate_records[lambda_key]["pareto_dominated_by"] = dominators
        candidate_records[lambda_key]["pareto_eligible"] = not dominators

    survivors = [
        key
        for key in lambda_keys
        if candidate_records[key]["gate_eligible"]
        and candidate_records[key]["pareto_eligible"]
    ]
    survivors.sort(
        key=lambda key: (
            candidate_records[key]["_worst_rank"],
            candidate_records[key]["_macro_rank"],
            -candidate_records[key]["total_vote"],
            Decimal(key),
        )
    )
    selected_key = survivors[0] if survivors else None

    candidate_ranking = []
    for key in survivors:
        record = candidate_records[key]
        candidate_ranking.append(
            {
                "lambda_req": record["lambda_req"],
                "worst_dataset_mean_rank": record["worst_dataset_mean_rank"],
                "macro_dataset_mean_rank": record["macro_dataset_mean_rank"],
                "total_vote": record["total_vote"],
                "tie_break_lambda": record["lambda_req"],
            }
        )

    for record in candidate_records.values():
        del record["_worst_rank"]
        del record["_macro_rank"]

    established = selected_key is not None
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "decision": (
            "GLOBAL_POSITIVE_TSS_RECIPE_ESTABLISHED"
            if established
            else "NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED"
        ),
        "global_tss_recipe_established": established,
        "global_tss_lambda": (
            candidate_records[selected_key]["lambda_req"] if selected_key else None
        ),
        "selection_split": SELECTION_SPLIT,
        "test_selected": True,
        "threshold": float(FIXED_THRESHOLD),
        "training_seed": TRAINING_SEED,
        "datasets": list(DATASETS),
        "dataset_test_identities": {
            dataset: {
                "img_idx_test_sha256": normalized["datasets"][dataset][
                    "img_idx_test_sha256"
                ],
                "img_idx_test_ordered_ids_sha256": normalized["datasets"][
                    dataset
                ]["img_idx_test_ordered_ids_sha256"],
            }
            for dataset in DATASETS
        },
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "candidate_lambdas": [float(value) for value in CANDIDATE_LAMBDAS],
        "metrics_used": list(METRICS),
        "aggregation": {
            "method": "hierarchical_equal_weight_rank_then_rank_vector_pareto",
            "rank_population": (
                "all_three_preregistered_candidates_before_eligibility_gates"
            ),
            "dataset_equal_weight": True,
            "checkpoint_role_equal_weight": True,
            "metric_equal_weight": True,
            "raw_metric_sum_used": False,
            "miou_niou_quantization": "q=floor(x/1e-4+0.5)",
            "count_metrics": [
                "matched_target_count",
                "unmatched_predicted_pixels",
                "matched_tiny_target_count",
            ],
            "pareto_space": "three_datasets_x_two_roles_x_five_metric_ranks",
            "nominal_rank_vector_dimension": (
                len(DATASETS) * len(CHECKPOINT_ROLES) * len(METRICS)
            ),
            "pareto_direction": "lower_rank_is_better",
            "unavailable_tiny_metric_policy": (
                "omit the tiny rank/vote cell for all candidates on that dataset"
            ),
            "tie_break_order": [
                "minimum_worst_dataset_mean_rank",
                "minimum_macro_dataset_mean_rank",
                "maximum_total_signed_metric_vote_vs_original",
                "smaller_lambda_req",
            ],
        },
        "severe_degradation_gate": {
            "matched_target_drop_at_least_2": "reject",
            "matched_tiny_target_drop_at_least_2": "reject",
            "miou_drop_quanta_at_least_50": "reject",
            "niou_drop_quanta_at_least_50": "reject",
            "fa_increase_rule": (
                "if Final unmatched pixels exceed 125% of Original, or "
                "Original is zero and Final is positive, require matched gain >=2"
            ),
            "original_dual_role_strict_dominance": "reject per dataset",
        },
        "fairness_and_search_budget": {
            "per_run_protocol_matched": True,
            "total_search_budget_matched": False,
            "total_search_budget_equal": False,
            "original_training_runs": ORIGINAL_RUN_COUNT,
            "final_training_runs": FINAL_SEARCH_RUN_COUNT,
            "final_to_original_run_budget_ratio": (
                FINAL_SEARCH_RUN_COUNT / ORIGINAL_RUN_COUNT
            ),
            "original_run_count": ORIGINAL_RUN_COUNT,
            "final_search_run_count": FINAL_SEARCH_RUN_COUNT,
            "total_run_count": ORIGINAL_RUN_COUNT + FINAL_SEARCH_RUN_COUNT,
            "permitted_claim": (
                "Original and Final use matched per-run training/evaluation protocols."
            ),
            "prohibited_claim": (
                "Original and Final have equal total hyperparameter-search budgets."
            ),
        },
        "candidate_ranking": candidate_ranking,
        "candidates": candidate_records,
        "input_sha256": _input_sha256(payload),
        "source_sha256": selector_source_sha256(),
        "launch_plan_binding": {
            "provided": False,
            "validated": False,
            "required_for_preregistered_source_binding": True,
            "launch_plan_path": None,
            "launch_plan_sha256": None,
            "frozen_selector_sha256": None,
            "frozen_data_protocol_sha256": None,
        },
        "no_fabricated_results": True,
        "claim_scope": "fixed_seed42_img_idx_test_selected_recipe_search",
    }


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SelectorInputError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda token: (_ for _ in ()).throw(
            SelectorInputError(f"non-finite JSON number: {token}")
        ),
    )
    _require(isinstance(payload, dict), "input JSON root must be an object")
    return payload


def validate_launch_plan_binding(
    path: Path,
    *,
    current_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the launch plan's preregistered selector/protocol source lock."""

    candidate = Path(path)
    launch_plan_sha256 = _file_sha256(candidate)
    plan = load_json(candidate)
    _require(
        _file_sha256(candidate) == launch_plan_sha256,
        "launch plan changed while its source binding was validated",
    )
    _require(
        plan.get("schema") == LAUNCH_PLAN_SCHEMA,
        f"launch plan schema must be {LAUNCH_PLAN_SCHEMA!r}",
    )
    _require(
        plan.get("dataset_order") == list(DATASETS),
        "launch plan dataset_order differs from the selector protocol",
    )
    for field, expected in (
        ("worker_count", 12),
        ("original_run_count", ORIGINAL_RUN_COUNT),
        ("final_run_count", FINAL_SEARCH_RUN_COUNT),
        ("total_search_budget_equal", False),
        ("final_to_original_run_budget_ratio", 3.0),
    ):
        _require(
            plan.get(field) == expected,
            f"launch plan {field} must be {expected!r}",
        )
    static_inputs = _mapping(plan.get("static_inputs"), "launch_plan.static_inputs")
    training_sources = _mapping(
        static_inputs.get("training_sources"),
        "launch_plan.static_inputs.training_sources",
    )
    observed_sources = selector_source_sha256()
    if current_sources is not None:
        _require(
            dict(current_sources) == observed_sources,
            "selector source files changed after the selection result was built",
        )
    sources = observed_sources
    required = {
        "global_recipe_selector": (
            "experiments/select_three_dataset_global_tss_recipe_v2.py",
            SELECTOR_SOURCE_PATH,
        ),
        "data_protocol": (
            "experiments/three_dataset_v2_protocol.py",
            DATA_PROTOCOL_SOURCE_PATH,
        ),
    }
    frozen: dict[str, str] = {}
    for source_name, (relative, expected_path) in required.items():
        record = _mapping(
            training_sources.get(source_name),
            f"launch_plan static source {source_name}",
        )
        observed_path = record.get("path")
        _require(
            isinstance(observed_path, str)
            and Path(observed_path).resolve() == expected_path,
            f"launch plan source path differs for {source_name}",
        )
        planned_digest = record.get("sha256")
        _require(
            isinstance(planned_digest, str)
            and len(planned_digest) == 64
            and all(character in "0123456789abcdef" for character in planned_digest),
            f"launch plan source SHA-256 is invalid for {source_name}",
        )
        _require(
            planned_digest == sources[relative],
            f"launch plan frozen {source_name} SHA-256 differs from current source",
        )
        frozen[relative] = planned_digest
    return {
        "provided": True,
        "validated": True,
        "required_for_preregistered_source_binding": True,
        "launch_plan_path": str(candidate.resolve()),
        "launch_plan_sha256": launch_plan_sha256,
        "launch_plan_schema": LAUNCH_PLAN_SCHEMA,
        "frozen_selector_sha256": frozen[
            "experiments/select_three_dataset_global_tss_recipe_v2.py"
        ],
        "frozen_data_protocol_sha256": frozen[
            "experiments/three_dataset_v2_protocol.py"
        ],
        "frozen_source_sha256": frozen,
        "matches_current_source_sha256": True,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--launch-plan",
        type=Path,
        help=(
            "Optionally require and record the prepared 12-run plan's frozen "
            "selector/data-protocol SHA-256 binding."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_json(args.input)
    result = select_global_recipe(payload)
    if args.launch_plan is not None:
        result["launch_plan_binding"] = validate_launch_plan_binding(
            args.launch_plan,
            current_sources=result["source_sha256"],
        )
    atomic_write_json(args.output, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
