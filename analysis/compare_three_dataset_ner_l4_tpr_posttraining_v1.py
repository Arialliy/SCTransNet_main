#!/usr/bin/env python3
"""Read-only post-training comparison for the three NER-L4-TPR runs.

The comparator reads the checkpoint-selection metrics embedded in each
method's own ``summary.json``.  ``best_miou`` is compared only with
``best_miou`` and ``best_pd`` only with ``best_pd``; checkpoint roles are
never crossed or replaced by another method's checkpoint.

The command fails closed while any NER-L4-TPR formal run is incomplete.
``--allow-partial`` is an explicitly non-final preview: it can show the
available reference rows and completion status, but it cannot emit a final
comparison classification.  The module never loads a model, starts an
evaluator, writes into a run directory, or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "sctransnet_three_dataset_ner_l4_tpr_posttraining_comparison_v1/v1"
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
CHECKPOINT_ROLES = ("best_miou", "best_pd")
METHODS = ("ner_l4_tpr", "current_final_tss_off", "original")
SEED = 42
FORMAL_EPOCHS = 1000
FIXED_THRESHOLD = 0.5

CANDIDATE_METHOD = "ner_l4_tpr"
CURRENT_METHOD = "current_final_tss_off"
ORIGINAL_METHOD = "original"

CANDIDATE_RECIPE = "final_tss_off_ner_l4_tpr_v1"
CURRENT_RECIPE = "final_tss_off"
ORIGINAL_RECIPE = "original"

CANDIDATE_SCHEMA = "sctransnet_three_dataset_l4_tpr_tss_off_seed42_v1/v1"
CURRENT_SCHEMA = "sctransnet_three_dataset_tss_off_seed42_v1/v1"
ORIGINAL_SCHEMA = "sctransnet_three_dataset_seed42_global_tss_v2/v1"

DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "results" / "three_dataset_l4_tpr_tss_off_seed42_v1"
)
DEFAULT_CURRENT_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_ORIGINAL_ROOT = (
    REPO_ROOT / "results" / "three_dataset_seed42_global_tss_v2"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_CANDIDATE_ROOT / "comparison" / "posttraining_seed42"
)

COMPLETE_JSON_FILENAME = "posttraining_comparison_v1.json"
COMPLETE_MARKDOWN_FILENAME = "posttraining_comparison_v1.md"
PARTIAL_JSON_FILENAME = "posttraining_partial_preview_v1.json"
PARTIAL_MARKDOWN_FILENAME = "posttraining_partial_preview_v1.md"

SOURCE_PATH = Path(__file__).resolve()
FLOAT_ABS_TOLERANCE = 1e-12
INTEGER_RECOVERY_ABS_TOLERANCE = 1e-6

# The relation is Pareto-style over named cells.  It does not average metrics
# with different units and does not count a repeated Pd ratio as a second vote.
RELATION_METRICS: dict[str, bool] = {
    "matched_target_count": True,
    "component_false_positive_pixels": False,
    "background_false_positive_pixels": False,
    "miou": True,
    "niou": True,
    "matched_tiny_target_count": True,
    "pixel_precision": True,
    "pixel_recall": True,
    "pixel_f1": True,
}


class NERL4TPRPosttrainingComparisonError(ValueError):
    """An input or output violates the post-training comparison contract."""


class FormalRunsIncompleteError(NERL4TPRPosttrainingComparisonError):
    """At least one required NER-L4-TPR summary is not complete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NERL4TPRPosttrainingComparisonError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    ready = float(value)
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _optional_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _nonnegative_int(value: Any, label: str, *, string_allowed: bool = False) -> int:
    if string_allowed and isinstance(value, str):
        _require(value.isdigit(), f"{label} must be a non-negative integer string")
        return int(value)
    _require(
        not isinstance(value, bool) and isinstance(value, int),
        f"{label} must be an integer",
    )
    _require(value >= 0, f"{label} must be non-negative")
    return value


def _close(left: float, right: float, *, tolerance: float = FLOAT_ABS_TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def file_sha256(path: Path) -> str:
    ready = Path(path)
    if not ready.is_file() or ready.is_symlink():
        raise FileNotFoundError(ready)
    digest = hashlib.sha256()
    with ready.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return _canonical_sha256(unsigned)


def seal_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = dict(payload)
    ready["artifact_sha256"] = _artifact_sha256(ready)
    return ready


def validate_artifact_sha256(payload: Mapping[str, Any]) -> None:
    declared = payload.get("artifact_sha256")
    _require(
        isinstance(declared, str) and len(declared) == 64,
        "artifact_sha256 must be a SHA-256",
    )
    _require(
        declared == _artifact_sha256(payload),
        "artifact_sha256 differs from canonical content",
    )


def _run_directory(root: Path, method: str, dataset: str) -> Path:
    recipe = {
        CANDIDATE_METHOD: CANDIDATE_RECIPE,
        CURRENT_METHOD: CURRENT_RECIPE,
        ORIGINAL_METHOD: ORIGINAL_RECIPE,
    }[method]
    return Path(root) / "runs" / dataset / recipe / "seed_42"


def _method_contract(method: str) -> dict[str, Any]:
    return {
        CANDIDATE_METHOD: {
            "schema": CANDIDATE_SCHEMA,
            "summary_method": "final",
            "recipe_id": CANDIDATE_RECIPE,
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
        },
        CURRENT_METHOD: {
            "schema": CURRENT_SCHEMA,
            "summary_method": "final",
            "recipe_id": CURRENT_RECIPE,
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
        },
        ORIGINAL_METHOD: {
            "schema": ORIGINAL_SCHEMA,
            "summary_method": "original",
            "recipe_id": "original_no_tss",
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
        },
    }[method]


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    ready = Path(path)
    if not ready.is_file() or ready.is_symlink():
        raise FileNotFoundError(ready)
    raw = ready.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise NERL4TPRPosttrainingComparisonError(
            f"{label} is not valid JSON: {ready}"
        ) from error
    _require(isinstance(payload, dict), f"{label} must contain an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _normalize_metrics(metrics_raw: Any, label: str) -> dict[str, Any]:
    metrics = _mapping(metrics_raw, f"{label}.metrics")
    valid_pixels = _nonnegative_int(
        metrics.get("valid_pixel_count"), f"{label}.valid_pixel_count"
    )
    _require(valid_pixels > 0, f"{label}.valid_pixel_count must be positive")
    target_count = _nonnegative_int(
        metrics.get("target_count"), f"{label}.target_count"
    )
    matched_count = _nonnegative_int(
        metrics.get("matched_target_count"), f"{label}.matched_target_count"
    )
    _require(target_count > 0, f"{label}.target_count must be positive")
    _require(matched_count <= target_count, f"{label} matched targets exceed total")

    tiny_count = _nonnegative_int(
        metrics.get("tiny_target_count"),
        f"{label}.tiny_target_count",
        string_allowed=True,
    )
    matched_tiny = _nonnegative_int(
        metrics.get("matched_tiny_target_count"),
        f"{label}.matched_tiny_target_count",
        string_allowed=True,
    )
    _require(matched_tiny <= tiny_count, f"{label} matched tiny targets exceed total")

    pd = _finite(metrics.get("pd"), f"{label}.pd")
    tiny_pd = _optional_finite(metrics.get("tiny_pd"), f"{label}.tiny_pd")
    fa = _finite(metrics.get("fa"), f"{label}.fa")
    miou = _finite(metrics.get("miou"), f"{label}.miou")
    niou = _finite(metrics.get("niou"), f"{label}.niou")
    for name, value in (("pd", pd), ("fa", fa), ("miou", miou), ("niou", niou)):
        _require(0.0 <= value <= 1.0, f"{label}.{name} must be in [0, 1]")
    _require(
        _close(pd, matched_count / target_count),
        f"{label}.pd differs from matched/total counts",
    )
    if tiny_count == 0:
        _require(
            matched_tiny == 0 and tiny_pd is None,
            f"{label}.tiny_pd must be unavailable when no tiny targets exist",
        )
    else:
        _require(tiny_pd is not None, f"{label}.tiny_pd is missing")
        _require(
            _close(float(tiny_pd), matched_tiny / tiny_count),
            f"{label}.tiny_pd differs from matched/total tiny counts",
        )

    raw_component_fp = fa * valid_pixels
    component_fp = int(round(raw_component_fp))
    _require(
        _close(
            raw_component_fp,
            float(component_fp),
            tolerance=INTEGER_RECOVERY_ABS_TOLERANCE,
        ),
        f"{label}.fa does not encode an integer component-FP numerator",
    )
    explicit_component = metrics.get("unmatched_predicted_pixels")
    if explicit_component is not None:
        _require(
            _nonnegative_int(
                explicit_component, f"{label}.unmatched_predicted_pixels"
            )
            == component_fp,
            f"{label}.unmatched_predicted_pixels differs from Fa numerator",
        )

    predicted_objects = _nonnegative_int(
        metrics.get("predicted_object_count"), f"{label}.predicted_object_count"
    )
    unmatched_objects = _nonnegative_int(
        metrics.get("unmatched_predicted_object_count"),
        f"{label}.unmatched_predicted_object_count",
    )
    _require(
        predicted_objects - matched_count == unmatched_objects,
        f"{label} predicted/matched/unmatched object counts differ",
    )
    false_objects_per_image = _finite(
        metrics.get("false_objects_per_image"),
        f"{label}.false_objects_per_image",
    )
    _require(false_objects_per_image >= 0.0, f"{label} false objects/image is negative")

    pixel_values: dict[str, float | None] = {}
    for field in ("pixel_precision", "pixel_recall", "pixel_f1"):
        value = _optional_finite(metrics.get(field), f"{label}.{field}")
        if value is not None:
            _require(0.0 <= value <= 1.0, f"{label}.{field} must be in [0, 1]")
        pixel_values[field] = value
    availability = [value is not None for value in pixel_values.values()]
    _require(
        all(availability) or not any(availability),
        f"{label} pixel precision/recall/F1 must be all present or all absent",
    )

    explicit_background = metrics.get("false_positive_pixels")
    background_fp = None
    background_source = "pending_cross_checked_reconstruction"
    if explicit_background is not None:
        background_fp = _nonnegative_int(
            explicit_background, f"{label}.false_positive_pixels"
        )
        background_source = "summary.false_positive_pixels"

    return {
        "threshold": FIXED_THRESHOLD,
        "pd": pd,
        "target_count": target_count,
        "matched_target_count": matched_count,
        "fa": fa,
        "miou": miou,
        "niou": niou,
        "tiny_pd": tiny_pd,
        "tiny_target_count": tiny_count,
        "matched_tiny_target_count": matched_tiny,
        "component_false_positive_pixels": component_fp,
        "component_fp_source": "exact_round(fa * valid_pixel_count)",
        "background_false_positive_pixels": background_fp,
        "background_fp_source": background_source,
        "predicted_object_count": predicted_objects,
        "unmatched_predicted_object_count": unmatched_objects,
        "false_objects_per_image": false_objects_per_image,
        **pixel_values,
        "pixel_metrics_available": all(availability),
        "valid_pixel_count": valid_pixels,
    }


def _validate_checkpoint_record(
    *,
    run_dir: Path,
    role: str,
    role_record: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
    verify_checkpoint_files: bool,
) -> dict[str, Any]:
    expected_path = (run_dir / "checkpoints" / f"{role}.pth.tar").resolve()
    paths: list[Path] = []
    for label, raw in (
        ("role path", role_record.get("path")),
        ("checkpoint path", checkpoint_record.get("path")),
    ):
        _require(isinstance(raw, str) and bool(raw), f"{role} {label} is invalid")
        paths.append(Path(raw).resolve())
    _require(
        paths[0] == expected_path and paths[1] == expected_path,
        f"{role} checkpoint path differs from its own run directory",
    )
    digest = checkpoint_record.get("sha256")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{role} checkpoint SHA-256 is invalid",
    )
    size = _nonnegative_int(checkpoint_record.get("bytes"), f"{role} checkpoint bytes")
    _require(size > 0, f"{role} checkpoint must be non-empty")
    if verify_checkpoint_files:
        _require(expected_path.is_file() and not expected_path.is_symlink(), f"missing checkpoint: {expected_path}")
        _require(expected_path.stat().st_size == size, f"checkpoint byte count differs: {expected_path}")
        _require(file_sha256(expected_path) == digest, f"checkpoint SHA differs: {expected_path}")
    return {"path": str(expected_path), "sha256": digest, "bytes": size}


def load_complete_summary(
    *,
    root: Path,
    method: str,
    dataset: str,
    verify_checkpoint_files: bool = True,
) -> dict[str, Any]:
    run_dir = _run_directory(root, method, dataset).resolve()
    summary_path = run_dir / "summary.json"
    summary, summary_sha = _load_json(summary_path, f"{method}/{dataset} summary")
    contract = _method_contract(method)
    for field, expected in (
        ("schema", contract["schema"]),
        ("status", "complete"),
        ("dataset", dataset),
        ("method", contract["summary_method"]),
        ("epochs", FORMAL_EPOCHS),
        ("seed", SEED),
        ("checkpoint_roles", list(CHECKPOINT_ROLES)),
        ("requested_tss_weight", contract["requested_tss_weight"]),
        ("test_selected", True),
        ("selection_is_optimistic", True),
    ):
        _require(
            summary.get(field) == expected,
            f"{method}/{dataset} summary {field} differs: "
            f"{summary.get(field)!r} != {expected!r}",
        )
    if method == CANDIDATE_METHOD:
        _require(
            summary.get("planned_total_epochs") == FORMAL_EPOCHS,
            f"{method}/{dataset} planned_total_epochs differs",
        )
    recipe = _mapping(summary.get("recipe"), f"{method}/{dataset}.recipe")
    for field, expected in (
        ("recipe_id", contract["recipe_id"]),
        ("requested_tss_weight", contract["requested_tss_weight"]),
        ("tss_enabled", contract["tss_enabled"]),
    ):
        _require(
            recipe.get(field) == expected,
            f"{method}/{dataset} recipe {field} differs",
        )

    checkpoints = _mapping(summary.get("checkpoints"), f"{method}/{dataset}.checkpoints")
    _require(
        set(checkpoints) == set(CHECKPOINT_ROLES),
        f"{method}/{dataset} checkpoint roles differ",
    )
    roles: dict[str, Any] = {}
    for role in CHECKPOINT_ROLES:
        role_record = _mapping(summary.get(role), f"{method}/{dataset}.{role}")
        epoch = _nonnegative_int(role_record.get("epoch"), f"{method}/{dataset}/{role}.epoch")
        _require(1 <= epoch <= FORMAL_EPOCHS, f"{method}/{dataset}/{role} epoch differs")
        checkpoint = _validate_checkpoint_record(
            run_dir=run_dir,
            role=role,
            role_record=role_record,
            checkpoint_record=_mapping(checkpoints[role], f"{method}/{dataset}/{role} checkpoint"),
            verify_checkpoint_files=verify_checkpoint_files,
        )
        roles[role] = {
            "epoch": epoch,
            "checkpoint": checkpoint,
            "metrics": _normalize_metrics(
                role_record.get("metrics"), f"{method}/{dataset}/{role}"
            ),
        }
    return {
        "method": method,
        "dataset": dataset,
        "run_directory": str(run_dir),
        "summary": {"path": str(summary_path), "sha256": summary_sha},
        "roles": roles,
    }


def _candidate_completion_record(root: Path, dataset: str) -> dict[str, Any]:
    run_dir = _run_directory(root, CANDIDATE_METHOD, dataset).resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            "dataset": dataset,
            "complete": False,
            "reason": "summary_missing",
            "run_directory": str(run_dir),
            "summary_path": str(summary_path),
        }
    if not summary_path.is_file() or summary_path.is_symlink():
        raise NERL4TPRPosttrainingComparisonError(
            f"candidate summary path is not a regular file: {summary_path}"
        )
    return {
        "dataset": dataset,
        "complete": True,
        "reason": None,
        "run_directory": str(run_dir),
        "summary_path": str(summary_path),
    }


def load_inputs(
    *,
    candidate_root: Path,
    current_root: Path,
    original_root: Path,
    allow_partial: bool,
    verify_checkpoint_files: bool = True,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    completion = [
        _candidate_completion_record(candidate_root, dataset) for dataset in DATASETS
    ]
    missing = [record for record in completion if not record["complete"]]
    if missing and not allow_partial:
        datasets = ", ".join(record["dataset"] for record in missing)
        raise FormalRunsIncompleteError(
            "NER-L4-TPR formal runs are incomplete; no final comparison was "
            f"created. Missing complete summary: {datasets}"
        )

    for dataset in DATASETS:
        loaded[CURRENT_METHOD][dataset] = load_complete_summary(
            root=current_root,
            method=CURRENT_METHOD,
            dataset=dataset,
            verify_checkpoint_files=verify_checkpoint_files,
        )
        loaded[ORIGINAL_METHOD][dataset] = load_complete_summary(
            root=original_root,
            method=ORIGINAL_METHOD,
            dataset=dataset,
            verify_checkpoint_files=verify_checkpoint_files,
        )
        if not any(
            record["dataset"] == dataset and not record["complete"]
            for record in completion
        ):
            loaded[CANDIDATE_METHOD][dataset] = load_complete_summary(
                root=candidate_root,
                method=CANDIDATE_METHOD,
                dataset=dataset,
                verify_checkpoint_files=verify_checkpoint_files,
            )
    return loaded, completion


def _reference_foreground_pixel_count(
    loaded: Mapping[str, Mapping[str, Mapping[str, Any]]], dataset: str
) -> int:
    denominators: list[int] = []
    valid_counts: set[int] = set()
    for method in (CURRENT_METHOD, ORIGINAL_METHOD):
        record = loaded[method][dataset]
        for role in CHECKPOINT_ROLES:
            point = record["roles"][role]["metrics"]
            valid = int(point["valid_pixel_count"])
            valid_counts.add(valid)
            recall = point.get("pixel_recall")
            _require(
                recall is not None,
                f"{method}/{dataset}/{role} pixel_recall is required to bind background FP",
            )
            reduced = Fraction(float(recall)).limit_denominator(valid)
            _require(
                _close(float(reduced), float(recall)),
                f"{method}/{dataset}/{role} recall does not recover an exact rational count",
            )
            denominators.append(reduced.denominator)
    _require(len(valid_counts) == 1, f"{dataset} reference valid-pixel counts differ")
    foreground = math.lcm(*denominators)
    valid = next(iter(valid_counts))
    _require(0 < foreground <= valid, f"{dataset} foreground-pixel count is invalid")
    return foreground


def _bind_background_fp(
    point_raw: Mapping[str, Any], *, foreground_pixels: int, label: str
) -> dict[str, Any]:
    point = dict(point_raw)
    precision = point.get("pixel_precision")
    recall = point.get("pixel_recall")
    f1 = point.get("pixel_f1")
    explicit = point.get("background_false_positive_pixels")
    if precision is None or recall is None or f1 is None:
        _require(
            explicit is not None,
            f"{label} lacks both pixel metrics and explicit background FP",
        )
        point["background_fp_source"] = "summary.false_positive_pixels"
        return point

    true_positive_raw = float(recall) * foreground_pixels
    true_positive = int(round(true_positive_raw))
    _require(
        _close(
            true_positive_raw,
            float(true_positive),
            tolerance=INTEGER_RECOVERY_ABS_TOLERANCE,
        ),
        f"{label} recall does not encode an integer true-positive pixel count",
    )
    _require(true_positive > 0, f"{label} cannot reconstruct background FP from zero TP")
    _require(float(precision) > 0.0, f"{label} precision must be positive")
    predicted_positive_raw = true_positive / float(precision)
    predicted_positive = int(round(predicted_positive_raw))
    _require(
        _close(
            predicted_positive_raw,
            float(predicted_positive),
            tolerance=INTEGER_RECOVERY_ABS_TOLERANCE,
        ),
        f"{label} precision does not encode an integer predicted-positive count",
    )
    background_fp = predicted_positive - true_positive
    _require(background_fp >= 0, f"{label} reconstructed background FP is negative")
    if explicit is not None:
        _require(
            int(explicit) == background_fp,
            f"{label} explicit and reconstructed background FP differ",
        )
        source = "summary.false_positive_pixels+pixel_metric_cross_check"
    else:
        source = (
            "exact_reconstruction_from_reference_gt_foreground_count_and_"
            "pixel_precision_recall"
        )

    expected_recall = true_positive / foreground_pixels
    expected_precision = true_positive / predicted_positive
    expected_miou = true_positive / (foreground_pixels + background_fp)
    denominator = expected_precision + expected_recall
    expected_f1 = (
        0.0 if denominator == 0.0 else 2.0 * expected_precision * expected_recall / denominator
    )
    for name, observed, expected in (
        ("pixel_recall", float(recall), expected_recall),
        ("pixel_precision", float(precision), expected_precision),
        ("miou", float(point["miou"]), expected_miou),
        ("pixel_f1", float(f1), expected_f1),
    ):
        _require(
            _close(observed, expected),
            f"{label}.{name} fails integer confusion-count cross-check",
        )
    point["foreground_positive_pixels"] = foreground_pixels
    point["true_positive_pixels"] = true_positive
    point["background_false_positive_pixels"] = background_fp
    point["background_fp_source"] = source
    return point


def bind_all_points(
    loaded: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    bound: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in METHODS}
    derivation: dict[str, Any] = {}
    for dataset in DATASETS:
        foreground = _reference_foreground_pixel_count(loaded, dataset)
        reference_valid = loaded[CURRENT_METHOD][dataset]["roles"][CHECKPOINT_ROLES[0]][
            "metrics"
        ]["valid_pixel_count"]
        derivation[dataset] = {
            "foreground_positive_pixels": foreground,
            "valid_pixel_count": reference_valid,
            "method": (
                "LCM of reduced pixel-recall denominators across current TSS-off "
                "and Original best_miou/best_pd, then integer confusion-matrix "
                "cross-check for every reported point"
            ),
        }
        for method in METHODS:
            record = loaded[method].get(dataset)
            if record is None:
                continue
            ready_record = {
                key: value for key, value in record.items() if key != "roles"
            }
            ready_roles: dict[str, Any] = {}
            for role in CHECKPOINT_ROLES:
                role_record = record["roles"][role]
                point = _bind_background_fp(
                    role_record["metrics"],
                    foreground_pixels=foreground,
                    label=f"{method}/{dataset}/{role}",
                )
                _require(
                    point["valid_pixel_count"] == reference_valid,
                    f"{method}/{dataset}/{role} valid-pixel count differs",
                )
                ready_roles[role] = {
                    "epoch": role_record["epoch"],
                    "checkpoint": dict(role_record["checkpoint"]),
                    "metrics": point,
                }
            ready_record["roles"] = ready_roles
            bound[method][dataset] = ready_record
    return bound, derivation


def _compare_value(left: Any, right: Any, *, higher_is_better: bool) -> int | None:
    if left is None or right is None:
        return None
    if isinstance(left, int) and isinstance(right, int):
        raw = (left > right) - (left < right)
    else:
        left_value = float(left)
        right_value = float(right)
        if _close(left_value, right_value):
            raw = 0
        else:
            raw = 1 if left_value > right_value else -1
    return raw if higher_is_better else -raw


def _relation_from_signs(signs: Sequence[int]) -> str:
    _require(bool(signs), "a pairwise relation requires at least one metric cell")
    any_better = any(sign > 0 for sign in signs)
    any_worse = any(sign < 0 for sign in signs)
    if any_better and not any_worse:
        return "candidate_dominates"
    if any_worse and not any_better:
        return "candidate_dominated"
    if not any_better and not any_worse:
        return "equal"
    return "incomparable"


def compare_points(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    signs: list[int] = []
    for metric, higher_is_better in RELATION_METRICS.items():
        candidate_value = candidate.get(metric)
        reference_value = reference.get(metric)
        sign = _compare_value(
            candidate_value, reference_value, higher_is_better=higher_is_better
        )
        if sign is None:
            cells[metric] = {
                "candidate": candidate_value,
                "reference": reference_value,
                "candidate_minus_reference": None,
                "higher_is_better": higher_is_better,
                "comparison_from_candidate_perspective": None,
                "available": False,
            }
            continue
        signs.append(sign)
        cells[metric] = {
            "candidate": candidate_value,
            "reference": reference_value,
            "candidate_minus_reference": candidate_value - reference_value,
            "higher_is_better": higher_is_better,
            "comparison_from_candidate_perspective": sign,
            "available": True,
        }
    return {
        "relation": _relation_from_signs(signs),
        "candidate_better_cell_count": sum(sign > 0 for sign in signs),
        "equal_cell_count": sum(sign == 0 for sign in signs),
        "candidate_worse_cell_count": sum(sign < 0 for sign in signs),
        "available_cell_count": len(signs),
        "cells": cells,
    }


def _aggregate_pairwise(
    per_dataset: Mapping[str, Any], reference_method: str
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    signs: list[int] = []
    unit_relations: dict[str, str] = {}
    for dataset in DATASETS:
        dataset_record = per_dataset.get(dataset)
        if dataset_record is None:
            continue
        for role in CHECKPOINT_ROLES:
            comparison = dataset_record["checkpoint_roles"][role]["comparisons"][
                f"candidate_vs_{reference_method}"
            ]
            unit_key = f"{dataset}/{role}"
            unit_relations[unit_key] = comparison["relation"]
            for metric, cell in comparison["cells"].items():
                if not cell["available"]:
                    continue
                key = f"{unit_key}/{metric}"
                cells[key] = cell
                signs.append(int(cell["comparison_from_candidate_perspective"]))
    if not signs:
        return {
            "relation": "not_available",
            "candidate_better_cell_count": 0,
            "equal_cell_count": 0,
            "candidate_worse_cell_count": 0,
            "available_cell_count": 0,
            "unit_relations": unit_relations,
            "cells": cells,
        }
    return {
        "relation": _relation_from_signs(signs),
        "candidate_better_cell_count": sum(sign > 0 for sign in signs),
        "equal_cell_count": sum(sign == 0 for sign in signs),
        "candidate_worse_cell_count": sum(sign < 0 for sign in signs),
        "available_cell_count": len(signs),
        "unit_relations": unit_relations,
        "cells": cells,
    }


def _complete_classification(candidate_vs_current: str, candidate_vs_original: str) -> str:
    if candidate_vs_current == "candidate_dominated":
        return "NER_L4_TPR_DOMINATED_BY_CURRENT_TSS_OFF_REPORTED_VECTOR"
    if candidate_vs_current in ("candidate_dominates", "equal") and candidate_vs_original in (
        "candidate_dominates",
        "equal",
    ):
        if candidate_vs_current == "equal" and candidate_vs_original == "equal":
            return "NER_L4_TPR_EQUAL_TO_BOTH_REFERENCES_REPORTED_VECTOR"
        return "NER_L4_TPR_NON_INFERIOR_TO_BOTH_REFERENCES_REPORTED_VECTOR"
    return "NER_L4_TPR_MIXED_TRADEOFF_REPORTED_VECTOR"


def build_comparison(
    loaded_raw: Mapping[str, Mapping[str, Mapping[str, Any]]],
    completion: Sequence[Mapping[str, Any]],
    *,
    roots: Mapping[str, Path],
    allow_partial: bool,
) -> dict[str, Any]:
    complete = all(bool(record.get("complete")) for record in completion)
    if not complete and not allow_partial:
        raise FormalRunsIncompleteError("formal runs are incomplete")
    loaded, foreground_derivation = bind_all_points(loaded_raw)
    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        candidate_record = loaded[CANDIDATE_METHOD].get(dataset)
        roles: dict[str, Any] = {}
        for role in CHECKPOINT_ROLES:
            methods: dict[str, Any] = {}
            for method in METHODS:
                method_record = loaded[method].get(dataset)
                if method_record is None:
                    methods[method] = None
                else:
                    methods[method] = method_record["roles"][role]
            comparisons: dict[str, Any] = {}
            if candidate_record is not None:
                candidate_point = candidate_record["roles"][role]["metrics"]
                for reference_method in (CURRENT_METHOD, ORIGINAL_METHOD):
                    comparisons[f"candidate_vs_{reference_method}"] = compare_points(
                        candidate_point,
                        loaded[reference_method][dataset]["roles"][role]["metrics"],
                    )
            else:
                for reference_method in (CURRENT_METHOD, ORIGINAL_METHOD):
                    comparisons[f"candidate_vs_{reference_method}"] = {
                        "relation": "not_available",
                        "candidate_better_cell_count": 0,
                        "equal_cell_count": 0,
                        "candidate_worse_cell_count": 0,
                        "available_cell_count": 0,
                        "cells": {},
                    }
            roles[role] = {"methods": methods, "comparisons": comparisons}
        per_dataset[dataset] = {"checkpoint_roles": roles}

    aggregate = {
        "candidate_vs_current_final_tss_off": _aggregate_pairwise(
            per_dataset, CURRENT_METHOD
        ),
        "candidate_vs_original": _aggregate_pairwise(per_dataset, ORIGINAL_METHOD),
    }
    if complete:
        classification = _complete_classification(
            aggregate["candidate_vs_current_final_tss_off"]["relation"],
            aggregate["candidate_vs_original"]["relation"],
        )
        status = "complete"
        final_made = True
    else:
        classification = "NOT_EVALUATED_INCOMPLETE_FORMAL_RUNS"
        status = "partial_preview"
        final_made = False

    bindings: dict[str, Any] = {}
    for method in METHODS:
        method_bindings: dict[str, Any] = {}
        for dataset in DATASETS:
            record = loaded[method].get(dataset)
            if record is None:
                method_bindings[dataset] = None
            else:
                method_bindings[dataset] = {
                    "run_directory": record["run_directory"],
                    "summary": record["summary"],
                    "checkpoints": {
                        role: record["roles"][role]["checkpoint"]
                        for role in CHECKPOINT_ROLES
                    },
                }
        bindings[method] = method_bindings

    output = {
        "schema": SCHEMA,
        "status": status,
        "classification": classification,
        "final_comparison_classification_made": final_made,
        "model_success_claim_made": False,
        "training_seed": SEED,
        "formal_epochs": FORMAL_EPOCHS,
        "threshold": FIXED_THRESHOLD,
        "datasets": list(DATASETS),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "checkpoint_role_policy": (
            "each method's own best_miou is compared only with best_miou; "
            "each method's own best_pd is compared only with best_pd"
        ),
        "selection_split": "img_idx/test",
        "test_selected": True,
        "selection_is_optimistic": True,
        "independent_test_confirmation": False,
        "completion": {
            "all_candidate_runs_complete": complete,
            "required_candidate_run_count": len(DATASETS),
            "complete_candidate_run_count": sum(
                bool(record.get("complete")) for record in completion
            ),
            "records": [dict(record) for record in completion],
            "allow_partial_requested": allow_partial,
            "partial_preview_cannot_make_final_classification": True,
        },
        "metric_semantics": {
            "pd": "matched_target_count / target_count; value and counts are both reported",
            "fa": "unmatched predicted-component pixels / valid pixels",
            "component_false_positive_pixels": (
                "pixels belonging to predicted connected components unmatched to a GT target"
            ),
            "background_false_positive_pixels": (
                "all thresholded predicted-foreground pixels located on GT background"
            ),
            "two_false_positive_metrics_are_not_interchangeable": True,
            "pixel_metrics": "precision, recall and F1 are reported when present in source summaries",
        },
        "relation_policy": {
            "type": "cellwise_pareto_without_scalarization",
            "metrics": dict(RELATION_METRICS),
            "weighted_sum_used": False,
            "raw_metric_sum_used": False,
            "pd_ratio_duplicated_as_a_second_relation_cell": False,
            "relation_labels": [
                "candidate_dominates",
                "candidate_dominated",
                "equal",
                "incomparable",
            ],
        },
        "foreground_pixel_count_derivation": foreground_derivation,
        "per_dataset": per_dataset,
        "aggregate_comparisons": aggregate,
        "input_roots": {key: str(Path(value).resolve()) for key, value in roots.items()},
        "input_bindings": bindings,
        "source_sha256": {str(SOURCE_PATH.relative_to(REPO_ROOT)): file_sha256(SOURCE_PATH)},
        "no_metrics_recomputed_by_model_inference": True,
        "no_checkpoint_loaded": True,
        "no_fabricated_results": True,
        "claim_scope": "three_dataset_seed42_img_idx_test_selected_checkpoint_role_matched",
    }
    return seal_artifact(output)


def validate_output_payload(payload_raw: Mapping[str, Any]) -> None:
    payload = _mapping(payload_raw, "output")
    validate_artifact_sha256(payload)
    for field, expected in (
        ("schema", SCHEMA),
        ("training_seed", SEED),
        ("formal_epochs", FORMAL_EPOCHS),
        ("threshold", FIXED_THRESHOLD),
        ("datasets", list(DATASETS)),
        ("checkpoint_roles", list(CHECKPOINT_ROLES)),
        ("test_selected", True),
        ("selection_is_optimistic", True),
        ("independent_test_confirmation", False),
        ("model_success_claim_made", False),
        ("no_checkpoint_loaded", True),
        ("no_fabricated_results", True),
    ):
        _require(payload.get(field) == expected, f"output.{field} differs")
    status = payload.get("status")
    _require(status in ("complete", "partial_preview"), "output.status differs")
    completion = _mapping(payload.get("completion"), "output.completion")
    all_complete = completion.get("all_candidate_runs_complete")
    _require(isinstance(all_complete, bool), "completion flag must be bool")
    _require(
        payload.get("final_comparison_classification_made") is all_complete,
        "final classification flag differs from completion state",
    )
    if all_complete:
        _require(status == "complete", "complete inputs require complete status")
        _require(
            payload.get("classification") != "NOT_EVALUATED_INCOMPLETE_FORMAL_RUNS",
            "complete output lacks a comparison classification",
        )
    else:
        _require(status == "partial_preview", "incomplete output must be a partial preview")
        _require(
            payload.get("classification") == "NOT_EVALUATED_INCOMPLETE_FORMAL_RUNS",
            "partial preview manufactured a final classification",
        )


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_scientific(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6e}"


def _method_label(method: str) -> str:
    return {
        CANDIDATE_METHOD: "NER-L4-TPR",
        CURRENT_METHOD: "当前 Final (TSS-off)",
        ORIGINAL_METHOD: "Original",
    }[method]


def render_markdown(payload_raw: Mapping[str, Any]) -> str:
    payload = _mapping(payload_raw, "output")
    validate_output_payload(payload)
    lines = [
        "# NER-L4-TPR 三数据集正式训练后性能比较",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 比较分类：`{payload['classification']}`",
        f"- 固定阈值：`{payload['threshold']}`",
        f"- seed / epochs：`{payload['training_seed']}` / `{payload['formal_epochs']}`",
        "- checkpoint 规则：各方法自己的 best_miou 只与 best_miou 比；best_pd 只与 best_pd 比。",
        "- 结果范围：img_idx/test 上的单 seed、test-selected 比较；不单独构成稳定性或最终模型成功主张。",
        "",
    ]
    completion = payload["completion"]
    if not completion["all_candidate_runs_complete"]:
        lines.extend(
            [
                "## 部分预览限制",
                "",
                "正式训练尚未全部完成。本文件只展示已有来源，不能据此给出最终比较裁决。",
                "",
                "| 数据集 | 完成 | 原因 | summary |",
                "| --- | --- | --- | --- |",
            ]
        )
        for record in completion["records"]:
            lines.append(
                f"| {record['dataset']} | {'是' if record['complete'] else '否'} | "
                f"{record['reason'] or '-'} | {record['summary_path']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 固定阈值 0.5 性能",
            "",
            "| 数据集 | checkpoint | 方法 | epoch | Pd（计数） | Fa | mIoU | nIoU | tiny-Pd（计数） | component FP(px) | background FP(px) | FP目标数 | Precision | Recall | F1 |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        for role in CHECKPOINT_ROLES:
            methods = payload["per_dataset"][dataset]["checkpoint_roles"][role]["methods"]
            for method in METHODS:
                record = methods[method]
                if record is None:
                    lines.append(
                        f"| {dataset} | {role} | {_method_label(method)} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
                    )
                    continue
                point = record["metrics"]
                tiny = (
                    "N/A"
                    if point["tiny_pd"] is None
                    else f"{point['tiny_pd']:.6f} ({point['matched_tiny_target_count']}/{point['tiny_target_count']})"
                )
                lines.append(
                    f"| {dataset} | {role} | {_method_label(method)} | {record['epoch']} | "
                    f"{point['pd']:.6f} ({point['matched_target_count']}/{point['target_count']}) | "
                    f"{_format_scientific(point['fa'])} | {_format_float(point['miou'])} | "
                    f"{_format_float(point['niou'])} | {tiny} | "
                    f"{point['component_false_positive_pixels']} | "
                    f"{point['background_false_positive_pixels']} | "
                    f"{point['unmatched_predicted_object_count']} | "
                    f"{_format_float(point['pixel_precision'])} | "
                    f"{_format_float(point['pixel_recall'])} | "
                    f"{_format_float(point['pixel_f1'])} |"
                )
    lines.append("")
    lines.extend(
        [
            "## 同角色逐项关系",
            "",
            "`candidate_dominates` 表示候选在该行所有可用指标上不差且至少一项更好；`incomparable` 表示存在性能权衡。这里不求不同量纲指标之和。",
            "",
            "| 数据集 | checkpoint | 对照 | 关系 | 更好项 | 相同项 | 更差项 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        for role in CHECKPOINT_ROLES:
            comparisons = payload["per_dataset"][dataset]["checkpoint_roles"][role]["comparisons"]
            for reference, key in (
                ("当前 Final (TSS-off)", f"candidate_vs_{CURRENT_METHOD}"),
                ("Original", f"candidate_vs_{ORIGINAL_METHOD}"),
            ):
                row = comparisons[key]
                lines.append(
                    f"| {dataset} | {role} | {reference} | {row['relation']} | "
                    f"{row['candidate_better_cell_count']} | {row['equal_cell_count']} | "
                    f"{row['candidate_worse_cell_count']} |"
                )
    lines.append("")
    lines.extend(["## 汇总关系", ""])
    for label, key in (
        ("NER-L4-TPR vs 当前 Final (TSS-off)", "candidate_vs_current_final_tss_off"),
        ("NER-L4-TPR vs Original", "candidate_vs_original"),
    ):
        row = payload["aggregate_comparisons"][key]
        lines.append(
            f"- {label}：`{row['relation']}`（更好 {row['candidate_better_cell_count']} / "
            f"相同 {row['equal_cell_count']} / 更差 {row['candidate_worse_cell_count']}）。"
        )
    lines.extend(
        [
            "",
            "## 指标说明",
            "",
            "- component FP：未与任何 GT 目标匹配的预测连通域像素数，也是 Fa 的分子。",
            "- background FP：阈值化预测前景中落在 GT 背景上的全部像素数。",
            "- 两类 FP 含义不同，不能互相替代；Pd 也必须与 Fa、IoU 和像素指标一起解释。",
            "- background FP 由同数据集固定 GT 前景像素数与 precision/recall/mIoU 的整数混淆矩阵关系恢复，并逐点交叉校验。",
            "",
            "未生成或补写任何实验数值。所有行均绑定到正式 summary 和各自同角色 checkpoint。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, raw: bytes, *, overwrite: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(
    json_path: Path,
    markdown_path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    validate_output_payload(payload)
    if (Path(json_path).exists() or Path(markdown_path).exists()) and not overwrite:
        existing = json_path if Path(json_path).exists() else markdown_path
        raise FileExistsError(existing)
    json_raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    markdown_raw = render_markdown(payload).encode("utf-8")
    _atomic_write(Path(json_path), json_raw, overwrite=overwrite)
    try:
        _atomic_write(Path(markdown_path), markdown_raw, overwrite=overwrite)
    except Exception:
        if not overwrite and Path(json_path).is_file():
            Path(json_path).unlink()
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--current-root", type=Path, default=DEFAULT_CURRENT_ROOT)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    loaded, completion = load_inputs(
        candidate_root=args.candidate_root,
        current_root=args.current_root,
        original_root=args.original_root,
        allow_partial=args.allow_partial,
        verify_checkpoint_files=True,
    )
    payload = build_comparison(
        loaded,
        completion,
        roots={
            CANDIDATE_METHOD: args.candidate_root,
            CURRENT_METHOD: args.current_root,
            ORIGINAL_METHOD: args.original_root,
        },
        allow_partial=args.allow_partial,
    )
    if payload["status"] == "complete":
        json_path = args.output_dir / COMPLETE_JSON_FILENAME
        markdown_path = args.output_dir / COMPLETE_MARKDOWN_FILENAME
    else:
        json_path = args.output_dir / PARTIAL_JSON_FILENAME
        markdown_path = args.output_dir / PARTIAL_MARKDOWN_FILENAME
    write_outputs(
        json_path,
        markdown_path,
        payload,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "artifact_sha256": payload["artifact_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
