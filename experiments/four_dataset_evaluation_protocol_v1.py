#!/usr/bin/env python3
"""Shared, strict protocol helpers for the four-dataset seed-42 experiment.

This module deliberately contains no dataset or model construction.  It is the
single source of truth for checkpoint selection, fixed-threshold metric
validation, Pd--Fa budget selection, and auditable JSON/checkpoint artefacts.
Unknown experimental values are never synthesized.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_ROOT = RESULTS_ROOT / "four_dataset_seed42_v1"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"
SELECTED_ROOT = EXPERIMENT_ROOT / "selected_checkpoints"

SCHEMA_PREFIX = "sctransnet_four_dataset_seed42"
TRAINING_SEED = 42
EXPECTED_EPOCHS = 1000
CANDIDATE_FIRST_EPOCH = 10
CANDIDATE_LAST_EPOCH = 1000
CANDIDATE_EVAL_EVERY = 10
CANDIDATE_EPOCHS = tuple(
    range(
        CANDIDATE_FIRST_EPOCH,
        CANDIDATE_LAST_EPOCH + 1,
        CANDIDATE_EVAL_EVERY,
    )
)
FIXED_THRESHOLD = 0.5
MATCH_RADIUS = 3.0
TINY_AREA = 9
FA_BUDGETS = (0.5e-6, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
DATASETS = ("SIRST3", "NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
SOURCE_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
METHODS = ("original", "final")
METHOD_LABELS = {
    "original": "Original",
    "final": "Final",
}
SELECTED_ROLES = ("best_miou", "best_pd")
CHECKPOINT_ROLES = SELECTED_ROLES
REPORTING_ROLES = (*SELECTED_ROLES, "last_epoch1000")
CHECKPOINT_FILENAMES = {
    "best_miou": "best_miou.pth.tar",
    "best_pd": "best_pd.pth.tar",
}

METRIC_ALIASES = {
    "test_loss": ("test_loss", "val_loss", "segmentation_loss"),
    "miou": ("miou", "mIoU"),
    "niou": ("niou", "nIoU"),
    "pixel_precision": ("pixel_precision", "precision"),
    "pixel_recall": ("pixel_recall", "recall"),
    "pixel_f1": ("pixel_f1", "f1", "F1", "f_measure"),
    "pd": ("pd", "Pd", "PD"),
    "tiny_pd": ("tiny_pd", "tiny-Pd", "tiny_Pd"),
    "fa": ("fa", "Fa", "FA"),
    "false_objects_per_image": (
        "false_objects_per_image",
        "false_object_rate",
    ),
    "target_count": ("target_count",),
    "matched_target_count": ("matched_target_count",),
    "tiny_target_count": ("tiny_target_count",),
    "matched_tiny_target_count": ("matched_tiny_target_count",),
    "predicted_object_count": ("predicted_object_count",),
    "unmatched_predicted_object_count": (
        "unmatched_predicted_object_count",
        "false_object_count",
    ),
    "valid_pixel_count": ("valid_pixel_count", "valid_pixels"),
}

REQUIRED_SELECTION_METRICS = (
    "test_loss",
    "miou",
    "niou",
    "pd",
    "tiny_pd",
    "fa",
)
STRICT_POINT_RATE_KEYS = (
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
)
STRICT_POINT_COUNT_KEYS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def load_json_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values cannot be serialized")
        return value
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    ready = json_ready(dict(payload))
    serialized = (
        json.dumps(
            ready,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy_file(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
) -> str:
    source = Path(source)
    destination = Path(destination)
    source_digest = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError(destination)
        if file_sha256(destination) == source_digest:
            return source_digest
        if not overwrite:
            raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        require(
            file_sha256(temporary) == source_digest,
            f"checkpoint copy hash mismatch: {source}",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    require(
        file_sha256(destination) == source_digest,
        f"frozen checkpoint hash mismatch: {destination}",
    )
    return source_digest


def _finite_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric, found {value!r}")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{location} must be finite, found {value!r}")
    return output


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be an integer, found {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    # The frozen trainer's JSON fallback serializes NumPy integer scalars as
    # base-10 strings. Accept only that lossless representation; floats and
    # arbitrary numeric text remain invalid.
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError(f"{location} must be an integer, found {value!r}")


def _metric_container(event: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("test_metrics", "evaluation", "metrics"):
        candidate = event.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return event


def _find_alias(container: Mapping[str, Any], canonical_name: str) -> Any:
    aliases = METRIC_ALIASES[canonical_name]
    present = [(alias, container[alias]) for alias in aliases if alias in container]
    if not present:
        raise KeyError(canonical_name)
    first_name, first_value = present[0]
    for alias, value in present[1:]:
        if canonical_json_bytes(value) != canonical_json_bytes(first_value):
            raise ValueError(
                f"metric aliases disagree for {canonical_name}: "
                f"{first_name}={first_value!r}, {alias}={value!r}"
            )
    return first_value


def normalize_metric_event(
    event: Mapping[str, Any],
    *,
    require_selection_fields: bool = True,
) -> dict[str, Any]:
    """Normalize a trainer event without changing metric semantics."""

    epoch = _integer(event.get("epoch"), "metrics.epoch")
    container = _metric_container(event)
    normalized: dict[str, Any] = {"epoch": epoch}
    for canonical_name in METRIC_ALIASES:
        try:
            value = _find_alias(container, canonical_name)
        except KeyError:
            continue
        if canonical_name in STRICT_POINT_COUNT_KEYS:
            normalized[canonical_name] = _integer(
                value, f"metrics[{epoch}].{canonical_name}"
            )
        elif canonical_name == "tiny_pd" and value is None:
            normalized[canonical_name] = None
        else:
            normalized[canonical_name] = _finite_float(
                value, f"metrics[{epoch}].{canonical_name}"
            )
    if require_selection_fields:
        missing = [
            key for key in REQUIRED_SELECTION_METRICS if key not in normalized
        ]
        require(
            not missing,
            f"metrics epoch {epoch} lacks selection fields: {missing}",
        )
    threshold = container.get("threshold", event.get("threshold"))
    if threshold is None:
        raise ValueError(f"metrics epoch {epoch} lacks evaluation threshold")
    normalized["threshold"] = _finite_float(
        threshold, f"metrics[{epoch}].threshold"
    )
    require(
        normalized["threshold"] == FIXED_THRESHOLD,
        f"metrics epoch {epoch} threshold must be exactly {FIXED_THRESHOLD}",
    )
    validate_metric_point(normalized, allow_partial=True)
    return normalized


def validate_metric_point(
    point: Mapping[str, Any],
    *,
    allow_partial: bool = False,
) -> None:
    required = set(REQUIRED_SELECTION_METRICS)
    if not allow_partial:
        required.update((*STRICT_POINT_RATE_KEYS, *STRICT_POINT_COUNT_KEYS))
    missing = sorted(key for key in required if key not in point)
    require(not missing, f"metric point lacks fields: {missing}")

    for key in STRICT_POINT_RATE_KEYS:
        if key not in point:
            continue
        value = point[key]
        if key == "tiny_pd" and value is None:
            continue
        number = _finite_float(value, f"metric point {key}")
        if key != "false_objects_per_image":
            require(0.0 <= number <= 1.0, f"{key} is outside [0, 1]")
        else:
            require(number >= 0.0, f"{key} must be non-negative")
    if "test_loss" in point:
        require(
            _finite_float(point["test_loss"], "metric point test_loss") >= 0.0,
            "test_loss must be non-negative",
        )
    for key in STRICT_POINT_COUNT_KEYS:
        if key in point:
            require(
                _integer(point[key], f"metric point {key}") >= 0,
                f"{key} must be non-negative",
            )

    if "target_count" in point and "matched_target_count" in point:
        require(
            int(point["matched_target_count"]) <= int(point["target_count"]),
            "matched_target_count exceeds target_count",
        )
        expected_pd = int(point["matched_target_count"]) / max(
            1, int(point["target_count"])
        )
        if "pd" in point:
            require(
                math.isclose(
                    float(point["pd"]),
                    expected_pd,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "pd differs from raw target counts",
            )
    if "tiny_target_count" in point and "matched_tiny_target_count" in point:
        tiny_total = int(point["tiny_target_count"])
        tiny_matched = int(point["matched_tiny_target_count"])
        require(
            tiny_matched <= tiny_total,
            "matched_tiny_target_count exceeds tiny_target_count",
        )
        if "tiny_pd" in point:
            expected_tiny = (
                tiny_matched / tiny_total if tiny_total > 0 else None
            )
            require(
                point["tiny_pd"] == expected_tiny
                or (
                    expected_tiny is not None
                    and point["tiny_pd"] is not None
                    and math.isclose(
                        float(point["tiny_pd"]),
                        expected_tiny,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ),
                "tiny_pd differs from raw tiny-target counts",
            )


def tiny_pd_for_selection(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("tiny_pd")
    return -1.0 if value is None else _finite_float(value, "tiny_pd")


def checkpoint_selection_key(
    role: str,
    metrics: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return the complete preregistered key, including earlier-epoch tie."""

    epoch = _integer(metrics.get("epoch"), "selection epoch")
    common_tail = (
        -_finite_float(metrics["test_loss"], "test_loss"),
        -float(epoch),
    )
    if role == "best_miou":
        return (
            _finite_float(metrics["miou"], "miou"),
            _finite_float(metrics["pd"], "pd"),
            -_finite_float(metrics["fa"], "fa"),
            _finite_float(metrics["niou"], "niou"),
            tiny_pd_for_selection(metrics),
            *common_tail,
        )
    if role == "best_pd":
        return (
            _finite_float(metrics["pd"], "pd"),
            -_finite_float(metrics["fa"], "fa"),
            tiny_pd_for_selection(metrics),
            _finite_float(metrics["miou"], "miou"),
            _finite_float(metrics["niou"], "niou"),
            *common_tail,
        )
    raise ValueError(f"unsupported selected role: {role!r}")


def selected_event(
    role: str,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(bool(events), "cannot select from an empty candidate set")
    normalized = [
        normalize_metric_event(event, require_selection_fields=True)
        for event in events
    ]
    return dict(max(normalized, key=lambda item: checkpoint_selection_key(role, item)))


def read_candidate_metrics(path: Path) -> list[dict[str, Any]]:
    """Read exactly the 100 fixed-threshold candidate epochs from JSONL."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    by_epoch: dict[int, dict[str, Any]] = {}
    previous_epoch = 0
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            raise ValueError(f"blank metrics row: {path}:{line_number}")
        event = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(event, dict):
            raise ValueError(f"metrics row is not an object: {path}:{line_number}")
        epoch = _integer(event.get("epoch"), f"{path}:{line_number}.epoch")
        require(
            epoch > previous_epoch,
            f"metrics epochs must be strictly increasing at {path}:{line_number}",
        )
        previous_epoch = epoch
        if epoch in CANDIDATE_EPOCHS:
            require(
                event.get("evaluated") is True,
                f"candidate epoch {epoch} is not marked evaluated",
            )
            require(epoch not in by_epoch, f"duplicate candidate epoch {epoch}")
            by_epoch[epoch] = event
        elif event.get("evaluated") is True:
            raise ValueError(f"unexpected evaluated non-candidate epoch {epoch}")
    missing = sorted(set(CANDIDATE_EPOCHS) - set(by_epoch))
    extra = sorted(set(by_epoch) - set(CANDIDATE_EPOCHS))
    require(
        not missing and not extra,
        f"candidate epoch coverage differs: missing={missing}, extra={extra}",
    )
    return [by_epoch[epoch] for epoch in CANDIDATE_EPOCHS]


def normalize_strict_point(raw: Mapping[str, Any]) -> dict[str, Any]:
    point = dict(raw)
    if "val_loss" in point and "test_loss" not in point:
        point["test_loss"] = point.pop("val_loss")
    if point.get("tiny_pd") is not None:
        tiny = float(point["tiny_pd"])
        if not math.isfinite(tiny):
            point["tiny_pd"] = None
    point = json_ready(point)
    validate_metric_point(point, allow_partial=False)
    return point


def strict_metric_points(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    losses: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Reuse the repository's frozen 8-connected, one-to-one metric core."""

    from experiments.train_tpd_pilot import ValidationMetrics

    require(
        len(probabilities) == len(targets) == len(losses) and probabilities,
        "prediction, target, and loss sequences must be non-empty and aligned",
    )
    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        accumulator = ValidationMetrics(
            float(threshold),
            MATCH_RADIUS,
            TINY_AREA,
        )
        for probability, target, loss in zip(probabilities, targets, losses):
            accumulator.update(probability, target, float(loss))
        point = normalize_strict_point(accumulator.compute())
        point["threshold"] = float(threshold)
        points.append(point)
    return points


def closed_interval_thresholds(
    probabilities: Sequence[np.ndarray],
) -> tuple[list[float], dict[str, Any]]:
    """Reuse the repository grid/adaptive-tail core and closed endpoints."""

    from experiments.evaluate_pd_fa_sweep import threshold_grid
    from experiments.evaluate_tpd_clean_v6_pd_fa import (
        adaptive_thresholds_closed_interval,
    )

    base = threshold_grid(
        0.01,
        0.99,
        0.01,
        (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999),
    )
    thresholds, provenance = adaptive_thresholds_closed_interval(
        probabilities,
        base,
        0.1,
    )
    require(FIXED_THRESHOLD in thresholds, "threshold sweep lacks 0.5")
    return list(map(float, thresholds)), json_ready(provenance)


def best_point_under_fa(
    points: Sequence[Mapping[str, Any]],
    budget: float,
) -> dict[str, Any] | None:
    feasible = [
        point for point in points if float(point["fa"]) <= float(budget)
    ]
    if not feasible:
        return None
    return dict(
        max(
            feasible,
            key=lambda point: (
                float(point["pd"]),
                -float(point["fa"]),
                tiny_pd_for_selection(point),
                float(point["miou"]),
                float(point["niou"]),
                -abs(float(point["threshold"]) - FIXED_THRESHOLD),
            ),
        )
    )


def fa_budget_points(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    return {
        f"{budget:.10g}": best_point_under_fa(points, budget)
        for budget in FA_BUDGETS
    }


def pareto_frontier(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return unique non-dominated (lower Fa, higher Pd) operating points."""

    unique: dict[tuple[float, float], Mapping[str, Any]] = {}
    for point in points:
        key = (float(point["fa"]), float(point["pd"]))
        current = unique.get(key)
        if current is None or (
            float(point["miou"]),
            tiny_pd_for_selection(point),
            -abs(float(point["threshold"]) - FIXED_THRESHOLD),
        ) > (
            float(current["miou"]),
            tiny_pd_for_selection(current),
            -abs(float(current["threshold"]) - FIXED_THRESHOLD),
        ):
            unique[key] = point
    ordered = sorted(
        unique.values(),
        key=lambda point: (
            float(point["fa"]),
            -float(point["pd"]),
            -float(point["miou"]),
        ),
    )
    frontier: list[dict[str, Any]] = []
    best_pd = -1.0
    for point in ordered:
        pd = float(point["pd"])
        if pd > best_pd:
            frontier.append(dict(point))
            best_pd = pd
    return frontier


def run_directory(
    dataset: str,
    method: str,
    *,
    runs_root: Path = RUNS_ROOT,
) -> Path:
    require(dataset in DATASETS, f"unsupported dataset: {dataset!r}")
    require(method in METHODS, f"unsupported method: {method!r}")
    return Path(runs_root) / dataset / method / f"seed_{TRAINING_SEED}"


def selected_checkpoint_path(
    dataset: str,
    method: str,
    role: str,
    *,
    selected_root: Path = SELECTED_ROOT,
) -> Path:
    require(role in CHECKPOINT_ROLES, f"unsupported checkpoint role: {role!r}")
    return (
        Path(selected_root)
        / dataset
        / method
        / CHECKPOINT_FILENAMES[role]
    )


def expected_selection_disclosure() -> dict[str, Any]:
    return {
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "candidate_epoch_rule": "10,20,...,1000",
        "candidate_epoch_count": len(CANDIDATE_EPOCHS),
        "eval_every": CANDIDATE_EVAL_EVERY,
        "selection_threshold": FIXED_THRESHOLD,
        "selection_source": "corresponding_official_test_split",
        "test_selected": True,
        "selection_is_optimistic": True,
        "best_miou_order": [
            "higher_miou",
            "higher_pd",
            "lower_fa",
            "higher_niou",
            "higher_tiny_pd",
            "lower_test_loss",
            "earlier_epoch",
        ],
        "best_pd_order": [
            "higher_pd",
            "lower_fa",
            "higher_tiny_pd",
            "higher_miou",
            "higher_niou",
            "lower_test_loss",
            "earlier_epoch",
        ],
    }


def source_hashes() -> dict[str, str]:
    from experiments import evaluate_pd_fa_sweep
    from experiments import train_tpd_pilot
    from experiments import evaluate_tpd_clean_v6_pd_fa

    sources = (
        Path(__file__).resolve(),
        Path(train_tpd_pilot.__file__).resolve(),
        Path(evaluate_pd_fa_sweep.__file__).resolve(),
        Path(evaluate_tpd_clean_v6_pd_fa.__file__).resolve(),
    )
    return {
        str(source.relative_to(REPO_ROOT)): file_sha256(source)
        for source in sources
    }
