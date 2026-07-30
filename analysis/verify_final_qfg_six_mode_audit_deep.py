#!/usr/bin/env python3
"""Independently deep-verify an F1 six-mode QFG audit artifact set.

The verifier is read-only with respect to the source audit.  It reloads the
six prediction-cache manifests and arrays, recomputes every cache-derived
quantity, and records explicit limits for quantities whose raw observations
were not persisted by the F1 runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from skimage import measure


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import audit_final_qfg_functional_use as audit_core  # noqa: E402
from analysis import (  # noqa: E402
    collect_final_model_validation_statistics as cache_core,
)
from analysis import run_final_qfg_six_mode_audit as audit_runner  # noqa: E402
from experiments import evaluate_pd_fa_sweep as sweep_core  # noqa: E402
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa as qfg_evaluator,
)
from experiments import (  # noqa: E402
    export_tpd_ner_v4_qfg_v2_croa_to_inference as exporter,
)
from experiments.evaluate_tpd_clean_v6_pd_fa import (  # noqa: E402
    adaptive_thresholds_closed_interval,
)


DEEP_SCHEMA = "sctransnet_final_model_qfg_six_mode_deep_verification_v1"
ACTION_SCHEMA = (
    "sctransnet_final_model_qfg_six_mode_deep_verification_action_v1"
)
DEFAULT_SOURCE_REPORT = (
    audit_runner.DEFAULT_OUTPUT_DIR / audit_runner.REPORT_FILENAME
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "analysis/results/"
    "final_model_qfg_six_mode_deep_verification_v1/"
    "final_model_qfg_six_mode_deep_verification_v1.json"
)
FLOAT_ATOL = 1e-14
FLOAT_RTOL = 2e-14


class DeepAuditVerificationError(ValueError):
    """The F1 report and its independently recomputed evidence differ."""


def _fail(message: str) -> None:
    raise DeepAuditVerificationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: expected={expected!r}, observed={observed!r}"
        )


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be one finite number")
    return float(value)


def _close(label: str, observed: Any, expected: Any) -> None:
    left = _finite_number(observed, f"{label} observed")
    right = _finite_number(expected, f"{label} expected")
    if not math.isclose(
        left,
        right,
        rel_tol=FLOAT_RTOL,
        abs_tol=FLOAT_ATOL,
    ):
        _fail(f"{label} differs: expected={right!r}, observed={left!r}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(f"expected a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> tuple[Path, dict[str, Any], bytes]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        _fail(f"{label} must not be a symlink: {requested}")
    source = requested.resolve()
    if not source.is_file():
        _fail(f"{label} must be a regular file: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepAuditVerificationError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    _require(isinstance(value, dict), f"{label} must contain one object")
    return source, value, raw


def _sha256_string(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be one lowercase SHA-256 digest",
    )
    return value


def _safe_report_child(report_path: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and relative, "cache path is missing")
    pure = PurePosixPath(relative)
    _require(
        relative == pure.as_posix()
        and not pure.is_absolute()
        and ".." not in pure.parts,
        f"cache path is not canonical report-relative: {relative!r}",
    )
    candidate = report_path.parent.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"cache manifest must be a regular file: {candidate}")
    resolved = candidate.resolve()
    _require(
        resolved.is_relative_to(report_path.parent.resolve()),
        f"cache manifest escapes source audit directory: {relative}",
    )
    return resolved


def _expected_source_bindings(repo_root: Path) -> dict[str, dict[str, str]]:
    root = Path(repo_root).resolve()
    sources = {
        "runner": Path(audit_runner.__file__).resolve(),
        "knockout_core": Path(audit_core.__file__).resolve(),
        "cache_core": Path(cache_core.__file__).resolve(),
        "qfg_evaluator": Path(qfg_evaluator.__file__).resolve(),
        "metric_core": Path(sweep_core.__file__).resolve(),
        "exporter": Path(exporter.__file__).resolve(),
    }
    result: dict[str, dict[str, str]] = {}
    for role, path in sources.items():
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise DeepAuditVerificationError(
                f"{role} source lies outside repository"
            ) from exc
        result[role] = {
            "path": relative,
            "sha256": _sha256_file(path),
        }
    return result


def _resolve_context(
    report: Mapping[str, Any],
    *,
    repo_root: Path,
    parent_lock: Path | None,
    source_lock: Path | None,
    expected_context: audit_runner.FrozenAuditContext | None,
) -> audit_runner.FrozenAuditContext:
    live_required = report.get("live_authority_required")
    _require(type(live_required) is bool, "live_authority_required is invalid")
    if live_required:
        context = audit_runner.load_frozen_audit_context(
            repo_root,
            parent_lock,
            source_lock,
            require_source_lock=True,
        )
        if expected_context is not None:
            _equal(
                "supplied/live authority",
                expected_context.authority_binding,
                context.authority_binding,
            )
    else:
        _require(
            expected_context is not None,
            "non-live deep verification requires an expected context",
        )
        context = expected_context
    _equal("report authority", report.get("authority"), context.authority_binding)
    return context


def _validate_execution_contract(
    report: Mapping[str, Any],
    context: audit_runner.FrozenAuditContext,
) -> None:
    contract = report.get("execution_contract")
    _require(isinstance(contract, Mapping), "execution contract is missing")
    _equal(
        "execution contract keys",
        set(contract),
        {
            "model_eval",
            "torch_inference_mode",
            "batch_size",
            "shuffle",
            "workers",
            "validation_count",
            "validation_ids_sha256",
            "checkpoint_sha256",
            "source_checkpoint_sha256",
            "certification_source_lock_sha256",
            "parent_lock_sha256",
            "prediction_comparison",
            "fixed_threshold",
            "fa_budgets",
            "match_radius",
            "tiny_area",
            "modes",
            "derived_checkpoint_written",
        },
    )
    expected = {
        "model_eval": True,
        "torch_inference_mode": True,
        "batch_size": 1,
        "shuffle": False,
        "workers": 0,
        "validation_count": len(context.validation_ids),
        "validation_ids_sha256": context.validation_ids_sha256,
        "checkpoint_sha256": context.checkpoint_sha256,
        "source_checkpoint_sha256": context.source_checkpoint_sha256,
        "certification_source_lock_sha256": context.source_lock_sha256,
        "parent_lock_sha256": context.authority_binding["parent_lock"][
            "sha256"
        ],
        "prediction_comparison": "probability > threshold",
        "fixed_threshold": audit_runner.FIXED_THRESHOLD,
        "fa_budgets": list(audit_runner.FA_BUDGETS),
        "match_radius": audit_runner.MATCH_RADIUS,
        "tiny_area": audit_runner.TINY_AREA,
        "modes": list(audit_runner.PUBLIC_MODES),
        "derived_checkpoint_written": False,
    }
    _equal("execution contract", dict(contract), expected)


def _load_six_caches(
    report_path: Path,
    report: Mapping[str, Any],
    context: audit_runner.FrozenAuditContext,
) -> tuple[
    dict[str, cache_core.PredictionCache],
    dict[str, dict[str, str]],
]:
    modes = report.get("modes")
    _require(isinstance(modes, Mapping), "report modes are missing")
    _equal("report mode set", set(modes), set(audit_runner.PUBLIC_MODES))
    caches: dict[str, cache_core.PredictionCache] = {}
    bindings: dict[str, dict[str, str]] = {}
    cache_keys: set[str] = set()
    for public_mode in audit_runner.PUBLIC_MODES:
        mode = modes[public_mode]
        _require(isinstance(mode, Mapping), f"{public_mode} record is invalid")
        _equal(f"{public_mode} public mode", mode.get("public_mode"), public_mode)
        primitive = audit_runner.PUBLIC_TO_PRIMITIVE_MODE[public_mode]
        _equal(f"{public_mode} primitive mode", mode.get("primitive_mode"), primitive)
        binding = mode.get("cache")
        _require(
            isinstance(binding, Mapping)
            and set(binding) == {"path", "sha256"},
            f"{public_mode} cache binding is invalid",
        )
        metadata_path = _safe_report_child(report_path, binding.get("path"))
        _equal(
            f"{public_mode} cache manifest SHA",
            _sha256_file(metadata_path),
            _sha256_string(
                binding.get("sha256"),
                f"{public_mode} cache manifest SHA",
            ),
        )
        cache = cache_core.load_prediction_cache(
            metadata_path,
            expected_identity=context.cache_identity(public_mode),
        )
        _equal(
            f"{public_mode} validation order",
            tuple(record.image_id for record in cache.records),
            tuple(context.validation_ids),
        )
        key = cache.identity["cache_key_sha256"]
        _require(key not in cache_keys, "six modes do not have unique cache keys")
        cache_keys.add(key)
        caches[public_mode] = cache
        bindings[public_mode] = {
            "manifest_path": binding["path"],
            "manifest_sha256": binding["sha256"],
            "array_content_sha256": cache.content_sha256,
            "cache_key_sha256": key,
        }

    full = caches["full"]
    for public_mode in audit_runner.COUNTERFACTUAL_MODES:
        other = caches[public_mode]
        _equal(
            f"{public_mode} compatibility SHA",
            other.identity["compatibility_sha256"],
            full.identity["compatibility_sha256"],
        )
        _equal(
            f"{public_mode} image count",
            len(other.records),
            len(full.records),
        )
        for index, (left, right) in enumerate(zip(full.records, other.records)):
            _equal(
                f"{public_mode} image[{index}] ID",
                right.image_id,
                left.image_id,
            )
            _equal(
                f"{public_mode} image[{index}] shape",
                right.probability.shape,
                left.probability.shape,
            )
            _require(
                np.array_equal(right.target, left.target),
                f"{public_mode} image[{index}] target differs",
            )
    return caches, bindings


def _recompute_fixed_metrics(
    report: Mapping[str, Any],
    caches: Mapping[str, cache_core.PredictionCache],
) -> dict[str, str]:
    status: dict[str, str] = {}
    for public_mode, cache in caches.items():
        expected = cache_core.recompute_metrics(
            cache,
            threshold=audit_runner.FIXED_THRESHOLD,
        )
        _equal(
            f"{public_mode} fixed-threshold metrics",
            report["modes"][public_mode].get("fixed_threshold_metrics"),
            expected,
        )
        status[public_mode] = "fully_recomputed_from_cache"
    return status


def recompute_formal_fa_budget_scan(
    cache: cache_core.PredictionCache,
) -> dict[str, Any]:
    probabilities = [record.probability for record in cache.records]
    base = sweep_core.threshold_grid(
        0.01,
        0.99,
        0.01,
        audit_runner.EXTRA_THRESHOLDS,
    )
    thresholds, provenance = adaptive_thresholds_closed_interval(
        probabilities,
        base,
        0.1,
    )
    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        point = cache_core.recompute_metrics(cache, threshold=threshold)
        point["threshold"] = float(threshold)
        points.append(point)
    return {
        "status": "complete",
        "formal_closed_interval_grid": True,
        "prediction_comparison": "probability > threshold",
        "fa_budgets": list(audit_runner.FA_BUDGETS),
        "budget_points": {
            f"{budget:.10g}": sweep_core.best_point_under_fa(points, budget)
            for budget in audit_runner.FA_BUDGETS
        },
        "threshold_provenance": provenance,
        "threshold_count": len(points),
    }


def _verify_fa_budget_scans(
    report: Mapping[str, Any],
    caches: Mapping[str, cache_core.PredictionCache],
    *,
    live_authority_required: bool,
) -> tuple[dict[str, str], list[str]]:
    status: dict[str, str] = {}
    limits: list[str] = []
    budget_keys = {
        f"{budget:.10g}" for budget in audit_runner.FA_BUDGETS
    }
    for public_mode, cache in caches.items():
        observed = report["modes"][public_mode].get("fa_budget_scan")
        _require(isinstance(observed, Mapping), f"{public_mode} Fa scan missing")
        _equal(
            f"{public_mode} Fa scan keys",
            set(observed),
            {
                "status",
                "formal_closed_interval_grid",
                "prediction_comparison",
                "fa_budgets",
                "budget_points",
                "threshold_provenance",
                "threshold_count",
            },
        )
        _equal(f"{public_mode} Fa scan status", observed["status"], "complete")
        _equal(
            f"{public_mode} Fa comparison",
            observed["prediction_comparison"],
            "probability > threshold",
        )
        _equal(
            f"{public_mode} Fa budgets",
            observed["fa_budgets"],
            list(audit_runner.FA_BUDGETS),
        )
        _equal(
            f"{public_mode} budget keys",
            set(observed["budget_points"]),
            budget_keys,
        )
        formal = observed.get("formal_closed_interval_grid")
        _require(type(formal) is bool, f"{public_mode} formal-grid flag invalid")
        if live_authority_required:
            _require(formal, f"{public_mode} live audit used a test threshold grid")
        if formal:
            expected = recompute_formal_fa_budget_scan(cache)
            _equal(f"{public_mode} formal Fa scan", dict(observed), expected)
            status[public_mode] = "full_grid_and_budget_optima_recomputed"
            continue

        provenance = observed.get("threshold_provenance")
        _require(
            isinstance(provenance, Mapping)
            and provenance.get("test_override") is True,
            f"{public_mode} non-formal threshold provenance is invalid",
        )
        _equal(
            f"{public_mode} non-formal threshold count",
            observed.get("threshold_count"),
            provenance.get("total_unique_threshold_count"),
        )
        for budget in audit_runner.FA_BUDGETS:
            key = f"{budget:.10g}"
            point = observed["budget_points"][key]
            _require(
                isinstance(point, Mapping),
                f"{public_mode} budget {key} has no selected point",
            )
            threshold = _finite_number(
                point.get("threshold"),
                f"{public_mode} budget {key} threshold",
            )
            expected_point = cache_core.recompute_metrics(
                cache,
                threshold=threshold,
            )
            expected_point["threshold"] = threshold
            _equal(
                f"{public_mode} budget {key} selected-point metrics",
                dict(point),
                expected_point,
            )
            _require(
                float(point["fa"]) <= budget,
                f"{public_mode} budget {key} point violates its Fa budget",
            )
        status[public_mode] = (
            "selected_points_recomputed_grid_optimality_not_recomputable"
        )
        limits.append(
            f"{public_mode}.fa_budget_scan optimality: non-formal override "
            "threshold values were not persisted"
        )
    return status, limits


def _recompute_probability_comparison(
    full: cache_core.PredictionCache,
    counterfactual: cache_core.PredictionCache,
) -> dict[str, Any]:
    per_image: list[dict[str, Any]] = []
    maximum = 0.0
    total = 0.0
    count = 0
    for left, right in zip(full.records, counterfactual.records):
        difference = np.abs(
            left.probability.astype(np.float64)
            - right.probability.astype(np.float64)
        )
        image_max = float(difference.max())
        image_mean = float(difference.mean())
        maximum = max(maximum, image_max)
        total += float(difference.sum())
        count += int(difference.size)
        per_image.append(
            {
                "image_id": left.image_id,
                "max_abs": image_max,
                "mean_abs": image_mean,
            }
        )
    mean_abs = total / count
    equivalent = (
        maximum <= audit_core.OUTPUT_EQUIVALENCE_MAX_ABS
        and mean_abs <= audit_core.OUTPUT_EQUIVALENCE_MEAN_ABS
    )
    return {
        "schema": audit_core.AUDIT_SCHEMA,
        "status": "complete",
        "scope": "internal_validation_same_checkpoint_counterfactual",
        "official_test_accessed": False,
        "checkpoint_sha256": full.identity["checkpoint_sha256"],
        "dataset_sha256": full.identity["dataset_sha256"],
        "evaluator_sha256": full.identity["evaluator_sha256"],
        "full_cache_key_sha256": full.identity["cache_key_sha256"],
        "counterfactual_cache_key_sha256": (
            counterfactual.identity["cache_key_sha256"]
        ),
        "counterfactual_mode": counterfactual.identity["mode"],
        "output_difference": {
            "max_abs": maximum,
            "mean_abs": mean_abs,
            "equivalent": equivalent,
            "functionally_different": not equivalent,
            "equivalence_max_abs_threshold": (
                audit_core.OUTPUT_EQUIVALENCE_MAX_ABS
            ),
            "equivalence_mean_abs_threshold": (
                audit_core.OUTPUT_EQUIVALENCE_MEAN_ABS
            ),
            "per_image": per_image,
        },
        "fixed_threshold": audit_runner.FIXED_THRESHOLD,
        "full_metrics": cache_core.recompute_metrics(
            full,
            threshold=audit_runner.FIXED_THRESHOLD,
        ),
        "counterfactual_metrics": cache_core.recompute_metrics(
            counterfactual,
            threshold=audit_runner.FIXED_THRESHOLD,
        ),
        "diagnostic_only": True,
        "derived_checkpoint_written": False,
    }


def _recompute_component_difference(
    full: cache_core.PredictionCache,
    counterfactual: cache_core.PredictionCache,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = {
        "changed_image_count": 0,
        "changed_pixel_count": 0,
        "full_only_component_count": 0,
        "counterfactual_only_component_count": 0,
        "overlapping_full_component_count": 0,
        "overlapping_counterfactual_component_count": 0,
    }
    threshold = audit_runner.FIXED_THRESHOLD
    for left, right in zip(full.records, counterfactual.records):
        full_binary = left.probability > threshold
        other_binary = right.probability > threshold
        changed_pixels = int(np.count_nonzero(full_binary != other_binary))
        full_labels = measure.label(full_binary, connectivity=2)
        other_labels = measure.label(other_binary, connectivity=2)
        full_count = int(full_labels.max())
        other_count = int(other_labels.max())
        overlapping_full = sum(
            bool(np.any(other_binary[full_labels == label]))
            for label in range(1, full_count + 1)
        )
        overlapping_other = sum(
            bool(np.any(full_binary[other_labels == label]))
            for label in range(1, other_count + 1)
        )
        row = {
            "image_id": left.image_id,
            "changed_pixel_count": changed_pixels,
            "full_component_count": full_count,
            "counterfactual_component_count": other_count,
            "full_only_component_count": full_count - overlapping_full,
            "counterfactual_only_component_count": (
                other_count - overlapping_other
            ),
            "overlapping_full_component_count": overlapping_full,
            "overlapping_counterfactual_component_count": overlapping_other,
        }
        rows.append(row)
        totals["changed_image_count"] += int(changed_pixels > 0)
        totals["changed_pixel_count"] += changed_pixels
        for field in tuple(totals)[2:]:
            totals[field] += int(row[field])
    return {
        "schema": audit_runner.COMPONENT_SCHEMA,
        "status": "complete",
        "threshold": threshold,
        "connectivity": 2,
        "overlap_definition": "at_least_one_shared_positive_pixel",
        **totals,
        "per_image": rows,
    }


def _bootstrap_stat_arrays(
    rows: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    def sums(field: str) -> np.ndarray:
        values = np.asarray([int(row[field]) for row in rows], dtype=np.int64)
        return values[indices].sum(axis=1, dtype=np.int64)

    target = sums("target_count")
    matched = sums("matched_target_count")
    tiny = sums("tiny_target_count")
    matched_tiny = sums("matched_tiny_target_count")
    intersection = sums("intersection")
    union = sums("union")
    unmatched_pixels = sums("unmatched_predicted_pixels")
    valid = sums("valid_pixel_count")
    unmatched_objects = sums("unmatched_predicted_object_count")
    return {
        "pd": matched / np.maximum(1, target),
        "fa": unmatched_pixels / np.maximum(1, valid),
        "miou": intersection / np.maximum(1, union),
        "tiny_pd": np.divide(
            matched_tiny,
            tiny,
            out=np.full(tiny.shape, np.nan, dtype=np.float64),
            where=tiny != 0,
        ),
        "false_objects_per_image": unmatched_objects / indices.shape[1],
    }


def recompute_paired_image_bootstrap(
    full: cache_core.PredictionCache,
    counterfactual: cache_core.PredictionCache,
    *,
    replicates: int,
    rng_seed: int,
) -> dict[str, Any]:
    _require(
        isinstance(replicates, int)
        and not isinstance(replicates, bool)
        and replicates > 0,
        "bootstrap replicate count is invalid",
    )
    _require(
        isinstance(rng_seed, int) and not isinstance(rng_seed, bool),
        "bootstrap seed is invalid",
    )
    full_rows = cache_core.image_sufficient_statistics(
        full,
        threshold=audit_runner.FIXED_THRESHOLD,
    )
    other_rows = cache_core.image_sufficient_statistics(
        counterfactual,
        threshold=audit_runner.FIXED_THRESHOLD,
    )
    point_full = cache_core.aggregate_sufficient_statistics(full_rows)
    point_other = cache_core.aggregate_sufficient_statistics(other_rows)
    point_delta = {
        key: float(point_full[key]) - float(point_other[key])
        for key in audit_runner.METRIC_KEYS
    }
    count = len(full_rows)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(
        0,
        count,
        size=(replicates, count),
        dtype=np.int64,
    )
    left = _bootstrap_stat_arrays(full_rows, indices)
    right = _bootstrap_stat_arrays(other_rows, indices)
    intervals: dict[str, Any] = {}
    for key in audit_runner.METRIC_KEYS:
        deltas = left[key] - right[key]
        finite = deltas[np.isfinite(deltas)]
        _require(finite.size > 0, f"bootstrap has no finite {key} deltas")
        lower, upper = np.quantile(finite, (0.005, 0.995))
        intervals[key] = {
            "delta_orientation": "full_minus_counterfactual",
            "point_delta": point_delta[key],
            "lower": float(lower),
            "upper": float(upper),
            "finite_replicates": int(finite.size),
        }
    return {
        "schema": audit_runner.BOOTSTRAP_SCHEMA,
        "status": "complete",
        "unit": "paired_image",
        "threshold": audit_runner.FIXED_THRESHOLD,
        "replicates": replicates,
        "rng_seed": rng_seed,
        "shared_resample_indices": True,
        "metric_family": list(audit_runner.METRIC_KEYS),
        "simultaneous_family_confidence": (
            audit_runner.SIMULTANEOUS_FAMILY_CI
        ),
        "per_metric_two_sided_confidence": (
            audit_runner.PER_METRIC_TWO_SIDED_CI
        ),
        "method": "Bonferroni percentile intervals",
        "intervals": intervals,
    }


def _verify_cache_pair_derivatives(
    report: Mapping[str, Any],
    caches: Mapping[str, cache_core.PredictionCache],
) -> dict[str, dict[str, str]]:
    full = caches["full"]
    status: dict[str, dict[str, str]] = {}
    _equal("full comparison_to_full", report["modes"]["full"].get("comparison_to_full"), None)
    _equal("full component_difference", report["modes"]["full"].get("component_difference"), None)
    _equal("full paired bootstrap", report["modes"]["full"].get("paired_image_bootstrap"), None)
    for public_mode in audit_runner.COUNTERFACTUAL_MODES:
        mode = report["modes"][public_mode]
        other = caches[public_mode]
        comparison = _recompute_probability_comparison(full, other)
        _equal(
            f"{public_mode} probability comparison",
            mode.get("comparison_to_full"),
            comparison,
        )
        component = _recompute_component_difference(full, other)
        _equal(
            f"{public_mode} component difference",
            mode.get("component_difference"),
            component,
        )
        observed_bootstrap = mode.get("paired_image_bootstrap")
        _require(
            isinstance(observed_bootstrap, Mapping),
            f"{public_mode} paired bootstrap is missing",
        )
        bootstrap = recompute_paired_image_bootstrap(
            full,
            other,
            replicates=observed_bootstrap.get("replicates"),
            rng_seed=observed_bootstrap.get("rng_seed"),
        )
        _equal(
            f"{public_mode} paired bootstrap",
            dict(observed_bootstrap),
            bootstrap,
        )
        status[public_mode] = {
            "probability_difference": "fully_recomputed_from_cache",
            "component_counts": "fully_recomputed_from_cache",
            "paired_image_bootstrap": "fully_recomputed_from_cache",
        }
    return status


_DISTRIBUTION_FIELDS = {
    "count",
    "mean",
    "rms",
    "p5",
    "p50",
    "p95",
    "minimum",
    "maximum",
}


def _validate_distribution(
    value: Any,
    *,
    factor: bool,
    label: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    expected_fields = set(_DISTRIBUTION_FIELDS)
    if factor:
        expected_fields.add("mean_abs_factor_minus_one")
    _equal(f"{label} fields", set(value), expected_fields)
    count = value.get("count")
    _require(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0,
        f"{label}.count is invalid",
    )
    numeric = [
        "mean",
        "rms",
        "p5",
        "p50",
        "p95",
        "minimum",
        "maximum",
    ]
    if factor:
        numeric.append("mean_abs_factor_minus_one")
    if count == 0:
        for name in numeric:
            _equal(f"{label}.{name}", value.get(name), None)
        return dict(value)
    numbers = {
        name: _finite_number(value.get(name), f"{label}.{name}")
        for name in numeric
    }
    _require(numbers["rms"] >= 0.0, f"{label}.rms is negative")
    _require(
        numbers["minimum"]
        <= numbers["p5"]
        <= numbers["p50"]
        <= numbers["p95"]
        <= numbers["maximum"],
        f"{label} quantile ordering is invalid",
    )
    _require(
        numbers["rms"] + FLOAT_ATOL >= abs(numbers["mean"]),
        f"{label} RMS/mean relation is invalid",
    )
    if factor:
        _require(
            numbers["mean_abs_factor_minus_one"] >= 0.0,
            f"{label} mean absolute factor deviation is negative",
        )
        _require(
            numbers["mean_abs_factor_minus_one"] + FLOAT_ATOL
            >= abs(numbers["mean"] - 1.0),
            f"{label} factor absolute-mean bound is invalid",
        )
    return dict(value)


def _weighted_region_consistency(
    regions: Mapping[str, Any],
    *,
    quantity: str,
    factor: bool,
    label: str,
) -> None:
    global_value = regions["global"][quantity]
    parts = [
        regions[name][quantity]
        for name in ("target", "hard_negative", "ordinary_background")
    ]
    _equal(
        f"{label} partition count",
        global_value["count"],
        sum(part["count"] for part in parts),
    )
    nonempty = [part for part in parts if part["count"] > 0]
    _require(bool(nonempty), f"{label} has no nonempty region")
    total = global_value["count"]
    mean = sum(part["count"] * part["mean"] for part in nonempty) / total
    rms = math.sqrt(
        sum(part["count"] * part["rms"] ** 2 for part in nonempty) / total
    )
    _close(f"{label} weighted mean", global_value["mean"], mean)
    _close(f"{label} weighted RMS", global_value["rms"], rms)
    _close(
        f"{label} partition minimum",
        global_value["minimum"],
        min(part["minimum"] for part in nonempty),
    )
    _close(
        f"{label} partition maximum",
        global_value["maximum"],
        max(part["maximum"] for part in nonempty),
    )
    if factor:
        mean_abs = (
            sum(
                part["count"] * part["mean_abs_factor_minus_one"]
                for part in nonempty
            )
            / total
        )
        _close(
            f"{label} weighted mean absolute factor deviation",
            global_value["mean_abs_factor_minus_one"],
            mean_abs,
        )


def _require_unit_factor_distribution(
    value: Mapping[str, Any],
    label: str,
) -> None:
    if value["count"] == 0:
        return
    for field in ("mean", "rms", "p5", "p50", "p95", "minimum", "maximum"):
        _equal(f"{label}.{field}", value[field], 1.0)
    _equal(
        f"{label}.mean_abs_factor_minus_one",
        value["mean_abs_factor_minus_one"],
        0.0,
    )


def _verify_factor_and_region_summaries(
    report: Mapping[str, Any],
    image_count: int,
) -> dict[str, Any]:
    full_regions: list[dict[str, Any]] | None = None
    full_factors: list[dict[str, Any]] | None = None
    source_alpha_sha: str | None = None
    for public_mode in audit_runner.PUBLIC_MODES:
        mode = report["modes"][public_mode]
        primitive = audit_runner.PUBLIC_TO_PRIMITIVE_MODE[public_mode]
        knockout = mode.get("alpha_knockout")
        _require(isinstance(knockout, Mapping), f"{public_mode} knockout missing")
        _equal(
            f"{public_mode} knockout fields",
            set(knockout),
            {
                "schema",
                "mode",
                "source_alpha_sha256",
                "active_alpha_sha256",
                "selected_level_indices_zero_based",
                "derived_checkpoint_written",
                "diagnostic_only",
            },
        )
        _equal(
            f"{public_mode} knockout schema",
            knockout["schema"],
            audit_core.ALPHA_KNOCKOUT_SCHEMA,
        )
        descriptor = cache_core.normalize_mode(primitive)
        _equal(f"{public_mode} knockout mode", knockout["mode"], descriptor)
        _equal(
            f"{public_mode} selected levels",
            knockout["selected_level_indices_zero_based"],
            descriptor["knockout_level_indices_zero_based"],
        )
        _equal(
            f"{public_mode} knockout diagnostic flag",
            knockout["diagnostic_only"],
            public_mode != "full",
        )
        _equal(
            f"{public_mode} derived checkpoint flag",
            knockout["derived_checkpoint_written"],
            False,
        )
        current_source_sha = _sha256_string(
            knockout["source_alpha_sha256"],
            f"{public_mode} source alpha SHA",
        )
        _sha256_string(
            knockout["active_alpha_sha256"],
            f"{public_mode} active alpha SHA",
        )
        if source_alpha_sha is None:
            source_alpha_sha = current_source_sha
        _equal(
            f"{public_mode} common source alpha SHA",
            current_source_sha,
            source_alpha_sha,
        )
        if public_mode == "full":
            _equal(
                "full active/source alpha SHA",
                knockout["active_alpha_sha256"],
                current_source_sha,
            )

        factor_summary = mode.get("factor_summary")
        _require(
            isinstance(factor_summary, Mapping),
            f"{public_mode} factor summary missing",
        )
        _equal(
            f"{public_mode} factor summary fields",
            set(factor_summary),
            {
                "schema",
                "forward_count",
                "levels",
                "maximum_level_mean_abs_factor_minus_one",
                "nontrivial_factor_use",
                "nontrivial_factor_threshold",
            },
        )
        _equal(
            f"{public_mode} factor schema",
            factor_summary["schema"],
            audit_core.FACTOR_SUMMARY_SCHEMA,
        )
        _equal(
            f"{public_mode} factor forward count",
            factor_summary["forward_count"],
            image_count,
        )
        factor_levels = factor_summary.get("levels")
        _require(
            isinstance(factor_levels, list) and len(factor_levels) == 4,
            f"{public_mode} factor levels are invalid",
        )

        region_summary = mode.get("factor_gate_region_statistics")
        _require(
            isinstance(region_summary, Mapping),
            f"{public_mode} region summary missing",
        )
        _equal(
            f"{public_mode} region summary fields",
            set(region_summary),
            {
                "schema",
                "status",
                "image_count",
                "region_definition",
                "levels",
            },
        )
        _equal(
            f"{public_mode} region schema",
            region_summary["schema"],
            audit_runner.REGION_SCHEMA,
        )
        _equal(f"{public_mode} region status", region_summary["status"], "complete")
        _equal(
            f"{public_mode} region image count",
            region_summary["image_count"],
            image_count,
        )
        _equal(
            f"{public_mode} region definition",
            region_summary["region_definition"],
            {
                "target": "adaptive_max_pool(target>0.5, gate_grid)>0",
                "hard_negative": (
                    "non_target gate cell containing a full-mode "
                    "false-positive pixel at threshold 0.5"
                ),
                "ordinary_background": (
                    "neither target nor fixed full-mode hard-negative"
                ),
                "mapping": "adaptive_max_pool_to_each_qfg_level",
                "hard_negative_reference_mode": "full",
            },
        )
        region_levels = region_summary.get("levels")
        _require(
            isinstance(region_levels, list) and len(region_levels) == 4,
            f"{public_mode} region levels are invalid",
        )

        mean_deviations: list[float] = []
        selected = set(knockout["selected_level_indices_zero_based"])
        for index, (factor_level, region_level) in enumerate(
            zip(factor_levels, region_levels)
        ):
            _require(
                isinstance(factor_level, Mapping),
                f"{public_mode} factor level {index + 1} is invalid",
            )
            _equal(
                f"{public_mode} factor level {index + 1} fields",
                set(factor_level),
                {
                    "level",
                    "element_count",
                    "mean",
                    "rms",
                    "p5",
                    "p50",
                    "p95",
                    "minimum",
                    "maximum",
                    "mean_abs_factor_minus_one",
                    "max_abs_factor_minus_one",
                },
            )
            _equal(
                f"{public_mode} factor level number",
                factor_level["level"],
                index + 1,
            )
            _require(
                isinstance(region_level, Mapping),
                f"{public_mode} region level {index + 1} is invalid",
            )
            _equal(
                f"{public_mode} region level {index + 1} fields",
                set(region_level),
                {"level", "regions", "factor_contrasts", "gate_contrasts"},
            )
            _equal(
                f"{public_mode} region level number",
                region_level["level"],
                index + 1,
            )
            regions = region_level.get("regions")
            _require(isinstance(regions, Mapping), "region map is missing")
            _equal(
                f"{public_mode} level {index + 1} region set",
                set(regions),
                {"global", "target", "hard_negative", "ordinary_background"},
            )
            for region_name, region in regions.items():
                _require(
                    isinstance(region, Mapping)
                    and set(region) == {"gate", "factor"},
                    f"{public_mode} level {index + 1} {region_name} invalid",
                )
                gate = _validate_distribution(
                    region["gate"],
                    factor=False,
                    label=(
                        f"{public_mode}.level{index + 1}."
                        f"{region_name}.gate"
                    ),
                )
                factor = _validate_distribution(
                    region["factor"],
                    factor=True,
                    label=(
                        f"{public_mode}.level{index + 1}."
                        f"{region_name}.factor"
                    ),
                )
                _equal(
                    f"{public_mode} level {index + 1} "
                    f"{region_name} gate/factor count",
                    gate["count"],
                    factor["count"],
                )
            _weighted_region_consistency(
                regions,
                quantity="gate",
                factor=False,
                label=f"{public_mode}.level{index + 1}.gate",
            )
            _weighted_region_consistency(
                regions,
                quantity="factor",
                factor=True,
                label=f"{public_mode}.level{index + 1}.factor",
            )
            global_factor = regions["global"]["factor"]
            _equal(
                f"{public_mode} level {index + 1} element count",
                factor_level["element_count"],
                global_factor["count"],
            )
            for field in (
                "mean",
                "rms",
                "p5",
                "p50",
                "p95",
                "minimum",
                "maximum",
                "mean_abs_factor_minus_one",
            ):
                _equal(
                    f"{public_mode} level {index + 1} "
                    f"factor/global {field}",
                    factor_level[field],
                    global_factor[field],
                )
            max_abs = max(
                abs(float(factor_level["minimum"]) - 1.0),
                abs(float(factor_level["maximum"]) - 1.0),
            )
            _close(
                f"{public_mode} level {index + 1} max factor deviation",
                factor_level["max_abs_factor_minus_one"],
                max_abs,
            )
            mean_deviations.append(
                float(factor_level["mean_abs_factor_minus_one"])
            )
            for contrast_name, region_name in (
                ("target_minus_background", "target"),
                ("hard_negative_minus_background", "hard_negative"),
            ):
                for quantity, contrast_group in (
                    ("factor", "factor_contrasts"),
                    ("gate", "gate_contrasts"),
                ):
                    left = regions[region_name][quantity]["mean"]
                    right = regions["ordinary_background"][quantity]["mean"]
                    expected_contrast = (
                        None
                        if left is None or right is None
                        else float(left) - float(right)
                    )
                    _equal(
                        f"{public_mode} level {index + 1} "
                        f"{contrast_group}.{contrast_name}",
                        region_level[contrast_group][contrast_name],
                        expected_contrast,
                    )
            if index in selected:
                _equal(
                    f"{public_mode} knocked factor max deviation",
                    factor_level["max_abs_factor_minus_one"],
                    0.0,
                )
                for region_name in regions:
                    _require_unit_factor_distribution(
                        regions[region_name]["factor"],
                        (
                            f"{public_mode}.level{index + 1}."
                            f"{region_name}.factor"
                        ),
                    )

        maximum = max(mean_deviations)
        _equal(
            f"{public_mode} maximum mean factor deviation",
            factor_summary["maximum_level_mean_abs_factor_minus_one"],
            maximum,
        )
        _equal(
            f"{public_mode} nontrivial threshold",
            factor_summary["nontrivial_factor_threshold"],
            audit_core.NONTRIVIAL_FACTOR_MEAN_ABS,
        )
        _equal(
            f"{public_mode} nontrivial factor flag",
            factor_summary["nontrivial_factor_use"],
            maximum > audit_core.NONTRIVIAL_FACTOR_MEAN_ABS,
        )

        if public_mode == "full":
            full_regions = region_levels
            full_factors = factor_levels
        else:
            _require(
                full_regions is not None and full_factors is not None,
                "full factor summaries were not processed first",
            )
            for index, region_level in enumerate(region_levels):
                _equal(
                    f"{public_mode} level {index + 1} gate invariance",
                    region_level["regions"]["global"]["gate"],
                    full_regions[index]["regions"]["global"]["gate"],
                )
                for region_name in (
                    "target",
                    "hard_negative",
                    "ordinary_background",
                ):
                    _equal(
                        f"{public_mode} level {index + 1} "
                        f"{region_name} gate invariance",
                        region_level["regions"][region_name]["gate"],
                        full_regions[index]["regions"][region_name]["gate"],
                    )
                if index not in selected:
                    _equal(
                        f"{public_mode} level {index + 1} "
                        "unselected factor invariance",
                        factor_levels[index],
                        full_factors[index],
                    )
                    for region_name in region_level["regions"]:
                        _equal(
                            f"{public_mode} level {index + 1} "
                            f"{region_name} unselected factor invariance",
                            region_level["regions"][region_name]["factor"],
                            full_regions[index]["regions"][region_name][
                                "factor"
                            ],
                        )
    return {
        "status": "derived_summary_consistency_verified",
        "mode_count": len(audit_runner.PUBLIC_MODES),
        "level_count_per_mode": 4,
        "checks": [
            "knockout mode and selected-level metadata",
            "factor-summary/global-region duplicate values",
            "region partition counts and weighted moments",
            "factor and gate contrasts",
            "unit factors at knocked levels",
            "gate invariance across all six modes",
            "unselected factor invariance",
            "nontrivial-factor derived flag",
        ],
        "raw_gate_factor_maps_recomputed": False,
        "region_membership_recomputed": False,
    }


def _verify_repeat_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    repeat = report.get("repeat_inference")
    _require(isinstance(repeat, Mapping), "repeat-inference summary missing")
    _equal(
        "repeat-inference fields",
        set(repeat),
        {
            "status",
            "max_abs",
            "mean_abs",
            "max_abs_tolerance",
            "equivalent",
            "same_cache_content_sha256",
        },
    )
    _equal("repeat-inference status", repeat["status"], "complete")
    maximum = _finite_number(repeat["max_abs"], "repeat max_abs")
    mean = _finite_number(repeat["mean_abs"], "repeat mean_abs")
    tolerance = _finite_number(
        repeat["max_abs_tolerance"],
        "repeat max_abs tolerance",
    )
    _require(maximum >= 0.0 and mean >= 0.0, "repeat differences are negative")
    _require(mean <= maximum + FLOAT_ATOL, "repeat mean exceeds maximum")
    _equal(
        "repeat tolerance",
        tolerance,
        audit_runner.REPEAT_MAX_ABS_TOLERANCE,
    )
    _equal("repeat equivalence derivation", repeat["equivalent"], maximum <= tolerance)
    _require(
        type(repeat["same_cache_content_sha256"]) is bool,
        "repeat cache-content flag is invalid",
    )
    if repeat["same_cache_content_sha256"]:
        _equal("identical repeat cache max_abs", maximum, 0.0)
        _equal("identical repeat cache mean_abs", mean, 0.0)
    return {
        "status": "reported_summary_internal_consistency_verified",
        "second_repeat_cache_available": False,
        "probability_differences_recomputed": False,
        "cache_content_sha_comparison_recomputed": False,
    }


def _verify_functional_gate(report: Mapping[str, Any]) -> None:
    repeat = report["repeat_inference"]
    qfg_difference = report["modes"]["qfg_off"]["comparison_to_full"][
        "output_difference"
    ]
    factor = report["modes"]["full"]["factor_summary"]
    expected = {
        "status": "complete",
        "repeat_inference_equivalent": repeat["equivalent"],
        "full_vs_qfg_off_functionally_different": (
            qfg_difference["functionally_different"]
        ),
        "nontrivial_factor_use": factor["nontrivial_factor_use"],
        "qfg_functionally_active": (
            repeat["equivalent"]
            and qfg_difference["functionally_different"]
            and factor["nontrivial_factor_use"]
        ),
        "performance_causal_claim_established": False,
    }
    _equal("functional gate", report.get("functional_gate"), expected)
    _equal(
        "claim boundary",
        report.get("claim_boundary"),
        {
            "diagnostic_only": True,
            "qfg_training_causal_contribution_supported": False,
            "paper_core_established_changed": False,
            "stability_claim_supported_changed": False,
            "official_test_claim": False,
        },
    )


def deep_verify_audit(
    report_path: Path = DEFAULT_SOURCE_REPORT,
    *,
    repo_root: Path = REPO_ROOT,
    parent_lock: Path | None = None,
    source_lock: Path | None = None,
    expected_context: audit_runner.FrozenAuditContext | None = None,
) -> dict[str, Any]:
    """Deep-check one report and return a deterministic verification object."""

    source, report, raw = _json_object(report_path, "F1 audit report")
    _equal("F1 report canonical bytes", raw, audit_runner.canonical_json_bytes(report))
    _equal(
        "F1 top-level fields",
        set(report),
        {
            "schema",
            "status",
            "scope",
            "official_test_accessed",
            "authority",
            "live_authority_required",
            "source_bindings",
            "execution_contract",
            "model_load_audit",
            "repeat_inference",
            "modes",
            "functional_gate",
            "claim_boundary",
            "write_once",
            "overwrite_forbidden",
        },
    )
    _equal("F1 report schema", report["schema"], audit_runner.REPORT_SCHEMA)
    _equal("F1 report status", report["status"], "complete")
    _equal(
        "F1 report scope",
        report["scope"],
        "internal_validation_same_checkpoint_counterfactual",
    )
    _equal("official-test boundary", report["official_test_accessed"], False)
    _equal("F1 write-once flag", report["write_once"], True)
    _equal("F1 overwrite flag", report["overwrite_forbidden"], True)
    context = _resolve_context(
        report,
        repo_root=Path(repo_root).resolve(),
        parent_lock=parent_lock,
        source_lock=source_lock,
        expected_context=expected_context,
    )
    _equal(
        "F1 source bindings",
        report["source_bindings"],
        _expected_source_bindings(context.repo_root),
    )
    _validate_execution_contract(report, context)
    caches, cache_bindings = _load_six_caches(source, report, context)
    fixed_status = _recompute_fixed_metrics(report, caches)
    fa_status, dynamic_limits = _verify_fa_budget_scans(
        report,
        caches,
        live_authority_required=context.live_authority_required,
    )
    pair_status = _verify_cache_pair_derivatives(report, caches)
    factor_status = _verify_factor_and_region_summaries(
        report,
        len(context.validation_ids),
    )
    repeat_status = _verify_repeat_summary(report)
    _verify_functional_gate(report)

    limitations = [
        {
            "field": "repeat_inference.max_abs/mean_abs/equivalent/"
            "same_cache_content_sha256",
            "reason": (
                "the second full-mode repeat cache and its content SHA were "
                "not persisted; only formula consistency can be checked"
            ),
        },
        {
            "field": "modes.*.factor_gate_region_statistics raw samples",
            "reason": (
                "per-image gate/factor maps and region masks were not "
                "persisted; duplicate summaries, partitions, moments, "
                "contrasts, and six-mode invariants are checked"
            ),
        },
        {
            "field": "modes.*.alpha_knockout runtime mutation/restoration",
            "reason": (
                "the report persists hashes and selected levels, not the "
                "four before/inside/after alpha tensors"
            ),
        },
        {
            "field": "model_load_audit and execution runtime assertions",
            "reason": (
                "cache artifacts cannot independently prove strict model "
                "loading, eval mode, inference mode, or loader runtime flags"
            ),
        },
        *(
            {"field": value.split(":", 1)[0], "reason": value}
            for value in dynamic_limits
        ),
    ]
    return {
        "schema": DEEP_SCHEMA,
        "status": "verified",
        "scope": "internal_validation_artifact_deep_verification",
        "official_test_accessed": False,
        "source_audit": {
            "path": str(source),
            "sha256": _sha256_file(source),
            "schema": report["schema"],
        },
        "verifier": {
            "path": Path(__file__).resolve().relative_to(
                context.repo_root
            ).as_posix(),
            "sha256": _sha256_file(Path(__file__).resolve()),
            "independent_of_runner_derivative_helpers": True,
        },
        "authority": report["authority"],
        "checks": {
            "cache_manifests_and_arrays": {
                "status": "fully_verified",
                "mode_count": len(caches),
                "bindings": cache_bindings,
            },
            "fixed_threshold_metrics": fixed_status,
            "fa_budget_scans": fa_status,
            "counterfactual_cache_derivatives": pair_status,
            "factor_gate_region_statistics": factor_status,
            "repeat_inference": repeat_status,
            "functional_gate": "derived_fields_recomputed",
        },
        "limitations": limitations,
        "no_invention_status": True,
        "gpu_used": False,
        "source_audit_modified": False,
        "write_once": True,
        "overwrite_forbidden": True,
    }


def _validate_separate_output(source_report: Path, output: Path) -> Path:
    requested = Path(output).expanduser()
    if requested.is_symlink():
        _fail(f"deep-verification output must not be a symlink: {requested}")
    destination = requested.resolve()
    source_root = Path(source_report).expanduser().resolve().parent
    _require(
        not destination.is_relative_to(source_root),
        "deep-verification output must be outside the source audit directory",
    )
    return destination


def _atomic_create(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"refusing to replace deep-verification output: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def write_deep_verification_once(
    report_path: Path = DEFAULT_SOURCE_REPORT,
    output: Path = DEFAULT_OUTPUT,
    *,
    repo_root: Path = REPO_ROOT,
    parent_lock: Path | None = None,
    source_lock: Path | None = None,
    expected_context: audit_runner.FrozenAuditContext | None = None,
) -> Path:
    destination = _validate_separate_output(report_path, output)
    result = deep_verify_audit(
        report_path,
        repo_root=repo_root,
        parent_lock=parent_lock,
        source_lock=source_lock,
        expected_context=expected_context,
    )
    _atomic_create(destination, canonical_json_bytes(result))
    verify_deep_verification(
        destination,
        report_path,
        repo_root=repo_root,
        parent_lock=parent_lock,
        source_lock=source_lock,
        expected_context=expected_context,
    )
    return destination


def verify_deep_verification(
    attestation_path: Path,
    report_path: Path = DEFAULT_SOURCE_REPORT,
    *,
    repo_root: Path = REPO_ROOT,
    parent_lock: Path | None = None,
    source_lock: Path | None = None,
    expected_context: audit_runner.FrozenAuditContext | None = None,
) -> dict[str, Any]:
    source, observed, raw = _json_object(
        attestation_path,
        "deep-verification attestation",
    )
    _equal(
        "deep-verification canonical bytes",
        raw,
        canonical_json_bytes(observed),
    )
    expected = deep_verify_audit(
        report_path,
        repo_root=repo_root,
        parent_lock=parent_lock,
        source_lock=source_lock,
        expected_context=expected_context,
    )
    _equal("deep-verification attestation", observed, expected)
    _equal("deep-verification path", source, Path(attestation_path).resolve())
    return observed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify-attestation", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parent-lock", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_only:
        result = deep_verify_audit(
            args.report,
            repo_root=args.repo_root,
            parent_lock=args.parent_lock,
            source_lock=args.source_lock,
        )
    elif args.write_once:
        output = write_deep_verification_once(
            args.report,
            args.output,
            repo_root=args.repo_root,
            parent_lock=args.parent_lock,
            source_lock=args.source_lock,
        )
        result = {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "write-once",
            "output": str(output),
            "output_sha256": _sha256_file(output),
            "verified": True,
            "gpu_used": False,
        }
    else:
        verify_deep_verification(
            args.output,
            args.report,
            repo_root=args.repo_root,
            parent_lock=args.parent_lock,
            source_lock=args.source_lock,
        )
        result = {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "verify-attestation",
            "output": str(args.output.resolve()),
            "output_sha256": _sha256_file(args.output),
            "verified": True,
            "gpu_used": False,
        }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "DEEP_SCHEMA",
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_REPORT",
    "DeepAuditVerificationError",
    "canonical_json_bytes",
    "deep_verify_audit",
    "main",
    "parse_args",
    "recompute_formal_fa_budget_scan",
    "recompute_paired_image_bootstrap",
    "verify_deep_verification",
    "write_deep_verification_once",
]


if __name__ == "__main__":
    main()
