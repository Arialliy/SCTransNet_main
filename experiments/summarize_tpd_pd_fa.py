#!/usr/bin/env python3
"""Strictly audit and aggregate four validation-only TPD Pd--Fa sweeps."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("original", "progressive", "spd", "tpd")
MANIFEST_VARIANTS = ("original", "progressive", "tpd", "spd")
COUNT_KEYS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
POINT_NUMERIC_KEYS = (
    "val_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    *COUNT_KEYS,
    "threshold",
)
VALIDATION_METRIC_KEYS = tuple(
    key for key in POINT_NUMERIC_KEYS if key != "threshold"
)
GT_INVARIANT_COUNT_KEYS = (
    "target_count",
    "tiny_target_count",
    "valid_pixel_count",
)
SPLIT_HASH_KEYS = (
    "full_internal_train_sha256",
    "full_internal_val_sha256",
    "used_train_sha256",
    "used_val_sha256",
)
CRITICAL_PROTOCOL_ARGUMENTS = (
    "dataset",
    "dataset_dir",
    "epochs",
    "batch_size",
    "patch_size",
    "workers",
    "seed",
    "split_seed",
    "val_fraction",
    "eval_every",
    "base_lr",
    "min_lr",
    "warmup_epochs",
    "threshold",
    "match_radius",
    "tiny_area",
    "amp",
    "max_train_images",
    "max_val_images",
)
PROTOCOL_CONTRACT_KEYS = {
    "primary_selection_rule",
    "secondary_selection_rule",
    "checkpoint_policy",
    "loss",
    "optimizer",
    "lr_schedule",
    "torch",
    "cuda_runtime",
    "device_name",
}
SPLIT_COUNT_KEYS = {
    "full_official_train_count",
    "full_internal_train_count",
    "full_internal_val_count",
    "used_train_count",
    "used_val_count",
}
REQUIRED_INTEGRITY_FLAGS = {
    "summary_complete",
    "metrics_complete_contiguous_finite",
    "metadata_consistent",
    "official_test_isolated",
    "split_hashes_recomputed_consistent",
    "checkpoint_role_epoch_metrics_consistent",
    "global_selection_keys_recomputed",
    "state_dict_strict_load",
    "fixed_threshold_object_metrics_exact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and aggregate four internal-validation Pd--Fa sweeps"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing aggregate outputs",
    )
    args = parser.parse_args()
    if args.expected_epochs < 1:
        parser.error("--expected-epochs must be >= 1")
    for option, value in (("--dataset", args.dataset), ("--run-name", args.run_name)):
        if (
            not value
            or value in {".", ".."}
            or Path(value).name != value
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            parser.error(f"{option} must be one safe directory name")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} is not a lowercase SHA-256 digest")
    return value


def read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def require_regular_file(path: Path, context: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{context} must be a regular non-symlink file: {path}")


def manifest_line(digest: str, name: str) -> str:
    require_sha256(digest, f"manifest digest for {name}")
    if (
        not name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ValueError(f"Unsupported manifest path characters: {name!r}")
    return f"{digest}  {name}\n"


def audit_source_manifests(
    root: Path, dataset: str, run_name: str
) -> Dict[str, Any]:
    comparison_candidate = root / dataset / "comparison"
    if not comparison_candidate.is_dir() or comparison_candidate.is_symlink():
        raise NotADirectoryError(
            f"Missing sealed non-symlink comparison directory: {comparison_candidate}"
        )
    comparison_dir = comparison_candidate.resolve()

    sweep_paths: Dict[str, Path] = {}
    for variant in MANIFEST_VARIANTS:
        candidate = (
            root
            / dataset
            / variant
            / run_name
            / "pd_fa_sweep_best.pth.json"
        )
        require_regular_file(candidate, f"{variant} sweep")
        sweep_paths[variant] = candidate.resolve()
    sweep_sha256: Dict[str, str] = {}
    sweep_manifest_text = ""
    for variant in MANIFEST_VARIANTS:
        path = sweep_paths[variant]
        require_regular_file(path, f"{variant} sweep")
        digest = file_sha256(path)
        sweep_sha256[variant] = digest
        sweep_manifest_text += manifest_line(digest, str(path))

    sweeps_manifest = comparison_dir / "SWEEPS.sha256"
    require_regular_file(sweeps_manifest, "sealed sweep manifest")
    if sweeps_manifest.read_bytes() != sweep_manifest_text.encode("utf-8"):
        raise ValueError(
            f"{sweeps_manifest} does not exactly match the four current sweeps"
        )

    base_names = (
        f"{run_name}.json",
        f"{run_name}.md",
        f"{run_name}.csv",
        "SWEEPS.sha256",
    )
    base_sha256: Dict[str, str] = {}
    complete_manifest_text = ""
    for name in base_names:
        path = comparison_dir / name
        require_regular_file(path, f"sealed comparison artifact {name}")
        digest = file_sha256(path)
        base_sha256[name] = digest
        complete_manifest_text += manifest_line(digest, name)

    complete_manifest = comparison_dir / "COMPLETE.sha256"
    require_regular_file(complete_manifest, "sealed comparison completion manifest")
    if complete_manifest.read_bytes() != complete_manifest_text.encode("utf-8"):
        raise ValueError(
            f"{complete_manifest} does not exactly match the sealed comparison set"
        )

    return {
        "comparison_dir": str(comparison_dir),
        "training_certificate_path": str(comparison_dir / f"{run_name}.json"),
        "training_certificate_sha256": base_sha256[f"{run_name}.json"],
        "sweeps_manifest_path": str(sweeps_manifest),
        "sweeps_manifest_sha256": base_sha256["SWEEPS.sha256"],
        "complete_manifest_path": str(complete_manifest),
        "complete_manifest_sha256": file_sha256(complete_manifest),
        "sealed_base_artifact_sha256": base_sha256,
        "sweep_paths": {
            variant: str(sweep_paths[variant]) for variant in MANIFEST_VARIANTS
        },
        "sweep_sha256": sweep_sha256,
    }


def require_mapping(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} is non-finite: {value!r}")
    return result


def assert_finite_tree(value: Any, context: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        finite_number(value, context)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}[{index}]")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assert_equal(actual: Any, expected: Any, context: str) -> None:
    if canonical(actual) != canonical(expected):
        raise ValueError(f"{context} mismatch: expected {expected!r}, got {actual!r}")


def assert_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{context} mismatch: expected {expected!r}, got {actual!r}")


def audit_point(
    value: Any, context: str, validation_count: int | None = None
) -> Dict[str, Any]:
    point = require_mapping(value, context)
    expected_keys = set(POINT_NUMERIC_KEYS)
    if set(point) != expected_keys:
        raise ValueError(
            f"{context} point fields differ: "
            f"missing={sorted(expected_keys - set(point))}, "
            f"extra={sorted(set(point) - expected_keys)}"
        )
    for key in POINT_NUMERIC_KEYS:
        finite_number(point[key], f"{context}.{key}")
    for key in COUNT_KEYS:
        if isinstance(point[key], bool) or not isinstance(point[key], int):
            raise ValueError(f"{context}.{key} must be an integer")
        if point[key] < 0:
            raise ValueError(f"{context}.{key} must be non-negative")
    threshold = float(point["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"{context}.threshold must lie in (0, 1)")
    for key in ("pd", "tiny_pd", "miou", "niou", "pixel_precision", "pixel_recall", "pixel_f1"):
        if not 0.0 <= float(point[key]) <= 1.0:
            raise ValueError(f"{context}.{key} must lie in [0, 1]")
    if float(point["fa"]) < 0.0 or float(point["false_objects_per_image"]) < 0.0:
        raise ValueError(f"{context} has a negative false-alarm metric")
    if point["matched_target_count"] > point["target_count"]:
        raise ValueError(f"{context} has matched_target_count > target_count")
    if point["matched_tiny_target_count"] > point["tiny_target_count"]:
        raise ValueError(f"{context} has matched_tiny_target_count > tiny_target_count")
    if point["tiny_target_count"] > point["target_count"]:
        raise ValueError(f"{context} has tiny_target_count > target_count")
    if point["matched_tiny_target_count"] > point["matched_target_count"]:
        raise ValueError(
            f"{context} has matched_tiny_target_count > matched_target_count"
        )
    if point["target_count"] < 1 or point["tiny_target_count"] < 1:
        raise ValueError(f"{context} requires positive target and tiny-target counts")
    if point["valid_pixel_count"] < 1:
        raise ValueError(f"{context}.valid_pixel_count must be positive")
    if (
        point["predicted_object_count"]
        != point["matched_target_count"] + point["unmatched_predicted_object_count"]
    ):
        raise ValueError(
            f"{context}: predicted objects do not equal matched plus unmatched objects"
        )
    assert_close(
        float(point["pd"]),
        point["matched_target_count"] / point["target_count"],
        f"{context}.pd/counts",
    )
    assert_close(
        float(point["tiny_pd"]),
        point["matched_tiny_target_count"] / point["tiny_target_count"],
        f"{context}.tiny_pd/counts",
    )
    if validation_count is not None:
        if validation_count < 1:
            raise ValueError(f"{context}: validation_count must be positive")
        assert_close(
            float(point["false_objects_per_image"]),
            point["unmatched_predicted_object_count"] / validation_count,
            f"{context}.false_objects_per_image/counts",
        )
    return point


def operating_point_key(point: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        float(point["pd"]),
        -float(point["fa"]),
        float(point["tiny_pd"]),
        float(point["miou"]),
        -abs(float(point["threshold"]) - 0.5),
    )


def recompute_budget_point(
    points: Sequence[Dict[str, Any]], budget: float
) -> Dict[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    return max(feasible, key=operating_point_key) if feasible else None


def expected_budget_key(budget: float) -> str:
    return f"{budget:.10g}"


def audit_artifact_hashes(
    sweep: Dict[str, Any], run_dir: Path, sweep_path: Path
) -> Dict[str, str]:
    audit = require_mapping(sweep.get("audit"), f"{sweep_path}.audit")
    recorded = require_mapping(
        audit.get("artifact_sha256"), f"{sweep_path}.audit.artifact_sha256"
    )
    expected_paths = {
        "protocol.json": run_dir / "protocol.json",
        "split.json": run_dir / "split.json",
        "summary.json": run_dir / "summary.json",
        "metrics.jsonl": run_dir / "metrics.jsonl",
        "checkpoint": run_dir / "best.pth.tar",
        "evaluator": REPO_ROOT / "experiments" / "evaluate_pd_fa_sweep.py",
    }
    verified: Dict[str, str] = {}
    for name, path in expected_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if recorded.get(name) != actual:
            raise ValueError(
                f"{sweep_path}: {name} SHA mismatch; recorded={recorded.get(name)!r}, actual={actual}"
            )
        verified[name] = actual
    if sweep.get("checkpoint_sha256") != verified["checkpoint"]:
        raise ValueError(f"{sweep_path}: top-level checkpoint SHA mismatch")
    return verified


def audit_fixed_checkpoint_evidence(
    sweep: Dict[str, Any],
    sweep_path: Path,
    checkpoint_epoch: int,
    checkpoint_metrics: Dict[str, Any],
    fixed_half: Dict[str, Any],
) -> None:
    evidence = require_mapping(
        sweep.get("fixed_threshold_0_5_checkpoint_audit"),
        f"{sweep_path}.fixed_threshold_0_5_checkpoint_audit",
    )
    exact_keys = list(
        dict.fromkeys(
            [
                "pd",
                "fa",
                "tiny_pd",
                "false_objects_per_image",
                *sorted(key for key in checkpoint_metrics if key.endswith("_count")),
            ]
        )
    )
    assert_equal(
        evidence.get("exact_match_keys"),
        exact_keys,
        f"{sweep_path}.fixed_threshold exact keys",
    )
    exact_matches = require_mapping(
        evidence.get("exact_matches"), f"{sweep_path}.fixed_threshold exact matches"
    )
    if set(exact_matches) != set(exact_keys):
        raise ValueError(f"{sweep_path}: fixed-threshold exact-match fields differ")
    for key in exact_keys:
        pair = require_mapping(
            exact_matches[key], f"{sweep_path}.fixed_threshold exact_matches.{key}"
        )
        assert_equal(
            pair,
            {"checkpoint": checkpoint_metrics[key], "sweep_0_5": fixed_half[key]},
            f"{sweep_path}.fixed_threshold exact_matches.{key}",
        )
        assert_equal(
            fixed_half[key],
            checkpoint_metrics[key],
            f"{sweep_path}.fixed_threshold checkpoint metric {key}",
        )

    expected_deltas = {
        key: float(fixed_half[key]) - float(checkpoint_metrics[key])
        for key in checkpoint_metrics
        if key in fixed_half and key not in exact_keys
    }
    reported_deltas = require_mapping(
        evidence.get("non_strict_numeric_deltas_sweep_minus_checkpoint"),
        f"{sweep_path}.fixed_threshold numeric deltas",
    )
    if set(reported_deltas) != set(expected_deltas):
        raise ValueError(f"{sweep_path}: fixed-threshold numeric-delta fields differ")
    for key, expected in expected_deltas.items():
        assert_close(
            finite_number(
                reported_deltas[key], f"{sweep_path}.fixed_threshold delta.{key}"
            ),
            expected,
            f"{sweep_path}.fixed_threshold delta.{key}",
        )
    assert_close(
        finite_number(
            evidence.get("max_abs_non_strict_numeric_delta"),
            f"{sweep_path}.fixed_threshold max delta",
        ),
        max((abs(value) for value in expected_deltas.values()), default=0.0),
        f"{sweep_path}.fixed_threshold max delta",
    )

    audit = require_mapping(sweep.get("audit"), f"{sweep_path}.audit")
    globally_recomputed = require_mapping(
        audit.get("globally_recomputed_selection"),
        f"{sweep_path}.audit.globally_recomputed_selection",
    )
    pd_primary = require_mapping(
        globally_recomputed.get("pd_primary"),
        f"{sweep_path}.audit.globally_recomputed_selection.pd_primary",
    )
    assert_equal(
        pd_primary.get("epoch"),
        checkpoint_epoch,
        f"{sweep_path}.globally recomputed Pd-primary epoch",
    )
    assert_equal(
        pd_primary.get("metrics"),
        checkpoint_metrics,
        f"{sweep_path}.globally recomputed Pd-primary metrics",
    )
    expected_key = [
        float(checkpoint_metrics["pd"]),
        -float(checkpoint_metrics["fa"]),
        float(checkpoint_metrics["tiny_pd"]),
        float(checkpoint_metrics["miou"]),
        -float(checkpoint_metrics["val_loss"]),
    ]
    assert_equal(
        pd_primary.get("key"),
        expected_key,
        f"{sweep_path}.globally recomputed Pd-primary key",
    )


def audit_sweep(
    root: Path,
    dataset: str,
    run_name: str,
    variant: str,
    expected_epochs: int,
) -> Dict[str, Any]:
    run_dir = (root / dataset / variant / run_name).resolve()
    sweep_path = run_dir / "pd_fa_sweep_best.pth.json"
    sweep = read_json_object(sweep_path)
    assert_finite_tree(sweep, str(sweep_path))

    expected_checkpoint = run_dir / "best.pth.tar"
    if sweep.get("variant") != variant or sweep.get("dataset") != dataset:
        raise ValueError(f"{sweep_path}: dataset/variant identity mismatch")
    if sweep.get("run_directory") != str(run_dir):
        raise ValueError(f"{sweep_path}: run_directory mismatch")
    if sweep.get("checkpoint") != str(expected_checkpoint):
        raise ValueError(f"{sweep_path}: checkpoint path mismatch")
    if sweep.get("checkpoint_role") != "best_validation_pd_primary":
        raise ValueError(f"{sweep_path}: checkpoint is not Pd-primary")
    if sweep.get("official_test_accessed") is not False:
        raise ValueError(f"{sweep_path}: official test isolation is not asserted")
    checkpoint_epoch = sweep.get("checkpoint_epoch")
    if (
        isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, int)
        or not 1 <= checkpoint_epoch <= expected_epochs
    ):
        raise ValueError(f"{sweep_path}: invalid checkpoint_epoch")
    seed = sweep.get("seed")
    split_seed = sweep.get("split_seed")
    validation_count = sweep.get("validation_count")
    tiny_area = sweep.get("tiny_area")
    for name, value in (
        ("seed", seed),
        ("split_seed", split_seed),
        ("validation_count", validation_count),
        ("tiny_area", tiny_area),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{sweep_path}: {name} must be an integer")
    if validation_count < 1 or tiny_area < 1:
        raise ValueError(f"{sweep_path}: validation_count/tiny_area must be positive")
    match_radius = finite_number(
        sweep.get("match_radius"), f"{sweep_path}.match_radius"
    )
    if match_radius <= 0.0:
        raise ValueError(f"{sweep_path}: match_radius must be positive")
    validation_split_sha256 = require_sha256(
        sweep.get("validation_split_sha256"),
        f"{sweep_path}.validation_split_sha256",
    )
    checkpoint_sha256 = require_sha256(
        sweep.get("checkpoint_sha256"), f"{sweep_path}.checkpoint_sha256"
    )

    audit = require_mapping(sweep.get("audit"), f"{sweep_path}.audit")
    if audit.get("expected_epochs") != expected_epochs:
        raise ValueError(f"{sweep_path}: expected epoch count mismatch")
    if audit.get("metrics_event_count") != expected_epochs:
        raise ValueError(f"{sweep_path}: metrics event count mismatch")
    if audit.get("metrics_epoch_range") != [1, expected_epochs]:
        raise ValueError(f"{sweep_path}: metrics epoch range mismatch")
    if audit.get("summary_status") != "complete":
        raise ValueError(f"{sweep_path}: source run is not complete")
    if audit.get("selection_source") != "internal_validation_only":
        raise ValueError(f"{sweep_path}: checkpoint selection source is invalid")
    flags = require_mapping(
        audit.get("integrity_checks_passed"),
        f"{sweep_path}.audit.integrity_checks_passed",
    )
    if not REQUIRED_INTEGRITY_FLAGS <= set(flags):
        raise ValueError(
            f"{sweep_path}: missing integrity flags {sorted(REQUIRED_INTEGRITY_FLAGS - set(flags))}"
        )
    if any(value is not True for value in flags.values()):
        raise ValueError(f"{sweep_path}: one or more integrity flags did not pass")
    embedded_protocol = require_mapping(
        audit.get("protocol"), f"{sweep_path}.audit.protocol"
    )
    artifact_hashes = audit_artifact_hashes(sweep, run_dir, sweep_path)

    raw_points = sweep.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"{sweep_path}: points must be a non-empty array")
    points = [
        audit_point(
            point,
            f"{sweep_path}.points[{index}]",
            validation_count=validation_count,
        )
        for index, point in enumerate(raw_points)
    ]
    thresholds = [float(point["threshold"]) for point in points]
    if len(thresholds) != len(set(thresholds)):
        raise ValueError(f"{sweep_path}: duplicate thresholds")
    if thresholds != sorted(thresholds):
        raise ValueError(f"{sweep_path}: thresholds are not sorted")
    provenance = require_mapping(
        sweep.get("threshold_provenance"), f"{sweep_path}.threshold_provenance"
    )
    if provenance.get("total_unique_threshold_count") != len(points):
        raise ValueError(f"{sweep_path}: threshold count provenance mismatch")
    invariant_counts = {
        key: points[0][key] for key in GT_INVARIANT_COUNT_KEYS
    }
    for index, point in enumerate(points[1:], start=1):
        for key, expected in invariant_counts.items():
            assert_equal(
                point[key],
                expected,
                f"{sweep_path}.points[{index}].{key} invariant",
            )

    half_matches = [point for point in points if float(point["threshold"]) == 0.5]
    if len(half_matches) != 1:
        raise ValueError(f"{sweep_path}: expected exactly one threshold-0.5 point")
    fixed_half = audit_point(
        sweep.get("fixed_threshold_0_5"),
        f"{sweep_path}.fixed_threshold_0_5",
        validation_count=validation_count,
    )
    assert_equal(fixed_half, half_matches[0], f"{sweep_path}.fixed_threshold_0_5")
    raw_checkpoint_metrics = require_mapping(
        sweep.get("checkpoint_validation_metrics"),
        f"{sweep_path}.checkpoint_validation_metrics",
    )
    if set(raw_checkpoint_metrics) != set(VALIDATION_METRIC_KEYS):
        raise ValueError(
            f"{sweep_path}: checkpoint validation metric fields differ"
        )
    checkpoint_metrics = audit_point(
        {**raw_checkpoint_metrics, "threshold": 0.5},
        f"{sweep_path}.checkpoint_validation_metrics",
        validation_count=validation_count,
    )
    checkpoint_metrics.pop("threshold")
    audit_fixed_checkpoint_evidence(
        sweep,
        sweep_path,
        checkpoint_epoch,
        checkpoint_metrics,
        fixed_half,
    )

    configuration = require_mapping(
        sweep.get("threshold_configuration"), f"{sweep_path}.threshold_configuration"
    )
    raw_budgets = configuration.get("fa_budgets")
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError(f"{sweep_path}: fa_budgets must be a non-empty array")
    budgets = [finite_number(value, f"{sweep_path}.fa_budgets") for value in raw_budgets]
    if any(budget < 0.0 for budget in budgets) or len(budgets) != len(set(budgets)):
        raise ValueError(f"{sweep_path}: invalid or duplicate Fa budgets")
    reported_budget_points = require_mapping(
        sweep.get("best_points_under_fa_budget"),
        f"{sweep_path}.best_points_under_fa_budget",
    )
    expected_keys = [expected_budget_key(budget) for budget in budgets]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError(f"{sweep_path}: formatted Fa-budget keys collide")
    if set(reported_budget_points) != set(expected_keys):
        raise ValueError(f"{sweep_path}: Fa-budget keys mismatch")
    verified_budget_points: Dict[str, Dict[str, Any] | None] = {}
    for budget, key in zip(budgets, expected_keys):
        recomputed = recompute_budget_point(points, budget)
        reported = reported_budget_points[key]
        if reported is not None:
            reported = audit_point(
                reported,
                f"{sweep_path}.budget[{key}]",
                validation_count=validation_count,
            )
        assert_equal(reported, recomputed, f"{sweep_path}.budget[{key}]")
        verified_budget_points[key] = recomputed

    return {
        "variant": variant,
        "run_dir": str(run_dir),
        "sweep_path": str(sweep_path),
        "sweep_sha256": file_sha256(sweep_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_validation_metrics": checkpoint_metrics,
        "seed": seed,
        "split_seed": split_seed,
        "validation_count": validation_count,
        "validation_split_sha256": validation_split_sha256,
        "match_radius": match_radius,
        "tiny_area": tiny_area,
        "threshold_configuration": configuration,
        "evaluator_sha256": artifact_hashes["evaluator"],
        "split_artifact_sha256": artifact_hashes["split.json"],
        "metric_notes": embedded_protocol.get("metric_notes"),
        "artifact_sha256": artifact_hashes,
        "fixed_threshold_0_5": fixed_half,
        "budget_points": verified_budget_points,
        "budgets": budgets,
        "points": points,
        "invariant_counts": invariant_counts,
    }


def require_cross_sweep_consistency(sweeps: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    fields = (
        "seed",
        "split_seed",
        "validation_count",
        "validation_split_sha256",
        "match_radius",
        "tiny_area",
        "threshold_configuration",
        "evaluator_sha256",
        "split_artifact_sha256",
        "metric_notes",
        "budgets",
        "invariant_counts",
    )
    reference = sweeps[VARIANTS[0]]
    for field in fields:
        for variant in VARIANTS[1:]:
            assert_equal(
                sweeps[variant][field],
                reference[field],
                f"cross-sweep {field} ({variant} vs original)",
            )
    return {field: reference[field] for field in fields}


def require_exact_keys(
    mapping: Mapping[str, Any], expected: Iterable[str], context: str
) -> None:
    expected_set = set(expected)
    actual_set = set(mapping)
    if actual_set != expected_set:
        raise ValueError(
            f"{context} fields differ: missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )


def audit_training_certificate(
    source_seal: Mapping[str, Any],
    sweeps: Mapping[str, Dict[str, Any]],
    dataset: str,
    run_name: str,
    expected_epochs: int,
) -> Dict[str, Any]:
    certificate_path = Path(str(source_seal["training_certificate_path"]))
    certificate = read_json_object(certificate_path)
    assert_finite_tree(certificate, str(certificate_path))
    assert_equal(certificate.get("dataset"), dataset, f"{certificate_path}.dataset")
    assert_equal(certificate.get("run_name"), run_name, f"{certificate_path}.run_name")
    assert_equal(
        certificate.get("expected_epochs"),
        expected_epochs,
        f"{certificate_path}.expected_epochs",
    )
    if certificate.get("official_test_accessed") is not False:
        raise ValueError(f"{certificate_path}: official test isolation is not asserted")
    variant_run_names = require_mapping(
        certificate.get("variant_run_names"),
        f"{certificate_path}.variant_run_names",
    )
    assert_equal(
        variant_run_names,
        {variant: run_name for variant in VARIANTS},
        f"{certificate_path}.variant_run_names",
    )

    integrity = require_mapping(
        certificate.get("integrity_audit"), f"{certificate_path}.integrity_audit"
    )
    require_exact_keys(
        integrity,
        {
            "seed",
            "split_hashes",
            "shared_initialization_sha256",
            "normalization",
            "critical_protocol_arguments",
            "protocol_contract",
            "split_counts",
        },
        f"{certificate_path}.integrity_audit",
    )
    seed = integrity.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{certificate_path}.integrity_audit.seed must be an integer")
    split_hashes = require_mapping(
        integrity.get("split_hashes"), f"{certificate_path}.integrity_audit.split_hashes"
    )
    require_exact_keys(
        split_hashes, SPLIT_HASH_KEYS, f"{certificate_path}.integrity_audit.split_hashes"
    )
    for key in SPLIT_HASH_KEYS:
        require_sha256(
            split_hashes[key], f"{certificate_path}.integrity_audit.split_hashes.{key}"
        )
    assert_equal(
        certificate.get("validation_split_sha256"),
        split_hashes["used_val_sha256"],
        f"{certificate_path}.validation_split_sha256",
    )
    assert_equal(
        certificate.get("training_split_sha256"),
        split_hashes["used_train_sha256"],
        f"{certificate_path}.training_split_sha256",
    )
    shared_initialization_sha256 = require_sha256(
        integrity.get("shared_initialization_sha256"),
        f"{certificate_path}.integrity_audit.shared_initialization_sha256",
    )
    assert_equal(
        certificate.get("shared_initialization_sha256"),
        shared_initialization_sha256,
        f"{certificate_path}.shared_initialization_sha256",
    )

    normalization = require_mapping(
        integrity.get("normalization"), f"{certificate_path}.integrity_audit.normalization"
    )
    require_exact_keys(
        normalization, {"mean", "std"}, f"{certificate_path}.integrity_audit.normalization"
    )
    finite_number(normalization["mean"], f"{certificate_path}.normalization.mean")
    if finite_number(normalization["std"], f"{certificate_path}.normalization.std") <= 0:
        raise ValueError(f"{certificate_path}: normalization std must be positive")

    critical_arguments = require_mapping(
        integrity.get("critical_protocol_arguments"),
        f"{certificate_path}.integrity_audit.critical_protocol_arguments",
    )
    require_exact_keys(
        critical_arguments,
        CRITICAL_PROTOCOL_ARGUMENTS,
        f"{certificate_path}.integrity_audit.critical_protocol_arguments",
    )
    assert_equal(critical_arguments["dataset"], dataset, f"{certificate_path}.protocol dataset")
    assert_equal(
        critical_arguments["epochs"], expected_epochs, f"{certificate_path}.protocol epochs"
    )
    assert_equal(critical_arguments["seed"], seed, f"{certificate_path}.protocol seed")
    assert_equal(
        finite_number(
            critical_arguments["threshold"], f"{certificate_path}.protocol threshold"
        ),
        0.5,
        f"{certificate_path}.protocol checkpoint threshold",
    )

    protocol_contract = require_mapping(
        integrity.get("protocol_contract"),
        f"{certificate_path}.integrity_audit.protocol_contract",
    )
    require_exact_keys(
        protocol_contract,
        PROTOCOL_CONTRACT_KEYS,
        f"{certificate_path}.integrity_audit.protocol_contract",
    )
    split_counts = require_mapping(
        integrity.get("split_counts"), f"{certificate_path}.integrity_audit.split_counts"
    )
    require_exact_keys(
        split_counts, SPLIT_COUNT_KEYS, f"{certificate_path}.integrity_audit.split_counts"
    )
    for key in SPLIT_COUNT_KEYS:
        value = split_counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{certificate_path}.split_counts.{key} must be positive integer")
    assert_equal(
        split_counts["full_official_train_count"],
        split_counts["full_internal_train_count"]
        + split_counts["full_internal_val_count"],
        f"{certificate_path}.split_counts full partition",
    )
    if (
        split_counts["used_train_count"] > split_counts["full_internal_train_count"]
        or split_counts["used_val_count"] > split_counts["full_internal_val_count"]
    ):
        raise ValueError(f"{certificate_path}: used split counts exceed full split counts")

    checkpoint_sha256 = require_mapping(
        certificate.get("checkpoint_sha256"), f"{certificate_path}.checkpoint_sha256"
    )
    require_exact_keys(
        checkpoint_sha256, VARIANTS, f"{certificate_path}.checkpoint_sha256"
    )
    raw_rows = certificate.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(VARIANTS):
        raise ValueError(f"{certificate_path}.rows must contain exactly four rows")
    rows: Dict[str, Dict[str, Any]] = {}
    for index, value in enumerate(raw_rows):
        row = require_mapping(value, f"{certificate_path}.rows[{index}]")
        variant = row.get("variant")
        if variant not in VARIANTS or variant in rows:
            raise ValueError(f"{certificate_path}: invalid or duplicate row variant {variant!r}")
        rows[str(variant)] = row
    require_exact_keys(rows, VARIANTS, f"{certificate_path}.rows variants")

    row_metric_bindings = {
        "pd": "pd",
        "tiny_pd": "tiny_pd",
        "fa": "fa",
        "false_objects_per_image": "false_objects_per_image",
        "miou_at_pd_best": "miou",
        "niou_at_pd_best": "niou",
        "f1_at_pd_best": "pixel_f1",
    }
    bindings: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        sweep = sweeps[variant]
        row = rows[variant]
        sealed_path = source_seal["sweep_paths"][variant]
        sealed_sweep_sha256 = source_seal["sweep_sha256"][variant]
        assert_equal(sweep["sweep_path"], sealed_path, f"{variant} sealed sweep path")
        assert_equal(
            sweep["sweep_sha256"], sealed_sweep_sha256, f"{variant} sealed sweep SHA"
        )
        checkpoint_entry = require_mapping(
            checkpoint_sha256[variant],
            f"{certificate_path}.checkpoint_sha256.{variant}",
        )
        require_exact_keys(
            checkpoint_entry,
            {"best.pth.tar", "best_miou.pth.tar"},
            f"{certificate_path}.checkpoint_sha256.{variant}",
        )
        pd_checkpoint_sha256 = require_sha256(
            checkpoint_entry["best.pth.tar"],
            f"{certificate_path}.checkpoint_sha256.{variant}.best.pth.tar",
        )
        require_sha256(
            checkpoint_entry["best_miou.pth.tar"],
            f"{certificate_path}.checkpoint_sha256.{variant}.best_miou.pth.tar",
        )
        assert_equal(
            sweep["checkpoint_sha256"],
            pd_checkpoint_sha256,
            f"{variant} sweep/certificate checkpoint SHA",
        )
        assert_equal(
            row.get("best_checkpoint_sha256"),
            pd_checkpoint_sha256,
            f"{variant} row/certificate checkpoint SHA",
        )
        assert_equal(
            row.get("pd_best_epoch"),
            sweep["checkpoint_epoch"],
            f"{variant} Pd-primary epoch",
        )
        assert_equal(row.get("seed"), seed, f"{variant} certificate row seed")
        assert_equal(sweep["seed"], seed, f"{variant} sweep seed")
        assert_equal(
            sweep["split_seed"],
            critical_arguments["split_seed"],
            f"{variant} sweep split seed",
        )
        assert_equal(
            sweep["validation_split_sha256"],
            split_hashes["used_val_sha256"],
            f"{variant} validation split SHA",
        )
        assert_equal(
            sweep["validation_count"],
            split_counts["used_val_count"],
            f"{variant} validation count",
        )
        assert_equal(
            sweep["match_radius"],
            critical_arguments["match_radius"],
            f"{variant} match radius",
        )
        assert_equal(
            sweep["tiny_area"],
            critical_arguments["tiny_area"],
            f"{variant} tiny area",
        )
        assert_equal(row.get("run_dir"), sweep["run_dir"], f"{variant} run directory")
        checkpoint_metrics = sweep["checkpoint_validation_metrics"]
        for row_key, checkpoint_key in row_metric_bindings.items():
            assert_equal(
                row.get(row_key),
                checkpoint_metrics[checkpoint_key],
                f"{variant} row/checkpoint metric {row_key}",
            )
        bindings[variant] = {
            "checkpoint_sha256": pd_checkpoint_sha256,
            "checkpoint_epoch": sweep["checkpoint_epoch"],
            "row_metric_binding_passed": True,
        }

    return {
        "integrity_audit": integrity,
        "training_to_sweep_binding": bindings,
        "hardware_note": (
            "Hardware identity is provenance, not a required cross-variant equality; "
            "the sealed formal certificate records the available environment fields."
        ),
    }


def global_pareto(
    sweeps: Mapping[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int], set[Tuple[float, float]]]:
    owners: Dict[Tuple[float, float], set[str]] = {}
    samples: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for variant in VARIANTS:
        for point in sweeps[variant]["points"]:
            coordinate = (float(point["pd"]), float(point["fa"]))
            owners.setdefault(coordinate, set()).add(variant)
            samples.setdefault(coordinate, []).append(
                {
                    "variant": variant,
                    "threshold": point["threshold"],
                    "tiny_pd": point["tiny_pd"],
                    "miou": point["miou"],
                }
            )
    coordinates = list(owners)
    frontier: set[Tuple[float, float]] = set()
    for pd_value, fa_value in coordinates:
        dominated = any(
            other_fa <= fa_value
            and other_pd >= pd_value
            and (other_fa < fa_value or other_pd > pd_value)
            for other_pd, other_fa in coordinates
        )
        if not dominated:
            frontier.add((pd_value, fa_value))

    ordered = sorted(frontier, key=lambda coordinate: (coordinate[1], -coordinate[0]))
    rows = [
        {
            "pd": coordinate[0],
            "fa": coordinate[1],
            "owners": sorted(owners[coordinate]),
            "samples": sorted(
                samples[coordinate], key=lambda sample: (sample["variant"], sample["threshold"])
            ),
        }
        for coordinate in ordered
    ]
    owner_counts = {
        variant: sum(variant in owners[coordinate] for coordinate in ordered)
        for variant in VARIANTS
    }
    return rows, owner_counts, frontier


def point_row(
    section: str,
    budget: str,
    variant: str,
    checkpoint_epoch: Any,
    point: Dict[str, Any] | None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "section": section,
        "fa_budget": budget,
        "variant": variant,
        "checkpoint_epoch": checkpoint_epoch,
        "available": point is not None,
    }
    for key in POINT_NUMERIC_KEYS:
        row[key] = "" if point is None else point[key]
    return row


def csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_report(
    dataset: str,
    run_name: str,
    sweeps: Mapping[str, Dict[str, Any]],
    budget_keys: Sequence[str],
    pareto_rows: Sequence[Dict[str, Any]],
    owner_counts: Mapping[str, int],
) -> str:
    lines = [
        f"# {dataset} Pd–Fa comparison — {run_name}",
        "",
        "Internal validation only; every curve uses that variant's Pd-primary `best.pth.tar`.",
        "No AUC is computed, and this aggregate does not make the TPD mainline decision.",
        "",
        "## Fixed threshold 0.5",
        "",
        "| Variant | Checkpoint epoch | Threshold | Pd ↑ | tiny-Pd ↑ | Fa ↓ | mIoU ↑ | Matched targets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        point = sweeps[variant]["fixed_threshold_0_5"]
        lines.append(
            f"| {variant} | {sweeps[variant]['checkpoint_epoch']} | {fmt(point['threshold'], 6)} | "
            f"{fmt(point['pd'])} | {fmt(point['tiny_pd'])} | {fmt(point['fa'], 8)} | "
            f"{fmt(point['miou'])} | {point['matched_target_count']}/{point['target_count']} |"
        )

    for key in budget_keys:
        lines += [
            "",
            f"## Fa budget ≤ {key}",
            "",
            "Discrete-grid operating point: Pd, then lower actual Fa, tiny-Pd, mIoU, and proximity to threshold 0.5.",
            "",
            "| Variant | Threshold | Pd ↑ | tiny-Pd ↑ | Actual Fa ↓ | mIoU ↑ | Matched targets |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for variant in VARIANTS:
            point = sweeps[variant]["budget_points"][key]
            if point is None:
                lines.append(f"| {variant} | NA | NA | NA | NA | NA | NA |")
            else:
                lines.append(
                    f"| {variant} | {fmt(point['threshold'], 9)} | {fmt(point['pd'])} | "
                    f"{fmt(point['tiny_pd'])} | {fmt(point['fa'], 10)} | {fmt(point['miou'])} | "
                    f"{point['matched_target_count']}/{point['target_count']} |"
                )

    lines += [
        "",
        "## Joint sampled discrete Pd–Fa Pareto frontier",
        "",
        "A coordinate is dominated when another coordinate has no larger Fa and no smaller Pd, with at least one strict inequality. Identical `(Pd, Fa)` coordinates retain all owners.",
        "",
        f"Unique sampled frontier coordinates: **{len(pareto_rows)}**.",
        "",
        "| Variant | Owned frontier coordinates |",
        "|---|---:|",
    ]
    for variant in VARIANTS:
        lines.append(f"| {variant} | {owner_counts[variant]} |")
    lines += [
        "",
        "| Pd ↑ | Fa ↓ | Owners |",
        "|---:|---:|---|",
    ]
    for row in pareto_rows:
        lines.append(
            f"| {fmt(row['pd'])} | {fmt(row['fa'], 10)} | {', '.join(row['owners'])} |"
        )
    lines += ["", "## Sweep provenance", ""]
    for variant in VARIANTS:
        lines.append(
            f"- `{variant}`: epoch {sweeps[variant]['checkpoint_epoch']}; "
            f"checkpoint `{sweeps[variant]['checkpoint_sha256']}`; "
            f"sweep `{sweeps[variant]['sweep_sha256']}`."
        )
    return "\n".join(lines) + "\n"


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_committed_output_set(
    payload_paths: Sequence[Path], marker_path: Path, marker_text: str
) -> None:
    require_regular_file(marker_path, "Pd--Fa completion marker")
    if marker_path.read_bytes() != marker_text.encode("utf-8"):
        raise ValueError(f"{marker_path} content does not match the expected output set")
    for path in payload_paths:
        require_regular_file(path, f"committed Pd--Fa output {path.name}")
    expected = "".join(
        manifest_line(file_sha256(path), path.name) for path in payload_paths
    )
    if expected != marker_text:
        raise ValueError(f"{marker_path} hashes do not match committed outputs")


def write_outputs_guarded(
    payloads: Mapping[Path, str],
    marker_path: Path,
    overwrite: bool,
    root: Path,
    dataset: str,
    run_name: str,
    source_snapshot: Mapping[str, Any],
) -> None:
    payload_paths = list(payloads)
    if not payload_paths:
        raise ValueError("No aggregate outputs were provided")
    output_dir = marker_path.parent
    if any(path.parent != output_dir for path in payload_paths):
        raise ValueError("All aggregate outputs and marker must share one directory")
    if marker_path in payloads:
        raise ValueError("Completion marker must not be a data payload")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = (root / dataset / f".pd_fa_{run_name}.aggregate.lock").resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another Pd--Fa aggregator holds {lock_path}") from exc

        current_source = audit_source_manifests(root, dataset, run_name)
        assert_equal(current_source, source_snapshot, "pre-write sealed source snapshot")
        targets = [*payload_paths, marker_path]
        existing = [
            str(path) for path in targets if path.exists() or path.is_symlink()
        ]
        if existing and not overwrite:
            raise FileExistsError(
                "Refusing to overwrite or recover existing aggregate outputs without "
                f"--overwrite: {existing}"
            )
        for path in targets:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(
                    f"Refusing to replace non-regular or symlink aggregate target: {path}"
                )

        stage_dir = Path(
            tempfile.mkdtemp(prefix=f".pd_fa_{run_name}.staging.", dir=output_dir)
        )
        try:
            staged_paths: List[Path] = []
            for destination, content in payloads.items():
                staged = stage_dir / destination.name
                with staged.open("x", encoding="utf-8", newline="") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                staged_paths.append(staged)
            marker_text = "".join(
                manifest_line(file_sha256(path), path.name) for path in staged_paths
            )
            staged_marker = stage_dir / marker_path.name
            with staged_marker.open("x", encoding="utf-8", newline="") as handle:
                handle.write(marker_text)
                handle.flush()
                os.fsync(handle.fileno())
            verify_committed_output_set(staged_paths, staged_marker, marker_text)
            fsync_directory(stage_dir)

            current_source = audit_source_manifests(root, dataset, run_name)
            assert_equal(
                current_source, source_snapshot, "pre-publish sealed source snapshot"
            )
            if marker_path.exists():
                invalidated_marker = stage_dir / f".previous.{marker_path.name}"
                os.replace(marker_path, invalidated_marker)
                fsync_directory(output_dir)
            for destination, staged in zip(payload_paths, staged_paths):
                os.replace(staged, destination)

            current_source = audit_source_manifests(root, dataset, run_name)
            assert_equal(
                current_source, source_snapshot, "pre-commit sealed source snapshot"
            )
            os.replace(staged_marker, marker_path)
            try:
                fsync_directory(output_dir)
                verify_committed_output_set(payload_paths, marker_path, marker_text)
                current_source = audit_source_manifests(root, dataset, run_name)
                assert_equal(
                    current_source,
                    source_snapshot,
                    "post-commit sealed source snapshot",
                )
            except Exception:
                if marker_path.exists() or marker_path.is_symlink():
                    invalidated_marker = stage_dir / f".failed.{marker_path.name}"
                    os.replace(marker_path, invalidated_marker)
                    fsync_directory(output_dir)
                raise
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_seal = audit_source_manifests(root, args.dataset, args.run_name)
    sweeps = {
        variant: audit_sweep(
            root, args.dataset, args.run_name, variant, args.expected_epochs
        )
        for variant in VARIANTS
    }
    common = require_cross_sweep_consistency(sweeps)
    training_certificate = audit_training_certificate(
        source_seal,
        sweeps,
        args.dataset,
        args.run_name,
        args.expected_epochs,
    )
    budgets = [float(value) for value in common["budgets"]]
    budget_keys = [expected_budget_key(value) for value in budgets]
    pareto_rows, owner_counts, pareto_coordinates = global_pareto(sweeps)
    pareto_owners_by_coordinate = {
        (float(row["pd"]), float(row["fa"])): ";".join(row["owners"])
        for row in pareto_rows
    }
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(str(source_seal["comparison_dir"]))
    )
    stem = f"pd_fa_{args.run_name}"
    marker_path = output_dir / f"{stem}.COMPLETE.sha256"

    operating_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        sweep = sweeps[variant]
        operating_rows.append(
            point_row(
                "fixed_threshold_0_5",
                "",
                variant,
                sweep["checkpoint_epoch"],
                sweep["fixed_threshold_0_5"],
            )
        )
        for key in budget_keys:
            operating_rows.append(
                point_row(
                    "fa_budget",
                    key,
                    variant,
                    sweep["checkpoint_epoch"],
                    sweep["budget_points"][key],
                )
            )
        for point in sweep["points"]:
            coordinate = (float(point["pd"]), float(point["fa"]))
            curve_rows.append(
                {
                    "variant": variant,
                    "checkpoint_epoch": sweep["checkpoint_epoch"],
                    **{key: point[key] for key in POINT_NUMERIC_KEYS},
                    "global_pareto": coordinate in pareto_coordinates,
                    "pareto_owners": pareto_owners_by_coordinate.get(coordinate, ""),
                }
            )

    aggregate = {
        "schema_version": "tpd-pd-fa-aggregate-v2",
        "dataset": args.dataset,
        "run_name": args.run_name,
        "expected_epochs": args.expected_epochs,
        "official_test_accessed": False,
        "selection_source": "internal_validation_only",
        "checkpoint_role": "best_validation_pd_primary",
        "operating_point_rule": [
            "maximum Pd under Fa budget",
            "minimum actual Fa on Pd ties",
            "maximum tiny-Pd",
            "maximum mIoU",
            "minimum distance from threshold 0.5",
        ],
        "auc_computed": False,
        "mainline_decision_made": False,
        "output_commit": {
            "marker": str(marker_path),
            "semantics": (
                "The four output files are valid only when the completion marker "
                "exists and all listed SHA-256 digests verify."
            ),
        },
        "sealed_source_evidence": source_seal,
        "sealed_training_certificate": training_certificate,
        "common_provenance": common,
        "source_sweeps": {
            variant: {
                key: sweeps[variant][key]
                for key in (
                    "run_dir",
                    "sweep_path",
                    "sweep_sha256",
                    "checkpoint_epoch",
                    "checkpoint_sha256",
                    "artifact_sha256",
                )
            }
            for variant in VARIANTS
        },
        "fixed_threshold_0_5": {
            variant: sweeps[variant]["fixed_threshold_0_5"] for variant in VARIANTS
        },
        "operating_points_by_fa_budget": {
            key: {variant: sweeps[variant]["budget_points"][key] for variant in VARIANTS}
            for key in budget_keys
        },
        "global_pareto": {
            "scope": "joint_sampled_discrete_threshold_coordinates",
            "dominance_definition": (
                "coordinate A dominates B iff A.Fa <= B.Fa and A.Pd >= B.Pd, "
                "with at least one strict inequality; identical coordinates retain all owners"
            ),
            "unique_coordinate_count": len(pareto_rows),
            "owner_coordinate_counts": owner_counts,
            "coordinates": pareto_rows,
        },
        "aggregator_sha256": file_sha256(Path(__file__).resolve()),
        "integrity_checks_passed": {
            "four_sweeps_present": True,
            "source_sweeps_manifest_verified": True,
            "source_complete_manifest_verified": True,
            "source_artifact_hashes_current": True,
            "all_source_audit_flags_true": True,
            "sealed_recorded_training_protocol_audit_verified": True,
            "training_checkpoint_sweep_binding_verified": True,
            "split_manifest_byte_identical": True,
            "thresholds_unique_sorted_finite": True,
            "point_count_identities_verified": True,
            "ground_truth_counts_invariant": True,
            "fixed_threshold_curve_point_exact": True,
            "fixed_threshold_checkpoint_object_metrics_exact": True,
            "fixed_threshold_checkpoint_numeric_deltas_recomputed": True,
            "fa_budget_points_recomputed_exact": True,
            "pareto_coordinates_recomputed": True,
            "hardware_timing_not_used_as_performance_evidence": True,
        },
    }

    operating_fields = (
        "section",
        "fa_budget",
        "variant",
        "checkpoint_epoch",
        "available",
        *POINT_NUMERIC_KEYS,
    )
    curve_fields = (
        "variant",
        "checkpoint_epoch",
        *POINT_NUMERIC_KEYS,
        "global_pareto",
        "pareto_owners",
    )
    payloads = {
        output_dir / f"{stem}.json": json.dumps(
            aggregate, ensure_ascii=False, indent=2
        )
        + "\n",
        output_dir / f"{stem}.md": markdown_report(
            args.dataset,
            args.run_name,
            sweeps,
            budget_keys,
            pareto_rows,
            owner_counts,
        ),
        output_dir / f"{stem}_operating_points.csv": csv_text(
            operating_rows, operating_fields
        ),
        output_dir / f"{stem}_curves.csv": csv_text(curve_rows, curve_fields),
    }
    write_outputs_guarded(
        payloads,
        marker_path,
        args.overwrite,
        root,
        args.dataset,
        args.run_name,
        source_seal,
    )
    print(
        f"WROTE {output_dir / stem}.[json|md] and "
        f"{output_dir / stem}_[operating_points|curves].csv; "
        f"COMMITTED {marker_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
