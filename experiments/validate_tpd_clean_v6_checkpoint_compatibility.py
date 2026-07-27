#!/usr/bin/env python3
"""Read-only validator for the eight post-freeze compatibility sweeps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import accept_tpd_clean_v6_formal800_results as old_acceptance  # noqa: E402
from experiments import evaluate_pd_fa_sweep as base  # noqa: E402
from experiments import evaluate_tpd_clean_v6_pd_fa_checkpoint_compat as compat  # noqa: E402
from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v6_checkpoint_compatibility_validation_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON payload is not an object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        base.json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _argument_value(argv: Sequence[Any], option: str) -> str:
    values = [
        argv[index + 1]
        for index, value in enumerate(argv[:-1])
        if value == option
    ]
    _require(len(values) == 1, f"runtime argv must contain {option} exactly once")
    _require(isinstance(values[0], str), f"runtime argv value is invalid: {option}")
    return str(values[0])


def expected_sweep_jobs(
    candidate_root: Path = summary.DEFAULT_CANDIDATE_ROOT,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    candidate_root = Path(candidate_root)
    for seed in summary.SEEDS:
        for variant in summary.VARIANTS:
            run_dir = (
                candidate_root
                / summary.DATASET
                / variant
                / f"seed_{seed}_{summary.RUN_TAG}"
            )
            for role_name, spec in summary.ROLE_SPECS.items():
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "role": role_name,
                        "run_dir": run_dir,
                        "path": run_dir / spec["sweep"],
                    }
                )
    return jobs


def validate_compatibility_sweep(
    path: Path,
    *,
    run_dir: Path,
    variant: str,
    seed: int,
    role_name: str,
    source_lock_path: Path = compat.DEFAULT_COMPATIBILITY_SOURCE_LOCK,
) -> dict[str, Any]:
    """Validate adapter provenance against current immutable input files."""

    path = Path(path).resolve()
    run_dir = Path(run_dir).resolve()
    payload = _load_json(path)
    spec = summary.ROLE_SPECS[role_name]
    _require(payload.get("variant") == variant, "compat sweep variant differs")
    _require(payload.get("seed") == seed, "compat sweep seed differs")

    provenance = payload.get("threshold_provenance")
    fixed_audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    audit = payload.get("audit")
    _require(isinstance(provenance, Mapping), "threshold provenance is missing")
    _require(isinstance(fixed_audit, Mapping), "fixed checkpoint audit is missing")
    _require(isinstance(audit, Mapping), "sweep audit is missing")
    compat.validate_non_strict_numeric_deltas(fixed_audit)
    normalized_deltas = fixed_audit[
        "non_strict_numeric_deltas_sweep_minus_checkpoint"
    ]
    _require(
        fixed_audit.get("max_abs_non_strict_numeric_delta") == 0.0
        and all(value == 0.0 for value in normalized_deltas.values()),
        "normalized fixed checkpoint audit is not exact",
    )
    records = [
        provenance.get(compat.COMPATIBILITY_KEY),
        fixed_audit.get(compat.COMPATIBILITY_KEY),
        audit.get(compat.COMPATIBILITY_KEY),
    ]
    _require(
        all(isinstance(record, Mapping) for record in records),
        "compatibility provenance is missing from one or more required locations",
    )
    _require(records[0] == records[1] == records[2], "compatibility records differ")
    record = dict(records[0])
    _require(
        record.get("schema") == compat.COMPATIBILITY_SCHEMA,
        "compatibility record schema differs",
    )
    protocol = _load_json(run_dir / "protocol.json")
    protocol_environment = (
        protocol.get("run_identity", {})
        .get("training_contract", {})
        .get("environment", {})
    )
    expected_inference_environment = {
        key: protocol_environment.get(key)
        for key in compat.FORMAL_INFERENCE_ENVIRONMENT_KEYS
    }
    _require(
        record.get("seed") == seed
        and record.get("training_inference_environment")
        == expected_inference_environment
        and record.get("formal_inference_determinism")
        == expected_inference_environment,
        "formal inference environment provenance differs",
    )

    _, lock_sha = compat.validate_compatibility_source_lock(source_lock_path)
    expected_lock_path = str(Path(source_lock_path).resolve())
    lock_record = record.get("compatibility_source_lock")
    _require(
        isinstance(lock_record, Mapping)
        and lock_record.get("path") == expected_lock_path
        and lock_record.get("sha256") == lock_sha,
        "compatibility source-lock provenance differs",
    )
    wrapper_path = Path(compat.__file__).resolve()
    wrapper_sha = compat.sha256_file(wrapper_path)
    wrapper_record = record.get("actual_wrapper")
    _require(
        isinstance(wrapper_record, Mapping)
        and wrapper_record.get("path") == str(wrapper_path)
        and wrapper_record.get("sha256") == wrapper_sha
        and record.get("runtime_compatibility_sha256") == wrapper_sha,
        "runtime compatibility wrapper provenance differs",
    )
    frozen_sha = compat.sha256_file(compat.FROZEN_EVALUATOR)
    generic_sha = compat.sha256_file(compat.GENERIC_BASE_EVALUATOR)
    frozen_record = record.get("frozen_v6_evaluator")
    generic_record = record.get("generic_base_evaluator")
    _require(
        isinstance(frozen_record, Mapping)
        and frozen_record.get("path")
        == str(compat.FROZEN_EVALUATOR.resolve())
        and frozen_record.get("sha256") == frozen_sha
        and record.get("base_evaluator_sha256") == frozen_sha,
        "frozen V6 evaluator provenance differs",
    )
    _require(
        isinstance(generic_record, Mapping)
        and generic_record.get("path")
        == str(compat.GENERIC_BASE_EVALUATOR.resolve())
        and generic_record.get("sha256") == generic_sha,
        "generic base evaluator provenance differs",
    )

    checkpoint_path = (run_dir / spec["checkpoint"]).resolve()
    metrics_path = (run_dir / "metrics.jsonl").resolve()
    _require(
        checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
        "compat checkpoint is not a regular file",
    )
    _require(
        metrics_path.is_file() and not metrics_path.is_symlink(),
        "compat metrics log is not a regular file",
    )
    checkpoint_sha = compat.sha256_file(checkpoint_path)
    metrics_sha = compat.sha256_file(metrics_path)
    checkpoint_record = record.get("checkpoint")
    metrics_record = record.get("metrics_log")
    _require(
        isinstance(checkpoint_record, Mapping)
        and checkpoint_record.get("path") == str(checkpoint_path)
        and checkpoint_record.get("sha256") == checkpoint_sha,
        "checkpoint compatibility provenance differs",
    )
    _require(
        isinstance(metrics_record, Mapping)
        and metrics_record.get("path") == str(metrics_path)
        and metrics_record.get("sha256") == metrics_sha,
        "metrics compatibility provenance differs",
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    _require(isinstance(checkpoint, dict), "checkpoint payload is not a dictionary")
    checkpoint_epoch = checkpoint.get("epoch")
    _require(
        type(checkpoint_epoch) is int
        and checkpoint_epoch == payload.get("checkpoint_epoch")
        and checkpoint_epoch == checkpoint_record.get("epoch")
        and checkpoint_epoch == metrics_record.get("authoritative_epoch"),
        "compatibility checkpoint epoch differs",
    )
    _require(
        checkpoint.get("checkpoint_role") == spec["checkpoint_role"]
        and checkpoint.get("checkpoint_role") == payload.get("checkpoint_role")
        and checkpoint.get("checkpoint_role") == checkpoint_record.get("role"),
        "compatibility checkpoint role differs",
    )
    checkpoint_metrics = checkpoint.get("validation_metrics")
    _require(
        isinstance(checkpoint_metrics, Mapping),
        "checkpoint validation metrics are missing",
    )
    checkpoint_metrics = dict(checkpoint_metrics)
    _require(
        payload.get("checkpoint_validation_metrics") == checkpoint_metrics
        and record.get("original_checkpoint_validation_metrics")
        == checkpoint_metrics
        and record.get("original_checkpoint_metric_keys")
        == sorted(checkpoint_metrics),
        "original checkpoint metrics were not preserved",
    )
    original_five = {
        key: checkpoint_metrics.get(key) for key in compat.SELECTION_METRIC_KEYS
    }
    _require(
        record.get("original_five_selection_metrics") == original_five
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in original_five.values()
        ),
        "original five selection metrics differ",
    )

    events = base.load_complete_metrics(metrics_path, summary.EXPECTED_EPOCHS)
    event = events[checkpoint_epoch - 1]
    for key, value in checkpoint_metrics.items():
        _require(
            event.get(key) == value,
            f"checkpoint metric differs from authoritative epoch: {key}",
        )
    expected_audit_fields = {
        key: event[key]
        for key in ["false_objects_per_image", *sorted(
            field for field in event if field.endswith("_count")
        )]
    }
    _require(
        set(compat.REQUIRED_AUDIT_ONLY_FIELDS) <= set(expected_audit_fields),
        "authoritative epoch lacks required audit fields",
    )
    _require(
        record.get("audit_only_fields") == expected_audit_fields,
        "audit-only field values differ from metrics.jsonl",
    )
    expected_supplemented = sorted(
        key for key in expected_audit_fields if key not in checkpoint_metrics
    )
    expected_preexisting = sorted(
        key for key in expected_audit_fields if key in checkpoint_metrics
    )
    _require(
        record.get("supplemented_fields") == expected_supplemented
        and record.get("preexisting_audit_fields") == expected_preexisting,
        "audit-only field classification differs",
    )
    field_sources = record.get("audit_only_field_sources")
    _require(
        isinstance(field_sources, Mapping)
        and set(field_sources) == set(expected_audit_fields),
        "audit-only field source set differs",
    )
    fixed_point = payload.get("fixed_threshold_0_5")
    _require(isinstance(fixed_point, Mapping), "fixed point is missing")
    for key, value in expected_audit_fields.items():
        source = field_sources[key]
        expected_mode = (
            "preexisting_checkpoint_value_verified_against_metrics_jsonl"
            if key in checkpoint_metrics
            else "supplemented_from_metrics_jsonl_for_audit_only"
        )
        _require(
            isinstance(source, Mapping)
            and source.get("source") == "metrics.jsonl"
            and source.get("path") == str(metrics_path)
            and source.get("sha256") == metrics_sha
            and source.get("epoch") == checkpoint_epoch
            and source.get("field") == key
            and source.get("value") == value
            and source.get("mode") == expected_mode,
            f"audit-only field source differs: {key}",
        )
        matches = fixed_audit.get("exact_matches", {}).get(key)
        _require(
            fixed_point.get(key) == value
            and isinstance(matches, Mapping)
            and matches.get("checkpoint") == value
            and matches.get("sweep_0_5") == value,
            f"fixed audit compatibility value differs: {key}",
        )

    points = payload.get("points")
    _require(isinstance(points, list) and points, "compat sweep points are missing")
    normalization = record.get("threshold_invariant_val_loss_normalization")
    raw_fixed_audit = record.get("raw_fixed_threshold_checkpoint_audit")
    _require(
        isinstance(normalization, Mapping)
        and normalization.get("schema") == compat.VAL_LOSS_NORMALIZATION_SCHEMA
        and normalization.get("field") == "val_loss"
        and normalization.get("authoritative_source")
        == "checkpoint.validation_metrics.val_loss",
        "val_loss normalization provenance differs",
    )
    _require(
        isinstance(raw_fixed_audit, Mapping),
        "raw fixed checkpoint audit provenance is missing",
    )
    compat.validate_non_strict_numeric_deltas(raw_fixed_audit)
    normalized_audit_without_compat = copy.deepcopy(dict(fixed_audit))
    normalized_audit_without_compat.pop(compat.COMPATIBILITY_KEY, None)
    expected_normalized_audit = copy.deepcopy(dict(raw_fixed_audit))
    expected_normalized_audit[
        "non_strict_numeric_deltas_sweep_minus_checkpoint"
    ] = {"miou": 0.0, "val_loss": 0.0}
    expected_normalized_audit["max_abs_non_strict_numeric_delta"] = 0.0
    _require(
        normalized_audit_without_compat == expected_normalized_audit
        and record.get("raw_fixed_threshold_checkpoint_audit_sha256")
        == _canonical_sha256(raw_fixed_audit)
        and record.get("normalized_fixed_threshold_checkpoint_audit_sha256")
        == _canonical_sha256(normalized_audit_without_compat),
        "raw-to-normalized fixed audit provenance differs",
    )
    checkpoint_val_loss = checkpoint_metrics["val_loss"]
    raw_val_loss = normalization.get("raw_recomputed_value")
    delta = raw_fixed_audit[
        "non_strict_numeric_deltas_sweep_minus_checkpoint"
    ]["val_loss"]
    _require(
        isinstance(raw_val_loss, (int, float))
        and not isinstance(raw_val_loss, bool)
        and math.isfinite(float(raw_val_loss))
        and float(raw_val_loss) - float(checkpoint_val_loss) == float(delta)
        and normalization.get("normalized_checkpoint_value")
        == checkpoint_val_loss
        and normalization.get("raw_minus_checkpoint_delta") == delta
        and normalization.get("absolute_delta") == abs(float(delta))
        and normalization.get("absolute_delta_limit")
        == compat.NON_STRICT_NUMERIC_DELTA_LIMITS["val_loss"],
        "val_loss normalization values differ",
    )
    budgets = payload.get("best_points_under_fa_budget")
    _require(
        isinstance(budgets, Mapping)
        and all(
            isinstance(point, Mapping)
            and point.get("val_loss") == checkpoint_val_loss
            for point in budgets.values()
        )
        and all(
            isinstance(point, Mapping)
            and point.get("val_loss") == checkpoint_val_loss
            for point in points
        )
        and fixed_point.get("val_loss") == checkpoint_val_loss,
        "persisted val_loss was not normalized to the checkpoint",
    )
    reconstructed_raw_points = copy.deepcopy(points)
    for point in reconstructed_raw_points:
        point["val_loss"] = raw_val_loss
    _require(
        record.get("points_sha256") == _canonical_sha256(points),
        "compat sweep point digest differs",
    )
    _require(
        record.get("raw_recomputed_points_sha256")
        == _canonical_sha256(reconstructed_raw_points)
        == normalization.get("raw_points_sha256"),
        "raw recomputed sweep point digest differs",
    )
    reconstructed_raw_fixed = copy.deepcopy(fixed_point)
    reconstructed_raw_fixed["val_loss"] = raw_val_loss
    _require(
        normalization.get("raw_fixed_threshold_0_5_sha256")
        == _canonical_sha256(reconstructed_raw_fixed)
        and normalization.get("point_count") == len(points)
        and normalization.get("budget_point_count") == len(budgets),
        "raw fixed-point normalization provenance differs",
    )
    _require(
        record.get("checkpoint_validation_metrics_sha256")
        == _canonical_sha256(checkpoint_metrics),
        "checkpoint validation metric digest differs",
    )
    for flag in (
        "temporary_audit_copy_only",
        "checkpoint_unchanged",
        "metrics_log_unchanged",
        "checkpoint_validation_metrics_unchanged",
        "task_metric_points_unchanged",
        "points_changed_only_by_threshold_invariant_val_loss",
    ):
        _require(record.get(flag) is True, f"compatibility flag differs: {flag}")
    _require(
        record.get("points_unchanged")
        is (raw_val_loss == checkpoint_val_loss),
        "points_unchanged flag differs",
    )

    argv = record.get("actual_runtime_argv")
    _require(
        isinstance(argv, list)
        and len(argv) >= 3
        and isinstance(argv[0], str)
        and argv[0]
        and Path(str(argv[1])).resolve() == wrapper_path,
        "actual runtime argv prefix differs",
    )
    _require("--overwrite" not in argv, "compatibility runtime enabled overwrite")
    _require(
        Path(_argument_value(argv, "--run-dir")).resolve() == run_dir,
        "runtime run-dir differs",
    )
    _require(
        _argument_value(argv, "--checkpoint") == spec["checkpoint"],
        "runtime checkpoint argument differs",
    )
    _require(
        int(_argument_value(argv, "--expected-epochs"))
        == summary.EXPECTED_EPOCHS,
        "runtime expected epoch argument differs",
    )

    artifact_hashes = audit.get("artifact_sha256")
    _require(
        isinstance(artifact_hashes, Mapping)
        and set(artifact_hashes)
        == {
            "protocol.json",
            "split.json",
            "summary.json",
            "metrics.jsonl",
            "checkpoint",
            "evaluator",
        }
        and artifact_hashes.get("evaluator") == frozen_sha,
        "frozen artifact evaluator digest was not preserved",
    )
    return {
        "variant": variant,
        "seed": seed,
        "role": role_name,
        "path": str(path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "metrics_sha256": metrics_sha,
        "wrapper_sha256": wrapper_sha,
        "compatibility_source_lock_sha256": lock_sha,
        "supplemented_fields": expected_supplemented,
        "valid": True,
    }


def inspect_compatibility_sweeps(
    candidate_root: Path = summary.DEFAULT_CANDIDATE_ROOT,
) -> dict[str, Any]:
    try:
        _, lock_sha = compat.validate_compatibility_source_lock()
        lock_error: str | None = None
    except Exception as exc:
        lock_sha = None
        lock_error = f"{type(exc).__name__}: {exc}"
    results: list[dict[str, Any]] = []
    for job in expected_sweep_jobs(candidate_root):
        if not job["path"].is_file() or job["path"].is_symlink():
            results.append(
                {
                    "variant": job["variant"],
                    "seed": job["seed"],
                    "role": job["role"],
                    "path": str(job["path"].resolve()),
                    "status": "missing"
                    if not job["path"].exists()
                    else "invalid",
                    "error": None
                    if not job["path"].exists()
                    else "sweep path is not a regular file",
                }
            )
            continue
        try:
            validate_compatibility_sweep(
                job["path"],
                run_dir=job["run_dir"],
                variant=job["variant"],
                seed=job["seed"],
                role_name=job["role"],
            )
            status = "compatibility_valid"
            error = None
        except Exception as exc:
            status = "invalid"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "variant": job["variant"],
                "seed": job["seed"],
                "role": job["role"],
                "path": str(job["path"].resolve()),
                "status": status,
                "error": error,
            }
        )
    valid = sum(item["status"] == "compatibility_valid" for item in results)
    return {
        "schema": SCHEMA,
        "mode": "preflight",
        "candidate_root": str(Path(candidate_root).resolve()),
        "compatibility_source_lock_sha256": lock_sha,
        "compatibility_source_lock_error": lock_error,
        "expected_sweeps": len(results),
        "valid_sweeps": valid,
        "complete_and_compatibility_valid": (
            lock_error is None and valid == len(results)
        ),
        "results": results,
    }


def validate_all_compatibility_sweeps(
    candidate_root: Path = summary.DEFAULT_CANDIDATE_ROOT,
) -> dict[str, Any]:
    """Require all three old locks and all eight compatibility records."""

    training_lock, _ = summary._validate_current_training_contract()
    summary.validate_postprocess_source_lock()
    old_acceptance.validate_supplemental_source_lock()
    _, lock_sha = compat.validate_compatibility_source_lock()
    evaluator_sha = training_lock["source_sha256"][
        "experiments/evaluate_tpd_clean_v6_pd_fa.py"
    ]
    records: list[dict[str, Any]] = []
    for job in expected_sweep_jobs(candidate_root):
        summary.validate_existing_sweep(
            job["run_dir"],
            variant=job["variant"],
            seed=job["seed"],
            role_name=job["role"],
            evaluator_sha256=evaluator_sha,
        )
        records.append(
            validate_compatibility_sweep(
                job["path"],
                run_dir=job["run_dir"],
                variant=job["variant"],
                seed=job["seed"],
                role_name=job["role"],
            )
        )
    _require(len(records) == 8, "compatibility sweep matrix is not 8/8")
    return {
        "schema": SCHEMA,
        "mode": "verify",
        "candidate_root": str(Path(candidate_root).resolve()),
        "compatibility_source_lock_sha256": lock_sha,
        "expected_sweeps": 8,
        "valid_sweeps": len(records),
        "complete_and_compatibility_valid": True,
        "results": records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate post-freeze checkpoint-metric compatibility"
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=summary.DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = (
        inspect_compatibility_sweeps(args.candidate_root)
        if args.preflight
        else validate_all_compatibility_sweeps(args.candidate_root)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


__all__ = [
    "SCHEMA",
    "expected_sweep_jobs",
    "inspect_compatibility_sweeps",
    "main",
    "parse_args",
    "validate_all_compatibility_sweeps",
    "validate_compatibility_sweep",
]


if __name__ == "__main__":
    main()
