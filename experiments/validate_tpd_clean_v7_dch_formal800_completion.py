#!/usr/bin/env python3
"""Validate and publish the V7-DCH formal800 completion transaction.

This module is the read-only matrix authority used by the DCH summarizer.  It
does not run evaluation, repair an artifact, select a checkpoint, or invent a
missing result.  ``inspect_training_readiness`` is safe while training is
active; ``validate_completion_matrix`` is strict and succeeds only for the
fixed 4-run / 12-checkpoint / 8-sweep matrix with native 17-field metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as evaluator_core  # noqa: E402
from experiments import freeze_tpd_clean_v7_dch_source_locks as locks  # noqa: E402
from experiments import tpd_exact_epoch_journal as epoch_journal  # noqa: E402
from experiments.train_tpd_clean_v7_dch import (  # noqa: E402
    build_clean_v7_dch_model,
)


DATASET = "NUDT-SIRST"
VARIANTS = (
    "tpd_clean_v7_dch_full",
    "tpd_clean_v7_dch_capacity",
)
PRIMARY_VARIANT = VARIANTS[0]
CONTROL_VARIANT = VARIANTS[1]
SEEDS = (42, 3407)
RUN_TAG = "formal800_exact_fp32_2x5090_v1"
EXPECTED_EPOCHS = 800
EXPECTED_TRAIN_COUNT = 530
EXPECTED_VAL_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
EXPECTED_SPLIT_SEED = 20260722
LAST_FLOAT32_BELOW_ONE = 0.9999999403953552
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
VALIDATION_FIELDS = (
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
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
ROLE_SPECS: Mapping[str, Mapping[str, str]] = {
    "pd_primary": {
        "checkpoint": "best.pth.tar",
        "sweep": "pd_fa_sweep_best.pth.json",
        "checkpoint_role": "best_validation_pd_primary",
        "summary_epoch": "best_pd_epoch",
        "summary_metrics": "best_pd_validation_metrics",
    },
    "miou_primary": {
        "checkpoint": "best_miou.pth.tar",
        "sweep": "pd_fa_sweep_best_miou.pth.json",
        "checkpoint_role": "best_validation_miou_secondary",
        "summary_epoch": "best_miou_epoch",
        "summary_metrics": "best_miou_validation_metrics",
    },
}
CHECKPOINT_SPECS: Mapping[str, Mapping[str, str | int]] = {
    "best.pth.tar": {
        "checkpoint_role": "best_validation_pd_primary",
        "summary_epoch": "best_pd_epoch",
        "summary_metrics": "best_pd_validation_metrics",
    },
    "best_miou.pth.tar": {
        "checkpoint_role": "best_validation_miou_secondary",
        "summary_epoch": "best_miou_epoch",
        "summary_metrics": "best_miou_validation_metrics",
    },
    "last.pth.tar": {
        "checkpoint_role": "last_evaluated_epoch",
        "epoch": EXPECTED_EPOCHS,
    },
}
GPU_ASSIGNMENTS = {
    (PRIMARY_VARIANT, 42): (
        2,
        "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    ),
    (CONTROL_VARIANT, 42): (
        3,
        "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    ),
    (CONTROL_VARIANT, 3407): (
        2,
        "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    ),
    (PRIMARY_VARIANT, 3407): (
        3,
        "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    ),
}

DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
)
DEFAULT_REFERENCE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_pe_formal800_4x5090_v1"
)
DEFAULT_EVALUATOR = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v7_dch_pd_fa.py"
)
DEFAULT_TRAINING_SOURCE_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v7_dch_exact_source_lock.json"
)
DEFAULT_ACCEPTANCE_SOURCE_LOCK = (
    REPO_ROOT / locks.DEFAULT_LOCK_RELATIVES["acceptance"]
)
DEFAULT_SMOKE_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_clean_v7_dch_preflight_v1/smoke_reports"
)
SMOKE_VERIFICATION = (
    REPO_ROOT
    / "experiments/results/tpd_clean_v7_dch_preflight_v1"
    / "smoke_verification.json"
)
SPD_RUN = (
    DEFAULT_REFERENCE_ROOT
    / DATASET
    / "spd"
    / "seed_42_formal800_pd_fp32_4x5090_v1"
)
SPD_REFERENCE_FILES = (
    SPD_RUN / "best.pth.tar",
    SPD_RUN / "pd_fa_sweep_best.pth.json",
    SPD_RUN / "protocol.json",
    SPD_RUN / "split.json",
    SPD_RUN / "summary.json",
    SPD_RUN / "metrics.jsonl",
)
DEFAULT_OUTPUT_DIR = DEFAULT_CANDIDATE_ROOT / DATASET / "comparison"
JSON_OUTPUT_NAME = "tpd_clean_v7_dch_formal800_comparison.json"
MARKDOWN_OUTPUT_NAME = "tpd_clean_v7_dch_formal800_comparison.md"
MANIFEST_SCHEMA = "sctransnet_tpd_clean_v7_dch_completion_inputs_v1"
MANIFEST_NAME = "completion_inputs.json"
MARKER_NAME = "COMPLETE.sha256"


class IncompleteArtifact(ValueError):
    """A required DCH formal artifact is absent or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteArtifact(message)


def sha256_file(path: Path) -> str:
    path = Path(path)
    _require(
        path.is_file() and not path.is_symlink(),
        f"not a regular file: {path}",
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tree(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"{label}: non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    _require(False, f"{label}: non-JSON value")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label}: missing regular file: {path}",
    )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise IncompleteArtifact(f"{label}: invalid JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"{label}: expected object")
    _finite_tree(payload, label)
    return payload


def _run_directory(candidate_root: Path, variant: str, seed: int) -> Path:
    return (
        Path(candidate_root)
        / DATASET
        / variant
        / f"seed_{seed}_{RUN_TAG}"
    )


def _run_key(variant: str, seed: int) -> str:
    return f"{variant}/seed_{seed}"


def inspect_training_readiness(
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
) -> dict[str, Any]:
    """Inspect the four training runs without loading live checkpoints."""

    runs: dict[str, Any] = {}
    ready = True
    checkpoint_count = 0
    for seed in SEEDS:
        for variant in VARIANTS:
            directory = _run_directory(candidate_root, variant, seed)
            metrics_path = directory / "metrics.jsonl"
            summary_path = directory / "summary.json"
            row_count = 0
            contiguous = False
            native_fields = False
            if metrics_path.is_file() and not metrics_path.is_symlink():
                lines = metrics_path.read_text(encoding="utf-8").splitlines()
                row_count = len(lines)
                if row_count == EXPECTED_EPOCHS:
                    try:
                        rows = [json.loads(line) for line in lines]
                        contiguous = [
                            row.get("epoch") for row in rows
                        ] == list(range(1, EXPECTED_EPOCHS + 1))
                        native_fields = all(
                            set(VALIDATION_FIELDS).issubset(row)
                            for row in rows
                        )
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        contiguous = False
                        native_fields = False
            summary_complete = False
            if summary_path.is_file() and not summary_path.is_symlink():
                try:
                    value = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary_complete = (
                        isinstance(value, dict)
                        and value.get("status") == "complete"
                        and value.get("stored_validation_metrics")
                        == list(VALIDATION_FIELDS)
                    )
                except (OSError, json.JSONDecodeError):
                    summary_complete = False
            checkpoints = {
                name: (directory / name).is_file()
                and not (directory / name).is_symlink()
                for name in CHECKPOINT_SPECS
            }
            checkpoint_count += sum(checkpoints.values())
            run_ready = (
                directory.is_dir()
                and not directory.is_symlink()
                and row_count == EXPECTED_EPOCHS
                and contiguous
                and native_fields
                and summary_complete
                and all(checkpoints.values())
            )
            ready = ready and run_ready
            runs[_run_key(variant, seed)] = {
                "variant": variant,
                "seed": seed,
                "run_directory": str(directory.resolve()),
                "metrics_rows": row_count,
                "metrics_contiguous_1_to_800": contiguous,
                "native_17_fields_present": native_fields,
                "summary_complete": summary_complete,
                "checkpoints_present": checkpoints,
                "ready_for_sweep": run_ready,
            }
    return {
        "schema": "sctransnet_tpd_clean_v7_dch_postprocess_preflight_v1",
        "mode": "preflight",
        "candidate_family": "tpd_clean_v7_dch",
        "candidate_root": str(Path(candidate_root).resolve()),
        "expected_runs": 4,
        "observed_checkpoints": checkpoint_count,
        "formal_matrix_complete": ready,
        "ready": ready,
        "status": "complete" if ready else "incomplete",
        "gate_evaluated": False,
        "engineering_gate_passed": None,
        "runs": runs,
        "writes_performed": 0,
    }


def inspect_completion_matrix(
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
) -> dict[str, Any]:
    """Report whether all eight sweep files exist, without validating results."""

    training = inspect_training_readiness(candidate_root)
    sweep_count = 0
    sweep_files: dict[str, bool] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            directory = _run_directory(candidate_root, variant, seed)
            for role, spec in ROLE_SPECS.items():
                path = directory / str(spec["sweep"])
                present = path.is_file() and not path.is_symlink()
                sweep_count += int(present)
                sweep_files[f"{_run_key(variant, seed)}/{role}"] = present
    ready = training["formal_matrix_complete"] is True and sweep_count == 8
    return {
        **training,
        "schema": "sctransnet_tpd_clean_v7_dch_completion_preflight_v1",
        "formal_matrix_complete": training["formal_matrix_complete"],
        "observed_sweeps": sweep_count,
        "expected_sweeps": 8,
        "sweep_files": sweep_files,
        "ready": ready,
        "status": "complete" if ready else "incomplete",
    }


def _validation_metrics(
    value: Any,
    label: str,
    *,
    exact_fields: bool = True,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label}: expected metrics object")
    metrics = dict(value)
    fields = set(VALIDATION_FIELDS)
    if exact_fields:
        _require(set(metrics) == fields, f"{label}: 17-field schema differs")
    else:
        _require(fields.issubset(metrics), f"{label}: missing validation fields")
        metrics = {name: metrics[name] for name in VALIDATION_FIELDS}
    for name, item in metrics.items():
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item)),
            f"{label}.{name}: expected finite numeric",
        )
    _require(
        int(metrics["target_count"]) == EXPECTED_TARGET_COUNT,
        f"{label}: target_count differs",
    )
    _require(
        int(metrics["tiny_target_count"]) == EXPECTED_TINY_TARGET_COUNT,
        f"{label}: tiny_target_count differs",
    )
    expected_pd = (
        int(metrics["matched_target_count"]) / EXPECTED_TARGET_COUNT
    )
    _require(
        math.isclose(
            float(metrics["pd"]), expected_pd, rel_tol=0.0, abs_tol=1e-15
        ),
        f"{label}: Pd/count mismatch",
    )
    return metrics


def _load_metrics(path: Path, variant: str) -> list[dict[str, Any]]:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{variant}: metrics journal missing",
    )
    raw = path.read_text(encoding="utf-8").splitlines()
    _require(
        len(raw) == EXPECTED_EPOCHS,
        f"{variant}: expected 800 metrics rows, found {len(raw)}",
    )
    rows: list[dict[str, Any]] = []
    for epoch, line in enumerate(raw, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IncompleteArtifact(
                f"{variant}: invalid metrics row {epoch}: {exc}"
            ) from exc
        _require(isinstance(row, dict), f"{variant}: metrics row is not object")
        _finite_tree(row, f"{variant}.metrics[{epoch}]")
        _require(row.get("epoch") == epoch, f"{variant}: noncontiguous epochs")
        _require(row.get("variant") == variant, f"{variant}: row identity")
        _require(
            row.get("processed_train_samples") == EXPECTED_TRAIN_COUNT,
            f"{variant}: processed_train_samples differs at epoch {epoch}",
        )
        _validation_metrics(
            row,
            f"{variant}.metrics[{epoch}]",
            exact_fields=False,
        )
        rows.append(row)
    return rows


def _validate_split(split: Mapping[str, Any]) -> str:
    _require(split.get("dataset") == DATASET, "split dataset differs")
    _require(
        split.get("split_seed") == EXPECTED_SPLIT_SEED,
        "split seed differs",
    )
    _require(
        split.get("used_train_count") == EXPECTED_TRAIN_COUNT
        and split.get("used_val_count") == EXPECTED_VAL_COUNT,
        "split counts differ",
    )
    try:
        recomputed = evaluator_core.validate_identifier_manifest(dict(split))
    except (ValueError, TypeError, KeyError) as exc:
        raise IncompleteArtifact(f"split manifest differs: {exc}") from exc
    hashes = split.get("hashes")
    _require(hashes == recomputed, "split hashes differ from identifiers")
    return str(recomputed["used_val_sha256"])


def _strict_load_state(
    variant: str,
    seed: int,
    state_dict: Mapping[str, Any],
) -> None:
    try:
        model, metadata = build_clean_v7_dch_model(variant, seed)
        _require(
            metadata.get("variant") == variant,
            "DCH builder metadata variant differs",
        )
        model.load_state_dict(state_dict, strict=True)
    except IncompleteArtifact:
        raise
    except Exception as exc:
        raise IncompleteArtifact(
            f"{variant}/seed={seed}: checkpoint strict-load failed: {exc}"
        ) from exc


def _load_checkpoint(
    path: Path,
    *,
    variant: str,
    seed: int,
    protocol: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = path.name
    spec = CHECKPOINT_SPECS[name]
    _require(
        path.is_file() and not path.is_symlink(),
        f"checkpoint missing: {path}",
    )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise IncompleteArtifact(f"checkpoint cannot be loaded: {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{name}: checkpoint must be mapping")
    _require(payload.get("variant") == variant, f"{name}: variant differs")
    _require(payload.get("seed") == seed, f"{name}: seed differs")
    _require(payload.get("dataset") == DATASET, f"{name}: dataset differs")
    _require(
        payload.get("split_seed") == EXPECTED_SPLIT_SEED,
        f"{name}: split_seed differs",
    )
    _require(
        payload.get("checkpoint_role") == spec["checkpoint_role"],
        f"{name}: checkpoint role differs",
    )
    _require(
        payload.get("selection_source") == "internal_validation_only"
        and payload.get("official_test_accessed") is False,
        f"{name}: validation-only identity differs",
    )
    expected_epoch = (
        int(spec["epoch"])
        if "epoch" in spec
        else int(summary[str(spec["summary_epoch"])])
    )
    _require(payload.get("epoch") == expected_epoch, f"{name}: epoch differs")
    metrics = _validation_metrics(
        payload.get("validation_metrics"),
        f"{name}.validation_metrics",
    )
    event_metrics = {
        field: rows[expected_epoch - 1][field]
        for field in VALIDATION_FIELDS
    }
    _require(metrics == event_metrics, f"{name}: metrics journal differs")
    if "summary_metrics" in spec:
        _require(
            metrics == summary[str(spec["summary_metrics"])],
            f"{name}: summary metrics differ",
        )
    protocol_identity = protocol.get("run_identity")
    _require(
        isinstance(protocol_identity, Mapping)
        and payload.get("run_identity") == dict(protocol_identity),
        f"{name}: exact run identity differs",
    )
    state_dict = payload.get("state_dict")
    _require(isinstance(state_dict, Mapping), f"{name}: state_dict missing")
    _strict_load_state(variant, seed, state_dict)
    record = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "role": spec["checkpoint_role"],
        "epoch": expected_epoch,
        "validation_metrics": metrics,
        "strict_load": True,
        "native_17_fields_complete": True,
    }
    return record, payload


def _validate_point(value: Any, label: str) -> dict[str, Any]:
    point = _validation_metrics(value, label, exact_fields=False)
    raw = dict(value)
    threshold = raw.get("threshold")
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold))
        and 0.0 <= float(threshold) <= 1.0,
        f"{label}: threshold differs",
    )
    return {**raw, "threshold": float(threshold)}


def _sweep_key(point: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(point["pd"]),
        -float(point["fa"]),
        float(point["tiny_pd"]),
        float(point["miou"]),
        -abs(float(point["threshold"]) - 0.5),
    )


def _best_under_budget(
    points: Sequence[Mapping[str, Any]], budget: float
) -> Mapping[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    return max(feasible, key=_sweep_key) if feasible else None


def _validate_existing_sweep_payload(
    path: Path,
    *,
    run_dir: Path,
    variant: str,
    seed: int,
    role_name: str,
    checkpoint: Mapping[str, Any],
    validation_split_sha256: str,
    evaluator_path: Path,
) -> dict[str, Any]:
    spec = ROLE_SPECS[role_name]
    payload = _load_json(path, f"{variant}/seed={seed}/{role_name} sweep")
    _require(payload.get("variant") == variant, "sweep variant differs")
    _require(payload.get("seed") == seed, "sweep seed differs")
    _require(payload.get("dataset") == DATASET, "sweep dataset differs")
    _require(payload.get("official_test_accessed") is False, "sweep test flag")
    _require(
        payload.get("checkpoint_role") == spec["checkpoint_role"],
        "sweep checkpoint role differs",
    )
    _require(
        payload.get("checkpoint_epoch") == checkpoint["epoch"],
        "sweep checkpoint epoch differs",
    )
    _require(
        payload.get("checkpoint_sha256") == checkpoint["sha256"],
        "sweep checkpoint digest differs",
    )
    _require(
        Path(str(payload.get("checkpoint", ""))).resolve()
        == Path(checkpoint["path"]).resolve(),
        "sweep checkpoint path differs",
    )
    checkpoint_metrics = _validation_metrics(
        payload.get("checkpoint_validation_metrics"),
        "sweep checkpoint metrics",
    )
    _require(
        checkpoint_metrics == checkpoint["validation_metrics"],
        "sweep checkpoint metrics differ",
    )
    _require(
        payload.get("validation_split_sha256") == validation_split_sha256,
        "sweep validation split differs",
    )

    points_raw = payload.get("points")
    _require(isinstance(points_raw, list) and points_raw, "sweep points missing")
    points = [
        _validate_point(point, f"sweep.points[{index}]")
        for index, point in enumerate(points_raw)
    ]
    thresholds = [float(point["threshold"]) for point in points]
    _require(
        thresholds == sorted(set(thresholds)),
        "sweep thresholds are not sorted and unique",
    )
    _require(0.5 in thresholds, "sweep threshold 0.5 missing")
    _require(
        LAST_FLOAT32_BELOW_ONE in thresholds and thresholds[-1] == 1.0,
        "sweep closed-interval endpoints missing",
    )
    endpoint = points[-1]
    _require(
        int(endpoint["matched_target_count"]) == 0
        and float(endpoint["pd"]) == 0.0
        and float(endpoint["fa"]) == 0.0
        and float(endpoint["miou"]) == 0.0
        and int(endpoint["predicted_object_count"]) == 0
        and int(endpoint["unmatched_predicted_object_count"]) == 0,
        "threshold 1.0 is not the empty-prediction endpoint",
    )
    provenance = payload.get("threshold_provenance")
    expected_provenance = {
        "posthoc_endpoint_completion": False,
        "preregistered_endpoint_completion": True,
        "endpoint_protocol_stage": "before_formal_training",
        "closed_probability_interval": True,
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": 1.0,
        "upper_boundary_comparison": "prediction > threshold",
        "upper_boundary_semantics": "empty_prediction_pd0_fa0",
    }
    _require(isinstance(provenance, Mapping), "sweep provenance missing")
    for key, expected in expected_provenance.items():
        _require(
            provenance.get(key) == expected,
            f"sweep endpoint provenance differs: {key}",
        )

    fixed = _validate_point(
        payload.get("fixed_threshold_0_5"),
        "fixed threshold 0.5",
    )
    _require(float(fixed["threshold"]) == 0.5, "fixed threshold differs")
    _require(fixed in points, "fixed threshold point absent from sweep")
    for field in VALIDATION_FIELDS:
        _require(
            fixed[field] == checkpoint_metrics[field],
            f"fixed threshold metric differs: {field}",
        )
    fixed_audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    _require(
        isinstance(fixed_audit, Mapping)
        and float(
            fixed_audit.get("max_abs_non_strict_numeric_delta", math.inf)
        )
        == 0.0,
        "fixed threshold reproduction is not exact",
    )

    configuration = payload.get("threshold_configuration")
    _require(isinstance(configuration, Mapping), "threshold configuration missing")
    observed_budgets = [
        f"{float(value):.10g}"
        for value in configuration.get("fa_budgets", [])
    ]
    _require(observed_budgets == list(BUDGET_KEYS), "Fa budgets differ")
    budgets_raw = payload.get("best_points_under_fa_budget")
    _require(
        isinstance(budgets_raw, Mapping)
        and set(budgets_raw) == set(BUDGET_KEYS),
        "registered budget point set differs",
    )
    budgets: dict[str, dict[str, Any]] = {}
    for key in BUDGET_KEYS:
        _require(budgets_raw[key] is not None, f"null budget point: {key}")
        point = _validate_point(budgets_raw[key], f"budget {key}")
        optimum = _best_under_budget(points, float(key))
        _require(
            optimum is not None and point == optimum,
            f"budget {key} is not the registered optimum",
        )
        budgets[key] = point

    audit = payload.get("audit")
    _require(isinstance(audit, Mapping), "sweep audit missing")
    _require(
        audit.get("expected_epochs") == EXPECTED_EPOCHS
        and audit.get("metrics_event_count") == EXPECTED_EPOCHS
        and audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS]
        and audit.get("summary_status") == "complete"
        and audit.get("selection_source") == "internal_validation_only",
        "sweep completeness audit differs",
    )
    checks = audit.get("integrity_checks_passed")
    _require(
        isinstance(checks, Mapping)
        and checks
        and all(value is True for value in checks.values()),
        "sweep evaluator integrity checks failed",
    )
    artifact_hashes = audit.get("artifact_sha256")
    _require(isinstance(artifact_hashes, Mapping), "sweep artifact hashes missing")
    expected_artifacts = {
        "protocol.json": run_dir / "protocol.json",
        "split.json": run_dir / "split.json",
        "summary.json": run_dir / "summary.json",
        "metrics.jsonl": run_dir / "metrics.jsonl",
        "checkpoint": Path(checkpoint["path"]),
        "evaluator": evaluator_path,
    }
    _require(
        set(artifact_hashes) == set(expected_artifacts),
        "sweep artifact hash set differs",
    )
    for name, artifact in expected_artifacts.items():
        _require(
            artifact_hashes[name] == sha256_file(artifact),
            f"sweep artifact hash differs: {name}",
        )
    return {
        "checkpoint": copy.deepcopy(dict(checkpoint)),
        "sweep": str(path.resolve()),
        "sweep_sha256": sha256_file(path),
        "fixed_threshold_0_5": fixed,
        "budgets": budgets,
        "closed_interval": True,
        "preregistered_endpoint_provenance": True,
        "fixed_threshold_reproduction_exact": True,
        "native_17_fields_complete": True,
    }


def validate_existing_sweep(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
    role_name: str,
    evaluator_path: Path = DEFAULT_EVALUATOR,
) -> dict[str, Any]:
    """Strictly validate one existing DCH sweep without replacing it."""

    _require(variant in VARIANTS, "unknown DCH variant")
    _require(seed in SEEDS, "unknown DCH seed")
    _require(role_name in ROLE_SPECS, "unknown DCH checkpoint role")
    run_dir = Path(run_dir)
    expected = _run_directory(run_dir.parents[2], variant, seed)
    _require(run_dir.resolve() == expected.resolve(), "run directory differs")
    protocol = _load_json(run_dir / "protocol.json", "run protocol")
    summary = _load_json(run_dir / "summary.json", "run summary")
    split = _load_json(run_dir / "split.json", "run split")
    rows = _load_metrics(run_dir / "metrics.jsonl", variant)
    validation_split = _validate_split(split)
    checkpoint, _ = _load_checkpoint(
        run_dir / str(ROLE_SPECS[role_name]["checkpoint"]),
        variant=variant,
        seed=seed,
        protocol=protocol,
        summary=summary,
        rows=rows,
    )
    return _validate_existing_sweep_payload(
        run_dir / str(ROLE_SPECS[role_name]["sweep"]),
        run_dir=run_dir,
        variant=variant,
        seed=seed,
        role_name=role_name,
        checkpoint=checkpoint,
        validation_split_sha256=validation_split,
        evaluator_path=Path(evaluator_path),
    )


def validate_acceptance_source_lock(
    path: Path = DEFAULT_ACCEPTANCE_SOURCE_LOCK,
) -> tuple[dict[str, Any], str]:
    """Validate the current v4 DCH acceptance lock and evaluator binding."""

    try:
        payload, digest = locks.validate_source_lock("acceptance", path)
    except (FileNotFoundError, ValueError) as exc:
        raise IncompleteArtifact(f"DCH acceptance source lock differs: {exc}") from exc
    _require(
        payload.get("schema") == locks.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
        "DCH acceptance source lock is not the current v4 schema",
    )
    source_hashes = payload.get("source_sha256")
    relative = str(DEFAULT_EVALUATOR.relative_to(REPO_ROOT))
    _require(
        isinstance(source_hashes, Mapping)
        and source_hashes.get(relative) == sha256_file(DEFAULT_EVALUATOR),
        "acceptance lock does not bind the DCH evaluator",
    )
    return payload, digest


def _validate_gpu_run_artifacts(
    candidate_root: Path,
    protocol: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    expected_index, expected_uuid = GPU_ASSIGNMENTS[(variant, seed)]
    assignment_path = (
        Path(candidate_root)
        / "lane_assignments"
        / f"{variant}_seed{seed}.json"
    )
    assignment = _load_json(assignment_path, "lane assignment")
    expected_assignment = {
        "schema": "sctransnet_tpd_clean_v7_dch_lane_assignment_v1",
        "variant": variant,
        "seed": seed,
        "physical_gpu_index": expected_index,
        "physical_gpu_uuid": expected_uuid,
        "logical_device": "cuda:0",
        "run_directory": str(
            _run_directory(candidate_root, variant, seed).resolve()
        ),
    }
    _require(assignment == expected_assignment, "lane assignment differs")
    identity = protocol.get("run_identity")
    training_contract = (
        identity.get("training_contract")
        if isinstance(identity, Mapping)
        else None
    )
    environment = (
        training_contract.get("environment")
        if isinstance(training_contract, Mapping)
        else None
    )
    _require(isinstance(environment, Mapping), "run environment is missing")
    _require(
        environment.get("physical_gpu_index") == expected_index
        and environment.get("physical_gpu_uuid") == expected_uuid
        and environment.get("device_uuid") == expected_uuid
        and environment.get("cuda_visible_devices") == expected_uuid
        and environment.get("logical_device") == "cuda:0"
        and environment.get("physical_gpu_assignment_source")
        == "verified_worker_environment",
        "run physical GPU identity differs",
    )
    log_path = Path(candidate_root) / "logs" / f"{variant}_seed{seed}.log"
    _require(
        log_path.is_file() and not log_path.is_symlink(),
        f"worker log missing: {log_path}",
    )
    completion = (
        f"TPDCLEANV7DCH_2X_COMPLETE variant={variant} seed={seed} "
        f"physical_gpu={expected_index} gpu_uuid={expected_uuid} "
        "epochs=800 stored_validation_metrics=17"
    )
    _require(
        completion in log_path.read_text(encoding="utf-8", errors="strict"),
        "worker completion record differs",
    )
    return {
        "assignment": str(assignment_path.resolve()),
        "assignment_sha256": sha256_file(assignment_path),
        "worker_log": str(log_path.resolve()),
        "worker_log_sha256": sha256_file(log_path),
        "physical_gpu_index": expected_index,
        "physical_gpu_uuid": expected_uuid,
        "logical_device": "cuda:0",
    }


def _validate_exact_journal(run_dir: Path) -> dict[str, Any]:
    root = run_dir / "exact_journal"
    names = (
        "active.json",
        "slot_a.metrics.jsonl",
        "slot_a.exact.pth",
        "slot_b.metrics.jsonl",
        "slot_b.exact.pth",
    )
    for name in names:
        path = root / name
        _require(
            path.is_file() and not path.is_symlink(),
            f"exact journal input missing: {path}",
        )
    try:
        active = epoch_journal.ExactEpochJournal(root).load_active()
    except Exception as exc:
        raise IncompleteArtifact(f"exact journal differs: {exc}") from exc
    _require(
        active is not None and active.epoch == EXPECTED_EPOCHS,
        "exact journal active epoch differs",
    )
    return {
        "active_epoch": active.epoch,
        "active_slot": active.slot,
        "files": {
            name: {
                "path": str((root / name).resolve()),
                "sha256": sha256_file(root / name),
            }
            for name in names
        },
        "complete": True,
    }


def _validate_run(
    candidate_root: Path,
    *,
    variant: str,
    seed: int,
    evaluator_path: Path,
) -> dict[str, Any]:
    run_dir = _run_directory(candidate_root, variant, seed)
    _require(
        run_dir.is_dir() and not run_dir.is_symlink(),
        f"run directory missing: {run_dir}",
    )
    protocol = _load_json(run_dir / "protocol.json", "run protocol")
    split = _load_json(run_dir / "split.json", "run split")
    summary = _load_json(run_dir / "summary.json", "run summary")
    arguments = protocol.get("arguments")
    _require(isinstance(arguments, Mapping), "protocol arguments missing")
    _require(
        protocol.get("schema")
        == "sctransnet_tpd_clean_v7_dch_exact_entry_v1",
        "protocol schema differs",
    )
    _require(
        arguments.get("variant") == variant
        and arguments.get("seed") == seed
        and arguments.get("dataset") == DATASET
        and arguments.get("run_tag") == RUN_TAG
        and arguments.get("epochs") == EXPECTED_EPOCHS
        and arguments.get("eval_every") == 1
        and arguments.get("workers") == 0
        and arguments.get("amp") is False
        and arguments.get("eps") == 1e-6,
        "protocol formal identity differs",
    )
    _require(
        protocol.get("stored_validation_metrics")
        == list(VALIDATION_FIELDS)
        and protocol.get("official_test_accessed") is False,
        "protocol native validation schema differs",
    )
    _require(
        summary.get("schema")
        == "sctransnet_tpd_clean_v7_dch_completion_summary_v1"
        and summary.get("status") == "complete"
        and summary.get("variant") == variant
        and summary.get("seed") == seed
        and summary.get("dataset") == DATASET
        and summary.get("stored_validation_metrics")
        == list(VALIDATION_FIELDS)
        and summary.get("official_test_accessed") is False
        and summary.get("selection_source") == "internal_validation_only",
        "completion summary identity/schema differs",
    )
    for name in (
        "best_validation_metrics",
        "best_pd_validation_metrics",
        "best_miou_validation_metrics",
    ):
        summary[name] = _validation_metrics(summary.get(name), f"summary.{name}")
    _require(
        summary["best_validation_metrics"]
        == summary["best_pd_validation_metrics"],
        "summary Pd aliases differ",
    )
    validation_split = _validate_split(split)
    _require(
        summary.get("split_hashes") == split.get("hashes"),
        "summary split hashes differ",
    )
    rows = _load_metrics(run_dir / "metrics.jsonl", variant)

    checkpoints: dict[str, dict[str, Any]] = {}
    for name in CHECKPOINT_SPECS:
        checkpoints[name], _ = _load_checkpoint(
            run_dir / name,
            variant=variant,
            seed=seed,
            protocol=protocol,
            summary=summary,
            rows=rows,
        )
    roles: dict[str, Any] = {}
    for role_name, spec in ROLE_SPECS.items():
        roles[role_name] = _validate_existing_sweep_payload(
            run_dir / str(spec["sweep"]),
            run_dir=run_dir,
            variant=variant,
            seed=seed,
            role_name=role_name,
            checkpoint=checkpoints[str(spec["checkpoint"])],
            validation_split_sha256=validation_split,
            evaluator_path=evaluator_path,
        )
    gpu = _validate_gpu_run_artifacts(
        candidate_root,
        protocol,
        variant=variant,
        seed=seed,
    )
    journal = _validate_exact_journal(run_dir)
    return {
        "variant": variant,
        "seed": seed,
        "run_directory": str(run_dir.resolve()),
        "validation_split_sha256": validation_split,
        "protocol": {
            "path": str((run_dir / "protocol.json").resolve()),
            "sha256": sha256_file(run_dir / "protocol.json"),
        },
        "split": {
            "path": str((run_dir / "split.json").resolve()),
            "sha256": sha256_file(run_dir / "split.json"),
        },
        "summary": {
            "path": str((run_dir / "summary.json").resolve()),
            "sha256": sha256_file(run_dir / "summary.json"),
        },
        "metrics": {
            "path": str((run_dir / "metrics.jsonl").resolve()),
            "sha256": sha256_file(run_dir / "metrics.jsonl"),
            "rows": EXPECTED_EPOCHS,
        },
        "checkpoints": checkpoints,
        "roles": roles,
        "gpu": gpu,
        "exact_journal": journal,
        "native_17_fields": {
            "required": list(VALIDATION_FIELDS),
            "protocol": True,
            "metrics_rows": EXPECTED_EPOCHS,
            "summary": True,
            "checkpoints": 3,
            "sweeps": 2,
            "complete": True,
        },
    }


def validate_completion_matrix(
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    evaluator_path: Path = DEFAULT_EVALUATOR,
    acceptance_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the fixed DCH matrix and return summarizer-ready records."""

    candidate_root = Path(candidate_root)
    evaluator_path = Path(evaluator_path)
    _require(
        evaluator_path.is_file() and not evaluator_path.is_symlink(),
        f"DCH evaluator missing: {evaluator_path}",
    )
    acceptance_lock_sha256 = None
    acceptance_lock_schema = None
    if acceptance_lock_path is not None:
        acceptance_payload, acceptance_lock_sha256 = (
            validate_acceptance_source_lock(
                Path(acceptance_lock_path)
            )
        )
        acceptance_lock_schema = acceptance_payload["schema"]
    runs: dict[str, dict[str, Any]] = {}
    validation_splits: set[str] = set()
    for seed in SEEDS:
        for variant in VARIANTS:
            record = _validate_run(
                candidate_root,
                variant=variant,
                seed=seed,
                evaluator_path=evaluator_path,
            )
            runs[_run_key(variant, seed)] = record
            validation_splits.add(record["validation_split_sha256"])
    _require(len(validation_splits) == 1, "validation splits differ across runs")
    checkpoint_count = sum(len(run["checkpoints"]) for run in runs.values())
    sweep_count = sum(len(run["roles"]) for run in runs.values())
    _require(checkpoint_count == 12, "checkpoint matrix count differs")
    _require(sweep_count == 8, "sweep matrix count differs")
    integrity = {
        "four_runs_contiguous_800_epochs": True,
        "twelve_checkpoints_present_and_strict_load": True,
        "eight_closed_interval_sweeps": True,
        "model_split_protocol_evaluator_hashes_consistent": True,
        "fixed_threshold_reproduction_exact": True,
        "all_five_budgets_available": True,
        "preregistered_endpoint_provenance": True,
        "exact_epoch_journals_complete": True,
        "worker_logs_complete_gpu_mapped": True,
        "native_17_fields_complete": True,
    }
    return {
        "schema": "sctransnet_tpd_clean_v7_dch_completion_matrix_v1",
        "candidate_family": "tpd_clean_v7_dch",
        "status": "complete",
        "ready": True,
        "candidate_root": str(candidate_root.resolve()),
        "evaluator": str(evaluator_path.resolve()),
        "evaluator_sha256": sha256_file(evaluator_path),
        "acceptance_source_lock_schema": acceptance_lock_schema,
        "acceptance_source_lock_sha256": acceptance_lock_sha256,
        "validation_split_sha256": next(iter(validation_splits)),
        "run_count": 4,
        "checkpoint_count": checkpoint_count,
        "sweep_count": sweep_count,
        "validation_field_count": len(VALIDATION_FIELDS),
        "validation_fields": list(VALIDATION_FIELDS),
        "runs": runs,
        "integrity": integrity,
        "gate_evaluated": False,
        "engineering_gate_passed": None,
    }


def _summary_module() -> Any:
    try:
        return importlib.import_module(
            "experiments.summarize_tpd_clean_v7_dch_formal800"
        )
    except ImportError as exc:
        raise IncompleteArtifact(
            "DCH formal summarizer is not available"
        ) from exc


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise IncompleteArtifact(
            f"completion input lies outside repository: {resolved}"
        ) from exc


def _input_paths(
    matrix: Mapping[str, Any],
    *,
    training_source_lock: Path = DEFAULT_TRAINING_SOURCE_LOCK,
    acceptance_source_lock: Path = DEFAULT_ACCEPTANCE_SOURCE_LOCK,
) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = [
        ("training_source_lock", "source_lock", training_source_lock),
        ("acceptance_source_lock", "source_lock", acceptance_source_lock),
    ]
    for path in SPD_REFERENCE_FILES:
        records.append(
            (
                f"frozen_spd:{path.name}",
                "frozen_reference",
                path,
            )
        )
    for name in ("cpu_all.json", "gpu2_full.json", "gpu3_capacity.json"):
        records.append(
            (f"smoke:{name}", "smoke_report", DEFAULT_SMOKE_ROOT / name)
        )
    records.append(
        (
            f"smoke:{SMOKE_VERIFICATION.name}",
            "smoke_verification",
            SMOKE_VERIFICATION,
        )
    )
    for key, run in matrix["runs"].items():
        for artifact in ("protocol", "split", "summary", "metrics"):
            records.append(
                (
                    f"{key}:{artifact}",
                    "candidate_training",
                    Path(run[artifact]["path"]),
                )
            )
        for name, checkpoint in run["checkpoints"].items():
            records.append(
                (
                    f"{key}:{name}",
                    "candidate_checkpoint",
                    Path(checkpoint["path"]),
                )
            )
        for role, result in run["roles"].items():
            records.append(
                (
                    f"{key}:sweep:{role}",
                    "candidate_sweep",
                    Path(result["sweep"]),
                )
            )
        records.extend(
            [
                (
                    f"{key}:lane_assignment",
                    "lane_assignment",
                    Path(run["gpu"]["assignment"]),
                ),
                (
                    f"{key}:worker_log",
                    "worker_log",
                    Path(run["gpu"]["worker_log"]),
                ),
            ]
        )
        for name, item in run["exact_journal"]["files"].items():
            records.append(
                (
                    f"{key}:exact_journal/{name}",
                    "exact_journal",
                    Path(item["path"]),
                )
            )
    identifiers = [identifier for identifier, _, _ in records]
    paths = [path.resolve() for _, _, path in records]
    _require(len(identifiers) == len(set(identifiers)), "duplicate manifest id")
    _require(len(paths) == len(set(paths)), "duplicate manifest path")
    return records


def build_manifest(
    matrix: Mapping[str, Any],
    *,
    training_source_lock: Path = DEFAULT_TRAINING_SOURCE_LOCK,
    acceptance_source_lock: Path = DEFAULT_ACCEPTANCE_SOURCE_LOCK,
) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    for identifier, category, path in _input_paths(
        matrix,
        training_source_lock=training_source_lock,
        acceptance_source_lock=acceptance_source_lock,
    ):
        _require(
            path.is_file() and not path.is_symlink(),
            f"completion input is not regular: {path}",
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        inputs.append(
            {
                "id": identifier,
                "category": category,
                "path": _relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    artifact_runs: dict[str, Any] = {}
    for key, run in matrix["runs"].items():
        checkpoint_records = {
            name: {
                "path": _relative(Path(record["path"])),
                "sha256": record["sha256"],
                "role": record["role"],
                "epoch": record["epoch"],
                "native_17_fields_complete": record[
                    "native_17_fields_complete"
                ],
            }
            for name, record in run["checkpoints"].items()
        }
        sweep_records = {
            role: {
                "path": _relative(Path(record["sweep"])),
                "sha256": record["sweep_sha256"],
                "checkpoint_role": record["checkpoint"]["role"],
                "checkpoint_sha256": record["checkpoint"]["sha256"],
                "fixed_threshold": 0.5,
                "fa_budget_keys": list(BUDGET_KEYS),
                "native_17_fields_complete": record[
                    "native_17_fields_complete"
                ],
            }
            for role, record in run["roles"].items()
        }
        artifact_runs[key] = {
            "variant": run["variant"],
            "seed": run["seed"],
            "run_directory": _relative(Path(run["run_directory"])),
            "validation_split_sha256": run["validation_split_sha256"],
            "native_17_fields_complete": run["native_17_fields"]["complete"],
            "checkpoints": checkpoint_records,
            "sweeps": sweep_records,
        }
    return {
        "schema": MANIFEST_SCHEMA,
        "candidate_family": "tpd_clean_v7_dch",
        "candidate_root": _relative(Path(matrix["candidate_root"])),
        "matrix_schema": matrix["schema"],
        "matrix_counts": {
            "runs": matrix["run_count"],
            "checkpoints": matrix["checkpoint_count"],
            "sweeps": matrix["sweep_count"],
            "validation_fields": matrix["validation_field_count"],
        },
        "validation_fields": list(VALIDATION_FIELDS),
        "artifact_matrix": {
            "runs": artifact_runs,
            "run_count": len(artifact_runs),
            "checkpoint_count": sum(
                len(run["checkpoints"]) for run in artifact_runs.values()
            ),
            "sweep_count": sum(
                len(run["sweeps"]) for run in artifact_runs.values()
            ),
            "native_17_fields_complete": all(
                run["native_17_fields_complete"]
                for run in artifact_runs.values()
            )
            if artifact_runs
            else False,
        },
        "training_source_lock_sha256": sha256_file(training_source_lock),
        "acceptance_source_lock_sha256": sha256_file(acceptance_source_lock),
        "input_count": len(inputs),
        "category_counts": category_counts,
        "inputs": inputs,
    }


def _without_generated_at(report: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(report))
    value.pop("generated_at_utc", None)
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _report_paths(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path]:
    root = Path(output_dir)
    return (
        root / JSON_OUTPUT_NAME,
        root / MARKDOWN_OUTPUT_NAME,
        root / MANIFEST_NAME,
        root / MARKER_NAME,
    )


def validate_published_report(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _summary_module()
    json_path, markdown_path, _, _ = _report_paths(output_dir)
    published = _load_json(json_path, "published DCH comparison")
    _require(
        published.get("status") == "complete"
        and published.get("gate_evaluated") is True,
        "published DCH report status differs",
    )
    matrix = validate_completion_matrix(
        DEFAULT_CANDIDATE_ROOT,
        DEFAULT_EVALUATOR,
        DEFAULT_ACCEPTANCE_SOURCE_LOCK,
    )
    derived = summary.build_report(matrix=matrix)
    _require(
        _without_generated_at(published) == _without_generated_at(derived),
        "published DCH report differs from exact inputs",
    )
    expected_markdown = summary.render_markdown(published).encode("utf-8")
    _require(
        markdown_path.is_file()
        and not markdown_path.is_symlink()
        and markdown_path.read_bytes() == expected_markdown,
        "published DCH Markdown differs",
    )
    return published, matrix


def _marker_bytes(
    json_path: Path,
    markdown_path: Path,
    manifest_path: Path,
) -> bytes:
    return (
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in (json_path, markdown_path, manifest_path)
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def publish_completion(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    json_path, markdown_path, manifest_path, marker_path = _report_paths(
        output_dir
    )
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    if marker_path.exists() or marker_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite marker: {marker_path}")
    report, matrix = validate_published_report(output_dir)
    manifest = build_manifest(matrix)
    _write_new(manifest_path, _canonical_json_bytes(manifest))
    try:
        _write_new(
            marker_path,
            _marker_bytes(json_path, markdown_path, manifest_path),
        )
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return {
        "status": "complete",
        "decision": report["decision"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "marker": str(marker_path.resolve()),
        "marker_sha256": sha256_file(marker_path),
        "input_count": manifest["input_count"],
    }


def verify_completion(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    json_path, markdown_path, manifest_path, marker_path = _report_paths(
        output_dir
    )
    report, matrix = validate_published_report(output_dir)
    published_manifest = _load_json(manifest_path, "completion manifest")
    expected_manifest = build_manifest(matrix)
    _require(
        published_manifest == expected_manifest,
        "completion manifest differs from exact inputs",
    )
    _require(
        marker_path.is_file()
        and not marker_path.is_symlink()
        and marker_path.read_bytes()
        == _marker_bytes(json_path, markdown_path, manifest_path),
        "completion marker digest set differs",
    )
    return {
        "status": "complete",
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
        "ner_stage_authorized": report["ner_stage_authorized"],
        "input_count": published_manifest["input_count"],
        "manifest_sha256": sha256_file(manifest_path),
        "marker_sha256": sha256_file(marker_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish or verify V7-DCH formal800 completion"
    )
    parser.add_argument("mode", choices=("preflight", "publish", "verify"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "preflight":
        result = inspect_completion_matrix()
    elif args.mode == "publish":
        readiness = inspect_completion_matrix()
        if readiness["ready"] is not True:
            raise SystemExit(
                "V7-DCH formal800 matrix is incomplete; only preflight is allowed"
            )
        result = publish_completion()
    else:
        result = verify_completion()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


__all__ = [
    "BUDGET_KEYS",
    "CHECKPOINT_SPECS",
    "DATASET",
    "DEFAULT_ACCEPTANCE_SOURCE_LOCK",
    "DEFAULT_CANDIDATE_ROOT",
    "DEFAULT_EVALUATOR",
    "DEFAULT_OUTPUT_DIR",
    "GPU_ASSIGNMENTS",
    "IncompleteArtifact",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "MARKER_NAME",
    "ROLE_SPECS",
    "RUN_TAG",
    "SEEDS",
    "VALIDATION_FIELDS",
    "VARIANTS",
    "build_manifest",
    "inspect_completion_matrix",
    "inspect_training_readiness",
    "main",
    "parse_args",
    "publish_completion",
    "sha256_file",
    "validate_acceptance_source_lock",
    "validate_completion_matrix",
    "validate_existing_sweep",
    "validate_published_report",
    "verify_completion",
]


if __name__ == "__main__":
    main()
