#!/usr/bin/env python3
"""Independent Mechanism Audit M for the TPD-Clean V7-DCH matrix.

The audit is deliberately separate from formal Gates A--E.  It reads the
four completed DCH runs and all three checkpoint roles in each run, evaluates
the frozen internal validation split, and reports topology measurements at a
common threshold registry.  It never trains, reselects a checkpoint, changes
a formal run directory, or accesses the official test set.

The common operating-point registry is fixed by the completed V6 Full
seed-3407 diagnostics:

* the frozen thresholds 0.5, 0.58, and 0.999;
* the thresholds selected by the V6 reference checkpoint at each registered
  Fa budget.

Duplicate numeric thresholds are evaluated once and retain every registry
label.  Thus every DCH model is evaluated at the same numeric threshold as
its corresponding V6 reference role; a model's own calibration cannot select
its topology comparison point.

Mechanism Audit M decides only
``fragmentation_mechanism_claim_supported``.  It does not recompute, replace,
relax, or imply any part of formal Gates A--E.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import diagnose_tpd_clean_v6_fragmentation as v6_diag  # noqa: E402
from experiments.train_tpd_clean_v7_dch import (  # noqa: E402
    build_clean_v7_dch_model,
)
from experiments.train_tpd_clean_v7_dch_exact import (  # noqa: E402
    ENTRY_SCHEMA as DCH_ENTRY_SCHEMA,
    STORED_VALIDATION_METRICS,
)
from experiments.train_tpd_pilot import ValidationSubset, json_ready  # noqa: E402
from model.tpd_clean_v7_dch import (  # noqa: E402
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
)


SCHEMA = "sctransnet_tpd_clean_v7_dch_mechanism_audit_v1"
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_clean_v7_dch_mechanism_checkpoint_audit_v1"
)
READINESS_SCHEMA = (
    "sctransnet_tpd_clean_v7_dch_mechanism_audit_readiness_v1"
)
MATRIX_SCHEMA = SCHEMA
SUMMARY_SCHEMA = "sctransnet_tpd_clean_v7_dch_completion_summary_v1"
REFERENCE_SCHEMA = "sctransnet_tpd_clean_v6_frozen_failure_diagnostic_v1"
CANDIDATE_FAMILY = "spd_anchored_tpd_clean_v7_deferred_context_headroom"
V6_REFERENCE_FAMILY = "spd_anchored_tpd_clean_v6_phase_tied_kcs_zero_mean_gain"
DATASET = "NUDT-SIRST"
SEEDS = (42, 3407)
VARIANTS = tuple(SUPPORTED_CLEAN_V7_DCH_VARIANTS)
PRIMARY_VARIANT = "tpd_clean_v7_dch_full"
CAPACITY_VARIANT = "tpd_clean_v7_dch_capacity"
EXPECTED_EPOCHS = 800
EXPECTED_VAL_COUNT = 133
EXPECTED_JOB_COUNT = 12
DEFAULT_RUN_TAG = "formal800_exact_fp32_2x5090_v1"
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / DATASET / "comparison"
DEFAULT_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "tpd_clean_v7_dch_mechanism_audit.json"
)
DEFAULT_REFERENCE_ROOT = (
    REPO_ROOT / "analysis/results/tpd_clean_v6_frozen_failure_atlas_v1"
)
DEFAULT_FIXED_THRESHOLDS = (0.5, 0.58, 0.999)
DEFAULT_FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
POSTPROCESS_GPUS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
CHECKPOINT_SPECS: Mapping[str, Mapping[str, str]] = {
    "pd_primary": {
        "filename": "best.pth.tar",
        "checkpoint_role": "best_validation_pd_primary",
        "summary_epoch": "best_pd_epoch",
        "summary_metrics": "best_pd_validation_metrics",
    },
    "miou_primary": {
        "filename": "best_miou.pth.tar",
        "checkpoint_role": "best_validation_miou_secondary",
        "summary_epoch": "best_miou_epoch",
        "summary_metrics": "best_miou_validation_metrics",
    },
    "last": {
        "filename": "last.pth.tar",
        "checkpoint_role": "last_evaluated_epoch",
        "summary_epoch": "",
        "summary_metrics": "",
    },
}
REFERENCE_RELATIVE_PATHS = {
    "pd_primary": Path(
        "gpu2_full/tpd_clean_v6_full/seed_3407/pd_primary.json"
    ),
    "miou_primary": Path(
        "gpu2_full/tpd_clean_v6_full/seed_3407/miou_primary.json"
    ),
}
AUDIT_MEASURE_KEYS = (
    "fragment_excess_total",
    "unmatched_pixels_in_gt",
    "split_target_count",
    "fragment_fa_fraction",
    "largest_fragment_fraction_mean",
    "largest_fragment_fraction_p10",
)


class AuditInputsUnavailable(RuntimeError):
    """Raised when a run is requested before all twelve inputs are ready."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def threshold_key(value: float) -> str:
    return f"{float(value):.10g}"


def _finite_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("At least one fixed threshold is required")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError("Fixed thresholds must be finite values in [0, 1]")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("Fixed thresholds must be unique")
    return thresholds


def _finite_budgets(values: Sequence[float]) -> tuple[float, ...]:
    budgets = tuple(float(value) for value in values)
    if not budgets:
        raise ValueError("At least one Fa budget is required")
    if any(not math.isfinite(value) or value < 0.0 for value in budgets):
        raise ValueError("Fa budgets must be finite and non-negative")
    if len(budgets) != len(set(budgets)):
        raise ValueError("Fa budgets must be unique")
    return budgets


def run_directory(
    results_root: Path,
    variant: str,
    seed: int,
    run_tag: str = DEFAULT_RUN_TAG,
) -> Path:
    return (
        Path(results_root)
        / DATASET
        / variant
        / f"seed_{seed}_{run_tag}"
    )


def diagnostic_output_path(
    output_dir: Path,
    variant: str,
    seed: int,
    role: str,
) -> Path:
    return (
        Path(output_dir)
        / "mechanism_checkpoints"
        / variant
        / f"seed_{seed}"
        / f"{role}.json"
    )


def expected_jobs(
    results_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_tag: str = DEFAULT_RUN_TAG,
) -> list[Dict[str, Any]]:
    jobs: list[Dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            directory = run_directory(results_root, variant, seed, run_tag)
            for role, spec in CHECKPOINT_SPECS.items():
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "role": role,
                        "run_dir": directory,
                        "checkpoint": directory / spec["filename"],
                        "output": diagnostic_output_path(
                            output_dir, variant, seed, role
                        ),
                    }
                )
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise RuntimeError("DCH Mechanism Audit M matrix is not 4x3")
    return jobs


def _required_job_paths(job: Mapping[str, Any]) -> Dict[str, Path]:
    directory = Path(job["run_dir"])
    return {
        "protocol": directory / "protocol.json",
        "split": directory / "split.json",
        "summary": directory / "summary.json",
        "metrics": directory / "metrics.jsonl",
        "checkpoint": Path(job["checkpoint"]),
    }


def inventory_contract(
    results_root: Path,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_tag: str = DEFAULT_RUN_TAG,
) -> Dict[str, Any]:
    """Describe absent, partial, or ready inputs without inventing results."""

    records: list[Dict[str, Any]] = []
    ready_count = 0
    present_file_count = 0
    for job in expected_jobs(results_root, output_dir, run_tag):
        paths = _required_job_paths(job)
        present = {
            label: path.is_file() and not path.is_symlink()
            for label, path in paths.items()
        }
        present_file_count += sum(present.values())
        if all(present.values()):
            state = "ready"
            ready_count += 1
        elif any(path.exists() or path.is_symlink() for path in paths.values()):
            state = "partial"
        else:
            state = "absent"
        records.append(
            {
                "variant": job["variant"],
                "seed": job["seed"],
                "checkpoint_role": job["role"],
                "state": state,
                "paths": {key: str(value.resolve()) for key, value in paths.items()},
                "regular_file_present": present,
                "missing_or_invalid": [
                    label for label, valid in present.items() if not valid
                ],
            }
        )

    if ready_count == EXPECTED_JOB_COUNT:
        status = "READY"
    elif present_file_count == 0:
        status = "NO_RESULTS_AVAILABLE"
    else:
        status = "INCOMPLETE_INPUTS"
    return {
        "schema": READINESS_SCHEMA,
        "mode": "availability",
        "status": status,
        "candidate_family": "tpd_clean_v7_dch",
        "expected_run_count": 4,
        "expected_checkpoints_per_run": 3,
        "expected_job_count": EXPECTED_JOB_COUNT,
        "ready_job_count": ready_count,
        "present_regular_file_count": present_file_count,
        "jobs": records,
        "audit_performed": False,
        "mechanism_audit_M_pass": None,
        "fragmentation_mechanism_claim_supported": None,
        "claim_status": "NOT_EVALUATED",
        "formal_gate_replacement": False,
        "gate_A_E_recomputed": False,
        "ner_authorization_decided": False,
        "performance_results": None,
        "no_result_contract": {
            "no_metric_value_prefilled": True,
            "no_audit_outcome_prefilled": True,
            "missing_results_are_not_a_failed_experiment": True,
        },
    }


def _require_identity_mapping(
    payload: Mapping[str, Any],
    key: str,
    location: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} lacks mapping {key!r}")
    return value


def require_validation_metrics(
    metrics: Mapping[str, Any],
    location: str,
) -> Dict[str, Any]:
    """Return the native 17-field record or reject an incomplete record."""

    if not isinstance(metrics, Mapping):
        raise ValueError(f"{location} validation metrics are not a mapping")
    missing = [name for name in STORED_VALIDATION_METRICS if name not in metrics]
    if missing:
        raise ValueError(
            f"{location} lacks native 17-field validation metrics: {missing}"
        )
    result: Dict[str, Any] = {}
    for name in STORED_VALIDATION_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{location} metric {name!r} is not numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{location} metric {name!r} is not finite")
        result[name] = copy.deepcopy(value)
    return result


def _same_metric_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    for name in STORED_VALIDATION_METRICS:
        left_value = left[name]
        right_value = right[name]
        if name.endswith("_count"):
            if int(left_value) != int(right_value):
                return False
        elif not math.isclose(
            float(left_value),
            float(right_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    return True


def _validate_source_locks(run_identity: Mapping[str, Any]) -> Dict[str, str]:
    source_locks = run_identity.get("source_locks")
    if not isinstance(source_locks, Mapping) or not source_locks:
        raise ValueError("DCH run identity has no source-lock mapping")
    normalized: Dict[str, str] = {}
    for name, digest in source_locks.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("DCH run identity contains an invalid source lock")
        normalized[name] = digest
    if "tpd_clean_v7_dch_exact_source_lock" not in normalized:
        raise ValueError("DCH run identity lacks its independent exact source lock")
    return normalized


def validate_job_artifacts(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate DCH-native identity, 17 fields, roles, and immutable inputs."""

    paths = _required_job_paths(job)
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} is not a regular file: {path}")

    protocol = load_json_object(paths["protocol"])
    split = load_json_object(paths["split"])
    summary = load_json_object(paths["summary"])
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {paths['checkpoint']}")
    if protocol.get("schema") != DCH_ENTRY_SCHEMA:
        raise ValueError("protocol is not an independent V7-DCH exact run")
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("summary is not a V7-DCH completion summary")

    arguments = _require_identity_mapping(protocol, "arguments", "protocol")
    expected = {
        "dataset": DATASET,
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
    }
    observed_sources = {
        "protocol": {
            "dataset": arguments.get("dataset"),
            "variant": arguments.get("variant"),
            "seed": arguments.get("seed"),
        },
        "summary": {
            "dataset": summary.get("dataset"),
            "variant": summary.get("variant"),
            "seed": summary.get("seed"),
        },
        "checkpoint": {
            "dataset": checkpoint.get("dataset"),
            "variant": checkpoint.get("variant"),
            "seed": checkpoint.get("seed"),
        },
    }
    for source, observed in observed_sources.items():
        if observed != expected:
            raise ValueError(
                f"{source} DCH identity mismatch: "
                f"expected={expected}, observed={observed}"
            )
    if summary.get("status") != "complete":
        raise ValueError("DCH run is not complete")
    if int(arguments.get("epochs", -1)) != EXPECTED_EPOCHS:
        raise ValueError("DCH protocol does not contain 800 epochs")
    if protocol.get("stored_validation_metrics") != list(
        STORED_VALIDATION_METRICS
    ):
        raise ValueError("protocol native validation-field contract differs")
    if summary.get("stored_validation_metrics") != list(
        STORED_VALIDATION_METRICS
    ):
        raise ValueError("summary native validation-field contract differs")

    validation_ids = split.get("used_val_ids")
    if not isinstance(validation_ids, list):
        raise ValueError("split lacks ordered used_val_ids")
    if split.get("used_val_count") != len(validation_ids):
        raise ValueError("split validation count does not match its ID list")
    if split.get("used_val_count") != EXPECTED_VAL_COUNT:
        raise ValueError("DCH run does not use the frozen 133-image split")

    for label, payload in {
        "protocol": protocol,
        "split": split,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        if payload.get("official_test_accessed") is not False:
            raise ValueError(
                f"{label} does not state official_test_accessed=false"
            )

    protocol_identity = _require_identity_mapping(
        protocol, "run_identity", "protocol"
    )
    checkpoint_identity = _require_identity_mapping(
        checkpoint, "run_identity", "checkpoint"
    )
    if dict(checkpoint_identity) != dict(protocol_identity):
        raise ValueError("checkpoint run identity differs from protocol")
    if (
        protocol_identity.get("variant") != job["variant"]
        or protocol_identity.get("dataset") != DATASET
        or protocol_identity.get("seed") != job["seed"]
    ):
        raise ValueError("protocol run identity is not the requested DCH job")
    run_id = protocol_identity.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(
        f"tpd-clean-v7-dch-exact:{DATASET}:{job['variant']}:"
    ):
        raise ValueError("run_id is not DCH-owned")
    source_locks = _validate_source_locks(protocol_identity)

    protocol_model = _require_identity_mapping(protocol, "model", "protocol")
    summary_model = _require_identity_mapping(summary, "model", "summary")
    checkpoint_model = _require_identity_mapping(
        checkpoint, "model_metadata", "checkpoint"
    )
    for label, model_metadata in {
        "protocol": protocol_model,
        "summary": summary_model,
        "checkpoint": checkpoint_model,
    }.items():
        if (
            model_metadata.get("candidate_family") != CANDIDATE_FAMILY
            or model_metadata.get("variant") != job["variant"]
            or model_metadata.get("mainline_contract")
            != "Keep-Context-Saliency"
        ):
            raise ValueError(f"{label} model identity is not V7-DCH K/C/S")

    expected_role = CHECKPOINT_SPECS[str(job["role"])]["checkpoint_role"]
    if checkpoint.get("checkpoint_role") != expected_role:
        raise ValueError("DCH checkpoint role mismatch")
    checkpoint_metrics = require_validation_metrics(
        _require_identity_mapping(
            checkpoint, "validation_metrics", "checkpoint"
        ),
        "checkpoint",
    )

    metrics_events: list[Dict[str, Any]] = []
    with paths["metrics"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"metrics line {line_number} is not an object")
            require_validation_metrics(event, f"metrics line {line_number}")
            metrics_events.append(event)
    if len(metrics_events) != EXPECTED_EPOCHS:
        raise ValueError("metrics.jsonl is not a complete 800-epoch record")
    if [event.get("epoch") for event in metrics_events] != list(
        range(1, EXPECTED_EPOCHS + 1)
    ):
        raise ValueError("metrics.jsonl epochs are not contiguous")

    spec = CHECKPOINT_SPECS[str(job["role"])]
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if job["role"] == "last":
        expected_epoch = EXPECTED_EPOCHS
        expected_metrics = require_validation_metrics(
            metrics_events[-1], "last metrics event"
        )
    else:
        expected_epoch = int(summary.get(spec["summary_epoch"], -1))
        expected_metrics = require_validation_metrics(
            _require_identity_mapping(
                summary, spec["summary_metrics"], "summary"
            ),
            f"summary {spec['summary_metrics']}",
        )
    if checkpoint_epoch != expected_epoch:
        raise ValueError("checkpoint epoch differs from its registered role")
    if not _same_metric_record(checkpoint_metrics, expected_metrics):
        raise ValueError("checkpoint native 17 fields differ from their source")

    input_sha256 = {label: file_sha256(path) for label, path in paths.items()}
    return {
        "paths": paths,
        "protocol": protocol,
        "split": split,
        "summary": summary,
        "checkpoint": checkpoint,
        "metrics_events": metrics_events,
        "checkpoint_metrics": checkpoint_metrics,
        "source_identity": {
            "candidate_family": CANDIDATE_FAMILY,
            "dataset": DATASET,
            "variant": job["variant"],
            "seed": job["seed"],
            "comparison_role": job["role"],
            "checkpoint_role": expected_role,
            "checkpoint_epoch": checkpoint_epoch,
            "run_id": run_id,
            "architecture_id": protocol_identity.get("architecture_id"),
            "source_locks": source_locks,
            "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
            "protocol_schema": protocol["schema"],
            "summary_schema": summary["schema"],
        },
        "input_sha256": input_sha256,
    }


def verify_inputs_unchanged(
    paths: Mapping[str, Path],
    before: Mapping[str, str],
) -> Dict[str, str]:
    after = {label: file_sha256(path) for label, path in paths.items()}
    if after != dict(before):
        raise RuntimeError("A formal DCH input changed during Mechanism Audit M")
    return after


def _audit_measures(point: Mapping[str, Any]) -> Dict[str, Any]:
    taxonomy = _require_identity_mapping(
        point, "component_taxonomy", "topology point"
    )
    topology = _require_identity_mapping(point, "gt_topology", "topology point")
    return {
        "fragment_excess_total": int(topology["fragment_excess_total"]),
        "unmatched_pixels_in_gt": int(taxonomy["unmatched_pixels_in_gt"]),
        "split_target_count": int(topology["split_target_count"]),
        "fragment_fa_fraction": float(taxonomy["fragment_fa_fraction"]),
        "largest_fragment_fraction_mean": float(
            topology["largest_fragment_fraction_mean"]
        ),
        "largest_fragment_fraction_p10": float(
            topology["largest_fragment_fraction_p10"]
        ),
    }


def audit_point_view(point: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = require_validation_metrics(point, "evaluated operating point")
    return {
        "threshold": float(point["threshold"]),
        "validation_metrics": metrics,
        "audit_measures": _audit_measures(point),
    }


def _merge_registry_point(
    registry: Dict[str, Dict[str, Any]],
    *,
    threshold: float,
    label: str,
    kind: str,
    reference_point: Mapping[str, Any],
) -> None:
    key = threshold_key(threshold)
    view = audit_point_view(reference_point)
    if key in registry:
        existing = registry[key]
        if float(existing["threshold"]) != float(threshold):
            raise RuntimeError("threshold key collision in operating registry")
        if existing["reference"]["audit_measures"] != view["audit_measures"]:
            raise RuntimeError(
                "one V6 reference checkpoint produced inconsistent duplicate "
                "threshold measurements"
            )
        existing["registry_labels"].append(label)
        existing["registry_kinds"].append(kind)
        return
    registry[key] = {
        "threshold": float(threshold),
        "registry_labels": [label],
        "registry_kinds": [kind],
        "matched_operating_point": kind == "v6_reference_fa_budget",
        "reference": view,
    }


def build_reference_registry(
    payload: Mapping[str, Any],
    *,
    role: str,
    fixed_thresholds: Sequence[float],
    fa_budgets: Sequence[float],
) -> Dict[str, Any]:
    """Build a unique, reference-determined threshold registry for one role."""

    if payload.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("V6 reference diagnostic schema mismatch")
    expected = {
        "variant": "tpd_clean_v6_full",
        "seed": 3407,
        "checkpoint_role": role,
    }
    observed = {name: payload.get(name) for name in expected}
    if observed != expected:
        raise ValueError(
            f"V6 reference identity mismatch: expected={expected}, "
            f"observed={observed}"
        )
    if (
        payload.get("official_test_accessed") is not False
        or payload.get("training_performed") is not False
        or payload.get("complete_validation_split") is not True
    ):
        raise ValueError("V6 reference is not a completed read-only diagnosis")
    metadata = _require_identity_mapping(payload, "model_metadata", "V6 reference")
    if metadata.get("candidate_family") != V6_REFERENCE_FAMILY:
        raise ValueError("V6 reference family mismatch")
    modes = _require_identity_mapping(payload, "modes", "V6 reference")
    as_trained = _require_identity_mapping(
        modes, "as_trained", "V6 reference modes"
    )
    fixed_points = _require_identity_mapping(
        as_trained, "fixed_threshold_points", "V6 reference"
    )
    budget_points = _require_identity_mapping(
        as_trained, "best_points_under_fa_budget", "V6 reference"
    )

    registry: Dict[str, Dict[str, Any]] = {}
    for threshold in fixed_thresholds:
        key = threshold_key(threshold)
        point = fixed_points.get(key)
        if not isinstance(point, Mapping):
            raise ValueError(f"V6 reference lacks frozen threshold {key}")
        _merge_registry_point(
            registry,
            threshold=float(threshold),
            label=f"fixed_threshold_{key}",
            kind="fixed_threshold",
            reference_point=point,
        )
    for budget in fa_budgets:
        budget_key = threshold_key(budget)
        point = budget_points.get(budget_key)
        if not isinstance(point, Mapping):
            raise ValueError(f"V6 reference lacks Fa budget {budget_key}")
        threshold = float(point["threshold"])
        _merge_registry_point(
            registry,
            threshold=threshold,
            label=f"v6_reference_fa_budget_{budget_key}",
            kind="v6_reference_fa_budget",
            reference_point=point,
        )
    for entry in registry.values():
        entry["registry_labels"] = sorted(set(entry["registry_labels"]))
        entry["registry_kinds"] = sorted(set(entry["registry_kinds"]))
        entry["matched_operating_point"] = (
            "v6_reference_fa_budget" in entry["registry_kinds"]
        )
    return {
        "role": role,
        "reference_variant": "tpd_clean_v6_full",
        "reference_seed": 3407,
        "reference_checkpoint_role": role,
        "selection_owner": "V6 reference only",
        "common_numeric_threshold_for_all_DCH_variants": True,
        "points": dict(
            sorted(registry.items(), key=lambda item: float(item[1]["threshold"]))
        ),
    }


def load_reference_registries(
    reference_root: Path,
    *,
    fixed_thresholds: Sequence[float],
    fa_budgets: Sequence[float],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    registries: Dict[str, Any] = {}
    hashes: Dict[str, str] = {}
    for role, relative in REFERENCE_RELATIVE_PATHS.items():
        path = Path(reference_root) / relative
        hashes[role] = file_sha256(path)
        registries[role] = build_reference_registry(
            load_json_object(path),
            role=role,
            fixed_thresholds=fixed_thresholds,
            fa_budgets=fa_budgets,
        )
        registries[role]["reference_path"] = str(path.resolve())
        registries[role]["reference_sha256"] = hashes[role]
    return registries, hashes


def thresholds_for_role(
    registries: Mapping[str, Mapping[str, Any]],
    role: str,
) -> Dict[str, Dict[str, Any]]:
    if role in ("pd_primary", "miou_primary"):
        return copy.deepcopy(dict(registries[role]["points"]))
    if role != "last":
        raise ValueError(f"Unknown checkpoint role: {role}")
    combined: Dict[str, Dict[str, Any]] = {}
    for source_role in ("pd_primary", "miou_primary"):
        for key, entry in registries[source_role]["points"].items():
            if key not in combined:
                combined[key] = {
                    "threshold": entry["threshold"],
                    "registry_labels": [],
                    "registry_kinds": [],
                    "reference_roles": [],
                }
            combined[key]["registry_labels"].extend(entry["registry_labels"])
            combined[key]["registry_kinds"].extend(entry["registry_kinds"])
            combined[key]["reference_roles"].append(source_role)
    for entry in combined.values():
        entry["registry_labels"] = sorted(set(entry["registry_labels"]))
        entry["registry_kinds"] = sorted(set(entry["registry_kinds"]))
        entry["reference_roles"] = sorted(set(entry["reference_roles"]))
        entry["matched_operating_point"] = (
            "v6_reference_fa_budget" in entry["registry_kinds"]
        )
    return dict(
        sorted(combined.items(), key=lambda item: float(item[1]["threshold"]))
    )


def configure_dch_inference(device: str) -> Dict[str, Any]:
    if device == "cuda:0" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != ":4096:8":
        raise RuntimeError(
            "CUDA audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def bind_requested_device(
    device: str,
    physical_gpu: str | None,
) -> Dict[str, Any]:
    if device == "cpu":
        if physical_gpu is not None:
            raise ValueError("--physical-gpu is only valid with --device cuda:0")
        return {
            "device": "cpu",
            "logical_device": "cpu",
            "physical_gpu_index": None,
            "physical_gpu_uuid": None,
        }
    if device != "cuda:0" or physical_gpu not in POSTPROCESS_GPUS:
        raise ValueError(
            "CUDA audit requires --device cuda:0 and --physical-gpu 2 or 3"
        )
    gpu_uuid = POSTPROCESS_GPUS[physical_gpu]
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [field.strip() for field in query.split(",")]
    expected = [physical_gpu, "NVIDIA GeForce RTX 5090", gpu_uuid]
    if fields != expected:
        raise RuntimeError(
            f"physical GPU identity differs: expected={expected}, observed={fields}"
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requested DCH audit GPU is not uniquely visible")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
        raise RuntimeError("visible cuda:0 is not an RTX 5090")
    return {
        "device": "cuda:0",
        "logical_device": "cuda:0",
        "physical_gpu_index": int(physical_gpu),
        "physical_gpu_uuid": gpu_uuid,
        "visible_device_name": torch.cuda.get_device_name(0),
    }


def evaluate_job(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    registry_points: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
    device_provenance: Mapping[str, Any],
    dilation_radius: int,
) -> Dict[str, Any]:
    checkpoint = artifacts["checkpoint"]
    protocol = artifacts["protocol"]
    split = artifacts["split"]
    arguments = protocol["arguments"]

    model, metadata = build_clean_v7_dch_model(
        str(job["variant"]), int(job["seed"])
    )
    if metadata.get("candidate_family") != CANDIDATE_FAMILY:
        raise ValueError("DCH builder family differs during audit")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    state_sha256 = v6_diag.model_state_sha256(model)

    validation_ids = list(split["used_val_ids"])
    dataset_dir = Path(arguments["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    validation_set = ValidationSubset(
        dataset_dir / DATASET,
        validation_ids,
        {key: float(value) for key, value in protocol["normalization"].items()},
    )
    loader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    probabilities, targets, losses = v6_diag.collect_predictions(
        model, loader, device
    )
    match_radius = float(arguments["match_radius"])
    tiny_area = int(arguments["tiny_area"])
    points: Dict[str, Any] = {}
    for key, registry in registry_points.items():
        threshold = float(registry["threshold"])
        point = v6_diag.decorated_point(
            probabilities,
            targets,
            losses,
            validation_ids,
            threshold,
            match_radius,
            tiny_area,
            dilation_radius,
        )
        points[key] = {
            "registry_labels": list(registry["registry_labels"]),
            "registry_kinds": list(registry["registry_kinds"]),
            "matched_operating_point": bool(
                registry["matched_operating_point"]
            ),
            **audit_point_view(point),
        }

    input_after = verify_inputs_unchanged(
        artifacts["paths"], artifacts["input_sha256"]
    )
    if v6_diag.model_state_sha256(model) != state_sha256:
        raise RuntimeError("DCH model state changed during read-only audit")
    return json_ready(
        {
            "schema": CHECKPOINT_SCHEMA,
            "audit": "Mechanism Audit M",
            "audit_scope": "frozen_internal_validation_topology_only",
            "candidate_family": "tpd_clean_v7_dch",
            "variant": job["variant"],
            "seed": job["seed"],
            "checkpoint_role": job["role"],
            "source_identity": artifacts["source_identity"],
            "checkpoint": str(Path(job["checkpoint"]).resolve()),
            "run_directory": str(Path(job["run_dir"]).resolve()),
            "checkpoint_validation_metrics_17": artifacts[
                "checkpoint_metrics"
            ],
            "native_validation_field_count": len(
                STORED_VALIDATION_METRICS
            ),
            "validation": {
                "dataset": DATASET,
                "validation_count": len(validation_ids),
                "validation_ids": validation_ids,
                "validation_split_sha256": split["hashes"][
                    "used_val_sha256"
                ],
                "match_radius": match_radius,
                "tiny_area": tiny_area,
                "component_connectivity": 2,
                "dilation_radius_pixels": dilation_radius,
                "prediction_comparison": "probability > threshold",
            },
            "device": dict(device_provenance),
            "operating_points": points,
            "loaded_model_state_sha256": state_sha256,
            "input_sha256_before": artifacts["input_sha256"],
            "input_sha256_after": input_after,
            "formal_inputs_unchanged": True,
            "training_performed": False,
            "checkpoint_reselection_permitted": False,
            "official_test_accessed": False,
            "formal_gate_replacement": False,
            "gate_A_E_recomputed": False,
            "fragmentation_mechanism_claim_supported": None,
            "per_checkpoint_claim_status": "DESCRIPTIVE_ONLY",
        }
    )


def _point_from_payload(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    points = _require_identity_mapping(
        payload, "operating_points", "DCH audit payload"
    )
    point = points.get(key)
    if not isinstance(point, Mapping):
        raise ValueError(f"DCH audit payload lacks operating point {key}")
    return point


def _measure(point: Mapping[str, Any], name: str) -> float:
    values = _require_identity_mapping(point, "audit_measures", "audit point")
    return float(values[name])


def build_mechanism_audit(
    payloads: Mapping[tuple[str, int, str], Mapping[str, Any]],
    registries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply preregistered M1--M4 without consulting Gates A--E."""

    required = {
        (variant, seed, role)
        for variant in VARIANTS
        for seed in SEEDS
        for role in CHECKPOINT_SPECS
    }
    if set(payloads) != required:
        return {
            "status": "INCOMPLETE_DIAGNOSTIC",
            "mechanism_audit_M_pass": None,
            "fragmentation_mechanism_claim_supported": None,
            "formal_gate_replacement": False,
            "gate_A_E_recomputed": False,
            "missing_jobs": [
                list(key) for key in sorted(required.difference(payloads))
            ],
        }

    baseline_comparisons: list[Dict[str, Any]] = []
    for role in ("pd_primary", "miou_primary"):
        dch_payload = payloads[(PRIMARY_VARIANT, 3407, role)]
        for key, registry_point in registries[role]["points"].items():
            dch_point = _point_from_payload(dch_payload, key)
            reference = registry_point["reference"]
            dch_measures = dch_point["audit_measures"]
            reference_measures = reference["audit_measures"]
            baseline_comparisons.append(
                {
                    "checkpoint_role": role,
                    "operating_point": key,
                    "threshold": float(registry_point["threshold"]),
                    "registry_labels": list(
                        registry_point["registry_labels"]
                    ),
                    "matched_operating_point": bool(
                        registry_point["matched_operating_point"]
                    ),
                    "dch": copy.deepcopy(dch_measures),
                    "v6_reference": copy.deepcopy(reference_measures),
                    "fragment_excess_delta_dch_minus_v6": (
                        int(dch_measures["fragment_excess_total"])
                        - int(reference_measures["fragment_excess_total"])
                    ),
                    "unmatched_pixels_in_gt_delta_dch_minus_v6": (
                        int(dch_measures["unmatched_pixels_in_gt"])
                        - int(reference_measures["unmatched_pixels_in_gt"])
                    ),
                    "split_target_delta_dch_minus_v6": (
                        int(dch_measures["split_target_count"])
                        - int(reference_measures["split_target_count"])
                    ),
                    "largest_fragment_mean_delta_dch_minus_v6": (
                        float(
                            dch_measures[
                                "largest_fragment_fraction_mean"
                            ]
                        )
                        - float(
                            reference_measures[
                                "largest_fragment_fraction_mean"
                            ]
                        )
                    ),
                }
            )

    m1_pass = all(
        row["fragment_excess_delta_dch_minus_v6"] <= 0
        for row in baseline_comparisons
    )
    dch_in_gt_mean = sum(
        int(row["dch"]["unmatched_pixels_in_gt"])
        for row in baseline_comparisons
    ) / len(baseline_comparisons)
    v6_in_gt_mean = sum(
        int(row["v6_reference"]["unmatched_pixels_in_gt"])
        for row in baseline_comparisons
    ) / len(baseline_comparisons)
    m2_pass = dch_in_gt_mean <= v6_in_gt_mean
    matched_rows = [
        row for row in baseline_comparisons if row["matched_operating_point"]
    ]
    m3_pass = bool(matched_rows) and any(
        row["fragment_excess_delta_dch_minus_v6"] < 0
        for row in matched_rows
    )

    capacity_comparisons: list[Dict[str, Any]] = []
    for seed in SEEDS:
        for role in CHECKPOINT_SPECS:
            full = payloads[(PRIMARY_VARIANT, seed, role)]
            capacity = payloads[(CAPACITY_VARIANT, seed, role)]
            common_keys = sorted(
                set(full["operating_points"]).intersection(
                    capacity["operating_points"]
                ),
                key=lambda key: float(
                    full["operating_points"][key]["threshold"]
                ),
            )
            if (
                common_keys != list(full["operating_points"])
                or set(common_keys) != set(capacity["operating_points"])
            ):
                raise ValueError("Full/Capacity operating registries differ")
            for key in common_keys:
                full_point = _point_from_payload(full, key)
                capacity_point = _point_from_payload(capacity, key)
                full_value = int(
                    _measure(full_point, "fragment_excess_total")
                )
                capacity_value = int(
                    _measure(capacity_point, "fragment_excess_total")
                )
                capacity_comparisons.append(
                    {
                        "seed": seed,
                        "checkpoint_role": role,
                        "operating_point": key,
                        "threshold": float(full_point["threshold"]),
                        "full_fragment_excess_total": full_value,
                        "capacity_fragment_excess_total": capacity_value,
                        "capacity_minus_full": capacity_value - full_value,
                    }
                )
    capacity_nonhigher_everywhere = all(
        row["capacity_minus_full"] <= 0 for row in capacity_comparisons
    )
    capacity_strictly_lower_somewhere = any(
        row["capacity_minus_full"] < 0 for row in capacity_comparisons
    )
    capacity_fully_covers_full = (
        capacity_nonhigher_everywhere and capacity_strictly_lower_somewhere
    )
    m4_pass = not capacity_fully_covers_full

    mechanism_pass = m1_pass and m2_pass and m3_pass and m4_pass
    return {
        "status": "COMPLETE",
        "audit_name": "Mechanism Audit M",
        "mechanism_audit_M_pass": mechanism_pass,
        "fragmentation_mechanism_claim_supported": mechanism_pass,
        "formal_gate_replacement": False,
        "gate_A_E_recomputed": False,
        "ner_authorization_decided": False,
        "primary_metric": "fragment_excess_total",
        "auxiliary_metrics_are_selection_neutral": True,
        "M1": {
            "pass": m1_pass,
            "rule": (
                "seed3407 Full fragment_excess_total is no higher than "
                "corresponding V6 Full at every registered point"
            ),
        },
        "M2": {
            "pass": m2_pass,
            "rule": (
                "discrete mean in-GT unmatched pixels for seed3407 Full "
                "is no higher than V6 Full"
            ),
            "dch_discrete_mean": dch_in_gt_mean,
            "v6_reference_discrete_mean": v6_in_gt_mean,
        },
        "M3": {
            "pass": m3_pass,
            "rule": (
                "fragment_excess_total is strictly lower at at least one "
                "V6-reference-matched operating point"
            ),
            "matched_point_count": len(matched_rows),
        },
        "M4": {
            "pass": m4_pass,
            "rule": (
                "pass iff Capacity does not have fragment_excess_total "
                "no higher everywhere and strictly lower somewhere"
            ),
            "capacity_nonhigher_everywhere": capacity_nonhigher_everywhere,
            "capacity_strictly_lower_somewhere": (
                capacity_strictly_lower_somewhere
            ),
            "capacity_fully_covers_full": capacity_fully_covers_full,
        },
        "directional_context": {
            "split_target_nonhigher_points": sum(
                row["split_target_delta_dch_minus_v6"] <= 0
                for row in baseline_comparisons
            ),
            "split_target_total_points": len(baseline_comparisons),
            "largest_fragment_mean_nonlower_points": sum(
                row["largest_fragment_mean_delta_dch_minus_v6"] >= 0.0
                for row in baseline_comparisons
            ),
            "largest_fragment_mean_total_points": len(
                baseline_comparisons
            ),
            "interpretation": (
                "split target and largest-fragment directions are reported "
                "for consistency only and do not replace the primary metric"
            ),
        },
        "v6_full_seed3407_comparisons": baseline_comparisons,
        "full_capacity_comparisons": capacity_comparisons,
    }


def _checkpoint_bindings(
    payloads: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    bindings: list[Dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for role in CHECKPOINT_SPECS:
                key = (variant, seed, role)
                if key not in payloads:
                    raise ValueError(
                        f"Mechanism report lacks checkpoint audit {key}"
                    )
                payload = payloads[key]
                source = _require_identity_mapping(
                    payload, "source_identity", "checkpoint audit"
                )
                input_hashes = _require_identity_mapping(
                    payload, "input_sha256_before", "checkpoint audit"
                )
                checkpoint_path = Path(str(payload["checkpoint"])).resolve()
                checkpoint_sha256 = input_hashes.get("checkpoint")
                if (
                    not isinstance(checkpoint_sha256, str)
                    or len(checkpoint_sha256) != 64
                ):
                    raise ValueError(
                        "checkpoint audit lacks its checkpoint SHA256"
                    )
                expected_role = CHECKPOINT_SPECS[role]["checkpoint_role"]
                if source.get("checkpoint_role") != expected_role:
                    raise ValueError(
                        "checkpoint audit source role differs from registry"
                    )
                bindings.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "comparison_role": role,
                        "checkpoint_role": expected_role,
                        "checkpoint_epoch": int(
                            source["checkpoint_epoch"]
                        ),
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha256,
                        "run_id": source.get("run_id"),
                    }
                )
    if len(bindings) != EXPECTED_JOB_COUNT:
        raise RuntimeError("Mechanism report does not bind 12 checkpoints")
    return bindings


def build_mechanism_report(
    payloads: Mapping[tuple[str, int, str], Mapping[str, Any]],
    registries: Mapping[str, Mapping[str, Any]],
    *,
    candidate_root: Path = DEFAULT_RESULTS_ROOT,
    reference_input_sha256: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Build the finalizer-facing, independent Mechanism Audit M report."""

    audit = build_mechanism_audit(payloads, registries)
    if audit.get("status") != "COMPLETE":
        raise ValueError("cannot build a mechanism report from incomplete audits")
    supported = audit.get("fragmentation_mechanism_claim_supported")
    if not isinstance(supported, bool):
        raise ValueError("complete Mechanism Audit M must produce a boolean")
    checkpoint_inputs = _checkpoint_bindings(payloads)
    run_pairs = {
        (entry["variant"], int(entry["seed"]))
        for entry in checkpoint_inputs
    }
    if run_pairs != {
        (variant, seed) for variant in VARIANTS for seed in SEEDS
    }:
        raise ValueError("Mechanism Audit M does not bind all four runs")
    return json_ready(
        {
            "schema": SCHEMA,
            "status": "complete",
            "candidate_family": "tpd_clean_v7_dch",
            "dataset": DATASET,
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "candidate_root": str(Path(candidate_root).resolve()),
            "artifact_counts": {
                "runs": 4,
                "checkpoints": EXPECTED_JOB_COUNT,
            },
            "directions": {
                "fragment_excess_total": {
                    "direction": "lower",
                    "role": "primary_audit_metric",
                },
                "in_gt_unmatched_pixels": {
                    "direction": "lower",
                    "role": "consistency_measure",
                },
                "split_target": {
                    "direction": "lower",
                    "role": "consistency_measure",
                },
                "largest_fragment": {
                    "direction": "higher",
                    "role": "consistency_measure",
                },
            },
            "fragmentation_mechanism_claim_supported": supported,
            "mechanism_audit_M_pass": bool(
                audit["mechanism_audit_M_pass"]
            ),
            "mechanism_audit_replaces_performance_gates": False,
            "formal_gate_replacement": False,
            "gate_A_E_recomputed": False,
            "ner_authorization_decided": False,
            "training_performed": False,
            "checkpoint_reselection_permitted": False,
            "official_test_accessed": False,
            "native_validation_fields": list(
                STORED_VALIDATION_METRICS
            ),
            "native_validation_field_count": len(
                STORED_VALIDATION_METRICS
            ),
            "input_checkpoints": checkpoint_inputs,
            "input_checkpoint_count": len(checkpoint_inputs),
            "reference_input_sha256": dict(
                reference_input_sha256 or {}
            ),
            "operating_point_contract": {
                "fixed_thresholds": list(DEFAULT_FIXED_THRESHOLDS),
                "fa_budgets": list(DEFAULT_FA_BUDGETS),
                "reference_owner": "V6 Full seed3407",
                "same_numeric_threshold_for_corresponding_models": True,
            },
            "audit_result": audit,
            "claim_boundary": (
                "Mechanism Audit M decides only the fragmentation mechanism "
                "claim and does not alter performance Gates A-E."
            ),
        }
    )


def validate_mechanism_report(path: Path) -> Dict[str, Any]:
    """Strictly validate the finalizer-facing report and all 12 bindings."""

    payload = load_json_object(Path(path))
    if payload.get("schema") != SCHEMA or payload.get("status") != "complete":
        raise ValueError("Mechanism Audit M schema/status differs")
    if (
        payload.get("candidate_family") != "tpd_clean_v7_dch"
        or payload.get("dataset") != DATASET
        or payload.get("variants") != list(VARIANTS)
        or payload.get("seeds") != list(SEEDS)
    ):
        raise ValueError("Mechanism Audit M candidate identity differs")
    if payload.get("artifact_counts") != {
        "runs": 4,
        "checkpoints": EXPECTED_JOB_COUNT,
    }:
        raise ValueError("Mechanism Audit M artifact counts differ")
    directions = payload.get("directions")
    expected_directions = {
        "fragment_excess_total": "lower",
        "in_gt_unmatched_pixels": "lower",
        "split_target": "lower",
        "largest_fragment": "higher",
    }
    if not isinstance(directions, Mapping) or any(
        not isinstance(directions.get(name), Mapping)
        or directions[name].get("direction") != direction
        for name, direction in expected_directions.items()
    ):
        raise ValueError("Mechanism Audit M directions differ")
    supported = payload.get("fragmentation_mechanism_claim_supported")
    if not isinstance(supported, bool):
        raise ValueError("Mechanism Audit M claim result is not boolean")
    if (
        payload.get("mechanism_audit_M_pass") is not supported
        or payload.get("mechanism_audit_replaces_performance_gates")
        is not False
        or payload.get("formal_gate_replacement") is not False
        or payload.get("gate_A_E_recomputed") is not False
    ):
        raise ValueError("Mechanism Audit M claim/gate boundary differs")
    if (
        payload.get("native_validation_fields")
        != list(STORED_VALIDATION_METRICS)
        or payload.get("native_validation_field_count")
        != len(STORED_VALIDATION_METRICS)
    ):
        raise ValueError("Mechanism Audit M native 17-field identity differs")

    checkpoints = payload.get("input_checkpoints")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != EXPECTED_JOB_COUNT
        or payload.get("input_checkpoint_count") != EXPECTED_JOB_COUNT
    ):
        raise ValueError("Mechanism Audit M must bind 12 checkpoint inputs")
    observed_keys: set[tuple[str, int, str]] = set()
    for index, record in enumerate(checkpoints):
        if not isinstance(record, Mapping):
            raise ValueError(f"checkpoint binding {index} is not a mapping")
        try:
            key = (
                str(record["variant"]),
                int(record["seed"]),
                str(record["comparison_role"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint binding {index} has invalid identity"
            ) from exc
        if key in observed_keys:
            raise ValueError("Mechanism Audit M checkpoint binding is duplicated")
        observed_keys.add(key)
        if (
            key[0] not in VARIANTS
            or key[1] not in SEEDS
            or key[2] not in CHECKPOINT_SPECS
            or record.get("checkpoint_role")
            != CHECKPOINT_SPECS[key[2]]["checkpoint_role"]
        ):
            raise ValueError("Mechanism Audit M checkpoint role differs")
        checkpoint_path = Path(str(record.get("path", "")))
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise ValueError(
                f"Mechanism Audit M checkpoint input is missing: {checkpoint_path}"
            )
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or file_sha256(checkpoint_path) != expected_sha256
        ):
            raise ValueError("Mechanism Audit M checkpoint SHA256 differs")
    expected_keys = {
        (variant, seed, role)
        for variant in VARIANTS
        for seed in SEEDS
        for role in CHECKPOINT_SPECS
    }
    if observed_keys != expected_keys:
        raise ValueError("Mechanism Audit M checkpoint matrix differs")

    audit = payload.get("audit_result")
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "COMPLETE"
        or audit.get("fragmentation_mechanism_claim_supported")
        is not supported
        or audit.get("formal_gate_replacement") is not False
    ):
        raise ValueError("Mechanism Audit M internal decision differs")
    return copy.deepcopy(payload)


def inspect_mechanism_readiness(
    candidate_root: Path = DEFAULT_RESULTS_ROOT,
) -> Dict[str, Any]:
    """Inspect the 4x3 matrix and V6 references without writing anything."""

    candidate_root = Path(candidate_root)
    output_dir = candidate_root / DATASET / "comparison"
    availability = inventory_contract(
        candidate_root,
        output_dir=output_dir,
        run_tag=DEFAULT_RUN_TAG,
    )
    references = {
        role: DEFAULT_REFERENCE_ROOT / relative
        for role, relative in REFERENCE_RELATIVE_PATHS.items()
    }
    reference_presence = {
        role: path.is_file() and not path.is_symlink()
        for role, path in references.items()
    }
    return {
        "schema": READINESS_SCHEMA,
        "mode": "preflight",
        "ready": (
            availability["status"] == "READY"
            and all(reference_presence.values())
        ),
        "candidate_root": str(candidate_root.resolve()),
        "availability": availability,
        "reference_inputs": {
            role: {
                "path": str(path.resolve()),
                "present": reference_presence[role],
            }
            for role, path in references.items()
        },
        "output_path": str(
            (
                output_dir
                / "tpd_clean_v7_dch_mechanism_audit.json"
            ).resolve()
        ),
        "writes_performed": 0,
        "fragmentation_mechanism_claim_supported": None,
    }


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite audit output: {path}; pass --overwrite"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preflight(args: argparse.Namespace) -> Dict[str, Any]:
    contract = inventory_contract(
        args.results_root,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
    )
    contract["mode"] = "preflight"
    contract["reference_inputs"] = {
        role: {
            "path": str((args.reference_root / relative).resolve()),
            "ready": (
                (args.reference_root / relative).is_file()
                and not (args.reference_root / relative).is_symlink()
            ),
        }
        for role, relative in REFERENCE_RELATIVE_PATHS.items()
    }
    contract["ready"] = (
        contract["status"] == "READY"
        and all(
            record["ready"]
            for record in contract["reference_inputs"].values()
        )
    )
    contract["output_path"] = str(
        (
            args.output_dir
            / "tpd_clean_v7_dch_mechanism_audit.json"
        ).resolve()
    )
    return contract


def run(args: argparse.Namespace) -> Dict[str, Any]:
    availability = inventory_contract(
        args.results_root,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
    )
    if availability["status"] != "READY":
        availability["mode"] = "run_not_started"
        return availability

    determinism = configure_dch_inference(args.device)
    device_provenance = bind_requested_device(
        args.device, args.physical_gpu
    )
    device_provenance["determinism"] = determinism
    device = torch.device(args.device)
    registries, reference_hashes_before = load_reference_registries(
        args.reference_root,
        fixed_thresholds=args.fixed_thresholds,
        fa_budgets=args.fa_budgets,
    )

    payloads: Dict[tuple[str, int, str], Dict[str, Any]] = {}
    jobs = expected_jobs(args.results_root, args.output_dir, args.run_tag)
    for job in jobs:
        artifacts = validate_job_artifacts(job)
        registry = thresholds_for_role(registries, str(job["role"]))
        payload = evaluate_job(
            job,
            artifacts,
            registry,
            device=device,
            device_provenance=device_provenance,
            dilation_radius=args.dilation_radius,
        )
        payloads[
            (str(job["variant"]), int(job["seed"]), str(job["role"]))
        ] = payload

    reference_hashes_after = {
        role: file_sha256(args.reference_root / relative)
        for role, relative in REFERENCE_RELATIVE_PATHS.items()
    }
    if reference_hashes_after != reference_hashes_before:
        raise RuntimeError("V6 reference inputs changed during Mechanism Audit M")

    report = build_mechanism_report(
        payloads,
        registries,
        candidate_root=args.results_root,
        reference_input_sha256=reference_hashes_before,
    )
    outputs: list[str] = []
    for job in jobs:
        key = (str(job["variant"]), int(job["seed"]), str(job["role"]))
        output = Path(job["output"])
        write_json(output, payloads[key], overwrite=args.overwrite)
        outputs.append(str(output.resolve()))
    report.update(
        {
            "mode": "run",
            "results_root": str(args.results_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "evaluated_job_count": len(payloads),
            "checkpoint_audit_outputs": outputs,
            "reference_input_sha256_before": reference_hashes_before,
            "reference_input_sha256_after": reference_hashes_after,
            "reference_inputs_unchanged": True,
            "reported_mechanism_measures": list(AUDIT_MEASURE_KEYS),
            "limitations": [
                "Mechanism Audit M does not replace or modify Gates A-E.",
                "The audit uses the internal 133-image validation split only.",
                "Auxiliary topology directions do not substitute for "
                "fragment_excess_total.",
                "The last checkpoint is descriptive and has no V6 selection-role "
                "baseline; it remains part of the 12-checkpoint coverage and M4.",
            ],
        }
    )
    write_json(
        args.output_dir / "tpd_clean_v7_dch_mechanism_audit.json",
        report,
        overwrite=args.overwrite,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent TPD-Clean V7-DCH Mechanism Audit M"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument(
        "--fixed-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_FIXED_THRESHOLDS),
    )
    parser.add_argument(
        "--fa-budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_FA_BUDGETS),
    )
    parser.add_argument("--dilation-radius", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-gpu", choices=tuple(POSTPROCESS_GPUS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.results_root = args.results_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.reference_root = args.reference_root.resolve()
    args.fixed_thresholds = _finite_thresholds(args.fixed_thresholds)
    args.fa_budgets = _finite_budgets(args.fa_budgets)
    if args.fixed_thresholds != DEFAULT_FIXED_THRESHOLDS:
        parser.error(
            "formal Mechanism Audit M requires fixed thresholds "
            f"{DEFAULT_FIXED_THRESHOLDS}"
        )
    if args.fa_budgets != DEFAULT_FA_BUDGETS:
        parser.error(
            "formal Mechanism Audit M requires Fa budgets "
            f"{DEFAULT_FA_BUDGETS}"
        )
    if args.dilation_radius != 3:
        parser.error("formal Mechanism Audit M requires dilation radius 3")
    if not args.run_tag:
        parser.error("--run-tag must not be empty")
    if args.device == "cuda:0" and args.physical_gpu is None:
        parser.error("CUDA audit requires --physical-gpu 2 or 3")
    if args.device == "cpu" and args.physical_gpu is not None:
        parser.error("--physical-gpu is only valid with --device cuda:0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        print(
            json.dumps(
                preflight(args),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    matrix = run(args)
    if matrix["status"] != "complete":
        print(
            json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=True),
            flush=True,
        )
        raise SystemExit(2)
    print(
        "MECHANISM_AUDIT_M_COMPLETE "
        f"checkpoints={matrix['evaluated_job_count']} "
        "fragmentation_mechanism_claim_supported="
        f"{matrix['fragmentation_mechanism_claim_supported']} "
        "summary="
        f"{args.output_dir / 'tpd_clean_v7_dch_mechanism_audit.json'}",
        flush=True,
    )


__all__ = [
    "AUDIT_MEASURE_KEYS",
    "CAPACITY_VARIANT",
    "CHECKPOINT_SPECS",
    "DEFAULT_FA_BUDGETS",
    "DEFAULT_FIXED_THRESHOLDS",
    "DEFAULT_OUTPUT_PATH",
    "EXPECTED_JOB_COUNT",
    "PRIMARY_VARIANT",
    "SCHEMA",
    "MATRIX_SCHEMA",
    "SEEDS",
    "VARIANTS",
    "audit_point_view",
    "build_mechanism_audit",
    "build_mechanism_report",
    "build_reference_registry",
    "expected_jobs",
    "inventory_contract",
    "inspect_mechanism_readiness",
    "parse_args",
    "require_validation_metrics",
    "run",
    "run_directory",
    "thresholds_for_role",
    "validate_job_artifacts",
    "validate_mechanism_report",
    "verify_inputs_unchanged",
]


if __name__ == "__main__":
    main()
