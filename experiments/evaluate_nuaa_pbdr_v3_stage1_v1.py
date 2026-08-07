#!/usr/bin/env python3
"""Leakage-closed NUAA PBDR-V3 Stage-1 deployment evaluator.

The completed trainer artifacts and candidate bytes are validated before the
internal decision is inspected.  A failed internal certification immediately
publishes a Current-fallback deployment and never constructs, validates, or
opens the official NUAA test dataset.  Only an internally certified candidate
may enter the strict inference conversion and one official-test evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import pbdr_v3_non_regression_gate as gate  # noqa: E402
from experiments import three_dataset_pbdr_v3_models_seed42_v1 as models  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import train_nuaa_pbdr_v3_stage1_v1 as trainer  # noqa: E402


SCHEMA = "sctransnet_nuaa_pbdr_v3_stage1_evaluation_v1/v1"
DEPLOYMENT_SCHEMA = "sctransnet_nuaa_pbdr_v3_stage1_deployment_v1/v1"
FIXED_THRESHOLD = 0.5
TARGET_COUNT = 263
PROTOCOL_DOCUMENT = REPO_ROOT / "experiments/PBDR_V3_PROTOCOL.md"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/nuaa_pbdr_v3_stage1_v1"
GPU0_UUID = trainer.GPU0_UUID
OFFICIAL_ACCESS_CLAIM_SCHEMA = (
    "sctransnet_nuaa_pbdr_v3_official_test_access_claim_v1/v1"
)

# Frozen verbatim from the role-specific official-test gate in
# experiments/PBDR_V3_PROTOCOL.md.  The evaluator also verifies the protocol
# document SHA recorded before training, so these values cannot silently drift
# independently of a completed run.
OFFICIAL_ROLE_THRESHOLDS: dict[str, dict[str, float | int]] = {
    "best_miou": {
        "minimum_matched_target_count": 256,
        "target_count": TARGET_COUNT,
        "maximum_fa": 1.5435192155794186e-5,
        "minimum_miou": 0.798482950889985,
        "minimum_niou": 0.795348496003674,
    },
    "best_pd": {
        "minimum_matched_target_count": 257,
        "target_count": TARGET_COUNT,
        "maximum_fa": 1.4749183615536667e-5,
        "minimum_miou": 0.7905534317984362,
        "minimum_niou": 0.7926679569324805,
    },
}


class PBDRV3EvaluationProtocolError(ValueError):
    """A trainer, checkpoint, data, or deployment binding is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV3EvaluationProtocolError(message)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV3EvaluationProtocolError(
            f"cannot read {label}: {candidate}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PBDRV3EvaluationProtocolError(f"{label} must be a JSON object")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON payload contains a non-finite float")
    return value


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise FileExistsError(f"refusing to replace symlink: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _json_ready(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        ready = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(ready):
        raise ValueError(f"{label} must be finite")
    return ready


def _canonical_equal(first: Any, second: Any) -> bool:
    # Trainer checkpoints retain NumPy scalar types while the paired JSON
    # summary necessarily stores their native-Python equivalents.  Normalize
    # both representations before hashing so semantically identical artifacts
    # compare equal without weakening any value or field checks.
    return models.canonical_sha256(_json_ready(first)) == models.canonical_sha256(
        _json_ready(second)
    )


def _validate_internal_decision(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping),
        "summary lacks internal_certification_fixed_0_5",
    )
    required = ("passed", "selected", "checks", "current", "candidate", "scope")
    _require(
        all(name in value for name in required),
        "internal certification fields are incomplete",
    )
    _require(
        value.get("scope") == "frozen_internal_validation_split",
        "internal certification scope differs",
    )
    current = gate.CertificationMetrics.from_mapping(value["current"])
    candidate = gate.CertificationMetrics.from_mapping(value["candidate"])
    decision = gate.CertificationDecision(
        passed=value["passed"],
        selected=value["selected"],
        checks=value["checks"],
        current=current,
        candidate=candidate,
    )
    expected = gate.certify(current, candidate)
    _require(decision == expected, "internal certification was not recomputed exactly")
    return {
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "current": dict(value["current"]),
        "candidate": dict(value["candidate"]),
        "scope": value["scope"],
    }


def _validate_runtime_sources(value: Any) -> dict[str, dict[str, Any]]:
    _require(isinstance(value, Mapping), "protocol lacks runtime source locks")
    current = models.runtime_source_records()
    _require(set(value) == set(current), "runtime source-lock set differs")
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in current.items():
        observed = value.get(relative)
        _require(
            isinstance(observed, Mapping),
            f"runtime source lock is malformed: {relative}",
        )
        for field in ("path", "sha256", "bytes"):
            _require(
                observed.get(field) == expected[field],
                f"runtime source lock {relative} {field} differs",
            )
        verified[relative] = dict(expected)
    return verified


def _validate_split_manifest(
    run_dir: Path,
    source_locks: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Recompute the formal train-only split binding without opening test."""

    path = run_dir / "split_manifest.json"
    split = _json_object(path, label="internal split manifest")
    _require(
        split.get("schema") == "sctransnet_nuaa_pbdr_v3_internal_split/v1",
        "internal split schema differs",
    )
    _require(split.get("dataset") == models.DATASET, "internal split dataset differs")
    _require(
        split.get("source_split") == "official_train_only"
        and split.get("official_test_index_opened") is False,
        "internal split is not train-only",
    )
    _require(
        split.get("split_seed") == trainer.SPLIT_SEED
        and split.get("val_fraction") == trainer.VAL_FRACTION,
        "internal split controls differ",
    )
    declared_sha = split.get("split_sha256")
    _require(
        isinstance(declared_sha, str) and len(declared_sha) == 64,
        "internal split SHA is malformed",
    )
    unsigned = dict(split)
    del unsigned["split_sha256"]
    _require(
        models.canonical_sha256(unsigned) == declared_sha,
        "internal split canonical SHA differs",
    )
    _require(
        source_locks.get("split_manifest") == declared_sha,
        "source-lock internal split SHA differs",
    )

    official = split.get("official_train_ids")
    development = split.get("development_train_ids")
    validation = split.get("internal_validation_ids")
    _require(
        all(isinstance(value, list) for value in (official, development, validation)),
        "internal split ID fields must be lists",
    )
    official_ids = list(official)
    development_ids = list(development)
    validation_ids = list(validation)
    expected_train = data_protocol.EXPECTED_SPLITS[models.DATASET]["train"]
    _require(
        len(official_ids) == int(expected_train["count"])
        and all(isinstance(identifier, str) for identifier in official_ids)
        and len(official_ids) == len(set(official_ids)),
        "official-train ID list differs",
    )
    _require(
        data_protocol.ordered_ids_sha256(official_ids)
        == expected_train["ordered_ids_sha256"],
        "official-train ordered-ID SHA differs",
    )
    _require(
        len(development_ids) == 170
        and len(validation_ids) == 43
        and all(
            isinstance(identifier, str)
            for identifier in (*development_ids, *validation_ids)
        )
        and not (set(development_ids) & set(validation_ids))
        and set(development_ids) | set(validation_ids) == set(official_ids),
        "170/43 internal split partition differs",
    )
    _require(
        split.get("official_train_index_sha256") == expected_train["file_sha256"],
        "official-train index SHA binding differs",
    )
    mask_records = split.get("mask_stats")
    _require(
        isinstance(mask_records, list)
        and len(mask_records) == len(official_ids)
        and [record.get("identifier") for record in mask_records if isinstance(record, Mapping)]
        == official_ids,
        "internal split mask-stat records differ",
    )
    try:
        mask_stats = [trainer.MaskStats(**dict(record)) for record in mask_records]
        expected_development, expected_validation = trainer.stratified_split(
            mask_stats,
            trainer.VAL_FRACTION,
            trainer.SPLIT_SEED,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise PBDRV3EvaluationProtocolError(
            f"cannot replay internal split: {error}"
        ) from error
    _require(
        development_ids == expected_development
        and validation_ids == expected_validation,
        "internal split does not replay from frozen mask stats",
    )

    manifest_binding = split.get("data_protocol_manifest")
    _require(
        isinstance(manifest_binding, Mapping)
        and isinstance(manifest_binding.get("path"), str)
        and isinstance(manifest_binding.get("sha256"), str),
        "data protocol manifest binding is malformed",
    )
    manifest_path = Path(str(manifest_binding["path"]))
    _require(not manifest_path.is_symlink(), "data protocol manifest is a symlink")
    manifest_path = manifest_path.resolve(strict=True)
    _require(
        models.file_sha256(manifest_path) == manifest_binding["sha256"],
        "data protocol manifest SHA differs",
    )
    return path, split


@dataclass(frozen=True, slots=True)
class ValidatedRun:
    run_dir: Path
    summary_path: Path
    protocol_path: Path
    split_path: Path
    candidate_path: Path
    summary: Mapping[str, Any]
    protocol: Mapping[str, Any]
    split_manifest: Mapping[str, Any]
    candidate: Mapping[str, Any]
    candidate_state: Mapping[str, torch.Tensor]
    candidate_sha256: str
    protocol_sha256: str
    parent_role: str
    recipe: str
    selected_threshold: float
    internal_decision: Mapping[str, Any]
    parent_checkpoint: Mapping[str, Any]
    runtime_sources: Mapping[str, Mapping[str, Any]]


def validate_completed_run(
    run_dir: Path,
    *,
    parent_checkpoint: Path | None = None,
) -> ValidatedRun:
    """Validate every training artifact without touching official test data."""

    resolved = Path(run_dir).resolve(strict=True)
    summary_path = resolved / "summary.json"
    protocol_path = resolved / "protocol.json"
    summary = _json_object(summary_path, label="trainer summary")
    protocol = _json_object(protocol_path, label="trainer protocol")

    for label, payload in (("summary", summary), ("protocol", protocol)):
        _require(payload.get("schema") == trainer.SCHEMA, f"{label} schema differs")
        _require(
            payload.get("parent_role") in models.PARENT_ROLES,
            f"{label} parent role differs",
        )
        _require(payload.get("recipe") in trainer.RECIPES, f"{label} recipe differs")
        _require(
            payload.get("official_test_accessed") is False,
            f"{label} claims official-test access",
        )
    _require(summary.get("status") == "complete", "trainer summary is incomplete")
    _require(
        summary.get("dataset") == models.DATASET
        and summary.get("seed") == trainer.TRAINING_SEED,
        "trainer summary dataset/seed differs",
    )
    parent_role = str(summary["parent_role"])
    recipe = str(summary["recipe"])
    _require(protocol.get("parent_role") == parent_role, "summary/protocol role differs")
    _require(protocol.get("recipe") == recipe, "summary/protocol recipe differs")
    formal_controls = {
        "mode": "formal",
        "dataset": models.DATASET,
        "training_seed": trainer.TRAINING_SEED,
        "batch_size": trainer.FORMAL_BATCH_SIZE,
        "workers": trainer.FORMAL_WORKERS,
        "device": "cuda:0",
        "expected_gpu_uuid": GPU0_UUID,
        "precision": "fp32",
        "fixed_threshold": trainer.FORMAL_THRESHOLD,
        "threshold_grid": list(trainer.THRESHOLDS),
        "split_seed": trainer.SPLIT_SEED,
        "val_fraction": trainer.VAL_FRACTION,
        "smoke_limits": {"max_train_images": None, "max_val_images": None},
    }
    for field, expected in formal_controls.items():
        _require(protocol.get(field) == expected, f"formal {field} differs")
    _require(protocol.get("epochs") == trainer.FORMAL_EPOCHS, "formal epoch count differs")
    _require(
        protocol.get("eval_every") == trainer.FORMAL_EVAL_EVERY,
        "formal evaluation cadence differs",
    )
    optimizer = protocol.get("optimizer")
    _require(isinstance(optimizer, Mapping), "protocol lacks optimizer")
    _require(
        optimizer.get("name") == "AdamW"
        and optimizer.get("lr") == trainer.FORMAL_LR
        and optimizer.get("weight_decay") == trainer.FORMAL_WEIGHT_DECAY,
        "formal optimizer differs",
    )

    declared_protocol_sha = protocol.get("protocol_sha256")
    _require(
        isinstance(declared_protocol_sha, str) and len(declared_protocol_sha) == 64,
        "protocol_sha256 is malformed",
    )
    unsigned_protocol = dict(protocol)
    del unsigned_protocol["protocol_sha256"]
    computed_protocol_sha = models.canonical_sha256(unsigned_protocol)
    _require(
        declared_protocol_sha == computed_protocol_sha,
        "protocol canonical SHA differs",
    )
    _require(
        summary.get("protocol_sha256") == declared_protocol_sha,
        "summary protocol_sha256 differs",
    )

    source_locks = protocol.get("source_locks")
    _require(isinstance(source_locks, Mapping), "protocol lacks source_locks")
    _require(
        source_locks.get("protocol_document") == models.file_sha256(PROTOCOL_DOCUMENT),
        "PBDR-V3 protocol document SHA differs",
    )
    runtime_sources = _validate_runtime_sources(source_locks.get("runtime_sources"))
    split_path, split_manifest = _validate_split_manifest(resolved, source_locks)
    protocol_data_root = protocol.get("data_root")
    _require(
        isinstance(protocol_data_root, str)
        and str(Path(protocol_data_root).resolve(strict=True)) == protocol_data_root,
        "trainer data-root binding differs",
    )
    _require(
        _canonical_equal(
            protocol.get("data_protocol_manifest"),
            split_manifest.get("data_protocol_manifest"),
        ),
        "trainer data protocol binding differs from internal split",
    )

    selected_binding = summary.get("selected_checkpoint")
    _require(
        isinstance(selected_binding, Mapping),
        "summary selected_checkpoint must bind path and sha256",
    )
    selected_path_value = selected_binding.get("path")
    selected_sha = selected_binding.get("sha256")
    _require(
        isinstance(selected_path_value, str) and bool(selected_path_value),
        "selected checkpoint path is malformed",
    )
    _require(
        isinstance(selected_sha, str) and len(selected_sha) == 64,
        "selected checkpoint SHA is malformed",
    )
    candidate_path = Path(selected_path_value)
    if not candidate_path.is_absolute():
        candidate_path = resolved / candidate_path
    _require(
        not candidate_path.is_symlink(),
        "selected checkpoint must not be a symlink",
    )
    candidate_path = candidate_path.resolve(strict=True)
    _require(
        candidate_path == (resolved / "selected_candidate.pth.tar").resolve(strict=True),
        "selected checkpoint path differs from the run-owned candidate",
    )
    candidate_sha = models.file_sha256(candidate_path)
    _require(candidate_sha == selected_sha, "selected checkpoint SHA differs")

    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    _require(isinstance(candidate, Mapping), "candidate checkpoint must be a mapping")
    for field, expected in (
        ("schema", trainer.SCHEMA),
        ("parent_role", parent_role),
        ("recipe", recipe),
        ("protocol_sha256", declared_protocol_sha),
    ):
        _require(candidate.get(field) == expected, f"candidate {field} differs")
    _require(
        candidate.get("source_locks") == source_locks,
        "candidate source locks differ from protocol",
    )
    selected_epoch = summary.get("selected_epoch")
    _require(
        isinstance(selected_epoch, int)
        and not isinstance(selected_epoch, bool)
        and candidate.get("epoch") == selected_epoch,
        "selected candidate epoch differs",
    )

    selected_threshold = _finite_float(
        summary.get("selected_threshold"),
        "summary.selected_threshold",
    )
    _require(
        selected_threshold in trainer.THRESHOLDS,
        "selected threshold is outside the frozen internal grid",
    )
    _require(
        candidate.get("selected_threshold") == selected_threshold,
        "candidate selected_threshold differs from summary",
    )
    validation = candidate.get("validation")
    _require(isinstance(validation, Mapping), "candidate lacks internal validation")
    fixed_validation = validation.get("fixed_0_5")
    threshold_sweep = validation.get("candidate_threshold_sweep")
    _require(
        isinstance(fixed_validation, Mapping)
        and isinstance(threshold_sweep, Mapping),
        "candidate internal validation is malformed",
    )
    _require(
        set(threshold_sweep) == {f"{value:.2f}" for value in trainer.THRESHOLDS},
        "candidate internal threshold grid differs",
    )
    try:
        recomputed_threshold, recomputed_selection = (
            trainer.select_validation_threshold(parent_role, validation)
        )
    except (TypeError, ValueError, KeyError) as error:
        raise PBDRV3EvaluationProtocolError(
            f"cannot replay internal threshold selection: {error}"
        ) from error
    _require(
        recomputed_threshold == selected_threshold,
        "selected threshold does not replay from internal validation",
    )
    summary_threshold_selection = summary.get("threshold_selection")
    candidate_threshold_selection = candidate.get("threshold_selection")
    _require(
        _canonical_equal(summary_threshold_selection, recomputed_selection)
        and _canonical_equal(candidate_threshold_selection, recomputed_selection),
        "internal threshold-selection artifact differs",
    )
    internal_decision = _validate_internal_decision(
        summary.get("internal_certification_fixed_0_5")
    )
    _require(
        summary.get("internal_gate_passed") is internal_decision["passed"],
        "summary internal gate flag differs",
    )
    try:
        fixed_current = gate.CertificationMetrics.from_mapping(
            fixed_validation["current"]
        )
        fixed_candidate = gate.CertificationMetrics.from_mapping(
            fixed_validation["candidate"]
        )
        decision_current = gate.CertificationMetrics.from_mapping(
            internal_decision["current"]
        )
        decision_candidate = gate.CertificationMetrics.from_mapping(
            internal_decision["candidate"]
        )
    except (TypeError, ValueError, KeyError) as error:
        raise PBDRV3EvaluationProtocolError(
            f"fixed internal validation binding differs: {error}"
        ) from error
    _require(
        fixed_current == decision_current and fixed_candidate == decision_candidate,
        "internal certification is not bound to selected checkpoint validation",
    )
    candidate_decision = candidate.get("internal_certification_fixed_0_5")
    _require(
        _canonical_equal(candidate_decision, internal_decision),
        "candidate internal certification differs from summary",
    )
    certification_path = resolved / "internal_certification.json"
    certification = _json_object(
        certification_path,
        label="internal certification",
    )
    certification_core = {
        name: certification.get(name)
        for name in ("passed", "selected", "checks", "current", "candidate")
    }
    decision_core = {
        name: internal_decision[name]
        for name in ("passed", "selected", "checks", "current", "candidate")
    }
    _require(
        certification.get("scope") == "frozen_certification_split_only",
        "gate artifact certification scope differs",
    )
    _require(
        _canonical_equal(certification_core, decision_core),
        "internal certification artifact differs from summary",
    )

    state = candidate.get("state_dict")
    _require(isinstance(state, Mapping), "candidate lacks state_dict")
    _require(
        len(state) == models.TRAINING_STATE_KEY_COUNT,
        "candidate training state-key count differs",
    )
    _require(
        all(isinstance(name, str) for name in state),
        "candidate state keys must be strings",
    )
    _require(
        all(isinstance(value, torch.Tensor) for value in state.values()),
        "candidate state values must be tensors",
    )
    for name, tensor in state.items():
        _require(bool(torch.isfinite(tensor).all()), f"candidate tensor {name} is non-finite")

    parent_payload, parent_state, parent_record = models.load_current_checkpoint(
        parent_role,
        parent_checkpoint,
    )
    del parent_payload
    parent_binding = protocol.get("model")
    _require(isinstance(parent_binding, Mapping), "protocol lacks model binding")
    frozen_parent = parent_binding.get("parent_checkpoint")
    _require(isinstance(frozen_parent, Mapping), "protocol lacks parent checkpoint binding")
    for field in ("path", "sha256", "state_sha256", "state_key_count", "checkpoint_role"):
        _require(
            frozen_parent.get(field) == parent_record[field],
            f"protocol parent checkpoint {field} differs",
        )
    _require(
        source_locks.get("parent_checkpoint") == parent_record["sha256"],
        "source-lock parent checkpoint SHA differs",
    )
    checkpoint_parent = candidate.get("parent_checkpoint")
    _require(
        isinstance(checkpoint_parent, Mapping)
        and checkpoint_parent.get("sha256") == parent_record["sha256"],
        "candidate parent checkpoint binding differs",
    )
    candidate_base = {
        name: tensor
        for name, tensor in state.items()
        if not name.startswith("pbdr_v3.")
    }
    _require(
        set(candidate_base) == set(parent_state),
        "candidate frozen-base key set differs from Current",
    )
    _require(
        models.tensor_mapping_sha256(candidate_base) == parent_record["state_sha256"],
        "candidate frozen-base SHA differs from Current",
    )
    _require(
        candidate.get("base_state_sha256") == parent_record["state_sha256"],
        "candidate base-state binding differs",
    )
    before_after = summary.get("base_state_sha256_before_after")
    _require(
        before_after == [parent_record["state_sha256"], parent_record["state_sha256"]],
        "summary frozen-base before/after SHA differs",
    )
    batchnorm_before_after = summary.get("batchnorm_buffer_sha256_before_after")
    _require(
        isinstance(batchnorm_before_after, list)
        and len(batchnorm_before_after) == 2
        and batchnorm_before_after[0] == batchnorm_before_after[1],
        "summary BatchNorm buffers changed",
    )
    freeze_before = protocol.get("freeze_before")
    _require(
        isinstance(freeze_before, Mapping)
        and freeze_before.get("trainable_parameter_count") == 6018
        and freeze_before.get("base_state_sha256") == parent_record["state_sha256"]
        and freeze_before.get("batchnorm_buffer_sha256")
        == batchnorm_before_after[0],
        "formal Stage-1 freeze audit differs",
    )
    _require(
        candidate.get("batchnorm_buffer_sha256_before")
        == batchnorm_before_after[0]
        and candidate.get("batchnorm_buffer_sha256_after")
        == batchnorm_before_after[1],
        "candidate BatchNorm audit differs",
    )

    return ValidatedRun(
        run_dir=resolved,
        summary_path=summary_path,
        protocol_path=protocol_path,
        split_path=split_path,
        candidate_path=candidate_path,
        summary=summary,
        protocol=protocol,
        split_manifest=split_manifest,
        candidate=candidate,
        candidate_state=state,
        candidate_sha256=candidate_sha,
        protocol_sha256=declared_protocol_sha,
        parent_role=parent_role,
        recipe=recipe,
        selected_threshold=selected_threshold,
        internal_decision=internal_decision,
        parent_checkpoint=parent_record,
        runtime_sources=runtime_sources,
    )


def _artifact_binding(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": models.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _base_deployment(validated: ValidatedRun) -> dict[str, Any]:
    return {
        "schema": DEPLOYMENT_SCHEMA,
        "status": "complete",
        "dataset": models.DATASET,
        "parent_role": validated.parent_role,
        "recipe": validated.recipe,
        "protocol_sha256": validated.protocol_sha256,
        "trainer_summary": _artifact_binding(validated.summary_path),
        "trainer_protocol": _artifact_binding(validated.protocol_path),
        "internal_split_manifest": _artifact_binding(validated.split_path),
        "internal_certification_fixed_0_5": dict(validated.internal_decision),
        "unseen_test_guarantee": False,
    }


def _current_fallback_deployment(validated: ValidatedRun) -> dict[str, Any]:
    deployment = _base_deployment(validated)
    deployment.update(
        {
            "selected": "current",
            "selection_reason": "internal_certification_fixed_0_5_failed",
            "selected_threshold": FIXED_THRESHOLD,
            "selected_artifact": dict(validated.parent_checkpoint),
            "candidate_artifact": {
                "path": str(validated.candidate_path),
                "sha256": validated.candidate_sha256,
                "evaluated_on_official_test": False,
            },
            "official_test_accessed": False,
            "evaluation": None,
        }
    )
    return deployment


def _extract_hw(sizes: Any) -> tuple[int, int]:
    if isinstance(sizes, torch.Tensor):
        value = sizes.detach().cpu()
        if value.ndim == 2:
            return int(value[0, 0]), int(value[0, 1])
        if value.ndim == 1 and value.numel() == 2:
            return int(value[0]), int(value[1])
    if isinstance(sizes, (tuple, list)) and len(sizes) == 2:
        values: list[int] = []
        for item in sizes:
            if isinstance(item, torch.Tensor):
                item = item.reshape(-1)[0].item()
            values.append(int(item))
        return values[0], values[1]
    raise TypeError(f"unsupported collated size value: {type(sizes)!r}")


class NUAAOfficialTestDataset(Dataset):
    """NUAA-only loader that opens its frozen official-test index once."""

    def __init__(self, dataset_root: Path) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.sample_ids = data_protocol.load_index(
            self.dataset_root,
            models.DATASET,
            "test",
        )
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = data_protocol.get_legacy_normalization(models.DATASET)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], str]:
        sample_id = self.sample_ids[index]
        resolved = data_protocol.resolve_sample(
            self.dataset_root,
            models.DATASET,
            sample_id,
            split="test",
            known_ids=self._known_ids,
        )
        with Image.open(resolved.image_path) as image:
            image_array = np.asarray(image.convert("I"), dtype=np.float32)
        with Image.open(resolved.mask_path) as mask:
            mask_array = np.asarray(mask, dtype=np.float32)
        if mask_array.ndim > 2:
            mask_array = mask_array[:, :, 0]
        _require(
            image_array.ndim == mask_array.ndim == 2
            and image_array.shape == mask_array.shape,
            f"official image/mask dimensions differ: {sample_id}",
        )
        _require(
            bool(np.isfinite(image_array).all())
            and bool(np.isfinite(mask_array).all()),
            f"official image/mask contains non-finite pixels: {sample_id}",
        )
        height, width = image_array.shape
        image_array = (
            image_array - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        mask_array = mask_array / np.float32(255.0)
        padded_height = ((height + 31) // 32) * 32
        padded_width = ((width + 31) // 32) * 32
        image_array = np.pad(
            image_array,
            ((0, padded_height - height), (0, padded_width - width)),
        )
        mask_array = np.pad(
            mask_array,
            ((0, padded_height - height), (0, padded_width - width)),
        )
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_array[None], dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_array[None], dtype=np.float32)
        )
        return image_tensor, mask_tensor, (height, width), sample_id


@torch.inference_mode()
def _collect_candidate_and_current(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Collect routed and exact router-bypass Current maps in one forward."""

    model.eval()
    model.mode = "test"
    criterion = nn.BCELoss(reduction="mean")
    candidate_probabilities: list[np.ndarray] = []
    current_probabilities: list[np.ndarray] = []
    candidate_losses: list[float] = []
    current_losses: list[float] = []
    targets: list[np.ndarray] = []
    identifiers: list[str] = []

    for images, masks, sizes, sample_ids in loader:
        _require(
            int(images.shape[0]) == int(masks.shape[0]) == 1,
            "official evaluator requires batch_size=1",
        )
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height, width = _extract_hw(sizes)
        _, auxiliary = model.forward_for_pbdr_v3_training(images)
        candidate = torch.sigmoid(auxiliary.routed_logits)[:, :, :height, :width]
        current = torch.sigmoid(auxiliary.base_logits)[:, :, :height, :width]
        target = masks[:, :, :height, :width]
        _require(
            candidate.shape == current.shape == target.shape,
            "candidate/Current/target shapes differ",
        )
        _require(
            bool(torch.isfinite(candidate).all())
            and bool(torch.isfinite(current).all())
            and bool(torch.isfinite(target).all()),
            "official inference produced non-finite values",
        )
        candidate_loss = criterion(candidate.float(), target.float())
        current_loss = criterion(current.float(), target.float())
        _require(
            math.isfinite(float(candidate_loss.item()))
            and math.isfinite(float(current_loss.item())),
            "official inference loss is non-finite",
        )
        candidate_probabilities.append(candidate[0, 0].float().cpu().numpy())
        current_probabilities.append(current[0, 0].float().cpu().numpy())
        targets.append(target[0, 0].float().cpu().numpy())
        candidate_losses.append(float(candidate_loss.item()))
        current_losses.append(float(current_loss.item()))
        _require(
            isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
            "official loader must yield one sample ID",
        )
        identifiers.append(str(sample_ids[0]))

    _require(
        bool(identifiers) and len(identifiers) == len(loader.dataset),
        "official prediction count differs from dataset",
    )
    _require(len(identifiers) == len(set(identifiers)), "duplicate official sample ID")
    return {
        "candidate_probabilities": candidate_probabilities,
        "current_probabilities": current_probabilities,
        "candidate_losses": candidate_losses,
        "current_losses": current_losses,
        "targets": targets,
        "identifiers": identifiers,
    }


def _metric_points(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    losses: Sequence[float],
    thresholds: Sequence[float],
) -> dict[str, dict[str, Any]]:
    from experiments import four_dataset_evaluation_protocol_v1 as metric_core

    unique = sorted({float(value) for value in thresholds})
    points = metric_core.strict_metric_points(
        probabilities,
        targets,
        losses,
        unique,
    )
    _require(len(points) == len(unique), "metric core returned an unexpected point count")
    return {f"{float(point['threshold']):.2f}": dict(point) for point in points}


def _official_decision(
    parent_role: str,
    candidate_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = OFFICIAL_ROLE_THRESHOLDS[parent_role]
    target_count = int(candidate_fixed.get("target_count", -1))
    matched = int(candidate_fixed.get("matched_target_count", -1))
    fa = _finite_float(candidate_fixed.get("fa"), "candidate fixed Fa")
    miou = _finite_float(candidate_fixed.get("miou"), "candidate fixed mIoU")
    niou = _finite_float(candidate_fixed.get("niou"), "candidate fixed nIoU")
    checks = {
        "target_count_exact": target_count == int(thresholds["target_count"]),
        "matched_target_count_minimum": matched
        >= int(thresholds["minimum_matched_target_count"]),
        "fa_maximum": fa <= float(thresholds["maximum_fa"]),
        "miou_minimum": miou >= float(thresholds["minimum_miou"]),
        "niou_minimum": niou >= float(thresholds["minimum_niou"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "selected": "candidate" if passed else "current",
        "checks": checks,
        "thresholds": dict(thresholds),
        "threshold": FIXED_THRESHOLD,
        "source": "experiments/PBDR_V3_PROTOCOL.md",
        "source_sha256": models.file_sha256(PROTOCOL_DOCUMENT),
    }


def _formal_official_device(device_name: str) -> torch.device:
    """Attest the sole GPU authorized for official-test inference."""

    _require(device_name == "cuda:0", "official-test evaluation requires cuda:0")
    _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one GPU must be visible")
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU0_UUID,
        "official-test CUDA_VISIBLE_DEVICES differs from GPU0 UUID",
    )
    observed = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if observed and not observed.startswith("GPU-"):
        observed = f"GPU-{observed}"
    _require(observed == GPU0_UUID, "visible GPU UUID differs from physical GPU0")
    return torch.device("cuda:0")


def _require_fresh_official_outputs(
    evaluation_path: Path,
    deployment_path: Path,
) -> None:
    resolved_evaluation = Path(evaluation_path).resolve(strict=False)
    resolved_deployment = Path(deployment_path).resolve(strict=False)
    _require(
        resolved_evaluation != resolved_deployment,
        "evaluation and deployment outputs must be distinct",
    )
    for label, path in (
        ("evaluation", Path(evaluation_path)),
        ("deployment", Path(deployment_path)),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"{label} output already exists before official-test access: {path}"
            )


def _claim_official_test_access(
    validated: ValidatedRun,
    claim_path: Path,
) -> dict[str, Any]:
    """Durably claim the run's one allowed official-test evaluation."""

    destination = Path(claim_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    payload = {
        "schema": OFFICIAL_ACCESS_CLAIM_SCHEMA,
        "status": "claimed_before_dataset_construction",
        "dataset": models.DATASET,
        "parent_role": validated.parent_role,
        "recipe": validated.recipe,
        "protocol_sha256": validated.protocol_sha256,
        "candidate_sha256": validated.candidate_sha256,
        "authorization": "internal_certification_fixed_0_5_passed",
        "internal_certification_fixed_0_5": dict(validated.internal_decision),
        "maximum_official_test_evaluations": 1,
    }
    content = (
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as error:
        raise PBDRV3EvaluationProtocolError(
            "official-test access was already claimed for this run"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Once O_EXCL succeeds, preserve the claim even if durability or later
        # evaluation fails.  A protocol amendment is required for another
        # official-test access.
        raise
    return _artifact_binding(destination)


def _evaluate_official_test(
    validated: ValidatedRun,
    *,
    data_root: Path,
    protocol_manifest: Path,
    device: torch.device,
    workers: int,
    access_claim_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enter the official-test path after internal certification only."""

    _require(
        validated.internal_decision.get("passed") is True,
        "official-test evaluation requires passed internal certification",
    )
    trainer.configure_determinism()
    _require(
        torch.backends.cuda.matmul.allow_tf32 is False
        and torch.backends.cudnn.allow_tf32 is False,
        "official-test TF32 controls differ",
    )
    model, model_metadata = models.build_inference_model_from_candidate_state(
        validated.candidate_state,
        parent_role=validated.parent_role,
        parent_checkpoint=Path(str(validated.parent_checkpoint["path"])),
    )
    _require(model_metadata.get("strict_load") is True, "candidate inference load not strict")
    _require(
        model_metadata.get("base_bitwise_equal_to_parent") is True,
        "candidate inference base differs from Current",
    )
    _require(
        model_metadata.get("inference_state_key_count") == models.INFERENCE_STATE_KEY_COUNT,
        "candidate inference state-key count differs",
    )
    model.to(device)

    manifest_path = Path(protocol_manifest).resolve(strict=True)
    data_root_path = Path(data_root).resolve(strict=True)
    frozen_manifest = validated.split_manifest["data_protocol_manifest"]
    _require(
        manifest_path == Path(str(frozen_manifest["path"])).resolve(strict=True)
        and models.file_sha256(manifest_path) == frozen_manifest["sha256"],
        "official evaluator data protocol differs from training binding",
    )
    _require(
        str(data_root_path) == validated.protocol.get("data_root"),
        "official evaluator data root differs from training binding",
    )

    access_claim = _claim_official_test_access(validated, access_claim_path)

    # Construct only the NUAA loader after the durable one-use claim.  This
    # intentionally avoids whole-manifest validation that would open NUDT and
    # IRSTD official indexes outside the NUAA Stage-1 scope.
    dataset = NUAAOfficialTestDataset(data_root_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    cache = _collect_candidate_and_current(model, loader, device)
    identifiers = cache["identifiers"]
    _require(
        identifiers == list(dataset.sample_ids),
        "official inference order differs from frozen test index",
    )
    thresholds = (FIXED_THRESHOLD, validated.selected_threshold)
    candidate_points = _metric_points(
        cache["candidate_probabilities"],
        cache["targets"],
        cache["candidate_losses"],
        thresholds,
    )
    current_points = _metric_points(
        cache["current_probabilities"],
        cache["targets"],
        cache["current_losses"],
        thresholds,
    )
    fixed_key = f"{FIXED_THRESHOLD:.2f}"
    selected_key = f"{validated.selected_threshold:.2f}"
    official_decision = _official_decision(
        validated.parent_role,
        candidate_points[fixed_key],
    )
    inference_order_newline_sha = hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()
    ordered_ids_sha = data_protocol.ordered_ids_sha256(identifiers)
    expected_test = data_protocol.EXPECTED_SPLITS[models.DATASET]["test"]
    _require(len(identifiers) == int(expected_test["count"]), "official test count differs")
    _require(
        ordered_ids_sha == expected_test["ordered_ids_sha256"],
        "official test ordered-ID SHA differs",
    )

    evaluation = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": models.DATASET,
        "parent_role": validated.parent_role,
        "recipe": validated.recipe,
        "seed": models.TRAINING_SEED,
        "official_test_accessed": True,
        "official_test_access_authorization": "internal_certification_fixed_0_5_passed",
        "official_test_access_claim": access_claim,
        "internal_certification_fixed_0_5": dict(validated.internal_decision),
        "thresholds": {
            "fixed_comparable": FIXED_THRESHOLD,
            "internal_validation_selected": validated.selected_threshold,
            "selection_from_official_test": False,
        },
        "metrics": {
            "current": {
                "fixed_0_5": current_points[fixed_key],
                "validation_selected": current_points[selected_key],
            },
            "candidate": {
                "fixed_0_5": candidate_points[fixed_key],
                "validation_selected": candidate_points[selected_key],
            },
        },
        "official_fixed_0_5_gate": official_decision,
        "candidate_checkpoint": {
            "path": str(validated.candidate_path),
            "sha256": validated.candidate_sha256,
            "epoch": int(validated.candidate["epoch"]),
        },
        "current_parent_checkpoint": dict(validated.parent_checkpoint),
        "model": model_metadata,
        "exact_current_evaluation": {
            "source": "same_strict_inference_forward_router_bypass_base_logits",
            "candidate_base_bitwise_equal_to_parent": True,
            "separate_current_approximation": False,
        },
        "data": {
            "dataset_root": str(Path(data_root).resolve()),
            "protocol_manifest": {
                "path": str(manifest_path),
                "sha256": models.file_sha256(manifest_path),
            },
            "split": "img_idx/test",
            "test_count": len(identifiers),
            "img_idx_test_sha256": expected_test["file_sha256"],
            "img_idx_test_ordered_ids_sha256": expected_test["ordered_ids_sha256"],
            "inference_order_newline_sha256": inference_order_newline_sha,
            "normalization": data_protocol.get_legacy_normalization(models.DATASET),
        },
        "metric_protocol": {
            "implementation": "experiments.train_tpd_pilot.ValidationMetrics",
            "prediction_comparison": "probability > threshold",
            "connectivity": 8,
            "match_radius": trainer.FORMAL_MATCH_RADIUS,
            "tiny_area": trainer.FORMAL_TINY_AREA,
        },
        "protocol_sha256": validated.protocol_sha256,
        "runtime_sources": dict(validated.runtime_sources),
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": models.file_sha256(Path(__file__).resolve()),
        },
        "no_fabricated_results": True,
    }

    del model, loader, cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    selected = str(official_decision["selected"])
    deployment = _base_deployment(validated)
    deployment.update(
        {
            "selected": selected,
            "selection_reason": (
                "official_fixed_0_5_gate_passed"
                if selected == "candidate"
                else "official_fixed_0_5_gate_failed"
            ),
            "selected_threshold": (
                validated.selected_threshold
                if selected == "candidate"
                else FIXED_THRESHOLD
            ),
            "selected_artifact": (
                {
                    "path": str(validated.candidate_path),
                    "sha256": validated.candidate_sha256,
                    "kind": "pbdr_v3_training_checkpoint_strict_inference_conversion",
                }
                if selected == "candidate"
                else dict(validated.parent_checkpoint)
            ),
            "candidate_artifact": {
                "path": str(validated.candidate_path),
                "sha256": validated.candidate_sha256,
                "evaluated_on_official_test": True,
            },
            "official_test_accessed": True,
            "official_test_access_claim": access_claim,
            "official_fixed_0_5_gate": official_decision,
        }
    )
    return evaluation, deployment


def run(
    *,
    run_dir: Path,
    data_root: Path = data_protocol.DEFAULT_DATASET_ROOT,
    protocol_manifest: Path = data_protocol.DEFAULT_MANIFEST_PATH,
    parent_checkpoint: Path | None = None,
    device_name: str = "cuda:0",
    workers: int = 0,
    evaluation_output: Path | None = None,
    deployment_output: Path | None = None,
    overwrite: bool = False,
) -> Path:
    if workers < 0:
        raise ValueError("workers must be non-negative")
    resolved_run = Path(run_dir).resolve(strict=True)
    lock_path = resolved_run / "evaluation.lock"
    if lock_path.is_symlink():
        raise FileExistsError(f"refusing evaluation lock symlink: {lock_path}")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError("another evaluator holds this run lock") from error
        try:
            validated = validate_completed_run(
                resolved_run,
                parent_checkpoint=parent_checkpoint,
            )
            evaluation_path = Path(
                evaluation_output or validated.run_dir / "evaluation.json"
            )
            deployment_path = Path(
                deployment_output or validated.run_dir / "deployment.json"
            )

            # This return precedes device checks, inference construction,
            # manifest validation, test-index loading, and all sample access.
            if validated.internal_decision.get("passed") is False:
                fallback = _current_fallback_deployment(validated)
                _atomic_write_json(
                    deployment_path,
                    fallback,
                    overwrite=overwrite,
                )
                return deployment_path.resolve()

            device = _formal_official_device(device_name)
            _require_fresh_official_outputs(evaluation_path, deployment_path)
            access_claim_path = (
                validated.run_dir / "official_test_access_claim.json"
            )
            evaluation, deployment = _evaluate_official_test(
                validated,
                data_root=data_root,
                protocol_manifest=protocol_manifest,
                device=device,
                workers=workers,
                access_claim_path=access_claim_path,
            )
            # Official outputs are immutable.  ``--overwrite`` remains useful
            # only for the no-test Current-fallback manifest.
            _atomic_write_json(evaluation_path, evaluation, overwrite=False)
            deployment["evaluation"] = _artifact_binding(evaluation_path)
            _atomic_write_json(deployment_path, deployment, overwrite=False)
            return deployment_path.resolve()
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_protocol.DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=data_protocol.DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--deployment-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    deployment = run(
        run_dir=args.run_dir,
        data_root=args.data_root,
        protocol_manifest=args.protocol_manifest,
        parent_checkpoint=args.parent_checkpoint,
        device_name=args.device,
        workers=args.workers,
        evaluation_output=args.evaluation_output,
        deployment_output=args.deployment_output,
        overwrite=args.overwrite,
    )
    payload = _json_object(deployment, label="deployment")
    print(
        json.dumps(
            {
                "status": "complete",
                "deployment": str(deployment),
                "deployment_sha256": models.file_sha256(deployment),
                "selected": payload["selected"],
                "official_test_accessed": payload["official_test_accessed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


__all__ = [
    "DEPLOYMENT_SCHEMA",
    "FIXED_THRESHOLD",
    "OFFICIAL_ROLE_THRESHOLDS",
    "PBDRV3EvaluationProtocolError",
    "SCHEMA",
    "ValidatedRun",
    "main",
    "parse_args",
    "run",
    "validate_completed_run",
]


if __name__ == "__main__":
    main()
