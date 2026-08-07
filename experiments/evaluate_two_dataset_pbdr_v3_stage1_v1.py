#!/usr/bin/env python3
"""Dataset-global PBDR-V3 evaluator for NUDT-SIRST and IRSTD-1K.

One invocation owns exactly one dataset.  It validates both formal ``core``
role runs and every Candidate/Current/Original checkpoint before creating a
durable dataset-level official-test claim.  After that claim, one test loader
is constructed and iterated exactly once; each sample is evaluated by both
PBDR-V3 candidates, both exact Current router bypasses, and both same-role
Original models.  Deployment uses the frozen role order at fixed probability
threshold 0.5 with no positive performance margin.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_nuaa_pbdr_v3_stage1_v1 as nuaa_tools  # noqa: E402
from experiments import pbdr_v3_zero_margin_role_gate as zero_gate  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import train_two_dataset_pbdr_v3_stage1_v1 as trainer  # noqa: E402
from experiments import two_dataset_pbdr_v3_models_seed42_v1 as models  # noqa: E402


SCHEMA = "sctransnet_two_dataset_pbdr_v3_stage1_evaluation_v1/v1"
DEPLOYMENT_SCHEMA = "sctransnet_two_dataset_pbdr_v3_stage1_deployment_v1/v1"
OFFICIAL_ACCESS_CLAIM_SCHEMA = (
    "sctransnet_two_dataset_pbdr_v3_official_test_access_claim_v1/v1"
)
PUBLICATION_BUNDLE_SCHEMA = (
    "sctransnet_two_dataset_pbdr_v3_official_publication_bundle_v1/v1"
)
FIXED_THRESHOLD = 0.5
DATASETS = models.DATASETS
PARENT_ROLES = models.PARENT_ROLES
PROTOCOL_DOCUMENT = trainer.PROTOCOL_DOCUMENT
DEFAULT_RESULTS_ROOT = trainer.DEFAULT_RESULTS_ROOT
DEFAULT_DATA_ROOT = trainer.DEFAULT_DATA_ROOT
DEFAULT_PROTOCOL_MANIFEST = trainer.DEFAULT_PROTOCOL_MANIFEST

# Reuse the completed NUAA evaluator's read-only/atomic primitives.  They do
# not carry a dataset binding and are covered by the cross-dataset source lock.
PBDRV3EvaluationProtocolError = nuaa_tools.PBDRV3EvaluationProtocolError
_require = nuaa_tools._require
_json_object = nuaa_tools._json_object
_json_ready = nuaa_tools._json_ready
_atomic_write_json = nuaa_tools._atomic_write_json
_finite_float = nuaa_tools._finite_float
_extract_hw = nuaa_tools._extract_hw
_metric_points = nuaa_tools._metric_points


def _canonical_equal(first: Any, second: Any) -> bool:
    return models.canonical_sha256(_json_ready(first)) == models.canonical_sha256(
        _json_ready(second)
    )


def _certification_metrics_equal(first: Any, second: Any) -> bool:
    """Compare only the frozen gate projection of full metric records."""

    try:
        first_ready = zero_gate.CertificationMetrics.from_mapping(first)
        second_ready = zero_gate.CertificationMetrics.from_mapping(second)
    except (TypeError, ValueError, KeyError):
        return False
    return first_ready == second_ready


def _artifact_binding(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": models.file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _checkpoint_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "dataset",
        "checkpoint_role",
        "path",
        "sha256",
        "bytes",
        "epoch",
        "state_key_count",
        "state_sha256",
        "schema",
        "protocol_sha256",
    )
    _require(
        all(name in record for name in required),
        "checkpoint record is incomplete",
    )
    return {name: _json_ready(record[name]) for name in required}


def _validate_internal_decision(role: str, value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "summary lacks internal certification")
    required = ("passed", "selected", "checks", "current", "candidate", "scope")
    _require(
        all(name in value for name in required),
        "internal certification fields are incomplete",
    )
    _require(
        value.get("scope") == "frozen_internal_validation_split",
        "internal certification scope differs",
    )
    current = zero_gate.CertificationMetrics.from_mapping(value["current"])
    candidate = zero_gate.CertificationMetrics.from_mapping(value["candidate"])
    expected = zero_gate.certify(role, current, candidate)
    _require(value.get("passed") is expected.passed, "internal pass flag differs")
    _require(value.get("selected") == expected.selected, "internal selection differs")
    _require(dict(value.get("checks", {})) == dict(expected.checks), "internal checks differ")
    _require(
        _certification_metrics_equal(value.get("current"), asdict(expected.current))
        and _certification_metrics_equal(
            value.get("candidate"), asdict(expected.candidate)
        ),
        "internal certification metric projection differs",
    )
    return {
        "passed": expected.passed,
        "selected": expected.selected,
        "checks": dict(expected.checks),
        "current": dict(value["current"]),
        "candidate": dict(value["candidate"]),
        "scope": str(value["scope"]),
        "role": role,
        "decisive_index": expected.decisive_index,
        "decisive_term": expected.decisive_term,
        "minimum_gain": 0.0,
    }


def _validate_runtime_sources(
    dataset_name: str,
    value: Any,
) -> dict[str, dict[str, Any]]:
    _require(isinstance(value, Mapping), "protocol lacks runtime source locks")
    expected = trainer.DatasetModelsAdapter(dataset_name).runtime_source_records()
    _require(set(value) == set(expected), "runtime source-lock set differs")
    verified: dict[str, dict[str, Any]] = {}
    for relative, current in expected.items():
        observed = value.get(relative)
        _require(isinstance(observed, Mapping), f"malformed source lock: {relative}")
        for field in ("path", "sha256", "bytes"):
            _require(
                observed.get(field) == current[field],
                f"runtime source lock {relative} {field} differs",
            )
        verified[relative] = dict(current)
    return verified


def _validate_split_manifest(
    dataset_name: str,
    run_dir: Path,
    source_locks: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Replay one train-only split without opening an official test index."""

    path = Path(run_dir) / "split_manifest.json"
    split = _json_object(path, label="cross-dataset internal split manifest")
    _require(
        split.get("schema") == "sctransnet_two_dataset_pbdr_v3_internal_split_v1/v1",
        "internal split schema differs",
    )
    _require(split.get("dataset") == dataset_name, "internal split dataset differs")
    _require(
        split.get("source_split") == "official_train_only"
        and split.get("official_test_index_opened") is False,
        "internal split is not train-only",
    )
    _require(
        split.get("split_seed") == trainer.engine.SPLIT_SEED
        and split.get("val_fraction") == trainer.engine.VAL_FRACTION,
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
        "source-lock split SHA differs",
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
    expected_train = data_protocol.EXPECTED_SPLITS[dataset_name]["train"]
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
        bool(development_ids)
        and bool(validation_ids)
        and len(development_ids) + len(validation_ids) == len(official_ids)
        and all(
            isinstance(identifier, str)
            for identifier in (*development_ids, *validation_ids)
        )
        and not (set(development_ids) & set(validation_ids))
        and set(development_ids) | set(validation_ids) == set(official_ids),
        "dynamic internal split partition differs",
    )
    _require(
        split.get("official_train_index_sha256") == expected_train["file_sha256"],
        "official-train index SHA binding differs",
    )
    mask_records = split.get("mask_stats")
    _require(
        isinstance(mask_records, list)
        and len(mask_records) == len(official_ids)
        and [
            record.get("identifier")
            for record in mask_records
            if isinstance(record, Mapping)
        ]
        == official_ids,
        "internal split mask-stat records differ",
    )
    try:
        stats = [trainer.engine.MaskStats(**dict(record)) for record in mask_records]
        replay_development, replay_validation = trainer.engine.stratified_split(
            stats,
            trainer.engine.VAL_FRACTION,
            trainer.engine.SPLIT_SEED,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise PBDRV3EvaluationProtocolError(
            f"cannot replay internal split: {error}"
        ) from error
    _require(
        development_ids == replay_development
        and validation_ids == replay_validation,
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
    candidate_state_sha256: str
    protocol_sha256: str
    dataset_name: str
    parent_role: str
    selected_threshold: float
    internal_decision: Mapping[str, Any]
    parent_checkpoint: Mapping[str, Any]
    runtime_sources: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ValidatedDatasetRuns:
    dataset_name: str
    formal_root: Path
    runs: Mapping[str, ValidatedRun]
    original_checkpoints: Mapping[str, Mapping[str, Any]]
    original_states: Mapping[str, Mapping[str, torch.Tensor]]
    original_authority: Mapping[str, Any]
    shared_split_sha256: str
    shared_runtime_sources_sha256: str


def validate_completed_run(
    dataset_name: str,
    run_dir: Path,
) -> ValidatedRun:
    """Validate one role run without constructing or opening official test."""

    _require(dataset_name in DATASETS, "unsupported dataset")
    resolved = Path(run_dir).resolve(strict=True)
    summary_path = resolved / "summary.json"
    protocol_path = resolved / "protocol.json"
    summary = _json_object(summary_path, label="trainer summary")
    protocol = _json_object(protocol_path, label="trainer protocol")

    for label, payload in (("summary", summary), ("protocol", protocol)):
        _require(payload.get("schema") == trainer.SCHEMA, f"{label} schema differs")
        _require(payload.get("dataset") == dataset_name, f"{label} dataset differs")
        _require(
            payload.get("parent_role") in PARENT_ROLES,
            f"{label} parent role differs",
        )
        _require(payload.get("recipe") == "core", f"{label} recipe is not core")
        _require(
            payload.get("official_test_accessed") is False,
            f"{label} claims official-test access",
        )
    _require(summary.get("status") == "complete", "trainer summary is incomplete")
    _require(summary.get("seed") == trainer.TRAINING_SEED, "trainer seed differs")
    role = str(summary["parent_role"])
    _require(protocol.get("parent_role") == role, "summary/protocol role differs")

    formal_controls = {
        "mode": "formal",
        "dataset": dataset_name,
        "training_seed": trainer.TRAINING_SEED,
        "batch_size": trainer.engine.FORMAL_BATCH_SIZE,
        "workers": trainer.engine.FORMAL_WORKERS,
        "device": "cuda:0",
        "expected_gpu_uuid": trainer.GPU_UUIDS[dataset_name],
        "precision": "fp32",
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_grid": [FIXED_THRESHOLD],
        "split_seed": trainer.engine.SPLIT_SEED,
        "val_fraction": trainer.engine.VAL_FRACTION,
        "smoke_limits": {"max_train_images": None, "max_val_images": None},
    }
    for field, expected in formal_controls.items():
        _require(protocol.get(field) == expected, f"formal {field} differs")
    _require(
        protocol.get("epochs") == trainer.engine.FORMAL_EPOCHS,
        "formal epoch count differs",
    )
    _require(
        protocol.get("eval_every") == trainer.engine.FORMAL_EVAL_EVERY,
        "formal evaluation cadence differs",
    )
    optimizer = protocol.get("optimizer")
    _require(isinstance(optimizer, Mapping), "protocol lacks optimizer")
    _require(
        optimizer.get("name") == "AdamW"
        and optimizer.get("lr") == trainer.engine.FORMAL_LR
        and optimizer.get("weight_decay") == trainer.engine.FORMAL_WEIGHT_DECAY,
        "formal optimizer differs",
    )

    declared_protocol_sha = protocol.get("protocol_sha256")
    _require(
        isinstance(declared_protocol_sha, str) and len(declared_protocol_sha) == 64,
        "protocol SHA is malformed",
    )
    unsigned_protocol = dict(protocol)
    del unsigned_protocol["protocol_sha256"]
    _require(
        models.canonical_sha256(unsigned_protocol) == declared_protocol_sha,
        "protocol canonical SHA differs",
    )
    _require(
        summary.get("protocol_sha256") == declared_protocol_sha,
        "summary protocol SHA differs",
    )

    source_locks = protocol.get("source_locks")
    _require(isinstance(source_locks, Mapping), "protocol lacks source locks")
    _require(
        source_locks.get("protocol_document") == models.file_sha256(PROTOCOL_DOCUMENT),
        "cross-dataset protocol document SHA differs",
    )
    runtime_sources = _validate_runtime_sources(
        dataset_name,
        source_locks.get("runtime_sources"),
    )
    split_path, split_manifest = _validate_split_manifest(
        dataset_name,
        resolved,
        source_locks,
    )
    data_root_value = protocol.get("data_root")
    _require(
        isinstance(data_root_value, str)
        and str(Path(data_root_value).resolve(strict=True)) == data_root_value,
        "trainer data-root binding differs",
    )
    manifest_binding = split_manifest.get("data_protocol_manifest")
    _require(
        isinstance(manifest_binding, Mapping)
        and isinstance(manifest_binding.get("path"), str),
        "trainer data protocol binding is malformed",
    )
    try:
        trainer.validate_frozen_data_binding(
            Path(data_root_value), Path(str(manifest_binding["path"]))
        )
    except (OSError, TypeError, ValueError) as error:
        raise PBDRV3EvaluationProtocolError(
            f"trainer frozen data binding differs: {error}"
        ) from error
    _require(
        _canonical_equal(
            protocol.get("data_protocol_manifest"),
            split_manifest.get("data_protocol_manifest"),
        ),
        "trainer data protocol differs from split binding",
    )

    selected_binding = summary.get("selected_checkpoint")
    _require(isinstance(selected_binding, Mapping), "summary lacks selected checkpoint")
    selected_path_value = selected_binding.get("path")
    selected_sha = selected_binding.get("sha256")
    _require(
        isinstance(selected_path_value, str)
        and isinstance(selected_sha, str)
        and len(selected_sha) == 64,
        "selected checkpoint binding is malformed",
    )
    candidate_path = Path(selected_path_value)
    if not candidate_path.is_absolute():
        candidate_path = resolved / candidate_path
    _require(not candidate_path.is_symlink(), "candidate checkpoint is a symlink")
    candidate_path = candidate_path.resolve(strict=True)
    _require(
        candidate_path == (resolved / "selected_candidate.pth.tar").resolve(strict=True),
        "candidate path differs from run-owned checkpoint",
    )
    candidate_sha = models.file_sha256(candidate_path)
    _require(candidate_sha == selected_sha, "candidate checkpoint SHA differs")

    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    _require(isinstance(candidate, Mapping), "candidate checkpoint must be a mapping")
    for field, expected in (
        ("schema", trainer.SCHEMA),
        ("parent_role", role),
        ("recipe", "core"),
        ("protocol_sha256", declared_protocol_sha),
    ):
        _require(candidate.get(field) == expected, f"candidate {field} differs")
    _require(candidate.get("source_locks") == source_locks, "candidate source locks differ")
    selected_epoch = summary.get("selected_epoch")
    _require(
        isinstance(selected_epoch, int)
        and not isinstance(selected_epoch, bool)
        and candidate.get("epoch") == selected_epoch,
        "selected candidate epoch differs",
    )

    selected_threshold = _finite_float(
        summary.get("selected_threshold"), "summary.selected_threshold"
    )
    _require(selected_threshold == FIXED_THRESHOLD, "selected threshold is not fixed 0.5")
    _require(
        candidate.get("selected_threshold") == FIXED_THRESHOLD,
        "candidate selected threshold differs",
    )
    validation = candidate.get("validation")
    _require(isinstance(validation, Mapping), "candidate lacks internal validation")
    fixed_validation = validation.get("fixed_0_5")
    threshold_sweep = validation.get("candidate_threshold_sweep")
    _require(
        isinstance(fixed_validation, Mapping)
        and isinstance(threshold_sweep, Mapping)
        and set(threshold_sweep) == {"0.50"},
        "candidate fixed-threshold validation differs",
    )
    try:
        recomputed_threshold, recomputed_selection = trainer.fixed_validation_threshold(
            role, validation
        )
    except (TypeError, ValueError, KeyError) as error:
        raise PBDRV3EvaluationProtocolError(
            f"cannot replay internal threshold selection: {error}"
        ) from error
    _require(recomputed_threshold == FIXED_THRESHOLD, "threshold replay differs")
    _require(
        _canonical_equal(summary.get("threshold_selection"), recomputed_selection)
        and _canonical_equal(candidate.get("threshold_selection"), recomputed_selection),
        "threshold selection artifact differs",
    )
    internal_decision = _validate_internal_decision(
        role,
        summary.get("internal_certification_fixed_0_5"),
    )
    _require(
        summary.get("internal_gate_passed") is internal_decision["passed"],
        "summary Current diagnostic flag differs",
    )
    for family in ("current", "candidate"):
        _require(
            _certification_metrics_equal(
                fixed_validation.get(family), internal_decision[family]
            ),
            f"internal {family} certification metric projection differs",
        )
    summary_decision_core = {
        name: internal_decision[name]
        for name in ("passed", "selected", "checks", "current", "candidate", "scope")
    }
    _require(
        _canonical_equal(
            candidate.get("internal_certification_fixed_0_5"),
            summary_decision_core,
        ),
        "candidate internal certification differs",
    )
    certification = _json_object(
        resolved / "internal_certification.json",
        label="internal certification artifact",
    )
    _require(certification.get("schema") == zero_gate.SCHEMA, "gate schema differs")
    _require(certification.get("role") == role, "gate role differs")
    _require(certification.get("minimum_gain") == 0.0, "gate margin differs")
    _require(
        certification.get("scope") == "frozen_certification_split_only",
        "gate scope differs",
    )
    for name in ("passed", "selected", "checks", "current", "candidate"):
        _require(
            _canonical_equal(certification.get(name), internal_decision[name]),
            f"gate artifact {name} differs",
        )
    _require(
        certification.get("decisive_index") == internal_decision["decisive_index"]
        and certification.get("decisive_term") == internal_decision["decisive_term"],
        "gate decisive term differs",
    )

    state = candidate.get("state_dict")
    _require(isinstance(state, Mapping), "candidate lacks state_dict")
    _require(
        len(state) == models.TRAINING_STATE_KEY_COUNT,
        "candidate training state-key count differs",
    )
    _require(
        all(isinstance(name, str) for name in state)
        and all(isinstance(value, torch.Tensor) for value in state.values()),
        "candidate state mapping is malformed",
    )
    for name, tensor in state.items():
        _require(bool(torch.isfinite(tensor).all()), f"candidate tensor {name} is non-finite")
    candidate_state_sha = models.tensor_mapping_sha256(state)  # type: ignore[arg-type]

    _, parent_state, parent_record = models.load_current_checkpoint(dataset_name, role)
    parent_binding = protocol.get("model")
    _require(isinstance(parent_binding, Mapping), "protocol lacks model binding")
    frozen_parent = parent_binding.get("parent_checkpoint")
    _require(isinstance(frozen_parent, Mapping), "protocol lacks Current binding")
    for field in ("path", "sha256", "state_sha256", "state_key_count", "checkpoint_role", "dataset"):
        _require(
            frozen_parent.get(field) == parent_record[field],
            f"protocol Current {field} differs",
        )
    _require(
        source_locks.get("parent_checkpoint") == parent_record["sha256"],
        "source-lock Current checkpoint differs",
    )
    checkpoint_parent = candidate.get("parent_checkpoint")
    _require(
        isinstance(checkpoint_parent, Mapping)
        and checkpoint_parent.get("sha256") == parent_record["sha256"]
        and checkpoint_parent.get("state_sha256") == parent_record["state_sha256"],
        "candidate Current binding differs",
    )
    candidate_base = {
        name: tensor for name, tensor in state.items() if not name.startswith("pbdr_v3.")
    }
    _require(set(candidate_base) == set(parent_state), "candidate base keys differ")
    _require(
        models.tensor_mapping_sha256(candidate_base) == parent_record["state_sha256"],
        "candidate frozen base differs from Current",
    )
    _require(
        candidate.get("base_state_sha256") == parent_record["state_sha256"],
        "candidate base-state binding differs",
    )
    _require(
        summary.get("base_state_sha256_before_after")
        == [parent_record["state_sha256"], parent_record["state_sha256"]],
        "summary frozen base changed",
    )
    batchnorm = summary.get("batchnorm_buffer_sha256_before_after")
    _require(
        isinstance(batchnorm, list)
        and len(batchnorm) == 2
        and batchnorm[0] == batchnorm[1],
        "summary BatchNorm buffers changed",
    )
    freeze = protocol.get("freeze_before")
    _require(
        isinstance(freeze, Mapping)
        and freeze.get("trainable_parameter_count") == 6018
        and freeze.get("base_state_sha256") == parent_record["state_sha256"]
        and freeze.get("batchnorm_buffer_sha256") == batchnorm[0],
        "formal Stage-1 freeze audit differs",
    )
    _require(
        candidate.get("batchnorm_buffer_sha256_before") == batchnorm[0]
        and candidate.get("batchnorm_buffer_sha256_after") == batchnorm[1],
        "candidate BatchNorm binding differs",
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
        candidate_state=state,  # type: ignore[arg-type]
        candidate_sha256=candidate_sha,
        candidate_state_sha256=candidate_state_sha,
        protocol_sha256=str(declared_protocol_sha),
        dataset_name=dataset_name,
        parent_role=role,
        selected_threshold=selected_threshold,
        internal_decision=internal_decision,
        parent_checkpoint=parent_record,
        runtime_sources=runtime_sources,
    )


def formal_run_directories(
    results_root: Path,
    dataset_name: str,
) -> dict[str, Path]:
    _require(dataset_name in DATASETS, "unsupported dataset")
    formal_root = (
        Path(results_root).resolve() / "runs" / dataset_name / "formal"
    )
    return {
        role: formal_root / role / "core"
        for role in PARENT_ROLES
    }


def validate_dataset_runs(
    dataset_name: str,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    run_directories: Mapping[str, Path] | None = None,
) -> ValidatedDatasetRuns:
    """Preflight both role runs and both Original checkpoints as one unit."""

    _require(dataset_name in DATASETS, "unsupported dataset")
    directories = dict(
        formal_run_directories(results_root, dataset_name)
        if run_directories is None
        else run_directories
    )
    _require(set(directories) == set(PARENT_ROLES), "both role run dirs are required")
    runs = {
        role: validate_completed_run(dataset_name, directories[role])
        for role in PARENT_ROLES
    }
    formal_roots: set[Path] = set()
    for role in PARENT_ROLES:
        _require(runs[role].parent_role == role, f"{role} run-directory role differs")
        _require(
            runs[role].run_dir.name == "core"
            and runs[role].run_dir.parent.name == role
            and runs[role].run_dir.parent.parent.name == "formal",
            f"{role} is not the formal role/core run",
        )
        formal_roots.add(runs[role].run_dir.parents[1])
    _require(len(formal_roots) == 1, "role runs do not share a dataset formal root")
    split_shas = {str(run.split_manifest["split_sha256"]) for run in runs.values()}
    _require(len(split_shas) == 1, "role runs do not share one frozen split")
    first = runs[PARENT_ROLES[0]]
    for run in runs.values():
        _require(
            _canonical_equal(run.split_manifest, first.split_manifest),
            "role split manifests differ",
        )
        _require(
            run.protocol.get("data_root") == first.protocol.get("data_root")
            and _canonical_equal(
                run.protocol.get("data_protocol_manifest"),
                first.protocol.get("data_protocol_manifest"),
            ),
            "role data bindings differ",
        )
        _require(
            _canonical_equal(run.runtime_sources, first.runtime_sources),
            "role runtime source locks differ",
        )

    originals: dict[str, Mapping[str, Any]] = {}
    original_states: dict[str, Mapping[str, torch.Tensor]] = {}
    authority: Mapping[str, Any] | None = None
    for role in PARENT_ROLES:
        _, state, record = models.load_original_checkpoint(dataset_name, role)
        originals[role] = record
        original_states[role] = state
        current_authority = record.get("authority_audit")
        _require(isinstance(current_authority, Mapping), "Original authority audit missing")
        if authority is None:
            authority = current_authority
        else:
            _require(
                _canonical_equal(current_authority, authority),
                "Original role authority audits differ",
            )
        policy = record.get("selection_policy")
        _require(
            isinstance(policy, Mapping)
            and policy.get("threshold") == FIXED_THRESHOLD
            and policy.get("test_selected") is True
            and policy.get("selection_is_optimistic") is True,
            "Original selection disclosure differs",
        )

    formal_root = next(iter(formal_roots))
    return ValidatedDatasetRuns(
        dataset_name=dataset_name,
        formal_root=formal_root,
        runs=runs,
        original_checkpoints=originals,
        original_states=original_states,
        original_authority=dict(authority or {}),
        shared_split_sha256=next(iter(split_shas)),
        shared_runtime_sources_sha256=models.canonical_sha256(first.runtime_sources),
    )


class OfficialTestDataset(Dataset):
    """One-dataset loader; its official index is opened only after claim."""

    def __init__(self, dataset_root: Path, dataset_name: str) -> None:
        super().__init__()
        _require(dataset_name in DATASETS, "unsupported dataset")
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.dataset_name = dataset_name
        self.sample_ids = data_protocol.load_index(
            self.dataset_root,
            self.dataset_name,
            "test",
        )
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = data_protocol.get_legacy_normalization(dataset_name)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], str]:
        sample_id = self.sample_ids[index]
        resolved = data_protocol.resolve_sample(
            self.dataset_root,
            self.dataset_name,
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
            f"official image/mask has non-finite pixels: {sample_id}",
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
        return (
            torch.from_numpy(np.ascontiguousarray(image_array[None], dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(mask_array[None], dtype=np.float32)),
            (height, width),
            sample_id,
        )


def _final_prediction(outputs: Any) -> torch.Tensor:
    evaluator = getattr(outputs, "evaluator_prediction", None)
    if callable(evaluator):
        return evaluator()
    if isinstance(outputs, (tuple, list)):
        return outputs[-1]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"unsupported Original model output: {type(outputs)!r}")


def _assert_fp32_model(model: nn.Module, label: str) -> None:
    non_fp32 = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.is_floating_point() and value.dtype != torch.float32
    ]
    _require(not non_fp32, f"{label} has non-FP32 tensors: {non_fp32[:5]}")


@torch.inference_mode()
def _collect_six_models_one_pass(
    candidate_models: Mapping[str, nn.Module],
    original_models: Mapping[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Iterate the loader once and collect all six fixed-input predictions."""

    _require(set(candidate_models) == set(PARENT_ROLES), "candidate model roles differ")
    _require(set(original_models) == set(PARENT_ROLES), "Original model roles differ")
    for model in (*candidate_models.values(), *original_models.values()):
        model.eval()
        model.mode = "test"
    criterion = nn.BCELoss(reduction="mean")
    families = ("candidate", "current", "original")
    probabilities = {
        family: {role: [] for role in PARENT_ROLES} for family in families
    }
    losses = {family: {role: [] for role in PARENT_ROLES} for family in families}
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    forward_counts = {
        family: {role: 0 for role in PARENT_ROLES} for family in families
    }

    for images, masks, sizes, sample_ids in loader:
        _require(
            int(images.shape[0]) == int(masks.shape[0]) == 1,
            "official evaluator requires batch_size=1",
        )
        _require(
            images.dtype == masks.dtype == torch.float32,
            "official inputs are not FP32",
        )
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height, width = _extract_hw(sizes)
        target = masks[:, :, :height, :width]
        _require(bool(torch.isfinite(target).all()), "official target is non-finite")

        for role in PARENT_ROLES:
            _, auxiliary = candidate_models[role].forward_for_pbdr_v3_training(images)
            outputs = {
                "candidate": torch.sigmoid(auxiliary.routed_logits)[
                    :, :, :height, :width
                ],
                "current": torch.sigmoid(auxiliary.base_logits)[
                    :, :, :height, :width
                ],
                "original": _final_prediction(original_models[role](images))[
                    :, :, :height, :width
                ],
            }
            forward_counts["candidate"][role] += 1
            forward_counts["current"][role] += 1
            forward_counts["original"][role] += 1
            for family, prediction in outputs.items():
                _require(
                    prediction.shape == target.shape,
                    f"{family}/{role} prediction shape differs",
                )
                _require(
                    prediction.dtype == torch.float32
                    and bool(torch.isfinite(prediction).all()),
                    f"{family}/{role} prediction is not finite FP32",
                )
                loss = criterion(prediction.float(), target.float())
                _require(
                    math.isfinite(float(loss.item())),
                    f"{family}/{role} official loss is non-finite",
                )
                probabilities[family][role].append(
                    prediction[0, 0].float().cpu().numpy()
                )
                losses[family][role].append(float(loss.item()))

        targets.append(target[0, 0].float().cpu().numpy())
        _require(
            isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
            "official loader must yield one sample ID",
        )
        identifiers.append(str(sample_ids[0]))

    _require(
        bool(identifiers) and len(identifiers) == len(loader.dataset),
        "official prediction count differs from dataset",
    )
    _require(len(identifiers) == len(set(identifiers)), "duplicate official ID")
    expected_forwards = len(identifiers)
    _require(
        all(
            count == expected_forwards
            for family in forward_counts.values()
            for count in family.values()
        ),
        "one-pass forward count differs",
    )
    return {
        "probabilities": probabilities,
        "losses": losses,
        "targets": targets,
        "identifiers": identifiers,
        "loader_iteration_count": 1,
        "forward_counts": forward_counts,
    }


_METRIC_DIRECTIONS = {
    "matched_target_count": "higher",
    "pd": "higher",
    "fa": "lower",
    "miou": "higher",
    "niou": "higher",
    "matched_tiny_target_count": "higher",
    "tiny_pd": "higher",
    "test_loss": "lower",
    "pixel_precision": "higher",
    "pixel_recall": "higher",
    "pixel_f1": "higher",
    "unmatched_predicted_object_count": "lower",
    "false_objects_per_image": "lower",
}
_COUNT_BINDINGS = ("target_count", "tiny_target_count", "valid_pixel_count")


def _metric_comparison(
    candidate: Mapping[str, Any],
    original: Mapping[str, Any],
) -> dict[str, Any]:
    common = sorted(set(candidate) & set(original))
    deltas: dict[str, Any] = {}
    improved: list[str] = []
    regressed: list[str] = []
    tied: list[str] = []
    for name in common:
        candidate_value = candidate[name]
        original_value = original[name]
        if isinstance(candidate_value, bool) or isinstance(original_value, bool):
            continue
        try:
            candidate_number = _finite_float(candidate_value, f"candidate {name}")
            original_number = _finite_float(original_value, f"Original {name}")
        except (TypeError, ValueError):
            continue
        delta = candidate_number - original_number
        direction = _METRIC_DIRECTIONS.get(name, "binding" if name in _COUNT_BINDINGS else "reported_only")
        status = "tied"
        if direction == "higher":
            status = "improved" if delta > 0 else "regressed" if delta < 0 else "tied"
        elif direction == "lower":
            status = "improved" if delta < 0 else "regressed" if delta > 0 else "tied"
        elif direction == "binding":
            status = "tied" if delta == 0 else "binding_mismatch"
        deltas[name] = {
            "candidate": candidate_number,
            "original": original_number,
            "candidate_minus_original": delta,
            "performance_direction": direction,
            "status": status,
        }
        if status == "improved":
            improved.append(name)
        elif status in ("regressed", "binding_mismatch"):
            regressed.append(name)
        elif status == "tied":
            tied.append(name)
    return {
        "metrics": deltas,
        "improved": improved,
        "regressed": regressed,
        "tied": tied,
    }


def _official_role_decision(
    role: str,
    candidate_fixed: Mapping[str, Any],
    original_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = zero_gate.CertificationMetrics.from_mapping(candidate_fixed)
    original = zero_gate.CertificationMetrics.from_mapping(original_fixed)
    decision = zero_gate.certify(role, original, candidate)
    _require(
        decision.checks["target_count_equal"],
        f"{role} Candidate/Original target counts differ",
    )
    _require(
        decision.checks["tiny_target_count_equal"],
        f"{role} Candidate/Original tiny-target counts differ",
    )
    selected = "candidate" if decision.passed else "original"
    return {
        "role": role,
        "passed": decision.passed,
        "selected": selected,
        "checks": dict(decision.checks),
        "decisive_index": decision.decisive_index,
        "decisive_term": decision.decisive_term,
        "comparison_order": list(zero_gate.ROLE_ORDERS[role]),
        "policy": "strict_role_performance_key_no_positive_margin",
        "minimum_gain": 0.0,
        "exact_tie_retains": "original",
        "epoch_participates_in_cross_model_comparison": False,
        "threshold": FIXED_THRESHOLD,
        "candidate": dict(candidate_fixed),
        "original": dict(original_fixed),
        "metric_comparison": _metric_comparison(candidate_fixed, original_fixed),
    }


def _formal_official_device(dataset_name: str, device_name: str) -> torch.device:
    _require(device_name == "cuda:0", "official-test evaluation requires cuda:0")
    _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one GPU must be visible")
    expected_uuid = trainer.GPU_UUIDS[dataset_name]
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid,
        "official-test CUDA_VISIBLE_DEVICES differs",
    )
    observed = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if observed and not observed.startswith("GPU-"):
        observed = f"GPU-{observed}"
    _require(observed == expected_uuid, "visible GPU UUID differs")
    return torch.device("cuda:0")


def _default_output_paths(
    validated: ValidatedDatasetRuns,
) -> tuple[Path, dict[str, Path], Path]:
    evaluation = validated.formal_root / "evaluation.json"
    deployments = {
        role: validated.runs[role].run_dir / "deployment.json"
        for role in PARENT_ROLES
    }
    claim = validated.formal_root / "official_test_access_claim.json"
    return evaluation, deployments, claim


def _require_fresh_official_outputs(
    evaluation_path: Path,
    deployment_paths: Mapping[str, Path],
) -> None:
    _require(set(deployment_paths) == set(PARENT_ROLES), "deployment roles differ")
    paths = [Path(evaluation_path), *(Path(deployment_paths[role]) for role in PARENT_ROLES)]
    _require(
        len({path.resolve(strict=False) for path in paths}) == len(paths),
        "official output paths must be distinct",
    )
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"official output already exists: {path}")


def _preflight_payload(validated: ValidatedDatasetRuns) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in PARENT_ROLES:
        run = validated.runs[role]
        roles[role] = {
            "trainer_summary": _artifact_binding(run.summary_path),
            "trainer_protocol": _artifact_binding(run.protocol_path),
            "internal_split": _artifact_binding(run.split_path),
            "candidate_checkpoint": {
                **_artifact_binding(run.candidate_path),
                "state_key_count": len(run.candidate_state),
                "state_sha256": run.candidate_state_sha256,
                "epoch": int(run.candidate["epoch"]),
            },
            "current_checkpoint": _checkpoint_binding(run.parent_checkpoint),
            "original_checkpoint": _checkpoint_binding(
                validated.original_checkpoints[role]
            ),
            "protocol_sha256": run.protocol_sha256,
            "internal_current_diagnostic": dict(run.internal_decision),
        }
    return {
        "dataset": validated.dataset_name,
        "roles": roles,
        "shared_split_sha256": validated.shared_split_sha256,
        "shared_runtime_sources_sha256": validated.shared_runtime_sources_sha256,
        "cross_dataset_protocol_document": _artifact_binding(PROTOCOL_DOCUMENT),
        "original_authority_manifest": dict(
            validated.original_authority.get("authority_manifest", {})
        ),
        "official_test_data_accessed": False,
    }


def _claim_official_test_access(
    validated: ValidatedDatasetRuns,
    claim_path: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(claim_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    payload = {
        "schema": OFFICIAL_ACCESS_CLAIM_SCHEMA,
        "status": "claimed_before_dataset_construction",
        "dataset": validated.dataset_name,
        "roles": list(PARENT_ROLES),
        "recipes": {role: "core" for role in PARENT_ROLES},
        "preflight_sha256": models.canonical_sha256(preflight),
        "candidate_checkpoint_sha256": {
            role: validated.runs[role].candidate_sha256 for role in PARENT_ROLES
        },
        "current_checkpoint_sha256": {
            role: validated.runs[role].parent_checkpoint["sha256"]
            for role in PARENT_ROLES
        },
        "original_checkpoint_sha256": {
            role: validated.original_checkpoints[role]["sha256"]
            for role in PARENT_ROLES
        },
        "shared_split_sha256": validated.shared_split_sha256,
        "authorization": "both_role_core_runs_and_all_comparators_integrity_valid",
        "maximum_dataset_official_test_loader_constructions": 1,
        "maximum_dataset_official_test_loader_passes": 1,
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
            "official-test access was already claimed for this dataset"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return _artifact_binding(destination)


@dataclass(slots=True)
class PreparedModels:
    candidates: Mapping[str, nn.Module]
    originals: Mapping[str, nn.Module]
    candidate_metadata: Mapping[str, Mapping[str, Any]]
    original_metadata: Mapping[str, Mapping[str, Any]]


def _prepare_models(
    validated: ValidatedDatasetRuns,
    device: torch.device,
) -> PreparedModels:
    trainer.engine.configure_determinism()
    _require(
        torch.backends.cuda.matmul.allow_tf32 is False
        and torch.backends.cudnn.allow_tf32 is False,
        "official-test TF32 controls differ",
    )
    candidates: dict[str, nn.Module] = {}
    originals: dict[str, nn.Module] = {}
    candidate_metadata: dict[str, Mapping[str, Any]] = {}
    original_metadata: dict[str, Mapping[str, Any]] = {}
    for role in PARENT_ROLES:
        run = validated.runs[role]
        candidate, candidate_meta = models.build_inference_model_from_candidate_state(
            run.candidate_state,
            dataset_name=validated.dataset_name,
            parent_role=role,
        )
        _require(candidate_meta.get("strict_load") is True, "candidate load not strict")
        _require(
            candidate_meta.get("base_bitwise_equal_to_parent") is True,
            "candidate base differs from Current",
        )
        _require(
            candidate_meta.get("inference_state_key_count")
            == models.INFERENCE_STATE_KEY_COUNT,
            "candidate inference state-key count differs",
        )
        expected_inference_state = models.strip_training_only_survival_state(
            run.candidate_state
        )
        installed_candidate_sha = models.tensor_mapping_sha256(candidate.state_dict())
        _require(
            installed_candidate_sha
            == models.tensor_mapping_sha256(expected_inference_state),
            "candidate installed inference state differs",
        )
        candidate_meta = dict(candidate_meta)
        candidate_meta["inference_state_sha256"] = installed_candidate_sha

        original, original_meta = models.build_original_inference_model(
            validated.dataset_name,
            role,
        )
        _require(original_meta.get("strict_load") is True, "Original load not strict")
        _require(
            original_meta.get("state_sha256")
            == validated.original_checkpoints[role]["state_sha256"],
            "Original installed state differs from preflight",
        )
        _assert_fp32_model(candidate, f"Candidate/{role}")
        _assert_fp32_model(original, f"Original/{role}")
        candidates[role] = candidate.to(device)
        originals[role] = original.to(device)
        candidate_metadata[role] = candidate_meta
        original_metadata[role] = original_meta
    return PreparedModels(
        candidates=candidates,
        originals=originals,
        candidate_metadata=candidate_metadata,
        original_metadata=original_metadata,
    )


def _evaluate_official_test(
    validated: ValidatedDatasetRuns,
    prepared: PreparedModels,
    *,
    data_root: Path,
    protocol_manifest: Path,
    device: torch.device,
    workers: int,
    access_claim_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Claim, construct one loader, and make exactly one six-model pass."""

    _require(workers >= 0, "workers must be non-negative")
    _require(
        torch.backends.cuda.matmul.allow_tf32 is False
        and torch.backends.cudnn.allow_tf32 is False,
        "official-test TF32 controls differ before claim",
    )
    first = validated.runs[PARENT_ROLES[0]]
    manifest_path = Path(protocol_manifest).resolve(strict=True)
    data_root_path = Path(data_root).resolve(strict=True)
    frozen_manifest = first.split_manifest["data_protocol_manifest"]
    _require(
        manifest_path == Path(str(frozen_manifest["path"])).resolve(strict=True)
        and models.file_sha256(manifest_path) == frozen_manifest["sha256"],
        "official evaluator data protocol differs from training binding",
    )
    _require(
        str(data_root_path) == first.protocol.get("data_root"),
        "official evaluator data root differs from training binding",
    )

    preflight = _preflight_payload(validated)
    access_claim = _claim_official_test_access(
        validated,
        access_claim_path,
        preflight,
    )

    # No dataset/index object exists before the durable O_EXCL claim above.
    dataset = OfficialTestDataset(data_root_path, validated.dataset_name)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    cache = _collect_six_models_one_pass(
        prepared.candidates,
        prepared.originals,
        loader,
        device,
    )
    identifiers = list(cache["identifiers"])
    _require(
        identifiers == list(dataset.sample_ids),
        "official inference order differs from frozen test index",
    )
    expected_test = data_protocol.EXPECTED_SPLITS[validated.dataset_name]["test"]
    ordered_ids_sha = data_protocol.ordered_ids_sha256(identifiers)
    _require(len(identifiers) == int(expected_test["count"]), "test count differs")
    _require(
        ordered_ids_sha == expected_test["ordered_ids_sha256"],
        "test ordered-ID SHA differs",
    )

    fixed_key = f"{FIXED_THRESHOLD:.2f}"
    metrics: dict[str, dict[str, Mapping[str, Any]]] = {
        family: {} for family in ("candidate", "current", "original")
    }
    for family in metrics:
        for role in PARENT_ROLES:
            point = _metric_points(
                cache["probabilities"][family][role],
                cache["targets"],
                cache["losses"][family][role],
                (FIXED_THRESHOLD,),
            )[fixed_key]
            metrics[family][role] = point

    valid_pixel_count = sum(int(target.size) for target in cache["targets"])
    positive_pixel_count = sum(
        int(np.count_nonzero(target > FIXED_THRESHOLD)) for target in cache["targets"]
    )
    target_counts = {
        int(metrics[family][role]["target_count"])
        for family in metrics
        for role in PARENT_ROLES
    }
    valid_counts = {
        int(metrics[family][role]["valid_pixel_count"])
        for family in metrics
        for role in PARENT_ROLES
    }
    _require(len(target_counts) == 1, "six-model target counts differ")
    _require(valid_counts == {valid_pixel_count}, "six-model valid-pixel counts differ")
    target_count = next(iter(target_counts))
    for role in PARENT_ROLES:
        historical = validated.original_checkpoints[role][
            "fixed_threshold_0_5_metrics"
        ]
        _require(
            int(historical["target_count"]) == target_count
            and int(historical["valid_pixel_count"]) == valid_pixel_count,
            f"Original historical {role} data counts differ",
        )

    decisions = {
        role: _official_role_decision(
            role,
            metrics["candidate"][role],
            metrics["original"][role],
        )
        for role in PARENT_ROLES
    }
    inference_order_newline_sha = hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()
    evaluation = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": validated.dataset_name,
        "seed": trainer.TRAINING_SEED,
        "roles": list(PARENT_ROLES),
        "recipe": "core",
        "official_test_accessed": True,
        "official_test_access_claim": access_claim,
        "official_test_loader_construction_count": 1,
        "official_test_loader_pass_count": cache["loader_iteration_count"],
        "inference_forward_counts": cache["forward_counts"],
        "threshold": {
            "probability_to_mask": FIXED_THRESHOLD,
            "comparison": "probability > threshold",
            "threshold_optimization_performed": False,
        },
        "metrics": {
            family: {
                role: dict(metrics[family][role]) for role in PARENT_ROLES
            }
            for family in metrics
        },
        "candidate_vs_same_role_original": decisions,
        "preflight": preflight,
        "training_artifacts": {
            role: {
                "protocol_sha256": validated.runs[role].protocol_sha256,
                "candidate_checkpoint_sha256": validated.runs[role].candidate_sha256,
                "candidate_training_state_sha256": validated.runs[
                    role
                ].candidate_state_sha256,
                "current_checkpoint": _checkpoint_binding(
                    validated.runs[role].parent_checkpoint
                ),
                "original_checkpoint": _checkpoint_binding(
                    validated.original_checkpoints[role]
                ),
                "internal_candidate_vs_current_diagnostic": dict(
                    validated.runs[role].internal_decision
                ),
            }
            for role in PARENT_ROLES
        },
        "models": {
            "candidate": {
                role: dict(prepared.candidate_metadata[role])
                for role in PARENT_ROLES
            },
            "original": {
                role: dict(prepared.original_metadata[role])
                for role in PARENT_ROLES
            },
            "current": {
                "source": "same_candidate_forward_router_bypass_base_logits",
                "separate_current_approximation": False,
                "base_bitwise_equal_to_same_role_parent": True,
            },
        },
        "original_selection_disclosure": {
            "test_selected": True,
            "selection_is_optimistic": True,
            "historical_metrics_used_for_deployment_comparison": False,
            "original_re_evaluated_under_matched_fp32_precision": True,
            "policy": validated.original_authority.get("selection_policy"),
        },
        "data": {
            "dataset_root": str(data_root_path),
            "protocol_manifest": {
                "path": str(manifest_path),
                "sha256": models.file_sha256(manifest_path),
            },
            "split": "img_idx/test",
            "test_count": len(identifiers),
            "ordered_test_ids": identifiers,
            "img_idx_test_sha256": expected_test["file_sha256"],
            "img_idx_test_ordered_ids_sha256": expected_test[
                "ordered_ids_sha256"
            ],
            "inference_order_newline_sha256": inference_order_newline_sha,
            "target_count": target_count,
            "positive_pixel_count": positive_pixel_count,
            "valid_pixel_count": valid_pixel_count,
            "normalization": data_protocol.get_legacy_normalization(
                validated.dataset_name
            ),
        },
        "metric_protocol": {
            "implementation": "experiments.train_tpd_pilot.ValidationMetrics",
            "connectivity": 8,
            "match_radius": trainer.engine.FORMAL_MATCH_RADIUS,
            "tiny_area": trainer.engine.FORMAL_TINY_AREA,
            "role_orders": {
                role: list(zero_gate.ROLE_ORDERS[role]) for role in PARENT_ROLES
            },
            "minimum_gain": 0.0,
            "exact_tie_retains": "original",
            "epoch_participates_in_cross_model_comparison": False,
        },
        "precision": {
            "mode": "fp32",
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "device": str(device),
            "gpu_uuid": trainer.GPU_UUIDS[validated.dataset_name],
        },
        "runtime_sources": dict(validated.runs[PARENT_ROLES[0]].runtime_sources),
        "runtime_sources_sha256": validated.shared_runtime_sources_sha256,
        "evaluator": _artifact_binding(Path(__file__)),
        "no_fabricated_results": True,
    }
    _require(
        evaluation["precision"]["cuda_matmul_allow_tf32"] is False
        and evaluation["precision"]["cudnn_allow_tf32"] is False,
        "recorded TF32 controls differ",
    )

    deployments: dict[str, dict[str, Any]] = {}
    for role in PARENT_ROLES:
        decision = decisions[role]
        selected = str(decision["selected"])
        candidate_artifact = {
            **_artifact_binding(validated.runs[role].candidate_path),
            "kind": "pbdr_v3_training_checkpoint_strict_inference_conversion",
            "training_state_sha256": validated.runs[role].candidate_state_sha256,
            "inference_state_sha256": prepared.candidate_metadata[role][
                "inference_state_sha256"
            ],
        }
        original_artifact = {
            **_checkpoint_binding(validated.original_checkpoints[role]),
            "kind": "same_role_original_checkpoint",
            "test_selected": True,
            "selection_is_optimistic": True,
        }
        deployments[role] = {
            "schema": DEPLOYMENT_SCHEMA,
            "status": "complete",
            "dataset": validated.dataset_name,
            "parent_role": role,
            "recipe": "core",
            "selected": selected,
            "selection_reason": "same_role_original_zero_margin_role_order",
            "selected_threshold": FIXED_THRESHOLD,
            "selected_artifact": (
                candidate_artifact if selected == "candidate" else original_artifact
            ),
            "candidate_artifact": candidate_artifact,
            "original_artifact": original_artifact,
            "current_diagnostic_artifact": _checkpoint_binding(
                validated.runs[role].parent_checkpoint
            ),
            "candidate_vs_same_role_original": dict(decision),
            "internal_candidate_vs_current_diagnostic": dict(
                validated.runs[role].internal_decision
            ),
            "official_test_accessed": True,
            "official_test_access_claim": access_claim,
            "original_selection_disclosure": {
                "test_selected": True,
                "selection_is_optimistic": True,
                "re_evaluated_under_matched_fp32_precision": True,
            },
        }

    del loader, dataset, cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return evaluation, deployments


def _make_publication_bundle(
    dataset_name: str,
    evaluation: Mapping[str, Any],
    deployment_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(set(deployment_payloads) == set(PARENT_ROLES), "bundle roles differ")
    ready_evaluation = _json_ready(evaluation)
    ready_deployments = {
        role: _json_ready(deployment_payloads[role]) for role in PARENT_ROLES
    }
    return {
        "schema": PUBLICATION_BUNDLE_SCHEMA,
        "status": "committed_official_fact",
        "dataset": dataset_name,
        "evaluation_canonical_sha256": models.canonical_sha256(ready_evaluation),
        "deployment_template_canonical_sha256": {
            role: models.canonical_sha256(ready_deployments[role])
            for role in PARENT_ROLES
        },
        "evaluation": ready_evaluation,
        "deployment_templates": ready_deployments,
        "official_test_reaccess_required_for_materialization": False,
    }


def _validate_publication_bundle(
    validated: ValidatedDatasetRuns,
    claim_path: Path,
    value: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _require(isinstance(value, Mapping), "publication bundle must be a mapping")
    _require(value.get("schema") == PUBLICATION_BUNDLE_SCHEMA, "bundle schema differs")
    _require(
        value.get("status") == "committed_official_fact"
        and value.get("dataset") == validated.dataset_name
        and value.get("official_test_reaccess_required_for_materialization") is False,
        "publication bundle identity/status differs",
    )
    evaluation = value.get("evaluation")
    templates = value.get("deployment_templates")
    _require(isinstance(evaluation, Mapping), "bundle evaluation is malformed")
    _require(
        isinstance(templates, Mapping) and set(templates) == set(PARENT_ROLES),
        "bundle deployment templates differ",
    )
    evaluation_ready = dict(evaluation)
    template_ready = {
        role: dict(templates[role])
        for role in PARENT_ROLES
        if isinstance(templates[role], Mapping)
    }
    _require(len(template_ready) == len(PARENT_ROLES), "bundle template malformed")
    _require(
        value.get("evaluation_canonical_sha256")
        == models.canonical_sha256(_json_ready(evaluation_ready)),
        "bundle evaluation hash differs",
    )
    declared_template_hashes = value.get("deployment_template_canonical_sha256")
    _require(
        isinstance(declared_template_hashes, Mapping)
        and set(declared_template_hashes) == set(PARENT_ROLES),
        "bundle deployment hashes differ",
    )
    for role in PARENT_ROLES:
        _require(
            declared_template_hashes.get(role)
            == models.canonical_sha256(_json_ready(template_ready[role])),
            f"bundle {role} deployment hash differs",
        )

    claim_binding = _artifact_binding(claim_path)
    _require(
        evaluation_ready.get("schema") == SCHEMA
        and evaluation_ready.get("status") == "complete"
        and evaluation_ready.get("dataset") == validated.dataset_name
        and evaluation_ready.get("roles") == list(PARENT_ROLES)
        and evaluation_ready.get("official_test_accessed") is True,
        "bundle evaluation identity/status differs",
    )
    _require(
        _canonical_equal(
            evaluation_ready.get("official_test_access_claim"), claim_binding
        ),
        "bundle evaluation claim binding differs",
    )
    _require(
        _canonical_equal(evaluation_ready.get("preflight"), _preflight_payload(validated)),
        "bundle evaluation preflight differs",
    )
    _require(
        evaluation_ready.get("runtime_sources_sha256")
        == validated.shared_runtime_sources_sha256
        and _canonical_equal(
            evaluation_ready.get("runtime_sources"),
            validated.runs[PARENT_ROLES[0]].runtime_sources,
        ),
        "bundle runtime-source binding differs",
    )
    metrics = evaluation_ready.get("metrics")
    decisions = evaluation_ready.get("candidate_vs_same_role_original")
    _require(
        isinstance(metrics, Mapping)
        and isinstance(metrics.get("candidate"), Mapping)
        and isinstance(metrics.get("original"), Mapping)
        and isinstance(decisions, Mapping),
        "bundle metrics/decisions are malformed",
    )
    for role in PARENT_ROLES:
        expected_decision = _official_role_decision(
            role,
            metrics["candidate"][role],
            metrics["original"][role],
        )
        _require(
            _canonical_equal(decisions.get(role), expected_decision),
            f"bundle {role} official decision differs",
        )
        template = template_ready[role]
        _require(
            template.get("schema") == DEPLOYMENT_SCHEMA
            and template.get("status") == "complete"
            and template.get("dataset") == validated.dataset_name
            and template.get("parent_role") == role
            and template.get("official_test_accessed") is True,
            f"bundle {role} deployment identity/status differs",
        )
        _require(
            _canonical_equal(template.get("official_test_access_claim"), claim_binding)
            and _canonical_equal(
                template.get("candidate_vs_same_role_original"), expected_decision
            ),
            f"bundle {role} deployment decision/claim differs",
        )
    return evaluation_ready, template_ready


def _materialize_publication_views(
    evaluation_path: Path,
    deployment_paths: Mapping[str, Path],
    evaluation: Mapping[str, Any],
    deployment_templates: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, Path]]:
    _require(set(deployment_paths) == set(PARENT_ROLES), "deployment roles differ")
    if evaluation_path.exists() or evaluation_path.is_symlink():
        observed_evaluation = _json_object(
            evaluation_path, label="materialized official evaluation"
        )
        _require(
            _canonical_equal(observed_evaluation, evaluation),
            "materialized evaluation differs from committed bundle",
        )
    else:
        _atomic_write_json(evaluation_path, evaluation, overwrite=False)
    evaluation_binding = _artifact_binding(evaluation_path)
    published: dict[str, Path] = {}
    for role in PARENT_ROLES:
        payload = dict(deployment_templates[role])
        payload["evaluation"] = evaluation_binding
        destination = Path(deployment_paths[role])
        if destination.exists() or destination.is_symlink():
            observed = _json_object(
                destination, label=f"materialized {role} deployment"
            )
            _require(
                _canonical_equal(observed, payload),
                f"materialized {role} deployment differs from committed bundle",
            )
        else:
            _atomic_write_json(destination, payload, overwrite=False)
        published[role] = destination.resolve(strict=True)
    return evaluation_path.resolve(strict=True), published


def run(
    *,
    dataset_name: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_manifest: Path = DEFAULT_PROTOCOL_MANIFEST,
    run_directories: Mapping[str, Path] | None = None,
    device_name: str = "cuda:0",
    workers: int = 0,
    evaluation_output: Path | None = None,
    deployment_outputs: Mapping[str, Path] | None = None,
    access_claim_output: Path | None = None,
) -> tuple[Path, dict[str, Path]]:
    if workers < 0:
        raise ValueError("workers must be non-negative")
    _require(dataset_name in DATASETS, "unsupported dataset")
    _require(
        Path(results_root).resolve() == DEFAULT_RESULTS_ROOT.resolve(),
        "formal evaluator results-root authority cannot be overridden",
    )
    _require(run_directories is None, "formal evaluator run directories cannot be overridden")
    _require(
        evaluation_output is None
        and deployment_outputs is None
        and access_claim_output is None,
        "formal evaluator claim/publication paths cannot be overridden",
    )
    trainer.validate_frozen_data_binding(data_root, protocol_manifest)
    formal_root = Path(results_root).resolve() / "runs" / dataset_name / "formal"
    formal_root.mkdir(parents=True, exist_ok=True)
    lock_path = formal_root / "evaluation.lock"
    if lock_path.is_symlink():
        raise FileExistsError(f"refusing evaluation lock symlink: {lock_path}")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another evaluator holds this dataset lock") from error
        try:
            validated = validate_dataset_runs(
                dataset_name,
                results_root=results_root,
                run_directories=run_directories,
            )
            default_evaluation, default_deployments, default_claim = (
                _default_output_paths(validated)
            )
            evaluation_path = default_evaluation
            deployments = dict(default_deployments)
            claim_path = default_claim
            bundle_path = validated.formal_root / "publication_bundle.json"
            _require(
                claim_path.resolve(strict=False)
                not in {
                    evaluation_path.resolve(strict=False),
                    bundle_path.resolve(strict=False),
                    *(
                        deployments[role].resolve(strict=False)
                        for role in PARENT_ROLES
                    ),
                },
                "official-test claim path must be distinct from outputs",
            )
            existing_state = any(
                path.exists() or path.is_symlink()
                for path in (
                    claim_path,
                    bundle_path,
                    evaluation_path,
                    *(deployments[role] for role in PARENT_ROLES),
                )
            )
            if existing_state:
                _require(
                    claim_path.is_file()
                    and not claim_path.is_symlink()
                    and bundle_path.is_file()
                    and not bundle_path.is_symlink(),
                    "official state exists without a recoverable committed bundle",
                )
                bundle = _json_object(
                    bundle_path, label="official publication bundle"
                )
                committed_evaluation, committed_templates = (
                    _validate_publication_bundle(validated, claim_path, bundle)
                )
                return _materialize_publication_views(
                    evaluation_path,
                    deployments,
                    committed_evaluation,
                    committed_templates,
                )

            _require_fresh_official_outputs(evaluation_path, deployments)
            if claim_path.exists() or claim_path.is_symlink():
                raise FileExistsError(f"official claim already exists: {claim_path}")
            if bundle_path.exists() or bundle_path.is_symlink():
                raise FileExistsError(f"publication bundle already exists: {bundle_path}")
            device = _formal_official_device(dataset_name, device_name)
            prepared = _prepare_models(validated, device)
            evaluation, deployment_payloads = _evaluate_official_test(
                validated,
                prepared,
                data_root=data_root,
                protocol_manifest=protocol_manifest,
                device=device,
                workers=workers,
                access_claim_path=claim_path,
            )
            bundle = _make_publication_bundle(
                dataset_name, evaluation, deployment_payloads
            )
            _validate_publication_bundle(validated, claim_path, bundle)
            _atomic_write_json(bundle_path, bundle, overwrite=False)
            return _materialize_publication_views(
                evaluation_path,
                deployments,
                evaluation,
                deployment_payloads,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--best-miou-deployment-output", type=Path)
    parser.add_argument("--best-pd-deployment-output", type=Path)
    parser.add_argument("--access-claim-output", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    deployment_outputs = None
    if args.best_miou_deployment_output or args.best_pd_deployment_output:
        if not (args.best_miou_deployment_output and args.best_pd_deployment_output):
            raise ValueError("both deployment output overrides are required")
        deployment_outputs = {
            "best_miou": args.best_miou_deployment_output,
            "best_pd": args.best_pd_deployment_output,
        }
    evaluation, deployments = run(
        dataset_name=args.dataset,
        results_root=args.results_root,
        data_root=args.data_root,
        protocol_manifest=args.protocol_manifest,
        device_name=args.device,
        workers=args.workers,
        evaluation_output=args.evaluation_output,
        deployment_outputs=deployment_outputs,
        access_claim_output=args.access_claim_output,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": args.dataset,
                "evaluation": str(evaluation),
                "evaluation_sha256": models.file_sha256(evaluation),
                "deployments": {
                    role: {
                        "path": str(path),
                        "sha256": models.file_sha256(path),
                        "selected": _json_object(path, label=f"{role} deployment")[
                            "selected"
                        ],
                    }
                    for role, path in deployments.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


__all__ = [
    "DATASETS",
    "DEPLOYMENT_SCHEMA",
    "FIXED_THRESHOLD",
    "OFFICIAL_ACCESS_CLAIM_SCHEMA",
    "PARENT_ROLES",
    "PBDRV3EvaluationProtocolError",
    "PreparedModels",
    "SCHEMA",
    "ValidatedDatasetRuns",
    "ValidatedRun",
    "formal_run_directories",
    "main",
    "parse_args",
    "run",
    "validate_completed_run",
    "validate_dataset_runs",
]


if __name__ == "__main__":
    main()
