#!/usr/bin/env python3
"""Frozen-Current PBDR-V3 registry for NUDT-SIRST and IRSTD-1K.

This module is intentionally separate from the completed NUAA registry.  A
caller must name both the dataset and checkpoint role.  Before a graph is
constructed, the registry verifies the frozen manifest, completed-run summary,
canonical protocol hash, every historical runtime source, checkpoint bytes,
payload contract, and exact tensor-state hash.

The registry never imports a dataset loader, opens an image/mask/index file, or
starts an optimizer/training loop.  It only prepares an audited Stage-1 graph
or converts a supplied candidate training state to the inference graph.
"""

from __future__ import annotations

import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import three_dataset_pbdr_v3_models_seed42_v1 as core  # noqa: E402
from experiments import four_dataset_models_seed42_v1 as original_models  # noqa: E402
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3 import (  # noqa: E402
    PBDR_V3_INTEGRATION_VERSION,
    PBDR_V3_STATE_KEYS,
    PBDR_V3_STATE_PREFIX,
    build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model,
    build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (  # noqa: E402
    SURVIVAL_STATE_KEYS,
)


SCHEMA = "sctransnet_two_dataset_pbdr_v3_models_seed42_v1/v1"
CURRENT_MANIFEST_SCHEMA = (
    "sctransnet_two_dataset_pbdr_v3_current_manifest_seed42_v1/v1"
)
HISTORICAL_RUN_SCHEMA = "sctransnet_three_dataset_tss_off_seed42_v1/v1"
DATASETS = ("NUDT-SIRST", "IRSTD-1K")
PARENT_ROLES = ("best_miou", "best_pd")
TRAINING_SEED = 42
CURRENT_STATE_KEY_COUNT = core.CURRENT_STATE_KEY_COUNT
TRAINING_STATE_KEY_COUNT = core.TRAINING_STATE_KEY_COUNT
INFERENCE_STATE_KEY_COUNT = core.INFERENCE_STATE_KEY_COUNT
CURRENT_MANIFEST_RELATIVE_PATH = (
    "experiments/two_dataset_pbdr_v3_current_manifest_seed42_v1.json"
)
CURRENT_MANIFEST_PATH = REPO_ROOT / CURRENT_MANIFEST_RELATIVE_PATH
CURRENT_MANIFEST_SHA256 = (
    "ca3867bb79a38dd15112c5edf4a34fe57e45f62569e1e89dcb81d15da82baf19"
)
CURRENT_RUN_ROOT = (
    REPO_ROOT / "results/three_dataset_tss_off_seed42_v1/runs"
)
ORIGINAL_MANIFEST_SCHEMA = (
    "sctransnet_two_dataset_pbdr_v3_original_manifest_seed42_v1/v1"
)
ORIGINAL_MANIFEST_RELATIVE_PATH = (
    "experiments/two_dataset_pbdr_v3_original_manifest_seed42_v1.json"
)
ORIGINAL_MANIFEST_PATH = REPO_ROOT / ORIGINAL_MANIFEST_RELATIVE_PATH
ORIGINAL_MANIFEST_SHA256 = (
    "45ececf78a1eb31a0551f67f61fb7ff8813383e1bcf6fd8aef8e91f8ceb9e4d5"
)
ORIGINAL_AUTHORITY_MANIFEST_RELATIVE_PATH = (
    "results/four_dataset_seed42_v1/selected_checkpoints/checkpoint_manifest.json"
)
ORIGINAL_AUTHORITY_MANIFEST_SHA256 = (
    "f286c2f07113be079a2f447b3a2a4e868c81df58ac06cc4acda2de2210249799"
)
ORIGINAL_RUN_SCHEMA = "sctransnet_four_dataset_seed42_exact_v1"
ORIGINAL_STATE_KEY_COUNT = original_models.ORIGINAL_STATE_KEY_COUNT

# Only files executed by this registry are listed here.  The historical
# Current source boundary is separately recovered from each signed protocol
# and verified byte-for-byte by ``audit_current_run``.
RUNTIME_DEPENDENCY_RELATIVE_PATHS = (
    "experiments/two_dataset_pbdr_v3_models_seed42_v1.py",
    CURRENT_MANIFEST_RELATIVE_PATH,
    ORIGINAL_MANIFEST_RELATIVE_PATH,
    "experiments/three_dataset_pbdr_v3_models_seed42_v1.py",
    "experiments/four_dataset_models_seed42_v1.py",
    "model/Config.py",
    "model/SCTransNet.py",
    "model/tpd_conservative_residual_calibrator_v3.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py",
    "model/tpd_frequency_gate_v2_croa.py",
    "model/tpd_query_frequency_bridge.py",
    "model/tpd_survival.py",
)

EXPECTED_CHECKPOINT_PAYLOAD_KEYS = frozenset(
    {
        "checkpoint_role",
        "dataset",
        "epoch",
        "method",
        "model_metadata",
        "protocol_sha256",
        "recipe",
        "requested_tss_weight",
        "schema",
        "seed",
        "selection_is_optimistic",
        "selection_source",
        "state_dict",
        "test_metrics",
        "test_selected",
        "tss_enabled",
    }
)
EXPECTED_ORIGINAL_CHECKPOINT_PAYLOAD_KEYS = frozenset(
    {
        "checkpoint_role",
        "dataset",
        "epoch",
        "method",
        "model_metadata",
        "protocol_sha256",
        "schema",
        "seed",
        "selection_is_optimistic",
        "selection_source",
        "state_dict",
        "test_metrics",
        "test_selected",
    }
)


class CrossDatasetPBDRV3ModelProtocolError(ValueError):
    """A frozen Current artifact or PBDR-V3 graph violated its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossDatasetPBDRV3ModelProtocolError(message)


def file_sha256(path: Path) -> str:
    return core.file_sha256(path)


def canonical_sha256(value: Any) -> str:
    return core.canonical_sha256(value)


def tensor_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return core.tensor_mapping_sha256(state)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _require_dataset(dataset_name: str) -> str:
    _require(
        type(dataset_name) is str and dataset_name in DATASETS,
        f"dataset_name must be one of {DATASETS}",
    )
    return dataset_name


def _require_role(parent_role: str) -> str:
    _require(
        type(parent_role) is str and parent_role in PARENT_ROLES,
        f"parent_role must be one of {PARENT_ROLES}",
    )
    return parent_role


def _require_seed(seed: int) -> int:
    _require(type(seed) is int and seed == TRAINING_SEED, "formal seed is 42")
    return seed


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(
        all(type(key) is str for key in value),
        f"{label} keys must be strings",
    )
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    actual = set(value)
    _require(
        actual == set(expected),
        f"{label} keys differ: missing={sorted(set(expected) - actual)}, "
        f"unexpected={sorted(actual - set(expected))}",
    )


def _repo_relative_file(relative_path: str, label: str) -> Path:
    _require(
        type(relative_path) is str and relative_path,
        f"{label} relative path must be a non-empty string",
    )
    relative = Path(relative_path)
    _require(not relative.is_absolute(), f"{label} path must be repo-relative")
    raw = REPO_ROOT / relative
    cursor = REPO_ROOT
    for component in relative.parts:
        _require(component not in ("", ".", ".."), f"{label} path is unsafe")
        cursor = cursor / component
        _require(not cursor.is_symlink(), f"{label} path traverses a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise CrossDatasetPBDRV3ModelProtocolError(
            f"{label} file is missing: {raw}"
        ) from error
    _require(
        resolved.is_relative_to(REPO_ROOT.resolve(strict=True)),
        f"{label} escapes repository root",
    )
    _require(resolved.is_file(), f"{label} is not a regular file")
    return resolved


def _absolute_repo_file(path_value: Any, label: str) -> Path:
    _require(type(path_value) is str and path_value, f"{label} path is invalid")
    supplied = Path(path_value)
    _require(supplied.is_absolute(), f"{label} path must be absolute")
    try:
        relative = supplied.relative_to(REPO_ROOT)
    except ValueError as error:
        raise CrossDatasetPBDRV3ModelProtocolError(
            f"{label} path is outside repository"
        ) from error
    resolved = _repo_relative_file(str(relative), label)
    _require(resolved == supplied, f"{label} path is not canonical")
    return resolved


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrossDatasetPBDRV3ModelProtocolError(
            f"could not read {label}: {path}"
        ) from error
    return dict(_require_mapping(value, label))


def _verify_file_record(record: Any, label: str) -> Path:
    item = _require_mapping(record, label)
    _require_exact_keys(item, {"relative_path", "sha256", "bytes"}, label)
    path = _repo_relative_file(item["relative_path"], label)
    _require(path.stat().st_size == item["bytes"], f"{label} byte count differs")
    _require(file_sha256(path) == item["sha256"], f"{label} SHA-256 differs")
    return path


def load_frozen_current_manifest() -> dict[str, Any]:
    """Load the code-pinned manifest; never infer checkpoint paths."""

    path = _repo_relative_file(
        CURRENT_MANIFEST_RELATIVE_PATH, "frozen Current manifest"
    )
    _require(
        file_sha256(path) == CURRENT_MANIFEST_SHA256,
        "frozen Current manifest SHA-256 differs",
    )
    manifest = _load_json_object(path, "frozen Current manifest")
    _require(manifest.get("schema") == CURRENT_MANIFEST_SCHEMA, "manifest schema differs")
    _require(manifest.get("authority") == "completed_three_dataset_tss_off_seed42_v1_artifacts", "manifest authority differs")
    _require(tuple(manifest.get("allowed_datasets", ())) == DATASETS, "manifest dataset order differs")
    _require(tuple(manifest.get("allowed_checkpoint_roles", ())) == PARENT_ROLES, "manifest role order differs")
    _require(manifest.get("historical_run_schema") == HISTORICAL_RUN_SCHEMA, "historical run schema differs")
    _require(manifest.get("root_relative") == "results/three_dataset_tss_off_seed42_v1/runs", "manifest root differs")
    _require(manifest.get("training_seed") == TRAINING_SEED, "manifest seed differs")
    _require(manifest.get("current_state_key_count") == CURRENT_STATE_KEY_COUNT, "manifest Current key count differs")
    _require(manifest.get("historical_runtime_source_count") == 35, "historical source count differs")
    datasets = _require_mapping(manifest.get("datasets"), "manifest datasets")
    _require(set(datasets) == set(DATASETS), "manifest dataset records differ")
    guards = _require_mapping(manifest.get("scope_guards"), "manifest scope guards")
    _require(
        guards
        == {
            "nuaa_registry_is_unchanged": True,
            "official_test_data_must_not_be_opened": True,
            "training_must_not_be_started_by_registry": True,
        },
        "manifest scope guards differ",
    )
    return manifest


def load_frozen_original_manifest() -> dict[str, Any]:
    """Load the code-pinned projection of the Original authority manifest."""

    path = _repo_relative_file(
        ORIGINAL_MANIFEST_RELATIVE_PATH, "frozen Original manifest"
    )
    _require(
        file_sha256(path) == ORIGINAL_MANIFEST_SHA256,
        "frozen Original manifest SHA-256 differs",
    )
    manifest = _load_json_object(path, "frozen Original manifest")
    _require(
        manifest.get("schema") == ORIGINAL_MANIFEST_SCHEMA,
        "Original manifest schema differs",
    )
    _require(
        manifest.get("authority")
        == "four_dataset_seed42_v1_selected_checkpoint_manifest",
        "Original manifest authority differs",
    )
    authority = _require_mapping(
        manifest.get("authority_manifest"), "Original authority manifest"
    )
    _require(
        authority
        == {
            "relative_path": ORIGINAL_AUTHORITY_MANIFEST_RELATIVE_PATH,
            "sha256": ORIGINAL_AUTHORITY_MANIFEST_SHA256,
            "bytes": 73520,
        },
        "Original authority manifest record differs",
    )
    _require(
        tuple(manifest.get("allowed_datasets", ())) == DATASETS,
        "Original manifest dataset order differs",
    )
    _require(
        tuple(manifest.get("allowed_checkpoint_roles", ())) == PARENT_ROLES,
        "Original manifest role order differs",
    )
    _require(
        manifest.get("training_seed") == TRAINING_SEED,
        "Original manifest seed differs",
    )
    _require(
        manifest.get("original_state_key_count") == ORIGINAL_STATE_KEY_COUNT,
        "Original manifest state-key count differs",
    )
    datasets = _require_mapping(
        manifest.get("datasets"), "Original manifest datasets"
    )
    _require(
        set(datasets) == set(DATASETS),
        "Original manifest dataset records differ",
    )
    policy = _require_mapping(
        manifest.get("selection_policy"), "Original selection policy"
    )
    _require(
        policy
        == {
            "threshold": 0.5,
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
            "test_selected": True,
            "selection_is_optimistic": True,
        },
        "Original selection policy differs",
    )
    _require(
        manifest.get("scope_guards")
        == {
            "official_test_data_must_not_be_opened": True,
            "evaluation_must_not_be_started_by_registry": True,
        },
        "Original manifest scope guards differ",
    )
    return manifest


def audit_original_authority(dataset_name: str) -> dict[str, Any]:
    """Audit Original records and files without reading an official split."""

    dataset = _require_dataset(dataset_name)
    frozen_manifest = load_frozen_original_manifest()
    authority_record = _require_mapping(
        frozen_manifest["authority_manifest"], "Original authority record"
    )
    authority_path = _repo_relative_file(
        authority_record["relative_path"], "Original authority manifest"
    )
    _require(
        authority_path.stat().st_size == authority_record["bytes"],
        "Original authority manifest byte count differs",
    )
    _require(
        file_sha256(authority_path) == authority_record["sha256"],
        "Original authority manifest SHA-256 differs",
    )
    authority = _load_json_object(authority_path, "Original authority manifest")
    _require(
        authority.get("schema")
        == "sctransnet_four_dataset_seed42_checkpoint_manifest_v1",
        "Original authority schema differs",
    )
    _require(authority.get("status") == "complete", "Original authority is incomplete")
    _require(authority.get("seed") == TRAINING_SEED, "Original authority seed differs")
    _require(authority.get("record_count") == 8, "Original authority record count differs")
    _require(
        authority.get("experiment") == "four_training_regimes_original_vs_final",
        "Original authority experiment differs",
    )
    _require(authority.get("no_fabricated_results") is True, "Original authority integrity disclosure differs")
    source_locks = _require_mapping(
        authority.get("source_sha256"), "Original authority source locks"
    )
    source_audit: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in sorted(source_locks.items()):
        path = _repo_relative_file(relative, f"Original authority source {relative}")
        actual_sha = file_sha256(path)
        _require(actual_sha == expected_sha, f"Original authority source {relative} SHA-256 differs")
        source_audit[relative] = {
            "path": str(path),
            "sha256": actual_sha,
            "bytes": path.stat().st_size,
        }

    records = authority.get("records")
    _require(isinstance(records, list), "Original authority records must be a list")
    matching = [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("dataset") == dataset
        and item.get("method") == "original"
    ]
    _require(len(matching) == 1, f"Original authority must contain one {dataset} record")
    record = _require_mapping(matching[0], f"Original authority {dataset}")
    _require(record.get("audit_passed") is True, f"Original {dataset} audit did not pass")
    _require(record.get("method_label") == "Original", f"Original {dataset} label differs")
    _require(record.get("seed") == TRAINING_SEED, f"Original {dataset} seed differs")

    global_disclosure = _require_mapping(
        authority.get("selection_disclosure"), "Original global selection disclosure"
    )
    record_disclosure = _require_mapping(
        record.get("selection_disclosure"), f"Original {dataset} selection disclosure"
    )
    _require(record_disclosure == global_disclosure, f"Original {dataset} selection disclosure differs")
    frozen_policy = _require_mapping(
        frozen_manifest["selection_policy"], "frozen Original policy"
    )
    for name in ("best_miou_order", "best_pd_order"):
        _require(record_disclosure.get(name) == frozen_policy[name], f"Original {dataset} {name} differs")
    _require(record_disclosure.get("selection_threshold") == 0.5, f"Original {dataset} threshold differs")
    _require(record_disclosure.get("test_selected") is True, f"Original {dataset} test disclosure differs")
    _require(record_disclosure.get("selection_is_optimistic") is True, f"Original {dataset} optimism disclosure differs")

    frozen_datasets = _require_mapping(
        frozen_manifest["datasets"], "frozen Original datasets"
    )
    frozen_dataset = _require_mapping(
        frozen_datasets[dataset], f"frozen Original {dataset}"
    )
    protocol_record = _require_mapping(
        frozen_dataset["protocol"], f"frozen Original {dataset} protocol"
    )
    protocol_path = _repo_relative_file(
        protocol_record["relative_path"], f"Original {dataset} protocol"
    )
    _require(file_sha256(protocol_path) == protocol_record["sha256"], f"Original {dataset} protocol SHA-256 differs")
    authority_protocol = _require_mapping(
        record.get("protocol_audit"), f"Original {dataset} protocol audit"
    )
    _require(authority_protocol.get("fresh_scratch") is True, f"Original {dataset} was not fresh scratch")
    _require(authority_protocol.get("path") == str(protocol_path), f"Original {dataset} protocol path differs")
    _require(authority_protocol.get("sha256") == protocol_record["sha256"], f"Original {dataset} protocol authority hash differs")
    verified_arguments = _require_mapping(
        authority_protocol.get("verified_arguments"),
        f"Original {dataset} protocol arguments",
    )
    _require(
        verified_arguments
        == {
            "begin_test": 10,
            "dataset": dataset,
            "epochs": 1000,
            "eval_every": 10,
            "match_radius": 3.0,
            "method": "original",
            "seed": 42,
            "threshold": 0.5,
            "tiny_area": 9,
        },
        f"Original {dataset} verified protocol arguments differ",
    )
    protocol = _load_json_object(protocol_path, f"Original {dataset} protocol")
    protocol_sha = _protocol_canonical_sha256(protocol)
    _require(protocol.get("dataset") == dataset, f"Original {dataset} protocol dataset differs")
    _require(protocol.get("method") == "original", f"Original {dataset} protocol method differs")
    _require(protocol.get("training_seed") == TRAINING_SEED, f"Original {dataset} protocol seed differs")

    authority_checkpoints = _require_mapping(
        record.get("checkpoints"), f"Original authority {dataset} checkpoints"
    )
    frozen_checkpoints = _require_mapping(
        frozen_dataset["checkpoints"], f"frozen Original {dataset} checkpoints"
    )
    _require(set(authority_checkpoints) == set(PARENT_ROLES), f"Original {dataset} authority roles differ")
    _require(set(frozen_checkpoints) == set(PARENT_ROLES), f"Original {dataset} frozen roles differ")
    checkpoints: dict[str, dict[str, Any]] = {}
    for role in PARENT_ROLES:
        expected = _require_mapping(
            frozen_checkpoints[role], f"frozen Original {dataset}/{role}"
        )
        selected = _require_mapping(
            authority_checkpoints[role], f"Original authority {dataset}/{role}"
        )
        frozen_path = _repo_relative_file(
            expected["frozen_relative_path"], f"Original {dataset}/{role} frozen checkpoint"
        )
        source_path = _repo_relative_file(
            expected["source_relative_path"], f"Original {dataset}/{role} source checkpoint"
        )
        _require(selected.get("checkpoint_role") == role, f"Original {dataset}/{role} role differs")
        _require(selected.get("epoch") == expected["epoch"], f"Original {dataset}/{role} epoch differs")
        _require(selected.get("frozen_path") == str(frozen_path), f"Original {dataset}/{role} frozen path differs")
        _require(selected.get("source_path") == str(source_path), f"Original {dataset}/{role} source path differs")
        _require(selected.get("sha256") == expected["sha256"], f"Original {dataset}/{role} authority SHA-256 differs")
        _require(selected.get("test_selected") is True, f"Original {dataset}/{role} test disclosure differs")
        _require(selected.get("selection_is_optimistic") is True, f"Original {dataset}/{role} optimism disclosure differs")
        fixed = _require_mapping(
            selected.get("fixed_threshold_0_5_metrics"),
            f"Original {dataset}/{role} fixed metrics",
        )
        _require(fixed.get("threshold") == 0.5, f"Original {dataset}/{role} metric threshold differs")
        for label, path in (("frozen", frozen_path), ("source", source_path)):
            _require(path.stat().st_size == expected["bytes"], f"Original {dataset}/{role} {label} bytes differ")
            _require(file_sha256(path) == expected["sha256"], f"Original {dataset}/{role} {label} SHA-256 differs")
        checkpoints[role] = {
            "frozen_path": str(frozen_path),
            "source_path": str(source_path),
            "sha256": expected["sha256"],
            "bytes": expected["bytes"],
            "epoch": expected["epoch"],
            "state_key_count": expected["state_key_count"],
            "state_sha256": expected["state_sha256"],
            "fixed_threshold_0_5_metrics": dict(fixed),
        }
    return {
        "dataset": dataset,
        "authority_manifest": {
            "path": str(authority_path),
            "sha256": authority_record["sha256"],
        },
        "frozen_projection_manifest": {
            "path": str(ORIGINAL_MANIFEST_PATH),
            "sha256": ORIGINAL_MANIFEST_SHA256,
        },
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": protocol_record["sha256"],
            "canonical_sha256": protocol_sha,
        },
        "source_locks": source_audit,
        "checkpoints": checkpoints,
        "selection_policy": dict(frozen_policy),
        "official_test_data_accessed": False,
        "dataset_loader_imported": False,
        "evaluation_started": False,
    }


def runtime_source_paths() -> dict[str, Path]:
    return {
        relative: _repo_relative_file(relative, f"runtime source {relative}")
        for relative in RUNTIME_DEPENDENCY_RELATIVE_PATHS
    }


def runtime_source_records() -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for relative, path in sorted(runtime_source_paths().items())
    }


def _audit_historical_runtime_sources(
    protocol: Mapping[str, Any], *, expected_count: int
) -> dict[str, dict[str, Any]]:
    sources = _require_mapping(
        protocol.get("runtime_sources"), "historical runtime sources"
    )
    _require(len(sources) == expected_count, "historical source count differs")
    audited: dict[str, dict[str, Any]] = {}
    for source_name, raw in sorted(sources.items()):
        entry = _require_mapping(raw, f"historical source {source_name}")
        _require_exact_keys(entry, {"path", "sha256"}, f"historical source {source_name}")
        path = _absolute_repo_file(entry["path"], f"historical source {source_name}")
        actual = file_sha256(path)
        _require(actual == entry["sha256"], f"historical source {source_name} SHA-256 differs")
        if source_name.startswith("architecture::"):
            expected_relative = source_name.removeprefix("architecture::")
            _require(path == REPO_ROOT / expected_relative, f"historical architecture source {source_name} path differs")
        audited[source_name] = {
            "path": str(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return audited


def _protocol_canonical_sha256(protocol: Mapping[str, Any]) -> str:
    unsigned = dict(protocol)
    declared = unsigned.pop("protocol_sha256", None)
    _require(type(declared) is str and len(declared) == 64, "protocol declared SHA-256 is invalid")
    computed = canonical_sha256(unsigned)
    _require(computed == declared, "protocol canonical SHA-256 differs")
    return computed


def audit_current_run(dataset_name: str) -> dict[str, Any]:
    """Audit signed result metadata and historical sources without test data."""

    dataset = _require_dataset(dataset_name)
    manifest = load_frozen_current_manifest()
    dataset_records = _require_mapping(manifest["datasets"], "manifest datasets")
    frozen = _require_mapping(dataset_records[dataset], f"manifest {dataset}")
    _require_exact_keys(frozen, {"summary", "protocol", "checkpoints"}, f"manifest {dataset}")

    summary_path = _verify_file_record(frozen["summary"], f"{dataset} summary")
    protocol_record = _require_mapping(frozen["protocol"], f"{dataset} protocol record")
    _require_exact_keys(protocol_record, {"relative_path", "sha256", "bytes", "canonical_sha256"}, f"{dataset} protocol record")
    protocol_path = _repo_relative_file(protocol_record["relative_path"], f"{dataset} protocol")
    _require(protocol_path.stat().st_size == protocol_record["bytes"], f"{dataset} protocol byte count differs")
    _require(file_sha256(protocol_path) == protocol_record["sha256"], f"{dataset} protocol file SHA-256 differs")

    summary = _load_json_object(summary_path, f"{dataset} summary")
    protocol = _load_json_object(protocol_path, f"{dataset} protocol")
    canonical_protocol = _protocol_canonical_sha256(protocol)
    _require(canonical_protocol == protocol_record["canonical_sha256"], f"{dataset} frozen protocol SHA-256 differs")

    for label, payload in (("summary", summary), ("protocol", protocol)):
        _require(payload.get("schema") == HISTORICAL_RUN_SCHEMA, f"{dataset} {label} schema differs")
        _require(payload.get("dataset") == dataset, f"{dataset} {label} dataset differs")
        _require(payload.get("method") == "final", f"{dataset} {label} method differs")
        seed_value = payload.get("seed", payload.get("training_seed"))
        _require(type(seed_value) is int and seed_value == TRAINING_SEED, f"{dataset} {label} seed differs")
        _require(payload.get("test_selected") is True, f"{dataset} {label} test-selected disclosure differs")
        _require(payload.get("selection_is_optimistic") is True, f"{dataset} {label} optimism disclosure differs")
        _require(tuple(payload.get("checkpoint_roles", ())) == PARENT_ROLES, f"{dataset} {label} role order differs")
    _require(summary.get("status") == "complete", f"{dataset} summary is incomplete")
    _require(summary.get("seed") == TRAINING_SEED, f"{dataset} summary seed differs")
    _require(summary.get("tss_enabled") is False, f"{dataset} summary TSS flag differs")
    protocol_tss = _require_mapping(protocol.get("tss"), f"{dataset} protocol TSS")
    _require(
        protocol_tss.get("enabled") is False
        and protocol_tss.get("requested_tss_weight") == 0.0
        and protocol_tss.get("statistics_consumed") is False,
        f"{dataset} protocol TSS-off contract differs",
    )
    protocol_recipe = _require_mapping(
        protocol.get("recipe"), f"{dataset} protocol recipe"
    )
    _require(
        protocol_recipe.get("tss_enabled") is False
        and protocol_recipe.get("requested_tss_weight") == 0.0,
        f"{dataset} protocol recipe is not TSS-off",
    )
    _require(summary.get("protocol_sha256") == canonical_protocol, f"{dataset} summary protocol SHA-256 differs")
    _require(summary.get("protocol") == str(protocol_path), f"{dataset} summary protocol path differs")
    _require(summary.get("recipe") == protocol.get("recipe"), f"{dataset} recipe differs between summary/protocol")

    checkpoint_records = _require_mapping(frozen["checkpoints"], f"manifest {dataset} checkpoints")
    summary_checkpoints = _require_mapping(summary.get("checkpoints"), f"{dataset} summary checkpoints")
    _require(set(checkpoint_records) == set(PARENT_ROLES), f"{dataset} manifest checkpoint roles differ")
    _require(set(summary_checkpoints) == set(PARENT_ROLES), f"{dataset} summary checkpoint roles differ")
    for role in PARENT_ROLES:
        record = _require_mapping(checkpoint_records[role], f"manifest {dataset}/{role}")
        _require_exact_keys(record, {"relative_path", "sha256", "bytes", "epoch", "state_key_count", "state_sha256"}, f"manifest {dataset}/{role}")
        path = _repo_relative_file(record["relative_path"], f"{dataset}/{role} checkpoint")
        summary_record = _require_mapping(summary_checkpoints[role], f"{dataset}/{role} summary checkpoint")
        _require(summary_record == {"path": str(path), "sha256": record["sha256"], "bytes": record["bytes"]}, f"{dataset}/{role} summary checkpoint binding differs")
        selected = _require_mapping(summary.get(role), f"{dataset} summary {role}")
        _require(selected.get("path") == str(path), f"{dataset}/{role} selected path differs")
        _require(selected.get("epoch") == record["epoch"], f"{dataset}/{role} selected epoch differs")

    sources = _audit_historical_runtime_sources(
        protocol,
        expected_count=int(manifest["historical_runtime_source_count"]),
    )
    return {
        "dataset": dataset,
        "manifest": {
            "path": str(CURRENT_MANIFEST_PATH),
            "sha256": CURRENT_MANIFEST_SHA256,
        },
        "summary": {
            "path": str(summary_path),
            "sha256": frozen["summary"]["sha256"],
        },
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": protocol_record["sha256"],
            "canonical_sha256": canonical_protocol,
        },
        "historical_runtime_sources": sources,
        "historical_runtime_sources_sha256": canonical_sha256(sources),
        "official_test_data_accessed": False,
        "dataset_loader_imported": False,
        "training_started": False,
        "files_read_scope": [
            "frozen_current_manifest",
            "completed_run_summary",
            "completed_run_protocol",
            "historical_runtime_source_files",
        ],
    }


def _checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    checkpoint = _require_mapping(payload, "Current checkpoint payload")
    _require_exact_keys(checkpoint, EXPECTED_CHECKPOINT_PAYLOAD_KEYS, "Current checkpoint payload")
    state = _require_mapping(checkpoint.get("state_dict"), "Current checkpoint state_dict")
    _require(all(isinstance(value, torch.Tensor) for value in state.values()), "Current checkpoint state values must be tensors")
    return state  # type: ignore[return-value]


def load_current_checkpoint(
    dataset_name: str,
    parent_role: str,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], dict[str, Any]]:
    """Load one exact dataset+role Current checkpoint after full auditing."""

    dataset = _require_dataset(dataset_name)
    role = _require_role(parent_role)
    run_audit = audit_current_run(dataset)
    manifest = load_frozen_current_manifest()
    frozen = _require_mapping(
        _require_mapping(manifest["datasets"], "manifest datasets")[dataset],
        f"manifest {dataset}",
    )
    record = _require_mapping(
        _require_mapping(frozen["checkpoints"], f"manifest {dataset} checkpoints")[role],
        f"manifest {dataset}/{role}",
    )
    path = _repo_relative_file(record["relative_path"], f"{dataset}/{role} checkpoint")
    _require(path.stat().st_size == record["bytes"], f"{dataset}/{role} checkpoint byte count differs")
    checkpoint_sha = file_sha256(path)
    _require(checkpoint_sha == record["sha256"], f"{dataset}/{role} checkpoint SHA-256 differs")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CrossDatasetPBDRV3ModelProtocolError(
            f"could not load frozen Current checkpoint: {path}"
        ) from error
    checkpoint = dict(_require_mapping(payload, "Current checkpoint payload"))
    state = _checkpoint_state(checkpoint)
    expected_metadata = {
        "schema": HISTORICAL_RUN_SCHEMA,
        "dataset": dataset,
        "method": "final",
        "seed": TRAINING_SEED,
        "checkpoint_role": role,
        "tss_enabled": False,
        "requested_tss_weight": 0.0,
        "test_selected": True,
        "selection_is_optimistic": True,
        "selection_source": f"test_{dataset}",
        "protocol_sha256": run_audit["protocol"]["canonical_sha256"],
        "epoch": record["epoch"],
    }
    for name, expected in expected_metadata.items():
        _require(checkpoint.get(name) == expected, f"{dataset}/{role} checkpoint {name} differs")

    summary_path = Path(run_audit["summary"]["path"])
    summary = _load_json_object(summary_path, f"{dataset} summary")
    _require(checkpoint.get("recipe") == summary.get("recipe"), f"{dataset}/{role} checkpoint recipe differs")
    _require(canonical_sha256(checkpoint.get("test_metrics")) == canonical_sha256(summary[role]["metrics"]), f"{dataset}/{role} checkpoint test-metric binding differs")
    metadata = _require_mapping(checkpoint.get("model_metadata"), f"{dataset}/{role} model metadata")
    _require(metadata.get("dataset_name") == dataset, f"{dataset}/{role} model metadata dataset differs")
    _require(metadata.get("training_seed") == TRAINING_SEED, f"{dataset}/{role} model metadata seed differs")
    _require(metadata.get("parent_checkpoint") is None and metadata.get("warm_start_used") is False, f"{dataset}/{role} historical scratch provenance differs")
    _require(metadata.get("selected_model_state_key_count") == CURRENT_STATE_KEY_COUNT, f"{dataset}/{role} model metadata state count differs")

    _require(len(state) == CURRENT_STATE_KEY_COUNT == record["state_key_count"], f"{dataset}/{role} Current state-key count differs")
    _require(not any(name.startswith(PBDR_V3_STATE_PREFIX) for name in state), f"{dataset}/{role} Current state contains PBDR-V3 keys")
    _require(set(SURVIVAL_STATE_KEYS) <= set(state), f"{dataset}/{role} Current state lacks Survival keys")
    _require(all(int(torch.count_nonzero(state[name])) == 0 for name in SURVIVAL_STATE_KEYS), f"{dataset}/{role} TSS-off Survival state is not exact zero")
    for name, tensor in state.items():
        _require(bool(torch.isfinite(tensor).all()), f"{dataset}/{role} state tensor {name!r} is non-finite")
    state_sha = tensor_mapping_sha256(state)
    _require(state_sha == record["state_sha256"], f"{dataset}/{role} Current state SHA-256 differs")
    parent_record = {
        "dataset": dataset,
        "checkpoint_role": role,
        "path": str(path),
        "sha256": checkpoint_sha,
        "bytes": path.stat().st_size,
        "epoch": int(checkpoint["epoch"]),
        "state_key_count": len(state),
        "state_sha256": state_sha,
        "schema": checkpoint["schema"],
        "protocol_sha256": checkpoint["protocol_sha256"],
        "current_run_audit": run_audit,
        "official_test_data_accessed": False,
    }
    return checkpoint, state, parent_record


def load_original_checkpoint(
    dataset_name: str,
    checkpoint_role: str,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], dict[str, Any]]:
    """Load one manifest-authorized Original checkpoint without test access."""

    dataset = _require_dataset(dataset_name)
    role = _require_role(checkpoint_role)
    authority = audit_original_authority(dataset)
    expected = _require_mapping(
        _require_mapping(authority["checkpoints"], "audited Original checkpoints")[role],
        f"audited Original {dataset}/{role}",
    )
    path = _absolute_repo_file(
        expected["frozen_path"], f"Original {dataset}/{role} frozen checkpoint"
    )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CrossDatasetPBDRV3ModelProtocolError(
            f"could not load frozen Original checkpoint: {path}"
        ) from error
    checkpoint = dict(_require_mapping(payload, "Original checkpoint payload"))
    _require_exact_keys(
        checkpoint,
        EXPECTED_ORIGINAL_CHECKPOINT_PAYLOAD_KEYS,
        "Original checkpoint payload",
    )
    state = _require_mapping(
        checkpoint.get("state_dict"), "Original checkpoint state_dict"
    )
    _require(
        all(isinstance(value, torch.Tensor) for value in state.values()),
        "Original checkpoint state values must be tensors",
    )
    expected_metadata = {
        "schema": ORIGINAL_RUN_SCHEMA,
        "dataset": dataset,
        "method": "original",
        "seed": TRAINING_SEED,
        "checkpoint_role": role,
        "epoch": expected["epoch"],
        "protocol_sha256": authority["protocol"]["canonical_sha256"],
        "test_selected": True,
        "selection_is_optimistic": True,
        "selection_source": f"test_{dataset}",
    }
    for name, expected_value in expected_metadata.items():
        _require(
            checkpoint.get(name) == expected_value,
            f"Original {dataset}/{role} checkpoint {name} differs",
        )
    metadata = _require_mapping(
        checkpoint.get("model_metadata"),
        f"Original {dataset}/{role} model metadata",
    )
    _require(metadata.get("dataset_name") == dataset, f"Original {dataset}/{role} model metadata dataset differs")
    _require(metadata.get("method") == "original_scratch", f"Original {dataset}/{role} model metadata method differs")
    _require(metadata.get("training_seed") == TRAINING_SEED, f"Original {dataset}/{role} model metadata seed differs")
    _require(metadata.get("parent_checkpoint") is None and metadata.get("warm_start_used") is False, f"Original {dataset}/{role} scratch provenance differs")
    _require(metadata.get("selected_model_state_key_count") == ORIGINAL_STATE_KEY_COUNT, f"Original {dataset}/{role} metadata state count differs")

    _require(len(state) == ORIGINAL_STATE_KEY_COUNT == expected["state_key_count"], f"Original {dataset}/{role} state-key count differs")
    for name, tensor in state.items():
        _require(bool(torch.isfinite(tensor).all()), f"Original {dataset}/{role} state tensor {name!r} is non-finite")
    state_sha = tensor_mapping_sha256(state)  # type: ignore[arg-type]
    _require(state_sha == expected["state_sha256"], f"Original {dataset}/{role} state SHA-256 differs")
    fixed = dict(
        _require_mapping(
            expected["fixed_threshold_0_5_metrics"],
            f"Original {dataset}/{role} fixed metrics",
        )
    )
    checkpoint_metrics = _require_mapping(
        checkpoint.get("test_metrics"),
        f"Original {dataset}/{role} checkpoint metrics",
    )
    fixed_payload_metrics = {
        name: value
        for name, value in fixed.items()
        if name not in {"epoch", "threshold"}
    }
    _require(
        canonical_sha256(_json_ready(checkpoint_metrics))
        == canonical_sha256(_json_ready(fixed_payload_metrics)),
        f"Original {dataset}/{role} checkpoint metric binding differs",
    )
    record = {
        "dataset": dataset,
        "checkpoint_role": role,
        "path": str(path),
        "source_path": expected["source_path"],
        "sha256": expected["sha256"],
        "bytes": expected["bytes"],
        "epoch": expected["epoch"],
        "state_key_count": len(state),
        "state_sha256": state_sha,
        "schema": checkpoint["schema"],
        "protocol_sha256": checkpoint["protocol_sha256"],
        "fixed_threshold_0_5_metrics": fixed,
        "selection_policy": authority["selection_policy"],
        "authority_audit": authority,
        "official_test_data_accessed": False,
    }
    return checkpoint, state, record


def build_original_inference_model(
    dataset_name: str,
    checkpoint_role: str,
    *,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build and strict-load the exact Original inference graph."""

    dataset = _require_dataset(dataset_name)
    role = _require_role(checkpoint_role)
    formal_seed = _require_seed(seed)
    _, state, checkpoint_record = load_original_checkpoint(dataset, role)
    with _preserve_process_rng():
        model, raw = original_models.build_paper_model(
            "original",
            dataset,
            seed=formal_seed,
            training=False,
        )
    _require(len(model.state_dict()) == ORIGINAL_STATE_KEY_COUNT, "Original inference graph state-key count differs")
    incompatible = model.load_state_dict(state, strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "Original inference strict load returned incompatible keys")
    model.eval()
    model.mode = "test"
    installed_sha = tensor_mapping_sha256(model.state_dict())
    _require(installed_sha == checkpoint_record["state_sha256"], "Original inference installed state SHA-256 differs")
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "checkpoint_role": role,
        "training_seed": formal_seed,
        "original_checkpoint": checkpoint_record,
        "state_key_count": len(model.state_dict()),
        "state_sha256": installed_sha,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "strict_load": True,
        "target_survival_registered": False,
        "raw_builder_metadata": raw,
        "official_test_data_accessed": False,
        "evaluation_started": False,
    }
    _require(metadata["parameter_count"] == original_models.ORIGINAL_PARAMETER_COUNT, "Original inference parameter count differs")
    return model, metadata


@contextmanager
def _preserve_process_rng() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def build_stage1_training_model(
    dataset_name: str,
    parent_role: str,
    *,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a frozen-base Stage-1 graph from one audited Current artifact."""

    dataset = _require_dataset(dataset_name)
    role = _require_role(parent_role)
    formal_seed = _require_seed(seed)
    checkpoint, current_state, parent = load_current_checkpoint(dataset, role)
    with _preserve_process_rng():
        model, raw = build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(formal_seed)
    candidate_keys = set(model.state_dict())
    _require(candidate_keys - set(PBDR_V3_STATE_KEYS) == set(current_state), "Current and PBDR-V3 inherited key sets differ")
    incompatible = model.load_state_dict(current_state, strict=False)
    _require(tuple(incompatible.missing_keys) == PBDR_V3_STATE_KEYS, "Current warm-start missing-key set is not exactly PBDR_V3_STATE_KEYS")
    _require(not incompatible.unexpected_keys, "Current warm-start returned unexpected keys")
    installed = model.state_dict()
    changed = [
        name
        for name, value in current_state.items()
        if not torch.equal(value.detach().cpu(), installed[name].detach().cpu())
    ]
    _require(not changed, f"Current tensors changed during warm-start: {changed[:5]}")
    _require(core.base_state_sha256(model) == parent["state_sha256"], "installed Current state hash differs from parent checkpoint")
    validated = validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(
        model,
        require_zero_initialized_heads=False,
        require_identity_initialized_qfg=False,
        require_identity_initialized_pbdr_v3=True,
    )
    freeze = core.configure_stage1(model)
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "training_seed": formal_seed,
        "parent_role": role,
        "parent_checkpoint": parent,
        "parent_checkpoint_payload_schema": checkpoint["schema"],
        "warm_start_used": True,
        "warm_start_strict_contract": "strict_false_with_exact_pbdr_v3_missing_and_no_unexpected",
        "current_state_key_count": CURRENT_STATE_KEY_COUNT,
        "training_state_key_count": len(model.state_dict()),
        "pbdr_v3_state_keys": list(PBDR_V3_STATE_KEYS),
        "all_current_tensors_bitwise_equal_after_load": True,
        "current_state_sha256_after_load": core.base_state_sha256(model),
        "initial_pbdr_v3_state_sha256": tensor_mapping_sha256(
            {name: model.state_dict()[name] for name in PBDR_V3_STATE_KEYS}
        ),
        "pbdr_v3_integration_version": PBDR_V3_INTEGRATION_VERSION,
        "architecture_manifest": model.architecture_manifest(),
        "architecture_id": canonical_sha256(model.architecture_manifest()),
        "builder_validation": validated,
        "stage1_freeze_audit": freeze,
        "raw_builder_metadata": raw,
        "official_test_data_accessed": False,
        "training_started": False,
    }
    _require(metadata["training_state_key_count"] == TRAINING_STATE_KEY_COUNT, "formal PBDR-V3 training state-key count differs")
    return model, metadata


def strip_training_only_survival_state(
    training_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return core.strip_training_only_survival_state(training_state)


def build_inference_model_from_candidate_state(
    training_state: Mapping[str, torch.Tensor],
    *,
    dataset_name: str,
    parent_role: str,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly bind a candidate state to its dataset+role Current parent."""

    dataset = _require_dataset(dataset_name)
    role = _require_role(parent_role)
    formal_seed = _require_seed(seed)
    _, parent_state, parent = load_current_checkpoint(dataset, role)
    stripped = strip_training_only_survival_state(training_state)
    parent_inference = {
        name: value
        for name, value in parent_state.items()
        if name not in set(SURVIVAL_STATE_KEYS)
    }
    candidate_base = {
        name: value
        for name, value in stripped.items()
        if not name.startswith(PBDR_V3_STATE_PREFIX)
    }
    _require(set(candidate_base) == set(parent_inference), "candidate inference base key set differs from Current")
    changed = [
        name
        for name, value in parent_inference.items()
        if not torch.equal(value.detach().cpu(), candidate_base[name].detach().cpu())
    ]
    _require(not changed, f"candidate modified frozen Current tensors: {changed[:5]}")
    with _preserve_process_rng():
        model, raw = build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(formal_seed)
    incompatible = model.load_state_dict(stripped, strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "candidate inference strict load returned incompatible keys")
    validated = validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(model)
    model.eval()
    model.mode = "test"
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "parent_role": role,
        "parent_checkpoint": parent,
        "training_state_key_count": len(training_state),
        "inference_state_key_count": len(stripped),
        "stripped_training_only_state_keys": list(SURVIVAL_STATE_KEYS),
        "base_bitwise_equal_to_parent": True,
        "strict_load": True,
        "builder_validation": validated,
        "raw_builder_metadata": raw,
        "official_test_data_accessed": False,
    }
    return model, metadata


__all__ = [
    "CURRENT_MANIFEST_PATH",
    "CURRENT_MANIFEST_SHA256",
    "CURRENT_STATE_KEY_COUNT",
    "CrossDatasetPBDRV3ModelProtocolError",
    "DATASETS",
    "INFERENCE_STATE_KEY_COUNT",
    "ORIGINAL_AUTHORITY_MANIFEST_SHA256",
    "ORIGINAL_MANIFEST_PATH",
    "ORIGINAL_MANIFEST_SHA256",
    "ORIGINAL_STATE_KEY_COUNT",
    "PARENT_ROLES",
    "RUNTIME_DEPENDENCY_RELATIVE_PATHS",
    "SCHEMA",
    "TRAINING_SEED",
    "TRAINING_STATE_KEY_COUNT",
    "audit_current_run",
    "audit_original_authority",
    "build_inference_model_from_candidate_state",
    "build_original_inference_model",
    "build_stage1_training_model",
    "canonical_sha256",
    "file_sha256",
    "load_current_checkpoint",
    "load_frozen_current_manifest",
    "load_frozen_original_manifest",
    "load_original_checkpoint",
    "runtime_source_paths",
    "runtime_source_records",
    "strip_training_only_survival_state",
    "tensor_mapping_sha256",
]
