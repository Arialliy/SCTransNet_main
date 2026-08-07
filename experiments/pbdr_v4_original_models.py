#!/usr/bin/env python3
"""Read-only three-dataset Original model registry for PBDR-V4.

NUDT-SIRST and IRSTD-1K are delegated to the already frozen two-dataset
PBDR-V3 authority.  NUAA-SIRST is loaded only from the code-pinned Original
records in the completed four-dataset checkpoint manifest.  This module has no
dataset, split-index, evaluation, optimizer, or training dependency.

Both public APIs require an explicit dataset and checkpoint role.  No path
override is accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from experiments import four_dataset_models_seed42_v1 as four_dataset_models
from experiments import two_dataset_pbdr_v3_models_seed42_v1 as two_dataset_models


SCHEMA = "sctransnet_pbdr_v4_original_models/v1"
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
CHECKPOINT_ROLES = ("best_miou", "best_pd")
TRAINING_SEED = 42
NUAA_DATASET = "NUAA-SIRST"
DELEGATED_DATASETS = ("NUDT-SIRST", "IRSTD-1K")

REPO_ROOT = Path(__file__).resolve().parents[1]
NUAA_AUTHORITY_MANIFEST_RELATIVE_PATH = (
    "results/four_dataset_seed42_v1/selected_checkpoints/"
    "checkpoint_manifest.json"
)
NUAA_AUTHORITY_MANIFEST_PATH = (
    REPO_ROOT / NUAA_AUTHORITY_MANIFEST_RELATIVE_PATH
)
NUAA_AUTHORITY_MANIFEST_BYTES = 73_520
NUAA_AUTHORITY_MANIFEST_SHA256 = (
    "f286c2f07113be079a2f447b3a2a4e868c81df58ac06cc4acda2de2210249799"
)
NUAA_PROTOCOL_SHA256 = (
    "7bf99ecefea60fec299f93e6c8b56d8139a967af6cbe4ac4af9c6d2675d33f8d"
)
NUAA_INITIAL_ORIGINAL_STATE_SHA256 = (
    "6e4d86a03b9ad912f91d046fe1959c85f543cfc63044d5bd0006495f71d46189"
)


@dataclass(frozen=True)
class FrozenOriginalCheckpointPin:
    """Code-pinned immutable identity for one NUAA Original checkpoint."""

    role: str
    frozen_relative_path: str
    source_relative_path: str
    file_bytes: int
    file_sha256: str
    epoch: int
    state_key_count: int
    state_sha256: str


NUAA_CHECKPOINT_PINS: Mapping[str, FrozenOriginalCheckpointPin] = (
    MappingProxyType(
        {
            "best_miou": FrozenOriginalCheckpointPin(
                role="best_miou",
                frozen_relative_path=(
                    "results/four_dataset_seed42_v1/selected_checkpoints/"
                    "NUAA-SIRST/original/best_miou.pth.tar"
                ),
                source_relative_path=(
                    "results/four_dataset_seed42_v1/runs/NUAA-SIRST/"
                    "original/seed_42/checkpoints/best_miou.pth.tar"
                ),
                file_bytes=45_528_975,
                file_sha256=(
                    "b5edfe46fc54d5e74c1896a43f0f44c8970c143d90eaebb1098cac760f119ead"
                ),
                epoch=830,
                state_key_count=510,
                state_sha256=(
                    "48a9ada9fae4b7e0fe9068916bf3a7011ca7379d4dfa8f3cd84cd593ebd986ac"
                ),
            ),
            "best_pd": FrozenOriginalCheckpointPin(
                role="best_pd",
                frozen_relative_path=(
                    "results/four_dataset_seed42_v1/selected_checkpoints/"
                    "NUAA-SIRST/original/best_pd.pth.tar"
                ),
                source_relative_path=(
                    "results/four_dataset_seed42_v1/runs/NUAA-SIRST/"
                    "original/seed_42/checkpoints/best_pd.pth.tar"
                ),
                file_bytes=45_527_943,
                file_sha256=(
                    "9638f92d6aac6114a5cfb7b8124f90e94c7569053735558c4b9cde73fb8ebd7d"
                ),
                epoch=440,
                state_key_count=510,
                state_sha256=(
                    "ebb31ad2e621ea12a479572600fcfadecab016dac2234e591efd17c999a88ee1"
                ),
            ),
        }
    )
)

EXPECTED_CHECKPOINT_PAYLOAD_KEYS = frozenset(
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
EXPECTED_MODEL_METADATA_KEYS = frozenset(
    {
        "dataset_name",
        "method",
        "pair",
        "parent_checkpoint",
        "schema",
        "selected_model_parameter_count",
        "selected_model_state_key_count",
        "selected_model_state_sha256",
        "training_graph_requested",
        "training_seed",
        "warm_start_used",
    }
)


class PBDRV4OriginalModelRegistryError(ValueError):
    """A requested Original authority, checkpoint, or graph is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4OriginalModelRegistryError(message)


def _require_dataset(dataset_name: str) -> str:
    _require(
        type(dataset_name) is str and dataset_name in DATASETS,
        f"dataset_name must be one of {DATASETS}",
    )
    return dataset_name


def _require_role(checkpoint_role: str) -> str:
    _require(
        type(checkpoint_role) is str and checkpoint_role in CHECKPOINT_ROLES,
        f"checkpoint_role must be one of {CHECKPOINT_ROLES}",
    )
    return checkpoint_role


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(
        all(type(key) is str for key in value),
        f"{label} keys must be strings",
    )
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    _require(set(value) == set(expected), f"{label} keys differ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _repo_file(relative_path: str, label: str) -> Path:
    _require(
        type(relative_path) is str
        and bool(relative_path)
        and not os.path.isabs(relative_path),
        f"{label} relative path is invalid",
    )
    relative = Path(relative_path)
    _require(".." not in relative.parts, f"{label} escapes repository root")
    candidate = REPO_ROOT / relative
    root = REPO_ROOT.resolve(strict=True)
    current = REPO_ROOT
    _require(not current.is_symlink(), "repository root must not be a symlink")
    for component in relative.parts:
        current = current / component
        _require(not current.is_symlink(), f"{label} path contains a symlink")
    _require(candidate.is_file(), f"{label} is not a regular file")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PBDRV4OriginalModelRegistryError(
            f"{label} escapes repository root"
        ) from error
    return resolved


def _load_nuaa_manifest() -> dict[str, Any]:
    path = _repo_file(
        NUAA_AUTHORITY_MANIFEST_RELATIVE_PATH,
        "NUAA Original authority manifest",
    )
    _require(
        path == NUAA_AUTHORITY_MANIFEST_PATH.resolve(strict=True),
        "NUAA Original authority manifest path differs",
    )
    _require(
        path.stat().st_size == NUAA_AUTHORITY_MANIFEST_BYTES,
        "NUAA Original authority manifest byte count differs",
    )
    _require(
        file_sha256(path) == NUAA_AUTHORITY_MANIFEST_SHA256,
        "NUAA Original authority manifest SHA-256 differs",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PBDRV4OriginalModelRegistryError(
            "could not parse NUAA Original authority manifest"
        ) from error
    manifest = dict(_require_mapping(value, "NUAA Original authority manifest"))
    expected = {
        "schema": "sctransnet_four_dataset_seed42_checkpoint_manifest_v1",
        "status": "complete",
        "experiment": "four_training_regimes_original_vs_final",
        "seed": TRAINING_SEED,
        "record_count": 8,
        "no_fabricated_results": True,
    }
    for name, expected_value in expected.items():
        _require(
            manifest.get(name) == expected_value,
            f"NUAA Original authority manifest {name} differs",
        )
    records = manifest.get("records")
    _require(isinstance(records, list), "authority records must be a list")
    _require(len(records) == manifest["record_count"], "authority record count differs")
    return manifest


def _nuaa_original_record(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    records = manifest.get("records")
    _require(isinstance(records, list), "authority records must be a list")
    matching = [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("dataset") == NUAA_DATASET
        and item.get("method") == "original"
    ]
    _require(
        len(matching) == 1,
        "authority must contain exactly one NUAA-SIRST Original record",
    )
    record = _require_mapping(matching[0], "NUAA Original authority record")
    expected = {
        "dataset": NUAA_DATASET,
        "method": "original",
        "method_label": "Original",
        "seed": TRAINING_SEED,
        "audit_passed": True,
    }
    for name, expected_value in expected.items():
        _require(
            record.get(name) == expected_value,
            f"NUAA Original authority record {name} differs",
        )
    checkpoints = _require_mapping(
        record.get("checkpoints"), "NUAA Original authority checkpoints"
    )
    _require(
        set(checkpoints) == set(CHECKPOINT_ROLES),
        "NUAA Original authority checkpoint roles differ",
    )
    return record


def _nuaa_checkpoint_authority(
    record: Mapping[str, Any], role: str
) -> tuple[Path, FrozenOriginalCheckpointPin, Mapping[str, Any]]:
    pin = NUAA_CHECKPOINT_PINS[role]
    _require(pin.role == role, f"NUAA Original {role} code pin role differs")
    checkpoints = _require_mapping(
        record.get("checkpoints"), "NUAA Original authority checkpoints"
    )
    selected = _require_mapping(
        checkpoints.get(role), f"NUAA Original authority {role} checkpoint"
    )
    frozen_path = _repo_file(
        pin.frozen_relative_path, f"NUAA Original {role} frozen checkpoint"
    )
    source_path = (REPO_ROOT / pin.source_relative_path).resolve(strict=False)
    expected = {
        "checkpoint_role": role,
        "epoch": pin.epoch,
        "frozen_path": str(frozen_path),
        "source_path": str(source_path),
        "sha256": pin.file_sha256,
        "test_selected": True,
        "selection_is_optimistic": True,
    }
    for name, expected_value in expected.items():
        _require(
            selected.get(name) == expected_value,
            f"NUAA Original {role} authority {name} differs",
        )
    fixed = _require_mapping(
        selected.get("fixed_threshold_0_5_metrics"),
        f"NUAA Original {role} fixed metrics",
    )
    _require(fixed.get("epoch") == pin.epoch, f"NUAA Original {role} metric epoch differs")
    _require(fixed.get("threshold") == 0.5, f"NUAA Original {role} metric threshold differs")
    _require(
        frozen_path.stat().st_size == pin.file_bytes,
        f"NUAA Original {role} checkpoint byte count differs",
    )
    _require(
        file_sha256(frozen_path) == pin.file_sha256,
        f"NUAA Original {role} checkpoint SHA-256 differs",
    )
    return frozen_path, pin, selected


def _validate_nuaa_checkpoint_payload(
    value: Any,
    *,
    role: str,
    pin: FrozenOriginalCheckpointPin,
    selected: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    payload = dict(_require_mapping(value, "NUAA Original checkpoint payload"))
    _require_exact_keys(
        payload, EXPECTED_CHECKPOINT_PAYLOAD_KEYS, "NUAA Original checkpoint payload"
    )
    expected = {
        "schema": "sctransnet_four_dataset_seed42_exact_v1",
        "dataset": NUAA_DATASET,
        "method": "original",
        "seed": TRAINING_SEED,
        "checkpoint_role": role,
        "epoch": pin.epoch,
        "protocol_sha256": NUAA_PROTOCOL_SHA256,
        "test_selected": True,
        "selection_is_optimistic": True,
        "selection_source": "test_NUAA-SIRST",
    }
    for name, expected_value in expected.items():
        _require(
            payload.get(name) == expected_value,
            f"NUAA Original {role} checkpoint {name} differs",
        )

    metadata = _require_mapping(
        payload.get("model_metadata"),
        f"NUAA Original {role} model metadata",
    )
    _require_exact_keys(
        metadata, EXPECTED_MODEL_METADATA_KEYS, f"NUAA Original {role} model metadata"
    )
    expected_metadata = {
        "schema": four_dataset_models.BUILDER_SCHEMA,
        "method": "original_scratch",
        "training_graph_requested": True,
        "dataset_name": NUAA_DATASET,
        "training_seed": TRAINING_SEED,
        "selected_model_state_key_count": pin.state_key_count,
        "selected_model_state_sha256": NUAA_INITIAL_ORIGINAL_STATE_SHA256,
        "selected_model_parameter_count": four_dataset_models.ORIGINAL_PARAMETER_COUNT,
        "warm_start_used": False,
        "parent_checkpoint": None,
    }
    for name, expected_value in expected_metadata.items():
        _require(
            metadata.get(name) == expected_value,
            f"NUAA Original {role} model metadata {name} differs",
        )
    pair = _require_mapping(
        metadata.get("pair"), f"NUAA Original {role} paired-model metadata"
    )
    expected_pair = {
        "schema": four_dataset_models.BUILDER_SCHEMA,
        "dataset_name": NUAA_DATASET,
        "training_seed": TRAINING_SEED,
        "initialization_mode": "true_scratch",
        "parent_checkpoint": None,
        "parent_checkpoint_load_count": 0,
        "warm_start_used": False,
        "paired_initialization": True,
    }
    for name, expected_value in expected_pair.items():
        _require(
            pair.get(name) == expected_value,
            f"NUAA Original {role} paired metadata {name} differs",
        )
    original = _require_mapping(
        pair.get("original"), f"NUAA Original {role} graph metadata"
    )
    expected_original = {
        "method": "original_scratch",
        "parameter_count": four_dataset_models.ORIGINAL_PARAMETER_COUNT,
        "state_key_count": pin.state_key_count,
        "state_sha256": NUAA_INITIAL_ORIGINAL_STATE_SHA256,
        "training_graph": "original_sctransnet",
        "inference_graph": "original_sctransnet",
    }
    for name, expected_value in expected_original.items():
        _require(
            original.get(name) == expected_value,
            f"NUAA Original {role} graph metadata {name} differs",
        )

    state = _require_mapping(
        payload.get("state_dict"), f"NUAA Original {role} state_dict"
    )
    _require(
        len(state) == pin.state_key_count == four_dataset_models.ORIGINAL_STATE_KEY_COUNT,
        f"NUAA Original {role} state-key count differs",
    )
    _require(
        all(type(name) is str and isinstance(tensor, torch.Tensor) for name, tensor in state.items()),
        f"NUAA Original {role} state_dict must map strings to tensors",
    )
    for name, tensor in state.items():
        _require(
            bool(torch.isfinite(tensor).all()),
            f"NUAA Original {role} state tensor {name!r} is non-finite",
        )
    state_sha = four_dataset_models.state_dict_sha256(state)  # type: ignore[arg-type]
    _require(
        state_sha == pin.state_sha256,
        f"NUAA Original {role} state SHA-256 differs",
    )

    fixed = _require_mapping(
        selected.get("fixed_threshold_0_5_metrics"),
        f"NUAA Original {role} fixed metrics",
    )
    checkpoint_metrics = _require_mapping(
        payload.get("test_metrics"), f"NUAA Original {role} checkpoint metrics"
    )
    fixed_payload_metrics = {
        name: value
        for name, value in fixed.items()
        if name not in {"epoch", "threshold"}
    }
    _require(
        canonical_sha256(checkpoint_metrics)
        == canonical_sha256(fixed_payload_metrics),
        f"NUAA Original {role} checkpoint metric binding differs",
    )
    return payload, state  # type: ignore[return-value]


def _validate_delegated_load(
    dataset: str,
    role: str,
    value: Any,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], dict[str, Any]]:
    _require(
        isinstance(value, tuple) and len(value) == 3,
        "delegated Original loader return contract differs",
    )
    payload = dict(_require_mapping(value[0], "delegated Original payload"))
    state = _require_mapping(value[1], "delegated Original state")
    record = dict(_require_mapping(value[2], "delegated Original record"))
    _require(payload.get("dataset") == dataset, "delegated Original payload dataset differs")
    _require(payload.get("checkpoint_role") == role, "delegated Original payload role differs")
    _require(record.get("dataset") == dataset, "delegated Original record dataset differs")
    _require(record.get("checkpoint_role") == role, "delegated Original record role differs")
    _require(
        all(type(name) is str and isinstance(tensor, torch.Tensor) for name, tensor in state.items()),
        "delegated Original state must map strings to tensors",
    )
    return payload, state, record  # type: ignore[return-value]


def load_original_checkpoint(
    dataset_name: str,
    checkpoint_role: str,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], dict[str, Any]]:
    """Return one role-bound Original payload/state/record without data access."""

    dataset = _require_dataset(dataset_name)
    role = _require_role(checkpoint_role)
    if dataset in DELEGATED_DATASETS:
        return _validate_delegated_load(
            dataset,
            role,
            two_dataset_models.load_original_checkpoint(dataset, role),
        )

    manifest = _load_nuaa_manifest()
    authority_record = _nuaa_original_record(manifest)
    path, pin, selected = _nuaa_checkpoint_authority(authority_record, role)
    try:
        raw_payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise PBDRV4OriginalModelRegistryError(
            f"could not load NUAA Original {role} checkpoint"
        ) from error
    payload, state = _validate_nuaa_checkpoint_payload(
        raw_payload,
        role=role,
        pin=pin,
        selected=selected,
    )
    record = {
        "dataset": dataset,
        "checkpoint_role": role,
        "path": str(path),
        "source_path": str((REPO_ROOT / pin.source_relative_path).resolve(strict=False)),
        "sha256": pin.file_sha256,
        "bytes": pin.file_bytes,
        "epoch": pin.epoch,
        "state_key_count": pin.state_key_count,
        "state_sha256": pin.state_sha256,
        "schema": payload["schema"],
        "protocol_sha256": payload["protocol_sha256"],
        "fixed_threshold_0_5_metrics": dict(
            _require_mapping(
                selected["fixed_threshold_0_5_metrics"],
                f"NUAA Original {role} fixed metrics",
            )
        ),
        "authority_manifest": {
            "path": str(NUAA_AUTHORITY_MANIFEST_PATH.resolve(strict=True)),
            "bytes": NUAA_AUTHORITY_MANIFEST_BYTES,
            "sha256": NUAA_AUTHORITY_MANIFEST_SHA256,
        },
        "official_test_data_accessed": False,
        "dataset_loader_imported": False,
        "evaluation_started": False,
    }
    return payload, state, record


def _validate_ready_model(model: Any, dataset: str, role: str) -> nn.Module:
    _require(isinstance(model, nn.Module), "Original builder did not return a Module")
    _require(not model.training, f"Original {dataset}/{role} model is not in eval mode")
    _require(getattr(model, "mode", None) == "test", f"Original {dataset}/{role} model mode differs")
    return model


def build_original_inference_model(
    dataset_name: str,
    checkpoint_role: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build an eval, ``mode='test'`` Original graph for one exact role."""

    dataset = _require_dataset(dataset_name)
    role = _require_role(checkpoint_role)
    if dataset in DELEGATED_DATASETS:
        value = two_dataset_models.build_original_inference_model(
            dataset, role, seed=TRAINING_SEED
        )
        _require(
            isinstance(value, tuple) and len(value) == 2,
            "delegated Original builder return contract differs",
        )
        model = _validate_ready_model(value[0], dataset, role)
        metadata = dict(_require_mapping(value[1], "delegated Original metadata"))
        _require(metadata.get("dataset") == dataset, "delegated Original metadata dataset differs")
        _require(metadata.get("checkpoint_role") == role, "delegated Original metadata role differs")
        return model, metadata

    _, state, checkpoint_record = load_original_checkpoint(dataset, role)
    try:
        model, raw_metadata = four_dataset_models.build_paper_model(
            "original",
            dataset,
            seed=TRAINING_SEED,
            training=False,
        )
    except Exception as error:
        raise PBDRV4OriginalModelRegistryError(
            "could not construct the NUAA Original inference graph"
        ) from error
    raw = _require_mapping(raw_metadata, "NUAA Original builder metadata")
    expected_raw = {
        "schema": four_dataset_models.BUILDER_SCHEMA,
        "method": "original_scratch",
        "training_graph_requested": False,
        "dataset_name": dataset,
        "training_seed": TRAINING_SEED,
        "selected_model_parameter_count": four_dataset_models.ORIGINAL_PARAMETER_COUNT,
        "selected_model_state_key_count": four_dataset_models.ORIGINAL_STATE_KEY_COUNT,
        "warm_start_used": False,
        "parent_checkpoint": None,
    }
    for name, expected_value in expected_raw.items():
        _require(
            raw.get(name) == expected_value,
            f"NUAA Original builder metadata {name} differs",
        )
    _require(
        isinstance(model, nn.Module), "NUAA Original builder did not return a Module"
    )
    _require(
        len(model.state_dict()) == four_dataset_models.ORIGINAL_STATE_KEY_COUNT,
        "NUAA Original inference graph state-key count differs",
    )
    incompatible = model.load_state_dict(state, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "NUAA Original strict load returned incompatible keys",
    )
    model.eval()
    model.mode = "test"
    model = _validate_ready_model(model, dataset, role)
    installed_sha = four_dataset_models.state_dict_sha256(model.state_dict())
    _require(
        installed_sha == checkpoint_record["state_sha256"],
        "NUAA Original installed state SHA-256 differs",
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(
        parameter_count == four_dataset_models.ORIGINAL_PARAMETER_COUNT,
        "NUAA Original inference parameter count differs",
    )
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "checkpoint_role": role,
        "training_seed": TRAINING_SEED,
        "original_checkpoint": checkpoint_record,
        "state_key_count": len(model.state_dict()),
        "state_sha256": installed_sha,
        "parameter_count": parameter_count,
        "strict_load": True,
        "raw_builder_metadata": dict(raw),
        "official_test_data_accessed": False,
        "dataset_loader_imported": False,
        "evaluation_started": False,
    }
    return model, metadata


__all__ = [
    "CHECKPOINT_ROLES",
    "DATASETS",
    "DELEGATED_DATASETS",
    "FrozenOriginalCheckpointPin",
    "NUAA_AUTHORITY_MANIFEST_BYTES",
    "NUAA_AUTHORITY_MANIFEST_PATH",
    "NUAA_AUTHORITY_MANIFEST_RELATIVE_PATH",
    "NUAA_AUTHORITY_MANIFEST_SHA256",
    "NUAA_CHECKPOINT_PINS",
    "PBDRV4OriginalModelRegistryError",
    "SCHEMA",
    "TRAINING_SEED",
    "build_original_inference_model",
    "load_original_checkpoint",
]
