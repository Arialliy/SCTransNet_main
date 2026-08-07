#!/usr/bin/env python3
"""Unified, source-bound PBDR-V4 model registry for three datasets.

This module is deliberately model-only.  It reuses the already-audited
PBDR-V3 Current checkpoint registries, but never imports a dataset module,
opens an index, constructs an optimizer, or evaluates the official test set.

Every public model builder requires an explicit dataset, role, and stage.
Stage 1 is the exact 27-state-key identity extension of Current.  Stage 2 can
only be initialized from a complete Stage-1 checkpoint payload whose parent,
Current state, source, split, atlas, initialization, and tensor-state hashes
all validate.  Frozen Current references are independent model objects; they
are never aliases of a Candidate model.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

import torch
import torch.nn as nn

from experiments import three_dataset_pbdr_v3_models_seed42_v1 as nuaa_registry
from experiments import two_dataset_pbdr_v3_models_seed42_v1 as cross_registry
from experiments.pbdr_v4_state_contract import (
    Stage,
    audit_candidate_against_current,
    audit_training_modes,
    configure_stage_training,
    state_semantic_sha256,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v4 import (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
    PBDR_V4_STATE_KEYS,
    PBDR_V4_STATE_PREFIX,
    SURVIVAL_STATE_KEYS,
    build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint,
    build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model,
    build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model,
    warm_start_formal_pbdr_v4_from_current,
)


SCHEMA = "sctransnet_pbdr_v4_models_seed42_v1/v1"
TRAINING_SEED = 42
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
ROLES = ("best_miou", "best_pd")
STAGES: tuple[Stage, ...] = ("stage1", "stage2")
CURRENT_STATE_KEY_COUNT = FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
TRAINING_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT
)
INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARENT_BINDING_KEYS = (
    "dataset",
    "checkpoint_role",
    "path",
    "sha256",
    "bytes",
    "state_key_count",
    "state_sha256",
    "epoch",
    "schema",
    "protocol_sha256",
)
_CANDIDATE_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "dataset",
        "role",
        "parent_role",
        "stage",
        "architecture_manifest",
        "state_dict",
        "state_key_count",
        "state_sha256",
        "parent_checkpoint",
        "parent_checkpoint_sha256",
        "parent_state_sha256",
        "current_state_sha256",
        "source_sha256",
        "split_sha256",
        "atlas_sha256",
        "initialization_sha256",
    }
)


Role = Literal["best_miou", "best_pd"]


class PBDRV4ModelRegistryError(ValueError):
    """A model request or checkpoint violates the frozen V4 registry line."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4ModelRegistryError(message)


def _require_dataset(dataset_name: str) -> str:
    _require(
        type(dataset_name) is str and dataset_name in DATASETS,
        f"dataset_name must be one of {DATASETS}",
    )
    return dataset_name


def _require_role(role: str) -> Role:
    _require(
        type(role) is str and role in ROLES,
        f"role must be one of {ROLES}",
    )
    return role  # type: ignore[return-value]


def _require_stage(stage: str, *, expected: Stage | None = None) -> Stage:
    _require(
        type(stage) is str and stage in STAGES,
        f"stage must be one of {STAGES}",
    )
    ready = stage  # type: ignore[assignment]
    if expected is not None:
        _require(ready == expected, f"stage must be {expected!r}")
    return ready  # type: ignore[return-value]


def _require_sha256(value: Any, *, name: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    _require(
        all(type(key) is str for key in value),
        f"{name} keys must be strings",
    )
    return value


def _clone_tensor_state(
    value: Any,
    *,
    name: str,
) -> dict[str, torch.Tensor]:
    state = _require_mapping(value, name=name)
    _require(bool(state), f"{name} must not be empty")
    _require(
        all(isinstance(tensor, torch.Tensor) for tensor in state.values()),
        f"{name} values must be tensors",
    )
    cloned: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        assert isinstance(tensor, torch.Tensor)
        _require(bool(torch.isfinite(tensor).all()), f"{name}[{key!r}] is non-finite")
        cloned[key] = tensor.detach().cpu().clone()
    return cloned


def _pbdr_initialization_sha256(model: nn.Module) -> str:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(PBDR_V4_STATE_PREFIX)
    }
    _require(
        tuple(state) == PBDR_V4_STATE_KEYS,
        "PBDR-V4 initialization state keys differ",
    )
    return state_semantic_sha256(state)


def _normalize_parent_record(
    record: Mapping[str, Any],
    *,
    dataset_name: str,
    role: Role,
) -> dict[str, Any]:
    _require(record.get("dataset", dataset_name) == dataset_name, "Current record dataset differs")
    _require(record.get("checkpoint_role") == role, "Current record role differs")
    _require(type(record.get("path")) is str and bool(record["path"]), "Current record path is invalid")
    checkpoint_sha = _require_sha256(record.get("sha256"), name="Current checkpoint sha256")
    _require(type(record.get("bytes")) is int and record["bytes"] > 0, "Current record byte count is invalid")
    _require(record.get("state_key_count") == CURRENT_STATE_KEY_COUNT, "Current record state-key count differs")
    state_sha = _require_sha256(record.get("state_sha256"), name="Current record state_sha256")
    _require(type(record.get("epoch")) is int and record["epoch"] > 0, "Current record epoch is invalid")
    protocol_sha = record.get("protocol_sha256")
    if protocol_sha is not None:
        _require_sha256(protocol_sha, name="Current record protocol_sha256")
    return {
        "dataset": dataset_name,
        "checkpoint_role": role,
        "path": record["path"],
        "sha256": checkpoint_sha,
        "bytes": record["bytes"],
        "state_key_count": record["state_key_count"],
        "state_sha256": state_sha,
        "epoch": record["epoch"],
        "schema": record.get("schema"),
        "protocol_sha256": protocol_sha,
    }


def load_current_checkpoint(
    dataset_name: str,
    role: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    """Route to an audited Current registry without touching dataset files."""

    dataset = _require_dataset(dataset_name)
    ready_role = _require_role(role)
    if dataset == "NUAA-SIRST":
        payload, raw_state, raw_record = nuaa_registry.load_current_checkpoint(
            ready_role
        )
        registry_state_sha256 = nuaa_registry.tensor_mapping_sha256
    else:
        payload, raw_state, raw_record = cross_registry.load_current_checkpoint(
            dataset,
            ready_role,
        )
        registry_state_sha256 = cross_registry.tensor_mapping_sha256

    checkpoint = dict(_require_mapping(payload, name="Current checkpoint payload"))
    state = _clone_tensor_state(raw_state, name="Current state_dict")
    record = _normalize_parent_record(
        _require_mapping(raw_record, name="Current checkpoint record"),
        dataset_name=dataset,
        role=ready_role,
    )
    _require(checkpoint.get("dataset") == dataset, "Current checkpoint dataset differs")
    _require(checkpoint.get("checkpoint_role") == ready_role, "Current checkpoint role differs")
    _require(len(state) == CURRENT_STATE_KEY_COUNT, "Current state-key count differs")
    _require(
        not any(name.startswith(PBDR_V4_STATE_PREFIX) for name in state),
        "Current state unexpectedly contains PBDR-V4 keys",
    )
    _require(
        set(SURVIVAL_STATE_KEYS) <= set(state),
        "Current state lacks training-only Survival keys",
    )
    _require(
        registry_state_sha256(state) == record["state_sha256"],
        "Current state hash differs from audited registry record",
    )
    checkpoint["state_dict"] = state
    return checkpoint, state, record


def _new_initialized_training_model(
    *,
    role: Role,
    current_state: Mapping[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    model, raw_metadata = (
        build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
            role=role,
            seed=TRAINING_SEED,
        )
    )
    warm_start = warm_start_formal_pbdr_v4_from_current(model, current_state)
    validation = validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
        model,
        expected_role=role,
        require_identity_initialized_pbdr_v4=True,
        current_state=current_state,
        stage="stage1",
    )
    metadata = {
        "raw_builder_metadata": raw_metadata,
        "warm_start": warm_start,
        "validation": validation,
        "architecture_manifest": model.architecture_manifest(),
        "initialization_sha256": _pbdr_initialization_sha256(model),
        "initial_full_state_sha256": state_semantic_sha256(model.state_dict()),
    }
    return model, metadata


def build_stage1_training_model(
    dataset_name: str,
    role: str,
    stage: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the exact Current + 27-key V4 Stage-1 identity extension."""

    dataset = _require_dataset(dataset_name)
    ready_role = _require_role(role)
    ready_stage = _require_stage(stage, expected="stage1")
    _, current_state, parent = load_current_checkpoint(dataset, ready_role)
    current_sha = state_semantic_sha256(current_state)
    model, initialized = _new_initialized_training_model(
        role=ready_role,
        current_state=current_state,
    )
    initial_state_sha = state_semantic_sha256(model.state_dict())
    _require(
        initial_state_sha == initialized["initial_full_state_sha256"],
        "Stage-1 warm-start state changed before configuration",
    )
    mutable = configure_stage_training(model, ready_stage)
    modes = audit_training_modes(model, ready_stage)
    _require(modes["base_training"] is False, "Stage-1 Current graph must remain in eval mode")
    state_contract = audit_candidate_against_current(
        model,
        current_state=current_state,
        stage=ready_stage,
    )
    _require(
        state_semantic_sha256(model.state_dict()) == initial_state_sha,
        "Stage-1 mode configuration changed tensor state",
    )
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "role": ready_role,
        "parent_role": ready_role,
        "stage": ready_stage,
        "seed": TRAINING_SEED,
        "parent_checkpoint": parent,
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_state_sha256": parent["state_sha256"],
        "current_state_sha256": current_sha,
        "initialization_sha256": initialized["initialization_sha256"],
        "initial_full_state_sha256": initial_state_sha,
        "architecture_manifest": initialized["architecture_manifest"],
        "warm_start": initialized["warm_start"],
        "validation": initialized["validation"],
        "state_contract": state_contract,
        "training_modes": modes,
        "trainable_parameter_names": list(mutable),
        "exact_current_extension_key_count": len(PBDR_V4_STATE_KEYS),
        "exact_current_extension_keys": list(PBDR_V4_STATE_KEYS),
        "dataset_loader_imported": False,
        "dataset_index_accessed": False,
        "official_test_data_accessed": False,
        "training_started": False,
    }
    return model, metadata


def _require_parent_binding(
    observed: Any,
    *,
    expected: Mapping[str, Any],
) -> None:
    parent = _require_mapping(observed, name="candidate parent_checkpoint")
    missing = sorted(set(_PARENT_BINDING_KEYS) - set(parent))
    _require(not missing, f"candidate parent_checkpoint is incomplete; missing={missing}")
    for name in _PARENT_BINDING_KEYS:
        _require(
            parent.get(name) == expected.get(name),
            f"candidate parent_checkpoint {name} differs",
        )


def _validate_candidate_payload(
    checkpoint: Any,
    *,
    dataset_name: str,
    role: Role,
    stage: Stage,
    current_state: Mapping[str, torch.Tensor],
    parent_checkpoint: Mapping[str, Any],
    expected_architecture_manifest: Mapping[str, Any],
    expected_source_sha256: str,
    expected_split_sha256: str,
    expected_atlas_sha256: str,
    expected_initialization_sha256: str,
    expected_state_sha256: str | None,
) -> tuple[Mapping[str, Any], dict[str, torch.Tensor], str]:
    payload = _require_mapping(checkpoint, name="candidate checkpoint")
    missing = sorted(_CANDIDATE_REQUIRED_KEYS - set(payload))
    _require(not missing, f"candidate checkpoint is incomplete; missing={missing}")
    _require(
        payload.get("schema") == PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
        "candidate checkpoint schema differs",
    )
    _require(payload.get("dataset") == dataset_name, "candidate dataset differs")
    _require(payload.get("role") == role, "candidate role differs")
    _require(payload.get("parent_role") == role, "candidate parent_role differs")
    _require(payload.get("stage") == stage, "candidate stage differs")
    _require(
        payload.get("architecture_manifest") == expected_architecture_manifest,
        "candidate architecture manifest differs",
    )
    _require_parent_binding(payload.get("parent_checkpoint"), expected=parent_checkpoint)
    _require(
        payload.get("parent_checkpoint_sha256") == parent_checkpoint["sha256"],
        "candidate parent checkpoint hash differs",
    )
    _require(
        payload.get("parent_state_sha256") == parent_checkpoint["state_sha256"],
        "candidate parent state hash differs",
    )
    current_sha = state_semantic_sha256(current_state)
    _require(
        payload.get("current_state_sha256") == current_sha,
        "candidate Current semantic state hash differs",
    )

    expected_locks = {
        "source_sha256": _require_sha256(expected_source_sha256, name="expected_source_sha256"),
        "split_sha256": _require_sha256(expected_split_sha256, name="expected_split_sha256"),
        "atlas_sha256": _require_sha256(expected_atlas_sha256, name="expected_atlas_sha256"),
        "initialization_sha256": _require_sha256(
            expected_initialization_sha256,
            name="expected_initialization_sha256",
        ),
    }
    for name, expected in expected_locks.items():
        observed = _require_sha256(payload.get(name), name=f"candidate {name}")
        _require(observed == expected, f"candidate {name} differs")

    state = _clone_tensor_state(payload.get("state_dict"), name="candidate state_dict")
    _require(
        len(state) == TRAINING_STATE_KEY_COUNT,
        "candidate training state-key count differs",
    )
    _require(
        payload.get("state_key_count") == len(state),
        "candidate state_key_count differs",
    )
    state_sha = state_semantic_sha256(state)
    _require_sha256(payload.get("state_sha256"), name="candidate state_sha256")
    _require(payload.get("state_sha256") == state_sha, "candidate tensor-state hash differs")
    if expected_state_sha256 is not None:
        expected_state = _require_sha256(
            expected_state_sha256,
            name="expected_state_sha256",
        )
        _require(state_sha == expected_state, "candidate state differs from expected hash")
    return payload, state, state_sha


def build_stage2_training_model(
    stage1_checkpoint: Mapping[str, Any],
    *,
    dataset_name: str,
    role: str,
    stage: str,
    expected_source_sha256: str,
    expected_split_sha256: str,
    expected_atlas_sha256: str,
    expected_initialization_sha256: str,
    expected_stage1_state_sha256: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly load a complete Stage-1 payload, then enable Stage-2 params."""

    dataset = _require_dataset(dataset_name)
    ready_role = _require_role(role)
    ready_stage = _require_stage(stage, expected="stage2")
    _, current_state, parent = load_current_checkpoint(dataset, ready_role)
    model, initialized = _new_initialized_training_model(
        role=ready_role,
        current_state=current_state,
    )
    payload, stage1_state, stage1_state_sha = _validate_candidate_payload(
        stage1_checkpoint,
        dataset_name=dataset,
        role=ready_role,
        stage="stage1",
        current_state=current_state,
        parent_checkpoint=parent,
        expected_architecture_manifest=initialized["architecture_manifest"],
        expected_source_sha256=expected_source_sha256,
        expected_split_sha256=expected_split_sha256,
        expected_atlas_sha256=expected_atlas_sha256,
        expected_initialization_sha256=expected_initialization_sha256,
        expected_state_sha256=expected_stage1_state_sha256,
    )
    _require(
        payload["initialization_sha256"] == initialized["initialization_sha256"],
        "Stage-1 initialization SHA-256 differs from a fresh identity graph",
    )
    incompatible = model.load_state_dict(stage1_state, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "Stage-1 strict state load failed",
    )
    stage1_validation = validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
        model,
        expected_role=ready_role,
        current_state=current_state,
        stage="stage1",
    )
    stage1_contract = audit_candidate_against_current(
        model,
        current_state=current_state,
        stage="stage1",
    )
    _require(
        state_semantic_sha256(model.state_dict()) == stage1_state_sha,
        "strict Stage-1 load changed tensor state",
    )
    mutable = configure_stage_training(model, ready_stage)
    modes = audit_training_modes(model, ready_stage)
    _require(modes["base_training"] is False, "Stage-2 Current graph must remain in eval mode")
    _require(
        state_semantic_sha256(model.state_dict()) == stage1_state_sha,
        "Stage-2 mode configuration changed initialization state",
    )
    stage2_contract = audit_candidate_against_current(
        model,
        current_state=current_state,
        stage=ready_stage,
    )
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "role": ready_role,
        "parent_role": ready_role,
        "stage": ready_stage,
        "seed": TRAINING_SEED,
        "parent_checkpoint": parent,
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_state_sha256": parent["state_sha256"],
        "current_state_sha256": state_semantic_sha256(current_state),
        "source_sha256": payload["source_sha256"],
        "split_sha256": payload["split_sha256"],
        "atlas_sha256": payload["atlas_sha256"],
        "initialization_sha256": initialized["initialization_sha256"],
        "stage1_state_sha256": stage1_state_sha,
        "stage2_initial_state_sha256": stage1_state_sha,
        "architecture_manifest": initialized["architecture_manifest"],
        "strict_stage1_state_load": True,
        "stage1_validation": stage1_validation,
        "stage1_state_contract": stage1_contract,
        "stage2_state_contract": stage2_contract,
        "training_modes": modes,
        "trainable_parameter_names": list(mutable),
        "dataset_loader_imported": False,
        "dataset_index_accessed": False,
        "official_test_data_accessed": False,
        "training_started": False,
    }
    return model, metadata


def _strip_current_survival_state(
    current_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _require(
        len(current_state) == CURRENT_STATE_KEY_COUNT,
        "Current survival state-key count differs",
    )
    _require(
        set(SURVIVAL_STATE_KEYS) <= set(current_state),
        "Current survival state lacks training-only keys",
    )
    stripped = {
        name: tensor.detach().cpu().clone()
        for name, tensor in current_state.items()
        if name not in set(SURVIVAL_STATE_KEYS)
    }
    _require(
        len(stripped) == FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
        "Current inference state-key count differs",
    )
    return stripped


def build_frozen_current_reference_model(
    dataset_name: str,
    role: str,
    stage: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build an independent, immutable Current-logit reference graph."""

    dataset = _require_dataset(dataset_name)
    ready_role = _require_role(role)
    ready_stage = _require_stage(stage)
    _, current_state, parent = load_current_checkpoint(dataset, ready_role)
    current_inference = _strip_current_survival_state(current_state)
    model, raw_metadata = build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
        role=ready_role,
        seed=TRAINING_SEED,
    )
    warm_start = warm_start_formal_pbdr_v4_from_current(model, current_inference)
    validation = validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
        model,
        expected_role=ready_role,
        require_identity_initialized_pbdr_v4=True,
        current_state=current_inference,
        stage="stage1",
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.mode = "test"
    _require(not model.training, "frozen Current reference must be in eval mode")
    _require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "frozen Current reference exposes trainable parameters",
    )
    exact_base = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith(PBDR_V4_STATE_PREFIX)
    }
    _require(
        all(torch.equal(exact_base[name], current_inference[name]) for name in current_inference),
        "frozen Current reference base state differs bitwise",
    )
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "role": ready_role,
        "parent_role": ready_role,
        "stage": ready_stage,
        "seed": TRAINING_SEED,
        "reference_kind": "independent_frozen_current",
        "candidate_object_reused": False,
        "candidate_state_reused": False,
        "base_logits_source": "audited_current_checkpoint",
        "base_logits_are_current": True,
        "parent_checkpoint": parent,
        "current_state_sha256": state_semantic_sha256(current_state),
        "current_inference_state_sha256": state_semantic_sha256(current_inference),
        "warm_start": warm_start,
        "validation": validation,
        "raw_builder_metadata": raw_metadata,
        "all_parameters_frozen": True,
        "base_training": False,
        "pbdr_v4_training": False,
        "dataset_loader_imported": False,
        "dataset_index_accessed": False,
        "official_test_data_accessed": False,
    }
    return model, metadata


def build_candidate_checkpoint_payload(
    model: nn.Module,
    *,
    dataset_name: str,
    role: str,
    stage: str,
    source_sha256: str,
    split_sha256: str,
    atlas_sha256: str,
    initialization_sha256: str,
) -> dict[str, Any]:
    """Create the complete registry/integration payload for one Candidate."""

    dataset = _require_dataset(dataset_name)
    ready_role = _require_role(role)
    ready_stage = _require_stage(stage)
    source = _require_sha256(source_sha256, name="source_sha256")
    split = _require_sha256(split_sha256, name="split_sha256")
    atlas = _require_sha256(atlas_sha256, name="atlas_sha256")
    initialization = _require_sha256(
        initialization_sha256,
        name="initialization_sha256",
    )
    _, current_state, parent = load_current_checkpoint(dataset, ready_role)
    validation = validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
        model,
        expected_role=ready_role,
        current_state=current_state,
        stage=ready_stage,
    )
    state_contract = audit_candidate_against_current(
        model,
        current_state=current_state,
        stage=ready_stage,
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    _require(len(state) == TRAINING_STATE_KEY_COUNT, "Candidate training state-key count differs")
    state_sha = state_semantic_sha256(state)
    return {
        "schema": PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
        "dataset": dataset,
        "role": ready_role,
        "parent_role": ready_role,
        "stage": ready_stage,
        "architecture_manifest": model.architecture_manifest(),
        "state_dict": state,
        "state_key_count": len(state),
        "state_sha256": state_sha,
        "parent_checkpoint": dict(parent),
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_state_sha256": parent["state_sha256"],
        "current_state_sha256": state_semantic_sha256(current_state),
        "source_sha256": source,
        "split_sha256": split,
        "atlas_sha256": atlas,
        "initialization_sha256": initialization,
        "model_validation": validation,
        "state_contract": state_contract,
        "official_test_data_accessed": False,
    }


def build_candidate_inference_model(
    checkpoint: Mapping[str, Any],
    *,
    dataset_name: str,
    role: str,
    stage: str,
    expected_source_sha256: str,
    expected_split_sha256: str,
    expected_atlas_sha256: str,
    expected_initialization_sha256: str,
    expected_state_sha256: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Validate a complete Candidate payload and export its inference graph."""

    dataset = _require_dataset(dataset_name)
    ready_role = _require_role(role)
    ready_stage = _require_stage(stage)
    _, current_state, parent = load_current_checkpoint(dataset, ready_role)
    initialized, init_metadata = _new_initialized_training_model(
        role=ready_role,
        current_state=current_state,
    )
    try:
        _, _, state_sha = _validate_candidate_payload(
            checkpoint,
            dataset_name=dataset,
            role=ready_role,
            stage=ready_stage,
            current_state=current_state,
            parent_checkpoint=parent,
            expected_architecture_manifest=init_metadata["architecture_manifest"],
            expected_source_sha256=expected_source_sha256,
            expected_split_sha256=expected_split_sha256,
            expected_atlas_sha256=expected_atlas_sha256,
            expected_initialization_sha256=expected_initialization_sha256,
            expected_state_sha256=expected_state_sha256,
        )
        _require(
            checkpoint["initialization_sha256"] == init_metadata["initialization_sha256"],
            "Candidate initialization SHA-256 differs from a fresh identity graph",
        )
    finally:
        del initialized
    model, integration_metadata = (
        build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint(
            checkpoint,
            expected_role=ready_role,
            expected_stage=ready_stage,
            current_state=current_state,
            seed=TRAINING_SEED,
        )
    )
    metadata = {
        "schema": SCHEMA,
        "dataset": dataset,
        "role": ready_role,
        "parent_role": ready_role,
        "stage": ready_stage,
        "seed": TRAINING_SEED,
        "parent_checkpoint": parent,
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_state_sha256": parent["state_sha256"],
        "current_state_sha256": state_semantic_sha256(current_state),
        "candidate_state_sha256": state_sha,
        "initialization_sha256": init_metadata["initialization_sha256"],
        "strict_complete_payload": True,
        "integration_export": integration_metadata,
        "dataset_loader_imported": False,
        "dataset_index_accessed": False,
        "official_test_data_accessed": False,
    }
    return model, metadata


__all__ = [
    "CURRENT_STATE_KEY_COUNT",
    "DATASETS",
    "INFERENCE_STATE_KEY_COUNT",
    "PBDRV4ModelRegistryError",
    "ROLES",
    "SCHEMA",
    "STAGES",
    "TRAINING_SEED",
    "TRAINING_STATE_KEY_COUNT",
    "build_candidate_checkpoint_payload",
    "build_candidate_inference_model",
    "build_frozen_current_reference_model",
    "build_stage1_training_model",
    "build_stage2_training_model",
    "load_current_checkpoint",
    "state_semantic_sha256",
]
