"""Strict current-Final -> NER-L4-TPR identity-audit bridge.

This module is deliberately outside the formal trainer.  Formal NER-L4-TPR
training is fresh seed-42 scratch training.  The functions here support only
zero-gate engineering identity checks and zero-training checkpoint screening.

Checkpoint selection is never inferred from a directory.  A caller must bind
all six ``dataset x {best_miou,best_pd}`` inputs in an immutable manifest and
must supply the expected manifest SHA-256.  The selected entry itself binds an
explicit checkpoint path, role, epoch, and file SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from experiments.four_dataset_models_seed42_v1 import state_dict_sha256
from experiments.three_dataset_v2_protocol import DATASETS
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr import (
    FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT,
    L4_TPR_STATE_KEYS,
    PRODUCTION_V4_QFG_V2_CROA_L4_TPR_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_PARAMETERS,
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet,
    build_formal_v4_qfg_v2_croa_l4_tpr_inference_model,
    build_formal_v4_qfg_v2_croa_l4_tpr_survival_model,
    validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model,
    validate_formal_v4_qfg_v2_croa_l4_tpr_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "sctransnet_ner_l4_tpr_identity_inputs/v1"
PARENT_CHECKPOINT_SCHEMA = "sctransnet_three_dataset_tss_off_seed42_v1/v1"
TRAINING_SEED = 42
CHECKPOINT_ROLES = ("best_miou", "best_pd")
PRIMARY_SCREENING_ROLE = "best_miou"
SUPPLEMENTAL_SCREENING_ROLE = "best_pd"
NEW_STATE_KEY = "ner_l4_tpr.reallocation_logits"
NEW_PARAMETER_SHAPE = (1, 256, 1, 1)


class NERL4TPRMigrationError(ValueError):
    """An identity-audit binding or state violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NERL4TPRMigrationError(message)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_repo_file(value: Any, *, name: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{name} must be a path")
    supplied = Path(value)
    if not supplied.is_absolute():
        supplied = REPO_ROOT / supplied
    ready = supplied.resolve(strict=True)
    _require(ready.is_file(), f"{name} is not a file")
    try:
        ready.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise NERL4TPRMigrationError(
            f"{name} must be inside the repository"
        ) from exc
    return ready


def load_manifest_binding(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    dataset: str,
    checkpoint_role: str,
) -> dict[str, Any]:
    """Return one explicitly bound Final checkpoint after full matrix checks."""

    _require(dataset in DATASETS, f"dataset must be one of {DATASETS}")
    _require(
        checkpoint_role in CHECKPOINT_ROLES,
        f"checkpoint_role must be one of {CHECKPOINT_ROLES}",
    )
    _require(
        _is_sha256(expected_manifest_sha256),
        "expected_manifest_sha256 must be lowercase SHA-256",
    )
    supplied = Path(manifest_path)
    _require(not supplied.is_symlink(), "manifest path cannot be a symlink")
    ready = supplied.resolve(strict=True)
    _require(ready.is_file(), "manifest path is not a file")
    observed_manifest_sha = file_sha256(ready)
    _require(
        observed_manifest_sha == expected_manifest_sha256,
        "manifest SHA-256 differs from the caller binding",
    )
    payload = json.loads(ready.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "manifest must be a JSON object")
    _require(payload.get("schema") == MANIFEST_SCHEMA, "manifest schema differs")
    _require(payload.get("status") == "frozen", "manifest is not frozen")
    _require(payload.get("seed") == TRAINING_SEED, "manifest seed differs")
    _require(
        payload.get("dataset_order") == list(DATASETS),
        "manifest dataset order differs",
    )
    _require(
        payload.get("checkpoint_role_order") == list(CHECKPOINT_ROLES),
        "manifest checkpoint-role order differs",
    )
    _require(
        payload.get("primary_screening_role") == PRIMARY_SCREENING_ROLE,
        "manifest primary screening role differs",
    )
    _require(
        payload.get("supplemental_screening_role")
        == SUPPLEMENTAL_SCREENING_ROLE,
        "manifest supplemental screening role differs",
    )
    data_protocol = payload.get("data_protocol_manifest")
    _require(
        isinstance(data_protocol, Mapping),
        "manifest data protocol binding differs",
    )
    _require(
        _is_sha256(data_protocol.get("sha256")),
        "manifest data protocol SHA-256 differs",
    )
    data_protocol_path = _explicit_repo_file(
        data_protocol.get("path"),
        name="manifest data_protocol_manifest.path",
    )
    _require(
        file_sha256(data_protocol_path) == data_protocol["sha256"],
        "manifest data protocol bytes changed",
    )

    entries = payload.get("entries")
    _require(isinstance(entries, list), "manifest entries must be a list")
    _require(
        len(entries) == len(DATASETS) * len(CHECKPOINT_ROLES),
        "manifest must bind six checkpoints",
    )
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in entries:
        _require(isinstance(raw, dict), "manifest entry must be an object")
        entry_dataset = raw.get("dataset")
        entry_role = raw.get("checkpoint_role")
        _require(entry_dataset in DATASETS, "manifest entry dataset differs")
        _require(entry_role in CHECKPOINT_ROLES, "manifest entry role differs")
        key = (str(entry_dataset), str(entry_role))
        _require(key not in indexed, f"duplicate manifest entry: {key}")
        _require(
            type(raw.get("epoch")) is int and int(raw["epoch"]) > 0,
            "manifest entry epoch differs",
        )
        _require(
            _is_sha256(raw.get("checkpoint_sha256")),
            "manifest checkpoint SHA-256 differs",
        )
        # Resolve the exact path written in the entry.  Never synthesize a
        # filename from dataset, directory, or checkpoint role.
        checkpoint_path = _explicit_repo_file(
            raw.get("checkpoint_path"),
            name="manifest checkpoint_path",
        )
        observed_checkpoint_sha = file_sha256(checkpoint_path)
        _require(
            observed_checkpoint_sha == raw["checkpoint_sha256"],
            f"manifest checkpoint SHA-256 changed: {key}",
        )
        _require(
            _is_sha256(raw.get("reference_evaluation_sha256")),
            "manifest reference-evaluation SHA-256 differs",
        )
        reference_path = _explicit_repo_file(
            raw.get("reference_evaluation_path"),
            name="manifest reference_evaluation_path",
        )
        _require(
            file_sha256(reference_path)
            == raw["reference_evaluation_sha256"],
            f"manifest reference evaluation changed: {key}",
        )
        normalized = dict(raw)
        normalized["checkpoint_path"] = str(checkpoint_path)
        normalized["reference_evaluation_path"] = str(reference_path)
        indexed[key] = normalized

    expected_keys = {
        (candidate_dataset, role)
        for candidate_dataset in DATASETS
        for role in CHECKPOINT_ROLES
    }
    _require(set(indexed) == expected_keys, "manifest checkpoint matrix differs")
    selected = dict(indexed[(dataset, checkpoint_role)])
    selected["manifest_path"] = str(ready)
    selected["manifest_sha256"] = observed_manifest_sha
    selected["data_protocol_manifest"] = {
        "path": str(data_protocol_path),
        "sha256": str(data_protocol["sha256"]),
    }
    return selected


def load_bound_parent_checkpoint(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Load one already-validated explicit binding and check payload identity."""

    _require(isinstance(binding, Mapping), "binding must be a mapping")
    dataset = binding.get("dataset")
    role = binding.get("checkpoint_role")
    _require(dataset in DATASETS, "bound dataset differs")
    _require(role in CHECKPOINT_ROLES, "bound checkpoint role differs")
    _require(
        _is_sha256(binding.get("manifest_sha256")),
        "bound manifest SHA-256 differs",
    )
    canonical = load_manifest_binding(
        Path(str(binding.get("manifest_path", ""))),
        expected_manifest_sha256=str(binding["manifest_sha256"]),
        dataset=str(dataset),
        checkpoint_role=str(role),
    )
    for field in ("checkpoint_path", "checkpoint_sha256", "epoch"):
        _require(
            binding.get(field) == canonical[field],
            f"binding {field} differs from its manifest entry",
        )
    _require(
        _is_sha256(binding.get("checkpoint_sha256")),
        "bound checkpoint SHA-256 differs",
    )
    checkpoint_path = _explicit_repo_file(
        binding.get("checkpoint_path"),
        name="bound checkpoint_path",
    )
    _require(
        file_sha256(checkpoint_path) == binding["checkpoint_sha256"],
        "bound checkpoint bytes changed",
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "bound checkpoint must be a dictionary")
    for field, expected in (
        ("schema", PARENT_CHECKPOINT_SCHEMA),
        ("dataset", dataset),
        ("checkpoint_role", role),
        ("epoch", binding.get("epoch")),
        ("method", "final"),
        ("seed", TRAINING_SEED),
        ("requested_tss_weight", 0.0),
        ("tss_enabled", False),
        ("test_selected", True),
    ):
        _require(payload.get(field) == expected, f"checkpoint {field} differs")
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping), "checkpoint state_dict is missing")
    _require(
        len(state) == FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
        "current Final training state-key count differs",
    )
    _require(
        all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ),
        "checkpoint state_dict must map string keys to tensors",
    )
    survival = {key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)}
    _require(survival == set(SURVIVAL_STATE_KEYS), "checkpoint TSS key set differs")
    ready_payload = dict(payload)
    ready_payload["state_dict"] = dict(state)
    ready_payload["checkpoint_path"] = str(checkpoint_path)
    ready_payload["checkpoint_sha256"] = str(binding["checkpoint_sha256"])
    return ready_payload


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _strict_one_key_extension(
    model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    target_state = model.state_dict()
    _require(
        set(L4_TPR_STATE_KEYS) == {NEW_STATE_KEY},
        "new state-key contract differs",
    )
    _require(NEW_STATE_KEY in target_state, "candidate lacks the new state key")
    _require(
        set(target_state) == set(source_state) | {NEW_STATE_KEY},
        "candidate is not an exact one-key extension",
    )
    full_state: dict[str, torch.Tensor] = {}
    for key, target_value in target_state.items():
        if key == NEW_STATE_KEY:
            _require(
                tuple(target_value.shape) == NEW_PARAMETER_SHAPE,
                "new parameter shape differs",
            )
            _require(
                target_value.is_floating_point(),
                "new parameter dtype is not floating",
            )
            _require(
                bool(torch.isfinite(target_value).all()),
                "new parameter is non-finite",
            )
            _require(
                int(torch.count_nonzero(target_value)) == 0,
                "new parameter is not exact zero",
            )
            full_state[key] = target_value.detach().clone()
            continue
        source_value = source_state[key]
        _require(
            tuple(source_value.shape) == tuple(target_value.shape),
            f"shared state shape differs: {key}",
        )
        _require(
            source_value.dtype == target_value.dtype,
            f"shared state dtype differs: {key}",
        )
        full_state[key] = source_value.detach().clone()

    # Unlike a permissive extension load, the assembled candidate state is
    # complete and must pass an actual strict=True load.
    incompatible = model.load_state_dict(full_state, strict=True)
    _require(not incompatible.missing_keys, "strict extension returned missing keys")
    _require(
        not incompatible.unexpected_keys,
        "strict extension returned unexpected keys",
    )
    loaded = model.state_dict()
    for key, source_value in source_state.items():
        _require(
            torch.equal(loaded[key], source_value),
            f"strict extension changed shared state: {key}",
        )
    _require(
        int(torch.count_nonzero(loaded[NEW_STATE_KEY])) == 0,
        "strict extension gate is not zero",
    )
    parameter = dict(model.named_parameters()).get(NEW_STATE_KEY)
    _require(isinstance(parameter, nn.Parameter), "new state is not an nn.Parameter")
    _require(parameter.requires_grad, "new parameter is not trainable")
    return {
        "load_mode": "complete_candidate_state_strict_true",
        "shared_state_key_count": len(source_state),
        "new_state_keys": [NEW_STATE_KEY],
        "new_state_key_count": 1,
        "new_parameter_elements": int(parameter.numel()),
        "new_parameter_shape": list(parameter.shape),
        "new_parameter_exact_zero": True,
        "formal_training_warm_start_authorized": False,
    }


def build_zero_extension_training_model(
    binding: Mapping[str, Any],
) -> tuple[TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet, dict[str, Any]]:
    """Build a 569-key identity-audit graph from one bound Final checkpoint."""

    checkpoint = load_bound_parent_checkpoint(binding)
    model, construction = build_formal_v4_qfg_v2_croa_l4_tpr_survival_model()
    report = _strict_one_key_extension(model, checkpoint["state_dict"])
    validation = validate_formal_v4_qfg_v2_croa_l4_tpr_survival_model(model)
    _require(
        len(model.state_dict())
        == FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT,
        "candidate training state-key count differs",
    )
    _require(
        _parameter_count(model) == PRODUCTION_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_PARAMETERS,
        "candidate training parameter count differs",
    )
    metadata = {
        "purpose": "zero_gate_engineering_identity_audit_only",
        "dataset": checkpoint["dataset"],
        "checkpoint_role": checkpoint["checkpoint_role"],
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "parent_state_dict_sha256": state_dict_sha256(checkpoint["state_dict"]),
        "candidate_state_dict_sha256": state_dict_sha256(model.state_dict()),
        "formal_training_construction": "fresh_seed42_scratch",
        "formal_training_uses_this_bridge": False,
        "parent_optimizer_loaded": False,
        "parent_rng_loaded": False,
        "migration": report,
        "construction": construction,
        "validation": validation,
    }
    return model, metadata


def strip_l4_tpr_tss_for_inference_state_dict(
    training_state: Mapping[str, torch.Tensor],
    *,
    to_cpu: bool = False,
) -> dict[str, torch.Tensor]:
    """Remove exactly four TSS keys from a trained 569-key candidate state."""

    _require(isinstance(training_state, Mapping), "training_state must be a mapping")
    _require(
        len(training_state) == FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT,
        "candidate training state-key count differs",
    )
    _require(NEW_STATE_KEY in training_state, "candidate training state lacks L4-TPR")
    survival = {key for key in training_state if key.startswith(SURVIVAL_STATE_PREFIX)}
    _require(survival == set(SURVIVAL_STATE_KEYS), "candidate training TSS keys differ")
    inference: dict[str, torch.Tensor] = {}
    for key, value in training_state.items():
        _require(
            isinstance(key, str) and isinstance(value, torch.Tensor),
            "candidate state must map string keys to tensors",
        )
        if key in survival:
            continue
        clone = value.detach().clone()
        inference[key] = clone.cpu() if to_cpu else clone
    _require(
        len(inference) == FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT,
        "candidate inference state-key count differs",
    )
    _require(NEW_STATE_KEY in inference, "inference state lost L4-TPR")
    _require(
        not any(key.startswith(SURVIVAL_STATE_PREFIX) for key in inference),
        "candidate inference state retains TSS",
    )
    return inference


def build_l4_tpr_inference_model_from_training_state_dict(
    training_state: Mapping[str, torch.Tensor],
) -> tuple[TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet, dict[str, Any]]:
    """Strict-load the 565-key deployment graph from a 569-key train state."""

    inference_state = strip_l4_tpr_tss_for_inference_state_dict(training_state)
    model, construction = build_formal_v4_qfg_v2_croa_l4_tpr_inference_model()
    _require(
        set(model.state_dict()) == set(inference_state),
        "inference graph key set differs",
    )
    incompatible = model.load_state_dict(inference_state, strict=True)
    _require(
        not incompatible.missing_keys,
        "strict inference load returned missing keys",
    )
    _require(
        not incompatible.unexpected_keys,
        "strict inference load returned unexpected keys",
    )
    validation = validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model(model)
    _require(
        len(model.state_dict())
        == FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT,
        "candidate inference state-key count differs",
    )
    _require(
        _parameter_count(model)
        == PRODUCTION_V4_QFG_V2_CROA_L4_TPR_INFERENCE_PARAMETERS,
        "candidate inference parameter count differs",
    )
    model.eval()
    model.mode = "test"
    return model, {
        "purpose": "strict_l4_tpr_inference_export",
        "training_state_dict_sha256": state_dict_sha256(training_state),
        "inference_state_dict_sha256": state_dict_sha256(model.state_dict()),
        "removed_state_keys": list(SURVIVAL_STATE_KEYS),
        "removed_state_key_count": len(SURVIVAL_STATE_KEYS),
        "retained_l4_tpr_state_keys": list(L4_TPR_STATE_KEYS),
        "strict_load": True,
        "construction": construction,
        "validation": validation,
    }


def build_zero_extension_inference_model(
    binding: Mapping[str, Any],
) -> tuple[TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet, dict[str, Any]]:
    """Build a 565-key zero-training screening graph from a bound checkpoint."""

    checkpoint = load_bound_parent_checkpoint(binding)
    parent_training_state = checkpoint["state_dict"]
    parent_inference_state = {
        key: value
        for key, value in parent_training_state.items()
        if key not in set(SURVIVAL_STATE_KEYS)
    }
    _require(
        len(parent_inference_state) == FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
        "parent inference state-key count differs",
    )
    model, construction = build_formal_v4_qfg_v2_croa_l4_tpr_inference_model()
    report = _strict_one_key_extension(model, parent_inference_state)
    validation = validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model(
        model,
        require_zero_initialized_l4_tpr=True,
    )
    model.eval()
    model.mode = "test"
    return model, {
        "purpose": "zero_training_checkpoint_screening_only",
        "dataset": checkpoint["dataset"],
        "checkpoint_role": checkpoint["checkpoint_role"],
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "parent_training_state_dict_sha256": state_dict_sha256(parent_training_state),
        "candidate_inference_state_dict_sha256": state_dict_sha256(model.state_dict()),
        "formal_training_uses_this_bridge": False,
        "migration": report,
        "construction": construction,
        "validation": validation,
    }


__all__ = [
    "CHECKPOINT_ROLES",
    "MANIFEST_SCHEMA",
    "NEW_PARAMETER_SHAPE",
    "NEW_STATE_KEY",
    "PRIMARY_SCREENING_ROLE",
    "SUPPLEMENTAL_SCREENING_ROLE",
    "build_l4_tpr_inference_model_from_training_state_dict",
    "build_zero_extension_inference_model",
    "build_zero_extension_training_model",
    "file_sha256",
    "load_bound_parent_checkpoint",
    "load_manifest_binding",
    "strip_l4_tpr_tss_for_inference_state_dict",
]
