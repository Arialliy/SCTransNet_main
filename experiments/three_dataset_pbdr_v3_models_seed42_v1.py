#!/usr/bin/env python3
"""Source-locked PBDR-V3 model registry for NUAA Stage 1.

The registry has deliberately narrow scope.  It constructs the formal
574-state-key training graph, installs one immutable 568-state-key Current
checkpoint, proves bitwise equality for every inherited tensor, and then
freezes everything except ``pbdr_v3.*``.  The inference builder performs the
same checks while removing only the four training-only Survival tensors.
"""

from __future__ import annotations

import hashlib
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

from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3 import (  # noqa: E402
    FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_STATE_KEY_COUNT,
    PBDR_V3_INTEGRATION_VERSION,
    PBDR_V3_STATE_KEYS,
    PBDR_V3_STATE_PREFIX,
    SURVIVAL_STATE_KEYS,
    build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model,
    build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model,
)


SCHEMA = "sctransnet_three_dataset_pbdr_v3_models_seed42_v1/v1"
DATASET = "NUAA-SIRST"
TRAINING_SEED = 42
PARENT_ROLES = ("best_miou", "best_pd")
CURRENT_STATE_KEY_COUNT = 568
TRAINING_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_STATE_KEY_COUNT
)
INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT
)
CURRENT_ROOT = REPO_ROOT / (
    "results/three_dataset_tss_off_seed42_v1/runs/NUAA-SIRST/"
    "final_tss_off/seed_42/checkpoints"
)
DEFAULT_PARENT_CHECKPOINTS = {
    role: CURRENT_ROOT / f"{role}.pth.tar" for role in PARENT_ROLES
}

# These are the executable dependency boundary, not a broad hash of every file
# in the repository.  A rolling resume refuses to continue when any entry
# changes.
RUNTIME_DEPENDENCY_RELATIVE_PATHS = (
    "experiments/three_dataset_pbdr_v3_models_seed42_v1.py",
    "experiments/train_nuaa_pbdr_v3_stage1_v1.py",
    "experiments/evaluate_nuaa_pbdr_v3_stage1_v1.py",
    "experiments/launch_nuaa_pbdr_v3_stage1_v1.py",
    "experiments/pbdr_v3_loss.py",
    "experiments/pbdr_v3_non_regression_gate.py",
    "experiments/PBDR_V3_PROTOCOL.md",
    "experiments/four_dataset_evaluation_protocol_v1.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/paper_three_dataset_v2.py",
    "experiments/train_tpd_pilot.py",
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


class PBDRV3ModelProtocolError(ValueError):
    """The candidate graph or parent checkpoint violates the frozen line."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV3ModelProtocolError(message)


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, dtype, shape, and exact dense tensor bytes in key order."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state[{name!r}] is not a tensor")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        # ``view(dtype)`` rejects zero-dimensional multi-byte tensors; the
        # one-dimensional byte view is equivalent and preserves exact bytes.
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def runtime_source_paths() -> dict[str, Path]:
    return {
        relative: (REPO_ROOT / relative).resolve(strict=True)
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


def _require_role(role: str) -> str:
    _require(role in PARENT_ROLES, f"parent_role must be one of {PARENT_ROLES}")
    return role


def _checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise PBDRV3ModelProtocolError("checkpoint payload must be a mapping")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise PBDRV3ModelProtocolError("checkpoint lacks state_dict")
    if not all(isinstance(key, str) for key in state):
        raise PBDRV3ModelProtocolError("checkpoint state keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise PBDRV3ModelProtocolError("checkpoint state values must be tensors")
    return state


def load_current_checkpoint(
    parent_role: str,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], dict[str, Any]]:
    """Load and validate one immutable Current training checkpoint."""

    role = _require_role(parent_role)
    path = Path(checkpoint_path or DEFAULT_PARENT_CHECKPOINTS[role]).resolve(
        strict=True
    )
    if path.is_symlink():
        raise PBDRV3ModelProtocolError("parent checkpoint must not be a symlink")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = _checkpoint_state(payload)
    expected = {
        "dataset": DATASET,
        "method": "final",
        "seed": TRAINING_SEED,
        "checkpoint_role": role,
        "tss_enabled": False,
    }
    for name, value in expected.items():
        _require(
            payload.get(name) == value,
            f"Current checkpoint {name} differs: {payload.get(name)!r}",
        )
    _require(
        len(state) == CURRENT_STATE_KEY_COUNT,
        f"Current checkpoint must have {CURRENT_STATE_KEY_COUNT} keys",
    )
    _require(
        not any(name.startswith(PBDR_V3_STATE_PREFIX) for name in state),
        "Current checkpoint unexpectedly contains PBDR-V3 state",
    )
    for name, tensor in state.items():
        _require(
            bool(torch.isfinite(tensor).all()),
            f"Current checkpoint tensor {name!r} is non-finite",
        )
    record = {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "state_key_count": len(state),
        "state_sha256": tensor_mapping_sha256(state),
        "checkpoint_role": role,
        "epoch": int(payload["epoch"]),
        "schema": payload.get("schema"),
        "protocol_sha256": payload.get("protocol_sha256"),
    }
    return dict(payload), state, record


def _base_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith(PBDR_V3_STATE_PREFIX)
    }


def batchnorm_buffer_state(model: nn.Module) -> dict[str, torch.Tensor]:
    buffers: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if module_name.startswith("pbdr_v3"):
            continue
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for local_name, value in module.named_buffers(recurse=False):
                if value is not None:
                    name = (
                        f"{module_name}.{local_name}"
                        if module_name
                        else local_name
                    )
                    buffers[name] = value
    _require(bool(buffers), "formal Current graph exposes no BatchNorm buffers")
    return buffers


def base_state_sha256(model: nn.Module) -> str:
    return tensor_mapping_sha256(_base_state(model))


def batchnorm_buffer_sha256(model: nn.Module) -> str:
    return tensor_mapping_sha256(batchnorm_buffer_state(model))


def configure_stage1(model: nn.Module) -> dict[str, Any]:
    """Freeze Current behavior and expose only calibrator parameters."""

    router = getattr(model, "pbdr_v3", None)
    _require(isinstance(router, nn.Module), "model has no pbdr_v3 module")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in router.parameters():
        parameter.requires_grad_(True)

    # Freezes both BatchNorm running statistics and any stochastic base layer.
    model.eval()
    router.train()
    return audit_stage1(model)


def audit_stage1(model: nn.Module) -> dict[str, Any]:
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    expected = tuple(
        name for name, _ in model.named_parameters() if name.startswith(PBDR_V3_STATE_PREFIX)
    )
    _require(trainable == expected, "Stage 1 trainable parameter set differs")
    _require(bool(trainable), "PBDR-V3 has no trainable parameters")
    _require(model.training is False, "Stage 1 base model must remain in eval mode")
    _require(model.pbdr_v3.training is True, "PBDR-V3 must remain in train mode")
    bad_modules = [
        name
        for name, module in model.named_modules()
        if not name.startswith("pbdr_v3") and module.training
    ]
    _require(not bad_modules, f"base modules remain in training mode: {bad_modules}")
    bad_bn = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
    ]
    _require(not bad_bn, f"BatchNorm modules remain in training mode: {bad_bn}")
    return {
        "trainable_parameter_names": list(trainable),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.pbdr_v3.parameters()
            if parameter.requires_grad
        ),
        "base_training": model.training,
        "pbdr_v3_training": model.pbdr_v3.training,
        "base_state_sha256": base_state_sha256(model),
        "batchnorm_buffer_sha256": batchnorm_buffer_sha256(model),
        "batchnorm_buffer_names": sorted(batchnorm_buffer_state(model)),
    }


def build_stage1_training_model(
    parent_role: str,
    *,
    parent_checkpoint: Path | None = None,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build, strictly warm-start, freeze, and audit the training graph."""

    _require(seed == TRAINING_SEED and type(seed) is int, "formal seed is 42")
    payload, current_state, parent = load_current_checkpoint(
        parent_role, parent_checkpoint
    )
    with _preserve_process_rng():
        model, raw = build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(seed)
    candidate_keys = set(model.state_dict())
    _require(
        candidate_keys - set(PBDR_V3_STATE_KEYS) == set(current_state),
        "Current and PBDR-V3 inherited key sets differ",
    )
    incompatible = model.load_state_dict(current_state, strict=False)
    _require(
        tuple(incompatible.missing_keys) == PBDR_V3_STATE_KEYS,
        "Current warm-start missing-key set is not exactly PBDR_V3_STATE_KEYS",
    )
    _require(
        not incompatible.unexpected_keys,
        "Current warm-start returned unexpected keys",
    )
    installed = model.state_dict()
    unequal = [
        name
        for name, value in current_state.items()
        if not torch.equal(value, installed[name].detach().cpu())
    ]
    _require(not unequal, f"Current tensors changed during warm-start: {unequal[:5]}")
    _require(
        tensor_mapping_sha256(_base_state(model)) == parent["state_sha256"],
        "installed Current state hash differs from parent checkpoint",
    )
    validated = validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(
        model,
        require_zero_initialized_heads=False,
        require_identity_initialized_qfg=False,
        require_identity_initialized_pbdr_v3=True,
    )
    freeze = configure_stage1(model)
    metadata = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "parent_role": _require_role(parent_role),
        "parent_checkpoint": parent,
        "parent_checkpoint_payload_schema": payload.get("schema"),
        "warm_start_used": True,
        "warm_start_strict_contract": (
            "strict_false_with_exact_pbdr_v3_missing_and_no_unexpected"
        ),
        "current_state_key_count": CURRENT_STATE_KEY_COUNT,
        "training_state_key_count": len(model.state_dict()),
        "pbdr_v3_state_keys": list(PBDR_V3_STATE_KEYS),
        "all_current_tensors_bitwise_equal_after_load": True,
        "current_state_sha256_after_load": base_state_sha256(model),
        "initial_pbdr_v3_state_sha256": tensor_mapping_sha256(
            {name: model.state_dict()[name] for name in PBDR_V3_STATE_KEYS}
        ),
        "pbdr_v3_integration_version": PBDR_V3_INTEGRATION_VERSION,
        "architecture_manifest": model.architecture_manifest(),
        "architecture_id": canonical_sha256(model.architecture_manifest()),
        "builder_validation": validated,
        "stage1_freeze_audit": freeze,
        "raw_builder_metadata": raw,
    }
    _require(
        metadata["training_state_key_count"] == TRAINING_STATE_KEY_COUNT,
        "formal PBDR-V3 training state-key count differs",
    )
    return model, metadata


def strip_training_only_survival_state(
    training_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _require(
        len(training_state) == TRAINING_STATE_KEY_COUNT,
        f"candidate training state must have {TRAINING_STATE_KEY_COUNT} keys",
    )
    _require(
        set(SURVIVAL_STATE_KEYS) <= set(training_state),
        "candidate training state lacks Survival keys",
    )
    _require(
        set(PBDR_V3_STATE_KEYS) <= set(training_state),
        "candidate training state lacks PBDR-V3 keys",
    )
    stripped = {
        name: value.detach().cpu().clone()
        for name, value in training_state.items()
        if name not in set(SURVIVAL_STATE_KEYS)
    }
    _require(
        len(stripped) == INFERENCE_STATE_KEY_COUNT,
        "stripped PBDR-V3 inference state-key count differs",
    )
    return stripped


def build_inference_model_from_candidate_state(
    training_state: Mapping[str, torch.Tensor],
    *,
    parent_role: str,
    parent_checkpoint: Path | None = None,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a deployment graph and prove its base still equals Current."""

    _, parent_state, parent = load_current_checkpoint(
        parent_role, parent_checkpoint
    )
    stripped = strip_training_only_survival_state(training_state)
    parent_inference = {
        name: value for name, value in parent_state.items() if name not in set(SURVIVAL_STATE_KEYS)
    }
    candidate_base = {
        name: value
        for name, value in stripped.items()
        if not name.startswith(PBDR_V3_STATE_PREFIX)
    }
    _require(
        set(candidate_base) == set(parent_inference),
        "candidate inference base key set differs from Current",
    )
    changed = [
        name
        for name, value in parent_inference.items()
        if not torch.equal(value.detach().cpu(), candidate_base[name].detach().cpu())
    ]
    _require(not changed, f"candidate modified frozen Current tensors: {changed[:5]}")
    with _preserve_process_rng():
        model, raw = build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(seed)
    incompatible = model.load_state_dict(stripped, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "candidate inference strict load returned incompatible keys",
    )
    validated = validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(model)
    model.eval()
    model.mode = "test"
    metadata = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "parent_role": _require_role(parent_role),
        "parent_checkpoint": parent,
        "training_state_key_count": len(training_state),
        "inference_state_key_count": len(stripped),
        "stripped_training_only_state_keys": list(SURVIVAL_STATE_KEYS),
        "base_bitwise_equal_to_parent": True,
        "strict_load": True,
        "builder_validation": validated,
        "raw_builder_metadata": raw,
    }
    return model, metadata


__all__ = [
    "CURRENT_STATE_KEY_COUNT",
    "DATASET",
    "DEFAULT_PARENT_CHECKPOINTS",
    "INFERENCE_STATE_KEY_COUNT",
    "PARENT_ROLES",
    "PBDRV3ModelProtocolError",
    "RUNTIME_DEPENDENCY_RELATIVE_PATHS",
    "SCHEMA",
    "TRAINING_SEED",
    "TRAINING_STATE_KEY_COUNT",
    "audit_stage1",
    "base_state_sha256",
    "batchnorm_buffer_sha256",
    "build_inference_model_from_candidate_state",
    "build_stage1_training_model",
    "canonical_sha256",
    "configure_stage1",
    "file_sha256",
    "load_current_checkpoint",
    "runtime_source_paths",
    "runtime_source_records",
    "strip_training_only_survival_state",
    "tensor_mapping_sha256",
]
