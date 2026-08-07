"""Stage-aware mutable/frozen state contract for PBDR-V4.

Stage-1 may train only ``pbdr_v4`` parameters.  Stage-2 may additionally
train parameters in ``outc`` and ``up_decoder1``.  BatchNorm modules remain in
evaluation mode in both stages, and every base-model buffer (including the BN
running state below ``up_decoder1``) must remain bitwise equal to Current.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal, Mapping

import torch
import torch.nn as nn

from experiments.pbdr_v4_zero_margin_selector import MetricRecord, Role, role_key


Stage = Literal["stage1", "stage2"]
PBDR_PREFIX = "pbdr_v4."
STAGE2_BASE_PARAMETER_PREFIXES = ("outc.", "up_decoder1.")


class PBDRV4StateContractError(RuntimeError):
    """A model, snapshot, or train/eval mode violates the stage contract."""


def _require_stage(stage: str) -> Stage:
    if stage not in ("stage1", "stage2"):
        raise PBDRV4StateContractError(f"unsupported V4 stage: {stage!r}")
    return stage  # type: ignore[return-value]


def _is_stage2_base_parameter(name: str) -> bool:
    return name.startswith(STAGE2_BASE_PARAMETER_PREFIXES)


def mutable_parameter_names(model: nn.Module, stage: Stage) -> tuple[str, ...]:
    ready = _require_stage(stage)
    names = []
    for name, _ in model.named_parameters():
        if name.startswith(PBDR_PREFIX) or (
            ready == "stage2" and _is_stage2_base_parameter(name)
        ):
            names.append(name)
    if not names or not any(name.startswith(PBDR_PREFIX) for name in names):
        raise PBDRV4StateContractError("model has no PBDR-V4 parameters")
    return tuple(names)


def configure_stage_training(model: nn.Module, stage: Stage) -> tuple[str, ...]:
    """Set the exact trainable set while keeping the Current graph in eval.

    Gradient eligibility is controlled by ``requires_grad`` and does not
    require the inherited modules to enter training mode.  Keeping the base in
    eval also freezes its Dropout behavior, not only BatchNorm statistics.
    """

    mutable = set(mutable_parameter_names(model, stage))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in mutable)
    model.eval()
    router = getattr(model, "pbdr_v4", None)
    if not isinstance(router, nn.Module):
        raise PBDRV4StateContractError("model has no pbdr_v4 module")
    router.train()
    audit_training_modes(model, stage)
    return tuple(sorted(mutable))


def audit_training_modes(model: nn.Module, stage: Stage) -> dict[str, object]:
    expected = set(mutable_parameter_names(model, stage))
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PBDRV4StateContractError(
            f"trainable parameter set differs; missing={missing}, extra={extra}"
        )
    if model.training:
        raise PBDRV4StateContractError("inherited Current graph is not in eval mode")
    router = getattr(model, "pbdr_v4", None)
    if not isinstance(router, nn.Module) or not router.training:
        raise PBDRV4StateContractError("PBDR-V4 router is not in training mode")
    unexpected_training = [
        name
        for name, module in model.named_modules()
        if name
        and not name.startswith("pbdr_v4")
        and module.training
    ]
    if unexpected_training:
        raise PBDRV4StateContractError(
            f"inherited modules remain in training mode: {unexpected_training}"
        )
    training_bn = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
    ]
    if training_bn:
        raise PBDRV4StateContractError(
            f"BatchNorm modules remain in training mode: {training_bn}"
        )
    return {
        "stage": stage,
        "trainable_parameter_names": sorted(actual),
        "base_training": False,
        "pbdr_v4_training": True,
        "batchnorm_training_modules": [],
    }


def clone_current_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Take an immutable CPU clone of the Current parent state."""

    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name.startswith(PBDR_PREFIX):
            raise PBDRV4StateContractError(
                "Current reference unexpectedly contains PBDR-V4 state"
            )
        state[name] = value.detach().cpu().clone()
    return state


def _tensor_bytes(value: torch.Tensor) -> bytes:
    ready = value.detach().cpu().contiguous()
    return ready.numpy().tobytes(order="C")


def state_semantic_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("state must map strings to tensors")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(len(value.shape).to_bytes(8, "little"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=True))
        raw = _tensor_bytes(value)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def audit_candidate_against_current(
    model: nn.Module,
    *,
    current_state: Mapping[str, torch.Tensor],
    stage: Stage,
) -> dict[str, object]:
    """Verify exact immutable/mutable partitions against frozen Current."""

    ready = _require_stage(stage)
    candidate = model.state_dict()
    parameter_names = set(dict(model.named_parameters()))
    base_names = {name for name in candidate if not name.startswith(PBDR_PREFIX)}
    if base_names != set(current_state):
        missing = sorted(set(current_state) - base_names)
        extra = sorted(base_names - set(current_state))
        raise PBDRV4StateContractError(
            f"base state keys differ from Current; missing={missing}, extra={extra}"
        )

    permitted_changes: list[str] = []
    checked_equal: list[str] = []
    for name in sorted(base_names):
        is_parameter = name in parameter_names
        may_change = (
            ready == "stage2"
            and is_parameter
            and _is_stage2_base_parameter(name)
        )
        if may_change:
            permitted_changes.append(name)
            continue
        observed = candidate[name].detach().cpu()
        expected = current_state[name].detach().cpu()
        if observed.dtype != expected.dtype or tuple(observed.shape) != tuple(expected.shape):
            raise PBDRV4StateContractError(f"immutable tensor metadata differs: {name}")
        if not torch.equal(observed, expected):
            kind = "buffer" if not is_parameter else "parameter"
            raise PBDRV4StateContractError(
                f"immutable {kind} differs from Current: {name}"
            )
        checked_equal.append(name)

    # No buffer is ever mutable, including BatchNorm buffers under an allowed
    # Stage-2 parameter prefix.
    buffer_names = set(dict(model.named_buffers())) - {
        name for name in candidate if name.startswith(PBDR_PREFIX)
    }
    forbidden_buffer_allowance = sorted(set(permitted_changes) & buffer_names)
    if forbidden_buffer_allowance:
        raise PBDRV4StateContractError(
            f"base buffers were incorrectly permitted: {forbidden_buffer_allowance}"
        )
    return {
        "stage": ready,
        "current_state_sha256": state_semantic_sha256(current_state),
        "immutable_tensor_count": len(checked_equal),
        "permitted_changed_parameter_names": permitted_changes,
        "all_base_buffers_bitwise_current": True,
    }


def l2sp_to_current(
    model: nn.Module,
    *,
    current_state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Mean squared Stage-2 base displacement from the Current parent."""

    terms: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not _is_stage2_base_parameter(name):
            continue
        if name not in current_state:
            raise PBDRV4StateContractError(f"Current anchor lacks parameter {name}")
        anchor = current_state[name].to(device=parameter.device, dtype=parameter.dtype)
        if anchor.shape != parameter.shape:
            raise PBDRV4StateContractError(f"Current anchor shape differs: {name}")
        terms.append((parameter - anchor).square().mean())
    if not terms:
        raise PBDRV4StateContractError("no Stage-2 base parameters found for L2-SP")
    return torch.stack(terms).mean()


def checkpoint_epoch_key(
    *,
    role: Role,
    metrics: MetricRecord,
    epoch: int,
) -> tuple[object, ...]:
    """Internal epoch key with no Candidate-vs-Current pass prefix."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise PBDRV4StateContractError("epoch must be a positive integer")
    return (*role_key(role, metrics), -epoch)


__all__ = [
    "PBDR_PREFIX",
    "PBDRV4StateContractError",
    "STAGE2_BASE_PARAMETER_PREFIXES",
    "Stage",
    "audit_candidate_against_current",
    "audit_training_modes",
    "checkpoint_epoch_key",
    "clone_current_state",
    "configure_stage_training",
    "l2sp_to_current",
    "mutable_parameter_names",
    "state_semantic_sha256",
]
