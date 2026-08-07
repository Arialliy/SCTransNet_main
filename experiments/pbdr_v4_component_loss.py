"""Role-aligned pixel, overlap, and component objectives for PBDR-V4.

Component-ID maps are generated from the frozen Current parent on the
development-train split with the canonical V4 matcher.  Positive IDs identify
independent components; zero denotes pixels outside that atlas class.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import torch
import torch.nn.functional as F


Role = Literal["best_miou", "best_pd"]
COMPONENT_PEAK_TEMPERATURE = 0.25

ROLE_TVERSKY: dict[Role, tuple[float, float]] = {
    "best_miou": (0.50, 0.50),
    "best_pd": (0.30, 0.70),
}
ROLE_WEIGHTS: dict[Role, dict[str, float]] = {
    "best_miou": {
        "bce": 1.00,
        "tversky": 1.00,
        "rescue": 2.00,
        "suppress": 1.50,
        "preserve": 1.00,
        "foreground_drop": 1.00,
        "background_increase": 1.00,
        "neutral_delta": 0.01,
    },
    "best_pd": {
        "bce": 0.50,
        "tversky": 0.75,
        "rescue": 5.00,
        "suppress": 0.50,
        "preserve": 2.00,
        "foreground_drop": 2.00,
        "background_increase": 0.25,
        "neutral_delta": 0.005,
    },
}


@dataclass(frozen=True, slots=True)
class PBDRV4LossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    tversky: torch.Tensor
    rescue_components: torch.Tensor
    suppress_components: torch.Tensor
    preserve_components: torch.Tensor
    foreground_drop: torch.Tensor
    background_increase: torch.Tensor
    neutral_delta: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in self.__dataclass_fields__
        }


def _require_role(role: str) -> Role:
    if role not in ROLE_WEIGHTS:
        raise ValueError(f"unsupported role: {role!r}")
    return role  # type: ignore[return-value]


def role_loss_manifest(role: Role) -> dict[str, object]:
    ready = _require_role(role)
    alpha, beta = ROLE_TVERSKY[ready]
    return {
        "role": ready,
        "component_peak_temperature": COMPONENT_PEAK_TEMPERATURE,
        "tversky_alpha": alpha,
        "tversky_beta": beta,
        "weights": dict(ROLE_WEIGHTS[ready]),
        "performance_acceptance_margin": None,
    }


def _require_float_bchw(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
        raise ValueError(f"{name} must be non-empty BCHW with C=1")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _require_component_ids(
    value: torch.Tensor,
    *,
    name: str,
    reference: torch.Tensor,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must share the routed-logit shape")
    if value.device != reference.device:
        raise ValueError(f"{name} must share the routed-logit device")
    if bool((value < 0).any()):
        raise ValueError(f"{name} contains a negative component ID")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    denominator = weight.sum().clamp_min(1.0)
    return (value * weight).sum() / denominator


def _soft_component_peaks(
    logits: torch.Tensor,
    component_ids: torch.Tensor,
    *,
    temperature: float = COMPONENT_PEAK_TEMPERATURE,
) -> list[torch.Tensor]:
    """Return one size-normalized smooth maximum for each positive ID."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    _require_float_bchw(logits, name="logits")
    _require_component_ids(
        component_ids,
        name="component_ids",
        reference=logits,
    )
    peaks: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        ids = torch.unique(component_ids[batch_index])
        ids = ids[ids > 0]
        for component_id in ids:
            mask = component_ids[batch_index] == component_id
            values = logits[batch_index][mask]
            peak = temperature * (
                torch.logsumexp(values / temperature, dim=0)
                - torch.log(values.new_tensor(float(values.numel())))
            )
            peaks.append(peak)
    return peaks


def _positive_component_loss(
    logits: torch.Tensor,
    component_ids: torch.Tensor,
) -> torch.Tensor:
    peaks = _soft_component_peaks(logits, component_ids)
    if not peaks:
        return logits.new_zeros(())
    return torch.stack([F.softplus(-peak) for peak in peaks]).mean()


def _negative_component_loss(
    logits: torch.Tensor,
    component_ids: torch.Tensor,
) -> torch.Tensor:
    peaks = _soft_component_peaks(logits, component_ids)
    if not peaks:
        return logits.new_zeros(())
    return torch.stack([F.softplus(peak) for peak in peaks]).mean()


def _tversky_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    reduce_dims = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=reduce_dims)
    false_positive = (probability * (1.0 - target)).sum(dim=reduce_dims)
    false_negative = ((1.0 - probability) * target).sum(dim=reduce_dims)
    score = (true_positive + epsilon) / (
        true_positive + alpha * false_positive + beta * false_negative + epsilon
    )
    return 1.0 - score.mean()


def _validate_loss_inputs(
    *,
    routed_logits: torch.Tensor,
    candidate_base_logits: torch.Tensor,
    reference_current_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    target: torch.Tensor,
    component_maps: Mapping[str, torch.Tensor],
) -> None:
    _require_float_bchw(routed_logits, name="routed_logits")
    for name, value in (
        ("candidate_base_logits", candidate_base_logits),
        ("reference_current_logits", reference_current_logits),
        ("delta_logits", delta_logits),
        ("target", target),
    ):
        _require_float_bchw(value, name=name)
        if value.shape != routed_logits.shape:
            raise ValueError(f"{name} must share the routed-logit shape")
        if value.device != routed_logits.device or value.dtype != routed_logits.dtype:
            raise ValueError(f"{name} must share routed-logit device and dtype")
    if bool((target < 0.0).any()) or bool((target > 1.0).any()):
        raise ValueError("target must lie in [0, 1]")
    for name, value in component_maps.items():
        _require_component_ids(value, name=name, reference=routed_logits)


def compute_pbdr_v4_loss(
    *,
    role: Role,
    routed_logits: torch.Tensor,
    candidate_base_logits: torch.Tensor,
    reference_current_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    target: torch.Tensor,
    rescue_component_ids: torch.Tensor,
    suppress_component_ids: torch.Tensor,
    preserve_component_ids: torch.Tensor,
) -> PBDRV4LossOutput:
    """Compute the frozen role-specific V4 training objective.

    ``candidate_base_logits`` is the base head inside the trainable candidate;
    in Stage-2 it moves with ``outc``/``up_decoder1``.  The one-sided
    foreground/background protections therefore use the separate, frozen
    ``reference_current_logits`` and never silently treat the moving candidate
    base as Current.
    """

    ready_role = _require_role(role)
    component_maps = {
        "rescue_component_ids": rescue_component_ids,
        "suppress_component_ids": suppress_component_ids,
        "preserve_component_ids": preserve_component_ids,
    }
    _validate_loss_inputs(
        routed_logits=routed_logits,
        candidate_base_logits=candidate_base_logits,
        reference_current_logits=reference_current_logits,
        delta_logits=delta_logits,
        target=target,
        component_maps=component_maps,
    )

    routed = routed_logits.float()
    # The candidate base is deliberately kept as an explicit contract input,
    # even though only its routed residual is optimized by this loss.  This
    # distinction prevents Stage-2 callers from passing it as Current.
    _candidate_base = candidate_base_logits.float()
    current_reference = reference_current_logits.detach().float()
    target_float = target.float()
    probability = torch.sigmoid(routed)
    current_probability = torch.sigmoid(current_reference)
    bce = F.binary_cross_entropy_with_logits(routed, target_float)
    alpha, beta = ROLE_TVERSKY[ready_role]
    tversky = _tversky_loss(
        probability,
        target_float,
        alpha=alpha,
        beta=beta,
    )
    rescue_components = _positive_component_loss(
        routed,
        rescue_component_ids,
    )
    suppress_components = _negative_component_loss(
        routed,
        suppress_component_ids,
    )
    preserve_components = _positive_component_loss(
        routed,
        preserve_component_ids,
    )
    foreground = target_float >= 0.5
    background = ~foreground
    foreground_drop = _masked_mean(
        F.relu(current_probability - probability).square(),
        foreground,
    )
    background_increase = _masked_mean(
        F.relu(probability - current_probability).square(),
        background,
    )
    edited = (
        (rescue_component_ids > 0)
        | (suppress_component_ids > 0)
        | (preserve_component_ids > 0)
    )
    neutral_delta = _masked_mean(delta_logits.float().abs(), ~edited)
    weights = ROLE_WEIGHTS[ready_role]
    total = (
        weights["bce"] * bce
        + weights["tversky"] * tversky
        + weights["rescue"] * rescue_components
        + weights["suppress"] * suppress_components
        + weights["preserve"] * preserve_components
        + weights["foreground_drop"] * foreground_drop
        + weights["background_increase"] * background_increase
        + weights["neutral_delta"] * neutral_delta
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("PBDR-V4 loss is non-finite")
    return PBDRV4LossOutput(
        total=total,
        bce=bce,
        tversky=tversky,
        rescue_components=rescue_components,
        suppress_components=suppress_components,
        preserve_components=preserve_components,
        foreground_drop=foreground_drop,
        background_increase=background_increase,
        neutral_delta=neutral_delta,
    )


__all__ = [
    "COMPONENT_PEAK_TEMPERATURE",
    "PBDRV4LossOutput",
    "ROLE_TVERSKY",
    "ROLE_WEIGHTS",
    "compute_pbdr_v4_loss",
    "role_loss_manifest",
]
