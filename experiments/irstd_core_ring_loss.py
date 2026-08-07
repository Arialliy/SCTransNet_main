"""Fixed IRSTD-only objective for the frozen-main BGCR repair head.

The loss directly supervises target cores, attached/detached halo pixels and
counterfactual rings while retaining a detached Current reference.  Relative
target peak/support no-drop terms prevent the repair arm from sacrificing an
existing Current target.  Cross-arm leakage explicitly penalizes positive-arm
growth on negative regions and negative-arm suppression on targets.

All numeric constants are fixed single-arm training weights.  The fixed
``logit > 0`` measurement boundary used by upstream atlas construction is not
a performance acceptance threshold; this module defines no acceptance margin.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F


IRSTD_CRR_LOSS_VERSION = "irstd_core_ring_loss_v1"
COMPONENT_PEAK_TEMPERATURE = 0.25
CENTROID_ROI_RADIUS = 4
LOSS_WEIGHTS: Mapping[str, float] = {
    "bce": 1.0,
    "soft_iou": 2.0,
    "core_gate": 0.75,
    "halo_gate": 0.75,
    "component_peak": 1.5,
    "centroid": 0.5,
    "target_peak_no_drop": 1.0,
    "target_support_no_drop": 1.0,
    "halo_probability": 2.0,
    "attached_halo_probability": 1.0,
    "far_background_no_increase": 0.5,
    "direction": 0.25,
    "cross_arm_leak": 0.5,
    "neutral_delta": 0.01,
}


@dataclass(frozen=True, slots=True)
class IRSTDCoreRingLossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    soft_iou: torch.Tensor
    core_gate: torch.Tensor
    halo_gate: torch.Tensor
    component_peak: torch.Tensor
    centroid: torch.Tensor
    target_peak_no_drop: torch.Tensor
    target_support_no_drop: torch.Tensor
    halo_probability: torch.Tensor
    attached_halo_probability: torch.Tensor
    far_background_no_increase: torch.Tensor
    direction: torch.Tensor
    cross_arm_leak: torch.Tensor
    neutral_delta: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in self.__dataclass_fields__
        }


def loss_manifest() -> dict[str, object]:
    """Return the complete non-tunable BGCR loss contract."""

    return {
        "version": IRSTD_CRR_LOSS_VERSION,
        "weights": dict(LOSS_WEIGHTS),
        "component_peak_temperature": COMPONENT_PEAK_TEMPERATURE,
        "centroid_roi_radius": CENTROID_ROI_RADIUS,
        "current_reference_gradient": "detached",
        "component_reduction": "equal_per_component",
        "attached_halo_supervision": "explicit_absolute_probability",
        "cross_arm_leak_protection": True,
        "baseline_maps_required": False,
        "performance_acceptance_margin": None,
    }


def _require_float_bchw(
    value: torch.Tensor,
    *,
    name: str,
    reference: torch.Tensor | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
        raise ValueError(f"{name} must be non-empty BCHW with C=1")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if reference is not None:
        if value.shape != reference.shape:
            raise ValueError(f"{name} must match routed_logits shape")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                f"{name} must match routed_logits device and dtype"
            )
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _binary_mask(
    value: torch.Tensor,
    *,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match routed_logits shape")
    if value.device != reference.device:
        raise ValueError(f"{name} must match routed_logits device")
    if value.is_floating_point():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains non-finite values")
    elif value.dtype not in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{name} must use bool or a numeric binary dtype")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain only binary values")
    return value.detach().to(dtype=torch.bool)


def _validated_component_ids(
    value: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("target_component_ids must be a tensor")
    if value.shape != reference.shape:
        raise ValueError("target_component_ids must match routed_logits shape")
    if value.device != reference.device:
        raise ValueError("target_component_ids must match routed_logits device")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_component_ids must use int32 or int64")
    if bool((value < 0).any()):
        raise ValueError("target_component_ids contains a negative ID")
    return value.detach()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _balanced_binary_logit_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    positive = positive_mask.bool()
    negative = ~positive
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive]).mean())
    if bool(negative.any()):
        terms.append(F.softplus(logits[negative]).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target_float = target.float()
    reduce_dims = tuple(range(1, probability.ndim))
    intersection = (probability * target_float).sum(dim=reduce_dims)
    union = (
        probability + target_float - probability * target_float
    ).sum(dim=reduce_dims)
    # This is a division stabilizer, not a performance acceptance margin.
    score = (intersection + 1.0e-6) / (union + 1.0e-6)
    return 1.0 - score.mean()


def _component_ids_for_sample(
    target_component_ids: torch.Tensor,
    batch_index: int,
) -> torch.Tensor:
    component_ids = torch.unique(target_component_ids[batch_index])
    return component_ids[component_ids > 0]


def _smooth_component_peak(
    values: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("component peak requires a non-empty vector")
    return temperature * (
        torch.logsumexp(values / temperature, dim=0)
        - torch.log(values.new_tensor(float(values.numel())))
    )


def _component_peak_loss(
    logits: torch.Tensor,
    target_component_ids: torch.Tensor,
    *,
    temperature: float = COMPONENT_PEAK_TEMPERATURE,
) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    losses: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        for component_id in _component_ids_for_sample(
            target_component_ids,
            batch_index,
        ):
            mask = target_component_ids[batch_index] == component_id
            peak = _smooth_component_peak(
                logits[batch_index][mask],
                temperature=temperature,
            )
            losses.append(F.softplus(-peak))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _component_peak_no_drop(
    candidate_logits: torch.Tensor,
    current_logits: torch.Tensor,
    target_component_ids: torch.Tensor,
    *,
    temperature: float = COMPONENT_PEAK_TEMPERATURE,
) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    losses: list[torch.Tensor] = []
    for batch_index in range(candidate_logits.shape[0]):
        for component_id in _component_ids_for_sample(
            target_component_ids,
            batch_index,
        ):
            mask = target_component_ids[batch_index] == component_id
            candidate_peak = _smooth_component_peak(
                candidate_logits[batch_index][mask],
                temperature=temperature,
            )
            current_peak = _smooth_component_peak(
                current_logits[batch_index][mask],
                temperature=temperature,
            )
            losses.append(F.relu(current_peak - candidate_peak).square())
    if not losses:
        return candidate_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _component_support_no_drop(
    candidate_logits: torch.Tensor,
    current_logits: torch.Tensor,
    target_component_ids: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for batch_index in range(candidate_logits.shape[0]):
        for component_id in _component_ids_for_sample(
            target_component_ids,
            batch_index,
        ):
            component = target_component_ids[batch_index] == component_id
            support = component & (current_logits[batch_index] > 0.0)
            if bool(support.any()):
                drop = F.relu(
                    current_logits[batch_index][support]
                    - candidate_logits[batch_index][support]
                )
                losses.append(drop.square().mean())
            else:
                # Keep equal-per-component reduction without an empty mean.
                losses.append(candidate_logits[batch_index].sum() * 0.0)
    if not losses:
        return candidate_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _component_centroid_loss(
    logits: torch.Tensor,
    target_component_ids: torch.Tensor,
    *,
    roi_radius: int = CENTROID_ROI_RADIUS,
) -> torch.Tensor:
    if type(roi_radius) is not int or roi_radius < 1:
        raise ValueError("roi_radius must be a positive integer")
    probability = torch.sigmoid(logits.float())
    height, width = probability.shape[-2:]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(
            height,
            device=probability.device,
            dtype=probability.dtype,
        ),
        torch.arange(
            width,
            device=probability.device,
            dtype=probability.dtype,
        ),
        indexing="ij",
    )
    losses: list[torch.Tensor] = []
    kernel_size = 2 * roi_radius + 1
    for batch_index in range(probability.shape[0]):
        for component_id in _component_ids_for_sample(
            target_component_ids,
            batch_index,
        ):
            component = target_component_ids[batch_index, 0] == component_id
            component_float = component.to(dtype=probability.dtype)
            roi = F.max_pool2d(
                component_float[None, None],
                kernel_size=kernel_size,
                stride=1,
                padding=roi_radius,
            )[0, 0]
            mass = probability[batch_index, 0] * roi
            denominator = mass.sum().clamp_min(1.0e-6)
            predicted_y = (mass * y_grid).sum() / denominator
            predicted_x = (mass * x_grid).sum() / denominator
            target_denominator = component_float.sum().clamp_min(1.0)
            target_y = (component_float * y_grid).sum() / target_denominator
            target_x = (component_float * x_grid).sum() / target_denominator
            scale = component_float.sum().sqrt().clamp_min(1.0) + float(
                roi_radius
            )
            losses.append(
                (
                    (predicted_y - target_y).square()
                    + (predicted_x - target_x).square()
                )
                / scale.square()
            )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_irstd_core_ring_loss(
    *,
    routed_logits: torch.Tensor,
    current_logits: torch.Tensor,
    target: torch.Tensor,
    target_component_ids: torch.Tensor,
    core_target: torch.Tensor,
    halo_target: torch.Tensor,
    attached_halo: torch.Tensor,
    far_background: torch.Tensor,
    core_gate_logits: torch.Tensor,
    halo_gate_logits: torch.Tensor,
    positive_delta: torch.Tensor,
    negative_delta: torch.Tensor,
    delta_logits: torch.Tensor,
) -> IRSTDCoreRingLossOutput:
    """Compute the fixed single-arm IRSTD BGCR training objective."""

    _require_float_bchw(routed_logits, name="routed_logits")
    for name, value in (
        ("current_logits", current_logits),
        ("core_gate_logits", core_gate_logits),
        ("halo_gate_logits", halo_gate_logits),
        ("positive_delta", positive_delta),
        ("negative_delta", negative_delta),
        ("delta_logits", delta_logits),
    ):
        _require_float_bchw(value, name=name, reference=routed_logits)

    target_mask = _binary_mask(
        target,
        name="target",
        reference=routed_logits,
    )
    core_mask = _binary_mask(
        core_target,
        name="core_target",
        reference=routed_logits,
    )
    halo_mask = _binary_mask(
        halo_target,
        name="halo_target",
        reference=routed_logits,
    )
    attached_halo_mask = _binary_mask(
        attached_halo,
        name="attached_halo",
        reference=routed_logits,
    )
    far_background_mask = _binary_mask(
        far_background,
        name="far_background",
        reference=routed_logits,
    )
    component_ids = _validated_component_ids(
        target_component_ids,
        reference=routed_logits,
    )

    if not torch.equal(component_ids > 0, target_mask):
        raise ValueError("target_component_ids must reproduce target support")
    if bool((core_mask & ~target_mask).any()):
        raise ValueError("core_target must be a subset of target")
    if bool((halo_mask & target_mask).any()):
        raise ValueError("halo_target must not overlap target")
    if bool((attached_halo_mask & target_mask).any()):
        raise ValueError("attached_halo must not overlap target")
    if bool((attached_halo_mask & ~halo_mask).any()):
        raise ValueError("attached_halo must be included in halo_target")
    if bool((far_background_mask & target_mask).any()):
        raise ValueError("far_background must not overlap target")
    if not torch.equal(delta_logits, positive_delta - negative_delta):
        raise ValueError(
            "delta_logits must equal positive_delta - negative_delta exactly"
        )

    routed = routed_logits.float()
    current = current_logits.detach().float()
    target_float = target_mask.to(dtype=routed.dtype)
    positive_delta_float = positive_delta.float()
    negative_delta_float = negative_delta.float()

    bce = F.binary_cross_entropy_with_logits(routed, target_float)
    soft_iou = _soft_iou_loss(routed, target_float)
    core_gate = _balanced_binary_logit_loss(
        core_gate_logits.float(),
        core_mask,
    )
    halo_gate = _balanced_binary_logit_loss(
        halo_gate_logits.float(),
        halo_mask,
    )
    component_peak = _component_peak_loss(routed, component_ids)
    centroid = _component_centroid_loss(routed, component_ids)
    target_peak_no_drop = _component_peak_no_drop(
        routed,
        current,
        component_ids,
    )
    target_support_no_drop = _component_support_no_drop(
        routed,
        current,
        component_ids,
    )

    probability = torch.sigmoid(routed)
    current_probability = torch.sigmoid(current)
    halo_probability = _masked_mean(probability.square(), halo_mask)
    attached_halo_probability = _masked_mean(
        probability.square(),
        attached_halo_mask,
    )
    far_background_no_increase = _masked_mean(
        F.relu(probability - current_probability).square(),
        far_background_mask,
    )

    # Direction constrains each arm on its intended supervision region.
    direction = (
        _masked_mean(
            F.relu(-positive_delta_float).square(),
            core_mask,
        )
        + _masked_mean(
            F.relu(-negative_delta_float).square(),
            halo_mask,
        )
    )
    # Positive-arm growth is harmful on explicit negative/far-background
    # regions.  Because negative_delta is subtracted from the base, a positive
    # negative-arm value is harmful on every target pixel.
    positive_leak_mask = halo_mask | far_background_mask
    cross_arm_leak = (
        _masked_mean(
            F.relu(positive_delta_float).square(),
            positive_leak_mask,
        )
        + _masked_mean(
            F.relu(negative_delta_float).square(),
            target_mask,
        )
    )
    edited = core_mask | halo_mask
    neutral_delta = _masked_mean(delta_logits.float().abs(), ~edited)

    components = {
        "bce": bce,
        "soft_iou": soft_iou,
        "core_gate": core_gate,
        "halo_gate": halo_gate,
        "component_peak": component_peak,
        "centroid": centroid,
        "target_peak_no_drop": target_peak_no_drop,
        "target_support_no_drop": target_support_no_drop,
        "halo_probability": halo_probability,
        "attached_halo_probability": attached_halo_probability,
        "far_background_no_increase": far_background_no_increase,
        "direction": direction,
        "cross_arm_leak": cross_arm_leak,
        "neutral_delta": neutral_delta,
    }
    if set(components) != set(LOSS_WEIGHTS):
        raise RuntimeError("IRSTD BGCR loss components and fixed weights differ")
    total = routed.sum() * 0.0
    for name, value in components.items():
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"IRSTD BGCR loss component {name} is non-finite")
        total = total + LOSS_WEIGHTS[name] * value
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("IRSTD core/ring loss is non-finite")
    return IRSTDCoreRingLossOutput(total=total, **components)


__all__ = [
    "CENTROID_ROI_RADIUS",
    "COMPONENT_PEAK_TEMPERATURE",
    "IRSTD_CRR_LOSS_VERSION",
    "IRSTDCoreRingLossOutput",
    "LOSS_WEIGHTS",
    "compute_irstd_core_ring_loss",
    "loss_manifest",
]
