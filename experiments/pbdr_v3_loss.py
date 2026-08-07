"""Metric-aligned objective for the conservative PBDR-V3 calibrator.

The Current prediction is a one-way reference: callers may pass a live tensor
for ``base_logits``, but this module always detaches it before computing the
relative constraints.  Consequently, only the routed candidate is optimized
to respect the frozen (or nearly frozen) Current model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class PBDRV3LossOutput:
    """Named loss terms returned by :func:`compute_pbdr_v3_loss`."""

    total: torch.Tensor
    final_bce: torch.Tensor
    soft_iou: torch.Tensor
    background_increase: torch.Tensor
    foreground_decrease: torch.Tensor
    trust_region: torch.Tensor
    residual_sparsity: torch.Tensor
    hard_negative: torch.Tensor
    deep_supervision: torch.Tensor


def _finite_float(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_float(value: Real, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_logit(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must have NCHW rank 4")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _validate_target(target: torch.Tensor) -> None:
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if target.ndim != 4:
        raise ValueError("target must have NCHW rank 4")
    if target.dtype == torch.bool:
        return
    if target.is_floating_point() or target.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return
    raise TypeError("target must have a boolean or real numeric dtype")


def _require_same_shape_and_device(
    reference_name: str,
    reference: torch.Tensor,
    other_name: str,
    other: torch.Tensor,
) -> None:
    if other.shape != reference.shape:
        raise ValueError(f"{other_name} shape must match {reference_name}")
    if other.device != reference.device:
        raise ValueError(f"{other_name} device must match {reference_name}")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    if selected.numel() == 0:
        return value.new_zeros(())
    return selected.mean()


def soft_iou_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return a batch-mean soft Jaccard loss for NCHW tensors."""

    _validate_logit("probability", probability)
    _validate_target(target)
    _require_same_shape_and_device(
        "probability",
        probability,
        "target",
        target,
    )
    epsilon = _finite_float(eps, "eps")
    if epsilon <= 0.0:
        raise ValueError("eps must be positive")

    probability_float = probability.float()
    target_float = target.float()
    dimensions = (1, 2, 3)
    intersection = (probability_float * target_float).sum(dim=dimensions)
    union = (
        probability_float
        + target_float
        - probability_float * target_float
    ).sum(dim=dimensions)
    return (1.0 - (intersection + epsilon) / (union + epsilon)).mean()


def topk_hard_negative_loss(
    routed_logits: torch.Tensor,
    routed_probability: torch.Tensor,
    base_probability: torch.Tensor,
    target: torch.Tensor,
    *,
    candidate_floor: float = 0.05,
    topk_fraction: float = 0.02,
) -> torch.Tensor:
    """Penalize the highest-loss plausible background pixels.

    A negative pixel becomes a candidate when either Current or the routed
    prediction assigns at least ``candidate_floor`` probability.  This keeps
    the term focused on hard negatives instead of the overwhelmingly easy
    background.
    """

    _validate_logit("routed_logits", routed_logits)
    _validate_logit("routed_probability", routed_probability)
    _validate_logit("base_probability", base_probability)
    _validate_target(target)
    for name, tensor in (
        ("routed_probability", routed_probability),
        ("base_probability", base_probability),
        ("target", target),
    ):
        _require_same_shape_and_device(
            "routed_logits",
            routed_logits,
            name,
            tensor,
        )

    floor = _finite_float(candidate_floor, "candidate_floor")
    fraction = _finite_float(topk_fraction, "topk_fraction")
    if not 0.0 <= floor < 1.0:
        raise ValueError("candidate_floor must be in [0, 1)")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")

    negative = target < 0.5
    candidate = negative & (
        (base_probability >= floor) | (routed_probability >= floor)
    )
    logits = routed_logits[candidate]
    if logits.numel() == 0:
        return routed_logits.new_zeros(())
    # BCEWithLogits(z, 0) == softplus(z).
    losses = F.softplus(logits.float())
    k = max(1, int(math.ceil(losses.numel() * fraction)))
    return losses.topk(k, sorted=False).values.mean()


def compute_pbdr_v3_loss(
    *,
    routed_logits: torch.Tensor,
    base_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    target: torch.Tensor,
    auxiliary_logits: Sequence[torch.Tensor] = (),
    soft_iou_weight: float = 1.0,
    background_increase_weight: float = 8.0,
    foreground_decrease_weight: float = 4.0,
    trust_region_weight: float = 0.25,
    residual_sparsity_weight: float = 0.05,
    hard_negative_weight: float = 2.0,
    deep_supervision_weight: float = 0.0,
    background_margin: float = 0.0,
    foreground_margin: float = 0.0,
    hard_negative_candidate_floor: float = 0.05,
    hard_negative_topk_fraction: float = 0.02,
) -> PBDRV3LossOutput:
    """Compute the conservative, metric-aligned PBDR-V3 objective.

    ``base_logits`` must be the Current logit from the same forward pass.
    Detaching it here makes the monotonic constraints one-way: the candidate
    must move relative to Current rather than moving both endpoints together.
    All weights are required to be finite and non-negative.
    """

    _validate_logit("routed_logits", routed_logits)
    _validate_logit("base_logits", base_logits)
    _validate_logit("delta_logits", delta_logits)
    _validate_target(target)
    for name, tensor in (
        ("base_logits", base_logits),
        ("delta_logits", delta_logits),
        ("target", target),
    ):
        _require_same_shape_and_device(
            "routed_logits",
            routed_logits,
            name,
            tensor,
        )

    weights = {
        "soft_iou_weight": _non_negative_float(
            soft_iou_weight,
            "soft_iou_weight",
        ),
        "background_increase_weight": _non_negative_float(
            background_increase_weight,
            "background_increase_weight",
        ),
        "foreground_decrease_weight": _non_negative_float(
            foreground_decrease_weight,
            "foreground_decrease_weight",
        ),
        "trust_region_weight": _non_negative_float(
            trust_region_weight,
            "trust_region_weight",
        ),
        "residual_sparsity_weight": _non_negative_float(
            residual_sparsity_weight,
            "residual_sparsity_weight",
        ),
        "hard_negative_weight": _non_negative_float(
            hard_negative_weight,
            "hard_negative_weight",
        ),
        "deep_supervision_weight": _non_negative_float(
            deep_supervision_weight,
            "deep_supervision_weight",
        ),
    }
    background_margin_value = _non_negative_float(
        background_margin,
        "background_margin",
    )
    foreground_margin_value = _non_negative_float(
        foreground_margin,
        "foreground_margin",
    )
    candidate_floor = _finite_float(
        hard_negative_candidate_floor,
        "hard_negative_candidate_floor",
    )
    topk_fraction = _finite_float(
        hard_negative_topk_fraction,
        "hard_negative_topk_fraction",
    )
    if not 0.0 <= candidate_floor < 1.0:
        raise ValueError("hard_negative_candidate_floor must be in [0, 1)")
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("hard_negative_topk_fraction must be in (0, 1]")

    target_float = target.float()
    routed_float = routed_logits.float()
    base_float = base_logits.detach().float()
    probability = torch.sigmoid(routed_float)
    base_probability = torch.sigmoid(base_float)

    final_bce = F.binary_cross_entropy_with_logits(
        routed_float,
        target_float,
        reduction="mean",
    )
    iou = soft_iou_loss(probability, target_float)

    background = target_float < 0.5
    foreground = ~background
    background_increase = _masked_mean(
        F.relu(
            probability - base_probability - background_margin_value
        ).square(),
        background,
    )
    foreground_decrease = _masked_mean(
        F.relu(
            base_probability - probability - foreground_margin_value
        ).square(),
        foreground,
    )
    trust_region = (probability - base_probability).square().mean()
    residual_sparsity = delta_logits.float().abs().mean()
    hard_negative = topk_hard_negative_loss(
        routed_float,
        probability,
        base_probability,
        target_float,
        candidate_floor=candidate_floor,
        topk_fraction=topk_fraction,
    )

    deep_supervision = routed_float.new_zeros(())
    if weights["deep_supervision_weight"] > 0.0:
        terms: list[torch.Tensor] = []
        for index, logits in enumerate(auxiliary_logits):
            _validate_logit(f"auxiliary_logits[{index}]", logits)
            _require_same_shape_and_device(
                "target",
                target,
                f"auxiliary_logits[{index}]",
                logits,
            )
            terms.append(
                F.binary_cross_entropy_with_logits(
                    logits.float(),
                    target_float,
                    reduction="mean",
                )
            )
        if terms:
            deep_supervision = torch.stack(terms).sum()

    total = (
        final_bce
        + weights["soft_iou_weight"] * iou
        + weights["background_increase_weight"] * background_increase
        + weights["foreground_decrease_weight"] * foreground_decrease
        + weights["trust_region_weight"] * trust_region
        + weights["residual_sparsity_weight"] * residual_sparsity
        + weights["hard_negative_weight"] * hard_negative
        + weights["deep_supervision_weight"] * deep_supervision
    )
    return PBDRV3LossOutput(
        total=total,
        final_bce=final_bce,
        soft_iou=iou,
        background_increase=background_increase,
        foreground_decrease=foreground_decrease,
        trust_region=trust_region,
        residual_sparsity=residual_sparsity,
        hard_negative=hard_negative,
        deep_supervision=deep_supervision,
    )


__all__ = [
    "PBDRV3LossOutput",
    "compute_pbdr_v3_loss",
    "soft_iou_loss",
    "topk_hard_negative_loss",
]
