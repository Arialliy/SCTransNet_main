"""EC-TSS V3.1 loss for the frozen composed SCTransNet training graph.

The segmentation objective is delegated to :mod:`experiments.tpd_training_loss`
with ``survival_weight=0``.  This keeps the historical one/six-output BCE
addition order unchanged.  EC-TSS then consumes the existing paired stride-16
survival logits and conditions their supervision on a detached final
segmentation probability map.

This module owns no parameter or buffer and deliberately does not accept the
historical ``survival_pos_weight`` argument.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import ContextManager

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tpd_training_loss import (
    SURVIVAL_DOWNSAMPLE,
    build_survival_target,
    compute_tpd_training_loss,
)
from model.tpd_forward_contract import (
    ForwardOutput,
    TPDForwardOutput,
    evaluator_prediction,
)


DEFAULT_SURVIVAL_WEIGHT = 0.005
DEFAULT_SURVIVAL_RATIO_CAP = 0.10
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_TARGET_DILATION_RADIUS = 3


class ECTSSV31LossError(ValueError):
    """The requested EC-TSS V3.1 objective violates its tensor contract."""


@dataclass(frozen=True, slots=True)
class ECTSSV31RiskMaps:
    """Detached target/background confidence maps and their gated risks."""

    target16: torch.Tensor
    target_neighborhood: torch.Tensor
    target_probability16: torch.Tensor
    background_probability16: torch.Tensor
    positive_risk: torch.Tensor
    negative_risk: torch.Tensor

    def __post_init__(self) -> None:
        low_resolution = (
            self.target16,
            self.target_probability16,
            self.background_probability16,
            self.positive_risk,
            self.negative_risk,
        )
        reference_shape = tuple(self.target16.shape)
        for name, value in (
            ("target16", self.target16),
            ("target_neighborhood", self.target_neighborhood),
            ("target_probability16", self.target_probability16),
            ("background_probability16", self.background_probability16),
            ("positive_risk", self.positive_risk),
            ("negative_risk", self.negative_risk),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise ECTSSV31LossError(f"{name} must be a Bx1xHxW Tensor")
            if value.shape[1] != 1 or min(value.shape) <= 0:
                raise ECTSSV31LossError(f"{name} has an invalid shape")
            if value.requires_grad:
                raise ECTSSV31LossError(f"{name} must be detached")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ECTSSV31LossError(f"{name} must be a finite floating Tensor")
            if torch.any(value < 0) or torch.any(value > 1):
                raise ECTSSV31LossError(f"{name} values must be in [0, 1]")
        for value in low_resolution[1:]:
            if tuple(value.shape) != reference_shape:
                raise ECTSSV31LossError("all stride-16 risk maps must share one shape")


@dataclass(frozen=True, slots=True)
class ECTSSV31TrainingLoss:
    """Scalar objective components and complete EC-TSS epoch-audit inputs."""

    total: torch.Tensor
    segmentation: torch.Tensor
    survival: torch.Tensor
    segmentation_terms: tuple[torch.Tensor, ...]
    positive_survival: torch.Tensor
    negative_survival: torch.Tensor
    endpoint_positive_terms: tuple[torch.Tensor, ...]
    endpoint_negative_terms: tuple[torch.Tensor, ...]
    effective_survival_weight: torch.Tensor
    weighted_survival: torch.Tensor
    positive_risk_mass: torch.Tensor
    negative_risk_mass: torch.Tensor
    positive_active_cells: torch.Tensor
    negative_active_cells: torch.Tensor

    def __post_init__(self) -> None:
        scalar_fields = (
            ("total", self.total),
            ("segmentation", self.segmentation),
            ("survival", self.survival),
            ("positive_survival", self.positive_survival),
            ("negative_survival", self.negative_survival),
            ("effective_survival_weight", self.effective_survival_weight),
            ("weighted_survival", self.weighted_survival),
            ("positive_risk_mass", self.positive_risk_mass),
            ("negative_risk_mass", self.negative_risk_mass),
            ("positive_active_cells", self.positive_active_cells),
            ("negative_active_cells", self.negative_active_cells),
        )
        for name, value in scalar_fields:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise ECTSSV31LossError(f"{name} must be a scalar Tensor")
            if not torch.isfinite(value):
                raise FloatingPointError(f"{name} is non-finite")
        for name, terms in (
            ("segmentation_terms", self.segmentation_terms),
            ("endpoint_positive_terms", self.endpoint_positive_terms),
            ("endpoint_negative_terms", self.endpoint_negative_terms),
        ):
            if name != "segmentation_terms" and len(terms) != 2:
                raise ECTSSV31LossError(f"{name} must contain exactly two terms")
            if name == "segmentation_terms" and len(terms) not in {1, 6}:
                raise ECTSSV31LossError(
                    "segmentation_terms must contain one or six terms"
                )
            for index, value in enumerate(terms):
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise ECTSSV31LossError(
                        f"{name}[{index}] must be a scalar Tensor"
                    )
                if not torch.isfinite(value):
                    raise FloatingPointError(f"{name}[{index}] is non-finite")
        if int(self.positive_active_cells.detach().item()) < 0:
            raise ECTSSV31LossError("positive_active_cells must be non-negative")
        if int(self.negative_active_cells.detach().item()) < 0:
            raise ECTSSV31LossError("negative_active_cells must be non-negative")


def _disabled_autocast(device_type: str) -> ContextManager[object]:
    if device_type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ECTSSV31LossError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ECTSSV31LossError(f"{name} must be finite and non-negative")
    return result


def _finite_positive_optional(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    result = _finite_nonnegative(name, value)
    if result == 0:
        raise ECTSSV31LossError(f"{name} must be positive when provided")
    return result


def _confidence_threshold(value: float) -> float:
    result = _finite_nonnegative("confidence_threshold", value)
    if not 0 < result < 1:
        raise ECTSSV31LossError("confidence_threshold must be strictly within (0, 1)")
    return result


def _dilation_radius(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ECTSSV31LossError(
            "target_dilation_radius must be a non-negative integer"
        )
    return value


def _validate_probability_pair(
    final_probability: torch.Tensor,
    segmentation_target: torch.Tensor,
) -> None:
    if not isinstance(final_probability, torch.Tensor):
        raise ECTSSV31LossError("final segmentation probability must be a Tensor")
    if final_probability.shape != segmentation_target.shape:
        raise ECTSSV31LossError(
            "final segmentation probability and target shapes differ"
        )
    if final_probability.device != segmentation_target.device:
        raise ECTSSV31LossError(
            "final segmentation probability and target devices differ"
        )
    if not final_probability.is_floating_point():
        raise ECTSSV31LossError("final segmentation probability must be floating")
    detached = final_probability.detach()
    if not torch.isfinite(detached).all():
        raise FloatingPointError("final segmentation probability is non-finite")
    if torch.any(detached < 0) or torch.any(detached > 1):
        raise ECTSSV31LossError(
            "final segmentation probability values must be in [0, 1]"
        )


def build_ec_tss_v3_1_risk_maps(
    final_probability: torch.Tensor,
    segmentation_target: torch.Tensor,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    target_dilation_radius: int = DEFAULT_TARGET_DILATION_RADIUS,
) -> ECTSSV31RiskMaps:
    """Build the exact detached V3.1 target-neighborhood/background risks."""

    threshold = _confidence_threshold(confidence_threshold)
    radius = _dilation_radius(target_dilation_radius)
    _validate_probability_pair(final_probability, segmentation_target)
    # Reuse the historical validation, divisibility check, and Y16 definition.
    target16 = build_survival_target(segmentation_target)
    kernel_size = 2 * radius + 1
    with torch.no_grad(), _disabled_autocast(segmentation_target.device.type):
        float_target = segmentation_target.detach().float()
        target_neighborhood = F.max_pool2d(
            float_target,
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )
        detached_probability = final_probability.detach().float()
        target_probability16 = F.max_pool2d(
            detached_probability * target_neighborhood,
            kernel_size=SURVIVAL_DOWNSAMPLE,
            stride=SURVIVAL_DOWNSAMPLE,
        )
        background_probability16 = F.max_pool2d(
            detached_probability * (1.0 - target_neighborhood),
            kernel_size=SURVIVAL_DOWNSAMPLE,
            stride=SURVIVAL_DOWNSAMPLE,
        )
        positive_risk = target16.float() * torch.clamp(
            (threshold - target_probability16) / threshold,
            min=0.0,
            max=1.0,
        )
        negative_risk = (1.0 - target16.float()) * torch.clamp(
            (background_probability16 - threshold) / (1.0 - threshold),
            min=0.0,
            max=1.0,
        )
    return ECTSSV31RiskMaps(
        target16=target16.detach().float(),
        target_neighborhood=target_neighborhood,
        target_probability16=target_probability16,
        background_probability16=background_probability16,
        positive_risk=positive_risk,
        negative_risk=negative_risk,
    )


def _risk_normalized_sum(
    weighted_term: torch.Tensor,
    risk: torch.Tensor,
) -> torch.Tensor:
    if weighted_term.shape != risk.shape:
        raise ECTSSV31LossError("weighted_term and risk shapes differ")
    denominator = risk.sum().clamp_min(1.0)
    return weighted_term.sum() / denominator


def compute_error_conditioned_endpoint_terms(
    logit: torch.Tensor,
    positive_risk: torch.Tensor,
    negative_risk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one endpoint's risk-mass-normalized positive/negative terms."""

    if not isinstance(logit, torch.Tensor) or logit.ndim != 4:
        raise ECTSSV31LossError("survival logit must be a Bx1xHxW Tensor")
    if logit.shape != positive_risk.shape or logit.shape != negative_risk.shape:
        raise ECTSSV31LossError("survival logit and risk-map shapes differ")
    if logit.device != positive_risk.device or logit.device != negative_risk.device:
        raise ECTSSV31LossError("survival logit and risk-map devices differ")
    if not logit.is_floating_point() or not torch.isfinite(logit).all():
        raise ECTSSV31LossError("survival logit must be finite and floating")
    with _disabled_autocast(logit.device.type):
        positive_element = F.softplus(-logit.float())
        negative_element = F.softplus(logit.float())
        positive_loss = _risk_normalized_sum(
            positive_risk * positive_element,
            positive_risk,
        )
        negative_loss = _risk_normalized_sum(
            negative_risk * negative_element,
            negative_risk,
        )
    return positive_loss, negative_loss


def compute_ec_tss_v3_1_training_loss(
    output: ForwardOutput,
    segmentation_target: torch.Tensor,
    segmentation_criterion: nn.Module,
    *,
    survival_weight: float = DEFAULT_SURVIVAL_WEIGHT,
    survival_ratio_cap: float | None = DEFAULT_SURVIVAL_RATIO_CAP,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    target_dilation_radius: int = DEFAULT_TARGET_DILATION_RADIUS,
) -> ECTSSV31TrainingLoss:
    """Compute the frozen segmentation objective plus EC-TSS V3.1."""

    weight = _finite_nonnegative("survival_weight", survival_weight)
    ratio_cap = _finite_positive_optional(
        "survival_ratio_cap", survival_ratio_cap
    )
    threshold = _confidence_threshold(confidence_threshold)
    radius = _dilation_radius(target_dilation_radius)

    # This is the sole authoritative segmentation calculation.  In particular,
    # it preserves the historical Python addition order for six BCE terms.
    base = compute_tpd_training_loss(
        output,
        segmentation_target,
        segmentation_criterion,
        survival_weight=0.0,
    )
    if not isinstance(output, TPDForwardOutput):
        raise ECTSSV31LossError(
            "EC-TSS requires a structured TPDForwardOutput"
        )
    logits = output.survival_logits
    if logits is None:
        raise ECTSSV31LossError("EC-TSS requires both survival logits")

    final_probability = evaluator_prediction(output)
    risks = build_ec_tss_v3_1_risk_maps(
        final_probability,
        segmentation_target,
        confidence_threshold=threshold,
        target_dilation_radius=radius,
    )
    endpoint_positive: list[torch.Tensor] = []
    endpoint_negative: list[torch.Tensor] = []
    for logit in logits:
        positive_term, negative_term = compute_error_conditioned_endpoint_terms(
            logit,
            risks.positive_risk,
            risks.negative_risk,
        )
        endpoint_positive.append(positive_term)
        endpoint_negative.append(negative_term)

    with _disabled_autocast(segmentation_target.device.type):
        positive_survival = 0.5 * (
            endpoint_positive[0] + endpoint_positive[1]
        )
        negative_survival = 0.5 * (
            endpoint_negative[0] + endpoint_negative[1]
        )
        # Exact frozen aggregation: endpoint mean of positive/negative means.
        survival_loss = 0.25 * (
            endpoint_positive[0]
            + endpoint_negative[0]
            + endpoint_positive[1]
            + endpoint_negative[1]
        )
        requested_weight = base.segmentation.new_tensor(weight)
        if ratio_cap is None:
            effective_weight = requested_weight
        else:
            epsilon = torch.finfo(survival_loss.dtype).eps
            capped_weight = (
                ratio_cap
                * base.segmentation.detach()
                / survival_loss.detach().clamp_min(epsilon)
            )
            effective_weight = torch.minimum(requested_weight, capped_weight)
        weighted_survival = effective_weight * survival_loss
        total = base.segmentation + weighted_survival

    return ECTSSV31TrainingLoss(
        total=total,
        segmentation=base.segmentation,
        survival=survival_loss,
        segmentation_terms=base.segmentation_terms,
        positive_survival=positive_survival,
        negative_survival=negative_survival,
        endpoint_positive_terms=tuple(endpoint_positive),
        endpoint_negative_terms=tuple(endpoint_negative),
        effective_survival_weight=effective_weight,
        weighted_survival=weighted_survival,
        positive_risk_mass=risks.positive_risk.sum(),
        negative_risk_mass=risks.negative_risk.sum(),
        positive_active_cells=(risks.positive_risk > 0).sum(),
        negative_active_cells=(risks.negative_risk > 0).sum(),
    )


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_SURVIVAL_RATIO_CAP",
    "DEFAULT_SURVIVAL_WEIGHT",
    "DEFAULT_TARGET_DILATION_RADIUS",
    "ECTSSV31LossError",
    "ECTSSV31RiskMaps",
    "ECTSSV31TrainingLoss",
    "build_ec_tss_v3_1_risk_maps",
    "compute_ec_tss_v3_1_training_loss",
    "compute_error_conditioned_endpoint_terms",
]
