"""Zero-margin target-preservation objective layered on frozen PBDR-V4.

The V5 objective deliberately reuses :func:`compute_pbdr_v4_loss` and the
unchanged V4 role weights.  It replaces exactly three V4 terms:

* absolute candidate-only preserve -> frozen-Current smooth-peak no-drop;
* pixel-averaged foreground drop -> equal-per-component Current-positive
  support logit no-drop;
* all-background probability increase -> equal-per-sample active-background
  probability increase.

The Current reference is detached inside every replacement.  A zero margin
means that equality with Current has zero loss; it is unrelated to model
selection, whose performance acceptance margin remains ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from experiments.pbdr_v4_component_loss import (
    COMPONENT_PEAK_TEMPERATURE,
    ROLE_TVERSKY,
    ROLE_WEIGHTS,
    Role,
    _soft_component_peaks,
    compute_pbdr_v4_loss,
)


PBDR_V5_TARGET_PRESERVATION_LOSS_VERSION = (
    "pbdr_v5_target_preservation_loss_v1"
)


@dataclass(frozen=True, slots=True)
class PBDRV5TargetPreservationLossOutput:
    """V5 total and its retained/replacement loss components."""

    total: torch.Tensor
    bce: torch.Tensor
    tversky: torch.Tensor
    rescue_components: torch.Tensor
    suppress_components: torch.Tensor
    preserve_peak_no_drop: torch.Tensor
    preserve_positive_support_logit_no_drop: torch.Tensor
    active_background_increase: torch.Tensor
    neutral_delta: torch.Tensor
    v4_total_before_replacement: torch.Tensor
    replaced_v4_absolute_preserve: torch.Tensor
    replaced_v4_pixel_foreground_drop: torch.Tensor
    replaced_v4_all_background_increase: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in self.__dataclass_fields__
        }


def target_preservation_loss_manifest(role: Role) -> dict[str, object]:
    """Return the complete, non-tunable V5 loss contract for one role."""

    if role not in ROLE_WEIGHTS:
        raise ValueError(f"unsupported role: {role!r}")
    alpha, beta = ROLE_TVERSKY[role]
    return {
        "version": PBDR_V5_TARGET_PRESERVATION_LOSS_VERSION,
        "role": role,
        "base_objective": "compute_pbdr_v4_loss",
        "component_peak_temperature": COMPONENT_PEAK_TEMPERATURE,
        "tversky_alpha": alpha,
        "tversky_beta": beta,
        "weights": dict(ROLE_WEIGHTS[role]),
        "replacements": {
            "preserve": "frozen_current_smooth_component_peak_zero_margin_no_drop",
            "foreground_drop": "equal_per_preserve_component_current_positive_support_logit_no_drop",
            "background_increase": "equal_per_sample_active_background_probability_increase",
        },
        "current_reference_gradient": "detached",
        "preserve_component_reduction": "equal_per_component",
        "active_background_reduction": "active_pixels_per_sample_then_equal_samples",
        "fixed_probability_comparison": ">",
        "fixed_probability_threshold": 0.5,
        "performance_acceptance_margin": None,
    }


def _component_peak_no_drop(
    candidate_logits: torch.Tensor,
    current_logits: torch.Tensor,
    preserve_component_ids: torch.Tensor,
) -> torch.Tensor:
    """Penalize only smooth component-peak drops below frozen Current.

    Candidate and reference peaks use the exact same size-normalized V4
    smooth maximum.  There is no additive logit margin.
    """

    candidate = candidate_logits.float()
    current = current_logits.detach().float()
    candidate_peaks = _soft_component_peaks(
        candidate,
        preserve_component_ids,
    )
    current_peaks = _soft_component_peaks(
        current,
        preserve_component_ids,
    )
    if len(candidate_peaks) != len(current_peaks):
        raise RuntimeError("candidate/Current preserve component counts differ")
    if not candidate_peaks:
        return candidate.sum() * 0.0
    losses = [
        F.relu(reference_peak - candidate_peak).square()
        for candidate_peak, reference_peak in zip(
            candidate_peaks,
            current_peaks,
            strict=True,
        )
    ]
    return torch.stack(losses).mean()


def _component_positive_support_logit_no_drop(
    candidate_logits: torch.Tensor,
    current_logits: torch.Tensor,
    preserve_component_ids: torch.Tensor,
) -> torch.Tensor:
    """Average Current-positive logit no-drop equally by component.

    Within each preserve component, only pixels whose frozen Current logit is
    strictly positive contribute.  Every preserve component contributes one
    scalar, irrespective of its area.  A component without Current-positive
    support contributes an exact graph-connected zero.
    """

    candidate = candidate_logits.float()
    current = current_logits.detach().float()
    component_losses: list[torch.Tensor] = []
    for batch_index in range(candidate.shape[0]):
        component_ids = torch.unique(preserve_component_ids[batch_index])
        component_ids = component_ids[component_ids > 0]
        for component_id in component_ids:
            component = preserve_component_ids[batch_index] == component_id
            support = component & (current[batch_index] > 0.0)
            if bool(support.any()):
                drop = F.relu(
                    current[batch_index][support]
                    - candidate[batch_index][support]
                )
                component_losses.append(drop.square().mean())
            else:
                component_losses.append(candidate[batch_index].sum() * 0.0)
    if not component_losses:
        return candidate.sum() * 0.0
    return torch.stack(component_losses).mean()


def _active_background_probability_increase(
    candidate_logits: torch.Tensor,
    current_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Average actual positive background changes without zero dilution.

    The reduction first averages squared positive probability changes over
    active background pixels within each sample, then averages sample losses.
    Samples without an active background increase contribute zero.
    """

    candidate_probability = torch.sigmoid(candidate_logits.float())
    current_probability = torch.sigmoid(current_logits.detach().float())
    target_float = target.float()
    increase = candidate_probability - current_probability
    background = target_float < 0.5
    sample_losses: list[torch.Tensor] = []
    for batch_index in range(candidate_probability.shape[0]):
        active = background[batch_index] & (increase[batch_index] > 0.0)
        if bool(active.any()):
            sample_losses.append(increase[batch_index][active].square().mean())
        else:
            sample_losses.append(candidate_probability[batch_index].sum() * 0.0)
    return torch.stack(sample_losses).mean()


def compute_pbdr_v5_target_preservation_loss(
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
) -> PBDRV5TargetPreservationLossOutput:
    """Compute V4, then replace exactly three terms with V5 no-drop terms."""

    v4 = compute_pbdr_v4_loss(
        role=role,
        routed_logits=routed_logits,
        candidate_base_logits=candidate_base_logits,
        reference_current_logits=reference_current_logits,
        delta_logits=delta_logits,
        target=target,
        rescue_component_ids=rescue_component_ids,
        suppress_component_ids=suppress_component_ids,
        preserve_component_ids=preserve_component_ids,
    )
    preserve_peak_no_drop = _component_peak_no_drop(
        routed_logits,
        reference_current_logits,
        preserve_component_ids,
    )
    preserve_positive_support_logit_no_drop = (
        _component_positive_support_logit_no_drop(
            routed_logits,
            reference_current_logits,
            preserve_component_ids,
        )
    )
    active_background_increase = _active_background_probability_increase(
        routed_logits,
        reference_current_logits,
        target,
    )
    weights: Mapping[str, float] = ROLE_WEIGHTS[role]
    total = (
        v4.total
        - weights["preserve"] * v4.preserve_components
        - weights["foreground_drop"] * v4.foreground_drop
        - weights["background_increase"] * v4.background_increase
        + weights["preserve"] * preserve_peak_no_drop
        + weights["foreground_drop"]
        * preserve_positive_support_logit_no_drop
        + weights["background_increase"] * active_background_increase
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("PBDR-V5 target-preservation loss is non-finite")
    return PBDRV5TargetPreservationLossOutput(
        total=total,
        bce=v4.bce,
        tversky=v4.tversky,
        rescue_components=v4.rescue_components,
        suppress_components=v4.suppress_components,
        preserve_peak_no_drop=preserve_peak_no_drop,
        preserve_positive_support_logit_no_drop=(
            preserve_positive_support_logit_no_drop
        ),
        active_background_increase=active_background_increase,
        neutral_delta=v4.neutral_delta,
        v4_total_before_replacement=v4.total,
        replaced_v4_absolute_preserve=v4.preserve_components,
        replaced_v4_pixel_foreground_drop=v4.foreground_drop,
        replaced_v4_all_background_increase=v4.background_increase,
    )


__all__ = [
    "PBDR_V5_TARGET_PRESERVATION_LOSS_VERSION",
    "PBDRV5TargetPreservationLossOutput",
    "compute_pbdr_v5_target_preservation_loss",
    "target_preservation_loss_manifest",
]
