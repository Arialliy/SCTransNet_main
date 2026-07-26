"""Loss routing for the composed TPD-SCTransNet training contract.

SCTransNet produces sigmoid probability maps for its segmentation objective,
whereas the optional target-survival heads produce low-resolution logits.
Those two families deliberately use different loss functions and must never be
flattened into one deep-supervision tuple.

The public entry point keeps the original segmentation objective unchanged:

    L = sum_j BCE(segmentation_j, target)
        + lambda_s * sum_i BCEWithLogits(survival_i, Y16)

where ``Y16`` is a fixed 16x max-pooled binary target.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import ContextManager

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tpd_forward_contract import ForwardOutput, TPDForwardOutput, legacy_output


SURVIVAL_DOWNSAMPLE = 16


class TPDTrainingLossError(ValueError):
    """The model output and requested training objective are incompatible."""


@dataclass(frozen=True, slots=True)
class TPDTrainingLoss:
    """Scalar loss components returned to the modular training loop."""

    total: torch.Tensor
    segmentation: torch.Tensor
    survival: torch.Tensor
    segmentation_terms: tuple[torch.Tensor, ...]
    survival_terms: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("total", self.total),
            ("segmentation", self.segmentation),
            ("survival", self.survival),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise TPDTrainingLossError(f"{name} must be a scalar Tensor")
            if not torch.isfinite(value):
                raise FloatingPointError(f"{name} loss is non-finite")
        for name, terms in (
            ("segmentation_terms", self.segmentation_terms),
            ("survival_terms", self.survival_terms),
        ):
            for index, value in enumerate(terms):
                if not isinstance(value, torch.Tensor) or value.ndim != 0:
                    raise TPDTrainingLossError(
                        f"{name}[{index}] must be a scalar Tensor"
                    )
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        f"{name}[{index}] loss is non-finite"
                    )


def _disabled_autocast(device_type: str) -> ContextManager[object]:
    if device_type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def _validate_binary_map(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TPDTrainingLossError(f"{name} must be a Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise TPDTrainingLossError(
            f"{name} must have shape Bx1xHxW, got {tuple(value.shape)}"
        )
    if min(value.shape) <= 0:
        raise TPDTrainingLossError(f"{name} dimensions must be positive")
    if not value.is_floating_point():
        raise TPDTrainingLossError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    detached = value.detach()
    if torch.any(detached < 0) or torch.any(detached > 1):
        raise TPDTrainingLossError(f"{name} values must be in [0, 1]")


def build_survival_target(
    segmentation_target: torch.Tensor,
    *,
    downsample: int = SURVIVAL_DOWNSAMPLE,
) -> torch.Tensor:
    """Build the exact fixed-stride ``Y16 = MaxPool16(Y)`` target."""

    _validate_binary_map("segmentation_target", segmentation_target)
    if not isinstance(downsample, int) or isinstance(downsample, bool) or downsample < 1:
        raise TPDTrainingLossError("downsample must be a positive integer")
    height, width = segmentation_target.shape[-2:]
    if height % downsample or width % downsample:
        raise TPDTrainingLossError(
            "segmentation target spatial size must be divisible by "
            f"{downsample}, got {(height, width)}"
        )
    with _disabled_autocast(segmentation_target.device.type):
        return F.max_pool2d(
            segmentation_target.float(),
            kernel_size=downsample,
            stride=downsample,
        )


def _segmentation_maps(output: ForwardOutput) -> tuple[torch.Tensor, ...]:
    segmentation = legacy_output(output)
    return (segmentation,) if isinstance(segmentation, torch.Tensor) else segmentation


def _validate_segmentation_pair(
    probability: torch.Tensor,
    target: torch.Tensor,
    index: int,
) -> None:
    if probability.shape != target.shape:
        raise TPDTrainingLossError(
            f"segmentation[{index}] shape {tuple(probability.shape)} does not "
            f"match target {tuple(target.shape)}"
        )
    if probability.device != target.device:
        raise TPDTrainingLossError(
            f"segmentation[{index}] and target must use the same device"
        )
    if not torch.isfinite(probability).all():
        raise FloatingPointError(
            f"segmentation[{index}] contains non-finite values"
        )


def _validated_survival_weight(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TPDTrainingLossError("survival_weight must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise TPDTrainingLossError(
            "survival_weight must be finite and non-negative"
        )
    return result


def _pos_weight_tensor(
    value: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TPDTrainingLossError("survival_pos_weight must be scalar")
        number = float(value.detach().cpu().item())
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TPDTrainingLossError("survival_pos_weight must be numeric")
    else:
        number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise TPDTrainingLossError(
            "survival_pos_weight must be finite and positive"
        )
    return torch.tensor(
        number,
        dtype=torch.float32,
        device=reference.device,
    ).reshape(1, 1, 1)


def compute_tpd_training_loss(
    output: ForwardOutput,
    segmentation_target: torch.Tensor,
    segmentation_criterion: nn.Module,
    *,
    survival_weight: float = 0.0,
    survival_pos_weight: float | torch.Tensor = 1.0,
) -> TPDTrainingLoss:
    """Compute the separated segmentation and optional survival objectives.

    ``survival_weight == 0`` is an exact no-auxiliary path: no survival target
    is built and no survival logit enters the returned total.
    """

    if not isinstance(segmentation_criterion, nn.Module):
        raise TPDTrainingLossError("segmentation_criterion must be an nn.Module")
    _validate_binary_map("segmentation_target", segmentation_target)
    weight = _validated_survival_weight(survival_weight)
    maps = _segmentation_maps(output)

    segmentation_terms: list[torch.Tensor] = []
    with _disabled_autocast(segmentation_target.device.type):
        float_target = segmentation_target.float()
        for index, probability in enumerate(maps):
            _validate_segmentation_pair(probability, segmentation_target, index)
            term = segmentation_criterion(probability.float(), float_target)
            if not isinstance(term, torch.Tensor) or term.ndim != 0:
                raise TPDTrainingLossError(
                    "segmentation criterion must return a scalar Tensor"
                )
            segmentation_terms.append(term)
        # Keep the exact Python addition order used by the frozen baseline
        # ``deep_supervision_loss``.  A stacked reduction can differ by one
        # FP32 ULP and would make the nominal no-auxiliary path diverge.
        segmentation_loss = sum(segmentation_terms)

    zero = segmentation_loss.new_zeros(())
    if weight == 0.0:
        return TPDTrainingLoss(
            total=segmentation_loss,
            segmentation=segmentation_loss,
            survival=zero,
            segmentation_terms=tuple(segmentation_terms),
            survival_terms=(),
        )

    if not isinstance(output, TPDForwardOutput):
        raise TPDTrainingLossError(
            "positive survival_weight requires a structured TPDForwardOutput"
        )
    logits = output.survival_logits
    if logits is None:
        raise TPDTrainingLossError(
            "positive survival_weight requires both survival logits"
        )
    # The public objective has one authoritative auxiliary target.  Do not
    # accept a caller-supplied substitute that could silently bypass Y16.
    survival_target = build_survival_target(segmentation_target)

    pos_weight = _pos_weight_tensor(survival_pos_weight, logits[0])
    survival_terms: list[torch.Tensor] = []
    with _disabled_autocast(segmentation_target.device.type):
        float_survival_target = survival_target.float()
        for index, logit in enumerate(logits):
            if logit.shape != float_survival_target.shape:
                raise TPDTrainingLossError(
                    f"survival_logits[{index}] shape {tuple(logit.shape)} "
                    "does not match survival_target "
                    f"{tuple(float_survival_target.shape)}"
                )
            if logit.device != segmentation_target.device:
                raise TPDTrainingLossError(
                    f"survival_logits[{index}] and target must use the same device"
                )
            if not torch.isfinite(logit).all():
                raise FloatingPointError(
                    f"survival_logits[{index}] contains non-finite values"
                )
            survival_terms.append(
                F.binary_cross_entropy_with_logits(
                    logit.float(),
                    float_survival_target,
                    pos_weight=pos_weight,
                    reduction="mean",
                )
            )
        survival_loss = sum(survival_terms)
        total = segmentation_loss + weight * survival_loss

    return TPDTrainingLoss(
        total=total,
        segmentation=segmentation_loss,
        survival=survival_loss,
        segmentation_terms=tuple(segmentation_terms),
        survival_terms=tuple(survival_terms),
    )


__all__ = [
    "SURVIVAL_DOWNSAMPLE",
    "TPDTrainingLoss",
    "TPDTrainingLossError",
    "build_survival_target",
    "compute_tpd_training_loss",
]
