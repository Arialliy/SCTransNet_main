"""Pre-registered zero-training recalibration of a PBDR-V3 residual.

This changes residual model parameters, not the fixed probability threshold.
The grid includes the exact Current anchor ``(0, 0, 0)`` and the exact
PBDR-V3 anchor ``(1, 1, 0)``.  Selection belongs on the frozen internal
validation split and uses the baseline-envelope selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterator

import torch
import torch.nn.functional as F


POSITIVE_SCALES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
NEGATIVE_SCALES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
BIASES: tuple[float, ...] = (
    -0.15,
    -0.10,
    -0.05,
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
)


@dataclass(frozen=True, slots=True)
class ResidualCalibration:
    positive_scale: float
    negative_scale: float
    bias: float

    def __post_init__(self) -> None:
        for name in ("positive_scale", "negative_scale", "bias"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.positive_scale < 0.0 or self.negative_scale < 0.0:
            raise ValueError("residual scales must be non-negative")

    @property
    def name(self) -> str:
        return (
            f"pos{self.positive_scale:g}_neg{self.negative_scale:g}"
            f"_bias{self.bias:+g}"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "positive_scale": float(self.positive_scale),
            "negative_scale": float(self.negative_scale),
            "bias": float(self.bias),
        }


CURRENT_ANCHOR = ResidualCalibration(0.0, 0.0, 0.0)
PBDR_V3_ANCHOR = ResidualCalibration(1.0, 1.0, 0.0)


def calibration_grid() -> tuple[ResidualCalibration, ...]:
    """Return the immutable 7 x 6 x 9 grid in deterministic order."""

    return tuple(
        ResidualCalibration(positive, negative, bias)
        for positive, negative, bias in itertools.product(
            POSITIVE_SCALES,
            NEGATIVE_SCALES,
            BIASES,
        )
    )


def iter_calibration_grid() -> Iterator[ResidualCalibration]:
    yield from calibration_grid()


def _validate_logits(
    base_logits: torch.Tensor,
    delta_logits: torch.Tensor,
) -> None:
    if not isinstance(base_logits, torch.Tensor) or not isinstance(
        delta_logits, torch.Tensor
    ):
        raise TypeError("base_logits and delta_logits must be tensors")
    if base_logits.shape != delta_logits.shape:
        raise ValueError("base_logits and delta_logits must share shape")
    if base_logits.ndim != 4 or base_logits.shape[1] != 1:
        raise ValueError("calibration logits must be BCHW with C=1")
    if not base_logits.is_floating_point() or not delta_logits.is_floating_point():
        raise TypeError("calibration logits must be floating point")
    if base_logits.device != delta_logits.device:
        raise ValueError("calibration logits must share device")
    if base_logits.dtype != delta_logits.dtype:
        raise ValueError("calibration logits must share dtype")
    if not bool(torch.isfinite(base_logits).all()) or not bool(
        torch.isfinite(delta_logits).all()
    ):
        raise FloatingPointError("calibration logits contain non-finite values")


def apply_residual_calibration(
    base_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    config: ResidualCalibration,
) -> torch.Tensor:
    """Apply separate positive/negative scaling while keeping threshold 0.5."""

    _validate_logits(base_logits, delta_logits)
    if not isinstance(config, ResidualCalibration):
        raise TypeError("config must be ResidualCalibration")
    if config == CURRENT_ANCHOR:
        return base_logits
    if config == PBDR_V3_ANCHOR:
        return base_logits + delta_logits
    routed = (
        base_logits
        + float(config.positive_scale) * F.relu(delta_logits)
        - float(config.negative_scale) * F.relu(-delta_logits)
        + float(config.bias)
    )
    if not bool(torch.isfinite(routed).all()):
        raise FloatingPointError("recalibrated logits contain non-finite values")
    return routed


__all__ = [
    "BIASES",
    "CURRENT_ANCHOR",
    "NEGATIVE_SCALES",
    "PBDR_V3_ANCHOR",
    "POSITIVE_SCALES",
    "ResidualCalibration",
    "apply_residual_calibration",
    "calibration_grid",
    "iter_calibration_grid",
]
