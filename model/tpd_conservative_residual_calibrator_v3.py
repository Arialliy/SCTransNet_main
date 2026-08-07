"""Conservative, exactly identity-initialized final-logit calibration.

PBDR-V3 treats the coarse ``q4`` evidence and auxiliary ``d0`` readout as
context only.  Neither tensor is allowed to inject a direct residual.  Two
non-negative spatial gates are initialized identically, so their difference
is exactly zero while both terminal gate branches retain a non-zero first
derivative::

    uncertainty = 4 * sigmoid(out) * (1 - sigmoid(out))
    budget = rho + (1 - rho) * uncertainty
    delta = limit * budget * (rescue_gate - suppression_gate)
    routed = out + delta

The q4 normalization is spatially centered per channel and uses an RMS floor;
weak evidence is therefore never amplified to unit energy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


PBDR_V3_VERSION = "pbdr_v3_conservative_twin_gate_calibrator_v1"
FORMAL_Q4_CHANNELS = 8
FORMAL_LOCAL_CHANNELS = 32
FORMAL_HIDDEN_CHANNELS = 16
FORMAL_RESIDUAL_LIMIT = 0.15
FORMAL_EVIDENCE_FLOOR = 1.0
FORMAL_UNCERTAINTY_FLOOR = 0.25
FORMAL_GATE_BIAS_INIT = -4.0
FORMAL_INTERPOLATION_MODE = "bilinear"
FORMAL_ALIGN_CORNERS = False
FORMAL_DETACH_LOCAL_FEATURE = True
PRODUCTION_PBDR_V3_PARAMETERS = 6_018
PRODUCTION_PBDR_V3_STATE_KEY_COUNT = 6
PRODUCTION_PBDR_V3_BUFFER_COUNT = 0
PBDR_V3_LOCAL_STATE_KEYS = (
    "local_projection.0.weight",
    "q_projection.0.weight",
    "routing_trunk.0.weight",
    "routing_trunk.0.bias",
    "routing_trunk.2.weight",
    "routing_trunk.2.bias",
)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if value.device.type == "cuda":
        torch._assert_async(
            torch.isfinite(value).all(),
            f"{name} contains non-finite values",
        )
        return
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _require_bchw_float(
    value: torch.Tensor,
    *,
    name: str,
    channels: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must be BCHW, got {tuple(value.shape)}")
    if min(value.shape) < 1:
        raise ValueError(f"{name} dimensions must be positive")
    if value.shape[1] != channels:
        raise ValueError(
            f"{name} requires C={channels}, got C={value.shape[1]}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    _require_finite_tensor(value, name=name)


@dataclass(frozen=True, slots=True)
class PBDRV3RoutingOutput:
    """Forward-local routing tensors; no diagnostic state is cached."""

    routed_logits: torch.Tensor
    delta_logits: torch.Tensor
    rescue_gate: torch.Tensor
    suppression_gate: torch.Tensor
    uncertainty: torch.Tensor


class ConservativeResidualCalibratorV3(nn.Module):
    """Small bounded calibrator for a trained Current checkpoint.

    Args:
        q_channels: Channels in the detached coarse q4 evidence.
        local_channels: Channels in the full-resolution decoder feature.
        hidden_channels: Width of each context projection and routing trunk.
        residual_limit: Maximum absolute logit correction.
        evidence_floor: Per-channel centered-RMS denominator floor.
        uncertainty_floor: Residual-budget fraction away from ``p=0.5``.
        gate_bias_init: Equal finite initialization of both gate logits.
        detach_local_feature: Stop the context path at the decoder feature.
            The additive routed-logit path remains differentiable to ``out``.
    """

    def __init__(
        self,
        *,
        q_channels: int = FORMAL_Q4_CHANNELS,
        local_channels: int = FORMAL_LOCAL_CHANNELS,
        hidden_channels: int = FORMAL_HIDDEN_CHANNELS,
        residual_limit: float = FORMAL_RESIDUAL_LIMIT,
        evidence_floor: float = FORMAL_EVIDENCE_FLOOR,
        uncertainty_floor: float = FORMAL_UNCERTAINTY_FLOOR,
        gate_bias_init: float = FORMAL_GATE_BIAS_INIT,
        detach_local_feature: bool = FORMAL_DETACH_LOCAL_FEATURE,
    ) -> None:
        super().__init__()
        if q_channels < 1 or local_channels < 1 or hidden_channels < 1:
            raise ValueError("all channel counts must be positive")
        if not math.isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("residual_limit must be finite and positive")
        if not math.isfinite(evidence_floor) or evidence_floor <= 0.0:
            raise ValueError("evidence_floor must be finite and positive")
        if not 0.0 <= uncertainty_floor <= 1.0:
            raise ValueError("uncertainty_floor must be in [0, 1]")
        if not math.isfinite(gate_bias_init):
            raise ValueError("gate_bias_init must be finite")

        self.q_channels = int(q_channels)
        self.local_channels = int(local_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_limit = float(residual_limit)
        self.evidence_floor = float(evidence_floor)
        self.uncertainty_floor = float(uncertainty_floor)
        self.gate_bias_init = float(gate_bias_init)
        self.detach_local_feature = bool(detach_local_feature)

        self.local_projection = nn.Sequential(
            nn.Conv2d(
                self.local_channels,
                self.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
        )
        self.q_projection = nn.Sequential(
            nn.Conv2d(
                self.q_channels,
                self.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
        )

        # p_out, p_d0, signed disagreement, absolute disagreement, and
        # uncertainty contribute five scalar context channels.
        context_channels = 2 * self.hidden_channels + 5
        self.routing_trunk = nn.Sequential(
            nn.Conv2d(
                context_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_channels,
                2,
                kernel_size=1,
                bias=True,
            ),
        )
        final_projection = self.routing_trunk[-1]
        if not isinstance(final_projection, nn.Conv2d):
            raise RuntimeError("unexpected routing trunk")
        nn.init.zeros_(final_projection.weight)
        nn.init.constant_(final_projection.bias, self.gate_bias_init)

    def _safe_q4(self, q4: torch.Tensor) -> torch.Tensor:
        """Center q4 per channel and normalize without amplifying weak maps."""

        detached = q4.detach()
        working = (
            detached.float()
            if detached.dtype in (torch.float16, torch.bfloat16)
            else detached
        )
        centered = working - working.mean(dim=(2, 3), keepdim=True)
        rms = torch.sqrt(
            centered.square().mean(dim=(2, 3), keepdim=True) + 1.0e-8
        )
        normalized = centered / rms.clamp_min(self.evidence_floor)
        normalized = normalized.to(dtype=detached.dtype).detach()
        _require_finite_tensor(normalized, name="normalized_q4")
        if normalized.requires_grad:
            raise RuntimeError("normalized q4 must be detached")
        return normalized

    def forward_with_diagnostics(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> PBDRV3RoutingOutput:
        _require_bchw_float(z_out, name="z_out", channels=1)
        _require_bchw_float(z_d0, name="z_d0", channels=1)
        _require_bchw_float(q4, name="q4", channels=self.q_channels)
        _require_bchw_float(
            local_feature,
            name="local_feature",
            channels=self.local_channels,
        )
        if z_out.shape != z_d0.shape:
            raise ValueError("z_out and z_d0 shapes must match")
        if q4.shape[0] != z_out.shape[0]:
            raise ValueError("q4 and readout batch sizes must match")
        if (
            local_feature.shape[0] != z_out.shape[0]
            or local_feature.shape[-2:] != z_out.shape[-2:]
        ):
            raise ValueError(
                "local_feature must share batch and spatial shape with z_out"
            )
        routing_inputs = (z_d0, q4, local_feature)
        if any(value.device != z_out.device for value in routing_inputs):
            raise ValueError("PBDR-V3 tensors must share one device")
        if any(value.dtype != z_out.dtype for value in routing_inputs):
            raise ValueError("PBDR-V3 tensors must share one dtype")

        local = (
            local_feature.detach()
            if self.detach_local_feature
            else local_feature
        )
        local_context = self.local_projection(local)
        q_context = self.q_projection(self._safe_q4(q4))
        if q_context.shape[-2:] != z_out.shape[-2:]:
            q_context = F.interpolate(
                q_context,
                size=z_out.shape[-2:],
                mode=FORMAL_INTERPOLATION_MODE,
                align_corners=FORMAL_ALIGN_CORNERS,
            )
        # Autocast can evaluate interpolation in FP32.  Restore the final
        # readout dtype before concatenation and residual arithmetic.
        local_context = local_context.to(dtype=z_out.dtype)
        q_context = q_context.to(dtype=z_out.dtype)

        # d0/out are immutable context anchors.  In a later fine-tuning stage,
        # z_out still receives the additive routed-logit gradient below.
        p_out = torch.sigmoid(z_out.detach())
        p_d0 = torch.sigmoid(z_d0.detach())
        disagreement = p_d0 - p_out
        uncertainty = 4.0 * p_out * (1.0 - p_out)
        uncertainty = uncertainty.clamp(min=0.0, max=1.0)
        residual_budget = self.uncertainty_floor + (
            1.0 - self.uncertainty_floor
        ) * uncertainty

        context = torch.cat(
            (
                local_context,
                q_context,
                p_out,
                p_d0,
                disagreement,
                disagreement.abs(),
                uncertainty,
            ),
            dim=1,
        )
        gates = torch.sigmoid(self.routing_trunk(context))
        rescue_gate = gates[:, 0:1]
        suppression_gate = gates[:, 1:2]

        delta = (
            self.residual_limit
            * residual_budget
            * (rescue_gate - suppression_gate)
        )
        delta = delta.clamp(
            min=-self.residual_limit,
            max=self.residual_limit,
        )
        routed = z_out + delta
        _require_finite_tensor(routed, name="routed_logits")
        return PBDRV3RoutingOutput(
            routed_logits=routed,
            delta_logits=delta,
            rescue_gate=rescue_gate,
            suppression_gate=suppression_gate,
            uncertainty=uncertainty,
        )

    def forward(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            z_out,
            z_d0,
            q4,
            local_feature,
        ).routed_logits

    def architecture_manifest(self) -> Dict[str, Any]:
        return {
            "pbdr_v3_version": PBDR_V3_VERSION,
            "q4_channels": self.q_channels,
            "local_channels": self.local_channels,
            "hidden_channels": self.hidden_channels,
            "q4_gradient_boundary": "stop_gradient_before_calibrator",
            "q4_normalization": "per_channel_spatial_centered_rms_floor",
            "q4_weak_evidence_amplified": False,
            "local_feature_gradient_boundary": (
                "stop_gradient_before_calibrator"
                if self.detach_local_feature
                else "open"
            ),
            "readout_context_gradient_boundary": "stop_gradient",
            "direct_q4_residual": False,
            "direct_d0_residual": False,
            "gate_mapping": "sigmoid_nonnegative_twin_gate",
            "gate_bias_initialization": self.gate_bias_init,
            "residual_formula": "L*(rho+(1-rho)*U)*(G_r-G_s)",
            "uncertainty_formula": "4*p_out*(1-p_out)",
            "residual_limit": self.residual_limit,
            "evidence_floor": self.evidence_floor,
            "uncertainty_floor": self.uncertainty_floor,
            "alignment": "bilinear_align_corners_false",
            "zero_anchor": "equal_finite_twin_gates_current_final_exact",
            "terminal_gate_first_derivative_nonzero": True,
            "parameters": _parameter_count(self),
            "state_key_count": len(self.state_dict()),
            "persistent_buffer_count": len(tuple(self.buffers())),
        }


def validate_formal_pbdr_v3_calibrator(
    module: nn.Module,
    *,
    require_identity_initialization: bool,
) -> Dict[str, Any]:
    """Validate the fixed production calibrator and optional identity state."""

    if type(module) is not ConservativeResidualCalibratorV3:
        raise TypeError("formal PBDR-V3 calibrator must use the exact class")
    expected_attributes = {
        "q_channels": FORMAL_Q4_CHANNELS,
        "local_channels": FORMAL_LOCAL_CHANNELS,
        "hidden_channels": FORMAL_HIDDEN_CHANNELS,
        "residual_limit": FORMAL_RESIDUAL_LIMIT,
        "evidence_floor": FORMAL_EVIDENCE_FLOOR,
        "uncertainty_floor": FORMAL_UNCERTAINTY_FLOOR,
        "gate_bias_init": FORMAL_GATE_BIAS_INIT,
        "detach_local_feature": FORMAL_DETACH_LOCAL_FEATURE,
    }
    for name, expected in expected_attributes.items():
        if getattr(module, name) != expected:
            raise RuntimeError(f"formal PBDR-V3 attribute {name!r} differs")
    if _parameter_count(module) != PRODUCTION_PBDR_V3_PARAMETERS:
        raise RuntimeError("formal PBDR-V3 parameter count differs")
    state = module.state_dict()
    if tuple(state) != PBDR_V3_LOCAL_STATE_KEYS:
        raise RuntimeError("formal PBDR-V3 state keys differ")
    if len(tuple(module.buffers())) != PRODUCTION_PBDR_V3_BUFFER_COUNT:
        raise RuntimeError("formal PBDR-V3 persistent buffers differ")

    reference = next(module.parameters())
    for name, parameter in module.named_parameters():
        if parameter.device != reference.device:
            raise RuntimeError(f"formal PBDR-V3 parameter {name} device differs")
        if parameter.dtype != reference.dtype:
            raise RuntimeError(f"formal PBDR-V3 parameter {name} dtype differs")
        _require_finite_tensor(parameter, name=name)

    if require_identity_initialization:
        final_projection = module.routing_trunk[-1]
        if not isinstance(final_projection, nn.Conv2d):
            raise RuntimeError("formal PBDR-V3 terminal projection differs")
        if int(torch.count_nonzero(final_projection.weight)) != 0:
            raise RuntimeError("formal PBDR-V3 terminal weight is not zero")
        expected_bias = torch.full_like(
            final_projection.bias,
            FORMAL_GATE_BIAS_INIT,
        )
        if not torch.equal(final_projection.bias, expected_bias):
            raise RuntimeError("formal PBDR-V3 twin-gate biases differ")

    return module.architecture_manifest()


__all__ = [
    "ConservativeResidualCalibratorV3",
    "FORMAL_ALIGN_CORNERS",
    "FORMAL_DETACH_LOCAL_FEATURE",
    "FORMAL_EVIDENCE_FLOOR",
    "FORMAL_GATE_BIAS_INIT",
    "FORMAL_HIDDEN_CHANNELS",
    "FORMAL_INTERPOLATION_MODE",
    "FORMAL_LOCAL_CHANNELS",
    "FORMAL_Q4_CHANNELS",
    "FORMAL_RESIDUAL_LIMIT",
    "FORMAL_UNCERTAINTY_FLOOR",
    "PBDR_V3_LOCAL_STATE_KEYS",
    "PBDR_V3_VERSION",
    "PBDRV3RoutingOutput",
    "PRODUCTION_PBDR_V3_BUFFER_COUNT",
    "PRODUCTION_PBDR_V3_PARAMETERS",
    "PRODUCTION_PBDR_V3_STATE_KEY_COUNT",
    "validate_formal_pbdr_v3_calibrator",
]
