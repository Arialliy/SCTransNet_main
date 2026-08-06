"""Adaptive persistent-evidence residual routing for the final readout.

PBDR-V2 reuses the existing NER ``q4`` evidence and the two already trained
raw readouts ``out`` and ``d0``.  It adds no encoder, Transformer, decoder, or
supervision head.  All five state tensors are exactly zero initialized, so the
initial routed logit is bitwise anchored to ``out``::

    q = rms_normalize(stopgrad(q4))
    C = 0.05 + 0.90 * sigmoid(up(conv_conf(q)))
    Q = C * tanh(up(conv_direct(q)))
    g_r = 0.5 * tanh(rescue_strength_raw)
    g_s = 0.5 * tanh(suppression_strength_raw)
    R+ = C * relu(d0 - out)
    R- = (1 - C) * relu(out - d0)
    routed = out + Q + g_r * R+ - g_s * R-

The confidence map is always soft and bounded away from zero and one.  This
removes the hard-protection failure of PBDR-V1 while retaining an explicit
target-evidence route.  The two signed coefficients are expected to become
non-negative for the intended rescue/suppression interpretation; their signs
are reported rather than projected because exact identity, non-negativity,
and a non-zero first derivative cannot all hold at the same finite parameter
initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PBDR_V2_VERSION = "pbdr_v2_adaptive_evidence_residual_router_v1"
FORMAL_Q4_CHANNELS = 8
FORMAL_CONFIDENCE_FLOOR = 0.05
FORMAL_DIRECT_RESIDUAL_LIMIT = 1.0
FORMAL_DISAGREEMENT_STRENGTH_LIMIT = 0.5
FORMAL_RMS_EPS = 1.0e-6
FORMAL_INTERPOLATION_MODE = "bilinear"
FORMAL_ALIGN_CORNERS = False
PRODUCTION_PBDR_V2_PARAMETERS = 19
PRODUCTION_PBDR_V2_STATE_KEY_COUNT = 5
PRODUCTION_PBDR_V2_BUFFER_COUNT = 0
PBDR_V2_LOCAL_STATE_KEYS = (
    "rescue_strength_raw",
    "suppression_strength_raw",
    "confidence_projection.weight",
    "confidence_projection.bias",
    "direct_residual_projection.weight",
)


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


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _stable_detached_rms_normalize(q4: torch.Tensor) -> torch.Tensor:
    """Return detached per-sample global-RMS normalized q4 evidence."""

    detached = q4.detach()
    working = (
        detached.float()
        if detached.dtype in (torch.float16, torch.bfloat16)
        else detached
    )
    dimensions = (1, 2, 3)
    maximum = working.abs().amax(dim=dimensions, keepdim=True)
    safe_maximum = torch.where(
        maximum > 0.0,
        maximum,
        torch.ones_like(maximum),
    )
    scaled = working / safe_maximum
    scaled_rms = torch.sqrt(
        torch.clamp(
            scaled.square().mean(dim=dimensions, keepdim=True),
            min=0.0,
        )
    )
    rms = maximum * scaled_rms
    denominator = torch.clamp(
        rms,
        min=working.new_tensor(FORMAL_RMS_EPS),
    )
    normalized = (working / denominator).to(dtype=detached.dtype)
    normalized = normalized.detach()
    _require_finite_tensor(normalized, name="normalized_q4")
    if normalized.requires_grad:
        raise RuntimeError("normalized q4 must be detached")
    return normalized


@dataclass(frozen=True, slots=True)
class PBDRV2RoutingOutput:
    """Forward-local tensors for diagnostics without persistent caches."""

    routed_logits: torch.Tensor
    confidence: torch.Tensor
    direct_residual: torch.Tensor
    target_rescue: torch.Tensor
    background_suppression: torch.Tensor
    rescue_strength: torch.Tensor
    suppression_strength: torch.Tensor


class PersistentEvidenceResidualRouterV2(nn.Module):
    """Nineteen-parameter adaptive final-readout router."""

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("PBDR-V2 parameters require a floating dtype")

        # Conv constructors normally consume RNG.  Construct on CPU inside a
        # fork, force the exact zero anchor, then move without touching CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            confidence = nn.Conv2d(
                FORMAL_Q4_CHANNELS,
                1,
                kernel_size=1,
                bias=True,
            )
            direct = nn.Conv2d(
                FORMAL_Q4_CHANNELS,
                1,
                kernel_size=1,
                bias=False,
            )
        nn.init.zeros_(confidence.weight)
        nn.init.zeros_(confidence.bias)
        nn.init.zeros_(direct.weight)
        confidence.to(device=device, dtype=dtype)
        direct.to(device=device, dtype=dtype)
        self.confidence_projection = confidence
        self.direct_residual_projection = direct
        self.rescue_strength_raw = nn.Parameter(
            torch.zeros(1, device=device, dtype=dtype)
        )
        self.suppression_strength_raw = nn.Parameter(
            torch.zeros(1, device=device, dtype=dtype)
        )

        if _parameter_count(self) != PRODUCTION_PBDR_V2_PARAMETERS:
            raise RuntimeError("unexpected PBDR-V2 parameter count")
        if len(self.state_dict()) != PRODUCTION_PBDR_V2_STATE_KEY_COUNT:
            raise RuntimeError("unexpected PBDR-V2 state-key count")
        if len(tuple(self.buffers())) != PRODUCTION_PBDR_V2_BUFFER_COUNT:
            raise RuntimeError("PBDR-V2 must not register persistent buffers")

    @property
    def confidence_floor(self) -> float:
        return FORMAL_CONFIDENCE_FLOOR

    @property
    def direct_residual_limit(self) -> float:
        return FORMAL_DIRECT_RESIDUAL_LIMIT

    @property
    def disagreement_strength_limit(self) -> float:
        return FORMAL_DISAGREEMENT_STRENGTH_LIMIT

    def strengths(self) -> Tuple[torch.Tensor, torch.Tensor]:
        _require_finite_tensor(
            self.rescue_strength_raw,
            name="rescue_strength_raw",
        )
        _require_finite_tensor(
            self.suppression_strength_raw,
            name="suppression_strength_raw",
        )
        rescue = torch.tanh(self.rescue_strength_raw).mul(
            FORMAL_DISAGREEMENT_STRENGTH_LIMIT
        )
        suppression = torch.tanh(self.suppression_strength_raw).mul(
            FORMAL_DISAGREEMENT_STRENGTH_LIMIT
        )
        return rescue, suppression

    def _project_evidence(
        self,
        q4: torch.Tensor,
        output_size: tuple[int, int],
        output_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = _stable_detached_rms_normalize(q4)
        confidence_logits = self.confidence_projection(normalized)
        direct_logits = self.direct_residual_projection(normalized)
        if tuple(confidence_logits.shape[-2:]) != output_size:
            confidence_logits = F.interpolate(
                confidence_logits,
                size=output_size,
                mode=FORMAL_INTERPOLATION_MODE,
                align_corners=FORMAL_ALIGN_CORNERS,
            )
            direct_logits = F.interpolate(
                direct_logits,
                size=output_size,
                mode=FORMAL_INTERPOLATION_MODE,
                align_corners=FORMAL_ALIGN_CORNERS,
            )
        span = 1.0 - 2.0 * FORMAL_CONFIDENCE_FLOOR
        confidence = torch.sigmoid(confidence_logits).mul(span).add(
            FORMAL_CONFIDENCE_FLOOR
        )
        direct = torch.tanh(direct_logits).mul(
            FORMAL_DIRECT_RESIDUAL_LIMIT
        )
        # CUDA autocast evaluates bilinear interpolation in FP32.  Route maps
        # must return to the final-readout dtype before any residual arithmetic
        # or the nominal zero extension changes dtype and sigmoid rounding.
        confidence = confidence.to(dtype=output_dtype)
        direct = direct.to(dtype=output_dtype)
        _require_finite_tensor(confidence, name="soft_confidence")
        _require_finite_tensor(direct, name="direct_q4_residual")
        return confidence, direct

    def forward_with_diagnostics(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
    ) -> PBDRV2RoutingOutput:
        _require_bchw_float(z_out, name="z_out", channels=1)
        _require_bchw_float(z_d0, name="z_d0", channels=1)
        _require_bchw_float(
            q4,
            name="q4",
            channels=FORMAL_Q4_CHANNELS,
        )
        if z_out.shape != z_d0.shape:
            raise ValueError("z_out and z_d0 shapes must match")
        if q4.shape[0] != z_out.shape[0]:
            raise ValueError("q4 and readout batch sizes must match")
        if q4.device != z_out.device or z_d0.device != z_out.device:
            raise ValueError("PBDR-V2 tensors must share one device")
        if q4.dtype != z_out.dtype or z_d0.dtype != z_out.dtype:
            raise ValueError("PBDR-V2 tensors must share one dtype")

        confidence, direct = self._project_evidence(
            q4,
            tuple(z_out.shape[-2:]),
            z_out.dtype,
        )
        if confidence.shape != z_out.shape or direct.shape != z_out.shape:
            raise RuntimeError("PBDR-V2 projection/output shapes differ")
        direct_residual = confidence * direct
        disagreement = z_d0 - z_out
        target_rescue = confidence * F.relu(disagreement)
        background_suppression = (1.0 - confidence) * F.relu(-disagreement)
        rescue_strength, suppression_strength = self.strengths()
        # Under autocast the convolution/readout activations can be FP16 or
        # BF16 while the scalar parameters remain FP32.  Explicitly align the
        # scalars with the routed activation so an exact-zero router cannot
        # silently promote the final logit and break the identity anchor.
        rescue_strength = rescue_strength.to(
            device=z_out.device,
            dtype=z_out.dtype,
        )
        suppression_strength = suppression_strength.to(
            device=z_out.device,
            dtype=z_out.dtype,
        )
        routed = (
            z_out
            + direct_residual
            + rescue_strength.view(1, 1, 1, 1) * target_rescue
            - suppression_strength.view(1, 1, 1, 1)
            * background_suppression
        )
        _require_finite_tensor(routed, name="routed_logits")
        return PBDRV2RoutingOutput(
            routed_logits=routed,
            confidence=confidence,
            direct_residual=direct_residual,
            target_rescue=target_rescue,
            background_suppression=background_suppression,
            rescue_strength=rescue_strength,
            suppression_strength=suppression_strength,
        )

    def forward(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(z_out, z_d0, q4).routed_logits

    def architecture_manifest(self) -> Dict[str, Any]:
        return {
            "pbdr_v2_version": PBDR_V2_VERSION,
            "q4_channels": FORMAL_Q4_CHANNELS,
            "q4_gradient_boundary": "stop_gradient_before_router",
            "q4_normalization": "per_sample_global_rms",
            "rms_eps": FORMAL_RMS_EPS,
            "confidence_projection": "conv1x1_8_to_1_with_bias",
            "confidence_floor": FORMAL_CONFIDENCE_FLOOR,
            "confidence_ceiling": 1.0 - FORMAL_CONFIDENCE_FLOOR,
            "confidence_is_soft": True,
            "direct_residual_projection": "conv1x1_8_to_1_no_bias",
            "direct_residual_limit": FORMAL_DIRECT_RESIDUAL_LIMIT,
            "rescue_strength": "0.5*tanh(rescue_strength_raw)",
            "suppression_strength": (
                "0.5*tanh(suppression_strength_raw)"
            ),
            "disagreement_strength_limit": (
                FORMAL_DISAGREEMENT_STRENGTH_LIMIT
            ),
            "alignment": "bilinear_align_corners_false",
            "zero_anchor": "routed_logits_equals_z_out_exactly",
            "parameters": PRODUCTION_PBDR_V2_PARAMETERS,
            "state_key_count": PRODUCTION_PBDR_V2_STATE_KEY_COUNT,
            "persistent_buffer_count": PRODUCTION_PBDR_V2_BUFFER_COUNT,
        }


def validate_formal_pbdr_v2_router(
    module: nn.Module,
    *,
    require_zero_initialization: bool,
) -> Dict[str, Any]:
    if type(module) is not PersistentEvidenceResidualRouterV2:
        raise TypeError("formal PBDR-V2 router must use the exact class")
    if _parameter_count(module) != PRODUCTION_PBDR_V2_PARAMETERS:
        raise RuntimeError("formal PBDR-V2 parameter count differs")
    state = module.state_dict()
    if tuple(state) != PBDR_V2_LOCAL_STATE_KEYS:
        raise RuntimeError("formal PBDR-V2 state keys differ")
    if len(tuple(module.buffers())) != PRODUCTION_PBDR_V2_BUFFER_COUNT:
        raise RuntimeError("formal PBDR-V2 buffer count differs")
    reference = next(module.parameters())
    for name, parameter in module.named_parameters():
        if parameter.device != reference.device or parameter.dtype != reference.dtype:
            raise RuntimeError(f"formal PBDR-V2 parameter {name} placement differs")
        _require_finite_tensor(parameter, name=name)
        if require_zero_initialization and int(torch.count_nonzero(parameter)) != 0:
            raise RuntimeError(f"formal PBDR-V2 parameter {name} is not zero")
    return module.architecture_manifest()


__all__ = [
    "FORMAL_ALIGN_CORNERS",
    "FORMAL_CONFIDENCE_FLOOR",
    "FORMAL_DIRECT_RESIDUAL_LIMIT",
    "FORMAL_DISAGREEMENT_STRENGTH_LIMIT",
    "FORMAL_INTERPOLATION_MODE",
    "FORMAL_Q4_CHANNELS",
    "FORMAL_RMS_EPS",
    "PBDR_V2_LOCAL_STATE_KEYS",
    "PBDR_V2_VERSION",
    "PBDRV2RoutingOutput",
    "PRODUCTION_PBDR_V2_BUFFER_COUNT",
    "PRODUCTION_PBDR_V2_PARAMETERS",
    "PRODUCTION_PBDR_V2_STATE_KEY_COUNT",
    "PersistentEvidenceResidualRouterV2",
    "validate_formal_pbdr_v2_router",
]
