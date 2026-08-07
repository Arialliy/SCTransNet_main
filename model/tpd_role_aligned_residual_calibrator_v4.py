"""Role-aligned, exactly identity-initialized residual calibration.

PBDR-V4 consumes only named tensors from a frozen Current graph.  The four
deep-supervision logits are explicit arguments so their semantic order cannot
be changed by passing an untyped sequence.  Role and logit limits are persistent
buffers and are guarded during ``load_state_dict``; a state from another role
or another capacity configuration is rejected even when ``strict=False``.

Runtime finite-value scans are optional.  They are disabled by default so the
formal CUDA path does not introduce host synchronization on every forward.
Shape, channel, dtype and device contracts are always checked.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Literal, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


Role = Literal["best_miou", "best_pd"]

PBDR_V4_VERSION = "pbdr_v4_role_aligned_component_calibrator_v1"
FORMAL_Q4_CHANNELS = 8
FORMAL_LOCAL_CHANNELS = 32
FORMAL_HIDDEN_CHANNELS = 24
FORMAL_EVIDENCE_FLOOR = 1.0
FORMAL_DETACH_CONTEXT = True
FORMAL_DEBUG_VALIDATE_FINITE = False
FORMAL_POSITIVE_LIMITS: Mapping[str, float] = {
    "best_miou": 0.60,
    "best_pd": 1.25,
}
FORMAL_NEGATIVE_LIMITS: Mapping[str, float] = {
    "best_miou": 0.50,
    "best_pd": 0.20,
}
ROLE_CODES: Mapping[str, int] = {
    "best_miou": 0,
    "best_pd": 1,
}

PRODUCTION_PBDR_V4_PARAMETERS = 11_497
PRODUCTION_PBDR_V4_STATE_KEY_COUNT = 27
PRODUCTION_PBDR_V4_BUFFER_COUNT = 3
PBDR_V4_LOCAL_STATE_KEYS = (
    "role_code",
    "positive_limit",
    "negative_limit",
    "local_projection.0.weight",
    "local_projection.1.weight",
    "local_projection.1.bias",
    "q_projection.0.weight",
    "q_projection.1.weight",
    "q_projection.1.bias",
    "context_stem.0.weight",
    "context_stem.0.bias",
    "context_stem.1.weight",
    "context_stem.1.bias",
    "context_branches.0.0.weight",
    "context_branches.0.1.weight",
    "context_branches.0.1.bias",
    "context_branches.1.0.weight",
    "context_branches.1.1.weight",
    "context_branches.1.1.bias",
    "context_branches.2.0.weight",
    "context_branches.2.1.weight",
    "context_branches.2.1.bias",
    "context_fuse.0.weight",
    "context_fuse.1.weight",
    "context_fuse.1.bias",
    "residual_head.weight",
    "residual_head.bias",
)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


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


def _semantic_buffer_equal(
    checkpoint_value: Any,
    expected_value: torch.Tensor,
) -> bool:
    if not isinstance(checkpoint_value, torch.Tensor):
        return False
    if checkpoint_value.shape != expected_value.shape:
        return False
    checkpoint_cpu = checkpoint_value.detach().cpu().to(dtype=expected_value.dtype)
    expected_cpu = expected_value.detach().cpu()
    return torch.equal(checkpoint_cpu, expected_cpu)


@dataclass(frozen=True, slots=True)
class PBDRV4RoutingOutput:
    """Forward-local tensors used by the V4 loss and diagnostics."""

    routed_logits: torch.Tensor
    delta_logits: torch.Tensor
    signed_score: torch.Tensor
    rescue_budget: torch.Tensor
    suppression_budget: torch.Tensor
    uncertainty: torch.Tensor
    consensus: torch.Tensor


class RoleAlignedResidualCalibratorV4(nn.Module):
    """Role-specific bounded residual head with an exact Current anchor.

    ``best_pd`` has a larger positive than negative logit capacity, while
    ``best_miou`` uses a balanced capacity.  The terminal projection is exactly
    zero at construction, so every finite input returns ``z_out`` unchanged.

    At the zero-score kink, the branch with the larger local evidence budget is
    used for the subgradient.  This leaves the forward identity untouched while
    allowing a saturated false-positive location to receive a first-step
    suppression gradient.
    """

    _SEMANTIC_BUFFER_NAMES = (
        "role_code",
        "positive_limit",
        "negative_limit",
    )

    def __init__(
        self,
        *,
        role: Role,
        q_channels: int = FORMAL_Q4_CHANNELS,
        local_channels: int = FORMAL_LOCAL_CHANNELS,
        hidden_channels: int = FORMAL_HIDDEN_CHANNELS,
        positive_limit: float | None = None,
        negative_limit: float | None = None,
        evidence_floor: float = FORMAL_EVIDENCE_FLOOR,
        detach_context: bool = FORMAL_DETACH_CONTEXT,
        debug_validate_finite: bool = FORMAL_DEBUG_VALIDATE_FINITE,
    ) -> None:
        super().__init__()
        if role not in ROLE_CODES:
            raise ValueError(f"unsupported role: {role!r}")
        channel_values = (q_channels, local_channels, hidden_channels)
        if any(type(value) is not int or value < 1 for value in channel_values):
            raise ValueError("all channel counts must be positive integers")
        if hidden_channels % 6 != 0:
            raise ValueError("hidden_channels must be divisible by 6")
        if isinstance(evidence_floor, bool) or not math.isfinite(evidence_floor):
            raise ValueError("evidence_floor must be finite and positive")
        if evidence_floor <= 0.0:
            raise ValueError("evidence_floor must be finite and positive")
        if type(detach_context) is not bool:
            raise TypeError("detach_context must be bool")
        if type(debug_validate_finite) is not bool:
            raise TypeError("debug_validate_finite must be bool")

        resolved_positive = (
            FORMAL_POSITIVE_LIMITS[role]
            if positive_limit is None
            else positive_limit
        )
        resolved_negative = (
            FORMAL_NEGATIVE_LIMITS[role]
            if negative_limit is None
            else negative_limit
        )
        for name, value in (
            ("positive_limit", resolved_positive),
            ("negative_limit", resolved_negative),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")

        self.role: Role = role
        self.q_channels = int(q_channels)
        self.local_channels = int(local_channels)
        self.hidden_channels = int(hidden_channels)
        self.evidence_floor = float(evidence_floor)
        self.detach_context = detach_context
        self.debug_validate_finite = debug_validate_finite

        # These values define checkpoint semantics, not merely diagnostics.
        # They are persistent and guarded before any state is installed.
        self.register_buffer(
            "role_code",
            torch.tensor(ROLE_CODES[role], dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "positive_limit",
            torch.tensor(float(resolved_positive), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "negative_limit",
            torch.tensor(float(resolved_negative), dtype=torch.float32),
            persistent=True,
        )

        self.local_projection = nn.Sequential(
            nn.Conv2d(self.local_channels, 16, kernel_size=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.q_projection = nn.Sequential(
            nn.Conv2d(self.q_channels, 8, kernel_size=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
        )

        # p_out, p_d0, four named auxiliary probabilities, and eight summary
        # maps make exactly fourteen scalar context channels.
        scalar_context_channels = 14
        self.context_stem = nn.Sequential(
            nn.Conv2d(
                16 + 8 + scalar_context_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GroupNorm(6, self.hidden_channels),
            nn.GELU(),
        )
        self.context_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.hidden_channels,
                        self.hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=self.hidden_channels,
                        bias=False,
                    ),
                    nn.GroupNorm(6, self.hidden_channels),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.context_fuse = nn.Sequential(
            nn.Conv2d(
                3 * self.hidden_channels,
                self.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(6, self.hidden_channels),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(
            self.hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        if _parameter_count(self) != PRODUCTION_PBDR_V4_PARAMETERS:
            raise RuntimeError("PBDR-V4 parameter count differs from production")

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject semantic checkpoint mismatches before copying buffers.

        Appending to ``error_msgs`` makes PyTorch raise for both strict modes.
        A guarded local copy also prevents a failed load from replacing this
        instance's role or capacity buffers before the exception is raised.
        """

        guarded_state = state_dict.copy()
        for local_name in self._SEMANTIC_BUFFER_NAMES:
            key = f"{prefix}{local_name}"
            expected = getattr(self, local_name)
            loaded = state_dict.get(key)
            if not _semantic_buffer_equal(loaded, expected):
                error_msgs.append(
                    "PBDR-V4 semantic checkpoint mismatch for "
                    f"{key!r}; cross-role and different-limit loads are forbidden"
                )
                guarded_state[key] = expected.detach().clone()

        super()._load_from_state_dict(
            guarded_state,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _require_finite(self, value: torch.Tensor, *, name: str) -> None:
        if not self.debug_validate_finite:
            return
        condition = torch.isfinite(value).all()
        if value.device.type == "cuda":
            torch._assert_async(condition, f"{name} contains non-finite values")
            return
        if not bool(condition):
            raise FloatingPointError(f"{name} contains non-finite values")

    def _normalize_q4(self, q4: torch.Tensor) -> torch.Tensor:
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
        self._require_finite(normalized, name="normalized_q4")
        return normalized

    def _validate_inputs(
        self,
        *,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> None:
        named_inputs = (
            ("z_out", z_out, 1),
            ("z_d0", z_d0, 1),
            ("z_gt2", z_gt2, 1),
            ("z_gt3", z_gt3, 1),
            ("z_gt4", z_gt4, 1),
            ("z_gt5", z_gt5, 1),
            ("q4", q4, self.q_channels),
            ("local_feature", local_feature, self.local_channels),
        )
        for name, value, channels in named_inputs:
            _require_bchw_float(value, name=name, channels=channels)
            self._require_finite(value, name=name)

        full_resolution = (
            z_d0,
            z_gt2,
            z_gt3,
            z_gt4,
            z_gt5,
            local_feature,
        )
        expected_batch = z_out.shape[0]
        expected_spatial = z_out.shape[-2:]
        for value in full_resolution:
            if (
                value.shape[0] != expected_batch
                or value.shape[-2:] != expected_spatial
            ):
                raise ValueError(
                    "z_d0, z_gt2-z_gt5 and local_feature must match "
                    "z_out batch and spatial shape"
                )
        if q4.shape[0] != expected_batch:
            raise ValueError("q4 batch size must match z_out")

        reference_device = z_out.device
        reference_dtype = z_out.dtype
        for name, value, _ in named_inputs[1:]:
            if value.device != reference_device:
                raise ValueError(f"{name} device must match z_out")
            if value.dtype != reference_dtype:
                raise ValueError(f"{name} dtype must match z_out")

    def forward_with_diagnostics(
        self,
        *,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> PBDRV4RoutingOutput:
        self._validate_inputs(
            z_out=z_out,
            z_d0=z_d0,
            z_gt2=z_gt2,
            z_gt3=z_gt3,
            z_gt4=z_gt4,
            z_gt5=z_gt5,
            q4=q4,
            local_feature=local_feature,
        )

        local = local_feature.detach() if self.detach_context else local_feature
        local_context = self.local_projection(local).to(dtype=z_out.dtype)
        q_context = self.q_projection(self._normalize_q4(q4))
        if q_context.shape[-2:] != z_out.shape[-2:]:
            q_context = F.interpolate(
                q_context,
                size=z_out.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        q_context = q_context.to(dtype=z_out.dtype)

        # Every Current-derived context tensor is immutable during Stage 1.
        p_out = torch.sigmoid(z_out.detach())
        p_d0 = torch.sigmoid(z_d0.detach())
        aux_probability = torch.cat(
            (
                torch.sigmoid(z_gt2.detach()),
                torch.sigmoid(z_gt3.detach()),
                torch.sigmoid(z_gt4.detach()),
                torch.sigmoid(z_gt5.detach()),
            ),
            dim=1,
        )
        aux_mean = aux_probability.mean(dim=1, keepdim=True)
        aux_max = aux_probability.amax(dim=1, keepdim=True)
        aux_min = aux_probability.amin(dim=1, keepdim=True)
        aux_std = aux_probability.std(dim=1, keepdim=True, unbiased=False)
        consensus = torch.sigmoid(8.0 * (aux_probability - 0.5)).mean(
            dim=1,
            keepdim=True,
        )
        uncertainty = (4.0 * p_out * (1.0 - p_out)).clamp(0.0, 1.0)
        support_gap = aux_mean - p_out
        spread = aux_max - aux_min

        scalar_context = torch.cat(
            (
                p_out,
                p_d0,
                aux_probability,
                aux_mean,
                aux_max,
                aux_min,
                aux_std,
                consensus,
                uncertainty,
                support_gap,
                spread,
            ),
            dim=1,
        ).to(dtype=z_out.dtype)
        context = torch.cat((local_context, q_context, scalar_context), dim=1)
        stem = self.context_stem(context)
        multi_scale = torch.cat(
            tuple(branch(stem) for branch in self.context_branches),
            dim=1,
        )
        fused = self.context_fuse(multi_scale)
        signed_score = torch.tanh(self.residual_head(fused))

        if self.role == "best_pd":
            rescue_budget = torch.ones_like(uncertainty)
        else:
            rescue_budget = (
                uncertainty + F.relu(aux_max - p_out)
            ).clamp(0.0, 1.0)
        suppression_budget = (
            uncertainty
            + F.relu(p_out - aux_mean)
            + F.relu(p_out - p_d0)
        ).clamp(0.0, 1.0)

        # Away from zero this is the planned sign-specific mapping.  At exact
        # identity, choose the larger evidence budget only to define a useful
        # subgradient; both branches still return an exact numeric zero.
        positive_branch = (signed_score > 0.0) | (
            (signed_score == 0.0) & (rescue_budget >= suppression_budget)
        )
        delta = torch.where(
            positive_branch,
            self.positive_limit * rescue_budget * signed_score,
            self.negative_limit * suppression_budget * signed_score,
        ).to(dtype=z_out.dtype)
        routed = z_out + delta
        self._require_finite(routed, name="routed_logits")

        return PBDRV4RoutingOutput(
            routed_logits=routed,
            delta_logits=delta,
            signed_score=signed_score,
            rescue_budget=rescue_budget,
            suppression_budget=suppression_budget,
            uncertainty=uncertainty,
            consensus=consensus,
        )

    def forward(
        self,
        *,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            z_out=z_out,
            z_d0=z_d0,
            z_gt2=z_gt2,
            z_gt3=z_gt3,
            z_gt4=z_gt4,
            z_gt5=z_gt5,
            q4=q4,
            local_feature=local_feature,
        ).routed_logits

    def architecture_manifest(self) -> Dict[str, Any]:
        return {
            "pbdr_v4_version": PBDR_V4_VERSION,
            "role": self.role,
            "role_code": int(self.role_code.detach().cpu()),
            "positive_limit": float(self.positive_limit.detach().cpu()),
            "negative_limit": float(self.negative_limit.detach().cpu()),
            "q_channels": self.q_channels,
            "local_channels": self.local_channels,
            "hidden_channels": self.hidden_channels,
            "scalar_context_channels": 14,
            "auxiliary_logit_order": ("gt2", "gt3", "gt4", "gt5"),
            "auxiliary_interface": "explicit_named_arguments",
            "evidence_floor": self.evidence_floor,
            "detach_context": self.detach_context,
            "debug_validate_finite": self.debug_validate_finite,
            "q4_normalization": "per_channel_spatial_centered_rms_floor",
            "residual_mapping": "signed_role_asymmetric_budgeted_tanh",
            "zero_subgradient": "larger_local_budget_branch",
            "zero_anchor": "zero_terminal_projection_current_final_exact",
            "parameters": _parameter_count(self),
            "state_key_count": len(self.state_dict()),
            "persistent_buffer_names": self._SEMANTIC_BUFFER_NAMES,
            "persistent_buffer_count": len(tuple(self.buffers())),
        }


def validate_formal_pbdr_v4_calibrator(
    module: nn.Module,
    *,
    expected_role: Role,
    require_identity_initialization: bool,
) -> Dict[str, Any]:
    """Validate the fixed production V4 module without running model data."""

    if type(module) is not RoleAlignedResidualCalibratorV4:
        raise TypeError("formal PBDR-V4 calibrator must use the exact class")
    expected_attributes = {
        "role": expected_role,
        "q_channels": FORMAL_Q4_CHANNELS,
        "local_channels": FORMAL_LOCAL_CHANNELS,
        "hidden_channels": FORMAL_HIDDEN_CHANNELS,
        "evidence_floor": FORMAL_EVIDENCE_FLOOR,
        "detach_context": FORMAL_DETACH_CONTEXT,
        "debug_validate_finite": FORMAL_DEBUG_VALIDATE_FINITE,
    }
    for name, expected in expected_attributes.items():
        if getattr(module, name) != expected:
            raise RuntimeError(f"formal PBDR-V4 attribute {name!r} differs")
    if int(module.role_code.detach().cpu()) != ROLE_CODES[expected_role]:
        raise RuntimeError("formal PBDR-V4 role buffer differs")
    expected_positive = torch.tensor(FORMAL_POSITIVE_LIMITS[expected_role])
    expected_negative = torch.tensor(FORMAL_NEGATIVE_LIMITS[expected_role])
    if not torch.equal(module.positive_limit.detach().cpu(), expected_positive):
        raise RuntimeError("formal PBDR-V4 positive limit differs")
    if not torch.equal(module.negative_limit.detach().cpu(), expected_negative):
        raise RuntimeError("formal PBDR-V4 negative limit differs")
    if _parameter_count(module) != PRODUCTION_PBDR_V4_PARAMETERS:
        raise RuntimeError("formal PBDR-V4 parameter count differs")
    if tuple(module.state_dict()) != PBDR_V4_LOCAL_STATE_KEYS:
        raise RuntimeError("formal PBDR-V4 state keys differ")
    if len(tuple(module.buffers())) != PRODUCTION_PBDR_V4_BUFFER_COUNT:
        raise RuntimeError("formal PBDR-V4 persistent buffer count differs")
    if require_identity_initialization:
        if int(torch.count_nonzero(module.residual_head.weight)) != 0:
            raise RuntimeError("formal PBDR-V4 terminal weight is not zero")
        if int(torch.count_nonzero(module.residual_head.bias)) != 0:
            raise RuntimeError("formal PBDR-V4 terminal bias is not zero")
    return module.architecture_manifest()


__all__ = [
    "FORMAL_DEBUG_VALIDATE_FINITE",
    "FORMAL_DETACH_CONTEXT",
    "FORMAL_EVIDENCE_FLOOR",
    "FORMAL_HIDDEN_CHANNELS",
    "FORMAL_LOCAL_CHANNELS",
    "FORMAL_NEGATIVE_LIMITS",
    "FORMAL_POSITIVE_LIMITS",
    "FORMAL_Q4_CHANNELS",
    "PBDR_V4_LOCAL_STATE_KEYS",
    "PBDR_V4_VERSION",
    "PBDRV4RoutingOutput",
    "PRODUCTION_PBDR_V4_BUFFER_COUNT",
    "PRODUCTION_PBDR_V4_PARAMETERS",
    "PRODUCTION_PBDR_V4_STATE_KEY_COUNT",
    "ROLE_CODES",
    "Role",
    "RoleAlignedResidualCalibratorV4",
    "validate_formal_pbdr_v4_calibrator",
]
