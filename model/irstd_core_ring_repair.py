"""IRSTD-only core/ring repair head for a frozen SCTransNet Current model.

The parent TPD8+NER4+QFG2/TSS-off graph is an immutable feature provider.  This
module consumes only full-resolution, detached Current tensors and adds one
small dataset-specific residual readout.  Both terminal residual projections
are exactly zero-initialized, so construction is bitwise identity at the logit
output: ``routed_logits == z_out``.

The positive and negative capacities are architectural limits, not model-
selection thresholds.  No performance acceptance margin is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Final, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


IRSTD_CRR_VERSION: Final[str] = "irstd_core_ring_repair_v1"
LOCAL_CHANNELS: Final[int] = 32
HIDDEN_CHANNELS: Final[int] = 32
POSITIVE_LOGIT_LIMIT: Final[float] = 2.25
NEGATIVE_LOGIT_LIMIT: Final[float] = 1.25
PRODUCTION_PARAMETER_COUNT: Final[int] = 27_220
PRODUCTION_STATE_KEY_COUNT: Final[int] = 31
PRODUCTION_PERSISTENT_BUFFER_COUNT: Final[int] = 2


@dataclass(frozen=True, slots=True)
class IRSTDCoreRingRepairOutput:
    """Forward-local tensors required by the BGCR loss and diagnostics."""

    routed_logits: torch.Tensor
    delta_logits: torch.Tensor
    positive_delta: torch.Tensor
    negative_delta: torch.Tensor
    core_gate_logits: torch.Tensor
    halo_gate_logits: torch.Tensor
    core_gate: torch.Tensor
    halo_gate: torch.Tensor


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class LocalGroupNorm2d(nn.Module):
    """Group-normalize channels independently at every spatial location.

    PyTorch ``GroupNorm`` also pools over H and W, which makes a cached context
    crop depend on pixels outside the repair head's convolutional receptive
    field.  BGCR training requires the center of a 272 crop to be identical to
    the same center computed on the full 512 context, so normalization may
    only mix channels at one pixel.  The affine parameter layout remains the
    usual one-weight/one-bias-per-channel contract.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if (
            type(num_groups) is not int
            or type(num_channels) is not int
            or num_groups < 1
            or num_channels < 1
            or num_channels % num_groups != 0
            or num_channels // num_groups < 2
        ):
            raise ValueError("local group-normalization dimensions differ")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("local group-normalization eps must be positive")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 4
            or value.shape[1] != self.num_channels
        ):
            raise ValueError("LocalGroupNorm2d requires its frozen BCHW channels")
        batch, channels, height, width = value.shape
        grouped = value.reshape(
            batch,
            self.num_groups,
            channels // self.num_groups,
            height,
            width,
        )
        working = grouped.float()
        mean = working.mean(dim=2, keepdim=True)
        variance = working.var(dim=2, keepdim=True, unbiased=False)
        normalized = (working - mean) * torch.rsqrt(variance + self.eps)
        normalized = normalized.reshape(batch, channels, height, width).to(
            dtype=value.dtype,
        )
        return (
            normalized * self.weight.to(dtype=value.dtype).view(1, -1, 1, 1)
            + self.bias.to(dtype=value.dtype).view(1, -1, 1, 1)
        )


def _conv_norm_act(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    dilation: int = 1,
) -> nn.Sequential:
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        LocalGroupNorm2d(_group_count(out_channels), out_channels),
        nn.GELU(),
    )


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _validated_limit(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    ready = float(value)
    if not math.isfinite(ready) or ready <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return ready


def _semantic_buffer_equal(
    checkpoint_value: Any,
    expected_value: torch.Tensor,
) -> bool:
    if not isinstance(checkpoint_value, torch.Tensor):
        return False
    if checkpoint_value.shape != expected_value.shape:
        return False
    checkpoint_cpu = checkpoint_value.detach().cpu().to(
        dtype=expected_value.dtype,
    )
    return torch.equal(checkpoint_cpu, expected_value.detach().cpu())


class IRSTDCoreRingRepairHead(nn.Module):
    """Compact dual-arm IRSTD repair head with exact Current initialization.

    ``detach_context=True`` is the formal frozen-main setting.  In that mode
    every supplied Current tensor, including the base ``z_out`` anchor, is
    detached before use.  Consequently only this module's parameters can
    receive gradients even if a caller accidentally supplies tensors attached
    to a parent autograd graph.
    """

    _SEMANTIC_BUFFER_NAMES = ("positive_limit", "negative_limit")

    def __init__(
        self,
        *,
        local_channels: int = LOCAL_CHANNELS,
        hidden_channels: int = HIDDEN_CHANNELS,
        positive_limit: float = POSITIVE_LOGIT_LIMIT,
        negative_limit: float = NEGATIVE_LOGIT_LIMIT,
        detach_context: bool = True,
    ) -> None:
        super().__init__()
        if type(local_channels) is not int or local_channels < 1:
            raise ValueError("local_channels must be a positive integer")
        if (
            type(hidden_channels) is not int
            or hidden_channels < 8
            or hidden_channels % 2 != 0
        ):
            raise ValueError("hidden_channels must be an even integer >= 8")
        if type(detach_context) is not bool:
            raise TypeError("detach_context must be bool")
        ready_positive = _validated_limit(positive_limit, name="positive_limit")
        ready_negative = _validated_limit(negative_limit, name="negative_limit")

        self.local_channels = local_channels
        self.hidden_channels = hidden_channels
        self.detach_context = detach_context
        self.register_buffer(
            "positive_limit",
            torch.tensor(ready_positive, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "negative_limit",
            torch.tensor(ready_negative, dtype=torch.float32),
            persistent=True,
        )

        self.local_projection = _conv_norm_act(local_channels, 16, kernel_size=1)
        self.contrast_projection = _conv_norm_act(4, 8, kernel_size=3)

        # Thirteen scalar maps: p_out, p_d0, four named auxiliary
        # probabilities, mean/max/min/std, uncertainty, support gap and spread.
        scalar_channels = 13
        context_channels = 16 + 8 + scalar_channels
        self.context_stem = _conv_norm_act(
            context_channels,
            hidden_channels,
            kernel_size=3,
        )
        branch_channels = hidden_channels // 2
        self.context_branches = nn.ModuleList(
            [
                _conv_norm_act(
                    hidden_channels,
                    branch_channels,
                    kernel_size=3,
                    dilation=dilation,
                )
                for dilation in (1, 2, 3)
            ]
        )
        self.context_fuse = _conv_norm_act(
            branch_channels * 3,
            hidden_channels,
            kernel_size=1,
        )

        self.core_gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.halo_gate_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.positive_residual_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )
        self.negative_residual_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )

        # Gates receive a sparse prior.  Exact identity is guaranteed solely by
        # the two all-zero terminal residual projections.
        nn.init.normal_(self.core_gate_head.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.core_gate_head.bias, -1.5)
        nn.init.normal_(self.halo_gate_head.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.halo_gate_head.bias, -1.5)
        nn.init.zeros_(self.positive_residual_head.weight)
        nn.init.zeros_(self.positive_residual_head.bias)
        nn.init.zeros_(self.negative_residual_head.weight)
        nn.init.zeros_(self.negative_residual_head.bias)

        if _parameter_count(self) != PRODUCTION_PARAMETER_COUNT:
            raise RuntimeError("IRSTD BGCR parameter count differs")
        if len(self.state_dict()) != PRODUCTION_STATE_KEY_COUNT:
            raise RuntimeError("IRSTD BGCR state-key count differs")
        if len(tuple(self.buffers())) != PRODUCTION_PERSISTENT_BUFFER_COUNT:
            raise RuntimeError("IRSTD BGCR persistent-buffer count differs")

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject a different residual-capacity checkpoint in either mode."""

        guarded_state = state_dict.copy()
        for local_name in self._SEMANTIC_BUFFER_NAMES:
            key = f"{prefix}{local_name}"
            expected = getattr(self, local_name)
            loaded = state_dict.get(key)
            if not _semantic_buffer_equal(loaded, expected):
                error_msgs.append(
                    "IRSTD BGCR semantic checkpoint mismatch for "
                    f"{key!r}; different residual limits are forbidden"
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

    @staticmethod
    def _validate_one_channel(
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
                raise ValueError(f"{name} must match z_out shape")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match z_out device/dtype")
        _require_finite(value, name=name)

    def _validate_inputs(
        self,
        *,
        image: torch.Tensor,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> None:
        self._validate_one_channel(z_out, name="z_out")
        for name, value in (
            ("image", image),
            ("z_d0", z_d0),
            ("z_gt2", z_gt2),
            ("z_gt3", z_gt3),
            ("z_gt4", z_gt4),
            ("z_gt5", z_gt5),
        ):
            self._validate_one_channel(value, name=name, reference=z_out)
        if not isinstance(local_feature, torch.Tensor):
            raise TypeError("local_feature must be a tensor")
        if (
            local_feature.ndim != 4
            or min(local_feature.shape) < 1
            or local_feature.shape[1] != self.local_channels
        ):
            raise ValueError(
                "local_feature must be non-empty BCHW with "
                f"C={self.local_channels}"
            )
        if not local_feature.is_floating_point():
            raise TypeError("local_feature must use a floating-point dtype")
        if (
            local_feature.shape[0] != z_out.shape[0]
            or local_feature.shape[-2:] != z_out.shape[-2:]
        ):
            raise ValueError("local_feature must match z_out batch/spatial shape")
        if (
            local_feature.device != z_out.device
            or local_feature.dtype != z_out.dtype
        ):
            raise ValueError("local_feature must match z_out device/dtype")
        _require_finite(local_feature, name="local_feature")

    @staticmethod
    def _local_contrast(image: torch.Tensor) -> torch.Tensor:
        mean3 = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        mean7 = F.avg_pool2d(image, kernel_size=7, stride=1, padding=3)
        mean5 = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        mean_sq5 = F.avg_pool2d(
            image.square(),
            kernel_size=5,
            stride=1,
            padding=2,
        )
        variance5 = (mean_sq5 - mean5.square()).clamp_min(0.0)
        std5 = torch.sqrt(variance5 + 1.0e-8)
        return torch.cat((image, image - mean3, image - mean7, std5), dim=1)

    def forward_with_diagnostics(
        self,
        *,
        image: torch.Tensor,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> IRSTDCoreRingRepairOutput:
        self._validate_inputs(
            image=image,
            z_out=z_out,
            z_d0=z_d0,
            z_gt2=z_gt2,
            z_gt3=z_gt3,
            z_gt4=z_gt4,
            z_gt5=z_gt5,
            local_feature=local_feature,
        )

        if self.detach_context:
            image_context = image.detach()
            local_context_input = local_feature.detach()
            readouts = tuple(
                value.detach()
                for value in (z_out, z_d0, z_gt2, z_gt3, z_gt4, z_gt5)
            )
        else:
            image_context = image
            local_context_input = local_feature
            readouts = (z_out, z_d0, z_gt2, z_gt3, z_gt4, z_gt5)

        out_ctx, d0_ctx, gt2_ctx, gt3_ctx, gt4_ctx, gt5_ctx = readouts
        p_out = torch.sigmoid(out_ctx)
        p_d0 = torch.sigmoid(d0_ctx)
        auxiliary = torch.cat(
            tuple(
                torch.sigmoid(value)
                for value in (gt2_ctx, gt3_ctx, gt4_ctx, gt5_ctx)
            ),
            dim=1,
        )
        auxiliary_mean = auxiliary.mean(dim=1, keepdim=True)
        auxiliary_max = auxiliary.amax(dim=1, keepdim=True)
        auxiliary_min = auxiliary.amin(dim=1, keepdim=True)
        auxiliary_std = auxiliary.std(dim=1, keepdim=True, unbiased=False)
        uncertainty = (4.0 * p_out * (1.0 - p_out)).clamp(0.0, 1.0)
        support_gap = auxiliary_mean - p_out
        spread = auxiliary_max - auxiliary_min
        scalar_context = torch.cat(
            (
                p_out,
                p_d0,
                auxiliary,
                auxiliary_mean,
                auxiliary_max,
                auxiliary_min,
                auxiliary_std,
                uncertainty,
                support_gap,
                spread,
            ),
            dim=1,
        )

        local = self.local_projection(local_context_input)
        contrast = self.contrast_projection(self._local_contrast(image_context))
        stem = self.context_stem(
            torch.cat((local, contrast, scalar_context), dim=1)
        )
        multi_scale = torch.cat(
            tuple(branch(stem) for branch in self.context_branches),
            dim=1,
        )
        fused = self.context_fuse(multi_scale)

        core_gate_logits = self.core_gate_head(fused)
        halo_gate_logits = self.halo_gate_head(fused)
        core_gate = torch.sigmoid(core_gate_logits)
        halo_gate = torch.sigmoid(halo_gate_logits)
        positive_signal = torch.tanh(self.positive_residual_head(fused))
        negative_signal = torch.tanh(self.negative_residual_head(fused))
        positive_delta = (
            self.positive_limit.to(dtype=out_ctx.dtype)
            * core_gate
            * positive_signal
        )
        negative_delta = (
            self.negative_limit.to(dtype=out_ctx.dtype)
            * halo_gate
            * negative_signal
        )
        delta = positive_delta - negative_delta
        routed = out_ctx + delta

        for name, value in (
            ("core_gate_logits", core_gate_logits),
            ("halo_gate_logits", halo_gate_logits),
            ("positive_delta", positive_delta),
            ("negative_delta", negative_delta),
            ("delta_logits", delta),
            ("routed_logits", routed),
        ):
            _require_finite(value, name=name)

        return IRSTDCoreRingRepairOutput(
            routed_logits=routed,
            delta_logits=delta,
            positive_delta=positive_delta,
            negative_delta=negative_delta,
            core_gate_logits=core_gate_logits,
            halo_gate_logits=halo_gate_logits,
            core_gate=core_gate,
            halo_gate=halo_gate,
        )

    def forward(
        self,
        *,
        image: torch.Tensor,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            image=image,
            z_out=z_out,
            z_d0=z_d0,
            z_gt2=z_gt2,
            z_gt3=z_gt3,
            z_gt4=z_gt4,
            z_gt5=z_gt5,
            local_feature=local_feature,
        ).routed_logits

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "version": IRSTD_CRR_VERSION,
            "local_channels": self.local_channels,
            "hidden_channels": self.hidden_channels,
            "positive_limit": float(self.positive_limit.detach().cpu()),
            "negative_limit": float(self.negative_limit.detach().cpu()),
            "detach_context": self.detach_context,
            "local_contrast": ("image", "highpass3", "highpass7", "std5"),
            "normalization": (
                "local_group_norm_channels_only_no_spatial_reduction"
            ),
            "normalization_groups_by_width": {
                "8": _group_count(8),
                "16": _group_count(16),
                "32": _group_count(32),
            },
            "maximum_spatial_receptive_radius": 8,
            "parameter_count": _parameter_count(self),
            "state_key_count": len(self.state_dict()),
            "persistent_buffer_count": len(tuple(self.buffers())),
            "terminal_initialization": "exact_zero",
            "performance_acceptance_margin": None,
        }


def validate_formal_irstd_core_ring_repair_head(
    module: nn.Module,
    *,
    require_identity_initialization: bool,
) -> dict[str, Any]:
    """Validate the exact frozen-main BGCR head and optional identity state."""

    if type(module) is not IRSTDCoreRingRepairHead:
        raise TypeError("formal IRSTD BGCR head must use the exact class")
    expected_attributes = {
        "local_channels": LOCAL_CHANNELS,
        "hidden_channels": HIDDEN_CHANNELS,
        "detach_context": True,
    }
    for name, expected in expected_attributes.items():
        if getattr(module, name) != expected:
            raise RuntimeError(f"formal IRSTD BGCR attribute {name!r} differs")
    expected_positive = torch.tensor(POSITIVE_LOGIT_LIMIT, dtype=torch.float32)
    expected_negative = torch.tensor(NEGATIVE_LOGIT_LIMIT, dtype=torch.float32)
    if not torch.equal(module.positive_limit.detach().cpu(), expected_positive):
        raise RuntimeError("formal IRSTD BGCR positive limit differs")
    if not torch.equal(module.negative_limit.detach().cpu(), expected_negative):
        raise RuntimeError("formal IRSTD BGCR negative limit differs")
    if _parameter_count(module) != PRODUCTION_PARAMETER_COUNT:
        raise RuntimeError("formal IRSTD BGCR parameter count differs")
    if len(module.state_dict()) != PRODUCTION_STATE_KEY_COUNT:
        raise RuntimeError("formal IRSTD BGCR state-key count differs")
    if len(tuple(module.buffers())) != PRODUCTION_PERSISTENT_BUFFER_COUNT:
        raise RuntimeError("formal IRSTD BGCR persistent-buffer count differs")
    normalizers = tuple(
        child for child in module.modules() if isinstance(child, LocalGroupNorm2d)
    )
    expected_normalizers = (
        (8, 16),
        (4, 8),
        (8, 32),
        (8, 16),
        (8, 16),
        (8, 16),
        (8, 32),
    )
    observed_normalizers = tuple(
        (child.num_groups, child.num_channels) for child in normalizers
    )
    if observed_normalizers != expected_normalizers:
        raise RuntimeError("formal IRSTD BGCR local-normalization layout differs")
    if any(isinstance(child, nn.GroupNorm) for child in module.modules()):
        raise RuntimeError("formal IRSTD BGCR cannot reduce normalization over H/W")
    if require_identity_initialization:
        for name, terminal in (
            ("positive", module.positive_residual_head),
            ("negative", module.negative_residual_head),
        ):
            if int(torch.count_nonzero(terminal.weight)) != 0:
                raise RuntimeError(
                    f"formal IRSTD BGCR {name} terminal weight is not zero"
                )
            if terminal.bias is None or int(torch.count_nonzero(terminal.bias)) != 0:
                raise RuntimeError(
                    f"formal IRSTD BGCR {name} terminal bias is not zero"
                )
    return module.architecture_manifest()


__all__ = [
    "HIDDEN_CHANNELS",
    "IRSTD_CRR_VERSION",
    "LocalGroupNorm2d",
    "IRSTDCoreRingRepairHead",
    "IRSTDCoreRingRepairOutput",
    "LOCAL_CHANNELS",
    "NEGATIVE_LOGIT_LIMIT",
    "POSITIVE_LOGIT_LIMIT",
    "PRODUCTION_PARAMETER_COUNT",
    "PRODUCTION_PERSISTENT_BUFFER_COUNT",
    "PRODUCTION_STATE_KEY_COUNT",
    "validate_formal_irstd_core_ring_repair_head",
]
