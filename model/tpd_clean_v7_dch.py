"""Deferred Context headroom for the phase-tied KCS tokenizer.

TPD-Clean V7-DCH preserves the V6 Keep--Context--Saliency representation,
phase-tied projection, parameter/state layout, and dense-SPD zero-scale
anchor.  The only model-formula change is the Context headroom schedule:

``H = 1 + gate * |a| * (1 - |a|) * V``

where ``a=tanh(saliency_scale)``, ``V`` is the centered bounded Context code,
and ``gate`` is the fixed Python constant one for Full and zero for Capacity.
Consequently, Full and Capacity have identical outputs and identical
first-order optimization at the zero-scale anchor.  The Capacity forward path
does not compute Context alignment, Context code, or Context headroom.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_V7_DCH_VARIANTS = (
    "tpd_clean_v7_dch_full",
    "tpd_clean_v7_dch_capacity",
)
PRIMARY_CLEAN_V7_DCH_VARIANT = "tpd_clean_v7_dch_full"

CONTEXT_HEADROOM_FLOOR = 0.75
CONTEXT_HEADROOM_CEILING = 1.25
_CONTEXT_MODULATION_SCALE = 0.5

_COMMON_SPEC: Mapping[str, object] = {
    "candidate_family": (
        "spd_anchored_tpd_clean_v7_deferred_context_headroom"
    ),
    "mainline_contract": "Keep-Context-Saliency",
    "fourth_parallel_branch_added": False,
    "semantic_sources": ("Keep", "Context", "Saliency"),
    "phase_tied_projection": "sum_keep_weights_over_four_contiguous_phases",
    "pixel_unshuffle_channel_order": (
        "input_channel_major_four_phases_contiguous"
    ),
    "saliency_representation": "max_pool_minus_avg_pool_unchanged_from_v6",
    "learned_scales_per_block": 1,
    "scale_parameter": "per_channel_saliency_scale",
    "zero_scale_reference": "dense_spd_exact",
    "zero_scale_first_order_reference": "capacity_exact",
    "state_compatible_with": "tpd_clean_v6",
    "shallow_embedding_parameters": 66_176,
    "full_model_parameters": 10_843_155,
}

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v7_dch_full": {
        **_COMMON_SPEC,
        "context_gate": 1.0,
        "context_reference": "phase_tied_deferred_zero_mean_gain",
        "context_code": (
            "phase_aligned_centered_spatial_rms_eps_tanh_"
            "formal_amp_off_fp32"
        ),
        "context_modulation": "half_centered_context_code",
        "context_headroom": (
            "one_plus_abs_scale_times_one_minus_abs_scale_times_modulation"
        ),
        "fusion_support": (
            "phase_tied_deferred_zero_mean_context_gain_modulated_saliency"
        ),
        "fusion_formula": (
            "K+Sa*(a*(1+abs(a)*(1-abs(a))*V));"
            "a=tanh(saliency_scale);V=0.5*(Q-mean_hw(Q))"
        ),
        "primary_candidate": True,
    },
    "tpd_clean_v7_dch_capacity": {
        **_COMMON_SPEC,
        "context_gate": 0.0,
        "context_reference": "capacity_control",
        "context_code": "not_computed_in_capacity_forward",
        "context_modulation": "not_computed_in_capacity_forward",
        "context_headroom": "neutral_one",
        "fusion_support": "phase_tied_saliency_capacity_control",
        "fusion_formula": "K+Sa*tanh(saliency_scale)",
        "primary_candidate": False,
    },
}


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


def clean_v7_dch_variant_spec(variant: str) -> Dict[str, object]:
    """Return a copy of the frozen V7-DCH variant contract."""

    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v7 DCH variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V7_DCH_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])


class TPDCleanV7DCHBlock(nn.Module):
    """One phase-tied KCS block with deferred Context headroom."""

    def __init__(
        self,
        channels: int,
        activate: bool,
        *,
        context_gate: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if context_gate not in (0.0, 1.0):
            raise ValueError(
                f"context_gate must be exactly 0.0 or 1.0, got {context_gate}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.channels = int(channels)
        self.context_gate = float(context_gate)
        self.eps = float(eps)
        self.phase_compress = nn.Conv2d(
            4 * channels,
            channels,
            kernel_size=1,
        )
        self.saliency_scale = nn.Parameter(torch.zeros(channels))
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "TPDCleanV7DCHBlock requires BxCxHxW input, "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TPDCleanV7DCHBlock expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "TPDCleanV7DCHBlock requires even H/W, "
                f"got {tuple(x.shape[-2:])}"
            )

    def branches(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the unchanged V6 Keep, Context, and Saliency sources."""

        self._validate_input(x)
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return keep, context, saliency

    def phase_tied_weight(self) -> torch.Tensor:
        """Derive the parameter-free V6 phase-tied projection in FP32."""

        weight = self.phase_compress.weight.float()
        return weight.reshape(
            self.phase_compress.out_channels,
            self.channels,
            4,
            1,
            1,
        ).sum(dim=2)

    def context_code(self, context_aligned: torch.Tensor) -> torch.Tensor:
        """Return the normalized and bounded V6 Context code in FP32."""

        context_fp32 = context_aligned.float()
        centered = context_fp32 - context_fp32.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.tanh(centered * inverse_rms)

    def context_modulation(
        self,
        context_aligned: torch.Tensor,
    ) -> torch.Tensor:
        """Return centered ``V`` or the Capacity-control zero."""

        if self.context_gate == 0.0:
            return torch.zeros_like(context_aligned, dtype=torch.float32)
        code = self.context_code(context_aligned)
        centered_code = code - code.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        return (
            self.context_gate
            * _CONTEXT_MODULATION_SCALE
            * centered_code
        )

    def headroom(
        self,
        context_aligned: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return saliency scale ``a``, Context modulation ``V``, and ``H``."""

        modulation = self.context_modulation(context_aligned)
        scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)
        magnitude = scale.abs()
        headroom = 1.0 + magnitude * (1.0 - magnitude) * modulation
        return scale, modulation, headroom

    def fusion_terms(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Keep, DCH residual, aligned Saliency, and modulation."""

        keep, context, saliency = self.branches(x)
        tied_weight = self.phase_tied_weight()
        saliency_aligned = F.conv2d(
            saliency.float(),
            tied_weight,
            bias=None,
        )
        scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)

        if self.context_gate == 0.0:
            # AvgPool remains necessary because Saliency is MaxPool-AvgPool.
            # Only the otherwise unused Context alignment/code/headroom path
            # is skipped by this fixed Python branch.
            modulation = torch.zeros_like(
                saliency_aligned,
                dtype=torch.float32,
            )
            headroom = torch.ones_like(
                saliency_aligned,
                dtype=torch.float32,
            )
        else:
            context_aligned = F.conv2d(
                context.float(),
                tied_weight,
                bias=None,
            )
            _, modulation, headroom = self.headroom(context_aligned)

        residual_fp32 = saliency_aligned * (scale * headroom)
        residual = residual_fp32.to(dtype=keep.dtype)
        return keep, residual, saliency_aligned, modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, residual, _, _ = self.fusion_terms(x)
        return self.activation(keep + residual)


class TPDCleanV7DCHPatchEmbedding(nn.Module):
    """Hierarchical V7-DCH embedding with an explicit evidence interface."""

    def __init__(
        self,
        channels: int,
        stride: int,
        *,
        context_gate: float,
    ) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            TPDCleanV7DCHBlock(
                channels,
                activate=index < steps - 1,
                context_gate=context_gate,
            )
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x

    def forward_with_evidence(
        self,
        x: torch.Tensor | None,
    ) -> Tuple[torch.Tensor | None, Tuple[torch.Tensor, ...]]:
        """Return the endpoint and only the non-terminal block states."""

        if x is None:
            return None, ()
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states[-1], tuple(states[:-1])


def build_clean_v7_dch_patch_embedding(
    variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    """Build one isolated V7-DCH shallow patch embedding."""

    variant = variant.lower()
    spec = clean_v7_dch_variant_spec(variant)
    return TPDCleanV7DCHPatchEmbedding(
        channels,
        stride,
        context_gate=float(spec["context_gate"]),
    )


def replace_shallow_embeddings_clean_v7_dch(
    model: nn.Module,
    variant: str,
) -> Dict[str, nn.Module]:
    """Replace only ``mtc.embeddings_1/2`` with V7-DCH modules."""

    variant = variant.lower()
    clean_v7_dch_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v7_dch_patch_embedding(
            variant,
            channels=32,
            stride=16,
        ),
        "embeddings_2": build_clean_v7_dch_patch_embedding(
            variant,
            channels=64,
            stride=8,
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    return replacements


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "CONTEXT_HEADROOM_CEILING",
    "CONTEXT_HEADROOM_FLOOR",
    "PRIMARY_CLEAN_V7_DCH_VARIANT",
    "SUPPORTED_CLEAN_V7_DCH_VARIANTS",
    "TPDCleanV7DCHBlock",
    "TPDCleanV7DCHPatchEmbedding",
    "build_clean_v7_dch_patch_embedding",
    "clean_v7_dch_variant_spec",
    "parameter_count",
    "replace_shallow_embeddings_clean_v7_dch",
]
