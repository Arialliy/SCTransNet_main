"""Phase-resolved KCS tokenization for the SCTransNet shallow token path.

V7 keeps the established three-source Keep--Context--Saliency mainline and
changes only the Saliency alignment used by V6.

For one 2x block, let ``Z = pixel_unshuffle(X, 2)`` and reshape it as
``Z[b, c, p, h, w]`` with phases ordered TL/TR/BL/BR.  The three sources are:

* Keep: ``K = Conv1x1(flatten(Z); Wk, bk)``;
* Context: ``C0 = AvgPool2(X)``;
* Saliency: ``D_p = relu(Z_p - C0)``.

Context remains aligned with the V6 tied projection
``Wt[o, c] = sum_p Wk[o, c, p]``.  Saliency preserves its phase identity and
is aligned with the complete dense Keep weights:

``Sa[o] = sum_{c,p} Wk[o,c,p] * D[c,p]``.

Thus ``max_p(D_p) == MaxPool2(X) - C0`` up to the exact pooling arithmetic:
Saliency remains one semantic source and no fourth parallel branch is added.
Full and the phase-capacity control share the same parameters, state layout,
source extraction, and projection.  Their only difference is a Python
constant multiplying the Context headroom modulation.  Zero
``saliency_scale`` is exactly the dense-SPD output when Keep weights are
shared.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_V7_VARIANTS = (
    "tpd_clean_v7_full",
    "tpd_clean_v7_phase_capacity",
)
PRIMARY_CLEAN_V7_VARIANT = "tpd_clean_v7_full"

CONTEXT_HEADROOM_FLOOR = 0.5
CONTEXT_HEADROOM_CEILING = 1.5
_CONTEXT_MODULATION_SCALE = 0.5
_HEADROOM_SCALE = 0.5

_COMMON_SPEC: Mapping[str, object] = {
    "candidate_family": (
        "spd_anchored_tpd_clean_v7_phase_resolved_kcs_zero_mean_gain"
    ),
    "mainline_contract": "Keep-Context-Saliency",
    "fourth_parallel_branch_added": False,
    "semantic_sources": ("Keep", "Context", "Saliency"),
    "phase_order": ("top_left", "top_right", "bottom_left", "bottom_right"),
    "pixel_unshuffle_channel_order": (
        "input_channel_major_four_phases_contiguous"
    ),
    "context_projection": "sum_keep_weights_over_four_contiguous_phases",
    "saliency_representation": "positive_phase_deviation_from_local_mean",
    "saliency_formula": "D_p=relu(Z_p-C0)",
    "saliency_projection": "complete_keep_weight_phase_projection",
    "learned_scales_per_block": 1,
    "scale_parameter": "per_channel_saliency_scale",
    "zero_scale_reference": "dense_spd_exact",
    "state_compatible_with": "tpd_clean_v6",
    "shallow_embedding_parameters": 66_176,
    "full_model_parameters": 10_843_155,
}

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v7_full": {
        **_COMMON_SPEC,
        "context_reference": "phase_tied_zero_mean_gain_redistribution",
        "context_code": (
            "phase_aligned_centered_spatial_rms_eps_tanh_"
            "formal_amp_off_fp32"
        ),
        "context_gate": 1.0,
        "context_modulation": "half_centered_context_code",
        "context_headroom": (
            "one_plus_half_one_minus_abs_scale_times_modulation"
        ),
        "fusion_support": (
            "phase_resolved_zero_mean_context_gain_modulated_saliency"
        ),
        "fusion_formula": (
            "K+Sa*(a*(1+0.5*(1-abs(a))*V));"
            "a=tanh(saliency_scale);V=0.5*(Q-mean_hw(Q))"
        ),
        "primary_candidate": True,
    },
    "tpd_clean_v7_phase_capacity": {
        **_COMMON_SPEC,
        "context_reference": "phase_resolved_capacity_control",
        "context_code": (
            "phase_aligned_centered_spatial_rms_eps_tanh_"
            "formal_amp_off_fp32"
        ),
        "context_gate": 0.0,
        "context_modulation": "computed_then_multiplied_by_zero",
        "context_headroom": "neutral_one",
        "fusion_support": "phase_resolved_saliency_capacity_control",
        "fusion_formula": "K+Sa*tanh(saliency_scale)",
        "primary_candidate": False,
    },
}


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


def clean_v7_variant_spec(variant: str) -> Dict[str, object]:
    """Return a copy of the isolated V7 design and metadata contract."""

    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v7 variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V7_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])


class TPDCleanV7Block(nn.Module):
    """One 2x KCS unit with phase-resolved Saliency alignment."""

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

        # Parameter and state-key compatible with V6 and dense SPD, apart from
        # the existing zero-initialized Saliency scale.
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
                f"TPDCleanV7Block requires BxCxHxW input, got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TPDCleanV7Block expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "TPDCleanV7Block requires even H/W, "
                f"got {tuple(x.shape[-2:])}"
            )

    def phase_sources(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return rearranged input, Context, and phase-resolved Saliency.

        ``rearranged`` has shape ``[B, 4C, H/2, W/2]``.  ``saliency_phases``
        has shape ``[B, C, 4, H/2, W/2]`` in TL/TR/BL/BR order.
        ``avg_pool2d`` is retained explicitly so the Keep and Context paths
        remain operation-for-operation aligned with V6.
        """

        self._validate_input(x)
        rearranged = F.pixel_unshuffle(x, 2)
        batch, _, height, width = rearranged.shape
        phases = rearranged.reshape(
            batch,
            self.channels,
            4,
            height,
            width,
        )
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency_phases = F.relu(phases - context.unsqueeze(2))
        return rearranged, context, saliency_phases

    def branches(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Keep, Context, and phase-resolved Saliency sources."""

        rearranged, context, saliency_phases = self.phase_sources(x)
        keep = self.phase_compress(rearranged)
        return keep, context, saliency_phases

    def phase_tied_weight(self) -> torch.Tensor:
        """Return the parameter-free Context projection derived from Keep."""

        weight = self.phase_compress.weight.float()
        return weight.reshape(
            self.phase_compress.out_channels,
            self.channels,
            4,
            1,
            1,
        ).sum(dim=2)

    def aligned_branches(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Keep plus aligned Context and phase-resolved Saliency."""

        keep, context, saliency_phases = self.branches(x)
        context_aligned = F.conv2d(
            context.float(),
            self.phase_tied_weight(),
            bias=None,
        )
        saliency_flat = saliency_phases.reshape(
            saliency_phases.shape[0],
            4 * self.channels,
            saliency_phases.shape[-2],
            saliency_phases.shape[-1],
        )
        saliency_aligned = F.conv2d(
            saliency_flat.float(),
            self.phase_compress.weight.float(),
            bias=None,
        )
        return keep, context_aligned, saliency_aligned

    def context_code(self, context_aligned: torch.Tensor) -> torch.Tensor:
        """Return the normalized, bounded Context code in FP32."""

        context_fp32 = context_aligned.float()
        centered = context_fp32 - context_fp32.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.tanh(centered * inverse_rms)

    def context_modulation(self, context_aligned: torch.Tensor) -> torch.Tensor:
        """Compute the common Context path, then apply the paired constant."""

        code = self.context_code(context_aligned)
        centered_code = code - code.mean(dim=(-2, -1), keepdim=True)
        return (
            self.context_gate
            * _CONTEXT_MODULATION_SCALE
            * centered_code
        )

    def headroom(
        self,
        context_aligned: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-channel scale ``a``, modulation ``V``, and headroom."""

        modulation = self.context_modulation(context_aligned)
        scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)
        headroom = (
            1.0
            + _HEADROOM_SCALE
            * (1.0 - scale.abs())
            * modulation
        )
        return scale, modulation, headroom

    def fusion_terms(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Keep, residual, aligned Saliency, and Context modulation."""

        keep, context_aligned, saliency_aligned = self.aligned_branches(x)
        scale, modulation, headroom = self.headroom(context_aligned)
        residual_fp32 = saliency_aligned * (scale * headroom)
        residual = residual_fp32.to(dtype=keep.dtype)
        return keep, residual, saliency_aligned, modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, residual, _, _ = self.fusion_terms(x)
        return self.activation(keep + residual)


class TPDCleanV7PatchEmbedding(nn.Module):
    """Hierarchical dense-Keep embedding using V7 KCS blocks."""

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
            TPDCleanV7Block(
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


def build_clean_v7_patch_embedding(
    variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    """Build one isolated V7 shallow patch embedding."""

    spec = clean_v7_variant_spec(variant.lower())
    return TPDCleanV7PatchEmbedding(
        channels,
        stride,
        context_gate=float(spec["context_gate"]),
    )


def replace_shallow_embeddings_clean_v7(
    model: nn.Module,
    variant: str,
) -> Dict[str, nn.Module]:
    """Replace only ``mtc.embeddings_1/2`` with isolated V7 modules."""

    variant = variant.lower()
    clean_v7_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v7_patch_embedding(
            variant,
            channels=32,
            stride=16,
        ),
        "embeddings_2": build_clean_v7_patch_embedding(
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
    "PRIMARY_CLEAN_V7_VARIANT",
    "SUPPORTED_CLEAN_V7_VARIANTS",
    "TPDCleanV7Block",
    "TPDCleanV7PatchEmbedding",
    "build_clean_v7_patch_embedding",
    "clean_v7_variant_spec",
    "parameter_count",
    "replace_shallow_embeddings_clean_v7",
]
