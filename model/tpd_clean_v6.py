"""Phase-tied, mean-neutral KCS fusion for TPD-Clean-v6.

V6 is an isolated shallow patch-embedding candidate.  It preserves exactly
three semantic sources and does not add a fourth tokenizer branch:

* Keep: ``Conv1x1(PixelUnshuffle2(X); Wk, bk)``;
* Context: ``AvgPool2(X)``;
* Saliency: ``MaxPool2(X) - Context``.

The Context and Saliency sources are aligned to the Keep output channels with
a parameter-free projection derived from the dense Keep weights.  PyTorch
``pixel_unshuffle(..., 2)`` orders its output channels as four contiguous
spatial phases for each input channel, so the tied projection is

``Wt[o, c] = sum(Wk[o, 4*c : 4*c + 4])``.

Full uses mean-neutral Context headroom to redistribute the aligned Saliency
gain spatially.  The paired capacity control sets that headroom to one.  Both
variants retain the exact v5 parameter/state layout: one per-channel
``saliency_scale`` plus the dense Keep projection in each block.  Consequently
zero scale is exactly dense SPD when the Keep weights are shared.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_V6_VARIANTS = (
    "tpd_clean_v6_full",
    "tpd_clean_v6_phase_capacity",
)
PRIMARY_CLEAN_V6_VARIANT = "tpd_clean_v6_full"

CONTEXT_HEADROOM_FLOOR = 0.5
CONTEXT_HEADROOM_CEILING = 1.5
_CONTEXT_MODULATION_SCALE = 0.5
_HEADROOM_SCALE = 0.5

_COMMON_SPEC: Mapping[str, object] = {
    "candidate_family": "spd_anchored_tpd_clean_v6_phase_tied_mean_neutral_kcs",
    "mainline_contract": "Keep-Context-Saliency",
    "fourth_parallel_branch_added": False,
    "semantic_sources": ("Keep", "Context", "Saliency"),
    "phase_tied_projection": "sum_keep_weights_over_four_contiguous_phases",
    "pixel_unshuffle_channel_order": "input_channel_major_four_phases_contiguous",
    "learned_scales_per_block": 1,
    "scale_parameter": "per_channel_saliency_scale",
    "zero_scale_reference": "dense_spd_exact",
    "shallow_embedding_parameters": 66_176,
    "full_model_parameters": 10_843_155,
}

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v6_full": {
        **_COMMON_SPEC,
        "context_reference": "phase_tied_mean_neutral",
        "context_code": "phase_aligned_centered_spatial_rms_tanh_fp32",
        "context_modulation": "half_centered_context_code",
        "context_headroom": "one_plus_half_one_minus_abs_scale_times_modulation",
        "fusion_support": "phase_tied_mean_neutral_context_modulated_saliency",
        "fusion_formula": (
            "K+Sa*(a*(1+0.5*(1-abs(a))*V));"
            "a=tanh(saliency_scale);V=0.5*(Q-mean_hw(Q))"
        ),
        "primary_candidate": True,
    },
    "tpd_clean_v6_phase_capacity": {
        **_COMMON_SPEC,
        "context_reference": "phase_tied_capacity_control",
        "context_code": "phase_aligned_context_computed_but_modulation_zero",
        "context_modulation": "zero",
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


def clean_v6_variant_spec(variant: str) -> Dict[str, object]:
    """Return a copy of the isolated v6 design/metadata contract."""

    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v6 variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V6_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])


class TPDCleanV6Block(nn.Module):
    """One aligned 2x phase-tied, mean-neutral KCS downsampling unit."""

    def __init__(
        self,
        channels: int,
        activate: bool,
        *,
        use_context_headroom: bool,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.channels = int(channels)
        self.use_context_headroom = bool(use_context_headroom)
        self.eps = float(eps)

        # This is state-key and parameter compatible with the v5 block.
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
                f"TPDCleanV6Block requires BxCxHxW input, got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TPDCleanV6Block expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "TPDCleanV6Block requires even H/W, "
                f"got {tuple(x.shape[-2:])}"
            )

    def branches(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three K/C/S semantic sources before tied alignment."""

        self._validate_input(x)
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return keep, context, saliency

    def phase_tied_weight(self) -> torch.Tensor:
        """Derive ``Wt`` from dense Keep weights without state or parameters.

        ``pixel_unshuffle(x, 2)`` stores the four phases of input channel ``c``
        at output channels ``4*c + p``.  Reshaping the Keep weight as
        ``[out, in, phase, 1, 1]`` therefore exposes the exact phase axis.
        The tied projection is evaluated in FP32 as required by the protocol.
        """

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
        """Return Keep plus FP32 phase-aligned Context and Saliency."""

        keep, context, saliency = self.branches(x)
        tied_weight = self.phase_tied_weight()
        context_aligned = F.conv2d(
            context.float(),
            tied_weight,
            bias=None,
        )
        saliency_aligned = F.conv2d(
            saliency.float(),
            tied_weight,
            bias=None,
        )
        return keep, context_aligned, saliency_aligned

    def context_code(self, context_aligned: torch.Tensor) -> torch.Tensor:
        """Return the normalized, bounded phase-aligned Context code in FP32."""

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
        """Return mean-neutral ``V`` or the paired capacity-control zero."""

        if not self.use_context_headroom:
            return torch.zeros_like(context_aligned, dtype=torch.float32)
        code = self.context_code(context_aligned)
        return _CONTEXT_MODULATION_SCALE * (
            code - code.mean(dim=(-2, -1), keepdim=True)
        )

    def headroom(
        self,
        context_aligned: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-channel scale ``a``, modulation ``V``, and headroom ``H``."""

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
        coefficient = scale * headroom
        residual_fp32 = saliency_aligned * coefficient
        residual = residual_fp32.to(dtype=keep.dtype)
        return keep, residual, saliency_aligned, modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, residual, _, _ = self.fusion_terms(x)
        return self.activation(keep + residual)


class TPDCleanV6PatchEmbedding(nn.Module):
    """Hierarchical dense-Keep embedding using v6 phase-tied KCS blocks."""

    def __init__(
        self,
        channels: int,
        stride: int,
        *,
        use_context_headroom: bool,
    ) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            TPDCleanV6Block(
                channels,
                activate=index < steps - 1,
                use_context_headroom=use_context_headroom,
            )
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


def build_clean_v6_patch_embedding(
    variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    """Build one isolated v6 shallow patch embedding."""

    variant = variant.lower()
    spec = clean_v6_variant_spec(variant)
    return TPDCleanV6PatchEmbedding(
        channels,
        stride,
        use_context_headroom=(
            spec["context_modulation"] == "half_centered_context_code"
        ),
    )


def replace_shallow_embeddings_clean_v6(
    model: nn.Module,
    variant: str,
) -> Dict[str, nn.Module]:
    """Replace only ``mtc.embeddings_1/2`` with isolated v6 modules."""

    variant = variant.lower()
    clean_v6_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v6_patch_embedding(
            variant,
            channels=32,
            stride=16,
        ),
        "embeddings_2": build_clean_v6_patch_embedding(
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
    "PRIMARY_CLEAN_V6_VARIANT",
    "SUPPORTED_CLEAN_V6_VARIANTS",
    "TPDCleanV6Block",
    "TPDCleanV6PatchEmbedding",
    "build_clean_v6_patch_embedding",
    "clean_v6_variant_spec",
    "parameter_count",
    "replace_shallow_embeddings_clean_v6",
]
