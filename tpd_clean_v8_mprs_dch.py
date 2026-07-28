"""TPD-Clean V8-MPRS-DCH.

Mass-preserving phase-resolved Saliency with the unchanged V7-DCH Context
headroom. The model keeps the SCTransNet TPD mainline: K/C/S only, no fourth
branch, no added learnable parameter or persistent buffer, and only
mtc.embeddings_1/2 are replaced.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS = (
    "tpd_clean_v8_mprs_dch_full",
    "tpd_clean_v8_mprs_dch_capacity",
)
PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT = "tpd_clean_v8_mprs_dch_full"

CONTEXT_HEADROOM_FLOOR = 0.75
CONTEXT_HEADROOM_CEILING = 1.25
_CONTEXT_MODULATION_SCALE = 0.5
_PHASE_COUNT = 4
_PHASE_CONTRAST_DENOMINATOR = 3.0

_COMMON_SPEC: Mapping[str, object] = {
    "candidate_family": "spd_anchored_tpd_clean_v8_mprs_dch",
    "mainline_contract": "Keep-Context-Saliency",
    "fourth_parallel_branch_added": False,
    "semantic_sources": ("Keep", "Context", "Saliency"),
    "phase_order": ("top_left", "top_right", "bottom_left", "bottom_right"),
    "pixel_unshuffle_channel_order": (
        "input_channel_major_four_phases_contiguous"
    ),
    "context_projection": "sum_keep_weights_over_four_contiguous_phases",
    "saliency_representation": "mass_preserving_phase_resolved",
    "saliency_formula": "S_p=(max_q(Z_q)-C0)+(Z_p-C0)/3",
    "saliency_mass_invariant": "sum_p(S_p)=4*(max_p(Z_p)-C0)",
    "saliency_nonnegative": True,
    "saliency_projection": "complete_keep_weight_phase_projection",
    "saliency_forward_implementation": (
        "algebraic_reuse_scalar_aligned_keep_linear_context_aligned"
    ),
    "learned_scales_per_block": 1,
    "scale_parameter": "per_channel_saliency_scale",
    "zero_scale_reference": "dense_spd_exact",
    "zero_scale_first_order_reference": "capacity_exact",
    "state_compatible_with": "tpd_clean_v7_dch",
    "shallow_embedding_parameters": 66_176,
    "full_model_parameters": 10_843_155,
}

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v8_mprs_dch_full": {
        **_COMMON_SPEC,
        "context_gate": 1.0,
        "context_reference": "phase_tied_deferred_zero_mean_gain",
        "context_headroom": "one_plus_abs_scale_times_one_minus_abs_scale_times_V",
        "fusion_formula": "K+Sa*(a*(1+abs(a)*(1-abs(a))*V))",
        "primary_candidate": True,
    },
    "tpd_clean_v8_mprs_dch_capacity": {
        **_COMMON_SPEC,
        "context_gate": 0.0,
        "context_reference": "capacity_control",
        "context_headroom": "neutral_one",
        "fusion_formula": "K+Sa*a",
        "primary_candidate": False,
    },
}


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


def clean_v8_mprs_dch_variant_spec(variant: str) -> Dict[str, object]:
    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v8 MPRS-DCH variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])


class TPDCleanV8MPRSDCHBlock(nn.Module):
    """One 2x KCS block with mass-preserving phase-resolved Saliency."""

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
        self.phase_compress = nn.Conv2d(4 * channels, channels, kernel_size=1)
        self.saliency_scale = nn.Parameter(torch.zeros(channels))
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(f"Expected even H/W, got {tuple(x.shape[-2:])}")

    def branches(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the shared Keep, Context, and scalar-Saliency sources."""

        self._validate_input(x)
        rearranged = F.pixel_unshuffle(x, 2)
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        scalar_saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(rearranged)
        return keep, context, scalar_saliency

    def phase_sources(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the explicit phase sources for tests and diagnostics only."""

        self._validate_input(x)
        rearranged = F.pixel_unshuffle(x, 2)
        batch, _, height, width = rearranged.shape
        phases = rearranged.reshape(
            batch, self.channels, _PHASE_COUNT, height, width
        )
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        scalar_saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        phase_saliency = (
            scalar_saliency.float().unsqueeze(2)
            + (phases.float() - context.float().unsqueeze(2))
            / _PHASE_CONTRAST_DENOMINATOR
        )
        return rearranged, context, scalar_saliency, phase_saliency

    def phase_tied_weight(self) -> torch.Tensor:
        """Parameter-free Context projection derived from Keep weights."""

        return self.phase_compress.weight.float().reshape(
            self.phase_compress.out_channels,
            self.channels,
            _PHASE_COUNT,
            1,
            1,
        ).sum(dim=2)

    def aligned_mprs_terms(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return K, Ca, V7 Sa, phase correction, and V8 Sa.

        The production path reuses Keep and Context alignment:

        ``Sa8 = Sa7 + ((K - bias) - Ca) / 3``.

        It does not materialize the explicit five-dimensional phase-Saliency
        tensor or execute a second 4C-to-C projection.
        """

        keep, context, scalar_saliency = self.branches(x)
        tied_weight = self.phase_tied_weight()
        scalar_aligned = F.conv2d(
            scalar_saliency.float(),
            tied_weight,
            bias=None,
        )
        context_aligned = F.conv2d(
            context.float(),
            tied_weight,
            bias=None,
        )
        bias = self.phase_compress.bias
        if bias is None:
            raise RuntimeError(
                "MPRS requires the V7-compatible phase_compress bias"
            )
        keep_linear = keep.float() - bias.float().view(1, -1, 1, 1)
        phase_correction = (
            keep_linear - context_aligned
        ) / _PHASE_CONTRAST_DENOMINATOR
        saliency_aligned = scalar_aligned + phase_correction
        return (
            keep,
            context_aligned,
            scalar_aligned,
            phase_correction,
            saliency_aligned,
        )

    def aligned_saliency_terms(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility view over the optimized production terms."""

        keep, context_aligned, scalar_aligned, _, saliency_aligned = (
            self.aligned_mprs_terms(x)
        )
        return keep, context_aligned, scalar_aligned, saliency_aligned

    def context_code(self, context_aligned: torch.Tensor) -> torch.Tensor:
        context_fp32 = context_aligned.float()
        centered = context_fp32 - context_fp32.mean(
            dim=(-2, -1), keepdim=True
        )
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.tanh(centered * inverse_rms)

    def context_modulation(
        self, context_aligned: torch.Tensor
    ) -> torch.Tensor:
        if self.context_gate == 0.0:
            return torch.zeros_like(context_aligned, dtype=torch.float32)
        code = self.context_code(context_aligned)
        centered_code = code - code.mean(dim=(-2, -1), keepdim=True)
        return self.context_gate * _CONTEXT_MODULATION_SCALE * centered_code

    def headroom(
        self, context_aligned: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modulation = self.context_modulation(context_aligned)
        scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)
        magnitude = scale.abs()
        headroom = 1.0 + magnitude * (1.0 - magnitude) * modulation
        return scale, modulation, headroom

    def fusion_terms(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep, context_aligned, _, _, saliency_aligned = (
            self.aligned_mprs_terms(x)
        )

        if self.context_gate == 0.0:
            scale = torch.tanh(
                self.saliency_scale.float()
            ).view(1, -1, 1, 1)
            modulation = torch.zeros_like(
                saliency_aligned, dtype=torch.float32
            )
            headroom = torch.ones_like(
                saliency_aligned, dtype=torch.float32
            )
        else:
            scale, modulation, headroom = self.headroom(context_aligned)

        residual = (saliency_aligned * scale * headroom).to(keep.dtype)
        return keep, residual, saliency_aligned, modulation

    def forward_with_mprs_diagnostics(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Return one forward result and its already-computed MPRS terms."""

        (
            keep,
            context_aligned,
            scalar_aligned,
            phase_correction,
            saliency_aligned,
        ) = self.aligned_mprs_terms(x)
        if self.context_gate == 0.0:
            scale = torch.tanh(
                self.saliency_scale.float()
            ).view(1, -1, 1, 1)
            modulation = torch.zeros_like(
                saliency_aligned,
                dtype=torch.float32,
            )
            headroom = torch.ones_like(
                saliency_aligned,
                dtype=torch.float32,
            )
        else:
            scale, modulation, headroom = self.headroom(context_aligned)
        residual = (saliency_aligned * scale * headroom).to(keep.dtype)
        output = self.activation(keep + residual)
        diagnostics = {
            "context_aligned": context_aligned,
            "saliency_v7": scalar_aligned,
            "phase_correction": phase_correction,
            "saliency_v8": saliency_aligned,
            "scale": scale,
            "modulation": modulation,
            "headroom": headroom,
        }
        return output, diagnostics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, residual, _, _ = self.fusion_terms(x)
        return self.activation(keep + residual)


class TPDCleanV8MPRSDCHPatchEmbedding(nn.Module):
    def __init__(
        self, channels: int, stride: int, *, context_gate: float
    ) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            TPDCleanV8MPRSDCHBlock(
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
        self, x: torch.Tensor | None
    ) -> Tuple[torch.Tensor | None, Tuple[torch.Tensor, ...]]:
        if x is None:
            return None, ()
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states[-1], tuple(states[:-1])


def build_clean_v8_mprs_dch_patch_embedding(
    variant: str, channels: int, stride: int
) -> nn.Module:
    spec = clean_v8_mprs_dch_variant_spec(variant.lower())
    return TPDCleanV8MPRSDCHPatchEmbedding(
        channels, stride, context_gate=float(spec["context_gate"])
    )


def replace_shallow_embeddings_clean_v8_mprs_dch(
    model: nn.Module, variant: str
) -> Dict[str, nn.Module]:
    clean_v8_mprs_dch_variant_spec(variant.lower())
    replacements = {
        "embeddings_1": build_clean_v8_mprs_dch_patch_embedding(
            variant, channels=32, stride=16
        ),
        "embeddings_2": build_clean_v8_mprs_dch_patch_embedding(
            variant, channels=64, stride=8
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
    "PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT",
    "SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS",
    "TPDCleanV8MPRSDCHBlock",
    "TPDCleanV8MPRSDCHPatchEmbedding",
    "build_clean_v8_mprs_dch_patch_embedding",
    "clean_v8_mprs_dch_variant_spec",
    "parameter_count",
    "replace_shallow_embeddings_clean_v8_mprs_dch",
]
