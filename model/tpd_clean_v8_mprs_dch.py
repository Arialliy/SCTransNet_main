"""Mass-preserving phase-resolved Saliency with V7-DCH headroom.

TPD-Clean V8-MPRS-DCH keeps the SCTransNet shallow-tokenizer mainline and
replaces only the Saliency representation inside the existing
Keep--Context--Saliency blocks:

``S_p = (max_q(Z_q) - C0) + (Z_p - C0) / 3``.

The production forward uses the algebraically equivalent reuse path

``Sa8 = Sa7 + ((K - bias) - Ca) / 3``

and therefore does not materialize the explicit five-dimensional
phase-Saliency tensor or execute a second ``4C -> C`` projection.  Full and
Capacity retain the V7-DCH parameter/state layout.  Both execute exactly three
convolutions per block: Keep, scalar-Saliency alignment, and Context alignment.
Capacity needs the last alignment for the MPRS correction but still skips the
Context code and headroom path.
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
    "phase_order": (
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ),
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
    "phase_contrast_parameters": 0,
    "phase_contrast_buffers": 0,
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
        "context_code": (
            "phase_aligned_centered_spatial_rms_eps_tanh_"
            "formal_amp_off_fp32"
        ),
        "context_modulation": "half_centered_context_code",
        "context_headroom": (
            "one_plus_abs_scale_times_one_minus_abs_scale_times_modulation"
        ),
        "fusion_support": (
            "phase_resolved_saliency_with_deferred_context_headroom"
        ),
        "fusion_formula": (
            "K+Sa8*(a*(1+abs(a)*(1-abs(a))*V));"
            "a=tanh(saliency_scale);V=0.5*(Q-mean_hw(Q))"
        ),
        "primary_candidate": True,
    },
    "tpd_clean_v8_mprs_dch_capacity": {
        **_COMMON_SPEC,
        "context_gate": 0.0,
        "context_reference": "capacity_control",
        "context_code": "not_computed_in_capacity_forward",
        "context_modulation": "not_computed_in_capacity_forward",
        "context_headroom": "neutral_one",
        "fusion_support": "phase_resolved_saliency_capacity_control",
        "fusion_formula": "K+Sa8*tanh(saliency_scale)",
        "primary_candidate": False,
    },
}


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


def clean_v8_mprs_dch_variant_spec(variant: str) -> Dict[str, object]:
    """Return a copy of the V8-MPRS-DCH variant contract."""

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
                "TPDCleanV8MPRSDCHBlock requires BxCxHxW input, "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TPDCleanV8MPRSDCHBlock expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "TPDCleanV8MPRSDCHBlock requires even H/W, "
                f"got {tuple(x.shape[-2:])}"
            )

    def branches(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the shared Keep, Context, and scalar-Saliency sources."""

        self._validate_input(x)
        rearranged = F.pixel_unshuffle(x, 2)
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        scalar_saliency = (
            F.max_pool2d(x, kernel_size=2, stride=2) - context
        )
        keep = self.phase_compress(rearranged)
        return keep, context, scalar_saliency

    def phase_sources(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return explicit MPRS sources for tests and diagnostics only."""

        self._validate_input(x)
        rearranged = F.pixel_unshuffle(x, 2)
        batch, _, height, width = rearranged.shape
        phases = rearranged.reshape(
            batch,
            self.channels,
            _PHASE_COUNT,
            height,
            width,
        )
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        scalar_saliency = (
            F.max_pool2d(x, kernel_size=2, stride=2) - context
        )
        phase_saliency = (
            scalar_saliency.float().unsqueeze(2)
            + (
                phases.float()
                - context.float().unsqueeze(2)
            )
            / _PHASE_CONTRAST_DENOMINATOR
        )
        return rearranged, context, scalar_saliency, phase_saliency

    def phase_tied_weight(self) -> torch.Tensor:
        """Derive the parameter-free Context/scalar projection in FP32."""

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
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the V7-compatible view of the optimized MPRS terms."""

        keep, context_aligned, scalar_aligned, _, saliency_aligned = (
            self.aligned_mprs_terms(x)
        )
        return keep, context_aligned, scalar_aligned, saliency_aligned

    def context_code(self, context_aligned: torch.Tensor) -> torch.Tensor:
        """Return the normalized and bounded V7-DCH Context code."""

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
        """Return Saliency scale ``a``, Context modulation ``V``, and ``H``."""

        modulation = self.context_modulation(context_aligned)
        scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)
        magnitude = scale.abs()
        headroom = 1.0 + magnitude * (1.0 - magnitude) * modulation
        return scale, modulation, headroom

    def fusion_terms(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Keep, DCH residual, MPRS-aligned Saliency, and modulation."""

        keep, context_aligned, _, _, saliency_aligned = (
            self.aligned_mprs_terms(x)
        )
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

        residual_fp32 = saliency_aligned * (scale * headroom)
        residual = residual_fp32.to(dtype=keep.dtype)
        return keep, residual, saliency_aligned, modulation

    def forward_with_mprs_diagnostics(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Return one forward result and the already-computed MPRS terms."""

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
    """Hierarchical V8 embedding with the five-node evidence interface."""

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


def build_clean_v8_mprs_dch_patch_embedding(
    variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    """Build one isolated V8-MPRS-DCH shallow patch embedding."""

    variant = variant.lower()
    spec = clean_v8_mprs_dch_variant_spec(variant)
    return TPDCleanV8MPRSDCHPatchEmbedding(
        channels,
        stride,
        context_gate=float(spec["context_gate"]),
    )


def replace_shallow_embeddings_clean_v8_mprs_dch(
    model: nn.Module,
    variant: str,
) -> Dict[str, nn.Module]:
    """Replace only ``mtc.embeddings_1/2`` with V8-MPRS-DCH modules."""

    variant = variant.lower()
    clean_v8_mprs_dch_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v8_mprs_dch_patch_embedding(
            variant,
            channels=32,
            stride=16,
        ),
        "embeddings_2": build_clean_v8_mprs_dch_patch_embedding(
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
    "PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT",
    "SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS",
    "TPDCleanV8MPRSDCHBlock",
    "TPDCleanV8MPRSDCHPatchEmbedding",
    "build_clean_v8_mprs_dch_patch_embedding",
    "clean_v8_mprs_dch_variant_spec",
    "parameter_count",
    "replace_shallow_embeddings_clean_v8_mprs_dch",
]
