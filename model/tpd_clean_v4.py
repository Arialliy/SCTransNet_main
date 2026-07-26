"""Stable single-logit KCS fusion for TPD-Clean-v4.

The completed TPD-v1, TPD-Clean-v2, and TPD-Clean-v3 sources are intentionally
left untouched.  V4 keeps exactly the same three semantic sources:

* Keep: dense SPD ``PixelUnshuffle -> 1x1`` projection;
* Context: aligned 2x2 average pooling;
* Saliency: aligned 2x2 max-minus-average response.

V3 injected Saliency and Context-conditioned Saliency through two independent
additive residuals.  V4 instead combines both controls into one spatial logit:

``logit = saliency_scale + 0.5 * tanh(context_scale) * context_code``

and injects only ``saliency * tanh(logit)``.  The residual is therefore
strictly bounded by the Saliency magnitude and Context cannot create support
outside the Saliency branch.

The paired capacity control replaces the Context code with one while keeping
the same parameters, state layout, initialization, and residual bound.  Both
learned scales start at zero, so every V4 unit is exactly equivalent to dense
SPD at initialization.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_V4_VARIANTS = (
    "tpd_clean_v4_full",
    "tpd_clean_v4_sal_capacity",
)

PRIMARY_CLEAN_V4_VARIANT = "tpd_clean_v4_full"

_CONTEXT_LOGIT_LIMIT = 0.5

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v4_full": {
        "context_reference": "contrast",
        "fusion_support": "single_bounded_saliency_logit",
        "context_code": "centered_spatial_rms_tanh_fp32",
        "primary_candidate": True,
    },
    "tpd_clean_v4_sal_capacity": {
        "context_reference": "capacity_control",
        "fusion_support": "single_bounded_saliency_logit",
        "context_code": "constant_one",
        "primary_candidate": False,
    },
}


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


def clean_v4_variant_spec(variant: str) -> Dict[str, object]:
    """Return a copy of the frozen design contract for ``variant``."""
    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v4 variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V4_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])


class TPDCleanV4Block(nn.Module):
    """One aligned 2x KCS downsampling unit with a single bounded logit."""

    def __init__(
        self,
        channels: int,
        activate: bool,
        *,
        use_context_code: bool,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.channels = channels
        self.use_context_code = use_context_code
        self.eps = float(eps)

        # Exact dense-SPD Keep anchor.  Its parameter layout matches
        # model.tpd.PhaseKeepBlock.
        self.phase_compress = nn.Conv2d(4 * channels, channels, kernel_size=1)
        self.context_scale = nn.Parameter(torch.zeros(channels))
        self.saliency_scale = nn.Parameter(torch.zeros(channels))
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def branches(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                f"TPDCleanV4Block requires even H/W, got "
                f"{tuple(x.shape[-2:])}"
            )
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return keep, context, saliency

    def context_code(self, context: torch.Tensor) -> torch.Tensor:
        """Return a bounded FP32 Context code or its paired counterfactual."""
        if not self.use_context_code:
            return torch.ones_like(context)

        # Keep normalization in FP32 under autocast.  Mean-square is used
        # directly rather than an unbiased variance estimate.
        context_fp32 = context.float()
        centered = context_fp32 - context_fp32.mean(
            dim=(-2, -1), keepdim=True
        )
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.tanh(centered * inverse_rms).to(dtype=context.dtype)

    def fusion_logit(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the spatial fusion logit and the Context code used by it."""
        code = self.context_code(context)
        saliency_logit = self.saliency_scale.float().view(1, -1, 1, 1)
        context_logit = (
            _CONTEXT_LOGIT_LIMIT
            * torch.tanh(self.context_scale.float()).view(1, -1, 1, 1)
            * code.float()
        )
        logit = saliency_logit + context_logit
        return logit, code

    def fusion_terms(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep, context, saliency = self.branches(x)
        logit, code = self.fusion_logit(context)
        coefficient = torch.tanh(logit).to(dtype=saliency.dtype)
        residual = saliency * coefficient
        return keep, residual, saliency, code

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, residual, _, _ = self.fusion_terms(x)
        return self.activation(keep + residual)


class TPDCleanV4PatchEmbedding(nn.Module):
    """Hierarchical dense-Keep KCS embedding with stable V4 fusion."""

    def __init__(
        self,
        channels: int,
        stride: int,
        *,
        use_context_code: bool,
    ) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            TPDCleanV4Block(
                channels,
                activate=index < steps - 1,
                use_context_code=use_context_code,
            )
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


def build_clean_v4_patch_embedding(
    variant: str, channels: int, stride: int
) -> nn.Module:
    variant = variant.lower()
    spec = clean_v4_variant_spec(variant)
    return TPDCleanV4PatchEmbedding(
        channels,
        stride,
        use_context_code=(
            spec["context_code"] == "centered_spatial_rms_tanh_fp32"
        ),
    )


def replace_shallow_embeddings_clean_v4(
    model: nn.Module, variant: str
) -> Dict[str, nn.Module]:
    """Replace only ``embeddings_1/2`` with an isolated Clean-v4 candidate."""
    variant = variant.lower()
    clean_v4_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v4_patch_embedding(
            variant, channels=32, stride=16
        ),
        "embeddings_2": build_clean_v4_patch_embedding(
            variant, channels=64, stride=8
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    return replacements


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
