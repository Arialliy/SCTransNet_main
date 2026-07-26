"""Isolated KCS fusion candidates for TPD-Clean-v3.

The completed TPD-v1 and TPD-Clean-v2 sources are intentionally left
untouched.  Every v3 unit keeps the same three semantic branches:

* Keep: dense SPD ``PixelUnshuffle -> 1x1`` projection;
* Context: aligned 2x2 average pooling;
* Saliency: aligned 2x2 max-minus-average response.

Clean-v2 showed that an unconstrained additive Context residual could move the
high-Pd part of the Pd--Fa curve towards higher false-alarm rates.  V3 changes
only Context calibration and KCS fusion.  Its full candidate uses
``Saliency * tanh(centered(Context) / RMS(centered(Context)))``.  Therefore
Context cannot create a residual outside Saliency support and its magnitude is
bounded by the Saliency response.

The only paired control is ``tpd_clean_v3_sal_capacity``.  It replaces the
Context code with one while retaining the same two learned scales, state keys,
parameter count, initialization, and maximum residual range.  This separates
actual Context conditioning from simply giving Saliency a second scale.

All candidates have identical trainable parameter layouts and are exactly
equivalent to dense SPD at initialization because both residual scales start
at zero.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_V3_VARIANTS = (
    "tpd_clean_v3_full",
    "tpd_clean_v3_sal_capacity",
)

PRIMARY_CLEAN_V3_VARIANT = "tpd_clean_v3_full"

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v3_full": {
        "context_reference": "contrast",
        "fusion_support": "saliency_conditioned",
        "context_code": "centered_spatial_rms_tanh",
        "primary_candidate": True,
    },
    "tpd_clean_v3_sal_capacity": {
        "context_reference": "capacity_control",
        "fusion_support": "saliency_conditioned",
        "context_code": "constant_one",
        "primary_candidate": False,
    },
}


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


def clean_v3_variant_spec(variant: str) -> Dict[str, object]:
    """Return a copy of the frozen design contract for ``variant``."""
    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v3 variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V3_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])


class TPDCleanV3Block(nn.Module):
    """One aligned 2x KCS downsampling unit with calibrated Context fusion."""

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

    @staticmethod
    def _scale(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(scale).view(1, -1, 1, 1)
        return bounded * values

    def branches(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                f"TPDCleanV3Block requires even H/W, got {tuple(x.shape[-2:])}"
            )
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return keep, context, saliency

    def context_code(
        self, context: torch.Tensor, saliency: torch.Tensor
    ) -> torch.Tensor:
        """Return bounded Context code or its capacity-matched counterfactual."""
        if not self.use_context_code:
            return torch.ones_like(saliency)
        centered = context - context.mean(dim=(-2, -1), keepdim=True)
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.tanh(centered * inverse_rms)

    def fusion_terms(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep, context, saliency = self.branches(x)
        context_term = saliency * self.context_code(context, saliency)
        return keep, context_term, saliency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, context_term, saliency = self.fusion_terms(x)
        output = keep
        output = output + self._scale(saliency, self.saliency_scale)
        output = output + self._scale(context_term, self.context_scale)
        return self.activation(output)


class TPDCleanV3PatchEmbedding(nn.Module):
    """Hierarchical dense-Keep KCS embedding with frozen v3 fusion semantics."""

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
            TPDCleanV3Block(
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


def build_clean_v3_patch_embedding(
    variant: str, channels: int, stride: int
) -> nn.Module:
    variant = variant.lower()
    spec = clean_v3_variant_spec(variant)
    return TPDCleanV3PatchEmbedding(
        channels,
        stride,
        use_context_code=spec["context_code"] == "centered_spatial_rms_tanh",
    )


def replace_shallow_embeddings_clean_v3(
    model: nn.Module, variant: str
) -> Dict[str, nn.Module]:
    """Replace only ``embeddings_1/2`` with an isolated Clean-v3 candidate."""
    variant = variant.lower()
    clean_v3_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v3_patch_embedding(
            variant, channels=32, stride=16
        ),
        "embeddings_2": build_clean_v3_patch_embedding(
            variant, channels=64, stride=8
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    return replacements


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
