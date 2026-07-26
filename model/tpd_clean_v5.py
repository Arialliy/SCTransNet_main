"""Positive Context-selector KCS fusion for the isolated Clean-v5 candidate.

V5 keeps the same three TPD semantic sources and no fourth tokenizer branch:

* Keep: ``Conv1x1(PixelUnshuffle2(X))``;
* Context: ``AvgPool2(X)``;
* Saliency: ``MaxPool2(X) - Context``.

The signed Context interaction used by v4 can attenuate or reverse the
Saliency injection at different spatial positions, while its two learned
scales also become redundant in the constant-code capacity control.  V5 uses
one learned Saliency scale and a parameter-free, strictly positive Context
selector:

```
Q = tanh((C - mean_hw(C)) / rms_hw(C - mean_hw(C)))
P = 1 + 0.5 * Q                             # P in [0.5, 1.5]
R = S * tanh(saliency_scale * P)
Y = activation(K + R)
```

The paired capacity control sets ``P=1`` and has exactly the same parameter
layout.  At zero scale both variants are exactly dense SPD.  Context can only
select the magnitude of an existing Saliency response; it cannot create new
support or change its sign.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_V5_VARIANTS = (
    "tpd_clean_v5_full",
    "tpd_clean_v5_sal_capacity",
)
PRIMARY_CLEAN_V5_VARIANT = "tpd_clean_v5_full"

CONTEXT_SELECTOR_FLOOR = 0.5
CONTEXT_SELECTOR_CEILING = 1.5
_CONTEXT_SELECTOR_STRENGTH = 0.5

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v5_full": {
        "context_reference": "positive_selector",
        "fusion_support": "positive_context_selected_saliency",
        "context_code": "centered_spatial_rms_tanh_fp32",
        "context_selector": "positive_centered_0p5_to_1p5",
        "primary_candidate": True,
    },
    "tpd_clean_v5_sal_capacity": {
        "context_reference": "capacity_control",
        "fusion_support": "positive_context_selected_saliency",
        "context_code": "centered_spatial_rms_tanh_fp32_ignored",
        "context_selector": "neutral_one",
        "primary_candidate": False,
    },
}


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


def clean_v5_variant_spec(variant: str) -> Dict[str, object]:
    """Return a copy of the isolated v5 design contract."""

    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v5 variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V5_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])


class TPDCleanV5Block(nn.Module):
    """One aligned 2x KCS unit with a nonnegative Context selector."""

    def __init__(
        self,
        channels: int,
        activate: bool,
        *,
        use_context_selector: bool,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.channels = int(channels)
        self.use_context_selector = bool(use_context_selector)
        self.eps = float(eps)

        self.phase_compress = nn.Conv2d(
            4 * channels,
            channels,
            kernel_size=1,
        )
        self.saliency_scale = nn.Parameter(torch.zeros(channels))
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def branches(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"TPDCleanV5Block requires BxCxHxW input, got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TPDCleanV5Block expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "TPDCleanV5Block requires even H/W, "
                f"got {tuple(x.shape[-2:])}"
            )
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return keep, context, saliency

    def context_code(self, context: torch.Tensor) -> torch.Tensor:
        """Return signed normalized Context evidence in FP32.

        Keeping the normalized code in FP32 is deliberate: under CUDA
        autocast, later hierarchy blocks can receive FP16 inputs.  Casting the
        code back to that dtype here would quantize both ``Q`` and the positive
        selector before the bounded residual is formed.
        """

        context_fp32 = context.float()
        centered = context_fp32 - context_fp32.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.tanh(centered * inverse_rms)

    def context_selector(self, context: torch.Tensor) -> torch.Tensor:
        """Map Context evidence to a selector in ``[0.5, 1.5]``."""

        code = self.context_code(context)
        if not self.use_context_selector:
            return torch.ones_like(code)
        return 1.0 + _CONTEXT_SELECTOR_STRENGTH * code

    def fusion_terms(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep, context, saliency = self.branches(x)
        selector = self.context_selector(context)
        scale = self.saliency_scale.float().view(1, -1, 1, 1)
        coefficient = torch.tanh(scale * selector)
        residual = (saliency.float() * coefficient).to(dtype=saliency.dtype)
        return keep, residual, saliency, selector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, residual, _, _ = self.fusion_terms(x)
        return self.activation(keep + residual)


class TPDCleanV5PatchEmbedding(nn.Module):
    """Hierarchical dense-Keep embedding using the v5 KCS selector."""

    def __init__(
        self,
        channels: int,
        stride: int,
        *,
        use_context_selector: bool,
    ) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            TPDCleanV5Block(
                channels,
                activate=index < steps - 1,
                use_context_selector=use_context_selector,
            )
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


def build_clean_v5_patch_embedding(
    variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    variant = variant.lower()
    spec = clean_v5_variant_spec(variant)
    return TPDCleanV5PatchEmbedding(
        channels,
        stride,
        use_context_selector=(
            spec["context_selector"]
            == "positive_centered_0p5_to_1p5"
        ),
    )


def replace_shallow_embeddings_clean_v5(
    model: nn.Module,
    variant: str,
) -> Dict[str, nn.Module]:
    """Replace only ``mtc.embeddings_1/2`` with isolated v5 modules."""

    variant = variant.lower()
    clean_v5_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v5_patch_embedding(
            variant,
            channels=32,
            stride=16,
        ),
        "embeddings_2": build_clean_v5_patch_embedding(
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
    "CONTEXT_SELECTOR_FLOOR",
    "CONTEXT_SELECTOR_CEILING",
    "PRIMARY_CLEAN_V5_VARIANT",
    "SUPPORTED_CLEAN_V5_VARIANTS",
    "TPDCleanV5Block",
    "TPDCleanV5PatchEmbedding",
    "build_clean_v5_patch_embedding",
    "clean_v5_variant_spec",
    "parameter_count",
    "replace_shallow_embeddings_clean_v5",
]
