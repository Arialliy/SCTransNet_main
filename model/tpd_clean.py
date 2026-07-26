"""SPD-anchored TPD candidates for the second controlled experiment.

This module is intentionally separate from :mod:`model.tpd`.  The completed
formal800 experiment fingerprints that file, so keeping the v2 candidates
here preserves the exact v1 source while allowing a clean follow-up.

Every ``CleanTPD2`` unit uses the same dense PixelUnshuffle projection as the
SPD control.  Context and saliency are bounded, zero-initialized residuals.
Consequently, a clean unit is exactly equivalent to its SPD counterpart at
initialization when their dense Keep weights are equal.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CLEAN_VARIANTS = (
    "tpd_clean_ctx",
    "tpd_clean_sal",
    "tpd_clean_full",
    "grouped_keep",
)


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


class CleanTPD2(nn.Module):
    """One SPD-anchored 2x downsampling unit with optional TPD residuals."""

    def __init__(
        self,
        channels: int,
        activate: bool,
        *,
        use_context: bool,
        use_saliency: bool,
    ) -> None:
        super().__init__()
        if not use_context and not use_saliency:
            raise ValueError("CleanTPD2 requires at least one residual branch")
        self.channels = channels
        self.use_context = use_context
        self.use_saliency = use_saliency
        # This projection deliberately matches PhaseKeepBlock in model/tpd.py.
        self.phase_compress = nn.Conv2d(4 * channels, channels, kernel_size=1)
        if use_context:
            self.context_scale = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("context_scale", None)
        if use_saliency:
            self.saliency_scale = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("saliency_scale", None)
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    @staticmethod
    def _scale(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(scale).view(1, -1, 1, 1)
        return bounded * values

    def branches(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(f"CleanTPD2 requires even H/W, got {tuple(x.shape[-2:])}")
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        context = (
            F.avg_pool2d(x, kernel_size=2, stride=2) if self.use_context else None
        )
        saliency = None
        if self.use_saliency:
            if context is None:
                context_for_saliency = F.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                context_for_saliency = context
            saliency = (
                F.max_pool2d(x, kernel_size=2, stride=2) - context_for_saliency
            )
        return keep, context, saliency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, context, saliency = self.branches(x)
        output = keep
        if context is not None:
            output = output + self._scale(context, self.context_scale)
        if saliency is not None:
            output = output + self._scale(saliency, self.saliency_scale)
        return self.activation(output)


class CleanTPDPatchEmbedding(nn.Module):
    """Hierarchical SPD-anchored TPD embedding."""

    def __init__(
        self,
        channels: int,
        stride: int,
        *,
        use_context: bool,
        use_saliency: bool,
    ) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            CleanTPD2(
                channels,
                activate=index < steps - 1,
                use_context=use_context,
                use_saliency=use_saliency,
            )
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


class GroupedKeepBlock(nn.Module):
    """The grouped Keep path from TPD-v1 without Context or Saliency."""

    def __init__(self, channels: int, activate: bool) -> None:
        super().__init__()
        self.phase_compress = nn.Conv2d(
            4 * channels,
            channels,
            kernel_size=1,
            groups=channels,
            bias=False,
        )
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                f"GroupedKeepBlock requires even H/W, got {tuple(x.shape[-2:])}"
            )
        return self.activation(self.phase_compress(F.pixel_unshuffle(x, 2)))


class GroupedKeepPatchEmbedding(nn.Module):
    """Hierarchical grouped-Keep-only control."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            GroupedKeepBlock(channels, activate=index < steps - 1)
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


def build_clean_patch_embedding(variant: str, channels: int, stride: int) -> nn.Module:
    variant = variant.lower()
    if variant == "tpd_clean_ctx":
        return CleanTPDPatchEmbedding(
            channels, stride, use_context=True, use_saliency=False
        )
    if variant == "tpd_clean_sal":
        return CleanTPDPatchEmbedding(
            channels, stride, use_context=False, use_saliency=True
        )
    if variant == "tpd_clean_full":
        return CleanTPDPatchEmbedding(
            channels, stride, use_context=True, use_saliency=True
        )
    if variant == "grouped_keep":
        return GroupedKeepPatchEmbedding(channels, stride)
    raise ValueError(
        f"Cannot build clean variant {variant!r}; choices={SUPPORTED_CLEAN_VARIANTS}"
    )


def replace_shallow_embeddings_clean(
    model: nn.Module, variant: str
) -> Dict[str, nn.Module]:
    """Replace only embeddings_1/2 with a v2 candidate."""
    variant = variant.lower()
    if variant not in SUPPORTED_CLEAN_VARIANTS:
        raise ValueError(
            f"Unknown clean variant {variant!r}; choices={SUPPORTED_CLEAN_VARIANTS}"
        )
    replacements = {
        "embeddings_1": build_clean_patch_embedding(
            variant, channels=32, stride=16
        ),
        "embeddings_2": build_clean_patch_embedding(
            variant, channels=64, stride=8
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    return replacements


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
