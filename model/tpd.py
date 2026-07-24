"""Target-preserving patch-embedding variants for controlled experiments.

The default SCTransNet code remains unchanged. Experiment runners replace only
``mtc.embeddings_1`` and ``mtc.embeddings_2`` with one of the modules here so
that encoder, SCTB, decoder, loss, and output interfaces stay fixed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_VARIANTS = ("original", "progressive", "spd", "tpd")


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))


class ProgressiveConvBlock(nn.Module):
    """Generic same-depth control: a learned overlapping stride-2 conv."""

    def __init__(self, channels: int, activate: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProgressivePatchEmbedding(nn.Module):
    """Replace one large-stride projection with repeated stride-2 convs."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            ProgressiveConvBlock(channels, activate=index < steps - 1)
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


class PhaseKeepBlock(nn.Module):
    """Canonical SPD control: space-to-depth plus dense 1x1 projection."""

    def __init__(self, channels: int, activate: bool) -> None:
        super().__init__()
        self.phase_compress = nn.Conv2d(4 * channels, channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(f"PhaseKeepBlock requires even H/W, got {tuple(x.shape[-2:])}")
        x = F.pixel_unshuffle(x, 2)
        x = self.phase_compress(x)
        return self.activation(x)


class SPDPatchEmbedding(nn.Module):
    """Canonical progressive SPD-Conv control."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            PhaseKeepBlock(channels, activate=index < steps - 1)
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


class TPD2(nn.Module):
    """One aligned 2x target-preserving downsampling unit.

    The three branches use the same non-overlapping 2x2 lattice:
    phase-aware rearrangement (keep), average context, and max-minus-average
    local saliency. Only non-terminal units use ReLU.
    """

    def __init__(self, channels: int, activate: bool) -> None:
        super().__init__()
        self.channels = channels
        self.phase_compress = nn.Conv2d(
            4 * channels,
            channels,
            kernel_size=1,
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def branches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(f"TPD2 requires even H/W, got {tuple(x.shape[-2:])}")
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return keep, context, saliency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, context, saliency = self.branches(x)
        output = self.fuse(torch.cat((keep, context, saliency), dim=1))
        return self.activation(output)


class TPDPatchEmbedding(nn.Module):
    """Hierarchical target-preserving embedding with optional intermediates."""

    def __init__(self, channels: int, stride: int, return_intermediates: bool = False) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.return_intermediates = return_intermediates
        self.blocks = nn.ModuleList(
            TPD2(channels, activate=index < steps - 1)
            for index in range(steps)
        )

    def forward(
        self, x: torch.Tensor | None
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, ...]] | None:
        if x is None:
            return None
        intermediates: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            intermediates.append(x)
        if self.return_intermediates:
            return x, tuple(intermediates[:-1])
        return x


def build_patch_embedding(variant: str, channels: int, stride: int) -> nn.Module:
    variant = variant.lower()
    if variant == "progressive":
        return ProgressivePatchEmbedding(channels, stride)
    if variant == "spd":
        return SPDPatchEmbedding(channels, stride)
    if variant == "tpd":
        return TPDPatchEmbedding(channels, stride)
    raise ValueError(f"Cannot build variant {variant!r}; expected one of {SUPPORTED_VARIANTS[1:]}")


def replace_shallow_embeddings(model: nn.Module, variant: str) -> Dict[str, nn.Module]:
    """Replace emb1/emb2 and return the newly created modules.

    ``original`` is a no-op. Call this after initializing the shared baseline
    model so every experiment starts from identical non-embedding weights.
    """
    variant = variant.lower()
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unknown embedding variant {variant!r}; choices={SUPPORTED_VARIANTS}")
    if variant == "original":
        return {}
    replacements = {
        "embeddings_1": build_patch_embedding(variant, channels=32, stride=16),
        "embeddings_2": build_patch_embedding(variant, channels=64, stride=8),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    return replacements


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
