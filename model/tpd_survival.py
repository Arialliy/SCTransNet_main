"""Target-survival heads for the composed TPD-SCTransNet model.

The two heads supervise the final ``emb1`` and ``emb2`` token endpoints.  Both
endpoints already live on the same stride-16 grid, so their logits share one
cell-presence target without resizing either feature tensor.

This module does not alter the segmentation path.  It only constructs the
optional :class:`~model.tpd_forward_contract.TPDForwardOutput` used by the
survival-training stage; evaluation continues to consume the unchanged final
segmentation probability map.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from model.tpd_forward_contract import (
    LegacySegmentationOutput,
    TPDForwardOutput,
)


SURVIVAL_ENDPOINT_CONTRACT = "post_tpd_emb1_emb2_stride16"


class TargetSurvivalHead(nn.Module):
    """A one-layer cell-presence classifier that returns raw logits."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        if (
            not isinstance(in_channels, int)
            or isinstance(in_channels, bool)
            or in_channels < 1
        ):
            raise ValueError(
                f"in_channels must be a positive integer, got {in_channels!r}"
            )
        self.in_channels = int(in_channels)
        self.classifier = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, endpoint: torch.Tensor) -> torch.Tensor:
        if not isinstance(endpoint, torch.Tensor):
            raise TypeError("survival endpoint must be a torch.Tensor")
        if endpoint.ndim != 4:
            raise ValueError(
                "survival endpoint must have shape BxCxHxW, "
                f"got {tuple(endpoint.shape)}"
            )
        if endpoint.shape[1] != self.in_channels:
            raise ValueError(
                f"survival endpoint requires {self.in_channels} channels, "
                f"got {endpoint.shape[1]}"
            )
        if min(endpoint.shape) < 1:
            raise ValueError("survival endpoint dimensions must be positive")
        if not endpoint.is_floating_point():
            raise TypeError("survival endpoint must use a floating-point dtype")
        if not torch.isfinite(endpoint).all():
            raise FloatingPointError("survival endpoint contains non-finite values")
        return self.classifier(endpoint)


class PairedTargetSurvivalHeads(nn.Module):
    """Independent ``emb1``/``emb2`` heads on one shared token grid."""

    endpoint_names = ("emb1", "emb2")

    def __init__(
        self,
        emb1_channels: int = 32,
        emb2_channels: int = 64,
    ) -> None:
        super().__init__()
        for name, value in (
            ("emb1_channels", emb1_channels),
            ("emb2_channels", emb2_channels),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        if emb2_channels != 2 * emb1_channels:
            raise ValueError(
                "emb2_channels must equal twice emb1_channels, "
                f"got {emb2_channels} and {emb1_channels}"
            )
        self.emb1_channels = int(emb1_channels)
        self.emb2_channels = int(emb2_channels)
        self.heads = nn.ModuleDict(
            {
                "emb1": TargetSurvivalHead(self.emb1_channels),
                "emb2": TargetSurvivalHead(self.emb2_channels),
            }
        )

    def forward(
        self,
        emb1_endpoint: torch.Tensor,
        emb2_endpoint: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(emb1_endpoint, torch.Tensor) or not isinstance(
            emb2_endpoint, torch.Tensor
        ):
            raise TypeError("both survival endpoints must be torch.Tensor values")
        if emb1_endpoint.ndim != 4 or emb2_endpoint.ndim != 4:
            raise ValueError("both survival endpoints must have shape BxCxHxW")
        if emb1_endpoint.shape[0] != emb2_endpoint.shape[0]:
            raise ValueError("survival endpoints must share one batch size")
        if emb1_endpoint.shape[-2:] != emb2_endpoint.shape[-2:]:
            raise ValueError("survival endpoints must share one spatial token grid")
        return (
            self.heads["emb1"](emb1_endpoint),
            self.heads["emb2"](emb2_endpoint),
        )

    def architecture_manifest(self) -> Dict[str, object]:
        return {
            "module": "PairedTargetSurvivalHeads",
            "supervised_endpoints": self.endpoint_names,
            "endpoint_channels": (
                self.emb1_channels,
                self.emb2_channels,
            ),
            "classifier": "independent_conv1x1_logits",
            "target_grid": "stride_16_max_presence",
            "endpoint_contract": SURVIVAL_ENDPOINT_CONTRACT,
            "segmentation_path_modified": False,
            "inference_heads_required": False,
        }


def build_structured_survival_output(
    segmentation: LegacySegmentationOutput,
    emb1_endpoint: torch.Tensor,
    emb2_endpoint: torch.Tensor,
    heads: PairedTargetSurvivalHeads,
) -> TPDForwardOutput:
    """Attach auxiliary logits without mixing them into segmentation outputs."""

    if not isinstance(heads, PairedTargetSurvivalHeads):
        raise TypeError("heads must be PairedTargetSurvivalHeads")
    emb1_logits, emb2_logits = heads(emb1_endpoint, emb2_endpoint)
    return TPDForwardOutput(
        segmentation=segmentation,
        emb1_endpoint=emb1_endpoint,
        emb2_endpoint=emb2_endpoint,
        emb1_survival_logits=emb1_logits,
        emb2_survival_logits=emb2_logits,
    )


def survival_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "PairedTargetSurvivalHeads",
    "SURVIVAL_ENDPOINT_CONTRACT",
    "TargetSurvivalHead",
    "build_structured_survival_output",
    "survival_parameter_count",
]
