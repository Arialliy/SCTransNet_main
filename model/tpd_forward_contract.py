"""Typed forward-output boundary for the composed TPD-SCTransNet model.

This module defines an output contract only.  It does not wrap or modify the
baseline model.  Consequently, the default model ``forward`` may keep returning
its legacy output: either one segmentation probability map or exactly six
full-resolution deep-supervision probability maps.

Token endpoints and survival logits belong to the optional structured result.
They can therefore never be mistaken for legacy segmentation outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch


SixSegmentationMaps: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
LegacySegmentationOutput: TypeAlias = torch.Tensor | SixSegmentationMaps


def _map_shape(
    name: str,
    value: torch.Tensor,
    *,
    channels: int | None = None,
) -> tuple[int, int, int, int]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(
            f"{name} must have shape BxCxHxW, got {tuple(value.shape)}"
        )
    batch, actual_channels, height, width = value.shape
    if min(batch, actual_channels, height, width) <= 0:
        raise ValueError(
            f"{name} must have positive B, C, H and W, "
            f"got {tuple(value.shape)}"
        )
    if channels is not None and actual_channels != channels:
        raise ValueError(
            f"{name} must have {channels} channel(s), "
            f"got {actual_channels}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    return batch, actual_channels, height, width


def _segmentation_maps(
    output: LegacySegmentationOutput,
) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        maps = (output,)
    elif isinstance(output, tuple):
        if len(output) != 6:
            raise ValueError(
                "segmentation must be one Tensor or exactly six "
                f"full-resolution Tensors, got {len(output)}"
            )
        maps = output
    else:
        raise TypeError(
            "segmentation must be one Tensor or a tuple of exactly six "
            "Tensors"
        )

    reference_shape = _map_shape("segmentation[0]", maps[0], channels=1)
    for index, probability_map in enumerate(maps[1:], start=1):
        current_shape = _map_shape(
            f"segmentation[{index}]",
            probability_map,
            channels=1,
        )
        if current_shape != reference_shape:
            raise ValueError(
                "all segmentation outputs must be full-resolution maps with "
                "the same Bx1xHxW shape; low-resolution auxiliary outputs "
                f"must not be mixed into segmentation (index {index}: "
                f"{current_shape}, expected {reference_shape})"
            )
    return maps


def _paired(
    first_name: str,
    first: torch.Tensor | None,
    second_name: str,
    second: torch.Tensor | None,
) -> bool:
    if (first is None) != (second is None):
        raise ValueError(
            f"{first_name} and {second_name} must either both be present "
            "or both be absent"
        )
    return first is not None


@dataclass(frozen=True, slots=True)
class TPDForwardOutput:
    """Optional structured result for the final composed model.

    ``segmentation`` remains exactly legacy-compatible.  The remaining fields
    are paired, optional training/debug information and are deliberately kept
    out of ``legacy_output`` and ``evaluator_prediction``.
    """

    segmentation: LegacySegmentationOutput
    emb1_endpoint: torch.Tensor | None = None
    emb2_endpoint: torch.Tensor | None = None
    emb1_survival_logits: torch.Tensor | None = None
    emb2_survival_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        segmentation_maps = _segmentation_maps(self.segmentation)
        batch = segmentation_maps[0].shape[0]

        endpoints_present = _paired(
            "emb1_endpoint",
            self.emb1_endpoint,
            "emb2_endpoint",
            self.emb2_endpoint,
        )
        endpoint_space: tuple[int, int] | None = None
        if endpoints_present:
            assert self.emb1_endpoint is not None
            assert self.emb2_endpoint is not None
            emb1_shape = _map_shape("emb1_endpoint", self.emb1_endpoint)
            emb2_shape = _map_shape("emb2_endpoint", self.emb2_endpoint)
            if emb1_shape[0] != batch or emb2_shape[0] != batch:
                raise ValueError(
                    "token endpoint batch size must match segmentation: "
                    f"got {emb1_shape[0]} and {emb2_shape[0]}, expected "
                    f"{batch}"
                )
            if emb1_shape[2:] != emb2_shape[2:]:
                raise ValueError(
                    "emb1_endpoint and emb2_endpoint must share one spatial "
                    f"token space, got {emb1_shape[2:]} and {emb2_shape[2:]}"
                )
            if emb2_shape[1] != 2 * emb1_shape[1]:
                raise ValueError(
                    "emb2_endpoint channels must equal twice emb1_endpoint "
                    f"channels, got {emb2_shape[1]} and {emb1_shape[1]}"
                )
            endpoint_space = emb1_shape[2:]

        logits_present = _paired(
            "emb1_survival_logits",
            self.emb1_survival_logits,
            "emb2_survival_logits",
            self.emb2_survival_logits,
        )
        if logits_present:
            assert self.emb1_survival_logits is not None
            assert self.emb2_survival_logits is not None
            logit1_shape = _map_shape(
                "emb1_survival_logits",
                self.emb1_survival_logits,
                channels=1,
            )
            logit2_shape = _map_shape(
                "emb2_survival_logits",
                self.emb2_survival_logits,
                channels=1,
            )
            if logit1_shape[0] != batch or logit2_shape[0] != batch:
                raise ValueError(
                    "survival-logit batch size must match segmentation: "
                    f"got {logit1_shape[0]} and {logit2_shape[0]}, expected "
                    f"{batch}"
                )
            if logit1_shape[2:] != logit2_shape[2:]:
                raise ValueError(
                    "the two survival logits must share one spatial token "
                    f"space, got {logit1_shape[2:]} and {logit2_shape[2:]}"
                )
            if endpoint_space is not None and logit1_shape[2:] != endpoint_space:
                raise ValueError(
                    "survival logits must use the token endpoint spatial "
                    f"space, got {logit1_shape[2:]}, expected {endpoint_space}"
                )

    @property
    def final_prediction(self) -> torch.Tensor:
        """Return the evaluator-facing final full-resolution probability map."""

        return _segmentation_maps(self.segmentation)[-1]

    @property
    def token_endpoints(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the paired token endpoints without changing legacy output."""

        if self.emb1_endpoint is None:
            return None
        assert self.emb2_endpoint is not None
        return self.emb1_endpoint, self.emb2_endpoint

    @property
    def survival_logits(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the paired low-resolution survival logits, when requested."""

        if self.emb1_survival_logits is None:
            return None
        assert self.emb2_survival_logits is not None
        return self.emb1_survival_logits, self.emb2_survival_logits

    def legacy_output(self) -> LegacySegmentationOutput:
        """Return exactly the single-Tensor or six-Tensor baseline form."""

        return self.segmentation

    def evaluator_prediction(self) -> torch.Tensor:
        """Return only the final full-resolution probability map."""

        return self.final_prediction


ForwardOutput: TypeAlias = LegacySegmentationOutput | TPDForwardOutput


def legacy_output(output: ForwardOutput) -> LegacySegmentationOutput:
    """Extract and validate a legacy-compatible model output."""

    if isinstance(output, TPDForwardOutput):
        return output.legacy_output()
    _segmentation_maps(output)
    return output


def evaluator_prediction(output: ForwardOutput) -> torch.Tensor:
    """Extract the final segmentation map from legacy or structured output."""

    if isinstance(output, TPDForwardOutput):
        return output.evaluator_prediction()
    return _segmentation_maps(output)[-1]


__all__ = [
    "ForwardOutput",
    "LegacySegmentationOutput",
    "SixSegmentationMaps",
    "TPDForwardOutput",
    "evaluator_prediction",
    "legacy_output",
]
