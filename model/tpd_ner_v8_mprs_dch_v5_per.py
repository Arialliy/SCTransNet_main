"""NER V5-PER: persistent-evidence routing of positive stage-2 gates.

V5-PER is a state-layout-preserving refinement of the frozen V4 Tail-Aware
relay.  Stages 4 and 3 delegate to V4 exactly.  At stage 2 only, the positive
part of the spatially centred gate is multiplied by the stop-gradient
persistent upper-tail support ``P`` while the negative part is retained:

``routed = centered - (1 - P) * relu(centered)``.

The inherited ``dc_support`` API can represent three different V4 policies,
but the expression above is valid only when it returns ``1 - P``.  Therefore
V5-PER deliberately accepts the formal ``complement_tail`` policy only.
There are no new parameters or buffers; raw tensor state is layout-compatible
with V4, while the checkpoint semantics are intentionally not interchangeable.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_WIDTH,
    RELAY_STAGE_ORDER,
)
from model.tpd_ner_v8_mprs_dch_v2 import (
    RELAY_RMS_EPS,
    arctangent_residual_mask,
    spatially_center_gate_logits,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_TAIL_Z_THRESHOLDS,
    TailAwarePersistentDCOffsetEvidenceRelay,
    TailDCSupportMode,
)


SpatialSize = Tuple[int, int]
V5_PER_RELAY_VERSION = "v5_stage2_persistent_evidence_positive_routing"
V5_PER_FORMAL_DC_SUPPORT_MODE = TailDCSupportMode.COMPLEMENT_TAIL.value


def _require_complement_tail(mode: str) -> None:
    if mode != V5_PER_FORMAL_DC_SUPPORT_MODE:
        raise ValueError(
            "NER V5-PER requires dc_support_mode='complement_tail'; "
            f"got {mode!r}"
        )


class PersistentEvidencePositiveRoutingRelay(
    TailAwarePersistentDCOffsetEvidenceRelay
):
    """V4 exact at stages 4/3; route positive stage-2 evidence through ``P``."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        width: int = DEFAULT_RELAY_WIDTH,
        eps: float = RELAY_RMS_EPS,
        dc_support_mode: Union[
            str,
            TailDCSupportMode,
        ] = V5_PER_FORMAL_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[
            int,
            float,
        ] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            base_channels=base_channels,
            width=width,
            eps=eps,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        _require_complement_tail(self.dc_support_mode)

    def forward_stage(
        self,
        stage: int,
        sources: Sequence[torch.Tensor],
        output_size: SpatialSize,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if stage not in RELAY_STAGE_ORDER:
            raise ValueError(f"relay stage must be 4, 3, or 2, got {stage}")
        if len(output_size) != 2 or min(output_size) < 1:
            raise ValueError(f"invalid relay output size: {output_size}")
        if stage != 2:
            return super().forward_stage(stage, sources, output_size)

        # Refuse a mutated/non-formal relay before interpreting dc_support as B.
        _require_complement_tail(self.dc_support_mode)
        relay_value = self.fusions["2"](sources, output_size)
        centered = spatially_center_gate_logits(
            self.gates["2"](relay_value)
        )
        background_support = self.dc_support(
            2,
            relay_value,
            sources,
            output_size,
        ).detach()

        if tuple(background_support.shape) != tuple(centered.shape):
            raise RuntimeError(
                "V5-PER stage-2 support shape differs from centered logits: "
                f"{tuple(background_support.shape)} != {tuple(centered.shape)}"
            )
        if (
            background_support.dtype != centered.dtype
            or background_support.device != centered.device
        ):
            raise RuntimeError(
                "V5-PER stage-2 support dtype/device differs from centered logits"
            )
        if background_support.requires_grad:
            raise RuntimeError("V5-PER persistent support must stop gradients")
        if not bool(torch.isfinite(background_support).all()):
            raise FloatingPointError("V5-PER stage-2 support must be finite")
        if not bool(
            ((background_support >= 0.0) & (background_support <= 1.0)).all()
        ):
            raise FloatingPointError(
                "V5-PER stage-2 complement support must lie in [0, 1]"
            )

        routed_centered = centered - background_support * F.relu(centered)
        shifted = routed_centered + (
            self.dc_offsets["2"].view(1, 1, 1, 1)
            * background_support
        )
        return relay_value, arctangent_residual_mask(shifted)


def replace_v4_relay_with_v5(
    relay: TailAwarePersistentDCOffsetEvidenceRelay,
) -> PersistentEvidencePositiveRoutingRelay:
    """Return an RNG-neutral V5 relay containing the exact supplied V4 state."""

    if type(relay) is not TailAwarePersistentDCOffsetEvidenceRelay:
        raise TypeError(
            "V5-PER replacement requires the exact V4 Tail-Aware relay"
        )
    _require_complement_tail(relay.dc_support_mode)
    reference = next(relay.parameters())
    before_keys = tuple(relay.state_dict())
    before_parameter_count = sum(
        parameter.numel() for parameter in relay.parameters()
    )

    # Conv construction consumes the CPU RNG.  V5 has no new random degree of
    # freedom, so isolate construction and immediately restore the V4 tensors.
    with torch.random.fork_rng(devices=[]):
        replacement = PersistentEvidencePositiveRoutingRelay(
            base_channels=relay.base_channels,
            width=relay.width,
            eps=relay.eps,
            dc_support_mode=V5_PER_FORMAL_DC_SUPPORT_MODE,
            tail_z_thresholds=relay.tail_z_thresholds,
        )
    replacement.to(device=reference.device, dtype=reference.dtype)
    incompatible = replacement.load_state_dict(relay.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V4-to-V5 strict relay load returned incompatible keys")
    replacement.train(relay.training)

    if tuple(replacement.state_dict()) != before_keys:
        raise RuntimeError("V5-PER relay state layout differs from V4")
    if (
        sum(parameter.numel() for parameter in replacement.parameters())
        != before_parameter_count
    ):
        raise RuntimeError("V5-PER relay parameter count differs from V4")
    return replacement


def v5_per_manifest_fields() -> Dict[str, object]:
    """Return the non-state semantic identity of the V5-PER relay."""

    return {
        "relay_version": V5_PER_RELAY_VERSION,
        "ner_version": "v5_per",
        "stage4_formula": "v4_exact",
        "stage3_formula": "v4_exact",
        "stage2_positive_route": (
            "centered-(1-persistent)*relu(centered)"
        ),
        "stage2_negative_route": "unchanged_identity_path",
        "stage2_dc_support": "one_minus_persistent_tail",
        "stage2_persistent_support_gradient": "stopped",
        "ner_v5_per_dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
        "parameters_added_vs_v4": 0,
        "buffers_added_vs_v4": 0,
        "state_layout_compatible_with": "ner_v4_tail_aware",
        "state_semantics_identical_to_v4": False,
        "checkpoint_semantically_interchangeable_with_v4": False,
        "v4_to_v5_optimizer_resume_allowed": False,
    }


__all__ = [
    "PersistentEvidencePositiveRoutingRelay",
    "V5_PER_FORMAL_DC_SUPPORT_MODE",
    "V5_PER_RELAY_VERSION",
    "replace_v4_relay_with_v5",
    "v5_per_manifest_fields",
]
