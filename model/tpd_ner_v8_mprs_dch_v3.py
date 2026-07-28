"""Post-centering DC-calibrated V3 relay for V8-MPRS-DCH SCTransNet.

V3 keeps the complete V2 model path unchanged except for one deliberately
small degree of freedom: each of the three spatial relay stages owns one
zero-initialized scalar DC offset.  The offset is added *after* per-sample
spatial centering and before the bounded arctangent mapping.  Consequently,
the centered spatial evidence remains V2-identical while each stage can learn
an independent global background calibration.

The SCTransNet parent, Keep--Context--Saliency tokenizer, MPRS-DCH blocks,
five evidence nodes, and ``q4 -> q3 -> q2`` topology are inherited unchanged.
Zero gate weights and zero DC offsets make relay-on V3 exactly identical to
the V2/relay-off path at step zero.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from model.SCTransNet import SCTransNet
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    EVIDENCE_NODE_NAMES,
    PRODUCTION_PARENT_PARAMETERS,
    RELAY_STAGE_ORDER,
    TPDNERV8MPRSDCHSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v2 import (
    RELAY_RMS_EPS,
    RMSBalancedCenteredEvidenceRelay,
    TPDNERV8MPRSDCHV2SCTransNet,
    V2_MASK_LIMIT,
    V2_SKIP_FACTOR_BOUNDS,
    adapt_v8_mprs_dch_parent_v2,
    arctangent_residual_mask,
    spatially_center_gate_logits,
)


SpatialSize = Tuple[int, int]
V3_RELAY_VERSION = "v3_rms_centered_arctangent_post_center_dc"
PRODUCTION_V3_RELAY_PARAMETERS = 11_291
PRODUCTION_V3_RELAY_ON_PARAMETERS = 10_854_446


class RMSBalancedCenteredDCOffsetEvidenceRelay(
    RMSBalancedCenteredEvidenceRelay
):
    """V2 relay plus one post-centering scalar DC offset per decoder stage."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        width: int = DEFAULT_RELAY_WIDTH,
        eps: float = RELAY_RMS_EPS,
    ) -> None:
        super().__init__(
            base_channels=base_channels,
            width=width,
            eps=eps,
        )
        self.dc_offsets = nn.ParameterDict(
            {
                str(stage): nn.Parameter(torch.zeros(1))
                for stage in RELAY_STAGE_ORDER
            }
        )
        self.zero_init_gates()

    def zero_init_gates(self) -> None:
        """Reset both the inherited spatial gates and the V3 DC calibration."""

        super().zero_init_gates()
        offsets = getattr(self, "dc_offsets", None)
        if offsets is not None:
            for offset in offsets.values():
                nn.init.zeros_(offset)

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

        relay_value = self.fusions[str(stage)](sources, output_size)
        logits = self.gates[str(stage)](relay_value)
        centered_logits = spatially_center_gate_logits(logits)
        shifted_logits = centered_logits + self.dc_offsets[str(stage)].view(
            1,
            1,
            1,
            1,
        )
        mask = arctangent_residual_mask(shifted_logits)
        return relay_value, mask


def _initialize_v3_relay(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TPDNERV8MPRSDCHV3SCTransNet(TPDNERV8MPRSDCHV2SCTransNet):
    """V2 complete model with post-centering stagewise DC calibration."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
        )
        embedding = self.mtc.embeddings_1
        base_channels = embedding.blocks[0].channels
        if not isinstance(base_channels, int) or base_channels < 1:
            raise RuntimeError("cannot infer positive V8 base channels")

        # Rebuild only the relay with the same isolated seed as V2.  The three
        # zero-valued offsets consume no random numbers, so every shared V2
        # relay tensor starts identically.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(relay_initialization_seed)
            relay = RMSBalancedCenteredDCOffsetEvidenceRelay(
                base_channels=base_channels,
                width=relay_width,
                eps=RELAY_RMS_EPS,
            )
            relay.apply(_initialize_v3_relay)
        reference = next(self.parameters())
        relay.to(device=reference.device, dtype=reference.dtype)
        relay.zero_init_gates()
        self.tpd_ner = relay

    def architecture_manifest(self) -> Dict[str, object]:
        manifest = dict(super().architecture_manifest())
        manifest.update(
            {
                "relay_version": V3_RELAY_VERSION,
                "gate_bias": False,
                "gate_spatial_centering": "per_sample_mean_hw",
                "gate_dc_offset": "learned_per_stage_post_centering",
                "gate_dc_offset_count": len(RELAY_STAGE_ORDER),
                "gate_dc_offset_initialization": "zero",
                "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
                "mask_mapping": "atan(pi*(centered+dc))/pi",
                "mask_bounds": (-V2_MASK_LIMIT, V2_MASK_LIMIT),
                "skip_factor_bounds": V2_SKIP_FACTOR_BOUNDS,
                "zero_gate_reference": "v2_and_relay_off_exact",
            }
        )
        return manifest


def adapt_v8_mprs_dch_parent_v3(
    parent: SCTransNet,
    *,
    variant: str,
    relay_enabled: bool,
    relay_width: int = DEFAULT_RELAY_WIDTH,
    relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
) -> TPDNERV8MPRSDCHSCTransNet:
    """Return unchanged relay-off or the post-centering-DC relay-on V3."""

    if relay_width != DEFAULT_RELAY_WIDTH:
        raise ValueError(
            f"V3 relay width is fixed to {DEFAULT_RELAY_WIDTH}, "
            f"got {relay_width}"
        )
    if not relay_enabled:
        return adapt_v8_mprs_dch_parent_v2(
            parent,
            variant=variant,
            relay_enabled=False,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
        )
    return TPDNERV8MPRSDCHV3SCTransNet(
        parent,
        variant=variant,
        relay_width=relay_width,
        relay_initialization_seed=relay_initialization_seed,
    )


def v3_relay_parameter_count(model: nn.Module) -> int:
    relay = getattr(model, "tpd_ner", None)
    if relay is None:
        return 0
    return sum(parameter.numel() for parameter in relay.parameters())


__all__ = [
    "DEFAULT_RELAY_INITIALIZATION_SEED",
    "DEFAULT_RELAY_WIDTH",
    "EVIDENCE_NODE_NAMES",
    "PRODUCTION_PARENT_PARAMETERS",
    "PRODUCTION_V3_RELAY_ON_PARAMETERS",
    "PRODUCTION_V3_RELAY_PARAMETERS",
    "RELAY_RMS_EPS",
    "RELAY_STAGE_ORDER",
    "RMSBalancedCenteredDCOffsetEvidenceRelay",
    "TPDNERV8MPRSDCHV3SCTransNet",
    "V2_MASK_LIMIT",
    "V2_SKIP_FACTOR_BOUNDS",
    "V3_RELAY_VERSION",
    "adapt_v8_mprs_dch_parent_v3",
    "v3_relay_parameter_count",
]
