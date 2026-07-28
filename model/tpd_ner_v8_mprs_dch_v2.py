"""RMS-balanced, centered-gate V2 relay for V8-MPRS-DCH SCTransNet.

V2 changes only the optional five-node Nested Evidence Relay (NER).  The
SCTransNet parent, Keep--Context--Saliency tokenizer, MPRS-DCH computation,
five evidence nodes, and ``q4 -> q3 -> q2`` topology remain the V1
implementation.

Relay-on V2 applies two numerical controls:

1. every aligned source projection and every fused relay value is normalized
   by its per-sample full-tensor RMS; and
2. each bias-free spatial gate is mean-centered before the bounded mapping
   ``atan(pi * z) / pi``.

The three gate weights remain exactly zero initialized.  Consequently the
installed relay is an exact step-zero identity relative to the unchanged V1
relay-off adapter.  Relay-off construction delegates directly to that V1
adapter and therefore keeps its class, state keys, and forward path unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    EVIDENCE_NODE_NAMES,
    PRODUCTION_PARENT_PARAMETERS,
    RELAY_STAGE_ORDER,
    TPDNERV8MPRSDCHSCTransNet,
    adapt_v8_mprs_dch_parent,
)
from model.tpd_sctransnet import ExplicitRelayUpBlock


SpatialSize = Tuple[int, int]
RELAY_RMS_EPS = 1e-6
V2_MASK_LIMIT = 0.5
V2_SKIP_FACTOR_BOUNDS = (0.5, 1.5)
PRODUCTION_V2_RELAY_PARAMETERS = 11_288
PRODUCTION_V2_RELAY_ON_PARAMETERS = 10_854_443


def _working_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def sample_full_tensor_rms_normalize(
    tensor: torch.Tensor,
    *,
    eps: float = RELAY_RMS_EPS,
) -> torch.Tensor:
    """Normalize each BCHW sample by one full-tensor RMS value.

    A zero sample stays exactly zero and finite.  FP16/BF16 reductions are
    evaluated in FP32, then returned in the input dtype.
    """

    if tensor.ndim != 4:
        raise ValueError(
            "RMS-balanced relay tensors must be BCHW, "
            f"got shape={tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError("RMS-balanced relay tensors must be floating point")
    if eps <= 0.0:
        raise ValueError(f"RMS epsilon must be positive, got {eps}")
    working = _working_float(tensor)
    inverse_rms = torch.rsqrt(
        working.square().mean(dim=(1, 2, 3), keepdim=True) + eps
    )
    normalized = working * inverse_rms
    return normalized.to(dtype=tensor.dtype)


def spatially_center_gate_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove only the per-sample spatial mean from one-channel gate logits."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "V2 gate logits must have shape Bx1xHxW, "
            f"got {tuple(logits.shape)}"
        )
    working = _working_float(logits)
    centered = working - working.mean(dim=(-2, -1), keepdim=True)
    return centered.to(dtype=logits.dtype)


def arctangent_residual_mask(centered_logits: torch.Tensor) -> torch.Tensor:
    """Map finite centered logits into ``(-0.5, 0.5)`` with unit slope at zero."""

    if not centered_logits.is_floating_point():
        raise TypeError("V2 centered gate logits must be floating point")
    working = _working_float(centered_logits)
    if not bool(torch.isfinite(working).all()):
        raise FloatingPointError("V2 centered gate logits must be finite")
    mask = torch.atan(math.pi * working) / math.pi
    mask = mask.to(dtype=centered_logits.dtype)
    # atan approaches +/-0.5 without reaching it over the real numbers, but
    # finite-precision rounding can produce an endpoint for very large finite
    # logits.  Clamping the mask to nextafter(0.5, 0) is not sufficient:
    # adding one can still round the skip factor to exactly 0.5 or 1.5.
    # Derive asymmetric mask limits from the adjacent representable factors
    # inside (0.5, 1.5), in the actual output dtype including FP16/BF16.
    one = mask.new_tensor(1.0)
    lower_factor = torch.nextafter(
        mask.new_tensor(V2_SKIP_FACTOR_BOUNDS[0]),
        one,
    )
    upper_factor = torch.nextafter(
        mask.new_tensor(V2_SKIP_FACTOR_BOUNDS[1]),
        one,
    )
    lower_mask = lower_factor - one
    upper_mask = upper_factor - one
    return torch.clamp(mask, min=lower_mask, max=upper_mask)


class RMSBalancedRelayFusionCell(nn.Module):
    """Project, align, RMS-balance, and fuse one fixed source set exactly once."""

    def __init__(
        self,
        source_channels: Sequence[int],
        width: int = DEFAULT_RELAY_WIDTH,
        *,
        eps: float = RELAY_RMS_EPS,
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(f"relay width must be positive, got {width}")
        if not source_channels or any(channels < 1 for channels in source_channels):
            raise ValueError(f"invalid relay source channels: {source_channels}")
        if eps <= 0.0:
            raise ValueError(f"RMS epsilon must be positive, got {eps}")
        self.width = int(width)
        self.source_channels = tuple(int(value) for value in source_channels)
        self.eps = float(eps)
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, self.width, kernel_size=1, bias=False)
            for channels in self.source_channels
        )
        self.fuse = nn.Conv2d(
            len(self.source_channels) * self.width,
            self.width,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        output_size: SpatialSize,
    ) -> torch.Tensor:
        if len(sources) != len(self.projections):
            raise ValueError(
                f"expected {len(self.projections)} relay sources, got {len(sources)}"
            )
        if len(output_size) != 2 or min(output_size) < 1:
            raise ValueError(f"invalid relay output size: {output_size}")

        normalized_sources = []
        for index, (source, projection, expected_channels) in enumerate(
            zip(sources, self.projections, self.source_channels)
        ):
            if source.ndim != 4:
                raise ValueError(
                    f"relay source {index} must be BCHW, "
                    f"got shape={tuple(source.shape)}"
                )
            if source.shape[1] != expected_channels:
                raise ValueError(
                    f"relay source {index} requires C={expected_channels}, "
                    f"got C={source.shape[1]}"
                )
            value = projection(source)
            if value.shape[-2:] != output_size:
                value = F.interpolate(
                    value,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            normalized_sources.append(
                sample_full_tensor_rms_normalize(value, eps=self.eps)
            )

        fused = self.activation(
            self.fuse(torch.cat(normalized_sources, dim=1))
        )
        return sample_full_tensor_rms_normalize(fused, eps=self.eps)


class RMSBalancedCenteredEvidenceRelay(nn.Module):
    """Bias-free V2 five-node relay with centered arctangent residual masks."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        width: int = DEFAULT_RELAY_WIDTH,
        eps: float = RELAY_RMS_EPS,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError(
                f"base_channels must be positive, got {base_channels}"
            )
        if width != DEFAULT_RELAY_WIDTH:
            raise ValueError(
                f"V2 relay width is fixed to {DEFAULT_RELAY_WIDTH}, got {width}"
            )
        if eps != RELAY_RMS_EPS:
            raise ValueError(
                f"V2 RMS epsilon is fixed to {RELAY_RMS_EPS}, got {eps}"
            )
        self.base_channels = int(base_channels)
        self.width = int(width)
        self.eps = float(eps)
        self.fusions = nn.ModuleDict(
            {
                "4": RMSBalancedRelayFusionCell(
                    (
                        self.base_channels,
                        2 * self.base_channels,
                        8 * self.base_channels,
                    ),
                    self.width,
                    eps=self.eps,
                ),
                "3": RMSBalancedRelayFusionCell(
                    (
                        self.base_channels,
                        2 * self.base_channels,
                        self.width,
                        4 * self.base_channels,
                    ),
                    self.width,
                    eps=self.eps,
                ),
                "2": RMSBalancedRelayFusionCell(
                    (
                        self.base_channels,
                        self.width,
                        2 * self.base_channels,
                    ),
                    self.width,
                    eps=self.eps,
                ),
            }
        )
        self.gates = nn.ModuleDict(
            {
                str(stage): nn.Conv2d(
                    self.width,
                    1,
                    kernel_size=1,
                    bias=False,
                )
                for stage in RELAY_STAGE_ORDER
            }
        )
        self.zero_init_gates()

    def zero_init_gates(self) -> None:
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)

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
        mask = arctangent_residual_mask(centered_logits)
        return relay_value, mask


def _initialize_v2_relay(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TPDNERV8MPRSDCHV2SCTransNet(TPDNERV8MPRSDCHSCTransNet):
    """Unchanged V8 parent plus the RMS-balanced centered-gate V2 relay."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
    ) -> None:
        if relay_width != DEFAULT_RELAY_WIDTH:
            raise ValueError(
                f"V2 relay width is fixed to {DEFAULT_RELAY_WIDTH}, "
                f"got {relay_width}"
            )
        if relay_initialization_seed < 0:
            raise ValueError("relay_initialization_seed must be non-negative")

        # Copy and validate the parent through V1's relay-off path.  No V1
        # relay is constructed or executed before the V2 relay is installed.
        super().__init__(
            parent,
            variant=variant,
            relay_enabled=False,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
        )
        embedding = self.mtc.embeddings_1
        base_channels = embedding.blocks[0].channels
        if not isinstance(base_channels, int) or base_channels < 1:
            raise RuntimeError("cannot infer positive V8 base channels")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(relay_initialization_seed)
            relay = RMSBalancedCenteredEvidenceRelay(
                base_channels=base_channels,
                width=relay_width,
                eps=RELAY_RMS_EPS,
            )
            relay.apply(_initialize_v2_relay)
        reference = next(self.parameters())
        relay.to(device=reference.device, dtype=reference.dtype)
        relay.zero_init_gates()

        self.up_decoder4 = ExplicitRelayUpBlock.from_existing(
            self.up_decoder4,
            stage=4,
        )
        self.up_decoder3 = ExplicitRelayUpBlock.from_existing(
            self.up_decoder3,
            stage=3,
        )
        self.up_decoder2 = ExplicitRelayUpBlock.from_existing(
            self.up_decoder2,
            stage=2,
        )
        self.add_module("tpd_ner", relay)
        self.relay_enabled = True
        self._nested_relay_installed = True
        self.relay_width = int(relay_width)
        self.relay_initialization_seed = int(relay_initialization_seed)

    def architecture_manifest(self) -> Dict[str, object]:
        manifest = dict(super().architecture_manifest())
        manifest.update(
            {
                "relay_version": "v2_rms_centered_arctangent",
                "relay_width": DEFAULT_RELAY_WIDTH,
                "relay_rms_scope": "per_sample_full_tensor",
                "source_projection_rms_normalized": True,
                "fusion_relu_output_rms_normalized": True,
                "relay_rms_eps": RELAY_RMS_EPS,
                "gate_bias": False,
                "gate_spatial_centering": "per_sample_mean_hw",
                "mask_mapping": "atan(pi*z)/pi",
                "mask_bounds": (-V2_MASK_LIMIT, V2_MASK_LIMIT),
                "skip_factor_bounds": V2_SKIP_FACTOR_BOUNDS,
                "mask_zero_derivative": 1.0,
                "zero_gate_reference": "v1_relay_off_exact",
            }
        )
        return manifest


def adapt_v8_mprs_dch_parent_v2(
    parent: SCTransNet,
    *,
    variant: str,
    relay_enabled: bool,
    relay_width: int = DEFAULT_RELAY_WIDTH,
    relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
) -> TPDNERV8MPRSDCHSCTransNet:
    """Return unchanged V1 relay-off or the single combined relay-on V2."""

    if relay_width != DEFAULT_RELAY_WIDTH:
        raise ValueError(
            f"V2 relay width is fixed to {DEFAULT_RELAY_WIDTH}, got {relay_width}"
        )
    if not relay_enabled:
        return adapt_v8_mprs_dch_parent(
            parent,
            variant=variant,
            relay_enabled=False,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
        )
    return TPDNERV8MPRSDCHV2SCTransNet(
        parent,
        variant=variant,
        relay_width=relay_width,
        relay_initialization_seed=relay_initialization_seed,
    )


def v2_relay_parameter_count(model: nn.Module) -> int:
    relay = getattr(model, "tpd_ner", None)
    if relay is None:
        return 0
    return sum(parameter.numel() for parameter in relay.parameters())


__all__ = [
    "DEFAULT_RELAY_INITIALIZATION_SEED",
    "DEFAULT_RELAY_WIDTH",
    "EVIDENCE_NODE_NAMES",
    "PRODUCTION_PARENT_PARAMETERS",
    "PRODUCTION_V2_RELAY_ON_PARAMETERS",
    "PRODUCTION_V2_RELAY_PARAMETERS",
    "RELAY_RMS_EPS",
    "RELAY_STAGE_ORDER",
    "RMSBalancedCenteredEvidenceRelay",
    "RMSBalancedRelayFusionCell",
    "TPDNERV8MPRSDCHV2SCTransNet",
    "V2_MASK_LIMIT",
    "V2_SKIP_FACTOR_BOUNDS",
    "adapt_v8_mprs_dch_parent_v2",
    "arctangent_residual_mask",
    "sample_full_tensor_rms_normalize",
    "spatially_center_gate_logits",
    "v2_relay_parameter_count",
]
