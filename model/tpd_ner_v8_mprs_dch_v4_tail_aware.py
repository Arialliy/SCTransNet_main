"""Tail-aware NER DC-support V4 relay for V8-MPRS-DCH SCTransNet.

V4 preserves the V3 parent, five evidence nodes, q4 -> q3 -> q2 relay,
RMS-balanced fusion, bias-free centered gates, arctangent mask bounds, and
three stagewise zero-initialized DC parameters.

Stage 4 remains exactly V3-global.  Stages 3 and 2 use a parameter-free,
stop-gradient persistent-tail map.  The formal default is the
target-protective complement ``1 - P``: learned DC calibration remains
available in background-like regions while high-confidence persistent target
responses are protected from a global offset.  Two strictly enumerated
diagnostic modes reproduce V3-global and the earlier direct-tail proposal.

No mode or threshold is persistent model state.  Consequently V4 adds no
parameter or buffer and remains strict-state compatible with V3.  The selected
mode and immutable thresholds are nevertheless explicit in the architecture
manifest and must be checked by checkpoint loaders.
"""

from __future__ import annotations

import math
from enum import Enum
from types import MappingProxyType
from typing import Dict, Mapping, Sequence, Tuple, Union

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
)
from model.tpd_ner_v8_mprs_dch_v2 import (
    RELAY_RMS_EPS,
    V2_MASK_LIMIT,
    V2_SKIP_FACTOR_BOUNDS,
    adapt_v8_mprs_dch_parent_v2,
    arctangent_residual_mask,
    spatially_center_gate_logits,
)
from model.tpd_ner_v8_mprs_dch_v3 import (
    RMSBalancedCenteredDCOffsetEvidenceRelay,
    TPDNERV8MPRSDCHV3SCTransNet,
)


SpatialSize = Tuple[int, int]
V4_RELAY_VERSION = "v4_tail_aware_persistent_post_center_dch"
PRODUCTION_V4_RELAY_PARAMETERS = 11_291
PRODUCTION_V4_RELAY_ON_PARAMETERS = 10_854_446


class TailDCSupportMode(str, Enum):
    """The only three V4 DC-support policies allowed by the architecture."""

    LEGACY_GLOBAL = "legacy_global"
    DIRECT_TAIL = "direct_tail"
    COMPLEMENT_TAIL = "complement_tail"


SUPPORTED_DC_SUPPORT_MODES = tuple(mode.value for mode in TailDCSupportMode)
DEFAULT_DC_SUPPORT_MODE = TailDCSupportMode.COMPLEMENT_TAIL.value

# Immutable architecture constants.  They are deliberately not registered as
# buffers so that V3 and V4 retain identical state_dict keys.
_TAIL_Z_THRESHOLD_ITEMS: Tuple[Tuple[int, float], ...] = (
    (4, 1.5),
    (3, 2.0),
    (2, 2.5),
)
DEFAULT_TAIL_Z_THRESHOLDS: Mapping[int, float] = MappingProxyType(
    dict(_TAIL_Z_THRESHOLD_ITEMS)
)


def _working_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def _positive_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive finite real number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(
            f"{name} must be positive and finite, got {normalized}"
        )
    return normalized


def _stable_regularized_rms(
    tensor: torch.Tensor,
    *,
    dimensions: Union[int, Tuple[int, ...]],
    eps: float,
) -> torch.Tensor:
    """Compute ``sqrt(mean(x**2) + eps)`` without squaring large values."""

    absolute = tensor.abs()
    scale = absolute.amax(dim=dimensions, keepdim=True)
    one = torch.ones_like(scale)
    safe_scale = torch.where(scale > 0.0, scale, one)
    scaled = tensor / safe_scale
    scaled_mean_square = scaled.square().mean(
        dim=dimensions,
        keepdim=True,
    )
    # Finite scaled values lie in [-1, 1].  Clamp protects the subsequent
    # multiplication from a possible one-ulp reduction overshoot.
    scaled_rms = torch.sqrt(
        torch.clamp(scaled_mean_square, min=0.0, max=1.0)
    )
    unregularized_rms = scale * scaled_rms
    sqrt_eps = tensor.new_tensor(eps).sqrt()
    return torch.hypot(unregularized_rms, sqrt_eps)


def _stable_nonnegative_spatial_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Return a finite spatial mean without overflowing an intermediate sum."""

    scale = tensor.amax(dim=(-2, -1), keepdim=True)
    one = torch.ones_like(scale)
    safe_scale = torch.where(scale > 0.0, scale, one)
    scaled_mean = (tensor / safe_scale).mean(
        dim=(-2, -1),
        keepdim=True,
    )
    return scale * torch.clamp(scaled_mean, min=0.0, max=1.0)


def relay_spatial_tail_support(
    relay_value: torch.Tensor,
    *,
    z_threshold: float,
    eps: float = RELAY_RMS_EPS,
) -> torch.Tensor:
    """Return deterministic Bx1xHxW upper-tail support in ``[0, 1)``.

    Channel energy is standardized within each image.  FP16/BF16 reductions
    are evaluated in FP32.  Scale-normalized RMS computations keep maximum
    finite floating-point responses finite instead of overflowing on a direct
    square.
    """

    if relay_value.ndim != 4:
        raise ValueError(
            "tail support requires BCHW relay_value, "
            f"got shape={tuple(relay_value.shape)}"
        )
    if not relay_value.is_floating_point():
        raise TypeError("tail support requires floating-point relay values")
    threshold = _positive_finite_float(
        z_threshold,
        name="z_threshold",
    )
    normalized_eps = _positive_finite_float(eps, name="eps")

    working = _working_float(relay_value)
    if not bool(torch.isfinite(working).all()):
        raise FloatingPointError(
            "tail support requires finite relay values"
        )
    energy = _stable_regularized_rms(
        working,
        dimensions=1,
        eps=normalized_eps,
    )
    centered = energy - _stable_nonnegative_spatial_mean(energy)
    spatial_rms = _stable_regularized_rms(
        centered,
        dimensions=(-2, -1),
        eps=normalized_eps,
    )
    standardized = centered / spatial_rms
    support = torch.tanh(F.relu(standardized - threshold))
    support = support.to(dtype=relay_value.dtype)

    # tanh may round to exactly 1 for a large finite response.  Preserve a
    # strict upper bound in the actual output dtype.
    one = support.new_tensor(1.0)
    zero = support.new_tensor(0.0)
    strict_upper = torch.nextafter(one, zero)
    return torch.clamp(support, min=zero, max=strict_upper)


def _normalize_fixed_tail_z_thresholds(
    thresholds: Mapping[int, float],
) -> Tuple[Tuple[int, float], ...]:
    if not isinstance(thresholds, Mapping):
        raise TypeError("tail_z_thresholds must be a mapping")
    normalized: Dict[int, float] = {}
    for stage, value in thresholds.items():
        if type(stage) is not int:
            raise TypeError("tail_z_threshold stage keys must be integers")
        if stage in normalized:
            raise ValueError(f"duplicate tail threshold stage {stage}")
        normalized[stage] = _positive_finite_float(
            value,
            name=f"tail_z_thresholds[{stage}]",
        )
    if set(normalized) != set(RELAY_STAGE_ORDER):
        raise ValueError(
            "tail_z_thresholds must define stages 4, 3, and 2 exactly"
        )
    ordered = tuple((stage, normalized[stage]) for stage in RELAY_STAGE_ORDER)
    if ordered != _TAIL_Z_THRESHOLD_ITEMS:
        raise ValueError(
            "V4 tail_z_thresholds are frozen to "
            f"{dict(_TAIL_Z_THRESHOLD_ITEMS)}, got {dict(ordered)}"
        )
    return ordered


def _normalize_dc_support_mode(
    mode: Union[str, TailDCSupportMode],
) -> TailDCSupportMode:
    if isinstance(mode, TailDCSupportMode):
        return mode
    if type(mode) is not str:
        raise TypeError(
            "dc_support_mode must be one of "
            f"{SUPPORTED_DC_SUPPORT_MODES}"
        )
    try:
        return TailDCSupportMode(mode)
    except ValueError as exc:
        raise ValueError(
            "dc_support_mode must be one of "
            f"{SUPPORTED_DC_SUPPORT_MODES}, got {mode!r}"
        ) from exc


def _align_support(
    support: torch.Tensor,
    output_size: SpatialSize,
) -> torch.Tensor:
    if tuple(support.shape[-2:]) == tuple(output_size):
        return support
    return F.interpolate(
        support,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )


class TailAwarePersistentDCOffsetEvidenceRelay(
    RMSBalancedCenteredDCOffsetEvidenceRelay
):
    """V3 relay with strictly selected stage-aware DC support."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        width: int = DEFAULT_RELAY_WIDTH,
        eps: float = RELAY_RMS_EPS,
        dc_support_mode: Union[
            str,
            TailDCSupportMode,
        ] = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[
            int,
            float,
        ] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            base_channels=base_channels,
            width=width,
            eps=eps,
        )
        self._dc_support_mode = _normalize_dc_support_mode(dc_support_mode)
        self._tail_z_threshold_items = _normalize_fixed_tail_z_thresholds(
            tail_z_thresholds
        )

    @property
    def dc_support_mode(self) -> str:
        return self._dc_support_mode.value

    @property
    def tail_z_thresholds(self) -> Mapping[int, float]:
        # A fresh immutable view prevents callers from mutating architecture
        # behavior after checkpoint/manifest creation.
        return MappingProxyType(dict(self._tail_z_threshold_items))

    def _tail_support(
        self,
        tensor: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        thresholds = dict(self._tail_z_threshold_items)
        return relay_spatial_tail_support(
            tensor,
            z_threshold=thresholds[stage],
            eps=self.eps,
        )

    def _persistent_tail_support(
        self,
        *,
        stage: int,
        relay_value: torch.Tensor,
        parent_relay: torch.Tensor,
        parent_stage: int,
        output_size: SpatialSize,
    ) -> torch.Tensor:
        # Computing from detached inputs avoids constructing a graph that would
        # immediately be discarded.  It also makes the routing policy's
        # stop-gradient contract explicit.
        with torch.no_grad():
            local_support = self._tail_support(relay_value.detach(), stage)
            parent_support = self._tail_support(
                parent_relay.detach(),
                parent_stage,
            )
            parent_support = _align_support(parent_support, output_size)
            persistent = torch.sqrt(
                torch.clamp(local_support * parent_support, min=0.0)
            )
        return persistent.detach()

    def dc_support(
        self,
        stage: int,
        relay_value: torch.Tensor,
        sources: Sequence[torch.Tensor],
        output_size: SpatialSize,
    ) -> torch.Tensor:
        """Return selected bounded NER DC-offset support without model state."""

        expected_sources = {4: 3, 3: 4, 2: 3}
        if stage not in expected_sources:
            raise ValueError(f"relay stage must be 4, 3, or 2, got {stage}")
        if len(sources) != expected_sources[stage]:
            raise ValueError(
                f"stage {stage} requires {expected_sources[stage]} sources, "
                f"got {len(sources)}"
            )

        batch = relay_value.shape[0]
        if (
            stage == 4
            or self._dc_support_mode is TailDCSupportMode.LEGACY_GLOBAL
        ):
            return relay_value.new_ones(
                (batch, 1, output_size[0], output_size[1])
            )

        if stage == 3:
            parent_relay = sources[2]  # q4
            parent_stage = 4
        else:
            parent_relay = sources[1]  # q3
            parent_stage = 3

        persistent = self._persistent_tail_support(
            stage=stage,
            relay_value=relay_value,
            parent_relay=parent_relay,
            parent_stage=parent_stage,
            output_size=output_size,
        )
        if self._dc_support_mode is TailDCSupportMode.DIRECT_TAIL:
            return persistent
        if self._dc_support_mode is TailDCSupportMode.COMPLEMENT_TAIL:
            one = persistent.new_tensor(1.0)
            return torch.clamp(one - persistent, min=0.0, max=1.0)
        raise RuntimeError(
            f"unhandled DC support mode {self._dc_support_mode!r}"
        )

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
        support = self.dc_support(
            stage,
            relay_value,
            sources,
            output_size,
        )
        shifted_logits = centered_logits + (
            self.dc_offsets[str(stage)].view(1, 1, 1, 1) * support
        )
        mask = arctangent_residual_mask(shifted_logits)
        return relay_value, mask


def _initialize_v4_relay(module: nn.Module) -> None:
    # V4 adds no random module, so common relay tensors remain exactly paired
    # with V3 under the same isolated seed.
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TPDNERV8MPRSDCHV4SCTransNet(TPDNERV8MPRSDCHV3SCTransNet):
    """V3 complete model with target-protective tail-aware NER DC support."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: Union[
            str,
            TailDCSupportMode,
        ] = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[
            int,
            float,
        ] = DEFAULT_TAIL_Z_THRESHOLDS,
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

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(relay_initialization_seed)
            relay = TailAwarePersistentDCOffsetEvidenceRelay(
                base_channels=base_channels,
                width=relay_width,
                eps=RELAY_RMS_EPS,
                dc_support_mode=dc_support_mode,
                tail_z_thresholds=tail_z_thresholds,
            )
            relay.apply(_initialize_v4_relay)
        reference = next(self.parameters())
        relay.to(device=reference.device, dtype=reference.dtype)
        relay.zero_init_gates()
        self.tpd_ner = relay

    def architecture_manifest(self) -> Dict[str, object]:
        manifest = dict(super().architecture_manifest())
        mode = self.tpd_ner.dc_support_mode
        if mode == TailDCSupportMode.LEGACY_GLOBAL.value:
            stage3_support = "global_v3_exact"
            stage2_support = "global_v3_exact"
            selected_support_formula = "1"
        elif mode == TailDCSupportMode.DIRECT_TAIL.value:
            stage3_support = "stopgrad_geomean_tail_q3_q4"
            stage2_support = "stopgrad_geomean_tail_q2_q3"
            selected_support_formula = "P"
        elif mode == TailDCSupportMode.COMPLEMENT_TAIL.value:
            stage3_support = "stopgrad_one_minus_geomean_tail_q3_q4"
            stage2_support = "stopgrad_one_minus_geomean_tail_q2_q3"
            selected_support_formula = "1-P"
        else:
            raise RuntimeError(f"unhandled DC support mode {mode!r}")

        manifest.update(
            {
                "relay_version": V4_RELAY_VERSION,
                "ner_dc_offset_support_mode": mode,
                "ner_dc_offset_support_stage4": "global_v3_exact",
                "ner_dc_offset_support_stage3": stage3_support,
                "ner_dc_offset_support_stage2": stage2_support,
                "ner_dc_offset_support_formula_stage4": "1",
                "ner_dc_offset_support_formula_stage3_2": (
                    selected_support_formula
                ),
                "ner_dc_offset_support_scope": (
                    "post_centering_ner_gate_offset_not_tokenizer_mprs_dch"
                ),
                "gate_dc_support_mode": mode,
                "gate_dc_support_mode_options": SUPPORTED_DC_SUPPORT_MODES,
                "gate_dc_support_formal_default": DEFAULT_DC_SUPPORT_MODE,
                "gate_dc_support_stage4": "global_v3_exact",
                "gate_dc_support_stage3": stage3_support,
                "gate_dc_support_stage2": stage2_support,
                "target_protective_complement": (
                    mode == TailDCSupportMode.COMPLEMENT_TAIL.value
                ),
                "tail_statistic": "per_sample_spatial_z_of_channel_rms",
                "tail_mapping": "tanh(relu(z-kappa))",
                "tail_z_thresholds": dict(
                    self.tpd_ner.tail_z_thresholds
                ),
                "tail_z_thresholds_frozen": True,
                "tail_support_parameters": 0,
                "tail_support_buffers": 0,
                "tail_support_gradient": "stopped_or_constant",
                "mask_mapping": (
                    "atan(pi*(centered+dc*selected_support))/pi"
                ),
                "mask_bounds": (-V2_MASK_LIMIT, V2_MASK_LIMIT),
                "skip_factor_bounds": V2_SKIP_FACTOR_BOUNDS,
                "state_compatible_with": "tpd_ner_v8_mprs_dch_v3",
                "zero_gate_reference": "v3_v2_and_relay_off_exact",
            }
        )
        return manifest


def adapt_v8_mprs_dch_parent_v4(
    parent: SCTransNet,
    *,
    variant: str,
    relay_enabled: bool,
    relay_width: int = DEFAULT_RELAY_WIDTH,
    relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
    dc_support_mode: Union[
        str,
        TailDCSupportMode,
    ] = DEFAULT_DC_SUPPORT_MODE,
    tail_z_thresholds: Mapping[
        int,
        float,
    ] = DEFAULT_TAIL_Z_THRESHOLDS,
) -> TPDNERV8MPRSDCHSCTransNet:
    """Return unchanged relay-off or the selected V4 relay-on model."""

    if relay_width != DEFAULT_RELAY_WIDTH:
        raise ValueError(
            f"V4 relay width is fixed to {DEFAULT_RELAY_WIDTH}, "
            f"got {relay_width}"
        )
    normalized_mode = _normalize_dc_support_mode(dc_support_mode)
    normalized_thresholds = _normalize_fixed_tail_z_thresholds(
        tail_z_thresholds
    )
    if not relay_enabled:
        return adapt_v8_mprs_dch_parent_v2(
            parent,
            variant=variant,
            relay_enabled=False,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
        )
    return TPDNERV8MPRSDCHV4SCTransNet(
        parent,
        variant=variant,
        relay_width=relay_width,
        relay_initialization_seed=relay_initialization_seed,
        dc_support_mode=normalized_mode,
        tail_z_thresholds=dict(normalized_thresholds),
    )


def v4_relay_parameter_count(model: nn.Module) -> int:
    relay = getattr(model, "tpd_ner", None)
    if relay is None:
        return 0
    return sum(parameter.numel() for parameter in relay.parameters())


__all__ = [
    "DEFAULT_DC_SUPPORT_MODE",
    "DEFAULT_RELAY_INITIALIZATION_SEED",
    "DEFAULT_RELAY_WIDTH",
    "DEFAULT_TAIL_Z_THRESHOLDS",
    "EVIDENCE_NODE_NAMES",
    "PRODUCTION_PARENT_PARAMETERS",
    "PRODUCTION_V4_RELAY_ON_PARAMETERS",
    "PRODUCTION_V4_RELAY_PARAMETERS",
    "RELAY_RMS_EPS",
    "RELAY_STAGE_ORDER",
    "SUPPORTED_DC_SUPPORT_MODES",
    "TailAwarePersistentDCOffsetEvidenceRelay",
    "TailDCSupportMode",
    "TPDNERV8MPRSDCHV4SCTransNet",
    "V2_MASK_LIMIT",
    "V2_SKIP_FACTOR_BOUNDS",
    "V4_RELAY_VERSION",
    "adapt_v8_mprs_dch_parent_v4",
    "relay_spatial_tail_support",
    "v4_relay_parameter_count",
]
