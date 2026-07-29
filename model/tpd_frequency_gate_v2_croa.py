"""CROA Query-only frequency gate with optimizer-anchored identity.

This module is the V2 successor of :mod:`model.tpd_frequency_gate`.  It keeps
the V1 forward-local ``prepare``/``apply_prepared`` boundary while changing
only the frequency-to-Query modulation:

* each aligned Haar prior is normalized by a per-sample full-tensor RMS;
* raw spatial logits are centered and RMS-normalized;
* an arctangent map is centered once more and contracted to ``(-0.5, 0.5)``;
* a detached frequency source prevents a side-branch gradient into encoder
  features; and
* a zero terminal projection plus effective alpha ``0.1`` gives an exact
  identity forward, identical shared gradients, and an anchored first update.

The prepared objects are explicitly scoped to one gate instance and one model
forward.  They must never be cached across optimizer steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tpd_frequency_gate import (
    FixedHaarAnalysis,
    SUPPORTED_FREQUENCY_MODES,
)


FrequencyMode = str
SpatialSize = Tuple[int, int]

QFG_V2_CROA_VERSION = "qfg_v2_croa_centered_rms_optimizer_anchor"
RMS_EPS = 1e-6
FORMAL_FEATURE_CHANNELS = (32, 64, 128, 256)
FORMAL_FREQUENCY_MODE = "high_low"
FORMAL_HIDDEN_CHANNELS = 8
FORMAL_ALIGNMENT_RATIOS = (8, 4, 2, 1)
FORMAL_ALPHA_EFFECTIVE_INIT = 0.1
FORMAL_GATE_LIMIT = 0.5
PRODUCTION_QFG_V2_CROA_PARAMETERS = 15_684
PRODUCTION_QFG_V2_CROA_PARAMETER_KEY_COUNT = 16
PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT = 20


def _formal_parameter_keys() -> Tuple[str, ...]:
    return tuple(
        key
        for index in range(4)
        for key in (
            f"levels.{index}.alpha",
            f"levels.{index}.prior_projection.weight",
            f"levels.{index}.spatial_projection.0.weight",
            f"levels.{index}.gate_out.weight",
        )
    )


def _formal_state_keys() -> Tuple[str, ...]:
    return tuple(
        key
        for index in range(4)
        for key in (
            f"levels.{index}.alpha",
            f"levels.{index}.haar.kernels",
            f"levels.{index}.prior_projection.weight",
            f"levels.{index}.spatial_projection.0.weight",
            f"levels.{index}.gate_out.weight",
        )
    )


FORMAL_QFG_V2_CROA_PARAMETER_KEYS = _formal_parameter_keys()
FORMAL_QFG_V2_CROA_STATE_KEYS = _formal_state_keys()


def _assert_runtime_condition(
    condition: torch.Tensor,
    *,
    message: str,
) -> None:
    """Assert one tensor condition without synchronizing a CUDA host thread."""

    if (
        not isinstance(condition, torch.Tensor)
        or condition.numel() != 1
        or condition.dtype != torch.bool
    ):
        raise TypeError("runtime assertion condition must be one bool Tensor")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise FloatingPointError(message)


def _positive_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be a positive finite real number"
        ) from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(
            f"{name} must be positive and finite, got {normalized!r}"
        )
    return normalized


def _effective_alpha(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("alpha_effective_init must be a finite real number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "alpha_effective_init must be a finite real number"
        ) from exc
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ValueError("alpha_effective_init must lie strictly inside (0, 1)")
    return normalized


def _validate_float_map(
    value: torch.Tensor,
    *,
    name: str,
    channels: int | None = None,
    validate_finite: bool,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(
            f"{name} must have shape BxCxHxW, got {tuple(value.shape)}"
        )
    if min(value.shape) < 1:
        raise ValueError(f"{name} dimensions must be positive")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(
            f"{name} requires {channels} channels, got {value.shape[1]}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if validate_finite:
        _assert_runtime_condition(
            torch.isfinite(value).all(),
            message=f"{name} contains non-finite values",
        )


def _working_float(value: torch.Tensor) -> torch.Tensor:
    """Promote only reduced-precision inputs; preserve FP32 and FP64."""

    if value.dtype in (torch.float16, torch.bfloat16):
        return value.float()
    return value


def _working_dtype(first: torch.dtype, second: torch.dtype) -> torch.dtype:
    promoted = torch.promote_types(first, second)
    if promoted in (torch.float16, torch.bfloat16):
        return torch.float32
    return promoted


def _stable_regularized_rms(
    value: torch.Tensor,
    *,
    dimensions: Tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    """Compute ``sqrt(mean(value**2)+eps)`` without squaring large values."""

    sqrt_eps = value.new_tensor(eps).sqrt()
    # The scale is a numerical device, not part of the requested formula.
    # Detaching it preserves the algebraic result and avoids the undefined
    # derivative of abs/amax at the all-zero identity anchor.
    scale = value.detach().abs().amax(dim=dimensions, keepdim=True)
    safe_scale = torch.maximum(scale, sqrt_eps)
    scaled = value / safe_scale
    scaled_mean_square = scaled.square().mean(
        dim=dimensions,
        keepdim=True,
    )
    scaled_eps = sqrt_eps / safe_scale
    scaled_regularized_rms = torch.sqrt(
        torch.clamp(scaled_mean_square, min=0.0)
        + scaled_eps.square()
    )
    return safe_scale * scaled_regularized_rms


def sample_full_tensor_rms_normalize(
    value: torch.Tensor,
    *,
    eps: float = RMS_EPS,
    validate_finite: bool = True,
) -> torch.Tensor:
    """Normalize each B sample across all C/H/W values with stable RMS."""

    _validate_float_map(
        value,
        name="frequency_prior",
        validate_finite=validate_finite,
    )
    normalized_eps = _positive_finite_float(eps, name="eps")
    working = _working_float(value)
    rms = _stable_regularized_rms(
        working,
        dimensions=(1, 2, 3),
        eps=normalized_eps,
    )
    normalized = working / rms
    if validate_finite:
        _assert_runtime_condition(
            torch.isfinite(normalized).all(),
            message=(
                "full-tensor RMS normalization produced non-finite values"
            ),
        )
    return normalized.to(dtype=value.dtype)


def spatial_center_rms_normalize(
    value: torch.Tensor,
    *,
    eps: float = RMS_EPS,
    validate_finite: bool = True,
) -> torch.Tensor:
    """Return spatially zero-centered, RMS-normalized logits.

    Centering is performed in a scale-normalized domain.  This avoids an
    overflowing ``value - mean(value)`` intermediate for extreme but finite
    values while remaining algebraically equivalent to the requested formula.
    """

    _validate_float_map(
        value,
        name="raw_gate_logits",
        channels=1,
        validate_finite=validate_finite,
    )
    normalized_eps = _positive_finite_float(eps, name="eps")
    working = _working_float(value)
    sqrt_eps = working.new_tensor(normalized_eps).sqrt()
    # See _stable_regularized_rms: the detached lower-bounded scale makes the
    # exact raw=0 backward finite while keeping the formula scale-stable.
    scale = working.detach().abs().amax(dim=(-2, -1), keepdim=True)
    safe_scale = torch.maximum(scale, sqrt_eps)
    scaled = working / safe_scale
    centered_scaled = scaled - scaled.mean(
        dim=(-2, -1),
        keepdim=True,
    )
    scaled_eps = sqrt_eps / safe_scale
    denominator = torch.sqrt(
        torch.clamp(
            centered_scaled.square().mean(
                dim=(-2, -1),
                keepdim=True,
            ),
            min=0.0,
        )
        + scaled_eps.square()
    )
    normalized = centered_scaled / denominator
    if validate_finite:
        _assert_runtime_condition(
            torch.isfinite(normalized).all(),
            message=(
                "spatial centered-RMS normalization produced non-finite values"
            ),
        )
    return normalized


def centered_bounded_arctangent_gate(
    normalized_logits: torch.Tensor,
    *,
    validate_finite: bool = True,
) -> torch.Tensor:
    """Map normalized logits to a zero-mean gate strictly inside ``±0.5``."""

    _validate_float_map(
        normalized_logits,
        name="normalized_gate_logits",
        channels=1,
        validate_finite=validate_finite,
    )
    pi = normalized_logits.new_tensor(math.pi)
    half_bounded = torch.atan(pi * normalized_logits) / pi
    gate = 0.5 * (
        half_bounded
        - half_bounded.mean(dim=(-2, -1), keepdim=True)
    )

    # The mathematical range is already strict.  A per-sample contraction
    # protects that invariant from endpoint rounding while preserving zero mean.
    half = gate.new_tensor(FORMAL_GATE_LIMIT)
    zero = gate.new_tensor(0.0)
    strict_limit = torch.nextafter(half, zero)
    maximum = gate.abs().amax(dim=(-2, -1), keepdim=True)
    safe_maximum = torch.where(
        maximum > 0.0,
        maximum,
        torch.ones_like(maximum),
    )
    contraction = torch.where(
        maximum >= half,
        strict_limit / safe_maximum,
        torch.ones_like(maximum),
    )
    gate = gate * contraction
    if validate_finite:
        _assert_runtime_condition(
            torch.isfinite(gate).all(),
            message="bounded arctangent gate produced non-finite values",
        )
        _assert_runtime_condition(
            torch.all(gate.abs() < half),
            message="bounded arctangent gate reached its limit",
        )
    return gate


def _mode_channels(channels: int, mode: FrequencyMode) -> int:
    if mode == "high":
        return 3 * channels
    if mode == "low":
        return channels
    if mode == "high_low":
        return 4 * channels
    raise ValueError(
        f"unknown frequency mode {mode!r}; "
        f"choices={SUPPORTED_FREQUENCY_MODES}"
    )


def _select_bands(
    bands: torch.Tensor,
    mode: FrequencyMode,
) -> torch.Tensor:
    if bands.ndim != 5 or bands.shape[2] != 4:
        raise ValueError(
            "Haar bands must have shape BxCx4xHxW, "
            f"got {tuple(bands.shape)}"
        )
    if mode == "high":
        selected = bands[:, :, 1:].abs()
    elif mode == "low":
        selected = bands[:, :, :1]
    elif mode == "high_low":
        selected = torch.cat(
            (bands[:, :, :1], bands[:, :, 1:].abs()),
            dim=2,
        )
    else:
        raise ValueError(
            f"unknown frequency mode {mode!r}; "
            f"choices={SUPPORTED_FREQUENCY_MODES}"
        )
    return selected.flatten(1, 2)


def _normalize_query_size(
    value: Sequence[int],
    *,
    name: str,
) -> SpatialSize:
    try:
        size = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a pair of positive integers") from exc
    if (
        len(size) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
            for dimension in size
        )
    ):
        raise ValueError(f"{name} must be a pair of positive integers")
    return int(size[0]), int(size[1])


def _normalize_alignment(
    value: int | Sequence[int],
    *,
    name: str = "expected_alignment",
) -> SpatialSize:
    if isinstance(value, int) and not isinstance(value, bool):
        raw = (value, value)
    else:
        try:
            raw = tuple(value)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be a positive integer or pair"
            ) from exc
    if (
        len(raw) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
            for dimension in raw
        )
    ):
        raise ValueError(f"{name} must be a positive integer or pair")
    return int(raw[0]), int(raw[1])


@dataclass(frozen=True, slots=True)
class PreparedQueryFrequencyLevelV2CROA:
    """One complete, query-independent modulation scoped to one gate level.

    The final three fields are optional only for constructor compatibility with
    the original raw-logit-only prepared object.  Objects returned by
    :meth:`QueryFrequencyLevelGateV2CROA.prepare` always populate all fields so
    every SCTB can reuse the same normalization, bounded gate, and factor.
    """

    raw_gate_logits: torch.Tensor
    query_size: SpatialSize
    batch_size: int
    _owner_token: object
    normalized_logits: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    factor: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class PreparedQueryFrequencyGateV2CROA:
    """Four prepared priors reusable by all SCTBs in one model forward."""

    levels: Tuple[
        PreparedQueryFrequencyLevelV2CROA,
        PreparedQueryFrequencyLevelV2CROA,
        PreparedQueryFrequencyLevelV2CROA,
        PreparedQueryFrequencyLevelV2CROA,
    ]
    _owner_token: object

    @property
    def raw_gate_logits(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(level.raw_gate_logits for level in self.levels)


@dataclass(frozen=True, slots=True)
class QueryFrequencyGateOutputV2CROA:
    """Explicit V2 outputs; raw and transformed gates cannot be confused."""

    queries: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    raw_gate_logits: Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    normalized_logits: Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    gates: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    factors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def gate_logits(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility alias whose meaning remains the raw projection."""

        return self.raw_gate_logits


class QueryFrequencyLevelGateV2CROA(nn.Module):
    """One centered/RMS Query frequency gate with a zero terminal map."""

    def __init__(
        self,
        feature_channels: int,
        *,
        mode: FrequencyMode = FORMAL_FREQUENCY_MODE,
        hidden_channels: int = FORMAL_HIDDEN_CHANNELS,
        expected_alignment: int | Sequence[int] = 1,
        detach_frequency_source: bool = True,
        alpha_effective_init: float = FORMAL_ALPHA_EFFECTIVE_INIT,
        eps: float = RMS_EPS,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("feature_channels", feature_channels),
            ("hidden_channels", hidden_channels),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        _mode_channels(feature_channels, mode)
        self.feature_channels = int(feature_channels)
        self.mode = str(mode)
        self.hidden_channels = int(hidden_channels)
        self.expected_alignment = _normalize_alignment(expected_alignment)
        self.detach_frequency_source = bool(detach_frequency_source)
        self.alpha_effective_init = _effective_alpha(alpha_effective_init)
        self.eps = _positive_finite_float(eps, name="eps")
        self.validate_finite = bool(validate_finite)
        self._prepared_owner_token = object()

        # ``prepare`` validates the feature immediately before Haar analysis.
        # Disabling the nested check removes an otherwise duplicate CUDA
        # reduction/host synchronization without weakening the public level
        # boundary.
        self.haar = FixedHaarAnalysis(validate_finite=False)
        self.prior_projection = nn.Conv2d(
            _mode_channels(self.feature_channels, self.mode),
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                groups=self.hidden_channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.gate_out = nn.Conv2d(
            self.hidden_channels,
            1,
            kernel_size=1,
            bias=False,
        )
        self.alpha = nn.Parameter(
            torch.tensor(math.atanh(self.alpha_effective_init))
        )
        nn.init.zeros_(self.gate_out.weight)

    def reset_identity(self) -> None:
        with torch.no_grad():
            self.gate_out.weight.zero_()
            self.alpha.fill_(math.atanh(self.alpha_effective_init))

    @staticmethod
    def _align_prior(
        prior: torch.Tensor,
        query_size: SpatialSize,
        expected_alignment: SpatialSize,
    ) -> torch.Tensor:
        prior_h, prior_w = prior.shape[-2:]
        query_h, query_w = query_size
        if prior_h % query_h or prior_w % query_w:
            raise ValueError(
                "Haar prior grid must be an integer multiple of Query grid, "
                f"got prior={(prior_h, prior_w)} query={query_size}"
            )
        ratio = prior_h // query_h, prior_w // query_w
        if ratio[0] < 1 or ratio[1] < 1:
            raise ValueError("Haar prior grid cannot be smaller than Query grid")
        if ratio != expected_alignment:
            raise ValueError(
                "Haar prior to Query alignment differs from the registered "
                f"level ratio: expected={expected_alignment} observed={ratio}"
            )
        if ratio == (1, 1):
            return prior
        return F.avg_pool2d(prior, kernel_size=ratio, stride=ratio)

    def prepare(
        self,
        feature: torch.Tensor,
        query_size: Sequence[int],
    ) -> PreparedQueryFrequencyLevelV2CROA:
        _validate_float_map(
            feature,
            name="frequency_feature",
            channels=self.feature_channels,
            validate_finite=self.validate_finite,
        )
        normalized_size = _normalize_query_size(
            query_size,
            name="query_size",
        )
        source = (
            feature.detach()
            if self.detach_frequency_source
            else feature
        )
        bands = self.haar(source)
        selected = _select_bands(bands, self.mode)
        selected = self._align_prior(
            selected,
            normalized_size,
            self.expected_alignment,
        )
        selected = sample_full_tensor_rms_normalize(
            selected,
            eps=self.eps,
            # The source was validated immediately before the fixed finite
            # Haar/pooling path.  Validate the resulting prepared tensors once
            # below instead of checking every intermediate twice.
            validate_finite=False,
        )
        hidden = self.prior_projection(selected)
        hidden = self.spatial_projection(hidden)
        raw_gate_logits = self.gate_out(hidden)
        normalized_logits = spatial_center_rms_normalize(
            raw_gate_logits,
            eps=self.eps,
            validate_finite=False,
        )
        gate = centered_bounded_arctangent_gate(
            normalized_logits,
            validate_finite=False,
        )
        compute_dtype = _working_dtype(raw_gate_logits.dtype, gate.dtype)
        gate_working = gate.to(dtype=compute_dtype)
        effective_alpha = torch.tanh(
            self.alpha.to(dtype=compute_dtype)
        )
        factor = 1.0 + effective_alpha * gate_working
        if self.validate_finite:
            for name, value in (
                ("raw gate logits", raw_gate_logits),
                ("normalized logits", normalized_logits),
                ("bounded gate", gate),
                ("Query frequency factor", factor),
            ):
                _assert_runtime_condition(
                    torch.isfinite(value).all(),
                    message=f"prepared {name} contain non-finite values",
                )
            half = gate.new_tensor(FORMAL_GATE_LIMIT)
            _assert_runtime_condition(
                torch.all(gate.abs() < half),
                message="bounded arctangent gate reached its limit",
            )
            factor_half = factor.new_tensor(FORMAL_GATE_LIMIT)
            factor_upper = factor.new_tensor(1.0 + FORMAL_GATE_LIMIT)
            _assert_runtime_condition(
                torch.logical_and(
                    torch.all(factor > factor_half),
                    torch.all(factor < factor_upper),
                ),
                message=(
                    "Query frequency factor left the strict (0.5, 1.5) range"
                ),
            )
        return PreparedQueryFrequencyLevelV2CROA(
            raw_gate_logits=raw_gate_logits,
            query_size=normalized_size,
            batch_size=int(feature.shape[0]),
            _owner_token=self._prepared_owner_token,
            normalized_logits=normalized_logits,
            gate=gate,
            factor=factor,
        )

    def apply_prepared(
        self,
        query: torch.Tensor,
        prepared: PreparedQueryFrequencyLevelV2CROA,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _validate_float_map(
            query,
            name="query",
            validate_finite=self.validate_finite,
        )
        if not isinstance(prepared, PreparedQueryFrequencyLevelV2CROA):
            raise TypeError(
                "prepared must be a PreparedQueryFrequencyLevelV2CROA"
            )
        if prepared._owner_token is not self._prepared_owner_token:
            raise ValueError(
                "prepared frequency prior belongs to a different gate level"
            )
        raw_gate_logits = prepared.raw_gate_logits
        normalized_logits = prepared.normalized_logits
        gate = prepared.gate
        factor = prepared.factor
        _validate_float_map(
            raw_gate_logits,
            name="prepared_raw_gate_logits",
            channels=1,
            # A standard prepared object was validated once in ``prepare``.
            validate_finite=False,
        )
        if tuple(query.shape[-2:]) != prepared.query_size:
            raise ValueError(
                "Query grid differs from the prepared grid: "
                f"prepared={prepared.query_size} "
                f"query={tuple(query.shape[-2:])}"
            )
        if (
            query.shape[0] != prepared.batch_size
            or raw_gate_logits.shape[0] != prepared.batch_size
        ):
            raise ValueError(
                "Query and prepared frequency prior batch sizes must match"
            )
        if tuple(raw_gate_logits.shape[-2:]) != prepared.query_size:
            raise ValueError(
                "prepared frequency prior has inconsistent grid metadata"
            )
        if query.device != raw_gate_logits.device:
            raise ValueError(
                "Query and prepared frequency prior devices must match"
            )

        # Compatibility fallback for callers that directly construct the
        # original four-field prepared dataclass.  Normal ``prepare`` output
        # always takes the cached branch.
        if normalized_logits is None or gate is None or factor is None:
            normalized_logits = spatial_center_rms_normalize(
                raw_gate_logits,
                eps=self.eps,
                validate_finite=self.validate_finite,
            )
            gate = centered_bounded_arctangent_gate(
                normalized_logits,
                validate_finite=self.validate_finite,
            )
            compute_dtype = _working_dtype(query.dtype, gate.dtype)
            gate_working = gate.to(dtype=compute_dtype)
            effective_alpha = torch.tanh(
                self.alpha.to(dtype=compute_dtype)
            )
            factor = 1.0 + effective_alpha * gate_working
        else:
            for name, value in (
                ("prepared_normalized_logits", normalized_logits),
                ("prepared_gate", gate),
                ("prepared_factor", factor),
            ):
                _validate_float_map(
                    value,
                    name=name,
                    channels=1,
                    validate_finite=False,
                )
                if tuple(value.shape[-2:]) != prepared.query_size:
                    raise ValueError(
                        f"{name} has inconsistent grid metadata"
                    )
                if value.shape[0] != prepared.batch_size:
                    raise ValueError(
                        f"{name} has inconsistent batch metadata"
                    )
                if value.device != query.device:
                    raise ValueError(
                        f"Query and {name} devices must match"
                    )

        compute_dtype = _working_dtype(query.dtype, gate.dtype)
        # Formal model Queries and encoder features share a dtype, so the
        # prepared factor is reused directly.  Retain the historical mixed-dtype
        # behavior for external callers by rebuilding only in that uncommon
        # compatibility case.
        if factor.dtype != compute_dtype:
            gate_working = gate.to(dtype=compute_dtype)
            effective_alpha = torch.tanh(
                self.alpha.to(dtype=compute_dtype)
            )
            factor = 1.0 + effective_alpha * gate_working
        modulated = (
            query.to(dtype=compute_dtype) * factor
        ).to(dtype=query.dtype)
        if self.validate_finite:
            _assert_runtime_condition(
                torch.isfinite(modulated).all(),
                message="modulated Query contains non-finite values",
            )
        return (
            modulated,
            raw_gate_logits,
            normalized_logits,
            gate,
            factor,
        )

    def forward(
        self,
        query: torch.Tensor,
        feature: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _validate_float_map(
            query,
            name="query",
            validate_finite=self.validate_finite,
        )
        if query.shape[0] != feature.shape[0]:
            raise ValueError(
                "Query and frequency feature batch sizes must match"
            )
        if query.device != feature.device:
            raise ValueError(
                "Query and frequency feature devices must match"
            )
        prepared = self.prepare(feature, tuple(query.shape[-2:]))
        return self.apply_prepared(query, prepared)


class QueryOnlyFrequencyGateV2CROA(nn.Module):
    """Four V2-CROA gates for already generated ``q1...q4`` tensors."""

    def __init__(
        self,
        feature_channels: Sequence[int] = FORMAL_FEATURE_CHANNELS,
        *,
        mode: FrequencyMode = FORMAL_FREQUENCY_MODE,
        hidden_channels: int = FORMAL_HIDDEN_CHANNELS,
        alignment_ratios: Sequence[
            int | Sequence[int]
        ] = FORMAL_ALIGNMENT_RATIOS,
        detach_frequency_source: bool = True,
        alpha_effective_init: float = FORMAL_ALPHA_EFFECTIVE_INIT,
        eps: float = RMS_EPS,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        try:
            channels = tuple(feature_channels)
        except TypeError as exc:
            raise TypeError(
                "feature_channels must be a finite sequence"
            ) from exc
        if len(channels) != 4:
            raise ValueError(
                f"QFG-V2-CROA requires four feature levels, got {len(channels)}"
            )
        for index, channels_at_level in enumerate(channels):
            if (
                not isinstance(channels_at_level, int)
                or isinstance(channels_at_level, bool)
                or channels_at_level < 1
            ):
                raise ValueError(
                    f"feature_channels[{index}] must be a positive integer"
                )
        try:
            raw_alignments = tuple(alignment_ratios)
        except TypeError as exc:
            raise TypeError(
                "alignment_ratios must be a finite sequence"
            ) from exc
        if len(raw_alignments) != 4:
            raise ValueError("QFG-V2-CROA requires four alignment ratios")
        alignments = tuple(
            _normalize_alignment(
                alignment,
                name=f"alignment_ratios[{index}]",
            )
            for index, alignment in enumerate(raw_alignments)
        )
        if (
            not isinstance(hidden_channels, int)
            or isinstance(hidden_channels, bool)
            or hidden_channels < 1
        ):
            raise ValueError("hidden_channels must be a positive integer")
        _mode_channels(channels[0], mode)

        self.feature_channels = channels
        self.mode = str(mode)
        self.hidden_channels = int(hidden_channels)
        self.alignment_ratios = alignments
        self.detach_frequency_source = bool(detach_frequency_source)
        self.alpha_effective_init = _effective_alpha(alpha_effective_init)
        self.eps = _positive_finite_float(eps, name="eps")
        self.validate_finite = bool(validate_finite)
        self._prepared_owner_token = object()
        self.levels = nn.ModuleList(
            QueryFrequencyLevelGateV2CROA(
                channels[index],
                mode=self.mode,
                hidden_channels=self.hidden_channels,
                expected_alignment=alignments[index],
                detach_frequency_source=self.detach_frequency_source,
                alpha_effective_init=self.alpha_effective_init,
                eps=self.eps,
                validate_finite=self.validate_finite,
            )
            for index in range(4)
        )

    def reset_identity(self) -> None:
        for level in self.levels:
            level.reset_identity()

    @staticmethod
    def _normalize_query_sizes(
        query_sizes: Sequence[Sequence[int]] | Sequence[int],
    ) -> Tuple[SpatialSize, SpatialSize, SpatialSize, SpatialSize]:
        try:
            raw_sizes = tuple(query_sizes)
        except TypeError as exc:
            raise ValueError(
                "query_sizes must be one H/W pair or four H/W pairs"
            ) from exc
        if (
            len(raw_sizes) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in raw_sizes
            )
        ):
            shared = _normalize_query_size(raw_sizes, name="query_sizes")
            return shared, shared, shared, shared
        if len(raw_sizes) != 4:
            raise ValueError(
                "query_sizes must be one H/W pair or four H/W pairs"
            )
        return tuple(
            _normalize_query_size(
                value,
                name=f"query_sizes[{index}]",
            )
            for index, value in enumerate(raw_sizes)
        )

    def prepare(
        self,
        encoder_features: Sequence[torch.Tensor],
        query_sizes: Sequence[Sequence[int]] | Sequence[int],
    ) -> PreparedQueryFrequencyGateV2CROA:
        """Prepare four complete modulation factors once per model forward."""

        encoder_features = tuple(encoder_features)
        if len(encoder_features) != 4:
            raise ValueError(
                "QFG-V2-CROA requires exactly four encoder features"
            )
        normalized_sizes = self._normalize_query_sizes(query_sizes)
        reference_batch = encoder_features[0].shape[0]
        reference_device = encoder_features[0].device
        for index, feature in enumerate(encoder_features):
            _validate_float_map(
                feature,
                name=f"encoder_features[{index}]",
                channels=self.feature_channels[index],
                # Each level validates its own feature immediately before the
                # fixed Haar path.  Keep wrapper shape/device checks without a
                # second full-tensor finite reduction.
                validate_finite=False,
            )
            if feature.shape[0] != reference_batch:
                raise ValueError(
                    "all encoder features must share one batch size"
                )
            if feature.device != reference_device:
                raise ValueError(
                    "all encoder features must share one device"
                )
        levels = tuple(
            level.prepare(feature, query_size)
            for level, feature, query_size in zip(
                self.levels,
                encoder_features,
                normalized_sizes,
            )
        )
        return PreparedQueryFrequencyGateV2CROA(
            levels=levels,
            _owner_token=self._prepared_owner_token,
        )

    def apply_prepared(
        self,
        queries: Sequence[torch.Tensor],
        prepared: PreparedQueryFrequencyGateV2CROA,
    ) -> QueryFrequencyGateOutputV2CROA:
        """Reuse one prepared modulation for one SCTB's four Query tensors."""

        queries = tuple(queries)
        if len(queries) != 4:
            raise ValueError(
                "QFG-V2-CROA requires exactly four Query maps"
            )
        if not isinstance(prepared, PreparedQueryFrequencyGateV2CROA):
            raise TypeError(
                "prepared must be a PreparedQueryFrequencyGateV2CROA"
            )
        if prepared._owner_token is not self._prepared_owner_token:
            raise ValueError(
                "prepared frequency gate belongs to a different gate instance"
            )
        if len(prepared.levels) != 4:
            raise ValueError("prepared frequency gate must contain four levels")
        outputs = tuple(
            level.apply_prepared(query, prepared_level)
            for level, query, prepared_level in zip(
                self.levels,
                queries,
                prepared.levels,
            )
        )
        (
            query_outputs,
            raw_gate_logits,
            normalized_logits,
            gates,
            factors,
        ) = zip(*outputs)
        return QueryFrequencyGateOutputV2CROA(
            queries=tuple(query_outputs),
            raw_gate_logits=tuple(raw_gate_logits),
            normalized_logits=tuple(normalized_logits),
            gates=tuple(gates),
            factors=tuple(factors),
        )

    def forward(
        self,
        queries: Sequence[torch.Tensor],
        encoder_features: Sequence[torch.Tensor],
    ) -> QueryFrequencyGateOutputV2CROA:
        """One-shot compatibility API: prepare once, then apply once."""

        queries = tuple(queries)
        encoder_features = tuple(encoder_features)
        if len(queries) != 4 or len(encoder_features) != 4:
            raise ValueError(
                "QFG-V2-CROA requires exactly four Query maps and four "
                "encoder features"
            )
        prepared = self.prepare(
            encoder_features,
            tuple(tuple(query.shape[-2:]) for query in queries),
        )
        return self.apply_prepared(queries, prepared)

    def architecture_manifest(self) -> Dict[str, object]:
        return {
            "module": "QueryOnlyFrequencyGateV2CROA",
            "version": QFG_V2_CROA_VERSION,
            "frequency_transform": "fixed_orthogonal_haar_2x2",
            "frequency_mode": self.mode,
            "high_frequency_representation": "absolute_magnitude",
            "feature_channels": self.feature_channels,
            "hidden_channels": self.hidden_channels,
            "query_levels": 4,
            "registered_alignment_ratios": tuple(
                value[0] if value[0] == value[1] else value
                for value in self.alignment_ratios
            ),
            "frequency_prior_normalization": (
                "per_sample_full_tensor_scale_stable_rms"
            ),
            "raw_logit_normalization": (
                "per_sample_spatial_center_scale_stable_rms"
            ),
            "gate_formula": (
                "0.5*(atan(pi*z)/pi-mean_hw(atan(pi*z)/pi))"
            ),
            "gate_bounds": (-FORMAL_GATE_LIMIT, FORMAL_GATE_LIMIT),
            "factor_formula": "1+tanh(alpha)*gate",
            "factor_bounds": (0.5, 1.5),
            "alpha_effective_initialization": self.alpha_effective_init,
            "alpha_parameter_initialization": math.atanh(
                self.alpha_effective_init
            ),
            "terminal_projection_initialization": "exact_zero",
            "terminal_projection_bias": False,
            "detach_frequency_source": self.detach_frequency_source,
            "rms_eps": self.eps,
            "reduced_precision_reduction_dtype": "float32",
            "float32_preserved": True,
            "float64_preserved": True,
            "modulation_location": "post_q_convolution_pre_normalization",
            "modified_attention_tensors": ("Q",),
            "kv_modified": False,
            "cfn_modified": False,
            "decoder_injection": False,
            "execution": (
                "prepare_complete_modulation_once_"
                "apply_query_many_per_model_forward"
            ),
            "prepared_level_payload": (
                "raw_gate_logits",
                "normalized_logits",
                "gate",
                "factor",
            ),
            "prepared_modulation_reused_across_sctb": True,
            "prepared_object_persistence": "forward_local_only",
            "prepared_owner_validation": "level_and_wrapper_identity_tokens",
            "output_contract": "QueryFrequencyGateOutputV2CROA",
            "parameter_count": frequency_gate_parameter_count(self),
            "state_key_count": len(self.state_dict()),
            "finite_validation": self.validate_finite,
        }


def frequency_gate_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def validate_formal_qfg_v2_croa(
    module: nn.Module,
    *,
    require_identity_initialization: bool = True,
) -> Dict[str, object]:
    """Validate the complete non-state and state contract of formal QFG V2."""

    if type(module) is not QueryOnlyFrequencyGateV2CROA:
        raise TypeError(
            "formal QFG-V2-CROA must use the exact production class"
        )
    expected = {
        "feature_channels": FORMAL_FEATURE_CHANNELS,
        "mode": FORMAL_FREQUENCY_MODE,
        "hidden_channels": FORMAL_HIDDEN_CHANNELS,
        "alignment_ratios": tuple(
            (ratio, ratio) for ratio in FORMAL_ALIGNMENT_RATIOS
        ),
        "detach_frequency_source": True,
        "alpha_effective_init": FORMAL_ALPHA_EFFECTIVE_INIT,
        "eps": RMS_EPS,
        "validate_finite": True,
    }
    for name, value in expected.items():
        if getattr(module, name) != value:
            raise ValueError(
                f"formal QFG-V2-CROA {name} differs: "
                f"{getattr(module, name)!r} != {value!r}"
            )
    parameter_keys = tuple(name for name, _ in module.named_parameters())
    state_keys = tuple(module.state_dict())
    if set(parameter_keys) != set(FORMAL_QFG_V2_CROA_PARAMETER_KEYS):
        raise ValueError("formal QFG-V2-CROA parameter keys differ")
    if set(state_keys) != set(FORMAL_QFG_V2_CROA_STATE_KEYS):
        raise ValueError("formal QFG-V2-CROA state keys differ")
    if len(parameter_keys) != PRODUCTION_QFG_V2_CROA_PARAMETER_KEY_COUNT:
        raise ValueError("formal QFG-V2-CROA parameter-key count differs")
    if len(state_keys) != PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT:
        raise ValueError("formal QFG-V2-CROA state-key count differs")
    parameters = frequency_gate_parameter_count(module)
    if parameters != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise ValueError("formal QFG-V2-CROA parameter count differs")
    for index, level in enumerate(module.levels):
        if level.gate_out.bias is not None:
            raise ValueError(
                f"formal QFG-V2-CROA level {index} terminal bias exists"
            )
        if require_identity_initialization:
            if int(torch.count_nonzero(level.gate_out.weight)) != 0:
                raise ValueError(
                    f"formal QFG-V2-CROA level {index} terminal is not zero"
                )
            observed_alpha = float(
                torch.tanh(level.alpha.detach().double())
            )
            if not math.isclose(
                observed_alpha,
                FORMAL_ALPHA_EFFECTIVE_INIT,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    f"formal QFG-V2-CROA level {index} alpha differs"
                )
    manifest = module.architecture_manifest()
    if manifest["parameter_count"] != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise ValueError("formal QFG-V2-CROA manifest parameters differ")
    if manifest["state_key_count"] != PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT:
        raise ValueError("formal QFG-V2-CROA manifest state keys differ")
    return manifest


__all__ = [
    "FORMAL_ALIGNMENT_RATIOS",
    "FORMAL_ALPHA_EFFECTIVE_INIT",
    "FORMAL_FEATURE_CHANNELS",
    "FORMAL_FREQUENCY_MODE",
    "FORMAL_GATE_LIMIT",
    "FORMAL_HIDDEN_CHANNELS",
    "FORMAL_QFG_V2_CROA_PARAMETER_KEYS",
    "FORMAL_QFG_V2_CROA_STATE_KEYS",
    "PRODUCTION_QFG_V2_CROA_PARAMETERS",
    "PRODUCTION_QFG_V2_CROA_PARAMETER_KEY_COUNT",
    "PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT",
    "PreparedQueryFrequencyGateV2CROA",
    "PreparedQueryFrequencyLevelV2CROA",
    "QFG_V2_CROA_VERSION",
    "QueryFrequencyGateOutputV2CROA",
    "QueryFrequencyLevelGateV2CROA",
    "QueryOnlyFrequencyGateV2CROA",
    "RMS_EPS",
    "centered_bounded_arctangent_gate",
    "frequency_gate_parameter_count",
    "sample_full_tensor_rms_normalize",
    "spatial_center_rms_normalize",
    "validate_formal_qfg_v2_croa",
]
