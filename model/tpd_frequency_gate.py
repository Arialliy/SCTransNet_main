"""Isolated Query-only frequency gate for the final TPD-SCTransNet stage.

The module implements only the frequency-to-Query modulation boundary:

1. fixed per-channel Haar analysis of ``x1...x4``;
2. level-specific projection to the shared Query grid, prepared once per
   whole-model forward; and
3. repeated spatial modulation of the Query maps produced by multiple SCTBs.

It has no K/V input and therefore cannot alter K, V, CFN, decoder features, or
the Keep-Context-Saliency tokenizer.  Integration into ``Attention_org`` must
call this module after ``q1...q4`` convolutions and before flatten/normalize.
Every level owns one scalar ``alpha`` initialized to exactly zero, making the
initial Query maps bitwise identical to the ungated path.

``prepare(...)`` returns a graph-connected, forward-local object.  It must not
be cached across optimizer steps.  ``apply_prepared(...)`` can then reuse that
object for every SCTB without repeating Haar analysis, alignment, or
projection.  ``forward(...)`` remains the one-shot compatibility API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FrequencyMode = str
SUPPORTED_FREQUENCY_MODES = ("high", "low", "high_low")


def _validate_float_map(
    value: torch.Tensor,
    *,
    name: str,
    channels: int | None = None,
    validate_finite: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape BxCxHxW, got {tuple(value.shape)}")
    if min(value.shape) < 1:
        raise ValueError(f"{name} dimensions must be positive")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(
            f"{name} requires {channels} channels, got {value.shape[1]}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if validate_finite and not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


class FixedHaarAnalysis(nn.Module):
    """Orthogonal 2x2 Haar analysis with no trainable parameters."""

    band_names = ("ll", "lh", "hl", "hh")

    def __init__(self, *, validate_finite: bool = False) -> None:
        super().__init__()
        self.validate_finite = bool(validate_finite)
        kernels = torch.tensor(
            (
                ((1.0, 1.0), (1.0, 1.0)),
                ((-1.0, -1.0), (1.0, 1.0)),
                ((-1.0, 1.0), (-1.0, 1.0)),
                ((1.0, -1.0), (-1.0, 1.0)),
            ),
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("kernels", kernels / 2.0, persistent=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _validate_float_map(
            feature,
            name="haar_feature",
            validate_finite=self.validate_finite,
        )
        if feature.shape[-2] % 2 or feature.shape[-1] % 2:
            raise ValueError(
                "Haar analysis requires even H/W, "
                f"got {tuple(feature.shape[-2:])}"
            )
        channels = feature.shape[1]
        weight = self.kernels.to(
            device=feature.device,
            dtype=feature.dtype,
        ).repeat(channels, 1, 1, 1)
        bands = F.conv2d(feature, weight, stride=2, groups=channels)
        return bands.reshape(
            feature.shape[0],
            channels,
            4,
            feature.shape[-2] // 2,
            feature.shape[-1] // 2,
        )


def _mode_channels(channels: int, mode: FrequencyMode) -> int:
    if mode == "high":
        return 3 * channels
    if mode == "low":
        return channels
    if mode == "high_low":
        return 4 * channels
    raise ValueError(
        f"unknown frequency mode {mode!r}; choices={SUPPORTED_FREQUENCY_MODES}"
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


class QueryFrequencyLevelGate(nn.Module):
    """One level-specific Haar prior and zero-strength Query modulation."""

    def __init__(
        self,
        feature_channels: int,
        *,
        mode: FrequencyMode = "high_low",
        hidden_channels: int = 8,
        expected_alignment: int | Tuple[int, int] | None = None,
        validate_finite: bool = False,
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
        self.mode = mode
        self.hidden_channels = int(hidden_channels)
        if isinstance(expected_alignment, int):
            expected_alignment = (
                expected_alignment,
                expected_alignment,
            )
        elif expected_alignment is not None:
            try:
                expected_alignment = tuple(expected_alignment)
            except TypeError as exc:
                raise ValueError(
                    "expected_alignment must be a positive integer or pair"
                ) from exc
        if expected_alignment is not None:
            if (
                len(expected_alignment) != 2
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 1
                    for value in expected_alignment
                )
            ):
                raise ValueError(
                    "expected_alignment must be a positive integer or pair"
                )
        self.expected_alignment = expected_alignment
        self.validate_finite = bool(validate_finite)
        self._prepared_owner_token = object()
        self.haar = FixedHaarAnalysis(validate_finite=validate_finite)
        self.prior_projection = nn.Conv2d(
            _mode_channels(self.feature_channels, self.mode),
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.gate_projection = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                groups=self.hidden_channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    def reset_identity(self) -> None:
        with torch.no_grad():
            self.alpha.zero_()

    @staticmethod
    def _align_prior(
        prior: torch.Tensor,
        query_size: Tuple[int, int],
        expected_alignment: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        prior_h, prior_w = prior.shape[-2:]
        query_h, query_w = query_size
        if query_h < 1 or query_w < 1:
            raise ValueError(f"invalid Query grid {query_size}")
        if prior_h % query_h or prior_w % query_w:
            raise ValueError(
                "Haar prior grid must be an integer multiple of Query grid, "
                f"got prior={(prior_h, prior_w)} query={query_size}"
            )
        factor_h = prior_h // query_h
        factor_w = prior_w // query_w
        if factor_h < 1 or factor_w < 1:
            raise ValueError(
                "Haar prior grid cannot be smaller than Query grid, "
                f"got prior={(prior_h, prior_w)} query={query_size}"
            )
        observed_alignment = (factor_h, factor_w)
        if (
            expected_alignment is not None
            and observed_alignment != expected_alignment
        ):
            raise ValueError(
                "Haar prior to Query alignment differs from the registered "
                f"level ratio: expected={expected_alignment} "
                f"observed={observed_alignment}"
            )
        if factor_h == 1 and factor_w == 1:
            return prior
        return F.avg_pool2d(
            prior,
            kernel_size=(factor_h, factor_w),
            stride=(factor_h, factor_w),
        )

    def prepare(
        self,
        feature: torch.Tensor,
        query_size: Tuple[int, int],
    ) -> "PreparedQueryFrequencyLevel":
        """Prepare this level's projected frequency logits exactly once."""

        _validate_float_map(
            feature,
            name="frequency_feature",
            channels=self.feature_channels,
            validate_finite=self.validate_finite,
        )
        query_size = _validate_query_size(query_size, name="query_size")
        bands = self.haar(feature)
        selected = _select_bands(bands, self.mode)
        selected = self._align_prior(
            selected,
            query_size,
            self.expected_alignment,
        )
        prior = self.prior_projection(selected)
        gate_logits = self.gate_projection(prior)
        if self.validate_finite and not torch.isfinite(gate_logits).all():
            raise FloatingPointError("prepared gate logits contain non-finite values")
        return PreparedQueryFrequencyLevel(
            gate_logits=gate_logits,
            query_size=query_size,
            batch_size=int(feature.shape[0]),
            _owner_token=self._prepared_owner_token,
        )

    def apply_prepared(
        self,
        query: torch.Tensor,
        prepared: "PreparedQueryFrequencyLevel",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a forward-local prepared prior to one Query map."""

        _validate_float_map(
            query,
            name="query",
            validate_finite=self.validate_finite,
        )
        if not isinstance(prepared, PreparedQueryFrequencyLevel):
            raise TypeError(
                "prepared must be a PreparedQueryFrequencyLevel from prepare()"
            )
        if prepared._owner_token is not self._prepared_owner_token:
            raise ValueError(
                "prepared frequency prior belongs to a different gate level"
            )
        gate_logits = prepared.gate_logits
        _validate_float_map(
            gate_logits,
            name="prepared_gate_logits",
            channels=1,
            validate_finite=self.validate_finite,
        )
        if tuple(query.shape[-2:]) != prepared.query_size:
            raise ValueError(
                "Query grid differs from the prepared grid: "
                f"prepared={prepared.query_size} "
                f"query={tuple(query.shape[-2:])}"
            )
        if query.shape[0] != prepared.batch_size:
            raise ValueError(
                "Query and prepared frequency prior batch sizes must match"
            )
        if gate_logits.shape[0] != prepared.batch_size:
            raise ValueError("prepared frequency prior has inconsistent batch metadata")
        if tuple(gate_logits.shape[-2:]) != prepared.query_size:
            raise ValueError("prepared frequency prior has inconsistent grid metadata")
        if query.device != gate_logits.device:
            raise ValueError("Query and prepared frequency prior devices must match")
        working_dtype = torch.promote_types(query.dtype, torch.float32)
        gate = torch.tanh(gate_logits.to(dtype=working_dtype))
        effective_alpha = torch.tanh(self.alpha.to(dtype=working_dtype))
        factor = 1.0 + effective_alpha * gate
        modulated = (
            query.to(dtype=working_dtype) * factor
        ).to(dtype=query.dtype)
        return modulated, gate_logits, factor

    def forward(
        self,
        query: torch.Tensor,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility path equivalent to prepare once followed by one apply."""

        _validate_float_map(
            query,
            name="query",
            validate_finite=self.validate_finite,
        )
        if query.shape[0] != feature.shape[0]:
            raise ValueError("Query and frequency feature batch sizes must match")
        if query.device != feature.device:
            raise ValueError("Query and frequency feature devices must match")
        prepared = self.prepare(feature, tuple(query.shape[-2:]))
        return self.apply_prepared(query, prepared)


def _validate_query_size(
    value: Sequence[int],
    *,
    name: str,
) -> Tuple[int, int]:
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


@dataclass(frozen=True, slots=True)
class PreparedQueryFrequencyLevel:
    """Projected frequency logits scoped to one whole-model forward."""

    gate_logits: torch.Tensor
    query_size: Tuple[int, int]
    batch_size: int
    _owner_token: object


@dataclass(frozen=True, slots=True)
class QueryFrequencyGateOutput:
    queries: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    gate_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    factors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class PreparedQueryFrequencyGate:
    """Four prepared level priors reusable by multiple SCTB Query groups."""

    levels: Tuple[
        PreparedQueryFrequencyLevel,
        PreparedQueryFrequencyLevel,
        PreparedQueryFrequencyLevel,
        PreparedQueryFrequencyLevel,
    ]
    _owner_token: object

    @property
    def gate_logits(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(level.gate_logits for level in self.levels)


class QueryOnlyFrequencyGate(nn.Module):
    """Four independent spatial gates for already generated ``q1...q4``."""

    def __init__(
        self,
        feature_channels: Sequence[int] = (32, 64, 128, 256),
        *,
        mode: FrequencyMode = "high_low",
        hidden_channels: int = 8,
        validate_finite: bool = False,
    ) -> None:
        super().__init__()
        try:
            channels = tuple(feature_channels)
        except TypeError as exc:
            raise TypeError("feature_channels must be a finite sequence") from exc
        if len(channels) != 4:
            raise ValueError(
                f"Query-only FG requires four feature levels, got {len(channels)}"
            )
        self.feature_channels = channels
        self.mode = mode
        self.hidden_channels = int(hidden_channels)
        self.validate_finite = bool(validate_finite)
        self._prepared_owner_token = object()
        self.levels = nn.ModuleList(
            QueryFrequencyLevelGate(
                channels[index],
                mode=mode,
                hidden_channels=hidden_channels,
                expected_alignment=(8, 4, 2, 1)[index],
                validate_finite=validate_finite,
            )
            for index in range(4)
        )

    def reset_identity(self) -> None:
        for level in self.levels:
            level.reset_identity()

    @staticmethod
    def _normalize_query_sizes(
        query_sizes: Sequence[Sequence[int]] | Sequence[int],
    ) -> Tuple[
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
    ]:
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
            shared_size = _validate_query_size(raw_sizes, name="query_sizes")
            return (shared_size, shared_size, shared_size, shared_size)
        if len(raw_sizes) != 4:
            raise ValueError("query_sizes must be one H/W pair or four H/W pairs")
        return tuple(
            _validate_query_size(value, name=f"query_sizes[{index}]")
            for index, value in enumerate(raw_sizes)
        )

    def prepare(
        self,
        encoder_features: Sequence[torch.Tensor],
        query_sizes: Sequence[Sequence[int]] | Sequence[int],
    ) -> PreparedQueryFrequencyGate:
        """Run Haar/alignment/projection once for one whole-model forward."""

        encoder_features = tuple(encoder_features)
        if len(encoder_features) != 4:
            raise ValueError(
                "Query-only FG requires exactly four encoder features"
            )
        normalized_sizes = self._normalize_query_sizes(query_sizes)
        prepared_levels = tuple(
            level.prepare(feature, query_size)
            for level, feature, query_size in zip(
                self.levels,
                encoder_features,
                normalized_sizes,
            )
        )
        return PreparedQueryFrequencyGate(
            levels=prepared_levels,
            _owner_token=self._prepared_owner_token,
        )

    def apply_prepared(
        self,
        queries: Sequence[torch.Tensor],
        prepared: PreparedQueryFrequencyGate,
    ) -> QueryFrequencyGateOutput:
        """Apply one prepared object to one SCTB's four Query maps."""

        queries = tuple(queries)
        if len(queries) != 4:
            raise ValueError(
                "Query-only FG requires exactly four Query maps"
            )
        if not isinstance(prepared, PreparedQueryFrequencyGate):
            raise TypeError(
                "prepared must be a PreparedQueryFrequencyGate from prepare()"
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
        query_outputs, gate_logits, factors = zip(*outputs)
        return QueryFrequencyGateOutput(
            queries=tuple(query_outputs),
            gate_logits=tuple(gate_logits),
            factors=tuple(factors),
        )

    def forward(
        self,
        queries: Sequence[torch.Tensor],
        encoder_features: Sequence[torch.Tensor],
    ) -> QueryFrequencyGateOutput:
        """One-shot compatibility API built from prepare/apply_prepared."""

        queries = tuple(queries)
        encoder_features = tuple(encoder_features)
        if len(queries) != 4 or len(encoder_features) != 4:
            raise ValueError(
                "Query-only FG requires exactly four Query maps and four "
                "encoder features"
            )
        prepared = self.prepare(
            encoder_features,
            tuple(tuple(query.shape[-2:]) for query in queries),
        )
        return self.apply_prepared(queries, prepared)

    def architecture_manifest(self) -> Dict[str, object]:
        return {
            "module": "QueryOnlyFrequencyGate",
            "frequency_transform": "fixed_orthogonal_haar_2x2",
            "frequency_mode": self.mode,
            "high_frequency_representation": "absolute_magnitude",
            "feature_channels": self.feature_channels,
            "query_levels": 4,
            "registered_alignment_ratios": (8, 4, 2, 1),
            "modulation_location": "post_q_convolution_pre_normalization",
            "modified_attention_tensors": ("Q",),
            "kv_modified": False,
            "cfn_modified": False,
            "decoder_injection": False,
            "tokenizer_branch_added": False,
            "alpha_initialization": 0.0,
            "alpha_parameterization": "tanh_bounded_no_sign_reversal",
            "finite_validation": self.validate_finite,
            "projection_order": "haar_align_then_1x1",
            "execution": "prepare_once_apply_many_per_model_forward",
            "prepared_object_persistence": "forward_local_only",
        }


def frequency_gate_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "FixedHaarAnalysis",
    "PreparedQueryFrequencyGate",
    "PreparedQueryFrequencyLevel",
    "QueryFrequencyGateOutput",
    "QueryFrequencyLevelGate",
    "QueryOnlyFrequencyGate",
    "SUPPORTED_FREQUENCY_MODES",
    "frequency_gate_parameter_count",
]
