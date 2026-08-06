"""Zero-anchored constant-sum fusion for the four SCTransNet skip levels.

``GlobalConstantSumSkipFusion`` reallocates a fixed coefficient sum of three
between the reconstructed transformer branch and the encoder identity branch.
The formal module has four channel-wise tensors, 480 parameters, four state
keys, and no persistent buffers.  Its zero initialization preserves the
historical operation order ``(T + E) + E`` and adds a zero delta, so both the
forward value and the shared first-step optimization anchor are unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn


GCSF_VERSION = "gcsf_v1_global_constant_sum_skip_reallocation"
FORMAL_GCSF_CHANNELS = (32, 64, 128, 256)
FORMAL_GCSF_GATE_LIMIT = 0.5
PRODUCTION_GCSF_PARAMETERS = 480
PRODUCTION_GCSF_STATE_KEY_COUNT = 4
PRODUCTION_GCSF_BUFFER_COUNT = 0
GCSF_LOCAL_STATE_KEYS = tuple(
    f"reallocation_logits.{level}" for level in range(4)
)


def _normalize_channels(channels: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(channels, (str, bytes)) or not isinstance(
        channels,
        Sequence,
    ):
        raise TypeError("channels must be a sequence of exactly four integers")
    normalized = tuple(channels)
    if len(normalized) != 4:
        raise ValueError("channels must contain exactly four integers")
    for level, value in enumerate(normalized):
        # Deliberately do not coerce floats or bools with int(value).  Channel
        # counts are architecture state even though they are not state_dict
        # tensors, and therefore require an exact integer contract.
        if type(value) is not int:
            raise TypeError(f"channels[{level}] must be an integer")
        if value <= 0:
            raise ValueError(f"channels[{level}] must be positive")
    return normalized


def _require_formal_gate_limit(gate_limit: float) -> float:
    if isinstance(gate_limit, bool):
        raise TypeError("gate_limit must be the finite formal value 0.5")
    try:
        normalized = float(gate_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "gate_limit must be the finite formal value 0.5"
        ) from exc
    if not torch.isfinite(torch.tensor(normalized, dtype=torch.float64)):
        raise ValueError("gate_limit must be finite")
    if normalized != FORMAL_GCSF_GATE_LIMIT:
        raise ValueError("GCSF V1 gate_limit is frozen to exactly 0.5")
    return FORMAL_GCSF_GATE_LIMIT


def _require_level(level: int) -> int:
    if type(level) is not int:
        raise TypeError("fusion level must be an integer")
    if not 0 <= level < 4:
        raise IndexError(f"invalid fusion level {level}")
    return level


def _require_finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if value.device.type == "cuda":
        torch._assert_async(
            torch.isfinite(value).all(),
            f"{name} contains non-finite values",
        )
        return
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _require_branch_tensor(
    value: torch.Tensor,
    *,
    name: str,
    channels: int,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must be a BCHW tensor")
    if any(dimension < 1 for dimension in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if value.shape[1] != channels:
        raise ValueError(
            f"{name} requires C={channels}, got C={value.shape[1]}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    _require_finite_tensor(value, name=name)


class GlobalConstantSumSkipFusion(nn.Module):
    """Channel-wise reallocation between transformed and encoder branches."""

    def __init__(
        self,
        channels: Sequence[int] = FORMAL_GCSF_CHANNELS,
        *,
        gate_limit: float = FORMAL_GCSF_GATE_LIMIT,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.channels = _normalize_channels(channels)
        # Validate the public argument, but calculate with the immutable module
        # constant.  Mutating an incidental Python attribute cannot change the
        # formal eta=0.5 computation.
        _require_formal_gate_limit(gate_limit)
        if dtype is not None:
            probe = torch.empty((), dtype=dtype)
            if not probe.is_floating_point():
                raise TypeError("GCSF parameters require a floating dtype")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.reallocation_logits = nn.ParameterList(
            nn.Parameter(
                torch.zeros(1, channels_i, 1, 1, **factory_kwargs)
            )
            for channels_i in self.channels
        )

    @property
    def gate_limit(self) -> float:
        """Return the immutable formal eta value."""

        return FORMAL_GCSF_GATE_LIMIT

    def gate(self, level: int) -> torch.Tensor:
        level = _require_level(level)
        logits = self.reallocation_logits[level]
        _require_finite_tensor(
            logits,
            name=f"reallocation_logits[{level}]",
        )
        gate = torch.tanh(logits).mul(FORMAL_GCSF_GATE_LIMIT)
        _require_finite_tensor(gate, name=f"gate[{level}]")
        return gate

    def coefficients(
        self,
        level: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return transformed/encoder coefficients with a closed FP bound."""

        gate = self.gate(level)
        transformed = gate.add(1.0)
        encoder = gate.neg().add(2.0)
        # Finite tanh can round to +/-1, so the executable contract is the
        # closed interval even though the real-valued mathematical bound is
        # open.  Clamp protects that exact numeric contract across dtypes.
        transformed = torch.clamp(transformed, min=0.5, max=1.5)
        encoder = torch.clamp(encoder, min=1.5, max=2.5)
        return transformed, encoder

    def forward_level(
        self,
        level: int,
        transformed: torch.Tensor,
        encoder: torch.Tensor,
    ) -> torch.Tensor:
        level = _require_level(level)
        expected_channels = self.channels[level]
        _require_branch_tensor(
            transformed,
            name="transformed",
            channels=expected_channels,
        )
        _require_branch_tensor(
            encoder,
            name="encoder",
            channels=expected_channels,
        )
        if transformed.shape != encoder.shape:
            raise ValueError(
                "transformed and encoder branches must have equal shapes"
            )
        if transformed.device != encoder.device:
            raise ValueError(
                "transformed and encoder branches must share one device"
            )
        if transformed.dtype != encoder.dtype:
            raise TypeError(
                "transformed and encoder branches must share one dtype"
            )
        logits = self.reallocation_logits[level]
        if logits.device != transformed.device:
            raise RuntimeError(
                f"GCSF level {level} parameter device differs from input"
            )
        if logits.dtype != transformed.dtype:
            raise RuntimeError(
                f"GCSF level {level} parameter dtype differs from input"
            )

        gate = self.gate(level)
        # Preserve the historical floating-point order exactly, then apply the
        # learnable reallocation as a delta.  Rewriting this as
        # (1+g)*T + (2-g)*E would destroy the bitwise zero anchor.
        baseline = transformed.add(encoder).add(encoder)
        correction = gate.mul(transformed).sub(gate.mul(encoder))
        fused = baseline.add(correction)
        _require_finite_tensor(fused, name=f"fused[{level}]")
        return fused

    def forward(
        self,
        transformed: Sequence[torch.Tensor],
        encoder: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        if isinstance(transformed, (str, bytes)) or not isinstance(
            transformed,
            Sequence,
        ):
            raise TypeError("transformed must be a four-tensor sequence")
        if isinstance(encoder, (str, bytes)) or not isinstance(
            encoder,
            Sequence,
        ):
            raise TypeError("encoder must be a four-tensor sequence")
        if len(transformed) != 4 or len(encoder) != 4:
            raise ValueError("GCSF requires exactly four branch scales")
        return tuple(
            self.forward_level(level, transformed_i, encoder_i)
            for level, (transformed_i, encoder_i) in enumerate(
                zip(transformed, encoder)
            )
        )

    def architecture_manifest(self) -> Dict[str, Any]:
        return {
            "gcsf_version": GCSF_VERSION,
            "module": "GlobalConstantSumSkipFusion",
            "levels": 4,
            "channels": self.channels,
            "gate_formula": "0.5*tanh(channel_logit)",
            "gate_limit": FORMAL_GCSF_GATE_LIMIT,
            "gate_numeric_bounds_closed": (-0.5, 0.5),
            "transformed_coefficient": "1+gate",
            "transformed_coefficient_numeric_bounds_closed": (0.5, 1.5),
            "encoder_coefficient": "2-gate",
            "encoder_coefficient_numeric_bounds_closed": (1.5, 2.5),
            "coefficient_sum": 3.0,
            "coefficient_sum_is_constant": True,
            "activation_norm_preserved": False,
            "zero_anchor_baseline_order": "(transformed+encoder)+encoder",
            "forward_form": "baseline_plus_gate_times_t_minus_gate_times_e",
            "parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_key_count": len(self.state_dict()),
            "persistent_buffer_count": len(tuple(self.named_buffers())),
            "initialization": "exact_zero",
        }


def validate_formal_global_constant_sum_skip_fusion(
    module: nn.Module,
    *,
    require_zero_initialization: bool = False,
) -> Dict[str, Any]:
    """Validate the exact formal 32/64/128/256 GCSF V1 module."""

    if type(module) is not GlobalConstantSumSkipFusion:
        raise TypeError("formal GCSF must use the exact module class")
    if module.channels != FORMAL_GCSF_CHANNELS:
        raise RuntimeError("formal GCSF channels differ")
    if module.gate_limit != FORMAL_GCSF_GATE_LIMIT:
        raise RuntimeError("formal GCSF gate limit differs")
    if sum(parameter.numel() for parameter in module.parameters()) != (
        PRODUCTION_GCSF_PARAMETERS
    ):
        raise RuntimeError("formal GCSF parameter count differs")
    if tuple(module.state_dict()) != GCSF_LOCAL_STATE_KEYS:
        raise RuntimeError("formal GCSF state keys differ")
    if len(module.state_dict()) != PRODUCTION_GCSF_STATE_KEY_COUNT:
        raise RuntimeError("formal GCSF state-key count differs")
    if len(tuple(module.named_buffers())) != PRODUCTION_GCSF_BUFFER_COUNT:
        raise RuntimeError("formal GCSF must not register persistent buffers")

    reference: nn.Parameter | None = None
    for level, parameter in enumerate(module.reallocation_logits):
        if tuple(parameter.shape) != (1, FORMAL_GCSF_CHANNELS[level], 1, 1):
            raise RuntimeError(f"formal GCSF level {level} shape differs")
        if not parameter.is_floating_point():
            raise RuntimeError(f"formal GCSF level {level} dtype is not floating")
        _require_finite_tensor(
            parameter,
            name=f"formal reallocation_logits[{level}]",
        )
        if reference is None:
            reference = parameter
        elif parameter.device != reference.device or parameter.dtype != (
            reference.dtype
        ):
            raise RuntimeError("formal GCSF parameters differ in device/dtype")
        if (
            require_zero_initialization
            and torch.count_nonzero(parameter).item() != 0
        ):
            raise RuntimeError(
                f"formal GCSF level {level} is not exactly zero initialized"
            )

    manifest = module.architecture_manifest()
    expected = {
        "gcsf_version": GCSF_VERSION,
        "levels": 4,
        "channels": FORMAL_GCSF_CHANNELS,
        "gate_limit": FORMAL_GCSF_GATE_LIMIT,
        "coefficient_sum": 3.0,
        "coefficient_sum_is_constant": True,
        "activation_norm_preserved": False,
        "parameters": PRODUCTION_GCSF_PARAMETERS,
        "state_key_count": PRODUCTION_GCSF_STATE_KEY_COUNT,
        "persistent_buffer_count": PRODUCTION_GCSF_BUFFER_COUNT,
        "initialization": "exact_zero",
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(f"formal GCSF manifest field {name!r} differs")
    return manifest


__all__ = [
    "FORMAL_GCSF_CHANNELS",
    "FORMAL_GCSF_GATE_LIMIT",
    "GCSF_LOCAL_STATE_KEYS",
    "GCSF_VERSION",
    "GlobalConstantSumSkipFusion",
    "PRODUCTION_GCSF_BUFFER_COUNT",
    "PRODUCTION_GCSF_PARAMETERS",
    "PRODUCTION_GCSF_STATE_KEY_COUNT",
    "validate_formal_global_constant_sum_skip_fusion",
]
