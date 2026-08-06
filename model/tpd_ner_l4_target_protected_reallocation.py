"""NER-conditioned target-protected reallocation at skip level four.

The module keeps the current Final L4 fusion ``(T + E) + E`` as an exact
zero-parameter anchor.  The existing NER q4 upper-tail evidence defines a
detached, binary target-protection map.  Outside the protected region, one
zero-initialized channel gate reallocates a constant coefficient sum between
the reconstructed Transformer branch and the encoder branch::

    B = (T + E) + E
    P = dilate_3x3(1[tail_support(stopgrad(q4), 1.5) > 0])
    G = 0.25 * tanh(a4)
    X = B + (1 - P) * G * T - (1 - P) * G * E

No persistent buffer is registered.  At ``a4 == 0`` the added delta is zero;
inside ``P == 1`` it remains zero for every value of ``a4``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    relay_spatial_tail_support,
)


NER_L4_TPR_VERSION = "ner_l4_tpr_v1_binary_tail_protected_reallocation"
FORMAL_L4_CHANNELS = 256
FORMAL_Q4_RELAY_CHANNELS = 8
FORMAL_L4_GATE_LIMIT = 0.25
FORMAL_L4_TAIL_Z_THRESHOLD = 1.5
FORMAL_L4_PROTECTION_DILATION_KERNEL = 3
PRODUCTION_NER_L4_TPR_PARAMETERS = 256
PRODUCTION_NER_L4_TPR_STATE_KEY_COUNT = 1
PRODUCTION_NER_L4_TPR_BUFFER_COUNT = 0
NER_L4_TPR_LOCAL_STATE_KEYS = ("reallocation_logits",)


def _require_exact_int(value: int, *, expected: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be the integer {expected}")
    if value != expected:
        raise ValueError(f"{name} is frozen to {expected}, got {value}")
    return expected


def _require_exact_float(
    value: float,
    *,
    expected: float,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be the finite value {expected}")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be the finite value {expected}"
        ) from exc
    if not torch.isfinite(torch.tensor(normalized, dtype=torch.float64)):
        raise ValueError(f"{name} must be finite")
    if normalized != expected:
        raise ValueError(f"{name} is frozen to {expected}, got {normalized}")
    return expected


def _require_finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if value.device.type == "cuda":
        torch._assert_async(
            torch.isfinite(value).all(),
            f"{name} contains non-finite values",
        )
        return
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _require_bchw_float(
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


class NERL4TargetProtectedReallocation(nn.Module):
    """One channel-wise L4 gate routed by existing NER q4 evidence."""

    def __init__(
        self,
        channels: int = FORMAL_L4_CHANNELS,
        *,
        gate_limit: float = FORMAL_L4_GATE_LIMIT,
        tail_z_threshold: float = FORMAL_L4_TAIL_Z_THRESHOLD,
        dilation_kernel: int = FORMAL_L4_PROTECTION_DILATION_KERNEL,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.channels = _require_exact_int(
            channels,
            expected=FORMAL_L4_CHANNELS,
            name="channels",
        )
        _require_exact_float(
            gate_limit,
            expected=FORMAL_L4_GATE_LIMIT,
            name="gate_limit",
        )
        _require_exact_float(
            tail_z_threshold,
            expected=FORMAL_L4_TAIL_Z_THRESHOLD,
            name="tail_z_threshold",
        )
        _require_exact_int(
            dilation_kernel,
            expected=FORMAL_L4_PROTECTION_DILATION_KERNEL,
            name="dilation_kernel",
        )
        if dtype is not None:
            probe = torch.empty((), dtype=dtype)
            if not probe.is_floating_point():
                raise TypeError("NER-L4-TPR parameter requires floating dtype")
        self.reallocation_logits = nn.Parameter(
            torch.zeros(
                1,
                FORMAL_L4_CHANNELS,
                1,
                1,
                device=device,
                dtype=dtype,
            )
        )

    @property
    def gate_limit(self) -> float:
        return FORMAL_L4_GATE_LIMIT

    @property
    def tail_z_threshold(self) -> float:
        return FORMAL_L4_TAIL_Z_THRESHOLD

    @property
    def dilation_kernel(self) -> int:
        return FORMAL_L4_PROTECTION_DILATION_KERNEL

    def gate(self) -> torch.Tensor:
        """Return the bounded per-channel reallocation gate."""

        _require_finite_tensor(
            self.reallocation_logits,
            name="reallocation_logits",
        )
        gate = torch.tanh(self.reallocation_logits).mul(
            FORMAL_L4_GATE_LIMIT
        )
        _require_finite_tensor(gate, name="gate")
        return gate

    def build_protection(self, q4: torch.Tensor) -> torch.Tensor:
        """Build a detached Bx1xHxW binary map from existing q4 tail support."""

        _require_bchw_float(
            q4,
            name="q4",
            channels=FORMAL_Q4_RELAY_CHANNELS,
        )
        # q4 is evidence, not a trainable shortcut for changing the routing
        # partition.  Both its tail calculation and the final map are detached.
        with torch.no_grad():
            tail = relay_spatial_tail_support(
                q4.detach(),
                z_threshold=FORMAL_L4_TAIL_Z_THRESHOLD,
            )
            binary = tail.gt(0.0).to(dtype=q4.dtype)
            protection = F.max_pool2d(
                binary,
                kernel_size=FORMAL_L4_PROTECTION_DILATION_KERNEL,
                stride=1,
                padding=FORMAL_L4_PROTECTION_DILATION_KERNEL // 2,
            )
        protection = protection.detach()
        if tuple(protection.shape) != (
            q4.shape[0],
            1,
            q4.shape[2],
            q4.shape[3],
        ):
            raise RuntimeError("NER-L4-TPR protection shape differs")
        if protection.requires_grad:
            raise RuntimeError("NER-L4-TPR protection must be detached")
        return protection

    def coefficients(
        self,
        q4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return transformed/encoder coefficients and the protection map."""

        protection = self.build_protection(q4)
        eligible = protection.neg().add(1.0)
        routed_gate = eligible.mul(self.gate())
        transformed = routed_gate.add(1.0)
        encoder = routed_gate.neg().add(2.0)
        return transformed, encoder, protection

    def forward(
        self,
        transformed: torch.Tensor,
        encoder: torch.Tensor,
        q4: torch.Tensor,
    ) -> torch.Tensor:
        """Apply target-protected constant-sum reallocation at L4."""

        _require_bchw_float(
            transformed,
            name="transformed",
            channels=FORMAL_L4_CHANNELS,
        )
        _require_bchw_float(
            encoder,
            name="encoder",
            channels=FORMAL_L4_CHANNELS,
        )
        _require_bchw_float(
            q4,
            name="q4",
            channels=FORMAL_Q4_RELAY_CHANNELS,
        )
        if transformed.shape != encoder.shape:
            raise ValueError(
                "transformed and encoder branches must have equal shapes"
            )
        if transformed.shape[0] != q4.shape[0] or transformed.shape[-2:] != (
            q4.shape[-2:]
        ):
            raise ValueError("q4 batch/spatial shape must match L4 branches")
        if transformed.device != encoder.device or transformed.device != (
            q4.device
        ):
            raise ValueError("NER-L4-TPR tensors must share one device")
        if transformed.dtype != encoder.dtype or transformed.dtype != q4.dtype:
            raise TypeError("NER-L4-TPR tensors must share one dtype")
        if self.reallocation_logits.device != transformed.device:
            raise RuntimeError("NER-L4-TPR parameter device differs from input")
        if self.reallocation_logits.dtype != transformed.dtype:
            raise RuntimeError("NER-L4-TPR parameter dtype differs from input")

        protection = self.build_protection(q4)
        eligible = protection.neg().add(1.0)
        routed_gate = eligible.mul(self.gate())

        # Preserve the current Final operation order exactly.  The correction
        # is deliberately written as two ordered products rather than folded
        # coefficients so a zero gate is a strict baseline-plus-zero anchor.
        baseline = transformed.add(encoder).add(encoder)
        correction = routed_gate.mul(transformed).sub(
            routed_gate.mul(encoder)
        )
        fused = baseline.add(correction)
        _require_finite_tensor(fused, name="fused")
        return fused

    def architecture_manifest(self) -> Dict[str, Any]:
        return {
            "ner_l4_tpr_version": NER_L4_TPR_VERSION,
            "module": "NERL4TargetProtectedReallocation",
            "level": 4,
            "channels": FORMAL_L4_CHANNELS,
            "q4_relay_channels": FORMAL_Q4_RELAY_CHANNELS,
            "gate_formula": "0.25*tanh(reallocation_logits)",
            "gate_limit": FORMAL_L4_GATE_LIMIT,
            "gate_numeric_bounds_closed": (-0.25, 0.25),
            "tail_support_source": "existing_ner_q4",
            "tail_z_threshold": FORMAL_L4_TAIL_Z_THRESHOLD,
            "protection_binarization": "tail_support>0",
            "protection_dilation": "max_pool_3x3_stride1_padding1",
            "protection_detached": True,
            "protected_region_fusion": "(T+E)+E",
            "eligible_region_fusion": (
                "((T+E)+E)+(1-P)*G*T-(1-P)*G*E"
            ),
            "transformed_coefficient": "1+(1-P)*G",
            "encoder_coefficient": "2-(1-P)*G",
            "coefficient_sum": 3.0,
            "coefficient_sum_is_constant": True,
            "zero_anchor_baseline_order": "(transformed+encoder)+encoder",
            "initialization": "exact_zero",
            "parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_key_count": len(self.state_dict()),
            "persistent_buffer_count": len(tuple(self.named_buffers())),
        }


def validate_formal_ner_l4_target_protected_reallocation(
    module: nn.Module,
    *,
    require_zero_initialization: bool = False,
) -> Dict[str, Any]:
    """Validate the exact formal NER-L4-TPR V1 module."""

    if type(module) is not NERL4TargetProtectedReallocation:
        raise TypeError("formal NER-L4-TPR must use the exact module class")
    if module.channels != FORMAL_L4_CHANNELS:
        raise RuntimeError("formal NER-L4-TPR channel count differs")
    if module.gate_limit != FORMAL_L4_GATE_LIMIT:
        raise RuntimeError("formal NER-L4-TPR gate limit differs")
    if module.tail_z_threshold != FORMAL_L4_TAIL_Z_THRESHOLD:
        raise RuntimeError("formal NER-L4-TPR tail threshold differs")
    if module.dilation_kernel != FORMAL_L4_PROTECTION_DILATION_KERNEL:
        raise RuntimeError("formal NER-L4-TPR dilation differs")
    if tuple(module.reallocation_logits.shape) != (
        1,
        FORMAL_L4_CHANNELS,
        1,
        1,
    ):
        raise RuntimeError("formal NER-L4-TPR parameter shape differs")
    if not module.reallocation_logits.is_floating_point():
        raise RuntimeError("formal NER-L4-TPR parameter must be floating")
    _require_finite_tensor(
        module.reallocation_logits,
        name="formal reallocation_logits",
    )
    if sum(parameter.numel() for parameter in module.parameters()) != (
        PRODUCTION_NER_L4_TPR_PARAMETERS
    ):
        raise RuntimeError("formal NER-L4-TPR parameter count differs")
    if tuple(module.state_dict()) != NER_L4_TPR_LOCAL_STATE_KEYS:
        raise RuntimeError("formal NER-L4-TPR state keys differ")
    if len(module.state_dict()) != PRODUCTION_NER_L4_TPR_STATE_KEY_COUNT:
        raise RuntimeError("formal NER-L4-TPR state-key count differs")
    if len(tuple(module.named_buffers())) != PRODUCTION_NER_L4_TPR_BUFFER_COUNT:
        raise RuntimeError("formal NER-L4-TPR must not register buffers")
    if (
        require_zero_initialization
        and torch.count_nonzero(module.reallocation_logits).item() != 0
    ):
        raise RuntimeError("formal NER-L4-TPR parameter is not exactly zero")

    manifest = module.architecture_manifest()
    expected = {
        "ner_l4_tpr_version": NER_L4_TPR_VERSION,
        "level": 4,
        "channels": FORMAL_L4_CHANNELS,
        "q4_relay_channels": FORMAL_Q4_RELAY_CHANNELS,
        "gate_limit": FORMAL_L4_GATE_LIMIT,
        "tail_z_threshold": FORMAL_L4_TAIL_Z_THRESHOLD,
        "protection_detached": True,
        "coefficient_sum": 3.0,
        "coefficient_sum_is_constant": True,
        "initialization": "exact_zero",
        "parameters": PRODUCTION_NER_L4_TPR_PARAMETERS,
        "state_key_count": PRODUCTION_NER_L4_TPR_STATE_KEY_COUNT,
        "persistent_buffer_count": PRODUCTION_NER_L4_TPR_BUFFER_COUNT,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(
                f"formal NER-L4-TPR manifest field {name!r} differs"
            )
    return manifest


__all__ = [
    "FORMAL_L4_CHANNELS",
    "FORMAL_L4_GATE_LIMIT",
    "FORMAL_L4_PROTECTION_DILATION_KERNEL",
    "FORMAL_L4_TAIL_Z_THRESHOLD",
    "FORMAL_Q4_RELAY_CHANNELS",
    "NER_L4_TPR_LOCAL_STATE_KEYS",
    "NER_L4_TPR_VERSION",
    "NERL4TargetProtectedReallocation",
    "PRODUCTION_NER_L4_TPR_BUFFER_COUNT",
    "PRODUCTION_NER_L4_TPR_PARAMETERS",
    "PRODUCTION_NER_L4_TPR_STATE_KEY_COUNT",
    "validate_formal_ner_l4_target_protected_reallocation",
]
