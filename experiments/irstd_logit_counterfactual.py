"""Deterministic logit counterfactuals for scarce IRSTD rescue/halo errors.

The frozen Current logit is never modified in-place.  Mode 0 is clean, mode 1
attenuates only target-core logits, and mode 2 injects only the supplied outer
ring.  Callers may provide explicit per-sample modes to guarantee a balanced
batch without relying on random mode counts; amplitudes remain reproducible
under the supplied generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch


IRSTD_COUNTERFACTUAL_VERSION = "irstd_logit_counterfactual_v1"
CLEAN_MODE = 0
CORE_DROP_MODE = 1
RING_INJECTION_MODE = 2
CORE_DROP_RANGE = (0.8, 2.2)
RING_INJECTION_RANGE = (0.5, 1.5)


@dataclass(frozen=True, slots=True)
class CounterfactualBatch:
    logits: torch.Tensor
    halo_target: torch.Tensor
    mode: torch.Tensor


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _validated_binary_mask(
    value: torch.Tensor,
    *,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match current_logits shape")
    if value.device != reference.device:
        raise ValueError(f"{name} must match current_logits device")
    if value.is_floating_point():
        _require_finite(value, name=name)
    elif value.dtype not in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{name} must use bool or a numeric binary dtype")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain only binary values")
    return value.detach().to(dtype=torch.bool)


def _generator_device(generator: torch.Generator) -> torch.device:
    raw_device = getattr(generator, "device", torch.device("cpu"))
    return torch.device(raw_device)


def _uniform_scales(
    batch: int,
    *,
    low: float,
    high: float,
    generator: torch.Generator,
    destination_device: torch.device,
    destination_dtype: torch.dtype,
) -> torch.Tensor:
    random_values = torch.rand(
        (batch, 1, 1, 1),
        generator=generator,
        device=_generator_device(generator),
        dtype=torch.float32,
    )
    scales = low + (high - low) * random_values
    return scales.to(device=destination_device, dtype=destination_dtype)


def _validated_modes(
    modes: torch.Tensor | Sequence[int] | None,
    *,
    batch: int,
    generator: torch.Generator,
    destination_device: torch.device,
) -> torch.Tensor:
    if modes is None:
        sampled = torch.randint(
            CLEAN_MODE,
            RING_INJECTION_MODE + 1,
            (batch,),
            generator=generator,
            device=_generator_device(generator),
            dtype=torch.int64,
        )
        return sampled.to(device=destination_device)
    if isinstance(modes, torch.Tensor):
        if modes.ndim != 1 or modes.shape[0] != batch:
            raise ValueError("modes must have shape [batch]")
        if modes.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("modes must use an integer dtype")
        ready = modes.detach().to(device=destination_device, dtype=torch.int64)
    else:
        if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence):
            raise TypeError("modes must be a tensor or integer sequence")
        if len(modes) != batch:
            raise ValueError("modes must have one value per sample")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in modes
        ):
            raise TypeError("modes must contain only integers")
        ready = torch.tensor(
            [int(value) for value in modes],
            device=destination_device,
            dtype=torch.int64,
        )
    if not bool(
        (
            (ready == CLEAN_MODE)
            | (ready == CORE_DROP_MODE)
            | (ready == RING_INJECTION_MODE)
        ).all()
    ):
        raise ValueError("modes may contain only 0, 1, or 2")
    return ready


def corrupt_irstd_logits(
    *,
    current_logits: torch.Tensor,
    core_target: torch.Tensor,
    outer_ring: torch.Tensor,
    observed_halo_target: torch.Tensor,
    generator: torch.Generator,
    modes: torch.Tensor | Sequence[int] | None = None,
) -> CounterfactualBatch:
    """Return clean/core-drop/ring-injection logits without mutating Current.

    Explicit ``modes`` make the clean/core/ring composition deterministic.  A
    generator is still mandatory because it controls the fixed-range corruption
    amplitudes.  Random values are generated on the generator's own device and
    then copied to the logit's device, so a CPU generator can drive cached CPU
    or CUDA training reproducibly.
    """

    if not isinstance(current_logits, torch.Tensor):
        raise TypeError("current_logits must be a tensor")
    if (
        current_logits.ndim != 4
        or current_logits.shape[1] != 1
        or min(current_logits.shape) < 1
    ):
        raise ValueError("current_logits must be non-empty BCHW with C=1")
    if not current_logits.is_floating_point():
        raise TypeError("current_logits must use a floating-point dtype")
    _require_finite(current_logits, name="current_logits")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    core_mask = _validated_binary_mask(
        core_target,
        name="core_target",
        reference=current_logits,
    )
    ring_mask = _validated_binary_mask(
        outer_ring,
        name="outer_ring",
        reference=current_logits,
    )
    observed_halo_mask = _validated_binary_mask(
        observed_halo_target,
        name="observed_halo_target",
        reference=current_logits,
    )
    if bool((core_mask & ring_mask).any()):
        raise ValueError("core_target and outer_ring must not overlap")
    if bool((core_mask & observed_halo_mask).any()):
        raise ValueError("core_target and observed_halo_target must not overlap")

    batch = current_logits.shape[0]
    device = current_logits.device
    dtype = current_logits.dtype
    ready_modes = _validated_modes(
        modes,
        batch=batch,
        generator=generator,
        destination_device=device,
    )
    drop_scale = _uniform_scales(
        batch,
        low=CORE_DROP_RANGE[0],
        high=CORE_DROP_RANGE[1],
        generator=generator,
        destination_device=device,
        destination_dtype=dtype,
    )
    ring_scale = _uniform_scales(
        batch,
        low=RING_INJECTION_RANGE[0],
        high=RING_INJECTION_RANGE[1],
        generator=generator,
        destination_device=device,
        destination_dtype=dtype,
    )
    core_mode = (ready_modes == CORE_DROP_MODE).view(batch, 1, 1, 1)
    ring_mode = (ready_modes == RING_INJECTION_MODE).view(batch, 1, 1, 1)

    base = current_logits.detach()
    corrupted = (
        base
        - core_mode.to(dtype) * drop_scale * core_mask.to(dtype)
        + ring_mode.to(dtype) * ring_scale * ring_mask.to(dtype)
    )
    halo_target = observed_halo_mask | (ring_mode & ring_mask)
    _require_finite(corrupted, name="counterfactual_logits")
    return CounterfactualBatch(
        logits=corrupted,
        halo_target=halo_target,
        mode=ready_modes,
    )


__all__ = [
    "CLEAN_MODE",
    "CORE_DROP_MODE",
    "CORE_DROP_RANGE",
    "CounterfactualBatch",
    "IRSTD_COUNTERFACTUAL_VERSION",
    "RING_INJECTION_MODE",
    "RING_INJECTION_RANGE",
    "corrupt_irstd_logits",
]
