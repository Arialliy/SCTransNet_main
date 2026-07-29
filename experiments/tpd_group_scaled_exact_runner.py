"""Exact-runner extension for deterministic per-parameter-group LR scaling.

This module is intentionally independent from every active training entry.
It does not change :mod:`experiments.tpd_exact_runner` or any source-lock
manifest.  A future training entry may opt in by constructing every optimizer
parameter group with these two identity-bound options:

``group_name``
    A unique, non-empty name whose optimizer order is authoritative.

``schedule_multiplier``
    A finite positive multiplier.  Every group must still be constructed with
    ``lr == ManualCosineSchedule.base_lr``.

For epoch ``e`` and optimizer group ``i``, the only LR formula is::

    group_lr(e, i) = manual_cosine_lr(e) * schedule_multiplier(i)

``GroupScaledExactRunner.next_epoch_control`` first delegates to the existing
runner and then applies this formula.  Before a commit, the scaled values and
identity-bound metadata are checked and recorded in the epoch event.  Group
LRs are then temporarily canonicalized to ``manual_cosine_lr(e)`` while the
base runner creates the exact checkpoint.  Consequently, exact-resume keeps
the existing checkpoint format and always resumes from a canonical base LR;
the next epoch control deterministically reapplies the group multipliers.

When every multiplier is ``1.0``, optimizer updates and manual scheduling are
numerically identical to the original :class:`ExactRunner`.  The added group
names, multipliers, and event evidence deliberately remain identity/audit
metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from experiments.tpd_exact_runner import (
    CompatibilityPayloadFactory,
    EpochControl,
    ExactRunSpec,
    ExactRunner,
    ExactRunnerError,
    RunnerSnapshot,
    SelectionPolicy,
)


GROUP_NAME_OPTION = "group_name"
SCHEDULE_MULTIPLIER_OPTION = "schedule_multiplier"
GROUP_NAMES_EVENT_FIELD = "optimizer_group_names"
GROUP_LEARNING_RATES_EVENT_FIELD = "group_learning_rates"
SCHEDULE_MULTIPLIERS_EVENT_FIELD = "schedule_multipliers"
GROUP_LR_FORMULA = (
    "group_lr(epoch,index)=manual_cosine_lr(epoch)"
    "*schedule_multiplier(index)"
)

_OWNED_EVENT_FIELDS = frozenset(
    {
        GROUP_NAMES_EVENT_FIELD,
        GROUP_LEARNING_RATES_EVENT_FIELD,
        SCHEDULE_MULTIPLIERS_EVENT_FIELD,
    }
)


def _fail(message: str) -> None:
    raise ExactRunnerError(message)


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be a finite number")
    return float(value)


def _validate_group_definitions(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        _fail("optimizer must be a torch.optim.Optimizer")
    if not optimizer.param_groups:
        _fail("optimizer must contain at least one parameter group")
    expected_base = _finite_number(base_lr, "manual schedule base_lr")
    if expected_base <= 0.0:
        _fail("manual schedule base_lr must be positive")

    records: list[tuple[str, float]] = []
    seen_names: set[str] = set()
    for index, group in enumerate(optimizer.param_groups):
        name = group.get(GROUP_NAME_OPTION)
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
        ):
            _fail(
                f"optimizer param-group {index} group_name must be a "
                "non-empty trimmed string"
            )
        if name in seen_names:
            _fail(f"optimizer group_name {name!r} appears more than once")
        seen_names.add(name)

        if SCHEDULE_MULTIPLIER_OPTION not in group:
            _fail(
                f"optimizer param-group {index} lacks "
                "schedule_multiplier"
            )
        multiplier = _finite_number(
            group[SCHEDULE_MULTIPLIER_OPTION],
            f"optimizer param-group {index} schedule_multiplier",
        )
        if multiplier <= 0.0:
            _fail(
                f"optimizer param-group {index} schedule_multiplier "
                "must be positive"
            )
        actual_lr = _finite_number(
            group.get("lr"),
            f"optimizer param-group {index} LR",
        )
        if actual_lr != expected_base:
            _fail(
                f"optimizer param-group {index} LR must equal manual "
                "schedule base_lr at runner construction"
            )
        records.append((name, multiplier))
    return tuple(records)


def group_scaled_determinism_contract() -> dict[str, Any]:
    """Return fields a caller can merge into ``ExactRunSpec.determinism``."""

    return {
        "manual_group_scaled_lr": True,
        "group_lr_formula": GROUP_LR_FORMULA,
        "group_order_source": "optimizer.param_groups",
        "checkpoint_group_lr": "manual_cosine_lr(epoch)",
        "next_epoch_reapplies_schedule_multiplier": True,
    }


def freeze_batchnorm_running_stats(model: nn.Module) -> int:
    """Put all BatchNorm modules in eval mode without freezing affine tensors.

    The function deliberately does not change ``weight.requires_grad`` or
    ``bias.requires_grad``.  Call it after any later ``model.train()`` call,
    because PyTorch's recursive ``train()`` method otherwise re-enables
    BatchNorm running-stat updates.

    Returns the number of BatchNorm modules placed in eval mode.
    """

    if not isinstance(model, nn.Module):
        _fail("BatchNorm freeze target must be an nn.Module")
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


class GroupScaledExactRunner(ExactRunner):
    """ExactRunner that applies identity-bound LR multipliers by group."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        loader_generator: torch.Generator,
        spec: ExactRunSpec,
        selection_policy: SelectionPolicy | None = None,
        compatibility_payload_factory: CompatibilityPayloadFactory | None = None,
        map_location: str | torch.device = "cpu",
    ) -> None:
        base_lr = spec.lr_schedule.normalized()["base_lr"]
        group_definitions = _validate_group_definitions(
            optimizer,
            base_lr=base_lr,
        )
        super().__init__(
            run_directory,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader_generator=loader_generator,
            spec=spec,
            selection_policy=selection_policy,
            compatibility_payload_factory=compatibility_payload_factory,
            map_location=map_location,
        )
        self._group_names = tuple(name for name, _ in group_definitions)
        self._schedule_multipliers = tuple(
            multiplier for _, multiplier in group_definitions
        )

    @property
    def optimizer_group_names(self) -> tuple[str, ...]:
        """Return the immutable optimizer-group order bound at construction."""

        return self._group_names

    @property
    def schedule_multipliers(self) -> tuple[float, ...]:
        """Return the immutable schedule multipliers bound at construction."""

        return self._schedule_multipliers

    def _validate_live_group_metadata(self) -> None:
        if len(self.optimizer.param_groups) != len(self._group_names):
            _fail("optimizer param-group count changed after construction")
        for index, (group, expected_name, expected_multiplier) in enumerate(
            zip(
                self.optimizer.param_groups,
                self._group_names,
                self._schedule_multipliers,
            )
        ):
            if group.get(GROUP_NAME_OPTION) != expected_name:
                _fail(
                    f"optimizer param-group {index} group_name changed "
                    "after construction"
                )
            actual_multiplier = _finite_number(
                group.get(SCHEDULE_MULTIPLIER_OPTION),
                f"optimizer param-group {index} schedule_multiplier",
            )
            if actual_multiplier != expected_multiplier:
                _fail(
                    f"optimizer param-group {index} schedule_multiplier "
                    "changed after construction"
                )

    def _expected_group_lrs(self, base_lr: float) -> tuple[float, ...]:
        base = _finite_number(base_lr, "open epoch base learning rate")
        return tuple(
            base * multiplier
            for multiplier in self._schedule_multipliers
        )

    def _set_group_lrs(self, values: tuple[float, ...]) -> None:
        if len(values) != len(self.optimizer.param_groups):
            _fail("group LR count differs from optimizer param-group count")
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value

    def _require_group_lrs(self, expected: tuple[float, ...]) -> None:
        if len(expected) != len(self.optimizer.param_groups):
            _fail("group LR count differs from optimizer param-group count")
        for index, (group, expected_lr) in enumerate(
            zip(self.optimizer.param_groups, expected)
        ):
            actual_lr = _finite_number(
                group.get("lr"),
                f"optimizer param-group {index} LR",
            )
            if actual_lr != expected_lr:
                _fail(
                    f"optimizer param-group {index} LR differs from the "
                    "scaled open epoch control"
                )

    def next_epoch_control(self) -> EpochControl:
        """Open the next base control, then apply deterministic group scaling."""

        self._validate_live_group_metadata()
        control = super().next_epoch_control()
        self._set_group_lrs(
            self._expected_group_lrs(control.learning_rate)
        )
        return control

    def commit_epoch(
        self,
        fields: Mapping[str, Any],
        *,
        extra_state: Mapping[str, Any] | None = None,
    ) -> RunnerSnapshot:
        """Validate scaled LRs, record evidence, and commit canonical base LRs."""

        if not isinstance(fields, Mapping):
            _fail("epoch fields must be a mapping")
        forged = sorted(_OWNED_EVENT_FIELDS & set(fields))
        if forged:
            _fail(
                "epoch fields contain group-scaled-runner-owned keys: "
                f"{forged}"
            )
        if self._open_control is None:
            _fail("next_epoch_control must be called before commit_epoch")

        self._validate_live_group_metadata()
        base_lr = self._open_control.learning_rate
        scaled_lrs = self._expected_group_lrs(base_lr)
        self._require_group_lrs(scaled_lrs)

        annotated_fields = dict(fields)
        annotated_fields[GROUP_NAMES_EVENT_FIELD] = list(
            self._group_names
        )
        annotated_fields[GROUP_LEARNING_RATES_EVENT_FIELD] = list(
            scaled_lrs
        )
        annotated_fields[SCHEDULE_MULTIPLIERS_EVENT_FIELD] = list(
            self._schedule_multipliers
        )

        canonical_lrs = tuple(
            base_lr for _ in self.optimizer.param_groups
        )
        self._set_group_lrs(canonical_lrs)
        try:
            return super().commit_epoch(
                annotated_fields,
                extra_state=extra_state,
            )
        except BaseException:
            # A pre-prepare rejection keeps the epoch open and has no pending
            # exact payload, so restore training LRs and permit corrected
            # fields to be retried.  A prepared/pending commit must retain the
            # canonical base LRs captured in its payload.  Likewise, once the
            # base runner closes the control (including post-commit derived
            # publication failure), the committed boundary remains canonical.
            if self._pending is None and self._open_control is not None:
                self._set_group_lrs(scaled_lrs)
            raise


__all__ = [
    "GROUP_LEARNING_RATES_EVENT_FIELD",
    "GROUP_LR_FORMULA",
    "GROUP_NAMES_EVENT_FIELD",
    "GROUP_NAME_OPTION",
    "GroupScaledExactRunner",
    "SCHEDULE_MULTIPLIERS_EVENT_FIELD",
    "SCHEDULE_MULTIPLIER_OPTION",
    "freeze_batchnorm_running_stats",
    "group_scaled_determinism_contract",
]
