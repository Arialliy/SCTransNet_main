"""Generic exact-training coordinator built on exact resume and A/B journal.

This module deliberately does not contain a model training loop.  A caller
performs one epoch, computes its metrics/best-selection state, and then uses
``prepare_epoch`` followed immediately by ``commit_epoch``.  Fresh and resume
startup are explicit and mutually exclusive; resume requires independently
reconstructed epoch, metrics-boundary, best-selection, and run-identity
expectations before any training state is restored.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from experiments import tpd_exact_resume as exact
from experiments.tpd_exact_epoch_journal import (
    ActiveEpochState,
    ExactEpochJournal,
    PreparedEpochEvent,
)


class ExactTrainingRuntimeError(RuntimeError):
    """The coordinator was used outside its explicit startup/commit state."""


@dataclass(frozen=True)
class ExactTrainingSnapshot:
    """Coordinator state at the last committed epoch."""

    mode: exact.InitializationMode
    completed_epoch: int
    run_identity: dict[str, Any]
    metrics_boundary: dict[str, Any] | None
    best_selection: dict[str, Any] | None
    extra_state: dict[str, Any]
    active: ActiveEpochState | None


@dataclass(frozen=True)
class PreparedExactEpoch:
    """One exact payload bound to one prepared journal event."""

    event: PreparedEpochEvent
    exact_payload: dict[str, Any]

    @property
    def epoch(self) -> int:
        return self.event.epoch

    @property
    def metrics_boundary(self) -> dict[str, Any]:
        return copy.deepcopy(self.event.metrics_boundary)


def _fail(message: str) -> None:
    raise ExactTrainingRuntimeError(message)


def _normalize_mode(
    mode: exact.InitializationMode | str,
) -> exact.InitializationMode:
    try:
        normalized = exact.InitializationMode(mode)
    except (TypeError, ValueError):
        _fail(f"unsupported initialization mode: {mode!r}")
    if normalized is exact.InitializationMode.PARENT_WARM_START:
        _fail("parent warm start is outside the exact-training runtime")
    return normalized


class ExactTrainingRuntime:
    """Coordinate exact startup and epoch commits without owning optimization."""

    def __init__(
        self,
        journal: ExactEpochJournal | str | os.PathLike[str],
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        loader_generator: torch.Generator,
        scheduler: Any | None = None,
        map_location: str | torch.device = "cpu",
    ) -> None:
        self.journal = (
            journal
            if isinstance(journal, ExactEpochJournal)
            else ExactEpochJournal(Path(journal))
        )
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.loader_generator = loader_generator
        self.map_location = map_location
        self._snapshot: ExactTrainingSnapshot | None = None
        self._pending: PreparedExactEpoch | None = None

    @property
    def started(self) -> bool:
        return self._snapshot is not None

    @property
    def snapshot(self) -> ExactTrainingSnapshot:
        if self._snapshot is None:
            _fail("runtime has not been started")
        return copy.deepcopy(self._snapshot)

    def startup(
        self,
        mode: exact.InitializationMode | str,
        *,
        run_identity: Mapping[str, Any],
        expected_epoch: int | None = None,
        expected_metrics_boundary: Mapping[str, Any] | None = None,
        expected_best_selection: Mapping[str, Any] | None = None,
    ) -> ExactTrainingSnapshot:
        """Start fresh or restore only after all external expectations agree."""

        if self._snapshot is not None:
            _fail("runtime startup may be called only once")
        normalized_mode = _normalize_mode(mode)
        identity = exact._validate_run_identity(run_identity)
        active = self.journal.load_active()

        if normalized_mode is exact.InitializationMode.FRESH:
            if any(
                value is not None
                for value in (
                    expected_epoch,
                    expected_metrics_boundary,
                    expected_best_selection,
                )
            ):
                _fail("fresh startup does not accept resume expectations")
            if active is not None:
                _fail("fresh startup requires an empty journal")
            snapshot = ExactTrainingSnapshot(
                mode=normalized_mode,
                completed_epoch=0,
                run_identity=identity,
                metrics_boundary=None,
                best_selection=None,
                extra_state={},
                active=None,
            )
        else:
            if active is None:
                _fail("exact-resume startup requires a committed journal")
            if expected_epoch is None:
                _fail("exact-resume startup requires expected_epoch")
            if expected_metrics_boundary is None:
                _fail(
                    "exact-resume startup requires expected_metrics_boundary"
                )
            if expected_best_selection is None:
                _fail("exact-resume startup requires expected_best_selection")
            if (
                not isinstance(expected_epoch, int)
                or isinstance(expected_epoch, bool)
                or expected_epoch < 1
            ):
                _fail("expected_epoch must be a positive integer")
            external_boundary = exact._validate_metrics_boundary(
                expected_metrics_boundary,
                expected_epoch,
            )
            external_best = exact._validate_best_selection(
                expected_best_selection,
                expected_epoch,
            )
            if active.epoch != expected_epoch:
                _fail("journal epoch differs from external expected_epoch")
            if active.metrics_boundary != external_boundary:
                _fail(
                    "journal boundary differs from external "
                    "expected_metrics_boundary"
                )
            restored = exact.restore_exact_resume(
                active.checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=self.scheduler,
                loader_generator=self.loader_generator,
                expected_run_identity=identity,
                expected_epoch=expected_epoch,
                expected_metrics_boundary=external_boundary,
                expected_best_selection=external_best,
                map_location=self.map_location,
            )
            snapshot = ExactTrainingSnapshot(
                mode=normalized_mode,
                completed_epoch=restored.epoch,
                run_identity=restored.run_identity,
                metrics_boundary=restored.metrics_boundary,
                best_selection=restored.best_selection,
                extra_state=restored.extra_state,
                active=active,
            )

        self._snapshot = snapshot
        return self.snapshot

    def prepare_epoch(
        self,
        event: Mapping[str, Any],
        *,
        best_selection: Mapping[str, Any],
        extra_state: Mapping[str, Any] | None = None,
    ) -> PreparedExactEpoch:
        """Capture one exact epoch payload at the caller's completed boundary."""

        if self._snapshot is None:
            _fail("runtime must be started before preparing an epoch")
        if self._pending is not None:
            _fail("the previous prepared epoch must be committed first")
        prepared_event = self.journal.prepare_next_event(event)
        expected_epoch = self._snapshot.completed_epoch + 1
        if prepared_event.epoch != expected_epoch:
            _fail("journal next epoch differs from runtime state")
        payload = exact.build_exact_resume_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.scheduler,
            epoch=prepared_event.epoch,
            run_identity=self._snapshot.run_identity,
            best_selection=best_selection,
            metrics_boundary=prepared_event.metrics_boundary,
            loader_generator=self.loader_generator,
            extra_state=extra_state,
        )
        pending = PreparedExactEpoch(
            event=prepared_event,
            exact_payload=payload,
        )
        self._pending = pending
        return pending

    def commit_epoch(
        self,
        prepared: PreparedExactEpoch,
    ) -> ExactTrainingSnapshot:
        """Commit the pending epoch; failures leave runtime state unadvanced."""

        if self._snapshot is None:
            _fail("runtime must be started before committing an epoch")
        if self._pending is None or prepared is not self._pending:
            _fail("prepared epoch is not this runtime's pending commit")
        payload = prepared.exact_payload
        if payload.get("run_identity") != self._snapshot.run_identity:
            _fail("pending payload run identity differs from runtime")
        if prepared.epoch != self._snapshot.completed_epoch + 1:
            _fail("pending payload epoch is not the next runtime epoch")
        active = self.journal.commit(prepared.event, payload)
        snapshot = ExactTrainingSnapshot(
            mode=self._snapshot.mode,
            completed_epoch=prepared.epoch,
            run_identity=copy.deepcopy(payload["run_identity"]),
            metrics_boundary=copy.deepcopy(payload["metrics_boundary"]),
            best_selection=copy.deepcopy(payload["best_selection"]),
            extra_state=copy.deepcopy(payload["extra_state"]),
            active=active,
        )
        self._snapshot = snapshot
        self._pending = None
        return self.snapshot


__all__ = [
    "ExactTrainingRuntime",
    "ExactTrainingRuntimeError",
    "ExactTrainingSnapshot",
    "PreparedExactEpoch",
]
