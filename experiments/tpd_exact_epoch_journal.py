"""Two-slot, crash-consistent epoch commits for exact TPD training.

The journal owns exactly five files: one metrics/checkpoint pair for each of
``slot_a`` and ``slot_b``, plus a single active marker.  A commit always
rewrites the inactive pair and atomically replaces the marker last, so a
failed commit cannot invalidate the previously active epoch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments import tpd_exact_resume as exact


JOURNAL_SCHEMA = "sctransnet_tpd_exact_epoch_journal_v1"
MARKER_FILENAME = "active.json"
SLOTS = ("slot_a", "slot_b")
SLOT_FILES = {
    "slot_a": ("slot_a.metrics.jsonl", "slot_a.exact.pth"),
    "slot_b": ("slot_b.metrics.jsonl", "slot_b.exact.pth"),
}
MARKER_KEYS = frozenset(
    {
        "schema",
        "active_slot",
        "epoch",
        "metrics_file",
        "checkpoint_file",
        "checkpoint_sha256",
        "metrics_sha256",
        "metrics_byte_count",
        "last_event_sha256",
        "metrics_boundary",
    }
)
BOUNDARY_KEYS = frozenset(
    {
        "completed_epoch",
        "event_count",
        "last_event_epoch",
        "metrics_sha256",
        "last_event_sha256",
    }
)


class ExactEpochJournalError(ValueError):
    """The journal or a proposed commit violates its exact-epoch contract."""


@dataclass(frozen=True)
class PreparedEpochEvent:
    """Immutable material prepared for one prospective epoch commit."""

    journal_root: Path
    base_marker_sha256: str | None
    previous_epoch: int
    previous_slot: str | None
    target_slot: str
    epoch: int
    event_bytes: bytes
    metrics_bytes: bytes
    metrics_boundary: dict[str, Any]


@dataclass(frozen=True)
class ActiveEpochState:
    """Validated active journal state."""

    slot: str
    epoch: int
    metrics_path: Path
    checkpoint_path: Path
    marker_path: Path
    marker_sha256: str
    checkpoint_sha256: str
    metrics_byte_count: int
    metrics_boundary: dict[str, Any]


def _fail(message: str) -> None:
    raise ExactEpochJournalError(message)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plain_json(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                _fail(f"{label} keys must be non-empty strings")
            normalized[key] = _plain_json(item, f"{label}.{key}")
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _plain_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(f"{label} contains a non-JSON value")


def _canonical_json(value: Any, label: str) -> bytes:
    return json.dumps(
        _plain_json(value, label),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parse_json(content: bytes, label: str) -> Any:
    try:
        text = content.decode("utf-8")
        return json.loads(
            text,
            parse_constant=lambda value: _fail(
                f"{label} contains non-finite constant {value}"
            ),
        )
    except ExactEpochJournalError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{label} is invalid JSON: {exc}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    if destination.is_symlink():
        _fail(f"refusing to replace symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink():
        _fail(f"{label} must not be a symlink")
    try:
        metadata = path.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} is not a regular file: {path}")
    return path.read_bytes()


def _validate_boundary(value: Any, epoch: int) -> dict[str, Any]:
    boundary = _plain_json(value, "metrics_boundary")
    if not isinstance(boundary, dict) or set(boundary) != BOUNDARY_KEYS:
        _fail("metrics_boundary has an invalid schema")
    for key in ("completed_epoch", "event_count", "last_event_epoch"):
        item = boundary[key]
        if not isinstance(item, int) or isinstance(item, bool):
            _fail(f"metrics_boundary.{key} must be an integer")
    if (
        boundary["completed_epoch"] != epoch
        or boundary["event_count"] != epoch
        or boundary["last_event_epoch"] != epoch
    ):
        _fail("metrics_boundary does not match epoch")
    _validate_digest(boundary["metrics_sha256"], "metrics_boundary.metrics_sha256")
    _validate_digest(
        boundary["last_event_sha256"],
        "metrics_boundary.last_event_sha256",
    )
    return boundary


def _validate_metrics(content: bytes, epoch: int) -> dict[str, Any]:
    if not content or not content.endswith(b"\n"):
        _fail("metrics JSONL must be non-empty and newline terminated")
    lines = content.splitlines(keepends=True)
    if len(lines) != epoch:
        _fail("metrics JSONL event count does not match epoch")
    for line_number, line in enumerate(lines, start=1):
        if not line.endswith(b"\n") or line == b"\n":
            _fail("metrics JSONL contains an incomplete or empty line")
        row = _parse_json(line[:-1], f"metrics line {line_number}")
        if not isinstance(row, dict):
            _fail(f"metrics line {line_number} must be an object")
        if row.get("epoch") != line_number:
            _fail("metrics JSONL epochs are not contiguous")
        if _canonical_json(row, f"metrics line {line_number}") + b"\n" != line:
            _fail("metrics JSONL is not canonical")
    return {
        "completed_epoch": epoch,
        "event_count": epoch,
        "last_event_epoch": epoch,
        "metrics_sha256": _sha256(content),
        "last_event_sha256": _sha256(lines[-1]),
    }


def _validate_exact_payload(
    payload: Any,
    *,
    epoch: int,
    boundary: Mapping[str, Any],
) -> None:
    if not isinstance(payload, Mapping):
        _fail("exact_payload must be a mapping")
    if set(payload) != exact.EXACT_RESUME_REQUIRED_KEYS:
        _fail("exact_payload has an invalid exact-resume schema")
    if payload.get("schema") != exact.EXACT_RESUME_SCHEMA:
        _fail("exact_payload schema mismatch")
    if payload.get("mode") != exact.EXACT_RESUME_MODE:
        _fail("exact_payload mode mismatch")
    payload_epoch = payload.get("epoch")
    if (
        not isinstance(payload_epoch, int)
        or isinstance(payload_epoch, bool)
        or payload_epoch != epoch
    ):
        _fail("exact_payload epoch mismatch")
    payload_boundary = _validate_boundary(payload.get("metrics_boundary"), epoch)
    if payload_boundary != dict(boundary):
        _fail("exact_payload metrics_boundary mismatch")


class ExactEpochJournal:
    """Fixed-path A/B epoch journal."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).absolute()
        for component in (self.root, *self.root.parents):
            if component.exists() and component.is_symlink():
                _fail(f"journal path contains a symlink: {component}")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            _fail("journal root must be a real directory")
        self.marker_path = self.root / MARKER_FILENAME
        self._runtime_cache_initialized = False
        self._runtime_active_cache: ActiveEpochState | None = None

    def _slot_paths(self, slot: str) -> tuple[Path, Path]:
        if slot not in SLOTS:
            _fail("invalid journal slot")
        metrics_name, checkpoint_name = SLOT_FILES[slot]
        metrics_path = self.root / metrics_name
        checkpoint_path = self.root / checkpoint_name
        if metrics_path.parent != self.root or checkpoint_path.parent != self.root:
            _fail("journal path escapes its root")
        return metrics_path, checkpoint_path

    def load_active(self) -> ActiveEpochState | None:
        """Load and fully verify the marker-selected slot."""

        if not self.marker_path.exists():
            if self.marker_path.is_symlink():
                _fail("active marker must not be a symlink")
            self._runtime_cache_initialized = True
            self._runtime_active_cache = None
            return None
        marker_bytes = _read_regular(self.marker_path, "active marker")
        if not marker_bytes.endswith(b"\n"):
            _fail("active marker must be newline terminated")
        marker = _parse_json(marker_bytes[:-1], "active marker")
        if not isinstance(marker, dict) or set(marker) != MARKER_KEYS:
            _fail("active marker has an invalid schema")
        if _canonical_json(marker, "active marker") + b"\n" != marker_bytes:
            _fail("active marker is not canonical")
        if marker["schema"] != JOURNAL_SCHEMA:
            _fail("active marker schema mismatch")
        slot = marker["active_slot"]
        if slot not in SLOTS:
            _fail("active marker slot is invalid")
        epoch = marker["epoch"]
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
            _fail("active marker epoch must be a positive integer")
        metrics_path, checkpoint_path = self._slot_paths(slot)
        expected_metrics_name, expected_checkpoint_name = SLOT_FILES[slot]
        if marker["metrics_file"] != expected_metrics_name:
            _fail("active marker metrics path is not the fixed slot path")
        if marker["checkpoint_file"] != expected_checkpoint_name:
            _fail("active marker checkpoint path is not the fixed slot path")

        metrics_bytes = _read_regular(metrics_path, "active metrics")
        checkpoint_bytes = _read_regular(checkpoint_path, "active checkpoint")
        boundary = _validate_metrics(metrics_bytes, epoch)
        marker_boundary = _validate_boundary(marker["metrics_boundary"], epoch)
        checkpoint_sha256 = _sha256(checkpoint_bytes)
        if marker["metrics_byte_count"] != len(metrics_bytes):
            _fail("active metrics byte count mismatch")
        if marker["metrics_sha256"] != boundary["metrics_sha256"]:
            _fail("active metrics SHA-256 mismatch")
        if marker["last_event_sha256"] != boundary["last_event_sha256"]:
            _fail("active last-event SHA-256 mismatch")
        if marker_boundary != boundary:
            _fail("active metrics boundary mismatch")
        if marker["checkpoint_sha256"] != checkpoint_sha256:
            _fail("active checkpoint SHA-256 mismatch")
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            _fail(f"active checkpoint cannot be loaded: {exc}")
        _validate_exact_payload(payload, epoch=epoch, boundary=boundary)
        state = ActiveEpochState(
            slot=slot,
            epoch=epoch,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            marker_path=self.marker_path,
            marker_sha256=_sha256(marker_bytes),
            checkpoint_sha256=checkpoint_sha256,
            metrics_byte_count=len(metrics_bytes),
            metrics_boundary=boundary,
        )
        self._runtime_cache_initialized = True
        self._runtime_active_cache = state
        return state

    def _load_runtime_active(self) -> ActiveEpochState | None:
        """Use a verified cache while the atomic marker remains unchanged."""

        if not self._runtime_cache_initialized:
            return self.load_active()
        cached = self._runtime_active_cache
        if not self.marker_path.exists():
            if self.marker_path.is_symlink():
                _fail("active marker must not be a symlink")
            if cached is None:
                return None
            _fail("active marker disappeared from a committed runtime journal")
        marker_bytes = _read_regular(self.marker_path, "active marker")
        marker_sha256 = _sha256(marker_bytes)
        if cached is not None and marker_sha256 == cached.marker_sha256:
            return cached
        return self.load_active()

    def prepare_next_event(self, event: Mapping[str, Any]) -> PreparedEpochEvent:
        """Canonicalize the next contiguous event without writing any file."""

        if not isinstance(event, Mapping):
            _fail("event must be a mapping")
        active = self._load_runtime_active()
        previous_epoch = 0 if active is None else active.epoch
        previous_slot = None if active is None else active.slot
        expected_epoch = previous_epoch + 1
        normalized = _plain_json(event, "event")
        if not isinstance(normalized, dict):
            _fail("event must be a mapping")
        if normalized.get("epoch") != expected_epoch:
            _fail(f"event epoch must be exactly {expected_epoch}")
        event_bytes = _canonical_json(normalized, "event") + b"\n"
        prefix = (
            b""
            if active is None
            else _read_regular(active.metrics_path, "active metrics")
        )
        if (
            active is not None
            and _sha256(prefix) != active.metrics_boundary["metrics_sha256"]
        ):
            _fail("cached active metrics SHA-256 mismatch")
        metrics_bytes = prefix + event_bytes
        boundary = _validate_metrics(metrics_bytes, expected_epoch)
        target_slot = "slot_a" if previous_slot in (None, "slot_b") else "slot_b"
        return PreparedEpochEvent(
            journal_root=self.root,
            base_marker_sha256=None if active is None else active.marker_sha256,
            previous_epoch=previous_epoch,
            previous_slot=previous_slot,
            target_slot=target_slot,
            epoch=expected_epoch,
            event_bytes=event_bytes,
            metrics_bytes=metrics_bytes,
            metrics_boundary=boundary,
        )

    def commit(
        self,
        prepared: PreparedEpochEvent,
        exact_payload: Mapping[str, Any],
    ) -> ActiveEpochState:
        """Durably write the inactive pair, then atomically switch the marker."""

        if not isinstance(prepared, PreparedEpochEvent):
            _fail("prepared must be a PreparedEpochEvent")
        if prepared.journal_root != self.root:
            _fail("prepared event belongs to a different journal")
        active = self._load_runtime_active()
        marker_sha = None if active is None else active.marker_sha256
        active_epoch = 0 if active is None else active.epoch
        active_slot = None if active is None else active.slot
        if (
            marker_sha != prepared.base_marker_sha256
            or active_epoch != prepared.previous_epoch
            or active_slot != prepared.previous_slot
        ):
            _fail("prepared event is stale")
        expected_slot = "slot_a" if active_slot in (None, "slot_b") else "slot_b"
        if prepared.target_slot != expected_slot or prepared.epoch != active_epoch + 1:
            _fail("prepared event does not target the inactive next slot")
        if _validate_metrics(prepared.metrics_bytes, prepared.epoch) != prepared.metrics_boundary:
            _fail("prepared metrics bytes and boundary differ")
        _validate_exact_payload(
            exact_payload,
            epoch=prepared.epoch,
            boundary=prepared.metrics_boundary,
        )

        metrics_path, checkpoint_path = self._slot_paths(prepared.target_slot)
        _atomic_write_bytes(metrics_path, prepared.metrics_bytes)
        exact.atomic_torch_save(exact_payload, checkpoint_path)
        checkpoint_bytes = _read_regular(checkpoint_path, "inactive checkpoint")
        marker = {
            "schema": JOURNAL_SCHEMA,
            "active_slot": prepared.target_slot,
            "epoch": prepared.epoch,
            "metrics_file": metrics_path.name,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": _sha256(checkpoint_bytes),
            "metrics_sha256": prepared.metrics_boundary["metrics_sha256"],
            "metrics_byte_count": len(prepared.metrics_bytes),
            "last_event_sha256": prepared.metrics_boundary[
                "last_event_sha256"
            ],
            "metrics_boundary": prepared.metrics_boundary,
        }
        marker_bytes = _canonical_json(marker, "active marker") + b"\n"
        _atomic_write_bytes(self.marker_path, marker_bytes)
        committed = ActiveEpochState(
            slot=prepared.target_slot,
            epoch=prepared.epoch,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            marker_path=self.marker_path,
            marker_sha256=_sha256(marker_bytes),
            checkpoint_sha256=marker["checkpoint_sha256"],
            metrics_byte_count=len(prepared.metrics_bytes),
            metrics_boundary=dict(prepared.metrics_boundary),
        )
        self._runtime_cache_initialized = True
        self._runtime_active_cache = committed
        return committed


__all__ = [
    "ActiveEpochState",
    "ExactEpochJournal",
    "ExactEpochJournalError",
    "JOURNAL_SCHEMA",
    "MARKER_FILENAME",
    "PreparedEpochEvent",
    "SLOT_FILES",
]
