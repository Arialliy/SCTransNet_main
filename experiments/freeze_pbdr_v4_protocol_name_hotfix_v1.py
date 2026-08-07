#!/usr/bin/env python3
"""Apply the post-lock PBDR-V4 selected-name contract amendment.

The frozen candidate-pool builder correctly replays the selected calibration
configuration and grid index, but its final metadata assertion compares the
grid-qualified sweep name (``grid-XXX-...``) with the unqualified calibration
name.  The defect was discovered after every formal training artifact had
already been bound to the original source lock and before any official-test
claim.

This runner therefore leaves the locked source bytes untouched.  It derives
the exact one-line corrected freezer in memory, records an immutable protocol
amendment, and adds that amendment binding to every candidate-pool manifest
before the manifest receives its canonical self-hash.  No model state,
calibration value, metric, split, threshold, or evaluator is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCKED_FREEZER_PATH = REPO_ROOT / "experiments/freeze_pbdr_v4_protocol.py"
AMENDMENT_SCHEMA = "sctransnet_pbdr_v4_protocol_amendment/v1"
AMENDMENT_ID = "freeze_selected_grid_name_contract_v1"
LOCKED_FREEZER_SHA256 = (
    "5a2050f935cc57cdc9fb45ef023382377b06352ec94814d5df5267b8f3f3ad91"
)
PATCHED_FREEZER_SHA256 = (
    "6854e46438fd697818927a59e2516ed8915c0ccc6e329f583ae2c1d126522585"
)
LOCKED_SOURCE_LOCK_FILE_SHA256 = (
    "0d3b0b26a482fcae3d1701205c45d83153828c03512367dde90c96c1a6d8edd8"
)
LOCKED_SOURCE_LOCK_SEMANTIC_SHA256 = (
    "96af51690eb9270f76a2a37cbb778ede45f57dcf6fb36e2eba357dfacdef8ba6"
)
OLD_LITERAL = 'and selected.get("name") == calibration.name,'
NEW_LITERAL = (
    'and selected.get("name") '
    '== f"grid-{grid_index:03d}-{calibration.name}",'
)


class PBDRV4FreezeNameHotfixError(RuntimeError):
    """The locked source, amendment, or derived freezer differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4FreezeNameHotfixError(message)


def _file_sha256(path: Path) -> str:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"required regular file is missing or unsafe: {candidate}",
    )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4FreezeNameHotfixError(
            f"amendment is not canonical JSON: {error}"
        ) from error
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _derive_patched_freezer_source() -> str:
    _require(
        _file_sha256(LOCKED_FREEZER_PATH) == LOCKED_FREEZER_SHA256,
        "locked freezer bytes differ",
    )
    try:
        source = LOCKED_FREEZER_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PBDRV4FreezeNameHotfixError(
            f"cannot read locked freezer: {error}"
        ) from error
    _require(
        source.count(OLD_LITERAL) == 1,
        "locked freezer defect literal must occur exactly once",
    )
    patched = source.replace(OLD_LITERAL, NEW_LITERAL)
    _require(
        hashlib.sha256(patched.encode("utf-8")).hexdigest()
        == PATCHED_FREEZER_SHA256,
        "derived patched freezer SHA-256 differs",
    )
    return patched


def _validated_source_lock(
    source_lock_path: Path,
    *,
    check_environment: bool,
) -> tuple[dict[str, object], Path]:
    from experiments import pbdr_v4_source_lock as source_lock_io
    from experiments import pbdr_v4_training_core as training_core

    training_core.configure_determinism(seed=training_core.TRAINING_SEED)
    path = Path(source_lock_path)
    _require(
        path.is_file() and not path.is_symlink(),
        "source lock is missing or unsafe",
    )
    resolved = path.resolve(strict=True)
    _require(
        _file_sha256(resolved) == LOCKED_SOURCE_LOCK_FILE_SHA256,
        "source-lock file SHA-256 differs",
    )
    payload = source_lock_io.load_source_lock(
        resolved,
        check_environment=check_environment,
    )
    _require(
        payload.get("source_lock_sha256")
        == LOCKED_SOURCE_LOCK_SEMANTIC_SHA256,
        "source-lock semantic SHA-256 differs",
    )
    sources = payload.get("sources")
    record = (
        sources.get("experiments/freeze_pbdr_v4_protocol.py")
        if isinstance(sources, Mapping)
        else None
    )
    _require(
        isinstance(record, Mapping)
        and record.get("sha256") == LOCKED_FREEZER_SHA256
        and record.get("path") == str(LOCKED_FREEZER_PATH),
        "source lock does not bind the expected frozen freezer",
    )
    _derive_patched_freezer_source()
    return dict(payload), resolved


def build_amendment_manifest(
    *,
    source_lock_path: Path,
    check_environment: bool = True,
) -> dict[str, object]:
    source_lock, resolved_lock = _validated_source_lock(
        source_lock_path,
        check_environment=check_environment,
    )
    runner_path = Path(__file__).resolve(strict=True)
    payload: dict[str, object] = {
        "schema": AMENDMENT_SCHEMA,
        "status": "frozen_before_official_claim",
        "amendment_id": AMENDMENT_ID,
        "scope": "candidate_pool_freeze_only",
        "defect": (
            "selected sweep names are grid-qualified, while the locked freezer "
            "compares against an unqualified calibration name"
        ),
        "locked_source_lock": {
            "path": str(resolved_lock),
            "file_sha256": LOCKED_SOURCE_LOCK_FILE_SHA256,
            "semantic_sha256": source_lock["source_lock_sha256"],
        },
        "locked_freezer": {
            "path": str(LOCKED_FREEZER_PATH),
            "file_sha256": LOCKED_FREEZER_SHA256,
        },
        "derived_patched_freezer": {
            "file_sha256": PATCHED_FREEZER_SHA256,
            "old_literal": OLD_LITERAL,
            "new_literal": NEW_LITERAL,
            "replacement_occurrences": 1,
        },
        "runner": {
            "path": str(runner_path),
            "file_sha256": _file_sha256(runner_path),
        },
        "invariants": {
            "training_artifacts_changed": False,
            "model_states_changed": False,
            "calibration_values_changed": False,
            "metrics_changed": False,
            "split_changed": False,
            "evaluator_changed": False,
            "fixed_probability_rule": "strict_greater_than_0.5",
            "performance_acceptance_margin": None,
        },
        "official_test_accessed": False,
        "official_probability_or_logit_cache_written": False,
        "official_sweep_performed": False,
    }
    payload["amendment_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_manifest_payload(
    payload: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> dict[str, object]:
    _require(
        payload.get("schema") == AMENDMENT_SCHEMA
        and payload.get("status") == "frozen_before_official_claim"
        and payload.get("amendment_id") == AMENDMENT_ID
        and payload.get("scope") == "candidate_pool_freeze_only",
        "protocol amendment identity/status differs",
    )
    declared = payload.get("amendment_sha256")
    unsigned = dict(payload)
    unsigned.pop("amendment_sha256", None)
    _require(
        declared == _canonical_sha256(unsigned),
        "protocol amendment canonical SHA-256 differs",
    )
    _require(
        _canonical_json_bytes(payload) == _canonical_json_bytes(expected),
        "protocol amendment differs from the live deterministic derivation",
    )
    return dict(payload)


def read_amendment_manifest(
    path: Path,
    *,
    source_lock_path: Path,
    check_environment: bool = True,
) -> dict[str, object]:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        "protocol amendment manifest is missing or unsafe",
    )
    try:
        raw = candidate.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4FreezeNameHotfixError(
            f"cannot read protocol amendment: {error}"
        ) from error
    _require(isinstance(payload, Mapping), "protocol amendment must be an object")
    _require(
        raw == _canonical_json_bytes(payload, newline=True),
        "protocol amendment is not canonical JSON",
    )
    expected = build_amendment_manifest(
        source_lock_path=source_lock_path,
        check_environment=check_environment,
    )
    return _validate_manifest_payload(payload, expected=expected)


def write_amendment_manifest_exclusive(
    path: Path,
    *,
    source_lock_path: Path,
    check_environment: bool = True,
) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        read_amendment_manifest(
            destination,
            source_lock_path=source_lock_path,
            check_environment=check_environment,
        )
        return destination.resolve(strict=True)
    payload = build_amendment_manifest(
        source_lock_path=source_lock_path,
        check_environment=check_environment,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not destination.parent.is_symlink(),
        "protocol amendment parent is a symlink",
    )
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload, newline=True))
        handle.flush()
        os.fsync(handle.fileno())
    return destination.resolve(strict=True)


def _amendment_binding(
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    locked_source = manifest.get("locked_source_lock")
    patched = manifest.get("derived_patched_freezer")
    runner = manifest.get("runner")
    _require(
        isinstance(locked_source, Mapping)
        and isinstance(patched, Mapping)
        and isinstance(runner, Mapping),
        "protocol amendment binding records differ",
    )
    path = Path(manifest_path).resolve(strict=True)
    return {
        "schema": AMENDMENT_SCHEMA,
        "amendment_id": AMENDMENT_ID,
        "scope": "candidate_pool_freeze_only",
        "manifest_path": str(path),
        "manifest_file_sha256": _file_sha256(path),
        "amendment_sha256": manifest["amendment_sha256"],
        "base_source_lock_sha256": locked_source["semantic_sha256"],
        "patched_freezer_source_sha256": patched["file_sha256"],
        "runner_source_sha256": runner["file_sha256"],
        "official_test_accessed": False,
    }


class _PoolModuleProxy:
    def __init__(self, base: Any, binding: Mapping[str, object]) -> None:
        self._base = base
        self._binding = dict(binding)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def build_candidate_pool(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        payload = dict(self._base.build_candidate_pool(*args, **kwargs))
        _require(
            "protocol_amendment_binding" not in payload,
            "candidate pool already carries a protocol amendment",
        )
        payload.pop("candidate_pool_sha256", None)
        payload["protocol_amendment_binding"] = dict(self._binding)
        payload["candidate_pool_sha256"] = self._base.canonical_sha256(payload)
        return self._base.validate_candidate_pool(payload)


def _load_patched_freezer(
    *,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> types.ModuleType:
    source = _derive_patched_freezer_source()
    runtime = types.ModuleType("experiments.freeze_pbdr_v4_protocol_hotfix_runtime")
    runtime.__file__ = str(LOCKED_FREEZER_PATH)
    runtime.__package__ = "experiments"
    sys.modules[runtime.__name__] = runtime
    try:
        exec(
            compile(source, str(LOCKED_FREEZER_PATH), "exec"),
            runtime.__dict__,
        )
    except BaseException:
        sys.modules.pop(runtime.__name__, None)
        raise
    runtime.pool_io = _PoolModuleProxy(
        runtime.pool_io,
        _amendment_binding(manifest_path, manifest),
    )
    return runtime


def _extract_required_option(
    argv: Sequence[str],
    option: str,
) -> tuple[str, tuple[str, ...]]:
    values = list(argv)
    _require(values.count(option) == 1, f"{option} must be supplied exactly once")
    index = values.index(option)
    _require(index + 1 < len(values), f"{option} value is missing")
    value = values[index + 1]
    del values[index : index + 2]
    return value, tuple(values)


def _run_freeze_pool(argv: Sequence[str]) -> int:
    manifest_raw, forwarded = _extract_required_option(argv, "--hotfix-manifest")
    source_lock_raw, _ = _extract_required_option(forwarded, "--source-lock")
    manifest_path = Path(manifest_raw)
    source_lock_path = Path(source_lock_raw)
    manifest = read_amendment_manifest(
        manifest_path,
        source_lock_path=source_lock_path,
        check_environment=True,
    )
    runtime = _load_patched_freezer(
        manifest_path=manifest_path,
        manifest=manifest,
    )
    return int(runtime.main(forwarded))


def parse_record_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="record the PBDR-V4 freeze amendment")
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ready = tuple(sys.argv[1:] if argv is None else argv)
    _require(bool(ready), "a hotfix command is required")
    if ready[0] == "record-hotfix":
        arguments = parse_record_args(ready[1:])
        destination = write_amendment_manifest_exclusive(
            arguments.output,
            source_lock_path=arguments.source_lock,
            check_environment=True,
        )
        print(destination)
        return 0
    _require(ready[0] == "freeze-pool", "only freeze-pool is permitted")
    return _run_freeze_pool(ready)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMENDMENT_ID",
    "AMENDMENT_SCHEMA",
    "PBDRV4FreezeNameHotfixError",
    "build_amendment_manifest",
    "main",
    "read_amendment_manifest",
    "write_amendment_manifest_exclusive",
]
