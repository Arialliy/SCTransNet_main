"""Source and environment lock shared by every formal PBDR-V4 stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK_SCHEMA = "sctransnet_pbdr_v4_source_lock/v1"
DEFAULT_SOURCE_RELATIVE_PATHS = (
    "experiments/PBDR_V4_PROTOCOL.md",
    "experiments/build_pbdr_v4_component_atlas.py",
    "experiments/component_matching_v2.py",
    "experiments/evaluate_three_dataset_pbdr_v4_v1.py",
    "experiments/freeze_pbdr_v4_protocol.py",
    "experiments/launch_three_dataset_pbdr_v4_v1.py",
    "experiments/pbdr_v3_residual_calibration.py",
    "experiments/pbdr_v4_atlas_dataset.py",
    "experiments/pbdr_v4_candidate_pool.py",
    "experiments/pbdr_v4_component_atlas.py",
    "experiments/pbdr_v4_component_loss.py",
    "experiments/pbdr_v4_internal_cache.py",
    "experiments/pbdr_v4_internal_dataset.py",
    "experiments/pbdr_v4_metric_core.py",
    "experiments/pbdr_v4_models_seed42_v1.py",
    "experiments/pbdr_v4_official_once.py",
    "experiments/pbdr_v4_original_models.py",
    "experiments/pbdr_v4_run_artifacts.py",
    "experiments/pbdr_v4_split_authority.py",
    "experiments/pbdr_v4_state_contract.py",
    "experiments/pbdr_v4_training_core.py",
    "experiments/pbdr_v4_zero_margin_selector.py",
    "experiments/pbdr_v4_source_lock.py",
    "experiments/prepare_pbdr_v4_internal_artifacts.py",
    "experiments/sweep_pbdr_v3_residual_calibration.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/train_three_dataset_pbdr_v4_v1.py",
    "model/tpd_role_aligned_residual_calibrator_v4.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v4.py",
)


class PBDRV4SourceLockError(RuntimeError):
    """A source, environment, or external artifact differs from its lock."""


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        content = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4SourceLockError(f"value cannot be canonical JSON: {error}") from error
    return content + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4SourceLockError(f"locked file is missing or unsafe: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PBDRV4SourceLockError("locked relative path is empty")
    rel = Path(relative)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise PBDRV4SourceLockError(f"locked path is unsafe: {relative!r}")
    root_ready = Path(root).resolve(strict=True)
    cursor = root_ready
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PBDRV4SourceLockError(f"locked path traverses a symlink: {relative}")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root_ready)
    except (FileNotFoundError, ValueError) as error:
        raise PBDRV4SourceLockError(f"locked path is missing/outside root: {relative}") from error
    if not resolved.is_file():
        raise PBDRV4SourceLockError(f"locked path is not a file: {relative}")
    return resolved


def environment_attestation() -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    cudnn_version = torch.backends.cudnn.version()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn_version,
        "cuda_available": cuda_available,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def require_formal_runtime_controls() -> dict[str, object]:
    environment = environment_attestation()
    required = {
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    differing = {
        name: {"expected": expected, "observed": environment[name]}
        for name, expected in required.items()
        if environment[name] != expected
    }
    if differing:
        raise PBDRV4SourceLockError(f"formal runtime controls differ: {differing}")
    return environment


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build_source_lock(
    *,
    source_root: Path = REPO_ROOT,
    source_relative_paths: Sequence[str] = DEFAULT_SOURCE_RELATIVE_PATHS,
    external_files: Mapping[str, Path] | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = Path(source_root).resolve(strict=True)
    paths = list(source_relative_paths)
    if not paths or len(paths) != len(set(paths)):
        raise PBDRV4SourceLockError("source path list is empty or contains duplicates")
    sources = {
        relative: _file_record(_safe_file(root, relative))
        for relative in sorted(paths)
    }
    external: dict[str, object] = {}
    for name, raw_path in sorted((external_files or {}).items()):
        if not isinstance(name, str) or not name:
            raise PBDRV4SourceLockError("external binding name is invalid")
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise PBDRV4SourceLockError(f"external file is missing or unsafe: {path}")
        external[name] = _file_record(path.resolve(strict=True))
    ready_environment = dict(
        environment if environment is not None else require_formal_runtime_controls()
    )
    payload: dict[str, object] = {
        "schema": SOURCE_LOCK_SCHEMA,
        "status": "frozen",
        "source_root": str(root),
        "sources": sources,
        "external_files": external,
        "environment": ready_environment,
        "tf32_disabled": (
            ready_environment.get("cuda_matmul_allow_tf32") is False
            and ready_environment.get("cudnn_allow_tf32") is False
        ),
        "official_test_accessed": False,
    }
    payload["source_lock_sha256"] = canonical_sha256(payload)
    return payload


def write_source_lock_exclusive(path: Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise PBDRV4SourceLockError(f"source lock already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise PBDRV4SourceLockError("source-lock parent is a symlink")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(dict(payload), newline=True))
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def validate_source_lock(
    payload: Mapping[str, object],
    *,
    check_environment: bool = True,
) -> dict[str, object]:
    if payload.get("schema") != SOURCE_LOCK_SCHEMA or payload.get("status") != "frozen":
        raise PBDRV4SourceLockError("source-lock identity/status differs")
    declared = payload.get("source_lock_sha256")
    unsigned = dict(payload)
    unsigned.pop("source_lock_sha256", None)
    if declared != canonical_sha256(unsigned):
        raise PBDRV4SourceLockError("source-lock canonical SHA-256 differs")
    for section_name in ("sources", "external_files"):
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            raise PBDRV4SourceLockError(f"source-lock {section_name} differs")
        for name, raw_record in section.items():
            if not isinstance(raw_record, Mapping):
                raise PBDRV4SourceLockError(f"locked record is malformed: {name}")
            path = Path(str(raw_record.get("path")))
            if path.is_symlink() or not path.is_file():
                raise PBDRV4SourceLockError(f"locked file is missing or unsafe: {name}")
            if path.stat().st_size != raw_record.get("bytes") or file_sha256(path) != raw_record.get("sha256"):
                raise PBDRV4SourceLockError(f"locked file bytes differ: {name}")
    if payload.get("tf32_disabled") is not True or payload.get("official_test_accessed") is not False:
        raise PBDRV4SourceLockError("source-lock scope/TF32 attestation differs")
    if check_environment and payload.get("environment") != environment_attestation():
        raise PBDRV4SourceLockError("runtime environment differs from source lock")
    return dict(payload)


def load_source_lock(path: Path, *, check_environment: bool = True) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4SourceLockError("source-lock file is missing or unsafe")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4SourceLockError(f"cannot read source lock: {error}") from error
    if not isinstance(value, Mapping):
        raise PBDRV4SourceLockError("source lock must contain an object")
    return validate_source_lock(value, check_environment=check_environment)


__all__ = [
    "DEFAULT_SOURCE_RELATIVE_PATHS",
    "PBDRV4SourceLockError",
    "REPO_ROOT",
    "SOURCE_LOCK_SCHEMA",
    "build_source_lock",
    "canonical_sha256",
    "environment_attestation",
    "file_sha256",
    "load_source_lock",
    "require_formal_runtime_controls",
    "validate_source_lock",
    "write_source_lock_exclusive",
]
