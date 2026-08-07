"""Immutable five-family candidate-pool manifest for PBDR-V4 evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.pbdr_v4_zero_margin_selector import FROZEN_TIE_ORDER


CANDIDATE_POOL_SCHEMA = "sctransnet_pbdr_v4_candidate_pool/v1"


class PBDRV4CandidatePoolError(RuntimeError):
    """A candidate or candidate-pool binding is missing, mutable, or invalid."""


def _require_sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PBDRV4CandidatePoolError(f"{name} must be lowercase SHA-256")
    return value


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4CandidatePoolError(f"candidate file is missing or unsafe: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4CandidatePoolError(f"candidate manifest is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    family: str
    name: str
    kind: str
    artifact_path: str
    artifact_sha256: str
    state_sha256: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if self.family not in FROZEN_TIE_ORDER:
            raise PBDRV4CandidatePoolError(f"unsupported family: {self.family!r}")
        if not isinstance(self.name, str) or not self.name:
            raise PBDRV4CandidatePoolError("candidate name is empty")
        if self.kind not in (
            "original_checkpoint",
            "current_checkpoint",
            "v3_residual_calibration",
            "v4_stage1_checkpoint",
            "v4_stage2_checkpoint",
        ):
            raise PBDRV4CandidatePoolError(f"candidate kind differs: {self.kind!r}")
        path = Path(self.artifact_path)
        if not path.is_absolute():
            raise PBDRV4CandidatePoolError("candidate artifact path must be absolute")
        _require_sha(self.artifact_sha256, name="artifact_sha256")
        _require_sha(self.state_sha256, name="state_sha256")
        _require_sha(self.configuration_sha256, name="configuration_sha256")

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


_EXPECTED_KIND = {
    "Original": "original_checkpoint",
    "Current": "current_checkpoint",
    "V3-calibrated": "v3_residual_calibration",
    "V4-Stage1": "v4_stage1_checkpoint",
    "V4-Stage2": "v4_stage2_checkpoint",
}


def _validate_artifacts(candidates: Sequence[CandidateArtifact]) -> None:
    if tuple(item.family for item in candidates) != tuple(FROZEN_TIE_ORDER):
        raise PBDRV4CandidatePoolError("candidate families/order differ from frozen pool")
    if len({item.name for item in candidates}) != len(candidates):
        raise PBDRV4CandidatePoolError("candidate names are not unique")
    for item in candidates:
        if item.kind != _EXPECTED_KIND[item.family]:
            raise PBDRV4CandidatePoolError(
                f"candidate family/kind mismatch: {item.family}/{item.kind}"
            )
        path = Path(item.artifact_path)
        if path.is_symlink() or not path.is_file():
            raise PBDRV4CandidatePoolError(f"candidate artifact is missing: {item.family}")
        if path.resolve(strict=True) != path:
            raise PBDRV4CandidatePoolError(f"candidate path is not canonical: {item.family}")
        if file_sha256(path) != item.artifact_sha256:
            raise PBDRV4CandidatePoolError(f"candidate artifact SHA differs: {item.family}")


def build_candidate_pool(
    *,
    dataset: str,
    role: str,
    source_lock_sha256: str,
    split_projection_sha256: str,
    candidates: Sequence[CandidateArtifact],
) -> dict[str, object]:
    if dataset not in ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"):
        raise PBDRV4CandidatePoolError("unsupported dataset")
    if role not in ("best_miou", "best_pd"):
        raise PBDRV4CandidatePoolError("unsupported role")
    _require_sha(source_lock_sha256, name="source_lock_sha256")
    _require_sha(split_projection_sha256, name="split_projection_sha256")
    ready = tuple(candidates)
    _validate_artifacts(ready)
    payload: dict[str, object] = {
        "schema": CANDIDATE_POOL_SCHEMA,
        "status": "frozen_before_official_claim",
        "dataset": dataset,
        "role": role,
        "fixed_probability_rule": "strict_greater_than_0.5",
        "performance_acceptance_margin": None,
        "source_lock_sha256": source_lock_sha256,
        "split_projection_sha256": split_projection_sha256,
        "family_order": list(FROZEN_TIE_ORDER),
        "candidate_count": len(ready),
        "candidates": [item.as_dict() for item in ready],
        "official_test_accessed": False,
    }
    payload["candidate_pool_sha256"] = canonical_sha256(payload)
    return payload


def validate_candidate_pool(payload: Mapping[str, object]) -> dict[str, object]:
    if payload.get("schema") != CANDIDATE_POOL_SCHEMA or payload.get("status") != "frozen_before_official_claim":
        raise PBDRV4CandidatePoolError("candidate-pool identity/status differs")
    if payload.get("official_test_accessed") is not False:
        raise PBDRV4CandidatePoolError("candidate pool crossed official boundary")
    if payload.get("dataset") not in ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"):
        raise PBDRV4CandidatePoolError("candidate-pool dataset differs")
    if payload.get("role") not in ("best_miou", "best_pd"):
        raise PBDRV4CandidatePoolError("candidate-pool role differs")
    if payload.get("fixed_probability_rule") != "strict_greater_than_0.5":
        raise PBDRV4CandidatePoolError("candidate-pool probability rule differs")
    if payload.get("performance_acceptance_margin") is not None:
        raise PBDRV4CandidatePoolError("candidate pool carries an acceptance margin")
    _require_sha(payload.get("source_lock_sha256"), name="source_lock_sha256")
    _require_sha(
        payload.get("split_projection_sha256"),
        name="split_projection_sha256",
    )
    declared = payload.get("candidate_pool_sha256")
    unsigned = dict(payload)
    unsigned.pop("candidate_pool_sha256", None)
    if declared != canonical_sha256(unsigned):
        raise PBDRV4CandidatePoolError("candidate-pool canonical SHA differs")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise PBDRV4CandidatePoolError("candidate records differ")
    try:
        candidates = tuple(CandidateArtifact(**dict(item)) for item in raw_candidates)
    except (TypeError, ValueError) as error:
        raise PBDRV4CandidatePoolError(f"candidate record is invalid: {error}") from error
    if payload.get("candidate_count") != 5 or payload.get("family_order") != list(FROZEN_TIE_ORDER):
        raise PBDRV4CandidatePoolError("candidate count/order differs")
    _validate_artifacts(candidates)
    return dict(payload)


def write_candidate_pool_exclusive(path: Path, payload: Mapping[str, object]) -> Path:
    validate_candidate_pool(payload)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise PBDRV4CandidatePoolError("candidate-pool manifest already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise PBDRV4CandidatePoolError("candidate-pool parent is a symlink")
    content = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def load_candidate_pool(path: Path) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4CandidatePoolError("candidate-pool manifest is missing or unsafe")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4CandidatePoolError(f"cannot read candidate pool: {error}") from error
    if not isinstance(payload, Mapping):
        raise PBDRV4CandidatePoolError("candidate-pool manifest must be an object")
    return validate_candidate_pool(payload)


__all__ = [
    "CANDIDATE_POOL_SCHEMA",
    "CandidateArtifact",
    "PBDRV4CandidatePoolError",
    "build_candidate_pool",
    "canonical_sha256",
    "file_sha256",
    "load_candidate_pool",
    "validate_candidate_pool",
    "write_candidate_pool_exclusive",
]
