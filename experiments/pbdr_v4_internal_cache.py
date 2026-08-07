#!/usr/bin/env python3
"""Fail-closed internal raw-logit cache for PBDR-V4 model selection.

The cache accepts only the development-train or internal-validation identity
from a PBDR-V4 split-authority projection.  It has no dataset-loader API and
cannot construct a split.  Samples are written in projection order to
pickle-free NumPy archives, committed once, and fully revalidated on read.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments import pbdr_v4_split_authority as split_authority


SCHEMA = "sctransnet_pbdr_v4_internal_raw_logit_cache_v1/v1"
COMMIT_SCHEMA = "sctransnet_pbdr_v4_internal_raw_logit_cache_commit_v1/v1"
ROLES = ("best_miou", "best_pd")
PARTITIONS = ("development_train", "internal_validation")
PARTITION_FIELDS = {
    "development_train": ("development_train", "development_train_ids"),
    "internal_validation": (
        "internal_validation",
        "internal_validation_ids",
    ),
}
TENSOR_FIELDS = (
    "base_logits",
    "delta_logits",
    "routed_logits",
    "current_logits",
    "original_logits",
    "target",
)
MANIFEST_NAME = "manifest.json"
COMMIT_NAME = "COMMITTED.json"
SAMPLES_DIRECTORY = "samples"
ADDITION_SEMANTICS = "numpy_add_float32_c_order_exact"


class PBDRV4InternalCacheError(ValueError):
    """A cache identity, sample, container, or commit violated its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4InternalCacheError(message)


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4InternalCacheError(
            f"value cannot be encoded as canonical JSON: {error}"
        ) from error
    return encoded + (b"\n" if trailing_newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, label: str) -> str:
    _require(type(value) is str and len(value) == 64, f"{label} is malformed")
    try:
        int(value, 16)
    except ValueError as error:
        raise PBDRV4InternalCacheError(f"{label} is malformed") from error
    _require(value == value.lower(), f"{label} must be lowercase hexadecimal")
    return value


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    _require(
        not candidate.is_symlink() and candidate.is_file(),
        f"file must be regular and non-symlink: {candidate}",
    )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} fields differ")


def _ordered_ids(value: Any, label: str) -> tuple[str, ...]:
    _require(isinstance(value, (list, tuple)), f"{label} must be a sequence")
    ready = tuple(value)
    _require(
        all(type(identifier) is str and identifier for identifier in ready),
        f"{label} contains an invalid ID",
    )
    _require(len(ready) == len(set(ready)), f"{label} contains duplicate IDs")
    _require(bool(ready), f"{label} cannot be empty")
    return ready


def _ordered_ids_sha256(identifiers: Sequence[str]) -> str:
    return canonical_sha256(list(identifiers))


def _validate_split_projection(value: Mapping[str, Any]) -> str:
    _require(isinstance(value, Mapping), "split projection must be a mapping")
    _require(
        value.get("schema") == split_authority.SCHEMA,
        "split projection schema differs",
    )
    _require(value.get("model_selection_only") is True, "split scope differs")
    _require(
        value.get("parent_seen_official_train") is True,
        "split parent-disclosure differs",
    )
    _require(
        value.get("official_test_accessed") is False,
        "split projection claims official access",
    )
    declared = _sha256(value.get("projection_sha256"), "split projection SHA-256")
    unsigned = dict(value)
    del unsigned["projection_sha256"]
    _require(
        canonical_sha256(unsigned) == declared,
        "split projection SHA-256 does not replay",
    )
    return declared


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    path: str
    bytes: int
    file_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.path) is str and self.path, "checkpoint path is invalid")
        _require(Path(self.path).is_absolute(), "checkpoint path must be absolute")
        _require(
            type(self.bytes) is int and not isinstance(self.bytes, bool) and self.bytes > 0,
            "checkpoint byte count is invalid",
        )
        _sha256(self.file_sha256, "checkpoint file SHA-256")
        _sha256(self.state_sha256, "checkpoint state SHA-256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "file_sha256": self.file_sha256,
            "state_sha256": self.state_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointBinding":
        _require(isinstance(value, Mapping), "checkpoint binding must be a mapping")
        _exact_keys(
            value,
            {"path", "bytes", "file_sha256", "state_sha256"},
            "checkpoint binding",
        )
        return cls(
            path=value["path"],
            bytes=value["bytes"],
            file_sha256=value["file_sha256"],
            state_sha256=value["state_sha256"],
        )


def capture_runtime_binding() -> dict[str, Any]:
    runtime = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    _require(
        runtime["cuda_matmul_allow_tf32"] is False
        and runtime["cudnn_allow_tf32"] is False,
        "both TF32 switches must be false",
    )
    return runtime


def _normalization(value: Mapping[str, Any]) -> dict[str, float]:
    _require(isinstance(value, Mapping), "normalization must be a mapping")
    _exact_keys(value, {"mean", "std"}, "normalization")
    ready: dict[str, float] = {}
    for field in ("mean", "std"):
        raw = value[field]
        _require(not isinstance(raw, bool), f"normalization {field} is invalid")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise PBDRV4InternalCacheError(
                f"normalization {field} is invalid"
            ) from error
        _require(math.isfinite(numeric), f"normalization {field} is non-finite")
        ready[field] = numeric
    _require(ready["std"] > 0.0, "normalization std must be positive")
    return ready


def build_cache_identity(
    *,
    dataset_name: str,
    parent_role: str,
    partition: str,
    split_projection: Mapping[str, Any],
    ordered_sample_ids: Sequence[str],
    v3_checkpoint: CheckpointBinding,
    current_checkpoint: CheckpointBinding,
    original_checkpoint: CheckpointBinding,
    normalization: Mapping[str, Any],
    metric_core_sha256: str,
    source_lock_sha256: str,
) -> dict[str, Any]:
    """Bind one cache to one projected internal partition and runtime."""

    _require(dataset_name in split_authority.DATASETS, "unsupported dataset")
    _require(parent_role in ROLES, "unsupported parent role")
    _require(partition in PARTITIONS, "unsupported internal partition")
    projection_sha = _validate_split_projection(split_projection)
    datasets = split_projection.get("datasets")
    _require(isinstance(datasets, Mapping), "split projection lacks datasets")
    projected = datasets.get(dataset_name)
    _require(isinstance(projected, Mapping), "dataset is absent from split projection")
    _require(projected.get("dataset") == dataset_name, "projected dataset differs")
    _require(projected.get("model_selection_only") is True, "dataset scope differs")
    _require(
        projected.get("parent_seen_official_train") is True,
        "dataset parent-disclosure differs",
    )
    _require(
        projected.get("official_test_accessed") is False,
        "projected dataset claims official access",
    )
    split_canonical_sha = _sha256(
        projected.get("canonical_split_sha256"),
        "canonical split SHA-256",
    )
    count_field, id_hash_field = PARTITION_FIELDS[partition]
    counts = projected.get("counts")
    hashes = projected.get("ordered_id_sha256")
    _require(isinstance(counts, Mapping), "projected counts are malformed")
    _require(isinstance(hashes, Mapping), "projected ID hashes are malformed")
    identifiers = _ordered_ids(ordered_sample_ids, "ordered sample IDs")
    _require(
        counts.get(count_field) == len(identifiers),
        "ordered sample count differs from split projection",
    )
    identifiers_sha = _ordered_ids_sha256(identifiers)
    _require(
        hashes.get(id_hash_field) == identifiers_sha,
        "ordered sample IDs differ from split projection",
    )
    metric_sha = _sha256(metric_core_sha256, "metric-core SHA-256")
    source_lock_sha = _sha256(source_lock_sha256, "source-lock SHA-256")
    for binding, label in (
        (v3_checkpoint, "V3"),
        (current_checkpoint, "Current"),
        (original_checkpoint, "Original"),
    ):
        _require(isinstance(binding, CheckpointBinding), f"{label} binding is invalid")
    identity: dict[str, Any] = {
        "dataset": dataset_name,
        "parent_role": parent_role,
        "partition": partition,
        "split_projection_sha256": projection_sha,
        "canonical_split_sha256": split_canonical_sha,
        "ordered_sample_ids": list(identifiers),
        "ordered_sample_ids_sha256": identifiers_sha,
        "checkpoints": {
            "v3_candidate": v3_checkpoint.as_dict(),
            "current": current_checkpoint.as_dict(),
            "original": original_checkpoint.as_dict(),
        },
        "normalization": _normalization(normalization),
        "metric_core_sha256": metric_sha,
        "source_lock_sha256": source_lock_sha,
        "runtime": capture_runtime_binding(),
        "tensor_contract": {
            "dtype": "float32",
            "layout": "C_contiguous",
            "rank": 2,
            "stored_fields": list(TENSOR_FIELDS),
            "residual_addition_semantics": ADDITION_SEMANTICS,
        },
        "official_test_accessed": False,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def validate_cache_identity(
    identity: Mapping[str, Any],
    split_projection: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(identity, Mapping), "cache identity must be a mapping")
    checkpoints = identity.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), "cache checkpoint bindings are malformed")
    _exact_keys(
        checkpoints,
        {"v3_candidate", "current", "original"},
        "cache checkpoint bindings",
    )
    expected = build_cache_identity(
        dataset_name=identity.get("dataset"),
        parent_role=identity.get("parent_role"),
        partition=identity.get("partition"),
        split_projection=split_projection,
        ordered_sample_ids=identity.get("ordered_sample_ids"),
        v3_checkpoint=CheckpointBinding.from_mapping(checkpoints["v3_candidate"]),
        current_checkpoint=CheckpointBinding.from_mapping(checkpoints["current"]),
        original_checkpoint=CheckpointBinding.from_mapping(checkpoints["original"]),
        normalization=identity.get("normalization"),
        metric_core_sha256=identity.get("metric_core_sha256"),
        source_lock_sha256=identity.get("source_lock_sha256"),
    )
    _require(dict(identity) == expected, "cache identity differs from live bindings")
    return expected


def _validate_array(
    name: str,
    value: Any,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    _require(isinstance(value, np.ndarray), f"{name} must be a NumPy array")
    _require(value.dtype == np.dtype(np.float32), f"{name} must be FP32")
    _require(value.ndim == 2, f"{name} must be rank two")
    _require(value.shape == (height, width), f"{name} shape differs from original size")
    _require(bool(value.flags.c_contiguous), f"{name} must be C-contiguous")
    _require(bool(np.isfinite(value).all()), f"{name} contains non-finite values")
    if name == "target":
        _require(
            bool(((value >= 0.0) & (value <= 1.0)).all()),
            "target must be in [0, 1]",
        )
    return value


def array_semantic_sha256(value: np.ndarray) -> str:
    _require(isinstance(value, np.ndarray), "semantic hash requires an array")
    _require(bool(value.flags.c_contiguous), "semantic hash requires C-contiguous data")
    metadata = {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _bitwise_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def _validated_sample_arrays(
    *,
    height: int,
    width: int,
    base_logits: np.ndarray,
    delta_logits: np.ndarray,
    routed_logits: np.ndarray,
    current_logits: np.ndarray,
    original_logits: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    _require(type(height) is int and height > 0, "sample height is invalid")
    _require(type(width) is int and width > 0, "sample width is invalid")
    supplied = {
        "base_logits": base_logits,
        "delta_logits": delta_logits,
        "routed_logits": routed_logits,
        "current_logits": current_logits,
        "original_logits": original_logits,
        "target": target,
    }
    arrays = {
        name: _validate_array(name, supplied[name], height=height, width=width)
        for name in TENSOR_FIELDS
    }
    expected_routed = np.add(
        arrays["base_logits"],
        arrays["delta_logits"],
        dtype=np.float32,
    )
    _require(
        _bitwise_equal(arrays["routed_logits"], expected_routed),
        "V3 routed logits are not the exact FP32 base-plus-delta result",
    )
    _require(
        _bitwise_equal(arrays["current_logits"], arrays["base_logits"]),
        "standalone Current logits are not bitwise equal to base logits",
    )
    return arrays


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, canonical_json_bytes(value, trailing_newline=True))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class InternalRawLogitCacheWriter:
    """Append projected samples in order and commit one immutable cache."""

    def __init__(
        self,
        destination: Path,
        *,
        dataset_name: str,
        parent_role: str,
        partition: str,
        split_projection: Mapping[str, Any],
        ordered_sample_ids: Sequence[str],
        v3_checkpoint: CheckpointBinding,
        current_checkpoint: CheckpointBinding,
        original_checkpoint: CheckpointBinding,
        normalization: Mapping[str, Any],
        metric_core_sha256: str,
        source_lock_sha256: str,
    ) -> None:
        supplied = Path(destination)
        supplied.parent.mkdir(parents=True, exist_ok=True)
        _require(not supplied.parent.is_symlink(), "cache parent cannot be a symlink")
        parent = supplied.parent.resolve(strict=True)
        self.destination = parent / supplied.name
        if self.destination.exists() or self.destination.is_symlink():
            raise FileExistsError(f"cache destination already exists: {self.destination}")
        self.identity = build_cache_identity(
            dataset_name=dataset_name,
            parent_role=parent_role,
            partition=partition,
            split_projection=split_projection,
            ordered_sample_ids=ordered_sample_ids,
            v3_checkpoint=v3_checkpoint,
            current_checkpoint=current_checkpoint,
            original_checkpoint=original_checkpoint,
            normalization=normalization,
            metric_core_sha256=metric_core_sha256,
            source_lock_sha256=source_lock_sha256,
        )
        self._expected_ids = tuple(self.identity["ordered_sample_ids"])
        self._records: list[dict[str, Any]] = []
        self._temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.destination.name}.tmp.",
                dir=parent,
            )
        )
        (self._temporary / SAMPLES_DIRECTORY).mkdir(mode=0o700)
        self._finalized = False

    def __enter__(self) -> "InternalRawLogitCacheWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._finalized:
            self.abort()

    def abort(self) -> None:
        if self._temporary.exists() and not self._temporary.is_symlink():
            shutil.rmtree(self._temporary)

    def append_sample(
        self,
        *,
        sample_id: str,
        height: int,
        width: int,
        base_logits: np.ndarray,
        delta_logits: np.ndarray,
        routed_logits: np.ndarray,
        current_logits: np.ndarray,
        original_logits: np.ndarray,
        target: np.ndarray,
    ) -> None:
        _require(not self._finalized, "cache is already finalized")
        index = len(self._records)
        _require(index < len(self._expected_ids), "cache received an extra sample")
        _require(type(sample_id) is str, "sample ID must be a string")
        _require(
            sample_id == self._expected_ids[index],
            "sample order differs from split projection",
        )
        arrays = _validated_sample_arrays(
            height=height,
            width=width,
            base_logits=base_logits,
            delta_logits=delta_logits,
            routed_logits=routed_logits,
            current_logits=current_logits,
            original_logits=original_logits,
            target=target,
        )
        relative = f"{SAMPLES_DIRECTORY}/{index:08d}.npz"
        path = self._temporary / relative
        with path.open("xb") as handle:
            np.savez(handle, **{name: arrays[name] for name in TENSOR_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        tensor_records = {
            name: {
                "dtype": "float32",
                "shape": [height, width],
                "layout": "C_contiguous",
                "semantic_sha256": array_semantic_sha256(arrays[name]),
            }
            for name in TENSOR_FIELDS
        }
        record: dict[str, Any] = {
            "index": index,
            "sample_id": sample_id,
            "height": height,
            "width": width,
            "file": relative,
            "file_bytes": path.stat().st_size,
            "file_sha256": file_sha256(path),
            "tensors": tensor_records,
            "v3_routed_equals_base_plus_delta_exact": True,
            "current_equals_base_exact": True,
        }
        record["sample_semantic_sha256"] = canonical_sha256(record)
        self._records.append(record)

    def finalize(self) -> Path:
        _require(not self._finalized, "cache is already finalized")
        _require(
            len(self._records) == len(self._expected_ids),
            "cache is missing projected samples",
        )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "complete_internal_raw_logit_cache",
            "identity": self.identity,
            "sample_count": len(self._records),
            "samples": list(self._records),
            "official_test_accessed": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        temporary_manifest = self._temporary / MANIFEST_NAME
        _write_json_exclusive(temporary_manifest, manifest)
        manifest_file_sha = file_sha256(temporary_manifest)
        manifest_bytes = temporary_manifest.stat().st_size

        try:
            os.mkdir(self.destination, mode=0o700)
        except FileExistsError as error:
            raise FileExistsError(
                f"cache destination already exists: {self.destination}"
            ) from error
        os.rename(
            self._temporary / SAMPLES_DIRECTORY,
            self.destination / SAMPLES_DIRECTORY,
        )
        os.rename(temporary_manifest, self.destination / MANIFEST_NAME)
        commit: dict[str, Any] = {
            "schema": COMMIT_SCHEMA,
            "status": "committed",
            "manifest_file": MANIFEST_NAME,
            "manifest_bytes": manifest_bytes,
            "manifest_file_sha256": manifest_file_sha,
            "manifest_sha256": manifest["manifest_sha256"],
            "sample_count": len(self._records),
            "official_test_accessed": False,
        }
        commit["commit_sha256"] = canonical_sha256(commit)
        _write_json_exclusive(self.destination / COMMIT_NAME, commit)
        _fsync_directory(self.destination / SAMPLES_DIRECTORY)
        _fsync_directory(self.destination)
        _fsync_directory(self.destination.parent)
        os.rmdir(self._temporary)
        self._finalized = True
        return self.destination.resolve(strict=True)


@dataclass(frozen=True, slots=True)
class ValidatedCacheSample:
    sample_id: str
    height: int
    width: int
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class ValidatedInternalRawLogitCache:
    path: Path
    manifest: Mapping[str, Any]
    samples: tuple[ValidatedCacheSample, ...]


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    _require(
        not path.is_symlink() and path.is_file(),
        f"{label} must be a regular non-symlink file",
    )
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4InternalCacheError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one object")
    return value


def _validate_self_hash(
    value: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    declared = _sha256(value.get(field), f"{label} SHA-256")
    unsigned = dict(value)
    del unsigned[field]
    _require(canonical_sha256(unsigned) == declared, f"{label} SHA-256 differs")
    return declared


def read_cache(
    path: Path,
    *,
    split_projection: Mapping[str, Any],
) -> ValidatedInternalRawLogitCache:
    """Read and fully revalidate every container byte and tensor semantic hash."""

    supplied = Path(path)
    _require(
        not supplied.is_symlink() and supplied.is_dir(),
        "cache must be a regular non-symlink directory",
    )
    root = supplied.resolve(strict=True)
    _require(
        {item.name for item in root.iterdir()}
        == {MANIFEST_NAME, COMMIT_NAME, SAMPLES_DIRECTORY},
        "cache top-level entries differ",
    )
    samples_directory = root / SAMPLES_DIRECTORY
    _require(
        not samples_directory.is_symlink() and samples_directory.is_dir(),
        "cache samples entry is not a regular directory",
    )
    commit = _read_json_object(root / COMMIT_NAME, "cache commit")
    _exact_keys(
        commit,
        {
            "schema",
            "status",
            "manifest_file",
            "manifest_bytes",
            "manifest_file_sha256",
            "manifest_sha256",
            "sample_count",
            "official_test_accessed",
            "commit_sha256",
        },
        "cache commit",
    )
    _require(commit.get("schema") == COMMIT_SCHEMA, "cache commit schema differs")
    _require(commit.get("status") == "committed", "cache is not committed")
    _require(commit.get("manifest_file") == MANIFEST_NAME, "manifest name differs")
    _require(
        commit.get("official_test_accessed") is False,
        "cache commit claims official access",
    )
    _validate_self_hash(commit, "commit_sha256", "cache commit")

    manifest_path = root / MANIFEST_NAME
    _require(
        manifest_path.stat().st_size == commit.get("manifest_bytes"),
        "manifest byte count differs",
    )
    _require(
        file_sha256(manifest_path) == commit.get("manifest_file_sha256"),
        "manifest file SHA-256 differs",
    )
    manifest = _read_json_object(manifest_path, "cache manifest")
    _exact_keys(
        manifest,
        {
            "schema",
            "status",
            "identity",
            "sample_count",
            "samples",
            "official_test_accessed",
            "manifest_sha256",
        },
        "cache manifest",
    )
    _require(manifest.get("schema") == SCHEMA, "cache manifest schema differs")
    _require(
        manifest.get("status") == "complete_internal_raw_logit_cache",
        "cache manifest status differs",
    )
    _require(
        manifest.get("official_test_accessed") is False,
        "cache manifest claims official access",
    )
    manifest_sha = _validate_self_hash(
        manifest,
        "manifest_sha256",
        "cache manifest",
    )
    _require(commit.get("manifest_sha256") == manifest_sha, "commit manifest SHA differs")
    identity = validate_cache_identity(manifest.get("identity"), split_projection)
    expected_ids = tuple(identity["ordered_sample_ids"])
    records = manifest.get("samples")
    _require(isinstance(records, list), "cache sample records must be a list")
    _require(
        manifest.get("sample_count") == len(expected_ids) == len(records),
        "cache sample count differs",
    )
    _require(commit.get("sample_count") == len(records), "commit sample count differs")
    expected_files = {f"{index:08d}.npz" for index in range(len(records))}
    _require(
        {item.name for item in samples_directory.iterdir()} == expected_files,
        "cache sample files are missing or extra",
    )

    validated_samples: list[ValidatedCacheSample] = []
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), "cache sample record must be a mapping")
        _exact_keys(
            record,
            {
                "index",
                "sample_id",
                "height",
                "width",
                "file",
                "file_bytes",
                "file_sha256",
                "tensors",
                "v3_routed_equals_base_plus_delta_exact",
                "current_equals_base_exact",
                "sample_semantic_sha256",
            },
            "cache sample record",
        )
        _require(record.get("index") == index, "cache sample index differs")
        _require(record.get("sample_id") == expected_ids[index], "cache sample ID differs")
        height = record.get("height")
        width = record.get("width")
        _require(type(height) is int and height > 0, "cache sample height is invalid")
        _require(type(width) is int and width > 0, "cache sample width is invalid")
        expected_relative = f"{SAMPLES_DIRECTORY}/{index:08d}.npz"
        _require(record.get("file") == expected_relative, "cache sample path differs")
        sample_path = root / expected_relative
        _require(
            not sample_path.is_symlink() and sample_path.is_file(),
            "cache sample must be a regular non-symlink file",
        )
        _require(
            sample_path.stat().st_size == record.get("file_bytes"),
            "cache sample byte count differs",
        )
        _require(
            file_sha256(sample_path) == record.get("file_sha256"),
            "cache sample file SHA-256 differs",
        )
        declared_sample_sha = _sha256(
            record.get("sample_semantic_sha256"),
            "sample semantic SHA-256",
        )
        unsigned_record = dict(record)
        del unsigned_record["sample_semantic_sha256"]
        _require(
            canonical_sha256(unsigned_record) == declared_sample_sha,
            "sample semantic record SHA-256 differs",
        )
        tensor_records = record.get("tensors")
        _require(isinstance(tensor_records, Mapping), "tensor records are malformed")
        _require(set(tensor_records) == set(TENSOR_FIELDS), "tensor record fields differ")
        try:
            with np.load(sample_path, allow_pickle=False) as archive:
                _require(
                    len(archive.files) == len(TENSOR_FIELDS)
                    and set(archive.files) == set(TENSOR_FIELDS),
                    "NPZ fields differ",
                )
                raw_arrays = {name: archive[name] for name in TENSOR_FIELDS}
        except (OSError, ValueError, KeyError) as error:
            raise PBDRV4InternalCacheError(
                f"cannot read cache sample archive: {error}"
            ) from error
        arrays = _validated_sample_arrays(
            height=height,
            width=width,
            **raw_arrays,
        )
        copied: dict[str, np.ndarray] = {}
        for name in TENSOR_FIELDS:
            metadata = tensor_records[name]
            _require(isinstance(metadata, Mapping), f"{name} metadata is malformed")
            _require(
                dict(metadata)
                == {
                    "dtype": "float32",
                    "shape": [height, width],
                    "layout": "C_contiguous",
                    "semantic_sha256": array_semantic_sha256(arrays[name]),
                },
                f"{name} semantic metadata differs",
            )
            ready = arrays[name].copy(order="C")
            ready.setflags(write=False)
            copied[name] = ready
        _require(
            record.get("v3_routed_equals_base_plus_delta_exact") is True
            and record.get("current_equals_base_exact") is True,
            "sample exact-identity attestations differ",
        )
        validated_samples.append(
            ValidatedCacheSample(
                sample_id=expected_ids[index],
                height=height,
                width=width,
                arrays=copied,
            )
        )
    return ValidatedInternalRawLogitCache(
        path=root,
        manifest=manifest,
        samples=tuple(validated_samples),
    )


__all__ = [
    "ADDITION_SEMANTICS",
    "COMMIT_NAME",
    "COMMIT_SCHEMA",
    "CheckpointBinding",
    "InternalRawLogitCacheWriter",
    "MANIFEST_NAME",
    "PARTITIONS",
    "PBDRV4InternalCacheError",
    "ROLES",
    "SCHEMA",
    "SAMPLES_DIRECTORY",
    "TENSOR_FIELDS",
    "ValidatedCacheSample",
    "ValidatedInternalRawLogitCache",
    "array_semantic_sha256",
    "build_cache_identity",
    "canonical_json_bytes",
    "canonical_sha256",
    "capture_runtime_binding",
    "file_sha256",
    "read_cache",
    "validate_cache_identity",
]
