#!/usr/bin/env python3
"""Build the train-only IRSTD BGCR frozen-context cache.

This module has a deliberately narrow data boundary.  It accepts only the
800 identifiers already frozen by the IRSTD official-train split manifest and
resolves image/mask pairs directly below ``datasets/IRSTD-1K``.  It neither
imports a dataset/evaluation module nor reads or constructs any non-training
index.

The Current graph is constructed through the formal BGCR builder.  The only
independent teacher admitted by the production entry point is the byte- and
state-bound epoch-1000 Original SCTransNet checkpoint.  Cache construction is
transactional: individual samples and records are atomically published in a
deterministic incomplete directory, verified on resume, and the complete tree
is renamed into place only after all 800 samples validate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Callable, Final, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from experiments.irstd_bgcr_run_contract import (
    OFFICIAL_FALSE_FLAGS as RUN_CONTRACT_OFFICIAL_FALSE_FLAGS,
)


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

CACHE_SCHEMA: Final[str] = "sctransnet_irstd_bgcr_frozen_context_cache_v1/v1"
BUILD_IDENTITY_SCHEMA: Final[str] = (
    "sctransnet_irstd_bgcr_frozen_context_build_identity_v1/v1"
)
SAMPLE_RECORD_SCHEMA: Final[str] = (
    "sctransnet_irstd_bgcr_frozen_context_sample_v1/v1"
)
COMMIT_SCHEMA: Final[str] = "sctransnet_irstd_bgcr_frozen_context_commit_v1/v1"

DATASET_NAME: Final[str] = "IRSTD-1K"
OFFICIAL_TRAIN_COUNT: Final[int] = 800
DEVELOPMENT_TRAIN_COUNT: Final[int] = 640
INTERNAL_VALIDATION_COUNT: Final[int] = 160
FULL_CONTEXT_HEIGHT: Final[int] = 512
FULL_CONTEXT_WIDTH: Final[int] = 512
LOCAL_FEATURE_CHANNELS: Final[int] = 32
OFFICIAL_TEST_ACCESSED: Final[bool] = False
PERFORMANCE_MARGIN: Final[None] = None
PERFORMANCE_ACCEPTANCE_MARGIN: Final[None] = None
FORMAL_SEED: Final[int] = 42
OFFICIAL_FALSE_FLAGS: Final[dict[str, bool]] = dict(
    RUN_CONTRACT_OFFICIAL_FALSE_FLAGS
)

IRSTD_IMAGE_MEAN: Final[float] = 87.4661865234375
IRSTD_IMAGE_STD: Final[float] = 39.71953201293945
TARGET_SOURCE_SCALE: Final[float] = 255.0
TARGET_BINARIZATION_THRESHOLD_SOURCE: Final[float] = 127.5
TARGET_BINARIZATION_THRESHOLD_NORMALIZED: Final[float] = 0.5
TARGET_BINARIZATION_COMPARISON: Final[str] = "strict_greater_than"
DEFAULT_DATA_ROOT: Final[Path] = REPO_ROOT / "datasets"

SPLIT_PROJECTION_PATH: Final[Path] = (
    REPO_ROOT / "results/pbdr_v4_v1/protocol/split_projection.json"
)
SPLIT_PROJECTION_FILE_SHA256: Final[str] = (
    "537b8b75d27629318579b5367b52799e78beba562a11cfed1d1d49bc5096af38"
)
SPLIT_PROJECTION_SHA256: Final[str] = (
    "edf6fffb47f52693dbdd6c82209ff2b2259095b7aed5b91b28aab0e83112d1fb"
)
SOURCE_SPLIT_MANIFEST_PATH: Final[Path] = (
    REPO_ROOT
    / "results/two_dataset_pbdr_v3_stage1_v1/runs/IRSTD-1K/formal/"
    "best_miou/core/split_manifest.json"
)
SOURCE_SPLIT_MANIFEST_FILE_SHA256: Final[str] = (
    "8bb2a0cb7cf7802c62ec54e1a43d8ff2524c1c2d45c5ffeb84cb88850f8bdeb4"
)
SOURCE_SPLIT_SCHEMA: Final[str] = (
    "sctransnet_two_dataset_pbdr_v3_internal_split_v1/v1"
)
CANONICAL_SPLIT_SHA256: Final[str] = (
    "9371a6be7a2671010a3eb014ef4763c97a6528757b2085e262e82237b2e14bac"
)
OFFICIAL_TRAIN_INDEX_SHA256: Final[str] = (
    "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744"
)
OFFICIAL_TRAIN_IDS_SHA256: Final[str] = (
    "681e4d741fb857703471d6555faa0d86e931aa790567c28f4254331ea9ba3d95"
)
DEVELOPMENT_TRAIN_IDS_SHA256: Final[str] = (
    "f4144772f01f7d373b450dd856c618f62d7aa0b3e1a4d14e47f7afff359ed589"
)
INTERNAL_VALIDATION_IDS_SHA256: Final[str] = (
    "22ceba9e2af438a66470f2cdd73a27586fdf1bed760f9d3eb12fdb80e4133734"
)

CURRENT_CHECKPOINT_PATH: Final[Path] = (
    REPO_ROOT
    / "results/three_dataset_tss_off_seed42_v1/runs/IRSTD-1K/"
    "final_tss_off/seed_42/checkpoints/best_miou.pth.tar"
)
CURRENT_CHECKPOINT_FILE_SHA256: Final[str] = (
    "e8e9401500502dda0bbdc9640b830a7934fb2bc97bde706fde9adca216d965b4"
)
CURRENT_CHECKPOINT_BYTES: Final[int] = 43_726_377
CURRENT_CHECKPOINT_EPOCH: Final[int] = 830
CURRENT_TRAINING_STATE_KEYS: Final[int] = 568
CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256: Final[str] = (
    "d7600f61ee3d0967dae899de72a28f2e7e9e4c6381f2687189e45d84dcb3e298"
)
CURRENT_INFERENCE_STATE_KEYS: Final[int] = 564
CURRENT_INFERENCE_STATE_SEMANTIC_SHA256: Final[str] = (
    "f3745109e889cc6f25e42a43e698c5a43516ddc96a1364ffc78ab4b6b09d7f4f"
)

BASELINE_TEACHER_CHECKPOINT_PATH: Final[Path] = Path(
    "/home/ly/SCTransNet/checkpoints/IRSTD-1K/SCTransNet_1000.pth.tar"
)
BASELINE_TEACHER_CHECKPOINT_FILE_SHA256: Final[str] = (
    "b4cb66be6e4a410dfd902ba050da82d0b666dd071bfb2c5477a7c3173ff07bc5"
)
BASELINE_TEACHER_CHECKPOINT_BYTES: Final[int] = 45_535_091
BASELINE_TEACHER_RAW_STATE_SEMANTIC_SHA256: Final[str] = (
    "972e7c15f8da8142da85112f535fb555a86293e12d7341d7c5be653fb4076d9b"
)
BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256: Final[str] = (
    "1961ed8ee278fde09508145fe537324172599bfa704c181dc53f756578070b5c"
)
BASELINE_TEACHER_EPOCH: Final[int] = 1000
BASELINE_TEACHER_STATE_KEYS: Final[int] = 510
BASELINE_TEACHER_RAW_STATE_PREFIX: Final[str] = "model."
OPERATIONAL_REFERENCE_EPOCH: Final[int] = 713
BASELINE_SOURCE_PATH: Final[Path] = Path(
    "/home/ly/SCTransNet/model/SCTransNet.py"
)
CURRENT_SCTransNET_SOURCE_PATH: Final[Path] = REPO_ROOT / "model/SCTransNet.py"
SCTransNET_SOURCE_SHA256: Final[str] = (
    "5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb"
)

FLOAT_ARRAY_KEYS: Final[tuple[str, ...]] = (
    "image",
    "target",
    "u1",
    "z_out",
    "z_d0",
    "z_gt2",
    "z_gt3",
    "z_gt4",
    "z_gt5",
    "baseline1000_logits",
)
ATLAS_ID_KEYS: Final[tuple[str, ...]] = (
    "target_component_ids",
    "rescue_component_ids",
)
ATLAS_MASK_KEYS: Final[tuple[str, ...]] = (
    "core_target",
    "attached_halo",
    "detached_false_positive",
    "outer_ring",
    "halo_target",
    "far_background",
    "baseline_rescue",
    "baseline_halo_advantage",
)
CACHE_ARRAY_KEYS: Final[tuple[str, ...]] = (
    *FLOAT_ARRAY_KEYS,
    *ATLAS_ID_KEYS,
    *ATLAS_MASK_KEYS,
)

_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_IMAGE_SUFFIXES: Final[tuple[str, ...]] = (
    ".png",
    ".bmp",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


class IRSTDBGCRCacheError(RuntimeError):
    """A source, model, cache item, or transactional cache contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IRSTDBGCRCacheError(message)


def expected_determinism_manifest() -> dict[str, Any]:
    return {
        "schema": "sctransnet_irstd_bgcr_cache_determinism_v1/v1",
        "seed": FORMAL_SEED,
        "precision": "fp32",
        "default_dtype": "torch.float32",
        "model_and_cache_dtype": "torch.float32",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": ":4096:8",
        "autocast": False,
    }


def configure_determinism(seed: int = FORMAL_SEED) -> dict[str, Any]:
    """Configure the only formal FP32/seed/TF32 policy before CUDA use."""

    _require(type(seed) is int and seed == FORMAL_SEED, "formal cache seed must be 42")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    observed = expected_determinism_manifest()
    _require(torch.get_default_dtype() == torch.float32, "torch default dtype differs")
    _require(
        torch.are_deterministic_algorithms_enabled(),
        "deterministic algorithms were not enabled",
    )
    warn_only = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )()
    _require(not warn_only, "deterministic algorithms are warn-only")
    _require(
        torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark
        and not torch.backends.cudnn.allow_tf32
        and not torch.backends.cuda.matmul.allow_tf32,
        "CUDNN/TF32 deterministic policy differs",
    )
    _require(
        torch.get_float32_matmul_precision() == "highest",
        "float32 matmul precision differs",
    )
    _require(not torch.is_autocast_enabled(), "autocast must be disabled")
    return observed


def _require_sha256(value: Any, *, name: str) -> str:
    _require(
        type(value) is str and _SHA256.fullmatch(value) is not None,
        f"{name} must be one lowercase SHA-256 digest",
    )
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IRSTDBGCRCacheError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} must be a regular non-symlink file: {candidate}",
    )
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IRSTDBGCRCacheError(f"cannot read {label}: {candidate}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_ready(value: Any) -> Any:
    """Convert builder metadata to a deterministic JSON-only representation."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        ready = [_json_ready(item) for item in value]
        return sorted(ready, key=lambda item: _canonical_json_bytes(item))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        ready = value.detach().cpu()
        return {
            "tensor_dtype": str(ready.dtype),
            "tensor_shape": list(ready.shape),
            "tensor_state_sha256": state_semantic_sha256({"value": ready}),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"builder metadata contains unsupported value: {type(value)!r}")


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"expected a regular non-symlink file: {candidate}",
    )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(identifiers: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(identifiers),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def tensor_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash state names, dtypes, shapes, and dense bytes using the frozen V3 rule."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if type(name) is not str or not isinstance(tensor, torch.Tensor):
            raise TypeError("state must map string keys to tensors")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def state_semantic_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a state with the semantic algorithm used by the teacher ledger."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if type(name) is not str or not isinstance(value, torch.Tensor):
            raise TypeError("state must map string keys to tensors")
        ready = value.detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(ready.dtype).encode("ascii"))
        digest.update(len(ready.shape).to_bytes(8, "little"))
        for dimension in ready.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=True))
        raw = ready.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def array_semantic_sha256(value: np.ndarray) -> str:
    """Hash one cache array independently of NPZ container metadata."""

    if not isinstance(value, np.ndarray):
        raise TypeError("cache value must be a numpy array")
    if value.dtype == np.dtype(np.float32):
        ready = np.ascontiguousarray(value.astype(np.dtype("<f4"), copy=False))
        semantic_dtype = "float32"
    elif value.dtype == np.dtype(np.int32):
        ready = np.ascontiguousarray(value.astype(np.dtype("<i4"), copy=False))
        semantic_dtype = "int32"
    elif value.dtype == np.dtype(np.bool_):
        ready = np.ascontiguousarray(value.astype(np.bool_, copy=False))
        semantic_dtype = "bool"
    else:
        raise TypeError(f"unsupported cache dtype: {value.dtype}")
    descriptor = _canonical_json_bytes(
        {"dtype": semantic_dtype, "shape": list(ready.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"sctransnet-irstd-bgcr-cache-array-v1\0")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    digest.update(ready.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainSplitAuthority:
    schema: str
    dataset: str
    split_seed: int
    validation_fraction: float
    official_count: int
    development_count: int
    internal_count: int
    canonical_split_sha256: str
    official_ids_sha256: str
    development_ids_sha256: str
    internal_ids_sha256: str
    official_train_index_sha256: str


IRSTD_TRAIN_SPLIT_AUTHORITY: Final[TrainSplitAuthority] = TrainSplitAuthority(
    schema=SOURCE_SPLIT_SCHEMA,
    dataset=DATASET_NAME,
    split_seed=20260722,
    validation_fraction=0.20,
    official_count=OFFICIAL_TRAIN_COUNT,
    development_count=DEVELOPMENT_TRAIN_COUNT,
    internal_count=INTERNAL_VALIDATION_COUNT,
    canonical_split_sha256=CANONICAL_SPLIT_SHA256,
    official_ids_sha256=OFFICIAL_TRAIN_IDS_SHA256,
    development_ids_sha256=DEVELOPMENT_TRAIN_IDS_SHA256,
    internal_ids_sha256=INTERNAL_VALIDATION_IDS_SHA256,
    official_train_index_sha256=OFFICIAL_TRAIN_INDEX_SHA256,
)


@dataclass(frozen=True, slots=True)
class BoundIRSTDTrainSplit:
    official_train_ids: tuple[str, ...]
    development_train_ids: tuple[str, ...]
    internal_validation_ids: tuple[str, ...]
    source_manifest_path: Path
    source_manifest_file_sha256: str
    projection_path: Path
    projection_file_sha256: str

    def membership(self, sample_id: str) -> str:
        if sample_id in frozenset(self.development_train_ids):
            return "development_train"
        if sample_id in frozenset(self.internal_validation_ids):
            return "internal_validation"
        raise IRSTDBGCRCacheError(f"sample is outside the bound train split: {sample_id}")


def _validated_ids(value: Any, *, name: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{name} must be an ordered JSON list")
    identifiers = tuple(value)
    _require(bool(identifiers), f"{name} must not be empty")
    _require(
        all(
            type(identifier) is str
            and _SAFE_SAMPLE_ID.fullmatch(identifier) is not None
            for identifier in identifiers
        ),
        f"{name} contains an unsafe sample identifier",
    )
    _require(len(identifiers) == len(set(identifiers)), f"{name} has duplicates")
    return identifiers


def validate_train_split_payload(
    payload: Mapping[str, Any],
    *,
    authority: TrainSplitAuthority = IRSTD_TRAIN_SPLIT_AUTHORITY,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate a parsed train-only split; parameters permit synthetic tests."""

    _require(isinstance(payload, Mapping), "split payload must be a mapping")
    _require(payload.get("schema") == authority.schema, "split schema differs")
    _require(payload.get("dataset") == authority.dataset, "split dataset differs")
    _require(
        payload.get("source_split") == "official_train_only",
        "split source is not official-train-only",
    )
    _require(
        payload.get("official_test_index_opened") is False,
        "split records prohibited index access",
    )
    _require(payload.get("split_seed") == authority.split_seed, "split seed differs")
    _require(
        payload.get("val_fraction") == authority.validation_fraction,
        "split validation fraction differs",
    )
    _require(
        payload.get("official_train_index_sha256")
        == authority.official_train_index_sha256,
        "official-train index binding differs",
    )
    declared = _require_sha256(payload.get("split_sha256"), name="split SHA")
    unsigned = dict(payload)
    unsigned.pop("split_sha256", None)
    replayed = canonical_sha256(unsigned)
    _require(declared == replayed, "split canonical SHA does not replay")
    _require(
        replayed == authority.canonical_split_sha256,
        "split canonical SHA differs from authority",
    )

    official = _validated_ids(payload.get("official_train_ids"), name="official IDs")
    development = _validated_ids(
        payload.get("development_train_ids"), name="development IDs"
    )
    internal = _validated_ids(
        payload.get("internal_validation_ids"), name="internal IDs"
    )
    _require(
        (len(official), len(development), len(internal))
        == (
            authority.official_count,
            authority.development_count,
            authority.internal_count,
        ),
        "split counts differ",
    )
    development_set = set(development)
    internal_set = set(internal)
    _require(
        not development_set & internal_set
        and development_set | internal_set == set(official),
        "development/internal IDs do not partition official train",
    )
    observed = (
        ordered_ids_sha256(official),
        ordered_ids_sha256(development),
        ordered_ids_sha256(internal),
    )
    expected = (
        authority.official_ids_sha256,
        authority.development_ids_sha256,
        authority.internal_ids_sha256,
    )
    _require(observed == expected, "ordered train ID hashes differ")
    return official, development, internal


def _validate_split_projection(payload: Mapping[str, Any]) -> None:
    _require(
        payload.get("schema") == "sctransnet_pbdr_v4_split_authority_v1/v1",
        "split projection schema differs",
    )
    _require(
        payload.get("status") == "frozen_v3_split_authority_projection",
        "split projection is not frozen",
    )
    _require(payload.get("model_selection_only") is True, "split scope differs")
    _require(
        payload.get("official_test_accessed") is False,
        "projection records prohibited access",
    )
    _require(
        payload.get("split_reconstruction_performed") is False,
        "split was unexpectedly reconstructed",
    )
    declared = _require_sha256(
        payload.get("projection_sha256"), name="projection SHA"
    )
    unsigned = dict(payload)
    unsigned.pop("projection_sha256", None)
    _require(declared == canonical_sha256(unsigned), "projection SHA does not replay")
    _require(declared == SPLIT_PROJECTION_SHA256, "projection authority differs")
    datasets = payload.get("datasets")
    _require(isinstance(datasets, Mapping), "projection datasets differ")
    record = datasets.get(DATASET_NAME)
    _require(isinstance(record, Mapping), "IRSTD projection record is absent")
    expected = {
        "dataset": DATASET_NAME,
        "canonical_split_sha256": CANONICAL_SPLIT_SHA256,
        "source_path": str(SOURCE_SPLIT_MANIFEST_PATH),
        "source_relative_path": str(SOURCE_SPLIT_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "source_file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "model_selection_only": True,
        "official_test_accessed": False,
        "parent_seen_official_train": True,
    }
    for name, value in expected.items():
        _require(record.get(name) == value, f"projection IRSTD {name} differs")
    _require(
        record.get("counts")
        == {
            "official_train": OFFICIAL_TRAIN_COUNT,
            "development_train": DEVELOPMENT_TRAIN_COUNT,
            "internal_validation": INTERNAL_VALIDATION_COUNT,
        },
        "projection counts differ",
    )
    _require(
        record.get("ordered_id_sha256")
        == {
            "official_train_ids": OFFICIAL_TRAIN_IDS_SHA256,
            "development_train_ids": DEVELOPMENT_TRAIN_IDS_SHA256,
            "internal_validation_ids": INTERNAL_VALIDATION_IDS_SHA256,
        },
        "projection ordered-ID binding differs",
    )


def load_bound_irstd_train_split(
    *,
    projection_path: Path = SPLIT_PROJECTION_PATH,
    source_manifest_path: Path = SOURCE_SPLIT_MANIFEST_PATH,
) -> BoundIRSTDTrainSplit:
    projection = Path(projection_path)
    source = Path(source_manifest_path)
    _require(
        projection.resolve(strict=True) == SPLIT_PROJECTION_PATH.resolve(strict=True),
        "only the frozen split projection is allowed",
    )
    _require(
        source.resolve(strict=True) == SOURCE_SPLIT_MANIFEST_PATH.resolve(strict=True),
        "only the frozen IRSTD source split is allowed",
    )
    _require(
        file_sha256(projection) == SPLIT_PROJECTION_FILE_SHA256,
        "split projection file SHA differs",
    )
    _require(
        file_sha256(source) == SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "source split file SHA differs",
    )
    projection_payload = _load_json_object(projection, label="split projection")
    _validate_split_projection(projection_payload)
    source_payload = _load_json_object(source, label="IRSTD source split")
    official, development, internal = validate_train_split_payload(source_payload)
    return BoundIRSTDTrainSplit(
        official_train_ids=official,
        development_train_ids=development,
        internal_validation_ids=internal,
        source_manifest_path=source.resolve(strict=True),
        source_manifest_file_sha256=SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        projection_path=projection.resolve(strict=True),
        projection_file_sha256=SPLIT_PROJECTION_FILE_SHA256,
    )


def _exact_regular_path(value: Path, expected: Path, *, label: str) -> Path:
    candidate = Path(value)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} is missing or unsafe: {candidate}",
    )
    ready = candidate.resolve(strict=True)
    _require(
        ready == expected.resolve(strict=True),
        f"{label} path differs from the frozen path",
    )
    return ready


def _load_torch_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise IRSTDBGCRCacheError(f"cannot load {label}: {path}") from error
    _require(
        isinstance(value, Mapping) and all(type(key) is str for key in value),
        f"{label} must contain a string-key mapping",
    )
    return dict(value)


def _validated_tensor_state(
    value: Any,
    *,
    expected_keys: int,
    expected_sha256: str,
    label: str,
    semantic_hasher: Callable[[Mapping[str, torch.Tensor]], str] = tensor_mapping_sha256,
) -> dict[str, torch.Tensor]:
    _require(isinstance(value, Mapping), f"{label} state is not a mapping")
    state = dict(value)
    _require(
        len(state) == expected_keys
        and all(type(key) is str for key in state)
        and all(isinstance(tensor, torch.Tensor) for tensor in state.values()),
        f"{label} state-key contract differs",
    )
    for name, tensor in state.items():
        _require(bool(torch.isfinite(tensor).all()), f"{label} tensor {name!r} is non-finite")
    observed = semantic_hasher(state)  # type: ignore[arg-type]
    _require(observed == expected_sha256, f"{label} tensor-state SHA differs")
    return state  # type: ignore[return-value]


def load_bound_current_training_state(
    checkpoint_path: Path = CURRENT_CHECKPOINT_PATH,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    ready = _exact_regular_path(
        checkpoint_path, CURRENT_CHECKPOINT_PATH, label="Current checkpoint"
    )
    observed_file_sha = file_sha256(ready)
    _require(ready.stat().st_size == CURRENT_CHECKPOINT_BYTES, "Current checkpoint bytes differ")
    _require(
        observed_file_sha == CURRENT_CHECKPOINT_FILE_SHA256,
        "Current checkpoint file SHA differs",
    )
    payload = _load_torch_mapping(ready, label="Current checkpoint")
    _require(payload.get("epoch") == CURRENT_CHECKPOINT_EPOCH, "Current epoch differs")
    state = _validated_tensor_state(
        payload.get("state_dict"),
        expected_keys=CURRENT_TRAINING_STATE_KEYS,
        expected_sha256=CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256,
        label="Current training",
    )
    return state, {
        "path": str(ready),
        "file_sha256": observed_file_sha,
        "file_bytes": ready.stat().st_size,
        "epoch": CURRENT_CHECKPOINT_EPOCH,
        "training_state_keys": len(state),
        "training_state_tensor_mapping_sha256": (
            CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256
        ),
        "training_state_tensor_mapping_hash_algorithm": "tensor_mapping_sha256",
    }


def normalize_baseline_teacher_state(
    raw_state: Mapping[str, Any],
    *,
    expected_keys: int = BASELINE_TEACHER_STATE_KEYS,
    expected_state_sha256: str = (
        BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256
    ),
    required_prefix: str = BASELINE_TEACHER_RAW_STATE_PREFIX,
) -> dict[str, torch.Tensor]:
    """Strip exactly the historical ``model.`` wrapper and validate state bytes."""

    _require(isinstance(raw_state, Mapping), "teacher state is not a mapping")
    _require(
        required_prefix == BASELINE_TEACHER_RAW_STATE_PREFIX,
        "teacher wrapper prefix must be exactly 'model.'",
    )
    _require(len(raw_state) == expected_keys, "teacher raw state-key count differs")
    _require(
        all(type(key) is str and key.startswith(required_prefix) for key in raw_state),
        "teacher raw state prefix differs",
    )
    normalized: dict[str, torch.Tensor] = {}
    for raw_name, value in raw_state.items():
        _require(isinstance(value, torch.Tensor), "teacher state contains a non-tensor")
        name = raw_name[len(required_prefix) :]
        _require(bool(name) and name not in normalized, "teacher state names collide")
        normalized[name] = value
    return _validated_tensor_state(
        normalized,
        expected_keys=expected_keys,
        expected_sha256=expected_state_sha256,
        label="Baseline epoch-1000 teacher",
        semantic_hasher=state_semantic_sha256,
    )


def _load_and_validate_teacher_checkpoint(
    checkpoint_path: Path,
    *,
    expected_path: Path,
    expected_file_sha256: str,
    expected_raw_state_semantic_sha256: str,
    expected_normalized_state_semantic_sha256: str,
    expected_state_keys: int,
    expected_epoch: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    _require(
        expected_epoch == BASELINE_TEACHER_EPOCH
        and expected_epoch != OPERATIONAL_REFERENCE_EPOCH,
        "teacher epoch must be the frozen epoch 1000",
    )
    ready = _exact_regular_path(checkpoint_path, expected_path, label="teacher checkpoint")
    observed_file_sha = file_sha256(ready)
    _require(
        ready.stat().st_size == BASELINE_TEACHER_CHECKPOINT_BYTES,
        "teacher checkpoint bytes differ",
    )
    _require(observed_file_sha == expected_file_sha256, "teacher checkpoint SHA differs")
    payload = _load_torch_mapping(ready, label="teacher checkpoint")
    _require(payload.get("epoch") == expected_epoch, "teacher checkpoint epoch differs")
    raw_state = payload.get("state_dict", {})
    _require(
        isinstance(raw_state, Mapping)
        and state_semantic_sha256(raw_state)
        == expected_raw_state_semantic_sha256,
        "teacher wrapped semantic state SHA differs",
    )
    normalized = normalize_baseline_teacher_state(
        raw_state,
        expected_keys=expected_state_keys,
        expected_state_sha256=expected_normalized_state_semantic_sha256,
    )
    return normalized, {
        "path": str(ready),
        "file_sha256": observed_file_sha,
        "file_bytes": ready.stat().st_size,
        "epoch": expected_epoch,
        "state_keys": len(normalized),
        "raw_state_semantic_sha256": expected_raw_state_semantic_sha256,
        "normalized_state_semantic_sha256": (
            expected_normalized_state_semantic_sha256
        ),
        "state_semantic_hash_algorithm": "state_semantic_sha256",
        "raw_state_prefix": BASELINE_TEACHER_RAW_STATE_PREFIX,
        "trainable": False,
        "enabled": True,
        "operational_reference_epoch": OPERATIONAL_REFERENCE_EPOCH,
        "operational_reference_is_teacher": False,
    }


def load_bound_baseline1000_teacher_state(
    checkpoint_path: Path = BASELINE_TEACHER_CHECKPOINT_PATH,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    return _load_and_validate_teacher_checkpoint(
        checkpoint_path,
        expected_path=BASELINE_TEACHER_CHECKPOINT_PATH,
        expected_file_sha256=BASELINE_TEACHER_CHECKPOINT_FILE_SHA256,
        expected_raw_state_semantic_sha256=(
            BASELINE_TEACHER_RAW_STATE_SEMANTIC_SHA256
        ),
        expected_normalized_state_semantic_sha256=(
            BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256
        ),
        expected_state_keys=BASELINE_TEACHER_STATE_KEYS,
        expected_epoch=BASELINE_TEACHER_EPOCH,
    )


@dataclass(frozen=True, slots=True)
class FormalCacheModels:
    current: nn.Module
    baseline1000: nn.Module
    current_binding: Mapping[str, Any]
    baseline1000_binding: Mapping[str, Any]
    source_binding: Mapping[str, Any]


def load_formal_bgcr_cache_models(
    *,
    device: torch.device | str,
    current_checkpoint_path: Path = CURRENT_CHECKPOINT_PATH,
    teacher_checkpoint_path: Path = BASELINE_TEACHER_CHECKPOINT_PATH,
) -> FormalCacheModels:
    """Construct both frozen graphs without importing any data/evaluation code."""

    ready_device = torch.device(device)
    current_state, current_binding = load_bound_current_training_state(
        current_checkpoint_path
    )
    teacher_state, teacher_binding = load_bound_baseline1000_teacher_state(
        teacher_checkpoint_path
    )
    for source in (BASELINE_SOURCE_PATH, CURRENT_SCTransNET_SOURCE_PATH):
        _require(
            file_sha256(source) == SCTransNET_SOURCE_SHA256,
            f"SCTransNet source binding differs: {source}",
        )

    # Delayed imports keep the cache module importable for static/synthetic
    # tests and ensure the formal builder is the sole Current construction path.
    from model import tpd8_ner4_qfg2_irstd_crr as bgcr_model
    from experiments import irstd_baseline_teacher as teacher_module

    inference_state = bgcr_model.strip_current_survival_state_strict(current_state)
    _require(
        len(inference_state) == CURRENT_INFERENCE_STATE_KEYS,
        "Current inference state-key count differs",
    )
    current_inference_sha = tensor_mapping_sha256(inference_state)
    current_training_semantic_sha = state_semantic_sha256(current_state)
    current_inference_semantic_sha = state_semantic_sha256(inference_state)
    _require(
        current_inference_semantic_sha == CURRENT_INFERENCE_STATE_SEMANTIC_SHA256,
        "Current inference semantic state SHA differs",
    )
    current, builder_metadata = bgcr_model.build_formal_irstd_bgcr_model(
        current_state,
        seed=42,
        repair_initialization_seed=42,
    )
    _require(
        builder_metadata.get("current_training_state_semantic_sha256")
        == current_training_semantic_sha
        and builder_metadata.get("current_inference_state_semantic_sha256")
        == current_inference_semantic_sha,
        "formal BGCR builder reports a different Current semantic state SHA",
    )
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        _require(
            builder_metadata.get(flag) is expected,
            f"formal BGCR builder official flag differs: {flag}",
        )
    bgcr_model.audit_frozen_current_base(current, inference_state)
    for parameter in current.parameters():
        parameter.requires_grad_(False)
    _require(
        all(parameter.dtype == torch.float32 for parameter in current.parameters()),
        "Current model parameters are not FP32",
    )
    current.eval()
    current.to(ready_device)

    _require(
        state_semantic_sha256(teacher_state)
        == BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256,
        "validated teacher state changed before formal construction",
    )
    baseline = teacher_module.build_formal_teacher()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    _require(
        all(parameter.dtype == torch.float32 for parameter in baseline.parameters()),
        "Baseline teacher parameters are not FP32",
    )
    baseline.eval()
    baseline.to(ready_device)

    wrapper_source = Path(bgcr_model.__file__).resolve(strict=True)
    teacher_builder_source = Path(teacher_module.__file__).resolve(strict=True)
    source_binding = {
        "formal_bgcr_wrapper": {
            "path": str(wrapper_source),
            "file_sha256": file_sha256(wrapper_source),
        },
        "baseline_teacher_builder": {
            "path": str(teacher_builder_source),
            "file_sha256": file_sha256(teacher_builder_source),
        },
        "current_sctransnet": {
            "path": str(CURRENT_SCTransNET_SOURCE_PATH.resolve(strict=True)),
            "file_sha256": SCTransNET_SOURCE_SHA256,
        },
        "baseline_sctransnet": {
            "path": str(BASELINE_SOURCE_PATH.resolve(strict=True)),
            "file_sha256": SCTransNET_SOURCE_SHA256,
        },
    }
    current_ready = {
        **current_binding,
        "training_state_semantic_sha256": current_training_semantic_sha,
        "training_state_semantic_hash_algorithm": "state_semantic_sha256",
        "inference_state_keys": len(inference_state),
        "inference_state_tensor_mapping_sha256": current_inference_sha,
        "inference_state_tensor_mapping_hash_algorithm": "tensor_mapping_sha256",
        "inference_state_semantic_sha256": current_inference_semantic_sha,
        "inference_state_semantic_hash_algorithm": "state_semantic_sha256",
        "formal_builder": "build_formal_irstd_bgcr_model",
        "formal_context_api": "forward_for_irstd_training",
        "seed": 42,
        "repair_initialization_seed": 42,
        "builder_metadata_sha256": canonical_sha256(_json_ready(builder_metadata)),
    }
    teacher_ready = {
        **teacher_binding,
        "builder": "experiments.irstd_baseline_teacher.build_formal_teacher",
        "builder_audit_sha256": canonical_sha256(_json_ready(baseline.audit())),
        "historical_official_test_evaluated": True,
        "historical_official_test_selected": False,
        "teacher_allowed": True,
    }
    return FormalCacheModels(
        current=current,
        baseline1000=baseline,
        current_binding=current_ready,
        baseline1000_binding=teacher_ready,
        source_binding=source_binding,
    )


@dataclass(frozen=True, slots=True)
class LoadedIRSTDTrainSample:
    sample_id: str
    image: np.ndarray
    target: np.ndarray
    image_path: Path
    target_path: Path
    image_file_sha256: str
    target_file_sha256: str


def _find_unique_sample_file(directory: Path, sample_id: str, *, label: str) -> Path:
    _require(directory.is_dir() and not directory.is_symlink(), f"{label} directory differs")
    candidates = [
        directory / f"{sample_id}{suffix}"
        for suffix in _SUPPORTED_IMAGE_SUFFIXES
        if (directory / f"{sample_id}{suffix}").is_file()
        and not (directory / f"{sample_id}{suffix}").is_symlink()
    ]
    _require(len(candidates) == 1, f"{label} file resolution is not unique: {sample_id}")
    return candidates[0].resolve(strict=True)


def load_irstd_train_image_target(
    sample_id: str,
    *,
    bound_ids: frozenset[str],
    data_root: Path = DEFAULT_DATA_ROOT,
) -> LoadedIRSTDTrainSample:
    """Load one manifest-authorized train sample without consulting an index."""

    _require(
        type(sample_id) is str and _SAFE_SAMPLE_ID.fullmatch(sample_id) is not None,
        "sample identifier is unsafe",
    )
    _require(sample_id in bound_ids, "sample is outside the bound official-train IDs")
    root = Path(data_root)
    _require(root.is_dir() and not root.is_symlink(), "data root is missing or unsafe")
    root = root.resolve(strict=True)
    _require(
        root == DEFAULT_DATA_ROOT.resolve(strict=True),
        "production cache requires the frozen repository data root",
    )
    dataset_root = root / DATASET_NAME
    image_path = _find_unique_sample_file(
        dataset_root / "images", sample_id, label="training image"
    )
    target_path = _find_unique_sample_file(
        dataset_root / "masks", sample_id, label="training mask"
    )
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("I"), dtype=np.float32)
    with Image.open(target_path) as handle:
        target_raw = np.asarray(handle, dtype=np.float32)
    if target_raw.ndim == 3:
        target_raw = target_raw[:, :, 0]
    expected_shape = (FULL_CONTEXT_HEIGHT, FULL_CONTEXT_WIDTH)
    _require(
        image.ndim == target_raw.ndim == 2
        and image.shape == target_raw.shape == expected_shape,
        f"{sample_id} is not an exact 512x512 aligned train pair",
    )
    _require(
        bool(np.isfinite(image).all()) and bool(np.isfinite(target_raw).all()),
        f"{sample_id} contains non-finite source pixels",
    )
    _require(
        bool((target_raw >= 0.0).all())
        and bool((target_raw <= np.float32(TARGET_SOURCE_SCALE)).all()),
        f"{sample_id} mask values are outside 0..255",
    )
    image = (image - np.float32(IRSTD_IMAGE_MEAN)) / np.float32(IRSTD_IMAGE_STD)
    # The original SCTransNet loader divides the PNG by 255 and all metric,
    # component and split-statistics code applies a strict >0.5 decision.  A
    # small set of IRSTD masks contains isolated antialiased grayscale pixels;
    # freezing the equivalent source-domain rule here reproduces the existing
    # train-only split statistics while giving the component losses binary IDs.
    target = (
        target_raw > np.float32(TARGET_BINARIZATION_THRESHOLD_SOURCE)
    ).astype(np.float32, copy=False)
    image_ready = np.ascontiguousarray(image[None], dtype=np.float32)
    target_ready = np.ascontiguousarray(target[None], dtype=np.float32)
    _require(
        bool(np.isfinite(image_ready).all()) and bool(np.isfinite(target_ready).all()),
        f"{sample_id} normalization produced non-finite values",
    )
    return LoadedIRSTDTrainSample(
        sample_id=sample_id,
        image=image_ready,
        target=target_ready,
        image_path=image_path,
        target_path=target_path,
        image_file_sha256=file_sha256(image_path),
        target_file_sha256=file_sha256(target_path),
    )


def _cpu_fp32_sample(tensor: torch.Tensor, *, name: str, channels: int) -> np.ndarray:
    _require(isinstance(tensor, torch.Tensor), f"{name} is not a tensor")
    expected = (1, channels, FULL_CONTEXT_HEIGHT, FULL_CONTEXT_WIDTH)
    _require(tuple(tensor.shape) == expected, f"{name} tensor shape differs")
    ready = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()[0]
    _require(bool(np.isfinite(ready).all()), f"{name} contains non-finite values")
    return np.ascontiguousarray(ready, dtype=np.float32)


def forward_formal_current_context(model: nn.Module, image: torch.Tensor) -> Any:
    """Use the public BGCR training/context entry on the actual 512 input."""

    _require(tuple(image.shape) == (1, 1, 512, 512), "Current input geometry differs")
    entry = getattr(model, "forward_for_irstd_training", None)
    _require(callable(entry), "Current model lacks the formal context API")
    with torch.inference_mode():
        routing, context = entry(image)
    _require(
        torch.equal(routing.routed_logits, context.out_logits),
        "identity BGCR output differs from frozen Current logits",
    )
    return context


def forward_baseline1000_logits(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Run the audited formal teacher, whose public result is raw ``outc`` logits."""

    _require(tuple(image.shape) == (1, 1, 512, 512), "teacher input geometry differs")
    _require(not model.training, "teacher model must remain in evaluation mode")
    _require(
        all(not parameter.requires_grad for parameter in model.parameters()),
        "teacher model contains trainable parameters",
    )
    # The authoritative helper retains the public probability and requires
    # ``torch.equal(torch.sigmoid(raw), probability)`` before returning raw.
    from experiments.irstd_baseline_teacher import (
        FrozenIRSTDBaselineTeacher,
        capture_outc_raw_logits,
    )

    _require(
        isinstance(model, FrozenIRSTDBaselineTeacher),
        "teacher must be the formal frozen epoch-1000 wrapper",
    )
    with torch.inference_mode():
        logits = capture_outc_raw_logits(model.model, image)
    _require(isinstance(logits, torch.Tensor), "formal teacher did not return logits")
    _require(tuple(logits.shape) == (1, 1, 512, 512), "teacher logits shape differs")
    _require(bool(torch.isfinite(logits).all()), "teacher logits are non-finite")
    return logits.detach()


def build_cache_sample_arrays(
    *,
    loaded: LoadedIRSTDTrainSample,
    current_context: Any,
    baseline1000_logits: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Materialize one exact per-sample tensor/atlas payload."""

    image = np.ascontiguousarray(loaded.image, dtype=np.float32)
    target = np.ascontiguousarray(loaded.target, dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "image": image,
        "target": target,
        "u1": _cpu_fp32_sample(current_context.local_feature, name="u1", channels=32),
        "z_out": _cpu_fp32_sample(current_context.out_logits, name="z_out", channels=1),
        "z_d0": _cpu_fp32_sample(current_context.d0_logits, name="z_d0", channels=1),
        "z_gt2": _cpu_fp32_sample(current_context.gt2_logits, name="z_gt2", channels=1),
        "z_gt3": _cpu_fp32_sample(current_context.gt3_logits, name="z_gt3", channels=1),
        "z_gt4": _cpu_fp32_sample(current_context.gt4_logits, name="z_gt4", channels=1),
        "z_gt5": _cpu_fp32_sample(current_context.gt5_logits, name="z_gt5", channels=1),
        "baseline1000_logits": _cpu_fp32_sample(
            baseline1000_logits, name="baseline1000_logits", channels=1
        ),
    }
    from experiments.irstd_error_atlas import build_irstd_error_atlas

    atlas = build_irstd_error_atlas(
        current_logits=arrays["z_out"][0],
        target_mask=arrays["target"][0] > np.float32(0.5),
        baseline_logits=arrays["baseline1000_logits"][0],
    )
    for name in (*ATLAS_ID_KEYS, *ATLAS_MASK_KEYS):
        value = getattr(atlas, name)
        dtype = np.int32 if name in ATLAS_ID_KEYS else np.bool_
        arrays[name] = np.ascontiguousarray(value, dtype=dtype)
    validate_cache_sample_arrays(arrays)
    return arrays


def cache_array_contract() -> dict[str, dict[str, Any]]:
    spatial = [FULL_CONTEXT_HEIGHT, FULL_CONTEXT_WIDTH]
    contract: dict[str, dict[str, Any]] = {}
    for name in ("image", "target"):
        contract[name] = {"dtype": "float32", "shape": [1, *spatial]}
    contract["u1"] = {
        "dtype": "float32",
        "shape": [LOCAL_FEATURE_CHANNELS, *spatial],
    }
    for name in (
        "z_out",
        "z_d0",
        "z_gt2",
        "z_gt3",
        "z_gt4",
        "z_gt5",
        "baseline1000_logits",
    ):
        contract[name] = {"dtype": "float32", "shape": [1, *spatial]}
    for name in ATLAS_ID_KEYS:
        contract[name] = {"dtype": "int32", "shape": spatial}
    for name in ATLAS_MASK_KEYS:
        contract[name] = {"dtype": "bool", "shape": spatial}
    return contract


def validate_cache_sample_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    _require(set(arrays) == set(CACHE_ARRAY_KEYS), "cache array keys differ")
    contract = cache_array_contract()
    dtype_by_name = {
        "float32": np.dtype(np.float32),
        "int32": np.dtype(np.int32),
        "bool": np.dtype(np.bool_),
    }
    for name in CACHE_ARRAY_KEYS:
        value = arrays[name]
        _require(isinstance(value, np.ndarray), f"cache {name} is not an array")
        expected = contract[name]
        _require(value.dtype == dtype_by_name[expected["dtype"]], f"cache {name} dtype differs")
        _require(tuple(value.shape) == tuple(expected["shape"]), f"cache {name} shape differs")
        if value.dtype == np.dtype(np.float32):
            _require(bool(np.isfinite(value).all()), f"cache {name} is non-finite")

    target = arrays["target"][0] > np.float32(0.5)
    _require(
        bool(np.logical_or(arrays["target"] == 0.0, arrays["target"] == 1.0).all()),
        "cache target is not binary",
    )
    target_ids = arrays["target_component_ids"]
    rescue_ids = arrays["rescue_component_ids"]
    _require(bool((target_ids >= 0).all()) and bool((rescue_ids >= 0).all()), "atlas IDs are negative")
    _require(bool((target_ids[~target] == 0).all()), "target IDs extend outside target")
    _require(
        bool(np.logical_or(rescue_ids == 0, rescue_ids == target_ids).all()),
        "rescue IDs are not a target-ID subset",
    )
    _require(bool((~arrays["core_target"] | target).all()), "core extends outside target")
    for name in (
        "attached_halo",
        "detached_false_positive",
        "outer_ring",
        "halo_target",
        "far_background",
        "baseline_halo_advantage",
    ):
        _require(bool((~arrays[name] | ~target).all()), f"{name} extends inside target")
    _require(
        bool((~arrays["baseline_rescue"] | target).all()),
        "baseline rescue extends outside target",
    )
    expected_halo = (
        arrays["attached_halo"]
        | arrays["detached_false_positive"]
        | arrays["baseline_halo_advantage"]
    )
    _require(bool(np.array_equal(arrays["halo_target"], expected_halo)), "halo target differs")
    current_prediction = arrays["z_out"][0] > np.float32(0.0)
    baseline_prediction = arrays["baseline1000_logits"][0] > np.float32(0.0)
    _require(
        bool(
            np.array_equal(
                arrays["baseline_rescue"],
                target & baseline_prediction & ~current_prediction,
            )
        ),
        "baseline rescue does not replay",
    )
    _require(
        bool(
            np.array_equal(
                arrays["baseline_halo_advantage"],
                ~target & current_prediction & ~baseline_prediction,
            )
        ),
        "baseline halo advantage does not replay",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(not destination.is_symlink(), f"destination is a symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(destination: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(destination, _canonical_json_bytes(dict(value), newline=True))


def _atomic_write_npz(destination: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(not destination.exists() and not destination.is_symlink(), "cache sample already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            # Store-only NPZ is intentional: this ~36 GiB cache is read every
            # epoch, and avoiding DEFLATE keeps formal build/preload CPU cost low.
            np.savez(handle, **{name: arrays[name] for name in CACHE_ARRAY_KEYS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    _require(path.is_file() and not path.is_symlink(), f"cache sample is missing: {path}")
    try:
        with zipfile.ZipFile(path, "r") as container:
            entries = container.infolist()
            expected_entries = {f"{name}.npy" for name in CACHE_ARRAY_KEYS}
            _require(
                len(entries) == len(expected_entries)
                and {entry.filename for entry in entries} == expected_entries,
                "NPZ container entries differ",
            )
            _require(
                all(
                    entry.compress_type == zipfile.ZIP_STORED
                    and entry.flag_bits & 0x1 == 0
                    for entry in entries
                ),
                "NPZ container must be uncompressed and unencrypted",
            )
        with np.load(path, allow_pickle=False) as archive:
            _require(
                len(archive.files) == len(CACHE_ARRAY_KEYS)
                and set(archive.files) == set(CACHE_ARRAY_KEYS),
                "NPZ keys differ",
            )
            arrays = {name: np.asarray(archive[name]) for name in CACHE_ARRAY_KEYS}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise IRSTDBGCRCacheError(f"cannot read cache sample: {path}") from error
    validate_cache_sample_arrays(arrays)
    return arrays


def _record_semantic_sha(record: Mapping[str, Any]) -> str:
    unsigned = dict(record)
    unsigned.pop("record_semantic_sha256", None)
    return canonical_sha256(unsigned)


def write_cache_sample_atomic(
    staging_root: Path,
    *,
    index: int,
    sample_id: str,
    split_membership: str,
    arrays: Mapping[str, np.ndarray],
    source: Mapping[str, Any],
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Write one NPZ then one signed sidecar; a one-sided pair is never resumed."""

    _require(type(index) is int and index >= 0, "sample index is invalid")
    _require(_SAFE_SAMPLE_ID.fullmatch(sample_id) is not None, "sample ID is unsafe")
    _require(
        split_membership in {"development_train", "internal_validation"},
        "sample split membership differs",
    )
    validate_cache_sample_arrays(arrays)
    stem = f"{index:06d}"
    sample_relative = Path("samples") / f"{stem}.npz"
    record_relative = Path("records") / f"{stem}.json"
    sample_path = staging_root / sample_relative
    record_path = staging_root / record_relative
    _require(
        not sample_path.exists()
        and not sample_path.is_symlink()
        and not record_path.exists()
        and not record_path.is_symlink(),
        "sample/record already exists",
    )
    _atomic_write_npz(sample_path, arrays)
    persisted = _load_npz_arrays(sample_path)
    array_records = {
        name: {
            "dtype": cache_array_contract()[name]["dtype"],
            "shape": cache_array_contract()[name]["shape"],
            "semantic_sha256": array_semantic_sha256(persisted[name]),
        }
        for name in CACHE_ARRAY_KEYS
    }
    record: dict[str, Any] = {
        "schema": SAMPLE_RECORD_SCHEMA,
        "index": index,
        "sample_id": sample_id,
        "split_membership": split_membership,
        "cache_relative_path": sample_relative.as_posix(),
        "cache_file_sha256": file_sha256(sample_path),
        "cache_bytes": sample_path.stat().st_size,
        "arrays": array_records,
        "source": dict(source),
        **OFFICIAL_FALSE_FLAGS,
    }
    record["record_semantic_sha256"] = _record_semantic_sha(record)
    _atomic_write_json(record_path, record)
    observed_record = _load_json_object(record_path, label="new cache sample record")
    _require(observed_record == record, "atomically written sample record differs")
    if source_root is not None:
        _validate_sample_source(
            source,
            sample_id=sample_id,
            source_root=source_root,
        )
    return observed_record


def _validate_sample_pair(
    root: Path,
    *,
    index: int,
    sample_id: str,
    split_membership: str,
    source_root: Path | None = None,
) -> dict[str, Any]:
    stem = f"{index:06d}"
    sample_path = root / "samples" / f"{stem}.npz"
    record_path = root / "records" / f"{stem}.json"
    sample_exists = sample_path.exists() or sample_path.is_symlink()
    record_exists = record_path.exists() or record_path.is_symlink()
    _require(sample_exists == record_exists, "one-sided cache sample/record pair")
    _require(sample_exists, "cache sample/record pair is absent")
    record = _load_json_object(record_path, label="cache sample record")
    _require(record.get("schema") == SAMPLE_RECORD_SCHEMA, "sample record schema differs")
    _require(record.get("index") == index, "sample record index differs")
    _require(record.get("sample_id") == sample_id, "sample record ID differs")
    _require(
        record.get("split_membership") == split_membership,
        "sample record split membership differs",
    )
    _validate_official_false_flags(record, label="sample record")
    _require(
        record.get("record_semantic_sha256") == _record_semantic_sha(record),
        "sample record semantic SHA differs",
    )
    _require(
        record.get("cache_relative_path") == f"samples/{stem}.npz",
        "sample relative path differs",
    )
    _require(record.get("cache_file_sha256") == file_sha256(sample_path), "sample file SHA differs")
    _require(record.get("cache_bytes") == sample_path.stat().st_size, "sample byte count differs")
    arrays = _load_npz_arrays(sample_path)
    array_records = record.get("arrays")
    _require(isinstance(array_records, Mapping), "sample array records differ")
    _require(set(array_records) == set(CACHE_ARRAY_KEYS), "sample array record keys differ")
    contract = cache_array_contract()
    for name in CACHE_ARRAY_KEYS:
        item = array_records[name]
        _require(isinstance(item, Mapping), f"sample {name} record differs")
        _require(item.get("dtype") == contract[name]["dtype"], f"sample {name} record dtype differs")
        _require(item.get("shape") == contract[name]["shape"], f"sample {name} record shape differs")
        _require(
            item.get("semantic_sha256") == array_semantic_sha256(arrays[name]),
            f"sample {name} semantic SHA differs",
        )
    source = record.get("source")
    _require(isinstance(source, Mapping), "sample source record differs")
    for name in ("image_file_sha256", "target_file_sha256"):
        _require_sha256(source.get(name), name=f"sample source {name}")
    if source_root is not None:
        _validate_sample_source(
            source,
            sample_id=sample_id,
            source_root=source_root,
        )
    return record


def _validate_sample_source(
    source: Mapping[str, Any],
    *,
    sample_id: str,
    source_root: Path,
) -> None:
    root = Path(source_root)
    _require(root.is_dir() and not root.is_symlink(), "sample source root is unsafe")
    root = root.resolve(strict=True)
    _require(root == DEFAULT_DATA_ROOT.resolve(strict=True), "sample source root differs")
    expected_parents = {
        "image_relative_path": Path(DATASET_NAME) / "images",
        "target_relative_path": Path(DATASET_NAME) / "masks",
    }
    for path_field, expected_parent in expected_parents.items():
        raw = source.get(path_field)
        _require(type(raw) is str and bool(raw), f"sample source {path_field} differs")
        relative = Path(raw)
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.parent == expected_parent
            and relative.stem == sample_id
            and relative.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES,
            f"sample source {path_field} escapes the train pair",
        )
        resolved = root / relative
        _require(
            resolved.is_file() and not resolved.is_symlink(),
            f"sample source file is missing or unsafe: {relative}",
        )
        sha_field = (
            "image_file_sha256"
            if path_field == "image_relative_path"
            else "target_file_sha256"
        )
        _require(
            file_sha256(resolved) == source.get(sha_field),
            f"sample source file SHA changed: {relative}",
        )


def _source_record(loaded: LoadedIRSTDTrainSample, data_root: Path) -> dict[str, Any]:
    root = Path(data_root).resolve(strict=True)
    return {
        "image_relative_path": loaded.image_path.relative_to(root).as_posix(),
        "target_relative_path": loaded.target_path.relative_to(root).as_posix(),
        "image_file_sha256": loaded.image_file_sha256,
        "target_file_sha256": loaded.target_file_sha256,
    }


def _build_identity(
    *,
    split: BoundIRSTDTrainSplit,
    models: FormalCacheModels,
    data_root: Path,
    determinism: Mapping[str, Any],
) -> dict[str, Any]:
    atlas_source = REPO_ROOT / "experiments/irstd_error_atlas.py"
    matcher_source = REPO_ROOT / "experiments/component_matching_v2.py"
    core_source = REPO_ROOT / "model/irstd_core_ring_repair.py"
    cache_source = Path(__file__).resolve(strict=True)
    identity: dict[str, Any] = {
        "schema": BUILD_IDENTITY_SCHEMA,
        "dataset": DATASET_NAME,
        "sample_count": OFFICIAL_TRAIN_COUNT,
        "geometry": {
            "context_height": FULL_CONTEXT_HEIGHT,
            "context_width": FULL_CONTEXT_WIDTH,
            "source_is_full_image": True,
            "padding": None,
            "resize": None,
        },
        "container": {
            "format": "npz",
            "container_compression": "store",
        },
        "container_compression": "store",
        "array_contract": cache_array_contract(),
        "normalization": {"mean": IRSTD_IMAGE_MEAN, "std": IRSTD_IMAGE_STD},
        "target_binarization": {
            "source_scale": TARGET_SOURCE_SCALE,
            "source_threshold": TARGET_BINARIZATION_THRESHOLD_SOURCE,
            "normalized_threshold": TARGET_BINARIZATION_THRESHOLD_NORMALIZED,
            "comparison": TARGET_BINARIZATION_COMPARISON,
            "reason": "match_existing_strict_gt_0_5_component_and_metric_contract",
        },
        "determinism": dict(determinism),
        "data_root": str(Path(data_root).resolve(strict=True)),
        "split": {
            "projection_path": str(split.projection_path),
            "projection_file_sha256": split.projection_file_sha256,
            "projection_sha256": SPLIT_PROJECTION_SHA256,
            "source_manifest_path": str(split.source_manifest_path),
            "source_manifest_file_sha256": split.source_manifest_file_sha256,
            "canonical_split_sha256": CANONICAL_SPLIT_SHA256,
            "official_train_index_sha256": OFFICIAL_TRAIN_INDEX_SHA256,
            "official_train_ids_sha256": OFFICIAL_TRAIN_IDS_SHA256,
            "development_train_ids_sha256": DEVELOPMENT_TRAIN_IDS_SHA256,
            "internal_validation_ids_sha256": INTERNAL_VALIDATION_IDS_SHA256,
            "official_train_ids": list(split.official_train_ids),
            "development_train_ids": list(split.development_train_ids),
            "internal_validation_ids": list(split.internal_validation_ids),
            "counts": {
                "official_train": OFFICIAL_TRAIN_COUNT,
                "development_train": DEVELOPMENT_TRAIN_COUNT,
                "internal_validation": INTERNAL_VALIDATION_COUNT,
            },
        },
        "models": {
            "current": dict(models.current_binding),
            "baseline1000": dict(models.baseline1000_binding),
        },
        "sources": {
            **dict(models.source_binding),
            "cache_builder": {
                "path": str(cache_source),
                "file_sha256": file_sha256(cache_source),
            },
            "error_atlas": {
                "path": str(atlas_source.resolve(strict=True)),
                "file_sha256": file_sha256(atlas_source),
            },
            "component_matcher": {
                "path": str(matcher_source.resolve(strict=True)),
                "file_sha256": file_sha256(matcher_source),
            },
            "repair_head": {
                "path": str(core_source.resolve(strict=True)),
                "file_sha256": file_sha256(core_source),
            },
        },
        **OFFICIAL_FALSE_FLAGS,
        "performance_acceptance_margin": None,
        "margin": None,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _prepare_staging(output_root: Path, identity: Mapping[str, Any]) -> tuple[Path, bool]:
    output = Path(output_root)
    _require(output.name not in {"", ".", ".."}, "output root name is unsafe")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(not output.is_symlink(), "output root is a symlink")
    if output.exists():
        _require(output.is_dir(), "output root exists but is not a directory")
        validate_committed_cache(output, expected_identity=identity)
        return output, True
    staging = output.parent / f".{output.name}.incomplete"
    _require(not staging.is_symlink(), "incomplete cache root is a symlink")
    if not staging.exists():
        staging.mkdir(mode=0o700)
        (staging / "samples").mkdir(mode=0o700)
        (staging / "records").mkdir(mode=0o700)
        _atomic_write_json(staging / "identity.json", dict(identity))
        _fsync_directory(staging)
    else:
        _require(staging.is_dir(), "incomplete cache root is not a directory")
        _require(
            {path.name for path in staging.iterdir()}
            <= {"identity.json", "samples", "records", "manifest.json", "COMMITTED.json"},
            "incomplete cache contains unexpected top-level entries",
        )
        observed = _load_json_object(staging / "identity.json", label="build identity")
        _require(observed == dict(identity), "resume identity differs; refusing mixed cache")
        _require(
            (staging / "samples").is_dir()
            and not (staging / "samples").is_symlink()
            and (staging / "records").is_dir()
            and not (staging / "records").is_symlink(),
            "resume directories are missing or unsafe",
        )
        manifest_exists = (staging / "manifest.json").exists()
        commit_exists = (staging / "COMMITTED.json").exists()
        _require(manifest_exists == commit_exists, "incomplete finalization is fail-closed")
        if manifest_exists:
            validate_committed_cache(staging, expected_identity=identity)
            os.replace(staging, output)
            _fsync_directory(output.parent)
            return output, True
    return staging, False


def _manifest_item(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    index = int(record["index"])
    record_path = root / "records" / f"{index:06d}.json"
    return {
        **dict(record),
        "record_relative_path": f"records/{index:06d}.json",
        "record_file_sha256": file_sha256(record_path),
        "record_bytes": record_path.stat().st_size,
    }


def _finalize_staging(
    staging: Path,
    output: Path,
    *,
    identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(len(records) == OFFICIAL_TRAIN_COUNT, "cannot finalize a partial cache")
    identity_path = staging / "identity.json"
    _require(
        _load_json_object(identity_path, label="build identity") == dict(identity),
        "staged identity differs before finalization",
    )
    identity_file_sha = file_sha256(identity_path)
    manifest: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "status": "complete",
        "identity": dict(identity),
        "identity_sha256": identity["identity_sha256"],
        "identity_file_sha256": identity_file_sha,
        "identity_file_bytes": identity_path.stat().st_size,
        "sample_count": len(records),
        "container_compression": "store",
        "items": [_manifest_item(staging, record) for record in records],
        **OFFICIAL_FALSE_FLAGS,
        "performance_acceptance_margin": None,
        "margin": None,
    }
    manifest["manifest_semantic_sha256"] = canonical_sha256(manifest)
    _atomic_write_json(staging / "manifest.json", manifest)
    manifest_file_sha = file_sha256(staging / "manifest.json")
    commit = {
        "schema": COMMIT_SCHEMA,
        "status": "complete",
        "manifest_file_sha256": manifest_file_sha,
        "manifest_semantic_sha256": manifest["manifest_semantic_sha256"],
        "identity_sha256": identity["identity_sha256"],
        "identity_file_sha256": identity_file_sha,
        "sample_count": len(records),
        "container_compression": "store",
        **OFFICIAL_FALSE_FLAGS,
        "performance_acceptance_margin": None,
        "margin": None,
    }
    _atomic_write_json(staging / "COMMITTED.json", commit)
    _fsync_directory(staging / "samples")
    _fsync_directory(staging / "records")
    _fsync_directory(staging)
    _require(not output.exists() and not output.is_symlink(), "output appeared during build")
    os.replace(staging, output)
    _fsync_directory(output.parent)
    return validate_committed_cache(output, expected_identity=identity)


def _validate_official_false_flags(payload: Mapping[str, Any], *, label: str) -> None:
    for name, expected in OFFICIAL_FALSE_FLAGS.items():
        _require(payload.get(name) is expected, f"{label} {name} must remain false")


def _validate_cache_identity(identity: Mapping[str, Any]) -> None:
    _require(identity.get("schema") == BUILD_IDENTITY_SCHEMA, "identity schema differs")
    _require(identity.get("dataset") == DATASET_NAME, "identity dataset differs")
    _require(identity.get("sample_count") == OFFICIAL_TRAIN_COUNT, "identity count differs")
    _require(
        identity.get("geometry")
        == {
            "context_height": FULL_CONTEXT_HEIGHT,
            "context_width": FULL_CONTEXT_WIDTH,
            "source_is_full_image": True,
            "padding": None,
            "resize": None,
        },
        "identity full-context geometry differs",
    )
    _require(
        identity.get("container")
        == {"format": "npz", "container_compression": "store"},
        "identity NPZ container contract differs",
    )
    _require(
        identity.get("container_compression") == "store",
        "identity container compression differs",
    )
    _require(identity.get("array_contract") == cache_array_contract(), "identity arrays differ")
    _require(
        identity.get("normalization")
        == {"mean": IRSTD_IMAGE_MEAN, "std": IRSTD_IMAGE_STD},
        "identity normalization differs",
    )
    _require(
        identity.get("target_binarization")
        == {
            "source_scale": TARGET_SOURCE_SCALE,
            "source_threshold": TARGET_BINARIZATION_THRESHOLD_SOURCE,
            "normalized_threshold": TARGET_BINARIZATION_THRESHOLD_NORMALIZED,
            "comparison": TARGET_BINARIZATION_COMPARISON,
            "reason": "match_existing_strict_gt_0_5_component_and_metric_contract",
        },
        "identity target binarization differs",
    )
    _require(
        identity.get("determinism") == expected_determinism_manifest(),
        "identity determinism manifest differs",
    )
    _validate_official_false_flags(identity, label="identity")
    _require(
        identity.get("performance_acceptance_margin") is None
        and identity.get("margin") is None,
        "identity must not contain a performance margin",
    )
    data_root = Path(str(identity.get("data_root")))
    _require(
        data_root.is_dir()
        and not data_root.is_symlink()
        and data_root.resolve(strict=True) == DEFAULT_DATA_ROOT.resolve(strict=True),
        "identity data root differs",
    )

    split = identity.get("split")
    _require(isinstance(split, Mapping), "identity split binding is missing")
    expected_split_scalars = {
        "projection_path": str(SPLIT_PROJECTION_PATH.resolve(strict=True)),
        "projection_file_sha256": SPLIT_PROJECTION_FILE_SHA256,
        "projection_sha256": SPLIT_PROJECTION_SHA256,
        "source_manifest_path": str(SOURCE_SPLIT_MANIFEST_PATH.resolve(strict=True)),
        "source_manifest_file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "canonical_split_sha256": CANONICAL_SPLIT_SHA256,
        "official_train_index_sha256": OFFICIAL_TRAIN_INDEX_SHA256,
        "official_train_ids_sha256": OFFICIAL_TRAIN_IDS_SHA256,
        "development_train_ids_sha256": DEVELOPMENT_TRAIN_IDS_SHA256,
        "internal_validation_ids_sha256": INTERNAL_VALIDATION_IDS_SHA256,
        "counts": {
            "official_train": OFFICIAL_TRAIN_COUNT,
            "development_train": DEVELOPMENT_TRAIN_COUNT,
            "internal_validation": INTERNAL_VALIDATION_COUNT,
        },
    }
    for name, expected in expected_split_scalars.items():
        _require(split.get(name) == expected, f"identity split {name} differs")
    _require(
        file_sha256(SPLIT_PROJECTION_PATH) == SPLIT_PROJECTION_FILE_SHA256
        and file_sha256(SOURCE_SPLIT_MANIFEST_PATH)
        == SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "bound split source files changed",
    )
    official = _validated_ids(split.get("official_train_ids"), name="identity official IDs")
    development = _validated_ids(
        split.get("development_train_ids"), name="identity development IDs"
    )
    internal = _validated_ids(
        split.get("internal_validation_ids"), name="identity internal IDs"
    )
    _require(
        len(official) == OFFICIAL_TRAIN_COUNT
        and len(development) == DEVELOPMENT_TRAIN_COUNT
        and len(internal) == INTERNAL_VALIDATION_COUNT,
        "identity split ID counts differ",
    )
    _require(
        ordered_ids_sha256(official) == OFFICIAL_TRAIN_IDS_SHA256
        and ordered_ids_sha256(development) == DEVELOPMENT_TRAIN_IDS_SHA256
        and ordered_ids_sha256(internal) == INTERNAL_VALIDATION_IDS_SHA256,
        "identity split ordered-ID hashes differ",
    )
    _require(
        not set(development) & set(internal)
        and set(development) | set(internal) == set(official),
        "identity split partition differs",
    )

    models = identity.get("models")
    _require(isinstance(models, Mapping), "identity model bindings are missing")
    current = models.get("current")
    teacher = models.get("baseline1000")
    _require(isinstance(current, Mapping) and isinstance(teacher, Mapping), "model bindings differ")
    expected_current = {
        "path": str(CURRENT_CHECKPOINT_PATH.resolve(strict=True)),
        "file_sha256": CURRENT_CHECKPOINT_FILE_SHA256,
        "file_bytes": CURRENT_CHECKPOINT_BYTES,
        "epoch": CURRENT_CHECKPOINT_EPOCH,
        "training_state_keys": CURRENT_TRAINING_STATE_KEYS,
        "training_state_tensor_mapping_sha256": (
            CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256
        ),
        "training_state_tensor_mapping_hash_algorithm": "tensor_mapping_sha256",
        "training_state_semantic_hash_algorithm": "state_semantic_sha256",
        "inference_state_keys": CURRENT_INFERENCE_STATE_KEYS,
        "inference_state_tensor_mapping_hash_algorithm": "tensor_mapping_sha256",
        "inference_state_semantic_hash_algorithm": "state_semantic_sha256",
        "formal_builder": "build_formal_irstd_bgcr_model",
        "formal_context_api": "forward_for_irstd_training",
        "seed": 42,
        "repair_initialization_seed": 42,
    }
    for name, expected in expected_current.items():
        _require(current.get(name) == expected, f"identity Current {name} differs")
    _require(
        file_sha256(CURRENT_CHECKPOINT_PATH) == CURRENT_CHECKPOINT_FILE_SHA256
        and CURRENT_CHECKPOINT_PATH.stat().st_size == CURRENT_CHECKPOINT_BYTES,
        "bound Current checkpoint changed",
    )
    _require_sha256(
        current.get("training_state_semantic_sha256"),
        name="Current training semantic state SHA",
    )
    _require_sha256(
        current.get("inference_state_tensor_mapping_sha256"),
        name="Current inference tensor-mapping state SHA",
    )
    _require(
        current.get("inference_state_semantic_sha256")
        == CURRENT_INFERENCE_STATE_SEMANTIC_SHA256,
        "Current inference semantic SHA differs",
    )
    _require_sha256(current.get("builder_metadata_sha256"), name="Current builder metadata SHA")
    expected_teacher = {
        "path": str(BASELINE_TEACHER_CHECKPOINT_PATH.resolve(strict=True)),
        "file_sha256": BASELINE_TEACHER_CHECKPOINT_FILE_SHA256,
        "file_bytes": BASELINE_TEACHER_CHECKPOINT_BYTES,
        "epoch": BASELINE_TEACHER_EPOCH,
        "state_keys": BASELINE_TEACHER_STATE_KEYS,
        "raw_state_semantic_sha256": (
            BASELINE_TEACHER_RAW_STATE_SEMANTIC_SHA256
        ),
        "normalized_state_semantic_sha256": (
            BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256
        ),
        "state_semantic_hash_algorithm": "state_semantic_sha256",
        "raw_state_prefix": BASELINE_TEACHER_RAW_STATE_PREFIX,
        "trainable": False,
        "enabled": True,
        "operational_reference_epoch": OPERATIONAL_REFERENCE_EPOCH,
        "operational_reference_is_teacher": False,
        "builder": "experiments.irstd_baseline_teacher.build_formal_teacher",
        "historical_official_test_evaluated": True,
        "historical_official_test_selected": False,
        "teacher_allowed": True,
    }
    for name, expected in expected_teacher.items():
        _require(teacher.get(name) == expected, f"identity teacher {name} differs")
    _require(
        file_sha256(BASELINE_TEACHER_CHECKPOINT_PATH)
        == BASELINE_TEACHER_CHECKPOINT_FILE_SHA256
        and BASELINE_TEACHER_CHECKPOINT_PATH.stat().st_size
        == BASELINE_TEACHER_CHECKPOINT_BYTES,
        "bound Baseline teacher checkpoint changed",
    )
    _require_sha256(teacher.get("builder_audit_sha256"), name="teacher builder audit SHA")

    sources = identity.get("sources")
    _require(isinstance(sources, Mapping) and bool(sources), "identity sources are missing")
    for name, record in sources.items():
        _require(isinstance(record, Mapping), f"identity source {name} differs")
        source_path = Path(str(record.get("path")))
        source_sha = _require_sha256(record.get("file_sha256"), name=f"source {name} SHA")
        _require(
            source_path.is_file()
            and not source_path.is_symlink()
            and file_sha256(source_path) == source_sha,
            f"identity source changed: {name}",
        )


def validate_committed_cache(
    root: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = Path(root)
    _require(candidate.is_dir() and not candidate.is_symlink(), "cache root is unsafe")
    _require(
        {path.name for path in candidate.iterdir()}
        == {"identity.json", "samples", "records", "manifest.json", "COMMITTED.json"},
        "cache root entries differ",
    )
    manifest_path = candidate / "manifest.json"
    commit_path = candidate / "COMMITTED.json"
    manifest = _load_json_object(manifest_path, label="cache manifest")
    commit = _load_json_object(commit_path, label="cache commit")
    _require(manifest.get("schema") == CACHE_SCHEMA, "cache manifest schema differs")
    _require(manifest.get("status") == "complete", "cache is incomplete")
    _require(commit.get("schema") == COMMIT_SCHEMA, "cache commit schema differs")
    _require(commit.get("status") == "complete", "cache commit is incomplete")
    _require(
        manifest.get("container_compression") == "store"
        and commit.get("container_compression") == "store",
        "cache container compression differs",
    )
    _validate_official_false_flags(manifest, label="manifest")
    _validate_official_false_flags(commit, label="commit")
    _require(
        manifest.get("margin") is None
        and commit.get("margin") is None
        and manifest.get("performance_acceptance_margin") is None
        and commit.get("performance_acceptance_margin") is None,
        "cache performance margin must be null",
    )
    unsigned = dict(manifest)
    declared_manifest_semantic = unsigned.pop("manifest_semantic_sha256", None)
    _require(
        declared_manifest_semantic == canonical_sha256(unsigned),
        "cache manifest semantic SHA differs",
    )
    _require(
        commit.get("manifest_file_sha256") == file_sha256(manifest_path),
        "cache manifest file SHA differs",
    )
    _require(
        commit.get("manifest_semantic_sha256") == declared_manifest_semantic,
        "cache commit semantic binding differs",
    )
    identity = manifest.get("identity")
    _require(isinstance(identity, Mapping), "cache identity is missing")
    identity_file = candidate / "identity.json"
    _require(
        identity_file.is_file()
        and not identity_file.is_symlink()
        and manifest.get("identity_file_bytes") == identity_file.stat().st_size
        and manifest.get("identity_file_sha256") == file_sha256(identity_file)
        == commit.get("identity_file_sha256")
        and _load_json_object(identity_file, label="cache identity file") == dict(identity),
        "cache identity file binding differs",
    )
    _validate_cache_identity(identity)
    _require(
        identity.get("identity_sha256") == manifest.get("identity_sha256")
        == commit.get("identity_sha256"),
        "cache identity bindings differ",
    )
    identity_unsigned = dict(identity)
    declared_identity_sha = identity_unsigned.pop("identity_sha256", None)
    _require(declared_identity_sha == canonical_sha256(identity_unsigned), "identity SHA differs")
    if expected_identity is not None:
        _require(dict(identity) == dict(expected_identity), "cache identity differs")
    items = manifest.get("items")
    _require(isinstance(items, list), "cache items are missing")
    _require(
        len(items) == manifest.get("sample_count") == commit.get("sample_count")
        == OFFICIAL_TRAIN_COUNT,
        "cache item count differs",
    )
    _require(
        (candidate / "samples").is_dir()
        and not (candidate / "samples").is_symlink()
        and (candidate / "records").is_dir()
        and not (candidate / "records").is_symlink(),
        "cache sample/record directories are unsafe",
    )
    _require(
        {path.name for path in (candidate / "samples").iterdir()}
        == {f"{index:06d}.npz" for index in range(OFFICIAL_TRAIN_COUNT)}
        and {path.name for path in (candidate / "records").iterdir()}
        == {f"{index:06d}.json" for index in range(OFFICIAL_TRAIN_COUNT)},
        "cache sample/record directory entries differ",
    )
    split_identity = identity["split"]
    official_ids = tuple(split_identity["official_train_ids"])
    development_ids = frozenset(split_identity["development_train_ids"])
    source_root = Path(str(identity["data_root"]))
    for index, item in enumerate(items):
        _require(isinstance(item, Mapping), f"cache item {index} differs")
        expected_sample_id = official_ids[index]
        expected_membership = (
            "development_train"
            if expected_sample_id in development_ids
            else "internal_validation"
        )
        _require(item.get("sample_id") == expected_sample_id, f"cache item {index} ID differs")
        _require(
            item.get("split_membership") == expected_membership,
            f"cache item {index} membership differs",
        )
        record = _validate_sample_pair(
            candidate,
            index=index,
            sample_id=expected_sample_id,
            split_membership=expected_membership,
            source_root=source_root,
        )
        expected_item = _manifest_item(candidate, record)
        _require(dict(item) == expected_item, f"cache manifest item {index} differs")
    return manifest


def build_irstd_frozen_context_cache(
    output_root: Path,
    *,
    device: torch.device | str,
    data_root: Path = DEFAULT_DATA_ROOT,
    projection_path: Path = SPLIT_PROJECTION_PATH,
    source_manifest_path: Path = SOURCE_SPLIT_MANIFEST_PATH,
    current_checkpoint_path: Path = CURRENT_CHECKPOINT_PATH,
    teacher_checkpoint_path: Path = BASELINE_TEACHER_CHECKPOINT_PATH,
) -> dict[str, Any]:
    """Create or strictly resume the complete 800-sample frozen cache."""

    determinism = configure_determinism(FORMAL_SEED)
    split = load_bound_irstd_train_split(
        projection_path=projection_path,
        source_manifest_path=source_manifest_path,
    )
    root = Path(data_root)
    _require(root.is_dir() and not root.is_symlink(), "data root is unsafe")
    root = root.resolve(strict=True)
    _require(root == DEFAULT_DATA_ROOT.resolve(strict=True), "data root differs")
    models = load_formal_bgcr_cache_models(
        device=device,
        current_checkpoint_path=current_checkpoint_path,
        teacher_checkpoint_path=teacher_checkpoint_path,
    )
    identity = _build_identity(
        split=split,
        models=models,
        data_root=root,
        determinism=determinism,
    )
    output = Path(output_root)
    staging, already_complete = _prepare_staging(output, identity)
    if already_complete:
        return validate_committed_cache(output, expected_identity=identity)

    bound_ids = frozenset(split.official_train_ids)
    development = frozenset(split.development_train_ids)
    permitted_samples = {
        f"{index:06d}.npz" for index in range(OFFICIAL_TRAIN_COUNT)
    }
    permitted_records = {
        f"{index:06d}.json" for index in range(OFFICIAL_TRAIN_COUNT)
    }
    _require(
        {path.name for path in (staging / "samples").iterdir()} <= permitted_samples,
        "resume sample directory contains unexpected files",
    )
    _require(
        {path.name for path in (staging / "records").iterdir()} <= permitted_records,
        "resume record directory contains unexpected files",
    )
    records: list[dict[str, Any]] = []
    for index, sample_id in enumerate(split.official_train_ids):
        membership = (
            "development_train" if sample_id in development else "internal_validation"
        )
        sample_path = staging / "samples" / f"{index:06d}.npz"
        record_path = staging / "records" / f"{index:06d}.json"
        sample_present = sample_path.exists() or sample_path.is_symlink()
        record_present = record_path.exists() or record_path.is_symlink()
        _require(sample_present == record_present, "resume found a one-sided sample pair")
        if sample_present:
            records.append(
                _validate_sample_pair(
                    staging,
                    index=index,
                    sample_id=sample_id,
                    split_membership=membership,
                    source_root=root,
                )
            )
            continue

        loaded = load_irstd_train_image_target(
            sample_id,
            bound_ids=bound_ids,
            data_root=root,
        )
        image = torch.from_numpy(loaded.image[None]).to(
            device=torch.device(device), dtype=torch.float32
        )
        context = forward_formal_current_context(models.current, image)
        teacher_logits = forward_baseline1000_logits(models.baseline1000, image)
        arrays = build_cache_sample_arrays(
            loaded=loaded,
            current_context=context,
            baseline1000_logits=teacher_logits,
        )
        records.append(
            write_cache_sample_atomic(
                staging,
                index=index,
                sample_id=sample_id,
                split_membership=membership,
                arrays=arrays,
                source=_source_record(loaded, root),
                source_root=root,
            )
        )

    _require(
        {path.name for path in (staging / "samples").iterdir()} == permitted_samples,
        "sample directory contains missing or unexpected files",
    )
    _require(
        {path.name for path in (staging / "records").iterdir()} == permitted_records,
        "record directory contains missing or unexpected files",
    )
    return _finalize_staging(
        staging,
        output,
        identity=identity,
        records=records,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True, help="explicit torch device, e.g. cuda:2")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-projection", type=Path, default=SPLIT_PROJECTION_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SOURCE_SPLIT_MANIFEST_PATH)
    parser.add_argument("--current-checkpoint", type=Path, default=CURRENT_CHECKPOINT_PATH)
    parser.add_argument(
        "--teacher-checkpoint", type=Path, default=BASELINE_TEACHER_CHECKPOINT_PATH
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_irstd_frozen_context_cache(
        args.output,
        device=args.device,
        data_root=args.data_root,
        projection_path=args.split_projection,
        source_manifest_path=args.split_manifest,
        current_checkpoint_path=args.current_checkpoint,
        teacher_checkpoint_path=args.teacher_checkpoint,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATLAS_ID_KEYS",
    "ATLAS_MASK_KEYS",
    "BASELINE_TEACHER_CHECKPOINT_FILE_SHA256",
    "BASELINE_TEACHER_CHECKPOINT_BYTES",
    "BASELINE_TEACHER_CHECKPOINT_PATH",
    "BASELINE_TEACHER_EPOCH",
    "BASELINE_TEACHER_STATE_KEYS",
    "BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256",
    "BASELINE_TEACHER_RAW_STATE_SEMANTIC_SHA256",
    "BUILD_IDENTITY_SCHEMA",
    "BoundIRSTDTrainSplit",
    "CACHE_ARRAY_KEYS",
    "CACHE_SCHEMA",
    "COMMIT_SCHEMA",
    "CURRENT_CHECKPOINT_FILE_SHA256",
    "CURRENT_CHECKPOINT_BYTES",
    "CURRENT_CHECKPOINT_PATH",
    "CURRENT_INFERENCE_STATE_KEYS",
    "CURRENT_INFERENCE_STATE_SEMANTIC_SHA256",
    "CURRENT_TRAINING_STATE_KEYS",
    "CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256",
    "DATASET_NAME",
    "FormalCacheModels",
    "FORMAL_SEED",
    "FULL_CONTEXT_HEIGHT",
    "FULL_CONTEXT_WIDTH",
    "IRSTDBGCRCacheError",
    "IRSTD_TRAIN_SPLIT_AUTHORITY",
    "LoadedIRSTDTrainSample",
    "OFFICIAL_TEST_ACCESSED",
    "OFFICIAL_FALSE_FLAGS",
    "OFFICIAL_TRAIN_COUNT",
    "OPERATIONAL_REFERENCE_EPOCH",
    "PERFORMANCE_ACCEPTANCE_MARGIN",
    "PERFORMANCE_MARGIN",
    "SAMPLE_RECORD_SCHEMA",
    "TARGET_BINARIZATION_COMPARISON",
    "TARGET_BINARIZATION_THRESHOLD_NORMALIZED",
    "TARGET_BINARIZATION_THRESHOLD_SOURCE",
    "TARGET_SOURCE_SCALE",
    "TrainSplitAuthority",
    "array_semantic_sha256",
    "build_cache_sample_arrays",
    "build_irstd_frozen_context_cache",
    "cache_array_contract",
    "canonical_sha256",
    "configure_determinism",
    "expected_determinism_manifest",
    "file_sha256",
    "forward_baseline1000_logits",
    "forward_formal_current_context",
    "load_bound_baseline1000_teacher_state",
    "load_bound_current_training_state",
    "load_bound_irstd_train_split",
    "load_formal_bgcr_cache_models",
    "load_irstd_train_image_target",
    "normalize_baseline_teacher_state",
    "ordered_ids_sha256",
    "state_semantic_sha256",
    "tensor_mapping_sha256",
    "validate_cache_sample_arrays",
    "validate_committed_cache",
    "validate_train_split_payload",
    "write_cache_sample_atomic",
]
