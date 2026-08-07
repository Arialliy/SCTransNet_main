#!/usr/bin/env python3
"""Formal cache-only trainer for the IRSTD frozen-main BGCR repair head.

This runner has exactly two modes.  ``fold`` trains one of the three frozen
OOF folds for 120 epochs and evaluates the complete held fold at epoch 0 and
every five epochs.  ``full`` starts a fresh identity head and retrains it on
all 800 cached train samples for the epoch selected by the completed OOF
contract.  It then emits one audited 595-key integrated candidate.

The module deliberately has no dataset-loader or official-test import.  Its
only sample authority is a committed frozen-context cache, and its only split
authority is a caller-supplied frozen fold manifest.  Full-image cached
Current tensors are used for validation.  Training extracts a zero-padded
272x272 context window and computes the objective only on its central
256x256 window; the formal repair head uses spatially local normalization, so
an eight-pixel halo is sufficient for exact central equivalence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import irstd_bgcr_run_contract as run_contract  # noqa: E402
from experiments import cache_irstd_frozen_context_v1 as cache_contract  # noqa: E402
from experiments.irstd_core_ring_loss import (  # noqa: E402
    compute_irstd_core_ring_loss,
    loss_manifest,
)
from experiments.irstd_logit_counterfactual import (  # noqa: E402
    corrupt_irstd_logits,
)
from experiments.pbdr_v4_metric_core import PBDRV4MetricAccumulator  # noqa: E402
from experiments.pbdr_v4_run_artifacts import (  # noqa: E402
    atomic_rolling_torch_save,
    exclusive_json,
    exclusive_torch_save,
    file_sha256,
    load_torch_artifact,
    optimizer_group_signature,
)
from experiments.pbdr_v4_state_contract import state_semantic_sha256  # noqa: E402
from experiments.pbdr_v4_training_core import (  # noqa: E402
    capture_rng_state,
    configure_determinism,
    restore_rng_state,
)
from model.irstd_core_ring_repair import (  # noqa: E402
    NEGATIVE_LOGIT_LIMIT,
    POSITIVE_LOGIT_LIMIT,
    PRODUCTION_PARAMETER_COUNT,
    PRODUCTION_PERSISTENT_BUFFER_COUNT,
    PRODUCTION_STATE_KEY_COUNT,
    validate_formal_irstd_core_ring_repair_head,
)
from model.tpd8_ner4_qfg2_irstd_crr import (  # noqa: E402
    CURRENT_INFERENCE_STATE_KEY_COUNT,
    CURRENT_TRAINING_STATE_KEY_COUNT,
    FORMAL_SEED,
    FrozenIRSTDContext,
    INTEGRATED_STATE_KEY_COUNT,
    IRSTD_CRR_STATE_PREFIX,
    TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    audit_frozen_current_base,
    build_formal_irstd_bgcr_model,
    strip_current_survival_state_strict,
    validate_formal_irstd_bgcr_model,
)


SCHEMA = "sctransnet_train_irstd_bgcr_v1/v1"
RUN_PROTOCOL_SCHEMA = f"{SCHEMA}/run_protocol"
ROLLING_SCHEMA = f"{SCHEMA}/rolling"
EVALUATION_CHECKPOINT_SCHEMA = f"{SCHEMA}/evaluation_checkpoint"
INTEGRATED_CANDIDATE_SCHEMA = f"{SCHEMA}/integrated_candidate"
FULL_SUMMARY_SCHEMA = "sctransnet_irstd_bgcr_training_v1/full_summary"
FOLD_SUMMARY_SCHEMA = "sctransnet_irstd_bgcr_training_v1/fold_summary"
OOF_SELECTOR_SCHEMA = "sctransnet_irstd_bgcr_oof_selector/v1"

DATASET = "IRSTD-1K"
ROLE = "best_miou"
MODES = ("fold", "full")
SEED = 42
TRAIN_EPOCHS = 120
EVALUATE_EVERY = 5
BATCH_SIZE = 16
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
COSINE_ETA_MIN = 1.0e-6
GRADIENT_CLIP_NORM = 1.0
PRECISION = "fp32"
CPU_INTRAOP_THREADS = 4
CPU_INTEROP_THREADS = 1

FULL_HEIGHT = 512
FULL_WIDTH = 512
OUTER_PATCH_SIZE = 272
CENTER_PATCH_SIZE = 256
CONTEXT_HALO = (OUTER_PATCH_SIZE - CENTER_PATCH_SIZE) // 2
PADDING_MODE = "constant_zero"
VALID_CENTER_MIN = CENTER_PATCH_SIZE // 2
VALID_CENTER_MAX = FULL_HEIGHT - CENTER_PATCH_SIZE // 2

CACHE_MANIFEST_NAME = "manifest.json"
CACHE_SAMPLE_DIRECTORY = "samples"
CACHE_SAMPLE_NAME_FORMAT = "{index:06d}.npz"
CACHE_SAMPLE_COUNT = 800
CACHE_FLOAT_FIELDS = (
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
CACHE_ID_FIELDS = (
    "target_component_ids",
    "rescue_component_ids",
)
CACHE_MASK_FIELDS = (
    "core_target",
    "attached_halo",
    "detached_false_positive",
    "outer_ring",
    "halo_target",
    "far_background",
    "baseline_rescue",
    "baseline_halo_advantage",
)
CACHE_FIELDS = (*CACHE_FLOAT_FIELDS, *CACHE_ID_FIELDS, *CACHE_MASK_FIELDS)

SOURCE_CLASSES = ("core_or_rescue", "attached_or_baseline_halo", "random")
COUNTERFACTUAL_MODES = (0, 1, 2)

RUN_PROTOCOL_NAME = "run_protocol.json"
ROLLING_NAME = "rolling_state.pth.tar"
SUMMARY_NAME = "summary.json"
METRIC_HISTORY_NAME = "metric_history.jsonl"
BASELINE_METRIC_NAME = "baseline1000_metric.json"
INTEGRATED_CANDIDATE_NAME = "integrated_candidate.pth.tar"
EVALUATION_CHECKPOINT_DIRECTORY = "evaluation_checkpoints"

CURRENT_CHECKPOINT_PATH = (
    REPO_ROOT
    / "results/three_dataset_tss_off_seed42_v1/runs/IRSTD-1K/"
    "final_tss_off/seed_42/checkpoints/best_miou.pth.tar"
)
CURRENT_CHECKPOINT_FILE_SHA256 = (
    "e8e9401500502dda0bbdc9640b830a7934fb2bc97bde706fde9adca216d965b4"
)
CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256 = (
    "d7600f61ee3d0967dae899de72a28f2e7e9e4c6381f2687189e45d84dcb3e298"
)
CURRENT_INFERENCE_STATE_SEMANTIC_SHA256 = (
    "f3745109e889cc6f25e42a43e698c5a43516ddc96a1364ffc78ab4b6b09d7f4f"
)
CURRENT_CHECKPOINT_EPOCH = 830
BASELINE_REFERENCE_NAME = "Baseline-epoch1000"

SOURCE_RELATIVE_PATHS = (
    "experiments/train_irstd_bgcr_v1.py",
    "experiments/cache_irstd_frozen_context_v1.py",
    "experiments/select_irstd_bgcr_oof_v1.py",
    "experiments/irstd_bgcr_run_contract.py",
    "experiments/irstd_core_ring_loss.py",
    "experiments/irstd_logit_counterfactual.py",
    "experiments/pbdr_v4_metric_core.py",
    "experiments/pbdr_v4_models_seed42_v1.py",
    "model/irstd_core_ring_repair.py",
    "model/tpd8_ner4_qfg2_irstd_crr.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IRSTDBGCRTrainingError(RuntimeError):
    """A cache, schedule, model, metric, or artifact violated the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IRSTDBGCRTrainingError(message)


def _sha256(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IRSTDBGCRTrainingError(
            f"value is not canonical-JSON serializable: {error}"
        ) from error
    return encoded + (b"\n" if newline else b"")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{name} is missing or unsafe: {candidate}",
    )
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IRSTDBGCRTrainingError(f"cannot parse {name}: {candidate}") from error
    _require(
        isinstance(value, Mapping)
        and all(isinstance(key, str) for key in value),
        f"{name} must be a string-key mapping",
    )
    return dict(value)


def _canonical_manifest_sha(
    value: Mapping[str, Any],
    *,
    field: str = "manifest_sha256",
) -> str:
    declared = _sha256(value.get(field), name=field)
    unsigned = dict(value)
    del unsigned[field]
    _require(
        canonical_json_sha256(unsigned) == declared,
        f"{field} does not replay",
    )
    return declared


def _ordered_identifiers(value: object, *, name: str) -> tuple[str, ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a sequence",
    )
    ready = tuple(value)
    _require(
        bool(ready)
        and all(
            isinstance(identifier, str)
            and identifier
            and "\x00" not in identifier
            for identifier in ready
        ),
        f"{name} contains an invalid sample ID",
    )
    _require(len(ready) == len(set(ready)), f"{name} contains duplicate IDs")
    return ready  # type: ignore[return-value]


def _source_bundle() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative_text in SOURCE_RELATIVE_PATHS:
        relative = Path(relative_text)
        _require(
            not relative.is_absolute()
            and all(part not in ("", ".", "..") for part in relative.parts),
            "source relative path is unsafe",
        )
        path = REPO_ROOT / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"source file is missing or unsafe: {relative_text}",
        )
        records.append(
            {
                "relative_path": relative_text,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    bundle: dict[str, Any] = {"files": records}
    bundle["bundle_sha256"] = canonical_json_sha256(bundle)
    return bundle


@dataclass(frozen=True, slots=True)
class CacheRecord:
    index: int
    sample_id: str
    relative_path: str
    bytes: int
    file_sha256: str
    record_relative_path: str
    record_bytes: int
    record_file_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenCacheSample:
    sample_id: str
    arrays: Mapping[str, np.ndarray]


def _cache_record(
    value: object,
    *,
    index: int,
) -> CacheRecord:
    _require(isinstance(value, Mapping), f"cache record {index} must be a mapping")
    sample_id = value.get("sample_id")
    _require(
        isinstance(sample_id, str) and sample_id and "\x00" not in sample_id,
        f"cache record {index} sample_id is invalid",
    )
    observed_index = value.get("index", index)
    _require(observed_index == index, f"cache record {index} index differs")
    expected_relative = f"{CACHE_SAMPLE_DIRECTORY}/{CACHE_SAMPLE_NAME_FORMAT.format(index=index)}"
    relative_path = value.get("cache_relative_path")
    _require(
        relative_path == expected_relative,
        f"cache record {index} path differs from six-digit sample contract",
    )
    byte_count = value.get("cache_bytes")
    _require(
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count > 0,
        f"cache record {index} byte count is invalid",
    )
    sample_sha = _sha256(
        value.get("cache_file_sha256"),
        name=f"cache record {index} file SHA",
    )
    expected_record_relative = f"records/{index:06d}.json"
    _require(
        value.get("record_relative_path") == expected_record_relative,
        f"cache record {index} sidecar path differs",
    )
    record_bytes = value.get("record_bytes")
    _require(
        isinstance(record_bytes, int)
        and not isinstance(record_bytes, bool)
        and record_bytes > 0,
        f"cache record {index} sidecar byte count is invalid",
    )
    record_file_sha = _sha256(
        value.get("record_file_sha256"),
        name=f"cache record {index} sidecar SHA",
    )
    return CacheRecord(
        index=index,
        sample_id=sample_id,
        relative_path=expected_relative,
        bytes=byte_count,
        file_sha256=sample_sha,
        record_relative_path=expected_record_relative,
        record_bytes=record_bytes,
        record_file_sha256=record_file_sha,
    )


def _validate_cache_array(name: str, value: np.ndarray) -> np.ndarray:
    _require(isinstance(value, np.ndarray), f"cache field {name} is not an array")
    if name == "u1":
        expected_shape = (32, FULL_HEIGHT, FULL_WIDTH)
        expected_dtype = np.dtype(np.float32)
    elif name in CACHE_FLOAT_FIELDS:
        expected_shape = (1, FULL_HEIGHT, FULL_WIDTH)
        expected_dtype = np.dtype(np.float32)
    else:
        expected_shape = (FULL_HEIGHT, FULL_WIDTH)
        expected_dtype = (
            np.dtype(np.int32) if name in CACHE_ID_FIELDS else np.dtype(np.bool_)
        )
    _require(value.shape == expected_shape, f"cache field {name} shape differs")
    _require(value.dtype == expected_dtype, f"cache field {name} dtype differs")
    ready = np.ascontiguousarray(value)
    _require(bool(ready.flags.c_contiguous), f"cache field {name} is not contiguous")
    if ready.dtype == np.dtype(np.float32):
        _require(bool(np.isfinite(ready).all()), f"cache field {name} is non-finite")
    if name == "target":
        _require(
            bool(((ready == 0.0) | (ready == 1.0)).all()),
            "cached target is not binary",
        )
    if name in CACHE_ID_FIELDS:
        _require(bool((ready >= 0).all()), f"cache field {name} contains negative IDs")
    ready.setflags(write=False)
    return ready


class FrozenContextCache:
    """Strict read-only, once-validated, host-RAM-resident context cache."""

    def __init__(self, root: Path | str) -> None:
        supplied = Path(root)
        _require(
            supplied.is_dir() and not supplied.is_symlink(),
            "cache root is missing or unsafe",
        )
        self.root = supplied.resolve(strict=True)
        manifest_path = self.root / CACHE_MANIFEST_NAME
        self.manifest = _load_json_mapping(manifest_path, name="cache manifest")
        _require(
            self.manifest.get("schema") == cache_contract.CACHE_SCHEMA
            and self.manifest.get("status") == "complete",
            "cache manifest schema/status differs",
        )
        _require(
            self.manifest.get("sample_count") == CACHE_SAMPLE_COUNT,
            "cache sample count must be 800",
        )
        for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
            _require(
                self.manifest.get(flag) is expected,
                f"cache official flag differs: {flag}",
            )
        _require(
            self.manifest.get("performance_acceptance_margin") is None
            and self.manifest.get("margin") is None,
            "cache performance margin must be null",
        )
        unsigned_manifest = dict(self.manifest)
        declared_manifest_sha = _sha256(
            unsigned_manifest.pop("manifest_semantic_sha256", None),
            name="cache manifest semantic SHA",
        )
        _require(
            cache_contract.canonical_sha256(unsigned_manifest)
            == declared_manifest_sha,
            "cache manifest semantic SHA does not replay",
        )
        commit_path = self.root / "COMMITTED.json"
        commit = _load_json_mapping(commit_path, name="cache commit")
        _require(
            commit.get("schema") == cache_contract.COMMIT_SCHEMA
            and commit.get("status") == "complete",
            "cache commit schema/status differs",
        )
        for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
            _require(commit.get(flag) is expected, f"cache commit flag differs: {flag}")
        _require(
            commit.get("manifest_file_sha256") == file_sha256(manifest_path)
            and commit.get("manifest_semantic_sha256") == declared_manifest_sha,
            "cache commit does not bind the manifest",
        )
        identity = self.manifest.get("identity")
        _require(isinstance(identity, Mapping), "cache identity is absent")
        _require(identity.get("dataset") == DATASET, "cache dataset differs")
        identity_unsigned = dict(identity)
        declared_identity_sha = _sha256(
            identity_unsigned.pop("identity_sha256", None),
            name="cache identity SHA",
        )
        _require(
            cache_contract.canonical_sha256(identity_unsigned)
            == declared_identity_sha
            == self.manifest.get("identity_sha256")
            == commit.get("identity_sha256"),
            "cache identity SHA does not replay",
        )
        model_bindings = identity.get("models")
        _require(isinstance(model_bindings, Mapping), "cache model bindings are absent")
        current_binding = model_bindings.get("current")
        baseline_binding = model_bindings.get("baseline1000")
        _require(
            isinstance(current_binding, Mapping)
            and current_binding.get("file_sha256") == CURRENT_CHECKPOINT_FILE_SHA256
            and current_binding.get("training_state_tensor_mapping_sha256")
            == CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256
            and current_binding.get("training_state_tensor_mapping_hash_algorithm")
            == "tensor_mapping_sha256"
            and current_binding.get("inference_state_semantic_sha256")
            == CURRENT_INFERENCE_STATE_SEMANTIC_SHA256
            and current_binding.get("inference_state_semantic_hash_algorithm")
            == "state_semantic_sha256",
            "cache Current binding differs",
        )
        _require(
            isinstance(baseline_binding, Mapping)
            and baseline_binding.get("file_sha256")
            == cache_contract.BASELINE_TEACHER_CHECKPOINT_FILE_SHA256
            and baseline_binding.get("raw_state_semantic_sha256")
            == cache_contract.BASELINE_TEACHER_RAW_STATE_SEMANTIC_SHA256
            and baseline_binding.get("normalized_state_semantic_sha256")
            == cache_contract.BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256
            and baseline_binding.get("state_semantic_hash_algorithm")
            == "state_semantic_sha256"
            and baseline_binding.get("epoch") == 1000,
            "cache Baseline-1000 binding differs",
        )
        raw_records = self.manifest.get("items")
        _require(isinstance(raw_records, list), "cache manifest lacks sample records")
        _require(len(raw_records) == CACHE_SAMPLE_COUNT, "cache record count differs")
        self.records = tuple(
            _cache_record(value, index=index)
            for index, value in enumerate(raw_records)
        )
        identifiers = tuple(record.sample_id for record in self.records)
        _require(len(set(identifiers)) == CACHE_SAMPLE_COUNT, "cache IDs are not unique")
        split_binding = identity.get("split")
        _require(isinstance(split_binding, Mapping), "cache split binding is absent")
        counts = split_binding.get("counts")
        _require(
            isinstance(counts, Mapping)
            and counts.get("official_train") == CACHE_SAMPLE_COUNT,
            "cache identity official-train count differs",
        )
        _require(
            split_binding.get("official_train_ids_sha256")
            == cache_contract.ordered_ids_sha256(identifiers),
            "cache item ordered-ID SHA differs from identity",
        )
        explicit_ids = split_binding.get("official_train_ids")
        if explicit_ids is not None:
            _require(
                _ordered_identifiers(
                    explicit_ids,
                    name="cache explicit official-train IDs",
                )
                == identifiers,
                "cache explicit ID order differs from items",
            )
        self.sample_ids = identifiers
        self._by_id = {record.sample_id: record for record in self.records}
        self.manifest_path = manifest_path.resolve(strict=True)
        self.manifest_file_sha256 = file_sha256(self.manifest_path)
        self.manifest_sha256 = declared_manifest_sha
        # Formal training is intentionally RAM resident.  Every NPZ/sidecar and
        # every per-array semantic hash is verified exactly once here; no epoch
        # performs disk decompression or file hashing again.
        self._samples: dict[str, FrozenCacheSample] = {}
        for item, record in zip(raw_records, self.records, strict=True):
            assert isinstance(item, Mapping)
            record_path = self.root / record.record_relative_path
            _require(
                record_path.is_file()
                and not record_path.is_symlink()
                and record_path.stat().st_size == record.record_bytes
                and file_sha256(record_path) == record.record_file_sha256,
                f"cache sidecar differs: {record.sample_id}",
            )
            sidecar = _load_json_mapping(record_path, name="cache sample sidecar")
            for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
                _require(
                    sidecar.get(flag) is expected,
                    f"cache sample official flag differs: {record.sample_id}/{flag}",
                )
            expected_item = {
                **sidecar,
                "record_relative_path": record.record_relative_path,
                "record_file_sha256": record.record_file_sha256,
                "record_bytes": record.record_bytes,
            }
            _require(dict(item) == expected_item, f"cache item differs: {record.sample_id}")
            unsigned_record = dict(sidecar)
            declared_record_sha = _sha256(
                unsigned_record.pop("record_semantic_sha256", None),
                name=f"cache record semantic SHA: {record.sample_id}",
            )
            _require(
                cache_contract.canonical_sha256(unsigned_record)
                == declared_record_sha,
                f"cache record semantic SHA differs: {record.sample_id}",
            )
            path = self._sample_path(record)
            try:
                with np.load(path, allow_pickle=False) as archive:
                    _require(set(archive.files) == set(CACHE_FIELDS), "cached NPZ fields differ")
                    arrays = {
                        name: np.ascontiguousarray(archive[name])
                        for name in CACHE_FIELDS
                    }
            except IRSTDBGCRTrainingError:
                raise
            except Exception as error:
                raise IRSTDBGCRTrainingError(
                    f"cannot preload cached sample {record.sample_id}: {error}"
                ) from error
            cache_contract.validate_cache_sample_arrays(arrays)
            array_records = sidecar.get("arrays")
            _require(isinstance(array_records, Mapping), "cache array records are absent")
            for name, value in arrays.items():
                value.setflags(write=False)
                metadata = array_records.get(name)
                _require(
                    isinstance(metadata, Mapping)
                    and metadata.get("semantic_sha256")
                    == cache_contract.array_semantic_sha256(value),
                    f"cache array semantic SHA differs: {record.sample_id}/{name}",
                )
            self._samples[record.sample_id] = FrozenCacheSample(
                sample_id=record.sample_id,
                arrays=arrays,
            )

    def _sample_path(self, record: CacheRecord) -> Path:
        candidate = self.root / record.relative_path
        _require(
            candidate.is_file() and not candidate.is_symlink(),
            f"cached sample is missing or unsafe: {record.relative_path}",
        )
        resolved = candidate.resolve(strict=True)
        _require(
            resolved.is_relative_to(self.root),
            f"cached sample escapes cache root: {record.relative_path}",
        )
        _require(resolved.stat().st_size == record.bytes, "cached sample bytes differ")
        _require(
            file_sha256(resolved) == record.file_sha256,
            "cached sample file SHA differs",
        )
        return resolved

    def load(self, sample_id: str) -> FrozenCacheSample:
        _require(sample_id in self._samples, f"sample is absent from cache: {sample_id}")
        return self._samples[sample_id]

    def support(self, sample_ids: Sequence[str]) -> dict[str, tuple[bool, bool]]:
        result: dict[str, tuple[bool, bool]] = {}
        for sample_id in _ordered_identifiers(sample_ids, name="support sample IDs"):
            sample = self.load(sample_id)
            arrays = sample.arrays
            core = (
                arrays["core_target"]
                | (arrays["rescue_component_ids"] > 0)
                | arrays["baseline_rescue"]
            )
            halo = arrays["attached_halo"] | arrays["baseline_halo_advantage"]
            result[sample_id] = (bool(core.any()), bool(halo.any()))
        return result


def load_frozen_fold_manifest(path: Path | str) -> dict[str, Any]:
    """Read a prebuilt fold manifest without rebuilding or opening an index."""

    manifest = _load_json_mapping(Path(path), name="frozen fold manifest")
    _require(
        manifest.get("schema") == run_contract.FOLD_MANIFEST_SCHEMA,
        "fold manifest schema differs",
    )
    _require(manifest.get("dataset") == DATASET, "fold manifest dataset differs")
    _require(
        manifest.get("sample_count") == CACHE_SAMPLE_COUNT,
        "fold manifest sample count differs",
    )
    _require(
        tuple(manifest.get("fold_sizes", ())) == run_contract.FOLD_SIZES,
        "fold sizes differ",
    )
    _require(
        manifest.get("assignment_sha256") == run_contract.FOLD_ASSIGNMENT_SHA256,
        "fold assignment SHA differs",
    )
    _canonical_manifest_sha(manifest)
    for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
        _require(manifest.get(flag) is expected, f"fold official flag differs: {flag}")
    _require(
        manifest.get("performance_acceptance_margin") is None,
        "fold performance margin must be null",
    )
    folds = manifest.get("folds")
    _require(isinstance(folds, list) and len(folds) == 3, "fold records differ")
    all_validation: list[str] = []
    for fold_index, raw in enumerate(folds):
        _require(isinstance(raw, Mapping), f"fold {fold_index} is invalid")
        _require(raw.get("fold_index") == fold_index, f"fold {fold_index} index differs")
        validation = _ordered_identifiers(
            raw.get("validation_ids"), name=f"fold {fold_index} validation IDs"
        )
        training = _ordered_identifiers(
            raw.get("training_ids"), name=f"fold {fold_index} training IDs"
        )
        _require(
            len(validation) == run_contract.FOLD_SIZES[fold_index]
            and len(training) == CACHE_SAMPLE_COUNT - len(validation),
            f"fold {fold_index} counts differ",
        )
        _require(
            not (set(validation) & set(training))
            and len(set(validation) | set(training)) == CACHE_SAMPLE_COUNT,
            f"fold {fold_index} is not a partition",
        )
        _require(
            raw.get("validation_ids_sha256")
            == canonical_json_sha256(list(validation)),
            f"fold {fold_index} validation-ID SHA differs",
        )
        _require(
            raw.get("training_ids_sha256") == canonical_json_sha256(list(training)),
            f"fold {fold_index} training-ID SHA differs",
        )
        all_validation.extend(validation)
    _require(
        len(all_validation) == CACHE_SAMPLE_COUNT
        and len(set(all_validation)) == CACHE_SAMPLE_COUNT,
        "held folds are not a disjoint cover of 800 IDs",
    )
    return manifest


def validate_cache_fold_alignment(
    cache: FrozenContextCache,
    fold_manifest: Mapping[str, Any],
) -> None:
    folds = fold_manifest["folds"]
    validation_union = {
        identifier
        for fold in folds
        for identifier in fold["validation_ids"]
    }
    _require(
        validation_union == set(cache.sample_ids),
        "cache IDs and frozen fold IDs differ",
    )


def _extract_zero_padded(
    value: np.ndarray,
    *,
    top: int,
    left: int,
    size: int = OUTER_PATCH_SIZE,
) -> np.ndarray:
    """Extract CHW/HW using exact constant-zero padding outside the image."""

    _require(isinstance(value, np.ndarray), "crop source must be an array")
    _require(value.ndim in (2, 3), "crop source must be HW or CHW")
    _require(type(top) is int and type(left) is int, "crop origin must be integral")
    _require(type(size) is int and size > 0, "crop size must be positive")
    height, width = value.shape[-2:]
    bottom = top + size
    right = left + size
    source_top = max(0, top)
    source_left = max(0, left)
    source_bottom = min(height, bottom)
    source_right = min(width, right)
    result_shape = (*value.shape[:-2], size, size)
    result = np.zeros(result_shape, dtype=value.dtype)
    if source_top < source_bottom and source_left < source_right:
        destination_top = source_top - top
        destination_left = source_left - left
        result[
            ...,
            destination_top : destination_top + source_bottom - source_top,
            destination_left : destination_left + source_right - source_left,
        ] = value[..., source_top:source_bottom, source_left:source_right]
    return np.ascontiguousarray(result)


def extract_outer_context_patch(
    value: np.ndarray,
    *,
    center_y: int,
    center_x: int,
) -> np.ndarray:
    _require(
        0 <= center_y < value.shape[-2] and 0 <= center_x < value.shape[-1],
        "crop center is outside the cached image",
    )
    central_top = center_y - CENTER_PATCH_SIZE // 2
    central_left = center_x - CENTER_PATCH_SIZE // 2
    return _extract_zero_padded(
        value,
        top=central_top - CONTEXT_HALO,
        left=central_left - CONTEXT_HALO,
    )


def center_crop_tensor(value: torch.Tensor) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), "center crop requires a tensor")
    _require(
        value.ndim >= 2
        and value.shape[-2:] == (OUTER_PATCH_SIZE, OUTER_PATCH_SIZE),
        "center crop source must be 272x272",
    )
    return value[
        ...,
        CONTEXT_HALO : CONTEXT_HALO + CENTER_PATCH_SIZE,
        CONTEXT_HALO : CONTEXT_HALO + CENTER_PATCH_SIZE,
    ]


def _stable_digest(seed: int, epoch: int, tag: str, sample_id: str = "") -> bytes:
    _require(type(seed) is int and seed == SEED, "formal stable-hash seed must be 42")
    _require(type(epoch) is int and epoch >= 0, "stable-hash epoch is invalid")
    payload = f"{seed}\0{epoch}\0{tag}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _stable_order(
    identifiers: Iterable[str],
    *,
    epoch: int,
    tag: str,
) -> list[str]:
    return sorted(
        identifiers,
        key=lambda identifier: (
            _stable_digest(SEED, epoch, tag, identifier),
            identifier.encode("utf-8"),
        ),
    )


def _balanced_cycle(length: int, *, phase: int) -> tuple[int, ...]:
    _require(type(length) is int and length > 0, "balanced cycle length is invalid")
    _require(phase in (0, 1, 2), "balanced cycle phase is invalid")
    return tuple((phase + index) % 3 for index in range(length))


@dataclass(frozen=True, slots=True)
class EpochPlanItem:
    sample_id: str
    source_class: str
    counterfactual_mode: int


def build_epoch_plan(
    sample_ids: Sequence[str],
    support: Mapping[str, tuple[bool, bool]],
    *,
    epoch: int,
    batch_size: int = BATCH_SIZE,
) -> tuple[EpochPlanItem, ...]:
    """Assign every sample once with exact deterministic 3-way schedules."""

    identifiers = _ordered_identifiers(sample_ids, name="epoch-plan sample IDs")
    _require(type(epoch) is int and 1 <= epoch <= TRAIN_EPOCHS, "epoch is invalid")
    _require(batch_size == BATCH_SIZE, "formal batch size must be 16")
    _require(set(support) == set(identifiers), "support map IDs differ")
    _require(
        all(
            isinstance(value, tuple)
            and len(value) == 2
            and all(type(flag) is bool for flag in value)
            for value in support.values()
        ),
        "support map values must be (core, halo) bool pairs",
    )

    category_phase = int.from_bytes(
        _stable_digest(SEED, epoch, "source-class-phase")[:2], "little"
    ) % 3
    mode_phase = int.from_bytes(
        _stable_digest(SEED, epoch, "counterfactual-mode-phase")[:2], "little"
    ) % 3
    category_slots = _balanced_cycle(len(identifiers), phase=category_phase)
    mode_slots = _balanced_cycle(len(identifiers), phase=mode_phase)
    category_counts = tuple(category_slots.count(index) for index in range(3))
    _require(
        max(category_counts) - min(category_counts) <= 1,
        "source-class totals are not balanced",
    )

    core_quota, halo_quota = category_counts[0], category_counts[1]
    core_only = [item for item in identifiers if support[item] == (True, False)]
    halo_only = [item for item in identifiers if support[item] == (False, True)]
    both = [item for item in identifiers if support[item] == (True, True)]
    core_only = _stable_order(core_only, epoch=epoch, tag="core-only")
    halo_only = _stable_order(halo_only, epoch=epoch, tag="halo-only")
    both = _stable_order(both, epoch=epoch, tag="core-halo-both")
    _require(
        core_quota <= len(core_only) + len(both),
        "insufficient core/rescue-support samples for strict balance",
    )
    _require(
        halo_quota <= len(halo_only) + len(both),
        "insufficient halo-support samples for strict balance",
    )
    _require(
        core_quota + halo_quota <= len(core_only) + len(halo_only) + len(both),
        "core and halo quotas cannot be assigned without sample reuse",
    )

    assigned_core = core_only[:core_quota]
    assigned_halo = halo_only[:halo_quota]
    used = set(assigned_core) | set(assigned_halo)
    both_remaining = [item for item in both if item not in used]
    core_deficit = core_quota - len(assigned_core)
    assigned_core.extend(both_remaining[:core_deficit])
    used.update(assigned_core)
    both_remaining = [item for item in both_remaining if item not in used]
    halo_deficit = halo_quota - len(assigned_halo)
    assigned_halo.extend(both_remaining[:halo_deficit])
    used.update(assigned_halo)
    _require(
        len(assigned_core) == core_quota and len(assigned_halo) == halo_quota,
        "strict source-class assignment failed",
    )
    assigned_random = [item for item in identifiers if item not in used]
    _require(
        len(assigned_random) == category_counts[2],
        "random source-class quota differs",
    )
    queues = {
        0: _stable_order(assigned_core, epoch=epoch, tag="core-queue"),
        1: _stable_order(assigned_halo, epoch=epoch, tag="halo-queue"),
        2: _stable_order(assigned_random, epoch=epoch, tag="random-queue"),
    }
    offsets = [0, 0, 0]
    plan: list[EpochPlanItem] = []
    for position, (category, mode) in enumerate(zip(category_slots, mode_slots, strict=True)):
        sample_id = queues[category][offsets[category]]
        offsets[category] += 1
        plan.append(
            EpochPlanItem(
                sample_id=sample_id,
                source_class=SOURCE_CLASSES[category],
                counterfactual_mode=mode,
            )
        )
        if (position + 1) % batch_size == 0 or position + 1 == len(identifiers):
            start = position + 1 - min(batch_size, position + 1)
            if position + 1 == len(identifiers) and len(identifiers) % batch_size:
                start = len(identifiers) - len(identifiers) % batch_size
            batch = plan[start : position + 1]
            source_batch_counts = [
                sum(item.source_class == source for item in batch)
                for source in SOURCE_CLASSES
            ]
            mode_batch_counts = [
                sum(item.counterfactual_mode == mode_value for item in batch)
                for mode_value in COUNTERFACTUAL_MODES
            ]
            _require(
                max(source_batch_counts) - min(source_batch_counts) <= 1,
                "one batch is not source-class balanced",
            )
            _require(
                max(mode_batch_counts) - min(mode_batch_counts) <= 1,
                "one batch is not counterfactual-mode balanced",
            )
    _require(
        len(plan) == len(identifiers)
        and {item.sample_id for item in plan} == set(identifiers),
        "epoch plan does not use every sample exactly once",
    )
    return tuple(plan)


def epoch_plan_sha256(plan: Sequence[EpochPlanItem]) -> str:
    return canonical_json_sha256([asdict(item) for item in plan])


def _coordinate_from_mask(
    mask: np.ndarray,
    *,
    epoch: int,
    tag: str,
    sample_id: str,
) -> tuple[int, int]:
    coordinates = np.argwhere(mask)
    _require(len(coordinates) > 0, f"source mask is empty: {sample_id}/{tag}")
    digest = _stable_digest(SEED, epoch, f"coordinate:{tag}", sample_id)
    selected = int.from_bytes(digest[:8], "little") % len(coordinates)
    y, x = coordinates[selected]
    return (
        min(VALID_CENTER_MAX, max(VALID_CENTER_MIN, int(y))),
        min(VALID_CENTER_MAX, max(VALID_CENTER_MIN, int(x))),
    )


def _random_coordinate(*, epoch: int, sample_id: str) -> tuple[int, int]:
    digest = _stable_digest(SEED, epoch, "coordinate:random", sample_id)
    valid_count = VALID_CENTER_MAX - VALID_CENTER_MIN + 1
    y = VALID_CENTER_MIN + int.from_bytes(digest[:8], "little") % valid_count
    x = VALID_CENTER_MIN + int.from_bytes(digest[8:16], "little") % valid_count
    return y, x


class CachedPatchDataset(Dataset[dict[str, object]]):
    """One deterministic, one-use-per-sample epoch over cached context."""

    def __init__(
        self,
        cache: FrozenContextCache,
        plan: Sequence[EpochPlanItem],
        *,
        epoch: int,
    ) -> None:
        self.cache = cache
        self.plan = tuple(plan)
        self.epoch = epoch
        _require(bool(self.plan), "cached patch epoch plan is empty")

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.plan[index]
        sample = self.cache.load(item.sample_id)
        arrays = sample.arrays
        if item.source_class == SOURCE_CLASSES[0]:
            mask = (
                arrays["core_target"]
                | (arrays["rescue_component_ids"] > 0)
                | arrays["baseline_rescue"]
            )
            center_y, center_x = _coordinate_from_mask(
                mask,
                epoch=self.epoch,
                tag=item.source_class,
                sample_id=item.sample_id,
            )
        elif item.source_class == SOURCE_CLASSES[1]:
            mask = arrays["attached_halo"] | arrays["baseline_halo_advantage"]
            center_y, center_x = _coordinate_from_mask(
                mask,
                epoch=self.epoch,
                tag=item.source_class,
                sample_id=item.sample_id,
            )
        else:
            _require(item.source_class == SOURCE_CLASSES[2], "source class differs")
            center_y, center_x = _random_coordinate(
                epoch=self.epoch,
                sample_id=item.sample_id,
            )
        result: dict[str, object] = {
            name: torch.from_numpy(
                extract_outer_context_patch(
                    arrays[name],
                    center_y=center_y,
                    center_x=center_x,
                )
            )
            for name in CACHE_FIELDS
        }
        result.update(
            {
                "sample_id": item.sample_id,
                "source_class": item.source_class,
                "counterfactual_mode": item.counterfactual_mode,
                "center_y": center_y,
                "center_x": center_x,
            }
        )
        return result


def _head_state(model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.irstd_repair.state_dict().items()
    }
    _require(len(state) == PRODUCTION_STATE_KEY_COUNT, "repair state count differs")
    return state


def _integrated_cpu_state(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    _require(len(state) == INTEGRATED_STATE_KEY_COUNT, "integrated state count differs")
    return state


def _audit_limit_buffers(state: Mapping[str, torch.Tensor], *, prefixed: bool) -> None:
    prefix = IRSTD_CRR_STATE_PREFIX if prefixed else ""
    positive_key = f"{prefix}positive_limit"
    negative_key = f"{prefix}negative_limit"
    _require(set((positive_key, negative_key)) <= set(state), "repair limit buffers are absent")
    expected_positive = torch.tensor(POSITIVE_LOGIT_LIMIT, dtype=torch.float32)
    expected_negative = torch.tensor(NEGATIVE_LOGIT_LIMIT, dtype=torch.float32)
    _require(
        torch.equal(state[positive_key].detach().cpu(), expected_positive),
        "repair positive limit differs",
    )
    _require(
        torch.equal(state[negative_key].detach().cpu(), expected_negative),
        "repair negative limit differs",
    )


def _load_immutable_current() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Load only the code-pinned Current checkpoint registry, never a dataset."""

    from experiments.pbdr_v4_models_seed42_v1 import load_current_checkpoint

    _, training_state, record = load_current_checkpoint(DATASET, ROLE)
    _require(len(training_state) == CURRENT_TRAINING_STATE_KEY_COUNT, "Current key count differs")
    _require(
        Path(str(record["path"])).resolve(strict=True)
        == CURRENT_CHECKPOINT_PATH.resolve(strict=True),
        "Current checkpoint path differs",
    )
    _require(record.get("sha256") == CURRENT_CHECKPOINT_FILE_SHA256, "Current file SHA differs")
    _require(
        record.get("state_sha256")
        == CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256,
        "Current tensor-mapping state SHA differs",
    )
    _require(record.get("epoch") == CURRENT_CHECKPOINT_EPOCH, "Current epoch differs")
    _require(
        cache_contract.tensor_mapping_sha256(training_state)
        == CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256,
        "live Current 568-key tensor-mapping SHA differs",
    )
    inference_state = strip_current_survival_state_strict(training_state)
    _require(len(inference_state) == CURRENT_INFERENCE_STATE_KEY_COUNT, "Current projection differs")
    _require(
        state_semantic_sha256(inference_state)
        == CURRENT_INFERENCE_STATE_SEMANTIC_SHA256,
        "Current inference state SHA differs",
    )
    return training_state, inference_state, dict(record)


def _device_fingerprint(device: torch.device) -> dict[str, Any]:
    if device.type == "cpu":
        return {"device_type": "cpu", "torch_version": torch.__version__}
    _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA is unavailable")
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    return {
        "device_type": "cuda",
        "visible_index": index,
        "uuid": None if uuid is None else str(uuid),
        "name": properties.name,
        "total_memory": int(properties.total_memory),
        "compute_capability": [properties.major, properties.minor],
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
    }


def _training_recipe() -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "role": ROLE,
        "trainable": f"{IRSTD_CRR_STATE_PREFIX}* only",
        "seed": SEED,
        "precision": PRECISION,
        "tf32": False,
        "cpu_intraop_threads": CPU_INTRAOP_THREADS,
        "cpu_interop_threads": CPU_INTEROP_THREADS,
        "epochs": TRAIN_EPOCHS,
        "evaluate_every": EVALUATE_EVERY,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": TRAIN_EPOCHS,
        "scheduler_eta_min": COSINE_ETA_MIN,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "outer_patch_size": OUTER_PATCH_SIZE,
        "center_loss_size": CENTER_PATCH_SIZE,
        "context_halo": CONTEXT_HALO,
        "padding": PADDING_MODE,
        "loss_center_range_in_full_image": [VALID_CENTER_MIN, VALID_CENTER_MAX],
        "cache_access": "validate_once_then_host_RAM_resident",
        "host_resident_cache": True,
        "source_classes": list(SOURCE_CLASSES),
        "source_class_schedule": "stable-hash explicit balanced; each sample once",
        "counterfactual_modes": list(COUNTERFACTUAL_MODES),
        "counterfactual_schedule": "explicit balanced cyclic",
        "probability_threshold": 0.5,
        "probability_comparison": "strict_greater_than",
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }


def build_optimizer(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
) -> torch.optim.AdamW:
    parameters = model.trainable_parameters()
    _require(
        sum(parameter.numel() for parameter in parameters) == PRODUCTION_PARAMETER_COUNT,
        "trainable parameter count differs from repair head",
    )
    return torch.optim.AdamW(
        [{"name": "irstd_repair", "params": list(parameters)}],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_EPOCHS,
        eta_min=COSINE_ETA_MIN,
    )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[key] = value.to(device=device)


def _counterfactual_generator(epoch: int, batch_index: int) -> torch.Generator:
    digest = _stable_digest(SEED, epoch, f"counterfactual-batch:{batch_index}")
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(seed)


def _restore_rng_after_hardware_check(
    rng_state: Mapping[str, Any],
    *,
    migrated: bool,
) -> None:
    if not migrated:
        restore_rng_state(rng_state)
        return
    _require(
        set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "migrated RNG state fields differ",
    )
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    # The repair graph has no stochastic layer, epoch plans are SHA-derived,
    # and counterfactual amplitudes use an explicit per-batch CPU generator.
    # CUDA global RNG therefore has no semantic consumer.  A migrated process
    # receives the formal seed instead of attempting to install a state vector
    # whose visible-device cardinality belongs to different hardware.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _to_device(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), "batch field is not a tensor")
    result = value.to(device=device, non_blocking=False)
    return result if dtype is None else result.to(dtype=dtype)


def train_one_epoch(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    loader: DataLoader[dict[str, object]],
    optimizer: torch.optim.AdamW,
    *,
    epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    model.train(True)
    _audit_limit_buffers(model.irstd_repair.state_dict(), prefixed=False)
    component_sums: dict[str, float] = {}
    batch_count = 0
    sample_count = 0
    max_gradient_norm = 0.0
    observed_ids: list[str] = []
    source_counts = {name: 0 for name in SOURCE_CLASSES}
    mode_counts = {mode: 0 for mode in COUNTERFACTUAL_MODES}
    for batch_index, batch in enumerate(loader):
        image = _to_device(batch["image"], device=device, dtype=torch.float32)
        target = _to_device(batch["target"], device=device, dtype=torch.float32)
        context = FrozenIRSTDContext(
            local_feature=_to_device(batch["u1"], device=device, dtype=torch.float32),
            out_logits=_to_device(batch["z_out"], device=device, dtype=torch.float32),
            d0_logits=_to_device(batch["z_d0"], device=device, dtype=torch.float32),
            gt2_logits=_to_device(batch["z_gt2"], device=device, dtype=torch.float32),
            gt3_logits=_to_device(batch["z_gt3"], device=device, dtype=torch.float32),
            gt4_logits=_to_device(batch["z_gt4"], device=device, dtype=torch.float32),
            gt5_logits=_to_device(batch["z_gt5"], device=device, dtype=torch.float32),
        )
        core_target = _to_device(batch["core_target"], device=device).unsqueeze(1).bool()
        baseline_rescue = _to_device(batch["baseline_rescue"], device=device).unsqueeze(1).bool()
        supervised_core = core_target | baseline_rescue
        outer_ring = _to_device(batch["outer_ring"], device=device).unsqueeze(1).bool()
        observed_halo = _to_device(batch["halo_target"], device=device).unsqueeze(1).bool()
        modes = _to_device(batch["counterfactual_mode"], device=device).long()
        corrupted = corrupt_irstd_logits(
            current_logits=context.out_logits,
            core_target=supervised_core,
            outer_ring=outer_ring,
            observed_halo_target=observed_halo,
            generator=_counterfactual_generator(epoch, batch_index),
            modes=modes,
        )
        routing = model.forward_repair_from_context(
            image,
            context,
            base_logits_override=corrupted.logits,
        )
        loss = compute_irstd_core_ring_loss(
            routed_logits=center_crop_tensor(routing.routed_logits),
            current_logits=center_crop_tensor(context.out_logits),
            target=center_crop_tensor(target),
            target_component_ids=center_crop_tensor(
                _to_device(batch["target_component_ids"], device=device).unsqueeze(1).int()
            ),
            core_target=center_crop_tensor(supervised_core),
            halo_target=center_crop_tensor(corrupted.halo_target),
            attached_halo=center_crop_tensor(
                _to_device(batch["attached_halo"], device=device).unsqueeze(1).bool()
            ),
            far_background=center_crop_tensor(
                _to_device(batch["far_background"], device=device).unsqueeze(1).bool()
            ),
            core_gate_logits=center_crop_tensor(routing.core_gate_logits),
            halo_gate_logits=center_crop_tensor(routing.halo_gate_logits),
            positive_delta=center_crop_tensor(routing.positive_delta),
            negative_delta=center_crop_tensor(routing.negative_delta),
            delta_logits=center_crop_tensor(routing.delta_logits),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        repair_gradients = {
            name: parameter.grad
            for name, parameter in model.irstd_repair.named_parameters()
        }
        _require(
            len(repair_gradients) == 29
            and all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for gradient in repair_gradients.values()
            ),
            "one of the 29 repair parameter tensors has a missing/non-finite gradient",
        )
        _require(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if not name.startswith(IRSTD_CRR_STATE_PREFIX)
            ),
            "a frozen Current parameter received a gradient",
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters(),
            GRADIENT_CLIP_NORM,
        )
        _require(bool(torch.isfinite(gradient_norm)), "gradient norm is non-finite")
        optimizer.step()
        _audit_limit_buffers(model.irstd_repair.state_dict(), prefixed=False)
        _require(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if not name.startswith(IRSTD_CRR_STATE_PREFIX)
            ),
            "a frozen Current parameter received a gradient",
        )
        scalars = loss.detached_scalars()
        for name, value in scalars.items():
            component_sums[name] = component_sums.get(name, 0.0) + value
        batch_count += 1
        batch_samples = int(image.shape[0])
        sample_count += batch_samples
        max_gradient_norm = max(max_gradient_norm, float(gradient_norm.detach().cpu()))
        batch_ids = batch["sample_id"]
        _require(isinstance(batch_ids, list), "collated sample IDs differ")
        observed_ids.extend(batch_ids)
        batch_sources = batch["source_class"]
        _require(isinstance(batch_sources, list), "collated source classes differ")
        for source in batch_sources:
            _require(source in source_counts, "collated source class differs")
            source_counts[source] += 1
        for mode in modes.detach().cpu().tolist():
            _require(mode in mode_counts, "collated counterfactual mode differs")
            mode_counts[mode] += 1
    _require(batch_count > 0 and sample_count == len(loader.dataset), "epoch coverage differs")
    _require(
        len(observed_ids) == len(set(observed_ids)) == sample_count,
        "one sample was repeated or omitted in the epoch",
    )
    _require(
        max(source_counts.values()) - min(source_counts.values()) <= 1,
        "epoch source classes are not balanced",
    )
    _require(
        max(mode_counts.values()) - min(mode_counts.values()) <= 1,
        "epoch counterfactual modes are not balanced",
    )
    return {
        "epoch": epoch,
        "batch_count": batch_count,
        "sample_count": sample_count,
        "sample_ids_sha256": canonical_json_sha256(observed_ids),
        "source_class_counts": source_counts,
        "counterfactual_mode_counts": {str(key): value for key, value in mode_counts.items()},
        "mean_loss_components": {
            name: value / batch_count for name, value in component_sums.items()
        },
        "max_preclip_gradient_norm": max_gradient_norm,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }


def _full_context(sample: FrozenCacheSample, device: torch.device) -> FrozenIRSTDContext:
    arrays = sample.arrays
    return FrozenIRSTDContext(
        local_feature=torch.from_numpy(np.array(arrays["u1"], copy=True)).unsqueeze(0).to(device),
        out_logits=torch.from_numpy(np.array(arrays["z_out"], copy=True)).unsqueeze(0).to(device),
        d0_logits=torch.from_numpy(np.array(arrays["z_d0"], copy=True)).unsqueeze(0).to(device),
        gt2_logits=torch.from_numpy(np.array(arrays["z_gt2"], copy=True)).unsqueeze(0).to(device),
        gt3_logits=torch.from_numpy(np.array(arrays["z_gt3"], copy=True)).unsqueeze(0).to(device),
        gt4_logits=torch.from_numpy(np.array(arrays["z_gt4"], copy=True)).unsqueeze(0).to(device),
        gt5_logits=torch.from_numpy(np.array(arrays["z_gt5"], copy=True)).unsqueeze(0).to(device),
    )


def _fraction_fields(prefix: str, value: Fraction) -> dict[str, int]:
    return {
        f"{prefix}_numerator": value.numerator,
        f"{prefix}_denominator": value.denominator,
    }


@torch.no_grad()
def evaluate_cached_fold(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    cache: FrozenContextCache,
    identifiers: Sequence[str],
    *,
    fold_index: int,
    epoch: int,
    fold_manifest: Mapping[str, Any],
    device: torch.device,
    candidate_kind: str = run_contract.BGCR_CANDIDATE_KIND,
    candidate_name: str = run_contract.BGCR_CANDIDATE_NAME,
) -> dict[str, Any]:
    """Evaluate a BGCR/Current/Baseline logit source on a complete held fold."""

    ids = _ordered_identifiers(identifiers, name="evaluation IDs")
    binding = run_contract.fold_metric_binding(
        fold_index,
        epoch,
        fold_manifest=fold_manifest,
    )
    _require(len(ids) == binding["sample_count"], "held-fold evaluation is incomplete")
    if candidate_kind == "bgcr":
        model.eval()
    elif candidate_kind == "frozen_reference":
        _require(
            candidate_name in (BASELINE_REFERENCE_NAME, "Current"),
            "unsupported frozen reference",
        )
    else:
        raise IRSTDBGCRTrainingError("unsupported evaluation candidate kind")
    accumulator = PBDRV4MetricAccumulator()
    niou_sum = Fraction(0, 1)
    loss_sum = Fraction(0, 1)
    for sample_id in ids:
        sample = cache.load(sample_id)
        target = torch.from_numpy(np.array(sample.arrays["target"], copy=True)).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        )
        if candidate_kind == "bgcr":
            image = torch.from_numpy(np.array(sample.arrays["image"], copy=True)).unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            )
            context = _full_context(sample, device)
            logits = model.forward_repair_from_context(image, context).routed_logits
        elif candidate_name == BASELINE_REFERENCE_NAME:
            logits = torch.from_numpy(
                np.array(sample.arrays["baseline1000_logits"], copy=True)
            ).unsqueeze(0).to(device=device, dtype=torch.float32)
        else:
            logits = torch.from_numpy(np.array(sample.arrays["z_out"], copy=True)).unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            )
        _require(logits.shape == target.shape == (1, 1, FULL_HEIGHT, FULL_WIDTH), "full-image shape differs")
        sample_loss_tensor = F.binary_cross_entropy_with_logits(logits.float(), target.float())
        sample_loss = float(sample_loss_tensor.detach().cpu())
        probability = torch.sigmoid(logits.float()).detach().cpu().numpy()[0, 0]
        target_numpy = target.detach().cpu().numpy()[0, 0]
        accumulator.update(
            probability=np.ascontiguousarray(probability, dtype=np.float32),
            target=np.ascontiguousarray(target_numpy, dtype=np.float32),
            loss=sample_loss,
            identifier=sample_id,
        )
        prediction_binary = probability > 0.5
        target_binary = target_numpy > 0.5
        intersection = int(np.logical_and(prediction_binary, target_binary).sum())
        union = int(np.logical_or(prediction_binary, target_binary).sum())
        niou_sum += Fraction(1, 1) if union == 0 else Fraction(intersection, union)
        loss_sum += Fraction.from_float(sample_loss)
    metrics = accumulator.compute()
    _require(metrics["sample_count"] == len(ids), "metric accumulator coverage differs")
    row: dict[str, Any] = dict(binding)
    for name in run_contract.ADDITIVE_COUNT_FIELDS:
        _require(name in metrics, f"metric accumulator lacks {name}")
        row[name] = int(metrics[name])
    row.update(
        {
            "candidate_kind": candidate_kind,
            "candidate_name": candidate_name,
            "miou": float(metrics["miou"]),
            "niou": float(metrics["niou"]),
            "test_loss": float(metrics["test_loss"]),
            "pd": float(metrics["pd"]),
            "fa": float(metrics["fa"]),
            "tiny_pd": metrics["tiny_pd"],
            "pixel_precision": float(metrics["pixel_precision"]),
            "pixel_recall": float(metrics["pixel_recall"]),
            "pixel_f1": float(metrics["pixel_f1"]),
            "sample_id_order_sha256": metrics["sample_id_order_sha256"],
            "target_sha256": metrics["target_sha256"],
            **_fraction_fields("niou_sum", niou_sum),
            **_fraction_fields("loss_sum", loss_sum),
        }
    )
    # Reuse the public pool validator's single-row internals indirectly at OOF
    # collection time; here the exact identities and arithmetic are explicit.
    _require(
        row["true_positive_pixels"] == row["intersection_pixels"]
        and row["union_pixels"]
        == row["true_positive_pixels"]
        + row["false_positive_pixels"]
        + row["false_negative_pixels"],
        "evaluation pixel confusion statistics differ",
    )
    return row


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(not destination.parent.is_symlink(), "JSONL parent is a symlink")
    encoded = canonical_json_bytes(dict(value), newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _require(path.is_file() and not path.is_symlink(), "JSONL history is unsafe")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            _require(isinstance(value, Mapping), "JSONL row is not a mapping")
            rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IRSTDBGCRTrainingError("cannot parse JSONL history") from error
    return rows


def _commit_or_validate_json(path: Path, payload: Mapping[str, Any]) -> Path:
    if not path.exists() and not path.is_symlink():
        return exclusive_json(path, payload)
    observed = _load_json_mapping(path, name=path.name)
    _require(observed == dict(payload), f"existing {path.name} differs")
    return path


def _repair_state_sha(state: Mapping[str, torch.Tensor]) -> str:
    _require(len(state) == PRODUCTION_STATE_KEY_COUNT, "repair state key count differs")
    _audit_limit_buffers(state, prefixed=False)
    _require(
        all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in state.values()),
        "repair state contains a non-finite tensor",
    )
    return state_semantic_sha256(state)


def _rolling_payload(
    *,
    identity: Mapping[str, Any],
    epoch: int,
    budget: int,
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    metric_history: Sequence[Mapping[str, Any]],
    baseline_metric: Mapping[str, Any] | None,
    epoch_plan_hashes: Mapping[str, str],
    event: Mapping[str, Any],
    hardware_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    state = _head_state(model)
    return {
        "schema": ROLLING_SCHEMA,
        "identity": dict(identity),
        "identity_sha256": identity["identity_sha256"],
        "epoch": epoch,
        "budget": budget,
        "repair_state_dict": state,
        "repair_state_sha256": _repair_state_sha(state),
        "optimizer": optimizer.state_dict(),
        "optimizer_group_signature": optimizer_group_signature(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "metric_history": [dict(row) for row in metric_history],
        "baseline1000_metric": None if baseline_metric is None else dict(baseline_metric),
        "epoch_plan_sha256": dict(epoch_plan_hashes),
        "event": dict(event),
        "hardware_history": [dict(item) for item in hardware_history],
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }


def _validate_rolling(
    payload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    budget: int,
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    optimizer: torch.optim.AdamW,
) -> dict[str, Any]:
    required = {
        "schema",
        "identity",
        "identity_sha256",
        "epoch",
        "budget",
        "repair_state_dict",
        "repair_state_sha256",
        "optimizer",
        "optimizer_group_signature",
        "scheduler",
        "rng_state",
        "metric_history",
        "baseline1000_metric",
        "epoch_plan_sha256",
        "event",
        "hardware_history",
        "performance_acceptance_margin",
        *run_contract.OFFICIAL_FALSE_FLAGS,
    }
    _require(set(payload) == required, "rolling payload fields differ")
    _require(payload.get("schema") == ROLLING_SCHEMA, "rolling schema differs")
    _require(payload.get("identity") == dict(identity), "rolling identity differs")
    _require(
        payload.get("identity_sha256") == identity["identity_sha256"],
        "rolling identity SHA differs",
    )
    epoch = payload.get("epoch")
    _require(
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and payload.get("budget") == budget
        and 1 <= epoch <= budget,
        "rolling epoch/budget differs",
    )
    state = payload.get("repair_state_dict")
    _require(isinstance(state, Mapping), "rolling repair state differs")
    expected_keys = tuple(model.irstd_repair.state_dict())
    _require(tuple(state) == expected_keys, "rolling repair state keys/order differ")
    _require(
        payload.get("repair_state_sha256") == _repair_state_sha(state),
        "rolling repair state SHA differs",
    )
    optimizer_state = payload.get("optimizer")
    _require(isinstance(optimizer_state, Mapping), "rolling optimizer differs")
    _require(
        payload.get("optimizer_group_signature")
        == optimizer_group_signature(optimizer_state)
        == optimizer_group_signature(optimizer.state_dict()),
        "rolling optimizer signature differs",
    )
    scheduler_state = payload.get("scheduler")
    _require(isinstance(scheduler_state, Mapping), "rolling scheduler differs")
    rng_state = payload.get("rng_state")
    _require(
        isinstance(rng_state, Mapping)
        and set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "rolling RNG state differs",
    )
    history = payload.get("metric_history")
    _require(isinstance(history, list), "rolling metric history differs")
    expected_evaluations = [
        ready
        for ready in run_contract.OOF_EVALUATION_EPOCHS
        if ready <= epoch
    ]
    if identity["mode"] == "fold":
        _require(
            [row.get("epoch") for row in history] == expected_evaluations,
            "rolling fold metric history schedule differs",
        )
        baseline = payload.get("baseline1000_metric")
        _require(
            isinstance(baseline, Mapping)
            and baseline.get("candidate_kind") == "frozen_reference"
            and baseline.get("candidate_name") == BASELINE_REFERENCE_NAME,
            "rolling Baseline-1000 metric differs",
        )
    else:
        _require(history == [] and payload.get("baseline1000_metric") is None, "full rolling metrics differ")
    plan_hashes = payload.get("epoch_plan_sha256")
    _require(
        isinstance(plan_hashes, Mapping)
        and tuple(plan_hashes) == tuple(str(value) for value in range(1, epoch + 1))
        and all(_SHA256_RE.fullmatch(str(value)) is not None for value in plan_hashes.values()),
        "rolling epoch-plan hashes differ",
    )
    _require(isinstance(payload.get("event"), Mapping), "rolling event differs")
    hardware_history = payload.get("hardware_history")
    _require(isinstance(hardware_history, list) and bool(hardware_history), "hardware history differs")
    for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
        _require(payload.get(flag) is expected, f"rolling official flag differs: {flag}")
    _require(payload.get("performance_acceptance_margin") is None, "rolling margin differs")
    return dict(payload)


def _require_no_orphan_training_sidecars(run_dir: Path) -> None:
    """Reject artifacts that are not anchored by an atomically saved rolling state."""

    for name in (METRIC_HISTORY_NAME, BASELINE_METRIC_NAME):
        path = run_dir / name
        _require(
            not path.exists() and not path.is_symlink(),
            f"orphan {name} exists without rolling state",
        )
    checkpoint_dir = run_dir / EVALUATION_CHECKPOINT_DIRECTORY
    if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
        _require(
            checkpoint_dir.is_dir()
            and not checkpoint_dir.is_symlink()
            and not any(checkpoint_dir.iterdir()),
            "orphan evaluation checkpoints exist without rolling state",
        )


def _validate_resume_sidecars(
    run_dir: Path,
    rolling: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
) -> None:
    """Bind append-only metrics and eval checkpoints to the rolling snapshot."""

    history = rolling.get("metric_history")
    _require(isinstance(history, list), "rolling metric history differs")
    _require(
        _read_json_lines(run_dir / METRIC_HISTORY_NAME) == history,
        "metric JSONL differs from rolling history",
    )
    baseline = rolling.get("baseline1000_metric")
    baseline_path = run_dir / BASELINE_METRIC_NAME
    checkpoint_dir = run_dir / EVALUATION_CHECKPOINT_DIRECTORY
    if identity.get("mode") != "fold":
        _require(history == [] and baseline is None, "full rolling sidecars differ")
        _require(
            not baseline_path.exists() and not baseline_path.is_symlink(),
            "full run contains a Baseline-1000 sidecar",
        )
        if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
            _require(
                checkpoint_dir.is_dir()
                and not checkpoint_dir.is_symlink()
                and not any(checkpoint_dir.iterdir()),
                "full run contains evaluation checkpoints",
            )
        return

    _require(isinstance(baseline, Mapping), "fold rolling baseline differs")
    _require(
        _load_json_mapping(baseline_path, name=BASELINE_METRIC_NAME)
        == dict(baseline),
        "Baseline-1000 JSON differs from rolling state",
    )
    _require(
        checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink(),
        "evaluation checkpoint directory is absent or unsafe",
    )
    expected_names = {_eval_checkpoint_path(run_dir, int(row["epoch"])).name for row in history}
    observed_paths = tuple(checkpoint_dir.iterdir())
    _require(
        all(path.is_file() and not path.is_symlink() for path in observed_paths)
        and {path.name for path in observed_paths} == expected_names,
        "evaluation checkpoint set differs from rolling history",
    )
    required = {
        "schema",
        "identity",
        "identity_sha256",
        "epoch",
        "repair_state_dict",
        "repair_state_sha256",
        "metric_row",
        "base_audit",
        "performance_acceptance_margin",
        *run_contract.OFFICIAL_FALSE_FLAGS,
    }
    for row in history:
        _require(isinstance(row, Mapping), "rolling metric row differs")
        epoch = row.get("epoch")
        _require(
            isinstance(epoch, int) and not isinstance(epoch, bool),
            "rolling metric epoch differs",
        )
        payload = load_torch_artifact(_eval_checkpoint_path(run_dir, epoch))
        _require(set(payload) == required, "evaluation checkpoint fields differ")
        _require(
            payload.get("schema") == EVALUATION_CHECKPOINT_SCHEMA
            and payload.get("identity") == dict(identity)
            and payload.get("identity_sha256") == identity["identity_sha256"]
            and payload.get("epoch") == epoch
            and payload.get("metric_row") == dict(row)
            and isinstance(payload.get("base_audit"), Mapping),
            "evaluation checkpoint binding differs",
        )
        state = payload.get("repair_state_dict")
        _require(
            isinstance(state, Mapping)
            and tuple(state) == tuple(model.irstd_repair.state_dict())
            and payload.get("repair_state_sha256") == _repair_state_sha(state),
            "evaluation checkpoint repair state differs",
        )
        for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
            _require(payload.get(flag) is expected, f"evaluation official flag differs: {flag}")
        _require(
            payload.get("performance_acceptance_margin") is None,
            "evaluation checkpoint margin differs",
        )


def _evaluation_checkpoint_payload(
    *,
    identity: Mapping[str, Any],
    epoch: int,
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    metric_row: Mapping[str, Any],
    base_audit: Mapping[str, Any],
) -> dict[str, Any]:
    state = _head_state(model)
    return {
        "schema": EVALUATION_CHECKPOINT_SCHEMA,
        "identity": dict(identity),
        "identity_sha256": identity["identity_sha256"],
        "epoch": epoch,
        "repair_state_dict": state,
        "repair_state_sha256": _repair_state_sha(state),
        "metric_row": dict(metric_row),
        "base_audit": dict(base_audit),
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }


def _eval_checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / EVALUATION_CHECKPOINT_DIRECTORY / f"epoch_{epoch:04d}.pth.tar"


def _commit_or_validate_eval_checkpoint(path: Path, payload: Mapping[str, Any]) -> Path:
    if not path.exists() and not path.is_symlink():
        return exclusive_torch_save(path, payload)
    observed = load_torch_artifact(path)
    _require(observed.keys() == payload.keys(), "existing evaluation checkpoint fields differ")
    for key in payload:
        expected = payload[key]
        actual = observed[key]
        if key == "repair_state_dict":
            _require(
                isinstance(actual, Mapping)
                and tuple(actual) == tuple(expected)
                and all(torch.equal(actual[name], expected[name]) for name in expected),
                "existing evaluation checkpoint state differs",
            )
        else:
            _require(actual == expected, f"existing evaluation checkpoint differs: {key}")
    return path


def _load_oof_selection(path: Path | str) -> tuple[dict[str, Any], int]:
    selection_path = Path(path)
    outer = _load_json_mapping(selection_path, name="OOF selection")
    _require(
        outer.get("schema") == OOF_SELECTOR_SCHEMA,
        "OOF selector schema differs",
    )
    unsigned_outer = dict(outer)
    declared_outer_sha = _sha256(
        unsigned_outer.pop("selection_sha256", None),
        name="OOF selector SHA",
    )
    _require(
        canonical_json_sha256(unsigned_outer) == declared_outer_sha,
        "OOF selector SHA does not replay",
    )
    _require(
        outer.get("dataset") == DATASET
        and outer.get("role") == ROLE
        and outer.get("source_scope") == run_contract.SOURCE_SCOPE,
        "OOF selector dataset/role/source scope differs",
    )
    _require(
        outer.get("fold_assignment_sha256")
        == run_contract.FOLD_ASSIGNMENT_SHA256
        and outer.get("source_split_manifest_file_sha256")
        == run_contract.SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "OOF selector split/fold binding differs",
    )
    _require(
        outer.get("probability_threshold")
        == run_contract.PROBABILITY_THRESHOLD
        and outer.get("probability_comparison")
        == run_contract.PROBABILITY_COMPARISON,
        "OOF selector probability contract differs",
    )
    _require(
        outer.get("performance_acceptance_margin") is None,
        "OOF selector margin differs",
    )
    for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
        _require(outer.get(flag) is expected, f"OOF selector official flag differs: {flag}")
    fold_summaries = outer.get("fold_summaries")
    _require(
        isinstance(fold_summaries, list)
        and len(fold_summaries) == 3
        and all(isinstance(item, Mapping) for item in fold_summaries)
        and [item.get("fold_index") for item in fold_summaries]
        == list(run_contract.FOLD_TIE_ORDER)
        and all(
            _SHA256_RE.fullmatch(str(item.get("file_sha256"))) is not None
            for item in fold_summaries
        ),
        "OOF selector fold-summary bindings differ",
    )
    selection = outer.get("selection")
    _require(
        isinstance(selection, Mapping)
        and selection.get("schema") == run_contract.SELECTION_SCHEMA,
        "inner OOF selection schema differs",
    )
    _require(selection.get("dataset") == DATASET and selection.get("role") == ROLE, "OOF selection scope differs")
    _require(
        selection.get("candidate_epochs") == list(run_contract.OOF_EVALUATION_EPOCHS),
        "OOF selection is incomplete",
    )
    selected_epoch = selection.get("selected_epoch")
    _require(
        isinstance(selected_epoch, int)
        and not isinstance(selected_epoch, bool)
        and selected_epoch in run_contract.OOF_EVALUATION_EPOCHS,
        "OOF selected epoch differs",
    )
    _require(
        selection.get("fold_assignment_sha256")
        == run_contract.FOLD_ASSIGNMENT_SHA256
        and selection.get("source_split_manifest_file_sha256")
        == run_contract.SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "inner OOF split/fold binding differs",
    )
    _require(selection.get("performance_acceptance_margin") is None, "OOF margin differs")
    for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
        _require(selection.get(flag) is expected, f"OOF official flag differs: {flag}")
    _require(outer.get("selected_epoch") == selected_epoch, "outer/inner selected epoch differs")
    return outer, selected_epoch


def _build_identity(
    *,
    mode: str,
    fold_index: int | None,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
    cache: FrozenContextCache,
    fold_manifest: Mapping[str, Any],
    current_record: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    selected_epoch: int | None,
    oof_selection_file_sha256: str | None,
) -> dict[str, Any]:
    identity = {
        "schema": f"{SCHEMA}/identity",
        "mode": mode,
        "fold_index": fold_index,
        "dataset": DATASET,
        "role": ROLE,
        "seed": SEED,
        "training_ids_sha256": canonical_json_sha256(list(training_ids)),
        "validation_ids_sha256": canonical_json_sha256(list(validation_ids)),
        "cache_manifest_sha256": cache.manifest_sha256,
        "cache_manifest_file_sha256": cache.manifest_file_sha256,
        "fold_manifest_sha256": fold_manifest["manifest_sha256"],
        "fold_assignment_sha256": run_contract.FOLD_ASSIGNMENT_SHA256,
        "current_checkpoint_path": str(CURRENT_CHECKPOINT_PATH.resolve(strict=True)),
        "current_checkpoint_file_sha256": current_record["sha256"],
        "current_training_state_tensor_mapping_sha256": (
            CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256
        ),
        "current_training_state_tensor_mapping_hash_algorithm": (
            "tensor_mapping_sha256"
        ),
        "current_inference_state_semantic_sha256": (
            CURRENT_INFERENCE_STATE_SEMANTIC_SHA256
        ),
        "current_inference_state_semantic_hash_algorithm": (
            "state_semantic_sha256"
        ),
        "source_bundle_sha256": source_bundle["bundle_sha256"],
        "loss_manifest_sha256": canonical_json_sha256(loss_manifest()),
        "training_recipe_sha256": canonical_json_sha256(_training_recipe()),
        "selected_epoch": selected_epoch,
        "oof_selection_file_sha256": oof_selection_file_sha256,
        "positive_logit_limit": POSITIVE_LOGIT_LIMIT,
        "negative_logit_limit": NEGATIVE_LOGIT_LIMIT,
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }
    if mode == "fold":
        _require(fold_index in run_contract.FOLD_TIE_ORDER, "fold index differs")
        _require(selected_epoch is None and oof_selection_file_sha256 is None, "fold selection binding differs")
    else:
        _require(mode == "full" and fold_index is None, "full mode identity differs")
        _require(selected_epoch is not None and oof_selection_file_sha256 is not None, "full selection binding is absent")
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def _protocol(
    *,
    identity: Mapping[str, Any],
    cache: FrozenContextCache,
    fold_manifest_path: Path,
    source_bundle: Mapping[str, Any],
    initial_hardware: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": RUN_PROTOCOL_SCHEMA,
        "identity": dict(identity),
        "identity_sha256": identity["identity_sha256"],
        "cache_manifest": {
            "path": str(cache.manifest_path),
            "file_sha256": cache.manifest_file_sha256,
            "semantic_sha256": cache.manifest_sha256,
        },
        "fold_manifest": {
            "path": str(fold_manifest_path.resolve(strict=True)),
            "file_sha256": file_sha256(fold_manifest_path),
        },
        "source_bundle": dict(source_bundle),
        "training_recipe": _training_recipe(),
        "loss_manifest": loss_manifest(),
        "initial_hardware": dict(initial_hardware),
        "hardware_migration_policy": "explicit --allow-hardware-migration",
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }
    payload["protocol_sha256"] = canonical_json_sha256(payload)
    return payload


def _hardware_history(
    protocol: Mapping[str, Any],
    rolling: Mapping[str, Any] | None,
    *,
    observed: Mapping[str, Any],
    allow_migration: bool,
) -> list[dict[str, Any]]:
    initial = protocol.get("initial_hardware")
    _require(isinstance(initial, Mapping), "protocol hardware binding differs")
    if rolling is None:
        _require(dict(initial) == dict(observed), "new-run hardware changed before training")
        return [{"event": "initial", "fingerprint": dict(observed)}]
    prior = rolling.get("hardware_history")
    _require(isinstance(prior, list) and bool(prior), "rolling hardware history differs")
    last = prior[-1]
    _require(isinstance(last, Mapping) and isinstance(last.get("fingerprint"), Mapping), "last hardware record differs")
    if dict(last["fingerprint"]) == dict(observed):
        return [dict(item) for item in prior]
    _require(allow_migration, "resume hardware differs; pass --allow-hardware-migration")
    return [
        *[dict(item) for item in prior],
        {
            "event": "explicit_resume_migration",
            "from": dict(last["fingerprint"]),
            "fingerprint": dict(observed),
            "source_rolling_sha256": canonical_json_sha256(
                {
                    "identity_sha256": rolling["identity_sha256"],
                    "epoch": rolling["epoch"],
                    "repair_state_sha256": rolling["repair_state_sha256"],
                }
            ),
        },
    ]


def _summary_if_complete(run_dir: Path, identity: Mapping[str, Any]) -> Path | None:
    path = run_dir / SUMMARY_NAME
    if not path.exists() and not path.is_symlink():
        return None
    summary = _load_json_mapping(path, name="completed summary")
    expected_schema = (
        FOLD_SUMMARY_SCHEMA if identity.get("mode") == "fold" else FULL_SUMMARY_SCHEMA
    )
    _require(summary.get("schema") == expected_schema, "summary schema differs")
    _require(summary.get("status") == "complete", "existing summary is not complete")
    _require(summary.get("identity") == dict(identity), "completed identity differs")
    unsigned = dict(summary)
    declared_sha = _sha256(unsigned.pop("summary_sha256", None), name="summary SHA")
    _require(
        canonical_json_sha256(unsigned) == declared_sha,
        "completed summary SHA does not replay",
    )
    for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
        _require(summary.get(flag) is expected, f"summary official flag differs: {flag}")
    _require(summary.get("performance_acceptance_margin") is None, "summary margin differs")
    if identity.get("mode") == "fold":
        history = summary.get("evaluation_history")
        baseline = summary.get("baseline1000_metric_row")
        _require(
            isinstance(history, list)
            and [row.get("epoch") for row in history]
            == list(run_contract.OOF_EVALUATION_EPOCHS)
            and isinstance(baseline, Mapping)
            and baseline.get("candidate_kind") == "frozen_reference"
            and baseline.get("candidate_name") == BASELINE_REFERENCE_NAME,
            "completed fold metric payload is incomplete",
        )
        _require(
            _read_json_lines(run_dir / METRIC_HISTORY_NAME) == history
            and _load_json_mapping(
                run_dir / BASELINE_METRIC_NAME,
                name=BASELINE_METRIC_NAME,
            )
            == dict(baseline),
            "completed fold metric sidecars differ",
        )
    else:
        candidate = summary.get("integrated_candidate")
        _require(isinstance(candidate, Mapping), "completed candidate binding is absent")
        candidate_path = run_dir / INTEGRATED_CANDIDATE_NAME
        _require(
            candidate.get("path") == str(candidate_path.resolve(strict=True))
            and candidate.get("file_sha256") == file_sha256(candidate_path)
            and candidate.get("state_key_count") == INTEGRATED_STATE_KEY_COUNT,
            "completed candidate file binding differs",
        )
    return path


def _run_training(
    *,
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    current_inference_state: Mapping[str, torch.Tensor],
    cache: FrozenContextCache,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
    fold_index: int | None,
    fold_manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    protocol: Mapping[str, Any],
    run_dir: Path,
    device: torch.device,
    budget: int,
    resume: str,
    allow_hardware_migration: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
    model.irstd_repair.to(device=device, dtype=torch.float32)
    model.train(True)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    signature = optimizer_group_signature(optimizer.state_dict())
    _require(len(signature) == 1 and signature[0]["parameter_count"] == 29, "optimizer parameter tensors differ")
    support = cache.support(training_ids) if budget > 0 else {}
    rolling_path = run_dir / ROLLING_NAME
    rolling: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    baseline_metric: dict[str, Any] | None = None
    plan_hashes: dict[str, str] = {}
    start_epoch = 1
    if rolling_path.exists() or rolling_path.is_symlink():
        _require(resume != "never", "rolling state exists but resume=never")
        rolling = _validate_rolling(
            load_torch_artifact(rolling_path),
            identity=identity,
            budget=budget,
            model=model,
            optimizer=optimizer,
        )
        model.irstd_repair.load_state_dict(rolling["repair_state_dict"], strict=True)
        optimizer.load_state_dict(rolling["optimizer"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(rolling["scheduler"])
        history = [dict(row) for row in rolling["metric_history"]]
        baseline_metric = (
            None
            if rolling["baseline1000_metric"] is None
            else dict(rolling["baseline1000_metric"])
        )
        plan_hashes = dict(rolling["epoch_plan_sha256"])
        start_epoch = int(rolling["epoch"]) + 1
        _validate_resume_sidecars(
            run_dir,
            rolling,
            identity=identity,
            model=model,
        )
    else:
        _require(resume != "required", "resume=required but rolling state is absent")
        _require_no_orphan_training_sidecars(run_dir)
    observed_hardware = _device_fingerprint(device)
    hardware_history = _hardware_history(
        protocol,
        rolling,
        observed=observed_hardware,
        allow_migration=allow_hardware_migration,
    )
    if rolling is not None:
        prior_hardware = rolling["hardware_history"]
        assert isinstance(prior_hardware, list)
        _restore_rng_after_hardware_check(
            rolling["rng_state"],
            migrated=len(hardware_history) > len(prior_hardware),
        )
    base_audit = audit_frozen_current_base(model, current_inference_state)
    _audit_limit_buffers(model.irstd_repair.state_dict(), prefixed=False)

    if identity["mode"] == "fold" and not history:
        assert fold_index is not None
        baseline_metric = evaluate_cached_fold(
            model,
            cache,
            validation_ids,
            fold_index=fold_index,
            epoch=0,
            fold_manifest=fold_manifest,
            device=device,
            candidate_kind="frozen_reference",
            candidate_name=BASELINE_REFERENCE_NAME,
        )
        _commit_or_validate_json(run_dir / BASELINE_METRIC_NAME, baseline_metric)
        epoch_zero = evaluate_cached_fold(
            model,
            cache,
            validation_ids,
            fold_index=fold_index,
            epoch=0,
            fold_manifest=fold_manifest,
            device=device,
        )
        history.append(epoch_zero)
        _append_json_line(run_dir / METRIC_HISTORY_NAME, epoch_zero)
        payload = _evaluation_checkpoint_payload(
            identity=identity,
            epoch=0,
            model=model,
            metric_row=epoch_zero,
            base_audit=base_audit,
        )
        _commit_or_validate_eval_checkpoint(_eval_checkpoint_path(run_dir, 0), payload)
        model.train(True)

    for epoch in range(start_epoch, budget + 1):
        plan = build_epoch_plan(training_ids, support, epoch=epoch)
        plan_sha = epoch_plan_sha256(plan)
        expected_prior = plan_hashes.get(str(epoch))
        _require(expected_prior in (None, plan_sha), "replayed epoch plan differs")
        plan_hashes[str(epoch)] = plan_sha
        dataset = CachedPatchDataset(cache, plan, epoch=epoch)
        loader: DataLoader[dict[str, object]] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )
        training = train_one_epoch(
            model,
            loader,
            optimizer,
            epoch=epoch,
            device=device,
        )
        scheduler.step()
        evaluation: dict[str, Any] | None = None
        if identity["mode"] == "fold" and epoch % EVALUATE_EVERY == 0:
            assert fold_index is not None
            evaluation = evaluate_cached_fold(
                model,
                cache,
                validation_ids,
                fold_index=fold_index,
                epoch=epoch,
                fold_manifest=fold_manifest,
                device=device,
            )
            history.append(evaluation)
            _append_json_line(run_dir / METRIC_HISTORY_NAME, evaluation)
            base_audit = audit_frozen_current_base(model, current_inference_state)
            payload = _evaluation_checkpoint_payload(
                identity=identity,
                epoch=epoch,
                model=model,
                metric_row=evaluation,
                base_audit=base_audit,
            )
            _commit_or_validate_eval_checkpoint(
                _eval_checkpoint_path(run_dir, epoch), payload
            )
            model.train(True)
        base_audit = audit_frozen_current_base(model, current_inference_state)
        event = {
            "epoch": epoch,
            "training": training,
            "evaluation": evaluation,
            "learning_rate_after_step": float(optimizer.param_groups[0]["lr"]),
            "base_audit": base_audit,
        }
        rolling_payload = _rolling_payload(
            identity=identity,
            epoch=epoch,
            budget=budget,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_history=history,
            baseline_metric=baseline_metric,
            epoch_plan_hashes=plan_hashes,
            event=event,
            hardware_history=hardware_history,
        )
        atomic_rolling_torch_save(rolling_path, rolling_payload)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "mode": identity["mode"],
                    "fold_index": fold_index,
                    "epoch": epoch,
                    "budget": budget,
                    "evaluated": evaluation is not None,
                    "mean_total_loss": training["mean_loss_components"]["total"],
                    "official_test_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    final_audit = audit_frozen_current_base(model, current_inference_state)
    return history, baseline_metric, final_audit, hardware_history


def run(args: argparse.Namespace) -> Path:
    _require(args.mode in MODES, "mode must be fold or full")
    _require(args.resume in ("auto", "never", "required"), "resume policy differs")
    determinism = configure_determinism(SEED)
    # Default PyTorch thread counts on this host are much larger than useful
    # for stacking the 20 cached crop tensors.  With 96 threads, a single
    # 16-sample collate is over 60x slower from scheduling/memory contention.
    # Fixing these execution-only counts leaves tensor values and the training
    # recipe unchanged while making the cache pipeline practically runnable.
    torch.set_num_threads(CPU_INTRAOP_THREADS)
    try:
        torch.set_num_interop_threads(CPU_INTEROP_THREADS)
    except RuntimeError as error:
        _require(
            torch.get_num_interop_threads() == CPU_INTEROP_THREADS,
            f"cannot install formal CPU interop thread count: {error}",
        )
    _require(
        torch.get_num_threads() == CPU_INTRAOP_THREADS
        and torch.get_num_interop_threads() == CPU_INTEROP_THREADS,
        "formal CPU execution thread counts differ",
    )
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    _require(
        torch.get_default_dtype() == torch.float32
        and torch.get_float32_matmul_precision() == "highest"
        and
        torch.backends.cuda.matmul.allow_tf32 is False
        and torch.backends.cudnn.allow_tf32 is False,
        "formal FP32/highest/TF32-off precision contract differs",
    )
    determinism.update(
        {
            "default_dtype": "torch.float32",
            "float32_matmul_precision": "highest",
            "cpu_intraop_threads": CPU_INTRAOP_THREADS,
            "cpu_interop_threads": CPU_INTEROP_THREADS,
        }
    )
    device = torch.device(args.device)
    _require(device.type in ("cpu", "cuda"), "unsupported device")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    hardware = _device_fingerprint(device)
    if args.expected_gpu_uuid is not None:
        _require(
            device.type == "cuda" and hardware.get("uuid") == args.expected_gpu_uuid,
            "observed GPU UUID differs",
        )

    cache = FrozenContextCache(args.cache_root)
    fold_manifest_path = Path(args.fold_manifest)
    fold_manifest = load_frozen_fold_manifest(fold_manifest_path)
    validate_cache_fold_alignment(cache, fold_manifest)
    if args.mode == "fold":
        _require(args.fold_index in run_contract.FOLD_TIE_ORDER, "fold mode requires --fold-index 0/1/2")
        _require(args.oof_selection is None, "fold mode cannot accept OOF selection")
        fold = fold_manifest["folds"][args.fold_index]
        training_ids = tuple(fold["training_ids"])
        validation_ids = tuple(fold["validation_ids"])
        budget = TRAIN_EPOCHS
        selected_epoch = None
        selection_file_sha = None
        selection = None
    else:
        _require(args.fold_index is None, "full mode cannot accept --fold-index")
        _require(args.oof_selection is not None, "full mode requires --oof-selection")
        selection, selected_epoch = _load_oof_selection(args.oof_selection)
        _require(
            selection.get("fold_manifest_sha256")
            == fold_manifest.get("manifest_sha256"),
            "OOF selector fold-manifest binding differs",
        )
        selection_file_sha = file_sha256(Path(args.oof_selection))
        training_ids = tuple(cache.sample_ids)
        validation_ids = ()
        budget = selected_epoch

    training_state, current_inference_state, current_record = _load_immutable_current()
    model, model_metadata = build_formal_irstd_bgcr_model(training_state)
    del training_state
    validate_formal_irstd_bgcr_model(
        model,
        expected_current_inference_state=current_inference_state,
        require_identity_initialization=True,
    )
    _audit_limit_buffers(model.state_dict(), prefixed=True)
    source_bundle = _source_bundle()
    identity = _build_identity(
        mode=args.mode,
        fold_index=args.fold_index,
        training_ids=training_ids,
        validation_ids=validation_ids,
        cache=cache,
        fold_manifest=fold_manifest,
        current_record=current_record,
        source_bundle=source_bundle,
        selected_epoch=selected_epoch,
        oof_selection_file_sha256=selection_file_sha,
    )
    run_dir = Path(args.run_dir)
    _require(not run_dir.is_symlink(), "run directory cannot be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = _summary_if_complete(run_dir, identity)
    if completed is not None:
        _require(args.resume != "never", "completed run exists but resume=never")
        return completed
    protocol_path = run_dir / RUN_PROTOCOL_NAME
    if protocol_path.exists() or protocol_path.is_symlink():
        observed_protocol = _load_json_mapping(protocol_path, name="run protocol")
        initial_hardware = observed_protocol.get("initial_hardware")
        _require(isinstance(initial_hardware, Mapping), "protocol initial hardware differs")
        protocol = _protocol(
            identity=identity,
            cache=cache,
            fold_manifest_path=fold_manifest_path,
            source_bundle=source_bundle,
            initial_hardware=initial_hardware,
        )
        _require(observed_protocol == protocol, "existing run protocol differs")
    else:
        protocol = _protocol(
            identity=identity,
            cache=cache,
            fold_manifest_path=fold_manifest_path,
            source_bundle=source_bundle,
            initial_hardware=hardware,
        )
        protocol_path = _commit_or_validate_json(protocol_path, protocol)
    history, baseline_metric, final_audit, hardware_history = _run_training(
        model=model,
        current_inference_state=current_inference_state,
        cache=cache,
        training_ids=training_ids,
        validation_ids=validation_ids,
        fold_index=args.fold_index,
        fold_manifest=fold_manifest,
        identity=identity,
        protocol=protocol,
        run_dir=run_dir,
        device=device,
        budget=budget,
        resume=args.resume,
        allow_hardware_migration=bool(args.allow_hardware_migration),
    )

    candidate_path: Path | None = None
    candidate_payload: dict[str, Any] | None = None
    if args.mode == "full":
        integrated_state = _integrated_cpu_state(model)
        _audit_limit_buffers(integrated_state, prefixed=True)
        _require(len(integrated_state) == 595, "formal integrated candidate must have 595 keys")
        base_state = {
            name: tensor
            for name, tensor in integrated_state.items()
            if not name.startswith(IRSTD_CRR_STATE_PREFIX)
        }
        repair_state = {
            name.removeprefix(IRSTD_CRR_STATE_PREFIX): tensor
            for name, tensor in integrated_state.items()
            if name.startswith(IRSTD_CRR_STATE_PREFIX)
        }
        _require(len(base_state) == 564 and len(repair_state) == 31, "candidate state partition differs")
        _require(
            state_semantic_sha256(base_state)
            == CURRENT_INFERENCE_STATE_SEMANTIC_SHA256,
            "candidate 564-key Current base changed",
        )
        candidate_payload = {
            "schema": INTEGRATED_CANDIDATE_SCHEMA,
            "dataset": DATASET,
            "role": ROLE,
            "mode": "full",
            "seed": SEED,
            "epoch": budget,
            "oof_selection": selection,
            "oof_selection_file_sha256": selection_file_sha,
            "identity": identity,
            "identity_sha256": identity["identity_sha256"],
            "state_dict": integrated_state,
            "state_key_count": len(integrated_state),
            "state_semantic_sha256": state_semantic_sha256(integrated_state),
            "state_hash_algorithm": "state_semantic_sha256",
            "current_base_state_key_count": len(base_state),
            "current_base_state_semantic_sha256": state_semantic_sha256(base_state),
            "current_base_state_hash_algorithm": "state_semantic_sha256",
            "repair_state_key_count": len(repair_state),
            "repair_state_semantic_sha256": state_semantic_sha256(repair_state),
            "repair_state_hash_algorithm": "state_semantic_sha256",
            "architecture_manifest": model.architecture_manifest(),
            "base_audit": final_audit,
            "model_builder_metadata": model_metadata,
            "training_recipe": _training_recipe(),
            "loss_manifest": loss_manifest(),
            "run_protocol_sha256": protocol["protocol_sha256"],
            "performance_acceptance_margin": None,
            **run_contract.OFFICIAL_FALSE_FLAGS,
        }
        candidate_path = run_dir / INTEGRATED_CANDIDATE_NAME
        if not candidate_path.exists() and not candidate_path.is_symlink():
            exclusive_torch_save(candidate_path, candidate_payload)
        else:
            observed = load_torch_artifact(candidate_path)
            observed_state = observed.get("state_dict")
            _require(
                isinstance(observed_state, Mapping)
                and tuple(observed_state) == tuple(integrated_state)
                and len(observed_state) == INTEGRATED_STATE_KEY_COUNT
                and all(
                    isinstance(tensor, torch.Tensor)
                    and bool(torch.isfinite(tensor).all())
                    for tensor in observed_state.values()
                ),
                "existing integrated candidate tensor contract differs",
            )
            _audit_limit_buffers(observed_state, prefixed=True)
            observed_base = {
                name: tensor
                for name, tensor in observed_state.items()
                if not name.startswith(IRSTD_CRR_STATE_PREFIX)
            }
            observed_repair = {
                name.removeprefix(IRSTD_CRR_STATE_PREFIX): tensor
                for name, tensor in observed_state.items()
                if name.startswith(IRSTD_CRR_STATE_PREFIX)
            }
            _require(
                observed.get("state_key_count") == len(observed_state) == 595
                and observed.get("state_hash_algorithm")
                == "state_semantic_sha256"
                and observed.get("state_semantic_sha256")
                == state_semantic_sha256(observed_state)
                == candidate_payload["state_semantic_sha256"]
                and observed.get("current_base_state_key_count")
                == len(observed_base)
                == 564
                and observed.get("current_base_state_hash_algorithm")
                == "state_semantic_sha256"
                and observed.get("current_base_state_semantic_sha256")
                == state_semantic_sha256(observed_base)
                == CURRENT_INFERENCE_STATE_SEMANTIC_SHA256
                and observed.get("repair_state_key_count")
                == len(observed_repair)
                == 31
                and observed.get("repair_state_hash_algorithm")
                == "state_semantic_sha256"
                and observed.get("repair_state_semantic_sha256")
                == state_semantic_sha256(observed_repair),
                "existing integrated candidate state hashes differ",
            )
            _require(
                all(
                    torch.equal(observed_state[name], integrated_state[name])
                    for name in integrated_state
                ),
                "existing integrated candidate tensors differ",
            )
            observed_manifest = dict(observed)
            expected_manifest = dict(candidate_payload)
            del observed_manifest["state_dict"]
            del expected_manifest["state_dict"]
            _require(
                canonical_json_sha256(observed_manifest)
                == canonical_json_sha256(expected_manifest)
                and observed_manifest == expected_manifest,
                "existing integrated candidate manifest differs",
            )

    summary: dict[str, Any] = {
        "schema": FOLD_SUMMARY_SCHEMA if args.mode == "fold" else FULL_SUMMARY_SCHEMA,
        "status": "complete",
        "mode": args.mode,
        "fold_index": args.fold_index,
        "dataset": DATASET,
        "role": ROLE,
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "run_protocol": str(protocol_path.resolve(strict=True)),
        "run_protocol_sha256": protocol["protocol_sha256"],
        "training_epoch_budget": budget,
        "integrated_candidate": (
            None
            if candidate_path is None
            else {
                "path": str(candidate_path.resolve(strict=True)),
                "file_sha256": file_sha256(candidate_path),
                "state_semantic_sha256": candidate_payload[
                    "state_semantic_sha256"
                ],
                "state_hash_algorithm": "state_semantic_sha256",
                "state_key_count": 595,
            }
        ),
        "final_base_audit": final_audit,
        "hardware_history": hardware_history,
        "determinism": determinism,
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }
    if args.mode == "fold":
        _require(
            args.fold_index is not None
            and baseline_metric is not None
            and [row["epoch"] for row in history]
            == list(run_contract.OOF_EVALUATION_EPOCHS),
            "completed fold history/baseline is incomplete",
        )
        summary.update(
            {
                "source_scope": run_contract.SOURCE_SCOPE,
                "fold_assignment_sha256": run_contract.FOLD_ASSIGNMENT_SHA256,
                "fold_manifest_sha256": fold_manifest["manifest_sha256"],
                "source_split_manifest_file_sha256": (
                    run_contract.SOURCE_SPLIT_MANIFEST_FILE_SHA256
                ),
                "probability_threshold": run_contract.PROBABILITY_THRESHOLD,
                "probability_comparison": run_contract.PROBABILITY_COMPARISON,
                "evaluation_history": history,
                "baseline1000_metric_row": baseline_metric,
            }
        )
    else:
        summary.update(
            {
                "source_scope": run_contract.SOURCE_SCOPE,
                "fold_assignment_sha256": run_contract.FOLD_ASSIGNMENT_SHA256,
                "fold_manifest_sha256": fold_manifest["manifest_sha256"],
                "oof_selection": selection,
                "selected_epoch": budget,
            }
        )
    summary["summary_sha256"] = canonical_json_sha256(summary)
    return exclusive_json(run_dir / SUMMARY_NAME, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--fold-index", type=int, choices=run_contract.FOLD_TIE_ORDER)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--oof-selection", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--allow-hardware-migration", action="store_true")
    return parser


def main() -> int:
    path = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "event": "irstd_bgcr_complete",
                "summary": str(path.resolve(strict=True)),
                "official_test_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_REFERENCE_NAME",
    "BATCH_SIZE",
    "CACHE_FIELDS",
    "CACHE_FLOAT_FIELDS",
    "CACHE_ID_FIELDS",
    "CACHE_MASK_FIELDS",
    "CPU_INTEROP_THREADS",
    "CPU_INTRAOP_THREADS",
    "CENTER_PATCH_SIZE",
    "CONTEXT_HALO",
    "COUNTERFACTUAL_MODES",
    "CachedPatchDataset",
    "EpochPlanItem",
    "FrozenCacheSample",
    "FrozenContextCache",
    "IRSTDBGCRTrainingError",
    "OUTER_PATCH_SIZE",
    "SEED",
    "SOURCE_CLASSES",
    "TRAIN_EPOCHS",
    "build_epoch_plan",
    "build_optimizer",
    "build_parser",
    "build_scheduler",
    "canonical_json_sha256",
    "center_crop_tensor",
    "epoch_plan_sha256",
    "evaluate_cached_fold",
    "extract_outer_context_patch",
    "load_frozen_fold_manifest",
    "run",
    "train_one_epoch",
    "validate_cache_fold_alignment",
]
