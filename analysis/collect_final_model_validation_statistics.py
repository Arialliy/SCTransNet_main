#!/usr/bin/env python3
"""Write-once internal-validation prediction cache for final-model F1 audits.

This module intentionally does not build a dataset, load a checkpoint, or run
model inference.  It accepts already collected per-image probabilities and
targets, binds them to checkpoint/dataset/evaluator/mode identities, and
stores a compact cache from which fixed-threshold sufficient statistics and
paired-bootstrap resamples can be recomputed without another model forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.evaluate_pd_fa_sweep import ValidationMetrics


CACHE_SCHEMA = "sctransnet_final_model_validation_prediction_cache_v1"
IDENTITY_SCHEMA = "sctransnet_final_model_validation_cache_identity_v1"
MODE_SCHEMA = "sctransnet_final_model_qfg_audit_mode_v1"
STATISTICS_SCHEMA = "sctransnet_final_model_image_sufficient_statistics_v1"
DATA_SCOPE = "internal_validation"
EXPECTED_VALIDATION_COUNT = 133
PREDICTION_COMPARISON = "probability > threshold"
ARRAY_KEYS = frozenset(
    {
        "probabilities",
        "targets",
        "offsets",
        "heights",
        "widths",
        "losses",
    }
)
_LEVEL_MODE = re.compile(r"level_([1-4])_off")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def validation_identifier_sha256(identifiers: Sequence[str]) -> str:
    if (
        isinstance(identifiers, (str, bytes))
        or not isinstance(identifiers, Sequence)
        or not identifiers
        or not all(isinstance(value, str) and value for value in identifiers)
    ):
        raise ValueError("validation identifiers must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("validation identifiers must be unique")
    canonical = "\n".join(sorted(identifiers))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evaluation_contract(
    *,
    match_radius: float,
    tiny_area: int,
) -> dict[str, Any]:
    if (
        not isinstance(match_radius, (int, float))
        or isinstance(match_radius, bool)
        or not math.isfinite(float(match_radius))
        or float(match_radius) <= 0
    ):
        raise ValueError("match_radius must be finite and positive")
    if (
        not isinstance(tiny_area, int)
        or isinstance(tiny_area, bool)
        or tiny_area < 0
    ):
        raise ValueError("tiny_area must be a non-negative integer")
    return {
        "prediction_comparison": PREDICTION_COMPARISON,
        "match_radius": float(match_radius),
        "tiny_area": tiny_area,
    }


def normalize_mode(mode: str) -> dict[str, Any]:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    if mode == "full":
        levels: tuple[int, ...] = ()
    elif mode == "all_off":
        levels = (0, 1, 2, 3)
    else:
        match = _LEVEL_MODE.fullmatch(mode)
        if match is None:
            raise ValueError(
                "mode must be full, all_off, or level_1_off..level_4_off"
            )
        levels = (int(match.group(1)) - 1,)
    core = {
        "schema": MODE_SCHEMA,
        "name": mode,
        "knockout_level_indices_zero_based": list(levels),
        "diagnostic_only": mode != "full",
    }
    return {
        **core,
        "sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def build_cache_identity(
    *,
    checkpoint_sha256: str,
    dataset_sha256: str,
    evaluator_sha256: str,
    mode: str,
    normalization_sha256: str,
    source_lock_sha256: str,
    validation_ids_sha256: str,
    validation_count: int,
    match_radius: float,
    tiny_area: int,
) -> dict[str, Any]:
    mode_binding = normalize_mode(mode)
    if (
        isinstance(validation_count, bool)
        or not isinstance(validation_count, int)
        or validation_count != EXPECTED_VALIDATION_COUNT
    ):
        raise ValueError(
            f"validation_count must equal {EXPECTED_VALIDATION_COUNT}"
        )
    core: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "data_scope": DATA_SCOPE,
        "checkpoint_sha256": _sha256(
            checkpoint_sha256,
            "checkpoint_sha256",
        ),
        "dataset_sha256": _sha256(dataset_sha256, "dataset_sha256"),
        "evaluator_sha256": _sha256(
            evaluator_sha256,
            "evaluator_sha256",
        ),
        "normalization_sha256": _sha256(
            normalization_sha256,
            "normalization_sha256",
        ),
        "source_lock_sha256": _sha256(
            source_lock_sha256,
            "source_lock_sha256",
        ),
        "validation_ids_sha256": _sha256(
            validation_ids_sha256,
            "validation_ids_sha256",
        ),
        "validation_count": validation_count,
        "evaluation_contract": _evaluation_contract(
            match_radius=match_radius,
            tiny_area=tiny_area,
        ),
        "mode": mode_binding,
    }
    compatibility_core = {
        key: value for key, value in core.items() if key != "mode"
    }
    return {
        **core,
        "compatibility_sha256": sha256_bytes(
            canonical_json_bytes(compatibility_core)
        ),
        "cache_key_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def validate_cache_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError("identity must be a mapping")
    _require(
        identity.get("schema") == IDENTITY_SCHEMA,
        "cache identity schema differs",
    )
    _require(
        identity.get("data_scope") == DATA_SCOPE,
        "F1 cache only permits internal_validation data",
    )
    mode = identity.get("mode")
    _require(isinstance(mode, Mapping), "cache identity mode is missing")
    evaluation = identity.get("evaluation_contract")
    _require(
        isinstance(evaluation, Mapping),
        "cache identity evaluation contract is missing",
    )
    expected = build_cache_identity(
        checkpoint_sha256=_sha256(
            identity.get("checkpoint_sha256"),
            "checkpoint_sha256",
        ),
        dataset_sha256=_sha256(
            identity.get("dataset_sha256"),
            "dataset_sha256",
        ),
        evaluator_sha256=_sha256(
            identity.get("evaluator_sha256"),
            "evaluator_sha256",
        ),
        normalization_sha256=_sha256(
            identity.get("normalization_sha256"),
            "normalization_sha256",
        ),
        source_lock_sha256=_sha256(
            identity.get("source_lock_sha256"),
            "source_lock_sha256",
        ),
        validation_ids_sha256=_sha256(
            identity.get("validation_ids_sha256"),
            "validation_ids_sha256",
        ),
        validation_count=identity.get("validation_count"),
        match_radius=evaluation.get("match_radius"),
        tiny_area=evaluation.get("tiny_area"),
        mode=str(mode.get("name")),
    )
    _require(
        canonical_json_bytes(dict(identity)) == canonical_json_bytes(expected),
        "cache identity conflicts with its hashes",
    )
    return expected


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    image_id: str
    probability: np.ndarray
    target: np.ndarray
    loss: float | None = None


@dataclass(frozen=True, slots=True)
class PredictionCache:
    identity: dict[str, Any]
    records: tuple[PredictionRecord, ...]
    match_radius: float
    tiny_area: int
    content_sha256: str


class PredictionCacheCollector:
    """Collect internal-validation outputs one image at a time, then seal once."""

    def __init__(
        self,
        *,
        identity: Mapping[str, Any],
        match_radius: float = 3.0,
        tiny_area: int = 9,
    ) -> None:
        self._identity = validate_cache_identity(identity)
        contract = _evaluation_contract(
            match_radius=match_radius,
            tiny_area=tiny_area,
        )
        _require(
            contract == self._identity["evaluation_contract"],
            "prediction collector metric contract differs from identity",
        )
        self._match_radius = match_radius
        self._tiny_area = tiny_area
        self._records: list[PredictionRecord] = []
        self._image_ids: set[str] = set()
        self._sealed = False

    @property
    def image_count(self) -> int:
        return len(self._records)

    def append(
        self,
        *,
        image_id: str,
        probability: np.ndarray,
        target: np.ndarray,
        loss: float | None = None,
    ) -> None:
        if self._sealed:
            raise RuntimeError("prediction collector is already sealed")
        if isinstance(image_id, str) and image_id in self._image_ids:
            raise ValueError(
                f"duplicate prediction image ID: {image_id!r}"
            )
        record = _normalize_record(
            PredictionRecord(
                image_id=image_id,
                probability=probability,
                target=target,
                loss=loss,
            ),
            len(self._records),
        )
        self._records.append(record)
        self._image_ids.add(record.image_id)

    def seal(self) -> PredictionCache:
        if self._sealed:
            raise RuntimeError("prediction collector is already sealed")
        cache = create_prediction_cache(
            self._records,
            identity=self._identity,
            match_radius=self._match_radius,
            tiny_area=self._tiny_area,
        )
        self._sealed = True
        return cache


def _normalize_record(record: PredictionRecord, index: int) -> PredictionRecord:
    if not isinstance(record, PredictionRecord):
        raise TypeError(f"record[{index}] must be a PredictionRecord")
    if not isinstance(record.image_id, str) or not record.image_id:
        raise ValueError(f"record[{index}] image_id must be non-empty")
    probability = np.asarray(record.probability)
    target = np.asarray(record.target)
    if probability.ndim != 2 or target.ndim != 2:
        raise ValueError(f"record[{index}] arrays must be two-dimensional")
    if probability.shape != target.shape or probability.size == 0:
        raise ValueError(f"record[{index}] probability/target shape differs")
    if probability.dtype != np.float32:
        raise ValueError(f"record[{index}] probability must be FP32")
    if not np.isfinite(probability).all():
        raise ValueError(f"record[{index}] probability must be finite")
    if bool(np.logical_or(probability < 0.0, probability > 1.0).any()):
        raise ValueError(f"record[{index}] probability must lie in [0, 1]")
    if not np.isfinite(target).all():
        raise ValueError(f"record[{index}] target must be finite")
    target_binary = (target > 0.5).astype(np.uint8, copy=False)
    if record.loss is None:
        raise ValueError(
            f"record[{index}] loss must be supplied by the formal FP32 "
            "BCELoss evaluation"
        )
    loss = float(record.loss)
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError(
            f"record[{index}] formal FP32 BCELoss must be finite "
            "and non-negative"
        )
    return PredictionRecord(
        image_id=record.image_id,
        probability=np.ascontiguousarray(probability),
        target=np.ascontiguousarray(target_binary),
        loss=loss,
    )


def prediction_content_sha256(
    records: Sequence[PredictionRecord],
) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            np.asarray(record.probability.shape, dtype="<i8").tobytes()
        )
        digest.update(record.probability.astype("<f4", copy=False).tobytes())
        digest.update(record.target.astype("u1", copy=False).tobytes())
        digest.update(np.asarray([record.loss], dtype="<f8").tobytes())
    return digest.hexdigest()


def create_prediction_cache(
    records: Iterable[PredictionRecord],
    *,
    identity: Mapping[str, Any],
    match_radius: float = 3.0,
    tiny_area: int = 9,
) -> PredictionCache:
    validated_identity = validate_cache_identity(identity)
    normalized = tuple(
        _normalize_record(record, index)
        for index, record in enumerate(records)
    )
    _require(bool(normalized), "prediction cache must contain at least one image")
    identifiers = [record.image_id for record in normalized]
    _require(
        len(identifiers) == len(set(identifiers)),
        "prediction cache image IDs must be unique",
    )
    contract = _evaluation_contract(
        match_radius=match_radius,
        tiny_area=tiny_area,
    )
    _require(
        len(normalized) == validated_identity["validation_count"],
        "prediction cache does not contain the complete validation set",
    )
    _require(
        validation_identifier_sha256(identifiers)
        == validated_identity["validation_ids_sha256"],
        "prediction cache validation image IDs differ from identity",
    )
    _require(
        contract == validated_identity["evaluation_contract"],
        "prediction cache metric contract differs from identity",
    )
    return PredictionCache(
        identity=validated_identity,
        records=normalized,
        match_radius=float(match_radius),
        tiny_area=tiny_area,
        content_sha256=prediction_content_sha256(normalized),
    )


def cache_paths(
    output_dir: Path,
    identity: Mapping[str, Any],
) -> tuple[Path, Path]:
    validated = validate_cache_identity(identity)
    key = validated["cache_key_sha256"]
    root = Path(output_dir).resolve()
    return (
        root / f"{key}.cache.json",
        root / f"{key}.arrays.npz",
    )


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace F1 cache output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_arrays(
    path: Path,
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
    offsets: np.ndarray,
    heights: np.ndarray,
    widths: np.ndarray,
    losses: np.ndarray,
) -> None:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace F1 cache output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                probabilities=probabilities,
                targets=targets,
                offsets=offsets,
                heights=heights,
                widths=widths,
                losses=losses,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def write_prediction_cache(
    cache: PredictionCache,
    output_dir: Path,
) -> Path:
    if not isinstance(cache, PredictionCache):
        raise TypeError("cache must be a PredictionCache")
    validated = create_prediction_cache(
        cache.records,
        identity=cache.identity,
        match_radius=cache.match_radius,
        tiny_area=cache.tiny_area,
    )
    _require(
        validated.content_sha256 == cache.content_sha256,
        "cache content SHA differs",
    )
    metadata_path, arrays_path = cache_paths(output_dir, cache.identity)
    if (
        metadata_path.exists()
        or metadata_path.is_symlink()
        or arrays_path.exists()
        or arrays_path.is_symlink()
    ):
        raise FileExistsError("refusing to replace existing F1 cache pair")

    sizes = np.asarray(
        [record.probability.shape for record in cache.records],
        dtype="<i8",
    )
    counts = np.asarray(
        [record.probability.size for record in cache.records],
        dtype="<i8",
    )
    offsets = np.concatenate(
        (np.zeros(1, dtype="<i8"), np.cumsum(counts, dtype="<i8"))
    )
    probabilities = np.concatenate(
        [record.probability.reshape(-1) for record in cache.records]
    ).astype("<f4", copy=False)
    targets = np.concatenate(
        [record.target.reshape(-1) for record in cache.records]
    ).astype("u1", copy=False)
    losses = np.asarray(
        [record.loss for record in cache.records],
        dtype="<f8",
    )
    arrays_created = False
    try:
        _atomic_create_arrays(
            arrays_path,
            probabilities=probabilities,
            targets=targets,
            offsets=offsets,
            heights=sizes[:, 0],
            widths=sizes[:, 1],
            losses=losses,
        )
        arrays_created = True
        metadata = {
            "schema": CACHE_SCHEMA,
            "status": "complete",
            "data_scope": DATA_SCOPE,
            "official_test_accessed": False,
            "identity": cache.identity,
            "image_count": len(cache.records),
            "image_ids": [record.image_id for record in cache.records],
            "evaluation_contract": {
                "prediction_comparison": PREDICTION_COMPARISON,
                "match_radius": cache.match_radius,
                "tiny_area": cache.tiny_area,
                "probability_dtype": "float32",
                "target_storage": "uint8_binary_target_gt_0.5",
            },
            "prediction_content_sha256": cache.content_sha256,
            "arrays": {
                "filename": arrays_path.name,
                "sha256": sha256_file(arrays_path),
                "allow_pickle": False,
                "keys": sorted(ARRAY_KEYS),
            },
            "write_once": True,
            "overwrite_forbidden": True,
        }
        _atomic_create_bytes(metadata_path, canonical_json_bytes(metadata))
    except BaseException:
        if (
            arrays_created
            and arrays_path.is_file()
            and not arrays_path.is_symlink()
        ):
            arrays_path.unlink()
        raise
    return metadata_path


def _load_json(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"expected a regular cache metadata file: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid cache metadata JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("cache metadata must be one JSON object")
    _require(
        raw == canonical_json_bytes(value),
        "cache metadata is not canonical JSON",
    )
    return value


def load_prediction_cache(
    metadata_path: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> PredictionCache:
    requested_metadata = Path(metadata_path)
    if requested_metadata.is_symlink():
        raise ValueError(
            f"cache metadata must not be a symlink: {requested_metadata}"
        )
    metadata_file = requested_metadata.resolve()
    metadata = _load_json(metadata_file)
    expected_metadata_fields = {
        "schema",
        "status",
        "data_scope",
        "official_test_accessed",
        "identity",
        "image_count",
        "image_ids",
        "evaluation_contract",
        "prediction_content_sha256",
        "arrays",
        "write_once",
        "overwrite_forbidden",
    }
    _require(
        set(metadata) == expected_metadata_fields,
        "cache metadata fields differ",
    )
    _require(metadata.get("schema") == CACHE_SCHEMA, "cache schema differs")
    _require(metadata.get("status") == "complete", "cache is incomplete")
    _require(
        metadata.get("data_scope") == DATA_SCOPE,
        "cache is not internal validation",
    )
    _require(
        metadata.get("official_test_accessed") is False,
        "cache official-test boundary differs",
    )
    _require(
        metadata.get("write_once") is True
        and metadata.get("overwrite_forbidden") is True,
        "cache write-once policy differs",
    )
    identity = validate_cache_identity(metadata.get("identity", {}))
    if expected_identity is not None:
        expected = validate_cache_identity(expected_identity)
        _require(
            canonical_json_bytes(identity) == canonical_json_bytes(expected),
            "cache identity differs from expected identity",
        )
    array_binding = metadata.get("arrays")
    _require(isinstance(array_binding, Mapping), "cache arrays binding missing")
    _require(
        set(array_binding) == {"filename", "sha256", "allow_pickle", "keys"},
        "cache arrays binding fields differ",
    )
    filename = array_binding.get("filename")
    _require(
        isinstance(filename, str)
        and filename
        and Path(filename).name == filename,
        "cache arrays filename is invalid",
    )
    arrays_path = metadata_file.parent / filename
    _require(
        sha256_file(arrays_path) == array_binding.get("sha256"),
        "cache arrays SHA differs",
    )
    _require(
        array_binding.get("allow_pickle") is False,
        "cache arrays pickle policy differs",
    )
    _require(
        array_binding.get("keys") == sorted(ARRAY_KEYS),
        "cache arrays key contract differs",
    )
    with np.load(arrays_path, allow_pickle=False) as arrays:
        _require(set(arrays.files) == ARRAY_KEYS, "cache array key set differs")
        probabilities = np.asarray(arrays["probabilities"])
        targets = np.asarray(arrays["targets"])
        offsets = np.asarray(arrays["offsets"])
        heights = np.asarray(arrays["heights"])
        widths = np.asarray(arrays["widths"])
        losses = np.asarray(arrays["losses"])

    image_ids = metadata.get("image_ids")
    count = metadata.get("image_count")
    _require(type(count) is int and count > 0, "cache image count is invalid")
    _require(
        isinstance(image_ids, list)
        and len(image_ids) == count
        and all(isinstance(value, str) and value for value in image_ids),
        "cache image IDs are invalid",
    )
    _require(
        probabilities.dtype == np.dtype("<f4"),
        "cache probability dtype differs",
    )
    _require(targets.dtype == np.dtype("u1"), "cache target dtype differs")
    _require(
        offsets.dtype == np.dtype("<i8")
        and heights.dtype == np.dtype("<i8")
        and widths.dtype == np.dtype("<i8"),
        "cache integer array dtype differs",
    )
    _require(losses.dtype == np.dtype("<f8"), "cache loss dtype differs")
    _require(
        offsets.shape == (count + 1,)
        and heights.shape == (count,)
        and widths.shape == (count,)
        and losses.shape == (count,),
        "cache array shapes differ",
    )
    _require(
        int(offsets[0]) == 0
        and np.all(offsets[1:] > offsets[:-1])
        and int(offsets[-1]) == probabilities.size
        and probabilities.size == targets.size,
        "cache offsets differ",
    )
    records: list[PredictionRecord] = []
    for index, image_id in enumerate(image_ids):
        height = int(heights[index])
        width = int(widths[index])
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        _require(
            height > 0 and width > 0 and stop - start == height * width,
            f"cache image[{index}] shape differs",
        )
        records.append(
            PredictionRecord(
                image_id=image_id,
                probability=probabilities[start:stop].reshape(
                    height,
                    width,
                ).copy(),
                target=targets[start:stop].reshape(height, width).copy(),
                loss=float(losses[index]),
            )
        )
    contract = metadata.get("evaluation_contract")
    _require(isinstance(contract, Mapping), "evaluation contract missing")
    _require(
        set(contract)
        == {
            "prediction_comparison",
            "match_radius",
            "tiny_area",
            "probability_dtype",
            "target_storage",
        },
        "evaluation contract fields differ",
    )
    _require(
        contract.get("prediction_comparison") == PREDICTION_COMPARISON,
        "prediction comparison differs",
    )
    _require(
        contract.get("probability_dtype") == "float32"
        and contract.get("target_storage")
        == "uint8_binary_target_gt_0.5",
        "cache array semantic contract differs",
    )
    cache = create_prediction_cache(
        records,
        identity=identity,
        match_radius=float(contract.get("match_radius")),
        tiny_area=contract.get("tiny_area"),
    )
    _require(
        cache.content_sha256 == metadata.get("prediction_content_sha256"),
        "prediction content SHA differs",
    )
    return cache


def image_sufficient_statistics(
    cache: PredictionCache,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("threshold must be finite and lie in [0, 1]")
    rows: list[dict[str, Any]] = []
    for record in cache.records:
        accumulator = ValidationMetrics(
            float(threshold),
            cache.match_radius,
            cache.tiny_area,
        )
        accumulator.update(
            record.probability,
            record.target,
            float(record.loss),
        )
        rows.append(
            {
                "schema": STATISTICS_SCHEMA,
                "image_id": record.image_id,
                "threshold": float(threshold),
                "loss": float(record.loss),
                "intersection": accumulator.intersection,
                "union": accumulator.union,
                "true_positive_pixels": accumulator.true_positive_pixels,
                "false_positive_pixels": accumulator.false_positive_pixels,
                "false_negative_pixels": accumulator.false_negative_pixels,
                "image_iou": accumulator.image_ious[0],
                "target_count": accumulator.target_count,
                "matched_target_count": accumulator.matched_target_count,
                "tiny_target_count": accumulator.tiny_target_count,
                "matched_tiny_target_count": (
                    accumulator.matched_tiny_target_count
                ),
                "predicted_object_count": accumulator.predicted_object_count,
                "unmatched_predicted_object_count": (
                    accumulator.unmatched_predicted_object_count
                ),
                "unmatched_predicted_pixels": (
                    accumulator.unmatched_predicted_pixels
                ),
                "valid_pixel_count": accumulator.valid_pixels,
            }
        )
    return rows


def aggregate_sufficient_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    _require(bool(rows), "statistics rows must not be empty")
    indices = (
        tuple(range(len(rows)))
        if sample_indices is None
        else tuple(sample_indices)
    )
    _require(bool(indices), "sample indices must not be empty")
    for index in indices:
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(rows)
        ):
            raise ValueError(f"invalid statistics sample index: {index!r}")
    selected = [rows[index] for index in indices]
    for row in selected:
        _require(
            row.get("schema") == STATISTICS_SCHEMA,
            "statistics row schema differs",
        )

    def total(name: str) -> int:
        return sum(int(row[name]) for row in selected)

    intersection = total("intersection")
    union = total("union")
    true_positive_pixels = total("true_positive_pixels")
    false_positive_pixels = total("false_positive_pixels")
    false_negative_pixels = total("false_negative_pixels")
    target_count = total("target_count")
    matched_target_count = total("matched_target_count")
    tiny_target_count = total("tiny_target_count")
    matched_tiny_target_count = total("matched_tiny_target_count")
    predicted_object_count = total("predicted_object_count")
    unmatched_object_count = total("unmatched_predicted_object_count")
    unmatched_pixels = total("unmatched_predicted_pixels")
    valid_pixels = total("valid_pixel_count")
    precision = true_positive_pixels / max(
        1,
        true_positive_pixels + false_positive_pixels,
    )
    recall = true_positive_pixels / max(
        1,
        true_positive_pixels + false_negative_pixels,
    )
    eps = np.finfo(np.float64).eps
    return {
        "val_loss": float(np.mean([float(row["loss"]) for row in selected])),
        "miou": intersection / max(1, union),
        "niou": float(
            np.mean([float(row["image_iou"]) for row in selected])
        ),
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": (
            2.0 * precision * recall / max(eps, precision + recall)
        ),
        "pd": matched_target_count / max(1, target_count),
        "tiny_pd": (
            matched_tiny_target_count / tiny_target_count
            if tiny_target_count
            else float("nan")
        ),
        "fa": unmatched_pixels / max(1, valid_pixels),
        "false_objects_per_image": unmatched_object_count / len(selected),
        "target_count": target_count,
        "matched_target_count": matched_target_count,
        "tiny_target_count": tiny_target_count,
        "matched_tiny_target_count": matched_tiny_target_count,
        "predicted_object_count": predicted_object_count,
        "unmatched_predicted_object_count": unmatched_object_count,
        "unmatched_predicted_pixels": unmatched_pixels,
        "valid_pixel_count": valid_pixels,
        "image_count": len(selected),
        "intersection": intersection,
        "union": union,
    }


def recompute_metrics(
    cache: PredictionCache,
    *,
    threshold: float,
    sample_indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    rows = image_sufficient_statistics(cache, threshold=threshold)
    return aggregate_sufficient_statistics(
        rows,
        sample_indices=sample_indices,
    )


__all__ = [
    "ARRAY_KEYS",
    "CACHE_SCHEMA",
    "DATA_SCOPE",
    "EXPECTED_VALIDATION_COUNT",
    "IDENTITY_SCHEMA",
    "MODE_SCHEMA",
    "PREDICTION_COMPARISON",
    "PredictionCache",
    "PredictionCacheCollector",
    "PredictionRecord",
    "STATISTICS_SCHEMA",
    "aggregate_sufficient_statistics",
    "build_cache_identity",
    "cache_paths",
    "canonical_json_bytes",
    "create_prediction_cache",
    "image_sufficient_statistics",
    "load_prediction_cache",
    "normalize_mode",
    "prediction_content_sha256",
    "recompute_metrics",
    "sha256_bytes",
    "sha256_file",
    "validate_cache_identity",
    "validation_identifier_sha256",
    "write_prediction_cache",
]
