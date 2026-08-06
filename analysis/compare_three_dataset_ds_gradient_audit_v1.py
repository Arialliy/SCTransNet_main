#!/usr/bin/env python3
"""Recompute the six-role DS-GA V1 gate from raw batch gradients.

This comparator is deliberately downstream of the train-only gradient audit.
It never loads a model, never reads ``img_idx/test``, and never changes a loss.
All PC/AC/PA predicates and the cross-dataset Trigger A are recomputed from the
four recorded batches in every available stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA = "sctransnet_three_dataset_ds_gradient_audit_comparison_v1/v1"
ANALYZER_SCHEMA = "sctransnet_three_dataset_ds_gradient_audit_v1/v1"
MANIFEST_SCHEMA = "sctransnet_three_dataset_ds_gradient_audit_manifest_v1/v1"
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLES = ("best_miou", "best_pd")
PRIMARY_ROLE = "best_miou"
CONFIRMATION_ROLE = "best_pd"
HEAD_ORDER = ("gt5", "gt4", "gt3", "gt2", "d0", "final")
AUXILIARY_HEADS = HEAD_ORDER[:-1]
REQUIRED_STRATA = ("tiny_positive", "normal_positive")
CONDITIONAL_STRATUM = "background_only"
STRATA = (*REQUIRED_STRATA, CONDITIONAL_STRATUM)
SHARED_GROUPS = (
    "encoder_shared",
    "tpd_qfg_sctb_shared",
    "ner_shared",
    "decoder_trunk_shared",
)
SEED = 42
BATCHES_PER_AVAILABLE_STRATUM = 4
SAMPLES_PER_AVAILABLE_STRATUM = 64
MIN_DISTINCT_SOURCE_IDS = 24
MIN_NATURAL_DIVERSITY_FLOOR = 16

PC_MAX_MEDIAN_COSINE = -0.10
PC_MIN_NEGATIVE_BATCHES = 3
PC_MIN_MEDIAN_NORM_RATIO = 0.25
AC_MAX_MEDIAN_COSINE = -0.10
AC_MIN_NEGATIVE_BATCHES = 3
AC_MIN_MEDIAN_NORM_RATIO = 1.50
PA_MIN_MEDIAN_COSINE = 0.20
PA_MIN_POSITIVE_BATCHES = 3
PA_MIN_MEDIAN_NORM_RATIO = 0.25
FINAL_NORM_MIN = 1e-12
FINAL_NORM_MAX_BAD_BATCHES = 1
SCALE_ANOMALY_RATIO = 1000.0
SCALE_ANOMALY_MIN_BATCHES = 2
FLOAT_REL_TOL = 1e-10
FLOAT_ABS_TOL = 1e-12

DECISION_AUTHORIZE = "DS_V2_DESIGN_AUTHORIZED_BY_PERSISTENT_GRADIENT_CONFLICT"
DECISION_ENGINEERING_INVALID = "DS_AUDIT_ENGINEERING_INVALID"
DECISION_DOMAIN_REVERSAL = "DS_GLOBAL_REWEIGHTING_BLOCKED_BY_DOMAIN_REVERSAL"
DECISION_SCALE_ANOMALY = "DS_GRADIENT_SCALE_ANOMALY_REQUIRES_DIAGNOSIS"
DECISION_NO_CONFLICT = "DS_NO_PERSISTENT_CROSS_DATASET_CONFLICT"

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "three_dataset_ds_gradient_audit_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison" / "seed42_six_role"


class DSGAComparisonError(ValueError):
    """The six-role DS gradient matrix differs from the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DSGAComparisonError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DSGAComparisonError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise DSGAComparisonError(f"{label} must be an array")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DSGAComparisonError(f"{label} must be numeric")
    ready = float(value)
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DSGAComparisonError(f"{label} must be an integer")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    ready = Path(path).resolve(strict=True)
    _require(ready.is_file() and not ready.is_symlink(), f"invalid JSON file: {ready}")
    raw = ready.read_bytes()
    value = json.loads(raw)
    _require(isinstance(value, dict), f"JSON root must be an object: {ready}")
    return value, hashlib.sha256(raw).hexdigest()


def _median(values: Sequence[float]) -> float:
    _require(bool(values), "cannot take median of an empty sequence")
    return float(statistics.median(values))


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _require_close(observed: Any, expected: float | None, label: str) -> None:
    if expected is None:
        _require(observed is None, f"{label} must be null")
        return
    ready = _finite(observed, label)
    _require(
        math.isclose(
            ready,
            expected,
            rel_tol=FLOAT_REL_TOL,
            abs_tol=FLOAT_ABS_TOL,
        ),
        f"{label} differs from gram_6x6",
    )


def _safe_cosine(dot: float, left_norm: float, right_norm: float) -> float | None:
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = dot / (left_norm * right_norm)
    _require(math.isfinite(value), "recomputed cosine is non-finite")
    _require(
        -1.0 - FLOAT_REL_TOL <= value <= 1.0 + FLOAT_REL_TOL,
        "gram_6x6 violates the cosine bound",
    )
    return max(-1.0, min(1.0, value))


def _validate_and_recompute_group(
    raw: Mapping[str, Any],
    *,
    expected_parameter_numel: int,
    label: str,
) -> dict[str, Any]:
    """Treat the serialized 6x6 Gram matrix as the sole numeric source."""

    _require(
        tuple(raw.get("gram_head_order", ())) == HEAD_ORDER,
        f"{label}.gram_head_order differs",
    )
    parameter_numel = _nonnegative_int(
        raw.get("parameter_numel"), f"{label}.parameter_numel"
    )
    _require(parameter_numel > 0, f"{label}.parameter_numel must be positive")
    _require(
        parameter_numel == expected_parameter_numel,
        f"{label}.parameter_numel differs from parameter_partition",
    )
    raw_gram = list(_sequence(raw.get("gram_6x6"), f"{label}.gram_6x6"))
    _require(len(raw_gram) == len(HEAD_ORDER), f"{label}.gram_6x6 row count differs")
    gram: list[list[float]] = []
    for row_index, raw_row in enumerate(raw_gram):
        row_values = list(_sequence(raw_row, f"{label}.gram_6x6[{row_index}]"))
        _require(
            len(row_values) == len(HEAD_ORDER),
            f"{label}.gram_6x6[{row_index}] column count differs",
        )
        gram.append(
            [
                _finite(value, f"{label}.gram_6x6[{row_index}][{column_index}]")
                for column_index, value in enumerate(row_values)
            ]
        )
    for left in range(len(HEAD_ORDER)):
        _require(gram[left][left] >= 0.0, f"{label}.gram diagonal is negative")
        for right in range(left + 1, len(HEAD_ORDER)):
            _require(
                math.isclose(
                    gram[left][right],
                    gram[right][left],
                    rel_tol=FLOAT_REL_TOL,
                    abs_tol=FLOAT_ABS_TOL,
                ),
                f"{label}.gram is not symmetric",
            )

    norms = [math.sqrt(value[index]) for index, value in enumerate(gram)]
    for left in range(len(HEAD_ORDER)):
        for right in range(left + 1, len(HEAD_ORDER)):
            bound = norms[left] * norms[right]
            _require(
                abs(gram[left][right])
                <= bound + FLOAT_ABS_TOL + FLOAT_REL_TOL * max(1.0, bound),
                f"{label}.gram violates Cauchy-Schwarz",
            )
    final_norm = norms[-1]
    serialized_heads = _mapping(raw.get("heads"), f"{label}.heads")
    _require(set(serialized_heads) == set(HEAD_ORDER), f"{label}.head scope differs")
    heads: dict[str, Any] = {}
    for index, head in enumerate(HEAD_ORDER):
        dot_final = gram[index][-1]
        expected = {
            "raw_l2_norm": norms[index],
            "gradient_rms": norms[index] / math.sqrt(parameter_numel),
            "norm_ratio_to_final": (
                None if final_norm == 0.0 else norms[index] / final_norm
            ),
            "cosine_to_final": _safe_cosine(dot_final, norms[index], final_norm),
            "dot_with_final": dot_final,
            "projection_onto_final": (
                None if final_norm == 0.0 else dot_final / (final_norm * final_norm)
            ),
        }
        serialized = _mapping(serialized_heads.get(head), f"{label}.heads.{head}")
        for metric, expected_value in expected.items():
            _require_close(
                serialized.get(metric), expected_value, f"{label}.heads.{head}.{metric}"
            )
        heads[head] = expected

    aux_square = sum(gram[left][right] for left in range(5) for right in range(5))
    aux_final_dot = sum(gram[index][5] for index in range(5))
    total_square = aux_square + gram[5][5] + 2.0 * aux_final_dot
    scale = max(1.0, sum(abs(value) for row in gram for value in row))
    _require(
        aux_square >= -(FLOAT_ABS_TOL + FLOAT_REL_TOL * scale),
        f"{label}.aux gram norm square is negative",
    )
    _require(
        total_square >= -(FLOAT_ABS_TOL + FLOAT_REL_TOL * scale),
        f"{label}.total gram norm square is negative",
    )
    aux_norm = math.sqrt(max(0.0, aux_square))
    total_norm = math.sqrt(max(0.0, total_square))
    individual_norm_sum = sum(norms)
    expected_aux = {
        "aux_l2_norm": aux_norm,
        "final_l2_norm": final_norm,
        "total_l2_norm": total_norm,
        "aux_final_dot": aux_final_dot,
        "cosine_aux_final": _safe_cosine(aux_final_dot, aux_norm, final_norm),
        "aux_to_final_norm_ratio": (
            None if final_norm == 0.0 else aux_norm / final_norm
        ),
        "cancellation": (
            None if individual_norm_sum == 0.0 else total_norm / individual_norm_sum
        ),
        "individual_head_norm_sum": individual_norm_sum,
    }
    serialized_aux = _mapping(raw.get("aux_total"), f"{label}.aux_total")
    for metric, expected_value in expected_aux.items():
        _require_close(
            serialized_aux.get(metric), expected_value, f"{label}.aux_total.{metric}"
        )
    return {
        "parameter_numel": parameter_numel,
        "heads": heads,
        "aux_total": expected_aux,
    }


def _scale_anomaly_batch_indices(
    ratios: Sequence[float | None],
    final_norms: Sequence[float],
) -> list[int]:
    _require(len(ratios) == len(final_norms), "scale-anomaly vectors differ")
    return [
        index
        for index, (value, final_norm) in enumerate(zip(ratios, final_norms))
        if final_norm >= FINAL_NORM_MIN
        and value is not None
        and value >= SCALE_ANOMALY_RATIO
    ]


def classify_head_relation(
    cosines: Sequence[float | None],
    norm_ratios: Sequence[float | None],
) -> dict[str, Any]:
    """Recompute PC and PA for one head/group/stratum unit."""

    _require(
        len(cosines) == len(norm_ratios) == BATCHES_PER_AVAILABLE_STRATUM,
        "head relation requires exactly four batches",
    )
    valid_cosines = [float(value) for value in cosines if value is not None]
    for value in valid_cosines:
        _require(math.isfinite(value) and -1.000001 <= value <= 1.000001, "invalid cosine")
    ready_ratios = [
        _optional_finite(value, "norm_ratio_to_final") for value in norm_ratios
    ]
    valid_ratios = [value for value in ready_ratios if value is not None]
    _require(all(value >= 0.0 for value in valid_ratios), "negative norm ratio")
    median_cosine = _median(valid_cosines) if valid_cosines else None
    median_ratio = _median(valid_ratios) if valid_ratios else None
    negative_count = sum(value < 0.0 for value in valid_cosines)
    positive_count = sum(value > 0.0 for value in valid_cosines)
    persistent_conflict = bool(
        len(valid_cosines) == BATCHES_PER_AVAILABLE_STRATUM
        and median_cosine is not None
        and median_cosine <= PC_MAX_MEDIAN_COSINE
        and negative_count >= PC_MIN_NEGATIVE_BATCHES
        and len(valid_ratios) == BATCHES_PER_AVAILABLE_STRATUM
        and median_ratio is not None
        and median_ratio >= PC_MIN_MEDIAN_NORM_RATIO
    )
    persistent_alignment = bool(
        len(valid_cosines) == BATCHES_PER_AVAILABLE_STRATUM
        and median_cosine is not None
        and median_cosine >= PA_MIN_MEDIAN_COSINE
        and positive_count >= PA_MIN_POSITIVE_BATCHES
        and len(valid_ratios) == BATCHES_PER_AVAILABLE_STRATUM
        and median_ratio is not None
        and median_ratio >= PA_MIN_MEDIAN_NORM_RATIO
    )
    return {
        "cosine_values": [None if value is None else float(value) for value in cosines],
        "norm_ratio_values": ready_ratios,
        "valid_cosine_count": len(valid_cosines),
        "valid_norm_ratio_count": len(valid_ratios),
        "median_cosine": median_cosine,
        "negative_cosine_batch_count": negative_count,
        "positive_cosine_batch_count": positive_count,
        "median_norm_ratio_to_final": median_ratio,
        "persistent_conflict": persistent_conflict,
        "persistent_alignment": persistent_alignment,
    }


def classify_auxiliary_relation(
    cosines: Sequence[float | None],
    norm_ratios: Sequence[float | None],
) -> dict[str, Any]:
    """Recompute AC for the sum of the five auxiliary gradients."""

    _require(
        len(cosines) == len(norm_ratios) == BATCHES_PER_AVAILABLE_STRATUM,
        "auxiliary relation requires exactly four batches",
    )
    valid_cosines = [float(value) for value in cosines if value is not None]
    for value in valid_cosines:
        _require(math.isfinite(value) and -1.000001 <= value <= 1.000001, "invalid aux cosine")
    ready_ratios = [
        _optional_finite(value, "aux_to_final_norm_ratio") for value in norm_ratios
    ]
    valid_ratios = [value for value in ready_ratios if value is not None]
    _require(all(value >= 0.0 for value in valid_ratios), "negative aux norm ratio")
    median_cosine = _median(valid_cosines) if valid_cosines else None
    median_ratio = _median(valid_ratios) if valid_ratios else None
    negative_count = sum(value < 0.0 for value in valid_cosines)
    aggregate_conflict = bool(
        len(valid_cosines) == BATCHES_PER_AVAILABLE_STRATUM
        and median_cosine is not None
        and median_cosine <= AC_MAX_MEDIAN_COSINE
        and negative_count >= AC_MIN_NEGATIVE_BATCHES
        and len(valid_ratios) == BATCHES_PER_AVAILABLE_STRATUM
        and median_ratio is not None
        and median_ratio >= AC_MIN_MEDIAN_NORM_RATIO
    )
    return {
        "cosine_values": [None if value is None else float(value) for value in cosines],
        "norm_ratio_values": ready_ratios,
        "valid_cosine_count": len(valid_cosines),
        "valid_norm_ratio_count": len(valid_ratios),
        "median_cosine": median_cosine,
        "negative_cosine_batch_count": negative_count,
        "median_aux_to_final_norm_ratio": median_ratio,
        "aggregate_conflict": aggregate_conflict,
    }


def _validate_checkpoint_binding(
    raw: Any,
    *,
    expected_role: str,
) -> tuple[str, str]:
    binding = _mapping(raw, "checkpoint_binding")
    checkpoint = _mapping(binding.get("checkpoint"), "checkpoint_binding.checkpoint")
    _require(
        checkpoint.get("role") == expected_role,
        "nested checkpoint role binding differs",
    )
    checkpoint_sha = checkpoint.get("sha256")
    _require(_is_sha256(checkpoint_sha), "nested checkpoint SHA missing")
    state_sha = binding.get("training_state_dict_sha256")
    _require(_is_sha256(state_sha), "training state-dict SHA differs")
    return str(checkpoint_sha), str(state_sha)


def _validate_partition_group(
    raw: Any,
    *,
    label: str,
) -> dict[str, Any]:
    group = _mapping(raw, label)
    names_raw = list(
        _sequence(group.get("ordered_parameter_names"), f"{label}.ordered_parameter_names")
    )
    _require(
        names_raw and all(isinstance(name, str) and bool(name) for name in names_raw),
        f"{label}.ordered_parameter_names differs",
    )
    names = [str(name) for name in names_raw]
    _require(len(names) == len(set(names)), f"{label} repeats a parameter name")
    tensor_count = _nonnegative_int(
        group.get("parameter_tensor_count"), f"{label}.parameter_tensor_count"
    )
    numel = _nonnegative_int(group.get("parameter_numel"), f"{label}.parameter_numel")
    _require(tensor_count == len(names), f"{label}.parameter_tensor_count differs")
    _require(numel > 0, f"{label}.parameter_numel must be positive")
    names_sha = group.get("ordered_parameter_names_sha256")
    _require(_is_sha256(names_sha), f"{label}.ordered parameter-name SHA missing")
    _require(
        names_sha == canonical_sha256(names),
        f"{label}.ordered parameter-name SHA is not reproducible",
    )
    return {
        "parameter_tensor_count": tensor_count,
        "parameter_numel": numel,
        "ordered_parameter_names": names,
        "ordered_parameter_names_sha256": names_sha,
    }


def _validate_parameter_partition(raw: Any) -> dict[str, Any]:
    partition = _mapping(raw, "parameter_partition")
    _require(
        partition.get("all_trainable_parameters_assigned_once") is True,
        "atomic parameter partition is not exhaustive",
    )
    _require(
        partition.get("shared_groups_mutually_exclusive") is True,
        "shared parameter groups overlap",
    )
    collections: dict[str, dict[str, dict[str, Any]]] = {}
    expected_scopes: dict[str, set[str] | None] = {
        "atomic_groups": None,
        "shared_groups": set(SHARED_GROUPS),
        "head_local_groups": set(HEAD_ORDER),
    }
    for collection_name, expected_scope in expected_scopes.items():
        serialized = _mapping(
            partition.get(collection_name), f"parameter_partition.{collection_name}"
        )
        _require(bool(serialized), f"parameter_partition.{collection_name} is empty")
        if expected_scope is not None:
            _require(
                set(serialized) == expected_scope,
                f"parameter_partition.{collection_name} scope differs",
            )
        collections[collection_name] = {
            name: _validate_partition_group(
                group,
                label=f"parameter_partition.{collection_name}.{name}",
            )
            for name, group in serialized.items()
        }

    atomic_names = [
        name
        for group in collections["atomic_groups"].values()
        for name in group["ordered_parameter_names"]
    ]
    _require(
        len(atomic_names) == len(set(atomic_names)),
        "atomic groups repeat a parameter name",
    )
    trainable_tensor_count = _nonnegative_int(
        partition.get("trainable_parameter_tensor_count"),
        "parameter_partition.trainable_parameter_tensor_count",
    )
    trainable_numel = _nonnegative_int(
        partition.get("trainable_parameter_numel"),
        "parameter_partition.trainable_parameter_numel",
    )
    _require(
        trainable_tensor_count
        == sum(
            group["parameter_tensor_count"]
            for group in collections["atomic_groups"].values()
        ),
        "trainable parameter tensor count differs from atomic partition",
    )
    _require(
        trainable_numel
        == sum(group["parameter_numel"] for group in collections["atomic_groups"].values()),
        "trainable parameter numel differs from atomic partition",
    )
    shared_names = [
        name
        for group in collections["shared_groups"].values()
        for name in group["ordered_parameter_names"]
    ]
    _require(
        len(shared_names) == len(set(shared_names)),
        "shared groups repeat a parameter name",
    )
    _require(
        set(shared_names).issubset(set(atomic_names)),
        "shared groups contain names outside the atomic partition",
    )
    local_names = [
        name
        for group in collections["head_local_groups"].values()
        for name in group["ordered_parameter_names"]
    ]
    _require(
        len(local_names) == len(set(local_names)),
        "head-local groups repeat a parameter name",
    )
    _require(
        set(local_names).issubset(set(atomic_names)),
        "head-local groups contain names outside the atomic partition",
    )
    _require(
        set(local_names).isdisjoint(set(shared_names)),
        "head-local and shared groups overlap",
    )
    contract = {
        "trainable_parameter_tensor_count": trainable_tensor_count,
        "trainable_parameter_numel": trainable_numel,
        "collections": collections,
    }
    return {
        "contract": contract,
        "sha256": canonical_sha256(contract),
        "shared_group_numel": {
            name: group["parameter_numel"]
            for name, group in collections["shared_groups"].items()
        },
    }


def _batch_group(batch: Mapping[str, Any], group: str, label: str) -> Mapping[str, Any]:
    shared = _mapping(batch.get("shared_groups"), f"{label}.shared_groups")
    return _mapping(shared.get(group), f"{label}.shared_groups.{group}")


def _extract_available_stratum(
    raw: Mapping[str, Any],
    *,
    dataset: str,
    role: str,
    stratum: str,
    shared_group_numel: Mapping[str, int],
) -> dict[str, Any]:
    label = f"{dataset}.{role}.{stratum}"
    _require(raw.get("available") is True, f"{label} must be available")
    batches = list(_sequence(raw.get("batches"), f"{label}.batches"))
    _require(len(batches) == BATCHES_PER_AVAILABLE_STRATUM, f"{label} batch count differs")
    sample_count = _nonnegative_int(raw.get("sample_count"), f"{label}.sample_count")
    distinct = _nonnegative_int(
        raw.get("distinct_source_count"), f"{label}.distinct_source_count"
    )
    _require(sample_count == SAMPLES_PER_AVAILABLE_STRATUM, f"{label} sample count differs")
    diversity_target = _nonnegative_int(
        raw.get("diversity_target", MIN_DISTINCT_SOURCE_IDS),
        f"{label}.diversity_target",
    )
    max_repeats = _nonnegative_int(
        raw.get("max_repeat_cap", raw.get("max_repeats_per_source", 3)),
        f"{label}.max_repeat_cap",
    )
    diversity_limited = raw.get(
        "diversity_target_limited_by_natural_availability", False
    )
    _require(isinstance(diversity_limited, bool), f"{label} diversity flag differs")
    if diversity_limited:
        natural_ceiling = _nonnegative_int(
            raw.get("natural_distinct_source_ceiling"),
            f"{label}.natural_distinct_source_ceiling",
        )
        _require(
            MIN_NATURAL_DIVERSITY_FLOOR <= natural_ceiling < MIN_DISTINCT_SOURCE_IDS,
            f"{label} natural diversity exception is outside the frozen range",
        )
        _require(diversity_target == natural_ceiling, f"{label} diversity target differs")
        _require(distinct == natural_ceiling, f"{label} did not cover every natural source")
        _require(
            max_repeats == math.ceil(SAMPLES_PER_AVAILABLE_STRATUM / natural_ceiling),
            f"{label} repeat cap is not the minimal feasible cap",
        )
        proof_sha = raw.get(
            "exhaustive_natural_availability_proof_sha256",
            raw.get("natural_diversity_exhaustive_proof_sha256"),
        )
        _require(_is_sha256(proof_sha), f"{label} exhaustive proof SHA missing")
    else:
        natural_ceiling = None
        proof_sha = None
        _require(diversity_target == MIN_DISTINCT_SOURCE_IDS, f"{label} diversity target differs")
        _require(distinct >= MIN_DISTINCT_SOURCE_IDS, f"{label} source coverage differs")
        _require(max_repeats == 3, f"{label} repeat cap differs")

    batch_contracts: list[dict[str, Any]] = []
    recomputed_batches: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(batches):
        batch = _mapping(raw_batch, f"{label}.batches[{batch_index}]")
        _require(
            batch.get("batch_index") == batch_index,
            f"{label}.batches[{batch_index}].batch_index differs",
        )
        sample_ids = list(
            _sequence(batch.get("sample_ids"), f"{label}.batches[{batch_index}].sample_ids")
        )
        _require(
            len(sample_ids) == 16
            and all(isinstance(sample_id, str) and bool(sample_id) for sample_id in sample_ids),
            f"{label}.batches[{batch_index}] must contain 16 sample IDs",
        )
        images_sha = batch.get("images_sha256")
        masks_sha = batch.get("masks_sha256")
        _require(_is_sha256(images_sha), f"{label}.batches[{batch_index}] image SHA differs")
        _require(_is_sha256(masks_sha), f"{label}.batches[{batch_index}] mask SHA differs")
        forward_seed = _nonnegative_int(
            batch.get("forward_seed"), f"{label}.batches[{batch_index}].forward_seed"
        )
        batch_contracts.append(
            {
                "batch_index": batch_index,
                "sample_ids": [str(value) for value in sample_ids],
                "images_sha256": images_sha,
                "masks_sha256": masks_sha,
                "forward_seed": forward_seed,
            }
        )
        recomputed_groups: dict[str, Any] = {}
        for group in SHARED_GROUPS:
            group_raw = _batch_group(batch, group, f"{label}.batches[{batch_index}]")
            recomputed_groups[group] = _validate_and_recompute_group(
                group_raw,
                expected_parameter_numel=int(shared_group_numel[group]),
                label=f"{label}.batches[{batch_index}].shared_groups.{group}",
            )
        recomputed_batches.append(recomputed_groups)

    group_results: dict[str, Any] = {}
    final_norm_failures: list[int] = []
    scale_anomalies: list[dict[str, Any]] = []
    for group in SHARED_GROUPS:
        head_relations: dict[str, Any] = {}
        for head in AUXILIARY_HEADS:
            cosines: list[float | None] = []
            ratios: list[float | None] = []
            final_norms_for_head: list[float] = []
            for recomputed in recomputed_batches:
                row = recomputed[group]["heads"][head]
                cosines.append(row["cosine_to_final"])
                ratios.append(row["norm_ratio_to_final"])
                final_norms_for_head.append(
                    recomputed[group]["aux_total"]["final_l2_norm"]
                )
            relation = classify_head_relation(cosines, ratios)
            anomaly_batches = _scale_anomaly_batch_indices(
                ratios, final_norms_for_head
            )
            relation["scale_anomaly_batch_indices"] = anomaly_batches
            relation["scale_anomaly"] = len(anomaly_batches) >= SCALE_ANOMALY_MIN_BATCHES
            if relation["scale_anomaly"]:
                scale_anomalies.append(
                    {"group": group, "head": head, "batch_indices": anomaly_batches}
                )
            head_relations[head] = relation

        aux_cosines: list[float | None] = []
        aux_ratios: list[float | None] = []
        final_norms: list[float] = []
        for recomputed in recomputed_batches:
            aggregate = recomputed[group]["aux_total"]
            aux_cosines.append(aggregate["cosine_aux_final"])
            aux_ratios.append(aggregate["aux_to_final_norm_ratio"])
            final_norms.append(aggregate["final_l2_norm"])
        final_norm_failure_indices = [
            index for index, value in enumerate(final_norms) if value < FINAL_NORM_MIN
        ]
        if len(final_norm_failure_indices) > FINAL_NORM_MAX_BAD_BATCHES:
            final_norm_failures.extend(final_norm_failure_indices)
        group_results[group] = {
            "heads": head_relations,
            "auxiliary": classify_auxiliary_relation(aux_cosines, aux_ratios),
            "final_norm_values": final_norms,
            "final_norm_failure_batch_indices": final_norm_failure_indices,
        }
    return {
        "required_or_conditional": raw.get("required_or_conditional"),
        "available": True,
        "sample_count": sample_count,
        "distinct_source_count": distinct,
        "diversity_target": diversity_target,
        "max_repeat_cap": max_repeats,
        "diversity_target_limited_by_natural_availability": diversity_limited,
        "natural_distinct_source_ceiling": natural_ceiling,
        "exhaustive_natural_availability_proof_sha256": proof_sha,
        "batch_count": len(batches),
        "batch_contracts": batch_contracts,
        "batch_contract_sha256": canonical_sha256(batch_contracts),
        "groups": group_results,
        "final_norm_gate_failed": bool(final_norm_failures),
        "scale_anomalies": scale_anomalies,
    }


def _validate_unavailable_background(raw: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require(raw.get("available") is False, f"{label} availability differs")
    _require(raw.get("structurally_unavailable") is True, f"{label} structural flag missing")
    _require(raw.get("candidate_count") == 0, f"{label} candidate_count must be zero")
    reason = raw.get("reason")
    _require(isinstance(reason, str) and bool(reason.strip()), f"{label} reason missing")
    batches = raw.get("batches", [])
    _require(isinstance(batches, (tuple, list)) and len(batches) == 0, f"{label} has partial batches")
    return {
        "required_or_conditional": "conditional",
        "available": False,
        "structurally_unavailable": True,
        "candidate_count": 0,
        "observed_natural_candidate_count": _nonnegative_int(
            raw.get("observed_natural_candidate_count", 0),
            f"{label}.observed_natural_candidate_count",
        ),
        "reason": reason,
        "batch_contracts": [],
        "batch_contract_sha256": canonical_sha256([]),
    }


def validate_analyzer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one analyzer output and recompute all unit predicates."""

    _require(payload.get("schema") == ANALYZER_SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer status differs")
    dataset = payload.get("dataset")
    role = payload.get("checkpoint_role")
    _require(dataset in DATASETS, "analyzer dataset differs")
    _require(role in CHECKPOINT_ROLES, "analyzer checkpoint role differs")
    _require(payload.get("seed") == SEED, "analyzer seed differs")
    _require(tuple(payload.get("head_order", ())) == HEAD_ORDER, "head order differs")

    checkpoint_sha, state_dict_sha = _validate_checkpoint_binding(
        payload.get("checkpoint_binding"), expected_role=str(role)
    )
    manifest = _mapping(payload.get("manifest_binding"), "manifest_binding")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema differs")
    manifest_sha = manifest.get("sha256")
    _require(_is_sha256(manifest_sha), "manifest SHA missing")
    partition = _validate_parameter_partition(payload.get("parameter_partition"))

    source_sha = _mapping(payload.get("source_sha256"), "source_sha256")
    _require(bool(source_sha), "source SHA map is empty")
    _require(all(_is_sha256(value) for value in source_sha.values()), "source SHA differs")
    sentinel = _mapping(payload.get("sentinel_replay"), "sentinel_replay")
    _require(sentinel.get("repeat_count") == 2, "sentinel repeat count differs")
    _require(sentinel.get("replay_exact") is True, "sentinel replay differs")
    _require(_is_sha256(sentinel.get("first_summary_sha256")), "sentinel first SHA differs")
    _require(_is_sha256(sentinel.get("second_summary_sha256")), "sentinel second SHA differs")
    _require(
        sentinel.get("first_summary_sha256") == sentinel.get("second_summary_sha256"),
        "sentinel summaries differ",
    )
    restoration = _mapping(payload.get("restoration_audit"), "restoration_audit")
    _require(restoration.get("all_batches_restored") is True, "model state was not restored")
    _require(restoration.get("all_parameter_grads_none") is True, "leaf gradients were modified")
    _require(restoration.get("rng_restored") is True, "RNG state was not restored")

    serialized_strata = _mapping(payload.get("strata"), "strata")
    _require(set(serialized_strata) == set(STRATA), "stratum set differs")
    strata: dict[str, Any] = {}
    engineering_reasons: list[str] = []
    for stratum in STRATA:
        raw = _mapping(serialized_strata[stratum], f"strata.{stratum}")
        if stratum in REQUIRED_STRATA:
            _require(raw.get("required_or_conditional") == "required", f"{stratum} role differs")
            _require(raw.get("available") is True, f"required stratum unavailable: {stratum}")
            ready = _extract_available_stratum(
                raw,
                dataset=str(dataset),
                role=str(role),
                stratum=stratum,
                shared_group_numel=partition["shared_group_numel"],
            )
        elif raw.get("available") is True:
            _require(raw.get("required_or_conditional") == "conditional", "background role differs")
            ready = _extract_available_stratum(
                raw,
                dataset=str(dataset),
                role=str(role),
                stratum=stratum,
                shared_group_numel=partition["shared_group_numel"],
            )
        else:
            ready = _validate_unavailable_background(raw, f"{dataset}.{role}.{stratum}")
        if ready.get("final_norm_gate_failed") is True:
            engineering_reasons.append(f"{stratum}:final_gradient_norm_gate")
        strata[stratum] = ready
    return {
        "dataset": dataset,
        "checkpoint_role": role,
        "checkpoint_sha256": checkpoint_sha,
        "training_state_dict_sha256": state_dict_sha,
        "manifest_sha256": manifest_sha,
        "parameter_partition_sha256": partition["sha256"],
        "parameter_partition_contract": partition["contract"],
        "strata": strata,
        "engineering_valid": not engineering_reasons,
        "engineering_failure_reasons": engineering_reasons,
        "source_sha256": dict(source_sha),
    }


def required_dataset_count(available_dataset_count: int) -> int:
    _require(available_dataset_count >= 0, "available dataset count is negative")
    return max(2, math.ceil(2 * available_dataset_count / 3))


def decide_from_units(units: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Apply Trigger A to validated dataset/role units."""

    engineering_failures: list[dict[str, Any]] = []
    manifest_shas: set[str] = set()
    parameter_partition_shas: set[str] = set()
    checkpoint_bindings: dict[str, dict[str, str]] = {}
    state_dict_bindings: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        _require(dataset in units, f"missing dataset unit: {dataset}")
        checkpoint_bindings[dataset] = {}
        state_dict_bindings[dataset] = {}
        for role in CHECKPOINT_ROLES:
            _require(role in units[dataset], f"missing role unit: {dataset}/{role}")
            unit = units[dataset][role]
            manifest_shas.add(str(unit["manifest_sha256"]))
            partition_sha = unit.get("parameter_partition_sha256")
            _require(_is_sha256(partition_sha), "parameter partition SHA missing")
            parameter_partition_shas.add(str(partition_sha))
            checkpoint_bindings[dataset][role] = str(unit["checkpoint_sha256"])
            state_dict_sha = unit.get("training_state_dict_sha256")
            _require(_is_sha256(state_dict_sha), "training state-dict SHA missing")
            state_dict_bindings[dataset][role] = str(state_dict_sha)
            if not unit.get("engineering_valid", False):
                engineering_failures.append(
                    {
                        "dataset": dataset,
                        "checkpoint_role": role,
                        "reasons": list(unit.get("engineering_failure_reasons", ())),
                    }
                )
    _require(len(manifest_shas) == 1, "six roles do not share one batch manifest")
    if len(parameter_partition_shas) != 1:
        engineering_failures.append(
            {
                "reason": "six_roles_do_not_share_one_parameter_partition",
                "parameter_partition_sha256": sorted(parameter_partition_shas),
            }
        )

    availability: dict[str, list[str]] = {}
    availability_mismatches: list[dict[str, Any]] = []
    for stratum in STRATA:
        ready: list[str] = []
        for dataset in DATASETS:
            flags = [
                bool(units[dataset][role]["strata"][stratum]["available"])
                for role in CHECKPOINT_ROLES
            ]
            if flags[0] != flags[1]:
                availability_mismatches.append(
                    {"dataset": dataset, "stratum": stratum, "role_flags": flags}
                )
            if all(flags):
                ready.append(dataset)
            role_batch_shas = [
                units[dataset][role]["strata"][stratum].get(
                    "batch_contract_sha256"
                )
                for role in CHECKPOINT_ROLES
            ]
            _require(
                all(_is_sha256(value) for value in role_batch_shas),
                "batch contract SHA missing",
            )
            if len(set(role_batch_shas)) != 1:
                engineering_failures.append(
                    {
                        "reason": "checkpoint_roles_do_not_reuse_identical_batches",
                        "dataset": dataset,
                        "stratum": stratum,
                        "role_batch_contract_sha256": dict(
                            zip(CHECKPOINT_ROLES, role_batch_shas)
                        ),
                    }
                )
        availability[stratum] = ready
    if availability_mismatches:
        engineering_failures.append(
            {"reason": "checkpoint_roles_do_not_share_stratum_availability", "details": availability_mismatches}
        )
    for stratum in REQUIRED_STRATA:
        if set(availability[stratum]) != set(DATASETS):
            engineering_failures.append(
                {"reason": "required_stratum_dataset_coverage_differs", "stratum": stratum}
            )

    signatures: list[dict[str, Any]] = []
    any_scale_anomaly = False
    any_domain_reversal = False
    authorized_signatures: list[dict[str, Any]] = []
    for stratum in STRATA:
        available_datasets = availability[stratum]
        k_s = required_dataset_count(len(available_datasets))
        for group in SHARED_GROUPS:
            for head in AUXILIARY_HEADS:
                unit_rows: dict[str, Any] = {}
                paired_pass_datasets: list[str] = []
                primary_pc: list[str] = []
                primary_pa: list[str] = []
                any_role_pc: list[str] = []
                any_role_pa: list[str] = []
                for dataset in available_datasets:
                    role_rows: dict[str, Any] = {}
                    for role in CHECKPOINT_ROLES:
                        group_row = units[dataset][role]["strata"][stratum]["groups"][group]
                        head_row = group_row["heads"][head]
                        aux_row = group_row["auxiliary"]
                        role_rows[role] = {
                            "persistent_conflict": bool(head_row["persistent_conflict"]),
                            "persistent_alignment": bool(head_row["persistent_alignment"]),
                            "aggregate_conflict": bool(aux_row["aggregate_conflict"]),
                            "scale_anomaly": bool(head_row["scale_anomaly"]),
                            "median_cosine": head_row["median_cosine"],
                            "median_norm_ratio_to_final": head_row[
                                "median_norm_ratio_to_final"
                            ],
                        }
                        any_scale_anomaly = any_scale_anomaly or bool(head_row["scale_anomaly"])
                    if role_rows[PRIMARY_ROLE]["persistent_conflict"]:
                        primary_pc.append(dataset)
                    if role_rows[PRIMARY_ROLE]["persistent_alignment"]:
                        primary_pa.append(dataset)
                    if any(
                        role_rows[role]["persistent_conflict"]
                        for role in CHECKPOINT_ROLES
                    ):
                        any_role_pc.append(dataset)
                    if any(
                        role_rows[role]["persistent_alignment"]
                        for role in CHECKPOINT_ROLES
                    ):
                        any_role_pa.append(dataset)
                    if all(
                        role_rows[role]["persistent_conflict"]
                        and role_rows[role]["aggregate_conflict"]
                        for role in CHECKPOINT_ROLES
                    ):
                        paired_pass_datasets.append(dataset)
                    unit_rows[dataset] = role_rows
                reversal = any(
                    conflict_dataset != aligned_dataset
                    for conflict_dataset in any_role_pc
                    for aligned_dataset in any_role_pa
                )
                any_domain_reversal = any_domain_reversal or reversal
                authorized = bool(
                    len(available_datasets) >= 2
                    and len(paired_pass_datasets) >= k_s
                    and not reversal
                )
                row = {
                    "head": head,
                    "group": group,
                    "stratum": stratum,
                    "available_datasets": list(available_datasets),
                    "available_dataset_count": len(available_datasets),
                    "required_dataset_count_k_s": k_s,
                    "paired_pc_ac_pass_datasets": paired_pass_datasets,
                    "paired_pc_ac_pass_count": len(paired_pass_datasets),
                    "primary_pc_datasets": primary_pc,
                    "primary_pa_datasets": primary_pa,
                    "any_role_pc_datasets": any_role_pc,
                    "any_role_pa_datasets": any_role_pa,
                    "domain_direction_reversal": reversal,
                    "authorized_signature": authorized,
                    "units": unit_rows,
                }
                signatures.append(row)
                if authorized:
                    authorized_signatures.append(row)

    if engineering_failures:
        decision = DECISION_ENGINEERING_INVALID
    elif any_scale_anomaly:
        decision = DECISION_SCALE_ANOMALY
    elif authorized_signatures:
        decision = DECISION_AUTHORIZE
    elif any_domain_reversal:
        decision = DECISION_DOMAIN_REVERSAL
    else:
        decision = DECISION_NO_CONFLICT
    return {
        "decision": decision,
        "engineering_valid": not engineering_failures,
        "engineering_failures": engineering_failures,
        "stratum_availability": availability,
        "signatures": signatures,
        "authorized_signatures": authorized_signatures,
        "trigger_a_passed": decision == DECISION_AUTHORIZE,
        "ds_v2_design_authorized": decision == DECISION_AUTHORIZE,
        "ds_v2_training_authorized": False,
        "loss_formula_changed": False,
        "tiny_gradient_conflict_supported": bool(
            decision == DECISION_AUTHORIZE
            and any(
                row["stratum"] == "tiny_positive"
                for row in authorized_signatures
            )
        ),
        "gradient_scale_anomaly_observed": any_scale_anomaly,
        "domain_direction_reversal_observed": any_domain_reversal,
        "manifest_sha256": next(iter(manifest_shas)),
        "parameter_partition_sha256": next(iter(parameter_partition_shas)),
        "checkpoint_sha256": checkpoint_bindings,
        "training_state_dict_sha256": state_dict_bindings,
    }


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    input_bindings: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> dict[str, Any]:
    units: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        units[dataset] = {}
        for role in CHECKPOINT_ROLES:
            units[dataset][role] = validate_analyzer_payload(payloads[dataset][role])
    decision = decide_from_units(units)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "seed": SEED,
        "datasets": list(DATASETS),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "primary_role": PRIMARY_ROLE,
        "confirmation_role": CONFIRMATION_ROLE,
        "head_order": list(HEAD_ORDER),
        "auxiliary_heads": list(AUXILIARY_HEADS),
        "shared_groups": list(SHARED_GROUPS),
        "thresholds": {
            "pc_max_median_cosine": PC_MAX_MEDIAN_COSINE,
            "pc_min_negative_batches": PC_MIN_NEGATIVE_BATCHES,
            "pc_min_median_norm_ratio": PC_MIN_MEDIAN_NORM_RATIO,
            "ac_max_median_cosine": AC_MAX_MEDIAN_COSINE,
            "ac_min_negative_batches": AC_MIN_NEGATIVE_BATCHES,
            "ac_min_median_norm_ratio": AC_MIN_MEDIAN_NORM_RATIO,
            "pa_min_median_cosine": PA_MIN_MEDIAN_COSINE,
            "pa_min_positive_batches": PA_MIN_POSITIVE_BATCHES,
            "pa_min_median_norm_ratio": PA_MIN_MEDIAN_NORM_RATIO,
            "scale_anomaly_ratio": SCALE_ANOMALY_RATIO,
            "scale_anomaly_min_batches": SCALE_ANOMALY_MIN_BATCHES,
        },
        "claim_scope": "seed42_img_idx_train_gradient_diagnostic_only",
        "test_split_used_for_head_or_signature_selection": False,
        "performance_claim_established": False,
        "decision": decision,
        "input_bindings": input_bindings or {},
        "source_sha256": {
            "analysis/compare_three_dataset_ds_gradient_audit_v1.py": file_sha256(
                Path(__file__).resolve()
            )
        },
    }


def validate_comparison_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "comparison schema differs")
    _require(payload.get("status") == "complete", "comparison status differs")
    _require(payload.get("seed") == SEED, "comparison seed differs")
    _require(payload.get("datasets") == list(DATASETS), "comparison datasets differ")
    _require(
        payload.get("checkpoint_roles") == list(CHECKPOINT_ROLES),
        "comparison roles differ",
    )
    decision = _mapping(payload.get("decision"), "decision")
    value = decision.get("decision")
    _require(
        value
        in {
            DECISION_AUTHORIZE,
            DECISION_ENGINEERING_INVALID,
            DECISION_DOMAIN_REVERSAL,
            DECISION_SCALE_ANOMALY,
            DECISION_NO_CONFLICT,
        },
        "comparison decision differs",
    )
    _require(decision.get("ds_v2_training_authorized") is False, "training was authorized")
    _require(decision.get("loss_formula_changed") is False, "loss formula changed")
    _require(
        decision.get("ds_v2_design_authorized") is (value == DECISION_AUTHORIZE),
        "design authorization differs",
    )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _decision_markdown(payload: Mapping[str, Any]) -> str:
    decision = _mapping(payload["decision"], "decision")
    lines = [
        "# DS-GA V1 六角色裁决",
        "",
        f"- decision: `{decision['decision']}`",
        f"- DS V2 设计授权: `{str(decision['ds_v2_design_authorized']).lower()}`",
        "- DS V2 训练授权: `false`",
        "- 结论范围: seed42、三数据集各自 img_idx/train 的梯度诊断",
        "",
        "## Stratum availability",
        "",
        "| Stratum | 可用数据集 |",
        "|---|---|",
    ]
    for stratum in STRATA:
        datasets = decision["stratum_availability"][stratum]
        lines.append(f"| {stratum} | {', '.join(datasets) if datasets else 'none'} |")
    lines.extend(
        [
            "",
            "## Trigger A",
            "",
            f"- 通过: `{str(decision['trigger_a_passed']).lower()}`",
            f"- 授权签名数: `{len(decision['authorized_signatures'])}`",
            f"- 梯度尺度异常: `{str(decision['gradient_scale_anomaly_observed']).lower()}`",
            f"- 跨数据集方向反转: `{str(decision['domain_direction_reversal_observed']).lower()}`",
            "",
            "该裁决只决定是否允许设计 DS V2；不代表性能提升，也不启动训练。",
            "",
        ]
    )
    return "\n".join(lines)


def _default_audit_path(root: Path, dataset: str, role: str) -> Path:
    return Path(root) / "runs" / dataset / role / "audit.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    bindings: dict[str, dict[str, dict[str, str]]] = {}
    for dataset in DATASETS:
        payloads[dataset] = {}
        bindings[dataset] = {}
        for role in CHECKPOINT_ROLES:
            path = _default_audit_path(args.input_root, dataset, role)
            payload, sha = _load_json(path)
            payloads[dataset][role] = payload
            bindings[dataset][role] = {
                "path": str(path.resolve()),
                "sha256": sha,
            }
    comparison = compare_payloads(payloads, input_bindings=bindings)
    validate_comparison_payload(comparison)
    output_dir = args.output_dir.resolve()
    aggregate_path = output_dir / "aggregate.json"
    decision_path = output_dir / "decision.json"
    markdown_path = output_dir / "decision.md"
    for path in (aggregate_path, decision_path, markdown_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    _atomic_write(aggregate_path, comparison)
    compact = {
        "schema": SCHEMA,
        "status": "complete",
        "decision": comparison["decision"],
        "aggregate": {
            "path": str(aggregate_path),
            "sha256": file_sha256(aggregate_path),
        },
        "input_bindings": bindings,
        "source_sha256": comparison["source_sha256"],
    }
    _atomic_write(decision_path, compact)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_decision_markdown(comparison), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "decision": comparison["decision"]["decision"],
                "aggregate": str(aggregate_path),
                "decision_json": str(decision_path),
                "decision_markdown": str(markdown_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
