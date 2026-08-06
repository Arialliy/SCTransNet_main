#!/usr/bin/env python3
"""Build the frozen train-only sample manifest for the DS gradient audit.

The builder is deliberately independent from model predictions and
checkpoints.  It reads only the three frozen ``img_idx/train`` splits through
``ThreeDatasetV2TrainDataset``, enumerates deterministic stateless training
crops, stratifies them from original-resolution ground truth components, and
stores exact FP32 tensor hashes for the samples that a later gradient analyzer
must consume.

``background_only`` is availability-conditional.  In particular, every
NUDT-SIRST train image is 256x256 and positive, so the formal 256x256 crop can
never produce a natural background-only sample.  A zero-candidate background
stratum is recorded as structurally unavailable instead of fabricating a
different crop protocol.  Non-empty background strata and both positive
strata must satisfy the complete 64-sample/24-source contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import paper_three_dataset_v2 as dataset_module  # noqa: E402
from experiments import three_dataset_v2_protocol as protocol  # noqa: E402


SCHEMA = "sctransnet_three_dataset_ds_gradient_audit_manifest_v1/v1"
AUDIT_NAMESPACE = "sctransnet-ds-gradient-audit-v1"
TRAINING_SEED = 42
SPLIT = "train"
EPOCH_START = 1
EPOCH_END = 1000
EPOCH_CANDIDATES_PER_DATASET = 32
TINY_COMPONENT_AREA = 9
MASK_FOREGROUND_THRESHOLD_UINT8 = 128
STRATA = ("background_only", "tiny_positive", "normal_positive")
POSITIVE_STRATA = ("tiny_positive", "normal_positive")
SAMPLES_PER_AVAILABLE_STRATUM = 64
MAX_SAMPLES_PER_SOURCE_PER_STRATUM = 3
MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM = 24
BATCH_SIZE = 16
BATCHES_PER_AVAILABLE_STRATUM = 4

DEFAULT_OUTPUT_PATH = (
    protocol.DEFAULT_RESULTS_ROOT
    / "manifests"
    / "three_dataset_ds_gradient_audit_manifest_v1.json"
)


class DSGradientAuditManifestError(ValueError):
    """The requested sample manifest violates the frozen audit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DSGradientAuditManifestError(message)


def stable_digest(*parts: Any) -> str:
    """Return the analyzer-shared deterministic SHA-256 selection key.

    This representation is intentionally not the training augmentation seed
    representation.  It is a separate, frozen audit-sampling namespace.
    """

    try:
        encoded = json.dumps(
            list(parts),
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DSGradientAuditManifestError(
            f"stable digest parts are not JSON encodable: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def ranked_candidate_epochs(dataset_name: str) -> tuple[dict[str, Any], ...]:
    """Hash-rank epoch 1..1000 and retain the first 32 for one dataset."""

    protocol.require_dataset(dataset_name)
    ranked = sorted(
        (
            stable_digest(
                AUDIT_NAMESPACE,
                TRAINING_SEED,
                dataset_name,
                epoch,
            ),
            epoch,
        )
        for epoch in range(EPOCH_START, EPOCH_END + 1)
    )[:EPOCH_CANDIDATES_PER_DATASET]
    return tuple(
        {
            "rank": rank,
            "epoch": epoch,
            "selection_sha256": digest,
        }
        for rank, (digest, epoch) in enumerate(ranked)
    )


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor bytes exactly as the downstream analyzer does."""

    _require(isinstance(value, torch.Tensor), "tensor SHA input is not a Tensor")
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _tensor_binding(value: torch.Tensor, *, expected_ndim: int) -> dict[str, Any]:
    _require(value.ndim == expected_ndim, f"tensor must have ndim={expected_ndim}")
    _require(value.dtype == torch.float32, "audit tensors must remain FP32")
    _require(bool(torch.isfinite(value).all()), "audit tensor contains non-finite values")
    return {
        "dtype": "float32",
        "shape": [int(size) for size in value.shape],
        "sha256": tensor_sha256(value),
        "hash_contract": (
            "sha256(t.detach().cpu().contiguous().view(torch.uint8)"
            ".numpy().tobytes())"
        ),
    }


@dataclass(frozen=True, slots=True)
class GroundTruthComponent:
    """One original-resolution, 8-connected foreground component."""

    pixels: frozenset[tuple[int, int]]
    area: int
    top: int
    left: int
    bottom_inclusive: int
    right_inclusive: int

    def intersects_crop(self, crop_top: int, crop_left: int, crop_size: int) -> bool:
        crop_bottom = crop_top + crop_size
        crop_right = crop_left + crop_size
        if (
            self.bottom_inclusive < crop_top
            or self.right_inclusive < crop_left
            or self.top >= crop_bottom
            or self.left >= crop_right
        ):
            return False
        return any(
            crop_top <= row < crop_bottom and crop_left <= column < crop_right
            for row, column in self.pixels
        )


def connected_components_8(binary_mask: np.ndarray) -> tuple[GroundTruthComponent, ...]:
    """Extract exact 8-connected components from a two-dimensional GT mask."""

    mask = np.asarray(binary_mask, dtype=np.bool_)
    _require(mask.ndim == 2, "ground-truth mask must be two-dimensional")
    remaining = {
        (int(row), int(column)) for row, column in np.argwhere(mask)
    }
    components: list[GroundTruthComponent] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        pixels: set[tuple[int, int]] = {seed}
        while stack:
            row, column = stack.pop()
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor = (row + row_delta, column + column_delta)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pixels.add(neighbor)
                        stack.append(neighbor)
        rows = [row for row, _ in pixels]
        columns = [column for _, column in pixels]
        components.append(
            GroundTruthComponent(
                pixels=frozenset(pixels),
                area=len(pixels),
                top=min(rows),
                left=min(columns),
                bottom_inclusive=max(rows),
                right_inclusive=max(columns),
            )
        )
    components.sort(
        key=lambda item: (
            item.top,
            item.left,
            item.bottom_inclusive,
            item.right_inclusive,
            item.area,
        )
    )
    return tuple(components)


def classify_crop(
    components: Sequence[GroundTruthComponent],
    *,
    crop_top: int,
    crop_left: int,
    crop_size: int,
) -> dict[str, Any]:
    """Classify a crop by original-GT intersection, with normal precedence."""

    _require(crop_top >= 0 and crop_left >= 0, "crop origin must be non-negative")
    _require(crop_size > 0, "crop size must be positive")
    intersected = tuple(
        component
        for component in components
        if component.intersects_crop(crop_top, crop_left, crop_size)
    )
    tiny = tuple(
        component
        for component in intersected
        if component.area <= TINY_COMPONENT_AREA
    )
    normal = tuple(
        component
        for component in intersected
        if component.area > TINY_COMPONENT_AREA
    )
    if normal:
        stratum = "normal_positive"
    elif tiny:
        stratum = "tiny_positive"
    else:
        stratum = "background_only"
    return {
        "stratum": stratum,
        "mixed_tiny": bool(normal and tiny),
        "intersected_component_count": len(intersected),
        "intersected_tiny_component_count": len(tiny),
        "intersected_normal_component_count": len(normal),
        "intersected_component_areas": sorted(
            int(component.area) for component in intersected
        ),
    }


@dataclass(frozen=True, slots=True)
class AuditCandidate:
    dataset_name: str
    dataset_index: int
    source_id: str
    namespaced_source_id: str
    epoch: int
    epoch_rank: int
    epoch_selection_sha256: str
    candidate_selection_sha256: str
    augmentation_seed: int
    transform_plan: Mapping[str, Any]
    original_height: int
    original_width: int
    source_component_count: int
    source_tiny_component_count: int
    source_normal_component_count: int
    stratum: str
    mixed_tiny: bool
    intersected_component_count: int
    intersected_tiny_component_count: int
    intersected_normal_component_count: int
    intersected_component_areas: tuple[int, ...]

    @property
    def identity(self) -> tuple[str, int]:
        return (self.source_id, self.epoch)


def _candidate_from_plan(
    *,
    dataset_name: str,
    dataset_index: int,
    source_id: str,
    epoch_binding: Mapping[str, Any],
    plan: protocol.StatelessTransformPlan,
    components: Sequence[GroundTruthComponent],
    original_height: int,
    original_width: int,
) -> AuditCandidate:
    classification = classify_crop(
        components,
        crop_top=plan.crop_top,
        crop_left=plan.crop_left,
        crop_size=plan.crop_size,
    )
    stratum = str(classification["stratum"])
    epoch = int(epoch_binding["epoch"])
    namespaced_source_id = f"{dataset_name}::{source_id}"
    return AuditCandidate(
        dataset_name=dataset_name,
        dataset_index=dataset_index,
        source_id=source_id,
        namespaced_source_id=namespaced_source_id,
        epoch=epoch,
        epoch_rank=int(epoch_binding["rank"]),
        epoch_selection_sha256=str(epoch_binding["selection_sha256"]),
        candidate_selection_sha256=stable_digest(
            AUDIT_NAMESPACE,
            TRAINING_SEED,
            dataset_name,
            stratum,
            source_id,
            epoch,
        ),
        augmentation_seed=int(plan.augmentation_seed),
        transform_plan=asdict(plan),
        original_height=original_height,
        original_width=original_width,
        source_component_count=len(components),
        source_tiny_component_count=sum(
            component.area <= TINY_COMPONENT_AREA for component in components
        ),
        source_normal_component_count=sum(
            component.area > TINY_COMPONENT_AREA for component in components
        ),
        stratum=stratum,
        mixed_tiny=bool(classification["mixed_tiny"]),
        intersected_component_count=int(
            classification["intersected_component_count"]
        ),
        intersected_tiny_component_count=int(
            classification["intersected_tiny_component_count"]
        ),
        intersected_normal_component_count=int(
            classification["intersected_normal_component_count"]
        ),
        intersected_component_areas=tuple(
            int(value) for value in classification["intersected_component_areas"]
        ),
    )


def enumerate_dataset_candidates(
    train_dataset: dataset_module.ThreeDatasetV2TrainDataset,
) -> dict[str, list[AuditCandidate]]:
    """Enumerate 32 hash-selected formal crops for every train source."""

    dataset_name = protocol.require_dataset(train_dataset.dataset_name)
    _require(train_dataset.seed == TRAINING_SEED, "train dataset seed differs")
    _require(train_dataset.patch_size == protocol.PATCH_SIZE, "patch size differs")
    _require(train_dataset.return_metadata, "audit dataset must return metadata")
    epoch_bindings = ranked_candidate_epochs(dataset_name)
    known_ids = frozenset(train_dataset.sample_ids)
    pools: dict[str, list[AuditCandidate]] = {stratum: [] for stratum in STRATA}
    for dataset_index, source_id in enumerate(train_dataset.sample_ids):
        sample = protocol.resolve_sample(
            train_dataset.dataset_root,
            dataset_name,
            source_id,
            split=SPLIT,
            known_ids=known_ids,
        )
        image, raw_mask = dataset_module._load_pair(sample)
        original_height, original_width = image.shape
        training_positive = raw_mask > 0
        components = connected_components_8(
            raw_mask >= np.float32(MASK_FOREGROUND_THRESHOLD_UINT8)
        )

        def has_positive(top: int, left: int, size: int) -> bool:
            return bool(
                np.any(training_positive[top : top + size, left : left + size])
            )

        namespaced_source_id = f"{dataset_name}::{source_id}"
        for epoch_binding in epoch_bindings:
            plan = protocol.derive_stateless_transform_plan(
                protocol_seed=TRAINING_SEED,
                dataset_name=dataset_name,
                epoch=int(epoch_binding["epoch"]),
                namespaced_id=namespaced_source_id,
                image_height=original_height,
                image_width=original_width,
                has_positive_in_crop=has_positive,
                patch_size=protocol.PATCH_SIZE,
            )
            candidate = _candidate_from_plan(
                dataset_name=dataset_name,
                dataset_index=dataset_index,
                source_id=source_id,
                epoch_binding=epoch_binding,
                plan=plan,
                components=components,
                original_height=original_height,
                original_width=original_width,
            )
            pools[candidate.stratum].append(candidate)
    return pools


def exhaustive_natural_source_availability_proof(
    train_dataset: dataset_module.ThreeDatasetV2TrainDataset,
    stratum: str,
) -> dict[str, Any]:
    """Prove the natural distinct-source ceiling across every epoch 1..1000.

    The proof is invoked only when the default 32-epoch candidate pool cannot
    reach 24 different original images.  Sources that lack a component type
    required by the requested stratum are ruled out directly from the
    original GT; all potentially matching sources enumerate every formal
    stateless transform plan for epochs 1..1000.
    """

    dataset_name = protocol.require_dataset(train_dataset.dataset_name)
    _require(stratum in STRATA, f"unknown proof stratum: {stratum!r}")
    _require(train_dataset.seed == TRAINING_SEED, "proof dataset seed differs")
    known_ids = frozenset(train_dataset.sample_ids)
    source_rows: list[dict[str, Any]] = []
    matching_source_ids: list[str] = []
    matching_candidate_count = 0
    derived_transform_plan_count = 0
    for dataset_index, source_id in enumerate(train_dataset.sample_ids):
        sample = protocol.resolve_sample(
            train_dataset.dataset_root,
            dataset_name,
            source_id,
            split=SPLIT,
            known_ids=known_ids,
        )
        image, raw_mask = dataset_module._load_pair(sample)
        original_height, original_width = image.shape
        training_positive = raw_mask > 0
        components = connected_components_8(
            raw_mask >= np.float32(MASK_FOREGROUND_THRESHOLD_UINT8)
        )
        has_tiny = any(
            component.area <= TINY_COMPONENT_AREA for component in components
        )
        has_normal = any(
            component.area > TINY_COMPONENT_AREA for component in components
        )
        can_rule_out_from_original_gt = (
            (stratum == "tiny_positive" and not has_tiny)
            or (stratum == "normal_positive" and not has_normal)
            or (stratum == "background_only" and original_height <= protocol.PATCH_SIZE
                and original_width <= protocol.PATCH_SIZE and bool(components))
        )
        matching_epochs: list[int] = []
        if not can_rule_out_from_original_gt:

            def has_positive(top: int, left: int, size: int) -> bool:
                return bool(
                    np.any(
                        training_positive[top : top + size, left : left + size]
                    )
                )

            namespaced_source_id = f"{dataset_name}::{source_id}"
            for epoch in range(EPOCH_START, EPOCH_END + 1):
                plan = protocol.derive_stateless_transform_plan(
                    protocol_seed=TRAINING_SEED,
                    dataset_name=dataset_name,
                    epoch=epoch,
                    namespaced_id=namespaced_source_id,
                    image_height=original_height,
                    image_width=original_width,
                    has_positive_in_crop=has_positive,
                    patch_size=protocol.PATCH_SIZE,
                )
                derived_transform_plan_count += 1
                observed = classify_crop(
                    components,
                    crop_top=plan.crop_top,
                    crop_left=plan.crop_left,
                    crop_size=plan.crop_size,
                )
                if observed["stratum"] == stratum:
                    matching_epochs.append(epoch)
        if matching_epochs:
            matching_source_ids.append(source_id)
            matching_candidate_count += len(matching_epochs)
        source_rows.append(
            {
                "dataset_index": dataset_index,
                "source_id": source_id,
                "original_gt_component_count": len(components),
                "original_gt_has_tiny_component_area_le_9": has_tiny,
                "original_gt_has_normal_component_area_gt_9": has_normal,
                "ruled_out_without_epoch_enumeration": (
                    can_rule_out_from_original_gt
                ),
                "matching_epoch_count": len(matching_epochs),
                "first_matching_epoch": (
                    matching_epochs[0] if matching_epochs else None
                ),
                "matching_epochs_sha256": hashlib.sha256(
                    protocol.compact_json_bytes(matching_epochs)
                ).hexdigest(),
            }
        )
    proof_body: dict[str, Any] = {
        "proof_schema": "sctransnet_ds_gradient_audit_natural_source_ceiling/v1",
        "audit_namespace": AUDIT_NAMESPACE,
        "training_seed": TRAINING_SEED,
        "dataset_name": dataset_name,
        "split": SPLIT,
        "stratum": stratum,
        "epoch_range_inclusive": [EPOCH_START, EPOCH_END],
        "epoch_count_per_source": EPOCH_END - EPOCH_START + 1,
        "train_source_count": len(train_dataset.sample_ids),
        "logical_source_epoch_count": (
            len(train_dataset.sample_ids) * (EPOCH_END - EPOCH_START + 1)
        ),
        "derived_transform_plan_count": derived_transform_plan_count,
        "original_gt_rule_out_is_exact": True,
        "full_epoch_range_covered_for_every_non_ruled_out_source": True,
        "distinct_matching_source_count": len(matching_source_ids),
        "matching_candidate_count": matching_candidate_count,
        "matching_source_ids": matching_source_ids,
        "matching_source_ids_sha256": hashlib.sha256(
            protocol.compact_json_bytes(matching_source_ids)
        ).hexdigest(),
        "source_rows": source_rows,
        "source_rows_sha256": hashlib.sha256(
            protocol.compact_json_bytes(source_rows)
        ).hexdigest(),
    }
    proof_body["proof_sha256"] = hashlib.sha256(
        protocol.compact_json_bytes(proof_body)
    ).hexdigest()
    return proof_body


def _unavailable_background_selection() -> dict[str, Any]:
    return {
        "stratum": "background_only",
        "availability_required": False,
        "structurally_unavailable": True,
        "availability_reason": (
            "no natural background-only crop exists among the formal "
            "seed-42 img_idx/train candidates; no synthetic crop is created"
        ),
        "candidate_count": 0,
        "candidate_distinct_source_count": 0,
        "natural_distinct_source_ceiling": 0,
        "diversity_target_limited_by_natural_availability": False,
        "default_min_distinct_sources": (
            MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM
        ),
        "effective_min_distinct_sources": 0,
        "default_max_samples_per_source": MAX_SAMPLES_PER_SOURCE_PER_STRATUM,
        "effective_max_samples_per_source": 0,
        "exhaustive_natural_availability_proof": None,
        "selected_count": 0,
        "selected_distinct_source_count": 0,
        "observed_max_samples_per_source": 0,
        "observed_min_samples_per_source": 0,
        "repetition_count_range": 0,
        "coverage_pass": True,
        "selected_candidates": [],
    }


def select_stratum_candidates(
    dataset_name: str,
    stratum: str,
    candidates: Sequence[AuditCandidate],
    *,
    natural_availability_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one 64-record stratum with deterministic diversity constraints."""

    protocol.require_dataset(dataset_name)
    _require(stratum in STRATA, f"unknown audit stratum: {stratum!r}")
    for candidate in candidates:
        _require(candidate.dataset_name == dataset_name, "candidate dataset differs")
        _require(candidate.stratum == stratum, "candidate stratum differs")
    if not candidates:
        if stratum == "background_only":
            return _unavailable_background_selection()
        raise DSGradientAuditManifestError(
            f"{dataset_name} {stratum} has no formal crop candidates"
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.candidate_selection_sha256,
            candidate.dataset_index,
            candidate.epoch,
        ),
    )
    identities = [candidate.identity for candidate in ordered]
    _require(
        len(identities) == len(set(identities)),
        f"{dataset_name} {stratum} candidate identities are not unique",
    )
    candidate_sources = {candidate.source_id for candidate in ordered}
    natural_ceiling = len(candidate_sources)
    diversity_limited = False
    effective_diversity_target = MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM
    effective_source_cap = MAX_SAMPLES_PER_SOURCE_PER_STRATUM
    proof_binding: dict[str, Any] | None = None
    if len(candidate_sources) < MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM:
        if natural_availability_proof is None:
            raise DSGradientAuditManifestError(
                f"{dataset_name} {stratum} has only {len(candidate_sources)} "
                "distinct candidate sources; an exhaustive natural-availability "
                "proof is required"
            )
        proof_binding = dict(natural_availability_proof)
        _require(
            proof_binding.get("dataset_name") == dataset_name,
            "natural-availability proof dataset differs",
        )
        _require(
            proof_binding.get("stratum") == stratum,
            "natural-availability proof stratum differs",
        )
        _require(
            proof_binding.get("full_epoch_range_covered_for_every_non_ruled_out_source")
            is True,
            "natural-availability proof is not exhaustive",
        )
        proof_sha = proof_binding.get("proof_sha256")
        _require(
            isinstance(proof_sha, str) and len(proof_sha) == 64,
            "natural-availability proof SHA is invalid",
        )
        natural_ceiling = int(
            proof_binding.get("distinct_matching_source_count", -1)
        )
        proof_sources_raw = proof_binding.get("matching_source_ids")
        _require(
            isinstance(proof_sources_raw, list)
            and all(isinstance(value, str) for value in proof_sources_raw),
            "natural-availability proof matching source IDs are invalid",
        )
        proof_sources = set(proof_sources_raw)
        _require(
            len(proof_sources) == natural_ceiling,
            "natural-availability proof source count differs",
        )
        _require(
            candidate_sources == proof_sources,
            "32-epoch candidate pool does not cover the complete proven "
            "natural source ceiling",
        )
        if natural_ceiling < 16:
            raise DSGradientAuditManifestError(
                f"{dataset_name} {stratum} natural distinct-source ceiling "
                f"{natural_ceiling} is below the minimum 16"
            )
        _require(
            natural_ceiling < MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM,
            "natural-availability proof does not justify a diversity exception",
        )
        diversity_limited = True
        effective_diversity_target = natural_ceiling
        effective_source_cap = max(
            MAX_SAMPLES_PER_SOURCE_PER_STRATUM,
            math.ceil(SAMPLES_PER_AVAILABLE_STRATUM / natural_ceiling),
        )
    capped_capacity = sum(
        min(effective_source_cap, count)
        for count in Counter(candidate.source_id for candidate in ordered).values()
    )
    if capped_capacity < SAMPLES_PER_AVAILABLE_STRATUM:
        raise DSGradientAuditManifestError(
            f"{dataset_name} {stratum} capped capacity is {capped_capacity}; "
            f"{SAMPLES_PER_AVAILABLE_STRATUM} samples are required"
        )

    selected: list[AuditCandidate] = []
    selected_ids: set[tuple[str, int]] = set()
    source_counts: Counter[str] = Counter()
    # First pass fixes the source universe from the same candidate digest order.
    selected_source_order: list[str] = []
    for candidate in ordered:
        if candidate.source_id in selected_source_order:
            continue
        selected_source_order.append(candidate.source_id)
        if len(selected_source_order) == effective_diversity_target:
            break
    _require(
        len(selected_source_order) == effective_diversity_target,
        f"{dataset_name} {stratum} cannot reach effective diversity target",
    )
    if diversity_limited:
        _require(
            set(selected_source_order) == candidate_sources,
            "availability-limited selection must cover every natural source",
        )
    candidates_by_source: dict[str, list[AuditCandidate]] = {
        source_id: [] for source_id in selected_source_order
    }
    for candidate in ordered:
        if candidate.source_id in candidates_by_source:
            candidates_by_source[candidate.source_id].append(candidate)

    # Add one candidate per source per round.  This minimizes repetition
    # imbalance; for NUAA tiny D=21 the final histogram is twenty 3s and one 4.
    while len(selected) < SAMPLES_PER_AVAILABLE_STRATUM:
        progress = False
        for source_id in selected_source_order:
            if len(selected) == SAMPLES_PER_AVAILABLE_STRATUM:
                break
            if source_counts[source_id] >= effective_source_cap:
                continue
            source_candidates = candidates_by_source[source_id]
            next_candidate = next(
                (
                    candidate
                    for candidate in source_candidates
                    if candidate.identity not in selected_ids
                ),
                None,
            )
            if next_candidate is None:
                continue
            selected.append(next_candidate)
            selected_ids.add(next_candidate.identity)
            source_counts[source_id] += 1
            progress = True
        if not progress:
            break

    _require(
        len(selected) == SAMPLES_PER_AVAILABLE_STRATUM,
        f"{dataset_name} {stratum} selection did not reach 64 samples",
    )
    _require(
        len(source_counts) >= effective_diversity_target,
        f"{dataset_name} {stratum} selection lost source diversity",
    )
    observed_max = max(source_counts.values())
    _require(
        observed_max <= effective_source_cap,
        f"{dataset_name} {stratum} selection exceeded source cap",
    )
    observed_min = min(source_counts.values())
    _require(
        observed_max - observed_min <= 1,
        f"{dataset_name} {stratum} repetition is not maximally balanced",
    )
    return {
        "stratum": stratum,
        "availability_required": stratum in POSITIVE_STRATA,
        "structurally_unavailable": False,
        "availability_reason": None,
        "candidate_count": len(ordered),
        "candidate_distinct_source_count": len(candidate_sources),
        "natural_distinct_source_ceiling": natural_ceiling,
        "diversity_target_limited_by_natural_availability": diversity_limited,
        "default_min_distinct_sources": (
            MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM
        ),
        "effective_min_distinct_sources": effective_diversity_target,
        "default_max_samples_per_source": MAX_SAMPLES_PER_SOURCE_PER_STRATUM,
        "effective_max_samples_per_source": effective_source_cap,
        "exhaustive_natural_availability_proof": proof_binding,
        "capped_candidate_capacity": capped_capacity,
        "selected_count": len(selected),
        "selected_distinct_source_count": len(source_counts),
        "observed_max_samples_per_source": observed_max,
        "observed_min_samples_per_source": observed_min,
        "repetition_count_range": observed_max - observed_min,
        "coverage_pass": True,
        "selected_source_histogram": dict(sorted(source_counts.items())),
        "selected_candidates": selected,
    }


def _relative_to(candidate: Path, root: Path, *, label: str) -> str:
    try:
        return candidate.resolve(strict=True).relative_to(
            root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise DSGradientAuditManifestError(
            f"{label} is outside its declared root: {candidate}: {exc}"
        ) from exc


def _source_file_binding(
    train_dataset: dataset_module.ThreeDatasetV2TrainDataset,
    candidate: AuditCandidate,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    namespaced = candidate.namespaced_source_id
    if namespaced in cache:
        return cache[namespaced]
    known_ids = frozenset(train_dataset.sample_ids)
    sample = protocol.resolve_sample(
        train_dataset.dataset_root,
        candidate.dataset_name,
        candidate.source_id,
        split=SPLIT,
        known_ids=known_ids,
    )
    binding = {
        "namespaced_source_id": namespaced,
        "image_relpath_from_dataset_root": _relative_to(
            sample.image_path,
            train_dataset.dataset_root,
            label="source image",
        ),
        "effective_mask_relpath_from_dataset_root": _relative_to(
            sample.mask_path,
            train_dataset.dataset_root,
            label="source mask",
        ),
        "image_file_sha256": protocol.sha256_file(sample.image_path),
        "effective_mask_file_sha256": protocol.sha256_file(sample.mask_path),
        "correction_id": sample.correction_id,
    }
    cache[namespaced] = binding
    return binding


def _materialize_candidate(
    train_dataset: dataset_module.ThreeDatasetV2TrainDataset,
    candidate: AuditCandidate,
    *,
    record_index: int,
    batch_index: int,
    stratum_batch_index: int,
    batch_position: int,
    source_binding_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    train_dataset.set_epoch(candidate.epoch)
    item = train_dataset[candidate.dataset_index]
    _require(isinstance(item, Mapping), "metadata-enabled dataset item is not a mapping")
    image = item.get("image")
    mask = item.get("mask")
    _require(isinstance(image, torch.Tensor), "dataset item lacks image Tensor")
    _require(isinstance(mask, torch.Tensor), "dataset item lacks mask Tensor")
    _require(image.shape == mask.shape, "image/mask tensor shapes differ")
    _require(
        tuple(image.shape) == (1, protocol.PATCH_SIZE, protocol.PATCH_SIZE),
        "formal sample tensor shape differs from CHW=(1,256,256)",
    )
    _require(item.get("dataset_name") == candidate.dataset_name, "dataset metadata differs")
    _require(item.get("sample_id") == candidate.source_id, "source metadata differs")
    _require(item.get("epoch") == candidate.epoch, "epoch metadata differs")
    _require(
        item.get("augmentation_seed") == candidate.augmentation_seed,
        "augmentation seed differs from enumerated plan",
    )
    item_plan = item.get("transform_plan")
    _require(
        isinstance(item_plan, Mapping) and dict(item_plan) == dict(candidate.transform_plan),
        "materialized transform plan differs from enumerated plan",
    )
    image_binding = _tensor_binding(image, expected_ndim=3)
    mask_binding = _tensor_binding(mask, expected_ndim=3)
    source_binding = _source_file_binding(
        train_dataset,
        candidate,
        source_binding_cache,
    )
    record = {
        "record_index": record_index,
        "dataset_name": candidate.dataset_name,
        "dataset_index": candidate.dataset_index,
        "source_id": candidate.source_id,
        "namespaced_source_id": candidate.namespaced_source_id,
        "epoch": candidate.epoch,
        "epoch_rank": candidate.epoch_rank,
        "epoch_selection_sha256": candidate.epoch_selection_sha256,
        "candidate_selection_sha256": candidate.candidate_selection_sha256,
        "augmentation_seed": candidate.augmentation_seed,
        "transform_plan": dict(candidate.transform_plan),
        "original_height": candidate.original_height,
        "original_width": candidate.original_width,
        "stratum": candidate.stratum,
        "mixed_tiny": candidate.mixed_tiny,
        "source_component_count": candidate.source_component_count,
        "source_tiny_component_count_area_le_9": (
            candidate.source_tiny_component_count
        ),
        "source_normal_component_count_area_gt_9": (
            candidate.source_normal_component_count
        ),
        "intersected_component_count": candidate.intersected_component_count,
        "intersected_tiny_component_count_area_le_9": (
            candidate.intersected_tiny_component_count
        ),
        "intersected_normal_component_count_area_gt_9": (
            candidate.intersected_normal_component_count
        ),
        "intersected_component_areas": list(
            candidate.intersected_component_areas
        ),
        "batch_index": batch_index,
        "stratum_batch_index": stratum_batch_index,
        "batch_position": batch_position,
        "image_tensor_sha256": image_binding["sha256"],
        "mask_tensor_sha256": mask_binding["sha256"],
        "image_tensor": image_binding,
        "mask_tensor": mask_binding,
        "source_image_file_sha256": source_binding["image_file_sha256"],
        "source_effective_mask_file_sha256": source_binding[
            "effective_mask_file_sha256"
        ],
    }
    return record, image, mask


def _dataset_train_index_binding(
    dataset_root: Path,
    dataset_name: str,
    train_ids: Sequence[str],
) -> dict[str, Any]:
    expected = protocol.EXPECTED_SPLITS[dataset_name][SPLIT]
    index = protocol.index_path(dataset_root, dataset_name, SPLIT).resolve(strict=True)
    observed_file_sha = protocol.sha256_file(index)
    observed_order_sha = protocol.ordered_ids_sha256(train_ids)
    _require(observed_file_sha == expected["file_sha256"], "train index file SHA differs")
    _require(
        observed_order_sha == expected["ordered_ids_sha256"],
        "train index ordered-ID SHA differs",
    )
    return {
        "split": SPLIT,
        "path": str(index),
        "relpath_from_dataset_root": _relative_to(
            index,
            dataset_root,
            label="train index",
        ),
        "count": len(train_ids),
        "file_sha256": observed_file_sha,
        "ordered_ids_sha256": observed_order_sha,
        "ids_are_in_authoritative_index_order": True,
    }


def _build_one_dataset(
    *,
    dataset_name: str,
    dataset_root: Path,
    protocol_manifest_path: Path,
) -> dict[str, Any]:
    train_dataset = dataset_module.ThreeDatasetV2TrainDataset(
        dataset_name,
        dataset_root=dataset_root,
        protocol_manifest=protocol_manifest_path,
        patch_size=protocol.PATCH_SIZE,
        seed=TRAINING_SEED,
        return_metadata=True,
    )
    epoch_bindings = ranked_candidate_epochs(dataset_name)
    candidate_pools = enumerate_dataset_candidates(train_dataset)
    selections: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        candidates = candidate_pools[stratum]
        proof: Mapping[str, Any] | None = None
        candidate_source_count = len(
            {candidate.source_id for candidate in candidates}
        )
        if (
            candidates
            and candidate_source_count
            < MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM
        ):
            proof = exhaustive_natural_source_availability_proof(
                train_dataset,
                stratum,
            )
        selections[stratum] = select_stratum_candidates(
            dataset_name,
            stratum,
            candidates,
            natural_availability_proof=proof,
        )

    records: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    source_binding_cache: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        selection = selections[stratum]
        selected = selection.pop("selected_candidates")
        _require(isinstance(selected, list), "selected candidate container differs")
        if selection["structurally_unavailable"]:
            selection["batch_indices"] = []
            continue
        _require(
            len(selected) == SAMPLES_PER_AVAILABLE_STRATUM,
            "available stratum does not contain 64 selected candidates",
        )
        stratum_batch_indices: list[int] = []
        for stratum_batch_index in range(BATCHES_PER_AVAILABLE_STRATUM):
            batch_index = len(batches)
            stratum_batch_indices.append(batch_index)
            start = stratum_batch_index * BATCH_SIZE
            batch_candidates = selected[start : start + BATCH_SIZE]
            _require(len(batch_candidates) == BATCH_SIZE, "audit batch is not size 16")
            batch_images: list[torch.Tensor] = []
            batch_masks: list[torch.Tensor] = []
            record_indices: list[int] = []
            for batch_position, candidate in enumerate(batch_candidates):
                record_index = len(records)
                record, image, mask = _materialize_candidate(
                    train_dataset,
                    candidate,
                    record_index=record_index,
                    batch_index=batch_index,
                    stratum_batch_index=stratum_batch_index,
                    batch_position=batch_position,
                    source_binding_cache=source_binding_cache,
                )
                records.append(record)
                record_indices.append(record_index)
                batch_images.append(image)
                batch_masks.append(mask)
            images = torch.stack(batch_images, dim=0)
            masks = torch.stack(batch_masks, dim=0)
            image_binding = _tensor_binding(images, expected_ndim=4)
            mask_binding = _tensor_binding(masks, expected_ndim=4)
            batches.append(
                {
                    "batch_index": batch_index,
                    "stratum": stratum,
                    "stratum_batch_index": stratum_batch_index,
                    "batch_size": BATCH_SIZE,
                    "record_indices": record_indices,
                    "source_ids": [
                        candidate.source_id for candidate in batch_candidates
                    ],
                    "epochs": [candidate.epoch for candidate in batch_candidates],
                    "images_tensor_sha256": image_binding["sha256"],
                    "masks_tensor_sha256": mask_binding["sha256"],
                    "images_tensor": image_binding,
                    "masks_tensor": mask_binding,
                }
            )
        selection["batch_indices"] = stratum_batch_indices

    expected_available = sum(
        not bool(selections[stratum]["structurally_unavailable"])
        for stratum in STRATA
    )
    _require(expected_available in (2, 3), "positive audit strata became unavailable")
    _require(
        len(records) == expected_available * SAMPLES_PER_AVAILABLE_STRATUM,
        "dataset selected-record count differs",
    )
    _require(
        len(batches) == expected_available * BATCHES_PER_AVAILABLE_STRATUM,
        "dataset batch count differs",
    )
    return {
        "dataset_name": dataset_name,
        "dataset_order_index": list(protocol.DATASETS).index(dataset_name),
        "split": SPLIT,
        "train_index_binding": _dataset_train_index_binding(
            dataset_root,
            dataset_name,
            train_dataset.sample_ids,
        ),
        "candidate_epoch_selection": {
            "epoch_range_inclusive": [EPOCH_START, EPOCH_END],
            "retained_epoch_count": EPOCH_CANDIDATES_PER_DATASET,
            "epochs": [binding["epoch"] for binding in epoch_bindings],
            "ranked_bindings": list(epoch_bindings),
        },
        "candidate_count": sum(len(pool) for pool in candidate_pools.values()),
        "candidate_count_by_stratum": {
            stratum: len(candidate_pools[stratum]) for stratum in STRATA
        },
        "strata": selections,
        "selected_record_count": len(records),
        "batch_count": len(batches),
        "records": records,
        "batches": batches,
        "selected_source_file_bindings": dict(sorted(source_binding_cache.items())),
    }


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DSGradientAuditManifestError(
            f"{label} must be a regular, non-symlink file: {path}"
        )
    return path.resolve(strict=True)


def _code_source_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(dataset_module.__file__).resolve(),
    )
    return {
        _relative_to(path, REPO_ROOT, label="audit source"): protocol.sha256_file(path)
        for path in paths
    }


def build_manifest(
    *,
    dataset_root: str | Path = protocol.DEFAULT_DATASET_ROOT,
    protocol_manifest: str | Path = protocol.DEFAULT_MANIFEST_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and optionally write the complete three-dataset audit manifest."""

    root = Path(dataset_root).resolve(strict=True)
    manifest_path = _regular_file(Path(protocol_manifest), label="protocol manifest")
    manifest_payload = protocol.load_protocol_manifest(
        manifest_path,
        dataset_root=root,
    )
    _require(manifest_payload.get("training_seed") == TRAINING_SEED, "seed differs")
    _require(
        tuple(manifest_payload.get("dataset_order", ())) == protocol.DATASETS,
        "protocol dataset order differs",
    )
    protocol_manifest_file_sha = protocol.sha256_file(manifest_path)
    protocol_manifest_canonical_sha = hashlib.sha256(
        protocol.canonical_json_bytes(manifest_payload)
    ).hexdigest()
    datasets = {
        dataset_name: _build_one_dataset(
            dataset_name=dataset_name,
            dataset_root=root,
            protocol_manifest_path=manifest_path,
        )
        for dataset_name in protocol.DATASETS
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "audit_namespace": AUDIT_NAMESPACE,
        "training_seed": TRAINING_SEED,
        "dataset_order": list(protocol.DATASETS),
        "split": SPLIT,
        "uses_only_img_idx_train": True,
        "uses_model_outputs": False,
        "uses_checkpoints": False,
        "sampling_contract": {
            "candidate_epoch_range_inclusive": [EPOCH_START, EPOCH_END],
            "candidate_epochs_per_dataset": EPOCH_CANDIDATES_PER_DATASET,
            "epoch_selection_key": (
                "stable_digest(namespace, 42, dataset_name, epoch)"
            ),
            "candidate_selection_key": (
                "stable_digest(namespace, 42, dataset_name, stratum, "
                "source_id, epoch)"
            ),
            "stable_digest_encoding": (
                "sha256(json.dumps(list(parts), ensure_ascii=True, "
                "separators=(',', ':')).encode('utf-8'))"
            ),
            "component_connectivity": 8,
            "component_coordinate_space": "original_resolution_before_crop",
            "component_mask_binarization": "effective_GT_uint8_value>=128",
            "training_positive_crop_callback": "effective_GT_value>0",
            "tiny_component_area_max_inclusive": TINY_COMPONENT_AREA,
            "strata_precedence": [
                "normal_positive",
                "tiny_positive",
                "background_only",
            ],
            "mixed_tiny_definition": (
                "normal_positive crop also intersects at least one area<=9 "
                "original-GT component"
            ),
            "samples_per_available_stratum": SAMPLES_PER_AVAILABLE_STRATUM,
            "max_samples_per_source_per_stratum": (
                MAX_SAMPLES_PER_SOURCE_PER_STRATUM
            ),
            "min_distinct_sources_per_available_stratum": (
                MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM
            ),
            "natural_availability_diversity_exception": (
                "if exhaustive epoch1..1000 enumeration proves a natural "
                "distinct-source ceiling D<24, require D>=16, set the "
                "effective diversity target to D, set the effective cap to "
                "max(3, ceil(64/D)), cover all D sources, and balance source "
                "repetitions to a count range of at most one"
            ),
            "batch_size": BATCH_SIZE,
            "batches_per_available_stratum": BATCHES_PER_AVAILABLE_STRATUM,
            "background_availability": (
                "descriptive availability-conditional; zero natural formal "
                "crop candidates are recorded as structurally unavailable; "
                "synthetic background windows are forbidden"
            ),
        },
        "bindings": {
            "dataset_root": str(root),
            "protocol_manifest_path": str(manifest_path),
            "protocol_manifest_file_sha256": protocol_manifest_file_sha,
            "protocol_manifest_canonical_payload_sha256": (
                protocol_manifest_canonical_sha
            ),
            "protocol_manifest_id": manifest_payload.get("manifest_id"),
            "source_sha256": _code_source_sha256(),
        },
        "datasets": datasets,
        "checks": {
            "dataset_scope_is_exactly_three": tuple(datasets) == protocol.DATASETS,
            "all_positive_strata_have_64_records": all(
                datasets[dataset_name]["strata"][stratum]["selected_count"]
                == SAMPLES_PER_AVAILABLE_STRATUM
                for dataset_name in protocol.DATASETS
                for stratum in POSITIVE_STRATA
            ),
            "all_available_strata_have_four_batches": all(
                len(binding["batch_indices"]) == BATCHES_PER_AVAILABLE_STRATUM
                for dataset in datasets.values()
                for binding in dataset["strata"].values()
                if not binding["structurally_unavailable"]
            ),
            "no_synthetic_background_created": True,
            "model_outputs_accessed": False,
            "checkpoint_accessed": False,
        },
    }
    protocol.canonical_json_bytes(payload)
    if output_path is not None:
        protocol.write_canonical_json(output_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=protocol.DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=protocol.DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_manifest(
        dataset_root=args.dataset_root,
        protocol_manifest=args.protocol_manifest,
        output_path=args.output,
    )
    summary = {
        dataset_name: {
            "records": dataset["selected_record_count"],
            "batches": dataset["batch_count"],
            "background_structurally_unavailable": dataset["strata"]
            ["background_only"]["structurally_unavailable"],
        }
        for dataset_name, dataset in payload["datasets"].items()
    }
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output": str(args.output.resolve()),
                "datasets": summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_NAMESPACE",
    "AuditCandidate",
    "BATCHES_PER_AVAILABLE_STRATUM",
    "BATCH_SIZE",
    "DEFAULT_OUTPUT_PATH",
    "DSGradientAuditManifestError",
    "GroundTruthComponent",
    "MAX_SAMPLES_PER_SOURCE_PER_STRATUM",
    "MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM",
    "SAMPLES_PER_AVAILABLE_STRATUM",
    "SCHEMA",
    "STRATA",
    "TINY_COMPONENT_AREA",
    "build_manifest",
    "classify_crop",
    "connected_components_8",
    "enumerate_dataset_candidates",
    "main",
    "parse_args",
    "ranked_candidate_epochs",
    "select_stratum_candidates",
    "stable_digest",
    "tensor_sha256",
]
