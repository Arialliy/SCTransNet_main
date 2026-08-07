"""Train-only 3-fold OOF and zero-margin selection contract for IRSTD-BGCR.

The sole data authority opened by this module is the already frozen
``official_train_only`` split manifest.  No dataset loader, official-test
index, or official-test artifact is imported or parsed.  The 800 train IDs are
projected into deterministic stratified folds of 267/267/266 samples.

OOF epoch selection pools additive sufficient statistics across the three
mutually exclusive validation folds.  It never averages three rounded fold
mIoUs.  Count ratios remain :class:`fractions.Fraction` objects in the role
key; nIoU and loss are reconstructed from exact rational sums supplied by the
fold evaluators.  The performance acceptance margin is always ``None``.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "sctransnet_irstd_bgcr_run_contract/v1"
FOLD_ASSIGNMENT_SCHEMA = "sctransnet_irstd_bgcr_fold_assignment/v1"
FOLD_MANIFEST_SCHEMA = "sctransnet_irstd_bgcr_fold_manifest/v1"
FOLD_METRIC_SCHEMA = "sctransnet_irstd_bgcr_fold_metric/v1"
POOLED_METRIC_SCHEMA = "sctransnet_irstd_bgcr_pooled_oof_metric/v1"
POOLED_REFERENCE_METRIC_SCHEMA = (
    "sctransnet_irstd_bgcr_pooled_oof_reference_metric/v1"
)
SELECTION_SCHEMA = "sctransnet_irstd_bgcr_oof_selection/v1"

DATASET = "IRSTD-1K"
ROLE = "best_miou"
BGCR_CANDIDATE_KIND = "bgcr"
BGCR_CANDIDATE_NAME = "IRSTD-BGCR"
SOURCE_SCOPE = "official_train_only"
SAMPLE_COUNT = 800
FOLD_COUNT = 3
FOLD_SIZES = (267, 267, 266)
FOLD_SEED = 42
FOLD_TIE_ORDER = (0, 1, 2)
STRATUM_TIE_ORDER = (
    "empty",
    "tiny_single",
    "tiny_multi",
    "small_non_tiny",
    "larger",
)
FOLD_ASSIGNMENT_ALGORITHM = (
    "rare-stratum-first; within each stratum sort by "
    "sha256(seed\\0stratum\\0identifier), then UTF-8 identifier; assign each "
    "sample to the non-full fold minimizing "
    "(same-stratum-count,total-count,fold-tie-rank)"
)

TRAIN_EPOCHS = 120
EVALUATE_EVERY = 5
OOF_EVALUATION_EPOCHS = (0, *range(EVALUATE_EVERY, TRAIN_EPOCHS + 1, EVALUATE_EVERY))
PROBABILITY_THRESHOLD = 0.5
PROBABILITY_COMPARISON = "strict_greater_than"
PERFORMANCE_ACCEPTANCE_MARGIN = None

SOURCE_SPLIT_MANIFEST_PATH = Path(
    "/home/ly/SCTransNet_main/results/two_dataset_pbdr_v3_stage1_v1/"
    "runs/IRSTD-1K/formal/best_miou/core/split_manifest.json"
)
SOURCE_SPLIT_MANIFEST_BYTES = 205_421
SOURCE_SPLIT_MANIFEST_FILE_SHA256 = (
    "8bb2a0cb7cf7802c62ec54e1a43d8ff2524c1c2d45c5ffeb84cb88850f8bdeb4"
)
SOURCE_SPLIT_MANIFEST_SCHEMA = "sctransnet_two_dataset_pbdr_v3_internal_split_v1/v1"
SOURCE_SPLIT_SEMANTIC_SHA256 = (
    "9371a6be7a2671010a3eb014ef4763c97a6528757b2085e262e82237b2e14bac"
)
OFFICIAL_TRAIN_INDEX_SHA256 = (
    "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744"
)

# SHA-256 of ``_assignment_hash_payload`` under the algorithm and source
# binding above.  Any change in an ID, assignment, seed, capacity, or tie order
# fails closed instead of silently creating a different experiment.
FOLD_ASSIGNMENT_SHA256 = (
    "a7ce375391e27e53bdad5f67599d470b336f70c304e22a96b8aa3fef6283c583"
)

OFFICIAL_FALSE_FLAGS = {
    "official_test_accessed": False,
    "official_test_index_opened": False,
    "official_test_index_parsed": False,
    "official_test_loader_built": False,
    "official_evaluation_performed": False,
}

EXACT_COUNT_FIELDS = (
    "intersection_pixels",
    "union_pixels",
    "matched_target_count",
    "target_count",
    "unmatched_component_pixels",
    "valid_pixel_count",
    "matched_tiny_target_count",
    "tiny_target_count",
)

PIXEL_CONFUSION_FIELDS = (
    "true_positive_pixels",
    "false_positive_pixels",
    "false_negative_pixels",
)

ADDITIVE_COUNT_FIELDS = (*EXACT_COUNT_FIELDS, *PIXEL_CONFUSION_FIELDS)


class IRSTDBGCRRunContractError(RuntimeError):
    """An artifact, fold, metric row, or selection violates the BGCR contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IRSTDBGCRRunContractError(message)


def canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IRSTDBGCRRunContractError(
            f"value cannot be represented as canonical JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_int(value: object, *, name: str, positive: bool = False) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    ready = int(value)
    _require(ready >= (1 if positive else 0), f"{name} is outside its valid range")
    return ready


def _finite_float(value: object, *, name: str) -> float:
    _require(not isinstance(value, bool), f"{name} must be a finite real")
    try:
        ready = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise IRSTDBGCRRunContractError(f"{name} must be a finite real") from error
    _require(math.isfinite(ready), f"{name} must be finite")
    return ready


def _identifier_sequence(value: object, *, name: str) -> tuple[str, ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a sequence",
    )
    identifiers = tuple(value)
    _require(
        all(isinstance(item, str) and item and "\x00" not in item for item in identifiers),
        f"{name} contains an invalid identifier",
    )
    _require(len(set(identifiers)) == len(identifiers), f"{name} contains duplicates")
    return identifiers  # type: ignore[return-value]


def _validate_mask_stats(
    value: object,
    *,
    identifiers: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "mask_stats must be a sequence",
    )
    stats: list[dict[str, object]] = []
    required = {
        "identifier",
        "height",
        "width",
        "target_count",
        "target_pixels",
        "minimum_target_area",
        "tiny_target_count",
        "stratum",
    }
    for index, item in enumerate(value):
        _require(isinstance(item, Mapping), f"mask_stats[{index}] must be a mapping")
        _require(required <= set(item), f"mask_stats[{index}] lacks required fields")
        identifier = item["identifier"]
        _require(isinstance(identifier, str) and identifier, "mask stat ID is invalid")
        height = _plain_int(item["height"], name=f"{identifier}.height", positive=True)
        width = _plain_int(item["width"], name=f"{identifier}.width", positive=True)
        target_count = _plain_int(item["target_count"], name=f"{identifier}.target_count")
        target_pixels = _plain_int(item["target_pixels"], name=f"{identifier}.target_pixels")
        minimum_area = _plain_int(
            item["minimum_target_area"], name=f"{identifier}.minimum_target_area"
        )
        tiny_count = _plain_int(
            item["tiny_target_count"], name=f"{identifier}.tiny_target_count"
        )
        stratum = item["stratum"]
        _require(stratum in STRATUM_TIE_ORDER, f"{identifier}.stratum is unsupported")
        _require(target_pixels <= height * width, f"{identifier}.target_pixels is invalid")
        _require(tiny_count <= target_count, f"{identifier}.tiny_target_count is invalid")
        if target_count == 0:
            _require(
                target_pixels == minimum_area == tiny_count == 0 and stratum == "empty",
                f"{identifier} empty-target statistics are inconsistent",
            )
        else:
            _require(
                target_pixels > 0 and minimum_area > 0 and stratum != "empty",
                f"{identifier} non-empty target statistics are inconsistent",
            )
        if stratum == "tiny_single":
            _require(target_count == 1 and tiny_count == 1, "tiny_single is inconsistent")
        if stratum == "tiny_multi":
            _require(target_count > 1 and tiny_count >= 1, "tiny_multi is inconsistent")
        stats.append(dict(item))
    _require(len(stats) == SAMPLE_COUNT, "mask_stats count must be 800")
    _require(
        tuple(item["identifier"] for item in stats) == identifiers,
        "mask_stats order must exactly match official_train_ids",
    )
    return tuple(stats)


def load_train_only_split_manifest(
    path: Path | str = SOURCE_SPLIT_MANIFEST_PATH,
) -> dict[str, object]:
    """Open and validate only the pre-audited train-only split manifest."""

    requested = Path(path)
    _require(requested.is_absolute(), "split manifest path must be absolute")
    _require(requested == SOURCE_SPLIT_MANIFEST_PATH, "unapproved split manifest path")
    _require(requested.is_file(), "approved train-only split manifest is missing")
    _require(
        requested.stat().st_size == SOURCE_SPLIT_MANIFEST_BYTES,
        "train-only split manifest byte size differs",
    )
    _require(
        _file_sha256(requested) == SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "train-only split manifest file SHA differs",
    )
    try:
        payload = json.loads(requested.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IRSTDBGCRRunContractError("cannot parse train-only split manifest") from error
    _require(isinstance(payload, Mapping), "split manifest must be a mapping")
    _require(payload.get("schema") == SOURCE_SPLIT_MANIFEST_SCHEMA, "split schema differs")
    _require(payload.get("dataset") == DATASET, "split dataset differs")
    _require(payload.get("source_split") == SOURCE_SCOPE, "split is not train-only")
    _require(
        payload.get("official_train_index_sha256") == OFFICIAL_TRAIN_INDEX_SHA256,
        "official-train index SHA differs",
    )
    _require(
        payload.get("split_sha256") == SOURCE_SPLIT_SEMANTIC_SHA256,
        "source split semantic SHA differs",
    )
    _require(
        payload.get("official_test_index_opened") is False,
        "source manifest reports official-test index access",
    )
    official_ids = _identifier_sequence(
        payload.get("official_train_ids"), name="official_train_ids"
    )
    _require(len(official_ids) == SAMPLE_COUNT, "official_train_ids count must be 800")
    development_ids = _identifier_sequence(
        payload.get("development_train_ids"), name="development_train_ids"
    )
    validation_ids = _identifier_sequence(
        payload.get("internal_validation_ids"), name="internal_validation_ids"
    )
    _require(
        len(development_ids) == 640 and len(validation_ids) == 160,
        "source 640/160 projection counts differ",
    )
    _require(
        not (set(development_ids) & set(validation_ids))
        and set(development_ids) | set(validation_ids) == set(official_ids),
        "source 640/160 projection is not an exact partition of the 800 train IDs",
    )
    stats = _validate_mask_stats(payload.get("mask_stats"), identifiers=official_ids)
    result = dict(payload)
    result["official_train_ids"] = list(official_ids)
    result["development_train_ids"] = list(development_ids)
    result["internal_validation_ids"] = list(validation_ids)
    result["mask_stats"] = [dict(item) for item in stats]
    return result


def _hashed_stratum_order(identifier: str, stratum: str) -> tuple[str, bytes]:
    digest = hashlib.sha256(
        f"{FOLD_SEED}\0{stratum}\0{identifier}".encode("utf-8")
    ).hexdigest()
    return digest, identifier.encode("utf-8")


def _fold_assignments(
    identifiers: tuple[str, ...],
    stats: tuple[dict[str, object], ...],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    stats_by_id = {str(item["identifier"]): item for item in stats}
    assignments: dict[str, int] = {}
    totals = [0 for _ in FOLD_TIE_ORDER]
    stratum_counts = {
        stratum: [0 for _ in FOLD_TIE_ORDER] for stratum in STRATUM_TIE_ORDER
    }
    for stratum in STRATUM_TIE_ORDER:
        members = [
            identifier
            for identifier in identifiers
            if stats_by_id[identifier]["stratum"] == stratum
        ]
        members.sort(key=lambda item: _hashed_stratum_order(item, stratum))
        for identifier in members:
            eligible = [
                fold
                for fold in FOLD_TIE_ORDER
                if totals[fold] < FOLD_SIZES[fold]
            ]
            _require(bool(eligible), "fold capacities were exhausted early")
            fold = min(
                eligible,
                key=lambda candidate: (
                    stratum_counts[stratum][candidate],
                    totals[candidate],
                    FOLD_TIE_ORDER.index(candidate),
                ),
            )
            assignments[identifier] = fold
            totals[fold] += 1
            stratum_counts[stratum][fold] += 1
    _require(tuple(totals) == FOLD_SIZES, "fold sizes differ from 267/267/266")
    _require(set(assignments) == set(identifiers), "fold assignment does not cover all IDs")
    return assignments, stratum_counts


def _assignment_hash_payload(
    identifiers: tuple[str, ...],
    assignments: Mapping[str, int],
) -> dict[str, object]:
    return {
        "schema": FOLD_ASSIGNMENT_SCHEMA,
        "source_split_manifest_file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "source_official_train_index_sha256": OFFICIAL_TRAIN_INDEX_SHA256,
        "seed": FOLD_SEED,
        "stratum_tie_order": list(STRATUM_TIE_ORDER),
        "fold_tie_order": list(FOLD_TIE_ORDER),
        "target_fold_sizes": list(FOLD_SIZES),
        "assignments": [
            {"identifier": identifier, "fold_index": assignments[identifier]}
            for identifier in identifiers
        ],
    }


def build_frozen_fold_manifest() -> dict[str, object]:
    """Return the sole permitted 3-fold projection and verify its frozen hash."""

    source = load_train_only_split_manifest()
    identifiers = tuple(source["official_train_ids"])  # type: ignore[arg-type]
    stats = tuple(dict(item) for item in source["mask_stats"])  # type: ignore[arg-type]
    assignments, stratum_counts = _fold_assignments(identifiers, stats)
    hash_payload = _assignment_hash_payload(identifiers, assignments)
    observed_assignment_sha256 = canonical_json_sha256(hash_payload)
    _require(
        observed_assignment_sha256 == FOLD_ASSIGNMENT_SHA256,
        "frozen fold assignment SHA differs",
    )
    folds: list[dict[str, object]] = []
    for fold_index in FOLD_TIE_ORDER:
        validation_ids = tuple(
            identifier
            for identifier in identifiers
            if assignments[identifier] == fold_index
        )
        training_ids = tuple(
            identifier
            for identifier in identifiers
            if assignments[identifier] != fold_index
        )
        _require(
            len(validation_ids) == FOLD_SIZES[fold_index]
            and len(training_ids) == SAMPLE_COUNT - FOLD_SIZES[fold_index],
            f"fold {fold_index} train/validation counts differ",
        )
        folds.append(
            {
                "fold_index": fold_index,
                "validation_count": len(validation_ids),
                "training_count": len(training_ids),
                "validation_ids": list(validation_ids),
                "training_ids": list(training_ids),
                "validation_ids_sha256": canonical_json_sha256(list(validation_ids)),
                "training_ids_sha256": canonical_json_sha256(list(training_ids)),
                "validation_stratum_counts": {
                    stratum: stratum_counts[stratum][fold_index]
                    for stratum in STRATUM_TIE_ORDER
                },
            }
        )
    manifest: dict[str, object] = {
        "schema": FOLD_MANIFEST_SCHEMA,
        "run_contract_schema": SCHEMA,
        "dataset": DATASET,
        "source_scope": SOURCE_SCOPE,
        "source_split_manifest": {
            "path": str(SOURCE_SPLIT_MANIFEST_PATH),
            "bytes": SOURCE_SPLIT_MANIFEST_BYTES,
            "file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
            "schema": SOURCE_SPLIT_MANIFEST_SCHEMA,
            "split_sha256": SOURCE_SPLIT_SEMANTIC_SHA256,
            "official_train_index_sha256": OFFICIAL_TRAIN_INDEX_SHA256,
        },
        "sample_count": SAMPLE_COUNT,
        "source_id_order_sha256": canonical_json_sha256(list(identifiers)),
        "source_mask_stats_sha256": canonical_json_sha256(list(stats)),
        "fold_count": FOLD_COUNT,
        "fold_sizes": list(FOLD_SIZES),
        "fold_seed": FOLD_SEED,
        "algorithm": FOLD_ASSIGNMENT_ALGORITHM,
        "stratum_tie_order": list(STRATUM_TIE_ORDER),
        "fold_tie_order": list(FOLD_TIE_ORDER),
        "assignment_sha256": observed_assignment_sha256,
        "assignment_hash_payload_schema": FOLD_ASSIGNMENT_SCHEMA,
        "stratum_counts_by_fold": {
            stratum: list(stratum_counts[stratum]) for stratum in STRATUM_TIE_ORDER
        },
        "folds": folds,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": PROBABILITY_COMPARISON,
        "performance_acceptance_margin": PERFORMANCE_ACCEPTANCE_MARGIN,
        **OFFICIAL_FALSE_FLAGS,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def fold_metric_binding(
    fold_index: int,
    epoch: int,
    *,
    fold_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return mandatory identity fields for one fold/epoch metric history row."""

    fold = _plain_int(fold_index, name="fold_index")
    _require(fold in FOLD_TIE_ORDER, "fold_index is unsupported")
    ready_epoch = _plain_int(epoch, name="epoch")
    _require(ready_epoch in OOF_EVALUATION_EPOCHS, "epoch is outside the OOF schedule")
    manifest = dict(fold_manifest or build_frozen_fold_manifest())
    _require(manifest.get("schema") == FOLD_MANIFEST_SCHEMA, "fold manifest schema differs")
    _require(
        manifest.get("assignment_sha256") == FOLD_ASSIGNMENT_SHA256,
        "fold manifest assignment SHA differs",
    )
    folds = manifest.get("folds")
    _require(isinstance(folds, Sequence), "fold manifest lacks folds")
    record = folds[fold]
    _require(isinstance(record, Mapping) and record.get("fold_index") == fold, "fold differs")
    return {
        "schema": FOLD_METRIC_SCHEMA,
        "dataset": DATASET,
        "role": ROLE,
        "evaluation_scope": "official_train_oof_validation",
        "fold_index": fold,
        "epoch": ready_epoch,
        "sample_count": FOLD_SIZES[fold],
        "validation_ids_sha256": record["validation_ids_sha256"],
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "source_split_manifest_file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": PROBABILITY_COMPARISON,
        "performance_acceptance_margin": PERFORMANCE_ACCEPTANCE_MARGIN,
        **OFFICIAL_FALSE_FLAGS,
    }


def _rational_sum(row: Mapping[str, object], prefix: str) -> Fraction:
    numerator = _plain_int(row.get(f"{prefix}_numerator"), name=f"{prefix}_numerator")
    denominator = _plain_int(
        row.get(f"{prefix}_denominator"),
        name=f"{prefix}_denominator",
        positive=True,
    )
    return Fraction(numerator, denominator)


def _validate_fold_metric_row(
    row: Mapping[str, object],
    *,
    expected_epoch: int,
    fold_manifest: Mapping[str, object],
) -> dict[str, object]:
    _require(isinstance(row, Mapping), "fold metric row must be a mapping")
    fold = _plain_int(row.get("fold_index"), name="fold_index")
    expected_binding = fold_metric_binding(
        fold,
        expected_epoch,
        fold_manifest=fold_manifest,
    )
    for field, expected in expected_binding.items():
        _require(row.get(field) == expected, f"fold metric binding differs: {field}")
    counts = {
        field: _plain_int(row.get(field), name=field) for field in ADDITIVE_COUNT_FIELDS
    }
    _require(counts["union_pixels"] > 0, "union_pixels must be positive")
    _require(counts["target_count"] > 0, "target_count must be positive")
    _require(counts["valid_pixel_count"] > 0, "valid_pixel_count must be positive")
    _require(
        counts["intersection_pixels"] <= counts["union_pixels"]
        <= counts["valid_pixel_count"],
        "pixel sufficient statistics are inconsistent",
    )
    _require(
        counts["matched_target_count"] <= counts["target_count"],
        "matched_target_count exceeds target_count",
    )
    _require(
        counts["matched_tiny_target_count"] <= counts["tiny_target_count"]
        and counts["matched_tiny_target_count"] <= counts["matched_target_count"],
        "tiny-target sufficient statistics are inconsistent",
    )
    _require(
        counts["unmatched_component_pixels"] <= counts["valid_pixel_count"],
        "unmatched_component_pixels exceeds valid_pixel_count",
    )
    _require(
        counts["true_positive_pixels"] == counts["intersection_pixels"],
        "true_positive_pixels must equal intersection_pixels",
    )
    _require(
        counts["union_pixels"]
        == counts["true_positive_pixels"]
        + counts["false_positive_pixels"]
        + counts["false_negative_pixels"],
        "union_pixels must equal TP + FP + FN",
    )
    _require(
        counts["unmatched_component_pixels"] <= counts["false_positive_pixels"],
        "unmatched component pixels cannot exceed false-positive pixels",
    )
    _require(
        counts["true_positive_pixels"]
        + counts["false_positive_pixels"]
        + counts["false_negative_pixels"]
        <= counts["valid_pixel_count"],
        "pixel confusion counts exceed valid pixels",
    )
    niou_sum = _rational_sum(row, "niou_sum")
    loss_sum = _rational_sum(row, "loss_sum")
    sample_count = _plain_int(row.get("sample_count"), name="sample_count", positive=True)
    _require(
        Fraction(0, 1) <= niou_sum <= sample_count,
        "niou_sum is outside [0, sample_count]",
    )
    _require(loss_sum >= 0, "loss_sum must be non-negative")
    result = dict(row)
    result.update(counts)
    return result


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def pool_oof_sufficient_statistics(
    fold_rows: Sequence[Mapping[str, object]],
    *,
    epoch: int,
    _expected_candidate_kind: str = BGCR_CANDIDATE_KIND,
    _expected_candidate_name: str = BGCR_CANDIDATE_NAME,
) -> dict[str, object]:
    """Pool one epoch over all held-out IDs using additive sufficient statistics."""

    ready_epoch = _plain_int(epoch, name="epoch")
    _require(ready_epoch in OOF_EVALUATION_EPOCHS, "epoch is outside the OOF schedule")
    _require(
        isinstance(fold_rows, Sequence) and not isinstance(fold_rows, (str, bytes)),
        "fold_rows must be a sequence",
    )
    _require(len(fold_rows) == FOLD_COUNT, "exactly three fold rows are required")
    _require(
        type(_expected_candidate_kind) is str
        and bool(_expected_candidate_kind)
        and type(_expected_candidate_name) is str
        and bool(_expected_candidate_name),
        "expected candidate identity is invalid",
    )
    for row in fold_rows:
        _require(isinstance(row, Mapping), "fold metric row must be a mapping")
        _require(
            row.get("candidate_kind") == _expected_candidate_kind,
            "fold metric candidate_kind differs",
        )
        _require(
            row.get("candidate_name") == _expected_candidate_name,
            "fold metric candidate_name differs",
        )
    fold_manifest = build_frozen_fold_manifest()
    validated = [
        _validate_fold_metric_row(
            row,
            expected_epoch=ready_epoch,
            fold_manifest=fold_manifest,
        )
        for row in fold_rows
    ]
    _require(
        sorted(int(row["fold_index"]) for row in validated) == list(FOLD_TIE_ORDER),
        "fold rows must contain each fold exactly once",
    )
    validated.sort(key=lambda row: int(row["fold_index"]))
    pooled_counts = {
        field: sum(int(row[field]) for row in validated)
        for field in ADDITIVE_COUNT_FIELDS
    }
    sample_count = sum(int(row["sample_count"]) for row in validated)
    _require(sample_count == SAMPLE_COUNT, "pooled OOF sample count must be 800")
    niou_sum = sum(
        (_rational_sum(row, "niou_sum") for row in validated),
        start=Fraction(0, 1),
    )
    loss_sum = sum(
        (_rational_sum(row, "loss_sum") for row in validated),
        start=Fraction(0, 1),
    )
    miou = Fraction(
        pooled_counts["intersection_pixels"], pooled_counts["union_pixels"]
    )
    pd = Fraction(
        pooled_counts["matched_target_count"], pooled_counts["target_count"]
    )
    fa = Fraction(
        pooled_counts["unmatched_component_pixels"],
        pooled_counts["valid_pixel_count"],
    )
    tiny_pd = (
        Fraction(
            pooled_counts["matched_tiny_target_count"],
            pooled_counts["tiny_target_count"],
        )
        if pooled_counts["tiny_target_count"]
        else Fraction(0, 1)
    )
    niou = niou_sum / sample_count
    loss = loss_sum / sample_count
    true_positive = pooled_counts["true_positive_pixels"]
    false_positive = pooled_counts["false_positive_pixels"]
    false_negative = pooled_counts["false_negative_pixels"]
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    pixel_precision = (
        Fraction(true_positive, precision_denominator)
        if precision_denominator
        else Fraction(0, 1)
    )
    pixel_recall = (
        Fraction(true_positive, recall_denominator)
        if recall_denominator
        else Fraction(0, 1)
    )
    pixel_f1 = (
        Fraction(2 * true_positive, f1_denominator)
        if f1_denominator
        else Fraction(0, 1)
    )
    return {
        "schema": POOLED_METRIC_SCHEMA,
        "dataset": DATASET,
        "role": ROLE,
        "evaluation_scope": "pooled_official_train_oof_validation",
        "candidate_kind": _expected_candidate_kind,
        "candidate_name": _expected_candidate_name,
        "epoch": ready_epoch,
        "fold_indices": list(FOLD_TIE_ORDER),
        "fold_sample_counts": list(FOLD_SIZES),
        "fold_validation_ids_sha256": [
            row["validation_ids_sha256"] for row in validated
        ],
        "sample_count": sample_count,
        **pooled_counts,
        "miou": float(miou),
        "pd": float(pd),
        "fa": float(fa),
        "niou": float(niou),
        "tiny_pd": float(tiny_pd),
        "test_loss": float(loss),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "pixel_f1": float(pixel_f1),
        "pixel_precision_exact": _fraction_payload(pixel_precision),
        "pixel_recall_exact": _fraction_payload(pixel_recall),
        "pixel_f1_exact": _fraction_payload(pixel_f1),
        "niou_sum": _fraction_payload(niou_sum),
        "loss_sum": _fraction_payload(loss_sum),
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "source_split_manifest_file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "fold_metrics_arithmetic_mean_used": False,
        "exact_counts_pooled_before_ratio": True,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": PROBABILITY_COMPARISON,
        "performance_acceptance_margin": PERFORMANCE_ACCEPTANCE_MARGIN,
        **OFFICIAL_FALSE_FLAGS,
    }


def pool_reference_sufficient_statistics(
    fold_rows: Sequence[Mapping[str, object]],
    *,
    reference_name: str,
    epoch: int = 0,
) -> dict[str, object]:
    """Pool a frozen reference on the identical held-out OOF projection.

    Reference rows use the same exact fold binding and additive statistics as
    BGCR rows.  The explicit ``candidate_name``/``candidate_kind`` fields stop
    a Current, Original, or other reference from being silently mixed.
    """

    _require(
        isinstance(reference_name, str)
        and reference_name.strip() == reference_name
        and bool(reference_name),
        "reference_name must be a non-empty stripped string",
    )
    _require(
        isinstance(fold_rows, Sequence) and not isinstance(fold_rows, (str, bytes)),
        "reference fold_rows must be a sequence",
    )
    for row in fold_rows:
        _require(isinstance(row, Mapping), "reference fold row must be a mapping")
        _require(
            row.get("candidate_kind") == "frozen_reference",
            "reference row candidate_kind differs",
        )
        _require(
            row.get("candidate_name") == reference_name,
            "reference row candidate_name differs",
        )
    pooled = pool_oof_sufficient_statistics(
        fold_rows,
        epoch=epoch,
        _expected_candidate_kind="frozen_reference",
        _expected_candidate_name=reference_name,
    )
    pooled["schema"] = POOLED_REFERENCE_METRIC_SCHEMA
    pooled["candidate_kind"] = "frozen_reference"
    pooled["candidate_name"] = reference_name
    return pooled


def irstd_best_miou_role_key(
    pooled_metrics: Mapping[str, object],
    *,
    epoch: int | None = None,
) -> tuple[object, ...]:
    """Frozen exact OOF key: mIoU, Pd, -Fa, nIoU, tiny-Pd, -loss, early epoch."""

    _require(isinstance(pooled_metrics, Mapping), "pooled_metrics must be a mapping")
    _require(
        pooled_metrics.get("schema")
        in (POOLED_METRIC_SCHEMA, POOLED_REFERENCE_METRIC_SCHEMA),
        "pooled schema differs",
    )
    _require(pooled_metrics.get("role") == ROLE, "pooled role differs")
    _require(
        pooled_metrics.get("performance_acceptance_margin") is None,
        "performance margin must be null",
    )
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        _require(pooled_metrics.get(flag) is expected, f"official flag differs: {flag}")
    ready_epoch = _plain_int(
        pooled_metrics.get("epoch") if epoch is None else epoch,
        name="epoch",
    )
    _require(pooled_metrics.get("epoch") == ready_epoch, "pooled epoch differs")
    intersection = _plain_int(pooled_metrics.get("intersection_pixels"), name="intersection")
    union = _plain_int(pooled_metrics.get("union_pixels"), name="union", positive=True)
    matched = _plain_int(pooled_metrics.get("matched_target_count"), name="matched")
    targets = _plain_int(pooled_metrics.get("target_count"), name="targets", positive=True)
    unmatched = _plain_int(
        pooled_metrics.get("unmatched_component_pixels"), name="unmatched"
    )
    valid = _plain_int(
        pooled_metrics.get("valid_pixel_count"), name="valid", positive=True
    )
    matched_tiny = _plain_int(
        pooled_metrics.get("matched_tiny_target_count"), name="matched_tiny"
    )
    tiny = _plain_int(pooled_metrics.get("tiny_target_count"), name="tiny")
    _require(
        intersection <= union <= valid
        and matched <= targets
        and matched_tiny <= tiny
        and matched_tiny <= matched
        and unmatched <= valid,
        "pooled sufficient statistics are inconsistent",
    )
    niou = _finite_float(pooled_metrics.get("niou"), name="niou")
    loss = _finite_float(pooled_metrics.get("test_loss"), name="test_loss")
    _require(0.0 <= niou <= 1.0 and loss >= 0.0, "nIoU/loss is outside its range")
    return (
        Fraction(intersection, union),
        Fraction(matched, targets),
        -Fraction(unmatched, valid),
        niou,
        Fraction(matched_tiny, tiny) if tiny else Fraction(0, 1),
        -loss,
        -ready_epoch,
    )


def serialize_role_key(key: Sequence[object]) -> list[dict[str, object]]:
    fields = ("miou", "pd", "negative_fa", "niou", "tiny_pd", "negative_loss", "negative_epoch")
    _require(len(key) == len(fields), "role key length differs")
    serialized: list[dict[str, object]] = []
    for field, value in zip(fields, key, strict=True):
        if isinstance(value, Fraction):
            serialized.append(
                {"field": field, "representation": "exact_fraction", **_fraction_payload(value)}
            )
        elif isinstance(value, int) and not isinstance(value, bool):
            serialized.append(
                {"field": field, "representation": "integer", "value": value}
            )
        else:
            serialized.append(
                {
                    "field": field,
                    "representation": "binary64_hex",
                    "hex": _finite_float(value, name=field).hex(),
                }
            )
    return serialized


def select_oof_epoch(
    fold_history: Sequence[Mapping[str, object]],
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Select the strict maximum pooled OOF epoch; epoch 0 is mandatory."""

    _require(
        isinstance(fold_history, Sequence)
        and not isinstance(fold_history, (str, bytes))
        and bool(fold_history),
        "fold_history must be a non-empty sequence",
    )
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in fold_history:
        _require(isinstance(row, Mapping), "fold history row must be a mapping")
        epoch = _plain_int(row.get("epoch"), name="epoch")
        grouped.setdefault(epoch, []).append(row)
    _require(0 in grouped, "epoch-0 identity must be in the OOF candidate pool")
    observed_epochs = tuple(sorted(grouped))
    if require_complete:
        _require(
            observed_epochs == OOF_EVALUATION_EPOCHS,
            "complete OOF history must contain epoch 0 and every 5 epochs through 120",
        )
    else:
        _require(
            all(epoch in OOF_EVALUATION_EPOCHS for epoch in observed_epochs),
            "partial OOF history contains an unscheduled epoch",
        )
    summaries = [
        pool_oof_sufficient_statistics(grouped[epoch], epoch=epoch)
        for epoch in observed_epochs
    ]
    winner = summaries[0]
    winner_key = irstd_best_miou_role_key(winner)
    for summary in summaries[1:]:
        candidate_key = irstd_best_miou_role_key(summary)
        if candidate_key > winner_key:
            winner = summary
            winner_key = candidate_key
    identity = summaries[0]
    identity_key = irstd_best_miou_role_key(identity)
    winner_miou = winner_key[0]
    identity_miou = identity_key[0]
    _require(
        isinstance(winner_miou, Fraction) and isinstance(identity_miou, Fraction),
        "mIoU role-key fields are not exact fractions",
    )
    return {
        "schema": SELECTION_SCHEMA,
        "dataset": DATASET,
        "role": ROLE,
        "selection": "strict_lexicographic_pooled_oof_role_key",
        "fold_metrics_arithmetic_mean_used": False,
        "candidate_epochs": list(observed_epochs),
        "identity_epoch": 0,
        "selected_epoch": int(winner["epoch"]),
        "selected_metrics": winner,
        "selected_role_key": serialize_role_key(winner_key),
        "identity_role_key": serialize_role_key(identity_key),
        "strictly_improves_epoch0_full_role_key": winner_key > identity_key,
        "strictly_improves_epoch0_miou": winner_miou > identity_miou,
        "performance_acceptance_margin": PERFORMANCE_ACCEPTANCE_MARGIN,
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "source_split_manifest_file_sha256": SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "epoch_summaries": [
            {
                "epoch": int(summary["epoch"]),
                "metrics": summary,
                "role_key": serialize_role_key(irstd_best_miou_role_key(summary)),
            }
            for summary in summaries
        ],
        **OFFICIAL_FALSE_FLAGS,
    }


def run_contract_manifest() -> dict[str, object]:
    """Return the immutable non-result protocol record consumed by runners."""

    fold_manifest = build_frozen_fold_manifest()
    return {
        "schema": SCHEMA,
        "dataset": DATASET,
        "role": ROLE,
        "source_scope": SOURCE_SCOPE,
        "fold_manifest_sha256": fold_manifest["manifest_sha256"],
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "fold_sizes": list(FOLD_SIZES),
        "fold_seed": FOLD_SEED,
        "stratum_tie_order": list(STRATUM_TIE_ORDER),
        "fold_tie_order": list(FOLD_TIE_ORDER),
        "fold_assignment_algorithm": FOLD_ASSIGNMENT_ALGORITHM,
        "train_epochs": TRAIN_EPOCHS,
        "evaluate_every": EVALUATE_EVERY,
        "evaluation_epochs": list(OOF_EVALUATION_EPOCHS),
        "selection_role_key": [
            "exact_miou",
            "exact_pd",
            "exact_negative_fa",
            "niou",
            "exact_tiny_pd",
            "negative_loss",
            "negative_epoch",
        ],
        "performance_acceptance_margin": PERFORMANCE_ACCEPTANCE_MARGIN,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": PROBABILITY_COMPARISON,
        "pooled_exact_sufficient_statistics": list(ADDITIVE_COUNT_FIELDS),
        "reported_pixel_metrics": ["pixel_precision", "pixel_recall", "pixel_f1"],
        "fold_metrics_arithmetic_mean_used": False,
        **OFFICIAL_FALSE_FLAGS,
    }


__all__ = [
    "ADDITIVE_COUNT_FIELDS",
    "BGCR_CANDIDATE_KIND",
    "BGCR_CANDIDATE_NAME",
    "DATASET",
    "EVALUATE_EVERY",
    "EXACT_COUNT_FIELDS",
    "FOLD_ASSIGNMENT_ALGORITHM",
    "FOLD_ASSIGNMENT_SHA256",
    "FOLD_COUNT",
    "FOLD_MANIFEST_SCHEMA",
    "FOLD_METRIC_SCHEMA",
    "FOLD_SEED",
    "FOLD_SIZES",
    "FOLD_TIE_ORDER",
    "IRSTDBGCRRunContractError",
    "OFFICIAL_FALSE_FLAGS",
    "OOF_EVALUATION_EPOCHS",
    "PERFORMANCE_ACCEPTANCE_MARGIN",
    "PIXEL_CONFUSION_FIELDS",
    "POOLED_METRIC_SCHEMA",
    "POOLED_REFERENCE_METRIC_SCHEMA",
    "PROBABILITY_COMPARISON",
    "PROBABILITY_THRESHOLD",
    "ROLE",
    "SOURCE_SPLIT_MANIFEST_FILE_SHA256",
    "SOURCE_SPLIT_MANIFEST_PATH",
    "SOURCE_SCOPE",
    "STRATUM_TIE_ORDER",
    "TRAIN_EPOCHS",
    "build_frozen_fold_manifest",
    "canonical_json_sha256",
    "fold_metric_binding",
    "irstd_best_miou_role_key",
    "load_train_only_split_manifest",
    "pool_oof_sufficient_statistics",
    "pool_reference_sufficient_statistics",
    "run_contract_manifest",
    "select_oof_epoch",
    "serialize_role_key",
]
