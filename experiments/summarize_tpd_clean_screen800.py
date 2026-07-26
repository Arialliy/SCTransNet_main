#!/usr/bin/env python3
"""Audit and summarize the TPD-Clean-v2 single-seed screening experiment.

The script compares the four fresh candidates with the frozen formal800 SPD
and TPD-v1 references.  It produces descriptive screening evidence only: a
single dataset and model seed cannot establish the paper core, mechanism
necessity, or cross-seed stability.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_VARIANTS = (
    "grouped_keep",
    "tpd_clean_ctx",
    "tpd_clean_sal",
    "tpd_clean_full",
)
REFERENCE_VARIANTS = ("tpd", "spd")
EXPECTED_BUDGETS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
EXPECTED_EPOCHS = 800
EXPECTED_DATASET = "NUDT-SIRST"
EXPECTED_SEED = 42
EXPECTED_SPLIT_SEED = 20260722
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
EXPECTED_VALID_PIXELS = 8_716_288
EXPECTED_TRAIN_COUNT = 530
EXPECTED_CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
    "last.pth.tar": "last_evaluated_epoch",
}
POINT_COUNT_KEYS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
POINT_RATE_KEYS = (
    "pd",
    "tiny_pd",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
)
POINT_KEYS = frozenset(
    (*POINT_COUNT_KEYS, *POINT_RATE_KEYS, "fa", "false_objects_per_image", "val_loss", "threshold")
)
SELECTION_METRIC_KEYS = ("pd", "fa", "tiny_pd", "miou", "val_loss")
CRITICAL_PROTOCOL_ARGUMENTS = (
    "dataset",
    "epochs",
    "batch_size",
    "patch_size",
    "workers",
    "seed",
    "split_seed",
    "val_fraction",
    "eval_every",
    "base_lr",
    "min_lr",
    "warmup_epochs",
    "threshold",
    "match_radius",
    "tiny_area",
    "amp",
    "max_train_images",
    "max_val_images",
)
FROZEN_DECISION = "INCONCLUSIVE_MIXED_TRADEOFF"
FROZEN_MIOU_CHECKPOINT_SHA256 = {
    "spd": "f932198ffa33408c8faa8801580bc8db6a337afa8544770d8c972f1c8bde232a",
    "tpd": "ce75b45494ada10ed3c2f8915a5e9be6223548fbce5e131acbb184d8d67b2676",
}
FROZEN_COMPARISON_FILES = {
    "seed_42_formal800_pd_fp32_4x5090_v1.json": (
        "c692ad664ae238dfa6fdf3ced8fd3e20361a3b68edad01288c5b583aa95f48b4"
    ),
    "COMPLETE.sha256": (
        "585028609eb7ca51c7232daa586cb0c2317c56b292cb424b2b96b11bdd37ab40"
    ),
    "pd_fa_seed_42_formal800_pd_fp32_4x5090_v1.json": (
        "1aea19d02a27cfd88f54d62a668b5a49156eea1a216a1e6b9fbb10fd42c08eae"
    ),
    "pd_fa_seed_42_formal800_pd_fp32_4x5090_v1.COMPLETE.sha256": (
        "20592973f91810fc65851e03ddcd8a66aca2e6c8a91c82ed0449889c7b2f5889"
    ),
    "mainline_decision_seed42.json": (
        "eb8c539c430c45e764935b16b08b1e6fdb4c91eb7d1519eba4c5f6aaf08eccca"
    ),
    "mainline_decision_seed42.COMPLETE.sha256": (
        "dccaac3394d62f4809e046fc9746a6c3466560de0efeda6feb8a9038aff75787"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize TPD-Clean screen800 candidates"
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/results/tpd_clean_screen800_4x5090_v1"
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/results/tpd_pe_formal800_4x5090_v1"
        ),
    )
    parser.add_argument(
        "--candidate-run-name",
        default="seed_42_screen800_pd_fp32_shared4x5090_v1",
    )
    parser.add_argument(
        "--reference-run-name",
        default="seed_42_formal800_pd_fp32_4x5090_v1",
    )
    parser.add_argument(
        "--reference-miou-root",
        type=Path,
        default=None,
        help=(
            "Read-only mirror containing derived SPD/TPD-v1 best-mIoU sweeps. "
            "Defaults to CANDIDATE_ROOT/frozen_reference_miou_runs."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.reference_miou_root is None:
        args.reference_miou_root = (
            args.candidate_root / "frozen_reference_miou_runs"
        )
    if args.output_dir is None:
        args.output_dir = args.candidate_root / EXPECTED_DATASET / "comparison"
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def canonical(value: Any) -> str:
    def json_default(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        scalar = getattr(item, "item", None)
        if callable(scalar):
            return scalar()
        raise TypeError(f"Cannot canonicalize {type(item).__name__}")

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )


def require_equal(actual: Any, expected: Any, message: str) -> None:
    require(canonical(actual) == canonical(expected), message)


def require_close(
    actual: float,
    expected: float,
    message: str,
    *,
    abs_tol: float = 1e-15,
) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=abs_tol),
        message,
    )


def verify_checksum_manifest(path: Path) -> Dict[str, str]:
    require(
        path.is_file() and not path.is_symlink(),
        f"Missing checksum manifest: {path}",
    )
    verified: Dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"{path}:{line_number}: malformed checksum line")
        expected, relative_name = parts
        relative_name = relative_name.lstrip("*")
        relative = Path(relative_name)
        require(
            len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            f"{path}:{line_number}: invalid SHA-256",
        )
        require(
            not relative.is_absolute()
            and relative.parent == Path(".")
            and relative.name not in {"", ".", ".."},
            f"{path}:{line_number}: checksum target must be a local basename",
        )
        target = path.parent / relative.name
        require(
            target.is_file() and not target.is_symlink(),
            f"{path}:{line_number}: missing or linked target {target}",
        )
        actual = file_sha256(target)
        require(
            actual == expected,
            f"{path}:{line_number}: checksum mismatch for {target.name}",
        )
        verified[target.name] = actual
    require(bool(verified), f"{path}: empty checksum manifest")
    return verified


def verify_frozen_comparison(reference_root: Path) -> Dict[str, Any]:
    comparison_dir = reference_root / EXPECTED_DATASET / "comparison"
    actual: Dict[str, str] = {}
    for name, expected in FROZEN_COMPARISON_FILES.items():
        artifact = comparison_dir / name
        require(
            artifact.is_file() and not artifact.is_symlink(),
            f"Missing frozen comparison artifact: {artifact}",
        )
        digest = file_sha256(artifact)
        require(digest == expected, f"Frozen comparison artifact changed: {name}")
        actual[name] = digest
    manifest_members = {
        "COMPLETE.sha256": verify_checksum_manifest(
            comparison_dir / "COMPLETE.sha256"
        ),
        "pd_fa_seed_42_formal800_pd_fp32_4x5090_v1.COMPLETE.sha256": (
            verify_checksum_manifest(
                comparison_dir
                / "pd_fa_seed_42_formal800_pd_fp32_4x5090_v1.COMPLETE.sha256"
            )
        ),
        "mainline_decision_seed42.COMPLETE.sha256": verify_checksum_manifest(
            comparison_dir / "mainline_decision_seed42.COMPLETE.sha256"
        ),
    }
    decision = load_json(comparison_dir / "mainline_decision_seed42.json")
    screening = decision.get("screening_decision")
    require(isinstance(screening, dict), "Frozen decision lacks screening_decision")
    require(
        screening.get("decision") == FROZEN_DECISION,
        "Frozen formal decision changed",
    )
    require(
        decision.get("paper_core_established") is False,
        "Frozen paper_core_established must remain false",
    )
    require(
        decision.get("stability_claim_supported") is False,
        "Frozen stability_claim_supported must remain false",
    )
    require(
        decision.get("official_test_accessed") is False,
        "Frozen decision official-test flag mismatch",
    )
    return {
        "comparison_directory": str(comparison_dir.resolve()),
        "pinned_sha256": actual,
        "verified_manifest_members": manifest_members,
        "formal_decision": FROZEN_DECISION,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def verify_source_lock(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    require(
        payload.get("schema") == "sctransnet_tpd_clean_screen800_source_lock_v1",
        "Unexpected TPD-Clean source-lock schema",
    )
    source_sha256 = payload.get("source_sha256")
    require(isinstance(source_sha256, dict) and source_sha256, "Empty source lock")
    verified: Dict[str, str] = {}
    for relative_name, expected in source_sha256.items():
        relative = Path(str(relative_name))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"Invalid source-lock path: {relative_name}",
        )
        source = REPO_ROOT / relative
        require(
            source.is_file() and not source.is_symlink(),
            f"Missing or linked locked source: {source}",
        )
        actual = file_sha256(source)
        require(actual == expected, f"Locked source changed: {relative_name}")
        verified[str(relative)] = actual
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "verified_source_sha256": verified,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"Non-finite value at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{location}.{key}")


def load_metrics(path: Path, variant: str) -> List[Dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"Missing metrics: {path}")
    raw = path.read_text(encoding="utf-8")
    require(raw.endswith("\n"), f"{variant}: metrics file lacks final newline")
    lines = raw.splitlines()
    require(
        len(lines) == EXPECTED_EPOCHS,
        f"{variant}: expected {EXPECTED_EPOCHS} metric rows, found {len(lines)}",
    )
    events: List[Dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        event = json.loads(line, object_pairs_hook=reject_duplicate_keys)
        require(isinstance(event, dict), f"{variant}: metrics row {index} is not an object")
        require(event.get("epoch") == index, f"{variant}: non-contiguous epoch {index}")
        require(event.get("variant") == variant, f"{variant}: metric variant mismatch")
        require(
            event.get("processed_train_samples") == EXPECTED_TRAIN_COUNT,
            f"{variant}: processed-train count mismatch at epoch {index}",
        )
        for key in SELECTION_METRIC_KEYS:
            require(key in event, f"{variant}: epoch {index} lacks validation metric {key}")
        require(
            isinstance(event.get("new_best_pd"), bool)
            and isinstance(event.get("new_best_miou"), bool),
            f"{variant}: epoch {index} lacks checkpoint selection flags",
        )
        require_finite(event, f"{variant}.metrics[{index}]")
        events.append(event)
    return events


def point_key(point: Mapping[str, Any] | None) -> Tuple[float, ...]:
    """Frozen cross-method order: availability, Pd, Fa, tiny-Pd, mIoU."""
    if point is None:
        return (0.0, -1.0, -float("inf"), -1.0, -1.0)
    tiny_pd = point.get("tiny_pd")
    return (
        1.0,
        float(point["pd"]),
        -float(point["fa"]),
        float(tiny_pd) if tiny_pd is not None else -1.0,
        float(point["miou"]),
    )


def compare_points(
    candidate: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    candidate_key = point_key(candidate)
    reference_key = point_key(reference)
    if candidate_key > reference_key:
        outcome = "candidate_better"
    elif candidate_key < reference_key:
        outcome = "reference_better"
    else:
        outcome = "tie"
    decisive = "tie"
    labels = ("availability", "pd", "fa", "tiny_pd", "miou")
    for label, candidate_value, reference_value in zip(
        labels, candidate_key, reference_key
    ):
        if candidate_value != reference_value:
            decisive = label
            break
    return {
        "outcome": outcome,
        "decisive_metric": decisive,
        # A missing operating point is already represented canonically as
        # null in the sweep.  Keep its diagnostic key null as well instead of
        # leaking the internal -Inf ordering sentinel into auditable JSON.
        "candidate_key": list(candidate_key) if candidate is not None else None,
        "reference_key": list(reference_key) if reference is not None else None,
    }


def operating_point_key(point: Mapping[str, Any]) -> Tuple[float, ...]:
    """Within-curve selection order; threshold proximity is the final tie-break."""
    return (
        float(point["pd"]),
        -float(point["fa"]),
        float(point["tiny_pd"]),
        float(point["miou"]),
        -abs(float(point["threshold"]) - 0.5),
    )


def recompute_budget_point(
    points: Sequence[Dict[str, Any]], budget: float
) -> Dict[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= float(budget)]
    return max(feasible, key=operating_point_key) if feasible else None


def pd_selection_key(metrics: Mapping[str, Any]) -> Tuple[float, ...]:
    return (
        float(metrics["pd"]),
        -float(metrics["fa"]),
        float(metrics["tiny_pd"]),
        float(metrics["miou"]),
        -float(metrics["val_loss"]),
    )


def miou_selection_key(metrics: Mapping[str, Any]) -> Tuple[float, ...]:
    return (
        float(metrics["miou"]),
        float(metrics["pd"]),
        -float(metrics["fa"]),
        float(metrics["tiny_pd"]),
        -float(metrics["val_loss"]),
    )


def pd_fa_dominates(
    left: Mapping[str, Any], right: Mapping[str, Any], *, strict: bool = True
) -> bool:
    no_worse = (
        float(left["pd"]) >= float(right["pd"])
        and float(left["fa"]) <= float(right["fa"])
    )
    if not no_worse:
        return False
    if not strict:
        return True
    return (
        float(left["pd"]) > float(right["pd"])
        or float(left["fa"]) < float(right["fa"])
    )


def unique_pd_fa_coordinates(
    named_points: Iterable[Tuple[str, Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    coordinates: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for owner, point in named_points:
        coordinate = (float(point["pd"]), float(point["fa"]))
        item = coordinates.setdefault(
            coordinate,
            {
                "pd": coordinate[0],
                "fa": coordinate[1],
                "matched_target_count": int(point["matched_target_count"]),
                "target_count": int(point["target_count"]),
                "owners": [],
            },
        )
        if owner not in item["owners"]:
            item["owners"].append(owner)
    return list(coordinates.values())


def pareto_frontier(
    named_points: Iterable[Tuple[str, Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    coordinates = unique_pd_fa_coordinates(named_points)
    frontier = [
        point
        for point in coordinates
        if not any(
            pd_fa_dominates(other, point)
            for other in coordinates
            if other is not point
        )
    ]
    return sorted(frontier, key=lambda point: (point["fa"], -point["pd"]))


def validate_point(point: Mapping[str, Any], location: str) -> None:
    require(
        set(point) == POINT_KEYS,
        f"{location}: point fields differ; missing={sorted(POINT_KEYS - set(point))}, "
        f"extra={sorted(set(point) - POINT_KEYS)}",
    )
    require_finite(dict(point), location)
    for key in POINT_COUNT_KEYS:
        value = point[key]
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{location}.{key}: expected non-negative integer",
        )
    threshold = float(point["threshold"])
    require(0.0 < threshold < 1.0, f"{location}: threshold must lie in (0,1)")
    for key in POINT_RATE_KEYS:
        require(
            0.0 <= float(point[key]) <= 1.0,
            f"{location}.{key}: expected value in [0,1]",
        )
    require(float(point["fa"]) >= 0.0, f"{location}: negative Fa")
    require(
        float(point["false_objects_per_image"]) >= 0.0,
        f"{location}: negative false_objects_per_image",
    )
    require(float(point["val_loss"]) >= 0.0, f"{location}: negative val_loss")
    require(
        int(point["target_count"]) == EXPECTED_TARGET_COUNT,
        f"{location}: target-count mismatch",
    )
    require(
        int(point["tiny_target_count"]) == EXPECTED_TINY_TARGET_COUNT,
        f"{location}: tiny-target-count mismatch",
    )
    require(
        int(point["valid_pixel_count"]) == EXPECTED_VALID_PIXELS,
        f"{location}: valid-pixel-count mismatch",
    )
    require(
        int(point["matched_target_count"]) <= int(point["target_count"]),
        f"{location}: matched targets exceed targets",
    )
    require(
        int(point["matched_tiny_target_count"]) <= int(point["tiny_target_count"]),
        f"{location}: matched tiny targets exceed tiny targets",
    )
    require(
        int(point["matched_tiny_target_count"]) <= int(point["matched_target_count"]),
        f"{location}: matched tiny targets exceed all matched targets",
    )
    require(
        int(point["predicted_object_count"])
        == int(point["matched_target_count"])
        + int(point["unmatched_predicted_object_count"]),
        f"{location}: predicted objects do not equal matched plus unmatched",
    )
    require_close(
        float(point["pd"]),
        int(point["matched_target_count"]) / EXPECTED_TARGET_COUNT,
        f"{location}: Pd/count identity mismatch",
    )
    require_close(
        float(point["tiny_pd"]),
        int(point["matched_tiny_target_count"]) / EXPECTED_TINY_TARGET_COUNT,
        f"{location}: tiny-Pd/count identity mismatch",
    )
    require_close(
        float(point["false_objects_per_image"]),
        int(point["unmatched_predicted_object_count"]) / EXPECTED_VALIDATION_COUNT,
        f"{location}: false-objects/image identity mismatch",
    )


def validate_sweep(
    path: Path,
    *,
    variant: str,
    split_sha256: str,
    expected_role: str,
    run_dir: Path | None = None,
    expected_checkpoint_name: str | None = None,
    expected_checkpoint_epoch: int | None = None,
    expected_checkpoint_metrics: Mapping[str, Any] | None = None,
    strict_artifacts: bool = False,
) -> Dict[str, Any]:
    sweep = load_json(path)
    require_finite(sweep, str(path))
    require(sweep.get("variant") == variant, f"{variant}: sweep variant mismatch")
    require(sweep.get("dataset") == EXPECTED_DATASET, f"{variant}: sweep dataset mismatch")
    require(sweep.get("seed") == EXPECTED_SEED, f"{variant}: sweep seed mismatch")
    require(
        sweep.get("split_seed") == EXPECTED_SPLIT_SEED,
        f"{variant}: sweep split-seed mismatch",
    )
    require(
        sweep.get("validation_count") == EXPECTED_VALIDATION_COUNT,
        f"{variant}: sweep validation-count mismatch",
    )
    require(
        sweep.get("validation_split_sha256") == split_sha256,
        f"{variant}: sweep split hash mismatch",
    )
    require(
        sweep.get("checkpoint_role") == expected_role,
        f"{variant}: sweep checkpoint-role mismatch",
    )
    require(
        sweep.get("official_test_accessed") is False,
        f"{variant}: sweep official-test flag mismatch",
    )
    checkpoint_path = Path(str(sweep["checkpoint"]))
    require(
        checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
        f"{variant}: missing or linked swept checkpoint",
    )
    if run_dir is not None:
        require(
            sweep.get("run_directory") == str(run_dir.resolve()),
            f"{variant}: sweep run-directory mismatch",
        )
    if run_dir is not None and expected_checkpoint_name is not None:
        expected_checkpoint = (run_dir / expected_checkpoint_name).resolve()
        require(
            checkpoint_path.resolve() == expected_checkpoint,
            f"{variant}: swept checkpoint path mismatch",
        )
    require(
        file_sha256(checkpoint_path) == sweep.get("checkpoint_sha256"),
        f"{variant}: swept checkpoint digest mismatch",
    )
    checkpoint_epoch = sweep.get("checkpoint_epoch")
    require(
        isinstance(checkpoint_epoch, int)
        and not isinstance(checkpoint_epoch, bool)
        and 1 <= checkpoint_epoch <= EXPECTED_EPOCHS,
        f"{variant}: invalid sweep checkpoint epoch",
    )
    if expected_checkpoint_epoch is not None:
        require(
            checkpoint_epoch == expected_checkpoint_epoch,
            f"{variant}: sweep checkpoint epoch mismatch",
        )

    points = sweep.get("points")
    require(isinstance(points, list) and points, f"{variant}: empty sweep points")
    for index, point in enumerate(points):
        require(isinstance(point, dict), f"{variant}: sweep point {index} is invalid")
        validate_point(point, f"{variant}.points[{index}]")
    thresholds = [float(point["threshold"]) for point in points]
    require(len(thresholds) == len(set(thresholds)), f"{variant}: duplicate thresholds")
    require(thresholds == sorted(thresholds), f"{variant}: thresholds are not sorted")
    provenance = sweep.get("threshold_provenance")
    if strict_artifacts:
        require(isinstance(provenance, dict), f"{variant}: missing threshold provenance")
        require(
            provenance.get("total_unique_threshold_count") == len(points),
            f"{variant}: threshold provenance count mismatch",
        )
    for count_key in ("target_count", "tiny_target_count", "valid_pixel_count"):
        invariant = points[0][count_key]
        require(
            all(point[count_key] == invariant for point in points),
            f"{variant}: non-invariant {count_key} across sweep",
        )

    half_matches = [point for point in points if float(point["threshold"]) == 0.5]
    require(len(half_matches) == 1, f"{variant}: expected one exact threshold 0.5")
    fixed = sweep.get("fixed_threshold_0_5")
    require(isinstance(fixed, dict), f"{variant}: missing fixed-threshold point")
    validate_point(fixed, f"{variant}.fixed_0_5")
    require_equal(fixed, half_matches[0], f"{variant}: fixed 0.5 point mismatch")

    configuration = sweep.get("threshold_configuration")
    if strict_artifacts:
        require(isinstance(configuration, dict), f"{variant}: missing threshold configuration")
        raw_budgets = configuration.get("fa_budgets")
        require(isinstance(raw_budgets, list), f"{variant}: missing configured budgets")
        configured_budget_keys = tuple(f"{float(value):.10g}" for value in raw_budgets)
        require(
            configured_budget_keys == EXPECTED_BUDGETS,
            f"{variant}: configured Fa budgets mismatch",
        )
    budgets = sweep.get("best_points_under_fa_budget")
    require(isinstance(budgets, dict), f"{variant}: missing budget points")
    require(set(budgets) == set(EXPECTED_BUDGETS), f"{variant}: budget set mismatch")
    for budget in EXPECTED_BUDGETS:
        point = budgets[budget]
        if point is not None:
            require(isinstance(point, dict), f"{variant}: invalid budget point {budget}")
            validate_point(point, f"{variant}.budget[{budget}]")
        recomputed = recompute_budget_point(points, float(budget))
        require_equal(
            point,
            recomputed,
            f"{variant}: budget {budget} optimal point mismatch after lexicographic recomputation",
        )

    checkpoint_metrics = sweep.get("checkpoint_validation_metrics")
    if strict_artifacts:
        require(
            isinstance(checkpoint_metrics, dict),
            f"{variant}: missing checkpoint validation metrics",
        )
        validate_point(
            {**checkpoint_metrics, "threshold": 0.5},
            f"{variant}.checkpoint_validation_metrics",
        )
        require_equal(
            checkpoint_metrics,
            {key: value for key, value in fixed.items() if key != "threshold"},
            f"{variant}: fixed 0.5 metrics differ from checkpoint metrics",
        )
        if expected_checkpoint_metrics is not None:
            require_equal(
                checkpoint_metrics,
                dict(expected_checkpoint_metrics),
                f"{variant}: sweep metrics differ from audited checkpoint metrics",
            )
        audit = sweep.get("audit")
        require(isinstance(audit, dict), f"{variant}: missing sweep audit")
        require(audit.get("expected_epochs") == EXPECTED_EPOCHS, f"{variant}: audit epoch mismatch")
        require(audit.get("metrics_event_count") == EXPECTED_EPOCHS, f"{variant}: audit row mismatch")
        require(audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS], f"{variant}: audit range mismatch")
        require(audit.get("summary_status") == "complete", f"{variant}: audit source incomplete")
        require(
            audit.get("selection_source") == "internal_validation_only",
            f"{variant}: audit selection source mismatch",
        )
        flags = audit.get("integrity_checks_passed")
        require(
            isinstance(flags, dict) and flags and all(value is True for value in flags.values()),
            f"{variant}: one or more sweep integrity checks did not pass",
        )
        require(run_dir is not None, f"{variant}: strict sweep audit requires run_dir")
        artifact_sha256 = audit.get("artifact_sha256")
        require(isinstance(artifact_sha256, dict), f"{variant}: missing sweep artifact hashes")
        expected_artifacts = {
            "protocol.json": run_dir / "protocol.json",
            "split.json": run_dir / "split.json",
            "summary.json": run_dir / "summary.json",
            "metrics.jsonl": run_dir / "metrics.jsonl",
            "checkpoint": checkpoint_path,
            "evaluator": REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py",
        }
        for name, artifact in expected_artifacts.items():
            require(
                artifact.is_file() and not artifact.is_symlink(),
                f"{variant}: missing or linked sweep source artifact {artifact}",
            )
            require(
                artifact_sha256.get(name) == file_sha256(artifact),
                f"{variant}: sweep artifact digest mismatch for {name}",
            )
        fixed_audit = sweep.get("fixed_threshold_0_5_checkpoint_audit")
        require(isinstance(fixed_audit, dict), f"{variant}: missing fixed-threshold audit")
        require_close(
            float(fixed_audit.get("max_abs_non_strict_numeric_delta")),
            0.0,
            f"{variant}: fixed-threshold numeric audit mismatch",
        )
    return sweep


def identifier_sha256(identifiers: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(identifier) for identifier in identifiers))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def learning_rate_for_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    decay_epochs = total_epochs - warmup_epochs
    if decay_epochs <= 0:
        return base_lr
    progress = (epoch - warmup_epochs) / decay_epochs
    return min_lr + 0.5 * (base_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def event_validation_metrics(event: Mapping[str, Any], location: str) -> Dict[str, Any]:
    keys = POINT_KEYS - {"threshold"}
    require(keys <= set(event), f"{location}: incomplete validation metrics")
    metrics = {key: event[key] for key in keys}
    validate_point({**metrics, "threshold": 0.5}, location)
    return metrics


def validate_selection_history(
    events: Sequence[Mapping[str, Any]], variant: str
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    best_pd_event: Mapping[str, Any] | None = None
    best_miou_event: Mapping[str, Any] | None = None
    best_pd_key: Tuple[float, ...] | None = None
    best_miou_key: Tuple[float, ...] | None = None
    for event in events:
        current_pd_key = pd_selection_key(event)
        current_miou_key = miou_selection_key(event)
        expected_new_pd = best_pd_key is None or current_pd_key > best_pd_key
        expected_new_miou = best_miou_key is None or current_miou_key > best_miou_key
        require(
            event.get("new_best_pd") is expected_new_pd,
            f"{variant}: incorrect new_best_pd flag at epoch {event['epoch']}",
        )
        require(
            event.get("new_best_miou") is expected_new_miou,
            f"{variant}: incorrect new_best_miou flag at epoch {event['epoch']}",
        )
        if expected_new_pd:
            best_pd_key = current_pd_key
            best_pd_event = event
        if expected_new_miou:
            best_miou_key = current_miou_key
            best_miou_event = event
    require(best_pd_event is not None, f"{variant}: no Pd-primary event")
    require(best_miou_event is not None, f"{variant}: no mIoU-secondary event")
    return best_pd_event, best_miou_event


def validate_checkpoint(
    path: Path,
    *,
    variant: str,
    expected_epoch: int,
    expected_role: str,
    expected_metrics: Mapping[str, Any],
    expected_model: Mapping[str, Any],
    expected_split_hashes: Mapping[str, Any],
) -> Dict[str, Any]:
    require(
        path.is_file() and not path.is_symlink(),
        f"{variant}: missing or linked checkpoint {path}",
    )
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), f"{variant}: checkpoint is not an object")
    require(checkpoint.get("variant") == variant, f"{variant}: checkpoint variant mismatch")
    require(checkpoint.get("dataset") == EXPECTED_DATASET, f"{variant}: checkpoint dataset mismatch")
    require(checkpoint.get("seed") == EXPECTED_SEED, f"{variant}: checkpoint seed mismatch")
    require(
        checkpoint.get("split_seed") == EXPECTED_SPLIT_SEED,
        f"{variant}: checkpoint split seed mismatch",
    )
    require(checkpoint.get("epoch") == expected_epoch, f"{variant}: checkpoint epoch mismatch")
    require(checkpoint.get("checkpoint_role") == expected_role, f"{variant}: checkpoint role mismatch")
    require(
        checkpoint.get("selection_source") == "internal_validation_only",
        f"{variant}: checkpoint selection source mismatch",
    )
    require(
        checkpoint.get("official_test_accessed") is False,
        f"{variant}: checkpoint official-test flag mismatch",
    )
    require_equal(
        checkpoint.get("validation_metrics"),
        dict(expected_metrics),
        f"{variant}: checkpoint validation metrics mismatch",
    )
    require_equal(
        checkpoint.get("model_metadata"),
        dict(expected_model),
        f"{variant}: checkpoint model metadata mismatch",
    )
    require_equal(
        checkpoint.get("split_hashes"),
        dict(expected_split_hashes),
        f"{variant}: checkpoint split hashes mismatch",
    )
    state_dict = checkpoint.get("state_dict")
    require(isinstance(state_dict, dict), f"{variant}: checkpoint lacks state_dict")
    if variant in CANDIDATE_VARIANTS:
        from experiments.train_tpd_clean_v2 import build_clean_model

        model, rebuilt_metadata = build_clean_model(variant, EXPECTED_SEED)
    else:
        from experiments.train_tpd_pilot import build_model

        model, rebuilt_metadata = build_model(variant, EXPECTED_SEED)
    require_equal(
        rebuilt_metadata,
        dict(expected_model),
        f"{variant}: rebuilt model metadata mismatch",
    )
    model.load_state_dict(state_dict, strict=True)
    del model, checkpoint
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "epoch": expected_epoch,
        "role": expected_role,
        "state_dict_strict_load": True,
    }


def validate_candidate_completion_log(root: Path, variant: str) -> Dict[str, Any]:
    log_path = root / "logs" / f"{variant}.log"
    require(
        log_path.is_file() and not log_path.is_symlink(),
        f"{variant}: missing completion log",
    )
    lines = log_path.read_text(encoding="utf-8", errors="strict").splitlines()
    expected = f"TPDCLEAN_COMPLETE variant={variant} "
    complete_lines = [line for line in lines if line.startswith(expected)]
    require(
        len(complete_lines) == 1,
        f"{variant}: expected exactly one TPDCLEAN_COMPLETE line",
    )
    forbidden_fragments = (
        "TPDCLEAN_ABORT",
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "/home/md0/ly",
        "SCTransNet copy",
        "/img_idx/test_",
        "/NUDT-SIRST/test",
    )
    for fragment in forbidden_fragments:
        require(
            not any(fragment in line for line in lines),
            f"{variant}: forbidden or failed-run log fragment: {fragment}",
        )
    require("epochs=800" in complete_lines[0], f"{variant}: completion epoch mismatch")
    return {
        "path": str(log_path.resolve()),
        "sha256": file_sha256(log_path),
        "completion_line": complete_lines[0],
    }


def validate_run(
    root: Path,
    run_name: str,
    variant: str,
    *,
    require_miou_sweep: bool,
) -> Dict[str, Any]:
    run_dir = (root / EXPECTED_DATASET / variant / run_name).resolve()
    require(run_dir.is_dir() and not run_dir.is_symlink(), f"Missing run: {run_dir}")
    summary = load_json(run_dir / "summary.json")
    protocol = load_json(run_dir / "protocol.json")
    split = load_json(run_dir / "split.json")
    metrics = load_metrics(run_dir / "metrics.jsonl", variant)
    require(summary.get("status") == "complete", f"{variant}: run not complete")
    require(summary.get("variant") == variant, f"{variant}: summary variant mismatch")
    require(summary.get("dataset") == EXPECTED_DATASET, f"{variant}: summary dataset mismatch")
    require(summary.get("seed") == EXPECTED_SEED, f"{variant}: summary seed mismatch")
    require(
        summary.get("selection_source") == "internal_validation_only",
        f"{variant}: selection-source mismatch",
    )
    require(
        summary.get("official_test_accessed") is False,
        f"{variant}: summary official-test flag mismatch",
    )
    arguments = protocol.get("arguments")
    require(isinstance(arguments, dict), f"{variant}: missing protocol arguments")
    require(arguments.get("variant") == variant, f"{variant}: protocol variant mismatch")
    for key in CRITICAL_PROTOCOL_ARGUMENTS:
        require(key in arguments, f"{variant}: missing protocol argument {key}")
    require(arguments.get("epochs") == EXPECTED_EPOCHS, f"{variant}: epoch protocol mismatch")
    require(arguments.get("seed") == EXPECTED_SEED, f"{variant}: seed protocol mismatch")
    require(
        arguments.get("split_seed") == EXPECTED_SPLIT_SEED,
        f"{variant}: split-seed protocol mismatch",
    )
    require(arguments.get("eval_every") == 1, f"{variant}: eval_every must be one")
    require(arguments.get("amp") is False, f"{variant}: FP32 protocol mismatch")
    require(float(arguments.get("threshold")) == 0.5, f"{variant}: threshold mismatch")
    require(
        protocol.get("run_directory") == str(run_dir),
        f"{variant}: protocol run-directory mismatch",
    )
    require(protocol.get("official_test_accessed") is False, f"{variant}: protocol official-test flag mismatch")

    require(split.get("dataset") == EXPECTED_DATASET, f"{variant}: split dataset mismatch")
    require(split.get("split_seed") == EXPECTED_SPLIT_SEED, f"{variant}: split seed mismatch")
    require(split.get("official_test_accessed") is False, f"{variant}: split official-test flag mismatch")
    require(
        split.get("full_internal_train_count") == EXPECTED_TRAIN_COUNT
        and split.get("full_internal_val_count") == EXPECTED_VALIDATION_COUNT
        and split.get("used_train_count") == EXPECTED_TRAIN_COUNT
        and split.get("used_val_count") == EXPECTED_VALIDATION_COUNT,
        f"{variant}: split-count mismatch",
    )
    list_hash_fields = {
        "full_internal_train_ids": "full_internal_train_sha256",
        "full_internal_val_ids": "full_internal_val_sha256",
        "used_train_ids": "used_train_sha256",
        "used_val_ids": "used_val_sha256",
    }
    split_hashes = split.get("hashes")
    require(isinstance(split_hashes, dict), f"{variant}: missing split hashes")
    for list_name, hash_name in list_hash_fields.items():
        identifiers = split.get(list_name)
        require(isinstance(identifiers, list), f"{variant}: missing split list {list_name}")
        require(len(identifiers) == len(set(identifiers)), f"{variant}: duplicate split IDs")
        require(
            identifier_sha256(identifiers) == split_hashes.get(hash_name),
            f"{variant}: recomputed split hash mismatch for {list_name}",
        )
    require_equal(
        split.get("full_internal_train_ids"),
        split.get("used_train_ids"),
        f"{variant}: unexpected training subset",
    )
    require_equal(
        split.get("full_internal_val_ids"),
        split.get("used_val_ids"),
        f"{variant}: unexpected validation subset",
    )
    validation_sha256 = str(split_hashes["used_val_sha256"])
    require_equal(summary.get("split_hashes"), split_hashes, f"{variant}: summary/split hash mismatch")
    require_finite(summary, f"{variant}.summary")
    require_finite(protocol, f"{variant}.protocol")
    require_finite(split, f"{variant}.split")

    for event in metrics:
        require(float(event["train_loss"]) >= 0.0, f"{variant}: negative train loss")
        require(float(event["epoch_seconds"]) > 0.0, f"{variant}: non-positive epoch time")
        expected_lr = learning_rate_for_epoch(
            int(event["epoch"]),
            int(arguments["epochs"]),
            float(arguments["base_lr"]),
            float(arguments["min_lr"]),
            int(arguments["warmup_epochs"]),
        )
        require_close(
            float(event["learning_rate"]),
            expected_lr,
            f"{variant}: learning-rate mismatch at epoch {event['epoch']}",
        )
        event_validation_metrics(event, f"{variant}.metrics[{event['epoch']}]")
    best_pd_event, best_miou_event = validate_selection_history(metrics, variant)
    best_pd_epoch = int(best_pd_event["epoch"])
    best_miou_epoch = int(best_miou_event["epoch"])
    best_pd_metrics = event_validation_metrics(best_pd_event, f"{variant}.best_pd")
    best_miou_metrics = event_validation_metrics(best_miou_event, f"{variant}.best_miou")
    last_metrics = event_validation_metrics(metrics[-1], f"{variant}.last")
    require(summary.get("best_epoch") == best_pd_epoch, f"{variant}: legacy best epoch mismatch")
    require(summary.get("best_pd_epoch") == best_pd_epoch, f"{variant}: best-Pd epoch mismatch")
    require(summary.get("best_miou_epoch") == best_miou_epoch, f"{variant}: best-mIoU epoch mismatch")
    require_equal(summary.get("best_validation_metrics"), best_pd_metrics, f"{variant}: best metrics mismatch")
    require_equal(summary.get("best_pd_validation_metrics"), best_pd_metrics, f"{variant}: best-Pd metrics mismatch")
    require_equal(summary.get("best_miou_validation_metrics"), best_miou_metrics, f"{variant}: best-mIoU metrics mismatch")

    model_metadata = summary.get("model")
    require(isinstance(model_metadata, dict), f"{variant}: missing model metadata")
    require_equal(protocol.get("model"), model_metadata, f"{variant}: protocol/summary model mismatch")
    checkpoint_expectations = {
        "best.pth.tar": (best_pd_epoch, best_pd_metrics),
        "best_miou.pth.tar": (best_miou_epoch, best_miou_metrics),
        "last.pth.tar": (EXPECTED_EPOCHS, last_metrics),
    }
    checkpoints: Dict[str, Any] = {}
    summary_paths = {
        "best.pth.tar": summary.get("best_checkpoint"),
        "best_miou.pth.tar": summary.get("best_miou_checkpoint"),
        "last.pth.tar": summary.get("last_checkpoint"),
    }
    for checkpoint_name, (checkpoint_epoch, checkpoint_metrics) in checkpoint_expectations.items():
        checkpoint_path = run_dir / checkpoint_name
        require(
            Path(str(summary_paths[checkpoint_name])).resolve() == checkpoint_path,
            f"{variant}: summary checkpoint path mismatch for {checkpoint_name}",
        )
        checkpoints[checkpoint_name] = validate_checkpoint(
            checkpoint_path,
            variant=variant,
            expected_epoch=checkpoint_epoch,
            expected_role=EXPECTED_CHECKPOINT_ROLES[checkpoint_name],
            expected_metrics=checkpoint_metrics,
            expected_model=model_metadata,
            expected_split_hashes=split_hashes,
        )

    best_sweep_path = run_dir / "pd_fa_sweep_best.pth.json"
    best_sweep = validate_sweep(
        best_sweep_path,
        variant=variant,
        split_sha256=validation_sha256,
        expected_role=EXPECTED_CHECKPOINT_ROLES["best.pth.tar"],
        run_dir=run_dir,
        expected_checkpoint_name="best.pth.tar",
        expected_checkpoint_epoch=best_pd_epoch,
        expected_checkpoint_metrics=best_pd_metrics,
        strict_artifacts=True,
    )
    best_miou_sweep = None
    if require_miou_sweep:
        best_miou_sweep = validate_sweep(
            run_dir / "pd_fa_sweep_best_miou.pth.json",
            variant=variant,
            split_sha256=validation_sha256,
            expected_role=EXPECTED_CHECKPOINT_ROLES["best_miou.pth.tar"],
            run_dir=run_dir,
            expected_checkpoint_name="best_miou.pth.tar",
            expected_checkpoint_epoch=best_miou_epoch,
            expected_checkpoint_metrics=best_miou_metrics,
            strict_artifacts=True,
        )

    completion_log = None
    launch_manifest = None
    if variant in CANDIDATE_VARIANTS:
        completion_log = validate_candidate_completion_log(root, variant)
        launch_path = root / "launch" / f"{variant}.json"
        launch_manifest = load_json(launch_path)
        require(launch_manifest.get("variant") == variant, f"{variant}: launch variant mismatch")
        require(launch_manifest.get("run_directory") == str(run_dir), f"{variant}: launch run mismatch")
        require(launch_manifest.get("training_data_sha256") == "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e", f"{variant}: training data fingerprint mismatch")
        require(launch_manifest.get("policy", {}).get("official_test_accessed") is False, f"{variant}: launch official-test flag mismatch")
        launch_manifest = {
            "path": str(launch_path.resolve()),
            "sha256": file_sha256(launch_path),
            "gpu_uuid": launch_manifest.get("gpu_uuid"),
            "training_data_sha256": launch_manifest.get("training_data_sha256"),
            "source_lock_sha256": launch_manifest.get("source_lock_sha256"),
        }

    artifact_names = [
        "protocol.json",
        "split.json",
        "metrics.jsonl",
        "summary.json",
        "best.pth.tar",
        "best_miou.pth.tar",
        "last.pth.tar",
        "pd_fa_sweep_best.pth.json",
    ]
    if require_miou_sweep:
        artifact_names.append("pd_fa_sweep_best_miou.pth.json")
    return {
        "variant": variant,
        "run_dir": str(run_dir),
        "summary": summary,
        "protocol": protocol,
        "critical_protocol": {key: arguments[key] for key in CRITICAL_PROTOCOL_ARGUMENTS},
        "protocol_contract": {
            "normalization": protocol.get("normalization"),
            "primary_selection_rule": protocol.get("primary_selection_rule"),
            "secondary_selection_rule": protocol.get("secondary_selection_rule"),
            "checkpoint_policy": protocol.get("checkpoint_policy"),
            "loss": protocol.get("loss"),
            "optimizer": protocol.get("optimizer"),
            "lr_schedule": protocol.get("lr_schedule"),
            "metric_notes": protocol.get("metric_notes"),
        },
        "split_hashes": split_hashes,
        "split_sha256": validation_sha256,
        "best_sweep": best_sweep,
        "best_miou_sweep": best_miou_sweep,
        "checkpoints": checkpoints,
        "completion_log": completion_log,
        "launch_manifest": launch_manifest,
        "artifact_sha256": {
            name: file_sha256(run_dir / name) for name in artifact_names
        },
    }


def validate_reference_miou_sweep(
    mirror_root: Path,
    run_name: str,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    variant = str(reference["variant"])
    source_run = Path(str(reference["run_dir"]))
    mirror_run = (mirror_root / EXPECTED_DATASET / variant / run_name).resolve()
    require(
        mirror_run.is_dir() and not mirror_run.is_symlink(),
        f"{variant}: missing derived best-mIoU reference mirror",
    )
    mirrored_sources = (
        "protocol.json",
        "split.json",
        "summary.json",
        "metrics.jsonl",
        "best_miou.pth.tar",
    )
    require(
        file_sha256(source_run / "best_miou.pth.tar")
        == FROZEN_MIOU_CHECKPOINT_SHA256[variant],
        f"{variant}: frozen best-mIoU checkpoint changed",
    )
    mirror_sha256: Dict[str, str] = {}
    for name in mirrored_sources:
        source = source_run / name
        mirror = mirror_run / name
        require(
            mirror.is_file() and not mirror.is_symlink(),
            f"{variant}: missing or linked derived reference artifact {mirror}",
        )
        source_digest = file_sha256(source)
        mirror_digest = file_sha256(mirror)
        require(
            mirror_digest == source_digest,
            f"{variant}: derived reference mirror differs for {name}",
        )
        mirror_sha256[name] = mirror_digest
    sweep_path = mirror_run / "pd_fa_sweep_best_miou.pth.json"
    sweep = validate_sweep(
        sweep_path,
        variant=variant,
        split_sha256=str(reference["split_sha256"]),
        expected_role=EXPECTED_CHECKPOINT_ROLES["best_miou.pth.tar"],
        run_dir=mirror_run,
        expected_checkpoint_name="best_miou.pth.tar",
        expected_checkpoint_epoch=int(reference["summary"]["best_miou_epoch"]),
        expected_checkpoint_metrics=reference["summary"]["best_miou_validation_metrics"],
        strict_artifacts=True,
    )
    reference["best_miou_sweep"] = sweep
    reference["derived_best_miou_reference"] = {
        "mirror_run_directory": str(mirror_run),
        "source_run_directory": str(source_run),
        "mirrored_source_sha256": mirror_sha256,
        "sweep_path": str(sweep_path),
        "sweep_sha256": file_sha256(sweep_path),
        "semantics": (
            "post-formal auxiliary sweep derived from the frozen best-mIoU "
            "checkpoint; it does not rewrite the formal800 decision"
        ),
    }
    return reference


def summarize_checkpoint_role(
    candidate_variant: str,
    candidate_sweep: Mapping[str, Any],
    references: Mapping[str, Dict[str, Any]],
    sweep_key: str,
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {"fixed_threshold_0_5": {}, "budgets": {}}
    for reference_name, reference in references.items():
        comparisons["fixed_threshold_0_5"][reference_name] = compare_points(
            candidate_sweep["fixed_threshold_0_5"],
            reference[sweep_key]["fixed_threshold_0_5"],
        )
    candidate_better_both = 0
    candidate_not_worse_both = 0
    reference_union_covers = 0
    noncovered_budget_keys: List[str] = []
    for budget in EXPECTED_BUDGETS:
        candidate_point = candidate_sweep["best_points_under_fa_budget"][budget]
        budget_result: Dict[str, Any] = {}
        outcomes: List[str] = []
        for reference_name, reference in references.items():
            result = compare_points(
                candidate_point,
                reference[sweep_key]["best_points_under_fa_budget"][budget],
            )
            budget_result[reference_name] = result
            outcomes.append(result["outcome"])
        comparisons["budgets"][budget] = budget_result
        if all(outcome == "candidate_better" for outcome in outcomes):
            candidate_better_both += 1
            if candidate_point is not None and float(candidate_point["pd"]) > 0.0:
                noncovered_budget_keys.append(budget)
        if all(outcome != "reference_better" for outcome in outcomes):
            candidate_not_worse_both += 1
        if not all(outcome == "candidate_better" for outcome in outcomes):
            reference_union_covers += 1

    reference_points = [
        (reference_name, point)
        for reference_name, reference in references.items()
        for point in reference[sweep_key]["points"]
    ]
    candidate_points = [
        (candidate_variant, point) for point in candidate_sweep["points"]
    ]
    joint_frontier = pareto_frontier([*reference_points, *candidate_points])
    unique_candidate_frontier = [
        point
        for point in joint_frontier
        if candidate_variant in point["owners"]
        and not any(owner in references for owner in point["owners"])
    ]
    if unique_candidate_frontier and candidate_better_both:
        evidence_class = "UNIQUE_SCREENING_SIGNAL"
    elif reference_union_covers == len(EXPECTED_BUDGETS) and not unique_candidate_frontier:
        evidence_class = "COVERED_BY_FROZEN_REFERENCES"
    else:
        evidence_class = "MIXED_SCREENING_SIGNAL"

    selected_point_groups: Dict[str, List[str]] = {}
    for budget in EXPECTED_BUDGETS:
        point = candidate_sweep["best_points_under_fa_budget"][budget]
        key = canonical(point)
        selected_point_groups.setdefault(key, []).append(budget)
    return {
        "fixed_threshold_0_5": candidate_sweep["fixed_threshold_0_5"],
        "budget_points": candidate_sweep["best_points_under_fa_budget"],
        "comparisons": comparisons,
        "candidate_better_than_both_budget_count": candidate_better_both,
        "candidate_not_worse_than_both_budget_count": candidate_not_worse_both,
        "reference_union_covers_budget_count": reference_union_covers,
        "adds_noncovered_budget_keys": noncovered_budget_keys,
        "joint_reference_candidate_pareto_frontier": joint_frontier,
        "unique_candidate_pareto_points": unique_candidate_frontier,
        "unique_selected_point_count": len(selected_point_groups),
        "budgets_grouped_by_identical_point": list(selected_point_groups.values()),
        "evidence_class": evidence_class,
    }


def summarize_candidate(
    candidate: Dict[str, Any],
    references: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    best = summarize_checkpoint_role(
        candidate["variant"], candidate["best_sweep"], references, "best_sweep"
    )
    best_miou = summarize_checkpoint_role(
        candidate["variant"],
        candidate["best_miou_sweep"],
        references,
        "best_miou_sweep",
    )
    return {
        "variant": candidate["variant"],
        "shallow_embedding_parameters": candidate["summary"]["model"][
            "shallow_embedding_parameters"
        ],
        "total_parameters": candidate["summary"]["model"]["total_parameters"],
        "best_pd_epoch": candidate["summary"]["best_pd_epoch"],
        "best_miou_epoch": candidate["summary"]["best_miou_epoch"],
        "fixed_threshold_0_5": best["fixed_threshold_0_5"],
        "best_miou_checkpoint_metrics": candidate["summary"][
            "best_miou_validation_metrics"
        ],
        "best_miou_fixed_threshold_0_5": best_miou["fixed_threshold_0_5"],
        "best_miou_budget_points": best_miou["budget_points"],
        "budget_points": best["budget_points"],
        "comparisons": best["comparisons"],
        "candidate_better_than_both_budget_count": best[
            "candidate_better_than_both_budget_count"
        ],
        "candidate_not_worse_than_both_budget_count": best[
            "candidate_not_worse_than_both_budget_count"
        ],
        "reference_union_covers_budget_count": best[
            "reference_union_covers_budget_count"
        ],
        "joint_reference_candidate_pareto_frontier": best[
            "joint_reference_candidate_pareto_frontier"
        ],
        "unique_candidate_pareto_points": best["unique_candidate_pareto_points"],
        "evidence_class": best["evidence_class"],
        "checkpoint_role_results": {
            "best_pd_primary": best,
            "best_miou_secondary": best_miou,
        },
    }


def point_delta(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    if left is None or right is None:
        return None
    return {
        "matched_target_count": int(left["matched_target_count"])
        - int(right["matched_target_count"]),
        "pd": float(left["pd"]) - float(right["pd"]),
        "fa": float(left["fa"]) - float(right["fa"]),
        "miou": float(left["miou"]) - float(right["miou"]),
        "matched_tiny_target_count": int(left["matched_tiny_target_count"])
        - int(right["matched_tiny_target_count"]),
        "false_objects_per_image": float(left["false_objects_per_image"])
        - float(right["false_objects_per_image"]),
    }


def compare_sweep_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    fixed_left = left["fixed_threshold_0_5"]
    fixed_right = right["fixed_threshold_0_5"]
    budgets: Dict[str, Any] = {}
    for budget in EXPECTED_BUDGETS:
        left_point = left["best_points_under_fa_budget"][budget]
        right_point = right["best_points_under_fa_budget"][budget]
        budgets[budget] = {
            "comparison": compare_points(left_point, right_point),
            "delta_left_minus_right": point_delta(left_point, right_point),
        }
    return {
        "fixed_threshold_0_5": {
            "comparison": compare_points(fixed_left, fixed_right),
            "delta_left_minus_right": point_delta(fixed_left, fixed_right),
        },
        "budgets": budgets,
    }


def build_branch_ablation_readout(
    candidates: Mapping[str, Dict[str, Any]],
    references: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    named = {**references, **candidates}
    relationships = {
        "context_increment_dense_keep": ("tpd_clean_ctx", "spd", "nested"),
        "saliency_increment_dense_keep": ("tpd_clean_sal", "spd", "nested"),
        "full_residual_increment_dense_keep": ("tpd_clean_full", "spd", "nested"),
        "context_necessity_clue": ("tpd_clean_full", "tpd_clean_sal", "nested"),
        "saliency_necessity_clue": ("tpd_clean_full", "tpd_clean_ctx", "nested"),
        "grouped_projection_effect": ("grouped_keep", "spd", "non_nested"),
        "full_vs_grouped_keep": ("tpd_clean_full", "grouped_keep", "non_nested"),
        "full_vs_tpd_v1": ("tpd_clean_full", "tpd", "non_nested"),
    }
    output: Dict[str, Any] = {}
    for label, (left_name, right_name, control_type) in relationships.items():
        left = named[left_name]
        right = named[right_name]
        output[label] = {
            "left": left_name,
            "right": right_name,
            "control_type": control_type,
            "checkpoint_roles": {
                "best_pd_primary": compare_sweep_pair(
                    left["best_sweep"], right["best_sweep"]
                ),
                "best_miou_secondary": compare_sweep_pair(
                    left["best_miou_sweep"], right["best_miou_sweep"]
                ),
            },
        }
    return {
        "relationships": output,
        "interpretation_scope": "descriptive_single_seed_screening_only",
        "tiny_pd_ceiling_note": (
            "39/39 is a ceiling and cannot create a branch-ranking advantage"
        ),
        "three_branch_necessity_established": False,
        "causal_mechanism_established": False,
    }


def evaluate_next_module_gate(
    candidates: Mapping[str, Dict[str, Any]],
    references: Mapping[str, Dict[str, Any]],
    candidate_conclusions: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    gate_path = REPO_ROOT / "experiments/tpd_clean_next_module_gate_v1.json"
    gate = load_json(gate_path)
    require(
        gate.get("schema") == "sctransnet_tpd_clean_next_module_gate_v1",
        "Unexpected next-module gate schema",
    )
    scope = gate.get("scope")
    require(isinstance(scope, dict), "Next-module gate lacks scope")
    require(scope.get("candidate_required_for_next_module") == "tpd_clean_full", "Next-module gate candidate changed")
    require(scope.get("mainline_change_allowed") is False, "Next-module gate cannot change mainline")
    require(scope.get("innovation_change_allowed") is False, "Next-module gate cannot change innovation")
    criteria = gate["formal_module_launch_gate_after_800"]["checks"]
    full = candidates["tpd_clean_full"]
    best = full["best_sweep"]
    best_miou = full["best_miou_sweep"]
    fixed = best["fixed_threshold_0_5"]
    fixed_miou = best_miou["fixed_threshold_0_5"]
    strict = best["best_points_under_fa_budget"]["1e-06"]
    wider_keys = tuple(criteria["wider_budget_keys"])
    wider = [best["best_points_under_fa_budget"][key] for key in wider_keys]
    conclusion = candidate_conclusions["tpd_clean_full"]["checkpoint_role_results"][
        "best_pd_primary"
    ]

    def full_vs(control_name: str, budget: str) -> str:
        return compare_points(
            best["best_points_under_fa_budget"][budget],
            candidates[control_name]["best_sweep"]["best_points_under_fa_budget"][budget],
        )["outcome"]

    full_better_ctx = sum(
        full_vs("tpd_clean_ctx", budget) == "candidate_better"
        for budget in EXPECTED_BUDGETS
    )
    full_better_sal = sum(
        full_vs("tpd_clean_sal", budget) == "candidate_better"
        for budget in EXPECTED_BUDGETS
    )
    fixed_vs_tpd = compare_points(
        fixed, references["tpd"]["best_sweep"]["fixed_threshold_0_5"]
    )["outcome"]
    wider_vs_tpd = [
        compare_points(
            best["best_points_under_fa_budget"][budget],
            references["tpd"]["best_sweep"]["best_points_under_fa_budget"][budget],
        )["outcome"]
        for budget in wider_keys
    ]
    checks = {
        "all_four_candidates_complete_800_and_two_sweeps_verified": all(
            candidate.get("best_sweep") is not None
            and candidate.get("best_miou_sweep") is not None
            for candidate in candidates.values()
        ),
        "full_pd_primary_fixed_matched_targets_min": int(fixed["matched_target_count"])
        >= int(criteria["full_pd_primary_fixed_matched_targets_min"]),
        "full_pd_primary_fixed_fa_max": float(fixed["fa"])
        <= float(criteria["full_pd_primary_fixed_fa_max"]),
        "full_pd_primary_fixed_miou_min": float(fixed["miou"])
        >= float(criteria["full_pd_primary_fixed_miou_min"]),
        "full_pd_primary_at_fa_1e_6_matched_targets_min": strict is not None
        and int(strict["matched_target_count"])
        >= int(criteria["full_pd_primary_at_fa_1e_6_matched_targets_min"]),
        "full_pd_primary_at_fa_1e_6_fa_max": strict is not None
        and float(strict["fa"])
        <= float(criteria["full_pd_primary_at_fa_1e_6_fa_max"]),
        "full_pd_primary_each_wider_budget_matched_targets_min": all(
            point is not None
            and int(point["matched_target_count"])
            >= int(criteria["full_pd_primary_each_wider_budget_matched_targets_min"])
            for point in wider
        ),
        "full_pd_primary_noncovered_budget_count_min": len(
            conclusion["adds_noncovered_budget_keys"]
        )
        >= int(criteria["full_pd_primary_noncovered_budget_count_min"]),
        "full_miou_secondary_fixed_miou_min": float(fixed_miou["miou"])
        >= float(criteria["full_miou_secondary_fixed_miou_min"]),
        "full_miou_secondary_fixed_matched_targets_min": int(
            fixed_miou["matched_target_count"]
        )
        >= int(criteria["full_miou_secondary_fixed_matched_targets_min"]),
        "full_miou_secondary_fixed_fa_max": float(fixed_miou["fa"])
        <= float(criteria["full_miou_secondary_fixed_fa_max"]),
        "full_better_than_ctx_only_budget_count_min": full_better_ctx
        >= int(criteria["full_better_than_ctx_only_budget_count_min"]),
        "full_better_than_sal_only_budget_count_min": full_better_sal
        >= int(criteria["full_better_than_sal_only_budget_count_min"]),
        "full_not_worse_than_ctx_only_at_fa_1e_6": full_vs(
            "tpd_clean_ctx", "1e-06"
        )
        != "reference_better",
        "full_not_worse_than_sal_only_at_fa_1e_6": full_vs(
            "tpd_clean_sal", "1e-06"
        )
        != "reference_better",
        "fixed_and_swept_direction_must_be_coherent": fixed_vs_tpd
        != "reference_better"
        and all(outcome != "reference_better" for outcome in wider_vs_tpd),
    }
    passed = all(checks.values())
    return {
        "gate_file": str(gate_path.resolve()),
        "gate_file_sha256": file_sha256(gate_path),
        "candidate": "tpd_clean_full",
        "formal_module_launch_gate_passed": passed,
        "checks": checks,
        "observed_counts": {
            "full_better_than_ctx_only_budget_count": full_better_ctx,
            "full_better_than_sal_only_budget_count": full_better_sal,
            "full_noncovered_budget_keys": conclusion["adds_noncovered_budget_keys"],
        },
        "permitted_action": (
            gate["formal_module_launch_gate_after_800"]["permission_if_passed"]
            if passed
            else gate["formal_module_launch_gate_after_800"]["action_if_failed"]
        ),
        "mainline_changed": False,
        "innovation_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def point_text(point: Mapping[str, Any] | None) -> str:
    if point is None:
        return "—"
    return (
        f"{int(point['matched_target_count'])}/{int(point['target_count'])}; "
        f"{float(point['fa']):.8g}; {float(point['miou']):.6f}"
    )


def render_markdown(output: Dict[str, Any]) -> str:
    lines = [
        "# TPD-Clean-v2 single-seed screening summary",
        "",
        "> This report completes single-seed structural screening only. The TPD-v1 mainline and frozen formal800 decision remain unchanged.",
        "",
        "NUDT-SIRST internal validation, model seed 42; official test data were not accessed.",
    ]
    for role_key, role_title in (
        ("best_pd_primary", "Pd-primary checkpoint"),
        ("best_miou_secondary", "mIoU-secondary checkpoint"),
    ):
        lines.extend(
            [
                "",
                f"## {role_title}: fixed threshold 0.5",
                "",
                "| Method | Pd | tiny-Pd | Fa | mIoU |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, record in output["methods"].items():
            point = record["checkpoint_roles"][role_key]["fixed_threshold_0_5"]
            lines.append(
                f"| `{method}` | {int(point['matched_target_count'])}/{int(point['target_count'])} "
                f"| {int(point['matched_tiny_target_count'])}/{int(point['tiny_target_count'])} "
                f"| {float(point['fa']):.8g} | {float(point['miou']):.6f} |"
            )
        lines.extend(
            [
                "",
                f"## {role_title}: Pd / actual Fa / mIoU at frozen Fa budgets",
                "",
                "| Method | 1e-6 | 5e-6 | 1e-5 | 5e-5 | 1e-4 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for method, record in output["methods"].items():
            budget_points = record["checkpoint_roles"][role_key]["budget_points"]
            cells = [point_text(budget_points.get(budget)) for budget in EXPECTED_BUDGETS]
            lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Candidate screening readout",
            "",
            "| Candidate | Pd-primary class | mIoU-secondary class | Noncovered budgets (Pd / mIoU) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for candidate, record in output["candidate_conclusions"].items():
        roles = record["checkpoint_role_results"]
        pd_role = roles["best_pd_primary"]
        miou_role = roles["best_miou_secondary"]
        lines.append(
            f"| `{candidate}` | `{pd_role['evidence_class']}` "
            f"| `{miou_role['evidence_class']}` "
            f"| {','.join(pd_role['adds_noncovered_budget_keys']) or 'none'} / "
            f"{','.join(miou_role['adds_noncovered_budget_keys']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"- Frozen formal decision: `{FROZEN_DECISION}`",
            "- `mainline_decision_made=false`",
            "- `mainline_changed=false`",
            "- `paper_core_established=false`",
            "- `stability_claim_supported=false`",
            "- `three_branch_necessity_established=false`",
            "- `causal_mechanism_established=false`",
            f"- `formal_module_launch_gate_passed={str(output['next_module_gate']['formal_module_launch_gate_passed']).lower()}`",
            "- A candidate may only be nominated for paired multi-seed confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_output(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary output exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    frozen_snapshot = verify_frozen_comparison(args.reference_root)
    source_lock_snapshot = verify_source_lock(
        REPO_ROOT / "experiments/tpd_clean_screen800_source_lock.json"
    )
    candidates = {
        variant: validate_run(
            args.candidate_root,
            args.candidate_run_name,
            variant,
            require_miou_sweep=True,
        )
        for variant in CANDIDATE_VARIANTS
    }
    references = {
        variant: validate_run(
            args.reference_root,
            args.reference_run_name,
            variant,
            require_miou_sweep=False,
        )
        for variant in REFERENCE_VARIANTS
    }
    for variant in REFERENCE_VARIANTS:
        references[variant] = validate_reference_miou_sweep(
            args.reference_miou_root,
            args.reference_run_name,
            references[variant],
        )

    all_records = [*candidates.values(), *references.values()]
    split_hashes = {record["split_sha256"] for record in all_records}
    require(len(split_hashes) == 1, "Candidate/reference validation splits differ")
    split_artifact_hashes = {
        record["artifact_sha256"]["split.json"] for record in all_records
    }
    require(len(split_artifact_hashes) == 1, "Candidate/reference split artifacts differ")
    critical_protocols = {canonical(record["critical_protocol"]) for record in all_records}
    require(len(critical_protocols) == 1, "Candidate/reference critical protocols differ")
    protocol_contracts = {canonical(record["protocol_contract"]) for record in all_records}
    require(len(protocol_contracts) == 1, "Candidate/reference protocol contracts differ")
    shared_initializations = {
        record["summary"]["model"]["shared_initialization_sha256"]
        for record in all_records
    }
    require(len(shared_initializations) == 1, "Shared model initialization differs")
    for variant, record in candidates.items():
        launch = record["launch_manifest"]
        require(isinstance(launch, dict), f"{variant}: missing launch manifest audit")
        require(
            launch["source_lock_sha256"] == source_lock_snapshot["sha256"],
            f"{variant}: launch source-lock digest mismatch",
        )

    candidate_conclusions = {
        variant: summarize_candidate(record, references)
        for variant, record in candidates.items()
    }
    methods: Dict[str, Any] = {}
    for variant, record in references.items():
        best = record["best_sweep"]
        best_miou = record["best_miou_sweep"]
        methods[variant] = {
            "role": "frozen_reference",
            "checkpoint_roles": {
                "best_pd_primary": {
                    "fixed_threshold_0_5": best["fixed_threshold_0_5"],
                    "budget_points": best["best_points_under_fa_budget"],
                    "checkpoint_epoch": best["checkpoint_epoch"],
                    "checkpoint_sha256": best["checkpoint_sha256"],
                },
                "best_miou_secondary": {
                    "fixed_threshold_0_5": best_miou["fixed_threshold_0_5"],
                    "budget_points": best_miou["best_points_under_fa_budget"],
                    "checkpoint_epoch": best_miou["checkpoint_epoch"],
                    "checkpoint_sha256": best_miou["checkpoint_sha256"],
                    "derived_auxiliary_sweep": True,
                },
            },
            "fixed_threshold_0_5": best["fixed_threshold_0_5"],
            "budget_points": best["best_points_under_fa_budget"],
            "best_miou_checkpoint_metrics": record["summary"][
                "best_miou_validation_metrics"
            ],
            "total_parameters": record["summary"]["model"]["total_parameters"],
            "shallow_embedding_parameters": record["summary"]["model"][
                "shallow_embedding_parameters"
            ],
        }
    for variant, record in candidates.items():
        conclusion = candidate_conclusions[variant]
        best = record["best_sweep"]
        best_miou = record["best_miou_sweep"]
        methods[variant] = {
            "role": "fresh_candidate",
            "checkpoint_roles": {
                "best_pd_primary": {
                    "fixed_threshold_0_5": best["fixed_threshold_0_5"],
                    "budget_points": best["best_points_under_fa_budget"],
                    "checkpoint_epoch": best["checkpoint_epoch"],
                    "checkpoint_sha256": best["checkpoint_sha256"],
                },
                "best_miou_secondary": {
                    "fixed_threshold_0_5": best_miou["fixed_threshold_0_5"],
                    "budget_points": best_miou["best_points_under_fa_budget"],
                    "checkpoint_epoch": best_miou["checkpoint_epoch"],
                    "checkpoint_sha256": best_miou["checkpoint_sha256"],
                    "derived_auxiliary_sweep": False,
                },
            },
            "fixed_threshold_0_5": conclusion["fixed_threshold_0_5"],
            "budget_points": conclusion["budget_points"],
            "best_miou_checkpoint_metrics": conclusion[
                "best_miou_checkpoint_metrics"
            ],
            "best_miou_fixed_threshold_0_5": conclusion[
                "best_miou_fixed_threshold_0_5"
            ],
            "best_miou_budget_points": conclusion["best_miou_budget_points"],
            "total_parameters": conclusion["total_parameters"],
            "shallow_embedding_parameters": conclusion[
                "shallow_embedding_parameters"
            ],
        }

    branch_readout = build_branch_ablation_readout(candidates, references)
    next_module_gate = evaluate_next_module_gate(
        candidates, references, candidate_conclusions
    )
    output = {
        "schema": "sctransnet_tpd_clean_screen800_comparison_v2",
        "status": "complete",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "dataset": EXPECTED_DATASET,
            "model_seed": EXPECTED_SEED,
            "split_seed": EXPECTED_SPLIT_SEED,
            "training_count": EXPECTED_TRAIN_COUNT,
            "validation_count": EXPECTED_VALIDATION_COUNT,
            "validation_split_sha256": next(iter(split_hashes)),
            "single_dataset_single_seed_screening_only": True,
            "shared_resource_run": True,
            "efficiency_comparison_allowed": False,
            "official_test_accessed": False,
        },
        "mainline_scope": {
            "mainline_before": "TPD-v1",
            "mainline_after": "TPD-v1",
            "mainline_changed": False,
            "mainline_decision_made": False,
            "formal_decision_preserved": FROZEN_DECISION,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "automatic_model_replacement": False,
            "report_semantics": "single-seed candidate screening only",
        },
        "comparison_policy": {
            "within_curve_order": [
                "higher Pd",
                "lower actual Fa",
                "higher tiny-Pd",
                "higher mIoU",
                "threshold closest to 0.5",
            ],
            "cross_method_order": [
                "available point",
                "higher Pd",
                "lower actual Fa",
                "higher tiny-Pd",
                "higher mIoU",
            ],
            "threshold_is_cross_method_advantage": False,
            "checkpoint_roles_compared_separately": True,
            "pareto_dimensions": ["Pd", "Fa"],
        },
        "fa_budgets": list(EXPECTED_BUDGETS),
        "common_provenance": {
            "critical_protocol": all_records[0]["critical_protocol"],
            "protocol_contract": all_records[0]["protocol_contract"],
            "split_artifact_sha256": next(iter(split_artifact_hashes)),
            "shared_initialization_sha256": next(iter(shared_initializations)),
            "source_lock": source_lock_snapshot,
        },
        "frozen_reference_snapshot": frozen_snapshot,
        "derived_best_miou_references": {
            variant: record["derived_best_miou_reference"]
            for variant, record in references.items()
        },
        "candidate_runs": {
            variant: {
                "run_directory": record["run_dir"],
                "artifact_sha256": record["artifact_sha256"],
                "checkpoints": record["checkpoints"],
                "completion_log": record["completion_log"],
                "launch_manifest": record["launch_manifest"],
            }
            for variant, record in candidates.items()
        },
        "methods": methods,
        "candidate_conclusions": candidate_conclusions,
        "branch_ablation_readout": branch_readout,
        "next_module_gate": next_module_gate,
        "decision_boundary": {
            "mainline_decision_made": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "three_branch_necessity_established": False,
            "causal_mechanism_established": False,
            "mainline_changed": False,
            "permitted_action": "nominate_candidate_for_paired_seed_confirmation_only",
        },
        "validation_checks": {
            "four_candidates_complete_800": True,
            "candidate_best_and_best_miou_sweeps_verified": True,
            "reference_best_sweeps_verified": True,
            "reference_best_miou_auxiliary_sweeps_verified": True,
            "candidate_checkpoints_strict_loaded": True,
            "frozen_formal_artifacts_verified": True,
            "source_lock_verified": True,
            "split_and_protocol_matched": True,
            "official_test_accessed": False,
        },
        "limitations": [
            "NUDT-SIRST official-training-set 530/133 internal split only",
            "single model seed 42 only",
            "shared GPU and CPU resources; no efficiency comparison",
            "tiny-Pd ceiling cannot establish target-preservation mechanism",
            "no paired multi-seed, multi-dataset, or official-test confirmation",
        ],
        "postprocessor_sha256": {
            "summarizer": file_sha256(Path(__file__).resolve()),
            "base_evaluator": file_sha256(
                REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
            ),
            "clean_evaluator_wrapper": file_sha256(
                REPO_ROOT / "experiments/evaluate_tpd_clean_v2_pd_fa.py"
            ),
            "next_module_gate_json": file_sha256(
                REPO_ROOT / "experiments/tpd_clean_next_module_gate_v1.json"
            ),
            "next_module_gate_markdown": file_sha256(
                REPO_ROOT / "experiments/TPD_CLEAN_V2_NEXT_MODULE_GATE.md"
            ),
        },
        "artifact_sha256": {
            variant: record["artifact_sha256"]
            for variant, record in {**references, **candidates}.items()
        },
    }
    require_finite(output, "output")
    json_content = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    markdown_content = render_markdown(output)
    output_dir = args.output_dir.resolve()
    json_path = output_dir / "tpd_clean_screen800_comparison_seed42.json"
    markdown_path = output_dir / "tpd_clean_screen800_comparison_seed42.md"
    marker_path = output_dir / "tpd_clean_screen800_comparison_seed42.COMPLETE.sha256"
    write_output(json_path, json_content, args.overwrite)
    write_output(markdown_path, markdown_content, args.overwrite)
    marker_content = (
        f"{file_sha256(json_path)}  {json_path.name}\n"
        f"{file_sha256(markdown_path)}  {markdown_path.name}\n"
    )
    write_output(marker_path, marker_content, args.overwrite)
    print(
        "TPDCLEAN_COMPARISON_COMPLETE"
        f" candidates={len(candidates)} references={len(references)}"
        f" output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
