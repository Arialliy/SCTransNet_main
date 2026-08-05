#!/usr/bin/env python3
"""Build an additive pixel-confusion sidecar for the frozen 36 evaluations.

This is a read-only, CPU-only postprocessor.  It does not run inference,
reselect a checkpoint, or modify any historical evaluation/comparison file.
The exact aggregate pixel confusion matrix is recovered from the frozen test
masks plus the already reported aggregate pixel precision/recall.  Reported
precision, recall, F1, and mIoU are then recomputed from the recovered integer
counts and must agree before the sidecar is written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import three_dataset_v2_protocol as data_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
POSITIVE_ROOT = RESULTS_ROOT / "three_dataset_seed42_global_tss_v2"
OFF_ROOT = RESULTS_ROOT / "three_dataset_tss_off_seed42_v1"
EC_ROOT = RESULTS_ROOT / "three_dataset_ec_tss_v3_1_seed42"
DEFAULT_OUTPUT_DIR = EC_ROOT / "comparison"
DEFAULT_JSON_NAME = "additive_joint_metrics_v1.json"
DEFAULT_MARKDOWN_NAME = "additive_joint_metrics_v1.md"

DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
ROLES = ("best_miou", "best_pd")
RECIPES = (
    "original",
    "tss_off",
    "lambda_0p0025",
    "lambda_0p005",
    "lambda_0p01",
    "ec_tss_v3_1",
)

INPUT_SCHEMA = "sctransnet_three_dataset_v2_evaluation_v1"
OUTPUT_SCHEMA = "sctransnet_additive_joint_metrics/v1"
FIXED_THRESHOLD = 0.5
SEED = 42
IDENTITY_ABS_TOLERANCE = 1e-12
COUNT_RECOVERY_ABS_TOLERANCE = 1e-6


class SidecarError(ValueError):
    """An input violates the frozen additive-metrics contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SidecarError(message)


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in output, f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def load_json(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"expected regular non-symlink JSON: {candidate}",
    )
    payload = json.loads(
        candidate.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            SidecarError(f"non-finite JSON token: {token}")
        ),
    )
    _require(isinstance(payload, dict), f"JSON root is not an object: {candidate}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        ready = float(value)
    except (TypeError, ValueError) as error:
        raise SidecarError(f"{label} must be numeric") from error
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _unit(value: Any, label: str) -> float:
    ready = _finite(value, label)
    _require(0.0 <= ready <= 1.0, f"{label} must be in [0, 1]")
    return ready


def _count(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be int")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def recipe_evaluation_path(recipe: str, dataset: str, role: str) -> Path:
    if recipe == "original":
        return (
            POSITIVE_ROOT
            / "runs"
            / dataset
            / "original"
            / "seed_42"
            / "evaluations"
            / f"{role}.json"
        )
    if recipe.startswith("lambda_"):
        return (
            POSITIVE_ROOT
            / "runs"
            / dataset
            / "final"
            / recipe
            / "seed_42"
            / "evaluations"
            / f"{role}.json"
        )
    if recipe == "tss_off":
        return (
            OFF_ROOT
            / "runs"
            / dataset
            / "final_tss_off"
            / "seed_42"
            / "evaluations"
            / f"{role}.json"
        )
    if recipe == "ec_tss_v3_1":
        return (
            EC_ROOT
            / "runs"
            / dataset
            / "final_ec_tss_v3_1"
            / "seed_42"
            / "evaluations"
            / f"{role}.json"
        )
    raise SidecarError(f"unknown recipe: {recipe}")


def _expected_method(recipe: str) -> str:
    if recipe == "original":
        return "original"
    if recipe == "tss_off":
        return "final_tss_off"
    if recipe == "ec_tss_v3_1":
        return "final_ec_tss_v3_1"
    return "final"


def _mask_statistics(
    dataset_root: Path,
    manifest_path: Path,
    dataset: str,
) -> dict[str, Any]:
    identifiers = data_protocol.load_frozen_index(
        dataset_root,
        dataset,
        "test",
        manifest_path,
    )
    known_ids = frozenset(identifiers)
    valid = 0
    positive = 0
    ordered_mask_digest = hashlib.sha256()
    for sample_id in identifiers:
        sample = data_protocol.resolve_sample(
            dataset_root,
            dataset,
            sample_id,
            split="test",
            known_ids=known_ids,
        )
        mask_bytes = sample.mask_path.read_bytes()
        mask_sha = hashlib.sha256(mask_bytes).hexdigest()
        ordered_mask_digest.update(sample_id.encode("utf-8"))
        ordered_mask_digest.update(b"\0")
        ordered_mask_digest.update(mask_sha.encode("ascii"))
        ordered_mask_digest.update(b"\n")
        with Image.open(io.BytesIO(mask_bytes)) as image:
            mask = np.asarray(image, dtype=np.float32)
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        _require(mask.ndim == 2, f"mask is not 2-D: {dataset}::{sample_id}")
        _require(bool(np.isfinite(mask).all()), f"mask is non-finite: {dataset}::{sample_id}")
        target = mask / np.float32(255.0) > np.float32(0.5)
        valid += int(target.size)
        positive += int(target.sum())
    _require(0 < positive < valid, f"invalid positive/valid counts for {dataset}")
    expected = data_protocol.EXPECTED_SPLITS[dataset]["test"]
    return {
        "split": "img_idx/test",
        "test_image_count": len(identifiers),
        "img_idx_test_file_sha256": expected["file_sha256"],
        "img_idx_test_ordered_ids_sha256": expected["ordered_ids_sha256"],
        "effective_masks_ordered_id_and_sha256_digest": ordered_mask_digest.hexdigest(),
        "mask_binarization": "float32(mask)/255 > float32(0.5)",
        "valid_pixel_count": valid,
        "positive_pixel_count": positive,
        "background_pixel_count": valid - positive,
    }


def _recover_integer(raw: float, label: str) -> int:
    _require(math.isfinite(raw) and raw >= 0.0, f"{label} is invalid")
    recovered = int(round(raw))
    _require(
        math.isclose(raw, recovered, rel_tol=0.0, abs_tol=COUNT_RECOVERY_ABS_TOLERANCE),
        f"{label} does not encode an integer: {raw}",
    )
    return recovered


def _ratio(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def _identity_check(observed: float, expected: float, label: str) -> float:
    error = abs(observed - expected)
    _require(
        error <= IDENTITY_ABS_TOLERANCE,
        f"{label} identity differs by {error} (observed={observed}, expected={expected})",
    )
    return error


def _normalize_evaluation(
    path: Path,
    payload: Mapping[str, Any],
    *,
    recipe: str,
    dataset: str,
    role: str,
    manifest_path: Path,
    manifest_sha256: str,
    mask_stats: Mapping[str, Any],
) -> dict[str, Any]:
    label = f"{recipe}/{dataset}/{role}"
    _require(payload.get("schema") == INPUT_SCHEMA, f"{label} schema differs")
    _require(payload.get("status") == "complete", f"{label} is not complete")
    _require(payload.get("dataset") == dataset, f"{label} dataset differs")
    _require(payload.get("method") == _expected_method(recipe), f"{label} method differs")
    _require(payload.get("checkpoint_role") == role, f"{label} role differs")
    _require(payload.get("seed") == SEED, f"{label} seed differs")

    data = payload.get("data")
    _require(isinstance(data, Mapping), f"{label} data binding is absent")
    declared_manifest = data.get("protocol_manifest")
    _require(isinstance(declared_manifest, Mapping), f"{label} manifest binding is absent")
    _require(
        Path(str(declared_manifest.get("path"))).resolve(strict=True) == manifest_path,
        f"{label} manifest path differs",
    )
    _require(
        declared_manifest.get("sha256") == manifest_sha256,
        f"{label} manifest SHA differs",
    )
    _require(data.get("split") == "img_idx/test", f"{label} split differs")
    _require(
        data.get("img_idx_test_sha256") == mask_stats["img_idx_test_file_sha256"],
        f"{label} img_idx file SHA differs",
    )
    _require(
        data.get("img_idx_test_ordered_ids_sha256")
        == mask_stats["img_idx_test_ordered_ids_sha256"],
        f"{label} ordered IDs SHA differs",
    )

    point = payload.get("fixed_threshold_0_5")
    _require(isinstance(point, Mapping), f"{label} lacks fixed_threshold_0_5")
    threshold = _finite(point.get("threshold"), f"{label}.threshold")
    _require(threshold == FIXED_THRESHOLD, f"{label} threshold is not exactly 0.5")
    valid = _count(point.get("valid_pixel_count"), f"{label}.valid_pixel_count")
    _require(valid == mask_stats["valid_pixel_count"], f"{label} valid pixels differ")

    precision = _unit(point.get("pixel_precision"), f"{label}.pixel_precision")
    recall = _unit(point.get("pixel_recall"), f"{label}.pixel_recall")
    f1 = _unit(point.get("pixel_f1"), f"{label}.pixel_f1")
    miou = _unit(point.get("miou"), f"{label}.miou")
    positive = int(mask_stats["positive_pixel_count"])
    background = int(mask_stats["background_pixel_count"])
    tp = _recover_integer(recall * positive, f"{label}.TP")
    _require(tp > 0 and precision > 0.0, f"{label} cannot uniquely recover predicted positives")
    predicted_positive = _recover_integer(tp / precision, f"{label}.predicted_positive")
    _require(predicted_positive >= tp, f"{label} predicted positives are below TP")
    fp = predicted_positive - tp
    fn = positive - tp
    _require(fp <= background, f"{label} FP exceeds background pixels")
    tn = background - fp

    recomputed_precision = _ratio(tp, tp + fp)
    recomputed_recall = _ratio(tp, tp + fn)
    recomputed_f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    recomputed_miou = _ratio(tp, tp + fp + fn, empty_value=1.0)
    errors = {
        "pixel_precision_abs_error": _identity_check(
            precision, recomputed_precision, f"{label}.pixel_precision"
        ),
        "pixel_recall_abs_error": _identity_check(
            recall, recomputed_recall, f"{label}.pixel_recall"
        ),
        "pixel_f1_abs_error": _identity_check(f1, recomputed_f1, f"{label}.pixel_f1"),
        "miou_abs_error": _identity_check(miou, recomputed_miou, f"{label}.miou"),
    }

    component_fields = (
        "pd",
        "tiny_pd",
        "fa",
        "false_objects_per_image",
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "unmatched_predicted_pixels",
        "valid_pixel_count",
    )
    overlap_fields = ("miou", "niou")
    pixel_fields = ("pixel_precision", "pixel_recall", "pixel_f1")
    _require(
        all(field in point for field in component_fields + overlap_fields + pixel_fields),
        f"{label} lacks inherited metrics",
    )
    checkpoint = payload.get("checkpoint_binding", {}).get("checkpoint", {})
    _require(isinstance(checkpoint, Mapping), f"{label} checkpoint binding is absent")
    source_sha = payload.get("source_sha256")
    _require(isinstance(source_sha, Mapping), f"{label} source_sha256 is absent")

    return {
        "recipe": recipe,
        "dataset": dataset,
        "checkpoint_role": role,
        "threshold": FIXED_THRESHOLD,
        "source": {
            "evaluation_path": str(path.resolve()),
            "evaluation_file_sha256": file_sha256(path),
            "evaluation_declared_source_sha256": dict(source_sha),
            "protocol_manifest_path": str(manifest_path),
            "protocol_manifest_file_sha256": manifest_sha256,
            "checkpoint_path": checkpoint.get("path"),
            "checkpoint_sha256": checkpoint.get("sha256"),
            "checkpoint_epoch": checkpoint.get("epoch"),
        },
        "inherited_metrics": {
            "object_and_component": {field: point[field] for field in component_fields},
            "overlap": {field: point[field] for field in overlap_fields},
            "pixel": {field: point[field] for field in pixel_fields},
        },
        "additive_pixel_confusion": {
            "true_positive_pixels": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "true_negative_pixels": tn,
            "predicted_positive_pixels": predicted_positive,
            "ground_truth_positive_pixels": positive,
            "ground_truth_background_pixels": background,
            "valid_pixel_count": valid,
            "false_positive_per_valid_pixel": fp / valid,
            "false_positive_per_background_pixel": fp / background,
        },
        "identity_validation": {
            "passed": True,
            "absolute_tolerance": IDENTITY_ABS_TOLERANCE,
            **errors,
        },
    }


def build_sidecar(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    data_protocol.load_protocol_manifest(manifest_path, dataset_root=dataset_root)
    manifest_sha = file_sha256(manifest_path)
    mask_statistics = {
        dataset: _mask_statistics(dataset_root, manifest_path, dataset)
        for dataset in DATASETS
    }
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for dataset in DATASETS:
        for recipe in RECIPES:
            for role in ROLES:
                path = recipe_evaluation_path(recipe, dataset, role).resolve(strict=True)
                _require(path not in seen_paths, f"duplicate evaluation input: {path}")
                seen_paths.add(path)
                records.append(
                    _normalize_evaluation(
                        path,
                        load_json(path),
                        recipe=recipe,
                        dataset=dataset,
                        role=role,
                        manifest_path=manifest_path,
                        manifest_sha256=manifest_sha,
                        mask_stats=mask_statistics[dataset],
                    )
                )
    _require(len(records) == 36 and len(seen_paths) == 36, "input matrix is not 36 unique points")
    max_errors = {
        key: max(record["identity_validation"][key] for record in records)
        for key in (
            "pixel_precision_abs_error",
            "pixel_recall_abs_error",
            "pixel_f1_abs_error",
            "miou_abs_error",
        )
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "scope": "seed42_three_dataset_fixed_threshold_0_5_additive_audit",
        "seed": SEED,
        "threshold": FIXED_THRESHOLD,
        "selection_effect": "none",
        "checkpoint_reselected": False,
        "historical_metrics_overwritten": False,
        "execution": {
            "device": "cpu",
            "inference_rerun": False,
            "derivation": (
                "GT positives come from the frozen effective img_idx/test masks; "
                "integer TP is recovered from reported pixel recall and integer "
                "predicted positives from reported pixel precision."
            ),
        },
        "source_manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_sha,
            "schema": data_protocol.SCHEMA,
            "manifest_id": data_protocol.MANIFEST_ID,
        },
        "dataset_mask_statistics": mask_statistics,
        "coverage": {
            "datasets": list(DATASETS),
            "recipes": list(RECIPES),
            "checkpoint_roles": list(ROLES),
            "evaluation_count": len(records),
            "expected_evaluation_count": 36,
        },
        "identity_validation": {
            "passed": True,
            "checked_evaluation_count": len(records),
            "absolute_tolerance": IDENTITY_ABS_TOLERANCE,
            "count_recovery_absolute_tolerance": COUNT_RECOVERY_ABS_TOLERANCE,
            "maximum_absolute_errors": max_errors,
        },
        "records": records,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    records = [record for record in payload["records"] if record["recipe"] == "ec_tss_v3_1"]
    lines = [
        "# Additive Joint Metrics V1",
        "",
        "只读 CPU sidecar；固定 seed 42、阈值 0.5，不重选 checkpoint，不覆盖历史指标。",
        "",
        "- 输入：36/36 个正式 evaluation JSON",
        "- 恒等式：36/36 个点的 pixel precision / recall / F1 / mIoU 全部通过",
        "- `pixel FP/valid` 统计所有背景误分像素，区别于只统计未匹配连通域的 component Fa。",
        "",
        "## EC-TSS V3.1 六个点",
        "",
        "| 数据集 | checkpoint | mIoU | nIoU | Pd | component Fa | pixel precision | pixel F1 | pixel FP/valid | pixel FP/background | false objects/image | TP | FP | FN | TN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        inherited = record["inherited_metrics"]
        obj = inherited["object_and_component"]
        overlap = inherited["overlap"]
        pixel = inherited["pixel"]
        confusion = record["additive_pixel_confusion"]
        lines.append(
            "| {dataset} | {role} | {miou:.6f} | {niou:.6f} | {pd:.6f} | "
            "{fa:.6e} | {precision:.6f} | {f1:.6f} | {fpv:.6e} | {fpb:.6e} | "
            "{fo:.6f} | {tp} | {fp} | {fn} | {tn} |".format(
                dataset=record["dataset"],
                role=record["checkpoint_role"],
                miou=overlap["miou"],
                niou=overlap["niou"],
                pd=obj["pd"],
                fa=obj["fa"],
                precision=pixel["pixel_precision"],
                f1=pixel["pixel_f1"],
                fpv=confusion["false_positive_per_valid_pixel"],
                fpb=confusion["false_positive_per_background_pixel"],
                fo=obj["false_objects_per_image"],
                tp=confusion["true_positive_pixels"],
                fp=confusion["false_positive_pixels"],
                fn=confusion["false_negative_pixels"],
                tn=confusion["true_negative_pixels"],
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该文件只补充同一批固定 checkpoint、固定阈值结果的像素混淆矩阵；不会改变任何历史模型排名或正式裁决。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    json_path = output_dir / DEFAULT_JSON_NAME
    markdown_path = output_dir / DEFAULT_MARKDOWN_NAME
    json_bytes = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    markdown_bytes = render_markdown(payload).encode("utf-8")
    _atomic_write(json_path, json_bytes)
    _atomic_write(markdown_path, markdown_bytes)
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=data_protocol.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_sidecar(args.dataset_root, args.manifest)
    json_path, markdown_path = write_outputs(payload, args.output_dir)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "evaluation_count": payload["coverage"]["evaluation_count"],
                "identity_validation_passed": payload["identity_validation"]["passed"],
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
