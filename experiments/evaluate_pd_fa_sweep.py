#!/usr/bin/env python3
"""Evaluate a validation-selected checkpoint across a Pd--Fa threshold sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_tpd_pilot import (  # noqa: E402
    ValidationMetrics,
    ValidationSubset,
    build_model,
    final_prediction,
    json_ready,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal-validation Pd--Fa sweep")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth.tar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--expected-epochs",
        type=int,
        default=None,
        help="Expected complete epoch count; defaults to protocol arguments.epochs",
    )
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--extra-thresholds",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
    )
    parser.add_argument("--tail-logit-step", type=float, default=0.1)
    parser.add_argument(
        "--fa-budgets",
        type=float,
        nargs="+",
        default=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    )
    parser.add_argument("--match-radius", type=float, default=None)
    parser.add_argument("--tiny-area", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing sweep JSON",
    )
    args = parser.parse_args()
    if args.expected_epochs is not None and args.expected_epochs < 1:
        parser.error("--expected-epochs must be >= 1")
    if not (
        math.isfinite(args.threshold_min)
        and math.isfinite(args.threshold_max)
        and 0.0 < args.threshold_min <= args.threshold_max < 1.0
    ):
        parser.error("threshold range must lie inside (0, 1)")
    if not math.isfinite(args.threshold_step) or args.threshold_step <= 0:
        parser.error("--threshold-step must be positive")
    if any(
        not math.isfinite(threshold) or not 0.0 < threshold < 1.0
        for threshold in args.extra_thresholds
    ):
        parser.error("--extra-thresholds must lie inside (0, 1)")
    if not math.isfinite(args.tail_logit_step) or args.tail_logit_step <= 0:
        parser.error("--tail-logit-step must be positive")
    if any(not math.isfinite(budget) or budget < 0 for budget in args.fa_budgets):
        parser.error("--fa-budgets must be non-negative")
    if args.match_radius is not None and (
        not math.isfinite(args.match_radius) or args.match_radius <= 0
    ):
        parser.error("--match-radius must be finite and positive")
    if args.tiny_area is not None and args.tiny_area < 0:
        parser.error("--tiny-area must be non-negative")
    return args


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identifier_sha256(identifiers: Sequence[str]) -> str:
    canonical = "\n".join(sorted(str(identifier) for identifier in identifiers))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_same(label: str, named_values: Dict[str, Any]) -> Any:
    iterator = iter(named_values.items())
    first_name, expected = next(iterator)
    for name, value in iterator:
        if value != expected:
            raise ValueError(
                f"{label} mismatch: {first_name}={expected!r}, {name}={value!r}"
            )
    return expected


def assert_finite_numbers(value: Any, location: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {location}: {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_numbers(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_numbers(item, f"{location}[{index}]")


def require_finite_numeric_fields(
    payload: Dict[str, Any], keys: Sequence[str], location: str
) -> None:
    for key in keys:
        value = payload.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"Expected finite numeric field at {location}.{key}, found {value!r}"
            )


def load_complete_metrics(path: Path, expected_epochs: int) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != expected_epochs:
        raise ValueError(
            f"Expected {expected_epochs} metrics rows in {path}, found {len(raw_lines)}"
        )
    events: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            raise ValueError(f"Blank metrics row at {path}:{line_number}")
        event = json.loads(raw_line)
        if not isinstance(event, dict):
            raise ValueError(f"Metrics row is not an object at {path}:{line_number}")
        if type(event.get("epoch")) is not int or event["epoch"] != line_number:
            raise ValueError(
                f"Metrics epochs must be exactly 1..{expected_epochs}; "
                f"row {line_number} has epoch={event.get('epoch')!r}"
            )
        assert_finite_numbers(event, f"metrics[{line_number}]")
        events.append(event)
    return events


def validate_identifier_manifest(split: Dict[str, Any]) -> Dict[str, str]:
    identifier_fields = {
        "full_internal_train_sha256": "full_internal_train_ids",
        "full_internal_val_sha256": "full_internal_val_ids",
        "used_train_sha256": "used_train_ids",
        "used_val_sha256": "used_val_ids",
    }
    hashes = split.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError("split.json is missing hashes")
    recomputed: Dict[str, str] = {}
    identifier_sets: Dict[str, set[str]] = {}
    for hash_key, identifier_key in identifier_fields.items():
        identifiers = split.get(identifier_key)
        if not isinstance(identifiers, list) or not all(
            isinstance(identifier, str) for identifier in identifiers
        ):
            raise ValueError(f"split.json has invalid {identifier_key}")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"split.json has duplicate IDs in {identifier_key}")
        recomputed[hash_key] = identifier_sha256(identifiers)
        if hashes.get(hash_key) != recomputed[hash_key]:
            raise ValueError(
                f"split hash mismatch for {identifier_key}: "
                f"manifest={hashes.get(hash_key)!r}, recomputed={recomputed[hash_key]!r}"
            )
        count_key = identifier_key.removesuffix("_ids") + "_count"
        if split.get(count_key) != len(identifiers):
            raise ValueError(
                f"split count mismatch for {identifier_key}: "
                f"declared={split.get(count_key)!r}, actual={len(identifiers)}"
            )
        identifier_sets[identifier_key] = set(identifiers)

    if identifier_sets["full_internal_train_ids"] & identifier_sets["full_internal_val_ids"]:
        raise ValueError("Full internal train/validation IDs overlap")
    if identifier_sets["used_train_ids"] & identifier_sets["used_val_ids"]:
        raise ValueError("Used internal train/validation IDs overlap")
    if not identifier_sets["used_train_ids"] <= identifier_sets["full_internal_train_ids"]:
        raise ValueError("used_train_ids is not a subset of full_internal_train_ids")
    if not identifier_sets["used_val_ids"] <= identifier_sets["full_internal_val_ids"]:
        raise ValueError("used_val_ids is not a subset of full_internal_val_ids")
    full_count = len(identifier_sets["full_internal_train_ids"]) + len(
        identifier_sets["full_internal_val_ids"]
    )
    if split.get("full_official_train_count") != full_count:
        raise ValueError(
            "Full internal train/validation union does not match full_official_train_count"
        )
    return recomputed


def write_output_json(path: Path, payload: Dict[str, Any], overwrite: bool) -> None:
    mode = "w" if overwrite else "x"
    try:
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(
                json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n"
            )
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite existing sweep output: {path}. "
            "Pass --overwrite explicitly to replace it."
        ) from error


def threshold_grid(
    minimum: float, maximum: float, step: float, extras: Sequence[float] = ()
) -> List[float]:
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    thresholds = [round(minimum + index * step, 10) for index in range(count + 1)]
    if thresholds[-1] < maximum - 1e-9:
        thresholds.append(float(maximum))
    if not any(abs(threshold - 0.5) < 1e-12 for threshold in thresholds):
        thresholds.append(0.5)
    thresholds.extend(float(threshold) for threshold in extras)
    return sorted(set(thresholds))


def best_point_under_fa(
    points: Sequence[Dict[str, Any]], budget: float
) -> Dict[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda point: (
            float(point["pd"]),
            -float(point["fa"]),
            float(point["tiny_pd"]) if point["tiny_pd"] is not None else -1.0,
            float(point["miou"]),
            -abs(float(point["threshold"]) - 0.5),
        ),
    )


def pd_primary_selection_key(metrics: Dict[str, Any]) -> Tuple[float, ...]:
    tiny_pd = float(metrics["tiny_pd"])
    return (
        float(metrics["pd"]),
        -float(metrics["fa"]),
        tiny_pd,
        float(metrics["miou"]),
        -float(metrics["val_loss"]),
    )


def miou_secondary_selection_key(metrics: Dict[str, Any]) -> Tuple[float, ...]:
    tiny_pd = float(metrics["tiny_pd"])
    return (
        float(metrics["miou"]),
        float(metrics["pd"]),
        -float(metrics["fa"]),
        tiny_pd,
        -float(metrics["val_loss"]),
    )


def audit_fixed_threshold_checkpoint(
    fixed_half: Dict[str, Any], checkpoint_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    if float(fixed_half["threshold"]) != 0.5:
        raise ValueError("Threshold sweep does not contain the exact 0.5 checkpoint threshold")
    count_metric_keys = sorted(
        key for key in checkpoint_metrics if key.endswith("_count")
    )
    exact_keys = list(
        dict.fromkeys(
            ["pd", "fa", "tiny_pd", "false_objects_per_image", *count_metric_keys]
        )
    )
    exact_matches: Dict[str, Any] = {}
    for key in exact_keys:
        if key not in checkpoint_metrics or key not in fixed_half:
            raise ValueError(f"Cannot audit fixed-threshold checkpoint metric {key!r}")
        exact_matches[key] = {
            "checkpoint": checkpoint_metrics[key],
            "sweep_0_5": fixed_half[key],
        }
        if fixed_half[key] != checkpoint_metrics[key]:
            raise ValueError(
                f"Fixed-threshold exact metric mismatch for {key}: "
                f"checkpoint={checkpoint_metrics[key]!r}, sweep={fixed_half[key]!r}"
            )
    numeric_deltas = {
        key: float(fixed_half[key]) - float(checkpoint_value)
        for key, checkpoint_value in checkpoint_metrics.items()
        if key in fixed_half
        and key not in exact_keys
        and isinstance(checkpoint_value, (int, float))
        and not isinstance(checkpoint_value, bool)
    }
    return {
        "exact_match_keys": exact_keys,
        "exact_matches": exact_matches,
        "non_strict_numeric_deltas_sweep_minus_checkpoint": numeric_deltas,
        "max_abs_non_strict_numeric_delta": max(
            (abs(delta) for delta in numeric_deltas.values()), default=0.0
        ),
    }


def adaptive_thresholds(
    probabilities: Sequence[np.ndarray],
    base_thresholds: Sequence[float],
    tail_logit_step: float,
) -> Tuple[List[float], Dict[str, Any]]:
    """Add dense log-odds tail samples and empirical score quantiles."""
    thresholds = set(float(threshold) for threshold in base_thresholds)
    lower_probability = 0.95
    upper_probability = 0.9999
    lower_logit = math.log(lower_probability / (1.0 - lower_probability))
    upper_logit = math.log(upper_probability / (1.0 - upper_probability))
    logit_values = np.arange(lower_logit, upper_logit + tail_logit_step / 2, tail_logit_step)
    logit_tail = [float(1.0 / (1.0 + math.exp(-value))) for value in logit_values]
    thresholds.update(logit_tail)

    quantile_levels = np.asarray(
        [
            0.90,
            0.95,
            0.98,
            0.99,
            0.995,
            0.999,
            0.9995,
            0.9999,
            0.99995,
            0.99999,
            0.999995,
            0.999999,
        ],
        dtype=np.float64,
    )
    flattened_scores = np.concatenate([probability.reshape(-1) for probability in probabilities])
    quantile_values = np.quantile(flattened_scores, quantile_levels)
    empirical_quantiles = {
        f"{level:.9g}": float(value)
        for level, value in zip(quantile_levels, quantile_values)
        if 0.0 < float(value) < 1.0
    }
    thresholds.update(empirical_quantiles.values())
    return sorted(thresholds), {
        "uniform_probability_grid_count": len(base_thresholds),
        "tail_logit_range": [lower_logit, upper_logit],
        "tail_logit_step": tail_logit_step,
        "tail_logit_threshold_count": len(logit_tail),
        "empirical_score_quantiles": empirical_quantiles,
        "total_unique_threshold_count": len(thresholds),
    }


@torch.inference_mode()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    model.eval()
    criterion = nn.BCELoss(reduction="mean")
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    losses: List[float] = []
    for images, masks, sizes, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height = int(sizes[0, 0].item())
        width = int(sizes[0, 1].item())
        prediction = final_prediction(model(images))[:, :, :height, :width]
        target = masks[:, :, :height, :width]
        losses.append(float(criterion(prediction.float(), target.float()).item()))
        probabilities.append(prediction[0, 0].float().cpu().numpy())
        targets.append(target[0, 0].float().cpu().numpy())
    return probabilities, targets, losses


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    protocol_path = run_dir / "protocol.json"
    split_path = run_dir / "split.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    protocol = load_json_object(protocol_path)
    split = load_json_object(split_path)
    summary = load_json_object(summary_path)
    assert_finite_numbers(protocol, "protocol")
    assert_finite_numbers(split, "split")
    assert_finite_numbers(summary, "summary")

    if summary.get("status") != "complete":
        raise ValueError(
            f"Run is not complete: summary.status={summary.get('status')!r}"
        )
    protocol_arguments = protocol.get("arguments")
    if not isinstance(protocol_arguments, dict):
        raise ValueError("protocol.json is missing arguments")
    protocol_epochs = protocol_arguments.get("epochs")
    if type(protocol_epochs) is not int or protocol_epochs < 1:
        raise ValueError(f"Invalid protocol epoch count: {protocol_epochs!r}")
    expected_epochs = (
        int(args.expected_epochs)
        if args.expected_epochs is not None
        else int(protocol_epochs)
    )
    if protocol_epochs != expected_epochs:
        raise ValueError(
            f"Protocol epochs={protocol_epochs} does not match "
            f"--expected-epochs={expected_epochs}"
        )
    metric_events = load_complete_metrics(metrics_path, expected_epochs)

    checkpoint_path = (run_dir / args.checkpoint).resolve()
    if checkpoint_path.parent != run_dir:
        raise ValueError("--checkpoint must name a file directly inside --run-dir")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = run_dir / f"pd_fa_sweep_{Path(args.checkpoint).stem}.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing sweep output: {output_path}. "
            "Pass --overwrite explicitly to replace it."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload is not a dictionary")
    dataset = str(
        require_same(
            "dataset",
            {
                "protocol": protocol_arguments.get("dataset"),
                "split": split.get("dataset"),
                "summary": summary.get("dataset"),
                "checkpoint": checkpoint.get("dataset"),
            },
        )
    )
    variant = str(
        require_same(
            "variant",
            {
                "protocol": protocol_arguments.get("variant"),
                "summary": summary.get("variant"),
                "checkpoint": checkpoint.get("variant"),
            },
        )
    )
    seed_value = require_same(
        "training seed",
        {
            "protocol": protocol_arguments.get("seed"),
            "summary": summary.get("seed"),
            "checkpoint": checkpoint.get("seed"),
        },
    )
    if type(seed_value) is not int:
        raise ValueError(f"Training seed is not an integer: {seed_value!r}")
    seed = int(seed_value)
    split_seed = require_same(
        "split seed",
        {
            "protocol": protocol_arguments.get("split_seed"),
            "split": split.get("split_seed"),
            "checkpoint": checkpoint.get("split_seed"),
        },
    )
    if type(split_seed) is not int:
        raise ValueError(f"Split seed is not an integer: {split_seed!r}")
    require_same(
        "validation fraction",
        {
            "protocol": protocol_arguments.get("val_fraction"),
            "split": split.get("val_fraction"),
        },
    )

    for artifact_name, artifact in {
        "protocol": protocol,
        "split": split,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        if artifact.get("official_test_accessed") is not False:
            raise ValueError(
                f"{artifact_name} does not assert official_test_accessed=false"
            )
    for artifact_name, artifact in {
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        if artifact.get("selection_source") != "internal_validation_only":
            raise ValueError(
                f"{artifact_name} was not selected on internal validation only"
            )

    if split.get("source") != f"img_idx/train_{dataset}.txt":
        raise ValueError(
            f"Unexpected split source for internal validation: {split.get('source')!r}"
        )
    expected_run_name = f"seed_{seed}_{protocol_arguments.get('run_tag')}"
    if run_dir.name != expected_run_name:
        raise ValueError(
            f"Run-directory name mismatch: expected {expected_run_name!r}, "
            f"found {run_dir.name!r}"
        )
    if run_dir.parent.name != variant or run_dir.parent.parent.name != dataset:
        raise ValueError(
            "Run-directory dataset/variant components do not match artifact metadata"
        )

    recomputed_split_hashes = validate_identifier_manifest(split)
    split_hashes = require_same(
        "split hashes",
        {
            "split": split.get("hashes"),
            "summary": summary.get("split_hashes"),
            "checkpoint": checkpoint.get("split_hashes"),
            "recomputed": recomputed_split_hashes,
        },
    )
    validation_ids = split["used_val_ids"]
    if split.get("used_val_count") != len(validation_ids):
        raise ValueError("Validation count does not match used_val_ids")

    for event in metric_events:
        if event.get("variant") != variant:
            raise ValueError(
                f"metrics epoch {event['epoch']} variant={event.get('variant')!r} "
                f"does not match {variant!r}"
            )
        require_finite_numeric_fields(
            event,
            ("train_loss", "learning_rate", "processed_train_samples", "epoch_seconds"),
            f"metrics[{event['epoch']}]",
        )

    pd_summary_metrics = json_ready(
        require_same(
            "summary Pd-primary metrics",
            {
                "best_validation_metrics": summary.get("best_validation_metrics"),
                "best_pd_validation_metrics": summary.get(
                    "best_pd_validation_metrics"
                ),
            },
        )
    )
    miou_summary_metrics = json_ready(summary.get("best_miou_validation_metrics"))
    for metric_role, metrics in {
        "Pd-primary": pd_summary_metrics,
        "mIoU-secondary": miou_summary_metrics,
    }.items():
        if not isinstance(metrics, dict) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in metrics.values()
        ):
            raise ValueError(f"Summary {metric_role} metrics must all be numeric")
        assert_finite_numbers(metrics, f"summary.{metric_role}")

    selection_keys = ("pd", "fa", "tiny_pd", "miou", "val_loss")
    evaluated_events = [
        event for event in metric_events if all(key in event for key in selection_keys)
    ]
    if not evaluated_events:
        raise ValueError("Metrics log contains no evaluated validation epochs")
    for event in evaluated_events:
        require_finite_numeric_fields(
            event, selection_keys, f"metrics[{event['epoch']}]"
        )
    recomputed_pd_event = max(evaluated_events, key=pd_primary_selection_key)
    recomputed_miou_event = max(evaluated_events, key=miou_secondary_selection_key)

    require_same(
        "globally recomputed Pd-primary epoch",
        {
            "summary.best_epoch": summary.get("best_epoch"),
            "summary.best_pd_epoch": summary.get("best_pd_epoch"),
            "recomputed_metrics": recomputed_pd_event["epoch"],
        },
    )
    require_same(
        "globally recomputed mIoU-secondary epoch",
        {
            "summary.best_miou_epoch": summary.get("best_miou_epoch"),
            "recomputed_metrics": recomputed_miou_event["epoch"],
        },
    )
    recomputed_pd_metrics = {
        key: recomputed_pd_event.get(key) for key in pd_summary_metrics
    }
    recomputed_miou_metrics = {
        key: recomputed_miou_event.get(key) for key in miou_summary_metrics
    }
    if recomputed_pd_metrics != pd_summary_metrics:
        raise ValueError(
            "Globally recomputed Pd-primary metrics do not match summary metrics"
        )
    if recomputed_miou_metrics != miou_summary_metrics:
        raise ValueError(
            "Globally recomputed mIoU-secondary metrics do not match summary metrics"
        )

    checkpoint_epoch = checkpoint.get("epoch")
    if type(checkpoint_epoch) is not int or not 1 <= checkpoint_epoch <= expected_epochs:
        raise ValueError(f"Invalid checkpoint epoch: {checkpoint_epoch!r}")
    checkpoint_role = checkpoint.get("checkpoint_role")
    checkpoint_name = checkpoint_path.name
    if checkpoint_name == "best.pth.tar":
        if checkpoint_role != "best_validation_pd_primary":
            raise ValueError(
                f"best.pth.tar has invalid checkpoint_role={checkpoint_role!r}"
            )
        require_same(
            "Pd-primary best epoch",
            {
                "summary.best_epoch": summary.get("best_epoch"),
                "summary.best_pd_epoch": summary.get("best_pd_epoch"),
                "checkpoint": checkpoint_epoch,
            },
        )
        summary_metrics = pd_summary_metrics
        if metric_events[checkpoint_epoch - 1].get("new_best_pd") is not True:
            raise ValueError("best.pth.tar epoch is not marked new_best_pd in metrics")
    elif checkpoint_name == "best_miou.pth.tar":
        if checkpoint_role != "best_validation_miou_secondary":
            raise ValueError(
                f"best_miou.pth.tar has invalid checkpoint_role={checkpoint_role!r}"
            )
        require_same(
            "mIoU-secondary best epoch",
            {
                "summary": summary.get("best_miou_epoch"),
                "checkpoint": checkpoint_epoch,
            },
        )
        summary_metrics = miou_summary_metrics
        if metric_events[checkpoint_epoch - 1].get("new_best_miou") is not True:
            raise ValueError(
                "best_miou.pth.tar epoch is not marked new_best_miou in metrics"
            )
    else:
        raise ValueError(
            "Only best.pth.tar or best_miou.pth.tar may be swept by the audited runner"
        )

    checkpoint_metrics = json_ready(checkpoint.get("validation_metrics"))
    summary_metrics = json_ready(summary_metrics)
    if not isinstance(checkpoint_metrics, dict) or not isinstance(summary_metrics, dict):
        raise ValueError("Checkpoint/summary validation metrics are not dictionaries")
    if checkpoint_metrics != summary_metrics:
        raise ValueError("Checkpoint validation metrics do not match summary metrics")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in checkpoint_metrics.values()
    ):
        raise ValueError("Checkpoint validation metrics must all be numeric")
    assert_finite_numbers(checkpoint_metrics, "checkpoint.validation_metrics")
    event_metrics = {
        key: metric_events[checkpoint_epoch - 1].get(key)
        for key in checkpoint_metrics
    }
    if event_metrics != checkpoint_metrics:
        raise ValueError(
            "Checkpoint validation metrics do not match the complete metrics log"
        )
    if int(protocol_arguments.get("eval_every", 0)) == 1:
        for event in metric_events:
            missing_metrics = [key for key in checkpoint_metrics if key not in event]
            if missing_metrics:
                raise ValueError(
                    f"metrics epoch {event['epoch']} is missing validation fields: "
                    f"{missing_metrics}"
                )
            require_finite_numeric_fields(
                event,
                tuple(checkpoint_metrics),
                f"metrics[{event['epoch']}]",
            )

    protocol_match_radius = float(protocol_arguments["match_radius"])
    protocol_tiny_area = int(protocol_arguments["tiny_area"])
    match_radius = (
        float(args.match_radius)
        if args.match_radius is not None
        else protocol_match_radius
    )
    tiny_area = (
        int(args.tiny_area) if args.tiny_area is not None else protocol_tiny_area
    )
    if match_radius != protocol_match_radius or tiny_area != protocol_tiny_area:
        raise ValueError(
            "Formal sweep match-radius/tiny-area overrides must equal the run protocol"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, _ = build_model(variant, seed)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)

    dataset_dir = Path(protocol_arguments["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    validation_set = ValidationSubset(
        dataset_dir / dataset,
        validation_ids,
        {key: float(value) for key, value in protocol["normalization"].items()},
    )
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
    probabilities, targets, losses = collect_predictions(model, validation_loader, device)
    if len(probabilities) != len(validation_ids) or len(targets) != len(validation_ids):
        raise ValueError("Prediction count does not match the audited validation split")
    for index, (probability, target, loss) in enumerate(
        zip(probabilities, targets, losses)
    ):
        if not np.isfinite(probability).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite prediction/target values at validation index {index}")
        if not math.isfinite(float(loss)):
            raise ValueError(f"Non-finite validation loss at index {index}")

    base_thresholds = threshold_grid(
        args.threshold_min, args.threshold_max, args.threshold_step, args.extra_thresholds
    )
    thresholds, threshold_provenance = adaptive_thresholds(
        probabilities, base_thresholds, args.tail_logit_step
    )
    points: List[Dict[str, Any]] = []
    for threshold in thresholds:
        accumulator = ValidationMetrics(threshold, match_radius, tiny_area)
        for probability, target, loss in zip(probabilities, targets, losses):
            accumulator.update(probability, target, loss)
        point = accumulator.compute()
        point["threshold"] = threshold
        ready_point = json_ready(point)
        assert_finite_numbers(ready_point, f"sweep threshold {threshold}")
        points.append(ready_point)

    fixed_half = min(points, key=lambda point: abs(float(point["threshold"]) - 0.5))
    fixed_half_checkpoint_audit = audit_fixed_threshold_checkpoint(
        fixed_half, checkpoint_metrics
    )
    budget_points = {
        f"{budget:.10g}": best_point_under_fa(points, budget) for budget in args.fa_budgets
    }
    artifact_hashes = {
        "protocol.json": file_sha256(protocol_path),
        "split.json": file_sha256(split_path),
        "summary.json": file_sha256(summary_path),
        "metrics.jsonl": file_sha256(metrics_path),
        "checkpoint": file_sha256(checkpoint_path),
        "evaluator": file_sha256(Path(__file__).resolve()),
    }
    output = {
        "run_directory": run_dir,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": artifact_hashes["checkpoint"],
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_role": checkpoint_role,
        "checkpoint_validation_metrics": checkpoint_metrics,
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "split_seed": split_seed,
        "validation_count": len(validation_set),
        "validation_split_sha256": split_hashes["used_val_sha256"],
        "official_test_accessed": False,
        "match_radius": match_radius,
        "tiny_area": tiny_area,
        "threshold_configuration": {
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "extra_thresholds": args.extra_thresholds,
            "tail_logit_step": args.tail_logit_step,
            "fa_budgets": args.fa_budgets,
        },
        "threshold_provenance": threshold_provenance,
        "fixed_threshold_0_5": fixed_half,
        "fixed_threshold_0_5_checkpoint_audit": fixed_half_checkpoint_audit,
        "best_points_under_fa_budget": budget_points,
        "points": points,
        "audit": {
            "invocation_argv": [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            "parsed_arguments": vars(args),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expected_epochs": expected_epochs,
            "metrics_event_count": len(metric_events),
            "metrics_epoch_range": [
                metric_events[0]["epoch"],
                metric_events[-1]["epoch"],
            ],
            "summary_status": summary["status"],
            "selection_source": checkpoint["selection_source"],
            "integrity_checks_passed": {
                "summary_complete": True,
                "metrics_complete_contiguous_finite": True,
                "metadata_consistent": True,
                "official_test_isolated": True,
                "split_hashes_recomputed_consistent": True,
                "checkpoint_role_epoch_metrics_consistent": True,
                "global_selection_keys_recomputed": True,
                "state_dict_strict_load": True,
                "fixed_threshold_object_metrics_exact": True,
            },
            "artifact_sha256": artifact_hashes,
            "protocol": {
                "arguments": protocol_arguments,
                "primary_selection_rule": protocol.get("primary_selection_rule"),
                "secondary_selection_rule": protocol.get("secondary_selection_rule"),
                "checkpoint_policy": protocol.get("checkpoint_policy"),
                "metric_notes": protocol.get("metric_notes"),
            },
            "globally_recomputed_selection": {
                "pd_primary": {
                    "epoch": recomputed_pd_event["epoch"],
                    "key": pd_primary_selection_key(recomputed_pd_event),
                    "metrics": recomputed_pd_metrics,
                },
                "miou_secondary": {
                    "epoch": recomputed_miou_event["epoch"],
                    "key": miou_secondary_selection_key(recomputed_miou_event),
                    "metrics": recomputed_miou_metrics,
                },
            },
        },
    }
    write_output_json(output_path, output, args.overwrite)
    print(
        f"COMPLETE variant={variant} epoch={checkpoint_epoch} "
        f"Pd@0.5={fixed_half['pd']:.6f} Fa@0.5={fixed_half['fa']:.8f} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
