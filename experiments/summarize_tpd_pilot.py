#!/usr/bin/env python3
"""Audit completed TPD experiment runs and create a comparison table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.Config import get_SCTrans_config  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402
from model.tpd import replace_shallow_embeddings  # noqa: E402


VARIANTS = ("original", "progressive", "spd", "tpd")

VALIDATION_METRIC_KEYS = (
    "val_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
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
    "valid_pixel_count",
)

PRIMARY_SELECTION_RULE = (
    "maximum val Pd",
    "minimum val Fa on Pd ties",
    "maximum val tiny-Pd",
    "maximum val mIoU",
    "minimum val loss",
)

SECONDARY_SELECTION_RULE = (
    "maximum val mIoU",
    "maximum val Pd",
    "minimum val Fa",
    "maximum val tiny-Pd",
    "minimum val loss",
)

CRITICAL_PROTOCOL_ARGUMENTS = (
    "dataset",
    "dataset_dir",
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

SPLIT_HASH_KEYS = (
    "full_internal_train_sha256",
    "full_internal_val_sha256",
    "used_train_sha256",
    "used_val_sha256",
)

SPLIT_ID_CONTRACT = (
    ("full_internal_train_ids", "full_internal_train_count", "full_internal_train_sha256"),
    ("full_internal_val_ids", "full_internal_val_count", "full_internal_val_sha256"),
    ("used_train_ids", "used_train_count", "used_train_sha256"),
    ("used_val_ids", "used_val_count", "used_val_sha256"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and summarize completed TPD runs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--run-name", default="seed_42_pilot_pd_fp32_v1")
    parser.add_argument(
        "--expected-epochs",
        type=int,
        required=True,
        help="Require exactly this many consecutive epochs in every metrics.jsonl",
    )
    parser.add_argument(
        "--report-title",
        default=None,
        help="Markdown report title (defaults to a dataset-specific generic title)",
    )
    parser.add_argument(
        "--variant-run-name",
        action="append",
        default=[],
        metavar="VARIANT=RUN_NAME",
        help="Override the run directory name for one variant (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.expected_epochs < 1:
        parser.error("--expected-epochs must be >= 1")
    return args


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def require_mapping(payload: Dict[str, Any], key: str, context: str) -> Dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def assert_finite_numbers(value: Any, context: str) -> None:
    """Reject NaN/Inf anywhere in an audited JSON-compatible payload."""

    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {context}: {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_numbers(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_numbers(item, f"{context}[{index}]")


def require_finite_metric_dict(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    expected = set(VALIDATION_METRIC_KEYS)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} metric keys differ: missing={missing}, extra={extra}")
    for key in VALIDATION_METRIC_KEYS:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{context}.{key} must be numeric, got {item!r}")
        if not math.isfinite(float(item)):
            raise ValueError(f"{context}.{key} is non-finite: {item!r}")
    return value


def assert_exact(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch: expected {expected!r}, got {actual!r}")


def assert_metrics_equal(actual: Any, expected: Dict[str, Any], context: str) -> None:
    actual_metrics = require_finite_metric_dict(actual, context)
    for key in VALIDATION_METRIC_KEYS:
        assert_exact(actual_metrics[key], expected[key], f"{context}.{key}")


def validation_metrics_from_event(event: Dict[str, Any], context: str) -> Dict[str, Any] | None:
    present = [key for key in VALIDATION_METRIC_KEYS if key in event]
    if not present:
        return None
    if len(present) != len(VALIDATION_METRIC_KEYS):
        missing = sorted(set(VALIDATION_METRIC_KEYS) - set(present))
        raise ValueError(f"Partial validation metrics at {context}: missing={missing}")
    return require_finite_metric_dict(
        {key: event[key] for key in VALIDATION_METRIC_KEYS}, context
    )


def load_metrics(
    path: Path,
    expected_epochs: int,
    variant: str,
    eval_every: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics log: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_epochs:
        raise ValueError(
            f"{path} has {len(lines)} rows; expected exactly {expected_epochs}"
        )

    evaluated: List[Tuple[int, Dict[str, Any]]] = []
    for expected_epoch, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Blank metrics row at {path}:{expected_epoch}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{expected_epoch}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"Metrics row must be an object at {path}:{expected_epoch}")
        assert_finite_numbers(event, f"{path}:{expected_epoch}")
        event_epoch = event.get("epoch")
        if isinstance(event_epoch, bool) or not isinstance(event_epoch, int):
            raise ValueError(f"{path}:{expected_epoch}.epoch must be an integer")
        assert_exact(event_epoch, expected_epoch, f"{path}:{expected_epoch}.epoch")
        assert_exact(event.get("variant"), variant, f"{path}:{expected_epoch}.variant")
        processed_samples = event.get("processed_train_samples")
        if (
            isinstance(processed_samples, bool)
            or not isinstance(processed_samples, int)
            or processed_samples < 1
        ):
            raise ValueError(
                f"{path}:{expected_epoch}.processed_train_samples must be a positive integer"
            )
        for key in ("train_loss", "learning_rate", "epoch_seconds"):
            item = event.get(key)
            if (
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
            ):
                raise ValueError(f"{path}:{expected_epoch}.{key} must be finite numeric")
        metrics = validation_metrics_from_event(event, f"{path}:{expected_epoch}")
        if metrics is not None:
            evaluated.append((expected_epoch, metrics))

    expected_evaluated_epochs = [
        epoch
        for epoch in range(1, expected_epochs + 1)
        if epoch == 1 or epoch % eval_every == 0 or epoch == expected_epochs
    ]
    actual_evaluated_epochs = [epoch for epoch, _ in evaluated]
    assert_exact(
        actual_evaluated_epochs,
        expected_evaluated_epochs,
        f"{path} evaluated epochs",
    )
    return evaluated


def pd_selection_key(metrics: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    tiny_pd = float(metrics["tiny_pd"])
    if not math.isfinite(tiny_pd):
        tiny_pd = -1.0
    return (
        float(metrics["pd"]),
        -float(metrics["fa"]),
        tiny_pd,
        float(metrics["miou"]),
        -float(metrics["val_loss"]),
    )


def miou_selection_key(metrics: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    tiny_pd = float(metrics["tiny_pd"])
    if not math.isfinite(tiny_pd):
        tiny_pd = -1.0
    return (
        float(metrics["miou"]),
        float(metrics["pd"]),
        -float(metrics["fa"]),
        tiny_pd,
        -float(metrics["val_loss"]),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256_mapping(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    if set(value) != set(SPLIT_HASH_KEYS):
        missing = sorted(set(SPLIT_HASH_KEYS) - set(value))
        extra = sorted(set(value) - set(SPLIT_HASH_KEYS))
        raise ValueError(f"{context} hash keys differ: missing={missing}, extra={extra}")
    for key in SPLIT_HASH_KEYS:
        digest = value[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{context}.{key} is not a lowercase SHA-256 digest")
    return value


def identifier_hash(identifiers: Sequence[str]) -> str:
    canonical_ids = "\n".join(sorted(identifiers)).encode("utf-8")
    return hashlib.sha256(canonical_ids).hexdigest()


def require_identifier_list(split: Dict[str, Any], key: str, context: str) -> List[str]:
    identifiers = split.get(key)
    if not isinstance(identifiers, list):
        raise ValueError(f"{context}.{key} must be a list")
    for index, identifier in enumerate(identifiers):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{context}.{key}[{index}] must be a non-empty string")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{context}.{key} contains duplicate identifiers")
    return identifiers


def audit_split_manifest(
    split: Dict[str, Any], expected_hashes: Dict[str, Any], context: str
) -> Dict[str, List[str]]:
    identifier_lists: Dict[str, List[str]] = {}
    recomputed_hashes: Dict[str, str] = {}
    for ids_key, count_key, hash_key in SPLIT_ID_CONTRACT:
        identifiers = require_identifier_list(split, ids_key, context)
        count = split.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{context}.{count_key} must be a positive integer")
        assert_exact(count, len(identifiers), f"{context}.{count_key}")
        identifier_lists[ids_key] = identifiers
        recomputed_hashes[hash_key] = identifier_hash(identifiers)

    assert_exact(recomputed_hashes, expected_hashes, f"{context} recomputed hashes")

    full_train = set(identifier_lists["full_internal_train_ids"])
    full_val = set(identifier_lists["full_internal_val_ids"])
    used_train = set(identifier_lists["used_train_ids"])
    used_val = set(identifier_lists["used_val_ids"])
    if full_train & full_val:
        raise ValueError(f"{context} full internal train/validation IDs overlap")
    if used_train & used_val:
        raise ValueError(f"{context} used train/validation IDs overlap")
    if not used_train <= full_train:
        raise ValueError(f"{context}.used_train_ids is not a subset of full train IDs")
    if not used_val <= full_val:
        raise ValueError(f"{context}.used_val_ids is not a subset of full validation IDs")

    official_count = split.get("full_official_train_count")
    if (
        isinstance(official_count, bool)
        or not isinstance(official_count, int)
        or official_count < 2
    ):
        raise ValueError(f"{context}.full_official_train_count must be an integer >= 2")
    assert_exact(
        official_count,
        len(full_train) + len(full_val),
        f"{context}.full_official_train_count",
    )
    return identifier_lists


def build_model_for_strict_load(variant: str) -> SCTransNet:
    model = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
    replace_shallow_embeddings(model, variant)
    return model


def audit_checkpoint(
    path: Path,
    model: SCTransNet,
    role: str,
    epoch: int,
    metrics: Dict[str, Any],
    variant: str,
    dataset: str,
    seed: int,
    split_seed: int,
    split_hashes: Dict[str, Any],
    model_metadata: Dict[str, Any],
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checksum = sha256_file(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a mapping: {path}")

    assert_exact(checkpoint.get("checkpoint_role"), role, f"{path}.checkpoint_role")
    assert_exact(checkpoint.get("epoch"), epoch, f"{path}.epoch")
    assert_exact(checkpoint.get("variant"), variant, f"{path}.variant")
    assert_exact(checkpoint.get("dataset"), dataset, f"{path}.dataset")
    assert_exact(checkpoint.get("seed"), seed, f"{path}.seed")
    assert_exact(checkpoint.get("split_seed"), split_seed, f"{path}.split_seed")
    assert_exact(
        checkpoint.get("official_test_accessed"), False, f"{path}.official_test_accessed"
    )
    assert_exact(
        checkpoint.get("selection_source"),
        "internal_validation_only",
        f"{path}.selection_source",
    )
    assert_exact(checkpoint.get("split_hashes"), split_hashes, f"{path}.split_hashes")
    assert_exact(checkpoint.get("model_metadata"), model_metadata, f"{path}.model_metadata")
    assert_metrics_equal(checkpoint.get("validation_metrics"), metrics, f"{path}.metrics")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path}.state_dict must be a mapping")
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path}.state_dict must map string keys to tensors")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Non-finite floating tensor in {path}.state_dict[{name!r}]")
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError(f"Strict state_dict load failed for {path}: {exc}") from exc
    return checksum


def require_protocol_arguments(protocol: Dict[str, Any], context: str) -> Dict[str, Any]:
    arguments = require_mapping(protocol, "arguments", context)
    missing = [key for key in CRITICAL_PROTOCOL_ARGUMENTS if key not in arguments]
    if missing:
        raise ValueError(f"{context}.arguments missing critical keys: {missing}")
    assert_finite_numbers(arguments, f"{context}.arguments")
    return arguments


def load_run(
    run_dir: Path,
    expected_epochs: int,
    expected_variant: str,
    expected_dataset: str,
) -> Dict[str, Any]:
    required_paths = {
        "summary": run_dir / "summary.json",
        "protocol": run_dir / "protocol.json",
        "split": run_dir / "split.json",
        "metrics": run_dir / "metrics.jsonl",
        "best": run_dir / "best.pth.tar",
        "best_miou": run_dir / "best_miou.pth.tar",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}; missing={missing}")

    summary = read_json(required_paths["summary"])
    protocol = read_json(required_paths["protocol"])
    split = read_json(required_paths["split"])
    assert_finite_numbers(summary, f"{required_paths['summary']}")
    assert_finite_numbers(protocol, f"{required_paths['protocol']}")
    assert_finite_numbers(split, f"{required_paths['split']}")

    assert_exact(summary.get("status"), "complete", f"{run_dir}.summary.status")
    assert_exact(summary.get("variant"), expected_variant, f"{run_dir}.summary.variant")
    assert_exact(summary.get("dataset"), expected_dataset, f"{run_dir}.summary.dataset")
    if isinstance(summary.get("seed"), bool) or not isinstance(summary.get("seed"), int):
        raise ValueError(f"{run_dir}.summary.seed must be an integer")
    seed = int(summary["seed"])
    assert_exact(
        summary.get("official_test_accessed"), False, f"{run_dir}.summary.test isolation"
    )
    assert_exact(
        summary.get("selection_source"),
        "internal_validation_only",
        f"{run_dir}.summary.selection_source",
    )
    assert_exact(
        summary.get("primary_selection_metric"),
        "validation Pd, then lower Fa",
        f"{run_dir}.summary.primary_selection_metric",
    )

    arguments = require_protocol_arguments(protocol, f"{run_dir}.protocol")
    assert_exact(arguments.get("variant"), expected_variant, f"{run_dir}.protocol.variant")
    assert_exact(arguments.get("dataset"), expected_dataset, f"{run_dir}.protocol.dataset")
    assert_exact(arguments.get("seed"), seed, f"{run_dir}.protocol.seed")
    assert_exact(arguments.get("epochs"), expected_epochs, f"{run_dir}.protocol.epochs")
    run_tag = arguments.get("run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError(f"{run_dir}.protocol.run_tag must be a non-empty string")
    assert_exact(run_dir.name, f"seed_{seed}_{run_tag}", f"{run_dir}.run name")
    eval_every = arguments.get("eval_every")
    if isinstance(eval_every, bool) or not isinstance(eval_every, int) or eval_every < 1:
        raise ValueError(f"{run_dir}.protocol.eval_every must be a positive integer")
    assert_exact(
        protocol.get("official_test_accessed"), False, f"{run_dir}.protocol.test isolation"
    )
    assert_exact(
        protocol.get("primary_selection_rule"),
        list(PRIMARY_SELECTION_RULE),
        f"{run_dir}.protocol.primary_selection_rule",
    )
    assert_exact(
        protocol.get("secondary_selection_rule"),
        list(SECONDARY_SELECTION_RULE),
        f"{run_dir}.protocol.secondary_selection_rule",
    )

    normalization = require_mapping(protocol, "normalization", f"{run_dir}.protocol")
    if set(normalization) != {"mean", "std"}:
        raise ValueError(f"{run_dir}.protocol.normalization must contain exactly mean/std")
    for key in ("mean", "std"):
        value = normalization[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{run_dir}.protocol.normalization.{key} must be finite")
    if float(normalization["std"]) <= 0.0:
        raise ValueError(f"{run_dir}.protocol.normalization.std must be positive")

    split_hashes = require_sha256_mapping(
        summary.get("split_hashes"), f"{run_dir}.summary.split_hashes"
    )
    split_file_hashes = require_sha256_mapping(
        split.get("hashes"), f"{run_dir}.split.hashes"
    )
    assert_exact(split.get("dataset"), expected_dataset, f"{run_dir}.split.dataset")
    assert_exact(split.get("split_seed"), arguments["split_seed"], f"{run_dir}.split.seed")
    assert_exact(
        split.get("official_test_accessed"), False, f"{run_dir}.split.test isolation"
    )
    assert_exact(split_file_hashes, split_hashes, f"{run_dir}.split.hashes")
    split_identifier_lists = audit_split_manifest(
        split, split_hashes, f"{run_dir}.split"
    )

    evaluated = load_metrics(
        required_paths["metrics"], expected_epochs, expected_variant, eval_every
    )
    best_pd_epoch, best_pd_metrics = max(evaluated, key=lambda item: pd_selection_key(item[1]))
    best_miou_epoch, best_miou_metrics = max(
        evaluated, key=lambda item: miou_selection_key(item[1])
    )

    assert_exact(summary.get("best_epoch"), best_pd_epoch, f"{run_dir}.summary.best_epoch")
    assert_exact(summary.get("best_pd_epoch"), best_pd_epoch, f"{run_dir}.summary.best_pd_epoch")
    assert_exact(
        summary.get("best_miou_epoch"), best_miou_epoch, f"{run_dir}.summary.best_miou_epoch"
    )
    assert_metrics_equal(
        summary.get("best_validation_metrics"), best_pd_metrics, f"{run_dir}.summary.best_metrics"
    )
    assert_metrics_equal(
        summary.get("best_pd_validation_metrics"),
        best_pd_metrics,
        f"{run_dir}.summary.best_pd_metrics",
    )
    assert_metrics_equal(
        summary.get("best_miou_validation_metrics"),
        best_miou_metrics,
        f"{run_dir}.summary.best_miou_metrics",
    )

    model_metadata = require_mapping(summary, "model", f"{run_dir}.summary")
    assert_exact(model_metadata.get("variant"), expected_variant, f"{run_dir}.model.variant")
    protocol_model = require_mapping(protocol, "model", f"{run_dir}.protocol")
    assert_exact(protocol_model, model_metadata, f"{run_dir}.protocol.model")
    model = build_model_for_strict_load(expected_variant)
    best_sha256 = audit_checkpoint(
        required_paths["best"],
        model,
        "best_validation_pd_primary",
        best_pd_epoch,
        best_pd_metrics,
        expected_variant,
        expected_dataset,
        seed,
        int(arguments["split_seed"]),
        split_hashes,
        model_metadata,
    )
    best_miou_sha256 = audit_checkpoint(
        required_paths["best_miou"],
        model,
        "best_validation_miou_secondary",
        best_miou_epoch,
        best_miou_metrics,
        expected_variant,
        expected_dataset,
        seed,
        int(arguments["split_seed"]),
        split_hashes,
        model_metadata,
    )

    expected_checkpoint_paths = {
        "best_checkpoint": required_paths["best"],
        "best_miou_checkpoint": required_paths["best_miou"],
    }
    for key, expected_path in expected_checkpoint_paths.items():
        summary_path = summary.get(key)
        if not isinstance(summary_path, str):
            raise ValueError(f"{run_dir}.summary.{key} must be a path string")
        assert_exact(
            Path(summary_path).resolve(), expected_path.resolve(), f"{run_dir}.summary.{key}"
        )

    return {
        "summary": summary,
        "protocol": protocol,
        "split": split,
        "arguments": arguments,
        "normalization": normalization,
        "split_identifier_lists": split_identifier_lists,
        "run_dir": run_dir,
        "best_checkpoint_sha256": best_sha256,
        "best_miou_checkpoint_sha256": best_miou_sha256,
    }


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def common_value(runs: Dict[str, Dict[str, Any]], label: str, getter: Any) -> Any:
    values = {variant: getter(run) for variant, run in runs.items()}
    reference = values[VARIANTS[0]]
    mismatched = {
        variant: value
        for variant, value in values.items()
        if canonical(value) != canonical(reference)
    }
    if mismatched:
        raise ValueError(f"{label} differ across variants: {values!r}")
    return reference


def audit_cross_variant_consistency(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    seed = common_value(runs, "Seeds", lambda run: run["summary"]["seed"])
    split_hashes = common_value(
        runs,
        "Train/validation split hashes",
        lambda run: run["summary"]["split_hashes"],
    )
    shared_hash = common_value(
        runs,
        "Shared initialization hashes",
        lambda run: run["summary"]["model"]["shared_initialization_sha256"],
    )
    normalization = common_value(
        runs, "Training-only normalization", lambda run: run["normalization"]
    )
    assert_finite_numbers(normalization, "cross-run normalization")
    critical_arguments = common_value(
        runs,
        "Critical protocol arguments",
        lambda run: {key: run["arguments"][key] for key in CRITICAL_PROTOCOL_ARGUMENTS},
    )
    protocol_contract = common_value(
        runs,
        "Protocol contracts",
        lambda run: {
            "primary_selection_rule": run["protocol"].get("primary_selection_rule"),
            "secondary_selection_rule": run["protocol"].get("secondary_selection_rule"),
            "checkpoint_policy": run["protocol"].get("checkpoint_policy"),
            "loss": run["protocol"].get("loss"),
            "optimizer": run["protocol"].get("optimizer"),
            "lr_schedule": run["protocol"].get("lr_schedule"),
            "torch": require_mapping(run["protocol"], "environment", "protocol").get("torch"),
            "cuda_runtime": require_mapping(
                run["protocol"], "environment", "protocol"
            ).get("cuda_runtime"),
            "device_name": require_mapping(
                run["protocol"], "environment", "protocol"
            ).get("device_name"),
        },
    )
    split_counts = common_value(
        runs,
        "Split counts",
        lambda run: {
            key: run["split"].get(key)
            for key in (
                "full_official_train_count",
                "full_internal_train_count",
                "full_internal_val_count",
                "used_train_count",
                "used_val_count",
            )
        },
    )
    return {
        "seed": seed,
        "split_hashes": split_hashes,
        "shared_initialization_sha256": shared_hash,
        "normalization": normalization,
        "critical_protocol_arguments": critical_arguments,
        "protocol_contract": protocol_contract,
        "split_counts": split_counts,
    }


def metric_row(variant: str, run: Dict[str, Any]) -> Dict[str, Any]:
    summary = run["summary"]
    pd_metrics = summary["best_pd_validation_metrics"]
    miou_metrics = summary["best_miou_validation_metrics"]
    model = summary["model"]
    return {
        "variant": variant,
        "seed": summary["seed"],
        "pd_best_epoch": summary["best_pd_epoch"],
        "pd": pd_metrics["pd"],
        "tiny_pd": pd_metrics["tiny_pd"],
        "fa": pd_metrics["fa"],
        "false_objects_per_image": pd_metrics["false_objects_per_image"],
        "miou_at_pd_best": pd_metrics["miou"],
        "niou_at_pd_best": pd_metrics["niou"],
        "f1_at_pd_best": pd_metrics["pixel_f1"],
        "miou_best_epoch": summary["best_miou_epoch"],
        "best_miou": miou_metrics["miou"],
        "pd_at_miou_best": miou_metrics["pd"],
        "fa_at_miou_best": miou_metrics["fa"],
        "parameters": model["total_parameters"],
        "shallow_parameters": model["shallow_embedding_parameters"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "best_checkpoint_sha256": run["best_checkpoint_sha256"],
        "best_miou_checkpoint_sha256": run["best_miou_checkpoint_sha256"],
        "run_dir": str(run["run_dir"]),
    }


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(
    rows: List[Dict[str, Any]], dataset: str, report_title: str | None = None
) -> str:
    title = report_title or f"{dataset} TPD-PE comparison (internal validation only)"
    lines = [
        f"# {title}",
        "",
        "Primary checkpoint: maximum validation Pd; ties use lower Fa, higher tiny-Pd, "
        "higher mIoU, then lower validation loss.",
        "",
        "| Variant | Epoch | Pd ↑ | tiny-Pd ↑ | Fa ↓ | False obj/img ↓ | "
        "mIoU@Pd-best ↑ | nIoU ↑ | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {epoch} | {pd} | {tiny_pd} | {fa} | {false_obj} | "
            "{miou} | {niou} | {parameters:,} |".format(
                variant=row["variant"],
                epoch=row["pd_best_epoch"],
                pd=format_float(row["pd"]),
                tiny_pd=format_float(row["tiny_pd"]),
                fa=format_float(row["fa"], 8),
                false_obj=format_float(row["false_objects_per_image"], 4),
                miou=format_float(row["miou_at_pd_best"]),
                niou=format_float(row["niou_at_pd_best"]),
                parameters=int(row["parameters"]),
            )
        )
    lines += [
        "",
        "Secondary mIoU-selected checkpoint (analysis only):",
        "",
        "| Variant | Epoch | best mIoU ↑ | Pd | Fa ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['miou_best_epoch']} | "
            f"{format_float(row['best_miou'])} | {format_float(row['pd_at_miou_best'])} | "
            f"{format_float(row['fa_at_miou_best'], 8)} |"
        )
    return "\n".join(lines) + "\n"


def parse_run_names(run_name: str, assignments: Sequence[str]) -> Dict[str, str]:
    run_names = {variant: run_name for variant in VARIANTS}
    seen = set()
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected VARIANT=RUN_NAME, got {assignment!r}")
        variant, variant_run_name = assignment.split("=", 1)
        if variant not in run_names or not variant_run_name or variant in seen:
            raise ValueError(f"Invalid variant run-name override: {assignment!r}")
        seen.add(variant)
        run_names[variant] = variant_run_name
    return run_names


def main() -> None:
    args = parse_args()
    run_names = parse_run_names(args.run_name, args.variant_run_name)
    runs = {
        variant: load_run(
            args.root.resolve() / args.dataset / variant / run_names[variant],
            args.expected_epochs,
            variant,
            args.dataset,
        )
        for variant in VARIANTS
    }
    consistency = audit_cross_variant_consistency(runs)

    rows = [metric_row(variant, runs[variant]) for variant in VARIANTS]
    baseline = rows[0]
    for row in rows:
        row["delta_pd_vs_original"] = float(row["pd"]) - float(baseline["pd"])
        row["delta_tiny_pd_vs_original"] = float(row["tiny_pd"]) - float(
            baseline["tiny_pd"]
        )
        row["delta_fa_vs_original"] = float(row["fa"]) - float(baseline["fa"])
        row["delta_miou_at_pd_best_vs_original"] = float(row["miou_at_pd_best"]) - float(
            baseline["miou_at_pd_best"]
        )

    output_dir = args.output_dir or args.root.resolve() / args.dataset / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_name
    checkpoint_sha256 = {
        variant: {
            "best.pth.tar": runs[variant]["best_checkpoint_sha256"],
            "best_miou.pth.tar": runs[variant]["best_miou_checkpoint_sha256"],
        }
        for variant in VARIANTS
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "run_name": args.run_name,
                "expected_epochs": args.expected_epochs,
                "report_title": args.report_title,
                "variant_run_names": run_names,
                "official_test_accessed": False,
                "validation_split_sha256": consistency["split_hashes"][
                    "used_val_sha256"
                ],
                "training_split_sha256": consistency["split_hashes"][
                    "used_train_sha256"
                ],
                "shared_initialization_sha256": consistency[
                    "shared_initialization_sha256"
                ],
                "checkpoint_sha256": checkpoint_sha256,
                "integrity_audit": consistency,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{stem}.md").write_text(
        markdown_table(rows, args.dataset, args.report_title), encoding="utf-8"
    )
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output_dir / stem}.[json|md|csv]")


if __name__ == "__main__":
    main()
