#!/usr/bin/env python3
"""Audit and freeze the 16 selected checkpoints for four seed-42 run pairs.

The trainer keeps only online ``best_miou`` and ``best_pd`` as formal model
checkpoints.  A single overwritten exact-resume state exists only while a run
is active.  This selector independently recomputes both best epochs from the
complete epoch-10,20,...,1000 test metric log, verifies the two selected payloads,
and copies immutable, SHA-bound snapshots into ``selected_checkpoints``.
Epoch 1000 remains a metric-only fixed endpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import four_dataset_evaluation_protocol_v1 as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=protocol.RUNS_ROOT,
    )
    parser.add_argument(
        "--selected-root",
        type=Path,
        default=protocol.SELECTED_ROOT,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=protocol.SELECTED_ROOT / "checkpoint_manifest.json",
    )
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--method", choices=protocol.METHODS)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Require and freeze all 4 datasets x 2 methods.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a differing frozen copy/manifest explicitly.",
    )
    args = parser.parse_args(argv)
    if args.all_runs:
        if args.dataset is not None or args.method is not None:
            parser.error("--all-runs cannot be combined with --dataset/--method")
    elif args.dataset is None or args.method is None:
        parser.error("use --all-runs or provide both --dataset and --method")
    return args


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {path}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"checkpoint lacks a non-empty state_dict: {path}")
    return payload


def _metadata_value(
    payload: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    location: str,
) -> Any:
    values = [(name, payload[name]) for name in aliases if name in payload]
    if not values:
        raise ValueError(f"{location} lacks any of {tuple(aliases)!r}")
    first_name, expected = values[0]
    for name, value in values[1:]:
        if protocol.canonical_json_bytes(value) != protocol.canonical_json_bytes(
            expected
        ):
            raise ValueError(
                f"{location} metadata differs: "
                f"{first_name}={expected!r}, {name}={value!r}"
            )
    return expected


def _checkpoint_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("test_metrics", "evaluation_metrics", "validation_metrics"):
        metrics = payload.get(key)
        if isinstance(metrics, Mapping):
            event = {"epoch": payload.get("epoch"), "test_metrics": dict(metrics)}
            if "threshold" not in metrics:
                event["threshold"] = payload.get(
                    "threshold",
                    protocol.FIXED_THRESHOLD,
                )
            return protocol.normalize_metric_event(event)
    raise ValueError("checkpoint lacks test/evaluation metrics")


def _require_checkpoint_matches(
    payload: Mapping[str, Any],
    *,
    path: Path,
    dataset: str,
    method: str,
    role: str,
    expected_metrics: Mapping[str, Any],
) -> None:
    observed_dataset = _metadata_value(
        payload,
        ("dataset", "dataset_name", "training_regime"),
        location=str(path),
    )
    observed_method = _metadata_value(
        payload,
        ("method", "variant"),
        location=str(path),
    )
    protocol.require(
        observed_dataset == dataset,
        f"{path}: dataset differs: {observed_dataset!r}",
    )
    protocol.require(
        observed_method == method,
        f"{path}: method differs: {observed_method!r}",
    )
    protocol.require(
        payload.get("seed") == protocol.TRAINING_SEED,
        f"{path}: seed must be {protocol.TRAINING_SEED}",
    )
    protocol.require(
        payload.get("epoch") == expected_metrics["epoch"],
        f"{path}: epoch differs from recomputed {role}",
    )
    observed_role = payload.get("checkpoint_role", payload.get("role"))
    accepted_roles = {
        role,
        f"test_selected_{role}",
        "best_test_miou" if role == "best_miou" else "best_test_pd",
    }
    protocol.require(
        observed_role in accepted_roles,
        f"{path}: checkpoint_role={observed_role!r} is invalid for {role}",
    )
    observed_metrics = _checkpoint_metrics(payload)
    for key, expected in expected_metrics.items():
        if key not in observed_metrics:
            continue
        protocol.require(
            protocol.canonical_json_bytes(observed_metrics[key])
            == protocol.canonical_json_bytes(expected),
            f"{path}: checkpoint metric {key} differs from metrics.jsonl",
        )
    for field, expected in (
        ("test_selected", True),
        ("selection_is_optimistic", True),
    ):
        if field in payload:
            protocol.require(
                payload[field] is expected,
                f"{path}: {field} must be {expected}",
            )


def _protocol_audit(run_dir: Path, dataset: str, method: str) -> dict[str, Any]:
    path = run_dir / "protocol.json"
    payload = protocol.load_json_object(path)
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = payload
    metric_arguments = payload.get("metrics")
    if not isinstance(metric_arguments, Mapping):
        metric_arguments = arguments
    expected = {
        "dataset": dataset,
        "method": method,
        "seed": protocol.TRAINING_SEED,
        "epochs": protocol.EXPECTED_EPOCHS,
        "begin_test": protocol.CANDIDATE_FIRST_EPOCH,
        "eval_every": protocol.CANDIDATE_EVAL_EVERY,
        "threshold": protocol.FIXED_THRESHOLD,
        "match_radius": protocol.MATCH_RADIUS,
        "tiny_area": protocol.TINY_AREA,
    }
    aliases = {
        "dataset": ("dataset", "dataset_name", "training_regime"),
        "method": ("method", "variant"),
        "seed": ("seed", "training_seed"),
        "epochs": ("epochs",),
        "begin_test": ("begin_test", "begin_test_epoch"),
        "eval_every": ("eval_every",),
        "threshold": ("threshold", "selection_threshold"),
        "match_radius": ("match_radius",),
        "tiny_area": ("tiny_area",),
    }
    observed: dict[str, Any] = {}
    for field, expected_value in expected.items():
        container = (
            metric_arguments
            if field in {"threshold", "match_radius", "tiny_area"}
            else arguments
        )
        observed[field] = _metadata_value(
            container,
            aliases[field],
            location=f"{path}.arguments",
        )
        protocol.require(
            observed[field] == expected_value,
            f"{path}: {field} differs: expected={expected_value!r}, "
            f"observed={observed[field]!r}",
        )
    fresh = arguments.get(
        "fresh_scratch",
        arguments.get(
            "scratch",
            arguments.get("initialization_mode") == "true_scratch",
        ),
    )
    protocol.require(fresh is True, f"{path}: run is not true scratch")
    return {
        "path": str(path.resolve()),
        "sha256": protocol.file_sha256(path),
        "verified_arguments": observed,
        "fresh_scratch": True,
    }


def audit_and_freeze_run(
    dataset: str,
    method: str,
    *,
    runs_root: Path,
    selected_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = protocol.run_directory(dataset, method, runs_root=runs_root)
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise NotADirectoryError(run_dir)
    protocol_audit = _protocol_audit(run_dir, dataset, method)
    metrics_path = run_dir / "metrics.jsonl"
    candidates = protocol.read_candidate_metrics(metrics_path)
    candidates_with_threshold = []
    for event in candidates:
        ready = dict(event)
        container = protocol._metric_container(ready)
        if "threshold" not in container and "threshold" not in ready:
            ready["threshold"] = protocol.FIXED_THRESHOLD
        candidates_with_threshold.append(ready)
    normalized_candidates = [
        protocol.normalize_metric_event(event)
        for event in candidates_with_threshold
    ]
    selected_metrics = {
        role: dict(
            max(
                normalized_candidates,
                key=lambda event: protocol.checkpoint_selection_key(role, event),
            )
        )
        for role in protocol.SELECTED_ROLES
    }
    fixed_endpoint_epoch1000 = dict(normalized_candidates[-1])

    records: dict[str, Any] = {}
    for role in protocol.CHECKPOINT_ROLES:
        source = (
            run_dir
            / "checkpoints"
            / protocol.CHECKPOINT_FILENAMES[role]
        )
        checkpoint = _load_checkpoint(source)
        _require_checkpoint_matches(
            checkpoint,
            path=source,
            dataset=dataset,
            method=method,
            role=role,
            expected_metrics=selected_metrics[role],
        )
        destination = protocol.selected_checkpoint_path(
            dataset,
            method,
            role,
            selected_root=selected_root,
        )
        digest = protocol.atomic_copy_file(
            source,
            destination,
            overwrite=overwrite,
        )
        records[role] = {
            "checkpoint_role": role,
            "source_path": str(source.resolve()),
            "frozen_path": str(destination.resolve()),
            "sha256": digest,
            "epoch": selected_metrics[role]["epoch"],
            "fixed_threshold_0_5_metrics": selected_metrics[role],
            "test_selected": True,
            "selection_is_optimistic": True,
        }

    return {
        "dataset": dataset,
        "method": method,
        "method_label": protocol.METHOD_LABELS[method],
        "seed": protocol.TRAINING_SEED,
        "run_directory": str(run_dir.resolve()),
        "protocol_audit": protocol_audit,
        "metrics_jsonl": {
            "path": str(metrics_path.resolve()),
            "sha256": protocol.file_sha256(metrics_path),
            "candidate_epochs": list(protocol.CANDIDATE_EPOCHS),
            "candidate_epoch_rule": "10,20,...,1000",
            "candidate_epoch_count": len(normalized_candidates),
        },
        "checkpoints": records,
        "fixed_endpoint_epoch1000": {
            "epoch": protocol.EXPECTED_EPOCHS,
            "checkpoint_saved": False,
            "checkpoint_role": None,
            "fixed_threshold_0_5_metrics": fixed_endpoint_epoch1000,
            "source": "metrics.jsonl",
            "test_selected": False,
            "selection_is_optimistic": False,
        },
        "selection_disclosure": protocol.expected_selection_disclosure(),
        "audit_passed": True,
    }


def build_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sorted_records = sorted(
        (copy.deepcopy(dict(record)) for record in records),
        key=lambda item: (
            protocol.DATASETS.index(str(item["dataset"])),
            protocol.METHODS.index(str(item["method"])),
        ),
    )
    return {
        "schema": f"{protocol.SCHEMA_PREFIX}_checkpoint_manifest_v1",
        "status": "complete",
        "experiment": "four_training_regimes_original_vs_final",
        "seed": protocol.TRAINING_SEED,
        "record_count": len(sorted_records),
        "records": sorted_records,
        "selection_disclosure": protocol.expected_selection_disclosure(),
        "source_sha256": protocol.source_hashes(),
        "no_fabricated_results": True,
        "stability_claim_supported": False,
        "multiseed_replication_supported": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pairs = (
        [
            (dataset, method)
            for dataset in protocol.DATASETS
            for method in protocol.METHODS
        ]
        if args.all_runs
        else [(args.dataset, args.method)]
    )
    records = [
        audit_and_freeze_run(
            dataset,
            method,
            runs_root=args.runs_root,
            selected_root=args.selected_root,
            overwrite=args.overwrite,
        )
        for dataset, method in pairs
    ]
    manifest = build_manifest(records)
    protocol.atomic_write_json(
        args.manifest,
        manifest,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "run_count": len(records),
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": protocol.file_sha256(args.manifest),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
