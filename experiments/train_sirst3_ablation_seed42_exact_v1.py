#!/usr/bin/env python3
"""Train one frozen A0--A4 SIRST3 ablation run from scratch.

Formal mode is fixed to seed 42, 1,000 epochs, SIRST3, FP32, and candidate
epochs 10, 20, ..., 1,000.  The only persistent model weights are ``best_miou`` and
``best_pd``.  A rolling optimizer/RNG state is overwritten for recovery and
removed after successful completion.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_four_dataset_original_final_seed42_exact_v1 as common
from experiments.four_dataset_ablation_models_seed42_v1 import (
    ABLATION_IDS,
    DATASET,
    GRAPH_BY_ABLATION,
    SCHEMA as MODEL_SCHEMA,
    TRAINING_SEED,
    TSS_WEIGHT_BY_ABLATION,
    build_ablation_model,
)
from experiments.paper_four_dataset_v1 import (
    FourDatasetTestDataset,
    FourDatasetTrainDataset,
)
from experiments.tpd_training_loss import compute_tpd_training_loss


SCHEMA = "sctransnet_sirst3_ablation_seed42_exact_train_v1"
DEFAULT_RESULTS_ROOT = (
    common.REPO_ROOT / "results" / "four_dataset_seed42_v1"
)
DEFAULT_MANIFEST_ROOT = DEFAULT_RESULTS_ROOT / "manifests"
DEFAULT_TSS_STATISTICS = (
    DEFAULT_MANIFEST_ROOT / "four_dataset_tss_seed42_v1.json"
)
GPU_UUIDS = dict(common.GPU_UUIDS)
FORMAL_BEGIN_TEST = 10
FORMAL_EVAL_EVERY = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=ABLATION_IDS, required=True)
    parser.add_argument("--data-root", type=Path, default=common.DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument(
        "--tss-statistics", type=Path, default=DEFAULT_TSS_STATISTICS
    )
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=common.FORMAL_EPOCHS)
    parser.add_argument(
        "--begin-test", type=int, default=FORMAL_BEGIN_TEST
    )
    parser.add_argument(
        "--eval-every", type=int, default=FORMAL_EVAL_EVERY
    )
    parser.add_argument(
        "--batch-size", type=int, default=common.FORMAL_BATCH_SIZE
    )
    parser.add_argument(
        "--patch-size", type=int, default=common.FORMAL_PATCH_SIZE
    )
    parser.add_argument("--workers", type=int, default=common.FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=common.FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=common.FORMAL_MIN_LR)
    parser.add_argument(
        "--warmup-epochs", type=int, default=common.FORMAL_WARMUP_EPOCHS
    )
    parser.add_argument(
        "--threshold", type=float, default=common.FORMAL_THRESHOLD
    )
    parser.add_argument(
        "--match-radius", type=float, default=common.FORMAL_MATCH_RADIUS
    )
    parser.add_argument(
        "--tiny-area", type=int, default=common.FORMAL_TINY_AREA
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--physical-gpu-index", choices=("2", "3"))
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.seed != TRAINING_SEED:
        raise ValueError("the ablation protocol has the sole seed 42")
    if args.epochs < 1 or args.begin_test < 1 or args.eval_every < 1:
        raise ValueError("epoch controls must be positive")
    if args.batch_size < 1 or args.patch_size < 32 or args.patch_size % 32:
        raise ValueError("invalid batch/patch configuration")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.physical_gpu_index is not None:
        expected = GPU_UUIDS[args.physical_gpu_index]
        if args.expected_gpu_uuid != expected:
            raise ValueError(
                f"GPU UUID differs for physical GPU "
                f"{args.physical_gpu_index}: {args.expected_gpu_uuid!r}"
            )
    if args.smoke:
        if args.epochs > 2:
            raise ValueError("smoke is limited to two epochs")
        if args.max_train_images is None or args.max_test_images is None:
            raise ValueError("smoke requires both image limits")
        return
    expected = {
        "epochs": 1000,
        "begin_test": FORMAL_BEGIN_TEST,
        "eval_every": FORMAL_EVAL_EVERY,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "base_lr": 1e-3,
        "min_lr": 1e-5,
        "warmup_epochs": 10,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "device": "cuda:0",
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(
                f"formal {name} differs: {getattr(args, name)!r} != {value!r}"
            )
    if args.physical_gpu_index not in GPU_UUIDS:
        raise ValueError("formal run must bind physical GPU 2 or 3")
    if args.ablation == "A4" and not args.tss_statistics.is_file():
        raise FileNotFoundError(args.tss_statistics)


def _run_directory(args: argparse.Namespace) -> Path:
    root = args.results_root.resolve() / "ablations"
    if args.smoke:
        root = root / "smoke"
    return root / "runs" / DATASET / args.ablation / "seed_42"


def _load_tss(args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if args.ablation != "A4":
        return 1.0, {
            "enabled": False,
            "weight": 0.0,
            "reason": "TSS is present only in A4",
        }
    path = args.tss_statistics.resolve()
    if args.smoke and not path.is_file():
        return 1.0, {
            "enabled": True,
            "weight": 0.005,
            "source": "smoke_default",
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    record = payload.get("datasets", {}).get(DATASET)
    if not isinstance(record, Mapping):
        raise ValueError("TSS statistics do not contain SIRST3")
    value = float(record["survival_pos_weight"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("SIRST3 survival_pos_weight is invalid")
    return value, {
        "enabled": True,
        "weight": 0.005,
        "source": str(path),
        "sha256": common.file_sha256(path),
        "dataset_record": copy.deepcopy(dict(record)),
    }


def _protocol(
    args: argparse.Namespace,
    *,
    model_metadata: Mapping[str, Any],
    data_manifests: Mapping[str, Any],
    tss_metadata: Mapping[str, Any],
    train_count: int,
    test_count: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model_schema": MODEL_SCHEMA,
        "dataset": DATASET,
        "ablation_id": args.ablation,
        "graph": GRAPH_BY_ABLATION[args.ablation],
        "training_seed": TRAINING_SEED,
        "scratch": True,
        "warm_start_used": False,
        "parent_checkpoint": None,
        "epochs": args.epochs,
        "candidate_epochs": list(
            range(args.begin_test, args.epochs + 1, args.eval_every)
        ),
        "candidate_epoch_count": len(
            range(args.begin_test, args.epochs + 1, args.eval_every)
        ),
        "eval_every": args.eval_every,
        "test_selected": True,
        "selection_is_optimistic": True,
        "selected_checkpoint_roles": ["best_miou", "best_pd"],
        "unselected_candidate_weights_saved": False,
        "epoch1000_checkpoint_saved": False,
        "rolling_resume_state": {
            "temporary": True,
            "overwritten_each_epoch": True,
            "removed_after_success": True,
        },
        "optimizer": "Adam",
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "workers": args.workers,
        "precision": "FP32",
        "amp": False,
        "segmentation_loss": "ordered sum BCE over six outputs",
        "tss": copy.deepcopy(dict(tss_metadata)),
        "tss_weight": TSS_WEIGHT_BY_ABLATION[args.ablation],
        "selection_threshold": args.threshold,
        "match_radius": args.match_radius,
        "tiny_area": args.tiny_area,
        "normalization": copy.deepcopy(common.LEGACY_NORMALIZATION[DATASET]),
        "dataset_counts": {"train": train_count, "test": test_count},
        "data_manifests": copy.deepcopy(dict(data_manifests)),
        "model": copy.deepcopy(dict(model_metadata)),
        "device": {
            "logical": str(device),
            "physical_index": args.physical_gpu_index,
            "uuid": args.expected_gpu_uuid,
            "name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else "cpu"
            ),
        },
        "protocol_document": {
            "path": str(common.PROTOCOL_DOCUMENT),
            "sha256": common.file_sha256(common.PROTOCOL_DOCUMENT),
        },
        "runtime_sources": {
            "runner": common.file_sha256(Path(__file__).resolve()),
            "builder": common.file_sha256(
                Path(
                    "/home/ly/SCTransNet_main/experiments/"
                    "four_dataset_ablation_models_seed42_v1.py"
                )
            ),
        },
        "attribution_lock": {
            "cumulative_order": list(ABLATION_IDS),
            "one_added_factor": True,
            "capacity_matching_claimed": False,
            "A0_to_A1_includes_frozen_tpd_parameterization": True,
        },
        "smoke": bool(args.smoke),
    }


def _selected_payload(
    args: argparse.Namespace,
    model: nn.Module,
    model_metadata: Mapping[str, Any],
    protocol_sha256: str,
    epoch: int,
    role: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dataset": DATASET,
        "ablation_id": args.ablation,
        "graph": GRAPH_BY_ABLATION[args.ablation],
        "seed": TRAINING_SEED,
        "epoch": epoch,
        "checkpoint_role": role,
        "selection_source": "test_SIRST3",
        "test_selected": True,
        "selection_is_optimistic": True,
        "test_metrics": copy.deepcopy(dict(metrics)),
        "state_dict": common.cpu_state_dict(model),
        "model_metadata": copy.deepcopy(dict(model_metadata)),
        "protocol_sha256": protocol_sha256,
    }


def _resume_payload(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_metadata: Mapping[str, Any],
    protocol_sha256: str,
    epoch: int,
    event: Mapping[str, Any],
    best_miou: Mapping[str, Any],
    best_pd: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dataset": DATASET,
        "ablation_id": args.ablation,
        "seed": TRAINING_SEED,
        "epoch": epoch,
        "checkpoint_role": "temporary_rolling_resume",
        "state_dict": common.cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "rng_state": common.rng_state(),
        "event": copy.deepcopy(dict(event)),
        "best_miou": copy.deepcopy(dict(best_miou)),
        "best_pd": copy.deepcopy(dict(best_pd)),
        "model_metadata": copy.deepcopy(dict(model_metadata)),
        "protocol_sha256": protocol_sha256,
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if value.get("epoch") != line_number:
                raise ValueError("metrics JSONL epoch sequence differs")
            values.append(value)
    return values


def _load_resume(
    args: argparse.Namespace,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    protocol_sha256: str,
    metrics_path: Path,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    events = _read_events(metrics_path)
    if not path.exists():
        if args.resume == "required":
            raise FileNotFoundError(path)
        if events:
            raise RuntimeError("metrics exist without exact resume state")
        return 1, {}, {}
    if args.resume == "never":
        raise FileExistsError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "ablation_id": args.ablation,
        "seed": TRAINING_SEED,
        "protocol_sha256": protocol_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"resume {key} differs")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    common.restore_rng_state(payload["rng_state"])
    epoch = int(payload["epoch"])
    event = dict(payload["event"])
    if len(events) == epoch - 1:
        common.append_jsonl(metrics_path, event)
    elif len(events) != epoch:
        raise RuntimeError("metrics/resume epoch mismatch")
    elif common.canonical_sha256(events[-1]) != common.canonical_sha256(event):
        raise RuntimeError("metrics tail differs from resume event")
    return (
        epoch + 1,
        dict(payload.get("best_miou", {})),
        dict(payload.get("best_pd", {})),
    )


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    common.configure_determinism()
    device = common.resolve_device(args)
    run_dir = _run_directory(args)
    checkpoint_dir = run_dir / "checkpoints"
    resume_path = run_dir / "resume" / "latest_training_state.pth.tar"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / "run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"ablation run is already active: {run_dir}") from error

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            if json.load(handle).get("status") == "complete":
                if resume_path.exists():
                    resume_path.unlink()
                return summary_path

    data_manifests = common._load_data_manifest_lock(args)
    manifest_files = data_manifests["files"]
    common.seed_everything(TRAINING_SEED)
    model, model_metadata = build_ablation_model(args.ablation, TRAINING_SEED)
    model.to(device)

    train_dataset = FourDatasetTrainDataset(
        DATASET,
        patch_size=args.patch_size,
        seed=TRAINING_SEED,
        dataset_root=args.data_root.resolve(),
        imgidx_manifest=manifest_files["imgidx"]["path"],
        normalization_manifest=manifest_files["normalization"]["path"],
        correction_manifest=manifest_files["correction"]["path"],
        return_metadata=False,
    )
    test_dataset = FourDatasetTestDataset(
        DATASET,
        DATASET,
        dataset_root=args.data_root.resolve(),
        imgidx_manifest=manifest_files["imgidx"]["path"],
        normalization_manifest=manifest_files["normalization"]["path"],
        correction_manifest=manifest_files["correction"]["path"],
        return_metadata=False,
    )
    expected_normalization = common.LEGACY_NORMALIZATION[DATASET]
    if train_dataset.normalization != expected_normalization:
        raise RuntimeError("SIRST3 ablation train normalization differs")
    if test_dataset.normalization != expected_normalization:
        raise RuntimeError("SIRST3 ablation test normalization differs")
    train_view = common._dataset_view(train_dataset, args.max_train_images)
    test_view = common._dataset_view(test_dataset, args.max_test_images)
    test_loader = DataLoader(
        test_view,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    pos_weight, tss_metadata = _load_tss(args)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    protocol = _protocol(
        args,
        model_metadata=model_metadata,
        data_manifests=data_manifests,
        tss_metadata=tss_metadata,
        train_count=len(train_view),
        test_count=len(test_view),
        device=device,
    )
    protocol_sha256 = common.canonical_sha256(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        with protocol_path.open("r", encoding="utf-8") as handle:
            if json.load(handle).get("protocol_sha256") != protocol_sha256:
                raise ValueError("existing ablation protocol differs")
    else:
        common.write_json_atomic(protocol_path, protocol)

    start_epoch, best_miou, best_pd = _load_resume(
        args,
        resume_path,
        model,
        optimizer,
        protocol_sha256,
        metrics_path,
    )
    if start_epoch == 1:
        common.seed_everything(TRAINING_SEED)
    started = time.time()
    print(
        f"START dataset={DATASET} ablation={args.ablation} seed=42 "
        f"epoch={start_epoch}/{args.epochs} device={device}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        lr = common.metric_base.learning_rate_for_epoch(
            epoch,
            args.epochs,
            args.base_lr,
            args.min_lr,
            args.warmup_epochs,
        )
        common.metric_base.set_learning_rate(optimizer, lr)
        common._set_dataset_epoch(train_view, epoch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            common.stable_uint63(TRAINING_SEED, DATASET, "shuffle", epoch)
        )
        train_loader = DataLoader(
            train_view,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=generator,
            drop_last=False,
        )
        model.train()
        totals = {"total": 0.0, "segmentation": 0.0, "survival": 0.0}
        sample_count = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            losses = compute_tpd_training_loss(
                output,
                masks,
                criterion,
                survival_weight=TSS_WEIGHT_BY_ABLATION[args.ablation],
                survival_pos_weight=pos_weight,
            )
            losses.total.backward()
            optimizer.step()
            count = int(images.shape[0])
            totals["total"] += float(losses.total.detach().item()) * count
            totals["segmentation"] += (
                float(losses.segmentation.detach().item()) * count
            )
            totals["survival"] += (
                float(losses.survival.detach().item()) * count
            )
            sample_count += count
        if sample_count != len(train_view):
            raise RuntimeError("processed training sample count differs")

        event: dict[str, Any] = {
            "schema": SCHEMA,
            "dataset": DATASET,
            "ablation_id": args.ablation,
            "seed": TRAINING_SEED,
            "epoch": epoch,
            "learning_rate": lr,
            "train_total_loss": totals["total"] / sample_count,
            "train_segmentation_loss": totals["segmentation"] / sample_count,
            "train_survival_loss": totals["survival"] / sample_count,
            "processed_train_samples": sample_count,
            "evaluated": False,
        }
        should_evaluate = (
            epoch >= args.begin_test
            and (
                (epoch - args.begin_test) % args.eval_every == 0
                or epoch == args.epochs
            )
        )
        if should_evaluate:
            metrics = common.evaluate(
                model,
                test_loader,
                device,
                criterion,
                threshold=args.threshold,
                match_radius=args.match_radius,
                tiny_area=args.tiny_area,
            )
            event.update(metrics)
            event["evaluated"] = True
            miou_key = common.best_miou_key(metrics, epoch)
            pd_key = common.best_pd_key(metrics, epoch)
            new_miou = not best_miou or miou_key > tuple(best_miou["key"])
            new_pd = not best_pd or pd_key > tuple(best_pd["key"])
            if new_miou:
                path = checkpoint_dir / "best_miou.pth.tar"
                common.torch_save_atomic(
                    path,
                    _selected_payload(
                        args,
                        model,
                        model_metadata,
                        protocol_sha256,
                        epoch,
                        "best_miou",
                        metrics,
                    ),
                )
                best_miou = {
                    "epoch": epoch,
                    "key": list(miou_key),
                    "metrics": copy.deepcopy(metrics),
                    "path": str(path),
                }
            if new_pd:
                path = checkpoint_dir / "best_pd.pth.tar"
                common.torch_save_atomic(
                    path,
                    _selected_payload(
                        args,
                        model,
                        model_metadata,
                        protocol_sha256,
                        epoch,
                        "best_pd",
                        metrics,
                    ),
                )
                best_pd = {
                    "epoch": epoch,
                    "key": list(pd_key),
                    "metrics": copy.deepcopy(metrics),
                    "path": str(path),
                }
            event.update(
                {
                    "new_best_miou": new_miou,
                    "new_best_pd": new_pd,
                    "best_miou_epoch": best_miou["epoch"],
                    "best_pd_epoch": best_pd["epoch"],
                }
            )
        event["epoch_seconds"] = time.time() - epoch_started
        common.torch_save_atomic(
            resume_path,
            _resume_payload(
                args,
                model,
                optimizer,
                model_metadata,
                protocol_sha256,
                epoch,
                event,
                best_miou,
                best_pd,
            ),
        )
        common.append_jsonl(metrics_path, event)
        common.write_json_atomic(
            run_dir / "progress.json",
            {
                "schema": SCHEMA,
                "status": "running",
                "dataset": DATASET,
                "ablation_id": args.ablation,
                "seed": TRAINING_SEED,
                "completed_epoch": epoch,
                "total_epochs": args.epochs,
                "best_miou": best_miou,
                "best_pd": best_pd,
                "updated_at_unix": time.time(),
            },
        )
        if should_evaluate:
            print(
                f"EPOCH ablation={args.ablation} epoch={epoch}/{args.epochs} "
                f"loss={event['train_total_loss']:.6f} "
                f"mIoU={event['miou']:.6f} Pd={event['pd']:.6f} "
                f"Fa={event['fa']:.8e}",
                flush=True,
            )
        elif epoch == 1 or epoch % 25 == 0:
            print(
                f"EPOCH ablation={args.ablation} epoch={epoch}/{args.epochs} "
                f"loss={event['train_total_loss']:.6f}",
                flush=True,
            )

    selected = {
        role: checkpoint_dir / f"{role}.pth.tar"
        for role in ("best_miou", "best_pd")
    }
    if any(not path.is_file() for path in selected.values()):
        raise RuntimeError("one or both selected checkpoints are missing")
    persistent_weights = sorted(checkpoint_dir.glob("*.pth*"))
    if set(persistent_weights) != set(selected.values()):
        raise RuntimeError("unexpected persistent ablation checkpoint")
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "ablation_id": args.ablation,
        "seed": TRAINING_SEED,
        "epochs": args.epochs,
        "best_miou": best_miou,
        "best_pd": best_pd,
        "test_selected": True,
        "selection_is_optimistic": True,
        "checkpoints": {
            role: {
                "path": str(path),
                "sha256": common.file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for role, path in selected.items()
        },
        "metrics": str(metrics_path),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "elapsed_seconds_this_invocation": time.time() - started,
        "completed_at_unix": time.time(),
    }
    common.write_json_atomic(summary_path, summary)
    if resume_path.exists():
        resume_path.unlink()
    common.write_json_atomic(
        run_dir / "progress.json",
        {
            "schema": SCHEMA,
            "status": "complete",
            "dataset": DATASET,
            "ablation_id": args.ablation,
            "seed": TRAINING_SEED,
            "completed_epoch": args.epochs,
            "total_epochs": args.epochs,
            "summary": str(summary_path),
            "updated_at_unix": time.time(),
        },
    )
    print(f"COMPLETE ablation={args.ablation} summary={summary_path}", flush=True)
    return summary_path


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
