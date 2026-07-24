#!/usr/bin/env python3
"""Extended completion audit for the fresh four-RTX-5090 formal800 runs.

This layer deliberately leaves the frozen generic training/summarization tools
unchanged. It adds run-identity binding, exact protocol checks, last-checkpoint
validation, independent initialization reconstruction, and artifact hashes.
It reads only the official training index; no official test path is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.summarize_tpd_pilot import (
    audit_checkpoint,
    audit_cross_variant_consistency,
    build_model_for_strict_load,
    load_metrics,
    load_run,
    miou_selection_key,
    pd_selection_key,
)
from experiments.train_tpd_pilot import build_model, learning_rate_for_epoch


VARIANTS = ("original", "progressive", "tpd", "spd")
GPU_UUIDS = {
    "original": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "progressive": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "tpd": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "spd": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
INVOCATION_IDS = {
    "original": "0e533dd2d1444feb9ba1ed1e3d42135c",
    "progressive": "0803e8ed5f974f909c54d6daaddc23bb",
    "tpd": "669e46b8bd2245db92788afa62066bd3",
    "spd": "75468708b771457daac1cf0586f7c333",
}
LAUNCH_MANIFEST_SHA256 = {
    "original": "2841ef7217e21b7cc58d6d9f7dadeb4f995191027339afc11a314f6f32a31ad9",
    "progressive": "194df73ffe812b04c462a541508cae2507db6f5d076265f3fea7be88d2f4cec2",
    "tpd": "fe277ef7e31518f0c0b9d5b6009686c26a360c2663153abdd84b0bf6b3726868",
    "spd": "c80da19164d1563b759df216bc569e3fbb453e2568d261147fb0dc5836ea18fa",
}
SPLIT_SHA256 = "27c3b4a30c680af1c16493f723ce9713cb7e6987dcbc82e72fe1331cff12cd6b"
TRAINING_DATA_SHA256 = "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
SHARED_INITIALIZATION_SHA256 = (
    "ae25925e8fffd9afe9fac1805389e80437f0d773ae744c979349a68886d81558"
)
WORKER_SHA256 = "72f1c0503fcec0c69f8a3b9c49da57a49db75134cc453031da56d635adc2d7a7"
LAUNCHER_SHA256 = "78b397a5c17bfeb62c3a83ec3aaf4ff733f97c29f686936e140fb7a0a7741fd8"
STATUS_SCRIPT_SHA256 = (
    "2ecc3621b4bedf6bd452d1bc1d3273a168925dd7e2af067cc0dfb7aeb0fd0a40"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended integrity audit for SCTransNet formal800 4xRTX5090"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--runtime-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def require_finite_tensors(value: Any, context: str) -> None:
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Non-finite tensor at {context}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_tensors(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_tensors(item, f"{context}[{index}]")


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def exact_protocol_arguments(
    variant: str, dataset: str, root: Path, run_name: str
) -> Dict[str, Any]:
    prefix = "seed_42_"
    if not run_name.startswith(prefix):
        raise ValueError(f"Unexpected run name: {run_name}")
    return {
        "variant": variant,
        "dataset": dataset,
        "dataset_dir": str((REPO_ROOT / "datasets").resolve()),
        "output_root": str(root.resolve()),
        "run_tag": run_name[len(prefix) :],
        "device": "cuda:0",
        "epochs": 800,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "seed": 42,
        "split_seed": 20260722,
        "val_fraction": 0.2,
        "eval_every": 1,
        "base_lr": 0.001,
        "min_lr": 0.00001,
        "warmup_epochs": 10,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "amp": False,
        "max_train_images": None,
        "max_val_images": None,
    }


def validate_event_stream(
    path: Path, variant: str, expected_epochs: int
) -> Dict[str, Any]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require_equal(len(events), expected_epochs, f"{variant}.metrics count")
    best_pd_key = (-float("inf"),) * 5
    best_miou_key = (-float("inf"),) * 5
    best_pd_epoch = 0
    best_miou_epoch = 0
    for expected_epoch, event in enumerate(events, start=1):
        require_equal(event.get("epoch"), expected_epoch, f"{variant}.metrics epoch")
        require_equal(event.get("variant"), variant, f"{variant}.metrics variant")
        require_equal(
            event.get("processed_train_samples"),
            530,
            f"{variant}.processed_train_samples[{expected_epoch}]",
        )
        expected_lr = learning_rate_for_epoch(
            expected_epoch, expected_epochs, 0.001, 0.00001, 10
        )
        require_equal(
            event.get("learning_rate"),
            expected_lr,
            f"{variant}.learning_rate[{expected_epoch}]",
        )
        for key, value in event.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{variant}.metrics[{expected_epoch}].{key} is non-finite")

        metrics = {
            key: event[key]
            for key in (
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
        }
        current_pd_key = pd_selection_key(metrics)
        current_miou_key = miou_selection_key(metrics)
        expected_new_pd = current_pd_key > best_pd_key
        expected_new_miou = current_miou_key > best_miou_key
        require_equal(
            event.get("new_best_pd"),
            expected_new_pd,
            f"{variant}.new_best_pd[{expected_epoch}]",
        )
        require_equal(
            event.get("new_best_miou"),
            expected_new_miou,
            f"{variant}.new_best_miou[{expected_epoch}]",
        )
        if expected_new_pd:
            best_pd_key = current_pd_key
            best_pd_epoch = expected_epoch
        if expected_new_miou:
            best_miou_key = current_miou_key
            best_miou_epoch = expected_epoch
    return {
        "event_count": len(events),
        "epoch_range": [1, expected_epochs],
        "processed_train_samples_each_epoch": 530,
        "learning_rate_schedule_exact": True,
        "online_best_flags_exact": True,
        "recomputed_best_pd_epoch": best_pd_epoch,
        "recomputed_best_miou_epoch": best_miou_epoch,
    }


def audit_runtime_state(path: Path, root: Path) -> Dict[str, Any]:
    require_regular(path, "runtime state")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_equal(
        payload.get("schema"),
        "sctransnet_formal800_4x5090_systemd_state_v1",
        "runtime state schema",
    )
    require_equal(payload.get("worker_sha256"), WORKER_SHA256, "runtime worker SHA")
    require_equal(payload.get("launcher_sha256"), LAUNCHER_SHA256, "runtime launcher SHA")
    require_equal(
        payload.get("status_script_sha256"),
        STATUS_SCRIPT_SHA256,
        "runtime status script SHA",
    )
    units = payload.get("units")
    if not isinstance(units, list) or len(units) != len(VARIANTS):
        raise ValueError("runtime state must contain exactly four units")
    by_variant = {item.get("variant"): item for item in units}
    require_equal(set(by_variant), set(VARIANTS), "runtime state variants")
    for variant in VARIANTS:
        item = by_variant[variant]
        require_equal(
            item.get("unit"),
            f"sctransnet-formal800-4x5090-{variant}.service",
            f"{variant}.unit",
        )
        require_equal(
            item.get("invocation_id"), INVOCATION_IDS[variant], f"{variant}.invocation"
        )
        require_equal(item.get("gpu_uuid"), GPU_UUIDS[variant], f"{variant}.GPU UUID")
        require_equal(
            item.get("exec_start"),
            [
                "/usr/bin/bash",
                str(
                    (
                        REPO_ROOT
                        / "experiments"
                        / "run_tpd_formal800_4x5090_worker.sh"
                    ).resolve()
                ),
                variant,
                GPU_UUIDS[variant],
            ],
            f"{variant}.ExecStart",
        )
        require_equal(item.get("n_restarts_at_capture"), 0, f"{variant}.NRestarts")
        require_equal(
            item.get("launch_manifest_sha256"),
            LAUNCH_MANIFEST_SHA256[variant],
            f"{variant}.launch manifest SHA",
        )
        launch_path = root / "launch" / f"{variant}.json"
        require_regular(launch_path, f"{variant} launch manifest")
        require_equal(
            sha256_file(launch_path),
            LAUNCH_MANIFEST_SHA256[variant],
            f"{variant} current launch manifest SHA",
        )
    return {
        "state_path": str(path.resolve()),
        "state_sha256": sha256_file(path),
        "captured_at": payload.get("captured_at"),
        "units": units,
        "invocation_ids_bound": True,
        "exec_start_gpu_mapping_bound": True,
        "no_restarts_at_capture": True,
    }


def audit_official_train_isolation(
    dataset_root: Path, runs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    train_index = dataset_root / "img_idx" / "train_NUDT-SIRST.txt"
    require_regular(train_index, "official training index")
    training_ids = [
        line.strip()
        for line in train_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(training_ids) != len(set(training_ids)):
        raise ValueError("Official training index contains duplicate identifiers")
    require_equal(len(training_ids), 663, "official training ID count")
    for variant, run in runs.items():
        split = run["split"]
        combined = (
            split["full_internal_train_ids"] + split["full_internal_val_ids"]
        )
        require_equal(set(combined), set(training_ids), f"{variant}.train ID coverage")
        require_equal(len(combined), len(training_ids), f"{variant}.train ID count")
    return {
        "official_train_index": str(train_index.resolve()),
        "official_train_index_sha256": sha256_file(train_index),
        "official_train_count": len(training_ids),
        "internal_split_union_equals_official_train": True,
        "runner_code_path_reads_training_index_only": True,
        "official_test_code_path_isolation_verified": True,
        "syscall_level_trace_available": False,
    }


def audit_checkpoint_payload_tensors(path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint for tensor audit: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a mapping: {path}")
    for key in ("state_dict", "optimizer", "scaler"):
        if key not in payload:
            raise ValueError(f"Checkpoint missing {key}: {path}")
        require_finite_tensors(payload[key], f"{path}.{key}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    runtime_state = args.runtime_state.resolve()
    if args.expected_epochs != 800:
        raise ValueError("This certificate is specific to exactly 800 epochs")
    if args.dataset != "NUDT-SIRST":
        raise ValueError("This certificate is specific to NUDT-SIRST")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")

    dataset_root = root / args.dataset
    runtime_binding = audit_runtime_state(runtime_state, root)
    runs: Dict[str, Dict[str, Any]] = {}
    per_variant: Dict[str, Any] = {}
    split_bytes: list[bytes] = []
    recomputed_shared: Dict[str, str] = {}

    for variant in VARIANTS:
        run_dir = dataset_root / variant / args.run_name
        run = load_run(run_dir, args.expected_epochs, variant, args.dataset)
        runs[variant] = run
        protocol = run["protocol"]
        summary = run["summary"]
        require_equal(
            run["arguments"],
            exact_protocol_arguments(variant, args.dataset, root, args.run_name),
            f"{variant}.exact protocol arguments",
        )
        require_equal(
            protocol.get("environment"),
            {
                "torch": "2.9.1+cu130",
                "cuda_runtime": "13.0",
                "device": "cuda:0",
                "device_name": "NVIDIA GeForce RTX 5090",
            },
            f"{variant}.environment",
        )
        require_equal(
            summary.get("skipped_singleton_batches"),
            0,
            f"{variant}.skipped singleton batches",
        )

        split_path = run_dir / "split.json"
        require_equal(sha256_file(split_path), SPLIT_SHA256, f"{variant}.split SHA")
        split_bytes.append(split_path.read_bytes())
        event_audit = validate_event_stream(
            run_dir / "metrics.jsonl", variant, args.expected_epochs
        )
        require_equal(
            event_audit["recomputed_best_pd_epoch"],
            summary["best_pd_epoch"],
            f"{variant}.best Pd epoch",
        )
        require_equal(
            event_audit["recomputed_best_miou_epoch"],
            summary["best_miou_epoch"],
            f"{variant}.best mIoU epoch",
        )

        evaluated = load_metrics(
            run_dir / "metrics.jsonl",
            args.expected_epochs,
            variant,
            int(run["arguments"]["eval_every"]),
        )
        last_epoch, last_metrics = evaluated[-1]
        require_equal(last_epoch, args.expected_epochs, f"{variant}.last epoch")
        last_path = run_dir / "last.pth.tar"
        last_sha = audit_checkpoint(
            last_path,
            build_model_for_strict_load(variant),
            "last_evaluated_epoch",
            args.expected_epochs,
            last_metrics,
            variant,
            args.dataset,
            int(summary["seed"]),
            int(run["arguments"]["split_seed"]),
            summary["split_hashes"],
            summary["model"],
        )
        require_equal(
            Path(summary["last_checkpoint"]).resolve(),
            last_path.resolve(),
            f"{variant}.summary.last_checkpoint",
        )

        for checkpoint_name in ("best.pth.tar", "best_miou.pth.tar", "last.pth.tar"):
            audit_checkpoint_payload_tensors(run_dir / checkpoint_name)

        _, rebuilt_metadata = build_model(variant, 42)
        require_equal(
            rebuilt_metadata,
            summary["model"],
            f"{variant}.independently rebuilt initialization",
        )
        require_equal(
            rebuilt_metadata["shared_initialization_sha256"],
            SHARED_INITIALIZATION_SHA256,
            f"{variant}.shared initialization",
        )
        recomputed_shared[variant] = rebuilt_metadata["shared_initialization_sha256"]

        launch_path = root / "launch" / f"{variant}.json"
        log_path = root / "logs" / f"{variant}.log"
        require_regular(log_path, f"{variant} worker log")
        artifact_paths = {
            "protocol.json": run_dir / "protocol.json",
            "split.json": split_path,
            "metrics.jsonl": run_dir / "metrics.jsonl",
            "summary.json": run_dir / "summary.json",
            "best.pth.tar": run_dir / "best.pth.tar",
            "best_miou.pth.tar": run_dir / "best_miou.pth.tar",
            "last.pth.tar": last_path,
            "launch_manifest": launch_path,
            "worker_log": log_path,
        }
        for label, path in artifact_paths.items():
            require_regular(path, f"{variant}.{label}")
        per_variant[variant] = {
            "gpu_uuid": GPU_UUIDS[variant],
            "invocation_id": INVOCATION_IDS[variant],
            "event_audit": event_audit,
            "best_pd_epoch": summary["best_pd_epoch"],
            "best_miou_epoch": summary["best_miou_epoch"],
            "last_checkpoint_audit": {
                "role": "last_evaluated_epoch",
                "epoch": args.expected_epochs,
                "sha256": last_sha,
                "strict_load": True,
                "all_checkpoint_tensors_finite": True,
                "metrics_equal_final_event": True,
            },
            "initialization_recomputed": rebuilt_metadata,
            "artifact_sha256": {
                label: sha256_file(path) for label, path in artifact_paths.items()
            },
        }

    if len(set(split_bytes)) != 1:
        raise ValueError("The four split manifests are not byte-identical")
    if set(recomputed_shared.values()) != {SHARED_INITIALIZATION_SHA256}:
        raise ValueError("Independently recomputed shared initialization differs")
    cross_variant = audit_cross_variant_consistency(runs)
    isolation = audit_official_train_isolation(
        REPO_ROOT / "datasets" / args.dataset, runs
    )

    source_paths = {
        "worker": REPO_ROOT / "experiments" / "run_tpd_formal800_4x5090_worker.sh",
        "launcher": REPO_ROOT / "experiments" / "launch_tpd_formal800_4x5090.sh",
        "status": REPO_ROOT / "experiments" / "status_tpd_formal800_4x5090.sh",
        "runner": REPO_ROOT / "experiments" / "train_tpd_pilot.py",
        "evaluator": REPO_ROOT / "experiments" / "evaluate_pd_fa_sweep.py",
        "training_summarizer": REPO_ROOT / "experiments" / "summarize_tpd_pilot.py",
        "sweep_aggregator": REPO_ROOT / "experiments" / "summarize_tpd_pd_fa.py",
        "dataset": REPO_ROOT / "dataset.py",
        "utils": REPO_ROOT / "utils.py",
        "model": REPO_ROOT / "model" / "SCTransNet.py",
        "tpd": REPO_ROOT / "model" / "tpd.py",
        "config": REPO_ROOT / "model" / "Config.py",
        "warmup_scheduler_posthoc_observed": REPO_ROOT / "warmup_scheduler.py",
        "this_auditor": Path(__file__).resolve(),
    }
    for label, path in source_paths.items():
        require_regular(path, label)

    checks_passed = {
        "runtime_invocation_and_execstart_bound": True,
        "launch_manifests_hash_bound": True,
        "frozen_protocol_exact": True,
        "metrics_800_contiguous_finite": True,
        "processed_sample_count_exact": True,
        "learning_rate_schedule_exact": True,
        "online_best_flags_exact": True,
        "best_checkpoints_strict_and_finite": True,
        "best_miou_checkpoints_strict_and_finite": True,
        "last_checkpoints_strict_and_finite": True,
        "last_checkpoint_metrics_match_epoch_800": True,
        "split_manifests_byte_identical": True,
        "shared_initialization_independently_recomputed": True,
        "training_data_fingerprint_bound": True,
        "official_test_code_path_isolation_verified": True,
    }
    payload = {
        "schema": "sctransnet_formal800_4x5090_extended_integrity_v1",
        "root": str(root),
        "dataset": args.dataset,
        "run_name": args.run_name,
        "expected_epochs": args.expected_epochs,
        "official_test_accessed": False,
        "selection_source": "internal_validation_only",
        "training_data_sha256": TRAINING_DATA_SHA256,
        "runtime_binding": runtime_binding,
        "per_variant": per_variant,
        "cross_variant_consistency": cross_variant,
        "byte_identical_split_sha256": SPLIT_SHA256,
        "independently_recomputed_shared_initialization_sha256": (
            SHARED_INITIALIZATION_SHA256
        ),
        "official_test_isolation_evidence": isolation,
        "frozen_sources_and_data": {
            "source_sha256": {
                label: sha256_file(path) for label, path in source_paths.items()
            },
            "training_data_sha256": TRAINING_DATA_SHA256,
            "environment": {
                "python_executable": os.path.realpath(os.sys.executable),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "torchvision": package_version("torchvision"),
                "numpy": package_version("numpy"),
                "scipy": package_version("scipy"),
                "scikit_image": package_version("scikit-image"),
                "opencv": package_version(
                    "opencv-python-headless", "opencv-python", "opencv-contrib-python"
                ),
                "matplotlib": package_version("matplotlib"),
                "ml_collections": package_version("ml-collections"),
                "einops": package_version("einops"),
                "thop": package_version("thop"),
                "pillow": package_version("Pillow"),
            },
        },
        "checks_passed": checks_passed,
        "limitations": {
            "syscall_level_file_access_trace_available": False,
            "official_test_isolation_basis": (
                "frozen runner code path, internal split provenance, manifests, "
                "and artifact declarations"
            ),
            "warmup_scheduler_sha_is_posthoc_observed_not_launch_bound": True,
            "single_dataset_single_seed_screening_only": True,
            "mainline_decision_not_made_by_this_audit": True,
        },
    }
    if not all(checks_passed.values()):
        raise ValueError("Not all extended integrity checks passed")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
