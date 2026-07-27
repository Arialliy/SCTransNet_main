#!/usr/bin/env python3
"""Exact epoch-boundary training entry for TPD-Clean V7-DCH.

The exact-resume control plane is reused from the frozen V6 implementation,
but every candidate-owned identity is rebound here: variants, builder, block
type, schemas, paths, run ID, architecture manifest, source-lock key, metrics
storage contract, protocol payload, checkpoint adapter, runner name, and
completion summary.  Importing this module creates no files and starts no
training process.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import train_tpd_clean_v6_exact as shared_exact  # noqa: E402
from experiments import train_tpd_pilot as base  # noqa: E402
from experiments.train_tpd_clean_v7_dch import (  # noqa: E402
    build_clean_v7_dch_model,
)
from model.tpd_clean_v7_dch import (  # noqa: E402
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
    TPDCleanV7DCHBlock,
)


ENTRY_SCHEMA = "sctransnet_tpd_clean_v7_dch_exact_entry_v1"
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_clean_v7_dch_exact_source_lock_v1"
)
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_clean_v7_dch_exact_architecture_manifest_v1"
)
DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT / "experiments/tpd_clean_v7_dch_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v7_dch_exact_v1"
)

FORMAL_EPOCHS = shared_exact.FORMAL_EPOCHS
FORMAL_EVAL_EVERY = shared_exact.FORMAL_EVAL_EVERY
FORMAL_WORKERS = shared_exact.FORMAL_WORKERS
FORMAL_AMP = shared_exact.FORMAL_AMP
FORMAL_EPS = shared_exact.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = (
    shared_exact.FORMAL_CUBLAS_WORKSPACE_CONFIG
)
FORMAL_INITIALIZATION_MODES = shared_exact.FORMAL_INITIALIZATION_MODES

RUNTIME_SOURCE_PATHS = (
    # DCH-owned formal training path and frozen protocol.
    REPO_ROOT / "experiments/train_tpd_clean_v7_dch_exact.py",
    REPO_ROOT / "experiments/train_tpd_clean_v7_dch.py",
    REPO_ROOT / "model/tpd_clean_v7_dch.py",
    REPO_ROOT / "experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md",
    # Formal launch/control path.  The worker imports the persistent smoke
    # verifier during every preflight, so all four files are part of the
    # executed training path rather than post-hoc acceptance code.
    REPO_ROOT
    / "experiments/run_tpd_clean_v7_dch_formal800_2x5090_worker.sh",
    REPO_ROOT
    / "experiments/run_tpd_clean_v7_dch_formal800_2x5090_lane.sh",
    REPO_ROOT
    / "experiments/launch_tpd_clean_v7_dch_formal800_2x5090.sh",
    REPO_ROOT / "experiments/verify_tpd_clean_v7_dch_smoke_reports.py",
    # The DCH adapter genuinely imports the V6 exact entry.  Python eagerly
    # imports its V6 builder/model even though DCH rebinds every serialized
    # identity before constructing a run.
    REPO_ROOT / "experiments/train_tpd_clean_v6_exact.py",
    REPO_ROOT / "experiments/train_tpd_clean_v6.py",
    REPO_ROOT / "model/tpd_clean_v6.py",
    # Exact-resume control plane reached by the reused helpers and DCH runner.
    REPO_ROOT / "experiments/tpd_exact_runner.py",
    REPO_ROOT / "experiments/tpd_exact_resume.py",
    REPO_ROOT / "experiments/tpd_exact_epoch_journal.py",
    REPO_ROOT / "experiments/tpd_exact_training_runtime.py",
    REPO_ROOT / "experiments/tpd_extension_warm_start.py",
    # Eager and runtime training/data/model dependencies.
    REPO_ROOT / "experiments/train_tpd_pilot.py",
    REPO_ROOT / "experiments/fingerprint_tpd_training_data.py",
    REPO_ROOT / "model/SCTransNet.py",
    REPO_ROOT / "model/Config.py",
    REPO_ROOT / "model/tpd.py",
    REPO_ROOT / "dataset.py",
    REPO_ROOT / "utils.py",
    REPO_ROOT / "warmup_scheduler.py",
)

# Ranking remains the original five-metric policy.  All seventeen validation
# fields are nevertheless stored in selection records, derived checkpoints,
# metrics.jsonl, and completion summaries.
SELECTION_METRICS = ("pd", "fa", "tiny_pd", "miou", "val_loss")
STORED_VALIDATION_METRICS = (
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


file_sha256 = shared_exact.file_sha256
canonical_sha256 = shared_exact.canonical_sha256
load_json_mapping = shared_exact.load_json_mapping
PreparedData = shared_exact.PreparedData
prepare_data = shared_exact.prepare_data
split_fingerprints = shared_exact.split_fingerprints
data_fingerprints = shared_exact.data_fingerprints
normalized_gpu_uuid = shared_exact.normalized_gpu_uuid
visible_gpu_identity = shared_exact.visible_gpu_identity
configure_determinism = shared_exact.configure_determinism
write_or_verify_json = shared_exact.write_or_verify_json

PHYSICAL_GPU_UUIDS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}


def environment_contract(device: torch.device) -> dict[str, Any]:
    """Extend the shared environment record with a verified physical GPU."""

    payload = shared_exact.environment_contract(device)
    if device.type != "cuda":
        payload.update(
            {
                "physical_gpu_index": None,
                "physical_gpu_uuid": None,
                "physical_gpu_assignment_source": None,
            }
        )
        return payload

    physical_index = os.environ.get("TPD_DCH_PHYSICAL_GPU_INDEX")
    physical_uuid = os.environ.get("TPD_DCH_PHYSICAL_GPU_UUID")
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "TPD_DCH_PHYSICAL_GPU_INDEX must identify physical GPU 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            "TPD_DCH_PHYSICAL_GPU_UUID differs from the registered "
            f"physical GPU {physical_index}"
        )
    if payload.get("device_uuid") != expected_uuid:
        raise RuntimeError(
            "visible cuda:0 UUID differs from the registered physical GPU"
        )
    if payload.get("cuda_visible_devices") != expected_uuid:
        raise RuntimeError(
            "formal DCH CUDA_VISIBLE_DEVICES must use the registered UUID"
        )
    payload.update(
        {
            "physical_gpu_index": int(physical_index),
            "physical_gpu_uuid": expected_uuid,
            "physical_gpu_assignment_source": (
                "verified_worker_environment"
            ),
        }
    )
    return payload


def formal_contract() -> dict[str, Any]:
    """Return the immutable V7-DCH formal-training axes."""

    return {
        "epochs": FORMAL_EPOCHS,
        "eval_every": FORMAL_EVAL_EVERY,
        "workers": FORMAL_WORKERS,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        "initialization_modes": list(FORMAL_INITIALIZATION_MODES),
    }


def _validate_formal_args(args: argparse.Namespace) -> None:
    if args.epochs != FORMAL_EPOCHS:
        raise ValueError(
            f"V7-DCH formal exact training requires epochs={FORMAL_EPOCHS}"
        )
    if args.eval_every != FORMAL_EVAL_EVERY:
        raise ValueError(
            "V7-DCH formal exact training requires "
            f"eval_every={FORMAL_EVAL_EVERY}"
        )
    if args.workers != FORMAL_WORKERS:
        raise ValueError(
            f"V7-DCH formal exact training requires workers={FORMAL_WORKERS}"
        )
    if args.amp is not FORMAL_AMP:
        raise ValueError("V7-DCH formal exact training requires AMP=false")
    if args.eps != FORMAL_EPS:
        raise ValueError(
            f"V7-DCH formal exact training requires eps={FORMAL_EPS}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-resume TPD-Clean V7-DCH validation training"
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_CLEAN_V7_DCH_VARIANTS,
        required=True,
    )
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--run-tag", default="formal800_exact_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--eval-every", type=int, default=FORMAL_EVAL_EVERY)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--tiny-area", type=int, default=9)
    parser.add_argument("--eps", type=float, default=FORMAL_EPS)
    parser.set_defaults(amp=FORMAL_AMP)
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="explicitly permit CPU-only contract tests",
    )
    parser.add_argument(
        "--exact-source-lock",
        type=Path,
        default=DEFAULT_EXACT_SOURCE_LOCK_PATH,
    )
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)

    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--fresh", action="store_true")
    initialization.add_argument("--exact-resume", action="store_true")

    args = parser.parse_args(argv)
    if args.epochs != FORMAL_EPOCHS:
        parser.error(
            f"formal V7-DCH exact training requires --epochs={FORMAL_EPOCHS}"
        )
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2")
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error("--patch-size must be a positive multiple of 32")
    if args.workers != FORMAL_WORKERS:
        parser.error(
            f"formal V7-DCH exact training requires --workers={FORMAL_WORKERS}"
        )
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")
    if args.eval_every != FORMAL_EVAL_EVERY:
        parser.error(
            "formal V7-DCH exact training requires "
            f"--eval-every={FORMAL_EVAL_EVERY}"
        )
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in (0, 1)")
    if args.match_radius <= 0:
        parser.error("--match-radius must be positive")
    if args.tiny_area < 1:
        parser.error("--tiny-area must be positive")
    if not 0.0 <= args.min_lr <= args.base_lr or args.base_lr <= 0.0:
        parser.error("learning rates must satisfy 0 <= min-lr <= base-lr")
    if args.eps != FORMAL_EPS:
        parser.error(
            f"formal V7-DCH exact training requires --eps={FORMAL_EPS}"
        )
    if args.max_train_images is not None and args.max_train_images < 2:
        parser.error("--max-train-images must be >= 2")
    if args.max_val_images is not None and args.max_val_images < 1:
        parser.error("--max-val-images must be >= 1")
    _validate_formal_args(args)
    return args


def run_directory(args: argparse.Namespace) -> Path:
    return (
        args.output_root.resolve()
        / args.dataset
        / args.variant
        / f"seed_{args.seed}_{args.run_tag}"
    )


@dataclass(frozen=True)
class InitializationPlan:
    request: exact_runner.InitializationRequest
    contract: Mapping[str, Any]
    initial_model_state_sha256: str
    initial_rng: Mapping[str, Any] | None = None
    selection_policy: Mapping[str, Any] | None = None


def _existing_training_contract(directory: Path) -> dict[str, Any]:
    protocol = load_json_mapping(
        directory / "protocol.json",
        "existing V7-DCH exact protocol",
    )
    if protocol.get("schema") != ENTRY_SCHEMA:
        raise ValueError("existing protocol is not a V7-DCH exact run")
    try:
        training = protocol["run_identity"]["training_contract"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "existing V7-DCH protocol has no exact training contract"
        ) from exc
    if not isinstance(training, dict):
        raise ValueError("existing exact training contract is not a mapping")
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(
            f"existing exact training contract lacks fields: {missing}"
        )
    return copy.deepcopy(training)


def initialization_plan(
    args: argparse.Namespace,
    directory: Path,
    model: nn.Module,
) -> InitializationPlan:
    if args.fresh:
        return InitializationPlan(
            request=exact_runner.InitializationRequest.fresh(),
            contract=exact_runner.fresh_initialization_contract(),
            initial_model_state_sha256=(
                exact_runner.initial_model_state_sha256(model)
            ),
        )
    if args.exact_resume:
        training = _existing_training_contract(directory)
        initial_rng = training["initial_rng"]
        selection_policy = training["selection_policy"]
        if not isinstance(initial_rng, Mapping):
            raise ValueError("existing initial_rng is not a mapping")
        if not isinstance(selection_policy, Mapping):
            raise ValueError("existing selection_policy is not a mapping")
        return InitializationPlan(
            request=exact_runner.InitializationRequest.exact(),
            contract=copy.deepcopy(training["initialization_contract"]),
            initial_model_state_sha256=str(
                training["initial_model_state_sha256"]
            ),
            initial_rng=copy.deepcopy(dict(initial_rng)),
            selection_policy=copy.deepcopy(dict(selection_policy)),
        )
    raise RuntimeError(
        "V7-DCH exact entry supports only fresh or exact-resume"
    )


def _dch_architecture_manifest(
    variant: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "model": "model.SCTransNet.SCTransNet",
        "block": "model.tpd_clean_v7_dch.TPDCleanV7DCHBlock",
        "candidate_family": metadata["candidate_family"],
        "mainline_contract": metadata["mainline_contract"],
        "semantic_sources": metadata["semantic_sources"],
        "embedding_replacements": metadata["replaced_embeddings"],
        "embedding_topology": {
            "mtc.embeddings_1": {
                "channels": 32,
                "stride": 16,
                "blocks": 4,
                "evidence_nodes": 3,
            },
            "mtc.embeddings_2": {
                "channels": 64,
                "stride": 8,
                "blocks": 3,
                "evidence_nodes": 2,
            },
        },
        "phase_tied_projection_formula": metadata[
            "phase_tied_projection_formula"
        ],
        "context_code_formula": metadata["context_code_formula"],
        "context_headroom_formula": metadata["context_headroom_formula"],
        "fusion_equation": metadata["fusion_equation"],
        "zero_scale_first_order_reference": metadata[
            "zero_scale_first_order_reference"
        ],
        "eps": FORMAL_EPS,
        "formal_amp": FORMAL_AMP,
        "derived_projection_parameters": metadata[
            "derived_projection_parameters"
        ],
        "derived_projection_buffers": metadata[
            "derived_projection_buffers"
        ],
        "shallow_embedding_parameters": metadata[
            "shallow_embedding_parameters"
        ],
        "total_parameters": metadata["total_parameters"],
    }


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    if variant not in SUPPORTED_CLEAN_V7_DCH_VARIANTS:
        raise ValueError(f"unsupported V7-DCH exact variant: {variant!r}")
    if eps != FORMAL_EPS:
        raise ValueError(
            f"V7-DCH exact builder requires eps={FORMAL_EPS}"
        )
    model, raw_metadata = build_clean_v7_dch_model(variant, seed)
    metadata = copy.deepcopy(dict(raw_metadata))
    if metadata.get("variant") != variant:
        raise ValueError("V7-DCH builder metadata variant mismatch")
    for name, expected_blocks in (("embeddings_1", 4), ("embeddings_2", 3)):
        embedding = getattr(model.mtc, name, None)
        blocks = getattr(embedding, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != expected_blocks:
            raise TypeError(f"V7-DCH {name} has an invalid block topology")
        for index, block in enumerate(blocks):
            if not isinstance(block, TPDCleanV7DCHBlock):
                raise TypeError(
                    f"V7-DCH {name}.blocks[{index}] has the wrong type"
                )
            if block.eps != FORMAL_EPS:
                raise ValueError(
                    f"V7-DCH {name}.blocks[{index}] eps differs "
                    "from the formal contract"
                )
    metadata["formal_eps"] = FORMAL_EPS
    metadata["formal_amp"] = FORMAL_AMP
    metadata["architecture_manifest"] = _dch_architecture_manifest(
        variant,
        metadata,
    )
    return model, metadata


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if str(device) != "cuda:0":
            raise ValueError(
                "each exact process must use its single visible GPU as cuda:0"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "exact V7-DCH requires exactly one process-visible GPU"
            )
        visible_gpu_identity()
        if os.environ.get("PYTHONHASHSEED") != str(args.seed):
            raise RuntimeError(
                "the process must start with PYTHONHASHSEED equal to --seed"
            )
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != FORMAL_CUBLAS_WORKSPACE_CONFIG
        ):
            raise RuntimeError(
                "the process must start with CUBLAS_WORKSPACE_CONFIG="
                f"{FORMAL_CUBLAS_WORKSPACE_CONFIG}"
            )
    elif device.type != "cpu":
        raise ValueError("V7-DCH exact entry supports only cpu or cuda:0")
    elif not args.allow_cpu_smoke:
        raise ValueError("CPU execution requires --allow-cpu-smoke")
    return device


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
) -> dict[str, str]:
    exact_source_lock_path = Path(exact_source_lock_path).resolve()
    payload = load_json_mapping(
        exact_source_lock_path,
        "V7-DCH exact source lock",
    )
    if payload.get("schema") != EXACT_SOURCE_LOCK_SCHEMA:
        raise ValueError("V7-DCH exact source-lock schema mismatch")
    if (
        tuple(payload.get("variants", ()))
        != SUPPORTED_CLEAN_V7_DCH_VARIANTS
    ):
        raise ValueError("V7-DCH exact source-lock variant matrix mismatch")
    if payload.get("formal_contract") != formal_contract():
        raise ValueError("V7-DCH exact source-lock formal contract mismatch")
    if payload.get("training_data_sha256") != training_data_sha256:
        raise ValueError(
            "training data differs from the V7-DCH exact source lock"
        )

    locked_sources = payload.get("source_sha256")
    if not isinstance(locked_sources, dict):
        raise ValueError(
            "V7-DCH exact source lock has no source_sha256 mapping"
        )
    required_relative = {
        str(path.relative_to(REPO_ROOT)) for path in RUNTIME_SOURCE_PATHS
    }
    missing_sources = sorted(required_relative - set(locked_sources))
    if missing_sources:
        raise ValueError(
            "V7-DCH exact source lock omits runtime sources: "
            f"{missing_sources}"
        )

    verified_sources: dict[str, str] = {}
    for relative, expected_digest in locked_sources.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError(
                "V7-DCH exact source lock has an invalid source path"
            )
        path = (REPO_ROOT / relative).resolve()
        try:
            canonical_relative = str(path.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError(
                "V7-DCH exact source path escapes the repository: "
                f"{relative!r}"
            ) from exc
        if canonical_relative != relative:
            raise ValueError(
                "V7-DCH exact source path is not canonical: "
                f"{relative!r}"
            )
        actual_digest = file_sha256(path)
        if expected_digest != actual_digest:
            raise ValueError(
                "V7-DCH exact source lock differs for runtime source "
                f"{relative}"
            )
        verified_sources[relative] = actual_digest

    result = {
        "tpd_clean_v7_dch_exact_source_lock": file_sha256(
            exact_source_lock_path
        ),
        "training_data": training_data_sha256,
    }
    result.update(
        {
            f"exact_source:{relative}": digest
            for relative, digest in verified_sources.items()
        }
    )
    return result


def make_exact_run_spec(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_metadata: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    initialization_contract: Mapping[str, Any],
    initial_model_state_sha256: str,
    initial_rng: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    source_locks: Mapping[str, str],
    split_records: Mapping[str, exact_runner.OrderedFingerprint],
    data_records: Mapping[str, exact_runner.OrderedFingerprint],
    environment: Mapping[str, Any],
) -> exact_runner.ExactRunSpec:
    _validate_formal_args(args)
    manifest = model_metadata.get("architecture_manifest")
    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError(
            "V7-DCH exact builder metadata has no architecture manifest"
        )
    if manifest.get("eps") != FORMAL_EPS:
        raise ValueError("V7-DCH architecture manifest eps mismatch")
    return exact_runner.ExactRunSpec(
        run_id=(
            f"tpd-clean-v7-dch-exact:{args.dataset}:{args.variant}:"
            f"seed-{args.seed}:{args.run_tag}"
        ),
        variant=args.variant,
        dataset=args.dataset,
        seed=args.seed,
        split_seed=args.split_seed,
        builder_metadata=copy.deepcopy(dict(model_metadata)),
        builder_manifest_sha256=canonical_sha256(manifest),
        source_locks=dict(source_locks),
        split_fingerprints=dict(split_records),
        data_fingerprints=dict(data_records),
        optimizer=exact_runner.optimizer_contract(model, optimizer),
        scaler=exact_runner.scaler_contract(scaler, amp=FORMAL_AMP),
        initialization_contract=copy.deepcopy(dict(initialization_contract)),
        lr_schedule=exact_runner.ManualCosineSchedule(
            total_epochs=FORMAL_EPOCHS,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
        ),
        loss={
            "class": "torch.nn.modules.loss.BCELoss",
            "reduction": "mean",
            "input": "post_sigmoid_probability",
            "aggregate": "sum",
            "compute_dtype": "float32",
        },
        deep_supervision={
            "enabled": True,
            "expected_outputs": 6,
            "training_uses_all_outputs": True,
            "validation_uses_final_output": True,
        },
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        workers=FORMAL_WORKERS,
        amp=FORMAL_AMP,
        total_epochs=FORMAL_EPOCHS,
        eval_interval=FORMAL_EVAL_EVERY,
        metric_config={
            "threshold": args.threshold,
            "match_radius": args.match_radius,
            "tiny_area": args.tiny_area,
            "validation_batch_size": 1,
            "official_test_accessed": False,
        },
        environment=dict(environment),
        determinism={
            "entry_schema": ENTRY_SCHEMA,
            "formal_contract": formal_contract(),
            "workers": FORMAL_WORKERS,
            "amp": FORMAL_AMP,
            "eps": FORMAL_EPS,
            "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
            "explicit_cpu_loader_generator": True,
            "loader_generator_seed": args.seed,
            "manual_lr_schedule": True,
            "scheduler": None,
            "drop_last": False,
            "skip_singleton_batches": True,
            "eval_every_epoch": True,
            "val_fraction": args.val_fraction,
            "max_train_images": args.max_train_images,
            "max_val_images": args.max_val_images,
            "cpu_smoke_explicitly_allowed": args.allow_cpu_smoke,
            "training_subset_class": (
                f"{base.TrainingSubset.__module__}."
                f"{base.TrainingSubset.__qualname__}"
            ),
            "validation_subset_class": (
                f"{base.ValidationSubset.__module__}."
                f"{base.ValidationSubset.__qualname__}"
            ),
        },
        initial_model_state_sha256=initial_model_state_sha256,
        initial_rng=copy.deepcopy(dict(initial_rng)),
        selection_policy=copy.deepcopy(dict(selection_policy)),
    )


def _require_complete_validation_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        name for name in STORED_VALIDATION_METRICS if name not in metrics
    ]
    if missing:
        raise ValueError(
            f"V7-DCH validation metrics lack stored fields: {missing}"
        )
    return {
        name: copy.deepcopy(metrics[name])
        for name in STORED_VALIDATION_METRICS
    }


@dataclass(frozen=True)
class EvaluatorCheckpointAdapter:
    model_metadata: Mapping[str, Any]
    split_hashes: Mapping[str, str]

    def __call__(
        self,
        context: exact_runner.CompatibilityPayloadContext,
    ) -> Mapping[str, Any]:
        exact_payload = context.exact_payload
        identity = context.run_identity
        validation_metrics = _require_complete_validation_metrics(
            context.metrics
        )
        return {
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": identity["variant"],
            "dataset": identity["dataset"],
            "seed": identity["seed"],
            "split_seed": identity["split_seed"],
            "state_dict": copy.deepcopy(
                exact_payload["model"]["state_dict"]
            ),
            "optimizer": copy.deepcopy(
                exact_payload["optimizer"]["state_dict"]
            ),
            "scaler": copy.deepcopy(
                exact_payload["scaler"]["state_dict"]
            ),
            "scheduler": None,
            "validation_metrics": validation_metrics,
            "model_metadata": copy.deepcopy(dict(self.model_metadata)),
            "split_hashes": copy.deepcopy(dict(self.split_hashes)),
            "run_identity": copy.deepcopy(context.run_identity),
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }


class DCHExactRunner(exact_runner.ExactRunner):
    """Publish the every-epoch last checkpoint with the evaluator role."""

    def _legacy_payload(
        self,
        source: Mapping[str, Any],
        *,
        source_exact_checkpoint_sha256: str,
        role: str,
        metrics: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        if role == "last_completed_epoch":
            if self.normalized_spec["eval_interval"] != FORMAL_EVAL_EVERY:
                raise RuntimeError(
                    "last_evaluated_epoch requires evaluation every epoch"
                )
            role = "last_evaluated_epoch"
        return super()._legacy_payload(
            source,
            source_exact_checkpoint_sha256=source_exact_checkpoint_sha256,
            role=role,
            metrics=metrics,
            event=event,
        )


def six_output_bce_loss(
    outputs: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError(
            "V7-DCH exact training requires exactly six outputs"
        )
    loss = base.deep_supervision_loss(
        tuple(output.float() for output in outputs),
        target.float(),
        criterion,
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError(
            "V7-DCH exact six-output BCE is non-finite"
        )
    return loss


def training_arguments(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "variant",
        "dataset",
        "dataset_dir",
        "output_root",
        "run_tag",
        "device",
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
        "eps",
        "amp",
        "allow_cpu_smoke",
        "exact_source_lock",
        "max_train_images",
        "max_val_images",
    )
    return {name: getattr(args, name) for name in names}


def protocol_payload(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    normalization: Mapping[str, float],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ENTRY_SCHEMA,
        "formal_contract": formal_contract(),
        "arguments": training_arguments(args),
        "run_directory": directory,
        "model": dict(model_metadata),
        "normalization": dict(normalization),
        "run_identity": dict(run_identity),
        "selection_order_metrics": list(SELECTION_METRICS),
        "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
        "primary_selection_rule": [
            "maximum val Pd",
            "minimum val Fa on Pd ties",
            "maximum val tiny-Pd",
            "maximum val mIoU",
            "minimum val loss",
        ],
        "secondary_selection_rule": [
            "maximum val mIoU",
            "maximum val Pd",
            "minimum val Fa",
            "maximum val tiny-Pd",
            "minimum val loss",
        ],
        "checkpoint_policy": (
            "best.pth.tar is Pd-primary; best_miou.pth.tar is "
            "mIoU-secondary; last.pth.tar is the last evaluated epoch; "
            "exact journal is authoritative"
        ),
        "loss": "sum of BCE over six deep-supervision outputs",
        "optimizer": "Adam",
        "lr_schedule": "manual warmup then cosine; no scheduler object",
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }


def _load_complete_events(path: Path, epochs: int) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(events) != epochs or [
        event.get("epoch") for event in events
    ] != list(range(1, epochs + 1)):
        raise RuntimeError(
            "V7-DCH exact metrics are not a complete contiguous run"
        )
    for event in events:
        _require_complete_validation_metrics(event)
    return events


def completion_summary(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    events = _load_complete_events(
        directory / exact_runner.METRICS_FILENAME,
        FORMAL_EPOCHS,
    )
    pd_epoch = int(selection["primary"]["epoch"])
    miou_epoch = int(selection["secondary"]["epoch"])
    pd_metrics = _require_complete_validation_metrics(
        events[pd_epoch - 1]
    )
    miou_metrics = _require_complete_validation_metrics(
        events[miou_epoch - 1]
    )
    return {
        "schema": "sctransnet_tpd_clean_v7_dch_completion_summary_v1",
        "status": "complete",
        "variant": args.variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "formal_contract": formal_contract(),
        "selection_order_metrics": list(SELECTION_METRICS),
        "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
        "best_epoch": pd_epoch,
        "best_validation_metrics": pd_metrics,
        "best_pd_epoch": pd_epoch,
        "best_pd_validation_metrics": pd_metrics,
        "best_miou_epoch": miou_epoch,
        "best_miou_validation_metrics": miou_metrics,
        "primary_selection_metric": "validation Pd, then lower Fa",
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "model": dict(model_metadata),
        "split_hashes": dict(split_hashes),
        "skipped_singleton_batches": sum(
            int(event.get("skipped_singleton_batches", 0))
            for event in events
        ),
        "elapsed_seconds": sum(
            float(event["epoch_seconds"]) for event in events
        ),
        "best_checkpoint": directory / exact_runner.BEST_FILENAME,
        "best_miou_checkpoint": directory / exact_runner.BEST_MIOU_FILENAME,
        "last_checkpoint": directory / exact_runner.LAST_FILENAME,
    }


def _check_metrics(metrics: Mapping[str, Any]) -> None:
    _require_complete_validation_metrics(metrics)
    for name, value in metrics.items():
        if (
            isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            raise FloatingPointError(
                f"non-finite validation metric {name!r}: {value!r}"
            )


def run_training(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    configure_determinism()
    device = resolve_device(args)
    file_sha256(args.exact_source_lock.resolve())
    directory = run_directory(args)
    prepared = prepare_data(args)
    sources = source_lock_contract(
        prepared.training_data_sha256,
        args.exact_source_lock,
    )

    model, model_metadata = build_selected_model(
        args.variant,
        args.seed,
        eps=args.eps,
    )
    plan = initialization_plan(args, directory, model)
    model.to(device)
    train_set = base.TrainingSubset(
        prepared.dataset_dir,
        args.dataset,
        args.patch_size,
        prepared.train_ids,
        prepared.normalization,
    )
    val_set = base.ValidationSubset(
        prepared.dataset_root,
        prepared.val_ids,
        prepared.normalization,
    )

    base.seed_everything(args.seed)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=FORMAL_WORKERS,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=FORMAL_WORKERS,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=FORMAL_AMP)
    selection_policy = exact_runner.pd_miou_selection_policy(
        stored_metrics=STORED_VALIDATION_METRICS
    )
    actual_selection_policy = selection_policy.normalized()
    spec_selection_policy = (
        copy.deepcopy(dict(plan.selection_policy))
        if plan.selection_policy is not None
        else actual_selection_policy
    )
    initial_rng = (
        copy.deepcopy(dict(plan.initial_rng))
        if plan.initial_rng is not None
        else exact_runner.initial_rng_contract()
    )
    spec = make_exact_run_spec(
        args,
        model=model,
        model_metadata=model_metadata,
        optimizer=optimizer,
        scaler=scaler,
        initialization_contract=plan.contract,
        initial_model_state_sha256=plan.initial_model_state_sha256,
        initial_rng=initial_rng,
        selection_policy=spec_selection_policy,
        source_locks=sources,
        split_records=split_fingerprints(prepared),
        data_records=data_fingerprints(prepared),
        environment=environment_contract(device),
    )
    adapter = EvaluatorCheckpointAdapter(
        model_metadata=model_metadata,
        split_hashes=prepared.split_hashes,
    )
    runner = DCHExactRunner(
        directory,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=loader_generator,
        spec=spec,
        selection_policy=selection_policy,
        compatibility_payload_factory=adapter,
    )
    snapshot = runner.startup(plan.request)
    write_or_verify_json(directory / "split.json", prepared.split_manifest)
    write_or_verify_json(
        directory / "protocol.json",
        protocol_payload(
            args,
            directory=directory,
            model_metadata=model_metadata,
            normalization=prepared.normalization,
            run_identity=snapshot.run_identity,
        ),
    )

    print(
        f"START variant={args.variant} mode={snapshot.initialization_mode.value} "
        f"completed={snapshot.completed_epoch} next={snapshot.next_epoch} "
        f"device={device}",
        flush=True,
    )
    while snapshot.next_epoch is not None:
        control = runner.next_epoch_control()
        if not control.should_evaluate:
            raise RuntimeError(
                "formal V7-DCH exact training must evaluate each epoch"
            )
        epoch_started = time.time()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        skipped_singletons = 0
        for images, masks in train_loader:
            if images.shape[0] == 1:
                skipped_singletons += 1
                continue
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=False):
                outputs = model(images)
                loss = six_output_bce_loss(outputs, masks, criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count = int(images.shape[0])
            loss_sum += float(loss.detach().item()) * batch_count
            sample_count += batch_count
        if not sample_count:
            raise RuntimeError("no training samples were processed in this epoch")

        metrics = base.validate(
            model,
            val_loader,
            device,
            criterion,
            args.threshold,
            args.match_radius,
            args.tiny_area,
            FORMAL_AMP,
        )
        _check_metrics(metrics)
        snapshot = runner.commit_epoch(
            {
                "variant": args.variant,
                "train_loss": loss_sum / sample_count,
                "processed_train_samples": sample_count,
                "epoch_seconds": time.time() - epoch_started,
                "skipped_singleton_batches": skipped_singletons,
                **metrics,
            },
            extra_state={
                "variant": args.variant,
                "formal_eps": FORMAL_EPS,
                "processed_train_samples": sample_count,
                "skipped_singleton_batches": skipped_singletons,
            },
        )
        print(
            f"EPOCH {control.epoch:03d}/{FORMAL_EPOCHS} "
            f"loss={loss_sum / sample_count:.6f} "
            f"mIoU={float(metrics['miou']):.6f} "
            f"Pd={float(metrics['pd']):.6f} "
            f"Fa={float(metrics['fa']):.8f}",
            flush=True,
        )

    if snapshot.best_selection is None:
        raise RuntimeError(
            "completed V7-DCH exact run has no best selection"
        )
    summary = completion_summary(
        args,
        directory=directory,
        model_metadata=model_metadata,
        split_hashes=prepared.split_hashes,
        selection=snapshot.best_selection,
    )
    write_or_verify_json(directory / "summary.json", summary)
    print(
        f"COMPLETE variant={args.variant} "
        f"bestPdEpoch={summary['best_pd_epoch']} "
        f"bestMiouEpoch={summary['best_miou_epoch']}",
        flush=True,
    )
    return directory


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


__all__ = [
    "ARCHITECTURE_MANIFEST_SCHEMA",
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DCHExactRunner",
    "ENTRY_SCHEMA",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EvaluatorCheckpointAdapter",
    "FORMAL_AMP",
    "FORMAL_CUBLAS_WORKSPACE_CONFIG",
    "FORMAL_EPOCHS",
    "FORMAL_EPS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_WORKERS",
    "InitializationPlan",
    "PreparedData",
    "RUNTIME_SOURCE_PATHS",
    "SELECTION_METRICS",
    "STORED_VALIDATION_METRICS",
    "build_selected_model",
    "canonical_sha256",
    "completion_summary",
    "configure_determinism",
    "data_fingerprints",
    "environment_contract",
    "file_sha256",
    "formal_contract",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "normalized_gpu_uuid",
    "parse_args",
    "prepare_data",
    "protocol_payload",
    "resolve_device",
    "run_directory",
    "run_training",
    "six_output_bce_loss",
    "source_lock_contract",
    "split_fingerprints",
    "visible_gpu_identity",
]


if __name__ == "__main__":
    main()
