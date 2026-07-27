#!/usr/bin/env python3
"""Exact epoch-boundary training entry for the isolated TPD-Clean-v6 pair.

The model remains owned by :mod:`experiments.train_tpd_clean_v6`; this module
only binds the fixed formal training contract to
:mod:`experiments.tpd_exact_runner`.  Importing it creates no directory and
starts no training process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import fingerprint_tpd_training_data as data_fingerprint  # noqa: E402
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import train_tpd_pilot as base  # noqa: E402
from experiments.train_tpd_clean_v6 import build_clean_v6_model  # noqa: E402
from model.tpd_clean_v6 import (  # noqa: E402
    SUPPORTED_CLEAN_V6_VARIANTS,
    TPDCleanV6Block,
)


ENTRY_SCHEMA = "sctransnet_tpd_clean_v6_exact_entry_v1"
EXACT_SOURCE_LOCK_SCHEMA = "sctransnet_tpd_clean_v6_exact_source_lock_v1"
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_clean_v6_exact_architecture_manifest_v1"
)
DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT / "experiments/tpd_clean_v6_exact_source_lock.json"
)

FORMAL_EPOCHS = 800
FORMAL_EVAL_EVERY = 1
FORMAL_WORKERS = 0
FORMAL_AMP = False
FORMAL_EPS = 1e-6
FORMAL_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FORMAL_INITIALIZATION_MODES = ("fresh", "exact_resume")

RUNTIME_SOURCE_PATHS = (
    # V6 entry, builder, block, evaluator, smoke, protocol and tests.
    REPO_ROOT / "experiments/train_tpd_clean_v6_exact.py",
    REPO_ROOT / "experiments/freeze_tpd_clean_v6_exact_source_lock.py",
    REPO_ROOT / "experiments/train_tpd_clean_v6.py",
    REPO_ROOT / "model/tpd_clean_v6.py",
    REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    REPO_ROOT / "experiments/smoke_tpd_clean_v6.py",
    REPO_ROOT / "experiments/capture_tpd_clean_v6_smoke_report.py",
    REPO_ROOT / "experiments/verify_tpd_clean_v6_smoke_reports.py",
    REPO_ROOT / "experiments/run_tpd_clean_v6_formal800_2x5090_worker.sh",
    REPO_ROOT / "experiments/run_tpd_clean_v6_formal800_2x5090_lane.sh",
    REPO_ROOT / "experiments/launch_tpd_clean_v6_formal800_2x5090.sh",
    REPO_ROOT / "experiments/status_tpd_clean_v6_formal800_2x5090.sh",
    REPO_ROOT / "experiments/TPD_CLEAN_V6_PROTOCOL.md",
    REPO_ROOT / "tests/test_tpd_clean_v6.py",
    REPO_ROOT / "tests/test_train_tpd_clean_v6.py",
    REPO_ROOT / "tests/test_evaluate_tpd_clean_v6_pd_fa.py",
    REPO_ROOT / "tests/test_smoke_tpd_clean_v6.py",
    REPO_ROOT / "tests/test_verify_tpd_clean_v6_smoke_reports.py",
    REPO_ROOT / "tests/test_tpd_clean_v6_2x_runtime.py",
    REPO_ROOT / "tests/test_train_tpd_clean_v6_exact.py",
    REPO_ROOT / "tests/test_freeze_tpd_clean_v6_exact_source_lock.py",
    # Shared exact-resume control plane.
    REPO_ROOT / "experiments/tpd_exact_runner.py",
    REPO_ROOT / "experiments/tpd_exact_resume.py",
    REPO_ROOT / "experiments/tpd_exact_epoch_journal.py",
    REPO_ROOT / "experiments/tpd_exact_training_runtime.py",
    REPO_ROOT / "experiments/tpd_extension_warm_start.py",
    REPO_ROOT / "tests/test_tpd_exact_runner.py",
    # Transitive local training and evaluation runtime.
    REPO_ROOT / "experiments/train_tpd_pilot.py",
    REPO_ROOT / "experiments/fingerprint_tpd_training_data.py",
    REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py",
    REPO_ROOT / "model/SCTransNet.py",
    REPO_ROOT / "model/Config.py",
    REPO_ROOT / "model/tpd.py",
    REPO_ROOT / "dataset.py",
    REPO_ROOT / "utils.py",
    REPO_ROOT / "warmup_scheduler.py",
)
SELECTION_METRICS = ("pd", "fa", "tiny_pd", "miou", "val_loss")


def formal_contract() -> dict[str, Any]:
    """Return the immutable V6 formal-training axes owned by this entry."""

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
        raise ValueError(f"V6 formal exact training requires epochs={FORMAL_EPOCHS}")
    if args.eval_every != FORMAL_EVAL_EVERY:
        raise ValueError(
            f"V6 formal exact training requires eval_every={FORMAL_EVAL_EVERY}"
        )
    if args.workers != FORMAL_WORKERS:
        raise ValueError(
            f"V6 formal exact training requires workers={FORMAL_WORKERS}"
        )
    if args.amp is not FORMAL_AMP:
        raise ValueError("V6 formal exact training requires AMP=false")
    if args.eps != FORMAL_EPS:
        raise ValueError(f"V6 formal exact training requires eps={FORMAL_EPS}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-resume TPD-Clean-v6 internal-validation training"
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_CLEAN_V6_VARIANTS,
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
        default=REPO_ROOT / "experiments/results/tpd_clean_v6_exact_v1",
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
        parser.error(f"formal V6 exact training requires --epochs={FORMAL_EPOCHS}")
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2")
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error("--patch-size must be a positive multiple of 32")
    if args.workers != FORMAL_WORKERS:
        parser.error(
            f"formal V6 exact training requires --workers={FORMAL_WORKERS}"
        )
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")
    if args.eval_every != FORMAL_EVAL_EVERY:
        parser.error(
            f"formal V6 exact training requires --eval-every={FORMAL_EVAL_EVERY}"
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
        parser.error(f"formal V6 exact training requires --eps={FORMAL_EPS}")
    if args.max_train_images is not None and args.max_train_images < 2:
        parser.error("--max-train-images must be >= 2")
    if args.max_val_images is not None and args.max_val_images < 1:
        parser.error("--max-val-images must be >= 1")
    _validate_formal_args(args)
    return args


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        base.json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


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
        "existing V6 exact protocol",
    )
    try:
        training = protocol["run_identity"]["training_contract"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "existing V6 protocol has no exact training contract"
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
    raise RuntimeError("V6 exact entry supports only fresh or exact-resume")


def _v6_architecture_manifest(
    variant: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "model": "model.SCTransNet.SCTransNet",
        "block": "model.tpd_clean_v6.TPDCleanV6Block",
        "candidate_family": metadata["candidate_family"],
        "mainline_contract": metadata["mainline_contract"],
        "semantic_sources": metadata["semantic_sources"],
        "embedding_replacements": metadata["replaced_embeddings"],
        "embedding_topology": {
            "mtc.embeddings_1": {
                "channels": 32,
                "stride": 16,
                "blocks": 4,
            },
            "mtc.embeddings_2": {
                "channels": 64,
                "stride": 8,
                "blocks": 3,
            },
        },
        "phase_tied_projection_formula": metadata[
            "phase_tied_projection_formula"
        ],
        "context_code_formula": metadata["context_code_formula"],
        "context_headroom_formula": metadata["context_headroom_formula"],
        "fusion_equation": metadata["fusion_equation"],
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
    if variant not in SUPPORTED_CLEAN_V6_VARIANTS:
        raise ValueError(f"unsupported V6 exact variant: {variant!r}")
    if eps != FORMAL_EPS:
        raise ValueError(f"V6 exact builder requires eps={FORMAL_EPS}")
    model, raw_metadata = build_clean_v6_model(variant, seed)
    metadata = copy.deepcopy(dict(raw_metadata))
    if metadata.get("variant") != variant:
        raise ValueError("V6 builder metadata variant mismatch")
    for name, expected_blocks in (("embeddings_1", 4), ("embeddings_2", 3)):
        embedding = getattr(model.mtc, name, None)
        blocks = getattr(embedding, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != expected_blocks:
            raise TypeError(f"V6 {name} has an invalid block topology")
        for index, block in enumerate(blocks):
            if not isinstance(block, TPDCleanV6Block):
                raise TypeError(f"V6 {name}.blocks[{index}] has the wrong type")
            if block.eps != FORMAL_EPS:
                raise ValueError(
                    f"V6 {name}.blocks[{index}] eps differs from formal contract"
                )
    metadata["formal_eps"] = FORMAL_EPS
    metadata["formal_amp"] = FORMAL_AMP
    metadata["architecture_manifest"] = _v6_architecture_manifest(
        variant,
        metadata,
    )
    return model, metadata


def configure_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def normalized_gpu_uuid(value: Any) -> str:
    """Normalize PyTorch/NVML UUID text to the ``GPU-...`` form."""

    text = str(value).strip()
    if not text:
        raise ValueError("GPU UUID is empty")
    return text if text.startswith("GPU-") else f"GPU-{text}"


def visible_gpu_identity() -> tuple[str, Any]:
    declared = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        not isinstance(declared, str)
        or not declared.startswith("GPU-")
        or "," in declared
        or declared.strip() != declared
        or len(declared) <= len("GPU-")
    ):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain one GPU-... UUID")
    properties = torch.cuda.get_device_properties(0)
    torch_uuid = getattr(properties, "uuid", None)
    if torch_uuid is None:
        raise RuntimeError("torch device properties do not expose a GPU UUID")
    if normalized_gpu_uuid(torch_uuid) != declared:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES UUID differs from torch cuda:0 UUID"
        )
    return declared, properties


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
                "exact V6 requires exactly one process-visible GPU"
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
        raise ValueError("V6 exact entry supports only cpu or cuda:0")
    elif not args.allow_cpu_smoke:
        raise ValueError("CPU execution requires --allow-cpu-smoke")
    return device


def _file_record(path: Path) -> str:
    return f"{path.name}:{file_sha256(path)}"


def ordered_sample_records(
    dataset_root: Path,
    identifiers: Sequence[str],
) -> list[str]:
    records: list[str] = []
    for identifier in identifiers:
        image = data_fingerprint.resolve_unique_file(
            dataset_root / "images",
            identifier,
        )
        mask = data_fingerprint.resolve_unique_file(
            dataset_root / "masks",
            identifier,
        )
        records.append(
            f"{identifier}|image={_file_record(image)}|mask={_file_record(mask)}"
        )
    return records


def official_training_data_sha256(
    dataset_root: Path,
    dataset: str,
    identifiers: Sequence[str],
    index_bytes: bytes,
) -> str:
    index_path = dataset_root / "img_idx" / f"train_{dataset}.txt"
    indexed = [
        line.strip()
        for line in index_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if list(identifiers) != indexed:
        raise ValueError("training IDs differ from the official index order")
    digest = hashlib.sha256()
    data_fingerprint.update_field(
        digest,
        "schema",
        b"tpd-training-data-fingerprint-v1",
    )
    data_fingerprint.update_field(
        digest,
        f"img_idx/{index_path.name}",
        index_bytes,
    )
    for identifier in indexed:
        for directory_name in ("images", "masks"):
            path = data_fingerprint.resolve_unique_file(
                dataset_root / directory_name,
                identifier,
            )
            data_fingerprint.update_file(
                digest,
                f"{directory_name}/{path.name}",
                path,
            )
    return digest.hexdigest()


def read_official_training_index(
    dataset_root: Path,
    dataset: str,
) -> tuple[bytes, list[str]]:
    index_path = dataset_root / "img_idx" / f"train_{dataset}.txt"
    if not index_path.is_file() or index_path.is_symlink():
        raise FileNotFoundError(f"training index is not regular: {index_path}")
    index_bytes = index_path.read_bytes()
    identifiers = [
        line.strip()
        for line in index_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(identifiers) < 2 or len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "training index must contain at least two unique identifiers"
        )
    return index_bytes, identifiers


@dataclass(frozen=True)
class PreparedData:
    dataset_dir: Path
    dataset_root: Path
    train_ids: list[str]
    val_ids: list[str]
    normalization: dict[str, float]
    split_hashes: dict[str, str]
    split_manifest: dict[str, Any]
    training_data_sha256: str
    train_sample_records: list[str]
    val_sample_records: list[str]


def prepare_data(args: argparse.Namespace) -> PreparedData:
    dataset_dir = args.dataset_dir.resolve()
    dataset_root = dataset_dir / args.dataset
    index_bytes, identifiers = read_official_training_index(
        dataset_root,
        args.dataset,
    )
    mask_stats = [
        base.inspect_mask(
            dataset_root / "masks",
            identifier,
            args.tiny_area,
        )
        for identifier in identifiers
    ]
    full_train_ids, full_val_ids = base.stratified_split(
        mask_stats,
        args.val_fraction,
        args.split_seed,
    )
    train_ids = base.subset_for_smoke(
        full_train_ids.copy(),
        args.max_train_images,
        args.seed + 101,
    )
    val_ids = base.subset_for_smoke(
        full_val_ids.copy(),
        args.max_val_images,
        args.seed + 202,
    )
    split_hashes = {
        "full_internal_train_sha256": base.identifier_hash(full_train_ids),
        "full_internal_val_sha256": base.identifier_hash(full_val_ids),
        "used_train_sha256": base.identifier_hash(train_ids),
        "used_val_sha256": base.identifier_hash(val_ids),
    }
    stats_by_id = {item.identifier: item for item in mask_stats}
    strata = sorted({item.stratum for item in mask_stats})
    split_manifest = {
        "dataset": args.dataset,
        "source": f"img_idx/train_{args.dataset}.txt",
        "official_test_accessed": False,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "full_official_train_count": len(identifiers),
        "full_internal_train_count": len(full_train_ids),
        "full_internal_val_count": len(full_val_ids),
        "used_train_count": len(train_ids),
        "used_val_count": len(val_ids),
        "hashes": split_hashes,
        "full_internal_train_ids": full_train_ids,
        "full_internal_val_ids": full_val_ids,
        "used_train_ids": train_ids,
        "used_val_ids": val_ids,
        "stratum_counts": {
            "official_train": {
                key: sum(item.stratum == key for item in mask_stats)
                for key in strata
            },
            "used_train": {
                key: sum(stats_by_id[item].stratum == key for item in train_ids)
                for key in strata
            },
            "used_val": {
                key: sum(stats_by_id[item].stratum == key for item in val_ids)
                for key in strata
            },
        },
        "mask_statistics": [asdict(item) for item in mask_stats],
    }
    normalization = base.training_only_normalization(dataset_root, train_ids)
    training_digest = official_training_data_sha256(
        dataset_root,
        args.dataset,
        identifiers,
        index_bytes,
    )
    return PreparedData(
        dataset_dir=dataset_dir,
        dataset_root=dataset_root,
        train_ids=train_ids,
        val_ids=val_ids,
        normalization=normalization,
        split_hashes=split_hashes,
        split_manifest=split_manifest,
        training_data_sha256=training_digest,
        train_sample_records=ordered_sample_records(dataset_root, train_ids),
        val_sample_records=ordered_sample_records(dataset_root, val_ids),
    )


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
) -> dict[str, str]:
    exact_source_lock_path = Path(exact_source_lock_path).resolve()
    payload = load_json_mapping(
        exact_source_lock_path,
        "V6 exact source lock",
    )
    if payload.get("schema") != EXACT_SOURCE_LOCK_SCHEMA:
        raise ValueError("V6 exact source-lock schema mismatch")
    if tuple(payload.get("variants", ())) != SUPPORTED_CLEAN_V6_VARIANTS:
        raise ValueError("V6 exact source-lock variant matrix mismatch")
    if payload.get("formal_contract") != formal_contract():
        raise ValueError("V6 exact source-lock formal contract mismatch")
    if payload.get("training_data_sha256") != training_data_sha256:
        raise ValueError("training data differs from the V6 exact source lock")

    locked_sources = payload.get("source_sha256")
    if not isinstance(locked_sources, dict):
        raise ValueError("V6 exact source lock has no source_sha256 mapping")
    required_relative = {
        str(path.relative_to(REPO_ROOT)) for path in RUNTIME_SOURCE_PATHS
    }
    missing_sources = sorted(required_relative - set(locked_sources))
    if missing_sources:
        raise ValueError(
            f"V6 exact source lock omits runtime sources: {missing_sources}"
        )

    verified_sources: dict[str, str] = {}
    for relative, expected_digest in locked_sources.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("V6 exact source lock has an invalid source path")
        path = (REPO_ROOT / relative).resolve()
        try:
            canonical_relative = str(path.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError(
                f"V6 exact source path escapes the repository: {relative!r}"
            ) from exc
        if canonical_relative != relative:
            raise ValueError(
                f"V6 exact source path is not canonical: {relative!r}"
            )
        actual_digest = file_sha256(path)
        if expected_digest != actual_digest:
            raise ValueError(
                f"V6 exact source lock differs for runtime source {relative}"
            )
        verified_sources[relative] = actual_digest

    result = {
        "tpd_clean_v6_exact_source_lock": file_sha256(
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


def split_fingerprints(
    prepared: PreparedData,
) -> dict[str, exact_runner.OrderedFingerprint]:
    manifest = prepared.split_manifest
    return {
        "full_train": exact_runner.OrderedFingerprint.from_values(
            "full_train",
            manifest["full_internal_train_ids"],
        ),
        "full_validation": exact_runner.OrderedFingerprint.from_values(
            "full_validation",
            manifest["full_internal_val_ids"],
        ),
        "train": exact_runner.OrderedFingerprint.from_values(
            "train",
            prepared.train_ids,
        ),
        "validation": exact_runner.OrderedFingerprint.from_values(
            "validation",
            prepared.val_ids,
        ),
    }


def data_fingerprints(
    prepared: PreparedData,
) -> dict[str, exact_runner.OrderedFingerprint]:
    normalization = json.dumps(
        prepared.normalization,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "official_training_data": exact_runner.OrderedFingerprint.from_values(
            "official_training_data",
            (prepared.training_data_sha256,),
        ),
        "train_samples": exact_runner.OrderedFingerprint.from_values(
            "train_samples",
            prepared.train_sample_records,
        ),
        "validation_samples": exact_runner.OrderedFingerprint.from_values(
            "validation_samples",
            prepared.val_sample_records,
        ),
        "normalization": exact_runner.OrderedFingerprint.from_values(
            "normalization",
            (normalization,),
        ),
    }


def environment_contract(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    device_uuid, _properties = (
        visible_gpu_identity() if cuda else (None, None)
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "logical_device": str(device),
        "visible_cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
        "device_uuid": device_uuid,
        "device_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda else None
        ),
        "cuda_visible_devices": (
            os.environ.get("CUDA_VISIBLE_DEVICES") if cuda else None
        ),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


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
        raise ValueError("V6 exact builder metadata has no architecture manifest")
    if manifest.get("eps") != FORMAL_EPS:
        raise ValueError("V6 exact architecture manifest eps mismatch")
    return exact_runner.ExactRunSpec(
        run_id=(
            f"tpd-clean-v6-exact:{args.dataset}:{args.variant}:"
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
            "validation_metrics": copy.deepcopy(context.metrics),
            "model_metadata": copy.deepcopy(dict(self.model_metadata)),
            "split_hashes": copy.deepcopy(dict(self.split_hashes)),
            "run_identity": copy.deepcopy(context.run_identity),
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }


class V6ExactRunner(exact_runner.ExactRunner):
    """Use the V6 protocol name for the derived every-epoch last checkpoint."""

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
        raise RuntimeError("V6 exact training requires exactly six outputs")
    loss = base.deep_supervision_loss(
        tuple(output.float() for output in outputs),
        target.float(),
        criterion,
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("V6 exact six-output BCE is non-finite")
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


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_or_verify_json(path: Path, payload: Mapping[str, Any]) -> None:
    normalized = base.json_ready(dict(payload))
    content = (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists() or path.is_symlink():
        existing = load_json_mapping(path, path.name)
        if existing != normalized:
            raise ValueError(f"existing {path.name} differs from exact contract")
        return
    _atomic_write(path, content)


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
            "mIoU-secondary; exact journal is authoritative"
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
        raise RuntimeError("V6 exact metrics are not a complete contiguous run")
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
    pd_event = events[pd_epoch - 1]
    miou_event = events[miou_epoch - 1]
    pd_metrics = {name: pd_event[name] for name in SELECTION_METRICS}
    miou_metrics = {name: miou_event[name] for name in SELECTION_METRICS}
    return {
        "status": "complete",
        "variant": args.variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "formal_contract": formal_contract(),
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
        "elapsed_seconds": sum(float(event["epoch_seconds"]) for event in events),
        "best_checkpoint": directory / exact_runner.BEST_FILENAME,
        "best_miou_checkpoint": directory / exact_runner.BEST_MIOU_FILENAME,
        "last_checkpoint": directory / exact_runner.LAST_FILENAME,
    }


def _check_metrics(metrics: Mapping[str, Any]) -> None:
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
    selection_policy = exact_runner.pd_miou_selection_policy()
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
    runner = V6ExactRunner(
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
            raise RuntimeError("formal V6 exact training must evaluate each epoch")
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
        raise RuntimeError("completed V6 exact run has no best selection")
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
    "V6ExactRunner",
    "build_selected_model",
    "canonical_sha256",
    "completion_summary",
    "data_fingerprints",
    "environment_contract",
    "file_sha256",
    "formal_contract",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "normalized_gpu_uuid",
    "official_training_data_sha256",
    "ordered_sample_records",
    "parse_args",
    "prepare_data",
    "protocol_payload",
    "read_official_training_index",
    "resolve_device",
    "run_training",
    "six_output_bce_loss",
    "source_lock_contract",
    "split_fingerprints",
    "visible_gpu_identity",
]


if __name__ == "__main__":
    main()
