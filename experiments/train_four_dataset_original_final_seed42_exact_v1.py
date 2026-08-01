#!/usr/bin/env python3
"""Fixed-seed four-dataset Original/Final training entry.

This runner implements the executable core of
``SCTransNet_三数据集1000Epoch论文实验完整方案.md``:

* four existing train/test ``img_idx`` regimes;
* Original and frozen Final trained from scratch;
* the sole training seed is 42;
* 1,000 epochs, FP32, Adam, warm-up plus cosine decay;
* test-selected checkpoints over epochs 10,20,...,1000;
* online preservation of only ``best_miou`` and ``best_pd`` under
  ``/home/ly/SCTransNet_main/results``.

A single full training state is atomically overwritten every epoch for
automatic recovery and removed after successful completion.  It is not a
selected checkpoint.  The two selected model checkpoints are never discarded
when the rolling resume state advances.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_tpd_pilot as metric_base  # noqa: E402
from experiments.tpd_training_loss import (  # noqa: E402
    TPDTrainingLoss,
    compute_tpd_training_loss,
)


SCHEMA = "sctransnet_four_dataset_seed42_exact_v1"
TRAINING_SEED = 42
DATASETS = ("SIRST3", "NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
METHODS = ("original", "final")
FORMAL_EPOCHS = 1000
FORMAL_BEGIN_TEST = 10
FORMAL_EVAL_EVERY = 10
FORMAL_BATCH_SIZE = 16
FORMAL_PATCH_SIZE = 256
FORMAL_WORKERS = 0
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_THRESHOLD = 0.5
FORMAL_MATCH_RADIUS = 3.0
FORMAL_TINY_AREA = 9
FORMAL_TSS_WEIGHT = 0.005
FORMAL_AMP = False
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "four_dataset_seed42_v1"
DEFAULT_MANIFEST_ROOT = DEFAULT_RESULTS_ROOT / "manifests"
PROTOCOL_DOCUMENT = (
    REPO_ROOT / "SCTransNet_三数据集1000Epoch论文实验完整方案.md"
)
GPU_UUIDS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
METHOD_GPU = {"original": "2", "final": "3"}
LEGACY_NORMALIZATION = {
    "SIRST3": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "NUAA-SIRST": {
        "mean": 101.06385040283203,
        "std": 34.619606018066406,
    },
    "NUDT-SIRST": {
        "mean": 107.80905151367188,
        "std": 33.02274703979492,
    },
    "IRSTD-1K": {"mean": 87.4661865234375, "std": 39.71953201293945},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--tss-statistics", type=Path)
    parser.add_argument("--survival-pos-weight", type=float)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--begin-test", type=int, default=FORMAL_BEGIN_TEST)
    parser.add_argument("--eval-every", type=int, default=FORMAL_EVAL_EVERY)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=FORMAL_PATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument(
        "--warmup-epochs", type=int, default=FORMAL_WARMUP_EPOCHS
    )
    parser.add_argument("--threshold", type=float, default=FORMAL_THRESHOLD)
    parser.add_argument(
        "--match-radius", type=float, default=FORMAL_MATCH_RADIUS
    )
    parser.add_argument("--tiny-area", type=int, default=FORMAL_TINY_AREA)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--physical-gpu-index", choices=("2", "3"))
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    return parser.parse_args(argv)


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs: {actual!r} != {expected!r}")


def validate_args(args: argparse.Namespace) -> None:
    _require_equal("training seed", args.seed, TRAINING_SEED)
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")
    if args.epochs < 1 or args.begin_test < 1:
        raise ValueError("epoch controls must be positive")
    if args.batch_size < 1 or args.patch_size < 32 or args.workers < 0:
        raise ValueError("invalid loader configuration")
    if args.patch_size % 32:
        raise ValueError("patch size must be divisible by 32")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if args.base_lr <= 0 or args.min_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.method == "final":
        if args.survival_pos_weight is not None and args.survival_pos_weight <= 0:
            raise ValueError("--survival-pos-weight must be positive")
        if not args.smoke and args.tss_statistics is None:
            raise ValueError("formal Final training requires --tss-statistics")
    if args.smoke:
        if args.epochs > 2:
            raise ValueError("smoke runs are limited to two epochs")
        if args.max_train_images is None or args.max_test_images is None:
            raise ValueError("smoke requires train/test limits")
        return
    formal = {
        "epochs": FORMAL_EPOCHS,
        "begin_test": FORMAL_BEGIN_TEST,
        "eval_every": FORMAL_EVAL_EVERY,
        "batch_size": FORMAL_BATCH_SIZE,
        "patch_size": FORMAL_PATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "threshold": FORMAL_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "device": "cuda:0",
    }
    for name, expected in formal.items():
        _require_equal(f"formal {name}", getattr(args, name), expected)
    expected_index = METHOD_GPU[args.method]
    _require_equal("physical GPU assignment", args.physical_gpu_index, expected_index)
    expected_uuid = GPU_UUIDS[expected_index]
    _require_equal("expected GPU UUID", args.expected_gpu_uuid, expected_uuid)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_uint63(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def torch_save_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def normalize_gpu_uuid(value: Any) -> str:
    text = str(value).strip()
    return text if text.startswith("GPU-") else f"GPU-{text}"


def resolve_device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cpu":
        if not args.smoke:
            raise ValueError("formal training cannot use CPU")
        return device
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each formal worker must see exactly one CUDA device through "
            "CUDA_VISIBLE_DEVICES"
        )
    declared = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected = args.expected_gpu_uuid or GPU_UUIDS.get(
        str(args.physical_gpu_index)
    )
    if declared != expected:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES differs: {declared!r} != {expected!r}"
        )
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = normalize_gpu_uuid(getattr(properties, "uuid", ""))
    if actual_uuid != expected:
        raise RuntimeError(
            f"visible torch device UUID differs: {actual_uuid!r} != {expected!r}"
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].detach().cpu())
    if torch.cuda.is_available() and value.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [state.detach().cpu() for state in value["torch_cuda"]]
        )


def _import_runtime_components():
    from experiments.four_dataset_models_seed42_v1 import build_method_model
    from experiments.paper_four_dataset_v1 import (
        FourDatasetTestDataset,
        FourDatasetTrainDataset,
    )

    return build_method_model, FourDatasetTrainDataset, FourDatasetTestDataset


def _dataset_view(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def _set_dataset_epoch(dataset: Any, epoch: int) -> None:
    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    setter = getattr(target, "set_epoch", None)
    if not callable(setter):
        raise TypeError("training dataset does not expose set_epoch(epoch)")
    setter(epoch)


def _extract_hw(sizes: Any) -> tuple[int, int]:
    if isinstance(sizes, torch.Tensor):
        value = sizes.detach().cpu()
        if value.ndim == 2:
            return int(value[0, 0]), int(value[0, 1])
        if value.ndim == 1 and value.numel() == 2:
            return int(value[0]), int(value[1])
    if isinstance(sizes, (tuple, list)) and len(sizes) == 2:
        first, second = sizes
        if isinstance(first, torch.Tensor):
            first = first.reshape(-1)[0].item()
        if isinstance(second, torch.Tensor):
            second = second.reshape(-1)[0].item()
        return int(first), int(second)
    raise TypeError(f"unsupported collated size value: {type(sizes)!r}")


def final_prediction(outputs: Any) -> torch.Tensor:
    evaluator = getattr(outputs, "evaluator_prediction", None)
    if callable(evaluator):
        return evaluator()
    if isinstance(outputs, (tuple, list)):
        return outputs[-1]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"unsupported model output: {type(outputs)!r}")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    threshold: float,
    match_radius: float,
    tiny_area: int,
) -> dict[str, Any]:
    model.eval()
    accumulator = metric_base.ValidationMetrics(
        threshold,
        match_radius,
        tiny_area,
    )
    for images, masks, sizes, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height, width = _extract_hw(sizes)
        prediction = final_prediction(model(images))
        prediction = prediction[:, :, :height, :width]
        target = masks[:, :, :height, :width]
        loss = criterion(prediction.float(), target.float())
        accumulator.update(
            prediction[0, 0].float().cpu().numpy(),
            target[0, 0].float().cpu().numpy(),
            float(loss.item()),
        )
    metrics = dict(accumulator.compute())
    metrics["test_loss"] = float(metrics.pop("val_loss"))
    tiny = float(metrics["tiny_pd"])
    if not math.isfinite(tiny):
        metrics["tiny_pd"] = None
    return metrics


def _tiny_for_key(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("tiny_pd")
    if value is None:
        return -1.0
    number = float(value)
    return number if math.isfinite(number) else -1.0


def best_miou_key(metrics: Mapping[str, Any], epoch: int) -> tuple[float, ...]:
    return (
        float(metrics["miou"]),
        float(metrics["pd"]),
        -float(metrics["fa"]),
        float(metrics["niou"]),
        _tiny_for_key(metrics),
        -float(metrics["test_loss"]),
        -float(epoch),
    )


def best_pd_key(metrics: Mapping[str, Any], epoch: int) -> tuple[float, ...]:
    return (
        float(metrics["pd"]),
        -float(metrics["fa"]),
        _tiny_for_key(metrics),
        float(metrics["miou"]),
        float(metrics["niou"]),
        -float(metrics["test_loss"]),
        -float(epoch),
    )


def _selected_checkpoint_payload(
    *,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    role: str,
    metrics: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "epoch": epoch,
        "dataset": args.dataset,
        "method": args.method,
        "seed": TRAINING_SEED,
        "checkpoint_role": role,
        "selection_source": f"test_{args.dataset}",
        "test_selected": role in {"best_miou", "best_pd"},
        "selection_is_optimistic": role in {"best_miou", "best_pd"},
        "test_metrics": copy.deepcopy(dict(metrics)),
        "state_dict": cpu_state_dict(model),
        "model_metadata": copy.deepcopy(dict(model_metadata)),
        "protocol_sha256": protocol_sha256,
    }


def _latest_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    event: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    protocol_sha256: str,
    best_miou: Mapping[str, Any],
    best_pd: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "epoch": epoch,
        "dataset": args.dataset,
        "method": args.method,
        "seed": TRAINING_SEED,
        "checkpoint_role": "latest_resume",
        "state_dict": cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "rng_state": rng_state(),
        "event": copy.deepcopy(dict(event)),
        "model_metadata": copy.deepcopy(dict(model_metadata)),
        "protocol_sha256": protocol_sha256,
        "best_miou": copy.deepcopy(dict(best_miou)),
        "best_pd": copy.deepcopy(dict(best_pd)),
    }


def _load_tss_pos_weight(args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if args.method == "original":
        return 1.0, {"enabled": False}
    if args.survival_pos_weight is not None:
        if not args.smoke:
            raise ValueError(
                "formal Final run must load the frozen TSS statistics artifact"
            )
        return float(args.survival_pos_weight), {
            "enabled": True,
            "source": "smoke_cli",
        }
    if args.tss_statistics is None:
        if args.smoke:
            return 1.0, {"enabled": True, "source": "smoke_default"}
        raise ValueError("missing TSS statistics")
    path = args.tss_statistics.resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    dataset_payload: Any = payload
    if isinstance(payload, Mapping):
        if "datasets" in payload:
            dataset_payload = payload["datasets"].get(args.dataset)
        elif args.dataset in payload:
            dataset_payload = payload[args.dataset]
    if not isinstance(dataset_payload, Mapping):
        raise ValueError(f"TSS statistics have no dataset {args.dataset!r}")
    value = float(dataset_payload["survival_pos_weight"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid survival_pos_weight")
    return value, {
        "enabled": True,
        "source": str(path),
        "sha256": file_sha256(path),
        "dataset_record": copy.deepcopy(dict(dataset_payload)),
    }


def _protocol_payload(
    args: argparse.Namespace,
    *,
    model_metadata: Mapping[str, Any],
    tss_metadata: Mapping[str, Any],
    data_manifests: Mapping[str, Any],
    train_count: int,
    test_count: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "method": args.method,
        "training_seed": TRAINING_SEED,
        "epochs": args.epochs,
        "begin_test": args.begin_test,
        "eval_every": args.eval_every,
        "candidate_epochs": list(
            range(args.begin_test, args.epochs + 1, args.eval_every)
        ),
        "test_selected": True,
        "selection_is_optimistic": True,
        "checkpoint_roles": ["best_miou", "best_pd"],
        "rolling_resume_state": {
            "enabled_during_training": True,
            "overwritten_each_epoch": True,
            "removed_after_success": True,
            "selected_checkpoint": False,
        },
        "training": {
            "optimizer": "Adam",
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "warmup_epochs": args.warmup_epochs,
            "schedule": "manual_linear_warmup_then_cosine",
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "workers": args.workers,
            "precision": "FP32",
            "amp": FORMAL_AMP,
            "segmentation_loss": "ordered sum BCE over six outputs",
            "tss_weight": FORMAL_TSS_WEIGHT if args.method == "final" else 0.0,
        },
        "metrics": {
            "threshold": args.threshold,
            "match_radius": args.match_radius,
            "tiny_area": args.tiny_area,
            "best_miou_key": [
                "miou",
                "pd",
                "-fa",
                "niou",
                "tiny_pd",
                "-test_loss",
                "-epoch",
            ],
            "best_pd_key": [
                "pd",
                "-fa",
                "tiny_pd",
                "miou",
                "niou",
                "-test_loss",
                "-epoch",
            ],
        },
        "normalization": copy.deepcopy(LEGACY_NORMALIZATION[args.dataset]),
        "data_manifests": copy.deepcopy(dict(data_manifests)),
        "dataset_counts": {"train": train_count, "test": test_count},
        "model": copy.deepcopy(dict(model_metadata)),
        "tss": copy.deepcopy(dict(tss_metadata)),
        "scratch": True,
        "parent_checkpoint": None,
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
            "path": str(PROTOCOL_DOCUMENT),
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
        },
        "runtime_sources": {
            "runner": file_sha256(Path(__file__).resolve()),
        },
        "smoke": bool(args.smoke),
    }


def _run_directory(args: argparse.Namespace) -> Path:
    root = args.results_root.resolve()
    if args.smoke:
        root = root / "smoke"
    return root / "runs" / args.dataset / args.method / "seed_42"


def _load_data_manifest_lock(args: argparse.Namespace) -> dict[str, Any]:
    root = args.manifest_root.resolve()
    paths = {
        "correction": root / "nuaa_misc111_correction_v1.json",
        "imgidx": root / "four_dataset_imgidx_v1.json",
        "normalization": root / "four_dataset_legacy_norm_v1.json",
        "pair_audit": root / "four_dataset_pair_audit_v1.json",
        "pair_records": root / "four_dataset_pair_records_v1.jsonl",
        "gate": root / "four_dataset_data_gate_v1.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"required data manifests are missing under {root}: {missing}"
        )
    with paths["gate"].open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    for field in (
        "nuaa_dataset_ready",
        "four_dataset_suite_ready",
        "formal_training_and_evaluation_ready",
    ):
        if gate.get(field) is not True:
            raise ValueError(f"data gate {field} is not true")
    return {
        "root": str(root),
        "files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }


def _load_resume(
    *,
    args: argparse.Namespace,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    protocol_sha256: str,
) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if not path.exists():
        if args.resume == "required":
            raise FileNotFoundError(path)
        return 1, {}, {}, None
    if args.resume == "never":
        raise FileExistsError(
            f"resume checkpoint exists but --resume=never: {path}"
        )
    # Keep RNG bytes on CPU. Optimizer.load_state_dict casts optimizer state
    # tensors to the already-created parameter devices below.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name, expected in (
        ("schema", SCHEMA),
        ("dataset", args.dataset),
        ("method", args.method),
        ("seed", TRAINING_SEED),
        ("protocol_sha256", protocol_sha256),
    ):
        _require_equal(f"resume {name}", payload.get(name), expected)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng_state"])
    completed_epoch = int(payload["epoch"])
    event = payload.get("event")
    if not isinstance(event, Mapping):
        raise ValueError("resume state lacks its completed epoch event")
    _require_equal("resume event epoch", event.get("epoch"), completed_epoch)
    return (
        completed_epoch + 1,
        dict(payload.get("best_miou", {})),
        dict(payload.get("best_pd", {})),
        dict(event),
    )


def _read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(
                    f"blank line in metrics log {path}:{line_number}"
                )
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(
                    f"non-object metrics event {path}:{line_number}"
                )
            expected_epoch = len(events) + 1
            _require_equal(
                f"metrics epoch at line {line_number}",
                value.get("epoch"),
                expected_epoch,
            )
            events.append(value)
    return events


def _reconcile_metrics_log(
    *,
    path: Path,
    args: argparse.Namespace,
    start_epoch: int,
    resume_event: Mapping[str, Any] | None,
) -> None:
    events = _read_metrics_jsonl(path)
    if resume_event is None:
        if events:
            raise RuntimeError(
                "metrics exist without a rolling resume state; refusing to "
                "guess the missing optimizer/RNG state"
            )
        return
    completed_epoch = start_epoch - 1
    for field, expected in (
        ("schema", SCHEMA),
        ("dataset", args.dataset),
        ("method", args.method),
        ("seed", TRAINING_SEED),
        ("epoch", completed_epoch),
    ):
        _require_equal(
            f"resume event {field}",
            resume_event.get(field),
            expected,
        )
    if len(events) == completed_epoch:
        if canonical_sha256(events[-1]) != canonical_sha256(resume_event):
            raise RuntimeError(
                "metrics tail differs from the rolling resume event"
            )
        return
    if len(events) == completed_epoch - 1:
        append_jsonl(path, resume_event)
        return
    raise RuntimeError(
        "metrics/resume epoch mismatch: "
        f"metrics={len(events)}, resume={completed_epoch}"
    )


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    configure_determinism()
    device = resolve_device(args)
    build_method_model, TrainDataset, TestDataset = _import_runtime_components()
    run_dir = _run_directory(args)
    checkpoint_dir = run_dir / "checkpoints"
    resume_dir = run_dir / "resume"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / "run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"run is already active: {run_dir}") from error

    complete_path = run_dir / "summary.json"
    if complete_path.exists():
        with complete_path.open("r", encoding="utf-8") as handle:
            complete = json.load(handle)
        if complete.get("status") == "complete":
            completed_resume = (
                run_dir / "resume" / "latest_training_state.pth.tar"
            )
            if completed_resume.exists():
                completed_resume.unlink()
            print(f"ALREADY_COMPLETE {run_dir}", flush=True)
            return complete_path

    data_manifests = _load_data_manifest_lock(args)
    manifest_files = data_manifests["files"]
    seed_everything(TRAINING_SEED)
    model, model_metadata = build_method_model(
        args.method,
        seed=TRAINING_SEED,
        dataset_name=args.dataset,
    )
    model.to(device)
    normalization = copy.deepcopy(LEGACY_NORMALIZATION[args.dataset])
    train_dataset = TrainDataset(
        args.dataset,
        patch_size=args.patch_size,
        seed=TRAINING_SEED,
        dataset_root=args.data_root.resolve(),
        imgidx_manifest=manifest_files["imgidx"]["path"],
        normalization_manifest=manifest_files["normalization"]["path"],
        correction_manifest=manifest_files["correction"]["path"],
        return_metadata=False,
    )
    test_dataset = TestDataset(
        args.dataset,
        args.dataset,
        dataset_root=args.data_root.resolve(),
        imgidx_manifest=manifest_files["imgidx"]["path"],
        normalization_manifest=manifest_files["normalization"]["path"],
        correction_manifest=manifest_files["correction"]["path"],
        return_metadata=False,
    )
    _require_equal(
        "train normalization",
        train_dataset.normalization,
        normalization,
    )
    _require_equal(
        "test normalization",
        test_dataset.normalization,
        normalization,
    )
    train_view = _dataset_view(train_dataset, args.max_train_images)
    test_view = _dataset_view(test_dataset, args.max_test_images)
    test_loader = DataLoader(
        test_view,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    tss_pos_weight, tss_metadata = _load_tss_pos_weight(args)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    protocol = _protocol_payload(
        args,
        model_metadata=model_metadata,
        tss_metadata=tss_metadata,
        data_manifests=data_manifests,
        train_count=len(train_view),
        test_count=len(test_view),
        device=device,
    )
    protocol_sha256 = canonical_sha256(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        with protocol_path.open("r", encoding="utf-8") as handle:
            existing_protocol = json.load(handle)
        _require_equal(
            "existing protocol sha256",
            existing_protocol.get("protocol_sha256"),
            protocol_sha256,
        )
    else:
        write_json_atomic(protocol_path, protocol)

    latest_path = resume_dir / "latest_training_state.pth.tar"
    start_epoch, best_miou, best_pd, resume_event = _load_resume(
        args=args,
        path=latest_path,
        model=model,
        optimizer=optimizer,
        device=device,
        protocol_sha256=protocol_sha256,
    )
    metrics_path = run_dir / "metrics.jsonl"
    _reconcile_metrics_log(
        path=metrics_path,
        args=args,
        start_epoch=start_epoch,
        resume_event=resume_event,
    )
    if start_epoch == 1:
        seed_everything(TRAINING_SEED)
    started_at = time.time()
    print(
        f"START dataset={args.dataset} method={args.method} seed=42 "
        f"epochs={args.epochs} start={start_epoch} train={len(train_view)} "
        f"test={len(test_view)} device={device}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        learning_rate = metric_base.learning_rate_for_epoch(
            epoch,
            args.epochs,
            args.base_lr,
            args.min_lr,
            args.warmup_epochs,
        )
        metric_base.set_learning_rate(optimizer, learning_rate)
        _set_dataset_epoch(train_view, epoch)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(
            stable_uint63(TRAINING_SEED, args.dataset, "shuffle", epoch)
        )
        train_loader = DataLoader(
            train_view,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=loader_generator,
            drop_last=False,
        )
        model.train()
        total_sum = 0.0
        segmentation_sum = 0.0
        survival_sum = 0.0
        sample_count = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses: TPDTrainingLoss = compute_tpd_training_loss(
                outputs,
                masks,
                criterion,
                survival_weight=(
                    FORMAL_TSS_WEIGHT if args.method == "final" else 0.0
                ),
                survival_pos_weight=tss_pos_weight,
            )
            losses.total.backward()
            optimizer.step()
            count = int(images.shape[0])
            total_sum += float(losses.total.detach().item()) * count
            segmentation_sum += (
                float(losses.segmentation.detach().item()) * count
            )
            survival_sum += float(losses.survival.detach().item()) * count
            sample_count += count
        if sample_count != len(train_view):
            raise RuntimeError(
                f"processed sample count differs: {sample_count} != "
                f"{len(train_view)}"
            )

        event: dict[str, Any] = {
            "schema": SCHEMA,
            "epoch": epoch,
            "dataset": args.dataset,
            "method": args.method,
            "seed": TRAINING_SEED,
            "learning_rate": learning_rate,
            "train_total_loss": total_sum / sample_count,
            "train_segmentation_loss": segmentation_sum / sample_count,
            "train_survival_loss": survival_sum / sample_count,
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
            metrics = evaluate(
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
            current_miou_key = best_miou_key(metrics, epoch)
            current_pd_key = best_pd_key(metrics, epoch)
            previous_miou_key = tuple(best_miou.get("key", []))
            previous_pd_key = tuple(best_pd.get("key", []))
            new_best_miou = (
                not previous_miou_key or current_miou_key > previous_miou_key
            )
            new_best_pd = (
                not previous_pd_key or current_pd_key > previous_pd_key
            )
            if new_best_miou:
                payload = _selected_checkpoint_payload(
                    model=model,
                    args=args,
                    epoch=epoch,
                    role="best_miou",
                    metrics=metrics,
                    model_metadata=model_metadata,
                    protocol_sha256=protocol_sha256,
                )
                path = checkpoint_dir / "best_miou.pth.tar"
                torch_save_atomic(path, payload)
                best_miou = {
                    "epoch": epoch,
                    "key": list(current_miou_key),
                    "metrics": copy.deepcopy(metrics),
                    "path": str(path),
                }
            if new_best_pd:
                payload = _selected_checkpoint_payload(
                    model=model,
                    args=args,
                    epoch=epoch,
                    role="best_pd",
                    metrics=metrics,
                    model_metadata=model_metadata,
                    protocol_sha256=protocol_sha256,
                )
                path = checkpoint_dir / "best_pd.pth.tar"
                torch_save_atomic(path, payload)
                best_pd = {
                    "epoch": epoch,
                    "key": list(current_pd_key),
                    "metrics": copy.deepcopy(metrics),
                    "path": str(path),
                }
            event["new_best_miou"] = new_best_miou
            event["new_best_pd"] = new_best_pd
            event["best_miou_epoch"] = best_miou["epoch"]
            event["best_pd_epoch"] = best_pd["epoch"]

        event["epoch_seconds"] = time.time() - epoch_started

        latest_payload = _latest_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            event=event,
            model_metadata=model_metadata,
            protocol_sha256=protocol_sha256,
            best_miou=best_miou,
            best_pd=best_pd,
        )
        torch_save_atomic(latest_path, latest_payload)
        append_jsonl(metrics_path, event)
        write_json_atomic(
            run_dir / "progress.json",
            {
                "schema": SCHEMA,
                "status": "running" if epoch < args.epochs else "finalizing",
                "dataset": args.dataset,
                "method": args.method,
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
                f"EPOCH dataset={args.dataset} method={args.method} "
                f"epoch={epoch}/{args.epochs} "
                f"loss={event['train_total_loss']:.6f} "
                f"mIoU={event['miou']:.6f} Pd={event['pd']:.6f} "
                f"Fa={event['fa']:.8e} "
                f"bestMiou={best_miou['epoch']} bestPd={best_pd['epoch']} "
                f"seconds={event['epoch_seconds']:.2f}",
                flush=True,
            )
        elif epoch == 1 or epoch % 25 == 0:
            print(
                f"EPOCH dataset={args.dataset} method={args.method} "
                f"epoch={epoch}/{args.epochs} "
                f"loss={event['train_total_loss']:.6f} "
                f"seconds={event['epoch_seconds']:.2f}",
                flush=True,
            )

    required = {
        "best_miou": checkpoint_dir / "best_miou.pth.tar",
        "best_pd": checkpoint_dir / "best_pd.pth.tar",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"selected checkpoints are missing: {missing}")
    checkpoint_manifest = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in required.items()
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": args.dataset,
        "method": args.method,
        "seed": TRAINING_SEED,
        "epochs": args.epochs,
        "test_selected": True,
        "selection_is_optimistic": True,
        "best_miou": best_miou,
        "best_pd": best_pd,
        "checkpoints": checkpoint_manifest,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "metrics": str(metrics_path),
        "elapsed_seconds_this_invocation": time.time() - started_at,
        "completed_at_unix": time.time(),
    }
    write_json_atomic(complete_path, summary)
    if latest_path.exists():
        latest_path.unlink()
    write_json_atomic(
        run_dir / "progress.json",
        {
            "schema": SCHEMA,
            "status": "complete",
            "dataset": args.dataset,
            "method": args.method,
            "seed": TRAINING_SEED,
            "completed_epoch": args.epochs,
            "total_epochs": args.epochs,
            "summary": str(complete_path),
            "updated_at_unix": time.time(),
        },
    )
    print(
        f"COMPLETE dataset={args.dataset} method={args.method} "
        f"bestMiouEpoch={best_miou['epoch']} bestPdEpoch={best_pd['epoch']} "
        f"summary={complete_path}",
        flush=True,
    )
    return complete_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
