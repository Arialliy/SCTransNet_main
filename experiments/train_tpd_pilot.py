#!/usr/bin/env python3
"""Leakage-free pilot runner for shallow SCTransNet embedding variants.

This runner deliberately reads only the official training index. It creates a
deterministic image-level validation split, selects ``best.pth.tar`` on that
validation split, and never opens the official test index or test samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage import measure
from torch.nn import init
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import TrainSetLoader  # noqa: E402
from model.Config import get_SCTrans_config  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402
from model.tpd import (  # noqa: E402
    SUPPORTED_VARIANTS,
    parameter_count,
    replace_shallow_embeddings,
)
from utils import Normalized, PadImg  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only SCTransNet TPD patch-embedding pilot"
    )
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments/results/tpd_pe_pilot_v1")
    parser.add_argument("--run-tag", default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--tiny-area", type=int, default=9)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2 because the reference runner skips singleton batches")
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error("--patch-size must be a positive multiple of 32")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")
    if args.eval_every < 1:
        parser.error("--eval-every must be >= 1")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in (0, 1)")
    if args.max_train_images is not None and args.max_train_images < 2:
        parser.error("--max-train-images must be >= 2")
    if args.max_val_images is not None and args.max_val_images < 1:
        parser.error("--max-val-images must be >= 1")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), ensure_ascii=False) + "\n")


def read_training_ids(dataset_root: Path, dataset_name: str) -> List[str]:
    train_index = dataset_root / "img_idx" / f"train_{dataset_name}.txt"
    if not train_index.is_file():
        raise FileNotFoundError(f"Training index not found: {train_index}")
    identifiers = [line.strip() for line in train_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate identifiers found in {train_index}")
    if len(identifiers) < 2:
        raise ValueError("At least two official-training images are required")
    return identifiers


def resolve_sample_file(directory: Path, identifier: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{identifier}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {identifier!r} under {directory}")


def training_only_normalization(
    dataset_root: Path, training_ids: Sequence[str]
) -> Dict[str, float]:
    """Compute normalization statistics from the actual internal train IDs."""
    means: List[float] = []
    standard_deviations: List[float] = []
    for identifier in training_ids:
        path = resolve_sample_file(dataset_root / "images", identifier)
        image = np.asarray(Image.open(path).convert("I"), dtype=np.float32)
        means.append(float(image.mean()))
        standard_deviations.append(float(image.std()))
    standard_deviation = float(np.mean(standard_deviations))
    if standard_deviation <= 0:
        raise ValueError("Non-positive internal-training-only normalization std")
    return {"mean": float(np.mean(means)), "std": standard_deviation}


@dataclass(frozen=True)
class MaskStats:
    identifier: str
    height: int
    width: int
    target_count: int
    target_pixels: int
    tiny_target_count: int
    minimum_target_area: int
    stratum: str


def inspect_mask(mask_dir: Path, identifier: str, tiny_area: int) -> MaskStats:
    path = resolve_sample_file(mask_dir, identifier)
    mask = np.asarray(Image.open(path), dtype=np.float32)
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    binary = (mask / 255.0) > 0.5
    regions = measure.regionprops(measure.label(binary, connectivity=2))
    areas = [int(region.area) for region in regions]
    target_count = len(areas)
    target_pixels = int(binary.sum())
    tiny_count = sum(area <= tiny_area for area in areas)
    minimum_area = min(areas) if areas else 0
    if not areas:
        stratum = "empty"
    elif tiny_count and target_count == 1:
        stratum = "tiny_single"
    elif tiny_count:
        stratum = "tiny_multi"
    elif minimum_area <= 25:
        stratum = "small_non_tiny"
    else:
        stratum = "larger"
    return MaskStats(
        identifier=identifier,
        height=int(binary.shape[0]),
        width=int(binary.shape[1]),
        target_count=target_count,
        target_pixels=target_pixels,
        tiny_target_count=tiny_count,
        minimum_target_area=minimum_area,
        stratum=stratum,
    )


def stratified_split(
    stats: Sequence[MaskStats], val_fraction: float, split_seed: int
) -> Tuple[List[str], List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for item in stats:
        groups[item.stratum].append(item.identifier)

    rng = random.Random(split_seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])

    target_val = min(len(stats) - 1, max(1, int(round(len(stats) * val_fraction))))
    allocation: Dict[str, int] = {}
    remainder: Dict[str, float] = {}
    for key, identifiers in groups.items():
        raw = len(identifiers) * val_fraction
        maximum = max(0, len(identifiers) - 1)
        initial = min(maximum, int(math.floor(raw)))
        if len(identifiers) >= 2 and initial == 0:
            initial = 1
        allocation[key] = initial
        remainder[key] = raw - math.floor(raw)

    while sum(allocation.values()) > target_val:
        candidates = [key for key, count in allocation.items() if count > 0]
        key = min(candidates, key=lambda item: (remainder[item], len(groups[item]), item))
        allocation[key] -= 1
    while sum(allocation.values()) < target_val:
        candidates = [
            key for key in groups if allocation[key] < max(0, len(groups[key]) - 1)
        ]
        if not candidates:
            break
        key = max(candidates, key=lambda item: (remainder[item], len(groups[item]), item))
        allocation[key] += 1

    validation: List[str] = []
    training: List[str] = []
    for key in sorted(groups):
        cut = allocation[key]
        validation.extend(groups[key][:cut])
        training.extend(groups[key][cut:])

    if len(validation) < target_val:
        rng.shuffle(training)
        needed = target_val - len(validation)
        validation.extend(training[:needed])
        training = training[needed:]

    rng.shuffle(training)
    rng.shuffle(validation)
    if not training or not validation or set(training) & set(validation):
        raise RuntimeError("Invalid internal training/validation split")
    return training, validation


def subset_for_smoke(identifiers: List[str], limit: int | None, seed: int) -> List[str]:
    if limit is None or limit >= len(identifiers):
        return identifiers
    shuffled = identifiers.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def identifier_hash(identifiers: Iterable[str]) -> str:
    canonical = "\n".join(sorted(identifiers)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class TrainingSubset(TrainSetLoader):
    """Reference augmentation pipeline restricted to explicit training IDs."""

    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        patch_size: int,
        identifiers: Sequence[str],
        img_norm_cfg: Dict[str, float],
    ) -> None:
        super().__init__(str(dataset_dir), dataset_name, patch_size, img_norm_cfg)
        self.train_list = list(identifiers)


class ValidationSubset(Dataset):
    """Full-resolution validation loader that never reads the test index."""

    def __init__(
        self,
        dataset_root: Path,
        identifiers: Sequence[str],
        img_norm_cfg: Dict[str, float],
    ) -> None:
        self.dataset_root = dataset_root
        self.identifiers = list(identifiers)
        self.img_norm_cfg = img_norm_cfg

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        identifier = self.identifiers[index]
        image_path = resolve_sample_file(self.dataset_root / "images", identifier)
        mask_path = resolve_sample_file(self.dataset_root / "masks", identifier)
        image = np.asarray(Image.open(image_path).convert("I"), dtype=np.float32)
        mask = np.asarray(Image.open(mask_path), dtype=np.float32)
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        if image.shape != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {identifier}: {image.shape} != {mask.shape}"
            )
        height, width = image.shape
        image = PadImg(Normalized(image, self.img_norm_cfg))
        mask = PadImg(mask / 255.0)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image[None, :, :]))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None, :, :]))
        return image_tensor, mask_tensor, torch.tensor([height, width]), identifier


def weights_init_kaiming(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname and getattr(module, "weight", None) is not None:
        init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
    elif "Linear" in classname and getattr(module, "weight", None) is not None:
        init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
    elif "BatchNorm" in classname and getattr(module, "weight", None) is not None:
        init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            init.constant_(module.bias.data, 0.0)


def model_checksum(model: nn.Module, exclude_shallow: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if exclude_shallow and name.startswith(("mtc.embeddings_1.", "mtc.embeddings_2.")):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(variant: str, seed: int) -> Tuple[SCTransNet, Dict[str, Any]]:
    seed_everything(seed)
    model = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(weights_init_kaiming)
    replacements = replace_shallow_embeddings(model, variant)
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    shallow_parameters = sum(parameter_count(module) for module in replacements.values())
    if variant == "original":
        shallow_parameters = parameter_count(model.mtc.embeddings_1) + parameter_count(model.mtc.embeddings_2)
    metadata = {
        "variant": variant,
        "total_parameters": parameter_count(model),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "shallow_embedding_parameters": shallow_parameters,
        "shared_initialization_sha256": model_checksum(model, exclude_shallow=True),
        "full_initialization_sha256": model_checksum(model),
    }
    return model, metadata


def deep_supervision_loss(outputs: Any, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        return sum(criterion(output, target) for output in outputs)
    return criterion(outputs, target)


def final_prediction(outputs: Any) -> torch.Tensor:
    return outputs[-1] if isinstance(outputs, (tuple, list)) else outputs


def learning_rate_for_epoch(
    epoch: int, total_epochs: int, base_lr: float, min_lr: float, warmup_epochs: int
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    decay_epochs = total_epochs - warmup_epochs
    if decay_epochs <= 0:
        return base_lr
    progress = (epoch - warmup_epochs) / decay_epochs
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def pd_selection_key(metrics: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """Primary checkpoint order requested for target detection.

    Pd is primary. Fa breaks Pd ties so an over-segmented checkpoint cannot
    win a tie; tiny-Pd, mIoU, and validation loss are subsequent tie-breakers.
    """
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
    """Secondary segmentation checkpoint retained for transparent comparison."""
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


class ValidationMetrics:
    def __init__(self, threshold: float, match_radius: float, tiny_area: int) -> None:
        self.threshold = threshold
        self.match_radius = match_radius
        self.tiny_area = tiny_area
        self.intersection = 0
        self.union = 0
        self.true_positive_pixels = 0
        self.false_positive_pixels = 0
        self.false_negative_pixels = 0
        self.image_ious: List[float] = []
        self.losses: List[float] = []
        self.target_count = 0
        self.matched_target_count = 0
        self.tiny_target_count = 0
        self.matched_tiny_target_count = 0
        self.predicted_object_count = 0
        self.unmatched_predicted_object_count = 0
        self.unmatched_predicted_pixels = 0
        self.valid_pixels = 0

    def update(self, probability: np.ndarray, target: np.ndarray, loss: float) -> None:
        prediction = probability > self.threshold
        target_binary = target > 0.5
        intersection = int(np.logical_and(prediction, target_binary).sum())
        union = int(np.logical_or(prediction, target_binary).sum())
        self.intersection += intersection
        self.union += union
        self.true_positive_pixels += intersection
        self.false_positive_pixels += int(np.logical_and(prediction, ~target_binary).sum())
        self.false_negative_pixels += int(np.logical_and(~prediction, target_binary).sum())
        self.image_ious.append(1.0 if union == 0 else intersection / union)
        self.losses.append(float(loss))
        self.valid_pixels += int(target_binary.size)

        predicted_regions = measure.regionprops(measure.label(prediction, connectivity=2))
        target_regions = measure.regionprops(measure.label(target_binary, connectivity=2))
        self.predicted_object_count += len(predicted_regions)
        self.target_count += len(target_regions)
        self.tiny_target_count += sum(region.area <= self.tiny_area for region in target_regions)

        matched_targets: set[int] = set()
        matched_predictions: set[int] = set()
        if target_regions and predicted_regions:
            distances = np.empty((len(target_regions), len(predicted_regions)), dtype=np.float64)
            for target_index, target_region in enumerate(target_regions):
                target_centroid = np.asarray(target_region.centroid)
                for predicted_index, predicted_region in enumerate(predicted_regions):
                    distances[target_index, predicted_index] = np.linalg.norm(
                        np.asarray(predicted_region.centroid) - target_centroid
                    )

            # Add one zero-cost dummy column per GT. Each valid real match has
            # a large negative reward, yielding maximum cardinality first and
            # minimum total centroid distance second.
            cardinality_reward = (
                (min(len(target_regions), len(predicted_regions)) + 1)
                * max(1.0, self.match_radius)
            )
            real_cost = np.where(
                distances < self.match_radius,
                distances - cardinality_reward,
                cardinality_reward,
            )
            assignment_cost = np.concatenate(
                (real_cost, np.zeros((len(target_regions), len(target_regions)))), axis=1
            )
            assigned_targets, assigned_columns = linear_sum_assignment(assignment_cost)
            for target_index, column_index in zip(assigned_targets, assigned_columns):
                if (
                    column_index < len(predicted_regions)
                    and distances[target_index, column_index] < self.match_radius
                ):
                    matched_targets.add(int(target_index))
                    matched_predictions.add(int(column_index))

        self.matched_target_count += len(matched_targets)
        self.matched_tiny_target_count += sum(
            target_regions[index].area <= self.tiny_area for index in matched_targets
        )
        unmatched = [
            region for index, region in enumerate(predicted_regions) if index not in matched_predictions
        ]
        self.unmatched_predicted_object_count += len(unmatched)
        self.unmatched_predicted_pixels += sum(int(region.area) for region in unmatched)

    def compute(self) -> Dict[str, float | int]:
        eps = np.finfo(np.float64).eps
        precision = self.true_positive_pixels / max(
            1, self.true_positive_pixels + self.false_positive_pixels
        )
        recall = self.true_positive_pixels / max(
            1, self.true_positive_pixels + self.false_negative_pixels
        )
        return {
            "val_loss": float(np.mean(self.losses)),
            "miou": self.intersection / max(1, self.union),
            "niou": float(np.mean(self.image_ious)),
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_f1": 2.0 * precision * recall / max(eps, precision + recall),
            "pd": self.matched_target_count / max(1, self.target_count),
            "tiny_pd": (
                self.matched_tiny_target_count / self.tiny_target_count
                if self.tiny_target_count
                else float("nan")
            ),
            "fa": self.unmatched_predicted_pixels / max(1, self.valid_pixels),
            "false_objects_per_image": self.unmatched_predicted_object_count
            / max(1, len(self.image_ious)),
            "target_count": self.target_count,
            "matched_target_count": self.matched_target_count,
            "tiny_target_count": self.tiny_target_count,
            "matched_tiny_target_count": self.matched_tiny_target_count,
            "predicted_object_count": self.predicted_object_count,
            "unmatched_predicted_object_count": self.unmatched_predicted_object_count,
            "valid_pixel_count": self.valid_pixels,
        }


@torch.inference_mode()
def validate(
    model: SCTransNet,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
    match_radius: float,
    tiny_area: int,
    amp: bool,
) -> Dict[str, float | int]:
    model.eval()
    accumulator = ValidationMetrics(threshold, match_radius, tiny_area)
    for images, masks, sizes, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height = int(sizes[0, 0].item())
        width = int(sizes[0, 1].item())
        with torch.autocast(device_type=device.type, enabled=amp):
            prediction = final_prediction(model(images))
        cropped_prediction = prediction[:, :, :height, :width]
        cropped_mask = masks[:, :, :height, :width]
        # SCTransNet emits post-sigmoid probabilities. PyTorch intentionally
        # rejects BCELoss under autocast, so preserve the reference BCE while
        # evaluating it in FP32.
        with torch.autocast(device_type=device.type, enabled=False):
            loss = criterion(cropped_prediction.float(), cropped_mask.float())
        accumulator.update(
            cropped_prediction[0, 0].float().cpu().numpy(),
            cropped_mask[0, 0].float().cpu().numpy(),
            float(loss.item()),
        )
    return accumulator.compute()


def checkpoint_payload(
    model: SCTransNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    variant: str,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    model_metadata: Dict[str, Any],
    split_hashes: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "variant": variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "validation_metrics": metrics,
        "model_metadata": model_metadata,
        "split_hashes": split_hashes,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp is supported only on CUDA in this runner")

    dataset_dir = args.dataset_dir.resolve()
    dataset_root = dataset_dir / args.dataset
    run_name = f"seed_{args.seed}_{args.run_tag}"
    run_dir = args.output_root.resolve() / args.dataset / args.variant / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "metrics.jsonl"

    started_at = time.time()
    identifiers = read_training_ids(dataset_root, args.dataset)
    mask_stats = [
        inspect_mask(dataset_root / "masks", identifier, args.tiny_area)
        for identifier in identifiers
    ]
    train_ids, val_ids = stratified_split(mask_stats, args.val_fraction, args.split_seed)
    full_train_ids, full_val_ids = train_ids.copy(), val_ids.copy()
    train_ids = subset_for_smoke(train_ids, args.max_train_images, args.seed + 101)
    val_ids = subset_for_smoke(val_ids, args.max_val_images, args.seed + 202)
    split_hashes = {
        "full_internal_train_sha256": identifier_hash(full_train_ids),
        "full_internal_val_sha256": identifier_hash(full_val_ids),
        "used_train_sha256": identifier_hash(train_ids),
        "used_val_sha256": identifier_hash(val_ids),
    }
    stats_by_id = {item.identifier: item for item in mask_stats}
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
            split: dict(sorted(counts.items()))
            for split, counts in {
                "official_train": {
                    key: sum(item.stratum == key for item in mask_stats)
                    for key in sorted({item.stratum for item in mask_stats})
                },
                "used_train": {
                    key: sum(stats_by_id[item].stratum == key for item in train_ids)
                    for key in sorted({item.stratum for item in mask_stats})
                },
                "used_val": {
                    key: sum(stats_by_id[item].stratum == key for item in val_ids)
                    for key in sorted({item.stratum for item in mask_stats})
                },
            }.items()
        },
        "mask_statistics": [asdict(item) for item in mask_stats],
    }
    write_json(run_dir / "split.json", split_manifest)

    model, model_metadata = build_model(args.variant, args.seed)
    model.to(device)
    img_norm_cfg = training_only_normalization(dataset_root, train_ids)
    train_set = TrainingSubset(
        dataset_dir, args.dataset, args.patch_size, train_ids, img_norm_cfg
    )
    val_set = ValidationSubset(dataset_root, val_ids, img_norm_cfg)

    # Reset all training-time RNG streams after variant-specific construction.
    # This keeps shuffle/crop/augmentation streams paired across model variants.
    seed_everything(args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    protocol = {
        "arguments": vars(args),
        "run_directory": run_dir,
        "model": model_metadata,
        "normalization": img_norm_cfg,
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
            "best.pth.tar is Pd-primary; best_miou.pth.tar is a secondary analysis "
            "checkpoint; all selection uses internal validation only"
        ),
        "official_test_accessed": False,
        "loss": "sum of BCE over six deep-supervision outputs",
        "optimizer": "Adam",
        "lr_schedule": "10-epoch linear warmup then cosine decay (or CLI overrides)",
        "metric_notes": {
            "miou": "dataset-level foreground intersection over union",
            "niou": "mean per-image foreground intersection over union",
            "pd": (
                "maximum-cardinality one-to-one centroid matching, then minimum distance, "
                f"at distance < {args.match_radius}"
            ),
            "tiny_pd": f"Pd restricted to GT connected components with area <= {args.tiny_area}",
            "fa": "pixels in unmatched predicted components divided by valid image pixels",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    write_json(run_dir / "protocol.json", protocol)

    best_pd_key = (-float("inf"),) * 5
    best_pd_epoch = 0
    best_pd_metrics: Dict[str, Any] = {}
    best_miou_key = (-float("inf"),) * 5
    best_miou_epoch = 0
    best_miou_metrics: Dict[str, Any] = {}
    skipped_singleton_batches = 0
    print(
        f"START variant={args.variant} dataset={args.dataset} train={len(train_set)} "
        f"val={len(val_set)} device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        learning_rate = learning_rate_for_epoch(
            epoch, args.epochs, args.base_lr, args.min_lr, args.warmup_epochs
        )
        set_learning_rate(optimizer, learning_rate)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, masks in train_loader:
            if images.shape[0] == 1:
                skipped_singleton_batches += 1
                continue
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                outputs = model(images)
            # The model returns sigmoid probabilities rather than logits;
            # compute the unchanged BCE objective outside autocast in FP32.
            with torch.autocast(device_type=device.type, enabled=False):
                float_outputs = (
                    tuple(output.float() for output in outputs)
                    if isinstance(outputs, tuple)
                    else [output.float() for output in outputs]
                    if isinstance(outputs, list)
                    else outputs.float()
                )
                loss = deep_supervision_loss(float_outputs, masks.float(), criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(images.shape[0])
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size
        if not sample_count:
            raise RuntimeError("No training samples were processed in this epoch")

        event: Dict[str, Any] = {
            "epoch": epoch,
            "variant": args.variant,
            "train_loss": loss_sum / sample_count,
            "learning_rate": learning_rate,
            "processed_train_samples": sample_count,
            "epoch_seconds": time.time() - epoch_started,
        }
        should_evaluate = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            metrics = validate(
                model,
                val_loader,
                device,
                criterion,
                args.threshold,
                args.match_radius,
                args.tiny_area,
                args.amp,
            )
            event.update(metrics)
            current_pd_key = pd_selection_key(metrics)
            current_miou_key = miou_selection_key(metrics)
            payload = checkpoint_payload(
                model,
                optimizer,
                scaler,
                epoch,
                args.variant,
                args,
                metrics,
                model_metadata,
                split_hashes,
            )
            payload["checkpoint_role"] = "last_evaluated_epoch"
            torch.save(payload, run_dir / "last.pth.tar")
            if current_pd_key > best_pd_key:
                best_pd_key = current_pd_key
                best_pd_epoch = epoch
                best_pd_metrics = dict(metrics)
                payload["checkpoint_role"] = "best_validation_pd_primary"
                torch.save(payload, run_dir / "best.pth.tar")
                event["new_best_pd"] = True
            else:
                event["new_best_pd"] = False
            if current_miou_key > best_miou_key:
                best_miou_key = current_miou_key
                best_miou_epoch = epoch
                best_miou_metrics = dict(metrics)
                payload["checkpoint_role"] = "best_validation_miou_secondary"
                torch.save(payload, run_dir / "best_miou.pth.tar")
                event["new_best_miou"] = True
            else:
                event["new_best_miou"] = False
            print(
                f"EPOCH {epoch:03d}/{args.epochs} loss={event['train_loss']:.6f} "
                f"mIoU={metrics['miou']:.6f} nIoU={metrics['niou']:.6f} "
                f"Pd={metrics['pd']:.6f} tinyPd={metrics['tiny_pd']:.6f} "
                f"Fa={metrics['fa']:.8f} bestPdEpoch={best_pd_epoch} "
                f"bestMiouEpoch={best_miou_epoch}",
                flush=True,
            )
        else:
            print(
                f"EPOCH {epoch:03d}/{args.epochs} loss={event['train_loss']:.6f} "
                f"lr={learning_rate:.8f}",
                flush=True,
            )
        append_jsonl(events_path, event)

    summary = {
        "status": "complete",
        "variant": args.variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_epoch": best_pd_epoch,
        "best_validation_metrics": best_pd_metrics,
        "best_pd_epoch": best_pd_epoch,
        "best_pd_validation_metrics": best_pd_metrics,
        "best_miou_epoch": best_miou_epoch,
        "best_miou_validation_metrics": best_miou_metrics,
        "primary_selection_metric": "validation Pd, then lower Fa",
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "model": model_metadata,
        "split_hashes": split_hashes,
        "skipped_singleton_batches": skipped_singleton_batches,
        "elapsed_seconds": time.time() - started_at,
        "best_checkpoint": run_dir / "best.pth.tar",
        "best_miou_checkpoint": run_dir / "best_miou.pth.tar",
        "last_checkpoint": run_dir / "last.pth.tar",
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"COMPLETE variant={args.variant} bestPdEpoch={best_pd_epoch} "
        f"bestPd={best_pd_metrics['pd']:.6f} bestPdFa={best_pd_metrics['fa']:.8f} "
        f"bestMiouEpoch={best_miou_epoch} bestMiou={best_miou_metrics['miou']:.6f} "
        f"elapsed={summary['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
