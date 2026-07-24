#!/usr/bin/env python3
"""Preliminary, falsification-oriented bottleneck validation for SCTransNet.

This script does not train or evaluate a TPD/FG model. It loads the best legacy
SCTransNet checkpoint (or an explicitly supplied checkpoint), freezes it, fits
identical 1x1 probes at selected model nodes, and measures how well target
presence remains locally rank-separable.

The output is engineering evidence for deciding whether to proceed. It is not a
publication result because the supplied checkpoints were produced by the legacy
training protocol and the diagnostic itself has not been preregistered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import measure


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.SCTransNet import SCTransNet  # noqa: E402
from model.Config import get_SCTrans_config  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg  # noqa: E402


STAGE_ORDER = (
    "input",
    "x1",
    "p1",
    "x2",
    "p2",
    "x3",
    "p3",
    "x4",
    "emb1",
    "emb2",
    "emb3",
    "emb4",
)

KNOWN_INVALID = {"NUAA-SIRST": {"Misc_111"}}


@dataclass(frozen=True)
class Sample:
    image_id: str
    image: torch.Tensor
    mask: torch.Tensor
    valid: torch.Tensor
    original_size: Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate where SCTransNet loses locally decodable target information."
    )
    parser.add_argument("--dataset", required=True, choices=("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"))
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-epochs", type=int, default=2)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--minimum-negatives", type=int, default=64)
    parser.add_argument("--histogram-bins", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-test-images", type=int, default=0)
    parser.add_argument(
        "--input-grid-ap",
        action="store_true",
        help="Also compute expensive original-pixel-grid localization AP.",
    )
    parser.add_argument("--include-known-invalid", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if finite.any() else float("nan")


def default_checkpoint(dataset: str) -> Path:
    checkpoint_dir = REPO_ROOT / "log" / "full_800_3datasets" / dataset
    candidates = sorted(checkpoint_dir.glob("SCTransNet_*_best.pth.tar"))
    if not candidates:
        fallback = checkpoint_dir / "SCTransNet_800.pth.tar"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No best or fixed-epoch checkpoint found in {checkpoint_dir}")

    scored: List[Tuple[float, int, Path]] = []
    for candidate in candidates:
        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        best_miou = checkpoint.get("best_mIOU")
        if isinstance(best_miou, (tuple, list)) and len(best_miou) >= 2:
            metric = float(best_miou[1])
        else:
            metric = float("-inf")
        epoch = int(checkpoint.get("epoch", -1))
        scored.append((metric, epoch, candidate))
    if any(np.isfinite(item[0]) for item in scored):
        metric, epoch, selected = max(scored, key=lambda item: (item[0], item[1]))
        selection_basis = f"highest stored best mIoU={metric:.9f}"
    else:
        # Some legacy NUAA checkpoints contain only epoch/state_dict/total_loss.
        # train.py emits a *_best file only when test mIoU improves, so the last
        # such file is the optimum of that single historical training trajectory.
        metric, epoch, selected = max(scored, key=lambda item: item[1])
        selection_basis = "latest monotonic *_best file; legacy checkpoint has no stored mIoU"
    print(
        f"[{dataset}] auto-selected best checkpoint: {selected.name} "
        f"(epoch={epoch}; basis: {selection_basis})",
        flush=True,
    )
    return selected


def read_split(dataset_dir: Path, dataset: str, split: str) -> List[str]:
    split_path = dataset_dir / dataset / "img_idx" / f"{split}_{dataset}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def resolve_image_path(base: Path, image_id: str) -> Path:
    for suffix in (".png", ".bmp"):
        candidate = base / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No PNG/BMP found for {base / image_id}")


def load_sample(dataset_dir: Path, dataset: str, image_id: str) -> Sample:
    root = dataset_dir / dataset
    image_path = resolve_image_path(root / "images", image_id)
    mask_path = resolve_image_path(root / "masks", image_id)

    image = np.asarray(Image.open(image_path).convert("I"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.float32)

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {dataset}/{image_id}: "
            f"image={image.shape}, mask={mask.shape}"
        )

    original_h, original_w = image.shape
    norm_cfg = get_img_norm_cfg(dataset, str(dataset_dir))
    image = PadImg(Normalized(image, norm_cfg))
    mask = PadImg(mask)
    valid = np.zeros_like(mask, dtype=np.float32)
    valid[:original_h, :original_w] = 1.0

    return Sample(
        image_id=image_id,
        image=torch.from_numpy(np.ascontiguousarray(image[None, None])),
        mask=torch.from_numpy(np.ascontiguousarray(mask[None, None])),
        valid=torch.from_numpy(np.ascontiguousarray(valid[None, None])),
        original_size=(original_h, original_w),
    )


def normalized_state_dict(checkpoint: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    raw = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping")
    result: Dict[str, torch.Tensor] = {}
    for raw_key, value in raw.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        result[key] = value
    return result


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[SCTransNet, Mapping[str, object]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SCTransNet(get_SCTrans_config(), mode="test", deepsuper=True)
    model.load_state_dict(normalized_state_dict(checkpoint), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


class FeatureCapture:
    def __init__(self, model: SCTransNet) -> None:
        self.model = model

    def extract(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run only the modules needed by this diagnostic, in model-forward order."""
        with torch.no_grad():
            x1 = self.model.inc(image)
            p1 = self.model.pool(x1)
            x2 = self.model.down_encoder1(p1)
            p2 = self.model.pool(x2)
            x3 = self.model.down_encoder2(p2)
            p3 = self.model.pool(x3)
            x4 = self.model.down_encoder3(p3)
            emb1 = self.model.mtc.embeddings_1(x1)
            emb2 = self.model.mtc.embeddings_2(x2)
            emb3 = self.model.mtc.embeddings_3(x3)
            emb4 = self.model.mtc.embeddings_4(x4)
        return {
            "input": image.detach(),
            "x1": x1,
            "p1": p1,
            "x2": x2,
            "p2": p2,
            "x3": x3,
            "p3": p3,
            "x4": x4,
            "emb1": emb1,
            "emb2": emb2,
            "emb3": emb3,
            "emb4": emb4,
        }

    def close(self) -> None:
        return None


def resize_presence(mask: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
    return F.adaptive_max_pool2d(mask, tuple(spatial_size))


def resize_valid(valid: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(valid, tuple(spatial_size)) >= 0.5


def standardize_feature_map(feature: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Standardize each channel over valid cells of one image."""
    valid_2d = valid[0, 0]
    feature_2d = feature[0]
    if int(valid_2d.sum()) < 2:
        raise ValueError("Not enough valid feature cells for standardization")
    values = feature_2d[:, valid_2d]
    center = values.mean(dim=1)[:, None, None]
    scale = values.std(dim=1, unbiased=False).clamp_min(1e-6)[:, None, None]
    return ((feature_2d - center) / scale)[None]


def initialize_probes(features: Mapping[str, torch.Tensor], device: torch.device) -> nn.ModuleDict:
    probes = nn.ModuleDict()
    for name in STAGE_ORDER:
        channels = int(features[name].shape[1])
        probe = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(probe.weight)
        nn.init.zeros_(probe.bias)
        probes[name] = probe
    return probes.to(device)


def sampled_probe_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    negative_ratio: int,
    minimum_negatives: int,
) -> torch.Tensor | None:
    flat_logits = logits.reshape(-1)
    flat_target = target.reshape(-1) >= 0.5
    flat_valid = valid.reshape(-1)
    positive = torch.nonzero(flat_valid & flat_target, as_tuple=False).flatten()
    negative = torch.nonzero(flat_valid & ~flat_target, as_tuple=False).flatten()
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    n_negative = min(
        int(negative.numel()),
        max(minimum_negatives, negative_ratio * int(positive.numel())),
    )
    choice = torch.randperm(negative.numel(), device=negative.device)[:n_negative]
    selected = torch.cat((positive, negative[choice]), dim=0)
    labels = flat_target[selected].to(dtype=flat_logits.dtype)
    return F.binary_cross_entropy_with_logits(flat_logits[selected], labels)


def fit_probes(
    model: SCTransNet,
    capture: FeatureCapture,
    probes: nn.ModuleDict,
    train_ids: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> List[float]:
    optimizer = torch.optim.Adam(probes.parameters(), lr=args.probe_lr)
    epoch_losses: List[float] = []
    for epoch in range(args.probe_epochs):
        order = list(train_ids)
        random.Random(args.seed + epoch).shuffle(order)
        running_loss = 0.0
        updates = 0
        probes.train()
        for sample_index, image_id in enumerate(order, start=1):
            sample = load_sample(args.dataset_dir, args.dataset, image_id)
            image = sample.image.to(device, non_blocking=True)
            target = sample.mask.to(device, non_blocking=True)
            valid = sample.valid.to(device, non_blocking=True)
            features = capture.extract(image)
            common_size = features["emb4"].shape[-2:]
            common_target = resize_presence(target, common_size)
            common_valid = resize_valid(valid, common_size)

            losses: List[torch.Tensor] = []
            for name in STAGE_ORDER:
                feature = features[name]
                spatial_size = feature.shape[-2:]
                valid_at_stage = resize_valid(valid, spatial_size)
                standardized = standardize_feature_map(feature, valid_at_stage)
                logits = probes[name](standardized)
                common_logits = F.adaptive_max_pool2d(logits, common_size)
                loss = sampled_probe_loss(
                    common_logits,
                    common_target,
                    common_valid,
                    args.negative_ratio,
                    args.minimum_negatives,
                )
                if loss is not None:
                    losses.append(loss)

            if losses:
                total_loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
                running_loss += float(total_loss.detach().cpu())
                updates += 1

            if sample_index % 100 == 0 or sample_index == len(order):
                print(
                    f"[{args.dataset}] probe epoch {epoch + 1}/{args.probe_epochs}: "
                    f"{sample_index}/{len(order)} images",
                    flush=True,
                )

        mean_loss = running_loss / max(updates, 1)
        epoch_losses.append(mean_loss)
        print(f"[{args.dataset}] probe epoch {epoch + 1} mean loss={mean_loss:.6f}", flush=True)
    return epoch_losses


def standardized_energy(feature: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return a one-channel, per-image channel-standardized energy map."""
    standardized = standardize_feature_map(feature, valid)[0]
    return standardized.square().mean(dim=0).sqrt()


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0


def local_object_statistics(
    score: torch.Tensor,
    object_masks: Sequence[Tuple[int, torch.Tensor]],
    all_target: torch.Tensor,
    valid: torch.Tensor,
) -> List[Dict[str, float]]:
    """Measure object rank on the shared 1/16 grid against a 2--6 cell ring."""
    feature_h, feature_w = score.shape[-2:]
    inner_radius = 1
    outer_radius = 6

    valid_2d = valid[0, 0]
    all_target_2d = all_target[0, 0] > 0
    rows: List[Dict[str, float]] = []
    for object_index, (area, full_resolution_object) in enumerate(object_masks):
        object_at_stage = resize_presence(full_resolution_object, (feature_h, feature_w)) > 0
        target_cells = object_at_stage[0, 0] & valid_2d
        if not bool(target_cells.any()):
            continue

        outer = dilate(object_at_stage, outer_radius)[0, 0]
        inner = dilate(object_at_stage, inner_radius)[0, 0]
        ring = outer & ~inner & valid_2d & ~all_target_2d
        if int(ring.sum()) < 16:
            ring = valid_2d & ~all_target_2d & ~target_cells
        if int(ring.sum()) < 16:
            continue

        target_values = score[target_cells]
        ring_values = score[ring]
        target_score = target_values.max()
        ring_mean = ring_values.mean()
        ring_std = ring_values.std(unbiased=False).clamp_min(1e-6)
        ring_median = ring_values.median()
        ring_mad = (ring_values - ring_median).abs().median().clamp_min(1e-6)
        percentile = (
            (ring_values < target_score).float().sum()
            + 0.5 * (ring_values == target_score).float().sum()
        ) / ring_values.numel()
        cnr = (target_score - ring_mean) / ring_std
        robust_cnr = (target_score - ring_median) / (1.4826 * ring_mad)
        rows.append(
            {
                "object_index": float(object_index),
                "area": float(area),
                "target_score": float(target_score.detach().cpu()),
                "percentile": float(percentile.detach().cpu()),
                "cnr": float(cnr.detach().cpu()),
                "robust_cnr": float(robust_cnr.detach().cpu()),
                "target_cells": float(target_cells.sum().detach().cpu()),
                "ring_cells": float(ring.sum().detach().cpu()),
            }
        )
    return rows


def component_masks(mask: torch.Tensor, original_size: Tuple[int, int]) -> List[Tuple[int, torch.Tensor]]:
    original_h, original_w = original_size
    mask_np = mask[0, 0, :original_h, :original_w].cpu().numpy() > 0
    labels = measure.label(mask_np, connectivity=2)
    output: List[Tuple[int, torch.Tensor]] = []
    padded_h, padded_w = mask.shape[-2:]
    for region in measure.regionprops(labels):
        component = np.zeros((padded_h, padded_w), dtype=np.float32)
        coordinates = region.coords
        component[coordinates[:, 0], coordinates[:, 1]] = 1.0
        tensor = torch.from_numpy(component[None, None]).to(mask.device)
        output.append((int(region.area), tensor))
    return output


class StreamingAveragePrecision:
    def __init__(self, bins: int) -> None:
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)

    def update(self, probabilities: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> None:
        selected_scores = probabilities[valid].detach().float().cpu().numpy()
        selected_labels = labels[valid].detach().bool().cpu().numpy()
        indices = np.minimum((selected_scores * (self.bins - 1)).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(indices[selected_labels], minlength=self.bins)
        self.negative += np.bincount(indices[~selected_labels], minlength=self.bins)

    def get(self) -> float:
        positives = self.positive[::-1].astype(np.float64)
        negatives = self.negative[::-1].astype(np.float64)
        total_positive = positives.sum()
        if total_positive == 0:
            return float("nan")
        true_positive = np.cumsum(positives)
        false_positive = np.cumsum(negatives)
        precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
        recall = true_positive / total_positive
        previous_recall = np.concatenate(([0.0], recall[:-1]))
        return float(np.sum((recall - previous_recall) * precision))


def summarize_object_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "objects": 0,
            "median_percentile": float("nan"),
            "median_cnr": float("nan"),
            "median_robust_cnr": float("nan"),
            "sr90": float("nan"),
            "sr95": float("nan"),
            "sr99": float("nan"),
            "sr99_eligible_objects": 0,
        }
    percentile = np.asarray([row["percentile"] for row in rows], dtype=np.float64)
    cnr = np.asarray([row["cnr"] for row in rows], dtype=np.float64)
    robust_cnr = np.asarray([row["robust_cnr"] for row in rows], dtype=np.float64)
    eligible_99 = np.asarray([row["ring_cells"] >= 100 for row in rows], dtype=bool)
    return {
        "objects": int(len(rows)),
        "median_percentile": float(np.median(percentile)),
        "median_cnr": float(np.median(cnr)),
        "median_robust_cnr": float(np.median(robust_cnr)),
        "sr90": float(np.mean(percentile >= 0.90)),
        "sr95": float(np.mean(percentile >= 0.95)),
        "sr99": float(np.mean(percentile[eligible_99] >= 0.99)) if eligible_99.any() else float("nan"),
        "sr99_eligible_objects": int(eligible_99.sum()),
    }


def stage_scale(stage: str) -> int:
    return {
        "input": 1,
        "x1": 1,
        "p1": 2,
        "x2": 2,
        "p2": 4,
        "x3": 4,
        "p3": 8,
        "x4": 8,
        "emb1": 16,
        "emb2": 16,
        "emb3": 16,
        "emb4": 16,
    }[stage]


def evaluate(
    model: SCTransNet,
    capture: FeatureCapture,
    probes: nn.ModuleDict,
    test_ids: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    probes.eval()
    object_rows: List[Dict[str, object]] = []
    ap_input_grid = (
        {name: StreamingAveragePrecision(args.histogram_bins) for name in STAGE_ORDER}
        if args.input_grid_ap
        else None
    )
    ap_sctb_grid = {name: StreamingAveragePrecision(args.histogram_bins) for name in STAGE_ORDER}

    for sample_index, image_id in enumerate(test_ids, start=1):
        sample = load_sample(args.dataset_dir, args.dataset, image_id)
        image = sample.image.to(device, non_blocking=True)
        mask = sample.mask.to(device, non_blocking=True)
        valid = sample.valid.to(device, non_blocking=True)
        features = capture.extract(image)
        objects = component_masks(mask, sample.original_size)
        common_size = features["emb4"].shape[-2:]
        common_target = resize_presence(mask, common_size)
        common_valid = resize_valid(valid, common_size)

        for name in STAGE_ORDER:
            feature = features[name]
            spatial_size = feature.shape[-2:]
            valid_at_stage = resize_valid(valid, spatial_size)
            with torch.no_grad():
                standardized = standardize_feature_map(feature, valid_at_stage)
                native_probability = torch.sigmoid(probes[name](standardized))
                common_probability = F.adaptive_max_pool2d(native_probability, common_size)[0, 0]
                native_energy = standardized.square().mean(dim=1, keepdim=True).sqrt()
                common_energy = F.adaptive_max_pool2d(native_energy, common_size)[0, 0]

            ap_sctb_grid[name].update(
                common_probability,
                common_target[0, 0] >= 0.5,
                common_valid[0, 0],
            )
            if ap_input_grid is not None:
                with torch.no_grad():
                    input_probability = F.interpolate(
                        native_probability,
                        size=image.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                ap_input_grid[name].update(
                    input_probability,
                    mask[0, 0] >= 0.5,
                    valid[0, 0] >= 0.5,
                )

            for source, score in (("probe", common_probability), ("energy", common_energy)):
                local_rows = local_object_statistics(
                    score,
                    objects,
                    common_target,
                    common_valid,
                )
                for row in local_rows:
                    object_rows.append(
                        {
                            "dataset": args.dataset,
                            "image_id": image_id,
                            "stage": name,
                            "scale": stage_scale(name),
                            "source": source,
                            **row,
                        }
                    )

        if sample_index % 50 == 0 or sample_index == len(test_ids):
            print(f"[{args.dataset}] evaluated {sample_index}/{len(test_ids)} images", flush=True)

    summary: Dict[str, object] = {}
    for name in STAGE_ORDER:
        stage_rows = [row for row in object_rows if row["stage"] == name]
        probe_rows = [row for row in stage_rows if row["source"] == "probe"]
        energy_rows = [row for row in stage_rows if row["source"] == "energy"]
        summary[name] = {
            "scale": stage_scale(name),
            "probe_ap_sctb_grid_approx": ap_sctb_grid[name].get(),
            "probe_ap_input_grid_approx": (
                ap_input_grid[name].get() if ap_input_grid is not None else float("nan")
            ),
            "probe_all": summarize_object_rows(probe_rows),
            "probe_tiny_le9": summarize_object_rows([row for row in probe_rows if row["area"] <= 9]),
            "energy_all": summarize_object_rows(energy_rows),
            "energy_tiny_le9": summarize_object_rows([row for row in energy_rows if row["area"] <= 9]),
        }
    return summary, object_rows


def transition_summary(stage_summary: Mapping[str, object], source: str = "probe_tiny_le9") -> Dict[str, object]:
    transitions = {
        "pool": (("x1", "p1"), ("x2", "p2"), ("x3", "p3")),
        "encoder": (("p1", "x2"), ("p2", "x3"), ("p3", "x4")),
        "embedding": (("x1", "emb1"), ("x2", "emb2"), ("x3", "emb3"), ("x4", "emb4")),
    }
    output: Dict[str, object] = {}
    for operation, pairs in transitions.items():
        rows = []
        for before, after in pairs:
            before_metrics = stage_summary[before][source]
            after_metrics = stage_summary[after][source]
            before_ap = stage_summary[before]["probe_ap_sctb_grid_approx"]
            after_ap = stage_summary[after]["probe_ap_sctb_grid_approx"]
            row = {
                "before": before,
                "after": after,
                "loss_ap_sctb": before_ap - after_ap,
                "delta_median_percentile": (
                    after_metrics["median_percentile"] - before_metrics["median_percentile"]
                ),
                "loss_median_percentile": (
                    before_metrics["median_percentile"] - after_metrics["median_percentile"]
                ),
                "delta_sr90": after_metrics["sr90"] - before_metrics["sr90"],
                "delta_sr95": after_metrics["sr95"] - before_metrics["sr95"],
                "delta_sr99": after_metrics["sr99"] - before_metrics["sr99"],
            }
            rows.append(row)
        output[operation] = {
            "pairs": rows,
            "mean_loss_ap_sctb": safe_nanmean([row["loss_ap_sctb"] for row in rows]),
            "mean_delta_median_percentile": safe_nanmean(
                [row["delta_median_percentile"] for row in rows]
            ),
            "mean_loss_median_percentile": safe_nanmean(
                [row["loss_median_percentile"] for row in rows]
            ),
            "mean_delta_sr90": safe_nanmean([row["delta_sr90"] for row in rows]),
            "mean_delta_sr95": safe_nanmean([row["delta_sr95"] for row in rows]),
            "mean_delta_sr99": safe_nanmean([row["delta_sr99"] for row in rows]),
        }
    return output


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    excluded_ids: Sequence[str],
    probe_losses: Sequence[float],
    stage_summary: Mapping[str, object],
    transitions: Mapping[str, object],
    object_rows: Sequence[Mapping[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    payload = {
        "status": "preliminary_engineering_diagnostic_not_publication_result",
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": args.seed,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "negative_ratio": args.negative_ratio,
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "excluded_ids": list(excluded_ids),
        "methodology": {
            "probe": "frozen SCTransNet + independently trained 1x1 logistic probe per stage",
            "probe_training": "per-image channel standardization; every native logit map is max-pooled to the shared stride-16 SCTB grid and trained against the same max-presence label",
            "local_ring": "shared stride-16 grid; Chebyshev-distance ring from 2 through 6 cells, excluding every target and padding",
            "survival": "target max score percentile within the local background ring",
            "tiny_definition": "connected-component area <= 9 pixels at original resolution",
            "ap_note": "streaming histogram approximation on a shared stride-16 grid; original-grid localization AP is secondary",
        },
        "limitations": [
            "The selected legacy best checkpoint was chosen by a protocol that evaluated the test set during training.",
            "Using the best checkpoint follows the requested preliminary diagnostic, but remains engineering evidence only.",
            "All primary probe comparisons use the same stride-16 labels and cells; local rank survival remains the primary diagnostic.",
            "No TPD or FG model is evaluated by this script.",
        ],
        "probe_epoch_losses": list(probe_losses),
        "stages": stage_summary,
        "transitions_tiny_probe": transitions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n"
    )

    fieldnames = [
        "dataset",
        "image_id",
        "object_index",
        "stage",
        "scale",
        "source",
        "area",
        "target_score",
        "percentile",
        "cnr",
        "robust_cnr",
        "target_cells",
        "ring_cells",
    ]
    with (output_dir / "per_object.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(object_rows)

    lines = [
        f"# {args.dataset} baseline information-bottleneck diagnostic",
        "",
        "> Preliminary engineering evidence only; not a publication result.",
        "",
        f"- Checkpoint: `{args.checkpoint}` (epoch {checkpoint.get('epoch')})",
        f"- Checkpoint SHA256: `{checkpoint_sha256}`",
        f"- Probe train/test images: {len(train_ids)} / {len(test_ids)}",
        f"- Excluded: {', '.join(excluded_ids) if excluded_ids else 'none'}",
        f"- Probe losses: {', '.join(f'{value:.6f}' for value in probe_losses)}",
        "",
        "## Per-stage results",
        "",
        "| Stage | Scale | AP@SCTB* | AP@input* | Tiny N | Median rank | SR@90 | SR@95 | SR@99† | rCNR | Energy SR@95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in STAGE_ORDER:
        stage = stage_summary[name]
        probe = stage["probe_tiny_le9"]
        energy = stage["energy_tiny_le9"]
        lines.append(
            f"| {name} | {stage['scale']}× | {stage['probe_ap_sctb_grid_approx']:.6f} | "
            f"{stage['probe_ap_input_grid_approx']:.6f} | {probe['objects']} | "
            f"{probe['median_percentile']:.6f} | {probe['sr90']:.6f} | {probe['sr95']:.6f} | "
            f"{probe['sr99']:.6f} ({probe['sr99_eligible_objects']}) | "
            f"{probe['median_robust_cnr']:.6f} | {energy['sr95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`AP@SCTB*` is the primary shared-grid probe AP; `AP@input*` is a secondary localization check. Both are streaming histogram approximations. † SR@99 uses only objects with at least 100 ring cells.",
            "",
            "## Tiny-target transition deltas (after − before)",
            "",
            "| Operation | Pair | AP loss@SCTB | Rank loss (before−after) | Δ SR@90 | Δ SR@95 | Δ SR@99 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for operation, block in transitions.items():
        for row in block["pairs"]:
            lines.append(
                f"| {operation} | {row['before']}→{row['after']} | "
                f"{row['loss_ap_sctb']:.6f} | {row['loss_median_percentile']:.6f} | "
                f"{row['delta_sr90']:.6f} | "
                f"{row['delta_sr95']:.6f} | "
                f"{row['delta_sr99']:.6f} |"
            )
        lines.append(
            f"| **{operation} mean** | — | {block['mean_loss_ap_sctb']:.6f} | "
            f"{block['mean_loss_median_percentile']:.6f} | {block['mean_delta_sr90']:.6f} | "
            f"{block['mean_delta_sr95']:.6f} | "
            f"{block['mean_delta_sr99']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This run can locate a candidate bottleneck and decide whether TPD deserves a controlled pilot. It cannot establish TPD effectiveness, novelty, or a publication claim.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    if args.checkpoint is None:
        args.checkpoint = default_checkpoint(args.dataset)
    else:
        args.checkpoint = args.checkpoint.resolve()

    train_ids = read_split(args.dataset_dir, args.dataset, "train")
    test_ids = read_split(args.dataset_dir, args.dataset, "test")
    excluded_ids: List[str] = []
    if not args.include_known_invalid:
        known_invalid = KNOWN_INVALID.get(args.dataset, set())
        excluded_ids = [image_id for image_id in test_ids if image_id in known_invalid]
        test_ids = [image_id for image_id in test_ids if image_id not in known_invalid]
    if args.max_train_images > 0:
        train_ids = train_ids[: args.max_train_images]
    if args.max_test_images > 0:
        test_ids = test_ids[: args.max_test_images]

    print(
        f"[{args.dataset}] device={device}, train={len(train_ids)}, test={len(test_ids)}, "
        f"excluded={excluded_ids}",
        flush=True,
    )
    model, checkpoint = load_model(args.checkpoint, device)
    capture = FeatureCapture(model)
    try:
        first_sample = load_sample(args.dataset_dir, args.dataset, train_ids[0])
        first_features = capture.extract(first_sample.image.to(device))
        shape_text = ", ".join(f"{name}={tuple(first_features[name].shape)}" for name in STAGE_ORDER)
        print(f"[{args.dataset}] captured shapes: {shape_text}", flush=True)
        probes = initialize_probes(first_features, device)
        probe_losses = fit_probes(model, capture, probes, train_ids, args, device)
        stage_summary, object_rows = evaluate(model, capture, probes, test_ids, args, device)
        transitions = transition_summary(stage_summary)
    finally:
        capture.close()

    output_dir = args.output_dir / args.dataset
    write_outputs(
        output_dir,
        args,
        checkpoint,
        train_ids,
        test_ids,
        excluded_ids,
        probe_losses,
        stage_summary,
        transitions,
        object_rows,
    )
    torch.save(probes.state_dict(), output_dir / "probe_state.pt")
    print(f"[{args.dataset}] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
