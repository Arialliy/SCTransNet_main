#!/usr/bin/env python3
"""Leakage-closed NUAA PBDR-V3 Stage-1 trainer.

Only the official NUAA training index is opened here.  Epoch/checkpoint and
threshold selection use a deterministic internal validation split; official
test construction belongs exclusively to the separate evaluator.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import measure
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import pbdr_v3_non_regression_gate as gate  # noqa: E402
from experiments import three_dataset_pbdr_v3_models_seed42_v1 as models  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments.pbdr_v3_loss import compute_pbdr_v3_loss  # noqa: E402
from experiments.train_tpd_pilot import (  # noqa: E402
    MaskStats,
    ValidationMetrics,
    stratified_split,
)

SCHEMA = "sctransnet_nuaa_pbdr_v3_stage1_v1/v1"
RECIPES = ("core", "constrained")
TRAINING_SEED = 42
SPLIT_SEED = 20260722
VAL_FRACTION = 0.20
FORMAL_EPOCHS = 150
FORMAL_EVAL_EVERY = 5
FORMAL_BATCH_SIZE = 16
FORMAL_PATCH_SIZE = 256
FORMAL_WORKERS = 0
FORMAL_LR = 1.0e-4
FORMAL_WEIGHT_DECAY = 1.0e-4
FORMAL_THRESHOLD = 0.5
FORMAL_MATCH_RADIUS = 3.0
FORMAL_TINY_AREA = 9
THRESHOLDS = tuple(round(0.20 + 0.01 * index, 2) for index in range(61))
GPU0_UUID = "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_PROTOCOL_MANIFEST = data_protocol.DEFAULT_MANIFEST_PATH
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/nuaa_pbdr_v3_stage1_v1"
PROTOCOL_DOCUMENT = REPO_ROOT / "experiments/PBDR_V3_PROTOCOL.md"


class Stage1ProtocolError(ValueError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise Stage1ProtocolError(message)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_json_once_or_equal(path: Path, value: Any) -> None:
    """Preserve immutable protocol artifacts across exact-resume attempts."""

    destination = Path(path)
    ready = _json_ready(value)
    if destination.exists() or destination.is_symlink():
        _require(
            destination.is_file() and not destination.is_symlink(),
            f"write-once JSON is not a regular file: {destination}",
        )
        try:
            observed = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Stage1ProtocolError(
                f"cannot validate write-once JSON {destination}: {error}"
            ) from error
        _require(observed == ready, f"write-once JSON conflicts: {destination}")
        return
    write_json_atomic(destination, ready)


def torch_save_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _pad(array: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.pad(array, ((0, height - array.shape[0]), (0, width - array.shape[1])))


def _load_pair(sample: data_protocol.ResolvedSample) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(sample.image_path) as handle:
        image = np.asarray(handle.convert("I"), dtype=np.float32)
    with Image.open(sample.mask_path) as handle:
        mask = np.asarray(handle, dtype=np.float32)
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    _require(image.ndim == mask.ndim == 2 and image.shape == mask.shape, "bad pair")
    _require(np.isfinite(image).all() and np.isfinite(mask).all(), "non-finite pair")
    return image, mask


def _mask_stats(root: Path, identifiers: Sequence[str]) -> list[MaskStats]:
    known = frozenset(identifiers)
    records: list[MaskStats] = []
    for identifier in identifiers:
        sample = data_protocol.resolve_sample(
            root, models.DATASET, identifier, split="train", known_ids=known
        )
        _, mask = _load_pair(sample)
        binary = mask > 127.5
        areas = [int(region.area) for region in measure.regionprops(measure.label(binary, connectivity=2))]
        tiny = sum(area <= FORMAL_TINY_AREA for area in areas)
        if not areas:
            stratum = "empty"
        elif tiny and len(areas) == 1:
            stratum = "tiny_single"
        elif tiny:
            stratum = "tiny_multi"
        elif min(areas) <= 25:
            stratum = "small_non_tiny"
        else:
            stratum = "larger"
        records.append(MaskStats(identifier, *binary.shape, len(areas), int(binary.sum()), tiny, min(areas) if areas else 0, stratum))
    return records


def build_internal_split_manifest(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_manifest: Path = DEFAULT_PROTOCOL_MANIFEST,
    split_seed: int = SPLIT_SEED,
    val_fraction: float = VAL_FRACTION,
) -> dict[str, Any]:
    """Read only ``train_NUAA-SIRST.txt`` and freeze its stratified split."""

    root = Path(data_root).resolve(strict=True)
    manifest_path = Path(protocol_manifest).resolve(strict=True)
    # ``load_frozen_index`` validates the whole three-dataset manifest and
    # therefore opens official test indexes.  Stage-1 binds that manifest by
    # SHA-256 but opens only this single, frozen official-train index.
    identifiers = data_protocol.load_index(root, models.DATASET, "train")
    stats = _mask_stats(root, identifiers)
    train_ids, val_ids = stratified_split(stats, val_fraction, split_seed)
    payload = {
        "schema": "sctransnet_nuaa_pbdr_v3_internal_split/v1",
        "dataset": models.DATASET,
        "source_split": "official_train_only",
        "official_test_index_opened": False,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "official_train_ids": identifiers,
        "development_train_ids": train_ids,
        "internal_validation_ids": val_ids,
        "mask_stats": [asdict(item) for item in stats],
        "official_train_index_sha256": data_protocol.EXPECTED_SPLITS[models.DATASET]["train"]["file_sha256"],
        "data_protocol_manifest": {
            "path": str(manifest_path),
            "sha256": models.file_sha256(manifest_path),
        },
    }
    payload["split_sha256"] = models.canonical_sha256(payload)
    return payload


class NUAAInternalTrainDataset(Dataset):
    """Frozen V2 crop/augmentation applied only to explicit dev-train IDs."""

    def __init__(
        self,
        identifiers: Sequence[str],
        *,
        data_root: Path = DEFAULT_DATA_ROOT,
        seed: int = TRAINING_SEED,
        known_train_ids: Sequence[str] | None = None,
    ) -> None:
        self.sample_ids = list(identifiers)
        self.root = Path(data_root).resolve(strict=True)
        verified = (
            list(known_train_ids)
            if known_train_ids is not None
            else data_protocol.load_index(self.root, models.DATASET, "train")
        )
        self._known = frozenset(verified)
        _require(
            len(self._known) == len(verified)
            and set(self.sample_ids).issubset(self._known),
            "internal train IDs are not a subset of the official-train index",
        )
        self.seed, self.epoch, self.normalization = seed, 0, data_protocol.get_legacy_normalization(models.DATASET)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        identifier = self.sample_ids[index]
        sample = data_protocol.resolve_sample(self.root, models.DATASET, identifier, split="train", known_ids=self._known)
        image, mask = _load_pair(sample)
        image = (image - np.float32(self.normalization["mean"])) / np.float32(self.normalization["std"])
        mask = mask / np.float32(255.0)
        padded_h, padded_w = max(image.shape[0], FORMAL_PATCH_SIZE), max(image.shape[1], FORMAL_PATCH_SIZE)
        positive = mask > 0
        plan = data_protocol.derive_stateless_transform_plan(
            protocol_seed=self.seed, dataset_name=models.DATASET, epoch=self.epoch,
            namespaced_id=f"{models.DATASET}::{identifier}", image_height=image.shape[0], image_width=image.shape[1],
            has_positive_in_crop=lambda top, left, size: bool(positive[top:top+size, left:left+size].any()),
        )
        image, mask = _pad(image, padded_h, padded_w), _pad(mask, padded_h, padded_w)
        top, left, size = plan.crop_top, plan.crop_left, plan.crop_size
        image, mask = image[top:top+size, left:left+size], mask[top:top+size, left:left+size]
        if plan.flip_axis0: image, mask = image[::-1], mask[::-1]
        if plan.flip_axis1: image, mask = image[:, ::-1], mask[:, ::-1]
        if plan.transpose: image, mask = image.T, mask.T
        return torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(mask[None], dtype=np.float32))


class NUAAInternalValidationDataset(Dataset):
    def __init__(
        self,
        identifiers: Sequence[str],
        *,
        data_root: Path = DEFAULT_DATA_ROOT,
        known_train_ids: Sequence[str] | None = None,
    ) -> None:
        self.sample_ids = list(identifiers)
        self.root = Path(data_root).resolve(strict=True)
        verified = (
            list(known_train_ids)
            if known_train_ids is not None
            else data_protocol.load_index(self.root, models.DATASET, "train")
        )
        self._known = frozenset(verified)
        _require(
            len(self._known) == len(verified)
            and set(self.sample_ids).issubset(self._known),
            "internal validation IDs are not a subset of the official-train index",
        )
        self.normalization = data_protocol.get_legacy_normalization(models.DATASET)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], str]:
        identifier = self.sample_ids[index]
        sample = data_protocol.resolve_sample(self.root, models.DATASET, identifier, split="train", known_ids=self._known)
        image, mask = _load_pair(sample); height, width = image.shape
        image = (image - np.float32(self.normalization["mean"])) / np.float32(self.normalization["std"])
        mask = mask / np.float32(255.0); padded_h, padded_w = ((height+31)//32)*32, ((width+31)//32)*32
        return torch.from_numpy(_pad(image, padded_h, padded_w)[None].astype(np.float32)), torch.from_numpy(_pad(mask, padded_h, padded_w)[None].astype(np.float32)), (height, width), identifier


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and value.get("torch_cuda"): torch.cuda.set_rng_state_all(value["torch_cuda"])


def configure_determinism(seed: int = TRAINING_SEED) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _metrics(probabilities: Sequence[np.ndarray], targets: Sequence[np.ndarray], threshold: float) -> dict[str, Any]:
    accumulator = ValidationMetrics(threshold, FORMAL_MATCH_RADIUS, FORMAL_TINY_AREA)
    for probability, target in zip(probabilities, targets):
        loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(probability), torch.from_numpy(target)))
        accumulator.update(probability, target, loss)
    return dict(accumulator.compute())


@torch.inference_mode()
def evaluate_internal(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval(); routed_maps, base_maps, targets = [], [], []
    for images, masks, sizes, _ in loader:
        images = images.to(device); height, width = int(sizes[0][0]), int(sizes[1][0])
        _, aux = model.forward_for_pbdr_v3_training(images)
        routed_maps.append(torch.sigmoid(aux.routed_logits)[0, 0, :height, :width].cpu().numpy())
        base_maps.append(torch.sigmoid(aux.base_logits)[0, 0, :height, :width].cpu().numpy())
        targets.append(masks[0, 0, :height, :width].numpy())
    fixed = {"current": _metrics(base_maps, targets, FORMAL_THRESHOLD), "candidate": _metrics(routed_maps, targets, FORMAL_THRESHOLD)}
    sweep = {f"{threshold:.2f}": _metrics(routed_maps, targets, threshold) for threshold in THRESHOLDS}
    return {"fixed_0_5": fixed, "candidate_threshold_sweep": sweep}


def _selection_key(role: str, metrics: Mapping[str, Any], epoch: int) -> tuple[float, ...]:
    if role == "best_miou": return (float(metrics["miou"]), float(metrics["pd"]), -float(metrics["fa"]), float(metrics["niou"]), -epoch)
    return (float(metrics["pd"]), -float(metrics["fa"]), float(metrics["miou"]), float(metrics["niou"]), -epoch)


def _checkpoint_selection_key(
    role: str,
    validation: Mapping[str, Any],
    epoch: int,
) -> tuple[float, ...]:
    """Prefer internally certified epochs, then apply the role objective."""

    fixed = validation["fixed_0_5"]
    decision = gate.certify(
        gate.CertificationMetrics.from_mapping(fixed["current"]),
        gate.CertificationMetrics.from_mapping(fixed["candidate"]),
    )
    return (float(decision.passed), *_selection_key(role, fixed["candidate"], epoch))


def _loss_kwargs(recipe: str, epoch: int) -> dict[str, float]:
    if recipe == "core": return {"background_increase_weight": 0.0, "foreground_decrease_weight": 0.0, "trust_region_weight": 0.0, "residual_sparsity_weight": 0.0, "hard_negative_weight": 0.0, "deep_supervision_weight": 0.0}
    return {"background_increase_weight": 8.0, "foreground_decrease_weight": 4.0, "trust_region_weight": 0.25, "residual_sparsity_weight": 0.05, "hard_negative_weight": 2.0 * min(epoch, 20) / 20.0, "deep_supervision_weight": 0.0}


def _decision_payload(decision: gate.CertificationDecision) -> dict[str, Any]:
    return {
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "current": asdict(decision.current),
        "candidate": asdict(decision.candidate),
        "scope": "frozen_internal_validation_split",
    }


def select_validation_threshold(
    role: str,
    validation: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Select a threshold only from the frozen internal validation sweep."""

    current = gate.CertificationMetrics.from_mapping(
        validation["fixed_0_5"]["current"]
    )
    points: list[tuple[float, Mapping[str, Any], gate.CertificationDecision]] = []
    for token, metrics in validation["candidate_threshold_sweep"].items():
        threshold = float(token)
        _require(threshold in THRESHOLDS, "validation threshold is off-grid")
        decision = gate.certify(
            current,
            gate.CertificationMetrics.from_mapping(metrics),
        )
        points.append((threshold, metrics, decision))
    passing = [point for point in points if point[2].passed]
    pool = passing or points
    selected = max(
        pool,
        key=lambda point: (
            _selection_key(role, point[1], 0),
            -abs(point[0] - FORMAL_THRESHOLD),
        ),
    )
    return selected[0], {
        "selection_source": "internal_validation_only",
        "passed_gate_pool_used": bool(passing),
        "threshold": selected[0],
        "metrics": dict(selected[1]),
        "certification": _decision_payload(selected[2]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-role", choices=models.PARENT_ROLES, required=True); parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--parent-checkpoint", type=Path); parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT); parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST); parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS); parser.add_argument("--eval-every", type=int, default=FORMAL_EVAL_EVERY); parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE); parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0"); parser.add_argument("--expected-gpu-uuid", default=GPU0_UUID); parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--max-train-images", type=int); parser.add_argument("--max-val-images", type=int)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    _require(args.epochs > 0 and args.eval_every > 0 and args.batch_size > 0 and args.workers >= 0, "invalid controls")
    if args.smoke:
        _require(args.epochs <= 2 and args.max_train_images and args.max_val_images, "smoke needs limits and <=2 epochs")
    else:
        _require((args.epochs, args.eval_every, args.batch_size, args.workers, args.device, args.expected_gpu_uuid) == (150, 5, 16, 0, "cuda:0", GPU0_UUID), "formal Stage-1 controls differ")
        _require(args.max_train_images is None and args.max_val_images is None, "formal run cannot limit data")


def _device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one GPU must be visible")
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU0_UUID, "GPU0 UUID binding differs")
        actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", "")); actual = actual if actual.startswith("GPU-") else f"GPU-{actual}"
        _require(actual == GPU0_UUID, "visible GPU UUID differs")
    return device


def run(args: argparse.Namespace) -> Path:
    validate_args(args); configure_determinism(); device = _device(args)
    started_at = time.time()
    run_dir = args.results_root.resolve() / ("smoke" if args.smoke else "formal") / args.parent_role / args.recipe
    run_dir.mkdir(parents=True, exist_ok=True); lock = (run_dir / "run.lock").open("a+"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    split = build_internal_split_manifest(
        data_root=args.data_root,
        protocol_manifest=args.protocol_manifest,
    )
    if args.max_train_images:
        split["development_train_ids"] = split["development_train_ids"][:args.max_train_images]
    if args.max_val_images:
        split["internal_validation_ids"] = split["internal_validation_ids"][:args.max_val_images]
    if args.max_train_images or args.max_val_images:
        split.pop("split_sha256")
        split["split_sha256"] = models.canonical_sha256(split)
    write_json_once_or_equal(run_dir / "split_manifest.json", split)
    model, metadata = models.build_stage1_training_model(args.parent_role, parent_checkpoint=args.parent_checkpoint); model.to(device); freeze = models.configure_stage1(model)
    train_set = NUAAInternalTrainDataset(
        split["development_train_ids"],
        data_root=args.data_root,
        known_train_ids=split["official_train_ids"],
    )
    val_set = NUAAInternalValidationDataset(
        split["internal_validation_ids"],
        data_root=args.data_root,
        known_train_ids=split["official_train_ids"],
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.pbdr_v3.parameters(), lr=FORMAL_LR, weight_decay=FORMAL_WEIGHT_DECAY)
    locks = {"parent_checkpoint": metadata["parent_checkpoint"]["sha256"], "split_manifest": split["split_sha256"], "protocol_document": models.file_sha256(PROTOCOL_DOCUMENT), "runtime_sources": models.runtime_source_records()}
    protocol = {
        "schema": SCHEMA,
        "mode": "smoke" if args.smoke else "formal",
        "dataset": models.DATASET,
        "training_seed": TRAINING_SEED,
        "parent_role": args.parent_role,
        "recipe": args.recipe,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "precision": "fp32",
        "fixed_threshold": FORMAL_THRESHOLD,
        "threshold_grid": list(THRESHOLDS),
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "data_root": str(Path(args.data_root).resolve(strict=True)),
        "data_protocol_manifest": dict(split["data_protocol_manifest"]),
        "smoke_limits": {
            "max_train_images": args.max_train_images,
            "max_val_images": args.max_val_images,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": FORMAL_LR,
            "weight_decay": FORMAL_WEIGHT_DECAY,
        },
        "official_test_accessed": False,
        "model": metadata,
        "source_locks": locks,
        "freeze_before": freeze,
    }
    protocol["protocol_sha256"] = models.canonical_sha256(protocol)
    write_json_once_or_equal(run_dir / "protocol.json", protocol)
    latest, selected_path = run_dir / "rolling_state.pth.tar", run_dir / "selected_candidate.pth.tar"; start, best_key, selected = 1, None, None
    if latest.exists():
        _require(args.resume != "never", "rolling state exists"); payload = torch.load(latest, map_location="cpu", weights_only=False)
        _require(payload["protocol_sha256"] == protocol["protocol_sha256"] and payload["source_locks"] == locks, "resume locks differ")
        model.load_state_dict(payload["state_dict"], strict=True); optimizer.load_state_dict(payload["optimizer"]); _restore_rng(payload["rng_state"]); start, best_key, selected = int(payload["epoch"])+1, payload["best_key"], payload["selected"]
    elif args.resume == "required": raise FileNotFoundError(latest)
    initial_base, initial_bn = freeze["base_state_sha256"], freeze["batchnorm_buffer_sha256"]
    for epoch in range(start, args.epochs + 1):
        models.configure_stage1(model); train_set.set_epoch(epoch)
        generator = torch.Generator().manual_seed(data_protocol.stable_sha256_uint64(TRAINING_SEED, "pbdr_v3", epoch))
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.workers)
        totals = []
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device); optimizer.zero_grad(set_to_none=True)
            _, aux = model.forward_for_pbdr_v3_training(images)
            loss = compute_pbdr_v3_loss(routed_logits=aux.routed_logits, base_logits=aux.base_logits, delta_logits=aux.routing.delta_logits, target=masks, soft_iou_weight=1.0, **_loss_kwargs(args.recipe, epoch))
            _require(bool(torch.isfinite(loss.total)), "non-finite loss"); loss.total.backward()
            _require(all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith("pbdr_v3.")), "base gradient detected")
            nn.utils.clip_grad_norm_(model.pbdr_v3.parameters(), 1.0, error_if_nonfinite=True); optimizer.step(); totals.append(float(loss.total.item()))
        _require(models.base_state_sha256(model) == initial_base and models.batchnorm_buffer_sha256(model) == initial_bn, "frozen base/BN changed")
        validation = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate_internal(model, val_loader, device)
            key = _checkpoint_selection_key(args.parent_role, validation, epoch)
            if best_key is None or key > tuple(best_key):
                best_key, selected = key, {"epoch": epoch, "validation": validation}
                checkpoint = {"schema": SCHEMA, "epoch": epoch, "parent_role": args.parent_role, "recipe": args.recipe, "state_dict": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, "validation": validation, "selection_key": key, "protocol_sha256": protocol["protocol_sha256"], "source_locks": locks, "parent_checkpoint": metadata["parent_checkpoint"], "base_state_sha256": initial_base, "batchnorm_buffer_sha256_before": initial_bn, "batchnorm_buffer_sha256_after": models.batchnorm_buffer_sha256(model)}
                torch_save_atomic(selected_path, checkpoint)
        event = {"schema": SCHEMA, "epoch": epoch, "loss": float(np.mean(totals)), "validation": validation, "selected_epoch": selected["epoch"] if selected else None}
        torch_save_atomic(latest, {"schema": SCHEMA, "epoch": epoch, "state_dict": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, "optimizer": optimizer.state_dict(), "rng_state": _rng_state(), "best_key": best_key, "selected": selected, "event": event, "protocol_sha256": protocol["protocol_sha256"], "source_locks": locks})
        write_json_atomic(run_dir / "progress.json", event)
    _require(selected_path.is_file(), "no selected checkpoint")
    selected_payload = torch.load(
        selected_path, map_location="cpu", weights_only=False
    )
    fixed = selected_payload["validation"]["fixed_0_5"]
    decision = gate.certify(
        gate.CertificationMetrics.from_mapping(fixed["current"]),
        gate.CertificationMetrics.from_mapping(fixed["candidate"]),
    )
    decision_ready = _decision_payload(decision)
    selected_threshold, threshold_selection = select_validation_threshold(
        args.parent_role,
        selected_payload["validation"],
    )
    selected_payload["internal_certification_fixed_0_5"] = decision_ready
    selected_payload["selected_threshold"] = selected_threshold
    selected_payload["threshold_selection"] = threshold_selection
    torch_save_atomic(selected_path, selected_payload)
    gate.write_decision(run_dir / "internal_certification.json", decision)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": models.DATASET,
        "seed": TRAINING_SEED,
        "parent_role": args.parent_role,
        "recipe": args.recipe,
        "selected_epoch": selected_payload["epoch"],
        "internal_gate_passed": decision.passed,
        "internal_certification_fixed_0_5": decision_ready,
        "selected_threshold": selected_threshold,
        "threshold_selection": threshold_selection,
        "official_test_accessed": False,
        "selected_checkpoint": {
            "path": str(selected_path.resolve()),
            "sha256": models.file_sha256(selected_path),
        },
        "parent_checkpoint": metadata["parent_checkpoint"],
        "protocol": str((run_dir / "protocol.json").resolve()),
        "protocol_sha256": protocol["protocol_sha256"],
        "rolling_state": str(latest.resolve()),
        "base_state_sha256_before_after": [
            initial_base,
            models.base_state_sha256(model),
        ],
        "batchnorm_buffer_sha256_before_after": [
            initial_bn,
            models.batchnorm_buffer_sha256(model),
        ],
        "elapsed_seconds": time.time() - started_at,
    }
    output = run_dir / "summary.json"; write_json_atomic(output, summary); return output


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()
