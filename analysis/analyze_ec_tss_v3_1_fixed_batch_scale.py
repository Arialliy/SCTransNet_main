#!/usr/bin/env python3
"""Train-only fixed-first-batch scale audit for EC-TSS V3.1.

The audit reconstructs the seed-42 Final model and the exact epoch-1 shuffled
training batch used by the formal runner.  It performs no optimizer step,
loads no checkpoint, constructs no test dataset, and does not select or tune a
loss weight.  Its purpose is limited to reporting scalar loss utilization and
the shared-parameter gradient scale induced by the frozen EC-TSS recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import four_dataset_models_seed42_v1 as models  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as engine,
)
from experiments import (  # noqa: E402
    train_three_dataset_ec_tss_v3_1_seed42 as runner,
)
from experiments.tpd_training_loss_ec_tss_v3_1 import (  # noqa: E402
    ECTSSV31TrainingLoss,
    compute_ec_tss_v3_1_training_loss,
)


SCHEMA = "sctransnet_ec_tss_v3_1_fixed_train_batch_scale_audit/v1"
AUDIT_EPOCH = 1
AUDIT_BATCH_INDEX = 0
AUDIT_BATCH_SIZE = runner.FORMAL_BATCH_SIZE
AUDIT_WORKERS = 0


class ECTSSV31ScaleAuditError(ValueError):
    """The requested fixed-batch audit is not the frozen train-only protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ECTSSV31ScaleAuditError(message)


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    _require(
        isinstance(value, torch.Tensor) and value.ndim == 0,
        f"{name} is not scalar",
    )
    number = float(value.detach().item())
    _require(math.isfinite(number), f"{name} is non-finite")
    return number


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _metadata_batch_collate(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stack tensors exactly as training while preserving uint64 seed metadata.

    Formal training requests no metadata and default-collates only image/mask
    tensors.  The audit requests metadata, whose deterministic augmentation
    seeds may exceed signed int64 and therefore cannot pass through PyTorch's
    default integer collation.  Keeping those Python integers in a list avoids
    that representational overflow without changing the image/mask stack.
    """

    _require(bool(samples), "cannot collate an empty audit batch")
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    identifiers: list[str] = []
    augmentation_seeds: list[int] = []
    for index, sample in enumerate(samples):
        _require(isinstance(sample, Mapping), f"sample[{index}] is not a mapping")
        image = sample.get("image")
        mask = sample.get("mask")
        identifier = sample.get("namespaced_sample_id")
        augmentation_seed = sample.get("augmentation_seed")
        _require(isinstance(image, torch.Tensor), f"sample[{index}] lacks image")
        _require(isinstance(mask, torch.Tensor), f"sample[{index}] lacks mask")
        _require(isinstance(identifier, str), f"sample[{index}] lacks sample ID")
        _require(
            type(augmentation_seed) is int,
            f"sample[{index}] lacks integer augmentation seed",
        )
        images.append(image)
        masks.append(mask)
        identifiers.append(identifier)
        augmentation_seeds.append(augmentation_seed)
    return {
        "image": torch.stack(images, dim=0),
        "mask": torch.stack(masks, dim=0),
        "namespaced_sample_id": identifiers,
        "augmentation_seed": augmentation_seeds,
    }


def shared_named_parameters(
    model: nn.Module,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Return trainable Final parameters excluding only the two TSS heads."""

    selected = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("target_survival.")
    )
    _require(bool(selected), "model has no non-survival-head trainable parameters")
    _require(
        all("target_survival" not in name for name, _ in selected),
        "survival-head parameter leaked into the shared set",
    )
    return selected


def _gradient_summary(
    gradients: Sequence[torch.Tensor | None],
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> dict[str, Any]:
    _require(
        len(gradients) == len(named_parameters),
        "gradient/parameter count differs",
    )
    used = 0
    squared_contributions: list[torch.Tensor] = []
    for gradient in gradients:
        if gradient is None:
            continue
        used += 1
        detached = gradient.detach().to(dtype=torch.float64)
        squared_contributions.append(torch.sum(detached * detached))
    if squared_contributions:
        packed = torch.stack(squared_contributions)
        finite = bool(torch.isfinite(packed).all().detach().cpu().item())
        nonzero = int(torch.count_nonzero(packed).detach().cpu().item())
        squared_norm = float(packed.sum().detach().cpu().item()) if finite else None
    else:
        finite = True
        nonzero = 0
        squared_norm = 0.0
    norm = math.sqrt(squared_norm) if squared_norm is not None else None
    return {
        "global_l2_norm": norm,
        "finite": finite,
        "parameter_tensor_count": len(named_parameters),
        "gradient_tensor_count": used,
        "nonzero_gradient_tensor_count": nonzero,
    }


def audit_losses_and_shared_gradients(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Audit one already-resolved training batch without changing parameters."""

    _require(images.ndim == 4 and masks.ndim == 4, "images/masks must be BxCxHxW")
    _require(images.shape == masks.shape, "images/masks shapes differ")
    _require(images.device == masks.device, "images/masks devices differ")
    criterion = criterion or nn.BCELoss(reduction="mean")
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(images)
    losses = compute_ec_tss_v3_1_training_loss(
        output,
        masks,
        criterion,
        survival_weight=runner.TSS_REQUESTED_WEIGHT,
        survival_ratio_cap=runner.TSS_RATIO_CAP,
        confidence_threshold=runner.CONFIDENCE_THRESHOLD,
        target_dilation_radius=runner.TARGET_DILATION_RADIUS,
    )

    named_parameters = shared_named_parameters(model)
    parameters = tuple(parameter for _, parameter in named_parameters)
    segmentation_gradients = torch.autograd.grad(
        losses.segmentation,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    segmentation_gradient = _gradient_summary(
        segmentation_gradients, named_parameters
    )
    del segmentation_gradients
    weighted_ec_gradients = torch.autograd.grad(
        losses.weighted_survival,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    weighted_ec_gradient = _gradient_summary(
        weighted_ec_gradients, named_parameters
    )
    del weighted_ec_gradients

    segmentation = _finite_scalar(losses.segmentation, "segmentation")
    ec_tss = _finite_scalar(losses.survival, "survival")
    weighted = _finite_scalar(losses.weighted_survival, "weighted_survival")
    effective_weight = _finite_scalar(
        losses.effective_survival_weight, "effective_survival_weight"
    )
    denominator = max(segmentation, torch.finfo(torch.float32).eps)
    segmentation_norm = segmentation_gradient["global_l2_norm"]
    weighted_norm = weighted_ec_gradient["global_l2_norm"]
    gradient_ratio = (
        float(weighted_norm) / float(segmentation_norm)
        if segmentation_norm is not None
        and weighted_norm is not None
        and float(segmentation_norm) > 0.0
        else None
    )
    cap_active = effective_weight < (
        runner.TSS_REQUESTED_WEIGHT * (1.0 - 1e-6)
    )
    return {
        "losses": {
            "segmentation": segmentation,
            "ec_tss": ec_tss,
            "weighted_ec_tss": weighted,
            "requested_survival_weight": runner.TSS_REQUESTED_WEIGHT,
            "effective_survival_weight": effective_weight,
            "survival_ratio_cap": runner.TSS_RATIO_CAP,
            "raw_requested_ec_to_segmentation_ratio": (
                runner.TSS_REQUESTED_WEIGHT * ec_tss / denominator
            ),
            "weighted_ec_to_segmentation_ratio": weighted / denominator,
            "cap_active": cap_active,
            "positive_survival": _finite_scalar(
                losses.positive_survival, "positive_survival"
            ),
            "negative_survival": _finite_scalar(
                losses.negative_survival, "negative_survival"
            ),
            "endpoint_positive_terms": [
                _finite_scalar(value, f"endpoint_positive_terms[{index}]")
                for index, value in enumerate(losses.endpoint_positive_terms)
            ],
            "endpoint_negative_terms": [
                _finite_scalar(value, f"endpoint_negative_terms[{index}]")
                for index, value in enumerate(losses.endpoint_negative_terms)
            ],
        },
        "risk": {
            "positive_risk_mass": _finite_scalar(
                losses.positive_risk_mass, "positive_risk_mass"
            ),
            "negative_risk_mass": _finite_scalar(
                losses.negative_risk_mass, "negative_risk_mass"
            ),
            "positive_active_cells": int(
                _finite_scalar(losses.positive_active_cells, "positive_active_cells")
            ),
            "negative_active_cells": int(
                _finite_scalar(losses.negative_active_cells, "negative_active_cells")
            ),
        },
        "shared_parameter_gradients": {
            "selection": "all_trainable_parameters_except_target_survival_prefix",
            "survival_head_parameters_excluded": True,
            "selected_parameter_count": sum(
                parameter.numel() for _, parameter in named_parameters
            ),
            "selected_parameter_tensor_count": len(named_parameters),
            "segmentation": segmentation_gradient,
            "weighted_ec_tss": weighted_ec_gradient,
            "weighted_ec_to_segmentation_global_l2_ratio": gradient_ratio,
        },
    }


def build_fixed_first_train_batch(
    dataset: Dataset[Any],
    dataset_name: str,
    *,
    batch_size: int = AUDIT_BATCH_SIZE,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Resolve the exact epoch-1, batch-0 sample order used by training."""

    _require(dataset_name in runner.DATASETS, "dataset is outside the frozen matrix")
    _require(batch_size == AUDIT_BATCH_SIZE, "audit batch size must remain 16")
    set_epoch = getattr(dataset, "set_epoch", None)
    _require(callable(set_epoch), "training dataset lacks set_epoch")
    set_epoch(AUDIT_EPOCH)
    generator = torch.Generator(device="cpu")
    shuffle_seed = engine.stable_uint63(
        runner.TRAINING_SEED,
        dataset_name,
        "shuffle",
        AUDIT_EPOCH,
    )
    generator.manual_seed(shuffle_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=AUDIT_WORKERS,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
        collate_fn=_metadata_batch_collate,
    )
    batch = next(iter(loader))
    _require(
        isinstance(batch, Mapping),
        "metadata-enabled training batch must be a mapping",
    )
    images = batch.get("image")
    masks = batch.get("mask")
    _require(isinstance(images, torch.Tensor), "batch lacks image Tensor")
    _require(isinstance(masks, torch.Tensor), "batch lacks mask Tensor")
    identifiers_raw = batch.get("namespaced_sample_id")
    _require(
        isinstance(identifiers_raw, (tuple, list)),
        "batch lacks namespaced sample IDs",
    )
    augmentation_raw = batch.get("augmentation_seed")
    if isinstance(augmentation_raw, torch.Tensor):
        augmentation_seeds = [int(value) for value in augmentation_raw.tolist()]
    elif isinstance(augmentation_raw, (tuple, list)):
        augmentation_seeds = [int(value) for value in augmentation_raw]
    else:
        raise ECTSSV31ScaleAuditError("batch lacks augmentation seeds")
    metadata = {
        "epoch": AUDIT_EPOCH,
        "batch_index": AUDIT_BATCH_INDEX,
        "batch_size": int(images.shape[0]),
        "configured_batch_size": batch_size,
        "shuffle_seed": shuffle_seed,
        "namespaced_sample_ids": [str(value) for value in identifiers_raw],
        "augmentation_seeds": augmentation_seeds,
        "images_sha256": _tensor_sha256(images),
        "masks_sha256": _tensor_sha256(masks),
    }
    return (
        images.to(device, non_blocking=True),
        masks.to(device, non_blocking=True),
        metadata,
    )


def _resolve_device(args: argparse.Namespace) -> tuple[torch.device, str | None]:
    if args.device == "cpu":
        return torch.device("cpu"), None
    _require(
        args.physical_gpu_index in runner.GPU_UUIDS,
        "physical GPU index differs",
    )
    expected = runner.GPU_UUIDS[args.physical_gpu_index]
    if args.expected_gpu_uuid is not None:
        _require(args.expected_gpu_uuid == expected, "expected GPU UUID differs")
    proxy = argparse.Namespace(
        device=args.device,
        smoke=True,
        physical_gpu_index=args.physical_gpu_index,
        expected_gpu_uuid=expected,
    )
    return engine.resolve_device(proxy), expected


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Construct the frozen train-only audit and return its JSON payload."""

    _require(args.dataset in runner.DATASETS, "dataset is outside the frozen matrix")
    data_root = args.data_root.resolve(strict=True)
    manifest_path = args.protocol_manifest.resolve(strict=True)
    engine.configure_determinism()
    device, gpu_uuid = _resolve_device(args)

    engine.seed_everything(runner.TRAINING_SEED)
    model, model_metadata = runner._build_method_model(
        runner.METHOD,
        runner.TRAINING_SEED,
        dataset_name=args.dataset,
    )
    initial_state_sha256 = models.state_dict_sha256(model.state_dict())
    model.to(device)
    train_dataset = runner.positive_runner._train_dataset_adapter(
        args.dataset,
        patch_size=runner.FORMAL_PATCH_SIZE,
        seed=runner.TRAINING_SEED,
        dataset_root=data_root,
        imgidx_manifest=manifest_path,
        normalization_manifest=manifest_path,
        correction_manifest=manifest_path,
        return_metadata=True,
    )
    engine.seed_everything(runner.TRAINING_SEED)
    images, masks, batch_metadata = build_fixed_first_train_batch(
        train_dataset,
        args.dataset,
        device=device,
    )
    audit = audit_losses_and_shared_gradients(model, images, masks)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "purpose": "train_only_fixed_first_batch_scale_audit",
        "dataset": args.dataset,
        "training_seed": runner.TRAINING_SEED,
        "objective_id": runner.OBJECTIVE_ID,
        "recipe": runner.recipe_identity(
            argparse.Namespace(
                method=runner.METHOD,
                tss_weight=runner.TSS_REQUESTED_WEIGHT,
            )
        ),
        "device": {
            "torch_device": str(device),
            "physical_gpu_index": (
                args.physical_gpu_index if device.type == "cuda" else None
            ),
            "expected_gpu_uuid": gpu_uuid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "data": {
            "split": "img_idx/train",
            "test_dataset_constructed": False,
            "test_metrics_read": False,
            "dataset_root": str(data_root),
            "protocol_manifest": str(manifest_path),
            "protocol_manifest_sha256": engine.file_sha256(manifest_path),
            "training_dataset_size": len(train_dataset),
            "fixed_batch": batch_metadata,
        },
        "model": {
            "builder": "train_three_dataset_ec_tss_v3_1_seed42._build_method_model",
            "initial_state_dict_sha256": initial_state_sha256,
            "model_metadata_objective_id": model_metadata[
                "formal_training_objective"
            ]["objective_id"],
            "optimizer_step_performed": False,
            "checkpoint_loaded": False,
        },
        **audit,
        "interpretation_scope": {
            "lambda_tuning_performed": False,
            "test_performance_used": False,
            "formal_recipe_modified": False,
            "permitted_use": "pretraining gradient-scale and cap-utilization audit",
            "performance_improvement_claim_supported": False,
        },
        "source_sha256": {
            "audit": engine.file_sha256(Path(__file__)),
            "runner": engine.file_sha256(Path(runner.__file__)),
            "loss": engine.file_sha256(
                REPO_ROOT / "experiments" / "tpd_training_loss_ec_tss_v3_1.py"
            ),
            "data_protocol": engine.file_sha256(
                Path(runner.positive_runner.data_protocol.__file__)
            ),
        },
    }
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=runner.DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=runner.DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=runner.DEFAULT_PROTOCOL_MANIFEST,
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument(
        "--physical-gpu-index", choices=tuple(runner.GPU_UUIDS), default="3"
    )
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_audit(args)
    if args.output is not None:
        destination = args.output.resolve()
        if destination.exists():
            raise FileExistsError(destination)
        engine.write_json_atomic(destination, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
