#!/usr/bin/env python3
"""Seed-42 three-dataset trainer for the frozen EC-TSS V3.1 recipe.

The runner owns a new artifact identity while reusing the frozen three-dataset
data adapters and the historical optimization/evaluation loop.  Its optional
``--pause-after-epoch=200`` is an invocation control, not a schedule change:
formal runs always keep ``args.epochs == 1000`` and therefore use the exact
1000-epoch learning-rate trajectory.  The pause is raised only after the
epoch event, metrics line, and rolling model/optimizer/RNG state are durable.
Continuation must use ``--resume=required`` in the same run directory.

The reused engine still passes ``survival_pos_weight`` to its loss callback.
EC-TSS V3.1 does not use that quantity; this runner validates the legacy
keyword and deliberately omits it from the new objective call.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as engine,
)
from experiments import (  # noqa: E402
    train_three_dataset_seed42_global_tss_v2 as positive_runner,
)
from experiments.tpd_training_loss_ec_tss_v3_1 import (  # noqa: E402
    ECTSSV31TrainingLoss,
    compute_ec_tss_v3_1_training_loss as _compute_ec_tss_v3_1_training_loss,
)


SCHEMA = "sctransnet_three_dataset_ec_tss_v3_1_seed42/v1"
OBJECTIVE_ID = "ec_tss_v3_1"
RECIPE_ID = "final_ec_tss_v3_1"
TRAINING_SEED = positive_runner.TRAINING_SEED
DATASETS = positive_runner.DATASETS
METHOD = "final"
TSS_REQUESTED_WEIGHT = 0.005
TSS_RATIO_CAP = 0.10
CONFIDENCE_THRESHOLD = 0.5
TARGET_DILATION_RADIUS = 3
CHECKPOINT_ROLES = positive_runner.CHECKPOINT_ROLES

FORMAL_EPOCHS = positive_runner.FORMAL_EPOCHS
FORMAL_PAUSE_EPOCH = 200
FORMAL_BEGIN_TEST = positive_runner.FORMAL_BEGIN_TEST
FORMAL_EVAL_EVERY = positive_runner.FORMAL_EVAL_EVERY
FORMAL_BATCH_SIZE = positive_runner.FORMAL_BATCH_SIZE
FORMAL_PATCH_SIZE = positive_runner.FORMAL_PATCH_SIZE
FORMAL_WORKERS = positive_runner.FORMAL_WORKERS
FORMAL_BASE_LR = positive_runner.FORMAL_BASE_LR
FORMAL_MIN_LR = positive_runner.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = positive_runner.FORMAL_WARMUP_EPOCHS
FORMAL_THRESHOLD = positive_runner.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = positive_runner.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = positive_runner.FORMAL_TINY_AREA

DEFAULT_DATA_ROOT = positive_runner.DEFAULT_DATA_ROOT
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "results" / "three_dataset_ec_tss_v3_1_seed42"
)
DEFAULT_PROTOCOL_MANIFEST = positive_runner.DEFAULT_PROTOCOL_MANIFEST
DEFAULT_TSS_STATISTICS = positive_runner.DEFAULT_TSS_STATISTICS
PROTOCOL_DOCUMENT = REPO_ROOT / (
    "SCTransNet_EC-TSS_V3性能提升与下一步方案.md"
)
GPU_UUIDS = {
    "0": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "1": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}


class ThreeDatasetECTSSV31ProtocolError(ValueError):
    """One command or persisted artifact violates the V3.1 contract."""


class _PauseAfterEpoch(RuntimeError):
    """Private control-flow signal emitted after a durable epoch boundary."""

    def __init__(self, progress_path: Path) -> None:
        super().__init__(f"paused after durable epoch: {progress_path}")
        self.progress_path = progress_path


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ThreeDatasetECTSSV31ProtocolError(
            f"{name} differs: {actual!r} != {expected!r}"
        )


def requested_tss_weight(args: argparse.Namespace) -> float:
    if args.method != METHOD:
        raise ThreeDatasetECTSSV31ProtocolError(
            "EC-TSS V3.1 accepts only method='final'"
        )
    if args.tss_weight is None:
        return TSS_REQUESTED_WEIGHT
    value = float(args.tss_weight)
    if value != TSS_REQUESTED_WEIGHT:
        raise ThreeDatasetECTSSV31ProtocolError(
            "EC-TSS V3.1 requires --tss-weight 0.005"
        )
    return value


def recipe_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": METHOD,
        "recipe_id": RECIPE_ID,
        "objective_id": OBJECTIVE_ID,
        "requested_tss_weight": requested_tss_weight(args),
        "tss_lambda_token": "0p005",
        "tss_ratio_cap": TSS_RATIO_CAP,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "target_dilation_radius": TARGET_DILATION_RADIUS,
        "positive_normalization": "risk_mass_clamp_min_1",
        "negative_normalization": "risk_mass_clamp_min_1",
        "tss_enabled": True,
        "survival_pos_weight_used": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=(METHOD,), required=True)
    parser.add_argument("--tss-weight", type=float)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_MANIFEST,
    )
    # Retained only because the shared engine exposes this CLI field.  The new
    # objective never reads the statistics artifact or its pos_weight.
    parser.add_argument(
        "--tss-statistics", type=Path, default=DEFAULT_TSS_STATISTICS
    )
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
    parser.add_argument("--physical-gpu-index", choices=tuple(GPU_UUIDS))
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--pause-after-epoch", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    args = parser.parse_args(argv)
    requested_tss_weight(args)
    args.tss_weight = TSS_REQUESTED_WEIGHT
    args.manifest_root = args.protocol_manifest.parent
    # Compatibility input passed by the shared engine.  The adapter below
    # validates and discards it before entering the EC-TSS loss.
    args.survival_pos_weight = None
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_runner.data_protocol.require_dataset(args.dataset)
    positive_runner.data_protocol.require_seed(args.seed)
    requested_tss_weight(args)
    if args.eval_every < 1 or args.epochs < 1 or args.begin_test < 1:
        raise ThreeDatasetECTSSV31ProtocolError(
            "epoch controls must be positive"
        )
    if args.batch_size < 1 or args.workers < 0:
        raise ThreeDatasetECTSSV31ProtocolError(
            "invalid loader configuration"
        )
    _require_equal("patch_size", args.patch_size, FORMAL_PATCH_SIZE)
    _require_equal("metric threshold", args.threshold, FORMAL_THRESHOLD)

    if args.smoke:
        if args.epochs > 3:
            raise ThreeDatasetECTSSV31ProtocolError(
                "smoke runs are limited to three epochs"
            )
        if args.max_train_images is None or args.max_test_images is None:
            raise ThreeDatasetECTSSV31ProtocolError(
                "smoke requires train/test image limits"
            )
        if args.pause_after_epoch is not None and not (
            1 <= args.pause_after_epoch < args.epochs
        ):
            raise ThreeDatasetECTSSV31ProtocolError(
                "smoke pause epoch must be before its planned final epoch"
            )
        if args.device == "cuda:0":
            if args.physical_gpu_index not in GPU_UUIDS:
                raise ThreeDatasetECTSSV31ProtocolError(
                    "CUDA smoke requires physical GPU 0, 1, 2, or 3"
                )
            _require_equal(
                "expected GPU UUID",
                args.expected_gpu_uuid,
                GPU_UUIDS[args.physical_gpu_index],
            )
        return

    if args.max_train_images is not None or args.max_test_images is not None:
        raise ThreeDatasetECTSSV31ProtocolError(
            "formal runs cannot limit train or test images"
        )
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
    for field, expected in formal.items():
        _require_equal(f"formal {field}", getattr(args, field), expected)
    if args.pause_after_epoch not in (None, FORMAL_PAUSE_EPOCH):
        raise ThreeDatasetECTSSV31ProtocolError(
            f"formal pause epoch must be {FORMAL_PAUSE_EPOCH} or omitted"
        )
    if args.physical_gpu_index not in GPU_UUIDS:
        raise ThreeDatasetECTSSV31ProtocolError(
            "formal runs use physical GPU 0, 1, 2, or 3"
        )
    _require_equal(
        "expected GPU UUID",
        args.expected_gpu_uuid,
        GPU_UUIDS[args.physical_gpu_index],
    )


def _run_directory(args: argparse.Namespace) -> Path:
    root = args.results_root.resolve()
    if args.smoke:
        root = root / "smoke"
    return root / "runs" / args.dataset / RECIPE_ID / "seed_42"


def _build_method_model(
    method: str,
    seed: int,
    *,
    dataset_name: str,
) -> tuple[nn.Module, dict[str, Any]]:
    if method != METHOD:
        raise ThreeDatasetECTSSV31ProtocolError(
            "EC-TSS V3.1 model builder accepts only Final"
        )
    positive_runner.data_protocol.require_dataset(dataset_name)
    model, raw_metadata = positive_runner._historical_model_builder(
        METHOD,
        seed=seed,
        dataset_name=dataset_name,
    )
    metadata = positive_runner._annotate_builder_metadata(
        raw_metadata,
        actual_weight=TSS_REQUESTED_WEIGHT,
        method=METHOD,
    )
    objective = metadata.get("formal_training_objective")
    if not isinstance(objective, dict):
        raise ThreeDatasetECTSSV31ProtocolError(
            "Final builder metadata lacks its training objective"
        )
    objective.update(
        {
            "authority": "three_dataset_ec_tss_v3_1_seed42_run_recipe",
            "objective_id": OBJECTIVE_ID,
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_ratio_cap": TSS_RATIO_CAP,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "target_dilation_radius": TARGET_DILATION_RADIUS,
            "survival_pos_weight_used": False,
        }
    )
    metadata["ec_tss_v3_1_recipe"] = copy.deepcopy(
        recipe_identity(argparse.Namespace(method=METHOD, tss_weight=0.005))
    )
    return model, metadata


def _import_runtime_components() -> tuple[Any, Any, Any]:
    return (
        _build_method_model,
        positive_runner._train_dataset_adapter,
        positive_runner._test_dataset_adapter,
    )


def _validate_tss_statistics(
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    requested_tss_weight(args)
    return 1.0, {
        "enabled": True,
        "objective_id": OBJECTIVE_ID,
        "requested_tss_weight": TSS_REQUESTED_WEIGHT,
        "statistics_consumed": False,
        "configured_legacy_path": (
            str(args.tss_statistics.resolve())
            if args.tss_statistics is not None
            else None
        ),
        "legacy_survival_pos_weight_keyword_absorbed": True,
        "survival_pos_weight_used": False,
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
    surrogate = copy.copy(args)
    surrogate.method = METHOD
    surrogate.tss_weight = TSS_REQUESTED_WEIGHT
    payload = positive_runner._protocol_payload(
        surrogate,
        model_metadata=model_metadata,
        tss_metadata=tss_metadata,
        data_manifests=data_manifests,
        train_count=train_count,
        test_count=test_count,
        device=device,
    )
    identity = recipe_identity(args)
    payload["schema"] = SCHEMA
    payload["method"] = METHOD
    payload["recipe"] = identity
    payload["objective_id"] = OBJECTIVE_ID
    payload["model"] = copy.deepcopy(dict(model_metadata))
    payload["tss"] = copy.deepcopy(dict(tss_metadata))
    payload["planned_total_epochs"] = args.epochs
    payload["pause_resume_contract"] = {
        "formal_pause_epoch": FORMAL_PAUSE_EPOCH,
        "pause_is_invocation_control_not_protocol_identity": True,
        "planned_total_epochs_remain_1000_for_formal": True,
        "rolling_state_saved_before_pause": True,
        "continuation_resume_mode": "required",
        "same_run_directory_required": True,
        "same_protocol_sha256_required": True,
        "pilot_creates_additional_run_identity": False,
    }
    payload["rolling_resume_state"].update(
        {
            "retained_on_intentional_pause": True,
            "resume_required_after_intentional_pause": True,
        }
    )
    training = payload["training"]
    training.pop("tss_epoch_diagnostics", None)
    training.update(
        {
            "objective_id": OBJECTIVE_ID,
            "tss_enabled": True,
            "tss_requested_weight": TSS_REQUESTED_WEIGHT,
            "tss_ratio_cap": TSS_RATIO_CAP,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "target_dilation_radius": TARGET_DILATION_RADIUS,
            "positive_normalization": "risk_mass_clamp_min_1",
            "negative_normalization": "risk_mass_clamp_min_1",
            "survival_pos_weight_used": False,
            "legacy_survival_pos_weight_keyword_absorbed": True,
            "tss_effective_weight_formula": (
                "min(0.005,0.10*stopgrad(Lseg)/"
                "max(stopgrad(Lec_tss_v3_1),float32_eps))"
            ),
            "ec_tss_epoch_diagnostics": {
                "participates_in_checkpoint_selection": False,
                "fields": [
                    "effective_weight_mean",
                    "survival_mean",
                    "weighted_survival_mean",
                    "raw_weighted_to_segmentation_ratio_mean",
                    "effective_weighted_to_segmentation_ratio_mean",
                    "positive_survival_mean",
                    "negative_survival_mean",
                    "positive_risk_mass_mean",
                    "negative_risk_mass_mean",
                    "positive_active_cells_mean",
                    "negative_active_cells_mean",
                    "cap_active_batch_fraction",
                ],
            },
        }
    )
    payload["metrics"]["threshold"] = FORMAL_THRESHOLD
    payload["search_budget_disclosure"] = {
        "existing_formal1000_training_runs": 15,
        "new_ec_tss_run_count": len(DATASETS),
        "completed_formal1000_training_runs": 18,
        "positive_and_off_final_family_runs_before_ec_tss": 12,
        "final_family_runs_after_ec_tss": 15,
        "original_training_runs": 3,
        "final_to_original_recipe_search_ratio_after_ec_tss": 5.0,
        "pilot_epochs_are_prefix_of_same_formal_run": True,
        "pilot_creates_additional_run_identity": False,
        "total_recipe_search_budget_equal": False,
        "test_selected": True,
        "selection_is_optimistic": True,
    }
    payload["protocol_document"] = {
        "path": str(PROTOCOL_DOCUMENT),
        "sha256": engine.file_sha256(PROTOCOL_DOCUMENT),
    }
    runtime_sources = payload["runtime_sources"]
    legacy_base_loss = runtime_sources.pop("training_loss", None)
    if not isinstance(legacy_base_loss, Mapping):
        raise ThreeDatasetECTSSV31ProtocolError(
            "reused protocol lacks the legacy segmentation-base loss binding"
        )
    legacy_base_loss = copy.deepcopy(dict(legacy_base_loss))
    legacy_base_loss.update(
        {
            "role": "segmentation-base semantics only; not the EC-TSS objective",
            "consumed_by_ec_tss_objective": False,
        }
    )
    runtime_sources["legacy_segmentation_base_loss"] = legacy_base_loss
    runtime_sources["runner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": engine.file_sha256(Path(__file__).resolve()),
    }
    runtime_sources["reused_positive_runner"] = {
        "path": str(Path(positive_runner.__file__).resolve()),
        "sha256": engine.file_sha256(Path(positive_runner.__file__).resolve()),
    }
    loss_path = REPO_ROOT / "experiments" / "tpd_training_loss_ec_tss_v3_1.py"
    runtime_sources["ec_tss_v3_1_loss"] = {
        "path": str(loss_path),
        "sha256": engine.file_sha256(loss_path),
    }
    ec_protocol_path = REPO_ROOT / "experiments" / "EC_TSS_V3_1_PROTOCOL.md"
    runtime_sources["ec_tss_v3_1_protocol"] = {
        "path": str(ec_protocol_path),
        "sha256": engine.file_sha256(ec_protocol_path),
    }
    return payload


def _scalar(value: torch.Tensor, name: str) -> float:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ThreeDatasetECTSSV31ProtocolError(
            f"{name} must be a scalar Tensor"
        )
    number = float(value.detach().item())
    if not math.isfinite(number):
        raise ThreeDatasetECTSSV31ProtocolError(f"{name} is non-finite")
    return number


class _EpochECTSSV31Audit:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples = 0
        self.batches = 0
        self.weighted_sums = {
            "effective_weight": 0.0,
            "survival": 0.0,
            "weighted_survival": 0.0,
            "raw_weighted_to_segmentation_ratio": 0.0,
            "effective_weighted_to_segmentation_ratio": 0.0,
            "positive_survival": 0.0,
            "negative_survival": 0.0,
            "positive_risk_mass": 0.0,
            "negative_risk_mass": 0.0,
            "positive_active_cells": 0.0,
            "negative_active_cells": 0.0,
        }
        self.cap_active_batches = 0
        self.cap_active_samples = 0

    def add(self, losses: ECTSSV31TrainingLoss, samples: int) -> None:
        if type(samples) is not int or samples < 1:
            raise ThreeDatasetECTSSV31ProtocolError(
                "EC-TSS audit sample count must be positive"
            )
        segmentation = _scalar(losses.segmentation, "segmentation")
        survival = _scalar(losses.survival, "survival")
        weighted_survival = _scalar(
            losses.weighted_survival, "weighted_survival"
        )
        denominator = max(segmentation, torch.finfo(torch.float32).eps)
        values = {
            "effective_weight": _scalar(
                losses.effective_survival_weight, "effective_survival_weight"
            ),
            "survival": survival,
            "weighted_survival": weighted_survival,
            "raw_weighted_to_segmentation_ratio": (
                TSS_REQUESTED_WEIGHT * survival / denominator
            ),
            "effective_weighted_to_segmentation_ratio": (
                weighted_survival / denominator
            ),
            "positive_survival": _scalar(
                losses.positive_survival, "positive_survival"
            ),
            "negative_survival": _scalar(
                losses.negative_survival, "negative_survival"
            ),
            "positive_risk_mass": _scalar(
                losses.positive_risk_mass, "positive_risk_mass"
            ),
            "negative_risk_mass": _scalar(
                losses.negative_risk_mass, "negative_risk_mass"
            ),
            "positive_active_cells": _scalar(
                losses.positive_active_cells, "positive_active_cells"
            ),
            "negative_active_cells": _scalar(
                losses.negative_active_cells, "negative_active_cells"
            ),
        }
        for name, number in values.items():
            if number < 0.0:
                raise ThreeDatasetECTSSV31ProtocolError(
                    f"{name} must be non-negative"
                )
            self.weighted_sums[name] += number * samples
        cap_active = values["effective_weight"] < (
            TSS_REQUESTED_WEIGHT * (1.0 - 1e-6)
        )
        if cap_active:
            self.cap_active_batches += 1
            self.cap_active_samples += samples
        self.samples += samples
        self.batches += 1

    def payload(self) -> dict[str, Any]:
        if self.samples < 1 or self.batches < 1:
            raise ThreeDatasetECTSSV31ProtocolError(
                "EC-TSS epoch audit has no minibatches"
            )
        means = {
            f"train_ec_tss_{name}_mean": total / self.samples
            for name, total in self.weighted_sums.items()
        }
        return {
            "train_ec_tss_enabled": True,
            "train_ec_tss_objective_id": OBJECTIVE_ID,
            "train_ec_tss_requested_weight": TSS_REQUESTED_WEIGHT,
            "train_ec_tss_ratio_cap": TSS_RATIO_CAP,
            "train_ec_tss_confidence_threshold": CONFIDENCE_THRESHOLD,
            "train_ec_tss_target_dilation_radius": TARGET_DILATION_RADIUS,
            "train_ec_tss_survival_pos_weight_used": False,
            "train_ec_tss_observed_batches": self.batches,
            "train_ec_tss_observed_samples": self.samples,
            "train_ec_tss_cap_active_batch_fraction": (
                self.cap_active_batches / self.batches
            ),
            "train_ec_tss_cap_active_sample_fraction": (
                self.cap_active_samples / self.samples
            ),
            **means,
        }


_AUDIT = _EpochECTSSV31Audit()


def _validate_ignored_survival_pos_weight(
    value: float | torch.Tensor,
) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ThreeDatasetECTSSV31ProtocolError(
                "legacy survival_pos_weight must be scalar"
            )
        number = float(value.detach().cpu().item())
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThreeDatasetECTSSV31ProtocolError(
            "legacy survival_pos_weight must be numeric"
        )
    else:
        number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ThreeDatasetECTSSV31ProtocolError(
            "legacy survival_pos_weight must be finite and positive"
        )


def _compute_loss_ec_tss_v3_1(
    output: Any,
    segmentation_target: torch.Tensor,
    segmentation_criterion: nn.Module,
    *,
    survival_weight: float = 0.0,
    survival_pos_weight: float | torch.Tensor = 1.0,
) -> ECTSSV31TrainingLoss:
    if float(survival_weight) != TSS_REQUESTED_WEIGHT:
        raise ThreeDatasetECTSSV31ProtocolError(
            "EC-TSS runtime weight differs from the frozen recipe"
        )
    _validate_ignored_survival_pos_weight(survival_pos_weight)
    losses = _compute_ec_tss_v3_1_training_loss(
        output,
        segmentation_target,
        segmentation_criterion,
        survival_weight=TSS_REQUESTED_WEIGHT,
        survival_ratio_cap=TSS_RATIO_CAP,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        target_dilation_radius=TARGET_DILATION_RADIUS,
    )
    _AUDIT.add(losses, int(segmentation_target.shape[0]))
    return losses


_ENGINE_SELECTED_PAYLOAD = engine._selected_checkpoint_payload
_ENGINE_LATEST_PAYLOAD = engine._latest_checkpoint_payload
_ENGINE_WRITE_JSON = engine.write_json_atomic
_ACTIVE_ARGS: argparse.Namespace | None = None


def _selected_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _ENGINE_SELECTED_PAYLOAD(*args, **kwargs)
    run_args = kwargs["args"]
    identity = recipe_identity(run_args)
    payload.update(
        {
            "schema": SCHEMA,
            "recipe": identity,
            "objective_id": OBJECTIVE_ID,
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_enabled": True,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "target_dilation_radius": TARGET_DILATION_RADIUS,
        }
    )
    return payload


def _latest_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    event = kwargs.get("event")
    if not isinstance(event, dict):
        raise TypeError("latest checkpoint event must be a mutable dictionary")
    run_args = kwargs["args"]
    identity = recipe_identity(run_args)
    event["recipe"] = identity
    event.update(_AUDIT.payload())
    payload = _ENGINE_LATEST_PAYLOAD(*args, **kwargs)
    payload.update(
        {
            "schema": SCHEMA,
            "recipe": identity,
            "objective_id": OBJECTIVE_ID,
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_enabled": True,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "target_dilation_radius": TARGET_DILATION_RADIUS,
            "planned_total_epochs": run_args.epochs,
        }
    )
    _AUDIT.reset()
    return payload


def _load_resume_ec_tss_v3_1(
    *,
    args: argparse.Namespace,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    protocol_sha256: str,
) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    del device
    if not path.exists():
        if args.resume == "required":
            raise FileNotFoundError(path)
        return 1, {}, {}, None
    if args.resume == "never":
        raise FileExistsError(
            f"resume checkpoint exists but --resume=never: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = recipe_identity(args)
    for field, expected in (
        ("schema", SCHEMA),
        ("dataset", args.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("protocol_sha256", protocol_sha256),
        ("recipe", identity),
        ("objective_id", OBJECTIVE_ID),
        ("requested_tss_weight", TSS_REQUESTED_WEIGHT),
        ("tss_enabled", True),
        ("confidence_threshold", CONFIDENCE_THRESHOLD),
        ("target_dilation_radius", TARGET_DILATION_RADIUS),
        ("planned_total_epochs", args.epochs),
    ):
        _require_equal(f"resume {field}", payload.get(field), expected)
    event = payload.get("event")
    if not isinstance(event, Mapping):
        raise ThreeDatasetECTSSV31ProtocolError(
            "resume state lacks its completed epoch event"
        )
    completed_epoch = int(payload["epoch"])
    _require_equal("resume event epoch", event.get("epoch"), completed_epoch)
    _require_equal("resume event recipe", event.get("recipe"), identity)
    _require_equal(
        "resume event objective",
        event.get("train_ec_tss_objective_id"),
        OBJECTIVE_ID,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    engine.restore_rng_state(payload["rng_state"])
    return (
        completed_epoch + 1,
        dict(payload.get("best_miou", {})),
        dict(payload.get("best_pd", {})),
        dict(event),
    )


def _event_diagnostics(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key.startswith("train_ec_tss_")
    }


def _paused_progress_payload(
    path: Path,
    value: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = path.parent
    latest_path = run_dir / "resume" / "latest_training_state.pth.tar"
    if not latest_path.is_file():
        raise ThreeDatasetECTSSV31ProtocolError(
            "pause boundary has no rolling resume state"
        )
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    completed_epoch = int(value["completed_epoch"])
    _require_equal("paused resume epoch", latest.get("epoch"), completed_epoch)
    event = latest.get("event")
    if not isinstance(event, Mapping):
        raise ThreeDatasetECTSSV31ProtocolError(
            "paused resume state lacks epoch diagnostics"
        )
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = protocol.get("protocol_sha256")
    _require_equal(
        "paused protocol binding",
        latest.get("protocol_sha256"),
        protocol_sha256,
    )
    paused = copy.deepcopy(dict(value))
    paused.update(
        {
            "schema": SCHEMA,
            "status": "paused",
            "planned_total_epochs": args.epochs,
            "pause_after_epoch": completed_epoch,
            "resume_required": True,
            "required_resume_mode": "required",
            "protocol_sha256": protocol_sha256,
            "recipe": recipe_identity(args),
            "objective_id": OBJECTIVE_ID,
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "target_dilation_radius": TARGET_DILATION_RADIUS,
            "diagnostics": _event_diagnostics(event),
            "rolling_resume_state": {
                "path": str(latest_path),
                "sha256": engine.file_sha256(latest_path),
                "bytes": latest_path.stat().st_size,
                "epoch": completed_epoch,
                "checkpoint_role": "latest_resume",
            },
        }
    )
    return paused


def _enrich_json_artifact(path: Path, value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if path.name not in {"progress.json", "summary.json"}:
        return value
    enriched = copy.deepcopy(dict(value))
    if enriched.get("method") != METHOD:
        return enriched
    identity = recipe_identity(
        argparse.Namespace(method=METHOD, tss_weight=TSS_REQUESTED_WEIGHT)
    )
    enriched.update(
        {
            "schema": SCHEMA,
            "recipe": identity,
            "objective_id": OBJECTIVE_ID,
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_enabled": True,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "target_dilation_radius": TARGET_DILATION_RADIUS,
            "planned_total_epochs": enriched.get(
                "total_epochs", enriched.get("epochs", FORMAL_EPOCHS)
            ),
            "checkpoint_roles": list(CHECKPOINT_ROLES),
        }
    )
    if path.name == "summary.json":
        enriched["search_budget_disclosure"] = {
            "existing_formal1000_training_runs": 15,
            "new_ec_tss_run_count": 3,
            "completed_formal1000_training_runs": 18,
            "final_family_runs_after_ec_tss": 15,
            "original_training_runs": 3,
            "final_to_original_recipe_search_ratio_after_ec_tss": 5.0,
            "pilot_epochs_are_prefix_of_same_formal_run": True,
            "pilot_creates_additional_run_identity": False,
            "total_recipe_search_budget_equal": False,
        }
    return enriched


def _write_json_atomic(path: Path, value: Any) -> None:
    enriched = _enrich_json_artifact(path, value)
    active = _ACTIVE_ARGS
    should_pause = (
        active is not None
        and active.pause_after_epoch is not None
        and path.name == "progress.json"
        and isinstance(enriched, Mapping)
        and enriched.get("status") in {"running", "finalizing"}
        and enriched.get("completed_epoch") == active.pause_after_epoch
        and active.pause_after_epoch < active.epochs
    )
    if not should_pause:
        _ENGINE_WRITE_JSON(path, enriched)
        return
    paused = _paused_progress_payload(path, enriched, active)
    _ENGINE_WRITE_JSON(path, paused)
    raise _PauseAfterEpoch(path)


def validate_paused_run(
    run_dir: Path,
    dataset: str,
    pause_epoch: int = FORMAL_PAUSE_EPOCH,
) -> dict[str, Any]:
    """Validate and return one intentional-pause artifact without mutation."""

    positive_runner.data_protocol.require_dataset(dataset)
    progress_path = Path(run_dir) / "progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError(progress_path)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    identity = recipe_identity(
        argparse.Namespace(method=METHOD, tss_weight=TSS_REQUESTED_WEIGHT)
    )
    for field, expected in (
        ("schema", SCHEMA),
        ("status", "paused"),
        ("dataset", dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("completed_epoch", pause_epoch),
        ("pause_after_epoch", pause_epoch),
        ("recipe", identity),
        ("objective_id", OBJECTIVE_ID),
        ("resume_required", True),
        ("required_resume_mode", "required"),
    ):
        _require_equal(f"paused progress {field}", progress.get(field), expected)
    planned = int(progress.get("planned_total_epochs", -1))
    if planned <= pause_epoch:
        raise ThreeDatasetECTSSV31ProtocolError(
            "paused run does not retain a later planned final epoch"
        )
    if pause_epoch == FORMAL_PAUSE_EPOCH:
        _require_equal(
            "formal paused planned_total_epochs", planned, FORMAL_EPOCHS
        )
        _require_equal(
            "formal paused total_epochs",
            progress.get("total_epochs"),
            FORMAL_EPOCHS,
        )
    diagnostics = progress.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ThreeDatasetECTSSV31ProtocolError(
            "paused progress lacks EC-TSS diagnostics"
        )
    rolling = progress.get("rolling_resume_state")
    if not isinstance(rolling, Mapping):
        raise ThreeDatasetECTSSV31ProtocolError(
            "paused progress lacks rolling-state binding"
        )
    latest_path = Path(str(rolling.get("path", "")))
    if not latest_path.is_file():
        raise FileNotFoundError(latest_path)
    _require_equal(
        "paused rolling sha256",
        engine.file_sha256(latest_path),
        rolling.get("sha256"),
    )
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    for field, expected in (
        ("schema", SCHEMA),
        ("dataset", dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("epoch", pause_epoch),
        ("recipe", identity),
        ("objective_id", OBJECTIVE_ID),
        ("protocol_sha256", progress.get("protocol_sha256")),
    ):
        _require_equal(f"paused rolling {field}", latest.get(field), expected)
    return progress


def _validate_existing_run_state(args: argparse.Namespace) -> None:
    run_dir = _run_directory(args)
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            for field, expected in (
                ("schema", SCHEMA),
                ("dataset", args.dataset),
                ("method", METHOD),
                ("seed", TRAINING_SEED),
                ("recipe", recipe_identity(args)),
                ("objective_id", OBJECTIVE_ID),
            ):
                _require_equal(
                    f"existing summary {field}", payload.get(field), expected
                )
        return
    progress_path = run_dir / "progress.json"
    if not progress_path.is_file():
        return
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") != "paused":
        return
    validate_paused_run(
        run_dir,
        args.dataset,
        pause_epoch=int(progress["completed_epoch"]),
    )
    if args.resume != "required":
        raise ThreeDatasetECTSSV31ProtocolError(
            "intentional pause continuation requires --resume=required"
        )


@contextmanager
def _patched_engine(args: argparse.Namespace) -> Iterator[None]:
    global _ACTIVE_ARGS
    recipe_identity(args)
    _AUDIT.reset()
    if _ACTIVE_ARGS is not None:
        raise RuntimeError("EC-TSS runner patch is already active")
    replacements = {
        "SCHEMA": SCHEMA,
        "DATASETS": DATASETS,
        "LEGACY_NORMALIZATION": positive_runner.data_protocol.LEGACY_NORMALIZATION,
        "PROTOCOL_DOCUMENT": PROTOCOL_DOCUMENT,
        "FORMAL_TSS_WEIGHT": TSS_REQUESTED_WEIGHT,
        "validate_args": validate_args,
        "_run_directory": _run_directory,
        "_load_data_manifest_lock": positive_runner._load_data_manifest_lock,
        "_import_runtime_components": _import_runtime_components,
        "_load_tss_pos_weight": _validate_tss_statistics,
        "_protocol_payload": _protocol_payload,
        "compute_tpd_training_loss": _compute_loss_ec_tss_v3_1,
        "_selected_checkpoint_payload": _selected_checkpoint_payload,
        "_latest_checkpoint_payload": _latest_checkpoint_payload,
        "_load_resume": _load_resume_ec_tss_v3_1,
        "write_json_atomic": _write_json_atomic,
    }
    previous = {name: getattr(engine, name) for name in replacements}
    for name, value in replacements.items():
        setattr(engine, name, value)
    _ACTIVE_ARGS = args
    try:
        yield
    finally:
        _ACTIVE_ARGS = None
        for name, value in previous.items():
            setattr(engine, name, value)
        _AUDIT.reset()


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    _validate_existing_run_state(args)
    try:
        with _patched_engine(args):
            output_path = engine.run(args)
    except _PauseAfterEpoch as paused:
        progress = validate_paused_run(
            _run_directory(args),
            args.dataset,
            pause_epoch=int(args.pause_after_epoch),
        )
        _require_equal("paused path", paused.progress_path, _run_directory(args) / "progress.json")
        _require_equal("paused total epochs", progress["planned_total_epochs"], args.epochs)
        print(
            f"PAUSED dataset={args.dataset} method={METHOD} "
            f"epoch={args.pause_after_epoch}/{args.epochs} "
            f"resume=required progress={paused.progress_path}",
            flush=True,
        )
        return paused.progress_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("schema", SCHEMA),
        ("recipe", recipe_identity(args)),
        ("objective_id", OBJECTIVE_ID),
        ("requested_tss_weight", TSS_REQUESTED_WEIGHT),
        ("tss_enabled", True),
        ("planned_total_epochs", args.epochs),
    ):
        _require_equal(f"summary {field}", payload.get(field), expected)
    return output_path


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
