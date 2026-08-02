#!/usr/bin/env python3
"""Formal seed-42 trainer for the three-dataset global-TSS V2 matrix.

This entry point intentionally reuses only the optimization/checkpoint loop of
the completed runner.  Its data identity, dataset classes, run identity, TSS
statistics, loss recipe, and artifact paths are replaced by the V2 contract.
It accepts exactly three datasets and creates only ``best_miou`` and
``best_pd`` selected checkpoints.
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

from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as engine,
)
from experiments.four_dataset_models_seed42_v1 import (  # noqa: E402
    build_method_model as _historical_model_builder,
)
from experiments.paper_three_dataset_v2 import (  # noqa: E402
    ThreeDatasetV2TestDataset,
    ThreeDatasetV2TrainDataset,
)
from experiments.tpd_training_loss import (  # noqa: E402
    TPDTrainingLoss,
    compute_tpd_training_loss as _compute_tpd_training_loss,
)


SCHEMA = "sctransnet_three_dataset_seed42_global_tss_v2/v1"
TRAINING_SEED = 42
DATASETS = data_protocol.DATASETS
METHODS = ("original", "final")
TSS_LAMBDAS = (0.0025, 0.005, 0.01)
TSS_RATIO_CAP = 0.10
CHECKPOINT_ROLES = ("best_miou", "best_pd")

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

DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / (
    "three_dataset_seed42_global_tss_v2"
)
DEFAULT_PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "results"
    / "three_dataset_v2"
    / "manifests"
    / "three_dataset_v2_protocol.json"
)
DEFAULT_TSS_STATISTICS = (
    REPO_ROOT
    / "results"
    / "three_dataset_v2"
    / "manifests"
    / "three_dataset_v2_tss_statistics.json"
)
PROTOCOL_DOCUMENT = REPO_ROOT / (
    "SCTransNet_V2全数据集混合结果复盘与全局TSS配方定型方案.md"
)

GPU_UUIDS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}


class ThreeDatasetTrainingProtocolError(ValueError):
    """One command or persisted artifact violates the V2 run contract."""


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ThreeDatasetTrainingProtocolError(
            f"{name} differs: {actual!r} != {expected!r}"
        )


def _lambda_token(value: float) -> str:
    mapping = {0.0025: "0p0025", 0.005: "0p005", 0.01: "0p01"}
    if value not in mapping:
        raise ThreeDatasetTrainingProtocolError(
            f"TSS weight must be one of {TSS_LAMBDAS}, got {value!r}"
        )
    return mapping[value]


def requested_tss_weight(args: argparse.Namespace) -> float:
    if args.method == "original":
        if args.tss_weight not in (None, 0, 0.0):
            raise ThreeDatasetTrainingProtocolError(
                "Original must not request a TSS weight"
            )
        return 0.0
    if args.tss_weight is None:
        raise ThreeDatasetTrainingProtocolError(
            f"Final requires --tss-weight in {TSS_LAMBDAS}"
        )
    value = float(args.tss_weight)
    _lambda_token(value)
    return value


def recipe_identity(args: argparse.Namespace) -> dict[str, Any]:
    weight = requested_tss_weight(args)
    if args.method == "original":
        recipe_id = "original_no_tss"
        token = None
    else:
        token = _lambda_token(weight)
        recipe_id = f"final_lambda_{token}"
    return {
        "method": args.method,
        "recipe_id": recipe_id,
        "requested_tss_weight": weight,
        "tss_lambda_token": token,
        "tss_ratio_cap": TSS_RATIO_CAP if args.method == "final" else None,
        "tss_enabled": args.method == "final",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
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
    parser.add_argument("--physical-gpu-index", choices=("2", "3"))
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    args = parser.parse_args(argv)
    # Compatibility fields consumed by the reused training-loop engine.  The
    # patched functions below never resolve its historical manifest inputs.
    args.manifest_root = args.protocol_manifest.parent
    args.survival_pos_weight = None
    return args


def validate_args(args: argparse.Namespace) -> None:
    data_protocol.require_dataset(args.dataset)
    data_protocol.require_seed(args.seed)
    if args.method not in METHODS:
        raise ThreeDatasetTrainingProtocolError(
            f"method must be one of {METHODS}"
        )
    requested_tss_weight(args)
    if args.method == "original" and args.tss_statistics not in (
        None,
        DEFAULT_TSS_STATISTICS,
    ):
        raise ThreeDatasetTrainingProtocolError(
            "Original does not accept a custom TSS statistics artifact"
        )
    if args.eval_every < 1 or args.epochs < 1 or args.begin_test < 1:
        raise ThreeDatasetTrainingProtocolError(
            "epoch controls must be positive"
        )
    if args.batch_size < 1 or args.workers < 0:
        raise ThreeDatasetTrainingProtocolError("invalid loader configuration")
    if args.patch_size != FORMAL_PATCH_SIZE:
        raise ThreeDatasetTrainingProtocolError(
            f"patch_size must be {FORMAL_PATCH_SIZE}"
        )
    _require_equal("metric threshold", args.threshold, FORMAL_THRESHOLD)
    if args.smoke:
        if args.epochs > 2:
            raise ThreeDatasetTrainingProtocolError(
                "smoke runs are limited to two epochs"
            )
        if args.max_train_images is None or args.max_test_images is None:
            raise ThreeDatasetTrainingProtocolError(
                "smoke requires train/test image limits"
            )
        if args.device == "cuda:0":
            if args.physical_gpu_index not in GPU_UUIDS:
                raise ThreeDatasetTrainingProtocolError(
                    "CUDA smoke requires physical GPU 2 or 3"
                )
            _require_equal(
                "expected GPU UUID",
                args.expected_gpu_uuid,
                GPU_UUIDS[args.physical_gpu_index],
            )
        return
    if args.max_train_images is not None or args.max_test_images is not None:
        raise ThreeDatasetTrainingProtocolError(
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
    if args.physical_gpu_index not in GPU_UUIDS:
        raise ThreeDatasetTrainingProtocolError(
            "formal runs use physical GPU 2 or 3"
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
    identity = recipe_identity(args)
    if args.method == "original":
        branch = Path("original")
    else:
        branch = Path("final") / f"lambda_{identity['tss_lambda_token']}"
    return root / "runs" / args.dataset / branch / "seed_42"


def _load_data_manifest_lock(args: argparse.Namespace) -> dict[str, Any]:
    path = args.protocol_manifest.resolve(strict=True)
    payload = data_protocol.load_protocol_manifest(
        path, dataset_root=args.data_root.resolve()
    )
    record = {
        "path": str(path),
        "sha256": engine.file_sha256(path),
        "bytes": path.stat().st_size,
    }
    # The engine asks for three legacy-named file roles.  All three are bound
    # to the same V2 manifest and translated by the adapters below.
    return {
        "root": str(path.parent),
        "manifest": copy.deepcopy(payload),
        "files": {
            "imgidx": dict(record),
            "normalization": dict(record),
            "correction": dict(record),
        },
    }


def _require_same_manifest(**values: Any) -> Path:
    paths = {Path(value).resolve(strict=True) for value in values.values()}
    if len(paths) != 1:
        raise ThreeDatasetTrainingProtocolError(
            "all data roles must resolve to one three-dataset V2 manifest"
        )
    return next(iter(paths))


def _train_dataset_adapter(
    dataset: str,
    *,
    patch_size: int,
    seed: int,
    dataset_root: Path,
    imgidx_manifest: str | Path,
    normalization_manifest: str | Path,
    correction_manifest: str | Path,
    return_metadata: bool,
) -> ThreeDatasetV2TrainDataset:
    manifest = _require_same_manifest(
        imgidx=imgidx_manifest,
        normalization=normalization_manifest,
        correction=correction_manifest,
    )
    return ThreeDatasetV2TrainDataset(
        dataset,
        dataset_root=dataset_root,
        protocol_manifest=manifest,
        patch_size=patch_size,
        seed=seed,
        return_metadata=return_metadata,
    )


def _test_dataset_adapter(
    train_dataset_name: str,
    test_dataset_name: str,
    *,
    dataset_root: Path,
    imgidx_manifest: str | Path,
    normalization_manifest: str | Path,
    correction_manifest: str | Path,
    return_metadata: bool,
) -> ThreeDatasetV2TestDataset:
    manifest = _require_same_manifest(
        imgidx=imgidx_manifest,
        normalization=normalization_manifest,
        correction=correction_manifest,
    )
    return ThreeDatasetV2TestDataset(
        train_dataset_name,
        test_dataset_name,
        dataset_root=dataset_root,
        protocol_manifest=manifest,
        return_metadata=return_metadata,
    )


def _annotate_builder_metadata(
    value: Any,
    *,
    actual_weight: float,
    method: str,
) -> dict[str, Any]:
    """Preserve shared-builder history while making the run recipe canonical.

    The reusable model builder records its historical default coefficient as
    ``tss_training_weight`` even though it does not construct the loss.  Rename
    that provenance field so it cannot disagree with the actual V2 recipe.
    """

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            cleaned: dict[str, Any] = {}
            for key, nested in item.items():
                output_key = (
                    "legacy_builder_default_tss_training_weight"
                    if key == "tss_training_weight"
                    else str(key)
                )
                cleaned[output_key] = visit(nested)
            return cleaned
        if isinstance(item, list):
            return [visit(nested) for nested in item]
        if isinstance(item, tuple):
            return tuple(visit(nested) for nested in item)
        return item

    annotated = visit(copy.deepcopy(value))
    if not isinstance(annotated, dict):
        raise TypeError("model builder metadata must be a mapping")
    annotated["formal_three_dataset_scope"] = list(DATASETS)
    annotated["formal_training_objective"] = {
        "authority": "three_dataset_v2_run_recipe",
        "method": method,
        "requested_tss_weight": actual_weight,
        "tss_ratio_cap": TSS_RATIO_CAP if method == "final" else None,
        "historical_builder_default_is_provenance_only": True,
    }
    return annotated


def _build_method_model(
    method: str,
    seed: int,
    *,
    dataset_name: str,
) -> tuple[nn.Module, dict[str, Any]]:
    data_protocol.require_dataset(dataset_name)
    model, metadata = _historical_model_builder(
        method, seed=seed, dataset_name=dataset_name
    )
    return model, _annotate_builder_metadata(
        metadata,
        actual_weight=_AUDIT.requested_weight,
        method=method,
    )


def _import_runtime_components() -> tuple[Any, Any, Any]:
    return _build_method_model, _train_dataset_adapter, _test_dataset_adapter


def _validate_tss_statistics(
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    if args.method == "original":
        return 1.0, {
            "enabled": False,
            "requested_tss_weight": 0.0,
            "statistics_consumed": False,
        }
    if args.tss_statistics is None:
        raise ThreeDatasetTrainingProtocolError(
            "Final requires the compact three-dataset V2 TSS statistics"
        )
    path = args.tss_statistics.resolve(strict=True)
    raw = path.read_text(encoding="utf-8")
    if "SIRST3" in raw:
        raise ThreeDatasetTrainingProtocolError(
            "TSS statistics contain a dataset outside the formal V2 scope"
        )
    payload = json.loads(raw)
    expected_top = {
        "schema": "sctransnet_three_dataset_v2_tss_statistics/v1",
        "training_seed": TRAINING_SEED,
        "epochs": FORMAL_EPOCHS,
        "datasets": list(DATASETS),
    }
    for field, expected in expected_top.items():
        _require_equal(f"TSS statistics {field}", payload.get(field), expected)
    records = payload.get("records")
    if not isinstance(records, Mapping) or set(records) != set(DATASETS):
        raise ThreeDatasetTrainingProtocolError(
            "TSS statistics records must contain exactly three datasets"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ThreeDatasetTrainingProtocolError(
            "TSS statistics lack extraction provenance"
        )
    for field, expected in (
        (
            "source_artifact_sha256",
            "885403ce46dff8f8b74a67237c225d2f11786c1b7e3dff9115bfaad081e753ae",
        ),
        ("extraction", "exact_three_dataset_records_only"),
        ("reused_after_exact_three_train_identity_validation", True),
        ("runtime_depends_on_source_artifact", False),
    ):
        _require_equal(f"TSS provenance {field}", provenance.get(field), expected)
    for dataset in DATASETS:
        candidate = records[dataset]
        if not isinstance(candidate, Mapping):
            raise ThreeDatasetTrainingProtocolError(
                f"invalid TSS dataset record: {dataset}"
            )
        for field, expected in (
            ("dataset", dataset),
            ("training_seed", TRAINING_SEED),
            ("epochs", FORMAL_EPOCHS),
            ("completed_through_epoch", FORMAL_EPOCHS),
            ("complete", True),
            (
                "train_ids_sha256",
                data_protocol.EXPECTED_SPLITS[dataset]["train"][
                    "ordered_ids_sha256"
                ],
            ),
            (
                "survival_pos_weight_formula",
                "negative_cells / positive_cells",
            ),
        ):
            _require_equal(
                f"TSS {dataset} {field}", candidate.get(field), expected
            )
        candidate_positive = int(candidate["positive_cells"])
        candidate_negative = int(candidate["negative_cells"])
        candidate_weight = float(candidate["survival_pos_weight"])
        if (
            candidate_positive <= 0
            or candidate_negative <= 0
            or not math.isfinite(candidate_weight)
            or candidate_weight != candidate_negative / candidate_positive
        ):
            raise ThreeDatasetTrainingProtocolError(
                f"invalid TSS arithmetic for {dataset}"
            )
        plan_sha = candidate.get("aggregate_plan_sha256")
        if (
            not isinstance(plan_sha, str)
            or len(plan_sha) != 64
            or any(character not in "0123456789abcdef" for character in plan_sha)
        ):
            raise ThreeDatasetTrainingProtocolError(
                f"invalid TSS aggregate plan identity for {dataset}"
            )
    record = records[args.dataset]
    if not isinstance(record, Mapping):
        raise ThreeDatasetTrainingProtocolError("invalid TSS dataset record")
    for field, expected in (
        ("dataset", args.dataset),
        ("training_seed", TRAINING_SEED),
        ("epochs", FORMAL_EPOCHS),
        ("completed_through_epoch", FORMAL_EPOCHS),
        ("complete", True),
        ("survival_pos_weight_formula", "negative_cells / positive_cells"),
    ):
        _require_equal(f"TSS {args.dataset} {field}", record.get(field), expected)
    positive = int(record["positive_cells"])
    negative = int(record["negative_cells"])
    weight = float(record["survival_pos_weight"])
    if positive <= 0 or negative <= 0 or not math.isfinite(weight):
        raise ThreeDatasetTrainingProtocolError("invalid TSS cell statistics")
    if weight != negative / positive:
        raise ThreeDatasetTrainingProtocolError(
            "TSS survival_pos_weight differs from negative/positive"
        )
    return weight, {
        "enabled": True,
        "source": str(path),
        "sha256": engine.file_sha256(path),
        "schema": payload["schema"],
        "dataset_record": copy.deepcopy(dict(record)),
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
    identity = recipe_identity(args)
    manifest_file = data_manifests["files"]["imgidx"]
    runtime_paths = {
        "runner": Path(__file__).resolve(),
        "training_engine": Path(engine.__file__).resolve(),
        "data_protocol": Path(data_protocol.__file__).resolve(),
        "torch_datasets": REPO_ROOT / "experiments" / "paper_three_dataset_v2.py",
        "model_builder": REPO_ROOT
        / "experiments"
        / "four_dataset_models_seed42_v1.py",
        "training_loss": REPO_ROOT / "experiments" / "tpd_training_loss.py",
        "training_metrics_and_schedule": REPO_ROOT
        / "experiments"
        / "train_tpd_pilot.py",
        "protocol_document": PROTOCOL_DOCUMENT,
    }
    for architecture_path in sorted((REPO_ROOT / "model").rglob("*.py")):
        relative = architecture_path.relative_to(REPO_ROOT).as_posix()
        runtime_paths[f"architecture::{relative}"] = architecture_path
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "method": args.method,
        "recipe": identity,
        "training_seed": TRAINING_SEED,
        "epochs": args.epochs,
        "begin_test": args.begin_test,
        "eval_every": args.eval_every,
        "candidate_epochs": list(
            range(args.begin_test, args.epochs + 1, args.eval_every)
        ),
        "test_selected": True,
        "selection_is_optimistic": True,
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "rolling_resume_state": {
            "enabled_during_training": True,
            "overwritten_each_epoch": True,
            "removed_after_success": True,
            "selected_checkpoint": False,
            "recipe_identity_explicit": True,
        },
        "three_dataset_v2_data_protocol": {
            "module": "experiments.three_dataset_v2_protocol",
            "schema": data_protocol.SCHEMA,
            "manifest_id": data_protocol.MANIFEST_ID,
            "manifest_path": manifest_file["path"],
            "manifest_sha256": manifest_file["sha256"],
            "datasets": list(DATASETS),
            "sirst3_in_formal_matrix": False,
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
            "amp": False,
            "segmentation_loss": "ordered sum BCE over six outputs",
            "tss_requested_weight": identity["requested_tss_weight"],
            "tss_ratio_cap": identity["tss_ratio_cap"],
            "tss_effective_weight_formula": (
                "min(lambda_req,0.10*stopgrad(Lseg)/"
                "max(stopgrad(Ltss),float32_eps))"
                if args.method == "final"
                else None
            ),
            "original_contains_tss_objective": False,
            "tss_epoch_diagnostics": {
                "fields": [
                    "requested_weight",
                    "effective_weight_mean",
                    "effective_weight_p10",
                    "effective_weight_p50",
                    "effective_weight_p90",
                    "effective_weight_std",
                    "effective_weight_max",
                    "raw_weighted_to_seg_ratio_mean",
                    "effective_weighted_to_seg_ratio_mean",
                    "cap_active_batch_fraction",
                    "cap_active_sample_fraction",
                ],
                "mean_and_std_weighting": "minibatch_sample_count",
                "std": "sample_weighted_population_std",
                "weighted_quantile": (
                    "smallest_value_with_cumulative_sample_weight_ge_q_times_N"
                ),
                "cap_active": (
                    "effective_weight < requested_weight*(1-1e-6)"
                ),
                "per_minibatch_pair_saved": [
                    "Lseg",
                    "Ltss",
                    "sample_count",
                    "raw_ratio",
                    "effective_ratio",
                    "counterfactual_effective_weights_for_all_three_lambdas",
                ],
                "participates_in_checkpoint_selection": False,
            },
        },
        "metrics": {
            "threshold": FORMAL_THRESHOLD,
            "threshold_role": "primary_checkpoint_selection_and_headline",
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
        "normalization": data_protocol.get_legacy_normalization(args.dataset),
        "data_manifest": copy.deepcopy(dict(data_manifests)),
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
        "search_budget_disclosure": {
            "per_run_schedule_data_evaluation_matched": True,
            "original_run_count": 3,
            "final_run_count": 9,
            "final_to_original_run_budget_ratio": 3.0,
            "total_search_budget_equal": False,
            "final_global_lambda_candidates": list(TSS_LAMBDAS),
        },
        "protocol_document": {
            "path": str(PROTOCOL_DOCUMENT),
            "sha256": engine.file_sha256(PROTOCOL_DOCUMENT),
        },
        "runtime_sources": {
            name: {
                "path": str(path),
                "sha256": engine.file_sha256(path),
            }
            for name, path in runtime_paths.items()
        },
        "smoke": bool(args.smoke),
    }
    return payload


class _EpochTSSAudit:
    def __init__(self) -> None:
        self.method = "original"
        self.requested_weight = 0.0
        self.reset()

    def configure(self, method: str, requested_weight: float) -> None:
        self.method = method
        self.requested_weight = requested_weight
        self.reset()

    def reset(self) -> None:
        self.samples = 0
        self.effective_sum = 0.0
        self.weighted_sum = 0.0
        self.segmentation_sum = 0.0
        self.cap_active_samples = 0
        self.cap_active_batches = 0
        self.batch_records: list[dict[str, Any]] = []

    def add(self, losses: TPDTrainingLoss, samples: int) -> None:
        effective = float(losses.effective_survival_weight.detach().item())
        weighted = float(losses.weighted_survival.detach().item())
        segmentation = float(losses.segmentation.detach().item())
        survival = float(losses.survival.detach().item())
        if not all(
            math.isfinite(x)
            for x in (effective, weighted, segmentation, survival)
        ):
            raise FloatingPointError("non-finite TSS audit value")
        self.samples += samples
        self.effective_sum += effective * samples
        self.weighted_sum += weighted * samples
        self.segmentation_sum += segmentation * samples
        raw_ratio = (
            self.requested_weight * survival / max(segmentation, 1e-12)
        )
        effective_ratio = weighted / max(segmentation, 1e-12)
        cap_active = effective < self.requested_weight * (1.0 - 1e-6)
        if cap_active:
            self.cap_active_samples += samples
            self.cap_active_batches += 1
        epsilon = torch.finfo(losses.survival.dtype).eps
        counterfactual = {
            _lambda_token(candidate): min(
                candidate,
                TSS_RATIO_CAP * segmentation / max(survival, epsilon),
            )
            for candidate in TSS_LAMBDAS
        }
        self.batch_records.append(
            {
                "batch_index": len(self.batch_records),
                "sample_count": samples,
                "segmentation_loss": segmentation,
                "survival_loss": survival,
                "requested_weight": self.requested_weight,
                "effective_weight": effective,
                "raw_weighted_to_seg_ratio": raw_ratio,
                "effective_weighted_to_seg_ratio": effective_ratio,
                "cap_active": cap_active,
                "counterfactual_effective_weights": counterfactual,
            }
        )

    @staticmethod
    def _weighted_quantile(
        records: list[dict[str, Any]], quantile: float
    ) -> float:
        ordered = sorted(
            (
                (float(record["effective_weight"]), int(record["sample_count"]))
                for record in records
            ),
            key=lambda item: item[0],
        )
        total = sum(weight for _, weight in ordered)
        target = quantile * total
        cumulative = 0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= target:
                return value
        return ordered[-1][0]

    @staticmethod
    def _sample_weighted_mean(
        records: list[dict[str, Any]], field: str
    ) -> float:
        total = sum(int(record["sample_count"]) for record in records)
        return sum(
            float(record[field]) * int(record["sample_count"])
            for record in records
        ) / total

    def payload(self) -> dict[str, Any]:
        if self.method == "original":
            return {
                "train_tss_enabled": False,
                "train_tss_requested_weight": 0.0,
                "requested_weight": 0.0,
            }
        if self.samples <= 0:
            raise RuntimeError("Final TSS audit has no minibatches")
        effective_mean = self.effective_sum / self.samples
        effective_variance = sum(
            int(record["sample_count"])
            * (float(record["effective_weight"]) - effective_mean) ** 2
            for record in self.batch_records
        ) / self.samples
        return {
            "train_tss_enabled": True,
            "train_tss_requested_weight": self.requested_weight,
            "train_tss_ratio_cap": TSS_RATIO_CAP,
            "train_tss_effective_weight_mean": effective_mean,
            "train_tss_effective_weight_p10": self._weighted_quantile(
                self.batch_records, 0.10
            ),
            "train_tss_effective_weight_p50": self._weighted_quantile(
                self.batch_records, 0.50
            ),
            "train_tss_effective_weight_p90": self._weighted_quantile(
                self.batch_records, 0.90
            ),
            "train_tss_effective_weight_std": math.sqrt(effective_variance),
            "train_tss_effective_weight_max": max(
                float(record["effective_weight"])
                for record in self.batch_records
            ),
            "train_tss_weighted_loss": self.weighted_sum / self.samples,
            "train_tss_raw_weighted_to_seg_ratio_mean": (
                self._sample_weighted_mean(
                    self.batch_records, "raw_weighted_to_seg_ratio"
                )
            ),
            "train_tss_effective_weighted_to_seg_ratio_mean": (
                self._sample_weighted_mean(
                    self.batch_records, "effective_weighted_to_seg_ratio"
                )
            ),
            "train_tss_weighted_to_segmentation_ratio": (
                self.weighted_sum / max(self.segmentation_sum, 1e-12)
            ),
            "train_tss_cap_active_batch_fraction": (
                self.cap_active_batches / len(self.batch_records)
            ),
            "train_tss_cap_active_sample_fraction": (
                self.cap_active_samples / self.samples
            ),
            "train_tss_weighted_quantile_algorithm": (
                "smallest_value_with_cumulative_sample_weight_ge_q_times_N"
            ),
            "train_tss_weighted_std_algorithm": "sample_weighted_population_std",
            "train_tss_batch_diagnostics": copy.deepcopy(self.batch_records),
            "requested_weight": self.requested_weight,
            "effective_weight_mean": effective_mean,
            "effective_weight_p10": self._weighted_quantile(
                self.batch_records, 0.10
            ),
            "effective_weight_p50": self._weighted_quantile(
                self.batch_records, 0.50
            ),
            "effective_weight_p90": self._weighted_quantile(
                self.batch_records, 0.90
            ),
            "effective_weight_std": math.sqrt(effective_variance),
            "effective_weight_max": max(
                float(record["effective_weight"])
                for record in self.batch_records
            ),
            "raw_weighted_to_seg_ratio_mean": self._sample_weighted_mean(
                self.batch_records, "raw_weighted_to_seg_ratio"
            ),
            "effective_weighted_to_seg_ratio_mean": (
                self._sample_weighted_mean(
                    self.batch_records, "effective_weighted_to_seg_ratio"
                )
            ),
            "cap_active_batch_fraction": (
                self.cap_active_batches / len(self.batch_records)
            ),
            "cap_active_sample_fraction": self.cap_active_samples / self.samples,
        }


_AUDIT = _EpochTSSAudit()


def _compute_loss_v2(
    output: Any,
    segmentation_target: torch.Tensor,
    segmentation_criterion: nn.Module,
    *,
    survival_weight: float = 0.0,
    survival_pos_weight: float | torch.Tensor = 1.0,
) -> TPDTrainingLoss:
    if _AUDIT.method == "original":
        if float(survival_weight) != 0.0:
            raise ThreeDatasetTrainingProtocolError(
                "Original received a nonzero TSS weight"
            )
        return _compute_tpd_training_loss(
            output,
            segmentation_target,
            segmentation_criterion,
            survival_weight=0.0,
            survival_pos_weight=survival_pos_weight,
        )
    if float(survival_weight) != _AUDIT.requested_weight:
        raise ThreeDatasetTrainingProtocolError(
            "Final runtime TSS weight differs from recipe identity"
        )
    losses = _compute_tpd_training_loss(
        output,
        segmentation_target,
        segmentation_criterion,
        survival_weight=survival_weight,
        survival_pos_weight=survival_pos_weight,
        survival_ratio_cap=TSS_RATIO_CAP,
    )
    _AUDIT.add(losses, int(segmentation_target.shape[0]))
    return losses


_ENGINE_SELECTED_PAYLOAD = engine._selected_checkpoint_payload
_ENGINE_LATEST_PAYLOAD = engine._latest_checkpoint_payload
_ENGINE_WRITE_JSON = engine.write_json_atomic


def _selected_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _ENGINE_SELECTED_PAYLOAD(*args, **kwargs)
    run_args = kwargs["args"]
    payload["schema"] = SCHEMA
    payload["recipe"] = recipe_identity(run_args)
    payload["requested_tss_weight"] = requested_tss_weight(run_args)
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
    payload["schema"] = SCHEMA
    payload["recipe"] = identity
    payload["requested_tss_weight"] = identity["requested_tss_weight"]
    _AUDIT.reset()
    return payload


def _enrich_json_artifact(path: Path, value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if path.name not in {"progress.json", "summary.json"}:
        return value
    enriched = copy.deepcopy(dict(value))
    method = enriched.get("method")
    if method not in METHODS:
        return enriched
    pseudo = argparse.Namespace(method=method, tss_weight=engine.FORMAL_TSS_WEIGHT)
    if method == "original":
        pseudo.tss_weight = None
    identity = recipe_identity(pseudo)
    enriched["schema"] = SCHEMA
    enriched["recipe"] = identity
    enriched["requested_tss_weight"] = identity["requested_tss_weight"]
    enriched["checkpoint_roles"] = list(CHECKPOINT_ROLES)
    if path.name == "summary.json":
        enriched["search_budget_disclosure"] = {
            "per_run_schedule_data_evaluation_matched": True,
            "original_run_count": 3,
            "final_run_count": 9,
            "final_to_original_run_budget_ratio": 3.0,
            "total_search_budget_equal": False,
        }
    return enriched


def _write_json_atomic(path: Path, value: Any) -> None:
    _ENGINE_WRITE_JSON(path, _enrich_json_artifact(path, value))


def _validate_existing_summary_identity(args: argparse.Namespace) -> None:
    path = _run_directory(args) / "summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = recipe_identity(args)
    for field, expected in (
        ("schema", SCHEMA),
        ("dataset", args.dataset),
        ("method", args.method),
        ("seed", TRAINING_SEED),
        ("recipe", identity),
        ("requested_tss_weight", identity["requested_tss_weight"]),
    ):
        _require_equal(f"existing summary {field}", payload.get(field), expected)


def _load_resume_v2(
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
        ("method", args.method),
        ("seed", TRAINING_SEED),
        ("protocol_sha256", protocol_sha256),
        ("recipe", identity),
        ("requested_tss_weight", identity["requested_tss_weight"]),
    ):
        _require_equal(f"resume {field}", payload.get(field), expected)
    event = payload.get("event")
    if not isinstance(event, Mapping):
        raise ThreeDatasetTrainingProtocolError(
            "resume state lacks its completed epoch event"
        )
    completed_epoch = int(payload["epoch"])
    _require_equal("resume event epoch", event.get("epoch"), completed_epoch)
    _require_equal("resume event recipe", event.get("recipe"), identity)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    engine.restore_rng_state(payload["rng_state"])
    return (
        completed_epoch + 1,
        dict(payload.get("best_miou", {})),
        dict(payload.get("best_pd", {})),
        dict(event),
    )


@contextmanager
def _patched_engine(args: argparse.Namespace) -> Iterator[None]:
    identity = recipe_identity(args)
    _AUDIT.configure(args.method, identity["requested_tss_weight"])
    replacements = {
        "SCHEMA": SCHEMA,
        "DATASETS": DATASETS,
        "LEGACY_NORMALIZATION": data_protocol.LEGACY_NORMALIZATION,
        "PROTOCOL_DOCUMENT": PROTOCOL_DOCUMENT,
        "FORMAL_TSS_WEIGHT": identity["requested_tss_weight"],
        "validate_args": validate_args,
        "_run_directory": _run_directory,
        "_load_data_manifest_lock": _load_data_manifest_lock,
        "_import_runtime_components": _import_runtime_components,
        "_load_tss_pos_weight": _validate_tss_statistics,
        "_protocol_payload": _protocol_payload,
        "compute_tpd_training_loss": _compute_loss_v2,
        "_selected_checkpoint_payload": _selected_checkpoint_payload,
        "_latest_checkpoint_payload": _latest_checkpoint_payload,
        "_load_resume": _load_resume_v2,
        "write_json_atomic": _write_json_atomic,
    }
    previous = {name: getattr(engine, name) for name in replacements}
    for name, value in replacements.items():
        setattr(engine, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)
        _AUDIT.configure("original", 0.0)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    _validate_existing_summary_identity(args)
    with _patched_engine(args):
        summary_path = engine.run(args)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = recipe_identity(args)
    _require_equal("summary schema", payload.get("schema"), SCHEMA)
    _require_equal("summary recipe", payload.get("recipe"), identity)
    _require_equal(
        "summary requested_tss_weight",
        payload.get("requested_tss_weight"),
        identity["requested_tss_weight"],
    )
    return summary_path


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
