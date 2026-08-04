#!/usr/bin/env python3
"""Formal seed-42 Final/TSS-off trainer for the three paper datasets.

This is a separate run identity layered over the frozen three-dataset training
engine.  It deliberately keeps the exact Final training model class and
forward path (including the registered, zero-initialized TSS heads), while the
loss takes its exact ``survival_weight == 0`` branch.  Consequently the model
still computes survival logits during training, but the loss neither consumes
those logits nor constructs the stride-16 survival target.

Only ``best_miou`` and ``best_pd`` are retained as selected checkpoints.  The
rolling model/optimizer/RNG state remains a resume artifact and is removed
after successful completion by the reused engine.
"""

from __future__ import annotations

import argparse
import copy
import json
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
from experiments.tpd_training_loss import (  # noqa: E402
    TPDTrainingLoss,
    compute_tpd_training_loss as _compute_tpd_training_loss,
)


SCHEMA = "sctransnet_three_dataset_tss_off_seed42_v1/v1"
TRAINING_SEED = positive_runner.TRAINING_SEED
DATASETS = positive_runner.DATASETS
METHOD = "final"
TSS_REQUESTED_WEIGHT = 0.0
TSS_RATIO_CAP = positive_runner.TSS_RATIO_CAP
CHECKPOINT_ROLES = positive_runner.CHECKPOINT_ROLES

FORMAL_EPOCHS = positive_runner.FORMAL_EPOCHS
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
    REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
)
DEFAULT_PROTOCOL_MANIFEST = positive_runner.DEFAULT_PROTOCOL_MANIFEST
DEFAULT_TSS_STATISTICS = positive_runner.DEFAULT_TSS_STATISTICS
PROTOCOL_DOCUMENT = REPO_ROOT / (
    "SCTransNet_正TSS全局配方失败后的TSS-Off因果诊断方案.md"
)
GPU_UUIDS = positive_runner.GPU_UUIDS


class ThreeDatasetTSSOffProtocolError(ValueError):
    """One command or persisted artifact violates the TSS-off contract."""


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ThreeDatasetTSSOffProtocolError(
            f"{name} differs: {actual!r} != {expected!r}"
        )


def requested_tss_weight(args: argparse.Namespace) -> float:
    if args.method != METHOD:
        raise ThreeDatasetTSSOffProtocolError(
            "the TSS-off entry accepts only method='final'"
        )
    if args.tss_weight not in (None, 0, 0.0):
        raise ThreeDatasetTSSOffProtocolError(
            "Final TSS-off requires --tss-weight 0 or an omitted weight"
        )
    return TSS_REQUESTED_WEIGHT


def recipe_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": METHOD,
        "recipe_id": "final_tss_off",
        "requested_tss_weight": requested_tss_weight(args),
        "tss_lambda_token": "off",
        "tss_ratio_cap": TSS_RATIO_CAP,
        "tss_ratio_cap_applied": False,
        "tss_enabled": False,
        "tss_heads_registered": True,
        "tss_training_forward_computes_logits": True,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
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
    # Persist one canonical numeric value regardless of whether the CLI omitted
    # the optional spelling.
    requested_tss_weight(args)
    args.tss_weight = TSS_REQUESTED_WEIGHT
    # Compatibility fields consumed by the reused training engine.
    args.manifest_root = args.protocol_manifest.parent
    args.survival_pos_weight = None
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_runner.data_protocol.require_dataset(args.dataset)
    positive_runner.data_protocol.require_seed(args.seed)
    requested_tss_weight(args)
    if args.eval_every < 1 or args.epochs < 1 or args.begin_test < 1:
        raise ThreeDatasetTSSOffProtocolError(
            "epoch controls must be positive"
        )
    if args.batch_size < 1 or args.workers < 0:
        raise ThreeDatasetTSSOffProtocolError(
            "invalid loader configuration"
        )
    _require_equal("patch_size", args.patch_size, FORMAL_PATCH_SIZE)
    _require_equal("metric threshold", args.threshold, FORMAL_THRESHOLD)
    if args.smoke:
        if args.epochs > 2:
            raise ThreeDatasetTSSOffProtocolError(
                "smoke runs are limited to two epochs"
            )
        if args.max_train_images is None or args.max_test_images is None:
            raise ThreeDatasetTSSOffProtocolError(
                "smoke requires train/test image limits"
            )
        if args.device == "cuda:0":
            if args.physical_gpu_index not in GPU_UUIDS:
                raise ThreeDatasetTSSOffProtocolError(
                    "CUDA smoke requires physical GPU 2 or 3"
                )
            _require_equal(
                "expected GPU UUID",
                args.expected_gpu_uuid,
                GPU_UUIDS[args.physical_gpu_index],
            )
        return

    if args.max_train_images is not None or args.max_test_images is not None:
        raise ThreeDatasetTSSOffProtocolError(
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
        raise ThreeDatasetTSSOffProtocolError(
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
    return root / "runs" / args.dataset / "final_tss_off" / "seed_42"


def _build_method_model(
    method: str,
    seed: int,
    *,
    dataset_name: str,
) -> tuple[nn.Module, dict[str, Any]]:
    if method != METHOD:
        raise ThreeDatasetTSSOffProtocolError(
            "TSS-off model builder accepts only Final"
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
        raise ThreeDatasetTSSOffProtocolError(
            "Final builder metadata lacks its training objective"
        )
    objective.update(
        {
            "authority": "three_dataset_tss_off_seed42_v1_run_recipe",
            "tss_enabled": False,
            "tss_heads_registered": True,
            "tss_training_forward_computes_logits": True,
            "tss_loss_consumes_logits": False,
            "tss_survival_target_constructed": False,
        }
    )
    metadata["tss_off_control"] = copy.deepcopy(recipe_identity(
        argparse.Namespace(method=METHOD, tss_weight=0.0)
    ))
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
    # The exact-zero loss branch does not consume pos_weight.  Keep the CLI
    # field for interface compatibility, but do not make an irrelevant
    # statistics artifact a runtime input.
    return 1.0, {
        "enabled": False,
        "requested_tss_weight": TSS_REQUESTED_WEIGHT,
        "statistics_consumed": False,
        "configured_path": (
            str(args.tss_statistics.resolve())
            if args.tss_statistics is not None
            else None
        ),
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
    # Reuse the complete frozen optimizer/data/checkpoint description with a
    # throwaway positive recipe, then replace every objective/run-identity field
    # by the authoritative TSS-off contract below.
    surrogate = copy.copy(args)
    surrogate.method = METHOD
    surrogate.tss_weight = positive_runner.TSS_LAMBDAS[0]
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
    payload["model"] = copy.deepcopy(dict(model_metadata))
    payload["tss"] = copy.deepcopy(dict(tss_metadata))
    training = payload["training"]
    training.update(
        {
            "tss_enabled": False,
            "tss_requested_weight": TSS_REQUESTED_WEIGHT,
            "tss_ratio_cap": TSS_RATIO_CAP,
            "tss_ratio_cap_applied": False,
            "tss_effective_weight_formula": (
                "exact_zero_short_circuit_before_survival_target_or_"
                "survival_logit_consumption"
            ),
            "tss_survival_target_constructed": False,
            "tss_survival_logits_consumed_by_loss": False,
            "tss_training_forward_computes_logits": True,
            "tss_head_gradients": "none",
            "shared_feature_tss_gradient": "none",
            "tss_epoch_diagnostics": {
                "fields": [
                    "train_tss_enabled",
                    "train_tss_requested_weight",
                    "train_tss_ratio_cap",
                    "train_tss_ratio_cap_applied",
                    "train_tss_survival_target_constructed",
                    "train_tss_survival_logits_consumed_by_loss",
                    "train_tss_training_forward_computes_logits",
                    "train_tss_observed_batches",
                    "train_tss_observed_samples",
                ],
                "participates_in_checkpoint_selection": False,
            },
        }
    )
    payload["search_budget_disclosure"] = {
        "experiment_role": "post_positive_tss_search_off_control",
        "tss_off_training_runs": len(DATASETS),
        "one_run_per_dataset": True,
        "positive_lambda_search_expanded": False,
        "comparison_uses_existing_original_and_positive_runs": True,
        "positive_final_training_runs": 9,
        "final_family_training_runs_after_tss_off": 12,
        "original_training_runs": 3,
        "final_to_original_recipe_search_ratio": 4.0,
        "per_run_protocol_matched": True,
        "pairwise_tss_off_vs_original_run_count_equal": True,
        "total_recipe_search_budget_equal": False,
        "tss_off_added_after_positive_test_results": True,
        "test_selected": True,
    }
    payload["protocol_document"] = {
        "path": str(PROTOCOL_DOCUMENT),
        "sha256": engine.file_sha256(PROTOCOL_DOCUMENT),
    }
    runtime_sources = payload["runtime_sources"]
    runtime_sources["runner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": engine.file_sha256(Path(__file__).resolve()),
    }
    runtime_sources["reused_positive_runner"] = {
        "path": str(Path(positive_runner.__file__).resolve()),
        "sha256": engine.file_sha256(Path(positive_runner.__file__).resolve()),
    }
    return payload


class _EpochTSSOffAudit:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples = 0
        self.batches = 0

    def add(self, losses: TPDTrainingLoss, samples: int) -> None:
        if type(samples) is not int or samples < 1:
            raise ThreeDatasetTSSOffProtocolError(
                "TSS-off audit sample count must be positive"
            )
        if not torch.equal(losses.total.detach(), losses.segmentation.detach()):
            raise ThreeDatasetTSSOffProtocolError(
                "TSS-off total loss differs from segmentation loss"
            )
        if losses.survival_terms != ():
            raise ThreeDatasetTSSOffProtocolError(
                "TSS-off unexpectedly constructed survival loss terms"
            )
        zero_values = (
            losses.survival,
            losses.effective_survival_weight,
            losses.weighted_survival,
        )
        if any(float(value.detach().item()) != 0.0 for value in zero_values):
            raise ThreeDatasetTSSOffProtocolError(
                "TSS-off loss returned a nonzero auxiliary value"
            )
        self.samples += samples
        self.batches += 1

    def payload(self) -> dict[str, Any]:
        if self.samples < 1 or self.batches < 1:
            raise ThreeDatasetTSSOffProtocolError(
                "TSS-off epoch audit has no minibatches"
            )
        return {
            "train_tss_enabled": False,
            "train_tss_requested_weight": TSS_REQUESTED_WEIGHT,
            "requested_weight": TSS_REQUESTED_WEIGHT,
            "train_tss_ratio_cap": TSS_RATIO_CAP,
            "train_tss_ratio_cap_applied": False,
            "train_tss_survival_target_constructed": False,
            "train_tss_survival_logits_consumed_by_loss": False,
            "train_tss_training_forward_computes_logits": True,
            "train_tss_head_gradients": "none",
            "train_tss_shared_feature_auxiliary_gradient": "none",
            "train_tss_loss_exact_segmentation_only": True,
            "train_tss_observed_batches": self.batches,
            "train_tss_observed_samples": self.samples,
        }


_AUDIT = _EpochTSSOffAudit()


def _compute_loss_off(
    output: Any,
    segmentation_target: torch.Tensor,
    segmentation_criterion: nn.Module,
    *,
    survival_weight: float = 0.0,
    survival_pos_weight: float | torch.Tensor = 1.0,
) -> TPDTrainingLoss:
    if float(survival_weight) != TSS_REQUESTED_WEIGHT:
        raise ThreeDatasetTSSOffProtocolError(
            "TSS-off runtime received a nonzero survival weight"
        )
    losses = _compute_tpd_training_loss(
        output,
        segmentation_target,
        segmentation_criterion,
        survival_weight=TSS_REQUESTED_WEIGHT,
        survival_pos_weight=survival_pos_weight,
        # Retain the frozen protocol field.  The loss returns before it can
        # influence any value because requested weight is exactly zero.
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
    payload["requested_tss_weight"] = TSS_REQUESTED_WEIGHT
    payload["tss_enabled"] = False
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
    payload["requested_tss_weight"] = TSS_REQUESTED_WEIGHT
    payload["tss_enabled"] = False
    _AUDIT.reset()
    return payload


def _load_resume_off(
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
        ("requested_tss_weight", TSS_REQUESTED_WEIGHT),
        ("tss_enabled", False),
    ):
        _require_equal(f"resume {field}", payload.get(field), expected)
    event = payload.get("event")
    if not isinstance(event, Mapping):
        raise ThreeDatasetTSSOffProtocolError(
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
    enriched["schema"] = SCHEMA
    enriched["recipe"] = identity
    enriched["requested_tss_weight"] = TSS_REQUESTED_WEIGHT
    enriched["tss_enabled"] = False
    enriched["checkpoint_roles"] = list(CHECKPOINT_ROLES)
    if path.name == "summary.json":
        enriched["experiment_role"] = "post_positive_tss_search_off_control"
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
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("recipe", identity),
        ("requested_tss_weight", TSS_REQUESTED_WEIGHT),
        ("tss_enabled", False),
    ):
        _require_equal(f"existing summary {field}", payload.get(field), expected)


@contextmanager
def _patched_engine(args: argparse.Namespace) -> Iterator[None]:
    recipe_identity(args)
    _AUDIT.reset()
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
        "compute_tpd_training_loss": _compute_loss_off,
        "_selected_checkpoint_payload": _selected_checkpoint_payload,
        "_latest_checkpoint_payload": _latest_checkpoint_payload,
        "_load_resume": _load_resume_off,
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
        _AUDIT.reset()


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    _validate_existing_summary_identity(args)
    with _patched_engine(args):
        summary_path = engine.run(args)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = recipe_identity(args)
    for field, expected in (
        ("schema", SCHEMA),
        ("recipe", identity),
        ("requested_tss_weight", TSS_REQUESTED_WEIGHT),
        ("tss_enabled", False),
    ):
        _require_equal(f"summary {field}", payload.get(field), expected)
    return summary_path


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
