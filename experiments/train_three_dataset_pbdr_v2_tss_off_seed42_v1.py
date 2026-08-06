#!/usr/bin/env python3
"""Three-dataset seed42 PBDR-V2 scratch trainer with TSS disabled.

This adapter reuses the frozen data, optimizer, evaluation, checkpoint, and
resume engine.  It owns an independent PBDR-V2 recipe/schema and accepts only
the 573-key PBDR-V2 training graph.  Formal checkpoint selection remains the
two-role ``best_miou`` / ``best_pd`` policy at threshold 0.5.
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

from experiments import three_dataset_pbdr_v2_models_seed42_v1 as models  # noqa: E402
from experiments import train_three_dataset_tss_off_seed42_v1 as base  # noqa: E402
from experiments import train_three_dataset_seed42_global_tss_v2 as positive  # noqa: E402
from experiments import train_four_dataset_original_final_seed42_exact_v1 as engine  # noqa: E402
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2 import (  # noqa: E402
    PBDR_V2_INTEGRATION_VERSION,
    PBDR_V2_STATE_KEYS,
    PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_PARAMETERS,
)


SCHEMA = "sctransnet_three_dataset_pbdr_v2_tss_off_seed42_v1/v1"
RECIPE_ID = models.RECIPE_ID
TRAINING_SEED = 42
DATASETS = models.DATASETS
METHOD = "final"
TSS_REQUESTED_WEIGHT = 0.0
FORMAL_EPOCHS = 1000
FORMAL_BEGIN_TEST = positive.FORMAL_BEGIN_TEST
FORMAL_EVAL_EVERY = positive.FORMAL_EVAL_EVERY
FORMAL_BATCH_SIZE = positive.FORMAL_BATCH_SIZE
FORMAL_PATCH_SIZE = positive.FORMAL_PATCH_SIZE
FORMAL_WORKERS = positive.FORMAL_WORKERS
FORMAL_BASE_LR = positive.FORMAL_BASE_LR
FORMAL_MIN_LR = positive.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = positive.FORMAL_WARMUP_EPOCHS
FORMAL_THRESHOLD = positive.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = positive.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = positive.FORMAL_TINY_AREA
CHECKPOINT_ROLES = positive.CHECKPOINT_ROLES
DEFAULT_DATA_ROOT = positive.DEFAULT_DATA_ROOT
DEFAULT_PROTOCOL_MANIFEST = positive.DEFAULT_PROTOCOL_MANIFEST
DEFAULT_TSS_STATISTICS = positive.DEFAULT_TSS_STATISTICS
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "results/three_dataset_pbdr_v2_tss_off_seed42_v1"
)
PROTOCOL_DOCUMENT = REPO_ROOT / "experiments/PBDR_V2_PROTOCOL.md"
GPU_UUIDS = {
    "0": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "1": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}

_BASE_PROTOCOL_PAYLOAD = base._protocol_payload
_ENGINE_WRITE_JSON = engine.write_json_atomic


class PBDRV2TrainingProtocolError(ValueError):
    """A command or persisted artifact violates the PBDR-V2 recipe."""


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise PBDRV2TrainingProtocolError(
            f"{name} differs: {actual!r} != {expected!r}"
        )


def recipe_identity(args: argparse.Namespace) -> dict[str, Any]:
    _require_equal("method", args.method, METHOD)
    if args.tss_weight not in (None, 0, 0.0):
        raise PBDRV2TrainingProtocolError("PBDR-V2 requires TSS weight 0")
    return {
        "method": METHOD,
        "recipe_id": RECIPE_ID,
        "architecture": "tpd8_ner4_qfg2_croa_pbdr_v2",
        "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
        "pbdr_v2_parameter_count": 19,
        "pbdr_v2_state_key_count": len(PBDR_V2_STATE_KEYS),
        "requested_tss_weight": TSS_REQUESTED_WEIGHT,
        "tss_enabled": False,
        "tss_heads_registered": True,
        "tss_training_forward_computes_logits": True,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
        "fresh_seed42_scratch": True,
        "warm_start_used": False,
        "parent_checkpoint": None,
        "resume_scope": "same_pbdr_v2_run_only",
        "current_shared_initial_state_bitwise_equal": True,
        "pbdr_v2_new_state_exact_zero": True,
    }


def runtime_source_paths() -> dict[str, Path]:
    return models.runtime_source_paths()


def runtime_source_records() -> dict[str, dict[str, str]]:
    return {
        key: {"path": str(path), "sha256": models.file_sha256(path)}
        for key, path in runtime_source_paths().items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=(METHOD,), required=True)
    parser.add_argument("--tss-weight", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
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
    parser.add_argument("--warmup-epochs", type=int, default=FORMAL_WARMUP_EPOCHS)
    parser.add_argument("--threshold", type=float, default=FORMAL_THRESHOLD)
    parser.add_argument("--match-radius", type=float, default=FORMAL_MATCH_RADIUS)
    parser.add_argument("--tiny-area", type=int, default=FORMAL_TINY_AREA)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--physical-gpu-index", choices=tuple(GPU_UUIDS))
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    args = parser.parse_args(argv)
    args.manifest_root = args.protocol_manifest.parent
    args.survival_pos_weight = None
    recipe_identity(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive.data_protocol.require_dataset(args.dataset)
    positive.data_protocol.require_seed(args.seed)
    recipe_identity(args)
    if args.eval_every < 1 or args.epochs < 1 or args.begin_test < 1:
        raise PBDRV2TrainingProtocolError("epoch controls must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise PBDRV2TrainingProtocolError("invalid loader configuration")
    _require_equal("patch_size", args.patch_size, FORMAL_PATCH_SIZE)
    _require_equal("threshold", args.threshold, FORMAL_THRESHOLD)
    if args.smoke:
        if args.epochs > 3:
            raise PBDRV2TrainingProtocolError("smoke is limited to 3 epochs")
        if args.max_train_images is None or args.max_test_images is None:
            raise PBDRV2TrainingProtocolError("smoke requires image limits")
        if args.device == "cuda:0":
            if args.physical_gpu_index not in GPU_UUIDS:
                raise PBDRV2TrainingProtocolError(
                    "CUDA smoke requires a physical GPU binding"
                )
            _require_equal(
                "GPU UUID",
                args.expected_gpu_uuid,
                GPU_UUIDS[args.physical_gpu_index],
            )
        return
    if args.max_train_images is not None or args.max_test_images is not None:
        raise PBDRV2TrainingProtocolError("formal runs cannot limit images")
    expected = {
        "epochs": FORMAL_EPOCHS,
        "begin_test": FORMAL_BEGIN_TEST,
        "eval_every": FORMAL_EVAL_EVERY,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "device": "cuda:0",
    }
    for field, value in expected.items():
        _require_equal(field, getattr(args, field), value)
    if args.physical_gpu_index not in GPU_UUIDS:
        raise PBDRV2TrainingProtocolError("formal run requires GPU binding")
    _require_equal(
        "GPU UUID",
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
    _require_equal("builder method", method, METHOD)
    model, metadata = models.build_pbdr_v2_training_model(dataset_name, seed)
    if metadata.get("all_pre_pbdr_state_bitwise_equal_to_current") is not True:
        raise PBDRV2TrainingProtocolError(
            "PBDR-V2 builder lacks paired Current-state attestation"
        )
    if metadata.get("pbdr_v2_new_state_zero_initialized") is not True:
        raise PBDRV2TrainingProtocolError(
            "PBDR-V2 builder lacks zero-extension attestation"
        )
    return model, metadata


def _import_runtime_components() -> tuple[Any, Any, Any]:
    return (
        _build_method_model,
        positive._train_dataset_adapter,
        positive._test_dataset_adapter,
    )


def _protocol_payload(args: argparse.Namespace, **kwargs: Any) -> dict[str, Any]:
    payload = _BASE_PROTOCOL_PAYLOAD(args, **kwargs)
    metadata = dict(kwargs["model_metadata"])
    identity = recipe_identity(args)
    payload.update(
        {
            "schema": SCHEMA,
            "method": METHOD,
            "recipe": identity,
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_enabled": False,
            "model": copy.deepcopy(metadata),
            "planned_total_epochs": args.epochs,
            "development_protocol": "seed42_img_idx_test_selected",
            "paper_unbiased_test_supported": False,
            "runtime_sources": runtime_source_records(),
            "pbdr_v2_architecture_binding": {
                "architecture_id": metadata["architecture_id"],
                "integration_version": PBDR_V2_INTEGRATION_VERSION,
                "training_parameter_count": (
                    PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_PARAMETERS
                ),
                "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
                "inference_state_key_count": models.INFERENCE_STATE_KEY_COUNT,
                "new_state_keys": list(PBDR_V2_STATE_KEYS),
            },
        }
    )
    payload["training"].update(identity)
    payload["training"]["selection_threshold"] = FORMAL_THRESHOLD
    payload["training"]["checkpoint_roles"] = list(CHECKPOINT_ROLES)
    payload["search_budget_disclosure"] = {
        "new_pbdr_v2_training_runs": len(DATASETS),
        "one_scratch_run_per_dataset": True,
        "checkpoint_roles_per_run": list(CHECKPOINT_ROLES),
        "test_selected": True,
        "fixed_seed": TRAINING_SEED,
    }
    return payload


def _architecture_binding(model: nn.Module) -> tuple[str, Mapping[str, Any]]:
    manifest = model.architecture_manifest()
    return models.canonical_sha256(manifest), manifest


def _selected_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = base._ENGINE_SELECTED_PAYLOAD(*args, **kwargs)
    run_args = kwargs["args"]
    architecture_id, _ = _architecture_binding(kwargs["model"])
    payload.update(
        {
            "schema": SCHEMA,
            "recipe": recipe_identity(run_args),
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_enabled": False,
            "architecture_id": architecture_id,
            "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
            "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
        }
    )
    return payload


def _latest_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    event = kwargs["event"]
    run_args = kwargs["args"]
    identity = recipe_identity(run_args)
    event["recipe"] = identity
    event.update(base._AUDIT.payload())
    payload = base._ENGINE_LATEST_PAYLOAD(*args, **kwargs)
    architecture_id, _ = _architecture_binding(kwargs["model"])
    payload.update(
        {
            "schema": SCHEMA,
            "recipe": identity,
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_enabled": False,
            "architecture_id": architecture_id,
            "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
            "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
            "planned_total_epochs": run_args.epochs,
        }
    )
    base._AUDIT.reset()
    return payload


def _load_resume_pbdr_v2(
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
        raise FileExistsError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture_id, _ = _architecture_binding(model)
    for field, expected in (
        ("schema", SCHEMA),
        ("dataset", args.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("protocol_sha256", protocol_sha256),
        ("recipe", recipe_identity(args)),
        ("architecture_id", architecture_id),
        ("pbdr_v2_integration_version", PBDR_V2_INTEGRATION_VERSION),
        ("training_state_key_count", models.TRAINING_STATE_KEY_COUNT),
        ("planned_total_epochs", args.epochs),
    ):
        _require_equal(f"resume {field}", payload.get(field), expected)
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or len(state) != models.TRAINING_STATE_KEY_COUNT:
        raise PBDRV2TrainingProtocolError(
            f"resume is not a {models.TRAINING_STATE_KEY_COUNT}-key PBDR-V2 state"
        )
    if not set(PBDR_V2_STATE_KEYS) <= set(state):
        raise PBDRV2TrainingProtocolError("resume lacks PBDR-V2 state keys")
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    engine.restore_rng_state(payload["rng_state"])
    completed = int(payload["epoch"])
    event = payload.get("event")
    if not isinstance(event, Mapping) or event.get("recipe") != recipe_identity(args):
        raise PBDRV2TrainingProtocolError("resume event is not PBDR-V2-only")
    return (
        completed + 1,
        dict(payload.get("best_miou", {})),
        dict(payload.get("best_pd", {})),
        dict(event),
    )


def _enrich_json(path: Path, value: Any) -> Any:
    if not isinstance(value, Mapping) or path.name not in {
        "progress.json",
        "summary.json",
    }:
        return value
    enriched = copy.deepcopy(dict(value))
    enriched.update(
        {
            "schema": SCHEMA,
            "recipe": recipe_identity(
                argparse.Namespace(method=METHOD, tss_weight=0.0)
            ),
            "requested_tss_weight": TSS_REQUESTED_WEIGHT,
            "tss_enabled": False,
            "planned_total_epochs": enriched.get(
                "total_epochs", enriched.get("epochs", FORMAL_EPOCHS)
            ),
            "checkpoint_roles": list(CHECKPOINT_ROLES),
            "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
            "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
        }
    )
    return enriched


def _write_json_atomic(path: Path, value: Any) -> None:
    _ENGINE_WRITE_JSON(path, _enrich_json(path, value))


@contextmanager
def _patched_base_and_engine(args: argparse.Namespace) -> Iterator[None]:
    replacements = {
        "SCHEMA": SCHEMA,
        "DATASETS": DATASETS,
        "PROTOCOL_DOCUMENT": PROTOCOL_DOCUMENT,
        "GPU_UUIDS": GPU_UUIDS,
        "recipe_identity": recipe_identity,
        "validate_args": validate_args,
        "_run_directory": _run_directory,
        "_build_method_model": _build_method_model,
        "_import_runtime_components": _import_runtime_components,
        "_protocol_payload": _protocol_payload,
        "_selected_checkpoint_payload": _selected_checkpoint_payload,
        "_latest_checkpoint_payload": _latest_checkpoint_payload,
        "_load_resume_off": _load_resume_pbdr_v2,
        "_write_json_atomic": _write_json_atomic,
    }
    previous = {key: getattr(base, key) for key in replacements}
    for key, value in replacements.items():
        setattr(base, key, value)
    try:
        with base._patched_engine(args):
            yield
    finally:
        for key, value in previous.items():
            setattr(base, key, value)
        base._AUDIT.reset()


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    with _patched_base_and_engine(args):
        output = engine.run(args)
    summary = json.loads(output.read_text(encoding="utf-8"))
    for field, expected in (
        ("schema", SCHEMA),
        ("recipe", recipe_identity(args)),
        ("requested_tss_weight", TSS_REQUESTED_WEIGHT),
        ("tss_enabled", False),
        ("planned_total_epochs", args.epochs),
        ("pbdr_v2_integration_version", PBDR_V2_INTEGRATION_VERSION),
        ("training_state_key_count", models.TRAINING_STATE_KEY_COUNT),
    ):
        _require_equal(f"summary {field}", summary.get(field), expected)
    return output


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()
