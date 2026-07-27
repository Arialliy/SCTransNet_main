#!/usr/bin/env python3
"""Run the frozen V6 sweep with an explicit checkpoint-metric audit adapter.

The formal V6 checkpoints intentionally persist only the five selection
metrics.  The frozen generic sweep evaluator additionally audits object-count
metrics at threshold 0.5.  This post-freeze adapter reads those fields from
the authoritative row of ``metrics.jsonl`` and adds them only to an in-memory
copy passed to the frozen audit function.  It never writes a checkpoint,
metrics log, summary, or enriched checkpoint metric dictionary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as base  # noqa: E402
from experiments import evaluate_tpd_clean_v6_pd_fa as frozen  # noqa: E402
from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402
from experiments import validate_tpd_clean_v6_strict_sweeps as strict  # noqa: E402


COMPATIBILITY_SCHEMA = (
    "sctransnet_tpd_clean_v6_postfreeze_checkpoint_metric_compatibility_v1"
)
COMPATIBILITY_KEY = "postfreeze_checkpoint_metric_compatibility"
SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_clean_v6_checkpoint_metric_compatibility_source_lock_v1"
)
DEFAULT_COMPATIBILITY_SOURCE_LOCK = (
    REPO_ROOT
    / "experiments/tpd_clean_v6_checkpoint_metric_compatibility_source_lock.json"
)
FROZEN_EVALUATOR = REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
GENERIC_BASE_EVALUATOR = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
SELECTION_METRIC_KEYS = ("pd", "fa", "tiny_pd", "miou", "val_loss")
NON_STRICT_NUMERIC_DELTA_LIMITS = {
    "miou": 0.0,
    "val_loss": 1.0e-7,
}
VAL_LOSS_NORMALIZATION_SCHEMA = (
    "sctransnet_tpd_clean_v6_threshold_invariant_val_loss_normalization_v1"
)
FORMAL_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FORMAL_INFERENCE_ENVIRONMENT_KEYS = (
    "pythonhashseed",
    "cublas_workspace_config",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "float32_matmul_precision",
    "torch_num_threads",
    "torch_num_interop_threads",
    "cuda_visible_devices",
    "device_uuid",
)
REQUIRED_AUDIT_ONLY_FIELDS = (
    "false_objects_per_image",
    "matched_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "target_count",
    "tiny_target_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
EXPECTED_LOCK_BINDINGS = {
    "training_source_lock": (
        REPO_ROOT / "experiments/tpd_clean_v6_exact_source_lock.json",
        "2de1a8f75deb321b5aec4cf5dfa6bc16df8443e858e1d48a3ab6bea34de526d2",
    ),
    "postprocess_source_lock": (
        REPO_ROOT / "experiments/tpd_clean_v6_postprocess_source_lock.json",
        "3cfbfda891d823c5b97d2d1a2364790c823fac9a548bbf0987444979619bd827",
    ),
    "supplemental_acceptance_source_lock": (
        REPO_ROOT
        / "experiments/tpd_clean_v6_supplemental_acceptance_source_lock.json",
        "dcaf2f1b32cff5096511ba090e3149327deea1f32f2a51d4d866bb0d0cf32696",
    ),
}
COMPATIBILITY_SOURCE_RELATIVES = frozenset(
    {
        "experiments/evaluate_tpd_clean_v6_pd_fa_checkpoint_compat.py",
        "experiments/run_tpd_clean_v6_formal800_checkpoint_compat_sweeps.py",
        "experiments/validate_tpd_clean_v6_checkpoint_compatibility.py",
        "experiments/accept_tpd_clean_v6_formal800_checkpoint_compat_results.py",
        "experiments/freeze_tpd_clean_v6_checkpoint_metric_compatibility_source_lock.py",
        "tests/test_evaluate_tpd_clean_v6_pd_fa_checkpoint_compat.py",
        "tests/test_run_tpd_clean_v6_formal800_checkpoint_compat_sweeps.py",
        "tests/test_validate_tpd_clean_v6_checkpoint_compatibility.py",
        "tests/test_freeze_tpd_clean_v6_checkpoint_metric_compatibility_source_lock.py",
    }
)

_FROZEN_AUDIT = base.audit_fixed_threshold_checkpoint
_FROZEN_WRITE_OUTPUT = base.write_output_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(f"path lies outside repository: {path}") from exc


def _require_regular(path: Path, label: str) -> None:
    if not Path(path).is_file() or Path(path).is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_regular(path, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        base.json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_compatibility_source_lock(
    path: Path = DEFAULT_COMPATIBILITY_SOURCE_LOCK,
) -> tuple[dict[str, Any], str]:
    """Validate the independent adapter lock and every bound source."""

    path = Path(path).resolve()
    payload = _load_json_object(path, "compatibility source lock")
    if payload.get("schema") != SOURCE_LOCK_SCHEMA:
        raise ValueError("compatibility source-lock schema differs")
    if payload.get("candidate_root") != _relative(summary.DEFAULT_CANDIDATE_ROOT):
        raise ValueError("compatibility source-lock candidate root differs")

    bindings = payload.get("frozen_lock_sha256")
    expected_binding_keys = set(EXPECTED_LOCK_BINDINGS)
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_keys:
        raise ValueError("compatibility source-lock frozen lock set differs")
    binding_paths = payload.get("frozen_lock_paths")
    if not isinstance(binding_paths, dict) or set(binding_paths) != expected_binding_keys:
        raise ValueError("compatibility source-lock frozen path set differs")
    for name, (bound_path, expected_sha) in EXPECTED_LOCK_BINDINGS.items():
        _require_regular(bound_path, name)
        actual_sha = sha256_file(bound_path)
        if actual_sha != expected_sha or bindings.get(name) != actual_sha:
            raise ValueError(f"compatibility source-lock binding differs: {name}")
        if binding_paths.get(name) != _relative(bound_path):
            raise ValueError(f"compatibility source-lock path differs: {name}")

    evaluator_sources = payload.get("base_evaluator_sha256")
    expected_evaluator_paths = {
        _relative(FROZEN_EVALUATOR): sha256_file(FROZEN_EVALUATOR),
        _relative(GENERIC_BASE_EVALUATOR): sha256_file(GENERIC_BASE_EVALUATOR),
    }
    if evaluator_sources != expected_evaluator_paths:
        raise ValueError("compatibility source-lock base evaluator binding differs")
    training_lock = _load_json_object(
        EXPECTED_LOCK_BINDINGS["training_source_lock"][0],
        "training source lock",
    )
    frozen_digest = training_lock.get("source_sha256", {}).get(
        _relative(FROZEN_EVALUATOR)
    )
    generic_digest = training_lock.get("source_sha256", {}).get(
        _relative(GENERIC_BASE_EVALUATOR)
    )
    if (
        frozen_digest != expected_evaluator_paths[_relative(FROZEN_EVALUATOR)]
        or generic_digest
        != expected_evaluator_paths[_relative(GENERIC_BASE_EVALUATOR)]
    ):
        raise ValueError("training lock does not bind the current base evaluators")

    sources = payload.get("source_sha256")
    if (
        not isinstance(sources, dict)
        or set(sources) != set(COMPATIBILITY_SOURCE_RELATIVES)
        or payload.get("source_count") != len(COMPATIBILITY_SOURCE_RELATIVES)
    ):
        raise ValueError("compatibility source-lock source set differs")
    for relative, expected_sha in sources.items():
        source = REPO_ROOT / relative
        _require_regular(source, f"compatibility source {relative}")
        if sha256_file(source) != expected_sha:
            raise ValueError(f"compatibility source differs: {relative}")

    policy = payload.get("policy")
    required_true = {
        "audit_supplement_is_in_memory_only",
        "checkpoint_rewrite_forbidden",
        "metrics_rewrite_forbidden",
        "sweep_overwrite_forbidden",
        "original_checkpoint_validation_metrics_preserved",
        "sweep_task_metric_points_preserved",
        "sweep_val_loss_normalized_to_checkpoint",
        "raw_fixed_audit_preserved_before_normalization",
        "formal_inference_replays_training_environment",
        "base_evaluator_artifact_digest_preserved",
        "old_acceptance_runs_before_compatibility_acceptance",
        "direct_wrapper_requires_frozen_arguments",
        "formal_runner_requires_cuda",
        "shared_postprocess_lock_required",
    }
    if (
        not isinstance(policy, dict)
        or any(policy.get(key) is not True for key in required_true)
        or policy.get("non_strict_numeric_delta_limits")
        != NON_STRICT_NUMERIC_DELTA_LIMITS
    ):
        raise ValueError("compatibility source-lock policy differs")
    return payload, sha256_file(path)


def _parse_runtime_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth.tar")
    parser.add_argument("--expected-epochs", type=int, default=None)
    args, _ = parser.parse_known_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    if args.expected_epochs is not None and args.expected_epochs < 1:
        raise ValueError("--expected-epochs must be >= 1")
    return args


def validate_formal_invocation(
    args: argparse.Namespace,
    actual_argv: Sequence[str],
) -> None:
    """Require the fixed runner contract before loading a checkpoint."""

    if args.overwrite is not False or "--overwrite" in actual_argv:
        raise ValueError("formal compatibility sweep overwrite is forbidden")
    for required in (
        "--run-dir",
        "--checkpoint",
        "--device",
        "--expected-epochs",
    ):
        if list(actual_argv).count(required) != 1:
            raise ValueError(
                f"formal compatibility invocation requires one explicit {required}"
            )
    if args.device != "cuda:0":
        raise ValueError("formal compatibility sweep requires device cuda:0")
    if args.expected_epochs != summary.EXPECTED_EPOCHS:
        raise ValueError(
            f"formal compatibility sweep requires "
            f"--expected-epochs={summary.EXPECTED_EPOCHS}"
        )
    if args.checkpoint not in {
        spec["checkpoint"] for spec in summary.ROLE_SPECS.values()
    }:
        raise ValueError("formal compatibility checkpoint name differs")
    expected_run_dirs = {
        (
            summary.DEFAULT_CANDIDATE_ROOT
            / summary.DATASET
            / variant
            / f"seed_{seed}_{summary.RUN_TAG}"
        ).resolve()
        for variant in summary.VARIANTS
        for seed in summary.SEEDS
    }
    if Path(args.run_dir).resolve() not in expected_run_dirs:
        raise ValueError("formal compatibility run directory differs")

    expected = strict.EXPECTED_THRESHOLD_CONFIGURATION
    scalar_fields = (
        "threshold_min",
        "threshold_max",
        "threshold_step",
        "tail_logit_step",
    )
    for key in scalar_fields:
        if float(getattr(args, key)) != float(expected[key]):
            raise ValueError(f"formal threshold argument differs: {key}")
    for key in ("extra_thresholds", "fa_budgets"):
        if [float(value) for value in getattr(args, key)] != [
            float(value) for value in expected[key]
        ]:
            raise ValueError(f"formal threshold argument differs: {key}")
    if args.match_radius is not None or args.tiny_area is not None:
        raise ValueError(
            "formal match-radius/tiny-area must come from the frozen protocol"
        )


def _numeric_mapping(value: Any, label: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    ready: dict[str, int | float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{label} contains a non-finite numeric field: {key!r}")
        ready[key] = item
    return ready


def _role_contract(checkpoint_name: str) -> dict[str, str]:
    if checkpoint_name == "best.pth.tar":
        return {
            "role": "best_validation_pd_primary",
            "summary_epoch": "best_pd_epoch",
            "summary_metrics": "best_pd_validation_metrics",
            "event_flag": "new_best_pd",
        }
    if checkpoint_name == "best_miou.pth.tar":
        return {
            "role": "best_validation_miou_secondary",
            "summary_epoch": "best_miou_epoch",
            "summary_metrics": "best_miou_validation_metrics",
            "event_flag": "new_best_miou",
        }
    raise ValueError(
        "Only best.pth.tar or best_miou.pth.tar may use the compatibility wrapper"
    )


def build_compatibility_context(
    run_dir: Path,
    checkpoint_name: str,
    expected_epochs: int | None,
    actual_argv: Sequence[str],
    *,
    source_lock_path: Path = DEFAULT_COMPATIBILITY_SOURCE_LOCK,
    source_lock_validator: Callable[
        [Path], tuple[dict[str, Any], str]
    ] = validate_compatibility_source_lock,
) -> dict[str, Any]:
    """Resolve and verify the authoritative checkpoint-epoch audit fields."""

    source_lock_path = Path(source_lock_path).resolve()
    source_lock_payload, source_lock_sha = source_lock_validator(source_lock_path)
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    checkpoint_path = (run_dir / checkpoint_name).resolve()
    if checkpoint_path.parent != run_dir:
        raise ValueError("--checkpoint must name a file directly inside --run-dir")
    _require_regular(checkpoint_path, "checkpoint")
    metrics_path = run_dir / "metrics.jsonl"
    protocol_path = run_dir / "protocol.json"
    summary_path = run_dir / "summary.json"
    protocol = _load_json_object(protocol_path, "protocol.json")
    run_summary = _load_json_object(summary_path, "summary.json")
    if run_summary.get("status") != "complete":
        raise ValueError("summary.json is not complete")
    protocol_arguments = protocol.get("arguments")
    if not isinstance(protocol_arguments, dict):
        raise ValueError("protocol.json is missing arguments")
    protocol_epochs = protocol_arguments.get("epochs")
    if type(protocol_epochs) is not int or protocol_epochs < 1:
        raise ValueError("protocol.json has an invalid epoch count")
    resolved_epochs = (
        protocol_epochs if expected_epochs is None else int(expected_epochs)
    )
    if resolved_epochs != protocol_epochs:
        raise ValueError(
            f"Protocol epochs={protocol_epochs} does not match "
            f"--expected-epochs={resolved_epochs}"
        )
    metric_events = base.load_complete_metrics(metrics_path, resolved_epochs)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    checkpoint_seed = checkpoint.get("seed")
    if type(checkpoint_seed) is not int:
        raise ValueError("checkpoint seed is invalid")
    if (
        run_summary.get("seed") != checkpoint_seed
        or protocol_arguments.get("seed") != checkpoint_seed
    ):
        raise ValueError("checkpoint seed differs from protocol or summary")
    training_contract = (
        protocol.get("run_identity", {}).get("training_contract", {})
        if isinstance(protocol.get("run_identity"), Mapping)
        else {}
    )
    training_environment = (
        training_contract.get("environment")
        if isinstance(training_contract, Mapping)
        else None
    )
    if not isinstance(training_environment, Mapping):
        raise ValueError("protocol training environment is missing")
    expected_training_environment = {
        key: training_environment.get(key)
        for key in FORMAL_INFERENCE_ENVIRONMENT_KEYS
    }
    required_training_environment = {
        "pythonhashseed": str(checkpoint_seed),
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "torch_num_threads": 1,
    }
    for key, expected in required_training_environment.items():
        if expected_training_environment.get(key) != expected:
            raise ValueError(f"protocol training environment differs: {key}")
    if not isinstance(
        expected_training_environment.get("torch_num_interop_threads"), int
    ):
        raise ValueError("protocol interop thread count is invalid")
    for key in ("cuda_visible_devices", "device_uuid"):
        value = expected_training_environment.get(key)
        if not isinstance(value, str) or not value.startswith("GPU-"):
            raise ValueError(f"protocol GPU environment differs: {key}")
    if (
        expected_training_environment["cuda_visible_devices"]
        != expected_training_environment["device_uuid"]
    ):
        raise ValueError("protocol visible GPU differs from device UUID")
    checkpoint_epoch = checkpoint.get("epoch")
    if (
        type(checkpoint_epoch) is not int
        or checkpoint_epoch < 1
        or checkpoint_epoch > resolved_epochs
    ):
        raise ValueError(f"invalid checkpoint epoch: {checkpoint_epoch!r}")
    contract = _role_contract(checkpoint_path.name)
    if checkpoint.get("checkpoint_role") != contract["role"]:
        raise ValueError(
            f"{checkpoint_path.name} has invalid checkpoint_role="
            f"{checkpoint.get('checkpoint_role')!r}"
        )
    if run_summary.get(contract["summary_epoch"]) != checkpoint_epoch:
        raise ValueError("checkpoint epoch does not match the selected summary epoch")
    event = metric_events[checkpoint_epoch - 1]
    if event.get(contract["event_flag"]) is not True:
        raise ValueError(
            f"checkpoint epoch is not marked {contract['event_flag']} in metrics"
        )

    checkpoint_metrics = _numeric_mapping(
        checkpoint.get("validation_metrics"),
        "checkpoint.validation_metrics",
    )
    missing_selection = [
        key for key in SELECTION_METRIC_KEYS if key not in checkpoint_metrics
    ]
    if missing_selection:
        raise ValueError(
            f"checkpoint is missing selection metrics: {missing_selection}"
        )
    selected_summary_metrics = _numeric_mapping(
        run_summary.get(contract["summary_metrics"]),
        f"summary.{contract['summary_metrics']}",
    )
    for key in SELECTION_METRIC_KEYS:
        values = {
            "checkpoint": checkpoint_metrics.get(key),
            "summary": selected_summary_metrics.get(key),
            "metrics.jsonl": event.get(key),
        }
        if len(set(values.values())) != 1:
            raise ValueError(
                f"selection metric mismatch for {key}: {values}"
            )
    for key, value in checkpoint_metrics.items():
        if event.get(key) != value:
            raise ValueError(
                f"checkpoint metric does not match metrics.jsonl at epoch "
                f"{checkpoint_epoch}: {key}"
            )

    dynamic_count_fields = sorted(
        key for key in event if key.endswith("_count")
    )
    audit_field_names = list(
        dict.fromkeys(
            [
                "false_objects_per_image",
                *dynamic_count_fields,
            ]
        )
    )
    missing_required = [
        key for key in REQUIRED_AUDIT_ONLY_FIELDS if key not in audit_field_names
    ]
    if missing_required:
        raise ValueError(
            f"metrics.jsonl epoch {checkpoint_epoch} lacks audit fields: "
            f"{missing_required}"
        )
    audit_fields: dict[str, int | float] = {}
    field_sources: dict[str, dict[str, Any]] = {}
    supplemented_fields: list[str] = []
    preexisting_fields: list[str] = []
    for key in audit_field_names:
        value = event.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"metrics.jsonl epoch {checkpoint_epoch} has invalid {key}"
            )
        audit_fields[key] = value
        if key in checkpoint_metrics:
            if checkpoint_metrics[key] != value:
                raise ValueError(
                    f"preexisting checkpoint audit metric differs: {key}"
                )
            mode = "preexisting_checkpoint_value_verified_against_metrics_jsonl"
            preexisting_fields.append(key)
        else:
            mode = "supplemented_from_metrics_jsonl_for_audit_only"
            supplemented_fields.append(key)
        field_sources[key] = {
            "source": "metrics.jsonl",
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
            "epoch": checkpoint_epoch,
            "field": key,
            "value": value,
            "mode": mode,
        }

    wrapper_path = Path(__file__).resolve()
    frozen_path = FROZEN_EVALUATOR.resolve()
    generic_path = GENERIC_BASE_EVALUATOR.resolve()
    source_hashes = source_lock_payload.get("source_sha256", {})
    wrapper_sha = sha256_file(wrapper_path)
    if source_hashes.get(_relative(wrapper_path)) != wrapper_sha:
        raise ValueError("runtime wrapper is not bound by the compatibility lock")
    frozen_sha = sha256_file(frozen_path)
    generic_sha = sha256_file(generic_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    metrics_sha = sha256_file(metrics_path)
    actual_runtime_argv = [
        sys.executable,
        str(wrapper_path),
        *list(actual_argv),
    ]
    return {
        "schema": COMPATIBILITY_SCHEMA,
        "actual_wrapper": {
            "path": str(wrapper_path),
            "repo_relative_path": _relative(wrapper_path),
            "sha256": wrapper_sha,
        },
        "compatibility_source_lock": {
            "path": str(source_lock_path),
            "repo_relative_path": _relative(source_lock_path),
            "sha256": source_lock_sha,
        },
        "frozen_v6_evaluator": {
            "path": str(frozen_path),
            "repo_relative_path": _relative(frozen_path),
            "sha256": frozen_sha,
        },
        "generic_base_evaluator": {
            "path": str(generic_path),
            "repo_relative_path": _relative(generic_path),
            "sha256": generic_sha,
        },
        "base_evaluator_sha256": frozen_sha,
        "runtime_compatibility_sha256": wrapper_sha,
        "seed": checkpoint_seed,
        "training_inference_environment": expected_training_environment,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "epoch": checkpoint_epoch,
            "role": checkpoint.get("checkpoint_role"),
        },
        "metrics_log": {
            "path": str(metrics_path.resolve()),
            "sha256": metrics_sha,
            "authoritative_epoch": checkpoint_epoch,
        },
        "original_five_selection_metrics": {
            key: checkpoint_metrics[key] for key in SELECTION_METRIC_KEYS
        },
        "original_checkpoint_validation_metrics": dict(checkpoint_metrics),
        "original_checkpoint_metric_keys": sorted(checkpoint_metrics),
        "audit_only_fields": audit_fields,
        "audit_only_field_sources": field_sources,
        "supplemented_fields": sorted(supplemented_fields),
        "preexisting_audit_fields": sorted(preexisting_fields),
        "actual_runtime_argv": actual_runtime_argv,
        "temporary_audit_copy_only": True,
        "checkpoint_unchanged": True,
        "metrics_log_unchanged": True,
        "checkpoint_validation_metrics_unchanged": True,
    }


def configure_formal_inference_determinism(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the exact inference-relevant environment used during training."""

    seed = context.get("seed")
    expected = context.get("training_inference_environment")
    if type(seed) is not int or not isinstance(expected, Mapping):
        raise ValueError("formal inference context is incomplete")
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError("formal inference PYTHONHASHSEED differs")
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != FORMAL_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError("formal inference CUBLAS workspace differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["device_uuid"]:
        raise RuntimeError("formal inference visible GPU differs from training")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"formal inference thread environment differs: {name}")

    torch.set_num_threads(int(expected["torch_num_threads"]))
    expected_interop = int(expected["torch_num_interop_threads"])
    if torch.get_num_interop_threads() != expected_interop:
        torch.set_num_interop_threads(expected_interop)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("formal inference requires exactly one visible GPU")
    torch.cuda.manual_seed_all(seed)

    attestation = {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if attestation != dict(expected):
        raise RuntimeError("formal inference environment does not match training")
    return attestation


def temporary_checkpoint_metrics_for_audit(
    checkpoint_metrics: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the sole enriched object: a temporary audit-call dictionary."""

    original = context["original_checkpoint_validation_metrics"]
    if dict(checkpoint_metrics) != original:
        raise ValueError(
            "runtime checkpoint metrics differ from the pre-inference context"
        )
    temporary = dict(checkpoint_metrics)
    for key in context["supplemented_fields"]:
        if key in temporary:
            raise ValueError(f"audit supplement would replace checkpoint field: {key}")
        temporary[key] = context["audit_only_fields"][key]
    return temporary


def normalize_threshold_invariant_val_loss(
    payload: MutableMapping[str, Any],
    checkpoint_metrics: Mapping[str, Any],
    fixed_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind threshold-independent loss to its authoritative checkpoint value."""

    validate_non_strict_numeric_deltas(fixed_audit)
    checkpoint_value = checkpoint_metrics.get("val_loss")
    if (
        not isinstance(checkpoint_value, (int, float))
        or isinstance(checkpoint_value, bool)
        or not math.isfinite(float(checkpoint_value))
    ):
        raise ValueError("checkpoint val_loss is invalid")

    points = payload.get("points")
    fixed = payload.get("fixed_threshold_0_5")
    budgets = payload.get("best_points_under_fa_budget")
    if not isinstance(points, list) or not points:
        raise ValueError("sweep points are missing before val_loss normalization")
    if not isinstance(fixed, MutableMapping):
        raise ValueError("fixed point is not mutable before val_loss normalization")
    if not isinstance(budgets, MutableMapping) or not budgets:
        raise ValueError("budget points are missing before val_loss normalization")
    if any(not isinstance(point, MutableMapping) for point in points):
        raise ValueError("sweep point is not mutable before val_loss normalization")
    if any(not isinstance(point, MutableMapping) for point in budgets.values()):
        raise ValueError("budget point is not mutable before val_loss normalization")

    raw_values = [point.get("val_loss") for point in points]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in raw_values
    ):
        raise ValueError("raw sweep val_loss is invalid")
    raw_value = raw_values[0]
    if any(value != raw_value for value in raw_values[1:]):
        raise ValueError("raw sweep val_loss is not threshold-invariant")
    if fixed.get("val_loss") != raw_value or any(
        point.get("val_loss") != raw_value for point in budgets.values()
    ):
        raise ValueError("raw copied operating-point val_loss differs")

    deltas = fixed_audit[
        "non_strict_numeric_deltas_sweep_minus_checkpoint"
    ]
    expected_delta = float(raw_value) - float(checkpoint_value)
    if float(deltas["val_loss"]) != expected_delta:
        raise ValueError("raw val_loss delta differs from fixed audit")

    raw_points_sha256 = _canonical_sha256(points)
    raw_fixed_sha256 = _canonical_sha256(fixed)
    for point in points:
        point["val_loss"] = checkpoint_value
    fixed["val_loss"] = checkpoint_value
    for point in budgets.values():
        point["val_loss"] = checkpoint_value

    return {
        "schema": VAL_LOSS_NORMALIZATION_SCHEMA,
        "field": "val_loss",
        "reason": "threshold_invariant_recomputed_loss_process_roundoff",
        "authoritative_source": "checkpoint.validation_metrics.val_loss",
        "raw_recomputed_value": raw_value,
        "normalized_checkpoint_value": checkpoint_value,
        "raw_minus_checkpoint_delta": expected_delta,
        "absolute_delta": abs(expected_delta),
        "absolute_delta_limit": NON_STRICT_NUMERIC_DELTA_LIMITS["val_loss"],
        "point_count": len(points),
        "budget_point_count": len(budgets),
        "raw_points_sha256": raw_points_sha256,
        "raw_fixed_threshold_0_5_sha256": raw_fixed_sha256,
        "normalized_collections": [
            "points",
            "fixed_threshold_0_5",
            "best_points_under_fa_budget",
        ],
    }


def enrich_output_payload(
    payload: MutableMapping[str, Any],
    context: Mapping[str, Any],
    *,
    source_lock_validator: Callable[
        [Path], tuple[dict[str, Any], str]
    ] = validate_compatibility_source_lock,
) -> None:
    """Attach provenance and normalize only threshold-independent val_loss."""

    points_before = copy.deepcopy(payload.get("points"))
    checkpoint_metrics_before = copy.deepcopy(
        payload.get("checkpoint_validation_metrics")
    )
    if checkpoint_metrics_before != context[
        "original_checkpoint_validation_metrics"
    ]:
        raise ValueError("output checkpoint_validation_metrics were altered")
    checkpoint_record = context["checkpoint"]
    if (
        str(Path(str(payload.get("checkpoint"))).resolve())
        != checkpoint_record["path"]
        or payload.get("checkpoint_sha256") != checkpoint_record["sha256"]
        or payload.get("checkpoint_epoch") != checkpoint_record["epoch"]
        or payload.get("checkpoint_role") != checkpoint_record["role"]
    ):
        raise ValueError("output checkpoint identity differs from compatibility context")
    if sha256_file(Path(checkpoint_record["path"])) != checkpoint_record["sha256"]:
        raise ValueError("checkpoint changed during compatibility sweep")
    metrics_record = context["metrics_log"]
    if sha256_file(Path(metrics_record["path"])) != metrics_record["sha256"]:
        raise ValueError("metrics.jsonl changed during compatibility sweep")
    _, current_lock_sha = source_lock_validator(
        Path(context["compatibility_source_lock"]["path"])
    )
    if current_lock_sha != context["compatibility_source_lock"]["sha256"]:
        raise ValueError("compatibility source lock changed during sweep")

    fixed_audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    if not isinstance(fixed_audit, MutableMapping):
        raise ValueError("fixed-threshold checkpoint audit is missing")
    fixed_point = payload.get("fixed_threshold_0_5")
    if not isinstance(fixed_point, Mapping):
        raise ValueError("fixed threshold point is missing")
    for key, expected in context["audit_only_fields"].items():
        matches = fixed_audit.get("exact_matches", {}).get(key)
        if (
            fixed_point.get(key) != expected
            or not isinstance(matches, Mapping)
            or matches.get("checkpoint") != expected
            or matches.get("sweep_0_5") != expected
        ):
            raise ValueError(
                f"fixed-threshold audit did not reproduce compatibility field {key}"
            )

    normalization = normalize_threshold_invariant_val_loss(
        payload,
        checkpoint_metrics_before,
        fixed_audit,
    )
    raw_fixed_audit = copy.deepcopy(dict(fixed_audit))
    normalized_fixed_audit = _FROZEN_AUDIT(
        dict(payload["fixed_threshold_0_5"]),
        temporary_checkpoint_metrics_for_audit(
            checkpoint_metrics_before,
            context,
        ),
    )
    fixed_audit.clear()
    fixed_audit.update(normalized_fixed_audit)
    validate_non_strict_numeric_deltas(fixed_audit)
    points_after = copy.deepcopy(payload.get("points"))
    if not isinstance(points_before, list) or not isinstance(points_after, list):
        raise ValueError("sweep points are missing around val_loss normalization")
    raw_task_points = [
        {key: value for key, value in point.items() if key != "val_loss"}
        for point in points_before
    ]
    normalized_task_points = [
        {key: value for key, value in point.items() if key != "val_loss"}
        for point in points_after
    ]
    if raw_task_points != normalized_task_points:
        raise RuntimeError("val_loss normalization changed a task metric")

    threshold_provenance = payload.get("threshold_provenance")
    audit = payload.get("audit")
    if not isinstance(threshold_provenance, MutableMapping):
        raise ValueError("threshold provenance is missing")
    if not isinstance(audit, MutableMapping):
        raise ValueError("sweep audit is missing")
    final_record = copy.deepcopy(dict(context))
    final_record.update(
        {
            "points_sha256": _canonical_sha256(points_after),
            "raw_recomputed_points_sha256": _canonical_sha256(points_before),
            "checkpoint_validation_metrics_sha256": _canonical_sha256(
                checkpoint_metrics_before
            ),
            "threshold_invariant_val_loss_normalization": normalization,
            "raw_fixed_threshold_checkpoint_audit": raw_fixed_audit,
            "raw_fixed_threshold_checkpoint_audit_sha256": _canonical_sha256(
                raw_fixed_audit
            ),
            "normalized_fixed_threshold_checkpoint_audit_sha256": (
                _canonical_sha256(normalized_fixed_audit)
            ),
            "task_metric_points_unchanged": True,
            "points_unchanged": points_after == points_before,
            "points_changed_only_by_threshold_invariant_val_loss": True,
        }
    )
    threshold_provenance[COMPATIBILITY_KEY] = copy.deepcopy(final_record)
    fixed_audit[COMPATIBILITY_KEY] = copy.deepcopy(final_record)
    audit[COMPATIBILITY_KEY] = copy.deepcopy(final_record)

    if payload.get("points") != points_after:
        raise RuntimeError("compatibility provenance changed normalized sweep points")
    if payload.get("checkpoint_validation_metrics") != checkpoint_metrics_before:
        raise RuntimeError(
            "compatibility provenance changed checkpoint validation metrics"
        )
    if not (
        threshold_provenance[COMPATIBILITY_KEY]
        == fixed_audit[COMPATIBILITY_KEY]
        == audit[COMPATIBILITY_KEY]
    ):
        raise RuntimeError("compatibility provenance copies differ")


def validate_non_strict_numeric_deltas(fixed_audit: Mapping[str, Any]) -> None:
    """Bound the two frozen evaluator fields that are intentionally non-exact."""

    deltas = fixed_audit.get(
        "non_strict_numeric_deltas_sweep_minus_checkpoint"
    )
    if not isinstance(deltas, Mapping) or set(deltas) != set(
        NON_STRICT_NUMERIC_DELTA_LIMITS
    ):
        raise ValueError("fixed-threshold non-strict numeric delta set differs")
    absolute_deltas: list[float] = []
    for key, limit in NON_STRICT_NUMERIC_DELTA_LIMITS.items():
        value = deltas.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"fixed-threshold non-strict numeric delta is invalid: {key}"
            )
        absolute = abs(float(value))
        if absolute > limit:
            raise ValueError(
                f"fixed-threshold non-strict numeric delta exceeds bound: {key}"
            )
        absolute_deltas.append(absolute)
    observed_max = fixed_audit.get("max_abs_non_strict_numeric_delta")
    expected_max = max(absolute_deltas, default=0.0)
    if (
        not isinstance(observed_max, (int, float))
        or isinstance(observed_max, bool)
        or not math.isfinite(float(observed_max))
        or float(observed_max) != expected_max
    ):
        raise ValueError("fixed-threshold maximum numeric delta differs")


def validate_payload_before_write(payload: Mapping[str, Any]) -> None:
    """Reject a non-canonical sweep before the exclusive output is created."""

    fixed_audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    if not isinstance(fixed_audit, Mapping):
        raise ValueError("fixed-threshold checkpoint audit is missing")
    validate_non_strict_numeric_deltas(fixed_audit)
    strict.validate_sweep_payload(
        payload,
        "postfreeze checkpoint-metric compatibility pre-write",
    )


def main() -> None:
    actual_argv = list(sys.argv[1:])
    runtime_args = base.parse_args()
    validate_formal_invocation(runtime_args, actual_argv)
    context = build_compatibility_context(
        runtime_args.run_dir,
        runtime_args.checkpoint,
        runtime_args.expected_epochs,
        actual_argv,
    )
    context["formal_inference_determinism"] = (
        configure_formal_inference_determinism(context)
    )

    original_adaptive = base.adaptive_thresholds
    original_audit = base.audit_fixed_threshold_checkpoint
    original_writer = base.write_output_json
    original_builder = base.build_model
    original_file = base.__file__

    def compatible_audit(
        fixed_half: dict[str, Any],
        checkpoint_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        temporary = temporary_checkpoint_metrics_for_audit(
            checkpoint_metrics, context
        )
        return _FROZEN_AUDIT(fixed_half, temporary)

    def compatible_writer(
        path: Path,
        payload: dict[str, Any],
        overwrite: bool,
    ) -> None:
        enrich_output_payload(payload, context)
        validate_payload_before_write(payload)
        _FROZEN_WRITE_OUTPUT(path, payload, overwrite)

    try:
        base.adaptive_thresholds = frozen.adaptive_thresholds_closed_interval
        base.audit_fixed_threshold_checkpoint = compatible_audit
        base.write_output_json = compatible_writer
        base.build_model = frozen.build_clean_v6_model
        # Preserve the evaluator digest expected by the frozen summarizer.
        base.__file__ = frozen.__file__
        base.main()
    finally:
        base.adaptive_thresholds = original_adaptive
        base.audit_fixed_threshold_checkpoint = original_audit
        base.write_output_json = original_writer
        base.build_model = original_builder
        base.__file__ = original_file


__all__ = [
    "COMPATIBILITY_KEY",
    "COMPATIBILITY_SCHEMA",
    "COMPATIBILITY_SOURCE_RELATIVES",
    "DEFAULT_COMPATIBILITY_SOURCE_LOCK",
    "EXPECTED_LOCK_BINDINGS",
    "FORMAL_CUBLAS_WORKSPACE_CONFIG",
    "FORMAL_INFERENCE_ENVIRONMENT_KEYS",
    "FROZEN_EVALUATOR",
    "GENERIC_BASE_EVALUATOR",
    "NON_STRICT_NUMERIC_DELTA_LIMITS",
    "REQUIRED_AUDIT_ONLY_FIELDS",
    "SELECTION_METRIC_KEYS",
    "SOURCE_LOCK_SCHEMA",
    "VAL_LOSS_NORMALIZATION_SCHEMA",
    "build_compatibility_context",
    "configure_formal_inference_determinism",
    "enrich_output_payload",
    "main",
    "normalize_threshold_invariant_val_loss",
    "sha256_file",
    "temporary_checkpoint_metrics_for_audit",
    "validate_formal_invocation",
    "validate_non_strict_numeric_deltas",
    "validate_payload_before_write",
    "validate_compatibility_source_lock",
]


if __name__ == "__main__":
    main()
