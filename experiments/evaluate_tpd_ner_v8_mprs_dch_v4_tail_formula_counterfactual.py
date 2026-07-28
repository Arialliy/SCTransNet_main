#!/usr/bin/env python3
"""Read-only V3-checkpoint evaluation of three V4 NER DC-support formulas.

Each invocation consumes exactly one immutable formal V3 checkpoint, strictly
loads its unchanged state into three freshly constructed V4 models, and
evaluates the modes in this fixed order:

``legacy_global -> direct_tail -> complement_tail``.

The legacy result must be canonically identical to the corresponding frozen
V3 sweep before either alternative formula is evaluated.  The evaluator writes
one new, non-overwritable JSON artifact per source checkpoint and never writes
a derived checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as metric_core  # noqa: E402
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout as frozen_diagnostic_core,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v3_pd_fa as formal_v3_evaluator,
)
from experiments import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as frozen_v3_spec,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v3_exact as exact_v3,
)
from experiments import (  # noqa: E402
    train_tpd_clean_v6_exact as gpu_identity_core,
)
from experiments import (  # noqa: E402
    train_tpd_clean_v8_mprs_dch as parent_builder_source,
)
from experiments import (  # noqa: E402
    evaluate_tpd_clean_v8_mprs_dch_pd_fa as determinism_core,
)
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    configure_v8_inference,
)
from model import SCTransNet as sctransnet_source  # noqa: E402
from model import tpd_clean_v8_mprs_dch as tokenizer_source  # noqa: E402
from model import tpd_ner_v8_mprs_dch as v1_ner_source  # noqa: E402
from model import tpd_ner_v8_mprs_dch_v2 as v2_ner_source  # noqa: E402
from model import tpd_ner_v8_mprs_dch_v3 as v3_ner_source  # noqa: E402
from model import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v4_tail_aware as v4_model_source,
)


EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_formula_"
    "counterfactual_evaluation_v1"
)
MODE_EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_formula_"
    "counterfactual_mode_v1"
)
PLAN_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_formula_"
    "counterfactual_plan_v1"
)
ARTIFACT_KIND = "v4_tail_formula_zero_training_counterfactual"
EVALUATOR_PATH = Path(__file__).resolve()
DATASET = frozen_v3_spec.DATASET
VARIANT = frozen_v3_spec.VARIANT
TRAINING_SEED = frozen_v3_spec.TRAINING_SEED
SPLIT_SEED = frozen_v3_spec.SPLIT_SEED
EXPECTED_EPOCHS = frozen_v3_spec.EXPECTED_EPOCHS
VALIDATION_COUNT = frozen_v3_spec.VALIDATION_COUNT
CHECKPOINT_ROLES = dict(frozen_v3_spec.CHECKPOINT_ROLES)
CHECKPOINTS = tuple(CHECKPOINT_ROLES)
FORMULA_MODES = (
    v4_model_source.TailDCSupportMode.LEGACY_GLOBAL.value,
    v4_model_source.TailDCSupportMode.DIRECT_TAIL.value,
    v4_model_source.TailDCSupportMode.COMPLEMENT_TAIL.value,
)
FORMULA_EXPRESSIONS = {
    "legacy_global": "d",
    "direct_tail": "d*P",
    "complement_tail": "d*(1-P)",
}
FA_BUDGETS = tuple(frozen_v3_spec.FA_BUDGETS)
BUDGET_KEYS = tuple(frozen_v3_spec.BUDGET_KEYS)
FIXED_THRESHOLD = 0.5
FORMAL_RUN_DIR = frozen_v3_spec.FORMAL_RUN_DIR.resolve()
CANONICAL_SWEEP_FILENAMES = {
    "best.pth.tar": "pd_fa_sweep_best.pth.json",
    "best_miou.pth.tar": "pd_fa_sweep_best_miou.pth.json",
}
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual_v1"
)
OUTPUT_FILENAMES = {
    "best.pth.tar": "tail_formula_counterfactual_best.pth.json",
    "best_miou.pth.tar": (
        "tail_formula_counterfactual_best_miou.pth.json"
    ),
}
CUDA_DEVICE_ORDER = frozen_v3_spec.CUDA_DEVICE_ORDER
CUBLAS_WORKSPACE_CONFIG_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_CONFIG_VALUE = frozen_v3_spec.CUBLAS_WORKSPACE_CONFIG
PYTHONHASHSEED_ENV = "PYTHONHASHSEED"
PYTHONHASHSEED_VALUE = str(TRAINING_SEED)
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"
PHYSICAL_GPU_INDEX_ENV = (
    "TPD_NER_V8_MPRS_DCH_V4_TAIL_FORMULA_PHYSICAL_GPU_INDEX"
)
PHYSICAL_GPU_UUID_ENV = (
    "TPD_NER_V8_MPRS_DCH_V4_TAIL_FORMULA_PHYSICAL_GPU_UUID"
)
CHECKPOINT_GPU_LANES = copy.deepcopy(frozen_v3_spec.CHECKPOINT_GPU_LANES)
CANONICAL_SWEEP_FIELDS = (
    "validation_count",
    "threshold_configuration",
    "threshold_provenance",
    "fixed_threshold_0_5",
    "best_points_under_fa_budget",
    "points",
    "final_metric_coverage",
)
METRIC_CONTRACT = {
    "source": "frozen_v3_pd_fa_contract",
    "prediction_collector": (
        "experiments.evaluate_pd_fa_sweep.collect_predictions"
    ),
    "sweep_core": (
        "experiments.evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout."
        "sweep_predictions"
    ),
    "closed_probability_interval": True,
    "fixed_threshold": FIXED_THRESHOLD,
    "fa_budgets": list(FA_BUDGETS),
    "final_metric_coverage_schema": (
        formal_v3_evaluator.FINAL_METRIC_COVERAGE_SCHEMA
    ),
}


def _require(condition: bool, message: str) -> None:
    """Optimization-independent invariant check."""

    if not condition:
        raise ValueError(message)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _sha256_file(path: Path) -> str:
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {value}")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise FileNotFoundError(value)
    payload = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {value}")
    return payload


def _canonical_json_bytes(value: Any) -> bytes:
    return frozen_v3_spec.canonical_json_bytes(
        metric_core.json_ready(value)
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def output_path(checkpoint: str) -> Path:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    return (
        DEFAULT_OUTPUT_ROOT
        / DATASET
        / OUTPUT_FILENAMES[checkpoint]
    )


def canonical_sweep_path(checkpoint: str) -> Path:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    return (
        FORMAL_RUN_DIR / CANONICAL_SWEEP_FILENAMES[checkpoint]
    ).resolve()


def _canonical_threshold_configuration() -> dict[str, Any]:
    return {
        "threshold_min": frozen_diagnostic_core.THRESHOLD_MIN,
        "threshold_max": frozen_diagnostic_core.THRESHOLD_MAX,
        "threshold_step": frozen_diagnostic_core.THRESHOLD_STEP,
        "extra_thresholds": list(frozen_v3_spec.EXTRA_THRESHOLDS),
        "tail_logit_step": frozen_diagnostic_core.TAIL_LOGIT_STEP,
        "fa_budgets": list(FA_BUDGETS),
    }


def canonical_sweep_projection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return every prediction-derived field used for exact equivalence."""

    if not isinstance(payload, Mapping):
        raise TypeError("sweep payload must be a mapping")
    missing = [name for name in CANONICAL_SWEEP_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"sweep payload lacks canonical fields: {missing}")
    projection = {
        name: copy.deepcopy(payload[name])
        for name in CANONICAL_SWEEP_FIELDS
    }
    _require_equal(
        "canonical validation count",
        projection["validation_count"],
        VALIDATION_COUNT,
    )
    _require_equal(
        "canonical threshold configuration",
        projection["threshold_configuration"],
        _canonical_threshold_configuration(),
    )
    points = projection["points"]
    if not isinstance(points, list) or not points:
        raise ValueError("canonical sweep points must be a non-empty list")
    fixed = projection["fixed_threshold_0_5"]
    if not isinstance(fixed, Mapping):
        raise ValueError("canonical fixed threshold must be an object")
    _require_equal(
        "canonical fixed threshold",
        fixed.get("threshold"),
        FIXED_THRESHOLD,
    )
    budgets = projection["best_points_under_fa_budget"]
    if not isinstance(budgets, Mapping):
        raise ValueError("canonical FA budgets must be an object")
    _require_equal("canonical FA budget keys", set(budgets), set(BUDGET_KEYS))
    formal_v3_evaluator._validate_closed_interval(projection)
    metric_core.assert_finite_numbers(projection, "canonical sweep projection")
    return metric_core.json_ready(projection)


def require_legacy_canonical_exact(
    observed: Mapping[str, Any],
    canonical: Mapping[str, Any],
) -> dict[str, Any]:
    """Require byte-exact canonical equality of all prediction-derived fields."""

    observed_projection = canonical_sweep_projection(observed)
    canonical_projection = canonical_sweep_projection(canonical)
    observed_bytes = _canonical_json_bytes(observed_projection)
    canonical_bytes = _canonical_json_bytes(canonical_projection)
    if observed_bytes != canonical_bytes:
        raise ValueError(
            "legacy_global output is not canonically exact to frozen V3 sweep"
        )
    digest = hashlib.sha256(observed_bytes).hexdigest()
    return {
        "legacy_global_canonical_exact": True,
        "canonical_projection_sha256": digest,
        "observed_projection_sha256": digest,
        "canonical_field_order": list(CANONICAL_SWEEP_FIELDS),
        "canonical_encoding": (
            "utf8_sort_keys_compact_json_newline_allow_nan_false"
        ),
    }


def _source_paths() -> dict[str, Path]:
    return {
        "counterfactual_evaluator": EVALUATOR_PATH,
        "v4_model": Path(v4_model_source.__file__).resolve(),
        "v3_ner_model": Path(v3_ner_source.__file__).resolve(),
        "v2_ner_model": Path(v2_ner_source.__file__).resolve(),
        "v1_ner_model": Path(v1_ner_source.__file__).resolve(),
        "v8_mprs_dch_tokenizer": Path(tokenizer_source.__file__).resolve(),
        "sctransnet_parent": Path(sctransnet_source.__file__).resolve(),
        "formal_v3_evaluator": Path(
            formal_v3_evaluator.__file__
        ).resolve(),
        "shared_metric_core": Path(metric_core.__file__).resolve(),
        "closed_interval_and_sweep_core": Path(
            frozen_diagnostic_core.__file__
        ).resolve(),
        "determinism_core": Path(determinism_core.__file__).resolve(),
        "gpu_identity_core": Path(gpu_identity_core.__file__).resolve(),
        "v8_parent_builder": Path(
            parent_builder_source.__file__
        ).resolve(),
        "v3_exact_builder": Path(exact_v3.__file__).resolve(),
        "v3_fixed_spec": Path(frozen_v3_spec.__file__).resolve(),
    }


def implementation_hashes() -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for name, path in _source_paths().items():
        records[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    return records


def _input_hashes(
    checkpoint_path: Path,
    canonical_path: Path,
) -> dict[str, str]:
    values = {
        "source_checkpoint": checkpoint_path,
        "canonical_v3_sweep": canonical_path,
        "protocol.json": FORMAL_RUN_DIR / "protocol.json",
        "split.json": FORMAL_RUN_DIR / "split.json",
        "summary.json": FORMAL_RUN_DIR / "summary.json",
        "metrics.jsonl": FORMAL_RUN_DIR / "metrics.jsonl",
    }
    return {name: _sha256_file(path) for name, path in values.items()}


def _assert_output_isolated(path: Path) -> None:
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("counterfactual output path must be absolute")
    resolved_target = target.resolve()
    resolved_root = DEFAULT_OUTPUT_ROOT.resolve()
    _require(
        resolved_target.is_relative_to(resolved_root),
        "counterfactual output lies outside its versioned root",
    )
    formal_root = exact_v3.DEFAULT_OUTPUT_ROOT.resolve()
    if (
        resolved_root == formal_root
        or resolved_root.is_relative_to(formal_root)
        or formal_root.is_relative_to(resolved_root)
    ):
        raise ValueError("counterfactual output overlaps formal V3 result root")
    current = target.parent
    while not current.exists():
        if current == current.parent:
            break
        current = current.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"output path contains a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent


def _validate_lane_environment(
    checkpoint: str,
    device_name: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    _require_equal("logical evaluation device", device_name, "cuda:0")
    env = os.environ if environment is None else environment
    lane = CHECKPOINT_GPU_LANES[checkpoint]
    expected_index = int(lane["physical_gpu_index"])
    expected_uuid = str(lane["physical_gpu_uuid"])
    _require_equal(
        "inherited CUBLAS workspace constant",
        CUBLAS_WORKSPACE_CONFIG,
        CUBLAS_WORKSPACE_CONFIG_VALUE,
    )
    expected_environment = {
        "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
        "CUDA_VISIBLE_DEVICES": expected_uuid,
        CUBLAS_WORKSPACE_CONFIG_ENV: CUBLAS_WORKSPACE_CONFIG_VALUE,
        PYTHONHASHSEED_ENV: PYTHONHASHSEED_VALUE,
        PHYSICAL_GPU_INDEX_ENV: str(expected_index),
        PHYSICAL_GPU_UUID_ENV: expected_uuid,
    }
    for name, expected in expected_environment.items():
        _require_equal(name, env.get(name), expected)
    return {
        "logical_device": "cuda:0",
        "physical_gpu_index": expected_index,
        "physical_gpu_uuid": expected_uuid,
        "required_environment": expected_environment,
        "checkpoint": checkpoint,
    }


def _validated_cuda_lane(
    checkpoint: str,
    device_name: str,
) -> dict[str, Any]:
    lane = _validate_lane_environment(checkpoint, device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _require_equal("visible CUDA device count", torch.cuda.device_count(), 1)
    actual_uuid, properties = gpu_identity_core.visible_gpu_identity()
    _require_equal(
        "logical cuda:0 actual GPU UUID",
        actual_uuid,
        lane["physical_gpu_uuid"],
    )
    _require_equal(
        "logical cuda:0 device name",
        str(properties.name),
        EXPECTED_GPU_NAME,
    )
    lane.update(
        {
            "actual_logical_cuda_0_uuid": actual_uuid,
            "device_name": str(properties.name),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_memory_bytes": int(properties.total_memory),
        }
    )
    return lane


def _preflight(checkpoint: str) -> dict[str, Any]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    destination = output_path(checkpoint)
    _assert_output_isolated(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite counterfactual output: {destination}"
        )
    artifact_audit = formal_v3_evaluator.validate_run_artifacts(
        FORMAL_RUN_DIR,
        checkpoint,
    )
    canonical_path = canonical_sweep_path(checkpoint)
    canonical_payload = _load_json(canonical_path)
    formal_v3_evaluator.validate_output_identity(
        canonical_payload,
        artifact_audit=artifact_audit,
    )
    checkpoint_path = (FORMAL_RUN_DIR / checkpoint).resolve()
    _require_equal(
        "preflight checkpoint path",
        checkpoint_path.parent,
        FORMAL_RUN_DIR,
    )
    _require_equal(
        "preflight checkpoint SHA",
        _sha256_file(checkpoint_path),
        artifact_audit["checkpoint_sha256"],
    )
    _require_equal(
        "canonical checkpoint role",
        canonical_payload.get("checkpoint_role"),
        CHECKPOINT_ROLES[checkpoint],
    )
    _require_equal(
        "canonical checkpoint epoch",
        canonical_payload.get("checkpoint_epoch"),
        artifact_audit["checkpoint_epoch"],
    )
    canonical_sweep_projection(canonical_payload)
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "canonical_path": canonical_path,
        "canonical_payload": canonical_payload,
        "artifact_audit": artifact_audit,
        "output_path": destination,
        "implementation_hashes": implementation_hashes(),
        "input_hashes": _input_hashes(checkpoint_path, canonical_path),
    }


def _command_environment(checkpoint: str) -> dict[str, str]:
    lane = CHECKPOINT_GPU_LANES[checkpoint]
    return {
        "CUDA_DEVICE_ORDER": CUDA_DEVICE_ORDER,
        "CUDA_VISIBLE_DEVICES": str(lane["physical_gpu_uuid"]),
        CUBLAS_WORKSPACE_CONFIG_ENV: CUBLAS_WORKSPACE_CONFIG_VALUE,
        PYTHONHASHSEED_ENV: PYTHONHASHSEED_VALUE,
        PHYSICAL_GPU_INDEX_ENV: str(lane["physical_gpu_index"]),
        PHYSICAL_GPU_UUID_ENV: str(lane["physical_gpu_uuid"]),
    }


def _build_plan_from_preflight(
    checkpoint: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    lane = CHECKPOINT_GPU_LANES[checkpoint]
    environment = _command_environment(checkpoint)
    command = [
        *[f"{name}={value}" for name, value in environment.items()],
        sys.executable,
        str(EVALUATOR_PATH),
        "--run",
        "--checkpoint",
        checkpoint,
        "--device",
        "cuda:0",
    ]
    artifact_audit = preflight["artifact_audit"]
    return {
        "schema": PLAN_SCHEMA,
        "status": "ready",
        "artifact_kind": ARTIFACT_KIND,
        "diagnostic_only": True,
        "zero_training": True,
        "official_test_accessed": False,
        "checkpoint": checkpoint,
        "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "source_checkpoint_sha256": artifact_audit["checkpoint_sha256"],
        "canonical_v3_sweep": str(preflight["canonical_path"]),
        "canonical_v3_sweep_sha256": preflight["input_hashes"][
            "canonical_v3_sweep"
        ],
        "formula_modes": list(FORMULA_MODES),
        "formula_expressions": dict(FORMULA_EXPRESSIONS),
        "mode_count": len(FORMULA_MODES),
        "metric_contract": copy.deepcopy(METRIC_CONTRACT),
        "legacy_canonical_exact_required_before_alternatives": True,
        "output": str(preflight["output_path"]),
        "output_overwrite_forbidden": True,
        "derived_checkpoint_written": False,
        "physical_gpu_index": lane["physical_gpu_index"],
        "physical_gpu_uuid": lane["physical_gpu_uuid"],
        "logical_device": "cuda:0",
        "required_environment": environment,
        "implementation_hashes": copy.deepcopy(
            preflight["implementation_hashes"]
        ),
        "input_hashes": copy.deepcopy(preflight["input_hashes"]),
        "command_tokens": command,
    }


def build_plan(checkpoint: str) -> dict[str, Any]:
    return _build_plan_from_preflight(checkpoint, _preflight(checkpoint))


def _load_source_checkpoint(
    checkpoint: str,
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    checkpoint_path = Path(preflight["checkpoint_path"])
    expected_sha = preflight["input_hashes"]["source_checkpoint"]
    _require_equal(
        "checkpoint SHA before load",
        _sha256_file(checkpoint_path),
        expected_sha,
    )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("V3 checkpoint must be an object")
    checkpoint_payload = exact_v3.require_evaluator_checkpoint_payload(
        payload,
        expected_variant=VARIANT,
    )
    state = checkpoint_payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("V3 checkpoint state_dict is missing")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("V3 state_dict must map string keys to tensors")
    state_sha = frozen_diagnostic_core.state_content_sha256(state)
    _require_equal(
        "checkpoint state_dict SHA",
        state_sha,
        checkpoint_payload.get("state_dict_sha256"),
    )
    artifact_audit = preflight["artifact_audit"]
    _require_equal(
        "checkpoint epoch",
        checkpoint_payload.get("epoch"),
        artifact_audit["checkpoint_epoch"],
    )
    _require_equal(
        "checkpoint role",
        checkpoint_payload.get("checkpoint_role"),
        CHECKPOINT_ROLES[checkpoint],
    )
    _require_equal(
        "checkpoint identity",
        checkpoint_payload.get("checkpoint_identity"),
        artifact_audit["checkpoint_identity"],
    )
    _require_equal(
        "checkpoint SHA after load",
        _sha256_file(checkpoint_path),
        expected_sha,
    )
    return dict(checkpoint_payload), state


def _build_v4_model(
    mode: str,
    source_state: Mapping[str, torch.Tensor],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if mode not in FORMULA_MODES:
        raise ValueError(f"unsupported formula mode: {mode}")
    parent, parent_metadata = exact_v3.build_clean_v8_mprs_dch_model(
        exact_v3.PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        TRAINING_SEED,
    )
    model = v4_model_source.adapt_v8_mprs_dch_parent_v4(
        parent,
        variant=exact_v3.PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_enabled=True,
        relay_width=exact_v3.RELAY_WIDTH,
        relay_initialization_seed=exact_v3.RELAY_INITIALIZATION_SEED,
        dc_support_mode=mode,
        tail_z_thresholds=v4_model_source.DEFAULT_TAIL_Z_THRESHOLDS,
    )
    if type(model) is not v4_model_source.TPDNERV8MPRSDCHV4SCTransNet:
        raise TypeError("counterfactual builder did not produce exact V4 class")
    _require_equal(
        "V4/source state keys",
        tuple(model.state_dict()),
        tuple(source_state),
    )
    _require_equal(
        "V4 relay parameter count",
        v4_model_source.v4_relay_parameter_count(model),
        v4_model_source.PRODUCTION_V4_RELAY_PARAMETERS,
    )
    _require_equal(
        "V4 total parameter count",
        sum(parameter.numel() for parameter in model.parameters()),
        v4_model_source.PRODUCTION_V4_RELAY_ON_PARAMETERS,
    )
    loaded = model.load_state_dict(source_state, strict=True)
    _require_equal("strict-load missing keys", list(loaded.missing_keys), [])
    _require_equal(
        "strict-load unexpected keys",
        list(loaded.unexpected_keys),
        [],
    )
    source_state_sha = frozen_diagnostic_core.state_content_sha256(
        source_state
    )
    _require_equal(
        "strict-loaded V4 state SHA",
        frozen_diagnostic_core.state_content_sha256(model.state_dict()),
        source_state_sha,
    )
    manifest = model.architecture_manifest()
    _require_equal(
        "V4 manifest formula mode",
        manifest.get("ner_dc_offset_support_mode"),
        mode,
    )
    _require_equal(
        "V4 manifest thresholds",
        manifest.get("tail_z_thresholds"),
        dict(v4_model_source.DEFAULT_TAIL_Z_THRESHOLDS),
    )
    _require_equal(
        "V4 manifest frozen thresholds",
        manifest.get("tail_z_thresholds_frozen"),
        True,
    )
    return model, {
        "model_class": (
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        ),
        "formula_mode": mode,
        "formula_expression": FORMULA_EXPRESSIONS[mode],
        "relay_parameters": v4_model_source.PRODUCTION_V4_RELAY_PARAMETERS,
        "total_parameters": v4_model_source.PRODUCTION_V4_RELAY_ON_PARAMETERS,
        "state_key_count": len(model.state_dict()),
        "state_dict_sha256": source_state_sha,
        "architecture_manifest": copy.deepcopy(manifest),
        "architecture_manifest_sha256": _canonical_sha256(manifest),
        "parent_metadata": metric_core.json_ready(parent_metadata),
    }


def _gpu_memory_record(device: torch.device) -> dict[str, Any]:
    return {
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }


def _formalize_sweep(
    raw_sweep: Mapping[str, Any],
) -> dict[str, Any]:
    sweep = copy.deepcopy(dict(raw_sweep))
    sweep["threshold_configuration"] = _canonical_threshold_configuration()
    fixed = sweep.get("fixed_threshold_0_5")
    if not isinstance(fixed, Mapping):
        raise ValueError("computed fixed threshold is missing")
    normalized_budgets = formal_v3_evaluator._normalize_budgets(sweep)
    sweep["final_metric_coverage"] = (
        formal_v3_evaluator._final_metric_coverage(
            fixed,
            normalized_budgets,
        )
    )
    canonical_sweep_projection(sweep)
    return metric_core.json_ready(sweep)


def _evaluate_formula_mode(
    *,
    mode: str,
    source_state: Mapping[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    source_state_sha = frozen_diagnostic_core.state_content_sha256(
        source_state
    )
    model, model_metadata = _build_v4_model(mode, source_state)
    model.to(device)
    _require_equal(
        f"{mode} state SHA after device transfer",
        frozen_diagnostic_core.state_content_sha256(model.state_dict()),
        source_state_sha,
    )
    torch.cuda.reset_peak_memory_stats(device)
    probabilities, targets, losses = metric_core.collect_predictions(
        model,
        loader,
        device,
    )
    memory = _gpu_memory_record(device)
    _require_equal(
        f"{mode} state SHA after inference",
        frozen_diagnostic_core.state_content_sha256(model.state_dict()),
        source_state_sha,
    )
    raw_sweep = frozen_diagnostic_core.sweep_predictions(
        probabilities,
        targets,
        losses,
        validation_count=VALIDATION_COUNT,
    )
    sweep = _formalize_sweep(raw_sweep)
    record = {
        "schema": MODE_EVALUATION_SCHEMA,
        "status": "complete",
        "formula_mode": mode,
        "formula_expression": FORMULA_EXPRESSIONS[mode],
        "formula_index": FORMULA_MODES.index(mode),
        "source_checkpoint_sha256_before": source_checkpoint_sha256,
        "source_checkpoint_sha256_after": source_checkpoint_sha256,
        "source_state_dict_sha256": source_state_sha,
        "evaluated_state_dict_sha256": source_state_sha,
        "strict_v3_state_load": True,
        "state_changed": False,
        "derived_checkpoint_written": False,
        "validation_count": VALIDATION_COUNT,
        "fixed_threshold": FIXED_THRESHOLD,
        "fa_budgets": list(FA_BUDGETS),
        "metric_contract": copy.deepcopy(METRIC_CONTRACT),
        "tail_z_thresholds": dict(
            v4_model_source.DEFAULT_TAIL_Z_THRESHOLDS
        ),
        "model": model_metadata,
        "gpu_memory": memory,
        **sweep,
    }
    metric_core.assert_finite_numbers(record, f"{mode} evaluation")
    del model, probabilities, targets, losses
    torch.cuda.empty_cache()
    return metric_core.json_ready(record)


def _environment_record(
    lane: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            None if not torch.backends.cudnn.is_available()
            else int(torch.backends.cudnn.version())
        ),
        "numpy_version": np.__version__,
        "lane": copy.deepcopy(dict(lane)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cublas_workspace_config": os.environ.get(
            CUBLAS_WORKSPACE_CONFIG_ENV
        ),
        "pythonhashseed": os.environ.get(PYTHONHASHSEED_ENV),
        "physical_gpu_index_env": os.environ.get(PHYSICAL_GPU_INDEX_ENV),
        "physical_gpu_uuid_env": os.environ.get(PHYSICAL_GPU_UUID_ENV),
        "determinism": copy.deepcopy(dict(determinism)),
    }


def validate_output_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("counterfactual payload must be a mapping")
    ready = copy.deepcopy(dict(payload))
    for name, expected in {
        "schema": EVALUATION_SCHEMA,
        "status": "complete",
        "artifact_kind": ARTIFACT_KIND,
        "diagnostic_only": True,
        "zero_training": True,
        "official_test_accessed": False,
        "checkpoint_filename": checkpoint,
        "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
        "formula_modes": list(FORMULA_MODES),
        "mode_count": len(FORMULA_MODES),
        "metric_contract": METRIC_CONTRACT,
        "derived_checkpoint_written": False,
        "output_overwrite_forbidden": True,
    }.items():
        _require_equal(f"payload {name}", ready.get(name), expected)
    evaluations = ready.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("payload evaluations must be a list")
    _require_equal("evaluation count", len(evaluations), len(FORMULA_MODES))
    _require_equal(
        "evaluation mode order",
        [item.get("formula_mode") for item in evaluations],
        list(FORMULA_MODES),
    )
    source_state_sha = ready.get("source_state_dict_sha256")
    for index, (mode, evaluation) in enumerate(
        zip(FORMULA_MODES, evaluations)
    ):
        if not isinstance(evaluation, Mapping):
            raise ValueError(f"evaluation[{index}] must be an object")
        _require_equal(
            f"evaluation[{index}] schema",
            evaluation.get("schema"),
            MODE_EVALUATION_SCHEMA,
        )
        _require_equal(
            f"evaluation[{index}] mode",
            evaluation.get("formula_mode"),
            mode,
        )
        _require_equal(
            f"evaluation[{index}] formula index",
            evaluation.get("formula_index"),
            index,
        )
        _require_equal(
            f"evaluation[{index}] source state",
            evaluation.get("source_state_dict_sha256"),
            source_state_sha,
        )
        _require_equal(
            f"evaluation[{index}] evaluated state",
            evaluation.get("evaluated_state_dict_sha256"),
            source_state_sha,
        )
        _require_equal(
            f"evaluation[{index}] strict load",
            evaluation.get("strict_v3_state_load"),
            True,
        )
        _require_equal(
            f"evaluation[{index}] state change",
            evaluation.get("state_changed"),
            False,
        )
        _require_equal(
            f"evaluation[{index}] metric contract",
            evaluation.get("metric_contract"),
            METRIC_CONTRACT,
        )
        coverage = evaluation.get("final_metric_coverage")
        if not isinstance(coverage, Mapping):
            raise ValueError(
                f"evaluation[{index}] final metric coverage is missing"
            )
        _require_equal(
            f"evaluation[{index}] final metric coverage schema",
            coverage.get("schema"),
            METRIC_CONTRACT["final_metric_coverage_schema"],
        )
        canonical_sweep_projection(evaluation)
    equivalence = ready.get("legacy_canonical_equivalence")
    if not isinstance(equivalence, Mapping):
        raise ValueError("legacy canonical equivalence record is missing")
    _require_equal(
        "legacy canonical exact",
        equivalence.get("legacy_global_canonical_exact"),
        True,
    )
    _require_equal(
        "implementation hashes before/after",
        ready.get("implementation_hashes_after"),
        ready.get("implementation_hashes_before"),
    )
    _require_equal(
        "input hashes before/after",
        ready.get("input_hashes_after"),
        ready.get("input_hashes_before"),
    )
    audit = ready.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("payload audit must be an object")
    for name in (
        "formal_v3_artifacts_read_only",
        "formal_v3_artifacts_unchanged",
        "all_modes_strict_loaded_from_same_pristine_v3_state",
        "legacy_checked_before_alternative_modes",
        "no_derived_checkpoint",
    ):
        _require_equal(f"audit {name}", audit.get(name), True)
    metric_core.assert_finite_numbers(ready, "counterfactual payload")
    return metric_core.json_ready(ready)


def run_checkpoint(
    checkpoint: str,
    device_name: str,
) -> dict[str, Any]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    _require_equal("run logical device", device_name, "cuda:0")
    preflight = _preflight(checkpoint)
    checkpoint_payload, source_state = _load_source_checkpoint(
        checkpoint,
        preflight,
    )
    source_state_sha = frozen_diagnostic_core.state_content_sha256(
        source_state
    )
    checkpoint_path = Path(preflight["checkpoint_path"])
    source_checkpoint_sha = preflight["input_hashes"]["source_checkpoint"]
    lane = _validated_cuda_lane(checkpoint, device_name)
    determinism = configure_v8_inference(device_name)
    device = torch.device(device_name)
    loader, data_contract = frozen_diagnostic_core._evaluation_data(
        preflight["artifact_audit"]
    )

    evaluations: list[dict[str, Any]] = []
    legacy_equivalence: dict[str, Any] | None = None
    for mode in FORMULA_MODES:
        _require_equal(
            f"checkpoint SHA before {mode}",
            _sha256_file(checkpoint_path),
            source_checkpoint_sha,
        )
        evaluation = _evaluate_formula_mode(
            mode=mode,
            source_state=source_state,
            loader=loader,
            device=device,
            source_checkpoint_sha256=source_checkpoint_sha,
        )
        if mode == v4_model_source.TailDCSupportMode.LEGACY_GLOBAL.value:
            legacy_equivalence = require_legacy_canonical_exact(
                evaluation,
                preflight["canonical_payload"],
            )
        elif legacy_equivalence is None:
            raise RuntimeError(
                "alternative formula reached before legacy canonical check"
            )
        evaluations.append(evaluation)
        _require_equal(
            f"checkpoint SHA after {mode}",
            _sha256_file(checkpoint_path),
            source_checkpoint_sha,
        )
        _require_equal(
            f"source state SHA after {mode}",
            frozen_diagnostic_core.state_content_sha256(source_state),
            source_state_sha,
        )
    if legacy_equivalence is None:
        raise RuntimeError("legacy canonical equivalence was not evaluated")

    implementation_after = implementation_hashes()
    input_after = _input_hashes(
        checkpoint_path,
        Path(preflight["canonical_path"]),
    )
    _require_equal(
        "implementation hashes before/after",
        implementation_after,
        preflight["implementation_hashes"],
    )
    _require_equal(
        "input hashes before/after",
        input_after,
        preflight["input_hashes"],
    )
    artifact_audit = preflight["artifact_audit"]
    output = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete",
        "artifact_kind": ARTIFACT_KIND,
        "scope": "same_v3_checkpoint_three_v4_forward_formulas",
        "diagnostic_only": True,
        "zero_training": True,
        "affects_v3_formal_decision": False,
        "formal_training_authorized_by_this_artifact": False,
        "official_test_accessed": False,
        "dataset": DATASET,
        "variant": VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "validation_count": VALIDATION_COUNT,
        "validation_split_sha256": artifact_audit[
            "validation_split_sha256"
        ],
        "run_directory": artifact_audit["run_directory"],
        "run_identity": copy.deepcopy(artifact_audit["run_identity"]),
        "checkpoint_filename": checkpoint,
        "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "checkpoint_validation_metrics": copy.deepcopy(
            artifact_audit["checkpoint_validation_metrics"]
        ),
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": source_checkpoint_sha,
            "identity": copy.deepcopy(
                artifact_audit["checkpoint_identity"]
            ),
            "state_dict_sha256": source_state_sha,
            "checkpoint_payload_state_dict_sha256": checkpoint_payload[
                "state_dict_sha256"
            ],
        },
        "source_state_dict_sha256": source_state_sha,
        "formula_modes": list(FORMULA_MODES),
        "formula_expressions": dict(FORMULA_EXPRESSIONS),
        "mode_count": len(FORMULA_MODES),
        "metric_contract": copy.deepcopy(METRIC_CONTRACT),
        "formal_default_mode": v4_model_source.DEFAULT_DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(
            v4_model_source.DEFAULT_TAIL_Z_THRESHOLDS
        ),
        "tail_z_thresholds_frozen": True,
        "fixed_threshold": FIXED_THRESHOLD,
        "fa_budgets": list(FA_BUDGETS),
        "data_contract": data_contract,
        "canonical_v3_sweep": {
            "path": str(preflight["canonical_path"]),
            "sha256": preflight["input_hashes"]["canonical_v3_sweep"],
            "schema": preflight["canonical_payload"].get("schema"),
        },
        "legacy_canonical_equivalence": legacy_equivalence,
        "evaluations": evaluations,
        "environment": _environment_record(lane, determinism),
        "implementation_hashes_before": copy.deepcopy(
            preflight["implementation_hashes"]
        ),
        "implementation_hashes_after": implementation_after,
        "input_hashes_before": copy.deepcopy(preflight["input_hashes"]),
        "input_hashes_after": input_after,
        "derived_checkpoint_written": False,
        "output_overwrite_forbidden": True,
        "audit": {
            "formal_v3_artifacts_read_only": True,
            "formal_v3_artifacts_unchanged": True,
            "all_modes_strict_loaded_from_same_pristine_v3_state": True,
            "mode_order": list(FORMULA_MODES),
            "legacy_checked_before_alternative_modes": True,
            "no_derived_checkpoint": True,
            "output_path": str(preflight["output_path"]),
            "invocation_argv": [
                sys.executable,
                str(EVALUATOR_PATH),
                *sys.argv[1:],
            ],
        },
    }
    ready = validate_output_payload(output, checkpoint=checkpoint)
    _require_equal(
        "checkpoint immediately before publication",
        _sha256_file(checkpoint_path),
        source_checkpoint_sha,
    )
    frozen_diagnostic_core.atomic_publish_new(
        Path(preflight["output_path"]),
        ready,
    )
    return ready


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run V3-checkpoint zero-training comparison of three "
            "V4 NER DC-support formulas"
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--plan",
        action="store_true",
        help="validate immutable inputs and print the fixed evaluation plan",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="evaluate all three formulas and publish one new JSON",
    )
    parser.add_argument(
        "--checkpoint",
        choices=CHECKPOINTS,
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("cuda:0",),
        default="cuda:0",
    )
    return parser.parse_args(
        list(sys.argv[1:] if argv is None else argv)
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.plan:
        payload = build_plan(args.checkpoint)
    else:
        payload = run_checkpoint(args.checkpoint, args.device)
    print(
        json.dumps(
            metric_core.json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACT_KIND",
    "BUDGET_KEYS",
    "CANONICAL_SWEEP_FIELDS",
    "CHECKPOINTS",
    "CHECKPOINT_GPU_LANES",
    "CHECKPOINT_ROLES",
    "CUBLAS_WORKSPACE_CONFIG_ENV",
    "CUBLAS_WORKSPACE_CONFIG_VALUE",
    "DEFAULT_OUTPUT_ROOT",
    "EVALUATION_SCHEMA",
    "EXPECTED_GPU_NAME",
    "FORMULA_EXPRESSIONS",
    "FORMULA_MODES",
    "METRIC_CONTRACT",
    "MODE_EVALUATION_SCHEMA",
    "PHYSICAL_GPU_INDEX_ENV",
    "PHYSICAL_GPU_UUID_ENV",
    "PLAN_SCHEMA",
    "PYTHONHASHSEED_ENV",
    "PYTHONHASHSEED_VALUE",
    "build_plan",
    "canonical_sweep_path",
    "canonical_sweep_projection",
    "implementation_hashes",
    "main",
    "output_path",
    "parse_args",
    "require_legacy_canonical_exact",
    "run_checkpoint",
    "validate_output_payload",
]
