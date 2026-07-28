#!/usr/bin/env python3
"""Formal internal-validation Pd/Fa evaluator for the sole V4 model.

The evaluator consumes exactly one completed seed-42 V4 trajectory and one of
its two validation-selected checkpoints:

* ``best.pth.tar`` (Pd-primary role);
* ``best_miou.pth.tar`` (mIoU-secondary role).

It deliberately reuses the repository's established object-matching and
closed-interval threshold-sweep cores.  The official test split is never read.
The output remains field-compatible with the V3 sweep so that an independent
V4 postprocessor can compare the same checkpoint roles and Fa budgets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact as exact,
)
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    DETERMINISM_SETTINGS,
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
    requested_device,
)


EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa_v1"
)
EVALUATION_SOURCE_BINDING_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "evaluation_source_binding_v1"
)
FINAL_METRIC_COVERAGE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "final_metric_coverage_v1"
)
TRAINING_SCHEMA = exact.EXACT_SOURCE_LOCK_SCHEMA
DATASET = "NUDT-SIRST"
VARIANT = exact.TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON
TRAINING_SEED = exact.TRAINING_SEED
SPLIT_SEED = exact.SPLIT_SEED
EXPECTED_EPOCHS = exact.FORMAL_EPOCHS
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
EXPECTED_RELAY_STATE_KEY_COUNT = 19
EXPECTED_RELAY_PARAMETERS = 11_291
EXPECTED_TOTAL_PARAMETERS = 10_854_446
EXPECTED_DC_OFFSET_STATE_KEYS = frozenset(
    {
        "tpd_ner.dc_offsets.4",
        "tpd_ner.dc_offsets.3",
        "tpd_ner.dc_offsets.2",
    }
)
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
DEFAULT_TRAINING_LOCK = exact.DEFAULT_EXACT_SOURCE_LOCK_PATH
BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
CLOSED_INTERVAL_CORE_PATH = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
)
DETERMINISM_CORE_PATH = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py"
)
ISOLATED_MODULE_NAME = (
    "_sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa"
)
PHYSICAL_GPU_INDEX_ENV = (
    "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_INDEX"
)
PHYSICAL_GPU_UUID_ENV = (
    "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_UUID"
)
PHYSICAL_GPU_UUIDS = {
    str(index): uuid for index, uuid in exact.PHYSICAL_GPU_UUIDS.items()
}
REQUIRED_INTEGRITY_CHECKS = frozenset(
    {
        "summary_complete",
        "metrics_complete_contiguous_finite",
        "metadata_consistent",
        "official_test_isolated",
        "split_hashes_recomputed_consistent",
        "checkpoint_role_epoch_metrics_consistent",
        "global_selection_keys_recomputed",
        "state_dict_strict_load",
        "fixed_threshold_object_metrics_exact",
    }
)
REQUIRED_POINT_FIELDS = (
    "threshold",
    "pd",
    "fa",
    "miou",
    "false_objects_per_image",
    "tiny_pd",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
FINAL_FIXED_FIELDS = (
    "threshold",
    "pd",
    "fa",
    "miou",
    "false_objects_per_image",
    "tiny_pd",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _require_mapping(location: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _require_finite(location: str, value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite")
    return float(value)


def _require_integer(
    location: str,
    value: Any,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _require_sha256(location: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON-compatible") from exc


def _canonical_equal(
    location: str,
    observed: Any,
    expected: Any,
) -> None:
    if _canonical_json(observed) != _canonical_json(expected):
        raise ValueError(f"{location} differs after JSON normalization")


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _repo_relative(path: Path, label: str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(
            f"{label} lies outside the repository: {resolved}"
        ) from exc


def _identifier_sha256(identifiers: Sequence[str]) -> str:
    canonical = "\n".join(sorted(str(identifier) for identifier in identifiers))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _current_evaluation_source_binding(
    *,
    training_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the V4 training lock and bind all executable evaluator sources."""

    training_lock = Path(
        DEFAULT_TRAINING_LOCK
        if training_lock_path is None
        else training_lock_path
    ).resolve()
    payload = _load_json(training_lock)
    training_sha256 = _sha256_file(training_lock)
    _require_equal("training lock schema", payload.get("schema"), TRAINING_SCHEMA)
    _require_equal("training lock kind", payload.get("lock_kind"), "training")
    _require_equal("training lock dataset", payload.get("dataset"), DATASET)
    _require_equal("training lock variants", payload.get("variants"), [VARIANT])
    _require_equal(
        "training lock official-test policy",
        _require_mapping("training lock policy", payload.get("policy")).get(
            "official_test_accessed"
        ),
        False,
    )
    _require_sha256(
        "training data SHA", payload.get("training_data_sha256")
    )
    locked_sources = _require_mapping(
        "training lock source_sha256", payload.get("source_sha256")
    )
    _require_equal(
        "training lock source count",
        payload.get("source_count"),
        len(locked_sources),
    )
    for relative, expected_digest in locked_sources.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("training lock source path is invalid")
        expected = _require_sha256(
            f"training lock source {relative}", expected_digest
        )
        source = (REPO_ROOT / relative).resolve()
        _require_equal(
            f"training lock source location {relative}",
            _repo_relative(source, relative),
            relative,
        )
        _require_equal(
            f"training lock current SHA {relative}",
            _sha256_file(source),
            expected,
        )
    _canonical_equal(
        "training lock formal contract",
        payload.get("formal_contract"),
        exact.formal_contract(),
    )

    closed_module = sys.modules.get(
        adaptive_thresholds_closed_interval.__module__
    )
    closed_file = (
        None if closed_module is None else getattr(closed_module, "__file__", None)
    )
    if closed_file is None:
        raise ValueError("closed-interval core has no source file")
    _require_equal(
        "closed-interval callable source",
        Path(closed_file).resolve(),
        CLOSED_INTERVAL_CORE_PATH.resolve(),
    )
    deterministic_module = sys.modules.get(configure_v8_inference.__module__)
    deterministic_file = (
        None
        if deterministic_module is None
        else getattr(deterministic_module, "__file__", None)
    )
    if deterministic_file is None:
        raise ValueError("determinism core has no source file")
    _require_equal(
        "determinism callable source",
        Path(deterministic_file).resolve(),
        DETERMINISM_CORE_PATH.resolve(),
    )

    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "shared_metric_core": BASE_EVALUATOR_PATH.resolve(),
        "closed_interval_core": CLOSED_INTERVAL_CORE_PATH.resolve(),
        "determinism_core": DETERMINISM_CORE_PATH.resolve(),
    }
    source_records: dict[str, Any] = {}
    for name, path in source_paths.items():
        source_records[name] = {
            "path": str(path),
            "relative_path": _repo_relative(path, name),
            "sha256": _sha256_file(path),
        }
    return {
        "schema": EVALUATION_SOURCE_BINDING_SCHEMA,
        "training_source_lock": {
            "path": str(training_lock),
            "sha256": training_sha256,
            "schema": TRAINING_SCHEMA,
            "training_data_sha256": payload["training_data_sha256"],
        },
        **source_records,
    }


def verify_frozen_training_sources(
    *,
    training_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Public read-only verifier used by preflight and tests."""

    return _current_evaluation_source_binding(
        training_lock_path=training_lock_path
    )


def evaluator_contract() -> dict[str, Any]:
    binding = _current_evaluation_source_binding()
    return {
        "schema": EVALUATION_SCHEMA,
        "dataset": DATASET,
        "formal_variant": VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "expected_validation_count": EXPECTED_VALIDATION_COUNT,
        "checkpoints": list(CHECKPOINT_ROLES),
        "checkpoint_roles": dict(CHECKPOINT_ROLES),
        "required_control": exact.REQUIRED_CONTROL,
        "paired_gate_predecessor": exact.PAIRED_GATE_PREDECESSOR,
        "structural_predecessor": exact.STRUCTURAL_PREDECESSOR,
        "dc_support_mode": exact.DC_SUPPORT_MODE,
        "dc_support_formula_stage4": exact.DC_SUPPORT_FORMULA_STAGE4,
        "dc_support_formula_stage3_2": (
            exact.DC_SUPPORT_FORMULA_STAGE3_2
        ),
        "tail_z_thresholds": dict(exact.TAIL_Z_THRESHOLDS),
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "fixed_threshold": 0.5,
        "fa_budgets": list(FA_BUDGETS),
        "metric_core": "experiments.evaluate_pd_fa_sweep",
        "metric_core_sha256": binding["shared_metric_core"]["sha256"],
        "closed_interval_core": (
            "experiments.evaluate_tpd_clean_v6_pd_fa."
            "adaptive_thresholds_closed_interval"
        ),
        "closed_interval_core_sha256": binding[
            "closed_interval_core"
        ]["sha256"],
        "determinism_core": (
            "experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa"
        ),
        "determinism_core_sha256": binding["determinism_core"]["sha256"],
        "training_source_lock_sha256": binding[
            "training_source_lock"
        ]["sha256"],
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
        "official_test_accessed": False,
        "split_source": "img_idx/train_NUDT-SIRST.txt",
        "physical_gpu_choices": [2, 3],
        "logical_cuda_device": "cuda:0",
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "determinism": dict(DETERMINISM_SETTINGS),
    }


def build_model(variant: str, seed: int):
    if variant != VARIANT:
        raise ValueError("V4 evaluator accepts only the sole formal V4 variant")
    if seed != TRAINING_SEED:
        raise ValueError("V4 evaluator accepts only seed 42")
    return exact.build_selected_model(variant, seed)


def validate_formal_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth.tar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-epochs", type=int, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--extra-thresholds",
        type=float,
        nargs="+",
        default=list(EXTRA_THRESHOLDS),
    )
    parser.add_argument("--tail-logit-step", type=float, default=0.1)
    parser.add_argument(
        "--fa-budgets",
        type=float,
        nargs="+",
        default=list(FA_BUDGETS),
    )
    parser.add_argument("--match-radius", type=float, default=None)
    parser.add_argument("--tiny-area", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args, _ = parser.parse_known_args(values)
    if args.overwrite:
        raise ValueError("V4 formal evaluator forbids --overwrite")
    if args.checkpoint not in CHECKPOINT_ROLES:
        raise ValueError("V4 evaluator accepts only best or best_miou")
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("V4 evaluator device must be cpu or cuda:0")
    if args.expected_epochs not in (None, EXPECTED_EPOCHS):
        raise ValueError("V4 evaluator requires expected_epochs=800")
    for name, observed, expected in (
        ("threshold_min", args.threshold_min, 0.01),
        ("threshold_max", args.threshold_max, 0.99),
        ("threshold_step", args.threshold_step, 0.01),
        ("extra_thresholds", tuple(args.extra_thresholds), EXTRA_THRESHOLDS),
        ("tail_logit_step", args.tail_logit_step, 0.1),
        ("fa_budgets", tuple(args.fa_budgets), FA_BUDGETS),
    ):
        _require_equal(name, observed, expected)
    if args.match_radius not in (None, 3.0):
        raise ValueError("V4 evaluator match_radius must be omitted or 3.0")
    if args.tiny_area not in (None, 9):
        raise ValueError("V4 evaluator tiny_area must be omitted or 9")
    return args


def _device_assignment(device: str) -> dict[str, Any]:
    """Enforce the physical-GPU 2/3 assignment for CUDA evaluation."""

    if device == "cpu":
        return {
            "device": "cpu",
            "physical_gpu_index": None,
            "physical_gpu_uuid": None,
            "cuda_visible_devices": None,
            "device_name": "cpu",
        }
    if device != "cuda:0":
        raise ValueError("unsupported V4 evaluation device")
    physical_index = os.environ.get(PHYSICAL_GPU_INDEX_ENV)
    physical_uuid = os.environ.get(PHYSICAL_GPU_UUID_ENV)
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "V4 evaluation physical GPU index must be 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError("V4 evaluation physical GPU UUID differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain the assigned GPU UUID"
        )
    environment = exact.shared_exact.environment_contract(
        torch.device("cuda:0")
    )
    _require_equal(
        "visible CUDA count",
        environment.get("visible_cuda_device_count"),
        1,
    )
    _require_equal(
        "visible CUDA UUID",
        environment.get("device_uuid"),
        expected_uuid,
    )
    _require_equal(
        "visible CUDA model",
        environment.get("device_name"),
        "NVIDIA GeForce RTX 5090",
    )
    return {
        "device": "cuda:0",
        "physical_gpu_index": int(physical_index),
        "physical_gpu_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "device_name": environment["device_name"],
    }


def _validate_split(split: Mapping[str, Any]) -> dict[str, str]:
    _require_equal("split dataset", split.get("dataset"), DATASET)
    _require_equal("split seed", split.get("split_seed"), SPLIT_SEED)
    _require_equal("split source", split.get("source"), "img_idx/train_NUDT-SIRST.txt")
    _require_equal("split official test", split.get("official_test_accessed"), False)
    _require_equal("split full official-train count", split.get("full_official_train_count"), 663)
    _require_equal("split used validation count", split.get("used_val_count"), EXPECTED_VALIDATION_COUNT)
    _require_equal("split used training count", split.get("used_train_count"), 530)
    identifier_fields = {
        "full_internal_train_sha256": "full_internal_train_ids",
        "full_internal_val_sha256": "full_internal_val_ids",
        "used_train_sha256": "used_train_ids",
        "used_val_sha256": "used_val_ids",
    }
    hashes = _require_mapping("split hashes", split.get("hashes"))
    sets: dict[str, set[str]] = {}
    recomputed: dict[str, str] = {}
    for hash_name, ids_name in identifier_fields.items():
        identifiers = split.get(ids_name)
        if not isinstance(identifiers, list) or not all(
            isinstance(identifier, str) and identifier
            for identifier in identifiers
        ):
            raise ValueError(f"split {ids_name} is invalid")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"split {ids_name} contains duplicates")
        count_name = ids_name.removesuffix("_ids") + "_count"
        _require_equal(
            f"split {ids_name} count",
            split.get(count_name),
            len(identifiers),
        )
        digest = _identifier_sha256(identifiers)
        _require_equal(
            f"split {ids_name} SHA",
            hashes.get(hash_name),
            digest,
        )
        recomputed[hash_name] = digest
        sets[ids_name] = set(identifiers)
    if sets["full_internal_train_ids"] & sets["full_internal_val_ids"]:
        raise ValueError("full internal train/validation identifiers overlap")
    if sets["used_train_ids"] & sets["used_val_ids"]:
        raise ValueError("used internal train/validation identifiers overlap")
    if not sets["used_train_ids"] <= sets["full_internal_train_ids"]:
        raise ValueError("used training identifiers leave the internal split")
    if not sets["used_val_ids"] <= sets["full_internal_val_ids"]:
        raise ValueError("used validation identifiers leave the internal split")
    _require_equal(
        "full internal split union count",
        len(sets["full_internal_train_ids"])
        + len(sets["full_internal_val_ids"]),
        663,
    )
    return recomputed


def _validate_metrics(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != EXPECTED_EPOCHS or any(not line.strip() for line in lines):
        raise ValueError("V4 metrics must contain exactly 800 nonblank rows")
    events = [json.loads(line) for line in lines]
    if (
        not all(isinstance(event, dict) for event in events)
        or [event.get("epoch") for event in events]
        != list(range(1, EXPECTED_EPOCHS + 1))
    ):
        raise ValueError("V4 metrics must be a contiguous 1..800 history")
    for event in events:
        _require_equal("metrics variant", event.get("variant"), VARIANT)
        for name in exact.STORED_VALIDATION_METRICS:
            _require_finite(
                f"metrics[{event['epoch']}].{name}",
                event.get(name),
            )
    policy = exact_runner.pd_miou_selection_policy(
        stored_metrics=exact.STORED_VALIDATION_METRICS
    )
    try:
        selection = policy.recompute(events, require_flags=True)
    except exact_runner.ExactRunnerError as exc:
        raise ValueError(
            f"V4 metric-selection history differs: {exc}"
        ) from exc
    return events, selection


def _validate_model_state(checkpoint: Mapping[str, Any]) -> None:
    model, metadata = build_model(VARIANT, TRAINING_SEED)
    _canonical_equal(
        "rebuilt/checkpoint model metadata",
        metadata,
        checkpoint.get("model_metadata"),
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    _require_equal("strict-load missing keys", list(incompatible.missing_keys), [])
    _require_equal(
        "strict-load unexpected keys",
        list(incompatible.unexpected_keys),
        [],
    )
    state_keys = tuple(model.state_dict())
    relay_keys = tuple(
        name for name in state_keys if name.startswith("tpd_ner.")
    )
    _require_equal(
        "V4 relay state-key count",
        len(relay_keys),
        EXPECTED_RELAY_STATE_KEY_COUNT,
    )
    _require_equal(
        "V4 DC-offset state keys",
        frozenset(
            name
            for name in relay_keys
            if name.startswith("tpd_ner.dc_offsets.")
        ),
        EXPECTED_DC_OFFSET_STATE_KEYS,
    )
    _require_equal(
        "V4 relay parameters",
        metadata.get("relay_parameters"),
        EXPECTED_RELAY_PARAMETERS,
    )
    _require_equal(
        "V4 total parameters",
        metadata.get("total_parameters"),
        EXPECTED_TOTAL_PARAMETERS,
    )


def _source_checkpoint_identity(
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Extend the training identity with evaluator-required V4 constants."""

    ready = copy.deepcopy(dict(checkpoint_identity))
    ready.update(
        {
            "dc_support_mode": exact.DC_SUPPORT_MODE,
            "dc_support_formula_stage4": (
                exact.DC_SUPPORT_FORMULA_STAGE4
            ),
            "dc_support_formula_stage3_2": (
                exact.DC_SUPPORT_FORMULA_STAGE3_2
            ),
            "tail_z_thresholds": dict(exact.TAIL_Z_THRESHOLDS),
            "tail_z_thresholds_frozen": True,
            "target_protective_complement": True,
            "required_control": exact.REQUIRED_CONTROL,
            "paired_gate_predecessor": (
                exact.PAIRED_GATE_PREDECESSOR
            ),
            "structural_predecessor": exact.STRUCTURAL_PREDECESSOR,
        }
    )
    return ready


def validate_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> dict[str, Any]:
    """Audit one completed V4 run before its checkpoint is evaluated."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise NotADirectoryError(run_dir)
    if checkpoint_name not in CHECKPOINT_ROLES:
        raise ValueError("V4 evaluator accepts only best or best_miou")
    checkpoint_path = (run_dir / checkpoint_name).resolve()
    if checkpoint_path.parent != run_dir or not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    source_binding = _current_evaluation_source_binding()
    protocol = _load_json(run_dir / "protocol.json")
    split = _load_json(run_dir / "split.json")
    summary = _load_json(run_dir / "summary.json")
    events, selection = _validate_metrics(run_dir / "metrics.jsonl")
    split_hashes = _validate_split(split)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _require_equal(
        "checkpoint SHA after load",
        _sha256_file(checkpoint_path),
        checkpoint_sha256,
    )

    _require_equal("protocol schema", protocol.get("schema"), exact.ENTRY_SCHEMA)
    _require_equal(
        "summary schema",
        summary.get("schema"),
        exact.COMPLETION_SUMMARY_SCHEMA,
    )
    _require_equal("summary status", summary.get("status"), "complete")
    _canonical_equal(
        "protocol formal contract",
        protocol.get("formal_contract"),
        exact.formal_contract(),
    )
    arguments = _require_mapping(
        "protocol arguments", protocol.get("arguments")
    )
    for name, expected in {
        "dataset": DATASET,
        "variant": VARIANT,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": EXPECTED_EPOCHS,
        "eval_every": 1,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "run_tag": exact.FORMAL_RUN_TAG,
    }.items():
        _require_equal(f"protocol argument {name}", arguments.get(name), expected)
    _require_equal(
        "protocol stored metrics",
        protocol.get("stored_validation_metrics"),
        list(exact.STORED_VALIDATION_METRICS),
    )

    identity = exact.require_v4_run_identity(
        protocol.get("run_identity"),
        label="V4 evaluation protocol",
        expected_variant=VARIANT,
    )
    source_locks = _require_mapping(
        "run identity source locks", identity.get("source_locks")
    )
    _require_equal(
        "run/training source lock",
        source_locks.get(exact.SOURCE_LOCK_KEY),
        source_binding["training_source_lock"]["sha256"],
    )
    _require_equal(
        "run/training data SHA",
        source_locks.get("training_data"),
        source_binding["training_source_lock"]["training_data_sha256"],
    )

    checkpoint = exact.require_evaluator_checkpoint_payload(
        checkpoint,
        expected_variant=VARIANT,
    )
    _require_equal(
        "checkpoint derived schema",
        checkpoint.get("derived_schema"),
        exact_runner.DERIVED_CHECKPOINT_SCHEMA,
    )
    for component, digest_name in {
        "state_dict": "state_dict_sha256",
        "optimizer": "optimizer_state_sha256",
        "scaler": "scaler_state_sha256",
    }.items():
        state = _require_mapping(
            f"checkpoint {component}", checkpoint.get(component)
        )
        _require_equal(
            f"checkpoint {digest_name}",
            checkpoint.get(digest_name),
            exact_runner._state_content_sha256(
                state,
                f"V4 evaluator {component}",
            ),
        )
    _require_sha256(
        "checkpoint source exact SHA",
        checkpoint.get("source_exact_checkpoint_sha256"),
    )
    _validate_model_state(checkpoint)

    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"{artifact_name} official test",
            artifact.get("official_test_accessed"),
            False,
        )
        _require_equal(
            f"{artifact_name} selection source",
            artifact.get("selection_source"),
            "internal_validation_only",
        )
    for artifact_name, artifact in {
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"{artifact_name} run identity",
            artifact.get("run_identity"),
            identity,
        )
        _require_equal(
            f"{artifact_name} split hashes",
            artifact.get("split_hashes"),
            split_hashes,
        )
        _require_equal(f"{artifact_name} variant", artifact.get("variant"), VARIANT)
        _require_equal(f"{artifact_name} dataset", artifact.get("dataset"), DATASET)
        _require_equal(
            f"{artifact_name} seed", artifact.get("seed"), TRAINING_SEED
        )
        _require_equal(
            f"{artifact_name} split seed",
            artifact.get("split_seed"),
            SPLIT_SEED,
        )
    _canonical_equal(
        "summary/checkpoint model metadata",
        summary.get("model"),
        checkpoint.get("model_metadata"),
    )
    _canonical_equal(
        "protocol/checkpoint model metadata",
        protocol.get("model"),
        checkpoint.get("model_metadata"),
    )

    role = CHECKPOINT_ROLES[checkpoint_name]
    _require_equal(
        "checkpoint role", checkpoint.get("checkpoint_role"), role
    )
    slot = (
        "primary"
        if role == "best_validation_pd_primary"
        else "secondary"
    )
    selected = _require_mapping(
        f"global selection {slot}", selection.get(slot)
    )
    _require_equal(
        "checkpoint epoch/global selection",
        checkpoint.get("epoch"),
        selected.get("epoch"),
    )
    _require_equal(
        "checkpoint role/global selection",
        checkpoint.get("checkpoint_role"),
        selected.get("role"),
    )
    _canonical_equal(
        "checkpoint metrics/global selection",
        checkpoint.get("validation_metrics"),
        selected.get("metrics"),
    )
    primary = _require_mapping(
        "global selection primary", selection.get("primary")
    )
    secondary = _require_mapping(
        "global selection secondary", selection.get("secondary")
    )
    for location, observed, expected in (
        ("summary best epoch", summary.get("best_epoch"), primary.get("epoch")),
        (
            "summary best Pd epoch",
            summary.get("best_pd_epoch"),
            primary.get("epoch"),
        ),
        (
            "summary best mIoU epoch",
            summary.get("best_miou_epoch"),
            secondary.get("epoch"),
        ),
    ):
        _require_equal(location, observed, expected)
    for location, observed, expected in (
        (
            "summary best metrics",
            summary.get("best_validation_metrics"),
            primary.get("metrics"),
        ),
        (
            "summary best Pd metrics",
            summary.get("best_pd_validation_metrics"),
            primary.get("metrics"),
        ),
        (
            "summary best mIoU metrics",
            summary.get("best_miou_validation_metrics"),
            secondary.get("metrics"),
        ),
    ):
        _canonical_equal(location, observed, expected)

    _require_equal(
        "run directory name",
        run_dir.name,
        f"seed_{TRAINING_SEED}_{exact.FORMAL_RUN_TAG}",
    )
    _require_equal("run variant directory", run_dir.parent.name, VARIANT)
    _require_equal("run dataset directory", run_dir.parent.parent.name, DATASET)
    _require_equal(
        "validation count",
        len(split["used_val_ids"]),
        EXPECTED_VALIDATION_COUNT,
    )
    return {
        "training_artifact_mode": "v4_exact_resume_primary",
        "run_directory": str(run_dir),
        "run_identity": copy.deepcopy(identity),
        "variant": VARIANT,
        "checkpoint_identity": _source_checkpoint_identity(
            checkpoint["checkpoint_identity"]
        ),
        "training_checkpoint_identity": copy.deepcopy(
            dict(checkpoint["checkpoint_identity"])
        ),
        "checkpoint_filename": checkpoint_name,
        "checkpoint_role": role,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_metrics": copy.deepcopy(
            dict(checkpoint["validation_metrics"])
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "global_selection": copy.deepcopy(selection),
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "validation_split_sha256": split_hashes["used_val_sha256"],
        "evaluation_source_binding": copy.deepcopy(source_binding),
        "required_control": exact.REQUIRED_CONTROL,
        "paired_gate_predecessor": exact.PAIRED_GATE_PREDECESSOR,
        "structural_predecessor": exact.STRUCTURAL_PREDECESSOR,
        "relay_off_retrained": False,
        "metric_event_count": len(events),
    }


def preflight_requested_artifacts(
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth.tar")
    args, _ = parser.parse_known_args(values)
    return validate_run_artifacts(args.run_dir, args.checkpoint)


def _require_checkpoint_unchanged(
    artifact_audit: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    checkpoint = (
        Path(str(artifact_audit.get("run_directory")))
        / str(artifact_audit.get("checkpoint_filename"))
    )
    _require_equal(
        f"checkpoint SHA {stage}",
        _sha256_file(checkpoint),
        artifact_audit.get("checkpoint_sha256"),
    )


def _normalize_point(
    point: Any,
    *,
    location: str,
    fixed_threshold: bool = False,
) -> dict[str, Any]:
    value = _require_mapping(location, point)
    missing = [name for name in REQUIRED_POINT_FIELDS if name not in value]
    if missing:
        raise ValueError(f"{location} lacks metrics: {missing}")
    ready = copy.deepcopy(dict(value))
    finite = {
        name: _require_finite(f"{location}.{name}", ready.get(name))
        for name in (
            "threshold",
            "pd",
            "fa",
            "miou",
            "false_objects_per_image",
            "tiny_pd",
        )
    }
    counts = {
        name: _require_integer(f"{location}.{name}", ready.get(name))
        for name in (
            "target_count",
            "matched_target_count",
            "tiny_target_count",
            "matched_tiny_target_count",
            "predicted_object_count",
            "unmatched_predicted_object_count",
            "valid_pixel_count",
        )
    }
    _require_equal(
        f"{location}.target_count",
        counts["target_count"],
        EXPECTED_TARGET_COUNT,
    )
    _require_equal(
        f"{location}.tiny_target_count",
        counts["tiny_target_count"],
        EXPECTED_TINY_TARGET_COUNT,
    )
    if counts["matched_target_count"] > counts["target_count"]:
        raise ValueError(f"{location}.matched_target_count exceeds total")
    if counts["matched_tiny_target_count"] > counts["tiny_target_count"]:
        raise ValueError(f"{location}.matched_tiny_target_count exceeds total")
    if (
        counts["unmatched_predicted_object_count"]
        > counts["predicted_object_count"]
    ):
        raise ValueError(f"{location}.unmatched predictions exceed total")
    for name in ("threshold", "pd", "fa", "miou", "tiny_pd"):
        if not 0.0 <= finite[name] <= 1.0:
            raise ValueError(f"{location}.{name} lies outside [0, 1]")
    if finite["false_objects_per_image"] < 0.0:
        raise ValueError(f"{location}.false_objects_per_image is negative")
    if not math.isclose(
        finite["pd"],
        counts["matched_target_count"] / counts["target_count"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"{location}.pd differs from object counts")
    if not math.isclose(
        finite["tiny_pd"],
        counts["matched_tiny_target_count"] / counts["tiny_target_count"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"{location}.tiny_pd differs from tiny-object counts")
    if not math.isclose(
        finite["false_objects_per_image"],
        counts["unmatched_predicted_object_count"]
        / EXPECTED_VALIDATION_COUNT,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{location}.false_objects_per_image differs from counts"
        )
    if fixed_threshold and finite["threshold"] != 0.5:
        raise ValueError(f"{location} is not the fixed 0.5 point")
    return ready


def _normalize_budgets(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = _require_mapping(
        "best_points_under_fa_budget",
        payload.get("best_points_under_fa_budget"),
    )
    _require_equal("Fa budget keys", tuple(raw), BUDGET_KEYS)
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("sweep points are missing")
    normalized_points = [
        _normalize_point(point, location=f"points[{index}]")
        for index, point in enumerate(points)
    ]
    ready: dict[str, dict[str, Any]] = {}
    for key, budget in zip(BUDGET_KEYS, FA_BUDGETS):
        point = _normalize_point(raw.get(key), location=f"Fa budget {key}")
        if float(point["fa"]) > budget + 1e-18:
            raise ValueError(f"Fa budget {key} is exceeded")
        eligible = [
            candidate
            for candidate in normalized_points
            if float(candidate["fa"]) <= budget
        ]
        expected = max(
            eligible,
            key=lambda candidate: (
                float(candidate["pd"]),
                -float(candidate["fa"]),
                float(candidate["tiny_pd"]),
                float(candidate["miou"]),
                -abs(float(candidate["threshold"]) - 0.5),
            ),
        )
        _canonical_equal(f"Fa budget {key} best point", point, expected)
        ready[key] = point
    return ready


def _fixed_threshold_checkpoint_audit(
    fixed: Mapping[str, Any],
    checkpoint_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    count_keys = sorted(
        key for key in checkpoint_metrics if key.endswith("_count")
    )
    exact_keys = list(
        dict.fromkeys(
            [
                "pd",
                "fa",
                "tiny_pd",
                "false_objects_per_image",
                *count_keys,
            ]
        )
    )
    exact_matches: dict[str, Any] = {}
    for key in exact_keys:
        if key not in fixed or key not in checkpoint_metrics:
            raise ValueError(f"cannot audit fixed metric {key!r}")
        _require_equal(
            f"fixed/checkpoint metric {key}",
            fixed[key],
            checkpoint_metrics[key],
        )
        exact_matches[key] = {
            "checkpoint": checkpoint_metrics[key],
            "sweep_0_5": fixed[key],
        }
    numeric_deltas = {
        key: float(fixed[key]) - float(checkpoint_value)
        for key, checkpoint_value in checkpoint_metrics.items()
        if key in fixed
        and key not in exact_keys
        and isinstance(checkpoint_value, (int, float))
        and not isinstance(checkpoint_value, bool)
    }
    return {
        "exact_match_keys": exact_keys,
        "exact_matches": exact_matches,
        "non_strict_numeric_deltas_sweep_minus_checkpoint": numeric_deltas,
        "max_abs_non_strict_numeric_delta": max(
            (abs(delta) for delta in numeric_deltas.values()),
            default=0.0,
        ),
    }


def _validate_point_collection(
    payload: Mapping[str, Any],
    checkpoint_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    raw = payload.get("points")
    if not isinstance(raw, list) or not raw:
        raise ValueError("sweep points are missing")
    points = [
        _normalize_point(point, location=f"points[{index}]")
        for index, point in enumerate(raw)
    ]
    thresholds = [float(point["threshold"]) for point in points]
    if thresholds != sorted(thresholds) or len(thresholds) != len(set(thresholds)):
        raise ValueError("sweep thresholds must be sorted and unique")
    fixed_points = [
        point for point in points if float(point["threshold"]) == 0.5
    ]
    if len(fixed_points) != 1:
        raise ValueError("sweep must contain exactly one threshold=0.5 point")
    declared = _normalize_point(
        payload.get("fixed_threshold_0_5"),
        location="fixed_threshold_0_5",
        fixed_threshold=True,
    )
    _canonical_equal(
        "fixed threshold/raw point",
        declared,
        fixed_points[0],
    )
    expected_audit = _fixed_threshold_checkpoint_audit(
        declared,
        checkpoint_metrics,
    )
    _canonical_equal(
        "fixed threshold checkpoint audit",
        payload.get("fixed_threshold_0_5_checkpoint_audit"),
        expected_audit,
    )
    provenance = _require_mapping(
        "threshold provenance", payload.get("threshold_provenance")
    )
    _require_equal(
        "threshold provenance point count",
        provenance.get("total_unique_threshold_count"),
        len(points),
    )
    _require_equal(
        "threshold provenance score count",
        provenance.get("score_count"),
        declared["valid_pixel_count"],
    )
    for index, point in enumerate(points):
        for name in ("target_count", "tiny_target_count", "valid_pixel_count"):
            _require_equal(
                f"points[{index}].{name}",
                point[name],
                declared[name],
            )
    return declared


def _validate_closed_interval(payload: Mapping[str, Any]) -> None:
    provenance = _require_mapping(
        "threshold provenance", payload.get("threshold_provenance")
    )
    for name, expected in {
        "posthoc_endpoint_completion": False,
        "preregistered_endpoint_completion": True,
        "endpoint_protocol_stage": "before_formal_training",
        "closed_probability_interval": True,
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
        "upper_boundary_comparison": "prediction > threshold",
        "upper_boundary_semantics": "empty_prediction_pd0_fa0",
    }.items():
        _require_equal(
            f"threshold provenance {name}", provenance.get(name), expected
        )
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("sweep points are missing")
    by_threshold = {
        float(point["threshold"]): point
        for point in points
        if isinstance(point, Mapping) and "threshold" in point
    }
    for endpoint in (LAST_FLOAT32_BELOW_ONE, UPPER_BOUNDARY_THRESHOLD):
        if endpoint not in by_threshold:
            raise ValueError(f"closed-interval endpoint {endpoint} is missing")
    upper = by_threshold[UPPER_BOUNDARY_THRESHOLD]
    for name, expected in {
        "pd": 0.0,
        "fa": 0.0,
        "matched_target_count": 0,
        "predicted_object_count": 0,
        "unmatched_predicted_object_count": 0,
    }.items():
        _require_equal(f"upper endpoint {name}", upper.get(name), expected)


def _final_metric_coverage(
    fixed: Mapping[str, Any],
    budgets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": FINAL_METRIC_COVERAGE_SCHEMA,
        "fixed_threshold_0_5": {
            name: copy.deepcopy(fixed[name]) for name in FINAL_FIXED_FIELDS
        },
        "fa_budget_points": {
            key: {
                name: copy.deepcopy(point[name])
                for name in FINAL_FIXED_FIELDS
            }
            for key, point in budgets.items()
        },
        "required_metrics": [
            "pd",
            "fa",
            "miou",
            "false_objects_per_image",
            "tiny_pd",
        ],
        "fa_budgets": list(FA_BUDGETS),
        "fixed_threshold_complete": True,
        "fa_budget_curve_complete": True,
        "official_test_accessed": False,
    }


def _validate_standard_audit(audit: Mapping[str, Any]) -> None:
    checks = _require_mapping(
        "audit integrity checks", audit.get("integrity_checks_passed")
    )
    if not REQUIRED_INTEGRITY_CHECKS <= {
        name for name, passed in checks.items() if passed is True
    }:
        raise ValueError("base evaluator did not pass all required checks")
    _require_equal("audit expected epochs", audit.get("expected_epochs"), 800)
    _require_equal("audit metric count", audit.get("metrics_event_count"), 800)
    _require_equal("audit epoch range", audit.get("metrics_epoch_range"), [1, 800])
    _require_equal("audit summary status", audit.get("summary_status"), "complete")
    _require_equal(
        "audit selection source",
        audit.get("selection_source"),
        "internal_validation_only",
    )


def finalize_evaluation_output(
    payload: Mapping[str, Any],
    artifact_audit: Mapping[str, Any],
    *,
    device_assignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the V4 identity and validate every reportable metric view."""

    ready = copy.deepcopy(dict(payload))
    _require_checkpoint_unchanged(
        artifact_audit, stage="after shared evaluator"
    )
    source_binding = _current_evaluation_source_binding()
    _canonical_equal(
        "preflight/current source binding",
        artifact_audit.get("evaluation_source_binding"),
        source_binding,
    )
    for name, expected in {
        "variant": VARIANT,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "official_test_accessed": False,
        "match_radius": 3.0,
        "tiny_area": 9,
    }.items():
        _require_equal(f"evaluation {name}", ready.get(name), expected)
    checkpoint_name = Path(str(ready.get("checkpoint"))).name
    if checkpoint_name not in CHECKPOINT_ROLES:
        raise ValueError("V4 output checkpoint filename differs")
    role = CHECKPOINT_ROLES[checkpoint_name]
    _require_equal("evaluation checkpoint role", ready.get("checkpoint_role"), role)
    _require_equal(
        "evaluation checkpoint epoch",
        ready.get("checkpoint_epoch"),
        artifact_audit.get("checkpoint_epoch"),
    )
    _require_equal(
        "evaluation checkpoint SHA",
        ready.get("checkpoint_sha256"),
        artifact_audit.get("checkpoint_sha256"),
    )
    _canonical_equal(
        "evaluation checkpoint validation metrics",
        ready.get("checkpoint_validation_metrics"),
        artifact_audit.get("checkpoint_validation_metrics"),
    )
    _require_equal(
        "evaluation validation split SHA",
        ready.get("validation_split_sha256"),
        artifact_audit.get("validation_split_sha256"),
    )

    checkpoint_metrics = _require_mapping(
        "checkpoint validation metrics",
        artifact_audit.get("checkpoint_validation_metrics"),
    )
    fixed = _validate_point_collection(ready, checkpoint_metrics)
    budgets = _normalize_budgets(ready)
    _validate_closed_interval(ready)
    audit = copy.deepcopy(
        dict(_require_mapping("audit", ready.get("audit")))
    )
    _validate_standard_audit(audit)
    artifact_hashes = dict(
        _require_mapping(
            "audit artifact hashes", audit.get("artifact_sha256")
        )
    )
    artifact_hashes.update(
        {
            "training_source_lock": source_binding[
                "training_source_lock"
            ]["sha256"],
            "shared_metric_core": source_binding[
                "shared_metric_core"
            ]["sha256"],
            "closed_interval_core": source_binding[
                "closed_interval_core"
            ]["sha256"],
            "determinism_core": source_binding[
                "determinism_core"
            ]["sha256"],
        }
    )
    audit["artifact_sha256"] = artifact_hashes
    audit["device_assignment"] = copy.deepcopy(dict(device_assignment))
    ready["audit"] = audit
    ready.update(
        {
            "schema": EVALUATION_SCHEMA,
            "run_identity": copy.deepcopy(
                artifact_audit["run_identity"]
            ),
            "training_artifact_mode": artifact_audit[
                "training_artifact_mode"
            ],
            "source_checkpoint_identity": copy.deepcopy(
                artifact_audit["checkpoint_identity"]
            ),
            "training_checkpoint_identity": copy.deepcopy(
                artifact_audit["training_checkpoint_identity"]
            ),
            "evaluated_checkpoint_identity": {
                "training_artifact_mode": artifact_audit[
                    "training_artifact_mode"
                ],
                "filename": checkpoint_name,
                "role": role,
                "sha256": ready["checkpoint_sha256"],
            },
            "artifact_identity_preflight_passed": True,
            "required_control": exact.REQUIRED_CONTROL,
            "paired_gate_predecessor": exact.PAIRED_GATE_PREDECESSOR,
            "structural_predecessor": exact.STRUCTURAL_PREDECESSOR,
            "relay_off_retrained": False,
            "dc_support_mode": exact.DC_SUPPORT_MODE,
            "dc_support_formula_stage4": exact.DC_SUPPORT_FORMULA_STAGE4,
            "dc_support_formula_stage3_2": (
                exact.DC_SUPPORT_FORMULA_STAGE3_2
            ),
            "tail_z_thresholds": dict(exact.TAIL_Z_THRESHOLDS),
            "tail_z_thresholds_frozen": True,
            "target_protective_complement": True,
            "evaluation_source_binding": copy.deepcopy(source_binding),
            "evaluator_contract": evaluator_contract(),
            "final_metric_coverage": _final_metric_coverage(
                fixed, budgets
            ),
        }
    )
    validate_output_identity(ready, artifact_audit=artifact_audit)
    return ready


def validate_output_identity(
    payload: Mapping[str, Any],
    *,
    artifact_audit: Mapping[str, Any],
) -> None:
    source_binding = _current_evaluation_source_binding()
    _canonical_equal(
        "output/current source binding",
        payload.get("evaluation_source_binding"),
        source_binding,
    )
    for name, expected in {
        "schema": EVALUATION_SCHEMA,
        "dataset": DATASET,
        "variant": VARIANT,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "official_test_accessed": False,
        "required_control": exact.REQUIRED_CONTROL,
        "paired_gate_predecessor": exact.PAIRED_GATE_PREDECESSOR,
        "structural_predecessor": exact.STRUCTURAL_PREDECESSOR,
        "relay_off_retrained": False,
        "dc_support_mode": exact.DC_SUPPORT_MODE,
        "dc_support_formula_stage4": exact.DC_SUPPORT_FORMULA_STAGE4,
        "dc_support_formula_stage3_2": exact.DC_SUPPORT_FORMULA_STAGE3_2,
        "tail_z_thresholds": dict(exact.TAIL_Z_THRESHOLDS),
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "artifact_identity_preflight_passed": True,
    }.items():
        _require_equal(f"output {name}", payload.get(name), expected)
    run_dir = Path(str(payload.get("run_directory")))
    checkpoint_path = Path(str(payload.get("checkpoint")))
    if (
        not run_dir.is_absolute()
        or run_dir != run_dir.resolve()
        or not checkpoint_path.is_absolute()
        or checkpoint_path != checkpoint_path.resolve()
        or checkpoint_path.parent != run_dir
    ):
        raise ValueError("V4 output run/checkpoint paths differ")
    _require_equal(
        "output run directory",
        str(run_dir),
        artifact_audit.get("run_directory"),
    )
    checkpoint_name = checkpoint_path.name
    _require_equal(
        "output checkpoint filename",
        checkpoint_name,
        artifact_audit.get("checkpoint_filename"),
    )
    role = CHECKPOINT_ROLES[checkpoint_name]
    for location, observed, expected in (
        ("output checkpoint role", payload.get("checkpoint_role"), role),
        (
            "output checkpoint epoch",
            payload.get("checkpoint_epoch"),
            artifact_audit.get("checkpoint_epoch"),
        ),
        (
            "output checkpoint SHA",
            payload.get("checkpoint_sha256"),
            artifact_audit.get("checkpoint_sha256"),
        ),
        (
            "output validation split SHA",
            payload.get("validation_split_sha256"),
            artifact_audit.get("validation_split_sha256"),
        ),
    ):
        _require_equal(location, observed, expected)
    _canonical_equal(
        "output run identity",
        payload.get("run_identity"),
        artifact_audit.get("run_identity"),
    )
    _canonical_equal(
        "output source checkpoint identity",
        payload.get("source_checkpoint_identity"),
        artifact_audit.get("checkpoint_identity"),
    )
    _canonical_equal(
        "output training checkpoint identity",
        payload.get("training_checkpoint_identity"),
        artifact_audit.get("training_checkpoint_identity"),
    )
    _canonical_equal(
        "output evaluated checkpoint identity",
        payload.get("evaluated_checkpoint_identity"),
        {
            "training_artifact_mode": artifact_audit[
                "training_artifact_mode"
            ],
            "filename": checkpoint_name,
            "role": role,
            "sha256": artifact_audit["checkpoint_sha256"],
        },
    )
    _canonical_equal(
        "output evaluator contract",
        payload.get("evaluator_contract"),
        evaluator_contract(),
    )
    _canonical_equal(
        "threshold configuration",
        payload.get("threshold_configuration"),
        {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": list(EXTRA_THRESHOLDS),
            "tail_logit_step": 0.1,
            "fa_budgets": list(FA_BUDGETS),
        },
    )
    fixed = _validate_point_collection(
        payload,
        _require_mapping(
            "artifact checkpoint validation metrics",
            artifact_audit.get("checkpoint_validation_metrics"),
        ),
    )
    budgets = _normalize_budgets(payload)
    _validate_closed_interval(payload)
    _canonical_equal(
        "final metric coverage",
        payload.get("final_metric_coverage"),
        _final_metric_coverage(fixed, budgets),
    )
    audit = _require_mapping("output audit", payload.get("audit"))
    _validate_standard_audit(audit)
    expected_hashes = {
        "protocol.json": _sha256_file(run_dir / "protocol.json"),
        "split.json": _sha256_file(run_dir / "split.json"),
        "summary.json": _sha256_file(run_dir / "summary.json"),
        "metrics.jsonl": _sha256_file(run_dir / "metrics.jsonl"),
        "checkpoint": _sha256_file(checkpoint_path),
        "evaluator": _sha256_file(Path(__file__).resolve()),
        "training_source_lock": source_binding[
            "training_source_lock"
        ]["sha256"],
        "shared_metric_core": source_binding[
            "shared_metric_core"
        ]["sha256"],
        "closed_interval_core": source_binding[
            "closed_interval_core"
        ]["sha256"],
        "determinism_core": source_binding[
            "determinism_core"
        ]["sha256"],
    }
    _require_equal(
        "output artifact hashes",
        audit.get("artifact_sha256"),
        expected_hashes,
    )
    invocation = audit.get("invocation_argv")
    if (
        not isinstance(invocation, list)
        or len(invocation) < 2
        or Path(str(invocation[1])).resolve() != Path(__file__).resolve()
    ):
        raise ValueError("V4 evaluator invocation identity differs")
    parsed = _require_mapping(
        "output parsed arguments", audit.get("parsed_arguments")
    )
    _require_equal(
        "output parsed run directory",
        Path(str(parsed.get("run_dir"))).resolve(),
        run_dir,
    )
    _require_equal(
        "output parsed checkpoint",
        parsed.get("checkpoint"),
        checkpoint_name,
    )
    parsed_device = parsed.get("device")
    if parsed_device not in ("cpu", "cuda:0"):
        raise ValueError("output parsed device differs")
    assignment = _require_mapping(
        "output device assignment", audit.get("device_assignment")
    )
    if parsed_device == "cpu":
        _require_equal(
            "output CPU assignment",
            dict(assignment),
            {
                "device": "cpu",
                "physical_gpu_index": None,
                "physical_gpu_uuid": None,
                "cuda_visible_devices": None,
                "device_name": "cpu",
            },
        )
    else:
        physical_index = str(assignment.get("physical_gpu_index"))
        if physical_index not in PHYSICAL_GPU_UUIDS:
            raise ValueError("output physical GPU is not 2 or 3")
        expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
        for name, expected in {
            "device": "cuda:0",
            "physical_gpu_uuid": expected_uuid,
            "cuda_visible_devices": expected_uuid,
            "device_name": "NVIDIA GeForce RTX 5090",
        }.items():
            _require_equal(
                f"output device assignment {name}",
                assignment.get(name),
                expected,
            )
    for name, expected in {
        "expected_epochs": 800,
        "threshold_min": 0.01,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "extra_thresholds": list(EXTRA_THRESHOLDS),
        "tail_logit_step": 0.1,
        "fa_budgets": list(FA_BUDGETS),
        "match_radius": None,
        "tiny_area": None,
        "overwrite": False,
    }.items():
        _require_equal(
            f"output parsed argument {name}",
            parsed.get(name),
            expected,
        )


def _atomic_write_output(
    path: Path,
    payload: Mapping[str, Any],
    overwrite: bool,
    *,
    artifact_audit: Mapping[str, Any],
    device_assignment: Mapping[str, Any],
    json_ready,
) -> None:
    if overwrite:
        raise ValueError("V4 formal evaluator forbids overwrite")
    ready = json_ready(
        finalize_evaluation_output(
            payload,
            artifact_audit,
            device_assignment=device_assignment,
        )
    )
    content = (
        json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing V4 sweep: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_isolated_base_evaluator(
    artifact_audit: Mapping[str, Any],
    device_assignment: Mapping[str, Any],
) -> ModuleType:
    if not BASE_EVALUATOR_PATH.is_file():
        raise FileNotFoundError(BASE_EVALUATOR_PATH)
    spec = importlib.util.spec_from_file_location(
        ISOLATED_MODULE_NAME,
        BASE_EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the shared Pd/Fa evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original_parse_args = evaluator.parse_args

    def bound_parse_args() -> argparse.Namespace:
        args = original_parse_args()
        validate_formal_arguments(sys.argv[1:])
        if args.expected_epochs is None:
            args.expected_epochs = EXPECTED_EPOCHS
        return args

    def bound_write_output_json(
        path: Path,
        payload: dict[str, Any],
        overwrite: bool,
    ) -> None:
        _atomic_write_output(
            path,
            payload,
            overwrite,
            artifact_audit=artifact_audit,
            device_assignment=device_assignment,
            json_ready=evaluator.json_ready,
        )

    evaluator.adaptive_thresholds = adaptive_thresholds_closed_interval
    evaluator.build_model = build_model
    evaluator.parse_args = bound_parse_args
    evaluator.write_output_json = bound_write_output_json
    evaluator.__file__ = __file__
    return evaluator


def main() -> None:
    argv = list(sys.argv[1:])
    help_requested = "-h" in argv or "--help" in argv
    if help_requested:
        if not BASE_EVALUATOR_PATH.is_file():
            raise FileNotFoundError(BASE_EVALUATOR_PATH)
        spec = importlib.util.spec_from_file_location(
            ISOLATED_MODULE_NAME,
            BASE_EVALUATOR_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the shared Pd/Fa evaluator")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        evaluator.adaptive_thresholds = adaptive_thresholds_closed_interval
        evaluator.build_model = build_model
        evaluator.__file__ = __file__
        evaluator.main()
        return
    args = validate_formal_arguments(argv)
    _current_evaluation_source_binding()
    device = requested_device(argv)
    configure_v8_inference(device)
    assignment = _device_assignment(args.device)
    artifact_audit = preflight_requested_artifacts(argv)
    evaluator = _load_isolated_base_evaluator(
        artifact_audit,
        assignment,
    )
    _require_checkpoint_unchanged(
        artifact_audit, stage="before shared evaluator"
    )
    try:
        evaluator.main()
    except BaseException:
        _require_checkpoint_unchanged(
            artifact_audit, stage="after failed shared evaluator"
        )
        raise
    _require_checkpoint_unchanged(
        artifact_audit, stage="after shared evaluator return"
    )


__all__ = [
    "BUDGET_KEYS",
    "CHECKPOINT_ROLES",
    "DATASET",
    "DEFAULT_TRAINING_LOCK",
    "EVALUATION_SCHEMA",
    "EVALUATION_SOURCE_BINDING_SCHEMA",
    "EXPECTED_EPOCHS",
    "EXPECTED_VALIDATION_COUNT",
    "FA_BUDGETS",
    "FINAL_METRIC_COVERAGE_SCHEMA",
    "LAST_FLOAT32_BELOW_ONE",
    "UPPER_BOUNDARY_THRESHOLD",
    "VARIANT",
    "build_model",
    "evaluator_contract",
    "finalize_evaluation_output",
    "main",
    "preflight_requested_artifacts",
    "validate_formal_arguments",
    "validate_output_identity",
    "validate_run_artifacts",
    "verify_frozen_training_sources",
]


if __name__ == "__main__":
    main()
