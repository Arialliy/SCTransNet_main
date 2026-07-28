#!/usr/bin/env python3
"""Formal closed-interval Pd/Fa evaluator for the sole V2 relay-on model."""

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

from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v2_source_locks as v2_freeze,
)
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import train_tpd_ner_v8_mprs_dch_v2 as trainer  # noqa: E402
from experiments import train_tpd_ner_v8_mprs_dch_v2_exact as exact  # noqa: E402
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    DETERMINISM_SETTINGS,
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
    requested_device,
)
performance_gate_contract = v2_freeze.performance_gate_contract


EVALUATION_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_pd_fa_v1"
EVALUATION_SOURCE_BINDING_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_evaluation_source_binding_v1"
)
FINAL_METRIC_COVERAGE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_final_metric_coverage_v1"
)
DATASET = trainer.DATASET
VARIANT = trainer.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON
V1_CONTROL = trainer.V1_RELAY_OFF_REFERENCE
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
FA_BUDGETS = tuple(trainer.FA_BUDGETS)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
CLOSED_INTERVAL_CORE_PATH = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
)
ISOLATED_MODULE_NAME = "_sctransnet_tpd_ner_v8_mprs_dch_v2_pd_fa"
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
RAW_POINT_FIELDS = frozenset(
    {
        "val_loss",
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
        "fa",
        "false_objects_per_image",
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
        "threshold",
    }
)
EMPIRICAL_QUANTILE_KEYS = (
    "0.9",
    "0.95",
    "0.98",
    "0.99",
    "0.995",
    "0.999",
    "0.9995",
    "0.9999",
    "0.99995",
    "0.99999",
    "0.999995",
    "0.999999",
)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, observed={observed!r}"
        )


def _require_canonical_json_equal(
    location: str,
    observed: Any,
    expected: Any,
) -> None:
    """Compare JSON and checkpoint views without conflating list and tuple."""

    try:
        observed_json = exact_runner._canonical_json(
            observed,
            f"{location}.observed",
        )
        expected_json = exact_runner._canonical_json(
            expected,
            f"{location}.expected",
        )
    except exact_runner.ExactRunnerError as exc:
        raise ValueError(
            f"{location} must be canonical JSON-compatible: {exc}"
        ) from exc
    if observed_json != expected_json:
        raise ValueError(f"{location} differs after canonical JSON normalization")


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
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
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
        raise ValueError(f"{label} lies outside the repository: {resolved}") from exc


def _current_evaluation_source_binding() -> dict[str, Any]:
    """Bind the executable evaluator cores to the two current source locks."""

    training_lock = Path(v2_freeze.DEFAULT_TRAINING_LOCK).resolve()
    acceptance_lock = Path(v2_freeze.DEFAULT_ACCEPTANCE_LOCK).resolve()
    training_payload = _load_json(training_lock)
    acceptance_payload = _load_json(acceptance_lock)
    training_sha256 = _sha256_file(training_lock)
    acceptance_sha256 = _sha256_file(acceptance_lock)
    _require_equal(
        "training source-lock schema",
        training_payload.get("schema"),
        exact.EXACT_SOURCE_LOCK_SCHEMA,
    )
    _require_equal(
        "training source-lock kind",
        training_payload.get("lock_kind"),
        "training",
    )
    _require_equal(
        "acceptance source-lock schema",
        acceptance_payload.get("schema"),
        v2_freeze.ACCEPTANCE_SCHEMA,
    )
    _require_equal(
        "acceptance source-lock kind",
        acceptance_payload.get("lock_kind"),
        "acceptance",
    )
    _require_equal(
        "acceptance/training source-lock binding",
        acceptance_payload.get("training_source_lock_sha256"),
        training_sha256,
    )

    closed_module = sys.modules.get(
        adaptive_thresholds_closed_interval.__module__
    )
    closed_module_file = (
        None if closed_module is None else getattr(closed_module, "__file__", None)
    )
    if closed_module_file is None:
        raise ValueError("closed-interval metric core has no source file")
    _require_equal(
        "closed-interval callable source",
        Path(closed_module_file).resolve(),
        CLOSED_INTERVAL_CORE_PATH.resolve(),
    )

    evaluator_path = Path(__file__).resolve()
    source_paths = {
        "evaluator": evaluator_path,
        "shared_metric_core": BASE_EVALUATOR_PATH.resolve(),
        "closed_interval_core": CLOSED_INTERVAL_CORE_PATH.resolve(),
    }
    acceptance_sources = _require_mapping(
        "acceptance source_sha256",
        acceptance_payload.get("source_sha256"),
    )
    source_records: dict[str, Any] = {}
    for name, path in source_paths.items():
        relative = _repo_relative(path, name)
        digest = _sha256_file(path)
        _require_equal(
            f"acceptance source binding for {relative}",
            acceptance_sources.get(relative),
            digest,
        )
        source_records[name] = {
            "path": str(path),
            "relative_path": relative,
            "sha256": digest,
        }
    return {
        "schema": EVALUATION_SOURCE_BINDING_SCHEMA,
        "training_source_lock": {
            "path": str(training_lock),
            "sha256": training_sha256,
        },
        "acceptance_source_lock": {
            "path": str(acceptance_lock),
            "sha256": acceptance_sha256,
            "training_source_lock_sha256": training_sha256,
        },
        **source_records,
    }


def evaluator_contract() -> dict[str, Any]:
    source_binding = _current_evaluation_source_binding()
    return {
        "schema": EVALUATION_SCHEMA,
        "dataset": DATASET,
        "formal_variant": VARIANT,
        "required_control": V1_CONTROL,
        "relay_off_retrained": False,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "checkpoints": list(CHECKPOINT_ROLES),
        "fixed_threshold": 0.5,
        "fa_budgets": list(FA_BUDGETS),
        "metric_core": "experiments.evaluate_pd_fa_sweep",
        "metric_core_sha256": source_binding["shared_metric_core"]["sha256"],
        "closed_interval_core": (
            "experiments.evaluate_tpd_clean_v6_pd_fa."
            "adaptive_thresholds_closed_interval"
        ),
        "closed_interval_core_sha256": source_binding[
            "closed_interval_core"
        ]["sha256"],
        "training_source_lock_sha256": source_binding[
            "training_source_lock"
        ]["sha256"],
        "acceptance_source_lock_sha256": source_binding[
            "acceptance_source_lock"
        ]["sha256"],
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
        "official_test_accessed": False,
        "performance_gates": performance_gate_contract(),
        "paired_gate_status": "requires_v2_postprocess_aggregate",
        "evaluator_may_authorize_final_success": False,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "determinism": dict(DETERMINISM_SETTINGS),
    }


def build_model(variant: str, seed: int):
    if variant != VARIANT:
        raise ValueError("V2 evaluator accepts only the relay-on candidate")
    return trainer.build_tpd_ner_v8_mprs_dch_v2_model(variant, seed)


def _validate_protocol_arguments(
    arguments: Mapping[str, Any],
) -> None:
    expected = {
        "dataset": DATASET,
        "variant": VARIANT,
        "parent_variant": "tpd_clean_v8_mprs_dch_full",
        "relay_enabled": True,
        "relay_width": 8,
        "relay_initialization_seed": 42,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": EXPECTED_EPOCHS,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "val_fraction": 0.20,
        "eval_every": 1,
        "base_lr": 1e-3,
        "min_lr": 1e-5,
        "warmup_epochs": 10,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "eps": exact.FORMAL_EPS,
        "amp": False,
        "allow_cpu_smoke": False,
        "max_train_images": None,
        "max_val_images": None,
        "run_tag": exact.FORMAL_RUN_TAG,
        "device": "cuda:0",
    }
    for name, value in expected.items():
        _require_equal(
            f"protocol.arguments.{name}",
            arguments.get(name),
            value,
        )


def _identifier_sha256(identifiers: Sequence[str]) -> str:
    canonical = "\n".join(sorted(str(identifier) for identifier in identifiers))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_split(split: Mapping[str, Any]) -> dict[str, str]:
    for name, expected in {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "full_official_train_count": 663,
        "full_internal_train_count": 530,
        "full_internal_val_count": 133,
        "used_train_count": 530,
        "used_val_count": 133,
        "official_test_accessed": False,
    }.items():
        _require_equal(f"split.{name}", split.get(name), expected)
    for field in (
        "full_internal_train_ids",
        "full_internal_val_ids",
        "used_train_ids",
        "used_val_ids",
    ):
        values = split.get(field)
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"split.{field} must be a unique string list")
    expected_lengths = {
        "full_internal_train_ids": 530,
        "full_internal_val_ids": 133,
        "used_train_ids": 530,
        "used_val_ids": 133,
    }
    for field, expected in expected_lengths.items():
        _require_equal(f"split.{field} length", len(split[field]), expected)
    full_train = set(split["full_internal_train_ids"])
    full_val = set(split["full_internal_val_ids"])
    used_train = set(split["used_train_ids"])
    used_val = set(split["used_val_ids"])
    if full_train & full_val:
        raise ValueError("full training and validation identifiers overlap")
    if used_train & used_val:
        raise ValueError("training and validation identifiers overlap")
    if not used_train <= full_train or not used_val <= full_val:
        raise ValueError("used split identifiers are not subsets of full split")

    expected_hashes = {
        "full_internal_train_sha256": _identifier_sha256(
            split["full_internal_train_ids"]
        ),
        "full_internal_val_sha256": _identifier_sha256(
            split["full_internal_val_ids"]
        ),
        "used_train_sha256": _identifier_sha256(split["used_train_ids"]),
        "used_val_sha256": _identifier_sha256(split["used_val_ids"]),
    }
    hashes = _require_mapping("split.hashes", split.get("hashes"))
    _require_equal("split hash keys", set(hashes), set(expected_hashes))
    _require_equal("split hashes", dict(hashes), expected_hashes)
    return expected_hashes


def _validate_run_identity(
    identity: Any,
    split: Mapping[str, Any],
    source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = exact.require_v2_run_identity(
        identity,
        label="formal evaluator run identity",
        expected_variant=VARIANT,
    )
    _require_equal(
        "run_identity.schema",
        value.get("schema"),
        exact_runner.RUN_IDENTITY_SCHEMA,
    )
    expected_run_id = (
        f"{exact.RUN_ID_PREFIX}{DATASET}:{VARIANT}:"
        f"seed-{TRAINING_SEED}:split-{SPLIT_SEED}:{exact.FORMAL_RUN_TAG}"
    )
    _require_equal("run_identity.run_id", value.get("run_id"), expected_run_id)
    _require_equal("run_identity.dataset", value.get("dataset"), DATASET)
    for field in (
        "architecture_id",
        "builder_manifest_sha256",
        "split_sha256",
        "data_sha256",
        "contract_sha256",
    ):
        _require_sha256(f"run_identity.{field}", value.get(field))

    training = _require_mapping(
        "run_identity.training_contract",
        value.get("training_contract"),
    )
    _require_equal(
        "training_contract.selection_policy",
        training.get("selection_policy"),
        exact_runner.pd_miou_selection_policy(
            stored_metrics=exact.STORED_VALIDATION_METRICS
        ).normalized(),
    )
    for name, expected in {
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "amp": False,
        "total_epochs": EXPECTED_EPOCHS,
        "eval_interval": 1,
    }.items():
        _require_equal(f"training_contract.{name}", training.get(name), expected)
    environment = _require_mapping(
        "training_contract.environment",
        training.get("environment"),
    )
    physical_index = environment.get("physical_gpu_index")
    if type(physical_index) is not int or str(physical_index) not in (
        exact.PHYSICAL_GPU_UUIDS
    ):
        raise ValueError("formal V2 trajectory must use physical GPU 2 or 3")
    expected_uuid = exact.PHYSICAL_GPU_UUIDS[str(physical_index)]
    for name, expected in {
        "device_type": "cuda",
        "logical_device": "cuda:0",
        "visible_cuda_device_count": 1,
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "physical_gpu_uuid": expected_uuid,
        "physical_gpu_assignment_source": (
            "verified_v2_ner_worker_environment"
        ),
        "pythonhashseed": "42",
        "cublas_workspace_config": exact.FORMAL_CUBLAS_WORKSPACE_CONFIG,
    }.items():
        _require_equal(
            f"training_contract.environment.{name}",
            environment.get(name),
            expected,
        )
    deep_supervision = _require_mapping(
        "training_contract.deep_supervision",
        training.get("deep_supervision"),
    )
    for name, expected in {
        "enabled": True,
        "expected_outputs": 6,
        "training_uses_all_outputs": True,
        "validation_uses_final_output": True,
    }.items():
        _require_equal(
            f"training_contract.deep_supervision.{name}",
            deep_supervision.get(name),
            expected,
        )
    loss = _require_mapping("training_contract.loss", training.get("loss"))
    for name, expected in {
        "input": "post_sigmoid_probability",
        "aggregate": "sum",
        "compute_dtype": "float32",
    }.items():
        _require_equal(f"training_contract.loss.{name}", loss.get(name), expected)
    metric = _require_mapping(
        "training_contract.metric_config",
        training.get("metric_config"),
    )
    for name, expected in {
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "validation_batch_size": 1,
        "official_test_accessed": False,
    }.items():
        _require_equal(
            f"training_contract.metric_config.{name}",
            metric.get(name),
            expected,
        )

    fingerprints = _require_mapping(
        "run_identity.ordered_split_fingerprints",
        value.get("ordered_split_fingerprints"),
    )
    split_fields = {
        "full_train": "full_internal_train_ids",
        "full_validation": "full_internal_val_ids",
        "train": "used_train_ids",
        "validation": "used_val_ids",
    }
    for fingerprint_name, split_field in split_fields.items():
        expected = exact_runner.OrderedFingerprint.from_values(
            fingerprint_name,
            split[split_field],
        ).normalized()
        _require_equal(
            f"ordered_split_fingerprints.{fingerprint_name}",
            fingerprints.get(fingerprint_name),
            expected,
        )
    _require_equal(
        "run_identity.split_sha256",
        value.get("split_sha256"),
        _canonical_sha256(fingerprints),
    )
    data_fingerprints = _require_mapping(
        "run_identity.ordered_data_fingerprints",
        value.get("ordered_data_fingerprints"),
    )
    _require_equal(
        "ordered_data_fingerprints keys",
        set(data_fingerprints),
        {
            "official_training_data",
            "train_samples",
            "validation_samples",
            "normalization",
        },
    )
    for name, record in data_fingerprints.items():
        _require_sha256(
            f"ordered_data_fingerprints.{name}.sha256",
            _require_mapping(name, record).get("sha256"),
        )
    _require_equal(
        "run_identity.data_sha256",
        value.get("data_sha256"),
        _canonical_sha256(data_fingerprints),
    )
    binding = (
        _current_evaluation_source_binding()
        if source_binding is None
        else dict(source_binding)
    )
    source_locks = _require_mapping(
        "run_identity.source_locks",
        value.get("source_locks"),
    )
    training_lock = _require_mapping(
        "evaluation source training lock",
        binding.get("training_source_lock"),
    )
    acceptance_lock = _require_mapping(
        "evaluation source acceptance lock",
        binding.get("acceptance_source_lock"),
    )
    training_lock_sha256 = _require_sha256(
        "evaluation training source-lock SHA",
        training_lock.get("sha256"),
    )
    _require_equal(
        "run identity/current training source lock",
        source_locks.get(exact.SOURCE_LOCK_KEY),
        training_lock_sha256,
    )
    _require_equal(
        "acceptance/current training source lock",
        acceptance_lock.get("training_source_lock_sha256"),
        training_lock_sha256,
    )
    _require_sha256(
        "evaluation acceptance source-lock SHA",
        acceptance_lock.get("sha256"),
    )
    identity_contract = {
        "schema": exact_runner.RUN_IDENTITY_SCHEMA,
        "architecture_id": value["architecture_id"],
        "builder_manifest_sha256": value["builder_manifest_sha256"],
        "source_locks": value["source_locks"],
        "ordered_split_fingerprints": fingerprints,
        "ordered_data_fingerprints": data_fingerprints,
        "data_sha256": value["data_sha256"],
        "training": training,
    }
    _require_equal(
        "run_identity.contract_sha256",
        value.get("contract_sha256"),
        _canonical_sha256(identity_contract),
    )
    return value


def _validate_metrics(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in raw_lines):
        raise ValueError("V2 metrics must not contain blank rows")
    events = [json.loads(line) for line in raw_lines]
    if (
        len(events) != EXPECTED_EPOCHS
        or not all(isinstance(event, dict) for event in events)
        or [event.get("epoch") for event in events]
        != list(range(1, EXPECTED_EPOCHS + 1))
    ):
        raise ValueError("V2 metrics must be a contiguous 1..800 history")
    required_metrics = tuple(exact.STORED_VALIDATION_METRICS)
    for event in events:
        _require_equal("metrics.variant", event.get("variant"), VARIANT)
        for name in required_metrics:
            _require_finite(
                f"metrics[{event['epoch']}].{name}",
                event.get(name),
            )
    policy = exact_runner.pd_miou_selection_policy(
        stored_metrics=required_metrics
    )
    try:
        selection = policy.recompute(events, require_flags=True)
    except exact_runner.ExactRunnerError as exc:
        raise ValueError(f"V2 metrics selection history differs: {exc}") from exc
    return events, selection


def validate_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> dict[str, Any]:
    """Validate one complete exact V2 run before model state is consumed."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise NotADirectoryError(run_dir)
    if checkpoint_name not in CHECKPOINT_ROLES:
        raise ValueError("V2 evaluator accepts only best or best_miou")
    checkpoint_path = (run_dir / checkpoint_name).resolve()
    if checkpoint_path.parent != run_dir or not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    protocol = _load_json(run_dir / "protocol.json")
    split = _load_json(run_dir / "split.json")
    summary = _load_json(run_dir / "summary.json")
    metric_events, global_selection = _validate_metrics(
        run_dir / "metrics.jsonl"
    )
    source_binding = _current_evaluation_source_binding()
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _require_equal(
        "checkpoint SHA during preflight load",
        _sha256_file(checkpoint_path),
        checkpoint_sha256,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V2 compatibility checkpoint must be an object")

    _require_equal("protocol.schema", protocol.get("schema"), exact.ENTRY_SCHEMA)
    _require_equal(
        "summary.schema",
        summary.get("schema"),
        exact.COMPLETION_SUMMARY_SCHEMA,
    )
    _require_equal("summary.status", summary.get("status"), "complete")
    _require_equal(
        "checkpoint.schema",
        checkpoint.get("schema"),
        exact.CHECKPOINT_SCHEMA,
    )
    _require_equal(
        "checkpoint.derived_schema",
        checkpoint.get("derived_schema"),
        exact_runner.DERIVED_CHECKPOINT_SCHEMA,
    )
    arguments = _require_mapping("protocol.arguments", protocol.get("arguments"))
    _validate_protocol_arguments(arguments)
    _require_equal(
        "protocol.formal_contract",
        protocol.get("formal_contract"),
        exact.formal_contract(),
    )
    split_hashes = _validate_split(split)
    identity = _validate_run_identity(
        protocol.get("run_identity"),
        split,
        source_binding=source_binding,
    )
    checkpoint = exact.require_evaluator_checkpoint_payload(
        checkpoint,
        expected_variant=VARIANT,
    )

    for artifact_name, artifact in {
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"{artifact_name}.run_identity",
            artifact.get("run_identity"),
            identity,
        )
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"{artifact_name}.official_test_accessed",
            artifact.get("official_test_accessed"),
            False,
        )
        _require_equal(
            f"{artifact_name}.selection_source",
            artifact.get("selection_source"),
            "internal_validation_only",
        )
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
    }.items():
        _require_equal(
            f"{artifact_name}.stored_validation_metrics",
            artifact.get("stored_validation_metrics"),
            list(exact.STORED_VALIDATION_METRICS),
        )
    relay = _require_mapping("protocol.relay_identity", protocol.get("relay_identity"))
    for name, expected in {
        "parent_variant": "tpd_clean_v8_mprs_dch_full",
        "enabled": True,
        "version": "v2_rms_centered_arctangent",
        "width": 8,
        "initialization_seed": 42,
        "rms_eps": 1e-6,
        "gate_bias": False,
        "spatial_centering": "per_sample_mean_hw",
        "mask_mapping": "atan(pi*z)/pi",
    }.items():
        _require_equal(f"protocol.relay_identity.{name}", relay.get(name), expected)
    comparison = _require_mapping(
        "protocol.comparison_design",
        protocol.get("comparison_design"),
    )
    _require_equal("comparison required control", comparison.get("required_control"), V1_CONTROL)
    _require_equal("comparison relay-off retrained", comparison.get("relay_off_retrained"), False)

    scalar_expected = {
        "variant": VARIANT,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "parent_variant": "tpd_clean_v8_mprs_dch_full",
        "relay_enabled": True,
        "relay_version": "v2_rms_centered_arctangent",
        "relay_width": 8,
        "required_control": V1_CONTROL,
        "relay_off_retrained": False,
    }
    for artifact_name, artifact in {
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        for name, expected in scalar_expected.items():
            _require_equal(
                f"{artifact_name}.{name}",
                artifact.get(name),
                expected,
            )
    model_records = {
        "protocol": protocol.get("model"),
        "summary": summary.get("model"),
        "checkpoint": checkpoint.get("model_metadata"),
    }
    if not all(isinstance(record, Mapping) for record in model_records.values()):
        raise ValueError("V2 model metadata views must be objects")
    reference_model = dict(model_records["protocol"])
    for name, record in model_records.items():
        _require_canonical_json_equal(
            f"{name}.model",
            dict(record),
            reference_model,
        )
        _require_equal(f"{name}.model.variant", record.get("variant"), VARIANT)
        _require_equal(
            f"{name}.model.relay_version",
            record.get("relay_version"),
            "v2_rms_centered_arctangent",
        )

    expected_role = CHECKPOINT_ROLES[checkpoint_name]
    _require_equal(
        "checkpoint.checkpoint_role",
        checkpoint.get("checkpoint_role"),
        expected_role,
    )
    validation_metrics = _require_mapping(
        "checkpoint.validation_metrics",
        checkpoint.get("validation_metrics"),
    )
    for name in exact.STORED_VALIDATION_METRICS:
        _require_finite(
            f"checkpoint.validation_metrics.{name}",
            validation_metrics.get(name),
        )
    selection_slot = (
        "primary"
        if expected_role == "best_validation_pd_primary"
        else "secondary"
    )
    selected = _require_mapping(
        f"global_selection.{selection_slot}",
        global_selection.get(selection_slot),
    )
    checkpoint_epoch = checkpoint.get("epoch")
    _require_equal(
        "checkpoint epoch/global selection",
        checkpoint_epoch,
        selected.get("epoch"),
    )
    _require_equal(
        "checkpoint role/global selection",
        checkpoint.get("checkpoint_role"),
        selected.get("role"),
    )
    _require_canonical_json_equal(
        "checkpoint metrics/global selection",
        dict(validation_metrics),
        selected.get("metrics"),
    )
    primary = _require_mapping(
        "global_selection.primary",
        global_selection.get("primary"),
    )
    secondary = _require_mapping(
        "global_selection.secondary",
        global_selection.get("secondary"),
    )
    for name, observed, expected in (
        ("summary.best_epoch", summary.get("best_epoch"), primary.get("epoch")),
        (
            "summary.best_pd_epoch",
            summary.get("best_pd_epoch"),
            primary.get("epoch"),
        ),
        (
            "summary.best_miou_epoch",
            summary.get("best_miou_epoch"),
            secondary.get("epoch"),
        ),
    ):
        _require_equal(name, observed, expected)
    for name, observed, expected in (
        (
            "summary.best_validation_metrics",
            summary.get("best_validation_metrics"),
            primary.get("metrics"),
        ),
        (
            "summary.best_pd_validation_metrics",
            summary.get("best_pd_validation_metrics"),
            primary.get("metrics"),
        ),
        (
            "summary.best_miou_validation_metrics",
            summary.get("best_miou_validation_metrics"),
            secondary.get("metrics"),
        ),
    ):
        _require_canonical_json_equal(name, observed, expected)
    for component, digest_field in {
        "state_dict": "state_dict_sha256",
        "optimizer": "optimizer_state_sha256",
        "scaler": "scaler_state_sha256",
    }.items():
        state = _require_mapping(f"checkpoint.{component}", checkpoint.get(component))
        if component == "state_dict" and not state:
            raise ValueError("checkpoint.state_dict must be non-empty")
        _require_equal(
            f"checkpoint.{digest_field}",
            checkpoint.get(digest_field),
            exact_runner._state_content_sha256(
                state,
                f"V2 evaluator {component}",
            ),
        )
    _require_sha256(
        "checkpoint.source_exact_checkpoint_sha256",
        checkpoint.get("source_exact_checkpoint_sha256"),
    )
    _require_equal("summary.split_hashes", summary.get("split_hashes"), split_hashes)
    _require_equal("checkpoint.split_hashes", checkpoint.get("split_hashes"), split_hashes)

    _require_equal(
        "run directory name",
        run_dir.name,
        f"seed_42_{exact.FORMAL_RUN_TAG}",
    )
    _require_equal("run directory variant", run_dir.parent.name, VARIANT)
    _require_equal("run directory dataset", run_dir.parent.parent.name, DATASET)
    return {
        "training_artifact_mode": "exact_resume_primary",
        "run_directory": str(run_dir),
        "run_identity": identity,
        "variant": VARIANT,
        "checkpoint_identity": dict(checkpoint["checkpoint_identity"]),
        "checkpoint_filename": checkpoint_name,
        "checkpoint_role": expected_role,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_validation_metrics": dict(validation_metrics),
        "checkpoint_sha256": checkpoint_sha256,
        "global_selection": copy.deepcopy(global_selection),
        "validation_count": len(split["used_val_ids"]),
        "validation_split_sha256": split_hashes["used_val_sha256"],
        "evaluation_source_binding": copy.deepcopy(source_binding),
        "architecture_id": identity["architecture_id"],
        "required_control": V1_CONTROL,
        "relay_off_retrained": False,
    }


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
        raise ValueError("V2 formal evaluator forbids --overwrite")
    if args.checkpoint not in CHECKPOINT_ROLES:
        raise ValueError("V2 evaluator accepts only best or best_miou")
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("V2 evaluator device must be cpu or cuda:0")
    if args.expected_epochs not in (None, EXPECTED_EPOCHS):
        raise ValueError("V2 evaluator requires expected_epochs=800")
    for name, observed, expected_value in (
        ("threshold_min", args.threshold_min, 0.01),
        ("threshold_max", args.threshold_max, 0.99),
        ("threshold_step", args.threshold_step, 0.01),
        ("extra_thresholds", tuple(args.extra_thresholds), EXTRA_THRESHOLDS),
        ("tail_logit_step", args.tail_logit_step, 0.1),
        ("fa_budgets", tuple(args.fa_budgets), FA_BUDGETS),
    ):
        _require_equal(name, observed, expected_value)
    if args.match_radius not in (None, 3.0):
        raise ValueError("V2 evaluator match_radius must be omitted or 3.0")
    if args.tiny_area not in (None, 9):
        raise ValueError("V2 evaluator tiny_area must be omitted or 9")
    return args


def preflight_requested_artifacts(
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth.tar")
    args, _ = parser.parse_known_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    return validate_run_artifacts(args.run_dir, args.checkpoint)


def _require_preflight_checkpoint_unchanged(
    artifact_audit: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    checkpoint_path = (
        Path(str(artifact_audit.get("run_directory")))
        / str(artifact_audit.get("checkpoint_filename"))
    )
    _require_equal(
        f"checkpoint SHA {stage}",
        _sha256_file(checkpoint_path),
        artifact_audit.get("checkpoint_sha256"),
    )


def _require_integer(
    location: str,
    value: Any,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")
    return value


def _validate_standard_integrity(audit: Mapping[str, Any]) -> None:
    checks = _require_mapping(
        "audit.integrity_checks_passed",
        audit.get("integrity_checks_passed"),
    )
    _require_equal(
        "audit integrity check keys",
        set(checks),
        set(REQUIRED_INTEGRITY_CHECKS),
    )
    if any(checks[name] is not True for name in REQUIRED_INTEGRITY_CHECKS):
        raise ValueError("V2 evaluator integrity checks are incomplete")


def _validate_raw_point(
    point: Any,
    *,
    index: int,
    validation_count: int,
    invariant_counts: Mapping[str, int],
) -> dict[str, Any]:
    value = _require_mapping(f"points[{index}]", point)
    _require_equal(
        f"points[{index}] keys",
        set(value),
        set(RAW_POINT_FIELDS),
    )
    finite = {
        name: _require_finite(f"points[{index}].{name}", value.get(name))
        for name in (
            "val_loss",
            "miou",
            "niou",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
            "pd",
            "tiny_pd",
            "fa",
            "false_objects_per_image",
            "threshold",
        )
    }
    for name in (
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
        "fa",
        "threshold",
    ):
        if not 0.0 <= finite[name] <= 1.0:
            raise ValueError(f"points[{index}].{name} lies outside [0, 1]")
    for name in ("val_loss", "false_objects_per_image"):
        if finite[name] < 0.0:
            raise ValueError(f"points[{index}].{name} must be non-negative")

    counts = {
        name: _require_integer(f"points[{index}].{name}", value.get(name))
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
    if counts["valid_pixel_count"] < 1:
        raise ValueError(f"points[{index}].valid_pixel_count must be positive")
    for name, expected in invariant_counts.items():
        _require_equal(f"points[{index}].{name}", counts[name], expected)
    if not 0 <= counts["matched_target_count"] <= counts["target_count"]:
        raise ValueError(f"points[{index}] matched target count is invalid")
    if not 0 <= counts["matched_tiny_target_count"] <= counts["tiny_target_count"]:
        raise ValueError(f"points[{index}] matched tiny-target count is invalid")
    if counts["predicted_object_count"] < counts["matched_target_count"]:
        raise ValueError(f"points[{index}] predicted object count is invalid")
    if (
        counts["unmatched_predicted_object_count"]
        != counts["predicted_object_count"] - counts["matched_target_count"]
    ):
        raise ValueError(f"points[{index}] unmatched object count is invalid")
    _require_equal(
        f"points[{index}] Pd/count",
        finite["pd"],
        counts["matched_target_count"] / counts["target_count"],
    )
    _require_equal(
        f"points[{index}] tiny-Pd/count",
        finite["tiny_pd"],
        counts["matched_tiny_target_count"] / counts["tiny_target_count"],
    )
    _require_equal(
        f"points[{index}] false objects/image",
        finite["false_objects_per_image"],
        counts["unmatched_predicted_object_count"] / validation_count,
    )
    return copy.deepcopy(dict(value))


def _expected_raw_thresholds(
    provenance: Mapping[str, Any],
) -> list[float]:
    base_count = int(math.floor((0.99 - 0.01) / 0.01 + 1e-9))
    base = [
        round(0.01 + index * 0.01, 10)
        for index in range(base_count + 1)
    ]
    base.extend(EXTRA_THRESHOLDS)
    base.append(0.5)
    base = sorted(set(base))
    _require_equal(
        "threshold provenance uniform grid count",
        provenance.get("uniform_probability_grid_count"),
        len(base),
    )

    expected_tail_range = [
        math.log(0.95 / (1.0 - 0.95)),
        math.log(0.9999 / (1.0 - 0.9999)),
    ]
    _require_equal(
        "threshold provenance tail range",
        provenance.get("tail_logit_range"),
        expected_tail_range,
    )
    _require_equal(
        "threshold provenance tail step",
        provenance.get("tail_logit_step"),
        0.1,
    )
    tail_count = _require_integer(
        "threshold provenance tail count",
        provenance.get("tail_logit_threshold_count"),
        minimum=1,
    )
    _require_equal("threshold provenance tail count", tail_count, 64)
    tail = [
        1.0 / (1.0 + math.exp(-(expected_tail_range[0] + index * 0.1)))
        for index in range(tail_count)
    ]

    quantiles = _require_mapping(
        "threshold provenance empirical quantiles",
        provenance.get("empirical_score_quantiles"),
    )
    observed_keys = tuple(quantiles)
    unknown_keys = set(observed_keys) - set(EMPIRICAL_QUANTILE_KEYS)
    if unknown_keys:
        raise ValueError(
            "threshold provenance empirical quantile keys differ: "
            f"unknown={sorted(unknown_keys)}"
        )
    if observed_keys:
        first = EMPIRICAL_QUANTILE_KEYS.index(observed_keys[0])
        expected_keys = EMPIRICAL_QUANTILE_KEYS[
            first : first + len(observed_keys)
        ]
        _require_equal(
            "threshold provenance empirical quantile key order",
            observed_keys,
            expected_keys,
        )
    empirical = []
    for key in observed_keys:
        value = _require_finite(
            f"threshold provenance empirical quantile {key}",
            quantiles.get(key),
        )
        if not 0.0 < value < 1.0:
            raise ValueError("empirical threshold lies outside (0, 1)")
        empirical.append(value)
    if empirical != sorted(empirical):
        raise ValueError(
            "threshold provenance empirical quantiles are not monotonic"
        )
    return sorted(
        set(
            (
                *base,
                *tail,
                *empirical,
                LAST_FLOAT32_BELOW_ONE,
                UPPER_BOUNDARY_THRESHOLD,
            )
        )
    )


def _fixed_threshold_checkpoint_audit(
    fixed: Mapping[str, Any],
    checkpoint_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    _require_equal("raw fixed threshold", fixed.get("threshold"), 0.5)
    count_metric_keys = sorted(
        key for key in checkpoint_metrics if key.endswith("_count")
    )
    exact_keys = list(
        dict.fromkeys(
            [
                "pd",
                "fa",
                "tiny_pd",
                "false_objects_per_image",
                *count_metric_keys,
            ]
        )
    )
    exact_matches: dict[str, Any] = {}
    for key in exact_keys:
        if key not in checkpoint_metrics or key not in fixed:
            raise ValueError(f"cannot audit fixed-threshold metric {key!r}")
        exact_matches[key] = {
            "checkpoint": checkpoint_metrics[key],
            "sweep_0_5": fixed[key],
        }
        _require_equal(
            f"fixed-threshold checkpoint metric {key}",
            fixed[key],
            checkpoint_metrics[key],
        )
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


def _best_raw_point_under_fa(
    points: Sequence[Mapping[str, Any]],
    budget: float,
) -> Mapping[str, Any]:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    if not feasible:
        raise ValueError(f"raw points contain no point under Fa budget {budget}")
    return max(
        feasible,
        key=lambda point: (
            float(point["pd"]),
            -float(point["fa"]),
            float(point["tiny_pd"]),
            float(point["miou"]),
            -abs(float(point["threshold"]) - 0.5),
        ),
    )


def _validate_raw_points_and_summaries(
    payload: Mapping[str, Any],
    *,
    artifact_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation_count = _require_integer(
        "validation_count",
        payload.get("validation_count"),
        minimum=1,
    )
    _require_equal(
        "validation count/artifact",
        validation_count,
        artifact_audit.get("validation_count"),
    )
    checkpoint_metrics = _require_mapping(
        "artifact checkpoint validation metrics",
        artifact_audit.get("checkpoint_validation_metrics"),
    )
    _require_canonical_json_equal(
        "output checkpoint validation metrics",
        payload.get("checkpoint_validation_metrics"),
        checkpoint_metrics,
    )
    invariant_counts = {
        name: _require_integer(
            f"checkpoint validation metrics {name}",
            checkpoint_metrics.get(name),
            minimum=1,
        )
        for name in (
            "target_count",
            "tiny_target_count",
            "valid_pixel_count",
        )
    }
    _require_equal("formal target count", invariant_counts["target_count"], 189)

    raw_points = payload.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("V2 sweep raw points are missing")
    points = [
        _validate_raw_point(
            point,
            index=index,
            validation_count=validation_count,
            invariant_counts=invariant_counts,
        )
        for index, point in enumerate(raw_points)
    ]
    thresholds = [float(point["threshold"]) for point in points]
    if thresholds != sorted(thresholds) or len(thresholds) != len(set(thresholds)):
        raise ValueError("V2 sweep raw thresholds must be unique and sorted")
    provenance = _require_mapping(
        "threshold_provenance",
        payload.get("threshold_provenance"),
    )
    expected_thresholds = _expected_raw_thresholds(provenance)
    _require_equal(
        "threshold provenance total point count",
        provenance.get("total_unique_threshold_count"),
        len(points),
    )
    _require_equal(
        "raw/expected threshold count",
        len(points),
        len(expected_thresholds),
    )
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=2e-15)
        for observed, expected in zip(thresholds, expected_thresholds)
    ):
        raise ValueError("V2 sweep raw threshold set is incomplete")
    _require_equal(
        "threshold provenance score count",
        provenance.get("score_count"),
        invariant_counts["valid_pixel_count"],
    )

    fixed_matches = [
        point for point in points if float(point["threshold"]) == 0.5
    ]
    if len(fixed_matches) != 1:
        raise ValueError("V2 sweep must contain exactly one threshold=0.5 point")
    raw_fixed = fixed_matches[0]
    declared_fixed = _require_mapping(
        "fixed_threshold_0_5",
        payload.get("fixed_threshold_0_5"),
    )
    _require_canonical_json_equal(
        "fixed_threshold_0_5/raw point",
        declared_fixed,
        raw_fixed,
    )
    expected_fixed_audit = _fixed_threshold_checkpoint_audit(
        raw_fixed,
        checkpoint_metrics,
    )
    _require_canonical_json_equal(
        "fixed_threshold_0_5_checkpoint_audit",
        payload.get("fixed_threshold_0_5_checkpoint_audit"),
        expected_fixed_audit,
    )

    declared_budgets = _require_mapping(
        "best_points_under_fa_budget",
        payload.get("best_points_under_fa_budget"),
    )
    _require_equal("budget keys", set(declared_budgets), set(BUDGET_KEYS))
    for budget, key in zip(FA_BUDGETS, BUDGET_KEYS):
        expected = _best_raw_point_under_fa(points, budget)
        _require_canonical_json_equal(
            f"budget {key}/raw optimum",
            declared_budgets.get(key),
            expected,
        )
    return _normalize_fixed(declared_fixed), _normalize_budgets(payload)


def _normalize_fixed(point: Any) -> dict[str, Any]:
    value = _require_mapping("fixed_threshold_0_5", point)
    _require_equal("fixed threshold", value.get("threshold"), 0.5)
    target_count = value.get("target_count")
    matched = value.get("matched_target_count")
    if type(target_count) is not int or target_count != 189:
        raise ValueError("fixed target count differs")
    if type(matched) is not int or not 0 <= matched <= target_count:
        raise ValueError("fixed matched target count differs")
    normalized = {
        name: _require_finite(f"fixed.{name}", value.get(name))
        for name in (
            "pd",
            "fa",
            "miou",
            "false_objects_per_image",
            "threshold",
        )
    }
    _require_equal("fixed Pd/count", normalized["pd"], matched / target_count)
    for name in ("pd", "fa", "miou", "threshold"):
        if not 0.0 <= normalized[name] <= 1.0:
            raise ValueError(f"fixed {name} lies outside [0, 1]")
    if normalized["false_objects_per_image"] < 0.0:
        raise ValueError("fixed false_objects_per_image must be non-negative")
    return {
        **normalized,
        "target_count": target_count,
        "matched_target_count": matched,
    }


def _normalize_budgets(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = _require_mapping(
        "best_points_under_fa_budget",
        payload.get("best_points_under_fa_budget"),
    )
    _require_equal("budget keys", set(raw), set(BUDGET_KEYS))
    result: dict[str, Any] = {}
    for budget, key in zip(FA_BUDGETS, BUDGET_KEYS):
        point = _require_mapping(f"budget {key}", raw[key])
        target_count = point.get("target_count")
        matched = point.get("matched_target_count")
        if type(target_count) is not int or target_count != 189:
            raise ValueError(f"budget {key} target count differs")
        if type(matched) is not int or not 0 <= matched <= target_count:
            raise ValueError(f"budget {key} matched count differs")
        pd = _require_finite(f"budget {key} Pd", point.get("pd"))
        fa = _require_finite(f"budget {key} Fa", point.get("fa"))
        threshold = _require_finite(
            f"budget {key} threshold",
            point.get("threshold"),
        )
        _require_equal(f"budget {key} Pd/count", pd, matched / target_count)
        if not 0.0 <= fa <= budget:
            raise ValueError(f"budget {key} exceeds Fa")
        result[key] = {
            "budget": budget,
            "pd": point["pd"],
            "achieved_fa": point["fa"],
            "threshold": point["threshold"],
            "matched_target_count": matched,
            "target_count": target_count,
        }
    return result


def _absolute_gate(
    role: str,
    fixed: Mapping[str, Any],
    budgets: Mapping[str, Any],
) -> dict[str, Any]:
    contract = performance_gate_contract()
    gate_name = (
        "pd_primary_fixed_threshold_0_5"
        if role == "best_validation_pd_primary"
        else "miou_secondary_fixed_threshold_0_5"
    )
    requirement = contract[gate_name]
    fixed_checks = {
        "matched_targets": (
            fixed["matched_target_count"]
            >= requirement["minimum_matched_targets"]
        ),
        "pd": fixed["pd"] >= requirement["minimum_pd"],
        "fa": fixed["fa"] <= requirement["maximum_fa"],
        "miou": fixed["miou"] >= requirement["minimum_miou"],
    }
    budget_checks = {}
    for key in BUDGET_KEYS:
        needed = contract["pd_at_fa_budget"][key]
        observed = budgets[key]
        checks = {
            "matched_targets": (
                observed["matched_target_count"]
                >= needed["minimum_matched_targets"]
            ),
            "pd": observed["pd"] >= needed["minimum_pd"],
        }
        budget_checks[key] = {
            "required_matched_target_count": needed[
                "minimum_matched_targets"
            ],
            "required_pd": needed["minimum_pd"],
            "observed_matched_target_count": observed[
                "matched_target_count"
            ],
            "observed_target_count": observed["target_count"],
            "observed_pd": observed["pd"],
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(fixed_checks.values()) and all(
        record["passed"] for record in budget_checks.values()
    )
    return {
        "contract": contract,
        "fixed_threshold_gate": gate_name,
        "fixed_threshold_observed": {
            "matched_target_count": fixed["matched_target_count"],
            "target_count": fixed["target_count"],
            "pd": fixed["pd"],
            "fa": fixed["fa"],
            "miou": fixed["miou"],
        },
        "fixed_threshold_checks": fixed_checks,
        "budget_checks": budget_checks,
        "absolute_checkpoint_gate_passed": passed,
        "paired_v2_on_vs_v1_off_gate_status": (
            "requires_same_role_aggregate"
        ),
        "formal_success_claim_authorized": False,
    }


def _validate_closed_interval(payload: Mapping[str, Any]) -> None:
    provenance = _require_mapping(
        "threshold_provenance",
        payload.get("threshold_provenance"),
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
        _require_equal(f"threshold_provenance.{name}", provenance.get(name), expected)
    _require_equal(
        "threshold_provenance.added_thresholds",
        provenance.get("added_thresholds"),
        [LAST_FLOAT32_BELOW_ONE, UPPER_BOUNDARY_THRESHOLD],
    )
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("V2 sweep points are missing")
    by_threshold = {
        float(point["threshold"]): point
        for point in points
        if isinstance(point, Mapping) and "threshold" in point
    }
    if (
        LAST_FLOAT32_BELOW_ONE not in by_threshold
        or UPPER_BOUNDARY_THRESHOLD not in by_threshold
    ):
        raise ValueError("V2 sweep endpoint points are missing")
    upper = by_threshold[UPPER_BOUNDARY_THRESHOLD]
    for name, expected in {
        "pd": 0.0,
        "fa": 0.0,
        "matched_target_count": 0,
        "predicted_object_count": 0,
        "unmatched_predicted_object_count": 0,
    }.items():
        _require_equal(f"upper endpoint {name}", upper.get(name), expected)


def finalize_evaluation_output(
    payload: Mapping[str, Any],
    artifact_audit: Mapping[str, Any],
) -> dict[str, Any]:
    ready = copy.deepcopy(dict(payload))
    _require_preflight_checkpoint_unchanged(
        artifact_audit,
        stage="after base evaluator",
    )
    source_binding = _current_evaluation_source_binding()
    _require_canonical_json_equal(
        "preflight/current evaluation source binding",
        artifact_audit.get("evaluation_source_binding"),
        source_binding,
    )
    _require_equal("evaluation variant", ready.get("variant"), VARIANT)
    _require_equal("evaluation seed", ready.get("seed"), TRAINING_SEED)
    _require_equal("evaluation split seed", ready.get("split_seed"), SPLIT_SEED)
    checkpoint_name = Path(str(ready.get("checkpoint"))).name
    if checkpoint_name not in CHECKPOINT_ROLES:
        raise ValueError("V2 evaluation checkpoint filename differs")
    role = CHECKPOINT_ROLES[checkpoint_name]
    _require_equal("evaluation checkpoint role", ready.get("checkpoint_role"), role)
    _require_equal(
        "evaluation checkpoint epoch",
        ready.get("checkpoint_epoch"),
        artifact_audit.get("checkpoint_epoch"),
    )
    _require_equal(
        "evaluation validation split SHA",
        ready.get("validation_split_sha256"),
        artifact_audit.get("validation_split_sha256"),
    )
    fixed, budgets = _validate_raw_points_and_summaries(
        ready,
        artifact_audit=artifact_audit,
    )
    _validate_closed_interval(ready)
    checkpoint_sha256 = _require_sha256(
        "checkpoint_sha256",
        ready.get("checkpoint_sha256"),
    )
    _require_equal(
        "evaluation/preflight checkpoint SHA",
        checkpoint_sha256,
        artifact_audit.get("checkpoint_sha256"),
    )
    audit = copy.deepcopy(
        dict(_require_mapping("audit", ready.get("audit")))
    )
    _validate_standard_integrity(audit)
    artifact_hashes = dict(
        _require_mapping(
            "audit.artifact_sha256",
            audit.get("artifact_sha256"),
        )
    )
    artifact_hashes.update(
        {
            "training_source_lock": source_binding[
                "training_source_lock"
            ]["sha256"],
            "acceptance_source_lock": source_binding[
                "acceptance_source_lock"
            ]["sha256"],
            "shared_metric_core": source_binding[
                "shared_metric_core"
            ]["sha256"],
            "closed_interval_core": source_binding[
                "closed_interval_core"
            ]["sha256"],
        }
    )
    audit["artifact_sha256"] = artifact_hashes
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
            "evaluated_checkpoint_identity": {
                "training_artifact_mode": artifact_audit[
                    "training_artifact_mode"
                ],
                "filename": checkpoint_name,
                "role": role,
                "sha256": checkpoint_sha256,
            },
            "artifact_identity_preflight_passed": True,
            "required_control": V1_CONTROL,
            "relay_off_retrained": False,
            "evaluation_source_binding": copy.deepcopy(source_binding),
            "evaluator_contract": evaluator_contract(),
            "performance_gate_assessment": _absolute_gate(
                role,
                fixed,
                budgets,
            ),
            "final_metric_coverage": {
                "schema": FINAL_METRIC_COVERAGE_SCHEMA,
                "fixed_threshold": 0.5,
                "fixed_threshold_0_5": {
                    name: ready["fixed_threshold_0_5"][name]
                    for name in (
                        "pd",
                        "fa",
                        "miou",
                        "false_objects_per_image",
                    )
                },
                "pd_at_fa_budget": budgets,
                "all_required_metrics_present": True,
            },
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
    _require_canonical_json_equal(
        "artifact/current evaluation source binding",
        artifact_audit.get("evaluation_source_binding"),
        source_binding,
    )
    _require_canonical_json_equal(
        "output/current evaluation source binding",
        payload.get("evaluation_source_binding"),
        source_binding,
    )
    _require_equal("schema", payload.get("schema"), EVALUATION_SCHEMA)
    _require_equal("dataset", payload.get("dataset"), DATASET)
    _require_equal("variant", payload.get("variant"), VARIANT)
    _require_equal("seed", payload.get("seed"), TRAINING_SEED)
    _require_equal("split_seed", payload.get("split_seed"), SPLIT_SEED)
    _require_equal("official test", payload.get("official_test_accessed"), False)
    _require_equal("required control", payload.get("required_control"), V1_CONTROL)
    _require_equal("relay-off retrained", payload.get("relay_off_retrained"), False)
    run_dir = Path(str(payload.get("run_directory")))
    checkpoint_path = Path(str(payload.get("checkpoint")))
    if (
        not run_dir.is_absolute()
        or run_dir != run_dir.resolve()
        or not checkpoint_path.is_absolute()
        or checkpoint_path != checkpoint_path.resolve()
        or checkpoint_path.parent != run_dir
    ):
        raise ValueError("V2 output run/checkpoint paths differ")
    _require_equal(
        "output run directory",
        str(run_dir),
        artifact_audit["run_directory"],
    )
    checkpoint_name = checkpoint_path.name
    _require_equal(
        "output checkpoint filename",
        checkpoint_name,
        artifact_audit["checkpoint_filename"],
    )
    role = CHECKPOINT_ROLES[checkpoint_name]
    _require_equal("output checkpoint role", payload.get("checkpoint_role"), role)
    _require_equal(
        "output checkpoint epoch",
        payload.get("checkpoint_epoch"),
        artifact_audit.get("checkpoint_epoch"),
    )
    _require_canonical_json_equal(
        "output checkpoint validation metrics",
        payload.get("checkpoint_validation_metrics"),
        artifact_audit.get("checkpoint_validation_metrics"),
    )
    _require_equal(
        "output validation count",
        payload.get("validation_count"),
        artifact_audit.get("validation_count"),
    )
    _require_equal(
        "output validation split SHA",
        payload.get("validation_split_sha256"),
        artifact_audit.get("validation_split_sha256"),
    )
    checkpoint_sha = _sha256_file(checkpoint_path)
    _require_equal(
        "current/preflight checkpoint SHA",
        checkpoint_sha,
        artifact_audit.get("checkpoint_sha256"),
    )
    _require_equal(
        "output checkpoint SHA",
        payload.get("checkpoint_sha256"),
        checkpoint_sha,
    )
    _require_equal(
        "output run identity",
        payload.get("run_identity"),
        artifact_audit["run_identity"],
    )
    _require_equal(
        "output source checkpoint identity",
        payload.get("source_checkpoint_identity"),
        artifact_audit["checkpoint_identity"],
    )
    _require_equal(
        "output evaluated checkpoint identity",
        payload.get("evaluated_checkpoint_identity"),
        {
            "training_artifact_mode": "exact_resume_primary",
            "filename": checkpoint_name,
            "role": role,
            "sha256": checkpoint_sha,
        },
    )
    _require_equal(
        "artifact preflight",
        payload.get("artifact_identity_preflight_passed"),
        True,
    )
    _require_equal(
        "evaluator contract",
        payload.get("evaluator_contract"),
        evaluator_contract(),
    )
    configuration = _require_mapping(
        "threshold_configuration",
        payload.get("threshold_configuration"),
    )
    _require_equal(
        "threshold configuration",
        dict(configuration),
        {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": list(EXTRA_THRESHOLDS),
            "tail_logit_step": 0.1,
            "fa_budgets": list(FA_BUDGETS),
        },
    )
    _validate_closed_interval(payload)
    fixed, budgets = _validate_raw_points_and_summaries(
        payload,
        artifact_audit=artifact_audit,
    )
    coverage = _require_mapping(
        "final_metric_coverage",
        payload.get("final_metric_coverage"),
    )
    _require_equal(
        "coverage schema",
        coverage.get("schema"),
        FINAL_METRIC_COVERAGE_SCHEMA,
    )
    _require_equal("coverage fixed threshold", coverage.get("fixed_threshold"), 0.5)
    _require_equal(
        "coverage fixed metrics",
        coverage.get("fixed_threshold_0_5"),
        {
            name: payload["fixed_threshold_0_5"][name]
            for name in (
                "pd",
                "fa",
                "miou",
                "false_objects_per_image",
            )
        },
    )
    _require_equal("coverage budgets", coverage.get("pd_at_fa_budget"), budgets)
    _require_equal(
        "coverage complete",
        coverage.get("all_required_metrics_present"),
        True,
    )
    _require_equal(
        "recorded gate",
        payload.get("performance_gate_assessment"),
        _absolute_gate(role, fixed, budgets),
    )
    audit = _require_mapping("audit", payload.get("audit"))
    _require_equal("audit expected epochs", audit.get("expected_epochs"), 800)
    _require_equal("audit metrics count", audit.get("metrics_event_count"), 800)
    _require_equal("audit epoch range", audit.get("metrics_epoch_range"), [1, 800])
    _require_equal("audit summary status", audit.get("summary_status"), "complete")
    _require_equal(
        "audit selection source",
        audit.get("selection_source"),
        "internal_validation_only",
    )
    _validate_standard_integrity(audit)
    selection = _require_mapping(
        "artifact global selection",
        artifact_audit.get("global_selection"),
    )
    primary = _require_mapping(
        "artifact global primary selection",
        selection.get("primary"),
    )
    secondary = _require_mapping(
        "artifact global secondary selection",
        selection.get("secondary"),
    )
    _require_canonical_json_equal(
        "audit globally recomputed selection",
        audit.get("globally_recomputed_selection"),
        {
            "pd_primary": {
                "epoch": primary["epoch"],
                "key": primary["key"],
                "metrics": primary["metrics"],
            },
            "miou_secondary": {
                "epoch": secondary["epoch"],
                "key": secondary["key"],
                "metrics": secondary["metrics"],
            },
        },
    )
    expected_hashes = {
        "protocol.json": _sha256_file(run_dir / "protocol.json"),
        "split.json": _sha256_file(run_dir / "split.json"),
        "summary.json": _sha256_file(run_dir / "summary.json"),
        "metrics.jsonl": _sha256_file(run_dir / "metrics.jsonl"),
        "checkpoint": checkpoint_sha,
        "evaluator": _sha256_file(Path(__file__).resolve()),
        "training_source_lock": source_binding[
            "training_source_lock"
        ]["sha256"],
        "acceptance_source_lock": source_binding[
            "acceptance_source_lock"
        ]["sha256"],
        "shared_metric_core": source_binding[
            "shared_metric_core"
        ]["sha256"],
        "closed_interval_core": source_binding[
            "closed_interval_core"
        ]["sha256"],
    }
    _require_equal(
        "audit artifact SHA",
        audit.get("artifact_sha256"),
        expected_hashes,
    )
    invocation = audit.get("invocation_argv")
    if (
        not isinstance(invocation, list)
        or len(invocation) < 2
        or not Path(str(invocation[1])).is_absolute()
        or Path(str(invocation[1])).resolve() != Path(__file__).resolve()
    ):
        raise ValueError("V2 evaluator invocation identity differs")
    parsed = _require_mapping(
        "audit.parsed_arguments",
        audit.get("parsed_arguments"),
    )
    parsed_run = Path(str(parsed.get("run_dir")))
    if not parsed_run.is_absolute() or parsed_run != run_dir:
        raise ValueError("V2 evaluator parsed run directory differs")
    _require_equal("parsed checkpoint", parsed.get("checkpoint"), checkpoint_name)
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
        _require_equal(f"parsed argument {name}", parsed.get(name), expected)


def _atomic_write_output(
    path: Path,
    payload: Mapping[str, Any],
    overwrite: bool,
    *,
    artifact_audit: Mapping[str, Any],
    json_ready,
) -> None:
    if overwrite:
        raise ValueError("V2 formal evaluator forbids overwrite")
    ready = json_ready(
        finalize_evaluation_output(payload, artifact_audit)
    )
    content = (
        json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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
                f"refusing to replace existing V2 sweep: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_isolated_base_evaluator(
    artifact_audit: Mapping[str, Any],
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
    if "-h" not in argv and "--help" not in argv:
        validate_formal_arguments(argv)
    configure_v8_inference(requested_device(argv))
    artifact_audit = preflight_requested_artifacts(argv)
    evaluator = _load_isolated_base_evaluator(artifact_audit)
    _require_preflight_checkpoint_unchanged(
        artifact_audit,
        stage="before base evaluator",
    )
    try:
        evaluator.main()
    except BaseException:
        _require_preflight_checkpoint_unchanged(
            artifact_audit,
            stage="after failed base evaluator",
        )
        raise
    _require_preflight_checkpoint_unchanged(
        artifact_audit,
        stage="after base evaluator return",
    )


__all__ = [
    "BUDGET_KEYS",
    "CHECKPOINT_ROLES",
    "EVALUATION_SCHEMA",
    "EXPECTED_EPOCHS",
    "FA_BUDGETS",
    "FINAL_METRIC_COVERAGE_SCHEMA",
    "LAST_FLOAT32_BELOW_ONE",
    "UPPER_BOUNDARY_THRESHOLD",
    "VARIANT",
    "V1_CONTROL",
    "build_model",
    "evaluator_contract",
    "finalize_evaluation_output",
    "main",
    "performance_gate_contract",
    "preflight_requested_artifacts",
    "validate_formal_arguments",
    "validate_output_identity",
    "validate_run_artifacts",
]


if __name__ == "__main__":
    main()
