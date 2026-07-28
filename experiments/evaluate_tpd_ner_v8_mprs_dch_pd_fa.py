#!/usr/bin/env python3
"""Formal Pd/Fa evaluator for V8-MPRS-DCH plus five-node NER.

The numerical metric implementation remains the audited shared
``evaluate_pd_fa_sweep`` runner.  This wrapper loads that runner into a
private module, binds the explicit relay-off/on builder and preregistered
closed-interval threshold function, and adds strict single-seed artifact and
final-metric coverage checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    DETERMINISM_SETTINGS,
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
    requested_device,
)
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import train_tpd_ner_v8_mprs_dch_exact as exact_trainer  # noqa: E402
from experiments.train_tpd_ner_v8_mprs_dch import (  # noqa: E402
    CANDIDATE_FAMILY,
    CHECKPOINT_IDENTITY_SCHEMA,
    CHECKPOINT_SCHEMA,
    DATASET,
    ENTRY_SCHEMA,
    FA_BUDGETS,
    FORMAL_EPOCHS,
    FORMAL_RUN_TAG,
    FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS,
    RUN_ID_PREFIX,
    SPLIT_SCHEMA,
    SPLIT_SEED,
    STORED_VALIDATION_METRICS,
    SUMMARY_SCHEMA,
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
    TRAINING_SEED,
    build_tpd_ner_v8_mprs_dch_model,
    formal_training_contract,
    ordered_identifier_sha256,
    variant_spec,
)


EVALUATION_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_pd_fa_v1"
FINAL_METRIC_COVERAGE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_final_metric_coverage_v1"
)
_BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
_ISOLATED_MODULE_NAME = "_sctransnet_tpd_ner_v8_mprs_dch_pd_fa_isolated"
_FORMAL_CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
_FORMAL_THRESHOLD_MIN = 0.01
_FORMAL_THRESHOLD_MAX = 0.99
_FORMAL_THRESHOLD_STEP = 0.01
_FORMAL_EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
_FORMAL_TAIL_LOGIT_STEP = 0.1
_EXACT_FORMAL_RUN_TAG = "formal800_exact_v1"
_FIXED_THRESHOLD_METRICS = (
    "pd",
    "fa",
    "miou",
    "false_objects_per_image",
)
_ANCHOR_TARGET_COUNT = 189
_PD_PRIMARY_MATCHED_TARGETS = 188
_MIOU_SELECTED_MATCHED_TARGETS = 187
_STRICT_BUDGET_MATCHED_TARGETS = 187
_OTHER_BUDGET_MATCHED_TARGETS = 188


def performance_gate_contract() -> Dict[str, Any]:
    """Return the result-independent absolute and paired acceptance gates."""

    return {
        "schema": "sctransnet_tpd_ner_v8_mprs_dch_performance_gates_v1",
        "anchor_target_count": _ANCHOR_TARGET_COUNT,
        "pd_primary_fixed_threshold_0_5": {
            "minimum_matched_targets": _PD_PRIMARY_MATCHED_TARGETS,
            "minimum_pd": (
                _PD_PRIMARY_MATCHED_TARGETS / _ANCHOR_TARGET_COUNT
            ),
            "maximum_fa": 1e-6,
            "minimum_miou": 0.933647,
        },
        "miou_selected_fixed_threshold_0_5": {
            "minimum_miou": 0.946542,
            "minimum_matched_targets": _MIOU_SELECTED_MATCHED_TARGETS,
            "minimum_pd": (
                _MIOU_SELECTED_MATCHED_TARGETS / _ANCHOR_TARGET_COUNT
            ),
            "maximum_fa": 1e-6,
        },
        "pd_at_fa_budget": {
            "1e-06": {
                "minimum_matched_targets": _STRICT_BUDGET_MATCHED_TARGETS,
                "minimum_pd": (
                    _STRICT_BUDGET_MATCHED_TARGETS / _ANCHOR_TARGET_COUNT
                ),
            },
            **{
                f"{budget:.10g}": {
                    "minimum_matched_targets": (
                        _OTHER_BUDGET_MATCHED_TARGETS
                    ),
                    "minimum_pd": (
                        _OTHER_BUDGET_MATCHED_TARGETS
                        / _ANCHOR_TARGET_COUNT
                    ),
                }
                for budget in FA_BUDGETS[1:]
            },
        },
        "relay_on_paired_budget_gate": {
            "reference": TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
            "candidate": TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
            "minimum_non_inferior_budget_count": 4,
            "minimum_strictly_better_budget_count": 1,
            "budget_count": 5,
        },
        "failure_action": (
            "return to code/training optimization; do not claim final success"
        ),
    }


def expected_run_identity(variant: str) -> Dict[str, Any]:
    """Return the only valid run identity for one formal candidate."""

    spec = variant_spec(variant)
    return {
        "schema": ENTRY_SCHEMA,
        "run_id": (
            f"{RUN_ID_PREFIX}{DATASET}:{variant}:"
            f"seed-{TRAINING_SEED}:split-{SPLIT_SEED}:{FORMAL_RUN_TAG}"
        ),
        "candidate_family": CANDIDATE_FAMILY,
        "dataset": DATASET,
        "variant": variant,
        "comparison_role": spec["comparison_role"],
        "parent_variant": spec["parent_variant"],
        "relay_enabled": spec["relay_enabled"],
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "run_tag": FORMAL_RUN_TAG,
    }


def evaluator_contract() -> Dict[str, Any]:
    """Describe the frozen formal evaluation and comparison contract."""

    return {
        "schema": EVALUATION_SCHEMA,
        "candidate_family": CANDIDATE_FAMILY,
        "formal_variants": list(FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS),
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": FORMAL_EPOCHS,
        "checkpoints": list(_FORMAL_CHECKPOINT_ROLES),
        "fixed_threshold": 0.5,
        "required_fixed_threshold_metrics": list(_FIXED_THRESHOLD_METRICS),
        "fa_budgets": list(FA_BUDGETS),
        "required_budget_metric": "pd",
        "preregistered_performance_gates": performance_gate_contract(),
        "main_comparison": [
            "baseline_sctransnet_external_same_split_reference",
            TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
            TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
        ],
        "historical_reference_not_formal_comparison_column": (
            "tpd_clean_v8_mprs_dch_full_external_same_split_reference"
        ),
        "required_control": TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
        "required_ablation": TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
        "preferred_training_entry": (
            "experiments.train_tpd_ner_v8_mprs_dch_exact"
        ),
        "accepted_training_artifact_modes": [
            "exact_resume_primary",
            "ordinary_compatibility",
        ],
        "metric_core": "experiments.evaluate_pd_fa_sweep",
        "closed_interval_core": (
            "experiments.evaluate_tpd_clean_v6_pd_fa."
            "adaptive_thresholds_closed_interval"
        ),
        "matching_or_metric_override": False,
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
        "official_test_accessed": False,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "determinism": dict(DETERMINISM_SETTINGS),
    }


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


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
    """Require semantic equality for two JSON-compatible artifact views.

    JSON files materialize tuples as arrays while ``torch.save`` preserves
    tuples.  The exact runner's canonicalizer intentionally maps both Python
    sequence representations to the same JSON array while retaining strict
    element order, length, scalar values, keys, and finite-number checks.
    """

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


def _require_finite_number(location: str, value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be a finite number, got {value!r}")
    return float(value)


def _require_nonnegative_integer(location: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{location} must be a non-negative integer, got {value!r}")
    return value


def _validate_protocol_arguments(
    arguments: Mapping[str, Any],
    variant: str,
) -> None:
    expected = {
        "dataset": DATASET,
        "variant": variant,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": FORMAL_EPOCHS,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "val_fraction": 0.20,
        "eval_every": 1,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "amp": False,
        "max_train_images": None,
        "max_val_images": None,
        "run_tag": FORMAL_RUN_TAG,
    }
    for name, value in expected.items():
        _require_equal(f"protocol.arguments.{name}", arguments.get(name), value)


def _validate_ordered_split(split: Mapping[str, Any]) -> None:
    expected_counts = {
        "full_official_train_count": 663,
        "used_train_count": 530,
        "used_val_count": 133,
    }
    for field, expected in expected_counts.items():
        _require_equal(f"split.{field}", split.get(field), expected)
    for field, checksum_field in (
        ("used_train_ids", "ordered_used_train_sha256"),
        ("used_val_ids", "ordered_used_val_sha256"),
    ):
        identifiers = split.get(field)
        if not isinstance(identifiers, list) or not all(
            isinstance(identifier, str) for identifier in identifiers
        ):
            raise ValueError(f"split.{field} must be a string list")
        count_field = field.removesuffix("_ids") + "_count"
        _require_equal(
            f"split.{count_field} versus len(split.{field})",
            split.get(count_field),
            len(identifiers),
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"split.{field} contains duplicate identifiers")
        _require_equal(
            f"split.{checksum_field}",
            split.get(checksum_field),
            ordered_identifier_sha256(identifiers),
        )
    if set(split["used_train_ids"]) & set(split["used_val_ids"]):
        raise ValueError("formal train and validation identifier lists overlap")


def _validate_ordinary_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> Dict[str, Any]:
    """Audit artifacts written by the ordinary compatibility trainer."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    if checkpoint_name not in _FORMAL_CHECKPOINT_ROLES:
        raise ValueError(
            "formal evaluation accepts only best.pth.tar or best_miou.pth.tar"
        )
    checkpoint_path = (run_dir / checkpoint_name).resolve()
    if checkpoint_path.parent != run_dir:
        raise ValueError("checkpoint must be directly inside the run directory")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    protocol = _load_json_object(run_dir / "protocol.json")
    split = _load_json_object(run_dir / "split.json")
    summary = _load_json_object(run_dir / "summary.json")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")

    _require_equal("protocol.schema", protocol.get("schema"), ENTRY_SCHEMA)
    _require_equal("split.schema", split.get("schema"), SPLIT_SCHEMA)
    _require_equal("summary.schema", summary.get("schema"), SUMMARY_SCHEMA)
    _require_equal("checkpoint.schema", checkpoint.get("schema"), CHECKPOINT_SCHEMA)
    _require_equal("summary.status", summary.get("status"), "complete")

    arguments = protocol.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("protocol.arguments must be an object")
    variant = arguments.get("variant")
    if variant not in FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS:
        raise ValueError(f"protocol variant is not formal: {variant!r}")
    variant = str(variant)
    _validate_protocol_arguments(arguments, variant)
    identity = expected_run_identity(variant)

    for artifact_name, artifact in {
        "protocol": protocol,
        "split": split,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"{artifact_name}.run_identity",
            artifact.get("run_identity"),
            identity,
        )
        _require_equal(
            f"{artifact_name}.official_test_accessed",
            artifact.get("official_test_accessed"),
            False,
        )

    training_contract = formal_training_contract()
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"{artifact_name}.training_contract",
            artifact.get("training_contract"),
            training_contract,
        )
        _require_equal(
            f"{artifact_name}.stored_validation_metrics",
            artifact.get("stored_validation_metrics"),
            list(STORED_VALIDATION_METRICS),
        )

    _require_equal("split.dataset", split.get("dataset"), DATASET)
    _require_equal("split.split_seed", split.get("split_seed"), SPLIT_SEED)
    _validate_ordered_split(split)

    for artifact_name, artifact in {
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(f"{artifact_name}.variant", artifact.get("variant"), variant)
        _require_equal(
            f"{artifact_name}.dataset",
            artifact.get("dataset"),
            DATASET,
        )
        _require_equal(
            f"{artifact_name}.seed",
            artifact.get("seed"),
            TRAINING_SEED,
        )
    _require_equal(
        "checkpoint.split_seed",
        checkpoint.get("split_seed"),
        SPLIT_SEED,
    )
    _require_equal(
        "summary.selection_source",
        summary.get("selection_source"),
        "internal_validation_only",
    )
    _require_equal(
        "checkpoint.selection_source",
        checkpoint.get("selection_source"),
        "internal_validation_only",
    )
    _require_equal(
        "checkpoint.six_output_training_semantics",
        checkpoint.get("six_output_training_semantics"),
        True,
    )

    model_records = {
        "protocol": protocol.get("model"),
        "summary": summary.get("model"),
        "checkpoint": checkpoint.get("model_metadata"),
    }
    if not all(isinstance(record, Mapping) for record in model_records.values()):
        raise ValueError("protocol/summary/checkpoint model metadata must be objects")
    architecture_ids = {
        name: record.get("architecture_id")
        for name, record in model_records.items()
    }
    architecture_id = architecture_ids["protocol"]
    if (
        not isinstance(architecture_id, str)
        or len(architecture_id) != 64
        or any(value != architecture_id for value in architecture_ids.values())
    ):
        raise ValueError(
            f"model architecture identity mismatch: {architecture_ids!r}"
        )
    spec = variant_spec(variant)
    for name, record in model_records.items():
        _require_equal(f"{name}.model.variant", record.get("variant"), variant)
        _require_equal(
            f"{name}.model.comparison_role",
            record.get("comparison_role"),
            spec["comparison_role"],
        )
        _require_equal(
            f"{name}.model.relay_enabled",
            record.get("relay_enabled"),
            spec["relay_enabled"],
        )

    expected_role = _FORMAL_CHECKPOINT_ROLES[checkpoint_name]
    _require_equal(
        "checkpoint.checkpoint_role",
        checkpoint.get("checkpoint_role"),
        expected_role,
    )
    expected_checkpoint_identity = {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "run_id": identity["run_id"],
        "variant": variant,
        "comparison_role": spec["comparison_role"],
        "relay_enabled": spec["relay_enabled"],
        "architecture_id": architecture_id,
        "checkpoint_role": expected_role,
        "checkpoint_filename": checkpoint_name,
    }
    _require_equal(
        "checkpoint.checkpoint_identity",
        checkpoint.get("checkpoint_identity"),
        expected_checkpoint_identity,
    )

    validation_metrics = checkpoint.get("validation_metrics")
    if not isinstance(validation_metrics, Mapping):
        raise ValueError("checkpoint.validation_metrics must be an object")
    missing_metrics = [
        name for name in STORED_VALIDATION_METRICS if name not in validation_metrics
    ]
    if missing_metrics:
        raise ValueError(
            "checkpoint.validation_metrics is incomplete: "
            f"missing={missing_metrics}"
        )
    for name in STORED_VALIDATION_METRICS:
        _require_finite_number(
            f"checkpoint.validation_metrics.{name}",
            validation_metrics[name],
        )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint.state_dict must be a non-empty mapping")

    expected_run_name = (
        f"seed_{TRAINING_SEED}_{FORMAL_RUN_TAG}"
    )
    _require_equal("run directory name", run_dir.name, expected_run_name)
    _require_equal("run directory variant", run_dir.parent.name, variant)
    _require_equal("run directory dataset", run_dir.parent.parent.name, DATASET)

    return {
        "training_artifact_mode": "ordinary_compatibility",
        "run_identity": identity,
        "variant": variant,
        "checkpoint_identity": expected_checkpoint_identity,
        "checkpoint_filename": checkpoint_name,
        "checkpoint_role": expected_role,
        "architecture_id": architecture_id,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(location: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _validate_exact_protocol_arguments(
    arguments: Mapping[str, Any],
    variant: str,
) -> None:
    candidate = exact_trainer.candidate_contract(variant)
    expected = {
        "dataset": DATASET,
        "variant": variant,
        "parent_variant": candidate["parent_variant"],
        "relay_enabled": candidate["relay_enabled"],
        "relay_width": exact_trainer.RELAY_WIDTH,
        "relay_initialization_seed": (
            exact_trainer.RELAY_INITIALIZATION_SEED
        ),
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": FORMAL_EPOCHS,
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
        "eps": exact_trainer.FORMAL_EPS,
        "amp": False,
        "allow_cpu_smoke": False,
        "max_train_images": None,
        "max_val_images": None,
        "run_tag": _EXACT_FORMAL_RUN_TAG,
        "device": "cuda:0",
    }
    for name, value in expected.items():
        _require_equal(
            f"exact protocol.arguments.{name}",
            arguments.get(name),
            value,
        )


def _validate_exact_run_identity(
    identity: Any,
    *,
    variant: str,
    split: Mapping[str, Any],
) -> Dict[str, Any]:
    value = exact_trainer._require_ner_run_identity(
        identity,
        label="formal evaluator exact identity",
        expected_variant=variant,
    )
    _require_equal(
        "exact run_identity.schema",
        value.get("schema"),
        exact_runner.RUN_IDENTITY_SCHEMA,
    )
    expected_run_id = (
        f"{exact_trainer.RUN_ID_PREFIX}{DATASET}:{variant}:"
        f"seed-{TRAINING_SEED}:{_EXACT_FORMAL_RUN_TAG}"
    )
    _require_equal("exact run_identity.run_id", value.get("run_id"), expected_run_id)
    _require_equal("exact run_identity.dataset", value.get("dataset"), DATASET)
    _require_equal("exact run_identity.seed", value.get("seed"), TRAINING_SEED)
    _require_equal(
        "exact run_identity.split_seed",
        value.get("split_seed"),
        SPLIT_SEED,
    )
    for field in (
        "architecture_id",
        "builder_manifest_sha256",
        "split_sha256",
        "data_sha256",
        "contract_sha256",
    ):
        _require_sha256(f"exact run_identity.{field}", value.get(field))

    training = value.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError("exact run_identity.training_contract must be an object")
    training_expected = {
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "amp": False,
        "total_epochs": FORMAL_EPOCHS,
        "eval_interval": 1,
    }
    for name, expected in training_expected.items():
        _require_equal(
            f"exact run_identity.training_contract.{name}",
            training.get(name),
            expected,
        )
    deep_supervision = training.get("deep_supervision")
    if not isinstance(deep_supervision, Mapping):
        raise ValueError("exact deep_supervision contract is missing")
    for name, expected in {
        "enabled": True,
        "expected_outputs": 6,
        "training_uses_all_outputs": True,
        "validation_uses_final_output": True,
    }.items():
        _require_equal(
            f"exact deep_supervision.{name}",
            deep_supervision.get(name),
            expected,
        )
    loss = training.get("loss")
    if not isinstance(loss, Mapping):
        raise ValueError("exact loss contract is missing")
    for name, expected in {
        "input": "post_sigmoid_probability",
        "aggregate": "sum",
        "compute_dtype": "float32",
    }.items():
        _require_equal(f"exact loss.{name}", loss.get(name), expected)
    metric_config = training.get("metric_config")
    if not isinstance(metric_config, Mapping):
        raise ValueError("exact metric_config is missing")
    for name, expected in {
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "validation_batch_size": 1,
        "official_test_accessed": False,
    }.items():
        _require_equal(
            f"exact metric_config.{name}",
            metric_config.get(name),
            expected,
        )

    fingerprints = value.get("ordered_split_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ValueError("exact ordered_split_fingerprints is missing")
    split_fields = {
        "full_train": "full_internal_train_ids",
        "full_validation": "full_internal_val_ids",
        "train": "used_train_ids",
        "validation": "used_val_ids",
    }
    for fingerprint_name, split_field in split_fields.items():
        identifiers = split.get(split_field)
        if not isinstance(identifiers, list) or not all(
            isinstance(identifier, str) for identifier in identifiers
        ):
            raise ValueError(f"split.{split_field} must be a string list")
        expected_fingerprint = exact_runner.OrderedFingerprint.from_values(
            fingerprint_name,
            identifiers,
        ).normalized()
        _require_equal(
            f"exact ordered_split_fingerprints.{fingerprint_name}",
            fingerprints.get(fingerprint_name),
            expected_fingerprint,
        )
    _require_equal(
        "exact run_identity.split_sha256",
        value.get("split_sha256"),
        _canonical_sha256(fingerprints),
    )

    data_fingerprints = value.get("ordered_data_fingerprints")
    if not isinstance(data_fingerprints, Mapping) or not data_fingerprints:
        raise ValueError("exact ordered_data_fingerprints is missing")
    _require_equal(
        "exact ordered_data_fingerprints keys",
        set(data_fingerprints),
        {
            "official_training_data",
            "train_samples",
            "validation_samples",
            "normalization",
        },
    )
    for name, record in data_fingerprints.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"exact data fingerprint {name!r} is invalid")
        _require_sha256(
            f"exact ordered_data_fingerprints.{name}.sha256",
            record.get("sha256"),
        )
    _require_equal(
        "exact run_identity.data_sha256",
        value.get("data_sha256"),
        _canonical_sha256(data_fingerprints),
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
        "exact run_identity.contract_sha256",
        value.get("contract_sha256"),
        _canonical_sha256(identity_contract),
    )
    return value


def _validate_exact_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> Dict[str, Any]:
    """Audit the preferred exact-resume protocol and compatibility view."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    if checkpoint_name not in _FORMAL_CHECKPOINT_ROLES:
        raise ValueError(
            "formal evaluation accepts only best.pth.tar or best_miou.pth.tar"
        )
    checkpoint_path = (run_dir / checkpoint_name).resolve()
    if checkpoint_path.parent != run_dir:
        raise ValueError("checkpoint must be directly inside the run directory")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    protocol = _load_json_object(run_dir / "protocol.json")
    split = _load_json_object(run_dir / "split.json")
    summary = _load_json_object(run_dir / "summary.json")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("exact compatibility checkpoint must be a dictionary")

    _require_equal(
        "exact protocol.schema",
        protocol.get("schema"),
        exact_trainer.ENTRY_SCHEMA,
    )
    _require_equal(
        "exact summary.schema",
        summary.get("schema"),
        exact_trainer.COMPLETION_SUMMARY_SCHEMA,
    )
    _require_equal(
        "exact checkpoint.schema",
        checkpoint.get("schema"),
        exact_trainer.CHECKPOINT_SCHEMA,
    )
    _require_equal(
        "exact checkpoint.derived_schema",
        checkpoint.get("derived_schema"),
        exact_runner.DERIVED_CHECKPOINT_SCHEMA,
    )
    _require_equal("exact summary.status", summary.get("status"), "complete")

    arguments = protocol.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("exact protocol.arguments must be an object")
    variant = arguments.get("variant")
    if variant not in FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS:
        raise ValueError(f"exact protocol variant is not formal: {variant!r}")
    variant = str(variant)
    checkpoint = exact_trainer.require_evaluator_checkpoint_payload(
        checkpoint,
        expected_variant=variant,
    )
    _validate_exact_protocol_arguments(arguments, variant)
    _require_equal(
        "exact protocol.formal_contract",
        protocol.get("formal_contract"),
        exact_trainer.formal_contract(),
    )

    _require_equal("exact split.dataset", split.get("dataset"), DATASET)
    _require_equal("exact split.split_seed", split.get("split_seed"), SPLIT_SEED)
    _require_equal(
        "exact split.full_official_train_count",
        split.get("full_official_train_count"),
        663,
    )
    _require_equal("exact split.used_train_count", split.get("used_train_count"), 530)
    _require_equal("exact split.used_val_count", split.get("used_val_count"), 133)
    _require_equal(
        "exact split.official_test_accessed",
        split.get("official_test_accessed"),
        False,
    )

    identity = _validate_exact_run_identity(
        protocol.get("run_identity"),
        variant=variant,
        split=split,
    )
    for artifact_name, artifact in {
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"exact {artifact_name}.run_identity",
            artifact.get("run_identity"),
            identity,
        )
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"exact {artifact_name}.official_test_accessed",
            artifact.get("official_test_accessed"),
            False,
        )
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
    }.items():
        _require_equal(
            f"exact {artifact_name}.stored_validation_metrics",
            artifact.get("stored_validation_metrics"),
            list(exact_trainer.STORED_VALIDATION_METRICS),
        )
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"exact {artifact_name}.selection_source",
            artifact.get("selection_source"),
            "internal_validation_only",
        )

    candidate = exact_trainer.candidate_contract(variant)
    relay_identity = protocol.get("relay_identity")
    if not isinstance(relay_identity, Mapping):
        raise ValueError("exact protocol.relay_identity must be an object")
    for name, expected in {
        "source": "candidate_variant_suffix",
        "parent_variant": candidate["parent_variant"],
        "enabled": candidate["relay_enabled"],
        "width": exact_trainer.RELAY_WIDTH,
        "initialization_seed": exact_trainer.RELAY_INITIALIZATION_SEED,
    }.items():
        _require_equal(
            f"exact protocol.relay_identity.{name}",
            relay_identity.get(name),
            expected,
        )
    scalar_expected = {
        "variant": variant,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "parent_variant": candidate["parent_variant"],
        "relay_enabled": candidate["relay_enabled"],
        "relay_width": exact_trainer.RELAY_WIDTH,
    }
    for artifact_name, artifact in {
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        for name, expected in scalar_expected.items():
            _require_equal(
                f"exact {artifact_name}.{name}",
                artifact.get(name),
                expected,
            )

    model_records = {
        "protocol": protocol.get("model"),
        "summary": summary.get("model"),
        "checkpoint": checkpoint.get("model_metadata"),
    }
    if not all(isinstance(record, Mapping) for record in model_records.values()):
        raise ValueError("exact model metadata views must be objects")
    reference_model = dict(model_records["protocol"])
    for name, record in model_records.items():
        _require_canonical_json_equal(
            f"exact {name}.model",
            dict(record),
            reference_model,
        )
        _require_equal(f"exact {name}.model.variant", record.get("variant"), variant)
        _require_equal(
            f"exact {name}.model.relay_enabled",
            record.get("relay_enabled"),
            candidate["relay_enabled"],
        )

    expected_role = _FORMAL_CHECKPOINT_ROLES[checkpoint_name]
    _require_equal(
        "exact checkpoint.checkpoint_role",
        checkpoint.get("checkpoint_role"),
        expected_role,
    )
    expected_checkpoint_identity = {
        "schema": exact_trainer.CHECKPOINT_IDENTITY_SCHEMA,
        "variant": variant,
        "parent_variant": candidate["parent_variant"],
        "relay_enabled": candidate["relay_enabled"],
        "relay_width": exact_trainer.RELAY_WIDTH,
        "run_id": identity["run_id"],
        "architecture_id": identity["architecture_id"],
        "builder_manifest_sha256": identity["builder_manifest_sha256"],
    }
    _require_equal(
        "exact checkpoint.checkpoint_identity",
        checkpoint.get("checkpoint_identity"),
        expected_checkpoint_identity,
    )
    validation_metrics = checkpoint.get("validation_metrics")
    if not isinstance(validation_metrics, Mapping):
        raise ValueError("exact checkpoint.validation_metrics must be an object")
    missing_metrics = [
        name
        for name in exact_trainer.STORED_VALIDATION_METRICS
        if name not in validation_metrics
    ]
    if missing_metrics:
        raise ValueError(
            "exact checkpoint.validation_metrics is incomplete: "
            f"missing={missing_metrics}"
        )
    for name in exact_trainer.STORED_VALIDATION_METRICS:
        _require_finite_number(
            f"exact checkpoint.validation_metrics.{name}",
            validation_metrics[name],
        )
    digest_fields = {
        "state_dict": "state_dict_sha256",
        "optimizer": "optimizer_state_sha256",
        "scaler": "scaler_state_sha256",
    }
    for component, digest_field in digest_fields.items():
        state = checkpoint.get(component)
        if not isinstance(state, Mapping):
            raise ValueError(f"exact checkpoint.{component} must be a mapping")
        if component == "state_dict" and not state:
            raise ValueError("exact checkpoint.state_dict must be non-empty")
        expected_digest = exact_runner._state_content_sha256(
            state,
            f"formal evaluator exact {component}",
        )
        _require_equal(
            f"exact checkpoint.{digest_field}",
            checkpoint.get(digest_field),
            expected_digest,
        )
    _require_sha256(
        "exact checkpoint.source_exact_checkpoint_sha256",
        checkpoint.get("source_exact_checkpoint_sha256"),
    )

    split_hashes = split.get("hashes")
    if not isinstance(split_hashes, Mapping):
        raise ValueError("exact split.hashes must be an object")
    _require_equal("exact summary.split_hashes", summary.get("split_hashes"), split_hashes)
    _require_equal(
        "exact checkpoint.split_hashes",
        checkpoint.get("split_hashes"),
        split_hashes,
    )

    expected_run_name = f"seed_{TRAINING_SEED}_{_EXACT_FORMAL_RUN_TAG}"
    _require_equal("exact run directory name", run_dir.name, expected_run_name)
    _require_equal("exact run directory variant", run_dir.parent.name, variant)
    _require_equal("exact run directory dataset", run_dir.parent.parent.name, DATASET)
    return {
        "training_artifact_mode": "exact_resume_primary",
        "run_identity": identity,
        "variant": variant,
        "checkpoint_identity": expected_checkpoint_identity,
        "checkpoint_filename": checkpoint_name,
        "checkpoint_role": expected_role,
        "architecture_id": identity["architecture_id"],
    }


def validate_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> Dict[str, Any]:
    """Strictly accept exact-primary or ordinary-compatibility artifacts."""

    protocol = _load_json_object(Path(run_dir).resolve() / "protocol.json")
    schema = protocol.get("schema")
    if schema == exact_trainer.ENTRY_SCHEMA:
        return _validate_exact_run_artifacts(run_dir, checkpoint_name)
    if schema == ENTRY_SCHEMA:
        return _validate_ordinary_run_artifacts(run_dir, checkpoint_name)
    raise ValueError(
        "run protocol is neither the exact-primary nor ordinary-compatible "
        "V8-MPRS-DCH NER schema"
    )


def preflight_requested_artifacts(
    argv: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Resolve the requested run/checkpoint without consuming other CLI flags."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth.tar")
    args, _ = parser.parse_known_args(
        list(sys.argv[1:] if argv is None else argv)
    )
    return validate_run_artifacts(args.run_dir, args.checkpoint)


def _validate_formal_evaluator_args(args: argparse.Namespace) -> None:
    if args.expected_epochs not in (None, FORMAL_EPOCHS):
        raise ValueError(
            f"formal evaluator requires expected_epochs={FORMAL_EPOCHS}"
        )
    args.expected_epochs = FORMAL_EPOCHS
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("formal evaluator device must be cpu or cuda:0")
    expected_values = {
        "threshold_min": _FORMAL_THRESHOLD_MIN,
        "threshold_max": _FORMAL_THRESHOLD_MAX,
        "threshold_step": _FORMAL_THRESHOLD_STEP,
        "tail_logit_step": _FORMAL_TAIL_LOGIT_STEP,
    }
    for name, expected in expected_values.items():
        _require_equal(f"evaluator argument {name}", getattr(args, name), expected)
    _require_equal(
        "evaluator argument extra_thresholds",
        tuple(args.extra_thresholds),
        _FORMAL_EXTRA_THRESHOLDS,
    )
    _require_equal(
        "evaluator argument fa_budgets",
        tuple(args.fa_budgets),
        FA_BUDGETS,
    )
    if args.match_radius not in (None, 3.0):
        raise ValueError("formal evaluator match_radius must be omitted or 3.0")
    if args.tiny_area not in (None, 9):
        raise ValueError("formal evaluator tiny_area must be omitted or 9")


def finalize_evaluation_output(
    payload: Mapping[str, Any],
    artifact_audit: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Require all formal final metrics and attach explicit coverage metadata."""

    ready = dict(payload)
    _require_equal("evaluation.dataset", ready.get("dataset"), DATASET)
    variant = ready.get("variant")
    if variant not in FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS:
        raise ValueError(f"evaluation variant is not formal: {variant!r}")
    variant = str(variant)
    _require_equal("evaluation.seed", ready.get("seed"), TRAINING_SEED)
    _require_equal("evaluation.split_seed", ready.get("split_seed"), SPLIT_SEED)
    _require_equal(
        "evaluation.official_test_accessed",
        ready.get("official_test_accessed"),
        False,
    )

    fixed = ready.get("fixed_threshold_0_5")
    if not isinstance(fixed, Mapping):
        raise ValueError("evaluation.fixed_threshold_0_5 must be an object")
    _require_equal("fixed_threshold_0_5.threshold", fixed.get("threshold"), 0.5)
    fixed_metrics: Dict[str, float | int] = {}
    for name in _FIXED_THRESHOLD_METRICS:
        value = _require_finite_number(f"fixed_threshold_0_5.{name}", fixed.get(name))
        if name == "false_objects_per_image":
            if value < 0.0:
                raise ValueError(f"fixed_threshold_0_5.{name} must be non-negative")
        elif not 0.0 <= value <= 1.0:
            raise ValueError(f"fixed_threshold_0_5.{name} must lie in [0, 1]")
        fixed_metrics[name] = fixed[name]
    fixed_target_count = _require_nonnegative_integer(
        "fixed_threshold_0_5.target_count",
        fixed.get("target_count"),
    )
    fixed_matched_target_count = _require_nonnegative_integer(
        "fixed_threshold_0_5.matched_target_count",
        fixed.get("matched_target_count"),
    )
    _require_equal(
        "fixed_threshold_0_5.target_count",
        fixed_target_count,
        _ANCHOR_TARGET_COUNT,
    )
    if fixed_matched_target_count > fixed_target_count:
        raise ValueError(
            "fixed_threshold_0_5.matched_target_count exceeds target_count"
        )

    threshold_configuration = ready.get("threshold_configuration")
    if not isinstance(threshold_configuration, Mapping):
        raise ValueError("evaluation.threshold_configuration must be an object")
    _require_equal(
        "threshold_configuration.fa_budgets",
        tuple(threshold_configuration.get("fa_budgets", ())),
        FA_BUDGETS,
    )
    budget_points = ready.get("best_points_under_fa_budget")
    if not isinstance(budget_points, Mapping):
        raise ValueError("evaluation.best_points_under_fa_budget must be an object")
    expected_budget_keys = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
    _require_equal(
        "best_points_under_fa_budget keys",
        set(budget_points),
        set(expected_budget_keys),
    )
    pd_by_budget: Dict[str, Dict[str, float | int]] = {}
    for budget, key in zip(FA_BUDGETS, expected_budget_keys):
        point = budget_points[key]
        if not isinstance(point, Mapping):
            raise ValueError(
                f"best_points_under_fa_budget[{key!r}] must be an object"
            )
        pd_value = _require_finite_number(f"budget[{key}].pd", point.get("pd"))
        fa_value = _require_finite_number(f"budget[{key}].fa", point.get("fa"))
        threshold = _require_finite_number(
            f"budget[{key}].threshold",
            point.get("threshold"),
        )
        target_count = _require_nonnegative_integer(
            f"budget[{key}].target_count",
            point.get("target_count"),
        )
        matched_target_count = _require_nonnegative_integer(
            f"budget[{key}].matched_target_count",
            point.get("matched_target_count"),
        )
        _require_equal(
            f"budget[{key}].target_count",
            target_count,
            _ANCHOR_TARGET_COUNT,
        )
        if matched_target_count > target_count:
            raise ValueError(
                f"budget[{key}].matched_target_count exceeds target_count"
            )
        if not 0.0 <= pd_value <= 1.0:
            raise ValueError(f"budget[{key}].pd must lie in [0, 1]")
        if not 0.0 <= fa_value <= budget:
            raise ValueError(
                f"budget[{key}].fa={fa_value} exceeds budget={budget}"
            )
        pd_by_budget[key] = {
            "budget": budget,
            "pd": point["pd"],
            "achieved_fa": point["fa"],
            "threshold": point["threshold"],
            "matched_target_count": matched_target_count,
            "target_count": target_count,
        }

    checkpoint_filename = Path(str(ready.get("checkpoint"))).name
    checkpoint_role = ready.get("checkpoint_role")
    if checkpoint_filename not in _FORMAL_CHECKPOINT_ROLES:
        raise ValueError("evaluation checkpoint filename is not formal")
    _require_equal(
        "evaluation checkpoint role",
        checkpoint_role,
        _FORMAL_CHECKPOINT_ROLES[checkpoint_filename],
    )
    if artifact_audit is not None:
        _require_equal(
            "preflight variant",
            artifact_audit.get("variant"),
            variant,
        )
        _require_equal(
            "preflight checkpoint filename",
            artifact_audit.get("checkpoint_filename"),
            checkpoint_filename,
        )
        _require_equal(
            "preflight checkpoint role",
            artifact_audit.get("checkpoint_role"),
            checkpoint_role,
        )
        checkpoint_sha256 = _require_sha256(
            "evaluation.checkpoint_sha256",
            ready.get("checkpoint_sha256"),
        )
        output_run_identity = dict(artifact_audit["run_identity"])
        training_artifact_mode = str(
            artifact_audit["training_artifact_mode"]
        )
    else:
        checkpoint_sha256 = ready.get("checkpoint_sha256")
        output_run_identity = expected_run_identity(variant)
        training_artifact_mode = "ordinary_compatibility_unverified_fixture"

    gate_contract = performance_gate_contract()
    fixed_gate_name = (
        "pd_primary_fixed_threshold_0_5"
        if checkpoint_role == "best_validation_pd_primary"
        else "miou_selected_fixed_threshold_0_5"
    )
    fixed_gate = gate_contract[fixed_gate_name]
    fixed_gate_checks = {
        "matched_targets": (
            fixed_matched_target_count
            >= int(fixed_gate["minimum_matched_targets"])
        ),
        "pd": (
            float(fixed["pd"]) >= float(fixed_gate["minimum_pd"])
        ),
        "fa": float(fixed["fa"]) <= float(fixed_gate["maximum_fa"]),
        "miou": (
            float(fixed["miou"]) >= float(fixed_gate["minimum_miou"])
        ),
    }
    budget_gate_assessment: Dict[str, Dict[str, Any]] = {}
    for key, observed in pd_by_budget.items():
        requirement = gate_contract["pd_at_fa_budget"][key]
        checks = {
            "matched_targets": (
                int(observed["matched_target_count"])
                >= int(requirement["minimum_matched_targets"])
            ),
            "pd": (
                float(observed["pd"])
                >= float(requirement["minimum_pd"])
            ),
        }
        budget_gate_assessment[key] = {
            "required_matched_target_count": requirement[
                "minimum_matched_targets"
            ],
            "required_pd": requirement["minimum_pd"],
            "observed_matched_target_count": observed[
                "matched_target_count"
            ],
            "observed_target_count": observed["target_count"],
            "observed_pd": observed["pd"],
            "checks": checks,
            "passed": all(checks.values()),
        }
    absolute_gate_passed = all(fixed_gate_checks.values()) and all(
        assessment["passed"]
        for assessment in budget_gate_assessment.values()
    )
    paired_gate_status = (
        "requires_relay_off_and_relay_on_aggregate"
        if variant == TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON
        else "reference_control_not_applicable"
    )

    ready.update(
        {
            "schema": EVALUATION_SCHEMA,
            "run_identity": output_run_identity,
            "training_artifact_mode": training_artifact_mode,
            "comparison_role": variant_spec(variant)["comparison_role"],
            "source_checkpoint_identity": (
                dict(artifact_audit["checkpoint_identity"])
                if artifact_audit is not None
                else None
            ),
            "evaluated_checkpoint_identity": {
                "training_artifact_mode": training_artifact_mode,
                "filename": checkpoint_filename,
                "role": checkpoint_role,
                "sha256": checkpoint_sha256,
            },
            "artifact_identity_preflight_passed": artifact_audit is not None,
            "evaluator_contract": evaluator_contract(),
            "performance_gate_assessment": {
                "contract": gate_contract,
                "fixed_threshold_gate": fixed_gate_name,
                "fixed_threshold_observed": {
                    "matched_target_count": fixed_matched_target_count,
                    "target_count": fixed_target_count,
                    "pd": fixed["pd"],
                    "fa": fixed["fa"],
                    "miou": fixed["miou"],
                },
                "fixed_threshold_checks": fixed_gate_checks,
                "budget_checks": budget_gate_assessment,
                "absolute_checkpoint_gate_passed": absolute_gate_passed,
                "paired_relay_on_gate_status": paired_gate_status,
                "formal_success_claim_authorized": False,
            },
            "final_metric_coverage": {
                "schema": FINAL_METRIC_COVERAGE_SCHEMA,
                "fixed_threshold": 0.5,
                "fixed_threshold_0_5": fixed_metrics,
                "pd_at_fa_budget": pd_by_budget,
                "all_required_metrics_present": True,
            },
        }
    )
    return ready


def _load_isolated_base_evaluator(
    artifact_audit: Mapping[str, Any] | None = None,
) -> ModuleType:
    """Load and bind a private evaluator instance without shared mutations."""

    if not _BASE_EVALUATOR_PATH.is_file():
        raise FileNotFoundError(_BASE_EVALUATOR_PATH)
    spec = importlib.util.spec_from_file_location(
        _ISOLATED_MODULE_NAME,
        _BASE_EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot create evaluator module spec for {_BASE_EVALUATOR_PATH}"
        )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original_parse_args = evaluator.parse_args
    original_write_output_json = evaluator.write_output_json

    def bound_parse_args() -> argparse.Namespace:
        args = original_parse_args()
        _validate_formal_evaluator_args(args)
        return args

    def bound_write_output_json(
        path: Path,
        payload: Dict[str, Any],
        overwrite: bool,
    ) -> None:
        original_write_output_json(
            path,
            finalize_evaluation_output(payload, artifact_audit),
            overwrite,
        )

    evaluator.adaptive_thresholds = adaptive_thresholds_closed_interval
    evaluator.build_model = build_tpd_ner_v8_mprs_dch_model
    evaluator.parse_args = bound_parse_args
    evaluator.write_output_json = bound_write_output_json
    # The shared evaluator reports and hashes ``__file__`` as provenance.
    evaluator.__file__ = __file__
    return evaluator


def main() -> None:
    if FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS != (
        TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
        TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
    ):
        raise RuntimeError("unexpected formal V8-MPRS-DCH NER variants")
    argv = list(sys.argv[1:])
    configure_v8_inference(requested_device(argv))
    artifact_audit = preflight_requested_artifacts(argv)
    evaluator = _load_isolated_base_evaluator(artifact_audit)
    evaluator.main()


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "DETERMINISM_SETTINGS",
    "EVALUATION_SCHEMA",
    "FINAL_METRIC_COVERAGE_SCHEMA",
    "FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS",
    "LAST_FLOAT32_BELOW_ONE",
    "UPPER_BOUNDARY_THRESHOLD",
    "adaptive_thresholds_closed_interval",
    "build_tpd_ner_v8_mprs_dch_model",
    "configure_v8_inference",
    "evaluator_contract",
    "expected_run_identity",
    "finalize_evaluation_output",
    "main",
    "preflight_requested_artifacts",
    "requested_device",
    "validate_run_artifacts",
]


if __name__ == "__main__":
    main()
