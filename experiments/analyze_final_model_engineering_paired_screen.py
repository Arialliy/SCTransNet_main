#!/usr/bin/env python3
"""Descriptive paired-image screen for the two engineering B/D seeds.

The input is the completed eight-result evaluator manifest plus the four
seed-by-checkpoint-policy B/D pairs of lossless PredictionCache artifacts.
All point estimates are recomputed from the locked per-image statistic core.
The paired and seed->image hierarchical bootstraps are CPU-only.

This screen is deliberately not Gate M-train.  Seed 3407 is a known historical
pressure trajectory and the matrix contains only two fixed-parent engineering
seeds.  Therefore no result produced here can establish a paper-core or
stability claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import collect_final_model_validation_statistics as cache_core
from analysis import run_final_qfg_six_mode_audit as bootstrap_contract
from experiments import (
    evaluate_final_model_engineering_replication_pd_fa as evaluator,
)
from experiments import final_model_replication_exact_core as replication_core
from experiments import final_model_replication_seed_contract as seed_contract
from experiments import watch_final_model_engineering_replication as watcher


SCHEMA = "sctransnet_final_model_engineering_paired_screen_v1"
ACTION_SCHEMA = "sctransnet_final_model_engineering_paired_screen_action_v1"
BOOTSTRAP_SCHEMA = (
    "sctransnet_final_model_engineering_paired_image_bootstrap_v1"
)
HIERARCHICAL_SCHEMA = (
    "sctransnet_final_model_engineering_seed_image_bootstrap_v1"
)
FIXED_THRESHOLD = evaluator.FIXED_THRESHOLD
BOOTSTRAP_REPLICATES = bootstrap_contract.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = bootstrap_contract.BOOTSTRAP_SEED
SIMULTANEOUS_FAMILY_CI = bootstrap_contract.SIMULTANEOUS_FAMILY_CI
PER_METRIC_TWO_SIDED_CI = (
    bootstrap_contract.PER_METRIC_TWO_SIDED_CI
)
METRIC_KEYS = tuple(bootstrap_contract.METRIC_KEYS)
DELTA_ORIENTATION = "D_minus_B"
PRIMARY_SELECTION_ROLE = "primary_best_miou"
SECONDARY_SELECTION_ROLE = "secondary_best_pd"
SELECTION_ROLES = (
    PRIMARY_SELECTION_ROLE,
    SECONDARY_SELECTION_ROLE,
)
POLICY_LABELS = {
    PRIMARY_SELECTION_ROLE: "primary_each_arm_own_best_miou",
    SECONDARY_SELECTION_ROLE: "secondary_each_arm_own_best",
}
DEFAULT_MANIFEST = evaluator.default_manifest_path(
    watcher.DEFAULT_OUTPUT_ROOT
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "analysis/results/final_model_engineering_paired_screen_v1.json"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_POINT_AUDIT_FIELDS = (
    *METRIC_KEYS,
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "unmatched_predicted_object_count",
    "unmatched_predicted_pixels",
    "valid_pixel_count",
    "intersection",
    "union",
    "image_count",
)


class EngineeringPairedScreenError(ValueError):
    """The paired engineering screen evidence violates its contract."""


def _fail(message: str) -> None:
    raise EngineeringPairedScreenError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"value is not finite canonical JSON: {exc}")


def _sha256_file(path: Path, label: str) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(f"{label} must be a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256 digest")
    return value


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: observed={observed!r}, expected={expected!r}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{label} must be an array")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be finite")
    return float(value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _base_payload(
    *,
    status: str,
    decision: str,
    manifest_path: Path,
    missing: Sequence[Mapping[str, str]] = (),
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "scope": "fixed_parent_engineering_b_d_descriptive_screen_only",
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "engineering_trajectory_seeds": list(
            seed_contract.ENGINEERING_TRAJECTORY_SEEDS
        ),
        "seed_roles": {
            str(seed_contract.HISTORICAL_PRESSURE_SEED): (
                "known_historical_pressure_seed"
            ),
            str(seed_contract.DEPLOYMENT_HASH_SEED): (
                "deployment_artifact_hash_replication_seed"
            ),
        },
        "arms": {
            replication_core.ARM_B: replication_core.arm_definition(
                replication_core.ARM_B
            ).variant,
            replication_core.ARM_D: replication_core.arm_definition(
                replication_core.ARM_D
            ).variant,
        },
        "checkpoint_policy": {
            "primary": "each_arm_own_best_miou",
            "secondary": "each_arm_own_best",
            "top_level_route_uses": "primary_only",
            "cross_arm_shared_epoch_required": False,
        },
        "fixed_threshold": FIXED_THRESHOLD,
        "metric_family": list(METRIC_KEYS),
        "fa_budgets": list(evaluator.FA_BUDGETS),
        "bootstrap_contract": {
            "replicates": BOOTSTRAP_REPLICATES,
            "rng_seed": BOOTSTRAP_SEED,
            "per_metric_two_sided_confidence": (
                PER_METRIC_TWO_SIDED_CI
            ),
            "simultaneous_family_confidence": SIMULTANEOUS_FAMILY_CI,
            "method": "Bonferroni percentile intervals",
            "delta_orientation": DELTA_ORIENTATION,
        },
        "missing_artifacts": [dict(item) for item in missing],
        "errors": list(errors),
        "engineering_paired_route_met": None,
        "establishes_gate_m_train": False,
        "gates": {
            "M-train": {
                "status": "insufficient_evidence",
                "passed": None,
                "establishes_gate_m_train": False,
                "required_evidence": (
                    "pre-registered confirmatory same-seed full-pipeline "
                    "paired runs satisfying Gate M-train"
                ),
                "available_evidence": (
                    "two fixed-parent engineering child-trajectory seeds, "
                    "including known pressure seed 3407"
                ),
            }
        },
        "claim_boundary": {
            "descriptive_engineering_screen_only": True,
            "historical_pressure_seed_3407_included": True,
            "fixed_parent_engineering_seed_count": 2,
            "establishes_gate_m_train": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "full_pipeline_stability_supported": False,
            "multiseed_replication_supported": False,
            "official_test_accessed": False,
        },
    }


def _validate_metric_point(
    value: Any,
    *,
    label: str,
    expected_threshold: float | None = None,
) -> dict[str, Any]:
    point = _mapping(value, label)
    required = {
        "threshold",
        "pd",
        "matched_target_count",
        "target_count",
        "fa",
        "miou",
        "tiny_pd",
        "matched_tiny_target_count",
        "tiny_target_count",
        "false_objects_per_image",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
    missing = sorted(required - set(point))
    if missing:
        _fail(f"{label} is missing fields: {missing}")
    threshold = _finite(point["threshold"], f"{label}.threshold")
    if not 0.0 <= threshold <= 1.0:
        _fail(f"{label}.threshold must lie in [0, 1]")
    if expected_threshold is not None and threshold != expected_threshold:
        _fail(f"{label}.threshold differs from {expected_threshold}")
    normalized: dict[str, Any] = copy.deepcopy(dict(point))
    for name in METRIC_KEYS:
        normalized[name] = _finite(point[name], f"{label}.{name}")
    for name in (
        "matched_target_count",
        "target_count",
        "matched_tiny_target_count",
        "tiny_target_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    ):
        normalized[name] = _integer(point[name], f"{label}.{name}")
        if normalized[name] < 0:
            _fail(f"{label}.{name} must be non-negative")
    return normalized


def _validate_manifest(
    manifest: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, str, str], dict[str, Any]],
    dict[tuple[int, str], dict[str, Any]],
]:
    required_top = {
        "schema",
        "status",
        "scope",
        "result_count",
        "expected_result_count",
        "all_checkpoint_local_results_valid",
        "threshold_selection_scope",
        "cross_checkpoint_point_pooling",
        "fixed_threshold",
        "fa_budgets",
        "source_lock_sha256",
        "seed_contract_sha256",
        "validation_count",
        "validation_ids_sha256",
        "formal_gpu_binding_policy",
        "all_results_expected_physical_gpu_bound",
        "paired_checkpoint_group_count",
        "paired_checkpoint_groups",
        "gate_m_train_image_level_inputs_ready",
        "paired_confidence_intervals_computed",
        "paired_confidence_intervals_claimed",
        "official_test_accessed",
        "evaluation_source_binding",
        "results",
    }
    _equal("manifest fields", set(manifest), required_top)
    for label, observed, expected in (
        ("manifest schema", manifest.get("schema"), evaluator.MANIFEST_SCHEMA),
        ("manifest status", manifest.get("status"), "complete"),
        (
            "manifest scope",
            manifest.get("scope"),
            "fixed_parent_engineering_b_d_only",
        ),
        (
            "manifest result count",
            manifest.get("result_count"),
            evaluator.EXPECTED_SWEEP_COUNT,
        ),
        (
            "manifest expected result count",
            manifest.get("expected_result_count"),
            evaluator.EXPECTED_SWEEP_COUNT,
        ),
        (
            "all checkpoint-local results valid",
            manifest.get("all_checkpoint_local_results_valid"),
            True,
        ),
        (
            "threshold selection scope",
            manifest.get("threshold_selection_scope"),
            "single_checkpoint_only",
        ),
        (
            "cross-checkpoint point pooling",
            manifest.get("cross_checkpoint_point_pooling"),
            False,
        ),
        (
            "fixed threshold",
            manifest.get("fixed_threshold"),
            FIXED_THRESHOLD,
        ),
        (
            "Fa budgets",
            manifest.get("fa_budgets"),
            list(evaluator.FA_BUDGETS),
        ),
        (
            "validation count",
            manifest.get("validation_count"),
            evaluator.EXPECTED_VALIDATION_COUNT,
        ),
        (
            "all results expected physical GPU bound",
            manifest.get("all_results_expected_physical_gpu_bound"),
            True,
        ),
        (
            "paired checkpoint group count",
            manifest.get("paired_checkpoint_group_count"),
            4,
        ),
        (
            "image-level inputs ready",
            manifest.get("gate_m_train_image_level_inputs_ready"),
            True,
        ),
        (
            "paired intervals precomputed",
            manifest.get("paired_confidence_intervals_computed"),
            False,
        ),
        (
            "paired intervals preclaimed",
            manifest.get("paired_confidence_intervals_claimed"),
            False,
        ),
        (
            "official test accessed",
            manifest.get("official_test_accessed"),
            False,
        ),
    ):
        _equal(label, observed, expected)
    _sha256(
        manifest.get("source_lock_sha256"),
        "manifest source-lock SHA-256",
    )
    _sha256(
        manifest.get("seed_contract_sha256"),
        "manifest seed-contract SHA-256",
    )
    expected_gpu_policy = {
        "cpu_results_accepted": False,
        "arm_assignments": {
            arm: {
                "physical_gpu_index": evaluator.ARM_PHYSICAL_GPU_INDICES[
                    arm
                ],
                "physical_gpu_uuid": replication_core.arm_definition(
                    arm
                ).trainer.PHYSICAL_GPU_UUIDS[
                    str(evaluator.ARM_PHYSICAL_GPU_INDICES[arm])
                ],
                "logical_device": "cuda:0",
            }
            for arm in replication_core.SUPPORTED_ARMS
        },
    }
    _equal(
        "manifest formal GPU binding policy",
        manifest.get("formal_gpu_binding_policy"),
        expected_gpu_policy,
    )
    gpu_policy = _mapping(
        manifest.get("formal_gpu_binding_policy"),
        "manifest formal GPU binding policy",
    )
    if gpu_policy.get("cpu_results_accepted") is not False:
        _fail("manifest CPU-result acceptance flag must be false")
    if manifest.get("all_results_expected_physical_gpu_bound") is not True:
        _fail("manifest physical-GPU-bound result flag must be true")
    validation_ids_sha256 = _sha256(
        manifest.get("validation_ids_sha256"),
        "manifest validation-ID SHA-256",
    )
    _mapping(
        manifest.get("evaluation_source_binding"),
        "manifest evaluation source binding",
    )
    evaluation_source_binding = _mapping(
        manifest.get("evaluation_source_binding"),
        "manifest evaluation source binding",
    )
    _equal(
        "manifest evaluation source-binding roles",
        set(evaluation_source_binding),
        set(evaluator.FROZEN_CORE_PATHS) | {"checkpoint_local_adapter"},
    )
    for role, raw_binding in evaluation_source_binding.items():
        binding = _mapping(
            raw_binding,
            f"manifest evaluation source binding {role}",
        )
        _equal(
            f"manifest evaluation source binding {role} fields",
            set(binding),
            {"path", "sha256"},
        )
        path_text = binding.get("path")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            _fail(
                f"manifest evaluation source binding {role} path "
                "must be absolute"
            )
        _sha256(
            binding.get("sha256"),
            f"manifest evaluation source binding {role} SHA-256",
        )

    expected_spec = {
        selection_role: (filename, checkpoint_role)
        for filename, selection_role, checkpoint_role in (
            evaluator.CHECKPOINT_SPECS
        )
    }
    expected_keys = {
        (trajectory_seed, arm, selection_role)
        for trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS
        for arm in replication_core.SUPPORTED_ARMS
        for selection_role in SELECTION_ROLES
    }
    result_fields = {
        "threshold_domain_id",
        "arm",
        "variant",
        "trajectory_seed",
        "selection_role",
        "checkpoint_filename",
        "checkpoint_role",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "result_path",
        "result_sha256",
        "execution_device_assignment",
        "fixed_threshold_0_5",
        "best_points_under_fa_budget",
        "prediction_cache",
    }
    cache_fields = {
        "metadata_path",
        "metadata_sha256",
        "arrays_path",
        "arrays_sha256",
        "identity",
        "engineering_request_identity",
        "prediction_content_sha256",
        "image_count",
        "image_ids_sha256",
        "paired_image_statistics_available",
    }
    results_raw = _sequence(manifest.get("results"), "manifest results")
    if len(results_raw) != evaluator.EXPECTED_SWEEP_COUNT:
        _fail("manifest does not contain exactly eight results")
    results: dict[tuple[int, str, str], dict[str, Any]] = {}
    for index, raw in enumerate(results_raw):
        result = dict(_mapping(raw, f"manifest result[{index}]"))
        _equal(f"manifest result[{index}] fields", set(result), result_fields)
        trajectory_seed = _integer(
            result.get("trajectory_seed"),
            f"manifest result[{index}] trajectory seed",
        )
        arm = result.get("arm")
        selection_role = result.get("selection_role")
        key = (trajectory_seed, arm, selection_role)
        if key not in expected_keys:
            _fail(f"manifest result[{index}] identity is not registered: {key}")
        if key in results:
            _fail(f"duplicate manifest result identity: {key}")
        expected_filename, expected_checkpoint_role = expected_spec[
            selection_role
        ]
        _equal(
            f"manifest result[{index}] variant",
            result.get("variant"),
            replication_core.arm_definition(arm).variant,
        )
        _equal(
            f"manifest result[{index}] checkpoint filename",
            result.get("checkpoint_filename"),
            expected_filename,
        )
        _equal(
            f"manifest result[{index}] checkpoint role",
            result.get("checkpoint_role"),
            expected_checkpoint_role,
        )
        epoch = _integer(
            result.get("checkpoint_epoch"),
            f"manifest result[{index}] checkpoint epoch",
        )
        if not 1 <= epoch <= evaluator.EXPECTED_EPOCHS:
            _fail(f"manifest result[{index}] checkpoint epoch is invalid")
        _sha256(
            result.get("checkpoint_sha256"),
            f"manifest result[{index}] checkpoint SHA-256",
        )
        _sha256(
            result.get("result_sha256"),
            f"manifest result[{index}] result SHA-256",
        )
        device_assignment = dict(
            _mapping(
                result.get("execution_device_assignment"),
                f"manifest result[{index}] execution device",
            )
        )
        _equal(
            f"manifest result[{index}] execution-device fields",
            set(device_assignment),
            {
                "device",
                "physical_gpu_index",
                "physical_gpu_uuid",
                "cuda_visible_devices",
                "visible_cuda_device_count",
                "device_name",
            },
        )
        expected_gpu = expected_gpu_policy["arm_assignments"][arm]
        _integer(
            device_assignment.get("physical_gpu_index"),
            f"manifest result[{index}] physical GPU index",
        )
        _integer(
            device_assignment.get("visible_cuda_device_count"),
            f"manifest result[{index}] visible CUDA device count",
        )
        for name, expected in (
            ("device", "cuda:0"),
            (
                "physical_gpu_index",
                expected_gpu["physical_gpu_index"],
            ),
            (
                "physical_gpu_uuid",
                expected_gpu["physical_gpu_uuid"],
            ),
            (
                "cuda_visible_devices",
                expected_gpu["physical_gpu_uuid"],
            ),
            ("visible_cuda_device_count", 1),
        ):
            _equal(
                f"manifest result[{index}] execution device {name}",
                device_assignment.get(name),
                expected,
            )
        if (
            not isinstance(device_assignment.get("device_name"), str)
            or not device_assignment["device_name"]
        ):
            _fail(
                f"manifest result[{index}] execution device name is missing"
            )
        result_path = result.get("result_path")
        if not isinstance(result_path, str) or not Path(result_path).is_absolute():
            _fail(f"manifest result[{index}] result path must be absolute")
        _validate_metric_point(
            result.get("fixed_threshold_0_5"),
            label=f"manifest result[{index}] fixed point",
            expected_threshold=FIXED_THRESHOLD,
        )
        budget_points = _mapping(
            result.get("best_points_under_fa_budget"),
            f"manifest result[{index}] Fa-budget points",
        )
        _equal(
            f"manifest result[{index}] Fa-budget keys",
            set(budget_points),
            set(evaluator.BUDGET_KEYS),
        )
        for budget_key, budget in zip(
            evaluator.BUDGET_KEYS,
            evaluator.FA_BUDGETS,
        ):
            point = _validate_metric_point(
                budget_points[budget_key],
                label=(
                    f"manifest result[{index}] Fa-budget "
                    f"{budget_key}"
                ),
            )
            if float(point["fa"]) > float(budget):
                _fail(
                    f"manifest result[{index}] point exceeds Fa budget "
                    f"{budget_key}"
                )
        cache_binding = dict(
            _mapping(
                result.get("prediction_cache"),
                f"manifest result[{index}] prediction cache",
            )
        )
        _equal(
            f"manifest result[{index}] prediction-cache fields",
            set(cache_binding),
            cache_fields,
        )
        for name in (
            "metadata_sha256",
            "arrays_sha256",
            "prediction_content_sha256",
            "image_ids_sha256",
        ):
            _sha256(
                cache_binding.get(name),
                f"manifest result[{index}] cache {name}",
            )
        for name in ("metadata_path", "arrays_path"):
            path_text = cache_binding.get(name)
            if (
                not isinstance(path_text, str)
                or not Path(path_text).is_absolute()
            ):
                _fail(
                    f"manifest result[{index}] cache {name} "
                    "must be absolute"
                )
        collector_identity = cache_core.validate_cache_identity(
            _mapping(
                cache_binding.get("identity"),
                f"manifest result[{index}] collector cache identity",
            )
        )
        engineering_identity = dict(
            _mapping(
                cache_binding.get("engineering_request_identity"),
                f"manifest result[{index}] engineering cache identity",
            )
        )
        engineering_fields = {
            "schema",
            "arm",
            "variant",
            "trajectory_seed",
            "run_id",
            "run_directory",
            "seed_contract_sha256",
            "child_manifest_sha256",
            "source_lock_sha256",
            "checkpoint_filename",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "threshold_domain_id",
            "adapter_source_sha256",
            "engineering_request_identity_sha256",
            "collector_evaluator_sha256",
            "collector_evaluator_sha256_derivation",
        }
        _equal(
            f"manifest result[{index}] engineering cache fields",
            set(engineering_identity),
            engineering_fields,
        )
        adapter_sha256 = evaluation_source_binding[
            "checkpoint_local_adapter"
        ]["sha256"]
        for name, expected in (
            ("schema", evaluator.CACHE_REQUEST_IDENTITY_SCHEMA),
            ("arm", arm),
            ("variant", result["variant"]),
            ("trajectory_seed", trajectory_seed),
            (
                "seed_contract_sha256",
                manifest["seed_contract_sha256"],
            ),
            ("source_lock_sha256", manifest["source_lock_sha256"]),
            ("checkpoint_filename", result["checkpoint_filename"]),
            ("checkpoint_epoch", result["checkpoint_epoch"]),
            ("checkpoint_sha256", result["checkpoint_sha256"]),
            ("threshold_domain_id", result["threshold_domain_id"]),
            ("adapter_source_sha256", adapter_sha256),
        ):
            _equal(
                f"manifest result[{index}] engineering cache {name}",
                engineering_identity.get(name),
                expected,
            )
        _sha256(
            engineering_identity.get("child_manifest_sha256"),
            f"manifest result[{index}] child-manifest SHA-256",
        )
        if (
            not isinstance(engineering_identity.get("run_id"), str)
            or not engineering_identity["run_id"]
            or not isinstance(
                engineering_identity.get("run_directory"),
                str,
            )
            or not Path(engineering_identity["run_directory"]).is_absolute()
        ):
            _fail(
                f"manifest result[{index}] engineering run identity is invalid"
            )
        engineering_core = {
            name: engineering_identity[name]
            for name in engineering_fields
            if name
            not in {
                "engineering_request_identity_sha256",
                "collector_evaluator_sha256",
                "collector_evaluator_sha256_derivation",
            }
        }
        engineering_sha256 = evaluator._canonical_digest(engineering_core)
        _equal(
            f"manifest result[{index}] engineering request SHA-256",
            engineering_identity["engineering_request_identity_sha256"],
            engineering_sha256,
        )
        derivation = {
            "schema": evaluator.CACHE_EVALUATOR_DERIVATION_SCHEMA,
            "algorithm": "sha256_of_canonical_json",
            "adapter_source_sha256": adapter_sha256,
            "engineering_request_identity_sha256": engineering_sha256,
        }
        _equal(
            f"manifest result[{index}] collector evaluator derivation",
            engineering_identity[
                "collector_evaluator_sha256_derivation"
            ],
            derivation,
        )
        derived_evaluator_sha256 = evaluator._canonical_digest(derivation)
        _equal(
            f"manifest result[{index}] collector evaluator SHA-256",
            engineering_identity["collector_evaluator_sha256"],
            derived_evaluator_sha256,
        )
        _equal(
            f"manifest result[{index}] collector identity evaluator SHA-256",
            collector_identity["evaluator_sha256"],
            derived_evaluator_sha256,
        )
        for name, expected in (
            ("checkpoint_sha256", result["checkpoint_sha256"]),
            ("source_lock_sha256", manifest["source_lock_sha256"]),
            ("validation_ids_sha256", validation_ids_sha256),
            ("validation_count", evaluator.EXPECTED_VALIDATION_COUNT),
        ):
            _equal(
                f"manifest result[{index}] collector identity {name}",
                collector_identity.get(name),
                expected,
            )
        _equal(
            f"manifest result[{index}] cache image count",
            cache_binding.get("image_count"),
            evaluator.EXPECTED_VALIDATION_COUNT,
        )
        _equal(
            f"manifest result[{index}] cache validation IDs",
            cache_binding.get("image_ids_sha256"),
            validation_ids_sha256,
        )
        _equal(
            f"manifest result[{index}] paired statistics flag",
            cache_binding.get("paired_image_statistics_available"),
            True,
        )
        results[key] = result
    _equal("manifest result identity matrix", set(results), expected_keys)
    expected_result_order = tuple(
        (trajectory_seed, arm, selection_role)
        for trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS
        for arm in replication_core.SUPPORTED_ARMS
        for _, selection_role, _ in evaluator.CHECKPOINT_SPECS
    )
    observed_result_order = tuple(
        (
            result["trajectory_seed"],
            result["arm"],
            result["selection_role"],
        )
        for result in results_raw
    )
    _equal(
        "manifest canonical result order",
        observed_result_order,
        expected_result_order,
    )

    group_fields = {
        "trajectory_seed",
        "selection_role",
        "validation_count",
        "validation_ids_sha256",
        "arm_b_cache_metadata_path",
        "arm_b_cache_metadata_sha256",
        "arm_d_cache_metadata_path",
        "arm_d_cache_metadata_sha256",
        "image_level_pairing_ready",
    }
    expected_group_keys = {
        (trajectory_seed, selection_role)
        for trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS
        for selection_role in SELECTION_ROLES
    }
    groups_raw = _sequence(
        manifest.get("paired_checkpoint_groups"),
        "manifest paired checkpoint groups",
    )
    if len(groups_raw) != len(expected_group_keys):
        _fail("manifest does not contain exactly four paired groups")
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for index, raw in enumerate(groups_raw):
        group = dict(_mapping(raw, f"manifest paired group[{index}]"))
        _equal(
            f"manifest paired group[{index}] fields",
            set(group),
            group_fields,
        )
        key = (
            _integer(
                group.get("trajectory_seed"),
                f"manifest paired group[{index}] trajectory seed",
            ),
            group.get("selection_role"),
        )
        if key not in expected_group_keys:
            _fail(f"manifest paired group identity is not registered: {key}")
        if key in groups:
            _fail(f"duplicate manifest paired group identity: {key}")
        _equal(
            f"manifest paired group[{index}] validation count",
            group.get("validation_count"),
            evaluator.EXPECTED_VALIDATION_COUNT,
        )
        _equal(
            f"manifest paired group[{index}] validation IDs",
            group.get("validation_ids_sha256"),
            validation_ids_sha256,
        )
        _equal(
            f"manifest paired group[{index}] readiness",
            group.get("image_level_pairing_ready"),
            True,
        )
        trajectory_seed, selection_role = key
        for arm, prefix in (
            (replication_core.ARM_B, "arm_b"),
            (replication_core.ARM_D, "arm_d"),
        ):
            result_cache = _mapping(
                results[
                    (trajectory_seed, arm, selection_role)
                ]["prediction_cache"],
                "result prediction cache",
            )
            _equal(
                f"manifest paired group[{index}] {arm} metadata path",
                group.get(f"{prefix}_cache_metadata_path"),
                result_cache["metadata_path"],
            )
            _equal(
                f"manifest paired group[{index}] {arm} metadata SHA-256",
                group.get(f"{prefix}_cache_metadata_sha256"),
                result_cache["metadata_sha256"],
            )
        groups[key] = group
    _equal("manifest paired-group matrix", set(groups), expected_group_keys)
    expected_group_order = tuple(
        (trajectory_seed, selection_role)
        for trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS
        for selection_role in SELECTION_ROLES
    )
    observed_group_order = tuple(
        (group["trajectory_seed"], group["selection_role"])
        for group in groups_raw
    )
    _equal(
        "manifest canonical paired-group order",
        observed_group_order,
        expected_group_order,
    )

    for result in results.values():
        cache_binding = result["prediction_cache"]
        _equal(
            "cache validation-ID binding",
            cache_binding["image_ids_sha256"],
            validation_ids_sha256,
        )
    return results, groups


def _evidence_inventory(
    results: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    missing: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, result in results.items():
        trajectory_seed, arm, selection_role = key
        result_path = Path(str(result["result_path"]))
        result_role = (
            f"seed_{trajectory_seed}_{arm}_{selection_role}_"
            "checkpoint_local_result"
        )
        result_record = {
            "role": result_role,
            "path": str(result_path),
        }
        if result_path.is_symlink():
            invalid.append({**result_record, "reason": "symlink"})
        elif not result_path.exists():
            missing.append(result_record)
        elif not result_path.is_file():
            invalid.append({**result_record, "reason": "not_regular_file"})
        cache = _mapping(result["prediction_cache"], "prediction cache")
        for kind in ("metadata", "arrays"):
            path = Path(str(cache[f"{kind}_path"]))
            role = (
                f"seed_{trajectory_seed}_{arm}_{selection_role}_"
                f"prediction_cache_{kind}"
            )
            identity = (role, str(path))
            if identity in seen:
                continue
            seen.add(identity)
            record = {"role": role, "path": str(path)}
            if path.is_symlink():
                invalid.append({**record, "reason": "symlink"})
            elif not path.exists():
                missing.append(record)
            elif not path.is_file():
                invalid.append({**record, "reason": "not_regular_file"})
    return missing, invalid


def _validate_result_files(
    manifest: Mapping[str, Any],
    results: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind all eight manifest records to their immutable result JSON files."""

    verified: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for key in sorted(results):
        trajectory_seed, arm, selection_role = key
        result = results[key]
        result_path = Path(str(result["result_path"])).resolve()
        if result_path in seen_paths:
            _fail(
                "checkpoint-local result path is reused across identities: "
                f"{result_path}"
            )
        seen_paths.add(result_path)
        before = _sha256_file(
            result_path,
            f"checkpoint-local result {key}",
        )
        _equal(
            f"checkpoint-local result SHA-256 {key}",
            before,
            result["result_sha256"],
        )
        payload = evaluator._load_result_object(result_path)
        after = _sha256_file(
            result_path,
            f"checkpoint-local result {key}",
        )
        _equal(
            f"checkpoint-local result stability {key}",
            after,
            before,
        )
        for label, observed, expected in (
            ("schema", payload.get("schema"), evaluator.RESULT_SCHEMA),
            ("execution complete", payload.get("execution_complete"), True),
            (
                "paired image statistics",
                payload.get("paired_image_statistics_available"),
                True,
            ),
            (
                "threshold selection scope",
                payload.get("threshold_selection_scope"),
                "single_checkpoint_only",
            ),
            (
                "cross-checkpoint pooling",
                payload.get("cross_checkpoint_point_pooling"),
                False,
            ),
            (
                "official test accessed",
                payload.get("official_test_accessed"),
                False,
            ),
            (
                "top-level checkpoint epoch",
                payload.get("checkpoint_epoch"),
                result["checkpoint_epoch"],
            ),
            (
                "top-level checkpoint role",
                payload.get("checkpoint_role"),
                result["checkpoint_role"],
            ),
            (
                "top-level checkpoint SHA-256",
                payload.get("checkpoint_sha256"),
                result["checkpoint_sha256"],
            ),
            ("top-level variant", payload.get("variant"), result["variant"]),
            (
                "top-level trajectory seed",
                payload.get("seed"),
                trajectory_seed,
            ),
            (
                "execution device assignment",
                payload.get("execution_device_assignment"),
                result["execution_device_assignment"],
            ),
        ):
            _equal(f"checkpoint-local result {key} {label}", observed, expected)

        checkpoint_identity = _mapping(
            payload.get("source_checkpoint_identity"),
            f"checkpoint-local result {key} source checkpoint identity",
        )
        for name, expected in (
            ("arm", arm),
            ("variant", result["variant"]),
            ("trajectory_seed", trajectory_seed),
            ("selection_role", selection_role),
            ("checkpoint_filename", result["checkpoint_filename"]),
            ("checkpoint_role", result["checkpoint_role"]),
            ("checkpoint_epoch", result["checkpoint_epoch"]),
            ("checkpoint_sha256", result["checkpoint_sha256"]),
            ("threshold_domain_id", result["threshold_domain_id"]),
        ):
            _equal(
                f"checkpoint-local result {key} identity {name}",
                checkpoint_identity.get(name),
                expected,
            )
        run_identity = _mapping(
            payload.get("source_run_identity"),
            f"checkpoint-local result {key} source run identity",
        )
        _equal(
            f"checkpoint-local result {key} run variant",
            run_identity.get("variant"),
            result["variant"],
        )
        _equal(
            f"checkpoint-local result {key} run seed",
            run_identity.get("seed"),
            trajectory_seed,
        )
        replication_binding = _mapping(
            payload.get("replication_input_binding"),
            f"checkpoint-local result {key} replication binding",
        )
        _equal(
            f"checkpoint-local result {key} source-lock binding",
            replication_binding.get("source_lock_sha256"),
            manifest["source_lock_sha256"],
        )
        _equal(
            f"checkpoint-local result {key} seed-contract binding",
            replication_binding.get("seed_contract_sha256"),
            manifest["seed_contract_sha256"],
        )
        _equal(
            f"checkpoint-local result {key} evaluation source binding",
            payload.get("evaluation_source_binding"),
            manifest["evaluation_source_binding"],
        )
        _equal(
            f"checkpoint-local result {key} fixed point",
            payload.get("fixed_threshold_0_5"),
            result["fixed_threshold_0_5"],
        )
        _equal(
            f"checkpoint-local result {key} Fa-budget points",
            payload.get("best_points_under_fa_budget"),
            result["best_points_under_fa_budget"],
        )
        payload_cache = _mapping(
            payload.get("prediction_cache"),
            f"checkpoint-local result {key} prediction cache",
        )
        for name, expected in result["prediction_cache"].items():
            _equal(
                f"checkpoint-local result {key} prediction cache {name}",
                payload_cache.get(name),
                expected,
            )
        verified.append(
            {
                "trajectory_seed": trajectory_seed,
                "arm": arm,
                "selection_role": selection_role,
                "path": str(result_path),
                "sha256": before,
            }
        )
    if len(verified) != evaluator.EXPECTED_SWEEP_COUNT:
        _fail("verified checkpoint-local result count differs from eight")
    return verified


def _load_caches(
    manifest: Mapping[str, Any],
    results: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> tuple[
    dict[tuple[int, str, str], cache_core.PredictionCache],
    dict[str, Any],
]:
    loaded: dict[
        tuple[int, str, str],
        cache_core.PredictionCache,
    ] = {}
    compatibility: dict[str, set[Any]] = {
        "dataset_sha256": set(),
        "adapter_source_sha256": set(),
        "normalization_sha256": set(),
        "source_lock_sha256": set(),
        "validation_ids_sha256": set(),
        "validation_count": set(),
        "match_radius": set(),
        "tiny_area": set(),
    }
    for key in sorted(results):
        result = results[key]
        binding = _mapping(
            result["prediction_cache"],
            f"prediction cache binding {key}",
        )
        metadata_path = Path(str(binding["metadata_path"]))
        arrays_path = Path(str(binding["arrays_path"]))
        _equal(
            f"prediction cache metadata SHA-256 {key}",
            _sha256_file(metadata_path, "prediction cache metadata"),
            binding["metadata_sha256"],
        )
        _equal(
            f"prediction cache arrays SHA-256 {key}",
            _sha256_file(arrays_path, "prediction cache arrays"),
            binding["arrays_sha256"],
        )
        cache = cache_core.load_prediction_cache(metadata_path)
        _equal(
            f"prediction cache content SHA-256 {key}",
            cache.content_sha256,
            binding["prediction_content_sha256"],
        )
        _equal(
            f"prediction cache image count {key}",
            len(cache.records),
            evaluator.EXPECTED_VALIDATION_COUNT,
        )
        _equal(
            f"prediction cache validation IDs {key}",
            cache_core.validation_identifier_sha256(
                tuple(record.image_id for record in cache.records)
            ),
            manifest["validation_ids_sha256"],
        )
        _equal(
            f"prediction cache checkpoint SHA-256 {key}",
            cache.identity["checkpoint_sha256"],
            result["checkpoint_sha256"],
        )
        _equal(
            f"prediction cache mode {key}",
            cache.identity["mode"]["name"],
            "full",
        )
        _equal(
            f"prediction cache source-lock SHA-256 {key}",
            cache.identity["source_lock_sha256"],
            manifest["source_lock_sha256"],
        )
        recorded_identity = cache_core.validate_cache_identity(
            _mapping(
                binding.get("identity"),
                f"prediction cache recorded collector identity {key}",
            )
        )
        _equal(
            f"prediction cache recorded/live collector identity {key}",
            recorded_identity,
            cache.identity,
        )
        engineering_identity = _mapping(
            binding.get("engineering_request_identity"),
            f"prediction cache engineering request identity {key}",
        )
        adapter_binding = _mapping(
            _mapping(
                manifest["evaluation_source_binding"],
                "manifest evaluation source binding",
            ).get("checkpoint_local_adapter"),
            "manifest checkpoint-local adapter binding",
        )
        _equal(
            f"prediction cache request-bound evaluator SHA-256 {key}",
            cache.identity["evaluator_sha256"],
            engineering_identity.get("collector_evaluator_sha256"),
        )
        _equal(
            f"prediction cache adapter-source SHA-256 {key}",
            engineering_identity.get("adapter_source_sha256"),
            adapter_binding.get("sha256"),
        )
        _equal(
            f"prediction cache match radius {key}",
            cache.match_radius,
            evaluator.FORMAL_MATCH_RADIUS,
        )
        _equal(
            f"prediction cache tiny area {key}",
            cache.tiny_area,
            evaluator.FORMAL_TINY_AREA,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded_arrays = metadata_path.parent / metadata["arrays"]["filename"]
        _equal(
            f"prediction cache arrays path {key}",
            recorded_arrays.resolve(),
            arrays_path.resolve(),
        )
        for name in (
            "dataset_sha256",
            "normalization_sha256",
            "source_lock_sha256",
            "validation_ids_sha256",
            "validation_count",
        ):
            compatibility[name].add(cache.identity[name])
        compatibility["adapter_source_sha256"].add(
            engineering_identity["adapter_source_sha256"]
        )
        compatibility["match_radius"].add(cache.match_radius)
        compatibility["tiny_area"].add(cache.tiny_area)
        loaded[key] = cache
    for name, values in compatibility.items():
        if len(values) != 1:
            _fail(f"prediction caches disagree on {name}")

    reference_targets: dict[str, np.ndarray] | None = None
    reference_shapes: dict[str, tuple[int, int]] | None = None
    reference_ordered_ids: tuple[str, ...] | None = None
    for key in sorted(loaded):
        cache = loaded[key]
        ordered_ids = tuple(record.image_id for record in cache.records)
        records = {record.image_id: record for record in cache.records}
        if len(records) != len(cache.records):
            _fail(f"prediction cache contains duplicate image IDs: {key}")
        targets = {
            image_id: record.target for image_id, record in records.items()
        }
        shapes = {
            image_id: tuple(record.probability.shape)
            for image_id, record in records.items()
        }
        if reference_targets is None:
            reference_targets = targets
            reference_shapes = shapes
            reference_ordered_ids = ordered_ids
            continue
        _equal(
            f"prediction cache ordered image IDs {key}",
            ordered_ids,
            reference_ordered_ids,
        )
        _equal(
            f"prediction cache image-ID set {key}",
            set(targets),
            set(reference_targets),
        )
        _equal(
            f"prediction cache image shapes {key}",
            shapes,
            reference_shapes,
        )
        for image_id in sorted(reference_targets):
            if not np.array_equal(
                targets[image_id],
                reference_targets[image_id],
            ):
                _fail(
                    "paired prediction-cache targets differ for "
                    f"{key}, image {image_id}"
                )
    return loaded, {
        name: next(iter(values))
        for name, values in compatibility.items()
    }


def _ordered_rows(
    cache: cache_core.PredictionCache,
) -> list[dict[str, Any]]:
    rows = cache_core.image_sufficient_statistics(
        cache,
        threshold=FIXED_THRESHOLD,
    )
    if len(rows) != evaluator.EXPECTED_VALIDATION_COUNT:
        _fail("image sufficient statistics do not contain 133 rows")
    return sorted(rows, key=lambda row: str(row["image_id"]))


def _point_projection(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: copy.deepcopy(point[name])
        for name in _POINT_AUDIT_FIELDS
        if name in point
    }


def _metric_delta(
    d_point: Mapping[str, Any],
    b_point: Mapping[str, Any],
    *,
    include_count: bool = False,
) -> dict[str, float | int]:
    names: tuple[str, ...] = METRIC_KEYS
    if include_count:
        names = (
            *names,
            "unmatched_predicted_object_count",
        )
    return {
        name: (
            int(d_point[name]) - int(b_point[name])
            if name == "unmatched_predicted_object_count"
            else float(d_point[name]) - float(b_point[name])
        )
        for name in names
    }


def _assert_manifest_point_matches_cache(
    manifest_point: Mapping[str, Any],
    cache_point: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for name in METRIC_KEYS:
        observed = _finite(manifest_point.get(name), f"{label}.{name}")
        expected = _finite(cache_point.get(name), f"{label}.cache.{name}")
        if not math.isclose(
            observed,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            _fail(
                f"{label}.{name} differs from the lossless cache: "
                f"{observed} != {expected}"
            )
    for name in (
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    ):
        _equal(
            f"{label}.{name}",
            manifest_point.get(name),
            cache_point.get(name),
        )


def _assert_manifest_budget_matches_cache(
    manifest_point: Mapping[str, Any],
    cache_point: Mapping[str, Any],
    *,
    label: str,
) -> None:
    observed_threshold = _finite(
        manifest_point.get("threshold"),
        f"{label}.threshold",
    )
    expected_threshold = _finite(
        cache_point.get("threshold"),
        f"{label}.cache.threshold",
    )
    if not math.isclose(
        observed_threshold,
        expected_threshold,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        _fail(
            f"{label}.threshold differs from the lossless-cache scan: "
            f"{observed_threshold} != {expected_threshold}"
        )
    _assert_manifest_point_matches_cache(
        manifest_point,
        cache_point,
        label=label,
    )


def _bootstrap_metric_arrays(
    rows: Sequence[Mapping[str, Any]],
    sample_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    if (
        sample_indices.ndim != 2
        or sample_indices.shape[1] != len(rows)
        or sample_indices.dtype.kind not in "iu"
    ):
        _fail("bootstrap sample-index matrix has an invalid shape or dtype")
    if (
        int(sample_indices.min(initial=0)) < 0
        or int(sample_indices.max(initial=0)) >= len(rows)
    ):
        _fail("bootstrap sample-index matrix is out of bounds")

    def totals(name: str) -> np.ndarray:
        values = np.asarray(
            [int(row[name]) for row in rows],
            dtype=np.float64,
        )
        return values[sample_indices].sum(axis=1, dtype=np.float64)

    matched = totals("matched_target_count")
    targets = totals("target_count")
    matched_tiny = totals("matched_tiny_target_count")
    tiny = totals("tiny_target_count")
    unmatched_pixels = totals("unmatched_predicted_pixels")
    valid_pixels = totals("valid_pixel_count")
    intersection = totals("intersection")
    union = totals("union")
    false_objects = totals("unmatched_predicted_object_count")
    return {
        "pd": matched / np.maximum(1.0, targets),
        "fa": unmatched_pixels / np.maximum(1.0, valid_pixels),
        "miou": intersection / np.maximum(1.0, union),
        "tiny_pd": np.divide(
            matched_tiny,
            tiny,
            out=np.full(tiny.shape, np.nan, dtype=np.float64),
            where=tiny != 0,
        ),
        "false_objects_per_image": (
            false_objects / float(sample_indices.shape[1])
        ),
    }


def _index_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        ready = np.ascontiguousarray(array, dtype="<i8")
        digest.update(np.asarray(ready.shape, dtype="<i8").tobytes())
        digest.update(ready.tobytes())
    return digest.hexdigest()


def _intervals(
    point_delta: Mapping[str, float],
    samples: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    intervals: dict[str, dict[str, Any]] = {}
    for name in METRIC_KEYS:
        values = np.asarray(samples[name], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size != BOOTSTRAP_REPLICATES:
            _fail(
                f"bootstrap metric {name} has {finite.size} finite "
                f"replicates, expected {BOOTSTRAP_REPLICATES}"
            )
        lower, upper = np.quantile(
            finite,
            (0.005, 0.995),
            method="linear",
        )
        intervals[name] = {
            "delta_orientation": DELTA_ORIENTATION,
            "point_delta": float(point_delta[name]),
            "lower": float(lower),
            "upper": float(upper),
            "finite_replicates": int(finite.size),
        }
    return intervals


def _miou_route(
    intervals: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    criteria = {
        "pd_noninferior_delta_0": (
            float(intervals["pd"]["lower"]) >= 0.0
        ),
        "tiny_pd_noninferior_delta_0": (
            float(intervals["tiny_pd"]["lower"]) >= 0.0
        ),
        "false_objects_noninferior_delta_0": (
            float(intervals["false_objects_per_image"]["upper"]) <= 0.0
        ),
        "miou_superior_delta_0": (
            float(intervals["miou"]["lower"]) > 0.0
        ),
        "fa_noninferior_delta_0": (
            float(intervals["fa"]["upper"]) <= 0.0
        ),
    }
    return {
        "name": "MIOU_ROUTE",
        "noninferiority_margins": {
            "pd": 0.0,
            "tiny_pd": 0.0,
            "miou": 0.0,
            "fa": 0.0,
            "false_objects_per_image": 0.0,
        },
        "criteria": criteria,
        "met": all(criteria.values()),
    }


def _paired_image_bootstrap(
    *,
    b_rows: Sequence[Mapping[str, Any]],
    d_rows: Sequence[Mapping[str, Any]],
    b_point: Mapping[str, Any],
    d_point: Mapping[str, Any],
    indices: np.ndarray,
    index_sha256: str,
) -> dict[str, Any]:
    b_arrays = _bootstrap_metric_arrays(b_rows, indices)
    d_arrays = _bootstrap_metric_arrays(d_rows, indices)
    point_delta = {
        name: float(d_point[name]) - float(b_point[name])
        for name in METRIC_KEYS
    }
    samples = {
        name: d_arrays[name] - b_arrays[name]
        for name in METRIC_KEYS
    }
    intervals = _intervals(point_delta, samples)
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "status": "complete",
        "unit": "paired_image",
        "threshold": FIXED_THRESHOLD,
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "rng_algorithm": "numpy.random.PCG64",
        "shared_b_d_image_indices": True,
        "shared_indices_across_seed_and_checkpoint_policy": True,
        "resample_index_sha256": index_sha256,
        "metric_family": list(METRIC_KEYS),
        "simultaneous_family_confidence": SIMULTANEOUS_FAMILY_CI,
        "per_metric_two_sided_confidence": PER_METRIC_TWO_SIDED_CI,
        "method": "Bonferroni percentile intervals",
        "quantile_method": "linear",
        "intervals": intervals,
        "miou_route_delta_0": _miou_route(intervals),
    }


def _budget_comparison(
    b_scan: Mapping[str, Any],
    d_scan: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for budget_key, budget in zip(
        evaluator.BUDGET_KEYS,
        evaluator.FA_BUDGETS,
    ):
        b_point = _mapping(
            b_scan["budget_points"][budget_key],
            f"B Fa-budget point {budget_key}",
        )
        d_point = _mapping(
            d_scan["budget_points"][budget_key],
            f"D Fa-budget point {budget_key}",
        )
        comparisons[budget_key] = {
            "fa_budget": budget,
            "b": {
                name: copy.deepcopy(b_point[name])
                for name in (
                    "threshold",
                    *METRIC_KEYS,
                    "unmatched_predicted_object_count",
                )
            },
            "d": {
                name: copy.deepcopy(d_point[name])
                for name in (
                    "threshold",
                    *METRIC_KEYS,
                    "unmatched_predicted_object_count",
                )
            },
            "d_minus_b": _metric_delta(
                d_point,
                b_point,
                include_count=True,
            ),
        }
    return {
        "participates_in_engineering_paired_route": False,
        "reason": (
            "pre-registered Fa-budget envelopes are reported in full but "
            "do not replace the fixed-threshold MIOU_ROUTE"
        ),
        "recomputed_from_lossless_prediction_cache": True,
        "b_threshold_count": b_scan["threshold_count"],
        "d_threshold_count": d_scan["threshold_count"],
        "formal_closed_interval_grid": True,
        "points": comparisons,
    }


def _hierarchical_policy_bootstrap(
    *,
    rows: Mapping[
        tuple[int, str, str],
        Sequence[Mapping[str, Any]],
    ],
    points: Mapping[tuple[int, str, str], Mapping[str, Any]],
    selection_role: str,
    seed_draws: np.ndarray,
    image_draws: np.ndarray,
    draw_sha256: str,
) -> dict[str, Any]:
    registered_seeds = tuple(seed_contract.ENGINEERING_TRAJECTORY_SEEDS)
    replicate_count, seed_draw_count = seed_draws.shape
    if (
        replicate_count != BOOTSTRAP_REPLICATES
        or seed_draw_count != len(registered_seeds)
        or image_draws.shape
        != (
            BOOTSTRAP_REPLICATES,
            len(registered_seeds),
            evaluator.EXPECTED_VALIDATION_COUNT,
        )
    ):
        _fail("hierarchical bootstrap draw shape differs")
    samples = {
        name: np.zeros(
            (BOOTSTRAP_REPLICATES, len(registered_seeds)),
            dtype=np.float64,
        )
        for name in METRIC_KEYS
    }
    for draw_slot in range(len(registered_seeds)):
        slot_indices = image_draws[:, draw_slot, :]
        for seed_index, trajectory_seed in enumerate(registered_seeds):
            mask = seed_draws[:, draw_slot] == seed_index
            if not bool(mask.any()):
                continue
            b_arrays = _bootstrap_metric_arrays(
                rows[
                    (
                        trajectory_seed,
                        replication_core.ARM_B,
                        selection_role,
                    )
                ],
                slot_indices[mask],
            )
            d_arrays = _bootstrap_metric_arrays(
                rows[
                    (
                        trajectory_seed,
                        replication_core.ARM_D,
                        selection_role,
                    )
                ],
                slot_indices[mask],
            )
            for name in METRIC_KEYS:
                samples[name][mask, draw_slot] = (
                    d_arrays[name] - b_arrays[name]
                )
    averaged_samples = {
        name: values.mean(axis=1)
        for name, values in samples.items()
    }
    seed_point_deltas = []
    for trajectory_seed in registered_seeds:
        b_point = points[
            (
                trajectory_seed,
                replication_core.ARM_B,
                selection_role,
            )
        ]
        d_point = points[
            (
                trajectory_seed,
                replication_core.ARM_D,
                selection_role,
            )
        ]
        seed_point_deltas.append(
            {
                name: float(d_point[name]) - float(b_point[name])
                for name in METRIC_KEYS
            }
        )
    point_delta = {
        name: float(
            np.mean(
                [delta[name] for delta in seed_point_deltas],
                dtype=np.float64,
            )
        )
        for name in METRIC_KEYS
    }
    intervals = _intervals(point_delta, averaged_samples)
    route = _miou_route(intervals)
    return {
        "schema": HIERARCHICAL_SCHEMA,
        "status": "complete",
        "unit": "seed_then_paired_image",
        "descriptive_only": True,
        "selection_role": selection_role,
        "checkpoint_policy": POLICY_LABELS[selection_role],
        "trajectory_seeds": list(registered_seeds),
        "seed_count": len(registered_seeds),
        "seed_draw_count_per_replicate": len(registered_seeds),
        "image_draw_count_per_sampled_seed": (
            evaluator.EXPECTED_VALIDATION_COUNT
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "rng_algorithm": "numpy.random.PCG64",
        "seed_sampling": "with_replacement",
        "within_seed_image_sampling": "with_replacement",
        "shared_b_d_seed_indices": True,
        "shared_b_d_image_indices": True,
        "shared_draws_across_checkpoint_policies": True,
        "draw_sha256": draw_sha256,
        "seed_effect_aggregation": "equal_weight_mean_of_seed_deltas",
        "metric_family": list(METRIC_KEYS),
        "simultaneous_family_confidence": SIMULTANEOUS_FAMILY_CI,
        "per_metric_two_sided_confidence": PER_METRIC_TWO_SIDED_CI,
        "method": "hierarchical Bonferroni percentile intervals",
        "quantile_method": "linear",
        "point_delta": point_delta,
        "intervals": intervals,
        "miou_route_delta_0": route,
        "engineering_paired_route_met": route["met"],
        "establishes_gate_m_train": False,
    }


def _source_binding() -> dict[str, dict[str, str]]:
    paths = {
        "paired_screen": Path(__file__).resolve(),
        "lossless_prediction_cache_core": Path(cache_core.__file__).resolve(),
        "bootstrap_contract": Path(
            bootstrap_contract.__file__
        ).resolve(),
        "eight_result_manifest_core": Path(evaluator.__file__).resolve(),
        "seed_contract": Path(seed_contract.__file__).resolve(),
    }
    return {
        role: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256_file(path, f"{role} source"),
        }
        for role, path in paths.items()
    }


def analyze(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser()
    if manifest_file.is_symlink():
        return _base_payload(
            status="invalid",
            decision="ENGINEERING_PAIRED_SCREEN_INVALID",
            manifest_path=manifest_file,
            errors=[
                f"manifest is a symlink: {manifest_file.resolve()}"
            ],
        )
    if not manifest_file.exists():
        return _base_payload(
            status="pending",
            decision="ENGINEERING_PAIRED_SCREEN_PENDING",
            manifest_path=manifest_file,
            missing=[
                {
                    "role": "eight_result_manifest",
                    "path": str(manifest_file.resolve()),
                }
            ],
        )
    if not manifest_file.is_file():
        return _base_payload(
            status="invalid",
            decision="ENGINEERING_PAIRED_SCREEN_INVALID",
            manifest_path=manifest_file,
            errors=[
                "manifest must be a regular file: "
                f"{manifest_file.resolve()}"
            ],
        )
    try:
        manifest = evaluator._load_manifest_object(manifest_file)
        results, groups = _validate_manifest(manifest)
        missing, invalid = _evidence_inventory(results)
        if invalid:
            return _base_payload(
                status="invalid",
                decision="ENGINEERING_PAIRED_SCREEN_INVALID",
                manifest_path=manifest_file,
                errors=[
                    f"{record['role']}: {record['reason']}: "
                    f"{record['path']}"
                    for record in invalid
                ],
            )
        if missing:
            return _base_payload(
                status="pending",
                decision="ENGINEERING_PAIRED_SCREEN_PENDING",
                manifest_path=manifest_file,
                missing=missing,
            )
        verified_result_files = _validate_result_files(
            manifest,
            results,
        )
        caches, compatibility = _load_caches(
            manifest,
            results,
        )
        rows: dict[
            tuple[int, str, str],
            list[dict[str, Any]],
        ] = {}
        points: dict[tuple[int, str, str], dict[str, Any]] = {}
        budget_scans: dict[
            tuple[int, str, str],
            dict[str, Any],
        ] = {}
        for key in sorted(caches):
            row_set = _ordered_rows(caches[key])
            rows[key] = row_set
            point = cache_core.aggregate_sufficient_statistics(row_set)
            points[key] = point
            _equal(
                f"cache full-sample target count {key}",
                point.get("target_count"),
                evaluator.EXPECTED_TARGET_COUNT,
            )
            _equal(
                f"cache full-sample tiny-target count {key}",
                point.get("tiny_target_count"),
                evaluator.EXPECTED_TINY_TARGET_COUNT,
            )
            _assert_manifest_point_matches_cache(
                _mapping(
                    results[key]["fixed_threshold_0_5"],
                    f"manifest fixed point {key}",
                ),
                point,
                label=f"manifest fixed point {key}",
            )
            budget_scan = bootstrap_contract.fa_budget_scan(caches[key])
            _equal(
                f"cache Fa-budget scan status {key}",
                budget_scan.get("status"),
                "complete",
            )
            _equal(
                f"cache Fa-budget scan formal grid {key}",
                budget_scan.get("formal_closed_interval_grid"),
                True,
            )
            _equal(
                f"cache Fa-budget scan budgets {key}",
                budget_scan.get("fa_budgets"),
                list(evaluator.FA_BUDGETS),
            )
            scan_points = _mapping(
                budget_scan.get("budget_points"),
                f"cache Fa-budget scan points {key}",
            )
            _equal(
                f"cache Fa-budget scan keys {key}",
                set(scan_points),
                set(evaluator.BUDGET_KEYS),
            )
            manifest_budget_points = _mapping(
                results[key]["best_points_under_fa_budget"],
                f"manifest Fa-budget points {key}",
            )
            for budget_key in evaluator.BUDGET_KEYS:
                _assert_manifest_budget_matches_cache(
                    _mapping(
                        manifest_budget_points[budget_key],
                        f"manifest Fa-budget point {key} {budget_key}",
                    ),
                    _mapping(
                        scan_points[budget_key],
                        f"cache Fa-budget point {key} {budget_key}",
                    ),
                    label=f"Fa-budget point {key} {budget_key}",
                )
            budget_scans[key] = budget_scan

        paired_rng = np.random.default_rng(BOOTSTRAP_SEED)
        paired_indices = paired_rng.integers(
            0,
            evaluator.EXPECTED_VALIDATION_COUNT,
            size=(
                BOOTSTRAP_REPLICATES,
                evaluator.EXPECTED_VALIDATION_COUNT,
            ),
            dtype=np.int64,
        )
        paired_index_sha256 = _index_sha256(paired_indices)
        per_seed_policy: list[dict[str, Any]] = []
        for trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS:
            for selection_role in SELECTION_ROLES:
                b_key = (
                    trajectory_seed,
                    replication_core.ARM_B,
                    selection_role,
                )
                d_key = (
                    trajectory_seed,
                    replication_core.ARM_D,
                    selection_role,
                )
                bootstrap = _paired_image_bootstrap(
                    b_rows=rows[b_key],
                    d_rows=rows[d_key],
                    b_point=points[b_key],
                    d_point=points[d_key],
                    indices=paired_indices,
                    index_sha256=paired_index_sha256,
                )
                per_seed_policy.append(
                    {
                        "trajectory_seed": trajectory_seed,
                        "seed_role": (
                            "known_historical_pressure_seed"
                            if trajectory_seed
                            == seed_contract.HISTORICAL_PRESSURE_SEED
                            else "deployment_artifact_hash_replication_seed"
                        ),
                        "selection_role": selection_role,
                        "checkpoint_policy": POLICY_LABELS[
                            selection_role
                        ],
                        "paired_group": copy.deepcopy(
                            groups[(trajectory_seed, selection_role)]
                        ),
                        "b_checkpoint": {
                            name: copy.deepcopy(results[b_key][name])
                            for name in (
                                "checkpoint_filename",
                                "checkpoint_role",
                                "checkpoint_epoch",
                                "checkpoint_sha256",
                                "result_sha256",
                            )
                        },
                        "d_checkpoint": {
                            name: copy.deepcopy(results[d_key][name])
                            for name in (
                                "checkpoint_filename",
                                "checkpoint_role",
                                "checkpoint_epoch",
                                "checkpoint_sha256",
                                "result_sha256",
                            )
                        },
                        "fixed_threshold_0_5": {
                            "b": _point_projection(points[b_key]),
                            "d": _point_projection(points[d_key]),
                            "d_minus_b": _metric_delta(
                                points[d_key],
                                points[b_key],
                                include_count=True,
                            ),
                        },
                        "paired_image_bootstrap": bootstrap,
                        "fa_budget_point_estimates": _budget_comparison(
                            budget_scans[b_key],
                            budget_scans[d_key],
                        ),
                    }
                )

        hierarchical_rng = np.random.default_rng(BOOTSTRAP_SEED)
        seed_draws = hierarchical_rng.integers(
            0,
            len(seed_contract.ENGINEERING_TRAJECTORY_SEEDS),
            size=(
                BOOTSTRAP_REPLICATES,
                len(seed_contract.ENGINEERING_TRAJECTORY_SEEDS),
            ),
            dtype=np.int64,
        )
        image_draws = hierarchical_rng.integers(
            0,
            evaluator.EXPECTED_VALIDATION_COUNT,
            size=(
                BOOTSTRAP_REPLICATES,
                len(seed_contract.ENGINEERING_TRAJECTORY_SEEDS),
                evaluator.EXPECTED_VALIDATION_COUNT,
            ),
            dtype=np.int64,
        )
        hierarchical_draw_sha256 = _index_sha256(
            seed_draws,
            image_draws,
        )
        hierarchical = {
            selection_role: _hierarchical_policy_bootstrap(
                rows=rows,
                points=points,
                selection_role=selection_role,
                seed_draws=seed_draws,
                image_draws=image_draws,
                draw_sha256=hierarchical_draw_sha256,
            )
            for selection_role in SELECTION_ROLES
        }
        primary_route_met = bool(
            hierarchical[PRIMARY_SELECTION_ROLE][
                "engineering_paired_route_met"
            ]
        )
        payload = _base_payload(
            status="complete",
            decision=(
                "ENGINEERING_PAIRED_SCREEN_ROUTE_MET"
                if primary_route_met
                else "ENGINEERING_PAIRED_SCREEN_ROUTE_NOT_MET"
            ),
            manifest_path=manifest_file,
        )
        payload.update(
            {
                "manifest": {
                    "path": str(manifest_file.resolve()),
                    "sha256": _sha256_file(
                        manifest_file,
                        "eight-result manifest",
                    ),
                    "schema": manifest["schema"],
                    "result_count": manifest["result_count"],
                    "paired_checkpoint_group_count": manifest[
                        "paired_checkpoint_group_count"
                    ],
                    "checkpoint_local_result_count": len(
                        verified_result_files
                    ),
                    "checkpoint_local_results": verified_result_files,
                },
                "source_binding": _source_binding(),
                "cache_compatibility": {
                    **copy.deepcopy(compatibility),
                    "all_eight_cache_targets_identical": True,
                    "all_eight_cache_image_ids_identical": True,
                    "all_eight_cache_shapes_identical": True,
                    "lossless_prediction_cache_count": len(caches),
                    "paired_checkpoint_group_count": len(groups),
                },
                "per_seed_checkpoint_policy_results": per_seed_policy,
                "hierarchical_seed_image_bootstrap": {
                    "descriptive_only": True,
                    "primary_selection_role": PRIMARY_SELECTION_ROLE,
                    "secondary_selection_role": SECONDARY_SELECTION_ROLE,
                    "policies": hierarchical,
                },
                "engineering_paired_route_met": primary_route_met,
                "establishes_gate_m_train": False,
                "interpretation": {
                    "primary_route_source": (
                        "descriptive two-seed hierarchical bootstrap over "
                        "each arm's own best_miou checkpoint"
                    ),
                    "secondary_policy_is_sensitivity_only": True,
                    "fa_budgets_are_reported_not_gate_inputs": True,
                    "seed_3407_is_known_pressure_seed": True,
                    "only_two_fixed_parent_engineering_seeds": True,
                    "engineering_route_does_not_establish_gate_m_train": True,
                },
            }
        )
        return payload
    except Exception as exc:
        return _base_payload(
            status="invalid",
            decision="ENGINEERING_PAIRED_SCREEN_INVALID",
            manifest_path=manifest_file,
            errors=[f"{type(exc).__name__}: {exc}"],
        )


def write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to replace paired-screen output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(dict(payload))
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = analyze(manifest_path=args.manifest)
    output_record: dict[str, str] | None = None
    if args.output is not None and payload["status"] == "complete":
        destination = write_once(args.output, payload)
        output_record = {
            "path": str(destination.resolve()),
            "sha256": _sha256_file(
                destination,
                "paired-screen output",
            ),
        }
    action = {
        "schema": ACTION_SCHEMA,
        "status": payload["status"],
        "decision": payload["decision"],
        "screen": payload,
        "output": output_record,
    }
    print(
        json.dumps(
            action,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    if payload["status"] == "invalid":
        raise SystemExit(1)


__all__ = [
    "ACTION_SCHEMA",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SCHEMA",
    "BOOTSTRAP_SEED",
    "DEFAULT_MANIFEST",
    "DEFAULT_OUTPUT",
    "EngineeringPairedScreenError",
    "HIERARCHICAL_SCHEMA",
    "METRIC_KEYS",
    "PER_METRIC_TWO_SIDED_CI",
    "SCHEMA",
    "SIMULTANEOUS_FAMILY_CI",
    "analyze",
    "canonical_json_bytes",
    "main",
    "write_once",
]


if __name__ == "__main__":
    main()
