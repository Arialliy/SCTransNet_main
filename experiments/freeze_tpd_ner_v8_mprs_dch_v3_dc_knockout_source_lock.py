#!/usr/bin/env python3
"""Freeze or verify the independent V3 DC-knockout diagnostic source lock.

There is deliberately no training lock for this evaluation-only package.
Importing this module is read-only.  ``--mode freeze`` is explicit,
no-overwrite, and remains unavailable until the versioned repaired formal V3
aggregate and both original formal checkpoint sweeps are fully valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v3_source_locks as v3_freeze,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_v3_formal800 as formal_post,
)
from experiments import (  # noqa: E402
    repair_postprocess_tpd_ner_v8_mprs_dch_v3_formal800_selection_contract_v1
    as repaired_formal,
)
from experiments import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)
from experiments.freeze_tpd_clean_v8_mprs_dch_source_locks import (  # noqa: E402
    file_sha256,
    hash_sources,
    load_json_object,
    publish_new_lock,
)


SOURCE_LOCK_SCHEMA = spec.SOURCE_LOCK_SCHEMA
SOURCE_BINDING_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_binding_v2"
)
SOURCE_LOCK_REVISION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_"
    "source_lock_revision_v2"
)
LEGACY_SOURCE_LOCK_V1 = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock.json"
)
LEGACY_SOURCE_LOCK_V1_SHA256 = (
    "89f98ecab9c1cbcd72f40b9ba9c2083076231ad240477d81a69528c0ef9c80f7"
)
LEGACY_OUTPUT_ROOT_V1 = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v8_mprs_dch_v3_dc_knockout_v1"
)
SOURCE_LOCK_REPAIR_REASON = (
    "v1 finalizer omitted required "
    "CUBLAS_WORKSPACE_CONFIG=:4096:8 from evaluator subprocess environment"
)
DEFAULT_SOURCE_LOCK = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock_v2.json"
)
DEFAULT_FORMAL_TRAINING_LOCK = v3_freeze.DEFAULT_TRAINING_LOCK
DEFAULT_FORMAL_ACCEPTANCE_LOCK = v3_freeze.DEFAULT_ACCEPTANCE_LOCK
DEFAULT_FORMAL_MARKER = repaired_formal.REPAIR_COMPLETE_MARKER
DEFAULT_FORMAL_REPORT = repaired_formal.REPAIR_JSON_OUTPUT
DEFAULT_FORMAL_MARKDOWN = repaired_formal.REPAIR_MARKDOWN_OUTPUT
DEFAULT_FORMAL_REPAIR_WRAPPER = Path(repaired_formal.__file__).resolve()
DEFAULT_FORMAL_REPAIR_PROTOCOL = repaired_formal.PROTOCOL
DEFAULT_FORMAL_REPAIR_ATTESTATION = repaired_formal.ATTESTATION
FORMAL_REPAIR_ID = repaired_formal.REPAIR_ID
EXPECTED_FORMAL_DECISION = "RETURN_TO_MODEL_OPTIMIZATION"
DIAGNOSTIC_SOURCE_RELATIVES = (
    "experiments/TPD_NER_V8_MPRS_DCH_V3_DC_KNOCKOUT_PROTOCOL.md",
    "experiments/tpd_ner_v8_mprs_dch_v3_dc_knockout_spec.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout.py",
    "experiments/postprocess_tpd_ner_v8_mprs_dch_v3_dc_knockout.py",
    (
        "experiments/"
        "finalize_tpd_ner_v8_mprs_dch_v3_dc_knockout_gpu23.py"
    ),
    (
        "experiments/"
        "freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock.py"
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    _require(
        value.is_file() and not value.is_symlink(),
        f"{label} must be a regular non-symlink file: {value}",
    )
    return value


def _load_regular_json(path: Path, label: str) -> dict[str, Any]:
    value = load_json_object(_regular_file(path, label))
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _sha256(path: Path, label: str) -> str:
    return file_sha256(_regular_file(path, label))


def source_lock_revision_contract() -> dict[str, Any]:
    """Bind v2 to the immutable failed-before-inference v1 audit parent."""

    parent = _regular_file(
        LEGACY_SOURCE_LOCK_V1,
        "legacy V1 DC-knockout source lock",
    )
    observed_parent_sha256 = _sha256(
        parent,
        "legacy V1 DC-knockout source lock",
    )
    _require(
        observed_parent_sha256 == LEGACY_SOURCE_LOCK_V1_SHA256,
        "legacy V1 DC-knockout source-lock SHA differs",
    )
    _require(
        not LEGACY_OUTPUT_ROOT_V1.exists()
        and not LEGACY_OUTPUT_ROOT_V1.is_symlink(),
        "legacy V1 diagnostic output root must remain absent because the "
        "failed launch published no sweep or aggregate",
    )
    return {
        "schema": SOURCE_LOCK_REVISION_SCHEMA,
        "revision": 2,
        "parent_source_lock": {
            "path": str(parent.resolve()),
            "sha256": observed_parent_sha256,
            "schema": (
                "sctransnet_tpd_ner_v8_mprs_dch_v3_"
                "dc_knockout_source_lock_v1"
            ),
            "retained_read_only_for_audit": True,
            "verified_against_current_v2_sources": False,
        },
        "repair_reason": SOURCE_LOCK_REPAIR_REASON,
        "failed_v1_attempt": {
            "failure_stage": (
                "evaluator_startup_before_model_or_data_inference"
            ),
            "inference_started": False,
            "published_sweep_count": 0,
            "published_aggregate_count": 0,
            "published_complete_marker_count": 0,
            "legacy_output_root": str(LEGACY_OUTPUT_ROOT_V1.resolve()),
            "legacy_output_root_observed_absent": True,
            "v1_artifacts_reused_by_v2": False,
        },
        "replacement": {
            "source_lock": str(DEFAULT_SOURCE_LOCK.resolve()),
            "diagnostic_output_root": str(
                spec.DEFAULT_OUTPUT_ROOT.resolve()
            ),
            "required_evaluator_environment": {
                spec.CUBLAS_WORKSPACE_CONFIG_ENV: (
                    spec.CUBLAS_WORKSPACE_CONFIG
                ),
                spec.PYTHONHASHSEED_ENV: spec.PYTHONHASHSEED,
            },
            "no_overwrite": True,
        },
    }


def policy_contract() -> dict[str, Any]:
    return {
        "artifact_kind": spec.ARTIFACT_KIND,
        "scope": "evaluation_only_same_checkpoint_counterfactual",
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_components": [],
        "training_performed": False,
        "derived_checkpoint_written": False,
        "formal_artifacts_read_only": True,
        "formal_artifacts_unchanged_required": True,
        "formal_aggregate_authority": (
            "versioned_selection_contract_repair_v1_only"
        ),
        "frozen_original_aggregate_accepted": False,
        "each_variant_uses_own_selected_checkpoints": True,
        "official_test_accessed": False,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "multi_seed_scheduled": False,
        "checkpoint_count": len(spec.CHECKPOINTS),
        "knockout_mode_count": len(spec.KNOCKOUT_MODES),
        "aggregate_row_count": spec.EXPECTED_ROW_COUNT,
        "existing_manifest_overwrite_forbidden": True,
        "source_symlinks_forbidden": True,
        "source_lock_revision": 2,
        "parent_source_lock_sha256": LEGACY_SOURCE_LOCK_V1_SHA256,
        "required_evaluator_environment": {
            spec.CUBLAS_WORKSPACE_CONFIG_ENV: (
                spec.CUBLAS_WORKSPACE_CONFIG
            ),
            spec.PYTHONHASHSEED_ENV: spec.PYTHONHASHSEED,
        },
        "diagnostic_output_root": str(spec.DEFAULT_OUTPUT_ROOT.resolve()),
        "formal_output_root": str(spec.FORMAL_RESULT_ROOT.resolve()),
    }


def validate_repaired_formal_aggregate(
    *,
    marker_path: Path,
    report_path: Path,
    markdown_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        Path(marker_path).resolve() == DEFAULT_FORMAL_MARKER.resolve()
        and Path(report_path).resolve() == DEFAULT_FORMAL_REPORT.resolve()
        and Path(markdown_path).resolve()
        == DEFAULT_FORMAL_MARKDOWN.resolve(),
        "only the canonical repaired formal aggregate is accepted",
    )
    marker = _load_regular_json(
        marker_path,
        "repaired formal completion marker",
    )
    report = _load_regular_json(
        report_path,
        "repaired formal aggregate report",
    )
    expected_marker_keys = {
        "schema",
        "status",
        "decision",
        "aggregate_full_model_gate_passed",
        "aggregate_tiny_pd_regressed",
        "tiny_pd_regression_affects_decision",
        "outputs",
    }
    _require(
        set(marker) == expected_marker_keys,
        "formal completion marker field set differs",
    )
    _require(
        marker.get("schema") == formal_post.COMPLETE_MARKER_SCHEMA
        and marker.get("status") == "complete",
        "repaired formal V3 postprocess marker is not complete",
    )
    _require(
        report.get("schema") == formal_post.SCHEMA
        and report.get("status") == "complete",
        "repaired formal V3 aggregate report is not complete",
    )
    _require(
        report.get("row_count") == 8
        and report.get("dataset") == spec.DATASET
        and report.get("training_seed") == spec.TRAINING_SEED
        and report.get("split_seed") == spec.SPLIT_SEED
        and report.get("multi_seed_scheduled") is False
        and report.get("official_test_accessed") is False,
        "repaired formal V3 aggregate identity differs",
    )
    expected_decision = {
        "decision": EXPECTED_FORMAL_DECISION,
        "aggregate_full_model_gate_passed": False,
        "aggregate_tiny_pd_regressed": False,
        "tiny_pd_regression_affects_decision": False,
    }
    for field, expected in expected_decision.items():
        _require(
            report.get(field) == expected,
            f"repaired formal decision field differs: {field}",
        )
    for field in (
        "decision",
        "aggregate_full_model_gate_passed",
        "aggregate_tiny_pd_regressed",
        "tiny_pd_regression_affects_decision",
    ):
        _require(
            marker.get(field) == report.get(field),
            f"repaired formal marker/report differs: {field}",
        )
    outputs = marker.get("outputs")
    expected_outputs = {
        Path(report_path).name: _sha256(
            report_path,
            "repaired formal report",
        ),
        Path(markdown_path).name: _sha256(
            markdown_path,
            "repaired formal markdown report",
        ),
    }
    _require(
        isinstance(outputs, Mapping) and dict(outputs) == expected_outputs,
        "repaired formal completion marker output hashes differ",
    )
    comparison_contract = report.get("comparison_contract")
    _require(
        isinstance(comparison_contract, Mapping),
        "repaired formal comparison contract is missing",
    )
    repair_contract = comparison_contract.get("selection_contract_repair")
    _require(
        isinstance(repair_contract, Mapping),
        "selection-contract repair evidence is missing",
    )
    for field, expected in {
        "schema": repaired_formal.CONTRACT_SCHEMA,
        "repair_id": repaired_formal.REPAIR_ID,
        "logic_override": "same_split_and_training_contract",
        "aggregate_implementation": "frozen_v3_aggregate_and_write",
        "modern_exact_equality_verified": True,
        "baseline_policy_text_equality_to_modern_required": False,
        "baseline_top_level_selection_source_required": False,
        "each_variant_uses_own_selected_checkpoints": True,
    }.items():
        _require(
            repair_contract.get(field) == expected,
            f"selection-contract repair field differs: {field}",
        )
    verified_attestation = repaired_formal.verify_repair_attestation(
        DEFAULT_FORMAL_REPAIR_ATTESTATION
    )
    _require(
        repair_contract.get("attestation") == verified_attestation,
        "repaired formal report attestation differs",
    )
    per_variant = repair_contract.get("per_variant_checkpoint_selection")
    expected_run_dirs = {
        formal_post.VARIANT_V3_ON: formal_post.V3_RUN_DIR,
        formal_post.VARIANT_V2_ON: formal_post.V2_RUN_DIR,
        formal_post.VARIANT_V1_OFF: formal_post.V1_OFF_RUN_DIR,
        formal_post.BASELINE_VARIANT: formal_post.BASELINE_RUN_DIR,
    }
    _require(
        isinstance(per_variant, Mapping)
        and set(per_variant) == set(expected_run_dirs),
        "per-variant selected-checkpoint matrix differs",
    )
    for variant, run_dir in expected_run_dirs.items():
        variant_selection = per_variant.get(variant)
        _require(
            isinstance(variant_selection, Mapping)
            and variant_selection.get("uses_own_checkpoint_directory")
            is True
            and variant_selection.get("checkpoint_count")
            == len(formal_post.CHECKPOINTS),
            f"{variant} own-checkpoint selection evidence differs",
        )
        checkpoints = variant_selection.get("checkpoints")
        _require(
            isinstance(checkpoints, Mapping)
            and set(checkpoints) == set(formal_post.CHECKPOINTS),
            f"{variant} selected-checkpoint set differs",
        )
        for checkpoint, role in formal_post.CHECKPOINT_ROLES.items():
            selection = checkpoints.get(checkpoint)
            expected_checkpoint = (run_dir / checkpoint).resolve()
            expected_sweep = formal_post.sweep_path(
                run_dir,
                checkpoint,
            ).resolve()
            sweep_payload = _load_regular_json(
                expected_sweep,
                f"{variant} {checkpoint} sweep",
            )
            _require(
                isinstance(selection, Mapping)
                and selection.get("selection_owner") == variant
                and selection.get("checkpoint_role") == role
                and selection.get("selection_source")
                == "internal_validation_only"
                and selection.get("official_test_accessed") is False
                and selection.get("checkpoint_path")
                == str(expected_checkpoint)
                and selection.get("sweep_path") == str(expected_sweep)
                and selection.get("checkpoint_sha256")
                == sweep_payload.get("checkpoint_sha256")
                and selection.get("sweep_sha256")
                == _sha256(
                    expected_sweep,
                    f"{variant} {checkpoint} sweep",
                ),
                f"{variant} {checkpoint} own-checkpoint binding differs",
            )
    return marker, report


def current_formal_artifact_binding(
    *,
    training_lock_path: Path = DEFAULT_FORMAL_TRAINING_LOCK,
    acceptance_lock_path: Path = DEFAULT_FORMAL_ACCEPTANCE_LOCK,
    marker_path: Path = DEFAULT_FORMAL_MARKER,
    report_path: Path = DEFAULT_FORMAL_REPORT,
    markdown_path: Path = DEFAULT_FORMAL_MARKDOWN,
    run_dir: Path = spec.FORMAL_RUN_DIR,
) -> dict[str, Any]:
    """Revalidate and hash every immutable formal input to the diagnostic."""

    training_lock = _regular_file(
        training_lock_path,
        "formal V3 training lock",
    )
    acceptance_lock = _regular_file(
        acceptance_lock_path,
        "formal V3 acceptance lock",
    )
    training = v3_freeze.verify_training_lock(training_lock)
    acceptance = v3_freeze.verify_acceptance_lock(
        acceptance_lock,
        training_lock,
    )
    marker, report = validate_repaired_formal_aggregate(
        marker_path=marker_path,
        report_path=report_path,
        markdown_path=markdown_path,
    )
    directory = Path(run_dir)
    _require(
        directory.resolve() == spec.FORMAL_RUN_DIR.resolve(),
        "formal V3 run directory differs from the canonical run",
    )
    _require(
        directory.is_dir() and not directory.is_symlink(),
        "formal V3 run directory is unavailable",
    )
    checkpoints: dict[str, Any] = {}
    sweeps: dict[str, Any] = {}
    for checkpoint in spec.CHECKPOINTS:
        checkpoint_path = directory / checkpoint
        binding = formal_post.current_v3_binding(checkpoint)
        _require(
            Path(binding["run_dir"]).resolve() == directory.resolve(),
            f"formal checkpoint binding run differs: {checkpoint}",
        )
        formal_sweep = formal_post.sweep_path(directory, checkpoint)
        formal_post.validate_v3_sweep(
            formal_sweep,
            checkpoint=checkpoint,
            binding=binding,
        )
        checkpoints[checkpoint] = {
            "path": str(_regular_file(
                checkpoint_path,
                f"formal checkpoint {checkpoint}",
            ).resolve()),
            "role": spec.CHECKPOINT_ROLES[checkpoint],
            "sha256": _sha256(
                checkpoint_path,
                f"formal checkpoint {checkpoint}",
            ),
            "artifact_identity_sha256": spec.canonical_sha256(
                binding["artifact_identity"]
            ),
        }
        sweeps[checkpoint] = {
            "path": str(_regular_file(
                formal_sweep,
                f"formal sweep {checkpoint}",
            ).resolve()),
            "sha256": _sha256(
                formal_sweep,
                f"formal sweep {checkpoint}",
            ),
        }
    binding = {
        "schema": (
            "sctransnet_tpd_ner_v8_mprs_dch_v3_"
            "dc_knockout_formal_input_binding_v2"
        ),
        "formal_training_source_lock": {
            "path": str(training_lock.resolve()),
            "sha256": _sha256(training_lock, "formal training lock"),
            "training_data_sha256": training["training_data_sha256"],
        },
        "formal_acceptance_source_lock": {
            "path": str(acceptance_lock.resolve()),
            "sha256": _sha256(
                acceptance_lock,
                "formal acceptance lock",
            ),
            "training_source_lock_sha256": acceptance[
                "training_source_lock_sha256"
            ],
        },
        "formal_completion_marker": {
            "path": str(Path(marker_path).resolve()),
            "sha256": _sha256(marker_path, "formal completion marker"),
            "schema": marker["schema"],
            "status": marker["status"],
        },
        "formal_aggregate_json": {
            "path": str(Path(report_path).resolve()),
            "sha256": _sha256(report_path, "formal aggregate JSON"),
            "schema": report["schema"],
            "status": report["status"],
        },
        "formal_aggregate_markdown": {
            "path": str(Path(markdown_path).resolve()),
            "sha256": _sha256(
                markdown_path,
                "formal aggregate Markdown",
            ),
        },
        "formal_selection_contract_repair": {
            "repair_id": repaired_formal.REPAIR_ID,
            "authority": (
                "versioned_selection_contract_repair_v1_only"
            ),
            "each_variant_uses_own_selected_checkpoints": True,
            "formal_aggregate_decision": report["decision"],
            "aggregate_full_model_gate_passed": report[
                "aggregate_full_model_gate_passed"
            ],
            "repair_wrapper": {
                "path": str(DEFAULT_FORMAL_REPAIR_WRAPPER),
                "sha256": _sha256(
                    DEFAULT_FORMAL_REPAIR_WRAPPER,
                    "formal repair wrapper",
                ),
            },
            "repair_protocol": {
                "path": str(DEFAULT_FORMAL_REPAIR_PROTOCOL.resolve()),
                "sha256": _sha256(
                    DEFAULT_FORMAL_REPAIR_PROTOCOL,
                    "formal repair protocol",
                ),
            },
            "repair_attestation": {
                "path": str(DEFAULT_FORMAL_REPAIR_ATTESTATION.resolve()),
                "sha256": _sha256(
                    DEFAULT_FORMAL_REPAIR_ATTESTATION,
                    "formal repair attestation",
                ),
                "schema": repaired_formal.ATTESTATION_SCHEMA,
                "status": "frozen",
            },
            "comparison_contract_sha256": spec.canonical_sha256(
                report["comparison_contract"][
                    "selection_contract_repair"
                ]
            ),
        },
        "formal_run_directory": str(directory.resolve()),
        "formal_checkpoints": checkpoints,
        "formal_sweeps": sweeps,
        "formal_outputs_read_only": True,
        "official_test_accessed": False,
    }
    binding["snapshot_sha256"] = spec.canonical_sha256(binding)
    return binding


def _verify_source_mapping(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    sources = payload.get("source_sha256")
    _require(
        isinstance(sources, Mapping) and bool(sources),
        "diagnostic source lock has no source mapping",
    )
    _require(
        payload.get("source_count") == len(sources),
        "diagnostic source_count differs",
    )
    expected = hash_sources(repo_root, tuple(sources))
    if dict(sources) != expected:
        changed = sorted(
            name
            for name in set(sources) | set(expected)
            if sources.get(name) != expected.get(name)
        )
        raise ValueError(f"diagnostic source digests differ: {changed}")


def build_source_lock(
    *,
    repo_root: Path = REPO_ROOT,
    source_relatives: Sequence[str] = DIAGNOSTIC_SOURCE_RELATIVES,
) -> dict[str, Any]:
    frozen_spec = spec.fixed_specification()
    spec.validate_specification(frozen_spec)
    formal_binding = current_formal_artifact_binding()
    sources = hash_sources(repo_root, tuple(source_relatives))
    return {
        "schema": SOURCE_LOCK_SCHEMA,
        "lock_kind": "diagnostic_acceptance",
        "artifact_kind": spec.ARTIFACT_KIND,
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_components": [],
        "source_lock_revision": source_lock_revision_contract(),
        "knockout_spec": frozen_spec,
        "knockout_spec_sha256": spec.specification_sha256(),
        "formal_artifact_binding": formal_binding,
        "source_count": len(sources),
        "source_sha256": sources,
        "policy": policy_contract(),
    }


def verify_source_lock(
    path: Path = DEFAULT_SOURCE_LOCK,
    *,
    repo_root: Path = REPO_ROOT,
    expected_source_relatives: Sequence[str] = DIAGNOSTIC_SOURCE_RELATIVES,
) -> dict[str, Any]:
    payload = _load_regular_json(path, "DC knockout diagnostic source lock")
    _require(
        payload.get("schema") == SOURCE_LOCK_SCHEMA
        and payload.get("lock_kind") == "diagnostic_acceptance"
        and payload.get("artifact_kind") == spec.ARTIFACT_KIND
        and payload.get("dataset") == spec.DATASET
        and payload.get("variant") == spec.VARIANT
        and payload.get("diagnostic_only") is True
        and payload.get("affects_formal_gate") is False
        and payload.get("formal_decision_authority") is False
        and payload.get("formal_gate_components") == [],
        "DC knockout diagnostic source-lock identity differs",
    )
    _require(
        "decision" not in payload
        and "performance_gate_assessment" not in payload,
        "diagnostic source lock may not contain a formal decision",
    )
    _require(
        payload.get("source_lock_revision")
        == source_lock_revision_contract(),
        "diagnostic source-lock revision contract differs",
    )
    frozen_spec = payload.get("knockout_spec")
    _require(
        isinstance(frozen_spec, Mapping),
        "diagnostic source lock has no knockout spec",
    )
    spec.validate_specification(frozen_spec)
    _require(
        payload.get("knockout_spec_sha256")
        == spec.specification_sha256(),
        "diagnostic knockout spec SHA differs",
    )
    _require(
        payload.get("policy") == policy_contract(),
        "diagnostic source-lock policy differs",
    )
    _require(
        set(payload.get("source_sha256", ()))
        == set(expected_source_relatives),
        "diagnostic source set differs",
    )
    _verify_source_mapping(payload, repo_root=repo_root)
    current_formal = current_formal_artifact_binding()
    _require(
        payload.get("formal_artifact_binding") == current_formal,
        "formal V3 inputs changed after diagnostic lock freeze",
    )
    return payload


def current_source_binding(
    path: Path = DEFAULT_SOURCE_LOCK,
) -> dict[str, Any]:
    payload = verify_source_lock(path)
    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "diagnostic_source_lock": {
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(path),
        },
        "source_lock_revision": payload["source_lock_revision"],
        "knockout_spec_sha256": payload["knockout_spec_sha256"],
        "formal_artifact_snapshot_sha256": payload[
            "formal_artifact_binding"
        ]["snapshot_sha256"],
        "formal_training_source_lock_sha256": payload[
            "formal_artifact_binding"
        ]["formal_training_source_lock"]["sha256"],
        "formal_acceptance_source_lock_sha256": payload[
            "formal_artifact_binding"
        ]["formal_acceptance_source_lock"]["sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the V3 DC-knockout diagnostic lock"
    )
    parser.add_argument(
        "--mode",
        choices=("freeze", "verify"),
        required=True,
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_SOURCE_LOCK,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "freeze":
        publish_new_lock(args.source_lock, build_source_lock())
    else:
        verify_source_lock(args.source_lock)
    output = {
        "status": "complete",
        "mode": args.mode,
        "artifact_kind": spec.ARTIFACT_KIND,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "source_lock": str(args.source_lock.resolve()),
        "source_lock_sha256": file_sha256(args.source_lock),
        "knockout_spec_sha256": spec.specification_sha256(),
    }
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_FORMAL_ACCEPTANCE_LOCK",
    "DEFAULT_FORMAL_MARKER",
    "DEFAULT_FORMAL_MARKDOWN",
    "DEFAULT_FORMAL_REPAIR_ATTESTATION",
    "DEFAULT_FORMAL_REPAIR_PROTOCOL",
    "DEFAULT_FORMAL_REPAIR_WRAPPER",
    "DEFAULT_FORMAL_REPORT",
    "DEFAULT_FORMAL_TRAINING_LOCK",
    "DEFAULT_SOURCE_LOCK",
    "DIAGNOSTIC_SOURCE_RELATIVES",
    "LEGACY_OUTPUT_ROOT_V1",
    "LEGACY_SOURCE_LOCK_V1",
    "LEGACY_SOURCE_LOCK_V1_SHA256",
    "SOURCE_LOCK_REPAIR_REASON",
    "SOURCE_LOCK_REVISION_SCHEMA",
    "SOURCE_BINDING_SCHEMA",
    "SOURCE_LOCK_SCHEMA",
    "EXPECTED_FORMAL_DECISION",
    "FORMAL_REPAIR_ID",
    "build_source_lock",
    "current_formal_artifact_binding",
    "current_source_binding",
    "file_sha256",
    "policy_contract",
    "source_lock_revision_contract",
    "validate_repaired_formal_aggregate",
    "verify_source_lock",
]
