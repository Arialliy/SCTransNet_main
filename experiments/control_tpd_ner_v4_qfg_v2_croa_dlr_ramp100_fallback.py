#!/usr/bin/env python3
"""Strict post-formal800 controller for the paired DLR+ramp100 fallback.

The controller never invents a metric threshold.  Its decision authority is
the frozen C/D final-selection policy and the deployment closure that was
published from that selection:

* a selected C or D full-QFG recipe with the policy's non-isolated,
  five-objective contribution is retained;
* every policy-selected non-QFG fallback launches the already validated
  paired E/F DLR+ramp100 formal800 launcher.

``--dry-run`` and ``--status`` are read-only.  ``--worker`` is the only mode
that may publish the controller receipt or invoke the paired launcher.
Neither this module nor its receipt is part of the frozen 48/51-source
training locks or the frozen 15-source current post-training closure.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CURRENT_RESULT_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized"
)
CURRENT_DATASET_ROOT = CURRENT_RESULT_ROOT / "NUDT-SIRST"
SELECTION_BASENAME = (
    "tpd_ner_v4_qfg_v2_croa_formal800_final_selection"
)
DEPLOYMENT_BASENAME = (
    "tpd_ner_v4_qfg_v2_croa_formal800"
)
DEFAULT_SELECTION = (
    CURRENT_DATASET_ROOT
    / "final_selection"
    / f"{SELECTION_BASENAME}.json"
)
DEFAULT_SELECTION_MARKDOWN = (
    CURRENT_DATASET_ROOT
    / "final_selection"
    / f"{SELECTION_BASENAME}.md"
)
DEFAULT_DEPLOYMENT_ARTIFACT = (
    CURRENT_DATASET_ROOT
    / "deployment"
    / f"{DEPLOYMENT_BASENAME}_inference.pth.tar"
)
DEFAULT_DEPLOYMENT_MANIFEST = (
    CURRENT_DATASET_ROOT
    / "deployment"
    / f"{DEPLOYMENT_BASENAME}_deployment_manifest.json"
)
DEFAULT_CLOSURE_LOCK = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json"
)
DEFAULT_LAUNCHER = (
    REPO_ROOT
    / "experiments/"
    "launch_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_2x5090.sh"
)
DEFAULT_DLR_SOURCE_LOCK = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json"
)
DEFAULT_DLR_RESULT_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_v1"
)
DEFAULT_RECEIPT = (
    CURRENT_DATASET_ROOT
    / "fallback_control/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback_receipt.json"
)
DEFAULT_PAIRED_LOCK = (
    CURRENT_DATASET_ROOT
    / "fallback_control/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback_paired.lock"
)

RECEIPT_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "fallback_receipt_v1"
)
STATUS_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "fallback_controller_status_v1"
)
CURRENT_SELECTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_final_selection_v2"
)
CURRENT_DEPLOYMENT_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_manifest_v1"
)
CURRENT_CLOSURE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "posttraining_closure_source_lock_v1"
)
DLR_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_source_lock_v1"
)

EXIT_OK = 0
EXIT_PERMANENT = 64
EXIT_RETRY = 75
TRAINING_SEED = 42
SPLIT_SEED = 20260722
FORMAL_EPOCHS = 800
EXPECTED_CURRENT_CLOSURE_SOURCE_COUNT = 15
EXPECTED_DLR_SOURCE_COUNT = 51
QFG_METHOD_CONTRACT = {
    "c_qfg_only": {
        "decision": "SELECT_C_QFG_ONLY",
        "variant": "qfg_only",
        "training_uses_tss": False,
    },
    "d_tss_qfg": {
        "decision": "SELECT_D_TSS_QFG",
        "variant": "tss_qfg",
        "training_uses_tss": True,
    },
}
SUCCESSFUL_CANDIDATE_STATES = {
    "RELATIVE_IMPROVED",
    "PARETO_MIXED_TRADEOFF",
}
PAIRED_RUN_SPECS = {
    "e_qfg_dlr": {
        "variant": "qfg_dlr",
        "run_tag": "formal800_qfg_dlr_control",
        "relative_run_directory": (
            "qfg_dlr_lane/NUDT-SIRST/qfg_dlr/"
            "seed_42_formal800_qfg_dlr_control"
        ),
    },
    "f_tss_qfg_dlr": {
        "variant": "tss_qfg_dlr",
        "run_tag": "formal800_tss_qfg_dlr_ramp100",
        "relative_run_directory": (
            "tss_qfg_dlr_lane/NUDT-SIRST/tss_qfg_dlr/"
            "seed_42_formal800_tss_qfg_dlr_ramp100"
        ),
    },
}
TERMINAL_RECEIPT_PHASES = {"no_fallback", "paired_training_complete"}
NONTERMINAL_RECEIPT_PHASES = {
    "paired_launching",
    "paired_launch_retryable_failure",
    "paired_launch_permanent_failure",
}
MEANINGFUL_IMPROVEMENT_BASIS = (
    "frozen_decision_f_f_non_isolated_five_objective_"
    "joint_pool_including_baseline"
)


class ControllerError(RuntimeError):
    """Base class for controller failures with an explicit exit contract."""

    exit_code = EXIT_PERMANENT


class ContractError(ControllerError):
    """A permanent artifact or configuration contract violation."""


class TerminalPending(ControllerError):
    """A retryable absence of a required terminal artifact."""

    exit_code = EXIT_RETRY


class PairedClaimBusy(ControllerError):
    """Another worker owns the one paired fallback claim."""

    exit_code = EXIT_RETRY


@dataclass(frozen=True)
class ControllerConfig:
    repo_root: Path = REPO_ROOT
    selection: Path = DEFAULT_SELECTION
    selection_markdown: Path = DEFAULT_SELECTION_MARKDOWN
    deployment_artifact: Path = DEFAULT_DEPLOYMENT_ARTIFACT
    deployment_manifest: Path = DEFAULT_DEPLOYMENT_MANIFEST
    closure_lock: Path = DEFAULT_CLOSURE_LOCK
    launcher: Path = DEFAULT_LAUNCHER
    dlr_source_lock: Path = DEFAULT_DLR_SOURCE_LOCK
    dlr_result_root: Path = DEFAULT_DLR_RESULT_ROOT
    receipt: Path = DEFAULT_RECEIPT
    paired_lock: Path = DEFAULT_PAIRED_LOCK

    def resolved(self) -> "ControllerConfig":
        return replace(
            self,
            **{
                field: Path(getattr(self, field)).expanduser().resolve()
                for field in (
                    "repo_root",
                    "selection",
                    "selection_markdown",
                    "deployment_artifact",
                    "deployment_manifest",
                    "closure_lock",
                    "launcher",
                    "dlr_source_lock",
                    "dlr_result_root",
                    "receipt",
                    "paired_lock",
                )
            },
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{label} must be one finite number")
    return float(value)


def _canonical(value: Any) -> Any:
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
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical JSON: {error}") from error


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_json(value: Any) -> str:
    return json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _regular_file(
    path: Path,
    label: str,
    *,
    missing_is_pending: bool,
) -> Path:
    value = Path(path)
    if not value.exists() and not value.is_symlink():
        if missing_is_pending:
            raise TerminalPending(f"{label} is not published: {value}")
        raise ContractError(f"{label} is missing: {value}")
    if value.is_symlink() or not value.is_file():
        raise ContractError(f"{label} must be a regular non-symlink file: {value}")
    return value


def _regular_directory(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_dir():
        raise ContractError(
            f"{label} must be a regular non-symlink directory: {value}"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(
    path: Path,
    label: str,
    *,
    missing_is_pending: bool,
    require_pretty_canonical: bool = False,
) -> dict[str, Any]:
    source = _regular_file(
        path,
        label,
        missing_is_pending=missing_is_pending,
    )

    def reject_constant(token: str) -> None:
        raise ContractError(f"{label} contains non-finite constant {token}")

    try:
        raw = source.read_bytes()
        payload = json.loads(
            raw,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain one JSON object")
    if require_pretty_canonical and raw != _pretty_json_bytes(payload):
        raise ContractError(f"{label} is not canonical pretty JSON")
    return payload


def _file_binding(
    path: Path,
    label: str,
    *,
    missing_is_pending: bool,
) -> dict[str, Any]:
    source = _regular_file(
        path,
        label,
        missing_is_pending=missing_is_pending,
    )
    return {
        "path": str(source.resolve()),
        "sha256": _sha256_file(source),
    }


def _strict_closure_loader(
    closure_lock: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Imported lazily so stub CPU tests need not import torch or model code.
    from experiments import (  # pylint: disable=import-outside-toplevel
        tpd_ner_v4_qfg_v2_croa_posttraining_policy as policy,
    )

    return policy.load_closure_lock(closure_lock, verify_sources=True)


def _strict_deployment_validator(
    *,
    selection_path: Path,
    artifact_path: Path,
    manifest_path: Path,
    closure_lock_path: Path,
) -> dict[str, Any]:
    # The deployer rebuilds the complete selection from live, evaluator-
    # validated inputs and then validates the exported checkpoint and manifest.
    from experiments import (  # pylint: disable=import-outside-toplevel
        deploy_tpd_ner_v4_qfg_v2_croa_formal800 as deploy,
    )

    return deploy.validate_deployment_closure(
        selection_path=selection_path,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        closure_lock_path=closure_lock_path,
    )


def _strict_selection_markdown(
    report: Mapping[str, Any],
    path: Path,
) -> None:
    from experiments import (  # pylint: disable=import-outside-toplevel
        postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as selector,
    )

    # The write-once JSON is canonicalized with sorted object keys, while the
    # published Markdown intentionally follows the formal experiment order.
    # Restore that presentation order before rendering; metric content and
    # every binding remain unchanged.
    method_order = (
        "baseline",
        "v1",
        "v2",
        "v3",
        "v4",
        "a_control",
        "b_tss",
        "c_qfg_only",
        "d_tss_qfg",
    )
    methods = report.get("methods")
    _require(
        isinstance(methods, Mapping) and set(methods) == set(method_order),
        "current final-selection method matrix differs",
    )
    ordered_report = copy.deepcopy(dict(report))
    ordered_report["methods"] = {
        method_id: copy.deepcopy(methods[method_id])
        for method_id in method_order
    }
    expected = selector.render_markdown(ordered_report).encode("utf-8")
    observed = _regular_file(
        path,
        "current final-selection Markdown",
        missing_is_pending=True,
    ).read_bytes()
    _require(
        observed == expected,
        "current final-selection Markdown conflicts with the live report",
    )


ClosureLoader = Callable[
    [Path],
    tuple[dict[str, Any], dict[str, Any]],
]
DeploymentValidator = Callable[..., dict[str, Any]]
MarkdownValidator = Callable[[Mapping[str, Any], Path], None]


def _validate_launcher_and_dlr_lock(
    config: ControllerConfig,
) -> dict[str, Any]:
    launcher = _regular_file(
        config.launcher,
        "paired DLR+ramp100 launcher",
        missing_is_pending=False,
    )
    mode = launcher.stat().st_mode
    _require(
        bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
        f"paired launcher is not executable: {launcher}",
    )
    source_lock = _load_json(
        config.dlr_source_lock,
        "DLR+ramp100 51-source lock",
        missing_is_pending=False,
        require_pretty_canonical=True,
    )
    _require(
        source_lock.get("schema") == DLR_SOURCE_LOCK_SCHEMA,
        "DLR+ramp100 source-lock schema differs",
    )
    _require(
        source_lock.get("lock_kind") == "training",
        "DLR+ramp100 source-lock kind differs",
    )
    _require(
        source_lock.get("source_count") == EXPECTED_DLR_SOURCE_COUNT,
        "DLR+ramp100 source count is not 51",
    )
    source_sha256 = source_lock.get("source_sha256")
    _require(
        isinstance(source_sha256, Mapping)
        and len(source_sha256) == EXPECTED_DLR_SOURCE_COUNT,
        "DLR+ramp100 source hash matrix differs",
    )
    _require(
        all(_is_sha256(value) for value in source_sha256.values()),
        "DLR+ramp100 source hash matrix contains an invalid SHA",
    )
    launcher_text = launcher.read_text(encoding="utf-8")
    for token in (
        "paired_gpu2_uuid=\"GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562\"",
        "paired_gpu3_uuid=\"GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3\"",
        "wait_for_gpu_idle=false",
        "flock -n 9",
        "--seed 42",
        "--epochs 800",
    ):
        _require(
            token in launcher_text,
            f"paired launcher fixed contract token is missing: {token}",
        )
    return {
        "path": str(launcher.resolve()),
        "sha256": _sha256_file(launcher),
        "verified_regular_executable": True,
        "fixed_physical_gpus": [2, 3],
        "wait_for_gpu_idle": False,
        "paired_flock": True,
        "result_root": str(config.dlr_result_root.resolve()),
        "source_lock": {
            "path": str(config.dlr_source_lock.resolve()),
            "sha256": _sha256_file(config.dlr_source_lock),
            "schema": source_lock["schema"],
            "source_count": EXPECTED_DLR_SOURCE_COUNT,
        },
    }


def _policy_action(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    deployment_validation: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema") == CURRENT_SELECTION_SCHEMA,
        "current final-selection schema differs",
    )
    _require(report.get("status") == "complete", "current selection is incomplete")
    _require(report.get("dataset") == "NUDT-SIRST", "selection dataset differs")
    _require(
        report.get("training_seed") == TRAINING_SEED,
        "selection training seed is not 42",
    )
    _require(
        report.get("split_seed") == SPLIT_SEED,
        "selection split seed differs",
    )
    _require(
        report.get("official_test_accessed") is False,
        "selection official-test field differs",
    )
    selection = report.get("selection")
    deployment = report.get("deployment_selection")
    assessments = report.get("candidate_assessments")
    methods = report.get("methods")
    _require(isinstance(selection, Mapping), "selection object is missing")
    _require(isinstance(deployment, Mapping), "deployment selection is missing")
    _require(isinstance(assessments, Mapping), "candidate assessments are missing")
    _require(isinstance(methods, Mapping), "method matrix is missing")
    selected_point = deployment.get("selected")
    _require(
        isinstance(selected_point, Mapping),
        "deployment selected point is missing",
    )
    selected_method = selection.get("selected_method_id")
    selected_variant = selection.get("selected_variant")
    decision = selection.get("decision")
    _require(
        report.get("decision") == decision,
        "top-level and selection decisions differ",
    )
    _require(
        selected_point.get("method_id") == selected_method,
        "deployment selected method differs",
    )
    _require(
        selected_point.get("variant") == selected_variant,
        "deployment selected variant differs",
    )
    _require(
        deployment.get("selected_point_is_checkpoint_local") is True
        and deployment.get("cross_checkpoint_metric_stitching") is False,
        "deployment is not one checkpoint-local atomic point",
    )
    _finite_number(
        selected_point.get("threshold"),
        "selected deployment threshold",
    )
    _require(
        deployment_validation.get("status") == "complete"
        and deployment_validation.get("verified") is True,
        "deployment validator did not establish a complete closure",
    )
    _require(
        deployment_validation.get("selected_method_id") == selected_method,
        "deployment validator selected a different method",
    )
    _require(
        manifest.get("schema") == CURRENT_DEPLOYMENT_MANIFEST_SCHEMA,
        "current deployment-manifest schema differs",
    )
    _require(
        manifest.get("status") == "complete",
        "current deployment manifest is incomplete",
    )
    _require(
        manifest.get("selected_method_id") == selected_method
        and manifest.get("selected_variant") == selected_variant,
        "deployment manifest selected recipe differs",
    )
    _require(
        manifest.get("official_test_accessed") is False,
        "deployment manifest official-test field differs",
    )
    manifest_selection = manifest.get("final_selection")
    _require(
        isinstance(manifest_selection, Mapping)
        and manifest_selection.get("sha256"),
        "deployment manifest selection binding is missing",
    )
    _require(
        manifest.get("cross_checkpoint_metric_stitching") is False
        and manifest.get("selected_point_is_checkpoint_local") is True,
        "deployment manifest atomic-point policy differs",
    )

    is_qfg_method = selected_method in QFG_METHOD_CONTRACT
    if is_qfg_method:
        contract = QFG_METHOD_CONTRACT[str(selected_method)]
        assessment = assessments.get(selected_method)
        method = methods.get(selected_method)
        _require(
            decision == contract["decision"],
            "selected QFG method has a different frozen decision",
        )
        _require(
            selected_variant == contract["variant"],
            "selected QFG method has a different variant",
        )
        _require(
            isinstance(method, Mapping)
            and method.get("variant") == selected_variant,
            "selected QFG method matrix entry differs",
        )
        _require(
            isinstance(assessment, Mapping),
            "selected QFG candidate assessment is missing",
        )
        candidate_status = assessment.get("status")
        _require(
            candidate_status in SUCCESSFUL_CANDIDATE_STATES,
            "selected QFG candidate has no successful frozen-policy state",
        )
        comparison_ids = assessment.get("comparison_method_ids")
        _require(
            isinstance(comparison_ids, list) and "baseline" in comparison_ids,
            "selected QFG assessment does not include baseline in its pool",
        )
        non_isolated_count = assessment.get("non_isolated_support_count")
        _require(
            isinstance(non_isolated_count, int)
            and not isinstance(non_isolated_count, bool)
            and non_isolated_count > 0,
            "selected QFG candidate has no non-isolated policy support",
        )
        for label, value in (
            (
                "selection query-FG stage success",
                selection.get("query_fg_stage_success"),
            ),
            (
                "top-level query-FG stage success",
                report.get("query_fg_stage_success"),
            ),
            (
                "final-model engineering selection",
                report.get("final_model_engineering_selected"),
            ),
            (
                "final-model establishment",
                report.get("final_model_established"),
            ),
        ):
            _require(value is True, f"{label} is not true")
        _require(
            selection.get("final_training_uses_tss")
            is contract["training_uses_tss"],
            "selected QFG TSS training state differs",
        )
        _require(
            selection.get("final_inference_uses_tss") is False
            and report.get("final_inference_uses_tss") is False,
            "TSS must not remain in final inference",
        )
        _require(
            manifest.get("export_mode") == "strict_head_free_qfg_export",
            "selected full QFG model was not exported in strict QFG mode",
        )
        return {
            "authoritative_action": "no_fallback",
            "selected_method_id": selected_method,
            "selected_variant": selected_variant,
            "selected_candidate_status": candidate_status,
            "decision": decision,
            "query_fg_stage_success": True,
            "final_model_engineering_selected": True,
            "final_model_established": True,
            "meaningful_overall_improvement_by_frozen_policy": True,
            "meaningful_improvement_basis": MEANINGFUL_IMPROVEMENT_BASIS,
            "paired_required": False,
        }

    # The frozen current selector's only non-QFG terminal choice is V4.
    # Keeping this branch expressed as "non-QFG" makes future old-method
    # fallbacks conservative, while the live deployer remains the strict
    # authority over which methods can actually be published.
    _require(
        selected_method == "v4",
        f"unsupported policy-selected non-QFG method: {selected_method!r}",
    )
    _require(
        decision == "FALLBACK_TO_FROZEN_V4",
        "V4 fallback decision differs",
    )
    _require(
        selection.get("query_fg_stage_success") is False
        and report.get("query_fg_stage_success") is False
        and report.get("final_model_engineering_selected") is False
        and report.get("final_model_established") is False,
        "non-QFG fallback has inconsistent QFG success fields",
    )
    _require(
        manifest.get("export_mode") == "write_once_native_v4_checkpoint_copy",
        "V4 fallback deployment mode differs",
    )
    return {
        "authoritative_action": "launch_paired",
        "selected_method_id": selected_method,
        "selected_variant": selected_variant,
        "selected_candidate_status": None,
        "decision": decision,
        "query_fg_stage_success": False,
        "final_model_engineering_selected": False,
        "final_model_established": False,
        "meaningful_overall_improvement_by_frozen_policy": False,
        "meaningful_improvement_basis": MEANINGFUL_IMPROVEMENT_BASIS,
        "paired_required": True,
    }


def evaluate_current_terminal(
    config: ControllerConfig,
    *,
    closure_loader: ClosureLoader = _strict_closure_loader,
    deployment_validator: DeploymentValidator = (
        _strict_deployment_validator
    ),
    markdown_validator: MarkdownValidator = _strict_selection_markdown,
) -> dict[str, Any]:
    """Validate the current terminal closure and return the policy action."""

    config = config.resolved()
    _regular_directory(config.repo_root, "repository root")
    # Explicit presence checks make every incomplete finalizer state retryable
    # before a deeper live validator can classify malformed artifacts.
    for path, label in (
        (config.selection, "current final selection"),
        (config.selection_markdown, "current final-selection Markdown"),
        (config.deployment_artifact, "current deployment artifact"),
        (config.deployment_manifest, "current deployment manifest"),
        (config.closure_lock, "current 15-source closure lock"),
    ):
        _regular_file(path, label, missing_is_pending=True)

    try:
        closure_payload, closure_binding = closure_loader(config.closure_lock)
    except FileNotFoundError as error:
        raise TerminalPending(
            f"current live closure dependency is not published: {error}"
        ) from error
    except TerminalPending:
        raise
    except Exception as error:  # strict external validator boundary
        raise ContractError(
            f"current 15-source closure validation failed: {error}"
        ) from error
    _require(
        isinstance(closure_payload, Mapping)
        and closure_payload.get("schema") == CURRENT_CLOSURE_LOCK_SCHEMA,
        "current closure-lock schema differs",
    )
    _require(
        closure_payload.get("status") == "complete",
        "current closure lock is incomplete",
    )
    _require(
        closure_payload.get("source_count")
        == EXPECTED_CURRENT_CLOSURE_SOURCE_COUNT,
        "current post-training closure source count is not 15",
    )
    _require(
        isinstance(closure_payload.get("source_sha256"), Mapping)
        and len(closure_payload["source_sha256"])
        == EXPECTED_CURRENT_CLOSURE_SOURCE_COUNT,
        "current closure source matrix differs",
    )
    _require(
        isinstance(closure_binding, Mapping)
        and closure_binding.get("verified_live") is True
        and closure_binding.get("source_count")
        == EXPECTED_CURRENT_CLOSURE_SOURCE_COUNT
        and _is_sha256(closure_binding.get("sha256")),
        "current closure live binding differs",
    )

    try:
        deployment_validation = deployment_validator(
            selection_path=config.selection,
            artifact_path=config.deployment_artifact,
            manifest_path=config.deployment_manifest,
            closure_lock_path=config.closure_lock,
        )
    except FileNotFoundError as error:
        raise TerminalPending(
            f"current deployment closure dependency is not published: {error}"
        ) from error
    except TerminalPending:
        raise
    except Exception as error:  # strict external validator boundary
        raise ContractError(
            f"current final selection/deployment validation failed: {error}"
        ) from error
    _require(
        isinstance(deployment_validation, Mapping),
        "deployment validator returned no object",
    )

    report = _load_json(
        config.selection,
        "current final selection",
        missing_is_pending=True,
        require_pretty_canonical=True,
    )
    manifest = _load_json(
        config.deployment_manifest,
        "current deployment manifest",
        missing_is_pending=True,
        require_pretty_canonical=True,
    )
    try:
        markdown_validator(report, config.selection_markdown)
    except FileNotFoundError as error:
        raise TerminalPending(
            f"current selection Markdown is not published: {error}"
        ) from error
    except TerminalPending:
        raise
    except Exception as error:
        raise ContractError(
            f"current final-selection Markdown validation failed: {error}"
        ) from error

    action = _policy_action(report, manifest, deployment_validation)
    selection_binding = _file_binding(
        config.selection,
        "current final selection",
        missing_is_pending=True,
    )
    manifest_selection = manifest["final_selection"]
    _require(
        manifest_selection.get("path") == selection_binding["path"]
        and manifest_selection.get("sha256") == selection_binding["sha256"],
        "deployment manifest does not bind the current final selection",
    )
    closure_file_binding = _file_binding(
        config.closure_lock,
        "current closure lock",
        missing_is_pending=True,
    )
    _require(
        closure_binding.get("path") == closure_file_binding["path"]
        and closure_binding.get("sha256") == closure_file_binding["sha256"],
        "live closure binding does not bind the requested 15-source lock",
    )
    manifest_closure = manifest.get("posttraining_closure_source_lock")
    _require(
        isinstance(manifest_closure, Mapping)
        and manifest_closure.get("path") == closure_file_binding["path"]
        and manifest_closure.get("sha256") == closure_file_binding["sha256"]
        and manifest_closure.get("source_count")
        == EXPECTED_CURRENT_CLOSURE_SOURCE_COUNT,
        "deployment manifest does not bind the current 15-source lock",
    )
    launcher_binding = _validate_launcher_and_dlr_lock(config)
    return {
        "schema": STATUS_SCHEMA,
        "status": "complete",
        "official_test_accessed": False,
        **copy.deepcopy(action),
        "current_terminal": {
            "final_selection": {
                **selection_binding,
                "schema": report["schema"],
            },
            "final_selection_markdown": _file_binding(
                config.selection_markdown,
                "current final-selection Markdown",
                missing_is_pending=True,
            ),
            "deployment_artifact": _file_binding(
                config.deployment_artifact,
                "current deployment artifact",
                missing_is_pending=True,
            ),
            "deployment_manifest": {
                **_file_binding(
                    config.deployment_manifest,
                    "current deployment manifest",
                    missing_is_pending=True,
                ),
                "schema": manifest["schema"],
            },
            "posttraining_closure_source_lock": {
                **closure_file_binding,
                "schema": closure_payload["schema"],
                "source_count": EXPECTED_CURRENT_CLOSURE_SOURCE_COUNT,
                "verified_live": True,
            },
        },
        "launcher_contract": launcher_binding,
    }


def _paired_run_binding(
    run_dir: Path,
    *,
    method_id: str,
    spec: Mapping[str, Any],
    dlr_source_lock_sha256: str,
) -> dict[str, Any]:
    _regular_directory(run_dir, f"{method_id} paired run directory")
    summary = _load_json(
        run_dir / "summary.json",
        f"{method_id} paired completion summary",
        missing_is_pending=True,
        require_pretty_canonical=True,
    )
    variant = spec["variant"]
    _require(summary.get("status") == "complete", f"{method_id} is incomplete")
    _require(summary.get("variant") == variant, f"{method_id} variant differs")
    _require(
        summary.get("candidate_variant") == variant,
        f"{method_id} candidate variant differs",
    )
    _require(
        summary.get("seed") == TRAINING_SEED,
        f"{method_id} training seed is not 42",
    )
    _require(
        summary.get("split_seed") == SPLIT_SEED,
        f"{method_id} split seed differs",
    )
    _require(
        summary.get("official_test_accessed") is False,
        f"{method_id} official-test field differs",
    )
    formal_contract = summary.get("formal_contract")
    _require(
        isinstance(formal_contract, Mapping)
        and formal_contract.get("epochs") == FORMAL_EPOCHS,
        f"{method_id} formal epoch contract differs",
    )
    identity = summary.get("run_identity")
    _require(isinstance(identity, Mapping), f"{method_id} run identity is missing")
    _require(
        identity.get("variant") == variant
        and identity.get("seed") == TRAINING_SEED
        and identity.get("split_seed") == SPLIT_SEED,
        f"{method_id} run identity differs",
    )
    expected_run_id_suffix = (
        f"NUDT-SIRST:{variant}:seed-42:split-20260722:"
        f"{spec['run_tag']}"
    )
    _require(
        isinstance(identity.get("run_id"), str)
        and identity["run_id"].endswith(expected_run_id_suffix),
        f"{method_id} run ID differs",
    )
    source_locks = identity.get("source_locks")
    _require(
        isinstance(source_locks, Mapping)
        and dlr_source_lock_sha256 in source_locks.values(),
        f"{method_id} does not bind the DLR 51-source lock",
    )
    metrics_path = _regular_file(
        run_dir / "metrics.jsonl",
        f"{method_id} metrics",
        missing_is_pending=True,
    )
    raw_lines = metrics_path.read_bytes().splitlines()
    _require(
        len(raw_lines) == FORMAL_EPOCHS
        and all(line.strip() for line in raw_lines),
        f"{method_id} metrics are not contiguous 800 rows",
    )
    try:
        events = [json.loads(line) for line in raw_lines]
    except json.JSONDecodeError as error:
        raise ContractError(f"{method_id} metrics are invalid JSONL: {error}") from error
    _require(
        [event.get("epoch") for event in events]
        == list(range(1, FORMAL_EPOCHS + 1)),
        f"{method_id} metric epochs are not 1..800",
    )
    active = _load_json(
        run_dir / "exact_journal/active.json",
        f"{method_id} exact active marker",
        missing_is_pending=True,
        require_pretty_canonical=True,
    )
    _require(
        active.get("epoch") == FORMAL_EPOCHS,
        f"{method_id} exact journal is not committed at epoch 800",
    )
    files = {
        "summary": run_dir / "summary.json",
        "metrics": metrics_path,
        "active_marker": run_dir / "exact_journal/active.json",
        "last_checkpoint": run_dir / "last.pth.tar",
        "pd_primary_checkpoint": run_dir / "best.pth.tar",
        "miou_secondary_checkpoint": run_dir / "best_miou.pth.tar",
    }
    return {
        "method_id": method_id,
        "variant": variant,
        "run_tag": spec["run_tag"],
        "run_directory": str(run_dir.resolve()),
        "formal_epochs_complete": FORMAL_EPOCHS,
        "files": {
            name: _file_binding(
                path,
                f"{method_id} {name}",
                missing_is_pending=True,
            )
            for name, path in files.items()
        },
    }


def collect_paired_training_closure(
    config: ControllerConfig,
) -> dict[str, Any]:
    config = config.resolved()
    source_lock_binding = _validate_launcher_and_dlr_lock(config)["source_lock"]
    runs = {}
    for method_id, spec in PAIRED_RUN_SPECS.items():
        run_dir = config.dlr_result_root / spec["relative_run_directory"]
        runs[method_id] = _paired_run_binding(
            run_dir,
            method_id=method_id,
            spec=spec,
            dlr_source_lock_sha256=source_lock_binding["sha256"],
        )
    return {
        "result_root": str(config.dlr_result_root.resolve()),
        "training_complete": True,
        "formal_epochs": FORMAL_EPOCHS,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "source_lock": source_lock_binding,
        "runs": runs,
        "posttraining_selection_complete": False,
        "posttraining_deployment_complete": False,
    }


def _receipt_payload(
    evaluation: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    launcher_invoked: bool,
    launcher_exit_status: int | None,
    paired_training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action = evaluation["authoritative_action"]
    paired_required = bool(evaluation["paired_required"])
    training_complete = (
        bool(paired_training and paired_training.get("training_complete"))
        if paired_required
        else False
    )
    if status == "complete":
        _require(
            phase in TERMINAL_RECEIPT_PHASES,
            "complete receipt has a nonterminal phase",
        )
    else:
        _require(
            phase in NONTERMINAL_RECEIPT_PHASES,
            "noncomplete receipt has an invalid phase",
        )
    _require(
        (phase == "no_fallback") == (action == "no_fallback"),
        "receipt phase conflicts with authoritative action",
    )
    _require(
        (phase == "paired_training_complete") == training_complete,
        "paired terminal phase conflicts with training completion",
    )
    launcher = copy.deepcopy(evaluation["launcher_contract"])
    launcher.update(
        {
            "invoked": bool(launcher_invoked),
            "exit_status": launcher_exit_status,
        }
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "phase": phase,
        "official_test_accessed": False,
        "authoritative_action": action,
        "selected_method_id": evaluation["selected_method_id"],
        "selected_variant": evaluation["selected_variant"],
        "selected_candidate_status": evaluation[
            "selected_candidate_status"
        ],
        "decision": evaluation["decision"],
        "query_fg_stage_success": evaluation["query_fg_stage_success"],
        "final_model_engineering_selected": evaluation[
            "final_model_engineering_selected"
        ],
        "final_model_established": evaluation["final_model_established"],
        "meaningful_overall_improvement_by_frozen_policy": evaluation[
            "meaningful_overall_improvement_by_frozen_policy"
        ],
        "meaningful_improvement_basis": evaluation[
            "meaningful_improvement_basis"
        ],
        "paired_required": paired_required,
        "current_terminal": copy.deepcopy(evaluation["current_terminal"]),
        "launcher": launcher,
        "paired": (
            {
                "required": False,
                "training_complete": False,
                "posttraining_selection_complete": False,
                "posttraining_deployment_complete": False,
                "result_root": evaluation["launcher_contract"][
                    "result_root"
                ],
                "runs": {},
            }
            if not paired_required
            else {
                "required": True,
                "training_complete": training_complete,
                "posttraining_selection_complete": False,
                "posttraining_deployment_complete": False,
                "result_root": (
                    paired_training.get("result_root")
                    if paired_training
                    else evaluation["launcher_contract"]["result_root"]
                ),
                "formal_epochs": (
                    paired_training.get("formal_epochs")
                    if paired_training
                    else FORMAL_EPOCHS
                ),
                "training_seed": TRAINING_SEED,
                "split_seed": SPLIT_SEED,
                "source_lock": (
                    copy.deepcopy(paired_training["source_lock"])
                    if paired_training
                    else copy.deepcopy(
                        evaluation["launcher_contract"]["source_lock"]
                    )
                ),
                "runs": (
                    copy.deepcopy(paired_training["runs"])
                    if paired_training
                    else {}
                ),
            }
        ),
        "terminal_for_fallback_controller": (
            status == "complete" and phase in TERMINAL_RECEIPT_PHASES
        ),
        # A paired-training receipt intentionally cannot close the wider
        # reproducibility package: the independent DLR final selection,
        # deployment export and deployment manifest must still be published
        # and validated by their own frozen post-training closure.
        "terminal_for_reproducibility_manifest": (
            status == "complete" and phase == "no_fallback"
        ),
        "receipt_write_policy": "atomic_state_transition",
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.is_symlink():
        raise ContractError(f"receipt path must not be a symlink: {destination}")
    parent = destination.parent
    if parent.is_symlink():
        raise ContractError(f"receipt parent must not be a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ContractError(f"receipt parent is not a regular directory: {parent}")
    data = _pretty_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _load_existing_receipt(path: Path) -> dict[str, Any] | None:
    value = Path(path)
    if not value.exists() and not value.is_symlink():
        return None
    receipt = _load_json(
        value,
        "fallback controller receipt",
        missing_is_pending=False,
        require_pretty_canonical=True,
    )
    _require(receipt.get("schema") == RECEIPT_SCHEMA, "receipt schema differs")
    _require(
        receipt.get("official_test_accessed") is False,
        "receipt official-test field differs",
    )
    status = receipt.get("status")
    phase = receipt.get("phase")
    if status == "complete":
        _require(phase in TERMINAL_RECEIPT_PHASES, "receipt terminal phase differs")
        _require(
            receipt.get("terminal_for_fallback_controller") is True,
            "complete receipt is not terminal for the fallback controller",
        )
        _require(
            receipt.get("terminal_for_reproducibility_manifest")
            is (phase == "no_fallback"),
            "receipt reproducibility-terminal field differs",
        )
    else:
        _require(
            phase in NONTERMINAL_RECEIPT_PHASES,
            "receipt nonterminal phase differs",
        )
        _require(
            receipt.get("terminal_for_fallback_controller") is False,
            "noncomplete receipt is terminal for the fallback controller",
        )
        _require(
            receipt.get("terminal_for_reproducibility_manifest") is False,
            "nonterminal receipt is marked terminal for reproducibility",
        )
    return receipt


def _receipt_matches_evaluation(
    receipt: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> None:
    for key in (
        "authoritative_action",
        "selected_method_id",
        "selected_variant",
        "selected_candidate_status",
        "decision",
        "query_fg_stage_success",
        "final_model_engineering_selected",
        "final_model_established",
        "meaningful_overall_improvement_by_frozen_policy",
        "meaningful_improvement_basis",
        "paired_required",
        "current_terminal",
    ):
        _require(
            _canonical(receipt.get(key)) == _canonical(evaluation.get(key)),
            f"receipt field conflicts with current live terminal: {key}",
        )
    launcher = receipt.get("launcher")
    _require(isinstance(launcher, Mapping), "receipt launcher binding is missing")
    for key, expected in evaluation["launcher_contract"].items():
        _require(
            _canonical(launcher.get(key)) == _canonical(expected),
            f"receipt launcher binding differs: {key}",
        )


class _NonBlockingFlock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def __enter__(self) -> "_NonBlockingFlock":
        if self.path.is_symlink():
            raise ContractError(f"paired claim must not be a symlink: {self.path}")
        parent = self.path.parent
        if parent.is_symlink():
            raise ContractError(
                f"paired claim parent must not be a symlink: {parent}"
            )
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ContractError(
                f"paired claim parent must be a regular directory: {parent}"
            )
        self._handle = self.path.open("a+b")
        try:
            fcntl.flock(
                self._handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise PairedClaimBusy(
                f"paired fallback claim is busy: {self.path}"
            ) from error
        return self

    def __exit__(self, *_: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


EvaluationFunction = Callable[[ControllerConfig], dict[str, Any]]
LauncherRunner = Callable[[ControllerConfig], int]
PairedCollector = Callable[[ControllerConfig], dict[str, Any]]


def _run_launcher(config: ControllerConfig) -> int:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TPD_NER_DLR_RAMP100_")
    }
    environment.update(
        {
            "TPD_NER_DLR_RAMP100_REPO": str(config.repo_root),
            "TPD_NER_DLR_RAMP100_SOURCE_LOCK": str(
                config.dlr_source_lock
            ),
            "TPD_NER_DLR_RAMP100_RESULT_ROOT": str(
                config.dlr_result_root
            ),
            "TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT": str(
                config.dlr_result_root / "qfg_dlr_lane"
            ),
            "TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT": str(
                config.dlr_result_root / "tss_qfg_dlr_lane"
            ),
        }
    )
    completed = subprocess.run(
        [str(config.launcher)],
        cwd=str(config.repo_root),
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def run_worker(
    config: ControllerConfig,
    *,
    evaluator: EvaluationFunction = evaluate_current_terminal,
    launcher_runner: LauncherRunner = _run_launcher,
    paired_collector: PairedCollector = collect_paired_training_closure,
) -> dict[str, Any]:
    """Execute one strict worker attempt under the paired nonblocking flock."""

    config = config.resolved()
    # Missing terminal artifacts return 75 without creating controller state.
    evaluator(config)
    with _NonBlockingFlock(config.paired_lock):
        # Re-evaluate under the claim to bind the exact artifacts used.
        evaluation = evaluator(config)
        existing = _load_existing_receipt(config.receipt)
        if existing is not None:
            _receipt_matches_evaluation(existing, evaluation)
            if existing["status"] == "complete":
                if existing["phase"] == "paired_training_complete":
                    observed = paired_collector(config)
                    _require(
                        _canonical(existing["paired"]["runs"])
                        == _canonical(observed["runs"]),
                        "paired terminal receipt conflicts with live E/F runs",
                    )
                return copy.deepcopy(existing)
            if existing["phase"] == "paired_launch_permanent_failure":
                raise ContractError(
                    "previous paired launcher attempt failed permanently"
                )

        if evaluation["authoritative_action"] == "no_fallback":
            receipt = _receipt_payload(
                evaluation,
                status="complete",
                phase="no_fallback",
                launcher_invoked=False,
                launcher_exit_status=None,
            )
            _write_receipt(config.receipt, receipt)
            return receipt

        launching = _receipt_payload(
            evaluation,
            status="in_progress",
            phase="paired_launching",
            launcher_invoked=True,
            launcher_exit_status=None,
        )
        _write_receipt(config.receipt, launching)
        launcher_status = int(launcher_runner(config))
        if launcher_status != EXIT_OK:
            permanent = launcher_status == EXIT_PERMANENT
            failure = _receipt_payload(
                evaluation,
                status="failed" if permanent else "retryable",
                phase=(
                    "paired_launch_permanent_failure"
                    if permanent
                    else "paired_launch_retryable_failure"
                ),
                launcher_invoked=True,
                launcher_exit_status=launcher_status,
            )
            _write_receipt(config.receipt, failure)
            if permanent:
                raise ContractError(
                    "paired launcher returned permanent exit status 64"
                )
            raise TerminalPending(
                "paired launcher was interrupted or returned a retryable "
                f"status: {launcher_status}"
            )
        try:
            paired_training = paired_collector(config)
        except ContractError:
            raise
        except Exception as error:
            raise TerminalPending(
                f"paired launcher exited 0 but E/F closure is incomplete: {error}"
            ) from error
        complete = _receipt_payload(
            evaluation,
            status="complete",
            phase="paired_training_complete",
            launcher_invoked=True,
            launcher_exit_status=EXIT_OK,
            paired_training=paired_training,
        )
        _write_receipt(config.receipt, complete)
        return complete


def read_status(
    config: ControllerConfig,
    *,
    evaluator: EvaluationFunction = evaluate_current_terminal,
) -> dict[str, Any]:
    config = config.resolved()
    evaluation = evaluator(config)
    receipt = _load_existing_receipt(config.receipt)
    if receipt is not None:
        _receipt_matches_evaluation(receipt, evaluation)
    return {
        "schema": STATUS_SCHEMA,
        "status": "complete",
        "mode": "status",
        "official_test_accessed": False,
        "writes_performed": False,
        "launcher_invoked": False,
        "evaluation": evaluation,
        "receipt": {
            "path": str(config.receipt),
            "state": "missing" if receipt is None else receipt["phase"],
            "fallback_controller_terminal": bool(
                receipt
                and receipt.get("terminal_for_fallback_controller")
                is True
            ),
            "reproducibility_manifest_terminal": bool(
                receipt
                and receipt.get("terminal_for_reproducibility_manifest")
                is True
            ),
            "sha256": (
                None
                if receipt is None
                else _sha256_file(config.receipt)
            ),
        },
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ContractError(f"command-line contract error: {message}")


def _argument_parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--selection-markdown",
        type=Path,
        default=DEFAULT_SELECTION_MARKDOWN,
    )
    parser.add_argument(
        "--deployment-artifact",
        type=Path,
        default=DEFAULT_DEPLOYMENT_ARTIFACT,
    )
    parser.add_argument(
        "--deployment-manifest",
        type=Path,
        default=DEFAULT_DEPLOYMENT_MANIFEST,
    )
    parser.add_argument(
        "--closure-source-lock",
        type=Path,
        default=DEFAULT_CLOSURE_LOCK,
    )
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument(
        "--dlr-source-lock",
        type=Path,
        default=DEFAULT_DLR_SOURCE_LOCK,
    )
    parser.add_argument(
        "--dlr-result-root",
        type=Path,
        default=DEFAULT_DLR_RESULT_ROOT,
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--paired-lock", type=Path, default=DEFAULT_PAIRED_LOCK)
    return parser


def _config_from_args(args: argparse.Namespace) -> ControllerConfig:
    return ControllerConfig(
        repo_root=args.repo_root,
        selection=args.selection,
        selection_markdown=args.selection_markdown,
        deployment_artifact=args.deployment_artifact,
        deployment_manifest=args.deployment_manifest,
        closure_lock=args.closure_source_lock,
        launcher=args.launcher,
        dlr_source_lock=args.dlr_source_lock,
        dlr_result_root=args.dlr_result_root,
        receipt=args.receipt,
        paired_lock=args.paired_lock,
    ).resolved()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _argument_parser().parse_args(argv)
        config = _config_from_args(args)
        if args.worker:
            result = run_worker(config)
            output = {
                "schema": STATUS_SCHEMA,
                "status": "complete",
                "mode": "worker",
                "official_test_accessed": False,
                "receipt_path": str(config.receipt),
                "receipt_sha256": _sha256_file(config.receipt),
                "phase": result["phase"],
                "authoritative_action": result["authoritative_action"],
                "paired_required": result["paired_required"],
                "launcher_invoked": result["launcher"]["invoked"],
                "launcher_exit_status": result["launcher"]["exit_status"],
            }
        elif args.dry_run:
            evaluation = evaluate_current_terminal(config)
            output = {
                "schema": STATUS_SCHEMA,
                "status": "complete",
                "mode": "dry-run",
                "official_test_accessed": False,
                "writes_performed": False,
                "launcher_invoked": False,
                "would_action": evaluation["authoritative_action"],
                "paired_required": evaluation["paired_required"],
                "selected_method_id": evaluation["selected_method_id"],
                "evaluation": evaluation,
            }
        else:
            output = read_status(config)
        print(_compact_json(output), flush=True)
        return EXIT_OK
    except ControllerError as error:
        print(
            "TPDNERV4QFG_DLR_FALLBACK_"
            f"{'RETRY' if error.exit_code == EXIT_RETRY else 'ABORT'} "
            f"reason={error}",
            file=sys.stderr,
            flush=True,
        )
        return error.exit_code
    except Exception as error:  # fail closed at the public entry point
        print(
            "TPDNERV4QFG_DLR_FALLBACK_ABORT "
            f"reason=unexpected_controller_failure:{error}",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_PERMANENT


__all__ = [
    "CURRENT_CLOSURE_LOCK_SCHEMA",
    "CURRENT_DEPLOYMENT_MANIFEST_SCHEMA",
    "CURRENT_SELECTION_SCHEMA",
    "ControllerConfig",
    "ContractError",
    "DEFAULT_RECEIPT",
    "EXIT_OK",
    "EXIT_PERMANENT",
    "EXIT_RETRY",
    "PairedClaimBusy",
    "RECEIPT_SCHEMA",
    "STATUS_SCHEMA",
    "TerminalPending",
    "collect_paired_training_closure",
    "evaluate_current_terminal",
    "main",
    "read_status",
    "run_worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
