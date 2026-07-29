#!/usr/bin/env python3
"""Independent formal800 comparison and selection for paired DLR arms E/F.

References are the live-validated baseline, V4 parent, and A/B/C/D methods
from the frozen QFG closure.  E/F each contribute their own ``best`` and
``best_miou`` checkpoint-local sweeps.  The report compares fixed threshold
0.5 and all five preregistered Fa budgets without pooling thresholds or
metrics across checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as reference_selector,
)
from experiments import (  # noqa: E402
    tpd_ner_v4_qfg_v2_croa_dlr_ramp100_posttraining_policy as policy,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as trainer,
)


SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "formal800_final_selection_v1"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "formal800_selection_action_v1"
)
PREFLIGHT_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "formal800_selection_preflight_v1"
)
REFERENCE_METHOD_IDS = (
    "baseline",
    "v4",
    "a_control",
    "b_tss",
    "c_qfg_only",
    "d_tss_qfg",
)
DLR_EVALUATOR_MODULE = (
    "experiments.evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa"
)
DLR_RESULT_ROOT = trainer.DEFAULT_OUTPUT_ROOT
DEFAULT_RUN_DIRS = {
    "e_qfg_dlr": (
        DLR_RESULT_ROOT
        / "qfg_dlr_lane"
        / policy.DATASET
        / trainer.QFG_DLR_VARIANT
        / f"seed_42_{trainer.FORMAL_RUN_TAGS[trainer.QFG_DLR_VARIANT]}"
    ),
    "f_tss_qfg_dlr": (
        DLR_RESULT_ROOT
        / "tss_qfg_dlr_lane"
        / policy.DATASET
        / trainer.TSS_QFG_DLR_VARIANT
        / f"seed_42_{trainer.FORMAL_RUN_TAGS[trainer.TSS_QFG_DLR_VARIANT]}"
    ),
}
DLR_METHOD_SPECS = {
    "e_qfg_dlr": {
        "display_name": "E: QFG + DLR",
        "variant": trainer.QFG_DLR_VARIANT,
    },
    "f_tss_qfg_dlr": {
        "display_name": "F: TSS-ramp100 + QFG + DLR",
        "variant": trainer.TSS_QFG_DLR_VARIANT,
    },
}
DEFAULT_OUTPUT_DIR = DLR_RESULT_ROOT / "final_selection"
DEFAULT_JSON_OUTPUT = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_final_selection.json"
)
DEFAULT_MARKDOWN_OUTPUT = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_final_selection.md"
)
DEFAULT_CLOSURE_SOURCE_LOCK = policy.DEFAULT_LOCK_PATH
OUTCOME_FIELDS = (
    "matched_target_count",
    "pd",
    "fa",
    "miou",
    "matched_tiny_target_count",
    "tiny_pd",
    "unmatched_predicted_object_count",
    "false_objects_per_image",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_equal(label: str, left: Any, right: Any) -> None:
    _require(
        policy.canonical(left) == policy.canonical(right),
        f"{label} differs",
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = policy.regular_file(path, label)
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain one object")
    return payload


def _sweep_filename(checkpoint: str) -> str:
    return f"pd_fa_sweep_{Path(checkpoint).stem}.json"


def _load_evaluator() -> ModuleType:
    return importlib.import_module(DLR_EVALUATOR_MODULE)


def _validate_summary(
    run_dir: Path,
    *,
    method_id: str,
    variant: str,
) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json", f"{method_id} summary")
    candidate = trainer.candidate_contract(variant)
    expected = {
        "schema": trainer.COMPLETION_SUMMARY_SCHEMA,
        "status": "complete",
        "variant": variant,
        "candidate_variant": variant,
        "base_model_variant": candidate["base_model_variant"],
        "qfg_variant": trainer.v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "family_recipe": trainer.FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "dataset": policy.DATASET,
        "seed": policy.TRAINING_SEED,
        "split_seed": policy.SPLIT_SEED,
        "formal_contract": trainer.formal_contract(),
        "official_test_accessed": False,
    }
    for name, required in expected.items():
        _canonical_equal(f"{method_id} summary {name}", summary.get(name), required)
    identity = trainer.require_paired_run_identity(
        summary.get("run_identity"),
        label=f"{method_id} summary",
        expected_variant=variant,
    )
    _canonical_equal(
        f"{method_id} summary source locks",
        summary.get("source_locks"),
        identity["source_locks"],
    )
    _require(
        identity["source_locks"][trainer.SOURCE_LOCK_KEY]
        == policy.TRAINING_LOCK_SHA256,
        f"{method_id} training source-lock digest differs",
    )
    return summary


def _validate_interface(
    payload: Mapping[str, Any],
    *,
    method_id: str,
    variant: str,
    checkpoint: str,
) -> None:
    missing = [
        field
        for field in policy.DLR_SWEEP_REQUIRED_FIELDS
        if field not in payload
    ]
    _require(not missing, f"{method_id}:{checkpoint} lacks fields: {missing}")
    expected = {
        "schema": policy.DLR_EVALUATION_SCHEMA,
        "dataset": policy.DATASET,
        "seed": policy.TRAINING_SEED,
        "split_seed": policy.SPLIT_SEED,
        "variant": variant,
        "checkpoint_role": policy.CHECKPOINT_ROLES[checkpoint],
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "evaluated_checkpoint_count": 1,
        "official_test_accessed": False,
    }
    for name, required in expected.items():
        _canonical_equal(
            f"{method_id}:{checkpoint} interface {name}",
            payload.get(name),
            required,
        )


def _public_role(role: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(role[key])
        for key in (
            "checkpoint",
            "checkpoint_role",
            "role_name",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "checkpoint_path",
            "run_directory",
            "fixed_threshold_0_5",
            "fa_budget_points",
            "raw_point_count",
            "sweep_binding",
        )
    } | {
        "evaluator_validation": copy.deepcopy(role["evaluator_validation"]),
        "sweep_interface_schema": policy.DLR_SWEEP_INTERFACE_SCHEMA,
    }


def validate_dlr_method(
    method_id: str,
    run_dir: Path,
    *,
    evaluator_module: ModuleType | None = None,
) -> dict[str, Any]:
    _require(method_id in DLR_METHOD_SPECS, f"unknown DLR method: {method_id}")
    spec = DLR_METHOD_SPECS[method_id]
    variant = str(spec["variant"])
    directory = Path(run_dir).resolve()
    summary = _validate_summary(
        directory,
        method_id=method_id,
        variant=variant,
    )
    evaluator = _load_evaluator() if evaluator_module is None else evaluator_module
    roles: dict[str, Any] = {}
    run_identity: Mapping[str, Any] | None = None
    for checkpoint, checkpoint_role in policy.CHECKPOINT_ROLES.items():
        audit = evaluator.validate_run_artifacts(directory, checkpoint)
        _require(isinstance(audit, Mapping), "DLR evaluator audit is invalid")
        expected_epoch_key = (
            "best_pd_epoch"
            if checkpoint == "best.pth.tar"
            else "best_miou_epoch"
        )
        expected_epoch = summary.get(expected_epoch_key)
        expected_audit = {
            "variant": variant,
            "checkpoint_filename": checkpoint,
            "checkpoint_role": checkpoint_role,
            "checkpoint_epoch": expected_epoch,
            "run_directory": str(directory),
        }
        for name, required in expected_audit.items():
            _canonical_equal(
                f"{method_id}:{checkpoint} evaluator {name}",
                audit.get(name),
                required,
            )
        checkpoint_path = directory / checkpoint
        _require(
            Path(str(audit.get("checkpoint_path"))).resolve()
            == checkpoint_path,
            f"{method_id}:{checkpoint} audit checkpoint path differs",
        )
        checkpoint_sha = audit.get("checkpoint_sha256")
        _require(
            policy.is_sha256(checkpoint_sha)
            and policy.sha256_file(checkpoint_path) == checkpoint_sha,
            f"{method_id}:{checkpoint} checkpoint SHA differs",
        )
        sweep_path = directory / _sweep_filename(checkpoint)
        payload = _load_json(
            sweep_path,
            f"{method_id}:{checkpoint} sweep",
        )
        _validate_interface(
            payload,
            method_id=method_id,
            variant=variant,
            checkpoint=checkpoint,
        )
        evaluator.validate_output_identity(payload, artifact_audit=audit)
        _require(
            Path(str(payload.get("run_directory"))).resolve() == directory,
            f"{method_id}:{checkpoint} sweep run directory differs",
        )
        _require(
            Path(str(payload.get("checkpoint"))).resolve() == checkpoint_path,
            f"{method_id}:{checkpoint} sweep checkpoint path differs",
        )
        for name, required in {
            "checkpoint_epoch": expected_epoch,
            "checkpoint_sha256": checkpoint_sha,
        }.items():
            _canonical_equal(
                f"{method_id}:{checkpoint} sweep {name}",
                payload.get(name),
                required,
            )
        _canonical_equal(
            f"{method_id}:{checkpoint} sweep/audit checkpoint identity",
            payload.get("source_checkpoint_identity"),
            audit.get("checkpoint_identity"),
        )
        _canonical_equal(
            f"{method_id}:{checkpoint} sweep/audit run identity",
            payload.get("run_identity"),
            audit.get("run_identity"),
        )
        normalized = reference_selector.normalize_sweep_payload(
            payload,
            method_id=method_id,
            display_name=str(spec["display_name"]),
            expected_variant=variant,
            checkpoint=checkpoint,
            sweep_path=sweep_path,
            sweep_sha256=policy.sha256_file(sweep_path),
        )
        normalized["evaluator_validation"] = {
            "module": DLR_EVALUATOR_MODULE,
            "schema": payload["schema"],
            "checkpoint_identity": copy.deepcopy(
                audit.get("checkpoint_identity")
            ),
            "run_identity": copy.deepcopy(audit.get("run_identity")),
        }
        role_name = normalized["role_name"]
        roles[role_name] = _public_role(normalized)
        current_identity = audit.get("run_identity")
        if run_identity is None:
            run_identity = copy.deepcopy(current_identity)
        else:
            _canonical_equal(
                f"{method_id} run identity across checkpoint roles",
                current_identity,
                run_identity,
            )
    _require(
        tuple(roles) == policy.CHECKPOINT_ROLE_ORDER,
        f"{method_id} checkpoint role order differs",
    )
    return {
        "method_id": method_id,
        "display_name": str(spec["display_name"]),
        "variant": variant,
        "roles": roles,
        "training_recipe": trainer.FAMILY_RECIPE,
        "training_source_lock": {
            "path": str(policy.TRAINING_LOCK_PATH.resolve()),
            "sha256": policy.TRAINING_LOCK_SHA256,
        },
        "run_identity": copy.deepcopy(run_identity),
    }


def _reference_methods(
    reference_report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(
        reference_report.get("schema") == reference_selector.SCHEMA
        and reference_report.get("status") == "complete",
        "reference final-selection report is invalid",
    )
    methods = reference_report.get("methods")
    _require(isinstance(methods, Mapping), "reference methods are missing")
    _require(
        set(REFERENCE_METHOD_IDS).issubset(methods),
        "reference method matrix is incomplete",
    )
    return {
        method_id: copy.deepcopy(dict(methods[method_id]))
        for method_id in REFERENCE_METHOD_IDS
    }


def collect_methods(
    *,
    run_directories: Mapping[str, Path] | None = None,
    evaluator_module: ModuleType | None = None,
    reference_report: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reference = (
        reference_selector.build_formal_report()
        if reference_report is None
        else copy.deepcopy(dict(reference_report))
    )
    reference_methods = _reference_methods(reference)
    directories = {
        method_id: Path(path).resolve()
        for method_id, path in (
            DEFAULT_RUN_DIRS
            if run_directories is None
            else run_directories
        ).items()
    }
    _require(
        set(directories) == set(DLR_METHOD_SPECS),
        "DLR run-directory method matrix differs",
    )
    dlr_methods = {
        method_id: validate_dlr_method(
            method_id,
            directories[method_id],
            evaluator_module=evaluator_module,
        )
        for method_id in DLR_METHOD_SPECS
    }
    e_run_id = dlr_methods["e_qfg_dlr"]["run_identity"].get("run_id")
    f_run_id = dlr_methods["f_tss_qfg_dlr"]["run_identity"].get("run_id")
    _require(e_run_id != f_run_id, "E/F must be independent run identities")
    methods = {
        **reference_methods,
        "e_qfg_dlr": dlr_methods["e_qfg_dlr"],
        "f_tss_qfg_dlr": dlr_methods["f_tss_qfg_dlr"],
    }
    _require(tuple(methods) == policy.METHOD_ORDER, "combined method order differs")
    reference_binding = {
        "schema": reference.get("schema"),
        "decision": reference.get("decision"),
        "selected_method_id": reference.get("selection", {}).get(
            "selected_method_id"
        ),
        "authority_binding": copy.deepcopy(reference.get("authority_binding")),
        "posttraining_closure_source_lock": copy.deepcopy(
            reference.get("posttraining_closure_source_lock")
        ),
        "input_bindings": copy.deepcopy(reference.get("input_bindings")),
    }
    return methods, reference_binding


def _point_delta(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, float | int]:
    return {
        field: left[field] - right[field]
        for field in OUTCOME_FIELDS
    }


def compare_method_pair(
    methods: Mapping[str, Mapping[str, Any]],
    left_id: str,
    right_id: str,
) -> dict[str, Any]:
    _require(left_id in methods and right_id in methods, "pair method is missing")
    locations: dict[str, Any] = {}
    left_wins = 0
    right_wins = 0
    exact_ties = 0
    left_dominates = 0
    right_dominates = 0
    for role_name in policy.CHECKPOINT_ROLE_ORDER:
        for location in policy.LOCATION_ORDER:
            left = policy.point_for_location(methods[left_id], role_name, location)
            right = policy.point_for_location(
                methods[right_id],
                role_name,
                location,
            )
            left_key = policy.objective_key(left)
            right_key = policy.objective_key(right)
            if left_key < right_key:
                outcome = "left"
                left_wins += 1
            elif right_key < left_key:
                outcome = "right"
                right_wins += 1
            else:
                outcome = "tie"
                exact_ties += 1
            left_strict = policy.dominates(left, right)
            right_strict = policy.dominates(right, left)
            left_dominates += int(left_strict)
            right_dominates += int(right_strict)
            key = f"{role_name}:{location}"
            locations[key] = {
                "role_name": role_name,
                "location": location,
                "left_checkpoint": methods[left_id]["roles"][role_name][
                    "checkpoint"
                ],
                "right_checkpoint": methods[right_id]["roles"][role_name][
                    "checkpoint"
                ],
                "left_threshold": float(left["threshold"]),
                "right_threshold": float(right["threshold"]),
                "left_minus_right": _point_delta(left, right),
                "lexicographic_winner": outcome,
                "left_strictly_dominates": left_strict,
                "right_strictly_dominates": right_strict,
                "checkpoint_local": True,
            }
    return {
        "left_method_id": left_id,
        "right_method_id": right_id,
        "location_count": len(locations),
        "left_lexicographic_win_count": left_wins,
        "right_lexicographic_win_count": right_wins,
        "exact_tie_count": exact_ties,
        "left_strict_dominance_count": left_dominates,
        "right_strict_dominance_count": right_dominates,
        "locations": locations,
        "cross_checkpoint_metric_stitching": False,
    }


def _public_method(method: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "method_id": method["method_id"],
        "display_name": method.get("display_name", method["method_id"]),
        "variant": method["variant"],
        "roles": {
            role_name: copy.deepcopy(method["roles"][role_name])
            for role_name in policy.CHECKPOINT_ROLE_ORDER
        },
    }
    for key in ("training_recipe", "training_source_lock"):
        if key in method:
            result[key] = copy.deepcopy(method[key])
    return result


def _snapshot_bindings(
    methods: Mapping[str, Mapping[str, Any]],
    reference_binding: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    snapshot: dict[str, dict[str, str]] = {}
    reference_inputs = reference_binding.get("input_bindings")
    _require(
        isinstance(reference_inputs, Mapping),
        "reference input bindings are missing",
    )
    for label, binding in reference_inputs.items():
        _require(isinstance(binding, Mapping), f"reference binding {label} differs")
        snapshot[f"reference:{label}"] = {
            "path": str(binding["path"]),
            "sha256": str(binding["sha256"]),
        }
    reference_lock = reference_binding.get("posttraining_closure_source_lock")
    _require(isinstance(reference_lock, Mapping), "reference closure lock missing")
    snapshot["reference:closure_source_lock"] = {
        "path": str(reference_lock["path"]),
        "sha256": str(reference_lock["sha256"]),
    }
    snapshot["dlr:training_source_lock"] = {
        "path": str(policy.TRAINING_LOCK_PATH.resolve()),
        "sha256": policy.TRAINING_LOCK_SHA256,
    }
    for method_id in ("e_qfg_dlr", "f_tss_qfg_dlr"):
        method = methods[method_id]
        run_dir = Path(method["roles"]["pd_primary"]["run_directory"])
        for artifact in ("summary.json", "protocol.json"):
            path = run_dir / artifact
            snapshot[f"{method_id}:{artifact}"] = {
                "path": str(path),
                "sha256": policy.sha256_file(path),
            }
        for role_name in policy.CHECKPOINT_ROLE_ORDER:
            role = method["roles"][role_name]
            snapshot[f"{method_id}:{role_name}:checkpoint"] = {
                "path": role["checkpoint_path"],
                "sha256": role["checkpoint_sha256"],
            }
            snapshot[f"{method_id}:{role_name}:sweep"] = copy.deepcopy(
                role["sweep_binding"]
            )
    return snapshot


def verify_snapshot(snapshot: Mapping[str, Mapping[str, str]]) -> None:
    for label, binding in snapshot.items():
        _require(policy.is_sha256(binding.get("sha256")), f"{label} SHA invalid")
        _require(
            policy.sha256_file(Path(binding["path"])) == binding["sha256"],
            f"selection input changed: {label}",
        )


def build_report(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    reference_binding: Mapping[str, Any],
    input_bindings: Mapping[str, Mapping[str, str]],
    closure_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(tuple(methods) == policy.METHOD_ORDER, "final method matrix differs")
    _require(
        closure_binding.get("schema") == policy.LOCK_SCHEMA
        and closure_binding.get("verified_live") is True
        and closure_binding.get("training_source_lock_sha256")
        == policy.TRAINING_LOCK_SHA256
        and closure_binding.get("reference_closure_lock_sha256")
        == policy.REFERENCE_CLOSURE_LOCK_SHA256,
        "DLR closure source-lock identity differs",
    )
    _require(
        closure_binding.get("policy_summary_sha256")
        == policy.policy_summary_sha256(),
        "DLR closure policy binding differs",
    )
    verify_snapshot(input_bindings)
    selection = policy.select_method(methods)
    selected_method_id = selection["selected_method_id"]
    deployment = policy.select_deployment_operating_point(
        methods[selected_method_id]
    )
    pairwise = {
        f"{candidate}_vs_{reference}": compare_method_pair(
            methods,
            candidate,
            reference,
        )
        for candidate in ("e_qfg_dlr", "f_tss_qfg_dlr")
        for reference in REFERENCE_METHOD_IDS
    }
    pairwise["f_tss_qfg_dlr_vs_e_qfg_dlr"] = compare_method_pair(
        methods,
        "f_tss_qfg_dlr",
        "e_qfg_dlr",
    )
    selected_vs_baseline = compare_method_pair(
        methods,
        selected_method_id,
        "baseline",
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": policy.DATASET,
        "training_seed": policy.TRAINING_SEED,
        "split_seed": policy.SPLIT_SEED,
        "scope": "single_seed_internal_validation",
        "official_test_accessed": False,
        "metric_contract": policy.policy_summary(),
        "dlr_sweep_public_interface": policy.interface_summary(),
        "reference_binding": copy.deepcopy(dict(reference_binding)),
        "input_bindings": {
            key: copy.deepcopy(dict(value))
            for key, value in sorted(input_bindings.items())
        },
        "methods": {
            method_id: _public_method(methods[method_id])
            for method_id in policy.METHOD_ORDER
        },
        "pairwise_comparisons": pairwise,
        "selection": selection,
        "deployment_selection": deployment,
        "selected_method_id": selected_method_id,
        "selected_variant": methods[selected_method_id]["variant"],
        "selected_checkpoint": deployment["selected"]["checkpoint"],
        "selected_checkpoint_role": deployment["selected"]["checkpoint_role"],
        "selected_threshold": deployment["selected"]["threshold"],
        "selected_vs_baseline": selected_vs_baseline,
        "meaningful_overall_improvement_under_frozen_policy": selection[
            "selected_outperforms_baseline_under_frozen_policy"
        ],
        "dlr_recipe_selected": selected_method_id
        in {"e_qfg_dlr", "f_tss_qfg_dlr"},
        "tss_training_selected": selected_method_id
        in {"b_tss", "d_tss_qfg", "f_tss_qfg_dlr"},
        "tss_inference_enabled": False,
        "cross_checkpoint_metric_stitching": False,
        "cross_method_metric_stitching": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "claim_boundary": {
            "single_seed_only": True,
            "internal_validation_only": True,
            "official_test_claim": False,
            "cross_seed_stability_claim": False,
            "universal_dominance_claim": False,
        },
        "posttraining_closure_source_lock": copy.deepcopy(
            dict(closure_binding)
        ),
        "write_once": True,
        "idempotent_resume": True,
    }


def build_formal_report(
    *,
    run_directories: Mapping[str, Path] | None = None,
    evaluator_module: ModuleType | None = None,
    reference_report: Mapping[str, Any] | None = None,
    closure_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = (
        policy.load_closure_lock(verify_sources=True)[1]
        if closure_binding is None
        else copy.deepcopy(dict(closure_binding))
    )
    methods, reference_binding = collect_methods(
        run_directories=run_directories,
        evaluator_module=evaluator_module,
        reference_report=reference_report,
    )
    snapshot = _snapshot_bindings(methods, reference_binding)
    verify_snapshot(snapshot)
    return build_report(
        methods,
        reference_binding=reference_binding,
        input_bindings=snapshot,
        closure_binding=binding,
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    _require(report.get("schema") == SCHEMA, "report schema differs")
    selected = report["deployment_selection"]["selected"]
    lines = [
        "# Paired DLR+ramp100 formal800 最终比较与选择",
        "",
        f"- 最终方法：`{report['selected_method_id']}`",
        f"- 最终 variant：`{report['selected_variant']}`",
        (
            f"- 唯一 checkpoint：`{selected['checkpoint']}` "
            f"({selected['checkpoint_role']}, epoch={selected['checkpoint_epoch']})"
        ),
        (
            f"- 唯一阈值：`{selected['threshold']:.10g}`，"
            f"来源 `{selected['operating_point_source']}`"
        ),
        (
            "- 相对 baseline 的冻结策略整体改进："
            f"`{report['meaningful_overall_improvement_under_frozen_policy']}`"
        ),
        f"- DLR recipe 被选择：`{report['dlr_recipe_selected']}`",
        "- 不跨 checkpoint、role 或方法拼接指标/阈值。",
        "",
        "## 方法级综合排序",
        "",
        "| 排名 | 方法 | 12位置 rank-sum | 第一名次数 | Pareto次数 | 严格支配次数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    summaries = report["selection"]["method_summaries"]
    for rank, method_id in enumerate(
        report["selection"]["ranked_method_ids"],
        start=1,
    ):
        summary = summaries[method_id]
        lines.append(
            f"| {rank} | {method_id} | {summary['aligned_rank_sum']} | "
            f"{summary['aligned_first_place_count']} | "
            f"{summary['aligned_pareto_membership_count']} | "
            f"{summary['aligned_pairwise_strict_dominance_count']} |"
        )
    lines.extend(
        [
            "",
            "## fixed0.5 与五个 Fa budget",
            "",
            (
                "| 方法 | checkpoint role | 位置 | threshold | Pd | Fa | "
                "mIoU | tiny-Pd | false objects/image |"
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method_id in policy.METHOD_ORDER:
        method = report["methods"][method_id]
        for role_name in policy.CHECKPOINT_ROLE_ORDER:
            for location in policy.LOCATION_ORDER:
                point = policy.point_for_location(method, role_name, location)
                lines.append(
                    f"| {method_id} | {role_name} | {location} | "
                    f"{float(point['threshold']):.10g} | "
                    f"{float(point['pd']):.10g} | "
                    f"{float(point['fa']):.10g} | "
                    f"{float(point['miou']):.10g} | "
                    f"{float(point['tiny_pd']):.10g} | "
                    f"{float(point['false_objects_per_image']):.10g} |"
                )
    return "\n".join(lines) + "\n"


def _absolute_output(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    absolute = Path(os.path.abspath(value))
    return absolute.parent.resolve(strict=False) / absolute.name


def _atomic_create(path: Path, content: bytes) -> bool:
    output = _absolute_output(path)
    if output.exists() or output.is_symlink():
        _require(
            output.is_file()
            and not output.is_symlink()
            and output.read_bytes() == content,
            f"existing write-once output conflicts: {output}",
        )
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            _require(
                output.is_file()
                and not output.is_symlink()
                and output.read_bytes() == content,
                f"concurrent write-once output conflicts: {output}",
            )
            return False
        directory_descriptor = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def publish_report(
    report: Mapping[str, Any],
    *,
    json_output: Path = DEFAULT_JSON_OUTPUT,
    markdown_output: Path = DEFAULT_MARKDOWN_OUTPUT,
) -> dict[str, Any]:
    json_path = _absolute_output(json_output)
    markdown_path = _absolute_output(markdown_output)
    _require(json_path != markdown_path, "JSON/Markdown outputs must differ")
    json_bytes = policy.canonical_json_bytes(report)
    markdown_bytes = render_markdown(report).encode("utf-8")
    json_written = _atomic_create(json_path, json_bytes)
    markdown_written = _atomic_create(markdown_path, markdown_bytes)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": (
            "publish"
            if json_written or markdown_written
            else "verify"
        ),
        "json_output": str(json_path),
        "json_sha256": policy.sha256_file(json_path),
        "markdown_output": str(markdown_path),
        "markdown_sha256": policy.sha256_file(markdown_path),
        "writes_performed": json_written or markdown_written,
        "write_once": True,
        "idempotent_resume": True,
    }


def preflight_manifest(
    *,
    methods: Mapping[str, Mapping[str, Any]],
    reference_binding: Mapping[str, Any],
    closure_binding: Mapping[str, Any],
) -> dict[str, Any]:
    selection = policy.select_method(methods)
    deployment = policy.select_deployment_operating_point(
        methods[selection["selected_method_id"]]
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "ready",
        "action": "preflight",
        "method_count": len(methods),
        "expected_method_count": len(policy.METHOD_ORDER),
        "checkpoint_count": len(methods) * 2,
        "aligned_location_count": (
            len(methods)
            * len(policy.CHECKPOINT_ROLE_ORDER)
            * len(policy.LOCATION_ORDER)
        ),
        "selected_method_id": selection["selected_method_id"],
        "selected_checkpoint": deployment["selected"]["checkpoint"],
        "selected_threshold": deployment["selected"]["threshold"],
        "reference_binding": copy.deepcopy(dict(reference_binding)),
        "closure_binding": copy.deepcopy(dict(closure_binding)),
        "writes_performed": False,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--e-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS["e_qfg_dlr"],
    )
    parser.add_argument(
        "--f-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS["f_tss_qfg_dlr"],
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    parser.add_argument(
        "--closure-source-lock",
        type=Path,
        default=DEFAULT_CLOSURE_SOURCE_LOCK,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    _, closure_binding = policy.load_closure_lock(
        args.closure_source_lock,
        verify_sources=True,
    )
    methods, reference_binding = collect_methods(
        run_directories={
            "e_qfg_dlr": args.e_run_dir,
            "f_tss_qfg_dlr": args.f_run_dir,
        }
    )
    if args.preflight:
        result = preflight_manifest(
            methods=methods,
            reference_binding=reference_binding,
            closure_binding=closure_binding,
        )
    else:
        snapshot = _snapshot_bindings(methods, reference_binding)
        report = build_report(
            methods,
            reference_binding=reference_binding,
            input_bindings=snapshot,
            closure_binding=closure_binding,
        )
        result = publish_report(
            report,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "DEFAULT_CLOSURE_SOURCE_LOCK",
    "DEFAULT_JSON_OUTPUT",
    "DEFAULT_MARKDOWN_OUTPUT",
    "DEFAULT_RUN_DIRS",
    "DLR_METHOD_SPECS",
    "OUTCOME_FIELDS",
    "PREFLIGHT_SCHEMA",
    "REFERENCE_METHOD_IDS",
    "SCHEMA",
    "build_formal_report",
    "build_report",
    "collect_methods",
    "compare_method_pair",
    "main",
    "preflight_manifest",
    "publish_report",
    "render_markdown",
    "validate_dlr_method",
    "verify_snapshot",
]


if __name__ == "__main__":
    main()
