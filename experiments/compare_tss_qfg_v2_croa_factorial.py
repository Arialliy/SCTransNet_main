#!/usr/bin/env python3
"""Strict seed-42 A/B/C/D factorial summary for TSS and QFG-V2-CROA.

This module never evaluates images and never selects a checkpoint.  It
consumes the eight checkpoint-local sweeps already produced for each arm's
own ``best.pth.tar`` and ``best_miou.pth.tar``.  The existing Survival and
QFG evaluators remain the authorities for run, checkpoint, and sweep
validation.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
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
    tpd_ner_v4_qfg_v2_croa_posttraining_policy as closure_policy,
)

SCHEMA = "sctransnet_tss_qfg_v2_croa_factorial_seed42_v2"
PREFLIGHT_SCHEMA = (
    "sctransnet_tss_qfg_v2_croa_factorial_preflight_v1"
)
ACTION_SCHEMA = "sctransnet_tss_qfg_v2_croa_factorial_action_v1"
DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
VALIDATION_COUNT = 133
TARGET_COUNT = 189
TINY_TARGET_COUNT = 39

CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
ROLE_NAMES = {
    "best.pth.tar": "pd_primary",
    "best_miou.pth.tar": "miou_secondary",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
OUTCOME_FIELDS = (
    "matched_target_count",
    "pd",
    "fa",
    "miou",
    "tiny_pd",
    "unmatched_predicted_object_count",
    "false_objects_per_image",
)
POINT_FIELDS = (
    "threshold",
    *OUTCOME_FIELDS,
    "target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "valid_pixel_count",
)
CONTRAST_ORDER = (
    "B_minus_A",
    "D_minus_C",
    "C_minus_A",
    "D_minus_B",
    "marginal_tss",
    "marginal_qfg",
    "interaction_tss_x_qfg",
)

SURVIVAL_ROOT = (
    REPO_ROOT / "experiments/results/tpd_ner_v4_survival_exact_v1"
)
QFG_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized"
)
DEFAULT_RUN_DIRS = {
    "A": (
        SURVIVAL_ROOT
        / DATASET
        / "tss_control"
        / "seed_42_formal800_control"
    ),
    "B": (
        SURVIVAL_ROOT
        / DATASET
        / "tss_on"
        / "seed_42_formal800_tss"
    ),
    "C": (
        QFG_ROOT
        / DATASET
        / "qfg_only"
        / "seed_42_formal800_qfg_only"
    ),
    "D": (
        QFG_ROOT
        / DATASET
        / "tss_qfg"
        / "seed_42_formal800_tss_qfg"
    ),
}
DEFAULT_OUTPUT_DIR = QFG_ROOT / DATASET / "comparison_factorial_v1"
DEFAULT_JSON_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "tss_qfg_v2_croa_factorial_seed42.json"
)
DEFAULT_MARKDOWN_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "tss_qfg_v2_croa_factorial_seed42.md"
)


@dataclass(frozen=True)
class ArmSpec:
    label: str
    variant: str
    evaluator_family: str
    tss_enabled: bool
    qfg_enabled: bool


ARM_SPECS = {
    "A": ArmSpec("A", "tss_control", "survival", False, False),
    "B": ArmSpec("B", "tss_on", "survival", True, False),
    "C": ArmSpec("C", "qfg_only", "qfg", False, True),
    "D": ArmSpec("D", "tss_qfg", "qfg", True, True),
}


@dataclass(frozen=True)
class SweepRecord:
    arm: str
    variant: str
    evaluator_family: str
    run_directory: Path
    checkpoint_filename: str
    checkpoint_role: str
    checkpoint_epoch: int
    checkpoint_sha256: str
    sweep_path: Path
    sweep_sha256: str
    validation_split_sha256: str
    run_identity: Mapping[str, Any]
    checkpoint_identity: Mapping[str, Any]
    parent_binding: Mapping[str, Any]
    fixed: Mapping[str, Any]
    budgets: Mapping[str, Mapping[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {value}")
    return value


def _sha256_file(path: Path) -> str:
    value = _regular_file(path, "SHA-256 input")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = _regular_file(path, label)
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    )


def _canonical_equal(location: str, observed: Any, expected: Any) -> None:
    if _canonical(observed) != _canonical(expected):
        raise ValueError(f"{location} differs after JSON normalization")


def _finite_number(value: Any, label: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _normalize_point(
    value: Any,
    *,
    label: str,
    fixed: bool = False,
    budget: float | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = [name for name in POINT_FIELDS if name not in value]
    if missing:
        raise ValueError(f"{label} lacks fields: {missing}")
    point = {
        name: copy.deepcopy(value[name])
        for name in POINT_FIELDS
    }
    for name in (
        "threshold",
        "pd",
        "fa",
        "miou",
        "tiny_pd",
        "false_objects_per_image",
    ):
        _finite_number(point[name], f"{label}.{name}")
    for name in (
        "matched_target_count",
        "unmatched_predicted_object_count",
        "target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "valid_pixel_count",
    ):
        _integer(point[name], f"{label}.{name}")
    _require_equal(f"{label}.target_count", point["target_count"], TARGET_COUNT)
    _require_equal(
        f"{label}.tiny_target_count",
        point["tiny_target_count"],
        TINY_TARGET_COUNT,
    )
    if fixed:
        _require_equal(f"{label}.threshold", float(point["threshold"]), 0.5)
    if budget is not None and float(point["fa"]) > budget + 1e-18:
        raise ValueError(f"{label}.fa exceeds budget {budget}")
    _require(
        0 <= point["matched_target_count"] <= TARGET_COUNT,
        f"{label}.matched_target_count is invalid",
    )
    _require(
        0 <= point["matched_tiny_target_count"] <= TINY_TARGET_COUNT,
        f"{label}.matched_tiny_target_count is invalid",
    )
    _require(
        0
        <= point["unmatched_predicted_object_count"]
        <= point["predicted_object_count"],
        f"{label}.unmatched_predicted_object_count is invalid",
    )
    _require(
        math.isclose(
            float(point["pd"]),
            point["matched_target_count"] / TARGET_COUNT,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{label}.pd differs from object counts",
    )
    _require(
        math.isclose(
            float(point["tiny_pd"]),
            point["matched_tiny_target_count"] / TINY_TARGET_COUNT,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{label}.tiny_pd differs from tiny-object counts",
    )
    _require(
        math.isclose(
            float(point["false_objects_per_image"]),
            point["unmatched_predicted_object_count"] / VALIDATION_COUNT,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{label}.false_objects_per_image differs from counts",
    )
    return point


def _load_evaluator_modules() -> dict[str, ModuleType]:
    return {
        "survival": importlib.import_module(
            "experiments."
            "evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa"
        ),
        "qfg": importlib.import_module(
            "experiments.evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa"
        ),
    }


def _validate_completion_summary(
    run_dir: Path,
    *,
    spec: ArmSpec,
) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json", f"arm {spec.label} summary")
    for name, expected in {
        "status": "complete",
        "variant": spec.variant,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "official_test_accessed": False,
    }.items():
        _require_equal(
            f"arm {spec.label} summary {name}",
            summary.get(name),
            expected,
        )
    return summary


def _sweep_filename(checkpoint: str) -> str:
    return f"pd_fa_sweep_{Path(checkpoint).stem}.json"


def _record_from_validated_artifacts(
    *,
    spec: ArmSpec,
    run_dir: Path,
    checkpoint: str,
    summary: Mapping[str, Any],
    evaluator: ModuleType,
) -> SweepRecord:
    audit = evaluator.validate_run_artifacts(run_dir, checkpoint)
    if not isinstance(audit, Mapping):
        raise ValueError(f"arm {spec.label} evaluator audit is not an object")
    role = CHECKPOINT_ROLES[checkpoint]
    expected_epoch_key = (
        "best_pd_epoch"
        if checkpoint == "best.pth.tar"
        else "best_miou_epoch"
    )
    expected_epoch = summary.get(expected_epoch_key)
    for name, observed, expected in (
        ("variant", audit.get("variant"), spec.variant),
        ("checkpoint filename", audit.get("checkpoint_filename"), checkpoint),
        ("checkpoint role", audit.get("checkpoint_role"), role),
        ("checkpoint epoch", audit.get("checkpoint_epoch"), expected_epoch),
    ):
        _require_equal(f"arm {spec.label} {name}", observed, expected)
    _require_equal(
        f"arm {spec.label} audit run directory",
        Path(str(audit.get("run_directory"))).resolve(),
        run_dir,
    )
    checkpoint_path = run_dir / checkpoint
    _require_equal(
        f"arm {spec.label} audit checkpoint path",
        Path(str(audit.get("checkpoint_path"))).resolve(),
        checkpoint_path,
    )
    checkpoint_sha256 = audit.get("checkpoint_sha256")
    if not _is_sha256(checkpoint_sha256):
        raise ValueError(f"arm {spec.label} checkpoint SHA-256 is invalid")

    sweep_path = run_dir / _sweep_filename(checkpoint)
    payload = _load_json(
        sweep_path,
        f"arm {spec.label} {checkpoint} sweep",
    )
    evaluator.validate_output_identity(payload, artifact_audit=audit)
    for name, expected in {
        "variant": spec.variant,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint_role": role,
        "checkpoint_epoch": expected_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "evaluated_checkpoint_count": 1,
        "official_test_accessed": False,
    }.items():
        _require_equal(
            f"arm {spec.label} sweep {name}",
            payload.get(name),
            expected,
        )
    _require_equal(
        f"arm {spec.label} sweep run directory",
        Path(str(payload.get("run_directory"))).resolve(),
        run_dir,
    )
    _require_equal(
        f"arm {spec.label} sweep checkpoint",
        Path(str(payload.get("checkpoint"))).resolve(),
        checkpoint_path,
    )

    raw_budgets = payload.get("best_points_under_fa_budget")
    if not isinstance(raw_budgets, Mapping):
        raise ValueError(f"arm {spec.label} sweep budgets are missing")
    _require_equal(
        f"arm {spec.label} sweep budget keys",
        tuple(raw_budgets),
        BUDGET_KEYS,
    )
    budgets = {
        key: _normalize_point(
            raw_budgets[key],
            label=f"arm {spec.label} {checkpoint} Fa<={key}",
            budget=budget,
        )
        for key, budget in zip(BUDGET_KEYS, FA_BUDGETS)
    }
    fixed_point = _normalize_point(
        payload.get("fixed_threshold_0_5"),
        label=f"arm {spec.label} {checkpoint} fixed0.5",
        fixed=True,
    )
    validation_split_sha256 = payload.get("validation_split_sha256")
    if not _is_sha256(validation_split_sha256):
        raise ValueError(
            f"arm {spec.label} validation-split SHA-256 is invalid"
        )
    run_identity = audit.get("run_identity")
    checkpoint_identity = audit.get("checkpoint_identity")
    if not isinstance(run_identity, Mapping):
        raise ValueError(f"arm {spec.label} run identity is missing")
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError(f"arm {spec.label} checkpoint identity is missing")
    source_identity = payload.get("source_checkpoint_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError(f"arm {spec.label} source checkpoint identity is missing")
    _canonical_equal(
        f"arm {spec.label} source/audit checkpoint identity",
        source_identity,
        checkpoint_identity,
    )
    parent_binding = {
        name: source_identity.get(name)
        for name in (
            "parent_checkpoint_sha256",
            "parent_checkpoint_state_dict_sha256",
            "parent_checkpoint_role",
            "parent_checkpoint_epoch",
        )
    }
    if not all(
        parent_binding[name] is not None for name in parent_binding
    ):
        raise ValueError(f"arm {spec.label} parent binding is incomplete")
    return SweepRecord(
        arm=spec.label,
        variant=spec.variant,
        evaluator_family=spec.evaluator_family,
        run_directory=run_dir,
        checkpoint_filename=checkpoint,
        checkpoint_role=role,
        checkpoint_epoch=int(expected_epoch),
        checkpoint_sha256=str(checkpoint_sha256),
        sweep_path=sweep_path,
        sweep_sha256=_sha256_file(sweep_path),
        validation_split_sha256=str(validation_split_sha256),
        run_identity=copy.deepcopy(dict(run_identity)),
        checkpoint_identity=copy.deepcopy(dict(checkpoint_identity)),
        parent_binding=parent_binding,
        fixed=fixed_point,
        budgets=budgets,
    )


def _validate_factorial_bindings(
    records: Mapping[tuple[str, str], SweepRecord],
) -> None:
    expected_keys = {
        (arm, checkpoint)
        for arm in ARM_SPECS
        for checkpoint in CHECKPOINT_ROLES
    }
    _require_equal("factorial record matrix", set(records), expected_keys)
    split_digests = {
        record.validation_split_sha256 for record in records.values()
    }
    _require_equal("factorial validation split count", len(split_digests), 1)
    parent_bindings = {
        json.dumps(
            _canonical(record.parent_binding),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records.values()
    }
    _require_equal("factorial parent binding count", len(parent_bindings), 1)
    run_ids: dict[str, str] = {}
    for arm in ARM_SPECS:
        arm_records = [
            records[(arm, checkpoint)] for checkpoint in CHECKPOINT_ROLES
        ]
        first_identity = arm_records[0].run_identity
        for record in arm_records[1:]:
            _canonical_equal(
                f"arm {arm} run identity across roles",
                record.run_identity,
                first_identity,
            )
        run_id = first_identity.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"arm {arm} run id is missing")
        run_ids[arm] = run_id
    _require_equal(
        "factorial independent run-id count",
        len(set(run_ids.values())),
        len(ARM_SPECS),
    )


def collect_validated_sweeps(
    run_directories: Mapping[str, Path] | None = None,
    *,
    evaluator_modules: Mapping[str, ModuleType] | None = None,
) -> dict[tuple[str, str], SweepRecord]:
    directories = {
        arm: Path(path).resolve()
        for arm, path in (
            DEFAULT_RUN_DIRS
            if run_directories is None
            else run_directories
        ).items()
    }
    _require_equal("factorial run-directory arms", set(directories), set(ARM_SPECS))
    evaluators = (
        _load_evaluator_modules()
        if evaluator_modules is None
        else dict(evaluator_modules)
    )
    _require_equal(
        "factorial evaluator families",
        set(evaluators),
        {"survival", "qfg"},
    )
    summaries = {
        arm: _validate_completion_summary(
            directories[arm],
            spec=ARM_SPECS[arm],
        )
        for arm in ARM_SPECS
    }
    records: dict[tuple[str, str], SweepRecord] = {}
    for arm, spec in ARM_SPECS.items():
        evaluator = evaluators[spec.evaluator_family]
        for checkpoint in CHECKPOINT_ROLES:
            records[(arm, checkpoint)] = _record_from_validated_artifacts(
                spec=spec,
                run_dir=directories[arm],
                checkpoint=checkpoint,
                summary=summaries[arm],
                evaluator=evaluator,
            )
    _validate_factorial_bindings(records)
    return records


def _subtract(left: int | float, right: int | float) -> int | float:
    result = left - right
    if result == 0:
        return 0 if isinstance(left, int) and isinstance(right, int) else 0.0
    return result


def factorial_effects(
    arm_points: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int | float]]:
    _require_equal("factorial point arms", set(arm_points), set(ARM_SPECS))
    effects: dict[str, dict[str, int | float]] = {
        name: {} for name in CONTRAST_ORDER
    }
    for field in OUTCOME_FIELDS:
        values = {
            arm: _finite_number(
                arm_points[arm].get(field),
                f"arm {arm}.{field}",
            )
            for arm in ARM_SPECS
        }
        effects["B_minus_A"][field] = _subtract(values["B"], values["A"])
        effects["D_minus_C"][field] = _subtract(values["D"], values["C"])
        effects["C_minus_A"][field] = _subtract(values["C"], values["A"])
        effects["D_minus_B"][field] = _subtract(values["D"], values["B"])
        effects["marginal_tss"][field] = (
            effects["B_minus_A"][field]
            + effects["D_minus_C"][field]
        ) / 2
        effects["marginal_qfg"][field] = (
            effects["C_minus_A"][field]
            + effects["D_minus_B"][field]
        ) / 2
        interaction = (
            effects["D_minus_C"][field]
            - effects["B_minus_A"][field]
        )
        equivalent = (
            effects["D_minus_B"][field]
            - effects["C_minus_A"][field]
        )
        if not math.isclose(
            float(interaction),
            float(equivalent),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"factorial interaction identity failed: {field}")
        effects["interaction_tss_x_qfg"][field] = interaction
    return effects


def _operating_point(
    arm_points: Mapping[str, Mapping[str, Any]],
    *,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = {
        arm: {
            name: copy.deepcopy(point[name])
            for name in POINT_FIELDS
        }
        for arm, point in arm_points.items()
    }
    return {
        "alignment": copy.deepcopy(dict(alignment)),
        "arm_points": normalized,
        "effects": factorial_effects(normalized),
    }


def build_factorial_report(
    records: Mapping[tuple[str, str], SweepRecord],
    *,
    posttraining_closure_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_factorial_bindings(records)
    closure_binding = (
        closure_policy.load_closure_lock(verify_sources=True)[1]
        if posttraining_closure_binding is None
        else copy.deepcopy(dict(posttraining_closure_binding))
    )
    _require(
        closure_policy.is_sha256(closure_binding.get("sha256")),
        "post-training closure source-lock SHA is invalid",
    )
    _require(
        closure_binding.get("schema") == closure_policy.LOCK_SCHEMA
        and closure_binding.get("source_count")
        == len(closure_policy.POSTTRAINING_SOURCE_PATHS)
        and closure_binding.get("training_source_lock_sha256")
        == closure_policy.TRAINING_LOCK_SHA256
        and closure_binding.get("verified_live") is True,
        "post-training closure source-lock identity differs",
    )
    _require_equal(
        "post-training policy summary SHA",
        closure_binding.get("policy_summary_sha256"),
        closure_policy.policy_summary_sha256(),
    )
    role_reports: dict[str, Any] = {}
    for checkpoint, role in CHECKPOINT_ROLES.items():
        fixed_points = {
            arm: records[(arm, checkpoint)].fixed for arm in ARM_SPECS
        }
        budget_reports = {
            key: _operating_point(
                {
                    arm: records[(arm, checkpoint)].budgets[key]
                    for arm in ARM_SPECS
                },
                alignment={
                    "kind": "same_fa_budget",
                    "fa_budget": budget,
                    "budget_key": key,
                    "checkpoint_local_thresholds": True,
                },
            )
            for key, budget in zip(BUDGET_KEYS, FA_BUDGETS)
        }
        role_reports[ROLE_NAMES[checkpoint]] = {
            "checkpoint_filename": checkpoint,
            "checkpoint_role": role,
            "cross_role_pooling": False,
            "fixed_threshold_0_5": _operating_point(
                fixed_points,
                alignment={
                    "kind": "same_fixed_threshold",
                    "threshold": 0.5,
                },
            ),
            "fa_budget_points": budget_reports,
        }

    bindings = {
        f"{arm}:{checkpoint}": {
            "arm": arm,
            "variant": record.variant,
            "evaluator_family": record.evaluator_family,
            "run_directory": str(record.run_directory),
            "checkpoint_filename": record.checkpoint_filename,
            "checkpoint_role": record.checkpoint_role,
            "checkpoint_epoch": record.checkpoint_epoch,
            "checkpoint_sha256": record.checkpoint_sha256,
            "sweep_path": str(record.sweep_path),
            "sweep_sha256": record.sweep_sha256,
            "validation_split_sha256": (
                record.validation_split_sha256
            ),
        }
        for (arm, checkpoint), record in records.items()
    }
    parent_binding = copy.deepcopy(
        dict(next(iter(records.values())).parent_binding)
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "official_test_accessed": False,
        "scope": (
            "single_seed_internal_validation_descriptive_factorial_differences"
        ),
        "seed42_descriptive_only": True,
        "stability_claim_supported": False,
        "causal_claim_supported": False,
        "arm_contract": {
            arm: {
                "variant": spec.variant,
                "tss_enabled": spec.tss_enabled,
                "qfg_enabled": spec.qfg_enabled,
            }
            for arm, spec in ARM_SPECS.items()
        },
        "checkpoint_contract": {
            "roles": dict(CHECKPOINT_ROLES),
            "each_arm_uses_own_selected_checkpoint": True,
            "cross_role_pooling": False,
            "cross_checkpoint_point_pooling": False,
            "checkpoint_selection_from_sweep": False,
        },
        "effect_contract": {
            "outcome_fields": list(OUTCOME_FIELDS),
            "B_minus_A": "TSS effect with QFG off",
            "D_minus_C": "TSS effect with QFG on",
            "C_minus_A": "QFG effect with TSS off",
            "D_minus_B": "QFG effect with TSS on",
            "marginal_tss": "((B-A)+(D-C))/2",
            "marginal_qfg": "((C-A)+(D-B))/2",
            "interaction_tss_x_qfg": "(D-C)-(B-A)",
            "delta_sign": "left_minus_right_raw_metric_units",
            "automatic_superiority_decision": False,
        },
        "common_parent_binding": parent_binding,
        "posttraining_closure_source_lock": closure_binding,
        "artifact_bindings": bindings,
        "role_reports": role_reports,
        "aggregator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
    }


def preflight_manifest(
    records: Mapping[tuple[str, str], SweepRecord],
    *,
    posttraining_closure_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_factorial_bindings(records)
    closure_binding = (
        closure_policy.load_closure_lock(verify_sources=True)[1]
        if posttraining_closure_binding is None
        else copy.deepcopy(dict(posttraining_closure_binding))
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "ready",
        "action": "preflight",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "validated_arm_count": len(ARM_SPECS),
        "validated_checkpoint_count": len(records),
        "expected_checkpoint_count": 8,
        "checkpoint_roles": dict(CHECKPOINT_ROLES),
        "each_arm_uses_own_selected_checkpoint": True,
        "cross_role_pooling": False,
        "cross_checkpoint_point_pooling": False,
        "would_write": False,
        "posttraining_closure_source_lock": closure_binding,
        "artifacts": {
            f"{arm}:{checkpoint}": {
                "variant": record.variant,
                "run_directory": str(record.run_directory),
                "checkpoint_epoch": record.checkpoint_epoch,
                "checkpoint_sha256": record.checkpoint_sha256,
                "sweep_path": str(record.sweep_path),
                "sweep_sha256": record.sweep_sha256,
            }
            for (arm, checkpoint), record in records.items()
        },
    }


def _format_number(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    _require_equal("Markdown report schema", report.get("schema"), SCHEMA)
    lines = [
        "# TSS × QFG-V2-CROA A/B/C/D factorial summary",
        "",
        "Seed 42、NUDT-SIRST 530/133 内部划分的描述性差值；不支持跨随机性、"
        "因果或全面优越性主张。",
        "",
        (
            "Post-training closure lock：`"
            f"{report['posttraining_closure_source_lock']['sha256']}`。"
        ),
        "",
        "A=TSS off/QFG off，B=TSS on/QFG off，"
        "C=TSS off/QFG on，D=TSS on/QFG on。",
        "",
        "两个 checkpoint role 独立报告，不跨 role 或 checkpoint 合并阈值点。",
        "",
        "## Checkpoint bindings",
        "",
        "| Arm | Variant | Role | Epoch | Checkpoint SHA-256 |",
        "|---|---|---|---:|---|",
    ]
    bindings = report["artifact_bindings"]
    for checkpoint in CHECKPOINT_ROLES:
        for arm in ARM_SPECS:
            binding = bindings[f"{arm}:{checkpoint}"]
            lines.append(
                f"| {arm} | {binding['variant']} | "
                f"{binding['checkpoint_role']} | "
                f"{binding['checkpoint_epoch']} | "
                f"`{binding['checkpoint_sha256']}` |"
            )

    for role_name, role_report in report["role_reports"].items():
        lines.extend(
            [
                "",
                f"## {role_name}",
                "",
            ]
        )
        operating_points = [
            ("fixed_threshold_0_5", role_report["fixed_threshold_0_5"]),
            *[
                (f"Fa <= {key}", value)
                for key, value in role_report[
                    "fa_budget_points"
                ].items()
            ],
        ]
        for title, point in operating_points:
            arm_points = point["arm_points"]
            effects = point["effects"]
            thresholds = ", ".join(
                f"{arm}={_format_number(arm_points[arm]['threshold'])}"
                for arm in ARM_SPECS
            )
            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"Checkpoint-local thresholds: {thresholds}",
                    "",
                    "| Metric | A | B | C | D | B-A | D-C | C-A | "
                    "D-B | marginal TSS | marginal QFG | interaction |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
                    "---:|---:|---:|",
                ]
            )
            for field in OUTCOME_FIELDS:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            field,
                            *[
                                _format_number(arm_points[arm][field])
                                for arm in ARM_SPECS
                            ],
                            *[
                                _format_number(effects[name][field])
                                for name in CONTRAST_ORDER
                            ],
                        ]
                    )
                    + " |"
                )
    return "\n".join(lines) + "\n"


def _absolute_output(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    absolute = Path(os.path.abspath(value))
    return absolute.parent.resolve(strict=False) / absolute.name


def _require_new_outputs(
    json_output: Path,
    markdown_output: Path,
    *,
    run_directories: Mapping[str, Path],
) -> tuple[Path, Path]:
    json_path = _absolute_output(json_output)
    markdown_path = _absolute_output(markdown_output)
    _require(json_path != markdown_path, "JSON and Markdown outputs must differ")
    for output in (json_path, markdown_path):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace output: {output}")
        for arm, run_dir in run_directories.items():
            resolved_run = Path(run_dir).resolve()
            if output == resolved_run or resolved_run in output.parents:
                raise ValueError(
                    f"output must not be inside arm {arm} run directory"
                )
    return json_path, markdown_path


def _stage_new_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise NotADirectoryError(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".factorial.tmp",
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def publish_report_once(
    report: Mapping[str, Any],
    *,
    json_output: Path,
    markdown_output: Path,
    run_directories: Mapping[str, Path],
) -> dict[str, Any]:
    json_path, markdown_path = _require_new_outputs(
        json_output,
        markdown_output,
        run_directories=run_directories,
    )
    json_content = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    markdown_content = render_markdown(report).encode("utf-8")
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for output, content in (
            (json_path, json_content),
            (markdown_path, markdown_content),
        ):
            staged.append((output, _stage_new_file(output, content)))
        for output, temporary in staged:
            os.link(temporary, output, follow_symlinks=False)
            published.append(output)
    except BaseException:
        for output in published:
            output.unlink(missing_ok=True)
        raise
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "aggregate",
        "json_output": str(json_path),
        "json_sha256": _sha256_file(json_path),
        "markdown_output": str(markdown_path),
        "markdown_sha256": _sha256_file(markdown_path),
        "write_once": True,
        "overwrite_forbidden": True,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--a-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS["A"],
    )
    parser.add_argument(
        "--b-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS["B"],
    )
    parser.add_argument(
        "--c-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS["C"],
    )
    parser.add_argument(
        "--d-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS["D"],
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _, closure_binding = closure_policy.load_closure_lock(
        verify_sources=True
    )
    run_directories = {
        "A": args.a_run_dir,
        "B": args.b_run_dir,
        "C": args.c_run_dir,
        "D": args.d_run_dir,
    }
    if args.aggregate:
        _require_new_outputs(
            args.json_output,
            args.markdown_output,
            run_directories=run_directories,
        )
    records = collect_validated_sweeps(run_directories)
    if args.preflight:
        print(
            json.dumps(
                preflight_manifest(
                    records,
                    posttraining_closure_binding=closure_binding,
                ),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        return
    report = build_factorial_report(
        records,
        posttraining_closure_binding=closure_binding,
    )
    action = publish_report_once(
        report,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        run_directories=run_directories,
    )
    print(
        json.dumps(
            action,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "ARM_SPECS",
    "BUDGET_KEYS",
    "CHECKPOINT_ROLES",
    "CONTRAST_ORDER",
    "DEFAULT_JSON_OUTPUT",
    "DEFAULT_MARKDOWN_OUTPUT",
    "DEFAULT_RUN_DIRS",
    "FA_BUDGETS",
    "OUTCOME_FIELDS",
    "PREFLIGHT_SCHEMA",
    "SCHEMA",
    "SweepRecord",
    "build_factorial_report",
    "collect_validated_sweeps",
    "factorial_effects",
    "main",
    "parse_args",
    "preflight_manifest",
    "publish_report_once",
    "render_markdown",
]


if __name__ == "__main__":
    main()
