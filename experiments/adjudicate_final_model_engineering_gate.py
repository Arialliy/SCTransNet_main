#!/usr/bin/env python3
"""Adjudicate the fixed-parent B/D engineering evidence matrix.

This is a read-only, post-execution gate.  It accepts only the live-validated
four-run summary and the eight checkpoint-local sweep artifacts selected by
the frozen engineering contracts.  Missing evidence yields ``pending``.

Gate S-E is an engineering-completeness gate.  The two engineering trajectory
seeds and their aggregate checkpoint metrics are not a substitute for the
paired image-level simultaneous confidence intervals required by Gate M-train,
nor do they establish paper-level or full-pipeline stability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    evaluate_final_model_engineering_replication_pd_fa as evaluation_core,
)
from experiments import final_model_replication_exact_core as core  # noqa: E402
from experiments import final_model_replication_seed_contract as seeds  # noqa: E402
from experiments import freeze_final_model_certification_source_lock as source_lock_core  # noqa: E402
from experiments import prepare_final_model_engineering_replication as prepare  # noqa: E402
from experiments import summarize_final_model_engineering_replication as summary_core  # noqa: E402
from experiments import watch_final_model_engineering_replication as watcher  # noqa: E402


SCHEMA = "sctransnet_final_model_engineering_gate_adjudication_v1"
ACTION_SCHEMA = "sctransnet_final_model_engineering_gate_action_v1"
DEFAULT_SUMMARY_FILENAME = "engineering_replication_summary_v1.json"
DEFAULT_OUTPUT = (
    watcher.DEFAULT_OUTPUT_ROOT / "engineering_gate_adjudication_v1.json"
)
FINAL_DESIGN_PATH = (
    REPO_ROOT / "SCTransNet_最终模型稳定性认证与论文级闭环方案.md"
)
PROTOCOL_PATH = (
    REPO_ROOT / "experiments/FINAL_MODEL_CERTIFICATION_PROTOCOL_V1.md"
)
PRIMARY_CHECKPOINT = "best_miou.pth.tar"
PRIMARY_SELECTION_ROLE = "primary_best_miou"
KEY_SECONDARY_FA_BUDGET = 5e-6
FIXED_METRIC_FIELDS = (
    "pd",
    "fa",
    "miou",
    "tiny_pd",
    "false_objects_per_image",
    "unmatched_predicted_object_count",
    "matched_target_count",
    "target_count",
    "matched_tiny_target_count",
    "tiny_target_count",
    "valid_pixel_count",
)
_LOG_FIXED_6 = r"[+-]?\d+\.\d{6}"
_LOG_FIXED_8 = r"[+-]?\d+\.\d{8}"
_EPOCH_LOG_PATTERN = re.compile(
    rf"^EPOCH (?P<epoch>\d{{3}})/800 "
    rf"total={_LOG_FIXED_6} "
    rf"seg={_LOG_FIXED_6} "
    rf"surv={_LOG_FIXED_6} "
    rf"mIoU={_LOG_FIXED_6} "
    rf"Pd={_LOG_FIXED_6} "
    rf"Fa={_LOG_FIXED_8}$"
)


class EngineeringGateError(ValueError):
    """Present engineering evidence is inconsistent or not authoritative."""


def _fail(message: str) -> None:
    raise EngineeringGateError(message)


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


def _sha256_file(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(f"expected a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(f"{label} must be a regular non-symlink file: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid UTF-8 JSON: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    return value, raw


def _canonical_equal(label: str, observed: Any, expected: Any) -> None:
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        _fail(f"{label} differs")


def _source_bindings() -> dict[str, dict[str, str]]:
    paths = {
        "adjudicator": Path(__file__).resolve(),
        "certification_protocol": PROTOCOL_PATH.resolve(),
        "final_design": FINAL_DESIGN_PATH.resolve(),
        "summary_contract": Path(summary_core.__file__).resolve(),
        "checkpoint_sweep_contract": Path(
            evaluation_core.__file__
        ).resolve(),
    }
    bindings: dict[str, dict[str, str]] = {}
    for role, path in paths.items():
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            _fail(f"{role} source lies outside the repository")
        bindings[role] = {
            "path": relative,
            "sha256": _sha256_file(path),
        }
    return bindings


def _expected_artifacts(
    *,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = [
        {"role": "certification_source_lock", "path": source_lock_path},
        {"role": "replication_seed_contract", "path": seed_contract_path},
        {"role": "four_run_summary", "path": summary_path},
    ]
    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
        for arm in core.SUPPORTED_ARMS:
            run_directory = watcher.run_directory(
                output_root,
                trajectory_seed,
                arm,
            )
            expected.append(
                {
                    "role": (
                        f"seed_{trajectory_seed}_{arm}_child_manifest"
                    ),
                    "path": prepare.manifest_path(
                        manifest_directory,
                        seed=trajectory_seed,
                        arm=arm,
                    ),
                }
            )
            for filename in (
                "protocol.json",
                "split.json",
                "summary.json",
                "metrics.jsonl",
                "exact_journal/active.json",
                "last.pth.tar",
                "best_miou.pth.tar",
                "best.pth.tar",
            ):
                expected.append(
                    {
                        "role": (
                            f"seed_{trajectory_seed}_{arm}_"
                            f"{filename.replace('/', '_')}"
                        ),
                        "path": run_directory / filename,
                    }
                )
            expected.append(
                {
                    "role": f"seed_{trajectory_seed}_{arm}_launcher_log",
                    "path": (
                        output_root
                        / "logs"
                        / f"seed_{trajectory_seed}_{arm}.log"
                    ),
                }
            )
            for checkpoint_filename, _, _ in (
                evaluation_core.CHECKPOINT_SPECS
            ):
                expected.append(
                    {
                        "role": (
                            f"seed_{trajectory_seed}_{arm}_"
                            f"{checkpoint_filename}_sweep"
                        ),
                        "path": _planned_sweep_path(
                            run_directory,
                            checkpoint_filename,
                        ),
                    }
                )
    return expected


def _planned_sweep_path(
    run_directory: Path,
    checkpoint_filename: str,
) -> Path:
    """Mirror ``CheckpointEvaluationRequest.planned_output_path`` exactly."""

    registered_checkpoints = {
        filename for filename, _, _ in evaluation_core.CHECKPOINT_SPECS
    }
    if checkpoint_filename not in registered_checkpoints:
        _fail(
            "unregistered checkpoint filename for engineering sweep: "
            f"{checkpoint_filename}"
        )
    return run_directory / (
        f"pd_fa_sweep_{Path(checkpoint_filename).stem}.json"
    )


def _inventory(
    expected: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    missing: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    for record in expected:
        role = str(record["role"])
        path = Path(record["path"])
        ready = {"role": role, "path": str(path.resolve())}
        if path.is_symlink():
            invalid.append({**ready, "reason": "symlink"})
        elif not path.exists():
            missing.append(ready)
        elif not path.is_file():
            invalid.append({**ready, "reason": "not_regular_file"})
    return missing, invalid


def _validated_launcher_log(
    *,
    output_root: Path,
    trajectory_seed: int,
    arm: str,
    summary_index: Mapping[
        tuple[int, str, str],
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """Bind one append-only launcher log to its completed run boundary."""

    run_directory = watcher.run_directory(
        output_root,
        trajectory_seed,
        arm,
    ).resolve()
    path = (
        Path(output_root)
        / "logs"
        / f"seed_{trajectory_seed}_{arm}.log"
    )
    if path.is_symlink() or not path.is_file():
        _fail(f"launcher log must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    if not raw:
        _fail(f"launcher log is empty: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"launcher log is not valid UTF-8: {path}: {exc}")
    lines = text.splitlines()
    nonempty_lines = [line for line in lines if line]
    if not nonempty_lines:
        _fail(f"launcher log has no non-empty lines: {path}")

    epoch_numbers = [
        int(match.group("epoch"))
        for line in lines
        if (match := _EPOCH_LOG_PATTERN.fullmatch(line)) is not None
    ]
    expected_epochs = list(range(1, watcher.FORMAL_EPOCHS + 1))
    if epoch_numbers != expected_epochs:
        _fail(
            "launcher log epoch sequence must be exactly 1..800 once "
            f"for seed={trajectory_seed}, arm={arm}"
        )

    definition = core.arm_definition(arm)
    start_prefix = f"START variant={definition.variant} "
    if not any(line.startswith(start_prefix) for line in lines):
        _fail(
            "launcher log has no run START boundary for "
            f"seed={trajectory_seed}, arm={arm}"
        )

    best_pd = summary_index.get(
        (trajectory_seed, arm, "best.pth.tar")
    )
    best_miou = summary_index.get(
        (trajectory_seed, arm, "best_miou.pth.tar")
    )
    if best_pd is None or best_miou is None:
        _fail(
            "four-run summary omits launcher completion epochs for "
            f"seed={trajectory_seed}, arm={arm}"
        )
    best_pd_epoch = _count(
        best_pd.get("epoch"),
        f"seed={trajectory_seed}, arm={arm} best-Pd epoch",
    )
    best_miou_epoch = _count(
        best_miou.get("epoch"),
        f"seed={trajectory_seed}, arm={arm} best-mIoU epoch",
    )
    if not (
        1 <= best_pd_epoch <= watcher.FORMAL_EPOCHS
        and 1 <= best_miou_epoch <= watcher.FORMAL_EPOCHS
    ):
        _fail(
            "launcher completion checkpoint epoch lies outside 1..800 "
            f"for seed={trajectory_seed}, arm={arm}"
        )
    complete_boundary = (
        f"COMPLETE variant={definition.variant} "
        f"bestPdEpoch={best_pd_epoch} "
        f"bestMiouEpoch={best_miou_epoch}"
    )
    output_boundary = f"OUTPUT {run_directory}"
    complete_lines = [
        line for line in lines if line.startswith("COMPLETE ")
    ]
    output_lines = [line for line in lines if line.startswith("OUTPUT ")]
    if complete_lines != [complete_boundary]:
        _fail(
            "launcher log COMPLETE boundary differs or is not unique "
            f"for seed={trajectory_seed}, arm={arm}"
        )
    if output_lines != [output_boundary]:
        _fail(
            "launcher log OUTPUT boundary differs or is not unique "
            f"for seed={trajectory_seed}, arm={arm}"
        )
    if nonempty_lines[-2:] != [complete_boundary, output_boundary]:
        _fail(
            "launcher log COMPLETE/OUTPUT must be the final two non-empty "
            f"lines for seed={trajectory_seed}, arm={arm}"
        )
    return {
        "trajectory_seed": trajectory_seed,
        "arm": arm,
        "variant": definition.variant,
        "run_directory": str(run_directory),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "line_count": len(lines),
        "nonempty_line_count": len(nonempty_lines),
        "epoch_line_count": len(epoch_numbers),
        "epoch_sequence": "exactly_1_through_800_once",
        "complete_boundary": complete_boundary,
        "output_boundary": output_boundary,
        "status": "verified_complete",
    }


def _base_payload(
    *,
    status: str,
    decision: str,
    missing: Sequence[Mapping[str, Any]] = (),
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "scope": "fixed_parent_engineering_b_d_only",
        "gate": "S-E_engineering_replication_completeness",
        "engineering_trajectory_seeds": list(
            seeds.ENGINEERING_TRAJECTORY_SEEDS
        ),
        "arms": {
            core.ARM_B: core.arm_definition(core.ARM_B).variant,
            core.ARM_D: core.arm_definition(core.ARM_D).variant,
        },
        "expected_run_count": 4,
        "expected_checkpoint_count": 12,
        "expected_sweep_count": evaluation_core.EXPECTED_SWEEP_COUNT,
        "checkpoint_policy": {
            "primary": "each_arm_own_best_miou",
            "secondary": "each_arm_own_best_pd",
            "cross_arm_shared_epoch_required": False,
        },
        "fixed_threshold": evaluation_core.FIXED_THRESHOLD,
        "fa_budgets": list(evaluation_core.FA_BUDGETS),
        "key_secondary_fa_budget": KEY_SECONDARY_FA_BUDGET,
        "missing_artifacts": [dict(item) for item in missing],
        "errors": list(errors),
        "gates": {
            "S-E": {
                "status": status,
                "passed": True if status == "complete" else None,
                "engineering_replication_complete": status == "complete",
            },
            "M-train": {
                "status": "insufficient_evidence",
                "passed": None,
                "required_evidence": (
                    "same-seed image-level paired sufficient statistics "
                    "with five-metric Bonferroni simultaneous 95% CI"
                ),
                "available_evidence": (
                    "checkpoint-level aggregate fixed points and five "
                    "checkpoint-local Fa-budget envelopes"
                ),
                "aggregate_point_estimates_used_as_ci": False,
            },
        },
        "claim_boundary": {
            "engineering_gate_is_paper_stability_gate": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "full_pipeline_stability_supported": False,
            "multiseed_replication_supported": False,
            "official_test_accessed": False,
        },
        "persistent_artifact_written": False,
    }


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be finite numeric")
    return float(value)


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _metric_view(point: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(point, Mapping):
        _fail(f"{label} is not a metric point")
    missing = [name for name in FIXED_METRIC_FIELDS if name not in point]
    if missing:
        _fail(f"{label} omits metrics: {missing}")
    return {
        "threshold": (
            _finite_number(point["threshold"], f"{label}.threshold")
            if "threshold" in point
            else evaluation_core.FIXED_THRESHOLD
        ),
        "pd": _finite_number(point["pd"], f"{label}.pd"),
        "matched_target_count": _count(
            point["matched_target_count"],
            f"{label}.matched_target_count",
        ),
        "target_count": _count(
            point["target_count"],
            f"{label}.target_count",
        ),
        "fa": _finite_number(point["fa"], f"{label}.fa"),
        "miou": _finite_number(point["miou"], f"{label}.miou"),
        "tiny_pd": _finite_number(
            point["tiny_pd"],
            f"{label}.tiny_pd",
        ),
        "matched_tiny_target_count": _count(
            point["matched_tiny_target_count"],
            f"{label}.matched_tiny_target_count",
        ),
        "tiny_target_count": _count(
            point["tiny_target_count"],
            f"{label}.tiny_target_count",
        ),
        "false_objects_per_image": _finite_number(
            point["false_objects_per_image"],
            f"{label}.false_objects_per_image",
        ),
        "unmatched_predicted_object_count": _count(
            point["unmatched_predicted_object_count"],
            f"{label}.unmatched_predicted_object_count",
        ),
        "valid_pixel_count": _count(
            point["valid_pixel_count"],
            f"{label}.valid_pixel_count",
        ),
    }


def _metric_delta(
    d_metrics: Mapping[str, Any],
    b_metrics: Mapping[str, Any],
) -> dict[str, float | int]:
    return {
        "pd": float(d_metrics["pd"]) - float(b_metrics["pd"]),
        "fa": float(d_metrics["fa"]) - float(b_metrics["fa"]),
        "miou": float(d_metrics["miou"]) - float(b_metrics["miou"]),
        "tiny_pd": (
            float(d_metrics["tiny_pd"]) - float(b_metrics["tiny_pd"])
        ),
        "false_objects_per_image": (
            float(d_metrics["false_objects_per_image"])
            - float(b_metrics["false_objects_per_image"])
        ),
        "unmatched_predicted_object_count": (
            int(d_metrics["unmatched_predicted_object_count"])
            - int(b_metrics["unmatched_predicted_object_count"])
        ),
    }


def _point_estimate_miou_route(
    d_metrics: Mapping[str, Any],
    b_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "pd_noninferior": float(d_metrics["pd"]) >= float(b_metrics["pd"]),
        "tiny_pd_noninferior": (
            float(d_metrics["tiny_pd"]) >= float(b_metrics["tiny_pd"])
        ),
        "false_objects_noninferior": (
            int(d_metrics["unmatched_predicted_object_count"])
            <= int(b_metrics["unmatched_predicted_object_count"])
        ),
        "miou_superior": (
            float(d_metrics["miou"]) > float(b_metrics["miou"])
        ),
        "fa_noninferior": float(d_metrics["fa"]) <= float(b_metrics["fa"]),
    }
    return {
        "status": (
            "direction_met" if all(checks.values()) else "direction_not_met"
        ),
        "checks": checks,
        "all_point_estimate_inequalities_met": all(checks.values()),
        "uses_confidence_interval": False,
        "establishes_gate_m_train": False,
    }


def _summary_checkpoint_index(
    summary: Mapping[str, Any],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    runs = summary.get("runs")
    if not isinstance(runs, list):
        _fail("four-run summary has no run list")
    index: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            _fail("four-run summary contains a non-object run")
        checkpoints = run.get("checkpoints")
        if not isinstance(checkpoints, list):
            _fail("four-run summary run has no checkpoint list")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                _fail("four-run summary checkpoint is not an object")
            key = (
                int(run["trajectory_seed"]),
                str(run["arm"]),
                str(checkpoint["filename"]),
            )
            if key in index:
                _fail(f"duplicate summary checkpoint identity: {key}")
            index[key] = checkpoint
    return index


def _validate_summary_request_binding(
    summary_index: Mapping[
        tuple[int, str, str],
        Mapping[str, Any],
    ],
    request: evaluation_core.CheckpointEvaluationRequest,
) -> None:
    key = (
        request.trajectory_seed,
        request.arm,
        request.checkpoint_filename,
    )
    record = summary_index.get(key)
    if record is None:
        _fail(f"summary omits checkpoint identity: {key}")
    for name, observed, expected in (
        ("SHA-256", record.get("sha256"), request.checkpoint_sha256),
        ("epoch", record.get("epoch"), request.checkpoint_epoch),
        (
            "selection role",
            record.get("selection_role"),
            request.selection_role,
        ),
        (
            "checkpoint role",
            record.get("checkpoint_role"),
            request.checkpoint_role,
        ),
    ):
        if observed != expected:
            _fail(f"summary/request {key} {name} differs")
    _canonical_equal(
        f"summary/request {key} validation metrics",
        record.get("metrics"),
        request.checkpoint_validation_metrics,
    )


def _comparison_record(
    *,
    trajectory_seed: int,
    checkpoint_filename: str,
    selection_role: str,
    b_result: Mapping[str, Any],
    d_result: Mapping[str, Any],
) -> dict[str, Any]:
    b_fixed = _metric_view(
        b_result["fixed_threshold_0_5"],
        "B fixed threshold",
    )
    d_fixed = _metric_view(
        d_result["fixed_threshold_0_5"],
        "D fixed threshold",
    )
    budget_records: dict[str, Any] = {}
    for key, budget in zip(
        evaluation_core.BUDGET_KEYS,
        evaluation_core.FA_BUDGETS,
    ):
        b_point = _metric_view(
            b_result["best_points_under_fa_budget"][key],
            f"B Fa budget {key}",
        )
        d_point = _metric_view(
            d_result["best_points_under_fa_budget"][key],
            f"D Fa budget {key}",
        )
        budget_records[key] = {
            "fa_budget": budget,
            "b": b_point,
            "d": d_point,
            "d_minus_b": _metric_delta(d_point, b_point),
            "pd_direction": (
                "D_higher"
                if d_point["pd"] > b_point["pd"]
                else "equal"
                if d_point["pd"] == b_point["pd"]
                else "D_lower"
            ),
            "participates_in_engineering_gate": False,
        }
    return {
        "trajectory_seed": trajectory_seed,
        "checkpoint_filename": checkpoint_filename,
        "selection_role": selection_role,
        "checkpoint_policy": "each_arm_own_selected_checkpoint",
        "b": b_fixed,
        "d": d_fixed,
        "d_minus_b": _metric_delta(d_fixed, b_fixed),
        "descriptive_miou_route_point_estimate": (
            _point_estimate_miou_route(d_fixed, b_fixed)
        ),
        "fa_budget_envelopes": budget_records,
        "key_secondary_pd_at_fa_le_5e_6": budget_records["5e-06"],
    }


def adjudicate(
    *,
    output_root: Path = watcher.DEFAULT_OUTPUT_ROOT,
    source_lock_path: Path = source_lock_core.DEFAULT_OUTPUT,
    seed_contract_path: Path = prepare.DEFAULT_SEED_CONTRACT,
    manifest_directory: Path = prepare.DEFAULT_MANIFEST_DIRECTORY,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Return ``pending``, ``invalid``, or a complete Gate S-E decision."""

    root = Path(output_root).expanduser().resolve()
    source_lock = Path(source_lock_path).expanduser()
    seed_contract = Path(seed_contract_path).expanduser()
    manifests = Path(manifest_directory).expanduser()
    aggregate_summary = (
        root / DEFAULT_SUMMARY_FILENAME
        if summary_path is None
        else Path(summary_path).expanduser()
    )
    expected = _expected_artifacts(
        output_root=root,
        source_lock_path=source_lock,
        seed_contract_path=seed_contract,
        manifest_directory=manifests,
        summary_path=aggregate_summary,
    )
    missing, invalid = _inventory(expected)
    if invalid:
        return _base_payload(
            status="invalid",
            decision="ENGINEERING_GATE_INVALID",
            errors=[
                f"{item['role']}: {item['reason']} ({item['path']})"
                for item in invalid
            ],
        )
    if missing:
        return _base_payload(
            status="pending",
            decision="ENGINEERING_GATE_PENDING",
            missing=missing,
        )

    try:
        stored_summary, summary_raw = _load_json(
            aggregate_summary,
            "four-run engineering summary",
        )
        if summary_raw != summary_core.canonical_json_bytes(
            stored_summary
        ):
            _fail("four-run engineering summary is not canonical JSON")
        live_summary = summary_core.build_summary(
            output_root=root,
            source_lock_path=source_lock,
            seed_contract_path=seed_contract,
            manifest_directory=manifests,
        )
        _canonical_equal(
            "stored/live four-run engineering summary",
            stored_summary,
            live_summary,
        )
        for name, expected_value in (
            ("schema", summary_core.SCHEMA),
            ("status", "complete"),
            ("scope", "fixed_parent_engineering_b_d_only"),
            ("run_count", 4),
            ("checkpoint_count", 8),
            ("fixed_threshold", evaluation_core.FIXED_THRESHOLD),
            ("official_test_accessed", False),
        ):
            if stored_summary.get(name) != expected_value:
                _fail(f"four-run summary {name} differs")

        requests: list[evaluation_core.CheckpointEvaluationRequest] = []
        for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
            for arm in core.SUPPORTED_ARMS:
                requests.extend(
                    evaluation_core.preflight_completed_run(
                        arm=arm,
                        trajectory_seed=trajectory_seed,
                        output_root=root,
                        source_lock_path=source_lock,
                        seed_contract_path=seed_contract,
                        manifest_directory=manifests,
                    )
                )
        evaluation_plan = evaluation_core.assemble_evaluation_plan(requests)
        if evaluation_plan.get("request_count") != 8:
            _fail("checkpoint-local evaluation plan does not contain 8 requests")
        data_contracts = {
            (
                request.training_data_sha256,
                request.normalization_sha256,
                request.validation_split_sha256,
                evaluation_core.statistics_cache
                .validation_identifier_sha256(request.validation_ids),
                len(request.validation_ids),
            )
            for request in requests
        }
        if len(data_contracts) != 1:
            _fail("eight checkpoint requests do not share one data contract")
        (
            training_data_sha256,
            normalization_sha256,
            validation_split_sha256,
            validation_ids_sha256,
            validation_count,
        ) = next(iter(data_contracts))
        summary_index = _summary_checkpoint_index(stored_summary)
        launcher_logs = [
            _validated_launcher_log(
                output_root=root,
                trajectory_seed=trajectory_seed,
                arm=arm,
                summary_index=summary_index,
            )
            for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
            for arm in core.SUPPORTED_ARMS
        ]
        validated_results: dict[
            tuple[int, str, str],
            dict[str, Any],
        ] = {}
        sweep_bindings: list[dict[str, Any]] = []
        for request in requests:
            _validate_summary_request_binding(summary_index, request)
            result_path = request.planned_output_path
            stored_result, _ = _load_json(
                result_path,
                "checkpoint-local sweep result",
            )
            if stored_result.get("schema") != evaluation_core.RESULT_SCHEMA:
                _fail(
                    "stored checkpoint-local sweep is not an adapter-"
                    "finalized result"
                )
            validated = evaluation_core.validate_checkpoint_local_result(
                stored_result,
                request,
            )
            _canonical_equal(
                "stored/validated checkpoint-local sweep",
                stored_result,
                validated,
            )
            key = (
                request.trajectory_seed,
                request.arm,
                request.checkpoint_filename,
            )
            if key in validated_results:
                _fail(f"duplicate validated sweep identity: {key}")
            validated_results[key] = validated
            sweep_bindings.append(
                {
                    "trajectory_seed": request.trajectory_seed,
                    "arm": request.arm,
                    "variant": request.variant,
                    "selection_role": request.selection_role,
                    "checkpoint_filename": request.checkpoint_filename,
                    "checkpoint_epoch": request.checkpoint_epoch,
                    "checkpoint_sha256": request.checkpoint_sha256,
                    "metrics_sha256": request.metrics_sha256,
                    "training_data_sha256": (
                        request.training_data_sha256
                    ),
                    "normalization_sha256": (
                        request.normalization_sha256
                    ),
                    "validation_split_sha256": (
                        request.validation_split_sha256
                    ),
                    "validation_ids_sha256": (
                        evaluation_core.statistics_cache
                        .validation_identifier_sha256(
                            request.validation_ids
                        )
                    ),
                    "validation_count": len(request.validation_ids),
                    "threshold_domain_id": request.threshold_domain_id,
                    "path": str(result_path.resolve()),
                    "sha256": _sha256_file(result_path),
                }
            )

        comparisons: list[dict[str, Any]] = []
        for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
            for checkpoint_filename, selection_role, _ in (
                evaluation_core.CHECKPOINT_SPECS
            ):
                comparisons.append(
                    _comparison_record(
                        trajectory_seed=trajectory_seed,
                        checkpoint_filename=checkpoint_filename,
                        selection_role=selection_role,
                        b_result=validated_results[
                            (
                                trajectory_seed,
                                core.ARM_B,
                                checkpoint_filename,
                            )
                        ],
                        d_result=validated_results[
                            (
                                trajectory_seed,
                                core.ARM_D,
                                checkpoint_filename,
                            )
                        ],
                    )
                )
        primary = [
            record
            for record in comparisons
            if record["checkpoint_filename"] == PRIMARY_CHECKPOINT
        ]
        direction_count = sum(
            int(
                record[
                    "descriptive_miou_route_point_estimate"
                ]["all_point_estimate_inequalities_met"]
            )
            for record in primary
        )
        payload = _base_payload(
            status="complete",
            decision="ENGINEERING_GATE_S_E_PASS",
        )
        payload.update(
            {
                "source_bindings": _source_bindings(),
                "evidence": {
                    "summary": {
                        "path": str(aggregate_summary.resolve()),
                        "sha256": _sha256_file(aggregate_summary),
                        "live_rebuilt_and_equal": True,
                    },
                    "run_count": 4,
                    "checkpoint_artifact_count": 12,
                    "selected_checkpoint_count": 8,
                    "validated_sweep_count": len(sweep_bindings),
                    "launcher_log_count": len(launcher_logs),
                    "launcher_logs": sorted(
                        launcher_logs,
                        key=lambda item: (
                            item["trajectory_seed"],
                            item["arm"],
                        ),
                    ),
                    "data_contract": {
                        "dataset": evaluation_core.DATASET,
                        "split_seed": seeds.SPLIT_SEED,
                        "training_data_sha256": training_data_sha256,
                        "normalization_sha256": normalization_sha256,
                        "validation_split_sha256": (
                            validation_split_sha256
                        ),
                        "validation_ids_sha256": validation_ids_sha256,
                        "validation_count": validation_count,
                        "official_test_accessed": False,
                    },
                    "sweeps": sorted(
                        sweep_bindings,
                        key=lambda item: (
                            item["trajectory_seed"],
                            item["arm"],
                            item["checkpoint_filename"],
                        ),
                    ),
                    "each_sweep_checkpoint_local": True,
                    "cross_checkpoint_point_pooling": False,
                },
                "fixed_threshold_and_budget_comparisons": comparisons,
                "engineering_performance_screen": {
                    "status": "descriptive_point_estimates_complete",
                    "primary_checkpoint": (
                        "each_arm_own_best_miou"
                    ),
                    "comparison": "D_minus_B",
                    "route": "MIOU_ROUTE_point_estimate_analogue",
                    "seed_count": len(primary),
                    "seed_direction_met_count": direction_count,
                    "all_engineering_seed_directions_met": (
                        direction_count == len(primary)
                    ),
                    "uses_paired_image_level_ci": False,
                    "establishes_gate_m_train": False,
                    "affects_gate_s_e_pass": False,
                },
            }
        )
        return payload
    except Exception as exc:
        return _base_payload(
            status="invalid",
            decision="ENGINEERING_GATE_INVALID",
            errors=[f"{type(exc).__name__}: {exc}"],
        )


def write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser()
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(
            f"refusing to replace engineering Gate output: {destination}"
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
        "--output-root",
        type=Path,
        default=watcher.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=source_lock_core.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=prepare.DEFAULT_SEED_CONTRACT,
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=prepare.DEFAULT_MANIFEST_DIRECTORY,
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = adjudicate(
        output_root=args.output_root,
        source_lock_path=args.source_lock,
        seed_contract_path=args.seed_contract,
        manifest_directory=args.manifest_directory,
        summary_path=args.summary,
    )
    output_record: dict[str, Any] | None = None
    if args.output is not None and payload["status"] == "complete":
        destination = write_once(args.output, payload)
        output_record = {
            "path": str(destination.resolve()),
            "sha256": _sha256_file(destination),
        }
    action = {
        "schema": ACTION_SCHEMA,
        "status": payload["status"],
        "decision": payload["decision"],
        "gate": payload,
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
    "DEFAULT_OUTPUT",
    "EngineeringGateError",
    "KEY_SECONDARY_FA_BUDGET",
    "PRIMARY_CHECKPOINT",
    "SCHEMA",
    "adjudicate",
    "canonical_json_bytes",
    "main",
    "write_once",
]


if __name__ == "__main__":
    main()
