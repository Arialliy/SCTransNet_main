#!/usr/bin/env python3
"""Publish and verify a closed TPD-Clean-v3 screen800 result bundle.

The canonical summarizer remains the sole producer of the scientific report.
This utility adds a strict, hash-bound completion layer around that report:

* ``publish`` accepts already-rendered canonical JSON/Markdown in a staging
  directory, re-runs the canonical audit against the current artifacts, and
  publishes the report, an exact input manifest, and a marker written last.
* ``verify`` re-derives the exact expected input set and checks every current
  byte against the published manifest and completion marker.

No training, checkpoint, sweep, frozen reference, or source-lock artifact is
modified by this program.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARIZER = (
    REPO_ROOT / "experiments/summarize_tpd_clean_v3_screen800.py"
)
DEFAULT_POSTPROCESS_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v3_postprocess_source_lock.json"
)

MANIFEST_NAME = "completion_inputs.json"
MARKER_NAME = "COMPLETE.sha256"
MANIFEST_SCHEMA = "sctransnet_tpd_clean_v3_completion_inputs_v1"

DATASET = "NUDT-SIRST"
VARIANTS = ("tpd_clean_v3_full", "tpd_clean_v3_sal_capacity")
SEEDS = (42, 3407)
RUN_TAG = "screen800_pd_fp32_shared4x5090_v1"
ROLES = ("pd_primary", "miou_primary")
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
REFERENCE_METHODS = ("spd", "tpd_v1", "v2_sal_only", "v2_full")
GATE_NAMES = (
    "gate_1_seed42_pd_primary_fixed",
    "gate_2_seed42_miou_primary_fixed",
    "gate_3_seed42_budget_floors",
    "gate_4_seed42_frozen_references",
    "gate_5_no_capacity_dominance",
    "gate_6_paired_advantage_and_wide_pd",
    "gate_7_fixed_sweep_direction_coherence",
)
POINT_FIELDS = ("matched_target_count", "target_count", "fa", "miou", "pd")

EXPECTED_INPUT_COUNTS = {
    "candidate_run_files": 36,
    "candidate_launch_manifests": 4,
    "candidate_worker_logs": 4,
    "frozen_reference_checkpoints": 8,
    "frozen_reference_sweeps": 8,
    "canonical_summarizers": 1,
    "source_locks": 4,
    "total_files": 65,
}

LOCK_SPECS = {
    "training": (
        "experiments/tpd_clean_v3_screen800_source_lock.json",
        "sctransnet_tpd_clean_v3_screen800_source_lock_v1",
    ),
    "v2": (
        "experiments/tpd_clean_screen800_source_lock.json",
        "sctransnet_tpd_clean_screen800_source_lock_v1",
    ),
    "ner": (
        "experiments/tpd_ner_v1_source_lock.json",
        "sctransnet_tpd_ner_v1_source_lock_v1",
    ),
}


class CompletionValidationError(ValueError):
    """The completion bundle or one of its bound inputs is invalid."""


def _fail(message: str) -> None:
    raise CompletionValidationError(message)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _reject_json_constant(value: str) -> None:
    _fail(f"JSON contains a non-finite constant: {value}")


def _require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label}: non-finite numeric value")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_tree(nested, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite_tree(nested, f"{label}[{index}]")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except CompletionValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}: invalid JSON at {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label}: JSON root must be an object")
    _require_finite_tree(payload, label)
    return payload


def _expect_exact_keys(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label}: expected an object")
    observed = set(value)
    if observed != expected:
        _fail(
            f"{label}: keys differ; "
            f"missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label}: missing, linked, or non-regular file: {path}")


def _resolve_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{label}: missing, linked, or non-directory path: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}: cannot resolve {path}: {exc}")


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _relative_regular_file(
    root: Path, relative: str, label: str
) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        _fail(f"{label}: unsafe relative path: {relative!r}")
    lexical = root.joinpath(*pure.parts)
    _require_regular_file(lexical, label)
    resolved = lexical.resolve(strict=True)
    if not _within(resolved, root):
        _fail(f"{label}: file escaped expected root {root}: {lexical}")
    return resolved


def _relative_to_root(path: Path, root: Path, label: str) -> str:
    _require_regular_file(path, label)
    resolved = path.resolve(strict=True)
    if not _within(resolved, root):
        _fail(f"{label}: file escaped expected root {root}: {path}")
    return resolved.relative_to(root).as_posix()


def _safe_output_directory(path: Path, candidate_root: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not _within(resolved, candidate_root):
        _fail(
            f"output directory must be inside candidate root "
            f"{candidate_root}: {path}"
        )
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        _fail(f"output directory is linked or not a directory: {path}")
    cursor = resolved
    while cursor != candidate_root:
        if cursor.exists() and cursor.is_symlink():
            _fail(f"output directory has a linked component: {cursor}")
        if candidate_root not in cursor.parents:
            _fail(f"output directory escaped candidate root: {path}")
        cursor = cursor.parent
    return resolved


def _parse_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label}: expected an ISO timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        _fail(f"{label}: invalid ISO timestamp: {exc}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{label}: timestamp must include a timezone")
    return value


def _load_source_lock(
    path: Path,
    *,
    repo: Path,
    expected_schema: str,
    label: str,
    validate_sources: bool,
) -> dict[str, Any]:
    payload = _load_json(path, label)
    if payload.get("schema") != expected_schema:
        _fail(
            f"{label}: schema={payload.get('schema')!r}, "
            f"expected {expected_schema!r}"
        )
    entries = payload.get("source_sha256")
    if not isinstance(entries, dict) or not entries:
        _fail(f"{label}: source_sha256 must be a non-empty object")
    seen: set[str] = set()
    for relative, expected_sha in entries.items():
        if not isinstance(relative, str) or relative in seen:
            _fail(f"{label}: invalid or duplicate source path {relative!r}")
        seen.add(relative)
        if not _valid_sha256(expected_sha):
            _fail(f"{label}: invalid SHA-256 for {relative!r}")
        if validate_sources:
            source = _relative_regular_file(
                repo, relative, f"{label}.source[{relative}]"
            )
            actual = _sha256_file(source)
            if actual != expected_sha:
                _fail(
                    f"{label}: source digest mismatch for {relative}; "
                    f"expected={expected_sha} actual={actual}"
                )
    return payload


def _load_canonical_summarizer(
    *,
    repo: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
) -> tuple[ModuleType, dict[str, Any]]:
    expected_relative = "experiments/summarize_tpd_clean_v3_screen800.py"
    observed_relative = _relative_to_root(
        summarizer_path, repo, "canonical summarizer"
    )
    if observed_relative != expected_relative:
        _fail(
            "canonical summarizer path mismatch: "
            f"{observed_relative!r} != {expected_relative!r}"
        )
    postprocess_relative = _relative_to_root(
        postprocess_lock_path, repo, "postprocess source lock"
    )
    if (
        postprocess_relative
        != "experiments/tpd_clean_v3_postprocess_source_lock.json"
    ):
        _fail(f"unexpected postprocess lock path: {postprocess_relative}")
    postprocess = _load_source_lock(
        postprocess_lock_path,
        repo=repo,
        expected_schema="sctransnet_tpd_clean_v3_postprocess_source_lock_v1",
        label="postprocess source lock",
        validate_sources=True,
    )
    entries = postprocess["source_sha256"]
    summarizer_sha = _sha256_file(summarizer_path)
    if entries.get(expected_relative) != summarizer_sha:
        _fail(
            "canonical summarizer is not bound by the postprocess source lock"
        )
    validator_relative = (
        "experiments/validate_tpd_clean_v3_completion.py"
    )
    validator_path = _relative_regular_file(
        repo, validator_relative, "canonical completion validator"
    )
    validator_sha = _sha256_file(validator_path)
    runtime_validator_sha = _sha256_file(Path(__file__).resolve(strict=True))
    if (
        entries.get(validator_relative) != validator_sha
        or runtime_validator_sha != validator_sha
    ):
        _fail(
            "canonical completion validator is not bound by the "
            "postprocess source lock"
        )
    training_relative = LOCK_SPECS["training"][0]
    training_path = _relative_regular_file(
        repo, training_relative, "training source lock"
    )
    if (
        postprocess.get("training_source_lock_sha256")
        != _sha256_file(training_path)
    ):
        _fail(
            "postprocess source lock does not bind the current training "
            "source lock"
        )
    policy = postprocess.get("policy")
    required_policy = {
        "separate_from_training_source_lock": True,
        "does_not_modify_frozen_training_results": True,
        "candidate_null_budget_points_forbidden": True,
        "unused_frozen_reference_null_points_disclosed": True,
        "required_gate_reference_null_points_forbidden": True,
        "automatic_mainline_replacement": False,
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) is not expected
        for key, expected in required_policy.items()
    ):
        _fail("postprocess source lock policy is incomplete or changed")

    module_name = (
        "_tpd_clean_v3_canonical_summary_"
        + summarizer_sha[:16]
        + "_"
        + str(os.getpid())
    )
    spec = importlib.util.spec_from_file_location(module_name, summarizer_path)
    if spec is None or spec.loader is None:
        _fail(f"cannot import canonical summarizer: {summarizer_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(f"canonical summarizer import failed: {exc}")
    required_attributes = (
        "SCHEMA",
        "JSON_OUTPUT_NAME",
        "MARKDOWN_OUTPUT_NAME",
        "VARIANTS",
        "SEEDS",
        "ROLE_SPECS",
        "BUDGET_KEYS",
        "REFERENCE_GATE_BUDGET_USAGE",
        "render_markdown",
        "build_report",
        "evaluate_engineering_gate",
        "_candidate_run_dir",
        "_reference_paths",
    )
    missing = [name for name in required_attributes if not hasattr(module, name)]
    if missing:
        _fail(f"canonical summarizer lacks required attributes: {missing}")
    if Path(module.REPO_ROOT).resolve() != repo:
        _fail(
            f"canonical summarizer repository mismatch: "
            f"{module.REPO_ROOT} != {repo}"
        )
    if tuple(module.VARIANTS) != VARIANTS or tuple(module.SEEDS) != SEEDS:
        _fail("canonical summarizer candidate scope changed")
    if tuple(module.BUDGET_KEYS) != BUDGET_KEYS:
        _fail("canonical summarizer budget scope changed")
    if set(module.ROLE_SPECS) != set(ROLES):
        _fail("canonical summarizer checkpoint roles changed")
    return module, postprocess


def _validate_point(point: Any, label: str) -> None:
    if not isinstance(point, Mapping):
        _fail(f"{label}: point must be an object")
    _require_finite_tree(point, label)
    missing = [field for field in POINT_FIELDS if field not in point]
    if missing:
        _fail(f"{label}: missing point fields {missing}")
    matched = point["matched_target_count"]
    target = point["target_count"]
    if (
        isinstance(matched, bool)
        or not isinstance(matched, int)
        or isinstance(target, bool)
        or not isinstance(target, int)
    ):
        _fail(f"{label}: target counts must be integers")
    if target != 189 or not 0 <= matched <= target:
        _fail(f"{label}: invalid matched/target count {matched}/{target}")
    for name in ("fa", "miou", "pd"):
        value = point[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            _fail(f"{label}: {name} must be finite numeric")
    if float(point["fa"]) < 0:
        _fail(f"{label}: Fa cannot be negative")
    if not 0 <= float(point["miou"]) <= 1:
        _fail(f"{label}: mIoU must be in [0, 1]")
    if abs(float(point["pd"]) - matched / target) > 1e-15:
        _fail(f"{label}: Pd is inconsistent with target counts")


def _validate_role(
    role: Any,
    *,
    label: str,
    candidate: bool,
    required_reference_usage: Mapping[tuple[str, str, str], str],
    method: str | None = None,
    role_name: str,
) -> list[dict[str, Any]]:
    expected_keys = {
        "checkpoint_epoch",
        "checkpoint_role",
        "fixed_threshold_0_5",
        "budgets",
        "checkpoint",
        "checkpoint_sha256",
        "sweep",
        "sweep_sha256",
        "validation_split_sha256",
        "integrity_checks_passed",
    }
    mapping = _expect_exact_keys(role, expected_keys, label)
    if isinstance(mapping["checkpoint_epoch"], bool) or not isinstance(
        mapping["checkpoint_epoch"], int
    ):
        _fail(f"{label}: checkpoint_epoch must be an integer")
    for key in ("checkpoint_sha256", "sweep_sha256"):
        if not _valid_sha256(mapping[key]):
            _fail(f"{label}: {key} must be lowercase SHA-256")
    if not _valid_sha256(mapping["validation_split_sha256"]):
        _fail(f"{label}: validation_split_sha256 must be lowercase SHA-256")
    for key in ("checkpoint", "sweep"):
        if not isinstance(mapping[key], str):
            _fail(f"{label}: {key} must be a path string")
    checks = mapping["integrity_checks_passed"]
    if not isinstance(checks, Mapping) or not checks or any(
        value is not True for value in checks.values()
    ):
        _fail(f"{label}: integrity checks must all be true")
    _validate_point(mapping["fixed_threshold_0_5"], f"{label}.fixed")
    budgets = _expect_exact_keys(
        mapping["budgets"], set(BUDGET_KEYS), f"{label}.budgets"
    )
    unavailable: list[dict[str, Any]] = []
    for budget in BUDGET_KEYS:
        point = budgets[budget]
        if point is None:
            if candidate:
                _fail(f"{label}: candidate budget {budget} is unavailable")
            assert method is not None
            usage = required_reference_usage.get((method, role_name, budget))
            if usage is not None:
                _fail(
                    f"{label}: budget {budget} is used by {usage} "
                    "but unavailable"
                )
            unavailable.append(
                {
                    "method": method,
                    "role": role_name,
                    "budget": budget,
                    "used_by_gates": False,
                    "gate_usage": "not_used_by_gates",
                }
            )
            continue
        _validate_point(point, f"{label}.budgets.{budget}")
        if float(point["fa"]) > float(budget) + 1e-15:
            _fail(
                f"{label}: actual Fa {point['fa']} exceeds budget {budget}"
            )
    return unavailable


def _validate_report_structure(
    report: dict[str, Any],
    *,
    module: ModuleType,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
) -> None:
    top_keys = {
        "schema",
        "generated_at_utc",
        "status",
        "scope",
        "candidate_runs",
        "reference_unavailable_points",
        "gate_evaluated",
        "engineering_gate_passed",
        "mainline_changed",
        "paper_core_established",
        "stability_claim_supported",
        "frozen_references",
        "engineering_gate",
        "validation",
        "decision",
        "decision_boundary",
    }
    _expect_exact_keys(report, top_keys, "report")
    if report["schema"] != module.SCHEMA:
        _fail("report schema is not the canonical summarizer schema")
    _parse_utc_timestamp(report["generated_at_utc"], "report.generated_at_utc")
    if report["status"] != "complete":
        _fail("incomplete reports cannot be published")
    if report["gate_evaluated"] is not True:
        _fail("engineering gate was not evaluated")
    if not isinstance(report["engineering_gate_passed"], bool):
        _fail("engineering_gate_passed must be boolean for a complete report")
    for key in (
        "mainline_changed",
        "paper_core_established",
        "stability_claim_supported",
    ):
        if report[key] is not False:
            _fail(f"report decision flag must remain false: {key}")

    scope = _expect_exact_keys(
        report["scope"],
        {
            "dataset",
            "candidate_variants",
            "model_seeds",
            "epochs",
            "split_seed",
            "official_test_accessed",
            "candidate_root",
            "formal_reference_root",
            "v2_reference_root",
            "reference_miou_root",
            "protocol",
        },
        "report.scope",
    )
    expected_scope = {
        "dataset": DATASET,
        "candidate_variants": list(VARIANTS),
        "model_seeds": list(SEEDS),
        "epochs": 800,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "candidate_root": str(candidate_root),
        "formal_reference_root": str(formal_root),
        "v2_reference_root": str(v2_root),
        "reference_miou_root": str(reference_miou_root),
        "protocol": str(
            (repo / "experiments/TPD_CLEAN_V3_PROTOCOL.md").resolve()
        ),
    }
    if dict(scope) != expected_scope:
        _fail("report scope does not exactly match the requested roots/protocol")

    expected_candidate_keys = {
        f"{variant}/seed_{seed}" for variant in VARIANTS for seed in SEEDS
    }
    candidates = _expect_exact_keys(
        report["candidate_runs"],
        expected_candidate_keys,
        "report.candidate_runs",
    )
    candidate_tuple_map: dict[tuple[str, int], Any] = {}
    validation_split_hashes: set[str] = set()
    for variant in VARIANTS:
        for seed in SEEDS:
            key = f"{variant}/seed_{seed}"
            record = _expect_exact_keys(
                candidates[key],
                {
                    "variant",
                    "seed",
                    "run_directory",
                    "roles",
                    "model",
                    "split_hashes",
                    "metrics_event_count",
                    "artifacts",
                },
                f"report.candidate_runs.{key}",
            )
            if record["variant"] != variant or record["seed"] != seed:
                _fail(f"{key}: variant/seed mismatch")
            expected_run = module._candidate_run_dir(
                candidate_root, variant, seed
            ).resolve()
            if Path(str(record["run_directory"])).resolve() != expected_run:
                _fail(f"{key}: run_directory mismatch")
            if record["metrics_event_count"] != 800:
                _fail(f"{key}: metrics_event_count must be 800")
            artifacts = _expect_exact_keys(
                record["artifacts"],
                {
                    "protocol.json",
                    "split.json",
                    "summary.json",
                    "metrics.jsonl",
                    "last.pth.tar",
                },
                f"{key}.artifacts",
            )
            if any(not _valid_sha256(value) for value in artifacts.values()):
                _fail(f"{key}: invalid common artifact digest")
            roles = _expect_exact_keys(
                record["roles"], set(ROLES), f"{key}.roles"
            )
            for role_name in ROLES:
                _validate_role(
                    roles[role_name],
                    label=f"{key}.{role_name}",
                    candidate=True,
                    required_reference_usage=module.REFERENCE_GATE_BUDGET_USAGE,
                    role_name=role_name,
                )
                validation_split_hashes.add(
                    roles[role_name]["validation_split_sha256"]
                )
            candidate_tuple_map[(variant, seed)] = record

    frozen = _expect_exact_keys(
        report["frozen_references"],
        set(REFERENCE_METHODS),
        "report.frozen_references",
    )
    unavailable: list[dict[str, Any]] = []
    for method in REFERENCE_METHODS:
        reference = _expect_exact_keys(
            frozen[method], {"seed", "roles"}, f"frozen.{method}"
        )
        if reference["seed"] != 42:
            _fail(f"frozen.{method}: seed must be 42")
        roles = _expect_exact_keys(
            reference["roles"], set(ROLES), f"frozen.{method}.roles"
        )
        for role_name in ROLES:
            unavailable.extend(
                _validate_role(
                    roles[role_name],
                    label=f"frozen.{method}.{role_name}",
                    candidate=False,
                    required_reference_usage=module.REFERENCE_GATE_BUDGET_USAGE,
                    method=method,
                    role_name=role_name,
                )
            )
            validation_split_hashes.add(
                roles[role_name]["validation_split_sha256"]
            )
    if len(validation_split_hashes) != 1:
        _fail(
            "candidate and frozen-reference roles do not share one "
            "validation split SHA-256"
        )
    if report["reference_unavailable_points"] != unavailable:
        _fail(
            "reference_unavailable_points is not the exact disclosure of "
            "unused frozen-reference null points"
        )
    for entry in unavailable:
        if entry["used_by_gates"] is not False:
            _fail("a frozen null point marked as gate-used cannot be published")

    validation = _expect_exact_keys(
        report["validation"],
        {
            "candidate_run_count",
            "candidate_checkpoint_count",
            "candidate_sweep_count",
            "candidate_metrics_event_count",
            "metrics_epochs_complete",
            "candidate_sweeps_integrity_checked",
            "frozen_reference_sweeps_integrity_checked",
            "paired_full_initialization_sha256",
            "paired_shared_non_shallow_initialization_sha256",
            "paired_split_fingerprint_equal",
        },
        "report.validation",
    )
    expected_counts = {
        "candidate_run_count": 4,
        "candidate_checkpoint_count": 8,
        "candidate_sweep_count": 8,
        "candidate_metrics_event_count": 3200,
    }
    for key, expected in expected_counts.items():
        if validation[key] != expected:
            _fail(f"report.validation.{key} must be {expected}")
    for key in (
        "metrics_epochs_complete",
        "candidate_sweeps_integrity_checked",
        "frozen_reference_sweeps_integrity_checked",
        "paired_split_fingerprint_equal",
    ):
        if validation[key] is not True:
            _fail(f"report.validation.{key} must be true")
    for key in (
        "paired_full_initialization_sha256",
        "paired_shared_non_shallow_initialization_sha256",
    ):
        pairs = _expect_exact_keys(
            validation[key], {"42", "3407"}, f"report.validation.{key}"
        )
        for seed, item in pairs.items():
            pair = _expect_exact_keys(
                item, {"equal", "sha256"}, f"{key}.{seed}"
            )
            if pair["equal"] is not True or not _valid_sha256(pair["sha256"]):
                _fail(f"{key}.{seed}: invalid paired initialization binding")

    engineering_gate = _expect_exact_keys(
        report["engineering_gate"],
        {"passed", "checks", "comparison_policy"},
        "report.engineering_gate",
    )
    checks = _expect_exact_keys(
        engineering_gate["checks"], set(GATE_NAMES), "engineering_gate.checks"
    )
    if any(
        not isinstance(check, Mapping)
        or not isinstance(check.get("passed"), bool)
        for check in checks.values()
    ):
        _fail("all seven engineering gates must contain boolean pass results")
    all_passed = all(bool(check["passed"]) for check in checks.values())
    if engineering_gate["passed"] is not all_passed:
        _fail("engineering gate aggregate is inconsistent with seven gates")
    if report["engineering_gate_passed"] is not all_passed:
        _fail("report engineering_gate_passed is inconsistent")
    expected_decision = (
        "ENGINEERING_GATE_PASS" if all_passed else "ENGINEERING_GATE_FAIL"
    )
    if report["decision"] != expected_decision:
        _fail("report decision is inconsistent with the engineering gate")
    recomputed_gate = module.evaluate_engineering_gate(
        candidate_tuple_map, frozen
    )
    if recomputed_gate != report["engineering_gate"]:
        _fail("reported seven-gate result differs from canonical recomputation")

    boundary = _expect_exact_keys(
        report["decision_boundary"],
        {
            "gate_only_controls_next_engineering_stage",
            "automatic_mainline_replacement",
            "requires_at_least_three_paired_seeds",
            "requires_more_datasets",
            "mainline_changed",
            "paper_core_established",
            "stability_claim_supported",
        },
        "report.decision_boundary",
    )
    expected_boundary = {
        "gate_only_controls_next_engineering_stage": True,
        "automatic_mainline_replacement": False,
        "requires_at_least_three_paired_seeds": True,
        "requires_more_datasets": True,
        "mainline_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }
    if dict(boundary) != expected_boundary:
        _fail("report decision boundary changed")


def _validate_canonical_report(
    *,
    staging_json: Path,
    staging_markdown: Path,
    module: ModuleType,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    report = _load_json(staging_json, "canonical report JSON")
    _validate_report_structure(
        report,
        module=module,
        repo=repo,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
    )
    _require_regular_file(staging_markdown, "canonical report Markdown")
    try:
        markdown_text = staging_markdown.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read canonical report Markdown: {exc}")
    canonical_markdown = module.render_markdown(report)
    if markdown_text != canonical_markdown:
        _fail("Markdown is not exactly canonical render_markdown(report)")

    try:
        current_report = module.build_report(
            candidate_root,
            formal_root,
            v2_root,
            reference_miou_root,
        )
    except Exception as exc:
        _fail(f"canonical current-artifact audit failed: {exc}")
    if not isinstance(current_report, dict):
        _fail("canonical current-artifact audit did not return an object")
    if current_report.get("status") != "complete":
        _fail(
            "current artifacts produce an incomplete canonical report: "
            + "; ".join(current_report.get("incomplete_reasons", []))
        )
    _validate_report_structure(
        current_report,
        module=module,
        repo=repo,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
    )
    staged_comparable = copy.deepcopy(report)
    current_comparable = copy.deepcopy(current_report)
    current_comparable["generated_at_utc"] = staged_comparable[
        "generated_at_utc"
    ]
    if current_comparable != staged_comparable:
        _fail("staged report differs from a current canonical rebuild")

    json_bytes = staging_json.read_bytes()
    markdown_bytes = staging_markdown.read_bytes()
    return report, json_bytes, markdown_bytes


def _descriptor(
    *,
    identifier: str,
    category: str,
    root_name: str,
    relative_path: str,
) -> dict[str, str]:
    return {
        "id": identifier,
        "category": category,
        "root": root_name,
        "relative_path": relative_path,
    }


def _validate_launch_and_log(
    *,
    launch_path: Path,
    log_path: Path,
    variant: str,
    seed: int,
    run_dir: Path,
    training_lock: Path,
) -> tuple[str, str]:
    launch = _load_json(launch_path, f"launch manifest {variant}/seed={seed}")
    expected = {
        "schema": "sctransnet_tpd_clean_v3_screen800_launch_v1",
        "variant": variant,
        "seed": seed,
        "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
        "gpu_name": "NVIDIA GeForce RTX 5090",
    }
    for key, value in expected.items():
        if launch.get(key) != value:
            _fail(
                f"launch manifest {variant}/seed={seed}: "
                f"{key}={launch.get(key)!r}, expected={value!r}"
            )
    if Path(str(launch.get("run_directory", ""))).resolve() != run_dir:
        _fail(f"launch manifest {variant}/seed={seed}: run directory mismatch")
    if Path(str(launch.get("source_lock", ""))).resolve() != training_lock:
        _fail(f"launch manifest {variant}/seed={seed}: source lock mismatch")
    if launch.get("source_lock_sha256") != _sha256_file(training_lock):
        _fail(
            f"launch manifest {variant}/seed={seed}: "
            "source-lock digest mismatch"
        )
    gpu_uuid = launch.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        _fail(f"launch manifest {variant}/seed={seed}: invalid GPU UUID")
    training_data_sha = launch.get("training_data_sha256")
    if not _valid_sha256(training_data_sha):
        _fail(
            f"launch manifest {variant}/seed={seed}: "
            "invalid training-data digest"
        )
    policy = launch.get("policy")
    required_policy = {
        "paired_variants": True,
        "pre_registered_seeds": [42, 3407],
        "fresh_run": True,
        "old_results_preserved": True,
        "shared_resource_screening": True,
        "efficiency_comparison_allowed": False,
        "official_test_accessed": False,
        "amp": False,
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != value for key, value in required_policy.items()
    ):
        _fail(f"launch manifest {variant}/seed={seed}: policy mismatch")

    _require_regular_file(log_path, f"worker log {variant}/seed={seed}")
    try:
        lines = [
            line
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        _fail(f"worker log {variant}/seed={seed}: cannot read: {exc}")
    pattern = re.compile(
        rf"^TPDCLEANV3_COMPLETE variant={re.escape(variant)} "
        rf"seed={seed} gpu_uuid=(?P<gpu_uuid>GPU-\S+) epochs=800$"
    )
    match = pattern.fullmatch(lines[-1]) if lines else None
    if match is None:
        _fail(
            f"worker log {variant}/seed={seed}: "
            "last non-empty line is not the registered completion record"
        )
    if match.group("gpu_uuid") != gpu_uuid:
        _fail(
            f"worker log {variant}/seed={seed}: completion GPU UUID "
            "does not match the launch manifest"
        )
    return gpu_uuid, training_data_sha


def _expected_input_descriptors(
    *,
    module: ModuleType,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
) -> list[dict[str, str]]:
    descriptors: list[dict[str, str]] = []
    training_lock = _relative_regular_file(
        repo, LOCK_SPECS["training"][0], "training source lock"
    )
    gpu_uuids: set[str] = set()
    data_hashes: set[str] = set()

    common_files = (
        "protocol.json",
        "split.json",
        "summary.json",
        "metrics.jsonl",
        "last.pth.tar",
    )
    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = module._candidate_run_dir(
                candidate_root, variant, seed
            ).resolve()
            if not _within(run_dir, candidate_root):
                _fail(f"candidate run escaped candidate root: {run_dir}")
            run_prefix = f"candidate/{variant}/seed_{seed}"
            for filename in common_files:
                path = run_dir / filename
                relative = _relative_to_root(
                    path, candidate_root, f"{run_prefix}/{filename}"
                )
                descriptors.append(
                    _descriptor(
                        identifier=f"{run_prefix}/{filename}",
                        category="candidate_run_file",
                        root_name="candidate",
                        relative_path=relative,
                    )
                )
            for role_name in ROLES:
                role_spec = module.ROLE_SPECS[role_name]
                for artifact_kind, filename, category in (
                    (
                        "checkpoint",
                        role_spec["checkpoint"],
                        "candidate_checkpoint",
                    ),
                    ("sweep", role_spec["sweep"], "candidate_sweep"),
                ):
                    path = run_dir / filename
                    relative = _relative_to_root(
                        path,
                        candidate_root,
                        f"{run_prefix}/{role_name}/{artifact_kind}",
                    )
                    descriptors.append(
                        _descriptor(
                            identifier=(
                                f"{run_prefix}/{role_name}/{artifact_kind}"
                            ),
                            category=category,
                            root_name="candidate",
                            relative_path=relative,
                        )
                    )
            launch_path = (
                candidate_root / "launch" / f"{variant}_seed{seed}.json"
            )
            log_path = (
                candidate_root / "logs" / f"{variant}_seed{seed}.log"
            )
            gpu_uuid, data_sha = _validate_launch_and_log(
                launch_path=launch_path,
                log_path=log_path,
                variant=variant,
                seed=seed,
                run_dir=run_dir,
                training_lock=training_lock,
            )
            gpu_uuids.add(gpu_uuid)
            data_hashes.add(data_sha)
            descriptors.append(
                _descriptor(
                    identifier=f"candidate/launch/{variant}/seed_{seed}",
                    category="candidate_launch_manifest",
                    root_name="candidate",
                    relative_path=_relative_to_root(
                        launch_path,
                        candidate_root,
                        f"launch manifest {variant}/seed={seed}",
                    ),
                )
            )
            descriptors.append(
                _descriptor(
                    identifier=f"candidate/log/{variant}/seed_{seed}",
                    category="candidate_worker_log",
                    root_name="candidate",
                    relative_path=_relative_to_root(
                        log_path,
                        candidate_root,
                        f"worker log {variant}/seed={seed}",
                    ),
                )
            )
    if len(gpu_uuids) != 4:
        _fail(f"launch manifests do not bind four distinct GPUs: {gpu_uuids}")
    if len(data_hashes) != 1:
        _fail("launch manifests do not share one training-data digest")

    reference_paths = module._reference_paths(
        formal_root, v2_root, reference_miou_root
    )
    if set(reference_paths) != set(REFERENCE_METHODS):
        _fail("canonical frozen-reference method set changed")
    for method in REFERENCE_METHODS:
        method_roles = reference_paths[method]
        if set(method_roles) != set(ROLES):
            _fail(f"frozen reference {method}: checkpoint role set changed")
        for role_name in ROLES:
            run_dir, _variant = method_roles[role_name]
            run_dir = Path(run_dir).resolve()
            if method in ("spd", "tpd_v1") and role_name == "pd_primary":
                root_name, expected_root = "formal", formal_root
            elif method in ("spd", "tpd_v1"):
                root_name, expected_root = "reference_miou", reference_miou_root
            else:
                root_name, expected_root = "v2", v2_root
            if not _within(run_dir, expected_root):
                _fail(
                    f"frozen reference {method}/{role_name} escaped "
                    f"{root_name} root"
                )
            role_spec = module.ROLE_SPECS[role_name]
            for artifact_kind, filename, category in (
                (
                    "checkpoint",
                    role_spec["checkpoint"],
                    "frozen_reference_checkpoint",
                ),
                ("sweep", role_spec["sweep"], "frozen_reference_sweep"),
            ):
                path = run_dir / filename
                descriptors.append(
                    _descriptor(
                        identifier=(
                            f"reference/{method}/{role_name}/{artifact_kind}"
                        ),
                        category=category,
                        root_name=root_name,
                        relative_path=_relative_to_root(
                            path,
                            expected_root,
                            f"frozen {method}/{role_name}/{artifact_kind}",
                        ),
                    )
                )

    descriptors.append(
        _descriptor(
            identifier="source/canonical_summarizer",
            category="canonical_summarizer",
            root_name="repo",
            relative_path=_relative_to_root(
                summarizer_path, repo, "canonical summarizer"
            ),
        )
    )
    lock_paths = {
        "training": training_lock,
        "postprocess": postprocess_lock_path.resolve(strict=True),
        "v2": _relative_regular_file(
            repo, LOCK_SPECS["v2"][0], "Clean-v2 source lock"
        ),
        "ner": _relative_regular_file(
            repo, LOCK_SPECS["ner"][0], "NER source lock"
        ),
    }
    for name, path in lock_paths.items():
        descriptors.append(
            _descriptor(
                identifier=f"lock/{name}",
                category="source_lock",
                root_name="repo",
                relative_path=_relative_to_root(
                    path, repo, f"{name} source lock"
                ),
            )
        )

    identifiers = [entry["id"] for entry in descriptors]
    if len(set(identifiers)) != len(identifiers):
        _fail("expected input descriptors contain duplicate identifiers")
    descriptors.sort(key=lambda entry: entry["id"])
    return descriptors


def _root_map(
    *,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
) -> dict[str, Path]:
    return {
        "repo": repo,
        "candidate": candidate_root,
        "formal": formal_root,
        "v2": v2_root,
        "reference_miou": reference_miou_root,
    }


def _materialize_inputs(
    descriptors: Sequence[Mapping[str, str]],
    roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    resolved_paths: set[Path] = set()
    for descriptor in descriptors:
        root_name = descriptor["root"]
        if root_name not in roots:
            _fail(f"unknown input root label: {root_name}")
        path = _relative_regular_file(
            roots[root_name],
            descriptor["relative_path"],
            f"completion input {descriptor['id']}",
        )
        if path in resolved_paths:
            _fail(f"completion input path is duplicated: {path}")
        resolved_paths.add(path)
        entries.append(
            {
                **dict(descriptor),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def _count_inputs(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for entry in entries:
        category = str(entry["category"])
        categories[category] = categories.get(category, 0) + 1
    counts = {
        "candidate_run_files": categories.get("candidate_run_file", 0)
        + categories.get("candidate_checkpoint", 0)
        + categories.get("candidate_sweep", 0),
        "candidate_launch_manifests": categories.get(
            "candidate_launch_manifest", 0
        ),
        "candidate_worker_logs": categories.get("candidate_worker_log", 0),
        "frozen_reference_checkpoints": categories.get(
            "frozen_reference_checkpoint", 0
        ),
        "frozen_reference_sweeps": categories.get(
            "frozen_reference_sweep", 0
        ),
        "canonical_summarizers": categories.get("canonical_summarizer", 0),
        "source_locks": categories.get("source_lock", 0),
        "total_files": len(entries),
    }
    return counts


def _validate_all_source_locks(
    *, repo: Path, postprocess_lock_path: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, (relative, schema) in LOCK_SPECS.items():
        path = _relative_regular_file(repo, relative, f"{name} source lock")
        _load_source_lock(
            path,
            repo=repo,
            expected_schema=schema,
            label=f"{name} source lock",
            validate_sources=True,
        )
        paths[name] = path
    postprocess = postprocess_lock_path.resolve(strict=True)
    paths["postprocess"] = postprocess
    return paths


def _manifest_payload(
    *,
    report: Mapping[str, Any],
    json_name: str,
    json_bytes: bytes,
    markdown_name: str,
    markdown_bytes: bytes,
    module: ModuleType,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    roots = _root_map(
        repo=repo,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
    )
    counts = _count_inputs(inputs)
    if counts != EXPECTED_INPUT_COUNTS:
        _fail(
            f"completion input counts differ: {counts} != "
            f"{EXPECTED_INPUT_COUNTS}"
        )
    summarizer_relative = _relative_to_root(
        summarizer_path, repo, "canonical summarizer"
    )
    postprocess_relative = _relative_to_root(
        postprocess_lock_path, repo, "postprocess source lock"
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "roots": {name: str(path) for name, path in roots.items()},
        "canonical_outputs": {
            "comparison_json": {
                "filename": json_name,
                "size_bytes": len(json_bytes),
                "sha256": _sha256_bytes(json_bytes),
            },
            "comparison_markdown": {
                "filename": markdown_name,
                "size_bytes": len(markdown_bytes),
                "sha256": _sha256_bytes(markdown_bytes),
            },
        },
        "source_binding": {
            "summarizer_relative_path": summarizer_relative,
            "summarizer_sha256": _sha256_file(summarizer_path),
            "postprocess_lock_relative_path": postprocess_relative,
            "postprocess_lock_sha256": _sha256_file(postprocess_lock_path),
            "postprocess_lock_schema": (
                "sctransnet_tpd_clean_v3_postprocess_source_lock_v1"
            ),
        },
        "report_binding": {
            "schema": module.SCHEMA,
            "generated_at_utc": report["generated_at_utc"],
            "status": "complete",
            "decision": report["decision"],
            "engineering_gate_passed": report["engineering_gate_passed"],
        },
        "validated_counts": {
            "candidate_runs": 4,
            "candidate_checkpoints": 8,
            "candidate_sweeps": 8,
            "candidate_metrics_events": 3200,
            "engineering_gates": 7,
            "frozen_reference_methods": 4,
            "frozen_reference_roles": 8,
        },
        "input_counts": counts,
        "inputs": inputs,
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        _fail(f"refusing to replace linked or non-regular output: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _marker_bytes(output_hashes: Mapping[str, str], names: Sequence[str]) -> bytes:
    return "".join(
        f"{output_hashes[name]}  {name}\n" for name in names
    ).encode("utf-8")


def _validate_marker(
    *,
    marker_path: Path,
    output_dir: Path,
    expected_names: Sequence[str],
) -> dict[str, str]:
    _require_regular_file(marker_path, "completion marker")
    try:
        lines = marker_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read completion marker: {exc}")
    if len(lines) != len(expected_names):
        _fail(
            f"completion marker must contain exactly "
            f"{len(expected_names)} rows"
        )
    pattern = re.compile(r"^([0-9a-f]{64})  ([^/]+)$")
    observed: dict[str, str] = {}
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            _fail(f"invalid completion marker row: {line!r}")
        name = match.group(2)
        if name in observed:
            _fail(f"duplicate completion marker output: {name}")
        observed[name] = match.group(1)
    if list(observed) != list(expected_names):
        _fail(
            f"completion marker output order/names differ: "
            f"{list(observed)} != {list(expected_names)}"
        )
    for name, expected_sha in observed.items():
        path = output_dir / name
        _require_regular_file(path, f"published output {name}")
        actual = _sha256_file(path)
        if actual != expected_sha:
            _fail(
                f"completion marker digest mismatch for {name}; "
                f"expected={expected_sha} actual={actual}"
            )
    return observed


def _resolve_configuration(
    *,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    resolved = {
        "repo": _resolve_directory(repo, "repository"),
        "candidate": _resolve_directory(candidate_root, "candidate root"),
        "formal": _resolve_directory(formal_root, "formal reference root"),
        "v2": _resolve_directory(v2_root, "Clean-v2 reference root"),
        "reference_miou": _resolve_directory(
            reference_miou_root, "reference-mIoU root"
        ),
    }
    if resolved["candidate"] in (
        resolved["formal"],
        resolved["v2"],
        resolved["reference_miou"],
    ):
        _fail("candidate root must differ from every frozen-reference root")
    for name in ("formal", "v2", "reference_miou"):
        other = resolved[name]
        if _within(resolved["candidate"], other) or _within(
            other, resolved["candidate"]
        ):
            _fail(f"candidate and {name} roots may not overlap")
    resolved["summarizer"] = Path(summarizer_path).resolve(strict=True)
    resolved["postprocess_lock"] = Path(postprocess_lock_path).resolve(
        strict=True
    )
    resolved["output"] = _safe_output_directory(
        output_dir, resolved["candidate"]
    )
    return resolved


def _validate_manifest(
    *,
    manifest: dict[str, Any],
    module: ModuleType,
    report: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
    roots: Mapping[str, Path],
    summarizer_path: Path,
    postprocess_lock_path: Path,
) -> None:
    top = _expect_exact_keys(
        manifest,
        {
            "schema",
            "created_at_utc",
            "roots",
            "canonical_outputs",
            "source_binding",
            "report_binding",
            "validated_counts",
            "input_counts",
            "inputs",
        },
        "completion manifest",
    )
    if top["schema"] != MANIFEST_SCHEMA:
        _fail("completion manifest schema mismatch")
    _parse_utc_timestamp(top["created_at_utc"], "manifest.created_at_utc")
    root_values = _expect_exact_keys(
        top["roots"],
        {"repo", "candidate", "formal", "v2", "reference_miou"},
        "manifest.roots",
    )
    expected_root_values = {
        name: str(roots[name])
        for name in ("repo", "candidate", "formal", "v2", "reference_miou")
    }
    if dict(root_values) != expected_root_values:
        _fail("completion manifest roots differ from requested roots")

    outputs = _expect_exact_keys(
        top["canonical_outputs"],
        {"comparison_json", "comparison_markdown"},
        "manifest.canonical_outputs",
    )
    expected_output_records = {
        "comparison_json": (
            module.JSON_OUTPUT_NAME,
            json_path,
        ),
        "comparison_markdown": (
            module.MARKDOWN_OUTPUT_NAME,
            markdown_path,
        ),
    }
    for key, (filename, path) in expected_output_records.items():
        record = _expect_exact_keys(
            outputs[key],
            {"filename", "size_bytes", "sha256"},
            f"manifest.canonical_outputs.{key}",
        )
        if (
            record["filename"] != filename
            or record["size_bytes"] != path.stat().st_size
            or record["sha256"] != _sha256_file(path)
        ):
            _fail(f"manifest canonical output binding changed: {key}")

    binding = _expect_exact_keys(
        top["source_binding"],
        {
            "summarizer_relative_path",
            "summarizer_sha256",
            "postprocess_lock_relative_path",
            "postprocess_lock_sha256",
            "postprocess_lock_schema",
        },
        "manifest.source_binding",
    )
    expected_binding = {
        "summarizer_relative_path": _relative_to_root(
            summarizer_path, roots["repo"], "canonical summarizer"
        ),
        "summarizer_sha256": _sha256_file(summarizer_path),
        "postprocess_lock_relative_path": _relative_to_root(
            postprocess_lock_path, roots["repo"], "postprocess source lock"
        ),
        "postprocess_lock_sha256": _sha256_file(postprocess_lock_path),
        "postprocess_lock_schema": (
            "sctransnet_tpd_clean_v3_postprocess_source_lock_v1"
        ),
    }
    if dict(binding) != expected_binding:
        _fail("completion manifest canonical source binding changed")

    report_binding = _expect_exact_keys(
        top["report_binding"],
        {
            "schema",
            "generated_at_utc",
            "status",
            "decision",
            "engineering_gate_passed",
        },
        "manifest.report_binding",
    )
    expected_report_binding = {
        "schema": module.SCHEMA,
        "generated_at_utc": report["generated_at_utc"],
        "status": "complete",
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
    }
    if dict(report_binding) != expected_report_binding:
        _fail("completion manifest report binding changed")
    expected_validated_counts = {
        "candidate_runs": 4,
        "candidate_checkpoints": 8,
        "candidate_sweeps": 8,
        "candidate_metrics_events": 3200,
        "engineering_gates": 7,
        "frozen_reference_methods": 4,
        "frozen_reference_roles": 8,
    }
    if top["validated_counts"] != expected_validated_counts:
        _fail("completion manifest validated counts changed")
    if top["input_counts"] != EXPECTED_INPUT_COUNTS:
        _fail("completion manifest input counts changed")

    expected_descriptors = _expected_input_descriptors(
        module=module,
        repo=roots["repo"],
        candidate_root=roots["candidate"],
        formal_root=roots["formal"],
        v2_root=roots["v2"],
        reference_miou_root=roots["reference_miou"],
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
    )
    inputs = top["inputs"]
    if not isinstance(inputs, list):
        _fail("completion manifest inputs must be a list")
    if len(inputs) != EXPECTED_INPUT_COUNTS["total_files"]:
        _fail("completion manifest has an unexpected number of inputs")
    if [entry.get("id") for entry in inputs] != [
        entry["id"] for entry in expected_descriptors
    ]:
        _fail("completion manifest input identifiers/order changed")
    current_inputs = _materialize_inputs(expected_descriptors, roots)
    if inputs != current_inputs:
        for observed, current in zip(inputs, current_inputs):
            if observed != current:
                _fail(
                    f"completion input changed: {current['id']}; "
                    f"published={observed!r} current={current!r}"
                )
        _fail("completion input set changed")
    if _count_inputs(inputs) != EXPECTED_INPUT_COUNTS:
        _fail("completion manifest input categories/counts changed")


def verify_completion(
    *,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    repo: Path = REPO_ROOT,
    summarizer_path: Path = DEFAULT_SUMMARIZER,
    postprocess_lock_path: Path = DEFAULT_POSTPROCESS_LOCK,
) -> dict[str, Any]:
    config = _resolve_configuration(
        repo=repo,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
        output_dir=output_dir,
    )
    module, _postprocess = _load_canonical_summarizer(
        repo=config["repo"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    _validate_all_source_locks(
        repo=config["repo"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    json_path = config["output"] / module.JSON_OUTPUT_NAME
    markdown_path = config["output"] / module.MARKDOWN_OUTPUT_NAME
    manifest_path = config["output"] / MANIFEST_NAME
    marker_path = config["output"] / MARKER_NAME
    expected_names = (
        module.JSON_OUTPUT_NAME,
        module.MARKDOWN_OUTPUT_NAME,
        MANIFEST_NAME,
    )
    marker_hashes = _validate_marker(
        marker_path=marker_path,
        output_dir=config["output"],
        expected_names=expected_names,
    )
    report, _json_bytes, _markdown_bytes = _validate_canonical_report(
        staging_json=json_path,
        staging_markdown=markdown_path,
        module=module,
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        v2_root=config["v2"],
        reference_miou_root=config["reference_miou"],
    )
    manifest = _load_json(manifest_path, "completion input manifest")
    roots = _root_map(
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        v2_root=config["v2"],
        reference_miou_root=config["reference_miou"],
    )
    _validate_manifest(
        manifest=manifest,
        module=module,
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        roots=roots,
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    if marker_hashes[MANIFEST_NAME] != _sha256_file(manifest_path):
        _fail("completion marker no longer binds the input manifest")
    return {
        "status": "verified",
        "output_dir": str(config["output"]),
        "marker": str(marker_path),
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
        "input_files": EXPECTED_INPUT_COUNTS["total_files"],
    }


def publish_completion(
    *,
    staging_dir: Path,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    repo: Path = REPO_ROOT,
    summarizer_path: Path = DEFAULT_SUMMARIZER,
    postprocess_lock_path: Path = DEFAULT_POSTPROCESS_LOCK,
) -> dict[str, Any]:
    config = _resolve_configuration(
        repo=repo,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
        output_dir=output_dir,
    )
    module, _postprocess = _load_canonical_summarizer(
        repo=config["repo"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    _validate_all_source_locks(
        repo=config["repo"],
        postprocess_lock_path=config["postprocess_lock"],
    )

    marker_path = config["output"] / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        verified = verify_completion(
            output_dir=config["output"],
            candidate_root=config["candidate"],
            formal_root=config["formal"],
            v2_root=config["v2"],
            reference_miou_root=config["reference_miou"],
            repo=config["repo"],
            summarizer_path=config["summarizer"],
            postprocess_lock_path=config["postprocess_lock"],
        )
        return {**verified, "status": "published", "reused": True}

    staging = _resolve_directory(staging_dir, "staging directory")
    staging_json = staging / module.JSON_OUTPUT_NAME
    staging_markdown = staging / module.MARKDOWN_OUTPUT_NAME
    report, json_bytes, markdown_bytes = _validate_canonical_report(
        staging_json=staging_json,
        staging_markdown=staging_markdown,
        module=module,
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        v2_root=config["v2"],
        reference_miou_root=config["reference_miou"],
    )
    descriptors = _expected_input_descriptors(
        module=module,
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        v2_root=config["v2"],
        reference_miou_root=config["reference_miou"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    roots = _root_map(
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        v2_root=config["v2"],
        reference_miou_root=config["reference_miou"],
    )
    inputs = _materialize_inputs(descriptors, roots)
    manifest = _manifest_payload(
        report=report,
        json_name=module.JSON_OUTPUT_NAME,
        json_bytes=json_bytes,
        markdown_name=module.MARKDOWN_OUTPUT_NAME,
        markdown_bytes=markdown_bytes,
        module=module,
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        v2_root=config["v2"],
        reference_miou_root=config["reference_miou"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
        inputs=inputs,
    )
    manifest_bytes = _canonical_json_bytes(manifest)

    # Detect input drift after the canonical rebuild and before publishing.
    if _materialize_inputs(descriptors, roots) != inputs:
        _fail("completion inputs changed while preparing publication")

    config["output"].mkdir(parents=True, exist_ok=True)
    json_path = config["output"] / module.JSON_OUTPUT_NAME
    markdown_path = config["output"] / module.MARKDOWN_OUTPUT_NAME
    manifest_path = config["output"] / MANIFEST_NAME
    _atomic_write_bytes(json_path, json_bytes)
    _atomic_write_bytes(markdown_path, markdown_bytes)
    _atomic_write_bytes(manifest_path, manifest_bytes)

    # The marker is deliberately the final write.  Recheck every bound input
    # and output immediately before exposing completion.
    if _materialize_inputs(descriptors, roots) != inputs:
        _fail("completion inputs changed before marker publication")
    output_hashes = {
        module.JSON_OUTPUT_NAME: _sha256_file(json_path),
        module.MARKDOWN_OUTPUT_NAME: _sha256_file(markdown_path),
        MANIFEST_NAME: _sha256_file(manifest_path),
    }
    expected_output_hashes = {
        module.JSON_OUTPUT_NAME: _sha256_bytes(json_bytes),
        module.MARKDOWN_OUTPUT_NAME: _sha256_bytes(markdown_bytes),
        MANIFEST_NAME: _sha256_bytes(manifest_bytes),
    }
    if output_hashes != expected_output_hashes:
        _fail("published outputs changed before marker publication")
    marker_names = (
        module.JSON_OUTPUT_NAME,
        module.MARKDOWN_OUTPUT_NAME,
        MANIFEST_NAME,
    )
    _atomic_write_bytes(
        marker_path, _marker_bytes(output_hashes, marker_names)
    )
    try:
        verified = verify_completion(
            output_dir=config["output"],
            candidate_root=config["candidate"],
            formal_root=config["formal"],
            v2_root=config["v2"],
            reference_miou_root=config["reference_miou"],
            repo=config["repo"],
            summarizer_path=config["summarizer"],
            postprocess_lock_path=config["postprocess_lock"],
        )
    except Exception:
        if marker_path.is_file() and not marker_path.is_symlink():
            marker_path.unlink()
        raise
    return {**verified, "status": "published", "reused": False}


def _add_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--formal-reference-root", type=Path, required=True)
    parser.add_argument("--v2-reference-root", type=Path, required=True)
    parser.add_argument("--reference-miou-root", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, default=DEFAULT_SUMMARIZER)
    parser.add_argument(
        "--postprocess-lock", type=Path, default=DEFAULT_POSTPROCESS_LOCK
    )
    parser.add_argument("--output-dir", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish or verify the exact hash-bound TPD-Clean-v3 "
            "screen800 completion bundle."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser(
        "publish", help="validate staged canonical outputs and publish marker-last"
    )
    _add_root_arguments(publish_parser)
    publish_parser.add_argument("--staging-dir", type=Path, required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="verify outputs and every current bound input"
    )
    _add_root_arguments(verify_parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        common = {
            "output_dir": args.output_dir,
            "candidate_root": args.candidate_root,
            "formal_root": args.formal_reference_root,
            "v2_root": args.v2_reference_root,
            "reference_miou_root": args.reference_miou_root,
            "repo": args.repo,
            "summarizer_path": args.summarizer,
            "postprocess_lock_path": args.postprocess_lock,
        }
        if args.command == "publish":
            result = publish_completion(
                staging_dir=args.staging_dir, **common
            )
        else:
            result = verify_completion(**common)
    except (CompletionValidationError, OSError, UnicodeError) as exc:
        print(f"TPDCLEANV3_COMPLETION_INVALID reason={exc}", file=sys.stderr)
        return 1
    print(
        "TPDCLEANV3_COMPLETION_OK"
        f" command={args.command}"
        f" status={result['status']}"
        f" decision={result['decision']}"
        f" gate={result['engineering_gate_passed']}"
        f" inputs={result['input_files']}"
        f" marker={result['marker']}"
        + (
            f" reused={str(result['reused']).lower()}"
            if "reused" in result
            else ""
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
