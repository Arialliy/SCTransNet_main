#!/usr/bin/env python3
"""Build the DCH formal800 Gate A--E comparison report.

This module owns no training or metric implementation.  It consumes the
strict DCH completion matrix, reuses the already-tested V6 gate equations
without changing a threshold, and emits a DCH-identified report.  Missing
formal artifacts remain missing: preflight is read-only and the write path
never fills, repairs, or replaces an experiment result.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    freeze_tpd_clean_v7_dch_source_locks as source_locks,
)
from experiments import summarize_tpd_clean_v6_formal800 as v6_core  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v7_dch_formal800_comparison_v1"
DATASET = "NUDT-SIRST"
VARIANTS = (
    "tpd_clean_v7_dch_full",
    "tpd_clean_v7_dch_capacity",
)
PRIMARY_VARIANT = VARIANTS[0]
CONTROL_VARIANT = VARIANTS[1]
SEEDS = (42, 3407)
RUN_TAG = "formal800_exact_fp32_2x5090_v1"
EXPECTED_EPOCHS = 800
EXPECTED_RUNS = 4
EXPECTED_CHECKPOINTS = 12
EXPECTED_SWEEPS = 8
EXPECTED_TARGET_COUNT = 189
VALIDATION_FIELDS = tuple(source_locks.VALIDATION_FIELDS)
ROLE_SPECS = v6_core.ROLE_SPECS
BUDGET_KEYS = v6_core.BUDGET_KEYS
LAST_FLOAT32_BELOW_ONE = v6_core.LAST_FLOAT32_BELOW_ONE

INTEGRITY_KEYS = (
    "four_runs_contiguous_800_epochs",
    "twelve_checkpoints_present_and_strict_load",
    "eight_closed_interval_sweeps",
    "model_split_protocol_evaluator_hashes_consistent",
    "cpu_and_rtx5090_smoke_passed",
    "fixed_threshold_reproduction_exact",
    "all_five_budgets_available",
    "preregistered_endpoint_provenance",
    "exact_epoch_journals_complete",
    "worker_logs_complete_gpu_mapped",
    "native_17_fields_complete",
)
GATE_CHECK_KEYS = (
    "gate_a_seed42_fixed_threshold",
    "gate_b_seed42_budget_and_spd",
    "gate_c_seed3407_stability",
    "gate_d_full_vs_capacity",
    "gate_e_engineering_integrity",
)
GATE_D_POINT_LABELS = tuple(
    label
    for role_name in ROLE_SPECS
    for label in (
        f"{role_name}.fixed_threshold_0_5",
        *(f"{role_name}.budget.{budget}" for budget in BUDGET_KEYS),
    )
)

DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CANDIDATE_ROOT / DATASET / "comparison"
DEFAULT_EVALUATOR = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v7_dch_pd_fa.py"
)
DEFAULT_ACCEPTANCE_SOURCE_LOCK = (
    REPO_ROOT
    / source_locks.DEFAULT_LOCK_RELATIVES["acceptance"]
)
DEFAULT_TRAINING_SOURCE_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v7_dch_exact_source_lock.json"
)
DEFAULT_SMOKE_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_clean_v7_dch_preflight_v1/smoke_reports"
)
JSON_OUTPUT_NAME = "tpd_clean_v7_dch_formal800_comparison.json"
MARKDOWN_OUTPUT_NAME = "tpd_clean_v7_dch_formal800_comparison.md"


class IncompleteArtifact(ValueError):
    """A required DCH formal artifact is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteArtifact(message)


def sha256_file(path: Path) -> str:
    path = Path(path)
    _require(
        path.is_file() and not path.is_symlink(),
        f"not a regular file: {path}",
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_key(variant: str, seed: int) -> str:
    return f"{variant}/seed_{seed}"


def _normalize_runs(
    value: Any,
) -> tuple[Dict[tuple[str, int], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    _require(isinstance(value, Mapping), "completion matrix runs are missing")
    tuple_runs: Dict[tuple[str, int], Dict[str, Any]] = {}
    keyed_runs: Dict[str, Dict[str, Any]] = {}
    for raw_key, raw_record in value.items():
        _require(isinstance(raw_record, Mapping), f"invalid run: {raw_key}")
        record = copy.deepcopy(dict(raw_record))
        variant = record.get("variant")
        seed = record.get("seed")
        if variant is None or seed is None:
            key_text = str(raw_key)
            for candidate in VARIANTS:
                prefix = f"{candidate}/seed_"
                if key_text.startswith(prefix):
                    variant = candidate
                    try:
                        seed = int(key_text.removeprefix(prefix))
                    except ValueError as exc:
                        raise IncompleteArtifact(
                            f"invalid run key: {raw_key}"
                        ) from exc
                    break
        _require(variant in VARIANTS, f"unexpected run variant: {variant}")
        _require(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed in SEEDS,
            f"unexpected run seed: {seed}",
        )
        record["variant"] = variant
        record["seed"] = seed
        tuple_key = (str(variant), int(seed))
        _require(tuple_key not in tuple_runs, f"duplicate run: {tuple_key}")
        tuple_runs[tuple_key] = record
        keyed_runs[_run_key(*tuple_key)] = record
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    _require(set(tuple_runs) == expected, "DCH four-run matrix differs")
    return tuple_runs, keyed_runs


def inspect_training_readiness(
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
) -> Dict[str, Any]:
    """Read the current matrix status without evaluating a performance gate."""

    from experiments import validate_tpd_clean_v7_dch_formal800_completion as completion

    inspected = completion.inspect_completion_matrix(
        candidate_root=Path(candidate_root)
    )
    # ``formal_matrix_complete`` covers training only; postprocess becomes
    # ready only after all eight sweep files also exist.
    ready = inspected.get("ready") is True
    return {
        "schema": (
            "sctransnet_tpd_clean_v7_dch_postprocess_preflight_v1"
        ),
        "mode": "preflight",
        "candidate_root": str(Path(candidate_root).resolve()),
        "formal_matrix_complete": ready,
        "gate_evaluated": False,
        "engineering_gate_passed": None,
        "ner_stage_authorized": False,
        "completion_inspection": inspected,
    }


def evaluate_engineering_gates(
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    spd_reference: Mapping[str, Any],
    engineering_integrity: Mapping[str, bool],
) -> Dict[str, Any]:
    """Evaluate the unchanged V6 Gate A--E equations under DCH identity."""

    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    _require(set(runs) == expected, "DCH gate run matrix differs")
    _require(
        set(engineering_integrity) == set(INTEGRITY_KEYS),
        "DCH Gate E integrity key set differs",
    )
    _require(
        all(isinstance(value, bool) for value in engineering_integrity.values()),
        "DCH Gate E integrity values must be boolean",
    )
    remapped = {
        (v6_core.PRIMARY_VARIANT, seed): runs[(PRIMARY_VARIANT, seed)]
        for seed in SEEDS
    }
    remapped.update(
        {
            (v6_core.CONTROL_VARIANT, seed): runs[(CONTROL_VARIANT, seed)]
            for seed in SEEDS
        }
    )
    v6_integrity = {
        key: bool(engineering_integrity[key])
        for key in v6_core.INTEGRITY_KEYS
    }
    result = copy.deepcopy(
        v6_core.evaluate_engineering_gates(
            remapped,
            spd_reference,
            v6_integrity,
        )
    )
    gate_e = {
        "passed": all(engineering_integrity.values()),
        "subchecks": dict(engineering_integrity),
    }
    result["checks"]["gate_e_engineering_integrity"] = gate_e
    result["passed"] = all(
        bool(item["passed"]) for item in result["checks"].values()
    )
    result["protocol"] = (
        "experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md section 7"
    )
    result["candidate_family"] = "tpd_clean_v7_dch"
    result["thresholds_inherited_without_change"] = True
    return result


def _integrity_from_matrix(
    matrix: Mapping[str, Any],
    *,
    smoke_passed: bool,
) -> Dict[str, bool]:
    observed = matrix.get("integrity")
    _require(isinstance(observed, Mapping), "matrix integrity audit missing")
    values: Dict[str, bool] = {}
    for key in INTEGRITY_KEYS:
        if key == "cpu_and_rtx5090_smoke_passed":
            values[key] = smoke_passed
            continue
        _require(key in observed, f"matrix integrity field missing: {key}")
        _require(
            isinstance(observed[key], bool),
            f"matrix integrity field is not boolean: {key}",
        )
        values[key] = bool(observed[key])
    return values


def _validation_split_sha256(
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
) -> str:
    digests = set()
    for key, record in runs.items():
        value = record.get("validation_split_sha256")
        if value is None and isinstance(record.get("split_hashes"), Mapping):
            value = record["split_hashes"].get("used_val_sha256")
        _require(
            isinstance(value, str) and len(value) == 64,
            f"{key}: validation split SHA-256 missing",
        )
        digests.add(value)
    _require(len(digests) == 1, "DCH validation splits differ across runs")
    return next(iter(digests))


def build_report_from_components(
    *,
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    keyed_runs: Mapping[str, Mapping[str, Any]],
    spd_reference: Mapping[str, Any],
    engineering_integrity: Mapping[str, bool],
    bindings: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pure report construction used by strict runtime and unit fixtures."""

    gates = evaluate_engineering_gates(
        runs,
        spd_reference,
        engineering_integrity,
    )
    passed = bool(gates["passed"])
    return {
        "schema": SCHEMA,
        "status": "complete",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gate_evaluated": True,
        "candidate_family": "tpd_clean_v7_dch",
        "dataset": DATASET,
        "official_test_accessed": False,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "run_tag": RUN_TAG,
        "formal_artifact_counts": {
            "runs": EXPECTED_RUNS,
            "checkpoints": EXPECTED_CHECKPOINTS,
            "sweeps": EXPECTED_SWEEPS,
        },
        "validation_fields": list(VALIDATION_FIELDS),
        "candidate_runs": copy.deepcopy(dict(keyed_runs)),
        "frozen_spd_reference": copy.deepcopy(dict(spd_reference)),
        "engineering_integrity": dict(engineering_integrity),
        "engineering_gate": gates,
        "engineering_gate_passed": passed,
        "ner_stage_authorized": passed,
        "decision": (
            "ENGINEERING_GATE_PASS"
            if passed
            else "ENGINEERING_GATE_FAIL"
        ),
        "mainline_contract": "Keep-Context-Saliency",
        "mainline_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "fragmentation_mechanism_claim_supported": None,
        "mechanism_audit_replaces_performance_gates": False,
        "bindings": copy.deepcopy(dict(bindings)),
    }


def build_report(
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    *,
    matrix: Mapping[str, Any] | None = None,
    evaluator_path: Path = DEFAULT_EVALUATOR,
    acceptance_source_lock: Path = DEFAULT_ACCEPTANCE_SOURCE_LOCK,
    training_source_lock: Path = DEFAULT_TRAINING_SOURCE_LOCK,
    smoke_root: Path = DEFAULT_SMOKE_ROOT,
) -> Dict[str, Any]:
    """Strictly validate all formal inputs and derive one DCH report."""

    from experiments import validate_tpd_clean_v7_dch_formal800_completion as completion
    from experiments import verify_tpd_clean_v7_dch_smoke_reports as smoke

    evaluator_path = Path(evaluator_path).resolve()
    acceptance_source_lock = Path(acceptance_source_lock).resolve()
    training_source_lock = Path(training_source_lock).resolve()
    lock_payload, lock_sha256 = source_locks.validate_source_lock(
        "acceptance",
        acceptance_source_lock,
        repo_root=REPO_ROOT,
    )
    training_lock_payload, training_lock_sha256 = (
        source_locks.validate_source_lock(
            "training",
            training_source_lock,
            repo_root=REPO_ROOT,
        )
    )
    matrix_payload = (
        copy.deepcopy(dict(matrix))
        if matrix is not None
        else completion.validate_completion_matrix(
            candidate_root=Path(candidate_root),
            evaluator_path=evaluator_path,
            acceptance_lock_path=acceptance_source_lock,
        )
    )
    _require(
        matrix_payload.get("status") == "complete"
        and matrix_payload.get("ready") is True,
        "DCH completion matrix is not complete",
    )
    _require(
        matrix_payload.get("candidate_family") == "tpd_clean_v7_dch"
        and matrix_payload.get("run_count") == EXPECTED_RUNS
        and matrix_payload.get("checkpoint_count") == EXPECTED_CHECKPOINTS
        and matrix_payload.get("sweep_count") == EXPECTED_SWEEPS
        and matrix_payload.get("validation_fields")
        == list(VALIDATION_FIELDS),
        "DCH completion matrix identity/counts differ",
    )
    tuple_runs, keyed_runs = _normalize_runs(matrix_payload.get("runs"))
    validation_split = _validation_split_sha256(tuple_runs)
    spd_reference = v6_core.load_spd_reference(validation_split)
    smoke_result = smoke.validate_smoke_reports(Path(smoke_root))
    smoke_passed = bool(
        smoke_result.get("status") == "complete"
        and smoke_result.get("passed") is True
        and smoke_result.get("physical_gpu_reports_verified")
        == {
            "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
            "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
        }
    )
    integrity = _integrity_from_matrix(
        matrix_payload,
        smoke_passed=smoke_passed,
    )
    bindings = {
        "candidate_root": str(Path(candidate_root).resolve()),
        "acceptance_source_lock": str(acceptance_source_lock),
        "acceptance_source_lock_schema": lock_payload["schema"],
        "acceptance_source_lock_sha256": lock_sha256,
        "acceptance_source_count": lock_payload["source_count"],
        "training_source_lock": str(training_source_lock),
        "training_source_lock_sha256": training_lock_sha256,
        "training_source_count": training_lock_payload["source_count"],
        "evaluator": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "training_data_sha256": training_lock_payload[
            "training_data_sha256"
        ],
        "validation_split_sha256": validation_split,
        "smoke_root": str(Path(smoke_root).resolve()),
        "smoke_report_sha256": dict(smoke_result["report_sha256"]),
        "completion_matrix_sha256": _canonical_sha256(matrix_payload),
    }
    return build_report_from_components(
        runs=tuple_runs,
        keyed_runs=keyed_runs,
        spd_reference=spd_reference,
        engineering_integrity=integrity,
        bindings=bindings,
    )


def _point_text(point: Mapping[str, Any]) -> str:
    return (
        f"{int(point['matched_target_count'])}/{int(point['target_count'])}"
        f" | {float(point['fa']):.9g}"
        f" | {float(point['miou']):.9f}"
        f" | {float(point['threshold']):.9g}"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the complete comparison without changing any reported value."""

    runs = report["candidate_runs"]
    gate = report["engineering_gate"]
    _require(
        set(gate["checks"]) == set(GATE_CHECK_KEYS),
        "Gate A--E check set differs",
    )
    lines = [
        "# TPD-Clean V7-DCH formal800 comparison",
        "",
        f"- Decision: `{report['decision']}`",
        (
            "- Engineering gate passed: "
            f"`{str(report['engineering_gate_passed']).lower()}`"
        ),
        (
            "- NER stage authorized: "
            f"`{str(report['ner_stage_authorized']).lower()}`"
        ),
        "- Mainline: `Keep-Context-Saliency` (unchanged)",
        "- Evidence: 4 runs / 12 checkpoints / 8 sweeps",
        "- Official test accessed: `false`",
        "",
        "## Fixed-threshold operating points",
        "",
        "| Variant | Seed | Role | Pd count | Fa | mIoU | Threshold |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        for variant in VARIANTS:
            record = runs[_run_key(variant, seed)]
            for role_name in ROLE_SPECS:
                point = record["roles"][role_name]["fixed_threshold_0_5"]
                lines.append(
                    f"| {variant} | {seed} | {role_name} | "
                    f"{_point_text(point)} |"
                )
    lines.extend(
        [
            "",
            "## Registered Fa-budget operating points (40)",
            "",
            "| Variant | Seed | Role | Budget | Pd count | Fa | mIoU | Threshold |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in SEEDS:
        for variant in VARIANTS:
            record = runs[_run_key(variant, seed)]
            for role_name in ROLE_SPECS:
                for budget in BUDGET_KEYS:
                    point = record["roles"][role_name]["budgets"][budget]
                    lines.append(
                        f"| {variant} | {seed} | {role_name} | {budget} | "
                        f"{_point_text(point)} |"
                    )
    lines.extend(["", "## Gate A–E", ""])
    for name in GATE_CHECK_KEYS:
        value = gate["checks"][name]
        lines.append(f"- `{name}`: `{str(value['passed']).lower()}`")
    gate_d = gate["checks"]["gate_d_full_vs_capacity"]
    lines.extend(
        [
            "",
            "## Gate D — Full versus Capacity (24 comparisons)",
            "",
            "| Seed | Point | Capacity covers Full | Full covers Capacity |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for seed in SEEDS:
        comparisons = gate_d["per_seed"][str(seed)]["comparisons"]
        _require(
            set(comparisons) == set(GATE_D_POINT_LABELS),
            f"Gate D comparison set differs for seed {seed}",
        )
        for label in GATE_D_POINT_LABELS:
            record = comparisons[label]
            lines.append(
                f"| {seed} | {label} | "
                f"{str(record['capacity_strictly_covers_full']).lower()} | "
                f"{str(record['full_strictly_covers_capacity']).lower()} |"
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Gate A–E controls only the next NER engineering stage.",
            "- Mechanism Audit M is independent and cannot replace Gate A–E.",
            "- This tokenizer-only report does not establish the paper core or "
            "cross-randomness stability claim.",
            "",
            "## Bindings",
            "",
            "```json",
            json.dumps(
                report["bindings"],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_new(path: Path, content: bytes) -> Path:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite DCH report: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NotADirectoryError(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def write_report_once(
    report: Mapping[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir).absolute()
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise NotADirectoryError(output_dir)
    json_path = output_dir / JSON_OUTPUT_NAME
    markdown_path = output_dir / MARKDOWN_OUTPUT_NAME
    if any(
        path.exists() or path.is_symlink()
        for path in (json_path, markdown_path)
    ):
        raise FileExistsError("refusing to overwrite a DCH formal report")
    write_new(json_path, _json_bytes(report))
    try:
        write_new(
            markdown_path,
            render_markdown(report).encode("utf-8"),
        )
    except BaseException:
        json_path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DCH formal800 and evaluate Gates A--E"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        print(
            json.dumps(
                inspect_training_readiness(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return
    readiness = inspect_training_readiness()
    if readiness["formal_matrix_complete"] is not True:
        raise SystemExit(
            "DCH formal800 matrix is incomplete; only --preflight is allowed"
        )
    report = build_report()
    paths = write_report_once(report)
    print(
        f"WROTE decision={report['decision']} "
        f"json={paths[0]} markdown={paths[1]}",
        flush=True,
    )


__all__ = [
    "BUDGET_KEYS",
    "CONTROL_VARIANT",
    "DEFAULT_ACCEPTANCE_SOURCE_LOCK",
    "DEFAULT_CANDIDATE_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TRAINING_SOURCE_LOCK",
    "EXPECTED_CHECKPOINTS",
    "EXPECTED_EPOCHS",
    "EXPECTED_RUNS",
    "EXPECTED_SWEEPS",
    "GATE_CHECK_KEYS",
    "GATE_D_POINT_LABELS",
    "INTEGRITY_KEYS",
    "IncompleteArtifact",
    "JSON_OUTPUT_NAME",
    "LAST_FLOAT32_BELOW_ONE",
    "MARKDOWN_OUTPUT_NAME",
    "PRIMARY_VARIANT",
    "ROLE_SPECS",
    "SEEDS",
    "VALIDATION_FIELDS",
    "VARIANTS",
    "build_report",
    "build_report_from_components",
    "evaluate_engineering_gates",
    "inspect_training_readiness",
    "main",
    "parse_args",
    "render_markdown",
    "sha256_file",
    "write_new",
    "write_report_once",
]


if __name__ == "__main__":
    main()
