#!/usr/bin/env python3
"""Aggregate the six PBDR-V1 zero-training roles and apply Trigger T1--T5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_three_dataset_pbdr_zero_training_v1 as analyzer  # noqa: E402
from analysis import compare_three_dataset_dorf_v1 as dorf_compare  # noqa: E402


SCHEMA = "sctransnet_three_dataset_pbdr_zero_training_comparison_v1/v1"
DATASETS = analyzer.DATASETS
CHECKPOINT_ROLES = analyzer.CHECKPOINT_ROLES
PRIMARY_ROLE = "best_miou"
CURRENT_MODE = analyzer.CURRENT_MODE
ORACLE_MODE = analyzer.ORACLE_MODE
AUTHORIZATION_G_EIGHTHS = analyzer.AUTHORIZATION_G_EIGHTHS
FLOAT_EQ_ATOL = analyzer.FLOAT_EQ_ATOL
FLOAT_EQ_RTOL = analyzer.FLOAT_EQ_RTOL

DECISION_PASS = "PBDR_ZERO_TRAINING_TRIGGER_PASSED"
DECISION_FAIL = "PBDR_GLOBAL_FIXED_G_SCREEN_FAILED"
DEFAULT_INPUT_ROOT = analyzer.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison/seed42_six_role"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_sha256() -> dict[str, str]:
    sources = {
        "analysis/compare_three_dataset_pbdr_zero_training_v1.py": Path(__file__),
        "analysis/analyze_three_dataset_pbdr_zero_training_v1.py": Path(
            analyzer.__file__
        ),
        "analysis/compare_three_dataset_dorf_v1.py": Path(dorf_compare.__file__),
    }
    return {
        relative: file_sha256(path.resolve(strict=True))
        for relative, path in sorted(sources.items())
    }


def role_key(dataset: str, checkpoint_role: str) -> str:
    _require(dataset in DATASETS, "dataset differs")
    _require(checkpoint_role in CHECKPOINT_ROLES, "role differs")
    return f"{dataset}::{checkpoint_role}"


def expected_role_keys() -> tuple[str, ...]:
    return tuple(
        role_key(dataset, checkpoint_role)
        for dataset in DATASETS
        for checkpoint_role in CHECKPOINT_ROLES
    )


def default_input_path(dataset: str, checkpoint_role: str) -> Path:
    return (
        DEFAULT_INPUT_ROOT
        / "runs"
        / dataset
        / f"final_tss_off_{checkpoint_role}_seed42"
        / "evaluation.json"
    )


def load_role_payloads(
    input_root: Path = DEFAULT_INPUT_ROOT,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    root = Path(input_root).resolve(strict=True)
    payloads: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for checkpoint_role in CHECKPOINT_ROLES:
            relative = (
                Path("runs")
                / dataset
                / f"final_tss_off_{checkpoint_role}_seed42"
                / "evaluation.json"
            )
            path = (root / relative).resolve(strict=True)
            _require(path.is_relative_to(root), "input path escapes root")
            _require(path.is_file() and not path.is_symlink(), f"missing input: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            _require(isinstance(payload, dict), f"input is not an object: {path}")
            serialized_modes = payload.get("modes")
            _require(
                isinstance(serialized_modes, Mapping)
                and set(serialized_modes) == set(analyzer.MODE_ORDER),
                "serialized mode set differs",
            )
            # Role artifacts use canonical sort_keys JSON.  Restore the frozen
            # semantic order in memory before calling the analyzer validator,
            # which intentionally checks that order as well as point equality.
            payload["modes"] = {
                mode: serialized_modes[mode] for mode in analyzer.MODE_ORDER
            }
            analyzer.validate_output_payload(payload)
            _require(payload.get("dataset") == dataset, "input dataset differs")
            _require(
                payload.get("checkpoint_role") == checkpoint_role,
                "input role differs",
            )
            key = role_key(dataset, checkpoint_role)
            _require(key not in payloads, f"duplicate input: {key}")
            payloads[key] = payload
            artifacts.append(
                {
                    "dataset": dataset,
                    "checkpoint_role": checkpoint_role,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
    _require(tuple(payloads) == expected_role_keys(), "input matrix order differs")
    return payloads, artifacts


def _mode(payload: Mapping[str, Any], g_eighths: int) -> Mapping[str, Any]:
    mode_name = analyzer.MODE_BY_G_EIGHTHS[g_eighths]
    modes = payload.get("modes")
    _require(isinstance(modes, Mapping), "payload modes are missing")
    point = modes.get(mode_name)
    _require(isinstance(point, Mapping), f"payload lacks {mode_name}")
    _require(point.get("g_eighths") == g_eighths, "point gate differs")
    return point


def _metrics(payload: Mapping[str, Any], g_eighths: int) -> Mapping[str, Any]:
    point = _mode(payload, g_eighths)
    metrics = point.get("fixed_threshold_0_5")
    _require(isinstance(metrics, Mapping), "point metrics are missing")
    return metrics


def float_not_lower(candidate: float, reference: float) -> bool:
    return float(candidate) >= float(reference) - FLOAT_EQ_ATOL


def float_strict_higher(candidate: float, reference: float) -> bool:
    return float(candidate) > float(reference) + FLOAT_EQ_ATOL


def evaluate_t1_dataset(
    candidate: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    nondegradation = {
        "matched_target_not_lower": int(candidate["matched_target_count"])
        >= int(current["matched_target_count"]),
        "unmatched_predicted_pixels_not_higher": int(
            candidate["unmatched_predicted_pixels"]
        )
        <= int(current["unmatched_predicted_pixels"]),
        "miou_not_lower": float_not_lower(candidate["miou"], current["miou"]),
        "niou_not_lower": float_not_lower(candidate["niou"], current["niou"]),
        "matched_tiny_not_lower": int(candidate["matched_tiny_target_count"])
        >= int(current["matched_tiny_target_count"]),
    }
    strict_core = {
        "matched_target_higher": int(candidate["matched_target_count"])
        > int(current["matched_target_count"]),
        "unmatched_predicted_pixels_lower": int(
            candidate["unmatched_predicted_pixels"]
        )
        < int(current["unmatched_predicted_pixels"]),
        "miou_higher": float_strict_higher(candidate["miou"], current["miou"]),
        "niou_higher": float_strict_higher(candidate["niou"], current["niou"]),
    }
    strict_count = sum(strict_core.values())
    passed = all(nondegradation.values()) and strict_count >= 2
    return {
        "nondegradation": nondegradation,
        "strict_core_improvements": strict_core,
        "strict_core_improvement_count": strict_count,
        "pass": passed,
        "delta": {
            "matched_target_count": int(candidate["matched_target_count"])
            - int(current["matched_target_count"]),
            "unmatched_predicted_pixels": int(
                candidate["unmatched_predicted_pixels"]
            )
            - int(current["unmatched_predicted_pixels"]),
            "miou": float(candidate["miou"]) - float(current["miou"]),
            "niou": float(candidate["niou"]) - float(current["niou"]),
            "matched_tiny_target_count": int(
                candidate["matched_tiny_target_count"]
            )
            - int(current["matched_tiny_target_count"]),
        },
    }


def evaluate_t1(
    payloads: Mapping[str, Mapping[str, Any]],
    g_eighths: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        payload = payloads[role_key(dataset, PRIMARY_ROLE)]
        result = evaluate_t1_dataset(
            _metrics(payload, g_eighths),
            _metrics(payload, analyzer.IDENTITY_G_EIGHTHS),
        )
        rows.append({"dataset": dataset, **result})
    passed_count = sum(bool(row["pass"]) for row in rows)
    return {
        "role": PRIMARY_ROLE,
        "required_dataset_count": 2,
        "passed_dataset_count": passed_count,
        "dataset_rows": rows,
        "pass": passed_count >= 2,
    }


def evaluate_t2(
    payloads: Mapping[str, Mapping[str, Any]],
    g_eighths: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    severe_condition_count = 0
    for dataset in DATASETS:
        for checkpoint_role in CHECKPOINT_ROLES:
            payload = payloads[role_key(dataset, checkpoint_role)]
            direction = dorf_compare.compare_direction(
                _metrics(payload, g_eighths),
                _metrics(payload, analyzer.IDENTITY_G_EIGHTHS),
            )
            conditions = direction["severe_degradation_conditions"]
            true_conditions = [name for name, value in conditions.items() if value]
            severe_condition_count += len(true_conditions)
            rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_role": checkpoint_role,
                    "severe": bool(direction["severe_degradation"]),
                    "true_severe_conditions": true_conditions,
                    "direction": direction,
                }
            )
    severe_role_count = sum(bool(row["severe"]) for row in rows)
    return {
        "role_count": len(rows),
        "severe_role_count": severe_role_count,
        "severe_condition_count": severe_condition_count,
        "role_rows": rows,
        "pass": severe_role_count == 0,
    }


def evaluate_t3(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    field = "missed_gt_objects_with_protected_rescue_pixels"
    for dataset in DATASETS:
        payload = payloads[role_key(dataset, PRIMARY_ROLE)]
        signals = payload["signals"]["target_rescue"]
        value = int(signals[field])
        rows.append({"dataset": dataset, "value": value, "pass": value > 0})
    passed = sum(bool(row["pass"]) for row in rows)
    return {
        "role": PRIMARY_ROLE,
        "field": field,
        "dataset_rows": rows,
        "passed_dataset_count": passed,
        "pass": passed == len(DATASETS),
    }


def evaluate_t4(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    field = "unmatched_fp_pixels_with_unprotected_suppression"
    for dataset in DATASETS:
        payload = payloads[role_key(dataset, PRIMARY_ROLE)]
        signals = payload["signals"]["background_suppression"]
        value = int(signals[field])
        rows.append({"dataset": dataset, "value": value, "pass": value > 0})
    passed = sum(bool(row["pass"]) for row in rows)
    return {
        "role": PRIMARY_ROLE,
        "field": field,
        "dataset_rows": rows,
        "passed_dataset_count": passed,
        "pass": passed == len(DATASETS),
    }


def evaluate_t5(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for checkpoint_role in CHECKPOINT_ROLES:
            protection = payloads[role_key(dataset, checkpoint_role)]["protection"]
            protected = int(protection["protected_background_pixel_count"])
            background = int(protection["background_pixel_count"])
            _require(background > 0, "T5 background denominator is zero")
            passed = 2 * protected < background
            fraction = float(protection["protected_background_fraction"])
            _require(
                math.isclose(
                    fraction,
                    protected / background,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "T5 fraction/count identity differs",
            )
            rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_role": checkpoint_role,
                    "protected_background_pixel_count": protected,
                    "background_pixel_count": background,
                    "protected_background_fraction": fraction,
                    "integer_rule": "2*protected_background < background",
                    "pass": passed,
                }
            )
    passed = sum(bool(row["pass"]) for row in rows)
    return {
        "role_rows": rows,
        "passed_role_count": passed,
        "pass": passed == len(rows),
    }


def evaluate_gate(
    payloads: Mapping[str, Mapping[str, Any]],
    g_eighths: int,
    *,
    t3: Mapping[str, Any],
    t4: Mapping[str, Any],
    t5: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        g_eighths in (*AUTHORIZATION_G_EIGHTHS, analyzer.ORACLE_G_EIGHTHS),
        "gate is not a candidate or oracle",
    )
    t1 = evaluate_t1(payloads, g_eighths)
    t2 = evaluate_t2(payloads, g_eighths)
    all_pass = all(
        (bool(t1["pass"]), bool(t2["pass"]), bool(t3["pass"]), bool(t4["pass"]), bool(t5["pass"]))
    )
    return {
        "g_eighths": g_eighths,
        "g": g_eighths / 8.0,
        "authorization_eligible": g_eighths in AUTHORIZATION_G_EIGHTHS,
        "t1": t1,
        "t2": t2,
        "t3_pass": bool(t3["pass"]),
        "t4_pass": bool(t4["pass"]),
        "t5_pass": bool(t5["pass"]),
        "all_t1_to_t5_pass": all_pass,
    }


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    input_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(tuple(payloads) == expected_role_keys(), "comparison matrix differs")
    _require(len(input_artifacts) == len(expected_role_keys()), "artifact count differs")
    for key in expected_role_keys():
        analyzer.validate_output_payload(payloads[key])
    protocol_signatures = {
        json.dumps(payloads[key]["protocol"], sort_keys=True, allow_nan=False)
        for key in expected_role_keys()
    }
    identity_passed = all(
        payloads[key]["identity"]["g0_raw_logit_bitwise_equal"] is True
        and payloads[key]["identity"]["g0_returned_probability_bitwise_equal"] is True
        for key in expected_role_keys()
    )
    _require(identity_passed, "one or more g=0 identity audits failed")
    _require(len(protocol_signatures) == 1, "role protocols differ")
    t3 = evaluate_t3(payloads)
    t4 = evaluate_t4(payloads)
    t5 = evaluate_t5(payloads)
    historical_rows = []
    for key in expected_role_keys():
        payload = payloads[key]
        audit = payload["current_reference"]["g0_historical_drift_audit"]
        historical_rows.append(
            {
                "dataset": payload["dataset"],
                "checkpoint_role": payload["checkpoint_role"],
                "historical_exact": bool(audit["historical_exact"]),
                "historical_within_frozen_dorf_tolerance": bool(
                    audit["historical_within_frozen_dorf_tolerance"]
                ),
                "historical_count_fields_exact": bool(
                    audit["historical_count_fields_exact"]
                ),
                "background_false_positive_pixel_delta": int(
                    audit["background_false_positive_pixel_delta"]
                ),
            }
        )
    candidate_evaluations = [
        evaluate_gate(payloads, g_eighths, t3=t3, t4=t4, t5=t5)
        for g_eighths in AUTHORIZATION_G_EIGHTHS
    ]
    oracle = evaluate_gate(
        payloads,
        analyzer.ORACLE_G_EIGHTHS,
        t3=t3,
        t4=t4,
        t5=t5,
    )
    _require(oracle["authorization_eligible"] is False, "oracle became eligible")
    passing = [
        int(candidate["g_eighths"])
        for candidate in candidate_evaluations
        if candidate["all_t1_to_t5_pass"]
    ]
    trigger_passed = bool(passing)
    decision = DECISION_PASS if trigger_passed else DECISION_FAIL
    source = _source_sha256()
    output = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "seed": analyzer.SEED,
        "split": "img_idx/test",
        "threshold": analyzer.FIXED_THRESHOLD,
        "float_eq_atol": FLOAT_EQ_ATOL,
        "float_eq_rtol": FLOAT_EQ_RTOL,
        "test_selected": True,
        "selection_is_optimistic": True,
        "input_artifacts": [dict(item) for item in input_artifacts],
        "matrix_validation": {
            "exact_cross_product_3x2": True,
            "duplicate_role_count": 0,
            "missing_role_count": 0,
            "all_role_artifacts_engineering_valid": all(
                payloads[key].get("engineering_valid") is True
                for key in expected_role_keys()
            ),
            "all_g0_identity_passed": identity_passed,
            "all_artifacts_share_protocol": len(protocol_signatures) == 1,
            "all_artifacts_share_g_grid": True,
        },
        "global_signal_gates": {"t3": t3, "t4": t4, "t5": t5},
        "bound_historical_reference_drift": {
            "authorization_gate": False,
            "uniform_inference_math_used_for_all_roles": True,
            "role_rows": historical_rows,
            "historical_exact_role_count": sum(
                row["historical_exact"] for row in historical_rows
            ),
            "within_frozen_dorf_tolerance_role_count": sum(
                row["historical_within_frozen_dorf_tolerance"]
                for row in historical_rows
            ),
        },
        "candidate_evaluations": candidate_evaluations,
        "oracle": {
            **oracle,
            "metrics_reported": True,
            "would_pass_t1_to_t5": oracle["all_t1_to_t5_pass"],
            "cannot_authorize": True,
        },
        "result": {
            "passing_authorization_g_eighths": passing,
            "passing_authorization_gates": [value / 8.0 for value in passing],
            "selected_fixed_g": None,
            "fixed_g_is_not_training_initialization": True,
            "zero_training_trigger_passed": trigger_passed,
            "pbdr_implementation_authorized": trigger_passed,
            "pbdr_training_authorized": False,
            "training_authorization_pending_code_tests": trigger_passed,
            "decision": decision,
        },
        "source_sha256": source,
        "severe_contract": {
            "authority": "analysis/compare_three_dataset_dorf_v1.py",
            "authority_sha256": source[
                "analysis/compare_three_dataset_dorf_v1.py"
            ],
            "condition_order": list(dorf_compare.SEVERE_CONDITION_ORDER),
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
    }
    validate_comparison(output)
    return output


def validate_comparison(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "comparison schema differs")
    _require(payload.get("status") == "complete", "comparison is incomplete")
    matrix = payload.get("matrix_validation")
    _require(isinstance(matrix, Mapping), "matrix validation is missing")
    _require(all(bool(value) for value in matrix.values() if isinstance(value, bool)), "matrix validation failed")
    candidates = payload.get("candidate_evaluations")
    _require(
        isinstance(candidates, list)
        and [item.get("g_eighths") for item in candidates]
        == list(AUTHORIZATION_G_EIGHTHS),
        "candidate gate order differs",
    )
    result = payload.get("result")
    _require(isinstance(result, Mapping), "comparison result is missing")
    passing = result.get("passing_authorization_g_eighths")
    _require(isinstance(passing, list), "passing gate list differs")
    recomputed = [
        int(item["g_eighths"])
        for item in candidates
        if item.get("all_t1_to_t5_pass") is True
    ]
    _require(passing == recomputed, "passing gate list differs")
    _require(result.get("selected_fixed_g") is None, "comparator selected a fixed g")
    expected_pass = bool(recomputed)
    _require(
        result.get("zero_training_trigger_passed") is expected_pass,
        "trigger result differs",
    )
    _require(
        result.get("decision") == (DECISION_PASS if expected_pass else DECISION_FAIL),
        "decision differs",
    )
    oracle = payload.get("oracle")
    _require(
        isinstance(oracle, Mapping)
        and oracle.get("g_eighths") == analyzer.ORACLE_G_EIGHTHS
        and oracle.get("authorization_eligible") is False
        and oracle.get("cannot_authorize") is True,
        "oracle contract differs",
    )
    json.dumps(payload, allow_nan=False)


def render_markdown(payload: Mapping[str, Any]) -> str:
    result = payload["result"]
    lines = [
        "# PBDR-V1 零训练审计裁决",
        "",
        f"- 决策：`{result['decision']}`",
        f"- Trigger：`{str(result['zero_training_trigger_passed']).lower()}`",
        f"- 通过的固定 g：`{result['passing_authorization_gates']}`",
        "- 固定 g 仅为统一方向筛选，不作为训练初始化；正式 routing_logit 仍从 0 开始。",
        "",
        "| g | T1 | T2 | T3 | T4 | T5 | 全部通过 |",
        "|---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for candidate in payload["candidate_evaluations"]:
        lines.append(
            "| {g:.3f} | {t1} | {t2} | {t3} | {t4} | {t5} | {all_pass} |".format(
                g=float(candidate["g"]),
                t1="是" if candidate["t1"]["pass"] else "否",
                t2="是" if candidate["t2"]["pass"] else "否",
                t3="是" if candidate["t3_pass"] else "否",
                t4="是" if candidate["t4_pass"] else "否",
                t5="是" if candidate["t5_pass"] else "否",
                all_pass="是" if candidate["all_t1_to_t5_pass"] else "否",
            )
        )
    lines.extend(
        [
            "",
            "## 信号门",
            "",
            f"- T3：`{payload['global_signal_gates']['t3']['pass']}`",
            f"- T4：`{payload['global_signal_gates']['t4']['pass']}`",
            f"- T5：`{payload['global_signal_gates']['t5']['pass']}`",
            "",
            "该结果仍是 seed42、img_idx/test-selected 开发审计，不建立随机性稳定性或论文核心结论。",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_create_text(path: Path, text: str) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    decision_path = args.output_dir / "decision.json"
    markdown_path = args.output_dir / "decision.md"
    for path in (decision_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    payloads, artifacts = load_role_payloads(args.input_root)
    output = compare_payloads(payloads, input_artifacts=artifacts)
    atomic_create_text(
        decision_path,
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    atomic_create_text(markdown_path, render_markdown(output))
    print(
        json.dumps(
            {
                "status": "complete",
                "decision": output["result"]["decision"],
                "trigger_passed": output["result"]["zero_training_trigger_passed"],
                "passing_gates": output["result"]["passing_authorization_gates"],
                "decision_json": str(decision_path.resolve()),
                "decision_sha256": file_sha256(decision_path.resolve()),
                "decision_markdown": str(markdown_path.resolve()),
                "decision_markdown_sha256": file_sha256(markdown_path.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_G_EIGHTHS",
    "CURRENT_MODE",
    "DECISION_FAIL",
    "DECISION_PASS",
    "SCHEMA",
    "compare_payloads",
    "evaluate_gate",
    "evaluate_t1",
    "evaluate_t1_dataset",
    "evaluate_t2",
    "evaluate_t3",
    "evaluate_t4",
    "evaluate_t5",
    "float_not_lower",
    "float_strict_higher",
    "load_role_payloads",
    "render_markdown",
    "validate_comparison",
]
