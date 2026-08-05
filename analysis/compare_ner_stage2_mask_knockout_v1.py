#!/usr/bin/env python3
"""Freeze and apply the NER stage-2 knockout trigger ``A AND (B OR C)``.

The comparator consumes exactly one seed-42, best-mIoU knockout result for
each of NUAA-SIRST, NUDT-SIRST, and IRSTD-1K.  It performs no inference.  The
thresholds below are executable versions of the approved development gate;
the output authorizes at most one test-selected V5-PER development candidate,
not a paper-level mechanism or stability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_ner_stage2_mask_knockout_v1 as analyzer  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA = "sctransnet_ner_stage2_mask_knockout_comparison_v1/v1"
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLE = "best_miou"
SEED = 42

A_REQUIRED_DATASET_COUNT = 2
A_RELATIVE_REDUCTION_MINIMUM = 0.05
A_MATCHED_TARGET_DROP_MAX_EXCLUSIVE = 2
A_MATCHED_TINY_TARGET_DROP_MAX_EXCLUSIVE = 2
A_MIOU_DROP_MAX_EXCLUSIVE = 0.005
A_NIOU_DROP_MAX_EXCLUSIVE = 0.005

B_REQUIRED_DATASET_COUNT = 2
B_DENSITY_RATIO_MINIMUM = 1.25

C_REQUIRED_DATASET_COUNT = 2
C_LOW_P2_THRESHOLD = 0.25
C_BACKGROUND_POSITIVE_LOCAL_MASS_SHARE_MINIMUM = 0.25

DEFAULT_INPUT_ROOT = analyzer.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "comparison" / "best_miou_seed42"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    ready = float(value)
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _optional_finite_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label)


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def evaluate_dataset_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one analyzer result and recompute its A/B/C decisions."""

    _require(payload.get("schema") == analyzer.SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer result is incomplete")
    dataset = payload.get("dataset")
    _require(dataset in DATASETS, f"unsupported dataset: {dataset!r}")
    _require(payload.get("checkpoint_role") == CHECKPOINT_ROLE, "checkpoint role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("intervention") == analyzer.INTERVENTION, "intervention differs")
    fixed = payload.get("fixed_threshold_0_5")
    _require(isinstance(fixed, Mapping), "analyzer result lacks fixed point")
    _require(fixed.get("threshold") == 0.5, "fixed threshold differs")
    audit = payload.get("intervention_audit")
    _require(isinstance(audit, Mapping), "analyzer result lacks intervention audit")
    _require(audit.get("model_state_unchanged") is True, "model state changed")
    _require(
        audit.get("returned_stage2_mask_abs_max") == 0.0,
        "stage2 return was not exactly zero",
    )

    gate_inputs = payload.get("gate_inputs")
    _require(isinstance(gate_inputs, Mapping), "analyzer result lacks gate inputs")
    raw_a = gate_inputs.get("A")
    raw_b = gate_inputs.get("B")
    raw_c = gate_inputs.get("C")
    _require(isinstance(raw_a, Mapping), "gate A input is missing")
    _require(isinstance(raw_b, Mapping), "gate B input is missing")
    _require(isinstance(raw_c, Mapping), "gate C input is missing")

    component_reduction = _optional_finite_float(
        raw_a.get("component_fa_relative_reduction"),
        "component-Fa relative reduction",
    )
    pixel_reduction = _optional_finite_float(
        raw_a.get("all_background_pixel_fp_relative_reduction"),
        "pixel-FP relative reduction",
    )
    matched_drop = int(raw_a["matched_target_drop"])
    tiny_drop = int(raw_a["matched_tiny_target_drop"])
    miou_drop = _finite_float(raw_a["miou_drop"], "mIoU drop")
    niou_drop = _finite_float(raw_a["niou_drop"], "nIoU drop")
    a_conditions = {
        "component_fa_or_pixel_fp_reduction_ge_5pct": (
            _at_least(component_reduction, A_RELATIVE_REDUCTION_MINIMUM)
            or _at_least(pixel_reduction, A_RELATIVE_REDUCTION_MINIMUM)
        ),
        "matched_target_drop_lt_2": (
            matched_drop < A_MATCHED_TARGET_DROP_MAX_EXCLUSIVE
        ),
        "matched_tiny_target_drop_lt_2": (
            tiny_drop < A_MATCHED_TINY_TARGET_DROP_MAX_EXCLUSIVE
        ),
        "miou_drop_lt_0_005": miou_drop < A_MIOU_DROP_MAX_EXCLUSIVE,
        "niou_drop_lt_0_005": niou_drop < A_NIOU_DROP_MAX_EXCLUSIVE,
    }
    a_pass = all(a_conditions.values())

    b_available = raw_b.get("available") is True
    b_reason = raw_b.get("reason")
    b_ratio = None
    b_denominator_zero = raw_b.get("denominator_is_zero") is True
    if b_available:
        b_ratio = _optional_finite_float(raw_b.get("density_ratio"), "B density ratio")
    b_pass = (
        b_available
        and not b_denominator_zero
        and _at_least(b_ratio, B_DENSITY_RATIO_MINIMUM)
    )

    c_threshold = _finite_float(raw_c.get("low_p2_threshold"), "C P2 threshold")
    _require(c_threshold == C_LOW_P2_THRESHOLD, "C P2 threshold differs")
    c_available = raw_c.get("available") is True
    c_denominator_zero = raw_c.get("denominator_is_zero") is True
    c_share = None
    if c_available:
        c_share = _optional_finite_float(raw_c.get("mass_share"), "C mass share")
    c_pass = (
        c_available
        and not c_denominator_zero
        and _at_least(c_share, C_BACKGROUND_POSITIVE_LOCAL_MASS_SHARE_MINIMUM)
    )

    return {
        "dataset": dataset,
        "A": {
            "pass": a_pass,
            "conditions": a_conditions,
            "component_fa_relative_reduction": component_reduction,
            "all_background_pixel_fp_relative_reduction": pixel_reduction,
            "matched_target_drop": matched_drop,
            "matched_tiny_target_drop": tiny_drop,
            "miou_drop": miou_drop,
            "niou_drop": niou_drop,
        },
        "B": {
            "pass": b_pass,
            "available": b_available,
            "reason": b_reason,
            "density_ratio": b_ratio,
            "denominator_is_zero": b_denominator_zero,
        },
        "C": {
            "pass": c_pass,
            "available": c_available,
            "low_p2_threshold": c_threshold,
            "background_positive_local_mass_share": c_share,
            "denominator_is_zero": c_denominator_zero,
        },
    }


def compare_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    input_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    _require(set(payloads) == set(DATASETS), "comparison requires exactly three datasets")
    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        result = evaluate_dataset_gate(payloads[dataset])
        _require(result["dataset"] == dataset, f"input binding differs: {dataset}")
        per_dataset[dataset] = result

    counts = {
        gate: sum(bool(per_dataset[dataset][gate]["pass"]) for dataset in DATASETS)
        for gate in ("A", "B", "C")
    }
    aggregate = {
        "A": {
            "pass": counts["A"] >= A_REQUIRED_DATASET_COUNT,
            "passed_dataset_count": counts["A"],
            "required_dataset_count": A_REQUIRED_DATASET_COUNT,
        },
        "B": {
            "pass": counts["B"] >= B_REQUIRED_DATASET_COUNT,
            "passed_dataset_count": counts["B"],
            "required_dataset_count": B_REQUIRED_DATASET_COUNT,
        },
        "C": {
            "pass": counts["C"] >= C_REQUIRED_DATASET_COUNT,
            "passed_dataset_count": counts["C"],
            "required_dataset_count": C_REQUIRED_DATASET_COUNT,
        },
    }
    authorized = bool(
        aggregate["A"]["pass"]
        and (aggregate["B"]["pass"] or aggregate["C"]["pass"])
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": (
            "AUTHORIZE_ONE_NER_V5_PER_DEVELOPMENT_CANDIDATE"
            if authorized
            else "DO_NOT_AUTHORIZE_NER_V5_PER_DEVELOPMENT_TRAINING"
        ),
        "ner_v5_per_development_training_authorized": authorized,
        "trigger_expression": "A AND (B OR C)",
        "checkpoint_role": CHECKPOINT_ROLE,
        "seed": SEED,
        "datasets": list(DATASETS),
        "threshold_contract": {
            "A": {
                "dataset_count": f">={A_REQUIRED_DATASET_COUNT}/{len(DATASETS)}",
                "component_fa_or_pixel_fp_relative_reduction": (
                    f">={A_RELATIVE_REDUCTION_MINIMUM}"
                ),
                "matched_target_drop": f"<{A_MATCHED_TARGET_DROP_MAX_EXCLUSIVE}",
                "matched_tiny_target_drop": (
                    f"<{A_MATCHED_TINY_TARGET_DROP_MAX_EXCLUSIVE}"
                ),
                "miou_drop": f"<{A_MIOU_DROP_MAX_EXCLUSIVE}",
                "niou_drop": f"<{A_NIOU_DROP_MAX_EXCLUSIVE}",
            },
            "B": {
                "dataset_count": f">={B_REQUIRED_DATASET_COUNT}/{len(DATASETS)}",
                "false_component_to_normal_background_positive_local_density_ratio": (
                    f">={B_DENSITY_RATIO_MINIMUM}"
                ),
                "zero_denominator_passes": False,
                "reference_probability_alignment_required": True,
            },
            "C": {
                "dataset_count": f">={C_REQUIRED_DATASET_COUNT}/{len(DATASETS)}",
                "low_p2_comparison": f"P2<={C_LOW_P2_THRESHOLD}",
                "low_p2_share_of_background_positive_local_mass": (
                    f">={C_BACKGROUND_POSITIVE_LOCAL_MASS_SHARE_MINIMUM}"
                ),
                "zero_denominator_passes": False,
            },
        },
        "aggregate_gates": aggregate,
        "per_dataset": per_dataset,
        "input_bindings": dict(input_bindings or {}),
        "scope": {
            "test_selected_development_gate_only": True,
            "authorizes_at_most_one_v5_candidate": True,
            "paper_mechanism_evidence": False,
            "stability_claim_supported": False,
        },
        "source_sha256": {
            "analysis/compare_ner_stage2_mask_knockout_v1.py": analyzer.file_sha256(
                Path(__file__)
            ),
            "analysis/analyze_ner_stage2_mask_knockout_v1.py": analyzer.file_sha256(
                Path(analyzer.__file__)
            ),
        },
        "no_fabricated_results": True,
    }


def _format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.3f}%"


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_markdown(result: Mapping[str, Any]) -> str:
    authorized = bool(result["ner_v5_per_development_training_authorized"])
    lines = [
        "# NER Stage-2 Mask Knockout V1 裁决",
        "",
        f"- 决策：`{result['decision']}`",
        f"- V5-PER development training authorized：`{str(authorized).lower()}`",
        "- 触发式：`A AND (B OR C)`",
        "- 范围：seed 42、test-selected、best_miou，仅授权一次开发候选。",
        "",
        "| 数据集 | component-Fa降幅 | pixel-FP降幅 | target下降 | tiny下降 | mIoU下降 | nIoU下降 | A | B | C低P2质量占比 | C |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|:---:|",
    ]
    for dataset in DATASETS:
        row = result["per_dataset"][dataset]
        a = row["A"]
        b = row["B"]
        c = row["C"]
        lines.append(
            "| "
            + " | ".join(
                (
                    dataset,
                    _format_percent(a["component_fa_relative_reduction"]),
                    _format_percent(a["all_background_pixel_fp_relative_reduction"]),
                    str(a["matched_target_drop"]),
                    str(a["matched_tiny_target_drop"]),
                    _format_float(a["miou_drop"]),
                    _format_float(a["niou_drop"]),
                    "PASS" if a["pass"] else "FAIL",
                    "PASS" if b["pass"] else "N/A" if not b["available"] else "FAIL",
                    _format_percent(c["background_positive_local_mass_share"]),
                    "PASS" if c["pass"] else "FAIL",
                )
            )
            + " |"
        )
    aggregate = result["aggregate_gates"]
    lines.extend(
        (
            "",
            "## 聚合门",
            "",
            f"- A：{aggregate['A']['passed_dataset_count']}/3（需至少 2/3）→ `{'PASS' if aggregate['A']['pass'] else 'FAIL'}`",
            f"- B：{aggregate['B']['passed_dataset_count']}/3（需至少 2/3）→ `{'PASS' if aggregate['B']['pass'] else 'FAIL'}`",
            f"- C：{aggregate['C']['passed_dataset_count']}/3（需至少 2/3）→ `{'PASS' if aggregate['C']['pass'] else 'FAIL'}`",
            "",
            "B 若显示 N/A，是因为单次 knockout 没有 V4 reference 概率缓存；不会把 knockout 假目标区域与 V4 mask 错配后计为通过。C 使用 `ReLU(centered local logits)`，不使用混入 DC 偏置的最终 arctan mask。",
            "",
        )
    )
    return "\n".join(lines)


def _default_input(input_root: Path, dataset: str) -> Path:
    return (
        Path(input_root)
        / "runs"
        / dataset
        / "v4_tss_off_best_miou_seed42"
        / analyzer.INTERVENTION
        / "evaluation.json"
    )


def _parse_bindings(values: Sequence[str], input_root: Path) -> dict[str, Path]:
    bindings = {dataset: _default_input(input_root, dataset) for dataset in DATASETS}
    for value in values:
        if "=" not in value:
            raise ValueError("--input must use DATASET=PATH")
        dataset, raw_path = value.split("=", 1)
        _require(dataset in DATASETS, f"unsupported --input dataset: {dataset}")
        _require(bool(raw_path), f"empty --input path for {dataset}")
        bindings[dataset] = Path(raw_path)
    return bindings


def atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="override one discovered analyzer result",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_DIR / "decision.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = _parse_bindings(args.input, args.input_root)
    payloads: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for dataset, path in input_paths.items():
        resolved = path.resolve(strict=True)
        payloads[dataset] = _load_json(resolved)
        bindings[dataset] = {
            "path": str(resolved),
            "sha256": analyzer.file_sha256(resolved),
        }
    result = compare_payloads(payloads, input_bindings=bindings)
    analyzer.atomic_write_json(args.output_json, result, overwrite=args.overwrite)
    atomic_write_text(
        args.output_md, render_markdown(result), overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "decision": result["decision"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            }
        )
    )


if __name__ == "__main__":
    main()
