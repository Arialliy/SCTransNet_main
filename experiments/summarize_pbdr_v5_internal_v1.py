#!/usr/bin/env python3
"""Summarize the three frozen PBDR-V5 internal runs without test access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pbdr_v4_run_artifacts import exclusive_json, file_sha256
from experiments.pbdr_v5_internal_selector import (
    FROZEN_FAMILY_ORDER,
    select_internal_candidate,
)
from experiments.pbdr_v5_run_contract import canonical_json_sha256


SCHEMA = "sctransnet_pbdr_v5_internal_summary/v1"
TRAINING_SCHEMA = "sctransnet_three_dataset_pbdr_v5_training_v1/v1"
RUNS = (
    ("NUDT-SIRST", "best_pd"),
    ("NUAA-SIRST", "best_miou"),
    ("IRSTD-1K", "best_miou"),
)
V5_RUN_RELATIVE_DIRS = {
    ("NUDT-SIRST", "best_pd"): Path(
        "results/pbdr_v5_v1/training_idle_gpu/NUDT-SIRST/best_pd"
    ),
    ("NUAA-SIRST", "best_miou"): Path(
        "results/pbdr_v5_v1/training/NUAA-SIRST/best_miou"
    ),
    ("IRSTD-1K", "best_miou"): Path(
        "results/pbdr_v5_v1/training_idle_gpu/IRSTD-1K/best_miou"
    ),
}


class PBDRV5SummaryError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV5SummaryError(message)


def _read(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(candidate.is_file() and not candidate.is_symlink(), f"{label} is missing or unsafe")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV5SummaryError(f"cannot read {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain one object")
    return payload


def _commit_or_validate(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if not destination.exists() and not destination.is_symlink():
        return exclusive_json(destination, payload).resolve(strict=True)
    observed = _read(destination, label=destination.name)
    _require(observed == dict(payload), f"existing {destination.name} differs")
    return destination.resolve(strict=True)


def _v3_selected_metrics(sweep: Mapping[str, Any]) -> dict[str, Any]:
    selected = sweep.get("selected")
    candidates = sweep.get("candidates")
    _require(isinstance(selected, Mapping) and isinstance(candidates, list), "V3 sweep differs")
    name = selected.get("name")
    matches = [item for item in candidates if isinstance(item, Mapping) and item.get("name") == name]
    _require(len(matches) == 1 and isinstance(matches[0].get("metrics"), Mapping), "V3 selected candidate differs")
    return dict(matches[0]["metrics"])


def _v5_summary_path(root: Path, dataset: str, role: str) -> Path:
    key = (dataset, role)
    _require(key in V5_RUN_RELATIVE_DIRS, f"unsupported V5 run: {dataset}/{role}")
    return root / V5_RUN_RELATIVE_DIRS[key] / "summary.json"


def _load_role(root: Path, dataset: str, role: str) -> dict[str, Any]:
    sweep_path = root / f"results/pbdr_v4_v1/v3_calibration/{dataset}/{role}/sweep_result.json"
    stage1_path = root / f"results/pbdr_v4_v1/training/{dataset}/{role}/stage1/summary.json"
    stage2_path = root / f"results/pbdr_v4_v1/training/{dataset}/{role}/stage2/summary.json"
    v5_path = _v5_summary_path(root, dataset, role)
    sweep = _read(sweep_path, label=f"{dataset}/{role} V3 sweep")
    stage1 = _read(stage1_path, label=f"{dataset}/{role} V4-Stage1 summary")
    stage2 = _read(stage2_path, label=f"{dataset}/{role} V4-Stage2 summary")
    v5 = _read(v5_path, label=f"{dataset}/{role} V5 summary")
    for label, payload in (("sweep", sweep), ("Stage1", stage1), ("Stage2", stage2), ("V5", v5)):
        _require(payload.get("official_test_accessed") is False, f"{label} official-access flag differs")
        _require(payload.get("performance_acceptance_margin") is None, f"{label} margin differs")
    _require(v5.get("schema") == TRAINING_SCHEMA and v5.get("status") == "complete", "V5 summary is incomplete")
    for label, payload in (("sweep", sweep), ("Stage1", stage1), ("Stage2", stage2), ("V5", v5)):
        _require(payload.get("dataset") == dataset, f"{label} dataset differs")
        _require(payload.get("role") == role, f"{label} role differs")
    baselines = sweep.get("baselines")
    _require(isinstance(baselines, Mapping), "V3 baselines differ")
    original = baselines.get("original")
    current = baselines.get("current")
    _require(isinstance(original, Mapping) and isinstance(original.get("metrics"), Mapping), "Original metrics differ")
    _require(isinstance(current, Mapping) and isinstance(current.get("metrics"), Mapping), "Current metrics differ")
    metrics_by_family = {
        "Original": dict(original["metrics"]),
        "Current": dict(current["metrics"]),
        "V3-calibrated": _v3_selected_metrics(sweep),
        "V4-Stage1": dict(stage1["selected_metrics"]),
        "V4-Stage2": dict(stage2["selected_metrics"]),
        "V5": dict(v5["selected_metrics"]),
    }
    report = select_internal_candidate(role, metrics_by_family)  # type: ignore[arg-type]
    return {
        "dataset": dataset,
        "role": role,
        "source_files": {
            "v3_sweep": {"path": str(sweep_path.resolve()), "sha256": file_sha256(sweep_path)},
            "v4_stage1": {"path": str(stage1_path.resolve()), "sha256": file_sha256(stage1_path)},
            "v4_stage2": {"path": str(stage2_path.resolve()), "sha256": file_sha256(stage2_path)},
            "v5": {"path": str(v5_path.resolve()), "sha256": file_sha256(v5_path)},
        },
        "selected_epochs": {
            "V4-Stage1": stage1["selected_epoch"],
            "V4-Stage2": stage2["selected_epoch"],
            "V5": v5["selected_epoch"],
        },
        "v5_initial_metrics": v5["initial_metrics"],
        "metrics_by_family": metrics_by_family,
        "selection": report,
    }


def _independent_baseline_comparison(
    diagnosis: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = diagnosis.get("user_supplied_independent_baseline")
    current_comparison = diagnosis.get("designed_current_vs_user_baseline_best_miou")
    _require(isinstance(baseline, Mapping), "user baseline differs")
    _require(isinstance(current_comparison, Mapping), "official Current comparison differs")
    _require(
        current_comparison.get("comparison_scope")
        == "best_miou_checkpoint_to_best_miou_checkpoint",
        "official Current comparison scope differs",
    )
    baseline_datasets = baseline.get("datasets")
    _require(isinstance(baseline_datasets, Mapping), "user baseline datasets differ")
    for dataset, _role in RUNS:
        _require(isinstance(baseline_datasets.get(dataset), Mapping), f"{dataset} baseline differs")
        record = current_comparison.get(dataset)
        _require(isinstance(record, Mapping), f"{dataset} official Current comparison differs")
        _require(
            isinstance(record.get("designed_current"), Mapping)
            and isinstance(record.get("delta_designed_minus_baseline"), Mapping),
            f"{dataset} official Current comparison payload differs",
        )
    return dict(baseline), dict(current_comparison)


def build_summary(root: Path = REPO_ROOT) -> dict[str, Any]:
    diagnosis_path = root / "results/pbdr_v5_v1/diagnostics/failure_localization_bundle.json"
    diagnosis = _read(diagnosis_path, label="failure-localization bundle")
    _require(diagnosis.get("official_test_accessed") is False, "diagnosis official flag differs")
    baseline, official_current_comparison = _independent_baseline_comparison(diagnosis)
    roles = [_load_role(root, dataset, role) for dataset, role in RUNS]
    strict_improvements = [
        item["selection"]["v5_strictly_improves_existing_envelope"] for item in roles
    ]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "development_train_plus_internal_validation_only",
        "candidate_family_order": list(FROZEN_FAMILY_ORDER),
        "roles": roles,
        "v5_strict_improvement_count": sum(bool(value) for value in strict_improvements),
        "all_v5_roles_strictly_improve": all(strict_improvements),
        "user_supplied_independent_baseline": baseline,
        "official_current_vs_user_baseline_best_miou": official_current_comparison,
        "independent_baseline_direct_comparison_policy": {
            "only_directly_comparable_model": "official Current best-mIoU checkpoint",
            "internal_v5_vs_user_baseline_direct_comparison_available": False,
            "reason": (
                "V5 has internal-validation results only; the user baseline is an external "
                "best-checkpoint scalar reference. Only the already-recorded official Current "
                "best-mIoU result has the matching direct-comparison scope."
            ),
        },
        "selection_is_optimistic": True,
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    return result


def _metric(metrics: Mapping[str, Any], name: str) -> str:
    value = metrics[name]
    if name in ("miou", "niou", "pixel_f1"):
        return f"{float(value):.9f}"
    if name == "fa":
        return f"{float(value):.9e}"
    return str(value)


def _transition(baseline: float, current: float, delta: float, *, suffix: str) -> str:
    return f"{baseline:.4f} → {current:.4f} ({delta:+.4f} {suffix})"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# PBDR-V5 内部微调结果",
        "",
        "> 仅 development-train 与 internal-validation；未访问 official test。",
        "",
    ]
    for role_record in summary["roles"]:
        dataset = role_record["dataset"]
        role = role_record["role"]
        selection = role_record["selection"]
        lines.extend(
            [
                f"## {dataset} / `{role}`",
                "",
                "| 家族 | mIoU | nIoU | F1 | Pd | Fa | tiny-Pd |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for family in FROZEN_FAMILY_ORDER:
            metrics = role_record["metrics_by_family"][family]
            lines.append(
                f"| {family} | {_metric(metrics, 'miou')} | {_metric(metrics, 'niou')} | "
                f"{_metric(metrics, 'pixel_f1')} | {metrics['matched_target_count']}/{metrics['target_count']} | "
                f"{_metric(metrics, 'fa')} | {metrics['matched_tiny_target_count']}/{metrics['tiny_target_count']} |"
            )
        lines.extend(
            [
                "",
                f"- 既有五族内部包络：`{selection['existing_envelope_winner']}`",
                f"- 六族胜者：`{selection['winner']}`",
                f"- V5 严格改善既有包络：`{str(selection['v5_strictly_improves_existing_envelope']).lower()}`",
                "",
            ]
        )
    baseline = summary["user_supplied_independent_baseline"]["datasets"]
    official_current = summary["official_current_vs_user_baseline_best_miou"]
    lines.extend(
        [
            "## 独立 Baseline 的可直接比较结果",
            "",
            "> 本表只比较同为 best-mIoU checkpoint 口径的用户 Baseline 与 official Current；"
            "不包含 internal V5。",
            "",
            "| 数据集 | mIoU (%) | nIoU (%) | F1 (%) | Pd (%) | Fa (×10⁻⁶) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, _role in RUNS:
        baseline_record = baseline[dataset]
        comparison_record = official_current[dataset]
        current_record = comparison_record["designed_current"]
        delta_record = comparison_record["delta_designed_minus_baseline"]
        lines.append(
            f"| {dataset} | "
            f"{_transition(baseline_record['best_miou_percent'], current_record['miou_percent'], delta_record['miou_percentage_points'], suffix='pp')} | "
            f"{_transition(baseline_record['niou_percent'], current_record['niou_percent'], delta_record['niou_percentage_points'], suffix='pp')} | "
            f"{_transition(baseline_record['f1_percent'], current_record['f1_percent'], delta_record['f1_percentage_points'], suffix='pp')} | "
            f"{_transition(baseline_record['pd_percent'], current_record['pd_percent'], delta_record['pd_percentage_points'], suffix='pp')} | "
            f"{_transition(baseline_record['fa_times_1e6'], current_record['fa_times_1e6'], delta_record['fa_times_1e6'], suffix='')} |"
        )
    lines.extend(
        [
            "",
            "## 口径边界",
            "",
            "V5 尚无新的独立测试结果，因此禁止把内部验证数值直接与用户给出的独立训练 "
            "Baseline 标量相减。V5 本轮只判断是否按完整 role key 严格超过既有内部包络；"
            "完全同键时保留冻结顺序中更早的家族。",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_dir: Path) -> tuple[Path, Path]:
    summary = build_summary()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _require(not destination.is_symlink(), "output directory cannot be a symlink")
    json_path = _commit_or_validate(destination / "internal_summary.json", summary)
    markdown_path = destination / "INTERNAL_SUMMARY.md"
    content = render_markdown(summary)
    if markdown_path.exists() or markdown_path.is_symlink():
        _require(markdown_path.is_file() and not markdown_path.is_symlink(), "Markdown path is unsafe")
        _require(markdown_path.read_text(encoding="utf-8") == content, "existing Markdown differs")
    else:
        # The JSON is the authoritative append-only artifact; Markdown is a
        # deterministic human-readable rendering written only after it commits.
        markdown_path.write_text(content, encoding="utf-8")
    return json_path, markdown_path.resolve(strict=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/pbdr_v5_v1/comparison",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print("\n".join(str(path) for path in run(parse_args(argv).output_dir)))


if __name__ == "__main__":
    main()
