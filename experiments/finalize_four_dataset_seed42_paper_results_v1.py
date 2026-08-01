#!/usr/bin/env python3
"""Initialize TBD tables or finalize the four-dataset seed-42 result package.

Initialization writes only explicit ``TBD``/null placeholders.  Finalization is
fail-closed: every expected fixed-point/source/sweep record must exist, bind a
frozen checkpoint SHA-256, and pass the protocol checks before a numeric table
is emitted.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import four_dataset_evaluation_protocol_v1 as protocol  # noqa: E402


TABLE_COLUMNS = (
    "training_dataset",
    "evaluation_dataset",
    "method",
    "checkpoint_role",
    "selected_epoch",
    "miou",
    "niou",
    "pixel_f1",
    "pd",
    "fa",
    "tiny_pd",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "unmatched_predicted_object_count",
    "checkpoint_sha256",
    "test_selected",
    "selection_is_optimistic",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=protocol.EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--initialize-templates",
        action="store_true",
        help="Write placeholder-only result tables and stop.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing generated tables/summary.",
    )
    args = parser.parse_args(argv)
    return args


def _table_row(
    training_dataset: str,
    evaluation_dataset: str,
    method: str,
    role: str,
    *,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "training_dataset": training_dataset,
        "evaluation_dataset": evaluation_dataset,
        "method": protocol.METHOD_LABELS[method],
        "checkpoint_role": role,
        "selected_epoch": "TBD",
        "miou": "TBD",
        "niou": "TBD",
        "pixel_f1": "TBD",
        "pd": "TBD",
        "fa": "TBD",
        "tiny_pd": "TBD",
        "false_objects_per_image": "TBD",
        "target_count": "TBD",
        "matched_target_count": "TBD",
        "tiny_target_count": "TBD",
        "matched_tiny_target_count": "TBD",
        "unmatched_predicted_object_count": "TBD",
        "checkpoint_sha256": "TBD",
        "test_selected": role in protocol.SELECTED_ROLES,
        "selection_is_optimistic": role in protocol.SELECTED_ROLES,
    }
    if result is None:
        return row
    fixed = result["fixed_threshold_0_5"]
    checkpoint = result.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        selected_epoch = checkpoint["epoch"]
        checkpoint_sha256 = checkpoint["sha256"]
    else:
        selected_epoch = result["epoch"]
        checkpoint_sha256 = "N/A"
    row.update(
        {
            "selected_epoch": selected_epoch,
            "miou": fixed["miou"],
            "niou": fixed["niou"],
            "pixel_f1": fixed["pixel_f1"],
            "pd": fixed["pd"],
            "fa": fixed["fa"],
            "tiny_pd": fixed["tiny_pd"],
            "false_objects_per_image": fixed[
                "false_objects_per_image"
            ],
            "target_count": fixed["target_count"],
            "matched_target_count": fixed["matched_target_count"],
            "tiny_target_count": fixed["tiny_target_count"],
            "matched_tiny_target_count": fixed[
                "matched_tiny_target_count"
            ],
            "unmatched_predicted_object_count": fixed[
                "unmatched_predicted_object_count"
            ],
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    return row


def _dataset_rows(
    role: str,
    loader: Callable[[str, str, str], Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    return [
        _table_row(
            dataset,
            dataset,
            method,
            role,
            result=loader(dataset, method, role),
        )
        for dataset in protocol.DATASETS
        for method in protocol.METHODS
    ]


def _source_rows(
    role: str,
    loader: Callable[[str, str, str], Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    return [
        _table_row(
            "SIRST3",
            source,
            method,
            role,
            result=loader(source, method, role),
        )
        for source in protocol.SOURCE_DATASETS
        for method in protocol.METHODS
    ]


def _format_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-4:
            return f"{value:.8g}"
        return f"{value:.6f}"
    return str(value)


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _format_cell(row[key]) for key in TABLE_COLUMNS})
    return output.getvalue()


def _markdown_text(
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    visible = (
        "training_dataset",
        "evaluation_dataset",
        "method",
        "checkpoint_role",
        "selected_epoch",
        "miou",
        "niou",
        "pixel_f1",
        "pd",
        "fa",
        "tiny_pd",
        "false_objects_per_image",
    )
    labels = {
        "training_dataset": "Train",
        "evaluation_dataset": "Test",
        "method": "Method",
        "checkpoint_role": "Role",
        "selected_epoch": "Epoch",
        "miou": "mIoU ↑",
        "niou": "nIoU ↑",
        "pixel_f1": "F1 ↑",
        "pd": "Pd ↑",
        "fa": "Fa ↓",
        "tiny_pd": "tiny-Pd ↑",
        "false_objects_per_image": "False obj./image ↓",
    }
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(labels[key] for key in visible) + " |",
        "| " + " | ".join("---" for _ in visible) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_cell(row[key]) for key in visible)
            + " |"
        )
    lines.extend(
        [
            "",
            "All selected best rows are test-selected "
            "(`test_selected=true`, `selection_is_optimistic=true`). "
            "Epoch-1000 rows are fixed endpoints.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    payload = {"text": text}
    temporary_json = path.with_name(f".{path.name}.payload.json")
    protocol.atomic_write_json(temporary_json, payload, overwrite=True)
    try:
        decoded = protocol.load_json_object(temporary_json)["text"]
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        temporary.write_text(decoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary_json.exists():
            temporary_json.unlink()


def _write_table_pair(
    directory: Path,
    stem: str,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    csv_path = directory / f"{stem}.csv"
    markdown_path = directory / f"{stem}.md"
    _atomic_write_text(csv_path, _csv_text(rows), overwrite=overwrite)
    _atomic_write_text(
        markdown_path,
        _markdown_text(title, rows),
        overwrite=overwrite,
    )
    return {
        "csv": {
            "path": str(csv_path.resolve()),
            "sha256": protocol.file_sha256(csv_path),
        },
        "markdown": {
            "path": str(markdown_path.resolve()),
            "sha256": protocol.file_sha256(markdown_path),
        },
    }


def _budget_rows_placeholder() -> list[dict[str, Any]]:
    return [
        {
            "training_dataset": dataset,
            "evaluation_dataset": dataset,
            "method": protocol.METHOD_LABELS[method],
            "checkpoint_role": "best_miou",
            **{f"{budget:.10g}": "TBD" for budget in protocol.FA_BUDGETS},
        }
        for dataset in protocol.DATASETS
        for method in protocol.METHODS
    ]


def _budget_table_text(rows: Sequence[Mapping[str, Any]]) -> str:
    budget_keys = tuple(f"{budget:.10g}" for budget in protocol.FA_BUDGETS)
    columns = (
        "training_dataset",
        "evaluation_dataset",
        "method",
        "checkpoint_role",
        *budget_keys,
    )
    labels = {
        "training_dataset": "Train",
        "evaluation_dataset": "Test",
        "method": "Method",
        "checkpoint_role": "Role",
        **{key: f"Pd@Fa≤{key}" for key in budget_keys},
    }
    lines = [
        "# Table 7: Pd under preregistered Fa budgets",
        "",
        "| " + " | ".join(labels[key] for key in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_cell(row[key]) for key in columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "Each cell reports the maximum Pd among thresholds whose achieved "
            "Fa is at or below the stated budget. Curves do not replace the "
            "fixed-threshold 0.5 result.",
            "",
        ]
    )
    return "\n".join(lines)


def initialize_templates(results_root: Path, *, overwrite: bool) -> dict[str, Any]:
    tables = Path(results_root) / "tables"

    def missing_loader(
        _dataset: str,
        _method: str,
        _role: str,
    ) -> None:
        return None

    generated: dict[str, Any] = {}
    specs = (
        (
            "table2_best_miou",
            "Table 2: dataset-specific best-mIoU checkpoints",
            _dataset_rows("best_miou", missing_loader),
        ),
        (
            "table4a_best_pd",
            "Table 4a: dataset-specific best-Pd checkpoints",
            _dataset_rows("best_pd", missing_loader),
        ),
        (
            "table4b_last_epoch1000",
            "Table 4b: fixed epoch-1000 metric endpoints (no checkpoint saved)",
            _dataset_rows("last_epoch1000", missing_loader),
        ),
    )
    for stem, title, rows in specs:
        generated[stem] = _write_table_pair(
            tables,
            stem,
            title,
            rows,
            overwrite=overwrite,
        )
    for role, suffix in (
        ("best_miou", "a_best_miou"),
        ("best_pd", "b_best_pd"),
    ):
        stem = f"table3{suffix}_sirst3_three_sources"
        generated[stem] = _write_table_pair(
            tables,
            stem,
            f"Table 3{suffix[0]}: one SIRST3 checkpoint on three sources",
            _source_rows(role, missing_loader),
            overwrite=overwrite,
        )
    budget_path = tables / "table7_pd_at_fa_budgets.md"
    _atomic_write_text(
        budget_path,
        _budget_table_text(_budget_rows_placeholder()),
        overwrite=overwrite,
    )
    generated["table7_pd_at_fa_budgets"] = {
        "markdown": {
            "path": str(budget_path.resolve()),
            "sha256": protocol.file_sha256(budget_path),
        }
    }
    template = {
        "schema": f"{protocol.SCHEMA_PREFIX}_results_template_v1",
        "status": "TBD",
        "seed": protocol.TRAINING_SEED,
        "training_regimes": list(protocol.DATASETS),
        "methods": list(protocol.METHODS),
        "checkpoint_roles": list(protocol.CHECKPOINT_ROLES),
        "reporting_roles": list(protocol.REPORTING_ROLES),
        "fa_budgets": list(protocol.FA_BUDGETS),
        "generated_tables": generated,
        "missing_values": "TBD",
        "no_experimental_result_generated": True,
        "no_fabricated_results": True,
        "test_selected_disclosure": protocol.expected_selection_disclosure(),
        "stability_claim_supported": False,
        "multiseed_replication_supported": False,
        "fixed_seed42_four_dataset_performance_supported": None,
        "paper_core_established": None,
    }
    template_path = Path(results_root) / "RESULTS_TEMPLATE.json"
    protocol.atomic_write_json(template_path, template, overwrite=overwrite)
    readme_path = Path(results_root) / "README.md"
    _atomic_write_text(
        readme_path,
        "\n".join(
            [
                "# Four-dataset seed-42 experiment results",
                "",
                "This directory is the formal output root. The current table "
                "cells are explicit `TBD` placeholders until audited experiment "
                "artifacts are available.",
                "",
                "- Seed: 42 only",
                "- Training regimes: SIRST3, NUAA-SIRST, NUDT-SIRST, IRSTD-1K",
                "- Methods: Original and Final, true scratch, 1000 epochs",
                "- Best checkpoint candidates: epochs 10,20,...,1000, threshold 0.5",
                "- Best results are test-selected and optimistic",
                "- `stability_claim_supported=false` (single seed)",
                "",
                "No numeric result in a `TBD` cell has been generated or inferred.",
                "",
            ]
        ),
        overwrite=overwrite,
    )
    return template


def _validate_fixed_result(
    payload: Mapping[str, Any],
    *,
    training_dataset: str,
    evaluation_dataset: str,
    method: str,
    role: str,
    manifest_record: Mapping[str, Any],
    source_subset: bool,
) -> dict[str, Any]:
    expected = {
        "training_dataset": training_dataset,
        "evaluation_dataset": evaluation_dataset,
        "method": method,
        "checkpoint_role": role,
        "seed": protocol.TRAINING_SEED,
        "normalization_dataset": training_dataset,
    }
    for field, value in expected.items():
        protocol.require(
            payload.get(field) == value,
            f"fixed result {training_dataset}/{evaluation_dataset}/"
            f"{method}/{role}: {field} differs",
        )
    checkpoint = payload.get("checkpoint")
    protocol.require(
        isinstance(checkpoint, Mapping),
        "fixed result lacks checkpoint metadata",
    )
    frozen = manifest_record["checkpoints"][role]
    for field in ("sha256", "epoch"):
        protocol.require(
            checkpoint.get(field) == frozen[field],
            f"fixed result checkpoint {field} differs from manifest",
        )
    fixed = payload.get("fixed_threshold_0_5")
    protocol.require(
        isinstance(fixed, Mapping),
        "fixed result lacks fixed_threshold_0_5",
    )
    protocol.require(
        float(fixed.get("threshold")) == protocol.FIXED_THRESHOLD,
        "fixed result threshold differs",
    )
    normalized_fixed = dict(fixed)
    protocol.validate_metric_point(normalized_fixed, allow_partial=False)
    if source_subset:
        protocol.require(
            payload.get("source_subset_of_selection") is True,
            "SIRST3 source result lacks subset disclosure",
        )
        protocol.require(
            payload.get("selection_parent") == "test_SIRST3",
            "SIRST3 source selection parent differs",
        )
        protocol.require(
            payload.get("checkpoint_reselected_for_source") is False,
            "SIRST3 source checkpoint was reselected",
        )
    return dict(payload)


def _manifest_index(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    records = manifest.get("records")
    protocol.require(isinstance(records, list), "manifest records are missing")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        protocol.require(isinstance(record, Mapping), "invalid manifest record")
        key = (str(record.get("dataset")), str(record.get("method")))
        protocol.require(key not in index, f"duplicate manifest record: {key}")
        index[key] = record
    expected = {
        (dataset, method)
        for dataset in protocol.DATASETS
        for method in protocol.METHODS
    }
    protocol.require(set(index) == expected, "manifest does not contain 8 runs")
    return index


def _load_fixed_path(
    results_root: Path,
    dataset: str,
    method: str,
    role: str,
) -> dict[str, Any]:
    path = (
        Path(results_root)
        / "evaluations"
        / "fixed_0_5"
        / dataset
        / method
        / f"{role}.json"
    )
    return protocol.load_json_object(path)


def _load_source_path(
    results_root: Path,
    source: str,
    method: str,
    role: str,
) -> dict[str, Any]:
    path = (
        Path(results_root)
        / "evaluations"
        / "sirst3_three_sources"
        / source
        / method
        / f"{role}.json"
    )
    return protocol.load_json_object(path)


def _load_sweep_path(
    results_root: Path,
    dataset: str,
    method: str,
    role: str,
) -> dict[str, Any]:
    path = (
        Path(results_root)
        / "evaluations"
        / "pd_fa_sweeps"
        / dataset
        / method
        / f"{role}.json"
    )
    return protocol.load_json_object(path)


def _load_source_sweep_path(
    results_root: Path,
    source: str,
    method: str,
    role: str,
) -> dict[str, Any]:
    path = (
        Path(results_root)
        / "evaluations"
        / "pd_fa_sweeps"
        / "sirst3_three_sources"
        / source
        / method
        / f"{role}.json"
    )
    return protocol.load_json_object(path)


def _validate_sweep_against_fixed(
    sweep: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    training_dataset: str,
    evaluation_dataset: str,
    method: str,
    role: str,
    source_subset: bool,
) -> dict[str, Any]:
    for field, expected in (
        ("training_dataset", training_dataset),
        ("evaluation_dataset", evaluation_dataset),
        ("method", method),
        ("checkpoint_role", role),
        ("seed", protocol.TRAINING_SEED),
        ("normalization_dataset", training_dataset),
    ):
        protocol.require(
            sweep.get(field) == expected,
            f"sweep {training_dataset}/{evaluation_dataset}/{method}/{role}: "
            f"{field} differs",
        )
    protocol.require(
        sweep.get("checkpoint") == fixed.get("checkpoint"),
        "sweep/fixed checkpoint metadata differs",
    )
    if source_subset:
        protocol.require(
            sweep.get("source_subset_of_selection") is True
            and sweep.get("selection_parent") == "test_SIRST3"
            and sweep.get("checkpoint_reselected_for_source") is False,
            "SIRST3 source sweep disclosure differs",
        )
    points = sweep.get("points")
    protocol.require(
        isinstance(points, list) and points,
        "sweep points are missing",
    )
    fixed_points = [
        point
        for point in points
        if float(point["threshold"]) == protocol.FIXED_THRESHOLD
    ]
    protocol.require(
        len(fixed_points) == 1,
        "sweep must contain exactly one threshold=0.5 point",
    )
    protocol.require(
        protocol.canonical_json_bytes(fixed_points[0])
        == protocol.canonical_json_bytes(fixed["fixed_threshold_0_5"]),
        "sweep threshold=0.5 differs from fixed evaluator",
    )
    expected_budgets = protocol.fa_budget_points(points)
    protocol.require(
        protocol.canonical_json_bytes(
            sweep.get("best_points_under_fa_budget")
        )
        == protocol.canonical_json_bytes(expected_budgets),
        "sweep Fa-budget points differ from recomputation",
    )
    protocol.require(
        protocol.canonical_json_bytes(sweep.get("pareto_frontier"))
        == protocol.canonical_json_bytes(protocol.pareto_frontier(points)),
        "sweep Pareto frontier differs from recomputation",
    )
    return dict(sweep)


def _paired_deltas(
    fixed_results: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = (
        "miou",
        "niou",
        "pixel_f1",
        "pd",
        "fa",
        "tiny_pd",
        "false_objects_per_image",
    )
    rows: list[dict[str, Any]] = []
    for dataset in protocol.DATASETS:
        for role in protocol.REPORTING_ROLES:
            original = fixed_results[
                (dataset, "original", role)
            ]["fixed_threshold_0_5"]
            final = fixed_results[
                (dataset, "final", role)
            ]["fixed_threshold_0_5"]
            deltas: dict[str, Any] = {}
            for key in keys:
                if original[key] is None or final[key] is None:
                    deltas[key] = None
                else:
                    deltas[key] = float(final[key]) - float(original[key])
            rows.append(
                {
                    "training_dataset": dataset,
                    "evaluation_dataset": dataset,
                    "checkpoint_role": role,
                    "delta_definition": "Final_minus_Original",
                    "deltas": deltas,
                }
            )
    return rows


def finalize(results_root: Path, *, overwrite: bool) -> dict[str, Any]:
    manifest_path = Path(results_root) / "selected_checkpoints" / (
        "checkpoint_manifest.json"
    )
    manifest = protocol.load_json_object(manifest_path)
    protocol.require(
        manifest.get("status") == "complete",
        "checkpoint manifest is not complete",
    )
    manifest_index = _manifest_index(manifest)

    fixed_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for dataset in protocol.DATASETS:
        for method in protocol.METHODS:
            manifest_record = manifest_index[(dataset, method)]
            for role in protocol.CHECKPOINT_ROLES:
                payload = _load_fixed_path(
                    results_root,
                    dataset,
                    method,
                    role,
                )
                fixed_results[(dataset, method, role)] = (
                    _validate_fixed_result(
                        payload,
                        training_dataset=dataset,
                        evaluation_dataset=dataset,
                        method=method,
                        role=role,
                        manifest_record=manifest_record,
                        source_subset=False,
                    )
                )
            endpoint = manifest_record.get("fixed_endpoint_epoch1000")
            protocol.require(
                isinstance(endpoint, Mapping),
                "manifest lacks fixed_endpoint_epoch1000",
            )
            protocol.require(
                endpoint.get("epoch") == protocol.EXPECTED_EPOCHS,
                "fixed endpoint epoch differs",
            )
            protocol.require(
                endpoint.get("checkpoint_saved") is False,
                "epoch-1000 endpoint must not be a saved checkpoint",
            )
            endpoint_metrics = endpoint.get("fixed_threshold_0_5_metrics")
            protocol.require(
                isinstance(endpoint_metrics, Mapping),
                "fixed endpoint metrics are missing",
            )
            protocol.validate_metric_point(
                endpoint_metrics,
                allow_partial=False,
            )
            fixed_results[(dataset, method, "last_epoch1000")] = {
                "training_dataset": dataset,
                "evaluation_dataset": dataset,
                "method": method,
                "checkpoint_role": "last_epoch1000",
                "epoch": protocol.EXPECTED_EPOCHS,
                "checkpoint_saved": False,
                "fixed_threshold_0_5": dict(endpoint_metrics),
                "test_selected": False,
                "selection_is_optimistic": False,
            }

    source_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in protocol.SOURCE_DATASETS:
        for method in protocol.METHODS:
            manifest_record = manifest_index[("SIRST3", method)]
            for role in protocol.CHECKPOINT_ROLES:
                payload = _load_source_path(
                    results_root,
                    source,
                    method,
                    role,
                )
                source_results[(source, method, role)] = (
                    _validate_fixed_result(
                        payload,
                        training_dataset="SIRST3",
                        evaluation_dataset=source,
                        method=method,
                        role=role,
                        manifest_record=manifest_record,
                        source_subset=True,
                    )
                )

    sweep_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for dataset in protocol.DATASETS:
        for method in protocol.METHODS:
            for role in protocol.SELECTED_ROLES:
                sweep = _load_sweep_path(
                    results_root,
                    dataset,
                    method,
                    role,
                )
                fixed = fixed_results[(dataset, method, role)]
                sweep_results[(dataset, method, role)] = (
                    _validate_sweep_against_fixed(
                        sweep,
                        fixed,
                        training_dataset=dataset,
                        evaluation_dataset=dataset,
                        method=method,
                        role=role,
                        source_subset=False,
                    )
                )

    source_sweep_results: dict[
        tuple[str, str, str], dict[str, Any]
    ] = {}
    for source in protocol.SOURCE_DATASETS:
        for method in protocol.METHODS:
            for role in protocol.CHECKPOINT_ROLES:
                sweep = _load_source_sweep_path(
                    results_root,
                    source,
                    method,
                    role,
                )
                fixed = source_results[(source, method, role)]
                source_sweep_results[(source, method, role)] = (
                    _validate_sweep_against_fixed(
                        sweep,
                        fixed,
                        training_dataset="SIRST3",
                        evaluation_dataset=source,
                        method=method,
                        role=role,
                        source_subset=True,
                    )
                )

    def fixed_loader(
        dataset: str,
        method: str,
        role: str,
    ) -> Mapping[str, Any]:
        return fixed_results[(dataset, method, role)]

    def source_loader(
        source: str,
        method: str,
        role: str,
    ) -> Mapping[str, Any]:
        return source_results[(source, method, role)]

    tables = Path(results_root) / "tables"
    generated: dict[str, Any] = {}
    for stem, title, rows in (
        (
            "table2_best_miou",
            "Table 2: dataset-specific best-mIoU checkpoints",
            _dataset_rows("best_miou", fixed_loader),
        ),
        (
            "table4a_best_pd",
            "Table 4a: dataset-specific best-Pd checkpoints",
            _dataset_rows("best_pd", fixed_loader),
        ),
        (
            "table4b_last_epoch1000",
            "Table 4b: fixed epoch-1000 metric endpoints (no checkpoint saved)",
            _dataset_rows("last_epoch1000", fixed_loader),
        ),
    ):
        generated[stem] = _write_table_pair(
            tables,
            stem,
            title,
            rows,
            overwrite=overwrite,
        )
    for role, suffix in (
        ("best_miou", "a_best_miou"),
        ("best_pd", "b_best_pd"),
    ):
        stem = f"table3{suffix}_sirst3_three_sources"
        generated[stem] = _write_table_pair(
            tables,
            stem,
            f"Table 3{suffix[0]}: one SIRST3 checkpoint on three sources",
            _source_rows(role, source_loader),
            overwrite=overwrite,
        )

    budget_rows: list[dict[str, Any]] = []
    for dataset in protocol.DATASETS:
        for method in protocol.METHODS:
            sweep = sweep_results[(dataset, method, "best_miou")]
            row = {
                "training_dataset": dataset,
                "evaluation_dataset": dataset,
                "method": protocol.METHOD_LABELS[method],
                "checkpoint_role": "best_miou",
            }
            for budget in protocol.FA_BUDGETS:
                key = f"{budget:.10g}"
                point = sweep["best_points_under_fa_budget"][key]
                row[key] = None if point is None else point["pd"]
            budget_rows.append(row)
    budget_path = tables / "table7_pd_at_fa_budgets.md"
    _atomic_write_text(
        budget_path,
        _budget_table_text(budget_rows),
        overwrite=overwrite,
    )
    generated["table7_pd_at_fa_budgets"] = {
        "markdown": {
            "path": str(budget_path.resolve()),
            "sha256": protocol.file_sha256(budget_path),
        }
    }

    deltas = _paired_deltas(fixed_results)
    delta_path = Path(results_root) / "paired_deltas_final_minus_original.json"
    protocol.atomic_write_json(
        delta_path,
        {
            "schema": f"{protocol.SCHEMA_PREFIX}_paired_deltas_v1",
            "seed": protocol.TRAINING_SEED,
            "rows": deltas,
            "stability_claim_supported": False,
        },
        overwrite=overwrite,
    )
    summary = {
        "schema": f"{protocol.SCHEMA_PREFIX}_paper_results_v1",
        "status": "complete",
        "seed": protocol.TRAINING_SEED,
        "checkpoint_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": protocol.file_sha256(manifest_path),
        },
        "dataset_specific_fixed_result_count": len(fixed_results),
        "sirst3_three_source_fixed_result_count": len(source_results),
        "dataset_specific_sweep_count": len(sweep_results),
        "sirst3_three_source_sweep_count": len(source_sweep_results),
        "generated_tables": generated,
        "paired_delta_artifact": {
            "path": str(delta_path.resolve()),
            "sha256": protocol.file_sha256(delta_path),
        },
        "test_selected_disclosure": protocol.expected_selection_disclosure(),
        "fixed_seed42_four_dataset_performance_supported": None,
        "paper_core_established": None,
        "stability_claim_supported": False,
        "multiseed_replication_supported": False,
        "claim_boundary": (
            "Single-seed, test-selected evidence. Numerical completion does "
            "not by itself establish multi-seed stability."
        ),
        "no_fabricated_results": True,
    }
    summary_path = Path(results_root) / "paper_results_summary.json"
    protocol.atomic_write_json(summary_path, summary, overwrite=overwrite)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.initialize_templates:
        payload = initialize_templates(
            args.results_root,
            overwrite=args.overwrite,
        )
        action = "initialized"
        artifact = Path(args.results_root) / "RESULTS_TEMPLATE.json"
    else:
        payload = finalize(args.results_root, overwrite=args.overwrite)
        action = "finalized"
        artifact = Path(args.results_root) / "paper_results_summary.json"
    print(
        json.dumps(
            {
                "status": action,
                "artifact": str(artifact.resolve()),
                "sha256": protocol.file_sha256(artifact),
                "no_fabricated_results": payload["no_fabricated_results"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
