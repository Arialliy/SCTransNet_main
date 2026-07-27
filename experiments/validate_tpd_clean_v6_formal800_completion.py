#!/usr/bin/env python3
"""Publish or verify the exact V6 formal800 completion transaction.

The summarizer owns the comparison JSON/Markdown.  This validator re-derives
their substantive content, binds the exact input files in a manifest, and
publishes a final digest marker.  It never replaces an existing report,
manifest, or marker.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402


MANIFEST_SCHEMA = "sctransnet_tpd_clean_v6_completion_inputs_v1"
MANIFEST_NAME = "completion_inputs.json"
MARKER_NAME = "COMPLETE.sha256"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise summary.IncompleteArtifact(f"{label}: missing regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise summary.IncompleteArtifact(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise summary.IncompleteArtifact(f"{label}: expected object")
    return value


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise summary.IncompleteArtifact(
            f"completion input lies outside repository: {resolved}"
        ) from exc


def _input_paths(
    report: Mapping[str, Any],
    *,
    postprocess_source_lock: Path = summary.DEFAULT_POSTPROCESS_SOURCE_LOCK,
) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = [
        (
            "training_source_lock",
            "source_lock",
            summary.DEFAULT_TRAINING_SOURCE_LOCK,
        ),
        ("postprocess_source_lock", "source_lock", postprocess_source_lock),
    ]
    for path in summary.SPD_REFERENCE_FILES:
        records.append(
            (
                f"frozen_spd:{path.name}",
                "frozen_reference",
                path,
            )
        )
    for name in ("cpu_all.json", "gpu2_full.json", "gpu3_capacity.json"):
        records.append(
            (
                f"smoke:{name}",
                "smoke_report",
                summary.DEFAULT_SMOKE_ROOT / name,
            )
        )
    records.append(
        (
            "smoke:v6_smoke_verification.json",
            "smoke_verification",
            summary.SMOKE_VERIFICATION,
        )
    )
    for seed in summary.SEEDS:
        for variant in summary.VARIANTS:
            key = f"{variant}/seed_{seed}"
            run = report["candidate_runs"][key]
            run_dir = Path(run["run_directory"])
            for name in ("protocol.json", "split.json", "summary.json", "metrics.jsonl"):
                records.append((f"{key}:{name}", "candidate_training", run_dir / name))
            for name in ("best.pth.tar", "best_miou.pth.tar", "last.pth.tar"):
                records.append((f"{key}:{name}", "candidate_checkpoint", run_dir / name))
            for spec in summary.ROLE_SPECS.values():
                name = spec["sweep"]
                records.append((f"{key}:{name}", "candidate_sweep", run_dir / name))
            records.append(
                (
                    f"{key}:worker_log",
                    "worker_log",
                    Path(run["worker_log"]["path"]),
                )
            )
            for name in (
                "active.json",
                "slot_a.metrics.jsonl",
                "slot_a.exact.pth",
                "slot_b.metrics.jsonl",
                "slot_b.exact.pth",
            ):
                records.append(
                    (
                        f"{key}:exact_journal/{name}",
                        "exact_journal",
                        run_dir / "exact_journal" / name,
                    )
                )
    identifiers = [identifier for identifier, _, _ in records]
    paths = [Path(path).resolve() for _, _, path in records]
    if len(identifiers) != len(set(identifiers)):
        raise summary.IncompleteArtifact("duplicate completion input identifier")
    if len(paths) != len(set(paths)):
        raise summary.IncompleteArtifact("duplicate completion input path")
    return records


def build_manifest(
    report: Mapping[str, Any],
    *,
    postprocess_source_lock: Path = summary.DEFAULT_POSTPROCESS_SOURCE_LOCK,
) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    for identifier, category, path in _input_paths(
        report, postprocess_source_lock=postprocess_source_lock
    ):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise summary.IncompleteArtifact(
                f"completion input is not a regular file: {path}"
            )
        category_counts[category] = category_counts.get(category, 0) + 1
        inputs.append(
            {
                "id": identifier,
                "category": category,
                "path": _relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": summary.sha256_file(path),
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "candidate_root": _relative(summary.DEFAULT_CANDIDATE_ROOT),
        "training_source_lock_sha256": summary.EXPECTED_TRAINING_LOCK_SHA256,
        "training_data_sha256": summary.EXPECTED_TRAINING_DATA_SHA256,
        "postprocess_source_lock_sha256": summary.sha256_file(
            postprocess_source_lock
        ),
        "input_count": len(inputs),
        "category_counts": category_counts,
        "inputs": inputs,
    }


def _without_generated_at(report: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(report))
    normalized.pop("generated_at_utc", None)
    return normalized


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
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


def _report_paths(
    output_dir: Path = summary.DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path]:
    root = Path(output_dir)
    return (
        root / summary.JSON_OUTPUT_NAME,
        root / summary.MARKDOWN_OUTPUT_NAME,
        root / MANIFEST_NAME,
        root / MARKER_NAME,
    )


def validate_published_report(
    output_dir: Path = summary.DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    json_path, markdown_path, _, _ = _report_paths(output_dir)
    published = _load_json(json_path, "published V6 report")
    if (
        published.get("schema") != summary.SCHEMA
        or published.get("status") != "complete"
        or published.get("gate_evaluated") is not True
    ):
        raise summary.IncompleteArtifact("published V6 report status differs")
    derived = summary.build_report()
    if _without_generated_at(published) != _without_generated_at(derived):
        raise summary.IncompleteArtifact(
            "published V6 report differs from current exact inputs"
        )
    expected_markdown = summary.render_markdown(published).encode("utf-8")
    if not markdown_path.is_file() or markdown_path.is_symlink():
        raise summary.IncompleteArtifact("published Markdown is missing")
    if markdown_path.read_bytes() != expected_markdown:
        raise summary.IncompleteArtifact(
            "published Markdown differs from the report JSON"
        )
    return published


def _marker_bytes(
    json_path: Path, markdown_path: Path, manifest_path: Path
) -> bytes:
    rows = [
        f"{summary.sha256_file(path)}  {path.name}"
        for path in (json_path, markdown_path, manifest_path)
    ]
    return ("\n".join(rows) + "\n").encode("utf-8")


def publish_completion(
    output_dir: Path = summary.DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    json_path, markdown_path, manifest_path, marker_path = _report_paths(
        output_dir
    )
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    if marker_path.exists() or marker_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite marker: {marker_path}")
    report = validate_published_report(output_dir)
    manifest = build_manifest(report)
    summary.write_new(manifest_path, _canonical_json_bytes(manifest))
    try:
        summary.write_new(
            marker_path,
            _marker_bytes(json_path, markdown_path, manifest_path),
        )
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise
    return {
        "status": "complete",
        "decision": report["decision"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": summary.sha256_file(manifest_path),
        "marker": str(marker_path.resolve()),
        "marker_sha256": summary.sha256_file(marker_path),
        "input_count": manifest["input_count"],
    }


def verify_completion(
    output_dir: Path = summary.DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    json_path, markdown_path, manifest_path, marker_path = _report_paths(
        output_dir
    )
    report = validate_published_report(output_dir)
    published_manifest = _load_json(manifest_path, "completion input manifest")
    expected_manifest = build_manifest(report)
    if published_manifest != expected_manifest:
        raise summary.IncompleteArtifact(
            "completion manifest differs from current exact inputs"
        )
    if not marker_path.is_file() or marker_path.is_symlink():
        raise summary.IncompleteArtifact("completion marker is missing")
    expected_marker = _marker_bytes(json_path, markdown_path, manifest_path)
    if marker_path.read_bytes() != expected_marker:
        raise summary.IncompleteArtifact("completion marker digest set differs")
    return {
        "status": "complete",
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
        "ner_stage_authorized": report["ner_stage_authorized"],
        "input_count": published_manifest["input_count"],
        "manifest_sha256": summary.sha256_file(manifest_path),
        "marker_sha256": summary.sha256_file(marker_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish or verify V6 formal800 completion"
    )
    parser.add_argument(
        "mode",
        choices=("preflight", "publish", "verify"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "preflight":
        result = summary.inspect_training_readiness()
    elif args.mode == "publish":
        readiness = summary.inspect_training_readiness()
        if readiness["formal_matrix_complete"] is not True:
            raise SystemExit(
                "V6 formal800 matrix is incomplete; only preflight is allowed"
            )
        result = publish_completion()
    else:
        result = verify_completion()
    print(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
        flush=True,
    )


__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "MARKER_NAME",
    "build_manifest",
    "main",
    "parse_args",
    "publish_completion",
    "validate_published_report",
    "verify_completion",
]


if __name__ == "__main__":
    main()
