#!/usr/bin/env python3
"""Finalize DCH formal800 without conflating performance and mechanism.

Gate A--E alone controls authorization of the next five-node NER engineering
stage.  Mechanism Audit M is reported alongside that decision but can neither
rescue a failed gate nor veto a passed engineering gate.  This entry is
read-only unless ``--write`` is explicitly selected, and every output is
exclusive-create.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
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
from experiments import summarize_tpd_clean_v7_dch_formal800 as summary  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v7_dch_final_decision_v1"
JSON_OUTPUT_NAME = "tpd_clean_v7_dch_final_decision.json"
MARKDOWN_OUTPUT_NAME = "tpd_clean_v7_dch_final_decision.md"
DEFAULT_COMPARISON_REPORT = (
    summary.DEFAULT_OUTPUT_DIR / summary.JSON_OUTPUT_NAME
)
DEFAULT_MECHANISM_REPORT = (
    summary.DEFAULT_OUTPUT_DIR
    / "tpd_clean_v7_dch_mechanism_audit.json"
)
DEFAULT_OUTPUT_DIR = summary.DEFAULT_OUTPUT_DIR
DEFAULT_ACCEPTANCE_SOURCE_LOCK = summary.DEFAULT_ACCEPTANCE_SOURCE_LOCK


class FinalizationError(RuntimeError):
    """Raised when a final decision input is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    path = Path(path)
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} is missing or non-regular: {path}",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{label} is invalid JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _validate_comparison(
    comparison: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = copy.deepcopy(dict(comparison))
    _require(
        payload.get("schema") == summary.SCHEMA
        and payload.get("status") == "complete",
        "DCH comparison schema/status differs",
    )
    _require(
        payload.get("candidate_family") == "tpd_clean_v7_dch"
        and payload.get("variants") == list(summary.VARIANTS)
        and payload.get("seeds") == list(summary.SEEDS),
        "DCH comparison identity differs",
    )
    _require(
        payload.get("formal_artifact_counts")
        == {"runs": 4, "checkpoints": 12, "sweeps": 8},
        "DCH comparison artifact counts differ",
    )
    _require(
        payload.get("validation_fields")
        == list(summary.VALIDATION_FIELDS),
        "DCH comparison does not bind the native 17 fields",
    )
    runs, _ = summary._normalize_runs(payload.get("candidate_runs"))
    integrity = payload.get("engineering_integrity")
    _require(
        isinstance(integrity, Mapping),
        "DCH comparison engineering integrity is missing",
    )
    spd = payload.get("frozen_spd_reference")
    _require(
        isinstance(spd, Mapping),
        "DCH comparison frozen SPD reference is missing",
    )
    recomputed = summary.evaluate_engineering_gates(runs, spd, integrity)
    _require(
        recomputed == payload.get("engineering_gate"),
        "DCH comparison Gate A--E payload is not reproducible",
    )
    gate_passed = bool(recomputed["passed"])
    _require(
        payload.get("engineering_gate_passed") is gate_passed,
        "DCH comparison gate boolean differs",
    )
    _require(
        payload.get("ner_stage_authorized") is gate_passed,
        "DCH comparison NER authorization differs from Gate A--E",
    )
    _require(
        payload.get("decision")
        == (
            "ENGINEERING_GATE_PASS"
            if gate_passed
            else "ENGINEERING_GATE_FAIL"
        ),
        "DCH comparison decision differs from Gate A--E",
    )
    _require(
        payload.get("mainline_contract") == "Keep-Context-Saliency"
        and payload.get("mainline_changed") is False
        and payload.get("paper_core_established") is False
        and payload.get("stability_claim_supported") is False,
        "DCH comparison claim boundary differs",
    )
    return payload


def _validate_mechanism(
    mechanism: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = copy.deepcopy(dict(mechanism))
    _require(
        payload.get("schema")
        == "sctransnet_tpd_clean_v7_dch_mechanism_audit_v1"
        and payload.get("status") == "complete",
        "Mechanism Audit M is not complete",
    )
    _require(
        payload.get("candidate_family") == "tpd_clean_v7_dch",
        "Mechanism Audit M candidate identity differs",
    )
    _require(
        payload.get("dataset") == summary.DATASET
        and payload.get("variants") == list(summary.VARIANTS)
        and payload.get("seeds") == list(summary.SEEDS),
        "Mechanism Audit M dataset/variant/seed identity differs",
    )
    counts = payload.get("artifact_counts")
    _require(
        isinstance(counts, Mapping)
        and counts.get("runs") == 4
        and counts.get("checkpoints") == 12,
        "Mechanism Audit M must bind 4 runs and 12 checkpoints",
    )
    directions = payload.get("directions")
    _require(
        isinstance(directions, Mapping)
        and {
            "fragment_excess_total",
            "in_gt_unmatched_pixels",
            "split_target",
            "largest_fragment",
        }.issubset(directions),
        "Mechanism Audit M directions are incomplete",
    )
    supported = payload.get("fragmentation_mechanism_claim_supported")
    _require(
        isinstance(supported, bool),
        "Mechanism Audit M claim result must be boolean",
    )
    _require(
        payload.get("mechanism_audit_replaces_performance_gates") is False,
        "Mechanism Audit M cannot replace Gate A--E",
    )
    _require(
        payload.get("native_validation_fields")
        == list(summary.VALIDATION_FIELDS)
        and payload.get("native_validation_field_count")
        == len(summary.VALIDATION_FIELDS),
        "Mechanism Audit M native 17-field identity differs",
    )
    return payload


def derive_final_decision(
    comparison: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Purely combine two independently validated evidence layers."""

    comparison_payload = _validate_comparison(comparison)
    mechanism_payload = _validate_mechanism(mechanism)
    gate_passed = bool(comparison_payload["engineering_gate_passed"])
    mechanism_supported = bool(
        mechanism_payload["fragmentation_mechanism_claim_supported"]
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_family": "tpd_clean_v7_dch",
        "dataset": summary.DATASET,
        "official_test_accessed": False,
        "mainline_contract": "Keep-Context-Saliency",
        "mainline_changed": False,
        "formal_artifact_counts": {
            "runs": 4,
            "checkpoints": 12,
            "sweeps": 8,
            "mechanism_checkpoint_audits": 12,
        },
        "validation_fields": list(summary.VALIDATION_FIELDS),
        "decision": (
            "ENGINEERING_GATE_PASS"
            if gate_passed
            else "ENGINEERING_GATE_FAIL"
        ),
        "engineering_gate_passed": gate_passed,
        "ner_stage_authorized": gate_passed,
        "fragmentation_mechanism_claim_supported": mechanism_supported,
        "mechanism_decision": (
            "MECHANISM_SUPPORTED"
            if mechanism_supported
            else "MECHANISM_NOT_SUPPORTED"
        ),
        "mechanism_audit_replaces_performance_gates": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "authoritative_result_accepted": True,
        "comparison_decision": comparison_payload["decision"],
        "comparison_gate": copy.deepcopy(
            comparison_payload["engineering_gate"]
        ),
        "mechanism_directions": copy.deepcopy(
            mechanism_payload["directions"]
        ),
        "bindings": copy.deepcopy(dict(bindings or {})),
    }


def inspect_readiness(
    *,
    comparison_path: Path = DEFAULT_COMPARISON_REPORT,
    mechanism_path: Path = DEFAULT_MECHANISM_REPORT,
    acceptance_source_lock: Path = DEFAULT_ACCEPTANCE_SOURCE_LOCK,
) -> Dict[str, Any]:
    """Return readiness without creating or evaluating missing outputs."""

    paths = {
        "comparison": Path(comparison_path),
        "mechanism_audit": Path(mechanism_path),
        "acceptance_source_lock": Path(acceptance_source_lock),
    }
    presence = {
        key: path.is_file() and not path.is_symlink()
        for key, path in paths.items()
    }
    acceptance_validation = {
        "expected_schema": source_locks.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
        "valid_current_lock": False,
        "sha256": None,
        "error": None,
    }
    if presence["acceptance_source_lock"]:
        try:
            payload, digest = source_locks.validate_source_lock(
                "acceptance",
                paths["acceptance_source_lock"],
                repo_root=REPO_ROOT,
            )
            acceptance_validation.update(
                {
                    "observed_schema": payload["schema"],
                    "valid_current_lock": True,
                    "sha256": digest,
                }
            )
        except (FileNotFoundError, ValueError) as exc:
            acceptance_validation["error"] = str(exc)
    ready = (
        presence["comparison"]
        and presence["mechanism_audit"]
        and acceptance_validation["valid_current_lock"] is True
    )
    return {
        "schema": "sctransnet_tpd_clean_v7_dch_finalizer_preflight_v1",
        "mode": "preflight",
        "ready": ready,
        "inputs": {
            key: {
                "path": str(path.resolve()),
                "present": presence[key],
                **(
                    {"validation": acceptance_validation}
                    if key == "acceptance_source_lock"
                    else {}
                ),
            }
            for key, path in paths.items()
        },
        "writes_performed": 0,
        "authoritative_result_accepted": False,
        "ner_stage_authorized": False,
    }


def build_final_report(
    *,
    comparison_path: Path = DEFAULT_COMPARISON_REPORT,
    mechanism_path: Path = DEFAULT_MECHANISM_REPORT,
    acceptance_source_lock: Path = DEFAULT_ACCEPTANCE_SOURCE_LOCK,
) -> Dict[str, Any]:
    """Validate the lock and both reports, then derive the final decision."""

    lock_payload, lock_sha256 = source_locks.validate_source_lock(
        "acceptance",
        Path(acceptance_source_lock),
        repo_root=REPO_ROOT,
    )
    comparison = _load_json(comparison_path, "DCH comparison report")
    from analysis import diagnose_tpd_clean_v7_dch_mechanism as mechanism_module

    mechanism = mechanism_module.validate_mechanism_report(
        Path(mechanism_path)
    )
    bindings = {
        "acceptance_source_lock": str(
            Path(acceptance_source_lock).resolve()
        ),
        "acceptance_source_lock_schema": lock_payload["schema"],
        "acceptance_source_lock_sha256": lock_sha256,
        "acceptance_source_count": lock_payload["source_count"],
        "comparison_report": str(Path(comparison_path).resolve()),
        "comparison_report_sha256": summary.sha256_file(comparison_path),
        "mechanism_report": str(Path(mechanism_path).resolve()),
        "mechanism_report_sha256": summary.sha256_file(mechanism_path),
        "input_count": 3,
    }
    return derive_final_decision(
        comparison,
        mechanism,
        bindings=bindings,
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TPD-Clean V7-DCH final decision",
            "",
            f"- Decision: `{report['decision']}`",
            (
                "- NER stage authorized: "
                f"`{str(report['ner_stage_authorized']).lower()}`"
            ),
            (
                "- Fragmentation mechanism supported: "
                f"`{str(report['fragmentation_mechanism_claim_supported']).lower()}`"
            ),
            "- Mainline: `Keep-Context-Saliency` (unchanged)",
            "- Paper core established: `false`",
            "- Stability claim supported: `false`",
            "",
            "Gate A–E and Mechanism Audit M remain independent. A mechanism "
            "result cannot override the engineering-gate decision.",
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


def _write_new(path: Path, content: bytes) -> Path:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite final report: {path}")
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


def write_final_report_once(
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
        raise FileExistsError("refusing to overwrite DCH final outputs")
    json_bytes = (
        json.dumps(
            dict(report),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_new(json_path, json_bytes)
    try:
        _write_new(
            markdown_path,
            render_markdown(report).encode("utf-8"),
        )
    except BaseException:
        json_path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize DCH Gate A--E and Mechanism Audit M"
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
                inspect_readiness(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return
    readiness = inspect_readiness()
    if readiness["ready"] is not True:
        raise SystemExit(
            "DCH finalization inputs are incomplete; use --preflight"
        )
    report = build_final_report()
    paths = write_final_report_once(report)
    print(
        f"WROTE decision={report['decision']} "
        f"json={paths[0]} markdown={paths[1]}",
        flush=True,
    )


__all__ = [
    "DEFAULT_ACCEPTANCE_SOURCE_LOCK",
    "DEFAULT_COMPARISON_REPORT",
    "DEFAULT_MECHANISM_REPORT",
    "DEFAULT_OUTPUT_DIR",
    "FinalizationError",
    "JSON_OUTPUT_NAME",
    "MARKDOWN_OUTPUT_NAME",
    "SCHEMA",
    "build_final_report",
    "derive_final_decision",
    "inspect_readiness",
    "main",
    "parse_args",
    "render_markdown",
    "write_final_report_once",
]


if __name__ == "__main__":
    main()
