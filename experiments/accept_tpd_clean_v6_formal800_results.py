#!/usr/bin/env python3
"""Authoritative read-only acceptance entry for V6 formal800 results.

A result is accepted only when both independent layers pass:

1. the frozen formal completion verifier;
2. the supplemental complete-grid and point-identity sweep verifier.

This wrapper never creates or replaces experiment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402
from experiments import validate_tpd_clean_v6_formal800_completion as completion  # noqa: E402
from experiments import validate_tpd_clean_v6_strict_sweeps as strict  # noqa: E402


SUPPLEMENTAL_SOURCE_LOCK = (
    REPO_ROOT
    / "experiments/tpd_clean_v6_supplemental_acceptance_source_lock.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_supplemental_source_lock() -> str:
    _require(
        SUPPLEMENTAL_SOURCE_LOCK.is_file()
        and not SUPPLEMENTAL_SOURCE_LOCK.is_symlink(),
        "supplemental acceptance source lock is missing",
    )
    payload = json.loads(SUPPLEMENTAL_SOURCE_LOCK.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("schema")
        == "sctransnet_tpd_clean_v6_supplemental_acceptance_source_lock_v1",
        "supplemental acceptance source-lock schema differs",
    )
    _require(
        payload.get("training_source_lock_sha256")
        == _sha256(summary.DEFAULT_TRAINING_SOURCE_LOCK),
        "supplemental lock training binding differs",
    )
    _require(
        payload.get("postprocess_source_lock_sha256")
        == _sha256(summary.DEFAULT_POSTPROCESS_SOURCE_LOCK),
        "supplemental lock postprocess binding differs",
    )
    sources = payload.get("source_sha256")
    _require(
        isinstance(sources, dict)
        and payload.get("source_count") == len(sources)
        and set(sources)
        == {
            "experiments/accept_tpd_clean_v6_formal800_results.py",
            "experiments/validate_tpd_clean_v6_strict_sweeps.py",
            "tests/test_accept_tpd_clean_v6_formal800_results.py",
            "tests/test_validate_tpd_clean_v6_strict_sweeps.py",
        },
        "supplemental source set differs",
    )
    for relative, expected in sources.items():
        path = REPO_ROOT / relative
        _require(
            path.is_file()
            and not path.is_symlink()
            and _sha256(path) == expected,
            f"supplemental source differs: {relative}",
        )
    return _sha256(SUPPLEMENTAL_SOURCE_LOCK)


def preflight() -> dict[str, Any]:
    supplemental_lock_sha256 = validate_supplemental_source_lock()
    training = summary.inspect_training_readiness()
    strict_sweeps = strict.inspect_strict_sweeps(
        summary.DEFAULT_CANDIDATE_ROOT
    )
    output_dir = summary.DEFAULT_OUTPUT_DIR
    expected_outputs = {
        summary.JSON_OUTPUT_NAME: output_dir / summary.JSON_OUTPUT_NAME,
        summary.MARKDOWN_OUTPUT_NAME: output_dir / summary.MARKDOWN_OUTPUT_NAME,
        completion.MANIFEST_NAME: output_dir / completion.MANIFEST_NAME,
        completion.MARKER_NAME: output_dir / completion.MARKER_NAME,
    }
    outputs = {
        name: path.is_file() and not path.is_symlink()
        for name, path in expected_outputs.items()
    }
    return {
        "schema": "sctransnet_tpd_clean_v6_result_acceptance_v1",
        "mode": "preflight",
        "training": training,
        "strict_sweeps": strict_sweeps,
        "completion_outputs": outputs,
        "supplemental_source_lock_sha256": supplemental_lock_sha256,
        "authoritative_result_accepted": False,
        "ner_stage_authorized": False,
    }


def verify_and_accept() -> dict[str, Any]:
    supplemental_lock_sha256 = validate_supplemental_source_lock()
    completion_result = completion.verify_completion(
        summary.DEFAULT_OUTPUT_DIR
    )
    strict_result = strict.validate_all_strict_sweeps(
        summary.DEFAULT_CANDIDATE_ROOT
    )
    _require(
        completion_result.get("status") == "complete",
        "frozen completion verification did not complete",
    )
    _require(
        strict_result.get("complete_and_strict_valid") is True
        and strict_result.get("strict_valid_sweeps") == 8,
        "supplemental strict sweep verification did not pass 8/8",
    )
    engineering_gate_passed = completion_result.get(
        "engineering_gate_passed"
    )
    ner_stage_authorized = completion_result.get("ner_stage_authorized")
    _require(
        isinstance(engineering_gate_passed, bool),
        "engineering gate result is not boolean",
    )
    _require(
        isinstance(ner_stage_authorized, bool),
        "NER authorization result is not boolean",
    )
    _require(
        not ner_stage_authorized or engineering_gate_passed,
        "NER was authorized without the full engineering gate",
    )
    return {
        "schema": "sctransnet_tpd_clean_v6_result_acceptance_v1",
        "mode": "verify",
        "authoritative_result_accepted": True,
        "decision": completion_result["decision"],
        "engineering_gate_passed": engineering_gate_passed,
        "ner_stage_authorized": ner_stage_authorized,
        "strict_valid_sweeps": strict_result["strict_valid_sweeps"],
        "supplemental_source_lock_sha256": supplemental_lock_sha256,
        "completion_input_count": completion_result["input_count"],
        "completion_manifest_sha256": completion_result[
            "manifest_sha256"
        ],
        "completion_marker_sha256": completion_result["marker_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accept V6 formal800 results only after both verifier layers"
    )
    parser.add_argument("mode", choices=("preflight", "verify"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = preflight() if args.mode == "preflight" else verify_and_accept()
    print(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
