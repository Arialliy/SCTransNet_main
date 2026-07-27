#!/usr/bin/env python3
"""Final read-only V6 acceptance including post-freeze adapter provenance.

The frozen two-layer acceptance entry always runs first.  This third layer
then requires the independently locked compatibility metadata in all eight
sweeps.  No experiment artifact is created or replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import accept_tpd_clean_v6_formal800_results as frozen_acceptance  # noqa: E402
from experiments import validate_tpd_clean_v6_checkpoint_compatibility as compat_validation  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v6_compatibility_result_acceptance_v1"


def preflight() -> dict[str, Any]:
    frozen_result = frozen_acceptance.preflight()
    compatibility_result = (
        compat_validation.inspect_compatibility_sweeps()
    )
    return {
        "schema": SCHEMA,
        "mode": "preflight",
        "frozen_acceptance": frozen_result,
        "compatibility_validation": compatibility_result,
        "authoritative_result_accepted": False,
        "compatibility_valid_sweeps": compatibility_result["valid_sweeps"],
    }


def verify_and_accept() -> dict[str, Any]:
    """Run frozen acceptance first, then require compatibility 8/8."""

    frozen_result = frozen_acceptance.verify_and_accept()
    if frozen_result.get("authoritative_result_accepted") is not True:
        raise RuntimeError("frozen acceptance did not accept the V6 result")
    compatibility_result = (
        compat_validation.validate_all_compatibility_sweeps()
    )
    if (
        compatibility_result.get("complete_and_compatibility_valid")
        is not True
        or compatibility_result.get("valid_sweeps") != 8
    ):
        raise RuntimeError("checkpoint compatibility validation did not pass 8/8")
    return {
        "schema": SCHEMA,
        "mode": "verify",
        "authoritative_result_accepted": True,
        "frozen_acceptance_ran_first": True,
        "decision": frozen_result["decision"],
        "engineering_gate_passed": frozen_result[
            "engineering_gate_passed"
        ],
        "ner_stage_authorized": frozen_result["ner_stage_authorized"],
        "strict_valid_sweeps": frozen_result["strict_valid_sweeps"],
        "compatibility_valid_sweeps": compatibility_result["valid_sweeps"],
        "supplemental_source_lock_sha256": frozen_result[
            "supplemental_source_lock_sha256"
        ],
        "compatibility_source_lock_sha256": compatibility_result[
            "compatibility_source_lock_sha256"
        ],
        "completion_manifest_sha256": frozen_result[
            "completion_manifest_sha256"
        ],
        "completion_marker_sha256": frozen_result[
            "completion_marker_sha256"
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accept V6 results after frozen and compatibility checks"
    )
    parser.add_argument("mode", choices=("preflight", "verify"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = preflight() if args.mode == "preflight" else verify_and_accept()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


__all__ = [
    "SCHEMA",
    "main",
    "parse_args",
    "preflight",
    "verify_and_accept",
]


if __name__ == "__main__":
    main()
