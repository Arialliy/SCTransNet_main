#!/usr/bin/env python3
"""Read-only V2 adjudication followed by an explicit V3 launcher handoff.

The default and ``--status`` paths never start a process.  ``--execute`` is
the only mutating mode, and even then this module delegates to the frozen V3
launcher instead of constructing a training command or touching a V1/V2
unit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v3_source_locks as v3_freeze,
)
from experiments import (  # noqa: E402
    handoff_tpd_ner_v8_v1_to_v2 as v2_handoff,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_v2_formal800 as v2_post,
)


SCHEMA = "sctransnet_tpd_ner_v8_v2_to_v3_handoff_v1"
READY_DECISION = "RETURN_TO_MODEL_OPTIMIZATION"
NOT_NEEDED_DECISION = "FULL_MODEL_GATE_PASSED"
V3_VARIANT = "tpd_ner_v8_mprs_dch_v3_full_relay_on"
V3_LAUNCHER = (
    REPO_ROOT
    / "experiments/launch_tpd_ner_v8_mprs_dch_v3_formal800_1x5090.sh"
)
PHYSICAL_GPU_UUIDS = {
    2: "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    3: "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
Runner = Callable[..., subprocess.CompletedProcess[Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_v3_closure() -> dict[str, Any]:
    training = v3_freeze.verify_training_lock(
        v3_freeze.DEFAULT_TRAINING_LOCK,
        dataset_dir=REPO_ROOT / "datasets",
    )
    acceptance = v3_freeze.verify_acceptance_lock(
        v3_freeze.DEFAULT_ACCEPTANCE_LOCK,
        v3_freeze.DEFAULT_TRAINING_LOCK,
        dataset_dir=REPO_ROOT / "datasets",
        upstream_v2_training_lock=(
            v3_freeze.UPSTREAM_V2_TRAINING_LOCK
        ),
        upstream_v2_acceptance_lock=(
            v3_freeze.UPSTREAM_V2_ACCEPTANCE_LOCK
        ),
    )
    return {
        "v3_training_source_lock_sha256": (
            v3_freeze.file_sha256(v3_freeze.DEFAULT_TRAINING_LOCK)
        ),
        "v3_acceptance_source_lock_sha256": (
            v3_freeze.file_sha256(v3_freeze.DEFAULT_ACCEPTANCE_LOCK)
        ),
        "upstream_v2_training_source_lock_sha256": acceptance[
            "upstream_v2_training_source_lock_sha256"
        ],
        "upstream_v2_acceptance_source_lock_sha256": acceptance[
            "upstream_v2_acceptance_source_lock_sha256"
        ],
        "training_data_sha256": training["training_data_sha256"],
    }


def inspect_v2_result() -> dict[str, Any]:
    readiness = v2_post.inspect_training_readiness()
    if readiness.get("required_runs_complete") is not True:
        return {
            "status": "waiting",
            "reason": "v2_formal800_or_required_control_incomplete",
            "readiness": readiness,
        }
    marker = Path(v2_post.COMPLETE_MARKER)
    if not marker.exists() and not marker.is_symlink():
        return {
            "status": "waiting",
            "reason": "v2_canonical_postprocess_marker_missing",
            "readiness": readiness,
        }
    triplet = v2_handoff.validate_v2_triplet()
    decision = triplet.get("decision")
    _require(
        decision in {READY_DECISION, NOT_NEEDED_DECISION},
        "V2 canonical decision is unsupported",
    )
    return {
        "status": "ready",
        "decision": decision,
        "readiness": readiness,
        "triplet": triplet,
    }


def build_handoff_plan(
    *,
    physical_gpu: int,
    closure_verifier: Callable[[], Mapping[str, Any]] = verify_v3_closure,
) -> dict[str, Any]:
    if physical_gpu not in PHYSICAL_GPU_UUIDS:
        raise ValueError("V3 handoff physical GPU must be 2 or 3")
    v2 = inspect_v2_result()
    if v2["status"] == "waiting":
        return {
            "schema": SCHEMA,
            "status": "waiting",
            "mutating": False,
            "v2": v2,
            "commands": [],
        }
    if v2["decision"] == NOT_NEEDED_DECISION:
        return {
            "schema": SCHEMA,
            "status": "v3_not_needed",
            "mutating": False,
            "v2": v2,
            "commands": [],
        }
    _require(
        V3_LAUNCHER.is_file()
        and not V3_LAUNCHER.is_symlink()
        and bool(V3_LAUNCHER.stat().st_mode & 0o111),
        "V3 launcher is missing or not executable",
    )
    closure = dict(closure_verifier())
    command = [
        str(V3_LAUNCHER.resolve()),
        "--physical-gpu",
        str(physical_gpu),
    ]
    return {
        "schema": SCHEMA,
        "status": "ready_to_launch_v3",
        "mutating": False,
        "variant": V3_VARIANT,
        "physical_gpu": physical_gpu,
        "physical_gpu_uuid": PHYSICAL_GPU_UUIDS[physical_gpu],
        "v2": v2,
        "closure": closure,
        "commands": [command],
        "v1_v2_services_modified": False,
        "v1_v2_artifacts_modified": False,
    }


def execute_handoff(
    plan: Mapping[str, Any],
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    _require(
        plan.get("status") == "ready_to_launch_v3",
        "V3 handoff plan is not launchable",
    )
    commands = plan.get("commands")
    _require(
        isinstance(commands, list) and len(commands) == 1,
        "V3 handoff must contain exactly one launcher command",
    )
    command = commands[0]
    _require(
        isinstance(command, list)
        and command
        and Path(command[0]).resolve() == V3_LAUNCHER.resolve(),
        "V3 handoff command does not use the frozen launcher",
    )
    runner(command, check=True, capture_output=False)
    result = dict(plan)
    result.update(
        {
            "status": "v3_launcher_invoked",
            "mutating": True,
            "executed_command_count": 1,
            "v1_v2_services_modified": False,
            "v1_v2_artifacts_modified": False,
        }
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate V2 closure and explicitly hand off to V3"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        choices=tuple(PHYSICAL_GPU_UUIDS),
        default=2,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_handoff_plan(physical_gpu=args.physical_gpu)
    result = execute_handoff(plan) if args.execute else plan
    print(
        json.dumps(result, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
