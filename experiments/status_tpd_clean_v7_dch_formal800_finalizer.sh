#!/usr/bin/env bash
set -euo pipefail

dch_repo="/home/ly/SCTransNet_main"
dch_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
dch_root="$dch_repo/experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
dch_comparison="$dch_root/NUDT-SIRST/comparison"
dch_finalizer_unit="sctransnet-tpd-clean-v7-dch-formal800-finalizer.service"
dch_training_units=(
    sctransnet-tpd-clean-v7-dch-gpu2-lane.service
    sctransnet-tpd-clean-v7-dch-gpu3-lane.service
)

dch_training_active="false"
for dch_unit in "$dch_finalizer_unit" "${dch_training_units[@]}"; do
    if systemctl --user cat "$dch_unit" >/dev/null 2>&1; then
        dch_active="$(systemctl --user show "$dch_unit" -p ActiveState --value)"
        dch_sub="$(systemctl --user show "$dch_unit" -p SubState --value)"
        dch_result="$(systemctl --user show "$dch_unit" -p Result --value)"
        dch_restarts="$(systemctl --user show "$dch_unit" -p NRestarts --value)"
        echo "TPDCLEANV7DCH_FINALIZER_UNIT unit=$dch_unit active=$dch_active sub=$dch_sub result=$dch_result restarts=$dch_restarts"
        if [[ "$dch_unit" != "$dch_finalizer_unit" \
            && "$dch_active" == "active" ]]; then
            dch_training_active="true"
        fi
    else
        echo "TPDCLEANV7DCH_FINALIZER_UNIT unit=$dch_unit status=not-found"
    fi
done
export DCH_FINALIZER_TRAINING_ACTIVE="$dch_training_active"

cd "$dch_repo"
"$dch_python" - "$dch_repo" "$dch_root" "$dch_comparison" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from analysis import diagnose_tpd_clean_v7_dch_mechanism as audit
from experiments import freeze_tpd_clean_v7_dch_source_locks as locks
from experiments import (
    validate_tpd_clean_v7_dch_formal800_completion as completion,
)


repo = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
comparison = Path(sys.argv[3]).resolve()
expected_artifact_paths: set[str] = set()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(stage: str, name: str, path: Path) -> str:
    if stage != "control_manifest":
        expected_artifact_paths.add(str(path.resolve()))
    if path.is_file() and not path.is_symlink():
        status = "regular"
        digest = sha256(path)
        print(
            "TPDCLEANV7DCH_FINALIZER_ARTIFACT "
            f"stage={stage} name={name} status={status} "
            f"sha256={digest} path={path}"
        )
        return status
    status = (
        "nonregular"
        if path.exists() or path.is_symlink()
        else "missing"
    )
    print(
        "TPDCLEANV7DCH_FINALIZER_ARTIFACT "
        f"stage={stage} name={name} status={status} path={path}"
    )
    return status


states: dict[str, list[str]] = {
    "sweeps": [],
    "summary_gates": [],
    "completion": [],
    "mechanism_checkpoint": [],
    "mechanism_report": [],
    "final_report": [],
    "control_manifest": [],
}
for seed in completion.SEEDS:
    for variant in completion.VARIANTS:
        run_dir = (
            root
            / completion.DATASET
            / variant
            / f"seed_{seed}_{completion.RUN_TAG}"
        )
        for role, spec in completion.ROLE_SPECS.items():
            states["sweeps"].append(
                emit(
                    "sweeps",
                    f"{variant}/seed_{seed}/{role}",
                    run_dir / str(spec["sweep"]),
                )
            )
for name in (
    "tpd_clean_v7_dch_formal800_comparison.json",
    "tpd_clean_v7_dch_formal800_comparison.md",
):
    states["summary_gates"].append(
        emit("summary_gates", name, comparison / name)
    )
for name in ("completion_inputs.json", "COMPLETE.sha256"):
    states["completion"].append(
        emit("completion", name, comparison / name)
    )
for job in audit.expected_jobs(root, comparison, audit.DEFAULT_RUN_TAG):
    states["mechanism_checkpoint"].append(
        emit(
            "mechanism_checkpoint",
            f"{job['variant']}/seed_{job['seed']}/{job['role']}",
            Path(job["output"]),
        )
    )
states["mechanism_report"].append(
    emit(
        "mechanism_report",
        "tpd_clean_v7_dch_mechanism_audit.json",
        comparison / "tpd_clean_v7_dch_mechanism_audit.json",
    )
)
for name in (
    "tpd_clean_v7_dch_final_decision.json",
    "tpd_clean_v7_dch_final_decision.md",
):
    states["final_report"].append(
        emit("final_report", name, comparison / name)
    )
control_manifest = (
    comparison / "tpd_clean_v7_dch_finalizer_control_manifest.json"
)
states["control_manifest"].append(
    emit(
        "control_manifest",
        control_manifest.name,
        control_manifest,
    )
)

lock_status: dict[str, str] = {}
lock_digests: dict[str, str] = {}
for kind, relative in locks.DEFAULT_LOCK_RELATIVES.items():
    path = repo / relative
    try:
        _, digest = locks.validate_source_lock(kind, path, repo_root=repo)
    except Exception as exc:
        lock_status[kind] = f"invalid:{type(exc).__name__}"
    else:
        lock_status[kind] = f"valid:{digest}"
        lock_digests[kind] = digest
    print(
        "TPDCLEANV7DCH_FINALIZER_LOCK "
        f"kind={kind} status={lock_status[kind]} path={path}"
    )

nonregular = any(
    value == "nonregular"
    for values in states.values()
    for value in values
)
training = completion.inspect_training_readiness(root)
locks_valid = all(
    value.startswith("valid:") for value in lock_status.values()
)
if nonregular:
    stage = "invalid_artifact"
elif not locks_valid:
    stage = "invalid_source_lock"
elif (
    os.environ.get("DCH_FINALIZER_TRAINING_ACTIVE") == "true"
    or training.get("formal_matrix_complete") is not True
):
    stage = "waiting_training"
elif all(value == "regular" for value in states["control_manifest"]):
    stage = "complete"
elif not all(value == "regular" for value in states["sweeps"]):
    stage = "sweeps"
elif not all(value == "regular" for value in states["summary_gates"]):
    stage = "summary_gates"
elif not all(value == "regular" for value in states["completion"]):
    stage = "completion"
elif not all(
    value == "regular" for value in states["mechanism_checkpoint"]
):
    stage = "mechanism_checkpoint"
elif not all(value == "regular" for value in states["mechanism_report"]):
    stage = "mechanism_report"
elif not all(value == "regular" for value in states["final_report"]):
    stage = "final_report"
else:
    stage = "control_manifest"

manifest_verification = "not_available"
if control_manifest.is_file() and not control_manifest.is_symlink():
    try:
        manifest = json.loads(control_manifest.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            != "sctransnet_tpd_clean_v7_dch_finalizer_control_manifest_v1"
            or manifest.get("status") != "complete"
        ):
            raise ValueError("identity")
        expected_control_sources = {
            "experiments/run_tpd_clean_v7_dch_formal800_finalizer.sh",
            "experiments/launch_tpd_clean_v7_dch_formal800_finalizer.sh",
            "experiments/status_tpd_clean_v7_dch_formal800_finalizer.sh",
            "tests/test_tpd_clean_v7_dch_formal800_finalizer.py",
        }
        if set(manifest.get("control_sources", {})) != expected_control_sources:
            raise ValueError("control source registry")
        for relative, digest in manifest["control_sources"].items():
            if sha256(repo / relative) != digest:
                raise ValueError(f"control source differs: {relative}")
        expected_lock_records = {
            kind: {
                "path": str(repo / relative),
                "sha256": lock_digests[kind],
            }
            for kind, relative in locks.DEFAULT_LOCK_RELATIVES.items()
        }
        if manifest.get("source_locks") != expected_lock_records:
            raise ValueError("source lock registry")
        records = manifest.get("artifacts")
        if (
            not isinstance(records, list)
            or manifest.get("artifact_count") != len(records)
            or len(records) != 27
            or {str(record.get("path")) for record in records}
            != expected_artifact_paths
        ):
            raise ValueError("artifact registry")
        for record in records:
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError(f"artifact differs: {record['name']}")
        manifest_verification = "valid"
    except Exception as exc:
        manifest_verification = f"invalid:{type(exc).__name__}"
        stage = "invalid_artifact"

print(
    "TPDCLEANV7DCH_FINALIZER_STAGE "
    f"stage={stage} manifest={manifest_verification} "
    f"locks={json.dumps(lock_status, sort_keys=True)}"
)
PY
