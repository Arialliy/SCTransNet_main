#!/usr/bin/env bash
set -Eeuo pipefail

dch_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    dch_mode="preflight"
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

dch_repo="/home/ly/SCTransNet_main"
dch_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
dch_root="$dch_repo/experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
dch_comparison="$dch_root/NUDT-SIRST/comparison"
dch_global_lock="$dch_root/.postprocess.lock"
dch_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
dch_training_units=(
    sctransnet-tpd-clean-v7-dch-gpu2-lane.service
    sctransnet-tpd-clean-v7-dch-gpu3-lane.service
)

dch_retry() {
    echo "TPDCLEANV7DCH_FINALIZER_RETRY reason=$1"
    exit 75
}

dch_abort() {
    echo "TPDCLEANV7DCH_FINALIZER_ABORT reason=$1" >&2
    exit 64
}

dch_map_unexpected_error() {
    local dch_status="$?"
    local dch_line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    if [[ "$dch_status" -eq 64 || "$dch_status" -eq 75 ]]; then
        exit "$dch_status"
    fi
    echo "TPDCLEANV7DCH_FINALIZER_ABORT reason=stage_command_failed original_exit=$dch_status line=$dch_line" >&2
    exit 64
}
trap dch_map_unexpected_error ERR

dch_require_training_lanes_inactive() {
    local dch_unit
    for dch_unit in "${dch_training_units[@]}"; do
        if systemctl --user is-active --quiet "$dch_unit"; then
            dch_retry "training_lane_active unit=$dch_unit"
        fi
    done
}

dch_training_ready() {
    "$dch_python" - <<'PY'
from experiments.validate_tpd_clean_v7_dch_formal800_completion import (
    inspect_training_readiness,
)

print(
    "true"
    if inspect_training_readiness()["formal_matrix_complete"] is True
    else "false"
)
PY
}

dch_validate_source_locks() {
    "$dch_python" - <<'PY'
from experiments import freeze_tpd_clean_v7_dch_source_locks as locks

for kind in locks.LOCK_KINDS:
    path = locks.REPO_ROOT / locks.DEFAULT_LOCK_RELATIVES[kind]
    payload, digest = locks.validate_source_lock(
        kind,
        path,
        repo_root=locks.REPO_ROOT,
    )
    print(
        "TPDCLEANV7DCH_FINALIZER_LOCK_VALID "
        f"kind={kind} sha256={digest} sources={payload['source_count']} "
        f"path={path}",
        flush=True,
    )
PY
}

dch_require_regular_pair_or_absent() {
    local dch_first="$1"
    local dch_second="$2"
    local dch_label="$3"
    if [[ ! -e "$dch_first" && ! -L "$dch_first" \
        && ! -e "$dch_second" && ! -L "$dch_second" ]]; then
        echo "absent"
        return
    fi
    if [[ -f "$dch_first" && ! -L "$dch_first" \
        && -f "$dch_second" && ! -L "$dch_second" ]]; then
        echo "regular"
        return
    fi
    dch_abort "partial_or_nonregular_${dch_label}"
}

cd "$dch_repo"
[[ -x "$dch_python" ]] || dch_abort "python_not_executable"

# No CUDA consumer is reached until both serial training lanes are inactive.
dch_require_training_lanes_inactive
dch_training_state="$(dch_training_ready)"
if [[ "$dch_training_state" != "true" ]]; then
    dch_retry "formal_matrix_incomplete"
fi
dch_validate_source_locks

if [[ "$dch_mode" == "preflight" ]]; then
    echo "TPDCLEANV7DCH_FINALIZER_PREFLIGHT_OK sweep_gpu_mode=training_gpu_replay audit_gpu=2 audit_gpu_uuid=$dch_gpu2_uuid locks=3"
    exit 0
fi

[[ -d "$dch_root" && ! -L "$dch_root" ]] \
    || dch_abort "candidate_root_nonregular"
if [[ -L "$dch_comparison" \
    || ( -e "$dch_comparison" && ! -d "$dch_comparison" ) ]]; then
    dch_abort "comparison_nonregular"
fi
mkdir -p "$dch_comparison"
[[ -d "$dch_comparison" && ! -L "$dch_comparison" ]] \
    || dch_abort "comparison_nonregular"

if [[ -L "$dch_global_lock" \
    || ( -e "$dch_global_lock" && ! -f "$dch_global_lock" ) ]]; then
    dch_abort "global_postprocess_lock_nonregular"
fi
exec 9>>"$dch_global_lock"
if [[ ! -f "$dch_global_lock" || -L "$dch_global_lock" ]]; then
    dch_abort "global_postprocess_lock_nonregular_after_open"
fi
if ! flock -n 9; then
    dch_retry "global_postprocess_lock_held"
fi

# Re-evaluate every transient prerequisite after acquiring the global lock.
dch_require_training_lanes_inactive
dch_training_state="$(dch_training_ready)"
if [[ "$dch_training_state" != "true" ]]; then
    dch_retry "formal_matrix_incomplete_after_lock"
fi
dch_validate_source_locks

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1

# Stage 1: eight closed-interval sweeps.  The existing sweep runner validates
# every regular output and refuses partial/non-regular paths.
dch_require_training_lanes_inactive
"$dch_python" experiments/run_tpd_clean_v7_dch_formal800_sweeps.py \
    --run \
    --device cuda:0

# Stage 2: comparison and Gate A--E report.
dch_comparison_json="$dch_comparison/tpd_clean_v7_dch_formal800_comparison.json"
dch_comparison_md="$dch_comparison/tpd_clean_v7_dch_formal800_comparison.md"
dch_pair_state="$(
    dch_require_regular_pair_or_absent \
        "$dch_comparison_json" \
        "$dch_comparison_md" \
        "comparison_report"
)"
if [[ "$dch_pair_state" == "absent" ]]; then
    "$dch_python" experiments/summarize_tpd_clean_v7_dch_formal800.py --write
fi
"$dch_python" - <<'PY'
from experiments.validate_tpd_clean_v7_dch_formal800_completion import (
    validate_published_report,
)

validate_published_report()
print("TPDCLEANV7DCH_FINALIZER_COMPARISON_VERIFIED", flush=True)
PY

# Stage 3: immutable completion manifest and marker.
dch_completion_manifest="$dch_comparison/completion_inputs.json"
dch_completion_marker="$dch_comparison/COMPLETE.sha256"
dch_pair_state="$(
    dch_require_regular_pair_or_absent \
        "$dch_completion_manifest" \
        "$dch_completion_marker" \
        "completion_publish"
)"
if [[ "$dch_pair_state" == "absent" ]]; then
    "$dch_python" \
        experiments/validate_tpd_clean_v7_dch_formal800_completion.py \
        publish
fi
"$dch_python" \
    experiments/validate_tpd_clean_v7_dch_formal800_completion.py \
    verify

dch_mechanism_stage() {
    "$dch_python" - "$dch_root" "$dch_comparison" <<'PY'
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from analysis import diagnose_tpd_clean_v7_dch_mechanism as audit


results_root = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
report_path = output_dir / "tpd_clean_v7_dch_mechanism_audit.json"


def require_regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not a regular file: {path}")


def validate_checkpoint_audit(
    payload: Mapping[str, Any],
    job: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    expected_identity = {
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
        "checkpoint_role": str(job["role"]),
    }
    if value.get("schema") != audit.CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint audit schema differs")
    if {
        key: value.get(key) for key in expected_identity
    } != expected_identity:
        raise ValueError("checkpoint audit identity differs")
    if value.get("candidate_family") != "tpd_clean_v7_dch":
        raise ValueError("checkpoint audit family differs")
    if Path(str(value.get("checkpoint", ""))).resolve() != Path(
        job["checkpoint"]
    ).resolve():
        raise ValueError("checkpoint audit path differs")
    if Path(str(value.get("run_directory", ""))).resolve() != Path(
        job["run_dir"]
    ).resolve():
        raise ValueError("checkpoint audit run directory differs")
    if value.get("source_identity") != artifacts["source_identity"]:
        raise ValueError("checkpoint audit source identity differs")
    if (
        value.get("input_sha256_before") != artifacts["input_sha256"]
        or value.get("input_sha256_after") != artifacts["input_sha256"]
        or value.get("formal_inputs_unchanged") is not True
    ):
        raise ValueError("checkpoint audit input hashes differ")
    if (
        value.get("checkpoint_validation_metrics_17")
        != artifacts["checkpoint_metrics"]
        or value.get("native_validation_field_count")
        != len(audit.STORED_VALIDATION_METRICS)
    ):
        raise ValueError("checkpoint audit native metrics differ")
    if (
        value.get("training_performed") is not False
        or value.get("checkpoint_reselection_permitted") is not False
        or value.get("official_test_accessed") is not False
        or value.get("formal_gate_replacement") is not False
        or value.get("gate_A_E_recomputed") is not False
    ):
        raise ValueError("checkpoint audit scope differs")
    device = value.get("device")
    if (
        not isinstance(device, Mapping)
        or device.get("physical_gpu_index") != 2
        or device.get("physical_gpu_uuid")
        != "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
    ):
        raise ValueError("checkpoint audit is not bound to physical GPU2")
    points = value.get("operating_points")
    if not isinstance(points, Mapping) or set(points) != set(registry):
        raise ValueError("checkpoint audit operating registry differs")
    for key, expected in registry.items():
        point = points[key]
        if (
            not isinstance(point, Mapping)
            or float(point.get("threshold", -1.0))
            != float(expected["threshold"])
            or point.get("registry_labels")
            != list(expected["registry_labels"])
            or point.get("registry_kinds")
            != list(expected["registry_kinds"])
            or point.get("matched_operating_point")
            != bool(expected["matched_operating_point"])
        ):
            raise ValueError(
                f"checkpoint audit registry point differs: {key}"
            )
    return value


jobs = audit.expected_jobs(results_root, output_dir, audit.DEFAULT_RUN_TAG)
registries, reference_before = audit.load_reference_registries(
    audit.DEFAULT_REFERENCE_ROOT,
    fixed_thresholds=audit.DEFAULT_FIXED_THRESHOLDS,
    fa_budgets=audit.DEFAULT_FA_BUDGETS,
)


def build_expected_report(
    payloads: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    reference_after = {
        role: audit.file_sha256(
            audit.DEFAULT_REFERENCE_ROOT / relative
        )
        for role, relative in audit.REFERENCE_RELATIVE_PATHS.items()
    }
    if reference_after != reference_before:
        raise RuntimeError("Mechanism Audit M reference inputs changed")
    report = audit.build_mechanism_report(
        payloads,
        registries,
        candidate_root=results_root,
        reference_input_sha256=reference_before,
    )
    report.update(
        {
            "mode": "run",
            "results_root": str(results_root),
            "output_dir": str(output_dir),
            "evaluated_job_count": len(payloads),
            "checkpoint_audit_outputs": [
                str(Path(job["output"]).resolve()) for job in jobs
            ],
            "reference_input_sha256_before": reference_before,
            "reference_input_sha256_after": reference_after,
            "reference_inputs_unchanged": True,
            "reported_mechanism_measures": list(
                audit.AUDIT_MEASURE_KEYS
            ),
            "limitations": [
                "Mechanism Audit M does not replace or modify Gates A-E.",
                "The audit uses the internal 133-image validation split only.",
                "Auxiliary topology directions do not substitute for "
                "fragment_excess_total.",
                "The last checkpoint is descriptive and has no V6 selection-role "
                "baseline; it remains part of the 12-checkpoint coverage and M4.",
            ],
        }
    )
    return report


if report_path.exists() or report_path.is_symlink():
    require_regular(report_path, "Mechanism Audit M report")
    report = audit.validate_mechanism_report(report_path)
    payloads: dict[tuple[str, int, str], dict[str, Any]] = {}
    for job in jobs:
        artifacts = audit.validate_job_artifacts(job)
        registry = audit.thresholds_for_role(
            registries, str(job["role"])
        )
        output = Path(job["output"])
        require_regular(output, "checkpoint audit")
        payload = validate_checkpoint_audit(
            audit.load_json_object(output),
            job,
            artifacts,
            registry,
        )
        payloads[
            (str(job["variant"]), int(job["seed"]), str(job["role"]))
        ] = payload
    expected_report = build_expected_report(payloads)
    if report != expected_report:
        raise ValueError(
            "Mechanism Audit M report differs from its 12 exact audits"
        )
    print(
        "TPDCLEANV7DCH_FINALIZER_MECHANISM_VERIFIED "
        f"checkpoints={len(payloads)}",
        flush=True,
    )
    raise SystemExit(0)

if report_path.exists() or report_path.is_symlink():
    raise RuntimeError("Mechanism Audit M report is non-regular")

device_provenance: dict[str, Any] | None = None
device: torch.device | None = None
payloads = {}
for job in jobs:
    artifacts = audit.validate_job_artifacts(job)
    registry = audit.thresholds_for_role(registries, str(job["role"]))
    output = Path(job["output"])
    if output.exists() or output.is_symlink():
        require_regular(output, "checkpoint audit")
        payload = validate_checkpoint_audit(
            audit.load_json_object(output),
            job,
            artifacts,
            registry,
        )
    else:
        if device_provenance is None:
            determinism = audit.configure_dch_inference("cuda:0")
            device_provenance = audit.bind_requested_device("cuda:0", "2")
            device_provenance["determinism"] = determinism
            device = torch.device("cuda:0")
        assert device is not None
        payload = audit.evaluate_job(
            job,
            artifacts,
            registry,
            device=device,
            device_provenance=device_provenance,
            dilation_radius=3,
        )
        audit.write_json(output, payload, overwrite=False)
        payload = validate_checkpoint_audit(
            audit.load_json_object(output),
            job,
            artifacts,
            registry,
        )
    payloads[
        (str(job["variant"]), int(job["seed"]), str(job["role"]))
    ] = payload

report = build_expected_report(payloads)
audit.write_json(report_path, report, overwrite=False)
published = audit.validate_mechanism_report(report_path)
if published != report:
    raise RuntimeError("Mechanism Audit M report readback differs")
print(
    "TPDCLEANV7DCH_FINALIZER_MECHANISM_COMPLETE "
    f"checkpoints={len(payloads)} report={report_path}",
    flush=True,
)
PY
}

# Stage 4: resumable Mechanism Audit M.  Existing checkpoint-audit JSON is
# validated against its current immutable inputs and reused.  Missing jobs are
# evaluated on physical GPU2 and published with exclusive-create; no verified
# audit is overwritten during interruption recovery.
dch_require_training_lanes_inactive
dch_mechanism_stage

# Stage 5: final decision.  Existing outputs are re-derived and byte-checked
# modulo the intentionally generated timestamp.
dch_validate_source_locks
dch_final_json="$dch_comparison/tpd_clean_v7_dch_final_decision.json"
dch_final_md="$dch_comparison/tpd_clean_v7_dch_final_decision.md"
dch_pair_state="$(
    dch_require_regular_pair_or_absent \
        "$dch_final_json" \
        "$dch_final_md" \
        "final_report"
)"
if [[ "$dch_pair_state" == "absent" ]]; then
    "$dch_python" experiments/finalize_tpd_clean_v7_dch.py --write
fi
dch_verify_final_report() {
    "$dch_python" - <<'PY'
from __future__ import annotations

import copy

from experiments import finalize_tpd_clean_v7_dch as finalizer


published = finalizer._load_json(
    finalizer.DEFAULT_OUTPUT_DIR / finalizer.JSON_OUTPUT_NAME,
    "published DCH final report",
)
derived = finalizer.build_final_report()
published_without_time = copy.deepcopy(published)
derived_without_time = copy.deepcopy(derived)
published_without_time.pop("created_at_utc", None)
derived_without_time.pop("created_at_utc", None)
if published_without_time != derived_without_time:
    raise RuntimeError("published DCH final report differs from exact inputs")
markdown = finalizer.DEFAULT_OUTPUT_DIR / finalizer.MARKDOWN_OUTPUT_NAME
if (
    not markdown.is_file()
    or markdown.is_symlink()
    or markdown.read_text(encoding="utf-8")
    != finalizer.render_markdown(published)
):
    raise RuntimeError("published DCH final Markdown differs")
print(
    "TPDCLEANV7DCH_FINALIZER_FINAL_REPORT_VERIFIED "
    f"decision={published['decision']}",
    flush=True,
)
PY
}
dch_verify_final_report

# The frozen acceptance lock intentionally excludes this control plane.
# Bind the four controller sources and every published result in a separate,
# deterministic exclusive-create manifest.
# First perform one final, unified revalidation after the potentially long
# Audit M stage and before sealing the controller manifest.
dch_validate_source_locks
"$dch_python" \
    experiments/validate_tpd_clean_v7_dch_formal800_completion.py \
    verify
dch_mechanism_stage
dch_verify_final_report
"$dch_python" - "$dch_repo" "$dch_root" "$dch_comparison" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from analysis import diagnose_tpd_clean_v7_dch_mechanism as audit
from experiments import (
    freeze_tpd_clean_v7_dch_source_locks as locks,
)
from experiments import (
    validate_tpd_clean_v7_dch_formal800_completion as completion,
)


repo = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
comparison = Path(sys.argv[3]).resolve()
manifest_path = (
    comparison / "tpd_clean_v7_dch_finalizer_control_manifest.json"
)


def sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"manifest input is not regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


control_relatives = (
    "experiments/run_tpd_clean_v7_dch_formal800_finalizer.sh",
    "experiments/launch_tpd_clean_v7_dch_formal800_finalizer.sh",
    "experiments/status_tpd_clean_v7_dch_formal800_finalizer.sh",
    "tests/test_tpd_clean_v7_dch_formal800_finalizer.py",
)
lock_relatives = {
    kind: relative
    for kind, relative in locks.DEFAULT_LOCK_RELATIVES.items()
}
artifacts: list[dict[str, str]] = []
for seed in completion.SEEDS:
    for variant in completion.VARIANTS:
        run_dir = (
            root
            / completion.DATASET
            / variant
            / f"seed_{seed}_{completion.RUN_TAG}"
        )
        for role, spec in completion.ROLE_SPECS.items():
            path = run_dir / str(spec["sweep"])
            artifacts.append(
                {
                    "stage": "sweeps",
                    "name": f"{variant}/seed_{seed}/{role}",
                    "path": str(path),
                    "sha256": sha256(path),
                }
            )
for name in (
    "tpd_clean_v7_dch_formal800_comparison.json",
    "tpd_clean_v7_dch_formal800_comparison.md",
):
    path = comparison / name
    artifacts.append(
        {
            "stage": "summary_gates",
            "name": name,
            "path": str(path),
            "sha256": sha256(path),
        }
    )
for name in ("completion_inputs.json", "COMPLETE.sha256"):
    path = comparison / name
    artifacts.append(
        {
            "stage": "completion",
            "name": name,
            "path": str(path),
            "sha256": sha256(path),
        }
    )
for job in audit.expected_jobs(root, comparison, audit.DEFAULT_RUN_TAG):
    path = Path(job["output"]).resolve()
    artifacts.append(
        {
            "stage": "mechanism_checkpoint",
            "name": (
                f"{job['variant']}/seed_{job['seed']}/{job['role']}"
            ),
            "path": str(path),
            "sha256": sha256(path),
        }
    )
for stage, name in (
    ("mechanism_report", "tpd_clean_v7_dch_mechanism_audit.json"),
    ("final_report", "tpd_clean_v7_dch_final_decision.json"),
    ("final_report", "tpd_clean_v7_dch_final_decision.md"),
):
    path = comparison / name
    artifacts.append(
        {
            "stage": stage,
            "name": name,
            "path": str(path),
            "sha256": sha256(path),
        }
    )

payload = {
    "schema": "sctransnet_tpd_clean_v7_dch_finalizer_control_manifest_v1",
    "status": "complete",
    "candidate_family": "tpd_clean_v7_dch",
    "postprocess_gpu_contract": {
        "sweeps": {
            "mode": "serial_training_gpu_replay",
            "physical_indices": [2, 3],
        },
        "mechanism_audit": {
            "physical_index": 2,
            "physical_uuid": (
                "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
            ),
        },
    },
    "control_sources": {
        relative: sha256(repo / relative)
        for relative in control_relatives
    },
    "source_locks": {
        kind: {
            "path": str(repo / relative),
            "sha256": sha256(repo / relative),
        }
        for kind, relative in lock_relatives.items()
    },
    "stage_order": [
        "sweeps",
        "summary_gates",
        "completion",
        "mechanism_checkpoint",
        "mechanism_report",
        "final_report",
    ],
    "artifact_count": len(artifacts),
    "artifacts": artifacts,
}
encoded = (
    json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
if manifest_path.exists() or manifest_path.is_symlink():
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or json.loads(manifest_path.read_text(encoding="utf-8")) != payload
    ):
        raise RuntimeError("finalizer control manifest differs")
else:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(manifest_path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
published_manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)
if published_manifest != payload:
    raise RuntimeError("finalizer control manifest readback differs")
print(
    "TPDCLEANV7DCH_FINALIZER_CONTROL_MANIFEST_VERIFIED "
    f"artifacts={len(artifacts)} path={manifest_path}",
    flush=True,
)
PY

echo "TPDCLEANV7DCH_FINALIZER_COMPLETE sweep_gpus=2,3 audit_gpu=2 audit_gpu_uuid=$dch_gpu2_uuid"
