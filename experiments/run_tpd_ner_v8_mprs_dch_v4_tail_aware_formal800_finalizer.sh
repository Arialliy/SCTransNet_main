#!/usr/bin/env bash
set -Eeuo pipefail

v4_finalizer_mode="run"
if [[ "${1:-}" == "--preflight" || "${1:-}" == "--status" ]]; then
    v4_finalizer_mode="${1#--}"
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight|--status]" >&2
    exit 2
fi

v4_finalizer_repo="${TPD_NER_V8_V4_TAIL_AWARE_REPO:-/home/ly/SCTransNet_main}"
v4_finalizer_python="${TPD_NER_V8_V4_TAIL_AWARE_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v4_finalizer_worker="$v4_finalizer_repo/experiments/run_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_eval_lane.sh"
v4_finalizer_evaluator="$v4_finalizer_repo/experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa.py"
v4_finalizer_postprocessor="$v4_finalizer_repo/experiments/postprocess_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800.py"
v4_finalizer_gpu23_manager="$v4_finalizer_repo/experiments/manage_gpu23_memory_reservation.sh"
v4_finalizer_result_root="${TPD_NER_V8_V4_TAIL_AWARE_RESULT_ROOT:-$v4_finalizer_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1}"
v4_finalizer_variant="tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
v4_finalizer_run_dir="$v4_finalizer_result_root/NUDT-SIRST/$v4_finalizer_variant/seed_42_formal800_exact_v4_tail_aware_seed42"
v4_finalizer_comparison="$v4_finalizer_result_root/NUDT-SIRST/comparison"
v4_finalizer_best="$v4_finalizer_run_dir/pd_fa_sweep_best.pth.json"
v4_finalizer_best_miou="$v4_finalizer_run_dir/pd_fa_sweep_best_miou.pth.json"
v4_finalizer_report_json="$v4_finalizer_comparison/tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_comparison.json"
v4_finalizer_report_md="$v4_finalizer_comparison/tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_comparison.md"
v4_finalizer_marker="$v4_finalizer_comparison/POSTPROCESS_COMPLETE.json"
v4_finalizer_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v4_finalizer_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v4_finalizer_unit="sctransnet-tpd-ner-v8-v4-tail-aware-formal800-finalizer.service"
v4_finalizer_best_unit="sctransnet-tpd-ner-v8-v4-tail-aware-eval-best-gpu2.service"
v4_finalizer_best_miou_unit="sctransnet-tpd-ner-v8-v4-tail-aware-eval-best-miou-gpu3.service"
v4_finalizer_gpu3_guard_unit="sctransnet-gpu-memory-reservation-gpu3.service"
v4_finalizer_gpu2_guard_unit="sctransnet-gpu-memory-reservation-gpu2.service"
v4_finalizer_training_units=(
    "sctransnet-tpd-ner-v8-v4-tail-aware-gpu2.service"
    "sctransnet-tpd-ner-v8-v4-tail-aware-gpu3.service"
)

v4_finalizer_retry() {
    echo "TPDNERV8V4TAIL_FINALIZER_RETRY reason=$1"
    exit 75
}

v4_finalizer_abort() {
    echo "TPDNERV8V4TAIL_FINALIZER_ABORT reason=$1" >&2
    exit 64
}

v4_finalizer_map_error() {
    local v4_finalizer_status="$?"
    local v4_finalizer_line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    if [[ "$v4_finalizer_status" -eq 64 \
        || "$v4_finalizer_status" -eq 75 ]]; then
        exit "$v4_finalizer_status"
    fi
    echo "TPDNERV8V4TAIL_FINALIZER_ABORT reason=stage_failed original_exit=$v4_finalizer_status line=$v4_finalizer_line" >&2
    exit 64
}
trap v4_finalizer_map_error ERR

v4_finalizer_require_static_contract() {
    [[ -d "$v4_finalizer_repo" && ! -L "$v4_finalizer_repo" ]] \
        || v4_finalizer_abort "invalid_repo"
    [[ -x "$v4_finalizer_python" ]] \
        || v4_finalizer_abort "python_not_executable"
    local v4_finalizer_file
    for v4_finalizer_file in \
        "$v4_finalizer_worker" \
        "$v4_finalizer_evaluator" \
        "$v4_finalizer_postprocessor" \
        "$v4_finalizer_gpu23_manager"
    do
        [[ -f "$v4_finalizer_file" && ! -L "$v4_finalizer_file" ]] \
            || v4_finalizer_abort "required_source_nonregular"
    done
    [[ -x "$v4_finalizer_worker" ]] \
        || v4_finalizer_abort "worker_not_executable"
    [[ -x "$v4_finalizer_gpu23_manager" ]] \
        || v4_finalizer_abort "gpu23_manager_not_executable"
    if [[ -L "$v4_finalizer_result_root" \
        || ( -e "$v4_finalizer_result_root" \
            && ! -d "$v4_finalizer_result_root" ) ]]; then
        v4_finalizer_abort "result_root_nonregular"
    fi
}

v4_finalizer_unit_field() {
    local v4_finalizer_name="$1"
    local v4_finalizer_property="$2"
    systemctl --user show "$v4_finalizer_name" \
        --property="$v4_finalizer_property" \
        --value \
        2>/dev/null || true
}

v4_finalizer_unit_active() {
    local v4_finalizer_state
    v4_finalizer_state="$(
        v4_finalizer_unit_field "$1" "ActiveState"
    )"
    [[ "$v4_finalizer_state" == "active" \
        || "$v4_finalizer_state" == "activating" ]]
}

v4_finalizer_require_training_inactive() {
    local v4_finalizer_training_unit
    for v4_finalizer_training_unit in \
        "${v4_finalizer_training_units[@]}"
    do
        if v4_finalizer_unit_active "$v4_finalizer_training_unit"; then
            v4_finalizer_retry \
                "formal_training_active unit=$v4_finalizer_training_unit"
        fi
    done
}

v4_finalizer_release_gpu_reservation() {
    local v4_finalizer_guard_index="$1"
    local v4_finalizer_guard_unit="$2"
    local v4_finalizer_guard_status
    v4_finalizer_guard_status="$(
        "$v4_finalizer_gpu23_manager" release \
            --physical-gpu "$v4_finalizer_guard_index"
    )"
    printf '%s\n' "$v4_finalizer_guard_status"
    if v4_finalizer_unit_active "$v4_finalizer_guard_unit"; then
        v4_finalizer_abort "gpu${v4_finalizer_guard_index}_guard_still_active_after_release"
    fi
    "$v4_finalizer_python" - \
        "$v4_finalizer_guard_status" \
        "$v4_finalizer_guard_index" \
        "$v4_finalizer_guard_unit" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
physical_index = int(sys.argv[2])
unit = sys.argv[3]
if payload.get("active") is not False:
    raise SystemExit(f"GPU{physical_index} guard status remains active")
if payload.get("holder_process_alive") is not False:
    raise SystemExit(f"GPU{physical_index} guard holder remains alive")
recorded = payload.get("recorded_state")
if recorded is not None:
    if not isinstance(recorded, dict):
        raise SystemExit("guard recorded state is malformed")
    if recorded.get("status") not in {"released", "self_released"}:
        raise SystemExit("guard recorded state is not released")
print(
    "TPDNERV8V4TAIL_FINALIZER_GPU_GUARD_RELEASED"
    f" physical_gpu={physical_index} unit={unit}"
    " active=false holder_process_alive=false restart_after_eval=false",
    flush=True,
)
PY
}

v4_finalizer_release_gpu23_reservations() {
    v4_finalizer_release_gpu_reservation 2 "$v4_finalizer_gpu2_guard_unit"
    v4_finalizer_release_gpu_reservation 3 "$v4_finalizer_gpu3_guard_unit"

}

v4_finalizer_training_state() {
    "$v4_finalizer_python" - "$v4_finalizer_run_dir" <<'PY'
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path


run_dir = Path(sys.argv[1])
summary = run_dir / "summary.json"
metrics = run_dir / "metrics.jsonl"
if not summary.exists() and not summary.is_symlink():
    count = 0
    last_epoch = 0
    if metrics.is_file() and not metrics.is_symlink():
        lines = [line for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        count = len(lines)
        if lines:
            try:
                last_epoch = json.loads(lines[-1]).get("epoch", 0)
            except (json.JSONDecodeError, AttributeError):
                last_epoch = -1
    print(f"waiting events={count} last_epoch={last_epoch}")
    raise SystemExit(0)
if summary.is_symlink() or not summary.is_file():
    raise SystemExit("V4 summary is nonregular")
if metrics.is_symlink() or not metrics.is_file():
    raise SystemExit("V4 metrics history is nonregular")

# Model construction inside artifact validation emits human-readable progress
# on stdout.  Keep that diagnostic visible on stderr while reserving stdout
# for the single machine-readable readiness line consumed by the shell.
with contextlib.redirect_stdout(sys.stderr):
    from experiments import (
        evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as evaluator,
    )

    best = evaluator.validate_run_artifacts(run_dir, "best.pth.tar")
    best_miou = evaluator.validate_run_artifacts(run_dir, "best_miou.pth.tar")
if (
    best["metric_event_count"] != 800
    or best_miou["metric_event_count"] != 800
):
    raise SystemExit("V4 metrics are not exactly epochs 1..800")
print(
    "ready events=800 last_epoch=800"
    f" best_epoch={best['checkpoint_epoch']}"
    f" best_miou_epoch={best_miou['checkpoint_epoch']}"
)
PY
}

v4_finalizer_validate_sweeps() {
    "$v4_finalizer_python" - \
        "$v4_finalizer_run_dir" \
        "$v4_finalizer_best" \
        "$v4_finalizer_best_miou" \
        "$v4_finalizer_gpu2_uuid" \
        "$v4_finalizer_gpu3_uuid" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import (
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as evaluator,
)
from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800 as postprocess,
)


run_dir = Path(sys.argv[1]).resolve()
jobs = (
    ("best.pth.tar", Path(sys.argv[2]).resolve(), 2, sys.argv[4]),
    ("best_miou.pth.tar", Path(sys.argv[3]).resolve(), 3, sys.argv[5]),
)
for checkpoint, path, physical_index, physical_uuid in jobs:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"V4 sweep is not regular: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = evaluator.validate_run_artifacts(run_dir, checkpoint)
    evaluator.validate_output_identity(payload, artifact_audit=audit)
    postprocess.validate_v4_sweep(
        path,
        checkpoint=checkpoint,
        expected_run_dir=run_dir,
        source_lock_path=postprocess.V4_SOURCE_LOCK,
        source_lock_sha256=postprocess.V4_SOURCE_LOCK_SHA256,
    )
    assignment = payload.get("audit", {}).get("device_assignment", {})
    expected = {
        "device": "cuda:0",
        "physical_gpu_index": physical_index,
        "physical_gpu_uuid": physical_uuid,
        "cuda_visible_devices": physical_uuid,
        "device_name": "NVIDIA GeForce RTX 5090",
    }
    if assignment != expected:
        raise SystemExit(
            f"{checkpoint} was not evaluated on its assigned GPU"
        )
    print(
        "TPDNERV8V4TAIL_FINALIZER_SWEEP_VERIFIED"
        f" checkpoint={checkpoint}"
        f" physical_gpu={physical_index}"
        f" path={path}",
        flush=True,
    )
PY
}

v4_finalizer_verify_report() {
    "$v4_finalizer_python" - \
        "$v4_finalizer_best" \
        "$v4_finalizer_best_miou" \
        "$v4_finalizer_comparison" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800 as postprocess,
)


best = Path(sys.argv[1]).resolve()
best_miou = Path(sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()
json_path = output_dir / postprocess.JSON_OUTPUT.name
markdown_path = output_dir / postprocess.MARKDOWN_OUTPUT.name
marker_path = output_dir / postprocess.COMPLETE_MARKER.name
for path in (json_path, markdown_path, marker_path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"published V4 artifact is not regular: {path}")
expected = postprocess.aggregate(
    best_sweep=best,
    best_miou_sweep=best_miou,
)
observed = json.loads(json_path.read_text(encoding="utf-8"))
if observed != expected:
    raise SystemExit("published V4 comparison JSON differs on recomputation")
if markdown_path.read_text(encoding="utf-8") != postprocess.render_markdown(
    expected
):
    raise SystemExit("published V4 comparison Markdown differs")
marker = json.loads(marker_path.read_text(encoding="utf-8"))
expected_marker = {
    "schema": postprocess.COMPLETE_MARKER_SCHEMA,
    "status": "complete",
    "decision": expected["decision"],
    "aggregate_full_model_gate_passed": expected[
        "aggregate_full_model_gate_passed"
    ],
    "outputs": {
        json_path.name: postprocess.sha256_file(json_path),
        markdown_path.name: postprocess.sha256_file(markdown_path),
    },
}
if marker != expected_marker:
    raise SystemExit("published V4 completion marker differs")
print(
    "TPDNERV8V4TAIL_FINALIZER_REPORT_VERIFIED"
    f" decision={expected['decision']}"
    f" gate_passed={expected['aggregate_full_model_gate_passed']}"
    f" marker={marker_path}",
    flush=True,
)
PY
}

v4_finalizer_artifact_state() {
    local v4_finalizer_path="$1"
    if [[ -f "$v4_finalizer_path" && ! -L "$v4_finalizer_path" ]]; then
        echo "regular"
    elif [[ ! -e "$v4_finalizer_path" \
        && ! -L "$v4_finalizer_path" ]]; then
        echo "absent"
    else
        echo "nonregular"
    fi
}

v4_finalizer_emit_status() {
    local v4_finalizer_name
    for v4_finalizer_name in \
        "$v4_finalizer_unit" \
        "$v4_finalizer_best_unit" \
        "$v4_finalizer_best_miou_unit" \
        "$v4_finalizer_gpu3_guard_unit" \
        "$v4_finalizer_gpu2_guard_unit" \
        "${v4_finalizer_training_units[@]}"
    do
        echo "TPDNERV8V4TAIL_FINALIZER_UNIT unit=$v4_finalizer_name active=$(v4_finalizer_unit_field "$v4_finalizer_name" ActiveState) sub=$(v4_finalizer_unit_field "$v4_finalizer_name" SubState) result=$(v4_finalizer_unit_field "$v4_finalizer_name" Result) restarts=$(v4_finalizer_unit_field "$v4_finalizer_name" NRestarts)"
    done
    local v4_finalizer_readiness
    v4_finalizer_readiness="$(v4_finalizer_training_state)"
    echo "TPDNERV8V4TAIL_FINALIZER_TRAINING $v4_finalizer_readiness"
    for v4_finalizer_name in \
        "$v4_finalizer_best" \
        "$v4_finalizer_best_miou" \
        "$v4_finalizer_report_json" \
        "$v4_finalizer_report_md" \
        "$v4_finalizer_marker"
    do
        echo "TPDNERV8V4TAIL_FINALIZER_ARTIFACT state=$(v4_finalizer_artifact_state "$v4_finalizer_name") path=$v4_finalizer_name"
    done
}

v4_finalizer_require_static_contract
cd "$v4_finalizer_repo"

if [[ "$v4_finalizer_mode" == "status" ]]; then
    v4_finalizer_emit_status
    exit 0
fi

v4_finalizer_readiness="$(v4_finalizer_training_state)"
if [[ "$v4_finalizer_mode" == "preflight" ]]; then
    echo "TPDNERV8V4TAIL_FINALIZER_PREFLIGHT_OK training=$v4_finalizer_readiness best_gpu=2 best_gpu_uuid=$v4_finalizer_gpu2_uuid best_miou_gpu=3 best_miou_gpu_uuid=$v4_finalizer_gpu3_uuid logical_device=cuda:0 gpu23_guard_release=deferred_until_formal800_complete restart_guard_after_eval=false writes_performed=false"
    exit 0
fi

v4_finalizer_require_training_inactive
if [[ "$v4_finalizer_readiness" != ready\ * ]]; then
    v4_finalizer_retry "formal800_incomplete state=$v4_finalizer_readiness"
fi

if [[ -L "$v4_finalizer_result_root" \
    || ( -e "$v4_finalizer_result_root" \
        && ! -d "$v4_finalizer_result_root" ) ]]; then
    v4_finalizer_abort "result_root_nonregular"
fi
mkdir -p "$v4_finalizer_result_root"
v4_finalizer_lock_dir="$v4_finalizer_result_root/.finalizer_locks"
if [[ -L "$v4_finalizer_lock_dir" \
    || ( -e "$v4_finalizer_lock_dir" \
        && ! -d "$v4_finalizer_lock_dir" ) ]]; then
    v4_finalizer_abort "lock_directory_nonregular"
fi
mkdir -p "$v4_finalizer_lock_dir"
v4_finalizer_lock="$v4_finalizer_lock_dir/formal800_finalizer.lock"
if [[ -L "$v4_finalizer_lock" \
    || ( -e "$v4_finalizer_lock" && ! -f "$v4_finalizer_lock" ) ]]; then
    v4_finalizer_abort "finalizer_lock_nonregular"
fi
exec 9>>"$v4_finalizer_lock"
[[ -f "$v4_finalizer_lock" && ! -L "$v4_finalizer_lock" ]] \
    || v4_finalizer_abort "finalizer_lock_nonregular_after_open"
if ! flock -n 9; then
    v4_finalizer_retry "finalizer_lock_held"
fi

# Recheck all transient prerequisites while holding the finalizer claim.
v4_finalizer_require_training_inactive
v4_finalizer_readiness="$(v4_finalizer_training_state)"
if [[ "$v4_finalizer_readiness" != ready\ * ]]; then
    v4_finalizer_retry \
        "formal800_incomplete_after_lock state=$v4_finalizer_readiness"
fi

# Release only after formal800 completion and immediately before evaluation.
# The reservation is intentionally not restarted after final evidence closes.
v4_finalizer_release_gpu23_reservations

v4_finalizer_launch_eval() {
    local v4_finalizer_checkpoint="$1"
    local v4_finalizer_index="$2"
    local v4_finalizer_uuid="$3"
    local v4_finalizer_eval_unit="$4"
    local v4_finalizer_output="$5"
    local v4_finalizer_output_state
    v4_finalizer_output_state="$(
        v4_finalizer_artifact_state "$v4_finalizer_output"
    )"
    if [[ "$v4_finalizer_output_state" == "nonregular" ]]; then
        v4_finalizer_abort \
            "sweep_nonregular checkpoint=$v4_finalizer_checkpoint"
    fi
    if [[ "$v4_finalizer_output_state" == "regular" ]]; then
        return
    fi
    if v4_finalizer_unit_active "$v4_finalizer_eval_unit"; then
        return
    fi
    local v4_finalizer_eval_result
    v4_finalizer_eval_result="$(
        v4_finalizer_unit_field "$v4_finalizer_eval_unit" Result
    )"
    if [[ "$v4_finalizer_eval_result" == "exit-code" \
        || "$v4_finalizer_eval_result" == "signal" \
        || "$v4_finalizer_eval_result" == "core-dump" ]]; then
        v4_finalizer_abort \
            "evaluation_unit_failed unit=$v4_finalizer_eval_unit"
    fi
    systemd-run --user \
        --collect \
        --unit="${v4_finalizer_eval_unit%.service}" \
        --description="SCTransNet V4 formal800 $v4_finalizer_checkpoint evaluation on physical GPU$v4_finalizer_index" \
        --property=Type=exec \
        --property=Restart=on-failure \
        --property=RestartPreventExitStatus=64 \
        --property=RestartSec=60 \
        --property=StartLimitIntervalSec=0 \
        --property=TimeoutStopSec=300 \
        "$v4_finalizer_worker" \
        "$v4_finalizer_checkpoint" \
        "$v4_finalizer_index" \
        "$v4_finalizer_uuid"
    echo "TPDNERV8V4TAIL_FINALIZER_EVAL_LAUNCHED checkpoint=$v4_finalizer_checkpoint physical_gpu=$v4_finalizer_index unit=$v4_finalizer_eval_unit"
}

v4_finalizer_launch_eval \
    "best.pth.tar" \
    "2" \
    "$v4_finalizer_gpu2_uuid" \
    "$v4_finalizer_best_unit" \
    "$v4_finalizer_best"
v4_finalizer_launch_eval \
    "best_miou.pth.tar" \
    "3" \
    "$v4_finalizer_gpu3_uuid" \
    "$v4_finalizer_best_miou_unit" \
    "$v4_finalizer_best_miou"

if v4_finalizer_unit_active "$v4_finalizer_best_unit" \
    || v4_finalizer_unit_active "$v4_finalizer_best_miou_unit"; then
    v4_finalizer_retry "parallel_sweeps_running"
fi
if [[ "$(v4_finalizer_artifact_state "$v4_finalizer_best")" != "regular" \
    || "$(v4_finalizer_artifact_state "$v4_finalizer_best_miou")" \
        != "regular" ]]; then
    v4_finalizer_retry "waiting_for_two_sweeps"
fi

v4_finalizer_validate_sweeps

v4_finalizer_report_states=(
    "$(v4_finalizer_artifact_state "$v4_finalizer_report_json")"
    "$(v4_finalizer_artifact_state "$v4_finalizer_report_md")"
    "$(v4_finalizer_artifact_state "$v4_finalizer_marker")"
)
if [[ "${v4_finalizer_report_states[*]}" == "absent absent absent" ]]; then
    "$v4_finalizer_python" "$v4_finalizer_postprocessor" \
        --aggregate \
        --best-sweep "$v4_finalizer_best" \
        --best-miou-sweep "$v4_finalizer_best_miou" \
        --output-dir "$v4_finalizer_comparison"
elif [[ "${v4_finalizer_report_states[*]}" != "regular regular regular" ]]; then
    v4_finalizer_abort \
        "partial_or_nonregular_postprocess_publish states=${v4_finalizer_report_states[*]}"
fi

v4_finalizer_verify_report
echo "TPDNERV8V4TAIL_FINALIZER_COMPLETE best_gpu=2 best_miou_gpu=3 comparison=$v4_finalizer_report_json marker=$v4_finalizer_marker"
