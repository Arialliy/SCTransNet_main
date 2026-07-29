#!/usr/bin/env bash
set -Eeuo pipefail

qfg_sweeps_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    qfg_sweeps_mode="preflight"
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

qfg_sweeps_repo="${TPD_NER_V4_QFG_V2_CROA_REPO:-/home/ly/SCTransNet_main}"
qfg_sweeps_python="${TPD_NER_V4_QFG_V2_CROA_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
qfg_sweeps_worker="$qfg_sweeps_repo/experiments/run_tpd_ner_v4_qfg_v2_croa_formal800_eval_lane.sh"
qfg_sweeps_source_lock="${TPD_NER_V4_QFG_V2_CROA_SOURCE_LOCK:-$qfg_sweeps_repo/experiments/tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json}"
qfg_sweeps_result_root="${TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT:-$qfg_sweeps_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized}"
qfg_sweeps_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
qfg_sweeps_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

qfg_sweeps_abort() {
    echo "TPDNERV4QFG_SWEEPS_ABORT reason=$1" >&2
    exit 64
}

[[ -d "$qfg_sweeps_repo" && ! -L "$qfg_sweeps_repo" ]] \
    || qfg_sweeps_abort "invalid_repo"
[[ -x "$qfg_sweeps_python" ]] \
    || qfg_sweeps_abort "python_not_executable"
[[ -f "$qfg_sweeps_worker" && ! -L "$qfg_sweeps_worker" \
    && -x "$qfg_sweeps_worker" ]] \
    || qfg_sweeps_abort "worker_nonregular_or_nonexecutable"
[[ -f "$qfg_sweeps_source_lock" && ! -L "$qfg_sweeps_source_lock" ]] \
    || qfg_sweeps_abort "source_lock_nonregular"
[[ -d "$qfg_sweeps_result_root" && ! -L "$qfg_sweeps_result_root" ]] \
    || qfg_sweeps_abort "result_root_nonregular"

# Propagate the same explicit optimized-V2 paths to every worker invocation.
export TPD_NER_V4_QFG_V2_CROA_REPO="$qfg_sweeps_repo"
export TPD_NER_V4_QFG_V2_CROA_PYTHON="$qfg_sweeps_python"
export TPD_NER_V4_QFG_V2_CROA_SOURCE_LOCK="$qfg_sweeps_source_lock"
export TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT="$qfg_sweeps_result_root"

qfg_sweeps_preflight_one() {
    local qfg_sweeps_variant="$1"
    local qfg_sweeps_checkpoint="$2"
    local qfg_sweeps_physical_index="$3"
    local qfg_sweeps_gpu_uuid="$4"
    local qfg_sweeps_status
    if "$qfg_sweeps_worker" \
        --preflight \
        "$qfg_sweeps_variant" \
        "$qfg_sweeps_checkpoint" \
        "$qfg_sweeps_physical_index" \
        "$qfg_sweeps_gpu_uuid"
    then
        return 0
    else
        qfg_sweeps_status="$?"
        echo "TPDNERV4QFG_SWEEPS_PREFLIGHT_FAILED variant=$qfg_sweeps_variant checkpoint=$qfg_sweeps_checkpoint physical_gpu=$qfg_sweeps_physical_index status=$qfg_sweeps_status outputs_started=false" >&2
        return "$qfg_sweeps_status"
    fi
}

qfg_sweeps_preflight_all() {
    # All four read-only checks finish before either output-producing lane is
    # launched.  The fixed ownership is GPU2=qfg_only and GPU3=tss_qfg.
    qfg_sweeps_preflight_one \
        qfg_only best.pth.tar 2 "$qfg_sweeps_gpu2_uuid"
    qfg_sweeps_preflight_one \
        qfg_only best_miou.pth.tar 2 "$qfg_sweeps_gpu2_uuid"
    qfg_sweeps_preflight_one \
        tss_qfg best.pth.tar 3 "$qfg_sweeps_gpu3_uuid"
    qfg_sweeps_preflight_one \
        tss_qfg best_miou.pth.tar 3 "$qfg_sweeps_gpu3_uuid"
    echo "TPDNERV4QFG_SWEEPS_PREFLIGHT_COMPLETE tasks=4 qfg_only_gpu=2 tss_qfg_gpu=3 outputs_started=false"
}

qfg_sweeps_run_lane() {
    local qfg_sweeps_variant="$1"
    local qfg_sweeps_physical_index="$2"
    local qfg_sweeps_gpu_uuid="$3"

    "$qfg_sweeps_worker" \
        "$qfg_sweeps_variant" \
        best.pth.tar \
        "$qfg_sweeps_physical_index" \
        "$qfg_sweeps_gpu_uuid"
    "$qfg_sweeps_worker" \
        "$qfg_sweeps_variant" \
        best_miou.pth.tar \
        "$qfg_sweeps_physical_index" \
        "$qfg_sweeps_gpu_uuid"
    echo "TPDNERV4QFG_SWEEPS_LANE_COMPLETE variant=$qfg_sweeps_variant physical_gpu=$qfg_sweeps_physical_index checkpoints=best,best_miou"
}

qfg_sweeps_preflight_all
if [[ "$qfg_sweeps_mode" == "preflight" ]]; then
    echo "TPDNERV4QFG_SWEEPS_PREFLIGHT_ONLY tasks=4 writes_performed=false"
    exit 0
fi

# A nonblocking orchestration claim prevents two copies of this matrix from
# racing.  It does not inspect utilization and never waits for a device.
qfg_sweeps_lock_dir="$qfg_sweeps_result_root/.evaluation_locks"
if [[ -L "$qfg_sweeps_lock_dir" \
    || ( -e "$qfg_sweeps_lock_dir" && ! -d "$qfg_sweeps_lock_dir" ) ]]; then
    qfg_sweeps_abort "lock_directory_nonregular"
fi
mkdir -p "$qfg_sweeps_lock_dir"
qfg_sweeps_lock="$qfg_sweeps_lock_dir/formal800_sweeps_2x5090.lock"
if [[ -L "$qfg_sweeps_lock" \
    || ( -e "$qfg_sweeps_lock" && ! -f "$qfg_sweeps_lock" ) ]]; then
    qfg_sweeps_abort "claim_nonregular"
fi
exec 7>>"$qfg_sweeps_lock"
[[ -f "$qfg_sweeps_lock" && ! -L "$qfg_sweeps_lock" ]] \
    || qfg_sweeps_abort "claim_nonregular_after_open"
if ! flock -n 7; then
    echo "TPDNERV4QFG_SWEEPS_RETRY reason=orchestration_claim_held" >&2
    exit 75
fi

(
    qfg_sweeps_run_lane \
        qfg_only 2 "$qfg_sweeps_gpu2_uuid"
) &
qfg_sweeps_qfg_pid="$!"
(
    qfg_sweeps_run_lane \
        tss_qfg 3 "$qfg_sweeps_gpu3_uuid"
) &
qfg_sweeps_tss_pid="$!"

qfg_sweeps_stop_children() {
    kill -TERM \
        "$qfg_sweeps_qfg_pid" \
        "$qfg_sweeps_tss_pid" \
        2>/dev/null || true
}
trap qfg_sweeps_stop_children INT TERM

echo "TPDNERV4QFG_SWEEPS_PARALLEL_STARTED qfg_only_gpu=2 qfg_only_pid=$qfg_sweeps_qfg_pid tss_qfg_gpu=3 tss_qfg_pid=$qfg_sweeps_tss_pid"
qfg_sweeps_qfg_status=0
qfg_sweeps_tss_status=0
wait "$qfg_sweeps_qfg_pid" || qfg_sweeps_qfg_status="$?"
wait "$qfg_sweeps_tss_pid" || qfg_sweeps_tss_status="$?"
trap - INT TERM

if [[ "$qfg_sweeps_qfg_status" -ne 0 \
    || "$qfg_sweeps_tss_status" -ne 0 ]]; then
    echo "TPDNERV4QFG_SWEEPS_FAILED qfg_only_status=$qfg_sweeps_qfg_status tss_qfg_status=$qfg_sweeps_tss_status" >&2
    exit 1
fi
echo "TPDNERV4QFG_SWEEPS_COMPLETE qfg_only_status=0 tss_qfg_status=0 tasks=4 output_root=$qfg_sweeps_result_root"
