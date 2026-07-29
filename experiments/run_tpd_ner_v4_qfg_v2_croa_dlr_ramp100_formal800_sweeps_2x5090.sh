#!/usr/bin/env bash
set -Eeuo pipefail

ramp_sweeps_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    ramp_sweeps_mode="preflight"
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

ramp_sweeps_repo="${TPD_NER_DLR_RAMP100_REPO:-/home/ly/SCTransNet_main}"
ramp_sweeps_python="${TPD_NER_DLR_RAMP100_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
ramp_sweeps_worker="$ramp_sweeps_repo/experiments/run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_eval_lane.sh"
ramp_sweeps_source_lock="${TPD_NER_DLR_RAMP100_SOURCE_LOCK:-$ramp_sweeps_repo/experiments/tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json}"
ramp_sweeps_result_root="${TPD_NER_DLR_RAMP100_RESULT_ROOT:-$ramp_sweeps_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_v1}"
ramp_sweeps_qfg_root="${TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT:-$ramp_sweeps_result_root/qfg_dlr_lane}"
ramp_sweeps_tss_root="${TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT:-$ramp_sweeps_result_root/tss_qfg_dlr_lane}"
ramp_sweeps_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
ramp_sweeps_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

ramp_sweeps_abort() {
    echo "TPDNER_DLR_RAMP100_SWEEPS_ABORT reason=$1" >&2
    exit 64
}

[[ -d "$ramp_sweeps_repo" && ! -L "$ramp_sweeps_repo" ]] \
    || ramp_sweeps_abort "invalid_repo"
[[ -x "$ramp_sweeps_python" ]] \
    || ramp_sweeps_abort "python_not_executable"
ramp_sweeps_python_real="$(readlink -f -- "$ramp_sweeps_python")"
[[ -n "$ramp_sweeps_python_real" \
    && -f "$ramp_sweeps_python_real" \
    && ! -L "$ramp_sweeps_python_real" \
    && -x "$ramp_sweeps_python_real" ]] \
    || ramp_sweeps_abort "python_target_nonregular"
[[ -f "$ramp_sweeps_worker" && ! -L "$ramp_sweeps_worker" \
    && -x "$ramp_sweeps_worker" ]] \
    || ramp_sweeps_abort "worker_nonregular_or_nonexecutable"
[[ -f "$ramp_sweeps_source_lock" && ! -L "$ramp_sweeps_source_lock" ]] \
    || ramp_sweeps_abort "source_lock_nonregular"
[[ -d "$ramp_sweeps_result_root" && ! -L "$ramp_sweeps_result_root" ]] \
    || ramp_sweeps_abort "result_root_nonregular"
[[ -d "$ramp_sweeps_qfg_root" && ! -L "$ramp_sweeps_qfg_root" ]] \
    || ramp_sweeps_abort "qfg_lane_root_nonregular"
[[ -d "$ramp_sweeps_tss_root" && ! -L "$ramp_sweeps_tss_root" ]] \
    || ramp_sweeps_abort "tss_lane_root_nonregular"

export TPD_NER_DLR_RAMP100_REPO="$ramp_sweeps_repo"
export TPD_NER_DLR_RAMP100_PYTHON="$ramp_sweeps_python"
export TPD_NER_DLR_RAMP100_SOURCE_LOCK="$ramp_sweeps_source_lock"
export TPD_NER_DLR_RAMP100_RESULT_ROOT="$ramp_sweeps_result_root"
export TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT="$ramp_sweeps_qfg_root"
export TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT="$ramp_sweeps_tss_root"

ramp_sweeps_preflight_one() {
    local ramp_sweeps_variant="$1"
    local ramp_sweeps_checkpoint="$2"
    local ramp_sweeps_physical_index="$3"
    local ramp_sweeps_gpu_uuid="$4"
    local ramp_sweeps_status
    if "$ramp_sweeps_worker" \
        --preflight \
        "$ramp_sweeps_variant" \
        "$ramp_sweeps_checkpoint" \
        "$ramp_sweeps_physical_index" \
        "$ramp_sweeps_gpu_uuid"
    then
        return 0
    else
        ramp_sweeps_status="$?"
        echo "TPDNER_DLR_RAMP100_SWEEPS_PREFLIGHT_FAILED variant=$ramp_sweeps_variant checkpoint=$ramp_sweeps_checkpoint physical_gpu=$ramp_sweeps_physical_index status=$ramp_sweeps_status outputs_started=false" >&2
        return "$ramp_sweeps_status"
    fi
}

ramp_sweeps_preflight_all() {
    ramp_sweeps_preflight_one \
        qfg_dlr best.pth.tar 2 "$ramp_sweeps_gpu2_uuid"
    ramp_sweeps_preflight_one \
        qfg_dlr best_miou.pth.tar 2 "$ramp_sweeps_gpu2_uuid"
    ramp_sweeps_preflight_one \
        tss_qfg_dlr best.pth.tar 3 "$ramp_sweeps_gpu3_uuid"
    ramp_sweeps_preflight_one \
        tss_qfg_dlr best_miou.pth.tar 3 "$ramp_sweeps_gpu3_uuid"
    echo "TPDNER_DLR_RAMP100_SWEEPS_PREFLIGHT_COMPLETE tasks=4 qfg_dlr_gpu=2 tss_qfg_dlr_gpu=3 outputs_started=false"
}

ramp_sweeps_run_lane() {
    local ramp_sweeps_variant="$1"
    local ramp_sweeps_physical_index="$2"
    local ramp_sweeps_gpu_uuid="$3"
    "$ramp_sweeps_worker" \
        "$ramp_sweeps_variant" \
        best.pth.tar \
        "$ramp_sweeps_physical_index" \
        "$ramp_sweeps_gpu_uuid"
    "$ramp_sweeps_worker" \
        "$ramp_sweeps_variant" \
        best_miou.pth.tar \
        "$ramp_sweeps_physical_index" \
        "$ramp_sweeps_gpu_uuid"
    echo "TPDNER_DLR_RAMP100_SWEEPS_LANE_COMPLETE variant=$ramp_sweeps_variant physical_gpu=$ramp_sweeps_physical_index checkpoints=best,best_miou"
}

ramp_sweeps_preflight_all
if [[ "$ramp_sweeps_mode" == "preflight" ]]; then
    echo "TPDNER_DLR_RAMP100_SWEEPS_PREFLIGHT_ONLY tasks=4 writes_performed=false"
    exit 0
fi

ramp_sweeps_lock_dir="$ramp_sweeps_result_root/.evaluation_locks"
if [[ -L "$ramp_sweeps_lock_dir" \
    || ( -e "$ramp_sweeps_lock_dir" && ! -d "$ramp_sweeps_lock_dir" ) ]]; then
    ramp_sweeps_abort "lock_directory_nonregular"
fi
mkdir -p "$ramp_sweeps_lock_dir"
ramp_sweeps_lock="$ramp_sweeps_lock_dir/formal800_sweeps_2x5090.lock"
if [[ -L "$ramp_sweeps_lock" \
    || ( -e "$ramp_sweeps_lock" && ! -f "$ramp_sweeps_lock" ) ]]; then
    ramp_sweeps_abort "claim_nonregular"
fi
exec 7>>"$ramp_sweeps_lock"
if ! flock -n 7; then
    echo "TPDNER_DLR_RAMP100_SWEEPS_RETRY reason=orchestration_claim_held" >&2
    exit 75
fi

(
    ramp_sweeps_run_lane \
        qfg_dlr 2 "$ramp_sweeps_gpu2_uuid"
) &
ramp_sweeps_qfg_pid="$!"
(
    ramp_sweeps_run_lane \
        tss_qfg_dlr 3 "$ramp_sweeps_gpu3_uuid"
) &
ramp_sweeps_tss_pid="$!"

ramp_sweeps_stop_children() {
    kill -TERM \
        "$ramp_sweeps_qfg_pid" \
        "$ramp_sweeps_tss_pid" \
        2>/dev/null || true
}
trap ramp_sweeps_stop_children INT TERM

echo "TPDNER_DLR_RAMP100_SWEEPS_PARALLEL_STARTED qfg_dlr_gpu=2 qfg_dlr_pid=$ramp_sweeps_qfg_pid tss_qfg_dlr_gpu=3 tss_qfg_dlr_pid=$ramp_sweeps_tss_pid"
ramp_sweeps_qfg_status=0
ramp_sweeps_tss_status=0
wait "$ramp_sweeps_qfg_pid" || ramp_sweeps_qfg_status="$?"
wait "$ramp_sweeps_tss_pid" || ramp_sweeps_tss_status="$?"
trap - INT TERM

if [[ "$ramp_sweeps_qfg_status" -ne 0 \
    || "$ramp_sweeps_tss_status" -ne 0 ]]; then
    echo "TPDNER_DLR_RAMP100_SWEEPS_FAILED qfg_dlr_status=$ramp_sweeps_qfg_status tss_qfg_dlr_status=$ramp_sweeps_tss_status" >&2
    exit 1
fi
echo "TPDNER_DLR_RAMP100_SWEEPS_COMPLETE qfg_dlr_status=0 tss_qfg_dlr_status=0 tasks=4 output_root=$ramp_sweeps_result_root"
