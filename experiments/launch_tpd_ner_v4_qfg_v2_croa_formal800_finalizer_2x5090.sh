#!/usr/bin/env bash
set -Eeuo pipefail

qfg_finalize_launch_mode="launch"
case "${1:-}" in
    --preflight)
        qfg_finalize_launch_mode="preflight"
        shift
        ;;
    --status)
        qfg_finalize_launch_mode="status"
        shift
        ;;
    --worker)
        qfg_finalize_launch_mode="worker"
        shift
        ;;
esac
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight|--status|--worker]" >&2
    exit 2
fi

qfg_finalize_launch_repo="${TPD_NER_V4_QFG_V2_CROA_REPO:-/home/ly/SCTransNet_main}"
qfg_finalize_launch_result_root="${TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT:-$qfg_finalize_launch_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized}"
qfg_finalize_launch_finalizer="${TPD_NER_V4_QFG_V2_CROA_FINALIZER:-$qfg_finalize_launch_repo/experiments/finalize_tpd_ner_v4_qfg_v2_croa_formal800_2x5090.sh}"
qfg_finalize_launch_systemd_run="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_SYSTEMD_RUN:-systemd-run}"
qfg_finalize_launch_systemctl="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_SYSTEMCTL:-systemctl}"
qfg_finalize_launch_unit="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_UNIT:-sctransnet-tpd-ner-v4-qfg-v2-croa-formal800-finalizer.service}"
qfg_finalize_launch_self="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    printf '%s/%s\n' "$PWD" "$(basename -- "${BASH_SOURCE[0]}")"
)"

qfg_finalize_launch_c_run="$qfg_finalize_launch_result_root/NUDT-SIRST/qfg_only/seed_42_formal800_qfg_only"
qfg_finalize_launch_d_run="$qfg_finalize_launch_result_root/NUDT-SIRST/tss_qfg/seed_42_formal800_tss_qfg"
qfg_finalize_launch_c_summary="$qfg_finalize_launch_c_run/summary.json"
qfg_finalize_launch_d_summary="$qfg_finalize_launch_d_run/summary.json"

qfg_finalize_launch_abort() {
    echo "TPDNERV4QFG_FINALIZER_LAUNCH_ABORT reason=$1" >&2
    exit 64
}

qfg_finalize_launch_retry() {
    echo "TPDNERV4QFG_FINALIZER_LAUNCH_RETRY reason=$1" >&2
    exit 75
}

qfg_finalize_launch_require_command() {
    command -v -- "$1" >/dev/null 2>&1 \
        || qfg_finalize_launch_abort "$2_not_found"
}

qfg_finalize_launch_static_contract() {
    [[ -d "$qfg_finalize_launch_repo" \
        && ! -L "$qfg_finalize_launch_repo" ]] \
        || qfg_finalize_launch_abort "invalid_repo"
    [[ -f "$qfg_finalize_launch_self" \
        && ! -L "$qfg_finalize_launch_self" ]] \
        || qfg_finalize_launch_abort "launcher_nonregular"
    [[ -f "$qfg_finalize_launch_finalizer" \
        && ! -L "$qfg_finalize_launch_finalizer" \
        && -x "$qfg_finalize_launch_finalizer" ]] \
        || qfg_finalize_launch_abort "finalizer_not_executable_regular_file"
    [[ -d "$qfg_finalize_launch_result_root" \
        && ! -L "$qfg_finalize_launch_result_root" ]] \
        || qfg_finalize_launch_abort "result_root_nonregular"
    [[ -d "$qfg_finalize_launch_c_run" \
        && ! -L "$qfg_finalize_launch_c_run" ]] \
        || qfg_finalize_launch_abort "qfg_only_run_nonregular"
    [[ -d "$qfg_finalize_launch_d_run" \
        && ! -L "$qfg_finalize_launch_d_run" ]] \
        || qfg_finalize_launch_abort "tss_qfg_run_nonregular"
}

qfg_finalize_launch_summary_state() {
    local qfg_finalize_launch_path="$1"
    if [[ -f "$qfg_finalize_launch_path" \
        && ! -L "$qfg_finalize_launch_path" ]]; then
        echo "complete"
        return
    fi
    if [[ ! -e "$qfg_finalize_launch_path" \
        && ! -L "$qfg_finalize_launch_path" ]]; then
        echo "missing"
        return
    fi
    qfg_finalize_launch_abort "summary_nonregular path=$qfg_finalize_launch_path"
}

qfg_finalize_launch_readiness() {
    local qfg_finalize_launch_c_state
    local qfg_finalize_launch_d_state
    qfg_finalize_launch_c_state="$(
        qfg_finalize_launch_summary_state "$qfg_finalize_launch_c_summary"
    )" || return $?
    qfg_finalize_launch_d_state="$(
        qfg_finalize_launch_summary_state "$qfg_finalize_launch_d_summary"
    )" || return $?
    echo "qfg_only=$qfg_finalize_launch_c_state tss_qfg=$qfg_finalize_launch_d_state"
}

qfg_finalize_launch_unit_field() {
    "$qfg_finalize_launch_systemctl" --user show \
        "$qfg_finalize_launch_unit" \
        --property="$1" \
        --value \
        2>/dev/null || true
}

qfg_finalize_launch_static_contract
qfg_finalize_launch_readiness="$(
    qfg_finalize_launch_readiness
)" || exit $?

if [[ "$qfg_finalize_launch_mode" == "preflight" ]]; then
    qfg_finalize_launch_require_command \
        "$qfg_finalize_launch_systemd_run" "systemd_run"
    qfg_finalize_launch_require_command \
        "$qfg_finalize_launch_systemctl" "systemctl"
    echo "TPDNERV4QFG_FINALIZER_LAUNCH_PREFLIGHT_OK readiness=$qfg_finalize_launch_readiness worker_incomplete_exit=75 finalizer_permanent_exit=64 restart_sec=60 writes_performed=false gpu_query=false"
    exit 0
fi

if [[ "$qfg_finalize_launch_mode" == "worker" ]]; then
    if [[ "$qfg_finalize_launch_readiness" \
        != "qfg_only=complete tss_qfg=complete" ]]; then
        qfg_finalize_launch_retry \
            "formal800_summaries_incomplete $qfg_finalize_launch_readiness"
    fi
    echo "TPDNERV4QFG_FINALIZER_LAUNCH_HANDOFF readiness=$qfg_finalize_launch_readiness finalizer=$qfg_finalize_launch_finalizer"
    exec "$qfg_finalize_launch_finalizer"
fi

qfg_finalize_launch_require_command \
    "$qfg_finalize_launch_systemctl" "systemctl"

if [[ "$qfg_finalize_launch_mode" == "status" ]]; then
    echo "TPDNERV4QFG_FINALIZER_LAUNCH_STATUS unit=$qfg_finalize_launch_unit load=$(qfg_finalize_launch_unit_field LoadState) active=$(qfg_finalize_launch_unit_field ActiveState) sub=$(qfg_finalize_launch_unit_field SubState) result=$(qfg_finalize_launch_unit_field Result) restarts=$(qfg_finalize_launch_unit_field NRestarts) readiness=$qfg_finalize_launch_readiness"
    exit 0
fi

qfg_finalize_launch_require_command \
    "$qfg_finalize_launch_systemd_run" "systemd_run"
qfg_finalize_launch_active="$(
    qfg_finalize_launch_unit_field ActiveState
)"
if [[ "$qfg_finalize_launch_active" == "active" \
    || "$qfg_finalize_launch_active" == "activating" ]]; then
    echo "TPDNERV4QFG_FINALIZER_LAUNCH_ALREADY_ACTIVE unit=$qfg_finalize_launch_unit readiness=$qfg_finalize_launch_readiness"
    exit 0
fi

"$qfg_finalize_launch_systemd_run" --user \
    --collect \
    --unit="${qfg_finalize_launch_unit%.service}" \
    --description="SCTransNet TPD+NER+QFG formal800 automatic 2x5090 finalizer" \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartPreventExitStatus=64 \
    --property=RestartSec=60 \
    --property=StartLimitIntervalSec=0 \
    --property=TimeoutStopSec=300 \
    "$qfg_finalize_launch_self" --worker

echo "TPDNERV4QFG_FINALIZER_LAUNCH_STARTED unit=$qfg_finalize_launch_unit readiness=$qfg_finalize_launch_readiness qfg_only_gpu=2 tss_qfg_gpu=3 wait_for_gpu_idle=false"
