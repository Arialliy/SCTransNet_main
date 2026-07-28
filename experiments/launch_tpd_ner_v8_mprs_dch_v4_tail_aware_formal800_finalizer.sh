#!/usr/bin/env bash
set -Eeuo pipefail

v4_finalizer_launch_mode="launch"
if [[ "${1:-}" == "--preflight" || "${1:-}" == "--status" ]]; then
    v4_finalizer_launch_mode="${1#--}"
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight|--status]" >&2
    exit 2
fi

v4_finalizer_launch_repo="${TPD_NER_V8_V4_TAIL_AWARE_REPO:-/home/ly/SCTransNet_main}"
v4_finalizer_launch_runner="$v4_finalizer_launch_repo/experiments/run_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_finalizer.sh"
v4_finalizer_launch_unit="sctransnet-tpd-ner-v8-v4-tail-aware-formal800-finalizer.service"
v4_finalizer_launch_marker="$v4_finalizer_launch_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/comparison/POSTPROCESS_COMPLETE.json"

[[ -d "$v4_finalizer_launch_repo" \
    && ! -L "$v4_finalizer_launch_repo" ]] || {
    echo "TPDNERV8V4TAIL_FINALIZER_LAUNCH_ABORT reason=invalid_repo" >&2
    exit 64
}
[[ -x "$v4_finalizer_launch_runner" \
    && ! -L "$v4_finalizer_launch_runner" ]] || {
    echo "TPDNERV8V4TAIL_FINALIZER_LAUNCH_ABORT reason=runner_not_executable" >&2
    exit 64
}
cd "$v4_finalizer_launch_repo"

if [[ "$v4_finalizer_launch_mode" == "preflight" ]]; then
    "$v4_finalizer_launch_runner" --preflight
    exit $?
fi
if [[ "$v4_finalizer_launch_mode" == "status" ]]; then
    "$v4_finalizer_launch_runner" --status
    exit $?
fi

v4_finalizer_preflight_output="$(
    "$v4_finalizer_launch_runner" --preflight
)"
printf '%s\n' "$v4_finalizer_preflight_output"
[[ "$v4_finalizer_preflight_output" \
    == *"TPDNERV8V4TAIL_FINALIZER_PREFLIGHT_OK"* ]] || {
    echo "TPDNERV8V4TAIL_FINALIZER_LAUNCH_ABORT reason=preflight_identity_missing" >&2
    exit 64
}

v4_finalizer_launch_state="$(
    systemctl --user show "$v4_finalizer_launch_unit" \
        --property=ActiveState \
        --value \
        2>/dev/null || true
)"
if [[ "$v4_finalizer_launch_state" == "active" \
    || "$v4_finalizer_launch_state" == "activating" ]]; then
    echo "TPDNERV8V4TAIL_FINALIZER_ALREADY_ACTIVE unit=$v4_finalizer_launch_unit"
    exit 0
fi

if [[ -L "$v4_finalizer_launch_marker" \
    || ( -e "$v4_finalizer_launch_marker" \
        && ! -f "$v4_finalizer_launch_marker" ) ]]; then
    echo "TPDNERV8V4TAIL_FINALIZER_LAUNCH_ABORT reason=completion_marker_nonregular" >&2
    exit 64
fi
if [[ -f "$v4_finalizer_launch_marker" \
    && ! -L "$v4_finalizer_launch_marker" ]]; then
    "$v4_finalizer_launch_runner"
    echo "TPDNERV8V4TAIL_FINALIZER_IDEMPOTENT_COMPLETE unit=$v4_finalizer_launch_unit marker=$v4_finalizer_launch_marker"
    exit 0
fi

v4_finalizer_launch_load_state="$(
    systemctl --user show "$v4_finalizer_launch_unit" \
        --property=LoadState \
        --value \
        2>/dev/null || true
)"
if [[ "$v4_finalizer_launch_load_state" == "loaded" ]]; then
    echo "TPDNERV8V4TAIL_FINALIZER_LAUNCH_ABORT reason=inactive_existing_unit_without_completion unit=$v4_finalizer_launch_unit" >&2
    exit 64
fi

systemd-run --user \
    --collect \
    --unit="${v4_finalizer_launch_unit%.service}" \
    --description="SCTransNet TPD-NER V4 Tail-Aware formal800 automatic evaluator and aggregator" \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartPreventExitStatus=64 \
    --property=RestartSec=60 \
    --property=StartLimitIntervalSec=0 \
    --property=TimeoutStopSec=300 \
    "$v4_finalizer_launch_runner"

echo "TPDNERV8V4TAIL_FINALIZER_LAUNCHED unit=$v4_finalizer_launch_unit best_gpu=2 best_miou_gpu=3 logical_device=cuda:0"
