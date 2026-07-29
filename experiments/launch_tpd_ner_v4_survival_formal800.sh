#!/usr/bin/env bash
set -euo pipefail

tss_mode="launch"
if [[ "$#" -gt 1 ]]; then
    echo "usage: $0 [--preflight|--status]" >&2
    exit 2
fi
case "${1:-}" in
    "")
        ;;
    --preflight)
        tss_mode="preflight"
        ;;
    --status)
        tss_mode="status"
        ;;
    *)
        echo "usage: $0 [--preflight|--status]" >&2
        exit 2
        ;;
esac

tss_repo="${TPD_NER_V4_SURVIVAL_REPO:-/home/ly/SCTransNet_main}"
tss_lane="$tss_repo/experiments/run_tpd_ner_v4_survival_formal800_lane.sh"
tss_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
tss_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
tss_control_unit="sctransnet-tss-control-gpu2"
tss_on_unit="sctransnet-tss-on-gpu3"

if [[ "$tss_mode" == "status" ]]; then
    for tss_unit in "$tss_control_unit" "$tss_on_unit"; do
        systemctl --user show "$tss_unit.service" \
            --property=Id,ActiveState,SubState,Result,ExecMainStatus,NRestarts
    done
    exit 0
fi

[[ -d "$tss_repo" && ! -L "$tss_repo" ]] || {
    echo "TSS_LAUNCH_ABORT reason=invalid_repo path=$tss_repo" >&2
    exit 1
}
[[ -x "$tss_lane" && ! -L "$tss_lane" ]] || {
    echo "TSS_LAUNCH_ABORT reason=lane_not_executable path=$tss_lane" >&2
    exit 1
}
cd "$tss_repo"

tss_control_preflight="$(
    "$tss_lane" --preflight tss_control 2 "$tss_gpu2_uuid"
)"
tss_on_preflight="$(
    "$tss_lane" --preflight tss_on 3 "$tss_gpu3_uuid"
)"
printf '%s\n' "$tss_control_preflight"
printf '%s\n' "$tss_on_preflight"
[[ "$tss_control_preflight" == *"TSS_LANE_READY variant=tss_control seed=42 epochs=800 physical_gpu=2 "* ]] || {
    echo "TSS_LAUNCH_ABORT reason=control_preflight_identity_missing" >&2
    exit 1
}
[[ "$tss_on_preflight" == *"TSS_LANE_READY variant=tss_on seed=42 epochs=800 physical_gpu=3 "* ]] || {
    echo "TSS_LAUNCH_ABORT reason=tss_preflight_identity_missing" >&2
    exit 1
}
if [[ "$tss_mode" == "preflight" ]]; then
    echo "TSS_PAIRED_PREFLIGHT_COMPLETE seed=42 epochs=800 physical_gpus=2,3"
    exit 0
fi

tss_result_root="${TPD_NER_V4_SURVIVAL_RESULT_ROOT:-$tss_repo/experiments/results/tpd_ner_v4_survival_exact_v1}"
mkdir -p "$tss_result_root/.launcher_locks"
tss_claim="$tss_result_root/.launcher_locks/formal800_seed42_paired.lock"
[[ ! -L "$tss_claim" ]] || {
    echo "TSS_LAUNCH_ABORT reason=invalid_claim_file path=$tss_claim" >&2
    exit 1
}
exec 9>"$tss_claim"
if ! flock -n 9; then
    echo "TSS_LAUNCH_ABORT reason=paired_launch_claim_busy path=$tss_claim" >&2
    exit 1
fi

for tss_unit in "$tss_control_unit" "$tss_on_unit"; do
    tss_state="$(
        systemctl --user show \
            "$tss_unit.service" \
            --property=ActiveState \
            --value \
            2>/dev/null || true
    )"
    if [[ "$tss_state" == "active" || "$tss_state" == "activating" ]]; then
        echo "TSS_LAUNCH_ABORT reason=unit_already_active unit=$tss_unit" >&2
        exit 1
    fi
    systemctl --user reset-failed "$tss_unit.service" >/dev/null 2>&1 || true
done

if [[ "$tss_control_preflight" != *" initialization=complete "* ]]; then
    systemd-run --user \
        --unit "$tss_control_unit" \
        --collect \
        --property=Type=exec \
        --property=Restart=on-failure \
        --property=RestartSec=10 \
        "$tss_lane" tss_control 2 "$tss_gpu2_uuid"
fi
if [[ "$tss_on_preflight" != *" initialization=complete "* ]]; then
    systemd-run --user \
        --unit "$tss_on_unit" \
        --collect \
        --property=Type=exec \
        --property=Restart=on-failure \
        --property=RestartSec=10 \
        "$tss_lane" tss_on 3 "$tss_gpu3_uuid"
fi

echo "TSS_PAIRED_LAUNCHED seed=42 epochs=800 control_gpu=2 tss_gpu=3 tss_weight=0.005"
