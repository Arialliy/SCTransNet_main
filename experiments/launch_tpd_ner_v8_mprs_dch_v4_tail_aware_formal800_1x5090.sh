#!/usr/bin/env bash
set -euo pipefail

v4_mode="launch"
v4_mode_seen=0
v4_physical_index=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --preflight)
            [[ "$v4_mode_seen" -eq 0 ]] || {
                echo "choose only one of --preflight and --status" >&2
                exit 2
            }
            v4_mode="preflight"
            v4_mode_seen=1
            shift
            ;;
        --status)
            [[ "$v4_mode_seen" -eq 0 ]] || {
                echo "choose only one of --preflight and --status" >&2
                exit 2
            }
            v4_mode="status"
            v4_mode_seen=1
            shift
            ;;
        --physical-gpu)
            [[ -z "$v4_physical_index" && "$#" -ge 2 ]] || {
                echo "provide --physical-gpu exactly once" >&2
                exit 2
            }
            v4_physical_index="$2"
            shift 2
            ;;
        *)
            echo "usage: $0 [--preflight|--status] --physical-gpu {2|3}" >&2
            exit 2
            ;;
    esac
done
if [[ "$v4_physical_index" != "2" && "$v4_physical_index" != "3" ]]; then
    echo "usage: $0 [--preflight|--status] --physical-gpu {2|3}" >&2
    exit 2
fi

v4_repo="${TPD_NER_V8_V4_TAIL_AWARE_REPO:-/home/ly/SCTransNet_main}"
v4_lane="$v4_repo/experiments/run_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_1x5090_lane.sh"
v4_result_root="${TPD_NER_V8_V4_TAIL_AWARE_RESULT_ROOT:-$v4_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1}"
v4_variant="tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
v4_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v4_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v4_gpu2_unit="sctransnet-tpd-ner-v8-v4-tail-aware-gpu2"
v4_gpu3_unit="sctransnet-tpd-ner-v8-v4-tail-aware-gpu3"

if [[ "$v4_physical_index" == "2" ]]; then
    v4_gpu_uuid="$v4_gpu2_uuid"
    v4_selected_unit="$v4_gpu2_unit"
else
    v4_gpu_uuid="$v4_gpu3_uuid"
    v4_selected_unit="$v4_gpu3_unit"
fi

if [[ "$v4_mode" == "status" ]]; then
    systemctl --user show "$v4_selected_unit.service" \
        --property=Id,ActiveState,SubState,Result,ExecMainStatus,NRestarts
    exit 0
fi

[[ -d "$v4_repo" && ! -L "$v4_repo" ]] || {
    echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=invalid_repo path=$v4_repo" >&2
    exit 1
}
cd "$v4_repo"
[[ -x "$v4_lane" && ! -L "$v4_lane" ]] || {
    echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=lane_not_executable path=$v4_lane" >&2
    exit 1
}
if [[ -L "$v4_result_root" || ( -e "$v4_result_root" && ! -d "$v4_result_root" ) ]]; then
    echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=invalid_result_root path=$v4_result_root" >&2
    exit 1
fi

# Preflight is read-only: it does not create result directories or units.
if [[ "$v4_mode" == "preflight" ]]; then
    v4_preflight_output="$(
        "$v4_lane" --preflight "$v4_physical_index" "$v4_gpu_uuid"
    )"
    printf '%s\n' "$v4_preflight_output"
    v4_ready_prefix="TPDNERV8V4TAIL_LANE_READY variant=$v4_variant seed=42 epochs=800 physical_gpu=$v4_physical_index initialization="
    [[ "$v4_preflight_output" == *"$v4_ready_prefix"* ]] || {
        echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=lane_preflight_identity_missing" >&2
        exit 1
    }
    echo "TPDNERV8V4TAIL_PREFLIGHT_COMPLETE variant=$v4_variant seed=42 epochs=800 physical_gpu=$v4_physical_index"
    exit 0
fi

mkdir -p "$v4_result_root"
v4_claim_dir="$v4_result_root/.launcher_locks"
if [[ -L "$v4_claim_dir" || ( -e "$v4_claim_dir" && ! -d "$v4_claim_dir" ) ]]; then
    echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=invalid_claim_directory path=$v4_claim_dir" >&2
    exit 1
fi
mkdir -p "$v4_claim_dir"
v4_claim_path="$v4_claim_dir/formal800_v4_tail_aware_global.lock"
if [[ -L "$v4_claim_path" ]]; then
    echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=invalid_claim_file path=$v4_claim_path" >&2
    exit 1
fi
exec 9>"$v4_claim_path"
if ! flock -n 9; then
    echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=v4_global_claim_busy path=$v4_claim_path" >&2
    exit 1
fi

v4_preflight_output="$(
    "$v4_lane" --preflight "$v4_physical_index" "$v4_gpu_uuid"
)"
printf '%s\n' "$v4_preflight_output"
v4_ready_prefix="TPDNERV8V4TAIL_LANE_READY variant=$v4_variant seed=42 epochs=800 physical_gpu=$v4_physical_index initialization="
[[ "$v4_preflight_output" == *"$v4_ready_prefix"* ]] || {
    echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=lane_preflight_identity_missing" >&2
    exit 1
}
if [[ "$v4_preflight_output" == *"$v4_ready_prefix""complete "* ]]; then
    echo "TPDNERV8V4TAIL_IDEMPOTENT_COMPLETE variant=$v4_variant seed=42 physical_gpu=$v4_physical_index"
    exit 0
fi

# The seed-42 V4 trajectory is singular; GPU2 and GPU3 are alternative lanes,
# not two independent copies of the same run.
for v4_unit in "$v4_gpu2_unit" "$v4_gpu3_unit"; do
    v4_state="$(
        systemctl --user show \
            "$v4_unit.service" \
            --property=ActiveState \
            --value \
            2>/dev/null || true
    )"
    if [[ "$v4_state" == "active" || "$v4_state" == "activating" ]]; then
        echo "TPDNERV8V4TAIL_LAUNCH_ABORT reason=v4_unit_already_active unit=$v4_unit" >&2
        exit 1
    fi
done
systemctl --user reset-failed "$v4_selected_unit.service" >/dev/null 2>&1 || true

systemd-run --user \
    --unit "$v4_selected_unit" \
    --collect \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartSec=10 \
    "$v4_lane" \
    "$v4_physical_index" \
    "$v4_gpu_uuid"

echo "TPDNERV8V4TAIL_LAUNCHED variant=$v4_variant seed=42 epochs=800 physical_gpu=$v4_physical_index"
