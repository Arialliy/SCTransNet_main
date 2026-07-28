#!/usr/bin/env bash
set -euo pipefail

v2_mode="launch"
v2_physical_index=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --preflight)
            v2_mode="preflight"
            shift
            ;;
        --status)
            v2_mode="status"
            shift
            ;;
        --physical-gpu)
            [[ "$#" -ge 2 ]] || {
                echo "missing value for --physical-gpu" >&2
                exit 2
            }
            v2_physical_index="$2"
            shift 2
            ;;
        *)
            echo "usage: $0 [--preflight|--status] --physical-gpu {2|3}" >&2
            exit 2
            ;;
    esac
done
if [[ "$v2_physical_index" != "2" && "$v2_physical_index" != "3" ]]; then
    echo "usage: $0 [--preflight|--status] --physical-gpu {2|3}" >&2
    exit 2
fi

v2_repo="${TPD_NER_V8_V2_REPO:-/home/ly/SCTransNet_main}"
v2_lane="$v2_repo/experiments/run_tpd_ner_v8_mprs_dch_v2_formal800_1x5090_lane.sh"
v2_manifest_tool="$v2_repo/experiments/freeze_tpd_ner_v8_mprs_dch_v2_source_locks.py"
v2_python="${TPD_NER_V8_V2_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v2_source_lock="${TPD_NER_V8_V2_SOURCE_LOCK:-$v2_repo/experiments/tpd_ner_v8_mprs_dch_v2_exact_source_lock.json}"
v2_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v2_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v2_gpu2_unit="sctransnet-tpd-ner-v8-v2-relay-on-gpu2"
v2_gpu3_unit="sctransnet-tpd-ner-v8-v2-relay-on-gpu3"

if [[ "$v2_physical_index" == "2" ]]; then
    v2_gpu_uuid="$v2_gpu2_uuid"
    v2_selected_unit="$v2_gpu2_unit"
else
    v2_gpu_uuid="$v2_gpu3_uuid"
    v2_selected_unit="$v2_gpu3_unit"
fi

cd "$v2_repo"
[[ -x "$v2_lane" ]] || {
    echo "TPDNERV8V2_LAUNCH_ABORT reason=lane_not_executable path=$v2_lane" >&2
    exit 1
}
[[ -x "$v2_python" ]] || {
    echo "TPDNERV8V2_LAUNCH_ABORT reason=python_not_executable path=$v2_python" >&2
    exit 1
}
[[ -f "$v2_manifest_tool" && ! -L "$v2_manifest_tool" ]] || {
    echo "TPDNERV8V2_LAUNCH_ABORT reason=manifest_tool_missing path=$v2_manifest_tool" >&2
    exit 1
}
[[ -f "$v2_source_lock" && ! -L "$v2_source_lock" ]] || {
    echo "TPDNERV8V2_LAUNCH_ABORT reason=training_manifest_missing path=$v2_source_lock" >&2
    exit 1
}

if [[ "$v2_mode" == "status" ]]; then
    systemctl --user show "$v2_selected_unit.service" \
        --property=Id,ActiveState,SubState,Result,ExecMainStatus,NRestarts
    exit 0
fi

"$v2_python" "$v2_manifest_tool" \
    --mode verify \
    --kind training \
    --training-lock "$v2_source_lock"

"$v2_lane" --preflight "$v2_physical_index" "$v2_gpu_uuid"
if [[ "$v2_mode" == "preflight" ]]; then
    echo "TPDNERV8V2_PREFLIGHT_COMPLETE variant=tpd_ner_v8_mprs_dch_v2_full_relay_on seed=42 physical_gpu=$v2_physical_index"
    exit 0
fi

for v2_unit in "$v2_gpu2_unit" "$v2_gpu3_unit"; do
    v2_state="$(
        systemctl --user show \
            "$v2_unit.service" \
            --property=ActiveState \
            --value \
            2>/dev/null || true
    )"
    if [[ "$v2_state" == "active" || "$v2_state" == "activating" ]]; then
        echo "TPDNERV8V2_LAUNCH_ABORT reason=v2_unit_already_active unit=$v2_unit" >&2
        exit 1
    fi
done
systemctl --user reset-failed "$v2_selected_unit.service" >/dev/null 2>&1 || true

systemd-run --user \
    --unit "$v2_selected_unit" \
    --collect \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartSec=10 \
    "$v2_lane" \
    "$v2_physical_index" \
    "$v2_gpu_uuid"

echo "TPDNERV8V2_LAUNCHED variant=tpd_ner_v8_mprs_dch_v2_full_relay_on seed=42 physical_gpu=$v2_physical_index"
