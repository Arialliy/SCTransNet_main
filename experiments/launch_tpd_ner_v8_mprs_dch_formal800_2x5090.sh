#!/usr/bin/env bash
set -euo pipefail

ner_mode="launch"
case "${1:-}" in
    --preflight)
        ner_mode="preflight"
        shift
        ;;
    --status)
        ner_mode="status"
        shift
        ;;
esac
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight|--status]" >&2
    exit 2
fi

ner_repo="${TPD_NER_V8_REPO:-/home/ly/SCTransNet_main}"
ner_lane="$ner_repo/experiments/run_tpd_ner_v8_mprs_dch_formal800_2x5090_lane.sh"
ner_manifest_tool="$ner_repo/experiments/freeze_tpd_ner_v8_mprs_dch_source_locks.py"
ner_python="${TPD_NER_V8_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
ner_source_lock="${TPD_NER_V8_SOURCE_LOCK:-$ner_repo/experiments/tpd_ner_v8_mprs_dch_exact_source_lock.json}"
ner_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
ner_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
ner_gpu2_unit="sctransnet-tpd-ner-v8-relay-off-gpu2"
ner_gpu3_unit="sctransnet-tpd-ner-v8-relay-on-gpu3"

cd "$ner_repo"
[[ -x "$ner_lane" ]] || {
    echo "TPDNERV8_LAUNCH_ABORT reason=lane_not_executable path=$ner_lane" >&2
    exit 1
}
[[ -x "$ner_python" ]] || {
    echo "TPDNERV8_LAUNCH_ABORT reason=python_not_executable path=$ner_python" >&2
    exit 1
}
[[ -f "$ner_manifest_tool" && ! -L "$ner_manifest_tool" ]] || {
    echo "TPDNERV8_LAUNCH_ABORT reason=manifest_tool_missing path=$ner_manifest_tool" >&2
    exit 1
}
[[ -f "$ner_source_lock" && ! -L "$ner_source_lock" ]] || {
    echo "TPDNERV8_LAUNCH_ABORT reason=training_manifest_missing path=$ner_source_lock" >&2
    exit 1
}

if [[ "$ner_mode" == "status" ]]; then
    systemctl --user show "$ner_gpu2_unit.service" \
        --property=Id,ActiveState,SubState,Result,ExecMainStatus,NRestarts
    systemctl --user show "$ner_gpu3_unit.service" \
        --property=Id,ActiveState,SubState,Result,ExecMainStatus,NRestarts
    exit 0
fi

"$ner_python" "$ner_manifest_tool" \
    --mode verify \
    --kind training \
    --training-lock "$ner_source_lock"

"$ner_lane" --preflight \
    tpd_ner_v8_mprs_dch_full_relay_off \
    2 \
    "$ner_gpu2_uuid"
"$ner_lane" --preflight \
    tpd_ner_v8_mprs_dch_full_relay_on \
    3 \
    "$ner_gpu3_uuid"

if [[ "$ner_mode" == "preflight" ]]; then
    echo "TPDNERV8_PREFLIGHT_COMPLETE variants=2 seed=42 gpus=2,3"
    exit 0
fi

for ner_unit in "$ner_gpu2_unit" "$ner_gpu3_unit"; do
    ner_state="$(systemctl --user show "$ner_unit.service" --property=ActiveState --value 2>/dev/null || true)"
    if [[ "$ner_state" == "active" || "$ner_state" == "activating" ]]; then
        echo "TPDNERV8_LAUNCH_ABORT reason=unit_already_active unit=$ner_unit" >&2
        exit 1
    fi
    systemctl --user reset-failed "$ner_unit.service" >/dev/null 2>&1 || true
done

systemd-run --user \
    --unit "$ner_gpu2_unit" \
    --collect \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartSec=10 \
    "$ner_lane" \
    tpd_ner_v8_mprs_dch_full_relay_off \
    2 \
    "$ner_gpu2_uuid"

systemd-run --user \
    --unit "$ner_gpu3_unit" \
    --collect \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartSec=10 \
    "$ner_lane" \
    tpd_ner_v8_mprs_dch_full_relay_on \
    3 \
    "$ner_gpu3_uuid"

echo "TPDNERV8_LAUNCHED relay_off_gpu=2 relay_on_gpu=3 seed=42"
