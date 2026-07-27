#!/usr/bin/env bash
set -euo pipefail

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
dch_lane_runner="$dch_repo/experiments/run_tpd_clean_v7_dch_formal800_2x5090_lane.sh"
dch_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
dch_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
dch_gpu2_unit="sctransnet-tpd-clean-v7-dch-gpu2-lane"
dch_gpu3_unit="sctransnet-tpd-clean-v7-dch-gpu3-lane"

cd "$dch_repo"
[[ -x "$dch_lane_runner" ]] || {
    echo "TPDCLEANV7DCH_2X_LAUNCH_ABORT reason=lane_runner_not_executable path=$dch_lane_runner" >&2
    exit 1
}

# Both checks are synchronous and do not start a training process.
"$dch_lane_runner" --preflight gpu2 "$dch_gpu2_uuid"
"$dch_lane_runner" --preflight gpu3 "$dch_gpu3_uuid"
echo "TPDCLEANV7DCH_2X_PREFLIGHT_ALL_OK lanes=2 tasks=4 physical_gpus=2,3 concurrent_tasks_per_gpu=1"
if [[ "$dch_mode" == "preflight" ]]; then
    exit 0
fi

for dch_unit in "$dch_gpu2_unit.service" "$dch_gpu3_unit.service"; do
    if systemctl --user is-active --quiet "$dch_unit"; then
        echo "TPDCLEANV7DCH_2X_LAUNCH_ABORT reason=lane_already_active unit=$dch_unit" >&2
        exit 1
    fi
done

systemd-run --user \
    --collect \
    --unit="$dch_gpu2_unit" \
    --description="SCTransNet V7-DCH physical GPU2 serial lane" \
    --property=Restart=on-failure \
    --property=RestartSec=30 \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$dch_lane_runner" gpu2 "$dch_gpu2_uuid"
echo "TPDCLEANV7DCH_2X_LANE_STARTED lane=gpu2 unit=$dch_gpu2_unit.service physical_gpu=2 gpu_uuid=$dch_gpu2_uuid"

systemd-run --user \
    --collect \
    --unit="$dch_gpu3_unit" \
    --description="SCTransNet V7-DCH physical GPU3 serial lane" \
    --property=Restart=on-failure \
    --property=RestartSec=30 \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$dch_lane_runner" gpu3 "$dch_gpu3_uuid"
echo "TPDCLEANV7DCH_2X_LANE_STARTED lane=gpu3 unit=$dch_gpu3_unit.service physical_gpu=3 gpu_uuid=$dch_gpu3_uuid"
