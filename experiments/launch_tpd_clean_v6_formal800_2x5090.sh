#!/usr/bin/env bash
set -euo pipefail

v6_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v6_mode="preflight"
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

v6_repo="/home/ly/SCTransNet_main"
v6_lane_runner="$v6_repo/experiments/run_tpd_clean_v6_formal800_2x5090_lane.sh"
v6_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v6_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v6_gpu2_unit="sctransnet-tpd-clean-v6-gpu2-lane"
v6_gpu3_unit="sctransnet-tpd-clean-v6-gpu3-lane"

cd "$v6_repo"
[[ -x "$v6_lane_runner" ]] || {
    echo "TPDCLEANV6_2X_LAUNCH_ABORT reason=lane_runner_not_executable path=$v6_lane_runner" >&2
    exit 1
}

# Both checks execute synchronously and never start a training process.
"$v6_lane_runner" --preflight gpu2 "$v6_gpu2_uuid"
"$v6_lane_runner" --preflight gpu3 "$v6_gpu3_uuid"
echo "TPDCLEANV6_2X_PREFLIGHT_ALL_OK lanes=2 tasks=4 physical_gpus=2,3 concurrent_tasks_per_gpu=1"
if [[ "$v6_mode" == "preflight" ]]; then
    exit 0
fi

for v6_unit in "$v6_gpu2_unit.service" "$v6_gpu3_unit.service"; do
    if systemctl --user is-active --quiet "$v6_unit"; then
        echo "TPDCLEANV6_2X_LAUNCH_ABORT reason=lane_already_active unit=$v6_unit" >&2
        exit 1
    fi
done

systemd-run --user \
    --collect \
    --unit="$v6_gpu2_unit" \
    --description="SCTransNet TPD-Clean-v6 physical GPU2 serial lane" \
    --property=Restart=on-failure \
    --property=RestartSec=30 \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v6_lane_runner" gpu2 "$v6_gpu2_uuid"
echo "TPDCLEANV6_2X_LANE_STARTED lane=gpu2 unit=$v6_gpu2_unit.service gpu_uuid=$v6_gpu2_uuid"

systemd-run --user \
    --collect \
    --unit="$v6_gpu3_unit" \
    --description="SCTransNet TPD-Clean-v6 physical GPU3 serial lane" \
    --property=Restart=on-failure \
    --property=RestartSec=30 \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v6_lane_runner" gpu3 "$v6_gpu3_uuid"
echo "TPDCLEANV6_2X_LANE_STARTED lane=gpu3 unit=$v6_gpu3_unit.service gpu_uuid=$v6_gpu3_uuid"
