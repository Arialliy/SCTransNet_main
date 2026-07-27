#!/usr/bin/env bash
set -euo pipefail

dch_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    dch_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 [--preflight] LANE GPU_UUID" >&2
    exit 2
fi

dch_lane="$1"
dch_gpu_uuid="$2"
dch_repo="/home/ly/SCTransNet_main"
dch_worker="$dch_repo/experiments/run_tpd_clean_v7_dch_formal800_2x5090_worker.sh"

case "$dch_lane:$dch_gpu_uuid" in
    gpu2:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        dch_variants=(
            tpd_clean_v7_dch_full
            tpd_clean_v7_dch_capacity
        )
        dch_seeds=(42 3407)
        ;;
    gpu3:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        dch_variants=(
            tpd_clean_v7_dch_capacity
            tpd_clean_v7_dch_full
        )
        dch_seeds=(42 3407)
        ;;
    *)
        echo "TPDCLEANV7DCH_2X_LANE_ABORT reason=invalid_lane_mapping lane=$dch_lane gpu_uuid=$dch_gpu_uuid" >&2
        exit 2
        ;;
esac

[[ -x "$dch_worker" ]] || {
    echo "TPDCLEANV7DCH_2X_LANE_ABORT reason=worker_not_executable path=$dch_worker" >&2
    exit 1
}

for dch_index in 0 1; do
    dch_variant="${dch_variants[$dch_index]}"
    dch_seed="${dch_seeds[$dch_index]}"
    if [[ "$dch_lane_mode" == "preflight" ]]; then
        "$dch_worker" --preflight "$dch_variant" "$dch_seed" "$dch_gpu_uuid"
    else
        "$dch_worker" "$dch_variant" "$dch_seed" "$dch_gpu_uuid"
    fi
done

echo "TPDCLEANV7DCH_2X_LANE_COMPLETE lane=$dch_lane gpu_uuid=$dch_gpu_uuid tasks=2 mode=$dch_lane_mode"
