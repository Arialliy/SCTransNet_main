#!/usr/bin/env bash
set -euo pipefail

v6_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v6_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 [--preflight] LANE GPU_UUID" >&2
    exit 2
fi

v6_lane="$1"
v6_gpu_uuid="$2"
v6_repo="/home/ly/SCTransNet_main"
v6_worker="$v6_repo/experiments/run_tpd_clean_v6_formal800_2x5090_worker.sh"

case "$v6_lane:$v6_gpu_uuid" in
    gpu2:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        v6_variants=(
            tpd_clean_v6_full
            tpd_clean_v6_phase_capacity
        )
        v6_seeds=(42 3407)
        ;;
    gpu3:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        v6_variants=(
            tpd_clean_v6_phase_capacity
            tpd_clean_v6_full
        )
        v6_seeds=(42 3407)
        ;;
    *)
        echo "TPDCLEANV6_2X_LANE_ABORT reason=invalid_lane_mapping lane=$v6_lane gpu_uuid=$v6_gpu_uuid" >&2
        exit 2
        ;;
esac

[[ -x "$v6_worker" ]] || {
    echo "TPDCLEANV6_2X_LANE_ABORT reason=worker_not_executable path=$v6_worker" >&2
    exit 1
}

for v6_index in 0 1; do
    v6_variant="${v6_variants[$v6_index]}"
    v6_seed="${v6_seeds[$v6_index]}"
    if [[ "$v6_lane_mode" == "preflight" ]]; then
        "$v6_worker" --preflight "$v6_variant" "$v6_seed" "$v6_gpu_uuid"
    else
        "$v6_worker" "$v6_variant" "$v6_seed" "$v6_gpu_uuid"
    fi
done

echo "TPDCLEANV6_2X_LANE_COMPLETE lane=$v6_lane gpu_uuid=$v6_gpu_uuid tasks=2 mode=$v6_lane_mode"
