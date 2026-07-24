#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_result_root="$formal_repo/experiments/results/tpd_pe_formal800_4x5090_v1"
formal_run_name="seed_42_formal800_pd_fp32_4x5090_v1"
formal_variants=(original progressive tpd spd)
formal_gpu_uuids=(
    GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70
    GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
)

for formal_index in "${!formal_variants[@]}"; do
    formal_variant="${formal_variants[$formal_index]}"
    formal_uuid="${formal_gpu_uuids[$formal_index]}"
    formal_unit="sctransnet-formal800-4x5090-$formal_variant.service"
    formal_run_dir="$formal_result_root/NUDT-SIRST/$formal_variant/$formal_run_name"
    formal_metrics="$formal_run_dir/metrics.jsonl"
    formal_state="$(systemctl --user show "$formal_unit" --property=ActiveState --value 2>/dev/null || true)"
    formal_substate="$(systemctl --user show "$formal_unit" --property=SubState --value 2>/dev/null || true)"
    formal_pid="$(systemctl --user show "$formal_unit" --property=MainPID --value 2>/dev/null || true)"
    formal_events=0
    formal_epoch=0
    if [[ -f "$formal_metrics" ]]; then
        formal_events="$(wc -l < "$formal_metrics")"
        formal_epoch="$(tail -n 1 "$formal_metrics" | jq -r '.epoch')"
    fi
    formal_memory="$(
        nvidia-smi -i "$formal_uuid" \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null || true
    )"
    printf 'variant=%s gpu_uuid=%s unit=%s/%s pid=%s epoch=%s events=%s gpu_used_mib_util_pct=%s\n' \
        "$formal_variant" \
        "$formal_uuid" \
        "${formal_state:-not-found}" \
        "${formal_substate:-not-found}" \
        "${formal_pid:-0}" \
        "$formal_epoch" \
        "$formal_events" \
        "${formal_memory:-unknown}"
done
