#!/usr/bin/env bash
set -euo pipefail

v4_repo="/home/ly/SCTransNet_main"
v4_result_root="$v4_repo/experiments/results/tpd_clean_v4_screen800_2x5090_v1"
v4_run_tag="screen800_pd_fp32_shared2x5090_v1"
v4_variants=(
    tpd_clean_v4_full
    tpd_clean_v4_sal_capacity
    tpd_clean_v4_full
    tpd_clean_v4_sal_capacity
)
v4_seeds=(42 42 3407 3407)
v4_gpu_uuids=(
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
)
v4_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)

cd "$v4_repo"
for v4_index in "${!v4_variants[@]}"; do
    v4_variant="${v4_variants[$v4_index]}"
    v4_seed="${v4_seeds[$v4_index]}"
    v4_gpu_uuid="${v4_gpu_uuids[$v4_index]}"
    v4_tag="${v4_unit_tags[$v4_index]}"
    v4_unit="sctransnet-tpd-clean-v4-2x-$v4_tag.service"
    v4_run_name="seed_${v4_seed}_${v4_run_tag}"
    v4_run_dir="$v4_result_root/NUDT-SIRST/$v4_variant/$v4_run_name"
    v4_state="$(systemctl --user is-active "$v4_unit" 2>/dev/null || true)"
    v4_epochs=0
    v4_last="not-started"
    if [[ -f "$v4_run_dir/metrics.jsonl" ]]; then
        v4_epochs="$(wc -l < "$v4_run_dir/metrics.jsonl")"
        v4_last="$(tail -n 1 "$v4_run_dir/metrics.jsonl")"
    fi
    v4_summary="pending"
    if [[ -f "$v4_run_dir/summary.json" ]]; then
        v4_summary="$(
            jq -c '{status,best_pd_epoch,best_miou_epoch}' \
                "$v4_run_dir/summary.json"
        )"
    fi
    echo "job=$v4_tag variant=$v4_variant seed=$v4_seed gpu_uuid=$v4_gpu_uuid unit=$v4_state epochs=$v4_epochs summary=$v4_summary"
    echo "last=$v4_last"
done

for v4_gpu_uuid in \
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562 \
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3; do
    nvidia-smi -i "$v4_gpu_uuid" \
        --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits
done
