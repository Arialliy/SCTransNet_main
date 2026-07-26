#!/usr/bin/env bash
set -euo pipefail

v3_repo="/home/ly/SCTransNet_main"
v3_result_root="$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1"
v3_run_tag="screen800_pd_fp32_shared4x5090_v1"
v3_variants=(
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
)
v3_seeds=(42 42 3407 3407)
v3_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)

cd "$v3_repo"
for v3_index in "${!v3_variants[@]}"; do
    v3_variant="${v3_variants[$v3_index]}"
    v3_seed="${v3_seeds[$v3_index]}"
    v3_tag="${v3_unit_tags[$v3_index]}"
    v3_unit="sctransnet-tpd-clean-v3-$v3_tag.service"
    v3_run_name="seed_${v3_seed}_${v3_run_tag}"
    v3_run_dir="$v3_result_root/NUDT-SIRST/$v3_variant/$v3_run_name"
    v3_state="$(systemctl --user is-active "$v3_unit" 2>/dev/null || true)"
    v3_epochs=0
    v3_last="not-started"
    if [[ -f "$v3_run_dir/metrics.jsonl" ]]; then
        v3_epochs="$(wc -l < "$v3_run_dir/metrics.jsonl")"
        v3_last="$(tail -n 1 "$v3_run_dir/metrics.jsonl")"
    fi
    v3_summary="pending"
    if [[ -f "$v3_run_dir/summary.json" ]]; then
        v3_summary="$(
            jq -c '{status,best_pd_epoch,best_miou_epoch}' \
                "$v3_run_dir/summary.json"
        )"
    fi
    echo "job=$v3_tag variant=$v3_variant seed=$v3_seed unit=$v3_state epochs=$v3_epochs summary=$v3_summary"
    echo "last=$v3_last"
done

nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits
