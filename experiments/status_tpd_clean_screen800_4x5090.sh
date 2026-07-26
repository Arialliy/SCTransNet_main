#!/usr/bin/env bash
set -euo pipefail

clean_repo="/home/ly/SCTransNet_main"
clean_result_root="$clean_repo/experiments/results/tpd_clean_screen800_4x5090_v1"
clean_run_name="seed_42_screen800_pd_fp32_shared4x5090_v1"
clean_variants=(grouped_keep tpd_clean_ctx tpd_clean_sal tpd_clean_full)

cd "$clean_repo"
for clean_variant in "${clean_variants[@]}"; do
    clean_unit="sctransnet-tpd-clean-screen800-$clean_variant.service"
    clean_run_dir="$clean_result_root/NUDT-SIRST/$clean_variant/$clean_run_name"
    clean_state="$(systemctl --user is-active "$clean_unit" 2>/dev/null || true)"
    clean_epochs=0
    clean_last="not-started"
    if [[ -f "$clean_run_dir/metrics.jsonl" ]]; then
        clean_epochs="$(wc -l < "$clean_run_dir/metrics.jsonl")"
        clean_last="$(tail -n 1 "$clean_run_dir/metrics.jsonl")"
    fi
    clean_summary="pending"
    if [[ -f "$clean_run_dir/summary.json" ]]; then
        clean_summary="$(jq -c '{status,best_pd_epoch,best_miou_epoch}' "$clean_run_dir/summary.json")"
    fi
    echo "variant=$clean_variant unit=$clean_state epochs=$clean_epochs summary=$clean_summary"
    echo "last=$clean_last"
done

nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits
