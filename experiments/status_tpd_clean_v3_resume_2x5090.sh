#!/usr/bin/env bash
set -euo pipefail

v3_repo="/home/ly/SCTransNet_main"
v3_result_root="$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1"
v3_run_tag="screen800_pd_fp32_shared4x5090_v1"
v3_resume_root="$v3_result_root/resume_2x5090_v1"
v3_resume_manifest_root="$v3_resume_root/manifests"
v3_resume_log_root="$v3_resume_root/logs"
v3_variants=(
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
)
v3_seeds=(42 42 3407 3407)
v3_gpu_uuids=(
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
)
v3_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)
v3_old_units=(
    sctransnet-tpd-clean-v3-full-s42.service
    sctransnet-tpd-clean-v3-cap-s42.service
    sctransnet-tpd-clean-v3-full-s3407.service
    sctransnet-tpd-clean-v3-cap-s3407.service
)

cd "$v3_repo"
for v3_index in "${!v3_variants[@]}"; do
    v3_variant="${v3_variants[$v3_index]}"
    v3_seed="${v3_seeds[$v3_index]}"
    v3_gpu_uuid="${v3_gpu_uuids[$v3_index]}"
    v3_tag="${v3_unit_tags[$v3_index]}"
    v3_old_unit="${v3_old_units[$v3_index]}"
    v3_new_unit="sctransnet-tpd-clean-v3-resume-2x-$v3_tag.service"
    v3_run_dir="$v3_result_root/NUDT-SIRST/$v3_variant/seed_${v3_seed}_${v3_run_tag}"
    v3_manifest="$v3_resume_manifest_root/${v3_variant}_seed${v3_seed}.json"
    v3_log="$v3_resume_log_root/${v3_variant}_seed${v3_seed}.log"

    v3_old_state="$(
        systemctl --user is-active "$v3_old_unit" 2>/dev/null || true
    )"
    v3_new_state="$(
        systemctl --user is-active "$v3_new_unit" 2>/dev/null || true
    )"
    v3_epochs=0
    v3_last="not-started"
    if [[ -f "$v3_run_dir/metrics.jsonl" ]]; then
        v3_epochs="$(wc -l < "$v3_run_dir/metrics.jsonl")"
        v3_last="$(tail -n 1 "$v3_run_dir/metrics.jsonl")"
    fi
    v3_resume_boundary="pending"
    if [[ -f "$v3_manifest" ]]; then
        v3_resume_boundary="$(
            jq -c \
                '{schema,boundary_epoch,target_epoch,original_gpu_uuid,resume_gpu_uuid,resume_gpu_index}' \
                "$v3_manifest"
        )"
    fi
    v3_summary="pending"
    if [[ -f "$v3_run_dir/summary.json" ]]; then
        v3_summary="$(
            jq -c '{status,best_pd_epoch,best_miou_epoch}' \
                "$v3_run_dir/summary.json"
        )"
    fi
    v3_sweeps=0
    for v3_sweep in \
        "$v3_run_dir/pd_fa_sweep_best.pth.json" \
        "$v3_run_dir/pd_fa_sweep_best_miou.pth.json"; do
        if [[ -f "$v3_sweep" ]]; then
            v3_sweeps="$((v3_sweeps + 1))"
        fi
    done
    v3_complete_marker="pending"
    if [[ -f "$v3_log" ]] &&
        rg -q \
            "^TPDCLEANV3_RESUME_2X_COMPLETE variant=$v3_variant seed=$v3_seed " \
            "$v3_log"; then
        v3_complete_marker="present"
    fi
    echo "job=$v3_tag variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_gpu_uuid old_unit=$v3_old_state resume_unit=$v3_new_state epochs=$v3_epochs sweeps=$v3_sweeps completion=$v3_complete_marker summary=$v3_summary"
    echo "resume_boundary=$v3_resume_boundary"
    echo "last=$v3_last"
done

for v3_gpu_uuid in \
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562 \
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3; do
    nvidia-smi -i "$v3_gpu_uuid" \
        --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits
done
