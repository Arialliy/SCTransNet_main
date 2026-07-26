#!/usr/bin/env bash
set -euo pipefail

v5_repo="/home/ly/SCTransNet_main"
v5_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v5_root="$v5_repo/experiments/results/tpd_clean_v5_screen800_2x5090_v1"
v5_tag="screen800_pd_fp32_shared2x5090_v1"

v5_variants=(
    tpd_clean_v5_full
    tpd_clean_v5_sal_capacity
    tpd_clean_v5_full
    tpd_clean_v5_sal_capacity
)
v5_seeds=(42 42 3407 3407)
v5_units=(
    sctransnet-tpd-clean-v5-2x-full-s42.service
    sctransnet-tpd-clean-v5-2x-cap-s42.service
    sctransnet-tpd-clean-v5-2x-full-s3407.service
    sctransnet-tpd-clean-v5-2x-cap-s3407.service
)

cd "$v5_repo"

nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader

for v5_index in "${!v5_variants[@]}"; do
    v5_variant="${v5_variants[$v5_index]}"
    v5_seed="${v5_seeds[$v5_index]}"
    v5_unit="${v5_units[$v5_index]}"
    v5_run_dir="$v5_root/NUDT-SIRST/$v5_variant/seed_${v5_seed}_${v5_tag}"
    v5_metrics="$v5_run_dir/metrics.jsonl"
    v5_summary="$v5_run_dir/summary.json"

    v5_active="not-found"
    v5_result="unknown"
    v5_restarts="0"
    v5_pid="0"
    if systemctl --user cat "$v5_unit" >/dev/null 2>&1; then
        v5_active="$(
            systemctl --user show "$v5_unit" -p ActiveState --value
        )"
        v5_result="$(
            systemctl --user show "$v5_unit" -p Result --value
        )"
        v5_restarts="$(
            systemctl --user show "$v5_unit" -p NRestarts --value
        )"
        v5_pid="$(
            systemctl --user show "$v5_unit" -p MainPID --value
        )"
    fi

    v5_events=0
    v5_last="none"
    if [[ -f "$v5_metrics" && ! -L "$v5_metrics" ]]; then
        v5_events="$(wc -l < "$v5_metrics")"
        v5_last="$(
            tail -n 1 "$v5_metrics" |
                "$v5_python" -c '
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    print("none")
else:
    event = json.loads(raw)
    print(
        "epoch={epoch} pd={pd:.9f} matched={matched} fa={fa:.9g} "
        "miou={miou:.9f} tiny={tiny:.9f}".format(
            epoch=event["epoch"],
            pd=event["pd"],
            matched=event["matched_target_count"],
            fa=event["fa"],
            miou=event["miou"],
            tiny=event["tiny_pd"],
        )
    )
'
        )"
    fi

    v5_final="pending"
    if [[ -f "$v5_summary" && ! -L "$v5_summary" ]]; then
        v5_final="$(
            "$v5_python" - "$v5_summary" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    "status={status} best_epoch={best_epoch} "
    "best_miou_epoch={best_miou_epoch}".format(
        status=summary.get("status"),
        best_epoch=summary.get("best_epoch"),
        best_miou_epoch=summary.get("best_miou_epoch"),
    )
)
PY
        )"
    fi

    echo "TPDCLEANV5_STATUS variant=$v5_variant seed=$v5_seed unit=$v5_unit active=$v5_active result=$v5_result restarts=$v5_restarts pid=$v5_pid events=$v5_events latest=[$v5_last] final=[$v5_final]"
done
