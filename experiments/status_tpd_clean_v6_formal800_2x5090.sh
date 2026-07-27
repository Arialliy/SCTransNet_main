#!/usr/bin/env bash
set -euo pipefail

v6_repo="/home/ly/SCTransNet_main"
v6_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v6_root="$v6_repo/experiments/results/tpd_clean_v6_formal800_2x5090_v1"
v6_tag="formal800_exact_fp32_2x5090_v1"
v6_variants=(
    tpd_clean_v6_full
    tpd_clean_v6_phase_capacity
    tpd_clean_v6_full
    tpd_clean_v6_phase_capacity
)
v6_seeds=(42 42 3407 3407)
v6_physical_gpus=(2 3 3 2)
v6_lane_units=(
    sctransnet-tpd-clean-v6-gpu2-lane.service
    sctransnet-tpd-clean-v6-gpu3-lane.service
    sctransnet-tpd-clean-v6-gpu3-lane.service
    sctransnet-tpd-clean-v6-gpu2-lane.service
)
v6_lane_positions=(0 0 1 1)

cd "$v6_repo"

for v6_index in "${!v6_variants[@]}"; do
    v6_variant="${v6_variants[$v6_index]}"
    v6_seed="${v6_seeds[$v6_index]}"
    v6_physical_gpu="${v6_physical_gpus[$v6_index]}"
    v6_unit="${v6_lane_units[$v6_index]}"
    v6_position="${v6_lane_positions[$v6_index]}"
    v6_run_dir="$v6_root/NUDT-SIRST/$v6_variant/seed_${v6_seed}_${v6_tag}"
    v6_metrics="$v6_run_dir/metrics.jsonl"
    v6_summary="$v6_run_dir/summary.json"

    v6_lane_active="not-found"
    if systemctl --user cat "$v6_unit" >/dev/null 2>&1; then
        v6_lane_active="$(
            systemctl --user show "$v6_unit" -p ActiveState --value
        )"
    fi

    v6_latest="$(
        "$v6_python" - "$v6_metrics" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    print("epoch=0")
    raise SystemExit(0)
events = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not events:
    print("epoch=0")
    raise SystemExit(0)
event = events[-1]
print(
    "epoch={epoch} pd={pd:.9f} fa={fa:.9g} "
    "miou={miou:.9f} tiny_pd={tiny_pd:.9f}".format(
        epoch=int(event["epoch"]),
        pd=float(event["pd"]),
        fa=float(event["fa"]),
        miou=float(event["miou"]),
        tiny_pd=float(event["tiny_pd"]),
    )
)
PY
    )"

    v6_completion="$(
        "$v6_python" - "$v6_summary" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    print("pending")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") == "complete":
    print(
        "complete"
        f":best_pd_epoch={payload.get('best_pd_epoch')}"
        f":best_miou_epoch={payload.get('best_miou_epoch')}"
    )
else:
    print(str(payload.get("status", "incomplete")))
PY
    )"

    v6_task_active="false"
    if [[ "$v6_lane_active" == "active" && "$v6_completion" != complete:* ]]; then
        if [[ "$v6_position" == "0" ]]; then
            v6_task_active="true"
        else
            if [[ "$v6_physical_gpu" == "2" ]]; then
                v6_prior="$v6_root/NUDT-SIRST/tpd_clean_v6_full/seed_42_${v6_tag}/summary.json"
            else
                v6_prior="$v6_root/NUDT-SIRST/tpd_clean_v6_phase_capacity/seed_42_${v6_tag}/summary.json"
            fi
            if "$v6_python" - "$v6_prior" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit(1)
raise SystemExit(
    0
    if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    else 1
)
PY
            then
                v6_task_active="true"
            fi
        fi
    fi

    echo "TPDCLEANV6_STATUS variant=$v6_variant seed=$v6_seed physical_gpu=$v6_physical_gpu unit=$v6_unit lane_active=$v6_lane_active active=$v6_task_active latest=[$v6_latest] summary=[$v6_completion]"
done
