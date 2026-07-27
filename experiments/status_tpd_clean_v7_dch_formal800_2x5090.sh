#!/usr/bin/env bash
set -euo pipefail

dch_repo="/home/ly/SCTransNet_main"
dch_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
dch_root="$dch_repo/experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
dch_tag="formal800_exact_fp32_2x5090_v1"
dch_variants=(
    tpd_clean_v7_dch_full
    tpd_clean_v7_dch_capacity
    tpd_clean_v7_dch_capacity
    tpd_clean_v7_dch_full
)
dch_seeds=(42 42 3407 3407)
dch_physical_gpus=(2 3 2 3)
dch_gpu_uuids=(
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
)
dch_lane_units=(
    sctransnet-tpd-clean-v7-dch-gpu2-lane.service
    sctransnet-tpd-clean-v7-dch-gpu3-lane.service
    sctransnet-tpd-clean-v7-dch-gpu2-lane.service
    sctransnet-tpd-clean-v7-dch-gpu3-lane.service
)
dch_lane_positions=(0 0 1 1)

cd "$dch_repo"

for dch_index in "${!dch_variants[@]}"; do
    dch_variant="${dch_variants[$dch_index]}"
    dch_seed="${dch_seeds[$dch_index]}"
    dch_physical_gpu="${dch_physical_gpus[$dch_index]}"
    dch_gpu_uuid="${dch_gpu_uuids[$dch_index]}"
    dch_unit="${dch_lane_units[$dch_index]}"
    dch_position="${dch_lane_positions[$dch_index]}"
    dch_run_dir="$dch_root/NUDT-SIRST/$dch_variant/seed_${dch_seed}_${dch_tag}"
    dch_metrics="$dch_run_dir/metrics.jsonl"
    dch_summary="$dch_run_dir/summary.json"
    dch_assignment="$dch_root/lane_assignments/${dch_variant}_seed${dch_seed}.json"

    dch_lane_active="not-found"
    if systemctl --user cat "$dch_unit" >/dev/null 2>&1; then
        dch_lane_active="$(
            systemctl --user show "$dch_unit" -p ActiveState --value
        )"
    fi

    dch_latest="$(
        "$dch_python" - "$dch_metrics" <<'PY'
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
    "miou={miou:.9f} tiny_pd={tiny_pd:.9f} "
    "false_objects_per_image={false_objects:.9f} "
    "pixel_f1={pixel_f1:.9f}".format(
        epoch=int(event["epoch"]),
        pd=float(event["pd"]),
        fa=float(event["fa"]),
        miou=float(event["miou"]),
        tiny_pd=float(event["tiny_pd"]),
        false_objects=float(event["false_objects_per_image"]),
        pixel_f1=float(event["pixel_f1"]),
    )
)
PY
    )"

    dch_completion="$(
        "$dch_python" - "$dch_summary" <<'PY'
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
        f":stored_metrics={len(payload.get('stored_validation_metrics', []))}"
    )
else:
    print(str(payload.get("status", "incomplete")))
PY
    )"

    dch_assignment_state="$(
        "$dch_python" - \
            "$dch_assignment" \
            "$dch_physical_gpu" \
            "$dch_gpu_uuid" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_index = int(sys.argv[2])
expected_uuid = sys.argv[3]
if not path.is_file() or path.is_symlink():
    print("pending")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("physical_gpu_index") == expected_index
    and payload.get("physical_gpu_uuid") == expected_uuid
):
    print("verified")
else:
    print("mismatch")
PY
    )"

    dch_task_active="false"
    if [[ "$dch_lane_active" == "active" && "$dch_completion" != complete:* ]]; then
        if [[ "$dch_position" == "0" ]]; then
            dch_task_active="true"
        else
            if [[ "$dch_physical_gpu" == "2" ]]; then
                dch_prior="$dch_root/NUDT-SIRST/tpd_clean_v7_dch_full/seed_42_${dch_tag}/summary.json"
            else
                dch_prior="$dch_root/NUDT-SIRST/tpd_clean_v7_dch_capacity/seed_42_${dch_tag}/summary.json"
            fi
            if "$dch_python" - "$dch_prior" <<'PY' >/dev/null 2>&1
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
                dch_task_active="true"
            fi
        fi
    fi

    echo "TPDCLEANV7DCH_STATUS variant=$dch_variant seed=$dch_seed physical_gpu=$dch_physical_gpu gpu_uuid=$dch_gpu_uuid unit=$dch_unit lane_active=$dch_lane_active active=$dch_task_active assignment=$dch_assignment_state latest=[$dch_latest] summary=[$dch_completion]"
done
