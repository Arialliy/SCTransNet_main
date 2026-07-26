#!/usr/bin/env bash
set -euo pipefail

v3_repo="/home/ly/SCTransNet_main"
v3_worker="$v3_repo/experiments/run_tpd_clean_v3_screen800_4x5090_worker.sh"
v3_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v3_result_root="$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1"
v3_run_tag="screen800_pd_fp32_shared4x5090_v1"
v3_source_lock="$v3_repo/experiments/tpd_clean_v3_screen800_source_lock.json"
v3_variants=(
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
)
v3_seeds=(42 42 3407 3407)
v3_gpu_uuids=(
    GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70
    GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
)
v3_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)

v3_mode="${1:-run}"
if [[ "$v3_mode" != "run" && "$v3_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$v3_repo"

[[ -x "$v3_worker" ]] || {
    echo "TPDCLEANV3_LAUNCH_ABORT reason=worker_not_executable path=$v3_worker" >&2
    exit 1
}
[[ -x "$v3_python" ]] || {
    echo "TPDCLEANV3_LAUNCH_ABORT reason=python_not_executable path=$v3_python" >&2
    exit 1
}
[[ -f "$v3_source_lock" && ! -L "$v3_source_lock" ]] || {
    echo "TPDCLEANV3_LAUNCH_ABORT reason=missing_source_lock path=$v3_source_lock" >&2
    exit 1
}

"$v3_python" -c '
import cv2
import einops
import ml_collections
import numpy
import scipy
import skimage
import thop
import torch
from experiments import train_tpd_clean_v3
assert torch.cuda.is_available()
'

"$v3_python" - "$v3_repo" "$v3_source_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v3_screen800_source_lock_v1":
    raise SystemExit("invalid source-lock schema")
for relative, expected in payload["source_sha256"].items():
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked source: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"source digest mismatch: {relative} "
            f"expected={expected} actual={actual}"
        )
print(f"TPDCLEANV3_PREFLIGHT_SOURCES_OK files={len(payload['source_sha256'])}")
PY

for v3_index in "${!v3_variants[@]}"; do
    v3_variant="${v3_variants[$v3_index]}"
    v3_seed="${v3_seeds[$v3_index]}"
    v3_uuid="${v3_gpu_uuids[$v3_index]}"
    v3_tag="${v3_unit_tags[$v3_index]}"
    v3_actual_name="$(
        nvidia-smi -i "$v3_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    v3_free_memory="$(
        nvidia-smi -i "$v3_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$v3_actual_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV3_LAUNCH_ABORT reason=gpu_mismatch job=$v3_tag gpu_uuid=$v3_uuid name=$v3_actual_name" >&2
        exit 1
    fi
    if (( v3_free_memory < 7500 )); then
        echo "TPDCLEANV3_LAUNCH_ABORT reason=insufficient_memory job=$v3_tag gpu_uuid=$v3_uuid free_mib=$v3_free_memory" >&2
        exit 1
    fi

    v3_run_name="seed_${v3_seed}_${v3_run_tag}"
    v3_run_dir="$v3_result_root/NUDT-SIRST/$v3_variant/$v3_run_name"
    if [[ -e "$v3_run_dir" || -L "$v3_run_dir" ]]; then
        echo "TPDCLEANV3_LAUNCH_ABORT reason=run_path_not_fresh job=$v3_tag path=$v3_run_dir" >&2
        exit 1
    fi

    v3_unit="sctransnet-tpd-clean-v3-$v3_tag.service"
    if systemctl --user cat "$v3_unit" >/dev/null 2>&1; then
        echo "TPDCLEANV3_LAUNCH_ABORT reason=unit_already_exists unit=$v3_unit" >&2
        exit 1
    fi
done

echo "TPDCLEANV3_PREFLIGHT_OK jobs=full-s42,cap-s42,full-s3407,cap-s3407 gpus=4"
if [[ "$v3_mode" == "--preflight" ]]; then
    exit 0
fi

mkdir -p "$v3_result_root"
for v3_index in "${!v3_variants[@]}"; do
    v3_variant="${v3_variants[$v3_index]}"
    v3_seed="${v3_seeds[$v3_index]}"
    v3_uuid="${v3_gpu_uuids[$v3_index]}"
    v3_tag="${v3_unit_tags[$v3_index]}"
    v3_unit="sctransnet-tpd-clean-v3-$v3_tag"
    systemd-run --user \
        --unit="$v3_unit" \
        --description="SCTransNet TPD-Clean-v3 $v3_variant seed $v3_seed" \
        --property=Restart=no \
        --property=TimeoutStopSec=120 \
        /usr/bin/bash "$v3_worker" "$v3_variant" "$v3_seed" "$v3_uuid"
    echo "TPDCLEANV3_UNIT_STARTED variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_uuid unit=$v3_unit.service"
done
