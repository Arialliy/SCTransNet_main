#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_worker="$formal_repo/experiments/run_tpd_formal800_4x5090_worker.sh"
formal_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
formal_result_root="$formal_repo/experiments/results/tpd_pe_formal800_4x5090_v1"
formal_run_name="seed_42_formal800_pd_fp32_4x5090_v1"
formal_variants=(original progressive tpd spd)
formal_gpu_uuids=(
    GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70
    GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
)

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$formal_repo"

[[ -x "$formal_worker" ]] || {
    echo "FORMAL4X5090_LAUNCH_ABORT reason=worker_not_executable path=$formal_worker" >&2
    exit 1
}
[[ -x "$formal_python" ]] || {
    echo "FORMAL4X5090_LAUNCH_ABORT reason=python_not_executable path=$formal_python" >&2
    exit 1
}

"$formal_python" -c '
import cv2
import einops
import ml_collections
import numpy
import scipy
import skimage
import thop
import torch
from experiments import train_tpd_pilot
assert torch.cuda.is_available()
'

for formal_index in "${!formal_variants[@]}"; do
    formal_variant="${formal_variants[$formal_index]}"
    formal_uuid="${formal_gpu_uuids[$formal_index]}"
    formal_actual_name="$(
        nvidia-smi -i "$formal_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    if [[ "$formal_actual_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "FORMAL4X5090_LAUNCH_ABORT reason=gpu_mismatch variant=$formal_variant gpu_uuid=$formal_uuid name=$formal_actual_name" >&2
        exit 1
    fi

    formal_run_dir="$formal_result_root/NUDT-SIRST/$formal_variant/$formal_run_name"
    if [[ -e "$formal_run_dir" || -L "$formal_run_dir" ]]; then
        echo "FORMAL4X5090_LAUNCH_ABORT reason=run_path_not_fresh variant=$formal_variant path=$formal_run_dir" >&2
        exit 1
    fi

    formal_unit="sctransnet-formal800-4x5090-$formal_variant.service"
    if systemctl --user cat "$formal_unit" >/dev/null 2>&1; then
        echo "FORMAL4X5090_LAUNCH_ABORT reason=unit_already_exists unit=$formal_unit" >&2
        exit 1
    fi
done

echo "FORMAL4X5090_PREFLIGHT_OK variants=original,progressive,tpd,spd gpus=4"
if [[ "$formal_mode" == "--preflight" ]]; then
    exit 0
fi

mkdir -p "$formal_result_root"
for formal_index in "${!formal_variants[@]}"; do
    formal_variant="${formal_variants[$formal_index]}"
    formal_uuid="${formal_gpu_uuids[$formal_index]}"
    formal_unit="sctransnet-formal800-4x5090-$formal_variant"
    systemd-run --user \
        --unit="$formal_unit" \
        --description="SCTransNet formal800 $formal_variant on RTX 5090" \
        --property=Restart=no \
        --property=TimeoutStopSec=120 \
        /usr/bin/bash "$formal_worker" "$formal_variant" "$formal_uuid"
    echo "FORMAL4X5090_UNIT_STARTED variant=$formal_variant gpu_uuid=$formal_uuid unit=$formal_unit.service"
done
