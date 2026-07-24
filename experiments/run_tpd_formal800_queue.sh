#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 VARIANT [VARIANT ...]" >&2
    exit 2
fi

python_bin="/home/ly/MSHNet/.venv/bin/python"
output_root="experiments/results/tpd_pe_formal800_v1"

for variant in "$@"; do
    echo "QUEUE_START variant=${variant} physical_cuda=${CUDA_VISIBLE_DEVICES:-unset}"
    "$python_bin" experiments/train_tpd_pilot.py \
        --variant "$variant" \
        --dataset NUDT-SIRST \
        --output-root "$output_root" \
        --device cuda:0 \
        --epochs 800 \
        --batch-size 16 \
        --patch-size 256 \
        --workers 0 \
        --seed 42 \
        --split-seed 20260722 \
        --val-fraction 0.20 \
        --eval-every 1 \
        --base-lr 0.001 \
        --min-lr 0.00001 \
        --warmup-epochs 10 \
        --threshold 0.5 \
        --match-radius 3 \
        --tiny-area 9 \
        --run-tag formal800_pd_fp32_v1
    echo "QUEUE_COMPLETE variant=${variant}"
done
