#!/usr/bin/env bash
set -euo pipefail

v4_repo="/home/ly/SCTransNet_main"
v4_worker="$v4_repo/experiments/run_tpd_clean_v4_screen800_2x5090_worker.sh"
v4_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v4_result_root="$v4_repo/experiments/results/tpd_clean_v4_screen800_2x5090_v1"
v4_run_tag="screen800_pd_fp32_shared2x5090_v1"
v4_source_lock="$v4_repo/experiments/tpd_clean_v4_screen800_2x_source_lock.json"
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
v4_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v4_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v4_expected_jobs=(
    "tpd_clean_v4_full:42:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562:full-s42"
    "tpd_clean_v4_sal_capacity:42:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3:cap-s42"
    "tpd_clean_v4_full:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3:full-s3407"
    "tpd_clean_v4_sal_capacity:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562:cap-s3407"
)

if (( ${#v4_variants[@]} != 4 ||
      ${#v4_seeds[@]} != 4 ||
      ${#v4_gpu_uuids[@]} != 4 ||
      ${#v4_unit_tags[@]} != 4 )); then
    echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=invalid_job_array_lengths" >&2
    exit 1
fi

declare -A v4_gpu_job_counts=()
for v4_index in "${!v4_expected_jobs[@]}"; do
    v4_actual_job="${v4_variants[$v4_index]}:${v4_seeds[$v4_index]}:${v4_gpu_uuids[$v4_index]}:${v4_unit_tags[$v4_index]}"
    if [[ "$v4_actual_job" != "${v4_expected_jobs[$v4_index]}" ]]; then
        echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=counterbalanced_mapping_mismatch index=$v4_index expected=${v4_expected_jobs[$v4_index]} actual=$v4_actual_job" >&2
        exit 1
    fi
    v4_uuid="${v4_gpu_uuids[$v4_index]}"
    v4_count="${v4_gpu_job_counts[$v4_uuid]:-0}"
    v4_gpu_job_counts["$v4_uuid"]="$((v4_count + 1))"
done
if (( ${#v4_gpu_job_counts[@]} != 2 )) ||
    [[ "${v4_gpu_job_counts[$v4_gpu2_uuid]:-0}" -ne 2 ]] ||
    [[ "${v4_gpu_job_counts[$v4_gpu3_uuid]:-0}" -ne 2 ]]; then
    echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=invalid_gpu_multiplicity expected_gpu_count=2 expected_jobs_per_gpu=2" >&2
    exit 1
fi

v4_mode="${1:-run}"
if [[ "$v4_mode" != "run" && "$v4_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$v4_repo"

[[ -x "$v4_worker" ]] || {
    echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=worker_not_executable path=$v4_worker" >&2
    exit 1
}
[[ -x "$v4_python" ]] || {
    echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=python_not_executable path=$v4_python" >&2
    exit 1
}
[[ -f "$v4_source_lock" && ! -L "$v4_source_lock" ]] || {
    echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=missing_source_lock path=$v4_source_lock" >&2
    exit 1
}

"$v4_python" -c '
import cv2
import einops
import ml_collections
import numpy
import scipy
import skimage
import thop
import torch
from experiments import evaluate_tpd_clean_v4_pd_fa
from experiments import train_tpd_clean_v4
from model.tpd_clean_v4 import SUPPORTED_CLEAN_V4_VARIANTS

assert SUPPORTED_CLEAN_V4_VARIANTS == (
    "tpd_clean_v4_full",
    "tpd_clean_v4_sal_capacity",
)
assert torch.cuda.is_available()
'

"$v4_python" - "$v4_repo" "$v4_source_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v4_screen800_2x_source_lock_v1":
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
print(f"TPDCLEANV4_2X_PREFLIGHT_SOURCES_OK files={len(payload['source_sha256'])}")
PY

for v4_uuid in "$v4_gpu2_uuid" "$v4_gpu3_uuid"; do
    v4_actual_index="$(
        nvidia-smi -i "$v4_uuid" \
            --query-gpu=index --format=csv,noheader,nounits
    )"
    v4_actual_name="$(
        nvidia-smi -i "$v4_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    v4_free_memory="$(
        nvidia-smi -i "$v4_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$v4_uuid" == "$v4_gpu2_uuid" && "$v4_actual_index" != "2" ]] ||
        [[ "$v4_uuid" == "$v4_gpu3_uuid" && "$v4_actual_index" != "3" ]]; then
        echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=gpu_index_mismatch gpu_uuid=$v4_uuid index=$v4_actual_index" >&2
        exit 1
    fi
    if [[ "$v4_actual_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=gpu_mismatch gpu_uuid=$v4_uuid name=$v4_actual_name" >&2
        exit 1
    fi
    if (( v4_free_memory < 15000 )); then
        echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=insufficient_memory_for_two_jobs gpu_uuid=$v4_uuid free_mib=$v4_free_memory required_mib=15000" >&2
        exit 1
    fi
done

for v4_index in "${!v4_variants[@]}"; do
    v4_variant="${v4_variants[$v4_index]}"
    v4_seed="${v4_seeds[$v4_index]}"
    v4_uuid="${v4_gpu_uuids[$v4_index]}"
    v4_tag="${v4_unit_tags[$v4_index]}"

    v4_run_name="seed_${v4_seed}_${v4_run_tag}"
    v4_run_dir="$v4_result_root/NUDT-SIRST/$v4_variant/$v4_run_name"
    if [[ -e "$v4_run_dir" || -L "$v4_run_dir" ]]; then
        echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=run_path_not_fresh job=$v4_tag path=$v4_run_dir" >&2
        exit 1
    fi

    v4_unit="sctransnet-tpd-clean-v4-2x-$v4_tag.service"
    if systemctl --user cat "$v4_unit" >/dev/null 2>&1; then
        echo "TPDCLEANV4_2X_LAUNCH_ABORT reason=unit_already_exists unit=$v4_unit" >&2
        exit 1
    fi
done

echo "TPDCLEANV4_2X_PREFLIGHT_OK jobs=full-s42,cap-s42,full-s3407,cap-s3407 gpus=2 concurrent_jobs_per_gpu=2 threads_per_job=1 counterbalanced_mapping=true"
if [[ "$v4_mode" == "--preflight" ]]; then
    exit 0
fi

mkdir -p "$v4_result_root"
for v4_index in "${!v4_variants[@]}"; do
    v4_variant="${v4_variants[$v4_index]}"
    v4_seed="${v4_seeds[$v4_index]}"
    v4_uuid="${v4_gpu_uuids[$v4_index]}"
    v4_tag="${v4_unit_tags[$v4_index]}"
    v4_unit="sctransnet-tpd-clean-v4-2x-$v4_tag"
    systemd-run --user \
        --unit="$v4_unit" \
        --description="SCTransNet TPD-Clean-v4 2GPU $v4_variant seed $v4_seed" \
        --property=Restart=no \
        --property=TimeoutStopSec=120 \
        /usr/bin/bash "$v4_worker" "$v4_variant" "$v4_seed" "$v4_uuid"
    echo "TPDCLEANV4_2X_UNIT_STARTED variant=$v4_variant seed=$v4_seed gpu_uuid=$v4_uuid unit=$v4_unit.service"
done
