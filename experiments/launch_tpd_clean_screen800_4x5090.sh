#!/usr/bin/env bash
set -euo pipefail

clean_repo="/home/ly/SCTransNet_main"
clean_worker="$clean_repo/experiments/run_tpd_clean_screen800_4x5090_worker.sh"
clean_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
clean_result_root="$clean_repo/experiments/results/tpd_clean_screen800_4x5090_v1"
clean_run_name="seed_42_screen800_pd_fp32_shared4x5090_v1"
clean_source_lock="$clean_repo/experiments/tpd_clean_screen800_source_lock.json"
clean_variants=(grouped_keep tpd_clean_ctx tpd_clean_sal tpd_clean_full)
clean_gpu_uuids=(
    GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70
    GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
)

clean_mode="${1:-run}"
if [[ "$clean_mode" != "run" && "$clean_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$clean_repo"

[[ -x "$clean_worker" ]] || {
    echo "TPDCLEAN_LAUNCH_ABORT reason=worker_not_executable path=$clean_worker" >&2
    exit 1
}
[[ -x "$clean_python" ]] || {
    echo "TPDCLEAN_LAUNCH_ABORT reason=python_not_executable path=$clean_python" >&2
    exit 1
}
[[ -f "$clean_source_lock" && ! -L "$clean_source_lock" ]] || {
    echo "TPDCLEAN_LAUNCH_ABORT reason=missing_source_lock path=$clean_source_lock" >&2
    exit 1
}

"$clean_python" -c '
import cv2
import einops
import ml_collections
import numpy
import scipy
import skimage
import thop
import torch
from experiments import train_tpd_clean_v2
assert torch.cuda.is_available()
'

"$clean_python" - "$clean_repo" "$clean_source_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_screen800_source_lock_v1":
    raise SystemExit("invalid source-lock schema")
for relative, expected in payload["source_sha256"].items():
    path = repo / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"source digest mismatch: {relative} expected={expected} actual={actual}"
        )
print(f"TPDCLEAN_PREFLIGHT_SOURCES_OK files={len(payload['source_sha256'])}")
PY

for clean_index in "${!clean_variants[@]}"; do
    clean_variant="${clean_variants[$clean_index]}"
    clean_uuid="${clean_gpu_uuids[$clean_index]}"
    clean_actual_name="$(
        nvidia-smi -i "$clean_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    clean_free_memory="$(
        nvidia-smi -i "$clean_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$clean_actual_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEAN_LAUNCH_ABORT reason=gpu_mismatch variant=$clean_variant gpu_uuid=$clean_uuid name=$clean_actual_name" >&2
        exit 1
    fi
    if (( clean_free_memory < 9000 )); then
        echo "TPDCLEAN_LAUNCH_ABORT reason=insufficient_free_memory variant=$clean_variant gpu_uuid=$clean_uuid free_mib=$clean_free_memory" >&2
        exit 1
    fi

    clean_run_dir="$clean_result_root/NUDT-SIRST/$clean_variant/$clean_run_name"
    if [[ -e "$clean_run_dir" || -L "$clean_run_dir" ]]; then
        echo "TPDCLEAN_LAUNCH_ABORT reason=run_path_not_fresh variant=$clean_variant path=$clean_run_dir" >&2
        exit 1
    fi

    clean_unit="sctransnet-tpd-clean-screen800-$clean_variant.service"
    if systemctl --user cat "$clean_unit" >/dev/null 2>&1; then
        echo "TPDCLEAN_LAUNCH_ABORT reason=unit_already_exists unit=$clean_unit" >&2
        exit 1
    fi
done

echo "TPDCLEAN_PREFLIGHT_OK variants=grouped_keep,tpd_clean_ctx,tpd_clean_sal,tpd_clean_full gpus=4 mode=shared_resource_screening"
if [[ "$clean_mode" == "--preflight" ]]; then
    exit 0
fi

mkdir -p "$clean_result_root"
for clean_index in "${!clean_variants[@]}"; do
    clean_variant="${clean_variants[$clean_index]}"
    clean_uuid="${clean_gpu_uuids[$clean_index]}"
    clean_unit="sctransnet-tpd-clean-screen800-$clean_variant"
    systemd-run --user \
        --unit="$clean_unit" \
        --description="SCTransNet TPD-Clean screen800 $clean_variant on RTX 5090" \
        --property=Restart=no \
        --property=TimeoutStopSec=120 \
        /usr/bin/bash "$clean_worker" "$clean_variant" "$clean_uuid"
    echo "TPDCLEAN_UNIT_STARTED variant=$clean_variant gpu_uuid=$clean_uuid unit=$clean_unit.service"
done
