#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 VARIANT GPU_UUID" >&2
    exit 2
fi

formal_variant="$1"
formal_gpu_uuid="$2"
formal_repo="/home/ly/SCTransNet_main"
formal_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
formal_result_root="$formal_repo/experiments/results/tpd_pe_formal800_4x5090_v1"
formal_dataset_root="$formal_result_root/NUDT-SIRST"
formal_run_name="seed_42_formal800_pd_fp32_4x5090_v1"
formal_run_dir="$formal_dataset_root/$formal_variant/$formal_run_name"
formal_log_root="$formal_result_root/logs"
formal_lock_root="$formal_result_root/.locks"
formal_launch_root="$formal_result_root/launch"
formal_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"

case "$formal_variant:$formal_gpu_uuid" in
    original:GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70) ;;
    progressive:GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640) ;;
    tpd:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    spd:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    *)
        echo "FORMAL4X5090_ABORT reason=invalid_variant_gpu_mapping variant=$formal_variant gpu_uuid=$formal_gpu_uuid" >&2
        exit 2
        ;;
esac

mkdir -p "$formal_log_root" "$formal_lock_root" "$formal_launch_root"
formal_log="$formal_log_root/$formal_variant.log"
exec > >(tee -a "$formal_log") 2>&1

exec 9>"$formal_lock_root/$formal_variant.lock"
if ! flock -n 9; then
    echo "FORMAL4X5090_ABORT reason=lock_held variant=$formal_variant" >&2
    exit 1
fi

cd "$formal_repo"

formal_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

formal_require_sha256() {
    local formal_path="$1"
    local formal_expected="$2"
    local formal_actual
    [[ -f "$formal_path" && ! -L "$formal_path" ]] || {
        echo "FORMAL4X5090_ABORT reason=missing_or_symlink path=$formal_path" >&2
        return 1
    }
    formal_actual="$(formal_sha256 "$formal_path")"
    if [[ "$formal_actual" != "$formal_expected" ]]; then
        echo "FORMAL4X5090_ABORT reason=sha_mismatch path=$formal_path expected=$formal_expected actual=$formal_actual" >&2
        return 1
    fi
}

formal_verify_sources() {
    formal_require_sha256 experiments/train_tpd_pilot.py 7532bdc3bcc777aa164e258ab21f78d38ed3a1eaa677a29c8256d900224a7f26
    formal_require_sha256 experiments/fingerprint_tpd_training_data.py 26382e38e899bdf4f97b77c6671929c391decef6e4bf4ac40094a7d4e6b0bc7d
    formal_require_sha256 dataset.py 516ea9c410f80cc9ae912cf0443126a067dd14b6cc5ad7945e83cfc497f4678d
    formal_require_sha256 utils.py afb6fc221072ddd082b53ccda132232bc9089afd0458d8f0e47a39b9c1e25c13
    formal_require_sha256 model/SCTransNet.py 5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb
    formal_require_sha256 model/tpd.py 18a5892edd18ab040e38f18c8d86a02bf3e50b7a4d12d0115ec9a97e8051c135
    formal_require_sha256 model/Config.py b7e3e67c379ef4638605ebe612336b0c3cdb1a97f4d6fe731dec80b4847d5596
}

formal_verify_data() {
    local formal_actual
    formal_actual="$(
        timeout 300s "$formal_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    if [[ "$formal_actual" != "$formal_training_data_sha256" ]]; then
        echo "FORMAL4X5090_ABORT reason=training_data_drift expected=$formal_training_data_sha256 actual=$formal_actual" >&2
        return 1
    fi
}

formal_verify_gpu() {
    local formal_name
    formal_name="$(
        nvidia-smi -i "$formal_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    if [[ "$formal_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "FORMAL4X5090_ABORT reason=unexpected_gpu_name gpu_uuid=$formal_gpu_uuid name=$formal_name" >&2
        return 1
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$formal_gpu_uuid" \
        "$formal_python" - "$formal_gpu_uuid" <<'PY'
import sys

import cv2
import einops
import ml_collections
import numpy
import scipy
import skimage
import thop
import torch

expected_uuid = sys.argv[1]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected CUDA device: {torch.cuda.get_device_name(0)}")
x = torch.ones((64, 64), device="cuda:0")
if float((x @ x).sum().item()) != 262144.0:
    raise SystemExit("CUDA health check returned an unexpected value")
torch.cuda.synchronize()
print(
    "FORMAL4X5090_GPU_OK"
    f" gpu_uuid={expected_uuid}"
    f" torch={torch.__version__}"
    f" cuda={torch.version.cuda}"
    f" capability={torch.cuda.get_device_capability(0)}",
    flush=True,
)
PY
}

formal_write_launch_manifest() {
    local formal_manifest="$formal_launch_root/$formal_variant.json"
    local formal_manifest_tmp="$formal_manifest.tmp.$$"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$formal_gpu_uuid" \
        "$formal_python" - \
        "$formal_variant" \
        "$formal_gpu_uuid" \
        "$formal_run_dir" \
        "$formal_training_data_sha256" \
        "$formal_manifest_tmp" <<'PY'
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import sys

import torch

variant, gpu_uuid, run_dir, data_sha256, output_path = sys.argv[1:]
repo = pathlib.Path("/home/ly/SCTransNet_main")
source_paths = (
    "experiments/train_tpd_pilot.py",
    "experiments/fingerprint_tpd_training_data.py",
    "dataset.py",
    "utils.py",
    "model/SCTransNet.py",
    "model/tpd.py",
    "model/Config.py",
)
source_sha256 = {
    path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
    for path in source_paths
}
packages = {}
for package in (
    "numpy",
    "scipy",
    "scikit-image",
    "opencv-python",
    "ml-collections",
    "einops",
    "thop",
    "Pillow",
):
    try:
        packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        packages[package] = None

payload = {
    "schema": "sctransnet_formal800_4x5090_launch_v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "variant": variant,
    "gpu_uuid": gpu_uuid,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "packages": packages,
    "run_directory": run_dir,
    "training_data_sha256": data_sha256,
    "source_sha256": source_sha256,
    "policy": {
        "one_variant_per_gpu": True,
        "fresh_run": True,
        "old_formal800_results_preserved": True,
        "reason": (
            "uniform four-RTX-5090 rerun; the old Original checkpoint lacks "
            "Python/NumPy/Torch/CUDA/DataLoader RNG state and cannot provide "
            "bit-exact continuation"
        ),
        "official_test_accessed": False,
        "amp": False,
    },
}
path = pathlib.Path(output_path)
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
    mv "$formal_manifest_tmp" "$formal_manifest"
}

formal_verify_sources
formal_verify_data
formal_verify_gpu

if [[ -e "$formal_run_dir" || -L "$formal_run_dir" ]]; then
    echo "FORMAL4X5090_ABORT reason=run_path_not_fresh variant=$formal_variant path=$formal_run_dir" >&2
    exit 1
fi

formal_write_launch_manifest

echo "FORMAL4X5090_START variant=$formal_variant gpu_uuid=$formal_gpu_uuid run_dir=$formal_run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$formal_gpu_uuid"
export PYTHONUNBUFFERED=1

"$formal_python" experiments/train_tpd_pilot.py \
    --variant "$formal_variant" \
    --dataset NUDT-SIRST \
    --output-root "$formal_result_root" \
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
    --run-tag formal800_pd_fp32_4x5090_v1

[[ "$(wc -l < "$formal_run_dir/metrics.jsonl")" -eq 800 ]]
jq -e --arg formal_variant "$formal_variant" '
    .status == "complete" and
    .variant == $formal_variant and
    .dataset == "NUDT-SIRST" and
    .seed == 42 and
    .selection_source == "internal_validation_only" and
    .official_test_accessed == false
' "$formal_run_dir/summary.json" >/dev/null
formal_verify_sources
formal_verify_data
echo "FORMAL4X5090_COMPLETE variant=$formal_variant gpu_uuid=$formal_gpu_uuid epochs=800"
