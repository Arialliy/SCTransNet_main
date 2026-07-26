#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 VARIANT GPU_UUID" >&2
    exit 2
fi

clean_variant="$1"
clean_gpu_uuid="$2"
clean_repo="/home/ly/SCTransNet_main"
clean_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
clean_result_root="$clean_repo/experiments/results/tpd_clean_screen800_4x5090_v1"
clean_dataset_root="$clean_result_root/NUDT-SIRST"
clean_run_tag="screen800_pd_fp32_shared4x5090_v1"
clean_run_name="seed_42_$clean_run_tag"
clean_run_dir="$clean_dataset_root/$clean_variant/$clean_run_name"
clean_log_root="$clean_result_root/logs"
clean_lock_root="$clean_result_root/.locks"
clean_launch_root="$clean_result_root/launch"
clean_source_lock="$clean_repo/experiments/tpd_clean_screen800_source_lock.json"
clean_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"

case "$clean_variant:$clean_gpu_uuid" in
    grouped_keep:GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70) ;;
    tpd_clean_ctx:GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640) ;;
    tpd_clean_sal:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    tpd_clean_full:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    *)
        echo "TPDCLEAN_ABORT reason=invalid_variant_gpu_mapping variant=$clean_variant gpu_uuid=$clean_gpu_uuid" >&2
        exit 2
        ;;
esac

mkdir -p "$clean_log_root" "$clean_lock_root" "$clean_launch_root"
clean_log="$clean_log_root/$clean_variant.log"
exec > >(tee -a "$clean_log") 2>&1

exec 9>"$clean_lock_root/$clean_variant.lock"
if ! flock -n 9; then
    echo "TPDCLEAN_ABORT reason=lock_held variant=$clean_variant" >&2
    exit 1
fi

cd "$clean_repo"

clean_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

clean_verify_sources() {
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
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked source: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"source digest mismatch: {relative} expected={expected} actual={actual}"
        )
print(f"TPDCLEAN_SOURCES_OK files={len(payload['source_sha256'])}", flush=True)
PY
}

clean_verify_data() {
    local clean_actual
    clean_actual="$(
        timeout 300s "$clean_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    if [[ "$clean_actual" != "$clean_training_data_sha256" ]]; then
        echo "TPDCLEAN_ABORT reason=training_data_drift expected=$clean_training_data_sha256 actual=$clean_actual" >&2
        return 1
    fi
}

clean_verify_gpu() {
    local clean_name
    local clean_free
    clean_name="$(
        nvidia-smi -i "$clean_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    clean_free="$(
        nvidia-smi -i "$clean_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$clean_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEAN_ABORT reason=unexpected_gpu_name gpu_uuid=$clean_gpu_uuid name=$clean_name" >&2
        return 1
    fi
    if (( clean_free < 9000 )); then
        echo "TPDCLEAN_ABORT reason=insufficient_free_memory gpu_uuid=$clean_gpu_uuid free_mib=$clean_free" >&2
        return 1
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$clean_gpu_uuid" \
        "$clean_python" - "$clean_gpu_uuid" <<'PY'
import sys

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
    "TPDCLEAN_GPU_OK"
    f" gpu_uuid={expected_uuid}"
    f" torch={torch.__version__}"
    f" cuda={torch.version.cuda}"
    f" capability={torch.cuda.get_device_capability(0)}",
    flush=True,
)
PY
}

clean_write_launch_manifest() {
    local clean_manifest="$clean_launch_root/$clean_variant.json"
    local clean_manifest_tmp="$clean_manifest.tmp.$$"
    local clean_memory_used
    local clean_memory_free
    local clean_utilization
    local clean_load_average
    clean_memory_used="$(
        nvidia-smi -i "$clean_gpu_uuid" \
            --query-gpu=memory.used --format=csv,noheader,nounits
    )"
    clean_memory_free="$(
        nvidia-smi -i "$clean_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    clean_utilization="$(
        nvidia-smi -i "$clean_gpu_uuid" \
            --query-gpu=utilization.gpu --format=csv,noheader,nounits
    )"
    clean_load_average="$(awk '{print $1","$2","$3}' /proc/loadavg)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$clean_gpu_uuid" \
        "$clean_python" - \
        "$clean_variant" \
        "$clean_gpu_uuid" \
        "$clean_run_dir" \
        "$clean_training_data_sha256" \
        "$clean_source_lock" \
        "$clean_memory_used" \
        "$clean_memory_free" \
        "$clean_utilization" \
        "$clean_load_average" \
        "$clean_manifest_tmp" <<'PY'
import datetime
import hashlib
import json
import pathlib
import platform
import sys

import torch

(
    variant,
    gpu_uuid,
    run_dir,
    data_sha256,
    source_lock_path,
    memory_used,
    memory_free,
    utilization,
    load_average,
    output_path,
) = sys.argv[1:]
source_lock = pathlib.Path(source_lock_path)
payload = {
    "schema": "sctransnet_tpd_clean_screen800_launch_v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "variant": variant,
    "candidate_family": "spd_anchored_tpd_clean_v2",
    "gpu_uuid": gpu_uuid,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "run_directory": run_dir,
    "training_data_sha256": data_sha256,
    "source_lock": str(source_lock),
    "source_lock_sha256": hashlib.sha256(source_lock.read_bytes()).hexdigest(),
    "resource_snapshot": {
        "gpu_memory_used_mib": int(memory_used),
        "gpu_memory_free_mib": int(memory_free),
        "gpu_utilization_percent": int(utilization),
        "load_average": [float(value) for value in load_average.split(",")],
    },
    "policy": {
        "one_candidate_per_gpu": True,
        "fresh_run": True,
        "old_formal800_results_preserved": True,
        "shared_resource_screening": True,
        "efficiency_comparison_allowed": False,
        "official_test_accessed": False,
        "amp": False,
        "reason": (
            "the user requested immediate four-GPU execution without waiting; "
            "model metrics remain screening evidence and wall-clock efficiency "
            "is excluded because the accelerators and CPUs are shared"
        ),
    },
}
path = pathlib.Path(output_path)
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
    mv "$clean_manifest_tmp" "$clean_manifest"
}

clean_verify_sources
clean_source_lock_sha256="$(clean_sha256 "$clean_source_lock")"
clean_verify_data
clean_verify_gpu

if [[ -e "$clean_run_dir" || -L "$clean_run_dir" ]]; then
    echo "TPDCLEAN_ABORT reason=run_path_not_fresh variant=$clean_variant path=$clean_run_dir" >&2
    exit 1
fi

clean_write_launch_manifest

echo "TPDCLEAN_START variant=$clean_variant gpu_uuid=$clean_gpu_uuid run_dir=$clean_run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$clean_gpu_uuid"
export PYTHONUNBUFFERED=1

"$clean_python" experiments/train_tpd_clean_v2.py \
    --variant "$clean_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$clean_repo/datasets" \
    --output-root "$clean_result_root" \
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
    --run-tag "$clean_run_tag"

[[ "$(wc -l < "$clean_run_dir/metrics.jsonl")" -eq 800 ]]
jq -e --arg clean_variant "$clean_variant" '
    .status == "complete" and
    .variant == $clean_variant and
    .dataset == "NUDT-SIRST" and
    .seed == 42 and
    .selection_source == "internal_validation_only" and
    .official_test_accessed == false
' "$clean_run_dir/summary.json" >/dev/null

for clean_checkpoint in best.pth.tar best_miou.pth.tar; do
    "$clean_python" experiments/evaluate_tpd_clean_v2_pd_fa.py \
        --run-dir "$clean_run_dir" \
        --checkpoint "$clean_checkpoint" \
        --device cuda:0 \
        --expected-epochs 800
done

clean_verify_sources
clean_verify_data
if [[ "$(clean_sha256 "$clean_source_lock")" != "$clean_source_lock_sha256" ]]; then
    echo "TPDCLEAN_ABORT reason=source_lock_changed_during_run" >&2
    exit 1
fi
echo "TPDCLEAN_COMPLETE variant=$clean_variant gpu_uuid=$clean_gpu_uuid epochs=800"
