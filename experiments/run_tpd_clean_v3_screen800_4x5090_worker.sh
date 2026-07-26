#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 VARIANT SEED GPU_UUID" >&2
    exit 2
fi

v3_variant="$1"
v3_seed="$2"
v3_gpu_uuid="$3"
v3_repo="/home/ly/SCTransNet_main"
v3_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v3_result_root="$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1"
v3_dataset_root="$v3_result_root/NUDT-SIRST"
v3_run_tag="screen800_pd_fp32_shared4x5090_v1"
v3_run_name="seed_${v3_seed}_${v3_run_tag}"
v3_run_dir="$v3_dataset_root/$v3_variant/$v3_run_name"
v3_log_root="$v3_result_root/logs"
v3_lock_root="$v3_result_root/.locks"
v3_launch_root="$v3_result_root/launch"
v3_source_lock="$v3_repo/experiments/tpd_clean_v3_screen800_source_lock.json"
v3_old_clean_lock="$v3_repo/experiments/tpd_clean_screen800_source_lock.json"
v3_old_ner_lock="$v3_repo/experiments/tpd_ner_v1_source_lock.json"
v3_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"

case "$v3_variant:$v3_seed:$v3_gpu_uuid" in
    tpd_clean_v3_full:42:GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70) ;;
    tpd_clean_v3_sal_capacity:42:GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640) ;;
    tpd_clean_v3_full:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    tpd_clean_v3_sal_capacity:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    *)
        echo "TPDCLEANV3_ABORT reason=invalid_job_mapping variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_gpu_uuid" >&2
        exit 2
        ;;
esac

mkdir -p "$v3_log_root" "$v3_lock_root" "$v3_launch_root"
v3_log="$v3_log_root/${v3_variant}_seed${v3_seed}.log"
exec > >(tee -a "$v3_log") 2>&1

exec 9>"$v3_lock_root/${v3_variant}_seed${v3_seed}.lock"
if ! flock -n 9; then
    echo "TPDCLEANV3_ABORT reason=lock_held variant=$v3_variant seed=$v3_seed" >&2
    exit 1
fi

cd "$v3_repo"

v3_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

v3_verify_source_locks() {
    "$v3_python" - "$v3_repo" \
        "$v3_source_lock" \
        "$v3_old_clean_lock" \
        "$v3_old_ner_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
expected_schemas = {
    "sctransnet_tpd_clean_v3_screen800_source_lock_v1",
    "sctransnet_tpd_clean_screen800_source_lock_v1",
    "sctransnet_tpd_ner_v1_source_lock_v1",
}
seen = set()
file_count = 0
for lock_text in sys.argv[2:]:
    lock_path = pathlib.Path(lock_text)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise SystemExit(f"missing or linked source lock: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in expected_schemas:
        raise SystemExit(f"unexpected source-lock schema: {schema}")
    seen.add(schema)
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
        file_count += 1
if seen != expected_schemas:
    raise SystemExit(f"incomplete source-lock schemas: {sorted(seen)}")
print(
    f"TPDCLEANV3_SOURCES_OK locks={len(seen)} checked_entries={file_count}",
    flush=True,
)
PY
}

v3_verify_data() {
    local v3_actual
    v3_actual="$(
        timeout 300s "$v3_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    if [[ "$v3_actual" != "$v3_training_data_sha256" ]]; then
        echo "TPDCLEANV3_ABORT reason=training_data_drift expected=$v3_training_data_sha256 actual=$v3_actual" >&2
        return 1
    fi
}

v3_verify_gpu() {
    local v3_name
    local v3_free
    v3_name="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    v3_free="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$v3_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV3_ABORT reason=unexpected_gpu gpu_uuid=$v3_gpu_uuid name=$v3_name" >&2
        return 1
    fi
    if (( v3_free < 7500 )); then
        echo "TPDCLEANV3_ABORT reason=insufficient_memory gpu_uuid=$v3_gpu_uuid free_mib=$v3_free" >&2
        return 1
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v3_gpu_uuid" \
        "$v3_python" - "$v3_gpu_uuid" <<'PY'
import sys
import torch

expected_uuid = sys.argv[1]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected CUDA device: {torch.cuda.get_device_name(0)}")
x = torch.ones((64, 64), device="cuda:0")
if float((x @ x).sum().item()) != 262144.0:
    raise SystemExit("CUDA calculation returned an unexpected value")
torch.cuda.synchronize()
print(
    "TPDCLEANV3_GPU_OK"
    f" gpu_uuid={expected_uuid}"
    f" torch={torch.__version__}"
    f" cuda={torch.version.cuda}"
    f" capability={torch.cuda.get_device_capability(0)}",
    flush=True,
)
PY
}

v3_write_launch_manifest() {
    local v3_manifest="$v3_launch_root/${v3_variant}_seed${v3_seed}.json"
    local v3_manifest_tmp="$v3_manifest.tmp.$$"
    local v3_memory_used
    local v3_memory_free
    local v3_utilization
    local v3_load_average
    v3_memory_used="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=memory.used --format=csv,noheader,nounits
    )"
    v3_memory_free="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    v3_utilization="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=utilization.gpu --format=csv,noheader,nounits
    )"
    v3_load_average="$(awk '{print $1","$2","$3}' /proc/loadavg)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v3_gpu_uuid" \
        "$v3_python" - \
        "$v3_variant" \
        "$v3_seed" \
        "$v3_gpu_uuid" \
        "$v3_run_dir" \
        "$v3_training_data_sha256" \
        "$v3_source_lock" \
        "$v3_memory_used" \
        "$v3_memory_free" \
        "$v3_utilization" \
        "$v3_load_average" \
        "$v3_manifest_tmp" <<'PY'
import datetime
import hashlib
import json
import pathlib
import platform
import sys

import torch

(
    variant,
    seed,
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
    "schema": "sctransnet_tpd_clean_v3_screen800_launch_v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "variant": variant,
    "seed": int(seed),
    "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
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
        "paired_variants": True,
        "pre_registered_seeds": [42, 3407],
        "fresh_run": True,
        "old_results_preserved": True,
        "shared_resource_screening": True,
        "efficiency_comparison_allowed": False,
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
    mv "$v3_manifest_tmp" "$v3_manifest"
}

v3_verify_source_locks
v3_source_lock_sha256="$(v3_sha256 "$v3_source_lock")"
v3_verify_data
v3_verify_gpu

if [[ -e "$v3_run_dir" || -L "$v3_run_dir" ]]; then
    echo "TPDCLEANV3_ABORT reason=run_path_not_fresh variant=$v3_variant seed=$v3_seed path=$v3_run_dir" >&2
    exit 1
fi

v3_write_launch_manifest

echo "TPDCLEANV3_START variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_gpu_uuid run_dir=$v3_run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$v3_gpu_uuid"
export PYTHONUNBUFFERED=1

"$v3_python" experiments/train_tpd_clean_v3.py \
    --variant "$v3_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$v3_repo/datasets" \
    --output-root "$v3_result_root" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed "$v3_seed" \
    --split-seed 20260722 \
    --val-fraction 0.20 \
    --eval-every 1 \
    --base-lr 0.001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --threshold 0.5 \
    --match-radius 3 \
    --tiny-area 9 \
    --run-tag "$v3_run_tag"

[[ "$(wc -l < "$v3_run_dir/metrics.jsonl")" -eq 800 ]]
jq -e --arg v3_variant "$v3_variant" --argjson v3_seed "$v3_seed" '
    .status == "complete" and
    .variant == $v3_variant and
    .dataset == "NUDT-SIRST" and
    .seed == $v3_seed and
    .selection_source == "internal_validation_only" and
    .official_test_accessed == false
' "$v3_run_dir/summary.json" >/dev/null

for v3_checkpoint in best.pth.tar best_miou.pth.tar; do
    "$v3_python" experiments/evaluate_tpd_clean_v3_pd_fa.py \
        --run-dir "$v3_run_dir" \
        --checkpoint "$v3_checkpoint" \
        --device cuda:0 \
        --expected-epochs 800
done

v3_verify_source_locks
v3_verify_data
if [[ "$(v3_sha256 "$v3_source_lock")" != "$v3_source_lock_sha256" ]]; then
    echo "TPDCLEANV3_ABORT reason=source_lock_changed_during_run" >&2
    exit 1
fi
echo "TPDCLEANV3_COMPLETE variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_gpu_uuid epochs=800"
