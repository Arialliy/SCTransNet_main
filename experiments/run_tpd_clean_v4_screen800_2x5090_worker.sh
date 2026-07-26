#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 VARIANT SEED GPU_UUID" >&2
    exit 2
fi

v4_variant="$1"
v4_seed="$2"
v4_gpu_uuid="$3"
v4_repo="/home/ly/SCTransNet_main"
v4_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v4_result_root="$v4_repo/experiments/results/tpd_clean_v4_screen800_2x5090_v1"
v4_dataset_root="$v4_result_root/NUDT-SIRST"
v4_run_tag="screen800_pd_fp32_shared2x5090_v1"
v4_run_name="seed_${v4_seed}_${v4_run_tag}"
v4_run_dir="$v4_dataset_root/$v4_variant/$v4_run_name"
v4_log_root="$v4_result_root/logs"
v4_lock_root="$v4_result_root/.locks"
v4_launch_root="$v4_result_root/launch"
v4_source_lock="$v4_repo/experiments/tpd_clean_v4_screen800_2x_source_lock.json"
v4_old_v3_lock="$v4_repo/experiments/tpd_clean_v3_screen800_source_lock.json"
v4_old_clean_lock="$v4_repo/experiments/tpd_clean_screen800_source_lock.json"
v4_old_ner_lock="$v4_repo/experiments/tpd_ner_v1_source_lock.json"
v4_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
v4_cpu_threads=1
export OMP_NUM_THREADS="$v4_cpu_threads"
export MKL_NUM_THREADS="$v4_cpu_threads"
export OPENBLAS_NUM_THREADS="$v4_cpu_threads"
export NUMEXPR_NUM_THREADS="$v4_cpu_threads"

case "$v4_variant:$v4_seed:$v4_gpu_uuid" in
    tpd_clean_v4_full:42:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    tpd_clean_v4_sal_capacity:42:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    tpd_clean_v4_full:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    tpd_clean_v4_sal_capacity:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    *)
        echo "TPDCLEANV4_2X_ABORT reason=invalid_job_mapping variant=$v4_variant seed=$v4_seed gpu_uuid=$v4_gpu_uuid" >&2
        exit 2
        ;;
esac

mkdir -p "$v4_log_root" "$v4_lock_root" "$v4_launch_root"
v4_log="$v4_log_root/${v4_variant}_seed${v4_seed}.log"
exec > >(tee -a "$v4_log") 2>&1

exec 9>"$v4_lock_root/${v4_variant}_seed${v4_seed}.lock"
if ! flock -n 9; then
    echo "TPDCLEANV4_2X_ABORT reason=lock_held variant=$v4_variant seed=$v4_seed" >&2
    exit 1
fi

cd "$v4_repo"

v4_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

v4_verify_source_locks() {
    "$v4_python" - "$v4_repo" \
        "$v4_source_lock" \
        "$v4_old_v3_lock" \
        "$v4_old_clean_lock" \
        "$v4_old_ner_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
expected_schemas = {
    "sctransnet_tpd_clean_v4_screen800_2x_source_lock_v1",
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
    f"TPDCLEANV4_2X_SOURCES_OK locks={len(seen)} checked_entries={file_count}",
    flush=True,
)
PY
}

v4_verify_data() {
    local v4_actual
    v4_actual="$(
        timeout 300s "$v4_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    if [[ "$v4_actual" != "$v4_training_data_sha256" ]]; then
        echo "TPDCLEANV4_2X_ABORT reason=training_data_drift expected=$v4_training_data_sha256 actual=$v4_actual" >&2
        return 1
    fi
}

v4_verify_gpu() {
    local v4_name
    local v4_free
    v4_name="$(
        nvidia-smi -i "$v4_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    v4_free="$(
        nvidia-smi -i "$v4_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$v4_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV4_2X_ABORT reason=unexpected_gpu gpu_uuid=$v4_gpu_uuid name=$v4_name" >&2
        return 1
    fi
    if (( v4_free < 7500 )); then
        echo "TPDCLEANV4_2X_ABORT reason=insufficient_memory gpu_uuid=$v4_gpu_uuid free_mib=$v4_free" >&2
        return 1
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v4_gpu_uuid" \
        "$v4_python" - "$v4_gpu_uuid" <<'PY'
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
    "TPDCLEANV4_2X_GPU_OK"
    f" gpu_uuid={expected_uuid}"
    f" torch={torch.__version__}"
    f" cuda={torch.version.cuda}"
    f" capability={torch.cuda.get_device_capability(0)}",
    flush=True,
)
PY
}

v4_write_launch_manifest() {
    local v4_manifest="$v4_launch_root/${v4_variant}_seed${v4_seed}.json"
    local v4_manifest_tmp="$v4_manifest.tmp.$$"
    local v4_memory_used
    local v4_memory_free
    local v4_utilization
    local v4_load_average
    v4_memory_used="$(
        nvidia-smi -i "$v4_gpu_uuid" \
            --query-gpu=memory.used --format=csv,noheader,nounits
    )"
    v4_memory_free="$(
        nvidia-smi -i "$v4_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    v4_utilization="$(
        nvidia-smi -i "$v4_gpu_uuid" \
            --query-gpu=utilization.gpu --format=csv,noheader,nounits
    )"
    v4_load_average="$(awk '{print $1","$2","$3}' /proc/loadavg)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v4_gpu_uuid" \
        "$v4_python" - \
        "$v4_variant" \
        "$v4_seed" \
        "$v4_gpu_uuid" \
        "$v4_run_dir" \
        "$v4_training_data_sha256" \
        "$v4_source_lock" \
        "$v4_memory_used" \
        "$v4_memory_free" \
        "$v4_utilization" \
        "$v4_load_average" \
        "$v4_manifest_tmp" <<'PY'
import datetime
import hashlib
import json
import os
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
    "schema": "sctransnet_tpd_clean_v4_screen800_2x5090_launch_v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "variant": variant,
    "seed": int(seed),
    "candidate_family": "spd_anchored_tpd_clean_v4_single_logit_kcs",
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
        "allowed_gpu_indices": [2, 3],
        "concurrent_jobs_per_gpu": 2,
        "counterbalanced_mapping": True,
        "cpu_threads_per_job": 1,
        "thread_environment": {
            key: int(os.environ[key])
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    },
}
path = pathlib.Path(output_path)
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
    mv "$v4_manifest_tmp" "$v4_manifest"
}

v4_verify_source_locks
v4_source_lock_sha256="$(v4_sha256 "$v4_source_lock")"
v4_verify_data
v4_verify_gpu

if [[ -e "$v4_run_dir" || -L "$v4_run_dir" ]]; then
    echo "TPDCLEANV4_2X_ABORT reason=run_path_not_fresh variant=$v4_variant seed=$v4_seed path=$v4_run_dir" >&2
    exit 1
fi

v4_write_launch_manifest

echo "TPDCLEANV4_2X_START variant=$v4_variant seed=$v4_seed gpu_uuid=$v4_gpu_uuid cpu_threads=$v4_cpu_threads run_dir=$v4_run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$v4_gpu_uuid"
export PYTHONUNBUFFERED=1

"$v4_python" experiments/train_tpd_clean_v4.py \
    --variant "$v4_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$v4_repo/datasets" \
    --output-root "$v4_result_root" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed "$v4_seed" \
    --split-seed 20260722 \
    --val-fraction 0.20 \
    --eval-every 1 \
    --base-lr 0.001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --threshold 0.5 \
    --match-radius 3 \
    --tiny-area 9 \
    --run-tag "$v4_run_tag"

[[ "$(wc -l < "$v4_run_dir/metrics.jsonl")" -eq 800 ]]
jq -e --arg v4_variant "$v4_variant" --argjson v4_seed "$v4_seed" '
    .status == "complete" and
    .variant == $v4_variant and
    .dataset == "NUDT-SIRST" and
    .seed == $v4_seed and
    .selection_source == "internal_validation_only" and
    .official_test_accessed == false
' "$v4_run_dir/summary.json" >/dev/null

for v4_checkpoint in best.pth.tar best_miou.pth.tar; do
    "$v4_python" experiments/evaluate_tpd_clean_v4_pd_fa.py \
        --run-dir "$v4_run_dir" \
        --checkpoint "$v4_checkpoint" \
        --device cuda:0 \
        --expected-epochs 800
    v4_sweep="$v4_run_dir/pd_fa_sweep_${v4_checkpoint%.tar}.json"
    jq -e '
        .threshold_provenance.posthoc_endpoint_completion == false and
        .threshold_provenance.preregistered_endpoint_completion == true and
        .threshold_provenance.endpoint_protocol_stage == "before_formal_training" and
        .threshold_provenance.closed_probability_interval == true and
        .threshold_provenance.score_dtype == "float32" and
        .threshold_provenance.added_thresholds[-1] == 1 and
        .threshold_provenance.upper_boundary_threshold == 1 and
        .threshold_provenance.upper_boundary_comparison == "prediction > threshold" and
        .threshold_provenance.upper_boundary_semantics == "empty_prediction_pd0_fa0" and
        .points[-1].threshold == 1 and
        .points[-1].pd == 0 and
        .points[-1].fa == 0
    ' "$v4_sweep" >/dev/null
done

v4_verify_source_locks
v4_verify_data
if [[ "$(v4_sha256 "$v4_source_lock")" != "$v4_source_lock_sha256" ]]; then
    echo "TPDCLEANV4_2X_ABORT reason=source_lock_changed_during_run" >&2
    exit 1
fi
echo "TPDCLEANV4_2X_COMPLETE variant=$v4_variant seed=$v4_seed gpu_uuid=$v4_gpu_uuid epochs=800"
