#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 VARIANT SEED GPU_UUID" >&2
    exit 2
fi

v5_variant="$1"
v5_seed="$2"
v5_gpu_uuid="$3"
v5_repo="/home/ly/SCTransNet_main"
v5_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v5_result_root="$v5_repo/experiments/results/tpd_clean_v5_screen800_2x5090_v1"
v5_dataset_root="$v5_result_root/NUDT-SIRST"
v5_run_tag="screen800_pd_fp32_shared2x5090_v1"
v5_run_name="seed_${v5_seed}_${v5_run_tag}"
v5_run_dir="$v5_dataset_root/$v5_variant/$v5_run_name"
v5_log_root="$v5_result_root/logs"
v5_lock_root="$v5_result_root/.locks"
v5_launch_root="$v5_result_root/launch"
v5_source_lock="$v5_repo/experiments/tpd_clean_v5_screen800_2x_source_lock.json"
v5_frozen_v4_lock="$v5_repo/experiments/tpd_clean_v4_screen800_2x_source_lock.json"
v5_frozen_v3_lock="$v5_repo/experiments/tpd_clean_v3_screen800_source_lock.json"
v5_frozen_v2_lock="$v5_repo/experiments/tpd_clean_screen800_source_lock.json"
v5_frozen_ner_lock="$v5_repo/experiments/tpd_ner_v1_source_lock.json"
v5_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
v5_cpu_threads=1

export OMP_NUM_THREADS="$v5_cpu_threads"
export MKL_NUM_THREADS="$v5_cpu_threads"
export OPENBLAS_NUM_THREADS="$v5_cpu_threads"
export NUMEXPR_NUM_THREADS="$v5_cpu_threads"

case "$v5_variant:$v5_seed:$v5_gpu_uuid" in
    tpd_clean_v5_full:42:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    tpd_clean_v5_sal_capacity:42:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    tpd_clean_v5_full:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    tpd_clean_v5_sal_capacity:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    *)
        echo "TPDCLEANV5_2X_ABORT reason=invalid_job_mapping variant=$v5_variant seed=$v5_seed gpu_uuid=$v5_gpu_uuid" >&2
        exit 2
        ;;
esac

mkdir -p "$v5_log_root" "$v5_lock_root" "$v5_launch_root"
v5_log="$v5_log_root/${v5_variant}_seed${v5_seed}.log"
exec > >(tee -a "$v5_log") 2>&1

exec 9>"$v5_lock_root/${v5_variant}_seed${v5_seed}.lock"
if ! flock -n 9; then
    echo "TPDCLEANV5_2X_ABORT reason=lock_held variant=$v5_variant seed=$v5_seed" >&2
    exit 1
fi

cd "$v5_repo"

v5_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

v5_verify_sources() {
    "$v5_python" - \
        "$v5_repo" \
        "$v5_source_lock" \
        "$v5_frozen_v4_lock" \
        "$v5_frozen_v3_lock" \
        "$v5_frozen_v2_lock" \
        "$v5_frozen_ner_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
expected_schemas = {
    "sctransnet_tpd_clean_v5_screen800_2x_source_lock_v1",
    "sctransnet_tpd_clean_v4_screen800_2x_source_lock_v1",
    "sctransnet_tpd_clean_v3_screen800_source_lock_v1",
    "sctransnet_tpd_clean_screen800_source_lock_v1",
    "sctransnet_tpd_ner_v1_source_lock_v1",
}
required_v5_sources = {
    "model/tpd_clean_v5.py",
    "experiments/train_tpd_clean_v5.py",
    "experiments/evaluate_tpd_clean_v5_pd_fa.py",
    "experiments/smoke_tpd_clean_v5.py",
    "experiments/capture_tpd_clean_v5_smoke_report.py",
    "experiments/run_tpd_clean_v5_screen800_2x5090_worker.sh",
    "experiments/launch_tpd_clean_v5_screen800_2x5090.sh",
    "experiments/status_tpd_clean_v5_screen800_2x5090.sh",
    "experiments/TPD_CLEAN_V5_PROTOCOL.md",
    "experiments/TPD_CLEAN_V5_2GPU_PROTOCOL.md",
    "tests/test_tpd_clean_v5.py",
    "tests/test_train_tpd_clean_v5.py",
    "tests/test_evaluate_tpd_clean_v5_pd_fa.py",
    "tests/test_smoke_tpd_clean_v5.py",
    "tests/test_tpd_clean_v5_runner.py",
    "tests/test_tpd_clean_v5_2x_runtime.py",
    "experiments/train_tpd_pilot.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/fingerprint_tpd_training_data.py",
    "dataset.py",
    "utils.py",
    "warmup_scheduler.py",
    "model/SCTransNet.py",
    "model/Config.py",
    "model/tpd.py",
    "experiments/smoke_tpd_clean_v3.py",
    "experiments/train_tpd_clean_v3.py",
    "model/tpd_clean_v3.py",
    "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
    "experiments/tpd_clean_v3_screen800_source_lock.json",
    "experiments/tpd_clean_screen800_source_lock.json",
    "experiments/tpd_ner_v1_source_lock.json",
}
seen = set()
checked_entries = 0
for lock_text in sys.argv[2:]:
    lock_path = pathlib.Path(lock_text)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise SystemExit(f"missing or linked source lock: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in expected_schemas:
        raise SystemExit(f"unexpected source-lock schema: {schema!r}")
    if schema in seen:
        raise SystemExit(f"duplicate source-lock schema: {schema!r}")
    seen.add(schema)
    if schema == "sctransnet_tpd_clean_v5_screen800_2x_source_lock_v1":
        if payload.get("variants") != [
            "tpd_clean_v5_full",
            "tpd_clean_v5_sal_capacity",
        ]:
            raise SystemExit("source-lock variant matrix differs")
        if set(payload.get("source_sha256", {})) != required_v5_sources:
            raise SystemExit("v5 source-lock path set differs")
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
        checked_entries += 1
if seen != expected_schemas:
    raise SystemExit(f"incomplete source-lock schemas: {sorted(seen)}")
print(
    f"TPDCLEANV5_2X_SOURCES_OK locks={len(seen)} "
    f"checked_entries={checked_entries}",
    flush=True,
)
PY
}

v5_verify_data() {
    local v5_actual
    v5_actual="$(
        timeout 300s "$v5_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    if [[ "$v5_actual" != "$v5_training_data_sha256" ]]; then
        echo "TPDCLEANV5_2X_ABORT reason=training_data_drift expected=$v5_training_data_sha256 actual=$v5_actual" >&2
        return 1
    fi
}

v5_verify_gpu() {
    local v5_name
    local v5_free
    v5_name="$(
        nvidia-smi -i "$v5_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    v5_free="$(
        nvidia-smi -i "$v5_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$v5_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV5_2X_ABORT reason=unexpected_gpu gpu_uuid=$v5_gpu_uuid name=$v5_name" >&2
        return 1
    fi
    if (( v5_free < 7500 )); then
        echo "TPDCLEANV5_2X_ABORT reason=insufficient_memory gpu_uuid=$v5_gpu_uuid free_mib=$v5_free" >&2
        return 1
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v5_gpu_uuid" \
        "$v5_python" - "$v5_gpu_uuid" <<'PY'
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
    "TPDCLEANV5_2X_GPU_OK"
    f" gpu_uuid={expected_uuid}"
    f" torch={torch.__version__}"
    f" cuda={torch.version.cuda}"
    f" capability={torch.cuda.get_device_capability(0)}",
    flush=True,
)
PY
}

v5_write_launch_manifest() {
    local v5_manifest="$v5_launch_root/${v5_variant}_seed${v5_seed}.json"
    local v5_manifest_tmp="$v5_manifest.tmp.$$"
    local v5_memory_used
    local v5_memory_free
    local v5_utilization
    local v5_load_average
    v5_memory_used="$(
        nvidia-smi -i "$v5_gpu_uuid" \
            --query-gpu=memory.used --format=csv,noheader,nounits
    )"
    v5_memory_free="$(
        nvidia-smi -i "$v5_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    v5_utilization="$(
        nvidia-smi -i "$v5_gpu_uuid" \
            --query-gpu=utilization.gpu --format=csv,noheader,nounits
    )"
    v5_load_average="$(awk '{print $1","$2","$3}' /proc/loadavg)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v5_gpu_uuid" \
        "$v5_python" - \
        "$v5_variant" \
        "$v5_seed" \
        "$v5_gpu_uuid" \
        "$v5_run_dir" \
        "$v5_training_data_sha256" \
        "$v5_source_lock" \
        "$v5_memory_used" \
        "$v5_memory_free" \
        "$v5_utilization" \
        "$v5_load_average" \
        "$v5_manifest_tmp" <<'PY'
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
    "schema": "sctransnet_tpd_clean_v5_screen800_2x5090_launch_v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "variant": variant,
    "seed": int(seed),
    "candidate_family": (
        "spd_anchored_tpd_clean_v5_positive_context_selector"
    ),
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
        "warm_start": False,
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
    mv "$v5_manifest_tmp" "$v5_manifest"
}

v5_verify_sources
v5_source_lock_sha256="$(v5_sha256 "$v5_source_lock")"
v5_verify_data
v5_verify_gpu

if [[ -e "$v5_run_dir" || -L "$v5_run_dir" ]]; then
    echo "TPDCLEANV5_2X_ABORT reason=run_path_not_fresh variant=$v5_variant seed=$v5_seed path=$v5_run_dir" >&2
    exit 1
fi

v5_write_launch_manifest

echo "TPDCLEANV5_2X_START variant=$v5_variant seed=$v5_seed gpu_uuid=$v5_gpu_uuid cpu_threads=$v5_cpu_threads run_dir=$v5_run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$v5_gpu_uuid"
export PYTHONUNBUFFERED=1

"$v5_python" experiments/train_tpd_clean_v5.py \
    --variant "$v5_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$v5_repo/datasets" \
    --output-root "$v5_result_root" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed "$v5_seed" \
    --split-seed 20260722 \
    --val-fraction 0.20 \
    --eval-every 1 \
    --base-lr 0.001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --threshold 0.5 \
    --match-radius 3 \
    --tiny-area 9 \
    --run-tag "$v5_run_tag"

[[ "$(wc -l < "$v5_run_dir/metrics.jsonl")" -eq 800 ]]
jq -e --arg v5_variant "$v5_variant" --argjson v5_seed "$v5_seed" '
    .status == "complete" and
    .variant == $v5_variant and
    .dataset == "NUDT-SIRST" and
    .seed == $v5_seed and
    .selection_source == "internal_validation_only" and
    .official_test_accessed == false
' "$v5_run_dir/summary.json" >/dev/null

[[ -f "$v5_run_dir/last.pth.tar" && ! -L "$v5_run_dir/last.pth.tar" ]]
"$v5_python" - \
    "$v5_run_dir/last.pth.tar" \
    "$v5_variant" \
    "$v5_seed" <<'PY'
import pathlib
import sys

import torch

checkpoint_path = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
seed = int(sys.argv[3])
payload = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)
required = {
    "checkpoint_role": "last_evaluated_epoch",
    "dataset": "NUDT-SIRST",
    "epoch": 800,
    "official_test_accessed": False,
    "seed": seed,
    "selection_source": "internal_validation_only",
    "variant": variant,
}
for key, expected in required.items():
    if payload.get(key) != expected:
        raise SystemExit(
            f"last checkpoint {key} differs: "
            f"expected={expected!r} actual={payload.get(key)!r}"
        )
metadata = payload.get("model_metadata", {})
if (
    metadata.get("candidate_family")
    != "spd_anchored_tpd_clean_v5_positive_context_selector"
):
    raise SystemExit("last checkpoint candidate family differs")
from experiments.train_tpd_clean_v5 import build_clean_v5_model

model, _ = build_clean_v5_model(variant, seed)
incompatible = model.load_state_dict(payload["state_dict"], strict=True)
if incompatible.missing_keys or incompatible.unexpected_keys:
    raise SystemExit("last checkpoint strict load reported incompatibility")
print(
    "TPDCLEANV5_2X_LAST_OK"
    f" variant={variant} seed={seed} epoch={payload['epoch']}",
    flush=True,
)
PY

for v5_checkpoint in best.pth.tar best_miou.pth.tar; do
    "$v5_python" experiments/evaluate_tpd_clean_v5_pd_fa.py \
        --run-dir "$v5_run_dir" \
        --checkpoint "$v5_checkpoint" \
        --device cuda:0 \
        --expected-epochs 800
    v5_sweep="$v5_run_dir/pd_fa_sweep_${v5_checkpoint%.tar}.json"
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
    ' "$v5_sweep" >/dev/null
done

v5_verify_sources
v5_verify_data
if [[ "$(v5_sha256 "$v5_source_lock")" != "$v5_source_lock_sha256" ]]; then
    echo "TPDCLEANV5_2X_ABORT reason=source_lock_changed_during_run" >&2
    exit 1
fi
echo "TPDCLEANV5_2X_COMPLETE variant=$v5_variant seed=$v5_seed gpu_uuid=$v5_gpu_uuid epochs=800"
