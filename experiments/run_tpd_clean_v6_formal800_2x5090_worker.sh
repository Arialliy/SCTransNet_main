#!/usr/bin/env bash
set -euo pipefail

v6_usage() {
    echo "usage: $0 [--preflight] VARIANT SEED GPU_UUID" >&2
}

v6_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v6_mode="preflight"
    shift
fi
if [[ "$#" -ne 3 ]]; then
    v6_usage
    exit 2
fi

v6_variant="$1"
v6_seed="$2"
v6_gpu_uuid="$3"
v6_repo="/home/ly/SCTransNet_main"
v6_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v6_result_root="$v6_repo/experiments/results/tpd_clean_v6_formal800_2x5090_v1"
v6_smoke_root="$v6_repo/experiments/results/tpd_clean_v6_preflight_v1/smoke_reports"
v6_run_tag="formal800_exact_fp32_2x5090_v1"
v6_run_dir="$v6_result_root/NUDT-SIRST/$v6_variant/seed_${v6_seed}_${v6_run_tag}"
v6_source_lock="$v6_repo/experiments/tpd_clean_v6_exact_source_lock.json"
v6_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
v6_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v6_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

case "$v6_variant:$v6_seed:$v6_gpu_uuid" in
    tpd_clean_v6_full:42:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        v6_physical_index="2"
        ;;
    tpd_clean_v6_phase_capacity:42:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        v6_physical_index="3"
        ;;
    tpd_clean_v6_full:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        v6_physical_index="3"
        ;;
    tpd_clean_v6_phase_capacity:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        v6_physical_index="2"
        ;;
    *)
        echo "TPDCLEANV6_2X_ABORT reason=invalid_job_mapping variant=$v6_variant seed=$v6_seed gpu_uuid=$v6_gpu_uuid" >&2
        exit 2
        ;;
esac

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$v6_gpu_uuid"
export PYTHONHASHSEED="$v6_seed"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1

cd "$v6_repo"

[[ -x "$v6_python" ]] || {
    echo "TPDCLEANV6_2X_ABORT reason=python_not_executable path=$v6_python" >&2
    exit 1
}
[[ -f "$v6_source_lock" && ! -L "$v6_source_lock" ]] || {
    echo "TPDCLEANV6_2X_ABORT reason=missing_formal_source_lock path=$v6_source_lock" >&2
    exit 1
}

v6_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

v6_verify_source_data_and_smoke() {
    "$v6_python" - \
        "$v6_repo" \
        "$v6_source_lock" \
        "$v6_smoke_root" \
        "$v6_training_data_sha256" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
source_lock = pathlib.Path(sys.argv[2]).resolve()
smoke_root = pathlib.Path(sys.argv[3]).resolve()
expected_data_sha256 = sys.argv[4]

if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from experiments import train_tpd_clean_v6_exact as exact
from experiments import verify_tpd_clean_v6_smoke_reports as smoke_verifier

dataset_root = repo / "datasets" / "NUDT-SIRST"
index_bytes, identifiers = exact.read_official_training_index(
    dataset_root,
    "NUDT-SIRST",
)
actual_data_sha256 = exact.official_training_data_sha256(
    dataset_root,
    "NUDT-SIRST",
    identifiers,
    index_bytes,
)
if actual_data_sha256 != expected_data_sha256:
    raise SystemExit(
        "training data differs: "
        f"expected={expected_data_sha256} actual={actual_data_sha256}"
    )

source_contract = exact.source_lock_contract(
    actual_data_sha256,
    source_lock,
)
if source_contract.get("training_data") != actual_data_sha256:
    raise SystemExit("formal source lock does not bind the training data")

smoke = smoke_verifier.validate_smoke_reports(smoke_root)
if (
    smoke.get("status") != "complete"
    or smoke.get("passed") is not True
    or smoke.get("physical_gpu_reports_verified")
    != {
        "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
        "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    }
):
    raise SystemExit("persistent CPU/GPU2/GPU3 smoke verification failed")

print(
    "TPDCLEANV6_2X_CONTRACTS_OK"
    f" exact_sources={len(source_contract) - 2}"
    f" smoke_reports={len(smoke['report_sha256'])}"
    f" training_data_sha256={actual_data_sha256}",
    flush=True,
)
PY
}

v6_verify_gpu() {
    local v6_actual_index
    local v6_actual_name
    v6_actual_index="$(
        nvidia-smi -i "$v6_gpu_uuid" \
            --query-gpu=index --format=csv,noheader,nounits
    )"
    v6_actual_name="$(
        nvidia-smi -i "$v6_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    if [[ "$v6_actual_index" != "$v6_physical_index" ]]; then
        echo "TPDCLEANV6_2X_ABORT reason=gpu_index_mismatch expected=$v6_physical_index actual=$v6_actual_index uuid=$v6_gpu_uuid" >&2
        return 1
    fi
    if [[ "$v6_actual_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV6_2X_ABORT reason=gpu_name_mismatch expected=NVIDIA_GeForce_RTX_5090 actual=$v6_actual_name uuid=$v6_gpu_uuid" >&2
        return 1
    fi

    "$v6_python" - "$v6_gpu_uuid" "$v6_physical_index" <<'PY'
import os
import sys

import torch

from experiments.train_tpd_clean_v6_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain the one assigned UUID")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the worker must expose exactly one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected cuda:0 model: {torch.cuda.get_device_name(0)}")
properties = torch.cuda.get_device_properties(0)
actual_uuid = normalized_gpu_uuid(getattr(properties, "uuid", ""))
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"cuda:0 UUID differs: expected={expected_uuid} actual={actual_uuid}"
    )
if os.environ.get("PYTHONHASHSEED") not in {"42", "3407"}:
    raise SystemExit("PYTHONHASHSEED is not a preregistered model seed")
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise SystemExit("CUBLAS_WORKSPACE_CONFIG differs from the exact contract")
if torch.get_num_threads() != 1:
    raise SystemExit(
        f"torch CPU thread count differs: {torch.get_num_threads()}"
    )
print(
    "TPDCLEANV6_2X_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " logical_device=cuda:0"
    f" name={torch.cuda.get_device_name(0)}",
    flush=True,
)
PY
}

v6_initialization_mode() {
    "$v6_python" - \
        "$v6_run_dir" \
        "$v6_variant" \
        "$v6_seed" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
seed = int(sys.argv[3])
if not run_dir.exists() and not run_dir.is_symlink():
    print("fresh")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"run path must be a regular directory: {run_dir}")

summary_path = run_dir / "summary.json"
if summary_path.is_symlink():
    raise SystemExit(f"summary must not be a symbolic link: {summary_path}")
if not summary_path.is_file():
    print("exact-resume")
    raise SystemExit(0)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") != "complete":
    print("exact-resume")
    raise SystemExit(0)
if summary.get("variant") != variant or summary.get("seed") != seed:
    raise SystemExit("complete summary identity differs from the requested task")
formal = summary.get("formal_contract", {})
if (
    formal.get("epochs") != 800
    or formal.get("eval_every") != 1
    or formal.get("workers") != 0
    or formal.get("amp") is not False
    or formal.get("eps") != 1e-6
):
    raise SystemExit("complete summary formal contract differs")
metrics_path = run_dir / "metrics.jsonl"
if not metrics_path.is_file() or metrics_path.is_symlink():
    raise SystemExit("complete summary has no regular metrics journal")
events = [
    json.loads(line)
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(events) != 800 or [
    event.get("epoch") for event in events
] != list(range(1, 801)):
    raise SystemExit("complete summary has no contiguous 800-epoch journal")
for name in ("best.pth.tar", "best_miou.pth.tar", "last.pth.tar"):
    path = run_dir / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"complete summary lacks regular checkpoint: {name}")
print("complete")
PY
}

v6_initialization="$(v6_initialization_mode)"
v6_verify_source_data_and_smoke
v6_source_lock_sha256="$(v6_sha256 "$v6_source_lock")"

if [[ "$v6_mode" == "run" && "$v6_initialization" == "complete" ]]; then
    echo "TPDCLEANV6_2X_IDEMPOTENT_COMPLETE variant=$v6_variant seed=$v6_seed run_dir=$v6_run_dir"
    exit 0
fi

v6_verify_gpu
echo "TPDCLEANV6_2X_PREFLIGHT_OK variant=$v6_variant seed=$v6_seed physical_gpu=$v6_physical_index gpu_uuid=$v6_gpu_uuid initialization=$v6_initialization"
if [[ "$v6_mode" == "preflight" ]]; then
    exit 0
fi

mkdir -p \
    "$v6_result_root/logs" \
    "$v6_result_root/.locks"
v6_log="$v6_result_root/logs/${v6_variant}_seed${v6_seed}.log"
exec > >(tee -a "$v6_log") 2>&1

exec 9>"$v6_result_root/.locks/${v6_variant}_seed${v6_seed}.lock"
if ! flock -n 9; then
    echo "TPDCLEANV6_2X_ABORT reason=task_lock_held variant=$v6_variant seed=$v6_seed" >&2
    exit 1
fi
exec 8>"$v6_result_root/.locks/physical_gpu${v6_physical_index}.lock"
if ! flock -n 8; then
    echo "TPDCLEANV6_2X_ABORT reason=gpu_lane_lock_held physical_gpu=$v6_physical_index uuid=$v6_gpu_uuid" >&2
    exit 1
fi

# Re-evaluate after both locks in case another invocation completed this task.
v6_initialization="$(v6_initialization_mode)"
if [[ "$v6_initialization" == "complete" ]]; then
    echo "TPDCLEANV6_2X_IDEMPOTENT_COMPLETE variant=$v6_variant seed=$v6_seed run_dir=$v6_run_dir"
    exit 0
fi
if [[ "$v6_initialization" == "fresh" ]]; then
    v6_init_flag="--fresh"
else
    v6_init_flag="--exact-resume"
fi

echo "TPDCLEANV6_2X_START variant=$v6_variant seed=$v6_seed physical_gpu=$v6_physical_index gpu_uuid=$v6_gpu_uuid mode=$v6_init_flag run_dir=$v6_run_dir"
"$v6_python" experiments/train_tpd_clean_v6_exact.py \
    --variant "$v6_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$v6_repo/datasets" \
    --output-root "$v6_result_root" \
    --run-tag "$v6_run_tag" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed "$v6_seed" \
    --split-seed 20260722 \
    --val-fraction 0.20 \
    --eval-every 1 \
    --base-lr 0.001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --threshold 0.5 \
    --match-radius 3 \
    --tiny-area 9 \
    --eps 0.000001 \
    --exact-source-lock "$v6_source_lock" \
    "$v6_init_flag"

if [[ "$(v6_initialization_mode)" != "complete" ]]; then
    echo "TPDCLEANV6_2X_ABORT reason=run_not_complete variant=$v6_variant seed=$v6_seed" >&2
    exit 1
fi
v6_verify_source_data_and_smoke
if [[ "$(v6_sha256 "$v6_source_lock")" != "$v6_source_lock_sha256" ]]; then
    echo "TPDCLEANV6_2X_ABORT reason=source_lock_changed_during_run" >&2
    exit 1
fi
echo "TPDCLEANV6_2X_COMPLETE variant=$v6_variant seed=$v6_seed physical_gpu=$v6_physical_index epochs=800"
