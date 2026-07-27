#!/usr/bin/env bash
set -euo pipefail

dch_usage() {
    echo "usage: $0 [--preflight] VARIANT SEED GPU_UUID" >&2
}

dch_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    dch_mode="preflight"
    shift
fi
if [[ "$#" -ne 3 ]]; then
    dch_usage
    exit 2
fi

dch_variant="$1"
dch_seed="$2"
dch_gpu_uuid="$3"
dch_repo="/home/ly/SCTransNet_main"
dch_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
dch_result_root="$dch_repo/experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
dch_smoke_root="$dch_repo/experiments/results/tpd_clean_v7_dch_preflight_v1/smoke_reports"
dch_run_tag="formal800_exact_fp32_2x5090_v1"
dch_run_dir="$dch_result_root/NUDT-SIRST/$dch_variant/seed_${dch_seed}_${dch_run_tag}"
dch_source_lock="$dch_repo/experiments/tpd_clean_v7_dch_exact_source_lock.json"
dch_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
dch_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
dch_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

# Counterbalanced two-lane schedule:
# physical GPU2: Full/42 -> Capacity/3407
# physical GPU3: Capacity/42 -> Full/3407
case "$dch_variant:$dch_seed:$dch_gpu_uuid" in
    tpd_clean_v7_dch_full:42:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        dch_physical_index="2"
        ;;
    tpd_clean_v7_dch_capacity:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        dch_physical_index="2"
        ;;
    tpd_clean_v7_dch_capacity:42:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        dch_physical_index="3"
        ;;
    tpd_clean_v7_dch_full:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        dch_physical_index="3"
        ;;
    *)
        echo "TPDCLEANV7DCH_2X_ABORT reason=invalid_job_mapping variant=$dch_variant seed=$dch_seed gpu_uuid=$dch_gpu_uuid" >&2
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
export CUDA_VISIBLE_DEVICES="$dch_gpu_uuid"
export TPD_DCH_PHYSICAL_GPU_INDEX="$dch_physical_index"
export TPD_DCH_PHYSICAL_GPU_UUID="$dch_gpu_uuid"
export PYTHONHASHSEED="$dch_seed"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1

cd "$dch_repo"

[[ -x "$dch_python" ]] || {
    echo "TPDCLEANV7DCH_2X_ABORT reason=python_not_executable path=$dch_python" >&2
    exit 1
}
[[ -f "$dch_source_lock" && ! -L "$dch_source_lock" ]] || {
    echo "TPDCLEANV7DCH_2X_ABORT reason=missing_formal_source_lock path=$dch_source_lock" >&2
    exit 1
}

dch_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

dch_verify_source_data_and_smoke() {
    "$dch_python" - \
        "$dch_repo" \
        "$dch_source_lock" \
        "$dch_smoke_root" \
        "$dch_training_data_sha256" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
source_lock = pathlib.Path(sys.argv[2]).resolve()
smoke_root = pathlib.Path(sys.argv[3]).resolve()
expected_data_sha256 = sys.argv[4]

if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from experiments import train_tpd_clean_v6_exact as shared_exact
from experiments import train_tpd_clean_v7_dch_exact as exact
from experiments import verify_tpd_clean_v7_dch_smoke_reports as smoke_verifier

dataset_root = repo / "datasets" / "NUDT-SIRST"
index_bytes, identifiers = shared_exact.read_official_training_index(
    dataset_root,
    "NUDT-SIRST",
)
actual_data_sha256 = shared_exact.official_training_data_sha256(
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
    raise SystemExit("DCH formal source lock does not bind the training data")

smoke = smoke_verifier.validate_smoke_reports(smoke_root)
expected_gpus = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
if (
    smoke.get("status") != "complete"
    or smoke.get("passed") is not True
    or smoke.get("physical_gpu_reports_verified") != expected_gpus
):
    raise SystemExit("persistent DCH CPU/GPU2/GPU3 smoke verification failed")

print(
    "TPDCLEANV7DCH_2X_CONTRACTS_OK"
    f" exact_sources={len(source_contract) - 2}"
    f" smoke_reports={len(smoke['report_sha256'])}"
    f" training_data_sha256={actual_data_sha256}",
    flush=True,
)
PY
}

dch_verify_gpu() {
    local dch_actual_index
    local dch_actual_name
    dch_actual_index="$(
        nvidia-smi -i "$dch_gpu_uuid" \
            --query-gpu=index --format=csv,noheader,nounits
    )"
    dch_actual_name="$(
        nvidia-smi -i "$dch_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    if [[ "$dch_actual_index" != "$dch_physical_index" ]]; then
        echo "TPDCLEANV7DCH_2X_ABORT reason=gpu_index_mismatch expected=$dch_physical_index actual=$dch_actual_index uuid=$dch_gpu_uuid" >&2
        return 1
    fi
    if [[ "$dch_actual_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV7DCH_2X_ABORT reason=gpu_name_mismatch expected=NVIDIA_GeForce_RTX_5090 actual=$dch_actual_name uuid=$dch_gpu_uuid" >&2
        return 1
    fi

    "$dch_python" - "$dch_gpu_uuid" "$dch_physical_index" <<'PY'
import os
import sys

import torch

from experiments.train_tpd_clean_v7_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain the assigned UUID")
if os.environ.get("TPD_DCH_PHYSICAL_GPU_INDEX") != expected_index:
    raise SystemExit("TPD_DCH_PHYSICAL_GPU_INDEX differs from the assignment")
if os.environ.get("TPD_DCH_PHYSICAL_GPU_UUID") != expected_uuid:
    raise SystemExit("TPD_DCH_PHYSICAL_GPU_UUID differs from the assignment")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the DCH worker must expose exactly one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected cuda:0 model: {torch.cuda.get_device_name(0)}")
properties = torch.cuda.get_device_properties(0)
actual_uuid = normalized_gpu_uuid(getattr(properties, "uuid", ""))
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"cuda:0 UUID differs: expected={expected_uuid} actual={actual_uuid}"
    )
if os.environ.get("PYTHONHASHSEED") not in {"42", "3407"}:
    raise SystemExit("PYTHONHASHSEED is not a preregistered DCH seed")
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise SystemExit("CUBLAS_WORKSPACE_CONFIG differs from the exact contract")
if torch.get_num_threads() != 1:
    raise SystemExit(
        f"torch CPU thread count differs: {torch.get_num_threads()}"
    )
print(
    "TPDCLEANV7DCH_2X_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " logical_device=cuda:0"
    f" name={torch.cuda.get_device_name(0)}",
    flush=True,
)
PY
}

dch_initialization_mode() {
    "$dch_python" - \
        "$dch_run_dir" \
        "$dch_variant" \
        "$dch_seed" <<'PY'
import json
import pathlib
import sys

from experiments.train_tpd_clean_v7_dch_exact import (
    STORED_VALIDATION_METRICS,
)

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
    raise SystemExit("complete DCH summary identity differs")
formal = summary.get("formal_contract", {})
if (
    formal.get("epochs") != 800
    or formal.get("eval_every") != 1
    or formal.get("workers") != 0
    or formal.get("amp") is not False
    or formal.get("eps") != 1e-6
):
    raise SystemExit("complete DCH summary formal contract differs")
if summary.get("stored_validation_metrics") != list(
    STORED_VALIDATION_METRICS
):
    raise SystemExit("complete DCH summary does not store all 17 metrics")

metrics_path = run_dir / "metrics.jsonl"
if not metrics_path.is_file() or metrics_path.is_symlink():
    raise SystemExit("complete DCH summary has no regular metrics journal")
events = [
    json.loads(line)
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(events) != 800 or [
    event.get("epoch") for event in events
] != list(range(1, 801)):
    raise SystemExit("complete DCH summary has no contiguous 800 epochs")
missing = [
    (int(event["epoch"]), sorted(set(STORED_VALIDATION_METRICS) - set(event)))
    for event in events
    if not set(STORED_VALIDATION_METRICS).issubset(event)
]
if missing:
    raise SystemExit(f"DCH metrics journal lacks validation fields: {missing[:3]}")
for name in ("best.pth.tar", "best_miou.pth.tar", "last.pth.tar"):
    path = run_dir / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"complete DCH run lacks checkpoint: {name}")
print("complete")
PY
}

dch_write_assignment() {
    local dch_assignment_dir
    local dch_assignment_path
    dch_assignment_dir="$dch_result_root/lane_assignments"
    dch_assignment_path="$dch_assignment_dir/${dch_variant}_seed${dch_seed}.json"
    mkdir -p "$dch_assignment_dir"
    "$dch_python" - \
        "$dch_assignment_path" \
        "$dch_variant" \
        "$dch_seed" \
        "$dch_physical_index" \
        "$dch_gpu_uuid" \
        "$dch_run_dir" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sctransnet_tpd_clean_v7_dch_lane_assignment_v1",
    "variant": sys.argv[2],
    "seed": int(sys.argv[3]),
    "physical_gpu_index": int(sys.argv[4]),
    "physical_gpu_uuid": sys.argv[5],
    "logical_device": "cuda:0",
    "run_directory": sys.argv[6],
}
if path.is_symlink():
    raise SystemExit(f"lane assignment must not be a symbolic link: {path}")
if path.exists():
    if not path.is_file():
        raise SystemExit(f"lane assignment must be a regular file: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit(f"lane assignment differs: {path}")
else:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
print(f"TPDCLEANV7DCH_2X_ASSIGNMENT_OK path={path}", flush=True)
PY
}

dch_initialization="$(dch_initialization_mode)"
dch_verify_source_data_and_smoke
dch_source_lock_sha256="$(dch_sha256 "$dch_source_lock")"

if [[ "$dch_mode" == "run" && "$dch_initialization" == "complete" ]]; then
    echo "TPDCLEANV7DCH_2X_IDEMPOTENT_COMPLETE variant=$dch_variant seed=$dch_seed run_dir=$dch_run_dir"
    exit 0
fi

dch_verify_gpu
echo "TPDCLEANV7DCH_2X_PREFLIGHT_OK variant=$dch_variant seed=$dch_seed physical_gpu=$dch_physical_index gpu_uuid=$dch_gpu_uuid initialization=$dch_initialization"
if [[ "$dch_mode" == "preflight" ]]; then
    exit 0
fi

mkdir -p \
    "$dch_result_root/logs" \
    "$dch_result_root/.locks"
dch_log="$dch_result_root/logs/${dch_variant}_seed${dch_seed}.log"
exec > >(tee -a "$dch_log") 2>&1

exec 9>"$dch_result_root/.locks/${dch_variant}_seed${dch_seed}.lock"
if ! flock -n 9; then
    echo "TPDCLEANV7DCH_2X_ABORT reason=task_lock_held variant=$dch_variant seed=$dch_seed" >&2
    exit 1
fi
exec 8>"$dch_result_root/.locks/physical_gpu${dch_physical_index}.lock"
if ! flock -n 8; then
    echo "TPDCLEANV7DCH_2X_ABORT reason=gpu_lane_lock_held physical_gpu=$dch_physical_index uuid=$dch_gpu_uuid" >&2
    exit 1
fi

# Re-evaluate after acquiring both locks.
dch_initialization="$(dch_initialization_mode)"
if [[ "$dch_initialization" == "complete" ]]; then
    echo "TPDCLEANV7DCH_2X_IDEMPOTENT_COMPLETE variant=$dch_variant seed=$dch_seed run_dir=$dch_run_dir"
    exit 0
fi
if [[ "$dch_initialization" == "fresh" ]]; then
    dch_init_flag="--fresh"
else
    dch_init_flag="--exact-resume"
fi

dch_write_assignment
echo "TPDCLEANV7DCH_2X_START variant=$dch_variant seed=$dch_seed physical_gpu=$dch_physical_index gpu_uuid=$dch_gpu_uuid mode=$dch_init_flag run_dir=$dch_run_dir"
"$dch_python" experiments/train_tpd_clean_v7_dch_exact.py \
    --variant "$dch_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$dch_repo/datasets" \
    --output-root "$dch_result_root" \
    --run-tag "$dch_run_tag" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed "$dch_seed" \
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
    --exact-source-lock "$dch_source_lock" \
    "$dch_init_flag"

if [[ "$(dch_initialization_mode)" != "complete" ]]; then
    echo "TPDCLEANV7DCH_2X_ABORT reason=run_not_complete variant=$dch_variant seed=$dch_seed" >&2
    exit 1
fi
dch_verify_source_data_and_smoke
if [[ "$(dch_sha256 "$dch_source_lock")" != "$dch_source_lock_sha256" ]]; then
    echo "TPDCLEANV7DCH_2X_ABORT reason=source_lock_changed_during_run" >&2
    exit 1
fi
echo "TPDCLEANV7DCH_2X_COMPLETE variant=$dch_variant seed=$dch_seed physical_gpu=$dch_physical_index gpu_uuid=$dch_gpu_uuid epochs=800 stored_validation_metrics=17"
