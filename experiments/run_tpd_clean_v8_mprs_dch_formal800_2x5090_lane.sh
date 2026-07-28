#!/usr/bin/env bash
set -euo pipefail

v8_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v8_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 [--preflight] LANE GPU_UUID" >&2
    exit 2
fi

v8_lane="$1"
v8_gpu_uuid="$2"
v8_repo="${TPD_V8_MPRS_DCH_REPO:-/home/ly/SCTransNet_main}"
v8_python="${TPD_V8_MPRS_DCH_TRAINING_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v8_launcher="$v8_repo/experiments/launch_tpd_clean_v8_mprs_dch_formal800_2x5090.sh"
v8_trainer="$v8_repo/experiments/train_tpd_clean_v8_mprs_dch_exact.py"
v8_source_lock="${TPD_V8_MPRS_DCH_SOURCE_LOCK:-$v8_repo/experiments/tpd_clean_v8_mprs_dch_exact_source_lock.json}"
v8_result_root="${TPD_V8_MPRS_DCH_RESULT_ROOT:-$v8_repo/experiments/results/tpd_clean_v8_mprs_dch_formal800_2x5090_v1}"
v8_run_tag="formal800_exact_fp32_2x5090_v1"
v8_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v8_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

# Counterbalanced serial schedule:
# physical GPU2: Full/42 -> Capacity/3407
# physical GPU3: Capacity/42 -> Full/3407
case "$v8_lane:$v8_gpu_uuid" in
    gpu2:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        v8_physical_index="2"
        v8_variants=(
            tpd_clean_v8_mprs_dch_full
            tpd_clean_v8_mprs_dch_capacity
        )
        v8_seeds=(42 3407)
        ;;
    gpu3:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        v8_physical_index="3"
        v8_variants=(
            tpd_clean_v8_mprs_dch_capacity
            tpd_clean_v8_mprs_dch_full
        )
        v8_seeds=(42 3407)
        ;;
    *)
        echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=invalid_lane_mapping lane=$v8_lane gpu_uuid=$v8_gpu_uuid" >&2
        exit 2
        ;;
esac

[[ -x "$v8_python" ]] || {
    echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=python_not_executable path=$v8_python" >&2
    exit 1
}
[[ -x "$v8_launcher" ]] || {
    echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=launcher_not_executable path=$v8_launcher" >&2
    exit 1
}
[[ -f "$v8_trainer" && ! -L "$v8_trainer" ]] || {
    echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=missing_v8_exact_trainer path=$v8_trainer" >&2
    exit 1
}

# A restarted systemd lane must re-check the same fail-closed authorization
# and artifact bindings as the parent launcher.
"$v8_launcher" --validate-only

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$v8_gpu_uuid"
export TPD_V8_MPRS_DCH_PHYSICAL_GPU_INDEX="$v8_physical_index"
export TPD_V8_MPRS_DCH_PHYSICAL_GPU_UUID="$v8_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1

cd "$v8_repo"

v8_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

v8_verify_gpu() {
    local v8_observed
    local v8_expected
    v8_observed="$(
        nvidia-smi -i "$v8_gpu_uuid" \
            --query-gpu=index,name,uuid \
            --format=csv,noheader,nounits
    )"
    v8_expected="$v8_physical_index, NVIDIA GeForce RTX 5090, $v8_gpu_uuid"
    if [[ "$v8_observed" != "$v8_expected" ]]; then
        echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=gpu_identity_mismatch expected=$v8_expected observed=$v8_observed" >&2
        return 1
    fi

    "$v8_python" - "$v8_gpu_uuid" "$v8_physical_index" <<'PY'
import os
import sys

import torch

from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain the assigned UUID")
if (
    os.environ.get("TPD_V8_MPRS_DCH_PHYSICAL_GPU_INDEX")
    != expected_index
):
    raise SystemExit("V8 physical GPU index differs from the assignment")
if (
    os.environ.get("TPD_V8_MPRS_DCH_PHYSICAL_GPU_UUID")
    != expected_uuid
):
    raise SystemExit("V8 physical GPU UUID differs from the assignment")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the V8 lane must expose exactly one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected cuda:0 model: {torch.cuda.get_device_name(0)}")
properties = torch.cuda.get_device_properties(0)
actual_uuid = normalized_gpu_uuid(getattr(properties, "uuid", ""))
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"cuda:0 UUID differs: expected={expected_uuid} actual={actual_uuid}"
    )
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise SystemExit("CUBLAS_WORKSPACE_CONFIG differs from the exact contract")
if torch.get_num_threads() != 1:
    raise SystemExit(
        f"torch CPU thread count differs: {torch.get_num_threads()}"
    )
print(
    "TPDCLEANV8MPRSDCH_2X_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " logical_device=cuda:0",
    flush=True,
)
PY
}

v8_initialization_mode() {
    local v8_run_dir="$1"
    local v8_variant="$2"
    local v8_seed="$3"
    "$v8_python" - "$v8_run_dir" "$v8_variant" "$v8_seed" <<'PY'
import json
import pathlib
import sys

ENTRY_SCHEMA = "sctransnet_tpd_clean_v8_mprs_dch_exact_entry_v1"
SUMMARY_SCHEMA = "sctransnet_tpd_clean_v8_mprs_dch_completion_summary_v1"
RUN_PREFIX = "tpd-clean-v8-mprs-dch-exact:"
SOURCE_LOCK_KEY = "tpd_clean_v8_mprs_dch_exact_source_lock"
V7_ENTRY_SCHEMA = "sctransnet_tpd_clean_v7_dch_exact_entry_v1"
STORED_METRICS = [
    "val_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
]

run_dir = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
seed = int(sys.argv[3])
if not run_dir.exists() and not run_dir.is_symlink():
    print("fresh")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"run path must be a regular directory: {run_dir}")

protocol_path = run_dir / "protocol.json"
if not protocol_path.is_file() or protocol_path.is_symlink():
    raise SystemExit(
        "existing run directory has no regular V8 protocol; refusing resume"
    )
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
if protocol.get("schema") == V7_ENTRY_SCHEMA:
    raise SystemExit(
        "cross-version exact resume from a V7 protocol/journal is forbidden"
    )
if protocol.get("schema") != ENTRY_SCHEMA:
    raise SystemExit("existing protocol is not V8-MPRS-DCH")
identity = protocol.get("run_identity")
if not isinstance(identity, dict):
    raise SystemExit("existing V8 protocol has no run identity")
training = identity.get("training_contract")
determinism = (
    training.get("determinism") if isinstance(training, dict) else None
)
source_locks = identity.get("source_locks")
if (
    identity.get("variant") != variant
    or identity.get("seed") != seed
    or not isinstance(identity.get("run_id"), str)
    or not identity["run_id"].startswith(RUN_PREFIX)
    or not isinstance(determinism, dict)
    or determinism.get("entry_schema") != ENTRY_SCHEMA
    or not isinstance(source_locks, dict)
    or SOURCE_LOCK_KEY not in source_locks
    or "tpd_clean_v7_dch_exact_source_lock" in source_locks
):
    raise SystemExit(
        "existing protocol identity is not the requested V8 run"
    )

summary_path = run_dir / "summary.json"
if summary_path.is_symlink():
    raise SystemExit(f"summary must not be a symbolic link: {summary_path}")
summary = None
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
if not isinstance(summary, dict) or summary.get("status") != "complete":
    marker = run_dir / "exact_journal" / "active.json"
    if not marker.is_file() or marker.is_symlink():
        raise SystemExit(
            "incomplete V8 directory has no committed exact journal"
        )
    print("exact-resume")
    raise SystemExit(0)

if (
    summary.get("schema") != SUMMARY_SCHEMA
    or summary.get("variant") != variant
    or summary.get("seed") != seed
):
    raise SystemExit("complete V8 summary identity differs")
formal = summary.get("formal_contract")
if formal != {
    "epochs": 800,
    "eval_every": 1,
    "workers": 0,
    "amp": False,
    "eps": 1e-6,
    "cublas_workspace_config": ":4096:8",
    "initialization_modes": ["fresh", "exact_resume"],
}:
    raise SystemExit("complete V8 summary formal contract differs")
if summary.get("stored_validation_metrics") != STORED_METRICS:
    raise SystemExit("complete V8 summary does not store all 17 metrics")

metrics_path = run_dir / "metrics.jsonl"
if not metrics_path.is_file() or metrics_path.is_symlink():
    raise SystemExit("complete V8 summary has no regular metrics journal")
events = [
    json.loads(line)
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(events) != 800 or [
    event.get("epoch") for event in events
] != list(range(1, 801)):
    raise SystemExit("complete V8 summary has no contiguous 800 epochs")
for event in events:
    if not set(STORED_METRICS).issubset(event):
        raise SystemExit(
            f"V8 metrics event {event.get('epoch')} lacks validation fields"
        )
for name in ("best.pth.tar", "best_miou.pth.tar", "last.pth.tar"):
    path = run_dir / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"complete V8 run lacks checkpoint: {name}")
print("complete")
PY
}

v8_run_one() {
    local v8_variant="$1"
    local v8_seed="$2"
    local v8_run_dir
    local v8_initialization
    local v8_init_flag
    local v8_source_lock_sha256
    local v8_task_log

    v8_run_dir="$v8_result_root/NUDT-SIRST/$v8_variant/seed_${v8_seed}_${v8_run_tag}"
    v8_initialization="$(v8_initialization_mode "$v8_run_dir" "$v8_variant" "$v8_seed")"
    echo "TPDCLEANV8MPRSDCH_2X_PREFLIGHT_OK variant=$v8_variant seed=$v8_seed physical_gpu=$v8_physical_index gpu_uuid=$v8_gpu_uuid initialization=$v8_initialization"
    if [[ "$v8_lane_mode" == "preflight" ]]; then
        return 0
    fi
    if [[ "$v8_initialization" == "complete" ]]; then
        echo "TPDCLEANV8MPRSDCH_2X_IDEMPOTENT_COMPLETE variant=$v8_variant seed=$v8_seed run_dir=$v8_run_dir"
        return 0
    fi

    mkdir -p "$v8_result_root/logs" "$v8_result_root/.locks"
    v8_task_log="$v8_result_root/logs/${v8_variant}_seed${v8_seed}.log"
    {
        exec 9>"$v8_result_root/.locks/${v8_variant}_seed${v8_seed}.lock"
        if ! flock -n 9; then
            echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=task_lock_held variant=$v8_variant seed=$v8_seed" >&2
            return 1
        fi

        # Re-evaluate only after both the lane and task locks are held.
        v8_initialization="$(v8_initialization_mode "$v8_run_dir" "$v8_variant" "$v8_seed")"
        if [[ "$v8_initialization" == "complete" ]]; then
            echo "TPDCLEANV8MPRSDCH_2X_IDEMPOTENT_COMPLETE variant=$v8_variant seed=$v8_seed run_dir=$v8_run_dir"
            return 0
        fi
        if [[ "$v8_initialization" == "fresh" ]]; then
            v8_init_flag="--fresh"
        else
            v8_init_flag="--exact-resume"
        fi
        export PYTHONHASHSEED="$v8_seed"
        v8_source_lock_sha256="$(v8_sha256 "$v8_source_lock")"
        echo "TPDCLEANV8MPRSDCH_2X_START variant=$v8_variant seed=$v8_seed physical_gpu=$v8_physical_index gpu_uuid=$v8_gpu_uuid mode=$v8_init_flag run_dir=$v8_run_dir"
        "$v8_python" experiments/train_tpd_clean_v8_mprs_dch_exact.py \
            --variant "$v8_variant" \
            --dataset NUDT-SIRST \
            --dataset-dir "$v8_repo/datasets" \
            --output-root "$v8_result_root" \
            --run-tag "$v8_run_tag" \
            --device cuda:0 \
            --epochs 800 \
            --batch-size 16 \
            --patch-size 256 \
            --workers 0 \
            --seed "$v8_seed" \
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
            --exact-source-lock "$v8_source_lock" \
            "$v8_init_flag"

        if [[ "$(v8_initialization_mode "$v8_run_dir" "$v8_variant" "$v8_seed")" != "complete" ]]; then
            echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=run_not_complete variant=$v8_variant seed=$v8_seed" >&2
            return 1
        fi
        if [[ "$(v8_sha256 "$v8_source_lock")" != "$v8_source_lock_sha256" ]]; then
            echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=source_lock_changed_during_run" >&2
            return 1
        fi
        echo "TPDCLEANV8MPRSDCH_2X_COMPLETE variant=$v8_variant seed=$v8_seed physical_gpu=$v8_physical_index gpu_uuid=$v8_gpu_uuid epochs=800 stored_validation_metrics=17"
    } > >(tee -a "$v8_task_log") 2>&1
}

v8_verify_gpu
if [[ "$v8_lane_mode" == "run" ]]; then
    mkdir -p "$v8_result_root/.locks"
    exec 8>"$v8_result_root/.locks/physical_gpu${v8_physical_index}.lock"
    if ! flock -n 8; then
        echo "TPDCLEANV8MPRSDCH_2X_LANE_ABORT reason=gpu_lane_lock_held physical_gpu=$v8_physical_index uuid=$v8_gpu_uuid" >&2
        exit 1
    fi
fi

for v8_index in 0 1; do
    v8_run_one "${v8_variants[$v8_index]}" "${v8_seeds[$v8_index]}"
done

echo "TPDCLEANV8MPRSDCH_2X_LANE_COMPLETE lane=$v8_lane gpu_uuid=$v8_gpu_uuid tasks=2 mode=$v8_lane_mode"
