#!/usr/bin/env bash
set -euo pipefail

repo_root=${FINAL_MODEL_SEED42_POST_REPO_ROOT:-/home/ly/SCTransNet_main}
python_bin=${FINAL_MODEL_SEED42_POST_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
module=experiments.final_model_seed42_certification_replay_posttraining
output_root="$repo_root/experiments/results/final_model_seed42_certification_replay_v1"

gpu2_uuid=GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
gpu3_uuid=GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3

usage() {
  echo "usage: $0 (--dry-run | --run)" >&2
}

if [[ "$#" -ne 1 || ( "$1" != "--dry-run" && "$1" != "--run" ) ]]; then
  usage
  exit 2
fi
mode="$1"

cd "$repo_root"

if [[ "$mode" == "--dry-run" ]]; then
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    "$python_bin" -m "$module" --dry-run
  printf '%s\n' \
    "DRY-RUN ONLY; no GPU command was launched." \
    "Future matrix: seed42 replay B/D x own best_miou/best = 4 sweeps." \
    "GPU2 is bound to B; GPU3 is bound to D." \
    "Then CPU paired-image analysis, locked Gate comparison, and closure."
  exit 0
fi

mkdir -p "$output_root"
exec 9>"$output_root/.seed42_replay_posttraining_gpu23.lock"
if ! flock -n 9; then
  echo "another seed42 replay post-training closure is active" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" --write-summary

temporary_root=$(mktemp -d "$output_root/.seed42-replay-posttraining.XXXXXX")
declare -a active_pids=()
declare -A pid_arms=()
declare -A pid_logs=()

cleanup() {
  local child_pid
  for child_pid in "${active_pids[@]}"; do
    if kill -0 "$child_pid" 2>/dev/null; then
      kill "$child_pid" 2>/dev/null || true
    fi
  done
  if [[ -d "$temporary_root" ]]; then
    rm -rf -- "$temporary_root"
  fi
}
trap cleanup INT TERM EXIT

launch_arm() {
  local arm="$1"
  local physical_index="$2"
  local physical_uuid="$3"
  local log_path="$temporary_root/arm_${arm}.log"
  CUDA_VISIBLE_DEVICES="$physical_uuid" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX="$physical_index" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID="$physical_uuid" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONHASHSEED=42 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  TORCH_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
    "$python_bin" -m "$module" \
      --execute \
      --arm "$arm" \
      --device cuda:0 \
      --physical-gpu-index "$physical_index" \
      --physical-gpu-uuid "$physical_uuid" \
      >"$log_path" 2>&1 &
  local child_pid=$!
  active_pids+=("$child_pid")
  pid_arms["$child_pid"]="$arm"
  pid_logs["$child_pid"]="$log_path"
}

launch_arm b 2 "$gpu2_uuid"
launch_arm d 3 "$gpu3_uuid"

parallel_status=0
while [[ "${#active_pids[@]}" -gt 0 ]]; do
  completed_pid=
  if wait -n -p completed_pid "${active_pids[@]}"; then
    completed_status=0
  else
    completed_status=$?
  fi
  remaining=()
  for child_pid in "${active_pids[@]}"; do
    if [[ "$child_pid" != "$completed_pid" ]]; then
      remaining+=("$child_pid")
    fi
  done
  active_pids=("${remaining[@]}")
  if [[ "$completed_status" -ne 0 ]]; then
    parallel_status="$completed_status"
    echo "seed42 arm ${pid_arms[$completed_pid]} sweep process failed" >&2
    sed "s/^/[seed42-${pid_arms[$completed_pid]}] /" \
      "${pid_logs[$completed_pid]}" >&2 || true
    for child_pid in "${active_pids[@]}"; do
      kill "$child_pid" 2>/dev/null || true
    done
    for child_pid in "${active_pids[@]}"; do
      wait "$child_pid" 2>/dev/null || true
    done
    active_pids=()
  fi
done

for arm in b d; do
  sed "s/^/[seed42-$arm] /" "$temporary_root/arm_${arm}.log"
done
if [[ "$parallel_status" -ne 0 ]]; then
  exit "$parallel_status"
fi

CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" --finalize-manifest
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" --analyze
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" --gate
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" --finalize-closure
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" --verify-closure

trap - INT TERM EXIT
rm -rf -- "$temporary_root"
