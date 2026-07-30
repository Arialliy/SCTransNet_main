#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/ly/SCTransNet_main
python_bin=/home/ly/BasicIRSTD/infrarenet/bin/python
source_lock_path="$repo_root/experiments/final_model_certification_source_lock_v1.json"
seed_contract_path="$repo_root/experiments/final_model_replication_seed_contract.json"
manifest_root="$repo_root/experiments/final_model_replication_manifests_v1"
output_root="$repo_root/experiments/results/final_model_engineering_replication_v1"
log_root="$output_root/logs"
gpu2_uuid=GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
gpu3_uuid=GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 TRAJECTORY_SEED" >&2
  exit 2
fi
trajectory_seed="$1"
if [[ "$trajectory_seed" != "3407" && "$trajectory_seed" != "426780603" ]]; then
  echo "trajectory seed is not in the engineering schedule" >&2
  exit 2
fi

cd "$repo_root"
mkdir -p "$log_root"
exec 9>"$output_root/.gpu23_engineering_replication.lock"
if ! flock -n 9; then
  echo "another GPU2/3 engineering replication pair is active" >&2
  exit 1
fi

b_manifest="$manifest_root/seed_${trajectory_seed}_b_child_init.json"
d_manifest="$manifest_root/seed_${trajectory_seed}_d_child_init.json"

mode_for_run() {
  local arm="$1"
  "$python_bin" -m experiments.watch_final_model_engineering_replication \
    --source-lock "$source_lock_path" \
    --output-root "$output_root" \
    --resolve-mode \
    --trajectory-seed "$trajectory_seed" \
    --arm "$arm"
}

b_mode="$(mode_for_run b)"
d_mode="$(mode_for_run d)"

b_pid=
d_pid=
pids=()
cleanup_children() {
  local child_pid
  for child_pid in "${pids[@]}"; do
    if kill -0 "$child_pid" 2>/dev/null; then
      kill "$child_pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_children INT TERM EXIT
if [[ "$b_mode" != "--complete" ]]; then
CUDA_VISIBLE_DEVICES="$gpu2_uuid" \
TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_INDEX=2 \
TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_UUID="$gpu2_uuid" \
PYTHONHASHSEED="$trajectory_seed" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
BLIS_NUM_THREADS=1 \
TORCH_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
"$python_bin" -m experiments.train_final_model_replication_b_exact \
  --trajectory-seed "$trajectory_seed" \
  --seed-contract "$seed_contract_path" \
  --child-initialization-manifest "$b_manifest" \
  --certification-source-lock "$source_lock_path" \
  "$b_mode" \
  --device cuda:0 \
  --output-root "$output_root" \
  >>"$log_root/seed_${trajectory_seed}_b.log" 2>&1 &
b_pid=$!
pids+=("$b_pid")
fi

if [[ "$d_mode" != "--complete" ]]; then
CUDA_VISIBLE_DEVICES="$gpu3_uuid" \
TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX=3 \
TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$gpu3_uuid" \
PYTHONHASHSEED="$trajectory_seed" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
BLIS_NUM_THREADS=1 \
TORCH_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
"$python_bin" -m experiments.train_final_model_replication_d_exact \
  --trajectory-seed "$trajectory_seed" \
  --seed-contract "$seed_contract_path" \
  --child-initialization-manifest "$d_manifest" \
  --certification-source-lock "$source_lock_path" \
  "$d_mode" \
  --device cuda:0 \
  --output-root "$output_root" \
  >>"$log_root/seed_${trajectory_seed}_d.log" 2>&1 &
d_pid=$!
pids+=("$d_pid")
fi

pair_status=0
while [[ "${#pids[@]}" -gt 0 ]]; do
  completed_pid=
  if wait -n -p completed_pid "${pids[@]}"; then
    completed_status=0
  else
    completed_status=$?
  fi
  remaining=()
  for active_pid in "${pids[@]}"; do
    if [[ "$active_pid" != "$completed_pid" ]]; then
      remaining+=("$active_pid")
    fi
  done
  pids=("${remaining[@]}")
  if [[ "$completed_status" -ne 0 ]]; then
    pair_status="$completed_status"
    cleanup_children
    for active_pid in "${pids[@]}"; do
      wait "$active_pid" 2>/dev/null || true
    done
    pids=()
  fi
done
trap - INT TERM EXIT
exit "$pair_status"
