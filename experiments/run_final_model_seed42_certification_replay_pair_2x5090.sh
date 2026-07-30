#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/ly/SCTransNet_main
python_bin=/home/ly/BasicIRSTD/infrarenet/bin/python
contract="$repo_root/experiments/final_model_seed42_certification_replay_contract_v2.json"
manifest_root="$repo_root/experiments/final_model_seed42_certification_replay_manifests_v2"
source_lock="$repo_root/experiments/final_model_seed42_certification_replay_source_lock_v4.json"
upstream_source_lock="$repo_root/experiments/final_model_certification_source_lock_v1.json"
parent_lock="$repo_root/experiments/final_model_certification_parent_lock_v1.json"
output_root="$repo_root/experiments/results/final_model_seed42_certification_replay_v1"
log_root="$output_root/logs"
gpu2_uuid=GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
gpu3_uuid=GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3

dry_run=false
if [[ "$#" -eq 1 && "$1" == "--dry-run" ]]; then
  dry_run=true
elif [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

cd "$repo_root"
"$python_bin" -m experiments.final_model_seed42_certification_replay_contract \
  --contract "$contract" \
  --manifest-directory "$manifest_root" \
  --verify-only
"$python_bin" \
  -m experiments.freeze_final_model_seed42_certification_replay_source_lock \
  --output "$source_lock" \
  --verify

common=(
  --replay-contract "$contract"
  --certification-source-lock "$upstream_source_lock"
  --certification-parent-lock "$parent_lock"
  --replay-source-lock "$source_lock"
)
b_manifest="$manifest_root/seed_42_b_certification_replay_init.json"
d_manifest="$manifest_root/seed_42_d_certification_replay_init.json"

if [[ "$dry_run" == true ]]; then
  "$python_bin" -m experiments.train_final_model_seed42_certification_replay_b_exact \
    "${common[@]}" \
    --child-initialization-manifest "$b_manifest" \
    --dry-run-contract
  "$python_bin" -m experiments.train_final_model_seed42_certification_replay_d_exact \
    "${common[@]}" \
    --child-initialization-manifest "$d_manifest" \
    --dry-run-contract
  exit 0
fi

mkdir -p "$log_root"
exec 9>"$output_root/.gpu23_seed42_certification_replay.lock"
if ! flock -n 9; then
  echo "another seed-42 certification replay pair is active" >&2
  exit 1
fi

b_mode=$(
  "$python_bin" -m experiments.train_final_model_seed42_certification_replay_b_exact \
    "${common[@]}" \
    --child-initialization-manifest "$b_manifest" \
    --resolve-mode
)
d_mode=$(
  "$python_bin" -m experiments.train_final_model_seed42_certification_replay_d_exact \
    "${common[@]}" \
    --child-initialization-manifest "$d_manifest" \
    --resolve-mode
)

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
PYTHONHASHSEED=42 \
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
"$python_bin" -m experiments.train_final_model_seed42_certification_replay_b_exact \
  "${common[@]}" \
  --child-initialization-manifest "$b_manifest" \
  "$b_mode" \
  >>"$log_root/seed_42_b.log" 2>&1 &
pids+=("$!")
fi

if [[ "$d_mode" != "--complete" ]]; then
CUDA_VISIBLE_DEVICES="$gpu3_uuid" \
TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX=3 \
TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$gpu3_uuid" \
PYTHONHASHSEED=42 \
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
"$python_bin" -m experiments.train_final_model_seed42_certification_replay_d_exact \
  "${common[@]}" \
  --child-initialization-manifest "$d_manifest" \
  "$d_mode" \
  >>"$log_root/seed_42_d.log" 2>&1 &
pids+=("$!")
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
