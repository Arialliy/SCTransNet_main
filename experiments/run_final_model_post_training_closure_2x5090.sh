#!/usr/bin/env bash
set -euo pipefail

repo_root=${FINAL_MODEL_POST_REPO_ROOT:-/home/ly/SCTransNet_main}
python_bin=${FINAL_MODEL_POST_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
output_root=${FINAL_MODEL_POST_OUTPUT_ROOT:-"$repo_root/experiments/results/final_model_engineering_replication_v1"}
source_lock=${FINAL_MODEL_POST_SOURCE_LOCK:-"$repo_root/experiments/final_model_certification_source_lock_v1.json"}
seed_contract=${FINAL_MODEL_POST_SEED_CONTRACT:-"$repo_root/experiments/final_model_replication_seed_contract.json"}
manifest_directory=${FINAL_MODEL_POST_MANIFEST_DIRECTORY:-"$repo_root/experiments/final_model_replication_manifests_v1"}
summary_path=${FINAL_MODEL_POST_SUMMARY:-"$output_root/engineering_replication_summary_v1.json"}
evaluation_manifest="$output_root/engineering_checkpoint_local_pd_fa_manifest_v1.json"
paired_output=${FINAL_MODEL_POST_PAIRED_OUTPUT:-"$repo_root/analysis/results/final_model_engineering_paired_screen_v1.json"}
f1_output_dir=${FINAL_MODEL_POST_F1_OUTPUT_DIR:-"$repo_root/analysis/results/final_model_qfg_six_mode_audit_v1"}
f1_report="$f1_output_dir/final_model_qfg_six_mode_audit_v1.json"
deep_output=${FINAL_MODEL_POST_DEEP_OUTPUT:-"$repo_root/analysis/results/final_model_qfg_six_mode_deep_verification_v1/final_model_qfg_six_mode_deep_verification_v1.json"}
gate_output=${FINAL_MODEL_POST_GATE_OUTPUT:-"$output_root/engineering_gate_adjudication_v1.json"}
closure_output=${FINAL_MODEL_POST_CLOSURE_OUTPUT:-"$output_root/post_training_closure_attestation_v1.json"}

gpu2_uuid=GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
gpu3_uuid=GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
evaluator_module=experiments.evaluate_final_model_engineering_replication_pd_fa
summary_module=experiments.summarize_final_model_engineering_replication
paired_module=experiments.analyze_final_model_engineering_paired_screen
f1_module=analysis.run_final_qfg_six_mode_audit
deep_module=analysis.verify_final_qfg_six_mode_audit_deep
gate_module=experiments.adjudicate_final_model_engineering_gate
helper_module=experiments.final_model_post_training_closure_preflight

usage() {
  echo "usage: $0 (--dry-run | --run) [--f1-gpu 2|3]" >&2
}

execution_mode=
f1_gpu_index=2
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run|--run)
      if [[ -n "$execution_mode" ]]; then
        usage
        exit 2
      fi
      execution_mode="$1"
      shift
      ;;
    --f1-gpu)
      if [[ "$#" -lt 2 ]]; then
        usage
        exit 2
      fi
      f1_gpu_index="$2"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
if [[ -z "$execution_mode" ]]; then
  usage
  exit 2
fi
if [[ "$f1_gpu_index" != "2" && "$f1_gpu_index" != "3" ]]; then
  echo "F1 physical GPU must be 2 or 3" >&2
  exit 2
fi
if [[ "$f1_gpu_index" == "2" ]]; then
  f1_gpu_uuid="$gpu2_uuid"
else
  f1_gpu_uuid="$gpu3_uuid"
fi

common_paths=(
  --repo-root "$repo_root"
  --output-root "$output_root"
  --source-lock "$source_lock"
  --seed-contract "$seed_contract"
  --manifest-directory "$manifest_directory"
  --summary "$summary_path"
  --paired-output "$paired_output"
  --f1-report "$f1_report"
  --deep-output "$deep_output"
  --gate-output "$gate_output"
  --closure-output "$closure_output"
  --f1-gpu-index "$f1_gpu_index"
)
evaluation_paths=(
  --output-root "$output_root"
  --source-lock "$source_lock"
  --seed-contract "$seed_contract"
  --manifest-directory "$manifest_directory"
)

cd "$repo_root"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  "$python_bin" -m "$helper_module" \
  --preflight \
  "${common_paths[@]}"

if [[ "$execution_mode" == "--dry-run" ]]; then
  printf '%s\n' \
    "DRY-RUN ONLY; no GPU command was launched." \
    "1. Validate/write the four-run summary on CPU." \
    "2. In parallel: GPU2 UUID $gpu2_uuid -> arm B (4 sweeps)." \
    "3. In parallel: GPU3 UUID $gpu3_uuid -> arm D (4 sweeps)." \
    "4. Explicitly finalize and verify the eight-result manifest on CPU." \
    "5. Run or validate the engineering B/D paired screen on CPU." \
    "6. Run or validate F1 six-mode audit on physical GPU$f1_gpu_index UUID $f1_gpu_uuid." \
    "7. Run or validate the independent deep verifier on CPU." \
    "8. Run or validate engineering Gate S-E adjudication on CPU." \
    "9. Write or validate the closure attestation."
  exit 0
fi

mkdir -p "$output_root"
exec 9>"$output_root/.post_training_closure_gpu23.lock"
if ! flock -n 9; then
  echo "another GPU2/3 post-training closure is active" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES= "$python_bin" -m "$summary_module" \
  --output-root "$output_root" \
  --source-lock "$source_lock" \
  --seed-contract "$seed_contract" \
  --manifest-directory "$manifest_directory" \
  --output "$summary_path"

temporary_root=$(mktemp -d "$output_root/.post-training-closure.XXXXXX")
f1_staging_container=
declare -a active_pids=()
declare -A pid_logs=()
declare -A pid_arms=()

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
  if [[ -n "$f1_staging_container" && -d "$f1_staging_container" ]]; then
    rm -rf -- "$f1_staging_container"
  fi
}
trap cleanup INT TERM EXIT

launch_arm_sweeps() {
  local arm="$1"
  local physical_index="$2"
  local physical_uuid="$3"
  local log_path="$temporary_root/arm_${arm}.json"
  CUDA_VISIBLE_DEVICES="$physical_uuid" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX="$physical_index" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID="$physical_uuid" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONHASHSEED=0 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  TORCH_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  "$python_bin" -m "$evaluator_module" \
    --execute \
    "${evaluation_paths[@]}" \
    --arm "$arm" \
    --device cuda:0 \
    --physical-gpu-index "$physical_index" \
    --physical-gpu-uuid "$physical_uuid" \
    >"$log_path" 2>&1 &
  local child_pid=$!
  active_pids+=("$child_pid")
  pid_logs["$child_pid"]="$log_path"
  pid_arms["$child_pid"]="$arm"
}

launch_arm_sweeps b 2 "$gpu2_uuid"
launch_arm_sweeps d 3 "$gpu3_uuid"

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
    echo "arm ${pid_arms[$completed_pid]} sweep process failed" >&2
    sed "s/^/[arm-${pid_arms[$completed_pid]}] /" \
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
if [[ "$parallel_status" -ne 0 ]]; then
  exit "$parallel_status"
fi
for arm in b d; do
  sed "s/^/[arm-$arm] /" "$temporary_root/arm_${arm}.json"
done

CUDA_VISIBLE_DEVICES= "$python_bin" -m "$evaluator_module" \
  --finalize-manifest \
  "${evaluation_paths[@]}"
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$evaluator_module" \
  --verify-results \
  "${evaluation_paths[@]}"

if [[ -L "$paired_output" ]]; then
  echo "engineering paired-screen output must not be a symlink: $paired_output" >&2
  exit 1
elif [[ -f "$paired_output" ]]; then
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$helper_module" \
    --verify-paired-screen-output \
    "${common_paths[@]}"
elif [[ -e "$paired_output" ]]; then
  echo "engineering paired-screen output is not a regular file: $paired_output" >&2
  exit 1
else
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$paired_module" \
    --manifest "$evaluation_manifest" \
    --output "$paired_output"
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$helper_module" \
    --verify-paired-screen-output \
    "${common_paths[@]}"
fi

if [[ -L "$f1_output_dir" ]]; then
  echo "F1 output directory must not be a symlink: $f1_output_dir" >&2
  exit 1
fi
if [[ -f "$f1_report" && ! -L "$f1_report" ]]; then
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$f1_module" \
    --verify \
    --repo-root "$repo_root" \
    --source-lock "$source_lock" \
    --report "$f1_report"
elif [[ -e "$f1_output_dir" || -L "$f1_report" ]]; then
  echo "incomplete or invalid F1 output already exists: $f1_output_dir" >&2
  exit 1
else
  f1_parent=$(dirname -- "$f1_output_dir")
  f1_basename=$(basename -- "$f1_output_dir")
  mkdir -p "$f1_parent"
  f1_staging_container=$(
    mktemp -d "$f1_parent/.${f1_basename}.closure-staging.XXXXXX"
  )
  f1_staged_output_dir="$f1_staging_container/payload"
  f1_staged_report="$f1_staged_output_dir/final_model_qfg_six_mode_audit_v1.json"
  CUDA_VISIBLE_DEVICES="$f1_gpu_uuid" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX="$f1_gpu_index" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID="$f1_gpu_uuid" \
  "$python_bin" -m "$helper_module" \
    --assert-runtime-gpu \
    --physical-gpu-index "$f1_gpu_index"
  CUDA_VISIBLE_DEVICES="$f1_gpu_uuid" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX="$f1_gpu_index" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID="$f1_gpu_uuid" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  "$python_bin" -m "$f1_module" \
    --run \
    --repo-root "$repo_root" \
    --source-lock "$source_lock" \
    --output-dir "$f1_staged_output_dir" \
    --device cuda:0
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$f1_module" \
    --verify \
    --repo-root "$repo_root" \
    --source-lock "$source_lock" \
    --report "$f1_staged_report"
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$helper_module" \
    --publish-f1-staging \
    --f1-staging-dir "$f1_staged_output_dir" \
    "${common_paths[@]}"
  rmdir -- "$f1_staging_container"
  f1_staging_container=
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$f1_module" \
    --verify \
    --repo-root "$repo_root" \
    --source-lock "$source_lock" \
    --report "$f1_report"
fi

if [[ -L "$deep_output" ]]; then
  echo "deep-verification output must not be a symlink: $deep_output" >&2
  exit 1
elif [[ -f "$deep_output" ]]; then
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$deep_module" \
    --verify-attestation \
    --repo-root "$repo_root" \
    --source-lock "$source_lock" \
    --report "$f1_report" \
    --output "$deep_output"
elif [[ -e "$deep_output" ]]; then
  echo "deep-verification output is not a regular file: $deep_output" >&2
  exit 1
else
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$deep_module" \
    --write-once \
    --repo-root "$repo_root" \
    --source-lock "$source_lock" \
    --report "$f1_report" \
    --output "$deep_output"
fi

if [[ -L "$gate_output" ]]; then
  echo "engineering Gate output must not be a symlink: $gate_output" >&2
  exit 1
elif [[ -f "$gate_output" ]]; then
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$helper_module" \
    --verify-gate-output \
    "${common_paths[@]}"
elif [[ -e "$gate_output" ]]; then
  echo "engineering Gate output is not a regular file: $gate_output" >&2
  exit 1
else
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$gate_module" \
    --output-root "$output_root" \
    --source-lock "$source_lock" \
    --seed-contract "$seed_contract" \
    --manifest-directory "$manifest_directory" \
    --summary "$summary_path" \
    --output "$gate_output"
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$helper_module" \
    --verify-gate-output \
    "${common_paths[@]}"
fi

CUDA_VISIBLE_DEVICES= "$python_bin" -m "$helper_module" \
  --finalize-closure \
  "${common_paths[@]}"

trap - INT TERM EXIT
rm -rf -- "$temporary_root"
