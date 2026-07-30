#!/usr/bin/env bash
set -euo pipefail

repo_root=${FINAL_MODEL_SEED42_COMPLETION_REPO_ROOT:-/home/ly/SCTransNet_main}
python_bin=${FINAL_MODEL_SEED42_COMPLETION_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
module=experiments.final_model_seed42_certification_completion
post_launcher="$repo_root/experiments/run_final_model_seed42_certification_replay_posttraining_2x5090.sh"
training_launcher="$repo_root/experiments/run_final_model_seed42_certification_replay_pair_2x5090.sh"
output_root="$repo_root/experiments/results/final_model_seed42_certification_replay_v1"
f1_output_dir="$repo_root/analysis/results/final_model_qfg_six_mode_audit_v1"
f1_report="$f1_output_dir/final_model_qfg_six_mode_audit_v1.json"
deep_output="$repo_root/analysis/results/final_model_qfg_six_mode_deep_verification_v1/final_model_qfg_six_mode_deep_verification_v1.json"
attestation="$output_root/final_model_seed42_certification_completion_attestation_v1.json"
training_pair_lock="$output_root/.gpu23_seed42_certification_replay.lock"
resume_record_root="$output_root/completion_watcher_resume_records_v1"
source_lock="$repo_root/experiments/final_model_certification_source_lock_v1.json"
parent_lock="$repo_root/experiments/final_model_certification_parent_lock_v1.json"

gpu2_uuid=GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
f1_module=analysis.run_final_qfg_six_mode_audit
deep_module=analysis.verify_final_qfg_six_mode_audit_deep

usage() {
  echo "usage: $0 (--dry-run | --run | --watch) [--poll-seconds N]" >&2
}

mode=
poll_seconds=30
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run|--run|--watch)
      if [[ -n "$mode" ]]; then
        usage
        exit 2
      fi
      mode="$1"
      shift
      ;;
    --poll-seconds)
      if [[ "$#" -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        usage
        exit 2
      fi
      poll_seconds="$2"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
if [[ -z "$mode" ]]; then
  usage
  exit 2
fi

cd "$repo_root"
common_completion_paths=(
  --f1-report "$f1_report"
  --deep-output "$deep_output"
  --output "$attestation"
)

# Dry-run is deliberately before mkdir/flock and invokes no CUDA assertion,
# evaluator, model loader, or GPU-facing command.
if [[ "$mode" == "--dry-run" ]]; then
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    "$python_bin" -m "$module" \
    --dry-run \
    "${common_completion_paths[@]}"
  printf '%s\n' \
    "DRY-RUN ONLY; no GPU command was launched." \
    "The new seed42 B/D summaries and exact journals must both prove epoch 800." \
    "Then GPU2/GPU3 run four replay sweeps; frozen deployment D F1 runs on GPU2; deep verification runs on CPU."
  exit 0
fi

mkdir -p "$output_root"
exec 9>"$output_root/.final_seed42_certification_completion.lock"
if ! flock -n 9; then
  echo "another final seed42 certification completion runner is active" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" \
  --verify-source-lock \
  "${common_completion_paths[@]}"

wait_for_training() {
  local resume_attempt=0
  local existing_attempt
  if [[ -L "$resume_record_root" ]]; then
    echo "resume-record directory must not be a symlink: $resume_record_root" >&2
    return 1
  fi
  if [[ -d "$resume_record_root" ]]; then
    for existing_attempt in 1 2 3; do
      existing_record="$resume_record_root/attempt_${existing_attempt}_preflight.json"
      if [[ -L "$existing_record" || ( -e "$existing_record" && ! -f "$existing_record" ) ]]; then
        echo "invalid resume-attempt record: $existing_record" >&2
        return 1
      elif [[ -f "$existing_record" ]]; then
        resume_attempt="$existing_attempt"
      fi
    done
  elif [[ -e "$resume_record_root" ]]; then
    echo "resume-record path is not a directory: $resume_record_root" >&2
    return 1
  fi
  while true; do
    set +e
    CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" \
      --training-ready \
      "${common_completion_paths[@]}"
    status=$?
    set -e
    case "$status" in
      0)
        return 0
        ;;
      3)
        if [[ "$mode" == "--run" ]]; then
          echo "new seed42 B/D formal800 replay is not complete; use --watch to wait" >&2
          return 3
        fi
        sleep "$poll_seconds"
        ;;
      4)
        if [[ "$mode" == "--run" ]]; then
          echo "new seed42 replay is incomplete and its pair launcher exited; use --watch for exact-resume recovery" >&2
          return 4
        fi
        resume_attempt=$((resume_attempt + 1))
        if [[ "$resume_attempt" -gt 3 ]]; then
          echo "seed42 exact-resume recovery exceeded three attempts" >&2
          return 1
        fi
        mkdir -p "$resume_record_root"
        resume_record="$resume_record_root/attempt_${resume_attempt}_preflight.json"
        if [[ -e "$resume_record" || -L "$resume_record" ]]; then
          echo "resume-attempt record already exists: $resume_record" >&2
          return 1
        fi
        set +e
        CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" \
          --training-ready \
          "${common_completion_paths[@]}" \
          >"$resume_record"
        record_status=$?
        set -e
        if [[ "$record_status" -ne 4 ]]; then
          echo "resume-attempt preflight changed unexpectedly" >&2
          return 1
        fi
        printf 'seed42 exact-resume recovery attempt %s/3 via frozen pair launcher\n' \
          "$resume_attempt"
        "$training_launcher"
        ;;
      *)
        echo "seed42 training completion validation failed" >&2
        return "$status"
        ;;
    esac
  done
}

wait_for_training

# Take and retain the replay pair's own lock before launching any evaluator.
# This both proves the pair launcher/children exited and closes the race where
# a second pair launcher could start after the read-only readiness check.
exec 8>"$training_pair_lock"
if [[ "$mode" == "--watch" ]]; then
  flock 8
elif ! flock -n 8; then
  echo "seed42 training pair still owns its GPU2/3 lock" >&2
  exit 3
fi
export FINAL_MODEL_SEED42_COMPLETION_TRAINING_LOCK_FD=8
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" \
  --training-ready \
  "${common_completion_paths[@]}"

# The posttraining launcher is itself locked, write-once, restartable, and
# terminates the peer sweep immediately if either parallel arm fails.
FINAL_MODEL_SEED42_POST_REPO_ROOT="$repo_root" \
FINAL_MODEL_SEED42_POST_PYTHON="$python_bin" \
  "$post_launcher" --run

CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" \
  --verify-posttraining \
  "${common_completion_paths[@]}"

# F1 audits the frozen deployment D, not the newly replayed D checkpoint.
# A complete existing artifact is live-verified and skipped; any partial or
# non-regular destination stops the closure.
if [[ -L "$f1_output_dir" || -L "$f1_report" ]]; then
  echo "F1 output must not be a symlink: $f1_output_dir" >&2
  exit 1
elif [[ -f "$f1_report" ]]; then
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$f1_module" \
    --verify \
    --repo-root "$repo_root" \
    --parent-lock "$parent_lock" \
    --source-lock "$source_lock" \
    --report "$f1_report"
elif [[ -e "$f1_output_dir" ]]; then
  echo "incomplete or invalid F1 output exists: $f1_output_dir" >&2
  exit 1
else
  CUDA_VISIBLE_DEVICES="$gpu2_uuid" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX=2 \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID="$gpu2_uuid" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$python_bin" -m "$module" \
    --assert-runtime-gpu2 \
    "${common_completion_paths[@]}"
  CUDA_VISIBLE_DEVICES="$gpu2_uuid" \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX=2 \
  FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID="$gpu2_uuid" \
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
    "$python_bin" -m "$f1_module" \
    --run \
    --repo-root "$repo_root" \
    --parent-lock "$parent_lock" \
    --source-lock "$source_lock" \
    --output-dir "$f1_output_dir" \
    --device cuda:0
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$f1_module" \
    --verify \
    --repo-root "$repo_root" \
    --parent-lock "$parent_lock" \
    --source-lock "$source_lock" \
    --report "$f1_report"
fi

# Deep verification is CPU-only and write-once.
if [[ -L "$deep_output" ]]; then
  echo "deep-verification output must not be a symlink: $deep_output" >&2
  exit 1
elif [[ -f "$deep_output" ]]; then
  CUDA_VISIBLE_DEVICES= "$python_bin" -m "$deep_module" \
    --verify-attestation \
    --repo-root "$repo_root" \
    --parent-lock "$parent_lock" \
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
    --parent-lock "$parent_lock" \
    --source-lock "$source_lock" \
    --report "$f1_report" \
    --output "$deep_output"
fi

CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" \
  --finalize-attestation \
  "${common_completion_paths[@]}"
CUDA_VISIBLE_DEVICES= "$python_bin" -m "$module" \
  --verify-attestation \
  "${common_completion_paths[@]}"

printf '%s\n' \
  "FINAL_SEED42_CERTIFICATION_COMPLETION_VERIFIED" \
  "attestation=$attestation" \
  "paper_core_established=false" \
  "stability_claim_supported=false"
