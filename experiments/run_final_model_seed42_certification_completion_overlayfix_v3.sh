#!/usr/bin/env bash
set -euo pipefail

repo_root=${FINAL_MODEL_SEED42_OVERLAYFIX_REPO_ROOT:-/home/ly/SCTransNet_main}
real_python=${FINAL_MODEL_SEED42_OVERLAYFIX_REAL_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
redirector="$repo_root/experiments/final_model_seed42_certification_python_overlayfix_v3.sh"
envfix_wrapper="$repo_root/experiments/run_final_model_seed42_certification_completion_envfix_v2.sh"
overlayfix_lock=${FINAL_MODEL_SEED42_OVERLAYFIX_SOURCE_LOCK:-"$repo_root/experiments/final_model_seed42_certification_completion_overlayfix_source_lock_v3.json"}
overlayfix_lock_module=experiments.freeze_final_model_seed42_certification_completion_overlayfix_source_lock
attestation_module=experiments.final_model_seed42_certification_completion_overlayfix_attestation_v3
attestation="$repo_root/experiments/results/final_model_seed42_certification_replay_v1/final_model_seed42_certification_overlayfix_attestation_v3.json"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export FINAL_MODEL_SEED42_OVERLAYFIX_REAL_PYTHON="$real_python"
export FINAL_MODEL_SEED42_ENVFIX_REPO_ROOT="$repo_root"
export FINAL_MODEL_SEED42_ENVFIX_PYTHON="$redirector"

cd "$repo_root"

"$real_python" -m "$overlayfix_lock_module" \
  --verify \
  --require-runtime-env \
  --output "$overlayfix_lock"

"$envfix_wrapper" "$@"

if [[ "${1:-}" == "--dry-run" ]]; then
  "$real_python" -m "$attestation_module" \
    --dry-run \
    --source-lock "$overlayfix_lock" \
    --output "$attestation"
  exit 0
fi

"$real_python" -m "$attestation_module" \
  --write-once \
  --source-lock "$overlayfix_lock" \
  --output "$attestation"
"$real_python" -m "$attestation_module" \
  --verify \
  --source-lock "$overlayfix_lock" \
  --output "$attestation"
