#!/usr/bin/env bash
set -euo pipefail

repo_root=${FINAL_MODEL_SEED42_GATEFIX_REPO_ROOT:-/home/ly/SCTransNet_main}
real_python=${FINAL_MODEL_SEED42_GATEFIX_REAL_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
redirector="$repo_root/experiments/final_model_seed42_certification_python_gatefix_v5.sh"
envfix_wrapper="$repo_root/experiments/run_final_model_seed42_certification_completion_envfix_v2.sh"
gatefix_lock=${FINAL_MODEL_SEED42_GATEFIX_SOURCE_LOCK:-"$repo_root/experiments/final_model_seed42_certification_completion_gatefix_source_lock_v5.json"}
metricsfix_lock="$repo_root/experiments/final_model_seed42_certification_completion_metricsfix_source_lock_v4.json"
gatefix_lock_module=experiments.freeze_final_model_seed42_certification_completion_gatefix_source_lock
metricsfix_attestation_module=experiments.final_model_seed42_certification_metricsfix_attestation_gatefix_v5
gatefix_attestation_module=experiments.final_model_seed42_certification_completion_gatefix_attestation_v5
metricsfix_attestation="$repo_root/experiments/results/final_model_seed42_certification_replay_v1/final_model_seed42_certification_metricsfix_attestation_v4.json"
gatefix_attestation="$repo_root/experiments/results/final_model_seed42_certification_replay_v1/final_model_seed42_certification_gatefix_attestation_v5.json"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export FINAL_MODEL_SEED42_GATEFIX_REAL_PYTHON="$real_python"
export FINAL_MODEL_SEED42_ENVFIX_REPO_ROOT="$repo_root"
export FINAL_MODEL_SEED42_ENVFIX_PYTHON="$redirector"

cd "$repo_root"

"$real_python" -m "$gatefix_lock_module" \
  --verify \
  --require-runtime-env \
  --output "$gatefix_lock"

"$envfix_wrapper" "$@"

if [[ "${1:-}" == "--dry-run" ]]; then
  "$real_python" -m "$metricsfix_attestation_module" \
    --dry-run \
    --source-lock "$metricsfix_lock" \
    --output "$metricsfix_attestation"
  "$real_python" -m "$gatefix_attestation_module" \
    --dry-run \
    --source-lock "$gatefix_lock" \
    --metricsfix-attestation "$metricsfix_attestation" \
    --output "$gatefix_attestation"
  exit 0
fi

"$real_python" -m "$metricsfix_attestation_module" \
  --write-once \
  --source-lock "$metricsfix_lock" \
  --output "$metricsfix_attestation"
"$real_python" -m "$metricsfix_attestation_module" \
  --verify \
  --source-lock "$metricsfix_lock" \
  --output "$metricsfix_attestation"
"$real_python" -m "$gatefix_attestation_module" \
  --write-once \
  --source-lock "$gatefix_lock" \
  --metricsfix-attestation "$metricsfix_attestation" \
  --output "$gatefix_attestation"
"$real_python" -m "$gatefix_attestation_module" \
  --verify \
  --source-lock "$gatefix_lock" \
  --metricsfix-attestation "$metricsfix_attestation" \
  --output "$gatefix_attestation"
