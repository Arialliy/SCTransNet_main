#!/usr/bin/env bash
set -euo pipefail

repo_root=${FINAL_MODEL_SEED42_ENVFIX_REPO_ROOT:-/home/ly/SCTransNet_main}
python_bin=${FINAL_MODEL_SEED42_ENVFIX_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
completion_shell="$repo_root/experiments/run_final_model_seed42_certification_completion.sh"
envfix_lock=${FINAL_MODEL_SEED42_ENVFIX_SOURCE_LOCK:-"$repo_root/experiments/final_model_seed42_certification_completion_envfix_source_lock_v2.json"}
envfix_module=experiments.freeze_final_model_seed42_certification_completion_envfix_source_lock

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export FINAL_MODEL_SEED42_COMPLETION_REPO_ROOT="$repo_root"
export FINAL_MODEL_SEED42_COMPLETION_PYTHON="$python_bin"

cd "$repo_root"

"$python_bin" -m "$envfix_module" \
  --verify \
  --require-runtime-env \
  --output "$envfix_lock"

exec "$completion_shell" "$@"

