#!/usr/bin/env bash
set -euo pipefail

real_python=${FINAL_MODEL_SEED42_GATEFIX_REAL_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
frozen_posttraining_module=experiments.final_model_seed42_certification_replay_posttraining
successor_posttraining_module=experiments.final_model_seed42_certification_replay_posttraining_gatefix_v5
frozen_completion_module=experiments.final_model_seed42_certification_completion
successor_completion_module=experiments.final_model_seed42_certification_completion_gatefix_v5

if [[ ! -x "$real_python" ]]; then
  echo "gatefix real Python is not executable: $real_python" >&2
  exit 1
fi
if [[ "$(readlink -f -- "$real_python")" == "$(readlink -f -- "$0")" ]]; then
  echo "gatefix real Python must not resolve to the redirector itself" >&2
  exit 1
fi

if [[ "$#" -ge 2 && "$1" == "-m" && "$2" == "$frozen_posttraining_module" ]]; then
  shift 2
  exec "$real_python" -m "$successor_posttraining_module" "$@"
fi
if [[ "$#" -ge 2 && "$1" == "-m" && "$2" == "$frozen_completion_module" ]]; then
  shift 2
  exec "$real_python" -m "$successor_completion_module" "$@"
fi

exec "$real_python" "$@"
