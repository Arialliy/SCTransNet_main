#!/usr/bin/env bash
set -euo pipefail

real_python=${FINAL_MODEL_SEED42_METRICSFIX_REAL_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
frozen_module=experiments.final_model_seed42_certification_replay_posttraining
successor_module=experiments.final_model_seed42_certification_replay_posttraining_metricsfix_v4

if [[ ! -x "$real_python" ]]; then
  echo "metricsfix real Python is not executable: $real_python" >&2
  exit 1
fi
if [[ "$(readlink -f -- "$real_python")" == "$(readlink -f -- "$0")" ]]; then
  echo "metricsfix real Python must not resolve to the redirector itself" >&2
  exit 1
fi

if [[ "$#" -ge 2 && "$1" == "-m" && "$2" == "$frozen_module" ]]; then
  shift 2
  exec "$real_python" -m "$successor_module" "$@"
fi

exec "$real_python" "$@"
