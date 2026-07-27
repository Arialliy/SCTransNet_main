#!/usr/bin/env bash
set -euo pipefail

v6_repo="/home/ly/SCTransNet_main"
v6_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v6_root="$v6_repo/experiments/results/tpd_clean_v6_formal800_2x5090_v1"
v6_comparison="$v6_root/NUDT-SIRST/comparison"
v6_lock="$v6_root/.postprocess.lock"

cd "$v6_repo"

v6_ready="$(
    "$v6_python" - <<'PY'
from experiments.summarize_tpd_clean_v6_formal800 import inspect_training_readiness
print("true" if inspect_training_readiness()["formal_matrix_complete"] else "false")
PY
)"
if [[ "$v6_ready" != "true" ]]; then
    echo "TPDCLEANV6_FINALIZER_RETRY reason=formal_matrix_incomplete"
    exit 75
fi

mkdir -p "$v6_comparison"
exec 9>"$v6_lock"
if ! flock -n 9; then
    echo "TPDCLEANV6_FINALIZER_RETRY reason=postprocess_lock_held"
    exit 75
fi

"$v6_python" experiments/run_tpd_clean_v6_formal800_sweeps.py \
    --run \
    --device cuda:0 \
    --physical-gpu 2

v6_json="$v6_comparison/tpd_clean_v6_formal800_comparison.json"
v6_markdown="$v6_comparison/tpd_clean_v6_formal800_comparison.md"
if [[ ! -e "$v6_json" && ! -L "$v6_json" && ! -e "$v6_markdown" && ! -L "$v6_markdown" ]]; then
    "$v6_python" experiments/summarize_tpd_clean_v6_formal800.py --write
elif [[ ! -f "$v6_json" || -L "$v6_json" || ! -f "$v6_markdown" || -L "$v6_markdown" ]]; then
    echo "TPDCLEANV6_FINALIZER_ABORT reason=partial_or_nonregular_report" >&2
    exit 1
fi

v6_manifest="$v6_comparison/completion_inputs.json"
v6_marker="$v6_comparison/COMPLETE.sha256"
if [[ ! -e "$v6_manifest" && ! -L "$v6_manifest" && ! -e "$v6_marker" && ! -L "$v6_marker" ]]; then
    "$v6_python" experiments/validate_tpd_clean_v6_formal800_completion.py publish
elif [[ ! -f "$v6_manifest" || -L "$v6_manifest" || ! -f "$v6_marker" || -L "$v6_marker" ]]; then
    echo "TPDCLEANV6_FINALIZER_ABORT reason=partial_or_nonregular_completion" >&2
    exit 1
fi

"$v6_python" experiments/validate_tpd_clean_v6_formal800_completion.py verify
echo "TPDCLEANV6_FINALIZER_COMPLETE physical_gpu=2 gpu_uuid=GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
