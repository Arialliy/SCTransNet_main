#!/usr/bin/env bash
set -euo pipefail

clean_repo="/home/ly/SCTransNet_main"
clean_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
clean_finalizer="$clean_repo/experiments/finalize_tpd_clean_screen800.py"
clean_result_root="$clean_repo/experiments/results/tpd_clean_screen800_4x5090_v1"
clean_log="$clean_result_root/logs/finalizer.log"
clean_poll_seconds="${TPDCLEAN_FINALIZER_POLL_SECONDS:-60}"

if [[ ! "$clean_poll_seconds" =~ ^[0-9]+$ ]] || (( clean_poll_seconds < 1 )); then
    echo "TPDCLEAN_FINALIZER_LAUNCH_FAILED reason=invalid_poll_seconds value=$clean_poll_seconds" >&2
    exit 2
fi
[[ -x "$clean_python" ]] || {
    echo "TPDCLEAN_FINALIZER_LAUNCH_FAILED reason=python_unavailable path=$clean_python" >&2
    exit 1
}
[[ -f "$clean_finalizer" && ! -L "$clean_finalizer" ]] || {
    echo "TPDCLEAN_FINALIZER_LAUNCH_FAILED reason=finalizer_unavailable path=$clean_finalizer" >&2
    exit 1
}

mkdir -p "$clean_result_root/logs" "$clean_result_root/.locks"
exec > >(tee -a "$clean_log") 2>&1
cd "$clean_repo"

echo "TPDCLEAN_FINALIZER_START poll_seconds=$clean_poll_seconds"
exec "$clean_python" "$clean_finalizer" --poll-seconds "$clean_poll_seconds"
