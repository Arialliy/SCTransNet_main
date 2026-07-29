#!/usr/bin/env bash
set -euo pipefail

# A lightweight outer watchdog for the paired formal800 launcher.  The paired
# launcher remains the sole owner of initialization/resume decisions and of
# the authoritative launcher lock.  While another launcher owns that lock,
# this process exits with EX_TEMPFAIL so a service manager can retry later.

qfg_watch_repo="${TPD_NER_V4_QFG_V2_CROA_REPO:-/home/ly/SCTransNet_main}"
qfg_watch_result_root="${TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT:-$qfg_watch_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized}"
qfg_watch_launcher="${TPD_NER_V4_QFG_V2_CROA_TRAIN_LAUNCHER:-$qfg_watch_repo/experiments/run_tpd_ner_v4_qfg_v2_croa_formal800_2x5090_lane.sh}"
qfg_watch_lock_dir="$qfg_watch_result_root/.launcher_locks"
qfg_watch_claim="$qfg_watch_lock_dir/formal800_seed42_paired.lock"

[[ -d "$qfg_watch_repo" && ! -L "$qfg_watch_repo" ]] || {
    echo "QFG2X_WATCH_ABORT reason=invalid_repo path=$qfg_watch_repo" >&2
    exit 64
}
[[ -f "$qfg_watch_launcher" && ! -L "$qfg_watch_launcher" && -x "$qfg_watch_launcher" ]] || {
    echo "QFG2X_WATCH_ABORT reason=invalid_launcher path=$qfg_watch_launcher" >&2
    exit 64
}
if [[ -L "$qfg_watch_result_root" || ( -e "$qfg_watch_result_root" && ! -d "$qfg_watch_result_root" ) ]]; then
    echo "QFG2X_WATCH_ABORT reason=invalid_result_root path=$qfg_watch_result_root" >&2
    exit 64
fi

mkdir -p "$qfg_watch_lock_dir"
[[ ! -L "$qfg_watch_claim" ]] || {
    echo "QFG2X_WATCH_ABORT reason=invalid_claim_file path=$qfg_watch_claim" >&2
    exit 64
}

exec 9>"$qfg_watch_claim"
if ! flock -n 9; then
    echo "QFG2X_WATCH_WAIT reason=paired_launcher_active path=$qfg_watch_claim"
    exit 75
fi
flock -u 9
exec 9>&-

echo "QFG2X_WATCH_RESTART launcher=$qfg_watch_launcher"
exec "$qfg_watch_launcher" --freeze verify
