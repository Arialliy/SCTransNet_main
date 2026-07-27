#!/usr/bin/env bash
set -euo pipefail

dch_repo="/home/ly/SCTransNet_main"
dch_runner="$dch_repo/experiments/run_tpd_clean_v7_dch_formal800_finalizer.sh"
dch_unit="sctransnet-tpd-clean-v7-dch-formal800-finalizer"

if [[ "${1:-}" == "--preflight" ]]; then
    shift
    if [[ "$#" -ne 0 ]]; then
        echo "usage: $0 [--preflight]" >&2
        exit 2
    fi
    /usr/bin/bash "$dch_runner" --preflight
    exit $?
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi
[[ -x "$dch_runner" ]] || {
    echo "TPDCLEANV7DCH_FINALIZER_LAUNCH_ABORT reason=runner_not_executable" >&2
    exit 64
}
if systemctl --user is-active --quiet "$dch_unit.service"; then
    echo "TPDCLEANV7DCH_FINALIZER_ALREADY_ACTIVE unit=$dch_unit.service"
    exit 0
fi

systemd-run --user \
    --collect \
    --unit="$dch_unit" \
    --description="SCTransNet TPD-Clean V7-DCH formal800 post-training finalizer" \
    --property=Restart=on-failure \
    --property=RestartPreventExitStatus=64 \
    --property=RestartSec=60 \
    --property=StartLimitIntervalSec=0 \
    --property=TimeoutStopSec=300 \
    /usr/bin/bash "$dch_runner"
echo "TPDCLEANV7DCH_FINALIZER_STARTED unit=$dch_unit.service sweep_gpus=2,3 audit_gpu=2"
