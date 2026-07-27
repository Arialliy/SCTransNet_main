#!/usr/bin/env bash
set -euo pipefail

v6_repo="/home/ly/SCTransNet_main"
v6_runner="$v6_repo/experiments/run_tpd_clean_v6_formal800_finalizer.sh"
v6_unit="sctransnet-tpd-clean-v6-formal800-finalizer"

if [[ "${1:-}" == "--preflight" ]]; then
    "$v6_repo/../BasicIRSTD/infrarenet/bin/python" \
        "$v6_repo/experiments/run_tpd_clean_v6_formal800_sweeps.py" \
        --preflight \
        --device cuda:0 \
        --physical-gpu 2
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi
if systemctl --user is-active --quiet "$v6_unit.service"; then
    echo "TPDCLEANV6_FINALIZER_ALREADY_ACTIVE unit=$v6_unit.service"
    exit 0
fi
systemd-run --user \
    --collect \
    --unit="$v6_unit" \
    --description="SCTransNet TPD-Clean-v6 formal800 postprocess finalizer" \
    --property=Restart=on-failure \
    --property=RestartSec=60 \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v6_runner"
echo "TPDCLEANV6_FINALIZER_STARTED unit=$v6_unit.service physical_gpu=2"
