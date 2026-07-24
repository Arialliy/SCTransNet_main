#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
formal_watchdog="$formal_repo/experiments/watch_tpd_formal800_4x5090.py"
formal_expected_sha256="c1dae28855dddb2f3d765b4165a8f013f2a0fc5b68ade44772686260564bd7e7"
formal_unit="sctransnet-formal800-4x5090-watchdog"

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$formal_repo"
[[ -x "$formal_watchdog" && ! -L "$formal_watchdog" ]] || {
    echo "FORMAL4X5090_WATCHDOG_LAUNCH_ABORT reason=watchdog_missing path=$formal_watchdog" >&2
    exit 1
}
formal_actual_sha256="$(sha256sum "$formal_watchdog" | awk '{print $1}')"
if [[ "$formal_actual_sha256" != "$formal_expected_sha256" ]]; then
    echo "FORMAL4X5090_WATCHDOG_LAUNCH_ABORT reason=sha_mismatch expected=$formal_expected_sha256 actual=$formal_actual_sha256" >&2
    exit 1
fi

"$formal_python" "$formal_watchdog" \
    --once \
    --poll-seconds 300 \
    --stale-seconds 3600
echo "FORMAL4X5090_WATCHDOG_LAUNCH_PREFLIGHT_OK script_sha256=$formal_actual_sha256"
if [[ "$formal_mode" == "--preflight" ]]; then
    exit 0
fi

if systemctl --user cat "$formal_unit.service" >/dev/null 2>&1; then
    echo "FORMAL4X5090_WATCHDOG_LAUNCH_ABORT reason=unit_already_exists unit=$formal_unit.service" >&2
    exit 1
fi

systemd-run --user \
    --unit="$formal_unit" \
    --description="SCTransNet formal800 4xRTX5090 live health watchdog" \
    --property=Restart=no \
    --property=TimeoutStopSec=30 \
    "$formal_python" "$formal_watchdog" \
    --poll-seconds 300 \
    --stale-seconds 3600
echo "FORMAL4X5090_WATCHDOG_UNIT_STARTED unit=$formal_unit.service"
