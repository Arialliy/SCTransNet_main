#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_script="$formal_repo/experiments/run_tpd_formal800_4x5090_postprocess.sh"
formal_expected_sha256="fb6f341c4e2af373b5d0d03ba5525e82dd07570d597a98101481edfc9fe6835d"
formal_unit="sctransnet-formal800-4x5090-postprocess"

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$formal_repo"
[[ -x "$formal_script" && ! -L "$formal_script" ]] || {
    echo "FORMAL4X5090_POSTPROCESS_LAUNCH_ABORT reason=script_missing_or_not_executable path=$formal_script" >&2
    exit 1
}
formal_actual_sha256="$(sha256sum "$formal_script" | awk '{print $1}')"
if [[ "$formal_actual_sha256" != "$formal_expected_sha256" ]]; then
    echo "FORMAL4X5090_POSTPROCESS_LAUNCH_ABORT reason=script_sha_mismatch expected=$formal_expected_sha256 actual=$formal_actual_sha256" >&2
    exit 1
fi

"$formal_script" --preflight
echo "FORMAL4X5090_POSTPROCESS_LAUNCH_PREFLIGHT_OK script_sha256=$formal_actual_sha256"
if [[ "$formal_mode" == "--preflight" ]]; then
    exit 0
fi

if systemctl --user cat "$formal_unit.service" >/dev/null 2>&1; then
    echo "FORMAL4X5090_POSTPROCESS_LAUNCH_ABORT reason=unit_already_exists unit=$formal_unit.service" >&2
    exit 1
fi

systemd-run --user \
    --unit="$formal_unit" \
    --description="SCTransNet formal800 4xRTX5090 completion audit and Pd-Fa postprocess" \
    --property=Restart=no \
    --property=TimeoutStopSec=300 \
    /usr/bin/bash "$formal_script"
echo "FORMAL4X5090_POSTPROCESS_UNIT_STARTED unit=$formal_unit.service"
