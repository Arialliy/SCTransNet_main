#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_runner="$formal_repo/experiments/run_tpd_formal800_4x5090_decision.sh"
formal_runner_sha256="a78a4b4e05c400157849302e3b5d865c5e67a4994254c83b7e0675bdf82a0101"
formal_decider="$formal_repo/experiments/decide_tpd_mainline_4x5090.py"
formal_decider_sha256="c7fbca1a57783c887391b343983d34985013e751e50fa7439d08c15550ad1393"
formal_unit="sctransnet-formal800-4x5090-decision"

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$formal_repo"
for formal_path in "$formal_runner" "$formal_decider"; do
    [[ -f "$formal_path" && ! -L "$formal_path" ]] || {
        echo "FORMAL4X5090_DECISION_LAUNCH_ABORT reason=missing_or_symlink path=$formal_path" >&2
        exit 1
    }
done
formal_actual_runner_sha256="$(sha256sum "$formal_runner" | awk '{print $1}')"
formal_actual_decider_sha256="$(sha256sum "$formal_decider" | awk '{print $1}')"
if [[ "$formal_actual_runner_sha256" != "$formal_runner_sha256" ]]; then
    echo "FORMAL4X5090_DECISION_LAUNCH_ABORT reason=runner_sha_mismatch expected=$formal_runner_sha256 actual=$formal_actual_runner_sha256" >&2
    exit 1
fi
if [[ "$formal_actual_decider_sha256" != "$formal_decider_sha256" ]]; then
    echo "FORMAL4X5090_DECISION_LAUNCH_ABORT reason=decider_sha_mismatch expected=$formal_decider_sha256 actual=$formal_actual_decider_sha256" >&2
    exit 1
fi

"/usr/bin/bash" "$formal_runner" --preflight
echo "FORMAL4X5090_DECISION_LAUNCH_PREFLIGHT_OK runner_sha256=$formal_actual_runner_sha256 decider_sha256=$formal_actual_decider_sha256"
if [[ "$formal_mode" == "--preflight" ]]; then
    exit 0
fi

if systemctl --user cat "$formal_unit.service" >/dev/null 2>&1; then
    echo "FORMAL4X5090_DECISION_LAUNCH_ABORT reason=unit_already_exists unit=$formal_unit.service" >&2
    exit 1
fi

systemd-run --user \
    --unit="$formal_unit" \
    --description="SCTransNet formal800 4xRTX5090 sealed seed42 mainline decision" \
    --property=Restart=no \
    --property=TimeoutStopSec=300 \
    /usr/bin/bash "$formal_runner"
echo "FORMAL4X5090_DECISION_UNIT_STARTED unit=$formal_unit.service"
