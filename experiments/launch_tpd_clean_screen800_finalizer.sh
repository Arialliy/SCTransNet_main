#!/usr/bin/env bash
set -euo pipefail

clean_repo="/home/ly/SCTransNet_main"
clean_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
clean_runner="$clean_repo/experiments/run_tpd_clean_screen800_finalizer.sh"
clean_finalizer="$clean_repo/experiments/finalize_tpd_clean_screen800.py"
clean_unit="sctransnet-tpd-clean-screen800-finalizer.service"
clean_mode="${1:-run}"

case "$clean_mode" in
    run)
        ;;
    --dry-run)
        exec "$clean_python" "$clean_finalizer" --dry-run
        ;;
    --status)
        exec "$clean_python" "$clean_finalizer" --status
        ;;
    *)
        echo "usage: $0 [--dry-run|--status]" >&2
        exit 2
        ;;
esac

[[ -x "$clean_runner" ]] || {
    echo "TPDCLEAN_FINALIZER_LAUNCH_FAILED reason=runner_not_executable path=$clean_runner" >&2
    exit 1
}
[[ -x "$clean_python" ]] || {
    echo "TPDCLEAN_FINALIZER_LAUNCH_FAILED reason=python_unavailable path=$clean_python" >&2
    exit 1
}

if systemctl --user is-active --quiet "$clean_unit"; then
    echo "TPDCLEAN_FINALIZER_ALREADY_ACTIVE unit=$clean_unit"
    exit 0
fi

if systemctl --user cat "$clean_unit" >/dev/null 2>&1; then
    systemctl --user reset-failed "$clean_unit" >/dev/null 2>&1 || true
    systemctl --user start "$clean_unit"
    echo "TPDCLEAN_FINALIZER_STARTED existing_unit=true unit=$clean_unit"
    exit 0
fi

systemd-run --user \
    --unit="${clean_unit%.service}" \
    --description="SCTransNet TPD-Clean screen800 automatic finalizer" \
    --property=Restart=no \
    --property=TimeoutStopSec=30 \
    /usr/bin/bash "$clean_runner"
echo "TPDCLEAN_FINALIZER_STARTED existing_unit=false unit=$clean_unit"
