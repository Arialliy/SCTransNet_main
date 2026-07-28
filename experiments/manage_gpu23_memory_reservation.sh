#!/usr/bin/env bash
set -euo pipefail

gpu23_action="${1:-}"
if [[ -n "$gpu23_action" ]]; then
    shift
fi

gpu23_physical_index=""
gpu23_reserve_mib=""
gpu23_min_free_mib=""
gpu23_poll_seconds="1"
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --physical-gpu)
            [[ -z "$gpu23_physical_index" && "$#" -ge 2 ]] || {
                echo "provide --physical-gpu exactly once" >&2
                exit 2
            }
            gpu23_physical_index="$2"
            shift 2
            ;;
        --reserve-mib)
            [[ -z "$gpu23_reserve_mib" && "$#" -ge 2 ]] || {
                echo "provide --reserve-mib at most once" >&2
                exit 2
            }
            gpu23_reserve_mib="$2"
            shift 2
            ;;
        --min-free-mib)
            [[ -z "$gpu23_min_free_mib" && "$#" -ge 2 ]] || {
                echo "provide --min-free-mib at most once" >&2
                exit 2
            }
            gpu23_min_free_mib="$2"
            shift 2
            ;;
        --poll-seconds)
            [[ "$#" -ge 2 ]] || {
                echo "provide a value after --poll-seconds" >&2
                exit 2
            }
            gpu23_poll_seconds="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

gpu23_usage() {
    echo "usage: $0 {preflight|direct-start|status|release} --physical-gpu {2|3} [--reserve-mib MiB] [--min-free-mib MiB] [--poll-seconds SEC]" >&2
}

if [[ "$gpu23_action" != "preflight" &&
      "$gpu23_action" != "direct-start" && "$gpu23_action" != "status" &&
      "$gpu23_action" != "release" ]]; then
    gpu23_usage
    exit 2
fi
if [[ "$gpu23_physical_index" != "2" &&
      "$gpu23_physical_index" != "3" ]]; then
    gpu23_usage
    exit 2
fi
for gpu23_value in \
    "$gpu23_reserve_mib" \
    "$gpu23_min_free_mib" \
    "$gpu23_poll_seconds"; do
    if [[ -n "$gpu23_value" &&
          ( ! "$gpu23_value" =~ ^[0-9]+$ || "$gpu23_value" -lt 1 ) ]]; then
        echo "memory and polling values must be positive integers" >&2
        exit 2
    fi
done
if [[ "$gpu23_action" == "status" || "$gpu23_action" == "release" ]]; then
    if [[ -n "$gpu23_reserve_mib" || -n "$gpu23_min_free_mib" ]]; then
        echo "status/release do not accept memory sizing arguments" >&2
        exit 2
    fi
fi

gpu23_repo="${SCTransNet_GPU23_RESERVATION_REPO:-/home/ly/SCTransNet_main}"
gpu23_python_fixed="/home/ly/BasicIRSTD/infrarenet/bin/python"
gpu23_python_expected_real="/usr/bin/python3.12"
gpu23_python="${SCTransNet_GPU23_RESERVATION_PYTHON:-$gpu23_python_fixed}"
gpu23_guard="$gpu23_repo/experiments/gpu23_memory_reservation_guard.py"
gpu23_state_root="${SCTransNet_GPU23_RESERVATION_STATE_ROOT:-$gpu23_repo/experiments/runtime/gpu23_memory_reservation}"
gpu23_unit="sctransnet-gpu-memory-reservation-gpu${gpu23_physical_index}"

[[ -d "$gpu23_repo" && ! -L "$gpu23_repo" ]] || {
    echo "GPU23_RESERVATION_ABORT reason=invalid_repo path=$gpu23_repo" >&2
    exit 1
}
if [[ "$gpu23_python" != "$gpu23_python_fixed" ]]; then
    echo "GPU23_RESERVATION_ABORT reason=unexpected_python_path path=$gpu23_python expected=$gpu23_python_fixed" >&2
    exit 1
fi
if [[ ! -x "$gpu23_python" ]]; then
    echo "GPU23_RESERVATION_ABORT reason=python_not_executable path=$gpu23_python" >&2
    exit 1
fi
if ! gpu23_python_real="$(readlink -f -- "$gpu23_python")"; then
    echo "GPU23_RESERVATION_ABORT reason=python_resolution_failed path=$gpu23_python" >&2
    exit 1
fi
case "$gpu23_python_real" in
    "$gpu23_python_expected_real")
        ;;
    *)
        echo "GPU23_RESERVATION_ABORT reason=python_target_outside_expected_path path=$gpu23_python real=$gpu23_python_real expected=$gpu23_python_expected_real" >&2
        exit 1
        ;;
esac
if [[ ! -f "$gpu23_python_real" || -L "$gpu23_python_real" || ! -x "$gpu23_python_real" ]]; then
    echo "GPU23_RESERVATION_ABORT reason=invalid_resolved_python path=$gpu23_python real=$gpu23_python_real" >&2
    exit 1
fi
[[ -f "$gpu23_guard" && ! -L "$gpu23_guard" ]] || {
    echo "GPU23_RESERVATION_ABORT reason=invalid_guard path=$gpu23_guard" >&2
    exit 1
}

gpu23_common=(
    --physical-gpu "$gpu23_physical_index"
    --state-root "$gpu23_state_root"
)
gpu23_sizing=()
if [[ -n "$gpu23_reserve_mib" ]]; then
    gpu23_sizing+=(--reserve-mib "$gpu23_reserve_mib")
fi
if [[ -n "$gpu23_min_free_mib" ]]; then
    gpu23_sizing+=(--min-free-mib "$gpu23_min_free_mib")
fi

case "$gpu23_action" in
    preflight)
        "$gpu23_python" "$gpu23_guard" preflight \
            "${gpu23_common[@]}" \
            "${gpu23_sizing[@]}"
        ;;
    status)
        systemctl --user show "$gpu23_unit.service" \
            --property=Id,LoadState,ActiveState,SubState,Result,ExecMainStatus,NRestarts \
            2>/dev/null || true
        "$gpu23_python" "$gpu23_guard" status "${gpu23_common[@]}"
        ;;
    release)
        systemctl --user stop "$gpu23_unit.service" 2>/dev/null || true
        "$gpu23_python" "$gpu23_guard" status "${gpu23_common[@]}"
        ;;
    direct-start)
        "$gpu23_python" "$gpu23_guard" preflight \
            "${gpu23_common[@]}" \
            "${gpu23_sizing[@]}"
        if systemctl --user is-active --quiet "$gpu23_unit.service"; then
            echo "GPU23_RESERVATION_ABORT reason=unit_already_active unit=$gpu23_unit" >&2
            exit 1
        fi
        systemctl --user reset-failed "$gpu23_unit.service" \
            >/dev/null 2>&1 || true
        systemd-run --user \
            --unit "$gpu23_unit" \
            --collect \
            --property=Type=exec \
            --property=Restart=no \
            "$gpu23_python" \
            "$gpu23_guard" \
            hold \
            "${gpu23_common[@]}" \
            "${gpu23_sizing[@]}" \
            --poll-seconds "$gpu23_poll_seconds"
        echo "GPU23_RESERVATION_DIRECT_LAUNCHED physical_gpu=$gpu23_physical_index unit=$gpu23_unit restart=no"
        ;;
esac

