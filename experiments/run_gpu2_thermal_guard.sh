#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_root="$formal_repo/experiments/results/tpd_pe_formal800_v1"
formal_gpu_uuid="GPU-3509f974-0eba-2bcb-9469-372003ae4f0a"
formal_recovery_unit="sctransnet-formal800-gpu2-recovery-s42-v1.service"
formal_postprocess_unit="sctransnet-formal800-gpu2-recovery-postprocess-s42-v1.service"
formal_pause_temp="${FORMAL_PAUSE_TEMP_C:-82}"
formal_emergency_temp="${FORMAL_EMERGENCY_TEMP_C:-87}"
formal_resume_temp="${FORMAL_RESUME_TEMP_C:-75}"
formal_sample_seconds="${FORMAL_SAMPLE_SECONDS:-5}"
formal_high_samples="${FORMAL_HIGH_SAMPLES:-2}"
formal_cool_samples="${FORMAL_COOL_SAMPLES:-3}"
formal_error_samples="${FORMAL_ERROR_SAMPLES:-3}"
formal_lock="$formal_root/.gpu2_thermal_guard.lock"
formal_events="$formal_root/GPU2_THERMAL_EVENTS.jsonl"

cd "$formal_repo"

exec 9>"$formal_lock"
if ! flock -n 9; then
    echo "GPU2_THERMAL_GUARD_ABORT reason=lock_held lock=$formal_lock" >&2
    exit 1
fi

formal_require_positive_integer() {
    local formal_name="$1"
    local formal_value="$2"
    if [[ ! "$formal_value" =~ ^[0-9]+$ ]] || (( formal_value < 1 )); then
        echo "GPU2_THERMAL_GUARD_ABORT reason=invalid_integer name=$formal_name value=$formal_value" >&2
        exit 2
    fi
}

for formal_pair in \
    "pause_temp:$formal_pause_temp" \
    "emergency_temp:$formal_emergency_temp" \
    "resume_temp:$formal_resume_temp" \
    "sample_seconds:$formal_sample_seconds" \
    "high_samples:$formal_high_samples" \
    "cool_samples:$formal_cool_samples" \
    "error_samples:$formal_error_samples"; do
    formal_require_positive_integer "${formal_pair%%:*}" "${formal_pair#*:}"
done

if (( formal_resume_temp >= formal_pause_temp || formal_pause_temp > formal_emergency_temp )); then
    echo "GPU2_THERMAL_GUARD_ABORT reason=invalid_hysteresis resume=$formal_resume_temp pause=$formal_pause_temp emergency=$formal_emergency_temp" >&2
    exit 2
fi

formal_log_event() {
    local formal_action="$1"
    local formal_unit="$2"
    local formal_temperature="$3"
    local formal_reason="$4"
    local formal_line
    formal_line="$(
        jq -cn \
            --arg formal_timestamp "$(date --iso-8601=seconds)" \
            --arg formal_action "$formal_action" \
            --arg formal_unit "$formal_unit" \
            --arg formal_gpu_uuid "$formal_gpu_uuid" \
            --arg formal_temperature "$formal_temperature" \
            --arg formal_reason "$formal_reason" \
            --argjson formal_pause_temp "$formal_pause_temp" \
            --argjson formal_emergency_temp "$formal_emergency_temp" \
            --argjson formal_resume_temp "$formal_resume_temp" '
            {
              timestamp: $formal_timestamp,
              action: $formal_action,
              unit: $formal_unit,
              gpu_uuid: $formal_gpu_uuid,
              temperature_c: (
                if $formal_temperature == "" then null
                else ($formal_temperature | tonumber)
                end
              ),
              reason: $formal_reason,
              thresholds_c: {
                pause: $formal_pause_temp,
                emergency: $formal_emergency_temp,
                resume: $formal_resume_temp
              }
            }
        '
    )"
    printf '%s\n' "$formal_line" >> "$formal_events"
    echo "GPU2_THERMAL_GUARD action=$formal_action unit=${formal_unit:-none} temp_c=${formal_temperature:-unknown} reason=$formal_reason"
}

formal_active_target() {
    if systemctl --user is-active --quiet "$formal_recovery_unit"; then
        printf '%s\n' "$formal_recovery_unit"
    elif systemctl --user is-active --quiet "$formal_postprocess_unit"; then
        printf '%s\n' "$formal_postprocess_unit"
    fi
}

formal_temperature() {
    local formal_value
    formal_value="$(
        timeout 10s nvidia-smi -i "$formal_gpu_uuid" \
            --query-gpu=temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null
    )" || return 1
    formal_value="${formal_value//[[:space:]]/}"
    [[ "$formal_value" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$formal_value"
}

formal_signal_unit() {
    local formal_signal="$1"
    local formal_unit="$2"
    systemctl --user kill \
        --kill-whom=all \
        --signal="$formal_signal" \
        "$formal_unit"
}

formal_paused_unit=""
formal_last_target=""
formal_high_count=0
formal_cool_count=0
formal_error_count=0

formal_cleanup() {
    if [[ -n "$formal_paused_unit" ]]; then
        formal_signal_unit SIGCONT "$formal_paused_unit" >/dev/null 2>&1 || true
        formal_log_event "resume_on_guard_exit" "$formal_paused_unit" "" "guard_exit"
        formal_paused_unit=""
    fi
}
trap formal_cleanup EXIT INT TERM

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

formal_initial_temp="$(formal_temperature)" || {
    echo "GPU2_THERMAL_GUARD_ABORT reason=temperature_probe_failed gpu_uuid=$formal_gpu_uuid" >&2
    exit 1
}

if [[ "$formal_mode" == "--preflight" ]]; then
    echo "GPU2_THERMAL_GUARD_PREFLIGHT gpu_uuid=$formal_gpu_uuid temp_c=$formal_initial_temp pause_c=$formal_pause_temp emergency_c=$formal_emergency_temp resume_c=$formal_resume_temp"
    exit 0
fi

formal_log_event "guard_start" "" "$formal_initial_temp" "configuration_armed"

while true; do
    formal_target="$(formal_active_target || true)"
    if [[ "$formal_target" != "$formal_last_target" ]]; then
        formal_log_event "target_change" "$formal_target" "" "active_unit_changed"
        formal_last_target="$formal_target"
        formal_high_count=0
        formal_error_count=0
    fi

    if ! formal_current_temp="$(formal_temperature)"; then
        formal_error_count=$((formal_error_count + 1))
        formal_high_count=0
        formal_cool_count=0
        if [[ -z "$formal_paused_unit" && -n "$formal_target" && "$formal_error_count" -ge "$formal_error_samples" ]]; then
            formal_signal_unit SIGSTOP "$formal_target"
            formal_paused_unit="$formal_target"
            formal_log_event "pause" "$formal_paused_unit" "" "temperature_probe_failed_${formal_error_count}_times"
        elif [[ "$formal_error_count" -eq 1 ]]; then
            formal_log_event "probe_error" "$formal_target" "" "temperature_probe_failed"
        fi
        sleep "$formal_sample_seconds"
        continue
    fi

    formal_error_count=0
    if [[ -n "$formal_paused_unit" ]]; then
        if (( formal_current_temp <= formal_resume_temp )); then
            formal_cool_count=$((formal_cool_count + 1))
        else
            formal_cool_count=0
        fi
        if (( formal_cool_count >= formal_cool_samples )); then
            formal_signal_unit SIGCONT "$formal_paused_unit"
            formal_log_event "resume" "$formal_paused_unit" "$formal_current_temp" "temperature_stable_below_resume_threshold"
            formal_paused_unit=""
            formal_cool_count=0
            formal_high_count=0
        fi
        sleep "$formal_sample_seconds"
        continue
    fi

    if [[ -z "$formal_target" ]]; then
        formal_high_count=0
        sleep "$formal_sample_seconds"
        continue
    fi

    if (( formal_current_temp >= formal_pause_temp )); then
        formal_high_count=$((formal_high_count + 1))
    else
        formal_high_count=0
    fi

    if (( formal_current_temp >= formal_emergency_temp || formal_high_count >= formal_high_samples )); then
        formal_reason="temperature_high_${formal_high_count}_samples"
        if (( formal_current_temp >= formal_emergency_temp )); then
            formal_reason="emergency_temperature"
        fi
        formal_signal_unit SIGSTOP "$formal_target"
        formal_paused_unit="$formal_target"
        formal_cool_count=0
        formal_log_event "pause" "$formal_paused_unit" "$formal_current_temp" "$formal_reason"
    fi

    sleep "$formal_sample_seconds"
done
