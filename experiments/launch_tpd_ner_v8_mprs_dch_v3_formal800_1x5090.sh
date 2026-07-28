#!/usr/bin/env bash
set -euo pipefail

v3_mode="launch"
v3_physical_index=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --preflight)
            v3_mode="preflight"
            shift
            ;;
        --status)
            v3_mode="status"
            shift
            ;;
        --physical-gpu)
            [[ "$#" -ge 2 ]] || {
                echo "missing value for --physical-gpu" >&2
                exit 2
            }
            v3_physical_index="$2"
            shift 2
            ;;
        *)
            echo "usage: $0 [--preflight|--status] --physical-gpu {2|3}" >&2
            exit 2
            ;;
    esac
done
if [[ "$v3_physical_index" != "2" && "$v3_physical_index" != "3" ]]; then
    echo "usage: $0 [--preflight|--status] --physical-gpu {2|3}" >&2
    exit 2
fi

v3_repo="${TPD_NER_V8_V3_REPO:-/home/ly/SCTransNet_main}"
v3_lane="$v3_repo/experiments/run_tpd_ner_v8_mprs_dch_v3_formal800_1x5090_lane.sh"
v3_manifest_tool="$v3_repo/experiments/freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py"
v3_python="${TPD_NER_V8_V3_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v3_training_lock="${TPD_NER_V8_V3_TRAINING_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v3_exact_source_lock.json}"
v3_acceptance_lock="${TPD_NER_V8_V3_ACCEPTANCE_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v3_acceptance_source_lock.json}"
v3_upstream_v2_training_lock="${TPD_NER_V8_V3_UPSTREAM_V2_TRAINING_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v2_exact_source_lock.json}"
v3_upstream_v2_acceptance_lock="${TPD_NER_V8_V3_UPSTREAM_V2_ACCEPTANCE_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v2_acceptance_source_lock.json}"
v3_result_root="${TPD_NER_V8_V3_RESULT_ROOT:-$v3_repo/experiments/results/tpd_ner_v8_mprs_dch_v3_exact_v1}"
v3_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v3_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v3_gpu2_unit="sctransnet-tpd-ner-v8-v3-relay-on-gpu2"
v3_gpu3_unit="sctransnet-tpd-ner-v8-v3-relay-on-gpu3"

if [[ "$v3_physical_index" == "2" ]]; then
    v3_gpu_uuid="$v3_gpu2_uuid"
    v3_selected_unit="$v3_gpu2_unit"
else
    v3_gpu_uuid="$v3_gpu3_uuid"
    v3_selected_unit="$v3_gpu3_unit"
fi

cd "$v3_repo"
[[ -x "$v3_lane" ]] || {
    echo "TPDNERV8V3_LAUNCH_ABORT reason=lane_not_executable path=$v3_lane" >&2
    exit 1
}
# The interpreter is allowed to be a symlink; only executability is required.
[[ -x "$v3_python" ]] || {
    echo "TPDNERV8V3_LAUNCH_ABORT reason=python_not_executable path=$v3_python" >&2
    exit 1
}
for v3_required_file in \
    "$v3_manifest_tool" \
    "$v3_training_lock" \
    "$v3_acceptance_lock" \
    "$v3_upstream_v2_training_lock" \
    "$v3_upstream_v2_acceptance_lock"
do
    [[ -f "$v3_required_file" && ! -L "$v3_required_file" ]] || {
        echo "TPDNERV8V3_LAUNCH_ABORT reason=missing_required_file path=$v3_required_file" >&2
        exit 1
    }
done

if [[ "$v3_mode" == "status" ]]; then
    systemctl --user show "$v3_selected_unit.service" \
        --property=Id,ActiveState,SubState,Result,ExecMainStatus,NRestarts
    exit 0
fi

if [[ -L "$v3_result_root" || ( -e "$v3_result_root" && ! -d "$v3_result_root" ) ]]; then
    echo "TPDNERV8V3_LAUNCH_ABORT reason=invalid_result_root path=$v3_result_root" >&2
    exit 1
fi
mkdir -p "$v3_result_root"
v3_claim_dir="$v3_result_root/.launcher_locks"
if [[ -L "$v3_claim_dir" || ( -e "$v3_claim_dir" && ! -d "$v3_claim_dir" ) ]]; then
    echo "TPDNERV8V3_LAUNCH_ABORT reason=invalid_claim_directory path=$v3_claim_dir" >&2
    exit 1
fi
mkdir -p "$v3_claim_dir"
v3_claim_path="$v3_claim_dir/formal800_v3_global.lock"
if [[ -L "$v3_claim_path" ]]; then
    echo "TPDNERV8V3_LAUNCH_ABORT reason=invalid_claim_file path=$v3_claim_path" >&2
    exit 1
fi
exec 9>"$v3_claim_path"
if ! flock -n 9; then
    echo "TPDNERV8V3_LAUNCH_ABORT reason=v3_global_claim_busy path=$v3_claim_path" >&2
    exit 1
fi

"$v3_python" "$v3_manifest_tool" \
    --mode verify \
    --kind all \
    --dataset-dir "$v3_repo/datasets" \
    --training-lock "$v3_training_lock" \
    --acceptance-lock "$v3_acceptance_lock" \
    --upstream-v2-training-lock "$v3_upstream_v2_training_lock" \
    --upstream-v2-acceptance-lock "$v3_upstream_v2_acceptance_lock"

v3_preflight_output="$(
    "$v3_lane" --preflight "$v3_physical_index" "$v3_gpu_uuid"
)"
printf '%s\n' "$v3_preflight_output"
v3_ready_prefix="TPDNERV8V3_LANE_READY variant=tpd_ner_v8_mprs_dch_v3_full_relay_on physical_gpu=$v3_physical_index initialization="
if [[ "$v3_preflight_output" != *"$v3_ready_prefix"* ]]; then
    echo "TPDNERV8V3_LAUNCH_ABORT reason=lane_preflight_identity_missing" >&2
    exit 1
fi

if [[ "$v3_mode" == "preflight" ]]; then
    echo "TPDNERV8V3_PREFLIGHT_COMPLETE variant=tpd_ner_v8_mprs_dch_v3_full_relay_on seed=42 physical_gpu=$v3_physical_index"
    exit 0
fi
if [[ "$v3_preflight_output" == *"$v3_ready_prefix""complete "* ]]; then
    echo "TPDNERV8V3_IDEMPOTENT_COMPLETE variant=tpd_ner_v8_mprs_dch_v3_full_relay_on seed=42 physical_gpu=$v3_physical_index"
    exit 0
fi

for v3_unit in "$v3_gpu2_unit" "$v3_gpu3_unit"; do
    v3_state="$(
        systemctl --user show \
            "$v3_unit.service" \
            --property=ActiveState \
            --value \
            2>/dev/null || true
    )"
    if [[ "$v3_state" == "active" || "$v3_state" == "activating" ]]; then
        echo "TPDNERV8V3_LAUNCH_ABORT reason=v3_unit_already_active unit=$v3_unit" >&2
        exit 1
    fi
done
systemctl --user reset-failed "$v3_selected_unit.service" >/dev/null 2>&1 || true

systemd-run --user \
    --unit "$v3_selected_unit" \
    --collect \
    --property=Type=exec \
    --property=Restart=on-failure \
    --property=RestartSec=10 \
    "$v3_lane" \
    "$v3_physical_index" \
    "$v3_gpu_uuid"

echo "TPDNERV8V3_LAUNCHED variant=tpd_ner_v8_mprs_dch_v3_full_relay_on seed=42 physical_gpu=$v3_physical_index"
