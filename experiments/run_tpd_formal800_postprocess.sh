#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_python="/home/ly/MSHNet/.venv/bin/python"
formal_root="$formal_repo/experiments/results/tpd_pe_formal800_v1"
formal_dataset_root="$formal_root/NUDT-SIRST"
formal_run_name="seed_42_formal800_pd_fp32_v1"
formal_expected_epochs=800
formal_poll_seconds=30
formal_gpu1_unit="sctransnet-formal800-gpu1-s42-v1.service"
formal_gpu2_unit="sctransnet-formal800-gpu2-s42-v1.service"
formal_gpu1_uuid="GPU-48b3f9d5-25d4-2398-5483-ee6bd406b655"
formal_gpu2_uuid="GPU-3509f974-0eba-2bcb-9469-372003ae4f0a"
formal_lock="$formal_root/.postprocess_s42_v1.lock"
formal_state_file="$formal_root/postprocess_state_s42_v1.json"

cd "$formal_repo"

exec 9>"$formal_lock"
if ! flock -n 9; then
    echo "POSTPROCESS_ABORT reason=lock_held lock=$formal_lock" >&2
    exit 1
fi

formal_run_dir() {
    local formal_variant="$1"
    printf '%s/%s/%s\n' "$formal_dataset_root" "$formal_variant" "$formal_run_name"
}

formal_file_sha256() {
    local formal_path="$1"
    sha256sum "$formal_path" | awk '{print $1}'
}

formal_verify_gpu_mapping() {
    local formal_index="$1"
    local formal_expected_uuid="$2"
    local formal_actual_uuid
    formal_actual_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | \
        awk -F', *' -v formal_index="$formal_index" '$1 == formal_index {print $2}')"
    if [[ "$formal_actual_uuid" != "$formal_expected_uuid" ]]; then
        echo "POSTPROCESS_ABORT reason=gpu_mapping_changed index=$formal_index expected_uuid=$formal_expected_uuid actual_uuid=$formal_actual_uuid" >&2
        return 1
    fi
}

formal_sweep_output() {
    local formal_variant="$1"
    printf '%s/pd_fa_sweep_best.pth.json\n' "$(formal_run_dir "$formal_variant")"
}

formal_sweep_valid() {
    local formal_physical_gpu="$1"
    local formal_variant="$2"
    local formal_dir
    local formal_output
    local formal_checkpoint
    local formal_checkpoint_sha
    local formal_evaluator_sha
    local formal_recorded
    local formal_artifact
    local formal_artifact_path
    formal_dir="$(formal_run_dir "$formal_variant")"
    formal_output="$(formal_sweep_output "$formal_variant")"
    formal_checkpoint="$formal_dir/best.pth.tar"
    [[ -f "$formal_output" && -f "$formal_checkpoint" ]] || return 1
    formal_checkpoint_sha="$(formal_file_sha256 "$formal_checkpoint")"
    formal_evaluator_sha="$(formal_file_sha256 "$formal_repo/experiments/evaluate_pd_fa_sweep.py")"

    jq -e \
        --arg formal_variant "$formal_variant" \
        --arg formal_dir "$formal_dir" \
        --arg formal_checkpoint "$formal_checkpoint" \
        --arg formal_checkpoint_sha "$formal_checkpoint_sha" \
        --arg formal_evaluator_sha "$formal_evaluator_sha" \
        --arg formal_gpu "$formal_physical_gpu" \
        --argjson formal_epochs "$formal_expected_epochs" '
        .variant == $formal_variant and
        .dataset == "NUDT-SIRST" and
        .run_directory == $formal_dir and
        .checkpoint == $formal_checkpoint and
        .checkpoint_role == "best_validation_pd_primary" and
        .checkpoint_sha256 == $formal_checkpoint_sha and
        .official_test_accessed == false and
        .audit.expected_epochs == $formal_epochs and
        .audit.metrics_event_count == $formal_epochs and
        .audit.metrics_epoch_range == [1, $formal_epochs] and
        .audit.summary_status == "complete" and
        .audit.selection_source == "internal_validation_only" and
        .audit.cuda_visible_devices == $formal_gpu and
        .audit.artifact_sha256.checkpoint == $formal_checkpoint_sha and
        .audit.artifact_sha256.evaluator == $formal_evaluator_sha and
        ((.audit.integrity_checks_passed | to_entries | map(.value == true)) | all) and
        .threshold_configuration.threshold_min == 0.01 and
        .threshold_configuration.threshold_max == 0.99 and
        .threshold_configuration.threshold_step == 0.01 and
        .threshold_configuration.extra_thresholds == [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999] and
        .threshold_configuration.tail_logit_step == 0.1 and
        .threshold_configuration.fa_budgets == [0.000001, 0.000005, 0.00001, 0.00005, 0.0001] and
        .fixed_threshold_0_5.threshold == 0.5 and
        (.points | type == "array" and length > 0)
        ' "$formal_output" >/dev/null || return 1

    for formal_artifact in protocol.json split.json summary.json metrics.jsonl; do
        formal_artifact_path="$formal_dir/$formal_artifact"
        [[ -f "$formal_artifact_path" ]] || return 1
        formal_recorded="$(jq -r --arg formal_key "$formal_artifact" \
            '.audit.artifact_sha256[$formal_key] // empty' "$formal_output")"
        [[ "$formal_recorded" == "$(formal_file_sha256 "$formal_artifact_path")" ]] || return 1
    done
}

formal_validate_sweep_set() {
    local formal_reference=""
    local formal_current
    local formal_variant
    local formal_gpu
    for formal_variant in original progressive tpd spd; do
        case "$formal_variant" in
            original|progressive) formal_gpu=1 ;;
            tpd|spd) formal_gpu=2 ;;
        esac
        formal_sweep_valid "$formal_gpu" "$formal_variant" || return 1
        formal_current="$(jq -cS '{dataset,seed,split_seed,validation_split_sha256,match_radius,tiny_area,threshold_configuration,evaluator_sha256:.audit.artifact_sha256.evaluator}' \
            "$(formal_sweep_output "$formal_variant")")"
        if [[ -z "$formal_reference" ]]; then
            formal_reference="$formal_current"
        elif [[ "$formal_current" != "$formal_reference" ]]; then
            echo "POSTPROCESS_ABORT reason=sweep_set_mismatch variant=$formal_variant" >&2
            return 1
        fi
    done
}

formal_summary_payload_valid() {
    local formal_dir="$1"
    [[ -d "$formal_dir" && -f "$formal_dir/COMPLETE.sha256" ]] || return 1
    (
        cd "$formal_dir"
        sha256sum -c COMPLETE.sha256 >/dev/null
        sha256sum -c SWEEPS.sha256 >/dev/null
    ) || return 1
    jq -e --argjson formal_epochs "$formal_expected_epochs" '
        .expected_epochs == $formal_epochs and
        .official_test_accessed == false and
        (.rows | type == "array" and length == 4) and
        ([.rows[].variant] | sort) == ["original", "progressive", "spd", "tpd"]
        ' "$formal_dir/$formal_run_name.json" >/dev/null
}

formal_comparison_complete() {
    formal_summary_payload_valid "$formal_dataset_root/comparison"
}

formal_variant_complete() {
    local formal_variant="$1"
    local formal_dir
    local formal_summary
    local formal_metrics
    local formal_rows
    formal_dir="$(formal_run_dir "$formal_variant")"
    formal_summary="$formal_dir/summary.json"
    formal_metrics="$formal_dir/metrics.jsonl"
    [[ -f "$formal_summary" && -f "$formal_metrics" ]] || return 1
    jq -e '.status == "complete" and .official_test_accessed == false' \
        "$formal_summary" >/dev/null || return 1
    formal_rows="$(wc -l < "$formal_metrics")"
    [[ "$formal_rows" -eq "$formal_expected_epochs" ]]
}

formal_snapshot_value() {
    local formal_snapshot="$1"
    local formal_key="$2"
    sed -n "s/^${formal_key}=//p" <<<"$formal_snapshot" | head -n 1
}

formal_capture_invocation() {
    local formal_unit="$1"
    local formal_snapshot
    local formal_load
    local formal_active
    local formal_invocation
    formal_snapshot="$(systemctl --user show "$formal_unit" \
        --property=LoadState,ActiveState,InvocationID 2>/dev/null || true)"
    formal_load="$(formal_snapshot_value "$formal_snapshot" LoadState)"
    formal_active="$(formal_snapshot_value "$formal_snapshot" ActiveState)"
    formal_invocation="$(formal_snapshot_value "$formal_snapshot" InvocationID)"
    if [[ "$formal_load" != "loaded" || "$formal_active" != "active" || ! "$formal_invocation" =~ ^[0-9a-f]{32}$ ]]; then
        echo "POSTPROCESS_ABORT reason=cannot_capture_training_invocation unit=$formal_unit load=$formal_load active=$formal_active invocation=$formal_invocation" >&2
        return 1
    fi
    printf '%s\n' "$formal_invocation"
}

formal_queue_journal_complete() {
    local formal_invocation="$1"
    shift
    local formal_messages
    local formal_variant
    formal_messages="$(journalctl --user "_SYSTEMD_INVOCATION_ID=$formal_invocation" \
        --no-pager --output=cat 2>/dev/null || true)"
    for formal_variant in "$@"; do
        grep -Fqx "QUEUE_COMPLETE variant=$formal_variant" <<<"$formal_messages" || return 1
    done
}

formal_queue_state() {
    local formal_unit="$1"
    shift
    local formal_expected_invocation="$1"
    shift
    local formal_snapshot
    local formal_load
    local formal_active
    local formal_result
    local formal_code
    local formal_status
    local formal_invocation
    local formal_artifacts_ready=0
    local formal_variant

    formal_snapshot="$(systemctl --user show "$formal_unit" \
        --property=LoadState,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus,InvocationID \
        2>/dev/null || true)"
    formal_load="$(formal_snapshot_value "$formal_snapshot" LoadState)"
    formal_active="$(formal_snapshot_value "$formal_snapshot" ActiveState)"
    formal_result="$(formal_snapshot_value "$formal_snapshot" Result)"
    formal_code="$(formal_snapshot_value "$formal_snapshot" ExecMainCode)"
    formal_status="$(formal_snapshot_value "$formal_snapshot" ExecMainStatus)"
    formal_invocation="$(formal_snapshot_value "$formal_snapshot" InvocationID)"

    formal_artifacts_ready=1
    for formal_variant in "$@"; do
        if ! formal_variant_complete "$formal_variant"; then
            formal_artifacts_ready=0
        fi
    done

    if [[ "$formal_load" == "not-found" ]]; then
        if [[ "$formal_artifacts_ready" -eq 1 ]] && \
            formal_queue_journal_complete "$formal_expected_invocation" "$@"; then
            return 0
        fi
        echo "POSTPROCESS_ABORT reason=training_unit_gone_without_proof unit=$formal_unit expected_invocation=$formal_expected_invocation artifacts_ready=$formal_artifacts_ready" >&2
        return 2
    fi

    if [[ "$formal_load" != "loaded" || "$formal_invocation" != "$formal_expected_invocation" ]]; then
        echo "POSTPROCESS_ABORT reason=training_unit_identity_changed unit=$formal_unit load=$formal_load expected_invocation=$formal_expected_invocation actual_invocation=$formal_invocation" >&2
        return 2
    fi

    case "$formal_active" in
        active|activating|deactivating|reloading)
            return 1
            ;;
        failed)
            echo "POSTPROCESS_ABORT reason=training_unit_failed unit=$formal_unit result=$formal_result status=$formal_status" >&2
            return 2
            ;;
        inactive)
            if [[ "$formal_result" == "success" && "$formal_code" == "1" && \
                "$formal_status" == "0" && "$formal_artifacts_ready" -eq 1 ]] && \
                formal_queue_journal_complete "$formal_expected_invocation" "$@"; then
                return 0
            fi
            echo "POSTPROCESS_ABORT reason=inactive_training_incomplete unit=$formal_unit result=$formal_result code=$formal_code status=$formal_status artifacts_ready=$formal_artifacts_ready" >&2
            return 2
            ;;
        *)
            echo "POSTPROCESS_ABORT reason=unexpected_training_state unit=$formal_unit load=$formal_load active=$formal_active result=$formal_result code=$formal_code status=$formal_status" >&2
            return 2
            ;;
    esac
}

formal_verify_gpu_mapping 1 "$formal_gpu1_uuid"
formal_verify_gpu_mapping 2 "$formal_gpu2_uuid"

if [[ -e "$formal_dataset_root/comparison" ]]; then
    if formal_validate_sweep_set && formal_comparison_complete; then
        echo "POSTPROCESS_ALREADY_COMPLETE comparison=$formal_dataset_root/comparison/$formal_run_name.[json|md|csv]"
        exit 0
    fi
    echo "POSTPROCESS_ABORT reason=existing_comparison_not_valid path=$formal_dataset_root/comparison" >&2
    exit 1
fi

formal_postprocess_sha="$(formal_file_sha256 "$formal_repo/experiments/run_tpd_formal800_postprocess.sh")"
if [[ -e "$formal_state_file" ]]; then
    jq -e \
        --arg formal_gpu1_unit "$formal_gpu1_unit" \
        --arg formal_gpu2_unit "$formal_gpu2_unit" \
        --arg formal_gpu1_uuid "$formal_gpu1_uuid" \
        --arg formal_gpu2_uuid "$formal_gpu2_uuid" \
        --arg formal_postprocess_sha "$formal_postprocess_sha" '
        .schema_version == 1 and
        .gpu1.unit == $formal_gpu1_unit and
        .gpu2.unit == $formal_gpu2_unit and
        .gpu1.uuid == $formal_gpu1_uuid and
        .gpu2.uuid == $formal_gpu2_uuid and
        (.gpu1.invocation_id | type == "string" and test("^[0-9a-f]{32}$")) and
        (.gpu2.invocation_id | type == "string" and test("^[0-9a-f]{32}$")) and
        .postprocess_script_sha256 == $formal_postprocess_sha
        ' "$formal_state_file" >/dev/null || {
            echo "POSTPROCESS_ABORT reason=invalid_persistent_state path=$formal_state_file" >&2
            exit 1
        }
    formal_gpu1_invocation="$(jq -r '.gpu1.invocation_id' "$formal_state_file")"
    formal_gpu2_invocation="$(jq -r '.gpu2.invocation_id' "$formal_state_file")"
    echo "POSTPROCESS_STATE_RESTORED path=$formal_state_file"
else
    formal_gpu1_invocation="$(formal_capture_invocation "$formal_gpu1_unit")"
    formal_gpu2_invocation="$(formal_capture_invocation "$formal_gpu2_unit")"
    formal_state_tmp="$(mktemp "$formal_root/.postprocess_state_s42_v1.tmp.XXXXXX")"
    jq -n \
        --arg formal_gpu1_unit "$formal_gpu1_unit" \
        --arg formal_gpu2_unit "$formal_gpu2_unit" \
        --arg formal_gpu1_uuid "$formal_gpu1_uuid" \
        --arg formal_gpu2_uuid "$formal_gpu2_uuid" \
        --arg formal_gpu1_invocation "$formal_gpu1_invocation" \
        --arg formal_gpu2_invocation "$formal_gpu2_invocation" \
        --arg formal_postprocess_sha "$formal_postprocess_sha" \
        --arg formal_created_at "$(date --iso-8601=seconds)" '
        {
          schema_version: 1,
          created_at: $formal_created_at,
          postprocess_script_sha256: $formal_postprocess_sha,
          gpu1: {unit: $formal_gpu1_unit, uuid: $formal_gpu1_uuid, invocation_id: $formal_gpu1_invocation},
          gpu2: {unit: $formal_gpu2_unit, uuid: $formal_gpu2_uuid, invocation_id: $formal_gpu2_invocation}
        }
        ' > "$formal_state_tmp"
    mv -- "$formal_state_tmp" "$formal_state_file"
    echo "POSTPROCESS_STATE_CREATED path=$formal_state_file"
fi
echo "POSTPROCESS_ARMED gpu1_invocation=$formal_gpu1_invocation gpu2_invocation=$formal_gpu2_invocation"

formal_wait_count=0
while true; do
    formal_gpu1_done=0
    formal_gpu2_done=0

    if formal_queue_state "$formal_gpu1_unit" "$formal_gpu1_invocation" original progressive; then
        formal_gpu1_done=1
    else
        formal_state_status=$?
        [[ "$formal_state_status" -eq 1 ]] || exit "$formal_state_status"
    fi

    if formal_queue_state "$formal_gpu2_unit" "$formal_gpu2_invocation" tpd spd; then
        formal_gpu2_done=1
    else
        formal_state_status=$?
        [[ "$formal_state_status" -eq 1 ]] || exit "$formal_state_status"
    fi

    if [[ "$formal_gpu1_done" -eq 1 && "$formal_gpu2_done" -eq 1 ]]; then
        break
    fi

    if (( formal_wait_count % 20 == 0 )); then
        echo "POSTPROCESS_WAIT gpu1_done=$formal_gpu1_done gpu2_done=$formal_gpu2_done poll_seconds=$formal_poll_seconds"
    fi
    formal_wait_count=$((formal_wait_count + 1))
    sleep "$formal_poll_seconds"
done

formal_comparison_dir="$formal_dataset_root/comparison"

formal_run_sweep() {
    local formal_physical_gpu="$1"
    local formal_variant="$2"
    local formal_dir
    local formal_output
    formal_dir="$(formal_run_dir "$formal_variant")"
    formal_output="$(formal_sweep_output "$formal_variant")"
    if [[ -e "$formal_output" ]]; then
        if formal_sweep_valid "$formal_physical_gpu" "$formal_variant"; then
            echo "SWEEP_SKIP_VALID variant=$formal_variant physical_cuda=$formal_physical_gpu output=$formal_output"
            return 0
        fi
        echo "POSTPROCESS_ABORT reason=existing_sweep_invalid variant=$formal_variant output=$formal_output" >&2
        return 1
    fi
    case "$formal_physical_gpu" in
        1) formal_verify_gpu_mapping 1 "$formal_gpu1_uuid" ;;
        2) formal_verify_gpu_mapping 2 "$formal_gpu2_uuid" ;;
        *) echo "POSTPROCESS_ABORT reason=invalid_physical_gpu gpu=$formal_physical_gpu" >&2; return 1 ;;
    esac
    echo "SWEEP_START variant=$formal_variant physical_cuda=$formal_physical_gpu"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$formal_physical_gpu" \
        "$formal_python" experiments/evaluate_pd_fa_sweep.py \
        --run-dir "$formal_dir" \
        --checkpoint best.pth.tar \
        --device cuda:0 \
        --expected-epochs "$formal_expected_epochs" \
        --threshold-min 0.01 \
        --threshold-max 0.99 \
        --threshold-step 0.01 \
        --extra-thresholds 0.001 0.005 0.995 0.999 0.9995 0.9999 \
        --tail-logit-step 0.1 \
        --fa-budgets 1e-6 5e-6 1e-5 5e-5 1e-4
    echo "SWEEP_COMPLETE variant=$formal_variant physical_cuda=$formal_physical_gpu"
}

echo "POSTPROCESS_START root=$formal_root"
(
    formal_run_sweep 1 original
    formal_run_sweep 1 progressive
) &
formal_gpu1_sweep_pid=$!
(
    formal_run_sweep 2 tpd
    formal_run_sweep 2 spd
) &
formal_gpu2_sweep_pid=$!

set +e
wait "$formal_gpu1_sweep_pid"
formal_gpu1_sweep_status=$?
wait "$formal_gpu2_sweep_pid"
formal_gpu2_sweep_status=$?
set -e

if [[ "$formal_gpu1_sweep_status" -ne 0 || "$formal_gpu2_sweep_status" -ne 0 ]]; then
    echo "POSTPROCESS_ABORT reason=sweep_failed gpu1_status=$formal_gpu1_sweep_status gpu2_status=$formal_gpu2_sweep_status" >&2
    exit 1
fi

formal_validate_sweep_set

formal_stage_dir="$(mktemp -d "$formal_dataset_root/.comparison_${formal_run_name}.staging.XXXXXX")"

"$formal_python" experiments/summarize_tpd_pilot.py \
    --root "$formal_root" \
    --dataset NUDT-SIRST \
    --run-name "$formal_run_name" \
    --expected-epochs "$formal_expected_epochs" \
    --report-title "NUDT-SIRST TPD-PE formal 800-epoch comparison" \
    --output-dir "$formal_stage_dir"

[[ -s "$formal_stage_dir/$formal_run_name.json" ]]
[[ -s "$formal_stage_dir/$formal_run_name.md" ]]
[[ -s "$formal_stage_dir/$formal_run_name.csv" ]]

formal_validate_sweep_set

for formal_variant in original progressive tpd spd; do
    formal_sweep_path="$(formal_sweep_output "$formal_variant")"
    printf '%s  %s\n' "$(formal_file_sha256 "$formal_sweep_path")" "$formal_sweep_path"
done > "$formal_stage_dir/SWEEPS.sha256"

(
    cd "$formal_stage_dir"
    sha256sum \
        "$formal_run_name.json" \
        "$formal_run_name.md" \
        "$formal_run_name.csv" \
        SWEEPS.sha256 > COMPLETE.sha256
    sha256sum -c COMPLETE.sha256 >/dev/null
    sha256sum -c SWEEPS.sha256 >/dev/null
)

formal_summary_payload_valid "$formal_stage_dir"

if [[ -e "$formal_comparison_dir" ]]; then
    echo "POSTPROCESS_ABORT reason=comparison_appeared_during_run path=$formal_comparison_dir" >&2
    exit 1
fi
mv -- "$formal_stage_dir" "$formal_comparison_dir"
formal_comparison_complete

echo "POSTPROCESS_COMPLETE comparison=$formal_comparison_dir/$formal_run_name.[json|md|csv]"
