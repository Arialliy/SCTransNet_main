#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_python="/home/ly/MSHNet/.venv/bin/python"
formal_root="$formal_repo/experiments/results/tpd_pe_formal800_v1"
formal_dataset_root="$formal_root/NUDT-SIRST"
formal_run_name="seed_42_formal800_pd_fp32_v1"
formal_expected_epochs=800
formal_gpu2_uuid="GPU-3509f974-0eba-2bcb-9469-372003ae4f0a"
formal_recovery_unit="sctransnet-formal800-gpu2-recovery-s42-v1.service"
formal_poll_seconds=30
formal_lock="$formal_root/.gpu2_recovery_postprocess_s42_v1.lock"
formal_recovery_lock="$formal_root/.gpu2_recovery_s42_v1.lock"
formal_state_file="$formal_root/gpu2_recovery_postprocess_state_s42_v1.json"
formal_comparison_dir="$formal_dataset_root/comparison"
formal_tpd_dir="$formal_dataset_root/tpd/$formal_run_name"
formal_archive="$formal_root/recovery_archive/20260723_112306_CST_gpu1_launch_failure"
formal_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
formal_variants=(original progressive tpd spd)

cd "$formal_repo"

exec 9>"$formal_lock"
if ! flock -n 9; then
    echo "GPU2_POSTPROCESS_ABORT reason=lock_held lock=$formal_lock" >&2
    exit 1
fi

formal_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

formal_require_sha256() {
    local formal_path="$1"
    local formal_expected="$2"
    local formal_actual
    [[ -f "$formal_path" && ! -L "$formal_path" ]] || {
        echo "GPU2_POSTPROCESS_ABORT reason=missing_or_symlink path=$formal_path" >&2
        return 1
    }
    formal_actual="$(formal_sha256 "$formal_path")"
    if [[ "$formal_actual" != "$formal_expected" ]]; then
        echo "GPU2_POSTPROCESS_ABORT reason=sha_mismatch path=$formal_path expected=$formal_expected actual=$formal_actual" >&2
        return 1
    fi
}

formal_verify_frozen_sources() {
    formal_require_sha256 experiments/run_tpd_formal800_gpu2_recovery.sh 746dbd7b7e6827b16984fb2dc8213118f354606548e7500825f27d81bc07a853
    formal_require_sha256 experiments/run_tpd_formal800_queue.sh a1c443770332daf1d9615f1f5b2756dd695749b808554bc0698380ba7eef66b7
    formal_require_sha256 experiments/train_tpd_pilot.py 7532bdc3bcc777aa164e258ab21f78d38ed3a1eaa677a29c8256d900224a7f26
    formal_require_sha256 experiments/fingerprint_tpd_training_data.py 26382e38e899bdf4f97b77c6671929c391decef6e4bf4ac40094a7d4e6b0bc7d
    formal_require_sha256 experiments/evaluate_pd_fa_sweep.py 0224ab44dc346ebdbd4cb4775c493bd6eecdc877019832dea0f16e59ab353537
    formal_require_sha256 experiments/summarize_tpd_pilot.py 49417a3fd43192e52308e7dc4343527bcda71495b78257c5464ba43d8eef7f3e
    formal_require_sha256 experiments/summarize_tpd_pd_fa.py 482903040cbe9a58f17444eee45aeb67c6763a6aef99bd54892f47be5e21b42e
    formal_require_sha256 dataset.py 516ea9c410f80cc9ae912cf0443126a067dd14b6cc5ad7945e83cfc497f4678d
    formal_require_sha256 utils.py afb6fc221072ddd082b53ccda132232bc9089afd0458d8f0e47a39b9c1e25c13
    formal_require_sha256 model/SCTransNet.py 5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb
    formal_require_sha256 model/tpd.py 18a5892edd18ab040e38f18c8d86a02bf3e50b7a4d12d0115ec9a97e8051c135
    formal_require_sha256 model/Config.py b7e3e67c379ef4638605ebe612336b0c3cdb1a97f4d6fe731dec80b4847d5596
}

formal_verify_training_data() {
    local formal_actual
    formal_actual="$(
        timeout 300s "$formal_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    [[ "$formal_actual" == "$formal_training_data_sha256" ]] || {
        echo "GPU2_POSTPROCESS_ABORT reason=training_data_drift expected=$formal_training_data_sha256 actual=$formal_actual" >&2
        return 1
    }
}

formal_verify_tpd_certificate() {
    formal_require_sha256 "$formal_tpd_dir/protocol.json" 6b974f73763aaa0d50ab666ea0e2ce7cd4ab526752082b4c2a64480603cf4773
    formal_require_sha256 "$formal_tpd_dir/split.json" 27c3b4a30c680af1c16493f723ce9713cb7e6987dcbc82e72fe1331cff12cd6b
    formal_require_sha256 "$formal_tpd_dir/metrics.jsonl" 0bd7e49bbe1d26e9592af39408a2798210d155852b0be4fcb6f8786ab38b464c
    formal_require_sha256 "$formal_tpd_dir/best.pth.tar" 9487fc82c3a2a8e6ab26ffb30c4b4cd6d6e3fb15d2ffbb7d0d77ef566ce78fa5
    formal_require_sha256 "$formal_tpd_dir/best_miou.pth.tar" 58e6171c7d7ccf9265c555371a7a92415d1bff391946ec2098e030b9855b7353
    formal_require_sha256 "$formal_tpd_dir/last.pth.tar" 4ed28c76ea51348b3f88993a061f1be4b5b2f5c53cb13ca88052f47fe6fe1fbe
    formal_require_sha256 "$formal_tpd_dir/summary.json" 0bb284cc87f8e1aa21d88bac7452d6cf4b7dee5ba94568d431dfd46577810a67
}

formal_verify_archive() {
    formal_require_sha256 "$formal_archive/FILES.relocated.sha256" 822a154e5cec2092b2c3d32a967be8bd55add90e5b1871bc4f439bd0af57031b
    (
        cd "$formal_archive"
        sha256sum -c FILES.relocated.sha256 >/dev/null
    )
}

formal_static_preflight() {
    formal_verify_frozen_sources
    formal_verify_training_data
    formal_verify_tpd_certificate
    formal_verify_archive
}

formal_snapshot_value() {
    local formal_snapshot="$1"
    local formal_key="$2"
    sed -n "s/^${formal_key}=//p" <<<"$formal_snapshot" | head -n 1
}

formal_capture_recovery_invocation() {
    local formal_snapshot
    local formal_load
    local formal_active
    local formal_invocation
    formal_snapshot="$(systemctl --user show "$formal_recovery_unit" \
        --property=LoadState,ActiveState,InvocationID 2>/dev/null || true)"
    formal_load="$(formal_snapshot_value "$formal_snapshot" LoadState)"
    formal_active="$(formal_snapshot_value "$formal_snapshot" ActiveState)"
    formal_invocation="$(formal_snapshot_value "$formal_snapshot" InvocationID)"
    if [[ "$formal_load" != "loaded" || "$formal_active" != "active" || ! "$formal_invocation" =~ ^[0-9a-f]{32}$ ]]; then
        echo "GPU2_POSTPROCESS_ABORT reason=cannot_capture_recovery_invocation load=$formal_load active=$formal_active invocation=$formal_invocation" >&2
        return 1
    fi
    printf '%s\n' "$formal_invocation"
}

formal_recovery_journal_complete() {
    local formal_invocation="$1"
    local formal_messages
    local formal_expected
    local formal_count
    formal_messages="$(journalctl --user "_SYSTEMD_INVOCATION_ID=$formal_invocation" \
        --no-pager --output=cat 2>/dev/null || true)"
    if grep -Fq "GPU2_RECOVERY_ABORT" <<<"$formal_messages"; then
        return 1
    fi
    for formal_expected in \
        "QUEUE_COMPLETE variant=original" \
        "QUEUE_COMPLETE variant=progressive" \
        "QUEUE_COMPLETE variant=spd" \
        "GPU2_RECOVERY_VARIANT_COMPLETE variant=original gpu_uuid=$formal_gpu2_uuid" \
        "GPU2_RECOVERY_VARIANT_COMPLETE variant=progressive gpu_uuid=$formal_gpu2_uuid" \
        "GPU2_RECOVERY_VARIANT_COMPLETE variant=spd gpu_uuid=$formal_gpu2_uuid" \
        "GPU2_RECOVERY_CROSS_AUDIT_COMPLETE variants=original,progressive,spd,tpd epochs=800" \
        "GPU2_RECOVERY_COMPLETE variants=original,progressive,spd gpu_uuid=$formal_gpu2_uuid"; do
        formal_count="$(grep -Fxc "$formal_expected" <<<"$formal_messages" || true)"
        [[ "$formal_count" -eq 1 ]] || {
            echo "GPU2_POSTPROCESS_ABORT reason=recovery_journal_proof_count expected=$formal_expected count=$formal_count" >&2
            return 1
        }
    done
}

formal_recovery_state() {
    local formal_expected_invocation="$1"
    local formal_snapshot
    local formal_load
    local formal_active
    local formal_result
    local formal_status
    local formal_invocation
    formal_snapshot="$(systemctl --user show "$formal_recovery_unit" \
        --property=LoadState,ActiveState,SubState,Result,ExecMainStatus,InvocationID \
        2>/dev/null || true)"
    formal_load="$(formal_snapshot_value "$formal_snapshot" LoadState)"
    formal_active="$(formal_snapshot_value "$formal_snapshot" ActiveState)"
    formal_result="$(formal_snapshot_value "$formal_snapshot" Result)"
    formal_status="$(formal_snapshot_value "$formal_snapshot" ExecMainStatus)"
    formal_invocation="$(formal_snapshot_value "$formal_snapshot" InvocationID)"

    if [[ "$formal_load" != "loaded" || "$formal_invocation" != "$formal_expected_invocation" ]]; then
        echo "GPU2_POSTPROCESS_ABORT reason=recovery_identity_changed expected=$formal_expected_invocation actual=$formal_invocation load=$formal_load" >&2
        return 2
    fi
    case "$formal_active" in
        active|activating|deactivating|reloading)
            return 1
            ;;
        failed)
            echo "GPU2_POSTPROCESS_ABORT reason=recovery_failed result=$formal_result status=$formal_status" >&2
            return 2
            ;;
        inactive)
            if [[ "$formal_result" == "success" && "$formal_status" == "0" ]] && \
                formal_recovery_journal_complete "$formal_expected_invocation"; then
                return 0
            fi
            echo "GPU2_POSTPROCESS_ABORT reason=recovery_inactive_without_proof result=$formal_result status=$formal_status" >&2
            return 2
            ;;
        *)
            echo "GPU2_POSTPROCESS_ABORT reason=unexpected_recovery_state active=$formal_active result=$formal_result status=$formal_status" >&2
            return 2
            ;;
    esac
}

formal_run_dir() {
    local formal_variant="$1"
    printf '%s/%s/%s\n' "$formal_dataset_root" "$formal_variant" "$formal_run_name"
}

formal_sweep_output() {
    local formal_variant="$1"
    printf '%s/pd_fa_sweep_best.pth.json\n' "$(formal_run_dir "$formal_variant")"
}

formal_sweep_valid() {
    local formal_variant="$1"
    local formal_dir
    local formal_output
    local formal_checkpoint
    local formal_checkpoint_sha
    local formal_recorded
    local formal_artifact
    local formal_artifact_path
    formal_dir="$(formal_run_dir "$formal_variant")"
    formal_output="$(formal_sweep_output "$formal_variant")"
    formal_checkpoint="$formal_dir/best.pth.tar"
    [[ -f "$formal_output" && ! -L "$formal_output" && -f "$formal_checkpoint" && ! -L "$formal_checkpoint" ]] || return 1
    formal_checkpoint_sha="$(formal_sha256 "$formal_checkpoint")"

    jq -e \
        --arg formal_variant "$formal_variant" \
        --arg formal_dir "$formal_dir" \
        --arg formal_checkpoint "$formal_checkpoint" \
        --arg formal_checkpoint_sha "$formal_checkpoint_sha" \
        --arg formal_evaluator_sha "0224ab44dc346ebdbd4cb4775c493bd6eecdc877019832dea0f16e59ab353537" \
        --arg formal_gpu "$formal_gpu2_uuid" \
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
        .audit.parsed_arguments.device == "cuda:0" and
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
        [[ -f "$formal_artifact_path" && ! -L "$formal_artifact_path" ]] || return 1
        formal_recorded="$(jq -r --arg formal_key "$formal_artifact" \
            '.audit.artifact_sha256[$formal_key] // empty' "$formal_output")"
        [[ "$formal_recorded" == "$(formal_sha256 "$formal_artifact_path")" ]] || return 1
    done
}

formal_validate_sweep_set() {
    local formal_reference=""
    local formal_current
    local formal_variant
    for formal_variant in "${formal_variants[@]}"; do
        formal_sweep_valid "$formal_variant" || return 1
        formal_current="$(jq -cS '{dataset,seed,split_seed,validation_split_sha256,match_radius,tiny_area,threshold_configuration,evaluator_sha256:.audit.artifact_sha256.evaluator}' \
            "$(formal_sweep_output "$formal_variant")")"
        if [[ -z "$formal_reference" ]]; then
            formal_reference="$formal_current"
        elif [[ "$formal_current" != "$formal_reference" ]]; then
            echo "GPU2_POSTPROCESS_ABORT reason=sweep_set_mismatch variant=$formal_variant" >&2
            return 1
        fi
    done
}

formal_gpu2_visible() {
    local formal_actual
    formal_actual="$(
        timeout 15s nvidia-smi -i "$formal_gpu2_uuid" \
            --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null
    )"
    [[ "$formal_actual" == "$formal_gpu2_uuid" ]]
}

formal_gpu2_cuda_healthy() {
    timeout 30s env CUDA_VISIBLE_DEVICES="$formal_gpu2_uuid" "$formal_python" -c '
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(1)
x = torch.ones((64, 64), device="cuda:0")
y = (x @ x).sum().item()
torch.cuda.synchronize()
raise SystemExit(0 if y == 262144.0 else 1)
' >/dev/null 2>&1
}

formal_gpu2_idle() {
    local formal_pids
    if ! formal_pids="$(
        timeout 15s nvidia-smi -i "$formal_gpu2_uuid" \
            --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null
    )"; then
        return 1
    fi
    [[ -z "$formal_pids" ]]
}

formal_wait_for_gpu2() {
    local formal_wait_count=0
    while true; do
        if formal_gpu2_visible && formal_gpu2_cuda_healthy && formal_gpu2_idle; then
            return 0
        fi
        if (( formal_wait_count % 20 == 0 )); then
            echo "GPU2_POSTPROCESS_GPU_WAIT gpu_uuid=$formal_gpu2_uuid poll_seconds=$formal_poll_seconds"
        fi
        formal_wait_count=$((formal_wait_count + 1))
        sleep "$formal_poll_seconds"
    done
}

formal_run_sweep() {
    local formal_variant="$1"
    local formal_dir
    local formal_output
    formal_dir="$(formal_run_dir "$formal_variant")"
    formal_output="$(formal_sweep_output "$formal_variant")"
    if [[ -e "$formal_output" || -L "$formal_output" ]]; then
        if formal_sweep_valid "$formal_variant"; then
            echo "GPU2_SWEEP_SKIP_VALID variant=$formal_variant output=$formal_output"
            return 0
        fi
        echo "GPU2_POSTPROCESS_ABORT reason=existing_sweep_invalid variant=$formal_variant output=$formal_output" >&2
        return 1
    fi
    formal_wait_for_gpu2
    formal_static_preflight
    formal_wait_for_gpu2
    echo "GPU2_SWEEP_START variant=$formal_variant gpu_uuid=$formal_gpu2_uuid"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$formal_gpu2_uuid" \
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
    formal_sweep_valid "$formal_variant"
    echo "GPU2_SWEEP_COMPLETE variant=$formal_variant gpu_uuid=$formal_gpu2_uuid"
}

formal_base_comparison_valid() {
    local formal_dir="$1"
    local formal_variant
    local formal_sweep_path
    [[ -d "$formal_dir" && ! -L "$formal_dir" ]] || return 1
    cmp -s <(
        for formal_variant in "${formal_variants[@]}"; do
            formal_sweep_path="$(formal_sweep_output "$formal_variant")"
            printf '%s  %s\n' "$(formal_sha256 "$formal_sweep_path")" "$formal_sweep_path"
        done
    ) "$formal_dir/SWEEPS.sha256" || return 1
    cmp -s <(
        cd "$formal_dir"
        sha256sum \
            "$formal_run_name.json" \
            "$formal_run_name.md" \
            "$formal_run_name.csv" \
            SWEEPS.sha256
    ) "$formal_dir/COMPLETE.sha256" || return 1
    (
        cd "$formal_dir"
        sha256sum -c SWEEPS.sha256 >/dev/null
        sha256sum -c COMPLETE.sha256 >/dev/null
    ) || return 1
    jq -e --argjson formal_epochs "$formal_expected_epochs" '
        .expected_epochs == $formal_epochs and
        .official_test_accessed == false and
        (.rows | type == "array" and length == 4) and
        ([.rows[].variant] | sort) == ["original", "progressive", "spd", "tpd"]
    ' "$formal_dir/$formal_run_name.json" >/dev/null
}

formal_aggregate_valid() {
    local formal_dir="$1"
    local formal_stem="pd_fa_$formal_run_name"
    local formal_marker="$formal_dir/$formal_stem.COMPLETE.sha256"
    local formal_path
    for formal_path in \
        "$formal_dir/$formal_stem.json" \
        "$formal_dir/$formal_stem.md" \
        "$formal_dir/${formal_stem}_operating_points.csv" \
        "$formal_dir/${formal_stem}_curves.csv" \
        "$formal_marker"; do
        [[ -s "$formal_path" && ! -L "$formal_path" ]] || return 1
    done
    cmp -s <(
        for formal_path in \
            "$formal_dir/$formal_stem.json" \
            "$formal_dir/$formal_stem.md" \
            "$formal_dir/${formal_stem}_operating_points.csv" \
            "$formal_dir/${formal_stem}_curves.csv"; do
            printf '%s  %s\n' "$(formal_sha256 "$formal_path")" "$(basename "$formal_path")"
        done
    ) "$formal_marker" || return 1
    (
        cd "$formal_dir"
        sha256sum -c "$formal_stem.COMPLETE.sha256" >/dev/null
    ) || return 1
    jq -e '
        .schema_version == "tpd-pd-fa-aggregate-v2" and
        .dataset == "NUDT-SIRST" and
        .run_name == "seed_42_formal800_pd_fp32_v1" and
        .expected_epochs == 800 and
        .official_test_accessed == false and
        .selection_source == "internal_validation_only" and
        .checkpoint_role == "best_validation_pd_primary" and
        .mainline_decision_made == false and
        .aggregator_sha256 == "482903040cbe9a58f17444eee45aeb67c6763a6aef99bd54892f47be5e21b42e" and
        .integrity_checks_passed.four_sweeps_present == true and
        .integrity_checks_passed.source_sweeps_manifest_verified == true and
        .integrity_checks_passed.source_complete_manifest_verified == true and
        .integrity_checks_passed.source_artifact_hashes_current == true and
        .integrity_checks_passed.all_source_audit_flags_true == true and
        .integrity_checks_passed.sealed_recorded_training_protocol_audit_verified == true and
        .integrity_checks_passed.training_checkpoint_sweep_binding_verified == true and
        .integrity_checks_passed.split_manifest_byte_identical == true and
        .integrity_checks_passed.thresholds_unique_sorted_finite == true and
        .integrity_checks_passed.point_count_identities_verified == true and
        .integrity_checks_passed.ground_truth_counts_invariant == true and
        .integrity_checks_passed.fixed_threshold_curve_point_exact == true and
        .integrity_checks_passed.fixed_threshold_checkpoint_object_metrics_exact == true and
        .integrity_checks_passed.fixed_threshold_checkpoint_numeric_deltas_recomputed == true and
        .integrity_checks_passed.fa_budget_points_recomputed_exact == true and
        .integrity_checks_passed.pareto_coordinates_recomputed == true and
        .integrity_checks_passed.hardware_timing_not_used_as_performance_evidence == true and
        (.integrity_checks_passed | length) == 17
    ' "$formal_dir/$formal_stem.json" >/dev/null
}

formal_final_complete() {
    formal_validate_sweep_set && \
        formal_base_comparison_valid "$formal_comparison_dir" && \
        formal_aggregate_valid "$formal_comparison_dir"
}

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

formal_static_preflight

if [[ "$formal_mode" == "--preflight" ]]; then
    echo "GPU2_POSTPROCESS_PREFLIGHT static_evidence_ok=1"
    exit 0
fi

if [[ -e "$formal_comparison_dir" || -L "$formal_comparison_dir" ]]; then
    echo "GPU2_POSTPROCESS_ABORT reason=existing_comparison_requires_manual_audit path=$formal_comparison_dir" >&2
    exit 1
fi

formal_postprocess_sha="$(formal_sha256 "$formal_repo/experiments/run_tpd_formal800_gpu2_recovery_postprocess.sh")"
formal_recovery_sha="$(formal_sha256 "$formal_repo/experiments/run_tpd_formal800_gpu2_recovery.sh")"
if [[ -e "$formal_state_file" || -L "$formal_state_file" ]]; then
    [[ -f "$formal_state_file" && ! -L "$formal_state_file" ]] || {
        echo "GPU2_POSTPROCESS_ABORT reason=invalid_state_file path=$formal_state_file" >&2
        exit 1
    }
    jq -e \
        --arg formal_unit "$formal_recovery_unit" \
        --arg formal_gpu "$formal_gpu2_uuid" \
        --arg formal_postprocess_sha "$formal_postprocess_sha" \
        --arg formal_recovery_sha "$formal_recovery_sha" '
        .schema_version == 1 and
        .recovery_unit == $formal_unit and
        .gpu_uuid == $formal_gpu and
        .postprocess_sha256 == $formal_postprocess_sha and
        .recovery_sha256 == $formal_recovery_sha and
        (.recovery_invocation_id | type == "string" and test("^[0-9a-f]{32}$"))
    ' "$formal_state_file" >/dev/null || {
        echo "GPU2_POSTPROCESS_ABORT reason=state_binding_mismatch path=$formal_state_file" >&2
        exit 1
    }
    formal_recovery_invocation="$(jq -r '.recovery_invocation_id' "$formal_state_file")"
    echo "GPU2_POSTPROCESS_STATE_RESTORED invocation=$formal_recovery_invocation"
else
    formal_recovery_invocation="$(formal_capture_recovery_invocation)"
    formal_state_tmp="$(mktemp "$formal_root/.gpu2_recovery_postprocess_state.tmp.XXXXXX")"
    jq -n \
        --arg formal_created_at "$(date --iso-8601=seconds)" \
        --arg formal_unit "$formal_recovery_unit" \
        --arg formal_invocation "$formal_recovery_invocation" \
        --arg formal_gpu "$formal_gpu2_uuid" \
        --arg formal_postprocess_sha "$formal_postprocess_sha" \
        --arg formal_recovery_sha "$formal_recovery_sha" '
        {
          schema_version: 1,
          created_at: $formal_created_at,
          recovery_unit: $formal_unit,
          recovery_invocation_id: $formal_invocation,
          gpu_uuid: $formal_gpu,
          postprocess_sha256: $formal_postprocess_sha,
          recovery_sha256: $formal_recovery_sha
        }
    ' > "$formal_state_tmp"
    mv -- "$formal_state_tmp" "$formal_state_file"
    echo "GPU2_POSTPROCESS_STATE_CREATED invocation=$formal_recovery_invocation"
fi

formal_wait_count=0
while true; do
    if formal_recovery_state "$formal_recovery_invocation"; then
        break
    else
        formal_status=$?
        [[ "$formal_status" -eq 1 ]] || exit "$formal_status"
    fi
    if (( formal_wait_count % 20 == 0 )); then
        echo "GPU2_POSTPROCESS_WAIT recovery_invocation=$formal_recovery_invocation poll_seconds=$formal_poll_seconds"
    fi
    formal_wait_count=$((formal_wait_count + 1))
    sleep "$formal_poll_seconds"
done

exec 8>"$formal_recovery_lock"
if ! flock -n 8; then
    echo "GPU2_POSTPROCESS_ABORT reason=recovery_lock_held lock=$formal_recovery_lock" >&2
    exit 1
fi

formal_static_preflight
formal_recovery_journal_complete "$formal_recovery_invocation"

echo "GPU2_POSTPROCESS_START recovery_invocation=$formal_recovery_invocation gpu_uuid=$formal_gpu2_uuid"
for formal_variant in "${formal_variants[@]}"; do
    formal_run_sweep "$formal_variant"
done
formal_validate_sweep_set
formal_static_preflight

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

for formal_variant in "${formal_variants[@]}"; do
    formal_sweep_path="$(formal_sweep_output "$formal_variant")"
    printf '%s  %s\n' "$(formal_sha256 "$formal_sweep_path")" "$formal_sweep_path"
done > "$formal_stage_dir/SWEEPS.sha256"

(
    cd "$formal_stage_dir"
    sha256sum \
        "$formal_run_name.json" \
        "$formal_run_name.md" \
        "$formal_run_name.csv" \
        SWEEPS.sha256 > COMPLETE.sha256
    sha256sum -c SWEEPS.sha256 >/dev/null
    sha256sum -c COMPLETE.sha256 >/dev/null
)
formal_base_comparison_valid "$formal_stage_dir"
formal_static_preflight
formal_validate_sweep_set

if [[ -e "$formal_comparison_dir" || -L "$formal_comparison_dir" ]]; then
    echo "GPU2_POSTPROCESS_ABORT reason=comparison_appeared_during_run path=$formal_comparison_dir" >&2
    exit 1
fi
mv -- "$formal_stage_dir" "$formal_comparison_dir"
formal_base_comparison_valid "$formal_comparison_dir"

"$formal_python" experiments/summarize_tpd_pd_fa.py \
    --root "$formal_root" \
    --dataset NUDT-SIRST \
    --run-name "$formal_run_name" \
    --expected-epochs "$formal_expected_epochs"

formal_static_preflight
formal_final_complete
formal_recovery_journal_complete "$formal_recovery_invocation"
echo "GPU2_POSTPROCESS_COMPLETE comparison=$formal_comparison_dir recovery_invocation=$formal_recovery_invocation"
