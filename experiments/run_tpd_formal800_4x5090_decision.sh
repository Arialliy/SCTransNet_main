#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
formal_root="$formal_repo/experiments/results/tpd_pe_formal800_4x5090_v1"
formal_dataset="NUDT-SIRST"
formal_run_name="seed_42_formal800_pd_fp32_4x5090_v1"
formal_expected_epochs=800
formal_comparison_dir="$formal_root/$formal_dataset/comparison"
formal_decider="$formal_repo/experiments/decide_tpd_mainline_4x5090.py"
formal_decider_sha256="c7fbca1a57783c887391b343983d34985013e751e50fa7439d08c15550ad1393"
formal_decider_test="$formal_repo/tests/test_decide_tpd_mainline_4x5090.py"
formal_decider_test_sha256="19c0f6e5ff18b4eef071e9b155e0606491aeaf2c2fb1f339809ee3a13368cd7f"
formal_postprocess_script="$formal_repo/experiments/run_tpd_formal800_4x5090_postprocess.sh"
formal_postprocess_script_sha256="fb6f341c4e2af373b5d0d03ba5525e82dd07570d597a98101481edfc9fe6835d"
formal_postprocess_launcher="$formal_repo/experiments/launch_tpd_formal800_4x5090_postprocess.sh"
formal_postprocess_launcher_sha256="008d3560138b217d4bdbfb01cd4fcf370514ec374b5dd4890d77363d5cd31bd6"
formal_replacement_state="$formal_root/launch/resilience_replacement_state_20260723_175419_CST.json"
formal_replacement_state_sha256="2945f5d84e3b73acf4eb840ca353b3f2b0edbd346ab922005f73ff97358de720"
formal_postprocess_unit="sctransnet-formal800-4x5090-postprocess.service"
formal_postprocess_invocation="fb873fd0610e4f5c92acaccd0afcad47"
formal_postprocess_lock="$formal_root/.postprocess_4x5090_v1.lock"
formal_decision_lock="$formal_root/.decision_4x5090_v1.wait.lock"
formal_log="$formal_root/logs/decision.log"
formal_poll_seconds=60
formal_output_stem="mainline_decision_seed42"

mkdir -p "$formal_root/logs"
exec > >(tee -a "$formal_log") 2>&1
cd "$formal_repo"

exec 9>"$formal_decision_lock"
if ! flock -n 9; then
    echo "FORMAL4X5090_DECISION_ABORT reason=lock_held lock=$formal_decision_lock" >&2
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
        echo "FORMAL4X5090_DECISION_ABORT reason=missing_or_symlink path=$formal_path" >&2
        return 1
    }
    formal_actual="$(formal_sha256 "$formal_path")"
    if [[ "$formal_actual" != "$formal_expected" ]]; then
        echo "FORMAL4X5090_DECISION_ABORT reason=sha_mismatch path=$formal_path expected=$formal_expected actual=$formal_actual" >&2
        return 1
    fi
}

formal_snapshot_value() {
    local formal_snapshot="$1"
    local formal_key="$2"
    sed -n "s/^${formal_key}=//p" <<<"$formal_snapshot" | head -n 1
}

formal_static_preflight() {
    formal_require_sha256 "$formal_decider" "$formal_decider_sha256"
    formal_require_sha256 "$formal_decider_test" "$formal_decider_test_sha256"
    formal_require_sha256 \
        "$formal_postprocess_script" \
        "$formal_postprocess_script_sha256"
    formal_require_sha256 \
        "$formal_postprocess_launcher" \
        "$formal_postprocess_launcher_sha256"
    formal_require_sha256 \
        "$formal_replacement_state" \
        "$formal_replacement_state_sha256"
    jq -e \
        --arg formal_invocation "$formal_postprocess_invocation" \
        --arg formal_script_sha "$formal_postprocess_script_sha256" \
        --arg formal_launcher_sha "$formal_postprocess_launcher_sha256" '
        .schema == "sctransnet_formal800_4x5090_resilience_replacement_state_v1" and
        .training_units_modified == false and
        .replacements.postprocess.new_invocation_id == $formal_invocation and
        .replacements.postprocess.new_script_sha256 == $formal_script_sha and
        .replacements.postprocess.new_launcher_sha256 == $formal_launcher_sha and
        .collected_unit_completion_policy.requires_frozen_runtime_identity == true and
        .collected_unit_completion_policy.requires_complete_800_epoch_artifacts == true and
        .collected_unit_completion_policy.requires_exact_success_markers_from_captured_invocation_journal == true and
        .collected_unit_completion_policy.all_other_identity_changes_fail_closed == true and
        .collected_unit_completion_policy.official_test_accessed == false
    ' "$formal_replacement_state" >/dev/null
    (
        cd "$formal_root/launch"
        sha256sum -c STATE_FILES.sha256 >/dev/null
    )
}

formal_postprocess_messages() {
    journalctl --user \
        "_SYSTEMD_INVOCATION_ID=$formal_postprocess_invocation" \
        --no-pager --output=cat 2>/dev/null || true
}

formal_postprocess_journal_complete() {
    local formal_messages
    local formal_complete_marker
    formal_messages="$(formal_postprocess_messages)"
    formal_complete_marker="FORMAL4X5090_POSTPROCESS_COMPLETE comparison=$formal_comparison_dir"
    if grep -Fq "FORMAL4X5090_POSTPROCESS_ABORT" <<<"$formal_messages"; then
        return 1
    fi
    if grep -Eq "Traceback|CUDA error|out of memory|OutOfMemory|Killed|No space left|NaN|Inf" <<<"$formal_messages"; then
        return 1
    fi
    [[ "$(grep -Fxc "$formal_complete_marker" <<<"$formal_messages" || true)" -eq 1 ]]
}

# Return 0 only for a completed postprocess, 1 while it is still running, and
# 2 for an identity change or terminal state without exact completion proof.
formal_postprocess_state() {
    local formal_snapshot
    local formal_load
    local formal_active
    local formal_result
    local formal_status
    local formal_main_code
    local formal_invocation
    local formal_restarts
    local formal_messages
    local formal_completion_count
    formal_snapshot="$(
        systemctl --user show "$formal_postprocess_unit" \
            --property=LoadState,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus,InvocationID,NRestarts \
            2>/dev/null || true
    )"
    formal_load="$(formal_snapshot_value "$formal_snapshot" LoadState)"
    formal_active="$(formal_snapshot_value "$formal_snapshot" ActiveState)"
    formal_result="$(formal_snapshot_value "$formal_snapshot" Result)"
    formal_status="$(formal_snapshot_value "$formal_snapshot" ExecMainStatus)"
    formal_main_code="$(formal_snapshot_value "$formal_snapshot" ExecMainCode)"
    formal_invocation="$(formal_snapshot_value "$formal_snapshot" InvocationID)"
    formal_restarts="$(formal_snapshot_value "$formal_snapshot" NRestarts)"
    formal_messages="$(formal_postprocess_messages)"
    formal_completion_count="$(
        grep -Fxc \
            "FORMAL4X5090_POSTPROCESS_COMPLETE comparison=$formal_comparison_dir" \
            <<<"$formal_messages" || true
    )"

    if grep -Fq "FORMAL4X5090_POSTPROCESS_ABORT" <<<"$formal_messages" || \
        grep -Eq "Traceback|CUDA error|out of memory|OutOfMemory|Killed|No space left|NaN|Inf" <<<"$formal_messages"; then
        echo "FORMAL4X5090_DECISION_ABORT reason=postprocess_journal_failure invocation=$formal_postprocess_invocation" >&2
        return 2
    fi
    if [[ "$formal_completion_count" -gt 1 ]]; then
        echo "FORMAL4X5090_DECISION_ABORT reason=duplicate_postprocess_completion count=$formal_completion_count" >&2
        return 2
    fi

    if [[ "$formal_load" == "not-found" && -z "$formal_invocation" ]]; then
        if [[ "$formal_completion_count" -eq 1 ]]; then
            return 0
        fi
        echo "FORMAL4X5090_DECISION_ABORT reason=postprocess_collected_without_completion_proof expected_invocation=$formal_postprocess_invocation" >&2
        return 2
    fi
    if [[ "$formal_load" != "loaded" || "$formal_invocation" != "$formal_postprocess_invocation" ]]; then
        echo "FORMAL4X5090_DECISION_ABORT reason=postprocess_identity_changed expected_invocation=$formal_postprocess_invocation actual_invocation=$formal_invocation load=$formal_load" >&2
        return 2
    fi
    if [[ "$formal_restarts" != "0" ]]; then
        echo "FORMAL4X5090_DECISION_ABORT reason=postprocess_restarted count=$formal_restarts" >&2
        return 2
    fi
    case "$formal_active" in
        active|activating|deactivating|reloading)
            return 1
            ;;
        failed)
            echo "FORMAL4X5090_DECISION_ABORT reason=postprocess_failed result=$formal_result status=$formal_status" >&2
            return 2
            ;;
        inactive)
            if [[ "$formal_result" == "success" && "$formal_main_code" == "1" && "$formal_status" == "0" && "$formal_completion_count" -eq 1 ]]; then
                return 0
            fi
            echo "FORMAL4X5090_DECISION_ABORT reason=postprocess_inactive_without_completion_proof result=$formal_result status=$formal_status" >&2
            return 2
            ;;
        *)
            echo "FORMAL4X5090_DECISION_ABORT reason=unexpected_postprocess_state active=$formal_active result=$formal_result status=$formal_status" >&2
            return 2
            ;;
    esac
}

formal_decider_command() {
    "$formal_python" "$formal_decider" \
        --root "$formal_root" \
        --dataset "$formal_dataset" \
        --run-name "$formal_run_name" \
        --expected-epochs "$formal_expected_epochs" \
        "$@"
}

formal_decision_valid() {
    local formal_json="$formal_comparison_dir/$formal_output_stem.json"
    local formal_md="$formal_comparison_dir/$formal_output_stem.md"
    local formal_marker="$formal_comparison_dir/$formal_output_stem.COMPLETE.sha256"
    local formal_path
    for formal_path in "$formal_json" "$formal_md" "$formal_marker"; do
        [[ -s "$formal_path" && ! -L "$formal_path" ]] || return 1
    done
    cmp -s <(
        cd "$formal_comparison_dir"
        sha256sum "$formal_output_stem.json" "$formal_output_stem.md"
    ) "$formal_marker" || return 1
    (
        cd "$formal_comparison_dir"
        sha256sum -c "$formal_output_stem.COMPLETE.sha256" >/dev/null
    ) || return 1
    jq -e \
        --arg formal_dataset "$formal_dataset" \
        --arg formal_run_name "$formal_run_name" \
        --arg formal_decider_sha "$formal_decider_sha256" '
        .schema_version == "tpd-mainline-seed42-decision-v1" and
        .dataset == $formal_dataset and
        .run_name == $formal_run_name and
        .expected_epochs == 800 and
        .seed == 42 and
        .selection_source == "internal_validation_only" and
        .official_test_accessed == false and
        .paper_core_established == false and
        .stability_claim_supported == false and
        .scope == "single_dataset_single_seed_screening_only" and
        .policy.name == "post_hoc_conservative_operational_policy_v2" and
        .policy.status == "post_hoc_not_launch_preregistered" and
        .policy.four_rtx5090_launch_natively_bound_this_policy == false and
        (.screening_decision.decision == "ADVANCE_TPD_TO_MULTI_SEED" or
         .screening_decision.decision == "DO_NOT_ESTABLISH_CURRENT_TPD_CORE" or
         .screening_decision.decision == "INCONCLUSIVE_MIXED_TRADEOFF") and
        .input_provenance.decision_implementation_sha256 == $formal_decider_sha and
        .evidence_gate.extended_audit_independently_reexecuted_byte_identical == true and
        .evidence_gate.training_data_fingerprint_independently_recomputed == true and
        .evidence_gate.current_sweep_curves_recomputed_exact == true and
        .evidence_gate.current_raw_artifact_hashes_recomputed_exact == true and
        .evidence_gate.current_frozen_source_hashes_recomputed_exact == true
    ' "$formal_json" >/dev/null || return 1
    cmp -s <(formal_decider_command --validate-only) "$formal_json"
}

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

formal_static_preflight

if [[ "$formal_mode" == "--preflight" ]]; then
    if formal_postprocess_state; then
        echo "FORMAL4X5090_DECISION_PREFLIGHT postprocess=complete"
    else
        formal_status=$?
        if [[ "$formal_status" -eq 1 ]]; then
            echo "FORMAL4X5090_DECISION_PREFLIGHT postprocess=running"
        else
            exit "$formal_status"
        fi
    fi
    if [[ -e "$formal_comparison_dir/$formal_output_stem.json" || \
          -e "$formal_comparison_dir/$formal_output_stem.md" || \
          -e "$formal_comparison_dir/$formal_output_stem.COMPLETE.sha256" ]]; then
        formal_decision_valid
        echo "FORMAL4X5090_DECISION_PREFLIGHT decision=complete_valid"
    else
        echo "FORMAL4X5090_DECISION_PREFLIGHT decision=pending"
    fi
    exit 0
fi

if [[ -e "$formal_comparison_dir/$formal_output_stem.json" || \
      -e "$formal_comparison_dir/$formal_output_stem.md" || \
      -e "$formal_comparison_dir/$formal_output_stem.COMPLETE.sha256" ]]; then
    if formal_decision_valid; then
        echo "FORMAL4X5090_DECISION_SKIP reason=already_complete_valid"
        exit 0
    fi
    echo "FORMAL4X5090_DECISION_ABORT reason=existing_decision_invalid" >&2
    exit 1
fi

formal_wait_count=0
while true; do
    if formal_postprocess_state; then
        break
    else
        formal_status=$?
        if [[ "$formal_status" -ne 1 ]]; then
            exit "$formal_status"
        fi
    fi
    if (( formal_wait_count % 20 == 0 )); then
        echo "FORMAL4X5090_DECISION_WAIT postprocess_invocation=$formal_postprocess_invocation poll_seconds=$formal_poll_seconds"
        formal_static_preflight
    fi
    formal_wait_count=$((formal_wait_count + 1))
    sleep "$formal_poll_seconds"
done

formal_static_preflight
formal_postprocess_journal_complete
if ! flock --shared --wait 300 "$formal_postprocess_lock" true; then
    echo "FORMAL4X5090_DECISION_ABORT reason=postprocess_lock_not_released lock=$formal_postprocess_lock" >&2
    exit 1
fi
formal_static_preflight
formal_postprocess_journal_complete

formal_decider_command
formal_decision_valid
formal_decision="$(
    jq -r '.screening_decision.decision' \
        "$formal_comparison_dir/$formal_output_stem.json"
)"
echo "FORMAL4X5090_DECISION_COMPLETE decision=$formal_decision paper_core_established=false stability_claim_supported=false"
