#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
formal_root="$formal_repo/experiments/results/tpd_pe_formal800_4x5090_v1"
formal_dataset_root="$formal_root/NUDT-SIRST"
formal_run_name="seed_42_formal800_pd_fp32_4x5090_v1"
formal_expected_epochs=800
formal_poll_seconds=60
formal_lock="$formal_root/.postprocess_4x5090_v1.lock"
formal_log="$formal_root/logs/postprocess.log"
formal_comparison_dir="$formal_dataset_root/comparison"
formal_runtime_state="$formal_root/launch/systemd_state_20260723_173022_CST.json"
formal_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
formal_split_sha256="27c3b4a30c680af1c16493f723ce9713cb7e6987dcbc82e72fe1331cff12cd6b"
formal_variants=(original progressive tpd spd)
formal_gpu_uuids=(
    GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70
    GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
)
formal_training_units=(
    sctransnet-formal800-4x5090-original.service
    sctransnet-formal800-4x5090-progressive.service
    sctransnet-formal800-4x5090-tpd.service
    sctransnet-formal800-4x5090-spd.service
)
formal_training_invocations=(
    0e533dd2d1444feb9ba1ed1e3d42135c
    0803e8ed5f974f909c54d6daaddc23bb
    669e46b8bd2245db92788afa62066bd3
    75468708b771457daac1cf0586f7c333
)

mkdir -p "$formal_root/logs"
exec > >(tee -a "$formal_log") 2>&1
cd "$formal_repo"

exec 9>"$formal_lock"
if ! flock -n 9; then
    echo "FORMAL4X5090_POSTPROCESS_ABORT reason=lock_held lock=$formal_lock" >&2
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
        echo "FORMAL4X5090_POSTPROCESS_ABORT reason=missing_or_symlink path=$formal_path" >&2
        return 1
    }
    formal_actual="$(formal_sha256 "$formal_path")"
    if [[ "$formal_actual" != "$formal_expected" ]]; then
        echo "FORMAL4X5090_POSTPROCESS_ABORT reason=sha_mismatch path=$formal_path expected=$formal_expected actual=$formal_actual" >&2
        return 1
    fi
}

formal_run_dir() {
    local formal_variant="$1"
    printf '%s/%s/%s\n' "$formal_dataset_root" "$formal_variant" "$formal_run_name"
}

formal_sweep_output() {
    local formal_variant="$1"
    printf '%s/pd_fa_sweep_best.pth.json\n' "$(formal_run_dir "$formal_variant")"
}

formal_gpu_for_variant() {
    local formal_requested="$1"
    local formal_index
    for formal_index in "${!formal_variants[@]}"; do
        if [[ "${formal_variants[$formal_index]}" == "$formal_requested" ]]; then
            printf '%s\n' "${formal_gpu_uuids[$formal_index]}"
            return 0
        fi
    done
    return 1
}

formal_verify_sources() {
    formal_require_sha256 experiments/run_tpd_formal800_4x5090_worker.sh 72f1c0503fcec0c69f8a3b9c49da57a49db75134cc453031da56d635adc2d7a7
    formal_require_sha256 experiments/launch_tpd_formal800_4x5090.sh 78b397a5c17bfeb62c3a83ec3aaf4ff733f97c29f686936e140fb7a0a7741fd8
    formal_require_sha256 experiments/status_tpd_formal800_4x5090.sh 2ecc3621b4bedf6bd452d1bc1d3273a168925dd7e2af067cc0dfb7aeb0fd0a40
    formal_require_sha256 experiments/train_tpd_pilot.py 7532bdc3bcc777aa164e258ab21f78d38ed3a1eaa677a29c8256d900224a7f26
    formal_require_sha256 experiments/fingerprint_tpd_training_data.py 26382e38e899bdf4f97b77c6671929c391decef6e4bf4ac40094a7d4e6b0bc7d
    formal_require_sha256 experiments/evaluate_pd_fa_sweep.py 0224ab44dc346ebdbd4cb4775c493bd6eecdc877019832dea0f16e59ab353537
    formal_require_sha256 experiments/summarize_tpd_pilot.py 49417a3fd43192e52308e7dc4343527bcda71495b78257c5464ba43d8eef7f3e
    formal_require_sha256 experiments/summarize_tpd_pd_fa.py 482903040cbe9a58f17444eee45aeb67c6763a6aef99bd54892f47be5e21b42e
    formal_require_sha256 experiments/audit_tpd_formal800_4x5090.py a5289b2d0cc8045d1514da5045e015b2d03f669e830817d3fcee1b7872f21c9d
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
    if [[ "$formal_actual" != "$formal_training_data_sha256" ]]; then
        echo "FORMAL4X5090_POSTPROCESS_ABORT reason=training_data_drift expected=$formal_training_data_sha256 actual=$formal_actual" >&2
        return 1
    fi
}

formal_verify_launch_manifests() {
    local formal_index
    local formal_variant
    local formal_gpu
    local formal_manifest
    for formal_index in "${!formal_variants[@]}"; do
        formal_variant="${formal_variants[$formal_index]}"
        formal_gpu="${formal_gpu_uuids[$formal_index]}"
        formal_manifest="$formal_root/launch/$formal_variant.json"
        [[ -f "$formal_manifest" && ! -L "$formal_manifest" ]] || {
            echo "FORMAL4X5090_POSTPROCESS_ABORT reason=missing_launch_manifest variant=$formal_variant" >&2
            return 1
        }
        jq -e \
            --arg formal_variant "$formal_variant" \
            --arg formal_gpu "$formal_gpu" \
            --arg formal_run_dir "$(formal_run_dir "$formal_variant")" \
            --arg formal_data "$formal_training_data_sha256" '
            .schema == "sctransnet_formal800_4x5090_launch_v1" and
            .variant == $formal_variant and
            .gpu_uuid == $formal_gpu and
            .gpu_name == "NVIDIA GeForce RTX 5090" and
            .gpu_capability == [12, 0] and
            .python_executable == "/home/ly/BasicIRSTD/infrarenet/bin/python" and
            .python_version == "3.12.3" and
            .torch == "2.9.1+cu130" and
            .cuda_runtime == "13.0" and
            .run_directory == $formal_run_dir and
            .training_data_sha256 == $formal_data and
            .policy.one_variant_per_gpu == true and
            .policy.fresh_run == true and
            .policy.old_formal800_results_preserved == true and
            .policy.official_test_accessed == false and
            .policy.amp == false
        ' "$formal_manifest" >/dev/null
    done
}

formal_verify_runtime_state() {
    formal_require_sha256 \
        "$formal_runtime_state" \
        7178a5157a4eae5641d410f9346bfe7ff847a90fca68a567e311cdc2baae23e2
    jq -e '
        .schema == "sctransnet_formal800_4x5090_systemd_state_v1" and
        .worker_sha256 == "72f1c0503fcec0c69f8a3b9c49da57a49db75134cc453031da56d635adc2d7a7" and
        .launcher_sha256 == "78b397a5c17bfeb62c3a83ec3aaf4ff733f97c29f686936e140fb7a0a7741fd8" and
        .status_script_sha256 == "2ecc3621b4bedf6bd452d1bc1d3273a168925dd7e2af067cc0dfb7aeb0fd0a40" and
        (.units | length) == 4 and
        [.units[] | {
            variant,
            unit,
            invocation_id,
            gpu_uuid,
            exec_start,
            n_restarts_at_capture,
            launch_manifest_sha256
        }] == [
          {
            "variant":"original",
            "unit":"sctransnet-formal800-4x5090-original.service",
            "invocation_id":"0e533dd2d1444feb9ba1ed1e3d42135c",
            "gpu_uuid":"GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
            "exec_start":[
              "/usr/bin/bash",
              "/home/ly/SCTransNet_main/experiments/run_tpd_formal800_4x5090_worker.sh",
              "original",
              "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70"
            ],
            "n_restarts_at_capture":0,
            "launch_manifest_sha256":"2841ef7217e21b7cc58d6d9f7dadeb4f995191027339afc11a314f6f32a31ad9"
          },
          {
            "variant":"progressive",
            "unit":"sctransnet-formal800-4x5090-progressive.service",
            "invocation_id":"0803e8ed5f974f909c54d6daaddc23bb",
            "gpu_uuid":"GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
            "exec_start":[
              "/usr/bin/bash",
              "/home/ly/SCTransNet_main/experiments/run_tpd_formal800_4x5090_worker.sh",
              "progressive",
              "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640"
            ],
            "n_restarts_at_capture":0,
            "launch_manifest_sha256":"194df73ffe812b04c462a541508cae2507db6f5d076265f3fea7be88d2f4cec2"
          },
          {
            "variant":"tpd",
            "unit":"sctransnet-formal800-4x5090-tpd.service",
            "invocation_id":"669e46b8bd2245db92788afa62066bd3",
            "gpu_uuid":"GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
            "exec_start":[
              "/usr/bin/bash",
              "/home/ly/SCTransNet_main/experiments/run_tpd_formal800_4x5090_worker.sh",
              "tpd",
              "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
            ],
            "n_restarts_at_capture":0,
            "launch_manifest_sha256":"fe277ef7e31518f0c0b9d5b6009686c26a360c2663153abdd84b0bf6b3726868"
          },
          {
            "variant":"spd",
            "unit":"sctransnet-formal800-4x5090-spd.service",
            "invocation_id":"75468708b771457daac1cf0586f7c333",
            "gpu_uuid":"GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
            "exec_start":[
              "/usr/bin/bash",
              "/home/ly/SCTransNet_main/experiments/run_tpd_formal800_4x5090_worker.sh",
              "spd",
              "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
            ],
            "n_restarts_at_capture":0,
            "launch_manifest_sha256":"c80da19164d1563b759df216bc569e3fbb453e2568d261147fb0dc5836ea18fa"
          }
        ]
    ' "$formal_runtime_state" >/dev/null

    local formal_variant
    local formal_expected
    local formal_actual
    for formal_variant in "${formal_variants[@]}"; do
        formal_expected="$(
            jq -r --arg formal_variant "$formal_variant" \
                '.units[] | select(.variant == $formal_variant) | .launch_manifest_sha256' \
                "$formal_runtime_state"
        )"
        formal_actual="$(formal_sha256 "$formal_root/launch/$formal_variant.json")"
        [[ "$formal_actual" == "$formal_expected" ]] || {
            echo "FORMAL4X5090_POSTPROCESS_ABORT reason=launch_manifest_state_drift variant=$formal_variant expected=$formal_expected actual=$formal_actual" >&2
            return 1
        }
    done
}

formal_static_preflight() {
    formal_verify_sources
    formal_verify_training_data
    formal_verify_launch_manifests
    formal_verify_runtime_state
}

formal_snapshot_value() {
    local formal_snapshot="$1"
    local formal_key="$2"
    sed -n "s/^${formal_key}=//p" <<<"$formal_snapshot" | head -n 1
}

formal_basic_run_complete() {
    local formal_variant="$1"
    local formal_dir
    local formal_artifact
    formal_dir="$(formal_run_dir "$formal_variant")"
    for formal_artifact in protocol.json split.json metrics.jsonl best.pth.tar best_miou.pth.tar last.pth.tar summary.json; do
        [[ -s "$formal_dir/$formal_artifact" && ! -L "$formal_dir/$formal_artifact" ]] || return 1
    done
    [[ "$(wc -l < "$formal_dir/metrics.jsonl")" -eq "$formal_expected_epochs" ]] || return 1
    [[ "$(formal_sha256 "$formal_dir/split.json")" == "$formal_split_sha256" ]] || return 1
    jq -e \
        --arg formal_variant "$formal_variant" '
        .status == "complete" and
        .variant == $formal_variant and
        .dataset == "NUDT-SIRST" and
        .seed == 42 and
        .selection_source == "internal_validation_only" and
        .official_test_accessed == false and
        .model.shared_initialization_sha256 == "ae25925e8fffd9afe9fac1805389e80437f0d773ae744c979349a68886d81558"
    ' "$formal_dir/summary.json" >/dev/null
    jq -e '
        .arguments.epochs == 800 and
        .arguments.eval_every == 1 and
        .arguments.seed == 42 and
        .arguments.split_seed == 20260722 and
        .arguments.amp == false and
        .official_test_accessed == false and
        .environment.device_name == "NVIDIA GeForce RTX 5090" and
        .environment.torch == "2.9.1+cu130" and
        .environment.cuda_runtime == "13.0"
    ' "$formal_dir/protocol.json" >/dev/null
}

formal_training_journal_complete() {
    local formal_index="$1"
    local formal_variant="${formal_variants[$formal_index]}"
    local formal_gpu="${formal_gpu_uuids[$formal_index]}"
    local formal_invocation="${formal_training_invocations[$formal_index]}"
    local formal_messages
    formal_messages="$(
        journalctl --user "_SYSTEMD_INVOCATION_ID=$formal_invocation" \
            --no-pager --output=cat 2>/dev/null || true
    )"
    if grep -Fq "FORMAL4X5090_ABORT" <<<"$formal_messages"; then
        return 1
    fi
    if grep -Eq "Traceback|CUDA error|out of memory|OutOfMemory|Killed|No space left|NaN|Inf" <<<"$formal_messages"; then
        return 1
    fi
    [[ "$(grep -Fxc "FORMAL4X5090_GPU_OK gpu_uuid=$formal_gpu torch=2.9.1+cu130 cuda=13.0 capability=(12, 0)" <<<"$formal_messages" || true)" -eq 1 ]] || return 1
    [[ "$(grep -Fxc "FORMAL4X5090_START variant=$formal_variant gpu_uuid=$formal_gpu run_dir=$(formal_run_dir "$formal_variant")" <<<"$formal_messages" || true)" -eq 1 ]] || return 1
    [[ "$(grep -Ec "^COMPLETE variant=$formal_variant " <<<"$formal_messages" || true)" -eq 1 ]] || return 1
    [[ "$(grep -Fxc "FORMAL4X5090_COMPLETE variant=$formal_variant gpu_uuid=$formal_gpu epochs=800" <<<"$formal_messages" || true)" -eq 1 ]] || return 1
}

formal_training_state() {
    local formal_index="$1"
    local formal_unit="${formal_training_units[$formal_index]}"
    local formal_expected_invocation="${formal_training_invocations[$formal_index]}"
    local formal_variant="${formal_variants[$formal_index]}"
    local formal_snapshot
    local formal_load
    local formal_active
    local formal_result
    local formal_status
    local formal_invocation
    local formal_restarts
    local formal_main_code
    formal_snapshot="$(
        systemctl --user show "$formal_unit" \
            --property=LoadState,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus,InvocationID,NRestarts \
            2>/dev/null || true
    )"
    formal_load="$(formal_snapshot_value "$formal_snapshot" LoadState)"
    formal_active="$(formal_snapshot_value "$formal_snapshot" ActiveState)"
    formal_result="$(formal_snapshot_value "$formal_snapshot" Result)"
    formal_status="$(formal_snapshot_value "$formal_snapshot" ExecMainStatus)"
    formal_invocation="$(formal_snapshot_value "$formal_snapshot" InvocationID)"
    formal_restarts="$(formal_snapshot_value "$formal_snapshot" NRestarts)"
    formal_main_code="$(formal_snapshot_value "$formal_snapshot" ExecMainCode)"

    # Transient services use CollectMode=inactive and RemainAfterExit=no.  A
    # successful service is therefore garbage-collected immediately, before a
    # 60-second poll can observe ActiveState=inactive.  In that one exact state,
    # accept completion only when both the pinned-invocation journal and the
    # complete run artifacts independently prove success.  Every other identity
    # change remains fail-closed.
    if [[ "$formal_load" == "not-found" && -z "$formal_invocation" ]]; then
        if formal_basic_run_complete "$formal_variant" && \
            formal_training_journal_complete "$formal_index"; then
            return 0
        fi
        echo "FORMAL4X5090_POSTPROCESS_ABORT reason=training_unit_collected_without_completion_proof variant=$formal_variant unit=$formal_unit expected_invocation=$formal_expected_invocation" >&2
        return 2
    fi
    if [[ "$formal_load" != "loaded" || "$formal_invocation" != "$formal_expected_invocation" ]]; then
        echo "FORMAL4X5090_POSTPROCESS_ABORT reason=training_identity_changed variant=$formal_variant unit=$formal_unit expected_invocation=$formal_expected_invocation actual_invocation=$formal_invocation load=$formal_load" >&2
        return 2
    fi
    case "$formal_active" in
        active|activating|deactivating|reloading)
            return 1
            ;;
        failed)
            echo "FORMAL4X5090_POSTPROCESS_ABORT reason=training_failed variant=$formal_variant result=$formal_result status=$formal_status" >&2
            return 2
            ;;
        inactive)
            if [[ "$formal_result" == "success" && "$formal_main_code" == "1" && "$formal_status" == "0" && "$formal_restarts" == "0" ]] && \
                formal_basic_run_complete "$formal_variant" && \
                formal_training_journal_complete "$formal_index"; then
                return 0
            fi
            echo "FORMAL4X5090_POSTPROCESS_ABORT reason=training_inactive_without_completion_proof variant=$formal_variant result=$formal_result status=$formal_status" >&2
            return 2
            ;;
        *)
            echo "FORMAL4X5090_POSTPROCESS_ABORT reason=unexpected_training_state variant=$formal_variant active=$formal_active result=$formal_result status=$formal_status" >&2
            return 2
            ;;
    esac
}

formal_sweep_valid() {
    local formal_variant="$1"
    local formal_gpu
    local formal_dir
    local formal_output
    local formal_checkpoint
    local formal_checkpoint_sha
    local formal_artifact
    local formal_recorded
    formal_gpu="$(formal_gpu_for_variant "$formal_variant")"
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
        --arg formal_gpu "$formal_gpu" \
        --arg formal_evaluator_sha "0224ab44dc346ebdbd4cb4775c493bd6eecdc877019832dea0f16e59ab353537" \
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
        formal_recorded="$(
            jq -r --arg formal_key "$formal_artifact" \
                '.audit.artifact_sha256[$formal_key] // empty' "$formal_output"
        )"
        [[ "$formal_recorded" == "$(formal_sha256 "$formal_dir/$formal_artifact")" ]] || return 1
    done
}

formal_validate_sweep_set() {
    local formal_reference=""
    local formal_current
    local formal_variant
    for formal_variant in "${formal_variants[@]}"; do
        formal_sweep_valid "$formal_variant" || return 1
        formal_current="$(
            jq -cS \
                '{dataset,seed,split_seed,validation_split_sha256,match_radius,tiny_area,threshold_configuration,evaluator_sha256:.audit.artifact_sha256.evaluator}' \
                "$(formal_sweep_output "$formal_variant")"
        )"
        if [[ -z "$formal_reference" ]]; then
            formal_reference="$formal_current"
        elif [[ "$formal_current" != "$formal_reference" ]]; then
            echo "FORMAL4X5090_POSTPROCESS_ABORT reason=sweep_set_mismatch variant=$formal_variant" >&2
            return 1
        fi
    done
}

formal_gpu_healthy() {
    local formal_gpu="$1"
    [[ "$(
        nvidia-smi -i "$formal_gpu" --query-gpu=name --format=csv,noheader,nounits
    )" == "NVIDIA GeForce RTX 5090" ]] || return 1
    timeout 30s env CUDA_VISIBLE_DEVICES="$formal_gpu" "$formal_python" -c '
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(1)
x = torch.ones((64, 64), device="cuda:0")
y = (x @ x).sum().item()
torch.cuda.synchronize()
raise SystemExit(0 if y == 262144.0 else 1)
' >/dev/null 2>&1
}

formal_run_sweep() {
    local formal_variant="$1"
    local formal_gpu
    local formal_dir
    local formal_output
    formal_gpu="$(formal_gpu_for_variant "$formal_variant")"
    formal_dir="$(formal_run_dir "$formal_variant")"
    formal_output="$(formal_sweep_output "$formal_variant")"
    if [[ -e "$formal_output" || -L "$formal_output" ]]; then
        if formal_sweep_valid "$formal_variant"; then
            echo "FORMAL4X5090_SWEEP_SKIP_VALID variant=$formal_variant output=$formal_output"
            return 0
        fi
        echo "FORMAL4X5090_POSTPROCESS_ABORT reason=existing_sweep_invalid variant=$formal_variant output=$formal_output" >&2
        return 1
    fi
    formal_gpu_healthy "$formal_gpu"
    formal_verify_sources
    echo "FORMAL4X5090_SWEEP_START variant=$formal_variant gpu_uuid=$formal_gpu"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$formal_gpu" \
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
    echo "FORMAL4X5090_SWEEP_COMPLETE variant=$formal_variant gpu_uuid=$formal_gpu"
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
    (
        cd "$formal_dir"
        sha256sum -c "$formal_stem.COMPLETE.sha256" >/dev/null
    ) || return 1
    jq -e \
        --arg formal_run_name "$formal_run_name" \
        --arg formal_aggregator_sha "482903040cbe9a58f17444eee45aeb67c6763a6aef99bd54892f47be5e21b42e" '
        .schema_version == "tpd-pd-fa-aggregate-v2" and
        .dataset == "NUDT-SIRST" and
        .run_name == $formal_run_name and
        .expected_epochs == 800 and
        .official_test_accessed == false and
        .selection_source == "internal_validation_only" and
        .checkpoint_role == "best_validation_pd_primary" and
        .mainline_decision_made == false and
        .aggregator_sha256 == $formal_aggregator_sha and
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

formal_extended_audit_valid() {
    local formal_dir="$1"
    local formal_payload="$formal_dir/extended_integrity_v1.json"
    local formal_marker="$formal_dir/EXTENDED_COMPLETE.sha256"
    [[ -s "$formal_payload" && ! -L "$formal_payload" ]] || return 1
    [[ -s "$formal_marker" && ! -L "$formal_marker" ]] || return 1
    cmp -s <(
        cd "$formal_dir"
        sha256sum COMPLETE.sha256 extended_integrity_v1.json
    ) "$formal_marker" || return 1
    (
        cd "$formal_dir"
        sha256sum -c EXTENDED_COMPLETE.sha256 >/dev/null
    ) || return 1
    jq -e '
        .schema == "sctransnet_formal800_4x5090_extended_integrity_v1" and
        .dataset == "NUDT-SIRST" and
        .expected_epochs == 800 and
        .official_test_accessed == false and
        .selection_source == "internal_validation_only" and
        .byte_identical_split_sha256 == "27c3b4a30c680af1c16493f723ce9713cb7e6987dcbc82e72fe1331cff12cd6b" and
        .independently_recomputed_shared_initialization_sha256 == "ae25925e8fffd9afe9fac1805389e80437f0d773ae744c979349a68886d81558" and
        (.per_variant | keys | sort) == ["original", "progressive", "spd", "tpd"] and
        ((.checks_passed | to_entries | map(.value == true)) | all)
    ' "$formal_payload" >/dev/null
}

formal_final_complete() {
    formal_validate_sweep_set && \
        formal_base_comparison_valid "$formal_comparison_dir" && \
        formal_extended_audit_valid "$formal_comparison_dir" && \
        formal_aggregate_valid "$formal_comparison_dir"
}

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

formal_static_preflight

if [[ "$formal_mode" == "--preflight" ]]; then
    for formal_index in "${!formal_variants[@]}"; do
        if formal_training_state "$formal_index"; then
            echo "FORMAL4X5090_POSTPROCESS_PREFLIGHT variant=${formal_variants[$formal_index]} state=complete"
        else
            formal_status=$?
            if [[ "$formal_status" -eq 1 ]]; then
                echo "FORMAL4X5090_POSTPROCESS_PREFLIGHT variant=${formal_variants[$formal_index]} state=running"
            else
                exit "$formal_status"
            fi
        fi
    done
    exit 0
fi

if [[ -e "$formal_comparison_dir" || -L "$formal_comparison_dir" ]]; then
    if formal_final_complete; then
        echo "FORMAL4X5090_POSTPROCESS_SKIP reason=already_complete comparison=$formal_comparison_dir"
        exit 0
    fi
    echo "FORMAL4X5090_POSTPROCESS_ABORT reason=existing_comparison_invalid path=$formal_comparison_dir" >&2
    exit 1
fi

formal_wait_count=0
while true; do
    formal_all_complete=1
    for formal_index in "${!formal_variants[@]}"; do
        if formal_training_state "$formal_index"; then
            continue
        else
            formal_status=$?
            if [[ "$formal_status" -eq 1 ]]; then
                formal_all_complete=0
            else
                exit "$formal_status"
            fi
        fi
    done
    if [[ "$formal_all_complete" -eq 1 ]]; then
        break
    fi
    if (( formal_wait_count % 20 == 0 )); then
        echo "FORMAL4X5090_POSTPROCESS_WAIT poll_seconds=$formal_poll_seconds"
        "$formal_repo/experiments/status_tpd_formal800_4x5090.sh"
        formal_static_preflight
    fi
    formal_wait_count=$((formal_wait_count + 1))
    sleep "$formal_poll_seconds"
done

formal_static_preflight
for formal_index in "${!formal_variants[@]}"; do
    formal_basic_run_complete "${formal_variants[$formal_index]}"
    formal_training_journal_complete "$formal_index"
done

echo "FORMAL4X5090_POSTPROCESS_TRAINING_COMPLETE variants=original,progressive,tpd,spd epochs=800"
formal_stage_dir="$(
    mktemp -d "$formal_dataset_root/.comparison_${formal_run_name}.staging.XXXXXX"
)"
echo "FORMAL4X5090_POSTPROCESS_STAGE path=$formal_stage_dir"

"$formal_python" experiments/summarize_tpd_pilot.py \
    --root "$formal_root" \
    --dataset NUDT-SIRST \
    --run-name "$formal_run_name" \
    --expected-epochs "$formal_expected_epochs" \
    --report-title "NUDT-SIRST TPD-PE formal 800-epoch 4xRTX5090 comparison" \
    --output-dir "$formal_stage_dir"

"$formal_python" experiments/audit_tpd_formal800_4x5090.py \
    --root "$formal_root" \
    --dataset NUDT-SIRST \
    --run-name "$formal_run_name" \
    --expected-epochs "$formal_expected_epochs" \
    --runtime-state "$formal_runtime_state" \
    --output "$formal_stage_dir/extended_integrity_v1.json"

formal_sweep_pids=()
for formal_variant in "${formal_variants[@]}"; do
    formal_run_sweep "$formal_variant" &
    formal_sweep_pids+=("$!")
done
formal_sweep_failed=0
for formal_index in "${!formal_sweep_pids[@]}"; do
    if ! wait "${formal_sweep_pids[$formal_index]}"; then
        echo "FORMAL4X5090_POSTPROCESS_ABORT reason=sweep_failed variant=${formal_variants[$formal_index]}" >&2
        formal_sweep_failed=1
    fi
done
[[ "$formal_sweep_failed" -eq 0 ]]

formal_validate_sweep_set
formal_static_preflight

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
(
    cd "$formal_stage_dir"
    sha256sum COMPLETE.sha256 extended_integrity_v1.json > EXTENDED_COMPLETE.sha256
    sha256sum -c EXTENDED_COMPLETE.sha256 >/dev/null
)
formal_extended_audit_valid "$formal_stage_dir"
formal_static_preflight
formal_validate_sweep_set

if [[ -e "$formal_comparison_dir" || -L "$formal_comparison_dir" ]]; then
    echo "FORMAL4X5090_POSTPROCESS_ABORT reason=comparison_appeared_during_run path=$formal_comparison_dir" >&2
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
echo "FORMAL4X5090_POSTPROCESS_COMPLETE comparison=$formal_comparison_dir"
