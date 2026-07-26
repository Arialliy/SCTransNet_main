#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 <gpu-uuid> <seed> <run-tag>" >&2
    exit 2
fi

ner_gpu_uuid="$1"
ner_seed="$2"
ner_run_tag="$3"

case "$ner_gpu_uuid" in
    GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70) ;;
    GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640) ;;
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562) ;;
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3) ;;
    *)
        echo "TPDNER_FORMAL800_ABORT reason=invalid_gpu_uuid actual=$ner_gpu_uuid" >&2
        exit 2
        ;;
esac
if [[ ! "$ner_seed" =~ ^(0|[1-9][0-9]*)$ ]] || (( ner_seed > 4294967295 )); then
    echo "TPDNER_FORMAL800_ABORT reason=invalid_seed expected=0..4294967295 actual=$ner_seed" >&2
    exit 2
fi
if [[ ! "$ner_run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=invalid_run_tag actual=$ner_run_tag" >&2
    exit 2
fi

ner_repo="/home/ly/SCTransNet_main"
ner_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
ner_variant="tpd_clean_full_ner"
ner_dataset="NUDT-SIRST"
ner_dataset_dir="$ner_repo/datasets"
ner_output_root="$ner_repo/experiments/results/tpd_ner_v1_formal800"
ner_run_dir="$ner_output_root/$ner_dataset/$ner_variant/seed_${ner_seed}_${ner_run_tag}"
ner_train_entry="$ner_repo/experiments/train_tpd_ner_v1.py"
ner_evaluate_entry="$ner_repo/experiments/evaluate_tpd_ner_v1_pd_fa.py"
ner_smoke_entry="$ner_repo/experiments/smoke_tpd_ner_v1.py"
ner_source_lock="$ner_repo/experiments/tpd_ner_v1_source_lock.json"
ner_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
ner_gate_definition="$ner_repo/experiments/tpd_clean_next_module_gate_v1.json"
ner_clean_root="$ner_repo/experiments/results/tpd_clean_screen800_4x5090_v1"
ner_finalizer_state="$ner_clean_root/launch/finalizer_state.json"
ner_comparison_dir="$ner_clean_root/NUDT-SIRST/comparison"
ner_comparison_json="$ner_comparison_dir/tpd_clean_screen800_comparison_seed42.json"
ner_comparison_marker="$ner_comparison_dir/tpd_clean_screen800_comparison_seed42.COMPLETE.sha256"
ner_lock_root="$ner_output_root/.locks"

if [[ ! -x "$ner_python" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=python_not_executable path=$ner_python" >&2
    exit 1
fi
if [[ ! -f "$ner_train_entry" || -L "$ner_train_entry" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=invalid_train_entry path=$ner_train_entry" >&2
    exit 1
fi
if [[ ! -f "$ner_evaluate_entry" || -L "$ner_evaluate_entry" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=invalid_evaluate_entry path=$ner_evaluate_entry" >&2
    exit 1
fi
if [[ ! -f "$ner_smoke_entry" || -L "$ner_smoke_entry" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=invalid_smoke_entry path=$ner_smoke_entry" >&2
    exit 1
fi
if [[ ! -d "$ner_dataset_dir/$ner_dataset" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=dataset_missing path=$ner_dataset_dir/$ner_dataset" >&2
    exit 1
fi
if [[ -e "$ner_run_dir" || -L "$ner_run_dir" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=run_path_exists path=$ner_run_dir" >&2
    exit 1
fi

ner_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

ner_verify_sources() {
    "$ner_python" - "$ner_repo" "$ner_source_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
payload = json.loads(lock_path.read_text(encoding="utf-8"))
sources = payload.get("source_sha256")
if payload.get("schema") != "sctransnet_tpd_ner_v1_source_lock_v1":
    raise SystemExit("invalid NER source-lock schema")
if not isinstance(sources, dict) or not sources:
    raise SystemExit("NER source-lock has no sources")
for relative, expected in sources.items():
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked NER source: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"NER source mismatch: {relative} expected={expected} actual={actual}"
        )
print(f"TPDNER_SOURCES_OK files={len(sources)}", flush=True)
PY
}

ner_verify_data() {
    local ner_actual
    ner_actual="$(
        timeout 300s "$ner_python" experiments/fingerprint_tpd_training_data.py --dataset "$ner_dataset"
    )"
    if [[ "$ner_actual" != "$ner_training_data_sha256" ]]; then
        echo "TPDNER_FORMAL800_ABORT reason=training_data_changed expected=$ner_training_data_sha256 actual=$ner_actual" >&2
        return 1
    fi
}

ner_verify_launch_gate() {
    local ner_required
    local ner_gate_sha
    for ner_required in \
        "$ner_finalizer_state" \
        "$ner_comparison_json" \
        "$ner_comparison_marker" \
        "$ner_gate_definition" \
        "$ner_source_lock"; do
        if [[ ! -f "$ner_required" || -L "$ner_required" ]]; then
            echo "TPDNER_FORMAL800_ABORT reason=launch_gate_artifact_missing path=$ner_required" >&2
            return 1
        fi
    done

    if ! jq -e '
            .schema == "sctransnet_tpd_clean_screen800_finalizer_state_v1" and
            .state == "complete" and
            .all_unique_completions_observed == true and
            all(.completion_evidence[]; .ready == true)
        ' "$ner_finalizer_state" >/dev/null; then
        echo "TPDNER_FORMAL800_ABORT reason=clean_v2_finalizer_not_complete" >&2
        return 1
    fi

    if ! (cd "$ner_comparison_dir" && sha256sum -c "$ner_comparison_marker" >/dev/null); then
        echo "TPDNER_FORMAL800_ABORT reason=invalid_clean_v2_comparison_marker" >&2
        return 1
    fi

    ner_gate_sha="$(ner_sha256 "$ner_gate_definition")"
    if ! jq -e --arg ner_gate_sha "$ner_gate_sha" '
            .schema == "sctransnet_tpd_clean_screen800_comparison_v2" and
            .next_module_gate.formal_module_launch_gate_passed == true and
            .next_module_gate.gate_file_sha256 == $ner_gate_sha and
            .next_module_gate.mainline_changed == false and
            .next_module_gate.innovation_changed == false
        ' "$ner_comparison_json" >/dev/null; then
        echo "TPDNER_FORMAL800_ABORT reason=next_module_gate_not_passed" >&2
        return 1
    fi
}

ner_verify_gpu() {
    local ner_compute_pids
    local ner_gpu_name
    local ner_free_mib
    ner_gpu_name="$(
        nvidia-smi -i "$ner_gpu_uuid" --query-gpu=name --format=csv,noheader,nounits
    )"
    ner_free_mib="$(
        nvidia-smi -i "$ner_gpu_uuid" --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$ner_gpu_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDNER_FORMAL800_ABORT reason=unexpected_gpu_name gpu_uuid=$ner_gpu_uuid name=$ner_gpu_name" >&2
        return 1
    fi
    ner_compute_pids="$(
        nvidia-smi -i "$ner_gpu_uuid" \
            --query-compute-apps=pid \
            --format=csv,noheader,nounits |
            awk 'NF {printf "%s%s", separator, $1; separator=","}'
    )"
    if [[ -n "$ner_compute_pids" ]]; then
        echo "TPDNER_FORMAL800_ABORT reason=gpu_has_compute_processes gpu_uuid=$ner_gpu_uuid pids=$ner_compute_pids" >&2
        return 1
    fi
    if (( ner_free_mib < 28000 )); then
        echo "TPDNER_FORMAL800_ABORT reason=insufficient_free_memory gpu_uuid=$ner_gpu_uuid free_mib=$ner_free_mib" >&2
        return 1
    fi
}

cd "$ner_repo"
ner_verify_launch_gate
ner_verify_sources
ner_verify_data
mkdir -p "$ner_lock_root"
exec 9>"$ner_lock_root/$ner_gpu_uuid.lock"
if ! flock -n 9; then
    echo "TPDNER_FORMAL800_ABORT reason=gpu_lock_held gpu_uuid=$ner_gpu_uuid" >&2
    exit 1
fi
ner_verify_gpu
ner_comparison_sha256="$(ner_sha256 "$ner_comparison_json")"
ner_comparison_marker_sha256="$(ner_sha256 "$ner_comparison_marker")"
ner_source_lock_sha256="$(ner_sha256 "$ner_source_lock")"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$ner_seed"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$ner_gpu_uuid"

ner_smoke_json="$(
    "$ner_python" "$ner_smoke_entry" \
        --device cuda:0 \
        --batch-size 16 \
        --patch-size 256 \
        --steps 2 \
        --seed "$ner_seed" \
        --learning-rate 0.001 \
        --expected-device-name "NVIDIA GeForce RTX 5090"
)"
if ! jq -e '
        .schema == "sctransnet_tpd_ner_v1_two_step_smoke_v1" and
        .status == "complete" and
        .variant == "tpd_clean_full_ner" and
        .device == "cuda:0" and
        .device_name == "NVIDIA GeForce RTX 5090" and
        .batch_size == 16 and
        .patch_size == 256 and
        .steps == 2 and
        .output_count == 6 and
        .strict_rebuild_load == true and
        .relay_parameters == 11291 and
        .total_parameters == 10854766 and
        (.losses | length) == 2 and
        all(.losses[]; isfinite and . > 0) and
        all(["2", "3", "4"][]; . as $stage |
            ($root.gate_gradient_l1[$stage] | isfinite and . > 0) and
            ($root.gate_update_l1[$stage] | isfinite and . > 0) and
            ($root.fusion_gradient_l1[$stage] | isfinite and . > 0) and
            ($root.fusion_update_l1[$stage] | isfinite and . > 0)
        ) and
        (.tpd_scale_update_l1 | isfinite and . > 0) and
        (.cuda_memory.peak_allocated_mib | isfinite and . > 0) and
        (.cuda_memory.peak_reserved_mib | isfinite and . > 0)
    ' --argjson root "$ner_smoke_json" <<<"$ner_smoke_json" >/dev/null; then
    echo "TPDNER_FORMAL800_ABORT reason=invalid_two_step_smoke_result" >&2
    exit 1
fi
ner_smoke_peak_reserved_mib="$(
    jq -r '.cuda_memory.peak_reserved_mib | ceil' <<<"$ner_smoke_json"
)"
ner_post_smoke_free_mib="$(
    nvidia-smi -i "$ner_gpu_uuid" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits
)"
ner_required_free_mib="$((ner_smoke_peak_reserved_mib + 4096))"
if (( ner_post_smoke_free_mib < ner_required_free_mib )); then
    echo "TPDNER_FORMAL800_ABORT reason=insufficient_post_smoke_headroom gpu_uuid=$ner_gpu_uuid free_mib=$ner_post_smoke_free_mib required_mib=$ner_required_free_mib peak_reserved_mib=$ner_smoke_peak_reserved_mib" >&2
    exit 1
fi
echo "TPDNER_TWO_STEP_SMOKE_COMPLETE $ner_smoke_json"
ner_verify_sources
ner_verify_data
ner_verify_launch_gate

"$ner_python" "$ner_train_entry" \
    --variant "$ner_variant" \
    --dataset "$ner_dataset" \
    --dataset-dir "$ner_dataset_dir" \
    --output-root "$ner_output_root" \
    --run-tag "$ner_run_tag" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed "$ner_seed" \
    --split-seed 20260722 \
    --val-fraction 0.20 \
    --eval-every 1 \
    --base-lr 0.001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --threshold 0.5 \
    --match-radius 3 \
    --tiny-area 9

ner_verify_sources
ner_verify_data
ner_verify_launch_gate
if [[ "$(ner_sha256 "$ner_source_lock")" != "$ner_source_lock_sha256" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=source_lock_changed_during_training" >&2
    exit 1
fi
if [[ "$(ner_sha256 "$ner_comparison_json")" != "$ner_comparison_sha256" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=launch_comparison_changed_during_training" >&2
    exit 1
fi
if [[ "$(ner_sha256 "$ner_comparison_marker")" != "$ner_comparison_marker_sha256" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=launch_marker_changed_during_training" >&2
    exit 1
fi

if [[ "$(wc -l < "$ner_run_dir/metrics.jsonl")" -ne 800 ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=incomplete_metrics path=$ner_run_dir/metrics.jsonl" >&2
    exit 1
fi

jq -e \
    --arg ner_variant "$ner_variant" \
    --arg ner_dataset "$ner_dataset" \
    --argjson ner_seed "$ner_seed" '
        .status == "complete" and
        .variant == $ner_variant and
        .dataset == $ner_dataset and
        .seed == $ner_seed and
        .selection_source == "internal_validation_only" and
        .official_test_accessed == false
    ' "$ner_run_dir/summary.json" >/dev/null

jq -e '
        .arguments.epochs == 800 and
        .arguments.batch_size == 16 and
        .arguments.patch_size == 256 and
        .arguments.workers == 0 and
        .arguments.split_seed == 20260722 and
        .arguments.val_fraction == 0.2 and
        .arguments.eval_every == 1 and
        .arguments.base_lr == 0.001 and
        .arguments.min_lr == 0.00001 and
        .arguments.warmup_epochs == 10 and
        .arguments.threshold == 0.5 and
        .arguments.match_radius == 3 and
        .arguments.tiny_area == 9 and
        .arguments.amp == false
    ' "$ner_run_dir/protocol.json" >/dev/null

for ner_checkpoint in best.pth.tar best_miou.pth.tar; do
    "$ner_python" "$ner_evaluate_entry" \
        --run-dir "$ner_run_dir" \
        --checkpoint "$ner_checkpoint" \
        --device cuda:0 \
        --expected-epochs 800
done

ner_verify_sources
ner_verify_data
ner_verify_launch_gate
if [[ "$(ner_sha256 "$ner_source_lock")" != "$ner_source_lock_sha256" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=source_lock_changed_during_evaluation" >&2
    exit 1
fi
if [[ "$(ner_sha256 "$ner_comparison_json")" != "$ner_comparison_sha256" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=launch_comparison_changed_during_evaluation" >&2
    exit 1
fi
if [[ "$(ner_sha256 "$ner_comparison_marker")" != "$ner_comparison_marker_sha256" ]]; then
    echo "TPDNER_FORMAL800_ABORT reason=launch_marker_changed_during_evaluation" >&2
    exit 1
fi

echo "TPDNER_FORMAL800_COMPLETE gpu_uuid=$ner_gpu_uuid seed=$ner_seed run_tag=$ner_run_tag epochs=800 run_dir=$ner_run_dir"
