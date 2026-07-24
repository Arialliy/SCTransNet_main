#!/usr/bin/env bash
set -euo pipefail

formal_repo="/home/ly/SCTransNet_main"
formal_python="/home/ly/MSHNet/.venv/bin/python"
formal_root="$formal_repo/experiments/results/tpd_pe_formal800_v1"
formal_dataset_root="$formal_root/NUDT-SIRST"
formal_run_name="seed_42_formal800_pd_fp32_v1"
formal_gpu2_uuid="GPU-3509f974-0eba-2bcb-9469-372003ae4f0a"
formal_poll_seconds=60
formal_wait_log_every=10
formal_lock="$formal_root/.gpu2_recovery_s42_v1.lock"
formal_archive="$formal_root/recovery_archive/20260723_112306_CST_gpu1_launch_failure"
formal_tpd_dir="$formal_dataset_root/tpd/$formal_run_name"
formal_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
formal_variants=(original progressive spd)
formal_all_variants=(original progressive spd tpd)

cd "$formal_repo"

exec 9>"$formal_lock"
if ! flock -n 9; then
    echo "GPU2_RECOVERY_ABORT reason=lock_held lock=$formal_lock" >&2
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
        echo "GPU2_RECOVERY_ABORT reason=missing_or_symlink path=$formal_path" >&2
        return 1
    }
    formal_actual="$(formal_sha256 "$formal_path")"
    if [[ "$formal_actual" != "$formal_expected" ]]; then
        echo "GPU2_RECOVERY_ABORT reason=sha_mismatch path=$formal_path expected=$formal_expected actual=$formal_actual" >&2
        return 1
    fi
}

formal_verify_frozen_sources() {
    formal_require_sha256 experiments/run_tpd_formal800_queue.sh a1c443770332daf1d9615f1f5b2756dd695749b808554bc0698380ba7eef66b7
    formal_require_sha256 experiments/train_tpd_pilot.py 7532bdc3bcc777aa164e258ab21f78d38ed3a1eaa677a29c8256d900224a7f26
    formal_require_sha256 experiments/fingerprint_tpd_training_data.py 26382e38e899bdf4f97b77c6671929c391decef6e4bf4ac40094a7d4e6b0bc7d
    formal_require_sha256 experiments/summarize_tpd_pilot.py 49417a3fd43192e52308e7dc4343527bcda71495b78257c5464ba43d8eef7f3e
    formal_require_sha256 dataset.py 516ea9c410f80cc9ae912cf0443126a067dd14b6cc5ad7945e83cfc497f4678d
    formal_require_sha256 utils.py afb6fc221072ddd082b53ccda132232bc9089afd0458d8f0e47a39b9c1e25c13
    formal_require_sha256 model/SCTransNet.py 5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb
    formal_require_sha256 model/tpd.py 18a5892edd18ab040e38f18c8d86a02bf3e50b7a4d12d0115ec9a97e8051c135
    formal_require_sha256 model/Config.py b7e3e67c379ef4638605ebe612336b0c3cdb1a97f4d6fe731dec80b4847d5596
}

formal_verify_tpd_certificate() {
    formal_require_sha256 "$formal_tpd_dir/protocol.json" 6b974f73763aaa0d50ab666ea0e2ce7cd4ab526752082b4c2a64480603cf4773
    formal_require_sha256 "$formal_tpd_dir/split.json" 27c3b4a30c680af1c16493f723ce9713cb7e6987dcbc82e72fe1331cff12cd6b
    formal_require_sha256 "$formal_tpd_dir/metrics.jsonl" 0bd7e49bbe1d26e9592af39408a2798210d155852b0be4fcb6f8786ab38b464c
    formal_require_sha256 "$formal_tpd_dir/best.pth.tar" 9487fc82c3a2a8e6ab26ffb30c4b4cd6d6e3fb15d2ffbb7d0d77ef566ce78fa5
    formal_require_sha256 "$formal_tpd_dir/best_miou.pth.tar" 58e6171c7d7ccf9265c555371a7a92415d1bff391946ec2098e030b9855b7353
    formal_require_sha256 "$formal_tpd_dir/last.pth.tar" 4ed28c76ea51348b3f88993a061f1be4b5b2f5c53cb13ca88052f47fe6fe1fbe
    formal_require_sha256 "$formal_tpd_dir/summary.json" 0bb284cc87f8e1aa21d88bac7452d6cf4b7dee5ba94568d431dfd46577810a67

    [[ "$(wc -l < "$formal_tpd_dir/metrics.jsonl")" -eq 800 ]]
    jq -e '
        .status == "complete" and
        .variant == "tpd" and
        .dataset == "NUDT-SIRST" and
        .seed == 42 and
        .best_pd_epoch == 289 and
        .selection_source == "internal_validation_only" and
        .official_test_accessed == false
    ' "$formal_tpd_dir/summary.json" >/dev/null
}

formal_verify_recovery_archive() {
    formal_require_sha256 \
        "$formal_archive/FILES.relocated.sha256" \
        822a154e5cec2092b2c3d32a967be8bd55add90e5b1871bc4f439bd0af57031b
    (
        cd "$formal_archive"
        sha256sum -c FILES.relocated.sha256 >/dev/null
    )
}

formal_verify_training_data() {
    local formal_actual
    formal_actual="$(
        timeout 300s "$formal_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    if [[ "$formal_actual" != "$formal_training_data_sha256" ]]; then
        echo "GPU2_RECOVERY_ABORT reason=training_data_drift expected=$formal_training_data_sha256 actual=$formal_actual" >&2
        return 1
    fi
}

formal_require_fresh_path() {
    local formal_variant="$1"
    local formal_path
    formal_path="$formal_dataset_root/$formal_variant/$formal_run_name"
    if [[ -e "$formal_path" || -L "$formal_path" ]]; then
        echo "GPU2_RECOVERY_ABORT reason=run_path_not_fresh variant=$formal_variant path=$formal_path" >&2
        return 1
    fi
}

formal_require_all_fresh_paths() {
    local formal_variant
    for formal_variant in "${formal_variants[@]}"; do
        formal_require_fresh_path "$formal_variant"
    done
}

formal_audit_completed_variant() {
    local formal_variant="$1"
    local formal_dir="$formal_dataset_root/$formal_variant/$formal_run_name"
    local formal_artifact
    for formal_artifact in protocol.json split.json metrics.jsonl best.pth.tar best_miou.pth.tar last.pth.tar summary.json; do
        [[ -s "$formal_dir/$formal_artifact" && ! -L "$formal_dir/$formal_artifact" ]] || {
            echo "GPU2_RECOVERY_ABORT reason=incomplete_variant_artifact variant=$formal_variant artifact=$formal_artifact" >&2
            return 1
        }
    done
    formal_require_sha256 \
        "$formal_dir/split.json" \
        27c3b4a30c680af1c16493f723ce9713cb7e6987dcbc82e72fe1331cff12cd6b
    timeout 1200s env CUDA_VISIBLE_DEVICES="" "$formal_python" -c '
import sys
from pathlib import Path
from experiments.summarize_tpd_pilot import (
    audit_checkpoint,
    build_model_for_strict_load,
    load_metrics,
    load_run,
)

run_dir = Path(sys.argv[1])
variant = sys.argv[2]
run = load_run(run_dir, 800, variant, "NUDT-SIRST")
evaluated = load_metrics(
    run_dir / "metrics.jsonl",
    800,
    variant,
    int(run["arguments"]["eval_every"]),
)
last_epoch, last_metrics = evaluated[-1]
if last_epoch != 800:
    raise ValueError(f"last evaluated epoch is {last_epoch}, expected 800")
audit_checkpoint(
    run_dir / "last.pth.tar",
    build_model_for_strict_load(variant),
    "last_evaluated_epoch",
    800,
    last_metrics,
    variant,
    "NUDT-SIRST",
    int(run["summary"]["seed"]),
    int(run["arguments"]["split_seed"]),
    run["summary"]["split_hashes"],
    run["summary"]["model"],
)
recorded_last = run["summary"].get("last_checkpoint")
if not isinstance(recorded_last, str):
    raise ValueError("summary.last_checkpoint must be a path string")
if Path(recorded_last).resolve() != (run_dir / "last.pth.tar").resolve():
    raise ValueError("summary.last_checkpoint does not point to this run last.pth.tar")
' "$formal_dir" "$formal_variant"
    jq -e --arg formal_variant "$formal_variant" '
        .status == "complete" and
        .variant == $formal_variant and
        .dataset == "NUDT-SIRST" and
        .seed == 42 and
        .selection_source == "internal_validation_only" and
        .official_test_accessed == false and
        .model.shared_initialization_sha256 == "ae25925e8fffd9afe9fac1805389e80437f0d773ae744c979349a68886d81558"
    ' "$formal_dir/summary.json" >/dev/null
    echo "GPU2_RECOVERY_AUDIT_COMPLETE variant=$formal_variant epochs=800"
}

formal_audit_cross_variant_consistency() {
    timeout 1800s env CUDA_VISIBLE_DEVICES="" "$formal_python" -c '
import sys
from pathlib import Path
from experiments.summarize_tpd_pilot import (
    audit_cross_variant_consistency,
    load_run,
)

dataset_root = Path(sys.argv[1])
run_name = sys.argv[2]
variants = ("original", "progressive", "spd", "tpd")
runs = {
    variant: load_run(
        dataset_root / variant / run_name,
        800,
        variant,
        "NUDT-SIRST",
    )
    for variant in variants
}
audit_cross_variant_consistency(runs)
' "$formal_dataset_root" "$formal_run_name"
    echo "GPU2_RECOVERY_CROSS_AUDIT_COMPLETE variants=original,progressive,spd,tpd epochs=800"
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
import sys
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
            --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null
    )"; then
        return 1
    fi
    [[ -z "$formal_pids" ]]
}

formal_static_preflight() {
    formal_verify_frozen_sources
    formal_verify_tpd_certificate
    formal_verify_recovery_archive
    formal_verify_training_data
}

formal_wait_for_gpu2() {
    local formal_poll_count=0
    while true; do
        if formal_gpu2_visible && formal_gpu2_cuda_healthy && formal_gpu2_idle; then
            break
        fi
        formal_poll_count=$((formal_poll_count + 1))
        if (( formal_poll_count == 1 || formal_poll_count % formal_wait_log_every == 0 )); then
            echo "GPU2_RECOVERY_WAIT cuda_healthy=0_or_busy gpu_uuid=$formal_gpu2_uuid poll_seconds=$formal_poll_seconds"
        fi
        sleep "$formal_poll_seconds"
    done
}

formal_prepare_variant_launch() {
    local formal_variant="$1"
    while true; do
        formal_wait_for_gpu2
        formal_static_preflight
        formal_require_fresh_path "$formal_variant"
        if formal_gpu2_visible && formal_gpu2_cuda_healthy && formal_gpu2_idle; then
            return 0
        fi
        echo "GPU2_RECOVERY_READY_RETRY variant=$formal_variant reason=health_changed_during_preflight"
    done
}

formal_mode="${1:-run}"
if [[ "$formal_mode" != "run" && "$formal_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

formal_static_preflight
formal_require_all_fresh_paths

if [[ "$formal_mode" == "--preflight" ]]; then
    if formal_gpu2_visible && formal_gpu2_cuda_healthy && formal_gpu2_idle; then
        echo "GPU2_RECOVERY_PREFLIGHT cuda_healthy=1 gpu_uuid=$formal_gpu2_uuid"
    else
        echo "GPU2_RECOVERY_PREFLIGHT cuda_healthy=0 gpu_uuid=$formal_gpu2_uuid"
    fi
    exit 0
fi

echo "GPU2_RECOVERY_ARMED variants=original,progressive,spd gpu_uuid=$formal_gpu2_uuid"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$formal_gpu2_uuid"
for formal_variant in "${formal_variants[@]}"; do
    formal_static_preflight
    formal_require_fresh_path "$formal_variant"
    formal_prepare_variant_launch "$formal_variant"
    if [[ "$formal_variant" == "original" ]]; then
        echo "GPU2_RECOVERY_START variants=original,progressive,spd gpu_uuid=$formal_gpu2_uuid"
    fi
    echo "GPU2_RECOVERY_VARIANT_START variant=$formal_variant gpu_uuid=$formal_gpu2_uuid"
    experiments/run_tpd_formal800_queue.sh "$formal_variant"
    formal_verify_frozen_sources
    formal_verify_tpd_certificate
    formal_verify_recovery_archive
    formal_verify_training_data
    formal_audit_completed_variant "$formal_variant"
    echo "GPU2_RECOVERY_VARIANT_COMPLETE variant=$formal_variant gpu_uuid=$formal_gpu2_uuid"
done
formal_static_preflight
for formal_variant in "${formal_all_variants[@]}"; do
    formal_audit_completed_variant "$formal_variant"
done
formal_audit_cross_variant_consistency
formal_static_preflight
echo "GPU2_RECOVERY_COMPLETE variants=original,progressive,spd gpu_uuid=$formal_gpu2_uuid"
