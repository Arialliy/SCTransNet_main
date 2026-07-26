#!/usr/bin/env bash
set -euo pipefail

v5_repo="/home/ly/SCTransNet_main"
v5_worker="$v5_repo/experiments/run_tpd_clean_v5_screen800_2x5090_worker.sh"
v5_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v5_result_root="$v5_repo/experiments/results/tpd_clean_v5_screen800_2x5090_v1"
v5_smoke_root="$v5_repo/experiments/results/tpd_clean_v5_preflight_v1"
v5_run_tag="screen800_pd_fp32_shared2x5090_v1"
v5_source_lock="$v5_repo/experiments/tpd_clean_v5_screen800_2x_source_lock.json"
v5_variants=(
    tpd_clean_v5_full
    tpd_clean_v5_sal_capacity
    tpd_clean_v5_full
    tpd_clean_v5_sal_capacity
)
v5_seeds=(42 42 3407 3407)
v5_gpu_uuids=(
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
)
v5_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)
v5_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v5_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v5_expected_jobs=(
    "tpd_clean_v5_full:42:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562:full-s42"
    "tpd_clean_v5_sal_capacity:42:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3:cap-s42"
    "tpd_clean_v5_full:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3:full-s3407"
    "tpd_clean_v5_sal_capacity:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562:cap-s3407"
)

if (( ${#v5_variants[@]} != 4 ||
      ${#v5_seeds[@]} != 4 ||
      ${#v5_gpu_uuids[@]} != 4 ||
      ${#v5_unit_tags[@]} != 4 )); then
    echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=invalid_job_array_lengths" >&2
    exit 1
fi

declare -A v5_gpu_job_counts=()
for v5_index in "${!v5_expected_jobs[@]}"; do
    v5_actual_job="${v5_variants[$v5_index]}:${v5_seeds[$v5_index]}:${v5_gpu_uuids[$v5_index]}:${v5_unit_tags[$v5_index]}"
    if [[ "$v5_actual_job" != "${v5_expected_jobs[$v5_index]}" ]]; then
        echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=counterbalanced_mapping_mismatch index=$v5_index expected=${v5_expected_jobs[$v5_index]} actual=$v5_actual_job" >&2
        exit 1
    fi
    v5_uuid="${v5_gpu_uuids[$v5_index]}"
    v5_count="${v5_gpu_job_counts[$v5_uuid]:-0}"
    v5_gpu_job_counts["$v5_uuid"]="$((v5_count + 1))"
done
if (( ${#v5_gpu_job_counts[@]} != 2 )) ||
    [[ "${v5_gpu_job_counts[$v5_gpu2_uuid]:-0}" -ne 2 ]] ||
    [[ "${v5_gpu_job_counts[$v5_gpu3_uuid]:-0}" -ne 2 ]]; then
    echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=invalid_gpu_multiplicity expected_gpu_count=2 expected_jobs_per_gpu=2" >&2
    exit 1
fi

v5_mode="${1:-run}"
if [[ "$v5_mode" != "run" && "$v5_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$v5_repo"

[[ -x "$v5_worker" ]] || {
    echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=worker_not_executable path=$v5_worker" >&2
    exit 1
}
[[ -x "$v5_python" ]] || {
    echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=python_not_executable path=$v5_python" >&2
    exit 1
}
[[ -f "$v5_source_lock" && ! -L "$v5_source_lock" ]] || {
    echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=missing_source_lock path=$v5_source_lock" >&2
    exit 1
}

"$v5_python" -c '
import cv2
import einops
import ml_collections
import numpy
import scipy
import skimage
import thop
import torch
from experiments import evaluate_tpd_clean_v5_pd_fa
from experiments import train_tpd_clean_v5
from model.tpd_clean_v5 import SUPPORTED_CLEAN_V5_VARIANTS

assert SUPPORTED_CLEAN_V5_VARIANTS == (
    "tpd_clean_v5_full",
    "tpd_clean_v5_sal_capacity",
)
assert torch.cuda.is_available()
'

"$v5_python" - "$v5_repo" "$v5_source_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v5_screen800_2x_source_lock_v1":
    raise SystemExit("invalid source-lock schema")
required_sources = {
    "model/tpd_clean_v5.py",
    "experiments/train_tpd_clean_v5.py",
    "experiments/evaluate_tpd_clean_v5_pd_fa.py",
    "experiments/smoke_tpd_clean_v5.py",
    "experiments/capture_tpd_clean_v5_smoke_report.py",
    "experiments/run_tpd_clean_v5_screen800_2x5090_worker.sh",
    "experiments/launch_tpd_clean_v5_screen800_2x5090.sh",
    "experiments/status_tpd_clean_v5_screen800_2x5090.sh",
    "experiments/TPD_CLEAN_V5_PROTOCOL.md",
    "experiments/TPD_CLEAN_V5_2GPU_PROTOCOL.md",
    "tests/test_tpd_clean_v5.py",
    "tests/test_train_tpd_clean_v5.py",
    "tests/test_evaluate_tpd_clean_v5_pd_fa.py",
    "tests/test_smoke_tpd_clean_v5.py",
    "tests/test_tpd_clean_v5_runner.py",
    "tests/test_tpd_clean_v5_2x_runtime.py",
    "experiments/train_tpd_pilot.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/fingerprint_tpd_training_data.py",
    "dataset.py",
    "utils.py",
    "warmup_scheduler.py",
    "model/SCTransNet.py",
    "model/Config.py",
    "model/tpd.py",
    "experiments/smoke_tpd_clean_v3.py",
    "experiments/train_tpd_clean_v3.py",
    "model/tpd_clean_v3.py",
    "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
    "experiments/tpd_clean_v3_screen800_source_lock.json",
    "experiments/tpd_clean_screen800_source_lock.json",
    "experiments/tpd_ner_v1_source_lock.json",
}
if set(payload.get("source_sha256", {})) != required_sources:
    raise SystemExit("source-lock path set differs")
for relative, expected in payload["source_sha256"].items():
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked source: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"source digest mismatch: {relative} "
            f"expected={expected} actual={actual}"
        )
print(
    f"TPDCLEANV5_2X_PREFLIGHT_SOURCES_OK files={len(payload['source_sha256'])}"
)
PY

"$v5_python" - "$v5_repo" "$v5_source_lock" "$v5_smoke_root" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
smoke_root = pathlib.Path(sys.argv[3])
payload = json.loads(lock_path.read_text(encoding="utf-8"))
expected_reports = {
    "cpu_all.json": {
        "device": "cpu",
        "device_name": "cpu",
        "cuda_visible_devices": None,
        "variants": [
            "tpd_clean_v5_full",
            "tpd_clean_v5_sal_capacity",
        ],
    },
    "gpu2_full.json": {
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "cuda_visible_devices": "2",
        "variants": ["tpd_clean_v5_full"],
    },
    "gpu3_capacity.json": {
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "cuda_visible_devices": "3",
        "variants": ["tpd_clean_v5_sal_capacity"],
    },
}
if set(payload.get("smoke_sha256", {})) != set(expected_reports):
    raise SystemExit("source lock does not bind the three smoke reports")
for name, expected in expected_reports.items():
    path = smoke_root / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked smoke report: {path}")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != payload["smoke_sha256"][name]:
        raise SystemExit(f"smoke report digest mismatch: {name}")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if (
        envelope.get("schema")
        != "sctransnet_tpd_clean_v5_persisted_smoke_v1"
        or envelope.get("status") != "complete"
        or envelope.get("cuda_visible_devices")
        != expected["cuda_visible_devices"]
    ):
        raise SystemExit(f"invalid smoke envelope: {name}")
    for source_key, relative in (
        ("model_source_sha256", "model/tpd_clean_v5.py"),
        ("train_source_sha256", "experiments/train_tpd_clean_v5.py"),
        ("smoke_source_sha256", "experiments/smoke_tpd_clean_v5.py"),
        (
            "capture_source_sha256",
            "experiments/capture_tpd_clean_v5_smoke_report.py",
        ),
    ):
        if envelope.get(source_key) != payload["source_sha256"][relative]:
            raise SystemExit(f"{name}: {source_key} differs from source lock")
    report = envelope.get("report", {})
    if (
        report.get("schema") != "sctransnet_tpd_clean_v5_smoke_v1"
        or report.get("status") != "complete"
        or report.get("device") != expected["device"]
        or report.get("device_name") != expected["device_name"]
        or report.get("paired_initialization") is not True
        or report.get("steps") != 2
    ):
        raise SystemExit(f"invalid smoke report contract: {name}")
    variants = report.get("variants", [])
    if [item.get("variant") for item in variants] != expected["variants"]:
        raise SystemExit(f"smoke variant matrix differs: {name}")
    for item in variants:
        evidence_maps = [
            item.get("scale_gradient_l1", {}),
            item.get("scale_update_l1", {}),
            item.get("phase_gradient_l1", {}),
            item.get("phase_update_l1", {}),
        ]
        evidence_values_valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for evidence in evidence_maps
            for value in evidence.values()
        )
        losses = item.get("losses", [])
        losses_valid = (
            len(losses) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in losses
            )
        )
        if (
            item.get("status") != "complete"
            or item.get("output_count") != 6
            or not losses_valid
            or item.get("step_zero_exact_spd") is not True
            or item.get("strict_rebuild_load") is not True
            or item.get("strict_reload_max_abs_difference") != 0.0
            or len(item.get("scale_gradient_l1", {})) != 7
            or len(item.get("scale_update_l1", {})) != 7
            or len(item.get("phase_gradient_l1", {})) != 14
            or len(item.get("phase_update_l1", {})) != 14
            or not evidence_values_valid
        ):
            raise SystemExit(f"incomplete smoke variant evidence: {name}")
print("TPDCLEANV5_2X_PREFLIGHT_SMOKE_OK reports=3", flush=True)
PY

for v5_uuid in "$v5_gpu2_uuid" "$v5_gpu3_uuid"; do
    v5_actual_index="$(
        nvidia-smi -i "$v5_uuid" \
            --query-gpu=index --format=csv,noheader,nounits
    )"
    v5_actual_name="$(
        nvidia-smi -i "$v5_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    v5_free_memory="$(
        nvidia-smi -i "$v5_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$v5_uuid" == "$v5_gpu2_uuid" && "$v5_actual_index" != "2" ]] ||
        [[ "$v5_uuid" == "$v5_gpu3_uuid" && "$v5_actual_index" != "3" ]]; then
        echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=gpu_index_mismatch gpu_uuid=$v5_uuid index=$v5_actual_index" >&2
        exit 1
    fi
    if [[ "$v5_actual_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=gpu_mismatch gpu_uuid=$v5_uuid name=$v5_actual_name" >&2
        exit 1
    fi
    if (( v5_free_memory < 15000 )); then
        echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=insufficient_memory_for_two_jobs gpu_uuid=$v5_uuid free_mib=$v5_free_memory required_mib=15000" >&2
        exit 1
    fi
done

for v5_index in "${!v5_variants[@]}"; do
    v5_variant="${v5_variants[$v5_index]}"
    v5_seed="${v5_seeds[$v5_index]}"
    v5_tag="${v5_unit_tags[$v5_index]}"
    v5_run_name="seed_${v5_seed}_${v5_run_tag}"
    v5_run_dir="$v5_result_root/NUDT-SIRST/$v5_variant/$v5_run_name"
    if [[ -e "$v5_run_dir" || -L "$v5_run_dir" ]]; then
        echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=run_path_not_fresh job=$v5_tag path=$v5_run_dir" >&2
        exit 1
    fi

    v5_unit="sctransnet-tpd-clean-v5-2x-$v5_tag.service"
    if systemctl --user cat "$v5_unit" >/dev/null 2>&1; then
        echo "TPDCLEANV5_2X_LAUNCH_ABORT reason=unit_already_exists unit=$v5_unit" >&2
        exit 1
    fi
done

echo "TPDCLEANV5_2X_PREFLIGHT_OK jobs=full-s42,cap-s42,full-s3407,cap-s3407 gpus=2,3 concurrent_jobs_per_gpu=2 threads_per_job=1 counterbalanced_mapping=true"
if [[ "$v5_mode" == "--preflight" ]]; then
    exit 0
fi

mkdir -p "$v5_result_root"
for v5_index in "${!v5_variants[@]}"; do
    v5_variant="${v5_variants[$v5_index]}"
    v5_seed="${v5_seeds[$v5_index]}"
    v5_uuid="${v5_gpu_uuids[$v5_index]}"
    v5_tag="${v5_unit_tags[$v5_index]}"
    v5_unit="sctransnet-tpd-clean-v5-2x-$v5_tag"
    systemd-run --user \
        --unit="$v5_unit" \
        --description="SCTransNet TPD-Clean-v5 2GPU $v5_variant seed $v5_seed" \
        --property=Restart=no \
        --property=TimeoutStopSec=120 \
        /usr/bin/bash "$v5_worker" "$v5_variant" "$v5_seed" "$v5_uuid"
    echo "TPDCLEANV5_2X_UNIT_STARTED variant=$v5_variant seed=$v5_seed gpu_uuid=$v5_uuid unit=$v5_unit.service"
done
