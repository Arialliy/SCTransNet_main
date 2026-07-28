#!/usr/bin/env bash
set -euo pipefail

ner_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    ner_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 [--preflight] VARIANT PHYSICAL_GPU_INDEX GPU_UUID" >&2
    exit 2
fi

ner_variant="$1"
ner_physical_index="$2"
ner_gpu_uuid="$3"
ner_repo="${TPD_NER_V8_REPO:-/home/ly/SCTransNet_main}"
ner_python="${TPD_NER_V8_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
ner_trainer="$ner_repo/experiments/train_tpd_ner_v8_mprs_dch_exact.py"
ner_manifest_tool="$ner_repo/experiments/freeze_tpd_ner_v8_mprs_dch_source_locks.py"
ner_source_lock="${TPD_NER_V8_SOURCE_LOCK:-$ner_repo/experiments/tpd_ner_v8_mprs_dch_exact_source_lock.json}"
ner_result_root="${TPD_NER_V8_RESULT_ROOT:-$ner_repo/experiments/results/tpd_ner_v8_mprs_dch_exact_v1}"
ner_run_tag="formal800_exact_v1"
ner_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
ner_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

case "$ner_variant:$ner_physical_index:$ner_gpu_uuid" in
    tpd_ner_v8_mprs_dch_full_relay_off:2:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        ;;
    tpd_ner_v8_mprs_dch_full_relay_on:3:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        ;;
    *)
        echo "TPDNERV8_LANE_ABORT reason=invalid_variant_gpu_mapping variant=$ner_variant physical_gpu=$ner_physical_index gpu_uuid=$ner_gpu_uuid" >&2
        exit 2
        ;;
esac

[[ -x "$ner_python" ]] || {
    echo "TPDNERV8_LANE_ABORT reason=python_not_executable path=$ner_python" >&2
    exit 1
}
for ner_required in "$ner_trainer" "$ner_manifest_tool" "$ner_source_lock"; do
    [[ -f "$ner_required" && ! -L "$ner_required" ]] || {
        echo "TPDNERV8_LANE_ABORT reason=missing_required_file path=$ner_required" >&2
        exit 1
    }
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$ner_gpu_uuid"
export TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_INDEX="$ner_physical_index"
export TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_UUID="$ner_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

cd "$ner_repo"

"$ner_python" "$ner_manifest_tool" \
    --mode verify \
    --kind training \
    --training-lock "$ner_source_lock"

"$ner_python" - "$ner_gpu_uuid" "$ner_physical_index" <<'PY'
import os
import sys

import torch

from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
    raise SystemExit("CUDA_VISIBLE_DEVICES differs from the assigned UUID")
if os.environ.get(
    "TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_INDEX"
) != expected_index:
    raise SystemExit("physical GPU index differs from the assignment")
if os.environ.get(
    "TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_UUID"
) != expected_uuid:
    raise SystemExit("physical GPU UUID differs from the assignment")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the lane must expose exactly one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected cuda:0 model: {torch.cuda.get_device_name(0)}")
properties = torch.cuda.get_device_properties(0)
actual_uuid = normalized_gpu_uuid(getattr(properties, "uuid", ""))
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"cuda:0 UUID differs: expected={expected_uuid} actual={actual_uuid}"
    )
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise SystemExit("CUBLAS_WORKSPACE_CONFIG differs")
if torch.get_num_threads() != 1:
    raise SystemExit(f"torch CPU thread count differs: {torch.get_num_threads()}")
print(
    "TPDNERV8_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " logical_device=cuda:0",
    flush=True,
)
PY

ner_run_dir="$ner_result_root/NUDT-SIRST/$ner_variant/seed_42_${ner_run_tag}"
ner_initialization="$(
    "$ner_python" - "$ner_run_dir" "$ner_variant" <<'PY'
import json
import pathlib
import sys

from experiments import train_tpd_ner_v8_mprs_dch_exact as exact

run_dir = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
if not run_dir.exists() and not run_dir.is_symlink():
    print("fresh")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"run path must be a regular directory: {run_dir}")

protocol_path = run_dir / "protocol.json"
if not protocol_path.is_file() or protocol_path.is_symlink():
    raise SystemExit("existing run directory has no exact protocol")
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
if protocol.get("schema") != exact.ENTRY_SCHEMA:
    raise SystemExit("existing protocol is not V8-MPRS-DCH+NER exact")
identity = exact._require_ner_run_identity(
    protocol.get("run_identity"),
    label="lane protocol",
    expected_variant=variant,
)
if identity.get("seed") != 42 or identity.get("split_seed") != 20260722:
    raise SystemExit("existing protocol seed identity differs")

summary_path = run_dir / "summary.json"
if summary_path.is_file() and not summary_path.is_symlink():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") == exact.COMPLETION_SUMMARY_SCHEMA
        and summary.get("status") == "complete"
        and summary.get("variant") == variant
        and summary.get("seed") == 42
    ):
        print("complete")
        raise SystemExit(0)

active_path = run_dir / "exact_journal" / "active.json"
if not active_path.is_file() or active_path.is_symlink():
    raise SystemExit("incomplete run has no committed exact epoch")
print("exact-resume")
PY
)"

echo "TPDNERV8_LANE_READY variant=$ner_variant physical_gpu=$ner_physical_index initialization=$ner_initialization run_dir=$ner_run_dir"

if [[ "$ner_lane_mode" == "preflight" || "$ner_initialization" == "complete" ]]; then
    exit 0
fi

ner_init_flag="--fresh"
if [[ "$ner_initialization" == "exact-resume" ]]; then
    ner_init_flag="--exact-resume"
fi

exec "$ner_python" "$ner_trainer" \
    --variant "$ner_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$ner_repo/datasets" \
    --output-root "$ner_result_root" \
    --run-tag "$ner_run_tag" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed 42 \
    --split-seed 20260722 \
    --val-fraction 0.20 \
    --eval-every 1 \
    --base-lr 0.001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --threshold 0.5 \
    --match-radius 3.0 \
    --tiny-area 9 \
    --eps 0.000001 \
    --exact-source-lock "$ner_source_lock" \
    "$ner_init_flag"
