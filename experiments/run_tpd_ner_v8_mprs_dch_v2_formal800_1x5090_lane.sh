#!/usr/bin/env bash
set -euo pipefail

v2_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v2_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 [--preflight] PHYSICAL_GPU_INDEX GPU_UUID" >&2
    exit 2
fi

v2_physical_index="$1"
v2_gpu_uuid="$2"
v2_variant="tpd_ner_v8_mprs_dch_v2_full_relay_on"
v2_repo="${TPD_NER_V8_V2_REPO:-/home/ly/SCTransNet_main}"
v2_python="${TPD_NER_V8_V2_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v2_trainer="$v2_repo/experiments/train_tpd_ner_v8_mprs_dch_v2_exact.py"
v2_manifest_tool="$v2_repo/experiments/freeze_tpd_ner_v8_mprs_dch_v2_source_locks.py"
v2_source_lock="${TPD_NER_V8_V2_SOURCE_LOCK:-$v2_repo/experiments/tpd_ner_v8_mprs_dch_v2_exact_source_lock.json}"
v2_result_root="${TPD_NER_V8_V2_RESULT_ROOT:-$v2_repo/experiments/results/tpd_ner_v8_mprs_dch_v2_exact_v1}"
v2_run_tag="formal800_exact_v2_seed42"
v2_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v2_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

case "$v2_physical_index:$v2_gpu_uuid" in
    "2:$v2_gpu2_uuid"|"3:$v2_gpu3_uuid")
        ;;
    *)
        echo "TPDNERV8V2_LANE_ABORT reason=invalid_gpu_mapping physical_gpu=$v2_physical_index gpu_uuid=$v2_gpu_uuid" >&2
        exit 2
        ;;
esac

[[ -x "$v2_python" ]] || {
    echo "TPDNERV8V2_LANE_ABORT reason=python_not_executable path=$v2_python" >&2
    exit 1
}
for v2_required in \
    "$v2_trainer" \
    "$v2_manifest_tool" \
    "$v2_source_lock"
do
    [[ -f "$v2_required" && ! -L "$v2_required" ]] || {
        echo "TPDNERV8V2_LANE_ABORT reason=missing_required_file path=$v2_required" >&2
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
export CUDA_VISIBLE_DEVICES="$v2_gpu_uuid"
export TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX="$v2_physical_index"
export TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID="$v2_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

cd "$v2_repo"

"$v2_python" "$v2_manifest_tool" \
    --mode verify \
    --kind training \
    --training-lock "$v2_source_lock"

"$v2_python" - "$v2_gpu_uuid" "$v2_physical_index" <<'PY'
import os
import sys

import torch

from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
    raise SystemExit("CUDA_VISIBLE_DEVICES differs from the selected UUID")
if os.environ.get(
    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX"
) != expected_index:
    raise SystemExit("physical GPU index differs from the selected lane")
if os.environ.get(
    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID"
) != expected_uuid:
    raise SystemExit("physical GPU UUID differs from the selected lane")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the V2 lane must expose exactly one CUDA device")
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
    "TPDNERV8V2_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " logical_device=cuda:0",
    flush=True,
)
PY

v2_run_dir="$v2_result_root/NUDT-SIRST/$v2_variant/seed_42_${v2_run_tag}"
v2_initialization="$(
    "$v2_python" - "$v2_run_dir" "$v2_variant" <<'PY'
import json
import pathlib
import sys

from experiments import train_tpd_ner_v8_mprs_dch_v2_exact as exact

run_dir = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
if variant != exact.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON:
    raise SystemExit("lane variant is not the sole formal V2 candidate")
if not run_dir.exists() and not run_dir.is_symlink():
    print("fresh")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"run path must be a regular directory: {run_dir}")

protocol_path = run_dir / "protocol.json"
if not protocol_path.is_file() or protocol_path.is_symlink():
    raise SystemExit("existing V2 run directory has no exact protocol")
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
if protocol.get("schema") != exact.ENTRY_SCHEMA:
    raise SystemExit("existing protocol is not V2 exact")
identity = exact.require_v2_run_identity(
    protocol.get("run_identity"),
    label="lane protocol",
    expected_variant=variant,
)
if identity.get("seed") != 42 or identity.get("split_seed") != 20260722:
    raise SystemExit("existing V2 protocol seed identity differs")

summary_path = run_dir / "summary.json"
if summary_path.is_file() and not summary_path.is_symlink():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") == exact.COMPLETION_SUMMARY_SCHEMA
        and summary.get("status") == "complete"
        and summary.get("variant") == variant
        and summary.get("seed") == 42
        and summary.get("split_seed") == 20260722
    ):
        print("complete")
        raise SystemExit(0)

active_path = run_dir / "exact_journal" / "active.json"
if not active_path.is_file() or active_path.is_symlink():
    raise SystemExit("incomplete V2 run has no committed exact epoch")
print("exact-resume")
PY
)"

echo "TPDNERV8V2_LANE_READY variant=$v2_variant physical_gpu=$v2_physical_index initialization=$v2_initialization run_dir=$v2_run_dir"

if [[ "$v2_lane_mode" == "preflight" || "$v2_initialization" == "complete" ]]; then
    exit 0
fi

v2_init_flag="--fresh"
if [[ "$v2_initialization" == "exact-resume" ]]; then
    v2_init_flag="--exact-resume"
fi

exec "$v2_python" "$v2_trainer" \
    --variant "$v2_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$v2_repo/datasets" \
    --output-root "$v2_result_root" \
    --run-tag "$v2_run_tag" \
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
    --exact-source-lock "$v2_source_lock" \
    "$v2_init_flag"
