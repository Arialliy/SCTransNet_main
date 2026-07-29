#!/usr/bin/env bash
set -euo pipefail

tss_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    tss_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 [--preflight] {tss_control|tss_on} PHYSICAL_GPU_INDEX GPU_UUID" >&2
    exit 2
fi

tss_variant="$1"
tss_physical_index="$2"
tss_gpu_uuid="$3"
tss_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
tss_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
case "$tss_variant:$tss_physical_index:$tss_gpu_uuid" in
    "tss_control:2:$tss_gpu2_uuid")
        tss_run_tag="formal800_control"
        tss_weight="0"
        ;;
    "tss_on:3:$tss_gpu3_uuid")
        tss_run_tag="formal800_tss"
        tss_weight="0.005"
        ;;
    *)
        echo "TSS_LANE_ABORT reason=invalid_variant_gpu_mapping variant=$tss_variant physical_gpu=$tss_physical_index uuid=$tss_gpu_uuid" >&2
        exit 2
        ;;
esac

tss_repo="${TPD_NER_V4_SURVIVAL_REPO:-/home/ly/SCTransNet_main}"
tss_python="${TPD_NER_V4_SURVIVAL_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
tss_trainer="$tss_repo/experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py"
tss_freezer="$tss_repo/experiments/freeze_tpd_ner_v4_survival_exact_source_lock.py"
tss_source_lock="${TPD_NER_V4_SURVIVAL_SOURCE_LOCK:-$tss_repo/experiments/tpd_ner_v4_survival_exact_source_lock.json}"
tss_statistics="$tss_repo/experiments/tpd_survival_target_statistics_nudt_sirst_v1.json"
tss_parent="$tss_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar"
tss_result_root="${TPD_NER_V4_SURVIVAL_RESULT_ROOT:-$tss_repo/experiments/results/tpd_ner_v4_survival_exact_v1}"

[[ -d "$tss_repo" && ! -L "$tss_repo" ]] || {
    echo "TSS_LANE_ABORT reason=invalid_repo path=$tss_repo" >&2
    exit 1
}
[[ -d "$tss_repo/datasets" && ! -L "$tss_repo/datasets" ]] || {
    echo "TSS_LANE_ABORT reason=invalid_dataset_dir path=$tss_repo/datasets" >&2
    exit 1
}
[[ -x "$tss_python" ]] || {
    echo "TSS_LANE_ABORT reason=python_not_executable path=$tss_python" >&2
    exit 1
}
for tss_required_file in \
    "$tss_trainer" \
    "$tss_freezer" \
    "$tss_source_lock" \
    "$tss_statistics" \
    "$tss_parent"
do
    [[ -f "$tss_required_file" && ! -L "$tss_required_file" ]] || {
        echo "TSS_LANE_ABORT reason=missing_required_file path=$tss_required_file" >&2
        exit 1
    }
done
if [[ -L "$tss_result_root" || ( -e "$tss_result_root" && ! -d "$tss_result_root" ) ]]; then
    echo "TSS_LANE_ABORT reason=invalid_result_root path=$tss_result_root" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$tss_gpu_uuid"
export TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_INDEX="$tss_physical_index"
export TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_UUID="$tss_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

cd "$tss_repo"

"$tss_python" "$tss_freezer" \
    --verify \
    --dataset-dir "$tss_repo/datasets" \
    --output "$tss_source_lock"

"$tss_python" - "$tss_gpu_uuid" "$tss_physical_index" <<'PY'
import os
import sys

import torch

from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as exact,
)
from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if exact.PHYSICAL_GPU_UUIDS.get(expected_index) != expected_uuid:
    raise SystemExit("selected GPU mapping differs from the TSS trainer")
expected_environment = {
    "CUDA_VISIBLE_DEVICES": expected_uuid,
    "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_INDEX": expected_index,
    "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_UUID": expected_uuid,
    "CUBLAS_WORKSPACE_CONFIG": exact.FORMAL_CUBLAS_WORKSPACE_CONFIG,
    "PYTHONHASHSEED": "42",
}
for name, expected in expected_environment.items():
    if os.environ.get(name) != expected:
        raise SystemExit(
            f"TSS lane environment differs for {name}: "
            f"expected={expected!r} observed={os.environ.get(name)!r}"
        )
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the TSS lane must expose exactly one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected cuda:0 model: {torch.cuda.get_device_name(0)}")
actual_uuid = normalized_gpu_uuid(
    getattr(torch.cuda.get_device_properties(0), "uuid", "")
)
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"cuda:0 UUID differs: expected={expected_uuid} actual={actual_uuid}"
    )
if torch.get_num_threads() != 1:
    raise SystemExit(f"torch CPU thread count differs: {torch.get_num_threads()}")
print(
    "TSS_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " model=NVIDIA_GeForce_RTX_5090"
    " logical_device=cuda:0",
    flush=True,
)
PY

tss_run_dir="$tss_result_root/NUDT-SIRST/$tss_variant/seed_42_$tss_run_tag"
tss_initialization="$(
    "$tss_python" - \
        "$tss_run_dir" \
        "$tss_variant" \
        "$tss_source_lock" <<'PY'
import json
from pathlib import Path
import sys

from experiments import tpd_exact_epoch_journal as epoch_journal
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as exact,
)


def read_object(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{label} is invalid: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain one JSON object")
    return value


run_dir = Path(sys.argv[1])
variant = sys.argv[2]
source_lock = Path(sys.argv[3])
exact.candidate_contract(variant)
if not run_dir.exists() and not run_dir.is_symlink():
    print("parent-warm-start")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"run path must be a regular directory: {run_dir}")

protocol = read_object(run_dir / "protocol.json", "TSS protocol")
identity = exact.require_tss_run_identity(
    protocol.get("run_identity"),
    label="TSS lane protocol",
    expected_variant=variant,
)
if (
    identity.get("source_locks", {}).get(exact.SOURCE_LOCK_KEY)
    != exact.file_sha256(source_lock)
):
    raise SystemExit("existing TSS protocol source-lock identity differs")

journal_root = run_dir / "exact_journal"
active = epoch_journal.ExactEpochJournal(journal_root).load_active()
if active is None:
    derived = [
        name
        for name in (
            "metrics.jsonl",
            "last.pth.tar",
            "best.pth.tar",
            "best_miou.pth.tar",
            "summary.json",
        )
        if (run_dir / name).exists() or (run_dir / name).is_symlink()
    ]
    if derived:
        raise SystemExit(
            "empty TSS journal has derived trajectory artifacts: "
            f"{derived}"
        )
    # The trainer writes protocol.json before entering epoch 1.  A process
    # failure in that narrow interval still represents a zero-epoch child
    # trajectory and can safely repeat the strict extension warm-start.
    print("parent-warm-start")
    raise SystemExit(0)
if not 1 <= active.epoch <= exact.FORMAL_EPOCHS:
    raise SystemExit("existing TSS run has an invalid committed exact epoch")

summary_path = run_dir / "summary.json"
if summary_path.exists() or summary_path.is_symlink():
    summary = read_object(summary_path, "TSS completion summary")
    if (
        summary.get("schema") != exact.COMPLETION_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("variant") != variant
        or active.epoch != exact.FORMAL_EPOCHS
    ):
        raise SystemExit("existing TSS completion identity differs")
    print("complete")
else:
    print("exact-resume")
PY
)"

case "$tss_initialization" in
    parent-warm-start|exact-resume|complete)
        ;;
    *)
        echo "TSS_LANE_ABORT reason=invalid_initialization value=$tss_initialization" >&2
        exit 1
        ;;
esac

echo "TSS_LANE_READY variant=$tss_variant seed=42 epochs=800 physical_gpu=$tss_physical_index initialization=$tss_initialization run_dir=$tss_run_dir"
if [[ "$tss_lane_mode" == "preflight" || "$tss_initialization" == "complete" ]]; then
    exit 0
fi

tss_init_flag="--parent-warm-start"
if [[ "$tss_initialization" == "exact-resume" ]]; then
    tss_init_flag="--exact-resume"
fi

exec "$tss_python" "$tss_trainer" \
    --variant "$tss_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$tss_repo/datasets" \
    --output-root "$tss_result_root" \
    --run-tag "$tss_run_tag" \
    --device cuda:0 \
    --epochs 800 \
    --batch-size 16 \
    --patch-size 256 \
    --workers 0 \
    --seed 42 \
    --split-seed 20260722 \
    --val-fraction 0.20 \
    --eval-every 1 \
    --base-lr 0.0001 \
    --min-lr 0.000001 \
    --warmup-epochs 10 \
    --threshold 0.5 \
    --match-radius 3.0 \
    --tiny-area 9 \
    --eps 0.000001 \
    --survival-weight "$tss_weight" \
    --survival-pos-weight 102.33587204874334 \
    --survival-target-statistics "$tss_statistics" \
    --parent-checkpoint "$tss_parent" \
    --exact-source-lock "$tss_source_lock" \
    "$tss_init_flag"
