#!/usr/bin/env bash
set -euo pipefail

v4_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v4_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 [--preflight] PHYSICAL_GPU_INDEX GPU_UUID" >&2
    exit 2
fi

v4_physical_index="$1"
v4_gpu_uuid="$2"
v4_variant="tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
v4_repo="${TPD_NER_V8_V4_TAIL_AWARE_REPO:-/home/ly/SCTransNet_main}"
v4_python="${TPD_NER_V8_V4_TAIL_AWARE_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v4_trainer="$v4_repo/experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact.py"
v4_freezer="$v4_repo/experiments/freeze_tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock.py"
v4_source_lock="${TPD_NER_V8_V4_TAIL_AWARE_SOURCE_LOCK:-$v4_repo/experiments/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock.json}"
v4_result_root="${TPD_NER_V8_V4_TAIL_AWARE_RESULT_ROOT:-$v4_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1}"
v4_run_tag="formal800_exact_v4_tail_aware_seed42"
v4_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v4_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

case "$v4_physical_index:$v4_gpu_uuid" in
    "2:$v4_gpu2_uuid"|"3:$v4_gpu3_uuid")
        ;;
    *)
        echo "TPDNERV8V4TAIL_LANE_ABORT reason=invalid_gpu_mapping physical_gpu=$v4_physical_index gpu_uuid=$v4_gpu_uuid" >&2
        exit 2
        ;;
esac

[[ -d "$v4_repo" && ! -L "$v4_repo" ]] || {
    echo "TPDNERV8V4TAIL_LANE_ABORT reason=invalid_repo path=$v4_repo" >&2
    exit 1
}
[[ -d "$v4_repo/datasets" && ! -L "$v4_repo/datasets" ]] || {
    echo "TPDNERV8V4TAIL_LANE_ABORT reason=invalid_dataset_dir path=$v4_repo/datasets" >&2
    exit 1
}
# The configured interpreter may intentionally be a symlink.
[[ -x "$v4_python" ]] || {
    echo "TPDNERV8V4TAIL_LANE_ABORT reason=python_not_executable path=$v4_python" >&2
    exit 1
}
for v4_required_file in \
    "$v4_trainer" \
    "$v4_freezer" \
    "$v4_source_lock"
do
    [[ -f "$v4_required_file" && ! -L "$v4_required_file" ]] || {
        echo "TPDNERV8V4TAIL_LANE_ABORT reason=missing_required_file path=$v4_required_file" >&2
        exit 1
    }
done
if [[ -L "$v4_result_root" || ( -e "$v4_result_root" && ! -d "$v4_result_root" ) ]]; then
    echo "TPDNERV8V4TAIL_LANE_ABORT reason=invalid_result_root path=$v4_result_root" >&2
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
export CUDA_VISIBLE_DEVICES="$v4_gpu_uuid"
export TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_INDEX="$v4_physical_index"
export TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_UUID="$v4_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

cd "$v4_repo"

# Every lane entry, including a systemd restart, verifies the immutable formal
# lock before touching CUDA or selecting fresh/exact-resume.
"$v4_python" "$v4_freezer" \
    --verify \
    --dataset-dir "$v4_repo/datasets" \
    --output "$v4_source_lock"

"$v4_python" - "$v4_gpu_uuid" "$v4_physical_index" <<'PY'
import os
import sys

import torch

from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact as exact,
)
from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if exact.PHYSICAL_GPU_UUIDS.get(expected_index) != expected_uuid:
    raise SystemExit("selected GPU mapping differs from the V4 trainer")
expected_environment = {
    "CUDA_VISIBLE_DEVICES": expected_uuid,
    (
        "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_INDEX"
    ): expected_index,
    (
        "TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_UUID"
    ): expected_uuid,
    "CUBLAS_WORKSPACE_CONFIG": exact.FORMAL_CUBLAS_WORKSPACE_CONFIG,
    "PYTHONHASHSEED": "42",
}
for name, expected in expected_environment.items():
    if os.environ.get(name) != expected:
        raise SystemExit(
            f"V4 lane environment differs for {name}: "
            f"expected={expected!r} observed={os.environ.get(name)!r}"
        )
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the V4 lane must expose exactly one CUDA device")
device_name = torch.cuda.get_device_name(0)
if device_name != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected cuda:0 model: {device_name}")
properties = torch.cuda.get_device_properties(0)
actual_uuid = normalized_gpu_uuid(getattr(properties, "uuid", ""))
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"cuda:0 UUID differs: expected={expected_uuid} actual={actual_uuid}"
    )
if torch.get_num_threads() != 1:
    raise SystemExit(f"torch CPU thread count differs: {torch.get_num_threads()}")
print(
    "TPDNERV8V4TAIL_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " model=NVIDIA_GeForce_RTX_5090"
    " logical_device=cuda:0",
    flush=True,
)
PY

v4_run_dir="$v4_result_root/NUDT-SIRST/$v4_variant/seed_42_${v4_run_tag}"
v4_initialization="$(
    "$v4_python" - \
        "$v4_run_dir" \
        "$v4_variant" \
        "$v4_source_lock" <<'PY'
import json
from pathlib import Path
import sys

import torch

from experiments import tpd_exact_epoch_journal as epoch_journal
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact as exact,
)


def read_json_object(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{label} is invalid or truncated: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain one JSON object")
    return payload


run_dir = Path(sys.argv[1])
variant = sys.argv[2]
source_lock = Path(sys.argv[3])
if variant != exact.TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON:
    raise SystemExit("lane variant is not the sole formal V4 candidate")
if not run_dir.exists() and not run_dir.is_symlink():
    print("fresh")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"run path must be a regular directory: {run_dir}")

protocol = read_json_object(run_dir / "protocol.json", "V4 protocol")
if protocol.get("schema") != exact.ENTRY_SCHEMA:
    raise SystemExit("existing protocol is not V4 tail-aware exact")
try:
    identity = exact.require_v4_run_identity(
        protocol.get("run_identity"),
        label="lane protocol",
        expected_variant=variant,
    )
except (TypeError, ValueError) as error:
    raise SystemExit(f"existing V4 protocol identity differs: {error}") from error
expected_run_id = (
    f"{exact.RUN_ID_PREFIX}NUDT-SIRST:{variant}:"
    f"seed-42:split-20260722:{exact.FORMAL_RUN_TAG}"
)
if (
    identity.get("dataset") != "NUDT-SIRST"
    or identity.get("seed") != exact.TRAINING_SEED
    or identity.get("split_seed") != exact.SPLIT_SEED
    or identity.get("run_id") != expected_run_id
):
    raise SystemExit("existing V4 protocol run identity differs")
if (
    identity["source_locks"].get(exact.SOURCE_LOCK_KEY)
    != exact.file_sha256(source_lock)
):
    raise SystemExit("existing V4 protocol source-lock identity differs")

journal_root = run_dir / "exact_journal"
active_path = journal_root / "active.json"
if (
    journal_root.is_symlink()
    or not journal_root.is_dir()
    or active_path.is_symlink()
    or not active_path.is_file()
):
    raise SystemExit("existing V4 run has no committed exact epoch")
try:
    active = epoch_journal.ExactEpochJournal(journal_root).load_active()
except (OSError, RuntimeError, ValueError) as error:
    raise SystemExit(f"existing V4 exact journal is invalid: {error}") from error
if active is None or not 1 <= active.epoch <= exact.FORMAL_EPOCHS:
    raise SystemExit("existing V4 exact journal epoch is invalid")
try:
    active_payload = torch.load(
        active.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
except Exception as error:
    raise SystemExit(
        f"existing V4 active checkpoint cannot be loaded: {error}"
    ) from error
try:
    active_identity = exact.require_v4_run_identity(
        active_payload.get("run_identity"),
        label="lane active exact journal",
        expected_variant=variant,
    )
except (AttributeError, TypeError, ValueError) as error:
    raise SystemExit(
        f"existing V4 active checkpoint identity differs: {error}"
    ) from error
if active_identity != identity:
    raise SystemExit("existing V4 active checkpoint identity differs")

summary_path = run_dir / "summary.json"
if summary_path.exists() or summary_path.is_symlink():
    summary = read_json_object(summary_path, "V4 completion summary")
    if (
        summary.get("schema") != exact.COMPLETION_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("variant") != variant
        or summary.get("seed") != exact.TRAINING_SEED
        or summary.get("split_seed") != exact.SPLIT_SEED
    ):
        raise SystemExit("existing V4 completion summary identity differs")
    try:
        summary_identity = exact.require_v4_run_identity(
            summary.get("run_identity"),
            label="lane completion summary",
            expected_variant=variant,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(
            f"existing V4 completion summary identity differs: {error}"
        ) from error
    if summary_identity != identity or active.epoch != exact.FORMAL_EPOCHS:
        raise SystemExit("completed V4 run state differs")
    print("complete")
    raise SystemExit(0)

print("exact-resume")
PY
)"

case "$v4_initialization" in
    fresh|exact-resume|complete)
        ;;
    *)
        echo "TPDNERV8V4TAIL_LANE_ABORT reason=invalid_initialization value=$v4_initialization" >&2
        exit 1
        ;;
esac

echo "TPDNERV8V4TAIL_LANE_READY variant=$v4_variant seed=42 epochs=800 physical_gpu=$v4_physical_index initialization=$v4_initialization run_dir=$v4_run_dir"

if [[ "$v4_lane_mode" == "preflight" || "$v4_initialization" == "complete" ]]; then
    exit 0
fi

v4_init_flag="--fresh"
if [[ "$v4_initialization" == "exact-resume" ]]; then
    v4_init_flag="--exact-resume"
fi

exec "$v4_python" "$v4_trainer" \
    --variant "$v4_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$v4_repo/datasets" \
    --output-root "$v4_result_root" \
    --run-tag "$v4_run_tag" \
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
    --exact-source-lock "$v4_source_lock" \
    "$v4_init_flag"
