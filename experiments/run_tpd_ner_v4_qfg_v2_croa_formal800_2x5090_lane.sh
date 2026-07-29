#!/usr/bin/env bash
set -euo pipefail

qfg2x_mode="run"
qfg2x_freeze_action="verify"
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --preflight)
            qfg2x_mode="preflight"
            shift
            ;;
        --freeze)
            [[ "$#" -ge 2 ]] || {
                echo "usage: $0 [--preflight] [--freeze verify|write-once]" >&2
                exit 2
            }
            qfg2x_freeze_action="$2"
            shift 2
            ;;
        --freeze=*)
            qfg2x_freeze_action="${1#--freeze=}"
            shift
            ;;
        *)
            echo "usage: $0 [--preflight] [--freeze verify|write-once]" >&2
            exit 2
            ;;
    esac
done
case "$qfg2x_freeze_action" in
    verify|write-once)
        ;;
    *)
        echo "QFG2X_LAUNCH_ABORT reason=invalid_freeze_action action=$qfg2x_freeze_action" >&2
        exit 2
        ;;
esac

qfg2x_repo="${TPD_NER_V4_QFG_V2_CROA_REPO:-/home/ly/SCTransNet_main}"
qfg2x_python="${TPD_NER_V4_QFG_V2_CROA_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
qfg2x_trainer="$qfg2x_repo/experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py"
qfg2x_freezer="$qfg2x_repo/experiments/freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock.py"
qfg2x_source_lock="$qfg2x_repo/experiments/tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
qfg2x_v1_source_lock="$qfg2x_repo/experiments/tpd_ner_v4_qfg_v2_croa_exact_source_lock.json"
qfg2x_result_root="$qfg2x_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized"
qfg2x_v1_result_root="$qfg2x_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v1"
qfg2x_statistics="$qfg2x_repo/experiments/tpd_survival_target_statistics_nudt_sirst_v1.json"
qfg2x_parent="$qfg2x_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar"
qfg2x_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
qfg2x_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

[[ -d "$qfg2x_repo" && ! -L "$qfg2x_repo" ]] || {
    echo "QFG2X_LAUNCH_ABORT reason=invalid_repo path=$qfg2x_repo" >&2
    exit 1
}
[[ -d "$qfg2x_repo/datasets" && ! -L "$qfg2x_repo/datasets" ]] || {
    echo "QFG2X_LAUNCH_ABORT reason=invalid_dataset_dir path=$qfg2x_repo/datasets" >&2
    exit 1
}
[[ -x "$qfg2x_python" ]] || {
    echo "QFG2X_LAUNCH_ABORT reason=python_not_executable path=$qfg2x_python" >&2
    exit 1
}
for qfg2x_required in \
    "$qfg2x_trainer" \
    "$qfg2x_freezer" \
    "$qfg2x_statistics" \
    "$qfg2x_parent"
do
    [[ -f "$qfg2x_required" && ! -L "$qfg2x_required" ]] || {
        echo "QFG2X_LAUNCH_ABORT reason=missing_required_file path=$qfg2x_required" >&2
        exit 1
    }
done

# V2 optimized owns new paths.  The active V1 slow trajectories are never
# accepted as an output root or source-lock destination by this launcher.
[[ "$qfg2x_result_root" != "$qfg2x_v1_result_root" ]] || {
    echo "QFG2X_LAUNCH_ABORT reason=v1_result_root_forbidden path=$qfg2x_result_root" >&2
    exit 1
}
[[ "$qfg2x_source_lock" != "$qfg2x_v1_source_lock" ]] || {
    echo "QFG2X_LAUNCH_ABORT reason=v1_source_lock_forbidden path=$qfg2x_source_lock" >&2
    exit 1
}
if [[ -L "$qfg2x_result_root" || ( -e "$qfg2x_result_root" && ! -d "$qfg2x_result_root" ) ]]; then
    echo "QFG2X_LAUNCH_ABORT reason=invalid_v2_result_root path=$qfg2x_result_root" >&2
    exit 1
fi

cd "$qfg2x_repo"

if [[ "$qfg2x_freeze_action" == "write-once" ]]; then
    "$qfg2x_python" "$qfg2x_freezer" \
        --write-once \
        --dataset-dir "$qfg2x_repo/datasets" \
        --output "$qfg2x_source_lock"
else
    "$qfg2x_python" "$qfg2x_freezer" \
        --verify \
        --dataset-dir "$qfg2x_repo/datasets" \
        --output "$qfg2x_source_lock"
fi
[[ -f "$qfg2x_source_lock" && ! -L "$qfg2x_source_lock" ]] || {
    echo "QFG2X_LAUNCH_ABORT reason=invalid_v2_source_lock path=$qfg2x_source_lock" >&2
    exit 1
}

qfg2x_initialization_mode() {
    local qfg2x_run_dir="$1"
    local qfg2x_variant="$2"
    "$qfg2x_python" - \
        "$qfg2x_run_dir" \
        "$qfg2x_variant" \
        "$qfg2x_source_lock" <<'PY'
import json
from pathlib import Path
import sys

from experiments import tpd_exact_epoch_journal as epoch_journal
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as exact


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

protocol = read_object(run_dir / "protocol.json", "QFG V2 optimized protocol")
identity = exact.require_qfg_run_identity(
    protocol.get("run_identity"),
    label="QFG V2 optimized lane protocol",
    expected_variant=variant,
)
if (
    identity.get("source_locks", {}).get(exact.SOURCE_LOCK_KEY)
    != exact.file_sha256(source_lock)
):
    raise SystemExit(
        "existing QFG V2 optimized protocol source-lock identity differs"
    )

active = epoch_journal.ExactEpochJournal(
    run_dir / "exact_journal"
).load_active()
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
            "empty QFG V2 optimized journal has derived artifacts: "
            f"{derived}"
        )
    print("parent-warm-start")
    raise SystemExit(0)
if not 1 <= active.epoch <= exact.FORMAL_EPOCHS:
    raise SystemExit(
        "existing QFG V2 optimized journal has an invalid epoch"
    )

summary_path = run_dir / "summary.json"
if summary_path.exists() or summary_path.is_symlink():
    summary = read_object(summary_path, "QFG V2 optimized summary")
    if (
        summary.get("schema") != exact.COMPLETION_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("variant") != variant
        or summary.get("seed") != exact.TRAINING_SEED
        or active.epoch != exact.FORMAL_EPOCHS
    ):
        raise SystemExit(
            "existing QFG V2 optimized completion identity differs"
        )
    print("complete")
else:
    print("exact-resume")
PY
}

qfg2x_lane() {
    local qfg2x_lane_mode="$1"
    local qfg2x_variant="$2"
    local qfg2x_physical_index="$3"
    local qfg2x_gpu_uuid="$4"
    local qfg2x_run_tag
    local qfg2x_run_dir
    local qfg2x_initialization
    local qfg2x_init_flag

    case "$qfg2x_variant:$qfg2x_physical_index:$qfg2x_gpu_uuid" in
        "qfg_only:2:$qfg2x_gpu2_uuid")
            qfg2x_run_tag="formal800_qfg_only"
            ;;
        "tss_qfg:3:$qfg2x_gpu3_uuid")
            qfg2x_run_tag="formal800_tss_qfg"
            ;;
        *)
            echo "QFG2X_LANE_ABORT reason=invalid_variant_gpu_mapping variant=$qfg2x_variant physical_gpu=$qfg2x_physical_index uuid=$qfg2x_gpu_uuid" >&2
            return 2
            ;;
    esac

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export TORCH_NUM_THREADS=1
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$qfg2x_gpu_uuid"
    export TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX="$qfg2x_physical_index"
    export TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$qfg2x_gpu_uuid"
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    export PYTHONHASHSEED=42
    export PYTHONUNBUFFERED=1

    "$qfg2x_python" - "$qfg2x_gpu_uuid" "$qfg2x_physical_index" <<'PY'
import os
import sys

import torch

from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as exact
from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if exact.PHYSICAL_GPU_UUIDS.get(expected_index) != expected_uuid:
    raise SystemExit("selected GPU mapping differs from the QFG trainer")
expected_environment = {
    "CUDA_VISIBLE_DEVICES": expected_uuid,
    "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX": expected_index,
    "TPD_NER_V4_QFG_PHYSICAL_GPU_UUID": expected_uuid,
    "CUBLAS_WORKSPACE_CONFIG": exact.FORMAL_CUBLAS_WORKSPACE_CONFIG,
    "PYTHONHASHSEED": "42",
}
for name, expected in expected_environment.items():
    if os.environ.get(name) != expected:
        raise SystemExit(
            f"QFG lane environment differs for {name}: "
            f"expected={expected!r} observed={os.environ.get(name)!r}"
        )
thread_environment = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
)
for name in thread_environment:
    if os.environ.get(name) != "1":
        raise SystemExit(f"QFG lane thread setting differs for {name}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the QFG lane must expose exactly one CUDA device")
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
    "QFG2X_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " model=NVIDIA_GeForce_RTX_5090"
    " logical_device=cuda:0",
    flush=True,
)
PY

    qfg2x_run_dir="$qfg2x_result_root/NUDT-SIRST/$qfg2x_variant/seed_42_$qfg2x_run_tag"
    qfg2x_initialization="$(
        qfg2x_initialization_mode "$qfg2x_run_dir" "$qfg2x_variant"
    )"
    case "$qfg2x_initialization" in
        parent-warm-start|exact-resume|complete)
            ;;
        *)
            echo "QFG2X_LANE_ABORT reason=invalid_initialization value=$qfg2x_initialization" >&2
            return 1
            ;;
    esac
    echo "QFG2X_LANE_READY variant=$qfg2x_variant seed=42 epochs=800 physical_gpu=$qfg2x_physical_index gpu_uuid=$qfg2x_gpu_uuid initialization=$qfg2x_initialization run_dir=$qfg2x_run_dir"
    if [[ "$qfg2x_lane_mode" == "preflight" || "$qfg2x_initialization" == "complete" ]]; then
        return 0
    fi

    qfg2x_init_flag="--parent-warm-start"
    if [[ "$qfg2x_initialization" == "exact-resume" ]]; then
        qfg2x_init_flag="--exact-resume"
    fi

    exec "$qfg2x_python" "$qfg2x_trainer" \
        --variant "$qfg2x_variant" \
        --dataset NUDT-SIRST \
        --dataset-dir "$qfg2x_repo/datasets" \
        --output-root "$qfg2x_result_root" \
        --run-tag "$qfg2x_run_tag" \
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
        --survival-target-statistics "$qfg2x_statistics" \
        --parent-checkpoint "$qfg2x_parent" \
        --exact-source-lock "$qfg2x_source_lock" \
        "$qfg2x_init_flag"
}

qfg2x_lane preflight qfg_only 2 "$qfg2x_gpu2_uuid"
qfg2x_lane preflight tss_qfg 3 "$qfg2x_gpu3_uuid"
echo "QFG2X_PAIRED_PREFLIGHT_COMPLETE seed=42 epochs=800 physical_gpus=2,3 output_root=$qfg2x_result_root v1_result_root_untouched=$qfg2x_v1_result_root"
if [[ "$qfg2x_mode" == "preflight" ]]; then
    exit 0
fi

mkdir -p "$qfg2x_result_root/.launcher_locks" "$qfg2x_result_root/logs"
qfg2x_claim="$qfg2x_result_root/.launcher_locks/formal800_seed42_paired.lock"
[[ ! -L "$qfg2x_claim" ]] || {
    echo "QFG2X_LAUNCH_ABORT reason=invalid_claim_file path=$qfg2x_claim" >&2
    exit 1
}
exec 9>"$qfg2x_claim"
if ! flock -n 9; then
    echo "QFG2X_LAUNCH_ABORT reason=paired_launch_claim_busy path=$qfg2x_claim" >&2
    exit 1
fi

qfg2x_qfg_log="$qfg2x_result_root/logs/qfg_only_gpu2.log"
qfg2x_tss_log="$qfg2x_result_root/logs/tss_qfg_gpu3.log"
(
    qfg2x_lane run qfg_only 2 "$qfg2x_gpu2_uuid"
) >"$qfg2x_qfg_log" 2>&1 &
qfg2x_qfg_pid="$!"
(
    qfg2x_lane run tss_qfg 3 "$qfg2x_gpu3_uuid"
) >"$qfg2x_tss_log" 2>&1 &
qfg2x_tss_pid="$!"

qfg2x_stop_children() {
    kill -TERM "$qfg2x_qfg_pid" "$qfg2x_tss_pid" 2>/dev/null || true
}
trap qfg2x_stop_children INT TERM

echo "QFG2X_PAIRED_LAUNCHED qfg_only_gpu=2 qfg_only_pid=$qfg2x_qfg_pid tss_qfg_gpu=3 tss_qfg_pid=$qfg2x_tss_pid output_root=$qfg2x_result_root"
echo "QFG2X_LOG variant=qfg_only path=$qfg2x_qfg_log"
echo "QFG2X_LOG variant=tss_qfg path=$qfg2x_tss_log"

set +e
wait "$qfg2x_qfg_pid"
qfg2x_qfg_status="$?"
wait "$qfg2x_tss_pid"
qfg2x_tss_status="$?"
set -e
trap - INT TERM

if [[ "$qfg2x_qfg_status" -ne 0 || "$qfg2x_tss_status" -ne 0 ]]; then
    echo "QFG2X_PAIRED_FAILED qfg_only_status=$qfg2x_qfg_status tss_qfg_status=$qfg2x_tss_status" >&2
    exit 1
fi
echo "QFG2X_PAIRED_COMPLETE qfg_only_status=0 tss_qfg_status=0 output_root=$qfg2x_result_root"
