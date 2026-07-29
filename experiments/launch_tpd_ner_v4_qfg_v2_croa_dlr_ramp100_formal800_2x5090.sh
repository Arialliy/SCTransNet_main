#!/usr/bin/env bash
set -Eeuo pipefail

# Exit contract for watchdogs:
#   0  requested action succeeded, or both lanes were already complete
#   2  command-line usage error
#   64 permanent contract/configuration/artifact error (do not restart)
#   75 retryable condition: paired claim busy, child interruption/failure,
#      or a nominally successful child did not publish a complete trajectory
paired_exit_permanent=64
paired_exit_retry=75

paired_mode="run"
paired_source_lock_mode="verify"
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --preflight)
            paired_mode="preflight"
            shift
            ;;
        --source-lock-mode|--freeze)
            [[ "$#" -ge 2 ]] || {
                echo "usage: $0 [--preflight] [--source-lock-mode verify|write-once]" >&2
                exit 2
            }
            paired_source_lock_mode="$2"
            shift 2
            ;;
        --source-lock-mode=*|--freeze=*)
            paired_source_lock_mode="${1#*=}"
            shift
            ;;
        *)
            echo "usage: $0 [--preflight] [--source-lock-mode verify|write-once]" >&2
            exit 2
            ;;
    esac
done
case "$paired_source_lock_mode" in
    verify|write-once)
        # The 51-source manifest is already frozen.  "write-once" is accepted
        # only as a compatibility spelling and performs the same read-only
        # verification; this launcher never creates or rewrites the lock.
        ;;
    *)
        echo "TPDNER_DLR_RAMP100_2X_ABORT reason=invalid_source_lock_mode mode=$paired_source_lock_mode" >&2
        exit "$paired_exit_permanent"
        ;;
esac

paired_repo="${TPD_NER_DLR_RAMP100_REPO:-/home/ly/SCTransNet_main}"
paired_python="${TPD_NER_DLR_RAMP100_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
paired_trainer="${TPD_NER_DLR_RAMP100_TRAINER:-$paired_repo/experiments/train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact.py}"
paired_source_lock="${TPD_NER_DLR_RAMP100_SOURCE_LOCK:-$paired_repo/experiments/tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json}"
paired_statistics="${TPD_NER_DLR_RAMP100_STATISTICS:-$paired_repo/experiments/tpd_survival_target_statistics_nudt_sirst_v1.json}"
paired_parent="${TPD_NER_DLR_RAMP100_PARENT:-$paired_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar}"
paired_result_root="${TPD_NER_DLR_RAMP100_RESULT_ROOT:-$paired_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_v1}"
paired_qfg_output_root="${TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT:-$paired_result_root/qfg_dlr_lane}"
paired_tss_output_root="${TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT:-$paired_result_root/tss_qfg_dlr_lane}"
paired_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
paired_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
paired_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"

# Keep every launcher-owned Python process, including source-lock and journal
# verification, at one CPU thread.  Each lane re-exports the same contract
# immediately before its isolated CUDA preflight and trainer exec.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1

paired_abort() {
    echo "TPDNER_DLR_RAMP100_2X_ABORT reason=$1" >&2
    exit "$paired_exit_permanent"
}

paired_retry() {
    echo "TPDNER_DLR_RAMP100_2X_RETRY reason=$1" >&2
    exit "$paired_exit_retry"
}

[[ -d "$paired_repo" && ! -L "$paired_repo" ]] \
    || paired_abort "invalid_repo path=$paired_repo"
[[ -d "$paired_repo/datasets" && ! -L "$paired_repo/datasets" ]] \
    || paired_abort "invalid_dataset_dir path=$paired_repo/datasets"
[[ -x "$paired_python" && ! -L "$paired_python" ]] \
    || paired_abort "python_not_executable_regular path=$paired_python"
for paired_required in \
    "$paired_trainer" \
    "$paired_source_lock" \
    "$paired_statistics" \
    "$paired_parent"
do
    [[ -f "$paired_required" && ! -L "$paired_required" ]] \
        || paired_abort "missing_required_regular_file path=$paired_required"
done
command -v flock >/dev/null 2>&1 || paired_abort "flock_not_found"
[[ "$paired_qfg_output_root" != "$paired_tss_output_root" ]] \
    || paired_abort "lane_output_roots_must_differ"
for paired_path in \
    "$paired_result_root" \
    "$paired_qfg_output_root" \
    "$paired_tss_output_root"
do
    if [[ -L "$paired_path" \
        || ( -e "$paired_path" && ! -d "$paired_path" ) ]]; then
        paired_abort "invalid_output_root path=$paired_path"
    fi
done

cd "$paired_repo"

paired_verify_source_lock() {
    "$paired_python" - \
        paired-source-lock-verify \
        "$paired_source_lock" \
        "$paired_training_data_sha256" \
        "$paired_statistics" \
        "$paired_source_lock_mode" <<'PY'
from pathlib import Path
import sys

from experiments import (
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as exact,
)

(
    marker,
    lock_text,
    training_data_sha256,
    statistics_text,
    requested_mode,
) = sys.argv[1:]
if marker != "paired-source-lock-verify":
    raise SystemExit("paired source-lock verifier marker differs")
lock_path = Path(lock_text)
if lock_path.is_symlink() or not lock_path.is_file():
    raise SystemExit(f"paired source lock must be regular: {lock_path}")
locks = exact.source_lock_contract(
    training_data_sha256,
    lock_path,
    Path(statistics_text),
)
if locks[exact.SOURCE_LOCK_KEY] != exact.file_sha256(lock_path):
    raise SystemExit("paired source-lock digest differs after verification")
print(
    "TPDNER_DLR_RAMP100_SOURCE_LOCK_OK"
    f" requested_mode={requested_mode}"
    " action=verify-existing"
    f" source_count={len(exact.RUNTIME_SOURCE_PATHS)}"
    f" sha256={locks[exact.SOURCE_LOCK_KEY]}",
    flush=True,
)
PY
}

paired_gpu_preflight() {
    local paired_physical_index="$1"
    local paired_gpu_uuid="$2"
    "$paired_python" - \
        paired-gpu-preflight \
        "$paired_physical_index" \
        "$paired_gpu_uuid" <<'PY'
import os
import sys

import torch

from experiments import (
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as exact,
)
from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

marker, expected_index, expected_uuid = sys.argv[1:]
if marker != "paired-gpu-preflight":
    raise SystemExit("paired GPU verifier marker differs")
if exact.v2.PHYSICAL_GPU_UUIDS.get(expected_index) != expected_uuid:
    raise SystemExit("paired physical GPU mapping differs from trainer")
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
            f"paired GPU environment differs for {name}: "
            f"expected={expected!r} observed={os.environ.get(name)!r}"
        )
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    if os.environ.get(name) != "1":
        raise SystemExit(f"paired CPU thread setting differs for {name}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("paired lane must expose exactly one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(
        f"paired lane has unexpected cuda:0 model: "
        f"{torch.cuda.get_device_name(0)}"
    )
actual_uuid = normalized_gpu_uuid(
    getattr(torch.cuda.get_device_properties(0), "uuid", "")
)
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"paired cuda:0 UUID differs: "
        f"expected={expected_uuid} actual={actual_uuid}"
    )
if torch.get_num_threads() != 1:
    raise SystemExit(
        f"paired torch CPU thread count differs: {torch.get_num_threads()}"
    )
print(
    "TPDNER_DLR_RAMP100_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " model=NVIDIA_GeForce_RTX_5090"
    " logical_device=cuda:0",
    flush=True,
)
PY
}

paired_initialization_mode() {
    local paired_run_dir="$1"
    local paired_variant="$2"
    local paired_run_tag="$3"
    local paired_lane_output_root="$4"
    "$paired_python" - \
        paired-run-state \
        "$paired_run_dir" \
        "$paired_variant" \
        "$paired_run_tag" \
        "$paired_lane_output_root" \
        "$paired_source_lock" <<'PY'
import json
from pathlib import Path
import sys

import torch

from experiments import tpd_exact_epoch_journal as epoch_journal
from experiments import (
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as exact,
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


(
    marker,
    run_dir_text,
    variant,
    run_tag,
    output_root_text,
    source_lock_text,
) = sys.argv[1:]
if marker != "paired-run-state":
    raise SystemExit("paired run-state marker differs")
candidate = exact.candidate_contract(variant)
if run_tag != candidate["formal_run_tag"]:
    raise SystemExit("paired lane run tag differs from formal contract")
run_dir = Path(run_dir_text).resolve()
output_root = Path(output_root_text).resolve()
expected_run_dir = (
    output_root
    / "NUDT-SIRST"
    / variant
    / f"seed_42_{run_tag}"
)
if run_dir != expected_run_dir:
    raise SystemExit("paired lane run directory differs")
source_lock = Path(source_lock_text).resolve()

if not run_dir.exists() and not run_dir.is_symlink():
    print("parent-warm-start")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"paired run path must be a real directory: {run_dir}")

protocol = read_object(run_dir / "protocol.json", "paired protocol")
if protocol.get("schema") != exact.ENTRY_SCHEMA:
    raise SystemExit("paired protocol schema differs")
if Path(protocol.get("run_directory", "")).resolve() != run_dir:
    raise SystemExit("paired protocol run directory differs")
arguments = protocol.get("arguments")
if not isinstance(arguments, dict):
    raise SystemExit("paired protocol arguments are missing")
if (
    arguments.get("variant") != variant
    or arguments.get("run_tag") != run_tag
    or Path(arguments.get("output_root", "")).resolve() != output_root
):
    raise SystemExit("paired protocol lane arguments differ")
identity = exact.require_paired_run_identity(
    protocol.get("run_identity"),
    label="paired lane protocol",
    expected_variant=variant,
)
if (
    identity.get("source_locks", {}).get(exact.SOURCE_LOCK_KEY)
    != exact.file_sha256(source_lock)
):
    raise SystemExit("paired protocol source-lock identity differs")

journal = epoch_journal.ExactEpochJournal(run_dir / "exact_journal")
active = journal.load_active()
derived_names = (
    "metrics.jsonl",
    "last.pth.tar",
    "best.pth.tar",
    "best_miou.pth.tar",
    "summary.json",
)
if active is None:
    unexpected = [
        name
        for name in derived_names
        if (run_dir / name).exists() or (run_dir / name).is_symlink()
    ]
    if unexpected:
        raise SystemExit(
            f"empty paired journal has derived artifacts: {unexpected}"
        )
    print("parent-warm-start")
    raise SystemExit(0)
if not 1 <= active.epoch <= exact.FORMAL_EPOCHS:
    raise SystemExit("paired active journal epoch is invalid")
payload = torch.load(
    active.checkpoint_path,
    map_location="cpu",
    weights_only=False,
)
if not isinstance(payload, dict):
    raise SystemExit("paired active checkpoint is invalid")
active_identity = exact.require_paired_run_identity(
    payload.get("run_identity"),
    label="paired active checkpoint",
    expected_variant=variant,
)
if active_identity != identity or payload.get("epoch") != active.epoch:
    raise SystemExit("paired active checkpoint boundary differs")
if not isinstance(payload.get("optimizer"), dict):
    raise SystemExit("paired active checkpoint optimizer is missing")

summary_path = run_dir / "summary.json"
if not summary_path.exists() and not summary_path.is_symlink():
    print("exact-resume")
    raise SystemExit(0)
summary = read_object(summary_path, "paired completion summary")
if active.epoch != exact.FORMAL_EPOCHS:
    raise SystemExit("paired summary exists before epoch800")
required_summary = {
    "schema": exact.COMPLETION_SUMMARY_SCHEMA,
    "status": "complete",
    "variant": variant,
    "candidate_variant": variant,
    "family_recipe": exact.FAMILY_RECIPE,
    "seed": exact.TRAINING_SEED,
    "split_seed": exact.SPLIT_SEED,
    "formal_contract": exact.formal_contract(),
    "run_identity": identity,
    "source_locks": identity["source_locks"],
}
for name, expected in required_summary.items():
    if summary.get(name) != expected:
        raise SystemExit(f"paired summary field {name} differs")
metrics_path = run_dir / "metrics.jsonl"
if metrics_path.is_symlink() or not metrics_path.is_file():
    raise SystemExit("paired completion metrics are not regular")
events = [
    json.loads(line)
    for line in metrics_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if (
    len(events) != exact.FORMAL_EPOCHS
    or [event.get("epoch") for event in events]
    != list(range(1, exact.FORMAL_EPOCHS + 1))
):
    raise SystemExit("paired completion metrics are not contiguous800")
for name in (
    "last.pth.tar",
    "best.pth.tar",
    "best_miou.pth.tar",
):
    path = run_dir / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"paired completion checkpoint is invalid: {name}")
print("complete")
PY
}

paired_lane() {
    local paired_lane_mode="$1"
    local paired_variant="$2"
    local paired_physical_index="$3"
    local paired_gpu_uuid="$4"
    local paired_run_tag
    local paired_lane_output_root
    local paired_weight_max
    local paired_run_dir
    local paired_initialization
    local paired_init_flag

    case "$paired_variant:$paired_physical_index:$paired_gpu_uuid" in
        "qfg_dlr:2:$paired_gpu2_uuid")
            paired_run_tag="formal800_qfg_dlr_control"
            paired_lane_output_root="$paired_qfg_output_root"
            paired_weight_max="0.0"
            ;;
        "tss_qfg_dlr:3:$paired_gpu3_uuid")
            paired_run_tag="formal800_tss_qfg_dlr_ramp100"
            paired_lane_output_root="$paired_tss_output_root"
            paired_weight_max="0.005"
            ;;
        *)
            echo "TPDNER_DLR_RAMP100_LANE_ABORT reason=invalid_variant_gpu_mapping variant=$paired_variant physical_gpu=$paired_physical_index uuid=$paired_gpu_uuid" >&2
            return "$paired_exit_permanent"
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
    export CUDA_VISIBLE_DEVICES="$paired_gpu_uuid"
    export TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX="$paired_physical_index"
    export TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$paired_gpu_uuid"
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    export PYTHONHASHSEED=42
    export PYTHONUNBUFFERED=1

    paired_gpu_preflight "$paired_physical_index" "$paired_gpu_uuid" \
        || return "$paired_exit_permanent"

    paired_run_dir="$paired_lane_output_root/NUDT-SIRST/$paired_variant/seed_42_$paired_run_tag"
    paired_initialization="$(
        paired_initialization_mode \
            "$paired_run_dir" \
            "$paired_variant" \
            "$paired_run_tag" \
            "$paired_lane_output_root"
    )" || return "$paired_exit_permanent"
    case "$paired_initialization" in
        parent-warm-start|exact-resume|complete)
            ;;
        *)
            echo "TPDNER_DLR_RAMP100_LANE_ABORT reason=invalid_initialization value=$paired_initialization" >&2
            return "$paired_exit_permanent"
            ;;
    esac
    echo "TPDNER_DLR_RAMP100_LANE_READY variant=$paired_variant seed=42 epochs=800 physical_gpu=$paired_physical_index gpu_uuid=$paired_gpu_uuid initialization=$paired_initialization run_tag=$paired_run_tag output_root=$paired_lane_output_root run_dir=$paired_run_dir"
    if [[ "$paired_lane_mode" == "preflight" \
        || "$paired_initialization" == "complete" ]]; then
        return 0
    fi

    paired_init_flag="--parent-warm-start"
    if [[ "$paired_initialization" == "exact-resume" ]]; then
        paired_init_flag="--exact-resume"
    fi

    exec "$paired_python" "$paired_trainer" \
        --variant "$paired_variant" \
        --dataset NUDT-SIRST \
        --dataset-dir "$paired_repo/datasets" \
        --output-root "$paired_lane_output_root" \
        --run-tag "$paired_run_tag" \
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
        --survival-weight-max "$paired_weight_max" \
        --survival-target-statistics "$paired_statistics" \
        --parent-checkpoint "$paired_parent" \
        --exact-source-lock "$paired_source_lock" \
        "$paired_init_flag"
}

paired_verify_source_lock || paired_abort "source_lock_live_verify_failed"
(
    paired_lane preflight qfg_dlr 2 "$paired_gpu2_uuid"
) || paired_abort "qfg_dlr_preflight_failed"
(
    paired_lane preflight tss_qfg_dlr 3 "$paired_gpu3_uuid"
) || paired_abort "tss_qfg_dlr_preflight_failed"
echo "TPDNER_DLR_RAMP100_PAIRED_PREFLIGHT_OK seed=42 epochs=800 qfg_dlr_gpu=2 tss_qfg_dlr_gpu=3 qfg_output_root=$paired_qfg_output_root tss_output_root=$paired_tss_output_root wait_for_gpu_idle=false source_lock_requested_mode=$paired_source_lock_mode source_lock_action=verify-existing exit_permanent=64 exit_retry=75 writes_performed=false"
if [[ "$paired_mode" == "preflight" ]]; then
    exit 0
fi

mkdir -p \
    "$paired_result_root/.launcher_locks" \
    "$paired_result_root/logs"
paired_claim="$paired_result_root/.launcher_locks/formal800_seed42_ramp100_paired.lock"
[[ ! -L "$paired_claim" ]] \
    || paired_abort "invalid_paired_claim path=$paired_claim"
exec 9>"$paired_claim"
if ! flock -n 9; then
    paired_retry "paired_launch_claim_busy path=$paired_claim"
fi

paired_qfg_log="$paired_result_root/logs/qfg_dlr_gpu2.log"
paired_tss_log="$paired_result_root/logs/tss_qfg_dlr_gpu3.log"
(
    paired_lane run qfg_dlr 2 "$paired_gpu2_uuid"
) >"$paired_qfg_log" 2>&1 &
paired_qfg_pid="$!"
(
    paired_lane run tss_qfg_dlr 3 "$paired_gpu3_uuid"
) >"$paired_tss_log" 2>&1 &
paired_tss_pid="$!"

paired_stop_children() {
    kill -TERM "$paired_qfg_pid" "$paired_tss_pid" 2>/dev/null || true
}
trap paired_stop_children INT TERM

echo "TPDNER_DLR_RAMP100_PAIRED_LAUNCHED qfg_dlr_gpu=2 qfg_dlr_pid=$paired_qfg_pid tss_qfg_dlr_gpu=3 tss_qfg_dlr_pid=$paired_tss_pid wait_for_gpu_idle=false"
echo "TPDNER_DLR_RAMP100_LOG variant=qfg_dlr path=$paired_qfg_log"
echo "TPDNER_DLR_RAMP100_LOG variant=tss_qfg_dlr path=$paired_tss_log"

set +e
wait "$paired_qfg_pid"
paired_qfg_status="$?"
wait "$paired_tss_pid"
paired_tss_status="$?"
set -e
trap - INT TERM

if [[ "$paired_qfg_status" -ne 0 \
    || "$paired_tss_status" -ne 0 ]]; then
    echo "TPDNER_DLR_RAMP100_PAIRED_RETRY reason=lane_failed qfg_dlr_status=$paired_qfg_status tss_qfg_dlr_status=$paired_tss_status" >&2
    exit "$paired_exit_retry"
fi

paired_qfg_run="$paired_qfg_output_root/NUDT-SIRST/qfg_dlr/seed_42_formal800_qfg_dlr_control"
paired_tss_run="$paired_tss_output_root/NUDT-SIRST/tss_qfg_dlr/seed_42_formal800_tss_qfg_dlr_ramp100"
paired_qfg_final="$(
    paired_initialization_mode \
        "$paired_qfg_run" \
        qfg_dlr \
        formal800_qfg_dlr_control \
        "$paired_qfg_output_root"
)" || paired_abort "qfg_dlr_postrun_state_invalid"
paired_tss_final="$(
    paired_initialization_mode \
        "$paired_tss_run" \
        tss_qfg_dlr \
        formal800_tss_qfg_dlr_ramp100 \
        "$paired_tss_output_root"
)" || paired_abort "tss_qfg_dlr_postrun_state_invalid"
if [[ "$paired_qfg_final" != "complete" \
    || "$paired_tss_final" != "complete" ]]; then
    paired_retry \
        "postrun_incomplete qfg_dlr=$paired_qfg_final tss_qfg_dlr=$paired_tss_final"
fi
echo "TPDNER_DLR_RAMP100_PAIRED_COMPLETE qfg_dlr_status=0 tss_qfg_dlr_status=0 qfg_output_root=$paired_qfg_output_root tss_output_root=$paired_tss_output_root"
