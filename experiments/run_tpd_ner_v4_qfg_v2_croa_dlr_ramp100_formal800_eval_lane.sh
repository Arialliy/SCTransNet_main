#!/usr/bin/env bash
set -Eeuo pipefail

ramp_eval_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    ramp_eval_mode="preflight"
    shift
fi
if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 [--preflight] {qfg_dlr|tss_qfg_dlr} {best.pth.tar|best_miou.pth.tar} {2|3} GPU_UUID" >&2
    exit 2
fi

ramp_eval_variant="$1"
ramp_eval_checkpoint="$2"
ramp_eval_physical_index="$3"
ramp_eval_gpu_uuid="$4"
ramp_eval_repo="${TPD_NER_DLR_RAMP100_REPO:-/home/ly/SCTransNet_main}"
ramp_eval_python="${TPD_NER_DLR_RAMP100_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
ramp_eval_evaluator="$ramp_eval_repo/experiments/evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa.py"
ramp_eval_source_lock="${TPD_NER_DLR_RAMP100_SOURCE_LOCK:-$ramp_eval_repo/experiments/tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json}"
ramp_eval_result_root="${TPD_NER_DLR_RAMP100_RESULT_ROOT:-$ramp_eval_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_v1}"
ramp_eval_qfg_root="${TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT:-$ramp_eval_result_root/qfg_dlr_lane}"
ramp_eval_tss_root="${TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT:-$ramp_eval_result_root/tss_qfg_dlr_lane}"
ramp_eval_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
ramp_eval_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

ramp_eval_abort() {
    echo "TPDNER_DLR_RAMP100_EVAL_ABORT reason=$1 variant=$ramp_eval_variant checkpoint=$ramp_eval_checkpoint physical_gpu=$ramp_eval_physical_index" >&2
    exit 64
}

ramp_eval_map_error() {
    local ramp_eval_status="$?"
    local ramp_eval_line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    if [[ "$ramp_eval_status" -eq 64 || "$ramp_eval_status" -eq 75 ]]; then
        exit "$ramp_eval_status"
    fi
    echo "TPDNER_DLR_RAMP100_EVAL_ABORT reason=stage_failed original_exit=$ramp_eval_status line=$ramp_eval_line variant=$ramp_eval_variant checkpoint=$ramp_eval_checkpoint physical_gpu=$ramp_eval_physical_index" >&2
    exit 64
}
trap ramp_eval_map_error ERR

case "$ramp_eval_variant:$ramp_eval_physical_index:$ramp_eval_gpu_uuid" in
    "qfg_dlr:2:$ramp_eval_gpu2_uuid")
        ramp_eval_lane_root="$ramp_eval_qfg_root"
        ramp_eval_run_tag="formal800_qfg_dlr_control"
        ;;
    "tss_qfg_dlr:3:$ramp_eval_gpu3_uuid")
        ramp_eval_lane_root="$ramp_eval_tss_root"
        ramp_eval_run_tag="formal800_tss_qfg_dlr_ramp100"
        ;;
    *)
        ramp_eval_abort "invalid_variant_gpu_mapping"
        ;;
esac

case "$ramp_eval_checkpoint" in
    best.pth.tar)
        ramp_eval_output_name="pd_fa_sweep_best.pth.json"
        ramp_eval_role="best_validation_pd_primary"
        ;;
    best_miou.pth.tar)
        ramp_eval_output_name="pd_fa_sweep_best_miou.pth.json"
        ramp_eval_role="best_validation_miou_secondary"
        ;;
    *)
        ramp_eval_abort "invalid_checkpoint"
        ;;
esac

ramp_eval_run_dir="$ramp_eval_lane_root/NUDT-SIRST/$ramp_eval_variant/seed_42_$ramp_eval_run_tag"
ramp_eval_output="$ramp_eval_run_dir/$ramp_eval_output_name"

[[ -d "$ramp_eval_repo" && ! -L "$ramp_eval_repo" ]] \
    || ramp_eval_abort "invalid_repo"
[[ -x "$ramp_eval_python" ]] \
    || ramp_eval_abort "python_not_executable"
ramp_eval_python_real="$(readlink -f -- "$ramp_eval_python")"
[[ -n "$ramp_eval_python_real" \
    && -f "$ramp_eval_python_real" \
    && ! -L "$ramp_eval_python_real" \
    && -x "$ramp_eval_python_real" ]] \
    || ramp_eval_abort "python_target_nonregular"
for ramp_eval_required in \
    "$ramp_eval_evaluator" \
    "$ramp_eval_source_lock"
do
    [[ -f "$ramp_eval_required" && ! -L "$ramp_eval_required" ]] \
        || ramp_eval_abort "required_source_nonregular"
done
[[ -d "$ramp_eval_result_root" && ! -L "$ramp_eval_result_root" ]] \
    || ramp_eval_abort "result_root_nonregular"
[[ -d "$ramp_eval_run_dir" && ! -L "$ramp_eval_run_dir" ]] \
    || ramp_eval_abort "run_directory_nonregular"
for ramp_eval_artifact in \
    protocol.json split.json summary.json metrics.jsonl "$ramp_eval_checkpoint"
do
    [[ -f "$ramp_eval_run_dir/$ramp_eval_artifact" \
        && ! -L "$ramp_eval_run_dir/$ramp_eval_artifact" ]] \
        || ramp_eval_abort "run_artifact_nonregular"
done
if [[ -L "$ramp_eval_output" \
    || ( -e "$ramp_eval_output" && ! -f "$ramp_eval_output" ) ]]; then
    ramp_eval_abort "sweep_output_nonregular"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$ramp_eval_gpu_uuid"
export TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX="$ramp_eval_physical_index"
export TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$ramp_eval_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

cd "$ramp_eval_repo"

ramp_eval_preflight() {
    "$ramp_eval_python" "$ramp_eval_evaluator" \
        --run-dir "$ramp_eval_run_dir" \
        --checkpoint "$ramp_eval_checkpoint" \
        --device cpu \
        --expected-epochs 800 \
        --preflight
}

ramp_eval_verify_existing() {
    "$ramp_eval_python" - \
        "$ramp_eval_run_dir" \
        "$ramp_eval_checkpoint" \
        "$ramp_eval_output" \
        "$ramp_eval_physical_index" \
        "$ramp_eval_gpu_uuid" <<'PY'
from pathlib import Path
import sys

from experiments import (
    evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa as evaluator,
)

run_dir = Path(sys.argv[1]).resolve()
checkpoint = sys.argv[2]
output = Path(sys.argv[3]).resolve()
physical_index = int(sys.argv[4])
gpu_uuid = sys.argv[5]
audit = evaluator.validate_run_artifacts(run_dir, checkpoint)
assignment = {
    "device": "cuda:0",
    "physical_gpu_index": physical_index,
    "physical_gpu_uuid": gpu_uuid,
    "cuda_visible_devices": gpu_uuid,
    "device_name": "NVIDIA GeForce RTX 5090",
}
evaluator.validate_existing_output(
    output,
    artifact_audit=audit,
    device_assignment=assignment,
)
print(
    "TPDNER_DLR_RAMP100_EVAL_OUTPUT_VERIFIED"
    f" variant={audit['variant']}"
    f" checkpoint={checkpoint}"
    f" role={audit['checkpoint_role']}"
    f" physical_gpu={physical_index}"
    f" output={output}",
    flush=True,
)
PY
}

ramp_eval_gpu_probe() {
    "$ramp_eval_python" - \
        "$ramp_eval_gpu_uuid" \
        "$ramp_eval_physical_index" \
        "$ramp_eval_variant" <<'PY'
import os
import sys

import torch

from experiments import (
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as exact,
)
from experiments.train_tpd_clean_v8_mprs_dch_exact import (
    normalized_gpu_uuid,
)

expected_uuid, expected_index, variant = sys.argv[1:]
expected_variant_index = {
    exact.QFG_DLR_VARIANT: "2",
    exact.TSS_QFG_DLR_VARIANT: "3",
}[variant]
if expected_index != expected_variant_index:
    raise SystemExit("ramp100 evaluator variant/GPU mapping differs")
if exact.v2.PHYSICAL_GPU_UUIDS.get(expected_index) != expected_uuid:
    raise SystemExit("ramp100 evaluator GPU UUID differs from trainer")
for name, expected in {
    "CUDA_VISIBLE_DEVICES": expected_uuid,
    "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX": expected_index,
    "TPD_NER_V4_QFG_PHYSICAL_GPU_UUID": expected_uuid,
    "CUBLAS_WORKSPACE_CONFIG": exact.FORMAL_CUBLAS_WORKSPACE_CONFIG,
    "PYTHONHASHSEED": "42",
}.items():
    if os.environ.get(name) != expected:
        raise SystemExit(f"ramp100 evaluator environment differs: {name}")
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
        raise SystemExit(f"ramp100 evaluator thread control differs: {name}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("ramp100 evaluator must expose one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit("ramp100 evaluator CUDA model differs")
actual_uuid = normalized_gpu_uuid(
    getattr(torch.cuda.get_device_properties(0), "uuid", "")
)
if actual_uuid != expected_uuid:
    raise SystemExit("ramp100 evaluator logical CUDA UUID differs")
if torch.get_num_threads() != 1:
    raise SystemExit("ramp100 evaluator torch threads differ")
print(
    "TPDNER_DLR_RAMP100_EVAL_GPU_OK"
    f" variant={variant}"
    f" physical_gpu={expected_index}"
    f" uuid={actual_uuid}"
    " logical_device=cuda:0",
    flush=True,
)
PY
}

ramp_eval_preflight
if [[ -f "$ramp_eval_output" && ! -L "$ramp_eval_output" ]]; then
    ramp_eval_verify_existing
    echo "TPDNER_DLR_RAMP100_EVAL_IDEMPOTENT_COMPLETE variant=$ramp_eval_variant checkpoint=$ramp_eval_checkpoint role=$ramp_eval_role physical_gpu=$ramp_eval_physical_index output=$ramp_eval_output"
    exit 0
fi
if [[ "$ramp_eval_mode" == "preflight" ]]; then
    echo "TPDNER_DLR_RAMP100_EVAL_PREFLIGHT_ONLY variant=$ramp_eval_variant checkpoint=$ramp_eval_checkpoint role=$ramp_eval_role physical_gpu=$ramp_eval_physical_index output_state=absent writes_performed=false"
    exit 0
fi

ramp_eval_lock_dir="$ramp_eval_result_root/.evaluation_locks"
if [[ -L "$ramp_eval_lock_dir" \
    || ( -e "$ramp_eval_lock_dir" && ! -d "$ramp_eval_lock_dir" ) ]]; then
    ramp_eval_abort "lock_directory_nonregular"
fi
mkdir -p "$ramp_eval_lock_dir"
ramp_eval_job_lock="$ramp_eval_lock_dir/${ramp_eval_variant}_${ramp_eval_output_name}.lock"
ramp_eval_gpu_lock="$ramp_eval_lock_dir/gpu${ramp_eval_physical_index}.lock"
for ramp_eval_lock in "$ramp_eval_job_lock" "$ramp_eval_gpu_lock"
do
    if [[ -L "$ramp_eval_lock" \
        || ( -e "$ramp_eval_lock" && ! -f "$ramp_eval_lock" ) ]]; then
        ramp_eval_abort "claim_nonregular"
    fi
done
exec 8>>"$ramp_eval_job_lock"
if ! flock -n 8; then
    echo "TPDNER_DLR_RAMP100_EVAL_RETRY reason=checkpoint_claim_held variant=$ramp_eval_variant checkpoint=$ramp_eval_checkpoint" >&2
    exit 75
fi
exec 9>>"$ramp_eval_gpu_lock"
if ! flock -n 9; then
    echo "TPDNER_DLR_RAMP100_EVAL_RETRY reason=gpu_claim_held variant=$ramp_eval_variant physical_gpu=$ramp_eval_physical_index" >&2
    exit 75
fi

if [[ -L "$ramp_eval_output" \
    || ( -e "$ramp_eval_output" && ! -f "$ramp_eval_output" ) ]]; then
    ramp_eval_abort "sweep_output_nonregular_after_claim"
fi
ramp_eval_preflight
if [[ -f "$ramp_eval_output" && ! -L "$ramp_eval_output" ]]; then
    ramp_eval_verify_existing
    echo "TPDNER_DLR_RAMP100_EVAL_IDEMPOTENT_COMPLETE variant=$ramp_eval_variant checkpoint=$ramp_eval_checkpoint role=$ramp_eval_role physical_gpu=$ramp_eval_physical_index output=$ramp_eval_output"
    exit 0
fi

ramp_eval_gpu_probe
"$ramp_eval_python" "$ramp_eval_evaluator" \
    --run-dir "$ramp_eval_run_dir" \
    --checkpoint "$ramp_eval_checkpoint" \
    --device cuda:0 \
    --expected-epochs 800
ramp_eval_verify_existing
echo "TPDNER_DLR_RAMP100_EVAL_COMPLETE variant=$ramp_eval_variant checkpoint=$ramp_eval_checkpoint role=$ramp_eval_role physical_gpu=$ramp_eval_physical_index logical_device=cuda:0 output=$ramp_eval_output"
