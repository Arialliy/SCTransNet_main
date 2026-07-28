#!/usr/bin/env bash
set -Eeuo pipefail

v4_eval_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v4_eval_mode="preflight"
    shift
fi
if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 [--preflight] {best.pth.tar|best_miou.pth.tar} PHYSICAL_GPU_INDEX GPU_UUID" >&2
    exit 2
fi

v4_eval_checkpoint="$1"
v4_eval_physical_index="$2"
v4_eval_gpu_uuid="$3"
v4_eval_repo="${TPD_NER_V8_V4_TAIL_AWARE_REPO:-/home/ly/SCTransNet_main}"
v4_eval_python="${TPD_NER_V8_V4_TAIL_AWARE_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v4_eval_script="$v4_eval_repo/experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa.py"
v4_eval_result_root="${TPD_NER_V8_V4_TAIL_AWARE_RESULT_ROOT:-$v4_eval_repo/experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1}"
v4_eval_variant="tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
v4_eval_run_dir="$v4_eval_result_root/NUDT-SIRST/$v4_eval_variant/seed_42_formal800_exact_v4_tail_aware_seed42"
v4_eval_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v4_eval_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

case "$v4_eval_checkpoint:$v4_eval_physical_index:$v4_eval_gpu_uuid" in
    "best.pth.tar:2:$v4_eval_gpu2_uuid")
        v4_eval_output="$v4_eval_run_dir/pd_fa_sweep_best.pth.json"
        v4_eval_role="best_validation_pd_primary"
        ;;
    "best_miou.pth.tar:3:$v4_eval_gpu3_uuid")
        v4_eval_output="$v4_eval_run_dir/pd_fa_sweep_best_miou.pth.json"
        v4_eval_role="best_validation_miou_secondary"
        ;;
    *)
        echo "TPDNERV8V4TAIL_EVAL_ABORT reason=invalid_role_gpu_mapping checkpoint=$v4_eval_checkpoint physical_gpu=$v4_eval_physical_index gpu_uuid=$v4_eval_gpu_uuid" >&2
        exit 64
        ;;
esac

v4_eval_abort() {
    echo "TPDNERV8V4TAIL_EVAL_ABORT reason=$1 checkpoint=$v4_eval_checkpoint physical_gpu=$v4_eval_physical_index" >&2
    exit 64
}

[[ -d "$v4_eval_repo" && ! -L "$v4_eval_repo" ]] \
    || v4_eval_abort "invalid_repo"
[[ -x "$v4_eval_python" ]] \
    || v4_eval_abort "python_not_executable"
[[ -f "$v4_eval_script" && ! -L "$v4_eval_script" ]] \
    || v4_eval_abort "evaluator_nonregular"
[[ -d "$v4_eval_run_dir" && ! -L "$v4_eval_run_dir" ]] \
    || v4_eval_abort "run_directory_nonregular"
[[ -f "$v4_eval_run_dir/$v4_eval_checkpoint" \
    && ! -L "$v4_eval_run_dir/$v4_eval_checkpoint" ]] \
    || v4_eval_abort "checkpoint_nonregular"
if [[ -L "$v4_eval_output" \
    || ( -e "$v4_eval_output" && ! -f "$v4_eval_output" ) ]]; then
    v4_eval_abort "sweep_output_nonregular"
fi

cd "$v4_eval_repo"

v4_eval_validate_output() {
    "$v4_eval_python" - \
        "$v4_eval_run_dir" \
        "$v4_eval_checkpoint" \
        "$v4_eval_output" \
        "$v4_eval_physical_index" \
        "$v4_eval_gpu_uuid" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import (
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as evaluator,
)
from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800 as postprocess,
)


run_dir = Path(sys.argv[1]).resolve()
checkpoint = sys.argv[2]
output = Path(sys.argv[3]).resolve()
physical_index = int(sys.argv[4])
physical_uuid = sys.argv[5]
if not output.is_file() or output.is_symlink():
    raise SystemExit(f"sweep is not a regular file: {output}")
payload = json.loads(output.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("sweep payload must be one JSON object")
audit = evaluator.validate_run_artifacts(run_dir, checkpoint)
evaluator.validate_output_identity(payload, artifact_audit=audit)
postprocess.validate_v4_sweep(
    output,
    checkpoint=checkpoint,
    expected_run_dir=run_dir,
    source_lock_path=postprocess.V4_SOURCE_LOCK,
    source_lock_sha256=postprocess.V4_SOURCE_LOCK_SHA256,
)
assignment = payload.get("audit", {}).get("device_assignment", {})
expected_assignment = {
    "device": "cuda:0",
    "physical_gpu_index": physical_index,
    "physical_gpu_uuid": physical_uuid,
    "cuda_visible_devices": physical_uuid,
    "device_name": "NVIDIA GeForce RTX 5090",
}
if assignment != expected_assignment:
    raise SystemExit(
        "sweep device assignment differs: "
        f"expected={expected_assignment!r} observed={assignment!r}"
    )
print(
    "TPDNERV8V4TAIL_EVAL_VERIFIED"
    f" checkpoint={checkpoint}"
    f" role={payload['checkpoint_role']}"
    f" physical_gpu={physical_index}"
    f" output={output}",
    flush=True,
)
PY
}

# Completion and the exact 1..800 metric history are checked on CPU before
# either a preflight succeeds or this worker touches CUDA.
"$v4_eval_python" - "$v4_eval_run_dir" "$v4_eval_checkpoint" <<'PY'
from pathlib import Path
import sys

from experiments import (
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as evaluator,
)

audit = evaluator.validate_run_artifacts(Path(sys.argv[1]), sys.argv[2])
if audit["metric_event_count"] != 800:
    raise SystemExit("V4 formal metric history is not exactly 800 events")
print(
    "TPDNERV8V4TAIL_EVAL_INPUT_READY"
    f" checkpoint={sys.argv[2]}"
    f" checkpoint_epoch={audit['checkpoint_epoch']}"
    " metric_epochs=1..800",
    flush=True,
)
PY

if [[ -f "$v4_eval_output" && ! -L "$v4_eval_output" ]]; then
    v4_eval_validate_output
    echo "TPDNERV8V4TAIL_EVAL_IDEMPOTENT_COMPLETE checkpoint=$v4_eval_checkpoint role=$v4_eval_role physical_gpu=$v4_eval_physical_index"
    exit 0
fi

if [[ "$v4_eval_mode" == "preflight" ]]; then
    echo "TPDNERV8V4TAIL_EVAL_PREFLIGHT_OK checkpoint=$v4_eval_checkpoint role=$v4_eval_role physical_gpu=$v4_eval_physical_index logical_device=cuda:0 output_state=absent"
    exit 0
fi

v4_eval_lock_dir="$v4_eval_result_root/.finalizer_locks"
if [[ -L "$v4_eval_lock_dir" \
    || ( -e "$v4_eval_lock_dir" && ! -d "$v4_eval_lock_dir" ) ]]; then
    v4_eval_abort "lock_directory_nonregular"
fi
mkdir -p "$v4_eval_lock_dir"
v4_eval_lock="$v4_eval_lock_dir/eval_gpu${v4_eval_physical_index}.lock"
if [[ -L "$v4_eval_lock" \
    || ( -e "$v4_eval_lock" && ! -f "$v4_eval_lock" ) ]]; then
    v4_eval_abort "gpu_claim_nonregular"
fi
exec 9>>"$v4_eval_lock"
[[ -f "$v4_eval_lock" && ! -L "$v4_eval_lock" ]] \
    || v4_eval_abort "gpu_claim_nonregular_after_open"
if ! flock -n 9; then
    echo "TPDNERV8V4TAIL_EVAL_RETRY reason=gpu_claim_held checkpoint=$v4_eval_checkpoint physical_gpu=$v4_eval_physical_index" >&2
    exit 75
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$v4_eval_gpu_uuid"
export TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_INDEX="$v4_eval_physical_index"
export TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_PHYSICAL_GPU_UUID="$v4_eval_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

"$v4_eval_python" "$v4_eval_script" \
    --run-dir "$v4_eval_run_dir" \
    --checkpoint "$v4_eval_checkpoint" \
    --device cuda:0 \
    --expected-epochs 800

v4_eval_validate_output
echo "TPDNERV8V4TAIL_EVAL_COMPLETE checkpoint=$v4_eval_checkpoint role=$v4_eval_role physical_gpu=$v4_eval_physical_index logical_device=cuda:0 output=$v4_eval_output"
