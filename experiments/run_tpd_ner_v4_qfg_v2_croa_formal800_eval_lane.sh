#!/usr/bin/env bash
set -Eeuo pipefail

qfg_eval_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    qfg_eval_mode="preflight"
    shift
fi
if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 [--preflight] {qfg_only|tss_qfg} {best.pth.tar|best_miou.pth.tar} {2|3} GPU_UUID" >&2
    exit 2
fi

qfg_eval_variant="$1"
qfg_eval_checkpoint="$2"
qfg_eval_physical_index="$3"
qfg_eval_gpu_uuid="$4"
qfg_eval_repo="${TPD_NER_V4_QFG_V2_CROA_REPO:-/home/ly/SCTransNet_main}"
qfg_eval_python="${TPD_NER_V4_QFG_V2_CROA_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
qfg_eval_evaluator="$qfg_eval_repo/experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py"
qfg_eval_freezer="$qfg_eval_repo/experiments/freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock.py"
qfg_eval_source_lock="${TPD_NER_V4_QFG_V2_CROA_SOURCE_LOCK:-$qfg_eval_repo/experiments/tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json}"
qfg_eval_result_root="${TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT:-$qfg_eval_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized}"
qfg_eval_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
qfg_eval_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

qfg_eval_abort() {
    echo "TPDNERV4QFG_EVAL_ABORT reason=$1 variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint physical_gpu=$qfg_eval_physical_index" >&2
    exit 64
}

qfg_eval_map_error() {
    local qfg_eval_status="$?"
    local qfg_eval_line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    if [[ "$qfg_eval_status" -eq 64 || "$qfg_eval_status" -eq 75 ]]; then
        exit "$qfg_eval_status"
    fi
    echo "TPDNERV4QFG_EVAL_ABORT reason=stage_failed original_exit=$qfg_eval_status line=$qfg_eval_line variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint physical_gpu=$qfg_eval_physical_index" >&2
    exit 64
}
trap qfg_eval_map_error ERR

case "$qfg_eval_variant" in
    qfg_only)
        qfg_eval_run_tag="formal800_qfg_only"
        ;;
    tss_qfg)
        qfg_eval_run_tag="formal800_tss_qfg"
        ;;
    *)
        qfg_eval_abort "invalid_variant"
        ;;
esac

case "$qfg_eval_checkpoint" in
    best.pth.tar)
        qfg_eval_output_name="pd_fa_sweep_best.pth.json"
        qfg_eval_role="best_validation_pd_primary"
        ;;
    best_miou.pth.tar)
        qfg_eval_output_name="pd_fa_sweep_best_miou.pth.json"
        qfg_eval_role="best_validation_miou_secondary"
        ;;
    *)
        qfg_eval_abort "invalid_checkpoint"
        ;;
esac

case "$qfg_eval_physical_index:$qfg_eval_gpu_uuid" in
    "2:$qfg_eval_gpu2_uuid"|"3:$qfg_eval_gpu3_uuid")
        ;;
    *)
        qfg_eval_abort "invalid_gpu_uuid_mapping"
        ;;
esac

qfg_eval_run_dir="$qfg_eval_result_root/NUDT-SIRST/$qfg_eval_variant/seed_42_$qfg_eval_run_tag"
qfg_eval_output="$qfg_eval_run_dir/$qfg_eval_output_name"

[[ -d "$qfg_eval_repo" && ! -L "$qfg_eval_repo" ]] \
    || qfg_eval_abort "invalid_repo"
[[ -x "$qfg_eval_python" ]] \
    || qfg_eval_abort "python_not_executable"
[[ -d "$qfg_eval_repo/datasets" && ! -L "$qfg_eval_repo/datasets" ]] \
    || qfg_eval_abort "dataset_directory_nonregular"
for qfg_eval_required_file in \
    "$qfg_eval_evaluator" \
    "$qfg_eval_freezer" \
    "$qfg_eval_source_lock"
do
    [[ -f "$qfg_eval_required_file" && ! -L "$qfg_eval_required_file" ]] \
        || qfg_eval_abort "required_source_nonregular"
done
[[ -d "$qfg_eval_result_root" && ! -L "$qfg_eval_result_root" ]] \
    || qfg_eval_abort "result_root_nonregular"
[[ -d "$qfg_eval_run_dir" && ! -L "$qfg_eval_run_dir" ]] \
    || qfg_eval_abort "run_directory_nonregular"
for qfg_eval_run_file in \
    "$qfg_eval_run_dir/protocol.json" \
    "$qfg_eval_run_dir/split.json" \
    "$qfg_eval_run_dir/summary.json" \
    "$qfg_eval_run_dir/metrics.jsonl" \
    "$qfg_eval_run_dir/$qfg_eval_checkpoint"
do
    [[ -f "$qfg_eval_run_file" && ! -L "$qfg_eval_run_file" ]] \
        || qfg_eval_abort "run_artifact_nonregular"
done
if [[ -L "$qfg_eval_output" \
    || ( -e "$qfg_eval_output" && ! -f "$qfg_eval_output" ) ]]; then
    qfg_eval_abort "sweep_output_nonregular"
fi

# These are exported before any Python process imports torch.  The worker
# always exposes one UUID-bound logical cuda:0 and fixes all seven CPU thread
# controls used elsewhere in the formal800 launchers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$qfg_eval_gpu_uuid"
export TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX="$qfg_eval_physical_index"
export TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$qfg_eval_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

cd "$qfg_eval_repo"

qfg_eval_verify_frozen_sources() {
    "$qfg_eval_python" "$qfg_eval_freezer" \
        --verify \
        --dataset-dir "$qfg_eval_repo/datasets" \
        --output "$qfg_eval_source_lock"
}

qfg_eval_validate_input() {
    "$qfg_eval_python" - \
        "$qfg_eval_run_dir" \
        "$qfg_eval_variant" \
        "$qfg_eval_checkpoint" \
        "$qfg_eval_source_lock" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments import (
    evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa as evaluator,
)
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as exact


run_dir = Path(sys.argv[1]).resolve()
variant = sys.argv[2]
checkpoint_name = sys.argv[3]
source_lock = Path(sys.argv[4]).resolve()
metrics_path = run_dir / "metrics.jsonl"
summary_path = run_dir / "summary.json"
protocol_path = run_dir / "protocol.json"

if metrics_path.is_symlink() or not metrics_path.is_file():
    raise SystemExit("QFG formal metrics must be a regular file")
raw_lines = metrics_path.read_text(encoding="utf-8").splitlines()
if (
    len(raw_lines) != exact.FORMAL_EPOCHS
    or any(not line.strip() for line in raw_lines)
):
    raise SystemExit("QFG formal metrics must have exactly 800 nonblank rows")
try:
    parsed = [json.loads(line) for line in raw_lines]
except json.JSONDecodeError as error:
    raise SystemExit(f"QFG formal metrics JSON is invalid: {error}") from error
if (
    not all(isinstance(event, dict) for event in parsed)
    or [event.get("epoch") for event in parsed]
    != list(range(1, exact.FORMAL_EPOCHS + 1))
):
    raise SystemExit("QFG formal metrics must be contiguous epochs 1..800")
if any(event.get("variant") != variant for event in parsed):
    raise SystemExit("QFG formal metric variant differs")

# The trainer validator checks every stored validation metric and all required
# QFG loss fields; the selection policy then verifies every new-best flag and
# reconstructs both globally selected checkpoints from the complete history.
events = exact._load_complete_events(metrics_path, exact.FORMAL_EPOCHS)
policy = exact.exact_runner.pd_miou_selection_policy(
    stored_metrics=exact.STORED_VALIDATION_METRICS,
)
try:
    selection = policy.recompute(events, require_flags=True)
except exact.exact_runner.ExactRunnerError as error:
    raise SystemExit(
        f"QFG formal metric selection differs: {error}"
    ) from error

audit = evaluator.validate_run_artifacts(run_dir, checkpoint_name)
if audit["variant"] != variant:
    raise SystemExit("QFG evaluator audit variant differs")
role = evaluator.CHECKPOINT_ROLES[checkpoint_name]
slot = (
    "primary"
    if role == "best_validation_pd_primary"
    else "secondary"
)
selected = selection[slot]
checks = (
    ("checkpoint role", audit["checkpoint_role"], role),
    ("checkpoint epoch", audit["checkpoint_epoch"], selected["epoch"]),
)
for label, observed, expected in checks:
    if observed != expected:
        raise SystemExit(
            f"QFG {label} differs: expected={expected!r} "
            f"observed={observed!r}"
        )
evaluator._canonical_equal(
    "QFG checkpoint/global-selection metrics",
    audit["checkpoint_validation_metrics"],
    selected["metrics"],
)

summary = evaluator._load_json(summary_path)
protocol = evaluator._load_json(protocol_path)
if summary.get("status") != "complete":
    raise SystemExit("QFG formal summary is not complete")
primary = selection["primary"]
secondary = selection["secondary"]
for label, observed, expected in (
    ("summary best epoch", summary.get("best_epoch"), primary["epoch"]),
    ("summary best Pd epoch", summary.get("best_pd_epoch"), primary["epoch"]),
    (
        "summary best mIoU epoch",
        summary.get("best_miou_epoch"),
        secondary["epoch"],
    ),
):
    if observed != expected:
        raise SystemExit(
            f"QFG {label} differs: expected={expected!r} "
            f"observed={observed!r}"
        )
for label, observed, expected in (
    (
        "summary best metrics",
        summary.get("best_validation_metrics"),
        primary["metrics"],
    ),
    (
        "summary best Pd metrics",
        summary.get("best_pd_validation_metrics"),
        primary["metrics"],
    ),
    (
        "summary best mIoU metrics",
        summary.get("best_miou_validation_metrics"),
        secondary["metrics"],
    ),
):
    evaluator._canonical_equal(label, observed, expected)

source_lock_sha256 = exact.file_sha256(source_lock)
run_source_locks = audit["run_identity"].get("source_locks")
if not isinstance(run_source_locks, dict):
    raise SystemExit("QFG run identity source locks are missing")
if (
    run_source_locks.get(exact.SOURCE_LOCK_KEY)
    != source_lock_sha256
):
    raise SystemExit("QFG run is not bound to the V2 optimized source lock")
for label, artifact in (
    ("protocol", protocol),
    ("summary", summary),
):
    evaluator._canonical_equal(
        f"QFG {label} source locks",
        artifact.get("source_locks"),
        run_source_locks,
    )

print(
    "TPDNERV4QFG_EVAL_INPUT_READY"
    f" variant={variant}"
    f" checkpoint={checkpoint_name}"
    f" role={role}"
    f" checkpoint_epoch={audit['checkpoint_epoch']}"
    " metric_events=800 metric_epochs=1..800"
    f" source_lock_sha256={source_lock_sha256}",
    flush=True,
)
PY
}

qfg_eval_validate_output() {
    "$qfg_eval_python" - \
        "$qfg_eval_run_dir" \
        "$qfg_eval_variant" \
        "$qfg_eval_checkpoint" \
        "$qfg_eval_output" \
        "$qfg_eval_source_lock" \
        "$qfg_eval_physical_index" \
        "$qfg_eval_gpu_uuid" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments import (
    evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa as evaluator,
)
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as exact


run_dir = Path(sys.argv[1]).resolve()
variant = sys.argv[2]
checkpoint_name = sys.argv[3]
output = Path(sys.argv[4]).resolve()
source_lock = Path(sys.argv[5]).resolve()
physical_index = int(sys.argv[6])
physical_uuid = sys.argv[7]
if output.is_symlink() or not output.is_file():
    raise SystemExit(f"QFG sweep must be a regular file: {output}")
before_sha256 = evaluator._sha256_file(output)
try:
    payload = json.loads(output.read_text(encoding="utf-8"))
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"QFG sweep JSON is invalid: {error}") from error
if not isinstance(payload, dict):
    raise SystemExit("QFG sweep must contain one JSON object")

audit = evaluator.validate_run_artifacts(run_dir, checkpoint_name)
evaluator.validate_output_identity(payload, artifact_audit=audit)
role = evaluator.CHECKPOINT_ROLES[checkpoint_name]
for label, observed, expected in (
    ("variant", payload.get("variant"), variant),
    ("checkpoint role", payload.get("checkpoint_role"), role),
    (
        "checkpoint epoch",
        payload.get("checkpoint_epoch"),
        audit["checkpoint_epoch"],
    ),
    (
        "checkpoint SHA",
        payload.get("checkpoint_sha256"),
        audit["checkpoint_sha256"],
    ),
):
    if observed != expected:
        raise SystemExit(
            f"QFG sweep {label} differs: expected={expected!r} "
            f"observed={observed!r}"
        )
if Path(str(payload.get("run_directory"))).resolve() != run_dir:
    raise SystemExit("QFG sweep run directory differs")
checkpoint_path = run_dir / checkpoint_name
if Path(str(payload.get("checkpoint"))).resolve() != checkpoint_path:
    raise SystemExit("QFG sweep checkpoint path differs")
evaluator._canonical_equal(
    "QFG sweep checkpoint metrics",
    payload.get("checkpoint_validation_metrics"),
    audit["checkpoint_validation_metrics"],
)
evaluator._canonical_equal(
    "QFG sweep evaluation source binding",
    payload.get("evaluation_source_binding"),
    audit["source_binding"],
)
evaluator._canonical_equal(
    "QFG sweep evaluator contract",
    payload.get("evaluator_contract"),
    evaluator.evaluator_contract(),
)

source_lock_sha256 = exact.file_sha256(source_lock)
if (
    audit["run_identity"].get("source_locks", {}).get(
        exact.SOURCE_LOCK_KEY
    )
    != source_lock_sha256
):
    raise SystemExit("QFG sweep run is not bound to the optimized source lock")
expected_assignment = {
    "device": "cuda:0",
    "physical_gpu_index": physical_index,
    "physical_gpu_uuid": physical_uuid,
    "cuda_visible_devices": physical_uuid,
    "device_name": "NVIDIA GeForce RTX 5090",
}
assignment = payload.get("audit", {}).get("device_assignment")
if assignment != expected_assignment:
    raise SystemExit(
        "QFG sweep device assignment differs: "
        f"expected={expected_assignment!r} observed={assignment!r}"
    )
if evaluator._sha256_file(output) != before_sha256:
    raise SystemExit("QFG sweep changed while it was being verified")
print(
    "TPDNERV4QFG_EVAL_OUTPUT_VERIFIED"
    f" variant={variant}"
    f" checkpoint={checkpoint_name}"
    f" role={role}"
    f" physical_gpu={physical_index}"
    f" output={output}",
    flush=True,
)
PY
}

qfg_eval_gpu_probe() {
    "$qfg_eval_python" - \
        "$qfg_eval_gpu_uuid" \
        "$qfg_eval_physical_index" <<'PY'
from __future__ import annotations

import os
import sys

import torch

from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as exact
from experiments.train_tpd_clean_v8_mprs_dch_exact import (
    normalized_gpu_uuid,
)


expected_uuid, expected_index = sys.argv[1:]
if exact.PHYSICAL_GPU_UUIDS.get(expected_index) != expected_uuid:
    raise SystemExit("QFG evaluator GPU mapping differs from the trainer")
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
            f"QFG evaluator environment differs for {name}: "
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
        raise SystemExit(f"QFG evaluator thread setting differs for {name}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("QFG evaluator must expose exactly one CUDA device")
device_name = torch.cuda.get_device_name(0)
if device_name != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected cuda:0 model: {device_name}")
actual_uuid = normalized_gpu_uuid(
    getattr(torch.cuda.get_device_properties(0), "uuid", "")
)
if actual_uuid != expected_uuid:
    raise SystemExit(
        f"cuda:0 UUID differs: expected={expected_uuid} "
        f"observed={actual_uuid}"
    )
if torch.get_num_threads() != 1:
    raise SystemExit(
        f"torch CPU thread count differs: {torch.get_num_threads()}"
    )
print(
    "TPDNERV4QFG_EVAL_GPU_OK"
    f" physical_gpu={expected_index}"
    f" uuid={actual_uuid}"
    " model=NVIDIA_GeForce_RTX_5090 logical_device=cuda:0",
    flush=True,
)
PY
}

# Existing checkpoint-local output is never replaced.  It is accepted as an
# idempotent completion only after the frozen sources, full training history,
# selected checkpoint, evaluator binding, metrics, and device assignment all
# pass live verification.
if [[ -f "$qfg_eval_output" && ! -L "$qfg_eval_output" ]]; then
    qfg_eval_verify_frozen_sources
    qfg_eval_validate_input
    qfg_eval_validate_output
    echo "TPDNERV4QFG_EVAL_IDEMPOTENT_COMPLETE variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint role=$qfg_eval_role physical_gpu=$qfg_eval_physical_index output=$qfg_eval_output"
    exit 0
fi

if [[ "$qfg_eval_mode" == "preflight" ]]; then
    qfg_eval_verify_frozen_sources
    qfg_eval_validate_input
    echo "TPDNERV4QFG_EVAL_PREFLIGHT_OK variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint role=$qfg_eval_role physical_gpu=$qfg_eval_physical_index logical_device=cuda:0 output_state=absent writes_performed=false"
    exit 0
fi

qfg_eval_lock_dir="$qfg_eval_result_root/.evaluation_locks"
if [[ -L "$qfg_eval_lock_dir" \
    || ( -e "$qfg_eval_lock_dir" && ! -d "$qfg_eval_lock_dir" ) ]]; then
    qfg_eval_abort "lock_directory_nonregular"
fi
mkdir -p "$qfg_eval_lock_dir"
qfg_eval_job_lock="$qfg_eval_lock_dir/${qfg_eval_variant}_${qfg_eval_output_name}.lock"
qfg_eval_gpu_lock="$qfg_eval_lock_dir/gpu${qfg_eval_physical_index}.lock"
for qfg_eval_lock in "$qfg_eval_job_lock" "$qfg_eval_gpu_lock"
do
    if [[ -L "$qfg_eval_lock" \
        || ( -e "$qfg_eval_lock" && ! -f "$qfg_eval_lock" ) ]]; then
        qfg_eval_abort "claim_nonregular"
    fi
done
exec 8>>"$qfg_eval_job_lock"
[[ -f "$qfg_eval_job_lock" && ! -L "$qfg_eval_job_lock" ]] \
    || qfg_eval_abort "job_claim_nonregular_after_open"
if ! flock -n 8; then
    echo "TPDNERV4QFG_EVAL_RETRY reason=checkpoint_claim_held variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint" >&2
    exit 75
fi
exec 9>>"$qfg_eval_gpu_lock"
[[ -f "$qfg_eval_gpu_lock" && ! -L "$qfg_eval_gpu_lock" ]] \
    || qfg_eval_abort "gpu_claim_nonregular_after_open"
if ! flock -n 9; then
    echo "TPDNERV4QFG_EVAL_RETRY reason=gpu_claim_held variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint physical_gpu=$qfg_eval_physical_index" >&2
    exit 75
fi

# Recheck the output after both nonblocking claims close the cross-GPU race.
if [[ -L "$qfg_eval_output" \
    || ( -e "$qfg_eval_output" && ! -f "$qfg_eval_output" ) ]]; then
    qfg_eval_abort "sweep_output_nonregular_after_claim"
fi
qfg_eval_verify_frozen_sources
qfg_eval_validate_input
if [[ -f "$qfg_eval_output" && ! -L "$qfg_eval_output" ]]; then
    qfg_eval_validate_output
    echo "TPDNERV4QFG_EVAL_IDEMPOTENT_COMPLETE variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint role=$qfg_eval_role physical_gpu=$qfg_eval_physical_index output=$qfg_eval_output"
    exit 0
fi

qfg_eval_gpu_probe
"$qfg_eval_python" "$qfg_eval_evaluator" \
    --run-dir "$qfg_eval_run_dir" \
    --checkpoint "$qfg_eval_checkpoint" \
    --device cuda:0 \
    --expected-epochs 800

qfg_eval_verify_frozen_sources
qfg_eval_validate_output
echo "TPDNERV4QFG_EVAL_COMPLETE variant=$qfg_eval_variant checkpoint=$qfg_eval_checkpoint role=$qfg_eval_role physical_gpu=$qfg_eval_physical_index logical_device=cuda:0 output=$qfg_eval_output"
