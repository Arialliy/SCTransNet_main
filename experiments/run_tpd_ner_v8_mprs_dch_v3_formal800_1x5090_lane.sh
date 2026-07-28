#!/usr/bin/env bash
set -euo pipefail

v3_lane_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    v3_lane_mode="preflight"
    shift
fi
if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 [--preflight] PHYSICAL_GPU_INDEX GPU_UUID" >&2
    exit 2
fi

v3_physical_index="$1"
v3_gpu_uuid="$2"
v3_variant="tpd_ner_v8_mprs_dch_v3_full_relay_on"
v3_repo="${TPD_NER_V8_V3_REPO:-/home/ly/SCTransNet_main}"
v3_python="${TPD_NER_V8_V3_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v3_trainer="$v3_repo/experiments/train_tpd_ner_v8_mprs_dch_v3_exact.py"
v3_manifest_tool="$v3_repo/experiments/freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py"
v3_training_lock="${TPD_NER_V8_V3_TRAINING_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v3_exact_source_lock.json}"
v3_acceptance_lock="${TPD_NER_V8_V3_ACCEPTANCE_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v3_acceptance_source_lock.json}"
v3_upstream_v2_training_lock="${TPD_NER_V8_V3_UPSTREAM_V2_TRAINING_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v2_exact_source_lock.json}"
v3_upstream_v2_acceptance_lock="${TPD_NER_V8_V3_UPSTREAM_V2_ACCEPTANCE_LOCK:-$v3_repo/experiments/tpd_ner_v8_mprs_dch_v2_acceptance_source_lock.json}"
v3_result_root="${TPD_NER_V8_V3_RESULT_ROOT:-$v3_repo/experiments/results/tpd_ner_v8_mprs_dch_v3_exact_v1}"
v3_run_tag="formal800_exact_v3_seed42"
v3_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v3_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"

case "$v3_physical_index:$v3_gpu_uuid" in
    "2:$v3_gpu2_uuid"|"3:$v3_gpu3_uuid")
        ;;
    *)
        echo "TPDNERV8V3_LANE_ABORT reason=invalid_gpu_mapping physical_gpu=$v3_physical_index gpu_uuid=$v3_gpu_uuid" >&2
        exit 2
        ;;
esac

# The configured interpreter may intentionally be a symlink.  Executability,
# rather than regular-file identity, is the interpreter contract.
[[ -x "$v3_python" ]] || {
    echo "TPDNERV8V3_LANE_ABORT reason=python_not_executable path=$v3_python" >&2
    exit 1
}
for v3_required_file in \
    "$v3_trainer" \
    "$v3_manifest_tool" \
    "$v3_training_lock" \
    "$v3_acceptance_lock" \
    "$v3_upstream_v2_training_lock" \
    "$v3_upstream_v2_acceptance_lock"
do
    [[ -f "$v3_required_file" && ! -L "$v3_required_file" ]] || {
        echo "TPDNERV8V3_LANE_ABORT reason=missing_required_file path=$v3_required_file" >&2
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
export CUDA_VISIBLE_DEVICES="$v3_gpu_uuid"
export TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_INDEX="$v3_physical_index"
export TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_UUID="$v3_gpu_uuid"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

cd "$v3_repo"

"$v3_python" "$v3_manifest_tool" \
    --mode verify \
    --kind all \
    --dataset-dir "$v3_repo/datasets" \
    --training-lock "$v3_training_lock" \
    --acceptance-lock "$v3_acceptance_lock" \
    --upstream-v2-training-lock "$v3_upstream_v2_training_lock" \
    --upstream-v2-acceptance-lock "$v3_upstream_v2_acceptance_lock"

"$v3_python" - "$v3_gpu_uuid" "$v3_physical_index" <<'PY'
import os
import sys

import torch

from experiments.train_tpd_clean_v8_mprs_dch_exact import normalized_gpu_uuid

expected_uuid, expected_index = sys.argv[1:]
if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
    raise SystemExit("CUDA_VISIBLE_DEVICES differs from the selected UUID")
if os.environ.get(
    "TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_INDEX"
) != expected_index:
    raise SystemExit("physical GPU index differs from the selected V3 lane")
if os.environ.get(
    "TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_UUID"
) != expected_uuid:
    raise SystemExit("physical GPU UUID differs from the selected V3 lane")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the V3 lane must expose exactly one CUDA device")
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
    "TPDNERV8V3_GPU_OK"
    f" physical_index={expected_index}"
    f" uuid={actual_uuid}"
    " logical_device=cuda:0",
    flush=True,
)
PY

v3_run_dir="$v3_result_root/NUDT-SIRST/$v3_variant/seed_42_${v3_run_tag}"
v3_initialization="$(
    "$v3_python" - \
        "$v3_run_dir" \
        "$v3_variant" \
        "$v3_training_lock" <<'PY'
import json
import pathlib
import sys

import torch

from experiments import tpd_exact_epoch_journal as epoch_journal
from experiments import train_tpd_ner_v8_mprs_dch_v3_exact as exact

run_dir = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
training_lock = pathlib.Path(sys.argv[3])
if variant != exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON:
    raise SystemExit("lane variant is not the sole formal V3 candidate")
if not run_dir.exists() and not run_dir.is_symlink():
    print("fresh")
    raise SystemExit(0)
if run_dir.is_symlink() or not run_dir.is_dir():
    raise SystemExit(f"run path must be a regular directory: {run_dir}")

protocol_path = run_dir / "protocol.json"
if not protocol_path.is_file() or protocol_path.is_symlink():
    raise SystemExit("existing V3 run directory has no exact protocol")
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
if protocol.get("schema") != exact.ENTRY_SCHEMA:
    raise SystemExit("existing protocol is not V3 exact")
identity = exact.require_v3_run_identity(
    protocol.get("run_identity"),
    label="lane protocol",
    expected_variant=variant,
)
expected_run_id = (
    f"{exact.RUN_ID_PREFIX}NUDT-SIRST:{variant}:"
    f"seed-42:split-20260722:{exact.FORMAL_RUN_TAG}"
)
if (
    identity.get("dataset") != "NUDT-SIRST"
    or identity.get("seed") != 42
    or identity.get("split_seed") != 20260722
    or identity.get("run_id") != expected_run_id
):
    raise SystemExit("existing V3 protocol run identity differs")
if (
    identity["source_locks"].get(exact.SOURCE_LOCK_KEY)
    != exact.file_sha256(training_lock)
):
    raise SystemExit("existing V3 protocol source-lock identity differs")

summary_path = run_dir / "summary.json"
if summary_path.is_symlink():
    raise SystemExit("existing V3 summary must not be a symbolic link")
if summary_path.exists() and not summary_path.is_file():
    raise SystemExit("existing V3 summary must be a regular file")
if summary_path.is_file():
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"existing V3 summary is invalid or truncated: {error}"
        ) from error
    if summary.get("status") == "complete":
        if (
            summary.get("schema") != exact.COMPLETION_SUMMARY_SCHEMA
            or summary.get("variant") != variant
            or summary.get("seed") != 42
            or summary.get("split_seed") != 20260722
        ):
            raise SystemExit("complete V3 summary identity differs")
        summary_identity = exact.require_v3_run_identity(
            summary.get("run_identity"),
            label="lane completion summary",
            expected_variant=variant,
        )
        if summary_identity != identity:
            raise SystemExit("complete V3 summary run identity differs")

        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file() or metrics_path.is_symlink():
            raise SystemExit(
                "complete V3 run has no regular metrics journal"
            )
        metrics_text = metrics_path.read_text(encoding="utf-8")
        if not metrics_text.endswith("\n"):
            raise SystemExit("complete V3 metrics journal is truncated")
        metric_lines = metrics_text.splitlines()
        if len(metric_lines) != exact.FORMAL_EPOCHS or any(
            not line.strip() for line in metric_lines
        ):
            raise SystemExit(
                "complete V3 metrics journal does not contain exactly "
                "800 events"
            )
        events = [
            json.loads(line)
            for line in metric_lines
        ]
        if any(
            not isinstance(event, dict)
            or event.get("epoch") != expected_epoch
            for expected_epoch, event in enumerate(events, start=1)
        ):
            raise SystemExit(
                "complete V3 metrics epochs are not contiguous 1..800"
            )
        if any(
            not set(exact.STORED_VALIDATION_METRICS).issubset(event)
            for event in events
        ):
            raise SystemExit(
                "complete V3 metrics journal lacks validation fields"
            )

        checkpoint_contracts = (
            ("best.pth.tar", "best_validation_pd_primary"),
            (
                "best_miou.pth.tar",
                "best_validation_miou_secondary",
            ),
            ("last.pth.tar", "last_evaluated_epoch"),
        )
        last_checkpoint_epoch = None
        last_checkpoint_role = None
        for checkpoint_name, expected_role in checkpoint_contracts:
            checkpoint_path = run_dir / checkpoint_name
            if (
                not checkpoint_path.is_file()
                or checkpoint_path.is_symlink()
            ):
                raise SystemExit(
                    f"complete V3 run lacks regular checkpoint: "
                    f"{checkpoint_name}"
                )
            checkpoint = exact.require_evaluator_checkpoint_payload(
                torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                ),
                expected_variant=variant,
            )
            if checkpoint.get("run_identity") != identity:
                raise SystemExit(
                    f"complete V3 checkpoint run identity differs: "
                    f"{checkpoint_name}"
                )
            if checkpoint.get("checkpoint_role") != expected_role:
                raise SystemExit(
                    f"complete V3 checkpoint role differs: "
                    f"{checkpoint_name}"
                )
            if checkpoint_name == "last.pth.tar":
                last_checkpoint_epoch = checkpoint.get("epoch")
                last_checkpoint_role = checkpoint.get("checkpoint_role")
            del checkpoint
        if (
            last_checkpoint_epoch != exact.FORMAL_EPOCHS
            or last_checkpoint_role != "last_evaluated_epoch"
        ):
            raise SystemExit(
                "complete V3 last checkpoint is not evaluated epoch 800"
            )

        journal_root = run_dir / "exact_journal"
        active_path = journal_root / "active.json"
        if (
            journal_root.is_symlink()
            or not journal_root.is_dir()
            or not active_path.is_file()
            or active_path.is_symlink()
        ):
            raise SystemExit(
                "complete V3 run has no regular active exact journal"
            )
        active = epoch_journal.ExactEpochJournal(
            journal_root
        ).load_active()
        if active is None or active.epoch != exact.FORMAL_EPOCHS:
            raise SystemExit(
                "complete V3 active journal is not committed at epoch 800"
            )
        if active.metrics_path.read_bytes() != metrics_path.read_bytes():
            raise SystemExit(
                "complete V3 derived metrics differ from active journal"
            )
        active_payload = torch.load(
            active.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        active_identity = exact.require_v3_run_identity(
            active_payload.get("run_identity"),
            label="lane active exact journal",
            expected_variant=variant,
        )
        if active_identity != identity:
            raise SystemExit(
                "complete V3 active journal run identity differs"
            )
        print("complete")
        raise SystemExit(0)

active_path = run_dir / "exact_journal" / "active.json"
if not active_path.is_file() or active_path.is_symlink():
    raise SystemExit("incomplete V3 run has no committed exact epoch")
print("exact-resume")
PY
)"

echo "TPDNERV8V3_LANE_READY variant=$v3_variant physical_gpu=$v3_physical_index initialization=$v3_initialization run_dir=$v3_run_dir"

if [[ "$v3_lane_mode" == "preflight" || "$v3_initialization" == "complete" ]]; then
    exit 0
fi

v3_init_flag="--fresh"
if [[ "$v3_initialization" == "exact-resume" ]]; then
    v3_init_flag="--exact-resume"
fi

exec "$v3_python" "$v3_trainer" \
    --variant "$v3_variant" \
    --dataset NUDT-SIRST \
    --dataset-dir "$v3_repo/datasets" \
    --output-root "$v3_result_root" \
    --run-tag "$v3_run_tag" \
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
    --exact-source-lock "$v3_training_lock" \
    "$v3_init_flag"
