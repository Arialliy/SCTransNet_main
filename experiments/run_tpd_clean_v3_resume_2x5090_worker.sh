#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 VARIANT SEED GPU_UUID RUN_DIR" >&2
    exit 2
fi

v3_variant="$1"
v3_seed="$2"
v3_gpu_uuid="$3"
v3_run_dir="$4"
v3_repo="/home/ly/SCTransNet_main"
v3_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v3_result_root="$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1"
v3_run_tag="screen800_pd_fp32_shared4x5090_v1"
v3_resume_program="$v3_repo/experiments/resume_tpd_clean_v3.py"
v3_resume_root="$v3_result_root/resume_2x5090_v1"
v3_resume_manifest_root="$v3_resume_root/manifests"
v3_resume_log_root="$v3_resume_root/logs"
v3_resume_boundary_root="$v3_resume_root/boundaries"
v3_resume_lock_root="$v3_resume_root/.locks"
v3_original_lock_root="$v3_result_root/.locks"
v3_source_lock="$v3_repo/experiments/tpd_clean_v3_resume_2x_source_lock.json"
v3_original_source_lock="$v3_repo/experiments/tpd_clean_v3_screen800_source_lock.json"
v3_old_clean_lock="$v3_repo/experiments/tpd_clean_screen800_source_lock.json"
v3_old_ner_lock="$v3_repo/experiments/tpd_ner_v1_source_lock.json"
v3_training_data_sha256="39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
v3_cpu_threads=1
export OMP_NUM_THREADS="$v3_cpu_threads"
export MKL_NUM_THREADS="$v3_cpu_threads"
export OPENBLAS_NUM_THREADS="$v3_cpu_threads"
export NUMEXPR_NUM_THREADS="$v3_cpu_threads"

case "$v3_variant:$v3_seed:$v3_gpu_uuid" in
    tpd_clean_v3_full:42:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        v3_gpu_index=3
        ;;
    tpd_clean_v3_sal_capacity:42:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        v3_gpu_index=2
        ;;
    tpd_clean_v3_full:3407:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)
        v3_gpu_index=2
        ;;
    tpd_clean_v3_sal_capacity:3407:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)
        v3_gpu_index=3
        ;;
    *)
        echo "TPDCLEANV3_RESUME_2X_ABORT reason=invalid_job_mapping variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_gpu_uuid" >&2
        exit 2
        ;;
esac

v3_expected_run_dir="$v3_result_root/NUDT-SIRST/$v3_variant/seed_${v3_seed}_${v3_run_tag}"
if [[ "$v3_run_dir" != "$v3_expected_run_dir" ]]; then
    echo "TPDCLEANV3_RESUME_2X_ABORT reason=noncanonical_run_dir expected=$v3_expected_run_dir actual=$v3_run_dir" >&2
    exit 2
fi

v3_original_launch="$v3_result_root/launch/${v3_variant}_seed${v3_seed}.json"
v3_original_log="$v3_result_root/logs/${v3_variant}_seed${v3_seed}.log"
v3_resume_manifest="$v3_resume_manifest_root/${v3_variant}_seed${v3_seed}.json"
v3_resume_log="$v3_resume_log_root/${v3_variant}_seed${v3_seed}.log"

mkdir -p \
    "$v3_resume_manifest_root" \
    "$v3_resume_log_root" \
    "$v3_resume_boundary_root" \
    "$v3_resume_lock_root"

for v3_new_artifact in "$v3_resume_manifest" "$v3_resume_log"; do
    if [[ -e "$v3_new_artifact" || -L "$v3_new_artifact" ]]; then
        echo "TPDCLEANV3_RESUME_2X_ABORT reason=resume_artifact_exists path=$v3_new_artifact" >&2
        exit 1
    fi
done

exec 8>>"$v3_original_lock_root/${v3_variant}_seed${v3_seed}.lock"
if ! flock -n 8; then
    echo "TPDCLEANV3_RESUME_2X_ABORT reason=original_worker_lock_held variant=$v3_variant seed=$v3_seed" >&2
    exit 1
fi
exec 9>"$v3_resume_lock_root/${v3_variant}_seed${v3_seed}.lock"
if ! flock -n 9; then
    echo "TPDCLEANV3_RESUME_2X_ABORT reason=resume_worker_lock_held variant=$v3_variant seed=$v3_seed" >&2
    exit 1
fi

exec > >(tee "$v3_resume_log") 2>&1
cd "$v3_repo"

v3_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

v3_verify_source_locks() {
    "$v3_python" - "$v3_repo" \
        "$v3_source_lock" \
        "$v3_original_source_lock" \
        "$v3_old_clean_lock" \
        "$v3_old_ner_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
expected_schemas = {
    "sctransnet_tpd_clean_v3_resume_2x_source_lock_v1",
    "sctransnet_tpd_clean_v3_screen800_source_lock_v1",
    "sctransnet_tpd_clean_screen800_source_lock_v1",
    "sctransnet_tpd_ner_v1_source_lock_v1",
}
seen = set()
file_count = 0
for lock_text in sys.argv[2:]:
    lock_path = pathlib.Path(lock_text)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise SystemExit(f"missing or linked source lock: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in expected_schemas:
        raise SystemExit(f"unexpected source-lock schema: {schema}")
    seen.add(schema)
    for relative, expected in payload["source_sha256"].items():
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing or linked source: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"source digest mismatch: {relative} "
                f"expected={expected} actual={actual}"
            )
        file_count += 1
if seen != expected_schemas:
    raise SystemExit(f"incomplete source-lock schemas: {sorted(seen)}")
print(
    "TPDCLEANV3_RESUME_2X_SOURCES_OK"
    f" locks={len(seen)} checked_entries={file_count}",
    flush=True,
)
PY
}

v3_verify_data() {
    local v3_actual
    v3_actual="$(
        timeout 300s "$v3_python" experiments/fingerprint_tpd_training_data.py \
            --dataset NUDT-SIRST
    )"
    if [[ "$v3_actual" != "$v3_training_data_sha256" ]]; then
        echo "TPDCLEANV3_RESUME_2X_ABORT reason=training_data_drift expected=$v3_training_data_sha256 actual=$v3_actual" >&2
        return 1
    fi
}

v3_verify_gpu() {
    local v3_actual_index
    local v3_name
    local v3_free
    v3_actual_index="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=index --format=csv,noheader,nounits
    )"
    v3_name="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=name --format=csv,noheader,nounits
    )"
    v3_free="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    if [[ "$v3_actual_index" != "$v3_gpu_index" ]]; then
        echo "TPDCLEANV3_RESUME_2X_ABORT reason=gpu_index_mismatch gpu_uuid=$v3_gpu_uuid expected=$v3_gpu_index actual=$v3_actual_index" >&2
        return 1
    fi
    if [[ "$v3_name" != "NVIDIA GeForce RTX 5090" ]]; then
        echo "TPDCLEANV3_RESUME_2X_ABORT reason=unexpected_gpu gpu_uuid=$v3_gpu_uuid name=$v3_name" >&2
        return 1
    fi
    if (( v3_free < 7500 )); then
        echo "TPDCLEANV3_RESUME_2X_ABORT reason=insufficient_memory gpu_uuid=$v3_gpu_uuid free_mib=$v3_free" >&2
        return 1
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v3_gpu_uuid" \
        "$v3_python" - "$v3_gpu_uuid" <<'PY'
import sys
import torch

expected_uuid = sys.argv[1]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"unexpected CUDA device: {torch.cuda.get_device_name(0)}")
x = torch.ones((64, 64), device="cuda:0")
if float((x @ x).sum().item()) != 262144.0:
    raise SystemExit("CUDA calculation returned an unexpected value")
torch.cuda.synchronize()
print(
    "TPDCLEANV3_RESUME_2X_GPU_OK"
    f" gpu_uuid={expected_uuid}"
    f" torch={torch.__version__}"
    f" cuda={torch.version.cuda}"
    f" capability={torch.cuda.get_device_capability(0)}",
    flush=True,
)
PY
}

v3_validate_boundary() {
    "$v3_python" - \
        "$v3_run_dir" \
        "$v3_variant" \
        "$v3_seed" \
        "$v3_original_launch" \
        "$v3_original_log" <<'PY'
import json
import math
import pathlib
import sys

import torch

run_dir = pathlib.Path(sys.argv[1])
variant = sys.argv[2]
seed = int(sys.argv[3])
original_launch_path = pathlib.Path(sys.argv[4])
original_log_path = pathlib.Path(sys.argv[5])

if not run_dir.is_dir() or run_dir.is_symlink():
    raise SystemExit(f"invalid run directory: {run_dir}")
required = (
    "metrics.jsonl",
    "last.pth.tar",
    "best.pth.tar",
    "best_miou.pth.tar",
    "protocol.json",
    "split.json",
)
for name in required:
    path = run_dir / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked boundary artifact: {path}")
for path in (original_launch_path, original_log_path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked original evidence: {path}")

metric_lines = (run_dir / "metrics.jsonl").read_text(
    encoding="utf-8"
).splitlines()
if not metric_lines:
    raise SystemExit("metrics.jsonl is empty")
events = [json.loads(line) for line in metric_lines]
epochs = [event.get("epoch") for event in events]
expected_epochs = list(range(1, len(events) + 1))
if epochs != expected_epochs:
    raise SystemExit("metrics epochs are not contiguous from 1")
boundary_epoch = len(events)
if not 1 <= boundary_epoch < 800:
    raise SystemExit(
        f"resume boundary must satisfy 1 <= epoch < 800: {boundary_epoch}"
    )
for event in events:
    if event.get("variant") != variant:
        raise SystemExit("metrics variant mismatch")
    for key, value in event.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise SystemExit(f"non-finite metric at epoch={event['epoch']}: {key}")

checkpoint = torch.load(
    run_dir / "last.pth.tar",
    map_location="cpu",
    weights_only=False,
)
if not isinstance(checkpoint, dict):
    raise SystemExit("last checkpoint is not a mapping")
if checkpoint.get("epoch") != boundary_epoch:
    raise SystemExit(
        "last checkpoint epoch does not match metrics boundary: "
        f"{checkpoint.get('epoch')} != {boundary_epoch}"
    )
expected_checkpoint_fields = (
    "state_dict",
    "optimizer",
    "scaler",
    "validation_metrics",
    "model_metadata",
    "split_hashes",
)
for key in expected_checkpoint_fields:
    if key not in checkpoint:
        raise SystemExit(f"last checkpoint missing field: {key}")
if checkpoint.get("variant") != variant:
    raise SystemExit("last checkpoint variant mismatch")
if checkpoint.get("seed") != seed:
    raise SystemExit("last checkpoint seed mismatch")
if checkpoint.get("dataset") != "NUDT-SIRST":
    raise SystemExit("last checkpoint dataset mismatch")
if checkpoint.get("selection_source") != "internal_validation_only":
    raise SystemExit("last checkpoint selection source mismatch")
if checkpoint.get("official_test_accessed") is not False:
    raise SystemExit("last checkpoint official-test policy mismatch")

last_event = events[-1]
checkpoint_metrics = checkpoint["validation_metrics"]
for key, value in checkpoint_metrics.items():
    if key not in last_event:
        raise SystemExit(f"last metrics event missing checkpoint metric: {key}")
    observed = last_event[key]
    if isinstance(value, (int, float)) and isinstance(observed, (int, float)):
        if float(value) != float(observed):
            raise SystemExit(
                f"checkpoint/metrics numeric mismatch for {key}: "
                f"{value} != {observed}"
            )
    elif value != observed:
        raise SystemExit(f"checkpoint/metrics mismatch for {key}")

protocol = json.loads(
    (run_dir / "protocol.json").read_text(encoding="utf-8")
)
arguments = protocol.get("arguments", {})
if arguments.get("variant") != variant or arguments.get("seed") != seed:
    raise SystemExit("protocol variant/seed mismatch")
if arguments.get("epochs") != 800:
    raise SystemExit("protocol target epochs mismatch")
if arguments.get("run_tag") != "screen800_pd_fp32_shared4x5090_v1":
    raise SystemExit("protocol run tag mismatch")
if protocol.get("official_test_accessed") is not False:
    raise SystemExit("protocol official-test policy mismatch")

split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
if split.get("hashes") != checkpoint["split_hashes"]:
    raise SystemExit("split hashes do not match last checkpoint")
if split.get("official_test_accessed") is not False:
    raise SystemExit("split official-test policy mismatch")

original_launch = json.loads(
    original_launch_path.read_text(encoding="utf-8")
)
if original_launch.get("schema") != "sctransnet_tpd_clean_v3_screen800_launch_v1":
    raise SystemExit("original launch schema mismatch")
if (
    original_launch.get("variant") != variant
    or original_launch.get("seed") != seed
):
    raise SystemExit("original launch variant/seed mismatch")
if pathlib.Path(original_launch.get("run_directory", "")).resolve() != run_dir.resolve():
    raise SystemExit("original launch run directory mismatch")

print(boundary_epoch)
PY
}

v3_create_boundary_snapshot() {
    "$v3_python" - \
        "$v3_run_dir" \
        "$v3_resume_boundary_root" \
        "$v3_variant" \
        "$v3_seed" \
        "$v3_boundary_epoch" \
        "$v3_original_launch" \
        "$v3_original_log" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import sys

run_dir = pathlib.Path(sys.argv[1])
boundary_root = pathlib.Path(sys.argv[2])
variant = sys.argv[3]
seed = int(sys.argv[4])
epoch = int(sys.argv[5])
original_launch = pathlib.Path(sys.argv[6])
original_log = pathlib.Path(sys.argv[7])

boundary_root.mkdir(parents=True, exist_ok=True)
target = boundary_root / f"{variant}_seed{seed}_epoch{epoch:03d}"
temporary = boundary_root / f".{target.name}.tmp.{os.getpid()}"
if target.exists() or target.is_symlink():
    raise SystemExit(f"resume boundary already exists: {target}")
if temporary.exists() or temporary.is_symlink():
    raise SystemExit(f"resume boundary temporary path exists: {temporary}")

sources = {
    "metrics.jsonl": run_dir / "metrics.jsonl",
    "last.pth.tar": run_dir / "last.pth.tar",
    "best.pth.tar": run_dir / "best.pth.tar",
    "best_miou.pth.tar": run_dir / "best_miou.pth.tar",
    "protocol.json": run_dir / "protocol.json",
    "split.json": run_dir / "split.json",
    "original_launch_manifest.json": original_launch,
    "original_worker.log": original_log,
}

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

temporary.mkdir(mode=0o700)
artifacts = {}
try:
    for snapshot_name, source in sources.items():
        destination = temporary / snapshot_name
        shutil.copy2(source, destination)
        source_sha = sha256(source)
        snapshot_sha = sha256(destination)
        if source_sha != snapshot_sha:
            raise SystemExit(f"boundary copy digest mismatch: {source}")
        artifacts[snapshot_name] = {
            "source": str(source.resolve()),
            "source_sha256": source_sha,
            "snapshot_sha256": snapshot_sha,
            "size_bytes": destination.stat().st_size,
        }
    payload = {
        "schema": "sctransnet_tpd_clean_v3_resume_boundary_v1",
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "variant": variant,
        "seed": seed,
        "boundary_epoch": epoch,
        "run_directory": str(run_dir.resolve()),
        "immutable_no_overwrite": True,
        "artifacts": artifacts,
    }
    boundary_manifest = temporary / "boundary.json"
    boundary_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in temporary.iterdir():
        path.chmod(0o444)
    temporary.rename(target)
    target.chmod(0o555)
except BaseException:
    if temporary.exists():
        shutil.rmtree(temporary)
    raise
print(target.resolve())
PY
}

v3_write_resume_manifest() {
    local v3_memory_used
    local v3_memory_free
    local v3_utilization
    local v3_load_average
    v3_memory_used="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=memory.used --format=csv,noheader,nounits
    )"
    v3_memory_free="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=memory.free --format=csv,noheader,nounits
    )"
    v3_utilization="$(
        nvidia-smi -i "$v3_gpu_uuid" \
            --query-gpu=utilization.gpu --format=csv,noheader,nounits
    )"
    v3_load_average="$(awk '{print $1","$2","$3}' /proc/loadavg)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$v3_gpu_uuid" \
        "$v3_python" - \
        "$v3_resume_manifest" \
        "$v3_variant" \
        "$v3_seed" \
        "$v3_gpu_uuid" \
        "$v3_gpu_index" \
        "$v3_run_dir" \
        "$v3_boundary_epoch" \
        "$v3_boundary_dir" \
        "$v3_original_launch" \
        "$v3_source_lock" \
        "$v3_original_source_lock" \
        "$v3_training_data_sha256" \
        "$v3_memory_used" \
        "$v3_memory_free" \
        "$v3_utilization" \
        "$v3_load_average" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import platform
import sys

import torch

(
    output_text,
    variant,
    seed_text,
    gpu_uuid,
    gpu_index_text,
    run_dir_text,
    boundary_epoch_text,
    boundary_dir_text,
    original_launch_text,
    source_lock_text,
    original_source_lock_text,
    data_sha256,
    memory_used,
    memory_free,
    utilization,
    load_average,
) = sys.argv[1:]
output = pathlib.Path(output_text)
boundary_dir = pathlib.Path(boundary_dir_text)
boundary_manifest = boundary_dir / "boundary.json"
original_launch_path = pathlib.Path(original_launch_text)
source_lock = pathlib.Path(source_lock_text)
original_source_lock = pathlib.Path(original_source_lock_text)

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

original_launch = json.loads(
    original_launch_path.read_text(encoding="utf-8")
)
payload = {
    "schema": "sctransnet_tpd_clean_v3_resume_2x5090_launch_v1",
    "created_at_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "variant": variant,
    "seed": int(seed_text),
    "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
    "run_directory": str(pathlib.Path(run_dir_text).resolve()),
    "run_tag": "screen800_pd_fp32_shared4x5090_v1",
    "boundary_epoch": int(boundary_epoch_text),
    "target_epoch": 800,
    "boundary_directory": str(boundary_dir.resolve()),
    "boundary_manifest_sha256": sha256(boundary_manifest),
    "original_launch_manifest": str(original_launch_path.resolve()),
    "original_launch_manifest_sha256": sha256(original_launch_path),
    "original_gpu_uuid": original_launch["gpu_uuid"],
    "resume_gpu_uuid": gpu_uuid,
    "resume_gpu_index": int(gpu_index_text),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "training_data_sha256": data_sha256,
    "source_lock": str(source_lock.resolve()),
    "source_lock_sha256": sha256(source_lock),
    "original_source_lock": str(original_source_lock.resolve()),
    "original_source_lock_sha256": sha256(original_source_lock),
    "resource_snapshot": {
        "gpu_memory_used_mib": int(memory_used),
        "gpu_memory_free_mib": int(memory_free),
        "gpu_utilization_percent": int(utilization),
        "load_average": [float(value) for value in load_average.split(",")],
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        "NUMEXPR_NUM_THREADS": os.environ["NUMEXPR_NUM_THREADS"],
    },
    "policy": {
        "in_place_resume": True,
        "fresh_run": False,
        "original_results_preserved_by_boundary": True,
        "immutable_resume_boundary": True,
        "paired_variants": True,
        "pre_registered_seeds": [42, 3407],
        "allowed_gpu_indices": [2, 3],
        "concurrent_jobs_per_gpu": 2,
        "counterbalanced_mapping": True,
        "efficiency_comparison_allowed": False,
        "official_test_accessed": False,
        "amp": False,
        "cpu_replay_thread_cap": 1,
    },
}
encoded = (
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
).encode("utf-8")
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
if output.exists() or output.is_symlink():
    raise SystemExit(f"resume manifest already exists: {output}")
with temporary.open("xb") as stream:
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
try:
    os.link(temporary, output)
finally:
    temporary.unlink(missing_ok=True)
output.chmod(0o444)
PY
}

v3_verify_source_locks
v3_source_lock_sha256="$(v3_sha256 "$v3_source_lock")"
v3_verify_data
v3_verify_gpu

if [[ ! -f "$v3_resume_program" || -L "$v3_resume_program" ]]; then
    echo "TPDCLEANV3_RESUME_2X_ABORT reason=missing_resume_program path=$v3_resume_program" >&2
    exit 1
fi

v3_boundary_epoch="$(v3_validate_boundary)"
v3_boundary_dir="$(
    v3_create_boundary_snapshot
)"
v3_write_resume_manifest

echo "TPDCLEANV3_RESUME_2X_START variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_gpu_uuid boundary_epoch=$v3_boundary_epoch target_epoch=800 cpu_threads=$v3_cpu_threads run_dir=$v3_run_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$v3_gpu_uuid"
export PYTHONUNBUFFERED=1

"$v3_python" experiments/resume_tpd_clean_v3.py \
    --run-dir "$v3_run_dir" \
    --device cuda:0 \
    --target-epoch 800 \
    --expected-resume-epoch "$v3_boundary_epoch" \
    --resume-gpu-uuid "$v3_gpu_uuid"

[[ "$(wc -l < "$v3_run_dir/metrics.jsonl")" -eq 800 ]]
jq -e --arg v3_variant "$v3_variant" --argjson v3_seed "$v3_seed" '
    .status == "complete" and
    .variant == $v3_variant and
    .dataset == "NUDT-SIRST" and
    .seed == $v3_seed and
    .selection_source == "internal_validation_only" and
    .official_test_accessed == false
' "$v3_run_dir/summary.json" >/dev/null

for v3_checkpoint in best.pth.tar best_miou.pth.tar; do
    "$v3_python" experiments/evaluate_tpd_clean_v3_pd_fa.py \
        --run-dir "$v3_run_dir" \
        --checkpoint "$v3_checkpoint" \
        --device cuda:0 \
        --expected-epochs 800
done
for v3_sweep in \
    "$v3_run_dir/pd_fa_sweep_best.pth.json" \
    "$v3_run_dir/pd_fa_sweep_best_miou.pth.json"; do
    if [[ ! -f "$v3_sweep" || -L "$v3_sweep" ]]; then
        echo "TPDCLEANV3_RESUME_2X_ABORT reason=missing_sweep path=$v3_sweep" >&2
        exit 1
    fi
done

v3_verify_source_locks
v3_verify_data
if [[ "$(v3_sha256 "$v3_source_lock")" != "$v3_source_lock_sha256" ]]; then
    echo "TPDCLEANV3_RESUME_2X_ABORT reason=source_lock_changed_during_run" >&2
    exit 1
fi
echo "TPDCLEANV3_RESUME_2X_COMPLETE variant=$v3_variant seed=$v3_seed gpu_uuid=$v3_gpu_uuid boundary_epoch=$v3_boundary_epoch epochs=800"
