#!/usr/bin/env bash
set -euo pipefail

v8_mode="run"
case "${1:-}" in
    --preflight)
        v8_mode="preflight"
        shift
        ;;
    --validate-only)
        v8_mode="validate-only"
        shift
        ;;
esac
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight|--validate-only]" >&2
    exit 2
fi

v8_repo="${TPD_V8_MPRS_DCH_REPO:-/home/ly/SCTransNet_main}"
v8_validation_python="${TPD_V8_MPRS_DCH_VALIDATION_PYTHON:-/usr/bin/python3}"
v8_lane_runner="$v8_repo/experiments/run_tpd_clean_v8_mprs_dch_formal800_2x5090_lane.sh"
v8_authorization="${TPD_V8_MPRS_DCH_FORMAL800_AUTHORIZATION:-$v8_repo/experiments/tpd_clean_v8_mprs_dch_formal800_authorization.json}"
v8_source_lock="${TPD_V8_MPRS_DCH_SOURCE_LOCK:-$v8_repo/experiments/tpd_clean_v8_mprs_dch_exact_source_lock.json}"
v8_acceptance_lock="${TPD_V8_MPRS_DCH_ACCEPTANCE_SOURCE_LOCK:-$v8_repo/experiments/tpd_clean_v8_mprs_dch_acceptance_source_lock.json}"
v8_protocol="${TPD_V8_MPRS_DCH_PROTOCOL:-$v8_repo/experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md}"
v8_counterfactual="${TPD_V8_MPRS_DCH_COUNTERFACTUAL_REPORT:-$v8_repo/analysis/results/tpd_clean_v8_mprs_counterfactual_v2/tpd_clean_v8_mprs_counterfactual.json}"
v8_benchmark="${TPD_V8_MPRS_DCH_COMPUTE_BENCHMARK:-$v8_repo/analysis/results/tpd_clean_v8_mprs_benchmark_v3/gpu2.json}"
v8_cpu_smoke="${TPD_V8_MPRS_DCH_CPU_SMOKE:-$v8_repo/analysis/results/tpd_clean_v8_mprs_smoke_v3/cpu.json}"
v8_gpu2_smoke="${TPD_V8_MPRS_DCH_GPU2_SMOKE:-$v8_repo/analysis/results/tpd_clean_v8_mprs_smoke_v3/gpu2.json}"
v8_gpu3_smoke="${TPD_V8_MPRS_DCH_GPU3_SMOKE:-$v8_repo/analysis/results/tpd_clean_v8_mprs_smoke_v3/gpu3.json}"
v8_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
v8_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
v8_gpu2_unit="sctransnet-tpd-clean-v8-mprs-dch-gpu2-lane"
v8_gpu3_unit="sctransnet-tpd-clean-v8-mprs-dch-gpu3-lane"

cd "$v8_repo"
[[ -x "$v8_validation_python" ]] || {
    echo "TPDCLEANV8MPRSDCH_2X_LAUNCH_ABORT reason=validation_python_not_executable path=$v8_validation_python" >&2
    exit 1
}
[[ -f "$v8_authorization" && ! -L "$v8_authorization" ]] || {
    echo "TPDCLEANV8MPRSDCH_2X_LAUNCH_ABORT reason=missing_formal_authorization path=$v8_authorization" >&2
    exit 1
}

# Authorization is deliberately checked before the source lock and every lane
# action.  The current counterfactual gate is false, so no true manifest may be
# synthesized merely to make this launcher pass.
"$v8_validation_python" - \
    "$v8_repo" \
    "$v8_authorization" \
    "$v8_source_lock" \
    "$v8_acceptance_lock" \
    "$v8_protocol" \
    "$v8_counterfactual" \
    "$v8_benchmark" \
    "$v8_cpu_smoke" \
    "$v8_gpu2_smoke" \
    "$v8_gpu3_smoke" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
from typing import Any


AUTHORIZATION_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_formal800_authorization_v1"
)
SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_exact_source_lock_v1"
)
ACCEPTANCE_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_acceptance_source_lock_v1"
)
COUNTERFACTUAL_SCHEMA = "sctransnet_tpd_clean_v8_mprs_counterfactual_v2"
BENCHMARK_SCHEMA = "sctransnet_tpd_clean_v8_mprs_block_benchmark_v2"
SMOKE_SCHEMA = "sctransnet_tpd_clean_v8_mprs_dch_smoke_v1"
GPU_UUIDS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
VARIANTS = [
    "tpd_clean_v8_mprs_dch_full",
    "tpd_clean_v8_mprs_dch_capacity",
]
REQUIRED_ACCEPTANCE_SOURCES = {
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
    "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "analysis/analyze_tpd_clean_v8_mprs_mechanism.py",
    "analysis/benchmark_tpd_clean_v8_mprs_dch.py",
    "experiments/smoke_tpd_clean_v8_mprs_dch.py",
    "experiments/launch_tpd_clean_v8_mprs_dch_formal800_2x5090.sh",
    "experiments/run_tpd_clean_v8_mprs_dch_formal800_2x5090_lane.sh",
    "experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py",
}
BENCHMARK_REPORT_SOURCES = {
    "analysis/benchmark_tpd_clean_v8_mprs_dch.py",
    "model/tpd_clean_v7_dch.py",
    "model/tpd_clean_v8_mprs_dch.py",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
}
SMOKE_REPORT_SOURCES = {
    "experiments/smoke_tpd_clean_v8_mprs_dch.py",
    "experiments/train_tpd_clean_v8_mprs_dch.py",
    "model/tpd_clean_v8_mprs_dch.py",
    "model/SCTransNet.py",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
}
FORMAL_CONTRACT = {
    "epochs": 800,
    "eval_every": 1,
    "workers": 0,
    "amp": False,
    "eps": 1e-6,
    "cublas_workspace_config": ":4096:8",
    "initialization_modes": ["fresh", "exact_resume"],
}
REQUIRED_GATES = {
    "formal_model_tests_passed",
    "v7_checkpoint_strict_load_passed",
    "zero_scale_spd_equivalence_passed",
    "paired_initialization_first_step_passed",
    "optimized_forward_contract_passed",
    "v8_exact_resume_passed",
    "counterfactual_finite_passed",
    "target_correction_lift_passed",
    "fragmentation_gate_passed",
    "shift_consistency_gate_passed",
    "compute_memory_gate_passed",
    "cpu_gpu_smoke_passed",
    "source_locks_passed",
}


def abort(reason: str) -> None:
    raise SystemExit(
        f"TPDCLEANV8MPRSDCH_2X_LAUNCH_ABORT reason={reason}"
    )


def regular(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_file() or path.is_symlink():
        abort(f"missing_or_nonregular_{label} path={path}")
    return path


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        abort(f"invalid_{label}_json error={exc}")
    if not isinstance(value, dict):
        abort(f"{label}_must_be_object")
    return value


def validate_report_sources(
    report: dict[str, Any],
    required: set[str],
    label: str,
) -> None:
    sources = report.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != required:
        abort(f"{label}_source_set_mismatch")
    for relative, expected in sources.items():
        path = regular(repo / relative, f"{label}_source")
        if sha256(path) != expected:
            abort(f"{label}_source_digest_mismatch path={relative}")


repo = pathlib.Path(sys.argv[1]).resolve()
authorization_path = pathlib.Path(sys.argv[2])
source_lock_path = pathlib.Path(sys.argv[3])
acceptance_lock_path = pathlib.Path(sys.argv[4])
protocol_path = pathlib.Path(sys.argv[5])
counterfactual_path = pathlib.Path(sys.argv[6])
benchmark_path = pathlib.Path(sys.argv[7])
cpu_smoke_path = pathlib.Path(sys.argv[8])
gpu2_smoke_path = pathlib.Path(sys.argv[9])
gpu3_smoke_path = pathlib.Path(sys.argv[10])

authorization = load_json(
    authorization_path,
    "formal_authorization",
)
if authorization.get("schema") != AUTHORIZATION_SCHEMA:
    abort("formal_authorization_schema_mismatch")
if authorization.get("formal_training_authorized") is not True:
    abort("formal_training_not_authorized")

gates = authorization.get("preflight_gates")
if not isinstance(gates, dict):
    abort("preflight_gates_missing")
missing_gates = sorted(REQUIRED_GATES - set(gates))
if missing_gates:
    abort(f"preflight_gates_missing_names names={missing_gates}")
false_gates = sorted(name for name, value in gates.items() if value is not True)
if false_gates:
    abort(f"preflight_gate_false names={false_gates}")

expected_runs = {
    ("tpd_clean_v8_mprs_dch_full", 42, 800),
    ("tpd_clean_v8_mprs_dch_capacity", 42, 800),
    ("tpd_clean_v8_mprs_dch_full", 3407, 800),
    ("tpd_clean_v8_mprs_dch_capacity", 3407, 800),
}
authorized_runs = authorization.get("authorized_runs")
if not isinstance(authorized_runs, list):
    abort("authorized_runs_missing")
try:
    actual_runs = {
        (item["variant"], int(item["seed"]), int(item["epochs"]))
        for item in authorized_runs
        if isinstance(item, dict)
    }
except (KeyError, TypeError, ValueError):
    abort("authorized_runs_invalid")
if len(actual_runs) != len(authorized_runs) or actual_runs != expected_runs:
    abort("authorized_runs_mismatch")
if authorization.get("physical_gpu_assignments") != GPU_UUIDS:
    abort("physical_gpu_assignments_mismatch")

artifacts = {
    "training_source_lock_sha256": regular(
        source_lock_path,
        "training_source_lock",
    ),
    "acceptance_source_lock_sha256": regular(
        acceptance_lock_path,
        "acceptance_source_lock",
    ),
    "protocol_sha256": regular(protocol_path, "protocol"),
    "counterfactual_report_sha256": regular(
        counterfactual_path,
        "counterfactual_report",
    ),
    "compute_benchmark_sha256": regular(
        benchmark_path,
        "compute_benchmark",
    ),
    "cpu_smoke_sha256": regular(cpu_smoke_path, "cpu_smoke"),
    "gpu2_smoke_sha256": regular(gpu2_smoke_path, "gpu2_smoke"),
    "gpu3_smoke_sha256": regular(gpu3_smoke_path, "gpu3_smoke"),
}
for field, path in artifacts.items():
    actual = sha256(path)
    if authorization.get(field) != actual:
        abort(f"{field}_mismatch")

source_lock = load_json(source_lock_path, "training_source_lock")
if source_lock.get("schema") != SOURCE_LOCK_SCHEMA:
    abort("training_source_lock_schema_mismatch")
if source_lock.get("variants") != VARIANTS:
    abort("training_source_lock_variant_matrix_mismatch")
if source_lock.get("formal_contract") != FORMAL_CONTRACT:
    abort("training_source_lock_formal_contract_mismatch")
training_data_sha256 = source_lock.get("training_data_sha256")
if (
    not isinstance(training_data_sha256, str)
    or len(training_data_sha256) != 64
    or any(character not in "0123456789abcdef" for character in training_data_sha256)
):
    abort("training_source_lock_data_digest_invalid")
source_sha256 = source_lock.get("source_sha256")
if (
    not isinstance(source_sha256, dict)
    or not source_sha256
    or source_lock.get("source_count") != len(source_sha256)
):
    abort("training_source_lock_sources_missing")
for relative, expected_digest in source_sha256.items():
    if not isinstance(relative, str) or not relative:
        abort("training_source_lock_path_invalid")
    candidate = repo / relative
    regular(candidate, "locked_source")
    path = candidate.resolve()
    try:
        canonical = str(path.relative_to(repo))
    except ValueError:
        abort(f"training_source_lock_path_escape path={relative}")
    if canonical != relative:
        abort(f"training_source_lock_path_noncanonical path={relative}")
    if sha256(path) != expected_digest:
        abort(f"training_source_digest_mismatch path={relative}")

acceptance_lock = load_json(
    acceptance_lock_path,
    "acceptance_source_lock",
)
if (
    acceptance_lock.get("schema") != ACCEPTANCE_SOURCE_LOCK_SCHEMA
    or acceptance_lock.get("lock_kind") != "acceptance"
    or acceptance_lock.get("variants") != VARIANTS
):
    abort("acceptance_source_lock_identity_mismatch")
if (
    acceptance_lock.get("training_source_lock_sha256")
    != sha256(source_lock_path)
):
    abort("acceptance_source_lock_training_binding_mismatch")
acceptance_sources = acceptance_lock.get("source_sha256")
if (
    not isinstance(acceptance_sources, dict)
    or set(acceptance_sources) != REQUIRED_ACCEPTANCE_SOURCES
    or acceptance_lock.get("source_count") != len(acceptance_sources)
):
    abort("acceptance_source_lock_sources_missing")
for relative, expected_digest in acceptance_sources.items():
    if not isinstance(relative, str) or not relative:
        abort("acceptance_source_lock_path_invalid")
    candidate = repo / relative
    regular(candidate, "locked_acceptance_source")
    path = candidate.resolve()
    try:
        canonical = str(path.relative_to(repo))
    except ValueError:
        abort(f"acceptance_source_lock_path_escape path={relative}")
    if canonical != relative:
        abort(f"acceptance_source_lock_path_noncanonical path={relative}")
    if sha256(path) != expected_digest:
        abort(f"acceptance_source_digest_mismatch path={relative}")

counterfactual = load_json(
    counterfactual_path,
    "counterfactual_report",
)
if (
    counterfactual.get("schema") != COUNTERFACTUAL_SCHEMA
    or counterfactual.get("status") != "complete"
    or counterfactual.get("counterfactual_gate_pass") is not True
    or counterfactual.get("training_performed") is not False
    or counterfactual.get("job_count") != 12
    or counterfactual.get("strict_load_count") != 12
    or counterfactual.get("finite_job_count") != 12
):
    abort("counterfactual_report_gate_failed")
hardening = counterfactual.get("audit_hardening")
if (
    not isinstance(hardening, dict)
    or hardening.get("expected_job_count") != 12
    or hardening.get("all_job_bindings_revalidated") is not True
    or hardening.get(
        "all_raw_outputs_losses_correlations_blocks_finite"
    ) is not True
    or hardening.get(
        "all_target_hard_negative_and_block_coverage_nonempty"
    ) is not True
    or hardening.get("target_priority_pooled_masks_disjoint") is not True
    or hardening.get("paired_topology_recomputed_from_per_gt") is not True
    or hardening.get(
        "reference_coverage_nondecrease_gate_included"
    ) is not True
    or hardening.get(
        "all_gate_probabilities_from_production_forward"
    ) is not True
    or hardening.get("shift_interpretation")
    != "toroidal_grid_offset_stress"
    or hardening.get("output_separate_from_formal_results_root") is not True
    or hardening.get("v8_protocol_sha256") != sha256(protocol_path)
    or hardening.get("v8_preflight_amendment_sha256")
    != sha256(
        repo
        / "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md"
    )
):
    abort("counterfactual_hardening_evidence_failed")
expected_groups = {
    "tpd_clean_v8_mprs_dch_full/seed_42",
    "tpd_clean_v8_mprs_dch_full/seed_3407",
    "tpd_clean_v8_mprs_dch_capacity/seed_42",
    "tpd_clean_v8_mprs_dch_capacity/seed_3407",
}
groups = counterfactual.get("groups")
if not isinstance(groups, dict) or set(groups) != expected_groups:
    abort("counterfactual_group_matrix_mismatch")
for group_name, group in groups.items():
    if (
        not isinstance(group, dict)
        or group.get("group_pass") is not True
        or group.get("target_correction_lift_pass") is not True
        or group.get("fragment_excess_nonincrease_pass") is not True
        or group.get("largest_fragment_nondecrease_pass") is not True
        or group.get("reference_coverage_nondecrease_pass") is not True
        or group.get("shift_ratio_pass") is not True
        or group.get("finite_pass") is not True
        or group.get("shift_stress_definition")
        != "toroidal_grid_offset_stress"
        or not isinstance(group.get("blocks"), dict)
        or len(group["blocks"]) != 7
        or not isinstance(group.get("coverage"), dict)
        or group["coverage"].get("checkpoint_role_count") != 3
        or group["coverage"].get("validation_image_evaluations") != 399
        or group["coverage"].get("block_count") != 7
        or group["coverage"].get("complete") is not True
    ):
        abort(f"counterfactual_group_gate_failed group={group_name}")
job_outputs = counterfactual.get("job_outputs")
if not isinstance(job_outputs, list) or len(job_outputs) != 12:
    abort("counterfactual_job_outputs_missing")
counterfactual_root = counterfactual_path.resolve().parent
for job in job_outputs:
    if not isinstance(job, dict):
        abort("counterfactual_job_output_invalid")
    job_path = pathlib.Path(str(job.get("path", ""))).resolve()
    try:
        job_path.relative_to(counterfactual_root)
    except ValueError:
        abort(f"counterfactual_job_output_escape path={job_path}")
    regular(job_path, "counterfactual_job_output")
    if sha256(job_path) != job.get("sha256"):
        abort(f"counterfactual_job_output_digest_mismatch path={job_path}")

benchmark = load_json(benchmark_path, "compute_benchmark")
if (
    benchmark.get("schema") != BENCHMARK_SCHEMA
    or benchmark.get("status") != "complete"
    or benchmark.get("compute_memory_gate_pass") is not True
    or benchmark.get("training_performed") is not False
):
    abort("compute_benchmark_gate_failed")
validate_report_sources(
    benchmark,
    BENCHMARK_REPORT_SOURCES,
    "compute_benchmark",
)
benchmark_variants = benchmark.get("variants")
if (
    not isinstance(benchmark_variants, list)
    or len(benchmark_variants) != 2
    or {
        item.get("variant")
        for item in benchmark_variants
        if isinstance(item, dict)
    } != set(VARIANTS)
):
    abort("compute_benchmark_variant_matrix_mismatch")
for variant_report in benchmark_variants:
    if (
        not isinstance(variant_report, dict)
        or variant_report.get("variant_compute_memory_gate_pass") is not True
        or not isinstance(variant_report.get("shapes"), list)
        or len(variant_report["shapes"]) != 2
    ):
        abort("compute_benchmark_variant_gate_failed")
    for shape_report in variant_report["shapes"]:
        if not isinstance(shape_report, dict):
            abort("compute_benchmark_shape_invalid")
        ratio_v7 = shape_report.get("v8_optimized_v7_peak_ratio")
        ratio_direct = shape_report.get(
            "v8_optimized_direct_peak_ratio"
        )
        if (
            not isinstance(ratio_v7, (int, float))
            or not math.isfinite(ratio_v7)
            or ratio_v7 > 1.10
            or not isinstance(ratio_direct, (int, float))
            or not math.isfinite(ratio_direct)
            or ratio_direct >= 1.0
            or shape_report.get("peak_memory_increase_pass") is not True
            or shape_report.get("optimized_below_direct_pass") is not True
            or shape_report.get("output_equivalence_pass") is not True
        ):
            abort("compute_benchmark_shape_gate_failed")

for label, path, expected_index in (
    ("cpu_smoke", cpu_smoke_path, None),
    ("gpu2_smoke", gpu2_smoke_path, 2),
    ("gpu3_smoke", gpu3_smoke_path, 3),
):
    smoke = load_json(path, label)
    if (
        smoke.get("schema") != SMOKE_SCHEMA
        or smoke.get("status") != "complete"
        or smoke.get("paired_initialization") is not True
        or smoke.get("paired_first_adam_step_exact") is not True
        or smoke.get("loss_head_count") != 6
    ):
        abort(f"{label}_gate_failed")
    validate_report_sources(smoke, SMOKE_REPORT_SOURCES, label)
    smoke_variants = smoke.get("variants")
    if (
        not isinstance(smoke_variants, list)
        or len(smoke_variants) != 2
        or {
            item.get("variant")
            for item in smoke_variants
            if isinstance(item, dict)
        } != set(VARIANTS)
    ):
        abort(f"{label}_variant_matrix_mismatch")
    for item in smoke_variants:
        if (
            not isinstance(item, dict)
            or item.get("status") != "complete"
            or item.get("output_count") != 6
            or item.get("optimizer_steps_completed") != 2
            or item.get("step_zero_exact_spd") is not True
            or item.get("strict_rebuild_load") is not True
            or item.get("all_observed_gradients_finite") is not True
            or item.get("all_updated_parameters_finite") is not True
            or item.get("mprs_block_count") != 7
            or item.get("mprs_diagnostics_verified") is not True
            or item.get("standard_forward_conv2d_calls_per_block") != 3
        ):
            abort(f"{label}_variant_evidence_failed")
    if expected_index is None:
        if smoke.get("device") != "cpu":
            abort("cpu_smoke_device_mismatch")
    elif (
        smoke.get("device") != "cuda:0"
        or str(smoke.get("physical_gpu_index")) != str(expected_index)
        or smoke.get("physical_gpu_uuid") != GPU_UUIDS[str(expected_index)]
        or not isinstance(smoke.get("deterministic_execution"), dict)
        or smoke["deterministic_execution"].get("enabled") is not True
        or smoke["deterministic_execution"].get(
            "cublas_workspace_config"
        ) != ":4096:8"
    ):
        abort(f"{label}_device_mismatch")

print(
    "TPDCLEANV8MPRSDCH_2X_AUTHORIZATION_OK"
    f" authorization_sha256={sha256(authorization_path)}"
    f" source_lock_sha256={sha256(source_lock_path)}"
    " authorized_runs=4 physical_gpus=2,3",
    flush=True,
)
PY

if [[ "$v8_mode" == "validate-only" ]]; then
    exit 0
fi

[[ -x "$v8_lane_runner" ]] || {
    echo "TPDCLEANV8MPRSDCH_2X_LAUNCH_ABORT reason=lane_runner_not_executable path=$v8_lane_runner" >&2
    exit 1
}

# Both lane preflights are synchronous and never start training.
"$v8_lane_runner" --preflight gpu2 "$v8_gpu2_uuid"
"$v8_lane_runner" --preflight gpu3 "$v8_gpu3_uuid"
echo "TPDCLEANV8MPRSDCH_2X_PREFLIGHT_ALL_OK lanes=2 tasks=4 epochs=800 physical_gpus=2,3 concurrent_tasks_per_gpu=1"
if [[ "$v8_mode" == "preflight" ]]; then
    exit 0
fi

for v8_unit in "$v8_gpu2_unit.service" "$v8_gpu3_unit.service"; do
    if systemctl --user is-active --quiet "$v8_unit"; then
        echo "TPDCLEANV8MPRSDCH_2X_LAUNCH_ABORT reason=lane_already_active unit=$v8_unit" >&2
        exit 1
    fi
done

systemd-run --user \
    --collect \
    --unit="$v8_gpu2_unit" \
    --description="SCTransNet V8-MPRS-DCH physical GPU2 serial lane" \
    --property=Restart=on-failure \
    --property=RestartSec=30 \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v8_lane_runner" gpu2 "$v8_gpu2_uuid"
echo "TPDCLEANV8MPRSDCH_2X_LANE_STARTED lane=gpu2 unit=$v8_gpu2_unit.service physical_gpu=2 gpu_uuid=$v8_gpu2_uuid"

systemd-run --user \
    --collect \
    --unit="$v8_gpu3_unit" \
    --description="SCTransNet V8-MPRS-DCH physical GPU3 serial lane" \
    --property=Restart=on-failure \
    --property=RestartSec=30 \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v8_lane_runner" gpu3 "$v8_gpu3_uuid"
echo "TPDCLEANV8MPRSDCH_2X_LANE_STARTED lane=gpu3 unit=$v8_gpu3_unit.service physical_gpu=3 gpu_uuid=$v8_gpu3_uuid"
