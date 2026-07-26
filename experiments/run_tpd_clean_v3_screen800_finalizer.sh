#!/usr/bin/env bash
set -euo pipefail

# This controller is deliberately outside the frozen training source lock.  It
# only observes the four pre-registered workers and writes under the isolated
# Clean-v3 result root.
v3_repo="${TPDCLEANV3_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v3_python="${TPDCLEANV3_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v3_result_root="${TPDCLEANV3_FINALIZER_RESULT_ROOT:-$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1}"
v3_formal_root="${TPDCLEANV3_FINALIZER_FORMAL_ROOT:-$v3_repo/experiments/results/tpd_pe_formal800_4x5090_v1}"
v3_v2_root="${TPDCLEANV3_FINALIZER_V2_ROOT:-$v3_repo/experiments/results/tpd_clean_screen800_4x5090_v1}"
v3_reference_miou_root="${TPDCLEANV3_FINALIZER_REFERENCE_MIOU_ROOT:-$v3_v2_root/frozen_reference_miou_runs}"
v3_summarizer="${TPDCLEANV3_FINALIZER_SUMMARIZER:-$v3_repo/experiments/summarize_tpd_clean_v3_screen800.py}"
v3_completion_validator="${TPDCLEANV3_FINALIZER_COMPLETION_VALIDATOR:-$v3_repo/experiments/validate_tpd_clean_v3_completion.py}"
v3_postprocess_lock="${TPDCLEANV3_FINALIZER_POSTPROCESS_LOCK:-$v3_repo/experiments/tpd_clean_v3_postprocess_source_lock.json}"
v3_systemctl="${TPDCLEANV3_FINALIZER_SYSTEMCTL:-systemctl}"
v3_sleep="${TPDCLEANV3_FINALIZER_SLEEP:-sleep}"
v3_poll_seconds="${TPDCLEANV3_FINALIZER_POLL_SECONDS:-60}"

v3_run_tag="screen800_pd_fp32_shared4x5090_v1"
v3_comparison_dir="$v3_result_root/NUDT-SIRST/comparison"
v3_launch_root="$v3_result_root/launch"
v3_log_root="$v3_result_root/logs"
v3_lock_root="$v3_result_root/.locks"
v3_state_file="$v3_launch_root/finalizer_state.json"
v3_marker="$v3_comparison_dir/COMPLETE.sha256"
v3_finalizer_log="$v3_log_root/finalizer.log"
v3_lock_file="$v3_lock_root/finalizer.lock"

v3_variants=(
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
)
v3_seeds=(42 42 3407 3407)
v3_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)
v3_units=(
    sctransnet-tpd-clean-v3-full-s42.service
    sctransnet-tpd-clean-v3-cap-s42.service
    sctransnet-tpd-clean-v3-full-s3407.service
    sctransnet-tpd-clean-v3-cap-s3407.service
)

if [[ ! "$v3_poll_seconds" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "TPDCLEANV3_FINALIZER_ABORT reason=invalid_poll_seconds value=$v3_poll_seconds" >&2
    exit 2
fi

mkdir -p "$v3_comparison_dir" "$v3_launch_root" "$v3_log_root" "$v3_lock_root"
exec > >(tee -a "$v3_finalizer_log") 2>&1
cd "$v3_repo"

[[ -x "$v3_python" ]] || {
    echo "TPDCLEANV3_FINALIZER_ABORT reason=python_not_executable path=$v3_python" >&2
    exit 1
}
[[ -f "$v3_summarizer" && ! -L "$v3_summarizer" ]] || {
    echo "TPDCLEANV3_FINALIZER_ABORT reason=missing_summarizer path=$v3_summarizer" >&2
    exit 1
}
[[ -f "$v3_completion_validator" && ! -L "$v3_completion_validator" ]] || {
    echo "TPDCLEANV3_FINALIZER_ABORT reason=missing_completion_validator path=$v3_completion_validator" >&2
    exit 1
}
[[ -f "$v3_postprocess_lock" && ! -L "$v3_postprocess_lock" ]] || {
    echo "TPDCLEANV3_FINALIZER_ABORT reason=missing_postprocess_lock path=$v3_postprocess_lock" >&2
    exit 1
}
command -v "$v3_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_FINALIZER_ABORT reason=missing_systemctl command=$v3_systemctl" >&2
    exit 1
}
command -v "$v3_sleep" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_FINALIZER_ABORT reason=missing_sleep command=$v3_sleep" >&2
    exit 1
}

v3_verify_postprocess_sources() {
    "$v3_python" - \
        "$v3_repo" \
        "$v3_postprocess_lock" \
        "$v3_summarizer" \
        "$v3_completion_validator" \
        "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
lock_path = pathlib.Path(sys.argv[2]).resolve(strict=True)
summarizer_path = pathlib.Path(sys.argv[3]).resolve(strict=True)
validator_path = pathlib.Path(sys.argv[4]).resolve(strict=True)
runner_path = pathlib.Path(sys.argv[5]).resolve(strict=True)
canonical = {
    "postprocess lock": repo
    / "experiments/tpd_clean_v3_postprocess_source_lock.json",
    "summarizer": repo
    / "experiments/summarize_tpd_clean_v3_screen800.py",
    "completion validator": repo
    / "experiments/validate_tpd_clean_v3_completion.py",
    "runner": repo
    / "experiments/run_tpd_clean_v3_screen800_finalizer.sh",
}
observed = {
    "postprocess lock": lock_path,
    "summarizer": summarizer_path,
    "completion validator": validator_path,
    "runner": runner_path,
}
for label, expected in canonical.items():
    if observed[label] != expected.resolve(strict=True):
        raise SystemExit(
            f"non-canonical Clean-v3 {label}: "
            f"expected={expected} observed={observed[label]}"
        )
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v3_postprocess_source_lock_v1":
    raise SystemExit("unexpected Clean-v3 postprocess source-lock schema")
entries = payload.get("source_sha256")
if not isinstance(entries, dict) or not entries:
    raise SystemExit("Clean-v3 postprocess source lock has no entries")
required = {
    "experiments/summarize_tpd_clean_v3_screen800.py",
    "experiments/validate_tpd_clean_v3_completion.py",
    "experiments/run_tpd_clean_v3_screen800_finalizer.sh",
    "experiments/launch_tpd_clean_v3_screen800_finalizer.sh",
}
if not required.issubset(entries):
    raise SystemExit("Clean-v3 postprocess source lock misses runtime entries")
for relative, expected in entries.items():
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked postprocess source: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            f"postprocess source digest mismatch: {relative} "
            f"expected={expected} actual={actual}"
        )
print(
    "TPDCLEANV3_POSTPROCESS_SOURCES_OK"
    f" checked_entries={len(entries)}",
    flush=True,
)
PY
}

v3_verify_postprocess_sources
"$v3_python" - \
    "$v3_result_root" \
    "$v3_formal_root" \
    "$v3_v2_root" \
    "$v3_comparison_dir" \
    "$v3_state_file" \
    "$v3_marker" <<'PY'
import pathlib
import sys

(
    result_text,
    formal_text,
    v2_text,
    comparison_text,
    state_text,
    marker_text,
) = sys.argv[1:]
result = pathlib.Path(result_text).resolve()
formal = pathlib.Path(formal_text).resolve()
v2 = pathlib.Path(v2_text).resolve()
comparison = pathlib.Path(comparison_text).resolve()
state = pathlib.Path(state_text).resolve()
marker = pathlib.Path(marker_text).resolve()

def within(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path == root or root in path.parents

if result == formal or result == v2:
    raise SystemExit("candidate and reference roots must differ")
if within(result, formal) or within(formal, result):
    raise SystemExit("candidate/formal roots may not overlap")
if within(result, v2) or within(v2, result):
    raise SystemExit("candidate/Clean-v2 roots may not overlap")
for path, label in (
    (comparison, "comparison"),
    (state, "state"),
    (marker, "marker"),
):
    if not within(path, result):
        raise SystemExit(f"{label} path escaped the Clean-v3 result root")
PY

v3_write_state() {
    local v3_state="$1"
    local v3_message="$2"
    local v3_snapshot="${3:--}"
    "$v3_python" - \
        "$v3_state_file" \
        "$v3_state" \
        "$v3_message" \
        "$v3_snapshot" \
        "$v3_result_root" \
        "$v3_marker" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile

state_path = pathlib.Path(sys.argv[1])
state = sys.argv[2]
message = sys.argv[3]
snapshot_text = sys.argv[4]
result_root = pathlib.Path(sys.argv[5]).resolve()
marker = pathlib.Path(sys.argv[6]).resolve()
units = []
if snapshot_text != "-":
    snapshot = pathlib.Path(snapshot_text)
    if snapshot.is_file():
        for raw in snapshot.read_text(encoding="utf-8").splitlines():
            fields = raw.split("\t")
            if len(fields) != 8:
                raise SystemExit(f"invalid finalizer unit snapshot row: {raw!r}")
            (
                tag,
                unit,
                load_state,
                active_state,
                sub_state,
                result,
                exec_main_code,
                exec_main_status,
            ) = fields
            units.append(
                {
                    "tag": tag,
                    "unit": unit,
                    "LoadState": load_state,
                    "ActiveState": active_state,
                    "SubState": sub_state,
                    "Result": result,
                    "ExecMainCode": exec_main_code,
                    "ExecMainStatus": exec_main_status,
                }
            )
payload = {
    "schema": "sctransnet_tpd_clean_v3_screen800_finalizer_state_v1",
    "updated_at_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "state": state,
    "message": message,
    "candidate_root": str(result_root),
    "completion_marker": str(marker),
    "units": units,
}
state_path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{state_path.name}.", suffix=".tmp", dir=state_path.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, state_path)
finally:
    temporary_path = pathlib.Path(temporary)
    if temporary_path.exists():
        temporary_path.unlink()
PY
}

v3_verify_complete_marker() {
    "$v3_python" "$v3_completion_validator" verify \
        --repo "$v3_repo" \
        --candidate-root "$v3_result_root" \
        --formal-reference-root "$v3_formal_root" \
        --v2-reference-root "$v3_v2_root" \
        --reference-miou-root "$v3_reference_miou_root" \
        --summarizer "$v3_summarizer" \
        --postprocess-lock "$v3_postprocess_lock" \
        --output-dir "$v3_comparison_dir"
}

v3_publish_complete_bundle() {
    local v3_staging_dir="$1"
    "$v3_python" "$v3_completion_validator" publish \
        --repo "$v3_repo" \
        --candidate-root "$v3_result_root" \
        --formal-reference-root "$v3_formal_root" \
        --v2-reference-root "$v3_v2_root" \
        --reference-miou-root "$v3_reference_miou_root" \
        --summarizer "$v3_summarizer" \
        --postprocess-lock "$v3_postprocess_lock" \
        --output-dir "$v3_comparison_dir" \
        --staging-dir "$v3_staging_dir"
}

exec 9>"$v3_lock_file"
if ! flock -n 9; then
    echo "TPDCLEANV3_FINALIZER_ALREADY_RUNNING lock=$v3_lock_file"
    exit 0
fi

if [[ -e "$v3_marker" || -L "$v3_marker" ]]; then
    if v3_verify_complete_marker; then
        v3_write_state \
            complete \
            "Validated and reused the existing Clean-v3 comparison marker."
        echo "TPDCLEANV3_FINALIZER_COMPLETE reused=true marker=$v3_marker"
        exit 0
    fi
    v3_write_state \
        failed \
        "An existing Clean-v3 completion marker failed validation."
    echo "TPDCLEANV3_FINALIZER_FAILED reason=invalid_existing_marker marker=$v3_marker" >&2
    exit 1
fi

v3_worker_completion_evidence() {
    local v3_variant="$1"
    local v3_seed="$2"
    local v3_run_dir="$v3_result_root/NUDT-SIRST/$v3_variant/seed_${v3_seed}_${v3_run_tag}"
    local v3_manifest="$v3_launch_root/${v3_variant}_seed${v3_seed}.json"
    local v3_log="$v3_log_root/${v3_variant}_seed${v3_seed}.log"
    "$v3_python" - \
        "$v3_variant" \
        "$v3_seed" \
        "$v3_run_dir" \
        "$v3_manifest" \
        "$v3_log" <<'PY'
import json
import pathlib
import sys

variant, seed_text, run_text, manifest_text, log_text = sys.argv[1:]
seed = int(seed_text)
run_dir = pathlib.Path(run_text).resolve()
manifest_path = pathlib.Path(manifest_text)
log_path = pathlib.Path(log_text)

def regular(path: pathlib.Path) -> bool:
    return path.is_file() and not path.is_symlink()

required = (
    run_dir / "metrics.jsonl",
    run_dir / "summary.json",
    run_dir / "best.pth.tar",
    run_dir / "best_miou.pth.tar",
    run_dir / "pd_fa_sweep_best.pth.json",
    run_dir / "pd_fa_sweep_best_miou.pth.json",
    manifest_path,
    log_path,
)
if not all(regular(path) for path in required):
    raise SystemExit(1)

metrics = (run_dir / "metrics.jsonl").read_text(
    encoding="utf-8"
).splitlines()
if len(metrics) != 800:
    raise SystemExit(1)
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
if not (
    isinstance(summary, dict)
    and summary.get("status") == "complete"
    and summary.get("variant") == variant
    and summary.get("seed") == seed
):
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not (
    isinstance(manifest, dict)
    and manifest.get("schema")
    == "sctransnet_tpd_clean_v3_screen800_launch_v1"
    and manifest.get("variant") == variant
    and manifest.get("seed") == seed
    and pathlib.Path(str(manifest.get("run_directory", ""))).resolve()
    == run_dir
):
    raise SystemExit(1)
completion_prefix = f"TPDCLEANV3_COMPLETE variant={variant} seed={seed} "
if not any(
    line.startswith(completion_prefix) and line.endswith("epochs=800")
    for line in log_path.read_text(encoding="utf-8").splitlines()
):
    raise SystemExit(1)
PY
}

v3_snapshot_file="-"
declare -A v3_collected_evidence_misses=()
while true; do
    v3_snapshot_file="$(mktemp "$v3_launch_root/.finalizer_units.XXXXXX")"
    v3_failure=""
    v3_all_success=1
    v3_any_active=0

    for v3_index in "${!v3_units[@]}"; do
        v3_tag="${v3_unit_tags[$v3_index]}"
        v3_unit="${v3_units[$v3_index]}"
        if ! v3_show="$(
            "$v3_systemctl" --user show "$v3_unit" --no-pager \
                --property=LoadState \
                --property=ActiveState \
                --property=SubState \
                --property=Result \
                --property=ExecMainCode \
                --property=ExecMainStatus
        )"; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$v3_tag" "$v3_unit" command-error unknown unknown unknown unknown unknown \
                >>"$v3_snapshot_file"
            v3_failure="$v3_tag: systemctl show failed"
            break
        fi

        declare -A v3_properties=()
        while IFS='=' read -r v3_key v3_value; do
            if [[ -n "$v3_key" ]]; then
                v3_properties["$v3_key"]="$v3_value"
            fi
        done <<<"$v3_show"
        v3_load_state="${v3_properties[LoadState]:-}"
        v3_active_state="${v3_properties[ActiveState]:-}"
        v3_sub_state="${v3_properties[SubState]:-}"
        v3_result="${v3_properties[Result]:-}"
        v3_exec_code="${v3_properties[ExecMainCode]:-}"
        v3_exec_status="${v3_properties[ExecMainStatus]:-}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$v3_tag" \
            "$v3_unit" \
            "$v3_load_state" \
            "$v3_active_state" \
            "$v3_sub_state" \
            "$v3_result" \
            "$v3_exec_code" \
            "$v3_exec_status" \
            >>"$v3_snapshot_file"

        if [[ -n "$v3_exec_status" && ! "$v3_exec_status" =~ ^[0-9]+$ ]]; then
            v3_failure="$v3_tag: invalid ExecMainStatus=$v3_exec_status"
            break
        fi
        if [[ -n "$v3_exec_status" && "$v3_exec_status" -ne 0 ]]; then
            v3_failure="$v3_tag: nonzero ExecMainStatus=$v3_exec_status"
            break
        fi
        if [[ "$v3_active_state" == "failed" ]]; then
            v3_failure="$v3_tag: ActiveState=failed Result=$v3_result"
            break
        fi
        case "$v3_result" in
            exit-code|signal|timeout|core-dump|watchdog|resources|protocol)
                v3_failure="$v3_tag: unsuccessful Result=$v3_result"
                break
                ;;
        esac

        case "$v3_active_state" in
            active|activating|deactivating|reloading)
                v3_any_active=1
                v3_all_success=0
                ;;
            inactive)
                if [[ "$v3_load_state" == "loaded" &&
                      "$v3_result" == "success" &&
                      "$v3_exec_status" == "0" ]]; then
                    v3_collected_evidence_misses["$v3_tag"]=0
                elif [[ "$v3_load_state" == "not-found" ]]; then
                    v3_variant="${v3_variants[$v3_index]}"
                    v3_seed="${v3_seeds[$v3_index]}"
                    if v3_worker_completion_evidence "$v3_variant" "$v3_seed"; then
                        v3_collected_evidence_misses["$v3_tag"]=0
                    else
                        v3_misses="$(( ${v3_collected_evidence_misses[$v3_tag]:-0} + 1 ))"
                        v3_collected_evidence_misses["$v3_tag"]="$v3_misses"
                        if (( v3_misses >= 5 )); then
                            v3_failure="$v3_tag: unit was unloaded without complete worker evidence after $v3_misses checks"
                            break
                        fi
                        v3_any_active=1
                        v3_all_success=0
                    fi
                else
                    v3_failure="$v3_tag: inactive without successful exit (LoadState=$v3_load_state Result=$v3_result ExecMainStatus=$v3_exec_status)"
                    break
                fi
                ;;
            *)
                v3_failure="$v3_tag: unexpected ActiveState=$v3_active_state"
                break
                ;;
        esac
    done

    if [[ -n "$v3_failure" ]]; then
        v3_write_state failed "$v3_failure" "$v3_snapshot_file"
        rm -f "$v3_snapshot_file"
        echo "TPDCLEANV3_FINALIZER_FAILED reason=$v3_failure" >&2
        exit 1
    fi
    if (( v3_all_success == 1 )); then
        v3_write_state \
            validating_candidates \
            "All four worker units exited successfully; validating artifacts." \
            "$v3_snapshot_file"
        break
    fi
    if (( v3_any_active != 1 )); then
        v3_write_state \
            failed \
            "No worker was active and the four-worker success condition was not met." \
            "$v3_snapshot_file"
        rm -f "$v3_snapshot_file"
        echo "TPDCLEANV3_FINALIZER_FAILED reason=no_active_or_complete_worker" >&2
        exit 1
    fi

    v3_write_state \
        waiting \
        "At least one fixed worker unit is active; checking again after ${v3_poll_seconds}s." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    v3_snapshot_file="-"
    echo "TPDCLEANV3_FINALIZER_WAITING poll_seconds=$v3_poll_seconds"
    "$v3_sleep" "$v3_poll_seconds"
done

if ! "$v3_python" - "$v3_result_root" "$v3_run_tag" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
run_tag = sys.argv[2]
jobs = (
    ("tpd_clean_v3_full", 42),
    ("tpd_clean_v3_sal_capacity", 42),
    ("tpd_clean_v3_full", 3407),
    ("tpd_clean_v3_sal_capacity", 3407),
)
checkpoint_roles = (
    ("best.pth.tar", "pd_fa_sweep_best.pth.json", "best_validation_pd_primary"),
    (
        "best_miou.pth.tar",
        "pd_fa_sweep_best_miou.pth.json",
        "best_validation_miou_secondary",
    ),
)
expected_integrity = {
    "summary_complete",
    "metrics_complete_contiguous_finite",
    "metadata_consistent",
    "official_test_isolated",
    "split_hashes_recomputed_consistent",
    "checkpoint_role_epoch_metrics_consistent",
    "global_selection_keys_recomputed",
    "state_dict_strict_load",
    "fixed_threshold_object_metrics_exact",
}
expected_budgets = {"1e-06", "5e-06", "1e-05", "5e-05", "0.0001"}

def regular(path: pathlib.Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing, linked, or not a file: {path}")

def load_json(path: pathlib.Path, label: str):
    regular(path, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

metrics_total = 0
checkpoint_total = 0
sweep_total = 0
for variant, seed in jobs:
    run_name = f"seed_{seed}_{run_tag}"
    run_dir = root / "NUDT-SIRST" / variant / run_name
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"invalid run directory: {run_dir}")

    metrics_path = run_dir / "metrics.jsonl"
    regular(metrics_path, "metrics")
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 800:
        raise ValueError(f"{variant}/seed{seed}: metrics rows={len(lines)}, expected 800")
    for expected_epoch, raw in enumerate(lines, start=1):
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise ValueError(f"{variant}/seed{seed}: metrics epoch is not an object")
        if event.get("epoch") != expected_epoch:
            raise ValueError(
                f"{variant}/seed{seed}: non-contiguous epoch "
                f"{event.get('epoch')!r}, expected {expected_epoch}"
            )
        if event.get("variant") != variant:
            raise ValueError(
                f"{variant}/seed{seed}: metrics variant mismatch at epoch {expected_epoch}"
            )
        for key in ("train_loss", "learning_rate", "epoch_seconds"):
            value = event.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{variant}/seed{seed}: metrics {key} is not numeric"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{variant}/seed{seed}: metrics {key} is not finite"
                )
    metrics_total += len(lines)

    summary = load_json(run_dir / "summary.json", "summary")
    expected_summary = {
        "status": "complete",
        "variant": variant,
        "dataset": "NUDT-SIRST",
        "seed": seed,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{variant}/seed{seed}: summary {key}={summary.get(key)!r}, "
                f"expected {expected!r}"
            )

    for checkpoint_name, sweep_name, checkpoint_role in checkpoint_roles:
        checkpoint = run_dir / checkpoint_name
        regular(checkpoint, "checkpoint")
        if checkpoint.stat().st_size <= 0:
            raise ValueError(f"empty checkpoint: {checkpoint}")
        checkpoint_total += 1

        sweep_path = run_dir / sweep_name
        sweep = load_json(sweep_path, "sweep")
        sweep_total += 1
        for key, expected in {
            "variant": variant,
            "seed": seed,
            "dataset": "NUDT-SIRST",
            "checkpoint_role": checkpoint_role,
            "official_test_accessed": False,
        }.items():
            if sweep.get(key) != expected:
                raise ValueError(
                    f"{variant}/seed{seed}: sweep {key}={sweep.get(key)!r}, "
                    f"expected {expected!r}"
                )
        checkpoint_path = pathlib.Path(str(sweep.get("checkpoint", ""))).resolve()
        if checkpoint_path != checkpoint.resolve():
            raise ValueError(f"{variant}/seed{seed}: sweep checkpoint path mismatch")
        checkpoint_sha = sha256(checkpoint)
        if sweep.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"{variant}/seed{seed}: sweep checkpoint digest mismatch")
        audit = sweep.get("audit")
        if not isinstance(audit, dict):
            raise ValueError(f"{variant}/seed{seed}: sweep audit is missing")
        if audit.get("expected_epochs") != 800:
            raise ValueError(f"{variant}/seed{seed}: sweep expected_epochs mismatch")
        if audit.get("metrics_event_count") != 800:
            raise ValueError(f"{variant}/seed{seed}: sweep metrics count mismatch")
        if audit.get("metrics_epoch_range") != [1, 800]:
            raise ValueError(f"{variant}/seed{seed}: sweep epoch range mismatch")
        checks = audit.get("integrity_checks_passed")
        if not isinstance(checks, dict) or not expected_integrity.issubset(checks):
            raise ValueError(f"{variant}/seed{seed}: sweep integrity checks missing")
        if any(checks[name] is not True for name in expected_integrity):
            raise ValueError(f"{variant}/seed{seed}: sweep integrity check failed")
        artifact_sha = audit.get("artifact_sha256")
        if not isinstance(artifact_sha, dict):
            raise ValueError(f"{variant}/seed{seed}: sweep artifact hashes missing")
        if artifact_sha.get("checkpoint") != checkpoint_sha:
            raise ValueError(
                f"{variant}/seed{seed}: sweep audit checkpoint digest mismatch"
            )
        fixed_audit = sweep.get("fixed_threshold_0_5_checkpoint_audit")
        if not isinstance(fixed_audit, dict):
            raise ValueError(f"{variant}/seed{seed}: fixed-threshold audit missing")
        budgets = sweep.get("best_points_under_fa_budget")
        if not isinstance(budgets, dict) or set(budgets) != expected_budgets:
            raise ValueError(f"{variant}/seed{seed}: sweep Fa budgets mismatch")

if metrics_total != 3200 or checkpoint_total != 8 or sweep_total != 8:
    raise ValueError(
        "unexpected artifact totals: "
        f"metrics={metrics_total} checkpoints={checkpoint_total} sweeps={sweep_total}"
    )
print(
    "TPDCLEANV3_ARTIFACTS_OK"
    f" runs={len(jobs)}"
    f" metrics={metrics_total}"
    f" checkpoints={checkpoint_total}"
    f" sweeps={sweep_total}",
    flush=True,
)
PY
then
    v3_write_state \
        failed \
        "Clean-v3 worker artifacts failed post-training validation." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_FINALIZER_FAILED reason=artifact_validation" >&2
    exit 1
fi

if ! v3_verify_postprocess_sources; then
    v3_write_state \
        failed \
        "The Clean-v3 postprocess sources changed before aggregation." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_FINALIZER_FAILED reason=postprocess_source_drift_before_aggregation" >&2
    exit 1
fi
v3_summarizer_sha_before="$(sha256sum "$v3_summarizer" | awk '{print $1}')"
v3_staging_dir="$(mktemp -d "$v3_comparison_dir/.staging.XXXXXX")"
v3_write_state \
    aggregating \
    "Validated four runs, 3200 metric rows, eight checkpoints, and eight sweeps; running the canonical summarizer in isolated staging: $v3_staging_dir" \
    "$v3_snapshot_file"

if ! "$v3_python" "$v3_summarizer" \
    --candidate-root "$v3_result_root" \
    --formal-reference-root "$v3_formal_root" \
    --v2-reference-root "$v3_v2_root" \
    --reference-miou-root "$v3_reference_miou_root" \
    --output-dir "$v3_staging_dir"
then
    v3_write_state \
        failed \
        "The Clean-v3 summarizer exited unsuccessfully; staged files were retained at $v3_staging_dir." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_FINALIZER_FAILED reason=summarizer_exit staging=$v3_staging_dir" >&2
    exit 1
fi

v3_summarizer_sha_after="$(sha256sum "$v3_summarizer" | awk '{print $1}')"
if [[ "$v3_summarizer_sha_before" != "$v3_summarizer_sha_after" ]]; then
    v3_write_state \
        failed \
        "The Clean-v3 summarizer changed while it was running." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_FINALIZER_FAILED reason=summarizer_changed" >&2
    exit 1
fi
if ! v3_verify_postprocess_sources; then
    v3_write_state \
        failed \
        "The Clean-v3 postprocess sources changed during aggregation." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_FINALIZER_FAILED reason=postprocess_source_drift_during_aggregation" >&2
    exit 1
fi

if ! v3_publish_complete_bundle "$v3_staging_dir"; then
    v3_write_state \
        failed \
        "The staged Clean-v3 comparison failed strict publication validation; production outputs were not marked complete and staging was retained at $v3_staging_dir." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_FINALIZER_FAILED reason=completion_publish staging=$v3_staging_dir" >&2
    exit 1
fi
if ! v3_verify_complete_marker; then
    v3_write_state \
        failed \
        "The published Clean-v3 completion bundle failed immediate full-input verification." \
        "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_FINALIZER_FAILED reason=completion_verify marker=$v3_marker" >&2
    exit 1
fi

rm -f \
    "$v3_staging_dir/tpd_clean_v3_screen800_comparison.json" \
    "$v3_staging_dir/tpd_clean_v3_screen800_comparison.md"
rmdir "$v3_staging_dir"
v3_write_state \
    complete \
    "Clean-v3 post-training comparison completed; report, manifest, marker, and all 65 bound inputs were verified." \
    "$v3_snapshot_file"
rm -f "$v3_snapshot_file"
echo "TPDCLEANV3_FINALIZER_COMPLETE reused=false marker=$v3_marker"
