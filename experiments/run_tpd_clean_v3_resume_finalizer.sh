#!/usr/bin/env bash
set -euo pipefail

v3_repo="${TPDCLEANV3_RESUME_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v3_python="${TPDCLEANV3_RESUME_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v3_candidate_root="${TPDCLEANV3_RESUME_FINALIZER_CANDIDATE_ROOT:-$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1}"
v3_formal_root="${TPDCLEANV3_RESUME_FINALIZER_FORMAL_ROOT:-$v3_repo/experiments/results/tpd_pe_formal800_4x5090_v1}"
v3_v2_root="${TPDCLEANV3_RESUME_FINALIZER_V2_ROOT:-$v3_repo/experiments/results/tpd_clean_screen800_4x5090_v1}"
v3_reference_miou_root="${TPDCLEANV3_RESUME_FINALIZER_REFERENCE_MIOU_ROOT:-$v3_v2_root/frozen_reference_miou_runs}"
v3_summarizer="${TPDCLEANV3_RESUME_FINALIZER_SUMMARIZER:-$v3_repo/experiments/summarize_tpd_clean_v3_screen800.py}"
v3_validator="${TPDCLEANV3_RESUME_FINALIZER_VALIDATOR:-$v3_repo/experiments/validate_tpd_clean_v3_resume_completion.py}"
v3_postprocess_lock="${TPDCLEANV3_RESUME_FINALIZER_POSTPROCESS_LOCK:-$v3_repo/experiments/tpd_clean_v3_resume_postprocess_source_lock.json}"
v3_systemctl="${TPDCLEANV3_RESUME_FINALIZER_SYSTEMCTL:-systemctl}"
v3_sleep="${TPDCLEANV3_RESUME_FINALIZER_SLEEP:-sleep}"
v3_poll_seconds="${TPDCLEANV3_RESUME_FINALIZER_POLL_SECONDS:-60}"

v3_resume_root="$v3_candidate_root/resume_2x5090_v1"
v3_comparison_dir="$v3_candidate_root/NUDT-SIRST/comparison"
v3_state_file="$v3_resume_root/finalizer_state.json"
v3_log_root="$v3_resume_root/logs"
v3_lock_root="$v3_resume_root/.locks"
v3_finalizer_log="$v3_log_root/finalizer.log"
v3_lock_file="$v3_lock_root/finalizer.lock"
v3_marker="$v3_comparison_dir/RESUME_COMPLETE.sha256"
v3_run_tag="screen800_pd_fp32_shared4x5090_v1"

v3_variants=(
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
    tpd_clean_v3_full
    tpd_clean_v3_sal_capacity
)
v3_seeds=(42 42 3407 3407)
v3_boundaries=(279 331 323 372)
v3_gpu_indices=(3 2 2 3)
v3_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)
v3_units=(
    sctransnet-tpd-clean-v3-resume-2x-full-s42.service
    sctransnet-tpd-clean-v3-resume-2x-cap-s42.service
    sctransnet-tpd-clean-v3-resume-2x-full-s3407.service
    sctransnet-tpd-clean-v3-resume-2x-cap-s3407.service
)

if [[ ! "$v3_poll_seconds" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "TPDCLEANV3_RESUME_FINALIZER_ABORT reason=invalid_poll_seconds value=$v3_poll_seconds" >&2
    exit 2
fi

mkdir -p "$v3_comparison_dir" "$v3_log_root" "$v3_lock_root"
exec > >(tee -a "$v3_finalizer_log") 2>&1
cd "$v3_repo"

[[ -x "$v3_python" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_ABORT reason=python_not_executable path=$v3_python" >&2
    exit 1
}
[[ -f "$v3_summarizer" && ! -L "$v3_summarizer" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_ABORT reason=missing_summarizer path=$v3_summarizer" >&2
    exit 1
}
[[ -f "$v3_validator" && ! -L "$v3_validator" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_ABORT reason=missing_validator path=$v3_validator" >&2
    exit 1
}
[[ -f "$v3_postprocess_lock" && ! -L "$v3_postprocess_lock" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_ABORT reason=missing_postprocess_lock path=$v3_postprocess_lock" >&2
    exit 1
}
command -v "$v3_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_RESUME_FINALIZER_ABORT reason=missing_systemctl command=$v3_systemctl" >&2
    exit 1
}
command -v "$v3_sleep" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_RESUME_FINALIZER_ABORT reason=missing_sleep command=$v3_sleep" >&2
    exit 1
}

v3_verify_postprocess_sources() {
    "$v3_python" - \
        "$v3_repo" \
        "$v3_postprocess_lock" \
        "$v3_summarizer" \
        "$v3_validator" \
        "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve(strict=True)
lock_path = pathlib.Path(sys.argv[2]).resolve(strict=True)
summarizer = pathlib.Path(sys.argv[3]).resolve(strict=True)
validator = pathlib.Path(sys.argv[4]).resolve(strict=True)
runner = pathlib.Path(sys.argv[5]).resolve(strict=True)
canonical = {
    "postprocess lock": repo / "experiments/tpd_clean_v3_resume_postprocess_source_lock.json",
    "summarizer": repo / "experiments/summarize_tpd_clean_v3_screen800.py",
    "validator": repo / "experiments/validate_tpd_clean_v3_resume_completion.py",
    "runner": repo / "experiments/run_tpd_clean_v3_resume_finalizer.sh",
}
observed = {
    "postprocess lock": lock_path,
    "summarizer": summarizer,
    "validator": validator,
    "runner": runner,
}
for label, expected in canonical.items():
    if observed[label] != expected.resolve(strict=True):
        raise SystemExit(
            f"non-canonical resume {label}: expected={expected} observed={observed[label]}"
        )
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v3_resume_postprocess_source_lock_v1":
    raise SystemExit("unexpected resume postprocess source-lock schema")
entries = payload.get("source_sha256")
required = {
    "experiments/summarize_tpd_clean_v3_screen800.py",
    "experiments/validate_tpd_clean_v3_completion.py",
    "experiments/validate_tpd_clean_v3_resume_completion.py",
    "experiments/run_tpd_clean_v3_resume_finalizer.sh",
    "experiments/launch_tpd_clean_v3_resume_finalizer.sh",
}
if not isinstance(entries, dict) or not required.issubset(entries):
    raise SystemExit("resume postprocess source lock misses runtime entries")

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

for relative, expected in entries.items():
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or linked postprocess source: {relative}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"postprocess source digest mismatch: {relative} expected={expected} actual={actual}"
        )
training_lock = repo / "experiments/tpd_clean_v3_screen800_source_lock.json"
resume_lock = repo / "experiments/tpd_clean_v3_resume_2x_source_lock.json"
if payload.get("training_source_lock_sha256") != sha256(training_lock):
    raise SystemExit("resume postprocess lock does not bind original training lock")
if payload.get("resume_source_lock_sha256") != sha256(resume_lock):
    raise SystemExit("resume postprocess lock does not bind resume source lock")
print(
    f"TPDCLEANV3_RESUME_POSTPROCESS_SOURCES_OK checked_entries={len(entries)}",
    flush=True,
)
PY
}

v3_verify_postprocess_sources
"$v3_python" - \
    "$v3_candidate_root" \
    "$v3_formal_root" \
    "$v3_v2_root" \
    "$v3_resume_root" \
    "$v3_comparison_dir" \
    "$v3_marker" <<'PY'
import pathlib
import sys

candidate, formal, v2, resume, comparison, marker = (
    pathlib.Path(value).resolve() for value in sys.argv[1:]
)

def within(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path == root or root in path.parents

if candidate in (formal, v2) or within(candidate, formal) or within(formal, candidate):
    raise SystemExit("candidate and formal reference roots overlap")
if within(candidate, v2) or within(v2, candidate):
    raise SystemExit("candidate and Clean-v2 reference roots overlap")
for path, label in ((resume, "resume"), (comparison, "comparison"), (marker, "marker")):
    if not within(path, candidate):
        raise SystemExit(f"{label} path escaped candidate root")
if resume != candidate / "resume_2x5090_v1":
    raise SystemExit("resume root is non-canonical")
if comparison != candidate / "NUDT-SIRST/comparison":
    raise SystemExit("comparison directory is non-canonical")
PY

v3_write_state() {
    local state="$1"
    local message="$2"
    local snapshot="${3:--}"
    "$v3_python" - \
        "$v3_state_file" "$state" "$message" "$snapshot" \
        "$v3_candidate_root" "$v3_marker" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile

state_path = pathlib.Path(sys.argv[1])
state = sys.argv[2]
message = sys.argv[3]
snapshot_path = pathlib.Path(sys.argv[4]) if sys.argv[4] != "-" else None
units = []
if snapshot_path is not None and snapshot_path.is_file():
    for raw in snapshot_path.read_text(encoding="utf-8").splitlines():
        fields = raw.split("\t")
        if len(fields) != 8:
            raise SystemExit(f"invalid finalizer snapshot row: {raw!r}")
        units.append(dict(zip(
            ("tag", "unit", "LoadState", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus"),
            fields,
        )))
payload = {
    "schema": "sctransnet_tpd_clean_v3_resume_finalizer_state_v1",
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "state": state,
    "message": message,
    "candidate_root": str(pathlib.Path(sys.argv[5]).resolve()),
    "completion_marker": str(pathlib.Path(sys.argv[6]).resolve()),
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

v3_validator_common=(
    --repo "$v3_repo"
    --candidate-root "$v3_candidate_root"
    --formal-reference-root "$v3_formal_root"
    --v2-reference-root "$v3_v2_root"
    --reference-miou-root "$v3_reference_miou_root"
    --summarizer "$v3_summarizer"
    --postprocess-lock "$v3_postprocess_lock"
    --output-dir "$v3_comparison_dir"
)

v3_verify_completion() {
    "$v3_python" "$v3_validator" verify "${v3_validator_common[@]}"
}

v3_audit_completion() {
    "$v3_python" "$v3_validator" audit "${v3_validator_common[@]}"
}

v3_publish_completion() {
    local staging_dir="$1"
    "$v3_python" "$v3_validator" publish \
        "${v3_validator_common[@]}" \
        --staging-dir "$staging_dir"
}

exec 9>"$v3_lock_file"
if ! flock -n 9; then
    echo "TPDCLEANV3_RESUME_FINALIZER_ALREADY_RUNNING lock=$v3_lock_file"
    exit 0
fi

if [[ -e "$v3_marker" || -L "$v3_marker" ]]; then
    if v3_verify_completion; then
        v3_write_state complete "Validated and reused the existing resume completion marker."
        echo "TPDCLEANV3_RESUME_FINALIZER_COMPLETE reused=true marker=$v3_marker"
        exit 0
    fi
    v3_write_state failed "The existing resume completion marker failed validation."
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=invalid_existing_marker marker=$v3_marker" >&2
    exit 1
fi

v3_worker_completion_evidence() {
    local variant="$1"
    local seed="$2"
    local boundary="$3"
    local run_dir="$v3_candidate_root/NUDT-SIRST/$variant/seed_${seed}_${v3_run_tag}"
    local manifest="$v3_resume_root/manifests/${variant}_seed${seed}.json"
    local log="$v3_resume_root/logs/${variant}_seed${seed}.log"
    "$v3_python" - "$variant" "$seed" "$boundary" "$run_dir" "$manifest" "$log" <<'PY'
import json
import pathlib
import sys

variant, seed_text, boundary_text, run_text, manifest_text, log_text = sys.argv[1:]
seed = int(seed_text)
boundary = int(boundary_text)
run_dir = pathlib.Path(run_text).resolve()
manifest_path = pathlib.Path(manifest_text)
log_path = pathlib.Path(log_text)
required = (
    run_dir / "metrics.jsonl",
    run_dir / "summary.json",
    run_dir / "resume_provenance.json",
    run_dir / "resume_segments.jsonl",
    manifest_path,
    log_path,
)
if any(not path.is_file() or path.is_symlink() for path in required):
    raise SystemExit(1)
if len((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) != 800:
    raise SystemExit(1)
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not (
    summary.get("status") == "complete"
    and summary.get("variant") == variant
    and summary.get("seed") == seed
    and manifest.get("schema") == "sctransnet_tpd_clean_v3_resume_2x5090_launch_v1"
    and manifest.get("variant") == variant
    and manifest.get("seed") == seed
    and manifest.get("boundary_epoch") == boundary
    and pathlib.Path(str(manifest.get("run_directory", ""))).resolve() == run_dir
):
    raise SystemExit(1)
completion = (
    f"TPDCLEANV3_RESUME_2X_COMPLETE variant={variant} seed={seed} "
    f"gpu_uuid={manifest.get('resume_gpu_uuid')} boundary_epoch={boundary} epochs=800"
)
lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines or lines[-1] != completion:
    raise SystemExit(1)
PY
}

v3_snapshot_file="-"
declare -A v3_collected_evidence_misses=()
while true; do
    v3_snapshot_file="$(mktemp "$v3_resume_root/.finalizer_units.XXXXXX")"
    v3_failure=""
    v3_all_success=1
    v3_any_active=0
    for v3_index in "${!v3_units[@]}"; do
        v3_tag="${v3_unit_tags[$v3_index]}"
        v3_unit="${v3_units[$v3_index]}"
        if ! v3_show="$(
            "$v3_systemctl" --user show "$v3_unit" --no-pager \
                --property=LoadState --property=ActiveState --property=SubState \
                --property=Result --property=ExecMainCode --property=ExecMainStatus
        )"; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$v3_tag" "$v3_unit" command-error unknown unknown unknown unknown unknown \
                >>"$v3_snapshot_file"
            v3_failure="$v3_tag: systemctl show failed"
            break
        fi
        declare -A v3_properties=()
        while IFS='=' read -r v3_key v3_value; do
            [[ -n "$v3_key" ]] && v3_properties["$v3_key"]="$v3_value"
        done <<<"$v3_show"
        v3_load_state="${v3_properties[LoadState]:-}"
        v3_active_state="${v3_properties[ActiveState]:-}"
        v3_sub_state="${v3_properties[SubState]:-}"
        v3_result="${v3_properties[Result]:-}"
        v3_exec_code="${v3_properties[ExecMainCode]:-}"
        v3_exec_status="${v3_properties[ExecMainStatus]:-}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$v3_tag" "$v3_unit" "$v3_load_state" "$v3_active_state" \
            "$v3_sub_state" "$v3_result" "$v3_exec_code" "$v3_exec_status" \
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
                if [[ "$v3_load_state" == "loaded" && "$v3_result" == "success" && "$v3_exec_status" == "0" ]]; then
                    v3_collected_evidence_misses["$v3_tag"]=0
                elif [[ "$v3_load_state" == "not-found" ]]; then
                    if v3_worker_completion_evidence \
                        "${v3_variants[$v3_index]}" \
                        "${v3_seeds[$v3_index]}" \
                        "${v3_boundaries[$v3_index]}"; then
                        v3_collected_evidence_misses["$v3_tag"]=0
                    else
                        v3_misses="$(( ${v3_collected_evidence_misses[$v3_tag]:-0} + 1 ))"
                        v3_collected_evidence_misses["$v3_tag"]="$v3_misses"
                        if (( v3_misses >= 5 )); then
                            v3_failure="$v3_tag: collected unit lacks completion evidence"
                            break
                        fi
                        v3_any_active=1
                        v3_all_success=0
                    fi
                else
                    v3_failure="$v3_tag: inactive without successful exit"
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
        echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=$v3_failure" >&2
        exit 1
    fi
    if (( v3_all_success == 1 )); then
        v3_write_state validating "All four resume workers completed; running the independent completion audit." "$v3_snapshot_file"
        break
    fi
    if (( v3_any_active != 1 )); then
        v3_write_state failed "No resume worker is active and completion is not established." "$v3_snapshot_file"
        rm -f "$v3_snapshot_file"
        echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=no_active_or_complete_worker" >&2
        exit 1
    fi
    v3_write_state waiting "Resume workers are active; polling again after ${v3_poll_seconds}s." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    v3_snapshot_file="-"
    echo "TPDCLEANV3_RESUME_FINALIZER_WAITING poll_seconds=$v3_poll_seconds"
    "$v3_sleep" "$v3_poll_seconds"
done

if ! v3_audit_completion; then
    v3_write_state failed "The resumed-run completion audit failed." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=completion_audit" >&2
    exit 1
fi
if ! v3_verify_postprocess_sources; then
    v3_write_state failed "Postprocess sources changed before aggregation." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=source_drift_before_aggregation" >&2
    exit 1
fi

v3_summarizer_sha_before="$(sha256sum "$v3_summarizer" | awk '{print $1}')"
v3_staging_dir="$(mktemp -d "$v3_comparison_dir/.resume-staging.XXXXXX")"
v3_write_state aggregating "Running the canonical seven-gate summarizer in isolated staging: $v3_staging_dir" "$v3_snapshot_file"
if ! "$v3_python" "$v3_summarizer" \
    --candidate-root "$v3_candidate_root" \
    --formal-reference-root "$v3_formal_root" \
    --v2-reference-root "$v3_v2_root" \
    --reference-miou-root "$v3_reference_miou_root" \
    --output-dir "$v3_staging_dir"
then
    v3_write_state failed "The canonical summarizer failed; staging was retained at $v3_staging_dir." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=summarizer_exit staging=$v3_staging_dir" >&2
    exit 1
fi
v3_summarizer_sha_after="$(sha256sum "$v3_summarizer" | awk '{print $1}')"
if [[ "$v3_summarizer_sha_before" != "$v3_summarizer_sha_after" ]]; then
    v3_write_state failed "The canonical summarizer changed while running." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=summarizer_changed" >&2
    exit 1
fi
if ! v3_verify_postprocess_sources; then
    v3_write_state failed "Postprocess sources changed during aggregation." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=source_drift_during_aggregation" >&2
    exit 1
fi
if ! v3_publish_completion "$v3_staging_dir"; then
    v3_write_state failed "Strict resume publication failed; staging was retained at $v3_staging_dir." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=completion_publish staging=$v3_staging_dir" >&2
    exit 1
fi
if ! v3_verify_completion; then
    v3_write_state failed "The published resume completion bundle failed immediate verification." "$v3_snapshot_file"
    rm -f "$v3_snapshot_file"
    echo "TPDCLEANV3_RESUME_FINALIZER_FAILED reason=completion_verify marker=$v3_marker" >&2
    exit 1
fi

rm -f \
    "$v3_staging_dir/tpd_clean_v3_screen800_comparison.json" \
    "$v3_staging_dir/tpd_clean_v3_screen800_comparison.md"
rmdir "$v3_staging_dir"
v3_write_state complete "Resume comparison, 116 bound input files, report, manifest, and marker were verified." "$v3_snapshot_file"
rm -f "$v3_snapshot_file"
echo "TPDCLEANV3_RESUME_FINALIZER_COMPLETE reused=false marker=$v3_marker"
