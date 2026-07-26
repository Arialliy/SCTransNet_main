#!/usr/bin/env bash
set -euo pipefail

# Post-training only: observe four frozen workers, audit their artifacts, and
# publish the Gate A--E report.  This controller never launches model training.
v4_repo="${TPDCLEANV4_2X_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v4_python="${TPDCLEANV4_2X_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v4_result_root="${TPDCLEANV4_2X_FINALIZER_RESULT_ROOT:-$v4_repo/experiments/results/tpd_clean_v4_screen800_2x5090_v1}"
v4_formal_root="${TPDCLEANV4_2X_FINALIZER_FORMAL_ROOT:-$v4_repo/experiments/results/tpd_pe_formal800_4x5090_v1}"
v4_reference_miou_root="${TPDCLEANV4_2X_FINALIZER_REFERENCE_MIOU_ROOT:-$v4_repo/experiments/results/tpd_clean_screen800_4x5090_v1/frozen_reference_miou_runs}"
v4_smoke_root="${TPDCLEANV4_2X_FINALIZER_SMOKE_ROOT:-$v4_repo/experiments/results/tpd_clean_v4_preflight_v1}"
v4_training_lock="${TPDCLEANV4_2X_FINALIZER_TRAINING_LOCK:-$v4_repo/experiments/tpd_clean_v4_screen800_2x_source_lock.json}"
v4_postprocess_lock="${TPDCLEANV4_2X_FINALIZER_POSTPROCESS_LOCK:-$v4_repo/experiments/tpd_clean_v4_2x_postprocess_source_lock.json}"
v4_summarizer="${TPDCLEANV4_2X_FINALIZER_SUMMARIZER:-$v4_repo/experiments/summarize_tpd_clean_v4_screen800.py}"
v4_completion_validator="${TPDCLEANV4_2X_FINALIZER_COMPLETION_VALIDATOR:-$v4_repo/experiments/validate_tpd_clean_v4_2x_completion.py}"
v4_systemctl="${TPDCLEANV4_2X_FINALIZER_SYSTEMCTL:-systemctl}"
v4_sleep="${TPDCLEANV4_2X_FINALIZER_SLEEP:-sleep}"
v4_poll_seconds="${TPDCLEANV4_2X_FINALIZER_POLL_SECONDS:-60}"
v4_max_polls="${TPDCLEANV4_2X_FINALIZER_MAX_POLLS:-0}"

v4_run_tag="screen800_pd_fp32_shared2x5090_v1"
v4_dataset_root="$v4_result_root/NUDT-SIRST"
v4_comparison_dir="$v4_dataset_root/comparison"
v4_launch_root="$v4_result_root/launch"
v4_log_root="$v4_result_root/logs"
v4_lock_root="$v4_result_root/.locks"
v4_state_file="$v4_launch_root/finalizer_state.json"
v4_finalizer_log="$v4_log_root/finalizer.log"
v4_mutex="$v4_lock_root/finalizer.lock"
v4_marker="$v4_comparison_dir/COMPLETE.sha256"
v4_json_name="tpd_clean_v4_screen800_comparison.json"
v4_markdown_name="tpd_clean_v4_screen800_comparison.md"

v4_variants=(
    tpd_clean_v4_full
    tpd_clean_v4_sal_capacity
    tpd_clean_v4_full
    tpd_clean_v4_sal_capacity
)
v4_seeds=(42 42 3407 3407)
v4_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)
v4_gpu_uuids=(
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
)

if [[ ! "$v4_poll_seconds" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=invalid_poll_seconds value=$v4_poll_seconds" >&2
    exit 2
fi
if [[ ! "$v4_max_polls" =~ ^[0-9]+$ ]]; then
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=invalid_max_polls value=$v4_max_polls" >&2
    exit 2
fi

mkdir -p "$v4_comparison_dir" "$v4_launch_root" "$v4_log_root" "$v4_lock_root"
exec > >(tee -a "$v4_finalizer_log") 2>&1
exec 9>"$v4_mutex"
if ! flock -n 9; then
    echo "TPDCLEANV4_2X_FINALIZER_REUSED reason=lock_held"
    exit 0
fi
cd "$v4_repo"

[[ -x "$v4_python" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=python_not_executable path=$v4_python" >&2
    exit 1
}
[[ -f "$v4_summarizer" && ! -L "$v4_summarizer" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=missing_summarizer path=$v4_summarizer" >&2
    exit 1
}
[[ -f "$v4_completion_validator" && ! -L "$v4_completion_validator" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=missing_completion_validator path=$v4_completion_validator" >&2
    exit 1
}
[[ -f "$v4_training_lock" && ! -L "$v4_training_lock" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=missing_training_lock path=$v4_training_lock" >&2
    exit 1
}
[[ -f "$v4_postprocess_lock" && ! -L "$v4_postprocess_lock" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=missing_postprocess_lock path=$v4_postprocess_lock" >&2
    exit 1
}
command -v "$v4_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=missing_systemctl command=$v4_systemctl" >&2
    exit 1
}
command -v "$v4_sleep" >/dev/null 2>&1 || {
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=missing_sleep command=$v4_sleep" >&2
    exit 1
}

v4_verify_postprocess_sources() {
    "$v4_python" - \
        "$v4_repo" \
        "$v4_postprocess_lock" \
        "$v4_training_lock" \
        "$v4_summarizer" \
        "$v4_completion_validator" \
        "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve(strict=True)
lock_path = pathlib.Path(sys.argv[2]).resolve(strict=True)
training_lock = pathlib.Path(sys.argv[3]).resolve(strict=True)
summarizer = pathlib.Path(sys.argv[4]).resolve(strict=True)
validator = pathlib.Path(sys.argv[5]).resolve(strict=True)
runner = pathlib.Path(sys.argv[6]).resolve(strict=True)
canonical = {
    "lock": repo / "experiments/tpd_clean_v4_2x_postprocess_source_lock.json",
    "training lock": repo / "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
    "summarizer": repo / "experiments/summarize_tpd_clean_v4_screen800.py",
    "validator": repo / "experiments/validate_tpd_clean_v4_2x_completion.py",
    "runner": repo / "experiments/run_tpd_clean_v4_screen800_2x5090_finalizer.sh",
}
observed = {
    "lock": lock_path,
    "training lock": training_lock,
    "summarizer": summarizer,
    "validator": validator,
    "runner": runner,
}
for label, expected in canonical.items():
    if observed[label] != expected.resolve(strict=True):
        raise SystemExit(
            f"non-canonical v4 postprocess {label}: "
            f"expected={expected} observed={observed[label]}"
        )
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v4_2x_postprocess_source_lock_v1":
    raise SystemExit("unexpected v4 postprocess source-lock schema")
entries = payload.get("source_sha256")
required = {
    "experiments/summarize_tpd_clean_v4_screen800.py",
    "experiments/validate_tpd_clean_v4_2x_completion.py",
    "experiments/run_tpd_clean_v4_screen800_2x5090_finalizer.sh",
    "experiments/launch_tpd_clean_v4_screen800_2x5090_finalizer.sh",
    "tests/test_summarize_tpd_clean_v4_screen800.py",
    "tests/test_validate_tpd_clean_v4_2x_completion.py",
    "tests/test_tpd_clean_v4_2x_finalizer.py",
    "experiments/TPD_CLEAN_V4_PROTOCOL.md",
    "experiments/TPD_CLEAN_V4_2GPU_PROTOCOL.md",
    "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
}
if not isinstance(entries, dict) or set(entries) != required:
    raise SystemExit("v4 postprocess source-lock entry set differs")
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
training_digest = hashlib.sha256(training_lock.read_bytes()).hexdigest()
if payload.get("training_source_lock_sha256") != training_digest:
    raise SystemExit("postprocess lock does not bind the current training lock")
print(
    f"TPDCLEANV4_2X_POSTPROCESS_SOURCES_OK checked_entries={len(entries)}",
    flush=True,
)
PY
}

v4_validate_roots() {
    "$v4_python" - \
        "$v4_result_root" \
        "$v4_formal_root" \
        "$v4_reference_miou_root" \
        "$v4_smoke_root" \
        "$v4_comparison_dir" \
        "$v4_state_file" \
        "$v4_marker" <<'PY'
import pathlib
import sys

result, formal, reference_miou, smoke, comparison, state, marker = (
    pathlib.Path(value).resolve() for value in sys.argv[1:]
)

def within(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path == root or root in path.parents

for reference, label in (
    (formal, "frozen formal"),
    (reference_miou, "frozen mIoU"),
    (smoke, "smoke"),
):
    if result == reference or within(result, reference) or within(reference, result):
        raise SystemExit(f"candidate and {label} roots overlap")
for path, label in (
    (comparison, "comparison"),
    (state, "state"),
    (marker, "marker"),
):
    if not within(path, result):
        raise SystemExit(f"{label} escaped the v4 result root")
PY
}

v4_write_state() {
    local v4_state="$1"
    local v4_message="$2"
    "$v4_python" - "$v4_state_file" "$v4_state" "$v4_message" "$v4_marker" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sctransnet_tpd_clean_v4_screen800_finalizer_state_v1",
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "state": sys.argv[2],
    "message": sys.argv[3],
    "completion_marker": str(pathlib.Path(sys.argv[4]).resolve()),
}
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

v4_verify_complete_marker() {
    "$v4_python" "$v4_completion_validator" verify \
        --repo "$v4_repo" \
        --candidate-root "$v4_result_root" \
        --formal-reference-root "$v4_formal_root" \
        --reference-miou-root "$v4_reference_miou_root" \
        --smoke-root "$v4_smoke_root" \
        --summarizer "$v4_summarizer" \
        --postprocess-lock "$v4_postprocess_lock" \
        --output-dir "$v4_comparison_dir"
}

v4_publish_complete_bundle() {
    local v4_staging_dir="$1"
    "$v4_python" "$v4_completion_validator" publish \
        --repo "$v4_repo" \
        --candidate-root "$v4_result_root" \
        --formal-reference-root "$v4_formal_root" \
        --reference-miou-root "$v4_reference_miou_root" \
        --smoke-root "$v4_smoke_root" \
        --summarizer "$v4_summarizer" \
        --postprocess-lock "$v4_postprocess_lock" \
        --output-dir "$v4_comparison_dir" \
        --staging-dir "$v4_staging_dir"
}

v4_worker_artifacts_ready() {
    local v4_variant="$1"
    local v4_seed="$2"
    local v4_gpu_uuid="$3"
    local v4_run_dir="$v4_dataset_root/$v4_variant/seed_${v4_seed}_${v4_run_tag}"
    local v4_worker_log="$v4_log_root/${v4_variant}_seed${v4_seed}.log"
    local v4_complete="TPDCLEANV4_2X_COMPLETE variant=$v4_variant seed=$v4_seed gpu_uuid=$v4_gpu_uuid epochs=800"
    local v4_complete_count
    local v4_metrics_count

    for v4_name in \
        protocol.json split.json summary.json metrics.jsonl \
        best.pth.tar best_miou.pth.tar last.pth.tar \
        pd_fa_sweep_best.pth.json pd_fa_sweep_best_miou.pth.json; do
        [[ -f "$v4_run_dir/$v4_name" && ! -L "$v4_run_dir/$v4_name" ]] ||
            return 1
    done
    [[ -f "$v4_worker_log" && ! -L "$v4_worker_log" ]] || return 1
    v4_metrics_count="$(wc -l < "$v4_run_dir/metrics.jsonl")"
    [[ "$v4_metrics_count" -eq 800 ]] || return 1
    jq -e '
        .status == "complete" and
        .selection_source == "internal_validation_only" and
        .official_test_accessed == false
    ' "$v4_run_dir/summary.json" >/dev/null || return 1
    v4_complete_count="$(grep -Fxc "$v4_complete" "$v4_worker_log" || true)"
    [[ "$v4_complete_count" -eq 1 ]] || return 1
    if grep -Eiq \
        'TPDCLEANV4_2X_ABORT|Traceback|out of memory|(^|[^[:alnum:]_])OOM([^[:alnum:]_]|$)|resume' \
        "$v4_worker_log"; then
        return 2
    fi
    return 0
}

v4_verify_postprocess_sources
v4_validate_roots
if [[ -e "$v4_marker" || -L "$v4_marker" ]]; then
    if v4_verify_complete_marker; then
        v4_write_state complete "validated and reused existing completion bundle"
        echo "TPDCLEANV4_2X_FINALIZER_COMPLETE reused=true marker=$v4_marker"
        exit 0
    fi
    v4_write_state failed "existing completion marker failed full-input verification"
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=invalid_existing_marker marker=$v4_marker" >&2
    exit 1
fi

v4_poll=0
while true; do
    v4_pending=()
    for v4_index in "${!v4_variants[@]}"; do
        v4_variant="${v4_variants[$v4_index]}"
        v4_seed="${v4_seeds[$v4_index]}"
        v4_tag="${v4_unit_tags[$v4_index]}"
        v4_gpu_uuid="${v4_gpu_uuids[$v4_index]}"
        v4_unit="sctransnet-tpd-clean-v4-2x-$v4_tag.service"

        set +e
        v4_worker_artifacts_ready "$v4_variant" "$v4_seed" "$v4_gpu_uuid"
        v4_ready_status=$?
        set -e
        if [[ "$v4_ready_status" -eq 0 ]]; then
            continue
        fi
        if [[ "$v4_ready_status" -eq 2 ]]; then
            v4_write_state failed "worker log failure evidence: $v4_tag"
            echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=worker_log_failure job=$v4_tag" >&2
            exit 1
        fi

        v4_properties="$(
            "$v4_systemctl" --user show "$v4_unit" --no-pager \
                --property=LoadState \
                --property=ActiveState \
                --property=SubState \
                --property=Result \
                --property=ExecMainCode \
                --property=ExecMainStatus 2>/dev/null || true
        )"
        v4_load_state="$(awk -F= '$1=="LoadState"{print $2}' <<<"$v4_properties")"
        v4_active_state="$(awk -F= '$1=="ActiveState"{print $2}' <<<"$v4_properties")"
        v4_result="$(awk -F= '$1=="Result"{print $2}' <<<"$v4_properties")"
        v4_exec_status="$(awk -F= '$1=="ExecMainStatus"{print $2}' <<<"$v4_properties")"
        case "$v4_active_state" in
            active|activating|deactivating|reloading)
                v4_pending+=("$v4_tag:$v4_active_state")
                ;;
            *)
                if [[ "$v4_load_state" == "not-found" || -z "$v4_load_state" ]]; then
                    v4_write_state failed "worker unit missing before artifacts: $v4_tag"
                else
                    v4_write_state failed \
                        "worker stopped before artifacts: $v4_tag result=$v4_result status=$v4_exec_status"
                fi
                echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=worker_incomplete job=$v4_tag active=$v4_active_state result=$v4_result exec_status=$v4_exec_status" >&2
                exit 1
                ;;
        esac
    done

    if (( ${#v4_pending[@]} == 0 )); then
        break
    fi
    v4_poll=$((v4_poll + 1))
    v4_pending_text="$(IFS=,; echo "${v4_pending[*]}")"
    v4_write_state waiting "poll=$v4_poll pending=$v4_pending_text"
    echo "TPDCLEANV4_2X_FINALIZER_WAIT poll=$v4_poll pending=$v4_pending_text"
    if (( v4_max_polls > 0 && v4_poll >= v4_max_polls )); then
        echo "TPDCLEANV4_2X_FINALIZER_WAIT_LIMIT polls=$v4_poll" >&2
        exit 3
    fi
    "$v4_sleep" "$v4_poll_seconds"
done

v4_verify_postprocess_sources
v4_write_state summarizing "four workers and eight sweeps complete"
v4_stage="$(mktemp -d "$v4_comparison_dir/.finalizer-stage.XXXXXX")"
"$v4_python" "$v4_summarizer" \
    --candidate-root "$v4_result_root" \
    --formal-reference-root "$v4_formal_root" \
    --smoke-root "$v4_smoke_root" \
    --training-source-lock "$v4_training_lock" \
    --output-dir "$v4_stage" \
    --overwrite \
    --require-complete

"$v4_python" - "$v4_stage/$v4_json_name" "$v4_stage/$v4_markdown_name" <<'PY'
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
markdown_path = pathlib.Path(sys.argv[2])
if (
    not json_path.is_file()
    or json_path.is_symlink()
    or not markdown_path.is_file()
    or markdown_path.is_symlink()
):
    raise SystemExit("staged comparison artifacts missing or linked")
report = json.loads(json_path.read_text(encoding="utf-8"))
gate = report.get("engineering_gate_passed")
expected_decision = "ENGINEERING_GATE_PASS" if gate is True else "ENGINEERING_GATE_FAIL"
boundary = report.get("decision_boundary")
if (
    report.get("schema") != "sctransnet_tpd_clean_v4_screen800_comparison_v1"
    or report.get("status") != "complete"
    or report.get("gate_evaluated") is not True
    or type(gate) is not bool
    or report.get("decision") != expected_decision
    or report.get("ner_stage_authorized") is not gate
    or not isinstance(boundary, dict)
    or boundary.get("automatic_mainline_replacement") is not False
    or boundary.get("mainline_changed") is not False
    or boundary.get("paper_core_established") is not False
    or boundary.get("stability_claim_supported") is not False
):
    raise SystemExit("staged v4 comparison contract differs")
PY

v4_verify_postprocess_sources
if ! v4_publish_complete_bundle "$v4_stage"; then
    v4_write_state failed "validator rejected staged comparison publication"
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=completion_publish staging=$v4_stage" >&2
    exit 1
fi
if ! v4_verify_complete_marker; then
    v4_write_state failed "published bundle failed immediate full-input verification"
    echo "TPDCLEANV4_2X_FINALIZER_ABORT reason=completion_verify marker=$v4_marker" >&2
    exit 1
fi

rm -f "$v4_stage/$v4_json_name" "$v4_stage/$v4_markdown_name"
rmdir "$v4_stage"
v4_write_state complete "published and verified JSON, Markdown, manifest, and three-row marker"
echo "TPDCLEANV4_2X_FINALIZER_COMPLETE reused=false marker=$v4_marker"
