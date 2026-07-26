#!/usr/bin/env bash
set -euo pipefail

# Post-training only: observe four frozen workers, audit their artifacts, and
# publish the Gate A--E report.  This controller never launches model training.
v5_repo="${TPDCLEANV5_2X_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v5_python="${TPDCLEANV5_2X_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v5_result_root="${TPDCLEANV5_2X_FINALIZER_RESULT_ROOT:-$v5_repo/experiments/results/tpd_clean_v5_screen800_2x5090_v1}"
v5_formal_root="${TPDCLEANV5_2X_FINALIZER_FORMAL_ROOT:-$v5_repo/experiments/results/tpd_pe_formal800_4x5090_v1}"
v5_reference_miou_root="${TPDCLEANV5_2X_FINALIZER_REFERENCE_MIOU_ROOT:-$v5_repo/experiments/results/tpd_clean_screen800_4x5090_v1/frozen_reference_miou_runs}"
v5_smoke_root="${TPDCLEANV5_2X_FINALIZER_SMOKE_ROOT:-$v5_repo/experiments/results/tpd_clean_v5_preflight_v1}"
v5_training_lock="${TPDCLEANV5_2X_FINALIZER_TRAINING_LOCK:-$v5_repo/experiments/tpd_clean_v5_screen800_2x_source_lock.json}"
v5_postprocess_lock="${TPDCLEANV5_2X_FINALIZER_POSTPROCESS_LOCK:-$v5_repo/experiments/tpd_clean_v5_2x_postprocess_source_lock.json}"
v5_summarizer="${TPDCLEANV5_2X_FINALIZER_SUMMARIZER:-$v5_repo/experiments/summarize_tpd_clean_v5_screen800.py}"
v5_completion_validator="${TPDCLEANV5_2X_FINALIZER_COMPLETION_VALIDATOR:-$v5_repo/experiments/validate_tpd_clean_v5_2x_completion.py}"
v5_systemctl="${TPDCLEANV5_2X_FINALIZER_SYSTEMCTL:-systemctl}"
v5_sleep="${TPDCLEANV5_2X_FINALIZER_SLEEP:-sleep}"
v5_poll_seconds="${TPDCLEANV5_2X_FINALIZER_POLL_SECONDS:-60}"
v5_max_polls="${TPDCLEANV5_2X_FINALIZER_MAX_POLLS:-0}"

v5_run_tag="screen800_pd_fp32_shared2x5090_v1"
v5_dataset_root="$v5_result_root/NUDT-SIRST"
v5_comparison_dir="$v5_dataset_root/comparison"
v5_launch_root="$v5_result_root/launch"
v5_log_root="$v5_result_root/logs"
v5_lock_root="$v5_result_root/.locks"
v5_state_file="$v5_launch_root/finalizer_state.json"
v5_finalizer_log="$v5_log_root/finalizer.log"
v5_mutex="$v5_lock_root/finalizer.lock"
v5_marker="$v5_comparison_dir/COMPLETE.sha256"
v5_json_name="tpd_clean_v5_screen800_comparison.json"
v5_markdown_name="tpd_clean_v5_screen800_comparison.md"

v5_variants=(
    tpd_clean_v5_full
    tpd_clean_v5_sal_capacity
    tpd_clean_v5_full
    tpd_clean_v5_sal_capacity
)
v5_seeds=(42 42 3407 3407)
v5_unit_tags=(full-s42 cap-s42 full-s3407 cap-s3407)
v5_gpu_uuids=(
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
    GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
)

if [[ ! "$v5_poll_seconds" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=invalid_poll_seconds value=$v5_poll_seconds" >&2
    exit 2
fi
if [[ ! "$v5_max_polls" =~ ^[0-9]+$ ]]; then
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=invalid_max_polls value=$v5_max_polls" >&2
    exit 2
fi

mkdir -p "$v5_comparison_dir" "$v5_launch_root" "$v5_log_root" "$v5_lock_root"
exec > >(tee -a "$v5_finalizer_log") 2>&1
exec 9>"$v5_mutex"
if ! flock -n 9; then
    echo "TPDCLEANV5_2X_FINALIZER_REUSED reason=lock_held"
    exit 0
fi
cd "$v5_repo"

[[ -x "$v5_python" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=python_not_executable path=$v5_python" >&2
    exit 1
}
[[ -f "$v5_summarizer" && ! -L "$v5_summarizer" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=missing_summarizer path=$v5_summarizer" >&2
    exit 1
}
[[ -f "$v5_completion_validator" && ! -L "$v5_completion_validator" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=missing_completion_validator path=$v5_completion_validator" >&2
    exit 1
}
[[ -f "$v5_training_lock" && ! -L "$v5_training_lock" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=missing_training_lock path=$v5_training_lock" >&2
    exit 1
}
[[ -f "$v5_postprocess_lock" && ! -L "$v5_postprocess_lock" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=missing_postprocess_lock path=$v5_postprocess_lock" >&2
    exit 1
}
command -v "$v5_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=missing_systemctl command=$v5_systemctl" >&2
    exit 1
}
command -v "$v5_sleep" >/dev/null 2>&1 || {
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=missing_sleep command=$v5_sleep" >&2
    exit 1
}

v5_verify_postprocess_sources() {
    "$v5_python" - \
        "$v5_repo" \
        "$v5_postprocess_lock" \
        "$v5_training_lock" \
        "$v5_summarizer" \
        "$v5_completion_validator" \
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
    "lock": repo / "experiments/tpd_clean_v5_2x_postprocess_source_lock.json",
    "training lock": repo / "experiments/tpd_clean_v5_screen800_2x_source_lock.json",
    "summarizer": repo / "experiments/summarize_tpd_clean_v5_screen800.py",
    "validator": repo / "experiments/validate_tpd_clean_v5_2x_completion.py",
    "runner": repo / "experiments/run_tpd_clean_v5_screen800_2x5090_finalizer.sh",
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
            f"non-canonical v5 postprocess {label}: "
            f"expected={expected} observed={observed[label]}"
        )
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v5_2x_postprocess_source_lock_v1":
    raise SystemExit("unexpected v5 postprocess source-lock schema")
entries = payload.get("source_sha256")
required = {
    "experiments/summarize_tpd_clean_v5_screen800.py",
    "experiments/validate_tpd_clean_v5_2x_completion.py",
    "experiments/run_tpd_clean_v5_screen800_2x5090_finalizer.sh",
    "experiments/launch_tpd_clean_v5_screen800_2x5090_finalizer.sh",
    "tests/test_summarize_tpd_clean_v5_screen800.py",
    "tests/test_validate_tpd_clean_v5_2x_completion.py",
    "tests/test_tpd_clean_v5_2x_finalizer.py",
    "experiments/TPD_CLEAN_V5_PROTOCOL.md",
    "experiments/TPD_CLEAN_V5_2GPU_PROTOCOL.md",
    "experiments/tpd_clean_v5_screen800_2x_source_lock.json",
}
if not isinstance(entries, dict) or set(entries) != required:
    raise SystemExit("v5 postprocess source-lock entry set differs")
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
    f"TPDCLEANV5_2X_POSTPROCESS_SOURCES_OK checked_entries={len(entries)}",
    flush=True,
)
PY
}

v5_validate_roots() {
    "$v5_python" - \
        "$v5_result_root" \
        "$v5_formal_root" \
        "$v5_reference_miou_root" \
        "$v5_smoke_root" \
        "$v5_comparison_dir" \
        "$v5_state_file" \
        "$v5_marker" <<'PY'
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
        raise SystemExit(f"{label} escaped the v5 result root")
PY
}

v5_write_state() {
    local v5_state="$1"
    local v5_message="$2"
    "$v5_python" - "$v5_state_file" "$v5_state" "$v5_message" "$v5_marker" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sctransnet_tpd_clean_v5_screen800_finalizer_state_v1",
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

v5_verify_complete_marker() {
    "$v5_python" "$v5_completion_validator" verify \
        --repo "$v5_repo" \
        --candidate-root "$v5_result_root" \
        --formal-reference-root "$v5_formal_root" \
        --reference-miou-root "$v5_reference_miou_root" \
        --smoke-root "$v5_smoke_root" \
        --summarizer "$v5_summarizer" \
        --postprocess-lock "$v5_postprocess_lock" \
        --output-dir "$v5_comparison_dir"
}

v5_publish_complete_bundle() {
    local v5_staging_dir="$1"
    "$v5_python" "$v5_completion_validator" publish \
        --repo "$v5_repo" \
        --candidate-root "$v5_result_root" \
        --formal-reference-root "$v5_formal_root" \
        --reference-miou-root "$v5_reference_miou_root" \
        --smoke-root "$v5_smoke_root" \
        --summarizer "$v5_summarizer" \
        --postprocess-lock "$v5_postprocess_lock" \
        --output-dir "$v5_comparison_dir" \
        --staging-dir "$v5_staging_dir"
}

v5_worker_artifacts_ready() {
    local v5_variant="$1"
    local v5_seed="$2"
    local v5_gpu_uuid="$3"
    local v5_run_dir="$v5_dataset_root/$v5_variant/seed_${v5_seed}_${v5_run_tag}"
    local v5_worker_log="$v5_log_root/${v5_variant}_seed${v5_seed}.log"
    local v5_complete="TPDCLEANV5_2X_COMPLETE variant=$v5_variant seed=$v5_seed gpu_uuid=$v5_gpu_uuid epochs=800"
    local v5_complete_count
    local v5_metrics_count

    for v5_name in \
        protocol.json split.json summary.json metrics.jsonl \
        best.pth.tar best_miou.pth.tar last.pth.tar \
        pd_fa_sweep_best.pth.json pd_fa_sweep_best_miou.pth.json; do
        [[ -f "$v5_run_dir/$v5_name" && ! -L "$v5_run_dir/$v5_name" ]] ||
            return 1
    done
    [[ -f "$v5_worker_log" && ! -L "$v5_worker_log" ]] || return 1
    v5_metrics_count="$(wc -l < "$v5_run_dir/metrics.jsonl")"
    [[ "$v5_metrics_count" -eq 800 ]] || return 1
    jq -e '
        .status == "complete" and
        .selection_source == "internal_validation_only" and
        .official_test_accessed == false
    ' "$v5_run_dir/summary.json" >/dev/null || return 1
    v5_complete_count="$(grep -Fxc "$v5_complete" "$v5_worker_log" || true)"
    [[ "$v5_complete_count" -eq 1 ]] || return 1
    if grep -Eiq \
        'TPDCLEANV5_2X_ABORT|Traceback|out of memory|(^|[^[:alnum:]_])OOM([^[:alnum:]_]|$)|resume' \
        "$v5_worker_log"; then
        return 2
    fi
    return 0
}

v5_verify_postprocess_sources
v5_validate_roots
if [[ -e "$v5_marker" || -L "$v5_marker" ]]; then
    if v5_verify_complete_marker; then
        v5_write_state complete "validated and reused existing completion bundle"
        echo "TPDCLEANV5_2X_FINALIZER_COMPLETE reused=true marker=$v5_marker"
        exit 0
    fi
    v5_write_state failed "existing completion marker failed full-input verification"
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=invalid_existing_marker marker=$v5_marker" >&2
    exit 1
fi

v5_poll=0
while true; do
    v5_pending=()
    for v5_index in "${!v5_variants[@]}"; do
        v5_variant="${v5_variants[$v5_index]}"
        v5_seed="${v5_seeds[$v5_index]}"
        v5_tag="${v5_unit_tags[$v5_index]}"
        v5_gpu_uuid="${v5_gpu_uuids[$v5_index]}"
        v5_unit="sctransnet-tpd-clean-v5-2x-$v5_tag.service"

        set +e
        v5_worker_artifacts_ready "$v5_variant" "$v5_seed" "$v5_gpu_uuid"
        v5_ready_status=$?
        set -e
        if [[ "$v5_ready_status" -eq 0 ]]; then
            continue
        fi
        if [[ "$v5_ready_status" -eq 2 ]]; then
            v5_write_state failed "worker log failure evidence: $v5_tag"
            echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=worker_log_failure job=$v5_tag" >&2
            exit 1
        fi

        v5_properties="$(
            "$v5_systemctl" --user show "$v5_unit" --no-pager \
                --property=LoadState \
                --property=ActiveState \
                --property=SubState \
                --property=Result \
                --property=ExecMainCode \
                --property=ExecMainStatus 2>/dev/null || true
        )"
        v5_load_state="$(awk -F= '$1=="LoadState"{print $2}' <<<"$v5_properties")"
        v5_active_state="$(awk -F= '$1=="ActiveState"{print $2}' <<<"$v5_properties")"
        v5_result="$(awk -F= '$1=="Result"{print $2}' <<<"$v5_properties")"
        v5_exec_status="$(awk -F= '$1=="ExecMainStatus"{print $2}' <<<"$v5_properties")"
        case "$v5_active_state" in
            active|activating|deactivating|reloading)
                v5_pending+=("$v5_tag:$v5_active_state")
                ;;
            *)
                if [[ "$v5_load_state" == "not-found" || -z "$v5_load_state" ]]; then
                    v5_write_state failed "worker unit missing before artifacts: $v5_tag"
                else
                    v5_write_state failed \
                        "worker stopped before artifacts: $v5_tag result=$v5_result status=$v5_exec_status"
                fi
                echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=worker_incomplete job=$v5_tag active=$v5_active_state result=$v5_result exec_status=$v5_exec_status" >&2
                exit 1
                ;;
        esac
    done

    if (( ${#v5_pending[@]} == 0 )); then
        break
    fi
    v5_poll=$((v5_poll + 1))
    v5_pending_text="$(IFS=,; echo "${v5_pending[*]}")"
    v5_write_state waiting "poll=$v5_poll pending=$v5_pending_text"
    echo "TPDCLEANV5_2X_FINALIZER_WAIT poll=$v5_poll pending=$v5_pending_text"
    if (( v5_max_polls > 0 && v5_poll >= v5_max_polls )); then
        echo "TPDCLEANV5_2X_FINALIZER_WAIT_LIMIT polls=$v5_poll" >&2
        exit 3
    fi
    "$v5_sleep" "$v5_poll_seconds"
done

v5_verify_postprocess_sources
v5_write_state summarizing "four workers and eight sweeps complete"
v5_stage="$(mktemp -d "$v5_comparison_dir/.finalizer-stage.XXXXXX")"
"$v5_python" "$v5_summarizer" \
    --candidate-root "$v5_result_root" \
    --formal-reference-root "$v5_formal_root" \
    --smoke-root "$v5_smoke_root" \
    --training-source-lock "$v5_training_lock" \
    --output-dir "$v5_stage" \
    --overwrite \
    --require-complete

"$v5_python" - "$v5_stage/$v5_json_name" "$v5_stage/$v5_markdown_name" <<'PY'
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
    report.get("schema") != "sctransnet_tpd_clean_v5_screen800_comparison_v1"
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
    raise SystemExit("staged v5 comparison contract differs")
PY

v5_verify_postprocess_sources
if ! v5_publish_complete_bundle "$v5_stage"; then
    v5_write_state failed "validator rejected staged comparison publication"
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=completion_publish staging=$v5_stage" >&2
    exit 1
fi
if ! v5_verify_complete_marker; then
    v5_write_state failed "published bundle failed immediate full-input verification"
    echo "TPDCLEANV5_2X_FINALIZER_ABORT reason=completion_verify marker=$v5_marker" >&2
    exit 1
fi

rm -f "$v5_stage/$v5_json_name" "$v5_stage/$v5_markdown_name"
rmdir "$v5_stage"
v5_write_state complete "published and verified JSON, Markdown, manifest, and three-row marker"
echo "TPDCLEANV5_2X_FINALIZER_COMPLETE reused=false marker=$v5_marker"
