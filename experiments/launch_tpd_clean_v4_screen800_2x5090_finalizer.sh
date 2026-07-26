#!/usr/bin/env bash
set -euo pipefail

v4_repo="${TPDCLEANV4_2X_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v4_runner="${TPDCLEANV4_2X_FINALIZER_RUNNER:-$v4_repo/experiments/run_tpd_clean_v4_screen800_2x5090_finalizer.sh}"
v4_summarizer="${TPDCLEANV4_2X_FINALIZER_SUMMARIZER:-$v4_repo/experiments/summarize_tpd_clean_v4_screen800.py}"
v4_completion_validator="${TPDCLEANV4_2X_FINALIZER_COMPLETION_VALIDATOR:-$v4_repo/experiments/validate_tpd_clean_v4_2x_completion.py}"
v4_python="${TPDCLEANV4_2X_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v4_postprocess_lock="${TPDCLEANV4_2X_FINALIZER_POSTPROCESS_LOCK:-$v4_repo/experiments/tpd_clean_v4_2x_postprocess_source_lock.json}"
v4_training_lock="${TPDCLEANV4_2X_FINALIZER_TRAINING_LOCK:-$v4_repo/experiments/tpd_clean_v4_screen800_2x_source_lock.json}"
v4_result_root="${TPDCLEANV4_2X_FINALIZER_RESULT_ROOT:-$v4_repo/experiments/results/tpd_clean_v4_screen800_2x5090_v1}"
v4_systemctl="${TPDCLEANV4_2X_FINALIZER_SYSTEMCTL:-systemctl}"
v4_systemd_run="${TPDCLEANV4_2X_FINALIZER_SYSTEMD_RUN:-systemd-run}"
v4_unit="sctransnet-tpd-clean-v4-2x-screen800-finalizer.service"
v4_marker="$v4_result_root/NUDT-SIRST/comparison/COMPLETE.sha256"

v4_mode="${1:-run}"
if [[ "$v4_mode" != "run" && "$v4_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$v4_repo"
[[ -x "$v4_runner" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_LAUNCH_ABORT reason=runner_not_executable path=$v4_runner" >&2
    exit 1
}
[[ -f "$v4_summarizer" && ! -L "$v4_summarizer" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_LAUNCH_ABORT reason=missing_summarizer path=$v4_summarizer" >&2
    exit 1
}
[[ -f "$v4_completion_validator" && ! -L "$v4_completion_validator" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_LAUNCH_ABORT reason=missing_completion_validator path=$v4_completion_validator" >&2
    exit 1
}
[[ -x "$v4_python" ]] || {
    echo "TPDCLEANV4_2X_FINALIZER_LAUNCH_ABORT reason=python_not_executable path=$v4_python" >&2
    exit 1
}
for v4_path in "$v4_postprocess_lock" "$v4_training_lock"; do
    [[ -f "$v4_path" && ! -L "$v4_path" ]] || {
        echo "TPDCLEANV4_2X_FINALIZER_LAUNCH_ABORT reason=missing_lock path=$v4_path" >&2
        exit 1
    }
done
command -v "$v4_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV4_2X_FINALIZER_LAUNCH_ABORT reason=missing_systemctl command=$v4_systemctl" >&2
    exit 1
}
command -v "$v4_systemd_run" >/dev/null 2>&1 || {
    echo "TPDCLEANV4_2X_FINALIZER_LAUNCH_ABORT reason=missing_systemd_run command=$v4_systemd_run" >&2
    exit 1
}
bash -n "$v4_runner"

"$v4_python" - \
    "$v4_repo" \
    "$v4_postprocess_lock" \
    "$v4_training_lock" \
    "$v4_runner" \
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
runner = pathlib.Path(sys.argv[4]).resolve(strict=True)
summarizer = pathlib.Path(sys.argv[5]).resolve(strict=True)
validator = pathlib.Path(sys.argv[6]).resolve(strict=True)
launcher = pathlib.Path(sys.argv[7]).resolve(strict=True)
canonical = {
    "lock": repo / "experiments/tpd_clean_v4_2x_postprocess_source_lock.json",
    "training lock": repo / "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
    "runner": repo / "experiments/run_tpd_clean_v4_screen800_2x5090_finalizer.sh",
    "summarizer": repo / "experiments/summarize_tpd_clean_v4_screen800.py",
    "validator": repo / "experiments/validate_tpd_clean_v4_2x_completion.py",
    "launcher": repo / "experiments/launch_tpd_clean_v4_screen800_2x5090_finalizer.sh",
}
observed = {
    "lock": lock_path,
    "training lock": training_lock,
    "runner": runner,
    "summarizer": summarizer,
    "validator": validator,
    "launcher": launcher,
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

echo "TPDCLEANV4_2X_FINALIZER_PREFLIGHT_OK unit=$v4_unit"
if [[ "$v4_mode" == "--preflight" ]]; then
    exit 0
fi

if [[ -e "$v4_marker" || -L "$v4_marker" ]]; then
    "$v4_runner"
    exit $?
fi

v4_active_state="$(
    "$v4_systemctl" --user show "$v4_unit" --no-pager \
        --property=ActiveState --value 2>/dev/null || true
)"
case "$v4_active_state" in
    active|activating|deactivating|reloading)
        echo "TPDCLEANV4_2X_FINALIZER_UNIT_REUSED state=$v4_active_state unit=$v4_unit"
        exit 0
        ;;
esac

mkdir -p "$v4_result_root"
"$v4_systemd_run" --user \
    --collect \
    --unit="${v4_unit%.service}" \
    --description="SCTransNet TPD-Clean-v4 screen800 post-training finalizer" \
    --property=Restart=no \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v4_runner"
echo "TPDCLEANV4_2X_FINALIZER_UNIT_STARTED unit=$v4_unit"
