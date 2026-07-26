#!/usr/bin/env bash
set -euo pipefail

v5_repo="${TPDCLEANV5_2X_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v5_runner="${TPDCLEANV5_2X_FINALIZER_RUNNER:-$v5_repo/experiments/run_tpd_clean_v5_screen800_2x5090_finalizer.sh}"
v5_summarizer="${TPDCLEANV5_2X_FINALIZER_SUMMARIZER:-$v5_repo/experiments/summarize_tpd_clean_v5_screen800.py}"
v5_completion_validator="${TPDCLEANV5_2X_FINALIZER_COMPLETION_VALIDATOR:-$v5_repo/experiments/validate_tpd_clean_v5_2x_completion.py}"
v5_python="${TPDCLEANV5_2X_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v5_postprocess_lock="${TPDCLEANV5_2X_FINALIZER_POSTPROCESS_LOCK:-$v5_repo/experiments/tpd_clean_v5_2x_postprocess_source_lock.json}"
v5_training_lock="${TPDCLEANV5_2X_FINALIZER_TRAINING_LOCK:-$v5_repo/experiments/tpd_clean_v5_screen800_2x_source_lock.json}"
v5_result_root="${TPDCLEANV5_2X_FINALIZER_RESULT_ROOT:-$v5_repo/experiments/results/tpd_clean_v5_screen800_2x5090_v1}"
v5_systemctl="${TPDCLEANV5_2X_FINALIZER_SYSTEMCTL:-systemctl}"
v5_systemd_run="${TPDCLEANV5_2X_FINALIZER_SYSTEMD_RUN:-systemd-run}"
v5_unit="sctransnet-tpd-clean-v5-2x-screen800-finalizer.service"
v5_marker="$v5_result_root/NUDT-SIRST/comparison/COMPLETE.sha256"

v5_mode="${1:-run}"
if [[ "$v5_mode" != "run" && "$v5_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$v5_repo"
[[ -x "$v5_runner" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_LAUNCH_ABORT reason=runner_not_executable path=$v5_runner" >&2
    exit 1
}
[[ -f "$v5_summarizer" && ! -L "$v5_summarizer" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_LAUNCH_ABORT reason=missing_summarizer path=$v5_summarizer" >&2
    exit 1
}
[[ -f "$v5_completion_validator" && ! -L "$v5_completion_validator" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_LAUNCH_ABORT reason=missing_completion_validator path=$v5_completion_validator" >&2
    exit 1
}
[[ -x "$v5_python" ]] || {
    echo "TPDCLEANV5_2X_FINALIZER_LAUNCH_ABORT reason=python_not_executable path=$v5_python" >&2
    exit 1
}
for v5_path in "$v5_postprocess_lock" "$v5_training_lock"; do
    [[ -f "$v5_path" && ! -L "$v5_path" ]] || {
        echo "TPDCLEANV5_2X_FINALIZER_LAUNCH_ABORT reason=missing_lock path=$v5_path" >&2
        exit 1
    }
done
command -v "$v5_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV5_2X_FINALIZER_LAUNCH_ABORT reason=missing_systemctl command=$v5_systemctl" >&2
    exit 1
}
command -v "$v5_systemd_run" >/dev/null 2>&1 || {
    echo "TPDCLEANV5_2X_FINALIZER_LAUNCH_ABORT reason=missing_systemd_run command=$v5_systemd_run" >&2
    exit 1
}
bash -n "$v5_runner"

"$v5_python" - \
    "$v5_repo" \
    "$v5_postprocess_lock" \
    "$v5_training_lock" \
    "$v5_runner" \
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
runner = pathlib.Path(sys.argv[4]).resolve(strict=True)
summarizer = pathlib.Path(sys.argv[5]).resolve(strict=True)
validator = pathlib.Path(sys.argv[6]).resolve(strict=True)
launcher = pathlib.Path(sys.argv[7]).resolve(strict=True)
canonical = {
    "lock": repo / "experiments/tpd_clean_v5_2x_postprocess_source_lock.json",
    "training lock": repo / "experiments/tpd_clean_v5_screen800_2x_source_lock.json",
    "runner": repo / "experiments/run_tpd_clean_v5_screen800_2x5090_finalizer.sh",
    "summarizer": repo / "experiments/summarize_tpd_clean_v5_screen800.py",
    "validator": repo / "experiments/validate_tpd_clean_v5_2x_completion.py",
    "launcher": repo / "experiments/launch_tpd_clean_v5_screen800_2x5090_finalizer.sh",
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

echo "TPDCLEANV5_2X_FINALIZER_PREFLIGHT_OK unit=$v5_unit"
if [[ "$v5_mode" == "--preflight" ]]; then
    exit 0
fi

if [[ -e "$v5_marker" || -L "$v5_marker" ]]; then
    "$v5_runner"
    exit $?
fi

v5_active_state="$(
    "$v5_systemctl" --user show "$v5_unit" --no-pager \
        --property=ActiveState --value 2>/dev/null || true
)"
case "$v5_active_state" in
    active|activating|deactivating|reloading)
        echo "TPDCLEANV5_2X_FINALIZER_UNIT_REUSED state=$v5_active_state unit=$v5_unit"
        exit 0
        ;;
esac

mkdir -p "$v5_result_root"
"$v5_systemd_run" --user \
    --collect \
    --unit="${v5_unit%.service}" \
    --description="SCTransNet TPD-Clean-v5 screen800 post-training finalizer" \
    --property=Restart=no \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v5_runner"
echo "TPDCLEANV5_2X_FINALIZER_UNIT_STARTED unit=$v5_unit"
