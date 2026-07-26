#!/usr/bin/env bash
set -euo pipefail

v3_repo="${TPDCLEANV3_2X_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v3_runner="${TPDCLEANV3_2X_FINALIZER_RUNNER:-$v3_repo/experiments/run_tpd_clean_v3_screen800_2x5090_finalizer.sh}"
v3_summarizer="${TPDCLEANV3_2X_FINALIZER_SUMMARIZER:-$v3_repo/experiments/summarize_tpd_clean_v3_screen800_2x5090.py}"
v3_completion_validator="${TPDCLEANV3_2X_FINALIZER_COMPLETION_VALIDATOR:-$v3_repo/experiments/validate_tpd_clean_v3_2x_completion.py}"
v3_python="${TPDCLEANV3_2X_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v3_postprocess_lock="${TPDCLEANV3_2X_FINALIZER_POSTPROCESS_LOCK:-$v3_repo/experiments/tpd_clean_v3_2x_postprocess_source_lock.json}"
v3_result_root="${TPDCLEANV3_2X_FINALIZER_RESULT_ROOT:-$v3_repo/experiments/results/tpd_clean_v3_screen800_2x5090_v1}"
v3_systemctl="${TPDCLEANV3_2X_FINALIZER_SYSTEMCTL:-systemctl}"
v3_systemd_run="${TPDCLEANV3_2X_FINALIZER_SYSTEMD_RUN:-systemd-run}"
v3_unit="sctransnet-tpd-clean-v3-2x-screen800-finalizer.service"
v3_marker="$v3_result_root/NUDT-SIRST/comparison/COMPLETE.sha256"

v3_mode="${1:-run}"
if [[ "$v3_mode" != "run" && "$v3_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$v3_repo"
[[ -x "$v3_runner" ]] || {
    echo "TPDCLEANV3_2X_FINALIZER_LAUNCH_ABORT reason=runner_not_executable path=$v3_runner" >&2
    exit 1
}
[[ -f "$v3_summarizer" && ! -L "$v3_summarizer" ]] || {
    echo "TPDCLEANV3_2X_FINALIZER_LAUNCH_ABORT reason=missing_summarizer path=$v3_summarizer" >&2
    exit 1
}
[[ -f "$v3_completion_validator" && ! -L "$v3_completion_validator" ]] || {
    echo "TPDCLEANV3_2X_FINALIZER_LAUNCH_ABORT reason=missing_completion_validator path=$v3_completion_validator" >&2
    exit 1
}
[[ -x "$v3_python" ]] || {
    echo "TPDCLEANV3_2X_FINALIZER_LAUNCH_ABORT reason=python_not_executable path=$v3_python" >&2
    exit 1
}
[[ -f "$v3_postprocess_lock" && ! -L "$v3_postprocess_lock" ]] || {
    echo "TPDCLEANV3_2X_FINALIZER_LAUNCH_ABORT reason=missing_postprocess_lock path=$v3_postprocess_lock" >&2
    exit 1
}
command -v "$v3_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_2X_FINALIZER_LAUNCH_ABORT reason=missing_systemctl command=$v3_systemctl" >&2
    exit 1
}
command -v "$v3_systemd_run" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_2X_FINALIZER_LAUNCH_ABORT reason=missing_systemd_run command=$v3_systemd_run" >&2
    exit 1
}
bash -n "$v3_runner"
"$v3_python" - \
    "$v3_repo" \
    "$v3_postprocess_lock" \
    "$v3_runner" \
    "$v3_summarizer" \
    "$v3_completion_validator" \
    "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
lock_path = pathlib.Path(sys.argv[2]).resolve(strict=True)
runner_path = pathlib.Path(sys.argv[3]).resolve(strict=True)
summarizer_path = pathlib.Path(sys.argv[4]).resolve(strict=True)
validator_path = pathlib.Path(sys.argv[5]).resolve(strict=True)
launcher_path = pathlib.Path(sys.argv[6]).resolve(strict=True)
canonical = {
    "postprocess lock": repo
    / "experiments/tpd_clean_v3_2x_postprocess_source_lock.json",
    "runner": repo
    / "experiments/run_tpd_clean_v3_screen800_2x5090_finalizer.sh",
    "summarizer": repo
    / "experiments/summarize_tpd_clean_v3_screen800_2x5090.py",
    "completion validator": repo
    / "experiments/validate_tpd_clean_v3_2x_completion.py",
    "launcher": repo
    / "experiments/launch_tpd_clean_v3_screen800_2x5090_finalizer.sh",
}
observed = {
    "postprocess lock": lock_path,
    "runner": runner_path,
    "summarizer": summarizer_path,
    "completion validator": validator_path,
    "launcher": launcher_path,
}
for label, expected in canonical.items():
    if observed[label] != expected.resolve(strict=True):
        raise SystemExit(
            f"non-canonical Clean-v3 {label}: "
            f"expected={expected} observed={observed[label]}"
        )
payload = json.loads(lock_path.read_text(encoding="utf-8"))
if payload.get("schema") != "sctransnet_tpd_clean_v3_2x_postprocess_source_lock_v1":
    raise SystemExit("unexpected Clean-v3 postprocess source-lock schema")
entries = payload.get("source_sha256")
if not isinstance(entries, dict) or not entries:
    raise SystemExit("Clean-v3 postprocess source lock has no entries")
required = {
    "experiments/summarize_tpd_clean_v3_screen800_2x5090.py",
    "experiments/validate_tpd_clean_v3_2x_completion.py",
    "experiments/run_tpd_clean_v3_screen800_2x5090_finalizer.sh",
    "experiments/launch_tpd_clean_v3_screen800_2x5090_finalizer.sh",
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
    "TPDCLEANV3_2X_POSTPROCESS_SOURCES_OK"
    f" checked_entries={len(entries)}",
    flush=True,
)
PY

echo "TPDCLEANV3_2X_FINALIZER_PREFLIGHT_OK poll_seconds=60 unit=$v3_unit"
if [[ "$v3_mode" == "--preflight" ]]; then
    exit 0
fi

if [[ -f "$v3_marker" && ! -L "$v3_marker" ]]; then
    # The runner performs the full hash and summary-schema validation.
    "$v3_runner"
    exit $?
fi

v3_active_state="$(
    "$v3_systemctl" --user show "$v3_unit" --no-pager \
        --property=ActiveState --value 2>/dev/null || true
)"
case "$v3_active_state" in
    active|activating|deactivating|reloading)
        echo "TPDCLEANV3_2X_FINALIZER_UNIT_REUSED state=$v3_active_state unit=$v3_unit"
        exit 0
        ;;
esac

mkdir -p "$v3_result_root"
"$v3_systemd_run" --user \
    --collect \
    --unit="${v3_unit%.service}" \
    --description="SCTransNet TPD-Clean-v3 screen800 post-training finalizer" \
    --property=Restart=no \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v3_runner"
echo "TPDCLEANV3_2X_FINALIZER_UNIT_STARTED unit=$v3_unit"
