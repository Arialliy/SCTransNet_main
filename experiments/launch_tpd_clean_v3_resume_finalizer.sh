#!/usr/bin/env bash
set -euo pipefail

v3_repo="${TPDCLEANV3_RESUME_FINALIZER_REPO:-/home/ly/SCTransNet_main}"
v3_runner="${TPDCLEANV3_RESUME_FINALIZER_RUNNER:-$v3_repo/experiments/run_tpd_clean_v3_resume_finalizer.sh}"
v3_summarizer="${TPDCLEANV3_RESUME_FINALIZER_SUMMARIZER:-$v3_repo/experiments/summarize_tpd_clean_v3_screen800.py}"
v3_validator="${TPDCLEANV3_RESUME_FINALIZER_VALIDATOR:-$v3_repo/experiments/validate_tpd_clean_v3_resume_completion.py}"
v3_python="${TPDCLEANV3_RESUME_FINALIZER_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
v3_postprocess_lock="${TPDCLEANV3_RESUME_FINALIZER_POSTPROCESS_LOCK:-$v3_repo/experiments/tpd_clean_v3_resume_postprocess_source_lock.json}"
v3_candidate_root="${TPDCLEANV3_RESUME_FINALIZER_CANDIDATE_ROOT:-$v3_repo/experiments/results/tpd_clean_v3_screen800_4x5090_v1}"
v3_systemctl="${TPDCLEANV3_RESUME_FINALIZER_SYSTEMCTL:-systemctl}"
v3_systemd_run="${TPDCLEANV3_RESUME_FINALIZER_SYSTEMD_RUN:-systemd-run}"
v3_unit="sctransnet-tpd-clean-v3-resume-finalizer.service"
v3_marker="$v3_candidate_root/NUDT-SIRST/comparison/RESUME_COMPLETE.sha256"

v3_mode="${1:-run}"
if [[ "$v3_mode" != "run" && "$v3_mode" != "--preflight" ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

cd "$v3_repo"
[[ -x "$v3_runner" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_LAUNCH_ABORT reason=runner_not_executable path=$v3_runner" >&2
    exit 1
}
[[ -f "$v3_summarizer" && ! -L "$v3_summarizer" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_LAUNCH_ABORT reason=missing_summarizer path=$v3_summarizer" >&2
    exit 1
}
[[ -f "$v3_validator" && ! -L "$v3_validator" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_LAUNCH_ABORT reason=missing_validator path=$v3_validator" >&2
    exit 1
}
[[ -x "$v3_python" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_LAUNCH_ABORT reason=python_not_executable path=$v3_python" >&2
    exit 1
}
[[ -f "$v3_postprocess_lock" && ! -L "$v3_postprocess_lock" ]] || {
    echo "TPDCLEANV3_RESUME_FINALIZER_LAUNCH_ABORT reason=missing_postprocess_lock path=$v3_postprocess_lock" >&2
    exit 1
}
command -v "$v3_systemctl" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_RESUME_FINALIZER_LAUNCH_ABORT reason=missing_systemctl command=$v3_systemctl" >&2
    exit 1
}
command -v "$v3_systemd_run" >/dev/null 2>&1 || {
    echo "TPDCLEANV3_RESUME_FINALIZER_LAUNCH_ABORT reason=missing_systemd_run command=$v3_systemd_run" >&2
    exit 1
}
bash -n "$v3_runner"

"$v3_python" - \
    "$v3_repo" \
    "$v3_postprocess_lock" \
    "$v3_runner" \
    "$v3_summarizer" \
    "$v3_validator" \
    "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve(strict=True)
lock_path = pathlib.Path(sys.argv[2]).resolve(strict=True)
runner = pathlib.Path(sys.argv[3]).resolve(strict=True)
summarizer = pathlib.Path(sys.argv[4]).resolve(strict=True)
validator = pathlib.Path(sys.argv[5]).resolve(strict=True)
launcher = pathlib.Path(sys.argv[6]).resolve(strict=True)
canonical = {
    "postprocess lock": repo / "experiments/tpd_clean_v3_resume_postprocess_source_lock.json",
    "runner": repo / "experiments/run_tpd_clean_v3_resume_finalizer.sh",
    "summarizer": repo / "experiments/summarize_tpd_clean_v3_screen800.py",
    "validator": repo / "experiments/validate_tpd_clean_v3_resume_completion.py",
    "launcher": repo / "experiments/launch_tpd_clean_v3_resume_finalizer.sh",
}
observed = {
    "postprocess lock": lock_path,
    "runner": runner,
    "summarizer": summarizer,
    "validator": validator,
    "launcher": launcher,
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

echo "TPDCLEANV3_RESUME_FINALIZER_PREFLIGHT_OK unit=$v3_unit"
if [[ "$v3_mode" == "--preflight" ]]; then
    exit 0
fi

if [[ -f "$v3_marker" && ! -L "$v3_marker" ]]; then
    "$v3_runner"
    exit $?
fi

v3_active_state="$(
    "$v3_systemctl" --user show "$v3_unit" --no-pager \
        --property=ActiveState --value 2>/dev/null || true
)"
case "$v3_active_state" in
    active|activating|deactivating|reloading)
        echo "TPDCLEANV3_RESUME_FINALIZER_UNIT_REUSED state=$v3_active_state unit=$v3_unit"
        exit 0
        ;;
esac

mkdir -p "$v3_candidate_root/resume_2x5090_v1"
"$v3_systemd_run" --user \
    --collect \
    --unit="${v3_unit%.service}" \
    --description="SCTransNet TPD-Clean-v3 resumed-run final audit" \
    --property=Restart=no \
    --property=TimeoutStopSec=120 \
    /usr/bin/bash "$v3_runner"
echo "TPDCLEANV3_RESUME_FINALIZER_UNIT_STARTED unit=$v3_unit"
