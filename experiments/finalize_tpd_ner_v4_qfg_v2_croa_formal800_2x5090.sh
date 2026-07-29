#!/usr/bin/env bash
set -Eeuo pipefail

qfg_finalize_mode="run"
if [[ "${1:-}" == "--preflight" ]]; then
    qfg_finalize_mode="preflight"
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--preflight]" >&2
    exit 2
fi

qfg_finalize_repo="${TPD_NER_V4_QFG_V2_CROA_REPO:-/home/ly/SCTransNet_main}"
qfg_finalize_python="${TPD_NER_V4_QFG_V2_CROA_PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}"
qfg_finalize_dataset_dir="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_DATASET_DIR:-$qfg_finalize_repo/datasets}"
qfg_finalize_source_lock="${TPD_NER_V4_QFG_V2_CROA_SOURCE_LOCK:-$qfg_finalize_repo/experiments/tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json}"
qfg_finalize_result_root="${TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT:-$qfg_finalize_repo/experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized}"
qfg_finalize_survival_root="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_SURVIVAL_RESULT_ROOT:-$qfg_finalize_repo/experiments/results/tpd_ner_v4_survival_exact_v1}"

qfg_finalize_default_freezer="$qfg_finalize_repo/experiments/freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock.py"
qfg_finalize_default_sweeps_runner="$qfg_finalize_repo/experiments/run_tpd_ner_v4_qfg_v2_croa_formal800_sweeps_2x5090.sh"
qfg_finalize_default_factorial="$qfg_finalize_repo/experiments/compare_tss_qfg_v2_croa_factorial.py"
qfg_finalize_default_postprocess="$qfg_finalize_repo/experiments/postprocess_tpd_ner_v4_qfg_v2_croa_formal800.py"
qfg_finalize_default_closure_freezer="$qfg_finalize_repo/experiments/freeze_tpd_ner_v4_qfg_v2_croa_posttraining_closure.py"
qfg_finalize_default_closure_lock="$qfg_finalize_repo/experiments/tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json"
qfg_finalize_default_deployer="$qfg_finalize_repo/experiments/deploy_tpd_ner_v4_qfg_v2_croa_formal800.py"
qfg_finalize_freezer="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_FREEZER:-$qfg_finalize_default_freezer}"
qfg_finalize_sweeps_runner="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_SWEEPS_RUNNER:-$qfg_finalize_default_sweeps_runner}"
qfg_finalize_factorial="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_FACTORIAL_COMPARE:-$qfg_finalize_default_factorial}"
qfg_finalize_postprocess="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_POSTPROCESS:-$qfg_finalize_default_postprocess}"
qfg_finalize_closure_freezer="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_CLOSURE_FREEZER:-$qfg_finalize_default_closure_freezer}"
qfg_finalize_closure_lock="${TPD_NER_V4_QFG_V2_CROA_POSTTRAINING_SOURCE_LOCK:-$qfg_finalize_default_closure_lock}"
qfg_finalize_deployer="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_DEPLOYER:-$qfg_finalize_default_deployer}"
qfg_finalize_completion_checker="${TPD_NER_V4_QFG_V2_CROA_FINALIZER_COMPLETION_CHECKER:-}"

qfg_finalize_a_run="$qfg_finalize_survival_root/NUDT-SIRST/tss_control/seed_42_formal800_control"
qfg_finalize_b_run="$qfg_finalize_survival_root/NUDT-SIRST/tss_on/seed_42_formal800_tss"
qfg_finalize_c_run="$qfg_finalize_result_root/NUDT-SIRST/qfg_only/seed_42_formal800_qfg_only"
qfg_finalize_d_run="$qfg_finalize_result_root/NUDT-SIRST/tss_qfg/seed_42_formal800_tss_qfg"

qfg_finalize_factorial_dir="$qfg_finalize_result_root/NUDT-SIRST/comparison_factorial_v1"
qfg_finalize_factorial_json="$qfg_finalize_factorial_dir/tss_qfg_v2_croa_factorial_seed42.json"
qfg_finalize_factorial_md="$qfg_finalize_factorial_dir/tss_qfg_v2_croa_factorial_seed42.md"
qfg_finalize_selection_dir="$qfg_finalize_result_root/NUDT-SIRST/final_selection"
qfg_finalize_selection_json="$qfg_finalize_selection_dir/tpd_ner_v4_qfg_v2_croa_formal800_final_selection.json"
qfg_finalize_selection_md="$qfg_finalize_selection_dir/tpd_ner_v4_qfg_v2_croa_formal800_final_selection.md"
qfg_finalize_deployment_dir="$qfg_finalize_result_root/NUDT-SIRST/deployment"
qfg_finalize_deployment_artifact="$qfg_finalize_deployment_dir/tpd_ner_v4_qfg_v2_croa_formal800_inference.pth.tar"
qfg_finalize_deployment_manifest="$qfg_finalize_deployment_dir/tpd_ner_v4_qfg_v2_croa_formal800_deployment_manifest.json"

qfg_finalize_abort() {
    echo "TPDNERV4QFG_FINALIZE_ABORT reason=$1" >&2
    exit 64
}

qfg_finalize_require_regular_file() {
    local qfg_finalize_path="$1"
    local qfg_finalize_label="$2"
    [[ -f "$qfg_finalize_path" && ! -L "$qfg_finalize_path" ]] \
        || qfg_finalize_abort "${qfg_finalize_label}_nonregular"
}

qfg_finalize_require_regular_dir() {
    local qfg_finalize_path="$1"
    local qfg_finalize_label="$2"
    [[ -d "$qfg_finalize_path" && ! -L "$qfg_finalize_path" ]] \
        || qfg_finalize_abort "${qfg_finalize_label}_nonregular"
}

qfg_finalize_pair_state() {
    local qfg_finalize_json="$1"
    local qfg_finalize_markdown="$2"
    local qfg_finalize_label="$3"
    local qfg_finalize_json_present="false"
    local qfg_finalize_markdown_present="false"

    if [[ -e "$qfg_finalize_json" || -L "$qfg_finalize_json" ]]; then
        qfg_finalize_json_present="true"
    fi
    if [[ -e "$qfg_finalize_markdown" || -L "$qfg_finalize_markdown" ]]; then
        qfg_finalize_markdown_present="true"
    fi
    if [[ "$qfg_finalize_json_present" == "false" \
        && "$qfg_finalize_markdown_present" == "false" ]]; then
        echo "missing"
        return 0
    fi
    if [[ "$qfg_finalize_json_present" != "$qfg_finalize_markdown_present" ]]; then
        qfg_finalize_abort "${qfg_finalize_label}_partial_pair"
    fi
    qfg_finalize_require_regular_file \
        "$qfg_finalize_json" "${qfg_finalize_label}_json"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_markdown" "${qfg_finalize_label}_markdown"
    echo "existing"
}

qfg_finalize_deployment_state() {
    local qfg_finalize_artifact_present="false"
    local qfg_finalize_manifest_present="false"
    if [[ -e "$qfg_finalize_deployment_artifact" \
        || -L "$qfg_finalize_deployment_artifact" ]]; then
        qfg_finalize_artifact_present="true"
    fi
    if [[ -e "$qfg_finalize_deployment_manifest" \
        || -L "$qfg_finalize_deployment_manifest" ]]; then
        qfg_finalize_manifest_present="true"
    fi
    if [[ "$qfg_finalize_manifest_present" == "true" \
        && "$qfg_finalize_artifact_present" != "true" ]]; then
        qfg_finalize_abort "deployment_manifest_without_artifact"
    fi
    if [[ "$qfg_finalize_artifact_present" == "true" ]]; then
        qfg_finalize_require_regular_file \
            "$qfg_finalize_deployment_artifact" "deployment_artifact"
    fi
    if [[ "$qfg_finalize_manifest_present" == "true" ]]; then
        qfg_finalize_require_regular_file \
            "$qfg_finalize_deployment_manifest" "deployment_manifest"
    fi
    if [[ "$qfg_finalize_artifact_present" == "true" \
        && "$qfg_finalize_manifest_present" == "true" ]]; then
        echo "existing"
    elif [[ "$qfg_finalize_artifact_present" == "true" ]]; then
        echo "artifact-only-recoverable"
    else
        echo "missing"
    fi
}

qfg_finalize_static_preflight() {
    qfg_finalize_require_regular_dir "$qfg_finalize_repo" "repo"
    [[ -x "$qfg_finalize_python" ]] \
        || qfg_finalize_abort "python_not_executable"
    qfg_finalize_require_regular_dir \
        "$qfg_finalize_dataset_dir" "dataset_directory"
    qfg_finalize_require_regular_dir \
        "$qfg_finalize_result_root" "result_root"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_source_lock" "optimized_source_lock"
    qfg_finalize_require_regular_file "$qfg_finalize_freezer" "freezer"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_sweeps_runner" "sweeps_runner"
    [[ -x "$qfg_finalize_sweeps_runner" ]] \
        || qfg_finalize_abort "sweeps_runner_not_executable"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_factorial" "factorial_compare"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_postprocess" "postprocess"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_closure_freezer" "posttraining_closure_freezer"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_closure_lock" "posttraining_closure_source_lock"
    qfg_finalize_require_regular_file \
        "$qfg_finalize_deployer" "deployer"
    if [[ -n "$qfg_finalize_completion_checker" ]]; then
        qfg_finalize_require_regular_file \
            "$qfg_finalize_completion_checker" "completion_checker"
        [[ -x "$qfg_finalize_completion_checker" ]] \
            || qfg_finalize_abort "completion_checker_not_executable"
    fi

    for qfg_finalize_run_spec in \
        "$qfg_finalize_a_run:a_run" \
        "$qfg_finalize_b_run:b_run" \
        "$qfg_finalize_c_run:c_run" \
        "$qfg_finalize_d_run:d_run"
    do
        qfg_finalize_require_regular_dir \
            "${qfg_finalize_run_spec%%:*}" \
            "${qfg_finalize_run_spec##*:}"
    done
    for qfg_finalize_reference_sweep in \
        "$qfg_finalize_a_run/pd_fa_sweep_best.pth.json" \
        "$qfg_finalize_a_run/pd_fa_sweep_best_miou.pth.json" \
        "$qfg_finalize_b_run/pd_fa_sweep_best.pth.json" \
        "$qfg_finalize_b_run/pd_fa_sweep_best_miou.pth.json"
    do
        qfg_finalize_require_regular_file \
            "$qfg_finalize_reference_sweep" "reference_sweep"
    done
    for qfg_finalize_candidate_run in \
        "$qfg_finalize_c_run" \
        "$qfg_finalize_d_run"
    do
        qfg_finalize_require_regular_file \
            "$qfg_finalize_candidate_run/summary.json" \
            "candidate_summary"
        qfg_finalize_require_regular_dir \
            "$qfg_finalize_candidate_run/exact_journal" \
            "candidate_exact_journal"
        qfg_finalize_require_regular_file \
            "$qfg_finalize_candidate_run/exact_journal/active.json" \
            "candidate_active_marker"
    done
}

qfg_finalize_verify_source_lock() {
    "$qfg_finalize_python" "$qfg_finalize_freezer" \
        --verify \
        --dataset-dir "$qfg_finalize_dataset_dir" \
        --output "$qfg_finalize_source_lock"
    echo "TPDNERV4QFG_FINALIZE_SOURCE_LOCK_VERIFIED live=true"
}

qfg_finalize_verify_closure_lock() {
    "$qfg_finalize_python" "$qfg_finalize_closure_freezer" \
        --verify \
        --output "$qfg_finalize_closure_lock"
    echo "TPDNERV4QFG_FINALIZE_POSTTRAINING_LOCK_VERIFIED live=true lock=$qfg_finalize_closure_lock"
}

qfg_finalize_verify_completion() {
    if [[ -n "$qfg_finalize_completion_checker" ]]; then
        "$qfg_finalize_completion_checker" \
            "$qfg_finalize_c_run" qfg_only \
            "$qfg_finalize_d_run" tss_qfg \
            "$qfg_finalize_source_lock"
        return
    fi

    "$qfg_finalize_python" - completion \
        "$qfg_finalize_repo" \
        "$qfg_finalize_c_run" qfg_only \
        "$qfg_finalize_d_run" tss_qfg \
        "$qfg_finalize_source_lock" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


if sys.argv[1] != "completion":
    raise SystemExit("invalid completion-check mode")
repo = Path(sys.argv[2]).resolve()
c_run = Path(sys.argv[3]).resolve()
c_variant = sys.argv[4]
d_run = Path(sys.argv[5]).resolve()
d_variant = sys.argv[6]
source_lock = Path(sys.argv[7]).resolve()
sys.path.insert(0, str(repo))

from experiments import tpd_exact_epoch_journal as epoch_journal
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as exact


def fail(message: str) -> None:
    raise SystemExit(message)


def regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            regular(path, label).read_text(encoding="utf-8"),
            parse_constant=lambda token: fail(
                f"{label} contains non-finite constant {token}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is invalid JSON: {error}")
    if not isinstance(payload, dict):
        fail(f"{label} must contain one JSON object")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with regular(path, "SHA-256 input").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        fail(
            f"{label} differs: expected={expected!r} "
            f"observed={observed!r}"
        )


expected_trainer = (
    repo / "experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py"
).resolve()
if Path(exact.__file__).resolve() != expected_trainer:
    fail("completion checker imported the wrong exact trainer")
if exact.FORMAL_EPOCHS != 800:
    fail("formal completion contract is not exactly 800 epochs")
source_lock_sha256 = sha256(source_lock)

for run_dir, variant, run_tag in (
    (c_run, c_variant, "formal800_qfg_only"),
    (d_run, d_variant, "formal800_tss_qfg"),
):
    if variant not in exact.SUPPORTED_CANDIDATE_VARIANTS:
        fail(f"unsupported formal candidate: {variant}")
    summary_path = regular(run_dir / "summary.json", f"{variant} summary")
    metrics_path = regular(
        run_dir / exact.exact_runner.METRICS_FILENAME,
        f"{variant} metrics",
    )
    journal_root = run_dir / exact.exact_runner.JOURNAL_DIRECTORY
    if journal_root.is_symlink() or not journal_root.is_dir():
        fail(f"{variant} exact journal must be a regular directory")
    regular(
        journal_root / epoch_journal.MARKER_FILENAME,
        f"{variant} active marker",
    )

    active = epoch_journal.ExactEpochJournal(journal_root).load_active()
    if active is None or active.epoch != exact.FORMAL_EPOCHS:
        fail(f"{variant} active journal is not committed at epoch 800")
    root_metrics = metrics_path.read_bytes()
    if root_metrics != active.metrics_path.read_bytes():
        fail(f"{variant} derived metrics differ from the active journal")
    if hashlib.sha256(root_metrics).hexdigest() != (
        active.metrics_boundary["metrics_sha256"]
    ):
        fail(f"{variant} active metrics digest differs")
    raw_lines = root_metrics.splitlines()
    if len(raw_lines) != 800 or any(not line.strip() for line in raw_lines):
        fail(f"{variant} metrics must contain exactly 800 nonblank rows")

    events = exact._load_complete_events(metrics_path, exact.FORMAL_EPOCHS)
    if any(event.get("variant") != variant for event in events):
        fail(f"{variant} metrics contain a different candidate variant")
    policy = exact.exact_runner.pd_miou_selection_policy(
        stored_metrics=exact.STORED_VALIDATION_METRICS,
    )
    try:
        selection = policy.recompute(events, require_flags=True)
    except exact.exact_runner.ExactRunnerError as error:
        fail(f"{variant} selection history is invalid: {error}")

    summary = load_json(summary_path, f"{variant} summary")
    candidate = exact.candidate_contract(variant)
    for label, observed, expected in (
        ("schema", summary.get("schema"), exact.COMPLETION_SUMMARY_SCHEMA),
        ("status", summary.get("status"), "complete"),
        ("variant", summary.get("variant"), variant),
        ("candidate variant", summary.get("candidate_variant"), variant),
        ("QFG variant", summary.get("qfg_variant"), candidate["qfg_variant"]),
        ("TSS variant", summary.get("tss_variant"), candidate["tss_variant"]),
        ("dataset", summary.get("dataset"), "NUDT-SIRST"),
        ("training seed", summary.get("seed"), 42),
        ("split seed", summary.get("split_seed"), 20260722),
        ("formal contract", summary.get("formal_contract"), exact.formal_contract()),
        (
            "stored validation metrics",
            summary.get("stored_validation_metrics"),
            list(exact.STORED_VALIDATION_METRICS),
        ),
        (
            "selection source",
            summary.get("selection_source"),
            "internal_validation_only",
        ),
        (
            "official test access",
            summary.get("official_test_accessed"),
            False,
        ),
        ("best epoch", summary.get("best_epoch"), selection["primary"]["epoch"]),
        (
            "best Pd epoch",
            summary.get("best_pd_epoch"),
            selection["primary"]["epoch"],
        ),
        (
            "best mIoU epoch",
            summary.get("best_miou_epoch"),
            selection["secondary"]["epoch"],
        ),
        (
            "best metrics",
            summary.get("best_validation_metrics"),
            selection["primary"]["metrics"],
        ),
        (
            "best Pd metrics",
            summary.get("best_pd_validation_metrics"),
            selection["primary"]["metrics"],
        ),
        (
            "best mIoU metrics",
            summary.get("best_miou_validation_metrics"),
            selection["secondary"]["metrics"],
        ),
    ):
        equal(f"{variant} summary {label}", observed, expected)

    try:
        identity = exact.require_qfg_run_identity(
            summary.get("run_identity"),
            label=f"{variant} completion summary",
            expected_variant=variant,
        )
    except (TypeError, ValueError) as error:
        fail(f"{variant} run identity is invalid: {error}")
    equal(f"{variant} identity dataset", identity.get("dataset"), "NUDT-SIRST")
    equal(f"{variant} identity seed", identity.get("seed"), 42)
    equal(f"{variant} identity split seed", identity.get("split_seed"), 20260722)
    expected_run_id = (
        f"{exact.RUN_ID_PREFIX}NUDT-SIRST:{variant}:"
        f"seed-42:split-20260722:{run_tag}"
    )
    equal(f"{variant} identity run_id", identity.get("run_id"), expected_run_id)
    source_locks = identity.get("source_locks")
    equal(f"{variant} summary source locks", summary.get("source_locks"), source_locks)
    if not isinstance(source_locks, dict):
        fail(f"{variant} source-lock binding is missing")
    equal(
        f"{variant} optimized source-lock digest",
        source_locks.get(exact.SOURCE_LOCK_KEY),
        source_lock_sha256,
    )
    print(
        "TPDNERV4QFG_FINALIZE_RUN_COMPLETE"
        f" variant={variant}"
        " active_epoch=800 summary=complete"
        f" active_marker_sha256={active.marker_sha256}",
        flush=True,
    )
PY
}

qfg_finalize_factorial_preflight() {
    "$qfg_finalize_python" "$qfg_finalize_factorial" \
        --preflight \
        --a-run-dir "$qfg_finalize_a_run" \
        --b-run-dir "$qfg_finalize_b_run" \
        --c-run-dir "$qfg_finalize_c_run" \
        --d-run-dir "$qfg_finalize_d_run" \
        --json-output "$qfg_finalize_factorial_json" \
        --markdown-output "$qfg_finalize_factorial_md"
}

qfg_finalize_factorial_aggregate() {
    "$qfg_finalize_python" "$qfg_finalize_factorial" \
        --aggregate \
        --a-run-dir "$qfg_finalize_a_run" \
        --b-run-dir "$qfg_finalize_b_run" \
        --c-run-dir "$qfg_finalize_c_run" \
        --d-run-dir "$qfg_finalize_d_run" \
        --json-output "$qfg_finalize_factorial_json" \
        --markdown-output "$qfg_finalize_factorial_md"
}

qfg_finalize_validate_terminal_reports() {
    local qfg_finalize_required_state="${1:-allow-missing}"
    "$qfg_finalize_python" - terminal-reports \
        "$qfg_finalize_repo" \
        "$qfg_finalize_factorial" \
        "$qfg_finalize_postprocess" \
        "$qfg_finalize_a_run" \
        "$qfg_finalize_b_run" \
        "$qfg_finalize_c_run" \
        "$qfg_finalize_d_run" \
        "$qfg_finalize_factorial_json" \
        "$qfg_finalize_factorial_md" \
        "$qfg_finalize_selection_json" \
        "$qfg_finalize_selection_md" \
        "$qfg_finalize_required_state" <<'PY'
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Mapping


if sys.argv[1] != "terminal-reports":
    raise SystemExit("invalid terminal-report check mode")
repo = Path(sys.argv[2]).resolve()
factorial_path = Path(sys.argv[3]).resolve()
postprocess_path = Path(sys.argv[4]).resolve()
run_directories = {
    "A": Path(sys.argv[5]).resolve(),
    "B": Path(sys.argv[6]).resolve(),
    "C": Path(sys.argv[7]).resolve(),
    "D": Path(sys.argv[8]).resolve(),
}
factorial_json = Path(sys.argv[9]).resolve()
factorial_markdown = Path(sys.argv[10]).resolve()
selection_json = Path(sys.argv[11]).resolve()
selection_markdown = Path(sys.argv[12]).resolve()
required_state = sys.argv[13]
if required_state not in {"allow-missing", "require-complete"}:
    raise SystemExit("invalid terminal-report required state")
sys.path.insert(0, str(repo))


def load_module(path: Path, name: str) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"terminal report source is not regular: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import terminal report source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pair_state(json_path: Path, markdown_path: Path, label: str) -> str:
    json_present = json_path.exists() or json_path.is_symlink()
    markdown_present = markdown_path.exists() or markdown_path.is_symlink()
    if not json_present and not markdown_present:
        return "missing"
    if json_present != markdown_present:
        raise SystemExit(f"{label} output pair is partial")
    for path in (json_path, markdown_path):
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"{label} output is not regular: {path}")
    return "existing"


def expected_json_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def verify_pair(
    *,
    label: str,
    state: str,
    report: Mapping[str, Any],
    render: Callable[[Mapping[str, Any]], str],
    json_path: Path,
    markdown_path: Path,
) -> None:
    if state == "missing":
        return
    expected_json = expected_json_bytes(report)
    expected_markdown = render(report).encode("utf-8")
    if json_path.read_bytes() != expected_json:
        raise SystemExit(f"existing {label} JSON conflicts with live inputs")
    if markdown_path.read_bytes() != expected_markdown:
        raise SystemExit(f"existing {label} Markdown conflicts with live inputs")


factorial = load_module(
    factorial_path,
    "_qfg_v2_croa_finalizer_factorial",
)
postprocess = load_module(
    postprocess_path,
    "_qfg_v2_croa_finalizer_postprocess",
)
factorial_records = factorial.collect_validated_sweeps(run_directories)
factorial_report = factorial.build_factorial_report(factorial_records)
selection_report = postprocess.build_formal_report()

factorial_state = pair_state(
    factorial_json,
    factorial_markdown,
    "factorial",
)
selection_state = pair_state(
    selection_json,
    selection_markdown,
    "final selection",
)
if selection_state == "existing" and factorial_state != "existing":
    raise SystemExit(
        "final-selection outputs exist without the preceding factorial outputs"
    )
verify_pair(
    label="factorial",
    state=factorial_state,
    report=factorial_report,
    render=factorial.render_markdown,
    json_path=factorial_json,
    markdown_path=factorial_markdown,
)
verify_pair(
    label="final selection",
    state=selection_state,
    report=selection_report,
    render=postprocess.render_markdown,
    json_path=selection_json,
    markdown_path=selection_markdown,
)
if required_state == "require-complete" and (
    factorial_state != "existing" or selection_state != "existing"
):
    raise SystemExit("terminal report closure is incomplete")
print(
    "TPDNERV4QFG_FINALIZE_TERMINAL_PREFLIGHT"
    f" factorial={factorial_state}"
    f" final_selection={selection_state}"
    " would_write=false",
    flush=True,
)
PY
}

qfg_finalize_static_preflight

if [[ "$qfg_finalize_mode" == "run" ]]; then
    qfg_finalize_lock_dir="$qfg_finalize_result_root/.finalization_locks"
    if [[ -L "$qfg_finalize_lock_dir" \
        || ( -e "$qfg_finalize_lock_dir" && ! -d "$qfg_finalize_lock_dir" ) ]]; then
        qfg_finalize_abort "lock_directory_nonregular"
    fi
    mkdir -p "$qfg_finalize_lock_dir"
    qfg_finalize_lock="$qfg_finalize_lock_dir/formal800_2x5090_finalize.lock"
    if [[ -L "$qfg_finalize_lock" \
        || ( -e "$qfg_finalize_lock" && ! -f "$qfg_finalize_lock" ) ]]; then
        qfg_finalize_abort "finalization_claim_nonregular"
    fi
    exec 9>>"$qfg_finalize_lock"
    [[ -f "$qfg_finalize_lock" && ! -L "$qfg_finalize_lock" ]] \
        || qfg_finalize_abort "finalization_claim_nonregular_after_open"
    if ! flock -n 9; then
        echo "TPDNERV4QFG_FINALIZE_RETRY reason=finalization_claim_held" >&2
        exit 75
    fi
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export TPD_NER_V4_QFG_V2_CROA_REPO="$qfg_finalize_repo"
export TPD_NER_V4_QFG_V2_CROA_PYTHON="$qfg_finalize_python"
export TPD_NER_V4_QFG_V2_CROA_SOURCE_LOCK="$qfg_finalize_source_lock"
export TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT="$qfg_finalize_result_root"
export TPD_NER_V4_QFG_V2_CROA_POSTTRAINING_SOURCE_LOCK="$qfg_finalize_closure_lock"

cd "$qfg_finalize_repo"
qfg_finalize_verify_source_lock
qfg_finalize_verify_closure_lock
qfg_finalize_verify_completion

qfg_finalize_factorial_state="$(
    qfg_finalize_pair_state \
        "$qfg_finalize_factorial_json" \
        "$qfg_finalize_factorial_md" \
        "factorial"
)"
qfg_finalize_selection_state="$(
    qfg_finalize_pair_state \
        "$qfg_finalize_selection_json" \
        "$qfg_finalize_selection_md" \
        "final_selection"
)"
qfg_finalize_deployment_state="$(
    qfg_finalize_deployment_state
)"
if [[ "$qfg_finalize_selection_state" == "existing" \
    && "$qfg_finalize_factorial_state" != "existing" ]]; then
    qfg_finalize_abort "final_selection_without_factorial"
fi

if [[ "$qfg_finalize_mode" == "preflight" ]]; then
    "$qfg_finalize_sweeps_runner" --preflight
    qfg_finalize_factorial_preflight
    qfg_finalize_validate_terminal_reports allow-missing
    if [[ "$qfg_finalize_selection_state" == "existing" ]]; then
        "$qfg_finalize_python" "$qfg_finalize_deployer" \
            --preflight \
            --selection "$qfg_finalize_selection_json" \
            --artifact "$qfg_finalize_deployment_artifact" \
            --manifest "$qfg_finalize_deployment_manifest" \
            --closure-source-lock "$qfg_finalize_closure_lock"
    fi
    echo "TPDNERV4QFG_FINALIZE_PREFLIGHT_COMPLETE writes_performed=false qfg_only_gpu=2 tss_qfg_gpu=3 deployment=$qfg_finalize_deployment_state"
    exit 0
fi

if [[ "$qfg_finalize_factorial_state" == "existing" \
    && "$qfg_finalize_selection_state" == "existing" \
    && "$qfg_finalize_deployment_state" == "existing" ]]; then
    qfg_finalize_factorial_preflight
    qfg_finalize_validate_terminal_reports require-complete
    "$qfg_finalize_python" "$qfg_finalize_deployer" \
        --verify \
        --selection "$qfg_finalize_selection_json" \
        --artifact "$qfg_finalize_deployment_artifact" \
        --manifest "$qfg_finalize_deployment_manifest" \
        --closure-source-lock "$qfg_finalize_closure_lock"
    echo "TPDNERV4QFG_FINALIZE_COMPLETE idempotent=true producers_run=false deployment_verified=true qfg_only_gpu=2 tss_qfg_gpu=3"
    exit 0
fi

# The existing runner owns the only CUDA work.  It binds QFG-only to physical
# GPU 2 and TSS+QFG to physical GPU 3, launches immediately, and never polls
# utilization or waits for either device to become idle.
if [[ "$qfg_finalize_factorial_state" != "existing" ]]; then
    "$qfg_finalize_sweeps_runner"
fi
qfg_finalize_factorial_preflight

# Recheck all four terminal paths after sweep production.  Then build both
# reports in memory and validate every existing byte before publishing either
# report.  A failed precheck therefore cannot create a terminal artifact.
qfg_finalize_factorial_state="$(
    qfg_finalize_pair_state \
        "$qfg_finalize_factorial_json" \
        "$qfg_finalize_factorial_md" \
        "factorial"
)"
qfg_finalize_selection_state="$(
    qfg_finalize_pair_state \
        "$qfg_finalize_selection_json" \
        "$qfg_finalize_selection_md" \
        "final_selection"
)"
if [[ "$qfg_finalize_selection_state" == "existing" \
    && "$qfg_finalize_factorial_state" != "existing" ]]; then
    qfg_finalize_abort "final_selection_without_factorial"
fi
qfg_finalize_validate_terminal_reports allow-missing

if [[ "$qfg_finalize_factorial_state" == "missing" ]]; then
    qfg_finalize_factorial_aggregate
    qfg_finalize_factorial_state="$(
        qfg_finalize_pair_state \
            "$qfg_finalize_factorial_json" \
            "$qfg_finalize_factorial_md" \
            "factorial"
    )"
    [[ "$qfg_finalize_factorial_state" == "existing" ]] \
        || qfg_finalize_abort "factorial_publish_incomplete"
fi

if [[ "$qfg_finalize_selection_state" == "missing" ]]; then
    "$qfg_finalize_python" "$qfg_finalize_postprocess" \
        --json-output "$qfg_finalize_selection_json" \
        --markdown-output "$qfg_finalize_selection_md"
    qfg_finalize_selection_state="$(
        qfg_finalize_pair_state \
            "$qfg_finalize_selection_json" \
            "$qfg_finalize_selection_md" \
            "final_selection"
    )"
    [[ "$qfg_finalize_selection_state" == "existing" ]] \
        || qfg_finalize_abort "final_selection_publish_incomplete"
fi

qfg_finalize_validate_terminal_reports require-complete
"$qfg_finalize_python" "$qfg_finalize_deployer" \
    --selection "$qfg_finalize_selection_json" \
    --artifact "$qfg_finalize_deployment_artifact" \
    --manifest "$qfg_finalize_deployment_manifest" \
    --closure-source-lock "$qfg_finalize_closure_lock"
"$qfg_finalize_python" "$qfg_finalize_deployer" \
    --verify \
    --selection "$qfg_finalize_selection_json" \
    --artifact "$qfg_finalize_deployment_artifact" \
    --manifest "$qfg_finalize_deployment_manifest" \
    --closure-source-lock "$qfg_finalize_closure_lock"
echo "TPDNERV4QFG_FINALIZE_COMPLETE idempotent=false producers_run=true deployment_verified=true qfg_only_gpu=2 tss_qfg_gpu=3"
