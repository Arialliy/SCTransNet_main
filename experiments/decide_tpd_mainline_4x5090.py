#!/usr/bin/env python3
"""Fail-closed seed-42 TPD mainline screening decision.

This tool is intentionally separate from training and from the frozen
post-processing pipeline.  It consumes the committed comparison,
extended-integrity audit, and Pd--Fa aggregate, then byte-hashes every source
artifact to which those sealed records refer.  It also re-runs the frozen
extended auditor under the current Python executable and requires its output to
be byte-identical to the sealed audit.  That isolated auditor loads checkpoints
for strict/finite validation but performs no inference.  The official-training
fingerprint is independently recomputed; no official-test path is opened.

The decision rule implemented here is a post-hoc conservative operational
policy.  It was not natively bound into the four-RTX-5090 launch and cannot
establish a paper-level mechanism or stability claim.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("original", "progressive", "spd", "tpd")
MANIFEST_VARIANTS = ("original", "progressive", "tpd", "spd")
STRONG_CONTROLS = ("progressive", "spd")
EXPECTED_EPOCHS = 800
EXPECTED_SEED = 42
EXPECTED_DATASET = "NUDT-SIRST"
EXPECTED_RUN_NAME = "seed_42_formal800_pd_fp32_4x5090_v1"
EXPECTED_SPLIT_SEED = 20260722
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_VALIDATION_SPLIT_SHA256 = (
    "86247e5970f93224c64005e1ac7f3a933bafb37baf279ab71fce5670ae925e06"
)
EXPECTED_SPLIT_ARTIFACT_SHA256 = (
    "27c3b4a30c680af1c16493f723ce9713cb7e6987dcbc82e72fe1331cff12cd6b"
)
EXPECTED_TRAINING_DATA_SHA256 = (
    "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
)
EXPECTED_SHARED_INITIALIZATION_SHA256 = (
    "ae25925e8fffd9afe9fac1805389e80437f0d773ae744c979349a68886d81558"
)
EXPECTED_EVALUATOR_SHA256 = (
    "0224ab44dc346ebdbd4cb4775c493bd6eecdc877019832dea0f16e59ab353537"
)
EXPECTED_AGGREGATOR_SHA256 = (
    "482903040cbe9a58f17444eee45aeb67c6763a6aef99bd54892f47be5e21b42e"
)
LEGACY_SCIENTIFIC_GATE_PATH = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "tpd_pe_formal800_v1"
    / "PREREGISTRATION.md"
)
LEGACY_SCIENTIFIC_GATE_SHA256 = (
    "3b266a6eb9c17e43ed42c2e078ad05842dafdeb6cedebfd2a105833a07966ede"
)
MAINLINE_CONTEXT_PATH = REPO_ROOT / "TPD_SCTransNet_主线修订版.md"
MAINLINE_CONTEXT_SHA256 = (
    "356b76143c87a60b46e2d572b2c945c702cab56314836b5c5a4f25af926dc49c"
)
FORMAL_FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
FORMAL_FA_BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FORMAL_FA_BUDGETS)
OUTPUT_STEM = "mainline_decision_seed42"
RUNTIME_STATE_NAME = "systemd_state_20260723_173022_CST.json"
EXPECTED_RUNTIME_STATE_SHA256 = (
    "7178a5157a4eae5641d410f9346bfe7ff847a90fca68a567e311cdc2baae23e2"
)
GPU_UUIDS = {
    "original": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "progressive": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "tpd": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "spd": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
INVOCATION_IDS = {
    "original": "0e533dd2d1444feb9ba1ed1e3d42135c",
    "progressive": "0803e8ed5f974f909c54d6daaddc23bb",
    "tpd": "669e46b8bd2245db92788afa62066bd3",
    "spd": "75468708b771457daac1cf0586f7c333",
}

COUNT_KEYS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
POINT_KEYS = (
    "val_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    *COUNT_KEYS,
    "threshold",
)
OPERATING_POINT_RULE = (
    "maximum Pd under Fa budget",
    "minimum actual Fa on Pd ties",
    "maximum tiny-Pd",
    "maximum mIoU",
    "minimum distance from threshold 0.5",
)
AGGREGATE_INTEGRITY_FLAGS = {
    "four_sweeps_present",
    "source_sweeps_manifest_verified",
    "source_complete_manifest_verified",
    "source_artifact_hashes_current",
    "all_source_audit_flags_true",
    "sealed_recorded_training_protocol_audit_verified",
    "training_checkpoint_sweep_binding_verified",
    "split_manifest_byte_identical",
    "thresholds_unique_sorted_finite",
    "point_count_identities_verified",
    "ground_truth_counts_invariant",
    "fixed_threshold_curve_point_exact",
    "fixed_threshold_checkpoint_object_metrics_exact",
    "fixed_threshold_checkpoint_numeric_deltas_recomputed",
    "fa_budget_points_recomputed_exact",
    "pareto_coordinates_recomputed",
    "hardware_timing_not_used_as_performance_evidence",
}
EXTENDED_INTEGRITY_FLAGS = {
    "runtime_invocation_and_execstart_bound",
    "launch_manifests_hash_bound",
    "frozen_protocol_exact",
    "metrics_800_contiguous_finite",
    "processed_sample_count_exact",
    "learning_rate_schedule_exact",
    "online_best_flags_exact",
    "best_checkpoints_strict_and_finite",
    "best_miou_checkpoints_strict_and_finite",
    "last_checkpoints_strict_and_finite",
    "last_checkpoint_metrics_match_epoch_800",
    "split_manifests_byte_identical",
    "shared_initialization_independently_recomputed",
    "training_data_fingerprint_bound",
    "official_test_code_path_isolation_verified",
}
AGGREGATE_TOP_LEVEL_KEYS = {
    "schema_version",
    "dataset",
    "run_name",
    "expected_epochs",
    "official_test_accessed",
    "selection_source",
    "checkpoint_role",
    "operating_point_rule",
    "auc_computed",
    "mainline_decision_made",
    "output_commit",
    "sealed_source_evidence",
    "sealed_training_certificate",
    "common_provenance",
    "source_sweeps",
    "fixed_threshold_0_5",
    "operating_points_by_fa_budget",
    "global_pareto",
    "aggregator_sha256",
    "integrity_checks_passed",
}
COMMON_PROVENANCE_KEYS = {
    "seed",
    "split_seed",
    "validation_count",
    "validation_split_sha256",
    "match_radius",
    "tiny_area",
    "threshold_configuration",
    "evaluator_sha256",
    "split_artifact_sha256",
    "metric_notes",
    "budgets",
    "invariant_counts",
}
SOURCE_SWEEP_KEYS = {
    "run_dir",
    "sweep_path",
    "sweep_sha256",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "artifact_sha256",
}
SOURCE_SWEEP_ARTIFACT_KEYS = {
    "protocol.json",
    "split.json",
    "summary.json",
    "metrics.jsonl",
    "checkpoint",
    "evaluator",
}
SEALED_SOURCE_EVIDENCE_KEYS = {
    "comparison_dir",
    "training_certificate_path",
    "training_certificate_sha256",
    "sweeps_manifest_path",
    "sweeps_manifest_sha256",
    "complete_manifest_path",
    "complete_manifest_sha256",
    "sealed_base_artifact_sha256",
    "sweep_paths",
    "sweep_sha256",
}
COMPARISON_TOP_LEVEL_KEYS = {
    "dataset",
    "run_name",
    "expected_epochs",
    "report_title",
    "variant_run_names",
    "official_test_accessed",
    "validation_split_sha256",
    "training_split_sha256",
    "shared_initialization_sha256",
    "checkpoint_sha256",
    "integrity_audit",
    "rows",
}
COMPARISON_ROW_KEYS = {
    "variant",
    "seed",
    "pd_best_epoch",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "miou_at_pd_best",
    "niou_at_pd_best",
    "f1_at_pd_best",
    "miou_best_epoch",
    "best_miou",
    "pd_at_miou_best",
    "fa_at_miou_best",
    "parameters",
    "shallow_parameters",
    "elapsed_seconds",
    "best_checkpoint_sha256",
    "best_miou_checkpoint_sha256",
    "run_dir",
    "delta_pd_vs_original",
    "delta_tiny_pd_vs_original",
    "delta_fa_vs_original",
    "delta_miou_at_pd_best_vs_original",
}
EXTENDED_TOP_LEVEL_KEYS = {
    "schema",
    "root",
    "dataset",
    "run_name",
    "expected_epochs",
    "official_test_accessed",
    "selection_source",
    "training_data_sha256",
    "runtime_binding",
    "per_variant",
    "cross_variant_consistency",
    "byte_identical_split_sha256",
    "independently_recomputed_shared_initialization_sha256",
    "official_test_isolation_evidence",
    "frozen_sources_and_data",
    "checks_passed",
    "limitations",
}
EXTENDED_VARIANT_KEYS = {
    "gpu_uuid",
    "invocation_id",
    "event_audit",
    "best_pd_epoch",
    "best_miou_epoch",
    "last_checkpoint_audit",
    "initialization_recomputed",
    "artifact_sha256",
}
EXTENDED_ARTIFACT_KEYS = {
    "protocol.json",
    "split.json",
    "metrics.jsonl",
    "summary.json",
    "best.pth.tar",
    "best_miou.pth.tar",
    "last.pth.tar",
    "launch_manifest",
    "worker_log",
}
EXTENDED_SOURCE_PATHS = {
    "worker": REPO_ROOT / "experiments" / "run_tpd_formal800_4x5090_worker.sh",
    "launcher": REPO_ROOT / "experiments" / "launch_tpd_formal800_4x5090.sh",
    "status": REPO_ROOT / "experiments" / "status_tpd_formal800_4x5090.sh",
    "runner": REPO_ROOT / "experiments" / "train_tpd_pilot.py",
    "evaluator": REPO_ROOT / "experiments" / "evaluate_pd_fa_sweep.py",
    "training_summarizer": REPO_ROOT / "experiments" / "summarize_tpd_pilot.py",
    "sweep_aggregator": REPO_ROOT / "experiments" / "summarize_tpd_pd_fa.py",
    "dataset": REPO_ROOT / "dataset.py",
    "utils": REPO_ROOT / "utils.py",
    "model": REPO_ROOT / "model" / "SCTransNet.py",
    "tpd": REPO_ROOT / "model" / "tpd.py",
    "config": REPO_ROOT / "model" / "Config.py",
    "warmup_scheduler_posthoc_observed": REPO_ROOT / "warmup_scheduler.py",
    "this_auditor": REPO_ROOT
    / "experiments"
    / "audit_tpd_formal800_4x5090.py",
}
POSTPROCESS_PRODUCER_SOURCE_PATHS = {
    "postprocess": REPO_ROOT
    / "experiments"
    / "run_tpd_formal800_4x5090_postprocess.sh",
    "postprocess_launcher": REPO_ROOT
    / "experiments"
    / "launch_tpd_formal800_4x5090_postprocess.sh",
    "training_data_fingerprint": REPO_ROOT
    / "experiments"
    / "fingerprint_tpd_training_data.py",
}
PRODUCER_SOURCE_PATHS = {
    **EXTENDED_SOURCE_PATHS,
    **POSTPROCESS_PRODUCER_SOURCE_PATHS,
}
EXPECTED_PRODUCER_SOURCE_SHA256 = {
    "worker": "72f1c0503fcec0c69f8a3b9c49da57a49db75134cc453031da56d635adc2d7a7",
    "launcher": "78b397a5c17bfeb62c3a83ec3aaf4ff733f97c29f686936e140fb7a0a7741fd8",
    "status": "2ecc3621b4bedf6bd452d1bc1d3273a168925dd7e2af067cc0dfb7aeb0fd0a40",
    "runner": "7532bdc3bcc777aa164e258ab21f78d38ed3a1eaa677a29c8256d900224a7f26",
    "evaluator": "0224ab44dc346ebdbd4cb4775c493bd6eecdc877019832dea0f16e59ab353537",
    "training_summarizer": "49417a3fd43192e52308e7dc4343527bcda71495b78257c5464ba43d8eef7f3e",
    "sweep_aggregator": "482903040cbe9a58f17444eee45aeb67c6763a6aef99bd54892f47be5e21b42e",
    "dataset": "516ea9c410f80cc9ae912cf0443126a067dd14b6cc5ad7945e83cfc497f4678d",
    "utils": "afb6fc221072ddd082b53ccda132232bc9089afd0458d8f0e47a39b9c1e25c13",
    "model": "5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb",
    "tpd": "18a5892edd18ab040e38f18c8d86a02bf3e50b7a4d12d0115ec9a97e8051c135",
    "config": "b7e3e67c379ef4638605ebe612336b0c3cdb1a97f4d6fe731dec80b4847d5596",
    "warmup_scheduler_posthoc_observed": (
        "d7ebc3f9568ebd2da6b141d2923dba22c70d523099320e8aee5e769426b62493"
    ),
    "this_auditor": (
        "a5289b2d0cc8045d1514da5045e015b2d03f669e830817d3fcee1b7872f21c9d"
    ),
    "postprocess": "fb6f341c4e2af373b5d0d03ba5525e82dd07570d597a98101481edfc9fe6835d",
    "postprocess_launcher": (
        "008d3560138b217d4bdbfb01cd4fcf370514ec374b5dd4890d77363d5cd31bd6"
    ),
    "training_data_fingerprint": (
        "26382e38e899bdf4f97b77c6671929c391decef6e4bf4ac40094a7d4e6b0bc7d"
    ),
}
OFFICIAL_TRAIN_INDEX_PATH = (
    REPO_ROOT / "datasets" / EXPECTED_DATASET / "img_idx" / "train_NUDT-SIRST.txt"
)
EXPECTED_OFFICIAL_TRAIN_INDEX_SHA256 = (
    "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3"
)
INVARIANT_COUNTS = {
    "target_count": 189,
    "tiny_target_count": 39,
    "valid_pixel_count": 8716288,
}
OFFICIAL_TEST_ISOLATION_KEYS = {
    "official_train_index",
    "official_train_index_sha256",
    "official_train_count",
    "internal_split_union_equals_official_train",
    "runner_code_path_reads_training_index_only",
    "official_test_code_path_isolation_verified",
    "syscall_level_trace_available",
}
VALID_DECISIONS = {
    "ADVANCE_TPD_TO_MULTI_SEED",
    "DO_NOT_ESTABLISH_CURRENT_TPD_CORE",
    "INCONCLUSIVE_MIXED_TRADEOFF",
}


class PendingEvidenceError(RuntimeError):
    """The producer has not committed the required evidence set yet."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sealed formal800 evidence and apply the post-hoc conservative "
            "seed-42 TPD screening policy"
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to ROOT/DATASET/comparison",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print the decision JSON without writing files",
    )
    args = parser.parse_args()
    for option, value in (("--dataset", args.dataset), ("--run-name", args.run_name)):
        require_safe_component(value, option)
    if args.dataset != EXPECTED_DATASET:
        parser.error(f"--dataset must be exactly {EXPECTED_DATASET!r}")
    if args.run_name != EXPECTED_RUN_NAME:
        parser.error(f"--run-name must be exactly {EXPECTED_RUN_NAME!r}")
    if args.expected_epochs != EXPECTED_EPOCHS:
        parser.error(
            f"--expected-epochs must be exactly {EXPECTED_EPOCHS} for this decision policy"
        )
    return args


def require_safe_component(value: str, option: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{option} must be one safe directory name")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} is not a lowercase SHA-256 digest")
    return value


def require_regular(path: Path, context: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{context} must be a regular non-symlink file: {path}")


def require_directory(path: Path, context: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{context} must be a non-symlink directory: {path}")


def reject_symlink_components(path: Path, context: str) -> None:
    absolute = path.absolute()
    chain = [absolute, *absolute.parents]
    for component in reversed(chain):
        if component.is_symlink():
            raise ValueError(f"{context} contains a symlink component: {component}")


def reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def read_json(path: Path, context: str) -> Dict[str, Any]:
    require_regular(path, context)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must contain a JSON object: {path}")
    assert_finite_tree(payload, context)
    return payload


def validate_frozen_producer_sources() -> Dict[str, str]:
    require_exact_keys(
        EXPECTED_PRODUCER_SOURCE_SHA256,
        PRODUCER_SOURCE_PATHS,
        "frozen producer source SHA policy",
    )
    current: Dict[str, str] = {}
    for label, path in PRODUCER_SOURCE_PATHS.items():
        reject_symlink_components(path, f"frozen producer source {label}")
        require_regular(path, f"frozen producer source {label}")
        digest = sha256_file(path)
        require_equal(
            digest,
            EXPECTED_PRODUCER_SOURCE_SHA256[label],
            f"frozen producer source SHA {label}",
        )
        current[label] = digest
    return current


def recompute_extended_audit(
    root: Path,
    dataset: str,
    run_name: str,
    expected_epochs: int,
    runtime_state: Path,
) -> Tuple[Dict[str, Any], bytes]:
    """Run the frozen auditor independently and return parsed and raw output."""
    reject_symlink_components(runtime_state, "independent-audit runtime state")
    require_regular(runtime_state, "independent-audit runtime state")
    require_equal(
        sha256_file(runtime_state),
        EXPECTED_RUNTIME_STATE_SHA256,
        "independent-audit runtime-state SHA",
    )
    auditor = EXTENDED_SOURCE_PATHS["this_auditor"]
    with tempfile.TemporaryDirectory(prefix="sctransnet-mainline-audit.") as temporary:
        output = Path(temporary) / "extended_integrity_v1.json"
        command = [
            sys.executable,
            str(auditor),
            "--root",
            str(root),
            "--dataset",
            dataset,
            "--run-name",
            run_name,
            "--expected-epochs",
            str(expected_epochs),
            "--runtime-state",
            str(runtime_state),
            "--output",
            str(output),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(
                f"Cannot execute independent extended auditor with {sys.executable}"
            ) from exc
        if completed.returncode != 0:
            raise ValueError(
                "Independent extended auditor failed: "
                f"returncode={completed.returncode}, "
                f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
            )
        require_regular(output, "independently recomputed extended audit")
        raw = output.read_bytes()
        parsed = read_json(output, "independently recomputed extended audit")
    return parsed, raw


def recompute_training_data_fingerprint(dataset: str) -> str:
    """Recompute the official-training-only data fingerprint in a subprocess."""
    tool = POSTPROCESS_PRODUCER_SOURCE_PATHS["training_data_fingerprint"]
    command = [
        sys.executable,
        str(tool),
        "--dataset-dir",
        str(REPO_ROOT / "datasets"),
        "--dataset",
        dataset,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            f"Cannot execute training-data fingerprint tool with {sys.executable}"
        ) from exc
    if completed.returncode != 0:
        raise ValueError(
            "Training-data fingerprint recomputation failed: "
            f"returncode={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise ValueError(
            "Training-data fingerprint tool must emit exactly one digest line"
        )
    digest = require_sha256(lines[0], "recomputed training-data fingerprint")
    require_equal(
        digest,
        EXPECTED_TRAINING_DATA_SHA256,
        "recomputed training-data fingerprint",
    )
    return digest


def validate_official_test_isolation(value: Any) -> Dict[str, Any]:
    evidence = require_mapping(value, "extended.official_test_isolation_evidence")
    require_exact_keys(
        evidence,
        OFFICIAL_TEST_ISOLATION_KEYS,
        "extended.official_test_isolation_evidence",
    )
    reject_symlink_components(OFFICIAL_TRAIN_INDEX_PATH, "official training index")
    require_regular(OFFICIAL_TRAIN_INDEX_PATH, "official training index")
    current_sha = sha256_file(OFFICIAL_TRAIN_INDEX_PATH)
    require_equal(
        current_sha,
        EXPECTED_OFFICIAL_TRAIN_INDEX_SHA256,
        "current official training-index SHA",
    )
    recorded_path = Path(str(evidence.get("official_train_index")))
    reject_symlink_components(recorded_path, "recorded official training index")
    require_equal(
        recorded_path.resolve(),
        OFFICIAL_TRAIN_INDEX_PATH.resolve(),
        "official-test isolation training-index path",
    )
    require_equal(
        evidence.get("official_train_index_sha256"),
        current_sha,
        "official-test isolation training-index SHA",
    )
    require_equal(
        evidence.get("official_train_count"),
        663,
        "official-test isolation training count",
    )
    for flag in (
        "internal_split_union_equals_official_train",
        "runner_code_path_reads_training_index_only",
        "official_test_code_path_isolation_verified",
    ):
        if evidence.get(flag) is not True:
            raise ValueError(f"Official-test isolation flag {flag} did not pass")
    if evidence.get("syscall_level_trace_available") is not False:
        raise ValueError(
            "Official-test isolation evidence must disclose absent syscall-level trace"
        )
    return evidence


def finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} is not finite: {value!r}")
    return result


def assert_finite_tree(value: Any, context: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        finite_number(value, context)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{context}.{key}")
        return
    raise ValueError(f"{context} contains unsupported JSON-like type {type(value)!r}")


def require_mapping(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def require_exact_keys(
    mapping: Mapping[str, Any], expected: Iterable[str], context: str
) -> None:
    expected_set = set(expected)
    actual_set = set(mapping)
    if actual_set != expected_set:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )


def canonical(value: Any) -> str:
    def encode_special(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported canonical value type: {type(item)!r}")

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=encode_special,
    )


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if canonical(actual) != canonical(expected):
        raise ValueError(f"{context} mismatch: expected={expected!r}, actual={actual!r}")


def manifest_line(digest: str, name: str) -> str:
    require_sha256(digest, f"manifest digest for {name}")
    if (
        not name
        or "\\" in name
        or "\n" in name
        or "\r" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ValueError(f"Unsafe manifest name: {name!r}")
    return f"{digest}  {name}\n"


def verify_exact_manifest(
    marker: Path, entries: Sequence[Tuple[Path, str]], context: str
) -> str:
    require_regular(marker, context)
    expected = ""
    for path, name in entries:
        require_regular(path, f"{context} payload {name}")
        expected += manifest_line(sha256_file(path), name)
    try:
        actual = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {context}: {marker}: {exc}") from exc
    if actual != expected:
        raise ValueError(f"{context} does not exactly seal the expected files: {marker}")
    return sha256_file(marker)


def require_committed_group(
    marker: Path, payloads: Sequence[Path], context: str
) -> None:
    if marker.is_symlink():
        raise ValueError(f"{context} marker must not be a symlink: {marker}")
    if not marker.exists():
        raise PendingEvidenceError(f"{context} has not been committed: {marker}")
    if not marker.is_file():
        raise ValueError(f"{context} marker is not a regular file: {marker}")
    missing = [str(path) for path in payloads if not path.exists()]
    if missing:
        raise ValueError(
            f"{context} marker exists but committed payloads are missing: {missing}"
        )
    for path in payloads:
        require_regular(path, f"{context} payload {path.name}")


def validate_point(
    value: Any, context: str, validation_count: int | None = None
) -> Dict[str, Any]:
    point = require_mapping(value, context)
    require_exact_keys(point, POINT_KEYS, context)
    for key in POINT_KEYS:
        finite_number(point[key], f"{context}.{key}")
    for key in COUNT_KEYS:
        count = point[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{context}.{key} must be a non-negative integer")
    threshold = float(point["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"{context}.threshold must lie in (0, 1)")
    for key in (
        "pd",
        "tiny_pd",
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    ):
        if not 0.0 <= float(point[key]) <= 1.0:
            raise ValueError(f"{context}.{key} must lie in [0, 1]")
    if float(point["fa"]) < 0.0 or float(point["false_objects_per_image"]) < 0.0:
        raise ValueError(f"{context} contains a negative false-alarm metric")
    if point["target_count"] < 1 or point["tiny_target_count"] < 1:
        raise ValueError(f"{context} requires positive target counts")
    if point["valid_pixel_count"] < 1:
        raise ValueError(f"{context}.valid_pixel_count must be positive")
    for key, expected in INVARIANT_COUNTS.items():
        if point[key] != expected:
            raise ValueError(
                f"{context}.{key} must equal the frozen value {expected}, "
                f"got {point[key]!r}"
            )
    if point["matched_target_count"] > point["target_count"]:
        raise ValueError(f"{context} matches more targets than exist")
    if point["matched_tiny_target_count"] > point["tiny_target_count"]:
        raise ValueError(f"{context} matches more tiny targets than exist")
    if point["matched_tiny_target_count"] > point["matched_target_count"]:
        raise ValueError(f"{context} tiny matches exceed all matches")
    if point["tiny_target_count"] > point["target_count"]:
        raise ValueError(f"{context} tiny targets exceed all targets")
    if (
        point["predicted_object_count"]
        != point["matched_target_count"] + point["unmatched_predicted_object_count"]
    ):
        raise ValueError(f"{context} predicted-object count identity fails")
    expected_pd = point["matched_target_count"] / point["target_count"]
    expected_tiny_pd = point["matched_tiny_target_count"] / point["tiny_target_count"]
    if not math.isclose(float(point["pd"]), expected_pd, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{context}.pd does not match object counts")
    if not math.isclose(
        float(point["tiny_pd"]), expected_tiny_pd, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(f"{context}.tiny_pd does not match tiny-object counts")
    if validation_count is not None:
        if validation_count < 1:
            raise ValueError(f"{context} validation count must be positive")
        expected_false_objects = (
            point["unmatched_predicted_object_count"] / validation_count
        )
        if not math.isclose(
            float(point["false_objects_per_image"]),
            expected_false_objects,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"{context}.false_objects_per_image does not match object counts"
            )
    return point


def raw_operating_point_key(
    point: Mapping[str, Any]
) -> Tuple[float, float, float, float, float]:
    return (
        float(point["pd"]),
        -float(point["fa"]),
        float(point["tiny_pd"]),
        float(point["miou"]),
        -abs(float(point["threshold"]) - 0.5),
    )


def recompute_budget_point(
    points: Sequence[Dict[str, Any]], budget: float
) -> Dict[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    return max(feasible, key=raw_operating_point_key) if feasible else None


def recompute_global_pareto(
    sweep_points: Mapping[str, Sequence[Dict[str, Any]]]
) -> Dict[str, Any]:
    owners: Dict[Tuple[float, float], set[str]] = {}
    samples: Dict[Tuple[float, float], list[Dict[str, Any]]] = {}
    for variant in VARIANTS:
        for point in sweep_points[variant]:
            coordinate = (float(point["pd"]), float(point["fa"]))
            owners.setdefault(coordinate, set()).add(variant)
            samples.setdefault(coordinate, []).append(
                {
                    "variant": variant,
                    "threshold": point["threshold"],
                    "tiny_pd": point["tiny_pd"],
                    "miou": point["miou"],
                }
            )
    coordinates = list(owners)
    frontier: set[Tuple[float, float]] = set()
    for pd_value, fa_value in coordinates:
        dominated = any(
            other_fa <= fa_value
            and other_pd >= pd_value
            and (other_fa < fa_value or other_pd > pd_value)
            for other_pd, other_fa in coordinates
        )
        if not dominated:
            frontier.add((pd_value, fa_value))
    ordered = sorted(frontier, key=lambda coordinate: (coordinate[1], -coordinate[0]))
    rows = [
        {
            "pd": coordinate[0],
            "fa": coordinate[1],
            "owners": sorted(owners[coordinate]),
            "samples": sorted(
                samples[coordinate],
                key=lambda sample: (sample["variant"], sample["threshold"]),
            ),
        }
        for coordinate in ordered
    ]
    owner_counts = {
        variant: sum(variant in owners[coordinate] for coordinate in ordered)
        for variant in VARIANTS
    }
    return {
        "scope": "joint_sampled_discrete_threshold_coordinates",
        "dominance_definition": (
            "coordinate A dominates B iff A.Fa <= B.Fa and A.Pd >= B.Pd, "
            "with at least one strict inequality; identical coordinates retain all owners"
        ),
        "unique_coordinate_count": len(rows),
        "owner_coordinate_counts": owner_counts,
        "coordinates": rows,
    }


def validate_comparison(
    payload: Dict[str, Any],
    dataset: str,
    run_name: str,
    expected_epochs: int,
) -> Dict[str, Dict[str, Any]]:
    require_exact_keys(payload, COMPARISON_TOP_LEVEL_KEYS, "comparison")
    require_equal(payload.get("dataset"), dataset, "comparison.dataset")
    require_equal(payload.get("run_name"), run_name, "comparison.run_name")
    require_equal(
        payload.get("expected_epochs"), expected_epochs, "comparison.expected_epochs"
    )
    if payload.get("official_test_accessed") is not False:
        raise ValueError("comparison does not assert official-test isolation")
    require_equal(
        payload.get("validation_split_sha256"),
        EXPECTED_VALIDATION_SPLIT_SHA256,
        "comparison validation split SHA",
    )
    require_equal(
        payload.get("shared_initialization_sha256"),
        EXPECTED_SHARED_INITIALIZATION_SHA256,
        "comparison shared initialization SHA",
    )
    require_equal(
        payload.get("variant_run_names"),
        {variant: run_name for variant in VARIANTS},
        "comparison.variant_run_names",
    )
    integrity = require_mapping(payload.get("integrity_audit"), "comparison.integrity_audit")
    require_equal(integrity.get("seed"), EXPECTED_SEED, "comparison seed")
    split_hashes = require_mapping(
        integrity.get("split_hashes"), "comparison.integrity_audit.split_hashes"
    )
    require_equal(
        split_hashes.get("used_val_sha256"),
        EXPECTED_VALIDATION_SPLIT_SHA256,
        "comparison used validation split SHA",
    )
    require_equal(
        integrity.get("shared_initialization_sha256"),
        EXPECTED_SHARED_INITIALIZATION_SHA256,
        "comparison integrity shared initialization SHA",
    )
    critical_arguments = require_mapping(
        integrity.get("critical_protocol_arguments"),
        "comparison.integrity_audit.critical_protocol_arguments",
    )
    require_equal(
        critical_arguments.get("seed"), EXPECTED_SEED, "comparison protocol seed"
    )
    require_equal(
        critical_arguments.get("split_seed"),
        EXPECTED_SPLIT_SEED,
        "comparison protocol split seed",
    )
    require_equal(
        critical_arguments.get("epochs"),
        expected_epochs,
        "comparison protocol epochs",
    )
    require_equal(
        critical_arguments.get("tiny_area"), 9, "comparison protocol tiny area"
    )
    require_equal(
        critical_arguments.get("match_radius"),
        3.0,
        "comparison protocol match radius",
    )
    require_equal(
        critical_arguments.get("threshold"),
        0.5,
        "comparison protocol fixed threshold",
    )
    split_counts = require_mapping(
        integrity.get("split_counts"), "comparison.integrity_audit.split_counts"
    )
    require_equal(
        split_counts.get("used_val_count"),
        EXPECTED_VALIDATION_COUNT,
        "comparison validation count",
    )
    checkpoint_sha = require_mapping(
        payload.get("checkpoint_sha256"), "comparison.checkpoint_sha256"
    )
    require_exact_keys(checkpoint_sha, VARIANTS, "comparison.checkpoint_sha256")
    for variant in VARIANTS:
        entry = require_mapping(
            checkpoint_sha[variant], f"comparison.checkpoint_sha256.{variant}"
        )
        require_exact_keys(
            entry,
            {"best.pth.tar", "best_miou.pth.tar"},
            f"comparison.checkpoint_sha256.{variant}",
        )
        for name, digest in entry.items():
            require_sha256(digest, f"comparison checkpoint {variant}.{name}")

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(VARIANTS):
        raise ValueError("comparison.rows must contain exactly four variants")
    rows: Dict[str, Dict[str, Any]] = {}
    for index, value in enumerate(raw_rows):
        row = require_mapping(value, f"comparison.rows[{index}]")
        require_exact_keys(row, COMPARISON_ROW_KEYS, f"comparison.rows[{index}]")
        variant = row.get("variant")
        if variant not in VARIANTS or variant in rows:
            raise ValueError(f"Invalid or duplicate comparison variant: {variant!r}")
        if row.get("seed") != EXPECTED_SEED:
            raise ValueError(f"comparison row {variant} has the wrong seed")
        for key, value in row.items():
            if key in {"variant", "run_dir", "best_checkpoint_sha256", "best_miou_checkpoint_sha256"}:
                continue
            finite_number(value, f"comparison row {variant}.{key}")
        require_equal(
            row.get("best_checkpoint_sha256"),
            checkpoint_sha[variant]["best.pth.tar"],
            f"comparison row {variant} Pd checkpoint SHA",
        )
        require_equal(
            row.get("best_miou_checkpoint_sha256"),
            checkpoint_sha[variant]["best_miou.pth.tar"],
            f"comparison row {variant} mIoU checkpoint SHA",
        )
        rows[str(variant)] = row
    require_exact_keys(rows, VARIANTS, "comparison row variants")
    return rows


def validate_aggregate(
    payload: Dict[str, Any],
    aggregate_path: Path,
    aggregate_marker: Path,
    comparison_dir: Path,
    comparison_path: Path,
    complete_path: Path,
    sweeps_manifest_path: Path,
    dataset: str,
    run_name: str,
    expected_epochs: int,
) -> Dict[str, Any]:
    require_exact_keys(payload, AGGREGATE_TOP_LEVEL_KEYS, "aggregate")
    require_equal(
        payload.get("schema_version"),
        "tpd-pd-fa-aggregate-v2",
        "aggregate.schema_version",
    )
    require_equal(payload.get("dataset"), dataset, "aggregate.dataset")
    require_equal(payload.get("run_name"), run_name, "aggregate.run_name")
    require_equal(
        payload.get("expected_epochs"), expected_epochs, "aggregate.expected_epochs"
    )
    if payload.get("official_test_accessed") is not False:
        raise ValueError("aggregate does not assert official-test isolation")
    require_equal(
        payload.get("selection_source"),
        "internal_validation_only",
        "aggregate.selection_source",
    )
    require_equal(
        payload.get("checkpoint_role"),
        "best_validation_pd_primary",
        "aggregate.checkpoint_role",
    )
    require_equal(
        payload.get("operating_point_rule"),
        list(OPERATING_POINT_RULE),
        "aggregate.operating_point_rule",
    )
    if payload.get("auc_computed") is not False:
        raise ValueError("aggregate unexpectedly reports an AUC")
    if payload.get("mainline_decision_made") is not False:
        raise ValueError("aggregate unexpectedly claims a mainline decision")
    require_equal(
        payload.get("aggregator_sha256"),
        EXPECTED_AGGREGATOR_SHA256,
        "aggregate.aggregator_sha256",
    )

    output_commit = require_mapping(payload.get("output_commit"), "aggregate.output_commit")
    require_equal(
        Path(str(output_commit.get("marker"))).resolve(),
        aggregate_marker.resolve(),
        "aggregate completion marker path",
    )

    integrity = require_mapping(
        payload.get("integrity_checks_passed"), "aggregate.integrity_checks_passed"
    )
    require_exact_keys(
        integrity, AGGREGATE_INTEGRITY_FLAGS, "aggregate.integrity_checks_passed"
    )
    if any(value is not True for value in integrity.values()):
        raise ValueError("One or more of the 17 aggregate integrity gates did not pass")

    common = require_mapping(payload.get("common_provenance"), "aggregate.common_provenance")
    require_exact_keys(common, COMMON_PROVENANCE_KEYS, "aggregate.common_provenance")
    require_equal(common.get("seed"), EXPECTED_SEED, "aggregate common seed")
    require_equal(
        common.get("split_seed"), EXPECTED_SPLIT_SEED, "aggregate common split seed"
    )
    require_equal(
        common.get("validation_count"),
        EXPECTED_VALIDATION_COUNT,
        "aggregate common validation count",
    )
    require_equal(
        common.get("validation_split_sha256"),
        EXPECTED_VALIDATION_SPLIT_SHA256,
        "aggregate validation split SHA",
    )
    require_equal(common.get("match_radius"), 3.0, "aggregate match radius")
    require_equal(common.get("tiny_area"), 9, "aggregate tiny area")
    require_equal(
        common.get("evaluator_sha256"),
        EXPECTED_EVALUATOR_SHA256,
        "aggregate evaluator SHA",
    )
    require_equal(
        common.get("split_artifact_sha256"),
        EXPECTED_SPLIT_ARTIFACT_SHA256,
        "aggregate split artifact SHA",
    )
    require_equal(
        common.get("invariant_counts"),
        INVARIANT_COUNTS,
        "aggregate invariant counts",
    )
    budgets = common.get("budgets")
    require_equal(budgets, list(FORMAL_FA_BUDGETS), "aggregate formal Fa budgets")
    threshold_configuration = require_mapping(
        common.get("threshold_configuration"),
        "aggregate.common_provenance.threshold_configuration",
    )
    require_exact_keys(
        threshold_configuration,
        {
            "threshold_min",
            "threshold_max",
            "threshold_step",
            "extra_thresholds",
            "tail_logit_step",
            "fa_budgets",
        },
        "aggregate threshold configuration",
    )
    require_equal(
        threshold_configuration.get("threshold_min"),
        0.01,
        "aggregate threshold minimum",
    )
    require_equal(
        threshold_configuration.get("threshold_max"),
        0.99,
        "aggregate threshold maximum",
    )
    require_equal(
        threshold_configuration.get("threshold_step"),
        0.01,
        "aggregate threshold step",
    )
    require_equal(
        threshold_configuration.get("extra_thresholds"),
        [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
        "aggregate extra thresholds",
    )
    require_equal(
        threshold_configuration.get("tail_logit_step"),
        0.1,
        "aggregate tail-logit step",
    )
    require_equal(
        threshold_configuration.get("fa_budgets"),
        list(FORMAL_FA_BUDGETS),
        "aggregate threshold-configuration Fa budgets",
    )

    fixed = require_mapping(
        payload.get("fixed_threshold_0_5"), "aggregate.fixed_threshold_0_5"
    )
    require_exact_keys(fixed, VARIANTS, "aggregate.fixed_threshold_0_5")
    for variant in VARIANTS:
        point = validate_point(
            fixed[variant],
            f"aggregate fixed 0.5 {variant}",
            EXPECTED_VALIDATION_COUNT,
        )
        require_equal(point["threshold"], 0.5, f"aggregate fixed threshold {variant}")

    operating = require_mapping(
        payload.get("operating_points_by_fa_budget"),
        "aggregate.operating_points_by_fa_budget",
    )
    require_exact_keys(
        operating,
        FORMAL_FA_BUDGET_KEYS,
        "aggregate.operating_points_by_fa_budget",
    )
    for budget, budget_key in zip(FORMAL_FA_BUDGETS, FORMAL_FA_BUDGET_KEYS):
        points = require_mapping(operating[budget_key], f"aggregate budget {budget_key}")
        require_exact_keys(points, VARIANTS, f"aggregate budget {budget_key}")
        for variant in VARIANTS:
            if points[variant] is None:
                continue
            point = validate_point(
                points[variant],
                f"aggregate budget {budget_key}.{variant}",
                EXPECTED_VALIDATION_COUNT,
            )
            if float(point["fa"]) > budget:
                raise ValueError(
                    f"aggregate budget {budget_key}.{variant} exceeds its Fa budget"
                )

    pareto = require_mapping(payload.get("global_pareto"), "aggregate.global_pareto")
    require_equal(
        pareto.get("scope"),
        "joint_sampled_discrete_threshold_coordinates",
        "aggregate global-Pareto scope",
    )
    coordinates = pareto.get("coordinates")
    if not isinstance(coordinates, list):
        raise ValueError("aggregate.global_pareto.coordinates must be an array")
    require_equal(
        pareto.get("unique_coordinate_count"),
        len(coordinates),
        "aggregate global-Pareto coordinate count",
    )
    recomputed_owner_counts = {variant: 0 for variant in VARIANTS}
    seen_coordinates: set[Tuple[float, float]] = set()
    for index, row_value in enumerate(coordinates):
        row = require_mapping(row_value, f"aggregate Pareto row {index}")
        pd_value = finite_number(row.get("pd"), f"aggregate Pareto row {index}.pd")
        fa_value = finite_number(row.get("fa"), f"aggregate Pareto row {index}.fa")
        if not 0.0 <= pd_value <= 1.0 or fa_value < 0.0:
            raise ValueError(f"aggregate Pareto row {index} has out-of-range metrics")
        coordinate = (pd_value, fa_value)
        if coordinate in seen_coordinates:
            raise ValueError(f"aggregate Pareto coordinate {coordinate} is duplicated")
        seen_coordinates.add(coordinate)
        owners = row.get("owners")
        if (
            not isinstance(owners, list)
            or not owners
            or owners != sorted(set(owners))
            or any(owner not in VARIANTS for owner in owners)
        ):
            raise ValueError(f"aggregate Pareto row {index} has invalid owners")
        samples = row.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"aggregate Pareto row {index} has no samples")
        sample_owners = {sample.get("variant") for sample in samples if isinstance(sample, dict)}
        if not set(owners) <= sample_owners:
            raise ValueError(f"aggregate Pareto row {index} owner/sample mismatch")
        for owner in owners:
            recomputed_owner_counts[owner] += 1
    require_equal(
        pareto.get("owner_coordinate_counts"),
        recomputed_owner_counts,
        "aggregate global-Pareto owner counts",
    )

    source_seal = require_mapping(
        payload.get("sealed_source_evidence"), "aggregate.sealed_source_evidence"
    )
    require_exact_keys(
        source_seal,
        SEALED_SOURCE_EVIDENCE_KEYS,
        "aggregate.sealed_source_evidence",
    )
    require_equal(
        Path(str(source_seal.get("comparison_dir"))).resolve(),
        comparison_dir.resolve(),
        "aggregate sealed comparison directory",
    )
    require_equal(
        Path(str(source_seal.get("training_certificate_path"))).resolve(),
        comparison_path.resolve(),
        "aggregate sealed training-certificate path",
    )
    require_equal(
        source_seal.get("training_certificate_sha256"),
        sha256_file(comparison_path),
        "aggregate sealed training-certificate SHA",
    )
    require_equal(
        Path(str(source_seal.get("sweeps_manifest_path"))).resolve(),
        sweeps_manifest_path.resolve(),
        "aggregate sealed sweep-manifest path",
    )
    require_equal(
        source_seal.get("sweeps_manifest_sha256"),
        sha256_file(sweeps_manifest_path),
        "aggregate sealed sweep-manifest SHA",
    )
    require_equal(
        Path(str(source_seal.get("complete_manifest_path"))).resolve(),
        complete_path.resolve(),
        "aggregate sealed comparison-completion path",
    )
    require_equal(
        source_seal.get("complete_manifest_sha256"),
        sha256_file(complete_path),
        "aggregate sealed comparison-completion SHA",
    )
    base_hashes = require_mapping(
        source_seal.get("sealed_base_artifact_sha256"),
        "aggregate sealed base artifact hashes",
    )
    expected_base_names = (
        f"{run_name}.json",
        f"{run_name}.md",
        f"{run_name}.csv",
        "SWEEPS.sha256",
    )
    require_exact_keys(base_hashes, expected_base_names, "aggregate sealed base hashes")
    for name in expected_base_names:
        require_equal(
            base_hashes[name],
            sha256_file(comparison_dir / name),
            f"aggregate sealed base SHA {name}",
        )

    source_sweeps = require_mapping(
        payload.get("source_sweeps"), "aggregate.source_sweeps"
    )
    require_exact_keys(source_sweeps, VARIANTS, "aggregate.source_sweeps")
    sealed_sweep_hashes = require_mapping(
        source_seal.get("sweep_sha256"), "aggregate sealed sweep hashes"
    )
    sealed_sweep_paths = require_mapping(
        source_seal.get("sweep_paths"), "aggregate sealed sweep paths"
    )
    require_exact_keys(sealed_sweep_hashes, VARIANTS, "aggregate sealed sweep hashes")
    require_exact_keys(sealed_sweep_paths, VARIANTS, "aggregate sealed sweep paths")
    expected_sweeps_manifest = ""
    sweep_points_by_variant: Dict[str, list[Dict[str, Any]]] = {}
    for variant in MANIFEST_VARIANTS:
        source = require_mapping(
            source_sweeps[variant], f"aggregate.source_sweeps.{variant}"
        )
        require_exact_keys(
            source, SOURCE_SWEEP_KEYS, f"aggregate.source_sweeps.{variant}"
        )
        expected_sweep_candidate = (
            comparison_dir.parent / variant / run_name / "pd_fa_sweep_best.pth.json"
        )
        reject_symlink_components(
            expected_sweep_candidate, f"{variant} expected sweep path"
        )
        expected_sweep_path = expected_sweep_candidate.resolve()
        expected_run_dir = expected_sweep_path.parent
        recorded_run_dir = Path(str(source.get("run_dir")))
        reject_symlink_components(recorded_run_dir, f"{variant} recorded run directory")
        require_equal(
            recorded_run_dir.resolve(),
            expected_run_dir,
            f"aggregate {variant} run directory",
        )
        recorded_sweep_path = Path(str(source.get("sweep_path")))
        reject_symlink_components(recorded_sweep_path, f"{variant} recorded sweep path")
        sweep_path = recorded_sweep_path.resolve()
        require_equal(sweep_path, expected_sweep_path, f"aggregate {variant} sweep path")
        sealed_recorded_sweep_path = Path(str(sealed_sweep_paths[variant]))
        reject_symlink_components(
            sealed_recorded_sweep_path, f"{variant} sealed recorded sweep path"
        )
        require_equal(
            sealed_recorded_sweep_path.resolve(),
            expected_sweep_path,
            f"aggregate sealed {variant} sweep path",
        )
        sweep_sha = require_sha256(
            source.get("sweep_sha256"), f"aggregate {variant} sweep SHA"
        )
        require_equal(
            sweep_sha,
            sealed_sweep_hashes[variant],
            f"aggregate sealed {variant} sweep SHA",
        )
        require_regular(expected_sweep_path, f"{variant} current sweep")
        require_equal(
            sha256_file(expected_sweep_path),
            sweep_sha,
            f"{variant} current sweep SHA",
        )
        checkpoint_sha = require_sha256(
            source.get("checkpoint_sha256"),
            f"aggregate {variant} Pd checkpoint SHA",
        )
        artifacts = require_mapping(
            source.get("artifact_sha256"),
            f"aggregate {variant} source artifact SHA",
        )
        require_exact_keys(
            artifacts,
            SOURCE_SWEEP_ARTIFACT_KEYS,
            f"aggregate {variant} source artifact SHA",
        )
        for name, digest in artifacts.items():
            require_sha256(digest, f"aggregate {variant} source artifact {name}")
        require_equal(
            artifacts["evaluator"],
            EXPECTED_EVALUATOR_SHA256,
            f"aggregate {variant} source evaluator SHA",
        )
        current_artifact_paths = {
            "protocol.json": expected_run_dir / "protocol.json",
            "split.json": expected_run_dir / "split.json",
            "summary.json": expected_run_dir / "summary.json",
            "metrics.jsonl": expected_run_dir / "metrics.jsonl",
            "checkpoint": expected_run_dir / "best.pth.tar",
            "evaluator": EXTENDED_SOURCE_PATHS["evaluator"],
        }
        for name, path in current_artifact_paths.items():
            reject_symlink_components(path, f"{variant} {name} path")
            require_regular(path, f"{variant} current {name}")
            require_equal(
                sha256_file(path),
                artifacts[name],
                f"{variant} current {name} SHA",
            )
        require_equal(
            artifacts["checkpoint"],
            checkpoint_sha,
            f"{variant} source/checkpoint SHA",
        )
        require_equal(
            artifacts["split.json"],
            EXPECTED_SPLIT_ARTIFACT_SHA256,
            f"{variant} source split SHA",
        )
        require_equal(
            artifacts["split.json"],
            common["split_artifact_sha256"],
            f"{variant} source/common split SHA",
        )

        sweep_payload = read_json(expected_sweep_path, f"{variant} current sweep JSON")
        require_equal(sweep_payload.get("variant"), variant, f"{variant} sweep variant")
        require_equal(sweep_payload.get("dataset"), dataset, f"{variant} sweep dataset")
        require_equal(
            sweep_payload.get("checkpoint_role"),
            "best_validation_pd_primary",
            f"{variant} sweep checkpoint role",
        )
        require_equal(
            sweep_payload.get("checkpoint_sha256"),
            checkpoint_sha,
            f"{variant} sweep checkpoint SHA",
        )
        require_equal(
            sweep_payload.get("checkpoint_epoch"),
            source.get("checkpoint_epoch"),
            f"{variant} sweep checkpoint epoch",
        )
        require_equal(sweep_payload.get("seed"), EXPECTED_SEED, f"{variant} sweep seed")
        require_equal(
            sweep_payload.get("split_seed"),
            EXPECTED_SPLIT_SEED,
            f"{variant} sweep split seed",
        )
        require_equal(
            sweep_payload.get("validation_count"),
            EXPECTED_VALIDATION_COUNT,
            f"{variant} sweep validation count",
        )
        require_equal(
            sweep_payload.get("validation_split_sha256"),
            EXPECTED_VALIDATION_SPLIT_SHA256,
            f"{variant} sweep validation split SHA",
        )
        if sweep_payload.get("official_test_accessed") is not False:
            raise ValueError(f"{variant} sweep does not assert official-test isolation")
        require_equal(
            sweep_payload.get("match_radius"), 3.0, f"{variant} sweep match radius"
        )
        require_equal(sweep_payload.get("tiny_area"), 9, f"{variant} sweep tiny area")
        require_equal(
            sweep_payload.get("threshold_configuration"),
            threshold_configuration,
            f"{variant} sweep threshold configuration",
        )
        raw_points = sweep_payload.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError(f"{variant} sweep points must be a non-empty array")
        current_points = [
            validate_point(
                value,
                f"{variant} sweep point {index}",
                EXPECTED_VALIDATION_COUNT,
            )
            for index, value in enumerate(raw_points)
        ]
        thresholds = [float(point["threshold"]) for point in current_points]
        if thresholds != sorted(thresholds) or len(thresholds) != len(set(thresholds)):
            raise ValueError(f"{variant} sweep thresholds are not unique and sorted")
        half_points = [
            point for point in current_points if float(point["threshold"]) == 0.5
        ]
        if len(half_points) != 1:
            raise ValueError(f"{variant} sweep must have exactly one threshold-0.5 point")
        require_equal(
            sweep_payload.get("fixed_threshold_0_5"),
            half_points[0],
            f"{variant} sweep fixed point",
        )
        require_equal(
            fixed[variant],
            half_points[0],
            f"{variant} aggregate/sweep fixed point",
        )
        reported_budget_points = require_mapping(
            sweep_payload.get("best_points_under_fa_budget"),
            f"{variant} sweep budget points",
        )
        require_exact_keys(
            reported_budget_points,
            FORMAL_FA_BUDGET_KEYS,
            f"{variant} sweep budget points",
        )
        for budget, budget_key in zip(FORMAL_FA_BUDGETS, FORMAL_FA_BUDGET_KEYS):
            recomputed = recompute_budget_point(current_points, budget)
            require_equal(
                reported_budget_points[budget_key],
                recomputed,
                f"{variant} sweep recomputed budget {budget_key}",
            )
            require_equal(
                operating[budget_key][variant],
                recomputed,
                f"{variant} aggregate/sweep budget {budget_key}",
            )
        sweep_points_by_variant[variant] = current_points
        expected_sweeps_manifest += manifest_line(sweep_sha, str(expected_sweep_path))
    if sweeps_manifest_path.read_text(encoding="utf-8") != expected_sweeps_manifest:
        raise ValueError("SWEEPS.sha256 does not match the aggregate's sealed sweep records")
    require_equal(
        pareto,
        recompute_global_pareto(sweep_points_by_variant),
        "aggregate global Pareto recomputed from current sweeps",
    )

    sealed_training_certificate = require_mapping(
        payload.get("sealed_training_certificate"),
        "aggregate.sealed_training_certificate",
    )
    require_exact_keys(
        sealed_training_certificate,
        {"integrity_audit", "training_to_sweep_binding", "hardware_note"},
        "aggregate.sealed_training_certificate",
    )
    return {
        "fixed": fixed,
        "operating": operating,
        "pareto": pareto,
        "common": common,
        "source_seal": source_seal,
        "source_sweeps": source_sweeps,
        "sealed_training_certificate": sealed_training_certificate,
        "aggregator_sha256": payload["aggregator_sha256"],
        "aggregate_sha256": sha256_file(aggregate_path),
    }


def validate_cross_bindings(
    comparison: Dict[str, Any],
    comparison_rows: Mapping[str, Dict[str, Any]],
    aggregate: Dict[str, Any],
    aggregate_evidence: Dict[str, Any],
    extended: Dict[str, Any],
    root: Path,
    dataset: str,
    run_name: str,
    expected_epochs: int,
) -> None:
    training_certificate = aggregate_evidence["sealed_training_certificate"]
    require_equal(
        training_certificate.get("integrity_audit"),
        comparison.get("integrity_audit"),
        "aggregate/comparison training integrity",
    )
    bindings = require_mapping(
        training_certificate.get("training_to_sweep_binding"),
        "aggregate training-to-sweep binding",
    )
    require_exact_keys(bindings, VARIANTS, "aggregate training-to-sweep binding")
    fixed = aggregate_evidence["fixed"]
    source_sweeps = aggregate_evidence["source_sweeps"]
    checkpoint_sha = require_mapping(
        comparison.get("checkpoint_sha256"), "comparison checkpoint SHA"
    )
    fixed_bindings = {
        "pd": "pd",
        "tiny_pd": "tiny_pd",
        "fa": "fa",
        "false_objects_per_image": "false_objects_per_image",
        "miou_at_pd_best": "miou",
        "niou_at_pd_best": "niou",
        "f1_at_pd_best": "pixel_f1",
    }
    for variant in VARIANTS:
        source = require_mapping(source_sweeps[variant], f"source sweep {variant}")
        binding = require_mapping(bindings[variant], f"training binding {variant}")
        row = comparison_rows[variant]
        require_equal(
            source.get("checkpoint_sha256"),
            checkpoint_sha[variant]["best.pth.tar"],
            f"{variant} aggregate/comparison Pd checkpoint SHA",
        )
        require_equal(
            binding.get("checkpoint_sha256"),
            checkpoint_sha[variant]["best.pth.tar"],
            f"{variant} training-binding checkpoint SHA",
        )
        require_equal(
            source.get("checkpoint_epoch"),
            row.get("pd_best_epoch"),
            f"{variant} aggregate/comparison checkpoint epoch",
        )
        require_equal(
            binding.get("checkpoint_epoch"),
            row.get("pd_best_epoch"),
            f"{variant} training-binding checkpoint epoch",
        )
        if binding.get("row_metric_binding_passed") is not True:
            raise ValueError(f"{variant} training-to-sweep row binding did not pass")
        for row_key, point_key_name in fixed_bindings.items():
            require_equal(
                row.get(row_key),
                fixed[variant][point_key_name],
                f"{variant} aggregate/comparison fixed metric {row_key}",
            )

    require_equal(
        extended.get("schema"),
        "sctransnet_formal800_4x5090_extended_integrity_v1",
        "extended.schema",
    )
    require_exact_keys(extended, EXTENDED_TOP_LEVEL_KEYS, "extended")
    require_equal(
        Path(str(extended.get("root"))).resolve(), root.resolve(), "extended.root"
    )
    require_equal(extended.get("dataset"), dataset, "extended.dataset")
    require_equal(extended.get("run_name"), run_name, "extended.run_name")
    require_equal(
        extended.get("expected_epochs"), expected_epochs, "extended.expected_epochs"
    )
    if extended.get("official_test_accessed") is not False:
        raise ValueError("extended audit does not assert official-test isolation")
    require_equal(
        extended.get("selection_source"),
        "internal_validation_only",
        "extended.selection_source",
    )
    require_equal(
        extended.get("training_data_sha256"),
        EXPECTED_TRAINING_DATA_SHA256,
        "extended training-data SHA",
    )
    require_equal(
        extended.get("byte_identical_split_sha256"),
        EXPECTED_SPLIT_ARTIFACT_SHA256,
        "extended byte-identical split SHA",
    )
    require_equal(
        extended.get("independently_recomputed_shared_initialization_sha256"),
        EXPECTED_SHARED_INITIALIZATION_SHA256,
        "extended shared initialization SHA",
    )
    runtime_binding = require_mapping(
        extended.get("runtime_binding"), "extended.runtime_binding"
    )
    runtime_state_candidate = root / "launch" / RUNTIME_STATE_NAME
    reject_symlink_components(runtime_state_candidate, "runtime state path")
    require_regular(runtime_state_candidate, "current runtime state")
    recorded_runtime_state = Path(str(runtime_binding.get("state_path")))
    reject_symlink_components(recorded_runtime_state, "recorded runtime state path")
    require_equal(
        recorded_runtime_state.resolve(),
        runtime_state_candidate.resolve(),
        "extended runtime-state path",
    )
    require_equal(
        runtime_binding.get("state_sha256"),
        sha256_file(runtime_state_candidate),
        "extended current runtime-state SHA",
    )
    require_equal(
        runtime_binding.get("state_sha256"),
        EXPECTED_RUNTIME_STATE_SHA256,
        "extended frozen runtime-state SHA",
    )
    for flag in (
        "invocation_ids_bound",
        "exec_start_gpu_mapping_bound",
        "no_restarts_at_capture",
    ):
        if runtime_binding.get(flag) is not True:
            raise ValueError(f"extended runtime binding {flag} did not pass")
    extended_checks = require_mapping(
        extended.get("checks_passed"), "extended.checks_passed"
    )
    require_exact_keys(
        extended_checks, EXTENDED_INTEGRITY_FLAGS, "extended.checks_passed"
    )
    if any(value is not True for value in extended_checks.values()):
        raise ValueError("One or more extended-integrity gates did not pass")
    limitations = require_mapping(extended.get("limitations"), "extended.limitations")
    if limitations.get("single_dataset_single_seed_screening_only") is not True:
        raise ValueError("extended audit lacks the single-dataset/single-seed limitation")
    if limitations.get("mainline_decision_not_made_by_this_audit") is not True:
        raise ValueError("extended audit unexpectedly claims a mainline decision")
    require_equal(
        extended.get("cross_variant_consistency"),
        comparison.get("integrity_audit"),
        "extended/comparison cross-variant consistency",
    )
    frozen = require_mapping(
        extended.get("frozen_sources_and_data"), "extended.frozen_sources_and_data"
    )
    require_exact_keys(
        frozen,
        {"source_sha256", "training_data_sha256", "environment"},
        "extended.frozen_sources_and_data",
    )
    source_hashes = require_mapping(
        frozen.get("source_sha256"), "extended frozen source SHA"
    )
    require_exact_keys(
        source_hashes, EXTENDED_SOURCE_PATHS, "extended frozen source SHA"
    )
    for label, path in EXTENDED_SOURCE_PATHS.items():
        reject_symlink_components(path, f"extended frozen source {label}")
        require_regular(path, f"extended frozen source {label}")
        require_equal(
            sha256_file(path),
            source_hashes[label],
            f"current frozen source SHA {label}",
        )
        require_equal(
            source_hashes[label],
            EXPECTED_PRODUCER_SOURCE_SHA256[label],
            f"recorded frozen producer source SHA {label}",
        )
    require_equal(
        source_hashes.get("sweep_aggregator"),
        aggregate_evidence["aggregator_sha256"],
        "extended/aggregate sweep-aggregator SHA",
    )
    require_equal(
        frozen.get("training_data_sha256"),
        extended.get("training_data_sha256"),
        "extended training-data SHA",
    )
    per_variant = require_mapping(extended.get("per_variant"), "extended.per_variant")
    require_exact_keys(per_variant, VARIANTS, "extended.per_variant")
    for variant in VARIANTS:
        variant_audit = require_mapping(
            per_variant[variant], f"extended.per_variant.{variant}"
        )
        require_exact_keys(
            variant_audit,
            EXTENDED_VARIANT_KEYS,
            f"extended.per_variant.{variant}",
        )
        require_equal(
            variant_audit.get("gpu_uuid"),
            GPU_UUIDS[variant],
            f"extended {variant} GPU UUID",
        )
        require_equal(
            variant_audit.get("invocation_id"),
            INVOCATION_IDS[variant],
            f"extended {variant} invocation ID",
        )
        event_audit = require_mapping(
            variant_audit.get("event_audit"),
            f"extended.per_variant.{variant}.event_audit",
        )
        require_equal(
            event_audit.get("event_count"),
            expected_epochs,
            f"extended {variant} event count",
        )
        require_equal(
            event_audit.get("epoch_range"),
            [1, expected_epochs],
            f"extended {variant} epoch range",
        )
        require_equal(
            event_audit.get("processed_train_samples_each_epoch"),
            530,
            f"extended {variant} processed samples per epoch",
        )
        if event_audit.get("learning_rate_schedule_exact") is not True:
            raise ValueError(f"extended {variant} LR schedule was not exact")
        if event_audit.get("online_best_flags_exact") is not True:
            raise ValueError(f"extended {variant} online best flags were not exact")
        require_equal(
            event_audit.get("recomputed_best_pd_epoch"),
            comparison_rows[variant].get("pd_best_epoch"),
            f"extended {variant} recomputed Pd epoch",
        )
        require_equal(
            event_audit.get("recomputed_best_miou_epoch"),
            comparison_rows[variant].get("miou_best_epoch"),
            f"extended {variant} recomputed mIoU epoch",
        )
        require_equal(
            variant_audit.get("best_pd_epoch"),
            comparison_rows[variant].get("pd_best_epoch"),
            f"extended {variant} best Pd epoch",
        )
        require_equal(
            variant_audit.get("best_miou_epoch"),
            comparison_rows[variant].get("miou_best_epoch"),
            f"extended {variant} best mIoU epoch",
        )
        last_audit = require_mapping(
            variant_audit.get("last_checkpoint_audit"),
            f"extended.per_variant.{variant}.last_checkpoint_audit",
        )
        require_equal(
            last_audit.get("role"),
            "last_evaluated_epoch",
            f"extended {variant} last-checkpoint role",
        )
        require_equal(
            last_audit.get("epoch"),
            expected_epochs,
            f"extended {variant} last-checkpoint epoch",
        )
        for flag in (
            "strict_load",
            "all_checkpoint_tensors_finite",
            "metrics_equal_final_event",
        ):
            if last_audit.get(flag) is not True:
                raise ValueError(f"extended {variant} last-checkpoint {flag} failed")
        require_sha256(
            last_audit.get("sha256"), f"extended {variant} last-checkpoint SHA"
        )
        artifact_sha = require_mapping(
            variant_audit.get("artifact_sha256"),
            f"extended.per_variant.{variant}.artifact_sha256",
        )
        require_exact_keys(
            artifact_sha,
            EXTENDED_ARTIFACT_KEYS,
            f"extended.per_variant.{variant}.artifact_sha256",
        )
        for name, digest in artifact_sha.items():
            require_sha256(digest, f"extended {variant} artifact {name}")
        run_dir = root / dataset / variant / run_name
        current_extended_artifacts = {
            "protocol.json": run_dir / "protocol.json",
            "split.json": run_dir / "split.json",
            "metrics.jsonl": run_dir / "metrics.jsonl",
            "summary.json": run_dir / "summary.json",
            "best.pth.tar": run_dir / "best.pth.tar",
            "best_miou.pth.tar": run_dir / "best_miou.pth.tar",
            "last.pth.tar": run_dir / "last.pth.tar",
            "launch_manifest": root / "launch" / f"{variant}.json",
            "worker_log": root / "logs" / f"{variant}.log",
        }
        for name, path in current_extended_artifacts.items():
            reject_symlink_components(path, f"{variant} extended artifact {name}")
            require_regular(path, f"{variant} current extended artifact {name}")
            require_equal(
                sha256_file(path),
                artifact_sha[name],
                f"{variant} current extended artifact SHA {name}",
            )
        require_equal(
            artifact_sha.get("best.pth.tar"),
            checkpoint_sha[variant]["best.pth.tar"],
            f"extended/comparison {variant} best checkpoint SHA",
        )
        require_equal(
            artifact_sha.get("best_miou.pth.tar"),
            checkpoint_sha[variant]["best_miou.pth.tar"],
            f"extended/comparison {variant} mIoU checkpoint SHA",
        )
        require_equal(
            artifact_sha["split.json"],
            EXPECTED_SPLIT_ARTIFACT_SHA256,
            f"extended {variant} frozen split SHA",
        )
        require_equal(
            last_audit.get("sha256"),
            artifact_sha["last.pth.tar"],
            f"extended {variant} last-checkpoint audit/artifact SHA",
        )
        aggregate_artifacts = require_mapping(
            source_sweeps[variant].get("artifact_sha256"),
            f"aggregate {variant} artifact SHA",
        )
        for name in ("protocol.json", "split.json", "summary.json", "metrics.jsonl"):
            require_equal(
                artifact_sha[name],
                aggregate_artifacts[name],
                f"extended/aggregate {variant} {name} SHA",
            )
        require_equal(
            artifact_sha["best.pth.tar"],
            aggregate_artifacts["checkpoint"],
            f"extended/aggregate {variant} best checkpoint SHA",
        )
    validate_official_test_isolation(
        extended.get("official_test_isolation_evidence")
    )


def point_key(point: Mapping[str, Any] | None) -> Tuple[float, ...]:
    """Cross-method order; calibration threshold is not a performance advantage."""
    if point is None:
        return (0.0,)
    return (
        1.0,
        float(point["pd"]),
        -float(point["fa"]),
        float(point["tiny_pd"]),
        float(point["miou"]),
    )


def comparison_label(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> str:
    left_key = point_key(left)
    right_key = point_key(right)
    if left_key > right_key:
        return "better"
    if left_key < right_key:
        return "worse"
    return "equal"


def fixed_dominates(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> bool:
    no_worse = (
        float(candidate["pd"]) >= float(reference["pd"])
        and float(candidate["tiny_pd"]) >= float(reference["tiny_pd"])
        and float(candidate["fa"]) <= float(reference["fa"])
    )
    strict = (
        float(candidate["pd"]) > float(reference["pd"])
        or float(candidate["tiny_pd"]) > float(reference["tiny_pd"])
        or float(candidate["fa"]) < float(reference["fa"])
    )
    return no_worse and strict


def decide(
    fixed: Mapping[str, Dict[str, Any]],
    operating: Mapping[str, Mapping[str, Dict[str, Any] | None]],
    pareto: Mapping[str, Any],
) -> Dict[str, Any]:
    budget_rows = []
    tpd_beats_original = []
    zero_detection_availability_advantages = []
    covered = []
    never_weaker_than_controls = True
    for budget_key in FORMAL_FA_BUDGET_KEYS:
        points = operating[budget_key]
        tpd_point = points["tpd"]
        tpd_vs_original = comparison_label(tpd_point, points["original"])
        control_labels = {
            control: comparison_label(tpd_point, points[control])
            for control in STRONG_CONTROLS
        }
        raw_tpd_advantage = tpd_vs_original == "better"
        zero_detection_availability_advantage = (
            raw_tpd_advantage
            and tpd_point is not None
            and float(tpd_point["pd"]) == 0.0
        )
        tpd_advantage = (
            raw_tpd_advantage and not zero_detection_availability_advantage
        )
        covering_controls = [
            control
            for control in STRONG_CONTROLS
            if point_key(points[control]) >= point_key(tpd_point)
        ]
        is_covered = tpd_advantage and bool(covering_controls)
        if tpd_advantage:
            tpd_beats_original.append(budget_key)
        if zero_detection_availability_advantage:
            zero_detection_availability_advantages.append(budget_key)
        if is_covered:
            covered.append(budget_key)
        if any(label == "worse" for label in control_labels.values()):
            never_weaker_than_controls = False
        budget_rows.append(
            {
                "fa_budget": budget_key,
                "tpd_vs_original": tpd_vs_original,
                "tpd_vs_progressive": control_labels["progressive"],
                "tpd_vs_spd": control_labels["spd"],
                "tpd_advantage_over_original": tpd_advantage,
                "zero_detection_availability_advantage": (
                    zero_detection_availability_advantage
                ),
                "strong_controls_reproducing_or_exceeding_tpd": covering_controls,
                "tpd_advantage_covered": is_covered,
                "points": {variant: points[variant] for variant in VARIANTS},
            }
        )

    fixed_dominators = [
        control
        for control in STRONG_CONTROLS
        if fixed_dominates(fixed[control], fixed["tpd"])
    ]
    exclusive_tpd_rows = [
        row
        for row in pareto["coordinates"]
        if row.get("owners") == ["tpd"]
    ]
    exclusive_tpd_pareto = bool(exclusive_tpd_rows)
    fixed_not_dominated = not fixed_dominators
    all_advantages_covered = bool(tpd_beats_original) and set(covered) == set(
        tpd_beats_original
    )

    advance_conditions = {
        "tpd_beats_original_at_one_or_more_formal_budgets": bool(
            tpd_beats_original
        ),
        "no_tpd_advantage_is_reproduced_or_exceeded_by_a_strong_control": not covered,
        "tpd_is_never_weaker_than_progressive_or_spd_at_any_formal_budget": (
            never_weaker_than_controls
        ),
        "tpd_owns_at_least_one_exclusive_global_pd_fa_pareto_coordinate": (
            exclusive_tpd_pareto
        ),
        "tpd_fixed_threshold_0_5_is_not_componentwise_dominated_by_a_strong_control": (
            fixed_not_dominated
        ),
    }
    if all(advance_conditions.values()):
        decision = "ADVANCE_TPD_TO_MULTI_SEED"
        reason_codes = ["ALL_CONSERVATIVE_ADVANCE_CONDITIONS_PASSED"]
    elif not tpd_beats_original and zero_detection_availability_advantages:
        decision = "INCONCLUSIVE_MIXED_TRADEOFF"
        reason_codes = ["ONLY_ZERO_DETECTION_AVAILABILITY_ADVANTAGE"]
    elif not tpd_beats_original:
        decision = "DO_NOT_ESTABLISH_CURRENT_TPD_CORE"
        reason_codes = ["NO_TPD_ADVANTAGE_OVER_ORIGINAL_AT_ANY_FORMAL_FA_BUDGET"]
    elif all_advantages_covered:
        decision = "DO_NOT_ESTABLISH_CURRENT_TPD_CORE"
        reason_codes = [
            "EVERY_TPD_ADVANTAGE_OVER_ORIGINAL_REPRODUCED_OR_EXCEEDED_BY_STRONG_CONTROL"
        ]
    else:
        decision = "INCONCLUSIVE_MIXED_TRADEOFF"
        reason_codes = [
            key
            for key, passed in advance_conditions.items()
            if not passed
        ]
    if decision not in VALID_DECISIONS:
        raise AssertionError(f"Unexpected decision: {decision}")
    return {
        "decision": decision,
        "reason_codes": reason_codes,
        "advance_conditions": advance_conditions,
        "tpd_better_than_original_budget_keys": tpd_beats_original,
        "zero_detection_availability_advantage_budget_keys": (
            zero_detection_availability_advantages
        ),
        "tpd_advantage_covered_budget_keys": covered,
        "fixed_threshold_0_5_strong_control_dominators": fixed_dominators,
        "exclusive_tpd_global_pareto_coordinates": exclusive_tpd_rows,
        "budget_comparisons": budget_rows,
    }


def markdown_report(payload: Mapping[str, Any]) -> str:
    screening = payload["screening_decision"]
    lines = [
        "# TPD-SCTransNet seed-42 mainline screening decision",
        "",
        f"- Decision: **{screening['decision']}**",
        "- Paper core established: **false**",
        "- Stability claim supported: **false**",
        "- Official test accessed: **false**",
        "",
        "## Policy boundary",
        "",
        "This is a post-hoc conservative operational policy applied to sealed "
        "single-dataset, single-seed internal-validation evidence. It was not "
        "natively bound into the four-RTX-5090 launch. A positive screening result "
        "only advances TPD to multi-seed and multi-dataset confirmation.",
        "",
        "The legacy `PREREGISTRATION.md` mechanism-interpretation gate is recorded "
        "as scientific context only. Its two-GPU serial queue was superseded by the "
        "separately recorded four-GPU parallel execution; neither this decision "
        "policy nor that execution is presented as natively hash-bound by the old "
        "document.",
        "",
        "## Formal Fa-budget comparisons",
        "",
        "| Fa budget | TPD vs Original | TPD vs Progressive | TPD vs SPD | "
        "TPD advantage covered |",
        "|---:|---|---|---|---|",
    ]
    for row in screening["budget_comparisons"]:
        covered = ", ".join(row["strong_controls_reproducing_or_exceeding_tpd"]) or "no"
        lines.append(
            f"| {row['fa_budget']} | {row['tpd_vs_original']} | "
            f"{row['tpd_vs_progressive']} | {row['tpd_vs_spd']} | {covered} |"
        )
    lines += [
        "",
        "Cross-method ordering is lexicographic: available point, higher Pd, "
        "lower actual Fa, higher tiny-Pd, then higher mIoU; an availability-only "
        "zero-Pd advantage is explicitly barred from advancing TPD. "
        "Threshold distance from 0.5 is used only to select a point within one "
        "variant's curve and never creates a cross-method advantage.",
        "",
        "## Conservative advance conditions",
        "",
    ]
    for name, passed in screening["advance_conditions"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines += [
        "",
        "## Reason codes",
        "",
    ]
    for reason in screening["reason_codes"]:
        lines.append(f"- `{reason}`")
    lines += [
        "",
        "## Evidence boundary",
        "",
        "The inputs passed the sealed comparison marker, extended-integrity marker, "
        "Pd–Fa aggregate completion marker, the aggregate's exact 17 integrity "
        "flags, and the extended audit's exact integrity checks. All four current "
        "sweep curves were parsed to recompute the five budget points, threshold-0.5 "
        "points, and joint Pareto frontier. Referenced run artifacts and frozen "
        "sources were re-hashed. The frozen extended auditor was independently "
        "re-executed and its output matched the sealed audit byte-for-byte; that "
        "subprocess strictly loaded checkpoints without inference. The "
        "official-training-only data fingerprint was recomputed before and after "
        "publication. No official-test file was read.",
    ]
    return "\n".join(lines) + "\n"


def collect_input_snapshot(paths: Sequence[Path]) -> Dict[str, str]:
    snapshot: Dict[str, str] = {}
    for path in paths:
        require_regular(path, f"decision input {path.name}")
        resolved = str(path.resolve())
        if resolved in snapshot:
            continue
        snapshot[resolved] = sha256_file(path)
    return snapshot


def verify_input_snapshot(snapshot: Mapping[str, str]) -> None:
    for name, expected in snapshot.items():
        path = Path(name)
        require_regular(path, f"decision input snapshot {path.name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Decision input changed during evaluation: {path}; "
                f"expected={expected}, actual={actual}"
            )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_decision_outputs(
    payload_paths: Sequence[Path], marker_path: Path
) -> None:
    entries = [(path, path.name) for path in payload_paths]
    verify_exact_manifest(marker_path, entries, "decision completion marker")


def write_outputs_atomically(
    payloads: Mapping[Path, str],
    marker_path: Path,
    input_snapshot: Mapping[str, str],
    expected_training_data_fingerprint: str | None = None,
) -> None:
    payload_paths = list(payloads)
    if not payload_paths:
        raise ValueError("No decision payloads were provided")
    output_dir = marker_path.parent
    if any(path.parent != output_dir for path in payload_paths):
        raise ValueError("Decision payloads and marker must share one directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    require_directory(output_dir, "decision output directory")
    lock_path = output_dir / f".{OUTPUT_STEM}.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        raise ValueError(f"Decision lock target is unsafe: {lock_path}")

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another decision process holds {lock_path}") from exc
        targets = [*payload_paths, marker_path]
        existing = [
            str(path) for path in targets if path.exists() or path.is_symlink()
        ]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing decision outputs: {existing}"
            )
        for path in targets:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"Unsafe decision output target: {path}")

        def verify_training_data_fingerprint() -> None:
            if expected_training_data_fingerprint is None:
                return
            require_equal(
                recompute_training_data_fingerprint(EXPECTED_DATASET),
                expected_training_data_fingerprint,
                "pre/post-publication training-data fingerprint",
            )

        verify_input_snapshot(input_snapshot)
        verify_training_data_fingerprint()
        stage_dir = Path(
            tempfile.mkdtemp(prefix=f".{OUTPUT_STEM}.staging.", dir=output_dir)
        )
        published: list[Tuple[Path, Path]] = []
        try:
            staged_payloads = []
            for destination, content in payloads.items():
                staged = stage_dir / destination.name
                with staged.open("x", encoding="utf-8", newline="") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                staged_payloads.append(staged)
            staged_marker = stage_dir / marker_path.name
            marker_text = "".join(
                manifest_line(sha256_file(path), path.name)
                for path in staged_payloads
            )
            with staged_marker.open("x", encoding="utf-8", newline="") as handle:
                handle.write(marker_text)
                handle.flush()
                os.fsync(handle.fileno())
            verify_decision_outputs(staged_payloads, staged_marker)
            fsync_directory(stage_dir)
            verify_input_snapshot(input_snapshot)
            verify_training_data_fingerprint()
            for destination, staged in zip(payload_paths, staged_payloads):
                os.link(staged, destination, follow_symlinks=False)
                published.append((destination, staged))
            fsync_directory(output_dir)
            verify_input_snapshot(input_snapshot)
            verify_training_data_fingerprint()
            os.link(staged_marker, marker_path, follow_symlinks=False)
            published.append((marker_path, staged_marker))
            try:
                fsync_directory(output_dir)
                verify_decision_outputs(payload_paths, marker_path)
                verify_input_snapshot(input_snapshot)
                verify_training_data_fingerprint()
            except Exception:
                raise
        except Exception:
            for destination, staged in reversed(published):
                try:
                    if destination.exists() and destination.samefile(staged):
                        destination.unlink()
                except (FileNotFoundError, OSError):
                    pass
            fsync_directory(output_dir)
            raise
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)


def build_decision_payload(
    root: Path,
    dataset: str,
    run_name: str,
    expected_epochs: int,
) -> Tuple[Dict[str, Any], Sequence[Path]]:
    root = root.resolve()
    require_directory(root, "formal result root")
    dataset_dir = root / dataset
    reject_symlink_components(dataset_dir, "formal dataset directory")
    require_directory(dataset_dir, "formal dataset directory")
    comparison_dir = dataset_dir / "comparison"
    reject_symlink_components(comparison_dir, "sealed comparison directory")
    if comparison_dir.is_symlink():
        raise ValueError(f"Sealed comparison directory must not be a symlink: {comparison_dir}")
    if not comparison_dir.exists():
        raise PendingEvidenceError(
            f"Sealed comparison directory has not been published: {comparison_dir}"
        )
    require_directory(comparison_dir, "sealed comparison directory")

    comparison_path = comparison_dir / f"{run_name}.json"
    comparison_md = comparison_dir / f"{run_name}.md"
    comparison_csv = comparison_dir / f"{run_name}.csv"
    sweeps_manifest = comparison_dir / "SWEEPS.sha256"
    complete_marker = comparison_dir / "COMPLETE.sha256"
    extended_path = comparison_dir / "extended_integrity_v1.json"
    extended_marker = comparison_dir / "EXTENDED_COMPLETE.sha256"

    aggregate_stem = f"pd_fa_{run_name}"
    aggregate_path = comparison_dir / f"{aggregate_stem}.json"
    aggregate_md = comparison_dir / f"{aggregate_stem}.md"
    operating_csv = comparison_dir / f"{aggregate_stem}_operating_points.csv"
    curves_csv = comparison_dir / f"{aggregate_stem}_curves.csv"
    aggregate_marker = comparison_dir / f"{aggregate_stem}.COMPLETE.sha256"
    runtime_state_path = root / "launch" / RUNTIME_STATE_NAME
    sealed_input_paths = (
        comparison_path,
        comparison_md,
        comparison_csv,
        sweeps_manifest,
        complete_marker,
        extended_path,
        extended_marker,
        aggregate_path,
        aggregate_md,
        operating_csv,
        curves_csv,
        aggregate_marker,
    )
    raw_artifact_paths = []
    for variant in VARIANTS:
        run_dir = dataset_dir / variant / run_name
        raw_artifact_paths.extend(
            (
                run_dir / "pd_fa_sweep_best.pth.json",
                run_dir / "protocol.json",
                run_dir / "split.json",
                run_dir / "metrics.jsonl",
                run_dir / "summary.json",
                run_dir / "best.pth.tar",
                run_dir / "best_miou.pth.tar",
                run_dir / "last.pth.tar",
                root / "launch" / f"{variant}.json",
                root / "logs" / f"{variant}.log",
            )
        )
    input_paths = tuple(
        dict.fromkeys(
            (
                *sealed_input_paths,
                *raw_artifact_paths,
                runtime_state_path,
                *PRODUCER_SOURCE_PATHS.values(),
                OFFICIAL_TRAIN_INDEX_PATH,
                LEGACY_SCIENTIFIC_GATE_PATH,
                MAINLINE_CONTEXT_PATH,
                Path(__file__).resolve(),
            )
        )
    )

    require_committed_group(
        complete_marker,
        (comparison_path, comparison_md, comparison_csv, sweeps_manifest),
        "base comparison",
    )
    require_committed_group(
        extended_marker,
        (complete_marker, extended_path),
        "extended-integrity audit",
    )
    require_committed_group(
        aggregate_marker,
        (aggregate_path, aggregate_md, operating_csv, curves_csv),
        "Pd--Fa aggregate",
    )
    for path in (
        *raw_artifact_paths,
        runtime_state_path,
        *PRODUCER_SOURCE_PATHS.values(),
        OFFICIAL_TRAIN_INDEX_PATH,
        LEGACY_SCIENTIFIC_GATE_PATH,
        MAINLINE_CONTEXT_PATH,
        Path(__file__).resolve(),
    ):
        reject_symlink_components(path, f"decision-bound input {path.name}")
        require_regular(path, f"decision-bound input {path.name}")
    require_equal(
        sha256_file(LEGACY_SCIENTIFIC_GATE_PATH),
        LEGACY_SCIENTIFIC_GATE_SHA256,
        "legacy scientific interpretation gate SHA",
    )
    require_equal(
        sha256_file(MAINLINE_CONTEXT_PATH),
        MAINLINE_CONTEXT_SHA256,
        "mainline research-context SHA",
    )
    producer_source_sha256 = validate_frozen_producer_sources()
    require_equal(
        sha256_file(runtime_state_path),
        EXPECTED_RUNTIME_STATE_SHA256,
        "frozen runtime-state SHA",
    )
    initial_input_snapshot = collect_input_snapshot(input_paths)
    verify_exact_manifest(
        complete_marker,
        (
            (comparison_path, comparison_path.name),
            (comparison_md, comparison_md.name),
            (comparison_csv, comparison_csv.name),
            (sweeps_manifest, sweeps_manifest.name),
        ),
        "base comparison completion marker",
    )
    verify_exact_manifest(
        extended_marker,
        (
            (complete_marker, complete_marker.name),
            (extended_path, extended_path.name),
        ),
        "extended-integrity completion marker",
    )
    verify_exact_manifest(
        aggregate_marker,
        (
            (aggregate_path, aggregate_path.name),
            (aggregate_md, aggregate_md.name),
            (operating_csv, operating_csv.name),
            (curves_csv, curves_csv.name),
        ),
        "Pd--Fa aggregate completion marker",
    )
    training_data_fingerprint = recompute_training_data_fingerprint(dataset)

    comparison = read_json(comparison_path, "sealed comparison JSON")
    comparison_rows = validate_comparison(
        comparison, dataset, run_name, expected_epochs
    )
    aggregate = read_json(aggregate_path, "sealed Pd--Fa aggregate JSON")
    aggregate_evidence = validate_aggregate(
        aggregate,
        aggregate_path,
        aggregate_marker,
        comparison_dir,
        comparison_path,
        complete_marker,
        sweeps_manifest,
        dataset,
        run_name,
        expected_epochs,
    )
    extended = read_json(extended_path, "sealed extended-integrity JSON")
    sealed_extended_raw = extended_path.read_bytes()
    recomputed_extended, recomputed_extended_raw = recompute_extended_audit(
        root,
        dataset,
        run_name,
        expected_epochs,
        runtime_state_path,
    )
    if recomputed_extended_raw != sealed_extended_raw:
        raise ValueError(
            "Independently recomputed extended audit is not byte-identical "
            "to the sealed extended audit"
        )
    require_equal(
        recomputed_extended,
        extended,
        "independently recomputed/sealed extended audit structure",
    )
    validate_cross_bindings(
        comparison,
        comparison_rows,
        aggregate,
        aggregate_evidence,
        extended,
        root,
        dataset,
        run_name,
        expected_epochs,
    )
    screening = decide(
        aggregate_evidence["fixed"],
        aggregate_evidence["operating"],
        aggregate_evidence["pareto"],
    )
    script_path = Path(__file__).resolve()
    verify_input_snapshot(initial_input_snapshot)
    payload = {
        "schema_version": "tpd-mainline-seed42-decision-v1",
        "dataset": dataset,
        "run_name": run_name,
        "expected_epochs": expected_epochs,
        "seed": EXPECTED_SEED,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "scope": "single_dataset_single_seed_screening_only",
        "policy": {
            "name": "post_hoc_conservative_operational_policy_v2",
            "status": "post_hoc_not_launch_preregistered",
            "four_rtx5090_launch_natively_bound_this_policy": False,
            "scientific_interpretation_context": {
                "legacy_preregistration_path": str(
                    LEGACY_SCIENTIFIC_GATE_PATH.resolve()
                ),
                "legacy_preregistration_sha256": LEGACY_SCIENTIFIC_GATE_SHA256,
                "inheritance_scope": (
                    "mechanism interpretation gate only; this policy and the "
                    "four-GPU execution were not natively hash-bound by that document"
                ),
                "superseded_logistics": (
                    "the legacy two-GPU serial queue was replaced by the separately "
                    "recorded four-RTX-5090 parallel launch"
                ),
                "mainline_context_path": str(MAINLINE_CONTEXT_PATH.resolve()),
                "mainline_context_sha256": MAINLINE_CONTEXT_SHA256,
                "not_claimed_as_this_launch_preregistration": True,
            },
            "invalid_evidence_behavior": (
                "fail closed with no decision artifact; invalid evidence is not "
                "converted into one of the three scientific screening states"
            ),
            "valid_input_decision_states": sorted(VALID_DECISIONS),
            "formal_fa_budgets": list(FORMAL_FA_BUDGETS),
            "curve_operating_point_selection_order": [
                *OPERATING_POINT_RULE,
            ],
            "cross_method_operating_point_order": [
                "available point before missing point",
                *OPERATING_POINT_RULE[:-1],
            ],
            "zero_detection_rule": (
                "an apparent advantage with TPD Pd == 0 is not an advance signal "
                "and yields ONLY_ZERO_DETECTION_AVAILABILITY_ADVANTAGE when it is "
                "the only apparent advantage"
            ),
            "fixed_threshold_dominance": (
                "candidate Pd >= reference Pd, tiny-Pd >= reference tiny-Pd, "
                "Fa <= reference Fa, with at least one strict inequality"
            ),
            "advance_semantics": (
                "advance to multi-seed/multi-dataset confirmation only; never "
                "establish the paper core from this screen"
            ),
        },
        "evidence_gate": {
            "base_comparison_complete_marker_verified": True,
            "extended_integrity_complete_marker_verified": True,
            "pd_fa_aggregate_complete_marker_verified": True,
            "aggregate_integrity_flags_exact_17_all_true": True,
            "extended_integrity_flags_exact_all_true": True,
            "formal_five_fa_budgets_exact": True,
            "comparison_aggregate_extended_cross_bindings_verified": True,
            "extended_audit_independently_reexecuted_byte_identical": True,
            "current_sweep_curves_recomputed_exact": True,
            "current_raw_artifact_hashes_recomputed_exact": True,
            "current_frozen_source_hashes_recomputed_exact": True,
            "training_data_fingerprint_independently_recomputed": True,
        },
        "screening_decision": screening,
        "fixed_threshold_0_5": aggregate_evidence["fixed"],
        "global_pareto_summary": {
            "unique_coordinate_count": aggregate_evidence["pareto"][
                "unique_coordinate_count"
            ],
            "owner_coordinate_counts": aggregate_evidence["pareto"][
                "owner_coordinate_counts"
            ],
        },
        "input_provenance": {
            "input_sha256": initial_input_snapshot,
            "comparison_complete_sha256": sha256_file(complete_marker),
            "extended_complete_sha256": sha256_file(extended_marker),
            "aggregate_complete_sha256": sha256_file(aggregate_marker),
            "aggregate_sha256": aggregate_evidence["aggregate_sha256"],
            "runtime_state_sha256": EXPECTED_RUNTIME_STATE_SHA256,
            "training_data_fingerprint_sha256": training_data_fingerprint,
            "frozen_producer_source_sha256": producer_source_sha256,
            "decision_implementation_path": str(script_path),
            "decision_implementation_sha256": sha256_file(script_path),
        },
        "limitations": {
            "single_dataset": True,
            "single_seed": True,
            "no_effect_size_delta_or_confidence_interval_in_this_screen": True,
            "no_unique_primary_fa_star_beyond_the_five_formal_budgets": True,
            "post_embedding_probe_not_part_of_this_aggregate": True,
            "capacity_matched_e_cap_not_part_of_this_four_variant_screen": True,
            "pilot_results_not_pooled_with_formal_results": True,
            "official_test_not_used": True,
            "checkpoints_loaded_only_by_independent_auditor_without_inference": True,
            "legacy_scientific_gate_not_native_four_gpu_launch_binding": True,
        },
    }
    return payload, input_paths


def main() -> None:
    args = parse_args()
    reject_symlink_components(args.root, "formal result root")
    root = args.root.resolve()
    try:
        payload, input_paths = build_decision_payload(
            root, args.dataset, args.run_name, args.expected_epochs
        )
    except PendingEvidenceError as exc:
        print(f"PENDING_EVIDENCE {exc}", file=sys.stderr, flush=True)
        raise SystemExit(75) from exc
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.validate_only:
        print(json_text, end="")
        return

    comparison_dir = root / args.dataset / "comparison"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else comparison_dir
    )
    reject_symlink_components(
        args.output_dir if args.output_dir is not None else comparison_dir,
        "decision output directory",
    )
    json_path = output_dir / f"{OUTPUT_STEM}.json"
    md_path = output_dir / f"{OUTPUT_STEM}.md"
    marker_path = output_dir / f"{OUTPUT_STEM}.COMPLETE.sha256"
    snapshot = collect_input_snapshot(input_paths)
    require_equal(
        snapshot,
        payload["input_provenance"]["input_sha256"],
        "pre-write decision input snapshot",
    )
    write_outputs_atomically(
        {
            json_path: json_text,
            md_path: markdown_report(payload),
        },
        marker_path,
        snapshot,
        payload["input_provenance"]["training_data_fingerprint_sha256"],
    )
    print(
        f"WROTE {json_path} and {md_path}; COMMITTED {marker_path}; "
        f"DECISION {payload['screening_decision']['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
