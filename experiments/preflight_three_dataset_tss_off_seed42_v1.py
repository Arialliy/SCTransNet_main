#!/usr/bin/env python3
"""Prepare and seal Gate O1 plus the existing-Original reuse audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import analyze_positive_tss_effective_weights_v1 as effective  # noqa: E402
from experiments import summarize_tss_violation_types_v1 as violations  # noqa: E402
from experiments import tss_off_diagnostic_common_v1 as common  # noqa: E402


SCHEMA = "sctransnet_three_dataset_tss_off_preflight_v1"
ORIGINAL_REUSE_SCHEMA = "sctransnet_three_dataset_original_reuse_audit_v1"
POSITIVE_TRAINING_SCHEMA = "sctransnet_three_dataset_seed42_global_tss_v2/v1"
PAIR_RECORDS = (
    REPO_ROOT
    / "results"
    / "four_dataset_seed42_v1"
    / "manifests"
    / "four_dataset_pair_records_v1.jsonl"
)
PAIR_DATA_GATE = (
    REPO_ROOT
    / "results"
    / "four_dataset_seed42_v1"
    / "manifests"
    / "four_dataset_data_gate_v1.json"
)
CORE_EVALUATOR_RELATIVE = "experiments/evaluate_three_dataset_v2.py"
METRIC_PROTOCOL_RELATIVE = "experiments/four_dataset_evaluation_protocol_v1.py"
NON_TSS_TRAINING_FIELDS = (
    "amp",
    "base_lr",
    "batch_size",
    "min_lr",
    "optimizer",
    "patch_size",
    "precision",
    "schedule",
    "segmentation_loss",
    "warmup_epochs",
    "workers",
)


def _verify_protocol_hash(payload: Mapping[str, Any], path: Path) -> str:
    declared = payload.get("protocol_sha256")
    common.require(isinstance(declared, str) and len(declared) == 64, f"bad protocol SHA: {path}")
    unsigned = dict(payload)
    del unsigned["protocol_sha256"]
    common.require(common.compact_sha256(unsigned) == declared, f"protocol SHA differs: {path}")
    return declared


def _full_data_sha_audit(dataset_root: Path) -> dict[str, Any]:
    gate = common.load_json(PAIR_DATA_GATE)
    expected_records_sha = gate.get("artifact_sha256", {}).get("pair_records")
    observed_records_sha = common.file_sha256(PAIR_RECORDS)
    common.require(observed_records_sha == expected_records_sha, "pair-record artifact SHA differs")
    cache: dict[Path, str] = {}
    record_counts = {dataset: 0 for dataset in common.DATASETS}
    reference_count = 0
    for _, record in common.iter_jsonl(PAIR_RECORDS):
        dataset = record.get("dataset_name")
        if dataset not in common.DATASETS:
            continue
        record_counts[dataset] += 1
        fields = (
            ("image_relpath", "image_sha256"),
            ("raw_mask_relpath", "raw_mask_sha256"),
            ("effective_mask_relpath", "effective_mask_sha256"),
        )
        for path_field, sha_field in fields:
            relative = record.get(path_field)
            expected = record.get(sha_field)
            common.require(isinstance(relative, str) and bool(relative), f"bad {path_field}")
            common.require(isinstance(expected, str) and len(expected) == 64, f"bad {sha_field}")
            path = (Path(dataset_root) / relative).resolve(strict=True)
            common.require(
                path.is_relative_to(Path(dataset_root).resolve(strict=True)),
                f"data path escapes dataset root: {path}",
            )
            if path not in cache:
                cache[path] = common.file_sha256(path)
            common.require(cache[path] == expected, f"data SHA differs: {path}")
            reference_count += 1
    expected_counts = {
        "NUAA-SIRST": 427,
        "NUDT-SIRST": 1327,
        "IRSTD-1K": 1001,
    }
    common.require(record_counts == expected_counts, "three-dataset pair-record counts differ")
    return {
        "status": "complete",
        "dataset_root": str(Path(dataset_root).resolve()),
        "pair_records": common.artifact_record(PAIR_RECORDS),
        "pair_data_gate": common.artifact_record(PAIR_DATA_GATE),
        "record_counts": record_counts,
        "hash_reference_count": reference_count,
        "unique_file_count": len(cache),
        "missing_file_count": 0,
        "sha_mismatch_count": 0,
    }


def _img_idx_and_correction_audit(
    manifest: Mapping[str, Any], dataset_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_records: dict[str, Any] = {}
    for dataset in common.DATASETS:
        index_records[dataset] = {}
        for split in ("train", "test"):
            frozen = manifest["datasets"][dataset]["splits"][split]
            path = Path(dataset_root) / frozen["index_relpath"]
            observed = common.file_sha256(path)
            common.require(
                observed == frozen["file_sha256"],
                f"img_idx SHA differs: {dataset}/{split}",
            )
            index_records[dataset][split] = {
                "path": str(path.resolve()),
                "sha256": observed,
                "ordered_ids_sha256": frozen["ordered_ids_sha256"],
                "count": frozen["count"],
            }
    correction = manifest["corrections"]["NUAA-SIRST::Misc_111"]
    correction_files: dict[str, Any] = {}
    for label, path_field, sha_field in (
        ("image", "image_relpath", "image_sha256"),
        ("raw_mask", "raw_mask_relpath", "raw_mask_sha256"),
        ("corrected_mask", "corrected_mask_relpath", "corrected_mask_sha256"),
    ):
        path = Path(dataset_root) / correction[path_field]
        observed = common.file_sha256(path)
        common.require(
            observed == correction[sha_field],
            f"Misc_111 {label} SHA differs",
        )
        correction_files[label] = {
            "path": str(path.resolve()),
            "sha256": observed,
        }
    return index_records, {
        "correction_id": correction["correction_id"],
        "operation": correction["operation"],
        "raw_mask_preserved": correction["raw_mask_preserved"],
        "files": correction_files,
    }
def _critical_training_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    training = protocol.get("training")
    common.require(isinstance(training, dict), "protocol lacks training contract")
    return {field: training.get(field) for field in NON_TSS_TRAINING_FIELDS}


def _initialization_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    pair = protocol.get("model", {}).get("pair")
    common.require(isinstance(pair, dict), "protocol lacks paired initialization")
    fields = (
        "schema",
        "initialization_mode",
        "paired_initialization",
        "shared_state_bitwise_equal",
        "model_construction_preserves_caller_rng_stream",
        "parent_checkpoint",
        "parent_checkpoint_load_count",
        "optimizer_state_inherited",
        "scheduler_state_inherited",
        "derived_initialization_seed_algorithm",
        "derived_initialization_seeds",
        "original_shared_state_sha256",
        "final_shared_state_sha256",
    )
    contract = {field: pair.get(field) for field in fields}
    common.require(contract["initialization_mode"] == "true_scratch", "initialization is not scratch")
    common.require(contract["paired_initialization"] is True, "paired initialization is disabled")
    common.require(contract["shared_state_bitwise_equal"] is True, "shared initialization differs")
    common.require(
        contract["original_shared_state_sha256"]
        == contract["final_shared_state_sha256"],
        "Original/Final shared state SHA differs",
    )
    common.require(contract["parent_checkpoint"] is None, "warm start is present")
    common.require(contract["parent_checkpoint_load_count"] == 0, "parent load count differs")
    common.require(contract["optimizer_state_inherited"] is False, "optimizer state was inherited")
    common.require(contract["scheduler_state_inherited"] is False, "scheduler state was inherited")
    common.require(
        contract["model_construction_preserves_caller_rng_stream"] is True,
        "model construction changes the training RNG stream",
    )
    return contract


def _validate_metric_rows(metrics_path: Path) -> dict[str, Any]:
    epochs: list[int] = []
    evaluated: list[int] = []
    for _, event in common.iter_jsonl(metrics_path):
        epoch = event.get("epoch")
        common.require(isinstance(epoch, int) and not isinstance(epoch, bool), "Original epoch is malformed")
        epochs.append(epoch)
        if event.get("evaluated") is True:
            evaluated.append(epoch)
        else:
            common.require(event.get("evaluated") is False, "Original evaluated flag is malformed")
    common.require(epochs == list(range(1, 1001)), "Original metrics sequence is incomplete")
    common.require(evaluated == list(range(10, 1001, 10)), "Original eval cadence differs")
    return {
        "epochs": len(epochs),
        "evaluated_epochs": len(evaluated),
        "first_evaluated_epoch": evaluated[0],
        "last_evaluated_epoch": evaluated[-1],
    }


def build_original_reuse_audit(
    positive_root: Path = common.POSITIVE_RESULTS_ROOT,
    *,
    dataset_root: Path = common.DATASET_ROOT,
    protocol_manifest_path: Path = common.DATA_PROTOCOL_MANIFEST,
) -> dict[str, Any]:
    manifest = common.load_json(protocol_manifest_path)
    manifest_sha = common.file_sha256(protocol_manifest_path)
    common.require(manifest.get("dataset_order") == list(common.DATASETS), "manifest dataset order differs")
    img_idx_audit, correction_audit = _img_idx_and_correction_audit(
        manifest, dataset_root
    )
    data_sha_audit = _full_data_sha_audit(dataset_root)
    evaluator_core_sha = common.file_sha256(REPO_ROOT / CORE_EVALUATOR_RELATIVE)
    metric_protocol_sha = common.file_sha256(REPO_ROOT / METRIC_PROTOCOL_RELATIVE)
    records: dict[str, Any] = {}
    common_runtime_sources: dict[str, Any] | None = None

    for dataset in common.DATASETS:
        original_dir = Path(positive_root) / "runs" / dataset / "original" / "seed_42"
        positive_dir = common.positive_run_directory(positive_root, dataset, 0.005)
        protocol_path = original_dir / "protocol.json"
        summary_path = original_dir / "summary.json"
        metrics_path = original_dir / "metrics.jsonl"
        protocol = common.load_json(protocol_path)
        summary = common.load_json(summary_path)
        positive_protocol = common.load_json(positive_dir / "protocol.json")
        protocol_sha = _verify_protocol_hash(protocol, protocol_path)
        _verify_protocol_hash(positive_protocol, positive_dir / "protocol.json")
        for label, payload in (("Original protocol", protocol), ("Original summary", summary)):
            common.require(payload.get("schema") == POSITIVE_TRAINING_SCHEMA, f"{label} schema differs")
            common.require(payload.get("dataset") == dataset, f"{label} dataset differs")
            common.require(payload.get("method") == "original", f"{label} method differs")
        common.require(summary.get("status") == "complete", "Original summary is incomplete")
        common.require(summary.get("seed") == 42, "Original seed differs")
        common.require(summary.get("epochs") == 1000, "Original epoch budget differs")
        common.require(summary.get("protocol_sha256") == protocol_sha, "Original summary protocol SHA differs")
        common.require(protocol.get("training_seed") == 42, "Original protocol seed differs")
        common.require(protocol.get("epochs") == 1000, "Original protocol epochs differ")
        common.require(protocol.get("begin_test") == 10, "Original eval start differs")
        common.require(protocol.get("eval_every") == 10, "Original eval cadence differs")
        common.require(protocol.get("checkpoint_roles") == list(common.CHECKPOINT_ROLES), "Original roles differ")
        common.require(protocol.get("metrics") == positive_protocol.get("metrics"), "Original/Final metric contract differs")
        common.require(protocol.get("normalization") == positive_protocol.get("normalization"), "Original/Final normalization differs")
        common.require(
            protocol.get("normalization")
            == {
                "mean": manifest["normalization"][dataset]["mean"],
                "std": manifest["normalization"][dataset]["std"],
            },
            "Original normalization differs from manifest",
        )
        common.require(
            protocol.get("three_dataset_v2_data_protocol")
            == positive_protocol.get("three_dataset_v2_data_protocol"),
            "Original/Final data binding differs",
        )
        binding = protocol["three_dataset_v2_data_protocol"]
        common.require(binding.get("manifest_sha256") == manifest_sha, "Original manifest SHA differs")
        common.require(
            _critical_training_contract(protocol)
            == _critical_training_contract(positive_protocol),
            "Original/Final optimizer, scheduler, or augmentation-facing training contract differs",
        )
        initialization = _initialization_contract(protocol)
        common.require(
            initialization == _initialization_contract(positive_protocol),
            "Original/Final paired initialization contract differs",
        )
        runtime_sources = protocol.get("runtime_sources")
        common.require(isinstance(runtime_sources, dict), "Original lacks runtime sources")
        common.require(runtime_sources == positive_protocol.get("runtime_sources"), "Original/Final runtime source lock differs")
        if common_runtime_sources is None:
            common_runtime_sources = runtime_sources
        else:
            common.require(runtime_sources == common_runtime_sources, "Original source locks differ by dataset")
        for source_name, entry in runtime_sources.items():
            common.require(isinstance(entry, dict), f"malformed Original source: {source_name}")
            source_path = Path(str(entry.get("path", "")))
            common.require(common.file_sha256(source_path) == entry.get("sha256"), f"Original source changed: {source_name}")
        metric_rows = _validate_metric_rows(metrics_path)

        evaluations: dict[str, Any] = {}
        for role in common.CHECKPOINT_ROLES:
            evaluation_path = original_dir / "evaluations" / f"{role}.json"
            evaluation = common.load_json(evaluation_path)
            common.require(evaluation.get("status") == "complete", "Original evaluation is incomplete")
            common.require(evaluation.get("dataset") == dataset, "Original evaluation dataset differs")
            common.require(evaluation.get("method") == "original", "Original evaluation method differs")
            common.require(evaluation.get("checkpoint_role") == role, "Original evaluation role differs")
            sources = evaluation.get("source_sha256")
            common.require(isinstance(sources, dict), "Original evaluation lacks source lock")
            common.require(sources.get(CORE_EVALUATOR_RELATIVE) == evaluator_core_sha, "Original evaluator core SHA differs")
            common.require(sources.get(METRIC_PROTOCOL_RELATIVE) == metric_protocol_sha, "Original metric protocol SHA differs")
            fixed = evaluation.get("fixed_threshold_0_5")
            common.require(isinstance(fixed, dict) and fixed.get("threshold") == 0.5, "Original fixed threshold differs")
            evaluations[role] = common.artifact_record(evaluation_path)

        records[dataset] = {
            "status": "eligible_for_reuse",
            "original_run_directory": str(original_dir.resolve()),
            "paired_positive_reference": str(positive_dir.resolve()),
            "protocol": common.artifact_record(protocol_path),
            "protocol_payload_sha256": protocol_sha,
            "summary": common.artifact_record(summary_path),
            "metrics": common.artifact_record(metrics_path),
            "metric_rows": metric_rows,
            "img_idx": img_idx_audit[dataset],
            "normalization": protocol["normalization"],
            "initialization": initialization,
            "training": _critical_training_contract(protocol),
            "checkpoint_tie_break": protocol["metrics"],
            "evaluations": evaluations,
        }

    common.require(common_runtime_sources is not None, "no Original source contract was found")
    return {
        "schema": ORIGINAL_REUSE_SCHEMA,
        "status": "complete",
        "reuse_decision": "REUSE_EXISTING_ORIGINALS",
        "reuse_is_conditional_on_tss_off_protocol_match": True,
        "retrain_originals_before_tss_off": False,
        "conditions": {
            "img_idx_file_sha": True,
            "all_data_file_sha": True,
            "misc_111_correction_manifest_sha": True,
            "normalization": True,
            "seed": True,
            "initialization_rule": True,
            "augmentation_rng_stream": True,
            "optimizer_scheduler": True,
            "epoch_budget_1000": True,
            "eval_every_10": True,
            "checkpoint_tie_break": True,
            "evaluator_core_sha": True,
        },
        "augmentation_rng_contract": {
            "stateless_seed": "sha256(seed,dataset,epoch,namespaced_id)[:8]_uint64_be",
            "epoch_shuffle_seed": (
                "stable_uint63(seed,dataset,'shuffle',epoch), where each str(part) "
                "is length-prefixed by 8-byte big-endian before SHA-256 and the "
                "first 8 digest bytes are masked to 63 bits"
            ),
            "source": common_runtime_sources["data_protocol"],
            "shuffle_source": common_runtime_sources["training_engine"],
        },
        "evaluator_contract": {
            "core_evaluator_path": CORE_EVALUATOR_RELATIVE,
            "core_evaluator_sha256": evaluator_core_sha,
            "metric_protocol_path": METRIC_PROTOCOL_RELATIVE,
            "metric_protocol_sha256": metric_protocol_sha,
            "tss_off_schema_adapter_may_differ": True,
            "metric_semantics_may_differ": False,
        },
        "protocol_manifest": common.artifact_record(protocol_manifest_path),
        "misc_111_correction": correction_audit,
        "full_data_sha_audit": data_sha_audit,
        "datasets": records,
        "source_sha256": {
            "experiments/preflight_three_dataset_tss_off_seed42_v1.py": common.file_sha256(Path(__file__)),
            "experiments/tss_off_diagnostic_common_v1.py": common.file_sha256(Path(common.__file__)),
        },
    }


def prepare_gate_o1(
    positive_root: Path = common.POSITIVE_RESULTS_ROOT,
    *,
    output_dir: Path | None = None,
    dataset_root: Path = common.DATASET_ROOT,
    protocol_manifest_path: Path = common.DATA_PROTOCOL_MANIFEST,
) -> dict[str, Any]:
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(positive_root) / "pre_tss_off_gate_o1"
    )
    effective_payload = effective.build_artifact(positive_root)
    violation_payload = violations.build_artifact(positive_root)
    original_payload = build_original_reuse_audit(
        positive_root,
        dataset_root=dataset_root,
        protocol_manifest_path=protocol_manifest_path,
    )
    effective_path = output_dir / "effective_lambda_summary_v1.json"
    violation_path = output_dir / "violation_matrix_v1.json"
    original_path = output_dir / "original_reuse_audit_v1.json"
    actions = {
        "effective_lambda_summary": common.write_once_or_identical(
            effective_path, effective_payload
        ),
        "violation_matrix": common.write_once_or_identical(
            violation_path, violation_payload
        ),
        "original_reuse_audit": common.write_once_or_identical(
            original_path, original_payload
        ),
    }
    seal = {
        "schema": SCHEMA,
        "status": "complete",
        "gate": "O1",
        "gate_passed": True,
        "positive_root": str(Path(positive_root).resolve()),
        "requirements": {
            "three_positive_lambda_results_complete": effective_payload[
                "gate_o1"
            ]["three_positive_lambda_results_complete"],
            "violation_type_matrix_complete": violation_payload["gate_o1"][
                "violation_type_matrix_complete"
            ],
            "dataset_checkpoint_role_matrix_complete": violation_payload[
                "gate_o1"
            ]["dataset_checkpoint_role_matrix_complete"],
            "effective_lambda_logs_complete": effective_payload["gate_o1"][
                "effective_lambda_logs_complete"
            ],
            "source_lock_complete": effective_payload["gate_o1"][
                "source_lock_complete"
            ],
            "existing_originals_reusable": original_payload["reuse_decision"]
            == "REUSE_EXISTING_ORIGINALS",
        },
        "lambda_005_vs_010_not_fully_identifiable": effective_payload[
            "identifiability"
        ]["lambda_005_vs_010_not_fully_identifiable"],
        "artifacts": {
            "effective_lambda_summary": common.artifact_record(effective_path),
            "violation_matrix": common.artifact_record(violation_path),
            "original_reuse_audit": common.artifact_record(original_path),
        },
        "source_sha256": {
            "experiments/preflight_three_dataset_tss_off_seed42_v1.py": common.file_sha256(Path(__file__)),
            "experiments/analyze_positive_tss_effective_weights_v1.py": common.file_sha256(Path(effective.__file__)),
            "experiments/summarize_tss_violation_types_v1.py": common.file_sha256(Path(violations.__file__)),
            "experiments/tss_off_diagnostic_common_v1.py": common.file_sha256(Path(common.__file__)),
        },
    }
    seal_path = output_dir / "gate_o1_seal_v1.json"
    actions["gate_o1_seal"] = common.write_once_or_identical(seal_path, seal)
    return {
        "status": "complete",
        "gate_passed": True,
        "actions": actions,
        "seal": common.artifact_record(seal_path),
        "artifacts": seal["artifacts"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, default=common.POSITIVE_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument(
        "--data-protocol-manifest",
        type=Path,
        default=common.DATA_PROTOCOL_MANIFEST,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = prepare_gate_o1(
        args.positive_root,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        protocol_manifest_path=args.data_protocol_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
