#!/usr/bin/env python3
"""Seal full-run effective-lambda diagnostics for the nine positive-TSS runs."""

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

from experiments import tss_off_diagnostic_common_v1 as common


SCHEMA = "sctransnet_positive_tss_effective_weights_v1"
TRAINING_SCHEMA = "sctransnet_three_dataset_seed42_global_tss_v2/v1"
RATIO_CAP = 0.10
IDENTIFIABILITY_THRESHOLD = 0.90
PAIR_KEYS = (
    ("0p0025", "0p005"),
    ("0p0025", "0p01"),
    ("0p005", "0p01"),
)


def _finite(value: Any, label: str) -> float:
    common.require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} is not numeric",
    )
    result = float(value)
    common.require(math.isfinite(result), f"{label} is not finite")
    return result


def _weighted_quantile(records: Sequence[tuple[float, int]], quantile: float) -> float:
    common.require(bool(records), "weighted quantile received no records")
    ordered = sorted(records, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    common.require(total > 0, "weighted quantile has zero total weight")
    target = quantile * total
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _close(left: float, right: Any, label: str) -> None:
    observed = _finite(right, label)
    common.require(
        math.isclose(left, observed, rel_tol=1e-12, abs_tol=1e-15),
        f"{label} differs: recomputed={left!r}, stored={observed!r}",
    )


def _verify_protocol_hash(protocol: Mapping[str, Any], path: Path) -> str:
    declared = protocol.get("protocol_sha256")
    common.require(
        isinstance(declared, str) and len(declared) == 64,
        f"protocol SHA is malformed: {path}",
    )
    unsigned = dict(protocol)
    del unsigned["protocol_sha256"]
    common.require(
        common.compact_sha256(unsigned) == declared,
        f"protocol canonical SHA differs: {path}",
    )
    return declared


def _source_contract(protocol: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = protocol.get("runtime_sources")
    common.require(isinstance(value, dict) and bool(value), f"{label} lacks sources")
    return dict(value)


def _aggregate_run(
    run_dir: Path,
    *,
    dataset: str,
    requested_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = run_dir / "protocol.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    protocol = common.load_json(protocol_path)
    summary = common.load_json(summary_path)
    protocol_sha = _verify_protocol_hash(protocol, protocol_path)
    expected_recipe = {
        "method": "final",
        "recipe_id": f"final_lambda_{common.POSITIVE_TOKENS[requested_weight]}",
        "requested_tss_weight": requested_weight,
        "tss_enabled": True,
        "tss_lambda_token": common.POSITIVE_TOKENS[requested_weight],
        "tss_ratio_cap": RATIO_CAP,
    }
    for label, payload in (("protocol", protocol), ("summary", summary)):
        common.require(payload.get("schema") == TRAINING_SCHEMA, f"{label} schema differs")
        common.require(payload.get("dataset") == dataset, f"{label} dataset differs")
        common.require(payload.get("method") == "final", f"{label} method differs")
        common.require(payload.get("recipe") == expected_recipe, f"{label} recipe differs")
    common.require(protocol.get("training_seed") == 42, "training seed differs")
    common.require(protocol.get("epochs") == 1000, "protocol epoch budget differs")
    common.require(protocol.get("eval_every") == 10, "eval cadence differs")
    common.require(protocol.get("begin_test") == 10, "eval start differs")
    common.require(protocol.get("smoke") is False, "positive run is a smoke run")
    common.require(summary.get("status") == "complete", "positive summary incomplete")
    common.require(summary.get("seed") == 42, "summary seed differs")
    common.require(summary.get("epochs") == 1000, "summary epoch budget differs")
    common.require(summary.get("protocol_sha256") == protocol_sha, "summary protocol SHA differs")
    training = protocol.get("training")
    common.require(isinstance(training, dict), "protocol lacks training contract")
    common.require(training.get("tss_ratio_cap") == RATIO_CAP, "ratio cap differs")
    common.require(
        training.get("tss_requested_weight") == requested_weight,
        "requested TSS weight differs",
    )
    expected_samples = int(protocol["dataset_counts"]["train"])

    effective: list[tuple[float, int]] = []
    raw_ratios: list[tuple[float, int]] = []
    effective_ratios: list[tuple[float, int]] = []
    equal_batches = {f"{left}_vs_{right}": 0 for left, right in PAIR_KEYS}
    equal_samples = {f"{left}_vs_{right}": 0 for left, right in PAIR_KEYS}
    cap_batches = 0
    cap_samples = 0
    total_batches = 0
    total_samples = 0
    epochs: list[int] = []

    for line_number, event in common.iter_jsonl(metrics_path):
        epoch = event.get("epoch")
        common.require(isinstance(epoch, int) and not isinstance(epoch, bool), f"bad epoch at line {line_number}")
        epochs.append(epoch)
        records = event.get("train_tss_batch_diagnostics")
        common.require(isinstance(records, list) and bool(records), f"missing batch diagnostics at epoch {epoch}")
        epoch_effective: list[tuple[float, int]] = []
        epoch_raw: list[tuple[float, int]] = []
        epoch_effective_ratio: list[tuple[float, int]] = []
        epoch_cap_batches = 0
        epoch_cap_samples = 0
        epoch_samples = 0
        for batch_index, record in enumerate(records):
            common.require(isinstance(record, dict), f"malformed batch record at epoch {epoch}")
            common.require(record.get("batch_index") == batch_index, f"batch index differs at epoch {epoch}")
            sample_count = record.get("sample_count")
            common.require(
                isinstance(sample_count, int)
                and not isinstance(sample_count, bool)
                and sample_count > 0,
                f"invalid sample count at epoch {epoch} batch {batch_index}",
            )
            requested = _finite(record.get("requested_weight"), "batch requested weight")
            _close(requested_weight, requested, "batch requested weight")
            value = _finite(record.get("effective_weight"), "effective weight")
            common.require(-1e-15 <= value <= requested_weight + 1e-12, "effective weight exceeds request")
            raw_ratio = _finite(record.get("raw_weighted_to_seg_ratio"), "raw TSS ratio")
            effective_ratio = _finite(
                record.get("effective_weighted_to_seg_ratio"),
                "effective TSS ratio",
            )
            common.require(effective_ratio <= RATIO_CAP + 1e-6, "effective TSS ratio exceeds cap")
            cap_active = value < requested_weight * (1.0 - 1e-6)
            common.require(record.get("cap_active") is cap_active, "cap-active flag differs")
            counterfactual = record.get("counterfactual_effective_weights")
            common.require(
                isinstance(counterfactual, dict)
                and set(counterfactual) == {"0p0025", "0p005", "0p01"},
                "counterfactual effective-weight set differs",
            )
            for token, counterfactual_value in counterfactual.items():
                _finite(counterfactual_value, f"counterfactual {token}")
            for left, right in PAIR_KEYS:
                key = f"{left}_vs_{right}"
                if counterfactual[left] == counterfactual[right]:
                    equal_batches[key] += 1
                    equal_samples[key] += sample_count
            epoch_effective.append((value, sample_count))
            epoch_raw.append((raw_ratio, sample_count))
            epoch_effective_ratio.append((effective_ratio, sample_count))
            effective.append((value, sample_count))
            raw_ratios.append((raw_ratio, sample_count))
            effective_ratios.append((effective_ratio, sample_count))
            epoch_cap_batches += int(cap_active)
            epoch_cap_samples += sample_count if cap_active else 0
            cap_batches += int(cap_active)
            cap_samples += sample_count if cap_active else 0
            epoch_samples += sample_count
            total_batches += 1
            total_samples += sample_count
        common.require(epoch_samples == expected_samples, f"epoch {epoch} sample count differs")
        epoch_mean = sum(value * weight for value, weight in epoch_effective) / epoch_samples
        epoch_raw_mean = sum(value * weight for value, weight in epoch_raw) / epoch_samples
        epoch_ratio_mean = sum(value * weight for value, weight in epoch_effective_ratio) / epoch_samples
        checks = {
            "train_tss_effective_weight_mean": epoch_mean,
            "train_tss_effective_weight_p10": _weighted_quantile(epoch_effective, 0.10),
            "train_tss_effective_weight_p50": _weighted_quantile(epoch_effective, 0.50),
            "train_tss_effective_weight_p90": _weighted_quantile(epoch_effective, 0.90),
            "train_tss_effective_weight_max": max(value for value, _ in epoch_effective),
            "train_tss_raw_weighted_to_seg_ratio_mean": epoch_raw_mean,
            "train_tss_effective_weighted_to_seg_ratio_mean": epoch_ratio_mean,
            "train_tss_cap_active_batch_fraction": epoch_cap_batches / len(records),
            "train_tss_cap_active_sample_fraction": epoch_cap_samples / epoch_samples,
        }
        for field, recomputed in checks.items():
            _close(recomputed, event.get(field), f"epoch {epoch} {field}")

    common.require(epochs == list(range(1, 1001)), "metrics epoch sequence is incomplete")
    common.require(total_samples == expected_samples * 1000, "full-run sample total differs")
    mean = sum(value * weight for value, weight in effective) / total_samples
    std = math.sqrt(
        sum(weight * (value - mean) ** 2 for value, weight in effective)
        / total_samples
    )
    pairwise = {
        key: {
            "equal_batch_count": equal_batches[key],
            "equal_batch_fraction": equal_batches[key] / total_batches,
            "equal_sample_count": equal_samples[key],
            "equal_sample_fraction": equal_samples[key] / total_samples,
        }
        for key in equal_batches
    }
    record = {
        "dataset": dataset,
        "requested_lambda": requested_weight,
        "epochs": len(epochs),
        "minibatches": total_batches,
        "sample_observations": total_samples,
        "effective_lambda": {
            "mean": mean,
            "p10": _weighted_quantile(effective, 0.10),
            "p50": _weighted_quantile(effective, 0.50),
            "p90": _weighted_quantile(effective, 0.90),
            "std": std,
            "max": max(value for value, _ in effective),
        },
        "cap_active_batch_fraction": cap_batches / total_batches,
        "cap_active_sample_fraction": cap_samples / total_samples,
        "raw_tss_ratio_mean": sum(value * weight for value, weight in raw_ratios) / total_samples,
        "effective_tss_ratio_mean": sum(value * weight for value, weight in effective_ratios) / total_samples,
        "counterfactual_pairwise_equal": pairwise,
        "inputs": {
            "run_directory": str(run_dir.resolve()),
            "protocol": common.artifact_record(protocol_path),
            "summary": common.artifact_record(summary_path),
            "metrics": common.artifact_record(metrics_path),
            "protocol_payload_sha256": protocol_sha,
        },
    }
    return record, _source_contract(protocol, str(run_dir))


def build_artifact(
    positive_root: Path = common.POSITIVE_RESULTS_ROOT,
    *,
    identifiability_threshold: float = IDENTIFIABILITY_THRESHOLD,
) -> dict[str, Any]:
    common.require(
        0.5 < identifiability_threshold <= 1.0,
        "identifiability threshold must be in (0.5, 1]",
    )
    runs: dict[str, Any] = {}
    source_contract: dict[str, Any] | None = None
    dataset_pool: dict[str, dict[str, list[int]]] = {
        dataset: {f"{left}_vs_{right}": [0, 0, 0, 0] for left, right in PAIR_KEYS}
        for dataset in common.DATASETS
    }
    global_pool = {f"{left}_vs_{right}": [0, 0, 0, 0] for left, right in PAIR_KEYS}
    for dataset in common.DATASETS:
        for requested_weight in common.POSITIVE_LAMBDAS:
            run_dir = common.positive_run_directory(
                positive_root, dataset, requested_weight
            )
            record, observed_sources = _aggregate_run(
                run_dir,
                dataset=dataset,
                requested_weight=requested_weight,
            )
            if source_contract is None:
                source_contract = observed_sources
            else:
                common.require(
                    observed_sources == source_contract,
                    "positive runs do not share one runtime source lock",
                )
            key = f"{dataset}__lambda_{common.POSITIVE_TOKENS[requested_weight]}"
            runs[key] = record
            for pair, equality in record["counterfactual_pairwise_equal"].items():
                counts = [
                    equality["equal_batch_count"],
                    record["minibatches"],
                    equality["equal_sample_count"],
                    record["sample_observations"],
                ]
                for index, value in enumerate(counts):
                    dataset_pool[dataset][pair][index] += value
                    global_pool[pair][index] += value
    common.require(source_contract is not None, "no positive runs were found")
    current_sources: dict[str, Any] = {}
    for name, entry in source_contract.items():
        common.require(isinstance(entry, dict), f"malformed source record: {name}")
        path = Path(str(entry.get("path", "")))
        observed = common.file_sha256(path)
        common.require(observed == entry.get("sha256"), f"positive source changed: {name}")
        current_sources[name] = {"path": str(path.resolve()), "sha256": observed}

    def pooled(records: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for pair, (equal_batches, batches, equal_samples, samples) in records.items():
            output[pair] = {
                "equal_batch_count": equal_batches,
                "batch_count": batches,
                "equal_batch_fraction": equal_batches / batches,
                "equal_sample_count": equal_samples,
                "sample_count": samples,
                "equal_sample_fraction": equal_samples / samples,
            }
        return output

    per_dataset = {dataset: pooled(records) for dataset, records in dataset_pool.items()}
    global_pairwise = pooled(global_pool)
    dataset_flags = {
        dataset: per_dataset[dataset]["0p005_vs_0p01"]["equal_batch_fraction"]
        >= identifiability_threshold
        for dataset in common.DATASETS
    }
    not_fully_identifiable = any(dataset_flags.values())
    return {
        "schema": SCHEMA,
        "status": "complete",
        "aggregation": {
            "mean_and_std_weighting": "minibatch_sample_count",
            "std": "sample_weighted_population_std",
            "weighted_quantile": (
                "smallest_value_with_cumulative_sample_weight_ge_q_times_N"
            ),
            "counterfactual_equality": "exact serialized numeric equality",
        },
        "positive_root": str(Path(positive_root).resolve()),
        "datasets": list(common.DATASETS),
        "positive_lambdas": list(common.POSITIVE_LAMBDAS),
        "run_count": len(runs),
        "runs": runs,
        "pairwise_pooled_by_dataset": per_dataset,
        "pairwise_pooled_all_runs": global_pairwise,
        "identifiability": {
            "overwhelming_majority_batch_fraction_threshold": identifiability_threshold,
            "decision_scope": "true_if_any_formal_dataset_meets_threshold",
            "lambda_005_vs_010_not_fully_identifiable_by_dataset": dataset_flags,
            "lambda_005_vs_010_not_fully_identifiable": not_fully_identifiable,
            "claim_limit": (
                "under the current ratio cap, increasing the requested upper "
                "weight did not establish an acceptable global recipe"
            ),
        },
        "gate_o1": {
            "three_positive_lambda_results_complete": True,
            "effective_lambda_logs_complete": True,
            "source_lock_complete": True,
        },
        "positive_runtime_sources": current_sources,
        "source_sha256": {
            "experiments/analyze_positive_tss_effective_weights_v1.py": common.file_sha256(Path(__file__)),
            "experiments/tss_off_diagnostic_common_v1.py": common.file_sha256(Path(common.__file__)),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, default=common.POSITIVE_RESULTS_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            common.POSITIVE_RESULTS_ROOT
            / "pre_tss_off_gate_o1"
            / "effective_lambda_summary_v1.json"
        ),
    )
    parser.add_argument(
        "--identifiability-threshold",
        type=float,
        default=IDENTIFIABILITY_THRESHOLD,
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = build_artifact(
        args.positive_root,
        identifiability_threshold=args.identifiability_threshold,
    )
    action = "checked_only"
    if not args.check_only:
        action = common.write_once_or_identical(args.output, artifact)
    print(
        json.dumps(
            {
                "status": "complete",
                "action": action,
                "run_count": artifact["run_count"],
                "lambda_005_vs_010_not_fully_identifiable": artifact[
                    "identifiability"
                ]["lambda_005_vs_010_not_fully_identifiable"],
                "output": None if args.check_only else str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
