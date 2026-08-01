#!/usr/bin/env python3
"""Closed-interval Pd--Fa sweeps for all frozen four-dataset checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_four_dataset_seed42_v1 as evaluator
from experiments import four_dataset_evaluation_protocol_v1 as protocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=evaluator.Path,
        default=protocol.EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--data-root",
        type=evaluator.Path,
        default=evaluator.REPO_ROOT / "datasets",
    )
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--source", choices=protocol.SOURCE_DATASETS)
    parser.add_argument("--method", choices=protocol.METHODS)
    parser.add_argument("--checkpoint-role", choices=protocol.CHECKPOINT_ROLES)
    parser.add_argument("--all-dataset-specific", action="store_true")
    parser.add_argument("--all-sirst3-sources", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    for name in (
        "imgidx_manifest",
        "normalization_manifest",
        "correction_manifest",
        "data_gate",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=evaluator.Path)
    args = parser.parse_args(argv)
    modes = sum(
        (
            bool(args.all_dataset_specific),
            bool(args.all_sirst3_sources),
            args.dataset is not None,
            args.source is not None,
        )
    )
    if modes != 1:
        parser.error(
            "choose exactly one of --all-dataset-specific, "
            "--all-sirst3-sources, --dataset, or --source"
        )
    if args.all_dataset_specific or args.all_sirst3_sources:
        if args.method is not None or args.checkpoint_role is not None:
            parser.error("all modes cannot take method/checkpoint-role")
    elif args.method is None or args.checkpoint_role is None:
        parser.error("single mode requires method and checkpoint-role")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def requests(args: argparse.Namespace) -> list[evaluator.EvaluationRequest]:
    if args.all_dataset_specific:
        return [
            evaluator.EvaluationRequest(dataset, dataset, method, role)
            for dataset in protocol.DATASETS
            for method in protocol.METHODS
            for role in protocol.CHECKPOINT_ROLES
        ]
    if args.all_sirst3_sources:
        return [
            evaluator.EvaluationRequest("SIRST3", source, method, role)
            for source in protocol.SOURCE_DATASETS
            for method in protocol.METHODS
            for role in protocol.CHECKPOINT_ROLES
        ]
    if args.source is not None:
        return [
            evaluator.EvaluationRequest(
                "SIRST3",
                args.source,
                args.method,
                args.checkpoint_role,
            )
        ]
    return [
        evaluator.EvaluationRequest(
            args.dataset,
            args.dataset,
            args.method,
            args.checkpoint_role,
        )
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = evaluator.run_requests(requests(args), args, sweep=True)
    print(
        json.dumps(
            {
                "status": "complete",
                "sweep_count": len(results),
                "fa_budgets": list(protocol.FA_BUDGETS),
                "outputs": results,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
