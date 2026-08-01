#!/usr/bin/env python3
"""Reuse each frozen SIRST3 best checkpoint on the three official test sets."""

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
    parser = evaluator.argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--source", choices=protocol.SOURCE_DATASETS)
    parser.add_argument("--method", choices=protocol.METHODS)
    parser.add_argument("--checkpoint-role", choices=protocol.CHECKPOINT_ROLES)
    parser.add_argument("--all", action="store_true")
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
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.all:
        if any(
            value is not None
            for value in (args.source, args.method, args.checkpoint_role)
        ):
            parser.error("--all cannot be combined with a single request")
    elif any(
        value is None
        for value in (args.source, args.method, args.checkpoint_role)
    ):
        parser.error("use --all or provide source/method/checkpoint-role")
    return args


def requests(args: argparse.Namespace) -> list[evaluator.EvaluationRequest]:
    if args.all:
        return [
            evaluator.EvaluationRequest("SIRST3", source, method, role)
            for source in protocol.SOURCE_DATASETS
            for method in protocol.METHODS
            for role in protocol.CHECKPOINT_ROLES
        ]
    return [
        evaluator.EvaluationRequest(
            "SIRST3",
            args.source,
            args.method,
            args.checkpoint_role,
        )
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = evaluator.run_requests(requests(args), args, sweep=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "evaluation_count": len(results),
                "same_sirst3_checkpoint_reused": True,
                "outputs": results,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
