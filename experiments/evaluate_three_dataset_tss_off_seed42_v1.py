#!/usr/bin/env python3
"""Thin, source-locked adapter for evaluating three-dataset TSS-off runs.

All inference, fixed-threshold metrics, descriptive sweeps, checkpoint audits,
and data identity checks remain implemented by ``evaluate_three_dataset_v2``.
This adapter changes only the admitted training schema and the one legal Final
requested weight (zero), and records its own SHA alongside the frozen core.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_three_dataset_v2 as core  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)
from experiments import tss_off_diagnostic_common_v1 as common  # noqa: E402


ADAPTER_SCHEMA = "sctransnet_three_dataset_tss_off_evaluator_adapter_v1"
TRAINING_RUN_SCHEMA = "sctransnet_three_dataset_tss_off_seed42_v1/v1"
REQUESTED_TSS_WEIGHT = 0.0
OUTPUT_METHOD = "final_tss_off"
RELATIVE_SOURCE = "experiments/evaluate_three_dataset_tss_off_seed42_v1.py"


def configure_core() -> None:
    """Narrow the frozen evaluator admission contract to the TSS-off recipe."""

    core.TRAINING_RUN_SCHEMA = TRAINING_RUN_SCHEMA
    core.TSS_CANDIDATES = (REQUESTED_TSS_WEIGHT,)
    if RELATIVE_SOURCE not in core._EVALUATOR_NON_MODEL_SOURCES:
        core._EVALUATOR_NON_MODEL_SOURCES = (
            *core._EVALUATOR_NON_MODEL_SOURCES,
            RELATIVE_SOURCE,
        )


def evaluate_run(
    *,
    dataset: str,
    checkpoint_role: str,
    run_dir: Path,
    dataset_root: Path = common.DATASET_ROOT,
    data_protocol_manifest: Path = common.DATA_PROTOCOL_MANIFEST,
    device_name: str = "cuda:0",
    workers: int = 0,
) -> dict[str, Any]:
    configure_core()
    # Reproduce the exact inference-kernel contract used during checkpoint
    # selection.  Without this call, a numerically sensitive checkpoint can
    # cross the frozen 1e-4 overlap-metric audit tolerance even though its
    # object counts, Pd and Fa remain identical.
    training_engine.configure_determinism()
    request = core.EvaluationRequest(
        dataset=dataset,
        method="final",
        checkpoint_role=checkpoint_role,
        requested_tss_weight=REQUESTED_TSS_WEIGHT,
    )
    request.validate()
    output = core.evaluate_run(
        request,
        run_dir=run_dir,
        dataset_root=dataset_root,
        data_protocol_manifest=data_protocol_manifest,
        device_name=device_name,
        workers=workers,
    )
    output["method"] = OUTPUT_METHOD
    output["training_model_method"] = "final"
    output["tss_off_evaluator_adapter"] = {
        "schema": ADAPTER_SCHEMA,
        "training_run_schema": TRAINING_RUN_SCHEMA,
        "core_method": "final",
        "output_method": OUTPUT_METHOD,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "semantic_change_to_metric_core": False,
        "training_determinism_contract_reapplied": True,
        "determinism_source": (
            "experiments/"
            "train_four_dataset_original_final_seed42_exact_v1.py"
        ),
        "determinism_source_sha256": common.file_sha256(
            Path(training_engine.__file__)
        ),
        "core_evaluator": RELATIVE_SOURCE.replace(
            "evaluate_three_dataset_tss_off_seed42_v1.py",
            "evaluate_three_dataset_v2.py",
        ),
        "core_evaluator_sha256": common.file_sha256(
            REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py"
        ),
        "adapter_sha256": common.file_sha256(Path(__file__)),
    }
    return output


def validate_completed_output(
    path: Path,
    *,
    dataset: str,
    checkpoint_role: str,
) -> dict[str, Any]:
    payload = common.load_json(path)
    common.require(payload.get("schema") == core.SCHEMA, "evaluation schema differs")
    common.require(payload.get("status") == "complete", "evaluation is incomplete")
    common.require(payload.get("dataset") == dataset, "evaluation dataset differs")
    common.require(payload.get("method") == OUTPUT_METHOD, "evaluation method differs")
    common.require(
        payload.get("training_model_method") == "final",
        "evaluation training-model method differs",
    )
    common.require(payload.get("checkpoint_role") == checkpoint_role, "evaluation role differs")
    common.require(
        payload.get("requested_tss_weight") == REQUESTED_TSS_WEIGHT,
        "evaluation requested weight differs",
    )
    point = payload.get("fixed_threshold_0_5")
    common.require(isinstance(point, dict), "evaluation lacks fixed 0.5 point")
    common.require(point.get("threshold") == 0.5, "fixed point threshold differs")
    adapter = payload.get("tss_off_evaluator_adapter")
    common.require(isinstance(adapter, dict), "evaluation lacks TSS-off adapter lock")
    common.require(adapter.get("schema") == ADAPTER_SCHEMA, "adapter schema differs")
    common.require(
        adapter.get("training_determinism_contract_reapplied") is True,
        "evaluation omitted the training determinism contract",
    )
    common.require(
        adapter.get("determinism_source_sha256")
        == common.file_sha256(Path(training_engine.__file__)),
        "evaluation determinism source SHA differs",
    )
    common.require(
        adapter.get("core_evaluator_sha256")
        == common.file_sha256(REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py"),
        "evaluation core SHA differs",
    )
    sources = payload.get("source_sha256")
    common.require(isinstance(sources, dict), "evaluation lacks source lock")
    common.require(
        sources.get(RELATIVE_SOURCE) == common.file_sha256(Path(__file__)),
        "evaluation adapter source SHA differs",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument(
        "--checkpoint-role", choices=core.CHECKPOINT_ROLES, required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--data-protocol-manifest",
        type=Path,
        default=data_protocol.DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = evaluate_run(
        dataset=args.dataset,
        checkpoint_role=args.checkpoint_role,
        run_dir=args.run_dir,
        dataset_root=args.dataset_root,
        data_protocol_manifest=args.data_protocol_manifest,
        device_name=args.device,
        workers=args.workers,
    )
    destination = (
        args.output
        if args.output is not None
        else args.run_dir / "evaluations" / f"{args.checkpoint_role}.json"
    )
    if destination.exists() and not args.overwrite:
        raise FileExistsError(destination)
    common.atomic_write_json(destination, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(destination.resolve()),
                "sha256": common.file_sha256(destination.resolve()),
                "fixed_threshold": 0.5,
                "requested_tss_weight": REQUESTED_TSS_WEIGHT,
                "descriptive_sweep_only": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
