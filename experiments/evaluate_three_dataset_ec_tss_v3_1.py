#!/usr/bin/env python3
"""Strict three-dataset evaluator adapter for EC-TSS V3.1.

The metric implementation, img_idx/test binding, fixed threshold, checkpoint
audit, and descriptive Pd--Fa sweep remain owned by
``evaluate_three_dataset_v2``.  This adapter changes only the admitted
training schema/recipe identity and reapplies the training-time deterministic
CUDA contract before the inference model is constructed.
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
from experiments import tss_off_diagnostic_common_v1 as artifacts  # noqa: E402


ADAPTER_SCHEMA = (
    "sctransnet_three_dataset_ec_tss_v3_1_evaluator_adapter_v1"
)
TRAINING_RUN_SCHEMA = "sctransnet_three_dataset_ec_tss_v3_1_seed42/v1"
OBJECTIVE_ID = "ec_tss_v3_1"
RECIPE_ID = "final_ec_tss_v3_1"
REQUESTED_TSS_WEIGHT = 0.005
SURVIVAL_RATIO_CAP = 0.10
CONFIDENCE_THRESHOLD = 0.5
TARGET_DILATION_RADIUS = 3
OUTPUT_METHOD = "final_ec_tss_v3_1"
TRAINING_MODEL_METHOD = "final"
RELATIVE_SOURCE = "experiments/evaluate_three_dataset_ec_tss_v3_1.py"


def expected_recipe() -> dict[str, Any]:
    """Return the complete frozen EC-TSS V3.1 training identity."""

    return {
        "method": TRAINING_MODEL_METHOD,
        "recipe_id": RECIPE_ID,
        "objective_id": OBJECTIVE_ID,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "tss_lambda_token": "0p005",
        "tss_ratio_cap": SURVIVAL_RATIO_CAP,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "target_dilation_radius": TARGET_DILATION_RADIUS,
        "positive_normalization": "risk_mass_clamp_min_1",
        "negative_normalization": "risk_mass_clamp_min_1",
        "tss_enabled": True,
        "survival_pos_weight_used": False,
    }


def _validate_training_identity(
    run_dir: Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Reject a same-weight checkpoint trained under another objective."""

    resolved = Path(run_dir).resolve(strict=True)
    summary_path = resolved / "summary.json"
    protocol_path = resolved / "protocol.json"
    summary = artifacts.load_json(summary_path)
    protocol = artifacts.load_json(protocol_path)
    recipe = expected_recipe()
    for label, container in (("summary", summary), ("protocol", protocol)):
        for field, expected in (
            ("schema", TRAINING_RUN_SCHEMA),
            ("dataset", dataset),
            ("method", TRAINING_MODEL_METHOD),
            ("objective_id", OBJECTIVE_ID),
            ("recipe", recipe),
        ):
            artifacts.require(
                container.get(field) == expected,
                f"EC-TSS {label} {field} differs",
            )
    for field, expected in (
        ("status", "complete"),
        ("seed", 42),
        ("epochs", 1000),
        ("planned_total_epochs", 1000),
    ):
        artifacts.require(
            summary.get(field) == expected,
            f"EC-TSS summary {field} differs",
        )
    for field, expected in (
        ("training_seed", 42),
        ("epochs", 1000),
        ("planned_total_epochs", 1000),
        ("begin_test", 10),
        ("eval_every", 10),
    ):
        artifacts.require(
            protocol.get(field) == expected,
            f"EC-TSS protocol {field} differs",
        )
    training = protocol.get("training")
    artifacts.require(
        isinstance(training, dict),
        "EC-TSS protocol lacks training objective metadata",
    )
    for field, expected in (
        ("objective_id", OBJECTIVE_ID),
        ("tss_requested_weight", REQUESTED_TSS_WEIGHT),
        ("tss_ratio_cap", SURVIVAL_RATIO_CAP),
        ("confidence_threshold", CONFIDENCE_THRESHOLD),
        ("target_dilation_radius", TARGET_DILATION_RADIUS),
        ("positive_normalization", "risk_mass_clamp_min_1"),
        ("negative_normalization", "risk_mass_clamp_min_1"),
        ("survival_pos_weight_used", False),
    ):
        artifacts.require(
            training.get(field) == expected,
            f"EC-TSS protocol training {field} differs",
        )
    metrics = protocol.get("metrics")
    artifacts.require(
        isinstance(metrics, dict) and metrics.get("threshold") == 0.5,
        "EC-TSS protocol metric threshold differs",
    )
    declared = protocol.get("protocol_sha256")
    artifacts.require(
        isinstance(declared, str) and declared,
        "EC-TSS protocol lacks protocol_sha256",
    )
    unsigned = dict(protocol)
    del unsigned["protocol_sha256"]
    artifacts.require(
        artifacts.compact_sha256(unsigned) == declared,
        "EC-TSS protocol payload SHA differs",
    )
    artifacts.require(
        summary.get("protocol_sha256") == declared,
        "EC-TSS summary protocol SHA differs",
    )
    return {
        "run_dir": str(resolved),
        "summary": artifacts.artifact_record(summary_path),
        "protocol": artifacts.artifact_record(protocol_path),
        "protocol_payload_sha256": declared,
        "recipe": recipe,
    }


def configure_core() -> None:
    """Narrow the shared evaluator admission contract to EC-TSS V3.1."""

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
    dataset_root: Path = data_protocol.DEFAULT_DATASET_ROOT,
    data_protocol_manifest: Path = data_protocol.DEFAULT_MANIFEST_PATH,
    device_name: str = "cuda:0",
    workers: int = 0,
) -> dict[str, Any]:
    """Evaluate one completed EC-TSS V3.1 selected checkpoint."""

    configure_core()
    identity_binding = _validate_training_identity(
        run_dir,
        dataset=dataset,
    )
    # The training engine enables deterministic algorithms before creating a
    # CUDA device or model.  Reapply exactly that contract here; the CUBLAS
    # workspace environment is set by the launcher before Python starts.
    training_engine.configure_determinism()
    request = core.EvaluationRequest(
        dataset=dataset,
        method=TRAINING_MODEL_METHOD,
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
    output["training_model_method"] = TRAINING_MODEL_METHOD
    output["ec_tss_v3_1_training_identity_binding"] = identity_binding
    output["ec_tss_v3_1_evaluator_adapter"] = {
        "schema": ADAPTER_SCHEMA,
        "training_run_schema": TRAINING_RUN_SCHEMA,
        "objective_id": OBJECTIVE_ID,
        "recipe_id": RECIPE_ID,
        "core_method": TRAINING_MODEL_METHOD,
        "output_method": OUTPUT_METHOD,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "survival_ratio_cap": SURVIVAL_RATIO_CAP,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "target_dilation_radius": TARGET_DILATION_RADIUS,
        "semantic_change_to_metric_core": False,
        "training_determinism_contract_reapplied": True,
        "determinism_source": (
            "experiments/"
            "train_four_dataset_original_final_seed42_exact_v1.py"
        ),
        "determinism_source_sha256": artifacts.file_sha256(
            Path(training_engine.__file__)
        ),
        "core_evaluator": "experiments/evaluate_three_dataset_v2.py",
        "core_evaluator_sha256": artifacts.file_sha256(
            REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py"
        ),
        "adapter_sha256": artifacts.file_sha256(Path(__file__)),
    }
    return output


def validate_completed_output(
    path: Path,
    *,
    dataset: str,
    checkpoint_role: str,
) -> dict[str, Any]:
    """Validate a reusable evaluator output against current source bytes."""

    payload = artifacts.load_json(path)
    artifacts.require(
        payload.get("schema") == core.SCHEMA,
        "evaluation schema differs",
    )
    for field, expected in (
        ("status", "complete"),
        ("dataset", dataset),
        ("method", OUTPUT_METHOD),
        ("training_model_method", TRAINING_MODEL_METHOD),
        ("checkpoint_role", checkpoint_role),
        ("requested_tss_weight", REQUESTED_TSS_WEIGHT),
    ):
        artifacts.require(
            payload.get(field) == expected,
            f"evaluation {field} differs",
        )
    fixed = payload.get("fixed_threshold_0_5")
    artifacts.require(
        isinstance(fixed, dict) and fixed.get("threshold") == 0.5,
        "evaluation lacks its fixed threshold=0.5 point",
    )
    adapter = payload.get("ec_tss_v3_1_evaluator_adapter")
    artifacts.require(isinstance(adapter, dict), "evaluation lacks adapter lock")
    expected_adapter = {
        "schema": ADAPTER_SCHEMA,
        "training_run_schema": TRAINING_RUN_SCHEMA,
        "objective_id": OBJECTIVE_ID,
        "recipe_id": RECIPE_ID,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "survival_ratio_cap": SURVIVAL_RATIO_CAP,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "target_dilation_radius": TARGET_DILATION_RADIUS,
        "training_determinism_contract_reapplied": True,
    }
    for field, expected in expected_adapter.items():
        artifacts.require(
            adapter.get(field) == expected,
            f"evaluation adapter {field} differs",
        )
    artifacts.require(
        adapter.get("determinism_source_sha256")
        == artifacts.file_sha256(Path(training_engine.__file__)),
        "evaluation determinism source SHA differs",
    )
    artifacts.require(
        adapter.get("core_evaluator_sha256")
        == artifacts.file_sha256(
            REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py"
        ),
        "evaluation core SHA differs",
    )
    artifacts.require(
        adapter.get("adapter_sha256")
        == artifacts.file_sha256(Path(__file__)),
        "evaluation adapter SHA differs",
    )
    sources = payload.get("source_sha256")
    artifacts.require(isinstance(sources, dict), "evaluation lacks source lock")
    artifacts.require(
        sources.get(RELATIVE_SOURCE) == artifacts.file_sha256(Path(__file__)),
        "evaluation source lock differs",
    )
    binding = payload.get("ec_tss_v3_1_training_identity_binding")
    artifacts.require(
        isinstance(binding, dict),
        "evaluation lacks EC-TSS training identity binding",
    )
    run_dir = Path(str(binding.get("run_dir", "")))
    expected_binding = _validate_training_identity(run_dir, dataset=dataset)
    artifacts.require(
        binding == expected_binding,
        "evaluation EC-TSS training identity binding differs",
    )
    core_binding = payload.get("checkpoint_binding")
    artifacts.require(
        isinstance(core_binding, dict),
        "evaluation lacks checkpoint binding",
    )
    artifacts.require(
        core_binding.get("run_dir") == expected_binding["run_dir"],
        "evaluation core run-directory binding differs",
    )
    for label in ("summary", "protocol"):
        record = core_binding.get(label)
        artifacts.require(
            isinstance(record, dict),
            f"evaluation lacks bound {label}",
        )
        artifacts.require(
            record.get("path") == expected_binding[label]["path"]
            and record.get("sha256") == expected_binding[label]["sha256"],
            f"evaluation core {label} binding differs",
        )
    checkpoint = core_binding.get("checkpoint")
    artifacts.require(
        isinstance(checkpoint, dict),
        "evaluation lacks bound checkpoint",
    )
    expected_checkpoint = (
        run_dir
        / "checkpoints"
        / core.CHECKPOINT_FILENAMES[checkpoint_role]
    ).resolve()
    artifacts.require(
        Path(str(checkpoint.get("path", ""))).resolve()
        == expected_checkpoint,
        "evaluation core checkpoint path differs",
    )
    artifacts.require(
        checkpoint.get("role") == checkpoint_role,
        "evaluation core checkpoint role differs",
    )
    artifacts.require(
        checkpoint.get("sha256")
        == artifacts.file_sha256(expected_checkpoint),
        "evaluation core checkpoint SHA differs",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument(
        "--checkpoint-role",
        choices=core.CHECKPOINT_ROLES,
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=data_protocol.DEFAULT_DATASET_ROOT,
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
    artifacts.atomic_write_json(destination, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(destination.resolve()),
                "sha256": artifacts.file_sha256(destination.resolve()),
                "fixed_threshold": 0.5,
                "requested_tss_weight": REQUESTED_TSS_WEIGHT,
                "objective_id": OBJECTIVE_ID,
                "descriptive_sweep_only": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
