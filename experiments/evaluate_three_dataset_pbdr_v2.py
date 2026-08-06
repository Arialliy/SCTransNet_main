#!/usr/bin/env python3
"""Strict three-dataset PBDR-V2 evaluator with explicit source closure.

The adapter deliberately delegates metric computation and checkpoint auditing
to :mod:`experiments.evaluate_three_dataset_v2`.  Consequently the selected
``best_miou`` / ``best_pd`` checkpoints are re-evaluated at the same frozen
``probability > 0.5`` operating point; the threshold sweep remains descriptive
only.  The only accepted model path is the PBDR-V2 TSS-off training graph and
its exact 573-to-569 training-to-inference conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_three_dataset_v2 as core  # noqa: E402
from experiments import three_dataset_pbdr_v2_models_seed42_v1 as models  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import train_four_dataset_original_final_seed42_exact_v1 as training_engine  # noqa: E402
from experiments import train_three_dataset_pbdr_v2_tss_off_seed42_v1 as trainer  # noqa: E402
from experiments import tss_off_diagnostic_common_v1 as artifacts  # noqa: E402
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2 import (  # noqa: E402
    PBDR_V2_INTEGRATION_VERSION,
    PBDR_V2_STATE_KEYS,
)


ADAPTER_SCHEMA = "sctransnet_three_dataset_pbdr_v2_evaluator_v1"
TRAINING_RUN_SCHEMA = "sctransnet_three_dataset_pbdr_v2_tss_off_seed42_v1/v1"
OUTPUT_METHOD = "pbdr_v2_tss_off"
TRAINING_MODEL_METHOD = "final"
REQUESTED_TSS_WEIGHT = 0.0
FIXED_THRESHOLD = 0.5
RELATIVE_SOURCE = "experiments/evaluate_three_dataset_pbdr_v2.py"
EVALUATOR_DEPENDENCY_RELATIVE_PATHS = (
    RELATIVE_SOURCE,
    "experiments/evaluate_three_dataset_v2.py",
    "experiments/three_dataset_pbdr_v2_models_seed42_v1.py",
    "experiments/train_three_dataset_pbdr_v2_tss_off_seed42_v1.py",
    "experiments/train_four_dataset_original_final_seed42_exact_v1.py",
    "experiments/tss_off_diagnostic_common_v1.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/four_dataset_evaluation_protocol_v1.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/train_tpd_pilot.py",
)


def _explicit_architecture_paths() -> dict[str, Path]:
    """Return only architecture files frozen by the PBDR-V2 registry."""

    return {
        key.removeprefix("architecture::"): path
        for key, path in models.runtime_source_paths().items()
        if key.startswith("architecture::")
    }


def evaluator_source_sha256() -> dict[str, str]:
    """Hash the evaluator implementation and its explicit dependency set."""

    paths = {
        relative: (REPO_ROOT / relative).resolve(strict=True)
        for relative in EVALUATOR_DEPENDENCY_RELATIVE_PATHS
    }
    for relative, path in _explicit_architecture_paths().items():
        if relative in paths:
            raise ValueError(f"duplicate evaluator dependency: {relative}")
        paths[relative] = path
    return {
        relative: models.file_sha256(path)
        for relative, path in sorted(paths.items())
    }


def _build_inference_model(
    request: core.EvaluationRequest,
    training_state_dict: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply the registry-owned exact 573-to-569 inference conversion."""

    if request.method != TRAINING_MODEL_METHOD:
        raise ValueError("PBDR-V2 evaluator accepts only the Final method")
    model, metadata = (
        models.build_pbdr_v2_inference_model_from_training_state_dict(
            training_state_dict,
            dataset_name=request.dataset,
            seed=42,
        )
    )
    expected = {
        "strict_load": True,
        "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
        "inference_state_key_count": models.INFERENCE_STATE_KEY_COUNT,
        "stripped_state_key_count": 4,
        "target_survival_registered": False,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"PBDR-V2 inference metadata {field} differs")
    if metadata.get("pbdr_v2_state_preserved") != list(PBDR_V2_STATE_KEYS):
        raise ValueError("PBDR-V2 inference conversion did not preserve router state")
    manifest = metadata.get("architecture_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("PBDR-V2 inference metadata lacks architecture manifest")
    if manifest.get("pbdr_v2_integration_version") != PBDR_V2_INTEGRATION_VERSION:
        raise ValueError("PBDR-V2 evaluator routed to a different integration")
    if manifest.get("target_survival_registered") is not False:
        raise ValueError("PBDR-V2 inference graph still registers TSS")
    return model, dict(metadata)


@contextmanager
def _configured_core() -> Iterator[None]:
    """Temporarily bind the frozen core to the independent PBDR-V2 recipe."""

    replacements = {
        "TRAINING_RUN_SCHEMA": TRAINING_RUN_SCHEMA,
        "TSS_CANDIDATES": (REQUESTED_TSS_WEIGHT,),
        "_model_source_paths": _explicit_architecture_paths,
        "_training_runtime_source_paths": trainer.runtime_source_paths,
        "evaluator_source_sha256": evaluator_source_sha256,
        "build_inference_model": _build_inference_model,
    }
    previous = {key: getattr(core, key) for key in replacements}
    for key, value in replacements.items():
        setattr(core, key, value)
    try:
        if core.FIXED_THRESHOLD != FIXED_THRESHOLD:
            raise ValueError("core evaluator fixed threshold is not 0.5")
        yield
    finally:
        for key, value in previous.items():
            setattr(core, key, value)


def _validate_training_identity(run_dir: Path, dataset: str) -> dict[str, Any]:
    """Validate PBDR-V2-specific identity before entering the shared core."""

    if dataset not in models.DATASETS:
        raise ValueError(f"dataset must be one of {models.DATASETS}")
    resolved = Path(run_dir).resolve(strict=True)
    summary_path = resolved / "summary.json"
    protocol_path = resolved / "protocol.json"
    summary = artifacts.load_json(summary_path)
    protocol = artifacts.load_json(protocol_path)
    for label, payload in (("summary", summary), ("protocol", protocol)):
        for field, expected in (
            ("schema", TRAINING_RUN_SCHEMA),
            ("dataset", dataset),
            ("method", TRAINING_MODEL_METHOD),
            ("requested_tss_weight", REQUESTED_TSS_WEIGHT),
            ("tss_enabled", False),
        ):
            artifacts.require(
                payload.get(field) == expected,
                f"PBDR-V2 {label} {field} differs",
            )
    artifacts.require(summary.get("status") == "complete", "PBDR-V2 run is not complete")
    artifacts.require(summary.get("seed") == 42, "PBDR-V2 seed differs")
    artifacts.require(summary.get("epochs") == 1000, "PBDR-V2 epochs differ")

    recipe = protocol.get("recipe")
    artifacts.require(isinstance(recipe, dict), "PBDR-V2 protocol lacks recipe")
    for field, expected in (
        ("recipe_id", models.RECIPE_ID),
        ("architecture", "tpd8_ner4_qfg2_croa_pbdr_v2"),
        ("pbdr_v2_integration_version", PBDR_V2_INTEGRATION_VERSION),
        ("pbdr_v2_parameter_count", 19),
        ("pbdr_v2_state_key_count", len(PBDR_V2_STATE_KEYS)),
        ("fresh_seed42_scratch", True),
        ("warm_start_used", False),
        ("parent_checkpoint", None),
        ("resume_scope", "same_pbdr_v2_run_only"),
        ("current_shared_initial_state_bitwise_equal", True),
        ("pbdr_v2_new_state_exact_zero", True),
    ):
        artifacts.require(
            recipe.get(field) == expected,
            f"PBDR-V2 recipe {field} differs",
        )

    binding = protocol.get("pbdr_v2_architecture_binding")
    artifacts.require(
        isinstance(binding, dict),
        "PBDR-V2 protocol lacks architecture binding",
    )
    for field, expected in (
        ("integration_version", PBDR_V2_INTEGRATION_VERSION),
        ("training_state_key_count", models.TRAINING_STATE_KEY_COUNT),
        ("inference_state_key_count", models.INFERENCE_STATE_KEY_COUNT),
        ("new_state_keys", list(PBDR_V2_STATE_KEYS)),
    ):
        artifacts.require(
            binding.get(field) == expected,
            f"PBDR-V2 architecture binding {field} differs",
        )
    architecture_id = binding.get("architecture_id")
    artifacts.require(
        isinstance(architecture_id, str) and len(architecture_id) == 64,
        "PBDR-V2 architecture binding lacks architecture_id",
    )
    return {
        "summary": artifacts.artifact_record(summary_path),
        "protocol": artifacts.artifact_record(protocol_path),
        "recipe": recipe,
        "architecture_binding": binding,
    }


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
    """Evaluate one completed PBDR-V2 checkpoint through the frozen core."""

    identity = _validate_training_identity(run_dir, dataset)
    training_engine.configure_determinism()
    request = core.EvaluationRequest(
        dataset=dataset,
        method=TRAINING_MODEL_METHOD,
        checkpoint_role=checkpoint_role,
        requested_tss_weight=REQUESTED_TSS_WEIGHT,
    )
    with _configured_core():
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
    output["pbdr_v2_training_identity_binding"] = identity
    output["pbdr_v2_evaluator_adapter"] = {
        "schema": ADAPTER_SCHEMA,
        "training_run_schema": TRAINING_RUN_SCHEMA,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "fixed_checkpoint_threshold": FIXED_THRESHOLD,
        "integration_version": PBDR_V2_INTEGRATION_VERSION,
        "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
        "inference_state_key_count": models.INFERENCE_STATE_KEY_COUNT,
        "tss_state_stripping": (
            "573_to_569_exact_four_tss_keys_preserve_five_pbdr_v2_keys"
        ),
        "runtime_dependency_policy": "explicit_closure_no_model_tree_glob",
        "runtime_dependency_manifest": {
            key: {"path": str(path), "sha256": models.file_sha256(path)}
            for key, path in trainer.runtime_source_paths().items()
        },
        "evaluator_source_sha256": evaluator_source_sha256(),
        "adapter_sha256": models.file_sha256(Path(__file__)),
    }
    return output


def validate_completed_output(
    path: Path,
    *,
    dataset: str,
    checkpoint_role: str,
) -> dict[str, Any]:
    payload = artifacts.load_json(path)
    for field, expected in (
        ("schema", core.SCHEMA),
        ("status", "complete"),
        ("dataset", dataset),
        ("method", OUTPUT_METHOD),
        ("checkpoint_role", checkpoint_role),
        ("requested_tss_weight", REQUESTED_TSS_WEIGHT),
    ):
        artifacts.require(
            payload.get(field) == expected,
            f"PBDR-V2 evaluation {field} differs",
        )
    adapter = payload.get("pbdr_v2_evaluator_adapter")
    artifacts.require(isinstance(adapter, dict), "PBDR-V2 evaluation lacks adapter")
    for field, expected in (
        ("training_run_schema", TRAINING_RUN_SCHEMA),
        ("fixed_checkpoint_threshold", FIXED_THRESHOLD),
        ("integration_version", PBDR_V2_INTEGRATION_VERSION),
        ("training_state_key_count", models.TRAINING_STATE_KEY_COUNT),
        ("inference_state_key_count", models.INFERENCE_STATE_KEY_COUNT),
        ("runtime_dependency_policy", "explicit_closure_no_model_tree_glob"),
        (
            "tss_state_stripping",
            "573_to_569_exact_four_tss_keys_preserve_five_pbdr_v2_keys",
        ),
    ):
        artifacts.require(
            adapter.get(field) == expected,
            f"PBDR-V2 evaluation adapter {field} differs",
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
                "fixed_threshold": FIXED_THRESHOLD,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
