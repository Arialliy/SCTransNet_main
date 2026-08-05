#!/usr/bin/env python3
"""Strict V5-PER evaluator with an explicit runtime dependency closure."""

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
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import three_dataset_ner_v5_per_models_seed42_v1 as models  # noqa: E402
from experiments import train_three_dataset_ner_v5_per_tss_off_seed42 as trainer  # noqa: E402
from experiments import train_four_dataset_original_final_seed42_exact_v1 as training_engine  # noqa: E402
from experiments import tss_off_diagnostic_common_v1 as artifacts  # noqa: E402
from model.tpd_ner_v8_mprs_dch_v5_per import (  # noqa: E402
    V5_PER_FORMAL_DC_SUPPORT_MODE,
    V5_PER_RELAY_VERSION,
)


ADAPTER_SCHEMA = "sctransnet_three_dataset_ner_v5_per_evaluator_v1"
TRAINING_RUN_SCHEMA = "sctransnet_three_dataset_ner_v5_per_tss_off_seed42/v1"
OUTPUT_METHOD = "ner_v5_per_tss_off"
TRAINING_MODEL_METHOD = "final"
REQUESTED_TSS_WEIGHT = 0.0
RELATIVE_SOURCE = "experiments/evaluate_three_dataset_ner_v5_per.py"
EVALUATOR_DEPENDENCY_RELATIVE_PATHS = (
    RELATIVE_SOURCE,
    "experiments/evaluate_three_dataset_v2.py",
    "experiments/three_dataset_ner_v5_per_models_seed42_v1.py",
    "experiments/train_three_dataset_ner_v5_per_tss_off_seed42.py",
    "experiments/train_four_dataset_original_final_seed42_exact_v1.py",
    "experiments/tss_off_diagnostic_common_v1.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/four_dataset_evaluation_protocol_v1.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/train_tpd_pilot.py",
)


def _explicit_architecture_paths() -> dict[str, Path]:
    return {
        key.removeprefix("architecture::"): path
        for key, path in models.runtime_source_paths().items()
        if key.startswith("architecture::")
    }


def evaluator_source_sha256() -> dict[str, str]:
    paths = {
        relative: (REPO_ROOT / relative).resolve(strict=True)
        for relative in EVALUATOR_DEPENDENCY_RELATIVE_PATHS
    }
    paths.update(_explicit_architecture_paths())
    return {
        relative: models.file_sha256(path)
        for relative, path in sorted(paths.items())
    }


def _build_inference_model(
    request: core.EvaluationRequest,
    training_state_dict: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if request.method != TRAINING_MODEL_METHOD:
        raise ValueError("V5 evaluator accepts only the Final training method")
    model, metadata = models.build_v5_inference_model_from_training_state_dict(
        training_state_dict,
        dataset_name=request.dataset,
        seed=42,
    )
    if metadata.get("relay_version") != V5_PER_RELAY_VERSION:
        raise ValueError("V5 evaluator routed to a non-V5 relay")
    if metadata.get("dc_support_mode") != V5_PER_FORMAL_DC_SUPPORT_MODE:
        raise ValueError("V5 evaluator routed to a non-complement-tail graph")
    return model, metadata


@contextmanager
def _configured_core() -> Iterator[None]:
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
        yield
    finally:
        for key, value in previous.items():
            setattr(core, key, value)


def _validate_training_identity(run_dir: Path, dataset: str) -> dict[str, Any]:
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
            artifacts.require(payload.get(field) == expected, f"V5 {label} {field} differs")
    artifacts.require(summary.get("status") == "complete", "V5 run is not complete")
    artifacts.require(summary.get("seed") == 42, "V5 seed differs")
    artifacts.require(summary.get("epochs") == 1000, "V5 epochs differ")
    recipe = protocol.get("recipe")
    artifacts.require(isinstance(recipe, dict), "V5 protocol lacks recipe")
    for field, expected in (
        ("recipe_id", models.RECIPE_ID),
        ("relay_version", V5_PER_RELAY_VERSION),
        ("dc_support_mode", V5_PER_FORMAL_DC_SUPPORT_MODE),
        ("fresh_seed42_scratch", True),
        ("warm_start_from_v4", False),
        ("resume_scope", "same_v5_run_only"),
        ("qfg_parameters_jointly_trainable", True),
    ):
        artifacts.require(recipe.get(field) == expected, f"V5 recipe {field} differs")
    binding = protocol.get("v5_architecture_binding")
    artifacts.require(isinstance(binding, dict), "V5 protocol lacks architecture binding")
    artifacts.require(
        binding.get("training_state_key_count") == models.TRAINING_STATE_KEY_COUNT,
        "V5 protocol training state count differs",
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
    output["v5_training_identity_binding"] = identity
    output["v5_evaluator_adapter"] = {
        "schema": ADAPTER_SCHEMA,
        "training_run_schema": TRAINING_RUN_SCHEMA,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "relay_version": V5_PER_RELAY_VERSION,
        "dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
        "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
        "inference_state_key_count": models.INFERENCE_STATE_KEY_COUNT,
        "tss_state_stripping": "568_to_564_exact_four_keys",
        "runtime_dependency_policy": "explicit_closure_no_model_tree_glob",
        "runtime_dependency_manifest": {
            key: {"path": str(path), "sha256": models.file_sha256(path)}
            for key, path in trainer.runtime_source_paths().items()
        },
        "evaluator_source_sha256": evaluator_source_sha256(),
        "adapter_sha256": models.file_sha256(Path(__file__)),
    }
    return output


def validate_completed_output(path: Path, *, dataset: str, checkpoint_role: str) -> dict[str, Any]:
    payload = artifacts.load_json(path)
    for field, expected in (
        ("schema", core.SCHEMA),
        ("status", "complete"),
        ("dataset", dataset),
        ("method", OUTPUT_METHOD),
        ("checkpoint_role", checkpoint_role),
        ("requested_tss_weight", 0.0),
    ):
        artifacts.require(payload.get(field) == expected, f"V5 evaluation {field} differs")
    adapter = payload.get("v5_evaluator_adapter")
    artifacts.require(isinstance(adapter, dict), "V5 evaluation lacks adapter")
    artifacts.require(
        adapter.get("runtime_dependency_policy") == "explicit_closure_no_model_tree_glob",
        "V5 evaluation used the wrong runtime closure policy",
    )
    artifacts.require(
        adapter.get("tss_state_stripping") == "568_to_564_exact_four_keys",
        "V5 evaluation stripping contract differs",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument("--checkpoint-role", choices=core.CHECKPOINT_ROLES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT)
    parser.add_argument("--data-protocol-manifest", type=Path, default=data_protocol.DEFAULT_MANIFEST_PATH)
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
    destination = args.output or args.run_dir / "evaluations" / f"{args.checkpoint_role}.json"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(destination)
    artifacts.atomic_write_json(destination, output)
    print(json.dumps({"status": "complete", "output": str(destination.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
