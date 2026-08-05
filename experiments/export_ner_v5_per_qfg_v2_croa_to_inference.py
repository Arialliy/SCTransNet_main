#!/usr/bin/env python3
"""Export a validated NER V5-PER checkpoint to its head-free V5 graph."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import three_dataset_ner_v5_per_models_seed42_v1 as registry  # noqa: E402
from experiments import train_three_dataset_ner_v5_per_tss_off_seed42 as trainer  # noqa: E402
from model.tpd_forward_contract import evaluator_prediction  # noqa: E402
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (  # noqa: E402
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)
from model.tpd_ner_v8_mprs_dch_v5_per import (  # noqa: E402
    V5_PER_FORMAL_DC_SUPPORT_MODE,
    V5_PER_RELAY_VERSION,
)
from model.tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival import (  # noqa: E402
    PRODUCTION_V5_PER_QFG_V2_CROA_INFERENCE_PARAMETERS,
    TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet,
    V5_PER_QFG_V2_CROA_INTEGRATION_VERSION,
    build_formal_v5_per_qfg_v2_croa_inference_model,
    build_formal_v5_per_qfg_v2_croa_survival_model,
    validate_formal_v5_per_qfg_v2_croa_inference_model,
    validate_formal_v5_per_qfg_v2_croa_survival_model,
)


EXPORT_SCHEMA = "sctransnet_ner_v5_per_qfg_v2_croa_inference_export/v1"
TRAINER_SCHEMA = trainer.SCHEMA
TRAINING_STATE_KEY_COUNT = registry.TRAINING_STATE_KEY_COUNT
INFERENCE_STATE_KEY_COUNT = registry.INFERENCE_STATE_KEY_COUNT
INFERENCE_PARAMETER_COUNT = (
    PRODUCTION_V5_PER_QFG_V2_CROA_INFERENCE_PARAMETERS
)
V5_ARCHITECTURE_NAME = "tpd8_ner_v5_per_qfg2_croa"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_is_valid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _expected_recipe() -> dict[str, Any]:
    return trainer.recipe_identity(
        argparse.Namespace(method=trainer.METHOD, tss_weight=0.0)
    )


def _validate_v5_manifest(
    manifest: Mapping[str, Any],
    *,
    training_graph: bool,
) -> None:
    expected = {
        "relay_version": V5_PER_RELAY_VERSION,
        "ner_version": "v5_per",
        "stage4_formula": "v4_exact",
        "stage3_formula": "v4_exact",
        "stage2_positive_route": (
            "centered-(1-persistent)*relu(centered)"
        ),
        "stage2_negative_route": "unchanged_identity_path",
        "stage2_dc_support": "one_minus_persistent_tail",
        "stage2_persistent_support_gradient": "stopped",
        "ner_v5_per_dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
        "qfg_integration_version": V5_PER_QFG_V2_CROA_INTEGRATION_VERSION,
        "parameters_added_vs_v4": 0,
        "buffers_added_vs_v4": 0,
        "state_layout_compatible_with": "ner_v4_tail_aware",
        "state_semantics_identical_to_v4": False,
        "checkpoint_semantically_interchangeable_with_v4": False,
        "v5_per_checkpoint_semantically_interchangeable_with_v4_qfg2": False,
        "qfg_enabled": True,
        "qfg_inference_required": True,
        "deployment_graph": "v5_per_qfg_v2_croa_no_tss",
    }
    for field, value in expected.items():
        _require(
            manifest.get(field) == value,
            f"V5 architecture manifest field {field!r} differs",
        )
    if training_graph:
        _require(
            manifest.get("survival_training_only") is True
            and manifest.get("survival_state_prefix")
            == SURVIVAL_STATE_PREFIX,
            "V5 training manifest lacks training-only TSS identity",
        )
    else:
        _require(
            manifest.get("target_survival_registered") is False
            and manifest.get("target_survival_state_removed") is True,
            "V5 inference manifest retains TSS identity",
        )


def require_v5_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Validate trainer, model-manifest, and V5-only semantic identity."""

    _require(isinstance(payload, Mapping), "V5 checkpoint must be a mapping")
    ready = copy.deepcopy(dict(payload))
    dataset = ready.get("dataset")
    expected_fields = {
        "schema": TRAINER_SCHEMA,
        "method": trainer.METHOD,
        "seed": trainer.TRAINING_SEED,
        "recipe": _expected_recipe(),
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "relay_version": V5_PER_RELAY_VERSION,
        "dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
        "training_state_key_count": TRAINING_STATE_KEY_COUNT,
    }
    for field, value in expected_fields.items():
        _require(
            ready.get(field) == value,
            f"V5 checkpoint field {field!r} differs",
        )
    _require(dataset in trainer.DATASETS, "V5 checkpoint dataset differs")
    _require(
        ready.get("checkpoint_role") in trainer.CHECKPOINT_ROLES,
        "V5 export accepts selected best_miou/best_pd checkpoints only",
    )
    _require(
        ready.get("selection_source") == f"test_{dataset}"
        and ready.get("test_selected") is True
        and ready.get("selection_is_optimistic") is True,
        "V5 checkpoint selection identity differs",
    )
    _require(
        _sha256_is_valid(ready.get("protocol_sha256")),
        "V5 checkpoint protocol SHA is invalid",
    )

    state = ready.get("state_dict")
    _require(isinstance(state, Mapping), "V5 checkpoint lacks state_dict")
    _require(
        len(state) == TRAINING_STATE_KEY_COUNT,
        "V5 checkpoint training state-key count differs",
    )
    # This also proves exact-zero TSS-off state and exactly four removable keys.
    registry.strip_tss_for_inference_state_dict(state)

    metadata = ready.get("model_metadata")
    _require(
        isinstance(metadata, Mapping),
        "V5 checkpoint lacks model_metadata",
    )
    metadata_expected = {
        "schema": registry.SCHEMA,
        "dataset_name": dataset,
        "method": registry.METHOD,
        "recipe_id": registry.RECIPE_ID,
        "training_seed": registry.TRAINING_SEED,
        "training_graph": True,
        "state_key_count": TRAINING_STATE_KEY_COUNT,
        "target_survival_registered": True,
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "tss_loss_consumes_logits": False,
        "relay_version": V5_PER_RELAY_VERSION,
        "dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
    }
    for field, value in metadata_expected.items():
        _require(
            metadata.get(field) == value,
            f"V5 model metadata field {field!r} differs",
        )
    manifest = metadata.get("architecture_manifest")
    _require(
        isinstance(manifest, Mapping),
        "V5 checkpoint lacks architecture manifest",
    )
    _validate_v5_manifest(manifest, training_graph=True)
    architecture_id = registry.canonical_sha256(manifest)
    _require(
        metadata.get("architecture_id") == architecture_id
        and ready.get("architecture_id") == architecture_id,
        "V5 checkpoint architecture id is not bound to its manifest",
    )
    _require(
        ready["recipe"].get("architecture") == V5_ARCHITECTURE_NAME,
        "V5 checkpoint recipe architecture differs",
    )
    return ready


def strip_v5_tss_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Delegate the exact-zero, exactly-four-key removal to the V5 registry."""

    return registry.strip_tss_for_inference_state_dict(state_dict)


def build_v5_inference_model_from_training_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    dataset_name: str,
    seed: int = registry.TRAINING_SEED,
) -> tuple[TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet, dict[str, Any]]:
    model, metadata = registry.build_v5_inference_model_from_training_state_dict(
        state_dict,
        dataset_name=dataset_name,
        seed=seed,
    )
    if type(model) is not TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet:
        raise TypeError("V5 export requires the exact V5 head-free class")
    if hasattr(model, "target_survival"):
        raise RuntimeError("V5 inference graph retains TSS heads")
    _require(
        len(model.state_dict()) == INFERENCE_STATE_KEY_COUNT,
        "V5 inference state-key count differs",
    )
    _require(
        _parameter_count(model) == INFERENCE_PARAMETER_COUNT,
        "V5 inference parameter count differs",
    )
    _validate_v5_manifest(model.architecture_manifest(), training_graph=False)
    return model, metadata


def assert_training_inference_equivalent(
    state_dict: Mapping[str, torch.Tensor],
    *,
    dataset_name: str,
    images: torch.Tensor | None = None,
) -> None:
    """Prove V5 training-eval segmentation equals head-free V5 inference."""

    training_model, _ = build_formal_v5_per_qfg_v2_croa_survival_model()
    incompatible = training_model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V5 training graph strict load is incompatible")
    validate_formal_v5_per_qfg_v2_croa_survival_model(training_model)
    inference_model, _ = build_v5_inference_model_from_training_state_dict(
        state_dict,
        dataset_name=dataset_name,
    )
    training_model.eval()
    inference_model.eval()
    if images is None:
        images = torch.linspace(-1.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
    _require(
        isinstance(images, torch.Tensor)
        and images.ndim == 4
        and images.shape[1] == 1,
        "V5 equivalence probe images must be Bx1xHxW",
    )
    with torch.no_grad():
        training_prediction = evaluator_prediction(training_model(images))
        inference_prediction = evaluator_prediction(inference_model(images))
    if not torch.equal(training_prediction, inference_prediction):
        raise RuntimeError(
            "V5 training-eval and head-free inference predictions differ"
        )


def _regular_file_bytes(path: Path) -> bytes:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"checkpoint is not a regular file: {candidate}")
    return candidate.read_bytes()


def export_v5_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write one validated, write-once, V5-only head-free checkpoint."""

    output_path = Path(output_path).resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite export: {output_path}")
    source_content = _regular_file_bytes(checkpoint_path)
    payload = torch.load(
        io.BytesIO(source_content),
        map_location="cpu",
        weights_only=False,
    )
    validated = require_v5_checkpoint_payload(payload)
    state_dict = validated["state_dict"]
    dataset = str(validated["dataset"])
    inference_state = strip_v5_tss_state_dict(state_dict)
    # The fixed probe catches training/inference graph drift before writing.
    assert_training_inference_equivalent(
        state_dict,
        dataset_name=dataset,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.is_symlink() or not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    manifest = validated["model_metadata"]["architecture_manifest"]
    exported = {
        "schema": EXPORT_SCHEMA,
        "source_trainer_schema": TRAINER_SCHEMA,
        "source_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "source_checkpoint_sha256": hashlib.sha256(source_content).hexdigest(),
        "source_checkpoint_role": validated["checkpoint_role"],
        "source_dataset": dataset,
        "source_architecture_id": validated["architecture_id"],
        "source_architecture_manifest": copy.deepcopy(dict(manifest)),
        "source_relay_version": V5_PER_RELAY_VERSION,
        "source_state_semantics": "ner_v5_per_only",
        "v4_semantic_interpretation_allowed": False,
        "state_dict": inference_state,
        "tss_state_removed": list(SURVIVAL_STATE_KEYS),
        "tss_state_removed_count": len(SURVIVAL_STATE_KEYS),
        "training_eval_inference_equivalence_probe": True,
        "inference_graph": "v5_per_qfg_v2_croa_no_tss",
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
    }
    serialized = io.BytesIO()
    torch.save(exported, serialized)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(serialized.getbuffer())
            output_file.flush()
            os.fsync(output_file.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite export: {output_path}"
            ) from error
        directory_descriptor = os.open(str(output_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return exported


def validate_exported_v5_checkpoint(
    output_path: Path,
    *,
    expected_source_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Re-open, identity-check, and strict-load one V5 inference export."""

    content = _regular_file_bytes(output_path)
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "V5 export must be a mapping")
    ready = copy.deepcopy(dict(payload))
    expected = {
        "schema": EXPORT_SCHEMA,
        "source_trainer_schema": TRAINER_SCHEMA,
        "source_relay_version": V5_PER_RELAY_VERSION,
        "source_state_semantics": "ner_v5_per_only",
        "v4_semantic_interpretation_allowed": False,
        "tss_state_removed": list(SURVIVAL_STATE_KEYS),
        "tss_state_removed_count": len(SURVIVAL_STATE_KEYS),
        "training_eval_inference_equivalence_probe": True,
        "inference_graph": "v5_per_qfg_v2_croa_no_tss",
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
    }
    for field, value in expected.items():
        _require(ready.get(field) == value, f"V5 export field {field!r} differs")
    _require(
        ready.get("source_dataset") in registry.DATASETS,
        "V5 export dataset differs",
    )
    manifest = ready.get("source_architecture_manifest")
    _require(isinstance(manifest, Mapping), "V5 export lacks source manifest")
    _validate_v5_manifest(manifest, training_graph=True)
    architecture_id = registry.canonical_sha256(manifest)
    _require(
        ready.get("source_architecture_id") == architecture_id,
        "V5 export architecture id differs",
    )
    state = ready.get("state_dict")
    _require(isinstance(state, Mapping), "V5 export lacks state_dict")
    _require(
        len(state) == INFERENCE_STATE_KEY_COUNT,
        "V5 export inference state-key count differs",
    )
    _require(
        not any(key.startswith(SURVIVAL_STATE_PREFIX) for key in state),
        "V5 export retains TSS state",
    )
    model, _ = build_formal_v5_per_qfg_v2_croa_inference_model()
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("V5 export strict inference load is incompatible")
    validate_formal_v5_per_qfg_v2_croa_inference_model(model)
    del model
    source_sha = ready.get("source_checkpoint_sha256")
    _require(_sha256_is_valid(source_sha), "V5 export source SHA is invalid")
    if expected_source_checkpoint is not None:
        expected_path = Path(expected_source_checkpoint).resolve()
        _require(
            Path(str(ready.get("source_checkpoint_path"))).resolve()
            == expected_path,
            "V5 export source path differs",
        )
        _require(
            hashlib.sha256(_regular_file_bytes(expected_path)).hexdigest()
            == source_sha,
            "V5 export source content differs",
        )
    return {
        "schema": EXPORT_SCHEMA,
        "path": str(Path(output_path).resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_checkpoint_sha256": source_sha,
        "source_dataset": ready["source_dataset"],
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
        "strict_v5_load": True,
        "tss_state_absent": True,
        "v4_semantic_interpretation_allowed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    export_v5_checkpoint(args.checkpoint.resolve(), args.output.resolve())


__all__ = [
    "EXPORT_SCHEMA",
    "INFERENCE_PARAMETER_COUNT",
    "INFERENCE_STATE_KEY_COUNT",
    "TRAINER_SCHEMA",
    "TRAINING_STATE_KEY_COUNT",
    "assert_training_inference_equivalent",
    "build_v5_inference_model_from_training_state_dict",
    "export_v5_checkpoint",
    "main",
    "parse_args",
    "require_v5_checkpoint_payload",
    "strip_v5_tss_state_dict",
    "validate_exported_v5_checkpoint",
]


if __name__ == "__main__":
    main()
