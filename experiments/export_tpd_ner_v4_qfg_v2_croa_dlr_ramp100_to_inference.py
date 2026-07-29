#!/usr/bin/env python3
"""Export paired DLR+ramp100 E/F checkpoints to the strict QFG graph.

DLR, the BatchNorm running-statistics policy, and TSS are training-time
choices.  The deployed graph is therefore the same strict head-free
TPD+NER+QFG inference graph used by C/D.  This adapter validates the
ramp100-owned checkpoint identity before reusing the frozen state filtering
and strict-load implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    export_tpd_ner_v4_qfg_v2_croa_to_inference as base_export,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as trainer,
)


EXPORT_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_inference_export_v1"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_export_action_v1"
)
SUPPORTED_VARIANTS = tuple(trainer.SUPPORTED_CANDIDATE_VARIANTS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _regular_file_bytes(path: Path, label: str) -> bytes:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"{label} is not a regular file: {value}")
    return value.read_bytes()


def _canonical(value: Any) -> Any:
    return json_ready(value)


def json_ready(value: Any) -> Any:
    import json

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    )


def require_ramp100_checkpoint_payload(
    payload: Any,
    *,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    """Validate the trainer-owned identity needed at the export boundary."""

    _require(isinstance(payload, Mapping), "DLR checkpoint is not a mapping")
    value = copy.deepcopy(dict(payload))
    variant = value.get("variant")
    _require(variant in SUPPORTED_VARIANTS, "DLR checkpoint variant differs")
    if expected_variant is not None:
        _require(variant == expected_variant, "DLR checkpoint variant differs")
    candidate = trainer.candidate_contract(str(variant))
    identity = trainer.require_paired_run_identity(
        value.get("run_identity"),
        label="DLR export checkpoint",
        expected_variant=str(variant),
    )
    expected = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "variant": variant,
        "candidate_variant": variant,
        "base_model_variant": candidate["base_model_variant"],
        "qfg_variant": trainer.v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "family_recipe": trainer.FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "dataset": "NUDT-SIRST",
        "seed": trainer.TRAINING_SEED,
        "split_seed": trainer.SPLIT_SEED,
        "architecture_id": identity["architecture_id"],
        "checkpoint_identity": trainer._checkpoint_identity(identity),
        "source_locks": identity["source_locks"],
        "exact_source_lock_sha256": identity["source_locks"][
            trainer.SOURCE_LOCK_KEY
        ],
        "upstream_source_lock_sha256": trainer.UPSTREAM_SOURCE_LOCK_SHA256,
        "parent_checkpoint_path": str(trainer.PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": trainer.PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_role": trainer.PARENT_CHECKPOINT_ROLE_SHORT,
        "parent_checkpoint_epoch": trainer.PARENT_CHECKPOINT_EPOCH,
        "parent_checkpoint_state_dict_sha256": (
            trainer.PARENT_STATE_DICT_SHA256
        ),
        "survival_weight_schedule": trainer.survival_schedule_contract(
            str(variant)
        ),
        trainer.TSS_WEIGHT_SCHEDULE_FIELD: candidate["weight_schedule_id"],
        trainer.SURVIVAL_WEIGHT_MAX_FIELD: candidate["survival_weight_max"],
        "optimizer_recipe": trainer.optimizer_recipe_contract(),
        "batchnorm_recipe": trainer.batchnorm_recipe_contract(),
        "selection_uses_survival_loss": False,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }
    for name, required in expected.items():
        _require(
            _canonical(value.get(name)) == _canonical(required),
            f"DLR checkpoint {name} differs",
        )
    epoch = value.get("epoch")
    _require(
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and 1 <= epoch <= trainer.FORMAL_EPOCHS,
        "DLR checkpoint epoch is invalid",
    )
    _require(
        value.get("checkpoint_role")
        in {
            "last_evaluated_epoch",
            "best_validation_pd_primary",
            "best_validation_miou_secondary",
        },
        "DLR checkpoint role is invalid",
    )
    _require(
        value.get(trainer.SURVIVAL_WEIGHT_FIELD)
        == trainer.survival_weight_for_epoch(str(variant), epoch),
        "DLR checkpoint effective TSS weight differs",
    )
    _require(
        value.get(trainer.TSS_RAMP_FRACTION_FIELD)
        == trainer.candidate_ramp_fraction(str(variant), epoch),
        "DLR checkpoint TSS ramp fraction differs",
    )
    expected_scheduler = {
        "kind": "identity_bound_manual_group_scaled_schedule",
        "completed_epoch": epoch,
        "checkpoint_group_lr": "manual_cosine_lr(completed_epoch)",
        "next_epoch_reapplies_multipliers": True,
        "tss_weight_state": "derived_from_epoch_not_serialized",
    }
    _require(
        value.get("scheduler") == expected_scheduler,
        "DLR checkpoint scheduler contract differs",
    )
    metadata = trainer._require_model_metadata(
        value.get("model_metadata"),
        variant=str(variant),
    )
    _require(
        _canonical(value.get("architecture_manifest"))
        == _canonical(metadata["architecture_manifest"]),
        "DLR checkpoint architecture manifest differs",
    )
    state_dict = value.get("state_dict")
    _require(isinstance(state_dict, Mapping), "DLR checkpoint lacks state_dict")
    # This exact filter also checks the complete training-state key set and all
    # QFG/TSS prefixes; no deployment file is written here.
    base_export.strip_survival_state_dict(state_dict)
    _require(
        isinstance(value.get("validation_metrics"), Mapping),
        "DLR checkpoint validation metrics are missing",
    )
    return value


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite DLR export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    serialized = io.BytesIO()
    torch.save(dict(payload), serialized)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized.getbuffer())
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite DLR export: {output}"
            ) from error
        directory_descriptor = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def export_ramp100_qfg_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    content = _regular_file_bytes(checkpoint_path, "DLR source checkpoint")
    payload = torch.load(
        io.BytesIO(content),
        map_location="cpu",
        weights_only=False,
    )
    validated = require_ramp100_checkpoint_payload(
        payload,
        expected_variant=expected_variant,
    )
    state_dict = validated["state_dict"]
    inference_state = base_export.strip_survival_state_dict(state_dict)
    base_export.build_inference_model_from_training_state_dict(state_dict)
    exported = {
        "schema": EXPORT_SCHEMA,
        "source_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "source_checkpoint_sha256": hashlib.sha256(content).hexdigest(),
        "source_checkpoint_role": validated["checkpoint_role"],
        "source_checkpoint_epoch": validated["epoch"],
        "source_variant": validated["variant"],
        "source_checkpoint_identity": copy.deepcopy(
            validated["checkpoint_identity"]
        ),
        "source_run_identity": copy.deepcopy(validated["run_identity"]),
        "training_recipe": trainer.FAMILY_RECIPE,
        "state_dict": inference_state,
        "survival_state_removed": list(base_export.SURVIVAL_STATE_KEYS),
        "qfg_state_preserved": list(base_export.QFG_STATE_KEYS),
        "qfg_inference_required": True,
        "inference_heads_required": False,
        "inference_state_key_count": base_export.INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": base_export.INFERENCE_PARAMETER_COUNT,
    }
    _atomic_save(output_path, exported)
    return exported


def validate_exported_ramp100_qfg_checkpoint(
    output_path: Path,
    *,
    expected_source_checkpoint: Path | None = None,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    content = _regular_file_bytes(output_path, "DLR inference export")
    payload = torch.load(
        io.BytesIO(content),
        map_location="cpu",
        weights_only=False,
    )
    _require(isinstance(payload, Mapping), "DLR export is not a mapping")
    ready = dict(payload)
    _require(ready.get("schema") == EXPORT_SCHEMA, "DLR export schema differs")
    _require(
        ready.get("source_variant") in SUPPORTED_VARIANTS,
        "DLR export source variant differs",
    )
    if expected_variant is not None:
        _require(
            ready.get("source_variant") == expected_variant,
            "DLR export expected variant differs",
        )
    state_dict = ready.get("state_dict")
    _require(isinstance(state_dict, Mapping), "DLR export lacks state_dict")
    model, metadata = base_export.build_frozen_qfg_inference_model()
    incompatible = model.load_state_dict(state_dict, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "DLR export strict load is incompatible",
    )
    _require(
        len(state_dict) == base_export.INFERENCE_STATE_KEY_COUNT,
        "DLR export state-key count differs",
    )
    _require(
        metadata["total_parameters"] == base_export.INFERENCE_PARAMETER_COUNT,
        "DLR export parameter count differs",
    )
    source_path = Path(str(ready.get("source_checkpoint_path"))).resolve()
    source_sha = ready.get("source_checkpoint_sha256")
    _require(
        isinstance(source_sha, str)
        and len(source_sha) == 64
        and all(character in "0123456789abcdef" for character in source_sha),
        "DLR export source SHA is invalid",
    )
    if expected_source_checkpoint is not None:
        expected = Path(expected_source_checkpoint).resolve()
        _require(source_path == expected, "DLR export source path differs")
        _require(
            hashlib.sha256(
                _regular_file_bytes(expected, "expected DLR source checkpoint")
            ).hexdigest()
            == source_sha,
            "DLR export source content differs",
        )
    return {
        "schema": EXPORT_SCHEMA,
        "path": str(Path(output_path).resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_checkpoint_path": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_role": ready.get("source_checkpoint_role"),
        "source_checkpoint_epoch": ready.get("source_checkpoint_epoch"),
        "source_variant": ready.get("source_variant"),
        "inference_state_key_count": base_export.INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": base_export.INFERENCE_PARAMETER_COUNT,
        "strict_load": True,
        "survival_state_absent": True,
        "qfg_state_preserved": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-variant", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.verify:
        result = validate_exported_ramp100_qfg_checkpoint(
            args.output,
            expected_source_checkpoint=args.checkpoint,
            expected_variant=args.expected_variant,
        )
    else:
        exported = export_ramp100_qfg_checkpoint(
            args.checkpoint,
            args.output,
            expected_variant=args.expected_variant,
        )
        result = {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "export",
            "source_variant": exported["source_variant"],
            "output": str(args.output.resolve()),
        }
    import json

    print(
        json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False),
        flush=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "EXPORT_SCHEMA",
    "SUPPORTED_VARIANTS",
    "export_ramp100_qfg_checkpoint",
    "main",
    "parse_args",
    "require_ramp100_checkpoint_payload",
    "validate_exported_ramp100_qfg_checkpoint",
]


if __name__ == "__main__":
    main()
