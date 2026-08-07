#!/usr/bin/env python3
"""Freeze the immutable PBDR-V4 source lock and five-family candidate pool.

The two subcommands are deliberately separate.  ``freeze-source`` is run
before any V4 training and locks the already persisted split projection,
every audited V3/Current/Original authority, and the executable source set.
``freeze-pool`` is run after both V4 stages and accepts only a fully replayed
internal-validation sweep plus two complete, non-smoke selected checkpoints.

This module never imports a dataset loader and never opens an official-test
index.  Each command writes exactly one O_EXCL manifest and never overwrites
an existing path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Sequence

import torch


_IMPORT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_REPO_ROOT))

from experiments import pbdr_v4_candidate_pool as pool_io
from experiments import pbdr_v4_component_loss as component_loss
from experiments import pbdr_v4_internal_cache as cache_io
from experiments import pbdr_v4_models_seed42_v1 as current_registry
from experiments import pbdr_v4_original_models as original_registry
from experiments import pbdr_v3_residual_calibration as residual_calibration
from experiments import pbdr_v4_run_artifacts as run_artifacts
from experiments import pbdr_v4_source_lock as source_lock_io
from experiments import pbdr_v4_split_authority as split_authority
from experiments import pbdr_v4_training_core as training_core
from experiments import sweep_pbdr_v3_residual_calibration as residual_sweep
from experiments.pbdr_v4_state_contract import state_semantic_sha256
from experiments.pbdr_v4_zero_margin_selector import FROZEN_TIE_ORDER


REPO_ROOT = _IMPORT_REPO_ROOT
DATASETS = split_authority.DATASETS
ROLES = ("best_miou", "best_pd")
TRAINER_SCHEMA = "sctransnet_three_dataset_pbdr_v4_training_v1/v1"
EVALUATOR_SCHEMA = "sctransnet_three_dataset_pbdr_v4_evaluator_v1/v1"
V3_CALIBRATED_SCHEMA = "sctransnet_pbdr_v4_v3_calibrated_candidate/v1"
SELECTED_CHECKPOINT_NAME = "selected_candidate.pth.tar"
SUMMARY_NAME = "summary.json"
RUN_PROTOCOL_NAME = "run_protocol.json"
FORMAL_BATCH_SIZE = 16
FORMAL_WORKERS = 0
FORMAL_GPU_UUIDS: Mapping[str, str] = {
    "NUDT-SIRST": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "IRSTD-1K": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "NUAA-SIRST": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
PROTOCOL_SOURCE_RELATIVE_PATHS = (
    "experiments/evaluate_three_dataset_pbdr_v4_v1.py",
    "experiments/freeze_pbdr_v4_protocol.py",
    "experiments/launch_three_dataset_pbdr_v4_v1.py",
    "experiments/pbdr_v4_models_seed42_v1.py",
    "experiments/pbdr_v4_original_models.py",
    "experiments/prepare_pbdr_v4_internal_artifacts.py",
    "experiments/sweep_pbdr_v3_residual_calibration.py",
    "experiments/train_three_dataset_pbdr_v4_v1.py",
)

V3_RUN_DIRECTORIES: Mapping[str, Mapping[str, Path]] = {
    "NUAA-SIRST": {
        role: REPO_ROOT
        / "results/nuaa_pbdr_v3_stage1_v1/formal"
        / role
        / "core"
        for role in ROLES
    },
    **{
        dataset: {
            role: REPO_ROOT
            / "results/two_dataset_pbdr_v3_stage1_v1/runs"
            / dataset
            / "formal"
            / role
            / "core"
            for role in ROLES
        }
        for dataset in ("NUDT-SIRST", "IRSTD-1K")
    },
}


class PBDRV4ProtocolFreezeError(ValueError):
    """A freeze input is mutable, incomplete, tampered, or out of scope."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4ProtocolFreezeError(message)


def _sha256(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    _require(
        not candidate.is_symlink() and candidate.is_file(),
        f"{label} must be a regular non-symlink file: {candidate}",
    )
    return candidate.resolve(strict=True)


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    candidate = _regular_file(path, label=label)
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4ProtocolFreezeError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value, raw


def _json_ready(value: Any, *, label: str) -> Any:
    """Return a deterministic JSON projection or fail instead of stringifying."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            _require(
                type(raw_key) in (str, int),
                f"{label} contains an unsupported mapping key",
            )
            key = raw_key if type(raw_key) is str else str(raw_key)
            _require(
                key not in normalized,
                f"{label} contains colliding JSON mapping keys",
            )
            normalized[key] = item
        return {
            key: _json_ready(item, label=f"{label}.{key}")
            for key, item in sorted(normalized.items())
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise PBDRV4ProtocolFreezeError(
        f"{label} is not replayable JSON configuration: {type(value).__name__}"
    )


def _configuration_sha256(
    *,
    family: str,
    dataset: str,
    role: str,
    configuration: Mapping[str, Any],
) -> str:
    """Mirror evaluator.candidate_configuration_sha256 exactly."""

    _require(family in FROZEN_TIE_ORDER, f"unsupported family: {family}")
    payload = {
        "schema": EVALUATOR_SCHEMA,
        "family": family,
        "dataset": dataset,
        "role": role,
        "fixed_probability_rule": "strict_greater_than_0.5",
        "details": _json_ready(configuration, label=f"{family} configuration"),
    }
    return pool_io.canonical_sha256(payload)


def _validate_persisted_projection(path: Path) -> tuple[dict[str, Any], Path]:
    """Require the persisted projection to equal all three live authorities."""

    observed, _ = _read_json_object(path, label="split projection")
    expected = split_authority.build_projection()
    _require(observed == expected, "persisted split projection differs from live authority")
    declared = _sha256(observed.get("projection_sha256"), name="split projection SHA")
    unsigned = dict(observed)
    del unsigned["projection_sha256"]
    _require(
        declared == split_authority.canonical_sha256(unsigned),
        "split projection SHA does not replay",
    )
    return observed, _regular_file(path, label="split projection")


def _validate_runtime_source_record(
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    _require(isinstance(record, Mapping), f"{label} runtime source record differs")
    path = _regular_file(Path(str(record.get("path"))), label=label)
    _require(
        type(record.get("bytes")) is int and path.stat().st_size == record["bytes"],
        f"{label} runtime source byte count differs",
    )
    _require(
        source_lock_io.file_sha256(path)
        == _sha256(record.get("sha256"), name=f"{label} runtime source SHA"),
        f"{label} runtime source SHA differs",
    )
    return path


def _merge_runtime_source_records(
    records_by_registry: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[tuple[str, ...], dict[str, Path]]:
    """Merge V3 runtime records into sources; route outside files externally."""

    root = Path(repo_root).resolve(strict=True)
    relative_paths = set(source_lock_io.DEFAULT_SOURCE_RELATIVE_PATHS)
    relative_paths.update(PROTOCOL_SOURCE_RELATIVE_PATHS)
    external: dict[str, Path] = {}
    relative_bindings: dict[str, tuple[int, str]] = {}
    for registry_name, records in sorted(records_by_registry.items()):
        _require(isinstance(records, Mapping), f"{registry_name} runtime records differ")
        for source_name, record in sorted(records.items()):
            path = _validate_runtime_source_record(
                record,
                label=f"{registry_name}:{source_name}",
            )
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                external[f"runtime_source::{registry_name}::{source_name}"] = path
                continue
            binding = (path.stat().st_size, source_lock_io.file_sha256(path))
            previous = relative_bindings.setdefault(relative, binding)
            _require(previous == binding, f"runtime source binding conflicts: {relative}")
            relative_paths.add(relative)
    return tuple(sorted(relative_paths)), external


def _merged_runtime_sources() -> tuple[tuple[str, ...], dict[str, Path]]:
    from experiments import three_dataset_pbdr_v3_models_seed42_v1 as nuaa_v3
    from experiments import two_dataset_pbdr_v3_models_seed42_v1 as cross_v3

    return _merge_runtime_source_records(
        {
            "nuaa_v3": nuaa_v3.runtime_source_records(),
            "cross_v3": cross_v3.runtime_source_records(),
        }
    )


def _add_external(
    destination: MutableMapping[str, Path],
    name: str,
    path: Path,
) -> None:
    _require(type(name) is str and bool(name), "external binding name is invalid")
    ready = _regular_file(path, label=f"external file {name}")
    previous = destination.get(name)
    _require(previous is None or previous == ready, f"external binding conflicts: {name}")
    destination[name] = ready


def _add_runtime_source_externals(
    destination: MutableMapping[str, Path],
    *,
    prefix: str,
    records: Mapping[str, Any],
) -> None:
    for source_name, raw in sorted(records.items()):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            continue
        path = _regular_file(Path(str(raw["path"])), label=f"{prefix}:{source_name}")
        _require(
            pool_io.file_sha256(path)
            == _sha256(raw.get("sha256"), name=f"{prefix}:{source_name} SHA"),
            f"{prefix}:{source_name} source SHA differs",
        )
        _add_external(destination, f"{prefix}::{source_name}", path)


def _validate_historical_protocol(
    path: Path,
    *,
    dataset: str,
    method: str,
) -> dict[str, Any]:
    protocol, _ = _read_json_object(path, label=f"{dataset}/{method} protocol")
    _require(
        protocol.get("dataset") == dataset
        and protocol.get("method") == method
        and protocol.get("training_seed") == 42
        and protocol.get("test_selected") is True
        and protocol.get("selection_is_optimistic") is True,
        f"{dataset}/{method} historical protocol identity differs",
    )
    declared = _sha256(
        protocol.get("protocol_sha256"), name=f"{dataset}/{method} protocol SHA"
    )
    unsigned = dict(protocol)
    del unsigned["protocol_sha256"]
    _require(
        declared == pool_io.canonical_sha256(unsigned),
        f"{dataset}/{method} historical protocol self-hash differs",
    )
    return protocol


def _validate_current_authority_files(
    dataset: str,
    *,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint_path = _regular_file(
        Path(str(checkpoint_record.get("path"))),
        label=f"{dataset} Current authority checkpoint",
    )
    _require(
        checkpoint_path.parent.name == "checkpoints",
        f"{dataset} Current checkpoint authority layout differs",
    )
    root = checkpoint_path.parent.parent
    summary_path = _regular_file(root / "summary.json", label=f"{dataset} Current summary")
    protocol_path = _regular_file(root / "protocol.json", label=f"{dataset} Current protocol")
    summary, _ = _read_json_object(summary_path, label=f"{dataset} Current summary")
    protocol = _validate_historical_protocol(
        protocol_path, dataset=dataset, method="final"
    )
    _require(
        summary.get("schema") == "sctransnet_three_dataset_tss_off_seed42_v1/v1"
        and summary.get("status") == "complete"
        and summary.get("dataset") == dataset
        and summary.get("method") == "final"
        and summary.get("seed") == 42
        and summary.get("tss_enabled") is False
        and summary.get("protocol") == str(protocol_path)
        and summary.get("protocol_sha256") == protocol.get("protocol_sha256"),
        f"{dataset} Current summary/protocol binding differs",
    )
    _require(
        checkpoint_payload.get("protocol_sha256")
        == checkpoint_record.get("protocol_sha256")
        == protocol.get("protocol_sha256"),
        f"{dataset} Current checkpoint/protocol authority binding differs",
    )
    if dataset != "NUAA-SIRST":
        from experiments import two_dataset_pbdr_v3_models_seed42_v1 as cross_v3

        audit = cross_v3.audit_current_run(dataset)
        _require(
            audit.get("summary", {}).get("path") == str(summary_path)
            and audit.get("protocol", {}).get("path") == str(protocol_path),
            f"{dataset} Current registry authority paths differ",
        )
    return summary_path, protocol_path, protocol


def _validate_original_protocol_from_record(
    dataset: str,
    *,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    audit = checkpoint_record.get("authority_audit")
    if isinstance(audit, Mapping):
        binding = audit.get("protocol")
        _require(isinstance(binding, Mapping), f"{dataset} Original authority protocol differs")
        protocol_path = Path(str(binding.get("path")))
    else:
        source_path = _regular_file(
            Path(str(checkpoint_record.get("source_path"))),
            label=f"{dataset} Original source checkpoint",
        )
        _require(
            source_path.parent.name == "checkpoints",
            f"{dataset} Original source authority layout differs",
        )
        protocol_path = source_path.parent.parent / "protocol.json"
    protocol_path = _regular_file(protocol_path, label=f"{dataset} Original protocol")
    protocol = _validate_historical_protocol(
        protocol_path, dataset=dataset, method="original"
    )
    _require(
        checkpoint_payload.get("protocol_sha256")
        == checkpoint_record.get("protocol_sha256")
        == protocol.get("protocol_sha256"),
        f"{dataset} Original checkpoint/protocol authority binding differs",
    )
    if isinstance(audit, Mapping):
        binding = audit["protocol"]
        _require(
            binding.get("canonical_sha256") == protocol.get("protocol_sha256")
            and binding.get("file_sha256") == pool_io.file_sha256(protocol_path),
            f"{dataset} Original registry protocol bytes differ",
        )
    return protocol_path, protocol


@dataclass(frozen=True, slots=True)
class V3Evidence:
    run: Any
    state_sha256: str
    architecture_binding: Mapping[str, Any]


def _validate_v3_run(dataset: str, role: str) -> V3Evidence:
    """Validate one complete V3 run and strict-load its deployment graph."""

    _require(dataset in DATASETS and role in ROLES, "V3 dataset/role differs")
    run_dir = V3_RUN_DIRECTORIES[dataset][role]
    if dataset == "NUAA-SIRST":
        from experiments import evaluate_nuaa_pbdr_v3_stage1_v1 as evaluator
        from experiments import three_dataset_pbdr_v3_models_seed42_v1 as models

        run = evaluator.validate_completed_run(run_dir)
        _require(run.parent_role == role and run.recipe == "core", "NUAA V3 run differs")
        model, metadata = models.build_inference_model_from_candidate_state(
            run.candidate_state,
            parent_role=role,
            parent_checkpoint=Path(str(run.parent_checkpoint["path"])),
        )
        state_sha = models.tensor_mapping_sha256(run.candidate_state)
        registry_schema = models.SCHEMA
    else:
        from experiments import evaluate_two_dataset_pbdr_v3_stage1_v1 as evaluator
        from experiments import two_dataset_pbdr_v3_models_seed42_v1 as models

        run = evaluator.validate_completed_run(dataset, run_dir)
        _require(run.parent_role == role, "cross-dataset V3 run role differs")
        model, metadata = models.build_inference_model_from_candidate_state(
            run.candidate_state,
            dataset_name=dataset,
            parent_role=role,
        )
        state_sha = models.tensor_mapping_sha256(run.candidate_state)
        _require(
            state_sha == run.candidate_state_sha256,
            "cross-dataset V3 candidate state SHA differs",
        )
        registry_schema = models.SCHEMA
    _require(metadata.get("strict_load") is True, "V3 inference load is not strict")
    _require(
        metadata.get("base_bitwise_equal_to_parent") is True,
        "V3 candidate base is not bitwise Current",
    )
    del model
    architecture = {
        "registry_schema": registry_schema,
        "trainer_protocol_sha256": run.protocol_sha256,
        "trainer_recipe": run.recipe if hasattr(run, "recipe") else "core",
        "training_state_key_count": metadata.get("training_state_key_count"),
        "inference_state_key_count": metadata.get("inference_state_key_count"),
        "builder_validation": metadata.get("builder_validation"),
        "raw_builder_metadata": metadata.get("raw_builder_metadata"),
        "strict_load": True,
        "base_bitwise_equal_to_parent": True,
    }
    return V3Evidence(
        run=run,
        state_sha256=_sha256(state_sha, name="V3 candidate state SHA"),
        architecture_binding=_json_ready(architecture, label="V3 architecture binding"),
    )


def _collect_authority_external_files(
    projection: Mapping[str, Any],
) -> dict[str, Path]:
    """Validate and collect every pre-V4 authority file for all six roles."""

    external: dict[str, Path] = {}
    datasets = projection.get("datasets")
    _require(isinstance(datasets, Mapping), "split projection datasets differ")
    for dataset in DATASETS:
        record = datasets.get(dataset)
        _require(isinstance(record, Mapping), f"split projection lacks {dataset}")
        _add_external(
            external,
            f"split_source::{dataset}",
            Path(str(record.get("source_path"))),
        )

    protocol_document = REPO_ROOT / "experiments/PBDR_V3_PROTOCOL.md"
    _add_external(external, "authority::pbdr_v3_protocol", protocol_document)
    _add_external(
        external,
        "authority::pbdr_v3_cross_dataset_protocol",
        REPO_ROOT / "experiments/PBDR_V3_CROSS_DATASET_PROTOCOL.md",
    )
    _add_external(
        external,
        "authority::cross_current_manifest",
        REPO_ROOT / "experiments/two_dataset_pbdr_v3_current_manifest_seed42_v1.json",
    )
    _add_external(
        external,
        "authority::cross_original_manifest",
        REPO_ROOT / "experiments/two_dataset_pbdr_v3_original_manifest_seed42_v1.json",
    )
    _add_external(
        external,
        "authority::original_checkpoint_manifest",
        original_registry.NUAA_AUTHORITY_MANIFEST_PATH,
    )

    for dataset in DATASETS:
        authority_current_payload, _, authority_current_record = (
            current_registry.load_current_checkpoint(dataset, ROLES[0])
        )
        current_summary, current_protocol_path, current_protocol = (
            _validate_current_authority_files(
                dataset,
                checkpoint_payload=authority_current_payload,
                checkpoint_record=authority_current_record,
            )
        )
        _add_external(external, f"current_summary::{dataset}", current_summary)
        _add_external(external, f"current_protocol::{dataset}", current_protocol_path)
        current_runtime = current_protocol.get("runtime_sources")
        _require(isinstance(current_runtime, Mapping), f"{dataset} Current runtime sources differ")
        _add_runtime_source_externals(
            external,
            prefix=f"current_runtime::{dataset}",
            records=current_runtime,
        )
        authority_original_payload, _, authority_original_record = (
            original_registry.load_original_checkpoint(dataset, ROLES[0])
        )
        original_protocol_path, _ = _validate_original_protocol_from_record(
            dataset,
            checkpoint_payload=authority_original_payload,
            checkpoint_record=authority_original_record,
        )
        _add_external(
            external,
            f"original_protocol::{dataset}",
            original_protocol_path,
        )
        for role in ROLES:
            v3 = _validate_v3_run(dataset, role)
            run = v3.run
            _add_external(external, f"v3_candidate::{dataset}::{role}", run.candidate_path)
            _add_external(external, f"v3_summary::{dataset}::{role}", run.summary_path)
            _add_external(external, f"v3_protocol::{dataset}::{role}", run.protocol_path)
            _add_external(external, f"v3_split::{dataset}::{role}", run.split_path)
            _add_external(
                external,
                f"v3_internal_certification::{dataset}::{role}",
                run.run_dir / "internal_certification.json",
            )
            _add_runtime_source_externals(
                external,
                prefix=f"v3_runtime::{dataset}::{role}",
                records=run.runtime_sources,
            )

            binding = run.split_manifest.get("data_protocol_manifest")
            _require(
                isinstance(binding, Mapping) and isinstance(binding.get("path"), str),
                f"V3 data-protocol manifest binding differs: {dataset}/{role}",
            )
            _add_external(
                external,
                f"data_protocol_manifest::{dataset}",
                Path(str(binding["path"])),
            )

            _, _, current_record = current_registry.load_current_checkpoint(dataset, role)
            current_model, current_metadata = (
                current_registry.build_frozen_current_reference_model(dataset, role, "stage1")
            )
            _require(
                current_metadata.get("base_logits_are_current") is True,
                f"Current strict reference differs: {dataset}/{role}",
            )
            del current_model
            _add_external(
                external,
                f"current_checkpoint::{dataset}::{role}",
                Path(str(current_record["path"])),
            )

            _, _, original_record = original_registry.load_original_checkpoint(dataset, role)
            original_model, original_metadata = original_registry.build_original_inference_model(
                dataset, role
            )
            _require(
                original_metadata.get("strict_load") is True,
                f"Original strict reference differs: {dataset}/{role}",
            )
            del original_model
            _add_external(
                external,
                f"original_checkpoint::{dataset}::{role}",
                Path(str(original_record["path"])),
            )

            authority = original_record.get("authority_manifest")
            if isinstance(authority, Mapping) and isinstance(authority.get("path"), str):
                _add_external(
                    external,
                    "authority::original_checkpoint_manifest",
                    Path(str(authority["path"])),
                )
    return external


def freeze_source(*, split_projection_path: Path, output_path: Path) -> Path:
    """Validate all pretraining authorities, then write one immutable lock."""

    training_core.configure_determinism()
    projection, projection_path = _validate_persisted_projection(split_projection_path)
    source_paths, runtime_external = _merged_runtime_sources()
    external = _collect_authority_external_files(projection)
    for name, path in runtime_external.items():
        _add_external(external, name, path)
    _add_external(external, "split_projection", projection_path)
    payload = source_lock_io.build_source_lock(
        source_root=REPO_ROOT,
        source_relative_paths=source_paths,
        external_files=external,
    )
    return source_lock_io.write_source_lock_exclusive(output_path, payload).resolve(
        strict=True
    )


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    family: str
    name: str
    kind: str
    artifact_path: Path
    state_sha256: str
    configuration: Mapping[str, Any]

    def candidate(self, *, dataset: str, role: str) -> pool_io.CandidateArtifact:
        path = _regular_file(self.artifact_path, label=f"{self.family} artifact")
        return pool_io.CandidateArtifact(
            family=self.family,
            name=self.name,
            kind=self.kind,
            artifact_path=str(path),
            artifact_sha256=pool_io.file_sha256(path),
            state_sha256=_sha256(
                self.state_sha256,
                name=f"{self.family} semantic state SHA",
            ),
            configuration_sha256=_configuration_sha256(
                family=self.family,
                dataset=dataset,
                role=role,
                configuration=self.configuration,
            ),
        )


@dataclass(frozen=True, slots=True)
class PoolEvidence:
    source_lock_sha256: str
    split_projection_sha256: str
    families: tuple[FamilyEvidence, ...]

    def __post_init__(self) -> None:
        _sha256(self.source_lock_sha256, name="source-lock SHA")
        _sha256(self.split_projection_sha256, name="split-projection SHA")
        _require(
            tuple(item.family for item in self.families) == tuple(FROZEN_TIE_ORDER),
            "family evidence order differs from frozen pool",
        )


def _assert_checkpoint_binding(
    record: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(isinstance(binding, Mapping), f"{label} cache checkpoint binding differs")
    path = _regular_file(Path(str(record.get("path"))), label=f"{label} checkpoint")
    expected = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": pool_io.file_sha256(path),
        "state_sha256": _sha256(record.get("state_sha256"), name=f"{label} state SHA"),
    }
    _require(dict(binding) == expected, f"{label} cache checkpoint binding differs")


def _baseline_configuration(*, state_sha256: str) -> dict[str, Any]:
    return {
        "state_sha256": _sha256(state_sha256, name="baseline state SHA"),
        "inference": "raw_final_logits",
    }


def _fraction_json(value: object) -> object:
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, (float, int, str)) or value is None:
        return value
    return str(value)


def _candidate_manifest_sha(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("state_dict", None)
    unsigned.pop("candidate_manifest_sha256", None)
    return pool_io.canonical_sha256(unsigned)


def _v3_calibrated_manifest_sha(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("state_dict", None)
    unsigned.pop("artifact_manifest_sha256", None)
    return pool_io.canonical_sha256(unsigned)


def _semantic_value_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and tuple(first.shape) == tuple(second.shape)
            and torch.equal(first.detach().cpu(), second.detach().cpu())
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_semantic_value_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
            and len(first) == len(second)
            and all(_semantic_value_equal(a, b) for a, b in zip(first, second))
        )
    return type(first) is type(second) and first == second


def _validate_v3_calibrated_artifact(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(
        payload.get("schema") == V3_CALIBRATED_SCHEMA
        and payload.get("family") == "V3-calibrated",
        "V3 calibrated artifact identity differs",
    )
    _require(
        payload.get("dataset") in DATASETS and payload.get("role") in ROLES,
        "V3 calibrated artifact dataset/role differs",
    )
    _require(
        payload.get("selected_on") == "internal_validation"
        and payload.get("official_test_accessed") is False
        and payload.get("performance_acceptance_margin") is None,
        "V3 calibrated artifact scope differs",
    )
    state = payload.get("state_dict")
    _require(
        isinstance(state, Mapping)
        and bool(state)
        and all(type(name) is str and isinstance(tensor, torch.Tensor) for name, tensor in state.items()),
        "V3 calibrated artifact state_dict differs",
    )
    state_sha = state_semantic_sha256(state)  # type: ignore[arg-type]
    _require(
        payload.get("state_key_count") == len(state)
        and payload.get("state_sha256") == state_sha
        and payload.get("state_semantic_sha256") == state_sha,
        "V3 calibrated artifact semantic state SHA differs",
    )
    calibration_raw = payload.get("calibration")
    _require(isinstance(calibration_raw, Mapping), "V3 calibration fields differ")
    _require(
        set(calibration_raw) == {"positive_scale", "negative_scale", "bias"},
        "V3 calibration must contain exactly three scalars",
    )
    try:
        calibration = residual_calibration.ResidualCalibration(**dict(calibration_raw))
    except (TypeError, ValueError) as error:
        raise PBDRV4ProtocolFreezeError(f"V3 calibration is invalid: {error}") from error
    _require(calibration.as_dict() == dict(calibration_raw), "V3 calibration scalar encoding differs")
    expected_configuration_sha = _configuration_sha256(
        family="V3-calibrated",
        dataset=str(payload["dataset"]),
        role=str(payload["role"]),
        configuration={
            "state_sha256": state_sha,
            "calibration": calibration.as_dict(),
            "selected_on": "internal_validation",
        },
    )
    _require(
        payload.get("configuration_sha256") == expected_configuration_sha,
        "V3 calibrated evaluator configuration SHA differs",
    )
    _sha256(payload.get("source_lock_sha256"), name="V3 source-lock SHA")
    _sha256(payload.get("split_projection_sha256"), name="V3 split-projection SHA")
    _require(
        isinstance(payload.get("sweep_binding"), Mapping)
        and isinstance(payload.get("cache_binding"), Mapping)
        and isinstance(payload.get("v3_candidate_binding"), Mapping),
        "V3 calibrated artifact provenance bindings differ",
    )
    _require(
        payload.get("artifact_manifest_sha256") == _v3_calibrated_manifest_sha(payload),
        "V3 calibrated artifact self-manifest SHA differs",
    )
    if expected is not None:
        _require(
            _semantic_value_equal(payload, expected),
            "existing V3 calibrated artifact differs from strict replay",
        )
    return dict(payload)


def _commit_or_replay_v3_calibrated_artifact(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Create with O_EXCL, or accept only an exact semantic prior commit."""

    expected = _validate_v3_calibrated_artifact(payload)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        _require(
            destination.is_file() and not destination.is_symlink(),
            "existing V3 calibrated artifact path is unsafe",
        )
        observed = run_artifacts.load_torch_artifact(destination)
        _validate_v3_calibrated_artifact(observed, expected=expected)
        return destination.resolve(strict=True)
    committed = run_artifacts.exclusive_torch_save(destination, expected)
    observed = run_artifacts.load_torch_artifact(committed)
    _validate_v3_calibrated_artifact(observed, expected=expected)
    return committed.resolve(strict=True)


def _build_v3_calibrated_artifact(
    *,
    path: Path,
    dataset: str,
    role: str,
    v3: V3Evidence,
    sweep: Mapping[str, Any],
    sweep_path: Path,
    cache: cache_io.ValidatedInternalRawLogitCache,
    configuration_sha256: str,
    source_lock_sha256: str,
    split_projection_sha256: str,
) -> tuple[Path, str]:
    selected = sweep.get("selected")
    _require(isinstance(selected, Mapping), "V3 sweep selected record differs")
    config_raw = selected.get("config")
    _require(isinstance(config_raw, Mapping), "V3 sweep selected calibration differs")
    _require(
        set(config_raw) == {"positive_scale", "negative_scale", "bias"},
        "V3 sweep selected calibration is not the three-scalar contract",
    )
    try:
        calibration = residual_calibration.ResidualCalibration(**dict(config_raw))
    except (TypeError, ValueError) as error:
        raise PBDRV4ProtocolFreezeError(f"V3 selected calibration is invalid: {error}") from error
    grid_index = selected.get("grid_index")
    grid = residual_calibration.calibration_grid()
    _require(
        type(grid_index) is int
        and 0 <= grid_index < len(grid)
        and grid[grid_index] == calibration
        and selected.get("name") == calibration.name,
        "V3 selected calibration does not replay frozen grid order",
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in v3.run.candidate_state.items()
    }
    semantic_sha = state_semantic_sha256(state)
    expected_configuration_sha = _configuration_sha256(
        family="V3-calibrated",
        dataset=dataset,
        role=role,
        configuration={
            "state_sha256": semantic_sha,
            "calibration": calibration.as_dict(),
            "selected_on": "internal_validation",
        },
    )
    _require(
        configuration_sha256 == expected_configuration_sha,
        "V3 supplied configuration SHA differs from evaluator contract",
    )
    sweep_path = _regular_file(sweep_path, label="V3 calibration sweep result")
    identity = cache.manifest.get("identity")
    _require(isinstance(identity, Mapping), "V3 cache identity differs")
    cache_binding = sweep.get("cache_binding")
    _require(isinstance(cache_binding, Mapping), "V3 sweep cache binding differs")
    payload: dict[str, Any] = {
        "schema": V3_CALIBRATED_SCHEMA,
        "family": "V3-calibrated",
        "dataset": dataset,
        "role": role,
        "state_dict": state,
        "state_key_count": len(state),
        "state_sha256": semantic_sha,
        "state_semantic_sha256": semantic_sha,
        "calibration": calibration.as_dict(),
        "configuration_sha256": expected_configuration_sha,
        "selected_on": "internal_validation",
        "v3_candidate_binding": {
            "path": str(v3.run.candidate_path),
            "bytes": v3.run.candidate_path.stat().st_size,
            "file_sha256": v3.run.candidate_sha256,
            "registry_state_sha256": v3.state_sha256,
            "state_semantic_sha256": semantic_sha,
        },
        "sweep_binding": {
            "path": str(sweep_path),
            "bytes": sweep_path.stat().st_size,
            "file_sha256": pool_io.file_sha256(sweep_path),
            "result_sha256": sweep.get("result_sha256"),
            "selected_grid_index": grid_index,
        },
        "cache_binding": dict(cache_binding),
        "cache_identity_sha256": identity.get("identity_sha256"),
        "source_lock_sha256": source_lock_sha256,
        "split_projection_sha256": split_projection_sha256,
        "fixed_probability_rule": "strict_greater_than_0.5",
        "performance_acceptance_margin": None,
        "official_test_accessed": False,
    }
    payload["artifact_manifest_sha256"] = _v3_calibrated_manifest_sha(payload)
    return _commit_or_replay_v3_calibrated_artifact(path, payload), semantic_sha


def _validate_selected_v4_checkpoint(
    path: Path,
    *,
    dataset: str,
    role: str,
    stage: str,
    source_lock_sha256: str,
    split_projection_sha256: str,
    current_record: Mapping[str, Any],
    expected_stage1_file_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Replay one complete selected checkpoint and its sibling summary."""

    checkpoint_path = _regular_file(path, label=f"V4 {stage} checkpoint")
    _require(
        checkpoint_path.name == SELECTED_CHECKPOINT_NAME,
        f"V4 {stage} path is not the selected checkpoint",
    )
    summary_path = checkpoint_path.parent / SUMMARY_NAME
    summary, raw_summary = _read_json_object(summary_path, label=f"V4 {stage} summary")
    expected_summary_bytes = (
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(raw_summary == expected_summary_bytes, f"V4 {stage} summary is not canonical")
    _require(
        summary.get("schema") == TRAINER_SCHEMA and summary.get("status") == "complete",
        f"V4 {stage} training summary is incomplete",
    )
    declared_summary_sha = _sha256(
        summary.get("summary_sha256"), name=f"V4 {stage} summary SHA"
    )
    unsigned_summary = dict(summary)
    del unsigned_summary["summary_sha256"]
    _require(
        declared_summary_sha == pool_io.canonical_sha256(unsigned_summary),
        f"V4 {stage} summary self-hash differs",
    )
    _require(
        (summary.get("dataset"), summary.get("role"), summary.get("stage"))
        == (dataset, role, stage),
        f"V4 {stage} summary identity differs",
    )
    _require(
        summary.get("official_test_accessed") is False
        and summary.get("performance_acceptance_margin") is None
        and summary.get("smoke") is False,
        f"V4 {stage} summary scope differs",
    )
    _require(
        Path(str(summary.get("selected_checkpoint"))) == checkpoint_path,
        f"V4 {stage} selected path differs",
    )

    protocol_path = checkpoint_path.parent / RUN_PROTOCOL_NAME
    protocol, raw_protocol = _read_json_object(
        protocol_path, label=f"V4 {stage} run protocol"
    )
    expected_protocol_bytes = (
        json.dumps(
            protocol,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(
        raw_protocol == expected_protocol_bytes,
        f"V4 {stage} run protocol is not canonical",
    )
    _require(
        protocol.get("schema") == f"{TRAINER_SCHEMA}/run_protocol",
        f"V4 {stage} run protocol schema differs",
    )
    declared_protocol_sha = _sha256(
        protocol.get("protocol_sha256"), name=f"V4 {stage} run protocol SHA"
    )
    unsigned_protocol = dict(protocol)
    del unsigned_protocol["protocol_sha256"]
    _require(
        declared_protocol_sha == pool_io.canonical_sha256(unsigned_protocol),
        f"V4 {stage} run protocol self-hash differs",
    )
    formal_epochs = 150 if stage == "stage1" else 50
    _require(
        protocol.get("epochs") == formal_epochs
        and protocol.get("eval_every") == 5
        and protocol.get("batch_size") == FORMAL_BATCH_SIZE
        and protocol.get("workers") == FORMAL_WORKERS,
        f"V4 {stage} formal 150/50-epoch, 5-eval, batch-16 recipe differs",
    )
    _require(
        protocol.get("device") == "cuda:0"
        and protocol.get("expected_gpu_uuid") == FORMAL_GPU_UUIDS[dataset],
        f"V4 {stage} formal GPU binding differs",
    )
    from experiments import three_dataset_v2_protocol as data_protocol

    expected_data_root = (REPO_ROOT / "datasets").resolve(strict=True)
    expected_normalization = dict(data_protocol.get_legacy_normalization(dataset))
    _require(
        protocol.get("data_root") == str(expected_data_root)
        and protocol.get("normalization") == expected_normalization,
        f"V4 {stage} formal data-root/normalization identity differs",
    )
    _require(
        protocol.get("smoke") is False
        and protocol.get("smoke_limits")
        == {"max_train_samples": None, "max_val_samples": None}
        and protocol.get("official_test_accessed") is False
        and protocol.get("performance_acceptance_margin") is None,
        f"V4 {stage} run protocol scope/budget differs",
    )
    for binding_name, expected_semantic_sha in (
        ("source_lock", source_lock_sha256),
        ("split_projection", split_projection_sha256),
    ):
        binding = protocol.get(binding_name)
        _require(
            isinstance(binding, Mapping),
            f"V4 {stage} {binding_name} protocol binding differs",
        )
        bound_path = _regular_file(
            Path(str(binding.get("path"))),
            label=f"V4 {stage} {binding_name} bound file",
        )
        _require(
            binding.get("semantic_sha256") == expected_semantic_sha
            and binding.get("file_sha256") == pool_io.file_sha256(bound_path),
            f"V4 {stage} {binding_name} protocol bytes/semantic SHA differ",
        )
    _require(
        summary.get("run_protocol") == str(protocol_path.resolve(strict=True))
        and summary.get("run_protocol_sha256") == declared_protocol_sha,
        f"V4 {stage} summary/run-protocol binding differs",
    )
    artifact_sha = pool_io.file_sha256(checkpoint_path)
    _require(
        summary.get("selected_checkpoint_sha256") == artifact_sha,
        f"V4 {stage} selected checkpoint bytes differ",
    )

    payload = run_artifacts.load_torch_artifact(checkpoint_path)
    _require(
        payload.get("candidate_manifest_sha256") == _candidate_manifest_sha(payload),
        f"V4 {stage} candidate manifest SHA differs",
    )
    _require(
        payload.get("run_protocol_sha256") == declared_protocol_sha,
        f"V4 {stage} candidate/run-protocol binding differs",
    )
    _require(
        (payload.get("dataset"), payload.get("role"), payload.get("parent_role"), payload.get("stage"))
        == (dataset, role, role, stage),
        f"V4 {stage} checkpoint identity differs",
    )
    _require(
        payload.get("official_test_data_accessed") is False
        and payload.get("official_test_accessed") is False
        and payload.get("performance_acceptance_margin") is None
        and payload.get("smoke") is False,
        f"V4 {stage} checkpoint scope differs",
    )
    state = payload.get("state_dict")
    _require(
        isinstance(state, Mapping)
        and bool(state)
        and all(type(name) is str and isinstance(tensor, torch.Tensor) for name, tensor in state.items()),
        f"V4 {stage} tensor state differs",
    )
    semantic_sha = state_semantic_sha256(state)  # type: ignore[arg-type]
    _require(
        payload.get("state_sha256") == semantic_sha
        and payload.get("state_key_count") == len(state)
        and summary.get("selected_state_sha256") == semantic_sha,
        f"V4 {stage} real semantic state SHA differs",
    )
    _require(
        payload.get("source_sha256") == source_lock_sha256
        and payload.get("source_lock_sha256") == source_lock_sha256
        and summary.get("source_lock_sha256") == source_lock_sha256,
        f"V4 {stage} source-lock binding differs",
    )
    _require(
        payload.get("split_sha256") == split_projection_sha256
        and payload.get("split_projection_sha256") == split_projection_sha256
        and summary.get("split_projection_sha256") == split_projection_sha256,
        f"V4 {stage} split-projection binding differs",
    )
    atlas_sha = _sha256(payload.get("atlas_sha256"), name=f"V4 {stage} atlas SHA")
    _require(
        payload.get("atlas_manifest_sha256") == atlas_sha
        and summary.get("atlas_manifest_sha256") == atlas_sha,
        f"V4 {stage} atlas binding differs",
    )
    atlas_binding = protocol.get("atlas_manifest")
    _require(
        isinstance(atlas_binding, Mapping),
        f"V4 {stage} atlas protocol binding differs",
    )
    atlas_path = _regular_file(
        Path(str(atlas_binding.get("path"))),
        label=f"V4 {stage} atlas manifest",
    )
    _require(
        atlas_binding.get("semantic_sha256") == atlas_sha
        and atlas_binding.get("file_sha256") == pool_io.file_sha256(atlas_path),
        f"V4 {stage} atlas protocol bytes/semantic SHA differ",
    )
    _require(
        payload.get("parent_checkpoint_sha256") == current_record.get("sha256")
        and payload.get("parent_state_sha256") == current_record.get("state_sha256"),
        f"V4 {stage} Current parent binding differs",
    )
    parent = payload.get("parent_checkpoint")
    _require(isinstance(parent, Mapping), f"V4 {stage} parent checkpoint differs")
    _require(
        parent.get("path") == str(Path(str(current_record["path"])).resolve(strict=True))
        and parent.get("sha256") == current_record.get("sha256")
        and parent.get("state_sha256") == current_record.get("state_sha256"),
        f"V4 {stage} complete parent record differs",
    )

    run_identity_raw = payload.get("run_identity")
    _require(isinstance(run_identity_raw, Mapping), f"V4 {stage} run identity differs")
    try:
        run_identity = run_artifacts.RunIdentity(**dict(run_identity_raw))
    except (TypeError, ValueError, run_artifacts.PBDRV4ArtifactError) as error:
        raise PBDRV4ProtocolFreezeError(f"V4 {stage} run identity is invalid: {error}") from error
    _require(summary.get("run_identity") == run_identity.as_dict(), f"V4 {stage} summary identity differs")
    _require(protocol.get("run_identity") == run_identity.as_dict(), f"V4 {stage} protocol identity differs")
    _require(
        run_identity.dataset == dataset
        and run_identity.role == role
        and run_identity.stage == stage
        and run_identity.source_lock_sha256 == source_lock_sha256
        and run_identity.split_projection_sha256 == split_projection_sha256
        and run_identity.atlas_manifest_sha256 == atlas_sha
        and run_identity.parent_checkpoint_sha256 == current_record.get("sha256")
        and run_identity.parent_state_sha256 == current_record.get("state_sha256"),
        f"V4 {stage} run identity bindings differ",
    )
    if stage == "stage1":
        _require(
            expected_stage1_file_sha256 is None
            and run_identity.initialization_checkpoint_sha256 is None,
            "V4 Stage1 initialization binding differs",
        )
    else:
        _require(
            run_identity.initialization_checkpoint_sha256
            == _sha256(expected_stage1_file_sha256, name="Stage1 artifact SHA"),
            "V4 Stage2 is not initialized from the selected Stage1 bytes",
        )

    expected_recipe = training_core.training_recipe(stage)  # type: ignore[arg-type]
    _require(
        payload.get("training_recipe") == expected_recipe
        and summary.get("training_recipe") == expected_recipe,
        f"V4 {stage} training recipe differs",
    )
    _require(
        expected_recipe.get("epochs") == formal_epochs
        and expected_recipe.get("eval_every") == 5
        and expected_recipe.get("batch_size") == FORMAL_BATCH_SIZE,
        f"V4 {stage} formal training budget differs",
    )
    _require(
        summary.get("expected_gpu_uuid") == FORMAL_GPU_UUIDS[dataset]
        and summary.get("observed_gpu_uuid") == FORMAL_GPU_UUIDS[dataset],
        f"V4 {stage} completed GPU UUID binding differs",
    )
    _require(
        summary.get("normalization") == expected_normalization,
        f"V4 {stage} summary normalization differs",
    )
    expected_loss = component_loss.role_loss_manifest(role)  # type: ignore[arg-type]
    _require(payload.get("loss_manifest") == expected_loss, f"V4 {stage} loss manifest differs")
    epoch = payload.get("epoch")
    metrics = payload.get("validation_metrics")
    _require(
        type(epoch) is int and epoch > 0 and isinstance(metrics, Mapping),
        f"V4 {stage} selected epoch/metrics differ",
    )
    exact_key = training_core.checkpoint_epoch_key(role, metrics, epoch)
    serialized_key = [_fraction_json(item) for item in exact_key]
    _require(payload.get("selection_key") == serialized_key, f"V4 {stage} selection key differs")
    _require(
        summary.get("selected_epoch") == epoch
        and summary.get("selected_metrics") == metrics
        and summary.get("selected_diagnostics") == payload.get("validation_diagnostics"),
        f"V4 {stage} summary selection differs",
    )
    architecture = payload.get("architecture_manifest")
    _require(isinstance(architecture, Mapping), f"V4 {stage} architecture manifest differs")
    initialization_sha = _sha256(
        payload.get("initialization_sha256"),
        name=f"V4 {stage} initialization semantic SHA",
    )
    model, metadata = current_registry.build_candidate_inference_model(
        payload,
        dataset_name=dataset,
        role=role,
        stage=stage,
        expected_source_sha256=source_lock_sha256,
        expected_split_sha256=split_projection_sha256,
        expected_atlas_sha256=atlas_sha,
        expected_initialization_sha256=initialization_sha,
        expected_state_sha256=semantic_sha,
    )
    _require(
        metadata.get("strict_complete_payload") is True
        and metadata.get("candidate_state_sha256") == semantic_sha,
        f"V4 {stage} strict inference export differs",
    )
    del model
    return payload, summary, checkpoint_path


def _validate_sweep_and_cache(
    *,
    cache_path: Path,
    sweep_path: Path,
    projection: Mapping[str, Any],
    dataset: str,
    role: str,
    source_lock_sha256: str,
    split_projection_sha256: str,
) -> tuple[cache_io.ValidatedInternalRawLogitCache, dict[str, Any], Path]:
    cache = cache_io.read_cache(cache_path, split_projection=projection)
    identity = cache.manifest.get("identity")
    _require(isinstance(identity, Mapping), "internal cache identity differs")
    _require(
        identity.get("dataset") == dataset
        and identity.get("parent_role") == role
        and identity.get("partition") == "internal_validation",
        "internal cache dataset/role/partition differs",
    )
    _require(
        identity.get("source_lock_sha256") == source_lock_sha256
        and identity.get("split_projection_sha256") == split_projection_sha256
        and identity.get("official_test_accessed") is False,
        "internal cache source/split/scope binding differs",
    )

    sweep_result, raw = _read_json_object(sweep_path, label="V3 calibration sweep result")
    _require(
        raw == cache_io.canonical_json_bytes(sweep_result, trailing_newline=True),
        "V3 calibration sweep result is not canonical JSON",
    )
    validated = residual_sweep.validate_sweep_result(sweep_result, cache)
    _require(
        validated.get("dataset") == dataset
        and validated.get("role") == role
        and validated.get("selection_scope") == "internal_validation_only"
        and validated.get("official_test_accessed") is False,
        "V3 calibration sweep identity/scope differs",
    )
    source_binding = validated.get("source_binding")
    split_binding = validated.get("split_and_target_binding")
    _require(
        isinstance(source_binding, Mapping)
        and source_binding.get("source_lock_sha256") == source_lock_sha256,
        "V3 calibration sweep source-lock binding differs",
    )
    _require(
        isinstance(split_binding, Mapping)
        and split_binding.get("split_projection_sha256") == split_projection_sha256,
        "V3 calibration sweep split binding differs",
    )
    return cache, validated, _regular_file(sweep_path, label="V3 calibration sweep result")


def _collect_pool_evidence(
    *,
    dataset: str,
    role: str,
    source_lock_path: Path,
    split_projection_path: Path,
    internal_cache_path: Path,
    v3_sweep_path: Path,
    v3_calibrated_artifact_path: Path,
    stage1_checkpoint_path: Path,
    stage2_checkpoint_path: Path,
) -> PoolEvidence:
    _require(dataset in DATASETS and role in ROLES, "pool dataset/role differs")
    source_lock = source_lock_io.load_source_lock(source_lock_path, check_environment=True)
    source_sha = _sha256(source_lock.get("source_lock_sha256"), name="source-lock SHA")
    projection, _ = _validate_persisted_projection(split_projection_path)
    split_sha = _sha256(projection.get("projection_sha256"), name="split-projection SHA")

    cache, sweep, sweep_path = _validate_sweep_and_cache(
        cache_path=internal_cache_path,
        sweep_path=v3_sweep_path,
        projection=projection,
        dataset=dataset,
        role=role,
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
    )
    identity = cache.manifest["identity"]
    checkpoints = identity.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), "cache checkpoint bindings differ")

    current_payload, _, current_record = current_registry.load_current_checkpoint(dataset, role)
    current_model, current_metadata = current_registry.build_frozen_current_reference_model(
        dataset, role, "stage1"
    )
    _require(
        current_metadata.get("base_logits_are_current") is True,
        "Current graph is not a strict frozen reference",
    )
    del current_model
    _assert_checkpoint_binding(
        current_record,
        checkpoints.get("current"),
        label="Current",
    )

    original_payload, _, original_record = original_registry.load_original_checkpoint(dataset, role)
    original_model, original_metadata = original_registry.build_original_inference_model(dataset, role)
    _require(original_metadata.get("strict_load") is True, "Original graph is not strict-loaded")
    del original_model
    _assert_checkpoint_binding(
        original_record,
        checkpoints.get("original"),
        label="Original",
    )

    v3 = _validate_v3_run(dataset, role)
    v3_record = {
        "path": str(v3.run.candidate_path),
        "state_sha256": v3.state_sha256,
    }
    _assert_checkpoint_binding(
        v3_record,
        checkpoints.get("v3_candidate"),
        label="V3 Candidate",
    )

    stage1_payload, _, stage1_path = _validate_selected_v4_checkpoint(
        stage1_checkpoint_path,
        dataset=dataset,
        role=role,
        stage="stage1",
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        current_record=current_record,
        expected_stage1_file_sha256=None,
    )
    stage1_file_sha = pool_io.file_sha256(stage1_path)
    stage2_payload, _, stage2_path = _validate_selected_v4_checkpoint(
        stage2_checkpoint_path,
        dataset=dataset,
        role=role,
        stage="stage2",
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        current_record=current_record,
        expected_stage1_file_sha256=stage1_file_sha,
    )
    _require(
        stage1_payload.get("atlas_sha256") == stage2_payload.get("atlas_sha256")
        and stage1_payload.get("initialization_sha256")
        == stage2_payload.get("initialization_sha256"),
        "V4 Stage1/Stage2 architecture initialization or atlas differs",
    )

    selected = sweep.get("selected")
    grid_binding = sweep.get("grid_binding")
    _require(
        isinstance(selected, Mapping)
        and isinstance(selected.get("config"), Mapping)
        and isinstance(selected.get("name"), str)
        and isinstance(grid_binding, Mapping),
        "V3 selected calibration configuration differs",
    )
    v3_semantic_state_sha = state_semantic_sha256(v3.run.candidate_state)
    v3_configuration = {
        "state_sha256": v3_semantic_state_sha,
        "calibration": dict(selected["config"]),
        "selected_on": "internal_validation",
    }
    v3_configuration_sha = _configuration_sha256(
        family="V3-calibrated",
        dataset=dataset,
        role=role,
        configuration=v3_configuration,
    )
    calibrated_path, calibrated_state_sha = _build_v3_calibrated_artifact(
        path=v3_calibrated_artifact_path,
        dataset=dataset,
        role=role,
        v3=v3,
        sweep=sweep,
        sweep_path=sweep_path,
        cache=cache,
        configuration_sha256=v3_configuration_sha,
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
    )

    def v4_configuration(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "stage": payload.get("stage"),
            "source_sha256": payload.get("source_sha256"),
            "split_sha256": payload.get("split_sha256"),
            "atlas_sha256": payload.get("atlas_sha256"),
            "initialization_sha256": payload.get("initialization_sha256"),
            "state_sha256": payload.get("state_sha256"),
        }

    families = (
        FamilyEvidence(
            family="Original",
            name=f"{dataset}/{role}/Original",
            kind="original_checkpoint",
            artifact_path=Path(str(original_record["path"])),
            state_sha256=str(original_record["state_sha256"]),
            configuration=_baseline_configuration(
                state_sha256=str(original_record["state_sha256"]),
            ),
        ),
        FamilyEvidence(
            family="Current",
            name=f"{dataset}/{role}/Current",
            kind="current_checkpoint",
            artifact_path=Path(str(current_record["path"])),
            state_sha256=str(current_record["state_sha256"]),
            configuration=_baseline_configuration(
                state_sha256=str(current_record["state_sha256"]),
            ),
        ),
        FamilyEvidence(
            family="V3-calibrated",
            name=f"{dataset}/{role}/V3-calibrated/{selected['name']}",
            kind="v3_residual_calibration",
            artifact_path=calibrated_path,
            state_sha256=calibrated_state_sha,
            configuration=v3_configuration,
        ),
        FamilyEvidence(
            family="V4-Stage1",
            name=f"{dataset}/{role}/V4-Stage1",
            kind="v4_stage1_checkpoint",
            artifact_path=stage1_path,
            state_sha256=str(stage1_payload["state_sha256"]),
            configuration=v4_configuration(stage1_payload),
        ),
        FamilyEvidence(
            family="V4-Stage2",
            name=f"{dataset}/{role}/V4-Stage2",
            kind="v4_stage2_checkpoint",
            artifact_path=stage2_path,
            state_sha256=str(stage2_payload["state_sha256"]),
            configuration=v4_configuration(stage2_payload),
        ),
    )
    return PoolEvidence(
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        families=families,
    )


def freeze_pool(
    *,
    dataset: str,
    role: str,
    source_lock_path: Path,
    split_projection_path: Path,
    internal_cache_path: Path,
    v3_sweep_path: Path,
    v3_calibrated_artifact_path: Path,
    stage1_checkpoint_path: Path,
    stage2_checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """Strictly replay every family, then write one ordered candidate pool."""

    training_core.configure_determinism()
    evidence = _collect_pool_evidence(
        dataset=dataset,
        role=role,
        source_lock_path=source_lock_path,
        split_projection_path=split_projection_path,
        internal_cache_path=internal_cache_path,
        v3_sweep_path=v3_sweep_path,
        v3_calibrated_artifact_path=v3_calibrated_artifact_path,
        stage1_checkpoint_path=stage1_checkpoint_path,
        stage2_checkpoint_path=stage2_checkpoint_path,
    )
    candidates = tuple(item.candidate(dataset=dataset, role=role) for item in evidence.families)
    payload = pool_io.build_candidate_pool(
        dataset=dataset,
        role=role,
        source_lock_sha256=evidence.source_lock_sha256,
        split_projection_sha256=evidence.split_projection_sha256,
        candidates=candidates,
    )
    return pool_io.write_candidate_pool_exclusive(output_path, payload).resolve(
        strict=True
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("freeze-source", help="freeze all pretraining authorities")
    source.add_argument("--split-projection", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)

    pool = commands.add_parser("freeze-pool", help="freeze one post-training candidate pool")
    pool.add_argument("--dataset", choices=DATASETS, required=True)
    pool.add_argument("--role", choices=ROLES, required=True)
    pool.add_argument("--source-lock", type=Path, required=True)
    pool.add_argument("--split-projection", type=Path, required=True)
    pool.add_argument("--internal-cache", "--v3-cache", dest="internal_cache", type=Path, required=True)
    pool.add_argument("--v3-sweep", "--v3-sweep-result", dest="v3_sweep", type=Path, required=True)
    pool.add_argument("--v3-calibrated-artifact", type=Path, required=True)
    pool.add_argument("--stage1-checkpoint", type=Path, required=True)
    pool.add_argument("--stage2-checkpoint", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.command == "freeze-source":
        destination = freeze_source(
            split_projection_path=arguments.split_projection,
            output_path=arguments.output,
        )
    else:
        destination = freeze_pool(
            dataset=arguments.dataset,
            role=arguments.role,
            source_lock_path=arguments.source_lock,
            split_projection_path=arguments.split_projection,
            internal_cache_path=arguments.internal_cache,
            v3_sweep_path=arguments.v3_sweep,
            v3_calibrated_artifact_path=arguments.v3_calibrated_artifact,
            stage1_checkpoint_path=arguments.stage1_checkpoint,
            stage2_checkpoint_path=arguments.stage2_checkpoint,
            output_path=arguments.output,
        )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FamilyEvidence",
    "PBDRV4ProtocolFreezeError",
    "PoolEvidence",
    "V3Evidence",
    "freeze_pool",
    "freeze_source",
    "main",
    "parse_args",
]
