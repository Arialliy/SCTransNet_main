#!/usr/bin/env python3
"""Deterministic internal-only PBDR-V5 target-preservation fine-tuner.

The single V5 arm starts from an immutable selected V4-Stage1 checkpoint and
uses the already-audited V4 Stage2 mutable set.  It never imports an official
test dataset, never opens an official test index, and selects epoch zero or a
trained epoch solely on the frozen internal-validation split at probability
``> 0.5``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import pbdr_v4_models_seed42_v1 as models
from experiments import train_three_dataset_pbdr_v4_v1 as v4_train
from experiments.pbdr_v4_atlas_dataset import PBDRV4AtlasTrainDataset
from experiments.pbdr_v4_internal_dataset import PBDRV4InternalInferenceDataset
from experiments.pbdr_v4_run_artifacts import (
    atomic_rolling_torch_save,
    exclusive_json,
    exclusive_torch_save,
    file_sha256,
    load_torch_artifact,
    optimizer_group_signature,
)
from experiments.pbdr_v4_source_lock import load_source_lock
from experiments import pbdr_v4_split_authority as split_authority
from experiments.pbdr_v4_state_contract import (
    audit_candidate_against_current,
    audit_training_modes,
    configure_stage_training,
    l2sp_to_current,
    mutable_parameter_names,
    state_semantic_sha256,
)
from experiments.pbdr_v4_training_core import (
    STAGE2_L2SP_WEIGHT,
    TRAINING_SEED,
    build_optimizer,
    capture_rng_state,
    configure_determinism,
    restore_rng_state,
)
from experiments.pbdr_v5_run_contract import (
    ARM,
    FORMAL_BATCH_SIZE,
    FORMAL_EPOCHS,
    FORMAL_EVAL_EVERY,
    V5RunIdentity,
    build_rolling_payload,
    canonical_json_sha256,
    epoch_selection_key,
    json_selection_key,
    ordered_strings_sha256,
    require_sha256,
    validate_rolling_payload,
)
from experiments.pbdr_v5_target_preservation_loss import (
    compute_pbdr_v5_target_preservation_loss,
    target_preservation_loss_manifest,
)


SCHEMA = "sctransnet_three_dataset_pbdr_v5_training_v1/v1"
CANDIDATE_SCHEMA = "sctransnet_pbdr_v5_target_preservation_candidate/v1"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/pbdr_v5_v1"
ROLLING_NAME = "rolling_state.pth.tar"
SELECTED_NAME = "selected_candidate.pth.tar"
SUMMARY_NAME = "summary.json"
RUN_PROTOCOL_NAME = "run_protocol.json"
FORMAL_WORKERS = 0
FOCUS_RUNS = frozenset(
    {
        ("NUDT-SIRST", "best_pd"),
        ("NUAA-SIRST", "best_miou"),
        ("IRSTD-1K", "best_miou"),
    }
)
V5_SOURCE_PATHS = (
    REPO_ROOT / "experiments/PBDR_V5_INTERNAL_PROTOCOL.md",
    REPO_ROOT / "experiments/pbdr_v5_target_preservation_loss.py",
    REPO_ROOT / "experiments/pbdr_v5_run_contract.py",
    REPO_ROOT / "experiments/pbdr_v5_internal_selector.py",
    REPO_ROOT / "experiments/train_three_dataset_pbdr_v5_v1.py",
    REPO_ROOT
    / "results/pbdr_v5_v1/diagnostics/failure_localization_bundle.json",
)


class PBDRV5TrainerError(RuntimeError):
    """A V5 input, training state, or artifact violates the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV5TrainerError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV5TrainerError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one object")
    return value


def _commit_or_validate_json(path: Path, expected: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if not destination.exists() and not destination.is_symlink():
        return exclusive_json(destination, expected).resolve(strict=True)
    _require(
        destination.is_file() and not destination.is_symlink(),
        f"existing JSON path is unsafe: {destination}",
    )
    observed = _read_json(destination, label=destination.name)
    _require(observed == dict(expected), f"existing {destination.name} differs")
    return destination.resolve(strict=True)


def _v5_source_manifest() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in V5_SOURCE_PATHS:
        _require(path.is_file() and not path.is_symlink(), f"V5 source is missing: {path}")
        relative = str(path.relative_to(REPO_ROOT))
        files[relative] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest: dict[str, Any] = {
        "schema": f"{SCHEMA}/source_manifest",
        "files": files,
    }
    manifest["semantic_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _validate_failure_localization_bundle() -> dict[str, Any]:
    path = REPO_ROOT / "results/pbdr_v5_v1/diagnostics/failure_localization_bundle.json"
    payload = _read_json(path, label="V5 failure-localization bundle")
    _require(
        payload.get("schema") == "sctransnet_pbdr_v5_failure_localization/v1",
        "failure-localization schema differs",
    )
    _require(payload.get("official_test_accessed") is False, "diagnosis official flag differs")
    _require(
        payload.get("official_test_index_accessed") is False
        and payload.get("official_test_loader_constructed") is False,
        "diagnosis test-access contract differs",
    )
    _require(
        payload.get("diagnosis_completed_before_v5_code") is True,
        "diagnosis order was not frozen",
    )
    declared = require_sha256(payload.get("bundle_sha256"), name="diagnosis bundle SHA")
    unsigned = dict(payload)
    del unsigned["bundle_sha256"]
    _require(canonical_json_sha256(unsigned) == declared, "diagnosis bundle SHA does not replay")
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha256(path),
        "semantic_sha256": declared,
    }


def _training_recipe(*, epochs: int, eval_every: int, batch_size: int) -> dict[str, Any]:
    return {
        "arm": ARM,
        "initialization": "immutable_selected_v4_stage1",
        "epochs": epochs,
        "eval_every": eval_every,
        "batch_size": batch_size,
        "workers": FORMAL_WORKERS,
        "seed": TRAINING_SEED,
        "precision": "fp32",
        "optimizer": "AdamW",
        "parameter_groups": [
            {"name": "pbdr_v4", "lr": 1.0e-4},
            {"name": "outc", "lr": 2.0e-6},
            {"name": "up_decoder1", "lr": 1.0e-6},
        ],
        "weight_decay": 1.0e-4,
        "l2sp_weight": STAGE2_L2SP_WEIGHT,
        "fixed_probability_comparison": ">",
        "fixed_probability_threshold": 0.5,
        "performance_acceptance_margin": None,
    }


def _initial_model(
    *,
    dataset: str,
    role: str,
    stage1_checkpoint: Path,
    source_sha256: str,
    split_sha256: str,
    atlas_sha256: str,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], str, str]:
    checkpoint_path = Path(stage1_checkpoint)
    _require(
        checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
        "V4-Stage1 checkpoint is missing or unsafe",
    )
    payload = load_torch_artifact(checkpoint_path)
    _require(payload.get("stage") == "stage1", "initial checkpoint is not V4-Stage1")
    _require(payload.get("smoke") is False, "formal V5 cannot initialize from a smoke checkpoint")
    _require(payload.get("official_test_accessed") is False, "Stage1 official flag differs")
    _require(payload.get("performance_acceptance_margin") is None, "Stage1 margin differs")
    initialization_sha = require_sha256(
        payload.get("initialization_sha256"), name="V4 initialization SHA"
    )
    stage1_state_sha = require_sha256(
        payload.get("state_sha256"), name="V4-Stage1 state SHA"
    )
    model, metadata = models.build_stage2_training_model(
        payload,
        dataset_name=dataset,
        role=role,
        stage="stage2",
        expected_source_sha256=source_sha256,
        expected_split_sha256=split_sha256,
        expected_atlas_sha256=atlas_sha256,
        expected_initialization_sha256=initialization_sha,
        expected_stage1_state_sha256=stage1_state_sha,
    )
    _require(
        state_semantic_sha256(model.state_dict()) == stage1_state_sha,
        "V5 initialization differs from selected V4-Stage1 state",
    )
    return (
        model,
        metadata,
        payload,
        file_sha256(checkpoint_path),
        stage1_state_sha,
    )


def _selected_state(
    *,
    model: torch.nn.Module,
    epoch: int,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    key = epoch_selection_key(role, metrics, epoch)
    return {
        "epoch": epoch,
        "metrics": dict(metrics),
        "diagnostics": dict(diagnostics),
        "selection_key": json_selection_key(key),
        "selection_key_raw": tuple(key),
        "state_dict": state,
        "state_sha256": state_semantic_sha256(state),
    }


def train_one_epoch(
    model: torch.nn.Module,
    current_reference: torch.nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    role: str,
    device: torch.device,
    current_state: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Train the one frozen V5 arm for one deterministic epoch."""

    configure_stage_training(model, "stage2")
    current_reference.eval()
    diagnostics = v4_train.TrainingDiagnostics()
    for batch in loader:
        _require(isinstance(batch, (tuple, list)) and len(batch) == 6, "training batch differs")
        image, target, rescue, suppress, preserve, _ = batch
        image = image.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        rescue = rescue.to(device=device)
        suppress = suppress.to(device=device)
        preserve = preserve.to(device=device)
        optimizer.zero_grad(set_to_none=True)
        _, auxiliary = model.forward_for_pbdr_v4_training(image)
        with torch.no_grad():
            _, reference_auxiliary = current_reference.forward_for_pbdr_v4_training(image)
            current_logits = reference_auxiliary.candidate_base_logits.detach()
        loss_output = compute_pbdr_v5_target_preservation_loss(
            role=role,  # type: ignore[arg-type]
            routed_logits=auxiliary.routed_logits,
            candidate_base_logits=auxiliary.candidate_base_logits,
            reference_current_logits=current_logits,
            delta_logits=auxiliary.delta_logits,
            target=target,
            rescue_component_ids=rescue,
            suppress_component_ids=suppress,
            preserve_component_ids=preserve,
        )
        l2sp = l2sp_to_current(model, current_state=current_state)
        total = loss_output.total + STAGE2_L2SP_WEIGHT * l2sp
        _require(bool(torch.isfinite(total)), "training loss is non-finite")
        total.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                _require(parameter.grad is not None, f"trainable parameter lacks gradient: {name}")
                _require(bool(torch.isfinite(parameter.grad).all()), f"gradient is non-finite: {name}")
            else:
                _require(parameter.grad is None, f"frozen parameter received gradient: {name}")
        optimizer.step()
        diagnostics.update(
            total=total,
            l2sp=l2sp,
            loss_components=loss_output.detached_scalars(),
            base=auxiliary.candidate_base_logits,
            routed=auxiliary.routed_logits,
            delta=auxiliary.delta_logits,
            component_maps={"rescue": rescue, "suppress": suppress, "preserve": preserve},
        )
    audit_training_modes(model, "stage2")
    audit_candidate_against_current(model, current_state=current_state, stage="stage2")
    return diagnostics.compute()


def _candidate_manifest(candidate: Mapping[str, Any]) -> dict[str, Any]:
    inner = candidate.get("v4_compatible_candidate")
    _require(isinstance(inner, Mapping), "candidate lacks its V4-compatible payload")
    return {
        "schema": candidate.get("schema"),
        "dataset": candidate.get("dataset"),
        "role": candidate.get("role"),
        "arm": candidate.get("arm"),
        "epoch": candidate.get("epoch"),
        "state_sha256": inner.get("state_sha256"),
        "run_identity": candidate.get("run_identity"),
        "run_identity_sha256": candidate.get("run_identity_sha256"),
        "run_protocol_sha256": candidate.get("run_protocol_sha256"),
        "v5_source_sha256": candidate.get("v5_source_sha256"),
        "loss_manifest_sha256": candidate.get("loss_manifest_sha256"),
        "stage1_checkpoint_sha256": candidate.get("stage1_checkpoint_sha256"),
        "stage1_state_sha256": candidate.get("stage1_state_sha256"),
        "validation_metrics": candidate.get("validation_metrics"),
        "selection_key": candidate.get("selection_key"),
        "official_test_accessed": candidate.get("official_test_accessed"),
        "performance_acceptance_margin": candidate.get("performance_acceptance_margin"),
    }


def _commit_or_validate_candidate(path: Path, expected: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if not destination.exists() and not destination.is_symlink():
        return exclusive_torch_save(destination, expected).resolve(strict=True)
    _require(
        destination.is_file() and not destination.is_symlink(),
        "existing selected candidate path is unsafe",
    )
    observed = load_torch_artifact(destination)
    _require(
        observed.get("candidate_manifest_sha256")
        == expected.get("candidate_manifest_sha256"),
        "existing selected candidate manifest differs",
    )
    _require(
        canonical_json_sha256(_candidate_manifest(observed))
        == observed.get("candidate_manifest_sha256"),
        "existing selected candidate manifest does not replay",
    )
    observed_inner = observed.get("v4_compatible_candidate")
    expected_inner = expected.get("v4_compatible_candidate")
    _require(
        isinstance(observed_inner, Mapping) and isinstance(expected_inner, Mapping),
        "candidate inner payload differs",
    )
    _require(
        observed_inner.get("state_sha256") == expected_inner.get("state_sha256"),
        "candidate state SHA differs",
    )
    observed_state = observed_inner.get("state_dict")
    expected_state = expected_inner.get("state_dict")
    _require(
        isinstance(observed_state, Mapping)
        and isinstance(expected_state, Mapping)
        and set(observed_state) == set(expected_state),
        "candidate state keys differ",
    )
    for name in expected_state:
        _require(
            isinstance(observed_state[name], torch.Tensor)
            and isinstance(expected_state[name], torch.Tensor)
            and torch.equal(observed_state[name], expected_state[name]),
            f"candidate tensor differs: {name}",
        )
    return destination.resolve(strict=True)


def _completed_summary(
    run_dir: Path,
    *,
    identity: V5RunIdentity,
    protocol_sha256: str,
) -> Path | None:
    path = run_dir / SUMMARY_NAME
    if not path.exists() and not path.is_symlink():
        return None
    summary = _read_json(path, label="completed V5 summary")
    _require(summary.get("schema") == SCHEMA and summary.get("status") == "complete", "summary status differs")
    _require(summary.get("run_identity") == identity.as_dict(), "summary identity differs")
    _require(summary.get("run_protocol_sha256") == protocol_sha256, "summary protocol differs")
    declared = require_sha256(summary.get("summary_sha256"), name="summary SHA")
    unsigned = dict(summary)
    del unsigned["summary_sha256"]
    _require(canonical_json_sha256(unsigned) == declared, "summary SHA does not replay")
    selected = run_dir / SELECTED_NAME
    _require(selected.is_file() and not selected.is_symlink(), "selected V5 candidate is missing")
    _require(summary.get("selected_checkpoint_sha256") == file_sha256(selected), "selected bytes differ")
    return path.resolve(strict=True)


def run(args: argparse.Namespace) -> Path:
    _require((args.dataset, args.role) in FOCUS_RUNS, "dataset/role is outside the frozen V5 scope")
    configure_determinism()
    diagnosis_binding = _validate_failure_localization_bundle()
    source_lock = load_source_lock(args.source_lock, check_environment=True)
    source_sha = require_sha256(source_lock.get("source_lock_sha256"), name="V4 source lock SHA")
    projection = v4_train.load_live_split_projection(args.split_projection)
    split_sha = require_sha256(projection.get("projection_sha256"), name="split projection SHA")
    official_ids, development_ids, validation_ids = v4_train.load_official_train_source_ids(
        projection, args.dataset
    )
    _, current_state, parent = models.load_current_checkpoint(args.dataset, args.role)
    parent_checkpoint_sha = require_sha256(parent.get("sha256"), name="Current checkpoint SHA")
    parent_state_sha = require_sha256(parent.get("state_sha256"), name="Current state SHA")
    atlas_manifest_path, atlas_manifest, atlas_sha = v4_train._atlas_bindings(
        args.atlas_root,
        dataset=args.dataset,
        role=args.role,
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        parent_checkpoint_sha256=parent_checkpoint_sha,
        parent_state_sha256=parent_state_sha,
    )
    model, model_metadata, stage1_payload, stage1_file_sha, stage1_state_sha = _initial_model(
        dataset=args.dataset,
        role=args.role,
        stage1_checkpoint=args.stage1_checkpoint,
        source_sha256=source_sha,
        split_sha256=split_sha,
        atlas_sha256=atlas_sha,
    )
    trainable_names = tuple(sorted(mutable_parameter_names(model, "stage2")))
    configure_stage_training(model, "stage2")
    loss_manifest = target_preservation_loss_manifest(args.role)  # type: ignore[arg-type]
    loss_manifest_sha = canonical_json_sha256(loss_manifest)
    source_manifest = _v5_source_manifest()
    v5_source_sha = require_sha256(source_manifest["semantic_sha256"], name="V5 source SHA")
    identity = V5RunIdentity(
        dataset=args.dataset,
        role=args.role,
        arm=ARM,
        v4_source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        atlas_manifest_sha256=atlas_sha,
        parent_checkpoint_sha256=parent_checkpoint_sha,
        parent_state_sha256=parent_state_sha,
        stage1_checkpoint_sha256=stage1_file_sha,
        stage1_state_sha256=stage1_state_sha,
        v5_source_sha256=v5_source_sha,
        loss_manifest_sha256=loss_manifest_sha,
        trainable_parameter_names_sha256=ordered_strings_sha256(trainable_names),
    )

    formal_epochs, formal_eval, formal_batch = (
        FORMAL_EPOCHS,
        FORMAL_EVAL_EVERY,
        FORMAL_BATCH_SIZE,
    )
    epochs = args.epochs if args.smoke else formal_epochs
    eval_every = args.eval_every if args.smoke else formal_eval
    batch_size = args.batch_size if args.smoke else formal_batch
    _require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (epochs, eval_every, batch_size)), "training budget differs")
    if not args.smoke:
        _require((epochs, eval_every, batch_size) == (formal_epochs, formal_eval, formal_batch), "formal recipe differs")
        _require(args.max_train_samples is None and args.max_val_samples is None, "formal run cannot limit samples")

    data_root = Path(args.data_root)
    _require(data_root.is_dir() and not data_root.is_symlink(), "data root is missing or unsafe")
    if not args.smoke:
        _require(data_root.resolve(strict=True) == DEFAULT_DATA_ROOT.resolve(strict=True), "formal data root differs")
    device = torch.device(args.device)
    _require(device.type in ("cpu", "cuda"), "unsupported device")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    if not args.smoke and device.type == "cuda":
        _require(
            args.expected_gpu_uuid == v4_train.FORMAL_GPU_UUIDS[args.dataset],
            "formal dataset/GPU UUID binding differs",
        )
    observed_gpu_uuid = v4_train.validate_cuda_uuid(device, args.expected_gpu_uuid)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _require(not run_dir.is_symlink(), "run directory cannot be a symlink")
    protocol: dict[str, Any] = {
        "schema": f"{SCHEMA}/run_protocol",
        "run_identity": identity.as_dict(),
        "run_identity_sha256": identity.semantic_sha256,
        "training_recipe": _training_recipe(epochs=epochs, eval_every=eval_every, batch_size=batch_size),
        "device": args.device,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "observed_gpu_uuid": observed_gpu_uuid,
        "data_root": str(data_root.resolve(strict=True)),
        "normalization": v4_train._normalization(args.dataset),
        "source_lock": {
            "path": str(Path(args.source_lock).resolve(strict=True)),
            "file_sha256": file_sha256(Path(args.source_lock)),
            "semantic_sha256": source_sha,
        },
        "split_projection": {
            "path": str(Path(args.split_projection).resolve(strict=True)),
            "file_sha256": file_sha256(Path(args.split_projection)),
            "semantic_sha256": split_sha,
        },
        "atlas_manifest": {
            "path": str(atlas_manifest_path),
            "file_sha256": file_sha256(atlas_manifest_path),
            "semantic_sha256": atlas_sha,
        },
        "stage1_checkpoint": {
            "path": str(Path(args.stage1_checkpoint).resolve(strict=True)),
            "file_sha256": stage1_file_sha,
            "state_sha256": stage1_state_sha,
            "selected_epoch": stage1_payload.get("epoch"),
        },
        "failure_localization": diagnosis_binding,
        "v5_source_manifest": source_manifest,
        "loss_manifest": loss_manifest,
        "loss_manifest_sha256": loss_manifest_sha,
        "trainable_parameter_names": list(trainable_names),
        "smoke": bool(args.smoke),
        "smoke_limits": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
    }
    protocol["protocol_sha256"] = canonical_json_sha256(protocol)
    protocol_path = _commit_or_validate_json(run_dir / RUN_PROTOCOL_NAME, protocol)
    protocol_sha = require_sha256(protocol["protocol_sha256"], name="run protocol SHA")
    completed = _completed_summary(run_dir, identity=identity, protocol_sha256=protocol_sha)
    if completed is not None:
        _require(args.resume != "never", "completed run exists but resume=never")
        return completed

    train_dataset = PBDRV4AtlasTrainDataset(
        development_ids,
        official_ids,
        dataset_name=args.dataset,
        data_root=data_root,
        atlas_root=args.atlas_root,
        atlas_manifest=atlas_manifest_path,
        parent_state_sha256=parent_state_sha,
        normalization=v4_train._normalization(args.dataset),
    )
    validation_dataset: Any = PBDRV4InternalInferenceDataset(
        validation_ids,
        official_ids,
        manifest_scope="internal_validation_ids",
        selected_ids_ordered_sha256=split_authority.ordered_ids_sha256(validation_ids),
        official_train_count=len(official_ids),
        official_train_ordered_ids_sha256=split_authority.ordered_ids_sha256(official_ids),
        dataset_name=args.dataset,
        data_root=data_root,
    )
    if args.max_val_samples is not None:
        _require(args.smoke and 0 < args.max_val_samples <= len(validation_dataset), "max validation samples differs")
        validation_dataset = Subset(validation_dataset, range(args.max_val_samples))
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.to(device)
    current_reference, current_reference_metadata = models.build_frozen_current_reference_model(
        args.dataset, args.role, "stage2"
    )
    current_reference.to(device)
    current_reference.eval()
    optimizer = build_optimizer(model, "stage2")
    optimizer_signature = optimizer_group_signature(optimizer.state_dict())

    rolling_path = run_dir / ROLLING_NAME
    evaluation_history: list[dict[str, Any]]
    best: dict[str, Any]
    start_epoch: int
    if rolling_path.exists() or rolling_path.is_symlink():
        _require(args.resume != "never", "rolling checkpoint exists but resume=never")
        rolling = validate_rolling_payload(
            load_torch_artifact(rolling_path),
            identity=identity,
            epochs=epochs,
            expected_optimizer_group_signature=optimizer_signature,
        )
        model.load_state_dict(rolling["state_dict"], strict=True)
        optimizer.load_state_dict(rolling["optimizer"])
        restore_rng_state(rolling["rng_state"])
        best = dict(rolling["selected"])
        evaluation_history = [dict(item) for item in rolling["evaluation_history"]]
        start_epoch = int(rolling["epoch"]) + 1
        audit_candidate_against_current(model, current_state=current_state, stage="stage2")
        configure_stage_training(model, "stage2")
    else:
        _require(args.resume != "required", "resume=required but rolling checkpoint is absent")
        initial_metrics, initial_diagnostics = v4_train.validate_candidate(
            model, validation_loader, device=device
        )
        best = _selected_state(
            model=model,
            epoch=0,
            metrics=initial_metrics,
            diagnostics=initial_diagnostics,
            role=args.role,
        )
        evaluation_history = [
            {
                "epoch": 0,
                "metrics": initial_metrics,
                "selection_key": best["selection_key"],
            }
        ]
        configure_stage_training(model, "stage2")
        start_epoch = 1

    for epoch in range(start_epoch, epochs + 1):
        train_loader = v4_train._make_train_loader(
            train_dataset,
            epoch=epoch,
            batch_size=batch_size,
            max_samples=args.max_train_samples,
        )
        training_diagnostics = train_one_epoch(
            model,
            current_reference,
            train_loader,
            optimizer,
            role=args.role,
            device=device,
            current_state=current_state,
        )
        evaluation: dict[str, Any] | None = None
        if epoch % eval_every == 0 or epoch == epochs:
            metrics, validation_diagnostics = v4_train.validate_candidate(
                model, validation_loader, device=device
            )
            key = epoch_selection_key(args.role, metrics, epoch)
            evaluation = {
                "epoch": epoch,
                "metrics": metrics,
                "selection_key": json_selection_key(key),
            }
            evaluation_history.append(evaluation)
            if key > tuple(best["selection_key_raw"]):
                best = _selected_state(
                    model=model,
                    epoch=epoch,
                    metrics=metrics,
                    diagnostics=validation_diagnostics,
                    role=args.role,
                )
            configure_stage_training(model, "stage2")
        event = {
            "epoch": epoch,
            "training": training_diagnostics,
            "evaluation": evaluation,
            "selected_epoch": best["epoch"],
        }
        rolling = build_rolling_payload(
            identity=identity,
            epoch=epoch,
            epochs=epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            rng_state=capture_rng_state(),
            selected=best,
            evaluation_history=evaluation_history,
            event=event,
        )
        atomic_rolling_torch_save(rolling_path, rolling)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "dataset": args.dataset,
                    "role": args.role,
                    "arm": ARM,
                    "epoch": epoch,
                    "epochs": epochs,
                    "mean_total_loss": training_diagnostics["mean_total_loss"],
                    "evaluated": evaluation is not None,
                    "selected_epoch": best["epoch"],
                    "official_test_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    model.load_state_dict(best["state_dict"], strict=True)
    configure_stage_training(model, "stage2")
    v4_candidate = models.build_candidate_checkpoint_payload(
        model,
        dataset_name=args.dataset,
        role=args.role,
        stage="stage2",
        source_sha256=source_sha,
        split_sha256=split_sha,
        atlas_sha256=atlas_sha,
        initialization_sha256=require_sha256(
            model_metadata.get("initialization_sha256"),
            name="model initialization SHA",
        ),
    )
    candidate: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "dataset": args.dataset,
        "role": args.role,
        "arm": ARM,
        "epoch": best["epoch"],
        "validation_metrics": best["metrics"],
        "validation_diagnostics": best["diagnostics"],
        "selection_key": best["selection_key"],
        "run_identity": identity.as_dict(),
        "run_identity_sha256": identity.semantic_sha256,
        "run_protocol_sha256": protocol_sha,
        "v5_source_sha256": v5_source_sha,
        "loss_manifest": loss_manifest,
        "loss_manifest_sha256": loss_manifest_sha,
        "stage1_checkpoint_sha256": stage1_file_sha,
        "stage1_state_sha256": stage1_state_sha,
        "v4_compatible_candidate": v4_candidate,
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
        "smoke": bool(args.smoke),
    }
    candidate["candidate_manifest_sha256"] = canonical_json_sha256(
        _candidate_manifest(candidate)
    )
    selected_path = _commit_or_validate_candidate(run_dir / SELECTED_NAME, candidate)

    initial_metrics = evaluation_history[0]["metrics"]
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": args.dataset,
        "role": args.role,
        "arm": ARM,
        "run_identity": identity.as_dict(),
        "run_identity_sha256": identity.semantic_sha256,
        "run_protocol": str(protocol_path),
        "run_protocol_sha256": protocol_sha,
        "initial_epoch": 0,
        "initial_metrics": initial_metrics,
        "selected_epoch": best["epoch"],
        "selected_metrics": best["metrics"],
        "selected_diagnostics": best["diagnostics"],
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": file_sha256(selected_path),
        "selected_state_sha256": v4_candidate["state_sha256"],
        "evaluation_history": evaluation_history,
        "source_lock_sha256": source_sha,
        "split_projection_sha256": split_sha,
        "atlas_manifest_sha256": atlas_sha,
        "v5_source_manifest": source_manifest,
        "loss_manifest": loss_manifest,
        "model_metadata": model_metadata,
        "current_reference_metadata": current_reference_metadata,
        "training_recipe": _training_recipe(epochs=epochs, eval_every=eval_every, batch_size=batch_size),
        "normalization": v4_train._normalization(args.dataset),
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "observed_gpu_uuid": observed_gpu_uuid,
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
        "smoke": bool(args.smoke),
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    return exclusive_json(run_dir / SUMMARY_NAME, summary).resolve(strict=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=models.DATASETS, required=True)
    parser.add_argument("--role", choices=models.ROLES, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--split-projection", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "ARM",
    "CANDIDATE_SCHEMA",
    "FOCUS_RUNS",
    "PBDRV5TrainerError",
    "SCHEMA",
    "parse_args",
    "run",
    "train_one_epoch",
]
