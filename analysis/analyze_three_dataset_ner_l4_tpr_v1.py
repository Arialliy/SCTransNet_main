#!/usr/bin/env python3
"""Zero-training three-dataset screening for NER-L4-TPR V1.

The analyzer consumes the six current-Final checkpoints frozen by the
identity-input manifest.  Per test batch it runs the CNN encoder, TPD8, QFG2,
L4 upsample, existing NER q4, and detached protection construction exactly
once.  Six modes then reuse those forward-local tensors and rerun only the L4
CCA and downstream decoder:

* current ``g=0``;
* protected ``g`` in ``{0.0625, 0.125, 0.1875, 0.25}``;
* the unprotected historical ``g=+0.25`` L4-only counterfactual.

All reported task metrics use the fixed probability threshold 0.5.  The
threshold-1.0 point is evaluated only to retain the repository evaluator's
closed-interval descriptive contract; it is not used to select a mode.
No checkpoint, probability cache, or feature cache is written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_ner_stage2_mask_knockout_v1 as state_audit  # noqa: E402
from analysis import analyze_three_dataset_gcsf_branch_audit_v1 as shared  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as metric_runner  # noqa: E402
from experiments import ner_l4_tpr_strict_migration_v1 as migration  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)
from model import tpd_ner_l4_target_protected_reallocation as tpr_core  # noqa: E402
from model import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr as tpr_model,
)


SCHEMA = "sctransnet_three_dataset_ner_l4_tpr_zero_training_v1/v1"
REFERENCE_METHOD = "final_tss_off"
SEED = 42
FIXED_THRESHOLD = 0.5
SWEEP_THRESHOLDS = (0.5, 1.0)
EVALUATION_PROTOCOL = "img_idx_test_selected_development"
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLES = tuple(metric_runner.CHECKPOINT_ROLES)
CURRENT_MODE = "current_g0"
UNPROTECTED_MODE = "gpos025_l4_only"
TPR_GATE_LIMIT = tpr_core.FORMAL_L4_GATE_LIMIT

MODE_GATES: dict[str, float] = {
    CURRENT_MODE: 0.0,
    "tpr_g00625": 0.0625,
    "tpr_g0125": 0.125,
    "tpr_g01875": 0.1875,
    "tpr_g025": 0.25,
    UNPROTECTED_MODE: 0.25,
}
PUBLIC_MODES = tuple(MODE_GATES)
TPR_MODES = tuple(mode for mode in PUBLIC_MODES if mode.startswith("tpr_"))

DEFAULT_IDENTITY_MANIFEST = (
    REPO_ROOT
    / "results/ner_l4_tpr_identity_audit_v1/manifests/identity_inputs.json"
)
FROZEN_IDENTITY_MANIFEST_SHA256 = (
    "3c0825f7a45984ecedd85edd32d207080244a943c77ff37d4f3a7f67a6897712"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results/three_dataset_ner_l4_tpr_zero_training_v1"
)

probability_difference = shared.probability_difference
atomic_create_json = shared.atomic_create_json
file_sha256 = migration.file_sha256

# Replaying an old checkpoint on the same RTX 5090/runtime can move a single
# probability sitting at exactly the 0.5 decision boundary.  The first V1
# execution established the bounded runtime envelope before any TPR result was
# accepted: matched total/tiny targets remained exact in all inspected roles,
# at most one predicted/false object or pixel changed, soft overlap differed by
# at most 3.25e-4, and BCE by at most 3.2e-8.  These constants conservatively
# round that measured envelope upward.  They validate numerical replay only;
# every candidate is still compared against current_g0 produced in the same
# forward pass, so they cannot make a TPR mode look better.
REPLAY_POLICY = "same_runtime_bounded_numeric_replay_v1"
REPLAY_EXACT_COUNT_KEYS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "valid_pixel_count",
)
REPLAY_ONE_COUNT_KEYS = (
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "unmatched_predicted_pixels",
)
REPLAY_ONE_COUNT_ABSOLUTE_TOLERANCE = 1
REPLAY_SOFT_METRIC_ABSOLUTE_TOLERANCE = 5e-4
REPLAY_FA_ABSOLUTE_TOLERANCE = 5e-8
REPLAY_TEST_LOSS_ABSOLUTE_TOLERANCE = 1e-7
REPLAY_FALSE_OBJECTS_PER_IMAGE_ABSOLUTE_TOLERANCE = 1e-2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_public_mode(public_mode: str) -> dict[str, Any]:
    _require(public_mode in MODE_GATES, f"unknown L4-TPR mode: {public_mode!r}")
    gate = MODE_GATES[public_mode]
    protected = public_mode in TPR_MODES
    unprotected_reference = public_mode == UNPROTECTED_MODE
    ratio = gate / TPR_GATE_LIMIT
    finite_logit = protected and abs(ratio) < 1.0
    required_logit = math.atanh(ratio) if finite_logit else None
    return {
        "public_mode": public_mode,
        "gate_value": gate,
        "level": 4,
        "protection_applied": protected,
        "unprotected_reference": unprotected_reference,
        "current_reference": public_mode == CURRENT_MODE,
        "protection_source": (
            "existing_q4_tail_z1.5_binary_dilated_3x3_detached"
            if protected
            else "none"
        ),
        "finite_l4_tpr_logit_representable": finite_logit,
        "finite_logit_representable": finite_logit,
        "required_reallocation_logit": required_logit,
        "required_logit": required_logit,
        "boundary_limit_counterfactual": (
            public_mode == "tpr_g025"
        ),
        "diagnostic_only": public_mode != CURRENT_MODE,
    }


@dataclass(frozen=True, slots=True)
class PreparedL4TPRBatch:
    """Forward-local state retained until all six decoder modes finish."""

    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor
    transformed4: torch.Tensor
    encoder4: torch.Tensor
    baseline4: torch.Tensor
    d5: torch.Tensor
    evidence1: tuple[torch.Tensor, ...]
    evidence2: tuple[torch.Tensor, ...]
    up4: torch.Tensor
    q4: torch.Tensor
    mask4: torch.Tensor
    protection4: torch.Tensor


def prepare_l4_tpr_batch(
    model: nn.Module,
    images: torch.Tensor,
) -> PreparedL4TPRBatch:
    """Prepare encoder/QFG/up4/q4/protection exactly once for one batch."""

    _require(not model.training, "NER-L4-TPR screening requires model.eval()")
    prepared = shared.prepare_forward_local_branches(model, images)
    baseline = shared.production_order_baseline(
        prepared.transformed,
        prepared.encoder,
    )
    h11, h12, h13 = prepared.evidence1
    h21, h22 = prepared.evidence2
    del h11, h12, h21
    up4 = model.up_decoder4.up(prepared.d5)
    if up4.shape[-2:] != prepared.encoder[3].shape[-2:]:
        up4 = F.interpolate(
            up4,
            size=prepared.encoder[3].shape[-2:],
            mode="nearest",
        )
    q4, mask4 = model.tpd_ner.forward_stage(
        4,
        (h13, h22, up4),
        tuple(up4.shape[-2:]),
    )
    protection = model.ner_l4_tpr.build_protection(q4)
    _require(not protection.requires_grad, "L4 protection must be detached")
    _require(
        tuple(protection.shape)
        == (q4.shape[0], 1, q4.shape[2], q4.shape[3]),
        "L4 protection shape differs",
    )
    _require(
        set(float(value) for value in torch.unique(protection)) <= {0.0, 1.0},
        "L4 protection is not binary",
    )
    return PreparedL4TPRBatch(
        x1=baseline[0],
        x2=baseline[1],
        x3=baseline[2],
        transformed4=prepared.transformed[3],
        encoder4=prepared.encoder[3],
        baseline4=baseline[3],
        d5=prepared.d5,
        evidence1=prepared.evidence1,
        evidence2=prepared.evidence2,
        up4=up4,
        q4=q4,
        mask4=mask4,
        protection4=protection,
    )


def fuse_l4_public_mode(
    prepared: PreparedL4TPRBatch,
    public_mode: str,
) -> torch.Tensor:
    """Apply one cached L4 counterfactual without regrouping the baseline."""

    binding = normalize_public_mode(public_mode)
    if public_mode == CURRENT_MODE:
        return prepared.baseline4
    gate = prepared.transformed4.new_tensor(float(binding["gate_value"]))
    if bool(binding["protection_applied"]):
        routed_gate = prepared.protection4.neg().add(1.0).mul(gate)
    else:
        _require(
            public_mode == UNPROTECTED_MODE,
            "non-current unprotected mode differs",
        )
        routed_gate = gate
    correction = routed_gate.mul(prepared.transformed4).sub(
        routed_gate.mul(prepared.encoder4)
    )
    output = prepared.baseline4.add(correction)
    _require(bool(torch.isfinite(output).all()), "L4 fusion is non-finite")
    return output


def decode_l4_tpr_mode(
    model: nn.Module,
    prepared: PreparedL4TPRBatch,
    public_mode: str,
) -> torch.Tensor:
    """Rerun L4 CCA and downstream decoder from cached q4/protection."""

    x4 = fuse_l4_public_mode(prepared, public_mode)
    h11, h12, _ = prepared.evidence1
    h21, _ = prepared.evidence2
    skip4 = model.up_decoder4.coatt(g=prepared.up4, x=x4)
    d4 = model.up_decoder4.finish(
        prepared.up4,
        skip4,
        prepared.mask4,
    )
    up3, skip3 = model.up_decoder3.prepare(d4, prepared.x3)
    q3, mask3 = model.tpd_ner.forward_stage(
        3,
        (h12, h21, prepared.q4, up3),
        tuple(up3.shape[-2:]),
    )
    d3 = model.up_decoder3.finish(up3, skip3, mask3)
    up2, skip2 = model.up_decoder2.prepare(d3, prepared.x2)
    _, mask2 = model.tpd_ner.forward_stage(
        2,
        (h11, q3, up2),
        tuple(up2.shape[-2:]),
    )
    d2 = model.up_decoder2.finish(up2, skip2, mask2)
    output = torch.sigmoid(model.outc(model.up_decoder1(d2, prepared.x1)))
    _require(
        output.ndim == 4 and output.shape[1] == 1,
        "decoder output shape differs",
    )
    _require(bool(torch.isfinite(output).all()), "decoder output is non-finite")
    return output


def _array_collection_sha256(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        digest.update(len(array.shape).to_bytes(8, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _annotate_fixed_metrics(
    fixed: Mapping[str, Any],
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> dict[str, Any]:
    ready = dict(fixed)
    component_fp = int(ready["unmatched_predicted_pixels"])
    background_fp = int(
        shared.qfg_audit._all_background_false_positive_pixels(
            probabilities,
            targets,
            threshold=FIXED_THRESHOLD,
        )
    )
    ready["component_false_positive_pixels"] = component_fp
    ready["background_false_positive_pixels"] = background_fp
    # Retain the historical alias used by prior branch-audit comparators.
    ready["false_positive_pixels"] = background_fp
    ready["pd_count_and_value"] = {
        "matched_target_count": int(ready["matched_target_count"]),
        "target_count": int(ready["target_count"]),
        "pd": float(ready["pd"]),
    }
    ready["tiny_pd_count_and_value"] = {
        "matched_tiny_target_count": int(
            ready["matched_tiny_target_count"]
        ),
        "tiny_target_count": int(ready["tiny_target_count"]),
        "tiny_pd": float(ready["tiny_pd"]),
    }
    return ready


def reference_replay_audit(
    observed: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the measured same-runtime replay envelope.

    Detection denominators and matched total/tiny target counts remain exact.
    Only a one-count boundary-pixel/object drift and bounded floating
    reductions are accepted.  This audit is intentionally separate from the
    TPR comparison, whose current and candidate modes share one execution.
    """

    required = {
        "threshold",
        "miou",
        "niou",
        "pd",
        "fa",
        *REPLAY_EXACT_COUNT_KEYS,
        *REPLAY_ONE_COUNT_KEYS,
    }
    _require(required <= set(observed), "current replay metrics are incomplete")
    _require(required <= set(reference), "reference replay metrics are incomplete")
    compared: dict[str, Any] = {}
    for key in sorted(reference):
        if key not in observed:
            continue
        expected = reference[key]
        actual = observed[key]
        if key in REPLAY_EXACT_COUNT_KEYS:
            absolute_difference = abs(int(actual) - int(expected))
            tolerance: float | int = 0
            _require(
                absolute_difference == 0,
                f"reference replay exact count differs: {key}",
            )
        elif key in REPLAY_ONE_COUNT_KEYS:
            absolute_difference = abs(int(actual) - int(expected))
            tolerance = REPLAY_ONE_COUNT_ABSOLUTE_TOLERANCE
            _require(
                absolute_difference <= tolerance,
                f"reference replay boundary count differs: {key}",
            )
        elif expected is None:
            absolute_difference = 0.0
            tolerance = 0.0
            _require(actual is None, f"reference replay null differs: {key}")
        else:
            if key in {
                "miou",
                "niou",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
            }:
                tolerance = REPLAY_SOFT_METRIC_ABSOLUTE_TOLERANCE
            elif key == "fa":
                tolerance = REPLAY_FA_ABSOLUTE_TOLERANCE
            elif key == "test_loss":
                tolerance = REPLAY_TEST_LOSS_ABSOLUTE_TOLERANCE
            elif key == "false_objects_per_image":
                tolerance = (
                    REPLAY_FALSE_OBJECTS_PER_IMAGE_ABSOLUTE_TOLERANCE
                )
            else:
                tolerance = 1e-15
            absolute_difference = abs(float(actual) - float(expected))
            _require(
                math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=float(tolerance),
                ),
                f"reference replay metric differs: {key}",
            )
        compared[key] = {
            "absolute_difference": absolute_difference,
            "absolute_tolerance": tolerance,
        }
    return {
        "passed": True,
        "policy": REPLAY_POLICY,
        "matched_total_and_tiny_target_counts_exact": True,
        "one_boundary_count_maximum": (
            REPLAY_ONE_COUNT_ABSOLUTE_TOLERANCE
        ),
        "soft_metric_absolute_tolerance": (
            REPLAY_SOFT_METRIC_ABSOLUTE_TOLERANCE
        ),
        "fa_absolute_tolerance": REPLAY_FA_ABSOLUTE_TOLERANCE,
        "test_loss_absolute_tolerance": (
            REPLAY_TEST_LOSS_ABSOLUTE_TOLERANCE
        ),
        "compared": compared,
    }


@torch.inference_mode()
def analyze_loaded_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Evaluate six modes while preparing q4/protection once per batch."""

    model.eval()
    model.mode = "test"
    _require(not model.training, "NER-L4-TPR model must be eval")
    state_before = state_audit.module_state_sha256(model)
    probabilities: dict[str, list[np.ndarray]] = {
        mode: [] for mode in PUBLIC_MODES
    }
    losses: dict[str, list[float]] = {mode: [] for mode in PUBLIC_MODES}
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    criterion = nn.BCELoss(reduction="mean")
    batch_count = 0
    protection_positive_cells = 0
    protection_total_cells = 0

    for images, masks, sizes, sample_ids in loader:
        _require(
            int(images.shape[0]) == int(masks.shape[0]) == 1,
            "NER-L4-TPR screening requires batch_size=1",
        )
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        height, width = metric_runner._extract_hw(sizes)
        prepared = prepare_l4_tpr_batch(model, images)
        protection_positive_cells += int(
            torch.count_nonzero(prepared.protection4).item()
        )
        protection_total_cells += int(prepared.protection4.numel())
        target = masks[:, :, :height, :width]
        for public_mode in PUBLIC_MODES:
            prediction = decode_l4_tpr_mode(model, prepared, public_mode)[
                :, :, :height, :width
            ]
            _require(
                prediction.shape == target.shape,
                "prediction/target shape differs",
            )
            one_loss = criterion(prediction.float(), target.float())
            _require(
                math.isfinite(float(one_loss.item())),
                "test loss is non-finite",
            )
            probabilities[public_mode].append(
                prediction[0, 0]
                .float()
                .cpu()
                .contiguous()
                .numpy()
                .astype(np.float32, copy=False)
            )
            losses[public_mode].append(float(one_loss.item()))
        targets.append(target[0, 0].float().cpu().contiguous().numpy())
        _require(
            isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
            "NER-L4-TPR screening requires one sample ID per batch",
        )
        identifiers.append(str(sample_ids[0]))
        batch_count += 1
        del prepared

    _require(identifiers == list(expected_identifiers), "inference order differs")
    _require(batch_count == len(loader.dataset), "inference count differs")
    _require(protection_total_cells > 0, "protection was not observed")
    current_probabilities = probabilities[CURRENT_MODE]
    unprotected_probabilities = probabilities[UNPROTECTED_MODE]
    modes: dict[str, Any] = {}
    for public_mode in PUBLIC_MODES:
        evaluated = metric_runner.evaluate_probability_arrays(
            probabilities[public_mode],
            targets,
            losses[public_mode],
            sweep_thresholds=SWEEP_THRESHOLDS,
        )
        shared._annotate_two_point_sweep(evaluated)
        fixed = _annotate_fixed_metrics(
            evaluated["fixed_threshold_0_5"],
            probabilities[public_mode],
            targets,
        )
        modes[public_mode] = {
            **normalize_public_mode(public_mode),
            "fixed_threshold_0_5": fixed,
            "descriptive_pd_fa": evaluated["descriptive_pd_fa"],
            "threshold_roles": evaluated["threshold_roles"],
            "probability_sha256": _array_collection_sha256(
                probabilities[public_mode]
            ),
            "probability_difference_to_current": probability_difference(
                current_probabilities,
                probabilities[public_mode],
            ),
            "probability_difference_to_unprotected_gpos025": (
                probability_difference(
                    unprotected_probabilities,
                    probabilities[public_mode],
                )
            ),
        }

    state_after = state_audit.module_state_sha256(model)
    _require(state_after == state_before, "screening changed model state")
    return {
        "modes": modes,
        "execution_audit": {
            "batch_count": batch_count,
            "encoder_tpd_qfg_prepare_count": batch_count,
            "up4_prepare_count": batch_count,
            "q4_forward_count": batch_count,
            "protection_build_count": batch_count,
            "decoder_mode_count_per_batch": len(PUBLIC_MODES),
            "l4_cca_execution_count": batch_count * len(PUBLIC_MODES),
            "decoder_execution_count": batch_count * len(PUBLIC_MODES),
            "encoder_tpd_qfg_recomputed_per_mode": False,
            "q4_recomputed_per_mode": False,
            "protection_recomputed_per_mode": False,
            "forward_local_feature_reuse_only": True,
        },
        "protection_statistics": {
            "positive_cell_count": protection_positive_cells,
            "total_cell_count": protection_total_cells,
            "positive_fraction": (
                protection_positive_cells / protection_total_cells
            ),
            "binary": True,
            "detached": True,
        },
        "restoration_audit": {
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_after == state_before,
        },
        "probability_arrays_persisted": False,
        "feature_tensors_persisted": False,
    }


def _source_sha256() -> dict[str, str]:
    sources = {
        "analysis/analyze_three_dataset_ner_l4_tpr_v1.py": Path(__file__),
        "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py": Path(
            shared.__file__
        ),
        "experiments/ner_l4_tpr_strict_migration_v1.py": Path(
            migration.__file__
        ),
        "experiments/evaluate_three_dataset_v2.py": Path(
            metric_runner.__file__
        ),
        "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": Path(
            adapter.__file__
        ),
        "experiments/three_dataset_v2_protocol.py": Path(
            data_protocol.__file__
        ),
        "model/tpd_ner_l4_target_protected_reallocation.py": Path(
            tpr_core.__file__
        ),
        (
            "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_"
            "croa_l4_tpr.py"
        ): Path(tpr_model.__file__),
    }
    return {
        name: file_sha256(path.resolve(strict=True))
        for name, path in sources.items()
    }


def analyze_run(
    *,
    dataset: str,
    checkpoint_role: str,
    identity_manifest: Path,
    identity_manifest_sha256: str,
    dataset_root: Path,
    device_name: str,
    workers: int,
) -> dict[str, Any]:
    _require(dataset in DATASETS, "dataset is outside formal scope")
    _require(checkpoint_role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(workers >= 0, "workers must be non-negative")
    _require(
        identity_manifest_sha256 == FROZEN_IDENTITY_MANIFEST_SHA256,
        "identity manifest SHA differs from the frozen V1 contract",
    )
    sources_before = _source_sha256()
    adapter.configure_core()
    training_engine.configure_determinism()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    binding = migration.load_manifest_binding(
        identity_manifest,
        expected_manifest_sha256=identity_manifest_sha256,
        dataset=dataset,
        checkpoint_role=checkpoint_role,
    )
    manifest_path = Path(binding["data_protocol_manifest"]["path"])
    manifest = data_protocol.load_protocol_manifest(
        manifest_path,
        dataset_root=dataset_root,
    )
    reference_path = Path(binding["reference_evaluation_path"])
    reference = adapter.validate_completed_output(
        reference_path,
        dataset=dataset,
        checkpoint_role=checkpoint_role,
    )
    _require(
        reference["checkpoint_binding"]["checkpoint"]["sha256"]
        == binding["checkpoint_sha256"],
        "reference evaluation/checkpoint SHA differs",
    )
    model, model_metadata = migration.build_zero_extension_inference_model(
        binding
    )
    model.to(device)
    model.eval()
    model.mode = "test"
    dataset_object = metric_runner.ThreeDatasetTestDataset(
        dataset_root,
        dataset,
        manifest_path,
    )
    loader = DataLoader(
        dataset_object,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    analyzed = analyze_loaded_model(
        model,
        loader,
        device,
        dataset_object.sample_ids,
    )
    replay = reference_replay_audit(
        analyzed["modes"][CURRENT_MODE]["fixed_threshold_0_5"],
        reference["fixed_threshold_0_5"],
    )
    replay["comparison"] = (
        f"current_g0_fixed_threshold_0_5_vs_existing_{checkpoint_role}"
    )
    _require(replay["passed"] is True, "current reference replay failed")
    ordered_id_sha = hashlib.sha256(
        ("\n".join(dataset_object.sample_ids) + "\n").encode("utf-8")
    ).hexdigest()
    _require(_source_sha256() == sources_before, "runtime source changed")

    output = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": REFERENCE_METHOD,
        "checkpoint_role": checkpoint_role,
        "seed": SEED,
        "test_selected": True,
        "selection_is_optimistic": True,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "fixed_threshold": FIXED_THRESHOLD,
        "mode_order": list(PUBLIC_MODES),
        **analyzed,
        "reference_replay_audit": replay,
        "identity_manifest_binding": {
            "path": str(Path(identity_manifest).resolve(strict=True)),
            "sha256": identity_manifest_sha256,
            "schema": migration.MANIFEST_SCHEMA,
            "dataset": dataset,
            "checkpoint_role": checkpoint_role,
            "primary_screening_role": migration.PRIMARY_SCREENING_ROLE,
            "supplemental_screening_role": (
                migration.SUPPLEMENTAL_SCREENING_ROLE
            ),
        },
        "checkpoint_binding": {
            "path": binding["checkpoint_path"],
            "sha256": binding["checkpoint_sha256"],
            "epoch": binding["epoch"],
            "role": checkpoint_role,
        },
        "reference_reuse": {
            "path": str(reference_path),
            "sha256": binding["reference_evaluation_sha256"],
            "checkpoint_role": checkpoint_role,
            "fixed_threshold_0_5": reference["fixed_threshold_0_5"],
        },
        "model": model_metadata,
        "data": {
            "dataset_root": str(Path(dataset_root).resolve()),
            "protocol_manifest": {
                "path": str(manifest_path),
                "sha256": binding["data_protocol_manifest"]["sha256"],
                "schema": manifest.get("schema"),
                "manifest_id": manifest.get("manifest_id"),
            },
            "split": "img_idx/test",
            "test_count": len(dataset_object.sample_ids),
            "inference_order_newline_sha256": ordered_id_sha,
            "normalization": metric_runner.NORMALIZATION[dataset],
            "sirst3_in_formal_matrix": False,
        },
        "metric_protocol": {
            "implementation": (
                "experiments.train_tpd_pilot.ValidationMetrics"
            ),
            "fixed_threshold": FIXED_THRESHOLD,
            "prediction_comparison": "probability > 0.5",
            "match_radius": metric_runner.MATCH_RADIUS,
            "tiny_area": metric_runner.TINY_AREA,
            "checkpoint_reselection": False,
            "descriptive_thresholds": list(SWEEP_THRESHOLDS),
        },
        "intervention_contract": {
            "family": "NER_L4_target_protected_reallocation_v1",
            "current_formula": "(T4+E4)+E4",
            "protected_formula": (
                "baseline+(1-P4)*g*T4-(1-P4)*g*E4"
            ),
            "unprotected_reference_formula": (
                "baseline+0.25*T4-0.25*E4"
            ),
            "q4_and_protection_prepared_once_per_batch": True,
            "model_state_modified": False,
            "derived_checkpoint_written": False,
            "formal_training_started": False,
        },
        "source_sha256": sources_before,
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
        "feature_cache_written": False,
        "formal_training_started": False,
        "no_fabricated_results": True,
        "stability_claim_supported": False,
    }
    validate_output_payload(output)
    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _finite(value: Any, name: str) -> float:
    _require(not isinstance(value, bool), f"{name} must be numeric")
    ready = float(value)
    _require(math.isfinite(ready), f"{name} must be finite")
    return ready


def validate_output_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer is incomplete")
    _require(payload.get("dataset") in DATASETS, "dataset differs")
    role = payload.get("checkpoint_role")
    _require(role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("fixed_threshold") == FIXED_THRESHOLD, "threshold differs")
    _require(payload.get("mode_order") == list(PUBLIC_MODES), "mode order differs")
    replay = payload.get("reference_replay_audit")
    _require(
        isinstance(replay, Mapping)
        and replay.get("passed") is True
        and replay.get("policy") == REPLAY_POLICY
        and replay.get("matched_total_and_tiny_target_counts_exact") is True
        and replay.get("comparison")
        == f"current_g0_fixed_threshold_0_5_vs_existing_{role}",
        "reference replay differs",
    )
    identity = payload.get("identity_manifest_binding")
    _require(
        isinstance(identity, Mapping)
        and identity.get("sha256") == FROZEN_IDENTITY_MANIFEST_SHA256
        and identity.get("schema") == migration.MANIFEST_SCHEMA
        and identity.get("dataset") == payload.get("dataset")
        and identity.get("checkpoint_role") == role,
        "identity manifest binding differs",
    )
    checkpoint = payload.get("checkpoint_binding")
    _require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("role") == role
        and migration._is_sha256(checkpoint.get("sha256"))
        and type(checkpoint.get("epoch")) is int,
        "checkpoint binding differs",
    )
    modes = payload.get("modes")
    _require(
        isinstance(modes, Mapping) and set(modes) == set(PUBLIC_MODES),
        "mode set differs",
    )
    invariant: tuple[int, int, int] | None = None
    required = {
        "threshold",
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "pd",
        "tiny_pd",
        "fa",
        "miou",
        "niou",
        "unmatched_predicted_pixels",
        "component_false_positive_pixels",
        "false_positive_pixels",
        "background_false_positive_pixels",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "valid_pixel_count",
        "pd_count_and_value",
        "tiny_pd_count_and_value",
    }
    for public_mode in PUBLIC_MODES:
        mode = modes[public_mode]
        _require(isinstance(mode, Mapping), f"{public_mode} is malformed")
        binding = normalize_public_mode(public_mode)
        for key in (
            "public_mode",
            "gate_value",
            "level",
            "protection_applied",
            "unprotected_reference",
            "current_reference",
            "finite_l4_tpr_logit_representable",
            "finite_logit_representable",
            "required_reallocation_logit",
            "required_logit",
            "boundary_limit_counterfactual",
        ):
            _require(mode.get(key) == binding[key], f"{public_mode}.{key} differs")
        fixed = mode.get("fixed_threshold_0_5")
        _require(
            isinstance(fixed, Mapping) and required <= set(fixed),
            f"{public_mode} fixed metrics differ",
        )
        _require(
            _finite(fixed["threshold"], "threshold") == FIXED_THRESHOLD,
            "fixed threshold differs",
        )
        target_count = int(fixed["target_count"])
        matched = int(fixed["matched_target_count"])
        tiny_count = int(fixed["tiny_target_count"])
        matched_tiny = int(fixed["matched_tiny_target_count"])
        valid_pixels = int(fixed["valid_pixel_count"])
        _require(target_count > 0 and valid_pixels > 0, "metric totals differ")
        _require(0 <= matched <= target_count, "matched target count differs")
        _require(0 <= matched_tiny <= tiny_count, "matched tiny count differs")
        _require(
            math.isclose(
                _finite(fixed["pd"], "pd"),
                matched / target_count,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "Pd count/value identity differs",
        )
        _require(
            math.isclose(
                _finite(fixed["tiny_pd"], "tiny_pd"),
                matched_tiny / tiny_count,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "tiny-Pd count/value identity differs",
        )
        component_fp = int(fixed["component_false_positive_pixels"])
        background_fp = int(fixed["background_false_positive_pixels"])
        _require(
            component_fp == int(fixed["unmatched_predicted_pixels"]),
            "component FP alias differs",
        )
        _require(
            background_fp == int(fixed["false_positive_pixels"]),
            "background FP alias differs",
        )
        _require(
            math.isclose(
                _finite(fixed["fa"], "fa"),
                component_fp / valid_pixels,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "Fa numerator identity differs",
        )
        for metric in (
            "miou",
            "niou",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
        ):
            value = _finite(fixed[metric], metric)
            _require(0.0 <= value <= 1.0, f"{metric} range differs")
        current_invariant = (target_count, tiny_count, valid_pixels)
        invariant = invariant or current_invariant
        _require(current_invariant == invariant, "mode totals differ")
        _require(
            migration._is_sha256(mode.get("probability_sha256")),
            "probability SHA differs",
        )
        for difference_key in (
            "probability_difference_to_current",
            "probability_difference_to_unprotected_gpos025",
        ):
            difference = mode.get(difference_key)
            _require(
                isinstance(difference, Mapping)
                and int(difference.get("element_count", 0)) == valid_pixels,
                f"{public_mode}.{difference_key} differs",
            )
        if public_mode == CURRENT_MODE:
            difference = mode["probability_difference_to_current"]
            _require(
                float(difference["max_abs"]) == 0.0
                and float(difference["absolute_difference_sum"]) == 0.0,
                "current self-difference is nonzero",
            )
        if public_mode == UNPROTECTED_MODE:
            difference = mode[
                "probability_difference_to_unprotected_gpos025"
            ]
            _require(
                float(difference["max_abs"]) == 0.0
                and float(difference["absolute_difference_sum"]) == 0.0,
                "unprotected self-difference is nonzero",
            )

    execution = payload.get("execution_audit")
    _require(isinstance(execution, Mapping), "execution audit missing")
    batch_count = int(execution.get("batch_count", 0))
    _require(batch_count > 0, "batch count differs")
    for key in (
        "encoder_tpd_qfg_prepare_count",
        "up4_prepare_count",
        "q4_forward_count",
        "protection_build_count",
    ):
        _require(execution.get(key) == batch_count, f"{key} differs")
    _require(
        execution.get("decoder_mode_count_per_batch") == len(PUBLIC_MODES)
        and execution.get("l4_cca_execution_count")
        == batch_count * len(PUBLIC_MODES)
        and execution.get("decoder_execution_count")
        == batch_count * len(PUBLIC_MODES)
        and execution.get("encoder_tpd_qfg_recomputed_per_mode") is False
        and execution.get("q4_recomputed_per_mode") is False
        and execution.get("protection_recomputed_per_mode") is False,
        "prepare-once execution contract differs",
    )
    restoration = payload.get("restoration_audit")
    _require(
        isinstance(restoration, Mapping)
        and restoration.get("model_state_unchanged") is True
        and restoration.get("model_state_sha256_before")
        == restoration.get("model_state_sha256_after"),
        "model restoration differs",
    )
    _require(
        payload.get("derived_checkpoint_written") is False
        and payload.get("probability_cache_written") is False
        and payload.get("feature_cache_written") is False
        and payload.get("formal_training_started") is False,
        "artifact/training flags differ",
    )


def _default_output(dataset: str, checkpoint_role: str) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / dataset
        / f"final_tss_off_{checkpoint_role}_seed42"
        / "evaluation.json"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--checkpoint-role",
        choices=CHECKPOINT_ROLES,
        required=True,
    )
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        default=DEFAULT_IDENTITY_MANIFEST,
    )
    parser.add_argument(
        "--identity-manifest-sha256",
        default=FROZEN_IDENTITY_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=data_protocol.DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not (args.device == "cpu" or args.device.startswith("cuda:")):
        parser.error("--device must be cpu or cuda:N")
    args.output = args.output or _default_output(
        args.dataset,
        args.checkpoint_role,
    )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(
            f"refusing existing output before inference: {args.output}"
        )
    output = analyze_run(
        dataset=args.dataset,
        checkpoint_role=args.checkpoint_role,
        identity_manifest=args.identity_manifest,
        identity_manifest_sha256=args.identity_manifest_sha256,
        dataset_root=args.dataset_root,
        device_name=args.device,
        workers=args.workers,
    )
    atomic_create_json(args.output, output)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete",
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output.resolve()),
                "dataset": args.dataset,
                "checkpoint_role": args.checkpoint_role,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
