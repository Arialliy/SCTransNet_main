#!/usr/bin/env python3
"""Strict fixed-0.5 evaluator for the four-dataset seed-42 experiment.

Only the two frozen, test-selected checkpoints are accepted.  Final checkpoints
are converted to the TSS-free inference graph before strict loading.  The
metric implementation is the repository's frozen 8-connected, one-to-one,
centroid-distance-<3 core used by the formal TPD experiments.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import four_dataset_evaluation_protocol_v1 as protocol  # noqa: E402


@dataclass(frozen=True)
class EvaluationRequest:
    training_dataset: str
    evaluation_dataset: str
    method: str
    checkpoint_role: str

    @property
    def is_sirst3_source(self) -> bool:
        return (
            self.training_dataset == "SIRST3"
            and self.evaluation_dataset in protocol.SOURCE_DATASETS
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=protocol.EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "datasets",
    )
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--method", choices=protocol.METHODS)
    parser.add_argument("--checkpoint-role", choices=protocol.CHECKPOINT_ROLES)
    parser.add_argument(
        "--all-dataset-specific",
        action="store_true",
        help="Evaluate 4 datasets x 2 methods x 2 selected roles.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--imgidx-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--normalization-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--correction-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--data-gate",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.all_dataset_specific:
        if any(
            value is not None
            for value in (args.dataset, args.method, args.checkpoint_role)
        ):
            parser.error(
                "--all-dataset-specific cannot be combined with a single request"
            )
    elif any(
        value is None
        for value in (args.dataset, args.method, args.checkpoint_role)
    ):
        parser.error(
            "provide --all-dataset-specific or dataset/method/checkpoint-role"
        )
    return args


def default_artifact_paths(
    results_root: Path,
) -> dict[str, Path]:
    manifests = Path(results_root) / "manifests"
    return {
        "imgidx_manifest": manifests / "four_dataset_imgidx_v1.json",
        "normalization_manifest": manifests / "four_dataset_legacy_norm_v1.json",
        "correction_manifest": manifests / "nuaa_misc111_correction_v1.json",
        "data_gate": manifests / "four_dataset_data_gate_v1.json",
    }


def _require_data_gate(path: Path) -> dict[str, Any]:
    gate = protocol.load_json_object(path)
    readiness_fields = (
        "formal_training_and_evaluation_ready",
        "ready",
        "passed",
    )
    observed = [gate.get(field) for field in readiness_fields if field in gate]
    status_ready = gate.get("status") in {"complete", "pass", "passed", "ready"}
    protocol.require(
        any(value is True for value in observed) or status_ready,
        f"data gate is not ready: {path}",
    )
    return {
        "path": str(path.resolve()),
        "sha256": protocol.file_sha256(path),
        "schema": gate.get("schema"),
        "status": gate.get("status"),
        "readiness": {
            field: gate.get(field)
            for field in readiness_fields
            if field in gate
        },
    }


def _manifest_index(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    records = manifest.get("records")
    protocol.require(isinstance(records, list), "checkpoint manifest lacks records")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        protocol.require(isinstance(record, Mapping), "invalid checkpoint record")
        key = (str(record.get("dataset")), str(record.get("method")))
        protocol.require(key not in index, f"duplicate checkpoint record: {key}")
        index[key] = record
    return index


def _load_checkpoint_binding(
    request: EvaluationRequest,
    results_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = (
        Path(results_root)
        / "selected_checkpoints"
        / "checkpoint_manifest.json"
    )
    manifest = protocol.load_json_object(manifest_path)
    protocol.require(
        manifest.get("status") == "complete",
        "checkpoint manifest is not complete",
    )
    index = _manifest_index(manifest)
    key = (request.training_dataset, request.method)
    protocol.require(key in index, f"checkpoint manifest lacks run {key}")
    run_record = index[key]
    checkpoints = run_record.get("checkpoints")
    protocol.require(
        isinstance(checkpoints, Mapping),
        f"checkpoint manifest run {key} lacks checkpoints",
    )
    binding = checkpoints.get(request.checkpoint_role)
    protocol.require(
        isinstance(binding, Mapping),
        f"checkpoint manifest lacks role {request.checkpoint_role}",
    )
    expected_path = protocol.selected_checkpoint_path(
        request.training_dataset,
        request.method,
        request.checkpoint_role,
        selected_root=Path(results_root) / "selected_checkpoints",
    )
    protocol.require(
        expected_path.is_file() and not expected_path.is_symlink(),
        f"frozen checkpoint is missing: {expected_path}",
    )
    digest = protocol.file_sha256(expected_path)
    protocol.require(
        digest == binding.get("sha256"),
        f"frozen checkpoint SHA differs: {expected_path}",
    )
    payload = torch.load(
        expected_path,
        map_location="cpu",
        weights_only=False,
    )
    protocol.require(
        isinstance(payload, dict),
        "selected checkpoint payload must be a dictionary",
    )
    for field, expected in (
        ("dataset", request.training_dataset),
        ("method", request.method),
        ("seed", protocol.TRAINING_SEED),
        ("checkpoint_role", request.checkpoint_role),
        ("epoch", binding.get("epoch")),
        ("test_selected", True),
        ("selection_is_optimistic", True),
    ):
        protocol.require(
            payload.get(field) == expected,
            f"selected checkpoint {field} differs: {expected_path}",
        )
    state_dict = payload.get("state_dict")
    protocol.require(
        isinstance(state_dict, Mapping) and state_dict,
        "selected checkpoint lacks a state_dict",
    )
    manifest_audit = {
        "path": str(manifest_path.resolve()),
        "sha256": protocol.file_sha256(manifest_path),
        "schema": manifest.get("schema"),
    }
    checkpoint = {
        "path": str(expected_path.resolve()),
        "sha256": digest,
        "epoch": int(binding["epoch"]),
        "role": request.checkpoint_role,
        "test_selected": True,
        "selection_is_optimistic": True,
    }
    return payload, checkpoint, manifest_audit


def build_inference_model(
    request: EvaluationRequest,
    training_state_dict: Mapping[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    from experiments import four_dataset_models_seed42_v1 as models

    if request.method == "final":
        model, metadata = (
            models.build_final_inference_model_from_training_state_dict(
                training_state_dict,
                dataset_name=request.training_dataset,
                seed=protocol.TRAINING_SEED,
            )
        )
        protocol.require(
            metadata.get("target_survival_registered") is False,
            "Final inference graph still registers TSS",
        )
        protocol.require(
            metadata.get("strict_load") is True,
            "Final inference state was not strictly loaded",
        )
        return model, dict(metadata)
    model, metadata = models.build_paper_model(
        request.method,
        request.training_dataset,
        seed=protocol.TRAINING_SEED,
        training=False,
    )
    incompatible = model.load_state_dict(training_state_dict, strict=True)
    protocol.require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "Original strict checkpoint load returned incompatible keys",
    )
    model.eval()
    model.mode = "test"
    ready = dict(metadata)
    ready.update(
        {
            "strict_load": True,
            "target_survival_registered": False,
        }
    )
    return model, ready


def _extract_hw(sizes: Any) -> tuple[int, int]:
    if isinstance(sizes, torch.Tensor):
        value = sizes.detach().cpu()
        if value.ndim == 2:
            return int(value[0, 0]), int(value[0, 1])
        if value.ndim == 1 and value.numel() == 2:
            return int(value[0]), int(value[1])
    if isinstance(sizes, (tuple, list)) and len(sizes) == 2:
        first, second = sizes
        if isinstance(first, torch.Tensor):
            first = first.reshape(-1)[0].item()
        if isinstance(second, torch.Tensor):
            second = second.reshape(-1)[0].item()
        return int(first), int(second)
    raise TypeError(f"unsupported collated size value: {type(sizes)!r}")


def final_prediction(outputs: Any) -> torch.Tensor:
    evaluator = getattr(outputs, "evaluator_prediction", None)
    if callable(evaluator):
        return evaluator()
    if isinstance(outputs, (tuple, list)):
        return outputs[-1]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"unsupported model output: {type(outputs)!r}")


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[str]]:
    model.eval()
    criterion = nn.BCELoss(reduction="mean")
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    identifiers: list[str] = []
    for images, masks, sizes, sample_ids in loader:
        protocol.require(
            int(images.shape[0]) == 1 and int(masks.shape[0]) == 1,
            "formal evaluator requires batch_size=1",
        )
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height, width = _extract_hw(sizes)
        prediction = final_prediction(model(images))[:, :, :height, :width]
        target = masks[:, :, :height, :width]
        protocol.require(
            prediction.shape == target.shape,
            "prediction and target shapes differ after unpadding",
        )
        protocol.require(
            bool(torch.isfinite(prediction).all()),
            "model prediction contains non-finite values",
        )
        loss = criterion(prediction.float(), target.float())
        protocol.require(
            math.isfinite(float(loss.item())),
            "test loss is non-finite",
        )
        probability = prediction[0, 0].float().cpu().numpy()
        target_array = target[0, 0].float().cpu().numpy()
        protocol.require(
            np.isfinite(probability).all() and np.isfinite(target_array).all(),
            "prediction or target array contains non-finite values",
        )
        probabilities.append(probability)
        targets.append(target_array)
        losses.append(float(loss.item()))
        if isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1:
            identifiers.append(str(sample_ids[0]))
        else:
            raise TypeError("test loader must collate one sample ID")
    protocol.require(
        bool(probabilities) and len(probabilities) == len(loader.dataset),
        "prediction count differs from test dataset",
    )
    protocol.require(
        len(identifiers) == len(set(identifiers)),
        "test loader yielded duplicate sample IDs",
    )
    return probabilities, targets, losses, identifiers


def _ordered_identifier_sha256(identifiers: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()


def build_test_loader(
    request: EvaluationRequest,
    *,
    data_root: Path,
    imgidx_manifest: Path,
    normalization_manifest: Path,
    correction_manifest: Path,
    workers: int,
    device: torch.device,
) -> tuple[DataLoader, dict[str, Any]]:
    from experiments.paper_four_dataset_v1 import build_test_dataset

    dataset = build_test_dataset(
        test_dataset_name=request.evaluation_dataset,
        normalization_dataset=request.training_dataset,
        dataset_root=data_root,
        imgidx_manifest=imgidx_manifest,
        normalization_manifest=normalization_manifest,
        correction_manifest=correction_manifest,
        return_metadata=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    data_binding = {
        "dataset_root": str(Path(data_root).resolve()),
        "training_dataset": request.training_dataset,
        "evaluation_dataset": request.evaluation_dataset,
        "normalization_dataset": request.training_dataset,
        "test_count": len(dataset),
        "imgidx_manifest": {
            "path": str(imgidx_manifest.resolve()),
            "sha256": protocol.file_sha256(imgidx_manifest),
        },
        "normalization_manifest": {
            "path": str(normalization_manifest.resolve()),
            "sha256": protocol.file_sha256(normalization_manifest),
        },
        "correction_manifest": {
            "path": str(correction_manifest.resolve()),
            "sha256": protocol.file_sha256(correction_manifest),
        },
    }
    return loader, data_binding


def _checkpoint_metric_audit(
    checkpoint_payload: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    raw = checkpoint_payload.get("test_metrics")
    protocol.require(
        isinstance(raw, Mapping),
        "checkpoint lacks fixed-threshold test metrics",
    )
    event = {
        "epoch": checkpoint_payload["epoch"],
        "test_metrics": dict(raw),
        "threshold": protocol.FIXED_THRESHOLD,
    }
    expected = protocol.normalize_metric_event(event)
    exact: dict[str, Any] = {}
    tolerances: dict[str, float] = {}
    for key, value in expected.items():
        if key in {"epoch", "threshold"} or key not in fixed:
            continue
        observed = fixed[key]
        if isinstance(value, int) and not isinstance(value, bool):
            protocol.require(
                observed == value,
                f"fixed evaluator raw count differs for {key}",
            )
            exact[key] = value
        elif value is None:
            protocol.require(
                observed is None,
                f"fixed evaluator null metric differs for {key}",
            )
            exact[key] = None
        else:
            if key == "test_loss":
                tolerance = 1e-7
            elif key in {
                "miou",
                "niou",
                "pixel_f1",
                "pixel_precision",
                "pixel_recall",
            }:
                # Re-evaluating a frozen CUDA checkpoint can change the
                # continuous pixel aggregates by a few 1e-5 even when every
                # thresholded prediction/count is identical.  Keep all raw
                # counts and count-derived detection metrics exact, while
                # allowing less than 0.01 percentage point numerical drift
                # for continuous pixel metrics.
                tolerance = 1e-4
            else:
                tolerance = 1e-15
            protocol.require(
                math.isclose(
                    float(observed),
                    float(value),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ),
                f"fixed evaluator metric differs for {key}: "
                f"checkpoint={value!r}, reevaluated={observed!r}",
            )
            tolerances[key] = tolerance
    return {
        "passed": True,
        "exact_count_or_null_matches": exact,
        "numeric_absolute_tolerances": tolerances,
    }


def output_path_for_request(
    request: EvaluationRequest,
    *,
    results_root: Path,
    sweep: bool,
) -> Path:
    base = Path(results_root) / "evaluations"
    if sweep:
        if request.is_sirst3_source:
            base = base / "pd_fa_sweeps" / "sirst3_three_sources"
        else:
            base = base / "pd_fa_sweeps"
    elif request.is_sirst3_source:
        base = base / "sirst3_three_sources"
    else:
        base = base / "fixed_0_5"
    return (
        base
        / request.evaluation_dataset
        / request.method
        / f"{request.checkpoint_role}.json"
    )


def evaluate_request(
    request: EvaluationRequest,
    *,
    results_root: Path,
    data_root: Path,
    imgidx_manifest: Path,
    normalization_manifest: Path,
    correction_manifest: Path,
    data_gate: Path,
    device_name: str,
    workers: int,
    sweep: bool,
    overwrite: bool,
) -> dict[str, Any]:
    protocol.require(
        request.training_dataset in protocol.DATASETS,
        "unsupported training dataset",
    )
    protocol.require(
        request.evaluation_dataset in protocol.DATASETS,
        "unsupported evaluation dataset",
    )
    protocol.require(request.method in protocol.METHODS, "unsupported method")
    protocol.require(
        request.checkpoint_role in protocol.CHECKPOINT_ROLES,
        "only best_miou/best_pd checkpoints may be evaluated",
    )
    protocol.require(
        request.training_dataset == request.evaluation_dataset
        or request.is_sirst3_source,
        "cross-dataset evaluation is allowed only for SIRST3 three-source reuse",
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    gate_binding = _require_data_gate(data_gate)
    checkpoint_payload, checkpoint, checkpoint_manifest = (
        _load_checkpoint_binding(request, results_root)
    )
    model, model_metadata = build_inference_model(
        request,
        checkpoint_payload["state_dict"],
    )
    model.to(device)
    loader, data_binding = build_test_loader(
        request,
        data_root=data_root,
        imgidx_manifest=imgidx_manifest,
        normalization_manifest=normalization_manifest,
        correction_manifest=correction_manifest,
        workers=workers,
        device=device,
    )
    probabilities, targets, losses, identifiers = collect_predictions(
        model,
        loader,
        device,
    )
    if sweep:
        thresholds, threshold_provenance = (
            protocol.closed_interval_thresholds(probabilities)
        )
    else:
        thresholds = [protocol.FIXED_THRESHOLD]
        threshold_provenance = {
            "fixed_threshold_only": True,
            "threshold": protocol.FIXED_THRESHOLD,
        }
    points = protocol.strict_metric_points(
        probabilities,
        targets,
        losses,
        thresholds,
    )
    fixed_candidates = [
        point
        for point in points
        if float(point["threshold"]) == protocol.FIXED_THRESHOLD
    ]
    protocol.require(
        len(fixed_candidates) == 1,
        "evaluation must contain exactly one threshold=0.5 point",
    )
    fixed = fixed_candidates[0]
    metric_audit = (
        _checkpoint_metric_audit(checkpoint_payload, fixed)
        if request.training_dataset == request.evaluation_dataset
        else {
            "passed": True,
            "checkpoint_metrics_not_compared": (
                "source subset is not the SIRST3 selection aggregate"
            ),
        }
    )
    data_binding["ordered_sample_id_sha256"] = _ordered_identifier_sha256(
        identifiers
    )
    data_binding["observed_test_count"] = len(identifiers)
    data_binding["data_gate"] = gate_binding
    output: dict[str, Any] = {
        "schema": (
            f"{protocol.SCHEMA_PREFIX}_pd_fa_sweep_v1"
            if sweep
            else f"{protocol.SCHEMA_PREFIX}_fixed_0_5_v1"
        ),
        "status": "complete",
        "training_dataset": request.training_dataset,
        "evaluation_dataset": request.evaluation_dataset,
        "normalization_dataset": request.training_dataset,
        "method": request.method,
        "method_label": protocol.METHOD_LABELS[request.method],
        "checkpoint_role": request.checkpoint_role,
        "seed": protocol.TRAINING_SEED,
        "checkpoint": checkpoint,
        "checkpoint_manifest": checkpoint_manifest,
        "fixed_threshold_0_5": fixed,
        "metric_protocol": {
            "implementation": (
                "experiments.train_tpd_pilot.ValidationMetrics"
            ),
            "connectivity": 8,
            "matching": "one_to_one_max_cardinality_min_distance",
            "centroid_radius_comparison": "distance < 3",
            "match_radius": protocol.MATCH_RADIUS,
            "tiny_area": protocol.TINY_AREA,
            "prediction_comparison": "probability > threshold",
            "score_dtype": "float32",
        },
        "checkpoint_metric_audit": metric_audit,
        "data": data_binding,
        "model": model_metadata,
        "source_subset_of_selection": request.is_sirst3_source,
        "selection_parent": (
            "test_SIRST3"
            if request.is_sirst3_source
            else f"test_{request.training_dataset}"
        ),
        "checkpoint_reselected_for_source": False,
        "test_selected": True,
        "selection_is_optimistic": True,
        "threshold_provenance": threshold_provenance,
        "source_sha256": {
            **protocol.source_hashes(),
            "experiments/evaluate_four_dataset_seed42_v1.py": (
                protocol.file_sha256(Path(__file__).resolve())
            ),
            "experiments/four_dataset_models_seed42_v1.py": (
                protocol.file_sha256(
                    REPO_ROOT
                    / "experiments/four_dataset_models_seed42_v1.py"
                )
            ),
            "experiments/paper_four_dataset_v1.py": protocol.file_sha256(
                REPO_ROOT / "experiments/paper_four_dataset_v1.py"
            ),
        },
        "no_fabricated_results": True,
        "stability_claim_supported": False,
    }
    if sweep:
        output.update(
            {
                "fa_budgets": list(protocol.FA_BUDGETS),
                "best_points_under_fa_budget": protocol.fa_budget_points(
                    points
                ),
                "pareto_frontier": protocol.pareto_frontier(points),
                "points": points,
            }
        )
    destination = output_path_for_request(
        request,
        results_root=results_root,
        sweep=sweep,
    )
    protocol.atomic_write_json(destination, output, overwrite=overwrite)
    del model, loader, probabilities, targets, losses
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "request": {
            "training_dataset": request.training_dataset,
            "evaluation_dataset": request.evaluation_dataset,
            "method": request.method,
            "checkpoint_role": request.checkpoint_role,
        },
        "output": str(destination.resolve()),
        "sha256": protocol.file_sha256(destination),
        "fixed_threshold_0_5": fixed,
    }


def resolve_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    defaults = default_artifact_paths(args.results_root)
    return {
        key: (
            Path(getattr(args, key)).resolve()
            if getattr(args, key) is not None
            else path.resolve()
        )
        for key, path in defaults.items()
    }


def dataset_specific_requests(
    args: argparse.Namespace,
) -> list[EvaluationRequest]:
    if args.all_dataset_specific:
        return [
            EvaluationRequest(dataset, dataset, method, role)
            for dataset in protocol.DATASETS
            for method in protocol.METHODS
            for role in protocol.CHECKPOINT_ROLES
        ]
    return [
        EvaluationRequest(
            args.dataset,
            args.dataset,
            args.method,
            args.checkpoint_role,
        )
    ]


def run_requests(
    requests: Sequence[EvaluationRequest],
    args: argparse.Namespace,
    *,
    sweep: bool,
) -> list[dict[str, Any]]:
    artifacts = resolve_artifact_paths(args)
    return [
        evaluate_request(
            request,
            results_root=args.results_root.resolve(),
            data_root=args.data_root.resolve(),
            imgidx_manifest=artifacts["imgidx_manifest"],
            normalization_manifest=artifacts["normalization_manifest"],
            correction_manifest=artifacts["correction_manifest"],
            data_gate=artifacts["data_gate"],
            device_name=args.device,
            workers=args.workers,
            sweep=sweep,
            overwrite=args.overwrite,
        )
        for request in requests
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = run_requests(dataset_specific_requests(args), args, sweep=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "evaluation_count": len(results),
                "outputs": results,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
