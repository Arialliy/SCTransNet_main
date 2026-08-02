#!/usr/bin/env python3
"""Strict evaluator for the three-dataset V2 TSS recipe experiment.

The evaluator has two deliberately separate threshold roles:

* ``threshold=0.5`` is the only point used by ``best_miou``/
  ``best_pd`` checkpoint selection, global-lambda selection, and main tables;
* a closed-interval threshold sweep is descriptive only.  It includes
  ``threshold=1.0``, which is the legal empty-prediction endpoint under the
  frozen ``probability > threshold`` comparison.

The formal dataset identity comes exclusively from
``experiments.three_dataset_v2_protocol``.  In particular, SIRST3 cannot be
passed through this entry point.  The existing four-dataset module is reused
only for its frozen metric/model implementation; none of its dataset matrix,
manifests, paths, or request generation is imported here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import four_dataset_evaluation_protocol_v1 as metric_core  # noqa: E402


SCHEMA = "sctransnet_three_dataset_v2_evaluation_v1"
TRAINING_RUN_SCHEMA = "sctransnet_three_dataset_seed42_global_tss_v2/v1"
FIXED_THRESHOLD = 0.5
UPPER_EMPTY_THRESHOLD = 1.0
TRAINING_SEED = 42
MATCH_RADIUS = 3.0
TINY_AREA = 9
PAD_MULTIPLE = 32
CHECKPOINT_ROLES = ("best_miou", "best_pd")
METHODS = ("original", "final")
TSS_CANDIDATES = (0.0025, 0.005, 0.01)
FA_BUDGETS = (0.5e-6, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
CHECKPOINT_FILENAMES = {
    "best_miou": "best_miou.pth.tar",
    "best_pd": "best_pd.pth.tar",
}
RUN_DATA_PROTOCOL_FIELD = "three_dataset_v2_data_protocol"
REQUIRED_CHECKPOINT_METRICS = (
    "test_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)

# Repository sources that directly define the evaluator, its threshold sweep,
# its frozen metric implementation, or its inference graph.  Keep repository-
# relative names as the public keys so the emitted map is location-independent.
_EVALUATOR_NON_MODEL_SOURCES = (
    "experiments/evaluate_three_dataset_v2.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/four_dataset_evaluation_protocol_v1.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/train_tpd_pilot.py",
    "experiments/four_dataset_models_seed42_v1.py",
)

# Keep the evaluator bound to the protocol's single normalization source.
# The per-dataset copies prevent accidental mutation of that frozen mapping.
NORMALIZATION = {
    dataset: dict(data_protocol.LEGACY_NORMALIZATION[dataset])
    for dataset in data_protocol.DATASETS
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Match the training engine's protocol-payload hash exactly."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_source_paths() -> dict[str, Path]:
    model_root = REPO_ROOT / "model"
    paths = {
        path.relative_to(REPO_ROOT).as_posix(): path.resolve()
        for path in sorted(model_root.rglob("*.py"))
    }
    _require(bool(paths), "repository model source set is empty")
    return dict(sorted(paths.items()))


def evaluator_source_sha256() -> dict[str, str]:
    """Hash every in-repository implementation used by formal evaluation."""

    paths = {
        relative: (REPO_ROOT / relative).resolve()
        for relative in _EVALUATOR_NON_MODEL_SOURCES
    }
    for relative, path in _model_source_paths().items():
        _require(relative not in paths, f"duplicate evaluator source: {relative}")
        paths[relative] = path
    return {
        relative: _file_sha256(path)
        for relative, path in sorted(paths.items())
    }


def _training_runtime_source_paths() -> dict[str, Path]:
    """Sources whose training-time bytes must still define inference."""

    paths: dict[str, Path] = {
        "model_builder": (
            REPO_ROOT / "experiments" / "four_dataset_models_seed42_v1.py"
        ).resolve(),
        "training_metrics_and_schedule": (
            REPO_ROOT / "experiments" / "train_tpd_pilot.py"
        ).resolve(),
    }
    for relative, path in _model_source_paths().items():
        paths[f"architecture::{relative}"] = path
    return dict(sorted(paths.items()))


def _validate_training_runtime_sources(
    run_protocol: Mapping[str, Any],
) -> dict[str, str]:
    """Reject inference under source bytes unlike those frozen at training."""

    frozen = run_protocol.get("runtime_sources")
    _require(isinstance(frozen, Mapping), "run protocol lacks runtime_sources")
    expected = _training_runtime_source_paths()
    expected_architecture = {
        key for key in expected if key.startswith("architecture::")
    }
    frozen_architecture = {
        key
        for key in frozen
        if isinstance(key, str) and key.startswith("architecture::")
    }
    _require(
        frozen_architecture == expected_architecture,
        "run protocol architecture source set differs from current model tree",
    )
    verified: dict[str, str] = {}
    for key, expected_path in expected.items():
        entry = frozen.get(key)
        _require(
            isinstance(entry, Mapping),
            f"run protocol runtime source is missing or malformed: {key}",
        )
        frozen_path = entry.get("path")
        _require(
            isinstance(frozen_path, str) and bool(frozen_path),
            f"run protocol runtime source path is malformed: {key}",
        )
        _require(
            Path(frozen_path).resolve(strict=True) == expected_path,
            f"run protocol runtime source path differs: {key}",
        )
        current_sha256 = _file_sha256(expected_path)
        _require(
            entry.get("sha256") == current_sha256,
            f"run protocol runtime source SHA differs: {key}",
        )
        verified[key] = current_sha256
    return dict(sorted(verified.items()))


def _validate_protocol_payload_and_summary_hash(
    summary: Mapping[str, Any],
    run_protocol: Mapping[str, Any],
) -> str:
    """Validate the trainer's self-excluding canonical protocol hash."""

    declared = run_protocol.get("protocol_sha256")
    _require(
        isinstance(declared, str) and len(declared) == 64,
        "protocol payload lacks a valid protocol_sha256",
    )
    unsigned_payload = dict(run_protocol)
    del unsigned_payload["protocol_sha256"]
    computed = _canonical_sha256(unsigned_payload)
    _require(
        declared == computed,
        "protocol payload protocol_sha256 differs from canonical payload hash",
    )
    _require(
        summary.get("protocol_sha256") == computed,
        "summary protocol_sha256 differs from protocol payload",
    )
    return computed


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    ready = float(value)
    if not math.isfinite(ready):
        raise ValueError(f"{label} must be finite")
    return ready


def _tiny_pd_for_selection(point: Mapping[str, Any]) -> float:
    value = point.get("tiny_pd")
    if value is None:
        return -1.0
    ready = float(value)
    return ready if math.isfinite(ready) else -1.0


def _point_is_empty(point: Mapping[str, Any]) -> bool:
    count = point.get("predicted_object_count")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("point.predicted_object_count must be an integer")
    if int(count) < 0:
        raise ValueError("point.predicted_object_count must be non-negative")
    return int(count) == 0


def _with_exact_unmatched_pixels(point: Mapping[str, Any]) -> dict[str, Any]:
    """Add the integer Fa numerator required by the lambda selector.

    The frozen metric core historically emitted only ``fa`` and
    ``valid_pixel_count`` even though its accumulator internally keeps the
    unmatched-pixel numerator.  Their product is an integer-valued quantity;
    recover it with a checked round instead of making downstream code compare
    floating Fa rates from differently sized datasets.
    """

    ready = dict(point)
    fa = _finite_float(ready.get("fa"), "point.fa")
    valid = ready.get("valid_pixel_count")
    if isinstance(valid, bool) or not isinstance(valid, (int, np.integer)):
        raise TypeError("point.valid_pixel_count must be an integer")
    _require(int(valid) > 0, "point.valid_pixel_count must be positive")
    raw = fa * int(valid)
    unmatched = int(round(raw))
    _require(
        math.isclose(raw, unmatched, rel_tol=0.0, abs_tol=1e-6),
        "Fa and valid_pixel_count do not encode an integer pixel numerator",
    )
    ready["unmatched_predicted_pixels"] = unmatched
    return ready


def _empty_endpoint(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        point
        for point in points
        if math.isclose(
            _finite_float(point.get("threshold"), "point.threshold"),
            UPPER_EMPTY_THRESHOLD,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ]
    _require(
        len(candidates) == 1,
        "descriptive sweep must contain exactly one threshold=1.0 point",
    )
    endpoint = dict(candidates[0])
    _require(
        _point_is_empty(endpoint),
        "threshold=1.0 must be an empty-prediction point",
    )
    _require(
        _finite_float(endpoint.get("pd"), "empty endpoint pd") == 0.0,
        "threshold=1.0 empty endpoint must have Pd=0",
    )
    _require(
        _finite_float(endpoint.get("fa"), "empty endpoint fa") == 0.0,
        "threshold=1.0 empty endpoint must have Fa=0",
    )
    return endpoint


def _best_nonempty_point(
    points: Sequence[Mapping[str, Any]],
    budget: float,
) -> dict[str, Any] | None:
    feasible = [
        point
        for point in points
        if not _point_is_empty(point)
        and _finite_float(point.get("fa"), "point.fa") <= budget
    ]
    if not feasible:
        return None
    selected = max(
        feasible,
        key=lambda point: (
            _finite_float(point.get("pd"), "point.pd"),
            -_finite_float(point.get("fa"), "point.fa"),
            _tiny_pd_for_selection(point),
            _finite_float(point.get("miou"), "point.miou"),
            _finite_float(point.get("niou"), "point.niou"),
            -abs(
                _finite_float(point.get("threshold"), "point.threshold")
                - FIXED_THRESHOLD
            ),
        ),
    )
    return dict(selected)


def pd_at_fa_budget(
    points: Sequence[Mapping[str, Any]],
    budget: float,
) -> dict[str, Any]:
    """Select one descriptive Pd--Fa point with explicit empty semantics.

    A feasible non-empty registered-grid point always takes precedence over
    the empty endpoint.  Only when no non-empty point satisfies the budget is
    the legal threshold-1.0 endpoint returned.  Therefore ``reachable`` is
    never overloaded to mean either "mathematically feasible" or "non-empty
    registered-grid point exists".
    """

    _require(bool(points), "Pd--Fa point sequence must not be empty")
    ready_budget = _finite_float(budget, "Fa budget")
    _require(ready_budget >= 0.0, "Fa budget must be non-negative")
    endpoint = _empty_endpoint(points)
    best_nonempty = _best_nonempty_point(points, ready_budget)
    if best_nonempty is None:
        selected = endpoint
        selected_is_empty = True
    else:
        selected = best_nonempty
        selected_is_empty = False
    return {
        "budget": ready_budget,
        "pd_at_fa_budget": _finite_float(selected.get("pd"), "selected pd"),
        "fa_at_selected_point": _finite_float(
            selected.get("fa"), "selected fa"
        ),
        "selected_threshold": _finite_float(
            selected.get("threshold"), "selected threshold"
        ),
        "selected_point_is_empty": selected_is_empty,
        "registered_grid_nonempty_feasible": best_nonempty is not None,
        "best_nonempty_point": best_nonempty,
    }


def pd_at_fa_budgets(
    points: Sequence[Mapping[str, Any]],
    budgets: Sequence[float] = FA_BUDGETS,
) -> dict[str, dict[str, Any]]:
    ready: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        numeric = _finite_float(budget, "Fa budget")
        key = f"{numeric:.10g}"
        _require(key not in ready, f"duplicate Fa budget: {numeric}")
        ready[key] = pd_at_fa_budget(points, numeric)
    return ready


def threshold_role_contract() -> dict[str, Any]:
    """Return the machine-readable separation of formal and sweep roles."""

    return {
        "checkpoint_selection_threshold": FIXED_THRESHOLD,
        "global_lambda_selection_threshold": FIXED_THRESHOLD,
        "main_table_threshold": FIXED_THRESHOLD,
        "descriptive_sweep_only": True,
        "descriptive_sweep_contains_threshold_1_0": True,
        "threshold_1_0_semantics": "empty_prediction_pd0_fa0",
        "sweep_reselects_checkpoint": False,
        "sweep_reselects_global_lambda": False,
    }


def evaluate_probability_arrays(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    losses: Sequence[float],
    *,
    sweep_thresholds: Sequence[float] | None = None,
    fa_budgets: Sequence[float] = FA_BUDGETS,
) -> dict[str, Any]:
    """Evaluate one inference cache once for fixed and descriptive outputs."""

    fixed_points = [
        _with_exact_unmatched_pixels(point)
        for point in metric_core.strict_metric_points(
            probabilities,
            targets,
            losses,
            [FIXED_THRESHOLD],
        )
    ]
    _require(len(fixed_points) == 1, "fixed evaluation produced !=1 point")
    if sweep_thresholds is None:
        thresholds, provenance = metric_core.closed_interval_thresholds(
            probabilities
        )
    else:
        thresholds = sorted({float(value) for value in sweep_thresholds})
        provenance = {
            "provided_thresholds": True,
            "closed_probability_interval": (
                UPPER_EMPTY_THRESHOLD in thresholds
            ),
        }
    _require(
        any(value == FIXED_THRESHOLD for value in thresholds),
        "descriptive sweep lacks threshold=0.5",
    )
    _require(
        any(value == UPPER_EMPTY_THRESHOLD for value in thresholds),
        "descriptive sweep lacks threshold=1.0",
    )
    points = [
        _with_exact_unmatched_pixels(point)
        for point in metric_core.strict_metric_points(
            probabilities,
            targets,
            losses,
            thresholds,
        )
    ]
    sweep_fixed = [
        point for point in points if float(point["threshold"]) == FIXED_THRESHOLD
    ]
    _require(
        len(sweep_fixed) == 1,
        "descriptive sweep must contain exactly one threshold=0.5 point",
    )
    _require(
        sweep_fixed[0] == fixed_points[0],
        "fixed-0.5 and sweep-0.5 metrics differ on the same cache",
    )
    budgets = pd_at_fa_budgets(points, fa_budgets)
    return {
        "threshold_roles": threshold_role_contract(),
        "fixed_threshold_0_5": fixed_points[0],
        "descriptive_pd_fa": {
            "selection_effect": "none",
            "threshold_provenance": provenance,
            "best_points_under_fa_budget": budgets,
            "pareto_frontier": metric_core.pareto_frontier(points),
            "points": points,
        },
    }


@dataclass(frozen=True)
class EvaluationRequest:
    dataset: str
    method: str
    checkpoint_role: str
    requested_tss_weight: float | None = None

    def validate(self) -> None:
        _require(
            self.dataset in data_protocol.DATASETS,
            f"dataset must be one of {data_protocol.DATASETS}",
        )
        _require(self.method in METHODS, f"method must be one of {METHODS}")
        _require(
            self.checkpoint_role in CHECKPOINT_ROLES,
            f"checkpoint_role must be one of {CHECKPOINT_ROLES}",
        )
        if self.method == "original":
            _require(
                self.requested_tss_weight is None,
                "Original must not have a requested TSS weight",
            )
        else:
            _require(
                self.requested_tss_weight in TSS_CANDIDATES,
                f"Final TSS weight must be one of {TSS_CANDIDATES}",
            )


def _extract_hw(sizes: Any) -> tuple[int, int]:
    if isinstance(sizes, torch.Tensor):
        value = sizes.detach().cpu()
        if value.ndim == 2:
            return int(value[0, 0]), int(value[0, 1])
        if value.ndim == 1 and value.numel() == 2:
            return int(value[0]), int(value[1])
    if isinstance(sizes, (tuple, list)) and len(sizes) == 2:
        values: list[int] = []
        for item in sizes:
            if isinstance(item, torch.Tensor):
                item = item.reshape(-1)[0].item()
            values.append(int(item))
        return values[0], values[1]
    raise TypeError(f"unsupported collated size value: {type(sizes)!r}")


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_bottom_right(
    array: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    height, width = array.shape
    _require(
        height <= target_height and width <= target_width,
        "padding target is smaller than input",
    )
    return np.pad(
        array,
        ((0, target_height - height), (0, target_width - width)),
        mode="constant",
    )


class ThreeDatasetTestDataset(Dataset):
    """Full-image test loader driven only by the three-dataset protocol."""

    def __init__(
        self,
        dataset_root: Path,
        dataset_name: str,
        protocol_manifest: Path | Mapping[str, Any],
    ) -> None:
        super().__init__()
        _require(
            dataset_name in data_protocol.DATASETS,
            f"unsupported formal dataset: {dataset_name!r}",
        )
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.dataset_name = dataset_name
        self.sample_ids = data_protocol.load_frozen_index(
            self.dataset_root,
            self.dataset_name,
            "test",
            protocol_manifest,
        )
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = dict(NORMALIZATION[self.dataset_name])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        sample_id = self.sample_ids[index]
        resolved = data_protocol.resolve_sample(
            self.dataset_root,
            self.dataset_name,
            sample_id,
            split="test",
            known_ids=self._known_ids,
        )
        with Image.open(resolved.image_path) as image:
            image_array = np.asarray(image.convert("I"), dtype=np.float32)
        with Image.open(resolved.mask_path) as mask:
            mask_array = np.asarray(mask, dtype=np.float32)
        if mask_array.ndim > 2:
            mask_array = mask_array[:, :, 0]
        _require(
            image_array.ndim == 2 and mask_array.ndim == 2,
            f"image/mask must be 2D: {self.dataset_name}::{sample_id}",
        )
        _require(
            image_array.shape == mask_array.shape,
            "image/mask dimensions differ after correction resolution: "
            f"{self.dataset_name}::{sample_id}",
        )
        _require(
            np.isfinite(image_array).all() and np.isfinite(mask_array).all(),
            f"non-finite pixels: {self.dataset_name}::{sample_id}",
        )
        height, width = image_array.shape
        image_array = (
            image_array - np.float32(self.normalization["mean"])
        ) / np.float32(self.normalization["std"])
        mask_array = mask_array / np.float32(255.0)
        padded_height = _next_multiple(height, PAD_MULTIPLE)
        padded_width = _next_multiple(width, PAD_MULTIPLE)
        image_array = _pad_bottom_right(
            image_array, padded_height, padded_width
        )
        mask_array = _pad_bottom_right(
            mask_array, padded_height, padded_width
        )
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_array[np.newaxis, :], dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_array[np.newaxis, :], dtype=np.float32)
        )
        return image_tensor, mask_tensor, (height, width), sample_id


def _final_prediction(outputs: Any) -> torch.Tensor:
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
        _require(
            int(images.shape[0]) == 1 and int(masks.shape[0]) == 1,
            "formal evaluator requires batch_size=1",
        )
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height, width = _extract_hw(sizes)
        prediction = _final_prediction(model(images))[:, :, :height, :width]
        target = masks[:, :, :height, :width]
        _require(prediction.shape == target.shape, "prediction/target differ")
        _require(
            bool(torch.isfinite(prediction).all()),
            "model prediction contains non-finite values",
        )
        loss = criterion(prediction.float(), target.float())
        _require(math.isfinite(float(loss.item())), "test loss is non-finite")
        probability = prediction[0, 0].float().cpu().numpy()
        target_array = target[0, 0].float().cpu().numpy()
        _require(
            np.isfinite(probability).all() and np.isfinite(target_array).all(),
            "prediction/target array contains non-finite values",
        )
        probabilities.append(probability)
        targets.append(target_array)
        losses.append(float(loss.item()))
        _require(
            isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
            "test loader must collate one sample ID",
        )
        identifiers.append(str(sample_ids[0]))
    _require(
        bool(probabilities) and len(probabilities) == len(loader.dataset),
        "prediction count differs from test dataset",
    )
    _require(
        len(identifiers) == len(set(identifiers)),
        "test loader yielded duplicate sample IDs",
    )
    return probabilities, targets, losses, identifiers


def _json_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _observed_requested_weight(
    checkpoint: Mapping[str, Any],
    run_protocol: Mapping[str, Any],
) -> float | None:
    values: list[float] = []
    for container in (checkpoint, run_protocol, run_protocol.get("training")):
        if not isinstance(container, Mapping):
            continue
        for key in (
            "requested_tss_weight",
            "tss_requested_weight",
            "tss_weight",
            "lambda_req",
        ):
            if key in container and container[key] is not None:
                values.append(_finite_float(container[key], key))
    if not values:
        return None
    first = values[0]
    _require(
        all(value == first for value in values[1:]),
        "run/checkpoint requested TSS weights disagree",
    )
    return first


def _validate_run_identity(
    request: EvaluationRequest,
    summary: Mapping[str, Any],
    run_protocol: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Reject checkpoints not trained under the new three-dataset lock."""

    for container_name, container in (
        ("summary", summary),
        ("run protocol", run_protocol),
    ):
        _require(
            container.get("schema") == TRAINING_RUN_SCHEMA,
            f"{container_name} schema differs",
        )
        for field, expected in (
            ("dataset", request.dataset),
            ("method", request.method),
        ):
            _require(
                container.get(field) == expected,
                f"{container_name} {field} differs",
            )
    _require(
        summary.get("seed") == TRAINING_SEED,
        "summary training seed differs",
    )
    _require(
        run_protocol.get("training_seed") == TRAINING_SEED,
        "run protocol training seed differs",
    )
    _require(summary.get("epochs") == 1000, "summary epochs differ")
    for field, expected in (
        ("epochs", 1000),
        ("begin_test", 10),
        ("eval_every", 10),
        ("smoke", False),
    ):
        _require(
            run_protocol.get(field) == expected,
            f"run protocol {field} differs",
        )
    expected_split_counts = {
        split: int(
            data_protocol.EXPECTED_SPLITS[request.dataset][split]["count"]
        )
        for split in data_protocol.SPLITS
    }
    _require(
        run_protocol.get("dataset_counts") == expected_split_counts,
        "run protocol dataset_counts differ from frozen img_idx",
    )
    _require(run_protocol.get("test_selected") is True, "run is not test-selected")
    _require(
        run_protocol.get("selection_is_optimistic") is True,
        "run does not disclose optimistic test selection",
    )
    _require(
        run_protocol.get("checkpoint_roles") == list(CHECKPOINT_ROLES),
        "run checkpoint roles differ",
    )
    metrics = run_protocol.get("metrics")
    _require(isinstance(metrics, Mapping), "run protocol lacks metrics")
    _require(
        metrics.get("threshold") == FIXED_THRESHOLD,
        "run checkpoint-selection threshold is not 0.5",
    )
    binding = run_protocol.get(RUN_DATA_PROTOCOL_FIELD)
    _require(
        isinstance(binding, Mapping),
        f"run protocol lacks {RUN_DATA_PROTOCOL_FIELD}",
    )
    expected_binding = {
        "module": "experiments.three_dataset_v2_protocol",
        "schema": data_protocol.SCHEMA,
        "manifest_id": data_protocol.MANIFEST_ID,
        "manifest_sha256": _file_sha256(manifest_path),
        "datasets": list(data_protocol.DATASETS),
        "sirst3_in_formal_matrix": False,
    }
    for field, expected in expected_binding.items():
        _require(
            binding.get(field) == expected,
            f"run data-protocol binding differs for {field}",
        )
    _require(
        manifest.get("dataset_order") == list(data_protocol.DATASETS),
        "loaded data manifest dataset order differs",
    )
    # Model-builder capability metadata may truthfully list historical
    # datasets supported by the shared implementation.  Formal matrix
    # identity is determined only by the structured V2 data binding above;
    # rejecting an unrelated metadata string would make valid V2 runs
    # unreadable without strengthening the data check.


def load_checkpoint(
    request: EvaluationRequest,
    run_dir: Path,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request.validate()
    run_dir = Path(run_dir).resolve(strict=True)
    _require("SIRST3" not in run_dir.parts, "formal run path contains SIRST3")
    summary_path = run_dir / "summary.json"
    protocol_path = run_dir / "protocol.json"
    summary = _json_object(summary_path)
    run_protocol = _json_object(protocol_path)
    _require(summary.get("status") == "complete", "run is not complete")
    _validate_run_identity(
        request,
        summary,
        run_protocol,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    protocol_payload_sha256 = _validate_protocol_payload_and_summary_hash(
        summary,
        run_protocol,
    )
    verified_runtime_sources = _validate_training_runtime_sources(run_protocol)
    checkpoint_path = (
        run_dir / "checkpoints" / CHECKPOINT_FILENAMES[request.checkpoint_role]
    )
    checkpoint_digest = _file_sha256(checkpoint_path)
    bindings = summary.get("checkpoints")
    _require(isinstance(bindings, Mapping), "summary lacks checkpoints")
    binding = bindings.get(request.checkpoint_role)
    _require(isinstance(binding, Mapping), "summary lacks checkpoint role")
    _require(
        binding.get("sha256") == checkpoint_digest,
        "checkpoint SHA differs from summary",
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "checkpoint payload must be a dictionary")
    for field, expected in (
        ("schema", TRAINING_RUN_SCHEMA),
        ("dataset", request.dataset),
        ("method", request.method),
        ("seed", TRAINING_SEED),
        ("checkpoint_role", request.checkpoint_role),
        ("test_selected", True),
        ("selection_is_optimistic", True),
    ):
        _require(
            payload.get(field) == expected,
            f"checkpoint {field} differs: {payload.get(field)!r} != {expected!r}",
        )
    _require(
        payload.get("protocol_sha256") == protocol_payload_sha256,
        "checkpoint protocol_sha256 differs from protocol payload",
    )
    epoch = payload.get("epoch")
    _require(
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and 10 <= epoch <= 1000
        and epoch % 10 == 0,
        "checkpoint epoch is not a frozen candidate epoch",
    )
    summary_selection = summary.get(request.checkpoint_role)
    _require(
        isinstance(summary_selection, Mapping),
        "summary lacks selected-role metadata",
    )
    _require(
        summary_selection.get("epoch") == epoch,
        "checkpoint epoch differs from summary selected role",
    )
    _require(
        payload.get("selection_source") == f"test_{request.dataset}",
        "checkpoint selection_source differs",
    )
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping) and state, "checkpoint lacks state_dict")
    observed_weight = _observed_requested_weight(payload, run_protocol)
    if request.method == "final":
        _require(
            observed_weight == request.requested_tss_weight,
            "Final checkpoint/run protocol TSS weight differs from request",
        )
    else:
        _require(
            observed_weight in (None, 0.0),
            "Original run unexpectedly registers a positive TSS weight",
        )
    return payload, {
        "run_dir": str(run_dir),
        "summary": {
            "path": str(summary_path),
            "sha256": _file_sha256(summary_path),
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": _file_sha256(protocol_path),
            "payload_sha256": protocol_payload_sha256,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
            "epoch": int(payload["epoch"]),
            "role": request.checkpoint_role,
        },
        "requested_tss_weight": observed_weight,
        "training_runtime_sources": {
            "validated": True,
            "source_sha256": verified_runtime_sources,
        },
    }


def build_inference_model(
    request: EvaluationRequest,
    training_state_dict: Mapping[str, torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    request.validate()
    from experiments import four_dataset_models_seed42_v1 as models

    if request.method == "final":
        model, metadata = (
            models.build_final_inference_model_from_training_state_dict(
                training_state_dict,
                dataset_name=request.dataset,
                seed=TRAINING_SEED,
            )
        )
        _require(
            metadata.get("target_survival_registered") is False,
            "Final inference graph still registers TSS",
        )
        _require(metadata.get("strict_load") is True, "Final load not strict")
        return model, dict(metadata)
    model, metadata = models.build_paper_model(
        "original",
        request.dataset,
        seed=TRAINING_SEED,
        training=False,
    )
    incompatible = model.load_state_dict(training_state_dict, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "Original strict checkpoint load returned incompatible keys",
    )
    model.eval()
    model.mode = "test"
    ready = dict(metadata)
    ready.update({"strict_load": True, "target_survival_registered": False})
    return model, ready


def _checkpoint_metric_audit(
    checkpoint_payload: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    raw = checkpoint_payload.get("test_metrics")
    _require(isinstance(raw, Mapping), "checkpoint lacks test_metrics")
    missing = [key for key in REQUIRED_CHECKPOINT_METRICS if key not in raw]
    _require(
        not missing,
        f"checkpoint test_metrics lacks required fields: {missing}",
    )
    count_keys = {
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
    compared: dict[str, float] = {}
    for key, expected in raw.items():
        if key not in fixed:
            continue
        observed = fixed[key]
        if key in count_keys:
            _require(observed == expected, f"checkpoint count differs: {key}")
            compared[key] = 0.0
            continue
        if expected is None:
            _require(observed is None, f"checkpoint null differs: {key}")
            compared[key] = 0.0
            continue
        tolerance = (
            1e-4
            if key
            in {
                "miou",
                "niou",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
            }
            else 1e-7
            if key == "test_loss"
            else 1e-15
        )
        _require(
            math.isclose(
                float(observed),
                float(expected),
                rel_tol=0.0,
                abs_tol=tolerance,
            ),
            f"checkpoint metric differs: {key}",
        )
        compared[key] = tolerance
    return {"passed": True, "absolute_tolerances": compared}


def evaluate_run(
    request: EvaluationRequest,
    *,
    run_dir: Path,
    dataset_root: Path,
    data_protocol_manifest: Path,
    device_name: str,
    workers: int,
    fa_budgets: Sequence[float] = FA_BUDGETS,
) -> dict[str, Any]:
    request.validate()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest_path = Path(data_protocol_manifest).resolve(strict=True)
    manifest = data_protocol.load_protocol_manifest(
        manifest_path,
        dataset_root=dataset_root,
    )
    checkpoint_payload, checkpoint_binding = load_checkpoint(
        request,
        run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    model, model_metadata = build_inference_model(
        request,
        checkpoint_payload["state_dict"],
    )
    model.to(device)
    dataset = ThreeDatasetTestDataset(
        dataset_root,
        request.dataset,
        manifest_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    probabilities, targets, losses, identifiers = collect_predictions(
        model, loader, device
    )
    _require(
        identifiers == list(dataset.sample_ids),
        "inference order differs from the frozen img_idx/test order",
    )
    evaluated = evaluate_probability_arrays(
        probabilities,
        targets,
        losses,
        fa_budgets=fa_budgets,
    )
    fixed = evaluated["fixed_threshold_0_5"]
    checkpoint_audit = _checkpoint_metric_audit(checkpoint_payload, fixed)
    inference_order_newline_sha256 = hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()
    expected_test_split = data_protocol.EXPECTED_SPLITS[request.dataset][
        "test"
    ]
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": request.dataset,
        "method": request.method,
        "requested_tss_weight": request.requested_tss_weight,
        "checkpoint_role": request.checkpoint_role,
        "seed": TRAINING_SEED,
        "test_selected": True,
        "selection_is_optimistic": True,
        **evaluated,
        "checkpoint_binding": checkpoint_binding,
        "checkpoint_metric_audit": checkpoint_audit,
        "model": model_metadata,
        "data": {
            "protocol_module": "experiments.three_dataset_v2_protocol",
            "dataset_root": str(Path(dataset_root).resolve()),
            "protocol_manifest": {
                "path": str(manifest_path),
                "sha256": _file_sha256(manifest_path),
                "schema": manifest.get("schema"),
                "manifest_id": manifest.get("manifest_id"),
            },
            "split": "img_idx/test",
            "test_count": len(identifiers),
            "img_idx_test_sha256": expected_test_split["file_sha256"],
            "img_idx_test_ordered_ids_sha256": expected_test_split[
                "ordered_ids_sha256"
            ],
            "inference_order_newline_sha256": (
                inference_order_newline_sha256
            ),
            "normalization": NORMALIZATION[request.dataset],
            "sirst3_in_formal_matrix": False,
        },
        "metric_protocol": {
            "implementation": "experiments.train_tpd_pilot.ValidationMetrics",
            "connectivity": 8,
            "matching": "one_to_one_max_cardinality_min_distance",
            "centroid_radius_comparison": "distance < 3",
            "match_radius": MATCH_RADIUS,
            "tiny_area": TINY_AREA,
            "prediction_comparison": "probability > threshold",
            "score_dtype": "float32",
        },
        "source_sha256": evaluator_source_sha256(),
        "no_fabricated_results": True,
        "stability_claim_supported": False,
    }
    del model, loader, probabilities, targets, losses
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--checkpoint-role", choices=CHECKPOINT_ROLES, required=True
    )
    parser.add_argument("--requested-tss-weight", type=float)
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
    try:
        EvaluationRequest(
            dataset=args.dataset,
            method=args.method,
            checkpoint_role=args.checkpoint_role,
            requested_tss_weight=args.requested_tss_weight,
        ).validate()
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    request = EvaluationRequest(
        dataset=args.dataset,
        method=args.method,
        checkpoint_role=args.checkpoint_role,
        requested_tss_weight=args.requested_tss_weight,
    )
    output = evaluate_run(
        request,
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
    _atomic_write_json(destination, output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(destination.resolve()),
                "sha256": _file_sha256(destination.resolve()),
                "fixed_threshold": FIXED_THRESHOLD,
                "descriptive_sweep_only": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
