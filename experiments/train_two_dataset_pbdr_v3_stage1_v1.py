#!/usr/bin/env python3
"""Cross-dataset PBDR-V3 Stage-1 trainer built on the frozen NUAA engine.

The completed NUAA source is imported as an execution engine but never edited.
This version supplies dataset-explicit model/data bindings, a fixed 0.5 metric
threshold, and a strict zero-positive-margin role gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import pbdr_v3_zero_margin_role_gate as zero_gate
from experiments import three_dataset_pbdr_v3_models_seed42_v1 as core_models
from experiments import train_nuaa_pbdr_v3_stage1_v1 as engine
from experiments import two_dataset_pbdr_v3_models_seed42_v1 as registry


SCHEMA = "sctransnet_two_dataset_pbdr_v3_stage1_v1/v1"
DATASETS = registry.DATASETS
PARENT_ROLES = registry.PARENT_ROLES
RECIPES = ("core",)
TRAINING_SEED = 42
FIXED_THRESHOLD = 0.5
THRESHOLDS = (FIXED_THRESHOLD,)
PROTOCOL_DOCUMENT = REPO_ROOT / "experiments/PBDR_V3_CROSS_DATASET_PROTOCOL.md"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_PROTOCOL_MANIFEST = (
    REPO_ROOT / "results/three_dataset_v2/manifests/three_dataset_v2_protocol.json"
)
FORMAL_PROTOCOL_MANIFEST_SHA256 = (
    "00edc6413dead3678f8b4c162c74ea7d8602f55ff413cb20ad1664587380319f"
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/two_dataset_pbdr_v3_stage1_v1"
GPU_UUIDS = {
    "NUDT-SIRST": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "IRSTD-1K": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
}
SMOKE_MIN_VAL_IMAGES = {
    # Frozen split prefixes through the first sample containing a tiny target.
    # This keeps tiny-Pd finite in the complete role key used by smoke.
    "NUDT-SIRST": 5,
    "IRSTD-1K": 6,
}
EXTRA_RUNTIME_SOURCE_PATHS = (
    "experiments/PBDR_V3_CROSS_DATASET_PROTOCOL.md",
    "experiments/pbdr_v3_loss.py",
    "experiments/pbdr_v3_non_regression_gate.py",
    "experiments/pbdr_v3_zero_margin_role_gate.py",
    "experiments/train_two_dataset_pbdr_v3_stage1_v1.py",
    "experiments/evaluate_two_dataset_pbdr_v3_stage1_v1.py",
    "experiments/launch_two_dataset_pbdr_v3_stage1_v1.py",
    "experiments/train_nuaa_pbdr_v3_stage1_v1.py",
    "experiments/evaluate_nuaa_pbdr_v3_stage1_v1.py",
)

_BASE_BUILD_INTERNAL_SPLIT = engine.build_internal_split_manifest
_BASE_METRICS = engine._metrics


class DatasetModelsAdapter:
    """Present the frozen NUAA engine's model API for one explicit dataset."""

    PARENT_ROLES = PARENT_ROLES
    TRAINING_SEED = TRAINING_SEED
    TRAINING_STATE_KEY_COUNT = registry.TRAINING_STATE_KEY_COUNT
    INFERENCE_STATE_KEY_COUNT = registry.INFERENCE_STATE_KEY_COUNT

    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset_name!r}")
        self.DATASET = dataset_name

    @staticmethod
    def file_sha256(path: Path) -> str:
        return registry.file_sha256(path)

    @staticmethod
    def canonical_sha256(value: Any) -> str:
        return registry.canonical_sha256(value)

    @staticmethod
    def tensor_mapping_sha256(value: Mapping[str, Any]) -> str:
        return registry.tensor_mapping_sha256(value)

    @staticmethod
    def configure_stage1(model: Any) -> dict[str, Any]:
        return core_models.configure_stage1(model)

    @staticmethod
    def audit_stage1(model: Any) -> dict[str, Any]:
        return core_models.audit_stage1(model)

    @staticmethod
    def base_state_sha256(model: Any) -> str:
        return core_models.base_state_sha256(model)

    @staticmethod
    def batchnorm_buffer_sha256(model: Any) -> str:
        return core_models.batchnorm_buffer_sha256(model)

    def build_stage1_training_model(
        self,
        parent_role: str,
        *,
        parent_checkpoint: Path | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if parent_checkpoint is not None:
            raise ValueError("cross-dataset Current checkpoint override is forbidden")
        return registry.build_stage1_training_model(self.DATASET, parent_role)

    def load_current_checkpoint(
        self,
        parent_role: str,
        parent_checkpoint: Path | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
        if parent_checkpoint is not None:
            raise ValueError("cross-dataset Current checkpoint override is forbidden")
        return registry.load_current_checkpoint(self.DATASET, parent_role)

    def build_inference_model_from_candidate_state(
        self,
        training_state: Mapping[str, Any],
        *,
        parent_role: str,
        parent_checkpoint: Path | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if parent_checkpoint is not None:
            expected = registry.load_current_checkpoint(self.DATASET, parent_role)[2]
            if Path(parent_checkpoint).resolve(strict=True) != Path(
                str(expected["path"])
            ).resolve(strict=True):
                raise ValueError("candidate parent checkpoint override differs")
        return registry.build_inference_model_from_candidate_state(
            training_state,
            dataset_name=self.DATASET,
            parent_role=parent_role,
        )

    def runtime_source_records(self) -> dict[str, dict[str, Any]]:
        records = dict(registry.runtime_source_records())
        for relative in EXTRA_RUNTIME_SOURCE_PATHS:
            path = REPO_ROOT / relative
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(path)
            records[relative] = {
                "path": str(path.resolve()),
                "sha256": registry.file_sha256(path),
                "bytes": path.stat().st_size,
            }
        return dict(sorted(records.items()))


def selection_key(
    role: str,
    metrics: Mapping[str, Any],
    epoch: int,
) -> tuple[float, ...]:
    ready = zero_gate.CertificationMetrics.from_mapping(metrics)
    return (*zero_gate.role_key(role, ready), -float(epoch))


def canonical_validation_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Use one loss-field name across training, replay, and official evaluation."""

    if not isinstance(value, Mapping):
        raise TypeError("validation metrics must be a mapping")
    ready = dict(value)
    validation_loss = ready.pop("val_loss", None)
    test_loss = ready.get("test_loss")
    if validation_loss is None and test_loss is None:
        raise ValueError("validation metrics lack val_loss/test_loss")
    if validation_loss is not None:
        if test_loss is not None and float(test_loss) != float(validation_loss):
            raise ValueError("val_loss and test_loss differ")
        ready["test_loss"] = validation_loss
    return ready


def cross_dataset_metrics(
    probabilities: Sequence[Any],
    targets: Sequence[Any],
    threshold: float,
) -> dict[str, Any]:
    """Run the frozen metric engine and canonicalize its split-neutral loss."""

    return canonical_validation_metrics(_BASE_METRICS(probabilities, targets, threshold))


def fixed_validation_threshold(
    role: str,
    validation: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    fixed = validation["fixed_0_5"]
    adapter = zero_gate.RoleGateAdapter(role)
    current = adapter.CertificationMetrics.from_mapping(fixed["current"])
    candidate = adapter.CertificationMetrics.from_mapping(fixed["candidate"])
    decision = adapter.certify(current, candidate)
    return FIXED_THRESHOLD, {
        "selection_source": "fixed_protocol_threshold_only",
        "threshold_optimization_performed": False,
        "passed_gate_pool_used": decision.passed,
        "threshold": FIXED_THRESHOLD,
        "metrics": dict(fixed["candidate"]),
        "certification": engine._decision_payload(decision),
    }


def configure_engine(dataset_name: str, parent_role: str) -> DatasetModelsAdapter:
    models = DatasetModelsAdapter(dataset_name)
    role_gate = zero_gate.RoleGateAdapter(parent_role)
    engine.models = models
    engine.gate = role_gate
    engine.SCHEMA = SCHEMA
    engine.TRAINING_SEED = TRAINING_SEED
    engine.THRESHOLDS = THRESHOLDS
    engine.FORMAL_THRESHOLD = FIXED_THRESHOLD
    engine.GPU0_UUID = GPU_UUIDS[dataset_name]
    engine.PROTOCOL_DOCUMENT = PROTOCOL_DOCUMENT
    engine.DEFAULT_RESULTS_ROOT = DEFAULT_RESULTS_ROOT / "runs" / dataset_name
    engine._metrics = cross_dataset_metrics
    engine._selection_key = selection_key
    engine.select_validation_threshold = fixed_validation_threshold

    def build_split(**kwargs: Any) -> dict[str, Any]:
        payload = _BASE_BUILD_INTERNAL_SPLIT(**kwargs)
        payload["schema"] = "sctransnet_two_dataset_pbdr_v3_internal_split_v1/v1"
        payload["dataset"] = dataset_name
        payload["official_test_index_opened"] = False
        payload.pop("split_sha256", None)
        payload["split_sha256"] = models.canonical_sha256(payload)
        return payload

    engine.build_internal_split_manifest = build_split
    return models


def dataset_results_root(results_root: Path, dataset_name: str) -> Path:
    return Path(results_root).resolve() / "runs" / dataset_name


def validate_frozen_data_binding(
    data_root: Path,
    protocol_manifest: Path,
) -> dict[str, str]:
    """Validate the predeclared data paths without opening a test index."""

    root = Path(data_root)
    manifest = Path(protocol_manifest)
    if root.is_symlink() or manifest.is_symlink():
        raise ValueError("frozen data binding cannot use a symlink")
    resolved_root = root.resolve(strict=True)
    resolved_manifest = manifest.resolve(strict=True)
    expected_root = DEFAULT_DATA_ROOT.resolve(strict=True)
    expected_manifest = DEFAULT_PROTOCOL_MANIFEST.resolve(strict=True)
    if resolved_root != expected_root:
        raise ValueError("dataset root differs from frozen formal path")
    if resolved_manifest != expected_manifest:
        raise ValueError("data protocol manifest differs from frozen formal path")
    observed_sha = registry.file_sha256(resolved_manifest)
    if observed_sha != FORMAL_PROTOCOL_MANIFEST_SHA256:
        raise ValueError("data protocol manifest SHA-256 differs")
    return {
        "dataset_root": str(resolved_root),
        "protocol_manifest": str(resolved_manifest),
        "protocol_manifest_sha256": observed_sha,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _load_torch_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"cannot load {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _validate_completed_certification(
    role: str,
    summary_decision: Any,
    candidate_decision: Any,
    certification: Mapping[str, Any],
) -> None:
    if not isinstance(summary_decision, Mapping) or not isinstance(
        candidate_decision, Mapping
    ):
        raise ValueError("completed internal certification is malformed")
    try:
        current = zero_gate.CertificationMetrics.from_mapping(
            summary_decision["current"]
        )
        candidate = zero_gate.CertificationMetrics.from_mapping(
            summary_decision["candidate"]
        )
        expected = zero_gate.certify(role, current, candidate)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot replay completed certification: {error}") from error
    expected_artifact = zero_gate._json_payload(expected)
    expected_core = {
        name: expected_artifact[name]
        for name in ("passed", "selected", "checks", "current", "candidate")
    }
    expected_core["scope"] = "frozen_internal_validation_split"
    if dict(summary_decision) != expected_core:
        raise ValueError("completed summary certification differs")
    if dict(candidate_decision) != expected_core:
        raise ValueError("completed candidate certification differs")
    if dict(certification) != expected_artifact:
        raise ValueError("completed certification artifact differs")


def _completed_run_summary(
    args: argparse.Namespace,
    models: DatasetModelsAdapter,
) -> Path | None:
    """Return a validated completed run without rewriting any final artifact."""

    run_dir = (
        dataset_results_root(args.results_root, args.dataset)
        / ("smoke" if args.smoke else "formal")
        / args.parent_role
        / args.recipe
    )
    summary_path = run_dir / "summary.json"
    if not summary_path.exists() and not summary_path.is_symlink():
        return None
    summary = _read_json_object(summary_path, "completed summary")
    if (
        summary.get("schema") != SCHEMA
        or summary.get("status") != "complete"
        or summary.get("dataset") != args.dataset
        or summary.get("parent_role") != args.parent_role
        or summary.get("recipe") != args.recipe
        or summary.get("official_test_accessed") is not False
    ):
        raise ValueError("completed summary identity/status differs")

    protocol_path = run_dir / "protocol.json"
    split_path = run_dir / "split_manifest.json"
    candidate_path = run_dir / "selected_candidate.pth.tar"
    for path, label in (
        (protocol_path, "completed protocol"),
        (split_path, "completed split"),
        (candidate_path, "completed candidate"),
        (run_dir / "rolling_state.pth.tar", "completed rolling state"),
        (run_dir / "internal_certification.json", "completed certification"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is missing or not a regular file")

    protocol = _read_json_object(protocol_path, "completed protocol")
    split = _read_json_object(split_path, "completed split")
    declared_protocol_sha = protocol.get("protocol_sha256")
    unsigned_protocol = dict(protocol)
    unsigned_protocol.pop("protocol_sha256", None)
    if (
        declared_protocol_sha != models.canonical_sha256(unsigned_protocol)
        or summary.get("protocol_sha256") != declared_protocol_sha
        or summary.get("protocol") != str(protocol_path.resolve(strict=True))
    ):
        raise ValueError("completed protocol hash binding differs")
    declared_split_sha = split.get("split_sha256")
    unsigned_split = dict(split)
    unsigned_split.pop("split_sha256", None)
    if declared_split_sha != models.canonical_sha256(unsigned_split):
        raise ValueError("completed split hash binding differs")

    expected_controls = {
        "mode": "smoke" if args.smoke else "formal",
        "dataset": args.dataset,
        "parent_role": args.parent_role,
        "recipe": args.recipe,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "data_root": str(Path(args.data_root).resolve(strict=True)),
    }
    if any(protocol.get(name) != value for name, value in expected_controls.items()):
        raise ValueError("completed run controls differ from requested controls")
    expected_smoke_limits = {
        "max_train_images": args.max_train_images,
        "max_val_images": args.max_val_images,
    }
    if protocol.get("smoke_limits") != expected_smoke_limits:
        raise ValueError("completed smoke limits differ")

    source_locks = protocol.get("source_locks")
    if not isinstance(source_locks, Mapping):
        raise ValueError("completed protocol lacks source locks")
    if (
        source_locks.get("split_manifest") != declared_split_sha
        or source_locks.get("protocol_document")
        != models.file_sha256(PROTOCOL_DOCUMENT)
        or source_locks.get("runtime_sources") != models.runtime_source_records()
    ):
        raise ValueError("completed run source locks differ")
    selected = summary.get("selected_checkpoint")
    if (
        not isinstance(selected, Mapping)
        or selected.get("path") != str(candidate_path.resolve(strict=True))
        or selected.get("sha256") != models.file_sha256(candidate_path)
    ):
        raise ValueError("completed candidate byte binding differs")
    candidate = _load_torch_mapping(candidate_path, "completed candidate")
    if (
        candidate.get("schema") != SCHEMA
        or candidate.get("parent_role") != args.parent_role
        or candidate.get("recipe") != args.recipe
        or candidate.get("protocol_sha256") != declared_protocol_sha
        or candidate.get("source_locks") != source_locks
        or candidate.get("epoch") != summary.get("selected_epoch")
        or candidate.get("selected_threshold") != FIXED_THRESHOLD
        or candidate.get("threshold_selection") != summary.get("threshold_selection")
    ):
        raise ValueError("completed candidate finalization differs")
    validation = candidate.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("completed candidate validation is malformed")
    try:
        selected_threshold, threshold_selection = fixed_validation_threshold(
            args.parent_role, validation
        )
        expected_selection_key = engine._checkpoint_selection_key(
            args.parent_role, validation, int(candidate["epoch"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot replay completed candidate selection: {error}") from error
    if (
        selected_threshold != FIXED_THRESHOLD
        or threshold_selection != summary.get("threshold_selection")
        or tuple(candidate.get("selection_key", ())) != tuple(expected_selection_key)
    ):
        raise ValueError("completed candidate selection replay differs")
    certification = _read_json_object(
        run_dir / "internal_certification.json", "completed certification"
    )
    _validate_completed_certification(
        args.parent_role,
        summary.get("internal_certification_fixed_0_5"),
        candidate.get("internal_certification_fixed_0_5"),
        certification,
    )
    if summary.get("internal_gate_passed") is not certification.get("passed"):
        raise ValueError("completed internal gate flag differs")
    candidate_state = candidate.get("state_dict")
    if (
        not isinstance(candidate_state, Mapping)
        or len(candidate_state) != models.TRAINING_STATE_KEY_COUNT
        or not all(isinstance(value, torch.Tensor) for value in candidate_state.values())
        or not all(bool(torch.isfinite(value).all()) for value in candidate_state.values())
    ):
        raise ValueError("completed candidate state mapping differs")

    rolling = _load_torch_mapping(
        run_dir / "rolling_state.pth.tar", "completed rolling state"
    )
    rolling_state = rolling.get("state_dict")
    rolling_selected = rolling.get("selected")
    rolling_event = rolling.get("event")
    if (
        rolling.get("schema") != SCHEMA
        or rolling.get("epoch") != args.epochs
        or rolling.get("protocol_sha256") != declared_protocol_sha
        or rolling.get("source_locks") != source_locks
        or tuple(rolling.get("best_key", ())) != tuple(expected_selection_key)
        or not isinstance(rolling_selected, Mapping)
        or rolling_selected.get("epoch") != summary.get("selected_epoch")
        or rolling_selected.get("validation") != validation
        or not isinstance(rolling_event, Mapping)
        or rolling_event.get("epoch") != args.epochs
        or rolling_event.get("selected_epoch") != summary.get("selected_epoch")
        or not isinstance(rolling_state, Mapping)
        or len(rolling_state) != models.TRAINING_STATE_KEY_COUNT
        or not all(isinstance(value, torch.Tensor) for value in rolling_state.values())
        or not all(bool(torch.isfinite(value).all()) for value in rolling_state.values())
        or not isinstance(rolling.get("optimizer"), Mapping)
        or not isinstance(rolling.get("rng_state"), Mapping)
    ):
        raise ValueError("completed rolling-state contract differs")
    return summary_path.resolve(strict=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--parent-role", choices=PARENT_ROLES, required=True)
    parser.add_argument("--recipe", choices=RECIPES, default="core")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=engine.FORMAL_EPOCHS)
    parser.add_argument("--eval-every", type=int, default=engine.FORMAL_EVAL_EVERY)
    parser.add_argument("--batch-size", type=int, default=engine.FORMAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=engine.FORMAL_WORKERS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    validate_frozen_data_binding(args.data_root, args.protocol_manifest)
    models = configure_engine(args.dataset, args.parent_role)
    expected_uuid = GPU_UUIDS[args.dataset]
    if args.expected_gpu_uuid is None:
        args.expected_gpu_uuid = expected_uuid
    if args.expected_gpu_uuid != expected_uuid:
        raise ValueError("dataset GPU UUID binding differs")
    if args.smoke and (
        args.max_val_images is None
        or args.max_val_images < SMOKE_MIN_VAL_IMAGES[args.dataset]
    ):
        raise ValueError(
            "smoke validation prefix must include a frozen tiny-target sample"
        )
    completed = _completed_run_summary(args, models)
    if completed is not None:
        if args.resume == "never":
            raise ValueError("completed run exists but resume=never")
        return completed
    forwarded = argparse.Namespace(**vars(args))
    delattr(forwarded, "dataset")
    forwarded.results_root = dataset_results_root(args.results_root, args.dataset)
    forwarded.parent_checkpoint = None
    return engine.run(forwarded)


def main(argv: Sequence[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DATASETS",
    "DEFAULT_RESULTS_ROOT",
    "DatasetModelsAdapter",
    "FIXED_THRESHOLD",
    "FORMAL_PROTOCOL_MANIFEST_SHA256",
    "GPU_UUIDS",
    "PARENT_ROLES",
    "PROTOCOL_DOCUMENT",
    "SCHEMA",
    "SMOKE_MIN_VAL_IMAGES",
    "THRESHOLDS",
    "configure_engine",
    "dataset_results_root",
    "fixed_validation_threshold",
    "run",
    "selection_key",
    "validate_frozen_data_binding",
]
