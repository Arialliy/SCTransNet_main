#!/usr/bin/env python3
"""Frozen V7-checkpoint counterfactual for V8-MPRS-DCH.

The analysis strict-loads each of the twelve completed V7-DCH checkpoints
into both V7 and V8.  It evaluates the unchanged validation split, registered
thresholds, MPRS correction selectivity, topology, and a preregistered
four-offset shift subset.  It never trains or writes into a formal run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import PIL
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
from skimage import measure, morphology
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import diagnose_tpd_clean_v6_fragmentation as v6_diag  # noqa: E402
from analysis import diagnose_tpd_clean_v7_dch_mechanism as v7_diag  # noqa: E402
from experiments.train_tpd_clean_v7_dch import (  # noqa: E402
    build_clean_v7_dch_model,
)
from experiments.train_tpd_clean_v8_mprs_dch import (  # noqa: E402
    build_clean_v8_mprs_dch_model,
)
from experiments.train_tpd_pilot import (  # noqa: E402
    ValidationSubset,
    final_prediction,
    json_ready,
)
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    TPDCleanV8MPRSDCHBlock,
    clean_v8_mprs_dch_variant_spec,
)


SCHEMA = "sctransnet_tpd_clean_v8_mprs_counterfactual_v2"
JOB_SCHEMA = "sctransnet_tpd_clean_v8_mprs_counterfactual_job_v2"
DATASET = "NUDT-SIRST"
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "analysis/results/tpd_clean_v8_mprs_counterfactual_v2"
)
DEFAULT_RUN_TAG = "formal800_exact_fp32_2x5090_v1"
V8_VARIANT_BY_V7 = {
    "tpd_clean_v7_dch_full": "tpd_clean_v8_mprs_dch_full",
    "tpd_clean_v7_dch_capacity": "tpd_clean_v8_mprs_dch_capacity",
}
REGISTERED_FIXED_THRESHOLD = 0.5
HARD_NEGATIVE_DILATION_RADIUS = 3
SHIFT_SUBSET_COUNT = 16
SHIFT_CROP = 16
SHIFT_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
TARGET_LIFT_FLOOR = 1.0
SHIFT_RATIO_CEILING = 1.10
DIAGNOSTIC_FORWARD_ATOL = 1e-6
DIAGNOSTIC_FORWARD_RTOL = 1e-6
EXPECTED_JOB_COUNT = 12
EXPECTED_BLOCK_NAMES = (
    "mtc.embeddings_1.blocks.0",
    "mtc.embeddings_1.blocks.1",
    "mtc.embeddings_1.blocks.2",
    "mtc.embeddings_1.blocks.3",
    "mtc.embeddings_2.blocks.0",
    "mtc.embeddings_2.blocks.1",
    "mtc.embeddings_2.blocks.2",
)
JOB_BINDING_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_counterfactual_job_binding_v3"
)
ORDERED_IDS_SCHEMA = "sctransnet_tpd_ordered_validation_ids_v1"
FORMAL_INPUT_LABELS = (
    "protocol",
    "split",
    "summary",
    "metrics",
    "checkpoint",
)
EXPECTED_ORDERED_DATA_FINGERPRINTS = {
    "normalization",
    "official_training_data",
    "train_samples",
    "validation_samples",
}
EXPECTED_ORDERED_SPLIT_FINGERPRINTS = {
    "full_train",
    "full_validation",
    "train",
    "validation",
}
EXPECTED_V7_CONTROL_MANIFEST_SHA256 = (
    "5e809ca8af8c18f6dd6783fa07ec015dfe4cd7e0d90e83b59c338528f324c236"
)
V7_CONTROL_MANIFEST_FILENAME = (
    "tpd_clean_v7_dch_finalizer_control_manifest.json"
)
COUNTERFACTUAL_IMPLEMENTATION_FILES = {
    "analyzer": "analysis/analyze_tpd_clean_v8_mprs_mechanism.py",
    "v6_topology_evaluator": (
        "analysis/diagnose_tpd_clean_v6_fragmentation.py"
    ),
    "v7_registry_validator": (
        "analysis/diagnose_tpd_clean_v7_dch_mechanism.py"
    ),
    "v7_model_builder": "experiments/train_tpd_clean_v7_dch.py",
    "v8_model_builder": "experiments/train_tpd_clean_v8_mprs_dch.py",
    "validation_loader": "experiments/train_tpd_pilot.py",
    "v8_protocol": "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    "v8_preflight_amendment": (
        "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md"
    ),
    "dataset_loader": "dataset.py",
    "sctransnet": "model/SCTransNet.py",
    "model_config": "model/Config.py",
    "tpd_baseline": "model/tpd.py",
    "v7_model": "model/tpd_clean_v7_dch.py",
    "v8_model": "model/tpd_clean_v8_mprs_dch.py",
    "metric_utils": "utils.py",
}
IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_with_sha256(path: Path) -> tuple[Dict[str, Any], str]:
    """Read, hash, and parse the same regular-file byte sequence once."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected a regular JSON file: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload, digest


def state_dict_model_sha256(state: Mapping[str, Any]) -> str:
    """Use the exact digest representation used for loaded model states."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state_dict is not a non-empty mapping")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Checkpoint state_dict must map names to tensors")
        contiguous = tensor.detach().cpu().contiguous()
        _require_all_numeric_finite(
            contiguous,
            f"checkpoint state_dict.{name}",
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _require_real(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} is not a finite real number")
    ready = float(value)
    if minimum is not None and ready < minimum:
        raise ValueError(f"{location} is below {minimum}")
    if maximum is not None and ready > maximum:
        raise ValueError(f"{location} is above {maximum}")
    return ready


def _require_integer(
    value: Any,
    location: str,
    *,
    minimum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{location} is not an integer")
    ready = int(value)
    if minimum is not None and ready < minimum:
        raise ValueError(f"{location} is below {minimum}")
    return ready


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value without accepting NaN/Inf spellings."""

    encoded = json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} is not a lowercase SHA256 digest")
    return value


def _require_all_numeric_finite(value: Any, location: str) -> None:
    """Reject any non-finite numeric leaf in a nested audit record."""

    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{location} contains a non-finite tensor")
        return
    if isinstance(value, np.ndarray):
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{location} contains a non-finite array")
        return
    if isinstance(value, np.generic):
        if not math.isfinite(float(value)):
            raise ValueError(f"{location} contains a non-finite scalar")
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{location} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_all_numeric_finite(item, f"{location}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _require_all_numeric_finite(item, f"{location}[{index}]")


def counterfactual_execution_binding() -> Dict[str, Any]:
    sources: Dict[str, Dict[str, str]] = {}
    for name, relative in COUNTERFACTUAL_IMPLEMENTATION_FILES.items():
        path = (REPO_ROOT / relative).resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"Counterfactual implementation source is invalid: {path}"
            )
        sources[name] = {
            "relative_path": relative,
            "sha256": file_sha256(path),
        }
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "skimage": str(skimage.__version__),
        "scipy": str(scipy.__version__),
        "pillow": str(PIL.__version__),
    }
    binding: Dict[str, Any] = {
        "sources": sources,
        "sources_sha256": canonical_json_sha256(sources),
        "runtime": runtime,
        "runtime_sha256": canonical_json_sha256(runtime),
    }
    binding["execution_sha256"] = canonical_json_sha256(binding)
    return binding


EXECUTION_BINDING_AT_PROCESS_START = counterfactual_execution_binding()


def process_execution_binding() -> Dict[str, Any]:
    current = counterfactual_execution_binding()
    if current != EXECUTION_BINDING_AT_PROCESS_START:
        raise RuntimeError(
            "Counterfactual implementation changed after process start"
        )
    return json_ready(EXECUTION_BINDING_AT_PROCESS_START)


def _resolve_validation_sample(
    directory: Path,
    identifier: str,
) -> Path:
    matches = [
        directory / f"{identifier}{extension}"
        for extension in IMAGE_EXTENSIONS
        if (directory / f"{identifier}{extension}").is_file()
        and not (directory / f"{identifier}{extension}").is_symlink()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one regular sample for {identifier!r} under "
            f"{directory}, found {len(matches)}"
        )
    return matches[0]


def current_validation_data_binding(
    dataset_root: Path,
    validation_ids: Sequence[str],
) -> Dict[str, Any]:
    records: list[str] = []
    for identifier in validation_ids:
        image = _resolve_validation_sample(
            Path(dataset_root) / "images",
            identifier,
        )
        mask = _resolve_validation_sample(
            Path(dataset_root) / "masks",
            identifier,
        )
        records.append(
            f"{identifier}|"
            f"image={image.name}:{file_sha256(image)}|"
            f"mask={mask.name}:{file_sha256(mask)}"
        )
    fingerprint = {
        "schema": "sctransnet_tpd_ordered_fingerprint_v1",
        "name": "validation_samples",
        "count": len(records),
        "sha256": canonical_json_sha256(records),
    }
    return {
        "fingerprint": fingerprint,
        "ordered_sample_records_sha256": canonical_json_sha256(records),
        "file_count": 2 * len(records),
    }


def ordered_validation_ids_sha256(identifiers: Sequence[str]) -> str:
    ready = list(identifiers)
    if (
        not ready
        or any(not isinstance(item, str) or not item for item in ready)
        or len(ready) != len(set(ready))
    ):
        raise ValueError(
            "ordered validation IDs must be non-empty, unique strings"
        )
    return canonical_json_sha256(
        {
            "schema": ORDERED_IDS_SCHEMA,
            "ordered_ids": ready,
        }
    )


def _path_is_within(path: Path, root: Path) -> bool:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    return (
        resolved_path == resolved_root
        or resolved_root in resolved_path.parents
    )


def require_analysis_output_separate(
    results_root: Path,
    output_dir: Path,
) -> None:
    if _path_is_within(output_dir, results_root):
        raise ValueError(
            "Counterfactual analysis output must be outside the formal "
            "results root"
        )


def load_json(path: Path) -> Dict[str, Any]:
    return load_json_with_sha256(path)[0]


def write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expected_jobs(
    results_root: Path,
) -> list[Dict[str, Any]]:
    comparison = Path(results_root) / DATASET / "comparison"
    jobs = v7_diag.expected_jobs(
        Path(results_root),
        comparison,
        DEFAULT_RUN_TAG,
    )
    expected_identities = {
        (variant, int(seed), role)
        for variant in V8_VARIANT_BY_V7
        for seed in v7_diag.SEEDS
        for role in v7_diag.CHECKPOINT_SPECS
    }
    observed_identities: set[tuple[str, int, str]] = set()
    for job in jobs:
        identity = (
            str(job.get("variant")),
            int(job.get("seed", -1)),
            str(job.get("role")),
        )
        if identity in observed_identities:
            raise ValueError(f"Duplicate expected job identity: {identity}")
        observed_identities.add(identity)
        variant, seed, role = identity
        if identity not in expected_identities:
            raise ValueError(f"Unexpected counterfactual job: {identity}")
        expected_run_dir = v7_diag.run_directory(
            Path(results_root),
            variant,
            seed,
            DEFAULT_RUN_TAG,
        ).resolve()
        expected_checkpoint = (
            expected_run_dir
            / v7_diag.CHECKPOINT_SPECS[role]["filename"]
        ).resolve()
        expected_registry = v7_diag.diagnostic_output_path(
            comparison,
            variant,
            seed,
            role,
        ).resolve()
        if Path(job["run_dir"]).resolve() != expected_run_dir:
            raise ValueError(f"Unexpected run directory for {identity}")
        if Path(job["checkpoint"]).resolve() != expected_checkpoint:
            raise ValueError(f"Unexpected checkpoint path for {identity}")
        if Path(job["output"]).resolve() != expected_registry:
            raise ValueError(f"Unexpected registry path for {identity}")
    if (
        len(jobs) != EXPECTED_JOB_COUNT
        or observed_identities != expected_identities
    ):
        raise ValueError("Counterfactual job matrix is not the exact 2x2x3 set")
    return jobs


def job_output_path(output_dir: Path, job: Mapping[str, Any]) -> Path:
    return (
        Path(output_dir)
        / "checkpoints"
        / str(job["variant"])
        / f"seed_{int(job['seed'])}"
        / f"{job['role']}.json"
    )


def sealed_v7_registry_binding(job: Mapping[str, Any]) -> Dict[str, Any]:
    registry_path = Path(job["output"]).resolve()
    try:
        comparison_dir = registry_path.parents[3]
    except IndexError as exc:
        raise ValueError("V7 registry path has no comparison root") from exc
    manifest_path = comparison_dir / V7_CONTROL_MANIFEST_FILENAME
    manifest, manifest_sha256 = load_json_with_sha256(manifest_path)
    if manifest_sha256 != EXPECTED_V7_CONTROL_MANIFEST_SHA256:
        raise ValueError("V7 finalizer control manifest SHA256 differs")
    if (
        manifest.get("schema")
        != "sctransnet_tpd_clean_v7_dch_finalizer_control_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("artifact_count") != 27
    ):
        raise ValueError("V7 finalizer control manifest is incomplete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("V7 finalizer control manifest lacks artifacts")
    expected_name = (
        f"{job['variant']}/seed_{int(job['seed'])}/{job['role']}"
    )
    matches = [
        record
        for record in artifacts
        if isinstance(record, Mapping)
        and record.get("stage") == "mechanism_checkpoint"
        and record.get("name") == expected_name
        and Path(str(record.get("path", ""))).resolve() == registry_path
    ]
    if len(matches) != 1:
        raise ValueError("V7 registry lacks one exact sealed manifest record")
    sealed_sha256 = _require_sha256(
        matches[0].get("sha256"),
        "sealed V7 registry SHA256",
    )
    if file_sha256(registry_path) != sealed_sha256:
        raise ValueError("V7 registry differs from its finalizer seal")

    mechanism_reports = [
        record
        for record in artifacts
        if isinstance(record, Mapping)
        and record.get("stage") == "mechanism_report"
        and record.get("name")
        == "tpd_clean_v7_dch_mechanism_audit.json"
    ]
    if len(mechanism_reports) != 1:
        raise ValueError("V7 mechanism report seal is absent")
    mechanism_report_path = Path(
        str(mechanism_reports[0].get("path", ""))
    ).resolve()
    mechanism_report_sha256 = _require_sha256(
        mechanism_reports[0].get("sha256"),
        "sealed V7 mechanism report SHA256",
    )
    if file_sha256(mechanism_report_path) != mechanism_report_sha256:
        raise ValueError("V7 mechanism report differs from its seal")

    source_locks = manifest.get("source_locks")
    if not isinstance(source_locks, Mapping) or not source_locks:
        raise ValueError("V7 control manifest lacks source-lock seals")
    normalized_locks: Dict[str, Dict[str, str]] = {}
    for name, record in source_locks.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("V7 control source-lock record is invalid")
        path = Path(str(record.get("path", ""))).resolve()
        digest = _require_sha256(
            record.get("sha256"),
            f"V7 control source lock {name}",
        )
        if file_sha256(path) != digest:
            raise ValueError(f"V7 control source lock changed: {name}")
        normalized_locks[name] = {
            "path": str(path),
            "sha256": digest,
        }
    return {
        "control_manifest": str(manifest_path.resolve()),
        "control_manifest_sha256": manifest_sha256,
        "registry_path": str(registry_path),
        "registry_sha256": sealed_sha256,
        "mechanism_report": str(mechanism_report_path),
        "mechanism_report_sha256": mechanism_report_sha256,
        "source_locks": normalized_locks,
        "source_locks_sha256": canonical_json_sha256(normalized_locks),
    }


def _registry_source(
    job: Mapping[str, Any],
) -> tuple[
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
]:
    path = Path(job["output"])
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"Expected a regular V7 registry source: {path}"
        )
    seal = sealed_v7_registry_binding(job)
    payload, source_sha256 = load_json_with_sha256(path)
    if source_sha256 != seal["registry_sha256"]:
        raise ValueError("Loaded V7 registry bytes differ from their seal")
    if (
        payload.get("schema") != v7_diag.CHECKPOINT_SCHEMA
        or payload.get("formal_inputs_unchanged") is not True
        or payload.get("training_performed") is not False
        or payload.get("checkpoint_reselection_permitted") is not False
        or payload.get("official_test_accessed") is not False
    ):
        raise ValueError("V7 mechanism checkpoint audit is incomplete")
    expected_role = v7_diag.CHECKPOINT_SPECS[
        str(job["role"])
    ]["checkpoint_role"]
    observed_identity = {
        "variant": payload.get("variant"),
        "seed": payload.get("seed"),
        "comparison_role": payload.get("checkpoint_role"),
        "checkpoint": str(
            Path(str(payload.get("checkpoint", ""))).resolve()
        ),
    }
    expected_identity = {
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
        "comparison_role": str(job["role"]),
        "checkpoint": str(Path(job["checkpoint"]).resolve()),
    }
    if observed_identity != expected_identity:
        raise ValueError(
            "V7 registry job identity mismatch: "
            f"expected={expected_identity}, observed={observed_identity}"
        )
    source_identity = payload.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("V7 registry lacks source identity")
    required_source_identity = {
        "dataset": DATASET,
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
        "comparison_role": str(job["role"]),
        "checkpoint_role": expected_role,
    }
    for key, expected in required_source_identity.items():
        if source_identity.get(key) != expected:
            raise ValueError(f"V7 registry source identity mismatch at {key}")
    before = payload.get("input_sha256_before")
    after = payload.get("input_sha256_after")
    if not isinstance(before, Mapping) or dict(after or {}) != dict(before):
        raise ValueError("V7 registry input hashes are absent or changed")
    for label in FORMAL_INPUT_LABELS:
        _require_sha256(before.get(label), f"registry input {label}")
    if (
        before.get("checkpoint") != file_sha256(Path(job["checkpoint"]))
    ):
        raise ValueError("V7 registry checkpoint SHA256 mismatch")
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("V7 registry lacks validation identity")
    registry_ids = validation.get("validation_ids")
    if (
        not isinstance(registry_ids, list)
        or validation.get("validation_count") != len(registry_ids)
    ):
        raise ValueError("V7 registry validation coverage is incomplete")
    ordered_validation_ids_sha256(registry_ids)
    points = payload.get("operating_points")
    if not isinstance(points, dict) or not points:
        raise ValueError("V7 mechanism checkpoint has no operating points")
    ready: Dict[str, Dict[str, Any]] = {}
    for key, value in points.items():
        if not isinstance(value, dict):
            raise ValueError("Invalid V7 operating point")
        threshold = float(value["threshold"])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Non-finite registered threshold")
        labels = value.get("registry_labels", ())
        kinds = value.get("registry_kinds", ())
        if (
            not isinstance(labels, list)
            or not isinstance(kinds, list)
            or any(not isinstance(item, str) for item in labels + kinds)
        ):
            raise ValueError("Invalid V7 operating-point registry labels")
        ready[str(key)] = {
            "threshold": threshold,
            "registry_labels": list(labels),
            "registry_kinds": list(kinds),
        }
    _require_all_numeric_finite(ready, "V7 operating-point registry")
    return payload, ready, seal


def _registry_points(job: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return _registry_source(job)[1]


def _validated_fingerprint_records(
    value: Any,
    location: str,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{location} is not a non-empty mapping")
    ready: Dict[str, Dict[str, Any]] = {}
    for key, record in value.items():
        if not isinstance(key, str) or not isinstance(record, Mapping):
            raise ValueError(f"{location} contains an invalid record")
        name = record.get("name")
        count = record.get("count")
        schema = record.get("schema")
        digest = _require_sha256(
            record.get("sha256"),
            f"{location}.{key}.sha256",
        )
        if (
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(schema, str)
            or not schema
        ):
            raise ValueError(f"{location}.{key} is incomplete")
        ready[key] = {
            "schema": schema,
            "name": name,
            "count": count,
            "sha256": digest,
        }
    return ready


def _job_binding(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    registry_seal: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Bind one output to the exact formal inputs and V7 registry."""

    input_sha256 = artifacts.get("input_sha256")
    if (
        not isinstance(input_sha256, Mapping)
        or set(input_sha256) != set(FORMAL_INPUT_LABELS)
    ):
        raise ValueError("Validated artifacts lack formal input SHA256 values")
    formal_hashes = {
        label: _require_sha256(
            input_sha256.get(label),
            f"formal input {label}",
        )
        for label in FORMAL_INPUT_LABELS
    }
    for label, path in artifacts["paths"].items():
        if label in formal_hashes and file_sha256(Path(path)) != formal_hashes[label]:
            raise ValueError(f"Formal input changed before evaluation: {label}")
    if (
        Path(artifacts["paths"]["checkpoint"]).resolve()
        != Path(job["checkpoint"]).resolve()
        or formal_hashes["checkpoint"]
        != file_sha256(Path(job["checkpoint"]))
    ):
        raise ValueError("Expected checkpoint path/SHA binding differs")
    checkpoint = artifacts.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Validated artifacts lack checkpoint content")
    checkpoint_state_sha256 = state_dict_model_sha256(
        checkpoint.get("state_dict")
    )

    split = artifacts["split"]
    validation_ids = split.get("used_val_ids")
    if not isinstance(validation_ids, list):
        raise ValueError("Validated split lacks ordered validation IDs")
    validation_ids = list(validation_ids)
    validation_ids_digest = ordered_validation_ids_sha256(validation_ids)
    registry_validation = registry_payload.get("validation")
    if (
        not isinstance(registry_validation, Mapping)
        or registry_validation.get("validation_ids") != validation_ids
        or registry_validation.get("validation_count") != len(validation_ids)
    ):
        raise ValueError("Registry and split ordered validation IDs differ")
    registry_hashes = registry_payload.get("input_sha256_before")
    if (
        not isinstance(registry_hashes, Mapping)
        or set(registry_hashes) != set(FORMAL_INPUT_LABELS)
        or {
            label: registry_hashes.get(label)
            for label in FORMAL_INPUT_LABELS
        }
        != formal_hashes
    ):
        raise ValueError("Registry and formal input fingerprints differ")

    source_identity = artifacts.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("Validated artifacts lack source identity")
    registry_source_identity = registry_payload.get("source_identity")
    if (
        not isinstance(registry_source_identity, Mapping)
        or dict(registry_source_identity) != dict(source_identity)
    ):
        raise ValueError("Registry and formal source identities differ")
    source_locks = source_identity.get("source_locks")
    if not isinstance(source_locks, Mapping) or not source_locks:
        raise ValueError("Formal source identity lacks source locks")
    normalized_source_locks = {
        str(name): _require_sha256(
            digest,
            f"source lock {name}",
        )
        for name, digest in source_locks.items()
    }
    protocol_identity = artifacts["protocol"].get("run_identity")
    if not isinstance(protocol_identity, Mapping):
        raise ValueError("Protocol lacks its run identity")
    if protocol_identity.get("source_locks") != normalized_source_locks:
        raise ValueError("Protocol and artifact source locks differ")
    data_identity = {
        "architecture_id": _require_sha256(
            protocol_identity.get("architecture_id"),
            "protocol architecture_id",
        ),
        "builder_manifest_sha256": _require_sha256(
            protocol_identity.get("builder_manifest_sha256"),
            "protocol builder manifest",
        ),
        "contract_sha256": _require_sha256(
            protocol_identity.get("contract_sha256"),
            "protocol contract",
        ),
        "data_sha256": _require_sha256(
            protocol_identity.get("data_sha256"),
            "protocol data",
        ),
        "split_sha256": _require_sha256(
            protocol_identity.get("split_sha256"),
            "protocol split",
        ),
        "ordered_data_fingerprints": _validated_fingerprint_records(
            protocol_identity.get("ordered_data_fingerprints"),
            "protocol ordered_data_fingerprints",
        ),
        "ordered_split_fingerprints": _validated_fingerprint_records(
            protocol_identity.get("ordered_split_fingerprints"),
            "protocol ordered_split_fingerprints",
        ),
    }
    if (
        set(data_identity["ordered_data_fingerprints"])
        != EXPECTED_ORDERED_DATA_FINGERPRINTS
        or set(data_identity["ordered_split_fingerprints"])
        != EXPECTED_ORDERED_SPLIT_FINGERPRINTS
    ):
        raise ValueError("Protocol ordered fingerprint registry differs")
    validation_split_fingerprint = data_identity[
        "ordered_split_fingerprints"
    ].get("validation")
    if (
        validation_split_fingerprint is None
        or validation_split_fingerprint["count"] != len(validation_ids)
        or data_identity["ordered_split_fingerprints"][
            "full_validation"
        ]["count"]
        != len(validation_ids)
        or data_identity["ordered_data_fingerprints"][
            "validation_samples"
        ]["count"]
        != len(validation_ids)
    ):
        raise ValueError(
            "Protocol ordered validation fingerprint coverage differs"
        )
    registered_validation_data = data_identity[
        "ordered_data_fingerprints"
    ]["validation_samples"]
    arguments = artifacts["protocol"].get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Protocol lacks arguments for data validation")
    dataset_dir = Path(str(arguments.get("dataset_dir", "")))
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    current_validation_data = current_validation_data_binding(
        dataset_dir / DATASET,
        validation_ids,
    )
    if current_validation_data["fingerprint"] != registered_validation_data:
        raise ValueError(
            "Current validation image/mask bytes differ from protocol"
        )
    split_hashes = split.get("hashes")
    if not isinstance(split_hashes, Mapping):
        raise ValueError("Split lacks registered hashes")
    used_val_sha256 = _require_sha256(
        split_hashes.get("used_val_sha256"),
        "split used_val_sha256",
    )

    expected_role = v7_diag.CHECKPOINT_SPECS[
        str(job["role"])
    ]["checkpoint_role"]
    expected_identity = {
        "dataset": DATASET,
        "variant": str(job["variant"]),
        "v8_variant": V8_VARIANT_BY_V7[str(job["variant"])],
        "seed": int(job["seed"]),
        "comparison_role": str(job["role"]),
        "checkpoint_role": expected_role,
        "run_directory": str(Path(job["run_dir"]).resolve()),
        "checkpoint": str(Path(job["checkpoint"]).resolve()),
        "checkpoint_sha256": formal_hashes["checkpoint"],
        "checkpoint_state_dict_sha256": checkpoint_state_sha256,
        "registry_source": str(Path(job["output"]).resolve()),
        "registry_source_sha256": registry_seal["registry_sha256"],
        "counterfactual_output": str(
            job_output_path(output_dir, job).resolve()
        ),
    }
    execution_binding = process_execution_binding()
    binding: Dict[str, Any] = {
        "schema": JOB_BINDING_SCHEMA,
        "expected_job": expected_identity,
        "source_identity": json_ready(source_identity),
        "formal_input_sha256": formal_hashes,
        "formal_input_fingerprint_sha256": canonical_json_sha256(
            formal_hashes
        ),
        "source_locks": normalized_source_locks,
        "source_locks_sha256": canonical_json_sha256(
            normalized_source_locks
        ),
        "protocol_data_identity": data_identity,
        "protocol_data_identity_sha256": canonical_json_sha256(
            data_identity
        ),
        "current_validation_data": current_validation_data,
        "current_validation_data_sha256": canonical_json_sha256(
            current_validation_data
        ),
        "counterfactual_execution": execution_binding,
        "v8_protocol_sha256": execution_binding["sources"][
            "v8_protocol"
        ]["sha256"],
        "v8_preflight_amendment_sha256": execution_binding["sources"][
            "v8_preflight_amendment"
        ]["sha256"],
        "ordered_validation": {
            "count": len(validation_ids),
            "ids": validation_ids,
            "ordered_ids_sha256": validation_ids_digest,
            "split_used_val_sha256": used_val_sha256,
            "protocol_ordered_validation_split_fingerprint": (
                validation_split_fingerprint
            ),
        },
        "registry": {
            "point_count": len(registry),
            "points_sha256": canonical_json_sha256(registry),
            "source_sha256": expected_identity[
                "registry_source_sha256"
            ],
            "v7_finalizer_seal": dict(registry_seal),
        },
    }
    binding["counterfactual_execution_sha256"] = binding[
        "counterfactual_execution"
    ]["execution_sha256"]
    binding["binding_sha256"] = canonical_json_sha256(binding)
    return json_ready(binding)


def preflight(results_root: Path, output_dir: Path) -> Dict[str, Any]:
    require_analysis_output_separate(results_root, output_dir)
    records: list[Dict[str, Any]] = []
    ready = True
    for job in expected_jobs(results_root):
        try:
            artifacts = v7_diag.validate_job_artifacts(job)
            registry_payload, registry, registry_seal = _registry_source(
                job
            )
            binding = _job_binding(
                job,
                artifacts,
                registry_payload,
                registry,
                registry_seal,
                output_dir,
            )
            state = artifacts["checkpoint"].get("state_dict")
            if not isinstance(state, Mapping):
                raise ValueError("checkpoint has no state_dict mapping")
            v8_variant = V8_VARIANT_BY_V7[str(job["variant"])]
            with contextlib.redirect_stdout(sys.stderr):
                model, _ = build_clean_v8_mprs_dch_model(
                    v8_variant,
                    int(job["seed"]),
                )
            model.load_state_dict(state, strict=True)
            records.append(
                {
                    "variant": job["variant"],
                    "v8_variant": v8_variant,
                    "seed": job["seed"],
                    "role": job["role"],
                    "checkpoint": str(Path(job["checkpoint"]).resolve()),
                    "checkpoint_sha256": file_sha256(
                        Path(job["checkpoint"])
                    ),
                    "job_binding": binding,
                    "registered_threshold_count": len(registry),
                    "strict_load": True,
                    "output": str(
                        job_output_path(output_dir, job).resolve()
                    ),
                }
            )
        except Exception as exc:
            ready = False
            records.append(
                {
                    "variant": job.get("variant"),
                    "seed": job.get("seed"),
                    "role": job.get("role"),
                    "strict_load": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema": SCHEMA,
        "mode": "preflight",
        "ready": ready and len(records) == 12,
        "job_count": len(records),
        "strict_load_count": sum(
            record.get("strict_load") is True for record in records
        ),
        "results_root": str(Path(results_root).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "jobs": records,
        "writes_performed": 0,
    }


@contextmanager
def capture_mprs_diagnostics(
    model: nn.Module,
) -> Iterator[
    tuple[
        Dict[str, Mapping[str, torch.Tensor]],
        Dict[str, Dict[str, float | bool]],
    ]
]:
    """Preserve formal forwards while collecting a tolerance-checked side pass."""

    records: Dict[str, Mapping[str, torch.Tensor]] = {}
    forward_checks: Dict[str, Dict[str, float | bool]] = {}
    originals: list[tuple[TPDCleanV8MPRSDCHBlock, Any]] = []
    blocks = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TPDCleanV8MPRSDCHBlock)
    ]
    if tuple(name for name, _ in blocks) != EXPECTED_BLOCK_NAMES:
        raise RuntimeError("V8 MPRS block identity/coverage differs")
    for name, module in blocks:
        original = module.forward

        def wrapped(
            x: torch.Tensor,
            *,
            block: TPDCleanV8MPRSDCHBlock = module,
            block_name: str = name,
            formal_forward: Any = original,
        ) -> torch.Tensor:
            formal_output = formal_forward(x)
            diagnostic_output, diagnostics = (
                block.forward_with_mprs_diagnostics(x)
            )
            absolute_difference = (
                diagnostic_output.float() - formal_output.float()
            ).abs()
            allowed = (
                DIAGNOSTIC_FORWARD_ATOL
                + DIAGNOSTIC_FORWARD_RTOL * formal_output.float().abs()
            )
            maximum = float(absolute_difference.max().item())
            maximum_allowed = float(allowed.max().item())
            within_tolerance = bool(
                (absolute_difference <= allowed).all().item()
            )
            if not within_tolerance:
                raise RuntimeError(
                    f"{block_name} diagnostic/production output differs "
                    f"(max_abs={maximum:.9g}, "
                    f"max_allowed={maximum_allowed:.9g})"
                )
            records[block_name] = diagnostics
            forward_checks[block_name] = {
                "max_abs_difference": maximum,
                "max_allowed_difference": maximum_allowed,
                "within_frozen_tolerance": True,
            }
            return formal_output

        originals.append((module, original))
        module.forward = wrapped  # type: ignore[method-assign]
    try:
        yield records, forward_checks
    finally:
        for module, original in originals:
            module.forward = original  # type: ignore[method-assign]


def hard_negative_mask(
    probability: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(probability) > REGISTERED_FIXED_THRESHOLD
    target_binary = np.asarray(target) > 0.5
    if target_binary.any():
        exclusion = morphology.binary_dilation(
            target_binary,
            footprint=morphology.disk(HARD_NEGATIVE_DILATION_RADIUS),
        )
    else:
        exclusion = target_binary
    labels = measure.label(prediction, connectivity=2)
    result = np.zeros_like(prediction, dtype=bool)
    for region in measure.regionprops(labels):
        rows, columns = region.coords.T
        if not bool(exclusion[rows, columns].any()):
            result[rows, columns] = True
    return result


def _padded_mask(
    mask: np.ndarray,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    output = torch.zeros(
        (1, 1, height, width),
        dtype=torch.float32,
        device=device,
    )
    source = torch.as_tensor(mask, dtype=torch.float32, device=device)
    output[0, 0, : source.shape[0], : source.shape[1]] = source
    return output


def _masked_correction_stats(
    diagnostics: Mapping[str, Mapping[str, torch.Tensor]],
    target_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> Dict[str, Any]:
    if tuple(diagnostics) != EXPECTED_BLOCK_NAMES:
        raise RuntimeError("MPRS diagnostic block coverage/order differs")
    _require_all_numeric_finite(target_mask, "target mask")
    _require_all_numeric_finite(negative_mask, "hard-negative mask")
    target_sum = 0.0
    target_count = 0
    negative_sum = 0.0
    negative_count = 0
    overlap_removed_count = 0
    corr_count = 0
    sx = sy = sxx = syy = sxy = 0.0
    blocks: Dict[str, Any] = {}
    for name, record in diagnostics.items():
        required = {"phase_correction", "context_aligned", "scale"}
        if not required.issubset(record):
            raise ValueError(f"{name} lacks required diagnostic tensors")
        _require_all_numeric_finite(record, f"diagnostics.{name}")
        correction = record["phase_correction"].float()
        context = record["context_aligned"].float()
        if correction.shape != context.shape:
            raise ValueError(f"{name} correction/context shapes differ")
        keep_linear = 3.0 * correction + context
        magnitude = correction.abs().mean(dim=1, keepdim=True)
        target_at_scale = F.adaptive_max_pool2d(
            target_mask,
            magnitude.shape[-2:],
        ) > 0.5
        negative_raw_at_scale = F.adaptive_max_pool2d(
            negative_mask,
            magnitude.shape[-2:],
        ) > 0.5
        pooled_overlap = target_at_scale & negative_raw_at_scale
        negative_at_scale = negative_raw_at_scale & ~target_at_scale
        if bool((target_at_scale & negative_at_scale).any().item()):
            raise RuntimeError(
                f"{name} target-priority pooled masks are not disjoint"
            )
        block_target_count = int(target_at_scale.sum().item())
        block_negative_count = int(negative_at_scale.sum().item())
        block_overlap_removed_count = int(pooled_overlap.sum().item())
        block_target_sum = float(
            magnitude.masked_select(target_at_scale).sum().item()
        )
        block_negative_sum = float(
            magnitude.masked_select(negative_at_scale).sum().item()
        )
        target_sum += block_target_sum
        target_count += block_target_count
        negative_sum += block_negative_sum
        negative_count += block_negative_count
        overlap_removed_count += block_overlap_removed_count

        x = correction.reshape(-1).double()
        y = keep_linear.reshape(-1).double()
        corr_count += x.numel()
        sx += float(x.sum().item())
        sy += float(y.sum().item())
        sxx += float(x.square().sum().item())
        syy += float(y.square().sum().item())
        sxy += float((x * y).sum().item())
        block_correlation_stats = {
            "count": x.numel(),
            "sum_correction": float(x.sum().item()),
            "sum_keep_linear": float(y.sum().item()),
            "sum_sq_correction": float(x.square().sum().item()),
            "sum_sq_keep_linear": float(y.square().sum().item()),
            "sum_product": float((x * y).sum().item()),
        }
        blocks[name] = {
            "target_sum": block_target_sum,
            "target_count": block_target_count,
            "hard_negative_sum": block_negative_sum,
            "hard_negative_count": block_negative_count,
            "pooled_overlap_removed_count": (
                block_overlap_removed_count
            ),
            "target_priority_masks_disjoint": True,
            "mean_abs_correction": float(magnitude.mean().item()),
            "mean_abs_scale": float(record["scale"].abs().mean().item()),
            "correlation_sufficient_statistics": (
                block_correlation_stats
            ),
        }
    result = {
        "target_sum": target_sum,
        "target_count": target_count,
        "hard_negative_sum": negative_sum,
        "hard_negative_count": negative_count,
        "pooled_overlap_removed_count": overlap_removed_count,
        "correlation_sufficient_statistics": {
            "count": corr_count,
            "sum_correction": sx,
            "sum_keep_linear": sy,
            "sum_sq_correction": sxx,
            "sum_sq_keep_linear": syy,
            "sum_product": sxy,
        },
        "blocks": blocks,
    }
    _require_all_numeric_finite(result, "masked correction statistics")
    return result


def _aligned_shift_error(
    base: torch.Tensor,
    shifted: torch.Tensor,
    dy: int,
    dx: int,
) -> float:
    height, width = base.shape[-2:]
    top = SHIFT_CROP
    left = SHIFT_CROP
    bottom = height - SHIFT_CROP
    right = width - SHIFT_CROP
    reference = base[
        ...,
        top : bottom - dy if dy else bottom,
        left : right - dx if dx else right,
    ]
    candidate = shifted[
        ...,
        top + dy : bottom,
        left + dx : right,
    ]
    if reference.shape != candidate.shape or not reference.numel():
        raise RuntimeError("Shift-aligned output crop is invalid")
    numerator = (candidate.float() - reference.float()).abs().mean()
    denominator = reference.float().abs().mean() + 1e-6
    return float((numerator / denominator).item())


@torch.inference_mode()
def _shift_errors(
    v7_model: nn.Module,
    v8_model: nn.Module,
    image: torch.Tensor,
    v7_base: torch.Tensor,
    v8_base: torch.Tensor,
) -> Dict[str, list[float]]:
    result = {"v7": [], "v8": []}
    for dy, dx in SHIFT_OFFSETS[1:]:
        shifted_image = torch.roll(
            image,
            shifts=(dy, dx),
            dims=(-2, -1),
        )
        v7_raw = v7_model(shifted_image)
        v8_raw = v8_model(shifted_image)
        if _validate_model_output_tree(
            v7_raw,
            f"V7 shifted raw output[{dy},{dx}]",
        )["tensor_count"] != 6:
            raise RuntimeError("V7 shifted raw output count is not six")
        if _validate_model_output_tree(
            v8_raw,
            f"V8 shifted raw output[{dy},{dx}]",
        )["tensor_count"] != 6:
            raise RuntimeError("V8 shifted raw output count is not six")
        v7_shift = final_prediction(v7_raw)
        v8_shift = final_prediction(v8_raw)
        result["v7"].append(
            _aligned_shift_error(v7_base, v7_shift, dy, dx)
        )
        result["v8"].append(
            _aligned_shift_error(v8_base, v8_shift, dy, dx)
        )
    _require_all_numeric_finite(result, "shift errors")
    return result


def _correlation(stats: Mapping[str, Any]) -> float:
    required = (
        "count",
        "sum_correction",
        "sum_keep_linear",
        "sum_sq_correction",
        "sum_sq_keep_linear",
        "sum_product",
    )
    if any(key not in stats for key in required):
        raise ValueError("Correlation sufficient statistics are incomplete")
    _require_all_numeric_finite(stats, "correlation sufficient statistics")
    count = _require_integer(
        stats["count"],
        "correlation.count",
        minimum=0,
    )
    if count < 2:
        return 0.0
    sx = _require_real(stats["sum_correction"], "correlation.sum_correction")
    sy = _require_real(
        stats["sum_keep_linear"],
        "correlation.sum_keep_linear",
    )
    sxx = _require_real(
        stats["sum_sq_correction"],
        "correlation.sum_sq_correction",
        minimum=0.0,
    )
    syy = _require_real(
        stats["sum_sq_keep_linear"],
        "correlation.sum_sq_keep_linear",
        minimum=0.0,
    )
    sxy = _require_real(
        stats["sum_product"],
        "correlation.sum_product",
    )
    covariance = sxy - sx * sy / count
    variance_x = max(0.0, sxx - sx * sx / count)
    variance_y = max(0.0, syy - sy * sy / count)
    denominator = math.sqrt(variance_x * variance_y)
    result = covariance / denominator if denominator else 0.0
    if not math.isfinite(result):
        raise ValueError("Derived correction correlation is non-finite")
    if result < -1.000001 or result > 1.000001:
        raise ValueError("Derived correction correlation is out of range")
    return result


def _add_correlation_stats(
    destination: Dict[str, float | int],
    source: Mapping[str, Any],
) -> None:
    _require_all_numeric_finite(source, "correlation accumulation source")
    for key in (
        "count",
        "sum_correction",
        "sum_keep_linear",
        "sum_sq_correction",
        "sum_sq_keep_linear",
        "sum_product",
    ):
        if key not in source:
            raise ValueError("Correlation accumulation source is incomplete")
        destination[key] += source[key]


def _finalize_block_reports(
    block_totals: Mapping[str, Mapping[str, Any]],
    expected_image_count: int,
) -> Dict[str, Dict[str, Any]]:
    if tuple(block_totals) != EXPECTED_BLOCK_NAMES:
        raise RuntimeError("Per-block MPRS coverage/order is incomplete")
    reports: Dict[str, Dict[str, Any]] = {}
    for name in EXPECTED_BLOCK_NAMES:
        totals = block_totals[name]
        target_count = int(totals["target_count"])
        negative_count = int(totals["hard_negative_count"])
        image_count = int(totals["image_count"])
        if image_count != expected_image_count:
            raise RuntimeError(
                f"{name} covered {image_count}/{expected_image_count} images"
            )
        if target_count <= 0:
            raise RuntimeError(f"{name} has empty target coverage")
        if negative_count <= 0:
            raise RuntimeError(f"{name} has empty hard-negative coverage")
        target_sum = float(totals["target_sum"])
        negative_sum = float(totals["hard_negative_sum"])
        target_mean = target_sum / target_count
        negative_mean = negative_sum / negative_count
        correlation_stats = dict(
            totals["correlation_sufficient_statistics"]
        )
        report = {
            "target_sum": target_sum,
            "target_count": target_count,
            "target_mean_abs": target_mean,
            "hard_negative_sum": negative_sum,
            "hard_negative_count": negative_count,
            "hard_negative_mean_abs": negative_mean,
            "pooled_overlap_removed_count": _require_integer(
                totals.get("pooled_overlap_removed_count", 0),
                f"{name}.pooled_overlap_removed_count",
                minimum=0,
            ),
            "target_priority_masks_disjoint": True,
            "target_correction_lift": (
                target_mean / (negative_mean + 1e-6)
            ),
            "mean_abs_correction": (
                float(totals["mean_abs_correction_sum"])
                / image_count
            ),
            "mean_abs_scale": (
                float(totals["mean_abs_scale_sum"]) / image_count
            ),
            "image_count": image_count,
            "image_coverage_complete": True,
            "correction_keep_correlation": _correlation(
                correlation_stats
            ),
            "correlation_sufficient_statistics": correlation_stats,
            "diagnostic_forward_max_abs_difference": float(
                totals.get(
                    "diagnostic_forward_max_abs_difference",
                    0.0,
                )
            ),
            "diagnostic_forward_max_allowed_difference": float(
                totals.get(
                    "diagnostic_forward_max_allowed_difference",
                    0.0,
                )
            ),
            "diagnostic_production_forward_within_frozen_tolerance": (
                float(
                    totals.get(
                        "diagnostic_forward_max_abs_difference",
                        0.0,
                    )
                )
                <= float(
                    totals.get(
                        "diagnostic_forward_max_allowed_difference",
                        0.0,
                    )
                )
            ),
        }
        _require_all_numeric_finite(report, f"block report {name}")
        reports[name] = report
    return reports


def _validate_model_output_tree(value: Any, location: str) -> Dict[str, int]:
    tensors: list[torch.Tensor] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            _require_all_numeric_finite(item, path)
            tensors.append(item)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            raise ValueError(f"{path} is not a tensor output container")

    visit(value, location)
    if not tensors:
        raise ValueError(f"{location} contains no tensor outputs")
    return {
        "tensor_count": len(tensors),
        "element_count": sum(tensor.numel() for tensor in tensors),
    }


def _topology_view(point: Mapping[str, Any]) -> Dict[str, Any]:
    topology = point["gt_topology"]
    fractions = [
        float(record["largest_fragment_fraction"])
        for record in topology["per_gt"]
        if int(record["overlapping_prediction_components"]) > 0
    ]
    overlap_covered_gt_count = sum(
        int(record["overlapping_prediction_components"]) > 0
        for record in topology["per_gt"]
    )
    if int(point["target_count"]) <= 0:
        raise RuntimeError("Topology point has empty target coverage")
    view = {
        "pd": float(point["pd"]),
        "fa": float(point["fa"]),
        "miou": float(point["miou"]),
        "matched_target_count": int(point["matched_target_count"]),
        "target_count": int(point["target_count"]),
        "unmatched_predicted_object_count": int(
            point["unmatched_predicted_object_count"]
        ),
        "fragment_excess_total": int(
            topology["fragment_excess_total"]
        ),
        "overlap_covered_gt_count": overlap_covered_gt_count,
        "largest_fragment_fraction_mean": float(
            topology["largest_fragment_fraction_mean"]
        ),
        "largest_fragment_fraction_p10": float(
            topology["largest_fragment_fraction_p10"]
        ),
        "largest_fragment_fractions": fractions,
    }
    _require_all_numeric_finite(view, "topology operating point")
    return view


def _paired_gt_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    if not records and not allow_empty:
        raise ValueError("Paired topology has no V7-covered reference GT")
    v7_fragments = 0
    v8_fragments = 0
    v7_fractions: list[float] = []
    v8_fractions: list[float] = []
    v8_covered = 0
    seen: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Paired topology record {index} is invalid")
        identifier = record.get("identifier")
        gt_index = _require_integer(
            record.get("gt_index"),
            f"paired_gt[{index}].gt_index",
            minimum=0,
        )
        _require_integer(
            record.get("gt_area"),
            f"paired_gt[{index}].gt_area",
            minimum=1,
        )
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"paired_gt[{index}] identifier is invalid")
        identity = (identifier, gt_index)
        if identity in seen:
            raise ValueError(f"Duplicate paired GT identity: {identity}")
        seen.add(identity)
        if record.get("v7_reference_coverage") != 1:
            raise ValueError("Paired GT is not V7-covered")
        v8_reference_coverage = _require_integer(
            record.get("v8_reference_coverage"),
            f"paired_gt[{index}].v8_reference_coverage",
            minimum=0,
        )
        if v8_reference_coverage not in (0, 1):
            raise ValueError("V8 reference coverage must be binary")
        v7_overlap = _require_integer(
            record.get("v7_overlapping_prediction_components"),
            f"paired_gt[{index}].v7_overlap",
            minimum=1,
        )
        v8_overlap = _require_integer(
            record.get("v8_overlapping_prediction_components"),
            f"paired_gt[{index}].v8_overlap",
            minimum=0,
        )
        if v8_reference_coverage != int(v8_overlap > 0):
            raise ValueError("V8 reference coverage/overlap differs")
        v7_fragment = _require_integer(
            record.get("v7_fragment_excess"),
            f"paired_gt[{index}].v7_fragment_excess",
            minimum=0,
        )
        v8_fragment = _require_integer(
            record.get("v8_fragment_excess"),
            f"paired_gt[{index}].v8_fragment_excess",
            minimum=0,
        )
        if (
            v7_fragment != max(0, v7_overlap - 1)
            or v8_fragment != max(0, v8_overlap - 1)
        ):
            raise ValueError("Paired GT fragment excess is not component-1")
        v7_fraction = _require_real(
            record.get("v7_largest_fragment_fraction"),
            f"paired_gt[{index}].v7_largest_fragment_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        v8_fraction = _require_real(
            record.get("v8_largest_fragment_fraction"),
            f"paired_gt[{index}].v8_largest_fragment_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if v7_fraction <= 0.0:
            raise ValueError("V7-covered reference GT has zero fraction")
        if v8_reference_coverage and v8_fraction <= 0.0:
            raise ValueError("Covered V8 reference GT has zero fraction")
        if not v8_reference_coverage and v8_fraction != 0.0:
            raise ValueError("Uncovered V8 reference GT fraction is not zero")
        v7_fragments += v7_fragment
        v8_fragments += v8_fragment
        v7_fractions.append(v7_fraction)
        v8_fractions.append(v8_fraction)
        v8_covered += v8_reference_coverage
    return {
        "reference_gt_count": len(records),
        "v7_covered_reference_gt_count": len(records),
        "v8_covered_reference_gt_count": v8_covered,
        "v7_fragment_excess_total": v7_fragments,
        "v8_fragment_excess_total": v8_fragments,
        "v7_largest_fragment_fractions": v7_fractions,
        "v8_largest_fragment_fractions": v8_fractions,
    }


def paired_topology_from_decorated_points(
    point_v7: Mapping[str, Any],
    point_v8: Mapping[str, Any],
) -> Dict[str, Any]:
    topology_v7 = point_v7.get("gt_topology")
    topology_v8 = point_v8.get("gt_topology")
    if not isinstance(topology_v7, Mapping) or not isinstance(
        topology_v8,
        Mapping,
    ):
        raise ValueError("Decorated points lack GT topology")
    per_gt_v7 = topology_v7.get("per_gt")
    per_gt_v8 = topology_v8.get("per_gt")
    if not isinstance(per_gt_v7, list) or not isinstance(per_gt_v8, list):
        raise ValueError("Decorated points lack per-GT topology")

    def indexed(records: Sequence[Mapping[str, Any]]) -> Dict[
        tuple[str, int],
        Mapping[str, Any],
    ]:
        ready: Dict[tuple[str, int], Mapping[str, Any]] = {}
        for record in records:
            identifier = record.get("identifier")
            gt_index = _require_integer(
                record.get("gt_index"),
                "decorated per_gt.gt_index",
                minimum=0,
            )
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Decorated per-GT identifier is invalid")
            key = (identifier, gt_index)
            if key in ready:
                raise ValueError(f"Duplicate decorated GT identity: {key}")
            ready[key] = record
        return ready

    indexed_v7 = indexed(per_gt_v7)
    indexed_v8 = indexed(per_gt_v8)
    if set(indexed_v7) != set(indexed_v8):
        raise ValueError("V7/V8 decorated GT identities differ")
    paired: list[Dict[str, Any]] = []
    for identity, v7_record in indexed_v7.items():
        v7_overlap = _require_integer(
            v7_record.get("overlapping_prediction_components"),
            f"{identity}.v7 overlap",
            minimum=0,
        )
        if v7_overlap <= 0:
            continue
        v8_record = indexed_v8[identity]
        v7_area = _require_integer(
            v7_record.get("gt_area"),
            f"{identity}.v7 gt_area",
            minimum=1,
        )
        v8_area = _require_integer(
            v8_record.get("gt_area"),
            f"{identity}.v8 gt_area",
            minimum=1,
        )
        if v8_area != v7_area:
            raise ValueError("V7/V8 paired GT area differs")
        v8_overlap = _require_integer(
            v8_record.get("overlapping_prediction_components"),
            f"{identity}.v8 overlap",
            minimum=0,
        )
        v8_covered = int(v8_overlap > 0)
        paired.append(
            {
                "identifier": identity[0],
                "gt_index": identity[1],
                "gt_area": v7_area,
                "v7_reference_coverage": 1,
                "v8_reference_coverage": v8_covered,
                "v7_overlapping_prediction_components": v7_overlap,
                "v8_overlapping_prediction_components": v8_overlap,
                "v7_fragment_excess": max(0, v7_overlap - 1),
                "v8_fragment_excess": max(0, v8_overlap - 1),
                "v7_largest_fragment_fraction": _require_real(
                    v7_record.get("largest_fragment_fraction"),
                    f"{identity}.v7 largest fraction",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "v8_largest_fragment_fraction": (
                    _require_real(
                        v8_record.get("largest_fragment_fraction"),
                        f"{identity}.v8 largest fraction",
                        minimum=0.0,
                        maximum=1.0,
                    )
                    if v8_covered
                    else 0.0
                ),
            }
        )
    summary = _paired_gt_summary(paired, allow_empty=True)
    return {
        "reference_definition": "V7_overlapping_prediction_components_gt0",
        "uncovered_v8_largest_fragment_fraction": 0.0,
        "paired_gt": paired,
        **summary,
    }


def topology_aggregate_from_operating_points(
    points: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recompute every gate field solely from saved paired per-GT values."""

    if not isinstance(points, Mapping) or not points:
        raise ValueError("Topology aggregation has no operating points")
    point_count = 0
    aggregate_summary = {
        "reference_gt_count": 0,
        "v7_covered_reference_gt_count": 0,
        "v8_covered_reference_gt_count": 0,
        "v7_fragment_excess_total": 0,
        "v8_fragment_excess_total": 0,
        "v7_largest_fragment_fractions": [],
        "v8_largest_fragment_fractions": [],
    }
    for point_name, point in points.items():
        if not isinstance(point, Mapping):
            raise ValueError(f"Operating point {point_name} is invalid")
        paired = point.get("paired_topology")
        if not isinstance(paired, Mapping):
            raise ValueError(f"{point_name} lacks paired topology")
        records = paired.get("paired_gt")
        if not isinstance(records, list):
            raise ValueError(f"{point_name} lacks paired per-GT values")
        recomputed = _paired_gt_summary(records, allow_empty=True)
        cached = {
            key: paired.get(key)
            for key in recomputed
        }
        if cached != recomputed:
            raise ValueError(
                f"{point_name} paired topology cache differs from per-GT"
            )
        v7_view = point.get("v7")
        if not isinstance(v7_view, Mapping):
            raise ValueError(f"{point_name} lacks its V7 topology view")
        if (
            _require_integer(
                v7_view.get("overlap_covered_gt_count"),
                f"{point_name}.v7.overlap_covered_gt_count",
                minimum=0,
            )
            != recomputed["reference_gt_count"]
            or _require_integer(
                v7_view.get("fragment_excess_total"),
                f"{point_name}.v7.fragment_excess_total",
                minimum=0,
            )
            != recomputed["v7_fragment_excess_total"]
            or v7_view.get("largest_fragment_fractions")
            != recomputed["v7_largest_fragment_fractions"]
        ):
            raise ValueError(
                f"{point_name} paired topology does not close against V7 view"
            )
        for key in (
            "reference_gt_count",
            "v7_covered_reference_gt_count",
            "v8_covered_reference_gt_count",
            "v7_fragment_excess_total",
            "v8_fragment_excess_total",
        ):
            aggregate_summary[key] += recomputed[key]
        aggregate_summary["v7_largest_fragment_fractions"].extend(
            recomputed["v7_largest_fragment_fractions"]
        )
        aggregate_summary["v8_largest_fragment_fractions"].extend(
            recomputed["v8_largest_fragment_fractions"]
        )
        point_count += 1
    aggregate = {
        "paired_operating_point_count": point_count,
        "reference_definition": "V7_overlapping_prediction_components_gt0",
        "uncovered_v8_largest_fragment_fraction": 0.0,
        **aggregate_summary,
    }
    if (
        aggregate["reference_gt_count"] <= 0
        or not aggregate["v7_largest_fragment_fractions"]
        or not aggregate["v8_largest_fragment_fractions"]
    ):
        raise ValueError("Paired topology job coverage is empty")
    _require_all_numeric_finite(aggregate, "paired topology aggregate")
    return aggregate


def evaluate_job(
    job: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    device: torch.device,
    device_provenance: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    registry_payload, registry, registry_seal = _registry_source(job)
    job_binding = _job_binding(
        job,
        artifacts,
        registry_payload,
        registry,
        registry_seal,
        output_dir,
    )
    checkpoint = artifacts["checkpoint"]
    state = checkpoint["state_dict"]
    v7_variant = str(job["variant"])
    v8_variant = V8_VARIANT_BY_V7[v7_variant]
    seed = int(job["seed"])
    with contextlib.redirect_stdout(sys.stderr):
        v7_model, _ = build_clean_v7_dch_model(v7_variant, seed)
        v8_model, v8_metadata = build_clean_v8_mprs_dch_model(
            v8_variant,
            seed,
        )
    v7_model.load_state_dict(state, strict=True)
    v8_model.load_state_dict(state, strict=True)
    if tuple(v7_model.state_dict()) != tuple(v8_model.state_dict()):
        raise RuntimeError("V7/V8 state layouts differ")
    v7_model.to(device).eval()
    v8_model.to(device).eval()
    state_before = {
        "v7": v6_diag.model_state_sha256(v7_model),
        "v8": v6_diag.model_state_sha256(v8_model),
    }
    expected_state_sha256 = job_binding["expected_job"][
        "checkpoint_state_dict_sha256"
    ]
    if set(state_before.values()) != {expected_state_sha256}:
        raise RuntimeError("Loaded model state differs from checkpoint state")

    protocol = artifacts["protocol"]
    split = artifacts["split"]
    arguments = protocol["arguments"]
    validation_ids = list(split["used_val_ids"])
    validation_ids_digest = ordered_validation_ids_sha256(validation_ids)
    if (
        job_binding["ordered_validation"]["ids"] != validation_ids
        or job_binding["ordered_validation"]["ordered_ids_sha256"]
        != validation_ids_digest
    ):
        raise RuntimeError("Job binding and evaluation validation IDs differ")
    dataset_dir = Path(arguments["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    validation_set = ValidationSubset(
        dataset_dir / DATASET,
        validation_ids,
        {
            key: float(value)
            for key, value in protocol["normalization"].items()
        },
    )
    loader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.BCELoss(reduction="mean")
    probabilities_v7: list[np.ndarray] = []
    probabilities_v8: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses_v7: list[float] = []
    losses_v8: list[float] = []
    processed_validation_ids: list[str] = []
    target_pixel_count = 0
    target_image_count = 0
    hard_negative_pixel_count = 0
    hard_negative_image_count = 0
    raw_output_audit = {
        "v7_tensor_count": 0,
        "v7_element_count": 0,
        "v8_tensor_count": 0,
        "v8_element_count": 0,
    }
    diagnostic_forward_max_abs = 0.0
    diagnostic_forward_max_allowed = 0.0
    diagnostic_forward_check_count = 0
    correction_totals = {
        "target_sum": 0.0,
        "target_count": 0,
        "hard_negative_sum": 0.0,
        "hard_negative_count": 0,
        "pooled_overlap_removed_count": 0,
    }
    correlation_totals = {
        "count": 0,
        "sum_correction": 0.0,
        "sum_keep_linear": 0.0,
        "sum_sq_correction": 0.0,
        "sum_sq_keep_linear": 0.0,
        "sum_product": 0.0,
    }
    block_totals: Dict[str, Dict[str, Any]] = {}
    shift_v7: list[float] = []
    shift_v8: list[float] = []

    with torch.inference_mode():
        for image_index, (images, masks, sizes, identifiers) in enumerate(
            loader
        ):
            if (
                not isinstance(identifiers, (list, tuple))
                or len(identifiers) != 1
            ):
                raise RuntimeError("Validation loader ID coverage is invalid")
            identifier = str(identifiers[0])
            if (
                image_index >= len(validation_ids)
                or identifier != validation_ids[image_index]
            ):
                raise RuntimeError("Validation loader order differs from split")
            processed_validation_ids.append(identifier)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            _require_all_numeric_finite(images, f"image[{identifier}]")
            _require_all_numeric_finite(masks, f"mask[{identifier}]")
            height = int(sizes[0, 0].item())
            width = int(sizes[0, 1].item())
            v7_raw_output = v7_model(images)
            v7_raw_audit = _validate_model_output_tree(
                v7_raw_output,
                f"v7 raw output[{identifier}]",
            )
            if v7_raw_audit["tensor_count"] != 6:
                raise RuntimeError("V7 raw output count is not six")
            v7_output = final_prediction(v7_raw_output)
            with capture_mprs_diagnostics(v8_model) as captured:
                diagnostics, diagnostic_forward_checks = captured
                v8_raw_output = v8_model(images)
                v8_raw_audit = _validate_model_output_tree(
                    v8_raw_output,
                    f"v8 raw output[{identifier}]",
                )
                if v8_raw_audit["tensor_count"] != 6:
                    raise RuntimeError("V8 raw output count is not six")
                v8_output = final_prediction(v8_raw_output)
            raw_output_audit["v7_tensor_count"] += v7_raw_audit[
                "tensor_count"
            ]
            raw_output_audit["v7_element_count"] += v7_raw_audit[
                "element_count"
            ]
            raw_output_audit["v8_tensor_count"] += v8_raw_audit[
                "tensor_count"
            ]
            raw_output_audit["v8_element_count"] += v8_raw_audit[
                "element_count"
            ]
            if len(diagnostics) != 7:
                raise RuntimeError("V8 diagnostic capture is incomplete")
            if tuple(diagnostics) != EXPECTED_BLOCK_NAMES:
                raise RuntimeError("V8 diagnostic block identity differs")
            if tuple(diagnostic_forward_checks) != EXPECTED_BLOCK_NAMES:
                raise RuntimeError(
                    "V8 diagnostic/production forward checks differ"
                )
            for check in diagnostic_forward_checks.values():
                if check["within_frozen_tolerance"] is not True:
                    raise RuntimeError(
                        "Diagnostic/production forward tolerance failed"
                    )
                diagnostic_forward_max_abs = max(
                    diagnostic_forward_max_abs,
                    float(check["max_abs_difference"]),
                )
                diagnostic_forward_max_allowed = max(
                    diagnostic_forward_max_allowed,
                    float(check["max_allowed_difference"]),
                )
                diagnostic_forward_check_count += 1
            _require_all_numeric_finite(v7_output, "V7 final output")
            _require_all_numeric_finite(v8_output, "V8 final output")
            v7_crop = v7_output[:, :, :height, :width]
            v8_crop = v8_output[:, :, :height, :width]
            target_crop = masks[:, :, :height, :width]
            probability_v7 = v7_crop[0, 0].float().cpu().numpy()
            probability_v8 = v8_crop[0, 0].float().cpu().numpy()
            target = target_crop[0, 0].float().cpu().numpy()
            _require_all_numeric_finite(
                probability_v7,
                f"v7 probability[{identifier}]",
            )
            _require_all_numeric_finite(
                probability_v8,
                f"v8 probability[{identifier}]",
            )
            _require_all_numeric_finite(target, f"target[{identifier}]")
            probabilities_v7.append(probability_v7)
            probabilities_v8.append(probability_v8)
            targets.append(target)
            loss_v7 = float(
                criterion(
                    v7_crop.float(),
                    target_crop.float(),
                ).item()
            )
            loss_v8 = float(
                criterion(
                    v8_crop.float(),
                    target_crop.float(),
                ).item()
            )
            if not math.isfinite(loss_v7) or not math.isfinite(loss_v8):
                raise ValueError("Validation BCE loss is non-finite")
            losses_v7.append(loss_v7)
            losses_v8.append(loss_v8)

            negative = hard_negative_mask(probability_v7, target)
            image_target_count = int((target > 0.5).sum())
            image_negative_count = int(negative.sum())
            target_pixel_count += image_target_count
            hard_negative_pixel_count += image_negative_count
            target_image_count += image_target_count > 0
            hard_negative_image_count += image_negative_count > 0
            target_padded = _padded_mask(
                target > 0.5,
                images.shape[-2],
                images.shape[-1],
                device=device,
            )
            negative_padded = _padded_mask(
                negative,
                images.shape[-2],
                images.shape[-1],
                device=device,
            )
            image_stats = _masked_correction_stats(
                diagnostics,
                target_padded,
                negative_padded,
            )
            for key in correction_totals:
                correction_totals[key] += image_stats[key]
            for key in correlation_totals:
                correlation_totals[key] += image_stats[
                    "correlation_sufficient_statistics"
                ][key]
            for name, values in image_stats["blocks"].items():
                totals = block_totals.setdefault(
                    name,
                    {
                        "target_sum": 0.0,
                        "target_count": 0.0,
                        "hard_negative_sum": 0.0,
                        "hard_negative_count": 0.0,
                        "pooled_overlap_removed_count": 0,
                        "mean_abs_correction_sum": 0.0,
                        "mean_abs_scale_sum": 0.0,
                        "image_count": 0.0,
                        "correlation_sufficient_statistics": {
                            "count": 0,
                            "sum_correction": 0.0,
                            "sum_keep_linear": 0.0,
                            "sum_sq_correction": 0.0,
                            "sum_sq_keep_linear": 0.0,
                            "sum_product": 0.0,
                        },
                        "diagnostic_forward_max_abs_difference": 0.0,
                        "diagnostic_forward_max_allowed_difference": 0.0,
                    },
                )
                totals["target_sum"] += values["target_sum"]
                totals["target_count"] += values["target_count"]
                totals["hard_negative_sum"] += values[
                    "hard_negative_sum"
                ]
                totals["hard_negative_count"] += values[
                    "hard_negative_count"
                ]
                totals["pooled_overlap_removed_count"] += values[
                    "pooled_overlap_removed_count"
                ]
                totals["mean_abs_correction_sum"] += values[
                    "mean_abs_correction"
                ]
                totals["mean_abs_scale_sum"] += values["mean_abs_scale"]
                totals["image_count"] += 1
                _add_correlation_stats(
                    totals["correlation_sufficient_statistics"],
                    values["correlation_sufficient_statistics"],
                )
                check = diagnostic_forward_checks[name]
                totals[
                    "diagnostic_forward_max_abs_difference"
                ] = max(
                    totals[
                        "diagnostic_forward_max_abs_difference"
                    ],
                    float(check["max_abs_difference"]),
                )
                totals[
                    "diagnostic_forward_max_allowed_difference"
                ] = max(
                    totals[
                        "diagnostic_forward_max_allowed_difference"
                    ],
                    float(check["max_allowed_difference"]),
                )

            if image_index < SHIFT_SUBSET_COUNT:
                shifts = _shift_errors(
                    v7_model,
                    v8_model,
                    images,
                    v7_output,
                    v8_output,
                )
                shift_v7.extend(shifts["v7"])
                shift_v8.extend(shifts["v8"])

    if processed_validation_ids != validation_ids:
        raise RuntimeError("Validation coverage/order is incomplete")
    if target_pixel_count <= 0 or correction_totals["target_count"] <= 0:
        raise RuntimeError("Counterfactual target coverage is empty")
    if (
        hard_negative_pixel_count <= 0
        or correction_totals["hard_negative_count"] <= 0
    ):
        raise RuntimeError("Counterfactual hard-negative coverage is empty")
    if len(losses_v7) != len(validation_ids) or len(losses_v8) != len(
        validation_ids
    ):
        raise RuntimeError("Counterfactual loss coverage is incomplete")
    expected_shift_count = min(
        SHIFT_SUBSET_COUNT,
        len(validation_ids),
    ) * (len(SHIFT_OFFSETS) - 1)
    if (
        len(shift_v7) != expected_shift_count
        or len(shift_v8) != expected_shift_count
    ):
        raise RuntimeError("Counterfactual shift coverage is incomplete")
    block_reports = _finalize_block_reports(
        block_totals,
        len(validation_ids),
    )
    target_mean = correction_totals["target_sum"] / correction_totals[
        "target_count"
    ]
    hard_negative_mean = correction_totals[
        "hard_negative_sum"
    ] / correction_totals["hard_negative_count"]
    target_lift = target_mean / (hard_negative_mean + 1e-6)
    points: Dict[str, Any] = {}
    match_radius = float(arguments["match_radius"])
    tiny_area = int(arguments["tiny_area"])
    for key, registration in registry.items():
        threshold = float(registration["threshold"])
        point_v7 = v6_diag.decorated_point(
            probabilities_v7,
            targets,
            losses_v7,
            validation_ids,
            threshold,
            match_radius,
            tiny_area,
            HARD_NEGATIVE_DILATION_RADIUS,
        )
        point_v8 = v6_diag.decorated_point(
            probabilities_v8,
            targets,
            losses_v8,
            validation_ids,
            threshold,
            match_radius,
            tiny_area,
            HARD_NEGATIVE_DILATION_RADIUS,
        )
        view_v7 = _topology_view(point_v7)
        view_v8 = _topology_view(point_v8)
        paired_topology = paired_topology_from_decorated_points(
            point_v7,
            point_v8,
        )
        points[key] = {
            **registration,
            "v7": view_v7,
            "v8": view_v8,
            "paired_topology": paired_topology,
            "delta_v8_minus_v7": {
                "pd": view_v8["pd"] - view_v7["pd"],
                "fa": view_v8["fa"] - view_v7["fa"],
                "miou": view_v8["miou"] - view_v7["miou"],
                "fragment_excess_total": (
                    view_v8["fragment_excess_total"]
                    - view_v7["fragment_excess_total"]
                ),
            },
        }
    topology_aggregate = topology_aggregate_from_operating_points(points)

    state_after = {
        "v7": v6_diag.model_state_sha256(v7_model),
        "v8": v6_diag.model_state_sha256(v8_model),
    }
    if state_after != state_before:
        raise RuntimeError("Model state changed during counterfactual")
    input_after = v7_diag.verify_inputs_unchanged(
        artifacts["paths"],
        artifacts["input_sha256"],
    )
    (
        registry_payload_after,
        registry_after,
        registry_seal_after,
    ) = _registry_source(job)
    binding_after = _job_binding(
        job,
        artifacts,
        registry_payload_after,
        registry_after,
        registry_seal_after,
        output_dir,
    )
    if binding_after != job_binding:
        raise RuntimeError("Job/registry binding changed during evaluation")
    payload = json_ready(
        {
            "schema": JOB_SCHEMA,
            "status": "complete",
            "variant": v7_variant,
            "v8_variant": v8_variant,
            "seed": seed,
            "checkpoint_role": job["role"],
            "checkpoint": str(Path(job["checkpoint"]).resolve()),
            "checkpoint_sha256": file_sha256(Path(job["checkpoint"])),
            "job_binding": job_binding,
            "strict_load_v7": True,
            "strict_load_v8": True,
            "state_layout_equal": True,
            "all_gate_probabilities_from_production_forward": True,
            "validation_count": len(validation_ids),
            "validation_ids": validation_ids,
            "ordered_validation_ids_sha256": validation_ids_digest,
            "v8_model_metadata": v8_metadata,
            "device": dict(device_provenance),
            "numeric_audit": {
                "raw_outputs": {
                    **raw_output_audit,
                    "shifted_outputs_all_finite": True,
                    "diagnostic_production_forward_within_frozen_tolerance": (
                        True
                    ),
                    "diagnostic_forward_atol": DIAGNOSTIC_FORWARD_ATOL,
                    "diagnostic_forward_rtol": DIAGNOSTIC_FORWARD_RTOL,
                    "diagnostic_forward_max_abs_difference": (
                        diagnostic_forward_max_abs
                    ),
                    "diagnostic_forward_max_allowed_difference": (
                        diagnostic_forward_max_allowed
                    ),
                    "diagnostic_forward_check_count": (
                        diagnostic_forward_check_count
                    ),
                    "all_finite": True,
                },
                "losses": {
                    "v7": losses_v7,
                    "v8": losses_v8,
                    "count_per_model": len(validation_ids),
                    "all_finite": True,
                },
                "correlations_all_finite": True,
                "block_values_all_finite": True,
            },
            "coverage": {
                "validation_images_expected": len(validation_ids),
                "validation_images_processed": len(
                    processed_validation_ids
                ),
                "processed_ordered_ids_sha256": (
                    ordered_validation_ids_sha256(
                        processed_validation_ids
                    )
                ),
                "target_pixel_count": target_pixel_count,
                "target_image_count": target_image_count,
                "hard_negative_pixel_count": hard_negative_pixel_count,
                "hard_negative_image_count": hard_negative_image_count,
                "block_count": len(block_reports),
                "block_image_evaluations": sum(
                    int(record["image_count"])
                    for record in block_reports.values()
                ),
                "shift_image_count": min(
                    SHIFT_SUBSET_COUNT,
                    len(validation_ids),
                ),
                "shift_error_count_per_model": expected_shift_count,
                "operating_point_count": len(points),
                "complete": True,
            },
            "correction_selectivity": {
                **correction_totals,
                "target_mean_abs": target_mean,
                "hard_negative_mean_abs": hard_negative_mean,
                "target_correction_lift": target_lift,
                "correction_keep_correlation": _correlation(
                    correlation_totals
                ),
                "correlation_sufficient_statistics": correlation_totals,
                "blocks": block_reports,
                "hard_negative_definition": (
                    "V7 probability>0.5 components disjoint from "
                    "radius-3 GT dilation"
                ),
            },
            "topology_aggregate": topology_aggregate,
            "operating_points": points,
            "toroidal_grid_offset_stress": {
                "definition": "toroidal_grid_offset_stress",
                "translation_equivariance_claim_permitted": False,
                "all_probabilities_from_production_forward": True,
                "subset_ids": validation_ids[:SHIFT_SUBSET_COUNT],
                "offsets": [list(offset) for offset in SHIFT_OFFSETS],
                "crop_pixels": SHIFT_CROP,
                "v7_errors": shift_v7,
                "v8_errors": shift_v8,
            },
            "finite": True,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "input_sha256_before": artifacts["input_sha256"],
            "input_sha256_after": input_after,
            "formal_inputs_unchanged": True,
            "training_performed": False,
            "checkpoint_reselection_permitted": False,
            "official_test_accessed": False,
        }
    )
    _require_all_numeric_finite(payload, "counterfactual job payload")
    return payload


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Median input coverage is empty")
    ready = [
        _require_real(
            value,
            f"median values[{index}]",
        )
        for index, value in enumerate(values)
    ]
    array = np.asarray(ready, dtype=np.float64)
    _require_all_numeric_finite(array, "median values")
    return float(np.median(array))


def _validate_device_identity(device: Any, location: Path) -> None:
    if not isinstance(device, Mapping):
        raise ValueError(f"Counterfactual device identity absent: {location}")
    physical_index = _require_integer(
        device.get("physical_gpu_index"),
        "device.physical_gpu_index",
    )
    index_text = str(physical_index)
    expected_uuid = v7_diag.POSTPROCESS_GPUS.get(index_text)
    if (
        device.get("device") != "cuda:0"
        or device.get("logical_device") != "cuda:0"
        or physical_index not in (2, 3)
        or device.get("physical_gpu_uuid") != expected_uuid
        or device.get("visible_device_name") != "NVIDIA GeForce RTX 5090"
    ):
        raise ValueError(f"Counterfactual GPU identity differs: {location}")
    determinism = device.get("determinism")
    expected_determinism = {
        "cublas_workspace_config": ":4096:8",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
    }
    if not isinstance(determinism, Mapping) or any(
        determinism.get(key) != value
        for key, value in expected_determinism.items()
    ):
        raise ValueError(
            f"Counterfactual determinism identity differs: {location}"
        )


def _validate_v8_model_metadata(
    metadata: Any,
    v8_variant: str,
    location: Path,
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"V8 model metadata absent: {location}")
    variant_spec = json_ready(clean_v8_mprs_dch_variant_spec(v8_variant))
    required = {
        "variant": v8_variant,
        "candidate_family": "spd_anchored_tpd_clean_v8_mprs_dch",
        "mainline_contract": "Keep-Context-Saliency",
        "saliency_formula": (
            "S_p=(max_q(Z_q)-C0)+(Z_p-C0)/3"
        ),
        "standard_forward_conv2d_calls_per_block": 3,
        "shallow_embedding_parameters": 66_176,
        "total_parameters": 10_843_155,
        "context_gate": variant_spec["context_gate"],
        "fusion_formula": variant_spec["fusion_formula"],
        "variant_spec": variant_spec,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"V8 model metadata differs at {key}: {location}"
            )
    for key in (
        "full_initialization_sha256",
        "shared_initialization_sha256",
    ):
        _require_sha256(metadata.get(key), f"V8 metadata {key}")


def _validate_job_payload(
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
) -> Dict[str, Any]:
    """Revalidate a saved job against the current exact formal inputs."""

    location = job_output_path(output_dir, job)
    if (
        payload.get("schema") != JOB_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("formal_inputs_unchanged") is not True
        or payload.get("training_performed") is not False
        or payload.get("checkpoint_reselection_permitted") is not False
        or payload.get("official_test_accessed") is not False
        or payload.get("strict_load_v7") is not True
        or payload.get("strict_load_v8") is not True
        or payload.get("state_layout_equal") is not True
        or payload.get(
            "all_gate_probabilities_from_production_forward"
        )
        is not True
        or payload.get("finite") is not True
    ):
        raise ValueError(f"Incomplete counterfactual job output: {location}")
    artifacts = v7_diag.validate_job_artifacts(job)
    registry_payload, registry, registry_seal = _registry_source(job)
    expected_binding = _job_binding(
        job,
        artifacts,
        registry_payload,
        registry,
        registry_seal,
        output_dir,
    )
    if payload.get("job_binding") != expected_binding:
        raise ValueError(f"Counterfactual job binding differs: {location}")
    expected_identity = {
        "variant": str(job["variant"]),
        "v8_variant": V8_VARIANT_BY_V7[str(job["variant"])],
        "seed": int(job["seed"]),
        "checkpoint_role": str(job["role"]),
        "checkpoint": str(Path(job["checkpoint"]).resolve()),
        "checkpoint_sha256": expected_binding["expected_job"][
            "checkpoint_sha256"
        ],
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Counterfactual job identity differs at {key}: {location}"
            )
    _validate_device_identity(payload.get("device"), location)
    _validate_v8_model_metadata(
        payload.get("v8_model_metadata"),
        expected_identity["v8_variant"],
        location,
    )
    before = payload.get("input_sha256_before")
    after = payload.get("input_sha256_after")
    if (
        before != expected_binding["formal_input_sha256"]
        or after != before
    ):
        raise ValueError(f"Counterfactual formal hashes differ: {location}")
    states_before = payload.get("model_state_sha256_before")
    states_after = payload.get("model_state_sha256_after")
    expected_state_sha256 = expected_binding["expected_job"][
        "checkpoint_state_dict_sha256"
    ]
    if (
        not isinstance(states_before, Mapping)
        or states_after != states_before
        or states_before.get("v7") != states_before.get("v8")
        or states_before.get("v7") != expected_state_sha256
    ):
        raise ValueError(f"Counterfactual model states differ: {location}")
    for key in ("v7", "v8"):
        _require_sha256(
            states_before.get(key),
            f"counterfactual model state {key}",
        )

    validation_ids = expected_binding["ordered_validation"]["ids"]
    validation_count = len(validation_ids)
    validation_digest = expected_binding["ordered_validation"][
        "ordered_ids_sha256"
    ]
    if (
        payload.get("validation_ids") != validation_ids
        or payload.get("validation_count") != validation_count
        or payload.get("ordered_validation_ids_sha256")
        != validation_digest
    ):
        raise ValueError(f"Counterfactual validation order differs: {location}")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError(f"Counterfactual coverage is absent: {location}")
    expected_shift_count = min(
        SHIFT_SUBSET_COUNT,
        validation_count,
    ) * (len(SHIFT_OFFSETS) - 1)
    coverage_expected = {
        "validation_images_expected": validation_count,
        "validation_images_processed": validation_count,
        "processed_ordered_ids_sha256": validation_digest,
        "block_count": len(EXPECTED_BLOCK_NAMES),
        "block_image_evaluations": (
            len(EXPECTED_BLOCK_NAMES) * validation_count
        ),
        "shift_image_count": min(
            SHIFT_SUBSET_COUNT,
            validation_count,
        ),
        "shift_error_count_per_model": expected_shift_count,
        "operating_point_count": len(registry),
        "complete": True,
    }
    for key, expected in coverage_expected.items():
        observed = coverage.get(key)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = (
                _require_integer(
                    observed,
                    f"coverage.{key}",
                    minimum=0,
                )
                == expected
            )
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"Counterfactual coverage differs at {key}: {location}"
            )
    for key in (
        "target_pixel_count",
        "target_image_count",
        "hard_negative_pixel_count",
        "hard_negative_image_count",
    ):
        if _require_integer(
            coverage.get(key),
            f"coverage.{key}",
            minimum=1,
        ) <= 0:
            raise ValueError(
                f"Counterfactual coverage is empty at {key}: {location}"
            )

    numeric_audit = payload.get("numeric_audit")
    if not isinstance(numeric_audit, Mapping):
        raise ValueError(f"Counterfactual numeric audit absent: {location}")
    raw_outputs = numeric_audit.get("raw_outputs")
    losses = numeric_audit.get("losses")
    if (
        not isinstance(raw_outputs, Mapping)
        or raw_outputs.get("all_finite") is not True
        or raw_outputs.get("shifted_outputs_all_finite") is not True
        or raw_outputs.get(
            "diagnostic_production_forward_within_frozen_tolerance"
        )
        is not True
        or raw_outputs.get("diagnostic_forward_atol")
        != DIAGNOSTIC_FORWARD_ATOL
        or raw_outputs.get("diagnostic_forward_rtol")
        != DIAGNOSTIC_FORWARD_RTOL
        or _require_integer(
            raw_outputs.get("diagnostic_forward_check_count"),
            "raw_outputs.diagnostic_forward_check_count",
            minimum=1,
        )
        != len(EXPECTED_BLOCK_NAMES) * validation_count
        or _require_integer(
            raw_outputs.get("v7_tensor_count"),
            "raw_outputs.v7_tensor_count",
            minimum=1,
        )
        != 6 * validation_count
        or _require_integer(
            raw_outputs.get("v8_tensor_count"),
            "raw_outputs.v8_tensor_count",
            minimum=1,
        )
        != 6 * validation_count
        or _require_integer(
            raw_outputs.get("v7_element_count"),
            "raw_outputs.v7_element_count",
            minimum=1,
        )
        <= 0
        or _require_integer(
            raw_outputs.get("v8_element_count"),
            "raw_outputs.v8_element_count",
            minimum=1,
        )
        <= 0
        or not isinstance(losses, Mapping)
        or losses.get("all_finite") is not True
        or losses.get("count_per_model") != validation_count
        or len(losses.get("v7", ())) != validation_count
        or len(losses.get("v8", ())) != validation_count
        or numeric_audit.get("correlations_all_finite") is not True
        or numeric_audit.get("block_values_all_finite") is not True
    ):
        raise ValueError(f"Counterfactual numeric coverage differs: {location}")
    diagnostic_max_abs = _require_real(
        raw_outputs.get("diagnostic_forward_max_abs_difference"),
        "raw_outputs.diagnostic_forward_max_abs_difference",
        minimum=0.0,
    )
    diagnostic_max_allowed = _require_real(
        raw_outputs.get("diagnostic_forward_max_allowed_difference"),
        "raw_outputs.diagnostic_forward_max_allowed_difference",
        minimum=0.0,
    )
    if diagnostic_max_abs > diagnostic_max_allowed:
        raise ValueError(
            f"Diagnostic/production tolerance evidence differs: {location}"
        )
    for side in ("v7", "v8"):
        for index, value in enumerate(losses[side]):
            _require_real(
                value,
                f"numeric_audit.losses.{side}[{index}]",
                minimum=0.0,
            )

    correction = payload.get("correction_selectivity")
    if not isinstance(correction, Mapping):
        raise ValueError(f"Counterfactual correction audit absent: {location}")
    target_count = _require_integer(
        correction.get("target_count"),
        "correction.target_count",
        minimum=1,
    )
    negative_count = _require_integer(
        correction.get("hard_negative_count"),
        "correction.hard_negative_count",
        minimum=1,
    )
    if target_count <= 0 or negative_count <= 0:
        raise ValueError(f"Counterfactual correction masks empty: {location}")
    blocks = correction.get("blocks")
    if not isinstance(blocks, Mapping) or tuple(blocks) != EXPECTED_BLOCK_NAMES:
        raise ValueError(f"Counterfactual block report differs: {location}")
    block_target_count = 0
    block_negative_count = 0
    block_target_sum = 0.0
    block_negative_sum = 0.0
    block_overlap_removed_count = 0
    for key in (
        "target_sum",
        "target_mean_abs",
        "hard_negative_sum",
        "hard_negative_mean_abs",
        "target_correction_lift",
    ):
        _require_real(
            correction.get(key),
            f"correction.{key}",
            minimum=0.0,
        )
    _require_integer(
        correction.get("pooled_overlap_removed_count"),
        "correction.pooled_overlap_removed_count",
        minimum=0,
    )
    for name in EXPECTED_BLOCK_NAMES:
        block = blocks[name]
        if not isinstance(block, Mapping):
            raise ValueError(f"Invalid block report {name}: {location}")
        if (
            _require_integer(
                block.get("image_count"),
                f"{name}.image_count",
                minimum=1,
            )
            != validation_count
            or block.get("image_coverage_complete") is not True
            or _require_integer(
                block.get("target_count"),
                f"{name}.target_count",
                minimum=1,
            )
            <= 0
            or _require_integer(
                block.get("hard_negative_count"),
                f"{name}.hard_negative_count",
                minimum=1,
            )
            <= 0
            or block.get("target_priority_masks_disjoint") is not True
        ):
            raise ValueError(f"Incomplete block report {name}: {location}")
        for key in (
            "target_sum",
            "target_mean_abs",
            "hard_negative_sum",
            "hard_negative_mean_abs",
            "target_correction_lift",
            "mean_abs_correction",
            "mean_abs_scale",
        ):
            _require_real(
                block.get(key),
                f"{name}.{key}",
                minimum=0.0,
            )
        _require_integer(
            block.get("pooled_overlap_removed_count"),
            f"{name}.pooled_overlap_removed_count",
            minimum=0,
        )
        block_diagnostic_max = _require_real(
            block.get("diagnostic_forward_max_abs_difference"),
            f"{name}.diagnostic_forward_max_abs_difference",
            minimum=0.0,
        )
        block_diagnostic_allowed = _require_real(
            block.get("diagnostic_forward_max_allowed_difference"),
            f"{name}.diagnostic_forward_max_allowed_difference",
            minimum=0.0,
        )
        if (
            block.get(
                "diagnostic_production_forward_within_frozen_tolerance"
            )
            is not True
            or block_diagnostic_max > block_diagnostic_allowed
        ):
            raise ValueError(
                f"Block diagnostic/production tolerance failed: {name}"
            )
        expected_correlation = _correlation(
            block["correlation_sufficient_statistics"]
        )
        if not math.isclose(
            float(block["correction_keep_correlation"]),
            expected_correlation,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Block correlation differs at {name}: {location}")
        block_target_count += _require_integer(
            block["target_count"],
            f"{name}.target_count",
            minimum=1,
        )
        block_negative_count += _require_integer(
            block["hard_negative_count"],
            f"{name}.hard_negative_count",
            minimum=1,
        )
        block_target_sum += _require_real(
            block["target_sum"],
            f"{name}.target_sum",
            minimum=0.0,
        )
        block_negative_sum += _require_real(
            block["hard_negative_sum"],
            f"{name}.hard_negative_sum",
            minimum=0.0,
        )
        block_overlap_removed_count += _require_integer(
            block["pooled_overlap_removed_count"],
            f"{name}.pooled_overlap_removed_count",
            minimum=0,
        )
    if (
        block_target_count != target_count
        or block_negative_count != negative_count
        or block_overlap_removed_count
        != _require_integer(
            correction["pooled_overlap_removed_count"],
            "correction.pooled_overlap_removed_count",
            minimum=0,
        )
        or not math.isclose(
            block_target_sum,
            _require_real(
                correction["target_sum"],
                "correction.target_sum",
                minimum=0.0,
            ),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            block_negative_sum,
            _require_real(
                correction["hard_negative_sum"],
                "correction.hard_negative_sum",
                minimum=0.0,
            ),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(f"Block/global correction totals differ: {location}")
    expected_correlation = _correlation(
        correction["correlation_sufficient_statistics"]
    )
    if not math.isclose(
        float(correction["correction_keep_correlation"]),
        expected_correlation,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Global correction correlation differs: {location}")

    points = payload.get("operating_points")
    if not isinstance(points, Mapping) or tuple(points) != tuple(registry):
        raise ValueError(f"Counterfactual registry coverage differs: {location}")
    for key, registration in registry.items():
        point = points[key]
        if not isinstance(point, Mapping):
            raise ValueError(f"Invalid operating point {key}: {location}")
        for field in ("threshold", "registry_labels", "registry_kinds"):
            if point.get(field) != registration[field]:
                raise ValueError(
                    f"Operating-point registry differs at {key}/{field}"
                )
        for side in ("v7", "v8"):
            view = point.get(side)
            if (
                not isinstance(view, Mapping)
                or _require_integer(
                    view.get("target_count"),
                    f"{key}.{side}.target_count",
                    minimum=1,
                )
                <= 0
            ):
                raise ValueError(
                    f"Topology coverage absent at {key}/{side}: {location}"
                )
            for metric, lower, upper in (
                ("pd", 0.0, 1.0),
                ("fa", 0.0, None),
                ("miou", 0.0, 1.0),
                ("largest_fragment_fraction_mean", 0.0, 1.0),
                ("largest_fragment_fraction_p10", 0.0, 1.0),
            ):
                _require_real(
                    view.get(metric),
                    f"{key}.{side}.{metric}",
                    minimum=lower,
                    maximum=upper,
                )
            for count_key in (
                "matched_target_count",
                "unmatched_predicted_object_count",
                "fragment_excess_total",
                "overlap_covered_gt_count",
            ):
                _require_integer(
                    view.get(count_key),
                    f"{key}.{side}.{count_key}",
                    minimum=0,
                )
            fractions = view.get("largest_fragment_fractions")
            if not isinstance(fractions, list):
                raise ValueError(
                    f"Topology fractions absent at {key}/{side}: {location}"
                )
            for index, value in enumerate(fractions):
                _require_real(
                    value,
                    f"{key}.{side}.largest_fragment_fractions[{index}]",
                    minimum=0.0,
                    maximum=1.0,
                )
    topology = payload.get("topology_aggregate")
    recomputed_topology = topology_aggregate_from_operating_points(points)
    if not isinstance(topology, Mapping) or dict(topology) != recomputed_topology:
        raise ValueError(
            f"Counterfactual topology aggregate differs from points: {location}"
        )
    shift = payload.get("toroidal_grid_offset_stress")
    if (
        not isinstance(shift, Mapping)
        or shift.get("definition") != "toroidal_grid_offset_stress"
        or shift.get("translation_equivariance_claim_permitted") is not False
        or shift.get("all_probabilities_from_production_forward") is not True
        or shift.get("subset_ids")
        != validation_ids[:SHIFT_SUBSET_COUNT]
        or shift.get("offsets")
        != [list(offset) for offset in SHIFT_OFFSETS]
        or shift.get("crop_pixels") != SHIFT_CROP
        or len(shift.get("v7_errors", ())) != expected_shift_count
        or len(shift.get("v8_errors", ())) != expected_shift_count
    ):
        raise ValueError(f"Counterfactual shift coverage differs: {location}")
    for side in ("v7_errors", "v8_errors"):
        for index, value in enumerate(shift[side]):
            _require_real(
                value,
                f"toroidal_grid_offset_stress.{side}[{index}]",
                minimum=0.0,
            )
    _require_all_numeric_finite(payload, f"counterfactual job {location}")
    return dict(payload)


def _aggregate_block_reports(
    selected: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not selected:
        raise ValueError("Cannot aggregate blocks from an empty group")
    totals: Dict[str, Dict[str, Any]] = {}
    for name in EXPECTED_BLOCK_NAMES:
        totals[name] = {
            "target_sum": 0.0,
            "target_count": 0,
            "hard_negative_sum": 0.0,
            "hard_negative_count": 0,
            "pooled_overlap_removed_count": 0,
            "mean_abs_correction_sum": 0.0,
            "mean_abs_scale_sum": 0.0,
            "image_count": 0,
            "correlation_sufficient_statistics": {
                "count": 0,
                "sum_correction": 0.0,
                "sum_keep_linear": 0.0,
                "sum_sq_correction": 0.0,
                "sum_sq_keep_linear": 0.0,
                "sum_product": 0.0,
            },
            "diagnostic_forward_max_abs_difference": 0.0,
            "diagnostic_forward_max_allowed_difference": 0.0,
        }
    for item in selected:
        blocks = item["correction_selectivity"]["blocks"]
        for name in EXPECTED_BLOCK_NAMES:
            source = blocks[name]
            destination = totals[name]
            image_count = int(source["image_count"])
            destination["target_sum"] += float(source["target_sum"])
            destination["target_count"] += int(source["target_count"])
            destination["hard_negative_sum"] += float(
                source["hard_negative_sum"]
            )
            destination["hard_negative_count"] += int(
                source["hard_negative_count"]
            )
            destination["pooled_overlap_removed_count"] += (
                _require_integer(
                    source.get("pooled_overlap_removed_count"),
                    f"{name}.pooled_overlap_removed_count",
                    minimum=0,
                )
            )
            destination["mean_abs_correction_sum"] += (
                float(source["mean_abs_correction"]) * image_count
            )
            destination["mean_abs_scale_sum"] += (
                float(source["mean_abs_scale"]) * image_count
            )
            destination["image_count"] += image_count
            _add_correlation_stats(
                destination["correlation_sufficient_statistics"],
                source["correlation_sufficient_statistics"],
            )
            destination[
                "diagnostic_forward_max_abs_difference"
            ] = max(
                destination[
                    "diagnostic_forward_max_abs_difference"
                ],
                _require_real(
                    source.get(
                        "diagnostic_forward_max_abs_difference"
                    ),
                    (
                        f"{name}."
                        "diagnostic_forward_max_abs_difference"
                    ),
                    minimum=0.0,
                ),
            )
            destination[
                "diagnostic_forward_max_allowed_difference"
            ] = max(
                destination[
                    "diagnostic_forward_max_allowed_difference"
                ],
                _require_real(
                    source.get(
                        "diagnostic_forward_max_allowed_difference"
                    ),
                    (
                        f"{name}."
                        "diagnostic_forward_max_allowed_difference"
                    ),
                    minimum=0.0,
                ),
            )
    expected_images = sum(
        int(item["validation_count"]) for item in selected
    )
    reports = _finalize_block_reports(totals, expected_images)
    for report in reports.values():
        report["checkpoint_role_count"] = len(selected)
    return reports


def aggregate(
    results_root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    require_analysis_output_separate(results_root, output_dir)
    report_path = (
        Path(output_dir) / "tpd_clean_v8_mprs_counterfactual.json"
    )
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    jobs = expected_jobs(results_root)
    payloads: list[Dict[str, Any]] = []
    job_output_sha256: Dict[str, str] = {}
    for job in jobs:
        path = job_output_path(output_dir, job)
        payload, payload_sha256 = load_json_with_sha256(path)
        job_output_sha256[str(path.resolve())] = payload_sha256
        payloads.append(
            _validate_job_payload(
                job,
                payload,
                output_dir=output_dir,
            )
        )
    validation_digests = {
        payload["ordered_validation_ids_sha256"] for payload in payloads
    }
    execution_digests = {
        payload["job_binding"]["counterfactual_execution_sha256"]
        for payload in payloads
    }
    validation_data_digests = {
        payload["job_binding"]["current_validation_data_sha256"]
        for payload in payloads
    }
    protocol_digests = {
        payload["job_binding"]["v8_protocol_sha256"]
        for payload in payloads
    }
    amendment_digests = {
        payload["job_binding"]["v8_preflight_amendment_sha256"]
        for payload in payloads
    }
    if (
        len(payloads) != EXPECTED_JOB_COUNT
        or len(validation_digests) != 1
        or len(execution_digests) != 1
        or len(validation_data_digests) != 1
        or len(protocol_digests) != 1
        or len(amendment_digests) != 1
    ):
        raise ValueError(
            "Counterfactual jobs mix validation/source/data identities"
        )

    group_results: Dict[str, Any] = {}
    all_groups_pass = True
    for v7_variant, v8_variant in V8_VARIANT_BY_V7.items():
        for seed in v7_diag.SEEDS:
            selected = [
                payload
                for payload in payloads
                if payload["variant"] == v7_variant
                and int(payload["seed"]) == int(seed)
            ]
            expected_roles = set(v7_diag.CHECKPOINT_SPECS)
            selected_roles = {
                str(item["checkpoint_role"]) for item in selected
            }
            if len(selected) != 3 or selected_roles != expected_roles:
                raise ValueError("Counterfactual group must contain 3 roles")
            target_sum = sum(
                float(item["correction_selectivity"]["target_sum"])
                for item in selected
            )
            target_count = sum(
                int(item["correction_selectivity"]["target_count"])
                for item in selected
            )
            negative_sum = sum(
                float(
                    item["correction_selectivity"][
                        "hard_negative_sum"
                    ]
                )
                for item in selected
            )
            negative_count = sum(
                int(
                    item["correction_selectivity"][
                        "hard_negative_count"
                    ]
                )
                for item in selected
            )
            if target_count <= 0 or negative_count <= 0:
                raise ValueError("Counterfactual group mask coverage is empty")
            target_mean = target_sum / target_count
            negative_mean = negative_sum / negative_count
            lift = target_mean / (negative_mean + 1e-6)
            fragments_v7 = sum(
                int(
                    item["topology_aggregate"][
                        "v7_fragment_excess_total"
                    ]
                )
                for item in selected
            )
            fragments_v8 = sum(
                int(
                    item["topology_aggregate"][
                        "v8_fragment_excess_total"
                    ]
                )
                for item in selected
            )
            reference_covered_v7 = sum(
                _require_integer(
                    item["topology_aggregate"][
                        "v7_covered_reference_gt_count"
                    ],
                    "group.v7_covered_reference_gt_count",
                    minimum=1,
                )
                for item in selected
            )
            reference_covered_v8 = sum(
                _require_integer(
                    item["topology_aggregate"][
                        "v8_covered_reference_gt_count"
                    ],
                    "group.v8_covered_reference_gt_count",
                    minimum=0,
                )
                for item in selected
            )
            fractions_v7 = [
                float(value)
                for item in selected
                for value in item["topology_aggregate"][
                    "v7_largest_fragment_fractions"
                ]
            ]
            fractions_v8 = [
                float(value)
                for item in selected
                for value in item["topology_aggregate"][
                    "v8_largest_fragment_fractions"
                ]
            ]
            shift_v7 = [
                float(value)
                for item in selected
                for value in item["toroidal_grid_offset_stress"][
                    "v7_errors"
                ]
            ]
            shift_v8 = [
                float(value)
                for item in selected
                for value in item["toroidal_grid_offset_stress"][
                    "v8_errors"
                ]
            ]
            if not shift_v7 or not shift_v8:
                raise ValueError("Counterfactual group shift coverage is empty")
            mean_shift_v7 = float(np.mean(shift_v7))
            mean_shift_v8 = float(np.mean(shift_v8))
            if mean_shift_v7 <= 0.0:
                raise ValueError("V7 toroidal shift-stress mean is zero")
            shift_ratio = mean_shift_v8 / mean_shift_v7
            block_reports = _aggregate_block_reports(selected)
            target_pass = lift > TARGET_LIFT_FLOOR
            fragment_pass = fragments_v8 <= fragments_v7
            largest_pass = _median(fractions_v8) >= _median(fractions_v7)
            reference_coverage_pass = (
                reference_covered_v8 >= reference_covered_v7
            )
            shift_pass = shift_ratio <= SHIFT_RATIO_CEILING
            finite_pass = all(item["finite"] is True for item in selected)
            group_pass = (
                target_pass
                and fragment_pass
                and largest_pass
                and reference_coverage_pass
                and shift_pass
                and finite_pass
            )
            all_groups_pass = all_groups_pass and group_pass
            group_record = {
                "checkpoint_roles": sorted(
                    str(item["checkpoint_role"]) for item in selected
                ),
                "checkpoint_bindings": [
                    {
                        "comparison_role": item["checkpoint_role"],
                        "checkpoint": item["checkpoint"],
                        "checkpoint_sha256": item["checkpoint_sha256"],
                        "job_binding_sha256": item["job_binding"][
                            "binding_sha256"
                        ],
                        "registry_points_sha256": item["job_binding"][
                            "registry"
                        ]["points_sha256"],
                        "registry_source_sha256": item["job_binding"][
                            "registry"
                        ]["source_sha256"],
                    }
                    for item in sorted(
                        selected,
                        key=lambda record: str(
                            record["checkpoint_role"]
                        ),
                    )
                ],
                "ordered_validation_ids_sha256": next(
                    iter(validation_digests)
                ),
                "target_mean_abs_correction": target_mean,
                "hard_negative_mean_abs_correction": negative_mean,
                "target_correction_lift": lift,
                "target_correction_lift_pass": target_pass,
                "v7_fragment_excess_total": fragments_v7,
                "v8_fragment_excess_total": fragments_v8,
                "fragment_excess_nonincrease_pass": fragment_pass,
                "v7_largest_fragment_median": _median(fractions_v7),
                "v8_largest_fragment_median": _median(fractions_v8),
                "largest_fragment_nondecrease_pass": largest_pass,
                "v7_covered_reference_gt_count": reference_covered_v7,
                "v8_covered_reference_gt_count": reference_covered_v8,
                "reference_coverage_nondecrease_pass": (
                    reference_coverage_pass
                ),
                "v7_shift_error_mean": mean_shift_v7,
                "v8_shift_error_mean": mean_shift_v8,
                "v8_v7_shift_ratio": shift_ratio,
                "shift_ratio_pass": shift_pass,
                "shift_stress_definition": (
                    "toroidal_grid_offset_stress"
                ),
                "blocks": block_reports,
                "coverage": {
                    "checkpoint_role_count": len(selected),
                    "validation_image_evaluations": sum(
                        int(item["validation_count"])
                        for item in selected
                    ),
                    "block_count": len(block_reports),
                    "complete": True,
                },
                "finite_pass": finite_pass,
                "group_pass": group_pass,
            }
            _require_all_numeric_finite(
                group_record,
                f"counterfactual group {v8_variant}/seed_{seed}",
            )
            group_results[f"{v8_variant}/seed_{seed}"] = group_record

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "candidate_family": "tpd_clean_v8_mprs_dch",
        "source_candidate_family": "tpd_clean_v7_dch",
        "dataset": DATASET,
        "job_count": len(payloads),
        "strict_load_count": sum(
            item.get("strict_load_v8") is True for item in payloads
        ),
        "finite_job_count": sum(item.get("finite") is True for item in payloads),
        "audit_hardening": {
            "job_binding_schema": JOB_BINDING_SCHEMA,
            "expected_job_count": EXPECTED_JOB_COUNT,
            "expected_block_names": list(EXPECTED_BLOCK_NAMES),
            "ordered_validation_ids_sha256": next(
                iter(validation_digests)
            ),
            "counterfactual_execution_sha256": next(
                iter(execution_digests)
            ),
            "validation_data_sha256": next(
                iter(validation_data_digests)
            ),
            "v8_protocol_sha256": next(iter(protocol_digests)),
            "v8_preflight_amendment_sha256": next(
                iter(amendment_digests)
            ),
            "formal_input_labels": list(FORMAL_INPUT_LABELS),
            "all_job_bindings_revalidated": True,
            "all_raw_outputs_losses_correlations_blocks_finite": True,
            "all_target_hard_negative_and_block_coverage_nonempty": True,
            "target_priority_pooled_masks_disjoint": True,
            "paired_topology_recomputed_from_per_gt": True,
            "reference_coverage_nondecrease_gate_included": True,
            "all_gate_probabilities_from_production_forward": True,
            "shift_interpretation": "toroidal_grid_offset_stress",
            "output_separate_from_formal_results_root": True,
        },
        "groups": group_results,
        "counterfactual_gate_pass": all_groups_pass,
        "training_performed": False,
        "checkpoint_reselection_permitted": False,
        "official_test_accessed": False,
        "decision_scope": (
            "amendment-v1 counterfactual-v2 formal800 investment gate "
            "only; does not replace Pd/Fa/mIoU Gates A-E"
        ),
        "job_outputs": [
            {
                "path": str(job_output_path(output_dir, job).resolve()),
                "sha256": job_output_sha256[
                    str(job_output_path(output_dir, job).resolve())
                ],
                "variant": job["variant"],
                "seed": job["seed"],
                "comparison_role": job["role"],
                "job_binding_sha256": payloads[index][
                    "job_binding"
                ]["binding_sha256"],
            }
            for index, job in enumerate(jobs)
        ],
    }
    _require_all_numeric_finite(report, "counterfactual aggregate report")
    process_execution_binding()
    for path_text, expected_sha256 in job_output_sha256.items():
        path = Path(path_text)
        if (
            not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != expected_sha256
        ):
            raise RuntimeError(
                f"Counterfactual job changed during aggregate: {path}"
            )
    write_json(
        report_path,
        report,
        overwrite=overwrite,
    )
    return report


def run_shard(args: argparse.Namespace) -> Dict[str, Any]:
    require_analysis_output_separate(args.results_root, args.output_dir)
    determinism = v7_diag.configure_dch_inference(args.device)
    device_provenance = v7_diag.bind_requested_device(
        args.device,
        args.physical_gpu,
    )
    device_provenance["determinism"] = determinism
    device = torch.device(args.device)
    selected = [
        job
        for index, job in enumerate(expected_jobs(args.results_root))
        if index % args.shard_count == args.shard_index
    ]
    outputs: list[Dict[str, Any]] = []
    for job in selected:
        artifacts = v7_diag.validate_job_artifacts(job)
        payload = evaluate_job(
            job,
            artifacts,
            device=device,
            device_provenance=device_provenance,
            output_dir=args.output_dir,
        )
        path = job_output_path(args.output_dir, job)
        write_json(path, payload, overwrite=args.overwrite)
        outputs.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "variant": job["variant"],
                "seed": job["seed"],
                "role": job["role"],
                "finite": payload["finite"],
            }
        )
        print(
            "V8_MPRS_COUNTERFACTUAL_JOB_COMPLETE "
            f"variant={job['variant']} seed={job['seed']} "
            f"role={job['role']} finite={payload['finite']}",
            flush=True,
        )
    shard = {
        "schema": SCHEMA,
        "mode": "run_shard",
        "status": "complete",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "job_count": len(outputs),
        "outputs": outputs,
        "counterfactual_execution_sha256": process_execution_binding()[
            "execution_sha256"
        ],
    }
    write_json(
        args.output_dir / f"shard_{args.shard_index}_of_{args.shard_count}.json",
        shard,
        overwrite=args.overwrite,
    )
    return shard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen V7-to-V8 MPRS counterfactual"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-gpu", choices=("2", "3"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.results_root = args.results_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must lie in [0, shard-count)")
    if args.run and (
        args.device != "cuda:0"
        or args.physical_gpu not in ("2", "3")
    ):
        parser.error(
            "--run requires --device cuda:0 and --physical-gpu 2 or 3"
        )
    if args.device == "cpu" and args.physical_gpu is not None:
        parser.error("--physical-gpu is only valid with cuda:0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        payload = preflight(args.results_root, args.output_dir)
    elif args.aggregate:
        payload = aggregate(
            args.results_root,
            args.output_dir,
            overwrite=args.overwrite,
        )
    else:
        payload = run_shard(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.preflight and not payload["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
