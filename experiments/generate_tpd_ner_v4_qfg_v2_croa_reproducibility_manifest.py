#!/usr/bin/env python3
"""Seal the terminal TPD+NER+QFG formal800 reproducibility evidence.

The generator is deliberately CPU-only and imports no training or model
module.  It validates every terminal input before creating anything.  A
missing terminal input exits with 75 and leaves no partial output.  Once all
inputs are complete, the JSON and Markdown files are staged together and the
whole directory is published with Linux ``RENAME_NOREPLACE`` semantics.

Two controller-authorized terminal families are supported:

``current``
    The C/D QFG closure is terminal and the controller explicitly records
    ``authoritative_action=no_fallback``.

``paired``
    The controller records ``paired_training_complete`` and the independent
    E/F DLR+ramp100 post-training selection and deployment are also complete.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
from string import Template
import sys
import tempfile
from typing import Any, Iterable, Mapping, NamedTuple, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "formal800_reproducibility_manifest_v1"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "formal800_reproducibility_manifest_action_v1"
)
CONTROLLER_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "fallback_receipt_v1"
)
DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
FORMAL_EPOCHS = 800
VALIDATION_SPLIT_SHA256 = (
    "86247e5970f93224c64005e1ac7f3a933bafb37baf279ab71fce5670ae925e06"
)
GPU_UUIDS = {
    2: "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    3: "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
CHECKPOINTS = {
    "best.pth.tar": ("best_validation_pd_primary", "best_epoch"),
    "best_miou.pth.tar": (
        "best_validation_miou_secondary",
        "best_miou_epoch",
    ),
}
SWEEP_NAMES = {
    "best.pth.tar": "pd_fa_sweep_best.pth.json",
    "best_miou.pth.tar": "pd_fa_sweep_best_miou.pth.json",
}
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
EXIT_PERMANENT = 64
EXIT_INCOMPLETE = 75

MAINLINE_SOURCES = (
    ("keep_context_saliency_tpd", "model/tpd_clean_v8_mprs_dch.py"),
    ("five_node_ner_core", "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py"),
    ("five_node_relay", "model/tpd_relay.py"),
    ("qfg_frequency_gate", "model/tpd_frequency_gate_v2_croa.py"),
    ("qfg_query_bridge", "model/tpd_query_frequency_bridge.py"),
    (
        "tpd_ner_qfg_tss_composite",
        "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
    ),
    ("training_only_tss", "model/tpd_survival.py"),
    ("forward_contract", "model/tpd_forward_contract.py"),
)

FROZEN_LOCK_DIGESTS = {
    "current_training_48": (
        "8d55464851db9441383854189eff64c05"
        "daf25e7ff3502c6c67cf06401996478"
    ),
    "current_posttraining_15": (
        "315f091b75078e65b871946cecae92893"
        "e8915bb3951b6fc4dcf3a52c984cbbd"
    ),
    "paired_training_51": (
        "88b4839b40484c881544614e60675c4d2"
        "805a4fd6de1cc2f0aad28bdcb1395e8"
    ),
    "paired_posttraining_v2_12": (
        "289e9dc3c03097dd96c2c5dbe6c637b"
        "0ed5b30bfcd3a3ad3c710b8f55497e2ab"
    ),
}


class IncompleteEvidence(RuntimeError):
    """A terminal producer has not published all required artifacts yet."""


class EvidenceConflict(RuntimeError):
    """Published evidence conflicts with the frozen terminal contract."""


class LockSpec(NamedTuple):
    name: str
    path: Path
    source_count: int
    expected_sha256: str | None


class RunSpec(NamedTuple):
    method_id: str
    variant: str
    run_dir: Path
    gpu_index: int
    gpu_uuid: str
    training_lock_name: str
    source_lock_key: str
    family: str


class ClosureSpec(NamedTuple):
    name: str
    selection_json: Path
    selection_markdown: Path
    deployment_manifest: Path
    deployment_artifact: Path
    closure_lock_name: str
    factorial_json: Path | None
    factorial_markdown: Path | None


class CommandSpec(NamedTuple):
    name: str
    source: Path
    argv: tuple[str, ...]


class EvidenceLayout(NamedTuple):
    repo_root: Path
    output_dir: Path
    template_path: Path
    controller_receipt: Path
    parent_checkpoint: Path
    locks: tuple[LockSpec, ...]
    current_runs: tuple[RunSpec, ...]
    paired_runs: tuple[RunSpec, ...]
    current_closure: ClosureSpec
    paired_closure: ClosureSpec
    commands: tuple[CommandSpec, ...]


def _path(root: Path, relative: str) -> Path:
    return root / PurePosixPath(relative)


def default_layout(
    *,
    repo_root: Path = REPO_ROOT,
    output_dir: Path | None = None,
    enforce_frozen_lock_digests: bool = True,
) -> EvidenceLayout:
    root = Path(repo_root).expanduser().resolve()
    current_root = _path(
        root,
        "experiments/results/"
        "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized/NUDT-SIRST",
    )
    paired_root = _path(
        root,
        "experiments/results/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_v1",
    )
    parent = _path(
        root,
        "experiments/results/"
        "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/"
        "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/"
        "seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar",
    )
    digest = (
        FROZEN_LOCK_DIGESTS
        if enforce_frozen_lock_digests
        else {name: None for name in FROZEN_LOCK_DIGESTS}
    )
    locks = (
        LockSpec(
            "current_training_48",
            _path(
                root,
                "experiments/"
                "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json",
            ),
            48,
            digest["current_training_48"],
        ),
        LockSpec(
            "current_posttraining_15",
            _path(
                root,
                "experiments/"
                "tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json",
            ),
            15,
            digest["current_posttraining_15"],
        ),
        LockSpec(
            "paired_training_51",
            _path(
                root,
                "experiments/"
                "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json",
            ),
            51,
            digest["paired_training_51"],
        ),
        LockSpec(
            "paired_posttraining_v2_12",
            _path(
                root,
                "experiments/"
                "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
                "posttraining_closure_source_lock_v2.json",
            ),
            12,
            digest["paired_posttraining_v2_12"],
        ),
    )
    current_runs = (
        RunSpec(
            "c_qfg_only",
            "qfg_only",
            current_root / "qfg_only/seed_42_formal800_qfg_only",
            2,
            GPU_UUIDS[2],
            "current_training_48",
            "tpd_ner_v4_qfg_v2_croa_exact_source_lock",
            "current",
        ),
        RunSpec(
            "d_tss_qfg",
            "tss_qfg",
            current_root / "tss_qfg/seed_42_formal800_tss_qfg",
            3,
            GPU_UUIDS[3],
            "current_training_48",
            "tpd_ner_v4_qfg_v2_croa_exact_source_lock",
            "current",
        ),
    )
    paired_runs = (
        RunSpec(
            "e_qfg_dlr",
            "qfg_dlr",
            paired_root
            / "qfg_dlr_lane/NUDT-SIRST/qfg_dlr/"
            "seed_42_formal800_qfg_dlr_control",
            2,
            GPU_UUIDS[2],
            "paired_training_51",
            "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock",
            "paired",
        ),
        RunSpec(
            "f_tss_qfg_dlr",
            "tss_qfg_dlr",
            paired_root
            / "tss_qfg_dlr_lane/NUDT-SIRST/tss_qfg_dlr/"
            "seed_42_formal800_tss_qfg_dlr_ramp100",
            3,
            GPU_UUIDS[3],
            "paired_training_51",
            "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock",
            "paired",
        ),
    )
    current_closure = ClosureSpec(
        "current_cd",
        current_root
        / "final_selection/"
        "tpd_ner_v4_qfg_v2_croa_formal800_final_selection.json",
        current_root
        / "final_selection/"
        "tpd_ner_v4_qfg_v2_croa_formal800_final_selection.md",
        current_root
        / "deployment/"
        "tpd_ner_v4_qfg_v2_croa_formal800_deployment_manifest.json",
        current_root
        / "deployment/"
        "tpd_ner_v4_qfg_v2_croa_formal800_inference.pth.tar",
        "current_posttraining_15",
        current_root
        / "comparison_factorial_v1/tss_qfg_v2_croa_factorial_seed42.json",
        current_root
        / "comparison_factorial_v1/tss_qfg_v2_croa_factorial_seed42.md",
    )
    paired_closure = ClosureSpec(
        "paired_ef",
        paired_root
        / "final_selection/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
        "formal800_final_selection.json",
        paired_root
        / "final_selection/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
        "formal800_final_selection.md",
        paired_root
        / "deployment/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
        "formal800_deployment_manifest.json",
        paired_root
        / "deployment/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
        "formal800_inference.pth.tar",
        "paired_posttraining_v2_12",
        None,
        None,
    )
    commands = (
        CommandSpec(
            "current_training_2x5090",
            _path(
                root,
                "experiments/"
                "run_tpd_ner_v4_qfg_v2_croa_formal800_2x5090_lane.sh",
            ),
            (
                "bash",
                "experiments/"
                "run_tpd_ner_v4_qfg_v2_croa_formal800_2x5090_lane.sh",
                "--freeze",
                "verify",
            ),
        ),
        CommandSpec(
            "current_finalize",
            _path(
                root,
                "experiments/"
                "finalize_tpd_ner_v4_qfg_v2_croa_formal800_2x5090.sh",
            ),
            (
                "bash",
                "experiments/"
                "finalize_tpd_ner_v4_qfg_v2_croa_formal800_2x5090.sh",
            ),
        ),
        CommandSpec(
            "fallback_controller",
            _path(
                root,
                "experiments/"
                "control_tpd_ner_v4_qfg_v2_croa_"
                "dlr_ramp100_fallback.py",
            ),
            (
                "/home/ly/BasicIRSTD/infrarenet/bin/python",
                "experiments/"
                "control_tpd_ner_v4_qfg_v2_croa_"
                "dlr_ramp100_fallback.py",
            ),
        ),
        CommandSpec(
            "paired_training_2x5090",
            _path(
                root,
                "experiments/"
                "launch_tpd_ner_v4_qfg_v2_croa_"
                "dlr_ramp100_formal800_2x5090.sh",
            ),
            (
                "bash",
                "experiments/"
                "launch_tpd_ner_v4_qfg_v2_croa_"
                "dlr_ramp100_formal800_2x5090.sh",
                "--source-lock-mode",
                "verify",
            ),
        ),
        CommandSpec(
            "paired_sweeps_2x5090",
            _path(
                root,
                "experiments/"
                "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
                "formal800_sweeps_2x5090.sh",
            ),
            (
                "bash",
                "experiments/"
                "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
                "formal800_sweeps_2x5090.sh",
            ),
        ),
        CommandSpec(
            "paired_final_selection",
            _path(
                root,
                "experiments/"
                "compare_tpd_ner_v4_qfg_v2_croa_dlr_ramp100.py",
            ),
            (
                "/home/ly/BasicIRSTD/infrarenet/bin/python",
                "experiments/"
                "compare_tpd_ner_v4_qfg_v2_croa_dlr_ramp100.py",
                "--publish",
            ),
        ),
        CommandSpec(
            "paired_deployment",
            _path(
                root,
                "experiments/"
                "deploy_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800.py",
            ),
            (
                "/home/ly/BasicIRSTD/infrarenet/bin/python",
                "experiments/"
                "deploy_tpd_ner_v4_qfg_v2_croa_"
                "dlr_ramp100_formal800.py",
            ),
        ),
    )
    return EvidenceLayout(
        repo_root=root,
        output_dir=(
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else current_root / "reproducibility_manifest_v1"
        ),
        template_path=_path(
            root,
            "experiments/"
            "tpd_ner_v4_qfg_v2_croa_reproducibility_manifest_template.md",
        ),
        controller_receipt=current_root
        / "fallback_control/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback_receipt.json",
        parent_checkpoint=parent,
        locks=locks,
        current_runs=current_runs,
        paired_runs=paired_runs,
        current_closure=current_closure,
        paired_closure=paired_closure,
        commands=commands,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceConflict(message)


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink():
        raise EvidenceConflict(f"{label} must not be a symlink: {value}")
    if not value.exists():
        raise IncompleteEvidence(f"{label} is not published yet: {value}")
    if not value.is_file():
        raise EvidenceConflict(f"{label} is not a regular file: {value}")
    return value


def _sha256_file(path: Path) -> str:
    value = _regular_file(path, "SHA-256 input")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical(value: Any) -> Any:
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


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = _regular_file(path, label)
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceConflict(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain one object")
    return payload


def _relative(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _binding(path: Path, root: Path, label: str) -> dict[str, Any]:
    value = _regular_file(path, label).resolve()
    return {
        "path": str(value),
        "repo_relative_path": _relative(value, root),
        "sha256": _sha256_file(value),
    }


def _assert_official_test_false(
    payload: Any,
    label: str,
    *,
    require_marker: bool = True,
) -> None:
    markers: list[tuple[str, Any]] = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                if key in {"official_test_accessed", "official_test_claim"}:
                    markers.append((child_location, child))
                visit(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(payload, label)
    if require_marker:
        _require(markers, f"{label} has no explicit official-test marker")
    for location, value in markers:
        _require(value is False, f"{location} must be exactly false")


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _path_matches(observed: Any, expected: Path, label: str) -> None:
    _require(
        isinstance(observed, (str, os.PathLike)),
        f"{label} path is missing",
    )
    _require(
        Path(observed).expanduser().resolve() == Path(expected).resolve(),
        f"{label} path differs",
    )


def _all_strings(value: Any) -> set[str]:
    result: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                result.add(str(key))
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return result


def _validate_source_lock(
    spec: LockSpec,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _regular_file(spec.path, f"{spec.name} source lock")
    raw = path.read_bytes()
    payload = _read_json(path, f"{spec.name} source lock")
    _require(
        _canonical_json_bytes(payload) == raw,
        f"{spec.name} source lock is not canonical JSON",
    )
    digest = hashlib.sha256(raw).hexdigest()
    if spec.expected_sha256 is not None:
        _require(
            digest == spec.expected_sha256,
            f"{spec.name} frozen SHA-256 differs",
        )
    _require(
        payload.get("source_count") == spec.source_count,
        f"{spec.name} source_count differs",
    )
    source_sha256 = payload.get("source_sha256")
    _require(
        isinstance(source_sha256, Mapping),
        f"{spec.name} source_sha256 is missing",
    )
    _require(
        len(source_sha256) == spec.source_count,
        f"{spec.name} source hash count differs",
    )
    verified: dict[str, str] = {}
    for relative, expected in source_sha256.items():
        _require(
            isinstance(relative, str) and relative,
            f"{spec.name} contains an invalid source path",
        )
        pure = PurePosixPath(relative)
        _require(
            not pure.is_absolute() and ".." not in pure.parts,
            f"{spec.name} source escapes repository: {relative}",
        )
        _require(
            _is_sha256(expected),
            f"{spec.name} source SHA is invalid: {relative}",
        )
        source = repo_root / pure
        observed = _sha256_file(source)
        _require(
            observed == expected,
            f"{spec.name} live source changed: {relative}",
        )
        verified[relative] = observed
    _assert_official_test_false(payload, f"{spec.name} source lock")
    record = {
        "name": spec.name,
        **_binding(path, repo_root, f"{spec.name} source lock"),
        "schema": payload.get("schema"),
        "lock_kind": payload.get("lock_kind"),
        "source_count": spec.source_count,
        "live_sources_verified": True,
        "official_test_accessed": False,
    }
    return payload, record


def _validate_lock_relations(
    payloads: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: Path,
    parent_checkpoint: Path,
) -> dict[str, Any]:
    current_training = payloads["current_training_48"]
    current_post = payloads["current_posttraining_15"]
    paired_training = payloads["paired_training_51"]
    paired_post = payloads["paired_posttraining_v2_12"]

    current_digest = records["current_training_48"]["sha256"]
    current_post_digest = records["current_posttraining_15"]["sha256"]
    paired_digest = records["paired_training_51"]["sha256"]

    _require(
        current_post.get("training_source_lock", {}).get("sha256")
        == current_digest,
        "15-source closure does not bind the 48-source training lock",
    )
    _require(
        paired_training.get("upstream_source_lock_sha256")
        == current_digest,
        "51-source paired lock does not bind the 48-source lock",
    )
    _require(
        paired_post.get("training_source_lock", {}).get("sha256")
        == paired_digest,
        "12-source paired closure does not bind the 51-source lock",
    )
    _require(
        paired_post.get("reference_closure_source_lock", {}).get("sha256")
        == current_post_digest,
        "12-source paired closure does not bind the 15-source closure",
    )

    parent = _regular_file(parent_checkpoint, "parent checkpoint").resolve()
    parent_sha256 = _sha256_file(parent)
    current_parent = current_training.get("parent_checkpoint")
    paired_parent = paired_training.get("parent_checkpoint")
    for label, binding in (
        ("48-source parent", current_parent),
        ("51-source parent", paired_parent),
    ):
        _require(isinstance(binding, Mapping), f"{label} binding is missing")
        observed_parent = Path(str(binding.get("path")))
        if not observed_parent.is_absolute():
            observed_parent = repo_root / observed_parent
        _path_matches(observed_parent, parent, label)
        _require(
            binding.get("sha256") == parent_sha256,
            f"{label} SHA-256 differs",
        )
        _require(binding.get("epoch") == 489, f"{label} epoch differs")
    for field in ("epoch", "path", "role", "sha256", "state_dict_sha256"):
        _require(
            current_parent.get(field) == paired_parent.get(field),
            f"current and paired parent {field} differs",
        )
    return {
        **_binding(parent, repo_root, "parent checkpoint"),
        "path": str(parent),
        "sha256": parent_sha256,
        "epoch": int(current_parent["epoch"]),
        "role": current_parent.get("role"),
        "state_dict_sha256": current_parent.get("state_dict_sha256"),
        "variant": current_parent.get("variant"),
        "used_only_as_independent_run_initialization": True,
        "parent_optimizer_inherited": False,
    }


def _validate_mainline_sources(
    *,
    repo_root: Path,
    lock_payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    current = lock_payloads["current_training_48"]["source_sha256"]
    paired = lock_payloads["paired_training_51"]["source_sha256"]
    records: list[dict[str, Any]] = []
    for component, relative in MAINLINE_SOURCES:
        _require(relative in current, f"48-source lock lacks {relative}")
        _require(relative in paired, f"51-source lock lacks {relative}")
        path = repo_root / PurePosixPath(relative)
        digest = _sha256_file(path)
        _require(current[relative] == digest, f"48-source digest differs: {relative}")
        _require(paired[relative] == digest, f"51-source digest differs: {relative}")
        records.append(
            {
                "component": component,
                **_binding(path, repo_root, component),
                "bound_by": [
                    "current_training_48",
                    "paired_training_51",
                ],
            }
        )
    return records


def _read_metrics_jsonl(path: Path, label: str) -> dict[str, Any]:
    value = _regular_file(path, label)
    epochs: list[int] = []
    with value.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvidenceConflict(
                    f"{label} contains an empty line at {line_number}"
                )
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceConflict(
                    f"{label} line {line_number} is invalid JSON: {error}"
                ) from error
            _require(
                isinstance(event, Mapping),
                f"{label} line {line_number} is not an object",
            )
            epoch = event.get("epoch")
            _require(
                isinstance(epoch, int) and not isinstance(epoch, bool),
                f"{label} line {line_number} lacks an integer epoch",
            )
            epochs.append(epoch)
    _require(
        epochs == list(range(1, FORMAL_EPOCHS + 1)),
        f"{label} is not the complete ordered 1..800 trajectory",
    )
    return {
        "path": str(value.resolve()),
        "sha256": _sha256_file(value),
        "epoch_count": len(epochs),
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],
    }


def _validate_sweep(
    path: Path,
    *,
    spec: RunSpec,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_role: str,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    payload = _read_json(path, f"{spec.method_id} sweep")
    _assert_official_test_false(payload, f"{spec.method_id} sweep")
    expected = {
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "variant": spec.variant,
        "checkpoint_role": checkpoint_role,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "evaluated_checkpoint_count": 1,
        "official_test_accessed": False,
    }
    for field, required in expected.items():
        _require(
            payload.get(field) == required,
            f"{spec.method_id} sweep {field} differs",
        )
    _path_matches(payload.get("run_directory"), spec.run_dir, "sweep run")
    _path_matches(payload.get("checkpoint"), checkpoint_path, "sweep checkpoint")
    fixed = payload.get("fixed_threshold_0_5")
    _require(isinstance(fixed, Mapping), "sweep fixed-threshold point is missing")
    _require(
        math.isclose(
            _finite(fixed.get("threshold"), "fixed threshold"),
            0.5,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "sweep fixed threshold differs from 0.5",
    )
    budgets = payload.get("best_points_under_fa_budget")
    _require(isinstance(budgets, Mapping), "sweep Fa-budget points are missing")
    _require(
        tuple(budgets.keys()) == BUDGET_KEYS
        or set(budgets.keys()) == set(BUDGET_KEYS),
        "sweep Fa-budget set differs",
    )
    points = payload.get("points")
    _require(isinstance(points, list) and points, "sweep point list is empty")
    return {
        **_binding(path, spec.run_dir, f"{spec.method_id} sweep"),
        "schema": payload.get("schema"),
        "checkpoint": checkpoint_path.name,
        "checkpoint_role": checkpoint_role,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "fixed_threshold": 0.5,
        "fa_budget_keys": list(BUDGET_KEYS),
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "official_test_accessed": False,
    }


def _validate_run(
    spec: RunSpec,
    *,
    repo_root: Path,
    training_lock_sha256: str,
    parent_binding: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = Path(spec.run_dir)
    if run_dir.is_symlink():
        raise EvidenceConflict(f"{spec.method_id} run directory is a symlink")
    if not run_dir.exists():
        raise IncompleteEvidence(
            f"{spec.method_id} formal800 run is not published yet: {run_dir}"
        )
    _require(run_dir.is_dir(), f"{spec.method_id} run path is not a directory")

    protocol_path = run_dir / "protocol.json"
    summary_path = run_dir / "summary.json"
    split_path = run_dir / "split.json"
    protocol = _read_json(protocol_path, f"{spec.method_id} protocol")
    summary = _read_json(summary_path, f"{spec.method_id} summary")
    split = _read_json(split_path, f"{spec.method_id} split")
    for label, payload in (
        ("protocol", protocol),
        ("summary", summary),
        ("split", split),
    ):
        _assert_official_test_false(payload, f"{spec.method_id} {label}")

    expected_summary = {
        "status": "complete",
        "variant": spec.variant,
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "official_test_accessed": False,
    }
    for field, required in expected_summary.items():
        _require(
            summary.get(field) == required,
            f"{spec.method_id} summary {field} differs",
        )
    formal_contract = summary.get("formal_contract")
    _require(
        isinstance(formal_contract, Mapping)
        and formal_contract.get("epochs") == FORMAL_EPOCHS,
        f"{spec.method_id} is not a formal800 summary",
    )

    arguments = protocol.get("arguments")
    _require(
        isinstance(arguments, Mapping),
        f"{spec.method_id} protocol arguments are missing",
    )
    for field, required in {
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": FORMAL_EPOCHS,
        "variant": spec.variant,
    }.items():
        _require(
            arguments.get(field) == required,
            f"{spec.method_id} protocol argument {field} differs",
        )
    _path_matches(
        arguments.get("parent_checkpoint"),
        Path(parent_binding["path"]),
        f"{spec.method_id} protocol parent",
    )

    protocol_parent = protocol.get("parent_checkpoint")
    _require(
        isinstance(protocol_parent, Mapping),
        f"{spec.method_id} protocol parent binding is missing",
    )
    _path_matches(
        protocol_parent.get("path"),
        Path(parent_binding["path"]),
        f"{spec.method_id} parent",
    )
    _require(
        protocol_parent.get("sha256") == parent_binding["sha256"],
        f"{spec.method_id} parent SHA differs",
    )
    _require(
        protocol_parent.get("epoch") == parent_binding["epoch"],
        f"{spec.method_id} parent epoch differs",
    )

    identity = protocol.get("run_identity")
    _require(isinstance(identity, Mapping), f"{spec.method_id} identity is missing")
    _require(
        _canonical(summary.get("run_identity")) == _canonical(identity),
        f"{spec.method_id} summary/protocol identity differs",
    )
    for field, required in {
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
    }.items():
        _require(
            identity.get(field) == required,
            f"{spec.method_id} identity {field} differs",
        )
    source_locks = identity.get("source_locks")
    _require(
        isinstance(source_locks, Mapping)
        and source_locks.get(spec.source_lock_key) == training_lock_sha256,
        f"{spec.method_id} training source-lock binding differs",
    )
    _require(
        source_locks.get("parent_checkpoint") == parent_binding["sha256"],
        f"{spec.method_id} identity parent binding differs",
    )
    environment = (
        identity.get("training_contract", {}).get("environment", {})
        if isinstance(identity.get("training_contract"), Mapping)
        else {}
    )
    _require(
        environment.get("physical_gpu_index") == spec.gpu_index,
        f"{spec.method_id} physical GPU index differs",
    )
    _require(
        environment.get("physical_gpu_uuid") == spec.gpu_uuid,
        f"{spec.method_id} physical GPU UUID differs",
    )
    _require(
        environment.get("visible_cuda_device_count") == 1,
        f"{spec.method_id} did not expose exactly one assigned GPU",
    )

    _require(
        split.get("dataset") == DATASET
        and split.get("split_seed") == SPLIT_SEED
        and split.get("official_test_accessed") is False,
        f"{spec.method_id} split contract differs",
    )
    _require(
        split.get("used_train_count") == 530
        and split.get("used_val_count") == 133,
        f"{spec.method_id} internal 530/133 split differs",
    )

    checkpoints: dict[str, dict[str, Any]] = {}
    sweeps: list[dict[str, Any]] = []
    for filename, (role, epoch_field) in CHECKPOINTS.items():
        checkpoint = run_dir / filename
        checkpoint_sha256 = _sha256_file(checkpoint)
        epoch = summary.get(epoch_field)
        if filename == "best.pth.tar" and epoch is None:
            epoch = summary.get("best_pd_epoch")
        _require(
            isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and 1 <= epoch <= FORMAL_EPOCHS,
            f"{spec.method_id} {filename} epoch is invalid",
        )
        summary_path_field = (
            "best_checkpoint"
            if filename == "best.pth.tar"
            else "best_miou_checkpoint"
        )
        if summary_path_field in summary:
            _path_matches(
                summary[summary_path_field],
                checkpoint,
                f"{spec.method_id} summary {filename}",
            )
        checkpoint_key = (
            "best" if filename == "best.pth.tar" else "best_miou"
        )
        summary_checkpoints = summary.get("checkpoints")
        if isinstance(summary_checkpoints, Mapping):
            entry = summary_checkpoints.get(checkpoint_key)
            _require(
                isinstance(entry, Mapping),
                f"{spec.method_id} summary lacks {checkpoint_key} checkpoint",
            )
            _path_matches(
                entry.get("path"),
                checkpoint,
                f"{spec.method_id} summary checkpoint",
            )
            _require(
                entry.get("sha256") == checkpoint_sha256
                and entry.get("epoch") == epoch
                and entry.get("role") == role,
                f"{spec.method_id} summary {checkpoint_key} binding differs",
            )
        checkpoints[checkpoint_key] = {
            **_binding(
                checkpoint,
                repo_root,
                f"{spec.method_id} {filename}",
            ),
            "filename": filename,
            "role": role,
            "epoch": epoch,
            "selected_from_own_run": True,
        }
        sweep = _validate_sweep(
            run_dir / SWEEP_NAMES[filename],
            spec=spec,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_role=role,
            checkpoint_epoch=epoch,
        )
        sweeps.append(sweep)

    metrics = _read_metrics_jsonl(
        run_dir / "metrics.jsonl",
        f"{spec.method_id} metrics",
    )
    return {
        "method_id": spec.method_id,
        "variant": spec.variant,
        "family": spec.family,
        "run_directory": str(run_dir.resolve()),
        "protocol": _binding(
            protocol_path,
            repo_root,
            f"{spec.method_id} protocol",
        ),
        "summary": {
            **_binding(
                summary_path,
                repo_root,
                f"{spec.method_id} summary",
            ),
            "schema": summary.get("schema"),
            "status": "complete",
        },
        "split": {
            **_binding(split_path, repo_root, f"{spec.method_id} split"),
            "training_count": 530,
            "validation_count": 133,
            "split_seed": SPLIT_SEED,
        },
        "metrics": metrics,
        "physical_gpu": {
            "index": spec.gpu_index,
            "uuid": spec.gpu_uuid,
            "logical_device": "cuda:0",
            "visible_device_count": 1,
        },
        "parent_checkpoint_sha256": parent_binding["sha256"],
        "training_source_lock_sha256": training_lock_sha256,
        "run_id": identity.get("run_id"),
        "run_identity_sha256": _canonical_sha256(identity),
        "trainer_arguments": _canonical(arguments),
        "checkpoints": checkpoints,
        "sweeps": sweeps,
        "sweep_count": len(sweeps),
        "own_best_and_best_miou": True,
        "official_test_accessed": False,
    }


def _validate_json_markdown_pair(
    json_path: Path,
    markdown_path: Path,
    *,
    repo_root: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(json_path, f"{label} JSON")
    markdown = _regular_file(markdown_path, f"{label} Markdown")
    return payload, {
        "json": _binding(json_path, repo_root, f"{label} JSON"),
        "markdown": _binding(markdown, repo_root, f"{label} Markdown"),
    }


def _validate_closure(
    spec: ClosureSpec,
    *,
    repo_root: Path,
    closure_lock_record: Mapping[str, Any],
    run_records: Sequence[Mapping[str, Any]],
    current_reference_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    factorial_payload: dict[str, Any] | None = None
    factorial_binding: dict[str, Any] | None = None
    if spec.factorial_json is not None or spec.factorial_markdown is not None:
        _require(
            spec.factorial_json is not None
            and spec.factorial_markdown is not None,
            f"{spec.name} factorial JSON/Markdown must be paired",
        )
        factorial_payload, factorial_binding = _validate_json_markdown_pair(
            spec.factorial_json,
            spec.factorial_markdown,
            repo_root=repo_root,
            label=f"{spec.name} factorial",
        )
        _assert_official_test_false(
            factorial_payload,
            f"{spec.name} factorial",
        )
        for field, required in {
            "status": "complete",
            "dataset": DATASET,
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "official_test_accessed": False,
        }.items():
            _require(
                factorial_payload.get(field) == required,
                f"{spec.name} factorial {field} differs",
            )

    selection, selection_binding = _validate_json_markdown_pair(
        spec.selection_json,
        spec.selection_markdown,
        repo_root=repo_root,
        label=f"{spec.name} final selection",
    )
    _assert_official_test_false(selection, f"{spec.name} final selection")
    for field, required in {
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "official_test_accessed": False,
    }.items():
        _require(
            selection.get(field) == required,
            f"{spec.name} final selection {field} differs",
        )
    deployment_selection = selection.get("deployment_selection")
    selected = (
        deployment_selection.get("selected")
        if isinstance(deployment_selection, Mapping)
        else None
    )
    _require(
        isinstance(selected, Mapping),
        f"{spec.name} final selection lacks one deployment point",
    )
    _require(
        selected.get("checkpoint_local_atomic_point") is True,
        f"{spec.name} selected point is not checkpoint-local",
    )
    _require(
        deployment_selection.get("cross_checkpoint_metric_stitching") is False,
        f"{spec.name} selection permits cross-checkpoint stitching",
    )
    selected_checkpoint = Path(str(selected.get("checkpoint_path"))).resolve()
    selected_checkpoint_sha256 = _sha256_file(selected_checkpoint)
    _require(
        selected.get("checkpoint_sha256") == selected_checkpoint_sha256,
        f"{spec.name} selected checkpoint SHA differs",
    )
    _finite(selected.get("threshold"), f"{spec.name} selected threshold")

    run_sweep_shas = [
        sweep["sha256"]
        for run in run_records
        for sweep in run["sweeps"]
    ]
    _require(
        len(run_sweep_shas) == 4,
        f"{spec.name} must consume exactly four own-checkpoint sweeps",
    )
    selection_strings = _all_strings(selection)
    for digest in run_sweep_shas:
        _require(
            digest in selection_strings,
            f"{spec.name} final selection does not bind sweep {digest}",
        )
    if factorial_payload is not None:
        factorial_strings = _all_strings(factorial_payload)
        for digest in run_sweep_shas:
            _require(
                digest in factorial_strings,
                f"{spec.name} factorial does not bind sweep {digest}",
            )
        _require(
            selection_binding["json"]["sha256"]
            != factorial_binding["json"]["sha256"],
            f"{spec.name} factorial and selection must be independent files",
        )
    if current_reference_record is not None:
        reference_shas = [
            current_reference_record[
                "posttraining_closure_source_lock_sha256"
            ],
            *current_reference_record["input_sweep_sha256"],
        ]
        for digest in reference_shas:
            _require(
                digest in selection_strings,
                "paired selection does not bind the live current closure "
                f"input {digest}",
            )

    deployment = _read_json(
        spec.deployment_manifest,
        f"{spec.name} deployment manifest",
    )
    _assert_official_test_false(
        deployment,
        f"{spec.name} deployment manifest",
    )
    for field, required in {
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "official_test_accessed": False,
        "selected_method_id": selected.get("method_id"),
    }.items():
        _require(
            deployment.get(field) == required,
            f"{spec.name} deployment {field} differs",
        )
    deployment_checkpoint = deployment.get("selected_checkpoint")
    _require(
        isinstance(deployment_checkpoint, Mapping)
        and deployment_checkpoint.get("checkpoint_sha256")
        == selected_checkpoint_sha256
        and deployment_checkpoint.get("checkpoint")
        == selected.get("checkpoint")
        and deployment_checkpoint.get("checkpoint_role")
        == selected.get("checkpoint_role"),
        f"{spec.name} deployment checkpoint differs from selection",
    )
    _path_matches(
        deployment_checkpoint.get("checkpoint_path"),
        selected_checkpoint,
        f"{spec.name} deployment selected checkpoint",
    )
    deployed_point = deployment.get("deployment_operating_point")
    _require(
        isinstance(deployed_point, Mapping)
        and _canonical(deployed_point.get("selected"))
        == _canonical(selected),
        f"{spec.name} deployment operating point differs from selection",
    )
    final_selection_binding = deployment.get("final_selection")
    _require(
        isinstance(final_selection_binding, Mapping),
        f"{spec.name} deployment final-selection binding is missing",
    )
    _path_matches(
        final_selection_binding.get("path"),
        spec.selection_json,
        f"{spec.name} deployment final selection",
    )
    _require(
        final_selection_binding.get("sha256")
        == selection_binding["json"]["sha256"],
        f"{spec.name} deployment final-selection SHA differs",
    )
    closure_binding = deployment.get("posttraining_closure_source_lock")
    _require(
        isinstance(closure_binding, Mapping)
        and closure_binding.get("sha256") == closure_lock_record["sha256"],
        f"{spec.name} deployment closure-lock binding differs",
    )

    artifact_binding = deployment.get("artifact")
    _require(
        isinstance(artifact_binding, Mapping),
        f"{spec.name} deployment artifact binding is missing",
    )
    _path_matches(
        artifact_binding.get("path"),
        spec.deployment_artifact,
        f"{spec.name} deployment artifact",
    )
    artifact_sha256 = _sha256_file(spec.deployment_artifact)
    _require(
        artifact_binding.get("sha256") == artifact_sha256,
        f"{spec.name} deployment artifact SHA differs",
    )
    deployment_manifest_binding = _binding(
        spec.deployment_manifest,
        repo_root,
        f"{spec.name} deployment manifest",
    )
    result = {
        "name": spec.name,
        "factorial": factorial_binding,
        "final_selection": selection_binding,
        "deployment_manifest": deployment_manifest_binding,
        "deployment_export": {
            **_binding(
                spec.deployment_artifact,
                repo_root,
                f"{spec.name} deployment export",
            ),
            "export_mode": deployment.get("export_mode"),
            "source_checkpoint_sha256": selected_checkpoint_sha256,
        },
        "selected_method_id": selected.get("method_id"),
        "selected_variant": selected.get("variant"),
        "selected_checkpoint": {
            "path": str(selected_checkpoint),
            "sha256": selected_checkpoint_sha256,
            "filename": selected.get("checkpoint"),
            "role": selected.get("checkpoint_role"),
            "epoch": selected.get("checkpoint_epoch"),
        },
        "selected_threshold": selected.get("threshold"),
        "posttraining_closure_source_lock_sha256": closure_lock_record[
            "sha256"
        ],
        "input_sweep_count": len(run_sweep_shas),
        "input_sweep_sha256": run_sweep_shas,
        "checkpoint_local_atomic_selection": True,
        "official_test_accessed": False,
    }
    return result


def _find_exact_binding(
    value: Any,
    *,
    expected_path: Path,
    expected_sha256: str,
) -> bool:
    expected = expected_path.resolve()
    if isinstance(value, Mapping):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and digest == expected_sha256:
            try:
                if Path(path).expanduser().resolve() == expected:
                    return True
            except OSError:
                pass
        return any(
            _find_exact_binding(
                child,
                expected_path=expected,
                expected_sha256=expected_sha256,
            )
            for child in value.values()
        )
    if isinstance(value, list):
        return any(
            _find_exact_binding(
                child,
                expected_path=expected,
                expected_sha256=expected_sha256,
            )
            for child in value
        )
    return False


def _validate_controller_receipt(
    path: Path,
    *,
    terminal_family: str,
    repo_root: Path,
    current_closure: ClosureSpec,
    current_closure_record: Mapping[str, Any],
    current_posttraining_lock: Mapping[str, Any],
    paired_training_lock: Mapping[str, Any],
    paired_training_command: Mapping[str, Any],
    paired_run_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = _read_json(path, "fallback controller receipt")
    _assert_official_test_false(receipt, "fallback controller receipt")
    _require(
        receipt.get("schema") == CONTROLLER_SCHEMA,
        "fallback controller receipt schema differs",
    )
    if receipt.get("status") != "complete":
        raise IncompleteEvidence(
            "fallback controller has not reached a terminal complete state"
        )
    _require(
        receipt.get("official_test_accessed") is False,
        "fallback controller receipt must explicitly forbid official test use",
    )
    phase = receipt.get("phase")
    action = receipt.get("authoritative_action")
    paired_required = receipt.get("paired_required")
    paired = receipt.get("paired")
    _require(isinstance(paired, Mapping), "receipt paired section is missing")
    paired_complete = paired.get("training_complete")
    _require(
        receipt.get("terminal_for_fallback_controller") is True,
        "complete receipt is not terminal for its controller",
    )
    _require(
        receipt.get("receipt_write_policy") == "atomic_state_transition",
        "receipt write policy differs",
    )
    if terminal_family == "current":
        _require(phase == "no_fallback", "current terminal phase differs")
        _require(
            action == "no_fallback",
            "current terminal action must be no_fallback",
        )
        _require(
            paired_required is False,
            "current terminal receipt unexpectedly requires paired training",
        )
        _require(
            receipt.get("terminal_for_reproducibility_manifest") is True,
            "no-fallback receipt is not terminal for reproducibility",
        )
        _require(
            paired.get("required") is False
            and paired.get("training_complete") is False
            and paired.get("posttraining_selection_complete") is False
            and paired.get("posttraining_deployment_complete") is False,
            "no-fallback paired receipt section differs",
        )
    elif terminal_family == "paired":
        _require(
            phase == "paired_training_complete",
            "paired controller phase is not terminal",
        )
        _require(
            action == "launch_paired",
            "paired terminal action must be launch_paired",
        )
        _require(
            paired_required is True and paired_complete is True,
            "paired training is not recorded complete",
        )
        _require(
            receipt.get("terminal_for_reproducibility_manifest") is False,
            "paired-training receipt must defer wider reproducibility sealing",
        )
        _require(
            paired.get("required") is True
            and paired.get("posttraining_selection_complete") is False
            and paired.get("posttraining_deployment_complete") is False
            and paired.get("formal_epochs") == FORMAL_EPOCHS
            and paired.get("training_seed") == TRAINING_SEED
            and paired.get("split_seed") == SPLIT_SEED,
            "paired receipt training contract differs",
        )
        paired_source_lock = paired.get("source_lock")
        _require(
            isinstance(paired_source_lock, Mapping)
            and paired_source_lock.get("sha256")
            == paired_training_lock["sha256"],
            "paired receipt 51-source lock differs",
        )
        launcher = receipt.get("launcher")
        _require(
            isinstance(launcher, Mapping)
            and launcher.get("invoked") is True
            and launcher.get("exit_status") == 0,
            "paired launcher completion binding differs",
        )
        _path_matches(
            launcher.get("path"),
            Path(paired_training_command["source"]["path"]),
            "paired launcher receipt",
        )
        _require(
            launcher.get("sha256")
            == paired_training_command["source"]["sha256"],
            "paired launcher receipt SHA differs",
        )
        _require(
            launcher.get("verified_regular_executable") is True
            and launcher.get("fixed_physical_gpus") == [2, 3]
            and launcher.get("wait_for_gpu_idle") is False
            and launcher.get("paired_flock") is True,
            "paired launcher fixed execution contract differs",
        )
        _require(
            isinstance(launcher.get("source_lock"), Mapping)
            and launcher["source_lock"].get("sha256")
            == paired_training_lock["sha256"],
            "paired launcher 51-source lock differs",
        )
        receipt_runs = paired.get("runs")
        _require(
            isinstance(receipt_runs, Mapping)
            and set(receipt_runs)
            == {run["method_id"] for run in paired_run_records},
            "paired receipt E/F run matrix differs",
        )
        for run in paired_run_records:
            receipt_run = receipt_runs[run["method_id"]]
            _require(
                isinstance(receipt_run, Mapping)
                and receipt_run.get("formal_epochs_complete")
                == FORMAL_EPOCHS,
                f"receipt {run['method_id']} completion differs",
            )
            files = receipt_run.get("files")
            _require(
                isinstance(files, Mapping),
                f"receipt {run['method_id']} files are missing",
            )
            for receipt_key, expected in (
                ("summary", run["summary"]),
                ("metrics", run["metrics"]),
                (
                    "pd_primary_checkpoint",
                    run["checkpoints"]["best"],
                ),
                (
                    "miou_secondary_checkpoint",
                    run["checkpoints"]["best_miou"],
                ),
            ):
                binding = files.get(receipt_key)
                _require(
                    isinstance(binding, Mapping)
                    and binding.get("sha256") == expected["sha256"],
                    f"receipt {run['method_id']} {receipt_key} differs",
                )
                _path_matches(
                    binding.get("path"),
                    Path(expected["path"]),
                    f"receipt {run['method_id']} {receipt_key}",
                )
    else:
        raise EvidenceConflict(f"unknown terminal family: {terminal_family}")

    expected_receipt_bindings = (
        (
            current_closure.selection_json,
            current_closure_record["final_selection"]["json"]["sha256"],
            "current final selection",
        ),
        (
            current_closure.deployment_manifest,
            current_closure_record["deployment_manifest"]["sha256"],
            "current deployment manifest",
        ),
        (
            current_closure.deployment_artifact,
            current_closure_record["deployment_export"]["sha256"],
            "current deployment artifact",
        ),
        (
            Path(current_posttraining_lock["path"]),
            current_posttraining_lock["sha256"],
            "current 15-source closure lock",
        ),
    )
    for expected_path, expected_sha256, label in expected_receipt_bindings:
        _require(
            _find_exact_binding(
                receipt,
                expected_path=expected_path,
                expected_sha256=expected_sha256,
            ),
            f"fallback controller receipt does not bind {label}",
        )
    return {
        **_binding(path, repo_root, "fallback controller receipt"),
        "schema": CONTROLLER_SCHEMA,
        "status": "complete",
        "phase": phase,
        "authoritative_action": action,
        "paired_required": paired_required,
        "paired_training_complete": paired_complete,
        "selected_method_id": receipt.get("selected_method_id"),
        "selected_candidate_status": receipt.get(
            "selected_candidate_status"
        ),
        "query_fg_stage_success": receipt.get("query_fg_stage_success"),
        "meaningful_overall_improvement_by_frozen_policy": receipt.get(
            "meaningful_overall_improvement_by_frozen_policy"
        ),
        "official_test_accessed": False,
    }


def _validate_commands(
    commands: Sequence[CommandSpec],
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for command in commands:
        _require(command.name not in records, f"duplicate command: {command.name}")
        _require(command.argv, f"{command.name} has an empty argv")
        records[command.name] = {
            "source": _binding(
                command.source,
                repo_root,
                f"{command.name} command source",
            ),
            "argv": list(command.argv),
            "working_directory": str(repo_root),
            "shell_string_for_display_only": " ".join(command.argv),
        }
    return records


def build_manifest(
    layout: EvidenceLayout,
    *,
    terminal_family: str,
) -> dict[str, Any]:
    _require(
        terminal_family in {"current", "paired"},
        "terminal family must be current or paired",
    )
    repo_root = Path(layout.repo_root).resolve()
    lock_payloads: dict[str, dict[str, Any]] = {}
    lock_records: dict[str, dict[str, Any]] = {}
    for spec in layout.locks:
        _require(spec.name not in lock_payloads, f"duplicate lock: {spec.name}")
        payload, record = _validate_source_lock(spec, repo_root=repo_root)
        lock_payloads[spec.name] = payload
        lock_records[spec.name] = record
    _require(
        set(lock_records)
        == {
            "current_training_48",
            "current_posttraining_15",
            "paired_training_51",
            "paired_posttraining_v2_12",
        },
        "the four required lock roles are incomplete",
    )
    parent = _validate_lock_relations(
        lock_payloads,
        lock_records,
        repo_root=repo_root,
        parent_checkpoint=layout.parent_checkpoint,
    )
    model_sources = _validate_mainline_sources(
        repo_root=repo_root,
        lock_payloads=lock_payloads,
    )
    commands = _validate_commands(layout.commands, repo_root=repo_root)
    _require(
        "paired_training_2x5090" in commands,
        "paired training command is missing",
    )

    current_runs = [
        _validate_run(
            spec,
            repo_root=repo_root,
            training_lock_sha256=lock_records[
                spec.training_lock_name
            ]["sha256"],
            parent_binding=parent,
        )
        for spec in layout.current_runs
    ]
    _require(
        len(current_runs) == 2
        and {run["method_id"] for run in current_runs}
        == {"c_qfg_only", "d_tss_qfg"},
        "current C/D run matrix differs",
    )
    _require(
        len({run["run_id"] for run in current_runs}) == 2,
        "C/D do not have independent run identities",
    )
    current_closure = _validate_closure(
        layout.current_closure,
        repo_root=repo_root,
        closure_lock_record=lock_records[
            layout.current_closure.closure_lock_name
        ],
        run_records=current_runs,
    )

    paired_runs: list[dict[str, Any]] = []
    paired_closure: dict[str, Any] | None = None
    if terminal_family == "paired":
        paired_runs = [
            _validate_run(
                spec,
                repo_root=repo_root,
                training_lock_sha256=lock_records[
                    spec.training_lock_name
                ]["sha256"],
                parent_binding=parent,
            )
            for spec in layout.paired_runs
        ]
        _require(
            len(paired_runs) == 2
            and {run["method_id"] for run in paired_runs}
            == {"e_qfg_dlr", "f_tss_qfg_dlr"},
            "paired E/F run matrix differs",
        )
        _require(
            len({run["run_id"] for run in paired_runs}) == 2,
            "E/F do not have independent run identities",
        )
        paired_closure = _validate_closure(
            layout.paired_closure,
            repo_root=repo_root,
            closure_lock_record=lock_records[
                layout.paired_closure.closure_lock_name
            ],
            run_records=paired_runs,
            current_reference_record=current_closure,
        )

    receipt = _validate_controller_receipt(
        layout.controller_receipt,
        terminal_family=terminal_family,
        repo_root=repo_root,
        current_closure=layout.current_closure,
        current_closure_record=current_closure,
        current_posttraining_lock=lock_records["current_posttraining_15"],
        paired_training_lock=lock_records["paired_training_51"],
        paired_training_command=commands["paired_training_2x5090"],
        paired_run_records=paired_runs,
    )
    _require(
        receipt["selected_method_id"] == current_closure["selected_method_id"],
        "controller selected method differs from current closure",
    )

    terminal_closure = (
        current_closure if terminal_family == "current" else paired_closure
    )
    _require(
        terminal_closure is not None,
        "terminal closure was not validated",
    )
    terminal_runs = (
        current_runs if terminal_family == "current" else paired_runs
    )
    terminal_sweeps = [
        sweep for run in terminal_runs for sweep in run["sweeps"]
    ]
    _require(
        len(terminal_sweeps) == 4,
        "terminal family must have exactly four own-checkpoint sweeps",
    )

    generator_path = Path(__file__).resolve()
    template_path = _regular_file(
        layout.template_path,
        "reproducibility Markdown template",
    ).resolve()
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "project": {
            "repo_root": str(repo_root),
            "dataset": DATASET,
            "scope": "single_seed_internal_validation",
            "official_training_split": "530_train_133_validation",
            "official_test_accessed": False,
        },
        "terminal_family": terminal_family,
        "model_design": {
            "tpd_mainline": "Keep-Context-Saliency",
            "ner_core": "five_node_3_plus_2",
            "qfg": "query_conditioned_frequency_gating_v2_croa",
            "tss": "training_only_target_survival_supervision",
            "tss_inference_enabled": False,
            "mainline_sources": model_sources,
        },
        "source_locks": {
            name: copy.deepcopy(record)
            for name, record in lock_records.items()
        },
        "training_contract": {
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "formal_epochs": FORMAL_EPOCHS,
            "validation_split_sha256": VALIDATION_SPLIT_SHA256,
            "physical_gpu_mapping": {
                "2": {
                    "uuid": GPU_UUIDS[2],
                    "assigned_methods": [
                        "c_qfg_only",
                        *(
                            ["e_qfg_dlr"]
                            if terminal_family == "paired"
                            else []
                        ),
                    ],
                },
                "3": {
                    "uuid": GPU_UUIDS[3],
                    "assigned_methods": [
                        "d_tss_qfg",
                        *(
                            ["f_tss_qfg_dlr"]
                            if terminal_family == "paired"
                            else []
                        ),
                    ],
                },
            },
            "only_physical_gpus_2_and_3": True,
            "parent_checkpoint": parent,
            "runs_are_independent_after_common_initialization": True,
            "official_test_accessed": False,
        },
        "runs": {
            "current": {
                run["method_id"]: copy.deepcopy(run)
                for run in current_runs
            },
            "paired": {
                run["method_id"]: copy.deepcopy(run)
                for run in paired_runs
            },
        },
        "evaluation_closures": {
            "current": current_closure,
            "paired": paired_closure,
        },
        "controller_receipt": receipt,
        "terminal_authority": {
            "family": terminal_family,
            "selection": copy.deepcopy(
                terminal_closure["final_selection"]
            ),
            "deployment_manifest": copy.deepcopy(
                terminal_closure["deployment_manifest"]
            ),
            "deployment_export": copy.deepcopy(
                terminal_closure["deployment_export"]
            ),
            "selected_method_id": terminal_closure[
                "selected_method_id"
            ],
            "selected_variant": terminal_closure["selected_variant"],
            "selected_checkpoint": copy.deepcopy(
                terminal_closure["selected_checkpoint"]
            ),
            "selected_threshold": terminal_closure[
                "selected_threshold"
            ],
            "own_best_best_miou_sweep_count": len(terminal_sweeps),
            "own_best_best_miou_sweep_sha256": [
                sweep["sha256"] for sweep in terminal_sweeps
            ],
            "checkpoint_local_atomic_selection": True,
        },
        "execution_commands": commands,
        "manifest_producer": {
            "generator": _binding(
                generator_path,
                repo_root,
                "reproducibility manifest generator",
            ),
            "markdown_template": _binding(
                template_path,
                repo_root,
                "reproducibility Markdown template",
            ),
            "output_mode": "atomic_directory_rename_noreplace",
            "write_once": True,
            "existing_identical_is_verify": True,
            "existing_conflict_is_refused": True,
        },
        "claim_boundary": {
            "official_test_accessed": False,
            "single_seed_only": True,
            "internal_validation_only": True,
            "cross_seed_stability_claim": False,
            "official_test_claim": False,
        },
    }
    _assert_official_test_false(payload, "final reproducibility manifest")
    return _canonical(payload)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_markdown(
    payload: Mapping[str, Any],
    *,
    template_path: Path,
) -> str:
    _require(payload.get("schema") == SCHEMA, "manifest schema differs")
    template_file = _regular_file(template_path, "Markdown template")
    template = Template(template_file.read_text(encoding="utf-8"))
    payload_sha256 = _canonical_sha256(payload)
    authority = payload["terminal_authority"]
    overview = "\n".join(
        [
            f"- 状态：`{payload['status']}`",
            f"- 终态分支：`{payload['terminal_family']}`",
            f"- 最终方法：`{authority['selected_method_id']}`",
            (
                "- 最终 checkpoint："
                f"`{authority['selected_checkpoint']['filename']}` / "
                f"`{authority['selected_checkpoint']['sha256']}`"
            ),
            f"- 最终阈值：`{authority['selected_threshold']}`",
            f"- JSON payload SHA-256：`{payload_sha256}`",
        ]
    )
    model_sources = _markdown_table(
        ("组件", "文件", "SHA-256"),
        (
            (
                source["component"],
                f"`{source['repo_relative_path']}`",
                f"`{source['sha256']}`",
            )
            for source in payload["model_design"]["mainline_sources"]
        ),
    )
    source_locks = _markdown_table(
        ("锁", "源数量", "SHA-256", "现场验证"),
        (
            (
                name,
                value["source_count"],
                f"`{value['sha256']}`",
                value["live_sources_verified"],
            )
            for name, value in payload["source_locks"].items()
        ),
    )
    contract = payload["training_contract"]
    training_contract = "\n".join(
        [
            f"- seed：`{contract['training_seed']}`",
            f"- split seed：`{contract['split_seed']}`",
            f"- epochs：`{contract['formal_epochs']}`",
            (
                "- validation split SHA-256："
                f"`{contract['validation_split_sha256']}`"
            ),
            (
                "- 父 checkpoint SHA-256："
                f"`{contract['parent_checkpoint']['sha256']}`"
            ),
            (
                "- GPU 2："
                f"`{contract['physical_gpu_mapping']['2']['uuid']}`；"
                "GPU 3："
                f"`{contract['physical_gpu_mapping']['3']['uuid']}`"
            ),
            "- 各 run 仅共享初始化起点，优化器、轨迹和 checkpoint 独立。",
        ]
    )
    run_rows: list[tuple[Any, ...]] = []
    for family, methods in payload["runs"].items():
        for method_id, run in methods.items():
            for checkpoint in ("best", "best_miou"):
                binding = run["checkpoints"][checkpoint]
                run_rows.append(
                    (
                        family,
                        method_id,
                        checkpoint,
                        binding["epoch"],
                        f"`{binding['sha256']}`",
                        run["physical_gpu"]["index"],
                        run["sweep_count"],
                    )
                )
    runs = _markdown_table(
        (
            "分支",
            "run",
            "own checkpoint",
            "epoch",
            "SHA-256",
            "GPU",
            "sweeps/run",
        ),
        run_rows,
    )
    closure_rows: list[tuple[Any, ...]] = []
    for family, closure in payload["evaluation_closures"].items():
        if closure is None:
            continue
        closure_rows.extend(
            [
                (
                    family,
                    "final_selection",
                    f"`{closure['final_selection']['json']['sha256']}`",
                ),
                (
                    family,
                    "deployment_manifest",
                    f"`{closure['deployment_manifest']['sha256']}`",
                ),
                (
                    family,
                    "deployment_export",
                    f"`{closure['deployment_export']['sha256']}`",
                ),
            ]
        )
        if closure["factorial"] is not None:
            closure_rows.append(
                (
                    family,
                    "factorial",
                    f"`{closure['factorial']['json']['sha256']}`",
                )
            )
    closure_outputs = _markdown_table(
        ("分支", "产物", "SHA-256"),
        closure_rows,
    )
    commands = _markdown_table(
        ("阶段", "源码 SHA-256", "argv"),
        (
            (
                name,
                f"`{command['source']['sha256']}`",
                f"`{' '.join(command['argv'])}`",
            )
            for name, command in payload["execution_commands"].items()
        ),
    )
    claim_boundary = "\n".join(
        [
            "- `official_test_accessed=false`。",
            "- 结论限定为 seed 42 的 NUDT-SIRST 官方训练集内部 530/133 划分。",
            "- 不建立跨 seed 稳定性或官方测试集结论。",
            "- 最终指标与阈值来自同一 checkpoint-local operating point。",
        ]
    )
    try:
        return template.substitute(
            overview=overview,
            model_sources=model_sources,
            source_locks=source_locks,
            training_contract=training_contract,
            runs=runs,
            closure_outputs=closure_outputs,
            commands=commands,
            claim_boundary=claim_boundary,
        )
    except KeyError as error:
        raise EvidenceConflict(
            f"Markdown template placeholder is unsupported: {error}"
        ) from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, content: bytes) -> None:
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise EvidenceConflict(
            "atomic directory publication requires Linux renameat2"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(destination)
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _bundle_bytes(
    payload: Mapping[str, Any],
    *,
    template_path: Path,
) -> dict[str, bytes]:
    return {
        "manifest.json": _canonical_json_bytes(payload),
        "manifest.md": render_markdown(
            payload,
            template_path=template_path,
        ).encode("utf-8"),
    }


def _verify_existing_bundle(
    output_dir: Path,
    expected: Mapping[str, bytes],
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.is_symlink():
        raise EvidenceConflict(f"manifest bundle is a symlink: {output}")
    if not output.exists():
        raise IncompleteEvidence(f"manifest bundle does not exist: {output}")
    _require(output.is_dir(), f"manifest bundle is not a directory: {output}")
    observed_names = {entry.name for entry in output.iterdir()}
    _require(
        observed_names == set(expected),
        "manifest bundle is partial or contains unexpected files",
    )
    bindings: dict[str, Any] = {}
    for filename, content in expected.items():
        path = _regular_file(output / filename, f"manifest {filename}")
        _require(
            path.read_bytes() == content,
            f"existing manifest output conflicts: {path}",
        )
        bindings[filename] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return bindings


def publish_bundle(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
    template_path: Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    expected = _bundle_bytes(payload, template_path=template_path)
    if output.exists() or output.is_symlink():
        bindings = _verify_existing_bundle(output, expected)
        return {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "verify",
            "writes_performed": False,
            "output_dir": str(output),
            "outputs": bindings,
            "payload_sha256": _canonical_sha256(payload),
            "write_once": True,
            "atomic_pair": True,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise EvidenceConflict(
            f"manifest parent is not a regular directory: {output.parent}"
        )
    stage = Path(
        tempfile.mkdtemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".staging",
        )
    )
    published = False
    try:
        for filename, content in expected.items():
            _write_fsynced(stage / filename, content)
        _fsync_directory(stage)
        try:
            _rename_noreplace(stage, output)
            published = True
        except FileExistsError:
            bindings = _verify_existing_bundle(output, expected)
            return {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "action": "verify",
                "writes_performed": False,
                "output_dir": str(output),
                "outputs": bindings,
                "payload_sha256": _canonical_sha256(payload),
                "write_once": True,
                "atomic_pair": True,
                "concurrent_identical_publication": True,
            }
        _fsync_directory(output.parent)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
    bindings = _verify_existing_bundle(output, expected)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "publish",
        "writes_performed": True,
        "output_dir": str(output),
        "outputs": bindings,
        "payload_sha256": _canonical_sha256(payload),
        "write_once": True,
        "atomic_pair": True,
    }


def execute(
    layout: EvidenceLayout,
    *,
    terminal_family: str,
    preflight: bool = False,
    verify: bool = False,
) -> dict[str, Any]:
    _require(not (preflight and verify), "preflight and verify are exclusive")
    payload = build_manifest(layout, terminal_family=terminal_family)
    expected = _bundle_bytes(payload, template_path=layout.template_path)
    if preflight:
        return {
            "schema": ACTION_SCHEMA,
            "status": "ready",
            "action": "preflight",
            "writes_performed": False,
            "output_dir": str(Path(layout.output_dir).resolve()),
            "payload_sha256": _canonical_sha256(payload),
            "json_sha256": hashlib.sha256(
                expected["manifest.json"]
            ).hexdigest(),
            "markdown_sha256": hashlib.sha256(
                expected["manifest.md"]
            ).hexdigest(),
            "terminal_family": terminal_family,
        }
    if verify:
        bindings = _verify_existing_bundle(layout.output_dir, expected)
        return {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "verify",
            "writes_performed": False,
            "output_dir": str(Path(layout.output_dir).resolve()),
            "outputs": bindings,
            "payload_sha256": _canonical_sha256(payload),
            "write_once": True,
            "atomic_pair": True,
        }
    return publish_bundle(
        payload,
        output_dir=layout.output_dir,
        template_path=layout.template_path,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terminal-family",
        choices=("current", "paired"),
        required=True,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--controller-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    layout = default_layout(output_dir=args.output_dir)
    if args.controller_receipt is not None:
        layout = layout._replace(
            controller_receipt=args.controller_receipt.expanduser().resolve()
        )
    try:
        result = execute(
            layout,
            terminal_family=args.terminal_family,
            preflight=args.preflight,
            verify=args.verify,
        )
    except IncompleteEvidence as error:
        print(
            json.dumps(
                {
                    "schema": ACTION_SCHEMA,
                    "status": "incomplete",
                    "action": "wait",
                    "writes_performed": False,
                    "reason": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_INCOMPLETE
    except EvidenceConflict as error:
        print(
            json.dumps(
                {
                    "schema": ACTION_SCHEMA,
                    "status": "conflict",
                    "action": "refuse",
                    "writes_performed": False,
                    "reason": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_PERMANENT
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "ACTION_SCHEMA",
    "CONTROLLER_SCHEMA",
    "ClosureSpec",
    "CommandSpec",
    "EvidenceConflict",
    "EvidenceLayout",
    "EXIT_INCOMPLETE",
    "EXIT_PERMANENT",
    "IncompleteEvidence",
    "LockSpec",
    "RunSpec",
    "SCHEMA",
    "build_manifest",
    "default_layout",
    "execute",
    "main",
    "publish_bundle",
    "render_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
