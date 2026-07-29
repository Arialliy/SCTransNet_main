#!/usr/bin/env python3
"""Publish and verify the selected formal800 deployment artifact and manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    export_tpd_ner_v4_qfg_v2_croa_to_inference as qfg_exporter,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as v4_evaluator,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as selector,
)
from experiments import (  # noqa: E402
    tpd_ner_v4_qfg_v2_croa_posttraining_policy as policy,
)


MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_manifest_v1"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_action_v1"
)
DEFAULT_SELECTION = selector.JSON_OUTPUT
DEFAULT_OUTPUT_DIR = selector.QFG_RESULT_ROOT / "deployment"
DEFAULT_ARTIFACT = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_formal800_inference.pth.tar"
)
DEFAULT_MANIFEST = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_formal800_deployment_manifest.json"
)
QFG_METHODS = frozenset(("c_qfg_only", "d_tss_qfg"))
V4_METHOD = "v4"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = policy.regular_file(path, label)
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace deployment output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory_descriptor = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    source_path = policy.regular_file(source, "native deployment source")
    destination_path = Path(destination).resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(
            f"refusing to replace deployment output: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with source_path.open("rb") as source_handle, os.fdopen(
            descriptor,
            "wb",
        ) as output_handle:
            shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.link(temporary, destination_path)
        directory_descriptor = os.open(str(destination_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_selection(
    selection_path: Path,
    closure_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(selection_path).resolve()
    report = _load_json(path, "formal final selection")
    live = selector.build_formal_report(
        posttraining_closure_binding=closure_binding,
    )
    _require(
        policy.canonical(report) == policy.canonical(live),
        "formal final selection conflicts with live validated inputs",
    )
    _require(report.get("status") == "complete", "final selection is incomplete")
    _require(
        report.get("posttraining_closure_source_lock") == dict(closure_binding),
        "final selection closure-lock binding differs",
    )
    deployment = report.get("deployment_selection")
    _require(
        isinstance(deployment, Mapping),
        "final selection has no deployment operating point",
    )
    selected_method = report.get("selection", {}).get("selected_method_id")
    selected = deployment.get("selected")
    _require(isinstance(selected, Mapping), "deployment selection is missing")
    _require(
        selected.get("method_id") == selected_method,
        "deployment method differs from final recipe",
    )
    checkpoint = Path(str(selected.get("checkpoint_path"))).resolve()
    _require(
        policy.sha256_file(checkpoint) == selected.get("checkpoint_sha256"),
        "deployment source checkpoint SHA differs",
    )
    return report, {
        "path": str(path),
        "sha256": policy.sha256_file(path),
        "schema": report.get("schema"),
    }


def _artifact_mode(method_id: str) -> str:
    if method_id in QFG_METHODS:
        return "strict_head_free_qfg_export"
    if method_id == V4_METHOD:
        return "write_once_native_v4_checkpoint_copy"
    raise ValueError(f"unsupported deployment method: {method_id!r}")


def _validate_artifact(
    artifact_path: Path,
    *,
    mode: str,
    source_checkpoint: Path,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = policy.regular_file(artifact_path, "deployment artifact").resolve()
    if mode == "strict_head_free_qfg_export":
        binding = qfg_exporter.validate_exported_qfg_checkpoint(
            artifact,
            expected_source_checkpoint=source_checkpoint,
        )
        _require(
            binding.get("source_checkpoint_sha256")
            == selected.get("checkpoint_sha256"),
            "QFG export source checkpoint binding differs",
        )
        _require(
            binding.get("source_checkpoint_role")
            == selected.get("checkpoint_role"),
            "QFG export source checkpoint role differs",
        )
        return binding
    source_sha256 = policy.sha256_file(source_checkpoint)
    artifact_sha256 = policy.sha256_file(artifact)
    _require(
        source_sha256 == selected.get("checkpoint_sha256"),
        "native V4 source checkpoint SHA differs",
    )
    _require(
        artifact_sha256 == source_sha256,
        "native V4 deployment copy differs from its selected checkpoint",
    )
    audit = v4_evaluator.validate_run_artifacts(
        source_checkpoint.parent,
        source_checkpoint.name,
    )
    for label, observed, expected in (
        ("checkpoint path", audit.get("run_directory"), str(source_checkpoint.parent)),
        ("checkpoint filename", audit.get("checkpoint_filename"), source_checkpoint.name),
        ("checkpoint SHA", audit.get("checkpoint_sha256"), source_sha256),
        ("checkpoint role", audit.get("checkpoint_role"), selected.get("checkpoint_role")),
        ("checkpoint epoch", audit.get("checkpoint_epoch"), selected.get("checkpoint_epoch")),
    ):
        _require(observed == expected, f"native V4 {label} differs")
    return {
        "schema": "sctransnet_native_v4_checkpoint_copy_v1",
        "path": str(artifact),
        "sha256": artifact_sha256,
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_role": selected.get("checkpoint_role"),
        "byte_identical_copy": True,
        "source_evaluator_validation": True,
    }


def _publish_artifact(
    artifact_path: Path,
    *,
    mode: str,
    source_checkpoint: Path,
) -> None:
    if mode == "strict_head_free_qfg_export":
        qfg_exporter.export_qfg_checkpoint(source_checkpoint, artifact_path)
    else:
        _atomic_copy(source_checkpoint, artifact_path)


def build_manifest(
    report: Mapping[str, Any],
    *,
    selection_binding: Mapping[str, Any],
    closure_binding: Mapping[str, Any],
    artifact_binding: Mapping[str, Any],
    export_mode: str,
) -> dict[str, Any]:
    deployment = report["deployment_selection"]
    selected = deployment["selected"]
    exporter_path = Path(qfg_exporter.__file__).resolve()
    deployer_path = Path(__file__).resolve()
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "selected_method_id": selected["method_id"],
        "selected_variant": selected["variant"],
        "selected_checkpoint": {
            key: copy.deepcopy(selected[key])
            for key in (
                "checkpoint",
                "checkpoint_role",
                "role_name",
                "checkpoint_epoch",
                "checkpoint_path",
                "checkpoint_sha256",
            )
        },
        "deployment_operating_point": copy.deepcopy(dict(deployment)),
        "export_mode": export_mode,
        "artifact": copy.deepcopy(dict(artifact_binding)),
        "final_selection": copy.deepcopy(dict(selection_binding)),
        "posttraining_closure_source_lock": copy.deepcopy(dict(closure_binding)),
        "exporter": {
            "path": str(exporter_path),
            "sha256": policy.sha256_file(exporter_path),
            "invoked": export_mode == "strict_head_free_qfg_export",
        },
        "deployer": {
            "path": str(deployer_path),
            "sha256": policy.sha256_file(deployer_path),
        },
        "cross_checkpoint_metric_stitching": False,
        "selected_point_is_checkpoint_local": True,
        "write_once": True,
        "overwrite_forbidden": True,
    }


def validate_deployment_closure(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    artifact_path: Path = DEFAULT_ARTIFACT,
    manifest_path: Path = DEFAULT_MANIFEST,
    closure_lock_path: Path | None = None,
) -> dict[str, Any]:
    _, closure_binding = policy.load_closure_lock(
        closure_lock_path,
        verify_sources=True,
    )
    report, selection_binding = _validate_selection(
        selection_path,
        closure_binding,
    )
    selected = report["deployment_selection"]["selected"]
    mode = _artifact_mode(str(selected["method_id"]))
    source_checkpoint = Path(str(selected["checkpoint_path"])).resolve()
    artifact_binding = _validate_artifact(
        artifact_path,
        mode=mode,
        source_checkpoint=source_checkpoint,
        selected=selected,
    )
    expected_manifest = build_manifest(
        report,
        selection_binding=selection_binding,
        closure_binding=closure_binding,
        artifact_binding=artifact_binding,
        export_mode=mode,
    )
    manifest = _load_json(manifest_path, "deployment manifest")
    _require(
        policy.canonical(manifest) == policy.canonical(expected_manifest),
        "deployment manifest conflicts with live validated closure",
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "verify",
        "artifact_path": str(Path(artifact_path).resolve()),
        "artifact_sha256": artifact_binding["sha256"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": policy.sha256_file(manifest_path),
        "selected_method_id": selected["method_id"],
        "selected_checkpoint_role": selected["checkpoint_role"],
        "selected_threshold": selected["threshold"],
        "posttraining_closure_source_lock_sha256": closure_binding["sha256"],
        "verified": True,
    }


def publish_deployment(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    artifact_path: Path = DEFAULT_ARTIFACT,
    manifest_path: Path = DEFAULT_MANIFEST,
    closure_lock_path: Path | None = None,
    preflight: bool = False,
) -> dict[str, Any]:
    _, closure_binding = policy.load_closure_lock(
        closure_lock_path,
        verify_sources=True,
    )
    report, selection_binding = _validate_selection(
        selection_path,
        closure_binding,
    )
    selected = report["deployment_selection"]["selected"]
    mode = _artifact_mode(str(selected["method_id"]))
    source_checkpoint = Path(str(selected["checkpoint_path"])).resolve()
    artifact = Path(artifact_path).resolve()
    manifest = Path(manifest_path).resolve()
    if manifest.exists() or manifest.is_symlink():
        _require(
            artifact.is_file() and not artifact.is_symlink(),
            "deployment manifest exists without a regular artifact",
        )
        return validate_deployment_closure(
            selection_path=selection_path,
            artifact_path=artifact,
            manifest_path=manifest,
            closure_lock_path=closure_lock_path,
        )
    if manifest.is_symlink():
        raise ValueError("deployment manifest must not be a symlink")
    if preflight:
        if artifact.exists() or artifact.is_symlink():
            _validate_artifact(
                artifact,
                mode=mode,
                source_checkpoint=source_checkpoint,
                selected=selected,
            )
        return {
            "schema": ACTION_SCHEMA,
            "status": "ready",
            "action": "preflight",
            "selected_method_id": selected["method_id"],
            "selected_checkpoint_role": selected["checkpoint_role"],
            "selected_threshold": selected["threshold"],
            "artifact_state": "existing" if artifact.exists() else "missing",
            "manifest_state": "missing",
            "writes_performed": False,
        }
    if not artifact.exists() and not artifact.is_symlink():
        _publish_artifact(
            artifact,
            mode=mode,
            source_checkpoint=source_checkpoint,
        )
    artifact_binding = _validate_artifact(
        artifact,
        mode=mode,
        source_checkpoint=source_checkpoint,
        selected=selected,
    )
    expected_manifest = build_manifest(
        report,
        selection_binding=selection_binding,
        closure_binding=closure_binding,
        artifact_binding=artifact_binding,
        export_mode=mode,
    )
    _atomic_create_bytes(manifest, policy.canonical_json_bytes(expected_manifest))
    verified = validate_deployment_closure(
        selection_path=selection_path,
        artifact_path=artifact,
        manifest_path=manifest,
        closure_lock_path=closure_lock_path,
    )
    verified["action"] = "publish"
    return verified


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--closure-source-lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    if args.preflight and args.verify:
        raise ValueError("--preflight and --verify are mutually exclusive")
    if args.verify:
        result = validate_deployment_closure(
            selection_path=args.selection,
            artifact_path=args.artifact,
            manifest_path=args.manifest,
            closure_lock_path=args.closure_source_lock,
        )
    else:
        result = publish_deployment(
            selection_path=args.selection,
            artifact_path=args.artifact,
            manifest_path=args.manifest,
            closure_lock_path=args.closure_source_lock,
            preflight=args.preflight,
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "DEFAULT_ARTIFACT",
    "DEFAULT_MANIFEST",
    "MANIFEST_SCHEMA",
    "build_manifest",
    "main",
    "publish_deployment",
    "validate_deployment_closure",
]


if __name__ == "__main__":
    main()
