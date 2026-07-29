#!/usr/bin/env python3
"""Publish/verify the unique artifact selected by the DLR+ramp100 closure."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    compare_tpd_ner_v4_qfg_v2_croa_dlr_ramp100 as selector,
)
from experiments import (  # noqa: E402
    export_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_to_inference as dlr_exporter,
)
from experiments import (  # noqa: E402
    export_tpd_ner_v4_qfg_v2_croa_to_inference as qfg_exporter,
)
from experiments import (  # noqa: E402
    export_tpd_ner_v4_survival_to_inference as survival_exporter,
)
from experiments import (  # noqa: E402
    tpd_ner_v4_qfg_v2_croa_dlr_ramp100_posttraining_policy as policy,
)


MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "deployment_manifest_v1"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "deployment_action_v1"
)
DEFAULT_SELECTION = selector.DEFAULT_JSON_OUTPUT
DEFAULT_OUTPUT_DIR = selector.DLR_RESULT_ROOT / "deployment"
DEFAULT_ARTIFACT = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_inference.pth.tar"
)
DEFAULT_MANIFEST = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "formal800_deployment_manifest.json"
)
DEFAULT_CLOSURE_SOURCE_LOCK = policy.DEFAULT_LOCK_PATH

NATIVE_METHODS = frozenset(("baseline", "v4"))
SURVIVAL_METHODS = frozenset(("a_control", "b_tss"))
QFG_METHODS = frozenset(("c_qfg_only", "d_tss_qfg"))
DLR_METHODS = frozenset(("e_qfg_dlr", "f_tss_qfg_dlr"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = policy.regular_file(path, label)
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain one object")
    return payload


def _atomic_create_bytes(path: Path, content: bytes) -> bool:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        _require(
            output.is_file()
            and not output.is_symlink()
            and output.read_bytes() == content,
            f"existing deployment output conflicts: {output}",
        )
        return False
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
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            _require(
                output.is_file()
                and not output.is_symlink()
                and output.read_bytes() == content,
                f"concurrent deployment output conflicts: {output}",
            )
            return False
        directory_descriptor = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _atomic_copy(source: Path, destination: Path) -> bool:
    source_path = policy.regular_file(source, "native deployment source")
    output = Path(destination).resolve()
    if output.exists() or output.is_symlink():
        _require(
            output.is_file()
            and not output.is_symlink()
            and policy.sha256_file(output) == policy.sha256_file(source_path),
            f"existing native deployment artifact conflicts: {output}",
        )
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
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
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            _require(
                output.is_file()
                and not output.is_symlink()
                and policy.sha256_file(output)
                == policy.sha256_file(source_path),
                f"concurrent native deployment artifact conflicts: {output}",
            )
            return False
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _validate_selection(
    selection_path: Path,
    closure_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(selection_path).resolve()
    report = _load_json(path, "DLR final selection")
    live = selector.build_formal_report(closure_binding=closure_binding)
    _require(
        policy.canonical(report) == policy.canonical(live),
        "DLR final selection conflicts with live validated inputs",
    )
    _require(report.get("status") == "complete", "selection is incomplete")
    _require(
        report.get("posttraining_closure_source_lock")
        == dict(closure_binding),
        "selection closure-lock binding differs",
    )
    deployment = report.get("deployment_selection")
    selected = (
        deployment.get("selected")
        if isinstance(deployment, Mapping)
        else None
    )
    _require(isinstance(selected, Mapping), "deployment selection is missing")
    _require(
        selected.get("method_id") == report.get("selected_method_id"),
        "selected method/deployment method differs",
    )
    checkpoint = Path(str(selected.get("checkpoint_path"))).resolve()
    _require(
        policy.sha256_file(checkpoint) == selected.get("checkpoint_sha256"),
        "selected checkpoint SHA differs",
    )
    return report, {
        "path": str(path),
        "sha256": policy.sha256_file(path),
        "schema": report.get("schema"),
    }


def artifact_mode(method_id: str) -> str:
    if method_id in DLR_METHODS:
        return "strict_head_free_dlr_qfg_export"
    if method_id in QFG_METHODS:
        return "strict_head_free_qfg_export"
    if method_id in SURVIVAL_METHODS:
        return "strict_head_free_survival_export"
    if method_id in NATIVE_METHODS:
        return "write_once_native_checkpoint_copy"
    raise ValueError(f"unsupported deployment method: {method_id!r}")


def _validate_survival_export(
    artifact_path: Path,
    *,
    source_checkpoint: Path,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = policy.regular_file(
        artifact_path,
        "Survival deployment artifact",
    ).resolve()
    content = artifact.read_bytes()
    payload = torch.load(
        io.BytesIO(content),
        map_location="cpu",
        weights_only=False,
    )
    _require(isinstance(payload, Mapping), "Survival export is not a mapping")
    _require(
        payload.get("schema") == survival_exporter.EXPORT_SCHEMA,
        "Survival export schema differs",
    )
    source_sha = policy.sha256_file(source_checkpoint)
    _require(
        payload.get("source_checkpoint_path")
        == str(source_checkpoint.resolve())
        and payload.get("source_checkpoint_sha256") == source_sha,
        "Survival export source binding differs",
    )
    _require(
        source_sha == selected.get("checkpoint_sha256"),
        "Survival selected checkpoint SHA differs",
    )
    state_dict = payload.get("state_dict")
    _require(isinstance(state_dict, Mapping), "Survival export lacks state_dict")
    model, _ = survival_exporter.build_frozen_v4_model()
    incompatible = model.load_state_dict(state_dict, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "Survival export strict load is incompatible",
    )
    return {
        "schema": survival_exporter.EXPORT_SCHEMA,
        "path": str(artifact),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_role": selected.get("checkpoint_role"),
        "strict_load": True,
        "survival_state_absent": True,
        "qfg_state_preserved": False,
    }


def _validate_artifact(
    artifact_path: Path,
    *,
    mode: str,
    source_checkpoint: Path,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    if mode == "strict_head_free_dlr_qfg_export":
        return dlr_exporter.validate_exported_ramp100_qfg_checkpoint(
            artifact_path,
            expected_source_checkpoint=source_checkpoint,
            expected_variant=str(selected["variant"]),
        )
    if mode == "strict_head_free_qfg_export":
        return qfg_exporter.validate_exported_qfg_checkpoint(
            artifact_path,
            expected_source_checkpoint=source_checkpoint,
        )
    if mode == "strict_head_free_survival_export":
        return _validate_survival_export(
            artifact_path,
            source_checkpoint=source_checkpoint,
            selected=selected,
        )
    artifact = policy.regular_file(
        artifact_path,
        "native deployment artifact",
    ).resolve()
    source_sha = policy.sha256_file(source_checkpoint)
    artifact_sha = policy.sha256_file(artifact)
    _require(
        source_sha == selected.get("checkpoint_sha256"),
        "native selected checkpoint SHA differs",
    )
    _require(artifact_sha == source_sha, "native artifact is not byte-identical")
    return {
        "schema": "sctransnet_native_checkpoint_copy_v1",
        "path": str(artifact),
        "sha256": artifact_sha,
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_role": selected.get("checkpoint_role"),
        "byte_identical_copy": True,
    }


def _publish_artifact(
    artifact_path: Path,
    *,
    mode: str,
    source_checkpoint: Path,
    selected: Mapping[str, Any],
) -> bool:
    output = Path(artifact_path).resolve()
    if output.exists() or output.is_symlink():
        _validate_artifact(
            output,
            mode=mode,
            source_checkpoint=source_checkpoint,
            selected=selected,
        )
        return False
    if mode == "strict_head_free_dlr_qfg_export":
        dlr_exporter.export_ramp100_qfg_checkpoint(
            source_checkpoint,
            output,
            expected_variant=str(selected["variant"]),
        )
    elif mode == "strict_head_free_qfg_export":
        qfg_exporter.export_qfg_checkpoint(source_checkpoint, output)
    elif mode == "strict_head_free_survival_export":
        survival_exporter.export_survival_checkpoint(
            source_checkpoint,
            output,
        )
    else:
        _atomic_copy(source_checkpoint, output)
    return True


def build_manifest(
    report: Mapping[str, Any],
    *,
    selection_binding: Mapping[str, Any],
    closure_binding: Mapping[str, Any],
    artifact_binding: Mapping[str, Any],
    export_mode: str,
) -> dict[str, Any]:
    selected = report["deployment_selection"]["selected"]
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "dataset": policy.DATASET,
        "training_seed": policy.TRAINING_SEED,
        "split_seed": policy.SPLIT_SEED,
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
        "selected_threshold": selected["threshold"],
        "deployment_operating_point": copy.deepcopy(
            report["deployment_selection"]
        ),
        "export_mode": export_mode,
        "artifact": copy.deepcopy(dict(artifact_binding)),
        "final_selection": copy.deepcopy(dict(selection_binding)),
        "posttraining_closure_source_lock": copy.deepcopy(
            dict(closure_binding)
        ),
        "deployer": {
            "path": str(Path(__file__).resolve()),
            "sha256": policy.sha256_file(Path(__file__).resolve()),
        },
        "exporters": {
            "dlr_qfg": {
                "path": str(Path(dlr_exporter.__file__).resolve()),
                "sha256": policy.sha256_file(
                    Path(dlr_exporter.__file__).resolve()
                ),
                "invoked": export_mode == "strict_head_free_dlr_qfg_export",
            },
            "qfg": {
                "path": str(Path(qfg_exporter.__file__).resolve()),
                "sha256": policy.sha256_file(
                    Path(qfg_exporter.__file__).resolve()
                ),
                "invoked": export_mode == "strict_head_free_qfg_export",
            },
            "survival": {
                "path": str(Path(survival_exporter.__file__).resolve()),
                "sha256": policy.sha256_file(
                    Path(survival_exporter.__file__).resolve()
                ),
                "invoked": export_mode == "strict_head_free_survival_export",
            },
        },
        "cross_checkpoint_metric_stitching": False,
        "selected_point_is_checkpoint_local": True,
        "write_once": True,
        "idempotent_resume": True,
    }


def validate_deployment(
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
    mode = artifact_mode(str(selected["method_id"]))
    source = Path(str(selected["checkpoint_path"])).resolve()
    artifact_binding = _validate_artifact(
        artifact_path,
        mode=mode,
        source_checkpoint=source,
        selected=selected,
    )
    expected_manifest = build_manifest(
        report,
        selection_binding=selection_binding,
        closure_binding=closure_binding,
        artifact_binding=artifact_binding,
        export_mode=mode,
    )
    manifest = _load_json(manifest_path, "DLR deployment manifest")
    _require(
        policy.canonical(manifest) == policy.canonical(expected_manifest),
        "DLR deployment manifest conflicts with live closure",
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
        "verified": True,
        "writes_performed": False,
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
    mode = artifact_mode(str(selected["method_id"]))
    source = Path(str(selected["checkpoint_path"])).resolve()
    artifact = Path(artifact_path).resolve()
    manifest = Path(manifest_path).resolve()
    if manifest.exists() or manifest.is_symlink():
        _require(
            artifact.is_file() and not artifact.is_symlink(),
            "deployment manifest exists without a regular artifact",
        )
        return validate_deployment(
            selection_path=selection_path,
            artifact_path=artifact,
            manifest_path=manifest,
            closure_lock_path=closure_lock_path,
        )
    if preflight:
        if artifact.exists() or artifact.is_symlink():
            _validate_artifact(
                artifact,
                mode=mode,
                source_checkpoint=source,
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
    artifact_written = _publish_artifact(
        artifact,
        mode=mode,
        source_checkpoint=source,
        selected=selected,
    )
    artifact_binding = _validate_artifact(
        artifact,
        mode=mode,
        source_checkpoint=source,
        selected=selected,
    )
    expected_manifest = build_manifest(
        report,
        selection_binding=selection_binding,
        closure_binding=closure_binding,
        artifact_binding=artifact_binding,
        export_mode=mode,
    )
    manifest_written = _atomic_create_bytes(
        manifest,
        policy.canonical_json_bytes(expected_manifest),
    )
    verified = validate_deployment(
        selection_path=selection_path,
        artifact_path=artifact,
        manifest_path=manifest,
        closure_lock_path=closure_lock_path,
    )
    verified["action"] = "publish"
    verified["writes_performed"] = artifact_written or manifest_written
    verified["artifact_written"] = artifact_written
    verified["manifest_written"] = manifest_written
    return verified


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--closure-source-lock",
        type=Path,
        default=DEFAULT_CLOSURE_SOURCE_LOCK,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    _require(
        not (args.preflight and args.verify),
        "--preflight and --verify are mutually exclusive",
    )
    if args.verify:
        result = validate_deployment(
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
    "DEFAULT_CLOSURE_SOURCE_LOCK",
    "DEFAULT_MANIFEST",
    "MANIFEST_SCHEMA",
    "artifact_mode",
    "build_manifest",
    "main",
    "publish_deployment",
    "validate_deployment",
]


if __name__ == "__main__":
    main()
