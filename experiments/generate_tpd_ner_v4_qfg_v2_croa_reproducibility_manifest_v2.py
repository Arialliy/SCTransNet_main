#!/usr/bin/env python3
"""Seal the corrected QFG-V2 operational default and full formal800 evidence.

The immutable v1 selection remains the authority for choosing method D and
the exported weights.  The additive deployment-v2 profile is the authority
for the practical default operating point.  This generator rebuilds the full
formal800 evidence with the legacy generator, validates the deployment-v2
overlay, and publishes one write-once JSON/Markdown bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
    freeze_tpd_ner_v4_qfg_v2_croa_operational_closure_v2 as freezer,
)
from experiments import (  # noqa: E402
    generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest as legacy,
)
from experiments import (  # noqa: E402
    publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2 as publisher,
)


SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "formal800_reproducibility_manifest_v2"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "formal800_reproducibility_manifest_action_v2"
)
DEFAULT_OUTPUT_DIR = (
    publisher.selector.QFG_RESULT_ROOT / "reproducibility_manifest_v2"
)
DEFAULT_OPERATIONAL_LOCK = freezer.DEFAULT_OUTPUT_PATH
EXPECTED_METHOD = "d_tss_qfg"
EXPECTED_VARIANT = "tss_qfg"
EXPECTED_CHECKPOINT_SHA256 = publisher.EXPECTED_CHECKPOINT_SHA256
EXPECTED_THRESHOLD = 0.5
EXPECTED_METRICS = {
    "pd": 0.9947089947089947,
    "fa": 0.0000041301985432330825,
    "miou": 0.9370177924736262,
    "tiny_pd": 1.0,
    "false_objects_per_image": 0.03759398496240601,
    "unmatched_predicted_object_count": 5,
}
OUTPUT_NAMES = ("manifest.json", "manifest.md")
EXIT_CONFLICT = 64
EXIT_INCOMPLETE = 75


class IncompleteEvidence(RuntimeError):
    """A required terminal artifact has not been published yet."""


class EvidenceConflict(RuntimeError):
    """A published artifact conflicts with the v2 closure contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceConflict(message)


def _canonical(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise EvidenceConflict(f"value is not canonical JSON: {error}") from error


def _canonical_bytes(value: Any) -> bytes:
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
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.exists() and not value.is_symlink():
        raise IncompleteEvidence(f"{label} is missing: {value}")
    if value.is_symlink() or not value.is_file():
        raise EvidenceConflict(f"{label} must be a regular file: {value}")
    return value


def _regular_directory(path: Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_symlink() or not value.is_dir():
        raise EvidenceConflict(f"{label} must be a regular directory: {value}")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)

    def reject_constant(token: str) -> None:
        raise EvidenceConflict(f"{label} contains non-finite constant {token}")

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceConflict(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(payload, dict), f"{label} must contain one object")
    return payload


def _binding(
    path: Path,
    label: str,
    *,
    schema: Any | None = None,
) -> dict[str, Any]:
    source = _regular_file(path, label)
    result: dict[str, Any] = {
        "path": str(source),
        "sha256": _sha256(source),
    }
    if schema is not None:
        result["schema"] = schema
    return result


def _finite(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _close(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    return abs(_finite(left, "observed metric") - float(right)) <= atol


def _assert_official_test_false(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "official_test_accessed":
                _require(child is False, f"{label} enables official test access")
            _assert_official_test_false(child, label)
    elif isinstance(value, list):
        for child in value:
            _assert_official_test_false(child, label)


def _validate_operational_lock(
    path: Path,
    *,
    allow_unfrozen: bool,
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists() and not target.is_symlink():
        if allow_unfrozen:
            return {
                "schema": freezer.LOCK_SCHEMA,
                "path": str(target),
                "sha256": None,
                "source_count": len(freezer.SOURCE_PATHS),
                "verified_live": False,
                "preflight_unfrozen": True,
            }
        raise IncompleteEvidence(f"operational source lock is missing: {target}")
    try:
        action = freezer.verify(REPO_ROOT, target)
    except FileNotFoundError as error:
        raise IncompleteEvidence(str(error)) from error
    except ValueError as error:
        raise EvidenceConflict(str(error)) from error
    _require(action.get("verified") is True, "operational lock was not verified")
    return {
        "schema": freezer.LOCK_SCHEMA,
        "path": str(target),
        "sha256": action["output_sha256"],
        "source_count": action["source_count"],
        "upstream_posttraining_closure_source_lock_sha256": action[
            "upstream_posttraining_closure_source_lock_sha256"
        ],
        "receipt_schema": action["receipt_schema"],
        "verified_live": True,
        "preflight_unfrozen": False,
    }


def _validate_overlay(
    *,
    legacy_payload: Mapping[str, Any],
    profile: Mapping[str, Any],
    deployment: Mapping[str, Any],
    profile_binding: Mapping[str, Any],
    deployment_binding: Mapping[str, Any],
    artifact_binding: Mapping[str, Any],
    operational_lock: Mapping[str, Any],
) -> dict[str, Any]:
    _require(profile.get("schema") == publisher.PROFILE_SCHEMA, "profile schema differs")
    _require(
        deployment.get("schema") == publisher.MANIFEST_SCHEMA,
        "deployment-v2 manifest schema differs",
    )
    for label, payload in (("profile", profile), ("deployment-v2", deployment)):
        _require(payload.get("status") == "complete", f"{label} is incomplete")
        _require(payload.get("dataset") == "NUDT-SIRST", f"{label} dataset differs")
        _require(payload.get("training_seed") == 42, f"{label} seed differs")
        _require(payload.get("split_seed") == 20260722, f"{label} split differs")
        _require(
            payload.get("official_test_accessed") is False,
            f"{label} official-test boundary differs",
        )
        _require(
            payload.get("selected_method_id") == EXPECTED_METHOD,
            f"{label} selected method differs",
        )
        _require(
            payload.get("selected_variant") == EXPECTED_VARIANT,
            f"{label} selected variant differs",
        )
        for key in (
            "method_unchanged",
            "checkpoint_unchanged",
            "weights_unchanged",
            "artifact_reused",
        ):
            _require(payload.get(key) is True, f"{label} {key} is not true")

    legacy_authority = legacy_payload.get("terminal_authority")
    _require(
        isinstance(legacy_authority, Mapping),
        "legacy terminal authority is missing",
    )
    _require(
        legacy_authority.get("selected_method_id") == EXPECTED_METHOD,
        "legacy generator did not select method D",
    )
    _require(
        legacy_payload.get("terminal_family") == "current",
        "legacy evidence family is not current",
    )
    selected = deployment.get("deployment_operating_point", {}).get("selected")
    _require(isinstance(selected, Mapping), "deployment-v2 selected point missing")
    _require(
        selected.get("candidate_id") == publisher.EXPECTED_CANDIDATE_ID,
        "deployment-v2 candidate differs",
    )
    _require(
        selected.get("operating_point_source") == "fixed_threshold_0_5",
        "deployment-v2 source is not fixed0.5",
    )
    _require(
        _close(selected.get("threshold"), EXPECTED_THRESHOLD),
        "deployment-v2 threshold is not 0.5",
    )
    _require(
        selected.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
        "deployment-v2 checkpoint SHA differs",
    )
    _require(
        legacy_authority.get("selected_checkpoint", {}).get("sha256")
        == EXPECTED_CHECKPOINT_SHA256,
        "legacy and v2 checkpoints differ",
    )
    metrics = selected.get("metrics")
    _require(isinstance(metrics, Mapping), "deployment-v2 metrics are missing")
    for field, expected in EXPECTED_METRICS.items():
        _require(
            field in metrics and _close(metrics[field], expected),
            f"deployment-v2 {field} differs",
        )
    _require(
        float(metrics["fa"]) <= publisher.MAX_FA,
        "deployment-v2 default violates Fa <= 5e-6",
    )

    manifest_profile = deployment.get("default_operating_point_profile")
    _require(
        isinstance(manifest_profile, Mapping)
        and manifest_profile.get("path") == profile_binding["path"]
        and manifest_profile.get("sha256") == profile_binding["sha256"],
        "deployment-v2 does not bind its profile",
    )
    profile_artifact = profile.get("artifact")
    deployment_artifact = deployment.get("artifact")
    _require(
        isinstance(profile_artifact, Mapping)
        and isinstance(deployment_artifact, Mapping),
        "reused artifact binding is missing",
    )
    for observed, label in (
        (profile_artifact, "profile"),
        (deployment_artifact, "deployment-v2"),
    ):
        _require(
            observed.get("path") == artifact_binding["path"]
            and observed.get("sha256") == artifact_binding["sha256"],
            f"{label} artifact binding differs",
        )
        _require(
            observed.get("source_checkpoint_sha256")
            == EXPECTED_CHECKPOINT_SHA256,
            f"{label} artifact source checkpoint differs",
        )
        _require(
            observed.get("reused_from_legacy_v1") is True
            and observed.get("bytes_unchanged") is True
            and observed.get("new_artifact_created") is False,
            f"{label} artifact reuse contract differs",
        )
    legacy_export = legacy_authority.get("deployment_export")
    _require(
        isinstance(legacy_export, Mapping)
        and legacy_export.get("path") == artifact_binding["path"]
        and legacy_export.get("sha256") == artifact_binding["sha256"],
        "legacy evidence export differs from reused artifact",
    )

    legacy_v1 = profile.get("legacy_v1")
    _require(isinstance(legacy_v1, Mapping), "profile legacy-v1 binding is missing")
    _require(
        legacy_v1.get("legacy_only") is True
        and legacy_v1.get("authoritative_default") is False,
        "legacy operating point was not demoted",
    )
    old_point = legacy_v1.get("deployment_operating_point")
    _require(isinstance(old_point, Mapping), "legacy operating point missing")
    _require(
        not _close(old_point.get("threshold"), EXPECTED_THRESHOLD),
        "legacy and corrected thresholds unexpectedly coincide",
    )
    _require(
        old_point.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
        "legacy and v2 operating points use different checkpoints",
    )
    _assert_official_test_false(profile, "operational profile")
    _assert_official_test_false(deployment, "deployment-v2 manifest")
    return {
        "authority": "authoritative_default",
        "default_for_inference": True,
        "selected_method_id": EXPECTED_METHOD,
        "selected_variant": EXPECTED_VARIANT,
        "selected_checkpoint": copy.deepcopy(
            deployment["selected_checkpoint"]
        ),
        "selected_operating_point": copy.deepcopy(dict(selected)),
        "profile": copy.deepcopy(dict(profile_binding)),
        "deployment_manifest": copy.deepcopy(dict(deployment_binding)),
        "artifact": copy.deepcopy(dict(artifact_binding)),
        "operational_source_lock": copy.deepcopy(dict(operational_lock)),
        "legacy_v1": copy.deepcopy(dict(legacy_v1)),
        "method_changed": False,
        "checkpoint_changed": False,
        "weights_changed": False,
        "only_default_operating_point_changed": True,
    }


def build_manifest(
    *,
    controller_receipt: Path | None = None,
    profile_path: Path = publisher.DEFAULT_PROFILE,
    deployment_path: Path = publisher.DEFAULT_MANIFEST,
    artifact_path: Path = publisher.DEFAULT_ARTIFACT,
    operational_lock_path: Path = DEFAULT_OPERATIONAL_LOCK,
    allow_unfrozen: bool = False,
) -> dict[str, Any]:
    layout = legacy.default_layout()
    if controller_receipt is not None:
        layout = layout._replace(
            controller_receipt=Path(controller_receipt).expanduser().resolve()
        )
    try:
        legacy_payload = legacy.build_manifest(layout, terminal_family="current")
        publication = publisher.validate_publication(
            profile_path=profile_path,
            manifest_path=deployment_path,
            artifact_path=artifact_path,
        )
    except legacy.IncompleteEvidence as error:
        raise IncompleteEvidence(str(error)) from error
    except (legacy.EvidenceConflict, ValueError) as error:
        raise EvidenceConflict(str(error)) from error
    _require(
        publication.get("verified") is True,
        "deployment-v2 publication is not verified",
    )
    profile = _read_json(profile_path, "deployment-v2 profile")
    deployment = _read_json(deployment_path, "deployment-v2 manifest")
    profile_binding = _binding(
        profile_path,
        "deployment-v2 profile",
        schema=profile["schema"],
    )
    deployment_binding = _binding(
        deployment_path,
        "deployment-v2 manifest",
        schema=deployment["schema"],
    )
    artifact_binding = _binding(artifact_path, "inference artifact")
    operational_lock = _validate_operational_lock(
        operational_lock_path,
        allow_unfrozen=allow_unfrozen,
    )
    authority_v2 = _validate_overlay(
        legacy_payload=legacy_payload,
        profile=profile,
        deployment=deployment,
        profile_binding=profile_binding,
        deployment_binding=deployment_binding,
        artifact_binding=artifact_binding,
        operational_lock=operational_lock,
    )

    payload = copy.deepcopy(dict(legacy_payload))
    legacy_authority = copy.deepcopy(payload["terminal_authority"])
    payload["schema"] = SCHEMA
    payload["legacy_terminal_authority_v1"] = legacy_authority
    payload["terminal_authority"] = {
        "family": "current",
        "authority": "deployment_v2_default_operating_point",
        "selected_method_id": EXPECTED_METHOD,
        "selected_variant": EXPECTED_VARIANT,
        "selected_checkpoint": copy.deepcopy(
            authority_v2["selected_checkpoint"]
        ),
        "selected_threshold": EXPECTED_THRESHOLD,
        "selected_metrics": copy.deepcopy(
            authority_v2["selected_operating_point"]["metrics"]
        ),
        "default_operating_point_profile": copy.deepcopy(
            authority_v2["profile"]
        ),
        "deployment_manifest": copy.deepcopy(
            authority_v2["deployment_manifest"]
        ),
        "deployment_export": copy.deepcopy(authority_v2["artifact"]),
        "checkpoint_local_atomic_selection": True,
        "cross_checkpoint_metric_stitching": False,
        "method_changed": False,
        "checkpoint_changed": False,
        "weights_changed": False,
        "legacy_v1_operating_point_retained_as_nondefault": True,
    }
    payload["operational_default_v2"] = authority_v2
    payload["source_locks"]["operational_closure_v2_4"] = copy.deepcopy(
        operational_lock
    )
    old_producer = copy.deepcopy(payload["manifest_producer"])
    producer_path = Path(__file__).resolve()
    payload["manifest_producer"] = {
        "generator": _binding(
            producer_path,
            "reproducibility-v2 generator",
        ),
        "legacy_generator": copy.deepcopy(old_producer["generator"]),
        "legacy_markdown_template": copy.deepcopy(
            old_producer["markdown_template"]
        ),
        "output_mode": "atomic_directory_rename_noreplace",
        "write_once": True,
        "existing_identical_is_verify": True,
        "existing_conflict_is_refused": True,
    }
    payload["claim_boundary"]["operational_default_corrected"] = True
    payload["claim_boundary"]["paper_core_established"] = False
    payload["claim_boundary"]["stability_claim_supported"] = False
    payload["write_once"] = True
    payload["overwrite_forbidden"] = True
    _assert_official_test_false(payload, "reproducibility-v2 manifest")
    return _canonical(payload)


def render_markdown(payload: Mapping[str, Any]) -> str:
    _require(payload.get("schema") == SCHEMA, "manifest-v2 schema differs")
    authority = payload["terminal_authority"]
    metrics = authority["selected_metrics"]
    legacy_point = payload["legacy_terminal_authority_v1"]
    old_threshold = legacy_point["selected_threshold"]
    lock = payload["operational_default_v2"]["operational_source_lock"]
    return "\n".join(
        [
            "# TPD+NER+QFG formal800 reproducibility manifest v2",
            "",
            "- Status: `complete`",
            "- Final method: `d_tss_qfg` (TPD + five-node NER + QFG)",
            "- TSS: training only; removed from inference state",
            "- Dataset scope: NUDT-SIRST 530/133 internal split, seed 42",
            "- Official test accessed: `false`",
            "",
            "## Authoritative default operating point",
            "",
            f"- Checkpoint: `{authority['selected_checkpoint']['checkpoint']}`",
            f"- Checkpoint epoch: `{authority['selected_checkpoint']['checkpoint_epoch']}`",
            f"- Checkpoint SHA256: `{authority['selected_checkpoint']['checkpoint_sha256']}`",
            f"- Threshold: `{authority['selected_threshold']:.10g}`",
            f"- Pd: `{metrics['pd']:.9f}` (188/189)",
            f"- Fa: `{metrics['fa']:.9g}`",
            f"- mIoU: `{metrics['miou']:.9f}`",
            f"- tiny-Pd: `{metrics['tiny_pd']:.9f}` (39/39)",
            f"- False objects/image: `{metrics['false_objects_per_image']:.9f}` (5 total)",
            "",
            "## Version boundary",
            "",
            f"- Legacy v1 threshold `{old_threshold:.10g}` is retained only as a high-recall profile.",
            "- Method, checkpoint, model weights, and inference artifact are unchanged.",
            "- Only the authoritative default operating point changed.",
            f"- Operational source lock SHA256: `{lock['sha256']}`",
            "",
            "## Claim boundary",
            "",
            "- `paper_core_established=false`",
            "- `stability_claim_supported=false`",
            "- This is a seed-42 internal-validation engineering selection.",
            "",
        ]
    )


def _bundle_bytes(payload: Mapping[str, Any]) -> dict[str, bytes]:
    return {
        "manifest.json": _canonical_bytes(payload),
        "manifest.md": render_markdown(payload).encode("utf-8"),
    }


def _verify_bundle(
    output_dir: Path,
    expected: Mapping[str, bytes],
) -> dict[str, dict[str, Any]]:
    output = _regular_directory(output_dir, "reproducibility-v2 output")
    observed_names = {
        child.name
        for child in output.iterdir()
        if child.is_file() and not child.is_symlink()
    }
    _require(
        observed_names == set(OUTPUT_NAMES)
        and all(not (output / name).is_symlink() for name in OUTPUT_NAMES),
        "reproducibility-v2 bundle file set differs",
    )
    bindings: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        path = _regular_file(output / name, name)
        observed = path.read_bytes()
        _require(observed == expected[name], f"{name} conflicts with live evidence")
        bindings[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(observed).hexdigest(),
        }
    return bindings


def _publish_bundle(
    output_dir: Path,
    expected: Mapping[str, bytes],
) -> tuple[dict[str, dict[str, Any]], bool]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        return _verify_bundle(output, expected), False
    output.parent.mkdir(parents=True, exist_ok=True)
    _regular_directory(output.parent, "reproducibility-v2 parent")
    stage = Path(
        tempfile.mkdtemp(
            dir=output.parent,
            prefix=f".{output.name}.",
        )
    )
    published = False
    try:
        for name, content in expected.items():
            path = stage / name
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        legacy._fsync_directory(stage)
        try:
            legacy._rename_noreplace(stage, output)
            published = True
        except FileExistsError:
            return _verify_bundle(output, expected), False
        legacy._fsync_directory(output.parent)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
    return _verify_bundle(output, expected), True


def execute(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    controller_receipt: Path | None = None,
    profile_path: Path = publisher.DEFAULT_PROFILE,
    deployment_path: Path = publisher.DEFAULT_MANIFEST,
    artifact_path: Path = publisher.DEFAULT_ARTIFACT,
    operational_lock_path: Path = DEFAULT_OPERATIONAL_LOCK,
    preflight: bool = False,
    verify: bool = False,
    allow_unfrozen: bool = False,
) -> dict[str, Any]:
    _require(not (preflight and verify), "preflight and verify are exclusive")
    _require(
        not allow_unfrozen or preflight,
        "--allow-unfrozen is permitted only with --preflight",
    )
    payload = build_manifest(
        controller_receipt=controller_receipt,
        profile_path=profile_path,
        deployment_path=deployment_path,
        artifact_path=artifact_path,
        operational_lock_path=operational_lock_path,
        allow_unfrozen=allow_unfrozen,
    )
    expected = _bundle_bytes(payload)
    output = Path(output_dir).expanduser().resolve()
    if preflight:
        return {
            "schema": ACTION_SCHEMA,
            "status": "ready",
            "action": "preflight",
            "writes_performed": False,
            "output_dir": str(output),
            "payload_sha256": _canonical_sha256(payload),
            "json_sha256": hashlib.sha256(expected["manifest.json"]).hexdigest(),
            "markdown_sha256": hashlib.sha256(expected["manifest.md"]).hexdigest(),
            "operational_source_lock_verified": not allow_unfrozen,
            "selected_method_id": EXPECTED_METHOD,
            "selected_threshold": EXPECTED_THRESHOLD,
        }
    if verify:
        bindings = _verify_bundle(output, expected)
        action = "verify"
        writes = False
    else:
        bindings, writes = _publish_bundle(output, expected)
        action = "publish" if writes else "verify"
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": action,
        "writes_performed": writes,
        "output_dir": str(output),
        "outputs": bindings,
        "payload_sha256": _canonical_sha256(payload),
        "selected_method_id": EXPECTED_METHOD,
        "selected_threshold": EXPECTED_THRESHOLD,
        "operational_source_lock_verified": True,
        "write_once": True,
        "atomic_pair": True,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--allow-unfrozen", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--controller-receipt", type=Path)
    parser.add_argument("--profile", type=Path, default=publisher.DEFAULT_PROFILE)
    parser.add_argument(
        "--deployment-v2-manifest",
        type=Path,
        default=publisher.DEFAULT_MANIFEST,
    )
    parser.add_argument("--artifact", type=Path, default=publisher.DEFAULT_ARTIFACT)
    parser.add_argument(
        "--operational-source-lock",
        type=Path,
        default=DEFAULT_OPERATIONAL_LOCK,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        result = execute(
            output_dir=args.output_dir,
            controller_receipt=args.controller_receipt,
            profile_path=args.profile,
            deployment_path=args.deployment_v2_manifest,
            artifact_path=args.artifact,
            operational_lock_path=args.operational_source_lock,
            preflight=args.preflight,
            verify=args.verify,
            allow_unfrozen=args.allow_unfrozen,
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
    except (EvidenceConflict, FileExistsError, OSError) as error:
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
        return EXIT_CONFLICT
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
