#!/usr/bin/env python3
"""Publish the authoritative low-Fa operating point without changing weights.

The v1 deployment closure remains immutable evidence.  This additive v2 layer
reuses its exact inference artifact and changes only the authoritative
operating point to the preregistered low-Fa choice from the selected D
checkpoint.  Publication is write-once; ``--preflight`` performs no writes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    deploy_tpd_ner_v4_qfg_v2_croa_formal800 as legacy_deployer,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa as qfg_evaluator,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as selector,
)
from experiments import (  # noqa: E402
    tpd_ner_v4_qfg_v2_croa_posttraining_policy as closure_policy,
)


PROFILE_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2"
)
MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_manifest_v2"
)
POLICY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_default_operating_point_policy_v2"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_default_operating_point_action_v2"
)

DEFAULT_SELECTION = legacy_deployer.DEFAULT_SELECTION
DEFAULT_LEGACY_MANIFEST = legacy_deployer.DEFAULT_MANIFEST
DEFAULT_ARTIFACT = legacy_deployer.DEFAULT_ARTIFACT
DEFAULT_OUTPUT_DIR = selector.QFG_RESULT_ROOT / "deployment_v2"
DEFAULT_PROFILE = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_formal800_default_operating_point_v2.json"
)
DEFAULT_MANIFEST = (
    DEFAULT_OUTPUT_DIR
    / "tpd_ner_v4_qfg_v2_croa_formal800_deployment_manifest_v2.json"
)

MAX_FA = 5e-6
EPSILON = 1e-12
EXPECTED_METHOD_ID = "d_tss_qfg"
EXPECTED_VARIANT = "tss_qfg"
EXPECTED_ROLE_NAME = "miou_secondary"
EXPECTED_CHECKPOINT = "best_miou.pth.tar"
EXPECTED_CHECKPOINT_ROLE = "best_validation_miou_secondary"
EXPECTED_CHECKPOINT_EPOCH = 3
EXPECTED_CHECKPOINT_SHA256 = (
    "890c8cf0e0f7c3a4c21e5772e69cd89e3038b308a1d77be58365f2254b89b678"
)
EXPECTED_CANDIDATE_ID = "miou_secondary:fixed_threshold_0_5"
EXPECTED_THRESHOLD = 0.5

ROLE_ORDER = ("pd_primary", "miou_secondary")
SOURCE_ORDER = {
    "fixed_threshold_0_5": 0,
    "fa_budget:1e-06": 1,
    "fa_budget:5e-06": 2,
    "fa_budget:1e-05": 3,
    "fa_budget:5e-05": 4,
    "fa_budget:0.0001": 5,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    source = closure_policy.regular_file(path, label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be one JSON object")
    return payload


def _binding(
    path: Path,
    *,
    label: str,
    schema: Any | None = None,
) -> dict[str, Any]:
    source = closure_policy.regular_file(path, label).resolve()
    result: dict[str, Any] = {
        "path": str(source),
        "sha256": closure_policy.sha256_file(source),
    }
    if schema is not None:
        result["schema"] = schema
    return result


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace v2 output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    content = closure_policy.canonical_json_bytes(payload)
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


def _point_candidate(
    role_name: str,
    role: Mapping[str, Any],
    *,
    source: str,
    point: Mapping[str, Any],
    fa_budget_key: str | None = None,
) -> dict[str, Any]:
    candidate_id = f"{role_name}:{source}"
    return {
        "candidate_id": candidate_id,
        "method_id": EXPECTED_METHOD_ID,
        "variant": EXPECTED_VARIANT,
        "checkpoint": role["checkpoint"],
        "checkpoint_role": role["checkpoint_role"],
        "role_name": role_name,
        "checkpoint_epoch": role["checkpoint_epoch"],
        "checkpoint_path": role["checkpoint_path"],
        "checkpoint_sha256": role["checkpoint_sha256"],
        "operating_point_source": source,
        "fa_budget_key": fa_budget_key,
        "threshold": point["threshold"],
        "metrics": copy.deepcopy(dict(point)),
        "checkpoint_local_atomic_point": True,
    }


def preregistered_candidates(
    method: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the 12 v1-registered operating points for selected method D."""

    _require(method.get("method_id") == EXPECTED_METHOD_ID, "method D is required")
    _require(method.get("variant") == EXPECTED_VARIANT, "method D variant differs")
    roles = method.get("roles")
    _require(isinstance(roles, Mapping), "method D roles are missing")
    candidates: list[dict[str, Any]] = []
    for role_name in ROLE_ORDER:
        role = roles.get(role_name)
        _require(isinstance(role, Mapping), f"method D role {role_name} is missing")
        candidates.append(
            _point_candidate(
                role_name,
                role,
                source="fixed_threshold_0_5",
                point=role["fixed_threshold_0_5"],
            )
        )
        budgets = role.get("fa_budget_points")
        _require(isinstance(budgets, Mapping), f"{role_name} budgets are missing")
        for budget_key in selector.BUDGET_KEYS:
            _require(
                budget_key in budgets,
                f"{role_name} budget {budget_key} is missing",
            )
            candidates.append(
                _point_candidate(
                    role_name,
                    role,
                    source=f"fa_budget:{budget_key}",
                    point=budgets[budget_key],
                    fa_budget_key=budget_key,
                )
            )
    _require(len(candidates) == 12, "v1 candidate universe must contain 12 points")
    return candidates


def _ranking_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    return (
        -float(metrics["pd"]),
        -float(metrics["tiny_pd"]),
        -float(metrics["miou"]),
        float(metrics["false_objects_per_image"]),
        int(metrics["unmatched_predicted_object_count"]),
        float(metrics["fa"]),
        ROLE_ORDER.index(str(candidate["role_name"])),
        SOURCE_ORDER[str(candidate["operating_point_source"])],
        float(candidate["threshold"]),
    )


def select_default_operating_point(
    method: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the registered low-Fa constraint and deterministic ranking."""

    candidates = preregistered_candidates(method)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["metrics"]["fa"]) <= MAX_FA + EPSILON
    ]
    _require(eligible, "no preregistered method-D point satisfies Fa <= 5e-6")
    ranked = sorted(eligible, key=_ranking_key)
    selected = copy.deepcopy(ranked[0])
    policy = {
        "schema": POLICY_SCHEMA,
        "authority": "authoritative_default",
        "default_for_inference": True,
        "candidate_registration": (
            "legacy_v1_method_d_checkpoint_local_fixed0.5_and_five_fa_budgets"
        ),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "eligibility": {
            "metric": "fa",
            "comparison": "less_than_or_equal",
            "maximum": MAX_FA,
            "epsilon": EPSILON,
            "uses_realized_candidate_fa": True,
        },
        "ranking_order": [
            "pd:maximize",
            "tiny_pd:maximize",
            "miou:maximize",
            "false_objects_per_image:minimize",
            "unmatched_predicted_object_count:minimize",
            "fa:minimize",
        ],
        "deterministic_suffix": [
            "checkpoint_role:pd_primary_before_miou_secondary",
            "source:fixed0.5_then_fa_budget_ascending",
            "threshold:minimize",
        ],
        "cross_checkpoint_metric_stitching": False,
        "selected_point_is_checkpoint_local": True,
    }
    return selected, policy


def _assert_expected_selected(selected: Mapping[str, Any]) -> None:
    expected = {
        "method_id": EXPECTED_METHOD_ID,
        "variant": EXPECTED_VARIANT,
        "role_name": EXPECTED_ROLE_NAME,
        "checkpoint": EXPECTED_CHECKPOINT,
        "checkpoint_role": EXPECTED_CHECKPOINT_ROLE,
        "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "operating_point_source": "fixed_threshold_0_5",
        "threshold": EXPECTED_THRESHOLD,
    }
    for field, value in expected.items():
        _require(selected.get(field) == value, f"selected {field} differs")
    _require(
        float(selected["metrics"]["fa"]) <= MAX_FA,
        "selected fixed0.5 point violates Fa <= 5e-6",
    )


def _validate_selected_sweep(
    role: Mapping[str, Any],
) -> dict[str, Any]:
    sweep_binding = role.get("sweep_binding")
    _require(isinstance(sweep_binding, Mapping), "D best_miou sweep binding missing")
    sweep_path = Path(str(sweep_binding.get("path"))).resolve()
    sweep = _load_json(sweep_path, "D best_miou sweep")
    _require(
        closure_policy.sha256_file(sweep_path) == sweep_binding.get("sha256"),
        "D best_miou sweep SHA differs",
    )
    audit = qfg_evaluator.validate_run_artifacts(
        Path(str(role["run_directory"])).resolve(),
        EXPECTED_CHECKPOINT,
    )
    qfg_evaluator.validate_output_identity(sweep, artifact_audit=audit)
    normalized = selector.normalize_sweep_payload(
        sweep,
        method_id=EXPECTED_METHOD_ID,
        display_name="D: TSS+QFG",
        expected_variant=EXPECTED_VARIANT,
        checkpoint=EXPECTED_CHECKPOINT,
        sweep_path=sweep_path,
        sweep_sha256=str(sweep_binding["sha256"]),
    )
    for field in (
        "checkpoint",
        "checkpoint_role",
        "role_name",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "checkpoint_path",
        "run_directory",
        "fixed_threshold_0_5",
        "fa_budget_points",
        "raw_point_count",
        "sweep_binding",
    ):
        _require(
            closure_policy.canonical(normalized[field])
            == closure_policy.canonical(role[field]),
            f"D best_miou sweep normalized {field} differs from v1 selection",
        )
    return {
        "path": str(sweep_path),
        "sha256": closure_policy.sha256_file(sweep_path),
        "schema": sweep.get("schema"),
        "checkpoint_identity_validated": True,
        "evaluator_output_identity_validated": True,
        "checkpoint_state_dict_strict_load": True,
    }


def _prepare_context(
    *,
    selection_path: Path,
    legacy_manifest_path: Path,
    artifact_path: Path,
    closure_lock_path: Path | None,
) -> dict[str, Any]:
    legacy_action = legacy_deployer.validate_deployment_closure(
        selection_path=selection_path,
        artifact_path=artifact_path,
        manifest_path=legacy_manifest_path,
        closure_lock_path=closure_lock_path,
    )
    _require(legacy_action.get("verified") is True, "v1 deployment is not verified")
    _, closure_binding = closure_policy.load_closure_lock(
        closure_lock_path,
        verify_sources=True,
    )
    _require(
        closure_binding.get("source_count") == 15,
        "posttraining closure must bind exactly 15 sources",
    )

    selection = _load_json(selection_path, "v1 final selection")
    legacy_manifest = _load_json(legacy_manifest_path, "v1 deployment manifest")
    _require(
        selection.get("schema") == selector.SCHEMA,
        "v1 final selection schema differs",
    )
    _require(
        legacy_manifest.get("schema") == legacy_deployer.MANIFEST_SCHEMA,
        "v1 deployment manifest schema differs",
    )
    _require(selection.get("status") == "complete", "v1 selection is incomplete")
    _require(
        selection.get("selection", {}).get("selected_method_id")
        == EXPECTED_METHOD_ID,
        "v1 selected method is not D",
    )
    method = selection.get("methods", {}).get(EXPECTED_METHOD_ID)
    _require(isinstance(method, Mapping), "v1 selection has no method D")
    role = method.get("roles", {}).get(EXPECTED_ROLE_NAME)
    _require(isinstance(role, Mapping), "v1 selection has no D best_miou role")
    sweep_binding = _validate_selected_sweep(role)

    selected, operating_policy = select_default_operating_point(method)
    _assert_expected_selected(selected)

    artifact = legacy_manifest.get("artifact")
    _require(isinstance(artifact, Mapping), "v1 manifest artifact is missing")
    artifact_source = closure_policy.regular_file(
        artifact_path,
        "v1 deployment artifact",
    ).resolve()
    _require(
        Path(str(artifact.get("path"))).resolve() == artifact_source,
        "v1 artifact path differs",
    )
    _require(
        closure_policy.sha256_file(artifact_source) == artifact.get("sha256"),
        "v1 artifact SHA differs",
    )
    _require(
        artifact.get("source_checkpoint_sha256")
        == selected["checkpoint_sha256"],
        "reused artifact source checkpoint differs",
    )
    _require(
        artifact.get("source_checkpoint_role")
        == selected["checkpoint_role"],
        "reused artifact source role differs",
    )
    legacy_selected = legacy_manifest.get(
        "deployment_operating_point",
        {},
    ).get("selected")
    _require(isinstance(legacy_selected, Mapping), "v1 operating point is missing")
    _require(
        legacy_selected.get("method_id") == selected["method_id"],
        "v2 method differs from v1",
    )
    _require(
        legacy_selected.get("checkpoint_sha256")
        == selected["checkpoint_sha256"],
        "v2 checkpoint differs from v1",
    )

    selection_binding = _binding(
        selection_path,
        label="v1 final selection",
        schema=selection["schema"],
    )
    legacy_manifest_binding = _binding(
        legacy_manifest_path,
        label="v1 deployment manifest",
        schema=legacy_manifest["schema"],
    )
    return {
        "selection": selection,
        "legacy_manifest": legacy_manifest,
        "legacy_selected": copy.deepcopy(dict(legacy_selected)),
        "selection_binding": selection_binding,
        "legacy_manifest_binding": legacy_manifest_binding,
        "closure_binding": copy.deepcopy(dict(closure_binding)),
        "sweep_binding": sweep_binding,
        "artifact": copy.deepcopy(dict(artifact)),
        "selected": selected,
        "policy": operating_policy,
    }


def _selected_checkpoint(selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(selected[key])
        for key in (
            "checkpoint",
            "checkpoint_role",
            "role_name",
            "checkpoint_epoch",
            "checkpoint_path",
            "checkpoint_sha256",
        )
    }


def _default_point(selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": selected["candidate_id"],
        "source": selected["operating_point_source"],
        "operating_point_source": selected["operating_point_source"],
        "threshold": selected["threshold"],
        "metrics": copy.deepcopy(dict(selected["metrics"])),
        "fa_ceiling": MAX_FA,
        "actual_fa_within_ceiling": True,
        "checkpoint_local_atomic_point": True,
    }


def _legacy_v1(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "legacy_only": True,
        "authoritative_default": False,
        "final_selection": copy.deepcopy(dict(context["selection_binding"])),
        "deployment_manifest": copy.deepcopy(
            dict(context["legacy_manifest_binding"])
        ),
        "deployment_operating_point": copy.deepcopy(
            dict(context["legacy_selected"])
        ),
        "old_source": context["legacy_selected"]["operating_point_source"],
        "old_threshold": context["legacy_selected"]["threshold"],
        "method_retained": True,
        "checkpoint_retained": True,
        "artifact_retained": True,
    }


def _artifact(context: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(context["artifact"]))
    result["reused_from_legacy_v1"] = True
    result["bytes_unchanged"] = True
    result["new_artifact_created"] = False
    return result


def _publisher_binding() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        "path": str(path),
        "sha256": closure_policy.sha256_file(path),
    }


def build_profile(context: Mapping[str, Any]) -> dict[str, Any]:
    selected = context["selected"]
    return {
        "schema": PROFILE_SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "selected_method_id": selected["method_id"],
        "selected_variant": selected["variant"],
        "selected_checkpoint": _selected_checkpoint(selected),
        "default_operating_point": _default_point(selected),
        "policy": copy.deepcopy(dict(context["policy"])),
        "legacy_v1": _legacy_v1(context),
        "evidence": {
            "posttraining_closure_source_lock": copy.deepcopy(
                dict(context["closure_binding"])
            ),
            "selected_checkpoint_sweep": copy.deepcopy(
                dict(context["sweep_binding"])
            ),
            "v1_deployment_closure_verified": True,
            "d_best_miou_sweep_validated": True,
            "source_count": 15,
        },
        "artifact": _artifact(context),
        "method_unchanged": True,
        "checkpoint_unchanged": True,
        "weights_unchanged": True,
        "artifact_reused": True,
        "publisher": _publisher_binding(),
        "write_once": True,
        "overwrite_forbidden": True,
    }


def build_manifest(
    context: Mapping[str, Any],
    *,
    profile_binding: Mapping[str, Any],
) -> dict[str, Any]:
    selected = context["selected"]
    deployment_selected = {
        "method_id": selected["method_id"],
        "variant": selected["variant"],
        **_selected_checkpoint(selected),
        **_default_point(selected),
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "selected_method_id": selected["method_id"],
        "selected_variant": selected["variant"],
        "selected_checkpoint": _selected_checkpoint(selected),
        "deployment_operating_point": {
            "authority": "authoritative_default",
            "default_for_inference": True,
            "selected": deployment_selected,
            "policy": copy.deepcopy(dict(context["policy"])),
        },
        "default_operating_point_profile": copy.deepcopy(
            dict(profile_binding)
        ),
        "legacy_v1": _legacy_v1(context),
        "evidence": {
            "posttraining_closure_source_lock": copy.deepcopy(
                dict(context["closure_binding"])
            ),
            "selected_checkpoint_sweep": copy.deepcopy(
                dict(context["sweep_binding"])
            ),
        },
        "artifact": _artifact(context),
        "method_unchanged": True,
        "checkpoint_unchanged": True,
        "weights_unchanged": True,
        "artifact_reused": True,
        "publisher": _publisher_binding(),
        "cross_checkpoint_metric_stitching": False,
        "selected_point_is_checkpoint_local": True,
        "write_once": True,
        "overwrite_forbidden": True,
    }


def _profile_binding(path: Path) -> dict[str, Any]:
    return _binding(
        path,
        label="v2 default operating-point profile",
        schema=PROFILE_SCHEMA,
    )


def _verify_outputs(
    context: Mapping[str, Any],
    *,
    profile_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    expected_profile = build_profile(context)
    profile = _load_json(profile_path, "v2 default operating-point profile")
    _require(
        closure_policy.canonical(profile)
        == closure_policy.canonical(expected_profile),
        "v2 default operating-point profile conflicts with live evidence",
    )
    profile_binding = _profile_binding(profile_path)
    expected_manifest = build_manifest(
        context,
        profile_binding=profile_binding,
    )
    manifest = _load_json(manifest_path, "v2 deployment manifest")
    _require(
        closure_policy.canonical(manifest)
        == closure_policy.canonical(expected_manifest),
        "v2 deployment manifest conflicts with live evidence",
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "verify",
        "profile_path": str(Path(profile_path).resolve()),
        "profile_sha256": profile_binding["sha256"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": closure_policy.sha256_file(manifest_path),
        "artifact_path": context["artifact"]["path"],
        "artifact_sha256": context["artifact"]["sha256"],
        "selected_method_id": context["selected"]["method_id"],
        "selected_checkpoint_role": context["selected"]["checkpoint_role"],
        "selected_threshold": context["selected"]["threshold"],
        "method_unchanged": True,
        "checkpoint_unchanged": True,
        "weights_unchanged": True,
        "artifact_reused": True,
        "verified": True,
    }


def validate_publication(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    legacy_manifest_path: Path = DEFAULT_LEGACY_MANIFEST,
    artifact_path: Path = DEFAULT_ARTIFACT,
    profile_path: Path = DEFAULT_PROFILE,
    manifest_path: Path = DEFAULT_MANIFEST,
    closure_lock_path: Path | None = None,
) -> dict[str, Any]:
    context = _prepare_context(
        selection_path=selection_path,
        legacy_manifest_path=legacy_manifest_path,
        artifact_path=artifact_path,
        closure_lock_path=closure_lock_path,
    )
    return _verify_outputs(
        context,
        profile_path=profile_path,
        manifest_path=manifest_path,
    )


def publish_default_operating_point(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    legacy_manifest_path: Path = DEFAULT_LEGACY_MANIFEST,
    artifact_path: Path = DEFAULT_ARTIFACT,
    profile_path: Path = DEFAULT_PROFILE,
    manifest_path: Path = DEFAULT_MANIFEST,
    closure_lock_path: Path | None = None,
    preflight: bool = False,
) -> dict[str, Any]:
    context = _prepare_context(
        selection_path=selection_path,
        legacy_manifest_path=legacy_manifest_path,
        artifact_path=artifact_path,
        closure_lock_path=closure_lock_path,
    )
    profile = Path(profile_path).resolve()
    manifest = Path(manifest_path).resolve()
    profile_exists = profile.exists() or profile.is_symlink()
    manifest_exists = manifest.exists() or manifest.is_symlink()

    if manifest_exists and not profile_exists:
        raise ValueError("v2 manifest exists without its bound profile")
    if profile_exists and manifest_exists:
        return _verify_outputs(
            context,
            profile_path=profile,
            manifest_path=manifest,
        )
    if profile_exists:
        expected_profile = build_profile(context)
        observed_profile = _load_json(
            profile,
            "partial v2 default operating-point profile",
        )
        _require(
            closure_policy.canonical(observed_profile)
            == closure_policy.canonical(expected_profile),
            "partial v2 profile conflicts with live evidence",
        )
    if preflight:
        return {
            "schema": ACTION_SCHEMA,
            "status": "ready",
            "action": "preflight",
            "profile_state": "existing" if profile_exists else "missing",
            "manifest_state": "missing",
            "would_publish_profile": not profile_exists,
            "would_publish_manifest": True,
            "writes_performed": False,
            "selected_method_id": context["selected"]["method_id"],
            "selected_checkpoint_role": context["selected"]["checkpoint_role"],
            "selected_threshold": context["selected"]["threshold"],
            "artifact_reused": True,
        }

    if not profile_exists:
        _atomic_create_json(profile, build_profile(context))
    profile_binding = _profile_binding(profile)
    _atomic_create_json(
        manifest,
        build_manifest(context, profile_binding=profile_binding),
    )
    verified = _verify_outputs(
        context,
        profile_path=profile,
        manifest_path=manifest,
    )
    verified["action"] = "publish"
    return verified


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--legacy-manifest",
        type=Path,
        default=DEFAULT_LEGACY_MANIFEST,
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--closure-source-lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    if args.preflight and args.verify:
        raise ValueError("--preflight and --verify are mutually exclusive")
    common = {
        "selection_path": args.selection,
        "legacy_manifest_path": args.legacy_manifest,
        "artifact_path": args.artifact,
        "profile_path": args.profile,
        "manifest_path": args.manifest,
        "closure_lock_path": args.closure_source_lock,
    }
    if args.verify:
        result = validate_publication(**common)
    else:
        result = publish_default_operating_point(
            **common,
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
    "DEFAULT_LEGACY_MANIFEST",
    "DEFAULT_MANIFEST",
    "DEFAULT_PROFILE",
    "DEFAULT_SELECTION",
    "MANIFEST_SCHEMA",
    "PROFILE_SCHEMA",
    "build_manifest",
    "build_profile",
    "main",
    "preregistered_candidates",
    "publish_default_operating_point",
    "select_default_operating_point",
    "validate_publication",
]


if __name__ == "__main__":
    main()
