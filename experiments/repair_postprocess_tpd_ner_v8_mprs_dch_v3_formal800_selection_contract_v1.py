#!/usr/bin/env python3
"""Versioned repair for the V3 formal800 selection-contract aggregation.

The frozen postprocessor and every frozen artifact remain byte-immutable.
This wrapper reuses its aggregate implementation while replacing only
``same_split_and_training_contract`` in memory.  Publication constants are
temporarily redirected to a repair-owned directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_v3_formal800 as locked,
)


REPAIR_ID = "v3_formal800_selection_contract_repair_v1"
CONTRACT_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_selection_contract_repair_v1"
)
ATTESTATION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_selection_contract_"
    "repair_attestation_v1"
)
PROTOCOL = (
    REPO_ROOT
    / "experiments/"
    "TPD_NER_V8_MPRS_DCH_V3_FORMAL800_SELECTION_CONTRACT_REPAIR_V1.md"
)
ATTESTATION = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v8_mprs_dch_v3_formal800_selection_contract_"
    "repair_v1_attestation.json"
)
REPAIR_COMPARISON_DIR = (
    locked.V3_RESULT_ROOT
    / locked.DATASET
    / "comparison_selection_contract_repair_v1"
)
REPAIR_JSON_OUTPUT = (
    REPAIR_COMPARISON_DIR
    / "tpd_ner_v8_mprs_dch_v3_formal800_comparison_"
    "selection_contract_repair_v1.json"
)
REPAIR_MARKDOWN_OUTPUT = (
    REPAIR_COMPARISON_DIR
    / "tpd_ner_v8_mprs_dch_v3_formal800_comparison_"
    "selection_contract_repair_v1.md"
)
REPAIR_COMPLETE_MARKER = (
    REPAIR_COMPARISON_DIR
    / "POSTPROCESS_COMPLETE_SELECTION_CONTRACT_REPAIR_V1.json"
)
BASELINE_CHECKPOINT_POLICY = (
    "best.pth.tar is Pd-primary; best_miou.pth.tar is a secondary "
    "analysis checkpoint; all selection uses internal validation only"
)
MODERN_VARIANTS = (
    locked.VARIANT_V3_ON,
    locked.VARIANT_V2_ON,
    locked.VARIANT_V1_OFF,
)
ALL_VARIANTS = (*MODERN_VARIANTS, locked.BASELINE_VARIANT)
_FROZEN_CONTRACT_FUNCTION = locked.same_split_and_training_contract


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def required_attestation_paths() -> dict[str, Path]:
    """Return the exact closure that the repair attestation must bind."""

    paths = {
        "source:frozen_v3_postprocess": Path(locked.__file__),
        "source:repair_wrapper": Path(__file__),
        "source:repair_protocol": PROTOCOL,
        "artifact:v3_sweep_best": locked.sweep_path(
            locked.V3_RUN_DIR,
            "best.pth.tar",
        ),
        "artifact:v3_sweep_best_miou": locked.sweep_path(
            locked.V3_RUN_DIR,
            "best_miou.pth.tar",
        ),
        "protocol:v3": locked.V3_RUN_DIR / "protocol.json",
        "protocol:v2": locked.V2_RUN_DIR / "protocol.json",
        "protocol:v1": locked.V1_OFF_RUN_DIR / "protocol.json",
        "protocol:baseline": locked.BASELINE_RUN_DIR / "protocol.json",
        "artifact:baseline_sweep_best": locked.sweep_path(
            locked.BASELINE_RUN_DIR,
            "best.pth.tar",
        ),
        "artifact:baseline_sweep_best_miou": locked.sweep_path(
            locked.BASELINE_RUN_DIR,
            "best_miou.pth.tar",
        ),
        "aggregate:v2_json": locked.v2_post.JSON_OUTPUT,
        "aggregate:v2_markdown": locked.v2_post.MARKDOWN_OUTPUT,
        "aggregate:v2_complete_marker": locked.v2_post.COMPLETE_MARKER,
    }
    return {name: path.resolve() for name, path in paths.items()}


def verify_repair_attestation(
    path: Path = ATTESTATION,
) -> dict[str, Any]:
    """Verify the no-extra/no-missing, path-exact repair closure."""

    payload = load_json(path)
    _require(
        payload.get("schema") == ATTESTATION_SCHEMA,
        "repair attestation schema differs",
    )
    _require(payload.get("status") == "frozen", "attestation is not frozen")
    _require(
        payload.get("repair_id") == REPAIR_ID,
        "repair attestation identity differs",
    )
    policy = payload.get("policy")
    _require(isinstance(policy, Mapping), "repair policy is missing")
    _require(
        policy.get("frozen_artifacts_modified") is False,
        "repair policy permits frozen mutation",
    )
    _require(
        policy.get("logic_override")
        == "same_split_and_training_contract",
        "repair logic override differs",
    )
    _require(
        policy.get("aggregate_implementation")
        == "frozen_v3_aggregate_and_write",
        "repair aggregate implementation differs",
    )
    bindings = payload.get("bindings")
    _require(isinstance(bindings, Mapping), "repair bindings are missing")
    expected = required_attestation_paths()
    _require(
        set(bindings) == set(expected),
        "repair attestation binding matrix differs",
    )
    verified: dict[str, Any] = {}
    for name, expected_path in expected.items():
        entry = bindings.get(name)
        _require(isinstance(entry, Mapping), f"{name} binding is missing")
        _require(
            set(entry) == {"path", "sha256"},
            f"{name} binding fields differ",
        )
        expected_relative = str(expected_path.relative_to(REPO_ROOT))
        _require(
            entry.get("path") == expected_relative,
            f"{name} path differs",
        )
        expected_sha = entry.get("sha256")
        _require(
            isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and all(
                character in "0123456789abcdef"
                for character in expected_sha
            ),
            f"{name} SHA-256 is invalid",
        )
        observed_sha = sha256_file(expected_path)
        _require(observed_sha == expected_sha, f"{name} SHA-256 differs")
        verified[name] = {
            "path": str(expected_path),
            "sha256": observed_sha,
        }
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "verified",
        "repair_id": REPAIR_ID,
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "bindings": verified,
    }


def _validate_v2_aggregate_and_baseline_evidence(
    *,
    baseline_protocol: Mapping[str, Any],
    baseline_split: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove legacy-baseline internal selection from immutable evidence."""

    _require(
        "selection_source" not in baseline_protocol,
        "legacy baseline unexpectedly has a top-level selection_source",
    )
    _require(
        baseline_protocol.get("checkpoint_policy")
        == BASELINE_CHECKPOINT_POLICY,
        "legacy baseline checkpoint policy differs",
    )
    report = load_json(locked.v2_post.JSON_OUTPUT)
    marker = load_json(locked.v2_post.COMPLETE_MARKER)
    _require(
        report.get("schema") == locked.v2_post.SCHEMA,
        "V2 aggregate schema differs",
    )
    for field, expected in {
        "status": "complete",
        "dataset": locked.DATASET,
        "training_seed": locked.TRAINING_SEED,
        "split_seed": locked.SPLIT_SEED,
        "scope": "single_seed_internal_validation",
        "official_test_accessed": False,
    }.items():
        _require(report.get(field) == expected, f"V2 aggregate differs: {field}")
    _require(
        marker.get("schema") == locked.v2_post.COMPLETE_MARKER_SCHEMA,
        "V2 aggregate completion-marker schema differs",
    )
    _require(
        marker.get("status") == "complete",
        "V2 aggregate completion marker is incomplete",
    )
    marker_outputs = marker.get("outputs")
    _require(
        isinstance(marker_outputs, Mapping),
        "V2 aggregate marker outputs are missing",
    )
    for output in (
        locked.v2_post.JSON_OUTPUT,
        locked.v2_post.MARKDOWN_OUTPUT,
    ):
        _require(
            marker_outputs.get(output.name) == sha256_file(output),
            f"V2 aggregate marker does not bind {output.name}",
        )
    report_bindings = report.get("bindings")
    _require(
        isinstance(report_bindings, Mapping),
        "V2 aggregate bindings are missing",
    )
    sweep_bindings = report_bindings.get("sweeps")
    _require(
        isinstance(sweep_bindings, Mapping),
        "V2 aggregate sweep bindings are missing",
    )
    expected_roles = dict(locked.CHECKPOINT_ROLES)
    validation_sha = baseline_split.get("hashes", {}).get("used_val_sha256")
    _require(
        isinstance(validation_sha, str),
        "baseline validation split SHA is missing",
    )
    sweep_evidence: dict[str, Any] = {}
    for checkpoint, role in expected_roles.items():
        path = locked.sweep_path(locked.BASELINE_RUN_DIR, checkpoint)
        payload = load_json(path)
        for field, expected in {
            "dataset": locked.DATASET,
            "seed": locked.TRAINING_SEED,
            "split_seed": locked.SPLIT_SEED,
            "checkpoint_role": role,
            "validation_count": 133,
            "validation_split_sha256": validation_sha,
            "official_test_accessed": False,
            "reference_artifact_validation_passed": True,
        }.items():
            _require(
                payload.get(field) == expected,
                f"baseline {checkpoint} sweep differs: {field}",
            )
        audit = payload.get("audit")
        _require(
            isinstance(audit, Mapping),
            f"baseline {checkpoint} audit is missing",
        )
        _require(
            audit.get("selection_source") == "internal_validation_only",
            f"baseline {checkpoint} selection source differs",
        )
        audit_protocol = audit.get("protocol")
        _require(
            isinstance(audit_protocol, Mapping)
            and audit_protocol.get("checkpoint_policy")
            == BASELINE_CHECKPOINT_POLICY,
            f"baseline {checkpoint} audited checkpoint policy differs",
        )
        checks = audit.get("integrity_checks_passed")
        _require(
            isinstance(checks, Mapping)
            and checks.get("official_test_isolated") is True
            and checks.get("checkpoint_role_epoch_metrics_consistent") is True
            and checks.get("global_selection_keys_recomputed") is True,
            f"baseline {checkpoint} selection audit is incomplete",
        )
        coverage = payload.get("final_metric_coverage")
        _require(
            isinstance(coverage, Mapping)
            and coverage.get("all_required_metrics_present") is True,
            f"baseline {checkpoint} metric coverage is incomplete",
        )
        key = f"{locked.BASELINE_VARIANT}:{checkpoint}"
        binding = sweep_bindings.get(key)
        _require(
            isinstance(binding, Mapping),
            f"V2 aggregate lacks baseline binding: {checkpoint}",
        )
        _require(
            binding.get("path") == str(path.resolve())
            and binding.get("sha256") == sha256_file(path),
            f"V2 aggregate baseline binding differs: {checkpoint}",
        )
        sweep_evidence[checkpoint] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "checkpoint_role": role,
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }
    rows = report.get("rows")
    _require(isinstance(rows, list), "V2 aggregate rows are missing")
    baseline_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("variant") == locked.BASELINE_VARIANT
    ]
    _require(
        len(baseline_rows) == len(expected_roles),
        "V2 aggregate baseline row count differs",
    )
    observed_roles = {
        row.get("checkpoint_role")
        for row in baseline_rows
        if row.get("source") == "same_protocol_external_reference"
        and row.get("run_directory") == str(locked.BASELINE_RUN_DIR.resolve())
    }
    _require(
        observed_roles == set(expected_roles.values()),
        "V2 aggregate baseline role mapping differs",
    )
    return {
        "legacy_schema_top_level_selection_source_absent": True,
        "checkpoint_policy": BASELINE_CHECKPOINT_POLICY,
        "checkpoint_policy_proves_internal_validation": True,
        "sweeps": sweep_evidence,
        "v2_aggregate": {
            "json_path": str(locked.v2_post.JSON_OUTPUT.resolve()),
            "json_sha256": sha256_file(locked.v2_post.JSON_OUTPUT),
            "markdown_path": str(locked.v2_post.MARKDOWN_OUTPUT.resolve()),
            "markdown_sha256": sha256_file(
                locked.v2_post.MARKDOWN_OUTPUT
            ),
            "completion_marker_path": str(
                locked.v2_post.COMPLETE_MARKER.resolve()
            ),
            "completion_marker_sha256": sha256_file(
                locked.v2_post.COMPLETE_MARKER
            ),
            "scope": "single_seed_internal_validation",
            "official_test_accessed": False,
            "baseline_role_mapping_verified": True,
        },
    }


def _validate_per_variant_checkpoint_selection(
    run_dirs: Mapping[str, Path],
) -> dict[str, Any]:
    """Record that every row uses that variant's own selected checkpoint."""

    expected_sweep_variants = {
        locked.VARIANT_V3_ON: locked.VARIANT_V3_ON,
        locked.VARIANT_V2_ON: locked.VARIANT_V2_ON,
        locked.VARIANT_V1_OFF: locked.VARIANT_V1_OFF,
        locked.BASELINE_VARIANT: "original",
    }
    matrix: dict[str, Any] = {}
    for variant, run_dir in run_dirs.items():
        checkpoints: dict[str, Any] = {}
        for checkpoint, role in locked.CHECKPOINT_ROLES.items():
            checkpoint_path = (run_dir / checkpoint).resolve()
            sweep_path = locked.sweep_path(run_dir, checkpoint).resolve()
            payload = load_json(sweep_path)
            _require(
                payload.get("checkpoint") == str(checkpoint_path),
                f"{variant} {checkpoint} sweep checkpoint path differs",
            )
            _require(
                payload.get("checkpoint_role") == role,
                f"{variant} {checkpoint} sweep role differs",
            )
            _require(
                payload.get("variant") == expected_sweep_variants[variant],
                f"{variant} {checkpoint} sweep variant differs",
            )
            _require(
                payload.get("official_test_accessed") is False,
                f"{variant} {checkpoint} accessed the official test set",
            )
            checkpoint_epoch = payload.get("checkpoint_epoch")
            _require(
                type(checkpoint_epoch) is int
                and 1 <= checkpoint_epoch <= locked.EXPECTED_EPOCHS,
                f"{variant} {checkpoint} epoch is invalid",
            )
            recorded_checkpoint_sha = payload.get("checkpoint_sha256")
            observed_checkpoint_sha = sha256_file(checkpoint_path)
            _require(
                recorded_checkpoint_sha == observed_checkpoint_sha,
                f"{variant} {checkpoint} checkpoint SHA-256 differs",
            )
            checkpoints[checkpoint] = {
                "selection_owner": variant,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_role": role,
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_sha256": observed_checkpoint_sha,
                "sweep_path": str(sweep_path),
                "sweep_sha256": sha256_file(sweep_path),
                "selection_source": "internal_validation_only",
                "official_test_accessed": False,
            }
        matrix[variant] = {
            "uses_own_checkpoint_directory": True,
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        }
    return matrix


def repaired_same_split_and_training_contract() -> dict[str, Any]:
    """V3 contract with exact modern equality and legacy baseline evidence."""

    attestation = verify_repair_attestation()
    upstream = locked.v2_post.same_split_and_training_contract()
    run_dirs = {
        locked.VARIANT_V3_ON: locked.V3_RUN_DIR,
        locked.VARIANT_V2_ON: locked.V2_RUN_DIR,
        locked.VARIANT_V1_OFF: locked.V1_OFF_RUN_DIR,
        locked.BASELINE_VARIANT: locked.BASELINE_RUN_DIR,
    }
    protocols = {
        name: load_json(path / "protocol.json")
        for name, path in run_dirs.items()
    }
    splits = {
        name: load_json(path / "split.json")
        for name, path in run_dirs.items()
    }
    arguments: dict[str, Mapping[str, Any]] = {}
    for name, protocol in protocols.items():
        value = protocol.get("arguments")
        _require(isinstance(value, Mapping), f"{name} arguments are missing")
        arguments[name] = value
    fixed_axes = dict(upstream["fixed_training_axes"])
    for source, source_arguments in arguments.items():
        for name, expected in fixed_axes.items():
            _require(
                source_arguments.get(name) == expected,
                f"{source} training axis differs: {name}",
            )
    reference_split = splits[locked.VARIANT_V1_OFF]
    for source, split in splits.items():
        for name, expected in {
            "dataset": locked.DATASET,
            "split_seed": locked.SPLIT_SEED,
            "used_train_count": 530,
            "used_val_count": 133,
            "full_official_train_count": 663,
            "official_test_accessed": False,
        }.items():
            _require(
                split.get(name) == expected,
                f"{source} split differs: {name}",
            )
        for name in ("used_train_ids", "used_val_ids", "hashes"):
            _require(
                split.get(name) == reference_split.get(name),
                f"{source} ordered split differs: {name}",
            )
    for field in (
        "normalization",
        "optimizer",
        "loss",
        "primary_selection_rule",
        "secondary_selection_rule",
    ):
        expected = protocols[locked.VARIANT_V1_OFF].get(field)
        _require(expected is not None, f"V1 off protocol lacks {field}")
        for source, protocol in protocols.items():
            _require(
                protocol.get(field) == expected,
                f"{source} protocol differs: {field}",
            )
    modern_selection_source = protocols[locked.VARIANT_V1_OFF].get(
        "selection_source"
    )
    _require(
        modern_selection_source == "internal_validation_only",
        "V1 selection source differs",
    )
    modern_checkpoint_policy = protocols[locked.VARIANT_V1_OFF].get(
        "checkpoint_policy"
    )
    _require(
        isinstance(modern_checkpoint_policy, str),
        "V1 checkpoint policy is missing",
    )
    for source in MODERN_VARIANTS:
        _require(
            protocols[source].get("selection_source")
            == modern_selection_source,
            f"{source} selection source differs from V1",
        )
        _require(
            protocols[source].get("checkpoint_policy")
            == modern_checkpoint_policy,
            f"{source} checkpoint policy differs from V1",
        )
    _require(
        arguments[locked.VARIANT_V3_ON].get("parent_variant")
        == arguments[locked.VARIANT_V2_ON].get("parent_variant")
        == arguments[locked.VARIANT_V1_OFF].get("parent_variant")
        == "tpd_clean_v8_mprs_dch_full",
        "V3/V2/V1 parent variants differ",
    )
    _require(
        arguments[locked.VARIANT_V3_ON].get("relay_enabled") is True
        and arguments[locked.VARIANT_V2_ON].get("relay_enabled") is True
        and arguments[locked.VARIANT_V1_OFF].get("relay_enabled") is False,
        "V3/V2/V1 relay identities differ",
    )
    design = protocols[locked.VARIANT_V3_ON].get("comparison_design")
    _require(isinstance(design, Mapping), "V3 comparison design is missing")
    _require(
        design.get("required_control") == locked.VARIANT_V1_OFF,
        "V3 required control differs",
    )
    _require(
        design.get("structural_predecessor") == locked.VARIANT_V2_ON,
        "V3 structural predecessor differs",
    )
    _require(
        design.get("relay_off_retrained") is False,
        "V3 relay-off retraining policy differs",
    )
    baseline_evidence = _validate_v2_aggregate_and_baseline_evidence(
        baseline_protocol=protocols[locked.BASELINE_VARIANT],
        baseline_split=splits[locked.BASELINE_VARIANT],
    )
    checkpoint_selection = _validate_per_variant_checkpoint_selection(
        run_dirs
    )
    return {
        "training_seed": locked.TRAINING_SEED,
        "split_seed": locked.SPLIT_SEED,
        "multi_seed_scheduled": False,
        "same_fixed_training_axes": True,
        "same_normalization": True,
        "same_optimizer": True,
        "same_loss": True,
        "same_selection_rules": True,
        "same_checkpoint_roles": True,
        "same_ordered_train_ids": True,
        "same_ordered_validation_ids": True,
        "same_split_hashes": True,
        "official_test_accessed": False,
        "required_control": locked.VARIANT_V1_OFF,
        "structural_predecessor": locked.VARIANT_V2_ON,
        "relay_off_retrained": False,
        "fixed_training_axes": fixed_axes,
        "upstream_v2_contract": upstream,
        "selection_contract_repair": {
            "schema": CONTRACT_SCHEMA,
            "repair_id": REPAIR_ID,
            "logic_override": "same_split_and_training_contract",
            "aggregate_implementation": "frozen_v3_aggregate_and_write",
            "modern_variants": list(MODERN_VARIANTS),
            "modern_selection_source_exact": modern_selection_source,
            "modern_checkpoint_policy_exact": modern_checkpoint_policy,
            "modern_exact_equality_verified": True,
            "baseline_policy_text_equality_to_modern_required": False,
            "baseline_top_level_selection_source_required": False,
            "baseline_internal_validation_evidence": baseline_evidence,
            "each_variant_uses_own_selected_checkpoints": True,
            "per_variant_checkpoint_selection": checkpoint_selection,
            "attestation": attestation,
        },
    }


@contextmanager
def _locked_aggregate_repair_scope() -> Iterator[None]:
    """Patch one logic function and repair-owned publication paths only."""

    _require(
        locked.same_split_and_training_contract
        is _FROZEN_CONTRACT_FUNCTION,
        "frozen V3 contract function was already replaced",
    )
    originals = {
        "same_split_and_training_contract": (
            locked.same_split_and_training_contract
        ),
        "COMPARISON_DIR": locked.COMPARISON_DIR,
        "JSON_OUTPUT": locked.JSON_OUTPUT,
        "MARKDOWN_OUTPUT": locked.MARKDOWN_OUTPUT,
        "COMPLETE_MARKER": locked.COMPLETE_MARKER,
    }
    locked.same_split_and_training_contract = (
        repaired_same_split_and_training_contract
    )
    locked.COMPARISON_DIR = REPAIR_COMPARISON_DIR
    locked.JSON_OUTPUT = REPAIR_JSON_OUTPUT
    locked.MARKDOWN_OUTPUT = REPAIR_MARKDOWN_OUTPUT
    locked.COMPLETE_MARKER = REPAIR_COMPLETE_MARKER
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(locked, name, value)


def aggregate_and_write() -> tuple[
    dict[str, Any],
    tuple[Path, Path, Path],
]:
    """Run the frozen aggregate under the one-function repair scope."""

    with _locked_aggregate_repair_scope():
        report, paths = locked.aggregate_and_write()
        _require(
            tuple(paths)
            == (
                REPAIR_JSON_OUTPUT,
                REPAIR_MARKDOWN_OUTPUT,
                REPAIR_COMPLETE_MARKER,
            ),
            "repair publication paths differ",
        )
        repair = report.get("comparison_contract", {}).get(
            "selection_contract_repair"
        )
        _require(
            isinstance(repair, Mapping)
            and repair.get("repair_id") == REPAIR_ID,
            "repair report lacks selection-contract attestation",
        )
        return report, paths


def verify_repair_closure() -> dict[str, Any]:
    """Read-only verification of locks, contract, sweeps, and row matrix."""

    frozen_bindings = locked.verify_frozen_manifests()
    before = locked.upstream_snapshot()
    contract = repaired_same_split_and_training_contract()
    attestation = contract["selection_contract_repair"]["attestation"]
    rows = locked.load_all_rows()
    after = locked.upstream_snapshot()
    expected_rows = {
        (variant, checkpoint)
        for variant in ALL_VARIANTS
        for checkpoint in locked.CHECKPOINTS
    }
    _require(set(rows) == expected_rows, "repair eight-row matrix differs")
    _require(before == after, "upstream artifacts changed during verification")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "verified",
        "repair_id": REPAIR_ID,
        "gpu_work": False,
        "frozen_artifacts_modified": False,
        "aggregate_implementation": "frozen_v3_aggregate_and_write",
        "logic_override": "same_split_and_training_contract",
        "row_count": len(rows),
        "output_directory": str(REPAIR_COMPARISON_DIR.resolve()),
        "outputs_exist": {
            str(path.resolve()): path.exists() or path.is_symlink()
            for path in (
                REPAIR_JSON_OUTPUT,
                REPAIR_MARKDOWN_OUTPUT,
                REPAIR_COMPLETE_MARKER,
            )
        },
        "frozen_bindings": frozen_bindings,
        "attestation": attestation,
        "selection_contract_repair": contract[
            "selection_contract_repair"
        ],
    }


def execution_plan() -> dict[str, Any]:
    attestation = verify_repair_attestation()
    return {
        "schema": CONTRACT_SCHEMA,
        "repair_id": REPAIR_ID,
        "mode": "aggregate_existing_locked_sweeps_only",
        "new_evaluation_count": 0,
        "gpu_work": False,
        "logic_overrides": ["same_split_and_training_contract"],
        "publication_path_overrides": {
            "comparison_dir": str(REPAIR_COMPARISON_DIR.resolve()),
            "json": str(REPAIR_JSON_OUTPUT.resolve()),
            "markdown": str(REPAIR_MARKDOWN_OUTPUT.resolve()),
            "marker": str(REPAIR_COMPLETE_MARKER.resolve()),
        },
        "frozen_v3_outputs_are_publication_targets": False,
        "attestation": attestation,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair only the V3 formal800 legacy-baseline selection contract"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def _print_json(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.plan:
        _print_json(execution_plan())
        return
    if args.verify_only:
        _print_json(verify_repair_closure())
        return
    report, paths = aggregate_and_write()
    print(
        f"REPAIRED_AGGREGATE decision={report['decision']} "
        f"json={paths[0]} markdown={paths[1]} marker={paths[2]}",
        flush=True,
    )


__all__ = [
    "ATTESTATION",
    "CONTRACT_SCHEMA",
    "REPAIR_COMPARISON_DIR",
    "REPAIR_COMPLETE_MARKER",
    "REPAIR_ID",
    "REPAIR_JSON_OUTPUT",
    "REPAIR_MARKDOWN_OUTPUT",
    "aggregate_and_write",
    "execution_plan",
    "repaired_same_split_and_training_contract",
    "required_attestation_paths",
    "verify_repair_attestation",
    "verify_repair_closure",
]


if __name__ == "__main__":
    main()
