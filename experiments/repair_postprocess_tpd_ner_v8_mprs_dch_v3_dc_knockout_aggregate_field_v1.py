#!/usr/bin/env python3
"""Versioned aggregate-field repair for the frozen V3 DC-knockout V2.

The six diagnostic sources, V2 source lock, two completed sweeps, and formal
comparison stay byte-identical.  This wrapper reuses the frozen sweep
validator, report builder, and Markdown renderer.  Its sole data-mapping
correction is:

    formal pd_at_fa_budget point["fa"] -> normalized["achieved_fa"]

Publication, when explicitly requested, is restricted to a new versioned
comparison directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock as freezer,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_v3_dc_knockout as frozen,
)
from experiments import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)


REPAIR_ID = "v3_dc_knockout_v2_aggregate_field_repair_v1"
REPAIR_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_"
    "aggregate_field_repair_v1"
)
ATTESTATION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_"
    "aggregate_field_repair_attestation_v1"
)
COMPLETE_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_"
    "aggregate_field_repair_complete_v1"
)
PROTOCOL = (
    REPO_ROOT
    / "experiments/"
    "TPD_NER_V8_MPRS_DCH_V3_DC_KNOCKOUT_AGGREGATE_FIELD_REPAIR_V1.md"
)
ATTESTATION = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v8_mprs_dch_v3_dc_knockout_"
    "aggregate_field_repair_v1_attestation.json"
)
V2_SOURCE_LOCK = freezer.DEFAULT_SOURCE_LOCK
FORMAL_REPAIRED_AGGREGATE = freezer.DEFAULT_FORMAL_REPORT
REPAIR_COMPARISON_DIR = (
    spec.DEFAULT_OUTPUT_ROOT
    / spec.DATASET
    / "comparison_aggregate_field_repair_v1"
)
REPAIR_JSON_OUTPUT = (
    REPAIR_COMPARISON_DIR
    / "tpd_ner_v8_mprs_dch_v3_dc_knockout_comparison_"
    "aggregate_field_repair_v1.json"
)
REPAIR_MARKDOWN_OUTPUT = (
    REPAIR_COMPARISON_DIR
    / "tpd_ner_v8_mprs_dch_v3_dc_knockout_comparison_"
    "aggregate_field_repair_v1.md"
)
REPAIR_COMPLETE_MARKER = (
    REPAIR_COMPARISON_DIR
    / "DC_KNOCKOUT_COMPLETE_AGGREGATE_FIELD_REPAIR_V1.json"
)
EXPECTED_V2_SOURCE_LOCK_SHA256 = (
    "ec72d67fbd992eacc2045a5cd5d39a1de9c7daed79a73a7cb92d26512436ea7a"
)
EXPECTED_SWEEP_SHA256 = {
    "best.pth.tar": (
        "92dc2a31a325a6a91382f0fd122d039c0730847dbbf84b4424cef08587a1866b"
    ),
    "best_miou.pth.tar": (
        "ee0f45566b5558db059efc3c1fcb09ed2f511b1f80ba070b03abcb05a4613a26"
    ),
}
EXPECTED_FORMAL_REPAIRED_AGGREGATE_SHA256 = (
    "68e36d8afc1620821b61cae76138d90b9ddeb8f2143e76a307c76164c63cb711"
)
EXPECTED_FROZEN_POSTPROCESSOR_SHA256 = (
    "eb88410b10ebb5e19191fdffa285c613a1b14363b5ca38ea1f4deace511fd5ef"
)
ATTESTATION_POLICY = {
    "frozen_artifacts_modified": False,
    "logic_override": (
        "load_formal_reference_rows.pd_at_fa_budget.field_mapping_only"
    ),
    "source_field": "fa",
    "normalized_field": "achieved_fa",
    "reused_frozen_functions": [
        "validate_checkpoint_sweep",
        "build_report",
        "render_markdown",
    ],
    "new_evaluation_count": 0,
    "gpu_work": False,
    "publication_scope": "comparison_aggregate_field_repair_v1",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    _require(
        value.is_file() and not value.is_symlink(),
        f"{label} must be a regular non-symlink file: {value}",
    )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path, "hashed artifact").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(
        _regular_file(path, label).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {token}")
        ),
    )
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def required_attestation_paths() -> dict[str, Path]:
    return {
        "source:frozen_v2_postprocessor": Path(frozen.__file__).resolve(),
        "source:repair_wrapper": Path(__file__).resolve(),
        "source:repair_protocol": PROTOCOL.resolve(),
        "artifact:v2_source_lock": V2_SOURCE_LOCK.resolve(),
        "artifact:v2_sweep_best": spec.sweep_path(
            "best.pth.tar",
            spec.DEFAULT_OUTPUT_ROOT,
        ).resolve(),
        "artifact:v2_sweep_best_miou": spec.sweep_path(
            "best_miou.pth.tar",
            spec.DEFAULT_OUTPUT_ROOT,
        ).resolve(),
        "artifact:formal_repaired_aggregate": (
            FORMAL_REPAIRED_AGGREGATE.resolve()
        ),
    }


def verify_repair_attestation(
    path: Path = ATTESTATION,
) -> dict[str, Any]:
    payload = _load_json(path, "aggregate-field repair attestation")
    _require(
        payload.get("schema") == ATTESTATION_SCHEMA,
        "aggregate-field repair attestation schema differs",
    )
    _require(payload.get("status") == "frozen", "attestation is not frozen")
    _require(payload.get("repair_id") == REPAIR_ID, "repair identity differs")
    _require(
        payload.get("policy") == ATTESTATION_POLICY,
        "aggregate-field repair policy differs",
    )
    bindings = payload.get("bindings")
    expected = required_attestation_paths()
    _require(isinstance(bindings, Mapping), "attestation bindings are missing")
    _require(
        set(bindings) == set(expected),
        "aggregate-field repair binding matrix differs",
    )
    verified: dict[str, dict[str, str]] = {}
    for name, expected_path in expected.items():
        entry = bindings.get(name)
        _require(
            isinstance(entry, Mapping)
            and set(entry) == {"path", "sha256"},
            f"{name} binding fields differ",
        )
        _require(
            entry.get("path") == str(expected_path.relative_to(REPO_ROOT)),
            f"{name} binding path differs",
        )
        observed = sha256_file(expected_path)
        _require(entry.get("sha256") == observed, f"{name} SHA-256 differs")
        verified[name] = {
            "path": str(expected_path),
            "sha256": observed,
        }
    _require(
        verified["artifact:v2_source_lock"]["sha256"]
        == EXPECTED_V2_SOURCE_LOCK_SHA256,
        "V2 diagnostic source-lock SHA-256 differs",
    )
    _require(
        verified["source:frozen_v2_postprocessor"]["sha256"]
        == EXPECTED_FROZEN_POSTPROCESSOR_SHA256,
        "frozen V2 postprocessor SHA-256 differs",
    )
    _require(
        verified["artifact:formal_repaired_aggregate"]["sha256"]
        == EXPECTED_FORMAL_REPAIRED_AGGREGATE_SHA256,
        "formal repaired aggregate SHA-256 differs",
    )
    for checkpoint, binding_name in {
        "best.pth.tar": "artifact:v2_sweep_best",
        "best_miou.pth.tar": "artifact:v2_sweep_best_miou",
    }.items():
        _require(
            verified[binding_name]["sha256"]
            == EXPECTED_SWEEP_SHA256[checkpoint],
            f"{checkpoint} V2 knockout sweep SHA-256 differs",
        )
    return {
        "schema": ATTESTATION_SCHEMA,
        "status": "verified",
        "repair_id": REPAIR_ID,
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "bindings": verified,
    }


def load_formal_reference_rows_repaired(
    path: Path = FORMAL_REPAIRED_AGGREGATE,
) -> dict[str, dict[str, Any]]:
    """Normalize formal rows with the one corrected canonical field mapping."""

    report = frozen._load_json(path, "repaired formal V3 aggregate report")
    frozen._require(
        report.get("status") == "complete"
        and report.get("row_count") == 8
        and report.get("dataset") == spec.DATASET
        and report.get("training_seed") == spec.TRAINING_SEED
        and report.get("split_seed") == spec.SPLIT_SEED,
        "repaired formal V3 aggregate identity differs",
    )
    repair = frozen._mapping(
        frozen._mapping(
            report.get("comparison_contract"),
            "repaired formal comparison contract",
        ).get("selection_contract_repair"),
        "repaired formal selection contract",
    )
    frozen._require(
        report.get("decision") == freezer.EXPECTED_FORMAL_DECISION
        and report.get("aggregate_full_model_gate_passed") is False
        and repair.get("repair_id") == freezer.FORMAL_REPAIR_ID
        and repair.get("each_variant_uses_own_selected_checkpoints") is True,
        "repaired formal aggregate authority differs",
    )
    rows = report.get("rows")
    frozen._require(
        isinstance(rows, list),
        "formal V3 aggregate rows are missing",
    )
    references: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            isinstance(row, Mapping)
            and row.get("variant") == spec.VARIANT
            and row.get("checkpoint_role") in spec.CHECKPOINT_ROLES.values()
        ):
            role = str(row["checkpoint_role"])
            frozen._require(
                role not in references,
                f"duplicate formal V3 role: {role}",
            )
            fixed = frozen._normalize_fixed(
                row.get("fixed_threshold_0_5"),
                location=f"formal reference {role}.fixed",
            )
            raw_budgets = frozen._mapping(
                row.get("pd_at_fa_budget"),
                f"formal reference {role}.budgets",
            )
            frozen._require(
                set(raw_budgets) == set(spec.BUDGET_KEYS),
                f"formal reference {role} budget keys differ",
            )
            budgets: dict[str, Any] = {}
            for budget, key in zip(spec.FA_BUDGETS, spec.BUDGET_KEYS):
                point = frozen._mapping(
                    raw_budgets[key],
                    f"formal reference {role}.{key}",
                )
                frozen._require(
                    set(point)
                    == {
                        "fa",
                        "pd",
                        "threshold",
                        "matched_target_count",
                        "target_count",
                    },
                    f"formal reference {role}.{key} canonical fields differ",
                )
                budgets[key] = {
                    "budget": budget,
                    "pd": frozen._finite(
                        point.get("pd"),
                        f"formal {role}.{key}.pd",
                    ),
                    # The sole semantic repair: canonical formal rows use
                    # ``fa`` while knockout-normalized rows use
                    # ``achieved_fa``.
                    "achieved_fa": frozen._finite(
                        point.get("fa"),
                        f"formal {role}.{key}.fa",
                    ),
                    "threshold": frozen._finite(
                        point.get("threshold"),
                        f"formal {role}.{key}.threshold",
                    ),
                    "matched_target_count": frozen._integer(
                        point.get("matched_target_count"),
                        f"formal {role}.{key}.matched",
                        minimum=0,
                        maximum=spec.TARGET_COUNT,
                    ),
                    "target_count": frozen._integer(
                        point.get("target_count"),
                        f"formal {role}.{key}.target_count",
                        minimum=spec.TARGET_COUNT,
                        maximum=spec.TARGET_COUNT,
                    ),
                }
            references[role] = {
                "fixed_threshold_0_5": fixed,
                "pd_at_fa_budget": budgets,
                "formal_row_sha256": spec.canonical_sha256(dict(row)),
            }
    frozen._require(
        set(references) == set(spec.CHECKPOINT_ROLES.values()),
        "formal V3 learned reference roles differ",
    )
    return references


def _repair_bindings(
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    values = attestation["bindings"]
    return {
        "v2_diagnostic_source_lock": copy.deepcopy(
            values["artifact:v2_source_lock"]
        ),
        "v2_knockout_sweeps": {
            "best.pth.tar": copy.deepcopy(
                values["artifact:v2_sweep_best"]
            ),
            "best_miou.pth.tar": copy.deepcopy(
                values["artifact:v2_sweep_best_miou"]
            ),
        },
        "formal_repaired_aggregate": copy.deepcopy(
            values["artifact:formal_repaired_aggregate"]
        ),
        "frozen_v2_postprocessor": copy.deepcopy(
            values["source:frozen_v2_postprocessor"]
        ),
        "repair_wrapper": copy.deepcopy(values["source:repair_wrapper"]),
        "repair_protocol": copy.deepcopy(values["source:repair_protocol"]),
        "repair_attestation": {
            "path": attestation["path"],
            "sha256": attestation["sha256"],
        },
    }


def build_repaired_report_in_memory() -> dict[str, Any]:
    """Read, validate, normalize, and build without publishing any file."""

    attestation = verify_repair_attestation()
    before = freezer.verify_source_lock(V2_SOURCE_LOCK)
    frozen._require(
        sha256_file(V2_SOURCE_LOCK) == EXPECTED_V2_SOURCE_LOCK_SHA256,
        "V2 source-lock digest differs",
    )
    expected_source_binding = freezer.current_source_binding(V2_SOURCE_LOCK)
    formal_binding = frozen._mapping(
        before.get("formal_artifact_binding"),
        "diagnostic source lock formal binding",
    )
    formal_report = frozen._validate_repaired_formal_report_input(
        FORMAL_REPAIRED_AGGREGATE,
        formal_binding=formal_binding,
    )
    rows: list[dict[str, Any]] = []
    sweep_bindings: dict[str, dict[str, str]] = {}
    for checkpoint in spec.CHECKPOINTS:
        path = spec.sweep_path(checkpoint, spec.DEFAULT_OUTPUT_ROOT)
        observed_sha256 = sha256_file(path)
        frozen._require(
            observed_sha256 == EXPECTED_SWEEP_SHA256[checkpoint],
            f"{checkpoint} V2 knockout sweep SHA-256 differs",
        )
        rows.extend(
            frozen.validate_checkpoint_sweep(
                path,
                checkpoint=checkpoint,
                expected_source_binding=expected_source_binding,
            )
        )
        sweep_bindings[checkpoint] = {
            "path": str(path.resolve()),
            "sha256": observed_sha256,
        }
    references = load_formal_reference_rows_repaired(formal_report)
    report = frozen.build_report(
        rows,
        references=references,
        source_lock_payload=before,
        source_lock_path=V2_SOURCE_LOCK,
        sweep_bindings=sweep_bindings,
        formal_report_path=formal_report,
    )
    bindings = _repair_bindings(attestation)
    report["aggregate_field_repair"] = {
        "schema": REPAIR_SCHEMA,
        "repair_id": REPAIR_ID,
        "logic_override": ATTESTATION_POLICY["logic_override"],
        "mapping": {
            "source_container": "formal_row.pd_at_fa_budget",
            "source_field": "fa",
            "normalized_field": "achieved_fa",
        },
        "reused_frozen_functions": copy.deepcopy(
            ATTESTATION_POLICY["reused_frozen_functions"]
        ),
        "frozen_artifacts_modified": False,
        "new_evaluation_count": 0,
        "gpu_work": False,
        "bindings": bindings,
    }
    report["source_binding"]["aggregate_field_repair"] = copy.deepcopy(
        bindings
    )
    after = freezer.verify_source_lock(V2_SOURCE_LOCK)
    frozen._require(
        before == after,
        "frozen diagnostic or formal inputs changed during repair validation",
    )
    return report


def render_repaired_markdown(report: Mapping[str, Any]) -> str:
    """Render the unchanged table through the frozen renderer."""

    base = frozen.render_markdown(report).rstrip()
    return (
        base
        + "\n\n## Aggregate field repair V1\n\n"
        + "- Formal budget mapping: `point[\"fa\"] -> achieved_fa`\n"
        + "- Frozen sweep validator, report builder, and table renderer reused\n"
        + "- New evaluation count: `0`; formal decision affected: `false`\n"
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _output_paths() -> tuple[Path, Path, Path]:
    expected_directory = (
        spec.validated_output_root(spec.DEFAULT_OUTPUT_ROOT)
        / spec.DATASET
        / "comparison_aggregate_field_repair_v1"
    )
    _require(
        REPAIR_COMPARISON_DIR.resolve() == expected_directory.resolve()
        and REPAIR_COMPARISON_DIR.resolve()
        != spec.DEFAULT_COMPARISON_DIR.resolve(),
        "aggregate-field repair publication directory differs",
    )
    return (
        REPAIR_JSON_OUTPUT,
        REPAIR_MARKDOWN_OUTPUT,
        REPAIR_COMPLETE_MARKER,
    )


def _marker_payload(
    report: Mapping[str, Any],
    json_bytes: bytes,
    markdown_bytes: bytes,
) -> dict[str, Any]:
    bindings = report["aggregate_field_repair"]["bindings"]
    return {
        "schema": COMPLETE_MARKER_SCHEMA,
        "status": "complete",
        "repair_id": REPAIR_ID,
        "artifact_kind": spec.ARTIFACT_KIND,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "row_count": spec.EXPECTED_ROW_COUNT,
        "mapping": {
            "source_field": "fa",
            "normalized_field": "achieved_fa",
        },
        "bindings": copy.deepcopy(bindings),
        "bindings_sha256": spec.canonical_sha256(bindings),
        "outputs": {
            REPAIR_JSON_OUTPUT.name: hashlib.sha256(json_bytes).hexdigest(),
            REPAIR_MARKDOWN_OUTPUT.name: hashlib.sha256(
                markdown_bytes
            ).hexdigest(),
        },
    }


def _atomic_publish_new(path: Path, content: bytes) -> None:
    _require(
        not path.exists() and not path.is_symlink(),
        f"refusing to overwrite repair artifact: {path}",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(path) from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_repaired_report(
    report: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    """Publish only the versioned repair package, never the V2 comparison."""

    json_path, markdown_path, marker_path = _output_paths()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        json_path.parent.is_dir() and not json_path.parent.is_symlink(),
        "aggregate-field repair comparison directory differs",
    )
    json_bytes = _json_bytes(report)
    markdown_bytes = render_repaired_markdown(report).encode("utf-8")
    marker_bytes = _json_bytes(
        _marker_payload(report, json_bytes, markdown_bytes)
    )
    for path, content in (
        (json_path, json_bytes),
        (markdown_path, markdown_bytes),
        (marker_path, marker_bytes),
    ):
        if path.exists() or path.is_symlink():
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == content,
                f"existing repair artifact conflicts: {path}",
            )
        else:
            _atomic_publish_new(path, content)
    return json_path, markdown_path, marker_path


def inspect_complete() -> dict[str, Any] | None:
    """Strictly validate the versioned package after publication."""

    json_path, markdown_path, marker_path = _output_paths()
    if not marker_path.exists():
        _require(
            not marker_path.is_symlink(),
            "repair completion marker may not be a symlink",
        )
        return None
    marker = _load_json(marker_path, "aggregate-field repair marker")
    expected_marker_fields = {
        "schema",
        "status",
        "repair_id",
        "artifact_kind",
        "diagnostic_only",
        "affects_formal_gate",
        "formal_decision_authority",
        "row_count",
        "mapping",
        "bindings",
        "bindings_sha256",
        "outputs",
    }
    _require(
        set(marker) == expected_marker_fields,
        "aggregate-field repair marker field set differs",
    )
    _require(
        marker.get("schema") == COMPLETE_MARKER_SCHEMA
        and marker.get("status") == "complete"
        and marker.get("repair_id") == REPAIR_ID
        and marker.get("artifact_kind") == spec.ARTIFACT_KIND
        and marker.get("diagnostic_only") is True
        and marker.get("affects_formal_gate") is False
        and marker.get("formal_decision_authority") is False
        and marker.get("row_count") == spec.EXPECTED_ROW_COUNT
        and marker.get("mapping")
        == {
            "source_field": "fa",
            "normalized_field": "achieved_fa",
        },
        "aggregate-field repair marker identity differs",
    )
    _require(
        not frozen.FORBIDDEN_DECISION_FIELDS.intersection(marker),
        "aggregate-field repair marker contains a formal decision field",
    )
    report = _load_json(json_path, "aggregate-field repaired report")
    _regular_file(markdown_path, "aggregate-field repaired Markdown")
    _require(
        report.get("schema") == frozen.SCHEMA
        and report.get("status") == "complete"
        and report.get("artifact_kind") == spec.ARTIFACT_KIND
        and report.get("diagnostic_only") is True
        and report.get("affects_formal_gate") is False
        and report.get("formal_decision_authority") is False
        and report.get("row_count") == spec.EXPECTED_ROW_COUNT
        and isinstance(report.get("rows"), list)
        and len(report["rows"]) == spec.EXPECTED_ROW_COUNT,
        "aggregate-field repaired report identity differs",
    )
    _require(
        not frozen.FORBIDDEN_DECISION_FIELDS.intersection(report),
        "aggregate-field repaired report contains a formal decision field",
    )
    repair = report.get("aggregate_field_repair")
    expected_repair_fields = {
        "schema",
        "repair_id",
        "logic_override",
        "mapping",
        "reused_frozen_functions",
        "frozen_artifacts_modified",
        "new_evaluation_count",
        "gpu_work",
        "bindings",
    }
    _require(
        isinstance(repair, Mapping)
        and set(repair) == expected_repair_fields,
        "aggregate-field repaired report repair field set differs",
    )
    _require(
        repair.get("schema") == REPAIR_SCHEMA
        and repair.get("repair_id") == REPAIR_ID
        and repair.get("logic_override") == ATTESTATION_POLICY["logic_override"]
        and repair.get("mapping")
        == {
            "source_container": "formal_row.pd_at_fa_budget",
            "source_field": "fa",
            "normalized_field": "achieved_fa",
        }
        and repair.get("reused_frozen_functions")
        == ATTESTATION_POLICY["reused_frozen_functions"]
        and repair.get("frozen_artifacts_modified") is False
        and repair.get("new_evaluation_count") == 0
        and repair.get("gpu_work") is False,
        "aggregate-field repaired report repair identity differs",
    )
    _require(
        not frozen.FORBIDDEN_DECISION_FIELDS.intersection(repair),
        "aggregate-field repair metadata contains a formal decision field",
    )
    expected_bindings = _repair_bindings(verify_repair_attestation())
    _require(
        repair.get("bindings") == expected_bindings
        and marker.get("bindings") == expected_bindings
        and marker.get("bindings_sha256")
        == spec.canonical_sha256(expected_bindings),
        "aggregate-field repair input bindings differ",
    )
    source_binding = report.get("source_binding")
    _require(
        isinstance(source_binding, Mapping)
        and source_binding.get("aggregate_field_repair")
        == expected_bindings,
        "aggregate-field repaired report source binding differs",
    )
    expected_outputs = {
        json_path.name: sha256_file(json_path),
        markdown_path.name: sha256_file(markdown_path),
    }
    _require(
        marker.get("outputs") == expected_outputs,
        "aggregate-field repair output hashes differ",
    )
    _require(
        json_path.read_bytes() == _json_bytes(report),
        "aggregate-field repaired report is not canonically encoded",
    )
    _require(
        markdown_path.read_text(encoding="utf-8")
        == render_repaired_markdown(report),
        "aggregate-field repaired Markdown content differs",
    )
    _require(
        marker_path.read_bytes() == _json_bytes(marker),
        "aggregate-field repair marker is not canonically encoded",
    )
    return marker


def verify_repair_closure() -> dict[str, Any]:
    """Perform the full CPU-only, read-only aggregate validation."""

    before_outputs = {
        str(path.resolve()): path.exists() or path.is_symlink()
        for path in _output_paths()
    }
    report = build_repaired_report_in_memory()
    markdown = render_repaired_markdown(report)
    after_outputs = {
        str(path.resolve()): path.exists() or path.is_symlink()
        for path in _output_paths()
    }
    _require(
        before_outputs == after_outputs,
        "verify-only changed repair publication state",
    )
    _require(
        report.get("row_count") == spec.EXPECTED_ROW_COUNT
        and len(report.get("rows", ())) == spec.EXPECTED_ROW_COUNT,
        "repaired aggregate row count differs",
    )
    _require(
        "point[\"fa\"] -> achieved_fa" in markdown,
        "repaired Markdown mapping note is missing",
    )
    return {
        "schema": REPAIR_SCHEMA,
        "status": "verified",
        "repair_id": REPAIR_ID,
        "mode": "read_only_existing_v2_sweeps",
        "row_count": spec.EXPECTED_ROW_COUNT,
        "formal_reference_role_count": len(spec.CHECKPOINT_ROLES),
        "mapping": {
            "source_field": "fa",
            "normalized_field": "achieved_fa",
        },
        "reused_frozen_functions": copy.deepcopy(
            ATTESTATION_POLICY["reused_frozen_functions"]
        ),
        "new_evaluation_count": 0,
        "gpu_work": False,
        "frozen_artifacts_modified": False,
        "bindings": copy.deepcopy(
            report["aggregate_field_repair"]["bindings"]
        ),
        "outputs_exist": after_outputs,
    }


def aggregate_and_write() -> tuple[
    dict[str, Any],
    tuple[Path, Path, Path],
]:
    report = build_repaired_report_in_memory()
    paths = write_repaired_report(report)
    marker = inspect_complete()
    _require(
        marker is not None and marker.get("repair_id") == REPAIR_ID,
        "published aggregate-field repair package is incomplete",
    )
    return report, paths


def execution_plan() -> dict[str, Any]:
    attestation = verify_repair_attestation()
    return {
        "schema": REPAIR_SCHEMA,
        "repair_id": REPAIR_ID,
        "mode": "aggregate_existing_v2_sweeps_only",
        "mapping": {
            "source_field": "fa",
            "normalized_field": "achieved_fa",
        },
        "reused_frozen_functions": copy.deepcopy(
            ATTESTATION_POLICY["reused_frozen_functions"]
        ),
        "new_evaluation_count": 0,
        "gpu_work": False,
        "publication_directory": str(REPAIR_COMPARISON_DIR.resolve()),
        "original_comparison_is_publication_target": False,
        "outputs": [str(path.resolve()) for path in _output_paths()],
        "attestation": attestation,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair only the V3 DC-knockout formal Fa field mapping"
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
        f"REPAIRED_DC_KNOCKOUT_AGGREGATE rows={report['row_count']} "
        f"json={paths[0]} markdown={paths[1]} marker={paths[2]}",
        flush=True,
    )


__all__ = [
    "ATTESTATION",
    "ATTESTATION_SCHEMA",
    "COMPLETE_MARKER_SCHEMA",
    "FORMAL_REPAIRED_AGGREGATE",
    "PROTOCOL",
    "REPAIR_COMPARISON_DIR",
    "REPAIR_COMPLETE_MARKER",
    "REPAIR_ID",
    "REPAIR_JSON_OUTPUT",
    "REPAIR_MARKDOWN_OUTPUT",
    "REPAIR_SCHEMA",
    "V2_SOURCE_LOCK",
    "aggregate_and_write",
    "build_repaired_report_in_memory",
    "execution_plan",
    "inspect_complete",
    "load_formal_reference_rows_repaired",
    "render_repaired_markdown",
    "required_attestation_paths",
    "sha256_file",
    "verify_repair_attestation",
    "verify_repair_closure",
    "write_repaired_report",
]


if __name__ == "__main__":
    main()
