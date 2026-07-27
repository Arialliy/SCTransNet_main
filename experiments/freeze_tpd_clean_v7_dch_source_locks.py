#!/usr/bin/env python3
"""Build, validate, or exclusively write the three V7-DCH source locks.

The locks are intentionally separate:

``diagnostic``
    The already executed frozen V6 failure-atlas path and its immutable
    diagnostic outputs.
``training``
    Exactly the runtime source set declared by the V7-DCH exact entry.  This
    produces the schema and default path consumed by that entry.
``acceptance``
    Fixed-threshold/sweep, Gates A--E, finalisation, and Mechanism Audit M.

Importing this module performs no hashing and writes nothing.  The CLI also
requires an explicit ``check``, ``freeze``, or ``validate`` action.  This makes
it possible to implement the lock contract before every future runtime file
exists, while refusing to freeze an incomplete set.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOCK_KINDS = ("diagnostic", "training", "acceptance")
DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1 = (
    "sctransnet_tpd_clean_v7_dch_diagnostic_source_lock_v1"
)
DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V2 = (
    "sctransnet_tpd_clean_v7_dch_diagnostic_source_lock_v2"
)
DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V3 = (
    "sctransnet_tpd_clean_v7_dch_diagnostic_source_lock_v3"
)
ACCEPTANCE_SOURCE_LOCK_SCHEMA_V1 = (
    "sctransnet_tpd_clean_v7_dch_acceptance_source_lock_v1"
)
ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2 = (
    "sctransnet_tpd_clean_v7_dch_acceptance_source_lock_v2"
)
ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3 = (
    "sctransnet_tpd_clean_v7_dch_acceptance_source_lock_v3"
)
ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4 = (
    "sctransnet_tpd_clean_v7_dch_acceptance_source_lock_v4"
)
SCHEMAS = {
    "diagnostic": DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V3,
    "training": "sctransnet_tpd_clean_v7_dch_exact_source_lock_v1",
    "acceptance": ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
}
SUPERSEDED_DIAGNOSTIC_LOCK_RELATIVE = (
    "experiments/tpd_clean_v7_dch_diagnostic_source_lock.json"
)
PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_RELATIVE = (
    "experiments/"
    "tpd_clean_v7_dch_diagnostic_source_lock_superseded_before_acceptance_v2.json"
)
DIAGNOSTIC_SUPERSESSION_RECORD_RELATIVE = (
    "experiments/"
    "tpd_clean_v7_dch_diagnostic_source_lock_supersession_acceptance_v2.json"
)
DIAGNOSTIC_SUPERSESSION_RECORD_SCHEMA = (
    "sctransnet_tpd_clean_v7_dch_diagnostic_source_lock_supersession_v1"
)
SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE = (
    "experiments/tpd_clean_v7_dch_acceptance_source_lock.json"
)
SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE = (
    "experiments/tpd_clean_v7_dch_acceptance_source_lock_v2.json"
)
SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE = (
    "experiments/tpd_clean_v7_dch_diagnostic_source_lock_v2.json"
)
SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE = (
    "experiments/tpd_clean_v7_dch_acceptance_source_lock_v3.json"
)
SUPERSEDED_DIAGNOSTIC_LOCK_SHA256 = (
    "5f99bb511cb140cd502dcf41329f698b338d41e7404e6f897cf84ce3ab241a92"
)
PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_SHA256 = (
    "edd670631f8e058e82d7bdddd68a21b1de46a1d3b02a35ae6fa7e2de22734695"
)
DIAGNOSTIC_SUPERSESSION_RECORD_SHA256 = (
    "86512d73fc6aa0bb8ebbf38b272392a55f56c0667af0d534f4bb0f4927a4219b"
)
SUPERSEDED_ACCEPTANCE_LOCK_SHA256 = (
    "4fb4668d1eb97e3c6a28a60efbfad4ea9ac3423d98f16f7d411a09cebb5b68d7"
)
SUPERSEDED_ACCEPTANCE_LOCK_V2_SHA256 = (
    "ee7be009081b1776b6e5068c9c39b7f4429c987a44cea0a25f7c95f27fc8f130"
)
SUPERSEDED_DIAGNOSTIC_LOCK_V2_SHA256 = (
    "902987310b86404b5cf72bb8e23359020508483ecc810cf6a881c67947e4b1d9"
)
SUPERSEDED_ACCEPTANCE_LOCK_V3_SHA256 = (
    "f319f4b4b1cd05ad97504b8fc317e8c24abb3736d5292ec64e85647731df5a45"
)
DEFAULT_LOCK_RELATIVES = {
    "diagnostic": (
        "experiments/tpd_clean_v7_dch_diagnostic_source_lock_v3.json"
    ),
    # This exact filename and schema are consumed by
    # experiments/train_tpd_clean_v7_dch_exact.py.
    "training": "experiments/tpd_clean_v7_dch_exact_source_lock.json",
    "acceptance": (
        "experiments/tpd_clean_v7_dch_acceptance_source_lock_v4.json"
    ),
}
VARIANTS = (
    "tpd_clean_v7_dch_full",
    "tpd_clean_v7_dch_capacity",
)
VALIDATION_FIELDS = (
    "val_loss",
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
GO_DECISION = {
    "status": "GO_DCH_TRAJECTORY_TEST",
    "context_direct_support": False,
    "dch_causal_mechanism_established": False,
    "paper_core_established": False,
    "stability_claim_supported": False,
}

DIAGNOSTIC_ROOT_RELATIVES = (
    "analysis/diagnose_tpd_clean_v6_fragmentation.py",
    "analysis/summarize_tpd_clean_v6_failure_atlas.py",
    "tests/test_diagnose_tpd_clean_v6_fragmentation.py",
    "tests/test_summarize_tpd_clean_v6_failure_atlas.py",
    "experiments/freeze_tpd_clean_v7_dch_source_locks.py",
    "tests/test_tpd_clean_v7_dch_source_locks.py",
)

TRAINING_MINIMUM_DECLARATIONS = (
    "model/tpd_clean_v7_dch.py",
    "experiments/train_tpd_clean_v7_dch.py",
    "experiments/train_tpd_clean_v7_dch_exact.py",
    # DCH exact imports this entry eagerly and reuses its data/exact helpers.
    "experiments/train_tpd_clean_v6_exact.py",
    "experiments/train_tpd_clean_v6.py",
    "model/tpd_clean_v6.py",
    "experiments/tpd_exact_runner.py",
    "experiments/tpd_exact_resume.py",
    "experiments/tpd_exact_epoch_journal.py",
    "experiments/tpd_exact_training_runtime.py",
    "experiments/tpd_extension_warm_start.py",
    "experiments/train_tpd_pilot.py",
    "experiments/fingerprint_tpd_training_data.py",
    "model/SCTransNet.py",
    "model/Config.py",
    "model/tpd.py",
    "dataset.py",
    "utils.py",
    "warmup_scheduler.py",
    "experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md",
    # Formal process control reached by the actual two-lane launch.
    "experiments/run_tpd_clean_v7_dch_formal800_2x5090_worker.sh",
    "experiments/run_tpd_clean_v7_dch_formal800_2x5090_lane.sh",
    "experiments/launch_tpd_clean_v7_dch_formal800_2x5090.sh",
    # Imported by worker preflight to validate persisted smoke artifacts.
    "experiments/verify_tpd_clean_v7_dch_smoke_reports.py",
)

ACCEPTANCE_ROOT_RELATIVES = (
    "experiments/evaluate_tpd_clean_v7_dch_pd_fa.py",
    "experiments/run_tpd_clean_v7_dch_formal800_sweeps.py",
    "experiments/summarize_tpd_clean_v7_dch_formal800.py",
    "experiments/validate_tpd_clean_v7_dch_formal800_completion.py",
    "experiments/finalize_tpd_clean_v7_dch.py",
    "analysis/diagnose_tpd_clean_v7_dch_mechanism.py",
    "experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md",
    "experiments/freeze_tpd_clean_v7_dch_source_locks.py",
    "tests/test_evaluate_tpd_clean_v7_dch_pd_fa.py",
    "tests/test_run_tpd_clean_v7_dch_formal800_sweeps.py",
    "tests/test_summarize_tpd_clean_v7_dch_formal800.py",
    "tests/test_validate_tpd_clean_v7_dch_formal800_completion.py",
    "tests/test_finalize_tpd_clean_v7_dch.py",
    "tests/test_tpd_clean_v7_dch_fragmentation_audit.py",
    "tests/test_tpd_clean_v7_dch_source_locks.py",
    "experiments/TPD_CLEAN_V7_DCH_ACCEPTANCE_AMENDMENT_V1.md",
    "experiments/TPD_CLEAN_V7_DCH_ACCEPTANCE_AMENDMENT_V2.md",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_kind(kind: str) -> str:
    if kind not in LOCK_KINDS:
        raise ValueError(f"unknown source-lock kind: {kind!r}")
    return kind


def _canonical_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    root = Path(repo_root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path lies outside repository: {path}") from exc
    text = relative.as_posix()
    _require(text not in {"", "."}, "repository root cannot be a lock input")
    return text


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"lock input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_candidates(
    module: str,
    aliases: Iterable[str],
    repo_root: Path,
) -> set[str]:
    """Resolve one absolute import to possible repository-local modules."""

    if not module:
        return set()
    pieces = module.split(".")
    base_file = repo_root.joinpath(*pieces).with_suffix(".py")
    base_package = repo_root.joinpath(*pieces, "__init__.py")
    candidates: set[str] = set()
    if base_file.is_file():
        candidates.add(_canonical_relative(base_file, repo_root))
    if base_package.is_file():
        candidates.add(_canonical_relative(base_package, repo_root))
    base_directory = repo_root.joinpath(*pieces)
    if base_directory.is_dir():
        for alias in aliases:
            if alias == "*":
                continue
            alias_file = base_directory / f"{alias}.py"
            alias_package = base_directory / alias / "__init__.py"
            if alias_file.is_file():
                candidates.add(_canonical_relative(alias_file, repo_root))
            if alias_package.is_file():
                candidates.add(_canonical_relative(alias_package, repo_root))
    return candidates


def _relative_module_name(
    current_relative: str,
    level: int,
    module: str | None,
) -> str:
    package_parts = list(Path(current_relative).parent.parts)
    if level > len(package_parts) + 1:
        return ""
    keep = len(package_parts) - max(level - 1, 0)
    prefix = package_parts[:keep]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


def direct_eager_local_imports(
    relative: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> set[str]:
    """Return statically imported local Python modules for one source."""

    path = Path(repo_root) / relative
    if path.suffix != ".py":
        return set()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"cannot inspect eager imports: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    import_nodes: list[ast.Import | ast.ImportFrom] = []

    class EagerImportVisitor(ast.NodeVisitor):
        """Visit import-time control flow, but skip deferred call bodies."""

        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            import_nodes.append(node)

        def visit_ImportFrom(  # noqa: N802
            self,
            node: ast.ImportFrom,
        ) -> None:
            import_nodes.append(node)

        def visit_FunctionDef(  # noqa: N802
            self,
            node: ast.FunctionDef,
        ) -> None:
            return None

        def visit_AsyncFunctionDef(  # noqa: N802
            self,
            node: ast.AsyncFunctionDef,
        ) -> None:
            return None

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return None

    EagerImportVisitor().visit(tree)
    discovered: set[str] = set()
    for node in import_nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                discovered.update(
                    _module_candidates(alias.name, (), Path(repo_root))
                )
        elif isinstance(node, ast.ImportFrom):
            module = (
                _relative_module_name(
                    relative,
                    node.level,
                    node.module,
                )
                if node.level
                else node.module or ""
            )
            discovered.update(
                _module_candidates(
                    module,
                    (alias.name for alias in node.names),
                    Path(repo_root),
                )
            )
    discovered.discard(relative)
    return discovered


def eager_local_import_closure(
    roots: Sequence[str],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, ...]:
    """Recursively expand present repository-local eager Python imports."""

    root = Path(repo_root).resolve()
    pending = list(dict.fromkeys(roots))
    discovered: set[str] = set()
    while pending:
        relative = pending.pop(0)
        if relative in discovered:
            continue
        discovered.add(relative)
        path = root / relative
        if path.is_file() and not path.is_symlink() and path.suffix == ".py":
            for dependency in sorted(
                direct_eager_local_imports(relative, repo_root=root)
            ):
                if dependency not in discovered:
                    pending.append(dependency)
    return tuple(sorted(discovered))


def _dch_exact_entry() -> Any:
    return importlib.import_module(
        "experiments.train_tpd_clean_v7_dch_exact"
    )


def formal_contract() -> Dict[str, Any]:
    exact_entry = _dch_exact_entry()
    contract = exact_entry.formal_contract()
    _require(
        contract
        == {
            "epochs": 800,
            "eval_every": 1,
            "workers": 0,
            "amp": False,
            "eps": 1e-6,
            "cublas_workspace_config": ":4096:8",
            "initialization_modes": ["fresh", "exact_resume"],
        },
        "DCH exact formal contract differs from the frozen protocol",
    )
    return dict(contract)


def training_source_relatives(
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, ...]:
    """Use the exact entry's runtime declaration as the sole authority."""

    exact_entry = _dch_exact_entry()
    root = Path(repo_root).resolve()
    _require(
        Path(exact_entry.REPO_ROOT).resolve() == root,
        "DCH exact entry repository root differs",
    )
    relatives = []
    for path in exact_entry.RUNTIME_SOURCE_PATHS:
        relative = _canonical_relative(Path(path), root)
        if relative in relatives:
            raise ValueError(f"duplicate DCH exact runtime source: {relative}")
        relatives.append(relative)
    _require(relatives, "DCH exact runtime source set is empty")
    return tuple(relatives)


def source_relatives(
    kind: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, ...]:
    kind = _validate_kind(kind)
    if kind == "training":
        return training_source_relatives(repo_root=repo_root)
    roots = (
        DIAGNOSTIC_ROOT_RELATIVES
        if kind == "diagnostic"
        else ACCEPTANCE_ROOT_RELATIVES
    )
    # Test files are bound as review evidence, but their imports are not part
    # of the executed diagnostic/acceptance runtime.  Expanding test imports
    # would incorrectly pull the exact-training entry into the other scopes.
    runtime_roots = tuple(
        relative for relative in roots if not relative.startswith("tests/")
    )
    closure = eager_local_import_closure(
        runtime_roots,
        repo_root=repo_root,
    )
    return tuple(sorted(set(closure).union(roots)))


def diagnostic_frozen_input_relatives() -> tuple[str, ...]:
    root = "analysis/results/tpd_clean_v6_frozen_failure_atlas_v1"
    paths = [
        f"{root}/V6_FAILURE_ATLAS.json",
        f"{root}/V6_FAILURE_ATLAS.md",
        f"{root}/gpu2_full/matrix_summary.json",
        f"{root}/gpu3_capacity/matrix_summary.json",
    ]
    lanes = (
        ("gpu2_full", "tpd_clean_v6_full"),
        ("gpu3_capacity", "tpd_clean_v6_phase_capacity"),
    )
    for lane, variant in lanes:
        for seed in (42, 3407):
            for role in ("pd_primary", "miou_primary"):
                paths.append(
                    f"{root}/{lane}/{variant}/seed_{seed}/{role}.json"
                )
    return tuple(paths)


def acceptance_frozen_input_relatives() -> tuple[str, ...]:
    root = (
        "experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1/"
        "NUDT-SIRST/comparison/"
        "superseded_acceptance_v3_markdown_order_v1"
    )
    return (
        f"{root}/tpd_clean_v7_dch_formal800_comparison.json",
        f"{root}/tpd_clean_v7_dch_formal800_comparison.md",
        f"{root}/ARCHIVE_EVIDENCE.json",
    )


def frozen_input_relatives(kind: str) -> tuple[str, ...]:
    kind = _validate_kind(kind)
    if kind == "diagnostic":
        return diagnostic_frozen_input_relatives()
    if kind == "acceptance":
        return acceptance_frozen_input_relatives()
    return ()


def _missing_regular_files(
    relatives: Sequence[str],
    repo_root: Path,
) -> list[str]:
    root = Path(repo_root)
    return [
        relative
        for relative in relatives
        if not (root / relative).is_file()
        or (root / relative).is_symlink()
    ]


def _historical_json_evidence(
    *,
    relative_path: str,
    expected_schema: str,
    expected_sha256: str,
    repo_root: Path,
) -> Dict[str, Any]:
    """Read and identify one immutable historical JSON artifact."""

    root = Path(repo_root).resolve()
    path = root / relative_path
    regular = path.is_file() and not path.is_symlink()
    record: Dict[str, Any] = {
        "relative_path": relative_path,
        "expected_schema": expected_schema,
        "observed_schema": None,
        "expected_sha256": expected_sha256,
        "sha256": None,
        "present": regular,
        "accepted_as_current": False,
        "evidence_error": None,
    }
    if not regular:
        if path.exists() or path.is_symlink():
            record["evidence_error"] = "historical artifact is not a regular file"
        else:
            record["evidence_error"] = "historical artifact is absent"
        return record
    try:
        payload = _load_json_mapping(path)
        observed_schema = payload.get("schema")
        observed_sha256 = file_sha256(path)
        record["observed_schema"] = observed_schema
        record["sha256"] = observed_sha256
        if observed_schema != expected_schema:
            record["evidence_error"] = "historical artifact schema differs"
        elif observed_sha256 != expected_sha256:
            record["evidence_error"] = "historical artifact SHA-256 differs"
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        record["evidence_error"] = str(exc)
    return record


def _superseded_lock_evidence(
    *,
    relative_path: str,
    expected_schema: str,
    expected_sha256: str,
    superseded_by_schema: str,
    superseded_by_relative_path: str,
    repo_root: Path,
) -> Dict[str, Any]:
    """Describe one immutable historical lock without accepting it as current."""

    record = _historical_json_evidence(
        relative_path=relative_path,
        expected_schema=expected_schema,
        expected_sha256=expected_sha256,
        repo_root=repo_root,
    )
    record.update(
        {
            "superseded": True,
            "superseded_by_schema": superseded_by_schema,
            "superseded_by_relative_path": superseded_by_relative_path,
        }
    )
    return record


def _diagnostic_predecessor_chain_evidence(
    *,
    repo_root: Path,
) -> list[Dict[str, Any]]:
    archived = _historical_json_evidence(
        relative_path=PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_RELATIVE,
        expected_schema=DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1,
        expected_sha256=PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_SHA256,
        repo_root=repo_root,
    )
    archived["evidence_role"] = "archived_diagnostic_lock"
    transition = _historical_json_evidence(
        relative_path=DIAGNOSTIC_SUPERSESSION_RECORD_RELATIVE,
        expected_schema=DIAGNOSTIC_SUPERSESSION_RECORD_SCHEMA,
        expected_sha256=DIAGNOSTIC_SUPERSESSION_RECORD_SHA256,
        repo_root=repo_root,
    )
    transition["evidence_role"] = "diagnostic_supersession_record"
    transition["relation_verified"] = False
    if transition["evidence_error"] is None:
        payload = _load_json_mapping(
            Path(repo_root) / DIAGNOSTIC_SUPERSESSION_RECORD_RELATIVE
        )
        expected_current = {
            "relative_path": SUPERSEDED_DIAGNOSTIC_LOCK_RELATIVE,
            "schema": DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1,
            "sha256": SUPERSEDED_DIAGNOSTIC_LOCK_SHA256,
            "source_count": 18,
        }
        expected_superseded = {
            "relative_path": PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_RELATIVE,
            "schema": DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1,
            "sha256": PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_SHA256,
            "source_count": 18,
            "accepted_as_current": False,
        }
        relation_verified = (
            payload.get("current") == expected_current
            and payload.get("superseded") == expected_superseded
            and payload.get("default_path_changed") is False
            and payload.get("schema_changed") is False
            and payload.get("source_set_changed") is False
            and payload.get("training_lock_changed") is False
        )
        transition["relation_verified"] = relation_verified
        if not relation_verified:
            transition["evidence_error"] = (
                "diagnostic supersession relation differs"
            )
    return [archived, transition]


def superseded_diagnostic_lock_evidence(
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Describe the immutable diagnostic v1 lock as historical evidence."""

    record = _superseded_lock_evidence(
        relative_path=SUPERSEDED_DIAGNOSTIC_LOCK_RELATIVE,
        expected_schema=DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1,
        expected_sha256=SUPERSEDED_DIAGNOSTIC_LOCK_SHA256,
        superseded_by_schema=DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V2,
        superseded_by_relative_path=SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE,
        repo_root=repo_root,
    )
    record["predecessor_chain_evidence"] = (
        _diagnostic_predecessor_chain_evidence(repo_root=repo_root)
    )
    return record


def superseded_diagnostic_v2_lock_evidence(
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Describe the immutable diagnostic v2 lock as historical evidence."""

    return _superseded_lock_evidence(
        relative_path=SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE,
        expected_schema=DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V2,
        expected_sha256=SUPERSEDED_DIAGNOSTIC_LOCK_V2_SHA256,
        superseded_by_schema=DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V3,
        superseded_by_relative_path=DEFAULT_LOCK_RELATIVES["diagnostic"],
        repo_root=repo_root,
    )


def superseded_acceptance_lock_evidence(
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Describe the immutable acceptance v1 lock as historical evidence."""

    return _superseded_lock_evidence(
        relative_path=SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE,
        expected_schema=ACCEPTANCE_SOURCE_LOCK_SCHEMA_V1,
        expected_sha256=SUPERSEDED_ACCEPTANCE_LOCK_SHA256,
        superseded_by_schema=ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2,
        superseded_by_relative_path=SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
        repo_root=repo_root,
    )


def superseded_acceptance_v2_lock_evidence(
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Describe the immutable acceptance v2 lock as historical evidence."""

    return _superseded_lock_evidence(
        relative_path=SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
        expected_schema=ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2,
        expected_sha256=SUPERSEDED_ACCEPTANCE_LOCK_V2_SHA256,
        superseded_by_schema=ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3,
        superseded_by_relative_path=SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE,
        repo_root=repo_root,
    )


def superseded_acceptance_v3_lock_evidence(
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Describe the immutable acceptance v3 lock as historical evidence."""

    return _superseded_lock_evidence(
        relative_path=SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE,
        expected_schema=ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3,
        expected_sha256=SUPERSEDED_ACCEPTANCE_LOCK_V3_SHA256,
        superseded_by_schema=ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
        superseded_by_relative_path=DEFAULT_LOCK_RELATIVES["acceptance"],
        repo_root=repo_root,
    )


def superseded_lock_evidence(
    kind: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[Dict[str, Any]]:
    """Return ordered historical-lock evidence for one current scope."""

    kind = _validate_kind(kind)
    if kind == "diagnostic":
        return [
            superseded_diagnostic_lock_evidence(repo_root=repo_root),
            superseded_diagnostic_v2_lock_evidence(repo_root=repo_root),
        ]
    if kind == "acceptance":
        return [
            superseded_acceptance_lock_evidence(repo_root=repo_root),
            superseded_acceptance_v2_lock_evidence(repo_root=repo_root),
            superseded_acceptance_v3_lock_evidence(repo_root=repo_root),
        ]
    return []


def _require_complete_superseded_evidence(
    kind: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected_records = (
        (
            (
                SUPERSEDED_DIAGNOSTIC_LOCK_RELATIVE,
                DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1,
                SUPERSEDED_DIAGNOSTIC_LOCK_SHA256,
                DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V2,
                SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE,
            ),
            (
                SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE,
                DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V2,
                SUPERSEDED_DIAGNOSTIC_LOCK_V2_SHA256,
                DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V3,
                DEFAULT_LOCK_RELATIVES["diagnostic"],
            ),
        )
        if kind == "diagnostic"
        else (
            (
                SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE,
                ACCEPTANCE_SOURCE_LOCK_SCHEMA_V1,
                SUPERSEDED_ACCEPTANCE_LOCK_SHA256,
                ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2,
                SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
            ),
            (
                SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
                ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2,
                SUPERSEDED_ACCEPTANCE_LOCK_V2_SHA256,
                ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3,
                SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE,
            ),
            (
                SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE,
                ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3,
                SUPERSEDED_ACCEPTANCE_LOCK_V3_SHA256,
                ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
                DEFAULT_LOCK_RELATIVES["acceptance"],
            ),
        )
    )
    _require(
        len(records) == len(expected_records),
        f"{kind} superseded-lock evidence count differs",
    )
    for record, expected in zip(records, expected_records, strict=True):
        (
            relative_path,
            schema,
            digest,
            successor_schema,
            successor_path,
        ) = expected
        _require(
            record.get("relative_path") == relative_path
            and record.get("expected_schema") == schema
            and record.get("expected_sha256") == digest
            and record.get("superseded_by_schema") == successor_schema
            and record.get("superseded_by_relative_path") == successor_path
            and record.get("present") is True
            and record.get("superseded") is True
            and record.get("accepted_as_current") is False
            and record.get("observed_schema") == record.get("expected_schema")
            and record.get("sha256") == record.get("expected_sha256")
            and record.get("evidence_error") is None,
            f"{kind} superseded-lock evidence is incomplete",
        )
    if kind == "diagnostic":
        chain = records[0].get("predecessor_chain_evidence")
        _require(
            isinstance(chain, list) and len(chain) == 2,
            "diagnostic predecessor-chain evidence count differs",
        )
        expected_chain = (
            (
                PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_RELATIVE,
                DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1,
                PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_SHA256,
                "archived_diagnostic_lock",
            ),
            (
                DIAGNOSTIC_SUPERSESSION_RECORD_RELATIVE,
                DIAGNOSTIC_SUPERSESSION_RECORD_SCHEMA,
                DIAGNOSTIC_SUPERSESSION_RECORD_SHA256,
                "diagnostic_supersession_record",
            ),
        )
        for item, expected in zip(chain, expected_chain, strict=True):
            relative_path, schema, digest, role = expected
            _require(
                isinstance(item, Mapping)
                and item.get("relative_path") == relative_path
                and item.get("expected_schema") == schema
                and item.get("expected_sha256") == digest
                and item.get("evidence_role") == role
                and item.get("present") is True
                and item.get("accepted_as_current") is False
                and item.get("observed_schema") == schema
                and item.get("sha256") == digest
                and item.get("evidence_error") is None,
                "diagnostic predecessor-chain evidence is incomplete",
            )
        _require(
            chain[1].get("relation_verified") is True,
            "diagnostic supersession relation is not verified",
        )


def source_lock_readiness(
    kind: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Report missing inputs without writing or accepting a partial lock."""

    kind = _validate_kind(kind)
    root = Path(repo_root).resolve()
    try:
        sources = source_relatives(kind, repo_root=root)
        declaration_error = None
    except (ImportError, FileNotFoundError, ValueError) as exc:
        sources = ()
        declaration_error = str(exc)
    frozen = frozen_input_relatives(kind)
    missing_sources = _missing_regular_files(sources, root)
    missing_frozen = _missing_regular_files(frozen, root)
    missing_declarations: list[str] = []
    if kind == "training" and sources:
        missing_declarations = sorted(
            set(TRAINING_MINIMUM_DECLARATIONS) - set(sources)
        )
    historical_evidence = superseded_lock_evidence(kind, repo_root=root)
    evidence_ready = True
    if kind in {"diagnostic", "acceptance"}:
        try:
            _require_complete_superseded_evidence(
                kind,
                historical_evidence,
            )
        except ValueError:
            evidence_ready = False
    ready = (
        declaration_error is None
        and not missing_sources
        and not missing_frozen
        and not missing_declarations
        and evidence_ready
    )
    return {
        "kind": kind,
        "ready": ready,
        "source_count": len(sources),
        "frozen_input_count": len(frozen),
        "missing_source_paths": missing_sources,
        "missing_frozen_input_paths": missing_frozen,
        "missing_exact_runtime_declarations": missing_declarations,
        "declaration_error": declaration_error,
        "expected_schema": SCHEMAS[kind],
        "default_lock_relative": DEFAULT_LOCK_RELATIVES[kind],
        "superseded_lock_evidence": historical_evidence,
        "superseded_lock_evidence_complete": evidence_ready,
        "writes_performed": 0,
    }


def _hash_relatives(
    relatives: Sequence[str],
    repo_root: Path,
) -> Dict[str, str]:
    records: Dict[str, str] = {}
    for relative in relatives:
        path = Path(repo_root) / relative
        canonical = _canonical_relative(path, repo_root)
        _require(
            canonical == relative,
            f"non-canonical source-lock relative path: {relative!r}",
        )
        if relative in records:
            raise ValueError(f"duplicate source-lock input: {relative}")
        records[relative] = file_sha256(path)
    _require(records, "source-lock input set is empty")
    return records


def _official_training_data_contract(
    dataset: str,
    dataset_dir: Path,
) -> Dict[str, Any]:
    if (
        not dataset
        or Path(dataset).name != dataset
        or dataset in {".", ".."}
    ):
        raise ValueError("dataset must be one directory name")
    shared_exact = importlib.import_module(
        "experiments.train_tpd_clean_v6_exact"
    )
    dataset_root = Path(dataset_dir).resolve() / dataset
    index_bytes, identifiers = shared_exact.read_official_training_index(
        dataset_root,
        dataset,
    )
    digest = shared_exact.official_training_data_sha256(
        dataset_root,
        dataset,
        identifiers,
        index_bytes,
    )
    return {
        "dataset": dataset,
        "official_training_index": f"img_idx/train_{dataset}.txt",
        "official_training_sample_count": len(identifiers),
        "training_data_sha256": digest,
    }


def _policy(kind: str) -> Dict[str, Any]:
    common = {
        "lock_scopes_are_separate": True,
        "existing_lock_overwrite_forbidden": True,
        "source_symlinks_forbidden": True,
        "official_test_accessed": False,
        "formula_change_after_lock_forbidden": True,
    }
    if kind == "diagnostic":
        return {
            **common,
            "training_performed": False,
            "frozen_counterfactual_is_training_trajectory": False,
            "formal_gate_replacement": False,
        }
    if kind == "training":
        return {
            **common,
            "fresh_or_exact_resume_only": True,
            "v6_warm_start_forbidden": True,
            "physical_gpus": [2, 3],
            "gpu0_gpu1_used": False,
            "native_17_validation_fields_required": True,
            "diagnostic_sources_excluded": True,
        }
    return {
        **common,
        "training_results_modified": False,
        "checkpoint_reselection_permitted": False,
        "gates_A_to_E_unchanged": True,
        "mechanism_audit_replaces_performance_gates": False,
        "comparison_manifest_binds_4_runs_12_checkpoints_8_sweeps": True,
    }


def build_source_lock_payload(
    kind: str,
    *,
    repo_root: Path = REPO_ROOT,
    dataset: str = "NUDT-SIRST",
    dataset_dir: Path | None = None,
    source_relatives_override: Sequence[str] | None = None,
    frozen_input_relatives_override: Sequence[str] | None = None,
    superseded_lock_evidence_override: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
    training_data_contract_override: Mapping[str, Any] | None = None,
    formal_contract_override: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build one deterministic complete payload without writing it."""

    kind = _validate_kind(kind)
    root = Path(repo_root).resolve()
    sources = tuple(
        source_relatives_override
        if source_relatives_override is not None
        else source_relatives(kind, repo_root=root)
    )
    frozen = tuple(
        frozen_input_relatives_override
        if frozen_input_relatives_override is not None
        else frozen_input_relatives(kind)
    )
    if kind == "training" and source_relatives_override is None:
        missing_declarations = sorted(
            set(TRAINING_MINIMUM_DECLARATIONS) - set(sources)
        )
        _require(
            not missing_declarations,
            "DCH exact RUNTIME_SOURCE_PATHS omits required execution paths: "
            f"{missing_declarations}",
        )
    source_hashes = _hash_relatives(sources, root)
    frozen_hashes = (
        _hash_relatives(frozen, root)
        if frozen
        else {}
    )
    contract = dict(
        formal_contract_override
        if formal_contract_override is not None
        else formal_contract()
    )
    payload: Dict[str, Any] = {
        "schema": SCHEMAS[kind],
        "lock_kind": kind,
        "candidate_family": "tpd_clean_v7_dch",
        "variants": list(VARIANTS),
        "formal_contract": contract,
        "go_decision": dict(GO_DECISION),
        "validation_fields": list(VALIDATION_FIELDS),
        "source_count": len(source_hashes),
        "source_sha256": source_hashes,
        "frozen_input_count": len(frozen_hashes),
        "frozen_input_sha256": frozen_hashes,
        "policy": _policy(kind),
    }
    if kind in {"diagnostic", "acceptance"}:
        historical_evidence = [
            dict(record)
            for record in (
                superseded_lock_evidence_override
                if superseded_lock_evidence_override is not None
                else superseded_lock_evidence(kind, repo_root=root)
            )
        ]
        _require_complete_superseded_evidence(kind, historical_evidence)
        payload["superseded_lock_evidence"] = historical_evidence
    if kind == "training":
        data_contract = dict(
            training_data_contract_override
            if training_data_contract_override is not None
            else _official_training_data_contract(
                dataset,
                dataset_dir
                if dataset_dir is not None
                else root / "datasets",
            )
        )
        required_data_fields = {
            "dataset",
            "official_training_index",
            "official_training_sample_count",
            "training_data_sha256",
        }
        _require(
            set(data_contract) == required_data_fields,
            "training data contract fields differ",
        )
        payload.update(data_contract)
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_new_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Exclusively create one canonical lock; never replace any path."""

    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing lock: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NotADirectoryError(
            f"lock parent must be an existing regular directory: {path.parent}"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    return path


def freeze_source_lock(
    kind: str,
    *,
    output: Path | None = None,
    repo_root: Path = REPO_ROOT,
    dataset: str = "NUDT-SIRST",
    dataset_dir: Path | None = None,
) -> tuple[Path, Dict[str, Any]]:
    """Build and exclusively write one complete source lock."""

    kind = _validate_kind(kind)
    root = Path(repo_root).resolve()
    output_path = (
        Path(output).absolute()
        if output is not None
        else root / DEFAULT_LOCK_RELATIVES[kind]
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"refusing to replace existing lock: {output_path}"
        )
    payload = build_source_lock_payload(
        kind,
        repo_root=root,
        dataset=dataset,
        dataset_dir=dataset_dir,
    )
    return write_new_json(output_path, payload), payload


def _load_json_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"source lock is not a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source lock must contain a JSON object")
    return payload


def validate_source_lock(
    kind: str,
    lock_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    dataset: str = "NUDT-SIRST",
    dataset_dir: Path | None = None,
    source_relatives_override: Sequence[str] | None = None,
    frozen_input_relatives_override: Sequence[str] | None = None,
    superseded_lock_evidence_override: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
    training_data_contract_override: Mapping[str, Any] | None = None,
    formal_contract_override: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], str]:
    """Validate schema, exact path sets, and every current digest."""

    kind = _validate_kind(kind)
    root = Path(repo_root).resolve()
    path = Path(lock_path).resolve()
    if kind == "diagnostic":
        superseded_paths = {
            (root / relative).resolve()
            for relative in (
                SUPERSEDED_DIAGNOSTIC_LOCK_RELATIVE,
                SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE,
                PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_RELATIVE,
                DIAGNOSTIC_SUPERSESSION_RECORD_RELATIVE,
            )
        }
        _require(
            path not in superseded_paths,
            "historical diagnostic v1/v2 evidence is superseded and cannot "
            "be validated as the current v3 diagnostic lock",
        )
    if kind == "acceptance":
        superseded_paths = {
            (root / relative).resolve()
            for relative in (
                SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE,
                SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
                SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE,
            )
        }
        _require(
            path not in superseded_paths,
            "acceptance source lock v1/v2/v3 is superseded and cannot be "
            "validated as the current v4 acceptance lock",
        )
    payload = _load_json_mapping(path)
    _require(
        payload.get("schema") == SCHEMAS[kind],
        f"{kind} source lock schema differs: expected {SCHEMAS[kind]!r}, "
        f"observed {payload.get('schema')!r}",
    )
    expected = build_source_lock_payload(
        kind,
        repo_root=root,
        dataset=dataset,
        dataset_dir=dataset_dir,
        source_relatives_override=source_relatives_override,
        frozen_input_relatives_override=frozen_input_relatives_override,
        superseded_lock_evidence_override=(
            superseded_lock_evidence_override
        ),
        training_data_contract_override=training_data_contract_override,
        formal_contract_override=formal_contract_override,
    )
    _require(
        payload == expected,
        f"{kind} source lock differs from the current complete contract",
    )
    return payload, file_sha256(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check, freeze, or validate one V7-DCH source lock"
    )
    parser.add_argument("action", choices=("check", "freeze", "validate"))
    parser.add_argument("--kind", choices=LOCK_KINDS, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset", default="NUDT-SIRST")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args(argv)
    if args.action == "validate" and args.lock is None:
        parser.error("validate requires --lock")
    if args.action != "validate" and args.lock is not None:
        parser.error("--lock is only valid with validate")
    if args.action != "freeze" and args.output is not None:
        parser.error("--output is only valid with freeze")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.action == "check":
        readiness = source_lock_readiness(
            args.kind,
            repo_root=repo_root,
        )
        print(
            json.dumps(
                readiness,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if not readiness["ready"]:
            raise SystemExit(2)
        return
    if args.action == "freeze":
        path, payload = freeze_source_lock(
            args.kind,
            output=args.output,
            repo_root=repo_root,
            dataset=args.dataset,
            dataset_dir=args.dataset_dir,
        )
        print(
            f"WROTE kind={args.kind} path={path} "
            f"sources={payload['source_count']}",
            flush=True,
        )
        return
    payload, digest = validate_source_lock(
        args.kind,
        args.lock,
        repo_root=repo_root,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
    )
    print(
        f"VALID kind={args.kind} path={Path(args.lock).resolve()} "
        f"sha256={digest} sources={payload['source_count']}",
        flush=True,
    )


__all__ = [
    "ACCEPTANCE_SOURCE_LOCK_SCHEMA_V1",
    "ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2",
    "ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3",
    "ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4",
    "ACCEPTANCE_ROOT_RELATIVES",
    "acceptance_frozen_input_relatives",
    "DEFAULT_LOCK_RELATIVES",
    "DIAGNOSTIC_SUPERSESSION_RECORD_RELATIVE",
    "DIAGNOSTIC_SUPERSESSION_RECORD_SCHEMA",
    "DIAGNOSTIC_SUPERSESSION_RECORD_SHA256",
    "DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1",
    "DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V2",
    "DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V3",
    "DIAGNOSTIC_ROOT_RELATIVES",
    "GO_DECISION",
    "LOCK_KINDS",
    "PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_RELATIVE",
    "PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_SHA256",
    "REPO_ROOT",
    "SCHEMAS",
    "SUPERSEDED_ACCEPTANCE_LOCK_SHA256",
    "SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE",
    "SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE",
    "SUPERSEDED_ACCEPTANCE_LOCK_V2_SHA256",
    "SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE",
    "SUPERSEDED_ACCEPTANCE_LOCK_V3_SHA256",
    "SUPERSEDED_DIAGNOSTIC_LOCK_RELATIVE",
    "SUPERSEDED_DIAGNOSTIC_LOCK_SHA256",
    "SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE",
    "SUPERSEDED_DIAGNOSTIC_LOCK_V2_SHA256",
    "TRAINING_MINIMUM_DECLARATIONS",
    "VALIDATION_FIELDS",
    "VARIANTS",
    "build_source_lock_payload",
    "diagnostic_frozen_input_relatives",
    "direct_eager_local_imports",
    "eager_local_import_closure",
    "file_sha256",
    "formal_contract",
    "freeze_source_lock",
    "frozen_input_relatives",
    "main",
    "parse_args",
    "source_lock_readiness",
    "source_relatives",
    "superseded_acceptance_lock_evidence",
    "superseded_acceptance_v2_lock_evidence",
    "superseded_acceptance_v3_lock_evidence",
    "superseded_diagnostic_lock_evidence",
    "superseded_diagnostic_v2_lock_evidence",
    "superseded_lock_evidence",
    "training_source_relatives",
    "validate_source_lock",
    "write_new_json",
]


if __name__ == "__main__":
    main()
