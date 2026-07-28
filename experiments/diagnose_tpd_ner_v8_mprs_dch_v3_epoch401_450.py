#!/usr/bin/env python3
"""Watch and diagnose the canonical V3 epoch-401--450 window.

This module is intentionally outside the locked V1/V2/V3 training and
acceptance closures.  It only reads the canonical V3/V2 run artifacts and,
once V3 reaches epoch 450, publishes the preregistered five-item diagnostic
inside the canonical V3 run directory.  The diagnostic is not the epoch-800
model decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
TARGET_COUNT = 189
WINDOW_START = 401
WINDOW_END = 450
WINDOW_EPOCHS = 50

V3_VARIANT = "tpd_ner_v8_mprs_dch_v3_full_relay_on"
V2_VARIANT = "tpd_ner_v8_mprs_dch_v2_full_relay_on"
V3_RUN_TAG = "formal800_exact_v3_seed42"
V2_RUN_TAG = "formal800_exact_v2_seed42"
V3_ENTRY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v3_exact_entry_v1"
V2_ENTRY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_exact_entry_v1"
RUN_IDENTITY_SCHEMA = "sctransnet_tpd_run_identity_v1"
ORDERED_FINGERPRINT_SCHEMA = "sctransnet_tpd_ordered_fingerprint_v1"

V3_RUN_DIR = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v8_mprs_dch_v3_exact_v1"
    / DATASET
    / V3_VARIANT
    / f"seed_{TRAINING_SEED}_{V3_RUN_TAG}"
)
V2_RUN_DIR = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v8_mprs_dch_v2_exact_v1"
    / DATASET
    / V2_VARIANT
    / f"seed_{TRAINING_SEED}_{V2_RUN_TAG}"
)
JSON_OUTPUT_NAME = "epoch450_diagnostic.json"
MARKDOWN_OUTPUT_NAME = "epoch450_diagnostic.md"

SERVICE = "sctransnet-tpd-ner-v8-v3-relay-on-gpu2.service"
ACTIVE_STATES = frozenset(
    {"active", "activating", "deactivating", "reloading"}
)

REPORT_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_epoch401_450_diagnostic_v1"
)
PASSED_DECISION = "EPOCH401_450_DIAGNOSTIC_PASSED"
FAILED_DECISION = "EPOCH401_450_DIAGNOSTIC_FAILED"


class DiagnosticError(RuntimeError):
    """The canonical epoch-401--450 diagnostic cannot safely proceed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_regular_bytes(path: Path, label: str) -> bytes:
    candidate = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise DiagnosticError(
            f"{label} is not a readable regular file: {candidate}: {exc}"
        ) from exc
    try:
        _require(
            stat.S_ISREG(os.fstat(descriptor).st_mode),
            f"{label} must be a regular file: {candidate}",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _reject_nonfinite_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token}")


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, label)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DiagnosticError(f"{label} is invalid JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value, raw


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
        serialized = json.dumps(value, **options) + "\n"
    else:
        options["separators"] = (",", ":")
        serialized = json.dumps(value, **options)
    return serialized.encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ordered_fingerprint(name: str, values: Any) -> dict[str, Any]:
    _require(isinstance(values, list), f"split {name} IDs must be a list")
    _require(
        all(isinstance(value, str) and value for value in values),
        f"split {name} IDs must be non-empty strings",
    )
    _require(
        len(values) == len(set(values)),
        f"split {name} IDs contain duplicates",
    )
    return {
        "schema": ORDERED_FINGERPRINT_SCHEMA,
        "name": name,
        "count": len(values),
        "sha256": _sha256_json(values),
    }


def _validate_split(
    split: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    expected_scalars = {
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "val_fraction": 0.2,
        "full_official_train_count": 663,
        "full_internal_train_count": 530,
        "full_internal_val_count": 133,
        "used_train_count": 530,
        "used_val_count": 133,
        "official_test_accessed": False,
    }
    for field, expected in expected_scalars.items():
        _require(
            split.get(field) == expected,
            f"{label} split field differs: {field}",
        )

    role_to_ids = {
        "full_train": "full_internal_train_ids",
        "full_validation": "full_internal_val_ids",
        "train": "used_train_ids",
        "validation": "used_val_ids",
    }
    records = {
        role: _ordered_fingerprint(role, split.get(field))
        for role, field in role_to_ids.items()
    }
    _require(
        records["full_train"]["count"] == 530
        and records["full_validation"]["count"] == 133
        and records["train"]["count"] == 530
        and records["validation"]["count"] == 133,
        f"{label} split ID counts differ",
    )
    _require(
        split.get("used_train_ids") == split.get("full_internal_train_ids")
        and split.get("used_val_ids") == split.get("full_internal_val_ids"),
        f"{label} split is not the complete fixed 530/133 split",
    )
    _require(
        set(split["used_train_ids"]).isdisjoint(split["used_val_ids"]),
        f"{label} train and validation IDs overlap",
    )
    _require(
        identity.get("ordered_split_fingerprints") == records,
        f"{label} protocol is not bound to split.json ID order",
    )
    _require(
        identity.get("split_sha256") == _sha256_json(records),
        f"{label} protocol split SHA differs",
    )
    return records


def _expected_run_id(variant: str, run_tag: str, version: str) -> str:
    return (
        f"tpd-ner-v8-mprs-dch-{version}-exact:{DATASET}:{variant}:"
        f"seed-{TRAINING_SEED}:split-{SPLIT_SEED}:{run_tag}"
    )


def _validate_run_binding(
    run_dir: Path,
    *,
    variant: str,
    run_tag: str,
    version: str,
    entry_schema: str,
) -> dict[str, Any]:
    directory = Path(run_dir)
    _require(
        directory.is_dir() and not directory.is_symlink(),
        f"{version.upper()} canonical run directory is unsafe: {directory}",
    )
    protocol, _ = _load_json_object(
        directory / "protocol.json",
        f"{version.upper()} protocol",
    )
    split, split_bytes = _load_json_object(
        directory / "split.json",
        f"{version.upper()} split",
    )
    _require(
        protocol.get("schema") == entry_schema,
        f"{version.upper()} protocol schema differs",
    )
    arguments = protocol.get("arguments")
    _require(
        isinstance(arguments, Mapping),
        f"{version.upper()} protocol arguments are missing",
    )
    for field, expected in {
        "dataset": DATASET,
        "variant": variant,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "run_tag": run_tag,
        "epochs": 800,
        "eval_every": 1,
    }.items():
        _require(
            arguments.get(field) == expected,
            f"{version.upper()} protocol argument differs: {field}",
        )

    formal = protocol.get("formal_contract")
    _require(
        isinstance(formal, Mapping),
        f"{version.upper()} formal contract is missing",
    )
    for field, expected in {
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": 800,
        "eval_every": 1,
        "candidate_variants": [variant],
        "multi_seed_scheduled": False,
    }.items():
        _require(
            formal.get(field) == expected,
            f"{version.upper()} formal contract differs: {field}",
        )
    _require(
        protocol.get("official_test_accessed") is False,
        f"{version.upper()} protocol accessed the official test set",
    )

    identity = protocol.get("run_identity")
    _require(
        isinstance(identity, Mapping),
        f"{version.upper()} run identity is missing",
    )
    for field, expected in {
        "schema": RUN_IDENTITY_SCHEMA,
        "dataset": DATASET,
        "variant": variant,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "run_id": _expected_run_id(variant, run_tag, version),
    }.items():
        _require(
            identity.get(field) == expected,
            f"{version.upper()} run identity differs: {field}",
        )
    _require(
        _valid_sha256(identity.get("split_sha256"))
        and _valid_sha256(identity.get("data_sha256")),
        f"{version.upper()} run identity digests are invalid",
    )
    ordered_data = identity.get("ordered_data_fingerprints")
    _require(
        isinstance(ordered_data, Mapping)
        and ordered_data
        and identity.get("data_sha256") == _sha256_json(ordered_data),
        f"{version.upper()} data fingerprint binding differs",
    )
    _validate_split(split, identity, label=version.upper())
    _require(
        Path(protocol.get("run_directory", "")).resolve()
        == directory.resolve(),
        f"{version.upper()} protocol run directory differs",
    )
    return {
        "run_dir": str(directory.resolve()),
        "variant": variant,
        "run_id": identity["run_id"],
        "split_sha256": identity["split_sha256"],
        "data_sha256": identity["data_sha256"],
        "split_bytes": split_bytes,
    }


def _load_metrics(run_dir: Path, variant: str, label: str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    raw = _read_regular_bytes(path, f"{label} metrics")
    _require(raw.endswith(b"\n"), f"{label} metrics journal is truncated")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DiagnosticError(f"{label} metrics journal is not UTF-8") from exc
    _require(
        lines and all(line.strip() for line in lines),
        f"{label} metrics journal has no complete events",
    )

    rows: list[dict[str, Any]] = []
    for expected_epoch, line in enumerate(lines, start=1):
        try:
            row = json.loads(
                line,
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DiagnosticError(
                f"{label} metrics epoch {expected_epoch} is invalid JSON: {exc}"
            ) from exc
        _require(
            isinstance(row, dict),
            f"{label} metrics epoch {expected_epoch} is not an object",
        )
        _require(
            row.get("epoch") == expected_epoch,
            f"{label} metrics epochs are not contiguous from 1",
        )
        _require(
            row.get("variant") == variant,
            f"{label} metrics epoch {expected_epoch} variant differs",
        )
        for field in (
            "fa",
            "pd",
            "miou",
            "target_count",
            "matched_target_count",
            "unmatched_predicted_object_count",
            "valid_pixel_count",
        ):
            _require(
                field in row,
                f"{label} metrics epoch {expected_epoch} lacks {field}",
            )
        _require(
            row["target_count"] == TARGET_COUNT,
            f"{label} metrics epoch {expected_epoch} target count differs",
        )
        for field in (
            "matched_target_count",
            "unmatched_predicted_object_count",
            "valid_pixel_count",
        ):
            _require(
                _is_integer(row[field]),
                f"{label} metrics epoch {expected_epoch} {field} is invalid",
            )
        _require(
            0 <= row["matched_target_count"] <= TARGET_COUNT
            and row["unmatched_predicted_object_count"] >= 0
            and row["valid_pixel_count"] > 0,
            f"{label} metrics epoch {expected_epoch} counts are invalid",
        )
        for field in ("fa", "pd", "miou"):
            _require(
                _is_number(row[field]),
                f"{label} metrics epoch {expected_epoch} {field} is invalid",
            )
        _require(
            0.0 <= float(row["fa"]) <= 1.0
            and 0.0 <= float(row["pd"]) <= 1.0
            and 0.0 <= float(row["miou"]) <= 1.0,
            f"{label} metrics epoch {expected_epoch} rates are out of range",
        )
        _require(
            math.isclose(
                float(row["pd"]),
                row["matched_target_count"] / TARGET_COUNT,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{label} metrics epoch {expected_epoch} Pd/count differs",
        )
        rows.append(row)
    return rows


def current_v3_epoch() -> int:
    """Return the strictly validated current V3 epoch, or zero before start."""

    path = V3_RUN_DIR / "metrics.jsonl"
    if not path.exists():
        _require(
            not path.is_symlink(),
            f"V3 metrics path may not be a symlink: {path}",
        )
        return 0
    return len(_load_metrics(V3_RUN_DIR, V3_VARIANT, "V3"))


def _at_most(observed: float, threshold: float) -> bool:
    return observed <= threshold or math.isclose(
        observed,
        threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def build_report(
    v3_rows: Sequence[Mapping[str, Any]],
    v2_rows: Sequence[Mapping[str, Any]],
    *,
    v3_binding: Mapping[str, Any],
    v2_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute exactly the five preregistered epoch-401--450 criteria."""

    _require(
        len(v3_rows) >= WINDOW_END and len(v2_rows) >= WINDOW_END,
        "both canonical metrics journals must reach epoch 450",
    )
    v3_window = list(v3_rows[WINDOW_START - 1 : WINDOW_END])
    v2_window = list(v2_rows[WINDOW_START - 1 : WINDOW_END])
    _require(
        len(v3_window) == WINDOW_EPOCHS
        and len(v2_window) == WINDOW_EPOCHS,
        "diagnostic windows must contain exactly 50 epochs",
    )

    false_pixels: list[int] = []
    for v3_row, v2_row in zip(v3_window, v2_window):
        epoch = v3_row["epoch"]
        _require(
            v2_row["epoch"] == epoch,
            "V3/V2 diagnostic windows are not epoch-aligned",
        )
        _require(
            v3_row["valid_pixel_count"] == v2_row["valid_pixel_count"],
            f"V3/V2 valid pixel count differs at epoch {epoch}",
        )
        reconstructed = (
            float(v3_row["fa"]) * int(v3_row["valid_pixel_count"])
        )
        nearest = round(reconstructed)
        _require(
            math.isclose(
                reconstructed,
                nearest,
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            f"V3 false pixels cannot be reconstructed at epoch {epoch}",
        )
        false_pixels.append(int(nearest))

    median_false_pixels = float(statistics.median(false_pixels))
    mean_unmatched_objects = (
        math.fsum(
            int(row["unmatched_predicted_object_count"])
            for row in v3_window
        )
        / WINDOW_EPOCHS
    )
    joint_pd_fa_count = sum(
        int(row["matched_target_count"]) >= 183
        and float(row["fa"]) <= 5e-6
        for row in v3_window
    )
    v3_mean_matched = (
        math.fsum(int(row["matched_target_count"]) for row in v3_window)
        / WINDOW_EPOCHS
    )
    v2_mean_matched = (
        math.fsum(int(row["matched_target_count"]) for row in v2_window)
        / WINDOW_EPOCHS
    )
    mean_matched_decrease = v2_mean_matched - v3_mean_matched
    v3_mean_miou = (
        math.fsum(float(row["miou"]) for row in v3_window)
        / WINDOW_EPOCHS
    )
    v2_mean_miou = (
        math.fsum(float(row["miou"]) for row in v2_window)
        / WINDOW_EPOCHS
    )
    mean_miou_decrease = v2_mean_miou - v3_mean_miou

    criteria = {
        "median_false_pixels_per_epoch": {
            "observed": median_false_pixels,
            "operator": "<=",
            "threshold": 40,
            "passed": _at_most(median_false_pixels, 40.0),
        },
        "mean_unmatched_predicted_objects_per_epoch": {
            "observed": mean_unmatched_objects,
            "operator": "<=",
            "threshold": 7.5,
            "passed": _at_most(mean_unmatched_objects, 7.5),
        },
        "epochs_with_matched_targets_ge_183_and_fa_le_5e_6": {
            "observed": joint_pd_fa_count,
            "operator": ">=",
            "threshold": 25,
            "denominator": WINDOW_EPOCHS,
            "passed": joint_pd_fa_count >= 25,
        },
        "mean_matched_targets_decrease_vs_v2": {
            "v3_mean": v3_mean_matched,
            "v2_mean": v2_mean_matched,
            "observed_decrease": mean_matched_decrease,
            "operator": "<=",
            "threshold": 1.0,
            "passed": _at_most(mean_matched_decrease, 1.0),
        },
        "mean_miou_decrease_vs_v2": {
            "v3_mean": v3_mean_miou,
            "v2_mean": v2_mean_miou,
            "observed_decrease": mean_miou_decrease,
            "operator": "<=",
            "threshold": 0.003,
            "passed": _at_most(mean_miou_decrease, 0.003),
        },
    }
    all_passed = all(
        criterion["passed"] is True for criterion in criteria.values()
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "scope": "diagnostic_only_not_epoch800_decision",
        "decision": PASSED_DECISION if all_passed else FAILED_DECISION,
        "all_five_criteria_passed": all_passed,
        "window": {
            "start_epoch": WINDOW_START,
            "end_epoch": WINDOW_END,
            "epoch_count": WINDOW_EPOCHS,
            "median_convention": "mean_of_sorted_positions_25_and_26",
            "false_pixels_reconstruction": (
                "round(fa * valid_pixel_count) after integer-closeness check"
            ),
        },
        "bindings": {
            "dataset": DATASET,
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "shared_split_sha256": v3_binding["split_sha256"],
            "shared_data_sha256": v3_binding["data_sha256"],
            "v3": {
                "variant": V3_VARIANT,
                "run_id": v3_binding["run_id"],
                "run_dir": v3_binding["run_dir"],
                "window_metrics_sha256": _sha256_json(v3_window),
            },
            "v2": {
                "variant": V2_VARIANT,
                "run_id": v2_binding["run_id"],
                "run_dir": v2_binding["run_dir"],
                "window_metrics_sha256": _sha256_json(v2_window),
            },
        },
        "criteria": criteria,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    criteria = report["criteria"]

    def result(name: str) -> str:
        return "PASS" if criteria[name]["passed"] else "FAIL"

    false_pixels = criteria["median_false_pixels_per_epoch"]
    unmatched = criteria["mean_unmatched_predicted_objects_per_epoch"]
    joint = criteria[
        "epochs_with_matched_targets_ge_183_and_fa_le_5e_6"
    ]
    matched = criteria["mean_matched_targets_decrease_vs_v2"]
    miou = criteria["mean_miou_decrease_vs_v2"]
    rows = [
        (
            "Median false pixels/epoch",
            f"{false_pixels['observed']:.12g}",
            "<= 40",
            result("median_false_pixels_per_epoch"),
        ),
        (
            "Mean unmatched predicted objects/epoch",
            f"{unmatched['observed']:.12g}",
            "<= 7.5",
            result("mean_unmatched_predicted_objects_per_epoch"),
        ),
        (
            "Epochs with matched targets >=183 and Fa <=5e-6",
            f"{joint['observed']}/50",
            ">= 25/50",
            result("epochs_with_matched_targets_ge_183_and_fa_le_5e_6"),
        ),
        (
            "Mean matched-target decrease vs V2",
            (
                f"{matched['observed_decrease']:.12g} "
                f"(V2 {matched['v2_mean']:.12g}, "
                f"V3 {matched['v3_mean']:.12g})"
            ),
            "<= 1",
            result("mean_matched_targets_decrease_vs_v2"),
        ),
        (
            "Mean mIoU decrease vs V2",
            (
                f"{miou['observed_decrease']:.12g} "
                f"(V2 {miou['v2_mean']:.12g}, "
                f"V3 {miou['v3_mean']:.12g})"
            ),
            "<= 0.003",
            result("mean_miou_decrease_vs_v2"),
        ),
    ]
    lines = [
        "# TPD-NER V8-MPRS-DCH V3 epoch 401–450 diagnostic",
        "",
        "**DIAGNOSTIC ONLY — NOT THE EPOCH-800 MODEL DECISION.**",
        "",
        f"- Decision: `{report['decision']}`",
        "- Window: closed interval epoch 401–450 (50/50 contiguous events)",
        "- Binding: canonical V3 and V2, seed 42, split seed 20260722",
        "",
        "| Criterion | Observed | Gate | Result |",
        "|---|---:|---:|:---:|",
    ]
    lines.extend(
        f"| {name} | {observed} | {gate} | **{outcome}** |"
        for name, observed, gate, outcome in rows
    )
    lines.extend(
        [
            "",
            "False pixels are reconstructed as "
            "`round(Fa * valid_pixel_count)` only after checking that the "
            "product is integer-valued within tolerance. For 50 values, the "
            "median is the mean of sorted positions 25 and 26.",
            "",
            "A failed diagnostic preserves the formal training artifacts and "
            "does not constitute the epoch-800 performance conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_publish_idempotent(path: Path, content: bytes) -> str:
    target = Path(path)
    _require(
        target.parent.is_dir() and not target.parent.is_symlink(),
        f"diagnostic output directory is unsafe: {target.parent}",
    )
    if target.exists() or target.is_symlink():
        existing = _read_regular_bytes(target, "existing diagnostic output")
        _require(
            existing == content,
            f"existing diagnostic output conflicts: {target}",
        )
        return "unchanged"

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _read_regular_bytes(
                target,
                "concurrently published diagnostic output",
            )
            _require(
                existing == content,
                f"concurrent diagnostic output conflicts: {target}",
            )
            return "unchanged"
        return "published"
    finally:
        temporary.unlink(missing_ok=True)


def diagnose_and_write() -> dict[str, Any]:
    """Validate canonical inputs, compute five gates, and publish two files."""

    v3_binding = _validate_run_binding(
        V3_RUN_DIR,
        variant=V3_VARIANT,
        run_tag=V3_RUN_TAG,
        version="v3",
        entry_schema=V3_ENTRY_SCHEMA,
    )
    v2_binding = _validate_run_binding(
        V2_RUN_DIR,
        variant=V2_VARIANT,
        run_tag=V2_RUN_TAG,
        version="v2",
        entry_schema=V2_ENTRY_SCHEMA,
    )
    _require(
        v3_binding["split_bytes"] == v2_binding["split_bytes"],
        "canonical V3/V2 split.json files differ",
    )
    _require(
        v3_binding["split_sha256"] == v2_binding["split_sha256"],
        "canonical V3/V2 split SHA differs",
    )
    _require(
        v3_binding["data_sha256"] == v2_binding["data_sha256"],
        "canonical V3/V2 training data SHA differs",
    )
    v3_rows = _load_metrics(V3_RUN_DIR, V3_VARIANT, "V3")
    v2_rows = _load_metrics(V2_RUN_DIR, V2_VARIANT, "V2")
    report = build_report(
        v3_rows,
        v2_rows,
        v3_binding=v3_binding,
        v2_binding=v2_binding,
    )
    json_path = V3_RUN_DIR / JSON_OUTPUT_NAME
    markdown_path = V3_RUN_DIR / MARKDOWN_OUTPUT_NAME
    publications = {
        JSON_OUTPUT_NAME: _atomic_publish_idempotent(
            json_path,
            _canonical_json_bytes(report, pretty=True),
        ),
        MARKDOWN_OUTPUT_NAME: _atomic_publish_idempotent(
            markdown_path,
            render_markdown(report).encode("utf-8"),
        ),
    }
    return {
        "status": (
            "already_complete"
            if set(publications.values()) == {"unchanged"}
            else "published"
        ),
        "decision": report["decision"],
        "all_five_criteria_passed": report["all_five_criteria_passed"],
        "outputs": {
            JSON_OUTPUT_NAME: str(json_path),
            MARKDOWN_OUTPUT_NAME: str(markdown_path),
        },
        "publications": publications,
    }


def inspect_training_service() -> dict[str, str]:
    """Read only the fixed canonical V3 GPU2 training service."""

    process = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            SERVICE,
            (
                "--property="
                "LoadState,ActiveState,SubState,Result,ExecMainStatus"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    properties: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    collected = properties.get("LoadState") == "not-found"
    if process.returncode and not collected:
        detail = process.stderr.strip() or process.stdout.strip()
        raise DiagnosticError(
            f"cannot inspect fixed training service {SERVICE}: {detail}"
        )
    _require(
        "ActiveState" in properties or collected,
        f"systemd returned no state for fixed training service {SERVICE}",
    )
    return properties


def watch_and_diagnose(*, poll_seconds: float = 30.0) -> dict[str, Any]:
    """Poll no slower than every 30 seconds until epoch 450 is available."""

    _require(
        _is_number(poll_seconds) and 0 < float(poll_seconds) <= 30,
        "poll interval must be greater than 0 and no more than 30 seconds",
    )
    while True:
        last_epoch = current_v3_epoch()
        if last_epoch >= WINDOW_END:
            return diagnose_and_write()

        service = inspect_training_service()
        active_state = service.get("ActiveState", "inactive")
        if active_state in ACTIVE_STATES:
            print(
                "TPDNERV8V3_EPOCH450_DIAGNOSTIC_WAIT"
                f" service={SERVICE}"
                f" active_state={active_state}"
                f" last_epoch={last_epoch}"
                f" poll_seconds={float(poll_seconds):g}",
                flush=True,
            )
            time.sleep(float(poll_seconds))
            continue

        # Close the small race between the metrics snapshot and service stop.
        last_epoch = current_v3_epoch()
        if last_epoch >= WINDOW_END:
            return diagnose_and_write()
        raise DiagnosticError(
            "fixed V3 training service stopped before epoch 450: "
            f"service={SERVICE} "
            f"LoadState={service.get('LoadState')!r} "
            f"ActiveState={active_state!r} "
            f"SubState={service.get('SubState')!r} "
            f"Result={service.get('Result')!r} "
            f"ExecMainStatus={service.get('ExecMainStatus')!r} "
            f"last_epoch={last_epoch}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch and diagnose only the canonical V3 epoch-401--450 window"
        )
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not 0 < args.poll_seconds <= 30:
        parser.error("--poll-seconds must be > 0 and <= 30")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = watch_and_diagnose(poll_seconds=args.poll_seconds)
    except (DiagnosticError, OSError) as exc:
        print(
            f"TPDNERV8V3_EPOCH450_DIAGNOSTIC_ERROR {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
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


if __name__ == "__main__":
    raise SystemExit(main())
