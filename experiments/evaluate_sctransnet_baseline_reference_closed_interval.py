#!/usr/bin/env python3
"""Closed-interval reference evaluation for the frozen SCTransNet baseline.

This wrapper deliberately reuses the existing ``evaluate_pd_fa_sweep``
implementation and the same closed-probability-interval threshold function
used by the formal V8-MPRS-DCH+NER evaluator.  It is intended for a separate
hard-linked reference view of the already completed baseline run, so it never
overwrites the historical sweep that predates endpoint completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as base  # noqa: E402
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
)


SCHEMA = "sctransnet_baseline_reference_closed_interval_evaluator_v1"
OUTPUT_SCHEMA = (
    "sctransnet_baseline_reference_closed_interval_evaluation_v1"
)
FINAL_METRIC_COVERAGE_SCHEMA = (
    "sctransnet_baseline_reference_final_metric_coverage_v1"
)
DATASET = "NUDT-SIRST"
VARIANT = "original"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
CHECKPOINTS = ("best.pth.tar", "best_miou.pth.tar")
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)


def evaluator_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dataset": DATASET,
        "variant": VARIANT,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "checkpoints": list(CHECKPOINTS),
        "metric_core": "experiments.evaluate_pd_fa_sweep",
        "closed_interval_core": (
            "experiments.evaluate_tpd_clean_v6_pd_fa."
            "adaptive_thresholds_closed_interval"
        ),
        "fixed_threshold": 0.5,
        "fa_budgets": list(FA_BUDGETS),
        "prediction_comparison": "prediction > threshold",
        "historical_sweep_overwrite_allowed": False,
        "reference_semantics": (
            "current closed-interval re-evaluation of an already trained "
            "historical checkpoint"
        ),
        "endpoint_protocol_preregistered_before_historical_training": False,
    }


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, observed={observed!r}"
        )


def validate_formal_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Validate the result-independent baseline reference CLI contract."""

    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pth.tar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-epochs", type=int, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--extra-thresholds",
        type=float,
        nargs="+",
        default=list(EXTRA_THRESHOLDS),
    )
    parser.add_argument("--tail-logit-step", type=float, default=0.1)
    parser.add_argument(
        "--fa-budgets",
        type=float,
        nargs="+",
        default=list(FA_BUDGETS),
    )
    parser.add_argument("--match-radius", type=float, default=None)
    parser.add_argument("--tiny-area", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args, _ = parser.parse_known_args(values)

    if args.overwrite:
        raise ValueError("baseline reference evaluation forbids --overwrite")
    if args.checkpoint not in CHECKPOINTS:
        raise ValueError(
            "baseline reference accepts only best.pth.tar or "
            "best_miou.pth.tar"
        )
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("baseline reference device must be cpu or cuda:0")
    if args.expected_epochs not in (None, EXPECTED_EPOCHS):
        raise ValueError(
            f"baseline reference requires expected_epochs={EXPECTED_EPOCHS}"
        )
    _require_equal("threshold_min", args.threshold_min, 0.01)
    _require_equal("threshold_max", args.threshold_max, 0.99)
    _require_equal("threshold_step", args.threshold_step, 0.01)
    _require_equal("extra_thresholds", tuple(args.extra_thresholds), EXTRA_THRESHOLDS)
    _require_equal("tail_logit_step", args.tail_logit_step, 0.1)
    _require_equal("fa_budgets", tuple(args.fa_budgets), FA_BUDGETS)
    if args.match_radius not in (None, 3.0):
        raise ValueError("baseline reference match_radius must be omitted or 3.0")
    if args.tiny_area not in (None, 9):
        raise ValueError("baseline reference tiny_area must be omitted or 9")
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(location: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _require_sha256(location: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _required_endpoint_points(payload: Mapping[str, Any]) -> None:
    provenance = _require_mapping(
        "threshold_provenance",
        payload.get("threshold_provenance"),
    )
    _require_equal(
        "closed_probability_interval",
        provenance.get("closed_probability_interval"),
        True,
    )
    _require_equal("score_dtype", provenance.get("score_dtype"), "float32")
    _require_equal(
        "last_float32_below_one",
        provenance.get("last_float32_below_one"),
        LAST_FLOAT32_BELOW_ONE,
    )
    _require_equal(
        "upper_boundary_threshold",
        provenance.get("upper_boundary_threshold"),
        UPPER_BOUNDARY_THRESHOLD,
    )
    _require_equal(
        "posthoc_endpoint_completion",
        provenance.get("posthoc_endpoint_completion"),
        True,
    )
    _require_equal(
        "preregistered_endpoint_completion",
        provenance.get("preregistered_endpoint_completion"),
        False,
    )
    _require_equal(
        "endpoint_protocol_stage",
        provenance.get("endpoint_protocol_stage"),
        "current_reference_reevaluation_after_historical_training",
    )
    _require_equal(
        "upper_boundary_comparison",
        provenance.get("upper_boundary_comparison"),
        "prediction > threshold",
    )
    _require_equal(
        "upper_boundary_semantics",
        provenance.get("upper_boundary_semantics"),
        "empty_prediction_pd0_fa0",
    )
    added = provenance.get("added_thresholds")
    if not isinstance(added, list) or list(map(float, added)) != [
        LAST_FLOAT32_BELOW_ONE,
        UPPER_BOUNDARY_THRESHOLD,
    ]:
        raise ValueError("closed-interval added thresholds differ")
    points = payload.get("points")
    if not isinstance(points, list):
        raise ValueError("baseline sweep points are missing")
    by_threshold = {
        float(point["threshold"]): point
        for point in points
        if isinstance(point, Mapping) and "threshold" in point
    }
    if (
        LAST_FLOAT32_BELOW_ONE not in by_threshold
        or UPPER_BOUNDARY_THRESHOLD not in by_threshold
    ):
        raise ValueError("baseline sweep endpoint points are missing")
    endpoint = by_threshold[UPPER_BOUNDARY_THRESHOLD]
    for name, expected in {
        "pd": 0.0,
        "fa": 0.0,
        "matched_target_count": 0,
        "predicted_object_count": 0,
        "unmatched_predicted_object_count": 0,
    }.items():
        _require_equal(f"upper endpoint {name}", endpoint.get(name), expected)


def validate_output_identity(
    payload: Mapping[str, Any],
    *,
    expected_run_dir: Path | None = None,
    expected_checkpoint: str | None = None,
) -> None:
    """Validate the actual fields emitted by the shared evaluator wrapper."""

    _require_equal("schema", payload.get("schema"), OUTPUT_SCHEMA)
    _require_equal("dataset", payload.get("dataset"), DATASET)
    _require_equal("variant", payload.get("variant"), VARIANT)
    _require_equal("seed", payload.get("seed"), TRAINING_SEED)
    _require_equal("split_seed", payload.get("split_seed"), SPLIT_SEED)
    _require_equal(
        "official_test_accessed",
        payload.get("official_test_accessed"),
        False,
    )
    configuration = payload.get("threshold_configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("baseline sweep threshold_configuration is missing")
    _require_equal(
        "threshold configuration",
        dict(configuration),
        {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": list(EXTRA_THRESHOLDS),
            "tail_logit_step": 0.1,
            "fa_budgets": list(FA_BUDGETS),
        },
    )
    _required_endpoint_points(payload)
    raw_run_dir = Path(str(payload.get("run_directory")))
    raw_checkpoint_path = Path(str(payload.get("checkpoint")))
    if (
        not raw_run_dir.is_absolute()
        or raw_run_dir != raw_run_dir.resolve()
        or not raw_checkpoint_path.is_absolute()
        or raw_checkpoint_path != raw_checkpoint_path.resolve()
    ):
        raise ValueError(
            "baseline run/checkpoint paths must be absolute and normalized"
        )
    run_dir = raw_run_dir
    checkpoint_path = raw_checkpoint_path
    if expected_run_dir is not None:
        _require_equal(
            "run_directory",
            run_dir,
            Path(expected_run_dir).resolve(),
        )
    if checkpoint_path.parent != run_dir:
        raise ValueError("baseline checkpoint is not inside the run directory")
    checkpoint_name = checkpoint_path.name
    if expected_checkpoint is not None:
        _require_equal("checkpoint filename", checkpoint_name, expected_checkpoint)
    if checkpoint_name not in CHECKPOINTS:
        raise ValueError("baseline checkpoint filename is not formal")
    expected_role = {
        "best.pth.tar": "best_validation_pd_primary",
        "best_miou.pth.tar": "best_validation_miou_secondary",
    }[checkpoint_name]
    _require_equal("checkpoint role", payload.get("checkpoint_role"), expected_role)
    checkpoint_sha256 = _require_sha256(
        "checkpoint_sha256",
        payload.get("checkpoint_sha256"),
    )
    _require_equal(
        "checkpoint_sha256 versus current file",
        checkpoint_sha256,
        _sha256_file(checkpoint_path),
    )
    audit = _require_mapping("audit", payload.get("audit"))
    artifact_hashes = _require_mapping(
        "audit.artifact_sha256",
        audit.get("artifact_sha256"),
    )
    expected_artifact_hashes = {
        "protocol.json": _sha256_file(run_dir / "protocol.json"),
        "split.json": _sha256_file(run_dir / "split.json"),
        "summary.json": _sha256_file(run_dir / "summary.json"),
        "metrics.jsonl": _sha256_file(run_dir / "metrics.jsonl"),
        "checkpoint": checkpoint_sha256,
        "evaluator": _sha256_file(Path(__file__).resolve()),
    }
    _require_equal(
        "audit artifact sha256",
        dict(artifact_hashes),
        expected_artifact_hashes,
    )
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    if not isinstance(split, Mapping):
        raise ValueError("baseline split is not an object")
    hashes = _require_mapping("split.hashes", split.get("hashes"))
    _require_equal(
        "validation split sha256",
        payload.get("validation_split_sha256"),
        hashes.get("used_val_sha256"),
    )
    invocation = audit.get("invocation_argv")
    if (
        not isinstance(invocation, list)
        or len(invocation) < 2
        or not Path(str(invocation[1])).is_absolute()
        or Path(str(invocation[1]))
        != Path(str(invocation[1])).resolve()
        or Path(str(invocation[1])).resolve() != Path(__file__).resolve()
    ):
        raise ValueError("baseline evaluator invocation identity differs")
    parsed = _require_mapping(
        "audit.parsed_arguments",
        audit.get("parsed_arguments"),
    )
    parsed_run_dir = Path(str(parsed.get("run_dir")))
    if not parsed_run_dir.is_absolute() or parsed_run_dir != parsed_run_dir.resolve():
        raise ValueError("parsed run directory must be absolute and normalized")
    _require_equal("parsed run directory", parsed_run_dir, run_dir)
    _require_equal(
        "parsed checkpoint",
        parsed.get("checkpoint"),
        checkpoint_name,
    )
    for name, expected in {
        "expected_epochs": EXPECTED_EPOCHS,
        "threshold_min": 0.01,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "extra_thresholds": list(EXTRA_THRESHOLDS),
        "tail_logit_step": 0.1,
        "fa_budgets": list(FA_BUDGETS),
        "match_radius": None,
        "tiny_area": None,
        "overwrite": False,
    }.items():
        _require_equal(
            f"parsed evaluator argument {name}",
            parsed.get(name),
            expected,
        )
    _require_equal(
        "audit expected epochs",
        audit.get("expected_epochs"),
        EXPECTED_EPOCHS,
    )
    _require_equal(
        "audit metrics count",
        audit.get("metrics_event_count"),
        EXPECTED_EPOCHS,
    )
    _require_equal(
        "audit metrics epoch range",
        audit.get("metrics_epoch_range"),
        [1, EXPECTED_EPOCHS],
    )
    _require_equal(
        "audit summary status",
        audit.get("summary_status"),
        "complete",
    )
    checks = _require_mapping(
        "audit.integrity_checks_passed",
        audit.get("integrity_checks_passed"),
    )
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("baseline evaluator checks are incomplete")
    _require_equal(
        "evaluator_contract",
        payload.get("evaluator_contract"),
        evaluator_contract(),
    )
    evaluated = _require_mapping(
        "evaluated_checkpoint_identity",
        payload.get("evaluated_checkpoint_identity"),
    )
    _require_equal("evaluated filename", evaluated.get("filename"), checkpoint_name)
    _require_equal("evaluated role", evaluated.get("role"), expected_role)
    _require_equal("evaluated sha256", evaluated.get("sha256"), checkpoint_sha256)
    _require_equal(
        "reference artifact validation",
        payload.get("reference_artifact_validation_passed"),
        True,
    )
    provenance = _require_mapping(
        "reference_provenance",
        payload.get("reference_provenance"),
    )
    _require_equal(
        "reference kind",
        provenance.get("kind"),
        "historical_checkpoint_current_reference_closed_interval_reevaluation",
    )
    _require_equal(
        "endpoint preregistration marker",
        provenance.get("endpoint_protocol_preregistered_before_historical_training"),
        False,
    )
    _require_equal(
        "historical checkpoint marker",
        provenance.get("historical_training_checkpoint_unchanged"),
        True,
    )
    _require_equal(
        "reference interpretation",
        provenance.get("interpretation"),
        (
            "the checkpoint is historical; the closed-interval sweep is a "
            "current same-metric reference evaluation"
        ),
    )
    coverage = _require_mapping(
        "final_metric_coverage",
        payload.get("final_metric_coverage"),
    )
    _require_equal(
        "final metric coverage schema",
        coverage.get("schema"),
        FINAL_METRIC_COVERAGE_SCHEMA,
    )
    _require_equal(
        "final metric coverage fixed threshold",
        coverage.get("fixed_threshold"),
        0.5,
    )
    _require_equal(
        "final metric coverage complete",
        coverage.get("all_required_metrics_present"),
        True,
    )
    fixed = _require_mapping(
        "fixed_threshold_0_5",
        payload.get("fixed_threshold_0_5"),
    )
    _require_equal(
        "fixed metric coverage",
        dict(coverage.get("fixed_threshold_0_5", {})),
        {
            name: fixed[name]
            for name in (
                "pd",
                "fa",
                "miou",
                "false_objects_per_image",
            )
        },
    )
    raw_budgets = _require_mapping(
        "best_points_under_fa_budget",
        payload.get("best_points_under_fa_budget"),
    )
    expected_budget_keys = tuple(f"{value:.10g}" for value in FA_BUDGETS)
    _require_equal(
        "raw budget keys",
        set(raw_budgets),
        set(expected_budget_keys),
    )
    expected_coverage = {
        key: {
            "budget": budget,
            "pd": raw_budgets[key]["pd"],
            "achieved_fa": raw_budgets[key]["fa"],
            "threshold": raw_budgets[key]["threshold"],
            "matched_target_count": raw_budgets[key][
                "matched_target_count"
            ],
            "target_count": raw_budgets[key]["target_count"],
        }
        for budget, key in zip(FA_BUDGETS, expected_budget_keys)
    }
    _require_equal(
        "budget metric coverage",
        dict(
            _require_mapping(
                "final_metric_coverage.pd_at_fa_budget",
                coverage.get("pd_at_fa_budget"),
            )
        ),
        expected_coverage,
    )


def finalize_reference_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach explicit identity and reference-scope fields to a base sweep."""

    ready = dict(payload)
    checkpoint_path = Path(str(ready.get("checkpoint"))).resolve()
    checkpoint_name = checkpoint_path.name
    if checkpoint_name not in CHECKPOINTS:
        raise ValueError("baseline checkpoint filename is not formal")
    role = {
        "best.pth.tar": "best_validation_pd_primary",
        "best_miou.pth.tar": "best_validation_miou_secondary",
    }[checkpoint_name]
    checkpoint_sha256 = _require_sha256(
        "checkpoint_sha256",
        ready.get("checkpoint_sha256"),
    )
    fixed = _require_mapping(
        "fixed_threshold_0_5",
        ready.get("fixed_threshold_0_5"),
    )
    raw_budgets = _require_mapping(
        "best_points_under_fa_budget",
        ready.get("best_points_under_fa_budget"),
    )
    threshold_provenance = dict(
        _require_mapping(
            "threshold_provenance",
            ready.get("threshold_provenance"),
        )
    )
    threshold_provenance.update(
        {
            "posthoc_endpoint_completion": True,
            "preregistered_endpoint_completion": False,
            "endpoint_protocol_stage": (
                "current_reference_reevaluation_after_historical_training"
            ),
        }
    )
    ready.update(
        {
            "schema": OUTPUT_SCHEMA,
            "threshold_provenance": threshold_provenance,
            "evaluator_contract": evaluator_contract(),
            "evaluated_checkpoint_identity": {
                "filename": checkpoint_name,
                "role": role,
                "sha256": checkpoint_sha256,
            },
            "reference_artifact_validation_passed": True,
            "reference_provenance": {
                "kind": (
                    "historical_checkpoint_current_reference_"
                    "closed_interval_reevaluation"
                ),
                "historical_training_checkpoint_unchanged": True,
                "endpoint_protocol_preregistered_before_historical_training": False,
                "interpretation": (
                    "the checkpoint is historical; the closed-interval sweep "
                    "is a current same-metric reference evaluation"
                ),
            },
            "final_metric_coverage": {
                "schema": FINAL_METRIC_COVERAGE_SCHEMA,
                "fixed_threshold": 0.5,
                "fixed_threshold_0_5": {
                    name: fixed[name]
                    for name in (
                        "pd",
                        "fa",
                        "miou",
                        "false_objects_per_image",
                    )
                },
                "pd_at_fa_budget": {
                    key: {
                        "budget": budget,
                        "pd": raw_budgets[key]["pd"],
                        "achieved_fa": raw_budgets[key]["fa"],
                        "threshold": raw_budgets[key]["threshold"],
                        "matched_target_count": raw_budgets[key][
                            "matched_target_count"
                        ],
                        "target_count": raw_budgets[key]["target_count"],
                    }
                    for budget, key in zip(
                        FA_BUDGETS,
                        (f"{value:.10g}" for value in FA_BUDGETS),
                    )
                },
                "all_required_metrics_present": True,
            },
        }
    )
    validate_output_identity(ready)
    return ready


def _atomic_write_output(
    path: Path,
    payload: Mapping[str, Any],
    overwrite: bool,
) -> None:
    if overwrite:
        raise ValueError("baseline reference evaluation forbids overwrite")
    ready = base.json_ready(finalize_reference_output(payload))
    content = (
        json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
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
            raise FileExistsError(
                f"refusing to replace existing baseline sweep: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    values = list(sys.argv[1:])
    if "-h" not in values and "--help" not in values:
        args = validate_formal_arguments(values)
        configure_v8_inference(str(args.device))
    base.adaptive_thresholds = adaptive_thresholds_closed_interval
    # ``base`` already owns the original/SCTransNet builder.
    base.__file__ = __file__
    base.write_output_json = _atomic_write_output
    base.main()


__all__ = [
    "CHECKPOINTS",
    "DATASET",
    "EXPECTED_EPOCHS",
    "EXTRA_THRESHOLDS",
    "FA_BUDGETS",
    "FINAL_METRIC_COVERAGE_SCHEMA",
    "OUTPUT_SCHEMA",
    "SCHEMA",
    "SPLIT_SEED",
    "TRAINING_SEED",
    "VARIANT",
    "evaluator_contract",
    "finalize_reference_output",
    "main",
    "validate_formal_arguments",
    "validate_output_identity",
]


if __name__ == "__main__":
    main()
