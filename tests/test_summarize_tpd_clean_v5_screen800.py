from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from experiments.summarize_tpd_clean_v5_screen800 import (
    BUDGET_KEYS,
    CONTROL_VARIANT,
    DEFAULT_FORMAL_REFERENCE_ROOT,
    DEFAULT_SMOKE_ROOT,
    DEFAULT_TRAINING_SOURCE_LOCK,
    ENGINEERING_INTEGRITY_KEYS,
    EXPECTED_FAMILY,
    EXPECTED_SHALLOW_PARAMETERS,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINING_SOURCE_FILES,
    IncompleteArtifact,
    JSON_OUTPUT_NAME,
    LAST_FLOAT32_BELOW_ONE,
    MARKDOWN_OUTPUT_NAME,
    PRIMARY_VARIANT,
    REPO_ROOT,
    REQUIRED_INTEGRITY_CHECKS,
    ROLE_SPECS,
    SCHEMA,
    SEEDS,
    _best_under_budget,
    _validate_model_metadata,
    _validate_sweep,
    build_report,
    evaluate_engineering_gate,
    sha256_file,
    validate_smoke_reports,
    validate_training_source_lock,
)


def _point(
    matched: int,
    fa: float,
    miou: float,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    return {
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "threshold": threshold,
    }


def _role(
    fixed: dict[str, float | int],
    budget_counts: tuple[int, int, int, int, int],
    *,
    miou: float,
) -> dict[str, object]:
    budgets = {
        budget: _point(
            count,
            min(float(budget), 2e-6) / 2,
            miou,
            0.6,
        )
        for budget, count in zip(BUDGET_KEYS, budget_counts)
    }
    return {"fixed_threshold_0_5": fixed, "budgets": budgets}


def _passing_inputs() -> tuple[
    dict[tuple[str, int], dict[str, object]],
    dict[str, object],
    dict[str, bool],
]:
    full42 = {
        "roles": {
            "pd_primary": _role(
                _point(188, 2e-6, 0.94),
                (187, 188, 188, 188, 188),
                miou=0.94,
            ),
            "miou_primary": _role(
                _point(187, 0.0, 0.947),
                (187, 188, 188, 188, 188),
                miou=0.947,
            ),
        }
    }
    full3407 = {
        "roles": {
            "pd_primary": _role(
                _point(188, 2e-6, 0.93),
                (186, 187, 187, 187, 187),
                miou=0.93,
            ),
            "miou_primary": _role(
                _point(186, 0.0, 0.945),
                (186, 187, 187, 187, 187),
                miou=0.945,
            ),
        }
    }
    runs: dict[tuple[str, int], dict[str, object]] = {
        (PRIMARY_VARIANT, 42): full42,
        (PRIMARY_VARIANT, 3407): full3407,
    }
    for seed in SEEDS:
        capacity = copy.deepcopy(runs[(PRIMARY_VARIANT, seed)])
        for role in capacity["roles"].values():
            for point in role["budgets"].values():
                point["matched_target_count"] -= 1
                point["pd"] = point["matched_target_count"] / 189
        runs[(CONTROL_VARIANT, seed)] = capacity

    spd_budgets = copy.deepcopy(
        full42["roles"]["pd_primary"]["budgets"]
    )
    for budget in BUDGET_KEYS[1:]:
        spd_budgets[budget]["matched_target_count"] = 187
        spd_budgets[budget]["pd"] = 187 / 189
    spd = {
        "roles": {
            "pd_primary": {
                "fixed_threshold_0_5": _point(187, 0.0, 0.946),
                "budgets": spd_budgets,
            }
        }
    }
    integrity = {key: True for key in ENGINEERING_INTEGRITY_KEYS}
    return runs, spd, integrity


def test_all_gates_pass_for_registered_boundary_values() -> None:
    runs, spd, integrity = _passing_inputs()
    result = evaluate_engineering_gate(runs, spd, integrity)

    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"].values())


def test_gate_b_treats_equal_spd_point_as_weak_coverage() -> None:
    runs, spd, integrity = _passing_inputs()
    spd["roles"]["pd_primary"]["budgets"] = copy.deepcopy(
        runs[(PRIMARY_VARIANT, 42)]["roles"]["pd_primary"]["budgets"]
    )

    gate_b = evaluate_engineering_gate(runs, spd, integrity)["checks"][
        "gate_b_seed42_budget_and_spd"
    ]

    assert gate_b["passed"] is False
    assert (
        gate_b["subchecks"][
            "at_least_one_budget_not_covered_by_frozen_spd"
        ]
        is False
    )
    assert all(
        not check["full_not_covered_by_spd"]
        for check in gate_b["frozen_spd_comparisons"].values()
    )


def test_gate_d_rejects_lexicographic_tradeoff_as_strict_advantage() -> None:
    runs, spd, integrity = _passing_inputs()
    full = runs[(PRIMARY_VARIANT, 42)]
    capacity = copy.deepcopy(full)
    for role_name in ("pd_primary", "miou_primary"):
        full_point = full["roles"][role_name]["budgets"][BUDGET_KEYS[0]]
        capacity_point = capacity["roles"][role_name]["budgets"][
            BUDGET_KEYS[0]
        ]
        full_point["matched_target_count"] = 188
        full_point["pd"] = 188 / 189
        full_point["fa"] = 9e-7
        capacity_point["matched_target_count"] = 187
        capacity_point["pd"] = 187 / 189
        capacity_point["fa"] = 0.0
    runs[(CONTROL_VARIANT, 42)] = capacity

    seed_gate = evaluate_engineering_gate(runs, spd, integrity)["checks"][
        "gate_d_full_vs_capacity"
    ]["per_seed"]["42"]

    assert seed_gate["passed"] is False
    assert (
        seed_gate["subchecks"][
            "full_strict_at_one_or_more_registered_budgets"
        ]
        is False
    )


def test_gate_d_rejects_advantage_only_at_threshold_one() -> None:
    runs, spd, integrity = _passing_inputs()
    full = runs[(PRIMARY_VARIANT, 3407)]
    capacity = copy.deepcopy(full)
    for role in full["roles"].values():
        for point in role["budgets"].values():
            point["matched_target_count"] = 0
            point["pd"] = 0.0
            point["fa"] = 0.0
            point["miou"] = 0.0
            point["threshold"] = 1.0
    for role in capacity["roles"].values():
        for point in role["budgets"].values():
            point["matched_target_count"] = 0
            point["pd"] = 0.0
            point["fa"] = 1e-7
            point["miou"] = 0.0
    runs[(CONTROL_VARIANT, 3407)] = capacity

    seed_gate = evaluate_engineering_gate(runs, spd, integrity)["checks"][
        "gate_d_full_vs_capacity"
    ]["per_seed"]["3407"]

    assert seed_gate["passed"] is False
    assert seed_gate["full_strict_budget_advantages"]
    assert seed_gate["nonempty_full_strict_budget_advantages"] == []


@pytest.mark.parametrize(
    ("failed_budget_count", "expected_passed"),
    [(1, True), (2, False)],
)
def test_gate_c_requires_at_least_four_of_five_budget_stability_checks(
    failed_budget_count: int, expected_passed: bool
) -> None:
    runs, spd, integrity = _passing_inputs()
    full42 = runs[(PRIMARY_VARIANT, 42)]["roles"]["pd_primary"]["budgets"]
    full3407 = runs[(PRIMARY_VARIANT, 3407)]["roles"]["pd_primary"][
        "budgets"
    ]
    for budget in BUDGET_KEYS[-failed_budget_count:]:
        failed_count = full42[budget]["matched_target_count"] - 2
        full3407[budget]["matched_target_count"] = failed_count
        full3407[budget]["pd"] = failed_count / 189

    gate_c = evaluate_engineering_gate(runs, spd, integrity)["checks"][
        "gate_c_seed3407_stability"
    ]

    assert gate_c["budget_stability_pass_count"] == 5 - failed_budget_count
    assert gate_c["passed"] is expected_passed


def test_gate_d_fails_when_capacity_dominates_only_one_fixed_point() -> None:
    runs, spd, integrity = _passing_inputs()
    capacity_fixed = runs[(CONTROL_VARIANT, 42)]["roles"]["pd_primary"][
        "fixed_threshold_0_5"
    ]
    capacity_fixed["fa"] = 1e-6

    seed_gate = evaluate_engineering_gate(runs, spd, integrity)["checks"][
        "gate_d_full_vs_capacity"
    ]["per_seed"]["42"]

    assert seed_gate["passed"] is False
    assert seed_gate["capacity_dominated_points"] == [
        "pd_primary.fixed_threshold_0_5"
    ]
    assert seed_gate["subchecks"]["capacity_never_dominates_full"] is False


@pytest.mark.parametrize(
    ("gate_name", "mutate"),
    [
        (
            "gate_a_seed42_fixed_threshold",
            lambda runs: runs[(PRIMARY_VARIANT, 42)]["roles"][
                "pd_primary"
            ]["fixed_threshold_0_5"].update(matched_target_count=187, pd=187 / 189),
        ),
        (
            "gate_c_seed3407_stability",
            lambda runs: runs[(PRIMARY_VARIANT, 3407)]["roles"][
                "miou_primary"
            ]["fixed_threshold_0_5"].update(miou=0.939999),
        ),
    ],
)
def test_fixed_threshold_gate_failure_is_localized(
    gate_name: str,
    mutate: Callable[[dict[tuple[str, int], dict[str, object]]], None],
) -> None:
    runs, spd, integrity = _passing_inputs()
    mutate(runs)

    checks = evaluate_engineering_gate(runs, spd, integrity)["checks"]

    assert checks[gate_name]["passed"] is False


def test_gate_e_requires_every_integrity_subcheck() -> None:
    runs, spd, integrity = _passing_inputs()
    integrity["fixed_threshold_reproduction_exact"] = False

    gate_e = evaluate_engineering_gate(runs, spd, integrity)["checks"][
        "gate_e_engineering_integrity"
    ]

    assert gate_e["passed"] is False
    assert gate_e["subchecks"]["fixed_threshold_reproduction_exact"] is False


def test_gate_e_missing_key_is_incomplete_not_gate_failure() -> None:
    runs, spd, integrity = _passing_inputs()
    integrity.pop("preregistered_endpoint_provenance")

    with pytest.raises(
        IncompleteArtifact,
        match="Gate E integrity key set differs",
    ):
        evaluate_engineering_gate(runs, spd, integrity)


def _complete_sweep_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("protocol.json", "split.json", "summary.json", "metrics.jsonl"):
        (run_dir / name).write_text(f"{name}\n", encoding="utf-8")
    checkpoint_path = run_dir / "best.pth.tar"
    checkpoint_path.write_bytes(b"checkpoint")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")

    checkpoint_metrics = {
        "val_loss": 0.1,
        "miou": 0.94,
        "pd": 188 / 189,
        "tiny_pd": 1.0,
        "fa": 2e-6,
        "target_count": 189,
        "matched_target_count": 188,
        "predicted_object_count": 188,
        "unmatched_predicted_object_count": 0,
    }
    fixed = {**checkpoint_metrics, "threshold": 0.5}
    empty_last_float = {
        **checkpoint_metrics,
        "miou": 0.0,
        "pd": 0.0,
        "tiny_pd": 0.0,
        "fa": 0.0,
        "matched_target_count": 0,
        "predicted_object_count": 0,
        "unmatched_predicted_object_count": 0,
        "threshold": LAST_FLOAT32_BELOW_ONE,
    }
    empty_boundary = {
        **empty_last_float,
        "threshold": 1.0,
    }
    points = [fixed, empty_last_float, empty_boundary]
    budget_points = {
        budget: copy.deepcopy(_best_under_budget(points, float(budget)))
        for budget in BUDGET_KEYS
    }
    artifact_paths = {
        "protocol.json": run_dir / "protocol.json",
        "split.json": run_dir / "split.json",
        "summary.json": run_dir / "summary.json",
        "metrics.jsonl": run_dir / "metrics.jsonl",
        "checkpoint": checkpoint_path,
        "evaluator": evaluator,
    }
    payload = {
        "variant": PRIMARY_VARIANT,
        "dataset": "NUDT-SIRST",
        "seed": 42,
        "split_seed": 20260722,
        "validation_count": 133,
        "official_test_accessed": False,
        "checkpoint_epoch": 10,
        "checkpoint_role": ROLE_SPECS["pd_primary"]["checkpoint_role"],
        "validation_split_sha256": "a" * 64,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_validation_metrics": checkpoint_metrics,
        "points": points,
        "fixed_threshold_0_5": fixed,
        "fixed_threshold_0_5_checkpoint_audit": {
            "max_abs_non_strict_numeric_delta": 0.0
        },
        "best_points_under_fa_budget": budget_points,
        "threshold_provenance": {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "added_thresholds": [LAST_FLOAT32_BELOW_ONE, 1.0],
            "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
            "last_float32_semantics": "exact_one_score_plateau",
            "upper_boundary_threshold": 1.0,
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
            "total_unique_threshold_count": len(points),
        },
        "audit": {
            "expected_epochs": 800,
            "metrics_event_count": 800,
            "metrics_epoch_range": [1, 800],
            "summary_status": "complete",
            "selection_source": "internal_validation_only",
            "integrity_checks_passed": {
                key: True for key in REQUIRED_INTEGRITY_CHECKS
            },
            "artifact_sha256": {
                name: sha256_file(path) for name, path in artifact_paths.items()
            },
        },
    }
    sweep_path = run_dir / ROLE_SPECS["pd_primary"]["sweep"]
    sweep_path.write_text(json.dumps(payload), encoding="utf-8")
    arguments = {
        "run_dir": run_dir,
        "path": sweep_path,
        "checkpoint": {
            "epoch": 10,
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "role_name": "pd_primary",
        "variant": PRIMARY_VARIANT,
        "seed": 42,
        "expected_metrics": checkpoint_metrics,
        "split_hashes": {"used_val_sha256": "a" * 64},
        "evaluator_sha": sha256_file(evaluator),
        "label": "synthetic",
    }
    return sweep_path, arguments


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("miou", 0.01),
        ("predicted_object_count", 1),
        ("unmatched_predicted_object_count", 1),
    ],
)
def test_sweep_rejects_nonempty_threshold_one_endpoint(
    tmp_path: Path, field: str, bad_value: float | int
) -> None:
    sweep_path, arguments = _complete_sweep_fixture(tmp_path)
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    payload["points"][-1][field] = bad_value
    sweep_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IncompleteArtifact, match="threshold-1"):
        _validate_sweep(**arguments)


def test_sweep_requires_exact_nine_integrity_keys(tmp_path: Path) -> None:
    sweep_path, arguments = _complete_sweep_fixture(tmp_path)
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    payload["audit"]["integrity_checks_passed"]["unexpected"] = True
    sweep_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IncompleteArtifact, match="integrity key set"):
        _validate_sweep(**arguments)


def _model_metadata(variant: str) -> dict[str, object]:
    primary = variant == PRIMARY_VARIANT
    return {
        "variant": variant,
        "candidate_family": EXPECTED_FAMILY,
        "primary_candidate": primary,
        "mainline_contract": "Keep-Context-Saliency",
        "fourth_parallel_branch_added": False,
        "context_reference": (
            "positive_selector" if primary else "capacity_control"
        ),
        "context_code": (
            "centered_spatial_rms_tanh_fp32"
            if primary
            else "centered_spatial_rms_tanh_fp32_ignored"
        ),
        "context_selector": (
            "positive_centered_0p5_to_1p5" if primary else "neutral_one"
        ),
        "context_selector_floor": 0.5,
        "context_selector_ceiling": 1.5,
        "fusion_support": "positive_context_selected_saliency",
        "fusion_formula": (
            "K+S*tanh(saliency_scale*(1+0.5*context_code))"
        ),
        "learned_scales_per_block": 1,
        "residual_bound": "absolute_residual_at_most_absolute_saliency",
        "zero_scale_reference": "dense_spd_exact",
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "trainable_parameters": EXPECTED_TOTAL_PARAMETERS,
        "shallow_embedding_parameters": EXPECTED_SHALLOW_PARAMETERS,
        "shared_initialization_sha256": "a" * 64,
        "full_initialization_sha256": "b" * 64,
    }


@pytest.mark.parametrize("variant", [PRIMARY_VARIANT, CONTROL_VARIANT])
def test_v5_model_metadata_contract_is_variant_specific(variant: str) -> None:
    metadata = _model_metadata(variant)
    _validate_model_metadata(metadata, variant, "synthetic")

    metadata["context_selector"] = "wrong"
    with pytest.raises(IncompleteArtifact, match="context_selector"):
        _validate_model_metadata(metadata, variant, "synthetic")


def test_training_source_lock_binds_exact_frozen_32_file_set() -> None:
    payload, digest = validate_training_source_lock(
        DEFAULT_TRAINING_SOURCE_LOCK
    )

    assert len(payload["source_sha256"]) == 32
    assert set(payload["source_sha256"]) == set(
        EXPECTED_TRAINING_SOURCE_FILES
    )
    assert digest == sha256_file(DEFAULT_TRAINING_SOURCE_LOCK)


def test_training_source_lock_rejects_a_31_file_subset(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        DEFAULT_TRAINING_SOURCE_LOCK.read_text(encoding="utf-8")
    )
    payload["source_sha256"].pop("warmup_scheduler.py")
    path = tmp_path / "incomplete-source-lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IncompleteArtifact, match="frozen 32-file"):
        validate_training_source_lock(path)


def test_persisted_smoke_envelopes_are_source_lock_bound() -> None:
    payload, _ = validate_training_source_lock(
        DEFAULT_TRAINING_SOURCE_LOCK
    )
    result = validate_smoke_reports(DEFAULT_SMOKE_ROOT, payload)

    assert result["passed"] is True
    assert result["binding"] == "training_source_lock.smoke_sha256"
    assert set(result["reports"]) == {
        "cpu_all.json",
        "gpu2_full.json",
        "gpu3_capacity.json",
    }
    assert result["reports"]["gpu2_full.json"]["cuda_visible_devices"] == "2"
    assert (
        result["reports"]["gpu3_capacity.json"]["cuda_visible_devices"] == "3"
    )


def _copy_smoke_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir()
    for name in ("cpu_all.json", "gpu2_full.json", "gpu3_capacity.json"):
        shutil.copy2(DEFAULT_SMOKE_ROOT / name, smoke_root / name)
    payload = json.loads(
        DEFAULT_TRAINING_SOURCE_LOCK.read_text(encoding="utf-8")
    )
    return smoke_root, payload


def test_smoke_rejects_wrong_physical_gpu_provenance(
    tmp_path: Path,
) -> None:
    smoke_root, payload = _copy_smoke_fixture(tmp_path)
    path = smoke_root / "gpu2_full.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["cuda_visible_devices"] = "0"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    payload["smoke_sha256"]["gpu2_full.json"] = sha256_file(path)

    with pytest.raises(IncompleteArtifact, match="CUDA visibility"):
        validate_smoke_reports(smoke_root, payload)


def test_smoke_rejects_zero_gradient_even_when_digest_is_rebound(
    tmp_path: Path,
) -> None:
    smoke_root, payload = _copy_smoke_fixture(tmp_path)
    path = smoke_root / "cpu_all.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    gradients = envelope["report"]["variants"][0]["scale_gradient_l1"]
    gradients[next(iter(gradients))] = 0.0
    path.write_text(json.dumps(envelope), encoding="utf-8")
    payload["smoke_sha256"]["cpu_all.json"] = sha256_file(path)

    with pytest.raises(IncompleteArtifact, match="scale_gradient_l1"):
        validate_smoke_reports(smoke_root, payload)


def test_missing_runs_return_incomplete_with_null_gate(
    tmp_path: Path,
) -> None:
    report = build_report(
        tmp_path / "missing-candidate-root",
        DEFAULT_FORMAL_REFERENCE_ROOT,
        DEFAULT_SMOKE_ROOT,
        DEFAULT_TRAINING_SOURCE_LOCK,
    )

    assert report["schema"] == SCHEMA
    assert report["status"] == "incomplete"
    assert report["decision"] == "INCOMPLETE"
    assert report["gate_evaluated"] is False
    assert report["engineering_gate_passed"] is None
    assert report["ner_stage_authorized"] is False
    assert len(report["incomplete_reasons"]) == 4


def test_absolute_cli_without_pythonpath_from_arbitrary_cwd(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    script = (
        REPO_ROOT / "experiments/summarize_tpd_clean_v5_screen800.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--candidate-root",
            str(tmp_path / "missing-runs"),
            "--output-dir",
            str(output_dir),
            "--require-complete",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert "TPDCLEANV5_SUMMARY status=incomplete gate=None" in completed.stdout
    report = json.loads(
        (output_dir / JSON_OUTPUT_NAME).read_text(encoding="utf-8")
    )
    markdown = (output_dir / MARKDOWN_OUTPUT_NAME).read_text(encoding="utf-8")
    assert report["status"] == "incomplete"
    assert report["gate_evaluated"] is False
    assert report["engineering_gate_passed"] is None
    assert "Gate A–E were not evaluated" in markdown
