from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from experiments import summarize_tpd_clean_v3_screen800 as summary


VARIANTS = (
    "tpd_clean_v3_full",
    "tpd_clean_v3_sal_capacity",
)
SEEDS = (42, 3407)
BUDGETS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")


def _point(matched: int, fa: float, miou: float) -> dict[str, Any]:
    target_count = 189
    return {
        "val_loss": 0.001,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.95,
        "pixel_recall": 0.95,
        "pixel_f1": 0.95,
        "pd": matched / target_count,
        "tiny_pd": 1.0,
        "fa": fa,
        "false_objects_per_image": 0.01,
        "target_count": target_count,
        "matched_target_count": matched,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 39,
        "predicted_object_count": matched + 1,
        "unmatched_predicted_object_count": 1,
        "valid_pixel_count": 8_716_288,
    }


def _budgets(
    p1: tuple[int, float, float],
    p5: tuple[int, float, float],
    p10: tuple[int, float, float],
    p50: tuple[int, float, float],
    p100: tuple[int, float, float],
) -> dict[str, dict[str, Any]]:
    values = (p1, p5, p10, p50, p100)
    return {
        budget: {**_point(*value), "threshold": 0.9}
        for budget, value in zip(BUDGETS, values)
    }


def _record(
    pd_fixed: tuple[int, float, float],
    miou_fixed: tuple[int, float, float],
    pd_budgets: dict[str, dict[str, Any]],
    miou_budgets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "roles": {
            "pd_primary": {
                "checkpoint_epoch": 1,
                "fixed_threshold_0_5": {
                    **_point(*pd_fixed),
                    "threshold": 0.5,
                },
                "budgets": copy.deepcopy(pd_budgets),
            },
            "miou_primary": {
                "checkpoint_epoch": 2,
                "fixed_threshold_0_5": {
                    **_point(*miou_fixed),
                    "threshold": 0.5,
                },
                "budgets": copy.deepcopy(miou_budgets or pd_budgets),
            },
        }
    }


def _consistent_sweep_points(
    fixed: dict[str, Any],
    requested_budgets: dict[str, dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any] | None]]:
    fixed_point = copy.deepcopy(fixed)
    fixed_point["threshold"] = 0.5
    points = [fixed_point]
    by_metrics = {
        json.dumps(
            {key: value for key, value in fixed_point.items() if key != "threshold"},
            sort_keys=True,
        ): fixed_point
    }
    next_threshold = 0.51
    for budget in BUDGETS:
        requested = requested_budgets[budget]
        if requested is None:
            continue
        candidate = copy.deepcopy(requested)
        signature = json.dumps(
            {key: value for key, value in candidate.items() if key != "threshold"},
            sort_keys=True,
        )
        if signature in by_metrics:
            continue
        candidate["threshold"] = next_threshold
        next_threshold += 0.01
        by_metrics[signature] = candidate
        points.append(candidate)
    points.sort(key=lambda point: float(point["threshold"]))
    recomputed = {
        budget: copy.deepcopy(
            summary._best_sweep_point_under_fa(points, float(budget))
        )
        for budget in BUDGETS
    }
    return points, recomputed


def _passing_gate_inputs() -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    full_budgets = _budgets(
        (187, 8e-7, 0.93),
        (188, 4e-6, 0.94),
        (189, 8e-6, 0.94),
        (189, 8e-6, 0.94),
        (189, 8e-6, 0.94),
    )
    control_budgets = _budgets(
        (186, 9e-7, 0.92),
        (187, 4.5e-6, 0.93),
        (188, 9e-6, 0.93),
        (188, 9e-6, 0.93),
        (188, 9e-6, 0.93),
    )
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in SEEDS:
        runs[("tpd_clean_v3_full", seed)] = _record(
            (188, 4e-6, 0.94),
            (187, 8e-7, 0.947),
            full_budgets,
        )
        runs[("tpd_clean_v3_sal_capacity", seed)] = _record(
            (187, 4.5e-6, 0.93),
            (186, 9e-7, 0.945),
            control_budgets,
        )
    references = {
        "spd": _record(
            (187, 0.0, 0.946),
            (187, 4e-7, 0.949),
            _budgets(
                (186, 0.0, 0.94),
                (187, 1e-6, 0.94),
                (188, 9e-6, 0.91),
                (188, 2e-5, 0.80),
                (188, 2e-5, 0.80),
            ),
        ),
        "tpd_v1": _record(
            (188, 1e-6, 0.933),
            (186, 4e-7, 0.942),
            full_budgets,
        ),
        "v2_sal_only": _record(
            (188, 5e-6, 0.915),
            (185, 3e-7, 0.935),
            _budgets(
                (186, 9e-7, 0.90),
                (188, 4.8e-6, 0.91),
                (189, 6e-6, 0.91),
                (189, 6e-6, 0.91),
                (189, 6e-6, 0.91),
            ),
        ),
        "v2_full": _record(
            (189, 1e-5, 0.91),
            (186, 3e-6, 0.936),
            full_budgets,
        ),
    }
    return runs, references


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics_rows(
    variant: str,
    pd_fixed: dict[str, Any],
    miou_fixed: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for epoch in range(1, 801):
        if epoch == 1:
            metrics = pd_fixed
        elif epoch == 2:
            metrics = miou_fixed
        else:
            metrics = _point(100, 1e-4, 0.5)
        row = {
            "epoch": epoch,
            "variant": variant,
            "train_loss": 0.1,
            "learning_rate": 1e-4,
            "processed_train_samples": 530,
            "epoch_seconds": 1.0,
            **metrics,
            "new_best_pd": epoch == 1,
            "new_best_miou": epoch in (1, 2),
        }
        rows.append(row)
    return rows


def _write_candidate_run(
    root: Path,
    variant: str,
    seed: int,
    record: dict[str, Any],
) -> Path:
    run_dir = (
        root
        / "NUDT-SIRST"
        / variant
        / f"seed_{seed}_screen800_pd_fp32_shared4x5090_v1"
    )
    run_dir.mkdir(parents=True)
    pd_fixed = copy.deepcopy(
        record["roles"]["pd_primary"]["fixed_threshold_0_5"]
    )
    pd_fixed.pop("threshold", None)
    miou_fixed = copy.deepcopy(
        record["roles"]["miou_primary"]["fixed_threshold_0_5"]
    )
    miou_fixed.pop("threshold", None)
    rows = _metrics_rows(variant, pd_fixed, miou_fixed)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    hashes = {
        "full_internal_train_sha256": "1" * 64,
        "full_internal_val_sha256": "2" * 64,
        "used_train_sha256": "1" * 64,
        "used_val_sha256": "2" * 64,
    }
    split = {
        "dataset": "NUDT-SIRST",
        "split_seed": 20260722,
        "used_train_count": 530,
        "used_val_count": 133,
        "hashes": hashes,
        "official_test_accessed": False,
        "source": "img_idx/train_NUDT-SIRST.txt",
    }
    _write_json(run_dir / "split.json", split)
    model = {
        "variant": variant,
        "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
        "primary_candidate": variant == "tpd_clean_v3_full",
        "mainline_contract": "Keep-Context-Saliency",
        "fourth_parallel_branch_added": False,
        "total_parameters": 10_843_475,
        "trainable_parameters": 10_843_475,
        "shallow_embedding_parameters": 66_496,
        "shared_initialization_sha256": f"{seed + 10_000:064x}",
        "full_initialization_sha256": f"{seed:064x}",
    }
    protocol = {
        "arguments": {
            "variant": variant,
            "dataset": "NUDT-SIRST",
            "epochs": 800,
            "batch_size": 16,
            "patch_size": 256,
            "workers": 0,
            "seed": seed,
            "split_seed": 20260722,
            "val_fraction": 0.2,
            "eval_every": 1,
            "base_lr": 0.001,
            "min_lr": 0.00001,
            "warmup_epochs": 10,
            "threshold": 0.5,
            "match_radius": 3.0,
            "tiny_area": 9,
            "amp": False,
            "max_train_images": None,
            "max_val_images": None,
        },
        "model": model,
        "official_test_accessed": False,
        "run_directory": str(run_dir),
        "checkpoint_policy": "internal validation only",
    }
    _write_json(run_dir / "protocol.json", protocol)
    summary_payload = {
        "status": "complete",
        "variant": variant,
        "dataset": "NUDT-SIRST",
        "seed": seed,
        "best_epoch": 1,
        "best_validation_metrics": pd_fixed,
        "best_pd_epoch": 1,
        "best_pd_validation_metrics": pd_fixed,
        "best_miou_epoch": 2,
        "best_miou_validation_metrics": miou_fixed,
        "primary_selection_metric": "validation Pd, then lower Fa",
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "model": model,
        "split_hashes": hashes,
        "best_checkpoint": str(run_dir / "best.pth.tar"),
        "best_miou_checkpoint": str(run_dir / "best_miou.pth.tar"),
        "last_checkpoint": str(run_dir / "last.pth.tar"),
    }
    _write_json(run_dir / "summary.json", summary_payload)
    (run_dir / "best.pth.tar").write_bytes(
        f"{variant}:{seed}:best".encode()
    )
    (run_dir / "best_miou.pth.tar").write_bytes(
        f"{variant}:{seed}:best_miou".encode()
    )
    (run_dir / "last.pth.tar").write_bytes(f"{variant}:{seed}:last".encode())

    common_hashes = {
        name: _sha256(run_dir / name)
        for name in ("protocol.json", "split.json", "summary.json", "metrics.jsonl")
    }
    for role, checkpoint_name, sweep_name, role_name, selected in (
        (
            "pd_primary",
            "best.pth.tar",
            "pd_fa_sweep_best.pth.json",
            "best_validation_pd_primary",
            pd_fixed,
        ),
        (
            "miou_primary",
            "best_miou.pth.tar",
            "pd_fa_sweep_best_miou.pth.json",
            "best_validation_miou_secondary",
            miou_fixed,
        ),
    ):
        checkpoint = run_dir / checkpoint_name
        fixed = {**selected, "threshold": 0.5}
        points, budget_points = _consistent_sweep_points(
            fixed,
            record["roles"][role]["budgets"],
        )
        sweep = {
            "variant": variant,
            "seed": seed,
            "dataset": "NUDT-SIRST",
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": record["roles"][role]["checkpoint_epoch"],
            "checkpoint_role": role_name,
            "checkpoint_sha256": _sha256(checkpoint),
            "fixed_threshold_0_5": fixed,
            "best_points_under_fa_budget": budget_points,
            "points": points,
            "audit": {
                "expected_epochs": 800,
                "metrics_event_count": 800,
                "metrics_epoch_range": [1, 800],
                "summary_status": "complete",
                "selection_source": "internal_validation_only",
                "integrity_checks_passed": {
                    "summary_complete": True,
                    "metrics_complete_contiguous_finite": True,
                    "metadata_consistent": True,
                    "official_test_isolated": True,
                    "split_hashes_recomputed_consistent": True,
                    "checkpoint_role_epoch_metrics_consistent": True,
                    "global_selection_keys_recomputed": True,
                    "state_dict_strict_load": True,
                    "fixed_threshold_object_metrics_exact": True,
                },
                "artifact_sha256": {
                    **common_hashes,
                    "checkpoint": _sha256(checkpoint),
                },
            },
            "fixed_threshold_0_5_checkpoint_audit": {
                "max_abs_non_strict_numeric_delta": 0.0,
            },
            "official_test_accessed": False,
            "validation_count": 133,
            "validation_split_sha256": "2" * 64,
            "split_seed": 20260722,
        }
        _write_json(run_dir / sweep_name, sweep)
    return run_dir


def _write_reference_sweeps(
    formal_root: Path,
    v2_root: Path,
    miou_root: Path,
    references: dict[str, dict[str, Any]],
) -> None:
    mappings = {
        "spd": (formal_root, miou_root, "spd", "formal800_pd_fp32_4x5090_v1"),
        "tpd_v1": (
            formal_root,
            miou_root,
            "tpd",
            "formal800_pd_fp32_4x5090_v1",
        ),
        "v2_sal_only": (
            v2_root,
            v2_root,
            "tpd_clean_sal",
            "screen800_pd_fp32_shared4x5090_v1",
        ),
        "v2_full": (
            v2_root,
            v2_root,
            "tpd_clean_full",
            "screen800_pd_fp32_shared4x5090_v1",
        ),
    }
    for name, (pd_root, m_root, variant, tag) in mappings.items():
        for role, root in (("pd_primary", pd_root), ("miou_primary", m_root)):
            run_dir = (
                root
                / "NUDT-SIRST"
                / variant
                / f"seed_42_{tag}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_name = (
                "best.pth.tar"
                if role == "pd_primary"
                else "best_miou.pth.tar"
            )
            sweep_name = (
                "pd_fa_sweep_best.pth.json"
                if role == "pd_primary"
                else "pd_fa_sweep_best_miou.pth.json"
            )
            checkpoint = run_dir / checkpoint_name
            checkpoint.write_bytes(f"{name}:{role}".encode())
            role_record = references[name]["roles"][role]
            points, budget_points = _consistent_sweep_points(
                role_record["fixed_threshold_0_5"],
                role_record["budgets"],
            )
            payload = {
                "variant": variant,
                "seed": 42,
                "dataset": "NUDT-SIRST",
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": role_record["checkpoint_epoch"],
                "checkpoint_role": (
                    "best_validation_pd_primary"
                    if role == "pd_primary"
                    else "best_validation_miou_secondary"
                ),
                "checkpoint_sha256": _sha256(checkpoint),
                "fixed_threshold_0_5": role_record["fixed_threshold_0_5"],
                "best_points_under_fa_budget": budget_points,
                "points": points,
                "audit": {
                    "expected_epochs": 800,
                    "metrics_event_count": 800,
                    "metrics_epoch_range": [1, 800],
                    "summary_status": "complete",
                    "selection_source": "internal_validation_only",
                    "integrity_checks_passed": {
                        "summary_complete": True,
                        "metrics_complete_contiguous_finite": True,
                        "metadata_consistent": True,
                        "official_test_isolated": True,
                        "split_hashes_recomputed_consistent": True,
                        "checkpoint_role_epoch_metrics_consistent": True,
                        "global_selection_keys_recomputed": True,
                        "state_dict_strict_load": True,
                        "fixed_threshold_object_metrics_exact": True,
                    },
                    "artifact_sha256": {
                        "checkpoint": _sha256(checkpoint),
                    },
                },
                "fixed_threshold_0_5_checkpoint_audit": {
                    "max_abs_non_strict_numeric_delta": 0.0
                },
                "official_test_accessed": False,
                "validation_count": 133,
                "validation_split_sha256": "2" * 64,
                "split_seed": 20260722,
            }
            _write_json(run_dir / sweep_name, payload)


class TPDV3SummaryTests(unittest.TestCase):
    def test_incomplete_run_writes_incomplete_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "comparison"
            exit_code = summary.main(
                [
                    "--candidate-root",
                    str(root / "candidate"),
                    "--formal-reference-root",
                    str(root / "formal"),
                    "--v2-reference-root",
                    str(root / "v2"),
                    "--reference-miou-root",
                    str(root / "miou"),
                    "--output-dir",
                    str(output),
                    "--overwrite",
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(
                (output / summary.JSON_OUTPUT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "incomplete")
            self.assertIsNone(payload["engineering_gate_passed"])
            self.assertFalse(payload["gate_evaluated"])
            self.assertEqual(payload["candidate_runs"], {})
            self.assertNotIn("methods", payload)
            self.assertGreaterEqual(len(payload["incomplete_reasons"]), 4)
            markdown = (output / summary.MARKDOWN_OUTPUT_NAME).read_text(
                encoding="utf-8"
            )
            self.assertIn("INCOMPLETE", markdown)
            self.assertIn("未计算", markdown)

    def test_complete_artifacts_evaluate_all_seven_gates(self) -> None:
        runs, references = _passing_gate_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            formal = root / "formal"
            v2 = root / "v2"
            miou = root / "miou"
            output = root / "comparison"
            for (variant, seed), record in runs.items():
                _write_candidate_run(candidate, variant, seed, record)
            _write_reference_sweeps(formal, v2, miou, references)
            self.assertEqual(
                summary.main(
                    [
                        "--candidate-root",
                        str(candidate),
                        "--formal-reference-root",
                        str(formal),
                        "--v2-reference-root",
                        str(v2),
                        "--reference-miou-root",
                        str(miou),
                        "--output-dir",
                        str(output),
                        "--overwrite",
                    ]
                ),
                0,
            )
            payload = json.loads(
                (output / summary.JSON_OUTPUT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(payload["gate_evaluated"])
            self.assertTrue(payload["engineering_gate_passed"])
            self.assertEqual(len(payload["candidate_runs"]), 4)
            self.assertEqual(
                set(payload["frozen_references"]),
                {"spd", "tpd_v1", "v2_sal_only", "v2_full"},
            )
            checks = payload["engineering_gate"]["checks"]
            self.assertEqual(len(checks), 7)
            self.assertTrue(all(check["passed"] for check in checks.values()))
            self.assertFalse(payload["mainline_changed"])
            self.assertFalse(payload["paper_core_established"])
            self.assertFalse(payload["stability_claim_supported"])
            self.assertEqual(
                payload["validation"]["candidate_metrics_event_count"], 3200
            )
            paired_shared = payload["validation"][
                "paired_shared_non_shallow_initialization_sha256"
            ]
            self.assertEqual(set(paired_shared), {"42", "3407"})
            self.assertTrue(
                all(item["equal"] is True for item in paired_shared.values())
            )
            self.assertNotEqual(
                paired_shared["42"]["sha256"],
                paired_shared["3407"]["sha256"],
            )
            markdown = (output / summary.MARKDOWN_OUTPUT_NAME).read_text(
                encoding="utf-8"
            )
            self.assertIn("PASS", markdown)
            self.assertIn("mainline_changed=false", markdown)

    def test_unused_reference_null_is_disclosed_but_required_null_is_incomplete(
        self,
    ) -> None:
        runs, references = _passing_gate_inputs()
        references["v2_full"]["roles"]["miou_primary"]["budgets"][
            "1e-06"
        ] = None
        self.assertTrue(
            summary.evaluate_engineering_gate(runs, references)["passed"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            formal = root / "formal"
            v2 = root / "v2"
            miou = root / "miou"
            output = root / "comparison"
            for (variant, seed), record in runs.items():
                _write_candidate_run(candidate, variant, seed, record)
            _write_reference_sweeps(formal, v2, miou, references)

            self.assertEqual(
                summary.main(
                    [
                        "--candidate-root",
                        str(candidate),
                        "--formal-reference-root",
                        str(formal),
                        "--v2-reference-root",
                        str(v2),
                        "--reference-miou-root",
                        str(miou),
                        "--output-dir",
                        str(output),
                        "--overwrite",
                    ]
                ),
                0,
            )
            payload = json.loads(
                (output / summary.JSON_OUTPUT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(
                payload["reference_unavailable_points"],
                [
                    {
                        "method": "v2_full",
                        "role": "miou_primary",
                        "budget": "1e-06",
                        "used_by_gates": False,
                        "gate_usage": "not_used_by_gates",
                    }
                ],
            )

            spd_run_dir = summary._reference_paths(formal, v2, miou)["spd"][
                "pd_primary"
            ][0]
            spd_sweep_path = (
                spd_run_dir / summary.ROLE_SPECS["pd_primary"]["sweep"]
            )
            spd_sweep = json.loads(spd_sweep_path.read_text(encoding="utf-8"))
            for point in spd_sweep["points"]:
                if float(point["fa"]) <= 1e-6:
                    point["fa"] = 2e-6
            spd_sweep["fixed_threshold_0_5"] = copy.deepcopy(
                next(
                    point
                    for point in spd_sweep["points"]
                    if float(point["threshold"]) == 0.5
                )
            )
            spd_sweep["best_points_under_fa_budget"] = {
                budget: copy.deepcopy(
                    summary._best_sweep_point_under_fa(
                        spd_sweep["points"],
                        float(budget),
                    )
                )
                for budget in BUDGETS
            }
            self.assertIsNone(
                spd_sweep["best_points_under_fa_budget"]["1e-06"]
            )
            _write_json(spd_sweep_path, spd_sweep)

            required_output = root / "required-null-comparison"
            self.assertEqual(
                summary.main(
                    [
                        "--candidate-root",
                        str(candidate),
                        "--formal-reference-root",
                        str(formal),
                        "--v2-reference-root",
                        str(v2),
                        "--reference-miou-root",
                        str(miou),
                        "--output-dir",
                        str(required_output),
                        "--overwrite",
                    ]
                ),
                0,
            )
            required_payload = json.loads(
                (
                    required_output / summary.JSON_OUTPUT_NAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(required_payload["status"], "incomplete")
            self.assertFalse(required_payload["gate_evaluated"])
            self.assertIsNone(required_payload["engineering_gate_passed"])
            self.assertEqual(len(required_payload["candidate_runs"]), 4)
            self.assertTrue(
                any(
                    point["method"] == "spd"
                    and point["role"] == "pd_primary"
                    and point["budget"] == "1e-06"
                    and point["used_by_gates"] is True
                    for point in required_payload[
                        "reference_unavailable_points"
                    ]
                )
            )
            self.assertTrue(
                any(
                    "required by the engineering gate" in reason
                    for reason in required_payload["incomplete_reasons"]
                )
            )

    def test_every_protocol_gate_has_an_independent_failure_path(self) -> None:
        base_runs, references = _passing_gate_inputs()
        baseline = summary.evaluate_engineering_gate(base_runs, references)
        self.assertTrue(baseline["passed"])
        self.assertEqual(len(baseline["checks"]), 7)

        cases = []

        runs = copy.deepcopy(base_runs)
        runs[("tpd_clean_v3_full", 42)]["roles"]["pd_primary"][
            "fixed_threshold_0_5"
        ]["matched_target_count"] = 187
        runs[("tpd_clean_v3_full", 42)]["roles"]["pd_primary"][
            "fixed_threshold_0_5"
        ]["pd"] = 187 / 189
        cases.append(("gate_1_seed42_pd_primary_fixed", runs, references))

        runs = copy.deepcopy(base_runs)
        runs[("tpd_clean_v3_full", 42)]["roles"]["miou_primary"][
            "fixed_threshold_0_5"
        ]["miou"] = 0.94
        cases.append(("gate_2_seed42_miou_primary_fixed", runs, references))

        runs = copy.deepcopy(base_runs)
        runs[("tpd_clean_v3_full", 42)]["roles"]["pd_primary"]["budgets"][
            "1e-06"
        ]["matched_target_count"] = 186
        runs[("tpd_clean_v3_full", 42)]["roles"]["pd_primary"]["budgets"][
            "1e-06"
        ]["pd"] = 186 / 189
        cases.append(("gate_3_seed42_budget_floors", runs, references))

        refs = copy.deepcopy(references)
        for point in refs["spd"]["roles"]["pd_primary"]["budgets"].values():
            point.update(_point(189, 0.0, 0.99))
        refs["v2_sal_only"]["roles"]["pd_primary"]["budgets"][
            "5e-06"
        ].update(_point(189, 0.0, 0.99))
        cases.append(("gate_4_seed42_frozen_references", base_runs, refs))

        runs = copy.deepcopy(base_runs)
        control_point = runs[("tpd_clean_v3_sal_capacity", 3407)]["roles"][
            "miou_primary"
        ]["fixed_threshold_0_5"]
        control_point.update(_point(189, 0.0, 0.99))
        cases.append(("gate_5_no_capacity_dominance", runs, references))

        runs = copy.deepcopy(base_runs)
        for seed in SEEDS:
            full = runs[("tpd_clean_v3_full", seed)]
            control = runs[("tpd_clean_v3_sal_capacity", seed)]
            full["roles"]["miou_primary"]["fixed_threshold_0_5"]["miou"] = (
                control["roles"]["miou_primary"]["fixed_threshold_0_5"]["miou"]
            )
            for budget in BUDGETS:
                full["roles"]["pd_primary"]["budgets"][budget] = copy.deepcopy(
                    control["roles"]["pd_primary"]["budgets"][budget]
                )
            for budget in ("1e-05", "5e-05", "0.0001"):
                point = full["roles"]["pd_primary"]["budgets"][budget]
                point["matched_target_count"] -= 2
                point["pd"] = point["matched_target_count"] / 189
        cases.append(("gate_6_paired_advantage_and_wide_pd", runs, references))

        runs = copy.deepcopy(base_runs)
        for seed in SEEDS:
            control = runs[("tpd_clean_v3_sal_capacity", seed)]
            for role in ("pd_primary", "miou_primary"):
                for budget in BUDGETS:
                    control["roles"][role]["budgets"][budget].update(
                        _point(189, 0.0, 0.99)
                    )
        cases.append(("gate_7_fixed_sweep_direction_coherence", runs, references))

        for expected_key, run_input, reference_input in cases:
            with self.subTest(gate=expected_key):
                result = summary.evaluate_engineering_gate(
                    run_input, reference_input
                )
                self.assertFalse(result["checks"][expected_key]["passed"])

    def test_metrics_truncation_is_incomplete_not_a_gate_failure(self) -> None:
        runs, references = _passing_gate_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            formal = root / "formal"
            v2 = root / "v2"
            miou = root / "miou"
            output = root / "comparison"
            run_dirs = {
                (variant, seed): _write_candidate_run(
                    candidate, variant, seed, record
                )
                for (variant, seed), record in runs.items()
            }
            _write_reference_sweeps(formal, v2, miou, references)
            truncated = run_dirs[("tpd_clean_v3_full", 42)] / "metrics.jsonl"
            lines = truncated.read_text(encoding="utf-8").splitlines()
            truncated.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            self.assertEqual(
                summary.main(
                    [
                        "--candidate-root",
                        str(candidate),
                        "--formal-reference-root",
                        str(formal),
                        "--v2-reference-root",
                        str(v2),
                        "--reference-miou-root",
                        str(miou),
                        "--output-dir",
                        str(output),
                        "--overwrite",
                    ]
                ),
                0,
            )
            payload = json.loads(
                (output / summary.JSON_OUTPUT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "incomplete")
            self.assertFalse(payload["gate_evaluated"])
            self.assertIsNone(payload["engineering_gate_passed"])
            self.assertTrue(
                any(
                    "metrics.jsonl" in reason and "800" in reason
                    for reason in payload["incomplete_reasons"]
                )
            )

    def test_budget_point_must_be_recomputed_from_sweep_points(self) -> None:
        runs, references = _passing_gate_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate"
            formal = root / "formal"
            v2 = root / "v2"
            miou = root / "miou"
            output = root / "comparison"
            run_dirs = {
                (variant, seed): _write_candidate_run(
                    candidate, variant, seed, record
                )
                for (variant, seed), record in runs.items()
            }
            _write_reference_sweeps(formal, v2, miou, references)
            sweep_path = (
                run_dirs[("tpd_clean_v3_full", 42)]
                / "pd_fa_sweep_best.pth.json"
            )
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
            sweep["best_points_under_fa_budget"]["1e-06"] = {
                **_point(189, 0.0, 0.99),
                "threshold": 0.123,
            }
            _write_json(sweep_path, sweep)

            self.assertEqual(
                summary.main(
                    [
                        "--candidate-root",
                        str(candidate),
                        "--formal-reference-root",
                        str(formal),
                        "--v2-reference-root",
                        str(v2),
                        "--reference-miou-root",
                        str(miou),
                        "--output-dir",
                        str(output),
                        "--overwrite",
                    ]
                ),
                0,
            )
            payload = json.loads(
                (output / summary.JSON_OUTPUT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "incomplete")
            self.assertFalse(payload["gate_evaluated"])
            self.assertTrue(
                any(
                    "not the exact optimum recomputed from sweep points"
                    in reason
                    for reason in payload["incomplete_reasons"]
                )
            )

    def test_nonfinite_or_missing_budget_is_never_defaulted(self) -> None:
        runs, references = _passing_gate_inputs()
        broken = copy.deepcopy(runs)
        broken[("tpd_clean_v3_full", 42)]["roles"]["pd_primary"][
            "fixed_threshold_0_5"
        ]["fa"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            summary.evaluate_engineering_gate(broken, references)

        broken = copy.deepcopy(runs)
        del broken[("tpd_clean_v3_full", 42)]["roles"]["pd_primary"][
            "budgets"
        ]["1e-06"]
        with self.assertRaisesRegex(ValueError, "1e-06"):
            summary.evaluate_engineering_gate(broken, references)

        broken = copy.deepcopy(runs)
        broken[("tpd_clean_v3_full", 42)]["roles"]["pd_primary"][
            "budgets"
        ]["1e-06"] = None
        with self.assertRaisesRegex(ValueError, "point must be an object"):
            summary.evaluate_engineering_gate(broken, references)


if __name__ == "__main__":
    unittest.main()
