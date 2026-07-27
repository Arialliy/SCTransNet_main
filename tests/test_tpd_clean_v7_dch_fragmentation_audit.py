from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from analysis import diagnose_tpd_clean_v7_dch_mechanism as audit
from experiments import finalize_tpd_clean_v7_dch as finalizer


def validation_metrics(offset: float = 0.0) -> dict[str, float | int]:
    return {
        "val_loss": 0.1 + offset,
        "miou": 0.9 + offset,
        "niou": 0.8 + offset,
        "pixel_precision": 0.95,
        "pixel_recall": 0.94,
        "pixel_f1": 0.945,
        "pd": 188 / 189,
        "tiny_pd": 1.0,
        "fa": 1e-6,
        "false_objects_per_image": 1 / 133,
        "target_count": 189,
        "matched_target_count": 188,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 39,
        "predicted_object_count": 189,
        "unmatched_predicted_object_count": 1,
        "valid_pixel_count": 8_716_288,
    }


def topology_point(
    *,
    threshold: float,
    fragment_excess: int,
    in_gt_pixels: int,
    split_targets: int,
    largest: float,
) -> dict[str, object]:
    return {
        **validation_metrics(),
        "threshold": threshold,
        "component_taxonomy": {
            "unmatched_pixels_in_gt": in_gt_pixels,
            "fragment_fa_fraction": 0.75,
        },
        "gt_topology": {
            "fragment_excess_total": fragment_excess,
            "split_target_count": split_targets,
            "largest_fragment_fraction_mean": largest,
            "largest_fragment_fraction_p10": largest - 0.05,
        },
    }


def reference_payload(role: str) -> dict[str, object]:
    fixed_05 = topology_point(
        threshold=0.5,
        fragment_excess=2,
        in_gt_pixels=20,
        split_targets=2,
        largest=0.7,
    )
    fixed_058 = topology_point(
        threshold=0.58,
        fragment_excess=2,
        in_gt_pixels=18,
        split_targets=2,
        largest=0.72,
    )
    fixed_0999 = topology_point(
        threshold=0.999,
        fragment_excess=3,
        in_gt_pixels=12,
        split_targets=3,
        largest=0.75,
    )
    matched = topology_point(
        threshold=0.8,
        fragment_excess=2,
        in_gt_pixels=15,
        split_targets=2,
        largest=0.74,
    )
    return {
        "schema": audit.REFERENCE_SCHEMA,
        "variant": "tpd_clean_v6_full",
        "seed": 3407,
        "checkpoint_role": role,
        "official_test_accessed": False,
        "training_performed": False,
        "complete_validation_split": True,
        "model_metadata": {
            "candidate_family": audit.V6_REFERENCE_FAMILY,
        },
        "modes": {
            "as_trained": {
                "fixed_threshold_points": {
                    "0.5": fixed_05,
                    "0.58": fixed_058,
                    "0.999": fixed_0999,
                },
                "best_points_under_fa_budget": {
                    "1e-06": matched,
                    "5e-06": copy.deepcopy(matched),
                    "1e-05": copy.deepcopy(matched),
                    # This duplicates a frozen numeric threshold and must
                    # merge labels rather than weight it twice.
                    "5e-05": copy.deepcopy(fixed_058),
                    "0.0001": copy.deepcopy(matched),
                },
            }
        },
    }


def reference_registries() -> dict[str, object]:
    return {
        role: audit.build_reference_registry(
            reference_payload(role),
            role=role,
            fixed_thresholds=audit.DEFAULT_FIXED_THRESHOLDS,
            fa_budgets=audit.DEFAULT_FA_BUDGETS,
        )
        for role in ("pd_primary", "miou_primary")
    }


def evaluated_point(
    registry: dict[str, object],
    *,
    fragment_excess: int,
    in_gt_pixels: int,
    split_targets: int,
    largest: float,
) -> dict[str, object]:
    return {
        "registry_labels": list(registry["registry_labels"]),
        "registry_kinds": list(registry["registry_kinds"]),
        "matched_operating_point": bool(
            registry["matched_operating_point"]
        ),
        "threshold": float(registry["threshold"]),
        "validation_metrics": validation_metrics(),
        "audit_measures": {
            "fragment_excess_total": fragment_excess,
            "unmatched_pixels_in_gt": in_gt_pixels,
            "split_target_count": split_targets,
            "fragment_fa_fraction": 0.5,
            "largest_fragment_fraction_mean": largest,
            "largest_fragment_fraction_p10": largest - 0.05,
        },
    }


def payload_matrix(
    registries: dict[str, object],
    checkpoint_paths: dict[tuple[str, int, str], Path] | None = None,
) -> dict[tuple[str, int, str], dict[str, object]]:
    payloads: dict[tuple[str, int, str], dict[str, object]] = {}
    last_registry = audit.thresholds_for_role(registries, "last")
    for variant in audit.VARIANTS:
        for seed in audit.SEEDS:
            for role in audit.CHECKPOINT_SPECS:
                registry = (
                    registries[role]["points"]
                    if role in ("pd_primary", "miou_primary")
                    else last_registry
                )
                points = {}
                for index, (key, record) in enumerate(registry.items()):
                    if variant == audit.PRIMARY_VARIANT and seed == 3407:
                        fragment_excess = 1
                        in_gt_pixels = 10
                        split_targets = 1
                        largest = 0.8
                    elif variant == audit.CAPACITY_VARIANT and index == 0:
                        # Capacity is worse at one point, so it does not
                        # comprehensively cover Full under M4.
                        fragment_excess = 4
                        in_gt_pixels = 25
                        split_targets = 4
                        largest = 0.6
                    else:
                        fragment_excess = 2
                        in_gt_pixels = 16
                        split_targets = 2
                        largest = 0.72
                    points[key] = evaluated_point(
                        record,
                        fragment_excess=fragment_excess,
                        in_gt_pixels=in_gt_pixels,
                        split_targets=split_targets,
                        largest=largest,
                    )
                checkpoint_path = (
                    checkpoint_paths[(variant, seed, role)]
                    if checkpoint_paths is not None
                    else Path(f"/tmp/{variant}-{seed}-{role}.pth.tar")
                )
                checkpoint_sha256 = (
                    hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                    if checkpoint_paths is not None
                    else "a" * 64
                )
                payloads[(variant, seed, role)] = {
                    "schema": audit.CHECKPOINT_SCHEMA,
                    "checkpoint": str(checkpoint_path),
                    "source_identity": {
                        "checkpoint_role": audit.CHECKPOINT_SPECS[role][
                            "checkpoint_role"
                        ],
                        "checkpoint_epoch": (
                            800 if role == "last" else 700
                        ),
                        "run_id": (
                            f"tpd-clean-v7-dch-exact:{audit.DATASET}:"
                            f"{variant}:seed-{seed}:formal"
                        ),
                    },
                    "input_sha256_before": {
                        "checkpoint": checkpoint_sha256,
                    },
                    "operating_points": points,
                }
    return payloads


class TPDCleanV7DCHFragmentationAuditTests(unittest.TestCase):
    def test_matrix_contract_is_four_runs_times_three_checkpoints(self) -> None:
        jobs = audit.expected_jobs(Path("/candidate"))
        self.assertEqual(len(jobs), 12)
        self.assertEqual(
            {
                (job["variant"], job["seed"], job["role"])
                for job in jobs
            },
            {
                (variant, seed, role)
                for variant in audit.VARIANTS
                for seed in audit.SEEDS
                for role in audit.CHECKPOINT_SPECS
            },
        )

    def test_no_results_contract_is_read_only_and_has_no_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.rglob("*"))
            contract = audit.inventory_contract(root)
            after = list(root.rglob("*"))
        self.assertEqual(contract["status"], "NO_RESULTS_AVAILABLE")
        self.assertEqual(contract["ready_job_count"], 0)
        self.assertIsNone(
            contract["fragmentation_mechanism_claim_supported"]
        )
        self.assertIsNone(contract["performance_results"])
        self.assertFalse(contract["formal_gate_replacement"])
        self.assertEqual(before, after)

    def test_run_does_not_create_placeholder_when_results_are_absent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = audit.parse_args(
                [
                    "--run",
                    "--results-root",
                    str(root / "candidate"),
                    "--output-dir",
                    str(root / "comparison"),
                ]
            )
            report = audit.run(args)
            self.assertEqual(report["status"], "NO_RESULTS_AVAILABLE")
            self.assertFalse((root / "comparison").exists())

    def test_native_validation_schema_requires_all_seventeen_fields(
        self,
    ) -> None:
        metrics = validation_metrics()
        accepted = audit.require_validation_metrics(metrics, "fixture")
        self.assertEqual(len(accepted), 17)
        broken = dict(metrics)
        del broken["fa"]
        with self.assertRaisesRegex(ValueError, "17-field"):
            audit.require_validation_metrics(broken, "fixture")

    def test_component_audit_reports_literal_fragmentation(self) -> None:
        target = np.zeros((9, 9), dtype=np.float32)
        target[2:7, 2:7] = 1.0
        probability = np.zeros_like(target)
        probability[2:7, 2:4] = 1.0
        probability[2:7, 5:7] = 1.0
        result = audit.v6_diag.component_diagnostics(
            probability,
            target,
            threshold=0.5,
            match_radius=3.0,
            dilation_radius=3,
        )
        self.assertEqual(
            result["unmatched_component_count_by_class"][
                "in_gt_fragment"
            ],
            1,
        )
        self.assertGreater(result["unmatched_pixels_in_gt"], 0)
        aggregate = audit.v6_diag.aggregate_component_diagnostics(
            [probability],
            [target],
            ["image"],
            threshold=0.5,
            match_radius=3.0,
            dilation_radius=3,
        )
        self.assertEqual(aggregate["fragment_excess_total"], 1)
        self.assertEqual(aggregate["split_target_count"], 1)
        self.assertAlmostEqual(
            aggregate["largest_fragment_fraction_mean"],
            0.5,
        )

    def test_reference_registry_deduplicates_numeric_thresholds(self) -> None:
        registry = audit.build_reference_registry(
            reference_payload("pd_primary"),
            role="pd_primary",
            fixed_thresholds=audit.DEFAULT_FIXED_THRESHOLDS,
            fa_budgets=audit.DEFAULT_FA_BUDGETS,
        )
        # 0.5, 0.58, 0.8 and 0.999: five budgets do not create
        # repeated observations at the same numeric threshold.
        self.assertEqual(len(registry["points"]), 4)
        duplicate = registry["points"]["0.58"]
        self.assertIn("fixed_threshold", duplicate["registry_kinds"])
        self.assertIn(
            "v6_reference_fa_budget", duplicate["registry_kinds"]
        )
        self.assertTrue(duplicate["matched_operating_point"])

    def test_reference_identity_is_independent_from_dch_identity(self) -> None:
        payload = reference_payload("pd_primary")
        payload["variant"] = audit.PRIMARY_VARIANT
        with self.assertRaisesRegex(ValueError, "reference identity"):
            audit.build_reference_registry(
                payload,
                role="pd_primary",
                fixed_thresholds=audit.DEFAULT_FIXED_THRESHOLDS,
                fa_budgets=audit.DEFAULT_FA_BUDGETS,
            )

    def test_M1_to_M4_complete_without_replacing_performance_gates(
        self,
    ) -> None:
        registries = reference_registries()
        result = audit.build_mechanism_audit(
            payload_matrix(registries), registries
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertTrue(result["M1"]["pass"])
        self.assertTrue(result["M2"]["pass"])
        self.assertTrue(result["M3"]["pass"])
        self.assertTrue(result["M4"]["pass"])
        self.assertTrue(result["mechanism_audit_M_pass"])
        self.assertTrue(
            result["fragmentation_mechanism_claim_supported"]
        )
        self.assertFalse(result["formal_gate_replacement"])
        self.assertFalse(result["gate_A_E_recomputed"])
        self.assertFalse(result["ner_authorization_decided"])

    def test_capacity_comprehensive_coverage_fails_M4(self) -> None:
        registries = reference_registries()
        payloads = payload_matrix(registries)
        for (variant, _, _), payload in payloads.items():
            if variant != audit.CAPACITY_VARIANT:
                continue
            for point in payload["operating_points"].values():
                point["audit_measures"]["fragment_excess_total"] = 0
        result = audit.build_mechanism_audit(payloads, registries)
        self.assertFalse(result["M4"]["pass"])
        self.assertTrue(result["M4"]["capacity_fully_covers_full"])
        self.assertFalse(result["mechanism_audit_M_pass"])

    def test_finalizer_report_binds_paths_hashes_roles_and_validates(
        self,
    ) -> None:
        registries = reference_registries()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for variant in audit.VARIANTS:
                for seed in audit.SEEDS:
                    for role in audit.CHECKPOINT_SPECS:
                        path = root / f"{variant}-{seed}-{role}.pth.tar"
                        path.write_bytes(
                            f"{variant}:{seed}:{role}".encode("utf-8")
                        )
                        paths[(variant, seed, role)] = path
            report = audit.build_mechanism_report(
                payload_matrix(registries, paths),
                registries,
                candidate_root=root,
                reference_input_sha256={
                    "pd_primary": "b" * 64,
                    "miou_primary": "c" * 64,
                },
            )
            report_path = root / "tpd_clean_v7_dch_mechanism_audit.json"
            report_path.write_text(
                json.dumps(report, sort_keys=True),
                encoding="utf-8",
            )
            validated = audit.validate_mechanism_report(report_path)
            finalizer_validated = finalizer._validate_mechanism(validated)

        self.assertEqual(report["schema"], audit.SCHEMA)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["artifact_counts"],
            {"runs": 4, "checkpoints": 12},
        )
        self.assertEqual(len(report["input_checkpoints"]), 12)
        self.assertIsInstance(
            report["fragmentation_mechanism_claim_supported"], bool
        )
        self.assertFalse(
            report["mechanism_audit_replaces_performance_gates"]
        )
        self.assertEqual(
            finalizer_validated["candidate_family"],
            "tpd_clean_v7_dch",
        )

    def test_mechanism_report_rejects_changed_checkpoint_input(self) -> None:
        registries = reference_registries()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for variant in audit.VARIANTS:
                for seed in audit.SEEDS:
                    for role in audit.CHECKPOINT_SPECS:
                        path = root / f"{variant}-{seed}-{role}.pth.tar"
                        path.write_bytes(b"checkpoint")
                        paths[(variant, seed, role)] = path
            report = audit.build_mechanism_report(
                payload_matrix(registries, paths),
                registries,
                candidate_root=root,
            )
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            paths[(audit.PRIMARY_VARIANT, 42, "pd_primary")].write_bytes(
                b"changed"
            )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                audit.validate_mechanism_report(report_path)

    def test_formal_input_hash_change_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            path.write_bytes(b"before")
            before = {"checkpoint": audit.file_sha256(path)}
            path.write_bytes(b"after")
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                audit.verify_inputs_unchanged(
                    {"checkpoint": path}, before
                )

    def test_DCH_artifact_identity_and_native_metrics_are_validated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_identity = {
                "candidate_family": audit.CANDIDATE_FAMILY,
                "variant": audit.PRIMARY_VARIANT,
                "mainline_contract": "Keep-Context-Saliency",
            }
            run_identity = {
                "run_id": (
                    f"tpd-clean-v7-dch-exact:{audit.DATASET}:"
                    f"{audit.PRIMARY_VARIANT}:seed-42:test"
                ),
                "dataset": audit.DATASET,
                "variant": audit.PRIMARY_VARIANT,
                "seed": 42,
                "architecture_id": "architecture",
                "source_locks": {
                    "tpd_clean_v7_dch_exact_source_lock": "a" * 64,
                },
            }
            protocol = {
                "schema": audit.DCH_ENTRY_SCHEMA,
                "arguments": {
                    "dataset": audit.DATASET,
                    "variant": audit.PRIMARY_VARIANT,
                    "seed": 42,
                    "epochs": 2,
                },
                "stored_validation_metrics": list(
                    audit.STORED_VALIDATION_METRICS
                ),
                "run_identity": run_identity,
                "model": model_identity,
                "normalization": {"mean": 0.0, "std": 1.0},
                "official_test_accessed": False,
            }
            split = {
                "used_val_ids": ["one"],
                "used_val_count": 1,
                "hashes": {"used_val_sha256": "split"},
                "official_test_accessed": False,
            }
            summary = {
                "schema": audit.SUMMARY_SCHEMA,
                "status": "complete",
                "dataset": audit.DATASET,
                "variant": audit.PRIMARY_VARIANT,
                "seed": 42,
                "stored_validation_metrics": list(
                    audit.STORED_VALIDATION_METRICS
                ),
                "best_pd_epoch": 1,
                "best_pd_validation_metrics": validation_metrics(),
                "best_miou_epoch": 2,
                "best_miou_validation_metrics": validation_metrics(),
                "model": model_identity,
                "official_test_accessed": False,
            }
            checkpoint = {
                "dataset": audit.DATASET,
                "variant": audit.PRIMARY_VARIANT,
                "seed": 42,
                "epoch": 1,
                "checkpoint_role": "best_validation_pd_primary",
                "validation_metrics": validation_metrics(),
                "run_identity": run_identity,
                "model_metadata": model_identity,
                "official_test_accessed": False,
            }
            (run_dir / "protocol.json").write_text(
                json.dumps(protocol), encoding="utf-8"
            )
            (run_dir / "split.json").write_text(
                json.dumps(split), encoding="utf-8"
            )
            (run_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            events = []
            for epoch in (1, 2):
                events.append(
                    json.dumps(
                        {"epoch": epoch, **validation_metrics()},
                        sort_keys=True,
                    )
                )
            (run_dir / "metrics.jsonl").write_text(
                "\n".join(events) + "\n", encoding="utf-8"
            )
            torch.save(checkpoint, run_dir / "best.pth.tar")
            job = {
                "variant": audit.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_dir": run_dir,
                "checkpoint": run_dir / "best.pth.tar",
            }
            with mock.patch.object(audit, "EXPECTED_EPOCHS", 2), mock.patch.object(
                audit, "EXPECTED_VAL_COUNT", 1
            ):
                artifacts = audit.validate_job_artifacts(job)
            self.assertEqual(
                artifacts["source_identity"]["candidate_family"],
                audit.CANDIDATE_FAMILY,
            )
            self.assertEqual(
                len(artifacts["checkpoint_metrics"]), 17
            )
            self.assertEqual(
                len(artifacts["input_sha256"]), 5
            )

            protocol["schema"] = "v6_identity"
            (run_dir / "protocol.json").write_text(
                json.dumps(protocol), encoding="utf-8"
            )
            with mock.patch.object(audit, "EXPECTED_EPOCHS", 2), mock.patch.object(
                audit, "EXPECTED_VAL_COUNT", 1
            ):
                with self.assertRaisesRegex(ValueError, "V7-DCH"):
                    audit.validate_job_artifacts(job)

    def test_readiness_api_exposes_default_finalizer_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = audit.inspect_mechanism_readiness(Path(directory))
        self.assertFalse(report["ready"])
        self.assertEqual(report["writes_performed"], 0)
        self.assertTrue(
            report["output_path"].endswith(
                "NUDT-SIRST/comparison/"
                "tpd_clean_v7_dch_mechanism_audit.json"
            )
        )
        self.assertEqual(
            audit.DEFAULT_OUTPUT_PATH.name,
            "tpd_clean_v7_dch_mechanism_audit.json",
        )


if __name__ == "__main__":
    unittest.main()
