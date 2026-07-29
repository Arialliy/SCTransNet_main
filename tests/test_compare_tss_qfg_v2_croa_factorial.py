from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments import compare_tss_qfg_v2_croa_factorial as factorial


PARENT_BINDING = {
    "parent_checkpoint_sha256": "1" * 64,
    "parent_checkpoint_state_dict_sha256": "2" * 64,
    "parent_checkpoint_role": "best_miou",
    "parent_checkpoint_epoch": 489,
}


def point(
    *,
    matched: int,
    unmatched: int,
    miou: float,
    fa: float = 0.0,
    threshold: float = 0.5,
) -> dict:
    return {
        "threshold": threshold,
        "matched_target_count": matched,
        "pd": matched / factorial.TARGET_COUNT,
        "fa": fa,
        "miou": miou,
        "tiny_pd": 1.0,
        "unmatched_predicted_object_count": unmatched,
        "false_objects_per_image": (
            unmatched / factorial.VALIDATION_COUNT
        ),
        "target_count": factorial.TARGET_COUNT,
        "tiny_target_count": factorial.TINY_TARGET_COUNT,
        "matched_tiny_target_count": factorial.TINY_TARGET_COUNT,
        "predicted_object_count": matched + unmatched,
        "valid_pixel_count": (
            factorial.VALIDATION_COUNT * 256 * 256
        ),
    }


def synthetic_records(root: Path) -> dict[tuple[str, str], factorial.SweepRecord]:
    matched = {"A": 180, "B": 181, "C": 182, "D": 185}
    unmatched = {"A": 4, "B": 3, "C": 2, "D": 1}
    records = {}
    for arm, spec in factorial.ARM_SPECS.items():
        run_dir = root / spec.variant / f"run_{arm}"
        for role_index, (checkpoint, role) in enumerate(
            factorial.CHECKPOINT_ROLES.items()
        ):
            fixed = point(
                matched=matched[arm],
                unmatched=unmatched[arm],
                miou=0.80 + 0.01 * "ABCD".index(arm) + 0.05 * role_index,
            )
            budgets = {
                key: point(
                    matched=matched[arm],
                    unmatched=unmatched[arm],
                    miou=fixed["miou"] + 0.001 * index,
                    fa=0.0,
                    threshold=0.60 + 0.01 * index,
                )
                for index, key in enumerate(factorial.BUDGET_KEYS)
            }
            records[(arm, checkpoint)] = factorial.SweepRecord(
                arm=arm,
                variant=spec.variant,
                evaluator_family=spec.evaluator_family,
                run_directory=run_dir,
                checkpoint_filename=checkpoint,
                checkpoint_role=role,
                checkpoint_epoch=10 + role_index,
                checkpoint_sha256=hashlib.sha256(
                    f"{arm}:{checkpoint}".encode()
                ).hexdigest(),
                sweep_path=run_dir / factorial._sweep_filename(checkpoint),
                sweep_sha256=hashlib.sha256(
                    f"sweep:{arm}:{checkpoint}".encode()
                ).hexdigest(),
                validation_split_sha256="3" * 64,
                run_identity={
                    "run_id": f"run-{arm}",
                    "variant": spec.variant,
                },
                checkpoint_identity={
                    "run_id": f"run-{arm}",
                    "variant": spec.variant,
                    **PARENT_BINDING,
                },
                parent_binding=dict(PARENT_BINDING),
                fixed=fixed,
                budgets=budgets,
            )
    return records


class FakeEvaluators:
    def __init__(self, audits: dict[tuple[str, str], dict]) -> None:
        self.audits = audits
        self.output_validation_count = 0

    def module(self):
        def validate_run_artifacts(run_dir: Path, checkpoint: str):
            return self.audits[(str(Path(run_dir).resolve()), checkpoint)]

        def validate_output_identity(payload, *, artifact_audit):
            self.output_validation_count += 1
            if payload["variant"] != artifact_audit["variant"]:
                raise ValueError("fixture variant differs")

        return SimpleNamespace(
            validate_run_artifacts=validate_run_artifacts,
            validate_output_identity=validate_output_identity,
        )


def fixture_tree(root: Path):
    run_dirs = {}
    audits = {}
    for arm, spec in factorial.ARM_SPECS.items():
        run_dir = (
            root
            / factorial.DATASET
            / spec.variant
            / f"seed_42_fixture_{arm}"
        ).resolve()
        run_dir.mkdir(parents=True)
        run_dirs[arm] = run_dir
        summary = {
            "status": "complete",
            "variant": spec.variant,
            "dataset": factorial.DATASET,
            "seed": factorial.TRAINING_SEED,
            "split_seed": factorial.SPLIT_SEED,
            "official_test_accessed": False,
            "best_pd_epoch": 10,
            "best_miou_epoch": 20,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        run_identity = {
            "run_id": f"fixture-{arm}",
            "variant": spec.variant,
        }
        checkpoint_identity = {
            "run_id": f"fixture-{arm}",
            "variant": spec.variant,
            **PARENT_BINDING,
        }
        for checkpoint, role in factorial.CHECKPOINT_ROLES.items():
            epoch = 10 if checkpoint == "best.pth.tar" else 20
            checkpoint_path = run_dir / checkpoint
            checkpoint_path.write_bytes(f"{arm}:{checkpoint}".encode())
            checkpoint_sha = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            audit = {
                "run_directory": str(run_dir),
                "variant": spec.variant,
                "checkpoint_filename": checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_epoch": epoch,
                "checkpoint_role": role,
                "checkpoint_identity": checkpoint_identity,
                "run_identity": run_identity,
            }
            audits[(str(run_dir), checkpoint)] = audit
            fixed = point(
                matched=180 + "ABCD".index(arm),
                unmatched=3,
                miou=0.8,
            )
            budgets = {
                key: point(
                    matched=180 + "ABCD".index(arm),
                    unmatched=3,
                    miou=0.8,
                    threshold=0.6,
                )
                for key in factorial.BUDGET_KEYS
            }
            payload = {
                "variant": spec.variant,
                "dataset": factorial.DATASET,
                "seed": factorial.TRAINING_SEED,
                "split_seed": factorial.SPLIT_SEED,
                "checkpoint_role": role,
                "checkpoint_epoch": epoch,
                "checkpoint_sha256": checkpoint_sha,
                "threshold_selection_scope": "single_checkpoint_only",
                "cross_checkpoint_point_pooling": False,
                "evaluated_checkpoint_count": 1,
                "official_test_accessed": False,
                "run_directory": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "fixed_threshold_0_5": fixed,
                "best_points_under_fa_budget": budgets,
                "validation_split_sha256": "3" * 64,
                "source_checkpoint_identity": checkpoint_identity,
            }
            (run_dir / factorial._sweep_filename(checkpoint)).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
    return run_dirs, audits


class FactorialComparisonTests(unittest.TestCase):
    def test_effect_formulas(self) -> None:
        points = {
            arm: point(
                matched=matched,
                unmatched=unmatched,
                miou=miou,
            )
            for arm, matched, unmatched, miou in (
                ("A", 180, 4, 0.80),
                ("B", 181, 3, 0.81),
                ("C", 182, 2, 0.82),
                ("D", 185, 1, 0.86),
            )
        }
        effects = factorial.factorial_effects(points)
        self.assertEqual(effects["B_minus_A"]["matched_target_count"], 1)
        self.assertEqual(effects["D_minus_C"]["matched_target_count"], 3)
        self.assertEqual(effects["C_minus_A"]["matched_target_count"], 2)
        self.assertEqual(effects["D_minus_B"]["matched_target_count"], 4)
        self.assertEqual(effects["marginal_tss"]["matched_target_count"], 2)
        self.assertEqual(effects["marginal_qfg"]["matched_target_count"], 3)
        self.assertEqual(
            effects["interaction_tss_x_qfg"]["matched_target_count"],
            2,
        )

    def test_report_keeps_checkpoint_roles_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = factorial.build_factorial_report(
                synthetic_records(Path(temporary))
            )
        self.assertTrue(report["seed42_descriptive_only"])
        self.assertFalse(report["stability_claim_supported"])
        self.assertFalse(report["causal_claim_supported"])
        self.assertEqual(
            report["posttraining_closure_source_lock"][
                "policy_summary_sha256"
            ],
            factorial.closure_policy.policy_summary_sha256(),
        )
        self.assertEqual(
            set(report["role_reports"]),
            {"pd_primary", "miou_secondary"},
        )
        for role in report["role_reports"].values():
            self.assertFalse(role["cross_role_pooling"])
            self.assertEqual(
                set(role["fa_budget_points"]),
                set(factorial.BUDGET_KEYS),
            )

    def test_collect_routes_all_eight_and_rejects_role_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dirs, audits = fixture_tree(Path(temporary))
            fake = FakeEvaluators(audits)
            module = fake.module()
            records = factorial.collect_validated_sweeps(
                run_dirs,
                evaluator_modules={
                    "survival": module,
                    "qfg": module,
                },
            )
            self.assertEqual(len(records), 8)
            self.assertEqual(fake.output_validation_count, 8)

            broken = dict(audits)
            key = (str(run_dirs["D"]), "best_miou.pth.tar")
            broken[key] = dict(broken[key])
            broken[key]["checkpoint_role"] = (
                "best_validation_pd_primary"
            )
            broken_fake = FakeEvaluators(broken)
            with self.assertRaisesRegex(ValueError, "checkpoint role"):
                factorial.collect_validated_sweeps(
                    run_dirs,
                    evaluator_modules={
                        "survival": broken_fake.module(),
                        "qfg": broken_fake.module(),
                    },
                )

            broken_variant = dict(audits)
            variant_key = (str(run_dirs["C"]), "best.pth.tar")
            broken_variant[variant_key] = dict(
                broken_variant[variant_key]
            )
            broken_variant[variant_key]["variant"] = "tss_qfg"
            broken_variant_fake = FakeEvaluators(broken_variant)
            with self.assertRaisesRegex(ValueError, "variant"):
                factorial.collect_validated_sweeps(
                    run_dirs,
                    evaluator_modules={
                        "survival": broken_variant_fake.module(),
                        "qfg": broken_variant_fake.module(),
                    },
                )

    def test_incomplete_arm_is_rejected_before_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dirs, audits = fixture_tree(Path(temporary))
            (run_dirs["C"] / "summary.json").unlink()
            fake = FakeEvaluators(audits)
            with self.assertRaisesRegex(FileNotFoundError, "summary"):
                factorial.collect_validated_sweeps(
                    run_dirs,
                    evaluator_modules={
                        "survival": fake.module(),
                        "qfg": fake.module(),
                    },
                )
            self.assertEqual(fake.output_validation_count, 0)

    def test_json_and_markdown_are_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = synthetic_records(root / "runs")
            report = factorial.build_factorial_report(records)
            run_dirs = {
                arm: records[(arm, "best.pth.tar")].run_directory
                for arm in factorial.ARM_SPECS
            }
            json_output = root / "output" / "factorial.json"
            markdown_output = root / "output" / "factorial.md"
            action = factorial.publish_report_once(
                report,
                json_output=json_output,
                markdown_output=markdown_output,
                run_directories=run_dirs,
            )
            self.assertTrue(json_output.is_file())
            self.assertTrue(markdown_output.is_file())
            self.assertEqual(action["status"], "complete")
            with self.assertRaises(FileExistsError):
                factorial.publish_report_once(
                    report,
                    json_output=json_output,
                    markdown_output=markdown_output,
                    run_directories=run_dirs,
                )

    def test_output_via_symlink_into_run_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = synthetic_records(root / "runs")
            report = factorial.build_factorial_report(records)
            run_dirs = {
                arm: records[(arm, "best.pth.tar")].run_directory
                for arm in factorial.ARM_SPECS
            }
            run_dirs["A"].mkdir(parents=True)
            output_link = root / "output_link"
            output_link.symlink_to(run_dirs["A"], target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "inside arm A"):
                factorial.publish_report_once(
                    report,
                    json_output=output_link / "factorial.json",
                    markdown_output=root / "factorial.md",
                    run_directories=run_dirs,
                )


if __name__ == "__main__":
    unittest.main()
