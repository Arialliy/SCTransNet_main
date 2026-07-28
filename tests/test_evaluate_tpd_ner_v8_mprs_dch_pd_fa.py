from __future__ import annotations

import argparse
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import evaluate_pd_fa_sweep as canonical_base
from experiments import evaluate_tpd_ner_v8_mprs_dch_pd_fa as evaluator
from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_ner_v8_mprs_dch as trainer
from experiments import train_tpd_ner_v8_mprs_dch_exact as exact_trainer


class TPDNERV8MPRSDCHFormalEvaluatorTests(unittest.TestCase):
    def test_contract_covers_fixed_metrics_budgets_and_formal_roles(self) -> None:
        contract = evaluator.evaluator_contract()
        self.assertEqual(
            contract["formal_variants"],
            [
                trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
                trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
            ],
        )
        self.assertEqual(contract["training_seed"], 42)
        self.assertEqual(contract["split_seed"], 20260722)
        self.assertEqual(
            contract["required_fixed_threshold_metrics"],
            ["pd", "fa", "miou", "false_objects_per_image"],
        )
        self.assertEqual(contract["fa_budgets"], list(trainer.FA_BUDGETS))
        self.assertEqual(contract["required_budget_metric"], "pd")
        self.assertEqual(
            contract["required_ablation"],
            trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
        )
        self.assertEqual(
            contract["main_comparison"],
            [
                "baseline_sctransnet_external_same_split_reference",
                trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
                trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
            ],
        )
        self.assertEqual(
            contract["accepted_training_artifact_modes"],
            ["exact_resume_primary", "ordinary_compatibility"],
        )
        gates = contract["preregistered_performance_gates"]
        self.assertEqual(gates["anchor_target_count"], 189)
        self.assertEqual(
            gates["pd_primary_fixed_threshold_0_5"][
                "minimum_matched_targets"
            ],
            188,
        )
        self.assertEqual(
            gates["miou_selected_fixed_threshold_0_5"]["minimum_miou"],
            0.946542,
        )
        self.assertIn(
            trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
            contract["main_comparison"],
        )
        self.assertFalse(contract["official_test_accessed"])

    def test_private_evaluator_binds_only_isolated_module(self) -> None:
        original = (
            canonical_base.adaptive_thresholds,
            canonical_base.build_model,
            canonical_base.parse_args,
            canonical_base.write_output_json,
            canonical_base.__file__,
        )
        isolated = evaluator._load_isolated_base_evaluator()
        self.assertIsNot(isolated, canonical_base)
        self.assertIs(
            isolated.adaptive_thresholds,
            evaluator.adaptive_thresholds_closed_interval,
        )
        self.assertIs(
            isolated.build_model,
            trainer.build_tpd_ner_v8_mprs_dch_model,
        )
        self.assertEqual(isolated.__file__, evaluator.__file__)
        self.assertEqual(
            (
                canonical_base.adaptive_thresholds,
                canonical_base.build_model,
                canonical_base.parse_args,
                canonical_base.write_output_json,
                canonical_base.__file__,
            ),
            original,
        )

    def test_formal_evaluator_arguments_reject_metric_protocol_changes(self) -> None:
        args = argparse.Namespace(
            expected_epochs=None,
            device="cpu",
            threshold_min=0.01,
            threshold_max=0.99,
            threshold_step=0.01,
            tail_logit_step=0.1,
            extra_thresholds=[0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
            fa_budgets=list(trainer.FA_BUDGETS),
            match_radius=None,
            tiny_area=None,
        )
        evaluator._validate_formal_evaluator_args(args)
        self.assertEqual(args.expected_epochs, 800)
        for name, invalid in (
            ("expected_epochs", 799),
            ("device", "cuda:1"),
            ("threshold_step", 0.02),
            ("fa_budgets", [1e-6]),
            ("match_radius", 4.0),
            ("tiny_area", 10),
        ):
            changed = argparse.Namespace(**vars(args))
            setattr(changed, name, invalid)
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    evaluator._validate_formal_evaluator_args(changed)

    @staticmethod
    def _complete_output(
        variant: str = trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
    ) -> dict:
        points = {
            f"{budget:.10g}": {
                "threshold": evaluator.UPPER_BOUNDARY_THRESHOLD,
                "pd": 0.0,
                "fa": 0.0,
                "miou": 0.0,
                "false_objects_per_image": 0.0,
                "target_count": 189,
                "matched_target_count": 0,
            }
            for budget in trainer.FA_BUDGETS
        }
        return {
            "checkpoint": Path("/tmp/best.pth.tar"),
            "checkpoint_role": "best_validation_pd_primary",
            "dataset": trainer.DATASET,
            "variant": variant,
            "seed": trainer.TRAINING_SEED,
            "split_seed": trainer.SPLIT_SEED,
            "official_test_accessed": False,
            "threshold_configuration": {
                "fa_budgets": list(trainer.FA_BUDGETS),
            },
            "fixed_threshold_0_5": {
                "threshold": 0.5,
                "pd": 0.4,
                "fa": 2e-6,
                "miou": 0.3,
                "false_objects_per_image": 0.2,
                "target_count": 189,
                "matched_target_count": 76,
            },
            "best_points_under_fa_budget": points,
        }

    def test_finalizer_materializes_all_required_metric_views(self) -> None:
        payload = self._complete_output()
        finalized = evaluator.finalize_evaluation_output(payload)
        self.assertEqual(finalized["schema"], evaluator.EVALUATION_SCHEMA)
        self.assertEqual(
            set(finalized["final_metric_coverage"]["fixed_threshold_0_5"]),
            {"pd", "fa", "miou", "false_objects_per_image"},
        )
        self.assertEqual(
            set(finalized["final_metric_coverage"]["pd_at_fa_budget"]),
            {f"{budget:.10g}" for budget in trainer.FA_BUDGETS},
        )
        self.assertTrue(
            finalized["final_metric_coverage"]["all_required_metrics_present"]
        )
        self.assertEqual(
            finalized["run_identity"],
            evaluator.expected_run_identity(payload["variant"]),
        )
        self.assertFalse(
            finalized["performance_gate_assessment"][
                "absolute_checkpoint_gate_passed"
            ]
        )
        self.assertFalse(
            finalized["performance_gate_assessment"][
                "formal_success_claim_authorized"
            ]
        )

    def test_finalizer_rejects_missing_or_over_budget_metrics(self) -> None:
        missing_fixed = self._complete_output()
        del missing_fixed["fixed_threshold_0_5"]["false_objects_per_image"]
        with self.assertRaises(ValueError):
            evaluator.finalize_evaluation_output(missing_fixed)

        missing_budget = self._complete_output()
        del missing_budget["best_points_under_fa_budget"]["1e-06"]
        with self.assertRaises(ValueError):
            evaluator.finalize_evaluation_output(missing_budget)

        over_budget = self._complete_output()
        over_budget["best_points_under_fa_budget"]["1e-06"]["fa"] = 2e-6
        with self.assertRaises(ValueError):
            evaluator.finalize_evaluation_output(over_budget)

    def _write_artifact_fixture(
        self,
        root: Path,
        variant: str = trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
    ) -> tuple[Path, dict]:
        run_dir = (
            root
            / trainer.DATASET
            / variant
            / f"seed_{trainer.TRAINING_SEED}_{trainer.FORMAL_RUN_TAG}"
        )
        run_dir.mkdir(parents=True)
        identity = evaluator.expected_run_identity(variant)
        spec = trainer.variant_spec(variant)
        architecture_id = "a" * 64
        model_metadata = {
            "variant": variant,
            "comparison_role": spec["comparison_role"],
            "relay_enabled": spec["relay_enabled"],
            "architecture_id": architecture_id,
        }
        protocol_arguments = {
            "dataset": trainer.DATASET,
            "variant": variant,
            "seed": trainer.TRAINING_SEED,
            "split_seed": trainer.SPLIT_SEED,
            "epochs": trainer.FORMAL_EPOCHS,
            "batch_size": trainer.FORMAL_BATCH_SIZE,
            "patch_size": trainer.FORMAL_PATCH_SIZE,
            "workers": trainer.FORMAL_WORKERS,
            "val_fraction": trainer.FORMAL_VAL_FRACTION,
            "eval_every": trainer.FORMAL_EVAL_EVERY,
            "threshold": trainer.FORMAL_THRESHOLD,
            "match_radius": trainer.FORMAL_MATCH_RADIUS,
            "tiny_area": trainer.FORMAL_TINY_AREA,
            "amp": False,
            "max_train_images": None,
            "max_val_images": None,
            "run_tag": trainer.FORMAL_RUN_TAG,
        }
        common = {
            "run_identity": identity,
            "official_test_accessed": False,
        }
        protocol = {
            **common,
            "schema": trainer.ENTRY_SCHEMA,
            "arguments": protocol_arguments,
            "model": model_metadata,
            "training_contract": trainer.formal_training_contract(),
            "stored_validation_metrics": list(trainer.STORED_VALIDATION_METRICS),
        }
        train_ids = [f"train-{index:03d}" for index in range(530)]
        val_ids = [f"val-{index:03d}" for index in range(133)]
        split = {
            **common,
            "schema": trainer.SPLIT_SCHEMA,
            "dataset": trainer.DATASET,
            "split_seed": trainer.SPLIT_SEED,
            "full_official_train_count": 663,
            "used_train_count": 530,
            "used_val_count": 133,
            "used_train_ids": train_ids,
            "used_val_ids": val_ids,
            "ordered_used_train_sha256": trainer.ordered_identifier_sha256(
                train_ids
            ),
            "ordered_used_val_sha256": trainer.ordered_identifier_sha256(
                val_ids
            ),
        }
        summary = {
            **common,
            "schema": trainer.SUMMARY_SCHEMA,
            "status": "complete",
            "variant": variant,
            "dataset": trainer.DATASET,
            "seed": trainer.TRAINING_SEED,
            "selection_source": "internal_validation_only",
            "model": model_metadata,
            "training_contract": trainer.formal_training_contract(),
            "stored_validation_metrics": list(trainer.STORED_VALIDATION_METRICS),
        }
        for filename, payload in (
            ("protocol.json", protocol),
            ("split.json", split),
            ("summary.json", summary),
        ):
            (run_dir / filename).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        checkpoint_role = "best_validation_pd_primary"
        checkpoint_identity = {
            "schema": trainer.CHECKPOINT_IDENTITY_SCHEMA,
            "run_id": identity["run_id"],
            "variant": variant,
            "comparison_role": spec["comparison_role"],
            "relay_enabled": spec["relay_enabled"],
            "architecture_id": architecture_id,
            "checkpoint_role": checkpoint_role,
            "checkpoint_filename": "best.pth.tar",
        }
        checkpoint = {
            **common,
            "schema": trainer.CHECKPOINT_SCHEMA,
            "variant": variant,
            "dataset": trainer.DATASET,
            "seed": trainer.TRAINING_SEED,
            "split_seed": trainer.SPLIT_SEED,
            "selection_source": "internal_validation_only",
            "six_output_training_semantics": True,
            "model_metadata": model_metadata,
            "training_contract": trainer.formal_training_contract(),
            "stored_validation_metrics": list(trainer.STORED_VALIDATION_METRICS),
            "checkpoint_role": checkpoint_role,
            "checkpoint_identity": checkpoint_identity,
            "validation_metrics": {
                name: 0 for name in trainer.STORED_VALIDATION_METRICS
            },
            "state_dict": {"weight": torch.tensor([1.0])},
        }
        torch.save(checkpoint, run_dir / "best.pth.tar")
        return run_dir, checkpoint

    def test_preflight_accepts_coherent_identity_and_rejects_relay_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, checkpoint = self._write_artifact_fixture(Path(temporary))
            audit = evaluator.validate_run_artifacts(run_dir)
            self.assertEqual(
                audit["variant"],
                trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
            )
            self.assertTrue(audit["checkpoint_identity"]["relay_enabled"])

            checkpoint["checkpoint_identity"]["relay_enabled"] = False
            torch.save(checkpoint, run_dir / "best.pth.tar")
            with self.assertRaises(ValueError):
                evaluator.validate_run_artifacts(run_dir)

    def _write_exact_artifact_fixture(
        self,
        root: Path,
        variant: str = trainer.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
    ) -> tuple[Path, dict]:
        run_dir = (
            root
            / trainer.DATASET
            / variant
            / "seed_42_formal800_exact_v1"
        )
        run_dir.mkdir(parents=True)
        candidate = exact_trainer.candidate_contract(variant)
        architecture_manifest = {
            "schema": exact_trainer.ARCHITECTURE_MANIFEST_SCHEMA,
            "variant": variant,
            "relay_enabled": candidate["relay_enabled"],
            "relay_width": exact_trainer.RELAY_WIDTH,
        }
        builder_manifest_sha256 = exact_trainer.canonical_sha256(
            architecture_manifest
        )
        train_ids = [f"train-{index:03d}" for index in range(530)]
        val_ids = [f"val-{index:03d}" for index in range(133)]
        split_ids = {
            "full_train": train_ids,
            "full_validation": val_ids,
            "train": train_ids,
            "validation": val_ids,
        }
        split_fingerprints = {
            name: exact_runner.OrderedFingerprint.from_values(
                name,
                identifiers,
            ).normalized()
            for name, identifiers in split_ids.items()
        }
        data_values = {
            "official_training_data": ["data-digest"],
            "train_samples": ["train-record"],
            "validation_samples": ["validation-record"],
            "normalization": ['{"mean":0.5,"std":0.25}'],
        }
        data_fingerprints = {
            name: exact_runner.OrderedFingerprint.from_values(
                name,
                values,
            ).normalized()
            for name, values in data_values.items()
        }
        training_contract = {
            "batch_size": 16,
            "patch_size": 256,
            "workers": 0,
            "amp": False,
            "total_epochs": 800,
            "eval_interval": 1,
            "deep_supervision": {
                "enabled": True,
                "expected_outputs": 6,
                "training_uses_all_outputs": True,
                "validation_uses_final_output": True,
            },
            "loss": {
                "input": "post_sigmoid_probability",
                "aggregate": "sum",
                "compute_dtype": "float32",
            },
            "metric_config": {
                "threshold": 0.5,
                "match_radius": 3.0,
                "tiny_area": 9,
                "validation_batch_size": 1,
                "official_test_accessed": False,
            },
            "determinism": {
                "entry_schema": exact_trainer.ENTRY_SCHEMA,
                "parent_variant": candidate["parent_variant"],
                "relay_enabled": candidate["relay_enabled"],
                "relay_width": exact_trainer.RELAY_WIDTH,
                "relay_initialization_seed": (
                    exact_trainer.RELAY_INITIALIZATION_SEED
                ),
            },
        }
        identity = {
            "run_id": (
                f"{exact_trainer.RUN_ID_PREFIX}{trainer.DATASET}:{variant}:"
                "seed-42:formal800_exact_v1"
            ),
            "variant": variant,
            "architecture_id": "a" * 64,
            "dataset": trainer.DATASET,
            "seed": 42,
            "split_seed": 20260722,
            "split_sha256": evaluator._canonical_sha256(
                split_fingerprints
            ),
            "schema": exact_runner.RUN_IDENTITY_SCHEMA,
            "builder_manifest_sha256": builder_manifest_sha256,
            "source_locks": {
                exact_trainer.SOURCE_LOCK_KEY: "c" * 64,
            },
            "ordered_split_fingerprints": split_fingerprints,
            "ordered_data_fingerprints": data_fingerprints,
            "data_sha256": evaluator._canonical_sha256(
                data_fingerprints
            ),
            "training_contract": training_contract,
        }
        identity_contract = {
            "schema": exact_runner.RUN_IDENTITY_SCHEMA,
            "architecture_id": identity["architecture_id"],
            "builder_manifest_sha256": identity[
                "builder_manifest_sha256"
            ],
            "source_locks": identity["source_locks"],
            "ordered_split_fingerprints": split_fingerprints,
            "ordered_data_fingerprints": data_fingerprints,
            "data_sha256": identity["data_sha256"],
            "training": training_contract,
        }
        identity["contract_sha256"] = evaluator._canonical_sha256(
            identity_contract
        )
        model_metadata = {
            "variant": variant,
            "parent_variant": candidate["parent_variant"],
            "relay_enabled": candidate["relay_enabled"],
            "relay_width": exact_trainer.RELAY_WIDTH,
            "relay_initialization_seed": (
                exact_trainer.RELAY_INITIALIZATION_SEED
            ),
            "architecture_manifest": architecture_manifest,
            "architecture_id": builder_manifest_sha256,
            "sequence_contract": {
                "evidence_nodes": ("h11", "h12", "h13", "h21", "h22"),
                "nested": (("keep", "context"), ("saliency",)),
            },
        }
        protocol_arguments = {
            "dataset": trainer.DATASET,
            "variant": variant,
            "parent_variant": candidate["parent_variant"],
            "relay_enabled": candidate["relay_enabled"],
            "relay_width": exact_trainer.RELAY_WIDTH,
            "relay_initialization_seed": (
                exact_trainer.RELAY_INITIALIZATION_SEED
            ),
            "seed": 42,
            "split_seed": 20260722,
            "epochs": 800,
            "batch_size": 16,
            "patch_size": 256,
            "workers": 0,
            "val_fraction": 0.20,
            "eval_every": 1,
            "base_lr": 1e-3,
            "min_lr": 1e-5,
            "warmup_epochs": 10,
            "threshold": 0.5,
            "match_radius": 3.0,
            "tiny_area": 9,
            "eps": exact_trainer.FORMAL_EPS,
            "amp": False,
            "allow_cpu_smoke": False,
            "max_train_images": None,
            "max_val_images": None,
            "run_tag": "formal800_exact_v1",
            "device": "cuda:0",
        }
        protocol = {
            "schema": exact_trainer.ENTRY_SCHEMA,
            "formal_contract": exact_trainer.formal_contract(),
            "arguments": protocol_arguments,
            "model": model_metadata,
            "run_identity": identity,
            "stored_validation_metrics": list(
                exact_trainer.STORED_VALIDATION_METRICS
            ),
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
            "relay_identity": {
                "source": "candidate_variant_suffix",
                "parent_variant": candidate["parent_variant"],
                "enabled": candidate["relay_enabled"],
                "width": exact_trainer.RELAY_WIDTH,
                "initialization_seed": (
                    exact_trainer.RELAY_INITIALIZATION_SEED
                ),
            },
        }
        split_hashes = {
            "full_internal_train_sha256": trainer.base.identifier_hash(
                train_ids
            ),
            "full_internal_val_sha256": trainer.base.identifier_hash(val_ids),
            "used_train_sha256": trainer.base.identifier_hash(train_ids),
            "used_val_sha256": trainer.base.identifier_hash(val_ids),
        }
        split = {
            "dataset": trainer.DATASET,
            "source": f"img_idx/train_{trainer.DATASET}.txt",
            "official_test_accessed": False,
            "split_seed": 20260722,
            "val_fraction": 0.20,
            "full_official_train_count": 663,
            "full_internal_train_count": 530,
            "full_internal_val_count": 133,
            "used_train_count": 530,
            "used_val_count": 133,
            "hashes": split_hashes,
            "full_internal_train_ids": train_ids,
            "full_internal_val_ids": val_ids,
            "used_train_ids": train_ids,
            "used_val_ids": val_ids,
        }
        summary = {
            "schema": exact_trainer.COMPLETION_SUMMARY_SCHEMA,
            "status": "complete",
            "variant": variant,
            "dataset": trainer.DATASET,
            "seed": 42,
            "split_seed": 20260722,
            "parent_variant": candidate["parent_variant"],
            "relay_enabled": candidate["relay_enabled"],
            "relay_width": exact_trainer.RELAY_WIDTH,
            "formal_contract": exact_trainer.formal_contract(),
            "stored_validation_metrics": list(
                exact_trainer.STORED_VALIDATION_METRICS
            ),
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
            "model": model_metadata,
            "split_hashes": split_hashes,
            "run_identity": identity,
        }
        for filename, payload in (
            ("protocol.json", protocol),
            ("split.json", split),
            ("summary.json", summary),
        ):
            (run_dir / filename).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        metrics = {
            name: 0 for name in exact_trainer.STORED_VALIDATION_METRICS
        }
        exact_payload = {
            "model": {"state_dict": {"weight": torch.tensor([1.0])}},
            "optimizer": {"state_dict": {"state": {}, "param_groups": []}},
            "scaler": {"state_dict": {}},
        }
        context = types.SimpleNamespace(
            run_identity=identity,
            exact_payload=exact_payload,
            metrics=metrics,
            epoch=1,
            role="best_validation_pd_primary",
        )
        checkpoint = dict(
            exact_trainer.EvaluatorCheckpointAdapter(
                model_metadata=model_metadata,
                split_hashes=split_hashes,
            )(context)
        )
        checkpoint["derived_schema"] = exact_runner.DERIVED_CHECKPOINT_SCHEMA
        checkpoint["source_exact_checkpoint_sha256"] = "e" * 64
        checkpoint["state_dict_sha256"] = (
            exact_runner._state_content_sha256(
                checkpoint["state_dict"],
                "fixture state_dict",
            )
        )
        checkpoint["optimizer_state_sha256"] = (
            exact_runner._state_content_sha256(
                checkpoint["optimizer"],
                "fixture optimizer",
            )
        )
        checkpoint["scaler_state_sha256"] = (
            exact_runner._state_content_sha256(
                checkpoint["scaler"],
                "fixture scaler",
            )
        )
        torch.save(checkpoint, run_dir / "best.pth.tar")
        return run_dir, checkpoint

    def test_preflight_accepts_exact_primary_artifact_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, checkpoint = self._write_exact_artifact_fixture(
                Path(temporary)
            )
            audit = evaluator.validate_run_artifacts(run_dir)
            self.assertEqual(
                audit["training_artifact_mode"],
                "exact_resume_primary",
            )
            self.assertEqual(
                audit["run_identity"],
                checkpoint["run_identity"],
            )
            self.assertEqual(
                audit["checkpoint_identity"]["schema"],
                exact_trainer.CHECKPOINT_IDENTITY_SCHEMA,
            )

            checkpoint["model_metadata"]["sequence_contract"]["nested"] = (
                ("keep", "context"),
                ("changed",),
            )
            torch.save(checkpoint, run_dir / "best.pth.tar")
            with self.assertRaisesRegex(
                ValueError,
                "exact checkpoint.model differs",
            ):
                evaluator.validate_run_artifacts(run_dir)

            checkpoint["model_metadata"]["sequence_contract"]["nested"] = (
                ("keep", "context"),
                ("saliency",),
            )
            checkpoint["run_identity"]["seed"] = 3407
            torch.save(checkpoint, run_dir / "best.pth.tar")
            with self.assertRaises(ValueError):
                evaluator.validate_run_artifacts(run_dir)

    def test_main_configures_preflights_and_runs_private_evaluator(self) -> None:
        observed: list[str] = []
        private = types.SimpleNamespace(main=lambda: observed.append("run"))
        audit = {"checkpoint_identity": {}}
        with (
            mock.patch.object(evaluator, "requested_device", return_value="cpu"),
            mock.patch.object(evaluator, "configure_v8_inference") as configure,
            mock.patch.object(
                evaluator,
                "preflight_requested_artifacts",
                return_value=audit,
            ) as preflight,
            mock.patch.object(
                evaluator,
                "_load_isolated_base_evaluator",
                return_value=private,
            ) as loader,
        ):
            evaluator.main()
        configure.assert_called_once_with("cpu")
        preflight.assert_called_once()
        loader.assert_called_once_with(audit)
        self.assertEqual(observed, ["run"])


if __name__ == "__main__":
    unittest.main()
