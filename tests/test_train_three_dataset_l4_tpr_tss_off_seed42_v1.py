from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import train_three_dataset_l4_tpr_tss_off_seed42_v1 as trainer
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr import (
    L4_TPR_STATE_KEYS,
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet,
)


torch.set_num_threads(1)


def _args(
    decision_path: Path | None = None,
    dataset: str = "NUAA-SIRST",
) -> argparse.Namespace:
    argv = [
        "--dataset",
        dataset,
        "--method",
        "final",
        "--physical-gpu-index",
        "0",
        "--expected-gpu-uuid",
        trainer.GPU_UUIDS["0"],
    ]
    if decision_path is not None:
        argv.extend(["--execution-decision-json", str(decision_path)])
    return trainer.parse_args(argv)


def _decision_payload() -> dict[str, object]:
    return {
        "schema": trainer.EXECUTION_DECISION_SCHEMA,
        "status": "complete",
        "decision": trainer.EXECUTION_DECISION_AUTHORIZE,
        "architecture": trainer.ARCHITECTURE,
        "recipe_id": trainer.RECIPE_ID,
        "seed": 42,
        "datasets": list(trainer.DATASETS),
        "training_authorized": True,
        "fresh_seed42_scratch": True,
        "parent_checkpoint": None,
        "warm_start_used": False,
        "requested_tss_weight": 0.0,
    }


class L4TPRProtocolAndBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = trainer._build_method_model(
            "final",
            42,
            dataset_name="NUAA-SIRST",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model
        del cls.metadata

    def test_frozen_three_dataset_seed42_protocol(self) -> None:
        args = _args()
        self.assertEqual(
            trainer.DATASETS,
            ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"),
        )
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.begin_test, 10)
        self.assertEqual(args.eval_every, 10)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.patch_size, 256)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.base_lr, 1e-3)
        self.assertEqual(args.min_lr, 1e-5)
        self.assertEqual(args.warmup_epochs, 10)
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(trainer.CHECKPOINT_ROLES, ("best_miou", "best_pd"))
        self.assertEqual(trainer.TRAINING_STATE_KEY_COUNT, 569)
        self.assertEqual(trainer.INFERENCE_STATE_KEY_COUNT, 565)
        self.assertIsNone(args.execution_decision_json)
        self.assertNotIn(
            "gcsf",
            str(trainer.DEFAULT_RESULTS_ROOT).lower(),
        )
        self.assertIn("l4_tpr", str(trainer.DEFAULT_RESULTS_ROOT))

        recipe = trainer.recipe_identity(args)
        self.assertEqual(recipe["recipe_id"], trainer.RECIPE_ID)
        self.assertEqual(recipe["architecture"], trainer.ARCHITECTURE)
        self.assertEqual(recipe["requested_tss_weight"], 0.0)
        self.assertFalse(recipe["tss_enabled"])
        self.assertTrue(recipe["fresh_seed42_scratch"])
        self.assertEqual(recipe["optimizer"], "Adam")
        self.assertTrue(recipe["optimizer_state_initialized_fresh"])
        self.assertIsNone(recipe["parent_checkpoint"])
        self.assertFalse(recipe["warm_start_used"])
        self.assertFalse(recipe["old_final_568_state_accepted"])

    def test_builder_is_exact_fresh_569_key_graph_and_gate_is_trainable(self) -> None:
        self.assertIs(
            type(self.model),
            TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet,
        )
        self.assertEqual(len(self.model.state_dict()), 569)
        self.assertEqual(
            self.metadata["initialization_mode"],
            "fresh_seed42_paired_scratch_extension",
        )
        self.assertEqual(
            self.metadata["construction"],
            "scratch_seed42_no_parent_checkpoint",
        )
        self.assertIsNone(self.metadata["parent_checkpoint"])
        self.assertFalse(self.metadata["warm_start_used"])
        self.assertFalse(self.metadata["learned_state_loaded"])
        self.assertTrue(
            self.metadata["all_pre_l4_tpr_state_bitwise_equal_to_reference"]
        )
        self.assertTrue(self.metadata["l4_tpr_new_state_zero_initialized"])
        self.assertEqual(set(L4_TPR_STATE_KEYS), {"ner_l4_tpr.reallocation_logits"})
        for key in L4_TPR_STATE_KEYS:
            self.assertEqual(
                int(torch.count_nonzero(self.model.state_dict()[key])),
                0,
            )
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        owned = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(
            {
                id(parameter)
                for parameter in self.model.ner_l4_tpr.parameters()
            }
            <= owned
        )

    def test_runtime_closure_contains_new_code_not_gcsf_or_future_decision(self) -> None:
        paths = trainer.runtime_source_paths()
        observed = {key.split("::", 1)[1] for key in paths}
        required = {
            "experiments/train_three_dataset_l4_tpr_tss_off_seed42_v1.py",
            "experiments/export_tpd_ner_v4_qfg_v2_croa_l4_tpr_to_inference.py",
            "experiments/train_three_dataset_tss_off_seed42_v1.py",
            "model/tpd_ner_l4_target_protected_reallocation.py",
            "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr.py",
        }
        self.assertTrue(required <= observed)
        self.assertFalse(any("gcsf" in relative for relative in observed))
        self.assertFalse(any("decision" in relative for relative in observed))
        self.assertTrue(all(path.is_file() for path in paths.values()))


class L4TPRExecutionAndArtifactTests(unittest.TestCase):
    def test_no_implicit_decision_file_and_explicit_decision_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "execution-decision-json"):
            trainer.validate_args(_args())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screening_decision.json"
            path.write_text(json.dumps(_decision_payload()), encoding="utf-8")
            args = _args(path)
            binding = trainer.validate_args(args)
            self.assertEqual(binding["path"], str(path.resolve()))
            self.assertEqual(binding["sha256"], trainer.engine.file_sha256(path))
            self.assertTrue(binding["training_authorized"])

            failed = _decision_payload()
            failed["training_authorized"] = False
            path.write_text(json.dumps(failed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training_authorized"):
                trainer.validate_args(_args(path))

    def test_protocol_records_scratch_adam_tss_off_and_decision_binding(self) -> None:
        args = _args()
        decision = {
            "path": "/later/screening/decision.json",
            "sha256": "a" * 64,
            "schema": trainer.EXECUTION_DECISION_SCHEMA,
            "decision": trainer.EXECUTION_DECISION_AUTHORIZE,
            "training_authorized": True,
        }
        args.execution_decision_binding = decision
        seed_payload = {
            "training": {},
            "runtime_sources": {"old": {}},
        }
        with (
            mock.patch.object(
                trainer,
                "_BASE_PROTOCOL_PAYLOAD",
                return_value=seed_payload,
            ),
            mock.patch.object(
                trainer,
                "runtime_source_records",
                return_value={"runtime::runner": {"sha256": "b" * 64}},
            ),
        ):
            payload = trainer._protocol_payload(
                args,
                model_metadata={
                    "architecture_id": "architecture",
                    "architecture_manifest": {},
                },
                tss_metadata={},
                data_manifests={},
                train_count=1,
                test_count=1,
                device=torch.device("cpu"),
            )
        self.assertEqual(payload["schema"], trainer.SCHEMA)
        self.assertEqual(payload["execution_decision"], decision)
        self.assertEqual(payload["training"]["optimizer"], "Adam")
        self.assertEqual(
            payload["training"]["optimizer_state_initialization"],
            "fresh",
        )
        self.assertEqual(
            payload["training"]["initialization"],
            "fresh_seed42_scratch",
        )
        self.assertIsNone(payload["training"]["parent_checkpoint"])
        self.assertFalse(payload["training"]["warm_start_used"])
        self.assertEqual(payload["training"]["requested_tss_weight"], 0.0)
        self.assertFalse(payload["training"]["tss_enabled"])
        self.assertFalse(payload["training"]["old_final_568_resume_allowed"])
        self.assertEqual(payload["training"]["resume_state_key_count"], 569)

    def test_old_568_state_is_rejected_even_with_current_wrapper_metadata(self) -> None:
        model, _ = trainer._build_method_model(
            "final", 42, dataset_name="NUDT-SIRST"
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        args = _args(dataset="NUDT-SIRST")
        args.resume = "required"
        args.execution_decision_binding = {"sha256": "d" * 64}
        architecture_id, _ = trainer._architecture_binding(model)
        payload = {
            "schema": trainer.SCHEMA,
            "dataset": args.dataset,
            "method": "final",
            "seed": 42,
            "protocol_sha256": "protocol",
            "recipe": trainer.recipe_identity(args),
            "architecture_id": architecture_id,
            "l4_tpr_integration_version": (
                trainer.l4_tpr.L4_TPR_INTEGRATION_VERSION
            ),
            "training_state_key_count": 569,
            "planned_total_epochs": 1000,
            "execution_decision_sha256": "d" * 64,
            # Deliberately model the historical 568-entry Final graph without
            # allocating another full model checkpoint.
            "state_dict": {
                f"historical.{index}": torch.zeros(())
                for index in range(568)
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_training_state.pth.tar"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "old 568-key Final"):
                trainer._load_resume_l4_tpr(
                    args=args,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    protocol_sha256="protocol",
                )

    def test_patch_context_restores_the_reused_runner(self) -> None:
        args = _args()
        args.execution_decision_binding = {"sha256": "e" * 64}
        before_builder = trainer.base._build_method_model
        before_resume = trainer.base._load_resume_off
        with trainer._patched_base_and_engine(args):
            self.assertIs(trainer.base._build_method_model, trainer._build_method_model)
            self.assertIs(trainer.base._load_resume_off, trainer._load_resume_l4_tpr)
        self.assertIs(trainer.base._build_method_model, before_builder)
        self.assertIs(trainer.base._load_resume_off, before_resume)

    def test_result_directory_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _args(dataset="IRSTD-1K")
            args.results_root = Path(directory)
            expected = (
                Path(directory).resolve()
                / "runs"
                / "IRSTD-1K"
                / trainer.RECIPE_ID
                / "seed_42"
            )
            self.assertEqual(trainer._run_directory(args), expected)


if __name__ == "__main__":
    unittest.main()
