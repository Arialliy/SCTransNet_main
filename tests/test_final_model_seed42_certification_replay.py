from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch.nn as nn

from experiments import final_model_replication_exact_core as frozen_core
from experiments import (
    final_model_seed42_certification_replay_contract as contract,
)
from experiments import (
    final_model_seed42_certification_replay_exact_core as core,
)
from experiments import (
    freeze_final_model_seed42_certification_replay_source_lock as source_lock,
)


EXPECTED_INITIAL_STATES = {
    core.ARM_B: "935b205b5eb19e9783c4d507e468d084746ce420ad61937e28daa3799c1890ea",
    core.ARM_D: "ef2097dde668e563e3d9a97527b3903436011367aa4051ce4ee080225d7084b5",
}


class Seed42ReplayContractTests(unittest.TestCase):
    def test_contract_fixes_new_seed42_scope_and_forbids_hash_seed(self) -> None:
        payload = contract.build_contract()
        self.assertEqual(payload["seed_roles"]["trajectory_seed"], 42)
        self.assertEqual(payload["seed_roles"]["builder_compatibility_seed"], 42)
        self.assertEqual(payload["seed_roles"]["split_seed"], 20260722)
        self.assertEqual(
            payload["seed_roles"]["supplementary_seed_not_in_primary_gate"],
            3407,
        )
        self.assertEqual(
            payload["seed_roles"]["forbidden_not_scheduled_seed"],
            426780603,
        )
        self.assertEqual(
            payload["data_and_metric_contract"]["threshold"],
            0.5,
        )
        self.assertFalse(
            payload["replay_identity"]["legacy_checkpoints_imported"]
        )
        self.assertFalse(
            payload["replay_identity"]["legacy_exact_journal_imported"]
        )
        self.assertEqual(
            [record["arm"] for record in payload["arms"]],
            ["b", "d"],
        )
        upstream = {
            record["arm"]: record["upstream_training_source_lock"]
            for record in payload["arms"]
        }
        self.assertEqual(
            upstream["b"]["path"],
            "experiments/tpd_ner_v4_survival_exact_source_lock.json",
        )
        self.assertEqual(
            upstream["b"]["sha256"],
            "23edf22eee2279dc59056ef4c4855ecd0d760fc3ee6856f902d44abecd9308cf",
        )
        self.assertEqual(
            upstream["d"]["path"],
            "experiments/"
            "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json",
        )
        self.assertEqual(
            upstream["d"]["sha256"],
            "8d55464851db9441383854189eff64c05daf25e7ff3502c6c67cf06401996478",
        )

    def test_contract_and_manifests_are_canonical_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            contract_path = directory / "contract.json"
            manifest_directory = directory / "manifests"
            first = contract.prepare(
                contract_path=contract_path,
                manifest_directory=manifest_directory,
            )
            second = contract.prepare(
                contract_path=contract_path,
                manifest_directory=manifest_directory,
            )
            self.assertEqual(first["contract_sha256"], second["contract_sha256"])
            self.assertEqual(first["run_count"], 2)
            for arm in ("b", "d"):
                path = contract.manifest_path(manifest_directory, arm)
                payload = contract.load_child_manifest(
                    path,
                    arm=arm,
                    contract_path=contract_path,
                )
                self.assertEqual(payload["trajectory_seed"], 42)
                self.assertEqual(payload["parent_load_count"], 1)
                self.assertFalse(payload["optimizer_inherited"])
                self.assertTrue(payload["all_child_parameters_trainable"])
                self.assertFalse(payload["legacy_checkpoint_imported"])
                self.assertEqual(
                    path.read_bytes(),
                    contract.canonical_json_bytes(
                        json.loads(path.read_text(encoding="utf-8"))
                    ),
                )

    def test_live_successor_source_lock_verifies(self) -> None:
        payload = source_lock.verify_source_lock()
        self.assertEqual(payload["trajectory_seed"], 42)
        self.assertEqual(payload["run_count"], 2)
        self.assertFalse(payload["seed_426780603_scheduled"])
        self.assertEqual(payload["source_count"], len(source_lock.SOURCE_PATHS))


class Seed42ReplayExactCoreTests(unittest.TestCase):
    def _inputs(self, arm: str) -> core.ReplayInputs:
        return core.validate_inputs(
            arm=arm,
            initialization_manifest_path=contract.manifest_path(
                contract.DEFAULT_MANIFEST_DIRECTORY,
                arm,
            ),
        )

    def test_b_and_d_dry_runs_have_new_independent_identities(self) -> None:
        legacy_paths = (
            "tpd_ner_v4_survival_exact_v1",
            "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized",
        )
        for arm in core.SUPPORTED_ARMS:
            inputs = self._inputs(arm)
            payload = core.dry_run_payload(inputs)
            self.assertEqual(payload["status"], "DRY_RUN_CONTRACT_VALID")
            self.assertEqual(payload["trajectory_seed"], 42)
            self.assertEqual(payload["default_threshold"], 0.5)
            self.assertEqual(payload["resolved_mode"], "--parent-warm-start")
            self.assertIn("seed42_certification_replay", payload["expected_run_id"])
            self.assertTrue(
                str(contract.DEFAULT_OUTPUT_ROOT)
                in payload["run_directory"]
            )
            self.assertFalse(
                any(value in payload["run_directory"] for value in legacy_paths)
            )
            self.assertFalse(payload["formal_training_started"])
            self.assertFalse(payload["gpu_used"])

    def test_loaded_child_start_matches_frozen_seed42_initial_state(self) -> None:
        for arm in core.SUPPORTED_ARMS:
            inputs = self._inputs(arm)
            with core.replay_trainer_overlay(inputs) as trainer:
                model, _ = trainer.build_selected_model(
                    inputs.definition.variant,
                    42,
                    eps=trainer.FORMAL_EPS,
                )
                plan = frozen_core.prepare_extension_parent_once(inputs, model)
            self.assertEqual(
                plan.initial_model_state_sha256,
                EXPECTED_INITIAL_STATES[arm],
            )

    def test_exact_resume_delegates_without_loading_parent(self) -> None:
        inputs = self._inputs(core.ARM_B)
        trainer = inputs.definition.trainer
        expected = object()
        fake_original = mock.Mock(return_value=expected)
        model = nn.Linear(1, 1)
        with mock.patch.object(trainer, "initialization_plan", fake_original):
            with core.replay_trainer_overlay(inputs) as overlaid:
                with mock.patch.object(
                    frozen_core,
                    "prepare_extension_parent_once",
                ) as parent_loader:
                    observed = overlaid.initialization_plan(
                        argparse.Namespace(
                            parent_warm_start=False,
                            exact_resume=True,
                        ),
                        core.run_directory(inputs),
                        model,
                    )
        self.assertIs(observed, expected)
        fake_original.assert_called_once()
        parent_loader.assert_not_called()

    def test_parent_warm_start_calls_parent_loader_exactly_once(self) -> None:
        inputs = self._inputs(core.ARM_D)
        expected = object()
        model = nn.Linear(1, 1)
        with core.replay_trainer_overlay(inputs) as trainer:
            with mock.patch.object(
                frozen_core,
                "prepare_extension_parent_once",
                return_value=expected,
            ) as parent_loader:
                observed = trainer.initialization_plan(
                    argparse.Namespace(
                        parent_warm_start=True,
                        exact_resume=False,
                    ),
                    core.run_directory(inputs),
                    model,
                )
        self.assertIs(observed, expected)
        parent_loader.assert_called_once()

    def test_overlay_source_lock_contract_uses_preverified_identity(self) -> None:
        inputs = self._inputs(core.ARM_B)
        with tempfile.TemporaryDirectory() as directory_text:
            statistics_path = Path(directory_text) / "statistics.json"
            statistics_path.write_text("{}\n", encoding="utf-8")
            with core.replay_trainer_overlay(inputs) as trainer:
                with mock.patch.object(
                    core.replay_source_lock,
                    "verify_source_lock",
                    side_effect=AssertionError(
                        "overlay must not recursively verify the source closure"
                    ),
                ) as verifier:
                    observed = trainer.source_lock_contract(
                        "a" * 64,
                        inputs.source_lock_path,
                        statistics_path,
                    )
        verifier.assert_not_called()
        self.assertEqual(
            observed[core.SOURCE_LOCK_KEY],
            inputs.source_lock_sha256,
        )
        self.assertEqual(observed["training_data"], "a" * 64)

    def test_controlled_identity_arguments_are_rejected(self) -> None:
        manifest = contract.manifest_path(
            contract.DEFAULT_MANIFEST_DIRECTORY,
            core.ARM_B,
        )
        for forbidden in (
            ["--seed", "3407"],
            ["--threshold", "0.4"],
            ["--output-root", "/tmp/elsewhere"],
        ):
            with self.assertRaisesRegex(
                core.Seed42ReplayExactError,
                "controlled",
            ):
                core.run_arm(
                    core.ARM_B,
                    [
                        "--child-initialization-manifest",
                        str(manifest),
                        "--dry-run-contract",
                        *forbidden,
                    ],
                )

    def test_non_trainable_child_parameter_is_rejected(self) -> None:
        model = nn.Linear(2, 1)
        model.weight.requires_grad_(False)
        with self.assertRaisesRegex(
            core.Seed42ReplayExactError,
            "non-trainable parameters",
        ):
            core.require_all_parameters_trainable(model)

    def test_wrong_existing_run_identity_is_not_resumable(self) -> None:
        inputs = self._inputs(core.ARM_B)
        with tempfile.TemporaryDirectory() as directory_text:
            replaced = dataclasses.replace(
                inputs,
                output_root=Path(directory_text),
            )
            directory = core.run_directory(replaced)
            directory.mkdir(parents=True)
            protocol = {
                "schema": inputs.definition.trainer.ENTRY_SCHEMA,
                "run_directory": str(directory),
                "run_identity": {
                    "run_id": "legacy-or-other-run",
                    "variant": inputs.definition.variant,
                    "dataset": contract.DATASET,
                    "seed": 42,
                    "split_seed": 20260722,
                },
            }
            (directory / "protocol.json").write_bytes(
                core._canonical_pretty_json_bytes(protocol)
            )
            with self.assertRaisesRegex(
                core.Seed42ReplayExactError,
                "run_id differs",
            ):
                core.resolve_initialization_mode(replaced)

    def test_launcher_is_seed42_only_and_exact_resume_aware(self) -> None:
        launcher = (
            contract.REPO_ROOT
            / "experiments/"
            "run_final_model_seed42_certification_replay_pair_2x5090.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("PYTHONHASHSEED=42", launcher)
        self.assertNotIn("3407", launcher)
        self.assertNotIn("426780603", launcher)
        self.assertIn("--resolve-mode", launcher)
        self.assertIn("TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_INDEX=2", launcher)
        self.assertIn("TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX=3", launcher)
        self.assertIn(
            "final_model_seed42_certification_replay_source_lock_v4.json",
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
