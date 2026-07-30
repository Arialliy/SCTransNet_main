from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import final_model_child_initialization_manifest as child
from experiments import final_model_replication_exact_core as core
from experiments import final_model_replication_seed_contract as seeds
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as b_trainer,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FakeWarmStartResult:
    def provenance(self) -> dict[str, object]:
        return {
            "schema": "sctransnet_tpd_extension_warm_start_v1",
            "parent_checkpoint_path": "/tmp/parent",
            "parent_checkpoint_sha256": "1" * 64,
            "parent_state_dict_path": ["state_dict"],
            "parent_state_key_count": 1,
            "preserved_new_state_key_count": 1,
            "new_module_prefixes": ["new"],
            "zero_init_prefixes": [],
        }


class FinalModelReplicationExactTests(unittest.TestCase):
    def _inputs(
        self,
        directory: Path,
        *,
        arm: str = core.ARM_B,
        trajectory_seed: int = seeds.DEPLOYMENT_HASH_SEED,
    ) -> core.ReplicationInputs:
        source_lock = directory / "source_lock.json"
        source_lock.write_text(
            json.dumps({"schema": "fixture"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_sha = seeds.file_sha256(source_lock)
        schedule = seeds.ReplicationSeedScheduleContract.derive(source_sha)
        schedule_path = directory / "seed_contract.json"
        seeds.write_contract_once(schedule_path, schedule)
        parent = b_trainer.PARENT_CHECKPOINT_PATH
        initialization = child.ChildInitializationManifest(
            arm=arm,
            trajectory_seed=trajectory_seed,
            seed_contract_sha256=seeds.file_sha256(schedule_path),
            certification_source_lock_sha256=source_sha,
            parent_seed=42,
            parent_checkpoint_path=str(parent.resolve()),
            parent_checkpoint_sha256=b_trainer.PARENT_CHECKPOINT_SHA256,
            parent_state_dict_sha256=b_trainer.PARENT_STATE_DICT_SHA256,
            parent_checkpoint_epoch=b_trainer.PARENT_CHECKPOINT_EPOCH,
        )
        initialization_path = directory / "child_init.json"
        child.write_manifest_once(initialization_path, initialization)
        return core.validate_replication_inputs(
            arm=arm,
            trajectory_seed=trajectory_seed,
            schedule_path=schedule_path,
            initialization_manifest_path=initialization_path,
            certification_source_lock_path=source_lock,
            verify_parent_lock_live=False,
            verify_source_lock_live=False,
        )

    def test_engineering_contract_cross_links_all_three_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            inputs = self._inputs(Path(directory_text))
            self.assertEqual(inputs.definition.variant, "tss_on")
            self.assertEqual(inputs.parent_seed, 42)
            self.assertEqual(
                inputs.initialization_scope,
                "fixed_parent_child_trajectory",
            )
            self.assertEqual(
                inputs.source_lock_sha256,
                inputs.schedule.certification_source_lock_sha256,
            )

    def test_dry_run_verifies_parent_without_constructing_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            inputs = self._inputs(Path(directory_text))
            payload = core.dry_run_payload(inputs)
            self.assertEqual(payload["status"], "DRY_RUN_CONTRACT_VALID")
            self.assertFalse(payload["formal_training_started"])
            self.assertEqual(payload["parent_load_count"], 1)

    def test_confirmatory_seed_is_rejected_by_engineering_v1_adapter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            source_lock = directory / "source_lock.json"
            source_lock.write_text('{"schema":"fixture"}\n', encoding="utf-8")
            source_sha = seeds.file_sha256(source_lock)
            schedule = seeds.ReplicationSeedScheduleContract.derive(source_sha)
            schedule_path = directory / "seed_contract.json"
            seeds.write_contract_once(schedule_path, schedule)
            confirmatory_seed = schedule.trajectory_seeds[0]
            initialization = child.ChildInitializationManifest(
                arm=core.ARM_B,
                trajectory_seed=confirmatory_seed,
                seed_contract_sha256=seeds.file_sha256(schedule_path),
                certification_source_lock_sha256=source_sha,
                parent_seed=42,
                parent_checkpoint_path=str(
                    b_trainer.PARENT_CHECKPOINT_PATH.resolve()
                ),
                parent_checkpoint_sha256=(
                    b_trainer.PARENT_CHECKPOINT_SHA256
                ),
                parent_state_dict_sha256=(
                    b_trainer.PARENT_STATE_DICT_SHA256
                ),
                parent_checkpoint_epoch=(
                    b_trainer.PARENT_CHECKPOINT_EPOCH
                ),
            )
            initialization_path = directory / "child_init.json"
            child.write_manifest_once(initialization_path, initialization)
            with self.assertRaisesRegex(
                core.ReplicationExactError,
                "confirmatory seeds require",
            ):
                core.validate_replication_inputs(
                    arm=core.ARM_B,
                    trajectory_seed=confirmatory_seed,
                    schedule_path=schedule_path,
                    initialization_manifest_path=initialization_path,
                    certification_source_lock_path=source_lock,
                    verify_parent_lock_live=False,
                    verify_source_lock_live=False,
                )

    def test_rng_reset_replays_all_cpu_streams_and_loader_order(self) -> None:
        first_generator = core.reset_trajectory_rng(3407)
        first = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.rand(3, generator=first_generator),
        )
        second_generator = core.reset_trajectory_rng(3407)
        second = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.rand(3, generator=second_generator),
        )
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.equal(first[3], second[3]))

    def test_parent_adapter_calls_strict_loader_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            inputs = self._inputs(Path(directory_text))
            fake_trainer = types.ModuleType("fake_replication_trainer")
            fake_trainer.survival_model = types.SimpleNamespace(
                build_formal_v4_reference=mock.Mock(
                    return_value=(TinyModel(), {})
                )
            )
            fake_trainer.load_parent_into_extension = mock.Mock(
                return_value=FakeWarmStartResult()
            )
            fake_trainer.InitializationPlan = b_trainer.InitializationPlan
            fake_definition = core.ArmDefinition(
                arm=core.ARM_B,
                variant="tss_on",
                trainer=fake_trainer,
                new_module_prefixes=("new",),
                zero_init_prefixes=(),
                validate_loaded_child=None,
            )
            fake_inputs = dataclasses.replace(
                inputs,
                definition=fake_definition,
            )
            with mock.patch.object(
                core,
                "verify_parent_checkpoint_payload",
            ) as verify:
                plan = core.prepare_extension_parent_once(
                    fake_inputs,
                    TinyModel(),
                )
            verify.assert_called_once_with(fake_inputs)
            fake_trainer.load_parent_into_extension.assert_called_once()
            self.assertEqual(
                plan.contract["mode"],
                "extension_parent_warm_start",
            )

    def test_overlay_restores_frozen_module_globals(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            inputs = self._inputs(Path(directory_text))
            original_seed = b_trainer.TRAINING_SEED
            original_source_key = b_trainer.SOURCE_LOCK_KEY
            with core.replication_trainer_overlay(inputs):
                self.assertEqual(
                    b_trainer.TRAINING_SEED,
                    inputs.trajectory_seed,
                )
                self.assertEqual(
                    b_trainer.SOURCE_LOCK_KEY,
                    core.SOURCE_LOCK_KEY,
                )
            self.assertEqual(b_trainer.TRAINING_SEED, original_seed)
            self.assertEqual(b_trainer.SOURCE_LOCK_KEY, original_source_key)

    def test_overlay_accepts_trajectory_seed_but_reuses_frozen_statistics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            inputs = self._inputs(Path(directory_text))
            with core.replication_trainer_overlay(inputs) as trainer:
                args = trainer.parse_args(
                    core._frozen_argv(
                        inputs,
                        [
                            "--parent-warm-start",
                            "--device",
                            "cpu",
                            "--allow-cpu-smoke",
                        ],
                    )
                )
                statistics = trainer.load_survival_target_statistics()
                self.assertEqual(args.seed, inputs.trajectory_seed)
                self.assertEqual(
                    args.run_tag,
                    core.ENGINEERING_RUN_TAGS[core.ARM_B],
                )
                # This artifact records its true derivation provenance; the
                # trajectory seed must not rewrite or relabel it.
                self.assertEqual(statistics["training_seed"], 42)

    def test_controlled_frozen_identity_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            core.ReplicationExactError,
            "--seed is controlled",
        ):
            core._parse_replication_args(
                [
                    "--trajectory-seed",
                    "3407",
                    "--seed-contract",
                    "/tmp/a",
                    "--child-initialization-manifest",
                    "/tmp/b",
                    "--certification-source-lock",
                    "/tmp/c",
                    "--seed",
                    "9",
                ]
            )

    def test_cuda_process_contract_uses_registered_gpu_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            inputs = self._inputs(Path(directory_text))
            gpu_uuid = inputs.definition.trainer.PHYSICAL_GPU_UUIDS["2"]
            environment = {
                "PYTHONHASHSEED": str(inputs.trajectory_seed),
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_INDEX": "2",
                "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_UUID": gpu_uuid,
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                core._require_process_seed_environment(
                    inputs,
                    require_cuda=True,
                )
            environment["CUDA_VISIBLE_DEVICES"] = "2"
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(
                    core.ReplicationExactError,
                    "registered physical GPU UUID",
                ),
            ):
                core._require_process_seed_environment(
                    inputs,
                    require_cuda=True,
                )

    def test_d_cuda_process_contract_uses_gpu3_qfg_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            inputs = self._inputs(
                Path(directory_text),
                arm=core.ARM_D,
            )
            gpu_uuid = inputs.definition.trainer.PHYSICAL_GPU_UUIDS["3"]
            environment = {
                "PYTHONHASHSEED": str(inputs.trajectory_seed),
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX": "3",
                "TPD_NER_V4_QFG_PHYSICAL_GPU_UUID": gpu_uuid,
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                core._require_process_seed_environment(
                    inputs,
                    require_cuda=True,
                )
            environment["TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX"] = "2"
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(
                    core.ReplicationExactError,
                    "physical GPU index differs",
                ),
            ):
                core._require_process_seed_environment(
                    inputs,
                    require_cuda=True,
                )


if __name__ == "__main__":
    unittest.main()
