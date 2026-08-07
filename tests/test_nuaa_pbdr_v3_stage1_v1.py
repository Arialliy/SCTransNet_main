from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments import launch_nuaa_pbdr_v3_stage1_v1 as launcher
from experiments import three_dataset_pbdr_v3_models_seed42_v1 as registry
from experiments import three_dataset_v2_protocol as data_protocol
from experiments import train_nuaa_pbdr_v3_stage1_v1 as trainer


class TestStage1Registry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = registry.build_stage1_training_model(
            "best_miou"
        )

    def test_exact_568_key_current_warm_start(self) -> None:
        self.assertEqual(
            self.metadata["current_state_key_count"],
            registry.CURRENT_STATE_KEY_COUNT,
        )
        self.assertEqual(
            self.metadata["training_state_key_count"],
            registry.TRAINING_STATE_KEY_COUNT,
        )
        self.assertTrue(
            self.metadata["all_current_tensors_bitwise_equal_after_load"]
        )
        self.assertEqual(
            len(self.metadata["parent_checkpoint"]["sha256"]), 64
        )
        self.assertEqual(
            self.metadata["current_state_sha256_after_load"],
            self.metadata["parent_checkpoint"]["state_sha256"],
        )

    def test_only_pbdr_v3_is_trainable_and_base_is_eval(self) -> None:
        audit = registry.audit_stage1(self.model)
        self.assertEqual(audit["trainable_parameter_count"], 6018)
        self.assertTrue(audit["trainable_parameter_names"])
        self.assertTrue(
            all(
                name.startswith("pbdr_v3.")
                for name in audit["trainable_parameter_names"]
            )
        )
        self.assertFalse(audit["base_training"])
        self.assertTrue(audit["pbdr_v3_training"])

    def test_freeze_audit_hashes_real_state_not_only_flags(self) -> None:
        before_base = registry.base_state_sha256(self.model)
        before_bn = registry.batchnorm_buffer_sha256(self.model)
        self.model.train()
        audit = registry.configure_stage1(self.model)
        self.assertEqual(audit["base_state_sha256"], before_base)
        self.assertEqual(audit["batchnorm_buffer_sha256"], before_bn)
        self.assertGreater(len(audit["batchnorm_buffer_names"]), 0)

    def test_574_to_570_inference_state_is_strict(self) -> None:
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        inference, metadata = registry.build_inference_model_from_candidate_state(
            state,
            parent_role="best_miou",
        )
        self.assertEqual(len(inference.state_dict()), registry.INFERENCE_STATE_KEY_COUNT)
        self.assertTrue(metadata["base_bitwise_equal_to_parent"])
        self.assertTrue(metadata["strict_load"])
        self.assertEqual(inference.mode, "test")


class TestInternalSplitAndData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        calls: list[tuple[str, str]] = []
        original = data_protocol.load_index

        def recording(root: Path, dataset: str, split: str) -> list[str]:
            calls.append((dataset, split))
            return original(root, dataset, split)

        with mock.patch.object(data_protocol, "load_index", side_effect=recording):
            cls.manifest = trainer.build_internal_split_manifest()
        cls.index_calls = calls

    def test_split_is_170_43_disjoint_and_train_only(self) -> None:
        train_ids = self.manifest["development_train_ids"]
        val_ids = self.manifest["internal_validation_ids"]
        self.assertEqual((len(train_ids), len(val_ids)), (170, 43))
        self.assertFalse(set(train_ids) & set(val_ids))
        self.assertEqual(set(train_ids) | set(val_ids), set(self.manifest["official_train_ids"]))
        self.assertEqual(self.index_calls, [(registry.DATASET, "train")])
        self.assertFalse(self.manifest["official_test_index_opened"])

    def test_split_replays_from_mask_stats_and_seed(self) -> None:
        from experiments.train_tpd_pilot import MaskStats, stratified_split

        stats = [MaskStats(**record) for record in self.manifest["mask_stats"]]
        train_ids, val_ids = stratified_split(
            stats, trainer.VAL_FRACTION, trainer.SPLIT_SEED
        )
        self.assertEqual(train_ids, self.manifest["development_train_ids"])
        self.assertEqual(val_ids, self.manifest["internal_validation_ids"])

    def test_split_hash_is_self_consistent(self) -> None:
        payload = dict(self.manifest)
        observed = payload.pop("split_sha256")
        self.assertEqual(observed, registry.canonical_sha256(payload))

    def test_internal_train_crop_is_stateless_and_repeatable(self) -> None:
        identifier = self.manifest["development_train_ids"][0]
        first = trainer.NUAAInternalTrainDataset([identifier])
        second = trainer.NUAAInternalTrainDataset([identifier])
        first.set_epoch(19)
        second.set_epoch(19)
        first_image, first_mask = first[0]
        second_image, second_mask = second[0]
        self.assertTrue(torch.equal(first_image, second_image))
        self.assertTrue(torch.equal(first_mask, second_mask))
        self.assertEqual(tuple(first_image.shape), (1, 256, 256))

    def test_internal_validation_uses_explicit_train_id(self) -> None:
        identifier = self.manifest["internal_validation_ids"][0]
        dataset = trainer.NUAAInternalValidationDataset([identifier])
        image, mask, size, returned = dataset[0]
        self.assertEqual(returned, identifier)
        self.assertEqual(image.shape, mask.shape)
        self.assertEqual(image.shape[-2] % 32, 0)
        self.assertEqual(image.shape[-1] % 32, 0)
        self.assertGreater(size[0] * size[1], 0)


class _SyntheticValidationDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        del index
        image = torch.zeros(1, 4, 4)
        mask = torch.zeros(1, 4, 4)
        mask[:, 1:3, 1:3] = 1.0
        return image, mask, (4, 4), "train_sample"


class _SyntheticAux:
    def __init__(self) -> None:
        base = torch.full((1, 1, 4, 4), -4.0)
        routed = base.clone()
        base[:, :, 1:3, 1:3] = 4.0
        routed[:, :, 1:3, 1:3] = 5.0
        self.base_logits = base
        self.routed_logits = routed


class _SyntheticModel(torch.nn.Module):
    def forward_for_pbdr_v3_training(self, image: torch.Tensor):
        del image
        return (), _SyntheticAux()


class TestTrainerContracts(unittest.TestCase):
    def test_threshold_sweep_is_validation_derived_and_complete(self) -> None:
        loader = DataLoader(_SyntheticValidationDataset(), batch_size=1)
        result = trainer.evaluate_internal(
            _SyntheticModel(), loader, torch.device("cpu")
        )
        self.assertEqual(
            tuple(result["candidate_threshold_sweep"]),
            tuple(f"{value:.2f}" for value in trainer.THRESHOLDS),
        )
        self.assertEqual(len(result["candidate_threshold_sweep"]), 61)
        self.assertEqual(
            result["fixed_0_5"]["candidate"]["matched_target_count"], 1
        )

    def test_core_and_constrained_recipes_are_not_silently_combined(self) -> None:
        core = trainer._loss_kwargs("core", 20)
        constrained_1 = trainer._loss_kwargs("constrained", 1)
        constrained_20 = trainer._loss_kwargs("constrained", 20)
        self.assertTrue(all(value == 0.0 for value in core.values()))
        self.assertEqual(constrained_1["background_increase_weight"], 8.0)
        self.assertEqual(constrained_1["foreground_decrease_weight"], 4.0)
        self.assertEqual(constrained_1["hard_negative_weight"], 0.1)
        self.assertEqual(constrained_20["hard_negative_weight"], 2.0)
        self.assertEqual(constrained_20["deep_supervision_weight"], 0.0)

    def test_selection_is_parent_role_specific(self) -> None:
        high_iou = {"miou": 0.9, "pd": 0.8, "fa": 0.01, "niou": 0.8}
        high_pd = {"miou": 0.8, "pd": 0.9, "fa": 0.01, "niou": 0.8}
        self.assertGreater(
            trainer._selection_key("best_miou", high_iou, 5),
            trainer._selection_key("best_miou", high_pd, 5),
        )
        self.assertGreater(
            trainer._selection_key("best_pd", high_pd, 5),
            trainer._selection_key("best_pd", high_iou, 5),
        )

    def test_checkpoint_selection_prefers_a_passing_internal_gate(self) -> None:
        current = {
            "matched_target_count": 10,
            "target_count": 10,
            "fa": 0.01,
            "miou": 0.80,
            "niou": 0.80,
            "pd": 1.0,
        }
        passing = dict(current, miou=0.802)
        failing_but_higher_miou = dict(current, miou=0.95, fa=0.02)
        passing_key = trainer._checkpoint_selection_key(
            "best_miou",
            {"fixed_0_5": {"current": current, "candidate": passing}},
            10,
        )
        failing_key = trainer._checkpoint_selection_key(
            "best_miou",
            {
                "fixed_0_5": {
                    "current": current,
                    "candidate": failing_but_higher_miou,
                }
            },
            5,
        )
        self.assertGreater(passing_key, failing_key)

    def test_rng_state_round_trip(self) -> None:
        trainer.configure_determinism()
        state = trainer._rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.random(); np.random.rand(); torch.rand(3)
        trainer._restore_rng(state)
        observed = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[0], observed[0])
        self.assertEqual(expected[1], observed[1])
        self.assertTrue(torch.equal(expected[2], observed[2]))

    def test_smoke_and_formal_argument_contracts(self) -> None:
        smoke = trainer.parse_args(
            [
                "--parent-role", "best_miou", "--recipe", "core",
                "--smoke", "--epochs", "1", "--max-train-images", "2",
                "--max-val-images", "1", "--device", "cpu",
            ]
        )
        trainer.validate_args(smoke)
        formal = trainer.parse_args(
            ["--parent-role", "best_pd", "--recipe", "constrained"]
        )
        trainer.validate_args(formal)
        formal.epochs = 149
        with self.assertRaises(trainer.Stage1ProtocolError):
            trainer.validate_args(formal)


class TestLauncher(unittest.TestCase):
    def test_default_plan_is_four_sequential_gpu0_workers(self) -> None:
        specs = launcher.build_worker_specs()
        self.assertEqual(len(specs), 4)
        self.assertEqual(
            [(spec.parent_role, spec.recipe) for spec in specs],
            [
                ("best_miou", "core"),
                ("best_miou", "constrained"),
                ("best_pd", "core"),
                ("best_pd", "constrained"),
            ],
        )
        for spec in specs:
            self.assertIn(launcher.GPU_UUID, spec.command)
            self.assertIn("--expected-gpu-uuid", spec.command)

    def test_dry_run_builds_user_systemd_command_without_execution(self) -> None:
        args = launcher.parse_args(["--parent-role", "best_miou", "--recipe", "core"])
        specs = launcher.build_worker_specs(parent_role=args.parent_role, recipe=args.recipe)
        payload = launcher.dry_run_payload(args, specs)
        command = payload["systemd_command"]
        self.assertEqual(command[:3], ["systemd-run", "--user", "--collect"])
        self.assertIn(
            f"--setenv=CUDA_VISIBLE_DEVICES={launcher.GPU_UUID}", command
        )
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["official_test_evaluation_launched"])

    def test_gpu0_uuid_attestation(self) -> None:
        output = f"0, {launcher.GPU_UUID}\n1, GPU-other\n"
        completed = mock.Mock(stdout=output)
        with mock.patch.object(launcher.subprocess, "run", return_value=completed):
            observed = launcher.verify_gpu0_uuid()
        self.assertEqual(observed["0"], launcher.GPU_UUID)

    def test_worker_sequence_stops_on_first_failure(self) -> None:
        specs = launcher.build_worker_specs(parent_role="best_miou")
        results = [mock.Mock(returncode=0), mock.Mock(returncode=7)]
        with (
            mock.patch.object(launcher.subprocess, "run", side_effect=results) as run,
            mock.patch.object(
                launcher,
                "read_internal_gate_result",
                return_value=False,
            ),
        ):
            code = launcher.execute_sequence(specs, environment={})
        self.assertEqual(code, 7)
        self.assertEqual(run.call_count, 2)

    def test_passing_core_skips_same_role_constrained_worker(self) -> None:
        specs = launcher.build_worker_specs()
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=completed,
            ) as run,
            mock.patch.object(
                launcher,
                "read_internal_gate_result",
                return_value=True,
            ) as gate_result,
        ):
            code = launcher.execute_sequence(specs, environment={})
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(gate_result.call_count, 2)
        launched = [call.args[0] for call in run.call_args_list]
        self.assertTrue(all("core" in command for command in launched))


if __name__ == "__main__":
    unittest.main()
