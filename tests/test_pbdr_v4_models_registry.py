from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from experiments import pbdr_v4_models_seed42_v1 as registry
from experiments.pbdr_v4_state_contract import mutable_parameter_names
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v4 import (
    PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
    PBDR_V4_STATE_KEYS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_survival_model,
)


torch.set_num_threads(1)

SOURCE_SHA = "1" * 64
SPLIT_SHA = "2" * 64
ATLAS_SHA = "3" * 64
PROTOCOL_SHA = "4" * 64
CHECKPOINT_SHA = "5" * 64


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


class PBDRV4ModelsRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current_model, _ = build_formal_v4_qfg_v2_croa_survival_model()
        cls.current_state = _clone_state(cls.current_model)
        cls.registry_state_sha = registry.nuaa_registry.tensor_mapping_sha256(
            cls.current_state
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.current_model
        del cls.current_state

    def setUp(self) -> None:
        self.nuaa_loader = patch.object(
            registry.nuaa_registry,
            "load_current_checkpoint",
            side_effect=self._load_nuaa,
        )
        self.cross_loader = patch.object(
            registry.cross_registry,
            "load_current_checkpoint",
            side_effect=self._load_cross,
        )
        self.mock_nuaa = self.nuaa_loader.start()
        self.mock_cross = self.cross_loader.start()

    def tearDown(self) -> None:
        self.nuaa_loader.stop()
        self.cross_loader.stop()

    @classmethod
    def _fixture(cls, dataset: str, role: str):
        payload = {
            "schema": "fixture_current/v1",
            "dataset": dataset,
            "checkpoint_role": role,
            "epoch": 17,
            "state_dict": cls.current_state,
        }
        record = {
            "dataset": dataset,
            "checkpoint_role": role,
            "path": f"/fixture/{dataset}/{role}.pth.tar",
            "sha256": CHECKPOINT_SHA,
            "bytes": 123456,
            "state_key_count": len(cls.current_state),
            "state_sha256": cls.registry_state_sha,
            "epoch": 17,
            "schema": "fixture_current/v1",
            "protocol_sha256": PROTOCOL_SHA,
        }
        return payload, cls.current_state, record

    @classmethod
    def _load_nuaa(cls, role: str):
        return cls._fixture("NUAA-SIRST", role)

    @classmethod
    def _load_cross(cls, dataset: str, role: str):
        return cls._fixture(dataset, role)

    def _stage1(self, dataset: str = "NUAA-SIRST", role: str = "best_miou"):
        return registry.build_stage1_training_model(dataset, role, "stage1")

    def _payload(
        self,
        model: torch.nn.Module,
        metadata: dict[str, object],
        *,
        dataset: str = "NUAA-SIRST",
        role: str = "best_miou",
        stage: str = "stage1",
    ) -> dict[str, object]:
        return registry.build_candidate_checkpoint_payload(
            model,
            dataset_name=dataset,
            role=role,
            stage=stage,
            source_sha256=SOURCE_SHA,
            split_sha256=SPLIT_SHA,
            atlas_sha256=ATLAS_SHA,
            initialization_sha256=str(metadata["initialization_sha256"]),
        )

    def _stage2(
        self,
        payload: dict[str, object],
        metadata: dict[str, object],
    ):
        return registry.build_stage2_training_model(
            payload,
            dataset_name="NUAA-SIRST",
            role="best_miou",
            stage="stage2",
            expected_source_sha256=SOURCE_SHA,
            expected_split_sha256=SPLIT_SHA,
            expected_atlas_sha256=ATLAS_SHA,
            expected_initialization_sha256=str(metadata["initialization_sha256"]),
            expected_stage1_state_sha256=str(payload["state_sha256"]),
        )

    def test_loader_routes_three_datasets_and_rejects_bad_coordinates(self) -> None:
        for dataset in registry.DATASETS:
            with self.subTest(dataset=dataset):
                payload, state, record = registry.load_current_checkpoint(
                    dataset,
                    "best_pd",
                )
                self.assertEqual(payload["dataset"], dataset)
                self.assertEqual(record["dataset"], dataset)
                self.assertEqual(len(state), registry.CURRENT_STATE_KEY_COUNT)
        self.mock_nuaa.assert_called_once_with("best_pd")
        self.assertEqual(
            self.mock_cross.call_args_list[0].args,
            ("NUDT-SIRST", "best_pd"),
        )
        self.assertEqual(
            self.mock_cross.call_args_list[1].args,
            ("IRSTD-1K", "best_pd"),
        )

        for dataset, role in (("bad", "best_miou"), ("NUAA-SIRST", "bad")):
            with self.subTest(dataset=dataset, role=role):
                with self.assertRaises(registry.PBDRV4ModelRegistryError):
                    registry.load_current_checkpoint(dataset, role)
        with self.assertRaisesRegex(registry.PBDRV4ModelRegistryError, "stage1"):
            registry.build_stage1_training_model(
                "NUAA-SIRST",
                "best_miou",
                "stage2",
            )
        with self.assertRaisesRegex(registry.PBDRV4ModelRegistryError, "stage"):
            registry.build_frozen_current_reference_model(
                "NUAA-SIRST",
                "best_miou",
                "bad",
            )

    def test_stage1_is_exact_27_key_extension_with_frozen_current_modes(self) -> None:
        model, metadata = self._stage1()
        state = model.state_dict()
        base = {
            name: tensor
            for name, tensor in state.items()
            if not name.startswith("pbdr_v4.")
        }
        self.assertEqual(tuple(name for name in state if name.startswith("pbdr_v4.")), PBDR_V4_STATE_KEYS)
        self.assertEqual(len(state), registry.TRAINING_STATE_KEY_COUNT)
        self.assertEqual(set(base), set(self.current_state))
        for name, expected in self.current_state.items():
            self.assertTrue(torch.equal(base[name], expected), name)

        expected_mutable = set(mutable_parameter_names(model, "stage1"))
        actual_mutable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(actual_mutable, expected_mutable)
        self.assertTrue(all(name.startswith("pbdr_v4.") for name in actual_mutable))
        self.assertFalse(model.training)
        self.assertTrue(model.pbdr_v4.training)
        self.assertFalse(
            any(
                module.training
                for name, module in model.named_modules()
                if name and not name.startswith("pbdr_v4")
            )
        )
        self.assertFalse(metadata["training_modes"]["base_training"])
        self.assertEqual(metadata["exact_current_extension_key_count"], 27)
        self.assertFalse(metadata["dataset_index_accessed"])

    def test_stage2_recomputes_initialization_sha_and_enables_exact_mutable_set(self) -> None:
        stage1, stage1_metadata = self._stage1()
        with torch.no_grad():
            stage1.pbdr_v4.residual_head.bias.add_(0.125)
        payload = self._payload(stage1, stage1_metadata)
        stage2, metadata = self._stage2(payload, stage1_metadata)
        self.assertEqual(
            registry.state_semantic_sha256(stage2.state_dict()),
            payload["state_sha256"],
        )
        expected_mutable = set(mutable_parameter_names(stage2, "stage2"))
        actual_mutable = {
            name
            for name, parameter in stage2.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(actual_mutable, expected_mutable)
        self.assertTrue(
            all(
                name.startswith(("pbdr_v4.", "outc.", "up_decoder1."))
                for name in actual_mutable
            )
        )
        self.assertFalse(stage2.training)
        self.assertTrue(stage2.pbdr_v4.training)
        self.assertFalse(metadata["training_modes"]["base_training"])
        self.assertEqual(metadata["stage2_initial_state_sha256"], payload["state_sha256"])

        bad = dict(payload)
        bad["initialization_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            registry.PBDRV4ModelRegistryError,
            "initialization_sha256 differs",
        ):
            registry.build_stage2_training_model(
                bad,
                dataset_name="NUAA-SIRST",
                role="best_miou",
                stage="stage2",
                expected_source_sha256=SOURCE_SHA,
                expected_split_sha256=SPLIT_SHA,
                expected_atlas_sha256=ATLAS_SHA,
                expected_initialization_sha256=str(stage1_metadata["initialization_sha256"]),
            )

        # Even when a caller repeats the forged value as its expected lock, a
        # fresh identity graph independently rejects it.
        with self.assertRaisesRegex(
            registry.PBDRV4ModelRegistryError,
            "fresh identity graph",
        ):
            registry.build_stage2_training_model(
                bad,
                dataset_name="NUAA-SIRST",
                role="best_miou",
                stage="stage2",
                expected_source_sha256=SOURCE_SHA,
                expected_split_sha256=SPLIT_SHA,
                expected_atlas_sha256=ATLAS_SHA,
                expected_initialization_sha256="f" * 64,
            )

    def test_stage2_validates_source_split_atlas_parent_current_and_state(self) -> None:
        stage1, metadata = self._stage1()
        payload = self._payload(stage1, metadata)
        cases = (
            ("source_sha256", "6" * 64, "source_sha256"),
            ("split_sha256", "7" * 64, "split_sha256"),
            ("atlas_sha256", "8" * 64, "atlas_sha256"),
            ("current_state_sha256", "9" * 64, "Current semantic"),
            ("state_sha256", "a" * 64, "tensor-state"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                bad = dict(payload)
                bad[field] = value
                with self.assertRaisesRegex(registry.PBDRV4ModelRegistryError, message):
                    self._stage2(bad, metadata)

        bad_parent = copy.deepcopy(payload)
        bad_parent["parent_checkpoint"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(registry.PBDRV4ModelRegistryError, "parent_checkpoint sha256"):
            self._stage2(bad_parent, metadata)

    def test_frozen_current_reference_is_independent_and_base_logits_are_current(self) -> None:
        stage1, metadata = self._stage1()
        payload = self._payload(stage1, metadata)
        stage2, _ = self._stage2(payload, metadata)
        reference, reference_metadata = registry.build_frozen_current_reference_model(
            "NUAA-SIRST",
            "best_miou",
            "stage2",
        )
        self.assertIsNot(reference, stage2)
        self.assertFalse(reference.training)
        self.assertFalse(reference.pbdr_v4.training)
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))
        self.assertTrue(reference_metadata["base_logits_are_current"])

        generator = torch.Generator().manual_seed(2026080701)
        image = torch.randn(1, 1, 32, 32, generator=generator)
        self.current_model.eval()
        self.current_model.mode = "train"
        with torch.no_grad():
            current_probabilities = self.current_model(image)
            _, reference_before = reference.forward_for_pbdr_v4_training(image)
        self.assertTrue(
            torch.equal(
                torch.sigmoid(reference_before.candidate_base_logits),
                current_probabilities[-1],
            )
        )
        with torch.no_grad():
            stage2.outc.bias.add_(0.25)
            _, candidate_after = stage2.forward_for_pbdr_v4_training(image)
            _, reference_after = reference.forward_for_pbdr_v4_training(image)
        self.assertTrue(
            torch.equal(
                reference_before.candidate_base_logits,
                reference_after.candidate_base_logits,
            )
        )
        self.assertFalse(
            torch.equal(
                candidate_after.candidate_base_logits,
                reference_after.candidate_base_logits,
            )
        )

    def test_bn_buffer_tamper_is_rejected_during_inference_export(self) -> None:
        stage1, metadata = self._stage1()
        stage1_payload = self._payload(stage1, metadata)
        stage2, _ = self._stage2(stage1_payload, metadata)
        stage2_payload = self._payload(
            stage2,
            metadata,
            stage="stage2",
        )
        buffer_name = next(
            name
            for name, _ in stage2.named_buffers()
            if name.startswith("up_decoder1.") and name.endswith("running_mean")
        )
        bad = dict(stage2_payload)
        bad_state = {
            name: tensor.detach().clone()
            for name, tensor in stage2_payload["state_dict"].items()
        }
        bad_state[buffer_name].add_(1.0)
        bad["state_dict"] = bad_state
        bad["state_sha256"] = registry.state_semantic_sha256(bad_state)
        with self.assertRaisesRegex(RuntimeError, "buffer differs from Current"):
            registry.build_candidate_inference_model(
                bad,
                dataset_name="NUAA-SIRST",
                role="best_miou",
                stage="stage2",
                expected_source_sha256=SOURCE_SHA,
                expected_split_sha256=SPLIT_SHA,
                expected_atlas_sha256=ATLAS_SHA,
                expected_initialization_sha256=str(metadata["initialization_sha256"]),
            )

    def test_complete_candidate_payload_exports_strict_inference_graph(self) -> None:
        stage1, metadata = self._stage1()
        payload = self._payload(stage1, metadata)
        self.assertEqual(payload["schema"], PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA)
        inference, export = registry.build_candidate_inference_model(
            payload,
            dataset_name="NUAA-SIRST",
            role="best_miou",
            stage="stage1",
            expected_source_sha256=SOURCE_SHA,
            expected_split_sha256=SPLIT_SHA,
            expected_atlas_sha256=ATLAS_SHA,
            expected_initialization_sha256=str(metadata["initialization_sha256"]),
            expected_state_sha256=str(payload["state_sha256"]),
        )
        self.assertFalse(inference.training)
        self.assertEqual(inference.mode, "test")
        self.assertEqual(len(inference.state_dict()), registry.INFERENCE_STATE_KEY_COUNT)
        self.assertTrue(export["strict_complete_payload"])
        self.assertTrue(export["integration_export"]["strict_training_state_load"])
        self.assertTrue(export["integration_export"]["strict_inference_state_load"])

        bad_role = dict(payload)
        bad_role["role"] = "best_pd"
        with self.assertRaisesRegex(registry.PBDRV4ModelRegistryError, "role differs"):
            registry.build_candidate_inference_model(
                bad_role,
                dataset_name="NUAA-SIRST",
                role="best_miou",
                stage="stage1",
                expected_source_sha256=SOURCE_SHA,
                expected_split_sha256=SPLIT_SHA,
                expected_atlas_sha256=ATLAS_SHA,
                expected_initialization_sha256=str(metadata["initialization_sha256"]),
            )


if __name__ == "__main__":
    unittest.main()
