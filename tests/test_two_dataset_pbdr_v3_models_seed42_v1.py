from __future__ import annotations

import gc
import unittest
from unittest import mock

import torch

from experiments import two_dataset_pbdr_v3_models_seed42_v1 as models
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3 import (
    PBDR_V3_STATE_KEYS,
)


torch.set_num_threads(1)


class FrozenArtifactAuditTests(unittest.TestCase):
    def test_scope_and_manifest_hashes_are_explicit(self) -> None:
        self.assertEqual(models.DATASETS, ("NUDT-SIRST", "IRSTD-1K"))
        self.assertEqual(models.PARENT_ROLES, ("best_miou", "best_pd"))
        self.assertEqual(models.TRAINING_SEED, 42)
        current = models.load_frozen_current_manifest()
        original = models.load_frozen_original_manifest()
        self.assertEqual(set(current["datasets"]), set(models.DATASETS))
        self.assertEqual(set(original["datasets"]), set(models.DATASETS))
        self.assertEqual(
            original["authority_manifest"]["sha256"],
            models.ORIGINAL_AUTHORITY_MANIFEST_SHA256,
        )

    def test_manifest_hash_tampering_is_rejected_before_artifact_use(self) -> None:
        with mock.patch.object(models, "CURRENT_MANIFEST_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                models.CrossDatasetPBDRV3ModelProtocolError,
                "manifest SHA-256 differs",
            ):
                models.load_frozen_current_manifest()
        with mock.patch.object(models, "ORIGINAL_MANIFEST_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                models.CrossDatasetPBDRV3ModelProtocolError,
                "manifest SHA-256 differs",
            ):
                models.load_frozen_original_manifest()

    def test_all_current_dataset_role_bindings_pass_file_and_state_audit(self) -> None:
        expected = {
            ("NUDT-SIRST", "best_miou"): (
                "0f5f6a5fe96fa86302807d132078d575495a3aff6690967785868a23400f3e84",
                "c10eebf67f3f25bd252881b61b5e3f9ad0a850dc594f79c451eb53325101e2da",
            ),
            ("NUDT-SIRST", "best_pd"): (
                "227d8133ac7427200daae8b2ed0920d19517a2ac9dda6d1c7f15f14cbac06929",
                "0d56ae2f268e29155b3beceed2bfb5541e100a93b6a5b769129501803c4745b4",
            ),
            ("IRSTD-1K", "best_miou"): (
                "e8e9401500502dda0bbdc9640b830a7934fb2bc97bde706fde9adca216d965b4",
                "d7600f61ee3d0967dae899de72a28f2e7e9e4c6381f2687189e45d84dcb3e298",
            ),
            ("IRSTD-1K", "best_pd"): (
                "f667376d930a933c905d80a02938f85ba8323017e5715df30ef40e6ef93875a2",
                "14abd9b707a443d7694c4e371cf7d85fb3c54e420bbaa0c64dbfe65529533314",
            ),
        }
        for (dataset, role), hashes in expected.items():
            with self.subTest(dataset=dataset, role=role):
                _, state, record = models.load_current_checkpoint(dataset, role)
                self.assertEqual(len(state), models.CURRENT_STATE_KEY_COUNT)
                self.assertEqual(
                    (record["sha256"], record["state_sha256"]), hashes
                )
                audit = record["current_run_audit"]
                self.assertEqual(len(audit["historical_runtime_sources"]), 35)
                self.assertFalse(audit["official_test_data_accessed"])
                self.assertFalse(audit["training_started"])

    def test_all_original_dataset_role_bindings_match_authority(self) -> None:
        expected = {
            ("NUDT-SIRST", "best_miou"): (
                "4f9d5132a9b1a8c62cfd4ae5bf6c4b43258e513973916a12465884ffea728c5e",
                "33d06a91515926379aae5440eae7cbd5742a0360694c4b0adaf5e59c93c08e82",
            ),
            ("NUDT-SIRST", "best_pd"): (
                "0c4c2990c1e99b82832e3ebe1f5d7af009819a566156475aff083f9289a5ff81",
                "bd12a5ef040bea5fd0c20ebd6b405f53499d8d84508e3c101d63806e94b03c0c",
            ),
            ("IRSTD-1K", "best_miou"): (
                "b82795652ec3ee11d28cd5703f737615676793ed63f653e952989518e472b6ba",
                "60113bc3c8baa02e44ea61e8ca63bfac25ee5554de4c9cf14776b3f84475ce47",
            ),
            ("IRSTD-1K", "best_pd"): (
                "09b3c399e6a4d0a75133a78204ea4e29864f7b108142b0e46d4e72e92429478a",
                "e08da4d1674850136bcc083d48dfcfa4580e0561567232c3ea818e77b0a46da4",
            ),
        }
        for (dataset, role), hashes in expected.items():
            with self.subTest(dataset=dataset, role=role):
                _, state, record = models.load_original_checkpoint(dataset, role)
                self.assertEqual(len(state), models.ORIGINAL_STATE_KEY_COUNT)
                self.assertEqual(
                    (record["sha256"], record["state_sha256"]), hashes
                )
                self.assertEqual(
                    record["fixed_threshold_0_5_metrics"]["threshold"],
                    0.5,
                )
                self.assertFalse(record["official_test_data_accessed"])
                self.assertEqual(
                    record["selection_policy"][role + "_order"][0],
                    "higher_miou" if role == "best_miou" else "higher_pd",
                )

    def test_unknown_dataset_or_role_is_rejected(self) -> None:
        with self.assertRaises(models.CrossDatasetPBDRV3ModelProtocolError):
            models.load_current_checkpoint("NUAA-SIRST", "best_miou")
        with self.assertRaises(models.CrossDatasetPBDRV3ModelProtocolError):
            models.load_original_checkpoint("NUDT-SIRST", "latest")


class CrossDatasetGraphBuilderTests(unittest.TestCase):
    def test_stage1_and_candidate_inference_are_bound_to_exact_parent(self) -> None:
        training, metadata = models.build_stage1_training_model(
            "NUDT-SIRST", "best_miou"
        )
        try:
            self.assertEqual(
                len(training.state_dict()), models.TRAINING_STATE_KEY_COUNT
            )
            self.assertTrue(metadata["all_current_tensors_bitwise_equal_after_load"])
            self.assertEqual(metadata["dataset"], "NUDT-SIRST")
            self.assertEqual(metadata["parent_role"], "best_miou")
            self.assertEqual(
                metadata["stage1_freeze_audit"]["trainable_parameter_count"],
                6018,
            )
            self.assertEqual(
                tuple(
                    name
                    for name, parameter in training.named_parameters()
                    if parameter.requires_grad
                ),
                tuple(
                    name
                    for name, _ in training.named_parameters()
                    if name in set(PBDR_V3_STATE_KEYS)
                ),
            )

            inference, inference_metadata = (
                models.build_inference_model_from_candidate_state(
                    training.state_dict(),
                    dataset_name="NUDT-SIRST",
                    parent_role="best_miou",
                )
            )
            try:
                self.assertEqual(
                    len(inference.state_dict()), models.INFERENCE_STATE_KEY_COUNT
                )
                self.assertTrue(inference_metadata["strict_load"])
                self.assertTrue(
                    inference_metadata["base_bitwise_equal_to_parent"]
                )
                self.assertFalse(inference.training)
                self.assertEqual(inference.mode, "test")
            finally:
                del inference

            with self.assertRaisesRegex(
                models.CrossDatasetPBDRV3ModelProtocolError,
                "candidate modified frozen Current tensors",
            ):
                models.build_inference_model_from_candidate_state(
                    training.state_dict(),
                    dataset_name="IRSTD-1K",
                    parent_role="best_miou",
                )
        finally:
            del training
            gc.collect()

    def test_original_inference_is_strict_and_test_ready(self) -> None:
        model, metadata = models.build_original_inference_model(
            "IRSTD-1K", "best_pd"
        )
        try:
            self.assertEqual(len(model.state_dict()), models.ORIGINAL_STATE_KEY_COUNT)
            self.assertTrue(metadata["strict_load"])
            self.assertFalse(metadata["target_survival_registered"])
            self.assertFalse(metadata["official_test_data_accessed"])
            self.assertFalse(metadata["evaluation_started"])
            self.assertFalse(model.training)
            self.assertEqual(model.mode, "test")
            self.assertEqual(
                metadata["state_sha256"],
                "e08da4d1674850136bcc083d48dfcfa4580e0561567232c3ea818e77b0a46da4",
            )
        finally:
            del model
            gc.collect()


if __name__ == "__main__":
    unittest.main()
