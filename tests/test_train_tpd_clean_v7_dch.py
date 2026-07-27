from __future__ import annotations

import unittest
from unittest import mock

import torch

from experiments import train_tpd_clean_v7_dch as entry
from experiments import train_tpd_pilot as base
from model.tpd_clean_v7_dch import (
    PRIMARY_CLEAN_V7_DCH_VARIANT,
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
    TPDCleanV7DCHPatchEmbedding,
)


class TrainTPDCleanV7DCHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))
        cls.models = {}
        cls.metadata = {}
        for variant in SUPPORTED_CLEAN_V7_DCH_VARIANTS:
            model, metadata = entry.build_clean_v7_dch_model(variant, seed=42)
            cls.models[variant] = model
            cls.metadata[variant] = metadata

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_import_has_no_runner_rebinding_side_effect(self) -> None:
        self.assertIsNot(base.build_model, entry.build_clean_v7_dch_model)
        self.assertNotEqual(
            tuple(base.SUPPORTED_VARIANTS),
            SUPPORTED_CLEAN_V7_DCH_VARIANTS,
        )

    def test_builder_owns_only_the_frozen_dch_pair(self) -> None:
        self.assertEqual(
            SUPPORTED_CLEAN_V7_DCH_VARIANTS,
            (
                "tpd_clean_v7_dch_full",
                "tpd_clean_v7_dch_capacity",
            ),
        )
        self.assertEqual(
            PRIMARY_CLEAN_V7_DCH_VARIANT,
            "tpd_clean_v7_dch_full",
        )
        for model in self.models.values():
            self.assertIsInstance(
                model.mtc.embeddings_1,
                TPDCleanV7DCHPatchEmbedding,
            )
            self.assertIsInstance(
                model.mtc.embeddings_2,
                TPDCleanV7DCHPatchEmbedding,
            )
            self.assertEqual(len(model.mtc.embeddings_1.blocks), 4)
            self.assertEqual(len(model.mtc.embeddings_2.blocks), 3)

    def test_pair_has_identical_initial_state_and_capacity(self) -> None:
        full = self.models["tpd_clean_v7_dch_full"].state_dict()
        capacity = self.models["tpd_clean_v7_dch_capacity"].state_dict()
        self.assertEqual(tuple(full), tuple(capacity))
        for name in full:
            with self.subTest(name=name):
                self.assertTrue(torch.equal(full[name], capacity[name]))

        metadata = tuple(self.metadata.values())
        self.assertEqual(
            {item["total_parameters"] for item in metadata},
            {entry.TOTAL_PARAMETERS},
        )
        self.assertEqual(
            {item["trainable_parameters"] for item in metadata},
            {entry.TOTAL_PARAMETERS},
        )
        self.assertEqual(
            {item["shallow_embedding_parameters"] for item in metadata},
            {entry.SHALLOW_EMBEDDING_PARAMETERS},
        )
        self.assertEqual(
            len({item["full_initialization_sha256"] for item in metadata}),
            1,
        )

    def test_metadata_has_dch_identity_and_formula(self) -> None:
        full = self.metadata["tpd_clean_v7_dch_full"]
        capacity = self.metadata["tpd_clean_v7_dch_capacity"]
        self.assertEqual(
            full["candidate_family"],
            "spd_anchored_tpd_clean_v7_deferred_context_headroom",
        )
        self.assertEqual(
            full["context_headroom_formula"],
            entry.FULL_HEADROOM_FORMULA,
        )
        self.assertEqual(
            capacity["context_headroom_formula"],
            entry.CAPACITY_HEADROOM_FORMULA,
        )
        self.assertEqual(full["context_gate"], 1.0)
        self.assertEqual(capacity["context_gate"], 0.0)
        self.assertEqual(
            capacity["context_code"],
            "not_computed_in_capacity_forward",
        )
        for item in (full, capacity):
            self.assertEqual(item["mainline_contract"], "Keep-Context-Saliency")
            self.assertEqual(
                item["semantic_sources"],
                ("Keep", "Context", "Saliency"),
            )
            self.assertFalse(item["fourth_parallel_branch_added"])
            self.assertEqual(
                item["replaced_embeddings"],
                ("mtc.embeddings_1", "mtc.embeddings_2"),
            )
            self.assertEqual(item["headroom_bound"], "0.75<=H<=1.25")
            self.assertEqual(
                item["zero_scale_first_order_reference"],
                "capacity_exact",
            )
            self.assertEqual(
                item["context_residual_near_zero_order"],
                "O(abs(a)^2)",
            )
            self.assertFalse(item["second_order_differentiability_claimed"])

    def test_main_rebinds_only_at_explicit_call(self) -> None:
        with (
            mock.patch.object(
                base,
                "SUPPORTED_VARIANTS",
                base.SUPPORTED_VARIANTS,
            ),
            mock.patch.object(base, "build_model", base.build_model),
            mock.patch.object(base, "main") as delegated,
        ):
            entry.main()
            self.assertEqual(
                base.SUPPORTED_VARIANTS,
                SUPPORTED_CLEAN_V7_DCH_VARIANTS,
            )
            self.assertIs(base.build_model, entry.build_clean_v7_dch_model)
            delegated.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
