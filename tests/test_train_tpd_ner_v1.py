from __future__ import annotations

import unittest
from unittest import mock

import torch

from experiments import evaluate_pd_fa_sweep as sweep_base
from experiments import evaluate_tpd_ner_v1_pd_fa as ner_sweep
from experiments import train_tpd_pilot as base
from experiments.train_tpd_ner_v1 import (
    CONSTRUCTION_SCHEMA,
    PARENT_VARIANT,
    RELAY_WIDTH,
    SUPPORTED_NER_VARIANTS,
    build_tpd_ner_model,
)
from model.tpd_sctransnet import EVIDENCE_NODE_NAMES, TPDSCTransNet


class TPDNERBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 73
        cls.spd, cls.spd_metadata = base.build_model("spd", cls.seed)
        cls.candidate, cls.metadata = build_tpd_ner_model(
            "tpd_clean_full_ner",
            cls.seed,
        )

    def test_builder_records_locked_innovation_contract(self) -> None:
        metadata = self.metadata
        self.assertEqual(SUPPORTED_NER_VARIANTS, ("tpd_clean_full_ner",))
        self.assertEqual(metadata["variant"], "tpd_clean_full_ner")
        self.assertEqual(metadata["parent_variant"], PARENT_VARIANT)
        self.assertEqual(metadata["construction_schema"], CONSTRUCTION_SCHEMA)
        self.assertIsInstance(self.candidate, TPDSCTransNet)
        self.assertEqual(metadata["model_class"], "TPDSCTransNet")
        self.assertEqual(metadata["tensor_handoff"], "forward_local_explicit")
        self.assertEqual(
            metadata["relay_taps"]["evidence_nodes"],
            EVIDENCE_NODE_NAMES,
        )
        self.assertEqual(metadata["relay_width"], RELAY_WIDTH)
        self.assertEqual(metadata["relay_topology"], "q4->q3->q2")
        self.assertEqual(metadata["relay_parameters"], 11_291)
        self.assertEqual(metadata["relay_gate_parameters"], 27)
        self.assertEqual(metadata["shallow_embedding_parameters"], 66_496)
        self.assertFalse(
            metadata["innovation_contract"]["primary_module_replaced"]
        )
        self.assertEqual(
            metadata["shared_initialization_sha256"],
            self.spd_metadata["shared_initialization_sha256"],
        )
        self.assertEqual(
            metadata["total_parameters"],
            metadata["trainable_parameters"],
        )

    def test_initial_full_model_is_exactly_spd_equivalent(self) -> None:
        self.spd.eval()
        self.candidate.eval()
        torch.manual_seed(101)
        inputs = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            expected = self.spd(inputs)
            actual = self.candidate(inputs)
        self.assertEqual(len(expected), 6)
        self.assertEqual(len(actual), 6)
        for expected_output, actual_output in zip(expected, actual):
            self.assertTrue(torch.equal(expected_output, actual_output))

    def test_checkpoint_rebuild_is_strict_and_deterministic(self) -> None:
        state = self.candidate.state_dict()
        rebuilt, rebuilt_metadata = build_tpd_ner_model(
            "tpd_clean_full_ner",
            self.seed + 1,
        )
        self.assertNotEqual(
            rebuilt_metadata["full_initialization_sha256"],
            self.metadata["full_initialization_sha256"],
        )
        self.assertEqual(set(state), set(rebuilt.state_dict()))
        source_checksum = base.model_checksum(self.candidate)
        rebuilt_checksum = base.model_checksum(rebuilt)
        self.assertNotEqual(rebuilt_checksum, source_checksum)

        self.candidate.eval()
        rebuilt.eval()
        torch.manual_seed(211)
        inputs = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            expected = self.candidate(inputs)
            before_load = rebuilt(inputs)
        self.assertTrue(
            any(
                not torch.equal(expected_output, before_output)
                for expected_output, before_output in zip(expected, before_load)
            )
        )

        rebuilt.load_state_dict(state, strict=True)
        self.assertEqual(
            base.model_checksum(rebuilt),
            source_checksum,
        )
        with torch.no_grad():
            actual = rebuilt(inputs)
        for expected_output, actual_output in zip(expected, actual):
            self.assertTrue(torch.equal(expected_output, actual_output))

    def test_pd_fa_wrapper_binds_the_ner_builder(self) -> None:
        original_builder = sweep_base.build_model
        try:
            with mock.patch.object(sweep_base, "main") as shared_main:
                ner_sweep.main()
                shared_main.assert_called_once_with()
                self.assertIs(sweep_base.build_model, build_tpd_ner_model)
        finally:
            sweep_base.build_model = original_builder

    def test_builder_rejects_every_non_ner_variant(self) -> None:
        for variant in ("spd", "tpd_clean_full", "original"):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "Unknown TPD-NER"):
                    build_tpd_ner_model(variant, self.seed)


if __name__ == "__main__":
    unittest.main()
