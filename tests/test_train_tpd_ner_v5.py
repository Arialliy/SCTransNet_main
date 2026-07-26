from __future__ import annotations

import unittest
from unittest import mock

import torch

from experiments import train_tpd_ner_v5 as entry
from experiments import train_tpd_pilot as base
from experiments import tpd_ner_runtime as runtime
from experiments.train_tpd_ner_v5 import (
    CONSTRUCTION_SCHEMA,
    SUPPORTED_TPD_NER_V5_VARIANTS,
    build_tpd_ner_v5_model,
)
from model.Config import get_SCTrans_config
from model.tpd_clean_v5 import PRIMARY_CLEAN_V5_VARIANT
from model.tpd_ner_v5 import (
    EVIDENCE_NODE_NAMES,
    PROGRESSIVE_TOKENIZER,
    TPDNERV5SCTransNet,
)


torch.set_num_threads(1)


def small_config():
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    return config


class TPDNERV5BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 73
        cls.models = {}
        cls.metadata = {}
        for variant in SUPPORTED_TPD_NER_V5_VARIANTS:
            model, metadata = build_tpd_ner_v5_model(
                variant,
                cls.seed,
                config=small_config(),
                img_size=32,
                relay_width=2,
            )
            cls.models[variant] = model
            cls.metadata[variant] = metadata

    def test_builder_exposes_the_complete_isolated_four_variant_matrix(self) -> None:
        self.assertEqual(
            SUPPORTED_TPD_NER_V5_VARIANTS,
            (
                "tpd_clean_v5_full_relay_off",
                "tpd_clean_v5_full_relay_on",
                "progressive_relay_off",
                "progressive_relay_on",
            ),
        )
        backbone_hashes = set()
        shallow_parameter_counts = set()
        for variant in SUPPORTED_TPD_NER_V5_VARIANTS:
            with self.subTest(variant=variant):
                model = self.models[variant]
                metadata = self.metadata[variant]
                self.assertIsInstance(model, TPDNERV5SCTransNet)
                self.assertEqual(metadata["variant"], variant)
                self.assertEqual(
                    metadata["construction_schema"],
                    CONSTRUCTION_SCHEMA,
                )
                self.assertEqual(
                    metadata["relay_taps"]["evidence_nodes"],
                    EVIDENCE_NODE_NAMES,
                )
                self.assertFalse(metadata["fourth_parallel_branch_added"])
                self.assertEqual(metadata["tensor_handoff"], "forward_local_explicit")
                self.assertEqual(metadata["relay_topology"], "q4->q3->q2")
                self.assertFalse(metadata["automatic_launch"])
                self.assertEqual(metadata["gate_connection"], "none")
                self.assertEqual(metadata["initialization_mode"], "fresh_shared_seed")
                self.assertFalse(metadata["warm_start_applied"])
                self.assertEqual(
                    metadata["total_parameters"],
                    metadata["trainable_parameters"],
                )
                backbone_hashes.add(metadata["backbone_initialization_sha256"])
                shallow_parameter_counts.add(
                    metadata["shallow_embedding_parameters"]
                )
                self.assertTrue(metadata["shallow_parameter_match_verified"])
                self.assertEqual(
                    metadata["shallow_embedding_parameters"],
                    metadata["matched_reference_shallow_parameters"],
                )
        self.assertEqual(len(backbone_hashes), 1)
        self.assertEqual(len(shallow_parameter_counts), 1)

    def test_v5_metadata_locks_three_sources_and_one_scale(self) -> None:
        for suffix in ("off", "on"):
            variant = f"tpd_clean_v5_full_relay_{suffix}"
            metadata = self.metadata[variant]
            self.assertEqual(
                metadata["tokenizer_variant"],
                PRIMARY_CLEAN_V5_VARIANT,
            )
            self.assertEqual(
                metadata["semantic_sources"],
                ("Keep", "Context", "Saliency"),
            )
            self.assertEqual(metadata["semantic_source_count"], 3)
            self.assertEqual(metadata["learned_scales_per_block"], 1)
            self.assertEqual(
                metadata["context_selector"],
                "positive_centered_0p5_to_1p5",
            )
            self.assertEqual(
                metadata["capacity_contract"],
                "v5_full_capacity_reference",
            )
            scale_names = [
                name
                for name, _ in self.models[variant].named_parameters()
                if name.endswith("saliency_scale")
            ]
            self.assertEqual(len(scale_names), 7)
            self.assertFalse(
                any("context_scale" in name for name in self.models[variant].state_dict())
            )

        for suffix in ("off", "on"):
            variant = f"progressive_relay_{suffix}"
            metadata = self.metadata[variant]
            self.assertEqual(metadata["tokenizer_variant"], PROGRESSIVE_TOKENIZER)
            self.assertEqual(
                metadata["capacity_contract"],
                "same_depth_strict_shallow_parameter_match_to_v5",
            )
            self.assertEqual(metadata["learned_scales_per_block"], 1)
            self.assertEqual(metadata["learned_capacity_gains_per_block"], 1)
            topology = metadata["progressive_topology"]
            self.assertEqual(topology["embedding_depths"], (4, 3))
            self.assertEqual(
                topology["spatial_projection"],
                "Conv2d(C,C,kernel=2,stride=2,bias=True)",
            )
            self.assertTrue(topology["all_capacity_parameters_forward_active"])

    def test_off_on_pairs_have_exact_common_state_and_step0_outputs(self) -> None:
        for stem in ("tpd_clean_v5_full", "progressive"):
            off_variant = f"{stem}_relay_off"
            on_variant = f"{stem}_relay_on"
            off = self.models[off_variant]
            on = self.models[on_variant]
            off_state = off.state_dict()
            on_state = on.state_dict()
            extra_keys = set(on_state) - set(off_state)
            self.assertFalse(set(off_state) - set(on_state))
            self.assertTrue(extra_keys)
            self.assertTrue(all(key.startswith("tpd_ner.") for key in extra_keys))
            for key, tensor in off_state.items():
                with self.subTest(pair=stem, state=key):
                    self.assertTrue(torch.equal(tensor, on_state[key]))

            off_parameters = dict(off.named_parameters())
            on_parameters = dict(on.named_parameters())
            extra_parameter_names = set(on_parameters) - set(off_parameters)
            self.assertTrue(
                all(
                    name.startswith("tpd_ner.")
                    for name in extra_parameter_names
                )
            )
            extra_parameter_count = sum(
                on_parameters[name].numel() for name in extra_parameter_names
            )
            self.assertEqual(
                extra_parameter_count,
                self.metadata[on_variant]["relay_parameters"],
            )
            self.assertEqual(
                self.metadata[off_variant]["common_initialization_sha256"],
                self.metadata[on_variant]["common_initialization_sha256"],
            )
            self.assertEqual(
                self.metadata[off_variant]["common_parameters"],
                self.metadata[on_variant]["common_parameters"],
            )
            self.assertEqual(
                self.metadata[on_variant]["total_parameters"]
                - self.metadata[off_variant]["total_parameters"],
                self.metadata[on_variant]["relay_parameters"],
            )

            off.eval()
            on.eval()
            inputs = torch.randn(2, 1, 32, 64)
            with torch.no_grad():
                expected = off(inputs)
                actual = on(inputs)
            self.assertEqual(len(expected), 6)
            self.assertEqual(len(actual), 6)
            for expected_output, actual_output in zip(expected, actual):
                self.assertTrue(torch.equal(expected_output, actual_output))

    def test_t_and_p_on_use_identical_relay_initialization(self) -> None:
        tpd = self.models["tpd_clean_v5_full_relay_on"]
        progressive = self.models["progressive_relay_on"]
        tpd_relay_state = tpd.tpd_ner.state_dict()
        progressive_relay_state = progressive.tpd_ner.state_dict()
        self.assertEqual(set(tpd_relay_state), set(progressive_relay_state))
        for key, tensor in tpd_relay_state.items():
            with self.subTest(state=key):
                self.assertTrue(torch.equal(tensor, progressive_relay_state[key]))
        self.assertEqual(
            self.metadata["tpd_clean_v5_full_relay_on"][
                "relay_initialization_sha256"
            ],
            self.metadata["progressive_relay_on"][
                "relay_initialization_sha256"
            ],
        )

    def test_checkpoint_rebuild_loads_strictly_with_stable_keys(self) -> None:
        source = self.models["tpd_clean_v5_full_relay_on"]
        source.eval()
        state = source.state_dict()
        rebuilt, rebuilt_metadata = build_tpd_ner_v5_model(
            "tpd_clean_v5_full_relay_on",
            self.seed + 1,
            config=small_config(),
            img_size=32,
            relay_width=2,
        )
        self.assertEqual(set(state), set(rebuilt.state_dict()))
        self.assertNotEqual(
            self.metadata["tpd_clean_v5_full_relay_on"][
                "full_initialization_sha256"
            ],
            rebuilt_metadata["full_initialization_sha256"],
        )
        rebuilt.load_state_dict(state, strict=True)
        rebuilt.eval()
        inputs = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            expected = source(inputs)
            actual = rebuilt(inputs)
        for expected_output, actual_output in zip(expected, actual):
            self.assertTrue(torch.equal(expected_output, actual_output))

    def test_main_delegates_only_when_called_explicitly(self) -> None:
        original_variants = base.SUPPORTED_VARIANTS
        original_builder = base.build_model
        original_loss = base.deep_supervision_loss
        original_validate = base.validate
        original_save = torch.save
        observed_runtime: list[bool] = []

        def capture_runtime() -> None:
            observed_runtime.append(True)
            self.assertIs(
                base.deep_supervision_loss,
                runtime.checked_deep_supervision_loss,
            )
            self.assertIs(base.validate, runtime.checked_validate)
            self.assertIs(torch.save, runtime.atomic_torch_save)

        try:
            with mock.patch.object(
                base,
                "main",
                side_effect=capture_runtime,
            ) as shared_main:
                entry.main()
                shared_main.assert_called_once_with()
                self.assertEqual(observed_runtime, [True])
                self.assertEqual(
                    base.SUPPORTED_VARIANTS,
                    SUPPORTED_TPD_NER_V5_VARIANTS,
                )
                self.assertIs(base.build_model, build_tpd_ner_v5_model)
                self.assertIs(base.deep_supervision_loss, original_loss)
                self.assertIs(base.validate, original_validate)
                self.assertIs(torch.save, original_save)
        finally:
            base.SUPPORTED_VARIANTS = original_variants
            base.build_model = original_builder

    def test_builder_rejects_unknown_variants(self) -> None:
        for variant in (
            "tpd_clean_v5_sal_capacity_relay_on",
            "tpd_clean_full_ner",
            "original",
        ):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "Unknown V5-NER"):
                    build_tpd_ner_v5_model(
                        variant,
                        self.seed,
                        config=small_config(),
                        img_size=32,
                        relay_width=2,
                    )


if __name__ == "__main__":
    unittest.main()
