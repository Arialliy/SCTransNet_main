from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_ner_v7_dch as builder
from experiments.train_tpd_clean_v7_dch import build_clean_v7_dch_model
from experiments.train_tpd_pilot import deep_supervision_loss
from model.Config import get_SCTrans_config
from model.tpd_clean_v7_dch import PRIMARY_CLEAN_V7_DCH_VARIANT
from model import tpd_ner_v7_dch as model_module


torch.set_num_threads(1)


def small_config():
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    return config


def build_small(
    variant: str,
    *,
    seed: int = 101,
):
    return builder.build_tpd_ner_v7_dch_model(
        variant,
        seed,
        config=small_config(),
        img_size=32,
        relay_width=2,
    )


class TPDNERV7DCHModelTests(unittest.TestCase):
    def test_source_has_no_v5_concrete_dependency(self) -> None:
        source = inspect.getsource(model_module)
        self.assertNotIn("model.tpd_ner_v5", source)
        self.assertNotIn("TPDCleanV5Block", source)

    def test_five_node_geometry_and_endpoint_identity(self) -> None:
        model, metadata = build_small(builder.V7_FULL_RELAY_OFF)
        self.assertEqual(metadata["evidence_layout"], (3, 2))
        self.assertFalse(metadata["fourth_parallel_branch_added"])
        self.assertEqual(
            metadata["semantic_sources"],
            ("Keep", "Context", "Saliency"),
        )

        input1 = torch.randn(2, 4, 64, 96)
        input2 = torch.randn(2, 8, 32, 48)
        for embedding, inputs, expected_nodes, expected_endpoint in (
            (
                model.mtc.embeddings_1,
                input1,
                ((2, 4, 32, 48), (2, 4, 16, 24), (2, 4, 8, 12)),
                (2, 4, 4, 6),
            ),
            (
                model.mtc.embeddings_2,
                input2,
                ((2, 8, 16, 24), (2, 8, 8, 12)),
                (2, 8, 4, 6),
            ),
        ):
            endpoint, evidence = embedding.forward_with_evidence(inputs)
            plain_endpoint = embedding(inputs)
            self.assertIsInstance(evidence, tuple)
            self.assertTrue(torch.equal(endpoint, plain_endpoint))
            self.assertEqual(
                tuple(tuple(node.shape) for node in evidence),
                expected_nodes,
            )
            self.assertEqual(tuple(endpoint.shape), expected_endpoint)
            self.assertTrue(
                all(node.dtype == inputs.dtype for node in evidence)
            )
            self.assertTrue(
                all(node.device == inputs.device for node in evidence)
            )

    def test_relay_pairs_have_exact_common_state_and_step_zero_output(
        self,
    ) -> None:
        pairs = (
            (builder.V7_FULL_RELAY_OFF, builder.V7_FULL_RELAY_ON),
            (builder.PROGRESSIVE_RELAY_OFF, builder.PROGRESSIVE_RELAY_ON),
        )
        inputs = torch.randn(2, 1, 32, 64)
        for off_name, on_name in pairs:
            with self.subTest(pair=(off_name, on_name)):
                off, off_metadata = build_small(off_name)
                on, on_metadata = build_small(on_name)
                off_state = off.state_dict()
                on_state = on.state_dict()
                self.assertEqual(
                    set(on_state) - set(off_state),
                    {
                        key
                        for key in on_state
                        if key.startswith("tpd_ner.")
                    },
                )
                self.assertFalse(set(off_state) - set(on_state))
                for key, value in off_state.items():
                    self.assertTrue(torch.equal(value, on_state[key]), key)
                self.assertEqual(
                    off_metadata["common_initialization_sha256"],
                    on_metadata["common_initialization_sha256"],
                )

                off.eval()
                on.eval()
                with torch.no_grad():
                    off_outputs = off(inputs)
                    on_outputs = on(inputs)
                self.assertEqual(len(off_outputs), 6)
                self.assertTrue(
                    all(
                        torch.equal(off_output, on_output)
                        for off_output, on_output in zip(
                            off_outputs,
                            on_outputs,
                        )
                    )
                )

    def test_builder_matches_frozen_v7_common_state(self) -> None:
        frozen, frozen_metadata = build_clean_v7_dch_model(
            PRIMARY_CLEAN_V7_DCH_VARIANT,
            42,
        )
        composed, composed_metadata = (
            builder.build_tpd_ner_v7_dch_model(
                builder.V7_FULL_RELAY_OFF,
                42,
            )
        )
        frozen_state = frozen.state_dict()
        composed_state = composed.state_dict()
        self.assertEqual(set(frozen_state), set(composed_state))
        for key, value in frozen_state.items():
            self.assertTrue(torch.equal(value, composed_state[key]), key)
        self.assertEqual(
            frozen_metadata["total_parameters"],
            composed_metadata["common_parameters"],
        )
        self.assertEqual(
            composed_metadata["total_parameters"],
            builder.EXPECTED_PRODUCTION_COMMON_PARAMETERS,
        )

    def test_parameter_groups_are_disjoint_complete_and_strict_reload(
        self,
    ) -> None:
        model, _ = build_small(builder.V7_FULL_RELAY_ON)
        groups = model.parameter_groups()
        names = {
            group: {name for name, _ in entries}
            for group, entries in groups.items()
        }
        self.assertFalse(names["backbone"] & names["tokenizer"])
        self.assertFalse(names["backbone"] & names["relay"])
        self.assertFalse(names["tokenizer"] & names["relay"])
        self.assertEqual(
            set().union(*names.values()),
            {name for name, _ in model.named_parameters()},
        )
        self.assertTrue(
            all(name.startswith("tpd_ner.") for name in names["relay"])
        )
        parameter_ids = [
            id(parameter)
            for entries in groups.values()
            for _, parameter in entries
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

        rebuilt, _ = build_small(builder.V7_FULL_RELAY_ON, seed=202)
        incompatible = rebuilt.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)

    def test_zeroing_relay_gates_preserves_trained_tokenizer(self) -> None:
        model, _ = build_small(builder.V7_FULL_RELAY_ON)
        with torch.no_grad():
            for embedding_name in ("embeddings_1", "embeddings_2"):
                for block in getattr(model.mtc, embedding_name).blocks:
                    block.saliency_scale.uniform_(-0.4, 0.4)
            for gate in model.tpd_ner.gates.values():
                gate.weight.fill_(1.0)
                gate.bias.fill_(1.0)
        tokenizer_before = {
            name: value.clone()
            for name, value in model.state_dict().items()
            if name.startswith(
                ("mtc.embeddings_1.", "mtc.embeddings_2.")
            )
        }
        model.zero_init_relay_gates()
        for name, value in tokenizer_before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        for gate in model.tpd_ner.gates.values():
            self.assertEqual(int(torch.count_nonzero(gate.weight)), 0)
            self.assertEqual(int(torch.count_nonzero(gate.bias)), 0)

    def test_two_steps_reach_gates_then_relay_fusions(self) -> None:
        model, _ = build_small(builder.V7_FULL_RELAY_ON)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss(reduction="mean")
        inputs = torch.randn(2, 1, 32, 32)
        targets = torch.rand(2, 1, 32, 32)

        optimizer.zero_grad(set_to_none=True)
        loss1 = deep_supervision_loss(model(inputs), targets, criterion)
        loss1.backward()
        self.assertTrue(torch.isfinite(loss1))
        self.assertTrue(
            all(
                gate.weight.grad is not None
                and float(gate.weight.grad.abs().sum()) > 0.0
                for gate in model.tpd_ner.gates.values()
            )
        )
        first_fusion_gradient = sum(
            0.0
            if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in model.tpd_ner.fusions.parameters()
        )
        self.assertEqual(first_fusion_gradient, 0.0)
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        loss2 = deep_supervision_loss(model(inputs), targets, criterion)
        loss2.backward()
        self.assertTrue(torch.isfinite(loss2))
        second_fusion_gradient = sum(
            0.0
            if parameter.grad is None
            else float(parameter.grad.abs().sum())
            for parameter in model.tpd_ner.fusions.parameters()
        )
        self.assertGreater(second_fusion_gradient, 0.0)

    def test_capacity_identity_is_rejected_before_state_load(self) -> None:
        payload = {
            "variant": "tpd_clean_v7_dch_capacity",
            "seed": 42,
            "checkpoint_role": "best_validation_pd_primary",
            "official_test_accessed": False,
            "run_identity": {
                "variant": "tpd_clean_v7_dch_capacity",
                "seed": 42,
            },
            "model_metadata": {
                "variant": "tpd_clean_v7_dch_capacity",
                "context_gate": 0.0,
                "mainline_contract": "Keep-Context-Saliency",
                "semantic_sources": ("Keep", "Context", "Saliency"),
                "architecture_manifest": {
                    "variant": "tpd_clean_v7_dch_capacity",
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "capacity.pth.tar"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "not V7-DCH Full"):
                builder._read_verified_parent_payload(
                    checkpoint,
                    expected_seed=42,
                    expected_checkpoint_role="best_validation_pd_primary",
                    expected_checkpoint_sha256=None,
                )

    def test_verified_parent_load_preserves_relay_state(self) -> None:
        checkpoint = Path(
            "experiments/results/"
            "tpd_clean_v7_dch_formal800_2x5090_v1/"
            "NUDT-SIRST/tpd_clean_v7_dch_full/"
            "seed_42_formal800_exact_fp32_2x5090_v1/best.pth.tar"
        )
        if not checkpoint.is_file():
            self.skipTest("formal V7-DCH Full seed-42 parent is unavailable")

        parent, _ = builder.build_tpd_ner_v7_dch_model(
            builder.V7_FULL_RELAY_OFF,
            42,
        )
        extension, _ = builder.build_tpd_ner_v7_dch_model(
            builder.V7_FULL_RELAY_ON,
            42,
        )
        relay_before = {
            name: value.clone()
            for name, value in extension.state_dict().items()
            if name.startswith("tpd_ner.")
        }
        builder.load_verified_v7_dch_full_parent(
            checkpoint,
            parent_model=parent,
            extension_model=parent,
            expected_seed=42,
            expected_checkpoint_role="best_validation_pd_primary",
        )
        provenance = builder.load_verified_v7_dch_full_parent(
            checkpoint,
            parent_model=parent,
            extension_model=extension,
            expected_seed=42,
            expected_checkpoint_role="best_validation_pd_primary",
        )
        self.assertEqual(
            provenance["transfer"]["mode"],
            "strict_parent_to_relay_extension",
        )
        for key, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, extension.state_dict()[key]))
        for key, value in relay_before.items():
            self.assertTrue(torch.equal(value, extension.state_dict()[key]))

        parent.eval()
        extension.eval()
        inputs = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            parent_outputs = parent(inputs)
            extension_outputs = extension(inputs)
        self.assertTrue(
            all(
                torch.equal(parent_output, extension_output)
                for parent_output, extension_output in zip(
                    parent_outputs,
                    extension_outputs,
                )
            )
        )

    def test_cli_is_disabled_before_gate_artifact(self) -> None:
        self.assertFalse(builder.FORMAL_LAUNCH_AUTHORIZED)
        self.assertEqual(
            builder.FORMAL_GATE_CONNECTION,
            "awaiting_v7_dch_gates_A_E",
        )
        with self.assertRaises(SystemExit):
            builder.main()


if __name__ == "__main__":
    unittest.main()
