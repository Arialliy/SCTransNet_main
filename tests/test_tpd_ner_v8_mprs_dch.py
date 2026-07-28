from __future__ import annotations

import gc
import inspect
import unittest
from unittest import mock

import torch
import torch.nn as nn

from experiments.train_tpd_clean_v8_mprs_dch import (
    TOTAL_PARAMETERS,
    build_clean_v8_mprs_dch_model,
)
from experiments.train_tpd_pilot import weights_init_kaiming
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    TPDCleanV8MPRSDCHBlock,
    TPDCleanV8MPRSDCHPatchEmbedding,
    build_clean_v8_mprs_dch_patch_embedding,
)
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_RELAY_ON_PARAMETERS,
    PRODUCTION_RELAY_PARAMETERS,
    TPDNERV8MPRSDCHSCTransNet,
    adapt_v8_mprs_dch_parent,
    relay_parameter_count,
)


FULL = "tpd_clean_v8_mprs_dch_full"
CAPACITY = "tpd_clean_v8_mprs_dch_capacity"

torch.set_num_threads(1)


def _small_config():
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    return config


def _small_parent(
    variant: str = FULL,
    *,
    seed: int = 42,
) -> SCTransNet:
    # ``seed`` is the model-construction seed for this test fixture.  Formal
    # construction and NER initialization both use the numeric value 42, but
    # NER keeps its role explicit; split_seed=20260722 belongs only to data.
    torch.manual_seed(seed)
    model = SCTransNet(
        _small_config(),
        img_size=32,
        mode="train",
        deepsuper=True,
    )
    model.apply(weights_init_kaiming)
    # The frozen production replacement helper intentionally fixes C=32/64.
    # This light fixture uses the same frozen V8 embedding builder at C=4/8.
    replacements = {
        "embeddings_1": build_clean_v8_mprs_dch_patch_embedding(
            variant,
            channels=4,
            stride=16,
        ),
        "embeddings_2": build_clean_v8_mprs_dch_patch_embedding(
            variant,
            channels=8,
            stride=8,
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    return model


def _production_parent(
    variant: str,
    *,
    seed: int = 42,
) -> SCTransNet:
    parent, _ = build_clean_v8_mprs_dch_model(
        variant,
        seed=seed,
    )
    return parent


def _adapt_pair(
    parent: SCTransNet,
    variant: str = FULL,
) -> tuple[
    TPDNERV8MPRSDCHSCTransNet,
    TPDNERV8MPRSDCHSCTransNet,
]:
    off = adapt_v8_mprs_dch_parent(
        parent,
        variant=variant,
        relay_enabled=False,
        relay_width=2,
    )
    on = adapt_v8_mprs_dch_parent(
        parent,
        variant=variant,
        relay_enabled=True,
        relay_width=2,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    return off, on


def _six_output_loss(
    outputs: object,
    targets: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("expected six deep-supervision outputs")
    criterion = nn.BCELoss(reduction="mean")
    return sum(criterion(output, targets) for output in outputs)


def _relay_fusion_gradient_l1(
    model: TPDNERV8MPRSDCHSCTransNet,
) -> float:
    total = 0.0
    for parameter in model.tpd_ner.fusions.parameters():
        if parameter.grad is None:
            raise RuntimeError("relay fusion parameter has no gradient tensor")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("relay fusion gradient is not finite")
        total += float(parameter.grad.detach().abs().sum())
    return total


def _relay_gate_gradient_l1(
    model: TPDNERV8MPRSDCHSCTransNet,
) -> float:
    total = 0.0
    for parameter in model.tpd_ner.gates.parameters():
        if parameter.grad is None:
            raise RuntimeError("relay gate parameter has no gradient tensor")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("relay gate gradient is not finite")
        total += float(parameter.grad.detach().abs().sum())
    return total


class TPDNERV8MPRSDCHTests(unittest.TestCase):
    def test_seed_roles_are_explicit(self) -> None:
        self.assertEqual(DEFAULT_RELAY_INITIALIZATION_SEED, 42)
        parent = _small_parent(seed=42)
        extension = adapt_v8_mprs_dch_parent(
            parent,
            variant=FULL,
            relay_enabled=True,
            relay_width=2,
        )
        self.assertEqual(extension.relay_initialization_seed, 42)

    def test_production_full_capacity_parameter_and_state_contract(self) -> None:
        self.assertEqual(PRODUCTION_PARENT_PARAMETERS, TOTAL_PARAMETERS)
        for variant in (FULL, CAPACITY):
            with self.subTest(variant=variant):
                parent = _production_parent(variant, seed=42)
                off = adapt_v8_mprs_dch_parent(
                    parent,
                    variant=variant,
                    relay_enabled=False,
                )
                on = adapt_v8_mprs_dch_parent(
                    parent,
                    variant=variant,
                    relay_enabled=True,
                )

                self.assertEqual(
                    sum(parameter.numel() for parameter in off.parameters()),
                    PRODUCTION_PARENT_PARAMETERS,
                )
                self.assertEqual(
                    relay_parameter_count(on),
                    PRODUCTION_RELAY_PARAMETERS,
                )
                self.assertEqual(
                    sum(parameter.numel() for parameter in on.parameters()),
                    PRODUCTION_RELAY_ON_PARAMETERS,
                )
                self.assertEqual(relay_parameter_count(off), 0)

                off_state = off.state_dict()
                on_state = on.state_dict()
                self.assertEqual(
                    set(on_state) - set(off_state),
                    {
                        name
                        for name in on_state
                        if name.startswith("tpd_ner.")
                    },
                )
                self.assertEqual(len(set(on_state) - set(off_state)), 19)
                self.assertFalse(set(off_state) - set(on_state))
                for name, value in off_state.items():
                    self.assertTrue(torch.equal(value, on_state[name]), name)

                manifest = on.architecture_manifest()
                self.assertEqual(manifest["evidence_layout"], (3, 2))
                self.assertEqual(manifest["evidence_node_count"], 5)
                self.assertEqual(
                    manifest["semantic_sources"],
                    ("Keep", "Context", "Saliency"),
                )
                self.assertFalse(manifest["fourth_parallel_branch_added"])
                self.assertFalse(
                    manifest["ordinary_forward_uses_mprs_diagnostics"]
                )
                del parent, off, on
                gc.collect()

    def test_parent_adapter_preserves_saliency_and_strict_reload(self) -> None:
        parent = _small_parent()
        with torch.no_grad():
            value = 0.03125
            for embedding_name in ("embeddings_1", "embeddings_2"):
                for block in getattr(parent.mtc, embedding_name).blocks:
                    block.saliency_scale.fill_(value)
                    value += 0.03125

        expected_parent_state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in parent.state_dict().items()
        }
        extension = adapt_v8_mprs_dch_parent(
            parent,
            variant=FULL,
            relay_enabled=True,
            relay_width=2,
        )
        extension_state = extension.state_dict()
        for name, value in expected_parent_state.items():
            self.assertTrue(torch.equal(value, extension_state[name]), name)

        saliency_before = {
            name: value.detach().cpu().contiguous()
            for name, value in extension_state.items()
            if name.endswith("saliency_scale")
        }
        with torch.no_grad():
            for gate in extension.tpd_ner.gates.values():
                gate.weight.fill_(1.0)
                if gate.bias is not None:
                    gate.bias.fill_(1.0)
        extension.zero_init_relay_gates()
        for name, value in saliency_before.items():
            self.assertTrue(
                torch.equal(value, extension.state_dict()[name]),
                name,
            )
        for gate in extension.tpd_ner.gates.values():
            self.assertEqual(int(torch.count_nonzero(gate.weight)), 0)
            if gate.bias is not None:
                self.assertEqual(int(torch.count_nonzero(gate.bias)), 0)

        rebuilt = adapt_v8_mprs_dch_parent(
            parent,
            variant=FULL,
            relay_enabled=True,
            relay_width=2,
        )
        incompatible = rebuilt.load_state_dict(
            extension.state_dict(),
            strict=True,
        )
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)

    def test_five_node_geometry_and_initial_six_output_identity(self) -> None:
        parent = _small_parent()
        off, on = _adapt_pair(parent)
        inputs = torch.randn(2, 1, 32, 32)

        on.eval()
        with torch.no_grad():
            x1 = on.inc(inputs)
            x2 = on.down_encoder1(on.pool(x1))
            x3 = on.down_encoder2(on.pool(x2))
            x4 = on.down_encoder3(on.pool(x3))
            emb1, emb2, _, _, evidence1, evidence2 = (
                on.explicit_embeddings(x1, x2, x3, x4)
            )
        self.assertEqual(
            tuple(tuple(node.shape) for node in evidence1),
            (
                (2, 4, 16, 16),
                (2, 4, 8, 8),
                (2, 4, 4, 4),
            ),
        )
        self.assertEqual(
            tuple(tuple(node.shape) for node in evidence2),
            (
                (2, 8, 8, 8),
                (2, 8, 4, 4),
            ),
        )
        self.assertEqual(tuple(emb1.shape), (2, 4, 2, 2))
        self.assertEqual(tuple(emb2.shape), (2, 8, 2, 2))

        off.eval()
        with torch.no_grad():
            off_outputs = off(inputs)
            on_outputs = on(inputs)
        self.assertIsInstance(off_outputs, tuple)
        self.assertIsInstance(on_outputs, tuple)
        self.assertEqual(len(off_outputs), 6)
        self.assertEqual(len(on_outputs), 6)
        for index, (off_output, on_output) in enumerate(
            zip(off_outputs, on_outputs)
        ):
            self.assertEqual(tuple(off_output.shape), (2, 1, 32, 32))
            self.assertTrue(
                torch.equal(off_output, on_output),
                f"step-zero output {index}",
            )

    def test_relay_forward_uses_each_v8_block_once(self) -> None:
        parent = _small_parent()
        _, model = _adapt_pair(parent)
        model.eval()

        block_calls: dict[int, int] = {}
        embedding_calls: dict[int, int] = {}
        real_block_forward = TPDCleanV8MPRSDCHBlock.forward
        real_evidence_forward = (
            TPDCleanV8MPRSDCHPatchEmbedding.forward_with_evidence
        )

        def counted_block_forward(
            block: TPDCleanV8MPRSDCHBlock,
            inputs: torch.Tensor,
        ) -> torch.Tensor:
            block_calls[id(block)] = block_calls.get(id(block), 0) + 1
            return real_block_forward(block, inputs)

        def counted_evidence_forward(
            embedding: TPDCleanV8MPRSDCHPatchEmbedding,
            inputs: torch.Tensor | None,
        ):
            embedding_calls[id(embedding)] = (
                embedding_calls.get(id(embedding), 0) + 1
            )
            return real_evidence_forward(embedding, inputs)

        def forbidden_mtc_forward(*_args, **_kwargs):
            raise RuntimeError("relay forward must not call mtc.forward")

        def forbidden_diagnostics(*_args, **_kwargs):
            raise RuntimeError("ordinary forward must not call diagnostics")

        with (
            mock.patch.object(
                TPDCleanV8MPRSDCHBlock,
                "forward",
                new=counted_block_forward,
            ),
            mock.patch.object(
                TPDCleanV8MPRSDCHPatchEmbedding,
                "forward_with_evidence",
                new=counted_evidence_forward,
            ),
            mock.patch.object(
                type(model.mtc),
                "forward",
                new=forbidden_mtc_forward,
            ),
            mock.patch.object(
                TPDCleanV8MPRSDCHBlock,
                "forward_with_mprs_diagnostics",
                new=forbidden_diagnostics,
            ),
        ):
            with torch.no_grad():
                outputs = model(torch.randn(1, 1, 32, 32))

        self.assertEqual(len(outputs), 6)
        self.assertEqual(len(block_calls), 7)
        self.assertTrue(
            all(count == 1 for count in block_calls.values()),
            block_calls,
        )
        self.assertEqual(len(embedding_calls), 2)
        self.assertTrue(
            all(count == 1 for count in embedding_calls.values()),
            embedding_calls,
        )

        relay_source = inspect.getsource(
            TPDNERV8MPRSDCHSCTransNet._forward_with_relay
        )
        for forbidden in (
            "mtc.forward(",
            "forward_with_mprs_diagnostics",
            "register_forward_hook",
            ".clone(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, relay_source)

    def test_zero_gate_first_adam_shared_state_and_two_step_gradient(self) -> None:
        parent = _small_parent(seed=73)
        off, on = _adapt_pair(parent)
        off.train()
        on.train()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(7301)
        inputs = torch.randn(2, 1, 32, 32, generator=generator)
        targets = torch.rand(2, 1, 32, 32, generator=generator)
        off_optimizer = torch.optim.Adam(off.parameters(), lr=1e-3)
        on_optimizer = torch.optim.Adam(on.parameters(), lr=1e-3)

        off_optimizer.zero_grad(set_to_none=True)
        on_optimizer.zero_grad(set_to_none=True)
        off_outputs = off(inputs)
        on_outputs = on(inputs)
        for index, (off_output, on_output) in enumerate(
            zip(off_outputs, on_outputs)
        ):
            self.assertTrue(
                torch.equal(off_output, on_output),
                f"pre-Adam output {index}",
            )
        off_loss = _six_output_loss(off_outputs, targets)
        on_loss = _six_output_loss(on_outputs, targets)
        self.assertTrue(torch.equal(off_loss, on_loss))
        off_loss.backward()
        on_loss.backward()

        off_parameters = dict(off.named_parameters())
        on_parameters = dict(on.named_parameters())
        for name, off_parameter in off_parameters.items():
            on_parameter = on_parameters[name]
            if off_parameter.grad is None:
                self.assertIsNone(on_parameter.grad, name)
            else:
                self.assertIsNotNone(on_parameter.grad, name)
                self.assertTrue(
                    torch.equal(off_parameter.grad, on_parameter.grad),
                    f"shared gradient {name}",
                )
        self.assertGreater(_relay_gate_gradient_l1(on), 0.0)
        self.assertEqual(_relay_fusion_gradient_l1(on), 0.0)

        off_optimizer.step()
        on_optimizer.step()
        off_state = off.state_dict()
        on_state = on.state_dict()
        for name, value in off_state.items():
            self.assertTrue(
                torch.equal(value, on_state[name]),
                f"first-Adam shared state {name}",
            )

        for name, off_parameter in off_parameters.items():
            on_parameter = on_parameters[name]
            off_adam = off_optimizer.state.get(off_parameter, {})
            on_adam = on_optimizer.state.get(on_parameter, {})
            self.assertEqual(set(off_adam), set(on_adam), name)
            for key, off_value in off_adam.items():
                on_value = on_adam[key]
                if isinstance(off_value, torch.Tensor):
                    self.assertTrue(
                        torch.equal(off_value, on_value),
                        f"Adam {name}.{key}",
                    )
                else:
                    self.assertEqual(off_value, on_value)

        on_optimizer.zero_grad(set_to_none=True)
        second_loss = _six_output_loss(on(inputs), targets)
        second_loss.backward()
        self.assertTrue(torch.isfinite(second_loss))
        self.assertGreater(_relay_fusion_gradient_l1(on), 0.0)

    def test_variant_identity_is_checked_before_adaptation(self) -> None:
        full_parent = _small_parent(FULL)
        with self.assertRaisesRegex(ValueError, "context gate"):
            adapt_v8_mprs_dch_parent(
                full_parent,
                variant=CAPACITY,
                relay_enabled=True,
                relay_width=2,
            )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            adapt_v8_mprs_dch_parent(
                full_parent,
                variant=FULL,
                relay_enabled=True,
                relay_width=2,
                relay_initialization_seed=-1,
            )


if __name__ == "__main__":
    unittest.main()
