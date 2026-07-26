from __future__ import annotations

import inspect
import unittest

import torch

from model.SCTransNet import Encoder, get_CTranS_config
from model.tpd_frequency_gate import QueryOnlyFrequencyGate
from model.tpd_query_frequency_bridge import (
    frequency_attention_forward,
    frequency_block_forward,
    frequency_encoder_forward,
)


torch.set_num_threads(1)

CHANNELS = (4, 8, 16, 32)


def small_config(*, num_layers: int = 4):
    config = get_CTranS_config()
    config.KV_size = sum(CHANNELS)
    config.transformer.num_layers = num_layers
    return config


def build_frozen_encoder(*, num_layers: int = 4, vis: bool = True) -> Encoder:
    encoder = Encoder(
        small_config(num_layers=num_layers),
        vis=vis,
        channel_num=list(CHANNELS),
    )
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def common_grid_embeddings(
    grid: tuple[int, int],
    *,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = grid
    return tuple(
        torch.randn(batch_size, channels, height, width)
        for channels in CHANNELS
    )


def encoder_features_for_grid(
    grid: tuple[int, int],
    *,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = grid
    # After the fixed stride-2 Haar analysis, these produce the registered
    # prior-to-Query ratios 8, 4, 2, 1.
    input_scales = (16, 8, 4, 2)
    return tuple(
        torch.randn(
            batch_size,
            channels,
            height * scale,
            width * scale,
        )
        for channels, scale in zip(CHANNELS, input_scales)
    )


def assert_tensor_bits_equal(
    testcase: unittest.TestCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    testcase.assertEqual(actual.dtype, expected.dtype)
    testcase.assertEqual(tuple(actual.shape), tuple(expected.shape))
    testcase.assertTrue(
        torch.equal(
            actual.detach().contiguous().view(torch.uint8),
            expected.detach().contiguous().view(torch.uint8),
        )
    )


class RecordingGate:
    def __init__(self, delegate: QueryOnlyFrequencyGate) -> None:
        self.delegate = delegate
        self.prepared_objects = []
        self.query_groups = []

    def apply_prepared(self, queries, prepared):
        self.prepared_objects.append(prepared)
        self.query_groups.append(tuple(queries))
        return self.delegate.apply_prepared(queries, prepared)


class QueryFrequencyBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2301)

    def test_public_functions_have_explicit_forward_local_contract(self) -> None:
        expected = {
            frequency_attention_forward: (
                "attention",
                "emb1",
                "emb2",
                "emb3",
                "emb4",
                "emb_all",
                "qfg",
                "prepared",
            ),
            frequency_block_forward: (
                "block",
                "emb1",
                "emb2",
                "emb3",
                "emb4",
                "qfg",
                "prepared",
            ),
            frequency_encoder_forward: (
                "encoder",
                "emb1",
                "emb2",
                "emb3",
                "emb4",
                "qfg",
                "prepared",
            ),
        }
        for function, names in expected.items():
            with self.subTest(function=function.__name__):
                self.assertEqual(
                    tuple(inspect.signature(function).parameters),
                    names,
                )

    def test_zero_alpha_matches_frozen_four_output_encoder_bit_for_bit(
        self,
    ) -> None:
        encoder = build_frozen_encoder(num_layers=4, vis=True)
        qfg = QueryOnlyFrequencyGate(CHANNELS, hidden_channels=3).eval()
        embeddings = common_grid_embeddings((3, 5))
        features = encoder_features_for_grid((3, 5))
        encoder_keys_before = tuple(encoder.state_dict())
        encoder_modules_before = tuple(encoder.named_modules())

        counts = [0, 0, 0, 0]
        handles = []
        for index, level in enumerate(qfg.levels):
            def count_haar(_module, _inputs, _output, *, index=index):
                counts[index] += 1

            handles.append(level.haar.register_forward_hook(count_haar))
        try:
            with torch.no_grad():
                expected = encoder(*embeddings)
                prepared = qfg.prepare(features, (3, 5))
                recording_gate = RecordingGate(qfg)
                actual = frequency_encoder_forward(
                    encoder,
                    *embeddings,
                    recording_gate,
                    prepared,
                )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(counts, [1, 1, 1, 1])
        self.assertEqual(len(recording_gate.prepared_objects), 4)
        self.assertTrue(
            all(
                value is prepared
                for value in recording_gate.prepared_objects
            )
        )
        self.assertEqual(len(recording_gate.query_groups), 4)
        for query_group in recording_gate.query_groups:
            self.assertEqual(
                [tuple(query.shape[-2:]) for query in query_group],
                [(3, 5)] * 4,
            )
        for actual_level, expected_level in zip(actual[:4], expected[:4]):
            assert_tensor_bits_equal(self, actual_level, expected_level)
        self.assertEqual(actual[4], expected[4])

        self.assertEqual(tuple(encoder.state_dict()), encoder_keys_before)
        self.assertEqual(tuple(encoder.named_modules()), encoder_modules_before)
        for module in (*tuple(encoder.modules()), *tuple(qfg.modules())):
            self.assertFalse(
                any(value is prepared for value in vars(module).values())
            )

    def test_nonzero_alpha_changes_query_path_but_not_attention_k_or_v(
        self,
    ) -> None:
        encoder = build_frozen_encoder(num_layers=1, vis=False)
        attention = encoder.layer[0].channel_attn
        qfg = QueryOnlyFrequencyGate(CHANNELS, hidden_channels=3).eval()
        for level in qfg.levels:
            level.alpha.data.fill_(0.6)

        embeddings = common_grid_embeddings((2, 3))
        emb_all = torch.cat(embeddings, dim=1)
        features = encoder_features_for_grid((2, 3))
        prepared = qfg.prepare(features, (2, 3))
        captures = {"k": [], "v": [], "q": [[], [], [], []]}
        handles = [
            attention.k.register_forward_hook(
                lambda _module, _inputs, output: captures["k"].append(
                    output.detach().clone()
                )
            ),
            attention.v.register_forward_hook(
                lambda _module, _inputs, output: captures["v"].append(
                    output.detach().clone()
                )
            ),
        ]
        for index, query_conv in enumerate(
            (attention.q1, attention.q2, attention.q3, attention.q4)
        ):
            handles.append(
                query_conv.register_forward_hook(
                    lambda _module, _inputs, output, *, index=index:
                    captures["q"][index].append(output.detach().clone())
                )
            )
        recording_gate = RecordingGate(qfg)
        try:
            with torch.no_grad():
                expected = attention(*embeddings, emb_all)
                actual = frequency_attention_forward(
                    attention,
                    *embeddings,
                    emb_all,
                    recording_gate,
                    prepared,
                )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(recording_gate.query_groups), 1)
        for index, query in enumerate(recording_gate.query_groups[0]):
            assert_tensor_bits_equal(self, query, captures["q"][index][1])
        assert_tensor_bits_equal(self, captures["k"][1], captures["k"][0])
        assert_tensor_bits_equal(self, captures["v"][1], captures["v"][0])
        self.assertTrue(
            any(
                not torch.equal(actual_level, expected_level)
                for actual_level, expected_level in zip(actual[:4], expected[:4])
            )
        )
        self.assertIsNone(actual[4])

    def test_nonzero_alpha_backpropagates_through_frequency_gate_only(
        self,
    ) -> None:
        encoder = build_frozen_encoder(num_layers=2, vis=False)
        qfg = QueryOnlyFrequencyGate(CHANNELS, hidden_channels=3)
        for level in qfg.levels:
            level.alpha.data.fill_(0.35)
        embeddings = common_grid_embeddings((2, 3))
        features = encoder_features_for_grid((2, 3))
        prepared = qfg.prepare(features, (2, 3))

        actual = frequency_encoder_forward(
            encoder,
            *embeddings,
            qfg,
            prepared,
        )
        loss = sum(
            (index + 1) * level.square().mean()
            for index, level in enumerate(actual[:4])
        )
        loss.backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in encoder.parameters())
        )
        for index, level in enumerate(qfg.levels):
            with self.subTest(level=index + 1):
                self.assertIsNotNone(level.alpha.grad)
                self.assertTrue(torch.isfinite(level.alpha.grad).all())
                self.assertGreater(float(level.alpha.grad.abs()), 0.0)
                projection_gradient = sum(
                    float(parameter.grad.abs().sum())
                    for name, parameter in level.named_parameters()
                    if name != "alpha" and parameter.grad is not None
                )
                self.assertGreater(projection_gradient, 0.0)

    def test_dynamic_common_query_grids_are_supported(self) -> None:
        encoder = build_frozen_encoder(num_layers=1, vis=False)
        qfg = QueryOnlyFrequencyGate(CHANNELS, hidden_channels=2).eval()
        for grid in ((2, 3), (3, 5), (4, 7)):
            with self.subTest(grid=grid):
                embeddings = common_grid_embeddings(grid)
                features = encoder_features_for_grid(grid)
                with torch.no_grad():
                    prepared = qfg.prepare(features, grid)
                    outputs = frequency_encoder_forward(
                        encoder,
                        *embeddings,
                        qfg,
                        prepared,
                    )
                self.assertEqual(
                    [tuple(output.shape[-2:]) for output in outputs[:4]],
                    [grid] * 4,
                )
                self.assertEqual(
                    [level.query_size for level in prepared.levels],
                    [grid] * 4,
                )


if __name__ == "__main__":
    unittest.main()
