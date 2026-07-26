from __future__ import annotations

import inspect
import unittest

import torch

from model.tpd_frequency_gate import (
    FixedHaarAnalysis,
    PreparedQueryFrequencyGate,
    QueryFrequencyLevelGate,
    QueryOnlyFrequencyGate,
    frequency_gate_parameter_count,
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


class FixedHaarAnalysisTests(unittest.TestCase):
    def test_constant_input_has_only_low_band(self) -> None:
        haar = FixedHaarAnalysis()
        feature = torch.ones(2, 3, 8, 12)
        bands = haar(feature)
        self.assertEqual(tuple(bands.shape), (2, 3, 4, 4, 6))
        torch.testing.assert_close(
            bands[:, :, 0],
            torch.full((2, 3, 4, 6), 2.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            bands[:, :, 1:],
            torch.zeros(2, 3, 3, 4, 6),
            rtol=0,
            atol=0,
        )

    def test_checkerboard_occupies_high_frequency_band(self) -> None:
        haar = FixedHaarAnalysis()
        tile = torch.tensor(((1.0, -1.0), (-1.0, 1.0)))
        feature = tile.repeat(1, 1, 4, 6)
        bands = haar(feature)
        self.assertEqual(float(bands[:, :, :3].abs().sum()), 0.0)
        self.assertGreater(float(bands[:, :, 3].abs().sum()), 0.0)

    def test_rejects_odd_or_nonfinite_features(self) -> None:
        haar = FixedHaarAnalysis(validate_finite=True)
        for feature in (
            torch.randn(1, 2, 7, 8),
            torch.full((1, 2, 8, 8), float("inf")),
            torch.ones(1, 2, 8, 8, dtype=torch.int64),
        ):
            with self.subTest(shape=tuple(feature.shape), dtype=feature.dtype):
                with self.assertRaises(
                    (ValueError, TypeError, FloatingPointError)
                ):
                    haar(feature)


class QueryFrequencyLevelGateTests(unittest.TestCase):
    def test_validates_registered_alignment_contract(self) -> None:
        for value in (0, (8,), (8, 0), (8, 4.0), "8"):
            with self.subTest(expected_alignment=value):
                with self.assertRaisesRegex(ValueError, "positive integer or pair"):
                    QueryFrequencyLevelGate(
                        4,
                        expected_alignment=value,  # type: ignore[arg-type]
                    )

    def test_zero_alpha_is_bitwise_identity(self) -> None:
        torch.manual_seed(7)
        level = QueryFrequencyLevelGate(
            4,
            mode="high_low",
            hidden_channels=3,
        )
        query = torch.randn(2, 4, 4, 6)
        query[0, 0, 0, 0] = -0.0
        feature = torch.randn(2, 4, 64, 96)
        modulated, logits, factor = level(query, feature)
        assert_tensor_bits_equal(self, modulated, query)
        self.assertEqual(tuple(logits.shape), (2, 1, 4, 6))
        self.assertTrue(torch.equal(factor, torch.ones_like(factor)))

    def test_zero_alpha_is_bitwise_identity_for_float64(self) -> None:
        torch.manual_seed(8)
        level = QueryFrequencyLevelGate(
            4,
            mode="high_low",
            hidden_channels=3,
        ).double()
        query = torch.randn(2, 4, 4, 6, dtype=torch.float64)
        query[0, 0, 0, 0] = -0.0
        feature = torch.randn(2, 4, 64, 96, dtype=torch.float64)
        modulated, _, factor = level(query, feature)
        assert_tensor_bits_equal(self, modulated, query)
        self.assertTrue(torch.equal(factor, torch.ones_like(factor)))

    def test_level_forward_is_prepare_then_apply_prepared(self) -> None:
        torch.manual_seed(9)
        level = QueryFrequencyLevelGate(
            4,
            mode="high_low",
            hidden_channels=3,
        )
        level.alpha.data.fill_(0.3)
        query = torch.randn(2, 7, 4, 6)
        feature = torch.randn(2, 4, 64, 96)
        prepared = level.prepare(feature, (4, 6))
        explicit = level.apply_prepared(query, prepared)
        compatible = level(query, feature)
        for actual, expected in zip(compatible, explicit):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nonzero_alpha_changes_only_query_values(self) -> None:
        torch.manual_seed(11)
        level = QueryFrequencyLevelGate(4, mode="high", hidden_channels=3)
        level.alpha.data.fill_(0.5)
        query = torch.randn(2, 6, 4, 6)
        feature = torch.randn(2, 4, 32, 48)
        query_before = query.clone()
        feature_before = feature.clone()
        modulated, logits, factor = level(query, feature)
        self.assertEqual(tuple(modulated.shape), tuple(query.shape))
        self.assertFalse(torch.equal(modulated, query))
        self.assertTrue(torch.equal(query, query_before))
        self.assertTrue(torch.equal(feature, feature_before))
        self.assertTrue(torch.all(factor > 0.5))
        self.assertTrue(torch.all(factor < 1.5))
        self.assertEqual(tuple(logits.shape), (2, 1, 4, 6))

    def test_two_steps_train_alpha_then_prior_projection(self) -> None:
        torch.manual_seed(13)
        level = QueryFrequencyLevelGate(4, hidden_channels=3)
        optimizer = torch.optim.SGD(level.parameters(), lr=0.1)
        query = torch.randn(2, 4, 2, 3)
        feature = torch.randn(2, 4, 32, 48)

        optimizer.zero_grad(set_to_none=True)
        first = level(query, feature)[0].square().mean()
        first.backward()
        self.assertIsNotNone(level.alpha.grad)
        self.assertGreater(float(level.alpha.grad.abs()), 0.0)
        first_prior_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in level.named_parameters()
            if name != "alpha" and parameter.grad is not None
        )
        self.assertEqual(first_prior_gradient, 0.0)
        optimizer.step()
        self.assertNotEqual(float(level.alpha.detach()), 0.0)

        optimizer.zero_grad(set_to_none=True)
        second = level(query, feature)[0].square().mean()
        second.backward()
        second_prior_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in level.named_parameters()
            if name != "alpha" and parameter.grad is not None
        )
        self.assertGreater(second_prior_gradient, 0.0)

    def test_rejects_nonintegral_prior_to_query_alignment(self) -> None:
        level = QueryFrequencyLevelGate(4)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            level(
                torch.randn(2, 4, 5, 5),
                torch.randn(2, 4, 32, 32),
            )

    def test_bounded_alpha_cannot_reverse_query_sign(self) -> None:
        torch.manual_seed(17)
        level = QueryFrequencyLevelGate(4)
        level.alpha.data.fill_(20.0)
        query = torch.ones(2, 4, 2, 3)
        feature = torch.randn(2, 4, 32, 48)
        modulated, _, factor = level(query, feature)
        self.assertTrue(torch.all(factor >= 0.0))
        self.assertTrue(torch.all(factor <= 2.0))
        self.assertTrue(torch.all(modulated >= 0.0))


class QueryOnlyFrequencyGateTests(unittest.TestCase):
    def _inputs(self):
        queries = tuple(
            torch.randn(2, channels, 4, 6)
            for channels in (4, 8, 16, 32)
        )
        features = (
            torch.randn(2, 4, 64, 96),
            torch.randn(2, 8, 32, 48),
            torch.randn(2, 16, 16, 24),
            torch.randn(2, 32, 8, 12),
        )
        return queries, features

    def test_four_levels_align_8_4_2_1_and_start_exact_identity(self) -> None:
        gate = QueryOnlyFrequencyGate(
            (4, 8, 16, 32),
            hidden_channels=3,
        )
        queries, features = self._inputs()
        output = gate(queries, features)
        self.assertEqual(len(output.queries), 4)
        for index in range(4):
            with self.subTest(level=index + 1):
                self.assertTrue(torch.equal(output.queries[index], queries[index]))
                self.assertEqual(
                    tuple(output.gate_logits[index].shape),
                    (2, 1, 4, 6),
                )
                self.assertTrue(
                    torch.equal(
                        output.factors[index],
                        torch.ones_like(output.factors[index]),
                    )
                )

    def test_prepare_once_can_be_applied_to_many_sctb_query_groups(self) -> None:
        torch.manual_seed(19)
        gate = QueryOnlyFrequencyGate(
            (4, 8, 16, 32),
            hidden_channels=3,
        )
        for level in gate.levels:
            level.alpha.data.fill_(0.25)
        _, features = self._inputs()
        counts = {
            "haar": [0, 0, 0, 0],
            "prior_projection": [0, 0, 0, 0],
            "gate_projection": [0, 0, 0, 0],
        }
        handles = []

        def count(kind, index):
            def hook(_module, _inputs, _output):
                counts[kind][index] += 1

            return hook

        for index, level in enumerate(gate.levels):
            handles.extend(
                (
                    level.haar.register_forward_hook(count("haar", index)),
                    level.prior_projection.register_forward_hook(
                        count("prior_projection", index)
                    ),
                    level.gate_projection.register_forward_hook(
                        count("gate_projection", index)
                    ),
                )
            )
        try:
            prepared = gate.prepare(features, (4, 6))
            self.assertIsInstance(prepared, PreparedQueryFrequencyGate)
            outputs = []
            for _ in range(4):
                queries, _ = self._inputs()
                outputs.append(gate.apply_prepared(queries, prepared))
        finally:
            for handle in handles:
                handle.remove()

        for kind in counts:
            self.assertEqual(counts[kind], [1, 1, 1, 1])
        for output in outputs:
            for index in range(4):
                self.assertIs(
                    output.gate_logits[index],
                    prepared.gate_logits[index],
                )

    def test_one_shot_forward_equals_new_composed_api(self) -> None:
        torch.manual_seed(23)
        gate = QueryOnlyFrequencyGate(
            (4, 8, 16, 32),
            hidden_channels=3,
        )
        for level in gate.levels:
            level.alpha.data.fill_(0.4)
        queries, features = self._inputs()
        prepared = gate.prepare(
            features,
            tuple(tuple(query.shape[-2:]) for query in queries),
        )
        composed = gate.apply_prepared(queries, prepared)
        compatible = gate(queries, features)
        for field in ("queries", "gate_logits", "factors"):
            for actual, expected in zip(
                getattr(compatible, field),
                getattr(composed, field),
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prepared_identity_is_bitwise_exact_for_float32_and_float64(
        self,
    ) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(29)
                gate = QueryOnlyFrequencyGate(
                    (4, 8, 16, 32),
                    hidden_channels=3,
                ).to(dtype=dtype)
                queries, features = self._inputs()
                queries = tuple(query.to(dtype=dtype) for query in queries)
                for query in queries:
                    query[0, 0, 0, 0] = -0.0
                features = tuple(feature.to(dtype=dtype) for feature in features)
                prepared = gate.prepare(features, (4, 6))
                for _ in range(3):
                    output = gate.apply_prepared(queries, prepared)
                    for query, modulated, factor in zip(
                        queries,
                        output.queries,
                        output.factors,
                    ):
                        assert_tensor_bits_equal(self, modulated, query)
                        self.assertTrue(
                            torch.equal(factor, torch.ones_like(factor))
                        )

    def test_many_applies_backpropagate_to_queries_projection_and_alpha(
        self,
    ) -> None:
        torch.manual_seed(31)
        gate = QueryOnlyFrequencyGate(
            (4, 8, 16, 32),
            hidden_channels=3,
        )
        for level in gate.levels:
            level.alpha.data.fill_(0.35)
        _, features = self._inputs()
        prepared = gate.prepare(features, (4, 6))
        query_groups = []
        loss = torch.zeros(())
        for _ in range(3):
            queries, _ = self._inputs()
            queries = tuple(query.requires_grad_() for query in queries)
            query_groups.append(queries)
            output = gate.apply_prepared(queries, prepared)
            loss = loss + sum(query.square().mean() for query in output.queries)
        loss.backward()

        for group in query_groups:
            for query in group:
                self.assertIsNotNone(query.grad)
                self.assertGreater(float(query.grad.abs().sum()), 0.0)
        for level in gate.levels:
            self.assertIsNotNone(level.alpha.grad)
            self.assertGreater(float(level.alpha.grad.abs()), 0.0)
            projection_gradient = sum(
                float(parameter.grad.abs().sum())
                for name, parameter in level.named_parameters()
                if name != "alpha" and parameter.grad is not None
            )
            self.assertGreater(projection_gradient, 0.0)

    def test_public_api_contains_query_path_only(self) -> None:
        expected = {
            "prepare": ("self", "encoder_features", "query_sizes"),
            "apply_prepared": ("self", "queries", "prepared"),
            "forward": ("self", "queries", "encoder_features"),
        }
        for method_name, parameter_names in expected.items():
            with self.subTest(method=method_name):
                signature = inspect.signature(
                    getattr(QueryOnlyFrequencyGate, method_name)
                )
                self.assertEqual(tuple(signature.parameters), parameter_names)
        visited = []
        gate = QueryOnlyFrequencyGate()
        returned = gate.apply(lambda module: visited.append(type(module).__name__))
        self.assertIs(returned, gate)
        self.assertIn("QueryOnlyFrequencyGate", visited)

    def test_rejects_prepared_grid_batch_and_type_mismatches(self) -> None:
        gate = QueryOnlyFrequencyGate(
            (4, 8, 16, 32),
            hidden_channels=3,
        )
        queries, features = self._inputs()
        prepared = gate.prepare(features, (4, 6))
        with self.assertRaisesRegex(TypeError, "PreparedQueryFrequencyGate"):
            gate.apply_prepared(queries, object())  # type: ignore[arg-type]
        foreign_gate = QueryOnlyFrequencyGate(
            (4, 8, 16, 32),
            hidden_channels=3,
        )
        foreign_prepared = foreign_gate.prepare(features, (4, 6))
        with self.assertRaisesRegex(ValueError, "different gate instance"):
            gate.apply_prepared(queries, foreign_prepared)
        wrong_grid = (
            torch.randn(2, 4, 2, 3),
            *queries[1:],
        )
        with self.assertRaisesRegex(ValueError, "prepared grid"):
            gate.apply_prepared(wrong_grid, prepared)
        wrong_batch = (queries[0][:1], *queries[1:])
        with self.assertRaisesRegex(ValueError, "batch"):
            gate.apply_prepared(wrong_batch, prepared)

    def test_manifest_locks_query_only_boundary_and_three_branch_mainline(self) -> None:
        gate = QueryOnlyFrequencyGate()
        manifest = gate.architecture_manifest()
        self.assertEqual(manifest["modified_attention_tensors"], ("Q",))
        self.assertFalse(manifest["kv_modified"])
        self.assertFalse(manifest["cfn_modified"])
        self.assertFalse(manifest["decoder_injection"])
        self.assertFalse(manifest["tokenizer_branch_added"])
        self.assertEqual(manifest["alpha_initialization"], 0.0)
        self.assertEqual(
            manifest["alpha_parameterization"],
            "tanh_bounded_no_sign_reversal",
        )
        self.assertEqual(manifest["registered_alignment_ratios"], (8, 4, 2, 1))
        self.assertEqual(
            manifest["high_frequency_representation"],
            "absolute_magnitude",
        )
        self.assertEqual(manifest["projection_order"], "haar_align_then_1x1")
        self.assertEqual(
            manifest["execution"],
            "prepare_once_apply_many_per_model_forward",
        )
        self.assertEqual(
            manifest["prepared_object_persistence"],
            "forward_local_only",
        )
        self.assertGreater(frequency_gate_parameter_count(gate), 0)
        self.assertEqual(
            set(name for name, _ in gate.named_buffers()),
            {
                "levels.0.haar.kernels",
                "levels.1.haar.kernels",
                "levels.2.haar.kernels",
                "levels.3.haar.kernels",
            },
        )

    def test_validates_four_level_input_contract(self) -> None:
        with self.assertRaisesRegex(TypeError, "finite sequence"):
            QueryOnlyFrequencyGate(4)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "four feature levels"):
            QueryOnlyFrequencyGate((4, 8, 16))
        gate = QueryOnlyFrequencyGate((4, 8, 16, 32))
        queries, features = self._inputs()
        with self.assertRaisesRegex(ValueError, "exactly four"):
            gate(queries[:3], features)
        with self.assertRaisesRegex(ValueError, "batch"):
            gate(
                queries,
                (features[0][:1], *features[1:]),
            )
        wrong_scale_features = (
            torch.randn(2, 4, 32, 48),
            *features[1:],
        )
        with self.assertRaisesRegex(ValueError, "registered level ratio"):
            gate(queries, wrong_scale_features)


if __name__ == "__main__":
    unittest.main()
