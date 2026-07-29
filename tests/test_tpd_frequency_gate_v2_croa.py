from __future__ import annotations

import copy
import inspect
import math
import unittest
from unittest import mock

import torch
import torch.nn as nn

import model.tpd_frequency_gate_v2_croa as qfg_core
from model.tpd_frequency_gate_v2_croa import (
    FORMAL_ALIGNMENT_RATIOS,
    FORMAL_ALPHA_EFFECTIVE_INIT,
    FORMAL_QFG_V2_CROA_PARAMETER_KEYS,
    FORMAL_QFG_V2_CROA_STATE_KEYS,
    PRODUCTION_QFG_V2_CROA_PARAMETERS,
    PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
    PreparedQueryFrequencyGateV2CROA,
    QueryFrequencyGateOutputV2CROA,
    QueryFrequencyLevelGateV2CROA,
    QueryOnlyFrequencyGateV2CROA,
    RMS_EPS,
    centered_bounded_arctangent_gate,
    frequency_gate_parameter_count,
    sample_full_tensor_rms_normalize,
    spatial_center_rms_normalize,
    validate_formal_qfg_v2_croa,
)


def assert_bits_equal(
    testcase: unittest.TestCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    testcase.assertEqual(actual.dtype, expected.dtype)
    testcase.assertEqual(tuple(actual.shape), tuple(expected.shape))
    testcase.assertTrue(
        torch.equal(
            actual.detach().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def assert_adam_parameter_state_equal(
    testcase: unittest.TestCase,
    left_optimizer: torch.optim.Optimizer,
    left_parameter: nn.Parameter,
    right_optimizer: torch.optim.Optimizer,
    right_parameter: nn.Parameter,
) -> None:
    left = left_optimizer.state[left_parameter]
    right = right_optimizer.state[right_parameter]
    testcase.assertEqual(set(left), set(right))
    for key in left:
        with testcase.subTest(adam_state=key):
            if isinstance(left[key], torch.Tensor):
                assert_bits_equal(testcase, left[key], right[key])
            else:
                testcase.assertEqual(left[key], right[key])


def legacy_reference_apply_prepared(
    module: QueryOnlyFrequencyGateV2CROA,
    queries: tuple[torch.Tensor, ...],
    prepared: PreparedQueryFrequencyGateV2CROA,
) -> QueryFrequencyGateOutputV2CROA:
    """Frozen pre-optimization apply path used only as a test oracle."""

    outputs = []
    for level, query, prepared_level in zip(
        module.levels,
        queries,
        prepared.levels,
    ):
        raw_gate_logits = prepared_level.raw_gate_logits
        normalized_logits = spatial_center_rms_normalize(
            raw_gate_logits,
            eps=level.eps,
            validate_finite=False,
        )
        bounded_gate = centered_bounded_arctangent_gate(
            normalized_logits,
            validate_finite=False,
        )
        compute_dtype = qfg_core._working_dtype(
            query.dtype,
            bounded_gate.dtype,
        )
        factor = 1.0 + torch.tanh(
            level.alpha.to(dtype=compute_dtype)
        ) * bounded_gate.to(dtype=compute_dtype)
        modulated = (
            query.to(dtype=compute_dtype) * factor
        ).to(dtype=query.dtype)
        outputs.append(
            (
                modulated,
                raw_gate_logits,
                normalized_logits,
                bounded_gate,
                factor,
            )
        )

    (
        query_outputs,
        raw_gate_logits,
        normalized_logits,
        gates,
        factors,
    ) = zip(*outputs)
    return QueryFrequencyGateOutputV2CROA(
        queries=tuple(query_outputs),
        raw_gate_logits=tuple(raw_gate_logits),
        normalized_logits=tuple(normalized_logits),
        gates=tuple(gates),
        factors=tuple(factors),
    )


class FrequencyMathTests(unittest.TestCase):
    def test_full_tensor_rms_matches_formula_and_is_scale_stable(self) -> None:
        source = torch.tensor(
            [
                [[[-3.0, -1.0], [2.0, 4.0]]],
                [[[0.5, -2.0], [1.5, 3.0]]],
            ],
            dtype=torch.float64,
        )
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                value = source.to(dtype=dtype)
                actual = sample_full_tensor_rms_normalize(value)
                expected = value / (
                    value.square().mean(dim=(1, 2, 3), keepdim=True)
                    + RMS_EPS
                ).sqrt()
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-6 if dtype == torch.float32 else 1e-12,
                    atol=2e-7 if dtype == torch.float32 else 1e-12,
                )

                maximum = torch.finfo(dtype).max
                huge = (source / source.abs().amax()).to(dtype=dtype)
                huge = huge * (maximum / 2.0)
                normalized_huge = sample_full_tensor_rms_normalize(huge)
                self.assertTrue(bool(torch.isfinite(normalized_huge).all()))
                rms = normalized_huge.square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                torch.testing.assert_close(
                    rms,
                    torch.ones_like(rms),
                    rtol=2e-5,
                    atol=2e-5,
                )

    def test_reduced_precision_reductions_promote_and_fp64_is_preserved(
        self,
    ) -> None:
        raw = torch.tensor(
            [[[[1.0, 2.0], [4.0, 8.0]]]],
            dtype=torch.float32,
        )
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                reduced = raw.to(dtype=dtype)
                normalized_prior = sample_full_tensor_rms_normalize(reduced)
                normalized_logits = spatial_center_rms_normalize(
                    reduced[:, :1]
                )
                self.assertEqual(normalized_prior.dtype, dtype)
                self.assertEqual(normalized_logits.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(normalized_logits).all()))

        fine = torch.tensor(
            [[[[1.0, 1.0 + 1e-10], [1.0 - 2e-10, 1.0 + 3e-10]]]],
            dtype=torch.float64,
        )
        normalized_fine = spatial_center_rms_normalize(fine)
        self.assertEqual(normalized_fine.dtype, torch.float64)
        self.assertGreater(float(normalized_fine.abs().max()), 0.0)

    def test_spatial_normalization_and_gate_match_locked_formula(self) -> None:
        raw = torch.tensor(
            [
                [[[1.0, 2.0, 5.0], [-3.0, 4.0, 8.0]]],
                [[[7.0, -2.0, 1.0], [0.0, 3.0, -6.0]]],
            ],
            dtype=torch.float64,
        )
        centered = raw - raw.mean(dim=(-2, -1), keepdim=True)
        expected_normalized = centered / (
            centered.square().mean(dim=(-2, -1), keepdim=True)
            + RMS_EPS
        ).sqrt()
        normalized = spatial_center_rms_normalize(raw)
        torch.testing.assert_close(
            normalized,
            expected_normalized,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            normalized.mean(dim=(-2, -1)),
            torch.zeros(2, 1, dtype=torch.float64),
            rtol=0.0,
            atol=2e-16,
        )

        half_bounded = torch.atan(math.pi * normalized) / math.pi
        expected_gate = 0.5 * (
            half_bounded
            - half_bounded.mean(dim=(-2, -1), keepdim=True)
        )
        gate = centered_bounded_arctangent_gate(normalized)
        torch.testing.assert_close(
            gate,
            expected_gate,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            gate.mean(dim=(-2, -1)),
            torch.zeros(2, 1, dtype=torch.float64),
            rtol=0.0,
            atol=2e-16,
        )
        self.assertTrue(bool(torch.all(gate > -0.5)))
        self.assertTrue(bool(torch.all(gate < 0.5)))

    def test_constant_logits_are_zero_and_extremes_stay_strict(self) -> None:
        constant = torch.full((2, 1, 3, 5), 17.0, dtype=torch.float64)
        normalized = spatial_center_rms_normalize(constant)
        gate = centered_bounded_arctangent_gate(normalized)
        self.assertTrue(torch.equal(normalized, torch.zeros_like(normalized)))
        self.assertTrue(torch.equal(gate, torch.zeros_like(gate)))

        extreme = torch.tensor(
            [[[[torch.finfo(torch.float64).max / 2.0, 0.0, 1.0],
               [-torch.finfo(torch.float64).max / 2.0, -1.0, 2.0]]]],
            dtype=torch.float64,
        )
        normalized_extreme = spatial_center_rms_normalize(extreme)
        gate_extreme = centered_bounded_arctangent_gate(normalized_extreme)
        self.assertTrue(bool(torch.isfinite(normalized_extreme).all()))
        self.assertTrue(bool(torch.isfinite(gate_extreme).all()))
        self.assertTrue(bool(torch.all(gate_extreme.abs() < 0.5)))

    def test_zero_point_factor_gradient_has_locked_gain_50(self) -> None:
        raw = torch.zeros(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)
        probe = torch.tensor(
            [[[[2.0, -1.0, 3.0], [-4.0, 1.0, -1.0]]]],
            dtype=torch.float64,
        )
        self.assertEqual(float(probe.mean()), 0.0)
        normalized = spatial_center_rms_normalize(raw)
        gate = centered_bounded_arctangent_gate(normalized)
        factor = 1.0 + FORMAL_ALPHA_EFFECTIVE_INIT * gate
        (factor * probe).sum().backward()
        expected = (
            FORMAL_ALPHA_EFFECTIVE_INIT
            * 0.5
            / math.sqrt(RMS_EPS)
            * probe
        )
        self.assertIsNotNone(raw.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        torch.testing.assert_close(
            raw.grad,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_finite_validation_cpu_errors_and_cuda_async_path(self) -> None:
        for operation, value in (
            (
                sample_full_tensor_rms_normalize,
                torch.full((1, 2, 2, 2), float("inf")),
            ),
            (
                spatial_center_rms_normalize,
                torch.full((1, 1, 2, 2), float("nan")),
            ),
            (
                centered_bounded_arctangent_gate,
                torch.full((1, 1, 2, 2), float("inf")),
            ),
        ):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(FloatingPointError):
                    operation(value)

        if not torch.cuda.is_available():
            return
        device = torch.device("cuda")
        gate = QueryOnlyFrequencyGateV2CROA(
            (2, 3, 4, 5),
            hidden_channels=3,
        ).to(device)
        queries = tuple(
            torch.randn(2, channels, 2, 3, device=device)
            for channels in (2, 3, 4, 5)
        )
        features = (
            torch.randn(2, 2, 32, 48, device=device),
            torch.randn(2, 3, 16, 24, device=device),
            torch.randn(2, 4, 8, 12, device=device),
            torch.randn(2, 5, 4, 6, device=device),
        )
        original_async_assert = torch._assert_async
        with mock.patch.object(
            torch,
            "_assert_async",
            wraps=original_async_assert,
        ) as async_assert:
            prepared = gate.prepare(features, (2, 3))
            output = gate.apply_prepared(queries, prepared)
            qfg_core._assert_runtime_condition(
                torch.ones((), dtype=torch.bool, device=device),
                message="CUDA async assertion smoke",
            )
            torch.cuda.synchronize(device)
        self.assertGreater(async_assert.call_count, 0)
        for call in async_assert.call_args_list:
            condition = call.args[0]
            self.assertIsInstance(condition, torch.Tensor)
            self.assertEqual(condition.device.type, "cuda")
        for tensor in (
            *output.queries,
            *output.raw_gate_logits,
            *output.normalized_logits,
            *output.gates,
            *output.factors,
        ):
            self.assertTrue(bool(torch.isfinite(tensor).all()))


class QueryGateContractTests(unittest.TestCase):
    CHANNELS = (2, 3, 4, 5)
    QUERY_SIZE = (2, 3)

    @classmethod
    def inputs(cls):
        queries = tuple(
            torch.randn(2, channels, *cls.QUERY_SIZE)
            for channels in cls.CHANNELS
        )
        features = (
            torch.randn(2, 2, 32, 48),
            torch.randn(2, 3, 16, 24),
            torch.randn(2, 4, 8, 12),
            torch.randn(2, 5, 4, 6),
        )
        return queries, features

    @classmethod
    def gate(cls) -> QueryOnlyFrequencyGateV2CROA:
        return QueryOnlyFrequencyGateV2CROA(
            cls.CHANNELS,
            hidden_channels=3,
        )

    def test_public_api_alignment_int_and_pair_contract(self) -> None:
        expected = {
            "prepare": ("self", "encoder_features", "query_sizes"),
            "apply_prepared": ("self", "queries", "prepared"),
            "forward": ("self", "queries", "encoder_features"),
        }
        for method, parameters in expected.items():
            with self.subTest(method=method):
                signature = inspect.signature(
                    getattr(QueryOnlyFrequencyGateV2CROA, method)
                )
                self.assertEqual(tuple(signature.parameters), parameters)

        scalar = QueryFrequencyLevelGateV2CROA(
            2,
            hidden_channels=3,
            expected_alignment=8,
        )
        pair = QueryFrequencyLevelGateV2CROA(
            2,
            hidden_channels=3,
            expected_alignment=(8, 4),
        )
        self.assertEqual(scalar.expected_alignment, (8, 8))
        self.assertEqual(pair.expected_alignment, (8, 4))
        mixed = QueryOnlyFrequencyGateV2CROA(
            self.CHANNELS,
            hidden_channels=3,
            alignment_ratios=(8, (4, 4), 2, (1, 1)),
        )
        self.assertEqual(
            mixed.alignment_ratios,
            ((8, 8), (4, 4), (2, 2), (1, 1)),
        )

    def test_identity_is_bitwise_and_outputs_are_semantically_separate(
        self,
    ) -> None:
        torch.manual_seed(101)
        gate = self.gate()
        queries, features = self.inputs()
        for query in queries:
            query[0, 0, 0, 0] = -0.0
        prepared = gate.prepare(features, self.QUERY_SIZE)
        output = gate.apply_prepared(queries, prepared)
        self.assertIsInstance(output, QueryFrequencyGateOutputV2CROA)
        self.assertIs(output.raw_gate_logits[0], prepared.raw_gate_logits[0])
        for index in range(4):
            with self.subTest(level=index):
                self.assertIs(
                    output.normalized_logits[index],
                    prepared.levels[index].normalized_logits,
                )
                self.assertIs(
                    output.gates[index],
                    prepared.levels[index].gate,
                )
                self.assertIs(
                    output.factors[index],
                    prepared.levels[index].factor,
                )
                assert_bits_equal(self, output.queries[index], queries[index])
                self.assertTrue(
                    torch.equal(
                        output.raw_gate_logits[index],
                        torch.zeros_like(output.raw_gate_logits[index]),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output.normalized_logits[index],
                        torch.zeros_like(output.normalized_logits[index]),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output.gates[index],
                        torch.zeros_like(output.gates[index]),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output.factors[index],
                        torch.ones_like(output.factors[index]),
                    )
                )

    def test_prepare_once_can_apply_many_without_recomputing_frequency(
        self,
    ) -> None:
        torch.manual_seed(103)
        gate = self.gate()
        _, features = self.inputs()
        counts = [0, 0, 0, 0]
        handles = []

        def hook(index):
            def count(_module, _inputs, _output):
                counts[index] += 1

            return count

        for index, level in enumerate(gate.levels):
            handles.append(level.haar.register_forward_hook(hook(index)))
        try:
            prepared = gate.prepare(features, self.QUERY_SIZE)
            self.assertEqual(counts, [1, 1, 1, 1])
            for _ in range(3):
                queries, _ = self.inputs()
                output = gate.apply_prepared(queries, prepared)
                self.assertEqual(len(output.queries), 4)
            self.assertEqual(counts, [1, 1, 1, 1])
        finally:
            for handle in handles:
                handle.remove()

    def test_prepare_caches_query_independent_math_and_finite_checks(
        self,
    ) -> None:
        torch.manual_seed(104)
        gate = self.gate()
        queries, features = self.inputs()
        original_normalize = qfg_core.spatial_center_rms_normalize
        original_bounded_gate = qfg_core.centered_bounded_arctangent_gate
        original_tanh = torch.tanh
        original_isfinite = torch.isfinite
        original_all = torch.all
        with (
            mock.patch.object(
                qfg_core,
                "spatial_center_rms_normalize",
                wraps=original_normalize,
            ) as normalize_call,
            mock.patch.object(
                qfg_core,
                "centered_bounded_arctangent_gate",
                wraps=original_bounded_gate,
            ) as bounded_gate_call,
            mock.patch.object(
                qfg_core.torch,
                "tanh",
                wraps=original_tanh,
            ) as tanh_call,
            mock.patch.object(
                qfg_core.torch,
                "isfinite",
                wraps=original_isfinite,
            ) as finite_call,
            mock.patch.object(
                qfg_core.torch,
                "all",
                wraps=original_all,
            ) as range_call,
        ):
            prepared = gate.prepare(features, self.QUERY_SIZE)
            self.assertEqual(normalize_call.call_count, 4)
            self.assertEqual(bounded_gate_call.call_count, 4)
            self.assertEqual(tanh_call.call_count, 4)
            self.assertEqual(finite_call.call_count, 20)
            self.assertEqual(range_call.call_count, 12)

            for _ in range(4):
                gate.apply_prepared(queries, prepared)

            self.assertEqual(normalize_call.call_count, 4)
            self.assertEqual(bounded_gate_call.call_count, 4)
            self.assertEqual(tanh_call.call_count, 4)
            self.assertEqual(finite_call.call_count, 52)
            self.assertEqual(range_call.call_count, 12)

    def test_cached_path_matches_legacy_forward_and_backward(self) -> None:
        torch.manual_seed(105)
        optimized = self.gate()
        with torch.no_grad():
            for index, level in enumerate(optimized.levels):
                values = torch.linspace(
                    -0.2 - 0.03 * index,
                    0.25 + 0.02 * index,
                    level.gate_out.weight.numel(),
                ).reshape_as(level.gate_out.weight)
                level.gate_out.weight.copy_(values)
        reference = copy.deepcopy(optimized)

        _, feature_values = self.inputs()
        optimized_features = tuple(
            value.detach().clone().requires_grad_(True)
            for value in feature_values
        )
        reference_features = tuple(
            value.detach().clone().requires_grad_(True)
            for value in feature_values
        )
        optimized_prepared = optimized.prepare(
            optimized_features,
            self.QUERY_SIZE,
        )
        reference_prepared = reference.prepare(
            reference_features,
            self.QUERY_SIZE,
        )

        optimized_query_groups = []
        reference_query_groups = []
        for block in range(4):
            query_values, _ = self.inputs()
            optimized_query_groups.append(
                tuple(
                    (value + 0.1 * block)
                    .detach()
                    .clone()
                    .requires_grad_(True)
                    for value in query_values
                )
            )
            reference_query_groups.append(
                tuple(
                    value.detach().clone().requires_grad_(True)
                    for value in optimized_query_groups[-1]
                )
            )

        optimized_outputs = [
            optimized.apply_prepared(queries, optimized_prepared)
            for queries in optimized_query_groups
        ]
        reference_outputs = [
            legacy_reference_apply_prepared(
                reference,
                queries,
                reference_prepared,
            )
            for queries in reference_query_groups
        ]

        optimized_loss = torch.zeros(())
        reference_loss = torch.zeros(())
        for block, (actual, expected) in enumerate(
            zip(optimized_outputs, reference_outputs)
        ):
            for level_index in range(4):
                with self.subTest(block=block, level=level_index):
                    assert_bits_equal(
                        self,
                        actual.queries[level_index],
                        expected.queries[level_index],
                    )
                    assert_bits_equal(
                        self,
                        actual.raw_gate_logits[level_index],
                        expected.raw_gate_logits[level_index],
                    )
                    assert_bits_equal(
                        self,
                        actual.normalized_logits[level_index],
                        expected.normalized_logits[level_index],
                    )
                    assert_bits_equal(
                        self,
                        actual.gates[level_index],
                        expected.gates[level_index],
                    )
                    assert_bits_equal(
                        self,
                        actual.factors[level_index],
                        expected.factors[level_index],
                    )
                probe = torch.linspace(
                    -1.5 + 0.1 * block,
                    2.0 + 0.2 * level_index,
                    actual.queries[level_index].numel(),
                ).reshape_as(actual.queries[level_index])
                optimized_loss = optimized_loss + (
                    actual.queries[level_index] * probe
                ).sum()
                reference_loss = reference_loss + (
                    expected.queries[level_index] * probe
                ).sum()

        assert_bits_equal(self, optimized_loss, reference_loss)
        optimized_loss.backward()
        reference_loss.backward()

        for optimized_group, reference_group in zip(
            optimized_query_groups,
            reference_query_groups,
        ):
            for actual, expected in zip(optimized_group, reference_group):
                self.assertIsNotNone(actual.grad)
                self.assertIsNotNone(expected.grad)
                assert_bits_equal(self, actual.grad, expected.grad)
        for actual, expected in zip(
            optimized_features,
            reference_features,
        ):
            self.assertIsNone(actual.grad)
            self.assertIsNone(expected.grad)

        optimized_parameters = dict(optimized.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        self.assertEqual(
            tuple(optimized_parameters),
            tuple(reference_parameters),
        )
        for name, actual in optimized_parameters.items():
            expected = reference_parameters[name]
            with self.subTest(parameter=name):
                self.assertIsNotNone(actual.grad)
                self.assertIsNotNone(expected.grad)
                torch.testing.assert_close(
                    actual.grad,
                    expected.grad,
                    rtol=3e-5,
                    atol=3e-6,
                )

    def test_wrapper_and_each_level_validate_independent_owner_tokens(
        self,
    ) -> None:
        torch.manual_seed(107)
        gate = self.gate()
        queries, features = self.inputs()
        prepared = gate.prepare(features, self.QUERY_SIZE)

        foreign = self.gate()
        foreign_prepared = foreign.prepare(features, self.QUERY_SIZE)
        with self.assertRaisesRegex(ValueError, "different gate instance"):
            gate.apply_prepared(queries, foreign_prepared)

        swapped = PreparedQueryFrequencyGateV2CROA(
            levels=(
                prepared.levels[1],
                prepared.levels[0],
                prepared.levels[2],
                prepared.levels[3],
            ),
            _owner_token=prepared._owner_token,
        )
        with self.assertRaisesRegex(ValueError, "different gate level"):
            gate.apply_prepared(queries, swapped)

        with self.assertRaisesRegex(
            TypeError,
            "PreparedQueryFrequencyGateV2CROA",
        ):
            gate.apply_prepared(queries, object())  # type: ignore[arg-type]

    def test_detached_source_and_zero_point_first_backward_contract(
        self,
    ) -> None:
        torch.manual_seed(109)
        level = QueryFrequencyLevelGateV2CROA(
            2,
            hidden_channels=3,
            expected_alignment=1,
        )
        query = torch.randn(2, 4, 2, 3, requires_grad=True)
        feature = torch.randn(2, 2, 4, 6, requires_grad=True)
        prepared = level.prepare(feature, (2, 3))
        prepared.raw_gate_logits.retain_grad()
        modulated, raw, normalized, gate, factor = level.apply_prepared(
            query,
            prepared,
        )
        probe = torch.linspace(
            -1.0,
            1.0,
            modulated.numel(),
        ).reshape_as(modulated)
        (modulated * probe).sum().backward()

        assert_bits_equal(self, modulated, query)
        self.assertTrue(torch.equal(raw, torch.zeros_like(raw)))
        self.assertTrue(torch.equal(normalized, torch.zeros_like(normalized)))
        self.assertTrue(torch.equal(gate, torch.zeros_like(gate)))
        self.assertTrue(torch.equal(factor, torch.ones_like(factor)))
        assert_bits_equal(self, query.grad, probe)
        self.assertIsNone(feature.grad)
        self.assertIsNotNone(prepared.raw_gate_logits.grad)
        self.assertTrue(
            bool(torch.isfinite(prepared.raw_gate_logits.grad).all())
        )
        self.assertGreater(
            float(prepared.raw_gate_logits.grad.abs().sum()),
            0.0,
        )

        self.assertIsNotNone(level.gate_out.weight.grad)
        self.assertTrue(bool(torch.isfinite(level.gate_out.weight.grad).all()))
        self.assertGreater(float(level.gate_out.weight.grad.norm()), 0.0)
        self.assertIsNotNone(level.alpha.grad)
        self.assertEqual(float(level.alpha.grad.abs().sum()), 0.0)
        for parameter in (
            level.prior_projection.weight,
            level.spatial_projection[0].weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertEqual(float(parameter.grad.abs().sum()), 0.0)

    def test_first_adam_step_preserves_shared_parameters_and_state(
        self,
    ) -> None:
        torch.manual_seed(113)
        shared_off = nn.Conv2d(3, 4, kernel_size=1)
        shared_on = copy.deepcopy(shared_off)
        gate = QueryFrequencyLevelGateV2CROA(
            2,
            hidden_channels=3,
            expected_alignment=1,
        )
        optimizer_off = torch.optim.Adam(shared_off.parameters(), lr=2e-4)
        optimizer_on = torch.optim.Adam(
            tuple(shared_on.parameters()) + tuple(gate.parameters()),
            lr=2e-4,
        )
        input_map = torch.randn(2, 3, 2, 3)
        feature = torch.randn(2, 2, 4, 6, requires_grad=True)
        target = torch.randn(2, 4, 2, 3)
        spatial_weight = torch.linspace(
            0.25,
            1.75,
            target.numel(),
        ).reshape_as(target)

        output_off = shared_off(input_map)
        query_on = shared_on(input_map)
        output_on = gate(query_on, feature)[0]
        assert_bits_equal(self, output_on, output_off)
        loss_off = ((output_off - target).square() * spatial_weight).mean()
        loss_on = ((output_on - target).square() * spatial_weight).mean()
        assert_bits_equal(self, loss_on, loss_off)
        loss_off.backward()
        loss_on.backward()

        off_parameters = tuple(shared_off.parameters())
        on_parameters = tuple(shared_on.parameters())
        for off_parameter, on_parameter in zip(
            off_parameters,
            on_parameters,
        ):
            self.assertIsNotNone(off_parameter.grad)
            self.assertIsNotNone(on_parameter.grad)
            assert_bits_equal(self, on_parameter.grad, off_parameter.grad)
        self.assertIsNone(feature.grad)
        self.assertGreater(float(gate.gate_out.weight.grad.norm()), 0.0)
        self.assertEqual(float(gate.alpha.grad.abs().sum()), 0.0)

        gate_before = {
            name: parameter.detach().clone()
            for name, parameter in gate.named_parameters()
        }
        optimizer_off.step()
        optimizer_on.step()
        for off_parameter, on_parameter in zip(
            off_parameters,
            on_parameters,
        ):
            assert_bits_equal(self, on_parameter, off_parameter)
            assert_adam_parameter_state_equal(
                self,
                optimizer_on,
                on_parameter,
                optimizer_off,
                off_parameter,
            )
        for name, parameter in gate.named_parameters():
            with self.subTest(qfg_parameter=name):
                changed = not torch.equal(parameter, gate_before[name])
                self.assertEqual(changed, name == "gate_out.weight")
                self.assertTrue(bool(torch.isfinite(parameter).all()))

    def test_second_step_can_train_hidden_branch_and_alpha(self) -> None:
        torch.manual_seed(127)
        level = QueryFrequencyLevelGateV2CROA(
            2,
            hidden_channels=3,
            expected_alignment=1,
        )
        optimizer = torch.optim.Adam(level.parameters(), lr=1e-4)
        query = torch.randn(2, 4, 2, 3)
        feature = torch.randn(2, 2, 4, 6)
        probe = torch.linspace(-2.0, 3.0, query.numel()).reshape_as(query)

        optimizer.zero_grad(set_to_none=True)
        first = (level(query, feature)[0] * probe).sum()
        first.backward()
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        second_output = level(query, feature)
        self.assertFalse(torch.equal(second_output[0], query))
        second = (second_output[0] * probe).sum()
        second.backward()
        self.assertGreater(float(level.alpha.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(level.prior_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(level.spatial_projection[0].weight.grad.abs().sum()),
            0.0,
        )
        for parameter in level.parameters():
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_formal_parameters_state_keys_and_manifest_are_frozen(self) -> None:
        gate = QueryOnlyFrequencyGateV2CROA()
        manifest = validate_formal_qfg_v2_croa(gate)
        self.assertEqual(
            frequency_gate_parameter_count(gate),
            PRODUCTION_QFG_V2_CROA_PARAMETERS,
        )
        self.assertEqual(PRODUCTION_QFG_V2_CROA_PARAMETERS, 15_684)
        self.assertEqual(len(gate.state_dict()), 20)
        self.assertEqual(PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT, 20)
        self.assertEqual(
            tuple(name for name, _ in gate.named_parameters()),
            FORMAL_QFG_V2_CROA_PARAMETER_KEYS,
        )
        self.assertEqual(
            tuple(gate.state_dict()),
            FORMAL_QFG_V2_CROA_STATE_KEYS,
        )
        self.assertEqual(
            manifest["registered_alignment_ratios"],
            FORMAL_ALIGNMENT_RATIOS,
        )
        self.assertEqual(manifest["rms_eps"], 1e-6)
        self.assertTrue(manifest["detach_frequency_source"])
        self.assertFalse(manifest["terminal_projection_bias"])
        for level in gate.levels:
            self.assertIsNone(level.gate_out.bias)
            self.assertEqual(
                int(torch.count_nonzero(level.gate_out.weight)),
                0,
            )
            self.assertAlmostEqual(
                float(torch.tanh(level.alpha.detach())),
                0.1,
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
