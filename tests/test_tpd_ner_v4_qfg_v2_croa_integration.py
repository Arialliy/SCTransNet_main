from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

from model.tpd_forward_contract import TPDForwardOutput, evaluator_prediction
from model.tpd_frequency_gate_v2_croa import (
    QueryFrequencyGateOutputV2CROA,
    _working_dtype,
    centered_bounded_arctangent_gate,
    spatial_center_rms_normalize,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    FORMAL_QFG_ALPHA_EFFECTIVE_INIT,
    FORMAL_QFG_DETACH_FREQUENCY_SOURCE,
    FORMAL_QFG_HIDDEN_CHANNELS,
    FORMAL_QFG_MODE,
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_QFG_V2_CROA_PARAMETERS,
    PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
    _build_formal_qfg,
    build_formal_v4_qfg_v2_croa_survival_model,
    validate_formal_qfg_v2_croa_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
    TPDNERV8MPRSDCHV4SurvivalSCTransNet,
    build_formal_v4_survival_model,
)


torch.set_num_threads(1)


def assert_tensor_bits_equal(
    testcase: unittest.TestCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    testcase.assertEqual(actual.dtype, expected.dtype, msg=name)
    testcase.assertEqual(actual.device, expected.device, msg=name)
    testcase.assertEqual(tuple(actual.shape), tuple(expected.shape), msg=name)
    testcase.assertTrue(
        torch.equal(
            actual.detach().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().contiguous().reshape(-1).view(torch.uint8),
        ),
        msg=name,
    )


def legacy_outputs(value) -> tuple[torch.Tensor, ...]:
    if isinstance(value, TPDForwardOutput):
        value = value.segmentation
    if not isinstance(value, tuple) or len(value) != 6:
        raise TypeError("expected the legacy six-output segmentation tuple")
    return value


def spatially_weighted_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    losses = []
    for index, output in enumerate(outputs):
        weights = torch.linspace(
            0.25,
            1.75,
            output.shape[-2] * output.shape[-1],
            device=output.device,
            dtype=output.dtype,
        ).reshape(1, 1, output.shape[-2], output.shape[-1])
        losses.append((index + 1) * (output * weights).mean())
    return torch.stack(losses).sum()


class LegacyQFGApplyAdapter:
    """Test-only pre-cache QFG apply path.

    The production ``prepare`` call still constructs the raw logits, while
    this adapter deliberately ignores its cached transforms and recomputes
    normalization, the bounded gate, and the factor for every SCTB use.
    """

    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def apply_prepared(self, queries, prepared) -> QueryFrequencyGateOutputV2CROA:
        outputs = []
        for query, level, prepared_level in zip(
            tuple(queries),
            self.delegate.levels,
            prepared.levels,
        ):
            raw_gate_logits = prepared_level.raw_gate_logits
            normalized_logits = spatial_center_rms_normalize(
                raw_gate_logits,
                eps=level.eps,
                validate_finite=level.validate_finite,
            )
            gate = centered_bounded_arctangent_gate(
                normalized_logits,
                validate_finite=level.validate_finite,
            )
            compute_dtype = _working_dtype(query.dtype, gate.dtype)
            factor = 1.0 + torch.tanh(
                level.alpha.to(dtype=compute_dtype)
            ) * gate.to(dtype=compute_dtype)
            modulated = (
                query.to(dtype=compute_dtype) * factor
            ).to(dtype=query.dtype)
            outputs.append(
                (
                    modulated,
                    raw_gate_logits,
                    normalized_logits,
                    gate,
                    factor,
                )
            )

        (
            modulated_queries,
            raw_gate_logits,
            normalized_logits,
            gates,
            factors,
        ) = zip(*outputs)
        return QueryFrequencyGateOutputV2CROA(
            queries=tuple(modulated_queries),
            raw_gate_logits=tuple(raw_gate_logits),
            normalized_logits=tuple(normalized_logits),
            gates=tuple(gates),
            factors=tuple(factors),
        )


def rng_snapshot() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    return (
        torch.get_rng_state().clone(),
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    )


def restore_rng(
    snapshot: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
) -> None:
    cpu_state, cuda_states = snapshot
    torch.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(list(cuda_states))


def assert_rng_bits_equal(
    testcase: unittest.TestCase,
    actual: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
    expected: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
    *,
    name: str,
) -> None:
    testcase.assertTrue(torch.equal(actual[0], expected[0]), msg=f"{name}.cpu")
    testcase.assertEqual(len(actual[1]), len(expected[1]), msg=f"{name}.cuda")
    for index, (actual_state, expected_state) in enumerate(
        zip(actual[1], expected[1])
    ):
        testcase.assertTrue(
            torch.equal(actual_state, expected_state),
            msg=f"{name}.cuda[{index}]",
        )


class V4QFGV2CROAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.control, cls.control_metadata = build_formal_v4_survival_model()
        cls.model, cls.metadata = (
            build_formal_v4_qfg_v2_croa_survival_model()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.control
        del cls.control_metadata
        del cls.model
        del cls.metadata

    def setUp(self) -> None:
        self.control.eval()
        self.model.eval()
        self.control.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)

    def test_strict_state_parameter_builder_validator_and_manifest(self) -> None:
        self.assertIs(
            type(self.model),
            TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
        )
        state = self.model.state_dict()
        qfg_keys = {
            key for key in state if key.startswith(QFG_STATE_PREFIX)
        }
        self.assertEqual(
            len(state),
            FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(
            len(state) - len(qfg_keys),
            FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(len(qfg_keys), PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT)
        self.assertEqual(qfg_keys, set(QFG_STATE_KEYS))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in self.model.tpd_qfg.parameters()
            ),
            PRODUCTION_QFG_V2_CROA_PARAMETERS,
        )

        validated = validate_formal_qfg_v2_croa_survival_model(
            self.model,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
        )
        self.assertEqual(
            validated["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(
            validated["qfg_state_key_count"],
            PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
        )
        self.assertEqual(
            validated["total_parameters"],
            PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
        )
        manifest = validated["architecture_manifest"]
        self.assertEqual(manifest["qfg_frequency_mode"], FORMAL_QFG_MODE)
        self.assertEqual(
            manifest["qfg_hidden_channels"],
            FORMAL_QFG_HIDDEN_CHANNELS,
        )
        self.assertIs(
            manifest["qfg_detach_frequency_source"],
            FORMAL_QFG_DETACH_FREQUENCY_SOURCE,
        )
        self.assertEqual(
            manifest["qfg_alpha_effective_initialization"],
            FORMAL_QFG_ALPHA_EFFECTIVE_INIT,
        )
        self.assertEqual(
            manifest["qfg_modified_attention_tensors"],
            ("Q",),
        )
        self.assertFalse(manifest["qfg_kv_modified"])
        self.assertFalse(manifest["qfg_cfn_modified"])
        self.assertFalse(manifest["qfg_decoder_injection"])
        self.assertTrue(manifest["qfg_inference_required"])
        self.assertTrue(manifest["segmentation_path_modified"])

    def test_trained_qfg_values_pass_nonidentity_validation_only(self) -> None:
        trained = copy.deepcopy(self.model)
        with torch.no_grad():
            trained.tpd_qfg.levels[0].alpha.add_(0.25)
            trained.tpd_qfg.levels[0].gate_out.weight.fill_(0.01)

        validated = validate_formal_qfg_v2_croa_survival_model(
            trained,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=False,
        )
        self.assertEqual(
            validated["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
        )
        with self.assertRaisesRegex(
            (ValueError, RuntimeError),
            "terminal",
        ):
            validate_formal_qfg_v2_croa_survival_model(
                trained,
                require_zero_initialized_heads=True,
                require_identity_initialized_qfg=True,
            )

        alpha_only = copy.deepcopy(self.model)
        with torch.no_grad():
            alpha_only.tpd_qfg.levels[0].alpha.add_(0.25)
        with self.assertRaisesRegex(
            (ValueError, RuntimeError),
            "alpha",
        ):
            validate_formal_qfg_v2_croa_survival_model(
                alpha_only,
                require_zero_initialized_heads=True,
                require_identity_initialized_qfg=True,
            )

        terminal_only = copy.deepcopy(self.model)
        with torch.no_grad():
            terminal_only.tpd_qfg.levels[0].gate_out.weight.fill_(0.01)
        with self.assertRaisesRegex(
            (ValueError, RuntimeError),
            "terminal",
        ):
            validate_formal_qfg_v2_croa_survival_model(
                terminal_only,
                require_zero_initialized_heads=True,
                require_identity_initialized_qfg=True,
            )

    def test_qfg_construction_is_cpu_and_cuda_rng_neutral_and_bitwise(
        self,
    ) -> None:
        reference = torch.empty((), dtype=torch.float32)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )

        qfg = _build_formal_qfg(reference, training=True)

        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_before))
        cuda_after = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )
        self.assertEqual(len(cuda_after), len(cuda_before))
        for index, (actual, expected) in enumerate(
            zip(cuda_after, cuda_before)
        ):
            self.assertTrue(torch.equal(actual, expected), msg=index)

        expected_state = self.model.tpd_qfg.state_dict()
        actual_state = qfg.state_dict()
        self.assertEqual(set(actual_state), set(expected_state))
        for name, expected in expected_state.items():
            assert_tensor_bits_equal(
                self,
                actual_state[name],
                expected,
                name=f"tpd_qfg.{name}",
            )

    def test_frozen_tpd_ner_survival_state_is_bitwise_identity(self) -> None:
        control_state = self.control.state_dict()
        integrated_state = self.model.state_dict()
        self.assertEqual(len(control_state), FORMAL_V4_SURVIVAL_STATE_KEY_COUNT)
        self.assertEqual(
            set(control_state),
            {
                key
                for key in integrated_state
                if not key.startswith(QFG_STATE_PREFIX)
            },
        )
        for name, expected in control_state.items():
            assert_tensor_bits_equal(
                self,
                integrated_state[name],
                expected,
                name=name,
            )

        for prefix in (
            "mtc.embeddings_1.",
            "mtc.embeddings_2.",
            "tpd_ner.",
            "target_survival.",
        ):
            matching = [
                name for name in control_state if name.startswith(prefix)
            ]
            self.assertTrue(matching, msg=prefix)
            for name in matching:
                assert_tensor_bits_equal(
                    self,
                    integrated_state[name],
                    control_state[name],
                    name=name,
                )

    def test_first_adam_step_is_a_bitwise_shared_parameter_anchor(
        self,
    ) -> None:
        control = copy.deepcopy(self.control).eval()
        integrated = copy.deepcopy(self.model).eval()
        control.zero_grad(set_to_none=True)
        integrated.zero_grad(set_to_none=True)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(2704)
        images = torch.randn(
            1,
            1,
            32,
            32,
            generator=generator,
        )
        control_optimizer = torch.optim.Adam(
            control.parameters(),
            lr=1.0e-4,
        )
        integrated_optimizer = torch.optim.Adam(
            integrated.parameters(),
            lr=1.0e-4,
        )
        qfg_before = {
            name: parameter.detach().clone()
            for name, parameter in integrated.tpd_qfg.named_parameters()
        }

        control_loss = spatially_weighted_loss(
            legacy_outputs(control(images))
        )
        integrated_loss = spatially_weighted_loss(
            legacy_outputs(integrated(images))
        )
        assert_tensor_bits_equal(
            self,
            integrated_loss,
            control_loss,
            name="first_adam_loss",
        )
        control_loss.backward()
        integrated_loss.backward()
        control_optimizer.step()
        integrated_optimizer.step()

        control_parameters = dict(control.named_parameters())
        integrated_parameters = dict(integrated.named_parameters())
        shared_names = {
            name
            for name in integrated_parameters
            if not name.startswith(QFG_STATE_PREFIX)
        }
        self.assertEqual(shared_names, set(control_parameters))
        expected_adam_state_keys = {"step", "exp_avg", "exp_avg_sq"}
        for name in sorted(shared_names):
            control_parameter = control_parameters[name]
            integrated_parameter = integrated_parameters[name]
            assert_tensor_bits_equal(
                self,
                integrated_parameter,
                control_parameter,
                name=f"{name}.first_adam_value",
            )

            control_state = control_optimizer.state.get(control_parameter)
            integrated_state = integrated_optimizer.state.get(
                integrated_parameter
            )
            self.assertIs(
                integrated_state is None,
                control_state is None,
                msg=f"{name}.first_adam_state_presence",
            )
            if control_state is None:
                continue
            self.assertEqual(
                set(control_state),
                expected_adam_state_keys,
                msg=f"{name}.control_adam_state_keys",
            )
            self.assertEqual(
                set(integrated_state),
                expected_adam_state_keys,
                msg=f"{name}.integrated_adam_state_keys",
            )
            for state_name in sorted(expected_adam_state_keys):
                assert_tensor_bits_equal(
                    self,
                    integrated_state[state_name],
                    control_state[state_name],
                    name=f"{name}.adam.{state_name}",
                )

        qfg_parameters = dict(integrated.tpd_qfg.named_parameters())
        gate_out_names = {
            name
            for name in qfg_parameters
            if name.endswith(".gate_out.weight")
        }
        self.assertEqual(len(gate_out_names), 4)
        changed_qfg_names = set()
        for name, parameter in qfg_parameters.items():
            state = integrated_optimizer.state.get(parameter)
            if name in gate_out_names:
                self.assertIsNotNone(
                    state,
                    msg=f"tpd_qfg.{name}.adam_state",
                )
                self.assertEqual(
                    set(state),
                    expected_adam_state_keys,
                    msg=f"tpd_qfg.{name}.adam_state_keys",
                )
                self.assertEqual(float(state["step"]), 1.0, msg=name)
                self.assertGreater(
                    int(torch.count_nonzero(state["exp_avg"])),
                    0,
                    msg=f"tpd_qfg.{name}.exp_avg",
                )
                self.assertGreater(
                    int(torch.count_nonzero(state["exp_avg_sq"])),
                    0,
                    msg=f"tpd_qfg.{name}.exp_avg_sq",
                )
                self.assertFalse(
                    torch.equal(parameter.detach(), qfg_before[name]),
                    msg=f"tpd_qfg.{name}.value",
                )
                changed_qfg_names.add(name)
                continue

            assert_tensor_bits_equal(
                self,
                parameter,
                qfg_before[name],
                name=f"tpd_qfg.{name}.first_adam_value",
            )
            if state is None:
                continue
            self.assertEqual(
                set(state),
                expected_adam_state_keys,
                msg=f"tpd_qfg.{name}.adam_state_keys",
            )
            self.assertEqual(
                int(torch.count_nonzero(state["exp_avg"])),
                0,
                msg=f"tpd_qfg.{name}.exp_avg",
            )
            self.assertEqual(
                int(torch.count_nonzero(state["exp_avg_sq"])),
                0,
                msg=f"tpd_qfg.{name}.exp_avg_sq",
            )

        self.assertEqual(changed_qfg_names, gate_out_names)

    def test_nonzero_qfg_cached_apply_matches_legacy_whole_model(self) -> None:
        legacy = copy.deepcopy(self.model)
        cached = copy.deepcopy(self.model)
        with torch.no_grad():
            for index, level in enumerate(legacy.tpd_qfg.levels):
                terminal = torch.linspace(
                    -0.035 - index * 0.002,
                    0.041 + index * 0.002,
                    level.gate_out.weight.numel(),
                    dtype=level.gate_out.weight.dtype,
                    device=level.gate_out.weight.device,
                ).reshape_as(level.gate_out.weight)
                level.gate_out.weight.copy_(terminal)
                level.alpha.fill_(0.2 + index * 0.05)
        cached.load_state_dict(legacy.state_dict(), strict=True)

        legacy_state = legacy.state_dict()
        cached_state = cached.state_dict()
        self.assertEqual(tuple(legacy_state), tuple(cached_state))
        for name in legacy_state:
            assert_tensor_bits_equal(
                self,
                cached_state[name],
                legacy_state[name],
                name=f"{name}.initial_state",
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(2711)
        images = torch.randn(
            1,
            1,
            32,
            32,
            generator=generator,
            dtype=torch.float32,
        )
        generator_before = generator.get_state().clone()
        original_rng = rng_snapshot()
        adapter = LegacyQFGApplyAdapter(legacy.tpd_qfg)
        try:
            legacy.eval()
            cached.eval()
            eval_rng = rng_snapshot()
            restore_rng(eval_rng)
            with (
                torch.no_grad(),
                mock.patch.object(
                    legacy.tpd_qfg,
                    "apply_prepared",
                    new=adapter.apply_prepared,
                ),
            ):
                legacy_eval = legacy_outputs(legacy(images))
            legacy_eval_rng = rng_snapshot()

            restore_rng(eval_rng)
            with torch.no_grad():
                cached_eval = legacy_outputs(cached(images))
            cached_eval_rng = rng_snapshot()
            assert_rng_bits_equal(
                self,
                cached_eval_rng,
                legacy_eval_rng,
                name="eval_rng",
            )
            for index, (actual, expected) in enumerate(
                zip(cached_eval, legacy_eval)
            ):
                assert_tensor_bits_equal(
                    self,
                    actual,
                    expected,
                    name=f"eval.segmentation[{index}]",
                )

            legacy.train()
            cached.train()
            legacy.zero_grad(set_to_none=True)
            cached.zero_grad(set_to_none=True)
            legacy_optimizer = torch.optim.Adam(
                legacy.parameters(),
                lr=1.0e-4,
            )
            cached_optimizer = torch.optim.Adam(
                cached.parameters(),
                lr=1.0e-4,
            )
            train_rng = rng_snapshot()

            restore_rng(train_rng)
            with mock.patch.object(
                legacy.tpd_qfg,
                "apply_prepared",
                new=adapter.apply_prepared,
            ):
                legacy_train = legacy(images)
            self.assertIsInstance(legacy_train, TPDForwardOutput)
            legacy_loss = spatially_weighted_loss(
                legacy_outputs(legacy_train)
            )
            legacy_loss.backward()
            legacy_optimizer.step()
            legacy_train_rng = rng_snapshot()

            restore_rng(train_rng)
            cached_train = cached(images)
            self.assertIsInstance(cached_train, TPDForwardOutput)
            cached_loss = spatially_weighted_loss(
                legacy_outputs(cached_train)
            )
            cached_loss.backward()
            cached_optimizer.step()
            cached_train_rng = rng_snapshot()
            assert_rng_bits_equal(
                self,
                cached_train_rng,
                legacy_train_rng,
                name="train_rng",
            )

            for index, (actual, expected) in enumerate(
                zip(
                    legacy_outputs(cached_train),
                    legacy_outputs(legacy_train),
                )
            ):
                assert_tensor_bits_equal(
                    self,
                    actual,
                    expected,
                    name=f"train.segmentation[{index}]",
                )
            for field in (
                "emb1_endpoint",
                "emb2_endpoint",
                "emb1_survival_logits",
                "emb2_survival_logits",
            ):
                actual = getattr(cached_train, field)
                expected = getattr(legacy_train, field)
                self.assertIsNotNone(actual, msg=field)
                self.assertIsNotNone(expected, msg=field)
                assert_tensor_bits_equal(
                    self,
                    actual,
                    expected,
                    name=f"train.{field}",
                )
            assert_tensor_bits_equal(
                self,
                cached_loss,
                legacy_loss,
                name="train.loss",
            )

            legacy_parameters = dict(legacy.named_parameters())
            cached_parameters = dict(cached.named_parameters())
            self.assertEqual(
                tuple(legacy_parameters),
                tuple(cached_parameters),
            )
            expected_adam_state_keys = {"step", "exp_avg", "exp_avg_sq"}
            for name, legacy_parameter in legacy_parameters.items():
                cached_parameter = cached_parameters[name]
                legacy_gradient = legacy_parameter.grad
                cached_gradient = cached_parameter.grad
                self.assertIs(
                    cached_gradient is None,
                    legacy_gradient is None,
                    msg=f"{name}.gradient_presence",
                )
                is_qfg = name.startswith(QFG_STATE_PREFIX)
                if cached_gradient is not None:
                    if is_qfg:
                        torch.testing.assert_close(
                            cached_gradient,
                            legacy_gradient,
                            rtol=1.0e-5,
                            atol=1.0e-7,
                            msg=f"{name}.grad",
                        )
                    else:
                        assert_tensor_bits_equal(
                            self,
                            cached_gradient,
                            legacy_gradient,
                            name=f"{name}.grad",
                        )

                if is_qfg:
                    torch.testing.assert_close(
                        cached_parameter,
                        legacy_parameter,
                        rtol=1.0e-5,
                        atol=1.0e-7,
                        msg=f"{name}.first_adam_value",
                    )
                else:
                    assert_tensor_bits_equal(
                        self,
                        cached_parameter,
                        legacy_parameter,
                        name=f"{name}.first_adam_value",
                    )

                legacy_adam = legacy_optimizer.state.get(legacy_parameter)
                cached_adam = cached_optimizer.state.get(cached_parameter)
                self.assertIs(
                    cached_adam is None,
                    legacy_adam is None,
                    msg=f"{name}.adam_state_presence",
                )
                if legacy_adam is None:
                    continue
                self.assertEqual(
                    set(legacy_adam),
                    expected_adam_state_keys,
                    msg=f"{name}.legacy_adam_state_keys",
                )
                self.assertEqual(
                    set(cached_adam),
                    expected_adam_state_keys,
                    msg=f"{name}.cached_adam_state_keys",
                )
                for state_name in sorted(expected_adam_state_keys):
                    actual = cached_adam[state_name]
                    expected = legacy_adam[state_name]
                    if is_qfg and state_name != "step":
                        torch.testing.assert_close(
                            actual,
                            expected,
                            rtol=1.0e-5,
                            atol=1.0e-7,
                            msg=f"{name}.adam.{state_name}",
                        )
                    else:
                        assert_tensor_bits_equal(
                            self,
                            actual,
                            expected,
                            name=f"{name}.adam.{state_name}",
                        )

            legacy_state = legacy.state_dict()
            cached_state = cached.state_dict()
            self.assertEqual(tuple(legacy_state), tuple(cached_state))
            for name, expected in legacy_state.items():
                actual = cached_state[name]
                if name.startswith(QFG_STATE_PREFIX):
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=1.0e-5,
                        atol=1.0e-7,
                        msg=f"{name}.trained_state",
                    )
                else:
                    assert_tensor_bits_equal(
                        self,
                        actual,
                        expected,
                        name=f"{name}.trained_state",
                    )
            self.assertTrue(
                torch.equal(generator.get_state(), generator_before),
            )
        finally:
            restore_rng(original_rng)

    def test_identity_qfg_matches_initial_six_outputs_bit_for_bit(self) -> None:
        torch.manual_seed(2701)
        images = torch.randn(1, 1, 32, 32)
        prepare = self.model.tpd_qfg.prepare
        apply_prepared = self.model.tpd_qfg.apply_prepared
        with (
            torch.no_grad(),
            mock.patch.object(
                self.model.tpd_qfg,
                "prepare",
                wraps=prepare,
            ) as prepare_mock,
            mock.patch.object(
                self.model.tpd_qfg,
                "apply_prepared",
                wraps=apply_prepared,
            ) as apply_mock,
        ):
            expected = legacy_outputs(self.control(images))
            actual = legacy_outputs(self.model(images))

        self.assertEqual(prepare_mock.call_count, 1)
        self.assertEqual(
            apply_mock.call_count,
            len(self.model.mtc.encoder.layer),
        )
        for index, (actual_level, expected_level) in enumerate(
            zip(actual, expected)
        ):
            assert_tensor_bits_equal(
                self,
                actual_level,
                expected_level,
                name=f"segmentation[{index}]",
            )

    def test_identity_qfg_preserves_every_shared_raw_gradient(self) -> None:
        torch.manual_seed(2702)
        images = torch.randn(1, 1, 32, 32)

        expected = legacy_outputs(self.control(images))
        spatially_weighted_loss(expected).backward()
        actual = legacy_outputs(self.model(images))
        spatially_weighted_loss(actual).backward()

        control_parameters = dict(self.control.named_parameters())
        integrated_parameters = dict(self.model.named_parameters())
        shared_names = tuple(
            name
            for name in integrated_parameters
            if not name.startswith(QFG_STATE_PREFIX)
        )
        self.assertEqual(set(shared_names), set(control_parameters))
        for name in shared_names:
            actual_gradient = integrated_parameters[name].grad
            expected_gradient = control_parameters[name].grad
            self.assertIs(
                actual_gradient is None,
                expected_gradient is None,
                msg=name,
            )
            if actual_gradient is not None:
                assert_tensor_bits_equal(
                    self,
                    actual_gradient,
                    expected_gradient,
                    name=f"{name}.grad",
                )

        terminal_gradient = 0.0
        for level in self.model.tpd_qfg.levels:
            self.assertIsNotNone(level.alpha.grad)
            self.assertEqual(
                int(torch.count_nonzero(level.alpha.grad)),
                0,
            )
            self.assertIsNotNone(level.gate_out.weight.grad)
            terminal_gradient += float(
                level.gate_out.weight.grad.detach().abs().sum()
            )
            for parameter in (
                level.prior_projection.weight,
                level.spatial_projection[0].weight,
            ):
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
        self.assertGreater(terminal_gradient, 0.0)

    def test_training_structured_output_eval_legacy_and_capture_contract(
        self,
    ) -> None:
        torch.manual_seed(2703)
        images = torch.randn(1, 1, 32, 32)
        self.model.train()
        prepare = self.model.tpd_qfg.prepare
        with mock.patch.object(
            self.model.tpd_qfg,
            "prepare",
            wraps=prepare,
        ) as prepare_mock:
            output = self.model(images)

        self.assertEqual(prepare_mock.call_count, 1)
        self.assertIsInstance(output, TPDForwardOutput)
        self.assertIsInstance(output.segmentation, tuple)
        self.assertEqual(len(output.segmentation), 6)
        self.assertEqual(tuple(output.emb1_endpoint.shape), (1, 32, 2, 2))
        self.assertEqual(tuple(output.emb2_endpoint.shape), (1, 64, 2, 2))
        self.assertEqual(
            tuple(output.emb1_survival_logits.shape),
            (1, 1, 2, 2),
        )
        self.assertEqual(
            tuple(output.emb2_survival_logits.shape),
            (1, 1, 2, 2),
        )
        self.assertFalse(self.model._survival_capture_active)
        self.assertIsNone(self.model._captured_survival_endpoints)

        self.model.eval()
        with (
            torch.no_grad(),
            mock.patch.object(
                self.model.target_survival,
                "forward",
                wraps=self.model.target_survival.forward,
            ) as survival_forward,
        ):
            legacy = self.model(images)
        self.assertIsInstance(legacy, tuple)
        self.assertEqual(len(legacy), 6)
        self.assertEqual(survival_forward.call_count, 0)
        self.assertIs(evaluator_prediction(legacy), legacy[-1])

    def test_integration_is_a_subclass_without_overriding_forward_or_capture(
        self,
    ) -> None:
        self.assertTrue(
            issubclass(
                TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
                TPDNERV8MPRSDCHV4SurvivalSCTransNet,
            )
        )
        self.assertNotIn(
            "forward",
            TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet.__dict__,
        )
        self.assertNotIn(
            "explicit_embeddings",
            TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet.__dict__,
        )
        self.assertIn(
            "_forward_with_relay",
            TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet.__dict__,
        )


if __name__ == "__main__":
    unittest.main()
