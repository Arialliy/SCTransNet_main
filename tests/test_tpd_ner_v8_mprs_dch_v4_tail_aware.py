from __future__ import annotations

import gc
import importlib
import math
import sys
import unittest
from pathlib import Path
from typing import Mapping, Sequence, Tuple, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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
    build_clean_v8_mprs_dch_patch_embedding,
)
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    PRODUCTION_PARENT_PARAMETERS,
    TPDNERV8MPRSDCHSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v2 import (
    RMSBalancedCenteredEvidenceRelay,
    V2_SKIP_FACTOR_BOUNDS,
)
from model.tpd_ner_v8_mprs_dch_v3 import (
    PRODUCTION_V3_RELAY_ON_PARAMETERS,
    PRODUCTION_V3_RELAY_PARAMETERS,
    RMSBalancedCenteredDCOffsetEvidenceRelay,
    TPDNERV8MPRSDCHV3SCTransNet,
    adapt_v8_mprs_dch_parent_v3,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_TAIL_Z_THRESHOLDS,
    PRODUCTION_V4_RELAY_ON_PARAMETERS,
    PRODUCTION_V4_RELAY_PARAMETERS,
    SUPPORTED_DC_SUPPORT_MODES,
    V4_RELAY_VERSION,
    TPDNERV8MPRSDCHV4SCTransNet,
    TailAwarePersistentDCOffsetEvidenceRelay,
    TailDCSupportMode,
    adapt_v8_mprs_dch_parent_v4,
    relay_spatial_tail_support,
    v4_relay_parameter_count,
)


FULL = "tpd_clean_v8_mprs_dch_full"
CAPACITY = "tpd_clean_v8_mprs_dch_capacity"
RELAY_OFFSET_KEYS = {
    "dc_offsets.4",
    "dc_offsets.3",
    "dc_offsets.2",
}
FULL_OFFSET_KEYS = {
    "tpd_ner.dc_offsets.4",
    "tpd_ner.dc_offsets.3",
    "tpd_ner.dc_offsets.2",
}

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
    torch.manual_seed(seed)
    model = SCTransNet(
        _small_config(),
        img_size=32,
        mode="train",
        deepsuper=True,
    )
    model.apply(weights_init_kaiming)
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


def _six_output_loss(
    outputs: object,
    target: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("expected exactly six deep-supervision outputs")
    criterion = nn.BCELoss(reduction="mean")
    return sum(criterion(output, target) for output in outputs)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _spike(
    *,
    shape: Tuple[int, int, int, int],
    point: Tuple[int, int],
    magnitude: float = 20.0,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> torch.Tensor:
    value = torch.zeros(shape, dtype=dtype)
    value[:, :, point[0], point[1]] = magnitude
    return value.requires_grad_(requires_grad)


def _stage_sources(
    stage: int,
    parent: torch.Tensor,
    *,
    spatial_size: Tuple[int, int],
) -> Tuple[torch.Tensor, ...]:
    height, width = spatial_size
    if stage == 3:
        return (
            torch.zeros(1, 2, height, width),
            torch.zeros(1, 4, height, width),
            parent,
            torch.zeros(1, 8, height, width),
        )
    if stage == 2:
        return (
            torch.zeros(1, 2, height, width),
            parent,
            torch.zeros(1, 4, height, width),
        )
    raise ValueError(f"test helper only accepts stage 3 or 2, got {stage}")


class _FixedFusion(nn.Module):
    """Return a prescribed relay value while retaining its autograd identity."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        del sources
        if tuple(self.value.shape[-2:]) != tuple(output_size):
            raise ValueError("fixed test relay value has the wrong spatial size")
        return self.value


class V4ModuleAndStateContractTests(unittest.TestCase):
    def test_module_import_exports_and_fixed_constants(self) -> None:
        module = importlib.import_module(
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware"
        )
        module_path = Path(cast(str, module.__file__)).resolve()
        self.assertEqual(module_path.parent.name, "model")
        self.assertIs(
            module.TailAwarePersistentDCOffsetEvidenceRelay,
            TailAwarePersistentDCOffsetEvidenceRelay,
        )
        self.assertIn("relay_spatial_tail_support", module.__all__)
        self.assertIn("adapt_v8_mprs_dch_parent_v4", module.__all__)
        self.assertEqual(DEFAULT_RELAY_WIDTH, 8)
        self.assertEqual(DEFAULT_RELAY_INITIALIZATION_SEED, 42)
        self.assertEqual(
            dict(DEFAULT_TAIL_Z_THRESHOLDS),
            {4: 1.5, 3: 2.0, 2: 2.5},
        )
        self.assertEqual(DEFAULT_DC_SUPPORT_MODE, "complement_tail")
        self.assertEqual(
            SUPPORTED_DC_SUPPORT_MODES,
            ("legacy_global", "direct_tail", "complement_tail"),
        )
        self.assertEqual(
            V4_RELAY_VERSION,
            "v4_tail_aware_persistent_post_center_dch",
        )

    def test_relay_parameters_buffers_offsets_and_fresh_v3_pairing(self) -> None:
        torch.manual_seed(4201)
        v2 = RMSBalancedCenteredEvidenceRelay(base_channels=2)
        torch.manual_seed(4201)
        v3 = RMSBalancedCenteredDCOffsetEvidenceRelay(base_channels=2)
        torch.manual_seed(4201)
        v4 = TailAwarePersistentDCOffsetEvidenceRelay(base_channels=2)

        self.assertEqual(
            _parameter_count(v4),
            _parameter_count(v3),
        )
        self.assertEqual(
            PRODUCTION_V4_RELAY_PARAMETERS,
            PRODUCTION_V3_RELAY_PARAMETERS,
        )
        self.assertEqual(
            set(v4.state_dict()) - set(v2.state_dict()),
            RELAY_OFFSET_KEYS,
        )
        self.assertFalse(set(v2.state_dict()) - set(v4.state_dict()))
        self.assertEqual(tuple(v3.state_dict()), tuple(v4.state_dict()))
        self.assertFalse(tuple(v4.named_buffers()))

        for name, v3_value in v3.state_dict().items():
            self.assertTrue(
                torch.equal(v3_value, v4.state_dict()[name]),
                f"fresh V3/V4 relay state differs: {name}",
            )
        for stage in ("4", "3", "2"):
            self.assertEqual(
                int(torch.count_nonzero(v4.dc_offsets[stage])),
                0,
            )
            self.assertEqual(
                int(torch.count_nonzero(v4.gates[stage].weight)),
                0,
            )
            self.assertIsNone(v4.gates[stage].bias)

        expected_offsets = {"4": -0.125, "3": 0.0625, "2": -0.03125}
        with torch.no_grad():
            for stage, value in expected_offsets.items():
                v3.dc_offsets[stage].fill_(value)
        incompatible = v4.load_state_dict(v3.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)
        for stage, value in expected_offsets.items():
            self.assertEqual(float(v4.dc_offsets[stage].detach()), value)

    def test_complete_model_fresh_state_strict_load_and_explicit_only_transfer(
        self,
    ) -> None:
        parent = _small_parent(seed=4202)
        v3 = adapt_v8_mprs_dch_parent_v3(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        v4 = adapt_v8_mprs_dch_parent_v4(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        self.assertIsInstance(v3, TPDNERV8MPRSDCHV3SCTransNet)
        self.assertIsInstance(v4, TPDNERV8MPRSDCHV4SCTransNet)
        self.assertEqual(tuple(v3.state_dict()), tuple(v4.state_dict()))
        self.assertEqual(
            tuple(name for name, _ in v3.named_buffers()),
            tuple(name for name, _ in v4.named_buffers()),
        )
        for name, value in v3.state_dict().items():
            self.assertTrue(
                torch.equal(value, v4.state_dict()[name]),
                f"fresh V3/V4 full state differs: {name}",
            )

        with torch.no_grad():
            v3.tpd_ner.dc_offsets["3"].fill_(0.375)
        fresh_v4 = adapt_v8_mprs_dch_parent_v4(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        self.assertEqual(
            int(torch.count_nonzero(fresh_v4.tpd_ner.dc_offsets["3"])),
            0,
            "fresh formal construction must not implicitly use V3 state",
        )
        incompatible = fresh_v4.load_state_dict(v3.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertEqual(
            float(fresh_v4.tpd_ner.dc_offsets["3"].detach()),
            0.375,
            "V3 transfer must occur only through an explicit strict load",
        )
        del parent, v3, v4, fresh_v4
        gc.collect()

    def test_production_parameter_counts_relay_off_and_manifest(self) -> None:
        self.assertEqual(PRODUCTION_PARENT_PARAMETERS, TOTAL_PARAMETERS)
        self.assertEqual(
            PRODUCTION_V4_RELAY_PARAMETERS,
            PRODUCTION_V3_RELAY_PARAMETERS,
        )
        self.assertEqual(
            PRODUCTION_V4_RELAY_ON_PARAMETERS,
            PRODUCTION_V3_RELAY_ON_PARAMETERS,
        )

        parent, _ = build_clean_v8_mprs_dch_model(FULL, seed=42)
        off = adapt_v8_mprs_dch_parent_v4(
            parent,
            variant=FULL,
            relay_enabled=False,
        )
        on = adapt_v8_mprs_dch_parent_v4(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        self.assertIs(type(off), TPDNERV8MPRSDCHSCTransNet)
        self.assertIsInstance(on, TPDNERV8MPRSDCHV4SCTransNet)
        self.assertEqual(v4_relay_parameter_count(off), 0)
        self.assertEqual(
            v4_relay_parameter_count(on),
            PRODUCTION_V4_RELAY_PARAMETERS,
        )
        self.assertEqual(
            _parameter_count(on),
            PRODUCTION_V4_RELAY_ON_PARAMETERS,
        )
        self.assertFalse(tuple(on.tpd_ner.named_buffers()))

        manifest = on.architecture_manifest()
        self.assertEqual(manifest["relay_version"], V4_RELAY_VERSION)
        self.assertEqual(
            manifest["gate_dc_support_mode"],
            DEFAULT_DC_SUPPORT_MODE,
        )
        self.assertEqual(
            manifest["gate_dc_support_mode_options"],
            SUPPORTED_DC_SUPPORT_MODES,
        )
        self.assertEqual(
            manifest["gate_dc_support_formal_default"],
            DEFAULT_DC_SUPPORT_MODE,
        )
        self.assertTrue(manifest["target_protective_complement"])
        self.assertEqual(manifest["evidence_node_count"], 5)
        self.assertEqual(manifest["evidence_layout"], (3, 2))
        self.assertEqual(manifest["relay_stage_order"], (4, 3, 2))
        self.assertEqual(
            manifest["semantic_sources"],
            ("Keep", "Context", "Saliency"),
        )
        self.assertFalse(manifest["fourth_parallel_branch_added"])
        self.assertEqual(
            manifest["tail_z_thresholds"],
            {4: 1.5, 3: 2.0, 2: 2.5},
        )
        self.assertEqual(manifest["tail_support_parameters"], 0)
        self.assertEqual(manifest["tail_support_buffers"], 0)
        self.assertEqual(
            manifest["tail_support_gradient"],
            "stopped_or_constant",
        )
        self.assertEqual(
            manifest["state_compatible_with"],
            "tpd_ner_v8_mprs_dch_v3",
        )
        del parent, off, on
        gc.collect()


class V4TailSupportNumericsTests(unittest.TestCase):
    def _check_extreme_dtype(self, dtype: torch.dtype) -> None:
        maximum = torch.finfo(dtype).max
        for sign in (1.0, -1.0):
            flat = torch.full(
                (1, 8, 8, 8),
                sign * maximum,
                dtype=dtype,
            )
            support = relay_spatial_tail_support(
                flat,
                z_threshold=2.0,
            )
            self.assertEqual(support.dtype, dtype)
            self.assertTrue(torch.isfinite(support).all())
            self.assertTrue(torch.equal(support, torch.zeros_like(support)))

        spike = torch.zeros((1, 8, 8, 8), dtype=dtype)
        spike[:, :, 3, 5] = maximum
        support = relay_spatial_tail_support(spike, z_threshold=2.0)
        self.assertEqual(support.dtype, dtype)
        self.assertTrue(torch.isfinite(support).all())
        self.assertGreater(float(support[0, 0, 3, 5]), 0.0)
        self.assertEqual(int(torch.count_nonzero(support)), 1)
        self.assertGreaterEqual(float(support.min()), 0.0)
        self.assertLess(float(support.max()), 1.0)

    def test_flat_spike_locality_and_fp32_bfloat16_extreme_stability(self) -> None:
        flat = torch.ones(2, 8, 16, 16)
        flat_support = relay_spatial_tail_support(
            flat,
            z_threshold=2.0,
        )
        self.assertTrue(
            torch.equal(flat_support, torch.zeros_like(flat_support))
        )

        spike = _spike(
            shape=(1, 8, 16, 16),
            point=(7, 9),
        )
        spike_support = relay_spatial_tail_support(
            spike,
            z_threshold=2.0,
        )
        self.assertGreater(float(spike_support[0, 0, 7, 9]), 0.0)
        self.assertEqual(int(torch.count_nonzero(spike_support)), 1)
        self.assertTrue(torch.isfinite(spike_support).all())
        self.assertGreaterEqual(float(spike_support.min()), 0.0)
        self.assertLess(float(spike_support.max()), 1.0)

        self._check_extreme_dtype(torch.float32)
        self._check_extreme_dtype(torch.bfloat16)

    def test_float16_extreme_stability_when_cpu_operations_are_available(
        self,
    ) -> None:
        try:
            torch.nextafter(
                torch.tensor(1.0, dtype=torch.float16),
                torch.tensor(0.0, dtype=torch.float16),
            )
        except (RuntimeError, NotImplementedError) as exc:
            self.skipTest(f"CPU FP16 nextafter is unavailable: {exc}")
        self._check_extreme_dtype(torch.float16)

    def test_invalid_shapes_dtypes_thresholds_and_epsilon_are_rejected(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            relay_spatial_tail_support(
                torch.zeros(1, 8, 8),
                z_threshold=2.0,
            )
        with self.assertRaises(TypeError):
            relay_spatial_tail_support(
                torch.zeros(1, 8, 8, 8, dtype=torch.int64),
                z_threshold=2.0,
            )

        value = torch.zeros(1, 8, 8, 8)
        for threshold in (
            0.0,
            -1.0,
            math.nan,
            math.inf,
            -math.inf,
        ):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    relay_spatial_tail_support(
                        value,
                        z_threshold=threshold,
                    )
                thresholds: Mapping[int, float] = {
                    4: 1.5,
                    3: 2.0,
                    2: threshold,
                }
                with self.assertRaises(ValueError):
                    TailAwarePersistentDCOffsetEvidenceRelay(
                        base_channels=2,
                        tail_z_thresholds=thresholds,
                    )

        for epsilon in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    relay_spatial_tail_support(
                        value,
                        z_threshold=2.0,
                        eps=epsilon,
                    )
        with self.assertRaises(ValueError):
            TailAwarePersistentDCOffsetEvidenceRelay(
                base_channels=2,
                tail_z_thresholds={4: 1.5, 3: 2.0},
            )
        with self.assertRaises(ValueError):
            TailAwarePersistentDCOffsetEvidenceRelay(
                base_channels=2,
                tail_z_thresholds={4: 1.5, 3: 2.0, 2: 2.5001},
            )

    def test_stage3_and_stage2_three_persistence_modes_and_detach(self) -> None:
        relay = TailAwarePersistentDCOffsetEvidenceRelay(
            base_channels=2,
            dc_support_mode=TailDCSupportMode.DIRECT_TAIL,
        )
        for stage in (3, 2):
            with self.subTest(stage=stage):
                local_spike = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                    requires_grad=True,
                )
                parent_spike = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                    requires_grad=True,
                )
                local_flat = torch.zeros(
                    1,
                    8,
                    8,
                    8,
                    requires_grad=True,
                )
                parent_flat = torch.zeros(
                    1,
                    8,
                    8,
                    8,
                    requires_grad=True,
                )

                local_only = relay.dc_support(
                    stage,
                    local_spike,
                    _stage_sources(
                        stage,
                        parent_flat,
                        spatial_size=(8, 8),
                    ),
                    (8, 8),
                )
                parent_only = relay.dc_support(
                    stage,
                    local_flat,
                    _stage_sources(
                        stage,
                        parent_spike,
                        spatial_size=(8, 8),
                    ),
                    (8, 8),
                )
                persistent = relay.dc_support(
                    stage,
                    local_spike,
                    _stage_sources(
                        stage,
                        parent_spike,
                        spatial_size=(8, 8),
                    ),
                    (8, 8),
                )
                self.assertTrue(
                    torch.equal(local_only, torch.zeros_like(local_only))
                )
                self.assertTrue(
                    torch.equal(parent_only, torch.zeros_like(parent_only))
                )
                self.assertGreater(float(persistent[0, 0, 3, 5]), 0.0)
                self.assertEqual(int(torch.count_nonzero(persistent)), 1)
                self.assertFalse(persistent.requires_grad)

                misaligned_parent = _spike(
                    shape=(1, 8, 8, 8),
                    point=(6, 1),
                    requires_grad=True,
                )
                misaligned = relay.dc_support(
                    stage,
                    local_spike,
                    _stage_sources(
                        stage,
                        misaligned_parent,
                        spatial_size=(8, 8),
                    ),
                    (8, 8),
                )
                self.assertTrue(
                    torch.equal(misaligned, torch.zeros_like(misaligned))
                )

    def test_parent_tail_alignment_preserves_local_persistence(self) -> None:
        relay = TailAwarePersistentDCOffsetEvidenceRelay(
            base_channels=2,
            dc_support_mode="direct_tail",
        )
        local = _spike(
            shape=(1, 8, 8, 8),
            point=(4, 4),
        )
        deeper = _spike(
            shape=(1, 8, 4, 4),
            point=(2, 2),
        )
        for stage in (3, 2):
            with self.subTest(stage=stage):
                support = relay.dc_support(
                    stage,
                    local,
                    _stage_sources(
                        stage,
                        deeper,
                        spatial_size=(8, 8),
                    ),
                    (8, 8),
                )
                self.assertEqual(tuple(support.shape), (1, 1, 8, 8))
                self.assertTrue(torch.isfinite(support).all())
                self.assertGreater(float(support[0, 0, 4, 4]), 0.0)
                self.assertFalse(support.requires_grad)


class V4ControlledRelayBehaviorTests(unittest.TestCase):
    def test_all_three_dc_support_modes_have_exact_controlled_behavior(
        self,
    ) -> None:
        relays = {
            mode: TailAwarePersistentDCOffsetEvidenceRelay(
                base_channels=2,
                dc_support_mode=mode,
            )
            for mode in SUPPORTED_DC_SUPPORT_MODES
        }
        for mode, relay in relays.items():
            self.assertEqual(relay.dc_support_mode, mode)
            self.assertEqual(
                dict(relay.tail_z_thresholds),
                {4: 1.5, 3: 2.0, 2: 2.5},
            )
            with self.assertRaises(TypeError):
                relay.tail_z_thresholds[2] = 3.0
        reference_state = relays["legacy_global"].state_dict()
        reference_parameter_count = _parameter_count(
            relays["legacy_global"]
        )
        for mode, relay in relays.items():
            self.assertEqual(tuple(relay.state_dict()), tuple(reference_state))
            self.assertEqual(
                _parameter_count(relay),
                reference_parameter_count,
            )
            self.assertFalse(tuple(relay.named_buffers()))

        for stage in (3, 2):
            with self.subTest(stage=stage):
                local = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                )
                parent = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                )
                sources = _stage_sources(
                    stage,
                    parent,
                    spatial_size=(8, 8),
                )
                supports = {
                    mode: relay.dc_support(
                        stage,
                        local,
                        sources,
                        (8, 8),
                    )
                    for mode, relay in relays.items()
                }
                legacy = supports["legacy_global"]
                direct = supports["direct_tail"]
                complement = supports["complement_tail"]
                self.assertTrue(
                    torch.equal(legacy, torch.ones_like(legacy))
                )
                self.assertGreater(float(direct[0, 0, 3, 5]), 0.0)
                self.assertEqual(int(torch.count_nonzero(direct)), 1)
                self.assertTrue(
                    torch.equal(complement, torch.ones_like(direct) - direct)
                )
                self.assertFalse(direct.requires_grad)
                self.assertFalse(complement.requires_grad)

                flat_parent = torch.zeros_like(parent)
                flat_sources = _stage_sources(
                    stage,
                    flat_parent,
                    spatial_size=(8, 8),
                )
                direct_without_persistence = relays[
                    "direct_tail"
                ].dc_support(
                    stage,
                    local,
                    flat_sources,
                    (8, 8),
                )
                complement_without_persistence = relays[
                    "complement_tail"
                ].dc_support(
                    stage,
                    local,
                    flat_sources,
                    (8, 8),
                )
                self.assertTrue(
                    torch.equal(
                        direct_without_persistence,
                        torch.zeros_like(direct_without_persistence),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        complement_without_persistence,
                        torch.ones_like(complement_without_persistence),
                    )
                )

        stage4_sources = (
            torch.randn(1, 2, 8, 8),
            torch.randn(1, 4, 8, 8),
            torch.randn(1, 16, 8, 8),
        )
        stage4_value = torch.randn(1, 8, 8, 8)
        for mode, relay in relays.items():
            with self.subTest(stage=4, mode=mode):
                support = relay.dc_support(
                    4,
                    stage4_value,
                    stage4_sources,
                    (8, 8),
                )
                self.assertTrue(
                    torch.equal(support, torch.ones_like(support))
                )

        for invalid_mode in (
            "",
            "DIRECT_TAIL",
            "unknown",
            1,
            None,
        ):
            with self.subTest(invalid_mode=invalid_mode):
                expected_exception = (
                    TypeError
                    if not isinstance(invalid_mode, str)
                    else ValueError
                )
                with self.assertRaises(expected_exception):
                    TailAwarePersistentDCOffsetEvidenceRelay(
                        base_channels=2,
                        dc_support_mode=invalid_mode,
                    )

    def test_stage4_is_v3_exact_for_negative_zero_positive_and_extreme_dc(
        self,
    ) -> None:
        torch.manual_seed(4301)
        v3 = RMSBalancedCenteredDCOffsetEvidenceRelay(base_channels=2)
        torch.manual_seed(4301)
        v4 = TailAwarePersistentDCOffsetEvidenceRelay(base_channels=2)
        incompatible = v4.load_state_dict(v3.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)

        with torch.no_grad():
            weights = torch.linspace(-0.5, 0.5, steps=8).view(
                1,
                8,
                1,
                1,
            )
            v3.gates["4"].weight.copy_(weights)
            v4.gates["4"].weight.copy_(weights)
        sources = (
            torch.randn(2, 2, 8, 8),
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 16, 8, 8),
        )
        offsets = (
            -torch.finfo(torch.float32).max,
            -0.25,
            0.0,
            0.25,
            torch.finfo(torch.float32).max,
        )
        for offset in offsets:
            with self.subTest(offset=offset):
                with torch.no_grad():
                    v3.dc_offsets["4"].fill_(offset)
                    v4.dc_offsets["4"].fill_(offset)
                q3, mask3 = v3.forward_stage(4, sources, (8, 8))
                q4, mask4 = v4.forward_stage(4, sources, (8, 8))
                self.assertTrue(torch.equal(q3, q4))
                self.assertTrue(torch.equal(mask3, mask4))
                self.assertTrue(torch.isfinite(mask4).all())
                factor = 1.0 + mask4
                self.assertTrue(bool((factor > V2_SKIP_FACTOR_BOUNDS[0]).all()))
                self.assertTrue(bool((factor < V2_SKIP_FACTOR_BOUNDS[1]).all()))

    def test_stage3_and_stage2_negative_zero_positive_modes_are_controlled(
        self,
    ) -> None:
        for stage in (3, 2):
            with self.subTest(stage=stage):
                local = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                )
                parent = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                )
                sources = _stage_sources(
                    stage,
                    parent,
                    spatial_size=(8, 8),
                )
                torch.manual_seed(4302 + stage)
                v3 = RMSBalancedCenteredDCOffsetEvidenceRelay(
                    base_channels=2
                )
                torch.manual_seed(4302 + stage)
                v4 = TailAwarePersistentDCOffsetEvidenceRelay(
                    base_channels=2,
                    dc_support_mode="direct_tail",
                )
                v4.load_state_dict(v3.state_dict(), strict=True)
                v3.fusions[str(stage)] = _FixedFusion(local)
                v4.fusions[str(stage)] = _FixedFusion(local)
                with torch.no_grad():
                    v3.gates[str(stage)].weight.zero_()
                    v4.gates[str(stage)].weight.zero_()
                support = v4.dc_support(
                    stage,
                    local,
                    sources,
                    (8, 8),
                )
                inside = support > 0
                outside = torch.logical_not(inside)
                self.assertTrue(bool(inside.any()))
                self.assertTrue(bool(outside.any()))

                for offset in (-0.25, 0.0, 0.25):
                    with self.subTest(stage=stage, offset=offset):
                        with torch.no_grad():
                            v3.dc_offsets[str(stage)].fill_(offset)
                            v4.dc_offsets[str(stage)].fill_(offset)
                        q3, mask3 = v3.forward_stage(
                            stage,
                            sources,
                            (8, 8),
                        )
                        q4, mask4 = v4.forward_stage(
                            stage,
                            sources,
                            (8, 8),
                        )
                        self.assertTrue(torch.equal(q3, q4))
                        self.assertTrue(torch.isfinite(mask4).all())
                        self.assertTrue(
                            torch.equal(
                                mask4[outside],
                                torch.zeros_like(mask4[outside]),
                            )
                        )
                        if offset == 0.0:
                            self.assertTrue(
                                torch.equal(
                                    mask3,
                                    torch.zeros_like(mask3),
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    mask4,
                                    torch.zeros_like(mask4),
                                )
                            )
                        elif offset > 0.0:
                            self.assertTrue(bool((mask3 > 0).all()))
                            self.assertTrue(bool((mask4[inside] > 0).all()))
                            self.assertFalse(torch.equal(mask3, mask4))
                        else:
                            self.assertTrue(bool((mask3 < 0).all()))
                            self.assertTrue(bool((mask4[inside] < 0).all()))
                            self.assertFalse(torch.equal(mask3, mask4))

    def test_stage3_and_stage2_support_is_detached_and_dc_gradient_is_local(
        self,
    ) -> None:
        for stage in (3, 2):
            with self.subTest(stage=stage):
                local = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                    requires_grad=True,
                )
                parent = _spike(
                    shape=(1, 8, 8, 8),
                    point=(3, 5),
                    requires_grad=True,
                )
                sources = _stage_sources(
                    stage,
                    parent,
                    spatial_size=(8, 8),
                )
                relay = TailAwarePersistentDCOffsetEvidenceRelay(
                    base_channels=2,
                    dc_support_mode="direct_tail",
                )
                relay.fusions[str(stage)] = _FixedFusion(local)
                with torch.no_grad():
                    relay.gates[str(stage)].weight.zero_()
                    relay.dc_offsets[str(stage)].fill_(0.25)
                support = relay.dc_support(
                    stage,
                    local,
                    sources,
                    (8, 8),
                )
                self.assertFalse(support.requires_grad)
                _, mask = relay.forward_stage(stage, sources, (8, 8))
                inside = (support > 0).to(dtype=mask.dtype)
                outside = (support == 0).to(dtype=mask.dtype)

                inside_gradient = torch.autograd.grad(
                    mask,
                    relay.dc_offsets[str(stage)],
                    grad_outputs=inside,
                    retain_graph=True,
                )[0]
                outside_gradient = torch.autograd.grad(
                    mask,
                    relay.dc_offsets[str(stage)],
                    grad_outputs=outside,
                    retain_graph=True,
                )[0]
                all_gradients = torch.autograd.grad(
                    mask.sum(),
                    (
                        relay.dc_offsets[str(stage)],
                        local,
                        parent,
                    ),
                    allow_unused=True,
                )
                dc_gradient = all_gradients[0]
                local_gradient = all_gradients[1]
                parent_gradient = all_gradients[2]
                self.assertIsNotNone(dc_gradient)
                dc_gradient = cast(torch.Tensor, dc_gradient)
                self.assertTrue(torch.isfinite(dc_gradient).all())
                self.assertGreater(float(dc_gradient.abs().sum()), 0.0)
                self.assertTrue(torch.isfinite(inside_gradient).all())
                self.assertGreater(float(inside_gradient.abs().sum()), 0.0)
                self.assertTrue(
                    torch.equal(
                        outside_gradient,
                        torch.zeros_like(outside_gradient),
                    )
                )
                self.assertIsNotNone(local_gradient)
                local_gradient = cast(torch.Tensor, local_gradient)
                self.assertTrue(
                    torch.equal(
                        local_gradient,
                        torch.zeros_like(local_gradient),
                    ),
                    "detached support must not route DC gradients to local q",
                )
                self.assertIsNone(
                    parent_gradient,
                    "detached support must not route DC gradients to parent q",
                )


class V4CompleteModelStepZeroTests(unittest.TestCase):
    def test_six_outputs_match_relay_off_and_v3_and_common_gradients_pair(
        self,
    ) -> None:
        parent = _small_parent(seed=4401)
        off = adapt_v8_mprs_dch_parent_v4(
            parent,
            variant=FULL,
            relay_enabled=False,
        )
        v3 = adapt_v8_mprs_dch_parent_v3(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        v4 = adapt_v8_mprs_dch_parent_v4(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        off.eval()
        v3.eval()
        v4.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(4402)
        inputs = torch.randn(2, 1, 32, 32, generator=generator)
        with torch.no_grad():
            off_outputs = off(inputs)
            v3_outputs = v3(inputs)
            v4_outputs = v4(inputs)
        self.assertIsInstance(off_outputs, tuple)
        self.assertIsInstance(v3_outputs, tuple)
        self.assertIsInstance(v4_outputs, tuple)
        self.assertEqual(len(off_outputs), 6)
        self.assertEqual(len(v3_outputs), 6)
        self.assertEqual(len(v4_outputs), 6)
        for index, (off_output, v3_output, v4_output) in enumerate(
            zip(off_outputs, v3_outputs, v4_outputs)
        ):
            self.assertTrue(
                torch.equal(off_output, v3_output),
                f"relay-off/V3 step-zero output {index}",
            )
            self.assertTrue(
                torch.equal(v3_output, v4_output),
                f"V3/V4 step-zero output {index}",
            )

        targets = torch.rand(
            2,
            1,
            32,
            32,
            generator=generator,
        )
        v3.train()
        v4.train()
        v3.zero_grad(set_to_none=True)
        v4.zero_grad(set_to_none=True)
        v3_loss = _six_output_loss(v3(inputs), targets)
        v4_loss = _six_output_loss(v4(inputs), targets)
        self.assertTrue(torch.equal(v3_loss, v4_loss))
        v3_loss.backward()
        v4_loss.backward()

        v3_parameters = dict(v3.named_parameters())
        v4_parameters = dict(v4.named_parameters())
        self.assertEqual(tuple(v3_parameters), tuple(v4_parameters))
        for name, v3_parameter in v3_parameters.items():
            v4_parameter = v4_parameters[name]
            if name in FULL_OFFSET_KEYS:
                self.assertIsNotNone(v3_parameter.grad)
                self.assertIsNotNone(v4_parameter.grad)
                v3_gradient = cast(torch.Tensor, v3_parameter.grad)
                v4_gradient = cast(torch.Tensor, v4_parameter.grad)
                self.assertTrue(torch.isfinite(v3_gradient).all())
                self.assertTrue(torch.isfinite(v4_gradient).all())
                if name.endswith(".4"):
                    self.assertTrue(torch.equal(v3_gradient, v4_gradient))
                continue
            self.assertEqual(
                v3_parameter.grad is None,
                v4_parameter.grad is None,
                f"common gradient presence differs: {name}",
            )
            if v3_parameter.grad is not None and v4_parameter.grad is not None:
                self.assertTrue(
                    torch.equal(v3_parameter.grad, v4_parameter.grad),
                    f"common step-zero gradient differs: {name}",
                )
                self.assertTrue(torch.isfinite(v4_parameter.grad).all())

        del parent, off, v3, v4
        gc.collect()


if __name__ == "__main__":
    unittest.main()
