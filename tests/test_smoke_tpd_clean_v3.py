from __future__ import annotations

import unittest

import torch

from experiments.smoke_tpd_clean_v3 import (
    UINT32_MAX,
    _next_seed,
    _paired_inputs,
    _resolve_device,
    _validate_outputs,
    run_smoke,
)


class SmokeTPDCleanV3Tests(unittest.TestCase):
    def test_paired_inputs_are_reproducible(self) -> None:
        inputs_a, targets_a = _paired_inputs(2, 32, 42)
        inputs_b, targets_b = _paired_inputs(2, 32, 42)
        inputs_c, targets_c = _paired_inputs(2, 32, 43)
        self.assertTrue(torch.equal(inputs_a, inputs_b))
        self.assertTrue(torch.equal(targets_a, targets_b))
        self.assertFalse(torch.equal(inputs_a, inputs_c))
        self.assertFalse(torch.equal(targets_a, targets_c))

    def test_validate_outputs_enforces_six_full_resolution_outputs(self) -> None:
        valid = tuple(torch.rand(2, 1, 32, 32) for _ in range(6))
        self.assertEqual(
            _validate_outputs(valid, batch_size=2, patch_size=32), valid
        )
        with self.assertRaisesRegex(RuntimeError, "expected six"):
            _validate_outputs(valid[:5], batch_size=2, patch_size=32)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            _validate_outputs(
                valid[:5] + (torch.rand(2, 1, 16, 16),),
                batch_size=2,
                patch_size=32,
            )
        invalid = list(valid)
        invalid[3] = invalid[3].clone()
        invalid[3][0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(FloatingPointError, "not finite"):
            _validate_outputs(
                tuple(invalid), batch_size=2, patch_size=32
            )

    def test_cpu_device_contract(self) -> None:
        device, name = _resolve_device("cpu", None)
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(name, "cpu")
        with self.assertRaisesRegex(ValueError, "only for CUDA"):
            _resolve_device("cpu", "NVIDIA GeForce RTX 5090")

    def test_rebuild_seed_wraps_in_uint32(self) -> None:
        self.assertEqual(_next_seed(42), 43)
        self.assertEqual(_next_seed(UINT32_MAX), 0)

    def test_programmatic_validation_fails_before_model_construction(self) -> None:
        base = {
            "variant": "all",
            "device_text": "cpu",
            "batch_size": 2,
            "patch_size": 32,
            "steps": 2,
            "seed": 42,
            "learning_rate": 1e-3,
        }
        cases = (
            ({"variant": "unknown"}, "unsupported variant"),
            ({"batch_size": 1}, "batch_size"),
            ({"patch_size": 48}, "patch_size"),
            ({"steps": 1}, "steps"),
            ({"seed": -1}, "seed"),
            ({"seed": UINT32_MAX + 1}, "seed"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"learning_rate": float("nan")}, "learning_rate"),
        )
        for override, pattern in cases:
            arguments = dict(base)
            arguments.update(override)
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, pattern):
                    run_smoke(**arguments)


if __name__ == "__main__":
    unittest.main()
