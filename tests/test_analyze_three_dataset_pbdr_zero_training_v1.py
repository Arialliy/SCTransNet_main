from __future__ import annotations

import unittest

import torch
from torch import nn

from analysis import analyze_three_dataset_pbdr_zero_training_v1 as subject


torch.set_num_threads(1)


class _ToyRelay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusions = nn.ModuleDict({"4": nn.Conv2d(8, 8, 1, bias=False)})


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tpd_ner = _ToyRelay()
        self.outconv = nn.Conv2d(8, 1, 1, bias=False)
        self.outc = nn.Conv2d(8, 1, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q4 = self.tpd_ner.fusions["4"](inputs)
        _ = self.outconv(q4)
        output = self.outc(q4)
        return torch.sigmoid(output)


class PBDRZeroTrainingAnalyzerTests(unittest.TestCase):
    def test_historical_drift_is_descriptive_not_same_forward_gate(self) -> None:
        reference = {
            "target_count": 10,
            "matched_target_count": 9,
            "tiny_target_count": 3,
            "matched_tiny_target_count": 2,
            "predicted_object_count": 11,
            "unmatched_predicted_object_count": 2,
            "unmatched_predicted_pixels": 8,
            "valid_pixel_count": 1000,
            "miou": 0.8,
            "niou": 0.79,
            "test_loss": 0.01,
        }
        observed = dict(reference)
        observed.update(
            {
                "miou": 0.8002,
                "background_false_positive_pixels": 7,
            }
        )
        audit = subject.historical_reference_drift_audit(
            observed,
            reference,
            expected_background_false_positive_pixels=8,
        )
        self.assertFalse(audit["same_forward_authorization_gate"])
        self.assertFalse(audit["historical_exact"])
        self.assertTrue(audit["historical_count_fields_exact"])
        self.assertEqual(audit["background_false_positive_pixel_delta"], -1)

    def test_gate_endpoints_and_identity(self) -> None:
        z_out = torch.tensor([[[[-2.0, 2.0], [3.0, -3.0]]]])
        z_d0 = torch.tensor([[[[1.0, 1.0], [-1.0, -2.0]]]])
        protection = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

        identity = subject.route_with_gate(z_out, z_d0, protection, 0)
        self.assertIs(identity, z_out)
        oracle = subject.route_with_gate(z_out, z_d0, protection, 8)
        expected = torch.where(
            protection.bool(),
            torch.maximum(z_out, z_d0),
            torch.minimum(z_out, z_d0),
        )
        self.assertTrue(torch.equal(oracle, expected))

        half = subject.route_with_gate(z_out, z_d0, protection, 4)
        self.assertTrue(torch.equal(half, z_out + 0.5 * (expected - z_out)))
        with self.assertRaisesRegex(ValueError, "frozen grid"):
            subject.route_with_gate(z_out, z_d0, protection, 3)

    def test_protection_is_binary_detached_and_resized(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026080601)
        q4 = torch.randn(2, 8, 5, 7, generator=generator, requires_grad=True)
        protection = subject.build_protection(q4, (11, 13))
        self.assertEqual(tuple(protection.shape), (2, 1, 11, 13))
        self.assertFalse(protection.requires_grad)
        self.assertLessEqual(
            set(float(value) for value in torch.unique(protection)),
            {0.0, 1.0},
        )

    def test_hook_capture_is_once_per_batch_and_restored(self) -> None:
        model = _ToyModel().eval()
        modules = (
            model.tpd_ner.fusions["4"],
            model.outconv,
            model.outc,
        )
        before = [tuple(module._forward_hooks) for module in modules]
        inputs = torch.randn(1, 8, 4, 4)
        with subject.RawQ4D0OutHookCapture(model) as capture:
            capture.begin_batch()
            model(inputs)
            q4, z_out, z_d0 = capture.finish_batch()
            self.assertEqual(tuple(q4.shape), (1, 8, 4, 4))
            self.assertEqual(tuple(z_out.shape), (1, 1, 4, 4))
            self.assertEqual(tuple(z_d0.shape), (1, 1, 4, 4))
            self.assertEqual(capture.total_counts, {"q4": 1, "out": 1, "d0": 1})
        self.assertTrue(capture.temporary_hooks_restored)
        self.assertEqual(
            [tuple(module._forward_hooks) for module in modules],
            before,
        )

    def test_hook_cleanup_on_exception(self) -> None:
        model = _ToyModel().eval()
        modules = (
            model.tpd_ner.fusions["4"],
            model.outconv,
            model.outc,
        )
        before = [tuple(module._forward_hooks) for module in modules]
        capture = subject.RawQ4D0OutHookCapture(model)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with capture:
                raise RuntimeError("synthetic")
        self.assertTrue(capture.temporary_hooks_restored)
        self.assertEqual(
            [tuple(module._forward_hooks) for module in modules],
            before,
        )


if __name__ == "__main__":
    unittest.main()
