from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from analysis import analyze_nuaa_pbdr_v2_failure_attribution_v1 as subject
from model.tpd_persistent_evidence_residual_router_v2 import (
    PersistentEvidenceResidualRouterV2,
)


torch.set_num_threads(1)


class _ToyRelay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusions = nn.ModuleDict({"4": nn.Conv2d(8, 8, 1, bias=False)})


class _ToyCaptureModel(nn.Module):
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


class NUAAPBDRV2FailureAttributionTests(unittest.TestCase):
    def _router_inputs(self):
        router = PersistentEvidenceResidualRouterV2()
        with torch.no_grad():
            router.confidence_projection.weight.copy_(
                torch.tensor([[[[0.15]], [[-0.10]], [[0.05]], [[0.20]],
                               [[-0.05]], [[0.12]], [[-0.08]], [[0.03]]]])
            )
            router.confidence_projection.bias.fill_(-0.2)
            router.direct_residual_projection.weight.copy_(
                torch.tensor([[[[0.08]], [[0.02]], [[-0.04]], [[0.10]],
                               [[0.01]], [[-0.03]], [[0.05]], [[-0.02]]]])
            )
            router.rescue_strength_raw.fill_(0.4)
            router.suppression_strength_raw.fill_(-0.3)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026080602)
        q4 = torch.randn(1, 8, 2, 3, generator=generator)
        z_out = torch.tensor(
            [[[[0.2, -0.4, 0.1], [-0.3, 0.5, -0.1]]]],
            dtype=torch.float32,
        )
        z_d0 = torch.tensor(
            [[[[0.7, -0.8, -0.2], [0.2, 0.1, 0.4]]]],
            dtype=torch.float32,
        )
        return router, z_out, z_d0, q4

    def test_module_state_hash_supports_zero_dimensional_integer_buffers(self) -> None:
        module = nn.BatchNorm2d(1)
        self.assertEqual(module.num_batches_tracked.ndim, 0)
        first = subject.module_state_sha256(module)
        self.assertEqual(len(first), 64)
        module.num_batches_tracked.add_(1)
        self.assertNotEqual(subject.module_state_sha256(module), first)

    def test_tf32_off_checkpoint_audit_bounds_soft_drift_but_not_counts(self) -> None:
        checkpoint = {
            "test_loss": 0.01,
            "miou": 0.80,
            "niou": 0.79,
            "pixel_precision": 0.90,
            "pixel_recall": 0.88,
            "pixel_f1": 0.89,
            "pd": 0.9,
            "tiny_pd": 0.8,
            "fa": 1.0e-5,
            "false_objects_per_image": 0.1,
            "target_count": 10,
            "matched_target_count": 9,
            "tiny_target_count": 5,
            "matched_tiny_target_count": 4,
            "predicted_object_count": 11,
            "unmatched_predicted_object_count": 2,
            "valid_pixel_count": 1000,
        }
        observed = dict(checkpoint)
        observed["miou"] += 5.0e-4
        observed["fa"] += 2.0 / observed["valid_pixel_count"]
        audit = subject.checkpoint_metric_audit_under_tf32_off(
            {"test_metrics": checkpoint}, observed
        )
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(
            audit["comparisons"]["fa"]["equivalent_absolute_pixel_delta"],
            2.0,
        )
        changed_count = dict(observed, matched_target_count=8)
        with self.assertRaisesRegex(ValueError, "count differs"):
            subject.checkpoint_metric_audit_under_tf32_off(
                {"test_metrics": checkpoint}, changed_count
            )
        excessive_drift = dict(observed, miou=0.81)
        with self.assertRaisesRegex(ValueError, "beyond tolerance"):
            subject.checkpoint_metric_audit_under_tf32_off(
                {"test_metrics": checkpoint}, excessive_drift
            )
        excessive_fa = dict(
            observed,
            fa=checkpoint["fa"]
            + (subject.STRICT_MATH_MAX_FA_PIXEL_DRIFT + 1) / 1000,
        )
        with self.assertRaisesRegex(ValueError, "beyond tolerance"):
            subject.checkpoint_metric_audit_under_tf32_off(
                {"test_metrics": checkpoint}, excessive_fa
            )

    def test_default_threshold_grid_is_exact_and_inclusive(self) -> None:
        grid = subject.build_threshold_grid()
        self.assertEqual(len(grid), 121)
        self.assertEqual(grid[0], 0.2)
        self.assertEqual(grid[-1], 0.8)
        self.assertIn(0.5, grid)
        self.assertNotIn(1.0, grid)
        with self.assertRaisesRegex(ValueError, "does not close"):
            subject.build_threshold_grid(0.2, 0.8, 0.07)
        with self.assertRaisesRegex(ValueError, "interval"):
            subject.build_threshold_grid(0.8, 0.2, 0.01)

    def test_all_eight_pbdr_ablation_formulas_are_exact(self) -> None:
        router, z_out, z_d0, q4 = self._router_inputs()
        diagnostics = router.forward_with_diagnostics(z_out, z_d0, q4)
        rescue = diagnostics.rescue_strength.reshape(1, 1, 1, 1)
        suppression = diagnostics.suppression_strength.reshape(1, 1, 1, 1)
        expected = {
            "identity": z_out,
            "full": diagnostics.routed_logits,
            "direct_only": z_out + diagnostics.direct_residual,
            "disagreement_only": (
                z_out
                + rescue * diagnostics.target_rescue
                - suppression * diagnostics.background_suppression
            ),
            "rescue_only": z_out + rescue * diagnostics.target_rescue,
            "suppression_only": (
                z_out - suppression * diagnostics.background_suppression
            ),
            "nonnegative_strengths": (
                z_out
                + diagnostics.direct_residual
                + rescue.clamp_min(0.0) * diagnostics.target_rescue
                - suppression.clamp_min(0.0)
                * diagnostics.background_suppression
            ),
            "auxiliary_d0": z_d0,
        }
        state_before = subject.module_state_sha256(router)
        for mode in subject.PBDR_ABLATION_MODES:
            actual = subject.ablation_logits_from_diagnostics(
                z_out, z_d0, diagnostics, mode
            )
            self.assertTrue(torch.equal(actual, expected[mode]), mode)
            wrapper = subject.PBDRV2AblationWrapper(router, mode).eval()
            wrapped = wrapper(z_out, z_d0, q4)
            self.assertTrue(torch.equal(wrapped, expected[mode]), mode)
            self.assertFalse(hasattr(wrapper, "last_diagnostics"))
        self.assertEqual(subject.module_state_sha256(router), state_before)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            subject.ablation_logits_from_diagnostics(
                z_out, z_d0, diagnostics, "unknown"  # type: ignore[arg-type]
            )

    def test_nonnegative_mode_neutralizes_negative_strength_semantics(self) -> None:
        router, z_out, z_d0, q4 = self._router_inputs()
        diagnostics = router.forward_with_diagnostics(z_out, z_d0, q4)
        self.assertLess(float(diagnostics.suppression_strength.detach()), 0.0)
        nonnegative = subject.ablation_logits_from_diagnostics(
            z_out, z_d0, diagnostics, "nonnegative_strengths"
        )
        expected = (
            z_out
            + diagnostics.direct_residual
            + diagnostics.rescue_strength.reshape(1, 1, 1, 1)
            * diagnostics.target_rescue
        )
        self.assertTrue(torch.equal(nonnegative, expected))

    def test_capture_is_once_per_batch_and_always_restores_hooks(self) -> None:
        model = _ToyCaptureModel().eval()
        modules = (
            model.tpd_ner.fusions["4"],
            model.outc,
            model.outconv,
        )
        hooks_before = [tuple(module._forward_hooks) for module in modules]
        with subject.RawQ4D0OutCapture(model) as capture:
            capture.begin_batch()
            model(torch.randn(1, 8, 4, 5))
            q4, z_out, z_d0 = capture.finish_batch()
            self.assertEqual(tuple(q4.shape), (1, 8, 4, 5))
            self.assertEqual(tuple(z_out.shape), (1, 1, 4, 5))
            self.assertEqual(tuple(z_d0.shape), (1, 1, 4, 5))
            self.assertEqual(
                capture.total_counts,
                {"q4": 1, "z_out": 1, "z_d0": 1},
            )
        self.assertTrue(capture.temporary_hooks_restored)
        self.assertEqual(
            [tuple(module._forward_hooks) for module in modules], hooks_before
        )

        failed = subject.RawQ4D0OutCapture(model)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with failed:
                raise RuntimeError("synthetic")
        self.assertTrue(failed.temporary_hooks_restored)
        self.assertEqual(
            [tuple(module._forward_hooks) for module in modules], hooks_before
        )

    def test_q4_and_router_diagnostics_expose_required_signals(self) -> None:
        router, _z_out, _z_d0, q4 = self._router_inputs()
        q4_stats = subject.q4_pre_normalization_statistics(q4)
        self.assertEqual([entry["channel"] for entry in q4_stats], list(range(8)))
        self.assertTrue(all(entry["rms"] >= 0.0 for entry in q4_stats))
        audit = subject.router_parameter_audit(router)
        self.assertEqual(audit["parameter_count"], 19)
        self.assertFalse(audit["rescue_semantics_reversed"])
        self.assertTrue(audit["suppression_semantics_reversed"])
        self.assertEqual(
            sum(
                len(record["values"])
                for record in audit["parameters"].values()
            ),
            19,
        )

    def test_background_crossing_and_new_component_attribution(self) -> None:
        current = np.full((8, 8), 0.1, dtype=np.float32)
        candidate = current.copy()
        candidate[5:7, 5:7] = 0.9
        target = np.zeros((8, 8), dtype=np.float32)
        counts = subject.binary_transition_counts(current, candidate, target)
        self.assertEqual(counts["background_off_to_on_pixels"], 4)
        self.assertEqual(counts["disjoint_new_candidate_components"], 1)

        direct = np.zeros((8, 8), dtype=np.float32)
        rescue = np.zeros((8, 8), dtype=np.float32)
        suppression = np.zeros((8, 8), dtype=np.float32)
        direct[5:7, 5:7] = 0.4
        rescue[5:7, 5:7] = 0.1
        records = subject.attribute_new_unmatched_components(
            identifier="toy",
            current_probability=current,
            full_probability=candidate,
            target=target,
            direct_contribution=direct,
            rescue_contribution=rescue,
            suppression_contribution=suppression,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["area"], 4)
        self.assertTrue(records[0]["disjoint_from_current"])
        self.assertEqual(records[0]["dominant_branch_by_absolute_mean"], "direct")

    def test_working_points_separate_exact_pd_from_fa_feasibility(self) -> None:
        current = {
            "threshold": 0.5,
            "target_count": 10,
            "matched_target_count": 9,
            "miou": 0.80,
            "niou": 0.79,
            "fa": 2.0e-5,
        }
        points = [
            {
                "threshold": 0.4,
                "target_count": 10,
                "matched_target_count": 9,
                "miou": 0.79,
                "niou": 0.78,
                "fa": 1.0e-5,
            },
            {
                "threshold": 0.5,
                "target_count": 10,
                "matched_target_count": 9,
                "miou": 0.81,
                "niou": 0.80,
                "fa": 1.5e-5,
            },
            {
                "threshold": 0.6,
                "target_count": 10,
                "matched_target_count": 8,
                "miou": 0.82,
                "niou": 0.81,
                "fa": 0.5e-5,
            },
        ]
        selected = subject.matched_working_points(points, current)
        self.assertEqual(
            selected["same_Pd_minimum_Fa"]["threshold"], 0.4
        )
        self.assertEqual(
            selected["same_Pd_maximum_mIoU"]["threshold"], 0.5
        )
        self.assertEqual(selected["same_Fa_maximum_Pd"]["threshold"], 0.4)
        self.assertTrue(selected["threshold_grid_can_fully_restore_A0"])

    def test_frozen_metric_core_supplies_fixed_sweep_and_empty_endpoint(self) -> None:
        probabilities = [
            np.array([[0.1, 0.9], [0.2, 0.1]], dtype=np.float32),
            np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
        ]
        targets = [
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ]
        output = subject.evaluate_attribution_mode(
            probabilities,
            targets,
            [0.1, 0.1],
            threshold_grid=(0.2, 0.5, 0.8),
            current_fixed=None,
        )
        self.assertEqual(output["fixed_threshold_0_5"]["threshold"], 0.5)
        self.assertEqual(output["threshold_sweep"]["registered_point_count"], 3)
        endpoint = output["threshold_sweep"]["threshold_1_0_empty_control"]
        self.assertEqual(endpoint["threshold"], 1.0)
        self.assertEqual(endpoint["predicted_object_count"], 0)
        self.assertEqual(endpoint["pd"], 0.0)
        self.assertEqual(endpoint["fa"], 0.0)

    def test_inference_contract_turns_tf32_off(self) -> None:
        contract = subject.configure_inference_math()
        self.assertEqual(
            contract,
            {
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "deterministic_algorithms": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
