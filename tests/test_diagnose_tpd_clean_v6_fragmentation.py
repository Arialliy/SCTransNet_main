from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis import diagnose_tpd_clean_v6_fragmentation as subject
from model.tpd_clean_v6 import TPDCleanV6Block


class ComponentTaxonomyTests(unittest.TestCase):
    def test_taxonomy_is_mutually_exclusive_and_pixel_complete(self) -> None:
        target = np.zeros((20, 20), dtype=np.float32)
        # One connected, thin GT whose centroid is (6, 6).  This geometry
        # leaves room for a non-overlapping component inside the centroid
        # match radius without connecting it to either in-GT fragment.
        target[6, 4:9] = 1.0
        probability = np.zeros_like(target)

        # Two disjoint components overlap the same GT: one primary and one
        # unmatched in-GT fragment.
        probability[6, 6] = 1.0
        probability[6, 4] = 1.0
        # Centroid is strictly within radius 3 of the GT centroid, no overlap.
        probability[4, 6] = 1.0
        # Intersects the 3-pixel dilation, but its centroid is not within 3 of
        # the GT centroid.
        probability[9, 6] = 1.0
        # Unrelated background object.
        probability[17, 17] = 1.0

        result = subject.component_diagnostics(
            probability,
            target,
            threshold=0.5,
            match_radius=3.0,
            dilation_radius=3,
        )

        counts = result["unmatched_component_count_by_class"]
        self.assertEqual(counts["in_gt_fragment"], 1)
        self.assertEqual(counts["near_gt_duplicate"], 1)
        self.assertEqual(counts["attached_or_near_gt"], 1)
        self.assertEqual(counts["background_false_object"], 1)
        self.assertEqual(sum(counts.values()), result["unmatched_component_count"])
        self.assertEqual(
            sum(result["unmatched_component_pixels_by_class"].values()),
            result["unmatched_pixels_total"],
        )
        self.assertAlmostEqual(
            result["fragment_fa_fraction"]
            + result["background_fa_fraction"],
            1.0,
        )

    def test_gt_fragmentation_summary_uses_intersection_areas(self) -> None:
        target = np.zeros((12, 12), dtype=np.float32)
        target[2:8, 2:8] = 1.0
        probability = np.zeros_like(target)
        probability[2:4, 2:5] = 1.0  # six in-GT pixels
        probability[6:8, 6:8] = 1.0  # four in-GT pixels

        result = subject.aggregate_component_diagnostics(
            [probability],
            [target],
            ["synthetic"],
            threshold=0.5,
            match_radius=3.0,
            dilation_radius=3,
        )

        self.assertEqual(result["fragmented_gt_count"], 1)
        self.assertEqual(result["split_target_count"], 1)
        self.assertEqual(result["extra_fragments"], 1)
        self.assertEqual(result["fragment_excess_total"], 1)
        record = result["per_gt"][0]
        self.assertEqual(record["overlapping_prediction_components"], 2)
        self.assertEqual(record["all_in_gt_prediction_area"], 10)
        self.assertEqual(record["largest_in_gt_fragment_area"], 6)
        self.assertAlmostEqual(record["largest_fragment_fraction"], 0.6)


class CounterfactualRestorationTests(unittest.TestCase):
    def test_context_off_restores_flag_and_state_after_exception(self) -> None:
        block = TPDCleanV6Block(
            channels=2,
            activate=False,
            use_context_headroom=True,
        )
        before = subject.model_state_sha256(block)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with subject.temporary_counterfactual(
                _SevenBlockContainer(block),
                "same_weights_context_off",
            ):
                self.assertFalse(block.use_context_headroom)
                raise RuntimeError("sentinel")
        self.assertTrue(block.use_context_headroom)
        self.assertEqual(subject.model_state_sha256(block), before)

    def test_residual_off_hook_is_keep_only_and_is_removed(self) -> None:
        torch.manual_seed(9)
        block = TPDCleanV6Block(
            channels=2,
            activate=False,
            use_context_headroom=True,
        )
        block.saliency_scale.data.fill_(0.8)
        model = _SevenBlockContainer(block)
        sample = torch.randn(1, 2, 8, 8)
        original = block(sample)
        expected_keep = block.branches(sample)[0]
        with subject.temporary_counterfactual(
            model,
            "same_weights_residual_off",
        ):
            observed = block(sample)
            self.assertTrue(torch.equal(observed, expected_keep))
        restored = block(sample)
        self.assertTrue(torch.equal(restored, original))

    def test_residual_off_preserves_intermediate_relu_contract(self) -> None:
        torch.manual_seed(11)
        block = TPDCleanV6Block(
            channels=2,
            activate=True,
            use_context_headroom=True,
        )
        block.saliency_scale.data.fill_(0.8)
        model = _SevenBlockContainer(block)
        sample = torch.randn(1, 2, 8, 8)
        keep = block.branches(sample)[0]
        expected = torch.relu(keep)
        with subject.temporary_counterfactual(
            model,
            "same_weights_residual_off",
        ):
            observed = block(sample)
        self.assertTrue(torch.equal(observed, expected))
        self.assertTrue(bool((observed >= 0).all()))

    def test_static_checkpoint_diagnostics_cover_all_blocks(self) -> None:
        torch.manual_seed(12)
        model = _SevenBlockContainer(
            TPDCleanV6Block(
                channels=2,
                activate=True,
                use_context_headroom=True,
            )
        )
        model.blocks[0].saliency_scale.data.copy_(
            torch.tensor([-1.0, 0.0])
        )
        result = subject.static_checkpoint_diagnostics(model)
        self.assertEqual(result["block_count"], 7)
        self.assertEqual(len(result["blocks"]), 7)
        first = result["blocks"][0]
        self.assertAlmostEqual(
            first["saliency_scale_effective_abs_tanh"]["max"],
            float(torch.tanh(torch.tensor(1.0))),
        )
        for row in result["blocks"]:
            cancellation = row["phase_sum_cancellation"]
            self.assertGreaterEqual(cancellation["rho_l1"], 0.0)
            self.assertLessEqual(cancellation["rho_l1"], 1.0 + 1e-7)
            self.assertGreaterEqual(cancellation["rho_l2"], 0.0)
            self.assertLessEqual(cancellation["rho_l2"], 1.0 + 1e-7)

    def test_static_rho_l2_uses_registered_sqrt4_weight_norm(self) -> None:
        model = _SevenBlockContainer(
            TPDCleanV6Block(
                channels=1,
                activate=True,
                use_context_headroom=True,
            ),
            channels=1,
        )
        with torch.no_grad():
            for block in model.blocks:
                block.phase_compress.weight.copy_(
                    torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])
                )
        result = subject.static_checkpoint_diagnostics(model)
        first = result["blocks"][0]["phase_sum_cancellation"]
        expected_numerator = 10.0
        expected_denominator = 2.0 * float(
            torch.linalg.vector_norm(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        ) + np.finfo(np.float64).eps
        self.assertAlmostEqual(first["l2_numerator"], expected_numerator)
        self.assertAlmostEqual(
            first["l2_denominator_sqrt4_weight_norm"],
            expected_denominator,
        )
        self.assertAlmostEqual(
            first["rho_l2"],
            expected_numerator / expected_denominator,
        )
        # Seven identical blocks: concatenating both numerator and weight
        # tensors preserves the same ratio.
        self.assertAlmostEqual(
            result["aggregate"]["phase_sum_cancellation"]["rho_l2"],
            expected_numerator / expected_denominator,
        )


class DecisionScreenTests(unittest.TestCase):
    def test_formal_inference_determinism_contract(self) -> None:
        contract = subject.configure_formal_inference_determinism("cpu")
        self.assertFalse(contract["cudnn_benchmark"])
        self.assertTrue(contract["cudnn_deterministic"])
        self.assertFalse(contract["cuda_matmul_allow_tf32"])
        self.assertFalse(contract["cudnn_allow_tf32"])
        self.assertTrue(contract["deterministic_algorithms"])
        self.assertEqual(contract["float32_matmul_precision"], "highest")

    def test_joint_task_and_topology_condition(self) -> None:
        delta = {
            "pd": 0.0,
            "fa": -1e-6,
            "miou": -0.01,
            "fragmented_gt_count": -1,
            "extra_fragments": 0,
        }
        self.assertTrue(subject.point_supports_context(delta))
        self.assertFalse(
            subject.point_supports_context({**delta, "pd": -0.01})
        )
        self.assertFalse(
            subject.point_supports_context(
                {
                    **delta,
                    "fa": 0.0,
                    "miou": 0.0,
                }
            )
        )
        self.assertFalse(
            subject.point_supports_context(
                {
                    **delta,
                    "fragmented_gt_count": 0,
                    "extra_fragments": 0,
                }
            )
        )

    def test_cli_requires_explicit_physical_gpu_for_cuda(self) -> None:
        with self.assertRaises(SystemExit):
            subject.parse_args(["--run", "--device", "cuda:0"])
        parsed = subject.parse_args(
            [
                "--run",
                "--device",
                "cuda:0",
                "--physical-gpu",
                "2",
                "--max-validation-images",
                "1",
            ]
        )
        self.assertEqual(parsed.physical_gpu, "2")


class _SevenBlockContainer(torch.nn.Module):
    """Expose seven references while tests exercise one representative block."""

    def __init__(
        self,
        representative: TPDCleanV6Block,
        channels: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                representative,
                *[
                    TPDCleanV6Block(
                        channels=channels,
                        activate=False,
                        use_context_headroom=True,
                    )
                    for _ in range(6)
                ],
            ]
        )


if __name__ == "__main__":
    unittest.main()
