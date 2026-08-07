from __future__ import annotations

import unittest

import numpy as np

from experiments.pbdr_v4_metric_core import (
    PBDRV4MetricAccumulator,
    PBDRV4MetricError,
)
from experiments.train_tpd_pilot import ValidationMetrics


class PBDRV4MetricCoreTests(unittest.TestCase):
    def test_matches_current_formal_metric_counts(self) -> None:
        rng = np.random.default_rng(2026080702)
        current = ValidationMetrics(threshold=0.5, match_radius=3.0, tiny_area=9)
        v4 = PBDRV4MetricAccumulator()
        for index in range(25):
            probability = rng.random((19, 23), dtype=np.float32)
            target = (rng.random((19, 23)) > 0.94).astype(np.float32)
            loss = float(rng.random())
            current.update(probability, target, loss)
            v4.update(
                probability=probability,
                target=target,
                loss=loss,
                identifier=f"sample-{index:03d}",
            )
        old = current.compute()
        new = v4.compute()
        for name in (
            "target_count",
            "matched_target_count",
            "tiny_target_count",
            "matched_tiny_target_count",
            "predicted_object_count",
            "unmatched_predicted_object_count",
            "valid_pixel_count",
        ):
            self.assertEqual(new[name], old[name])
        self.assertEqual(
            new["unmatched_component_pixels"],
            old["unmatched_predicted_pixel_count"]
            if "unmatched_predicted_pixel_count" in old
            else current.unmatched_predicted_pixels,
        )
        self.assertAlmostEqual(float(new["miou"]), float(old["miou"]))
        self.assertAlmostEqual(float(new["niou"]), float(old["niou"]))
        self.assertAlmostEqual(float(new["pd"]), float(old["pd"]))
        self.assertAlmostEqual(float(new["fa"]), float(old["fa"]))

    def test_probability_threshold_is_strictly_greater_than_half(self) -> None:
        target = np.asarray([[1.0, 0.0]], dtype=np.float32)
        probability = np.asarray([[0.5, np.nextafter(np.float32(0.5), np.float32(1.0))]], dtype=np.float32)
        metric = PBDRV4MetricAccumulator()
        metric.update(
            probability=probability,
            target=target,
            loss=0.1,
            identifier="boundary",
        )
        result = metric.compute()
        self.assertEqual(result["intersection_pixels"], 0)
        self.assertEqual(result["union_pixels"], 2)
        self.assertEqual(result["false_positive_pixels"], 1)
        self.assertEqual(result["false_negative_pixels"], 1)

    def test_order_and_targets_are_semantically_hashed(self) -> None:
        target = np.asarray([[1.0, 0.0]], dtype=np.float32)
        probability = target.copy()
        first = PBDRV4MetricAccumulator()
        second = PBDRV4MetricAccumulator()
        for identifier in ("a", "b"):
            first.update(probability=probability, target=target, loss=0.0, identifier=identifier)
        for identifier in ("b", "a"):
            second.update(probability=probability, target=target, loss=0.0, identifier=identifier)
        left, right = first.compute(), second.compute()
        self.assertNotEqual(left["sample_id_order_sha256"], right["sample_id_order_sha256"])
        self.assertNotEqual(left["target_sha256"], right["target_sha256"])

    def test_duplicate_or_empty_inputs_fail_closed(self) -> None:
        metric = PBDRV4MetricAccumulator()
        with self.assertRaisesRegex(PBDRV4MetricError, "empty"):
            metric.compute()
        value = np.ones((2, 2), dtype=np.float32)
        metric.update(probability=value, target=value, loss=0.0, identifier="same")
        with self.assertRaisesRegex(PBDRV4MetricError, "duplicate"):
            metric.update(probability=value, target=value, loss=0.0, identifier="same")


if __name__ == "__main__":
    unittest.main()
