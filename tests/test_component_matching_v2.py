from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.component_matching_v2 import match_components_v2


class ComponentMatchingV2Tests(unittest.TestCase):
    def test_duplicate_area_unmatched_component_is_not_dropped(self) -> None:
        target = np.zeros((16, 16), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[2, 2] = True
        prediction[2, 2] = True
        prediction[12, 12] = True

        result = match_components_v2(
            prediction_mask=prediction,
            target_mask=target,
        )

        self.assertEqual(tuple(record.area for record in result.predictions), (1, 1))
        self.assertEqual(result.matched_prediction_ids, (1,))
        self.assertEqual(result.unmatched_prediction_ids, (2,))
        self.assertEqual(result.unmatched_prediction_pixels, 1)

    def test_assignment_maximizes_cardinality_before_distance(self) -> None:
        # Target A can use both predictions. Target B can use only prediction 1.
        # A scan-order greedy matcher consumes prediction 1 for A and returns
        # one match; the canonical assignment returns both.
        target = np.zeros((16, 16), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[5, 5] = True
        target[5, 9] = True
        prediction[5, 7] = True
        prediction[7, 5] = True

        result = match_components_v2(
            prediction_mask=prediction,
            target_mask=target,
        )

        self.assertEqual(result.matched_target_ids, (1, 2))
        self.assertEqual(result.matched_prediction_ids, (1, 2))
        self.assertEqual(result.unmatched_target_ids, ())
        self.assertEqual(result.unmatched_prediction_ids, ())
        self.assertEqual(
            tuple((pair.target_id, pair.prediction_id) for pair in result.matches),
            ((1, 2), (2, 1)),
        )

    def test_one_target_cannot_match_two_predictions(self) -> None:
        target = np.zeros((16, 16), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[5, 5] = True
        prediction[5, 3] = True
        prediction[5, 7] = True

        result = match_components_v2(
            prediction_mask=prediction,
            target_mask=target,
        )

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matched_target_ids, (1,))
        self.assertEqual(len(result.matched_prediction_ids), 1)
        self.assertEqual(len(result.unmatched_prediction_ids), 1)
        self.assertEqual(result.unmatched_prediction_pixels, 1)

    def test_match_radius_comparison_is_strict(self) -> None:
        target = np.zeros((16, 16), dtype=np.bool_)
        target[5, 5] = True

        at_boundary = np.zeros_like(target)
        at_boundary[5, 8] = True
        boundary = match_components_v2(
            prediction_mask=at_boundary,
            target_mask=target,
            match_radius=3.0,
        )
        self.assertEqual(boundary.matches, ())
        self.assertEqual(boundary.unmatched_prediction_pixels, 1)

        inside = np.zeros_like(target)
        inside[5, 7] = True
        matched = match_components_v2(
            prediction_mask=inside,
            target_mask=target,
            match_radius=3.0,
        )
        self.assertEqual(len(matched.matches), 1)
        self.assertEqual(matched.matches[0].centroid_distance, 2.0)

    def test_diagonal_pixels_are_one_eight_connected_component(self) -> None:
        target = np.zeros((8, 8), dtype=np.bool_)
        target[2, 2] = True
        target[3, 3] = True

        result = match_components_v2(
            prediction_mask=target.copy(),
            target_mask=target,
        )

        self.assertEqual(len(result.targets), 1)
        self.assertEqual(result.targets[0].area, 2)
        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(len(result.matches), 1)

    def test_empty_component_cases(self) -> None:
        empty = np.zeros((8, 8), dtype=np.bool_)
        one = empty.copy()
        one[3, 3] = True

        both_empty = match_components_v2(
            prediction_mask=empty,
            target_mask=empty,
        )
        self.assertEqual(both_empty.matches, ())
        self.assertEqual(both_empty.unmatched_prediction_pixels, 0)

        target_only = match_components_v2(
            prediction_mask=empty,
            target_mask=one,
        )
        self.assertEqual(target_only.unmatched_target_ids, (1,))
        self.assertEqual(target_only.unmatched_prediction_ids, ())

        prediction_only = match_components_v2(
            prediction_mask=one,
            target_mask=empty,
        )
        self.assertEqual(prediction_only.unmatched_target_ids, ())
        self.assertEqual(prediction_only.unmatched_prediction_ids, (1,))
        self.assertEqual(prediction_only.unmatched_prediction_pixels, 1)

    def test_inputs_are_strictly_validated(self) -> None:
        valid = np.zeros((8, 8), dtype=np.bool_)
        with self.assertRaisesRegex(TypeError, "prediction_mask.*bool"):
            match_components_v2(
                prediction_mask=valid.astype(np.uint8),
                target_mask=valid,
            )
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            match_components_v2(
                prediction_mask=valid[None],
                target_mask=valid[None],
            )
        with self.assertRaisesRegex(ValueError, "share shape"):
            match_components_v2(
                prediction_mask=valid,
                target_mask=np.zeros((7, 8), dtype=np.bool_),
            )
        for bad_radius in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(match_radius=bad_radius), self.assertRaises(
                ValueError
            ):
                match_components_v2(
                    prediction_mask=valid,
                    target_mask=valid,
                    match_radius=bad_radius,
                )
        with self.assertRaises(TypeError):
            match_components_v2(
                prediction_mask=valid,
                target_mask=valid,
                match_radius=True,
            )
        with self.assertRaisesRegex(ValueError, "8-connected"):
            match_components_v2(
                prediction_mask=valid,
                target_mask=valid,
                connectivity=1,
            )

    def test_results_are_stably_ordered_and_maps_are_read_only_int32(self) -> None:
        target = np.zeros((12, 12), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[2, 2] = target[8, 8] = True
        prediction[2, 3] = prediction[8, 9] = True

        first = match_components_v2(
            prediction_mask=prediction,
            target_mask=target,
        )
        second = match_components_v2(
            prediction_mask=prediction.copy(),
            target_mask=target.copy(),
        )

        self.assertEqual(first.targets, second.targets)
        self.assertEqual(first.predictions, second.predictions)
        self.assertEqual(first.matches, second.matches)
        np.testing.assert_array_equal(first.target_id_map, second.target_id_map)
        np.testing.assert_array_equal(
            first.prediction_id_map, second.prediction_id_map
        )
        self.assertEqual(first.target_id_map.dtype, np.dtype(np.int32))
        self.assertEqual(first.prediction_id_map.dtype, np.dtype(np.int32))
        self.assertFalse(first.target_id_map.flags.writeable)
        self.assertFalse(first.prediction_id_map.flags.writeable)


if __name__ == "__main__":
    unittest.main()
