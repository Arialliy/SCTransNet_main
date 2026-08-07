from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from experiments.component_matching_v2 import match_components_v2
from experiments.pbdr_v4_component_atlas import (
    atlas_maps_from_match,
    build_component_atlas,
)


class PBDRV4ComponentAtlasTests(unittest.TestCase):
    def test_atlas_maps_form_the_required_id_partitions(self) -> None:
        target = np.zeros((16, 16), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[2, 2] = True
        target[8, 8] = True
        prediction[2, 2] = True
        prediction[13, 13] = True

        result = match_components_v2(
            prediction_mask=prediction,
            target_mask=target,
        )
        atlas = atlas_maps_from_match(result)

        self.assertEqual(result.matched_target_ids, (1,))
        self.assertEqual(result.unmatched_target_ids, (2,))
        self.assertEqual(result.unmatched_prediction_ids, (2,))
        np.testing.assert_array_equal(
            atlas.preserve_ids > 0,
            result.target_id_map == 1,
        )
        np.testing.assert_array_equal(
            atlas.rescue_ids > 0,
            result.target_id_map == 2,
        )
        np.testing.assert_array_equal(
            atlas.suppress_ids > 0,
            result.prediction_id_map == 2,
        )
        np.testing.assert_array_equal(
            (atlas.rescue_ids > 0) | (atlas.preserve_ids > 0),
            target,
        )
        self.assertFalse(
            bool(np.any((atlas.rescue_ids > 0) & (atlas.preserve_ids > 0)))
        )

    def test_unmatched_suppress_component_may_overlap_target(self) -> None:
        target = np.zeros((12, 12), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[5, 1:8] = True  # centroid is (5, 4)
        prediction[5, 7] = True  # overlaps target but centroid distance is 3

        result, atlas = build_component_atlas(
            prediction_mask=prediction,
            target_mask=target,
        )

        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_target_ids, (1,))
        self.assertEqual(result.unmatched_prediction_ids, (1,))
        self.assertTrue(atlas.rescue_ids[5, 7] > 0)
        self.assertTrue(atlas.suppress_ids[5, 7] > 0)
        self.assertEqual(
            int(np.count_nonzero((atlas.suppress_ids > 0) & target)),
            1,
        )

    def test_atlas_preserves_source_ids_dtype_and_read_only_contract(self) -> None:
        target = np.zeros((16, 16), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[1, 1] = target[5, 5] = target[10, 10] = True
        prediction[1, 1] = prediction[10, 10] = True

        result, atlas = build_component_atlas(
            prediction_mask=prediction,
            target_mask=target,
        )

        # Target ID 2 is rescued while IDs 1 and 3 are preserved. IDs are not
        # compacted independently inside the filtered maps.
        self.assertEqual(
            tuple(int(value) for value in np.unique(atlas.rescue_ids)),
            (0, 2),
        )
        self.assertEqual(
            tuple(int(value) for value in np.unique(atlas.preserve_ids)),
            (0, 1, 3),
        )
        self.assertEqual(result.unmatched_target_ids, (2,))
        for id_map in (
            atlas.rescue_ids,
            atlas.suppress_ids,
            atlas.preserve_ids,
        ):
            self.assertEqual(id_map.dtype, np.dtype(np.int32))
            self.assertTrue(id_map.flags.c_contiguous)
            self.assertFalse(id_map.flags.writeable)

    def test_direct_builder_matches_two_step_pure_functions(self) -> None:
        target = np.zeros((10, 10), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[2, 2] = target[7, 7] = True
        prediction[2, 3] = prediction[9, 1] = True

        expected_result = match_components_v2(
            prediction_mask=prediction,
            target_mask=target,
        )
        expected_atlas = atlas_maps_from_match(expected_result)
        observed_result, observed_atlas = build_component_atlas(
            prediction_mask=prediction,
            target_mask=target,
        )

        self.assertEqual(expected_result.matches, observed_result.matches)
        np.testing.assert_array_equal(
            expected_atlas.rescue_ids, observed_atlas.rescue_ids
        )
        np.testing.assert_array_equal(
            expected_atlas.suppress_ids, observed_atlas.suppress_ids
        )
        np.testing.assert_array_equal(
            expected_atlas.preserve_ids, observed_atlas.preserve_ids
        )

    def test_atlas_rejects_tampered_match_partitions_and_pixel_count(self) -> None:
        target = np.zeros((12, 12), dtype=np.bool_)
        prediction = np.zeros_like(target)
        target[2, 2] = True
        prediction[2, 2] = prediction[9, 9] = True
        result = match_components_v2(
            prediction_mask=prediction,
            target_mask=target,
        )

        bad_partition = replace(result, unmatched_prediction_ids=())
        with self.assertRaisesRegex(ValueError, "do not form a partition"):
            atlas_maps_from_match(bad_partition)

        bad_pixels = replace(result, unmatched_prediction_pixels=0)
        with self.assertRaisesRegex(ValueError, "pixel count differs"):
            atlas_maps_from_match(bad_pixels)

    def test_empty_atlas_maps_are_valid(self) -> None:
        empty = np.zeros((6, 7), dtype=np.bool_)
        result, atlas = build_component_atlas(
            prediction_mask=empty,
            target_mask=empty,
        )

        self.assertEqual(result.matches, ())
        self.assertEqual(atlas.rescue_ids.shape, empty.shape)
        self.assertEqual(int(atlas.rescue_ids.sum()), 0)
        self.assertEqual(int(atlas.suppress_ids.sum()), 0)
        self.assertEqual(int(atlas.preserve_ids.sum()), 0)


if __name__ == "__main__":
    unittest.main()
