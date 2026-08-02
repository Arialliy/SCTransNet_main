from __future__ import annotations

import unittest

import torch

from experiments import paper_three_dataset_v2 as paper
from experiments import three_dataset_v2_protocol as protocol


class ThreeDatasetV2DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = protocol.DEFAULT_MANIFEST_PATH

    def test_train_and_test_use_frozen_order_and_shapes(self) -> None:
        train = paper.ThreeDatasetV2TrainDataset(
            "NUAA-SIRST",
            protocol_manifest=self.manifest,
            return_metadata=False,
        )
        self.assertEqual(train.sample_ids[0], protocol.load_index(
            protocol.DEFAULT_DATASET_ROOT, "NUAA-SIRST", "train"
        )[0])
        image, mask = train[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(tuple(image.shape), (1, 256, 256))
        self.assertEqual(tuple(mask.shape), (1, 256, 256))

        test = paper.ThreeDatasetV2TestDataset(
            "NUAA-SIRST",
            protocol_manifest=self.manifest,
        )
        image, mask, original_hw, sample_id = test[0]
        self.assertEqual(image.shape, mask.shape)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(len(original_hw), 2)
        self.assertEqual(sample_id, test.sample_ids[0])

    def test_misc111_uses_only_internal_effective_overlay(self) -> None:
        test = paper.ThreeDatasetV2TestDataset(
            "NUAA-SIRST",
            protocol_manifest=self.manifest,
        )
        index = test.sample_ids.index("Misc_111")
        record = test.sample_record(index)
        self.assertTrue(record["correction_applied"])
        self.assertEqual(
            record["correction_id"], protocol.NUAA_MISC111_CORRECTION_ID
        )
        self.assertEqual(
            record["image_size_width_height"],
            record["effective_mask_size_width_height"],
        )

    def test_dataset_outside_formal_matrix_is_rejected(self) -> None:
        with self.assertRaises(protocol.ThreeDatasetV2ProtocolError):
            paper.ThreeDatasetV2TrainDataset(
                "SIRST3", protocol_manifest=self.manifest
            )


if __name__ == "__main__":
    unittest.main()

