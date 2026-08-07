from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from experiments import pbdr_v4_internal_dataset as subject
from experiments import three_dataset_v2_protocol as data_protocol


DATASET = "NUDT-SIRST"


class SyntheticInternalFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.selected_ids = ["sample_b", "sample_a"]
        self.official_ids = ["sample_a", "sample_b", "heldout_train"]
        self.paths: dict[str, tuple[Path, Path]] = {}
        self.calls: list[tuple[str, str, str, frozenset[str]]] = []
        for offset, identifier in enumerate(("sample_a", "sample_b")):
            height, width = 35 + offset, 65 + offset
            image = np.full((height, width), 100 + offset, dtype=np.uint16)
            image[0, 0] = 120 + offset
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[2:4, 3:5] = 255
            image_path = self.data_root / f"{identifier}_image.png"
            mask_path = self.data_root / f"{identifier}_mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            self.paths[identifier] = (image_path, mask_path)
        self.returned_split = "train"
        self.returned_dataset = DATASET
        self.returned_sample_id: str | None = None
        self.correction_id: str | None = None

    def __enter__(self) -> "SyntheticInternalFixture":
        return self

    def __exit__(self, *_: object) -> None:
        self.temporary.cleanup()

    def resolver(
        self,
        _data_root: Path,
        dataset_name: str,
        sample_id: str,
        *,
        split: str,
        known_ids: frozenset[str],
    ) -> data_protocol.ResolvedSample:
        self.calls.append((dataset_name, sample_id, split, known_ids))
        image_path, mask_path = self.paths[sample_id]
        return data_protocol.ResolvedSample(
            dataset_name=self.returned_dataset,
            split=self.returned_split,
            sample_id=self.returned_sample_id or sample_id,
            image_path=image_path,
            raw_mask_path=mask_path,
            mask_path=mask_path,
            correction_id=self.correction_id,
        )

    def dataset(self, **overrides: object) -> subject.PBDRV4InternalInferenceDataset:
        arguments: dict[str, object] = {
            "selected_ids": self.selected_ids,
            "known_official_train_ids": self.official_ids,
            "manifest_scope": "development_train_ids",
            "selected_ids_ordered_sha256": data_protocol.ordered_ids_sha256(
                self.selected_ids
            ),
            "official_train_count": len(self.official_ids),
            "official_train_ordered_ids_sha256": data_protocol.ordered_ids_sha256(
                self.official_ids
            ),
            "dataset_name": DATASET,
            "data_root": self.data_root,
            "sample_resolver": self.resolver,
        }
        arguments.update(overrides)
        return subject.PBDRV4InternalInferenceDataset(**arguments)  # type: ignore[arg-type]


class PBDRV4InternalDatasetTests(unittest.TestCase):
    def test_ordered_projection_is_no_augmentation_train_only_and_padded(self) -> None:
        with SyntheticInternalFixture() as fixture:
            dataset = fixture.dataset()
            self.assertEqual(fixture.calls, [])
            image, target, size, identifier = dataset[0]

            self.assertEqual(identifier, "sample_b")
            self.assertEqual(size, (36, 66))
            self.assertEqual(tuple(image.shape), (1, 64, 96))
            self.assertEqual(tuple(target.shape), (1, 64, 96))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(target.dtype, torch.float32)
            normalization = data_protocol.get_legacy_normalization(DATASET)
            expected = (
                np.float32(121.0) - np.float32(normalization["mean"])
            ) / np.float32(normalization["std"])
            self.assertAlmostEqual(float(image[0, 0, 0]), float(expected), places=6)
            self.assertEqual(float(target[0, 2, 3]), 1.0)
            self.assertTrue(torch.count_nonzero(image[:, 36:, :]) == 0)
            self.assertTrue(torch.count_nonzero(image[:, :, 66:]) == 0)
            self.assertTrue(torch.count_nonzero(target[:, 36:, :]) == 0)
            self.assertEqual(len(fixture.calls), 1)
            dataset_name, sample_id, split, known_ids = fixture.calls[0]
            self.assertEqual((dataset_name, sample_id), (DATASET, "sample_b"))
            self.assertEqual(split, "train")
            self.assertEqual(known_ids, frozenset(fixture.official_ids))

            repeated = dataset[0]
            self.assertTrue(torch.equal(image, repeated[0]))
            self.assertTrue(torch.equal(target, repeated[1]))

    def test_both_and_only_both_manifest_scopes_are_allowed(self) -> None:
        with SyntheticInternalFixture() as fixture:
            development = fixture.dataset(
                manifest_scope="development_train_ids"
            )
            validation = fixture.dataset(
                manifest_scope="internal_validation_ids"
            )
            self.assertEqual(development.manifest_scope, "development_train_ids")
            self.assertEqual(validation.manifest_scope, "internal_validation_ids")
            for invalid in ("train", "validation", "test", "official_test", ""):
                with self.subTest(scope=invalid), self.assertRaisesRegex(
                    subject.PBDRV4InternalDatasetError,
                    "manifest_scope",
                ):
                    fixture.dataset(manifest_scope=invalid)

    def test_selected_projection_must_be_ordered_unique_subset(self) -> None:
        with SyntheticInternalFixture() as fixture:
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "duplicate"):
                fixture.dataset(selected_ids=["sample_a", "sample_a"])
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "not a subset"):
                outside = ["outside"]
                fixture.dataset(
                    selected_ids=outside,
                    selected_ids_ordered_sha256=data_protocol.ordered_ids_sha256(
                        outside
                    ),
                )
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "selected ordered-ID SHA"):
                fixture.dataset(selected_ids_ordered_sha256="0" * 64)

    def test_complete_official_train_count_order_and_uniqueness_are_bound(self) -> None:
        with SyntheticInternalFixture() as fixture:
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "count differs"):
                fixture.dataset(official_train_count=4)
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "ordered-ID SHA"):
                fixture.dataset(official_train_ordered_ids_sha256="0" * 64)
            duplicate = ["sample_a", "sample_a", "heldout_train"]
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "duplicate"):
                fixture.dataset(known_official_train_ids=duplicate)

    def test_resolver_identity_scope_and_correction_are_fail_closed(self) -> None:
        with SyntheticInternalFixture() as fixture:
            fixture.returned_split = "test"
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "non-training"):
                dataset[0]
            self.assertEqual(fixture.calls[0][2], "train")

        with SyntheticInternalFixture() as fixture:
            fixture.returned_dataset = "IRSTD-1K"
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "dataset differs"):
                dataset[0]

        with SyntheticInternalFixture() as fixture:
            fixture.returned_sample_id = "sample_a"
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "sample ID differs"):
                dataset[0]

        with SyntheticInternalFixture() as fixture:
            fixture.correction_id = "test-only-overlay"
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "correction overlay"):
                dataset[0]

    def test_missing_symlink_and_shape_mismatch_are_rejected(self) -> None:
        with SyntheticInternalFixture() as fixture:
            image_path, _ = fixture.paths["sample_b"]
            image_path.unlink()
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "regular non-symlink"):
                dataset[0]

        with SyntheticInternalFixture() as fixture:
            image_path, _ = fixture.paths["sample_b"]
            external = fixture.root / "external.png"
            image_path.replace(external)
            try:
                os.symlink(external, image_path)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "non-symlink"):
                dataset[0]

        with SyntheticInternalFixture() as fixture:
            _, mask_path = fixture.paths["sample_b"]
            Image.fromarray(np.zeros((10, 11), dtype=np.uint8)).save(mask_path)
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "dimensions differ"):
                dataset[0]

    def test_nonfinite_mask_is_rejected(self) -> None:
        with SyntheticInternalFixture() as fixture:
            _, old_mask_path = fixture.paths["sample_b"]
            float_mask_path = fixture.data_root / "sample_b_mask.tiff"
            mask = np.zeros((36, 66), dtype=np.float32)
            mask[0, 0] = np.nan
            Image.fromarray(mask, mode="F").save(float_mask_path)
            image_path, _ = fixture.paths["sample_b"]
            fixture.paths["sample_b"] = (image_path, float_mask_path)
            old_mask_path.unlink()
            dataset = fixture.dataset()
            with self.assertRaisesRegex(subject.PBDRV4InternalDatasetError, "non-finite"):
                dataset[0]

    def test_source_has_no_index_loader_or_root_dataset_dependency(self) -> None:
        source = inspect.getsource(subject)
        self.assertNotIn("load_" + "index(", source)
        self.assertNotIn("from " + "dataset import", source)
        self.assertNotIn("import " + "dataset\n", source)


if __name__ == "__main__":
    unittest.main()
