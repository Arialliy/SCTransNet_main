from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from experiments import three_dataset_v2_protocol as data_protocol
from experiments.pbdr_v4_atlas_dataset import (
    ATLAS_MANIFEST_SCHEMA,
    ATLAS_MAP_NAMES,
    ATLAS_SPLIT_SCOPE,
    PBDRV4AtlasDatasetError,
    PBDRV4AtlasTrainDataset,
    apply_stateless_geometry,
    array_semantic_sha256,
    file_sha256,
    matcher_source_sha256,
    ordered_ids_sha256,
)


PARENT_SHA = "a" * 64
DATASET = "NUDT-SIRST"


class SyntheticAtlasFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.atlas_root = self.root / "atlas"
        self.atlas_root.mkdir()
        self.manifest_path = self.atlas_root / "manifest.json"
        self.development_ids = ["sample_a", "sample_b"]
        self.official_ids = ["sample_a", "sample_b", "heldout_train"]
        self.matcher_sha = matcher_source_sha256()
        self.calls: list[tuple[str, str, str, frozenset[str]]] = []
        self.sample_paths: dict[str, tuple[Path, Path]] = {}
        self.arrays: dict[str, dict[str, np.ndarray]] = {}
        self.manifest: dict[str, object] = {}
        self._build()

    def close(self) -> None:
        self.temporary.cleanup()

    def __enter__(self) -> "SyntheticAtlasFixture":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _build(self) -> None:
        records: list[dict[str, object]] = []
        for offset, identifier in enumerate(self.development_ids):
            height, width = 260 + offset, 270 + 2 * offset
            image = (
                np.arange(height * width, dtype=np.uint32).reshape(height, width)
                % 4096
            ).astype(np.uint16)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[10:13, 20:23] = 255
            mask[220:224, 240:244] = 255
            image_path = self.data_root / f"{identifier}_image.png"
            mask_path = self.data_root / f"{identifier}_mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            self.sample_paths[identifier] = (image_path, mask_path)

            preserve = np.zeros((height, width), dtype=np.int32)
            rescue = np.zeros_like(preserve)
            suppress = np.zeros_like(preserve)
            preserve[10:13, 20:23] = 1
            rescue[220:224, 240:244] = 2
            suppress[100:102, 110:112] = 1
            arrays = {
                "rescue_ids": rescue,
                "suppress_ids": suppress,
                "preserve_ids": preserve,
                "image_id": np.asarray(identifier),
                "parent_state_sha256": np.asarray(PARENT_SHA),
                "matcher_source_sha256": np.asarray(self.matcher_sha),
            }
            self.arrays[identifier] = arrays
            filename = f"{identifier}.npz"
            path = self.atlas_root / filename
            np.savez_compressed(path, **arrays)
            records.append(
                {
                    "image_id": identifier,
                    "filename": filename,
                    "file_sha256": file_sha256(path),
                    "parent_state_sha256": PARENT_SHA,
                    "matcher_source_sha256": self.matcher_sha,
                    "maps": {
                        name: {
                            "semantic_sha256": array_semantic_sha256(arrays[name]),
                            "shape": list(arrays[name].shape),
                            "dtype": "int32",
                        }
                        for name in ATLAS_MAP_NAMES
                    },
                }
            )
        self.manifest = {
            "schema": ATLAS_MANIFEST_SCHEMA,
            "dataset": DATASET,
            "split_scope": ATLAS_SPLIT_SCOPE,
            "official_test_accessed": False,
            "development_train_ids": list(self.development_ids),
            "development_train_ids_sha256": ordered_ids_sha256(
                self.development_ids
            ),
            "official_train_ids_sha256": ordered_ids_sha256(self.official_ids),
            "parent_state_sha256": PARENT_SHA,
            "matcher_source_sha256": self.matcher_sha,
            "samples": records,
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def rewrite_npz(
        self,
        identifier: str,
        updates: dict[str, np.ndarray],
        *,
        rebind_maps: bool = False,
    ) -> None:
        arrays = dict(self.arrays[identifier])
        arrays.update(updates)
        self.arrays[identifier] = arrays
        path = self.atlas_root / f"{identifier}.npz"
        np.savez_compressed(path, **arrays)
        record = next(
            item
            for item in self.manifest["samples"]  # type: ignore[index]
            if item["image_id"] == identifier
        )
        record["file_sha256"] = file_sha256(path)
        if rebind_maps:
            for name in ATLAS_MAP_NAMES:
                record["maps"][name] = {
                    "semantic_sha256": array_semantic_sha256(arrays[name]),
                    "shape": list(arrays[name].shape),
                    "dtype": "int32",
                }
        self.write_manifest()

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
        image_path, mask_path = self.sample_paths[sample_id]
        return data_protocol.ResolvedSample(
            dataset_name=dataset_name,
            split=split,
            sample_id=sample_id,
            image_path=image_path,
            raw_mask_path=mask_path,
            mask_path=mask_path,
            correction_id=None,
        )

    def dataset(self, **overrides: object) -> PBDRV4AtlasTrainDataset:
        arguments: dict[str, object] = {
            "development_train_ids": self.development_ids,
            "known_official_train_ids": self.official_ids,
            "dataset_name": DATASET,
            "data_root": self.data_root,
            "atlas_root": self.atlas_root,
            "atlas_manifest": self.manifest_path,
            "parent_state_sha256": PARENT_SHA,
            "normalization": {"mean": 0.0, "std": 1.0},
            "sample_resolver": self.resolver,
        }
        arguments.update(overrides)
        return PBDRV4AtlasTrainDataset(**arguments)  # type: ignore[arg-type]


class PBDRV4AtlasDatasetTests(unittest.TestCase):
    def test_pure_geometry_uses_exact_v3_order_for_all_arrays(self) -> None:
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        plan = data_protocol.StatelessTransformPlan(
            augmentation_seed=1,
            crop_top=1,
            crop_left=2,
            crop_size=4,
            padded_height=6,
            padded_width=7,
            crop_attempts=1,
            flip_axis0=True,
            flip_axis1=True,
            transpose=True,
        )
        categorical = np.arange(20, dtype=np.int32).reshape(4, 5)
        observed = apply_stateless_geometry(
            plan,
            image=base,
            target=base + 100.0,
            rescue_ids=categorical,
            suppress_ids=categorical + 100,
            preserve_ids=categorical + 200,
        )

        def expected(value: np.ndarray) -> np.ndarray:
            padded = np.pad(value, ((0, 2), (0, 2)))
            cropped = padded[1:5, 2:6]
            return np.ascontiguousarray(cropped[::-1, ::-1].T)

        np.testing.assert_array_equal(observed.image, expected(base))
        np.testing.assert_array_equal(observed.target, expected(base + 100.0))
        np.testing.assert_array_equal(
            observed.rescue_ids, expected(categorical)
        )
        np.testing.assert_array_equal(
            observed.suppress_ids, expected(categorical + 100)
        )
        np.testing.assert_array_equal(
            observed.preserve_ids, expected(categorical + 200)
        )
        for value in (
            observed.image,
            observed.target,
            observed.rescue_ids,
            observed.suppress_ids,
            observed.preserve_ids,
        ):
            self.assertTrue(value.flags.c_contiguous)
        self.assertEqual(observed.rescue_ids.dtype, np.dtype(np.int32))

    def test_pure_geometry_rejects_wrong_id_dtype_and_shape(self) -> None:
        base = np.zeros((4, 5), dtype=np.float32)
        ids = np.zeros((4, 5), dtype=np.int32)
        plan = data_protocol.StatelessTransformPlan(
            1, 0, 0, 4, 4, 5, 1, False, False, False
        )
        with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "must be int32"):
            apply_stateless_geometry(
                plan,
                image=base,
                target=base,
                rescue_ids=ids.astype(np.int64),
                suppress_ids=ids,
                preserve_ids=ids,
            )
        with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "share shape"):
            apply_stateless_geometry(
                plan,
                image=base,
                target=base,
                rescue_ids=ids[:-1],
                suppress_ids=ids,
                preserve_ids=ids,
            )

    def test_dataset_returns_six_fields_and_resolves_train_only(self) -> None:
        with SyntheticAtlasFixture() as fixture:
            dataset = fixture.dataset()
            self.assertEqual(fixture.calls, [])
            dataset.set_epoch(7)
            image, target, rescue, suppress, preserve, identifier = dataset[0]

            self.assertEqual(identifier, "sample_a")
            self.assertEqual(tuple(image.shape), (1, 256, 256))
            self.assertEqual(tuple(target.shape), (1, 256, 256))
            self.assertEqual(tuple(rescue.shape), (1, 256, 256))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(target.dtype, torch.float32)
            self.assertEqual(rescue.dtype, torch.int32)
            self.assertEqual(suppress.dtype, torch.int32)
            self.assertEqual(preserve.dtype, torch.int32)
            self.assertEqual(len(fixture.calls), 1)
            dataset_name, sample_id, split, known_ids = fixture.calls[0]
            self.assertEqual((dataset_name, sample_id), (DATASET, "sample_a"))
            self.assertEqual(split, "train")
            self.assertEqual(known_ids, frozenset(fixture.official_ids))
            self.assertNotIn("test", [call[2] for call in fixture.calls])

    def test_development_ids_must_be_unique_official_train_subset(self) -> None:
        with SyntheticAtlasFixture() as fixture:
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "duplicate"):
                fixture.dataset(
                    development_train_ids=["sample_a", "sample_a"]
                )
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "not a subset"):
                fixture.dataset(development_train_ids=["outside"])

    def test_missing_extra_and_symlink_npz_are_rejected(self) -> None:
        with SyntheticAtlasFixture() as fixture:
            (fixture.atlas_root / "sample_b.npz").unlink()
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "missing or extra"):
                fixture.dataset()

        with SyntheticAtlasFixture() as fixture:
            np.savez_compressed(
                fixture.atlas_root / "extra_sample.npz",
                value=np.asarray(1),
            )
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "missing or extra"):
                fixture.dataset()

        with SyntheticAtlasFixture() as fixture:
            original = fixture.atlas_root / "sample_a.npz"
            external = fixture.root / "external.npz"
            original.replace(external)
            try:
                os.symlink(external, original)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "symlink"):
                fixture.dataset()

    def test_npz_image_parent_and_matcher_identities_are_checked(self) -> None:
        cases = (
            ("image_id", np.asarray("wrong"), "image_id differs"),
            (
                "parent_state_sha256",
                np.asarray("b" * 64),
                "parent_state_sha256 differs",
            ),
            (
                "matcher_source_sha256",
                np.asarray("c" * 64),
                "matcher_source_sha256 differs",
            ),
        )
        for key, value, message in cases:
            with self.subTest(key=key), SyntheticAtlasFixture() as fixture:
                fixture.rewrite_npz("sample_a", {key: value})
                with self.assertRaisesRegex(PBDRV4AtlasDatasetError, message):
                    fixture.dataset()

    def test_map_dtype_shape_and_semantic_sha_are_checked(self) -> None:
        with SyntheticAtlasFixture() as fixture:
            bad = fixture.arrays["sample_a"]["rescue_ids"].astype(np.int64)
            fixture.rewrite_npz("sample_a", {"rescue_ids": bad})
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "dtype differs"):
                fixture.dataset()

        with SyntheticAtlasFixture() as fixture:
            bad = fixture.arrays["sample_a"]["rescue_ids"][:-1]
            fixture.rewrite_npz("sample_a", {"rescue_ids": bad})
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "shape differs"):
                fixture.dataset()

        with SyntheticAtlasFixture() as fixture:
            bad = fixture.arrays["sample_a"]["rescue_ids"].copy()
            bad[0, 0] = 99
            fixture.rewrite_npz("sample_a", {"rescue_ids": bad})
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "semantic SHA differs"):
                fixture.dataset()

    def test_manifest_parent_matcher_and_sample_set_are_checked(self) -> None:
        with SyntheticAtlasFixture() as fixture:
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "parent state SHA"):
                fixture.dataset(parent_state_sha256="b" * 64)

        with SyntheticAtlasFixture() as fixture:
            fixture.manifest["matcher_source_sha256"] = "c" * 64
            fixture.write_manifest()
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "matcher source SHA"):
                fixture.dataset()

        with SyntheticAtlasFixture() as fixture:
            fixture.manifest["samples"] = fixture.manifest["samples"][:-1]
            fixture.write_manifest()
            with self.assertRaisesRegex(PBDRV4AtlasDatasetError, "missing, extra"):
                fixture.dataset()

    def test_target_partition_is_checked_before_transform(self) -> None:
        with SyntheticAtlasFixture() as fixture:
            preserve = fixture.arrays["sample_a"]["preserve_ids"].copy()
            preserve[10, 20] = 0
            fixture.rewrite_npz(
                "sample_a",
                {"preserve_ids": preserve},
                rebind_maps=True,
            )
            dataset = fixture.dataset()
            with self.assertRaisesRegex(
                PBDRV4AtlasDatasetError, "do not partition target"
            ):
                dataset[0]
            self.assertEqual(fixture.calls[0][2], "train")

    def test_semantic_hash_is_layout_independent_but_value_sensitive(self) -> None:
        base = np.arange(20, dtype=np.int32).reshape(4, 5)
        fortran = np.asfortranarray(base)
        self.assertEqual(
            array_semantic_sha256(base),
            array_semantic_sha256(fortran),
        )
        changed = base.copy()
        changed[0, 0] += 1
        self.assertNotEqual(
            array_semantic_sha256(base),
            array_semantic_sha256(changed),
        )


if __name__ == "__main__":
    unittest.main()
