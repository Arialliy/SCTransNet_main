from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments import three_dataset_v2_protocol as data_protocol
from experiments.build_pbdr_v4_component_atlas import (
    FrozenCurrentSample,
    PBDRV4AtlasBuildError,
    build_pbdr_v4_component_atlas,
    validate_component_atlas_artifact,
)
from experiments.pbdr_v4_atlas_dataset import (
    ATLAS_NPZ_KEYS,
    PBDRV4AtlasTrainDataset,
    matcher_source_sha256,
    ordered_ids_sha256,
)


DATASET = "NUDT-SIRST"
ROLE = "best_pd"
PARENT_CHECKPOINT_SHA = "1" * 64
PARENT_STATE_SHA = "2" * 64
SPLIT_PROJECTION_SHA = "3" * 64
METRIC_SOURCE_SHA = "4" * 64


def synthetic_samples() -> dict[str, FrozenCurrentSample]:
    target_a = np.zeros((24, 28), dtype=np.float32)
    probability_a = np.zeros_like(target_a)
    target_a[2, 2] = 1.0
    target_a[10, 10] = 1.0
    target_a[5, 5] = 0.5  # strict target > 0.5 excludes this pixel
    probability_a[2, 2] = 0.9
    probability_a[10, 10] = 0.5  # strict probability > 0.5 misses target 2
    probability_a[18, 20] = np.nextafter(
        np.float32(0.5), np.float32(1.0)
    )

    target_b = np.zeros((17, 19), dtype=np.float64)
    probability_b = np.zeros_like(target_b)
    target_b[8:10, 8:10] = 1.0
    probability_b[8:10, 8:10] = 1.0
    return {
        "sample_a": FrozenCurrentSample(probability_a, target_a),
        "sample_b": FrozenCurrentSample(probability_b, target_b),
    }


def build_arguments(
    destination: Path,
    *,
    samples: dict[str, FrozenCurrentSample] | None = None,
) -> dict[str, object]:
    official_ids = ["sample_a", "sample_b", "heldout_train"]
    return {
        "dataset_name": DATASET,
        "role": ROLE,
        "development_train_ids": ["sample_a", "sample_b"],
        "frozen_samples": synthetic_samples() if samples is None else samples,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "parent_state_sha256": PARENT_STATE_SHA,
        "split_projection_sha256": SPLIT_PROJECTION_SHA,
        "official_train_ids_sha256": ordered_ids_sha256(official_ids),
        "metric_source_sha256": METRIC_SOURCE_SHA,
        "matcher_source_sha256": matcher_source_sha256(),
        "source_lock_sha256": "9" * 64,
        "output_root": destination,
    }


class BuildPBDRV4ComponentAtlasTests(unittest.TestCase):
    def test_build_uses_strict_half_threshold_and_commits_valid_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            destination = parent / "atlas"
            result = build_pbdr_v4_component_atlas(
                **build_arguments(destination)  # type: ignore[arg-type]
            )

            self.assertEqual(result.root, destination)
            self.assertEqual(result.sample_count, 2)
            self.assertTrue(result.manifest_path.is_file())
            manifest = validate_component_atlas_artifact(destination)
            self.assertEqual(manifest["manifest_sha256"], result.manifest_sha256)
            self.assertEqual(manifest["role"], ROLE)
            self.assertEqual(manifest["source_lock_sha256"], "9" * 64)
            self.assertIs(manifest["official_test_accessed"], False)
            self.assertEqual(
                manifest["threshold_contract"],
                {
                    "probability_threshold": 0.5,
                    "probability_comparison": ">",
                    "target_threshold": 0.5,
                    "target_comparison": ">",
                },
            )
            first = manifest["samples"][0]
            statistics = first["component_statistics"]
            self.assertEqual(statistics["target_component_count"], 2)
            self.assertEqual(statistics["matched_target_component_count"], 1)
            self.assertEqual(statistics["unmatched_target_component_count"], 1)
            self.assertEqual(statistics["prediction_component_count"], 2)
            self.assertEqual(statistics["unmatched_prediction_component_count"], 1)
            self.assertEqual(statistics["unmatched_prediction_pixel_count"], 1)

            with np.load(destination / "sample_a.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), ATLAS_NPZ_KEYS)
                self.assertTrue(all(archive[name].dtype.kind != "O" for name in archive.files))
                self.assertEqual(archive["image_id"].item(), "sample_a")
                self.assertEqual(
                    archive["parent_state_sha256"].item(), PARENT_STATE_SHA
                )
                self.assertEqual(
                    archive["matcher_source_sha256"].item(),
                    matcher_source_sha256(),
                )
                # Target (10,10) is a strict-half rescue; probability (18,20)
                # is a strict-above-half suppress component.
                self.assertGreater(archive["rescue_ids"][10, 10], 0)
                self.assertGreater(archive["suppress_ids"][18, 20], 0)
                self.assertEqual(archive["preserve_ids"][5, 5], 0)

            self.assertEqual(
                list(parent.glob(".atlas.stage.*")),
                [],
            )

    def test_generated_artifact_is_directly_compatible_with_atlas_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "atlas"
            samples = synthetic_samples()
            result = build_pbdr_v4_component_atlas(
                **build_arguments(destination, samples=samples)  # type: ignore[arg-type]
            )
            data_root = root / "data"
            data_root.mkdir()
            paths: dict[str, tuple[Path, Path]] = {}
            for image_id, sample in samples.items():
                image = np.arange(
                    sample.target.size, dtype=np.uint16
                ).reshape(sample.target.shape)
                mask = (sample.target > 0.5).astype(np.uint8) * 255
                image_path = data_root / f"{image_id}_image.png"
                mask_path = data_root / f"{image_id}_mask.png"
                Image.fromarray(image).save(image_path)
                Image.fromarray(mask).save(mask_path)
                paths[image_id] = (image_path, mask_path)

            calls: list[str] = []

            def resolver(
                _root: Path,
                dataset_name: str,
                sample_id: str,
                *,
                split: str,
                known_ids: frozenset[str],
            ) -> data_protocol.ResolvedSample:
                self.assertEqual(split, "train")
                self.assertEqual(
                    known_ids,
                    frozenset(("sample_a", "sample_b", "heldout_train")),
                )
                calls.append(split)
                image_path, mask_path = paths[sample_id]
                return data_protocol.ResolvedSample(
                    dataset_name=dataset_name,
                    split=split,
                    sample_id=sample_id,
                    image_path=image_path,
                    raw_mask_path=mask_path,
                    mask_path=mask_path,
                    correction_id=None,
                )

            dataset = PBDRV4AtlasTrainDataset(
                ["sample_a", "sample_b"],
                ["sample_a", "sample_b", "heldout_train"],
                dataset_name=DATASET,
                data_root=data_root,
                atlas_root=result.root,
                atlas_manifest=result.manifest_path,
                parent_state_sha256=PARENT_STATE_SHA,
                normalization={"mean": 0.0, "std": 1.0},
                sample_resolver=resolver,
            )
            image, target, rescue, suppress, preserve, image_id = dataset[0]
            self.assertEqual(image_id, "sample_a")
            self.assertEqual(tuple(image.shape), (1, 256, 256))
            self.assertEqual(rescue.dtype, np_to_torch_int32())
            self.assertEqual(suppress.dtype, np_to_torch_int32())
            self.assertEqual(preserve.dtype, np_to_torch_int32())
            self.assertEqual(calls, ["train"])
            self.assertEqual(target.shape, image.shape)

    def test_missing_extra_and_duplicate_inputs_are_rejected_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "missing"
            samples = synthetic_samples()
            samples.pop("sample_b")
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "missing or extra"):
                build_pbdr_v4_component_atlas(
                    **build_arguments(destination, samples=samples)  # type: ignore[arg-type]
                )
            self.assertFalse(destination.exists())

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "extra"
            samples = synthetic_samples()
            samples["extra"] = next(iter(samples.values()))
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "missing or extra"):
                build_pbdr_v4_component_atlas(
                    **build_arguments(destination, samples=samples)  # type: ignore[arg-type]
                )

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "duplicate"
            arguments = build_arguments(destination)
            arguments["development_train_ids"] = ["sample_a", "sample_a"]
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "duplicates"):
                build_pbdr_v4_component_atlas(**arguments)  # type: ignore[arg-type]

    def test_invalid_sample_cleans_unique_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            destination = parent / "atlas"
            samples = synthetic_samples()
            bad_probability = samples["sample_b"].probability.copy()
            bad_probability[0, 0] = np.nan
            samples["sample_b"] = FrozenCurrentSample(
                bad_probability, samples["sample_b"].target
            )
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "non-finite"):
                build_pbdr_v4_component_atlas(
                    **build_arguments(destination, samples=samples)  # type: ignore[arg-type]
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.glob(".atlas.stage.*")), [])

    def test_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "atlas"
            first = build_pbdr_v4_component_atlas(
                **build_arguments(destination)  # type: ignore[arg-type]
            )
            original = first.manifest_path.read_bytes()
            with self.assertRaises(FileExistsError):
                build_pbdr_v4_component_atlas(
                    **build_arguments(destination)  # type: ignore[arg-type]
                )
            self.assertEqual(first.manifest_path.read_bytes(), original)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "atlas"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                build_pbdr_v4_component_atlas(
                    **build_arguments(destination)  # type: ignore[arg-type]
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "atlas"
            external = root / "external"
            external.mkdir()
            try:
                os.symlink(external, destination)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(FileExistsError):
                build_pbdr_v4_component_atlas(
                    **build_arguments(destination)  # type: ignore[arg-type]
                )

    def test_matcher_binding_and_normalized_probability_contract_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "atlas"
            arguments = build_arguments(destination)
            arguments["matcher_source_sha256"] = "f" * 64
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "active canonical"):
                build_pbdr_v4_component_atlas(**arguments)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "atlas"
            samples = synthetic_samples()
            bad = samples["sample_a"].probability.copy()
            bad[0, 0] = 1.1
            samples["sample_a"] = FrozenCurrentSample(
                bad, samples["sample_a"].target
            )
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, r"\[0, 1\]"):
                build_pbdr_v4_component_atlas(
                    **build_arguments(destination, samples=samples)  # type: ignore[arg-type]
                )

    def test_artifact_validator_rejects_byte_tamper_missing_extra_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "atlas"
            build_pbdr_v4_component_atlas(
                **build_arguments(destination)  # type: ignore[arg-type]
            )
            with (destination / "sample_a.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "file SHA differs"):
                validate_component_atlas_artifact(destination)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "atlas"
            build_pbdr_v4_component_atlas(
                **build_arguments(destination)  # type: ignore[arg-type]
            )
            (destination / "sample_b.npz").unlink()
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "missing or extra"):
                validate_component_atlas_artifact(destination)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "atlas"
            build_pbdr_v4_component_atlas(
                **build_arguments(destination)  # type: ignore[arg-type]
            )
            (destination / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "missing or extra"):
                validate_component_atlas_artifact(destination)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "atlas"
            build_pbdr_v4_component_atlas(
                **build_arguments(destination)  # type: ignore[arg-type]
            )
            external = root / "external"
            external.write_text("x", encoding="utf-8")
            try:
                os.symlink(external, destination / "bad-link")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(PBDRV4AtlasBuildError, "symlink"):
                validate_component_atlas_artifact(destination)


def np_to_torch_int32():
    import torch

    return torch.int32


if __name__ == "__main__":
    unittest.main()
