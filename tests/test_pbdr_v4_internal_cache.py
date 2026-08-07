from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments import pbdr_v4_internal_cache as cache
from experiments import pbdr_v4_split_authority as split_authority


DATASET = "NUAA-SIRST"
ROLE = "best_miou"
DEVELOPMENT_IDS = ("synthetic-dev-0", "synthetic-dev-1")
VALIDATION_IDS = ("synthetic-val-0", "synthetic-val-1")


def _synthetic_projection() -> dict[str, object]:
    projection: dict[str, object] = {
        "schema": split_authority.SCHEMA,
        "status": "synthetic_split_projection",
        "dataset_order": [DATASET],
        "model_selection_only": True,
        "parent_seen_official_train": True,
        "official_test_accessed": False,
        "split_reconstruction_performed": False,
        "datasets": {
            DATASET: {
                "dataset": DATASET,
                "canonical_split_sha256": "1" * 64,
                "counts": {
                    "official_train": 4,
                    "development_train": len(DEVELOPMENT_IDS),
                    "internal_validation": len(VALIDATION_IDS),
                },
                "ordered_id_sha256": {
                    "official_train_ids": cache.canonical_sha256(
                        list(DEVELOPMENT_IDS + VALIDATION_IDS)
                    ),
                    "development_train_ids": cache.canonical_sha256(
                        list(DEVELOPMENT_IDS)
                    ),
                    "internal_validation_ids": cache.canonical_sha256(
                        list(VALIDATION_IDS)
                    ),
                },
                "model_selection_only": True,
                "parent_seen_official_train": True,
                "official_test_accessed": False,
            }
        },
    }
    projection["projection_sha256"] = cache.canonical_sha256(projection)
    return projection


def _checkpoint(name: str, token: str) -> cache.CheckpointBinding:
    return cache.CheckpointBinding(
        path=f"/synthetic/{name}.pth.tar",
        bytes=100 + len(name),
        file_sha256=token * 64,
        state_sha256=token.upper().lower() * 64,
    )


def _arrays(offset: float = 0.0) -> dict[str, np.ndarray]:
    base = np.ascontiguousarray(
        np.array([[0.25, -0.5], [1.0, -1.5]], dtype=np.float32) + offset
    )
    delta = np.ascontiguousarray(
        np.array([[0.125, 0.25], [-0.5, 0.75]], dtype=np.float32)
    )
    routed = np.add(base, delta, dtype=np.float32)
    return {
        "base_logits": base,
        "delta_logits": delta,
        "routed_logits": np.ascontiguousarray(routed),
        "current_logits": base.copy(order="C"),
        "original_logits": np.ascontiguousarray(base - np.float32(0.1)),
        "target": np.ascontiguousarray(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        ),
    }


class PBDRV4InternalRawLogitCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        self._cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.projection = _synthetic_projection()
        self.v3 = _checkpoint("v3", "a")
        self.current = _checkpoint("current", "b")
        self.original = _checkpoint("original", "c")

    def tearDown(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self._matmul_tf32
        torch.backends.cudnn.allow_tf32 = self._cudnn_tf32

    def _writer(
        self,
        destination: Path,
        *,
        partition: str = "internal_validation",
        identifiers: tuple[str, ...] = VALIDATION_IDS,
    ) -> cache.InternalRawLogitCacheWriter:
        return cache.InternalRawLogitCacheWriter(
            destination,
            dataset_name=DATASET,
            parent_role=ROLE,
            partition=partition,
            split_projection=self.projection,
            ordered_sample_ids=identifiers,
            v3_checkpoint=self.v3,
            current_checkpoint=self.current,
            original_checkpoint=self.original,
            normalization={"mean": 10.0, "std": 2.0},
            metric_core_sha256="d" * 64,
            source_lock_sha256="e" * 64,
        )

    @staticmethod
    def _append(
        writer: cache.InternalRawLogitCacheWriter,
        sample_id: str,
        *,
        offset: float = 0.0,
        arrays: dict[str, np.ndarray] | None = None,
    ) -> None:
        writer.append_sample(
            sample_id=sample_id,
            height=2,
            width=2,
            **(_arrays(offset) if arrays is None else arrays),
        )

    def _commit(self, destination: Path) -> Path:
        with self._writer(destination) as writer:
            for index, sample_id in enumerate(VALIDATION_IDS):
                self._append(writer, sample_id, offset=float(index))
            return writer.finalize()

    def test_identity_binds_projection_checkpoints_runtime_and_scope(self) -> None:
        identity = cache.build_cache_identity(
            dataset_name=DATASET,
            parent_role=ROLE,
            partition="development_train",
            split_projection=self.projection,
            ordered_sample_ids=DEVELOPMENT_IDS,
            v3_checkpoint=self.v3,
            current_checkpoint=self.current,
            original_checkpoint=self.original,
            normalization={"mean": 10.0, "std": 2.0},
            metric_core_sha256="d" * 64,
            source_lock_sha256="e" * 64,
        )
        self.assertEqual(
            identity["split_projection_sha256"],
            self.projection["projection_sha256"],
        )
        self.assertEqual(identity["ordered_sample_ids"], list(DEVELOPMENT_IDS))
        self.assertEqual(identity["checkpoints"]["v3_candidate"], self.v3.as_dict())
        self.assertEqual(identity["checkpoints"]["current"], self.current.as_dict())
        self.assertEqual(identity["checkpoints"]["original"], self.original.as_dict())
        self.assertEqual(identity["normalization"], {"mean": 10.0, "std": 2.0})
        self.assertEqual(identity["metric_core_sha256"], "d" * 64)
        self.assertEqual(identity["source_lock_sha256"], "e" * 64)
        self.assertIs(identity["runtime"]["cuda_matmul_allow_tf32"], False)
        self.assertIs(identity["runtime"]["cudnn_allow_tf32"], False)
        self.assertIs(identity["official_test_accessed"], False)

    def test_only_two_internal_projection_partitions_are_accepted(self) -> None:
        self.assertEqual(
            cache.PARTITIONS,
            ("development_train", "internal_validation"),
        )
        with tempfile.TemporaryDirectory() as directory_text:
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "unsupported internal partition",
            ):
                self._writer(
                    Path(directory_text) / "cache",
                    partition="other",
                )

    def test_module_has_no_dataset_loader_or_split_builder_symbols(self) -> None:
        source = Path(cache.__file__).resolve().read_text(encoding="utf-8")
        self.assertNotIn("load_index", source)
        self.assertNotIn("stratified_split", source)
        self.assertNotIn("img_idx", source)

    def test_projection_order_duplicates_missing_and_extra_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "duplicate IDs",
            ):
                self._writer(
                    root / "duplicate",
                    identifiers=(VALIDATION_IDS[0], VALIDATION_IDS[0]),
                )
            writer = self._writer(root / "strict-order")
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "sample order differs",
            ):
                self._append(writer, VALIDATION_IDS[1])
            self._append(writer, VALIDATION_IDS[0])
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "sample order differs",
            ):
                self._append(writer, VALIDATION_IDS[0])
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "missing projected samples",
            ):
                writer.finalize()
            self._append(writer, VALIDATION_IDS[1])
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "extra sample",
            ):
                self._append(writer, "synthetic-extra")
            writer.abort()

    def test_exact_residual_and_current_identities_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            wrong_routed = _arrays()
            wrong_routed["routed_logits"] = wrong_routed["routed_logits"].copy()
            wrong_routed["routed_logits"][0, 0] += np.float32(1.0e-3)
            writer = self._writer(root / "wrong-routed")
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "exact FP32 base-plus-delta",
            ):
                self._append(writer, VALIDATION_IDS[0], arrays=wrong_routed)
            writer.abort()

            wrong_current = _arrays()
            wrong_current["current_logits"] = wrong_current["current_logits"].copy()
            wrong_current["current_logits"][0, 0] += np.float32(1.0e-3)
            writer = self._writer(root / "wrong-current")
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "bitwise equal to base",
            ):
                self._append(writer, VALIDATION_IDS[0], arrays=wrong_current)
            writer.abort()

    def test_arrays_must_be_fp32_c_contiguous_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            wrong_dtype = _arrays()
            wrong_dtype["original_logits"] = wrong_dtype["original_logits"].astype(
                np.float64
            )
            writer = self._writer(root / "dtype")
            with self.assertRaisesRegex(cache.PBDRV4InternalCacheError, "must be FP32"):
                self._append(writer, VALIDATION_IDS[0], arrays=wrong_dtype)
            writer.abort()

            noncontiguous = _arrays()
            backing = np.zeros((2, 4), dtype=np.float32)
            noncontiguous["original_logits"] = backing[:, ::2]
            writer = self._writer(root / "layout")
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "must be C-contiguous",
            ):
                self._append(writer, VALIDATION_IDS[0], arrays=noncontiguous)
            writer.abort()

            nonfinite = _arrays()
            nonfinite["original_logits"] = nonfinite["original_logits"].copy()
            nonfinite["original_logits"][0, 0] = np.inf
            writer = self._writer(root / "finite")
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "non-finite",
            ):
                self._append(writer, VALIDATION_IDS[0], arrays=nonfinite)
            writer.abort()

    def test_commit_and_read_fully_validate_synthetic_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            destination = self._commit(Path(directory_text) / "cache")
            validated = cache.read_cache(
                destination,
                split_projection=self.projection,
            )
            self.assertEqual(
                tuple(sample.sample_id for sample in validated.samples),
                VALIDATION_IDS,
            )
            self.assertEqual(validated.manifest["sample_count"], len(VALIDATION_IDS))
            for sample in validated.samples:
                self.assertEqual(set(sample.arrays), set(cache.TENSOR_FIELDS))
                for value in sample.arrays.values():
                    self.assertEqual(value.dtype, np.float32)
                    self.assertTrue(value.flags.c_contiguous)
                    self.assertFalse(value.flags.writeable)

    def test_existing_or_symlink_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            destination = self._commit(root / "cache")
            manifest_before = (destination / cache.MANIFEST_NAME).read_bytes()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                self._writer(destination)
            self.assertEqual(
                (destination / cache.MANIFEST_NAME).read_bytes(),
                manifest_before,
            )

            target = root / "target"
            target.mkdir()
            sentinel = target / "sentinel"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            symlink = root / "symlink-cache"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                self._writer(symlink)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")

    def test_reader_rejects_changed_archive_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            destination = self._commit(Path(directory_text) / "cache")
            sample = destination / cache.SAMPLES_DIRECTORY / "00000000.npz"
            with sample.open("r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                byte = handle.read(1)
                handle.seek(-1, os.SEEK_END)
                handle.write(bytes([byte[0] ^ 0x01]))
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "sample file SHA-256 differs",
            ):
                cache.read_cache(destination, split_projection=self.projection)

    def test_reader_rejects_self_consistent_wrong_tensor_semantic_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            destination = self._commit(Path(directory_text) / "cache")
            manifest_path = destination / cache.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = manifest["samples"][0]
            record["tensors"]["base_logits"]["semantic_sha256"] = "0" * 64
            unsigned_record = dict(record)
            unsigned_record.pop("sample_semantic_sha256")
            record["sample_semantic_sha256"] = cache.canonical_sha256(unsigned_record)
            unsigned_manifest = dict(manifest)
            unsigned_manifest.pop("manifest_sha256")
            manifest["manifest_sha256"] = cache.canonical_sha256(unsigned_manifest)
            manifest_path.write_bytes(
                cache.canonical_json_bytes(manifest, trailing_newline=True)
            )

            commit_path = destination / cache.COMMIT_NAME
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            commit["manifest_bytes"] = manifest_path.stat().st_size
            commit["manifest_file_sha256"] = cache.file_sha256(manifest_path)
            commit["manifest_sha256"] = manifest["manifest_sha256"]
            unsigned_commit = dict(commit)
            unsigned_commit.pop("commit_sha256")
            commit["commit_sha256"] = cache.canonical_sha256(unsigned_commit)
            commit_path.write_bytes(cache.canonical_json_bytes(commit, trailing_newline=True))

            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "base_logits semantic metadata differs",
            ):
                cache.read_cache(destination, split_projection=self.projection)

    def test_reader_rejects_symlink_sample_even_with_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            destination = self._commit(Path(directory_text) / "cache")
            sample = destination / cache.SAMPLES_DIRECTORY / "00000000.npz"
            backup = destination.parent / "sample-backup.npz"
            sample.rename(backup)
            sample.symlink_to(backup)
            with self.assertRaisesRegex(
                cache.PBDRV4InternalCacheError,
                "sample must be a regular non-symlink file",
            ):
                cache.read_cache(destination, split_projection=self.projection)


if __name__ == "__main__":
    unittest.main()
