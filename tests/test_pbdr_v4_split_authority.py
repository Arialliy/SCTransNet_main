from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import pbdr_v4_split_authority as authority


def _source_payload(dataset_name: str) -> dict[str, object]:
    path = authority.source_manifest_path(dataset_name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("source manifest is not an object")
    return value


class PBDRV4SplitAuthorityTests(unittest.TestCase):
    def test_registered_sources_are_exact_unique_v3_formal_manifests(self) -> None:
        self.assertEqual(tuple(authority.SOURCE_AUTHORITIES), authority.DATASETS)
        paths = [
            record.relative_path for record in authority.SOURCE_AUTHORITIES.values()
        ]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(paths), 3)
        self.assertEqual(
            paths,
            [
                (
                    "results/nuaa_pbdr_v3_stage1_v1/formal/best_miou/core/"
                    "split_manifest.json"
                ),
                (
                    "results/two_dataset_pbdr_v3_stage1_v1/runs/NUDT-SIRST/"
                    "formal/best_miou/core/split_manifest.json"
                ),
                (
                    "results/two_dataset_pbdr_v3_stage1_v1/runs/IRSTD-1K/"
                    "formal/best_miou/core/split_manifest.json"
                ),
            ],
        )

    def test_projection_binds_all_frozen_split_identities_and_scope_flags(
        self,
    ) -> None:
        projection = authority.build_projection()
        self.assertEqual(projection["schema"], authority.SCHEMA)
        self.assertEqual(projection["dataset_order"], list(authority.DATASETS))
        self.assertIs(projection["model_selection_only"], True)
        self.assertIs(projection["parent_seen_official_train"], True)
        self.assertIs(projection["official_test_accessed"], False)
        self.assertIs(projection["split_reconstruction_performed"], False)
        self.assertEqual(
            projection["projection_sha256"],
            authority.canonical_sha256(
                {
                    key: value
                    for key, value in projection.items()
                    if key != "projection_sha256"
                }
            ),
        )

        datasets = projection["datasets"]
        self.assertIsInstance(datasets, dict)
        for dataset_name in authority.DATASETS:
            expected = authority.SOURCE_AUTHORITIES[dataset_name]
            record = datasets[dataset_name]
            self.assertEqual(record["source_relative_path"], expected.relative_path)
            self.assertEqual(record["source_file_sha256"], expected.file_sha256)
            self.assertEqual(
                record["canonical_split_sha256"],
                expected.canonical_split_sha256,
            )
            self.assertEqual(
                record["counts"],
                {
                    "official_train": expected.official_train_count,
                    "development_train": expected.development_train_count,
                    "internal_validation": expected.internal_validation_count,
                },
            )
            self.assertEqual(
                record["ordered_id_sha256"],
                {
                    "official_train_ids": expected.official_train_ids_sha256,
                    "development_train_ids": (
                        expected.development_train_ids_sha256
                    ),
                    "internal_validation_ids": (
                        expected.internal_validation_ids_sha256
                    ),
                },
            )
            self.assertIs(record["model_selection_only"], True)
            self.assertIs(record["parent_seen_official_train"], True)
            self.assertIs(record["official_test_accessed"], False)

    def test_projection_and_json_are_deterministic(self) -> None:
        first = authority.build_projection()
        second = authority.build_projection()
        self.assertEqual(first, second)
        self.assertEqual(
            authority.canonical_json_bytes(first, trailing_newline=True),
            authority.canonical_json_bytes(second, trailing_newline=True),
        )

    def test_same_counts_but_different_partition_and_hash_are_rejected(
        self,
    ) -> None:
        dataset_name = "NUDT-SIRST"
        payload = copy.deepcopy(_source_payload(dataset_name))
        development = payload["development_train_ids"]
        validation = payload["internal_validation_ids"]
        self.assertIsInstance(development, list)
        self.assertIsInstance(validation, list)
        development[0], validation[0] = validation[0], development[0]
        unsigned = dict(payload)
        unsigned.pop("split_sha256")
        payload["split_sha256"] = authority.canonical_sha256(unsigned)

        frozen = authority.SOURCE_AUTHORITIES[dataset_name]
        self.assertEqual(len(development), frozen.development_train_count)
        self.assertEqual(len(validation), frozen.internal_validation_count)
        with self.assertRaisesRegex(
            authority.PBDRV4SplitAuthorityError,
            "canonical split SHA-256 differs",
        ):
            authority.validate_split_payload(dataset_name, payload)

    def test_changed_source_file_digest_is_rejected_before_projection(self) -> None:
        with mock.patch.object(authority, "file_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(
                authority.PBDRV4SplitAuthorityError,
                "source manifest file SHA-256 differs",
            ):
                authority.build_projection()

    def test_write_projection_is_exclusive_and_preserves_first_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            destination = Path(directory_text) / "projection.json"
            projection = authority.build_projection()
            written = authority.write_projection_once(destination, projection)
            first_bytes = written.read_bytes()
            self.assertEqual(
                first_bytes,
                authority.canonical_json_bytes(projection, trailing_newline=True),
            )
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                authority.write_projection_once(destination, projection)
            self.assertEqual(written.read_bytes(), first_bytes)

    def test_write_projection_rejects_a_self_consistent_but_wrong_payload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            destination = Path(directory_text) / "projection.json"
            projection = authority.build_projection()
            projection["official_test_accessed"] = True
            unsigned = dict(projection)
            unsigned.pop("projection_sha256")
            projection["projection_sha256"] = authority.canonical_sha256(unsigned)
            with self.assertRaisesRegex(
                authority.PBDRV4SplitAuthorityError,
                "differs from the live frozen authority",
            ):
                authority.write_projection_once(destination, projection)
            self.assertFalse(destination.exists())

    def test_write_projection_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            target = directory / "target.json"
            target.write_text("sentinel\n", encoding="utf-8")
            destination = directory / "projection.json"
            destination.symlink_to(target)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                authority.write_projection_once(destination)
            self.assertEqual(target.read_text(encoding="utf-8"), "sentinel\n")

    def test_module_has_no_dataset_loader_or_split_reconstruction_imports(
        self,
    ) -> None:
        source_path = Path(authority.__file__).resolve()
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertLessEqual(
            imported_roots,
            {
                "__future__",
                "argparse",
                "dataclasses",
                "hashlib",
                "json",
                "os",
                "pathlib",
                "typing",
            },
        )
        self.assertNotIn("load_index", source)
        self.assertNotIn("stratified_split", source)


if __name__ == "__main__":
    unittest.main()
