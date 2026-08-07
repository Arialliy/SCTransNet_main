from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import prepare_pbdr_v4_internal_artifacts as subject
from experiments import pbdr_v4_internal_cache as cache_io
from experiments import pbdr_v4_split_authority as split_authority


torch.set_num_threads(1)

DATASET = "NUAA-SIRST"
ROLE = "best_miou"
DEV_IDS = ("dev_0", "dev_1")
VAL_IDS = ("val_0",)
OFFICIAL_IDS = DEV_IDS + VAL_IDS
CANONICAL_SPLIT_SHA = "a" * 64
METRIC_SHA = "b" * 64
SOURCE_LOCK_SHA = "c" * 64


def _projection(source_path: Path | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "dataset": DATASET,
        "canonical_split_sha256": CANONICAL_SPLIT_SHA,
        "counts": {
            "official_train": len(OFFICIAL_IDS),
            "development_train": len(DEV_IDS),
            "internal_validation": len(VAL_IDS),
        },
        "ordered_id_sha256": {
            "official_train_ids": split_authority.ordered_ids_sha256(OFFICIAL_IDS),
            "development_train_ids": split_authority.ordered_ids_sha256(DEV_IDS),
            "internal_validation_ids": split_authority.ordered_ids_sha256(VAL_IDS),
        },
        "model_selection_only": True,
        "parent_seen_official_train": True,
        "official_test_accessed": False,
    }
    if source_path is not None:
        record.update(
            {
                "source_path": str(source_path),
                "source_bytes": source_path.stat().st_size,
                "source_file_sha256": split_authority.file_sha256(source_path),
            }
        )
    projection: dict[str, object] = {
        "schema": split_authority.SCHEMA,
        "status": "synthetic",
        "source_policy": "read_only_existing_v3_split_manifests",
        "dataset_order": [DATASET],
        "model_selection_only": True,
        "parent_seen_official_train": True,
        "official_test_accessed": False,
        "split_reconstruction_performed": False,
        "datasets": {DATASET: record},
    }
    projection["projection_sha256"] = split_authority.canonical_sha256(projection)
    return projection


class FakeProjectionDataset:
    def __init__(self, identifiers: tuple[str, ...]) -> None:
        self.sample_ids = identifiers
        self.normalization = {"mean": 10.0, "std": 2.0}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        identifier = self.sample_ids[index]
        offset = float(OFFICIAL_IDS.index(identifier))
        image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) + offset
        target = torch.zeros(1, 4, 4, dtype=torch.float32)
        target[0, 0, 0] = 1.0
        if index % 2:
            target[0, 1, 1] = 1.0
        return image, target, (2, 3), identifier


class FakeV3(nn.Module):
    def __init__(self, counters: dict[str, int]) -> None:
        super().__init__()
        self.counters = counters
        self.mode = "test"

    def forward_for_pbdr_v3_training(self, image: torch.Tensor):
        self.counters["v3"] += 1
        base = image + np.float32(0.25)
        delta = torch.full_like(base, np.float32(-0.125))
        routed = torch.add(base, delta)
        auxiliary = SimpleNamespace(
            base_logits=base,
            routed_logits=routed,
            routing=SimpleNamespace(delta_logits=delta),
        )
        return (), auxiliary


class FakeCurrent(nn.Module):
    def __init__(self, counters: dict[str, int], *, mismatch: bool = False) -> None:
        super().__init__()
        self.counters = counters
        self.mismatch = mismatch
        self.mode = "test"

    def forward_for_pbdr_v4_training(self, image: torch.Tensor):
        self.counters["current"] += 1
        logits = image + np.float32(0.25)
        if self.mismatch:
            logits = logits.clone()
            logits[..., 0, 0] += np.float32(0.01)
        return (), SimpleNamespace(candidate_base_logits=logits)


class FakeOriginal(nn.Module):
    def __init__(self, counters: dict[str, int]) -> None:
        super().__init__()
        self.counters = counters
        self.outc = nn.Identity()
        self.mode = "test"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.counters["original"] += 1
        raw = self.outc(image - np.float32(0.75))
        return torch.sigmoid(raw)


def _binding(name: str, token: str) -> cache_io.CheckpointBinding:
    return cache_io.CheckpointBinding(
        path=f"/synthetic/{name}.pth.tar",
        bytes=100 + len(name),
        file_sha256=token * 64,
        state_sha256=token * 64,
    )


def _bundle(
    counters: dict[str, int],
    data_root: Path,
    *,
    mismatch: bool = False,
) -> subject.StrictModelBundle:
    return subject.StrictModelBundle(
        v3_candidate=FakeV3(counters),
        current=FakeCurrent(counters, mismatch=mismatch),
        original=FakeOriginal(counters),
        v3_checkpoint=_binding("v3", "1"),
        current_checkpoint=_binding("current", "2"),
        original_checkpoint=_binding("original", "3"),
        data_root=data_root,
        candidate_split_sha256=CANONICAL_SPLIT_SHA,
        attestations={
            "v3_strict_load": True,
            "current_strict_load": True,
            "original_strict_load": True,
            "v3_base_bitwise_current_state": True,
            "current_base_logits_from_current": True,
            "official_test_accessed": False,
        },
    )


def _source(lock_path: Path) -> subject.SourceContext:
    return subject.SourceContext(
        path=lock_path,
        bytes=lock_path.stat().st_size,
        file_sha256=cache_io.file_sha256(lock_path),
        source_lock_sha256=SOURCE_LOCK_SHA,
        metric_source_sha256=METRIC_SHA,
        matcher_source_sha256=subject.atlas_dataset.matcher_source_sha256(),
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class PreparePBDRV4InternalArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def tearDown(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self.matmul_tf32
        torch.backends.cudnn.allow_tf32 = self.cudnn_tf32

    def test_projection_ids_are_read_only_from_validated_official_train_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            manifest_path = root / "split_manifest.json"
            payload = {
                "schema": "synthetic",
                "dataset": DATASET,
                "source_split": "official_train_only",
                "official_test_index_opened": False,
                "official_train_ids": list(OFFICIAL_IDS),
                "development_train_ids": list(DEV_IDS),
                "internal_validation_ids": list(VAL_IDS),
                "split_sha256": CANONICAL_SPLIT_SHA,
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            projection = _projection(manifest_path)
            validated = {
                "dataset": DATASET,
                "canonical_split_sha256": CANONICAL_SPLIT_SHA,
                "counts": projection["datasets"][DATASET]["counts"],
                "ordered_id_sha256": projection["datasets"][DATASET][
                    "ordered_id_sha256"
                ],
            }
            with mock.patch.object(
                subject.split_authority,
                "source_manifest_path",
                return_value=manifest_path,
            ), mock.patch.object(
                subject.split_authority,
                "validate_split_payload",
                return_value=validated,
            ) as validate:
                identifiers = subject.load_projection_ids(DATASET, projection)
            self.assertEqual(identifiers.official_train, OFFICIAL_IDS)
            self.assertEqual(identifiers.development_train, DEV_IDS)
            self.assertEqual(identifiers.internal_validation, VAL_IDS)
            validate.assert_called_once()
            self.assertIs(validate.call_args.args[1]["official_test_index_opened"], False)

    def test_source_context_binds_one_lock_to_live_metric_and_matcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            lock_path = Path(directory_text) / "source_lock.json"
            lock_path.write_text("fixture\n", encoding="utf-8")
            metric_path = Path(subject.metric_core.__file__).resolve()
            metric_sha = subject.source_lock_io.file_sha256(metric_path)
            matcher_sha = subject.atlas_dataset.matcher_source_sha256()
            payload = {
                "source_lock_sha256": "d" * 64,
                "sources": {
                    "experiments/pbdr_v4_metric_core.py": {"sha256": metric_sha},
                    "experiments/component_matching_v2.py": {"sha256": matcher_sha},
                },
            }
            with mock.patch.object(
                subject.source_lock_io,
                "load_source_lock",
                return_value=payload,
            ) as loader:
                context = subject.load_source_context(lock_path)
            loader.assert_called_once_with(lock_path, check_environment=True)
            self.assertEqual(context.metric_source_sha256, metric_sha)
            self.assertEqual(context.matcher_source_sha256, matcher_sha)
            self.assertEqual(context.source_lock_sha256, "d" * 64)

            bad = json.loads(json.dumps(payload))
            bad["sources"]["experiments/pbdr_v4_metric_core.py"]["sha256"] = "e" * 64
            with mock.patch.object(
                subject.source_lock_io,
                "load_source_lock",
                return_value=bad,
            ), self.assertRaisesRegex(
                subject.PBDRV4InternalPreparationError,
                "metric_core.py.*SHA differs",
            ):
                subject.load_source_context(lock_path)

    def test_single_partition_forward_is_once_raw_and_bitwise_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            lock_path = root / "lock.json"
            lock_path.write_text("lock\n", encoding="utf-8")
            counters = {"v3": 0, "current": 0, "original": 0}
            bundle = _bundle(counters, root)
            projection = _projection()
            dataset = FakeProjectionDataset(DEV_IDS)
            destination = root / "cache"
            subject.write_partition_cache(
                destination,
                dataset_name=DATASET,
                role=ROLE,
                partition="development_train",
                dataset=dataset,
                split_projection=projection,
                models=bundle,
                device=torch.device("cpu"),
                source=_source(lock_path),
            )
            self.assertEqual(counters, {"v3": 2, "current": 2, "original": 2})
            validated = cache_io.read_cache(
                destination,
                split_projection=projection,
            )
            first = validated.samples[0].arrays
            expected_image = np.arange(16, dtype=np.float32).reshape(4, 4)[:2, :3]
            np.testing.assert_array_equal(first["base_logits"], expected_image + 0.25)
            np.testing.assert_array_equal(first["current_logits"], first["base_logits"])
            np.testing.assert_array_equal(first["original_logits"], expected_image - 0.75)
            np.testing.assert_array_equal(
                first["routed_logits"],
                np.add(first["base_logits"], first["delta_logits"], dtype=np.float32),
            )

    def test_current_mismatch_fails_before_original_and_leaves_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            lock_path = root / "lock.json"
            lock_path.write_text("lock\n", encoding="utf-8")
            counters = {"v3": 0, "current": 0, "original": 0}
            with self.assertRaisesRegex(
                subject.PBDRV4InternalPreparationError,
                "not bitwise Current",
            ):
                subject.write_partition_cache(
                    root / "cache",
                    dataset_name=DATASET,
                    role=ROLE,
                    partition="development_train",
                    dataset=FakeProjectionDataset(DEV_IDS),
                    split_projection=_projection(),
                    models=_bundle(counters, root, mismatch=True),
                    device=torch.device("cpu"),
                    source=_source(lock_path),
                )
            self.assertEqual(counters, {"v3": 1, "current": 1, "original": 0})
            self.assertFalse((root / "cache").exists())

    def test_prepare_builds_two_caches_and_atlas_then_replays_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            data_root = root / "data"
            data_root.mkdir()
            lock_path = root / "source_lock.json"
            lock_path.write_text("lock\n", encoding="utf-8")
            output = root / "artifacts"
            projection = _projection()
            identifiers = subject.ProjectionIds(
                official_train=OFFICIAL_IDS,
                development_train=DEV_IDS,
                internal_validation=VAL_IDS,
            )
            counters = {"v3": 0, "current": 0, "original": 0}
            bundle = _bundle(counters, data_root)
            source = _source(lock_path)

            def dataset_builder(**arguments: object) -> FakeProjectionDataset:
                partition = arguments["partition"]
                return FakeProjectionDataset(
                    DEV_IDS if partition == "development_train" else VAL_IDS
                )

            patches = (
                mock.patch.object(subject, "load_source_context", return_value=source),
                mock.patch.object(
                    subject,
                    "projection_and_ids",
                    return_value=(projection, identifiers),
                ),
                mock.patch.object(subject, "load_strict_models", return_value=bundle),
                mock.patch.object(subject, "build_partition_dataset", side_effect=dataset_builder),
            )
            for item in patches:
                item.start()
            try:
                result = subject.prepare_internal_artifacts(
                    dataset_name=DATASET,
                    role=ROLE,
                    output_root=output,
                    source_lock_path=lock_path,
                    device_name="cpu",
                )
                self.assertFalse(result.replayed)
                self.assertEqual(counters, {"v3": 3, "current": 3, "original": 3})
                development = cache_io.read_cache(
                    result.development_cache,
                    split_projection=projection,
                )
                validation = cache_io.read_cache(
                    result.validation_cache,
                    split_projection=projection,
                )
                atlas = subject.atlas_builder.validate_component_atlas_artifact(
                    result.atlas
                )
                self.assertEqual(len(development.samples), len(DEV_IDS))
                self.assertEqual(len(validation.samples), len(VAL_IDS))
                self.assertEqual(len(atlas["samples"]), len(DEV_IDS))
                self.assertEqual(
                    development.manifest["identity"]["source_lock_sha256"],
                    SOURCE_LOCK_SHA,
                )
                self.assertEqual(
                    validation.manifest["identity"]["metric_core_sha256"],
                    METRIC_SHA,
                )
                self.assertEqual(atlas["source_lock_sha256"], SOURCE_LOCK_SHA)
                self.assertEqual(
                    atlas["matcher_source_sha256"], source.matcher_source_sha256
                )
                self.assertEqual(
                    atlas["parent_state_sha256"], bundle.current_checkpoint.state_sha256
                )

                before = _tree_bytes(output)
                replayed = subject.replay_internal_artifacts(
                    dataset_name=DATASET,
                    role=ROLE,
                    output_root=output,
                    source_lock_path=lock_path,
                )
                self.assertTrue(replayed.replayed)
                self.assertEqual(before, _tree_bytes(output))
                self.assertEqual(counters, {"v3": 3, "current": 3, "original": 3})
                with self.assertRaises(FileExistsError):
                    subject.prepare_internal_artifacts(
                        dataset_name=DATASET,
                        role=ROLE,
                        output_root=output,
                        source_lock_path=lock_path,
                        device_name="cpu",
                    )
                self.assertEqual(before, _tree_bytes(output))
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_cli_is_single_dataset_role_and_source_avoids_index_loading(self) -> None:
        args = subject.parse_args(
            [
                "--dataset",
                "IRSTD-1K",
                "--role",
                "best_pd",
                "--output-root",
                "/tmp/artifacts",
                "--source-lock",
                "/tmp/source-lock.json",
                "--device",
                "cpu",
                "--replay-existing",
            ]
        )
        self.assertEqual((args.dataset, args.role), ("IRSTD-1K", "best_pd"))
        self.assertTrue(args.replay_existing)
        self.assertEqual(set(subject.V3_RUN_DIRECTORIES), set(subject.DATASETS))
        self.assertTrue(
            all(
                set(role_paths) == set(subject.ROLES)
                for role_paths in subject.V3_RUN_DIRECTORIES.values()
            )
        )
        source = inspect.getsource(subject)
        self.assertNotIn("load_" + "index(", source)
        self.assertNotIn("resolve_sample(" + "*", source)


if __name__ == "__main__":
    unittest.main()
