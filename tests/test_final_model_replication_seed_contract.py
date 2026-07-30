from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments import final_model_child_initialization_manifest as child
from experiments import final_model_replication_seed_contract as seeds


class FinalModelReplicationSeedContractTests(unittest.TestCase):
    def test_deployment_hash_seed_uses_inference_artifact(self) -> None:
        self.assertEqual(
            seeds.hash_seed_from_deployment_artifact(
                seeds.DEPLOYMENT_INFERENCE_ARTIFACT_SHA256
            ),
            426780603,
        )
        self.assertNotEqual(
            seeds.hash_seed_from_deployment_artifact(
                "890c8cf0" + "0" * 56
            ),
            426780603,
        )

    def test_confirmatory_derivation_is_ordered_unique_and_excludes_history(
        self,
    ) -> None:
        digest = hashlib.sha256(b"frozen-source-lock").hexdigest()
        first = seeds.derive_confirmatory_seeds(digest, count=12)
        second = seeds.derive_confirmatory_seeds(digest, count=12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(set(first)), 12)
        self.assertFalse(
            set(first) & set(seeds.CONFIRMATORY_EXCLUDED_SEEDS)
        )

    def test_write_once_and_canonical_round_trip(self) -> None:
        digest = hashlib.sha256(b"source-lock").hexdigest()
        contract = seeds.ReplicationSeedScheduleContract.derive(digest)
        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "schedule.json"
            seeds.write_contract_once(path, contract)
            self.assertEqual(
                seeds.load_contract(path).normalized(),
                contract.normalized(),
            )
            seeds.write_contract_once(path, contract)
            changed = seeds.ReplicationSeedScheduleContract.derive(
                hashlib.sha256(b"different-lock").hexdigest()
            )
            with self.assertRaises(FileExistsError):
                seeds.write_contract_once(path, changed)

    def test_contract_rejects_manually_reordered_seed_sequence(self) -> None:
        digest = hashlib.sha256(b"source-lock").hexdigest()
        payload = seeds.ReplicationSeedScheduleContract.derive(
            digest
        ).normalized()
        payload["confirmatory_trajectory_seeds"] = list(
            reversed(payload["confirmatory_trajectory_seeds"])
        )
        with self.assertRaises(seeds.ReplicationSeedContractError):
            seeds.parse_contract(payload)

    def test_python_optimized_mode_does_not_disable_validation(self) -> None:
        # Contract validation must use explicit exceptions, never assert.
        digest = hashlib.sha256(b"source-lock").hexdigest()
        payload = seeds.ReplicationSeedScheduleContract.derive(
            digest
        ).normalized()
        payload["builder_compatibility_seed"] = 43
        with self.assertRaises(seeds.ReplicationSeedContractError):
            seeds.parse_contract(payload)


class FinalModelChildInitializationManifestTests(unittest.TestCase):
    def _manifest(
        self,
        directory: Path,
    ) -> child.ChildInitializationManifest:
        parent = directory / "parent.pth.tar"
        parent.write_bytes(b"immutable-parent-checkpoint")
        source_digest = hashlib.sha256(b"source-lock").hexdigest()
        seed_schedule = seeds.ReplicationSeedScheduleContract.derive(
            source_digest
        )
        schedule_path = directory / "schedule.json"
        seeds.write_contract_once(schedule_path, seed_schedule)
        return child.ChildInitializationManifest(
            arm="d",
            trajectory_seed=seeds.DEPLOYMENT_HASH_SEED,
            seed_contract_sha256=seeds.file_sha256(schedule_path),
            certification_source_lock_sha256=source_digest,
            parent_seed=42,
            parent_checkpoint_path=str(parent.resolve()),
            parent_checkpoint_sha256=child.file_sha256(parent),
            parent_state_dict_sha256=hashlib.sha256(
                b"parent-state"
            ).hexdigest(),
            parent_checkpoint_epoch=489,
        )

    def test_manifest_binds_exactly_one_parent_load_and_fresh_optimizer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            manifest = self._manifest(directory)
            payload = manifest.normalized()
            self.assertEqual(payload["parent_load_count"], 1)
            self.assertFalse(payload["optimizer_inherited"])
            self.assertFalse(payload["scheduler_inherited"])
            self.assertTrue(payload["all_child_parameters_trainable"])
            self.assertEqual(
                payload["initialization_scope"],
                "fixed_parent_child_trajectory",
            )

    def test_manifest_detects_parent_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            manifest = self._manifest(directory)
            path = directory / "child_init.json"
            child.write_manifest_once(path, manifest)
            parent_path = Path(manifest.parent_checkpoint_path)
            parent_path.write_bytes(b"changed-parent-checkpoint")
            with self.assertRaises(
                child.ChildInitializationManifestError
            ):
                child.load_manifest(path)

    def test_manifest_rejects_noncanonical_extra_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            manifest = self._manifest(directory)
            payload = manifest.normalized()
            payload["unexpected"] = True
            with self.assertRaises(
                child.ChildInitializationManifestError
            ):
                child.parse_manifest(payload)

    def test_manifest_rejects_parent_seed_unrelated_to_child_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            manifest = dataclasses.replace(
                self._manifest(directory),
                trajectory_seed=100,
                parent_seed=99,
            )
            with self.assertRaisesRegex(
                child.ChildInitializationManifestError,
                "parent seed must be",
            ):
                manifest.normalized()

    def test_manifest_file_is_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            manifest = self._manifest(directory)
            path = directory / "child_init.json"
            child.write_manifest_once(path, manifest)
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                path.read_bytes(),
                child.canonical_json_bytes(parsed),
            )


if __name__ == "__main__":
    unittest.main()
