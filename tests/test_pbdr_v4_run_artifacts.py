from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from experiments.pbdr_v4_run_artifacts import (
    PBDRV4ArtifactError,
    RunIdentity,
    atomic_rolling_torch_save,
    checkpoint_payload,
    exclusive_json,
    exclusive_torch_save,
    file_sha256,
    load_torch_artifact,
    optimizer_group_signature,
    validate_checkpoint_payload,
)


def _identity(stage: str = "stage1") -> RunIdentity:
    return RunIdentity(
        dataset="NUDT-SIRST",
        role="best_pd",
        stage=stage,
        source_lock_sha256="a" * 64,
        split_projection_sha256="b" * 64,
        atlas_manifest_sha256="c" * 64,
        parent_checkpoint_sha256="d" * 64,
        parent_state_sha256="e" * 64,
        initialization_checkpoint_sha256=None if stage == "stage1" else "f" * 64,
    )


def _optimizer() -> torch.optim.Optimizer:
    parameters = [torch.nn.Parameter(torch.ones(2))]
    return torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)


def _payload(identity: RunIdentity | None = None) -> dict[str, object]:
    optimizer = _optimizer()
    return checkpoint_payload(
        identity=identity or _identity(),
        epoch=1,
        epochs=2,
        model_state={"weight": torch.arange(2, dtype=torch.float32)},
        optimizer_state=optimizer.state_dict(),
        rng_state={
            "python": (3, (), None),
            "numpy": ("MT19937", torch.zeros(4, dtype=torch.int64).numpy(), 0, 0, 0.0),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
        },
        selected={"epoch": 1, "checkpoint_sha256": "0" * 64},
        event={"epoch": 1},
    )


class PBDRV4RunArtifactTests(unittest.TestCase):
    def test_exclusive_commits_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch_path = root / "epoch.pth.tar"
            json_path = root / "selection.json"
            exclusive_torch_save(torch_path, _payload())
            first_sha = file_sha256(torch_path)
            with self.assertRaisesRegex(PBDRV4ArtifactError, "exists"):
                exclusive_torch_save(torch_path, _payload())
            self.assertEqual(file_sha256(torch_path), first_sha)
            exclusive_json(json_path, {"status": "complete"})
            first_bytes = json_path.read_bytes()
            with self.assertRaisesRegex(PBDRV4ArtifactError, "exists"):
                exclusive_json(json_path, {"status": "changed"})
            self.assertEqual(json_path.read_bytes(), first_bytes)

    def test_rolling_state_is_atomic_and_replaceable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rolling.pth.tar"
            atomic_rolling_torch_save(path, {"value": 1})
            first = file_sha256(path)
            atomic_rolling_torch_save(path, {"value": 2})
            self.assertNotEqual(file_sha256(path), first)
            self.assertEqual(load_torch_artifact(path)["value"], 2)

    def test_symlink_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"safe")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(PBDRV4ArtifactError, "exists"):
                exclusive_json(link, {"bad": True})
            with self.assertRaisesRegex(PBDRV4ArtifactError, "symlink"):
                atomic_rolling_torch_save(link, {"bad": True})
            self.assertEqual(target.read_bytes(), b"safe")

    def test_payload_replay_rejects_identity_state_optimizer_and_rng_changes(self) -> None:
        payload = _payload()
        optimizer = _optimizer()
        signature = optimizer_group_signature(optimizer.state_dict())
        validate_checkpoint_payload(
            payload,
            identity=_identity(),
            epochs=2,
            expected_optimizer_group_signature=signature,
        )

        changed = dict(payload)
        changed["identity"] = _identity("stage2").as_dict()
        with self.assertRaisesRegex(PBDRV4ArtifactError, "identity"):
            validate_checkpoint_payload(changed, identity=_identity(), epochs=2)

        changed = dict(payload)
        changed["state_dict"] = {"weight": torch.ones(2)}
        with self.assertRaisesRegex(PBDRV4ArtifactError, "state SHA"):
            validate_checkpoint_payload(changed, identity=_identity(), epochs=2)

        changed = dict(payload)
        changed["optimizer_group_signature"] = []
        with self.assertRaisesRegex(PBDRV4ArtifactError, "optimizer"):
            validate_checkpoint_payload(changed, identity=_identity(), epochs=2)

        changed = dict(payload)
        changed["rng_state"] = {}
        with self.assertRaisesRegex(PBDRV4ArtifactError, "RNG"):
            validate_checkpoint_payload(changed, identity=_identity(), epochs=2)

    def test_stage2_requires_initialization_checkpoint_binding(self) -> None:
        with self.assertRaisesRegex(PBDRV4ArtifactError, "initialization"):
            RunIdentity(
                dataset="NUDT-SIRST",
                role="best_pd",
                stage="stage2",
                source_lock_sha256="a" * 64,
                split_projection_sha256="b" * 64,
                atlas_manifest_sha256="c" * 64,
                parent_checkpoint_sha256="d" * 64,
                parent_state_sha256="e" * 64,
                initialization_checkpoint_sha256=None,
            )


if __name__ == "__main__":
    unittest.main()
