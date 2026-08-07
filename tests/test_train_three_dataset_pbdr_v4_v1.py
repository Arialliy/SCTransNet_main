from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from experiments import pbdr_v4_split_authority as split_authority
from experiments import train_three_dataset_pbdr_v4_v1 as trainer
from experiments.pbdr_v4_run_artifacts import (
    RunIdentity,
    checkpoint_payload,
    epoch_checkpoint_path,
    exclusive_torch_save,
    file_sha256,
    optimizer_group_signature,
)
from experiments.pbdr_v4_state_contract import state_semantic_sha256
from experiments.pbdr_v4_training_core import (
    capture_rng_state,
    checkpoint_epoch_key,
)


class _ValidationDataset(Dataset):
    def __init__(self) -> None:
        self.values = (
            (
                torch.zeros(1, 2, 2),
                torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                (2, 2),
                "a",
            ),
            (
                torch.zeros(1, 2, 2),
                torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]),
                (2, 2),
                "b",
            ),
        )

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        return self.values[index]


@dataclass
class _Aux:
    routed_logits: torch.Tensor
    candidate_base_logits: torch.Tensor
    delta_logits: torch.Tensor


class _ValidationModel(torch.nn.Module):
    def forward_for_pbdr_v4_training(self, image: torch.Tensor):
        logits = torch.tensor(
            [[[[1.0, -1.0], [-1.0, 1.0]]]],
            dtype=image.dtype,
            device=image.device,
        )
        base = torch.zeros_like(logits)
        return (), _Aux(logits, base, logits - base)


class TrainThreeDatasetPBDRV4Tests(unittest.TestCase):
    def test_split_projection_and_source_ids_replay_without_test_access(self) -> None:
        projection = split_authority.build_projection()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.json"
            path.write_text(
                json.dumps(projection, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            observed = trainer.load_live_split_projection(path)
        official, development, validation = trainer.load_official_train_source_ids(
            observed, "NUDT-SIRST"
        )
        self.assertEqual((len(official), len(development), len(validation)), (663, 530, 133))
        self.assertFalse(set(development) & set(validation))
        self.assertEqual(set(development) | set(validation), set(official))

    def test_validation_is_one_ordered_fixed_half_pass(self) -> None:
        loader = DataLoader(_ValidationDataset(), batch_size=1, shuffle=False)
        metrics, diagnostics = trainer.validate_candidate(
            _ValidationModel(), loader, device=torch.device("cpu")
        )
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["matched_target_count"], 2)
        self.assertEqual(metrics["target_count"], 2)
        self.assertEqual(metrics["intersection_pixels"], 2)
        self.assertEqual(metrics["union_pixels"], 4)
        self.assertEqual(metrics["miou"], 0.5)
        self.assertEqual(diagnostics["threshold_up_crossings"], 4)
        self.assertEqual(diagnostics["threshold_down_crossings"], 0)
        self.assertGreater(diagnostics["positive_delta_count"], 0)
        self.assertEqual(
            [item["sample_id"] for item in diagnostics["component_transitions"]],
            ["a", "b"],
        )
        self.assertTrue(
            all(item["target_component_peaks"] for item in diagnostics["component_transitions"])
        )

    def test_training_diagnostics_reports_atlas_regions_and_crossings(self) -> None:
        diagnostics = trainer.TrainingDiagnostics()
        base = torch.tensor([[[[-0.1, 0.1], [0.0, 0.0]]]])
        delta = torch.tensor([[[[0.2, -0.2], [0.0, 0.1]]]])
        routed = base + delta
        maps = {
            "rescue": torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.int32),
            "suppress": torch.tensor([[[[0, 2], [0, 0]]]], dtype=torch.int32),
            "preserve": torch.tensor([[[[0, 0], [3, 0]]]], dtype=torch.int32),
        }
        diagnostics.update(
            total=torch.tensor(1.0),
            l2sp=torch.tensor(0.25),
            loss_components={"bce": 0.5},
            base=base,
            routed=routed,
            delta=delta,
            component_maps=maps,
        )
        result = diagnostics.compute()
        self.assertEqual(result["threshold_up_crossings"], 2)
        self.assertEqual(result["threshold_down_crossings"], 1)
        self.assertEqual(result["atlas_regions"]["rescue"]["pixel_count"], 1)
        self.assertAlmostEqual(result["atlas_regions"]["rescue"]["delta_mean"], 0.2)
        self.assertAlmostEqual(result["mean_loss_components"]["bce"], 0.5)

    def test_cli_requires_all_frozen_artifact_bindings(self) -> None:
        with self.assertRaises(SystemExit):
            trainer.parse_args([])
        args = trainer.parse_args(
            [
                "--dataset", "NUDT-SIRST",
                "--role", "best_pd",
                "--stage", "stage1",
                "--source-lock", "source.json",
                "--split-projection", "split.json",
                "--atlas-root", "atlas",
                "--run-dir", "run",
            ]
        )
        self.assertEqual(args.dataset, "NUDT-SIRST")
        self.assertEqual(args.role, "best_pd")
        self.assertEqual(args.stage, "stage1")

    def test_cpu_cannot_claim_a_cuda_uuid(self) -> None:
        self.assertIsNone(trainer.validate_cuda_uuid(torch.device("cpu"), None))
        with self.assertRaises(trainer.PBDRV4TrainerError):
            trainer.validate_cuda_uuid(
                torch.device("cpu"),
                "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
            )

    def test_trainer_has_no_official_test_loader_literal(self) -> None:
        source = Path(trainer.__file__).read_text(encoding="utf-8")
        self.assertNotIn('split="test"', source)
        self.assertNotIn("split='test'", source)
        self.assertNotIn("test.txt", source)

    def test_rolling_selection_binds_state_key_and_epoch_checkpoint_bytes(self) -> None:
        digest = "a" * 64
        identity = RunIdentity(
            dataset="NUDT-SIRST",
            role="best_pd",
            stage="stage1",
            source_lock_sha256=digest,
            split_projection_sha256="b" * 64,
            atlas_manifest_sha256="c" * 64,
            parent_checkpoint_sha256="d" * 64,
            parent_state_sha256="e" * 64,
            initialization_checkpoint_sha256=None,
        )
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        metrics = {
            "intersection_pixels": 1,
            "union_pixels": 2,
            "matched_target_count": 1,
            "target_count": 1,
            "unmatched_component_pixels": 0,
            "valid_pixel_count": 4,
            "matched_tiny_target_count": 1,
            "tiny_target_count": 1,
            "niou": 0.5,
            "test_loss": 0.25,
        }
        key = checkpoint_epoch_key("best_pd", metrics, 1)
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            artifact = checkpoint_payload(
                identity=identity,
                epoch=1,
                epochs=1,
                model_state=state,
                optimizer_state=optimizer.state_dict(),
                rng_state=capture_rng_state(),
                selected={
                    "epoch": 1,
                    "state_sha256": state_semantic_sha256(state),
                    "selection_key": trainer._json_role_key(key),
                },
                event={},
            )
            path = exclusive_torch_save(epoch_checkpoint_path(run_dir, 1), artifact)
            selected = {
                "epoch": 1,
                "metrics": metrics,
                "diagnostics": {},
                "selection_key": trainer._json_role_key(key),
                "selection_key_raw": key,
                "state_dict": state,
                "state_sha256": state_semantic_sha256(state),
                "epoch_checkpoint_path": str(path.resolve(strict=True)),
                "epoch_checkpoint_sha256": file_sha256(path),
            }
            replayed = trainer._validate_rolling_selection(
                selected,
                run_dir=run_dir,
                identity=identity,
                epochs=1,
                optimizer_signature=optimizer_group_signature(optimizer.state_dict()),
            )
            self.assertEqual(replayed["epoch"], 1)
            tampered = dict(selected)
            tampered_state = dict(state)
            tampered_state["weight"] = tampered_state["weight"] + 1.0
            tampered["state_dict"] = tampered_state
            with self.assertRaises(trainer.PBDRV4TrainerError):
                trainer._validate_rolling_selection(
                    tampered,
                    run_dir=run_dir,
                    identity=identity,
                    epochs=1,
                    optimizer_signature=optimizer_group_signature(optimizer.state_dict()),
                )

    def test_selected_candidate_partial_finalize_is_idempotent_and_bound(self) -> None:
        state = {"weight": torch.tensor([1.0])}
        payload = {
            "schema": "candidate",
            "state_dict": state,
            "state_sha256": state_semantic_sha256(state),
            "dataset": "NUDT-SIRST",
        }
        payload["candidate_manifest_sha256"] = trainer._candidate_manifest_sha(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.pth.tar"
            first = trainer._commit_or_replay_selected(path, payload)
            original_bytes = first.read_bytes()
            second = trainer._commit_or_replay_selected(path, payload)
            self.assertEqual(first, second)
            self.assertEqual(second.read_bytes(), original_bytes)
            changed = dict(payload)
            changed["dataset"] = "IRSTD-1K"
            changed["candidate_manifest_sha256"] = trainer._candidate_manifest_sha(changed)
            with self.assertRaises(trainer.PBDRV4TrainerError):
                trainer._commit_or_replay_selected(path, changed)


if __name__ == "__main__":
    unittest.main()
