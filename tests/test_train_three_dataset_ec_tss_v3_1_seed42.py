from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from experiments import train_three_dataset_seed42_global_tss_v2 as positive
from experiments import train_three_dataset_ec_tss_v3_1_seed42 as trainer
from model.tpd_forward_contract import TPDForwardOutput


torch.set_num_threads(1)


def _formal_args(
    dataset: str = "NUAA-SIRST",
    *,
    gpu: str = "0",
    results_root: str | Path | None = None,
    pause: int | None = None,
) -> argparse.Namespace:
    argv = [
        "--dataset",
        dataset,
        "--method",
        "final",
        "--tss-weight",
        "0.005",
        "--physical-gpu-index",
        gpu,
        "--expected-gpu-uuid",
        trainer.GPU_UUIDS[gpu],
    ]
    if results_root is not None:
        argv.extend(["--results-root", str(results_root)])
    if pause is not None:
        argv.extend(["--pause-after-epoch", str(pause)])
    return trainer.parse_args(argv)


def _smoke_args(
    root: Path,
    *,
    epochs: int,
    resume: str,
    pause: int | None = None,
) -> argparse.Namespace:
    argv = [
        "--dataset",
        "NUAA-SIRST",
        "--method",
        "final",
        "--tss-weight",
        "0.005",
        "--results-root",
        str(root),
        "--smoke",
        "--device",
        "cpu",
        "--epochs",
        str(epochs),
        "--begin-test",
        "1",
        "--eval-every",
        "1",
        "--batch-size",
        "1",
        "--max-train-images",
        "2",
        "--max-test-images",
        "1",
        "--resume",
        resume,
    ]
    if pause is not None:
        argv.extend(["--pause-after-epoch", str(pause)])
    return trainer.parse_args(argv)


class ECTSSV31ArgumentTests(unittest.TestCase):
    def test_frozen_recipe_and_four_gpu_identity(self) -> None:
        self.assertEqual(set(trainer.GPU_UUIDS), {"0", "1", "2", "3"})
        for gpu in trainer.GPU_UUIDS:
            args = _formal_args(gpu=gpu, pause=200)
            trainer.validate_args(args)
            self.assertEqual(args.epochs, 1000)
            self.assertEqual(args.pause_after_epoch, 200)
            self.assertEqual(
                trainer.recipe_identity(args),
                {
                    "method": "final",
                    "recipe_id": "final_ec_tss_v3_1",
                    "objective_id": "ec_tss_v3_1",
                    "requested_tss_weight": 0.005,
                    "tss_lambda_token": "0p005",
                    "tss_ratio_cap": 0.10,
                    "confidence_threshold": 0.5,
                    "target_dilation_radius": 3,
                    "positive_normalization": "risk_mass_clamp_min_1",
                    "negative_normalization": "risk_mass_clamp_min_1",
                    "tss_enabled": True,
                    "survival_pos_weight_used": False,
                },
            )

    def test_formal_pause_does_not_shorten_schedule(self) -> None:
        args = _formal_args(pause=200)
        args.epochs = 200
        with self.assertRaisesRegex(ValueError, "formal epochs"):
            trainer.validate_args(args)
        args = _formal_args(pause=199)
        with self.assertRaisesRegex(ValueError, "formal pause epoch"):
            trainer.validate_args(args)

    def test_rejects_other_methods_weights_datasets_and_gpu_uuid(self) -> None:
        with self.assertRaises(SystemExit):
            trainer.parse_args(
                ["--dataset", "NUAA-SIRST", "--method", "original"]
            )
        with self.assertRaisesRegex(ValueError, "requires --tss-weight 0.005"):
            trainer.parse_args(
                [
                    "--dataset",
                    "NUAA-SIRST",
                    "--method",
                    "final",
                    "--tss-weight",
                    "0.01",
                ]
            )
        with self.assertRaises(SystemExit):
            trainer.parse_args(
                ["--dataset", "SIRST3", "--method", "final"]
            )
        args = _formal_args(gpu="1")
        args.expected_gpu_uuid = trainer.GPU_UUIDS["0"]
        with self.assertRaisesRegex(ValueError, "expected GPU UUID"):
            trainer.validate_args(args)

    def test_result_identity_is_disjoint_from_prior_runs(self) -> None:
        self.assertEqual(
            trainer.DEFAULT_RESULTS_ROOT,
            trainer.REPO_ROOT
            / "results"
            / "three_dataset_ec_tss_v3_1_seed42",
        )
        with tempfile.TemporaryDirectory() as directory:
            args = _formal_args(results_root=directory)
            path = trainer._run_directory(args)
            self.assertEqual(
                path.relative_to(Path(directory).resolve()).as_posix(),
                "runs/NUAA-SIRST/final_ec_tss_v3_1/seed_42",
            )


class ECTSSV31AdapterAndArtifactTests(unittest.TestCase):
    def tearDown(self) -> None:
        trainer._AUDIT.reset()

    @staticmethod
    def _structured_output() -> tuple[TPDForwardOutput, torch.Tensor]:
        target = torch.zeros(2, 1, 32, 32)
        probability = torch.full_like(target, 0.75, requires_grad=True)
        segmentation = tuple(probability for _ in range(6))
        logit1 = torch.zeros(2, 1, 2, 2, requires_grad=True)
        logit2 = torch.zeros(2, 1, 2, 2, requires_grad=True)
        return (
            TPDForwardOutput(
                segmentation=segmentation,
                emb1_survival_logits=logit1,
                emb2_survival_logits=logit2,
            ),
            target,
        )

    def test_adapter_absorbs_and_ignores_legacy_pos_weight(self) -> None:
        output, target = self._structured_output()
        first = trainer._compute_loss_ec_tss_v3_1(
            output,
            target,
            nn.BCELoss(),
            survival_weight=0.005,
            survival_pos_weight=1.0,
        )
        trainer._AUDIT.reset()
        second = trainer._compute_loss_ec_tss_v3_1(
            output,
            target,
            nn.BCELoss(),
            survival_weight=0.005,
            survival_pos_weight=987.0,
        )
        self.assertTrue(torch.equal(first.total, second.total))
        self.assertTrue(torch.equal(first.survival, second.survival))
        audit = trainer._AUDIT.payload()
        self.assertFalse(audit["train_ec_tss_survival_pos_weight_used"])
        self.assertIn("train_ec_tss_survival_mean", audit)
        self.assertIn("train_ec_tss_weighted_survival_mean", audit)
        self.assertIn(
            "train_ec_tss_raw_weighted_to_segmentation_ratio_mean", audit
        )
        self.assertIn(
            "train_ec_tss_effective_weighted_to_segmentation_ratio_mean",
            audit,
        )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            trainer._compute_loss_ec_tss_v3_1(
                output,
                target,
                nn.BCELoss(),
                survival_weight=0.005,
                survival_pos_weight=float("nan"),
            )

    def test_protocol_freezes_recipe_schedule_budget_and_sources(self) -> None:
        args = _formal_args(pause=200)
        args.smoke = True
        args.device = "cpu"
        args.epochs = 2
        args.begin_test = 1
        args.eval_every = 1
        args.batch_size = 1
        args.max_train_images = 2
        args.max_test_images = 1
        payload = trainer._protocol_payload(
            args,
            model_metadata={"formal_training_objective": {}},
            tss_metadata=trainer._validate_tss_statistics(args)[1],
            data_manifests={
                "files": {
                    "imgidx": {"path": "manifest.json", "sha256": "sha"}
                }
            },
            train_count=2,
            test_count=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(payload["schema"], trainer.SCHEMA)
        self.assertEqual(payload["recipe"]["recipe_id"], trainer.RECIPE_ID)
        self.assertEqual(payload["objective_id"], trainer.OBJECTIVE_ID)
        self.assertEqual(payload["planned_total_epochs"], 2)
        pause = payload["pause_resume_contract"]
        self.assertTrue(pause["pause_is_invocation_control_not_protocol_identity"])
        self.assertFalse(pause["pilot_creates_additional_run_identity"])
        training = payload["training"]
        self.assertFalse(training["survival_pos_weight_used"])
        self.assertEqual(training["confidence_threshold"], 0.5)
        self.assertEqual(training["target_dilation_radius"], 3)
        budget = payload["search_budget_disclosure"]
        self.assertEqual(budget["existing_formal1000_training_runs"], 15)
        self.assertEqual(budget["completed_formal1000_training_runs"], 18)
        self.assertEqual(
            budget["final_to_original_recipe_search_ratio_after_ec_tss"], 5.0
        )
        sources = payload["runtime_sources"]
        self.assertEqual(
            Path(sources["ec_tss_v3_1_protocol"]["path"]).name,
            "EC_TSS_V3_1_PROTOCOL.md",
        )
        self.assertEqual(
            sources["legacy_segmentation_base_loss"]["role"],
            "segmentation-base semantics only; not the EC-TSS objective",
        )
        self.assertFalse(
            sources["legacy_segmentation_base_loss"][
                "consumed_by_ec_tss_objective"
            ]
        )

    def test_resume_rejects_a_prior_recipe_state(self) -> None:
        args = _formal_args()
        args.resume = "required"
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_training_state.pth.tar"
            torch.save(
                {
                    "schema": "sctransnet_three_dataset_tss_off_seed42_v1/v1",
                    "dataset": args.dataset,
                    "method": "final",
                    "seed": 42,
                    "protocol_sha256": "sha",
                    "epoch": 200,
                    "event": {"epoch": 200},
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": trainer.engine.rng_state(),
                    "best_miou": {},
                    "best_pd": {},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "resume schema"):
                trainer._load_resume_ec_tss_v3_1(
                    args=args,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    protocol_sha256="sha",
                )


class ECTSSV31CpuRunTests(unittest.TestCase):
    class TinyTrain(Dataset):
        def __init__(self, dataset: str, **kwargs: object) -> None:
            self.normalization = positive.data_protocol.get_legacy_normalization(
                dataset
            )
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            image = torch.zeros(1, 32, 32)
            return image, torch.zeros_like(image)

    class TinyTest(Dataset):
        def __init__(
            self,
            train_dataset: str,
            test_dataset: str,
            **kwargs: object,
        ) -> None:
            self.normalization = positive.data_protocol.get_legacy_normalization(
                train_dataset
            )

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            image = torch.zeros(1, 32, 32)
            return image, torch.zeros_like(image), (32, 32), "mock_0"

    class TinyFinal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.segmentation_logit = nn.Parameter(torch.tensor(1.0))
            self.survival_logit1 = nn.Parameter(torch.tensor(0.0))
            self.survival_logit2 = nn.Parameter(torch.tensor(0.0))

        def forward(self, image: torch.Tensor) -> TPDForwardOutput:
            probability = torch.sigmoid(self.segmentation_logit) * torch.ones_like(
                image
            )
            batch, _, height, width = image.shape
            shape = (batch, 1, height // 16, width // 16)
            logit1 = self.survival_logit1 * torch.ones(
                shape, dtype=image.dtype, device=image.device
            )
            logit2 = self.survival_logit2 * torch.ones(
                shape, dtype=image.dtype, device=image.device
            )
            return TPDForwardOutput(
                segmentation=tuple(probability for _ in range(6)),
                emb1_survival_logits=logit1,
                emb2_survival_logits=logit2,
            )

    @staticmethod
    def _builder(method: str, seed: int, *, dataset_name: str):
        return ECTSSV31CpuRunTests.TinyFinal(), {
            "method": method,
            "seed": seed,
            "dataset_name": dataset_name,
            "formal_training_objective": {
                "objective_id": trainer.OBJECTIVE_ID,
            },
        }

    def tearDown(self) -> None:
        trainer._AUDIT.reset()

    @staticmethod
    def _events(summary: Mapping[str, object]) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in Path(str(summary["metrics"])).read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def _run_with_tiny_components(self, args: argparse.Namespace) -> Path:
        with mock.patch.object(
            trainer,
            "_import_runtime_components",
            return_value=(self._builder, self.TinyTrain, self.TinyTest),
        ):
            return trainer.run(args)

    def test_pause_is_durable_and_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _smoke_args(root, epochs=2, resume="never", pause=1)
            progress_path = self._run_with_tiny_components(args)
            self.assertEqual(progress_path.name, "progress.json")
            progress = trainer.validate_paused_run(
                trainer._run_directory(args), args.dataset, pause_epoch=1
            )
            self.assertEqual(progress["status"], "paused")
            self.assertEqual(progress["completed_epoch"], 1)
            self.assertEqual(progress["planned_total_epochs"], 2)
            self.assertEqual(progress["required_resume_mode"], "required")
            self.assertTrue(progress["diagnostics"])
            self.assertTrue(
                Path(progress["rolling_resume_state"]["path"]).is_file()
            )
            auto = _smoke_args(root, epochs=2, resume="auto")
            with self.assertRaisesRegex(ValueError, "requires --resume=required"):
                self._run_with_tiny_components(auto)

    def test_continuous_and_paused_resume_training_are_exact(self) -> None:
        def nested_equal(left: object, right: object, label: str) -> None:
            if isinstance(left, torch.Tensor):
                self.assertIsInstance(right, torch.Tensor, msg=label)
                self.assertTrue(torch.equal(left, right), msg=label)
                return
            if isinstance(left, np.ndarray):
                self.assertIsInstance(right, np.ndarray, msg=label)
                self.assertTrue(np.array_equal(left, right), msg=label)
                return
            if isinstance(left, dict):
                self.assertIsInstance(right, dict, msg=label)
                self.assertEqual(set(left), set(right), msg=label)
                for key in left:
                    nested_equal(left[key], right[key], f"{label}.{key}")
                return
            if isinstance(left, (list, tuple)):
                self.assertIsInstance(right, type(left), msg=label)
                self.assertEqual(len(left), len(right), msg=label)
                for index, (a, b) in enumerate(zip(left, right)):
                    nested_equal(a, b, f"{label}[{index}]")
                return
            self.assertEqual(left, right, msg=label)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous_args = _smoke_args(
                root / "continuous", epochs=2, resume="never"
            )
            continuous_path = self._run_with_tiny_components(continuous_args)
            continuous = json.loads(
                continuous_path.read_text(encoding="utf-8")
            )

            first_args = _smoke_args(
                root / "split", epochs=2, resume="never", pause=1
            )
            self._run_with_tiny_components(first_args)
            paused_protocol = json.loads(
                (
                    trainer._run_directory(first_args) / "protocol.json"
                ).read_text(encoding="utf-8")
            )["protocol_sha256"]
            resume_args = _smoke_args(
                root / "split", epochs=2, resume="required"
            )
            resumed_path = self._run_with_tiny_components(resume_args)
            resumed = json.loads(resumed_path.read_text(encoding="utf-8"))

            self.assertEqual(continuous["status"], "complete")
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(continuous["epochs"], 2)
            self.assertEqual(continuous["planned_total_epochs"], 2)
            self.assertEqual(resumed["epochs"], 2)
            self.assertEqual(resumed["planned_total_epochs"], 2)
            self.assertEqual(resumed["protocol_sha256"], paused_protocol)
            self.assertEqual(
                continuous["protocol_sha256"], resumed["protocol_sha256"]
            )
            continuous_events = self._events(continuous)
            resumed_events = self._events(resumed)
            self.assertEqual(len(continuous_events), 2)
            self.assertEqual(len(resumed_events), 2)
            for left, right in zip(continuous_events, resumed_events):
                left = copy.deepcopy(left)
                right = copy.deepcopy(right)
                left.pop("epoch_seconds", None)
                right.pop("epoch_seconds", None)
                nested_equal(left, right, "event")
            for role in trainer.CHECKPOINT_ROLES:
                left = torch.load(
                    continuous["checkpoints"][role]["path"],
                    map_location="cpu",
                    weights_only=False,
                )
                right = torch.load(
                    resumed["checkpoints"][role]["path"],
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(left["epoch"], right["epoch"])
                nested_equal(left["state_dict"], right["state_dict"], role)
                nested_equal(left["test_metrics"], right["test_metrics"], role)
            self.assertFalse(
                (
                    trainer._run_directory(resume_args)
                    / "resume"
                    / "latest_training_state.pth.tar"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
