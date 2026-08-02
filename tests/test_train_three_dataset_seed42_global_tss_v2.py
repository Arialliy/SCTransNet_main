from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset
from unittest import mock

from experiments import train_three_dataset_seed42_global_tss_v2 as trainer
from experiments import three_dataset_seed42_launch_v2 as launcher


def _formal_args(
    dataset: str = "NUAA-SIRST",
    method: str = "original",
    weight: float | None = None,
) -> argparse.Namespace:
    argv = [
        "--dataset",
        dataset,
        "--method",
        method,
        "--physical-gpu-index",
        "2",
        "--expected-gpu-uuid",
        trainer.GPU_UUIDS["2"],
    ]
    if weight is not None:
        argv.extend(["--tss-weight", str(weight)])
    return trainer.parse_args(argv)


class ArgumentContractTests(unittest.TestCase):
    def test_only_three_datasets_seed42_and_fixed_threshold(self) -> None:
        with self.assertRaises(SystemExit):
            trainer.parse_args(
                ["--dataset", "SIRST3", "--method", "original"]
            )
        args = _formal_args()
        trainer.validate_args(args)
        for field, value in (
            ("seed", 7),
            ("threshold", 0.51),
            ("epochs", 999),
            ("eval_every", 50),
        ):
            changed = copy.copy(args)
            setattr(changed, field, value)
            with self.assertRaises(ValueError):
                trainer.validate_args(changed)

    def test_formal_subset_limits_are_rejected(self) -> None:
        args = _formal_args()
        args.max_train_images = 2
        with self.assertRaises(ValueError):
            trainer.validate_args(args)
        args.max_train_images = None
        args.max_test_images = 2
        with self.assertRaises(ValueError):
            trainer.validate_args(args)

    def test_original_has_no_tss_and_final_accepts_only_frozen_lambdas(self) -> None:
        self.assertEqual(trainer.requested_tss_weight(_formal_args()), 0.0)
        for weight in trainer.TSS_LAMBDAS:
            args = _formal_args(method="final", weight=weight)
            trainer.validate_args(args)
            self.assertEqual(trainer.requested_tss_weight(args), weight)
        with self.assertRaises(ValueError):
            trainer.validate_args(_formal_args(method="final", weight=0.02))
        with self.assertRaises(ValueError):
            trainer.validate_args(_formal_args(method="final"))


class RecipeIdentityTests(unittest.TestCase):
    def test_lambda_run_directories_are_disjoint(self) -> None:
        paths = {
            trainer._run_directory(_formal_args(method="final", weight=weight))
            for weight in trainer.TSS_LAMBDAS
        }
        self.assertEqual(len(paths), 3)
        self.assertTrue(all("lambda_" in str(path) for path in paths))
        original = trainer._run_directory(_formal_args())
        self.assertNotIn(original, paths)

    def test_resume_protocol_hash_rejects_a_different_lambda(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        args = _formal_args(method="final", weight=0.0025)
        args.resume = "required"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pth.tar"
            payload = {
                "schema": trainer.SCHEMA,
                "epoch": 1,
                "dataset": args.dataset,
                "method": "final",
                "seed": 42,
                "protocol_sha256": "lambda-0p0025-hash",
                "recipe": trainer.recipe_identity(args),
                "requested_tss_weight": 0.0025,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": trainer.engine.rng_state(),
                "event": {
                    "epoch": 1,
                    "recipe": trainer.recipe_identity(args),
                },
                "best_miou": {},
                "best_pd": {},
            }
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                trainer._load_resume_v2(
                    args=args,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    protocol_sha256="lambda-0p005-hash",
                )

            payload["protocol_sha256"] = "lambda-0p0025-hash"
            payload["requested_tss_weight"] = 0.005
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                trainer._load_resume_v2(
                    args=args,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    protocol_sha256="lambda-0p0025-hash",
                )

    def test_checkpoint_payloads_explicitly_lock_lambda(self) -> None:
        args = _formal_args(method="final", weight=0.01)
        model = torch.nn.Linear(1, 1)
        payload = trainer._selected_checkpoint_payload(
            model=model,
            args=args,
            epoch=10,
            role="best_pd",
            metrics={"pd": 1.0},
            model_metadata={},
            protocol_sha256="sha",
        )
        self.assertEqual(payload["requested_tss_weight"], 0.01)
        self.assertEqual(payload["recipe"]["recipe_id"], "final_lambda_0p01")


class TssAuditTests(unittest.TestCase):
    @staticmethod
    def _loss(effective: float, segmentation: float, survival: float):
        scalar = lambda value: torch.tensor(float(value))
        weighted = effective * survival
        return trainer.TPDTrainingLoss(
            total=scalar(segmentation + weighted),
            segmentation=scalar(segmentation),
            survival=scalar(survival),
            segmentation_terms=(scalar(segmentation),),
            survival_terms=(scalar(survival),),
            effective_survival_weight=scalar(effective),
            weighted_survival=scalar(weighted),
        )

    def test_epoch_diagnostics_include_frozen_fields(self) -> None:
        audit = trainer._EpochTSSAudit()
        audit.configure("final", 0.01)
        audit.add(self._loss(0.002, 2.0, 10.0), 1)
        audit.add(self._loss(0.01, 2.0, 1.0), 3)
        payload = audit.payload()
        required = {
            "train_tss_effective_weight_mean",
            "train_tss_effective_weight_p10",
            "train_tss_effective_weight_p50",
            "train_tss_effective_weight_p90",
            "train_tss_effective_weight_std",
            "train_tss_effective_weight_max",
            "train_tss_raw_weighted_to_seg_ratio_mean",
            "train_tss_effective_weighted_to_seg_ratio_mean",
            "train_tss_cap_active_batch_fraction",
            "train_tss_cap_active_sample_fraction",
            "train_tss_batch_diagnostics",
        }
        self.assertTrue(required.issubset(payload))
        self.assertAlmostEqual(
            payload["train_tss_effective_weight_p10"], 0.002, places=8
        )
        self.assertAlmostEqual(
            payload["train_tss_effective_weight_p50"], 0.01, places=8
        )
        self.assertEqual(len(payload["train_tss_batch_diagnostics"]), 2)

    def test_statistics_with_wrong_train_identity_are_rejected(self) -> None:
        payload = launcher.build_tss_statistics_payload()
        payload["records"]["NUAA-SIRST"]["train_ids_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-tss.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            args = _formal_args(method="final", weight=0.005)
            args.tss_statistics = path
            with self.assertRaises(ValueError):
                trainer._validate_tss_statistics(args)


class CpuSmokeTests(unittest.TestCase):
    class TinyTrain(Dataset):
        def __init__(self, dataset: str, **kwargs: object) -> None:
            self.dataset = dataset
            self.normalization = trainer.data_protocol.get_legacy_normalization(
                dataset
            )
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            image = torch.zeros(1, 256, 256)
            mask = torch.zeros(1, 256, 256)
            return image, mask

    class TinyTest(Dataset):
        def __init__(self, train_dataset: str, test_dataset: str, **kwargs: object):
            self.normalization = trainer.data_protocol.get_legacy_normalization(
                train_dataset
            )

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            image = torch.zeros(1, 32, 32)
            mask = torch.zeros(1, 32, 32)
            return image, mask, (32, 32), "mock_0"

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(-1.0))

        def forward(self, image: torch.Tensor):
            probability = torch.sigmoid(self.logit) * torch.ones_like(image)
            return tuple(probability for _ in range(6))

    @staticmethod
    def _builder(method: str, seed: int, *, dataset_name: str):
        return CpuSmokeTests.TinyModel(), {
            "method": method,
            "seed": seed,
            "dataset_name": dataset_name,
        }

    def test_one_epoch_mock_cpu_smoke_writes_only_two_selected_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = trainer.parse_args(
                [
                    "--dataset",
                    "NUAA-SIRST",
                    "--method",
                    "original",
                    "--results-root",
                    directory,
                    "--smoke",
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
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
                ]
            )
            with mock.patch.object(
                trainer,
                "_import_runtime_components",
                return_value=(self._builder, self.TinyTrain, self.TinyTest),
            ):
                summary_path = trainer.run(args)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(
                set(summary["checkpoints"]), {"best_miou", "best_pd"}
            )
            self.assertEqual(summary["requested_tss_weight"], 0.0)
            self.assertFalse(
                (summary_path.parent / "resume" / "latest_training_state.pth.tar").exists()
            )


if __name__ == "__main__":
    unittest.main()
