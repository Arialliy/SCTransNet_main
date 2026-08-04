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
from experiments import train_three_dataset_tss_off_seed42_v1 as trainer
from model.tpd_forward_contract import TPDForwardOutput
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
)


torch.set_num_threads(1)


def _args(
    dataset: str = "NUAA-SIRST",
    *,
    weight: float | None = 0.0,
    results_root: str | Path | None = None,
) -> argparse.Namespace:
    argv = [
        "--dataset",
        dataset,
        "--method",
        "final",
        "--physical-gpu-index",
        "2",
        "--expected-gpu-uuid",
        trainer.GPU_UUIDS["2"],
    ]
    if weight is not None:
        argv.extend(["--tss-weight", str(weight)])
    if results_root is not None:
        argv.extend(["--results-root", str(results_root)])
    return trainer.parse_args(argv)


class TSSOffArgumentTests(unittest.TestCase):
    def test_recipe_is_one_exact_final_off_identity(self) -> None:
        for weight in (None, 0.0):
            args = _args(weight=weight)
            trainer.validate_args(args)
            self.assertEqual(trainer.requested_tss_weight(args), 0.0)
            self.assertEqual(
                trainer.recipe_identity(args),
                {
                    "method": "final",
                    "recipe_id": "final_tss_off",
                    "requested_tss_weight": 0.0,
                    "tss_lambda_token": "off",
                    "tss_ratio_cap": 0.10,
                    "tss_ratio_cap_applied": False,
                    "tss_enabled": False,
                    "tss_heads_registered": True,
                    "tss_training_forward_computes_logits": True,
                    "tss_loss_consumes_logits": False,
                    "tss_survival_target_constructed": False,
                },
            )

    def test_rejects_original_positive_weight_and_sirst3(self) -> None:
        with self.assertRaises(SystemExit):
            trainer.parse_args(
                ["--dataset", "NUAA-SIRST", "--method", "original"]
            )
        with self.assertRaisesRegex(ValueError, "requires --tss-weight 0"):
            _args(weight=0.0025)
        with self.assertRaises(SystemExit):
            trainer.parse_args(
                ["--dataset", "SIRST3", "--method", "final"]
            )

    def test_run_directory_is_disjoint_from_positive_search(self) -> None:
        self.assertEqual(
            trainer.DEFAULT_RESULTS_ROOT,
            trainer.REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            off_args = _args(results_root=directory)
            positive_args = positive.parse_args(
                [
                    "--dataset",
                    "NUAA-SIRST",
                    "--method",
                    "final",
                    "--tss-weight",
                    "0.0025",
                    "--results-root",
                    directory,
                ]
            )
            off_path = trainer._run_directory(off_args)
            positive_path = positive._run_directory(positive_args)
            self.assertNotEqual(off_path, positive_path)
            self.assertEqual(
                off_path.relative_to(Path(directory).resolve()).as_posix(),
                "runs/NUAA-SIRST/final_tss_off/seed_42",
            )


class TSSOffLossAndModelTests(unittest.TestCase):
    def tearDown(self) -> None:
        trainer._AUDIT.reset()

    def test_zero_loss_does_not_consume_nan_logits(self) -> None:
        target = torch.zeros(2, 1, 32, 32)
        segmentation = tuple(
            torch.sigmoid(torch.randn_like(target)).requires_grad_()
            for _ in range(6)
        )
        logit1 = torch.full((2, 1, 2, 2), float("nan"), requires_grad=True)
        logit2 = torch.full((2, 1, 2, 2), float("nan"), requires_grad=True)
        output = TPDForwardOutput(
            segmentation=segmentation,
            emb1_survival_logits=logit1,
            emb2_survival_logits=logit2,
        )
        losses = trainer._compute_loss_off(
            output,
            target,
            nn.BCELoss(),
            survival_weight=0.0,
        )
        self.assertTrue(torch.equal(losses.total, losses.segmentation))
        self.assertEqual(losses.survival_terms, ())
        losses.total.backward()
        self.assertIsNone(logit1.grad)
        self.assertIsNone(logit2.grad)
        audit = trainer._AUDIT.payload()
        self.assertFalse(audit["train_tss_enabled"])
        self.assertFalse(audit["train_tss_survival_target_constructed"])
        self.assertFalse(audit["train_tss_survival_logits_consumed_by_loss"])
        self.assertTrue(audit["train_tss_training_forward_computes_logits"])

    def test_real_final_forward_is_retained_but_heads_do_not_update(self) -> None:
        model, metadata = trainer._build_method_model(
            "final",
            42,
            dataset_name="NUAA-SIRST",
        )
        self.assertIs(
            type(model),
            TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
        )
        self.assertTrue(model.training)
        self.assertTrue(hasattr(model, "target_survival"))
        self.assertFalse(
            metadata["formal_training_objective"]["tss_enabled"]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        images = torch.zeros(2, 1, 32, 32)
        target = torch.zeros_like(images)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("target_survival.")
        }
        output = model(images)
        self.assertIsInstance(output, TPDForwardOutput)
        self.assertIsNotNone(output.survival_logits)
        losses = trainer._compute_loss_off(
            output,
            target,
            nn.BCELoss(),
            survival_weight=0.0,
        )
        losses.total.backward()
        head_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if name.startswith("target_survival.")
        }
        self.assertEqual(set(before), set(head_parameters))
        self.assertTrue(
            all(parameter.grad is None for parameter in head_parameters.values())
        )
        optimizer.step()
        for name, parameter in head_parameters.items():
            self.assertTrue(torch.equal(parameter.detach(), before[name]), msg=name)
            self.assertNotIn(parameter, optimizer.state)


class TSSOffArtifactTests(unittest.TestCase):
    def tearDown(self) -> None:
        trainer._AUDIT.reset()

    def test_protocol_has_off_semantics_and_new_runner_binding(self) -> None:
        args = _args()
        args.smoke = True
        args.device = "cpu"
        args.epochs = 1
        args.begin_test = 1
        args.eval_every = 1
        args.batch_size = 1
        args.max_train_images = 1
        args.max_test_images = 1
        payload = trainer._protocol_payload(
            args,
            model_metadata={"formal_training_objective": {"tss_enabled": False}},
            tss_metadata=trainer._validate_tss_statistics(args)[1],
            data_manifests={
                "files": {
                    "imgidx": {"path": "manifest.json", "sha256": "sha"}
                }
            },
            train_count=1,
            test_count=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(payload["schema"], trainer.SCHEMA)
        self.assertEqual(payload["recipe"]["recipe_id"], "final_tss_off")
        training = payload["training"]
        self.assertFalse(training["tss_enabled"])
        self.assertEqual(training["tss_requested_weight"], 0.0)
        self.assertEqual(training["tss_ratio_cap"], 0.10)
        self.assertFalse(training["tss_ratio_cap_applied"])
        self.assertFalse(training["tss_survival_target_constructed"])
        self.assertFalse(training["tss_survival_logits_consumed_by_loss"])
        self.assertTrue(training["tss_training_forward_computes_logits"])
        budget = payload["search_budget_disclosure"]
        self.assertEqual(budget["final_family_training_runs_after_tss_off"], 12)
        self.assertEqual(budget["original_training_runs"], 3)
        self.assertEqual(budget["final_to_original_recipe_search_ratio"], 4.0)
        self.assertFalse(budget["total_recipe_search_budget_equal"])
        self.assertTrue(budget["tss_off_added_after_positive_test_results"])
        self.assertEqual(
            Path(payload["runtime_sources"]["runner"]["path"]).name,
            "train_three_dataset_tss_off_seed42_v1.py",
        )
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("final_lambda_0p0025", serialized)

    def test_resume_rejects_positive_recipe_identity(self) -> None:
        args = _args()
        args.resume = "required"
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_training_state.pth.tar"
            event = {
                "epoch": 1,
                "recipe": {
                    **trainer.recipe_identity(args),
                    "recipe_id": "final_lambda_0p0025",
                },
            }
            torch.save(
                {
                    "schema": trainer.SCHEMA,
                    "dataset": args.dataset,
                    "method": "final",
                    "seed": 42,
                    "protocol_sha256": "sha",
                    "recipe": event["recipe"],
                    "requested_tss_weight": 0.0,
                    "tss_enabled": False,
                    "epoch": 1,
                    "event": event,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": trainer.engine.rng_state(),
                    "best_miou": {},
                    "best_pd": {},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "resume recipe"):
                trainer._load_resume_off(
                    args=args,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    protocol_sha256="sha",
                )


class TSSOffCpuSmokeTests(unittest.TestCase):
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
            image = torch.zeros(1, 256, 256)
            return image, torch.zeros_like(image)

    class TinyTest(Dataset):
        def __init__(self, train_dataset: str, test_dataset: str, **kwargs: object):
            self.normalization = positive.data_protocol.get_legacy_normalization(
                train_dataset
            )

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            image = torch.zeros(1, 32, 32)
            return image, torch.zeros_like(image), (32, 32), "mock_0"

    class TinyFinal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(-1.0))

        def forward(self, image: torch.Tensor):
            probability = torch.sigmoid(self.logit) * torch.ones_like(image)
            return tuple(probability for _ in range(6))

    @staticmethod
    def _builder(method: str, seed: int, *, dataset_name: str):
        return TSSOffCpuSmokeTests.TinyFinal(), {
            "method": method,
            "seed": seed,
            "dataset_name": dataset_name,
            "formal_training_objective": {"tss_enabled": False},
        }

    def tearDown(self) -> None:
        trainer._AUDIT.reset()

    def test_one_epoch_writes_two_selected_roles_and_off_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = trainer.parse_args(
                [
                    "--dataset",
                    "NUAA-SIRST",
                    "--method",
                    "final",
                    "--tss-weight",
                    "0",
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
            self.assertEqual(summary["schema"], trainer.SCHEMA)
            self.assertEqual(summary["recipe"]["recipe_id"], "final_tss_off")
            self.assertFalse(summary["tss_enabled"])
            self.assertEqual(
                set(summary["checkpoints"]), {"best_miou", "best_pd"}
            )
            self.assertIn("final_tss_off", summary_path.as_posix())
            self.assertFalse(
                (
                    summary_path.parent
                    / "resume"
                    / "latest_training_state.pth.tar"
                ).exists()
            )
            events = [
                json.loads(line)
                for line in Path(summary["metrics"]).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(events), 1)
            self.assertFalse(events[0]["train_tss_enabled"])
            self.assertEqual(events[0]["train_tss_requested_weight"], 0.0)
            self.assertFalse(events[0]["train_tss_ratio_cap_applied"])
            for checkpoint in summary["checkpoints"].values():
                payload = torch.load(
                    checkpoint["path"], map_location="cpu", weights_only=False
                )
                self.assertEqual(payload["recipe"]["recipe_id"], "final_tss_off")
                self.assertFalse(payload["tss_enabled"])

    def test_two_epoch_continuous_and_epoch_boundary_resume_are_exact(self) -> None:
        def make_args(root: Path, *, resume: str) -> argparse.Namespace:
            return trainer.parse_args(
                [
                    "--dataset",
                    "NUAA-SIRST",
                    "--method",
                    "final",
                    "--tss-weight",
                    "0",
                    "--results-root",
                    str(root),
                    "--smoke",
                    "--device",
                    "cpu",
                    "--epochs",
                    "2",
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
            )

        def interrupt_after(root: Path, epoch: int, *, resume: str):
            args = make_args(root, resume=resume)
            original_append = trainer.engine.append_jsonl

            def append_then_interrupt(path: Path, event: dict[str, object]) -> None:
                original_append(path, event)
                if event["epoch"] == epoch:
                    raise RuntimeError(f"planned stop after epoch {epoch}")

            with (
                mock.patch.object(
                    trainer,
                    "_import_runtime_components",
                    return_value=(self._builder, self.TinyTrain, self.TinyTest),
                ),
                mock.patch.object(
                    trainer.engine,
                    "append_jsonl",
                    side_effect=append_then_interrupt,
                ),
                self.assertRaisesRegex(RuntimeError, f"planned stop after epoch {epoch}"),
            ):
                trainer.run(args)
            latest = (
                trainer._run_directory(args)
                / "resume"
                / "latest_training_state.pth.tar"
            )
            self.assertTrue(latest.is_file())
            return torch.load(latest, map_location="cpu", weights_only=False)

        def assert_nested_equal(left: object, right: object, label: str) -> None:
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
                    assert_nested_equal(left[key], right[key], f"{label}.{key}")
                return
            if isinstance(left, (list, tuple)):
                self.assertIsInstance(right, type(left), msg=label)
                self.assertEqual(len(left), len(right), msg=label)
                for index, (left_item, right_item) in enumerate(zip(left, right)):
                    assert_nested_equal(
                        left_item, right_item, f"{label}[{index}]"
                    )
                return
            self.assertEqual(left, right, msg=label)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = interrupt_after(root / "continuous", 2, resume="never")
            first = interrupt_after(root / "split", 1, resume="never")
            self.assertEqual(first["epoch"], 1)
            resumed = interrupt_after(root / "split", 2, resume="required")

            self.assertEqual(continuous["epoch"], 2)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(
                continuous["protocol_sha256"], resumed["protocol_sha256"]
            )
            for field in ("state_dict", "optimizer", "rng_state"):
                assert_nested_equal(
                    continuous[field], resumed[field], f"resume.{field}"
                )
            for role in ("best_miou", "best_pd"):
                for field in ("epoch", "key", "metrics"):
                    assert_nested_equal(
                        continuous[role][field],
                        resumed[role][field],
                        f"resume.{role}.{field}",
                    )
            continuous_event = copy.deepcopy(continuous["event"])
            resumed_event = copy.deepcopy(resumed["event"])
            continuous_event.pop("epoch_seconds", None)
            resumed_event.pop("epoch_seconds", None)
            assert_nested_equal(continuous_event, resumed_event, "resume.event")


if __name__ == "__main__":
    unittest.main()
