"""Architecture, loss, state, and launch tests for SIRST3 A0--A4."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from experiments.four_dataset_ablation_models_seed42_v1 import (
    ABLATION_IDS,
    EXPECTED_PARAMETER_COUNTS,
    MODULE_FLAGS,
    TSS_WEIGHT_BY_ABLATION,
    build_all_ablation_models,
)
from experiments.launch_sirst3_ablation_seed42_after_main_v1 import (
    dry_run_manifest,
)
from experiments.tpd_training_loss import compute_tpd_training_loss
from experiments.train_sirst3_ablation_seed42_exact_v1 import (
    parse_args,
    validate_args,
)
from model.tpd_forward_contract import TPDForwardOutput


class TestSirst3AblationModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models, cls.metadata = build_all_ablation_models()

    def test_exact_graph_parameter_and_state_contract(self) -> None:
        self.assertEqual(tuple(self.models), ABLATION_IDS)
        for ablation_id, model in self.models.items():
            flags = MODULE_FLAGS[ablation_id]
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                EXPECTED_PARAMETER_COUNTS[ablation_id],
            )
            self.assertEqual(hasattr(model, "tpd_ner"), flags["ner"])
            self.assertEqual(hasattr(model, "tpd_qfg"), flags["qfg"])
            self.assertEqual(
                hasattr(model, "target_survival"),
                flags["tss"],
            )
            record = self.metadata["models"][ablation_id]
            self.assertEqual(record["state_key_count"], len(model.state_dict()))
            self.assertEqual(
                record["tss_training_weight"],
                TSS_WEIGHT_BY_ABLATION[ablation_id],
            )
        self.assertFalse(self.metadata["capacity_matching_claimed"])
        self.assertIn("A0->A1", self.metadata["capacity_note"])

    def test_every_cumulative_shared_tensor_is_bitwise_paired(self) -> None:
        for transition in self.metadata["paired_transitions"].values():
            self.assertGreater(transition["shared_key_count"], 0)
            self.assertTrue(transition["shared_state_bitwise_equal"])
            self.assertEqual(
                transition["source_shared_state_sha256"],
                transition["target_shared_state_sha256"],
            )

    def test_strict_state_reload_for_every_graph(self) -> None:
        for ablation_id, model in self.models.items():
            replica = copy.deepcopy(model)
            incompatible = replica.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(incompatible.missing_keys, [], ablation_id)
            self.assertEqual(incompatible.unexpected_keys, [], ablation_id)
            del replica

    def test_all_graphs_forward_and_loss_are_finite(self) -> None:
        torch.manual_seed(42)
        image = torch.randn(1, 1, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 30:33, 31:34] = 1.0
        criterion = nn.BCELoss()
        for ablation_id, model in self.models.items():
            model.train()
            with torch.no_grad():
                output = model(image)
                loss = compute_tpd_training_loss(
                    output,
                    target,
                    criterion,
                    survival_weight=TSS_WEIGHT_BY_ABLATION[ablation_id],
                    survival_pos_weight=2.0,
                )
            self.assertTrue(torch.isfinite(loss.total), ablation_id)
            self.assertEqual(
                len(loss.segmentation_terms),
                6,
                ablation_id,
            )
            if ablation_id == "A4":
                self.assertIsInstance(output, TPDForwardOutput)
                self.assertEqual(len(loss.survival_terms), 2)
                self.assertGreater(float(loss.survival), 0.0)
            else:
                self.assertNotIsInstance(output, TPDForwardOutput)
                self.assertEqual(loss.survival_terms, ())
                self.assertEqual(float(loss.survival), 0.0)

    def test_full_graph_backward_reaches_all_added_modules(self) -> None:
        model = self.models["A4"]
        model.train()
        model.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(4205)
        image = torch.randn(1, 1, 64, 64, generator=generator)
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 7:10, 12:15] = 1.0
        output = model(image)
        loss = compute_tpd_training_loss(
            output,
            target,
            nn.BCELoss(),
            survival_weight=0.005,
            survival_pos_weight=2.0,
        )
        loss.total.backward()

        groups = {
            "tpd": "mtc.embeddings_1.blocks.",
            "ner": "tpd_ner.",
            "qfg": "tpd_qfg.",
            "tss": "target_survival.",
        }
        for group, prefix in groups.items():
            gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            self.assertTrue(gradients, group)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients),
                group,
            )
            self.assertTrue(
                any(int(torch.count_nonzero(gradient).item()) > 0 for gradient in gradients),
                group,
            )
        self.assertTrue(math.isfinite(float(loss.total.detach())))

    def test_a4_eval_does_not_execute_training_only_output_contract(self) -> None:
        model = self.models["A4"]
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 64, 64))
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 6)


class TestSirst3AblationRunnerAndLauncher(unittest.TestCase):
    def test_formal_cli_is_frozen_and_has_only_two_selected_roles(self) -> None:
        args = parse_args(
            [
                "--ablation",
                "A2",
                "--physical-gpu-index",
                "2",
                "--expected-gpu-uuid",
                "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
            ]
        )
        validate_args(args)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.begin_test, 10)
        self.assertEqual(args.eval_every, 10)

    def test_dry_run_waits_for_main_and_schedules_only_a1_a3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = dry_run_manifest(root)
        self.assertFalse(manifest["main_gate_complete_now"])
        self.assertTrue(manifest["will_wait_for_main_matrix"])
        self.assertFalse(manifest["waits_for_gpu_utilization"])
        self.assertTrue(manifest["A0_A4_reused_from_main"])
        self.assertEqual(
            manifest["candidate_epochs"],
            list(range(10, 1001, 10)),
        )
        self.assertEqual(manifest["candidate_epoch_count"], 100)
        scheduled = [
            task["ablation_id"]
            for wave in manifest["waves"]
            for task in wave
        ]
        self.assertEqual(scheduled, ["A1", "A2", "A3"])
        self.assertEqual(
            manifest["persistent_checkpoint_roles_per_run"],
            ["best_miou", "best_pd"],
        )
        gpu = {
            task["ablation_id"]: task["physical_gpu"]
            for wave in manifest["waves"]
            for task in wave
        }
        self.assertEqual(gpu, {"A1": "2", "A2": "3", "A3": "2"})
        for wave in manifest["waves"]:
            for task in wave:
                command = task["command"]
                self.assertEqual(command[command.index("--begin-test") + 1], "10")
                self.assertEqual(command[command.index("--eval-every") + 1], "10")


if __name__ == "__main__":
    unittest.main()
