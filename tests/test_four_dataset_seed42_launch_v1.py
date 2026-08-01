from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments import four_dataset_seed42_launch_v1 as launch


def _option(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


class WorkerCommandTests(unittest.TestCase):
    def test_formal_matrix_and_wave_order_are_frozen(self) -> None:
        specs = launch.build_all_worker_specs(
            mode="formal",
            base_environment={},
        )
        self.assertEqual(len(specs), 8)
        self.assertEqual(
            [(spec.dataset, spec.method) for spec in specs],
            [
                (dataset, method)
                for dataset in launch.DATASETS
                for method in launch.METHODS
            ],
        )

    def test_method_to_gpu_assignment_never_uses_gpu_zero_or_one(self) -> None:
        for mode in ("smoke", "formal"):
            for dataset in launch.DATASETS:
                original = launch.build_worker_spec(
                    dataset,
                    "original",
                    mode=mode,
                    base_environment={},
                )
                final = launch.build_worker_spec(
                    dataset,
                    "final",
                    mode=mode,
                    base_environment={},
                )
                self.assertEqual(
                    original.environment["CUDA_VISIBLE_DEVICES"],
                    launch.GPU_ASSIGNMENT["original"]["uuid"],
                )
                self.assertEqual(
                    final.environment["CUDA_VISIBLE_DEVICES"],
                    launch.GPU_ASSIGNMENT["final"]["uuid"],
                )
                self.assertEqual(
                    _option(original.command, "--physical-gpu-index"), "2"
                )
                self.assertEqual(
                    _option(final.command, "--physical-gpu-index"), "3"
                )
                self.assertEqual(
                    _option(original.command, "--device"), "cuda:0"
                )
                self.assertEqual(_option(final.command, "--device"), "cuda:0")
                joined = " ".join(original.command + final.command)
                self.assertNotIn("cuda:2", joined)
                self.assertNotIn("cuda:3", joined)

    def test_formal_command_is_exact_and_final_gets_tss(self) -> None:
        original = launch.build_worker_spec(
            "SIRST3",
            "original",
            mode="formal",
            base_environment={},
        )
        final = launch.build_worker_spec(
            "SIRST3",
            "final",
            mode="formal",
            base_environment={},
        )
        for spec in (original, final):
            self.assertEqual(spec.command[0], str(launch.PYTHON))
            self.assertEqual(spec.command[1], str(launch.RUNNER))
            self.assertEqual(_option(spec.command, "--seed"), "42")
            self.assertEqual(_option(spec.command, "--epochs"), "1000")
            self.assertEqual(_option(spec.command, "--begin-test"), "10")
            self.assertEqual(_option(spec.command, "--eval-every"), "10")
            self.assertEqual(_option(spec.command, "--batch-size"), "16")
            self.assertEqual(_option(spec.command, "--patch-size"), "256")
            self.assertEqual(_option(spec.command, "--workers"), "0")
            self.assertEqual(_option(spec.command, "--resume"), "auto")
            self.assertNotIn("--smoke", spec.command)
            self.assertNotIn("--max-train-images", spec.command)
            self.assertNotIn("--max-test-images", spec.command)
        self.assertNotIn("--tss-statistics", original.command)
        self.assertEqual(
            _option(final.command, "--tss-statistics"),
            str(launch.TSS_STATISTICS),
        )

    def test_smoke_command_is_two_epoch_and_small(self) -> None:
        spec = launch.build_worker_spec(
            "NUDT-SIRST",
            "final",
            mode="smoke",
            max_train_images=4,
            max_test_images=3,
            base_environment={},
        )
        self.assertIn("--smoke", spec.command)
        self.assertEqual(_option(spec.command, "--epochs"), "2")
        self.assertEqual(_option(spec.command, "--begin-test"), "1")
        self.assertEqual(_option(spec.command, "--eval-every"), "1")
        self.assertEqual(_option(spec.command, "--batch-size"), "2")
        self.assertEqual(_option(spec.command, "--max-train-images"), "4")
        self.assertEqual(_option(spec.command, "--max-test-images"), "3")
        self.assertEqual(
            _option(spec.command, "--tss-statistics"),
            str(launch.TSS_STATISTICS),
        )

    def test_command_record_names_only_two_selected_roles(self) -> None:
        spec = launch.build_worker_spec(
            "IRSTD-1K",
            "original",
            mode="formal",
            base_environment={},
        )
        record = launch.command_record(spec)
        self.assertEqual(
            record["checkpoint_roles"], ["best_miou", "best_pd"]
        )
        self.assertFalse(
            record["rolling_resume_state_is_selected_checkpoint"]
        )


class TssManifestValidationTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        datasets = {}
        for index, dataset in enumerate(launch.DATASETS, start=1):
            datasets[dataset] = {
                "dataset": dataset,
                "training_seed": 42,
                "epochs": 1000,
                "completed_through_epoch": 1000,
                "complete": True,
                "survival_pos_weight": 10.0 + index,
                "positive_cells": index,
                "negative_cells": index * 10,
                "aggregate_plan_sha256": f"sha-{index}",
            }
        return {
            "schema": (
                "sctransnet_four_dataset_exact_tss_statistics/v1"
            ),
            "training_seed": 42,
            "epochs": 1000,
            "datasets": datasets,
        }

    def test_valid_four_dataset_tss_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tss.json"
            path.write_text(
                json.dumps(self._payload()), encoding="utf-8"
            )
            record = launch.validate_tss_statistics(path)
            self.assertEqual(record["training_seed"], 42)
            self.assertEqual(set(record["datasets"]), set(launch.DATASETS))
            self.assertEqual(len(record["sha256"]), 64)

    def test_wrong_seed_or_nonfinite_weight_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tss.json"
            payload = self._payload()
            payload["training_seed"] = 7
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(launch.LaunchProtocolError):
                launch.validate_tss_statistics(path)
            payload["training_seed"] = 42
            payload["datasets"]["SIRST3"][  # type: ignore[index]
                "survival_pos_weight"
            ] = math.inf
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(launch.LaunchProtocolError):
                launch.validate_tss_statistics(path)


if __name__ == "__main__":
    unittest.main()
