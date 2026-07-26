from __future__ import annotations

import argparse
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments import resume_tpd_clean_v3 as resume
from experiments import train_tpd_pilot as base


VARIANT = "tpd_clean_v3_full"
RUN_TAG = "resume_test"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validation_metrics(
    *,
    pd: float,
    fa: float,
    miou: float,
    matched: int,
) -> dict[str, object]:
    return {
        "val_loss": 1.0 - miou,
        "miou": miou,
        "niou": miou - 0.01,
        "pixel_precision": 0.91,
        "pixel_recall": 0.92,
        "pixel_f1": 0.915,
        "pd": pd,
        "tiny_pd": 1.0,
        "fa": fa,
        "false_objects_per_image": 0.01,
        "target_count": 10,
        "matched_target_count": matched,
        "tiny_target_count": 2,
        "matched_tiny_target_count": 2,
        "predicted_object_count": matched + 1,
        "unmatched_predicted_object_count": 1,
        "valid_pixel_count": 10000,
    }


def _event(
    epoch: int,
    arguments: dict[str, object],
    *,
    pd: float,
    fa: float,
    miou: float,
    matched: int,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "variant": arguments["variant"],
        "train_loss": 0.1 / epoch,
        "learning_rate": base.learning_rate_for_epoch(
            epoch,
            int(arguments["epochs"]),
            float(arguments["base_lr"]),
            float(arguments["min_lr"]),
            int(arguments["warmup_epochs"]),
        ),
        "processed_train_samples": 4,
        "epoch_seconds": 0.1,
        **_validation_metrics(
            pd=pd,
            fa=fa,
            miou=miou,
            matched=matched,
        ),
        "new_best_pd": True,
        "new_best_miou": True,
        # Extra event keys are deliberately allowed for resumed histories.
        "fixture_extra_event_key": {"kept": True},
    }


def _checkpoint(
    *,
    epoch: int,
    role: str,
    metrics: dict[str, object],
    arguments: dict[str, object],
    model_metadata: dict[str, object],
    split_hashes: dict[str, str],
) -> dict[str, object]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return {
        "epoch": epoch,
        "variant": arguments["variant"],
        "dataset": arguments["dataset"],
        "seed": arguments["seed"],
        "split_seed": arguments["split_seed"],
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": {},
        "validation_metrics": metrics,
        "model_metadata": model_metadata,
        "split_hashes": split_hashes,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "checkpoint_role": role,
    }


class ResumeFixture:
    def __init__(self, root: Path) -> None:
        output_root = root / "results"
        self.run_dir = (
            output_root
            / "NUDT-SIRST"
            / VARIANT
            / f"seed_42_{RUN_TAG}"
        )
        self.run_dir.mkdir(parents=True)
        self.arguments: dict[str, object] = {
            "variant": VARIANT,
            "dataset": "NUDT-SIRST",
            "dataset_dir": str(root / "datasets"),
            "output_root": str(output_root),
            "run_tag": RUN_TAG,
            "device": "cuda:0",
            "epochs": 4,
            "batch_size": 2,
            "patch_size": 32,
            "workers": 0,
            "seed": 42,
            "split_seed": 20260722,
            "val_fraction": 0.2,
            "eval_every": 1,
            "base_lr": 0.001,
            "min_lr": 0.00001,
            "warmup_epochs": 1,
            "threshold": 0.5,
            "match_radius": 3.0,
            "tiny_area": 9,
            "amp": False,
            "max_train_images": None,
            "max_val_images": None,
        }
        self.model_metadata: dict[str, object] = {
            "variant": VARIANT,
            "candidate_family": "fixture",
        }
        protocol = {
            "arguments": self.arguments,
            "run_directory": str(self.run_dir),
            "model": self.model_metadata,
            "normalization": {"mean": 1.0, "std": 2.0},
        }
        _write_json(self.run_dir / "protocol.json", protocol)

        train_ids = ["train-a", "train-b", "train-c", "train-d"]
        val_ids = ["val-a"]
        self.split_hashes = {
            "full_internal_train_sha256": base.identifier_hash(train_ids),
            "full_internal_val_sha256": base.identifier_hash(val_ids),
            "used_train_sha256": base.identifier_hash(train_ids),
            "used_val_sha256": base.identifier_hash(val_ids),
        }
        split = {
            "dataset": "NUDT-SIRST",
            "split_seed": 20260722,
            "val_fraction": 0.2,
            "used_train_count": len(train_ids),
            "used_val_count": len(val_ids),
            "used_train_ids": train_ids,
            "used_val_ids": val_ids,
            "hashes": self.split_hashes,
        }
        _write_json(self.run_dir / "split.json", split)

        self.rows = [
            _event(
                1,
                self.arguments,
                pd=0.8,
                fa=2e-5,
                miou=0.70,
                matched=8,
            ),
            _event(
                2,
                self.arguments,
                pd=0.9,
                fa=5e-6,
                miou=0.80,
                matched=9,
            ),
        ]
        self.write_metrics()
        selection = resume.rebuild_best_selection(self.rows)
        latest_metrics = resume._extract_validation_metrics(self.rows[-1])
        last = _checkpoint(
            epoch=2,
            role="last_evaluated_epoch",
            metrics=latest_metrics,
            arguments=self.arguments,
            model_metadata=self.model_metadata,
            split_hashes=self.split_hashes,
        )
        best = _checkpoint(
            epoch=selection.best_pd_epoch,
            role="best_validation_pd_primary",
            metrics=selection.best_pd_metrics,
            arguments=self.arguments,
            model_metadata=self.model_metadata,
            split_hashes=self.split_hashes,
        )
        best_miou = _checkpoint(
            epoch=selection.best_miou_epoch,
            role="best_validation_miou_secondary",
            metrics=selection.best_miou_metrics,
            arguments=self.arguments,
            model_metadata=self.model_metadata,
            split_hashes=self.split_hashes,
        )
        torch.save(last, self.run_dir / "last.pth.tar")
        torch.save(best, self.run_dir / "best.pth.tar")
        torch.save(best_miou, self.run_dir / "best_miou.pth.tar")

    def write_metrics(self) -> None:
        (self.run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )

    def load(self) -> resume.ResumeState:
        return resume.load_resume_state(
            self.run_dir,
            expected_resume_epoch=2,
            target_epoch=4,
        )


class RandomTraceDataset(Dataset):
    def __len__(self) -> int:
        return 7

    def __getitem__(self, index: int) -> tuple[int, int, float]:
        return (
            index,
            random.randint(0, 1_000_000),
            random.random(),
        )


def _trace_loader(seed: int) -> DataLoader:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        RandomTraceDataset(),
        batch_size=3,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def _collect_trace_epoch(loader: DataLoader) -> list[tuple[int, int, float]]:
    observed: list[tuple[int, int, float]] = []
    for indices, integers, floats in loader:
        observed.extend(
            (
                int(index),
                int(integer),
                float(random_value),
            )
            for index, integer, random_value in zip(
                indices.tolist(),
                integers.tolist(),
                floats.tolist(),
            )
        )
    return observed


class ResumeTPDCleanV3Tests(unittest.TestCase):
    def test_valid_prefix_accepts_extra_event_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            state = fixture.load()
            self.assertEqual(state.completed_epoch, 2)
            self.assertEqual(state.target_epoch, 4)
            self.assertEqual(state.selection.best_pd_epoch, 2)

    def test_rejects_wrong_expected_checkpoint_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            with self.assertRaises(resume.ResumeValidationError):
                resume.load_resume_state(
                    fixture.run_dir,
                    expected_resume_epoch=3,
                    target_epoch=4,
                )

    def test_rejects_noncontiguous_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            fixture.rows[1]["epoch"] = 3
            fixture.write_metrics()
            with self.assertRaises(resume.ResumeValidationError):
                fixture.load()

    def test_rejects_wrong_metrics_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            fixture.rows[1]["variant"] = "tpd_clean_v3_sal_capacity"
            fixture.write_metrics()
            with self.assertRaises(resume.ResumeValidationError):
                fixture.load()

    def test_best_state_is_rebuilt_with_original_selection_order(self) -> None:
        arguments = {
            "variant": VARIANT,
            "epochs": 4,
            "base_lr": 0.001,
            "min_lr": 0.00001,
            "warmup_epochs": 1,
        }
        rows = [
            _event(
                1,
                arguments,
                pd=0.9,
                fa=9e-6,
                miou=0.80,
                matched=9,
            ),
            _event(
                2,
                arguments,
                pd=0.9,
                fa=2e-6,
                miou=0.81,
                matched=9,
            ),
            _event(
                3,
                arguments,
                pd=0.8,
                fa=1e-7,
                miou=0.92,
                matched=8,
            ),
        ]
        state = resume.rebuild_best_selection(rows)
        self.assertEqual(state.best_pd_epoch, 2)
        self.assertEqual(state.best_miou_epoch, 3)

    def test_loader_replay_reaches_same_next_data_random_stream(self) -> None:
        uninterrupted = _trace_loader(3407)
        for _ in range(3):
            _collect_trace_epoch(uninterrupted)
        expected_next = _collect_trace_epoch(uninterrupted)

        replayed = _trace_loader(3407)
        stats = resume.replay_loader_epochs(
            replayed,
            3,
            progress_every=0,
        )
        actual_next = _collect_trace_epoch(replayed)
        self.assertEqual(actual_next, expected_next)
        self.assertEqual(stats.epochs, 3)
        self.assertEqual(stats.samples, 21)

    def test_checkpoints_and_summary_bind_resume_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeFixture(Path(temporary))
            state = fixture.load()
            replay = resume.ReplayStats(
                epochs=2,
                batches=4,
                samples=8,
                elapsed_seconds=0.01,
            )
            binding = resume._write_segment_and_provenance(
                state,
                replay=replay,
                device_text="cpu",
                resume_gpu_uuid="CPU-test",
            )
            resume._bind_existing_checkpoints(state, binding)
            provenance_digest = resume._sha256_file(
                fixture.run_dir / resume.PROVENANCE_NAME
            )
            self.assertEqual(
                binding["resume_provenance_sha256"],
                provenance_digest,
            )
            for name in (
                "last.pth.tar",
                "best.pth.tar",
                "best_miou.pth.tar",
            ):
                checkpoint = torch.load(
                    fixture.run_dir / name,
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(
                    checkpoint["resume_provenance_sha256"],
                    provenance_digest,
                )
                self.assertFalse(
                    checkpoint["resume_disclosure"][
                        "cuda_bitwise_continuity_claimed"
                    ]
                )
            summary = resume._final_summary(
                state,
                state.selection,
                binding=binding,
                skipped_singleton_batches=0,
                process_elapsed_seconds=1.0,
            )
            self.assertEqual(
                summary["resume_provenance_sha256"],
                provenance_digest,
            )
            self.assertFalse(
                summary["resume_disclosure"][
                    "cuda_bitwise_continuity_claimed"
                ]
            )

            # A second invocation can validate the provenance-bound prefix.
            reloaded = fixture.load()
            self.assertEqual(reloaded.completed_epoch, 2)

    def test_fixed_cli_names(self) -> None:
        args = resume._parse_args(
            [
                "--run-dir",
                "/tmp/example",
                "--device",
                "cuda:0",
                "--target-epoch",
                "800",
                "--expected-resume-epoch",
                "279",
                "--resume-gpu-uuid",
                "GPU-example",
            ]
        )
        self.assertEqual(args.target_epoch, 800)
        self.assertEqual(args.expected_resume_epoch, 279)
        self.assertEqual(args.resume_gpu_uuid, "GPU-example")


if __name__ == "__main__":
    unittest.main()
