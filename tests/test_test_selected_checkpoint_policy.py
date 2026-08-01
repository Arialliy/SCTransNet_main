from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments import four_dataset_evaluation_protocol_v1 as protocol
from experiments import select_four_dataset_test_checkpoints_v1 as selector


def metric_event(epoch: int, **overrides: float) -> dict:
    metrics = {
        "threshold": 0.5,
        "test_loss": 0.1,
        "miou": 0.5,
        "niou": 0.4,
        "pixel_f1": 0.6,
        "pd": 0.7,
        "tiny_pd": 0.5,
        "fa": 1e-6,
    }
    metrics.update(overrides)
    return {
        "epoch": epoch,
        "evaluated": True,
        "test_metrics": metrics,
    }


class SelectionPolicyTests(unittest.TestCase):
    def test_formal_candidate_grid_is_every_ten_epochs(self) -> None:
        self.assertEqual(protocol.CANDIDATE_FIRST_EPOCH, 10)
        self.assertEqual(protocol.CANDIDATE_LAST_EPOCH, 1000)
        self.assertEqual(protocol.CANDIDATE_EVAL_EVERY, 10)
        self.assertEqual(
            protocol.CANDIDATE_EPOCHS,
            tuple(range(10, 1001, 10)),
        )
        self.assertEqual(len(protocol.CANDIDATE_EPOCHS), 100)

    def test_complete_selection_orders_and_earlier_epoch_tie(self) -> None:
        events = [metric_event(epoch) for epoch in protocol.CANDIDATE_EPOCHS]
        events[10] = metric_event(
            110,
            miou=0.8,
            pd=0.7,
            niou=0.6,
            tiny_pd=0.6,
            fa=2e-6,
        )
        events[11] = metric_event(
            120,
            miou=0.8,
            pd=0.7,
            niou=0.6,
            tiny_pd=0.6,
            fa=2e-6,
        )
        events[20] = metric_event(
            210,
            miou=0.7,
            pd=0.9,
            niou=0.7,
            tiny_pd=0.8,
            fa=1e-7,
        )
        self.assertEqual(
            protocol.selected_event("best_miou", events)["epoch"],
            110,
        )
        self.assertEqual(
            protocol.selected_event("best_pd", events)["epoch"],
            210,
        )

    def test_selection_includes_niou_before_tiny_pd_for_best_miou(self) -> None:
        better_niou = metric_event(
            10,
            miou=0.8,
            pd=0.8,
            fa=1e-6,
            niou=0.7,
            tiny_pd=0.1,
        )
        better_tiny = metric_event(
            20,
            miou=0.8,
            pd=0.8,
            fa=1e-6,
            niou=0.6,
            tiny_pd=1.0,
        )
        self.assertEqual(
            protocol.selected_event(
                "best_miou",
                [better_tiny, better_niou],
            )["epoch"],
            10,
        )

    def test_reader_rejects_missing_candidate_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            events = [
                metric_event(epoch)
                for epoch in protocol.CANDIDATE_EPOCHS
                if epoch != 770
            ]
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing=.*770"):
                protocol.read_candidate_metrics(path)

    def test_threshold_must_be_exactly_half(self) -> None:
        event = metric_event(10)
        event["test_metrics"]["threshold"] = 0.5000001
        with self.assertRaisesRegex(ValueError, "exactly 0.5"):
            protocol.normalize_metric_event(event)

    def test_tiny_pd_null_is_allowed_only_with_consistent_counts(self) -> None:
        point = metric_event(10)["test_metrics"]
        point.update(
            {
                "tiny_pd": None,
                "tiny_target_count": 0,
                "matched_tiny_target_count": 0,
            }
        )
        normalized = protocol.normalize_metric_event(
            {"epoch": 10, "test_metrics": point}
        )
        self.assertIsNone(normalized["tiny_pd"])
        invalid = copy.deepcopy(point)
        invalid["tiny_target_count"] = 1
        with self.assertRaisesRegex(ValueError, "tiny_pd differs"):
            protocol.normalize_metric_event(
                {"epoch": 10, "test_metrics": invalid}
            )

    def test_full_run_freezes_verified_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root = root / "runs"
            selected_root = root / "selected"
            run_dir = protocol.run_directory(
                "SIRST3",
                "original",
                runs_root=runs_root,
            )
            (run_dir / "checkpoints").mkdir(parents=True)
            protocol_payload = {
                "dataset": "SIRST3",
                "method": "original",
                "training_seed": 42,
                "epochs": 1000,
                "begin_test": 10,
                "eval_every": 10,
                "metrics": {
                    "threshold": 0.5,
                    "match_radius": 3.0,
                    "tiny_area": 9,
                },
                "scratch": True,
            }
            (run_dir / "protocol.json").write_text(
                json.dumps(protocol_payload),
                encoding="utf-8",
            )
            events = [metric_event(epoch) for epoch in protocol.CANDIDATE_EPOCHS]
            for event in events:
                event["test_metrics"].pop("threshold")
            events[12] = metric_event(130, miou=0.9)
            events[23] = metric_event(240, pd=0.95)
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            selection_events = [
                {**event, "threshold": 0.5} for event in events
            ]
            selected = {
                "best_miou": protocol.selected_event(
                    "best_miou", selection_events
                ),
                "best_pd": protocol.selected_event(
                    "best_pd", selection_events
                ),
            }
            for role, metrics in selected.items():
                torch.save(
                    {
                        "dataset": "SIRST3",
                        "method": "original",
                        "seed": 42,
                        "epoch": metrics["epoch"],
                        "checkpoint_role": role,
                        "test_metrics": {
                            key: value
                            for key, value in metrics.items()
                            if key not in {"epoch", "threshold"}
                        },
                        "state_dict": {"weight": torch.tensor([1.0])},
                        "test_selected": True,
                        "selection_is_optimistic": True,
                    },
                    run_dir / "checkpoints" / protocol.CHECKPOINT_FILENAMES[role],
                )
            record = selector.audit_and_freeze_run(
                "SIRST3",
                "original",
                runs_root=runs_root,
                selected_root=selected_root,
            )
            self.assertTrue(record["audit_passed"])
            self.assertEqual(
                record["checkpoints"]["best_miou"]["epoch"],
                130,
            )
            self.assertEqual(
                record["fixed_endpoint_epoch1000"]["epoch"],
                1000,
            )
            self.assertFalse(
                record["fixed_endpoint_epoch1000"]["checkpoint_saved"]
            )
            for role in protocol.CHECKPOINT_ROLES:
                frozen = protocol.selected_checkpoint_path(
                    "SIRST3",
                    "original",
                    role,
                    selected_root=selected_root,
                )
                self.assertTrue(frozen.is_file())
                self.assertEqual(
                    protocol.file_sha256(frozen),
                    record["checkpoints"][role]["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
