from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import tpd_exact_epoch_journal as journal_module
from experiments import tpd_exact_resume as exact


def exact_payload(prepared: journal_module.PreparedEpochEvent) -> dict:
    """Minimal serializable payload with the exact-resume top-level schema."""

    return {
        "schema": exact.EXACT_RESUME_SCHEMA,
        "mode": exact.EXACT_RESUME_MODE,
        "epoch": prepared.epoch,
        "run_identity": {},
        "model": {},
        "optimizer": {},
        "scaler": {},
        "scheduler": None,
        "best_selection": {},
        "metrics_boundary": copy.deepcopy(prepared.metrics_boundary),
        "rng_state": {},
        "extra_state": {},
    }


class ExactEpochJournalTest(unittest.TestCase):
    def make_journal(
        self,
        directory: str,
    ) -> journal_module.ExactEpochJournal:
        return journal_module.ExactEpochJournal(Path(directory) / "journal")

    def commit_epoch(
        self,
        journal: journal_module.ExactEpochJournal,
        epoch: int,
    ) -> journal_module.ActiveEpochState:
        prepared = journal.prepare_next_event(
            {"epoch": epoch, "loss": 1.0 / epoch}
        )
        return journal.commit(prepared, exact_payload(prepared))

    def test_no_marker_is_empty_and_first_commit_uses_slot_a(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            self.assertIsNone(journal.load_active())
            prepared = journal.prepare_next_event({"loss": 1.0, "epoch": 1})
            self.assertEqual(prepared.target_slot, "slot_a")
            self.assertEqual(
                prepared.event_bytes,
                b'{"epoch":1,"loss":1.0}\n',
            )
            state = journal.commit(prepared, exact_payload(prepared))
            self.assertEqual((state.slot, state.epoch), ("slot_a", 1))
            self.assertEqual(
                state.metrics_boundary,
                prepared.metrics_boundary,
            )

    def test_commits_switch_a_b_and_keep_only_fixed_slot_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            first = self.commit_epoch(journal, 1)
            first_checkpoint = first.checkpoint_path.read_bytes()
            second = self.commit_epoch(journal, 2)
            self.assertEqual((second.slot, second.epoch), ("slot_b", 2))
            self.assertEqual(first.checkpoint_path.read_bytes(), first_checkpoint)
            third = self.commit_epoch(journal, 3)
            self.assertEqual((third.slot, third.epoch), ("slot_a", 3))
            self.assertEqual(
                {path.name for path in journal.root.iterdir()},
                {
                    journal_module.MARKER_FILENAME,
                    "slot_a.metrics.jsonl",
                    "slot_a.exact.pth",
                    "slot_b.metrics.jsonl",
                    "slot_b.exact.pth",
                },
            )

    def test_metrics_write_failure_before_checkpoint_keeps_old_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            old = self.commit_epoch(journal, 1)
            prepared = journal.prepare_next_event({"epoch": 2, "loss": 0.5})
            with mock.patch.object(
                exact,
                "atomic_torch_save",
                side_effect=OSError("injected checkpoint failure"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected checkpoint failure",
                ):
                    journal.commit(prepared, exact_payload(prepared))
            active = journal.load_active()
            self.assertIsNotNone(active)
            self.assertEqual((active.slot, active.epoch), (old.slot, old.epoch))

    def test_checkpoint_write_then_marker_failure_keeps_old_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            old = self.commit_epoch(journal, 1)
            prepared = journal.prepare_next_event({"epoch": 2, "loss": 0.5})
            real_atomic_write = journal_module._atomic_write_bytes

            def fail_marker(destination: Path, content: bytes) -> None:
                if destination.name == journal_module.MARKER_FILENAME:
                    raise OSError("injected marker failure")
                real_atomic_write(destination, content)

            with mock.patch.object(
                journal_module,
                "_atomic_write_bytes",
                side_effect=fail_marker,
            ):
                with self.assertRaisesRegex(OSError, "injected marker failure"):
                    journal.commit(prepared, exact_payload(prepared))
            active = journal.load_active()
            self.assertIsNotNone(active)
            self.assertEqual((active.slot, active.epoch), (old.slot, old.epoch))

    def test_uncommitted_inactive_pair_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            old = self.commit_epoch(journal, 1)
            inactive_metrics, inactive_checkpoint = journal._slot_paths("slot_b")
            inactive_metrics.write_bytes(b'{"epoch":99}\n')
            inactive_checkpoint.write_bytes(b"not a checkpoint")
            active = journal.load_active()
            self.assertIsNotNone(active)
            self.assertEqual((active.slot, active.epoch), (old.slot, old.epoch))

    def test_runtime_cache_avoids_reloading_checkpoint_each_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            real_torch_load = journal_module.torch.load
            with mock.patch.object(
                journal_module.torch,
                "load",
                wraps=real_torch_load,
            ) as load:
                first = journal.prepare_next_event({"epoch": 1, "loss": 1.0})
                journal.commit(first, exact_payload(first))
                second = journal.prepare_next_event({"epoch": 2, "loss": 0.5})
                journal.commit(second, exact_payload(second))
                self.assertEqual(load.call_count, 0)

                journal.load_active()
                self.assertEqual(load.call_count, 1)

                restarted = journal_module.ExactEpochJournal(journal.root)
                third = restarted.prepare_next_event(
                    {"epoch": 3, "loss": 0.25}
                )
                self.assertEqual(load.call_count, 2)
                restarted.commit(third, exact_payload(third))
                restarted.prepare_next_event({"epoch": 4, "loss": 0.125})
                self.assertEqual(load.call_count, 2)

    def test_runtime_rejects_disappeared_committed_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            self.commit_epoch(journal, 1)
            journal.marker_path.unlink()
            with self.assertRaisesRegex(
                journal_module.ExactEpochJournalError,
                "marker disappeared",
            ):
                journal.prepare_next_event({"epoch": 2, "loss": 0.5})

            restarted = journal_module.ExactEpochJournal(journal.root)
            self.assertIsNone(restarted.load_active())

    def test_active_metrics_or_checkpoint_corruption_is_rejected(self) -> None:
        for target in ("metrics", "checkpoint"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                journal = self.make_journal(directory)
                active = self.commit_epoch(journal, 1)
                path = (
                    active.metrics_path
                    if target == "metrics"
                    else active.checkpoint_path
                )
                path.write_bytes(path.read_bytes() + b"corrupt")
                with self.assertRaises(journal_module.ExactEpochJournalError):
                    journal.load_active()

    def test_payload_epoch_and_boundary_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            prepared = journal.prepare_next_event({"epoch": 1, "loss": 1.0})
            wrong_epoch = exact_payload(prepared)
            wrong_epoch["epoch"] = 2
            with self.assertRaisesRegex(
                journal_module.ExactEpochJournalError,
                "epoch mismatch",
            ):
                journal.commit(prepared, wrong_epoch)
            wrong_boundary = exact_payload(prepared)
            wrong_boundary["metrics_boundary"] = copy.deepcopy(
                prepared.metrics_boundary
            )
            wrong_boundary["metrics_boundary"]["metrics_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                journal_module.ExactEpochJournalError,
                "metrics_boundary mismatch",
            ):
                journal.commit(prepared, wrong_boundary)

    def test_rejects_noncontiguous_nonfinite_tail_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = self.make_journal(directory)
            with self.assertRaisesRegex(
                journal_module.ExactEpochJournalError,
                "exactly 1",
            ):
                journal.prepare_next_event({"epoch": 2, "loss": 1.0})
            with self.assertRaisesRegex(
                journal_module.ExactEpochJournalError,
                "non-finite",
            ):
                journal.prepare_next_event({"epoch": 1, "loss": float("nan")})

            active = self.commit_epoch(journal, 1)
            active.metrics_path.write_bytes(
                active.metrics_path.read_bytes().rstrip(b"\n")
            )
            with self.assertRaisesRegex(
                journal_module.ExactEpochJournalError,
                "newline terminated",
            ):
                journal.load_active()

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.mkdir()
            linked_root = Path(directory) / "linked"
            linked_root.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                journal_module.ExactEpochJournalError,
                "symlink",
            ):
                journal_module.ExactEpochJournal(linked_root)


if __name__ == "__main__":
    unittest.main()
