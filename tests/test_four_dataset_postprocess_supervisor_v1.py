from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments import four_dataset_evaluation_protocol_v1 as protocol
from experiments import supervise_four_dataset_seed42_postprocess_v1 as post


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PostprocessCommandPlanTests(unittest.TestCase):
    def test_six_stages_have_exact_output_cardinalities(self) -> None:
        stages = post.build_stages(device="cpu", workers=0)
        self.assertEqual(len(stages), 6)
        self.assertEqual(
            [len(stage.outputs) for stage in stages],
            [17, 16, 12, 16, 12, 13],
        )
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "freeze_checkpoints",
                "fixed_0_5_dataset_specific",
                "fixed_0_5_sirst3_sources",
                "pd_fa_sweep_dataset_specific",
                "pd_fa_sweep_sirst3_sources",
                "finalize_tables",
            ],
        )

    def test_every_command_uses_module_entrypoint_and_fixed_root(self) -> None:
        stages = post.build_stages(device="cuda:0", workers=0)
        for stage in stages:
            self.assertEqual(stage.command[0], str(post.PYTHON))
            self.assertEqual(stage.command[1], "-m")
            self.assertEqual(stage.command[2], stage.module)
            self.assertTrue(
                any(
                    argument.startswith(str(post.RESULTS_ROOT))
                    for argument in stage.command
                )
            )
        self.assertIn("--all-runs", stages[0].command)
        self.assertIn("--all-dataset-specific", stages[1].command)
        self.assertIn("--all", stages[2].command)
        self.assertIn("--all-dataset-specific", stages[3].command)
        self.assertIn("--all-sirst3-sources", stages[4].command)
        self.assertNotIn("--initialize-templates", stages[5].command)

    def test_only_two_checkpoint_roles_are_planned(self) -> None:
        self.assertEqual(
            protocol.CHECKPOINT_ROLES,
            ("best_miou", "best_pd"),
        )
        names = {
            path.name
            for path in post._checkpoint_paths()
        }
        self.assertEqual(
            names,
            {"best_miou.pth.tar", "best_pd.pth.tar"},
        )


class FormalTrainingGateTests(unittest.TestCase):
    def _make_complete_matrix(self, root: Path) -> None:
        for dataset in protocol.DATASETS:
            for method in protocol.METHODS:
                run = root / "runs" / dataset / method / "seed_42"
                checkpoints = run / "checkpoints"
                checkpoints.mkdir(parents=True)
                records = {}
                for role in protocol.CHECKPOINT_ROLES:
                    path = checkpoints / protocol.CHECKPOINT_FILENAMES[role]
                    path.write_bytes(f"{dataset}/{method}/{role}".encode())
                    records[role] = {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": _sha(path),
                    }
                (run / "protocol.json").write_text(
                    json.dumps({"status": "complete"}),
                    encoding="utf-8",
                )
                (run / "metrics.jsonl").write_text(
                    json.dumps({"epoch": 10}) + "\n",
                    encoding="utf-8",
                )
                (run / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "dataset": dataset,
                            "method": method,
                            "seed": 42,
                            "epochs": 1000,
                            "test_selected": True,
                            "selection_is_optimistic": True,
                            "checkpoints": records,
                        }
                    ),
                    encoding="utf-8",
                )

    def test_complete_matrix_passes_and_counts_eight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_complete_matrix(root)
            gate = post.audit_training_gate(root=root)
            self.assertTrue(gate["ready"])
            self.assertEqual(gate["complete_run_count"], 8)
            self.assertEqual(len(gate["records"]), 8)
            self.assertEqual(len(gate["gate_sha256"]), 64)

    def test_extra_checkpoint_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_complete_matrix(root)
            extra = (
                root
                / "runs"
                / "SIRST3"
                / "original"
                / "seed_42"
                / "checkpoints"
                / "last_epoch1000.pth.tar"
            )
            extra.write_bytes(b"not permitted")
            gate = post.audit_training_gate(
                root=root,
                raise_on_error=False,
            )
            self.assertFalse(gate["ready"])
            self.assertEqual(gate["complete_run_count"], 7)
            self.assertTrue(
                any(
                    "checkpoint directory entries differ" in error
                    for error in gate["errors"]
                )
            )

    def test_incomplete_summary_is_rejected_without_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_complete_matrix(root)
            summary = (
                root
                / "runs"
                / "NUDT-SIRST"
                / "final"
                / "seed_42"
                / "summary.json"
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["status"] = "running"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(post.PostprocessError):
                post.audit_training_gate(root=root)


if __name__ == "__main__":
    unittest.main()
