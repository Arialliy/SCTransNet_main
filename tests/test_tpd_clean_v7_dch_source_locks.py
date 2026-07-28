from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import freeze_tpd_clean_v7_dch_source_locks as subject
from experiments import train_tpd_clean_v7_dch_exact as exact_entry


SYNTHETIC_FORMAL_CONTRACT = {
    "epochs": 800,
    "eval_every": 1,
    "workers": 0,
    "amp": False,
    "eps": 1e-6,
    "cublas_workspace_config": ":4096:8",
    "initialization_modes": ["fresh", "exact_resume"],
}
SYNTHETIC_DATA_CONTRACT = {
    "dataset": "NUDT-SIRST",
    "official_training_index": "img_idx/train_NUDT-SIRST.txt",
    "official_training_sample_count": 2,
    "training_data_sha256": "a" * 64,
}


def make_tiny_dataset(root: Path) -> Path:
    dataset_dir = root / "datasets"
    dataset_root = dataset_dir / "NUDT-SIRST"
    for name in ("img_idx", "images", "masks"):
        (dataset_root / name).mkdir(parents=True, exist_ok=True)
    (
        dataset_root / "img_idx" / "train_NUDT-SIRST.txt"
    ).write_bytes(b"sample_b\nsample_a\n")
    for identifier, suffix in (("sample_b", b"b"), ("sample_a", b"a")):
        (dataset_root / "images" / f"{identifier}.png").write_bytes(
            b"image-" + suffix
        )
        (dataset_root / "masks" / f"{identifier}.png").write_bytes(
            b"mask-" + suffix
        )
    return dataset_dir


class TPDCleanV7DCHSourceLockTests(unittest.TestCase):
    def test_current_governance_is_diagnostic_v3_acceptance_v4(self) -> None:
        self.assertEqual(
            subject.SCHEMAS["diagnostic"],
            subject.DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V3,
        )
        self.assertEqual(
            subject.SCHEMAS["training"],
            "sctransnet_tpd_clean_v7_dch_exact_source_lock_v1",
        )
        self.assertEqual(
            subject.DEFAULT_LOCK_RELATIVES["diagnostic"],
            "experiments/tpd_clean_v7_dch_diagnostic_source_lock_v3.json",
        )
        self.assertEqual(
            subject.DEFAULT_LOCK_RELATIVES["training"],
            "experiments/tpd_clean_v7_dch_exact_source_lock.json",
        )
        self.assertEqual(
            subject.SCHEMAS["acceptance"],
            subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
        )
        self.assertEqual(
            subject.DEFAULT_LOCK_RELATIVES["acceptance"],
            "experiments/tpd_clean_v7_dch_acceptance_source_lock_v4.json",
        )
        self.assertEqual(
            subject.SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE,
            "experiments/tpd_clean_v7_dch_acceptance_source_lock.json",
        )
        self.assertNotEqual(
            subject.DEFAULT_LOCK_RELATIVES["acceptance"],
            subject.SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE,
        )

        diagnostic = subject.superseded_diagnostic_lock_evidence()
        self.assertEqual(
            diagnostic["observed_schema"],
            subject.DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V1,
        )
        self.assertEqual(
            diagnostic["sha256"],
            subject.SUPERSEDED_DIAGNOSTIC_LOCK_SHA256,
        )
        self.assertIsNone(diagnostic["evidence_error"])
        self.assertEqual(
            diagnostic["superseded_by_relative_path"],
            subject.SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE,
        )
        chain = diagnostic["predecessor_chain_evidence"]
        self.assertEqual(len(chain), 2)
        self.assertEqual(
            chain[0]["sha256"],
            subject.PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_SHA256,
        )
        self.assertEqual(
            chain[1]["sha256"],
            subject.DIAGNOSTIC_SUPERSESSION_RECORD_SHA256,
        )
        self.assertTrue(chain[1]["relation_verified"])
        diagnostic_v2 = subject.superseded_diagnostic_v2_lock_evidence()
        self.assertEqual(
            diagnostic_v2["observed_schema"],
            subject.DIAGNOSTIC_SOURCE_LOCK_SCHEMA_V2,
        )
        self.assertEqual(
            diagnostic_v2["sha256"],
            subject.SUPERSEDED_DIAGNOSTIC_LOCK_V2_SHA256,
        )
        self.assertIsNone(diagnostic_v2["evidence_error"])
        self.assertEqual(
            diagnostic_v2["superseded_by_relative_path"],
            subject.DEFAULT_LOCK_RELATIVES["diagnostic"],
        )

        evidence = subject.superseded_lock_evidence("acceptance")
        self.assertEqual(len(evidence), 3)
        self.assertTrue(all(item["present"] for item in evidence))
        self.assertTrue(all(item["superseded"] for item in evidence))
        self.assertTrue(
            all(not item["accepted_as_current"] for item in evidence)
        )
        self.assertEqual(
            evidence[0]["observed_schema"],
            subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V1,
        )
        self.assertEqual(
            evidence[0]["sha256"],
            "4fb4668d1eb97e3c6a28a60efbfad4ea9ac3423d98f16f7d411a09cebb5b68d7",
        )
        self.assertEqual(
            evidence[1]["observed_schema"],
            subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2,
        )
        self.assertEqual(
            evidence[1]["sha256"],
            "ee7be009081b1776b6e5068c9c39b7f4429c987a44cea0a25f7c95f27fc8f130",
        )
        self.assertEqual(
            evidence[2]["observed_schema"],
            subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3,
        )
        self.assertEqual(
            evidence[2]["sha256"],
            "f319f4b4b1cd05ad97504b8fc317e8c24abb3736d5292ec64e85647731df5a45",
        )
        self.assertTrue(
            all(item["evidence_error"] is None for item in evidence)
        )
        self.assertEqual(
            [item["superseded_by_relative_path"] for item in evidence],
            [
                subject.SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
                subject.SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE,
                subject.DEFAULT_LOCK_RELATIVES["acceptance"],
            ],
        )

    def test_training_paths_are_exact_entry_runtime_authority(self) -> None:
        observed = subject.training_source_relatives()
        expected = tuple(
            str(path.resolve().relative_to(exact_entry.REPO_ROOT.resolve()))
            for path in exact_entry.RUNTIME_SOURCE_PATHS
        )
        self.assertEqual(observed, expected)
        self.assertIn(
            "experiments/train_tpd_clean_v7_dch_exact.py",
            observed,
        )
        # The adapter eagerly reuses the V6 exact entry; it cannot be omitted
        # merely because the model and variant identity are rebound.
        self.assertIn(
            "experiments/train_tpd_clean_v6_exact.py",
            observed,
        )

    def test_scope_roots_separate_diagnostic_training_and_acceptance(self) -> None:
        diagnostic = set(subject.source_relatives("diagnostic"))
        training = set(subject.source_relatives("training"))
        acceptance_roots = set(subject.ACCEPTANCE_ROOT_RELATIVES)

        self.assertIn(
            "analysis/diagnose_tpd_clean_v6_fragmentation.py",
            diagnostic,
        )
        self.assertIn(
            "analysis/summarize_tpd_clean_v6_failure_atlas.py",
            diagnostic,
        )
        self.assertNotIn(
            "analysis/diagnose_tpd_clean_v6_fragmentation.py",
            training,
        )
        self.assertNotIn(
            "experiments/train_tpd_clean_v7_dch_exact.py",
            diagnostic,
        )
        self.assertNotIn("model/tpd_clean_v7.py", training)
        self.assertIn(
            "experiments/evaluate_tpd_clean_v7_dch_pd_fa.py",
            acceptance_roots,
        )
        self.assertIn(
            "experiments/finalize_tpd_clean_v7_dch.py",
            acceptance_roots,
        )
        self.assertIn(
            "analysis/diagnose_tpd_clean_v7_dch_mechanism.py",
            acceptance_roots,
        )
        self.assertIn(
            "experiments/TPD_CLEAN_V7_DCH_ACCEPTANCE_AMENDMENT_V1.md",
            acceptance_roots,
        )
        self.assertIn(
            "experiments/TPD_CLEAN_V7_DCH_ACCEPTANCE_AMENDMENT_V2.md",
            acceptance_roots,
        )
        archived = subject.acceptance_frozen_input_relatives()
        self.assertEqual(len(archived), 3)
        self.assertTrue(
            all(
                "superseded_acceptance_v3_markdown_order_v1" in path
                for path in archived
            )
        )

    def test_eager_import_closure_adds_nested_local_sources_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "main.py").write_text(
                "import json\nfrom pkg import helper\n",
                encoding="utf-8",
            )
            (root / "pkg/helper.py").write_text(
                "from nested import value\n",
                encoding="utf-8",
            )
            (root / "nested.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )
            (root / "lazy.py").write_text(
                "VALUE = 2\n",
                encoding="utf-8",
            )
            with (root / "main.py").open("a", encoding="utf-8") as handle:
                handle.write(
                    "\ndef deferred():\n"
                    "    import lazy\n"
                )
            closure = subject.eager_local_import_closure(
                ("main.py",),
                repo_root=root,
            )
        self.assertEqual(
            set(closure),
            {"main.py", "pkg/helper.py", "nested.py"},
        )

    def test_training_payload_uses_exact_schema_and_seventeen_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "b.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            payload = subject.build_source_lock_payload(
                "training",
                repo_root=root,
                source_relatives_override=("a.py", "b.sh"),
                frozen_input_relatives_override=(),
                training_data_contract_override=SYNTHETIC_DATA_CONTRACT,
                formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
            )

        self.assertEqual(payload["schema"], exact_entry.EXACT_SOURCE_LOCK_SCHEMA)
        self.assertEqual(
            payload["variants"],
            list(exact_entry.SUPPORTED_CLEAN_V7_DCH_VARIANTS),
        )
        self.assertEqual(payload["formal_contract"], SYNTHETIC_FORMAL_CONTRACT)
        self.assertEqual(payload["training_data_sha256"], "a" * 64)
        self.assertEqual(
            tuple(payload["validation_fields"]),
            subject.VALIDATION_FIELDS,
        )
        self.assertEqual(set(payload["source_sha256"]), {"a.py", "b.sh"})
        self.assertEqual(
            payload["go_decision"]["status"],
            "GO_DCH_TRAJECTORY_TEST",
        )
        self.assertFalse(
            payload["go_decision"]["dch_causal_mechanism_established"]
        )

    def test_temporary_lock_round_trip_and_source_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            lock = root / "lock.json"
            payload = subject.build_source_lock_payload(
                "training",
                repo_root=root,
                source_relatives_override=("source.py",),
                frozen_input_relatives_override=(),
                training_data_contract_override=SYNTHETIC_DATA_CONTRACT,
                formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
            )
            subject.write_new_json(lock, payload)
            loaded, digest = subject.validate_source_lock(
                "training",
                lock,
                repo_root=root,
                source_relatives_override=("source.py",),
                frozen_input_relatives_override=(),
                training_data_contract_override=SYNTHETIC_DATA_CONTRACT,
                formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
            )
            self.assertEqual(loaded, payload)
            self.assertEqual(digest, subject.file_sha256(lock))

            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                subject.validate_source_lock(
                    "training",
                    lock,
                    repo_root=root,
                    source_relatives_override=("source.py",),
                    frozen_input_relatives_override=(),
                    training_data_contract_override=(
                        SYNTHETIC_DATA_CONTRACT
                    ),
                    formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
                )

    def test_acceptance_v1_v2_v3_are_rejected_as_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )
            evidence = subject.superseded_lock_evidence("acceptance")
            payload = subject.build_source_lock_payload(
                "acceptance",
                repo_root=root,
                source_relatives_override=("source.py",),
                frozen_input_relatives_override=(),
                superseded_lock_evidence_override=evidence,
                formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
            )
            self.assertEqual(
                payload["schema"],
                subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
            )

            current = root / "current-v4.json"
            subject.write_new_json(current, payload)
            loaded, _ = subject.validate_source_lock(
                "acceptance",
                current,
                repo_root=root,
                source_relatives_override=("source.py",),
                frozen_input_relatives_override=(),
                superseded_lock_evidence_override=evidence,
                formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
            )
            self.assertEqual(loaded, payload)

            for version, schema in (
                ("v1", subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V1),
                ("v2", subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V2),
                ("v3", subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V3),
            ):
                old_payload = dict(payload)
                old_payload["schema"] = schema
                old_lock = root / f"acceptance-{version}.json"
                subject.write_new_json(old_lock, old_payload)
                with self.assertRaisesRegex(
                    ValueError,
                    "expected .*acceptance_source_lock_v4.*observed "
                    f".*acceptance_source_lock_{version}",
                ):
                    subject.validate_source_lock(
                        "acceptance",
                        old_lock,
                        repo_root=root,
                        source_relatives_override=("source.py",),
                        frozen_input_relatives_override=(),
                        superseded_lock_evidence_override=evidence,
                        formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
                    )

            for relative in (
                subject.SUPERSEDED_ACCEPTANCE_LOCK_RELATIVE,
                subject.SUPERSEDED_ACCEPTANCE_LOCK_V2_RELATIVE,
                subject.SUPERSEDED_ACCEPTANCE_LOCK_V3_RELATIVE,
            ):
                superseded = root / relative
                superseded.parent.mkdir(parents=True, exist_ok=True)
                subject.write_new_json(superseded, payload)
                with self.assertRaisesRegex(ValueError, "superseded"):
                    subject.validate_source_lock(
                        "acceptance",
                        superseded,
                        repo_root=root,
                        source_relatives_override=("source.py",),
                        frozen_input_relatives_override=(),
                        superseded_lock_evidence_override=evidence,
                        formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
                    )

    def test_diagnostic_v1_v2_paths_are_rejected_as_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                subject.SUPERSEDED_DIAGNOSTIC_LOCK_RELATIVE,
                subject.SUPERSEDED_DIAGNOSTIC_LOCK_V2_RELATIVE,
                subject.PRE_ACCEPTANCE_V2_DIAGNOSTIC_LOCK_RELATIVE,
                subject.DIAGNOSTIC_SUPERSESSION_RECORD_RELATIVE,
            ):
                superseded = root / relative
                superseded.parent.mkdir(parents=True, exist_ok=True)
                superseded.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "superseded.*current v3",
                ):
                    subject.validate_source_lock(
                        "diagnostic",
                        superseded,
                        repo_root=root,
                        source_relatives_override=("source.py",),
                        frozen_input_relatives_override=(),
                        formal_contract_override=SYNTHETIC_FORMAL_CONTRACT,
                    )

    def test_real_training_payload_is_consumed_by_dch_exact_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_tiny_dataset(root)
            lock = root / "dch-exact-source-lock.json"
            payload = subject.build_source_lock_payload(
                "training",
                dataset_dir=dataset_dir,
            )
            subject.write_new_json(lock, payload)
            contract = exact_entry.source_lock_contract(
                payload["training_data_sha256"],
                lock,
            )
        self.assertEqual(
            contract["training_data"],
            payload["training_data_sha256"],
        )
        self.assertIn(
            "exact_source:experiments/train_tpd_clean_v6_exact.py",
            contract,
        )

    def test_exclusive_writer_refuses_existing_file_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.json"
            existing.write_bytes(b"unchanged")
            with self.assertRaises(FileExistsError):
                subject.write_new_json(existing, {"value": 1})
            self.assertEqual(existing.read_bytes(), b"unchanged")

            target = root / "target.json"
            target.write_bytes(b"target")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                subject.write_new_json(link, {"value": 1})
            self.assertEqual(target.read_bytes(), b"target")

    def test_readiness_is_read_only_and_reports_future_acceptance_files(self) -> None:
        before = set(
            Path(subject.REPO_ROOT / "experiments").glob(
                "tpd_clean_v7_dch_*_source_lock.json"
            )
        )
        readiness = subject.source_lock_readiness("acceptance")
        after = set(
            Path(subject.REPO_ROOT / "experiments").glob(
                "tpd_clean_v7_dch_*_source_lock.json"
            )
        )
        self.assertEqual(before, after)
        self.assertEqual(readiness["writes_performed"], 0)
        self.assertEqual(
            readiness["expected_schema"],
            subject.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
        )
        self.assertEqual(
            readiness["default_lock_relative"],
            "experiments/tpd_clean_v7_dch_acceptance_source_lock_v4.json",
        )
        self.assertEqual(len(readiness["superseded_lock_evidence"]), 3)
        self.assertTrue(readiness["superseded_lock_evidence_complete"])
        self.assertTrue(
            all(
                not item["accepted_as_current"]
                for item in readiness["superseded_lock_evidence"]
            )
        )
        diagnostic = subject.source_lock_readiness("diagnostic")
        training = subject.source_lock_readiness("training")
        self.assertEqual(diagnostic["source_count"], 18)
        self.assertEqual(training["source_count"], 24)
        self.assertEqual(readiness["source_count"], 40)
        self.assertEqual(readiness["frozen_input_count"], 3)
        self.assertEqual(len(diagnostic["superseded_lock_evidence"]), 2)
        self.assertTrue(diagnostic["superseded_lock_evidence_complete"])
        self.assertIn("ready", readiness)
        if not readiness["ready"]:
            self.assertTrue(
                readiness["missing_source_paths"]
                or readiness["declaration_error"]
            )


if __name__ == "__main__":
    unittest.main()
