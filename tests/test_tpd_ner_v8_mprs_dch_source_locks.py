from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import freeze_tpd_ner_v8_mprs_dch_source_locks as subject
from experiments import train_tpd_ner_v8_mprs_dch_exact as exact


SYNTHETIC_CONTRACT = {
    "epochs": 800,
    "eval_every": 1,
    "workers": 0,
    "amp": False,
    "eps": 1e-6,
    "cublas_workspace_config": ":4096:8",
    "initialization_modes": ["fresh", "exact_resume"],
    "training_seed": 42,
    "split_seed": 20260722,
    "candidate_variants": list(subject.VARIANTS),
    "relay_identity_source": "candidate_variant_suffix",
    "relay_width": 8,
    "relay_initialization_seed": 42,
    "scheduler_restore": (
        "identity_bound_manual_schedule_from_completed_epoch"
    ),
}


def make_dataset(root: Path) -> Path:
    dataset_dir = root / "datasets"
    dataset_root = dataset_dir / "NUDT-SIRST"
    for name in ("img_idx", "images", "masks"):
        (dataset_root / name).mkdir(parents=True)
    (
        dataset_root / "img_idx" / "train_NUDT-SIRST.txt"
    ).write_bytes(b"sample_b\nsample_a\n")
    for identifier in ("sample_b", "sample_a"):
        (dataset_root / "images" / f"{identifier}.png").write_bytes(
            f"image:{identifier}".encode("utf-8")
        )
        (dataset_root / "masks" / f"{identifier}.png").write_bytes(
            f"mask:{identifier}".encode("utf-8")
        )
    return dataset_dir


class TPDNERV8MPRSDCHSourceLockTests(unittest.TestCase):
    def test_training_sources_match_exact_runtime(self) -> None:
        expected = tuple(
            str(path.resolve().relative_to(exact.REPO_ROOT.resolve()))
            for path in exact.RUNTIME_SOURCE_PATHS
        )
        self.assertEqual(subject.training_source_relatives(), expected)
        self.assertEqual(len(expected), 32)
        for relative in (
            "model/tpd_ner_v8_mprs_dch.py",
            "model/tpd_sctransnet.py",
            "model/tpd_relay.py",
            "model/tpd_clean.py",
            "experiments/train_tpd_ner_v8_mprs_dch.py",
            "experiments/train_tpd_ner_v8_mprs_dch_exact.py",
            "experiments/TPD_NER_V8_MPRS_DCH_PROTOCOL.md",
        ):
            with self.subTest(relative=relative):
                self.assertIn(relative, expected)

    def test_acceptance_sources_cover_execution_and_evaluation(self) -> None:
        expected = set(subject.ACCEPTANCE_SOURCE_RELATIVES)
        for relative in (
            "experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py",
            "experiments/smoke_tpd_ner_v8_mprs_dch.py",
            "experiments/launch_tpd_ner_v8_mprs_dch_formal800_2x5090.sh",
            "experiments/run_tpd_ner_v8_mprs_dch_formal800_2x5090_lane.sh",
            "experiments/freeze_tpd_ner_v8_mprs_dch_source_locks.py",
            "experiments/TPD_NER_V8_MPRS_DCH_PROTOCOL.md",
        ):
            with self.subTest(relative=relative):
                self.assertIn(relative, expected)

    def test_synthetic_freeze_verify_and_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "analysis").mkdir()
            runtime = root / "model/runtime.py"
            acceptance_source = root / "analysis/accept.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")
            acceptance_source.write_text("VALUE = 2\n", encoding="utf-8")
            dataset_dir = make_dataset(root)
            training_path = root / "training.json"
            acceptance_path = root / "acceptance.json"

            subject.publish_new_lock(
                training_path,
                subject.build_training_lock(
                    repo_root=root,
                    dataset_dir=dataset_dir,
                    source_relatives=("model/runtime.py",),
                    contract=SYNTHETIC_CONTRACT,
                ),
            )
            subject.publish_new_lock(
                acceptance_path,
                subject.build_acceptance_lock(
                    training_path,
                    repo_root=root,
                    source_relatives=("analysis/accept.py",),
                ),
            )
            training = subject.verify_training_lock(
                training_path,
                repo_root=root,
                dataset_dir=dataset_dir,
                expected_source_relatives=("model/runtime.py",),
                expected_contract=SYNTHETIC_CONTRACT,
            )
            acceptance = subject.verify_acceptance_lock(
                acceptance_path,
                training_path,
                repo_root=root,
                expected_source_relatives=("analysis/accept.py",),
            )
            self.assertEqual(training["official_training_sample_count"], 2)
            self.assertEqual(training["policy"]["training_seed"], 42)
            self.assertFalse(training["policy"]["multi_seed_scheduled"])
            self.assertEqual(
                acceptance["training_source_lock_sha256"],
                subject.file_sha256(training_path),
            )

            runtime.write_text("VALUE = 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source digests"):
                subject.verify_training_lock(
                    training_path,
                    repo_root=root,
                    dataset_dir=dataset_dir,
                    expected_source_relatives=("model/runtime.py",),
                    expected_contract=SYNTHETIC_CONTRACT,
                )

    def test_existing_manifest_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            subject.publish_new_lock(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                subject.publish_new_lock(path, {"value": 2})

            target = root / "target.py"
            alias = root / "alias.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                subject.hash_sources(root, ("alias.py",))

    def test_live_data_contract_and_single_seed_formal_contract(self) -> None:
        data = subject.training_data_contract(subject.REPO_ROOT / "datasets")
        self.assertEqual(data["official_training_sample_count"], 663)
        self.assertEqual(
            data["training_data_sha256"],
            "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e",
        )
        contract = subject.formal_contract()
        self.assertEqual(contract["training_seed"], 42)
        self.assertEqual(contract["split_seed"], 20260722)
        self.assertEqual(
            tuple(contract["candidate_variants"]),
            subject.VARIANTS,
        )


if __name__ == "__main__":
    unittest.main()
