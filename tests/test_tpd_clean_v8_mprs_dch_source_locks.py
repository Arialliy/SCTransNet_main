from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import freeze_tpd_clean_v8_mprs_dch_source_locks as subject
from experiments import train_tpd_clean_v8_mprs_dch_exact as exact


SYNTHETIC_CONTRACT = {
    "epochs": 800,
    "eval_every": 1,
    "workers": 0,
    "amp": False,
    "eps": 1e-6,
    "cublas_workspace_config": ":4096:8",
    "initialization_modes": ["fresh", "exact_resume"],
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


class TPDCleanV8MPRSDCHSourceLockTests(unittest.TestCase):
    def test_training_sources_are_exact_runtime_authority(self) -> None:
        expected = tuple(
            str(
                path.resolve().relative_to(
                    exact.REPO_ROOT.resolve()
                )
            )
            for path in exact.RUNTIME_SOURCE_PATHS
        )
        self.assertEqual(subject.training_source_relatives(), expected)
        self.assertIn(
            "model/tpd_clean_v8_mprs_dch.py",
            expected,
        )
        self.assertIn(
            "experiments/train_tpd_clean_v8_mprs_dch_exact.py",
            expected,
        )
        self.assertIn(
            "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
            expected,
        )
        self.assertIn(
            "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
            expected,
        )

    def test_acceptance_scope_covers_every_preflight_consumer(self) -> None:
        expected = set(subject.ACCEPTANCE_SOURCE_RELATIVES)
        for relative in (
            "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
            "analysis/analyze_tpd_clean_v8_mprs_mechanism.py",
            "analysis/benchmark_tpd_clean_v8_mprs_dch.py",
            "experiments/smoke_tpd_clean_v8_mprs_dch.py",
            "experiments/launch_tpd_clean_v8_mprs_dch_formal800_2x5090.sh",
            "experiments/run_tpd_clean_v8_mprs_dch_formal800_2x5090_lane.sh",
            "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
            "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
        ):
            with self.subTest(relative=relative):
                self.assertIn(relative, expected)

    def test_synthetic_freeze_and_verify_are_separate_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "analysis").mkdir()
            (root / "model/runtime.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )
            (root / "analysis/accept.py").write_text(
                "VALUE = 2\n",
                encoding="utf-8",
            )
            dataset_dir = make_dataset(root)
            training_path = root / "training-lock.json"
            acceptance_path = root / "acceptance-lock.json"

            training = subject.build_training_lock(
                repo_root=root,
                dataset_dir=dataset_dir,
                source_relatives=("model/runtime.py",),
                contract=SYNTHETIC_CONTRACT,
            )
            subject.publish_new_lock(training_path, training)
            acceptance = subject.build_acceptance_lock(
                training_path,
                repo_root=root,
                source_relatives=("analysis/accept.py",),
            )
            subject.publish_new_lock(acceptance_path, acceptance)

            verified_training = subject.verify_training_lock(
                training_path,
                repo_root=root,
                dataset_dir=dataset_dir,
                expected_source_relatives=("model/runtime.py",),
                expected_contract=SYNTHETIC_CONTRACT,
            )
            verified_acceptance = subject.verify_acceptance_lock(
                acceptance_path,
                training_path,
                repo_root=root,
                expected_source_relatives=("analysis/accept.py",),
            )
            self.assertEqual(
                verified_training["official_training_sample_count"],
                2,
            )
            self.assertEqual(
                verified_acceptance[
                    "training_source_lock_sha256"
                ],
                subject.file_sha256(training_path),
            )
            self.assertNotEqual(
                set(verified_training["source_sha256"]),
                set(verified_acceptance["source_sha256"]),
            )

            with self.assertRaises(FileExistsError):
                subject.publish_new_lock(training_path, training)

    def test_verify_rejects_changed_source_and_training_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "analysis").mkdir()
            runtime = root / "model/runtime.py"
            acceptance_source = root / "analysis/accept.py"
            runtime.write_text("VALUE = 1\n", encoding="utf-8")
            acceptance_source.write_text("VALUE = 2\n", encoding="utf-8")
            dataset_dir = make_dataset(root)
            training_path = root / "training-lock.json"
            acceptance_path = root / "acceptance-lock.json"
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

            acceptance_source.write_text(
                "VALUE = 3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source digests"):
                subject.verify_acceptance_lock(
                    acceptance_path,
                    training_path,
                    repo_root=root,
                    expected_source_relatives=("analysis/accept.py",),
                )

            runtime.write_text("VALUE = 4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source digests"):
                subject.verify_training_lock(
                    training_path,
                    repo_root=root,
                    dataset_dir=dataset_dir,
                    expected_source_relatives=("model/runtime.py",),
                    expected_contract=SYNTHETIC_CONTRACT,
                )

    def test_source_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "alias.py").symlink_to(root / "real.py")
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                subject.hash_sources(root, ("alias.py",))

    def test_live_data_contract_matches_frozen_v7_training_data(self) -> None:
        contract = subject.training_data_contract(
            subject.REPO_ROOT / "datasets"
        )
        self.assertEqual(
            contract["official_training_sample_count"],
            663,
        )
        self.assertEqual(
            contract["training_data_sha256"],
            "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e",
        )


if __name__ == "__main__":
    unittest.main()
