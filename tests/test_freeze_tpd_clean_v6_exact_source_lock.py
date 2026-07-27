from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import freeze_tpd_clean_v6_exact_source_lock as freezer
from experiments import train_tpd_clean_v6_exact as exact_entry


REQUIRED_SOURCES = {
    "experiments/train_tpd_clean_v6_exact.py",
    "experiments/freeze_tpd_clean_v6_exact_source_lock.py",
    "experiments/train_tpd_clean_v6.py",
    "model/tpd_clean_v6.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/smoke_tpd_clean_v6.py",
    "experiments/capture_tpd_clean_v6_smoke_report.py",
    "experiments/verify_tpd_clean_v6_smoke_reports.py",
    "experiments/run_tpd_clean_v6_formal800_2x5090_worker.sh",
    "experiments/run_tpd_clean_v6_formal800_2x5090_lane.sh",
    "experiments/launch_tpd_clean_v6_formal800_2x5090.sh",
    "experiments/status_tpd_clean_v6_formal800_2x5090.sh",
    "experiments/TPD_CLEAN_V6_PROTOCOL.md",
    "experiments/tpd_exact_runner.py",
    "experiments/tpd_exact_resume.py",
    "experiments/tpd_exact_epoch_journal.py",
    "experiments/tpd_exact_training_runtime.py",
    "experiments/tpd_extension_warm_start.py",
    "experiments/train_tpd_pilot.py",
    "experiments/fingerprint_tpd_training_data.py",
    "model/SCTransNet.py",
    "model/Config.py",
    "model/tpd.py",
    "dataset.py",
    "utils.py",
    "warmup_scheduler.py",
    "tests/test_tpd_clean_v6.py",
    "tests/test_train_tpd_clean_v6.py",
    "tests/test_evaluate_tpd_clean_v6_pd_fa.py",
    "tests/test_smoke_tpd_clean_v6.py",
    "tests/test_verify_tpd_clean_v6_smoke_reports.py",
    "tests/test_tpd_clean_v6_2x_runtime.py",
    "tests/test_train_tpd_clean_v6_exact.py",
    "tests/test_freeze_tpd_clean_v6_exact_source_lock.py",
}


def make_tiny_dataset(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "datasets"
    dataset_root = dataset_dir / freezer.DEFAULT_DATASET
    for directory in ("img_idx", "images", "masks"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    index_path = (
        dataset_root
        / "img_idx"
        / f"train_{freezer.DEFAULT_DATASET}.txt"
    )
    index_path.write_bytes(b"sample_b\nsample_a\n")
    for identifier, suffix in (("sample_b", b"b"), ("sample_a", b"a")):
        (dataset_root / "images" / f"{identifier}.png").write_bytes(
            b"image-" + suffix
        )
        (dataset_root / "masks" / f"{identifier}.png").write_bytes(
            b"mask-" + suffix
        )
    return dataset_dir, dataset_root


class FreezeCleanV6ExactSourceLockTests(unittest.TestCase):
    def test_cli_defaults_match_the_exact_entry_and_output_is_overridable(
        self,
    ) -> None:
        defaults = freezer.parse_args([])
        self.assertEqual(defaults.dataset, "NUDT-SIRST")
        self.assertEqual(
            defaults.dataset_dir,
            exact_entry.REPO_ROOT / "datasets",
        )
        self.assertEqual(
            defaults.output,
            exact_entry.DEFAULT_EXACT_SOURCE_LOCK_PATH,
        )
        custom = freezer.parse_args(["--output", "/tmp/v6-test-lock.json"])
        self.assertEqual(custom.output, Path("/tmp/v6-test-lock.json"))

    def test_payload_uses_the_exact_entry_data_contract_and_all_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir, dataset_root = make_tiny_dataset(Path(directory))
            expected_index_bytes = b"sample_b\nsample_a\n"
            expected_identifiers = ["sample_b", "sample_a"]
            expected_data_digest = (
                exact_entry.official_training_data_sha256(
                    dataset_root,
                    freezer.DEFAULT_DATASET,
                    expected_identifiers,
                    expected_index_bytes,
                )
            )
            with (
                mock.patch.object(
                    exact_entry,
                    "read_official_training_index",
                    wraps=exact_entry.read_official_training_index,
                ) as read_index,
                mock.patch.object(
                    exact_entry,
                    "official_training_data_sha256",
                    wraps=exact_entry.official_training_data_sha256,
                ) as data_digest,
            ):
                payload = freezer.build_source_lock_payload(
                    dataset_dir=dataset_dir,
                )

            read_index.assert_called_once_with(
                dataset_root,
                freezer.DEFAULT_DATASET,
            )
            data_digest.assert_called_once_with(
                dataset_root,
                freezer.DEFAULT_DATASET,
                expected_identifiers,
                expected_index_bytes,
            )
            self.assertEqual(
                payload["schema"],
                exact_entry.EXACT_SOURCE_LOCK_SCHEMA,
            )
            self.assertEqual(
                payload["variants"],
                list(exact_entry.SUPPORTED_CLEAN_V6_VARIANTS),
            )
            self.assertEqual(
                payload["formal_contract"],
                exact_entry.formal_contract(),
            )
            self.assertEqual(payload["official_training_sample_count"], 2)
            self.assertEqual(
                payload["training_data_sha256"],
                expected_data_digest,
            )

            declared_sources = {
                str(path.relative_to(exact_entry.REPO_ROOT))
                for path in exact_entry.RUNTIME_SOURCE_PATHS
            }
            self.assertEqual(set(payload["source_sha256"]), declared_sources)
            self.assertTrue(REQUIRED_SOURCES <= declared_sources)
            for relative, digest in payload["source_sha256"].items():
                with self.subTest(relative=relative):
                    self.assertEqual(
                        digest,
                        exact_entry.file_sha256(
                            exact_entry.REPO_ROOT / relative
                        ),
                    )

    def test_temporary_lock_is_accepted_and_existing_output_is_refused(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = make_tiny_dataset(root)
            output = root / "temporary-v6-exact-source-lock.json"
            written, payload = freezer.freeze_source_lock(
                dataset_dir=dataset_dir,
                output=output,
            )
            self.assertEqual(written, output)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                payload,
            )
            self.assertEqual(
                exact_entry.source_lock_contract(
                    payload["training_data_sha256"],
                    output,
                )["training_data"],
                payload["training_data_sha256"],
            )
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                freezer.freeze_source_lock(
                    dataset_dir=dataset_dir,
                    output=output,
                )
            self.assertEqual(output.read_bytes(), original)

    def test_symlink_output_is_refused_without_changing_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = make_tiny_dataset(root)
            target = root / "target.json"
            target.write_bytes(b"unchanged")
            output = root / "output.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                freezer.freeze_source_lock(
                    dataset_dir=dataset_dir,
                    output=output,
                )
            self.assertEqual(target.read_bytes(), b"unchanged")


if __name__ == "__main__":
    unittest.main()
