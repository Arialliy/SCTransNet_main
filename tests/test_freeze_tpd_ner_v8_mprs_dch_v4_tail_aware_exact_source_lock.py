from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments import (
    freeze_tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock as freezer,
)
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact as trainer,
)


def make_tiny_dataset(root: Path) -> Path:
    dataset_dir = root / "datasets"
    dataset_root = dataset_dir / freezer.DATASET
    for directory in ("img_idx", "images", "masks"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    (
        dataset_root
        / "img_idx"
        / f"train_{freezer.DATASET}.txt"
    ).write_bytes(b"sample_b\nsample_a\n")
    for identifier, suffix in (("sample_b", b"b"), ("sample_a", b"a")):
        (dataset_root / "images" / f"{identifier}.png").write_bytes(
            b"image-" + suffix
        )
        (dataset_root / "masks" / f"{identifier}.png").write_bytes(
            b"mask-" + suffix
        )
    return dataset_dir


class FreezeV4TailAwareExactSourceLockTests(unittest.TestCase):
    def test_cli_has_exactly_three_actions_and_trainer_default(self) -> None:
        for option, attribute in (
            ("--plan", "plan"),
            ("--write-once", "write_once"),
            ("--verify", "verify"),
        ):
            with self.subTest(option=option):
                args = freezer.parse_args([option])
                self.assertTrue(getattr(args, attribute))
                self.assertEqual(
                    args.output,
                    trainer.DEFAULT_EXACT_SOURCE_LOCK_PATH,
                )
                self.assertEqual(
                    args.dataset_dir,
                    trainer.REPO_ROOT / "datasets",
                )
        for argv in (
            [],
            ["--plan", "--verify"],
            ["--plan", "--write-once"],
            ["--verify", "--write-once"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    freezer.parse_args(argv)

    def test_runtime_sources_are_unique_regular_repo_files_and_exclude_freezer(
        self,
    ) -> None:
        records = freezer.runtime_source_sha256()
        declared = {
            str(path.resolve().relative_to(trainer.REPO_ROOT))
            for path in trainer.RUNTIME_SOURCE_PATHS
        }
        self.assertEqual(set(records), declared)
        self.assertNotIn(
            str(freezer.FREEZER_PATH.relative_to(trainer.REPO_ROOT)),
            records,
        )
        for relative, digest in records.items():
            with self.subTest(relative=relative):
                path = trainer.REPO_ROOT / relative
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

        duplicate = Path(trainer.__file__).resolve()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freezer.runtime_source_sha256(
                source_paths=(duplicate, duplicate),
            )
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "outside.py"
            outside.write_text("value = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                freezer.runtime_source_sha256(source_paths=(outside,))
            symlink = Path(directory) / "link.py"
            symlink.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                freezer.runtime_source_sha256(source_paths=(symlink,))

    def test_payload_binds_data_sources_contract_and_formula_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = make_tiny_dataset(Path(directory))
            payload = freezer.build_source_lock_payload(
                dataset_dir=dataset_dir,
            )
        self.assertEqual(payload["schema"], trainer.EXACT_SOURCE_LOCK_SCHEMA)
        self.assertEqual(payload["lock_kind"], "training")
        self.assertEqual(
            payload["variants"],
            list(trainer.supported_candidate_variants()),
        )
        self.assertEqual(
            payload["formal_contract"],
            trainer.formal_contract(),
        )
        self.assertEqual(payload["official_training_sample_count"], 2)
        self.assertRegex(
            payload["training_data_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            payload["source_count"],
            len(trainer.RUNTIME_SOURCE_PATHS),
        )
        self.assertEqual(
            payload["formula_selection"]["aggregate_sha256"],
            trainer.FORMULA_SELECTION_AGGREGATE_SHA256,
        )
        self.assertEqual(
            payload["formula_selection"]["completion_marker_sha256"],
            trainer.FORMULA_SELECTION_MARKER_SHA256,
        )
        self.assertEqual(
            payload["formula_selection"]["selected_formula_mode"],
            "complement_tail",
        )
        self.assertFalse(
            payload["formula_selection"][
                "formal_training_authorized_by_selection_artifact"
            ]
        )
        self.assertFalse(payload["policy"]["freezer_in_runtime_source_set"])

    def test_plan_is_read_only_and_reports_deterministic_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_tiny_dataset(root)
            output = root / "v4-source-lock.json"
            first = freezer.plan_source_lock(
                dataset_dir=dataset_dir,
                output=output,
            )
            second = freezer.plan_source_lock(
                dataset_dir=dataset_dir,
                output=output,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "ready")
            self.assertEqual(first["action"], "plan")
            self.assertFalse(first["would_write"])
            self.assertFalse(output.exists())
            self.assertRegex(first["payload_sha256"], r"^[0-9a-f]{64}$")

    def test_write_once_postverifies_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_tiny_dataset(root)
            output = root / "v4-source-lock.json"
            result = freezer.write_source_lock_once(
                dataset_dir=dataset_dir,
                output=output,
            )
            self.assertTrue(result["post_write_verified"])
            self.assertEqual(result["action"], "write-once")
            verified = freezer.verify_source_lock(
                output,
                dataset_dir=dataset_dir,
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                verified,
            )
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                freezer.write_source_lock_once(
                    dataset_dir=dataset_dir,
                    output=output,
                )
            self.assertEqual(output.read_bytes(), original)

    def test_verify_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_tiny_dataset(root)
            output = root / "v4-source-lock.json"
            payload = freezer.build_source_lock_payload(
                dataset_dir=dataset_dir,
            )
            tampered = copy.deepcopy(payload)
            tampered["formula_selection"]["selected_formula_mode"] = (
                "direct_tail"
            )
            output.write_text(
                json.dumps(tampered, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "payload"):
                freezer.verify_source_lock(
                    output,
                    dataset_dir=dataset_dir,
                )

    def test_symlink_output_is_refused_without_changing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_tiny_dataset(root)
            target = root / "target.json"
            target.write_bytes(b"unchanged")
            output = root / "v4-source-lock.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                freezer.write_source_lock_once(
                    dataset_dir=dataset_dir,
                    output=output,
                )
            self.assertEqual(target.read_bytes(), b"unchanged")


if __name__ == "__main__":
    unittest.main()
