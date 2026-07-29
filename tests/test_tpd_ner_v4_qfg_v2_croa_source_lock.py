from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments import (
    freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock as freezer,
)
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as trainer


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


class FreezeV4QFGV2CROAExactSourceLockTests(unittest.TestCase):
    def test_cli_has_three_exclusive_actions_and_trainer_defaults(
        self,
    ) -> None:
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

    def test_runtime_sources_are_unique_repo_files_and_exclude_freezer(
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
        self.assertIn(
            "model/tpd_frequency_gate_v2_croa.py",
            records,
        )
        self.assertIn(
            (
                "model/tpd_ner_v8_mprs_dch_v4_tail_aware_"
                "qfg_v2_croa_survival.py"
            ),
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

    def test_payload_binds_data_sources_formal_stats_parent_and_arms(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = make_tiny_dataset(Path(directory))
            first = freezer.build_source_lock_payload(
                dataset_dir=dataset_dir,
            )
            second = freezer.build_source_lock_payload(
                dataset_dir=dataset_dir,
            )
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], trainer.EXACT_SOURCE_LOCK_SCHEMA)
        self.assertEqual(first["lock_kind"], "training")
        self.assertEqual(
            first["variants"],
            [trainer.QFG_ONLY_VARIANT, trainer.TSS_QFG_VARIANT],
        )
        self.assertEqual(first["qfg_variant"], trainer.QFG_VARIANT)
        self.assertEqual(
            first["tss_variants"],
            dict(trainer.FORMAL_TSS_VARIANTS),
        )
        self.assertEqual(first["formal_contract"], trainer.formal_contract())
        self.assertEqual(first["official_training_sample_count"], 2)
        self.assertRegex(first["training_data_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            first["source_count"],
            len(trainer.RUNTIME_SOURCE_PATHS),
        )
        self.assertEqual(
            first["survival_target_statistics_sha256"],
            trainer.file_sha256(trainer.DEFAULT_TARGET_STATISTICS_PATH),
        )
        self.assertEqual(
            first["parent_checkpoint_sha256"],
            trainer.PARENT_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            first["parent_checkpoint"]["state_dict_sha256"],
            trainer.PARENT_STATE_DICT_SHA256,
        )
        self.assertFalse(first["policy"]["fresh_weight_initialization"])
        self.assertTrue(
            first["policy"]["extension_parent_initialization"]
        )
        self.assertEqual(first["policy"]["physical_gpu_choices"], [2, 3])
        self.assertFalse(first["policy"]["freezer_in_runtime_source_set"])

    def test_plan_is_read_only_and_write_once_postverifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_tiny_dataset(root)
            output = root / "qfg-source-lock.json"
            plan = freezer.plan_source_lock(
                dataset_dir=dataset_dir,
                output=output,
            )
            self.assertEqual(plan["status"], "ready")
            self.assertEqual(plan["action"], "plan")
            self.assertFalse(plan["would_write"])
            self.assertFalse(output.exists())
            self.assertRegex(plan["payload_sha256"], r"^[0-9a-f]{64}$")

            result = freezer.write_source_lock_once(
                dataset_dir=dataset_dir,
                output=output,
            )
            self.assertTrue(result["post_write_verified"])
            verified = freezer.verify_source_lock(
                output,
                dataset_dir=dataset_dir,
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                verified,
            )
            consumer = trainer.source_lock_contract(
                verified["training_data_sha256"],
                output,
                trainer.DEFAULT_TARGET_STATISTICS_PATH,
            )
            self.assertEqual(
                set(consumer),
                set(freezer.EXPECTED_CONSUMER_KEYS),
            )
            self.assertEqual(
                consumer["survival_target_statistics"],
                verified["survival_target_statistics_sha256"],
            )
            self.assertEqual(
                consumer["parent_checkpoint"],
                verified["parent_checkpoint_sha256"],
            )
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                freezer.write_source_lock_once(
                    dataset_dir=dataset_dir,
                    output=output,
                )
            self.assertEqual(output.read_bytes(), original)

    def test_verify_rejects_external_binding_and_source_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_tiny_dataset(root)
            payload = freezer.build_source_lock_payload(
                dataset_dir=dataset_dir,
            )
            mutations = {
                "training_data_sha256": "0" * 64,
                "survival_target_statistics_sha256": "0" * 64,
                "parent_checkpoint_sha256": "0" * 64,
            }
            source_name = next(iter(payload["source_sha256"]))
            for key, replacement in mutations.items():
                with self.subTest(key=key):
                    output = root / f"{key}.json"
                    tampered = copy.deepcopy(payload)
                    tampered[key] = replacement
                    output.write_text(
                        json.dumps(tampered, sort_keys=True),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "payload"):
                        freezer.verify_source_lock(
                            output,
                            dataset_dir=dataset_dir,
                        )
            output = root / "source.json"
            tampered = copy.deepcopy(payload)
            tampered["source_sha256"][source_name] = "0" * 64
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
            output = root / "qfg-source-lock.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                freezer.write_source_lock_once(
                    dataset_dir=dataset_dir,
                    output=output,
                )
            self.assertEqual(target.read_bytes(), b"unchanged")


if __name__ == "__main__":
    unittest.main()
