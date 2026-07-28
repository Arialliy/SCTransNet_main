from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from experiments import (
    freeze_tpd_ner_v8_mprs_dch_v2_source_locks as v2_freeze,
)
from experiments import (
    freeze_tpd_ner_v8_mprs_dch_v3_source_locks as subject,
)
from experiments import train_tpd_ner_v8_mprs_dch_v3_exact as exact


EXPECTED_ACCEPTANCE_SOURCE_RELATIVES = (
    "experiments/TPD_NER_V8_MPRS_DCH_V3_PROTOCOL.md",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v3_pd_fa.py",
    "experiments/postprocess_tpd_ner_v8_mprs_dch_v3_formal800.py",
    "experiments/smoke_tpd_ner_v8_mprs_dch_v3.py",
    "experiments/handoff_tpd_ner_v8_v2_to_v3.py",
    "experiments/launch_tpd_ner_v8_mprs_dch_v3_formal800_1x5090.sh",
    "experiments/run_tpd_ner_v8_mprs_dch_v3_formal800_1x5090_lane.sh",
    "experiments/freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v2_pd_fa.py",
    "experiments/postprocess_tpd_ner_v8_mprs_dch_v2_formal800.py",
    "experiments/handoff_tpd_ner_v8_v1_to_v2.py",
    "experiments/freeze_tpd_ner_v8_mprs_dch_v2_source_locks.py",
    "experiments/postprocess_tpd_ner_v8_mprs_dch_formal800.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_sctransnet_baseline_reference_closed_interval.py",
    "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py",
)

EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES = (
    "experiments/freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py",
)


def make_dataset(root: Path) -> Path:
    dataset_dir = root / "datasets"
    dataset_root = dataset_dir / subject.DATASET
    for name in ("img_idx", "images", "masks"):
        (dataset_root / name).mkdir(parents=True)
    (
        dataset_root / "img_idx" / f"train_{subject.DATASET}.txt"
    ).write_bytes(b"sample_b\nsample_a\n")
    for identifier in ("sample_b", "sample_a"):
        (dataset_root / "images" / f"{identifier}.png").write_bytes(
            f"image:{identifier}".encode("utf-8")
        )
        (dataset_root / "masks" / f"{identifier}.png").write_bytes(
            f"mask:{identifier}".encode("utf-8")
        )
    return dataset_dir


class V3SourceLockContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.temporary_root = Path(cls._temporary.name)
        cls.training_path = cls.temporary_root / "v3-training.json"
        cls.acceptance_path = cls.temporary_root / "v3-acceptance.json"
        cls.training_relatives = subject.training_source_relatives()

        cls.training_payload = subject.build_training_lock(
            source_relatives=cls.training_relatives,
        )
        subject.publish_new_lock(
            cls.training_path,
            cls.training_payload,
        )
        cls.acceptance_payload = subject.build_acceptance_lock(
            cls.training_path,
            source_relatives=EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES,
        )
        subject.publish_new_lock(
            cls.acceptance_path,
            cls.acceptance_payload,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def write_payload(
        self,
        name: str,
        payload: Mapping[str, Any],
    ) -> Path:
        path = self.temporary_root / name
        path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def test_single_seed_single_variant_and_comparison_contract(self) -> None:
        formal = subject.formal_contract()
        self.assertEqual(formal, exact.formal_contract())
        self.assertEqual(
            tuple(formal["candidate_variants"]),
            (exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,),
        )
        self.assertEqual(formal["training_seed"], 42)
        self.assertEqual(formal["split_seed"], 20260722)
        self.assertFalse(formal["multi_seed_scheduled"])
        self.assertEqual(formal["required_control"], subject.V1_CONTROL)
        self.assertEqual(
            formal["structural_predecessor"],
            subject.V2_PREDECESSOR,
        )
        self.assertFalse(formal["relay_off_retrained"])

        self.assertEqual(
            subject.TRAINING_SCHEMA,
            exact.EXACT_SOURCE_LOCK_SCHEMA,
        )
        self.assertEqual(subject.CANDIDATE_FAMILY, exact.CANDIDATE_FAMILY)
        self.assertEqual(
            subject.DEFAULT_TRAINING_LOCK,
            exact.DEFAULT_EXACT_SOURCE_LOCK_PATH,
        )
        self.assertEqual(
            subject.VARIANTS,
            exact.supported_candidate_variants(),
        )

    def test_training_runtime_is_the_exact_36_file_closure(self) -> None:
        raw_paths = exact.RUNTIME_SOURCE_PATHS
        relatives = subject.training_source_relatives()
        expected = tuple(
            str(path.resolve().relative_to(exact.REPO_ROOT.resolve()))
            for path in raw_paths
        )
        self.assertEqual(relatives, expected)
        self.assertEqual(len(raw_paths), 36)
        self.assertEqual(len(relatives), 36)
        self.assertEqual(len(set(path.resolve() for path in raw_paths)), 36)
        self.assertEqual(len(set(relatives)), 36)

        required = {
            "experiments/train_tpd_ner_v8_mprs_dch_v3_exact.py",
            "experiments/TPD_NER_V8_MPRS_DCH_V3_PROTOCOL.md",
            "model/tpd_ner_v8_mprs_dch_v3.py",
            "model/tpd_ner_v8_mprs_dch_v2.py",
            "experiments/train_tpd_ner_v8_mprs_dch_exact.py",
            "model/tpd_ner_v8_mprs_dch.py",
        }
        self.assertTrue(required.issubset(relatives))
        for path, relative in zip(raw_paths, relatives):
            with self.subTest(relative=relative):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(
                    path.resolve(),
                    (subject.REPO_ROOT / relative).resolve(),
                )

    def test_acceptance_sources_are_the_complete_19_file_closure(
        self,
    ) -> None:
        relatives = subject.ACCEPTANCE_SOURCE_RELATIVES
        self.assertEqual(relatives, EXPECTED_ACCEPTANCE_SOURCE_RELATIVES)
        self.assertEqual(len(relatives), 19)
        self.assertEqual(len(relatives), len(set(relatives)))
        for relative in relatives:
            with self.subTest(relative=relative):
                path = subject.REPO_ROOT / relative
                self.assertTrue(path.is_file(), relative)
                self.assertFalse(path.is_symlink(), relative)

    def test_performance_gate_is_six_components_and_two_4_of_5_pairs(
        self,
    ) -> None:
        contract = subject.performance_gate_contract()
        self.assertEqual(
            contract["all_required_components"],
            [
                "pd_primary_absolute",
                "miou_secondary_absolute",
                "pd_primary_v3_vs_v1",
                "miou_secondary_v3_vs_v1",
                "pd_primary_v3_vs_v2",
                "miou_secondary_v3_vs_v2",
            ],
        )
        pairs = {
            "paired_v3_on_vs_v1_off_each_checkpoint_role": (
                subject.V1_CONTROL
            ),
            "paired_v3_on_vs_v2_on_each_checkpoint_role": (
                subject.V2_PREDECESSOR
            ),
        }
        for name, reference in pairs.items():
            with self.subTest(name=name):
                paired = contract[name]
                self.assertEqual(paired["reference"], reference)
                self.assertEqual(
                    paired["candidate"],
                    exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
                )
                self.assertEqual(
                    paired["minimum_non_inferior_budget_count"],
                    4,
                )
                self.assertEqual(
                    paired["minimum_strictly_better_budget_count"],
                    1,
                )
                self.assertEqual(paired["budget_count"], 5)

        self.assertFalse(contract["v1_off_absolute_gate_required"])
        self.assertFalse(
            contract["v2_predecessor_absolute_gate_required"]
        )
        self.assertFalse(contract["baseline_affects_decision"])
        self.assertTrue(
            contract["tiny_pd_reported_not_independent_gate"]
        )

    def test_training_and_acceptance_round_trip_with_explicit_sources(
        self,
    ) -> None:
        training = subject.verify_training_lock(
            self.training_path,
            expected_source_relatives=self.training_relatives,
        )
        acceptance = subject.verify_acceptance_lock(
            self.acceptance_path,
            self.training_path,
            expected_source_relatives=(
                EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
            ),
        )
        self.assertEqual(training["source_count"], 36)
        self.assertEqual(acceptance["source_count"], 1)
        self.assertEqual(
            acceptance["training_source_lock_sha256"],
            subject.file_sha256(self.training_path),
        )
        self.assertEqual(
            acceptance["training_data_sha256"],
            training["training_data_sha256"],
        )

        v2_training = v2_freeze.verify_training_lock(
            subject.UPSTREAM_V2_TRAINING_LOCK
        )
        self.assertEqual(
            acceptance["upstream_v2_training_source_lock_sha256"],
            subject.file_sha256(subject.UPSTREAM_V2_TRAINING_LOCK),
        )
        self.assertEqual(
            acceptance["upstream_v2_acceptance_source_lock_sha256"],
            subject.file_sha256(subject.UPSTREAM_V2_ACCEPTANCE_LOCK),
        )
        self.assertEqual(
            acceptance["upstream_v2_training_data_sha256"],
            v2_training["training_data_sha256"],
        )
        self.assertEqual(
            acceptance["upstream_v2_training_data_sha256"],
            training["training_data_sha256"],
        )

    def test_every_training_policy_field_is_frozen(self) -> None:
        bad_values = {
            "official_test_accessed": True,
            "physical_gpu_choices": [0, 1],
            "simultaneous_v3_training_tasks": 2,
            "gpu0_gpu1_used": True,
            "training_seed": 7,
            "split_seed": 17,
            "multi_seed_scheduled": True,
            "required_control": "wrong-control",
            "structural_predecessor": "wrong-predecessor",
            "relay_off_retrained": True,
            "fresh_or_exact_resume_only": False,
            "cross_version_exact_resume_supported": True,
            "existing_manifest_overwrite_forbidden": False,
            "source_symlinks_forbidden": False,
        }
        self.assertEqual(
            set(bad_values),
            set(subject.training_policy_contract()),
        )
        for name, bad_value in bad_values.items():
            with self.subTest(name=name):
                payload = copy.deepcopy(self.training_payload)
                payload["policy"][name] = bad_value
                path = self.write_payload(
                    f"bad-training-policy-{name}.json",
                    payload,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    f"training policy differs: {name}",
                ):
                    subject.verify_training_lock(path)

        missing = copy.deepcopy(self.training_payload)
        missing["policy"].pop("training_seed")
        with self.assertRaisesRegex(ValueError, "policy field set"):
            subject.verify_training_lock(
                self.write_payload(
                    "bad-training-policy-missing.json",
                    missing,
                )
            )
        extra = copy.deepcopy(self.training_payload)
        extra["policy"]["unregistered_policy"] = True
        with self.assertRaisesRegex(ValueError, "policy field set"):
            subject.verify_training_lock(
                self.write_payload(
                    "bad-training-policy-extra.json",
                    extra,
                )
            )

    def test_every_acceptance_policy_field_is_frozen(self) -> None:
        bad_values = {
            "official_test_accessed": True,
            "training_seed": 7,
            "split_seed": 17,
            "multi_seed_scheduled": True,
            "new_sweeps": [],
            "v1_off_sweeps_read_only": False,
            "v2_predecessor_sweeps_read_only": False,
            "baseline_sweeps_read_only": False,
            "all_gate_components_required": False,
            "existing_manifest_overwrite_forbidden": False,
            "source_symlinks_forbidden": False,
        }
        self.assertEqual(
            set(bad_values),
            set(subject.acceptance_policy_contract()),
        )
        for name, bad_value in bad_values.items():
            with self.subTest(name=name):
                payload = copy.deepcopy(self.acceptance_payload)
                payload["policy"][name] = bad_value
                path = self.write_payload(
                    f"bad-acceptance-policy-{name}.json",
                    payload,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    f"acceptance policy differs: {name}",
                ):
                    subject.verify_acceptance_lock(
                        path,
                        self.training_path,
                        expected_source_relatives=(
                            EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                        ),
                    )

        missing = copy.deepcopy(self.acceptance_payload)
        missing["policy"].pop("training_seed")
        with self.assertRaisesRegex(ValueError, "policy field set"):
            subject.verify_acceptance_lock(
                self.write_payload(
                    "bad-acceptance-policy-missing.json",
                    missing,
                ),
                self.training_path,
                expected_source_relatives=(
                    EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                ),
            )
        extra = copy.deepcopy(self.acceptance_payload)
        extra["policy"]["unregistered_policy"] = True
        with self.assertRaisesRegex(ValueError, "policy field set"):
            subject.verify_acceptance_lock(
                self.write_payload(
                    "bad-acceptance-policy-extra.json",
                    extra,
                ),
                self.training_path,
                expected_source_relatives=(
                    EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                ),
            )

    def test_source_mapping_missing_and_digest_tampering_are_rejected(
        self,
    ) -> None:
        training_relative = self.training_relatives[0]
        missing_training = copy.deepcopy(self.training_payload)
        missing_training["source_sha256"].pop(training_relative)
        missing_training["source_count"] -= 1
        with self.assertRaisesRegex(ValueError, "runtime source set"):
            subject.verify_training_lock(
                self.write_payload(
                    "bad-training-source-missing.json",
                    missing_training,
                )
            )

        changed_training = copy.deepcopy(self.training_payload)
        changed_training["source_sha256"][training_relative] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source digests differ"):
            subject.verify_training_lock(
                self.write_payload(
                    "bad-training-source-digest.json",
                    changed_training,
                )
            )

        acceptance_relative = EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES[0]
        missing_acceptance = copy.deepcopy(self.acceptance_payload)
        missing_acceptance["source_sha256"].pop(acceptance_relative)
        missing_acceptance["source_count"] -= 1
        with self.assertRaisesRegex(ValueError, "acceptance source set"):
            subject.verify_acceptance_lock(
                self.write_payload(
                    "bad-acceptance-source-missing.json",
                    missing_acceptance,
                ),
                self.training_path,
                expected_source_relatives=(
                    EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                ),
            )

        changed_acceptance = copy.deepcopy(self.acceptance_payload)
        changed_acceptance["source_sha256"][
            acceptance_relative
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source digests differ"):
            subject.verify_acceptance_lock(
                self.write_payload(
                    "bad-acceptance-source-digest.json",
                    changed_acceptance,
                ),
                self.training_path,
                expected_source_relatives=(
                    EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                ),
            )

    def test_temporary_repo_rejects_missing_and_symlink_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = make_dataset(root)
            source = root / "runtime.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            contract = {"fixture": "explicit-source-contract"}
            payload = subject.build_training_lock(
                repo_root=root,
                dataset_dir=dataset_dir,
                source_relatives=("runtime.py",),
                contract=contract,
            )
            lock_path = root / "training.json"
            subject.publish_new_lock(lock_path, payload)
            verified = subject.verify_training_lock(
                lock_path,
                repo_root=root,
                dataset_dir=dataset_dir,
                expected_source_relatives=("runtime.py",),
                expected_contract=contract,
            )
            self.assertEqual(verified["source_count"], 1)

            with self.assertRaisesRegex(ValueError, "non-symlink"):
                subject.build_training_lock(
                    repo_root=root,
                    dataset_dir=dataset_dir,
                    source_relatives=("missing.py",),
                    contract=contract,
                )

            alias = root / "alias.py"
            alias.symlink_to(source.name)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                subject.build_training_lock(
                    repo_root=root,
                    dataset_dir=dataset_dir,
                    source_relatives=("alias.py",),
                    contract=contract,
                )

            lock_alias = root / "training-alias.json"
            lock_alias.symlink_to(lock_path.name)
            with self.assertRaisesRegex(
                ValueError,
                "lock must be a regular non-symlink",
            ):
                subject.verify_training_lock(
                    lock_alias,
                    repo_root=root,
                    dataset_dir=dataset_dir,
                    expected_source_relatives=("runtime.py",),
                    expected_contract=contract,
                )

    def test_v3_and_upstream_v2_bindings_reject_tampering(self) -> None:
        wrong_dataset = copy.deepcopy(self.acceptance_payload)
        wrong_dataset["dataset"] = "wrong-dataset"
        with self.assertRaisesRegex(
            ValueError,
            "acceptance source-manifest identity differs",
        ):
            subject.verify_acceptance_lock(
                self.write_payload(
                    "bad-acceptance-dataset.json",
                    wrong_dataset,
                ),
                self.training_path,
                expected_source_relatives=(
                    EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                ),
            )

        direct_bindings = (
            "training_source_lock_sha256",
            "training_data_sha256",
        )
        for name in direct_bindings:
            with self.subTest(name=name):
                payload = copy.deepcopy(self.acceptance_payload)
                payload[name] = "0" * 64
                path = self.write_payload(
                    f"bad-v3-binding-{name}.json",
                    payload,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "acceptance training binding",
                ):
                    subject.verify_acceptance_lock(
                        path,
                        self.training_path,
                        expected_source_relatives=(
                            EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                        ),
                    )

        upstream_bindings = (
            "upstream_v2_training_source_lock_sha256",
            "upstream_v2_acceptance_source_lock_sha256",
            "upstream_v2_training_data_sha256",
        )
        for name in upstream_bindings:
            with self.subTest(name=name):
                payload = copy.deepcopy(self.acceptance_payload)
                payload[name] = "0" * 64
                path = self.write_payload(
                    f"bad-v2-binding-{name}.json",
                    payload,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    f"upstream binding differs: {name}",
                ):
                    subject.verify_acceptance_lock(
                        path,
                        self.training_path,
                        expected_source_relatives=(
                            EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES
                        ),
                    )

        wrong_data = copy.deepcopy(self.training_payload)
        wrong_data["training_data_sha256"] = "0" * 64
        wrong_training_path = self.write_payload(
            "bad-v2-training-data-binding.json",
            wrong_data,
        )
        with self.assertRaisesRegex(
            ValueError,
            "training data contract differs: training_data_sha256",
        ):
            subject.build_acceptance_lock(
                wrong_training_path,
                source_relatives=EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES,
            )

        fake_training_path = self.write_payload(
            "incomplete-v3-training.json",
            {
                "schema": subject.TRAINING_SCHEMA,
                "lock_kind": "training",
                "training_data_sha256": self.training_payload[
                    "training_data_sha256"
                ],
            },
        )
        with self.assertRaisesRegex(
            ValueError,
            "training source-manifest identity differs",
        ):
            subject.build_acceptance_lock(
                fake_training_path,
                source_relatives=EXPLICIT_ACCEPTANCE_SOURCE_RELATIVES,
            )

    def test_import_does_not_publish_or_modify_default_locks(self) -> None:
        paths = (
            subject.DEFAULT_TRAINING_LOCK,
            subject.DEFAULT_ACCEPTANCE_LOCK,
        )

        def state(path: Path) -> tuple[bool, bool, int | None, int | None, str]:
            symlink = path.is_symlink()
            if not path.exists() and not symlink:
                return False, False, None, None, ""
            stat = path.lstat()
            digest = (
                subject.file_sha256(path)
                if path.is_file() and not symlink
                else ""
            )
            return True, symlink, stat.st_size, stat.st_mtime_ns, digest

        before = {str(path): state(path) for path in paths}
        importlib.reload(subject)
        after = {str(path): state(path) for path in paths}
        self.assertEqual(after, before)

    def test_freeze_is_explicit_and_never_overwrites(self) -> None:
        path = self.temporary_root / "freeze-no-overwrite.json"
        arguments = [
            "--mode",
            "freeze",
            "--kind",
            "training",
            "--dataset-dir",
            str(subject.REPO_ROOT / "datasets"),
            "--training-lock",
            str(path),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            subject.main(arguments)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                subject.main(arguments)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
