from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock as subject,
)
from experiments import (
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)


def formal_binding(suffix: str = "0") -> dict:
    digest = suffix * 64
    binding = {
        "schema": (
            "sctransnet_tpd_ner_v8_mprs_dch_v3_"
            "dc_knockout_formal_input_binding_v2"
        ),
        "formal_training_source_lock": {
            "path": "/formal/training-lock.json",
            "sha256": "1" * 64,
            "training_data_sha256": "2" * 64,
        },
        "formal_acceptance_source_lock": {
            "path": "/formal/acceptance-lock.json",
            "sha256": "3" * 64,
            "training_source_lock_sha256": "1" * 64,
        },
        "formal_completion_marker": {
            "path": "/formal/POSTPROCESS_COMPLETE.json",
            "sha256": "4" * 64,
            "schema": "formal-marker",
            "status": "complete",
        },
        "formal_aggregate_json": {
            "path": "/formal/report.json",
            "sha256": "5" * 64,
            "schema": "formal-report",
            "status": "complete",
        },
        "formal_aggregate_markdown": {
            "path": "/formal/report.md",
            "sha256": "6" * 64,
        },
        "formal_selection_contract_repair": {
            "repair_id": subject.FORMAL_REPAIR_ID,
            "authority": "versioned_selection_contract_repair_v1_only",
            "each_variant_uses_own_selected_checkpoints": True,
            "formal_aggregate_decision": (
                subject.EXPECTED_FORMAL_DECISION
            ),
            "aggregate_full_model_gate_passed": False,
            "repair_wrapper": {
                "path": "/formal/repair.py",
                "sha256": "a" * 64,
            },
            "repair_protocol": {
                "path": "/formal/repair.md",
                "sha256": "b" * 64,
            },
            "repair_attestation": {
                "path": "/formal/attestation.json",
                "sha256": "c" * 64,
                "schema": "repair-attestation",
                "status": "frozen",
            },
            "comparison_contract_sha256": "d" * 64,
        },
        "formal_run_directory": str(spec.FORMAL_RUN_DIR.resolve()),
        "formal_checkpoints": {
            checkpoint: {
                "path": f"/formal/{checkpoint}",
                "role": spec.CHECKPOINT_ROLES[checkpoint],
                "sha256": str(index) * 64,
                "artifact_identity_sha256": str(index + 2) * 64,
            }
            for index, checkpoint in enumerate(spec.CHECKPOINTS, start=7)
        },
        "formal_sweeps": {
            checkpoint: {
                "path": f"/formal/{checkpoint}.json",
                "sha256": str(index) * 64,
            }
            for index, checkpoint in enumerate(spec.CHECKPOINTS, start=5)
        },
        "formal_outputs_read_only": True,
        "official_test_accessed": False,
        "fixture_suffix": digest,
    }
    binding["snapshot_sha256"] = spec.canonical_sha256(binding)
    return binding


class V3DCKnockoutSourceLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.relative = "diagnostic_source.py"
        (self.root / self.relative).write_text(
            "# diagnostic fixture\n",
            encoding="utf-8",
        )
        self.lock = self.root / "diagnostic-lock.json"
        self.formal = formal_binding("a")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_and_publish(self) -> dict:
        with mock.patch.object(
            subject,
            "current_formal_artifact_binding",
            return_value=copy.deepcopy(self.formal),
        ):
            payload = subject.build_source_lock(
                repo_root=self.root,
                source_relatives=(self.relative,),
            )
        subject.publish_new_lock(self.lock, payload)
        return payload

    def test_build_is_one_diagnostic_lock_without_training_authority(
        self,
    ) -> None:
        payload = self.build_and_publish()
        self.assertEqual(payload["schema"], spec.SOURCE_LOCK_SCHEMA)
        self.assertEqual(payload["lock_kind"], "diagnostic_acceptance")
        self.assertTrue(payload["diagnostic_only"])
        self.assertFalse(payload["affects_formal_gate"])
        self.assertFalse(payload["formal_decision_authority"])
        self.assertEqual(payload["formal_gate_components"], [])
        self.assertNotIn("decision", payload)
        self.assertNotIn("performance_gate_assessment", payload)
        self.assertEqual(payload["source_count"], 1)
        self.assertEqual(payload["knockout_spec"], spec.fixed_specification())
        self.assertEqual(
            payload["knockout_spec_sha256"],
            spec.specification_sha256(),
        )
        self.assertFalse(payload["policy"]["training_performed"])
        self.assertFalse(payload["policy"]["multi_seed_scheduled"])
        revision = payload["source_lock_revision"]
        self.assertEqual(revision["revision"], 2)
        self.assertEqual(
            revision["parent_source_lock"]["sha256"],
            subject.LEGACY_SOURCE_LOCK_V1_SHA256,
        )
        self.assertFalse(
            revision["failed_v1_attempt"]["inference_started"]
        )
        self.assertEqual(
            revision["failed_v1_attempt"]["published_sweep_count"],
            0,
        )
        self.assertEqual(
            revision["failed_v1_attempt"]["published_aggregate_count"],
            0,
        )
        self.assertEqual(
            revision["replacement"]["required_evaluator_environment"],
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": "42",
            },
        )
        self.assertEqual(
            payload["policy"]["parent_source_lock_sha256"],
            subject.LEGACY_SOURCE_LOCK_V1_SHA256,
        )

    def test_round_trip_and_current_binding(self) -> None:
        self.build_and_publish()
        with mock.patch.object(
            subject,
            "current_formal_artifact_binding",
            return_value=copy.deepcopy(self.formal),
        ):
            verified = subject.verify_source_lock(
                self.lock,
                repo_root=self.root,
                expected_source_relatives=(self.relative,),
            )
        self.assertEqual(verified["formal_artifact_binding"], self.formal)

        with (
            mock.patch.object(
                subject,
                "verify_source_lock",
                return_value=verified,
            ),
            mock.patch.object(
                subject,
                "file_sha256",
                return_value="f" * 64,
            ),
        ):
            binding = subject.current_source_binding(self.lock)
        self.assertEqual(binding["schema"], subject.SOURCE_BINDING_SCHEMA)
        self.assertEqual(
            binding["diagnostic_source_lock"]["sha256"],
            "f" * 64,
        )
        self.assertEqual(
            binding["knockout_spec_sha256"],
            spec.specification_sha256(),
        )
        self.assertEqual(
            binding["source_lock_revision"],
            verified["source_lock_revision"],
        )

    def test_source_mutation_is_rejected(self) -> None:
        self.build_and_publish()
        (self.root / self.relative).write_text(
            "# changed after freeze\n",
            encoding="utf-8",
        )
        with (
            mock.patch.object(
                subject,
                "current_formal_artifact_binding",
                return_value=copy.deepcopy(self.formal),
            ),
            self.assertRaisesRegex(ValueError, "source digests differ"),
        ):
            subject.verify_source_lock(
                self.lock,
                repo_root=self.root,
                expected_source_relatives=(self.relative,),
            )

    def test_formal_binding_mutation_is_rejected(self) -> None:
        self.build_and_publish()
        changed = formal_binding("b")
        with (
            mock.patch.object(
                subject,
                "current_formal_artifact_binding",
                return_value=changed,
            ),
            self.assertRaisesRegex(
                ValueError,
                "formal V3 inputs changed",
            ),
        ):
            subject.verify_source_lock(
                self.lock,
                repo_root=self.root,
                expected_source_relatives=(self.relative,),
            )

    def test_lock_identity_and_spec_tampering_are_rejected(self) -> None:
        payload = self.build_and_publish()
        for name, mutate, pattern in (
            (
                "authority",
                lambda value: value.__setitem__(
                    "formal_decision_authority",
                    True,
                ),
                "identity differs",
            ),
            (
                "spec",
                lambda value: value["knockout_spec"].__setitem__(
                    "row_count",
                    7,
                ),
                "specification differs",
            ),
            (
                "decision",
                lambda value: value.__setitem__("decision", "ACCEPT"),
                "formal decision",
            ),
            (
                "revision",
                lambda value: value["source_lock_revision"].__setitem__(
                    "repair_reason",
                    "different",
                ),
                "revision contract differs",
            ),
        ):
            with self.subTest(name=name):
                changed = copy.deepcopy(payload)
                mutate(changed)
                path = self.root / f"{name}.json"
                path.write_text(
                    json.dumps(changed, sort_keys=True),
                    encoding="utf-8",
                )
                with (
                    mock.patch.object(
                        subject,
                        "current_formal_artifact_binding",
                        return_value=copy.deepcopy(self.formal),
                    ),
                    self.assertRaisesRegex(ValueError, pattern),
                ):
                    subject.verify_source_lock(
                        path,
                        repo_root=self.root,
                        expected_source_relatives=(self.relative,),
                    )

    def test_freeze_publish_is_no_overwrite(self) -> None:
        payload = self.build_and_publish()
        original = self.lock.read_bytes()
        with self.assertRaises(FileExistsError):
            subject.publish_new_lock(self.lock, payload)
        self.assertEqual(self.lock.read_bytes(), original)

    def test_import_contract_does_not_create_default_lock(self) -> None:
        # Import has no publication call.  Once a future explicit freeze has
        # occurred, the same test remains valid by only accepting a regular
        # non-symlink file at the versioned default path.
        if (
            subject.DEFAULT_SOURCE_LOCK.exists()
            or subject.DEFAULT_SOURCE_LOCK.is_symlink()
        ):
            self.assertTrue(subject.DEFAULT_SOURCE_LOCK.is_file())
            self.assertFalse(subject.DEFAULT_SOURCE_LOCK.is_symlink())
        self.assertTrue(
            str(subject.DEFAULT_SOURCE_LOCK).endswith(
                "tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock_v2.json"
            )
        )
        self.assertTrue(subject.LEGACY_SOURCE_LOCK_V1.is_file())
        self.assertFalse(subject.LEGACY_SOURCE_LOCK_V1.is_symlink())
        self.assertEqual(
            subject.file_sha256(subject.LEGACY_SOURCE_LOCK_V1),
            subject.LEGACY_SOURCE_LOCK_V1_SHA256,
        )
        self.assertFalse(
            subject.LEGACY_OUTPUT_ROOT_V1.exists()
            or subject.LEGACY_OUTPUT_ROOT_V1.is_symlink()
        )
        self.assertNotEqual(
            subject.DEFAULT_SOURCE_LOCK,
            subject.DEFAULT_FORMAL_TRAINING_LOCK,
        )
        self.assertNotEqual(
            subject.DEFAULT_SOURCE_LOCK,
            subject.DEFAULT_FORMAL_ACCEPTANCE_LOCK,
        )

    def test_v2_revision_contract_strictly_binds_failed_v1_parent(
        self,
    ) -> None:
        revision = subject.source_lock_revision_contract()
        self.assertEqual(
            revision["schema"],
            subject.SOURCE_LOCK_REVISION_SCHEMA,
        )
        self.assertEqual(revision["revision"], 2)
        self.assertEqual(
            revision["parent_source_lock"]["path"],
            str(subject.LEGACY_SOURCE_LOCK_V1.resolve()),
        )
        self.assertEqual(
            revision["parent_source_lock"]["sha256"],
            (
                "89f98ecab9c1cbcd72f40b9ba9c2083076231ad240477d81a"
                "69528c0ef9c80f7"
            ),
        )
        self.assertIn(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            revision["repair_reason"],
        )
        self.assertFalse(revision["failed_v1_attempt"]["inference_started"])
        self.assertEqual(
            revision["failed_v1_attempt"]["published_complete_marker_count"],
            0,
        )
        with (
            mock.patch.object(subject, "_sha256", return_value="0" * 64),
            self.assertRaisesRegex(ValueError, "legacy V1.*SHA differs"),
        ):
            subject.source_lock_revision_contract()

    def test_source_closure_includes_gpu23_orchestrator(self) -> None:
        self.assertIn(
            (
                "experiments/"
                "finalize_tpd_ner_v8_mprs_dch_v3_dc_knockout_gpu23.py"
            ),
            subject.DIAGNOSTIC_SOURCE_RELATIVES,
        )

    def test_only_versioned_repaired_aggregate_is_the_default_authority(
        self,
    ) -> None:
        self.assertEqual(
            subject.DEFAULT_FORMAL_MARKER,
            subject.repaired_formal.REPAIR_COMPLETE_MARKER,
        )
        self.assertEqual(
            subject.DEFAULT_FORMAL_REPORT,
            subject.repaired_formal.REPAIR_JSON_OUTPUT,
        )
        self.assertEqual(
            subject.DEFAULT_FORMAL_MARKDOWN,
            subject.repaired_formal.REPAIR_MARKDOWN_OUTPUT,
        )
        self.assertNotEqual(
            subject.DEFAULT_FORMAL_MARKER,
            subject.formal_post.COMPLETE_MARKER,
        )
        policy = subject.policy_contract()
        self.assertEqual(
            policy["formal_aggregate_authority"],
            "versioned_selection_contract_repair_v1_only",
        )
        self.assertFalse(policy["frozen_original_aggregate_accepted"])
        self.assertTrue(
            policy["each_variant_uses_own_selected_checkpoints"]
        )

    def test_real_repaired_marker_report_and_selection_contract_validate(
        self,
    ) -> None:
        marker, report = subject.validate_repaired_formal_aggregate(
            marker_path=subject.DEFAULT_FORMAL_MARKER,
            report_path=subject.DEFAULT_FORMAL_REPORT,
            markdown_path=subject.DEFAULT_FORMAL_MARKDOWN,
        )
        self.assertEqual(
            marker["decision"],
            subject.EXPECTED_FORMAL_DECISION,
        )
        self.assertFalse(report["aggregate_full_model_gate_passed"])
        repair = report["comparison_contract"][
            "selection_contract_repair"
        ]
        self.assertTrue(
            repair["each_variant_uses_own_selected_checkpoints"]
        )
        self.assertEqual(repair["repair_id"], subject.FORMAL_REPAIR_ID)

    def test_repaired_decision_or_own_checkpoint_flag_tampering_fails(
        self,
    ) -> None:
        marker = json.loads(
            subject.DEFAULT_FORMAL_MARKER.read_text(encoding="utf-8")
        )
        original_report = json.loads(
            subject.DEFAULT_FORMAL_REPORT.read_text(encoding="utf-8")
        )
        mutations = (
            (
                "decision",
                lambda report: report.__setitem__(
                    "decision",
                    "FULL_MODEL_GATE_PASSED",
                ),
                "decision field differs",
            ),
            (
                "own_checkpoint",
                lambda report: report["comparison_contract"][
                    "selection_contract_repair"
                ].__setitem__(
                    "each_variant_uses_own_selected_checkpoints",
                    False,
                ),
                "repair field differs",
            ),
        )
        for name, mutate, pattern in mutations:
            with self.subTest(name=name):
                report = copy.deepcopy(original_report)
                mutate(report)
                with (
                    mock.patch.object(
                        subject,
                        "_load_regular_json",
                        side_effect=[copy.deepcopy(marker), report],
                    ),
                    self.assertRaisesRegex(ValueError, pattern),
                ):
                    subject.validate_repaired_formal_aggregate(
                        marker_path=subject.DEFAULT_FORMAL_MARKER,
                        report_path=subject.DEFAULT_FORMAL_REPORT,
                        markdown_path=subject.DEFAULT_FORMAL_MARKDOWN,
                    )

    def test_current_binding_uses_current_v3_binding_run_dir_and_binds_repair(
        self,
    ) -> None:
        marker = {
            "schema": subject.formal_post.COMPLETE_MARKER_SCHEMA,
            "status": "complete",
        }
        report = {
            "schema": subject.formal_post.SCHEMA,
            "status": "complete",
            "decision": subject.EXPECTED_FORMAL_DECISION,
            "aggregate_full_model_gate_passed": False,
            "comparison_contract": {
                "selection_contract_repair": {"fixture": True},
            },
        }
        current_bindings = {
            checkpoint: {
                "run_dir": spec.FORMAL_RUN_DIR,
                "artifact_identity": {"checkpoint": checkpoint},
            }
            for checkpoint in spec.CHECKPOINTS
        }
        with (
            mock.patch.object(
                subject.v3_freeze,
                "verify_training_lock",
                return_value={"training_data_sha256": "1" * 64},
            ),
            mock.patch.object(
                subject.v3_freeze,
                "verify_acceptance_lock",
                return_value={"training_source_lock_sha256": "2" * 64},
            ),
            mock.patch.object(
                subject,
                "validate_repaired_formal_aggregate",
                return_value=(marker, report),
            ),
            mock.patch.object(
                subject.formal_post,
                "current_v3_binding",
                side_effect=lambda checkpoint: current_bindings[checkpoint],
            ),
            mock.patch.object(
                subject.formal_post,
                "validate_v3_sweep",
            ),
            mock.patch.object(
                subject,
                "_sha256",
                return_value="f" * 64,
            ),
        ):
            binding = subject.current_formal_artifact_binding()
        repair = binding["formal_selection_contract_repair"]
        self.assertEqual(repair["repair_id"], subject.FORMAL_REPAIR_ID)
        self.assertEqual(
            repair["formal_aggregate_decision"],
            subject.EXPECTED_FORMAL_DECISION,
        )
        self.assertTrue(
            repair["each_variant_uses_own_selected_checkpoints"]
        )


if __name__ == "__main__":
    unittest.main()
