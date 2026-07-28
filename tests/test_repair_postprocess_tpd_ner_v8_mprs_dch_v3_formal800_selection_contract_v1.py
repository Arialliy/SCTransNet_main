from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    repair_postprocess_tpd_ner_v8_mprs_dch_v3_formal800_selection_contract_v1
    as subject,
)


FIXED_AXES = {
    "dataset": subject.locked.DATASET,
    "epochs": 800,
    "batch_size": 16,
    "patch_size": 256,
    "workers": 0,
    "seed": subject.locked.TRAINING_SEED,
    "split_seed": subject.locked.SPLIT_SEED,
    "val_fraction": 0.2,
    "eval_every": 1,
    "base_lr": 1e-3,
    "min_lr": 1e-5,
    "warmup_epochs": 10,
    "threshold": 0.5,
    "match_radius": 3.0,
    "tiny_area": 9,
    "amp": False,
    "max_train_images": None,
    "max_val_images": None,
    "device": "cuda:0",
}


class SelectionContractRepairV1Tests(unittest.TestCase):
    def test_actual_bug_is_reproduced_and_repaired_contract_passes(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "baseline_sctransnet selection source differs",
        ):
            subject.locked.same_split_and_training_contract()
        contract = subject.repaired_same_split_and_training_contract()
        repair = contract["selection_contract_repair"]
        self.assertTrue(repair["modern_exact_equality_verified"])
        self.assertEqual(
            repair["modern_selection_source_exact"],
            "internal_validation_only",
        )
        self.assertFalse(
            repair["baseline_policy_text_equality_to_modern_required"]
        )
        self.assertFalse(
            repair["baseline_top_level_selection_source_required"]
        )
        evidence = repair["baseline_internal_validation_evidence"]
        self.assertTrue(
            evidence[
                "legacy_schema_top_level_selection_source_absent"
            ]
        )
        self.assertTrue(
            evidence["v2_aggregate"]["baseline_role_mapping_verified"]
        )
        self.assertEqual(set(evidence["sweeps"]), set(subject.locked.CHECKPOINTS))
        selection = repair["per_variant_checkpoint_selection"]
        self.assertTrue(
            repair["each_variant_uses_own_selected_checkpoints"]
        )
        self.assertEqual(set(selection), set(subject.ALL_VARIANTS))
        run_dirs = {
            subject.locked.VARIANT_V3_ON: subject.locked.V3_RUN_DIR,
            subject.locked.VARIANT_V2_ON: subject.locked.V2_RUN_DIR,
            subject.locked.VARIANT_V1_OFF: subject.locked.V1_OFF_RUN_DIR,
            subject.locked.BASELINE_VARIANT: subject.locked.BASELINE_RUN_DIR,
        }
        for variant, run_dir in run_dirs.items():
            self.assertTrue(
                selection[variant]["uses_own_checkpoint_directory"]
            )
            for checkpoint, role in subject.locked.CHECKPOINT_ROLES.items():
                recorded = selection[variant]["checkpoints"][checkpoint]
                self.assertEqual(
                    recorded["checkpoint_path"],
                    str((run_dir / checkpoint).resolve()),
                )
                self.assertEqual(recorded["checkpoint_role"], role)
                self.assertEqual(recorded["selection_owner"], variant)

    def test_attestation_binds_exact_required_closure(self) -> None:
        result = subject.verify_repair_attestation()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["repair_id"], subject.REPAIR_ID)
        self.assertEqual(
            set(result["bindings"]),
            set(subject.required_attestation_paths()),
        )
        for required in (
            "source:frozen_v3_postprocess",
            "source:repair_wrapper",
            "artifact:v3_sweep_best",
            "artifact:v3_sweep_best_miou",
            "artifact:baseline_sweep_best",
            "artifact:baseline_sweep_best_miou",
            "aggregate:v2_json",
            "aggregate:v2_markdown",
            "aggregate:v2_complete_marker",
        ):
            self.assertIn(required, result["bindings"])

    def test_attestation_hash_tamper_is_rejected(self) -> None:
        payload = json.loads(subject.ATTESTATION.read_text(encoding="utf-8"))
        payload["bindings"]["artifact:v3_sweep_best"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attestation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "artifact:v3_sweep_best SHA-256 differs",
            ):
                subject.verify_repair_attestation(path)

    def test_attestation_extra_binding_is_rejected(self) -> None:
        payload = json.loads(subject.ATTESTATION.read_text(encoding="utf-8"))
        payload["bindings"]["extra"] = {
            "path": "experiments/extra",
            "sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attestation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "binding matrix differs",
            ):
                subject.verify_repair_attestation(path)

    def test_modern_selection_source_must_remain_exactly_equal(self) -> None:
        original_load = subject.load_json

        def altered(path: Path) -> dict:
            value = original_load(path)
            if Path(path) == subject.locked.V3_RUN_DIR / "protocol.json":
                value = copy.deepcopy(value)
                value["selection_source"] = "different"
            return value

        with (
            mock.patch.object(
                subject.locked.v2_post,
                "same_split_and_training_contract",
                return_value={"fixed_training_axes": FIXED_AXES},
            ),
            mock.patch.object(subject, "load_json", side_effect=altered),
            self.assertRaisesRegex(
                ValueError,
                "v3.*selection source differs from V1",
            ),
        ):
            subject.repaired_same_split_and_training_contract()

    def test_modern_checkpoint_policy_must_remain_exactly_equal(self) -> None:
        original_load = subject.load_json

        def altered(path: Path) -> dict:
            value = original_load(path)
            if Path(path) == subject.locked.V2_RUN_DIR / "protocol.json":
                value = copy.deepcopy(value)
                value["checkpoint_policy"] = "different"
            return value

        with (
            mock.patch.object(
                subject.locked.v2_post,
                "same_split_and_training_contract",
                return_value={"fixed_training_axes": FIXED_AXES},
            ),
            mock.patch.object(subject, "load_json", side_effect=altered),
            self.assertRaisesRegex(
                ValueError,
                "v2.*checkpoint policy differs from V1",
            ),
        ):
            subject.repaired_same_split_and_training_contract()

    def test_baseline_internal_selection_audit_is_required(self) -> None:
        original_load = subject.load_json
        target = subject.locked.sweep_path(
            subject.locked.BASELINE_RUN_DIR,
            "best.pth.tar",
        )

        def altered(path: Path) -> dict:
            value = original_load(path)
            if Path(path) == target:
                value = copy.deepcopy(value)
                value["audit"]["selection_source"] = "different"
            return value

        with (
            mock.patch.object(
                subject.locked.v2_post,
                "same_split_and_training_contract",
                return_value={"fixed_training_axes": FIXED_AXES},
            ),
            mock.patch.object(subject, "load_json", side_effect=altered),
            self.assertRaisesRegex(
                ValueError,
                "baseline best.pth.tar selection source differs",
            ),
        ):
            subject.repaired_same_split_and_training_contract()

    def test_repair_scope_replaces_one_logic_function_and_restores_all(
        self,
    ) -> None:
        original = {
            "contract": subject.locked.same_split_and_training_contract,
            "comparison": subject.locked.COMPARISON_DIR,
            "json": subject.locked.JSON_OUTPUT,
            "markdown": subject.locked.MARKDOWN_OUTPUT,
            "marker": subject.locked.COMPLETE_MARKER,
        }
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with subject._locked_aggregate_repair_scope():
                self.assertIs(
                    subject.locked.same_split_and_training_contract,
                    subject.repaired_same_split_and_training_contract,
                )
                self.assertEqual(
                    subject.locked.COMPARISON_DIR,
                    subject.REPAIR_COMPARISON_DIR,
                )
                self.assertEqual(
                    subject.locked.JSON_OUTPUT,
                    subject.REPAIR_JSON_OUTPUT,
                )
                raise RuntimeError("fixture")
        self.assertIs(
            subject.locked.same_split_and_training_contract,
            original["contract"],
        )
        self.assertEqual(subject.locked.COMPARISON_DIR, original["comparison"])
        self.assertEqual(subject.locked.JSON_OUTPUT, original["json"])
        self.assertEqual(subject.locked.MARKDOWN_OUTPUT, original["markdown"])
        self.assertEqual(subject.locked.COMPLETE_MARKER, original["marker"])

    def test_aggregate_wrapper_calls_frozen_aggregate_with_repair_scope(
        self,
    ) -> None:
        called = []

        def frozen_stub():
            called.append(True)
            self.assertIs(
                subject.locked.same_split_and_training_contract,
                subject.repaired_same_split_and_training_contract,
            )
            self.assertEqual(
                subject.locked.COMPARISON_DIR,
                subject.REPAIR_COMPARISON_DIR,
            )
            report = {
                "comparison_contract": {
                    "selection_contract_repair": {
                        "repair_id": subject.REPAIR_ID,
                    }
                }
            }
            return report, (
                subject.locked.JSON_OUTPUT,
                subject.locked.MARKDOWN_OUTPUT,
                subject.locked.COMPLETE_MARKER,
            )

        frozen_contract = subject.locked.same_split_and_training_contract
        frozen_output = subject.locked.JSON_OUTPUT
        with (
            mock.patch.object(
                subject,
                "verify_repair_attestation",
                return_value={"status": "fixture"},
            ),
            mock.patch.object(
                subject.locked,
                "aggregate_and_write",
                side_effect=frozen_stub,
            ),
        ):
            report, paths = subject.aggregate_and_write()
        self.assertEqual(called, [True])
        self.assertEqual(
            report["comparison_contract"]["selection_contract_repair"][
                "repair_id"
            ],
            subject.REPAIR_ID,
        )
        self.assertEqual(paths[0], subject.REPAIR_JSON_OUTPUT)
        self.assertIs(
            subject.locked.same_split_and_training_contract,
            frozen_contract,
        )
        self.assertEqual(subject.locked.JSON_OUTPUT, frozen_output)

    def test_plan_is_aggregate_only_gpu_free_and_path_isolated(self) -> None:
        plan = subject.execution_plan()
        self.assertEqual(plan["new_evaluation_count"], 0)
        self.assertFalse(plan["gpu_work"])
        self.assertEqual(
            plan["logic_overrides"],
            ["same_split_and_training_contract"],
        )
        self.assertFalse(
            plan["frozen_v3_outputs_are_publication_targets"]
        )
        paths = plan["publication_path_overrides"]
        self.assertIn(
            "comparison_selection_contract_repair_v1",
            paths["comparison_dir"],
        )
        self.assertNotEqual(
            Path(paths["json"]),
            subject.locked.JSON_OUTPUT.resolve(),
        )

    def test_wrapper_has_no_evaluation_or_training_execution_path(self) -> None:
        source = inspect.getsource(subject)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("run_v3_evaluations(", source)
        self.assertNotIn("_run_v3_evaluation(", source)
        self.assertNotIn("train(", source)
        self.assertEqual(source.count("locked.aggregate_and_write()"), 1)
        self.assertNotIn("\nassert ", source)

    def test_protocol_documents_only_allowed_override(self) -> None:
        text = subject.PROTOCOL.read_text(encoding="utf-8")
        self.assertIn("same_split_and_training_contract", text)
        self.assertIn("No other comparison or gate logic may be replaced", text)
        self.assertIn("performs no", text)
        self.assertIn("The repair introduces no", text)
        self.assertIn(
            "comparison_selection_contract_repair_v1",
            text,
        )


if __name__ == "__main__":
    unittest.main()
