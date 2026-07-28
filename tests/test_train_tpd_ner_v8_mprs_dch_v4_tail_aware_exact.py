from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from experiments import tpd_exact_runner as exact_runner
from experiments import (
    train_tpd_ner_v8_mprs_dch_v3_exact as v3_exact,
)
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact as entry,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    TPDNERV8MPRSDCHV4SCTransNet,
)


V4_ON = entry.TPD_NER_V8_MPRS_DCH_V4_TAIL_AWARE_FULL_RELAY_ON


class StateScaler:
    def state_dict(self) -> dict[str, int]:
        return {"updates": 0}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if state != {"updates": 0}:
            raise ValueError("unexpected scaler fixture state")


def parse(trailing: list[str]):
    return entry.parse_args(
        [
            "--variant",
            V4_ON,
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            *trailing,
        ]
    )


def validation_metrics() -> dict[str, int | float]:
    counts = {
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
    return {
        name: 1 if name in counts else 0.5
        for name in entry.STORED_VALIDATION_METRICS
    }


def make_spec_and_identity(
    args,
    model: nn.Module,
    metadata: dict,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=entry.FORMAL_BASE_LR,
    )
    spec = entry.make_exact_run_spec(
        args,
        model=model,
        model_metadata=metadata,
        optimizer=optimizer,
        scaler=StateScaler(),
        initialization_contract=exact_runner.fresh_initialization_contract(),
        initial_model_state_sha256=(
            exact_runner.initial_model_state_sha256(model)
        ),
        initial_rng=exact_runner.initial_rng_contract(),
        selection_policy=exact_runner.pd_miou_selection_policy(
            stored_metrics=entry.STORED_VALIDATION_METRICS
        ).normalized(),
        source_locks={
            entry.SOURCE_LOCK_KEY: "1" * 64,
            "training_data": "2" * 64,
        },
        split_records={
            "train": exact_runner.OrderedFingerprint.from_values(
                "train", ("a", "b")
            ),
            "validation": exact_runner.OrderedFingerprint.from_values(
                "validation", ("c",)
            ),
        },
        data_records={
            "train_samples": exact_runner.OrderedFingerprint.from_values(
                "train_samples", ("a:image", "b:image")
            ),
            "normalization": exact_runner.OrderedFingerprint.from_values(
                "normalization", ('{"mean":1.0,"std":2.0}',)
            ),
        },
        environment={"name": "cpu-v4-fixture"},
    )
    identity = exact_runner.build_run_identity(model, spec)
    return spec, identity


class V4TailAwareExactTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.args = parse(["--fresh"])
        cls.model, cls.metadata = entry.build_selected_model(V4_ON, 42)
        cls.spec, cls.identity = make_spec_and_identity(
            cls.args,
            cls.model,
            cls.metadata,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.spec, cls.identity, cls.metadata, cls.model, cls.args
        torch.set_num_threads(cls.previous_threads)

    def test_parser_freezes_candidate_formula_thresholds_and_training_axes(
        self,
    ) -> None:
        args = self.args
        self.assertEqual(args.variant, V4_ON)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.split_seed, 20260722)
        self.assertEqual(args.epochs, 800)
        self.assertFalse(args.amp)
        self.assertEqual(args.dc_support_mode, "complement_tail")
        self.assertEqual(
            args.tail_z_thresholds,
            {"4": 1.5, "3": 2.0, "2": 2.5},
        )
        self.assertEqual(args.run_tag, entry.FORMAL_RUN_TAG)
        self.assertEqual(
            entry.formal_contract()["mask_mapping"],
            "atan(pi*(centered+dc*selected_support))/pi",
        )
        self.assertTrue(
            entry.run_directory(args).is_relative_to(
                entry.DEFAULT_OUTPUT_ROOT
            )
        )

        invalid = (
            ["--fresh", "--seed", "7"],
            ["--fresh", "--variant", v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON],
            ["--fresh", "--dc-support-mode", "legacy_global"],
            ["--fresh", "--tail-z-threshold-stage3", "1.0"],
            ["--fresh", "--relay-width", "8"],
        )
        for trailing in invalid:
            with self.subTest(trailing=trailing):
                with self.assertRaises((ValueError, SystemExit)):
                    parse(trailing)

    def test_production_builder_is_fresh_v4_complement_only(self) -> None:
        self.assertIs(type(self.model), TPDNERV8MPRSDCHV4SCTransNet)
        self.assertEqual(self.model.tpd_ner.dc_support_mode, "complement_tail")
        self.assertEqual(
            {str(k): float(v) for k, v in self.model.tpd_ner.tail_z_thresholds.items()},
            entry.TAIL_Z_THRESHOLDS,
        )
        metadata = self.metadata
        self.assertTrue(metadata["fresh_training"])
        self.assertFalse(metadata["warm_start_applied"])
        self.assertFalse(metadata["v3_warm_start"])
        self.assertEqual(metadata["dc_support_formula_stage3_2"], "1-P")
        manifest = metadata["architecture_manifest"]
        self.assertEqual(manifest["dc_support_mode"], "complement_tail")
        self.assertEqual(manifest["dc_support_formula_stage3_2"], "1-P")
        self.assertEqual(
            manifest["ner_dc_offset_support_scope"],
            entry.DC_SUPPORT_SCOPE,
        )
        self.assertEqual(manifest["tail_z_thresholds"], entry.TAIL_Z_THRESHOLDS)
        self.assertEqual(manifest["tail_support_parameters"], 0)
        self.assertEqual(manifest["tail_support_buffers"], 0)
        self.assertEqual(
            manifest["formula_selection_aggregate_sha256"],
            entry.FORMULA_SELECTION_AGGREGATE_SHA256,
        )
        self.assertRegex(metadata["architecture_id"], r"^[0-9a-f]{64}$")

    def test_exact_run_identity_is_v4_owned_and_json_normalizable(self) -> None:
        normalized = self.spec.normalized()
        json.dumps(normalized, sort_keys=True, allow_nan=False)
        identity = entry.require_v4_run_identity(
            self.identity,
            label="fixture",
            expected_variant=V4_ON,
        )
        self.assertTrue(identity["run_id"].startswith(entry.RUN_ID_PREFIX))
        determinism = identity["training_contract"]["determinism"]
        self.assertEqual(determinism["entry_schema"], entry.ENTRY_SCHEMA)
        self.assertEqual(determinism["relay_version"], entry.V4_RELAY_VERSION)
        self.assertEqual(determinism["dc_support_mode"], "complement_tail")
        self.assertEqual(determinism["dc_support_formula_stage3_2"], "1-P")
        self.assertEqual(
            determinism["tail_z_thresholds"],
            entry.TAIL_Z_THRESHOLDS,
        )
        self.assertEqual(
            determinism["required_control"],
            v3_exact.V8_PARENT_RELAY_OFF_REFERENCE,
        )
        self.assertEqual(
            determinism["paired_gate_predecessor"],
            v3_exact.V2_RELAY_ON_VARIANT,
        )
        self.assertEqual(
            determinism["structural_predecessor"],
            v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        )
        self.assertNotEqual(entry.ENTRY_SCHEMA, v3_exact.ENTRY_SCHEMA)
        self.assertNotEqual(entry.SOURCE_LOCK_KEY, v3_exact.SOURCE_LOCK_KEY)
        self.assertNotEqual(entry.DEFAULT_OUTPUT_ROOT, v3_exact.DEFAULT_OUTPUT_ROOT)

    def test_v3_or_wrong_formula_identity_is_rejected(self) -> None:
        v3_identity = copy.deepcopy(self.identity)
        v3_identity["variant"] = (
            v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
        )
        v3_identity["run_id"] = f"{v3_exact.RUN_ID_PREFIX}fixture"
        v3_identity["source_locks"] = {
            v3_exact.SOURCE_LOCK_KEY: "1" * 64,
            "training_data": "2" * 64,
        }
        v3_identity["training_contract"]["determinism"][
            "entry_schema"
        ] = v3_exact.ENTRY_SCHEMA
        with self.assertRaises(ValueError):
            entry.require_v4_run_identity(v3_identity, label="v3")

        wrong_formula = copy.deepcopy(self.identity)
        wrong_formula["training_contract"]["determinism"][
            "dc_support_mode"
        ] = "direct_tail"
        with self.assertRaisesRegex(ValueError, "dc_support_mode"):
            entry.require_v4_run_identity(
                wrong_formula,
                label="wrong formula",
            )

        wrong_threshold = copy.deepcopy(self.identity)
        wrong_threshold["training_contract"]["determinism"][
            "tail_z_thresholds"
        ]["3"] = 1.0
        with self.assertRaisesRegex(ValueError, "thresholds"):
            entry.require_v4_run_identity(
                wrong_threshold,
                label="wrong threshold",
            )

    def test_checkpoint_roles_record_and_validate_v4_nonstate_identity(
        self,
    ) -> None:
        adapter = entry.EvaluatorCheckpointAdapter(
            model_metadata=self.metadata,
            split_hashes={"train": "a" * 64},
        )
        for role in (
            "last_evaluated_epoch",
            "best_validation_pd_primary",
            "best_validation_miou_secondary",
        ):
            with self.subTest(role=role):
                checkpoint = adapter(
                    exact_runner.CompatibilityPayloadContext(
                        role=role,
                        epoch=1,
                        metrics=validation_metrics(),
                        event={"epoch": 1, **validation_metrics()},
                        exact_payload={
                            "model": {
                                "state_dict": {
                                    "w": torch.tensor([1.0])
                                }
                            },
                            "optimizer": {"state_dict": {"state": {}}},
                            "scaler": {
                                "state_dict": {"updates": 0}
                            },
                        },
                        run_identity=self.identity,
                        normalized_spec=self.spec.normalized(),
                    )
                )
                self.assertEqual(checkpoint["schema"], entry.CHECKPOINT_SCHEMA)
                self.assertEqual(checkpoint["checkpoint_role"], role)
                self.assertEqual(checkpoint["dc_support_mode"], "complement_tail")
                self.assertEqual(
                    checkpoint["dc_support_formula_stage3_2"],
                    "1-P",
                )
                self.assertEqual(
                    checkpoint["tail_z_thresholds"],
                    entry.TAIL_Z_THRESHOLDS,
                )
                self.assertEqual(
                    checkpoint["formula_selection_decision"],
                    "COMPLEMENT_TAIL_SELECTED",
                )
                self.assertEqual(
                    checkpoint["required_control"],
                    v3_exact.V8_PARENT_RELAY_OFF_REFERENCE,
                )
                self.assertEqual(
                    checkpoint["paired_gate_predecessor"],
                    v3_exact.V2_RELAY_ON_VARIANT,
                )
                entry.require_evaluator_checkpoint_payload(
                    checkpoint,
                    expected_variant=V4_ON,
                )

                changed = copy.deepcopy(checkpoint)
                changed["dc_support_formula_stage3_2"] = "P"
                with self.assertRaisesRegex(ValueError, "formula"):
                    entry.require_evaluator_checkpoint_payload(changed)

    def test_fresh_and_exact_resume_are_v4_only(self) -> None:
        fresh = entry.initialization_plan(
            self.args,
            Path("/unused"),
            self.model,
        )
        self.assertEqual(fresh.request.mode.value, "fresh")

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "protocol.json").write_text(
                json.dumps(
                    {
                        "schema": entry.ENTRY_SCHEMA,
                        "run_identity": self.identity,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            resume_args = parse(
                [
                    "--exact-resume",
                    "--output-root",
                    directory,
                ]
            )
            plan = entry.initialization_plan(
                resume_args,
                run_dir,
                self.model,
            )
            self.assertEqual(plan.request.mode.value, "exact_resume")

            v3_protocol = copy.deepcopy(self.identity)
            v3_protocol["training_contract"]["determinism"][
                "entry_schema"
            ] = v3_exact.ENTRY_SCHEMA
            (run_dir / "protocol.json").write_text(
                json.dumps(
                    {
                        "schema": v3_exact.ENTRY_SCHEMA,
                        "run_identity": v3_protocol,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema"):
                entry.initialization_plan(
                    resume_args,
                    run_dir,
                    self.model,
                )

    def test_protocol_keeps_v1_v2_v3_roles_separate(self) -> None:
        payload = entry.protocol_payload(
            self.args,
            directory=Path("/tmp/v4-tail-aware-protocol-fixture"),
            model_metadata=self.metadata,
            normalization={"mean": 0.1, "std": 0.2},
            run_identity=self.identity,
        )
        design = payload["comparison_design"]
        self.assertEqual(
            design["required_control"],
            v3_exact.V8_PARENT_RELAY_OFF_REFERENCE,
        )
        self.assertEqual(
            design["paired_gate_predecessor"],
            v3_exact.V2_RELAY_ON_VARIANT,
        )
        self.assertEqual(
            design["structural_predecessor"],
            v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        )
        for variant in (
            "baseline_sctransnet",
            v3_exact.V8_PARENT_RELAY_OFF_REFERENCE,
            v3_exact.V2_RELAY_ON_VARIANT,
            v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
            V4_ON,
        ):
            self.assertIn(variant, design["primary"])
        self.assertEqual(
            design["postprocess_requirements"],
            {
                "v1_role_gate": True,
                "v2_role_gate": True,
                "v3_additional_delta_report": True,
            },
        )

    def test_active_journal_guard_rejects_v3_before_restore(self) -> None:
        runner = object.__new__(entry.TPDNERV8V4TailAwareExactRunner)
        runner.journal = mock.Mock()
        runner.journal.load_active.return_value = SimpleNamespace(
            checkpoint_path=Path("/tmp/fixture.pth")
        )
        runner.spec = SimpleNamespace(variant=V4_ON)
        v3_identity = copy.deepcopy(self.identity)
        v3_identity["variant"] = (
            v3_exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
        )
        v3_identity["training_contract"]["determinism"][
            "entry_schema"
        ] = v3_exact.ENTRY_SCHEMA
        runner._load_exact_payload = mock.Mock(
            return_value=(
                {
                    "run_identity": v3_identity,
                    "optimizer": {"state_dict": {}},
                },
                "fixture-digest",
            )
        )
        with self.assertRaisesRegex(ValueError, "entry schema"):
            runner._require_v8_active_journal()
        runner._load_exact_payload.assert_called_once()

    def test_protocol_draft_and_runtime_sources_are_v4_bound(self) -> None:
        selection = entry.formula_selection_contract()
        self.assertEqual(
            selection["selected_formula_mode"],
            "complement_tail",
        )
        self.assertEqual(
            selection["aggregate_sha256"],
            entry.FORMULA_SELECTION_AGGREGATE_SHA256,
        )
        self.assertTrue(entry.PROTOCOL_DRAFT_PATH.is_file())
        runtime = set(entry.RUNTIME_SOURCE_PATHS)
        self.assertIn(Path(entry.__file__).resolve(), runtime)
        self.assertIn(entry.PROTOCOL_DRAFT_PATH.resolve(), runtime)
        self.assertIn(
            (
                Path(entry.REPO_ROOT)
                / "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py"
            ).resolve(),
            runtime,
        )
        self.assertEqual(
            entry.DEFAULT_EXACT_SOURCE_LOCK_PATH.name,
            "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock.json",
        )


if __name__ == "__main__":
    unittest.main()
