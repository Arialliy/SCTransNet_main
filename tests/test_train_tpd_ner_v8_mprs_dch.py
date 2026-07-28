from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainTPDNERV8MPRSDCHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = importlib.import_module(
            "experiments.train_tpd_ner_v8_mprs_dch"
        )

    def test_import_is_side_effect_free_and_does_not_bind_shared_runner(self) -> None:
        script = r"""
import json
from experiments import train_tpd_pilot as base
before = (
    base.SUPPORTED_VARIANTS,
    base.build_model,
    base.parse_args,
    base.checkpoint_payload,
    base.write_json,
    base.append_jsonl,
)
from experiments import train_tpd_ner_v8_mprs_dch as entry
after = (
    base.SUPPORTED_VARIANTS,
    base.build_model,
    base.parse_args,
    base.checkpoint_payload,
    base.write_json,
    base.append_jsonl,
)
print(json.dumps({
    "same": all(left is right for left, right in zip(before, after)),
    "variants": list(entry.SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn('"same": true', completed.stdout)
        self.assertIn(
            "tpd_ner_v8_mprs_dch_full_relay_off",
            completed.stdout,
        )
        self.assertIn(
            "tpd_ner_v8_mprs_dch_full_relay_on",
            completed.stdout,
        )

    def test_formal_parser_hard_rejects_other_seeds_and_settings(self) -> None:
        subject = self.subject
        valid = subject.parse_args(
            [
                "--variant",
                subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(valid.seed, 42)
        self.assertEqual(valid.split_seed, 20260722)
        self.assertEqual(valid.epochs, 800)
        self.assertEqual(valid.batch_size, 16)
        self.assertEqual(valid.patch_size, 256)
        self.assertEqual(valid.workers, 0)
        self.assertEqual(valid.eval_every, 1)
        self.assertFalse(valid.amp)
        self.assertIsNone(valid.max_train_images)
        self.assertIsNone(valid.max_val_images)

        invalid_arguments = (
            ("--seed", "3407"),
            ("--split-seed", "42"),
            ("--epochs", "1"),
            ("--batch-size", "8"),
            ("--workers", "1"),
        )
        for option, value in invalid_arguments:
            with self.subTest(option=option):
                with contextlib.redirect_stderr(None):
                    with self.assertRaises(SystemExit):
                        subject.parse_args(
                            [
                                "--variant",
                                subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
                                option,
                                value,
                            ]
                        )

    def test_full_relay_pair_has_paired_parent_and_distinct_identity(self) -> None:
        subject = self.subject
        off, off_metadata = subject.build_tpd_ner_v8_mprs_dch_model(
            subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
            42,
        )
        on, on_metadata = subject.build_tpd_ner_v8_mprs_dch_model(
            subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
            42,
        )
        try:
            self.assertFalse(off_metadata["relay_enabled"])
            self.assertTrue(on_metadata["relay_enabled"])
            self.assertEqual(
                off_metadata["comparison_role"],
                "tpd_only_v8_full_relay_off_control",
            )
            self.assertEqual(
                on_metadata["comparison_role"],
                "tpd_plus_ner_v8_full",
            )
            self.assertEqual(off_metadata["relay_parameters"], 0)
            self.assertEqual(on_metadata["relay_parameters"], 11_291)
            self.assertEqual(off_metadata["total_parameters"], 10_843_155)
            self.assertEqual(on_metadata["total_parameters"], 10_854_446)
            self.assertEqual(
                off_metadata["common_initialization_sha256"],
                on_metadata["common_initialization_sha256"],
            )
            self.assertEqual(
                off_metadata["parent_model_metadata"][
                    "full_initialization_sha256"
                ],
                on_metadata["parent_model_metadata"][
                    "full_initialization_sha256"
                ],
            )
            self.assertNotEqual(
                off_metadata["architecture_id"],
                on_metadata["architecture_id"],
            )
            self.assertFalse(hasattr(off, "tpd_ner"))
            self.assertTrue(hasattr(on, "tpd_ner"))
            self.assertEqual(
                on_metadata["architecture_manifest"]["evidence_layout"],
                (3, 2),
            )
            self.assertEqual(
                on_metadata["architecture_manifest"][
                    "deep_supervision_outputs"
                ],
                6,
            )
        finally:
            del off, on
            gc.collect()

        with self.assertRaisesRegex(ValueError, "only model seed=42"):
            subject.build_tpd_ner_v8_mprs_dch_model(
                subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
                3407,
            )

    def _formal_args(self) -> argparse.Namespace:
        return self.subject.parse_args(
            [
                "--variant",
                self.subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
                "--device",
                "cpu",
            ]
        )

    def test_artifact_annotations_bind_schema_run_and_ordered_split(self) -> None:
        subject = self.subject
        args = self._formal_args()
        train_ids = [f"train-{index:03d}" for index in range(530)]
        val_ids = [f"val-{index:03d}" for index in range(133)]
        split = subject.annotate_json_artifact(
            Path("split.json"),
            {
                "full_official_train_count": 663,
                "used_train_count": 530,
                "used_val_count": 133,
                "used_train_ids": train_ids,
                "used_val_ids": val_ids,
            },
            args,
        )
        self.assertEqual(split["schema"], subject.SPLIT_SCHEMA)
        self.assertEqual(split["run_identity"]["seed"], 42)
        self.assertEqual(split["run_identity"]["split_seed"], 20260722)
        self.assertEqual(
            split["ordered_used_val_sha256"],
            subject.ordered_identifier_sha256(val_ids),
        )
        self.assertNotEqual(
            split["ordered_used_val_sha256"],
            subject.ordered_identifier_sha256(list(reversed(val_ids))),
        )

        protocol = subject.annotate_json_artifact(
            Path("protocol.json"),
            {
                "model": {
                    "variant": args.variant,
                    "architecture_id": "a" * 64,
                }
            },
            args,
        )
        self.assertEqual(protocol["schema"], subject.ENTRY_SCHEMA)
        self.assertEqual(
            protocol["comparison_design"]["required_ablation"],
            subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
        )
        self.assertEqual(
            protocol["comparison_design"]["primary"],
            [
                "baseline_sctransnet",
                subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
                subject.TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
            ],
        )
        self.assertEqual(
            tuple(protocol["stored_validation_metrics"]),
            subject.STORED_VALIDATION_METRICS,
        )

        summary = subject.annotate_json_artifact(
            Path("summary.json"),
            {"status": "complete"},
            args,
        )
        self.assertEqual(summary["schema"], subject.SUMMARY_SCHEMA)
        self.assertEqual(
            summary["run_identity"]["run_id"],
            subject.formal_run_id(args),
        )

    def test_checkpoint_payload_has_ner_owned_identity(self) -> None:
        subject = self.subject
        args = self._formal_args()
        model = nn.Conv2d(1, 1, kernel_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        metrics = {
            name: 1 if name.endswith("_count") else 0.25
            for name in subject.STORED_VALIDATION_METRICS
        }
        metadata = {
            "variant": args.variant,
            "comparison_role": "tpd_plus_ner",
            "relay_enabled": True,
            "architecture_id": "a" * 64,
        }
        payload = subject.checkpoint_payload(
            model,
            optimizer,
            scaler,
            1,
            args.variant,
            args,
            metrics,
            metadata,
            {"used_val_sha256": "b" * 64},
        )
        self.assertEqual(payload["schema"], subject.CHECKPOINT_SCHEMA)
        self.assertEqual(
            payload["checkpoint_identity"]["schema"],
            subject.CHECKPOINT_IDENTITY_SCHEMA,
        )
        self.assertEqual(payload["run_identity"]["seed"], 42)
        self.assertEqual(payload["run_identity"]["split_seed"], 20260722)
        self.assertTrue(payload["six_output_training_semantics"])
        self.assertEqual(
            tuple(payload["stored_validation_metrics"]),
            subject.STORED_VALIDATION_METRICS,
        )

    def test_six_output_contract_rejects_any_other_training_semantics(self) -> None:
        subject = self.subject
        target = torch.zeros(2, 1, 8, 8)
        outputs = tuple(torch.full_like(target, 0.5) for _ in range(6))
        self.assertEqual(
            subject.require_six_output_loss_inputs(outputs, target),
            outputs,
        )
        for count in (1, 5, 7):
            with self.subTest(count=count):
                with self.assertRaisesRegex(RuntimeError, "exactly six"):
                    subject.require_six_output_loss_inputs(
                        tuple(outputs[0] for _ in range(count)),
                        target,
                    )
        bad_shape = list(outputs)
        bad_shape[3] = torch.zeros(2, 1, 4, 4)
        with self.assertRaisesRegex(ValueError, "differs from target"):
            subject.require_six_output_loss_inputs(tuple(bad_shape), target)

    def test_runtime_binding_restores_every_shared_global(self) -> None:
        subject = self.subject
        base = importlib.import_module("experiments.train_tpd_pilot")
        before = {
            name: getattr(base, name)
            for name in (
                "SUPPORTED_VARIANTS",
                "build_model",
                "parse_args",
                "checkpoint_payload",
                "write_json",
                "append_jsonl",
            )
        }
        with subject._formal_runtime_bindings():
            self.assertIs(
                base.build_model,
                subject.build_tpd_ner_v8_mprs_dch_model,
            )
            self.assertEqual(
                base.SUPPORTED_VARIANTS,
                subject.SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS,
            )
        for name, value in before.items():
            self.assertIs(getattr(base, name), value, name)

    def test_main_delegates_without_persistent_global_mutation(self) -> None:
        subject = self.subject
        base = importlib.import_module("experiments.train_tpd_pilot")
        original_build = base.build_model
        observed = {}

        def fake_main() -> None:
            observed["builder"] = base.build_model
            observed["variants"] = base.SUPPORTED_VARIANTS

        with (
            mock.patch.object(base, "main", side_effect=fake_main),
            mock.patch.object(
                subject,
                "guarded_training_runtime",
                side_effect=lambda: contextlib.nullcontext(),
            ),
        ):
            subject.main()
        self.assertIs(
            observed["builder"],
            subject.build_tpd_ner_v8_mprs_dch_model,
        )
        self.assertEqual(
            observed["variants"],
            subject.SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS,
        )
        self.assertIs(base.build_model, original_build)


if __name__ == "__main__":
    unittest.main()
