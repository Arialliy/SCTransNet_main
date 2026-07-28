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


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _six_output_loss(outputs: object, target: torch.Tensor) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("fixture requires exactly six outputs")
    criterion = nn.BCELoss(reduction="mean")
    return sum(criterion(output, target) for output in outputs)


class TrainTPDNERV8MPRSDCHV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = importlib.import_module(
            "experiments.train_tpd_ner_v8_mprs_dch_v2"
        )
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def _formal_args(self) -> argparse.Namespace:
        return self.subject.parse_args(
            [
                "--variant",
                self.subject.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
            ]
        )

    def test_import_is_side_effect_free_and_variant_matrix_is_on_only(
        self,
    ) -> None:
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
from experiments import train_tpd_ner_v8_mprs_dch_v2 as entry
after = (
    base.SUPPORTED_VARIANTS,
    base.build_model,
    base.parse_args,
    base.checkpoint_payload,
    base.write_json,
    base.append_jsonl,
)
print(json.dumps({
    "same": all(a is b for a, b in zip(before, after)),
    "variants": list(entry.SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS),
    "control": entry.V1_RELAY_OFF_REFERENCE,
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
        self.assertEqual(
            self.subject.SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS,
            (self.subject.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,),
        )
        self.assertNotIn(
            self.subject.V1_RELAY_OFF_REFERENCE,
            self.subject.SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS,
        )

    def test_parser_freezes_single_seed_full_fp32_and_rejects_other_axes(
        self,
    ) -> None:
        subject = self.subject
        args = self._formal_args()
        self.assertEqual(args.variant, subject.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.split_seed, 20260722)
        self.assertEqual(args.epochs, 800)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.patch_size, 256)
        self.assertEqual(args.workers, 0)
        self.assertFalse(args.amp)
        self.assertEqual(args.run_tag, subject.FORMAL_RUN_TAG)

        invalid = (
            ("--variant", subject.V1_RELAY_OFF_REFERENCE),
            ("--variant", "tpd_ner_v8_mprs_dch_v2_full_relay_off"),
            ("--seed", "3407"),
            ("--split-seed", "42"),
            ("--batch-size", "8"),
            ("--epochs", "1"),
            ("--device", "cpu"),
        )
        for option, value in invalid:
            arguments = [
                "--variant",
                subject.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
                option,
                value,
            ]
            if option == "--variant":
                arguments = [option, value]
            with self.subTest(option=option, value=value):
                with contextlib.redirect_stderr(None):
                    with self.assertRaises(SystemExit):
                        subject.parse_args(arguments)

    def test_production_builder_has_v2_type_parameters_and_manifest(self) -> None:
        subject = self.subject
        model, metadata = subject.build_tpd_ner_v8_mprs_dch_v2_model(
            subject.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
            42,
        )
        try:
            from model.tpd_ner_v8_mprs_dch_v2 import (
                TPDNERV8MPRSDCHV2SCTransNet,
            )

            self.assertIs(type(model), TPDNERV8MPRSDCHV2SCTransNet)
            self.assertEqual(metadata["relay_parameters"], 11_288)
            self.assertEqual(metadata["total_parameters"], 10_854_443)
            self.assertEqual(
                len(
                    [
                        key
                        for key in model.state_dict()
                        if key.startswith("tpd_ner.")
                    ]
                ),
                16,
            )
            self.assertEqual(
                metadata["relay_version"],
                "v2_rms_centered_arctangent",
            )
            self.assertEqual(
                metadata["required_control"],
                subject.V1_RELAY_OFF_REFERENCE,
            )
            self.assertFalse(metadata["relay_off_retrained"])
            manifest = metadata["architecture_manifest"]
            self.assertEqual(
                manifest["schema"],
                subject.ARCHITECTURE_MANIFEST_SCHEMA,
            )
            self.assertEqual(manifest["relay_rms_eps"], 1e-6)
            self.assertFalse(manifest["gate_bias"])
            self.assertEqual(manifest["mask_mapping"], "atan(pi*z)/pi")
            self.assertEqual(manifest["deep_supervision_outputs"], 6)
        finally:
            del model
            gc.collect()

        with self.assertRaisesRegex(ValueError, "only model seed=42"):
            subject.build_tpd_ner_v8_mprs_dch_v2_model(
                subject.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
                7,
            )

    def test_ordinary_runtime_binds_registered_physical_gpu2_or_gpu3(
        self,
    ) -> None:
        subject = self.subject
        args = self._formal_args()
        expected_uuid = subject.PHYSICAL_GPU_UUIDS["2"]
        with (
            mock.patch.dict(
                os.environ,
                {
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX": "2",
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID": expected_uuid,
                    "CUDA_VISIBLE_DEVICES": expected_uuid,
                },
                clear=False,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 5090",
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=mock.Mock(uuid=expected_uuid),
            ),
        ):
            identity = subject.validate_physical_gpu_runtime(args)
        self.assertEqual(identity["physical_gpu_index"], 2)
        self.assertEqual(identity["physical_gpu_uuid"], expected_uuid)
        self.assertEqual(identity["logical_device"], "cuda:0")

        with (
            mock.patch.dict(
                os.environ,
                {
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX": "0",
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID": "wrong",
                    "CUDA_VISIBLE_DEVICES": "wrong",
                },
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "GPU 2 or 3"),
        ):
            subject.validate_physical_gpu_runtime(args)

    def test_v2_off_probe_is_v1_off_class_state_output_and_first_step_exact(
        self,
    ) -> None:
        subject = self.subject
        direct, _ = subject.build_v1_relay_off_reference()
        probe, probe_metadata = subject.build_v2_relay_off_identity_probe()
        try:
            self.assertIs(type(direct), type(probe))
            self.assertFalse(hasattr(direct, "tpd_ner"))
            self.assertFalse(hasattr(probe, "tpd_ner"))
            self.assertFalse(probe_metadata["formal_training_scheduled"])
            self.assertTrue(
                _nested_equal(direct.state_dict(), probe.state_dict())
            )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(12_345)
            inputs = torch.randn(2, 1, 32, 32, generator=generator)
            targets = torch.rand(2, 1, 32, 32, generator=generator)
            direct.eval()
            probe.eval()
            with torch.no_grad():
                direct_outputs = tuple(direct(inputs))
                probe_outputs = tuple(probe(inputs))
            self.assertEqual(len(direct_outputs), 6)
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(direct_outputs, probe_outputs)
                )
            )

            direct.train()
            probe.train()
            direct_optimizer = torch.optim.Adam(direct.parameters(), lr=1e-3)
            probe_optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
            direct_optimizer.zero_grad(set_to_none=True)
            probe_optimizer.zero_grad(set_to_none=True)
            cpu_rng = torch.get_rng_state()
            direct_loss = _six_output_loss(direct(inputs), targets)
            direct_loss.backward()
            torch.set_rng_state(cpu_rng)
            probe_loss = _six_output_loss(probe(inputs), targets)
            probe_loss.backward()
            self.assertTrue(torch.equal(direct_loss, probe_loss))
            direct_optimizer.step()
            probe_optimizer.step()
            self.assertTrue(
                _nested_equal(direct.state_dict(), probe.state_dict())
            )
            self.assertTrue(
                _nested_equal(
                    direct_optimizer.state_dict(),
                    probe_optimizer.state_dict(),
                )
            )
        finally:
            del direct, probe
            gc.collect()

    def test_artifacts_and_checkpoint_use_v2_identity_and_existing_control(
        self,
    ) -> None:
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
        comparison = protocol["comparison_design"]
        self.assertEqual(
            comparison["required_control"],
            subject.V1_RELAY_OFF_REFERENCE,
        )
        self.assertFalse(comparison["relay_off_retrained"])
        self.assertNotIn(
            "tpd_ner_v8_mprs_dch_v2_full_relay_off",
            comparison["primary"],
        )

        model = nn.Conv2d(1, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        metadata = {
            "variant": args.variant,
            "architecture_id": "a" * 64,
        }
        metrics = {
            name: 1 if name.endswith("_count") else 0.25
            for name in subject.STORED_VALIDATION_METRICS
        }
        checkpoint = subject.checkpoint_payload(
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
        self.assertEqual(checkpoint["schema"], subject.CHECKPOINT_SCHEMA)
        self.assertTrue(
            checkpoint["run_identity"]["run_id"].startswith(
                subject.RUN_ID_PREFIX
            )
        )
        self.assertEqual(
            checkpoint["checkpoint_identity"]["relay_version"],
            "v2_rms_centered_arctangent",
        )

    def test_runtime_bindings_restore_shared_runner_on_success_and_error(
        self,
    ) -> None:
        subject = self.subject
        base = importlib.import_module("experiments.train_tpd_pilot")
        names = (
            "SUPPORTED_VARIANTS",
            "build_model",
            "parse_args",
            "checkpoint_payload",
            "write_json",
            "append_jsonl",
        )
        before = {name: getattr(base, name) for name in names}
        with subject._formal_runtime_bindings():
            self.assertIs(
                base.build_model,
                subject.build_tpd_ner_v8_mprs_dch_v2_model,
            )
            self.assertEqual(
                base.SUPPORTED_VARIANTS,
                subject.SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS,
            )
        self.assertTrue(
            all(getattr(base, name) is value for name, value in before.items())
        )

        physical_identity = {
            "logical_device": "cuda:0",
            "physical_gpu_index": 2,
            "physical_gpu_uuid": subject.PHYSICAL_GPU_UUIDS["2"],
        }
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "train_tpd_ner_v8_mprs_dch_v2.py",
                    "--variant",
                    subject.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
                ],
            ),
            mock.patch.object(
                subject,
                "validate_physical_gpu_runtime",
                return_value=physical_identity,
            ),
            subject._formal_runtime_bindings(),
        ):
            parsed = base.parse_args()
            self.assertEqual(
                parsed.physical_gpu_identity,
                physical_identity,
            )

        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with subject._formal_runtime_bindings():
                raise RuntimeError("fixture")
        self.assertTrue(
            all(getattr(base, name) is value for name, value in before.items())
        )

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
            subject.build_tpd_ner_v8_mprs_dch_v2_model,
        )
        self.assertEqual(
            observed["variants"],
            subject.SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS,
        )


if __name__ == "__main__":
    unittest.main()
