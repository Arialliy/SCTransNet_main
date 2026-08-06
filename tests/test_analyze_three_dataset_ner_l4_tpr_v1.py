from __future__ import annotations

import copy
import math
from pathlib import Path
import unittest

import torch

from analysis import analyze_three_dataset_ner_l4_tpr_v1 as subject
from experiments import ner_l4_tpr_strict_migration_v1 as migration
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr import (
    build_formal_v4_qfg_v2_croa_l4_tpr_inference_model,
)


torch.set_num_threads(1)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


class NERL4TPRAnalyzerUnitTests(unittest.TestCase):
    def test_bounded_reference_replay_keeps_detection_counts_exact(self) -> None:
        reference = {
            "threshold": 0.5,
            "target_count": 100,
            "matched_target_count": 97,
            "tiny_target_count": 20,
            "matched_tiny_target_count": 18,
            "predicted_object_count": 110,
            "unmatched_predicted_object_count": 13,
            "unmatched_predicted_pixels": 250,
            "valid_pixel_count": 1_000_000,
            "pd": 0.97,
            "tiny_pd": 0.9,
            "fa": 0.00025,
            "miou": 0.8,
            "niou": 0.79,
            "pixel_precision": 0.81,
            "pixel_recall": 0.82,
            "pixel_f1": 0.815,
            "false_objects_per_image": 0.13,
            "test_loss": 0.001,
        }
        observed = copy.deepcopy(reference)
        observed["predicted_object_count"] += 1
        observed["unmatched_predicted_object_count"] += 1
        observed["unmatched_predicted_pixels"] += 1
        observed["fa"] += 4e-8
        observed["miou"] += 4e-4
        observed["niou"] -= 4e-4
        observed["pixel_f1"] += 4e-4
        observed["false_objects_per_image"] += 0.009
        observed["test_loss"] += 9e-8
        audit = subject.reference_replay_audit(observed, reference)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["policy"], subject.REPLAY_POLICY)
        self.assertTrue(audit["matched_total_and_tiny_target_counts_exact"])

        bad_target = copy.deepcopy(observed)
        bad_target["matched_target_count"] -= 1
        bad_target["pd"] = 0.96
        with self.assertRaisesRegex(ValueError, "exact count"):
            subject.reference_replay_audit(bad_target, reference)

        bad_boundary = copy.deepcopy(observed)
        bad_boundary["unmatched_predicted_pixels"] += 1
        with self.assertRaisesRegex(ValueError, "boundary count"):
            subject.reference_replay_audit(bad_boundary, reference)

        bad_soft = copy.deepcopy(observed)
        bad_soft["miou"] = reference["miou"] + 6e-4
        with self.assertRaisesRegex(ValueError, "metric differs"):
            subject.reference_replay_audit(bad_soft, reference)

    def test_mode_contract_and_finite_logit_boundary(self) -> None:
        self.assertEqual(
            subject.PUBLIC_MODES,
            (
                "current_g0",
                "tpr_g00625",
                "tpr_g0125",
                "tpr_g01875",
                "tpr_g025",
                "gpos025_l4_only",
            ),
        )
        expected = {
            "tpr_g00625": math.atanh(0.25),
            "tpr_g0125": math.atanh(0.5),
            "tpr_g01875": math.atanh(0.75),
        }
        for mode, logit in expected.items():
            with self.subTest(mode=mode):
                binding = subject.normalize_public_mode(mode)
                self.assertTrue(binding["protection_applied"])
                self.assertTrue(binding["finite_logit_representable"])
                self.assertEqual(binding["required_logit"], logit)
                self.assertFalse(binding["boundary_limit_counterfactual"])
        endpoint = subject.normalize_public_mode("tpr_g025")
        self.assertFalse(endpoint["finite_logit_representable"])
        self.assertIsNone(endpoint["required_logit"])
        self.assertTrue(endpoint["boundary_limit_counterfactual"])
        unprotected = subject.normalize_public_mode(subject.UNPROTECTED_MODE)
        self.assertFalse(unprotected["protection_applied"])
        self.assertTrue(unprotected["unprotected_reference"])

    def test_cached_l4_fusion_protects_only_the_binary_partition(self) -> None:
        transformed = torch.arange(256 * 4, dtype=torch.float64).reshape(
            1, 256, 2, 2
        )
        encoder = transformed.flip(-1).add(3.0)
        baseline = transformed.add(encoder).add(encoder)
        protection = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        protection[..., 0, 0] = 1.0
        placeholder = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        prepared = subject.PreparedL4TPRBatch(
            x1=placeholder,
            x2=placeholder,
            x3=placeholder,
            transformed4=transformed,
            encoder4=encoder,
            baseline4=baseline,
            d5=placeholder,
            evidence1=(placeholder, placeholder, placeholder),
            evidence2=(placeholder, placeholder),
            up4=placeholder,
            q4=placeholder,
            mask4=placeholder,
            protection4=protection,
        )

        current = subject.fuse_l4_public_mode(prepared, subject.CURRENT_MODE)
        self.assertTrue(torch.equal(_bits(current), _bits(baseline)))
        unprotected = subject.fuse_l4_public_mode(
            prepared, subject.UNPROTECTED_MODE
        )
        unprotected_expected = baseline.add(
            transformed.new_tensor(0.25).mul(transformed).sub(
                transformed.new_tensor(0.25).mul(encoder)
            )
        )
        self.assertTrue(
            torch.equal(_bits(unprotected), _bits(unprotected_expected))
        )

        protected = subject.fuse_l4_public_mode(prepared, "tpr_g025")
        protected_pixels = protection.bool().expand_as(protected)
        self.assertTrue(
            torch.equal(
                _bits(protected[protected_pixels]),
                _bits(baseline[protected_pixels]),
            )
        )
        eligible_pixels = torch.logical_not(protected_pixels)
        self.assertTrue(
            torch.equal(
                _bits(protected[eligible_pixels]),
                _bits(unprotected_expected[eligible_pixels]),
            )
        )

    def test_identity_manifest_binds_all_six_inputs_explicitly(self) -> None:
        manifest = subject.DEFAULT_IDENTITY_MANIFEST
        self.assertTrue(manifest.is_file())
        self.assertEqual(
            migration.file_sha256(manifest),
            subject.FROZEN_IDENTITY_MANIFEST_SHA256,
        )
        binding = migration.load_manifest_binding(
            manifest,
            expected_manifest_sha256=(
                subject.FROZEN_IDENTITY_MANIFEST_SHA256
            ),
            dataset="NUDT-SIRST",
            checkpoint_role="best_miou",
        )
        self.assertEqual(binding["epoch"], 420)
        self.assertEqual(binding["checkpoint_role"], "best_miou")
        self.assertTrue(Path(binding["checkpoint_path"]).is_file())
        self.assertTrue(Path(binding["reference_evaluation_path"]).is_file())
        self.assertEqual(
            binding["data_protocol_manifest"]["sha256"],
            "00edc6413dead3678f8b4c162c74ea7d8602f55ff413cb20ad1664587380319f",
        )

    def test_prepare_once_current_replay_and_decoder_call_counts(self) -> None:
        model, _ = build_formal_v4_qfg_v2_croa_l4_tpr_inference_model()
        model.eval()
        model.mode = "test"
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026080505)
        image = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.inference_mode():
            reference = model(image)

        counts = {"qfg": 0, "q4": 0, "q3": 0, "q2": 0, "p": 0, "cca4": 0}
        original_qfg = model.tpd_qfg.prepare
        original_stage = model.tpd_ner.forward_stage
        original_protection = model.ner_l4_tpr.build_protection
        original_cca4 = model.up_decoder4.coatt.forward

        def qfg_wrapper(*args, **kwargs):
            counts["qfg"] += 1
            return original_qfg(*args, **kwargs)

        def stage_wrapper(stage, *args, **kwargs):
            counts[f"q{stage}"] += 1
            return original_stage(stage, *args, **kwargs)

        def protection_wrapper(*args, **kwargs):
            counts["p"] += 1
            return original_protection(*args, **kwargs)

        def cca4_wrapper(*args, **kwargs):
            counts["cca4"] += 1
            return original_cca4(*args, **kwargs)

        model.tpd_qfg.prepare = qfg_wrapper
        model.tpd_ner.forward_stage = stage_wrapper
        model.ner_l4_tpr.build_protection = protection_wrapper
        model.up_decoder4.coatt.forward = cca4_wrapper
        try:
            with torch.inference_mode():
                prepared = subject.prepare_l4_tpr_batch(model, image)
                outputs = {
                    mode: subject.decode_l4_tpr_mode(model, prepared, mode)
                    for mode in subject.PUBLIC_MODES
                }
        finally:
            model.tpd_qfg.prepare = original_qfg
            model.tpd_ner.forward_stage = original_stage
            model.ner_l4_tpr.build_protection = original_protection
            model.up_decoder4.coatt.forward = original_cca4

        self.assertTrue(
            torch.equal(_bits(outputs[subject.CURRENT_MODE]), _bits(reference))
        )
        self.assertEqual(counts["qfg"], 1)
        self.assertEqual(counts["q4"], 1)
        self.assertEqual(counts["p"], 1)
        self.assertEqual(counts["cca4"], len(subject.PUBLIC_MODES))
        self.assertEqual(counts["q3"], len(subject.PUBLIC_MODES))
        self.assertEqual(counts["q2"], len(subject.PUBLIC_MODES))

    @staticmethod
    def _fixed_metrics() -> dict[str, object]:
        return {
            "threshold": 0.5,
            "target_count": 10,
            "matched_target_count": 8,
            "tiny_target_count": 4,
            "matched_tiny_target_count": 3,
            "pd": 0.8,
            "tiny_pd": 0.75,
            "fa": 0.1,
            "miou": 0.7,
            "niou": 0.71,
            "unmatched_predicted_pixels": 10,
            "component_false_positive_pixels": 10,
            "false_positive_pixels": 14,
            "background_false_positive_pixels": 14,
            "pixel_precision": 0.8,
            "pixel_recall": 0.75,
            "pixel_f1": 0.7741935483870968,
            "valid_pixel_count": 100,
            "pd_count_and_value": {
                "matched_target_count": 8,
                "target_count": 10,
                "pd": 0.8,
            },
            "tiny_pd_count_and_value": {
                "matched_tiny_target_count": 3,
                "tiny_target_count": 4,
                "tiny_pd": 0.75,
            },
        }

    @classmethod
    def _synthetic_payload(cls) -> dict[str, object]:
        modes: dict[str, object] = {}
        for index, mode in enumerate(subject.PUBLIC_MODES):
            difference = {
                "element_count": 100,
                "max_abs": 0.0 if mode == subject.CURRENT_MODE else 0.01,
                "absolute_difference_sum": (
                    0.0 if mode == subject.CURRENT_MODE else 0.5
                ),
            }
            unprotected_difference = {
                "element_count": 100,
                "max_abs": (
                    0.0 if mode == subject.UNPROTECTED_MODE else 0.01
                ),
                "absolute_difference_sum": (
                    0.0 if mode == subject.UNPROTECTED_MODE else 0.5
                ),
            }
            modes[mode] = {
                **subject.normalize_public_mode(mode),
                "fixed_threshold_0_5": cls._fixed_metrics(),
                "probability_sha256": f"{index + 1:064x}",
                "probability_difference_to_current": difference,
                "probability_difference_to_unprotected_gpos025": (
                    unprotected_difference
                ),
            }
        return {
            "schema": subject.SCHEMA,
            "status": "complete",
            "dataset": "NUDT-SIRST",
            "checkpoint_role": "best_miou",
            "seed": 42,
            "test_selected": True,
            "fixed_threshold": 0.5,
            "mode_order": list(subject.PUBLIC_MODES),
            "modes": modes,
            "reference_replay_audit": {
                "passed": True,
                "policy": subject.REPLAY_POLICY,
                "matched_total_and_tiny_target_counts_exact": True,
                "comparison": (
                    "current_g0_fixed_threshold_0_5_vs_existing_best_miou"
                ),
            },
            "identity_manifest_binding": {
                "sha256": subject.FROZEN_IDENTITY_MANIFEST_SHA256,
                "schema": migration.MANIFEST_SCHEMA,
                "dataset": "NUDT-SIRST",
                "checkpoint_role": "best_miou",
            },
            "checkpoint_binding": {
                "role": "best_miou",
                "sha256": "a" * 64,
                "epoch": 420,
            },
            "execution_audit": {
                "batch_count": 2,
                "encoder_tpd_qfg_prepare_count": 2,
                "up4_prepare_count": 2,
                "q4_forward_count": 2,
                "protection_build_count": 2,
                "decoder_mode_count_per_batch": len(subject.PUBLIC_MODES),
                "l4_cca_execution_count": 2 * len(subject.PUBLIC_MODES),
                "decoder_execution_count": 2 * len(subject.PUBLIC_MODES),
                "encoder_tpd_qfg_recomputed_per_mode": False,
                "q4_recomputed_per_mode": False,
                "protection_recomputed_per_mode": False,
            },
            "restoration_audit": {
                "model_state_unchanged": True,
                "model_state_sha256_before": "b" * 64,
                "model_state_sha256_after": "b" * 64,
            },
            "derived_checkpoint_written": False,
            "probability_cache_written": False,
            "feature_cache_written": False,
            "formal_training_started": False,
        }

    def test_output_validator_checks_joint_metric_and_execution_contract(self) -> None:
        payload = self._synthetic_payload()
        subject.validate_output_payload(payload)

        from analysis import compare_three_dataset_ner_l4_tpr_v1 as comparator

        normalized = comparator.validate_analyzer_payload(payload)
        self.assertEqual(normalized["dataset"], "NUDT-SIRST")
        self.assertEqual(normalized["checkpoint_role"], "best_miou")

        bad_alias = copy.deepcopy(payload)
        bad_alias["modes"]["tpr_g0125"]["fixed_threshold_0_5"][
            "component_false_positive_pixels"
        ] = 11
        with self.assertRaises(ValueError):
            subject.validate_output_payload(bad_alias)

        bad_prepare = copy.deepcopy(payload)
        bad_prepare["execution_audit"]["q4_forward_count"] = 12
        with self.assertRaises(ValueError):
            subject.validate_output_payload(bad_prepare)

        bad_boundary = copy.deepcopy(payload)
        bad_boundary["modes"]["tpr_g025"][
            "finite_logit_representable"
        ] = True
        with self.assertRaises(ValueError):
            subject.validate_output_payload(bad_boundary)


if __name__ == "__main__":
    unittest.main()
