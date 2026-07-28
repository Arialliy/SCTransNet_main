import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from analysis import analyze_tpd_clean_v8_mprs_mechanism as subject


def sha(character: str = "a") -> str:
    return character * 64


def correlation_stats(multiplier: int = 1) -> dict:
    return {
        "count": 2 * multiplier,
        "sum_correction": 1.0 * multiplier,
        "sum_keep_linear": 1.0 * multiplier,
        "sum_sq_correction": 1.0 * multiplier,
        "sum_sq_keep_linear": 1.0 * multiplier,
        "sum_product": 1.0 * multiplier,
    }


def block_report(image_count: int = 2) -> dict:
    return {
        "target_sum": 1.0,
        "target_count": 1,
        "target_mean_abs": 1.0,
        "hard_negative_sum": 0.5,
        "hard_negative_count": 1,
        "hard_negative_mean_abs": 0.5,
        "target_correction_lift": 1.0 / (0.5 + 1e-6),
        "mean_abs_correction": 0.25,
        "mean_abs_scale": 0.1,
        "image_count": image_count,
        "image_coverage_complete": True,
        "pooled_overlap_removed_count": 0,
        "target_priority_masks_disjoint": True,
        "diagnostic_forward_max_abs_difference": 1e-7,
        "diagnostic_forward_max_allowed_difference": 1e-6,
        "diagnostic_production_forward_within_frozen_tolerance": True,
        "correction_keep_correlation": 1.0,
        "correlation_sufficient_statistics": correlation_stats(),
    }


class AnalyzerHardeningTests(unittest.TestCase):
    def test_ordered_ids_hash_is_order_sensitive_and_rejects_bad_ids(self):
        first = subject.ordered_validation_ids_sha256(["a", "b"])
        second = subject.ordered_validation_ids_sha256(["b", "a"])
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 64)
        with self.assertRaises(ValueError):
            subject.ordered_validation_ids_sha256([])
        with self.assertRaises(ValueError):
            subject.ordered_validation_ids_sha256(["a", "a"])
        with self.assertRaises(ValueError):
            subject.ordered_validation_ids_sha256(["a", ""])

    def test_recursive_finite_audit_rejects_all_numeric_nonfinite_forms(self):
        subject._require_all_numeric_finite(
            {
                "python": [1, 2.0],
                "numpy": np.asarray([3.0], dtype=np.float32),
                "torch": torch.ones(1),
            },
            "finite",
        )
        for value in (
            float("nan"),
            np.asarray([np.inf]),
            torch.tensor([float("-inf")]),
        ):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    subject._require_all_numeric_finite(value, "bad")

    def test_output_directory_must_be_outside_formal_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            inside = root / "analysis"
            sibling = Path(temporary) / "analysis"
            subject.require_analysis_output_separate(root, sibling)
            with self.assertRaises(ValueError):
                subject.require_analysis_output_separate(root, root)
            with self.assertRaises(ValueError):
                subject.require_analysis_output_separate(root, inside)

    def test_run_cli_requires_cuda_zero_and_registered_gpu(self):
        with self.assertRaises(SystemExit):
            subject.parse_args(["--run"])
        with self.assertRaises(SystemExit):
            subject.parse_args(["--run", "--device", "cuda:0"])
        args = subject.parse_args(
            [
                "--run",
                "--device",
                "cuda:0",
                "--physical-gpu",
                "2",
            ]
        )
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.physical_gpu, "2")

    def _binding_fixture(self, root: Path):
        run_dir = root / "formal" / "run"
        run_dir.mkdir(parents=True)
        paths = {
            label: run_dir / f"{label}.bin"
            for label in subject.FORMAL_INPUT_LABELS
        }
        for index, path in enumerate(paths.values()):
            path.write_bytes(f"input-{index}".encode("utf-8"))
        registry_path = root / "registry.json"
        registry_path.write_text("{}\n", encoding="utf-8")
        output_dir = root / "analysis"
        validation_ids = ["id-1", "id-2"]
        dataset_dir = root / "datasets"
        dataset_root = dataset_dir / subject.DATASET
        (dataset_root / "images").mkdir(parents=True)
        (dataset_root / "masks").mkdir(parents=True)
        for index, identifier in enumerate(validation_ids):
            (dataset_root / "images" / f"{identifier}.png").write_bytes(
                f"image-{index}".encode("utf-8")
            )
            (dataset_root / "masks" / f"{identifier}.png").write_bytes(
                f"mask-{index}".encode("utf-8")
            )
        current_validation = subject.current_validation_data_binding(
            dataset_root,
            validation_ids,
        )
        source_locks = {
            "tpd_clean_v7_dch_exact_source_lock": sha("1"),
            "training_data": sha("2"),
        }
        source_identity = {
            "dataset": subject.DATASET,
            "variant": "tpd_clean_v7_dch_full",
            "seed": 42,
            "comparison_role": "pd_primary",
            "checkpoint_role": "best_validation_pd_primary",
            "source_locks": source_locks,
        }

        def fingerprint(name: str, count: int) -> dict:
            return {
                "schema": "ordered-v1",
                "name": name,
                "count": count,
                "sha256": sha("3"),
            }

        protocol = {
            "run_identity": {
                "architecture_id": sha("4"),
                "builder_manifest_sha256": sha("5"),
                "contract_sha256": sha("6"),
                "data_sha256": sha("7"),
                "split_sha256": sha("8"),
                "source_locks": source_locks,
                "ordered_data_fingerprints": {
                    key: (
                        current_validation["fingerprint"]
                        if key == "validation_samples"
                        else fingerprint(key, 1)
                    )
                    for key in (
                        subject.EXPECTED_ORDERED_DATA_FINGERPRINTS
                    )
                },
                "ordered_split_fingerprints": {
                    key: fingerprint(
                        key,
                        2
                        if key in {"validation", "full_validation"}
                        else 3,
                    )
                    for key in (
                        subject.EXPECTED_ORDERED_SPLIT_FINGERPRINTS
                    )
                },
            },
            "arguments": {"dataset_dir": str(dataset_dir)},
        }
        split = {
            "used_val_ids": validation_ids,
            "hashes": {"used_val_sha256": sha("9")},
        }
        input_hashes = {
            label: subject.file_sha256(path)
            for label, path in paths.items()
        }
        artifacts = {
            "paths": paths,
            "input_sha256": input_hashes,
            "source_identity": source_identity,
            "protocol": protocol,
            "split": split,
            "checkpoint": {
                "state_dict": {
                    "weight": torch.tensor([1.0], dtype=torch.float32)
                }
            },
        }
        registry_payload = {
            "source_identity": source_identity,
            "validation": {
                "validation_ids": validation_ids,
                "validation_count": 2,
            },
            "input_sha256_before": input_hashes,
        }
        registry = {
            "fixed": {
                "threshold": 0.5,
                "registry_labels": ["fixed_0.5"],
                "registry_kinds": ["fixed_threshold"],
            }
        }
        job = {
            "variant": "tpd_clean_v7_dch_full",
            "seed": 42,
            "role": "pd_primary",
            "run_dir": run_dir,
            "checkpoint": paths["checkpoint"],
            "output": registry_path,
        }
        registry_seal = {
            "control_manifest": str(root / "control_manifest.json"),
            "control_manifest_sha256": sha("a"),
            "registry_path": str(registry_path.resolve()),
            "registry_sha256": subject.file_sha256(registry_path),
            "mechanism_report": str(root / "mechanism_report.json"),
            "mechanism_report_sha256": sha("b"),
            "source_locks": {},
            "source_locks_sha256": sha("c"),
        }
        return (
            job,
            artifacts,
            registry_payload,
            registry,
            registry_seal,
            output_dir,
        )

    def test_job_binding_covers_formal_registry_split_and_source_fingerprints(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._binding_fixture(Path(temporary))
            binding = subject._job_binding(*fixture)
            self.assertEqual(binding["schema"], subject.JOB_BINDING_SCHEMA)
            self.assertEqual(
                binding["expected_job"]["checkpoint_sha256"],
                fixture[1]["input_sha256"]["checkpoint"],
            )
            self.assertEqual(
                binding["ordered_validation"]["ids"],
                ["id-1", "id-2"],
            )
            self.assertEqual(
                binding["registry"]["points_sha256"],
                subject.canonical_json_sha256(fixture[3]),
            )
            self.assertEqual(
                binding["source_locks_sha256"],
                subject.canonical_json_sha256(
                    fixture[1]["source_identity"]["source_locks"]
                ),
            )
            self.assertEqual(len(binding["binding_sha256"]), 64)

            changed = copy.deepcopy(fixture[2])
            changed["validation"]["validation_ids"].reverse()
            with self.assertRaises(ValueError):
                subject._job_binding(
                    fixture[0],
                    fixture[1],
                    changed,
                    fixture[3],
                    fixture[4],
                    fixture[5],
                )

    def test_registry_source_rejects_wrong_job_identity_and_binds_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                job,
                artifacts,
                registry_payload,
                registry,
                registry_seal,
                _,
            ) = (
                self._binding_fixture(root)
            )
            payload = {
                "schema": subject.v7_diag.CHECKPOINT_SCHEMA,
                "formal_inputs_unchanged": True,
                "training_performed": False,
                "checkpoint_reselection_permitted": False,
                "official_test_accessed": False,
                "variant": job["variant"],
                "seed": job["seed"],
                "checkpoint_role": job["role"],
                "checkpoint": str(job["checkpoint"].resolve()),
                "source_identity": registry_payload["source_identity"],
                "input_sha256_before": artifacts["input_sha256"],
                "input_sha256_after": artifacts["input_sha256"],
                "validation": registry_payload["validation"],
                "operating_points": registry,
            }
            Path(job["output"]).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            registry_seal["registry_sha256"] = subject.file_sha256(
                Path(job["output"])
            )
            with mock.patch.object(
                subject,
                "sealed_v7_registry_binding",
                return_value=registry_seal,
            ):
                loaded, points, seal = subject._registry_source(job)
            self.assertEqual(loaded["seed"], 42)
            self.assertEqual(points, registry)
            self.assertEqual(seal, registry_seal)

            payload["seed"] = 3407
            Path(job["output"]).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            registry_seal["registry_sha256"] = subject.file_sha256(
                Path(job["output"])
            )
            with (
                mock.patch.object(
                    subject,
                    "sealed_v7_registry_binding",
                    return_value=registry_seal,
                ),
                self.assertRaises(ValueError),
            ):
                subject._registry_source(job)

    def test_masked_stats_are_per_block_and_reject_nonfinite_diagnostics(self):
        diagnostics = {}
        for name in subject.EXPECTED_BLOCK_NAMES:
            correction = torch.tensor(
                [[[[0.0, 1.0], [2.0, 3.0]]]],
                dtype=torch.float32,
            )
            diagnostics[name] = {
                "phase_correction": correction,
                "context_aligned": torch.zeros_like(correction),
                "scale": torch.tensor(0.1),
            }
        target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        negative = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]])
        result = subject._masked_correction_stats(
            diagnostics,
            target,
            negative,
        )
        self.assertEqual(
            tuple(result["blocks"]),
            subject.EXPECTED_BLOCK_NAMES,
        )
        self.assertGreater(result["target_count"], 0)
        self.assertGreater(result["hard_negative_count"], 0)
        self.assertIn(
            "correlation_sufficient_statistics",
            result["blocks"][subject.EXPECTED_BLOCK_NAMES[0]],
        )

        broken = copy.deepcopy(diagnostics)
        broken[subject.EXPECTED_BLOCK_NAMES[0]][
            "phase_correction"
        ][0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            subject._masked_correction_stats(broken, target, negative)

    def test_hard_negative_mask_is_strict_and_excludes_whole_near_component(
        self,
    ):
        probability = np.zeros((12, 12), dtype=np.float32)
        target = np.zeros_like(probability)
        target[4, 4] = 1.0
        probability[0, 0] = 0.5
        probability[4, 7:10] = 0.9
        probability[10, 10] = 0.9
        result = subject.hard_negative_mask(probability, target)
        self.assertFalse(result[0, 0])
        self.assertFalse(bool(result[4, 7:10].any()))
        self.assertTrue(result[10, 10])
        self.assertEqual(int(result.sum()), 1)

    def test_target_priority_is_applied_after_pooling_per_block(self):
        diagnostics = {}
        for name in subject.EXPECTED_BLOCK_NAMES:
            correction = torch.ones((1, 1, 2, 2))
            diagnostics[name] = {
                "phase_correction": correction,
                "context_aligned": torch.zeros_like(correction),
                "scale": torch.tensor(0.1),
            }
        target = torch.zeros((1, 1, 4, 4))
        negative = torch.zeros_like(target)
        target[0, 0, 0, 0] = 1.0
        negative[0, 0, 1, 1] = 1.0
        negative[0, 0, 3, 3] = 1.0
        result = subject._masked_correction_stats(
            diagnostics,
            target,
            negative,
        )
        self.assertEqual(
            result["pooled_overlap_removed_count"],
            len(subject.EXPECTED_BLOCK_NAMES),
        )
        for block in result["blocks"].values():
            self.assertEqual(block["target_count"], 1)
            self.assertEqual(block["hard_negative_count"], 1)
            self.assertEqual(block["pooled_overlap_removed_count"], 1)
            self.assertTrue(block["target_priority_masks_disjoint"])

    def test_diagnostic_capture_returns_formal_output_and_restores_forward(self):
        model = torch.nn.Module()
        model.mtc = torch.nn.Module()
        for embedding_name, block_count in (
            ("embeddings_1", 4),
            ("embeddings_2", 3),
        ):
            embedding = torch.nn.Module()
            embedding.blocks = torch.nn.ModuleList(
                [
                    subject.TPDCleanV8MPRSDCHBlock(
                        1,
                        activate=False,
                        context_gate=1.0,
                    )
                    for _ in range(block_count)
                ]
            )
            setattr(model.mtc, embedding_name, embedding)
        first = model.mtc.embeddings_1.blocks[0]
        original_function = first.forward.__func__
        x = torch.randn(1, 1, 4, 4)
        expected = first(x)
        with subject.capture_mprs_diagnostics(model) as (
            diagnostics,
            checks,
        ):
            observed = first(x)
            self.assertTrue(torch.equal(observed, expected))
            self.assertIn(subject.EXPECTED_BLOCK_NAMES[0], diagnostics)
            self.assertTrue(
                checks[subject.EXPECTED_BLOCK_NAMES[0]][
                    "within_frozen_tolerance"
                ]
            )
        self.assertIs(first.forward.__func__, original_function)

        with self.assertRaisesRegex(RuntimeError, "forced"):
            with subject.capture_mprs_diagnostics(model):
                raise RuntimeError("forced")
        self.assertIs(first.forward.__func__, original_function)

    def test_block_finalizer_hard_fails_empty_coverage(self):
        totals = {}
        for name in subject.EXPECTED_BLOCK_NAMES:
            totals[name] = {
                "target_sum": 1.0,
                "target_count": 1,
                "hard_negative_sum": 0.5,
                "hard_negative_count": 1,
                "mean_abs_correction_sum": 0.5,
                "mean_abs_scale_sum": 0.2,
                "image_count": 2,
                "correlation_sufficient_statistics": correlation_stats(),
            }
        reports = subject._finalize_block_reports(totals, 2)
        self.assertEqual(tuple(reports), subject.EXPECTED_BLOCK_NAMES)
        self.assertEqual(
            reports[subject.EXPECTED_BLOCK_NAMES[0]][
                "correction_keep_correlation"
            ],
            1.0,
        )
        totals[subject.EXPECTED_BLOCK_NAMES[-1]][
            "hard_negative_count"
        ] = 0
        with self.assertRaises(RuntimeError):
            subject._finalize_block_reports(totals, 2)

    def test_paired_topology_uses_v7_reference_and_closes_to_v7_view(self):
        def point(records):
            return {"gt_topology": {"per_gt": records}}

        v7 = point(
            [
                {
                    "identifier": "a",
                    "gt_index": 1,
                    "gt_area": 4,
                    "overlapping_prediction_components": 2,
                    "largest_fragment_fraction": 0.75,
                },
                {
                    "identifier": "b",
                    "gt_index": 2,
                    "gt_area": 3,
                    "overlapping_prediction_components": 0,
                    "largest_fragment_fraction": 0.0,
                },
            ]
        )
        v8 = point(
            [
                {
                    "identifier": "a",
                    "gt_index": 1,
                    "gt_area": 4,
                    "overlapping_prediction_components": 0,
                    "largest_fragment_fraction": 0.9,
                },
                {
                    "identifier": "b",
                    "gt_index": 2,
                    "gt_area": 3,
                    "overlapping_prediction_components": 1,
                    "largest_fragment_fraction": 1.0,
                },
            ]
        )
        paired = subject.paired_topology_from_decorated_points(v7, v8)
        self.assertEqual(paired["reference_gt_count"], 1)
        self.assertEqual(paired["v8_covered_reference_gt_count"], 0)
        self.assertEqual(
            paired["paired_gt"][0]["v8_largest_fragment_fraction"],
            0.0,
        )
        points = {
            "fixed": {
                "v7": {
                    "overlap_covered_gt_count": 1,
                    "fragment_excess_total": 1,
                    "largest_fragment_fractions": [0.75],
                },
                "paired_topology": paired,
            }
        }
        aggregate = subject.topology_aggregate_from_operating_points(points)
        self.assertEqual(aggregate["v7_fragment_excess_total"], 1)
        self.assertEqual(aggregate["v8_fragment_excess_total"], 0)

        tampered = copy.deepcopy(points)
        record = tampered["fixed"]["paired_topology"]["paired_gt"][0]
        record["v7_largest_fragment_fraction"] = 0.5
        tampered["fixed"]["paired_topology"].update(
            subject._paired_gt_summary([record])
        )
        with self.assertRaises(ValueError):
            subject.topology_aggregate_from_operating_points(tampered)

    def test_paired_topology_rejects_invalid_area_and_zero_covered_fraction(self):
        record = {
            "identifier": "a",
            "gt_index": 1,
            "gt_area": 0,
            "v7_reference_coverage": 1,
            "v8_reference_coverage": 1,
            "v7_overlapping_prediction_components": 1,
            "v8_overlapping_prediction_components": 1,
            "v7_fragment_excess": 0,
            "v8_fragment_excess": 0,
            "v7_largest_fragment_fraction": 1.0,
            "v8_largest_fragment_fraction": 1.0,
        }
        with self.assertRaises(ValueError):
            subject._paired_gt_summary([record])
        record["gt_area"] = 1
        record["v8_largest_fragment_fraction"] = 0.0
        with self.assertRaises(ValueError):
            subject._paired_gt_summary([record])

    def _valid_saved_payload(self):
        validation_ids = ["id-1", "id-2"]
        validation_digest = subject.ordered_validation_ids_sha256(
            validation_ids
        )
        formal_hashes = {
            label: sha(str(index + 1))
            for index, label in enumerate(subject.FORMAL_INPUT_LABELS)
        }
        state_sha = sha("e")
        binding = {
            "expected_job": {
                "checkpoint_sha256": formal_hashes["checkpoint"],
                "checkpoint_state_dict_sha256": state_sha,
            },
            "formal_input_sha256": formal_hashes,
            "ordered_validation": {
                "ids": validation_ids,
                "ordered_ids_sha256": validation_digest,
            },
        }
        blocks = {
            name: block_report() for name in subject.EXPECTED_BLOCK_NAMES
        }
        global_correlation = correlation_stats(
            len(subject.EXPECTED_BLOCK_NAMES)
        )
        registration = {
            "threshold": 0.5,
            "registry_labels": ["fixed_0.5"],
            "registry_kinds": ["fixed_threshold"],
        }
        topology_view = {
            "pd": 1.0,
            "fa": 0.0,
            "miou": 1.0,
            "matched_target_count": 1,
            "target_count": 1,
            "unmatched_predicted_object_count": 0,
            "fragment_excess_total": 0,
            "overlap_covered_gt_count": 1,
            "largest_fragment_fraction_mean": 1.0,
            "largest_fragment_fraction_p10": 1.0,
            "largest_fragment_fractions": [1.0],
        }
        paired_record = {
            "identifier": "id-1",
            "gt_index": 1,
            "gt_area": 1,
            "v7_reference_coverage": 1,
            "v8_reference_coverage": 1,
            "v7_overlapping_prediction_components": 1,
            "v8_overlapping_prediction_components": 1,
            "v7_fragment_excess": 0,
            "v8_fragment_excess": 0,
            "v7_largest_fragment_fraction": 1.0,
            "v8_largest_fragment_fraction": 1.0,
        }
        paired_topology = {
            "reference_definition": (
                "V7_overlapping_prediction_components_gt0"
            ),
            "uncovered_v8_largest_fragment_fraction": 0.0,
            "paired_gt": [paired_record],
            **subject._paired_gt_summary([paired_record]),
        }
        shift_count = 2 * (len(subject.SHIFT_OFFSETS) - 1)
        job = {
            "variant": "tpd_clean_v7_dch_full",
            "seed": 42,
            "role": "pd_primary",
            "checkpoint": Path("/tmp/checkpoint.pth"),
        }
        payload = {
            "schema": subject.JOB_SCHEMA,
            "status": "complete",
            "variant": job["variant"],
            "v8_variant": subject.V8_VARIANT_BY_V7[job["variant"]],
            "seed": job["seed"],
            "checkpoint_role": job["role"],
            "checkpoint": str(job["checkpoint"].resolve()),
            "checkpoint_sha256": formal_hashes["checkpoint"],
            "job_binding": binding,
            "strict_load_v7": True,
            "strict_load_v8": True,
            "state_layout_equal": True,
            "all_gate_probabilities_from_production_forward": True,
            "finite": True,
            "validation_count": 2,
            "validation_ids": validation_ids,
            "ordered_validation_ids_sha256": validation_digest,
            "input_sha256_before": formal_hashes,
            "input_sha256_after": formal_hashes,
            "model_state_sha256_before": {
                "v7": state_sha,
                "v8": state_sha,
            },
            "model_state_sha256_after": {
                "v7": state_sha,
                "v8": state_sha,
            },
            "coverage": {
                "validation_images_expected": 2,
                "validation_images_processed": 2,
                "processed_ordered_ids_sha256": validation_digest,
                "target_pixel_count": 2,
                "target_image_count": 2,
                "hard_negative_pixel_count": 2,
                "hard_negative_image_count": 2,
                "block_count": len(subject.EXPECTED_BLOCK_NAMES),
                "block_image_evaluations": (
                    len(subject.EXPECTED_BLOCK_NAMES) * 2
                ),
                "shift_image_count": 2,
                "shift_error_count_per_model": shift_count,
                "operating_point_count": 1,
                "complete": True,
            },
            "device": {
                "device": "cuda:0",
                "logical_device": "cuda:0",
                "physical_gpu_index": 2,
                "physical_gpu_uuid": subject.v7_diag.POSTPROCESS_GPUS["2"],
                "visible_device_name": "NVIDIA GeForce RTX 5090",
                "determinism": {
                    "cublas_workspace_config": ":4096:8",
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "deterministic_algorithms": True,
                    "float32_matmul_precision": "highest",
                },
            },
            "v8_model_metadata": {
                "variant": subject.V8_VARIANT_BY_V7[job["variant"]],
                "candidate_family": (
                    "spd_anchored_tpd_clean_v8_mprs_dch"
                ),
                "mainline_contract": "Keep-Context-Saliency",
                "saliency_formula": (
                    "S_p=(max_q(Z_q)-C0)+(Z_p-C0)/3"
                ),
                "standard_forward_conv2d_calls_per_block": 3,
                "shallow_embedding_parameters": 66_176,
                "total_parameters": 10_843_155,
                "context_gate": subject.clean_v8_mprs_dch_variant_spec(
                    subject.V8_VARIANT_BY_V7[job["variant"]]
                )["context_gate"],
                "fusion_formula": subject.clean_v8_mprs_dch_variant_spec(
                    subject.V8_VARIANT_BY_V7[job["variant"]]
                )["fusion_formula"],
                "variant_spec": subject.json_ready(
                    subject.clean_v8_mprs_dch_variant_spec(
                        subject.V8_VARIANT_BY_V7[job["variant"]]
                    )
                ),
                "full_initialization_sha256": sha("f"),
                "shared_initialization_sha256": sha("0"),
            },
            "numeric_audit": {
                "raw_outputs": {
                    "v7_tensor_count": 12,
                    "v7_element_count": 12,
                    "v8_tensor_count": 12,
                    "v8_element_count": 12,
                    "shifted_outputs_all_finite": True,
                    "diagnostic_forward_atol": (
                        subject.DIAGNOSTIC_FORWARD_ATOL
                    ),
                    "diagnostic_forward_rtol": (
                        subject.DIAGNOSTIC_FORWARD_RTOL
                    ),
                    "diagnostic_forward_max_abs_difference": 1e-7,
                    "diagnostic_forward_max_allowed_difference": 1e-6,
                    "diagnostic_forward_check_count": (
                        len(subject.EXPECTED_BLOCK_NAMES) * 2
                    ),
                    "diagnostic_production_forward_within_frozen_tolerance": (
                        True
                    ),
                    "all_finite": True,
                },
                "losses": {
                    "v7": [0.1, 0.2],
                    "v8": [0.1, 0.2],
                    "count_per_model": 2,
                    "all_finite": True,
                },
                "correlations_all_finite": True,
                "block_values_all_finite": True,
            },
            "correction_selectivity": {
                "target_sum": 7.0,
                "target_count": 7,
                "hard_negative_sum": 3.5,
                "hard_negative_count": 7,
                "target_mean_abs": 1.0,
                "hard_negative_mean_abs": 0.5,
                "target_correction_lift": 1.0 / (0.5 + 1e-6),
                "pooled_overlap_removed_count": 0,
                "correction_keep_correlation": 1.0,
                "correlation_sufficient_statistics": global_correlation,
                "blocks": blocks,
            },
            "operating_points": {
                "fixed": {
                    **registration,
                    "v7": copy.deepcopy(topology_view),
                    "v8": copy.deepcopy(topology_view),
                    "paired_topology": paired_topology,
                    "delta_v8_minus_v7": {
                        "pd": 0.0,
                        "fa": 0.0,
                        "miou": 0.0,
                        "fragment_excess_total": 0,
                    },
                }
            },
            "toroidal_grid_offset_stress": {
                "definition": "toroidal_grid_offset_stress",
                "translation_equivariance_claim_permitted": False,
                "all_probabilities_from_production_forward": True,
                "subset_ids": validation_ids,
                "offsets": [
                    list(offset) for offset in subject.SHIFT_OFFSETS
                ],
                "crop_pixels": subject.SHIFT_CROP,
                "v7_errors": [0.1] * shift_count,
                "v8_errors": [0.1] * shift_count,
            },
            "formal_inputs_unchanged": True,
            "training_performed": False,
            "checkpoint_reselection_permitted": False,
            "official_test_accessed": False,
        }
        payload["topology_aggregate"] = (
            subject.topology_aggregate_from_operating_points(
                payload["operating_points"]
            )
        )
        return job, payload, binding, {"fixed": registration}

    def test_saved_job_revalidation_rejects_identity_nonfinite_and_empty_masks(
        self,
    ):
        job, payload, binding, registry = self._valid_saved_payload()
        patches = (
            mock.patch.object(
                subject.v7_diag,
                "validate_job_artifacts",
                return_value={},
            ),
            mock.patch.object(
                subject,
                "_registry_source",
                return_value=({}, registry, {}),
            ),
            mock.patch.object(
                subject,
                "_job_binding",
                return_value=binding,
            ),
        )
        with patches[0], patches[1], patches[2]:
            validated = subject._validate_job_payload(
                job,
                payload,
                output_dir=Path("/tmp/analysis"),
            )
            self.assertEqual(validated["checkpoint_role"], "pd_primary")

            mutations = []
            wrong_role = copy.deepcopy(payload)
            wrong_role["checkpoint_role"] = "last"
            mutations.append(wrong_role)
            nonfinite = copy.deepcopy(payload)
            nonfinite["numeric_audit"]["losses"]["v7"][0] = float("nan")
            mutations.append(nonfinite)
            empty_target = copy.deepcopy(payload)
            empty_target["coverage"]["target_pixel_count"] = 0
            mutations.append(empty_target)
            empty_negative = copy.deepcopy(payload)
            empty_negative["correction_selectivity"][
                "hard_negative_count"
            ] = 0
            mutations.append(empty_negative)
            missing_block = copy.deepcopy(payload)
            missing_block["correction_selectivity"]["blocks"].pop(
                subject.EXPECTED_BLOCK_NAMES[-1]
            )
            mutations.append(missing_block)
            stale_topology = copy.deepcopy(payload)
            stale_topology["topology_aggregate"][
                "v8_fragment_excess_total"
            ] += 1
            mutations.append(stale_topology)
            boolean_count = copy.deepcopy(payload)
            boolean_count["coverage"]["target_pixel_count"] = True
            mutations.append(boolean_count)
            string_probability = copy.deepcopy(payload)
            string_probability["operating_points"]["fixed"]["v8"][
                "pd"
            ] = "1.0"
            mutations.append(string_probability)
            state_mismatch = copy.deepcopy(payload)
            state_mismatch["model_state_sha256_before"]["v8"] = sha("d")
            mutations.append(state_mismatch)
            tolerance_failure = copy.deepcopy(payload)
            tolerance_failure["numeric_audit"]["raw_outputs"][
                "diagnostic_forward_max_abs_difference"
            ] = 2e-6
            mutations.append(tolerance_failure)
            for changed in mutations:
                with self.subTest(keys=tuple(changed)):
                    with self.assertRaises(ValueError):
                        subject._validate_job_payload(
                            job,
                            changed,
                            output_dir=Path("/tmp/analysis"),
                        )

    def _aggregate_payload(self, job: dict, validation_digest: str) -> dict:
        blocks = {
            name: block_report() for name in subject.EXPECTED_BLOCK_NAMES
        }
        return {
            "variant": job["variant"],
            "seed": job["seed"],
            "checkpoint_role": job["role"],
            "checkpoint": str(job["checkpoint"]),
            "checkpoint_sha256": sha("a"),
            "job_binding": {
                "binding_sha256": sha("b"),
                "counterfactual_execution_sha256": sha("1"),
                "current_validation_data_sha256": sha("2"),
                "v8_protocol_sha256": sha("3"),
                "v8_preflight_amendment_sha256": sha("4"),
                "registry": {
                    "points_sha256": sha("c"),
                    "source_sha256": sha("d"),
                },
            },
            "ordered_validation_ids_sha256": validation_digest,
            "validation_count": 2,
            "finite": True,
            "correction_selectivity": {
                "target_sum": 7.0,
                "target_count": 7,
                "hard_negative_sum": 3.5,
                "hard_negative_count": 7,
                "blocks": blocks,
            },
            "topology_aggregate": {
                "v7_fragment_excess_total": 1,
                "v8_fragment_excess_total": 1,
                "v7_covered_reference_gt_count": 1,
                "v8_covered_reference_gt_count": 1,
                "v7_largest_fragment_fractions": [1.0],
                "v8_largest_fragment_fractions": [1.0],
            },
            "toroidal_grid_offset_stress": {
                "v7_errors": [0.1],
                "v8_errors": [0.1],
            },
            "strict_load_v8": True,
        }

    def test_block_aggregation_uses_sums_and_counts_not_role_means(self):
        first_blocks = {
            name: block_report(image_count=1)
            for name in subject.EXPECTED_BLOCK_NAMES
        }
        second_blocks = {
            name: block_report(image_count=2)
            for name in subject.EXPECTED_BLOCK_NAMES
        }
        for block in second_blocks.values():
            block["target_sum"] = 9.0
            block["target_count"] = 3
            block["target_mean_abs"] = 3.0
        reports = subject._aggregate_block_reports(
            [
                {
                    "validation_count": 1,
                    "correction_selectivity": {"blocks": first_blocks},
                },
                {
                    "validation_count": 2,
                    "correction_selectivity": {"blocks": second_blocks},
                },
            ]
        )
        for report in reports.values():
            self.assertEqual(report["target_sum"], 10.0)
            self.assertEqual(report["target_count"], 4)
            self.assertEqual(report["target_mean_abs"], 2.5)

    def _aggregate_with_mutation(self, mutation=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results_root = root / "formal"
            output_dir = root / "analysis"
            output_dir.mkdir()
            jobs = []
            validation_digest = sha("f")
            for variant in subject.V8_VARIANT_BY_V7:
                for seed in subject.v7_diag.SEEDS:
                    for role in subject.v7_diag.CHECKPOINT_SPECS:
                        job = {
                            "variant": variant,
                            "seed": seed,
                            "role": role,
                            "checkpoint": (
                                results_root
                                / variant
                                / f"seed-{seed}"
                                / f"{role}.pth"
                            ),
                        }
                        jobs.append(job)
                        payload = self._aggregate_payload(
                            job,
                            validation_digest,
                        )
                        if mutation is not None:
                            mutation(job, payload)
                        path = subject.job_output_path(output_dir, job)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(
                            json.dumps(payload),
                            encoding="utf-8",
                        )
            with (
                mock.patch.object(
                    subject,
                    "expected_jobs",
                    return_value=jobs,
                ),
                mock.patch.object(
                    subject,
                    "_validate_job_payload",
                    side_effect=lambda job, payload, output_dir: payload,
                ),
            ):
                return subject.aggregate(
                    results_root,
                    output_dir,
                    overwrite=False,
                )

    def test_counterfactual_gate_boundaries_are_exact(self):
        def exact_lift(_job, payload):
            correction = payload["correction_selectivity"]
            correction["target_sum"] = 1e-6
            correction["target_count"] = 1
            correction["hard_negative_sum"] = 0.0
            correction["hard_negative_count"] = 1

        lift_report = self._aggregate_with_mutation(exact_lift)
        self.assertFalse(lift_report["counterfactual_gate_pass"])
        self.assertTrue(
            all(
                group["target_correction_lift"] == 1.0
                and not group["target_correction_lift_pass"]
                for group in lift_report["groups"].values()
            )
        )

        def fragment_increase(job, payload):
            if job["role"] == "pd_primary":
                payload["topology_aggregate"][
                    "v8_fragment_excess_total"
                ] += 1

        fragment_report = self._aggregate_with_mutation(fragment_increase)
        self.assertFalse(fragment_report["counterfactual_gate_pass"])
        self.assertTrue(
            all(
                not group["fragment_excess_nonincrease_pass"]
                for group in fragment_report["groups"].values()
            )
        )

        def median_decline(_job, payload):
            payload["topology_aggregate"][
                "v8_largest_fragment_fractions"
            ] = [0.9]

        median_report = self._aggregate_with_mutation(median_decline)
        self.assertFalse(median_report["counterfactual_gate_pass"])
        self.assertTrue(
            all(
                not group["largest_fragment_nondecrease_pass"]
                for group in median_report["groups"].values()
            )
        )

        def coverage_decline(job, payload):
            if job["role"] == "pd_primary":
                payload["topology_aggregate"][
                    "v8_covered_reference_gt_count"
                ] = 0

        coverage_report = self._aggregate_with_mutation(coverage_decline)
        self.assertFalse(coverage_report["counterfactual_gate_pass"])
        self.assertTrue(
            all(
                not group["reference_coverage_nondecrease_pass"]
                for group in coverage_report["groups"].values()
            )
        )

        def shift_boundary(_job, payload):
            payload["toroidal_grid_offset_stress"]["v8_errors"] = [0.11]

        boundary_report = self._aggregate_with_mutation(shift_boundary)
        self.assertTrue(boundary_report["counterfactual_gate_pass"])
        self.assertTrue(
            all(
                group["shift_ratio_pass"]
                for group in boundary_report["groups"].values()
            )
        )

        def shift_above(_job, payload):
            payload["toroidal_grid_offset_stress"]["v8_errors"] = [
                0.110001
            ]

        above_report = self._aggregate_with_mutation(shift_above)
        self.assertFalse(above_report["counterfactual_gate_pass"])
        self.assertTrue(
            all(
                not group["shift_ratio_pass"]
                for group in above_report["groups"].values()
            )
        )

    def test_aggregate_respects_overwrite_and_writes_hardened_group_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results_root = root / "formal"
            output_dir = root / "analysis"
            output_dir.mkdir()
            report_path = (
                output_dir / "tpd_clean_v8_mprs_counterfactual.json"
            )
            report_path.write_text('{"old": true}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                subject.aggregate(
                    results_root,
                    output_dir,
                    overwrite=False,
                )

            jobs = []
            validation_digest = sha("f")
            for variant in subject.V8_VARIANT_BY_V7:
                for seed in subject.v7_diag.SEEDS:
                    for role in subject.v7_diag.CHECKPOINT_SPECS:
                        job = {
                            "variant": variant,
                            "seed": seed,
                            "role": role,
                            "checkpoint": (
                                results_root
                                / variant
                                / f"seed-{seed}"
                                / f"{role}.pth"
                            ),
                        }
                        jobs.append(job)
                        path = subject.job_output_path(output_dir, job)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(
                            json.dumps(
                                self._aggregate_payload(
                                    job,
                                    validation_digest,
                                )
                            ),
                            encoding="utf-8",
                        )
            with (
                mock.patch.object(
                    subject,
                    "expected_jobs",
                    return_value=jobs,
                ),
                mock.patch.object(
                    subject,
                    "_validate_job_payload",
                    side_effect=lambda job, payload, output_dir: payload,
                ),
            ):
                report = subject.aggregate(
                    results_root,
                    output_dir,
                    overwrite=True,
                )
            self.assertTrue(report["counterfactual_gate_pass"])
            self.assertTrue(
                report["audit_hardening"][
                    "all_job_bindings_revalidated"
                ]
            )
            self.assertEqual(len(report["groups"]), 4)
            for group in report["groups"].values():
                self.assertEqual(
                    tuple(group["blocks"]),
                    subject.EXPECTED_BLOCK_NAMES,
                )
            written = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], subject.SCHEMA)


if __name__ == "__main__":
    unittest.main()
