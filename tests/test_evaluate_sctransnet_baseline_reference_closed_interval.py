from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    evaluate_sctransnet_baseline_reference_closed_interval as entry,
)
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    LAST_FLOAT32_BELOW_ONE,
)


class BaselineClosedIntervalReferenceTests(unittest.TestCase):
    @staticmethod
    def _base_output(root: Path) -> dict:
        root = root.resolve()
        split_hash = "a" * 64
        artifacts = {
            "protocol.json": b"protocol",
            "split.json": (
                '{"hashes":{"used_val_sha256":"' + split_hash + '"}}'
            ).encode(),
            "summary.json": b"summary",
            "metrics.jsonl": b"metrics",
            "best.pth.tar": b"checkpoint",
        }
        for name, content in artifacts.items():
            (root / name).write_bytes(content)
        checkpoint_sha = entry._sha256_file(root / "best.pth.tar")
        endpoint = {
            "threshold": 1.0,
            "pd": 0.0,
            "fa": 0.0,
            "miou": 0.0,
            "false_objects_per_image": 0.0,
            "target_count": 189,
            "matched_target_count": 0,
            "predicted_object_count": 0,
            "unmatched_predicted_object_count": 0,
        }
        ordinary = {
            **endpoint,
            "threshold": 0.5,
            "pd": 187 / 189,
            "fa": 0.0,
            "miou": 0.94,
            "target_count": 189,
            "matched_target_count": 187,
            "predicted_object_count": 187,
        }
        budget_points = {
            f"{budget:.10g}": dict(ordinary)
            for budget in entry.FA_BUDGETS
        }
        return {
            "run_directory": root,
            "checkpoint": root / "best.pth.tar",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_role": "best_validation_pd_primary",
            "dataset": "NUDT-SIRST",
            "variant": "original",
            "seed": 42,
            "split_seed": 20260722,
            "validation_count": 133,
            "validation_split_sha256": split_hash,
            "official_test_accessed": False,
            "threshold_configuration": {
                "threshold_min": 0.01,
                "threshold_max": 0.99,
                "threshold_step": 0.01,
                "extra_thresholds": list(entry.EXTRA_THRESHOLDS),
                "tail_logit_step": 0.1,
                "fa_budgets": list(entry.FA_BUDGETS),
            },
            "threshold_provenance": {
                "closed_probability_interval": True,
                "score_dtype": "float32",
                "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
                "upper_boundary_threshold": 1.0,
                "upper_boundary_comparison": "prediction > threshold",
                "upper_boundary_semantics": "empty_prediction_pd0_fa0",
                "added_thresholds": [LAST_FLOAT32_BELOW_ONE, 1.0],
            },
            "points": [
                {
                    **endpoint,
                    "threshold": LAST_FLOAT32_BELOW_ONE,
                },
                endpoint,
            ],
            "fixed_threshold_0_5": ordinary,
            "best_points_under_fa_budget": budget_points,
            "audit": {
                "invocation_argv": [
                    sys.executable,
                    str(Path(entry.__file__).resolve()),
                ],
                "parsed_arguments": {
                    "run_dir": root,
                    "checkpoint": "best.pth.tar",
                    "expected_epochs": 800,
                    "threshold_min": 0.01,
                    "threshold_max": 0.99,
                    "threshold_step": 0.01,
                    "extra_thresholds": list(entry.EXTRA_THRESHOLDS),
                    "tail_logit_step": 0.1,
                    "fa_budgets": list(entry.FA_BUDGETS),
                    "match_radius": None,
                    "tiny_area": None,
                    "overwrite": False,
                },
                "expected_epochs": 800,
                "metrics_event_count": 800,
                "metrics_epoch_range": [1, 800],
                "summary_status": "complete",
                "integrity_checks_passed": {"all": True},
                "artifact_sha256": {
                    "protocol.json": entry._sha256_file(
                        root / "protocol.json"
                    ),
                    "split.json": entry._sha256_file(root / "split.json"),
                    "summary.json": entry._sha256_file(
                        root / "summary.json"
                    ),
                    "metrics.jsonl": entry._sha256_file(
                        root / "metrics.jsonl"
                    ),
                    "checkpoint": checkpoint_sha,
                    "evaluator": entry._sha256_file(
                        Path(entry.__file__).resolve()
                    ),
                },
            },
        }

    def test_contract_reuses_shared_metric_and_closed_interval(self) -> None:
        contract = entry.evaluator_contract()
        self.assertEqual(contract["dataset"], "NUDT-SIRST")
        self.assertEqual(contract["variant"], "original")
        self.assertEqual(contract["training_seed"], 42)
        self.assertEqual(contract["split_seed"], 20260722)
        self.assertEqual(contract["expected_epochs"], 800)
        self.assertEqual(
            contract["metric_core"],
            "experiments.evaluate_pd_fa_sweep",
        )
        self.assertIn("adaptive_thresholds_closed_interval", contract["closed_interval_core"])
        self.assertFalse(contract["historical_sweep_overwrite_allowed"])
        self.assertFalse(
            contract[
                "endpoint_protocol_preregistered_before_historical_training"
            ]
        )

    def test_formal_arguments_are_fixed_and_overwrite_is_rejected(self) -> None:
        args = entry.validate_formal_arguments(
            ["--run-dir", "/tmp/reference", "--device", "cpu"]
        )
        self.assertEqual(args.checkpoint, "best.pth.tar")
        self.assertEqual(tuple(args.fa_budgets), entry.FA_BUDGETS)
        with self.assertRaisesRegex(ValueError, "forbids --overwrite"):
            entry.validate_formal_arguments(
                ["--run-dir", "/tmp/reference", "--overwrite"]
            )
        with self.assertRaisesRegex(ValueError, "only best"):
            entry.validate_formal_arguments(
                ["--run-dir", "/tmp/reference", "--checkpoint", "last.pth.tar"]
            )
        with self.assertRaisesRegex(ValueError, "fa_budgets"):
            entry.validate_formal_arguments(
                [
                    "--run-dir",
                    "/tmp/reference",
                    "--fa-budgets",
                    "0.000001",
                ]
            )

    def test_output_identity_requires_closed_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = entry.finalize_reference_output(
                self._base_output(root)
            )
            entry.validate_output_identity(
                payload,
                expected_run_dir=root,
                expected_checkpoint="best.pth.tar",
            )
            payload["threshold_provenance"][
                "closed_probability_interval"
            ] = False
            with self.assertRaisesRegex(
                ValueError,
                "closed_probability_interval",
            ):
                entry.validate_output_identity(payload)

    def test_output_rejects_stale_identity_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = entry.finalize_reference_output(
                self._base_output(root)
            )
            cases = (
                (
                    "checkpoint sha",
                    lambda value: value.__setitem__(
                        "checkpoint_sha256",
                        "0" * 64,
                    ),
                    "checkpoint_sha256",
                ),
                (
                    "absolute run",
                    lambda value: value.__setitem__(
                        "run_directory",
                        "/tmp/wrong-baseline-run",
                    ),
                    "inside the run directory",
                ),
                (
                    "checkpoint filename",
                    lambda value: value.__setitem__(
                        "checkpoint",
                        str(root / "last.pth.tar"),
                    ),
                    "filename",
                ),
                (
                    "checkpoint role",
                    lambda value: value.__setitem__(
                        "checkpoint_role",
                        "wrong",
                    ),
                    "role",
                ),
                (
                    "seed",
                    lambda value: value.__setitem__("seed", 43),
                    "seed",
                ),
                (
                    "split seed",
                    lambda value: value.__setitem__("split_seed", 1),
                    "split_seed",
                ),
                (
                    "validation split",
                    lambda value: value.__setitem__(
                        "validation_split_sha256",
                        "0" * 64,
                    ),
                    "validation split",
                ),
                (
                    "evaluator sha",
                    lambda value: value["audit"]["artifact_sha256"].__setitem__(
                        "evaluator",
                        "0" * 64,
                    ),
                    "artifact sha256",
                ),
                (
                    "evaluator invocation",
                    lambda value: value["audit"]["invocation_argv"].__setitem__(
                        1,
                        "/tmp/wrong-evaluator.py",
                    ),
                    "invocation",
                ),
                (
                    "parsed checkpoint",
                    lambda value: value["audit"]["parsed_arguments"].__setitem__(
                        "checkpoint",
                        "best_miou.pth.tar",
                    ),
                    "parsed checkpoint",
                ),
                (
                    "preregistration marker",
                    lambda value: value["reference_provenance"].__setitem__(
                        "endpoint_protocol_preregistered_before_historical_training",
                        True,
                    ),
                    "preregistration",
                ),
                (
                    "raw endpoint provenance",
                    lambda value: value["threshold_provenance"].__setitem__(
                        "preregistered_endpoint_completion",
                        True,
                    ),
                    "preregistered_endpoint_completion",
                ),
                (
                    "budget coverage",
                    lambda value: value["final_metric_coverage"][
                        "pd_at_fa_budget"
                    ]["1e-06"].__setitem__("pd", 0.0),
                    "budget metric coverage",
                ),
            )
            for label, mutate, message in cases:
                with self.subTest(label=label):
                    value = json.loads(json.dumps(entry.base.json_ready(original)))
                    mutate(value)
                    with self.assertRaisesRegex(ValueError, message):
                        entry.validate_output_identity(value)

    def test_every_audit_artifact_hash_is_bound_to_current_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = entry.finalize_reference_output(
                self._base_output(Path(directory))
            )
            for artifact_name in (
                "protocol.json",
                "split.json",
                "summary.json",
                "metrics.jsonl",
                "checkpoint",
                "evaluator",
            ):
                with self.subTest(artifact=artifact_name):
                    value = json.loads(
                        json.dumps(entry.base.json_ready(original))
                    )
                    value["audit"]["artifact_sha256"][artifact_name] = "0" * 64
                    with self.assertRaisesRegex(
                        ValueError,
                        "artifact sha256",
                    ):
                        entry.validate_output_identity(value)

    def test_checkpoint_file_change_invalidates_completed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = entry.finalize_reference_output(
                self._base_output(root)
            )
            (root / "best.pth.tar").write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "current file"):
                entry.validate_output_identity(payload)

    def test_help_is_side_effect_free(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(Path(entry.__file__)), "--help"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--run-dir", completed.stdout)
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
