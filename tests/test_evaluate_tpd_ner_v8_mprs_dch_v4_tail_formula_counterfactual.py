from __future__ import annotations

import ast
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments import (
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual as subject,
)


_CANONICAL: dict | None = None


def canonical_fixture() -> dict:
    global _CANONICAL
    if _CANONICAL is None:
        _CANONICAL = json.loads(
            subject.canonical_sweep_path("best.pth.tar").read_text(
                encoding="utf-8"
            )
        )
    return copy.deepcopy(_CANONICAL)


class TailFormulaCounterfactualContractTests(unittest.TestCase):
    def test_fixed_modes_lanes_and_metric_contract(self) -> None:
        self.assertEqual(
            subject.FORMULA_MODES,
            ("legacy_global", "direct_tail", "complement_tail"),
        )
        self.assertEqual(
            subject.FORMULA_EXPRESSIONS,
            {
                "legacy_global": "d",
                "direct_tail": "d*P",
                "complement_tail": "d*(1-P)",
            },
        )
        self.assertEqual(
            subject.CHECKPOINT_GPU_LANES["best.pth.tar"][
                "physical_gpu_index"
            ],
            2,
        )
        self.assertEqual(
            subject.CHECKPOINT_GPU_LANES["best_miou.pth.tar"][
                "physical_gpu_index"
            ],
            3,
        )
        self.assertEqual(
            subject.METRIC_CONTRACT["final_metric_coverage_schema"],
            subject.formal_v3_evaluator.FINAL_METRIC_COVERAGE_SCHEMA,
        )

    def test_cli_requires_exactly_one_action(self) -> None:
        plan = subject.parse_args(
            ["--plan", "--checkpoint", "best.pth.tar"]
        )
        self.assertTrue(plan.plan)
        self.assertFalse(plan.run)
        run = subject.parse_args(
            ["--run", "--checkpoint", "best_miou.pth.tar"]
        )
        self.assertTrue(run.run)
        self.assertEqual(run.device, "cuda:0")
        for argv in (
            ["--checkpoint", "best.pth.tar"],
            ["--plan", "--run", "--checkpoint", "best.pth.tar"],
            ["--plan", "--checkpoint", "unknown.pth.tar"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    subject.parse_args(argv)

    def test_legacy_projection_is_exact_and_any_change_is_rejected(self) -> None:
        canonical = canonical_fixture()
        result = subject.require_legacy_canonical_exact(
            copy.deepcopy(canonical),
            canonical,
        )
        self.assertTrue(result["legacy_global_canonical_exact"])
        self.assertEqual(
            result["observed_projection_sha256"],
            result["canonical_projection_sha256"],
        )

        changed = copy.deepcopy(canonical)
        changed["fixed_threshold_0_5"]["pd"] -= 0.01
        with self.assertRaisesRegex(ValueError, "not canonically exact"):
            subject.require_legacy_canonical_exact(changed, canonical)

    def test_projection_rejects_missing_field_and_wrong_budget_keys(self) -> None:
        missing = canonical_fixture()
        del missing["points"]
        with self.assertRaisesRegex(ValueError, "lacks canonical fields"):
            subject.canonical_sweep_projection(missing)

        wrong_budgets = canonical_fixture()
        del wrong_budgets["best_points_under_fa_budget"][
            next(iter(subject.BUDGET_KEYS))
        ]
        with self.assertRaisesRegex(ValueError, "FA budget keys"):
            subject.canonical_sweep_projection(wrong_budgets)

    def test_environment_and_actual_gpu_identity_are_both_required(self) -> None:
        checkpoint = "best.pth.tar"
        required = subject._command_environment(checkpoint)
        lane_spec = subject.CHECKPOINT_GPU_LANES[checkpoint]
        properties = SimpleNamespace(
            name=subject.EXPECTED_GPU_NAME,
            major=12,
            minor=0,
            total_memory=32 * 1024**3,
            uuid=lane_spec["physical_gpu_uuid"],
        )
        with (
            mock.patch.dict(os.environ, required, clear=True),
            mock.patch.object(subject.torch.cuda, "is_available", return_value=True),
            mock.patch.object(subject.torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                subject.gpu_identity_core,
                "visible_gpu_identity",
                return_value=(lane_spec["physical_gpu_uuid"], properties),
            ),
        ):
            lane = subject._validated_cuda_lane(checkpoint, "cuda:0")
        self.assertEqual(
            lane["actual_logical_cuda_0_uuid"],
            lane_spec["physical_gpu_uuid"],
        )
        self.assertEqual(lane["device_name"], subject.EXPECTED_GPU_NAME)

        invalid_environment = dict(required)
        invalid_environment["PYTHONHASHSEED"] = "7"
        with self.assertRaisesRegex(ValueError, "PYTHONHASHSEED"):
            subject._validate_lane_environment(
                checkpoint,
                "cuda:0",
                invalid_environment,
            )

        with (
            mock.patch.dict(os.environ, required, clear=True),
            mock.patch.object(subject.torch.cuda, "is_available", return_value=True),
            mock.patch.object(subject.torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                subject.gpu_identity_core,
                "visible_gpu_identity",
                return_value=("GPU-not-the-declared-device", properties),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "actual GPU UUID"):
                subject._validated_cuda_lane(checkpoint, "cuda:0")

    def test_implementation_hashes_bind_builder_determinism_and_gpu_identity(
        self,
    ) -> None:
        records = subject.implementation_hashes()
        for name in (
            "v8_parent_builder",
            "determinism_core",
            "gpu_identity_core",
        ):
            with self.subTest(name=name):
                self.assertIn(name, records)
                self.assertRegex(records[name]["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(Path(records[name]["path"]).is_file())

    def test_output_is_versioned_and_atomic_publication_never_overwrites(
        self,
    ) -> None:
        for checkpoint in subject.CHECKPOINTS:
            target = subject.output_path(checkpoint)
            self.assertTrue(
                target.is_relative_to(subject.DEFAULT_OUTPUT_ROOT)
            )
            subject._assert_output_isolated(target)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "result.json"
            payload = {"status": "complete"}
            subject.frozen_diagnostic_core.atomic_publish_new(
                destination,
                payload,
            )
            original = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.frozen_diagnostic_core.atomic_publish_new(
                    destination,
                    {"status": "replacement"},
                )
            self.assertEqual(destination.read_bytes(), original)

    def test_no_optimization_sensitive_assert_statements(self) -> None:
        source = Path(subject.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertEqual(
            [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)],
            [],
        )


if __name__ == "__main__":
    unittest.main()
