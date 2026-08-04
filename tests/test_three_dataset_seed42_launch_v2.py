from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from experiments import three_dataset_seed42_launch_v2 as launch


def _option(command: tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


class MatrixTests(unittest.TestCase):
    def test_exact_12_run_matrix_and_dataset_local_wave_order(self) -> None:
        specs = launch.build_all_worker_specs(base_environment={})
        self.assertEqual(len(specs), 12)
        self.assertEqual(sum(spec.method == "original" for spec in specs), 3)
        self.assertEqual(sum(spec.method == "final" for spec in specs), 9)
        self.assertEqual(
            {spec.requested_tss_weight for spec in specs if spec.method == "final"},
            set(launch.TSS_LAMBDAS),
        )
        for dataset_index, dataset in enumerate(launch.DATASETS):
            members = [spec for spec in specs if spec.dataset == dataset]
            self.assertEqual(len(members), 4)
            self.assertEqual(
                {spec.global_wave for spec in members},
                {dataset_index * 2, dataset_index * 2 + 1},
            )
        for wave in range(6):
            members = [spec for spec in specs if spec.global_wave == wave]
            self.assertEqual({spec.gpu_index for spec in members}, {"2", "3"})

    def test_commands_lock_seed_schedule_threshold_lambda_and_gpu(self) -> None:
        for spec in launch.build_all_worker_specs(base_environment={}):
            self.assertEqual(_option(spec.command, "--seed"), "42")
            self.assertEqual(_option(spec.command, "--epochs"), "1000")
            self.assertEqual(_option(spec.command, "--eval-every"), "10")
            self.assertEqual(_option(spec.command, "--threshold"), "0.5")
            self.assertEqual(
                _option(spec.command, "--physical-gpu-index"), spec.gpu_index
            )
            self.assertEqual(
                spec.environment["CUDA_VISIBLE_DEVICES"],
                launch.GPU_ASSIGNMENTS[spec.gpu_index]["uuid"],
            )
            self.assertNotIn("--max-train-images", spec.command)
            self.assertNotIn("--max-test-images", spec.command)
            if spec.method == "final":
                self.assertEqual(
                    float(_option(spec.command, "--tss-weight")),
                    spec.requested_tss_weight,
                )
            else:
                self.assertNotIn("--tss-weight", spec.command)

    def test_run_directories_isolate_all_final_lambdas(self) -> None:
        specs = launch.build_all_worker_specs(base_environment={})
        self.assertEqual(len({spec.run_directory for spec in specs}), 12)


class ArtifactTests(unittest.TestCase):
    def test_static_lock_includes_launcher_evaluator_selector_and_metric_protocol(
        self,
    ) -> None:
        static = launch.validate_static_inputs()
        sources = static["training_sources"]
        expected = {
            "launcher": Path(launch.__file__).resolve(),
            "posttraining_evaluator": (
                launch.REPO_ROOT
                / "experiments"
                / "evaluate_three_dataset_v2.py"
            ),
            "global_recipe_selector": (
                launch.REPO_ROOT
                / "experiments"
                / "select_three_dataset_global_tss_recipe_v2.py"
            ),
            "evaluation_metric_protocol": (
                launch.REPO_ROOT
                / "experiments"
                / "four_dataset_evaluation_protocol_v1.py"
            ),
        }
        for key, path in expected.items():
            self.assertIn(key, sources)
            self.assertEqual(sources[key]["path"], str(path))
            self.assertEqual(sources[key]["sha256"], launch.file_sha256(path))

    def test_completed_run_uses_exact_training_runtime_source_subset(self) -> None:
        planned = launch.validate_static_inputs()["training_sources"]
        runtime = {
            name: dict(record)
            for name, record in launch._planned_training_runtime_sources(
                planned
            ).items()
        }
        self.assertTrue(launch.PLAN_ONLY_SOURCE_NAMES.issubset(planned))
        self.assertTrue(launch.PLAN_ONLY_SOURCE_NAMES.isdisjoint(runtime))
        launch._validate_run_runtime_sources(
            runtime,
            planned,
            spec_key="mock-complete-run",
        )

        missing = dict(runtime)
        missing.pop("runner")
        with self.assertRaises(launch.LaunchProtocolError):
            launch._validate_run_runtime_sources(
                missing,
                planned,
                spec_key="mock-complete-run",
            )

        tampered = {name: dict(record) for name, record in runtime.items()}
        tampered["runner"]["sha256"] = "f" * 64
        with self.assertRaises(launch.LaunchProtocolError):
            launch._validate_run_runtime_sources(
                tampered,
                planned,
                spec_key="mock-complete-run",
            )

    def test_compact_tss_artifact_has_exact_three_records(self) -> None:
        payload = launch.build_tss_statistics_payload()
        self.assertEqual(payload["datasets"], list(launch.DATASETS))
        self.assertEqual(set(payload["records"]), set(launch.DATASETS))
        self.assertNotIn("SIRST3", json.dumps(payload))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tss.json"
            record = launch.prepare_tss_statistics(path)
            self.assertEqual(record["schema"], launch.TSS_STATISTICS_SCHEMA)
            self.assertEqual(json.loads(path.read_text()), payload)

    def test_full_pair_preflight_and_tiny_counts(self) -> None:
        payload = launch.audit_all_indexed_pairs(output_path=None)
        self.assertEqual(payload["pair_count"], 2755)
        self.assertEqual(payload["error_count"], 0)
        expected_tiny = {
            "NUAA-SIRST": 35,
            "NUDT-SIRST": 259,
            "IRSTD-1K": 30,
        }
        for dataset, expected in expected_tiny.items():
            record = payload["datasets"][dataset]["test"]
            self.assertEqual(
                record["tiny_gt_component_count_area_le_9"], expected
            )
            self.assertTrue(record["tiny_metric_defined"])
        self.assertEqual(
            payload["tiny_gt_component_count_area_le_9_test_total"], 324
        )

    def test_default_main_does_not_execute_workers(self) -> None:
        prepared = {
            "worker_count": 12,
            "wave_count": 6,
            "training_started": False,
        }
        with mock.patch.object(
            launch, "prepare_launch_plan", return_value=prepared
        ), mock.patch.object(launch, "execute_prepared_plan") as execute:
            launch.main([])
        execute.assert_not_called()

    def test_malformed_complete_summary_is_not_silently_skipped(self) -> None:
        original = launch.build_all_worker_specs(base_environment={})[0]
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "run"
            run_directory.mkdir(parents=True)
            spec = replace(original, run_directory=run_directory)
            (run_directory / "summary.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            with self.assertRaises(launch.LaunchProtocolError):
                launch._spec_is_complete(
                    spec,
                    static_inputs={},
                    tss_sha256="sha",
                )


if __name__ == "__main__":
    unittest.main()
