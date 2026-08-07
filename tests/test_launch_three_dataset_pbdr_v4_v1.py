from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import launch_three_dataset_pbdr_v4_v1 as launcher


class ThreeDatasetPBDRV4LauncherTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> tuple[Path, Path, Path]:
        source = root / "source_lock.json"
        split = root / "split_projection.json"
        source.write_text("{}\n", encoding="utf-8")
        split.write_text("{}\n", encoding="utf-8")
        pool_root = root / "pools"
        for dataset in launcher.DATASETS:
            for role in launcher.ROLES:
                path = launcher.candidate_pool_path(pool_root, dataset, role)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
        return source, split, pool_root

    def test_three_dataset_workers_use_fixed_gpu_mapping_and_joint_official(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, split, pool_root = self._artifacts(root)
            calls: list[tuple[str, str]] = []
            events: list[str] = []

            def preflight(**kwargs: object) -> dict[str, object]:
                events.append("preflight")
                calls.append((str(kwargs["dataset"]), str(kwargs["role"])))
                return {
                    "dataset": kwargs["dataset"],
                    "role": kwargs["role"],
                    "official_test_accessed": False,
                }

            with mock.patch.object(
                launcher.training_core,
                "configure_determinism",
                side_effect=lambda seed: events.append(f"determinism:{seed}") or {},
            ), mock.patch.object(
                launcher.evaluator,
                "preflight_artifacts_only",
                side_effect=preflight,
            ):
                specs = launcher.build_worker_specs(
                    results_root=root / "results",
                    data_root=root / "data",
                    source_lock_path=source,
                    split_projection_path=split,
                    candidate_pool_root=pool_root,
                    python=Path("/venv/python"),
                    evaluator_path=Path("/repo/evaluator.py"),
                    launcher_path=Path("/repo/launcher.py"),
                    base_environment={"PATH": "/bin"},
                    preflight_frozen_pools=True,
                )

            self.assertEqual(
                [(item.dataset, item.gpu_index) for item in specs],
                [
                    ("NUDT-SIRST", "0"),
                    ("IRSTD-1K", "1"),
                    ("NUAA-SIRST", "3"),
                ],
            )
            self.assertEqual(events[0], "determinism:42")
            self.assertEqual(len(calls), 6)
            self.assertEqual(
                set(calls),
                {
                    (dataset, role)
                    for dataset in launcher.DATASETS
                    for role in launcher.ROLES
                },
            )
            for spec in specs:
                self.assertEqual(spec.run_directory, (root / "results" / "official" / spec.dataset).resolve())
                self.assertEqual(tuple(spec.candidate_pool_paths), launcher.ROLES)
                self.assertEqual(
                    tuple(phase.phase for phase in spec.phases),
                    launcher.PHASE_ORDER,
                )
                smoke = spec.phases[launcher.PHASE_ORDER.index("smoke")]
                self.assertEqual(len(smoke.commands), 4)
                self.assertEqual(
                    [command[command.index("--stage") + 1] for command in smoke.commands],
                    ["stage1", "stage1", "stage2", "stage2"],
                )
                for command in smoke.commands[2:]:
                    self.assertIn("--stage1-checkpoint", command)
                    self.assertIn("--smoke", command)
                freeze = spec.phases[launcher.PHASE_ORDER.index("freeze_pools")]
                self.assertEqual(len(freeze.commands), 2)
                self.assertTrue(
                    all("freeze-pool" in command for command in freeze.commands)
                )
                joint = spec.phases[-1]
                self.assertEqual(len(joint.commands), 1)
                command = joint.commands[0]
                self.assertEqual(command[0], "/venv/python")
                self.assertIn("--best-miou-candidate-pool", command)
                self.assertIn("--best-pd-candidate-pool", command)
                self.assertIn("--expected-gpu-uuid", command)
                self.assertIn(spec.expected_gpu_uuid, command)
                self.assertNotIn("--role", command)
                self.assertNotEqual(spec.gpu_index, "2")

    def test_full_phase_plan_and_status_reject_skipping(self) -> None:
        spec = launcher.synthetic_worker_spec_for_tests(
            dataset="NUDT-SIRST",
            gpu_index="0",
        )
        self.assertEqual(
            tuple(phase.phase for phase in spec.phases),
            (
                "prepare",
                "sweep",
                "smoke",
                "stage1",
                "stage2",
                "freeze_pools",
                "joint_official",
            ),
        )
        self.assertTrue(
            all(command and command[0] for phase in spec.phases for command in phase.commands)
        )
        manifest = launcher.phase_status_manifest(spec)
        self.assertEqual(
            [item["status"] for item in manifest["phases"]],
            ["pending"] * len(launcher.PHASE_ORDER),
        )
        valid = {
            phase: ("complete" if index < 2 else "pending")
            for index, phase in enumerate(launcher.PHASE_ORDER)
        }
        launcher.validate_phase_statuses(valid)
        self.assertEqual(launcher.next_pending_phase(spec, valid).phase, "smoke")

        skipped = dict(valid)
        skipped["prepare"] = "pending"
        with self.assertRaisesRegex(ValueError, "skip"):
            launcher.validate_phase_statuses(skipped)

    def test_gpu2_is_not_configured_or_accepted(self) -> None:
        self.assertEqual(launcher.ALLOWED_GPU_INDICES, ("0", "1", "3"))
        self.assertEqual(
            launcher.DATASET_GPU_LAYOUT,
            (("NUDT-SIRST", "0"), ("IRSTD-1K", "1"), ("NUAA-SIRST", "3")),
        )
        self.assertEqual(
            {
                dataset: launcher.GPU_ASSIGNMENTS[gpu]["uuid"]
                for dataset, gpu in launcher.DATASET_GPU_LAYOUT
            },
            launcher.evaluator.FORMAL_GPU_UUIDS,
        )
        with self.assertRaisesRegex(ValueError, "GPU"):
            launcher.gpu_environment("2", {})

    def test_dry_run_contains_full_status_manifest_without_official_access(self) -> None:
        spec = launcher.synthetic_worker_spec_for_tests(
            dataset="IRSTD-1K",
            gpu_index="1",
        )
        payload = launcher.dry_run_payload((spec,))
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["official_test_accessed"])
        self.assertEqual(payload["phase_order"], list(launcher.PHASE_ORDER))
        self.assertEqual(payload["worker_count"], 1)
        self.assertEqual(
            [item["phase"] for item in payload["workers"][0]["phases"]],
            list(launcher.PHASE_ORDER),
        )
        arguments = launcher.parse_args(
            [
                "--source-lock",
                "/tmp/source.json",
                "--split-projection",
                "/tmp/split.json",
            ]
        )
        self.assertFalse(arguments.execute)

    def test_launcher_never_constructs_dataset_index_or_loader(self) -> None:
        source = inspect.getsource(launcher)
        self.assertNotIn("load_" + "index(", source)
        self.assertNotIn("Data" + "Loader(", source)
        self.assertNotIn("resolve_" + "sample(", source)
        self.assertNotIn("GPU-4a0f4ab5", source)


if __name__ == "__main__":
    unittest.main()
