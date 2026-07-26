from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import finalize_tpd_clean_screen800 as finalizer


class FinalizeTPDCleanScreen800Tests(unittest.TestCase):
    def make_candidate_tree(self, root: Path, *, duplicate: bool = False) -> None:
        (root / "logs").mkdir(parents=True)
        for index, variant in enumerate(finalizer.VARIANTS):
            run_dir = root / finalizer.DATASET / variant / finalizer.DEFAULT_RUN_NAME
            run_dir.mkdir(parents=True)
            for name in finalizer.REQUIRED_RUN_FILES:
                path = run_dir / name
                if name == "metrics.jsonl":
                    path.write_text(
                        "".join(
                            json.dumps({"epoch": epoch}) + "\n"
                            for epoch in range(1, finalizer.EXPECTED_EPOCHS + 1)
                        ),
                        encoding="utf-8",
                    )
                else:
                    path.write_text("fixture\n", encoding="utf-8")
            completion = (
                f"TPDCLEAN_COMPLETE variant={variant} "
                f"gpu_uuid=GPU-fixture-{index} epochs=800\n"
            )
            if duplicate and variant == finalizer.VARIANTS[0]:
                completion += completion
            (root / "logs" / f"{variant}.log").write_text(
                completion, encoding="utf-8"
            )

    def test_four_unique_completions_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate_root = Path(temporary)
            self.make_candidate_tree(candidate_root)
            observations = finalizer.inspect_all(
                candidate_root,
                finalizer.DEFAULT_RUN_NAME,
                include_unit_state=False,
            )
            self.assertTrue(finalizer.observations_ready(observations))
            self.assertEqual(finalizer.observation_problems(observations), [])

    def test_duplicate_completion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate_root = Path(temporary)
            self.make_candidate_tree(candidate_root, duplicate=True)
            observation = finalizer.inspect_variant(
                candidate_root,
                finalizer.DEFAULT_RUN_NAME,
                finalizer.VARIANTS[0],
                include_unit_state=False,
            )
            self.assertFalse(observation.ready)
            self.assertEqual(observation.completion_count, 2)
            self.assertTrue(
                any("expected one completion" in item for item in observation.problems)
            )

    def test_atomic_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "launch/finalizer_state.json"
            payload = {
                "schema": finalizer.STATE_SCHEMA,
                "state": "waiting",
                "message": "fixture",
            }
            finalizer.atomic_write_json(state_path, payload)
            self.assertEqual(json.loads(state_path.read_text(encoding="utf-8")), payload)
            self.assertFalse(
                any(state_path.parent.glob(f".{state_path.name}.tmp.*"))
            )

    def test_aggregator_environment_preserves_paths_and_deduplicates_repo(self) -> None:
        repo_root = str(finalizer.REPO_ROOT)
        existing = os.pathsep.join(
            ("/fixture/one", repo_root, f"{repo_root}/", "/fixture/two")
        )
        environment = finalizer._aggregator_subprocess_environment(
            {"PYTHONPATH": existing, "UNCHANGED": "yes"}
        )
        self.assertEqual(environment["UNCHANGED"], "yes")
        self.assertEqual(
            environment["PYTHONPATH"].split(os.pathsep),
            [repo_root, "/fixture/one", "/fixture/two"],
        )

    def test_run_aggregator_imports_repo_without_external_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregator = root / "aggregator.py"
            aggregator.write_text(
                "import json, os\n"
                "from experiments import finalize_tpd_clean_screen800 as imported\n"
                "print(json.dumps({\"repo_root\": str(imported.REPO_ROOT), "
                "\"pythonpath\": os.environ.get(\"PYTHONPATH\")}))\n",
                encoding="utf-8",
            )
            args = finalizer.parse_args(
                [
                    "--candidate-root",
                    str(root / "candidates"),
                    "--reference-root",
                    str(root / "references"),
                    "--reference-miou-root",
                    str(root / "candidates/reference_miou"),
                    "--aggregator",
                    str(aggregator),
                    "--output-dir",
                    str(root / "candidates/comparison"),
                    "--aggregate-timeout-seconds",
                    "30",
                ]
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                result = finalizer.run_aggregator(args)

            self.assertEqual(result["returncode"], 0)
            payload = json.loads(result["stdout"].strip())
            self.assertEqual(payload["repo_root"], str(finalizer.REPO_ROOT))
            pythonpath = payload["pythonpath"].split(os.pathsep)
            self.assertEqual(pythonpath[0], str(finalizer.REPO_ROOT))
            self.assertEqual(pythonpath.count(str(finalizer.REPO_ROOT)), 1)

    def test_dry_run_does_not_write_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_root = root / "candidates"
            reference_root = root / "references"
            candidate_root.mkdir()
            reference_root.mkdir()
            state_path = candidate_root / "launch/finalizer_state.json"
            aggregator = root / "aggregator.py"
            aggregator.write_text("# fixture\n", encoding="utf-8")
            exit_code = finalizer.main(
                [
                    "--candidate-root",
                    str(candidate_root),
                    "--reference-root",
                    str(reference_root),
                    "--aggregator",
                    str(aggregator),
                    "--state-file",
                    str(state_path),
                    "--dry-run",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse(state_path.exists())


if __name__ == "__main__":
    unittest.main()
