from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from experiments import validate_tpd_clean_v3_2x_completion as validator


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_HELPERS_PATH = (
    REPO_ROOT / "tests/test_summarize_tpd_clean_v3_screen800_2x5090.py"
)
CANONICAL_SUMMARIZER_PATH = (
    REPO_ROOT / "experiments/summarize_tpd_clean_v3_screen800_2x5090.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUMMARY_HELPERS = _load_module(
    SUMMARY_HELPERS_PATH, "_tpd_clean_v3_completion_test_helpers"
)


def _write_source_lock(
    path: Path, schema: str, source_sha256: dict[str, str], **extra: object
) -> None:
    _write_json(
        path,
        {
            "schema": schema,
            **extra,
            "source_sha256": source_sha256,
        },
    )


class CompletionFixture:
    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.experiments = self.repo / "experiments"
        self.experiments.mkdir(parents=True)
        self.summarizer_path = (
            self.experiments / "summarize_tpd_clean_v3_screen800_2x5090.py"
        )
        shutil.copyfile(CANONICAL_SUMMARIZER_PATH, self.summarizer_path)
        self.validator_path = (
            self.experiments / "validate_tpd_clean_v3_2x_completion.py"
        )
        shutil.copyfile(Path(validator.__file__), self.validator_path)
        (self.experiments / "TPD_CLEAN_V3_PROTOCOL.md").write_text(
            "# test protocol\n", encoding="utf-8"
        )
        dummy = self.repo / "model/dummy.py"
        dummy.parent.mkdir(parents=True)
        dummy.write_text("VALUE = 1\n", encoding="utf-8")
        dummy_sources = {"model/dummy.py": _sha256(dummy)}

        self.training_lock = (
            self.experiments / "tpd_clean_v3_screen800_2x_source_lock.json"
        )
        self.v2_lock = (
            self.experiments / "tpd_clean_screen800_source_lock.json"
        )
        self.ner_lock = self.experiments / "tpd_ner_v1_source_lock.json"
        _write_source_lock(
            self.training_lock,
            "sctransnet_tpd_clean_v3_screen800_2x_source_lock_v1",
            dummy_sources,
        )
        _write_source_lock(
            self.v2_lock,
            "sctransnet_tpd_clean_screen800_source_lock_v1",
            dummy_sources,
        )
        _write_source_lock(
            self.ner_lock,
            "sctransnet_tpd_ner_v1_source_lock_v1",
            dummy_sources,
        )
        self.postprocess_lock = (
            self.experiments / "tpd_clean_v3_2x_postprocess_source_lock.json"
        )
        _write_source_lock(
            self.postprocess_lock,
            "sctransnet_tpd_clean_v3_2x_postprocess_source_lock_v1",
            {
                "experiments/summarize_tpd_clean_v3_screen800_2x5090.py": _sha256(
                    self.summarizer_path
                ),
                "experiments/validate_tpd_clean_v3_2x_completion.py": _sha256(
                    self.validator_path
                ),
                "experiments/tpd_clean_v3_screen800_2x_source_lock.json": (
                    _sha256(self.training_lock)
                ),
            },
            training_source_lock_sha256=_sha256(self.training_lock),
            policy={
                "separate_from_training_source_lock": True,
                "does_not_modify_frozen_training_results": True,
                "candidate_null_budget_points_forbidden": True,
                "unused_frozen_reference_null_points_disclosed": True,
                "required_gate_reference_null_points_forbidden": True,
                "automatic_mainline_replacement": False,
            },
        )

        self.candidate = self.repo / "results/candidate"
        self.formal = self.repo / "results/formal"
        self.v2 = self.repo / "results/v2"
        self.miou = self.repo / "results/miou"
        self.staging = self.candidate / "staging"
        self.output = self.candidate / "published"

        runs, references = SUMMARY_HELPERS._passing_gate_inputs()
        for (variant, seed), record in runs.items():
            SUMMARY_HELPERS._write_candidate_run(
                self.candidate, variant, seed, record
            )
        SUMMARY_HELPERS._write_reference_sweeps(
            self.formal, self.v2, self.miou, references
        )
        self._write_launch_evidence()

        self.summary = _load_module(
            self.summarizer_path,
            f"_tpd_clean_v3_fixture_summary_{id(self)}",
        )
        exit_code = self.summary.main(
            [
                "--candidate-root",
                str(self.candidate),
                "--formal-reference-root",
                str(self.formal),
                "--v2-reference-root",
                str(self.v2),
                "--reference-miou-root",
                str(self.miou),
                "--output-dir",
                str(self.staging),
                "--overwrite",
            ]
        )
        if exit_code != 0:
            raise RuntimeError("fixture canonical summarizer failed")

    def _write_launch_evidence(self) -> None:
        launch_root = self.candidate / "launch"
        log_root = self.candidate / "logs"
        launch_root.mkdir(parents=True)
        log_root.mkdir(parents=True)
        training_lock_sha = _sha256(self.training_lock)
        jobs = [
            (variant, seed)
            for variant in validator.VARIANTS
            for seed in validator.SEEDS
        ]
        for variant, seed in jobs:
            run_dir = (
                self.candidate
                / "NUDT-SIRST"
                / variant
                / f"seed_{seed}_{validator.RUN_TAG}"
            ).resolve()
            gpu_uuid = validator.EXPECTED_GPU_ASSIGNMENTS[(variant, seed)]
            _write_json(
                launch_root / f"{variant}_seed{seed}.json",
                {
                    "schema": "sctransnet_tpd_clean_v3_screen800_2x5090_launch_v1",
                    "variant": variant,
                    "seed": seed,
                    "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
                    "gpu_uuid": gpu_uuid,
                    "gpu_name": "NVIDIA GeForce RTX 5090",
                    "run_directory": str(run_dir),
                    "training_data_sha256": "a" * 64,
                    "source_lock": str(self.training_lock),
                    "source_lock_sha256": training_lock_sha,
                    "policy": {
                        "paired_variants": True,
                        "pre_registered_seeds": [42, 3407],
                        "fresh_run": True,
                        "old_results_preserved": True,
                        "shared_resource_screening": True,
                        "efficiency_comparison_allowed": False,
                        "official_test_accessed": False,
                        "amp": False,
                    },
                },
            )
            (
                log_root / f"{variant}_seed{seed}.log"
            ).write_text(
                "worker output\n"
                f"TPDCLEANV3_2X_COMPLETE variant={variant} seed={seed} "
                f"gpu_uuid={gpu_uuid} epochs=800\n",
                encoding="utf-8",
            )

    def kwargs(self) -> dict[str, Path]:
        return {
            "candidate_root": self.candidate,
            "formal_root": self.formal,
            "v2_root": self.v2,
            "reference_miou_root": self.miou,
            "repo": self.repo,
            "summarizer_path": self.summarizer_path,
            "postprocess_lock_path": self.postprocess_lock,
        }


class TPDV3TwoGPUCompletionValidatorTests(unittest.TestCase):
    def test_fabricated_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            report_path = (
                fixture.staging / fixture.summary.JSON_OUTPUT_NAME
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["candidate_runs"]["fabricated/seed_42"] = {}
            _write_json(report_path, report)

            with self.assertRaises(validator.CompletionValidationError):
                validator.publish_completion(
                    staging_dir=fixture.staging,
                    output_dir=fixture.output,
                    **fixture.kwargs(),
                )
            self.assertFalse(
                (fixture.output / validator.MARKER_NAME).exists()
            )

    def test_incomplete_report_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            report_path = (
                fixture.staging / fixture.summary.JSON_OUTPUT_NAME
            )
            markdown_path = (
                fixture.staging / fixture.summary.MARKDOWN_OUTPUT_NAME
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["status"] = "incomplete"
            report["gate_evaluated"] = False
            report["engineering_gate_passed"] = None
            report["incomplete_reasons"] = ["test fixture is incomplete"]
            # The extra incomplete_reasons key itself is legitimate only for
            # an incomplete canonical report; publication must still reject it.
            _write_json(report_path, report)
            markdown_path.write_text(
                fixture.summary.render_markdown(report), encoding="utf-8"
            )

            with self.assertRaises(validator.CompletionValidationError):
                validator.publish_completion(
                    staging_dir=fixture.staging,
                    output_dir=fixture.output,
                    **fixture.kwargs(),
                )
            self.assertFalse(
                (fixture.output / validator.MARKER_NAME).exists()
            )

    def test_worker_log_gpu_uuid_must_match_launch_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            log_path = (
                fixture.candidate
                / "logs/tpd_clean_v3_full_seed42.log"
            )
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(
                    f"gpu_uuid={validator.GPU2_UUID}",
                    f"gpu_uuid={validator.GPU3_UUID}",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(validator.CompletionValidationError):
                validator.publish_completion(
                    staging_dir=fixture.staging,
                    output_dir=fixture.output,
                    **fixture.kwargs(),
                )
            self.assertFalse(
                (fixture.output / validator.MARKER_NAME).exists()
            )

    def test_two_gpu_scope_and_65_input_contract(self) -> None:
        self.assertEqual(
            validator.RUN_TAG, "screen800_pd_fp32_shared2x5090_v1"
        )
        self.assertEqual(
            validator.EXPECTED_GPU_UUIDS,
            frozenset((validator.GPU2_UUID, validator.GPU3_UUID)),
        )
        self.assertEqual(len(validator.EXPECTED_GPU_ASSIGNMENTS), 4)
        self.assertEqual(validator.EXPECTED_INPUT_COUNTS["total_files"], 65)
        self.assertEqual(len(validator.GATE_NAMES), 7)

    def test_registered_uuid_set_is_not_enough_without_job_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            launch_path = (
                fixture.candidate
                / "launch/tpd_clean_v3_full_seed42.json"
            )
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            self.assertEqual(launch["gpu_uuid"], validator.GPU2_UUID)
            launch["gpu_uuid"] = validator.GPU3_UUID
            _write_json(launch_path, launch)
            log_path = (
                fixture.candidate
                / "logs/tpd_clean_v3_full_seed42.log"
            )
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(
                    f"gpu_uuid={validator.GPU2_UUID}",
                    f"gpu_uuid={validator.GPU3_UUID}",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(validator.CompletionValidationError):
                validator.publish_completion(
                    staging_dir=fixture.staging,
                    output_dir=fixture.output,
                    **fixture.kwargs(),
                )
            self.assertFalse(
                (fixture.output / validator.MARKER_NAME).exists()
            )

    def test_marker_write_failure_never_exposes_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            original_atomic_write = validator._atomic_write_bytes

            def fail_marker(path: Path, content: bytes) -> None:
                if path.name == validator.MARKER_NAME:
                    raise OSError("injected marker write failure")
                original_atomic_write(path, content)

            with mock.patch.object(
                validator,
                "_atomic_write_bytes",
                side_effect=fail_marker,
            ):
                with self.assertRaises(OSError):
                    validator.publish_completion(
                        staging_dir=fixture.staging,
                        output_dir=fixture.output,
                        **fixture.kwargs(),
                    )
            self.assertFalse(
                (fixture.output / validator.MARKER_NAME).exists()
            )

    def test_marker_last_reuse_and_checkpoint_drift_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            writes: list[str] = []
            original_atomic_write = validator._atomic_write_bytes

            def recording_write(path: Path, content: bytes) -> None:
                writes.append(path.name)
                original_atomic_write(path, content)

            with mock.patch.object(
                validator,
                "_atomic_write_bytes",
                side_effect=recording_write,
            ):
                result = validator.publish_completion(
                    staging_dir=fixture.staging,
                    output_dir=fixture.output,
                    **fixture.kwargs(),
                )
            self.assertFalse(result["reused"])
            self.assertEqual(
                writes,
                [
                    fixture.summary.JSON_OUTPUT_NAME,
                    fixture.summary.MARKDOWN_OUTPUT_NAME,
                    validator.MANIFEST_NAME,
                    validator.MARKER_NAME,
                ],
            )
            marker_lines = (
                fixture.output / validator.MARKER_NAME
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(marker_lines), 3)
            self.assertTrue(
                (fixture.output / validator.MANIFEST_NAME).is_file()
            )

            reused = validator.publish_completion(
                staging_dir=fixture.staging,
                output_dir=fixture.output,
                **fixture.kwargs(),
            )
            self.assertTrue(reused["reused"])

            checkpoint = (
                fixture.candidate
                / "NUDT-SIRST/tpd_clean_v3_full"
                / f"seed_42_{validator.RUN_TAG}/best.pth.tar"
            )
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaises(validator.CompletionValidationError):
                validator.verify_completion(
                    output_dir=fixture.output,
                    **fixture.kwargs(),
                )


if __name__ == "__main__":
    unittest.main()
