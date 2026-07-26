from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from experiments import validate_tpd_clean_v4_2x_completion as validator


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_SOURCE = Path(validator.__file__).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_SUMMARIZER = '''\
from __future__ import annotations

import datetime as dt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sctransnet_tpd_clean_v4_screen800_comparison_v1"
JSON_OUTPUT_NAME = "tpd_clean_v4_screen800_comparison.json"
MARKDOWN_OUTPUT_NAME = "tpd_clean_v4_screen800_comparison.md"
VARIANTS = ("tpd_clean_v4_full", "tpd_clean_v4_sal_capacity")
SEEDS = (42, 3407)
RUN_TAG = "screen800_pd_fp32_shared2x5090_v1"


def build_report(candidate_root, formal_root, smoke_root, training_lock):
    runs = {
        f"{variant}/seed_{seed}": {
            "variant": variant,
            "seed": seed,
        }
        for variant in VARIANTS
        for seed in SEEDS
    }
    return {
        "schema": SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
        "scope": {
            "dataset": "NUDT-SIRST",
            "candidate_variants": list(VARIANTS),
            "model_seeds": list(SEEDS),
            "official_test_accessed": False,
            "candidate_root": str(Path(candidate_root).resolve()),
            "formal_reference_root": str(Path(formal_root).resolve()),
            "smoke_root": str(Path(smoke_root).resolve()),
            "training_source_lock": str(Path(training_lock).resolve()),
        },
        "candidate_runs": runs,
        "gate_evaluated": True,
        "engineering_gate_passed": True,
        "ner_stage_authorized": True,
        "mainline_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "decision": "ENGINEERING_GATE_PASS",
        "decision_boundary": {
            "gate_only_controls_next_engineering_stage": True,
            "automatic_mainline_replacement": False,
            "mainline_changed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }


def render_markdown(report):
    return (
        "# TPD-Clean-v4 fixture\\n\\n"
        f"Decision: {report['decision']}\\n"
    )
'''


class CompletionFixture:
    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.experiments = self.repo / "experiments"
        self.tests = self.repo / "tests"
        self.experiments.mkdir(parents=True)
        self.tests.mkdir(parents=True)

        self.summarizer_path = (
            self.experiments / "summarize_tpd_clean_v4_screen800.py"
        )
        self.summarizer_path.write_text(FAKE_SUMMARIZER, encoding="utf-8")
        self.validator_path = (
            self.experiments / "validate_tpd_clean_v4_2x_completion.py"
        )
        shutil.copyfile(VALIDATOR_SOURCE, self.validator_path)

        self.dummy = self.repo / "model/dummy.py"
        self.dummy.parent.mkdir(parents=True)
        self.dummy.write_text("VALUE = 1\n", encoding="utf-8")
        dummy_sources = {"model/dummy.py": _sha256(self.dummy)}

        self.v3_lock = (
            self.experiments / "tpd_clean_v3_screen800_source_lock.json"
        )
        _write_json(
            self.v3_lock,
            {
                "schema": "sctransnet_tpd_clean_v3_screen800_source_lock_v1",
                "source_sha256": dummy_sources,
            },
        )
        self.training_lock = (
            self.experiments / "tpd_clean_v4_screen800_2x_source_lock.json"
        )
        _write_json(
            self.training_lock,
            {
                "schema": (
                    "sctransnet_tpd_clean_v4_screen800_2x_source_lock_v1"
                ),
                "frozen_v3_source_lock_sha256": _sha256(self.v3_lock),
                "source_sha256": dummy_sources,
            },
        )
        self.v2_lock = (
            self.experiments / "tpd_clean_screen800_source_lock.json"
        )
        self.ner_lock = self.experiments / "tpd_ner_v1_source_lock.json"
        _write_json(
            self.v2_lock,
            {
                "schema": "sctransnet_tpd_clean_screen800_source_lock_v1",
                "source_sha256": dummy_sources,
            },
        )
        _write_json(
            self.ner_lock,
            {
                "schema": "sctransnet_tpd_ner_v1_source_lock_v1",
                "source_sha256": dummy_sources,
            },
        )

        for relative in validator.POSTPROCESS_SOURCE_SET:
            path = self.repo / relative
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture source: {relative}\n", encoding="utf-8")
        self.postprocess_lock = (
            self.experiments
            / "tpd_clean_v4_2x_postprocess_source_lock.json"
        )
        postprocess_sources = {
            relative: _sha256(self.repo / relative)
            for relative in validator.POSTPROCESS_SOURCE_SET
        }
        _write_json(
            self.postprocess_lock,
            {
                "schema": (
                    "sctransnet_tpd_clean_v4_2x_postprocess_source_lock_v1"
                ),
                "training_source_lock_sha256": _sha256(self.training_lock),
                "policy": dict(validator.REQUIRED_POSTPROCESS_POLICY),
                "source_sha256": postprocess_sources,
            },
        )

        self.candidate = self.repo / "results/candidate"
        self.formal = self.repo / "results/formal"
        self.reference_miou = self.repo / "results/reference_miou"
        self.smoke = self.repo / "results/smoke"
        for path in (
            self.candidate,
            self.formal,
            self.reference_miou,
            self.smoke,
        ):
            path.mkdir(parents=True)
        self.staging = self.candidate / "staging"
        self.output = self.candidate / "published"
        self.staging.mkdir()

        self._write_candidates()
        self._write_references()
        self._write_smoke()
        self.summary = _load_module(
            self.summarizer_path, f"_v4_completion_fixture_{id(self)}"
        )
        report = self.summary.build_report(
            self.candidate,
            self.formal,
            self.smoke,
            self.training_lock,
        )
        _write_json(
            self.staging / self.summary.JSON_OUTPUT_NAME, report
        )
        (
            self.staging / self.summary.MARKDOWN_OUTPUT_NAME
        ).write_text(self.summary.render_markdown(report), encoding="utf-8")

    def _write_candidates(self) -> None:
        candidate_filenames = (
            "protocol.json",
            "split.json",
            "summary.json",
            "metrics.jsonl",
            "last.pth.tar",
            "best.pth.tar",
            "best_miou.pth.tar",
            "pd_fa_sweep_best.pth.json",
            "pd_fa_sweep_best_miou.pth.json",
        )
        launch_root = self.candidate / "launch"
        log_root = self.candidate / "logs"
        launch_root.mkdir()
        log_root.mkdir()
        for variant in validator.VARIANTS:
            for seed in validator.SEEDS:
                run = (
                    self.candidate
                    / validator.DATASET
                    / variant
                    / f"seed_{seed}_{validator.RUN_TAG}"
                )
                run.mkdir(parents=True)
                for filename in candidate_filenames:
                    (run / filename).write_bytes(
                        f"{variant}:{seed}:{filename}\n".encode()
                    )
                gpu_uuid = validator.EXPECTED_GPU_ASSIGNMENTS[(variant, seed)]
                _write_json(
                    launch_root / f"{variant}_seed{seed}.json",
                    {
                        "schema": (
                            "sctransnet_tpd_clean_v4_screen800_2x5090_launch_v1"
                        ),
                        "variant": variant,
                        "seed": seed,
                        "candidate_family": (
                            "spd_anchored_tpd_clean_v4_single_logit_kcs"
                        ),
                        "gpu_uuid": gpu_uuid,
                        "gpu_name": "NVIDIA GeForce RTX 5090",
                        "run_directory": str(run.resolve()),
                        "training_data_sha256": "a" * 64,
                        "source_lock": str(self.training_lock.resolve()),
                        "source_lock_sha256": _sha256(self.training_lock),
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
                    f"TPDCLEANV4_2X_COMPLETE variant={variant} seed={seed} "
                    f"gpu_uuid={gpu_uuid} epochs=800\n",
                    encoding="utf-8",
                )

    def _write_references(self) -> None:
        for variant in validator.REFERENCE_VARIANTS:
            formal_run = (
                self.formal
                / validator.DATASET
                / variant
                / f"seed_42_{validator.REFERENCE_RUN_TAG}"
            )
            miou_run = (
                self.reference_miou
                / validator.DATASET
                / variant
                / f"seed_42_{validator.REFERENCE_RUN_TAG}"
            )
            formal_run.mkdir(parents=True)
            miou_run.mkdir(parents=True)
            for run, filenames in (
                (
                    formal_run,
                    ("best.pth.tar", "pd_fa_sweep_best.pth.json"),
                ),
                (
                    miou_run,
                    (
                        "best_miou.pth.tar",
                        "pd_fa_sweep_best_miou.pth.json",
                    ),
                ),
            ):
                for filename in filenames:
                    (run / filename).write_bytes(
                        f"{variant}:{filename}\n".encode()
                    )

    def _write_smoke(self) -> None:
        for filename in (
            "SMOKE_REPORTS.sha256",
            "cpu_all.json",
            "gpu2_full.json",
            "gpu3_capacity.json",
        ):
            (self.smoke / filename).write_text(
                f"smoke fixture: {filename}\n", encoding="utf-8"
            )

    def kwargs(self) -> dict[str, Path]:
        return {
            "candidate_root": self.candidate,
            "formal_root": self.formal,
            "reference_miou_root": self.reference_miou,
            "smoke_root": self.smoke,
            "repo": self.repo,
            "summarizer_path": self.summarizer_path,
            "postprocess_lock_path": self.postprocess_lock,
        }


def test_exact_62_input_contract() -> None:
    assert validator.EXPECTED_INPUT_COUNTS == {
        "candidate_run_files": 36,
        "candidate_launch_manifests": 4,
        "candidate_worker_logs": 4,
        "frozen_reference_checkpoints": 4,
        "frozen_reference_sweeps": 4,
        "canonical_summarizers": 1,
        "source_locks": 5,
        "smoke_files": 4,
        "total_files": 62,
    }
    assert len(validator.POSTPROCESS_SOURCE_SET) == 10
    assert (
        validator.DEFAULT_POSTPROCESS_LOCK.name
        == "tpd_clean_v4_2x_postprocess_source_lock.json"
    )


def test_publish_verify_reuse_and_three_row_marker() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = CompletionFixture(Path(temporary))
        result = validator.publish_completion(
            staging_dir=fixture.staging,
            output_dir=fixture.output,
            **fixture.kwargs(),
        )
        assert result["status"] == "published"
        assert result["reused"] is False
        assert result["input_files"] == 62

        marker = fixture.output / validator.MARKER_NAME
        lines = marker.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert [line.split("  ", 1)[1] for line in lines] == [
            fixture.summary.JSON_OUTPUT_NAME,
            fixture.summary.MARKDOWN_OUTPUT_NAME,
            validator.MANIFEST_NAME,
        ]
        manifest = json.loads(
            (fixture.output / validator.MANIFEST_NAME).read_text(
                encoding="utf-8"
            )
        )
        assert manifest["input_counts"] == validator.EXPECTED_INPUT_COUNTS
        assert len(manifest["inputs"]) == 62

        verified = validator.verify_completion(
            output_dir=fixture.output, **fixture.kwargs()
        )
        assert verified["status"] == "verified"

        staged_json = fixture.staging / fixture.summary.JSON_OUTPUT_NAME
        staged_json.write_text("not used after marker\n", encoding="utf-8")
        reused = validator.publish_completion(
            staging_dir=fixture.staging,
            output_dir=fixture.output,
            **fixture.kwargs(),
        )
        assert reused["status"] == "reused"
        assert reused["reused"] is True


def test_fabricated_staged_report_is_rejected_without_marker() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = CompletionFixture(Path(temporary))
        report_path = fixture.staging / fixture.summary.JSON_OUTPUT_NAME
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["fabricated"] = True
        _write_json(report_path, report)
        (
            fixture.staging / fixture.summary.MARKDOWN_OUTPUT_NAME
        ).write_text(
            fixture.summary.render_markdown(report), encoding="utf-8"
        )

        with pytest.raises(validator.CompletionValidationError):
            validator.publish_completion(
                staging_dir=fixture.staging,
                output_dir=fixture.output,
                **fixture.kwargs(),
            )
        assert not (fixture.output / validator.MARKER_NAME).exists()


def test_missing_candidate_input_is_clear_and_never_marks_complete() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = CompletionFixture(Path(temporary))
        missing = (
            fixture.candidate
            / validator.DATASET
            / validator.VARIANTS[0]
            / f"seed_42_{validator.RUN_TAG}"
            / "best_miou.pth.tar"
        )
        missing.unlink()

        with pytest.raises(
            validator.CompletionValidationError, match="best_miou"
        ):
            validator.publish_completion(
                staging_dir=fixture.staging,
                output_dir=fixture.output,
                **fixture.kwargs(),
            )
        assert not (fixture.output / validator.MARKER_NAME).exists()


def test_verify_detects_bound_input_drift() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = CompletionFixture(Path(temporary))
        validator.publish_completion(
            staging_dir=fixture.staging,
            output_dir=fixture.output,
            **fixture.kwargs(),
        )
        checkpoint = (
            fixture.formal
            / validator.DATASET
            / "tpd"
            / f"seed_42_{validator.REFERENCE_RUN_TAG}"
            / "best.pth.tar"
        )
        checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")

        with pytest.raises(validator.CompletionValidationError):
            validator.verify_completion(
                output_dir=fixture.output, **fixture.kwargs()
            )


def test_linked_input_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = CompletionFixture(Path(temporary))
        target = fixture.smoke / "cpu_all.json"
        target.unlink()
        target.symlink_to(fixture.smoke / "gpu2_full.json")

        with pytest.raises(validator.CompletionValidationError, match="linked"):
            validator.publish_completion(
                staging_dir=fixture.staging,
                output_dir=fixture.output,
                **fixture.kwargs(),
            )
        assert not (fixture.output / validator.MARKER_NAME).exists()


def test_postprocess_bound_source_drift_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = CompletionFixture(Path(temporary))
        bound_source = (
            fixture.repo / "experiments/TPD_CLEAN_V4_PROTOCOL.md"
        )
        bound_source.write_text("changed after lock\n", encoding="utf-8")

        with pytest.raises(
            validator.CompletionValidationError, match="digest mismatch"
        ):
            validator.publish_completion(
                staging_dir=fixture.staging,
                output_dir=fixture.output,
                **fixture.kwargs(),
            )
        assert not (fixture.output / validator.MARKER_NAME).exists()


def test_marker_is_last_and_marker_write_failure_exposes_no_completion() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = CompletionFixture(Path(temporary))
        writes: list[str] = []
        original = validator._atomic_write_bytes

        def fail_marker(path: Path, content: bytes) -> None:
            writes.append(path.name)
            if path.name == validator.MARKER_NAME:
                raise OSError("injected marker failure")
            original(path, content)

        with mock.patch.object(
            validator, "_atomic_write_bytes", side_effect=fail_marker
        ):
            with pytest.raises(OSError, match="injected marker failure"):
                validator.publish_completion(
                    staging_dir=fixture.staging,
                    output_dir=fixture.output,
                    **fixture.kwargs(),
                )
        assert writes == [
            fixture.summary.JSON_OUTPUT_NAME,
            fixture.summary.MARKDOWN_OUTPUT_NAME,
            validator.MANIFEST_NAME,
            validator.MARKER_NAME,
        ]
        assert not (fixture.output / validator.MARKER_NAME).exists()
