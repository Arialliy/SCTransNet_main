from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from experiments import validate_tpd_clean_v3_resume_completion as completion


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class ResumeCompletionFixture:
    def __init__(self, root: Path, *, segment_starts: tuple[int, ...]) -> None:
        self.repo = root / "repo"
        self.candidate = root / "candidate"
        self.repo.joinpath("experiments").mkdir(parents=True)
        self.candidate.mkdir(parents=True)
        self.job = completion.JobSpec(
            "tpd_clean_v3_full", 42, 2, 0, 3, "fixture-full-s42"
        )
        if not segment_starts or segment_starts[0] != self.job.boundary_epoch:
            raise ValueError("fixture segments must start at the fixed boundary")
        self.run_dir = completion._run_dir(self.candidate, self.job)
        self.run_dir.mkdir(parents=True)
        self.resume_root = completion._resume_root(self.candidate)
        self.training_lock = self.repo / "training-lock.json"
        self.resume_lock = self.repo / "resume-lock.json"
        _write_json(self.training_lock, {"fixture": "training"})
        _write_json(self.resume_lock, {"fixture": "resume"})
        self.engine = self.repo / "experiments/resume_tpd_clean_v3.py"
        self.engine.write_text("# fixture resume engine\n", encoding="utf-8")

        _write_json(self.run_dir / "protocol.json", {"fixture": "protocol"})
        _write_json(self.run_dir / "split.json", {"fixture": "split"})
        self.boundary_rows = [self._base_event(1), self._base_event(2)]
        _write_jsonl(self.run_dir / "metrics.jsonl", self.boundary_rows)
        torch.save(
            {
                "epoch": self.job.boundary_epoch,
                "variant": self.job.variant,
                "seed": self.job.seed,
            },
            self.run_dir / "last.pth.tar",
        )
        torch.save({"fixture": "best"}, self.run_dir / "best.pth.tar")
        torch.save(
            {"fixture": "best_miou"}, self.run_dir / "best_miou.pth.tar"
        )

        self.original_launch_path = (
            self.candidate
            / "launch"
            / f"{self.job.variant}_seed{self.job.seed}.json"
        )
        self.original_log_path = (
            self.candidate
            / "logs"
            / f"{self.job.variant}_seed{self.job.seed}.log"
        )
        _write_json(
            self.original_launch_path,
            {
                "schema": completion.ORIGINAL_LAUNCH_SCHEMA,
                "variant": self.job.variant,
                "seed": self.job.seed,
                "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
                "gpu_uuid": self.job.original_gpu_uuid,
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "run_directory": str(self.run_dir),
                "source_lock": str(self.training_lock),
                "source_lock_sha256": _sha256(self.training_lock),
                "training_data_sha256": "1" * 64,
            },
        )
        self.original_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.original_log_path.write_text(
            "TPDCLEANV3_START "
            f"variant={self.job.variant} seed={self.job.seed} "
            f"gpu_uuid={self.job.original_gpu_uuid} "
            f"run_dir={self.run_dir}\n",
            encoding="utf-8",
        )
        self._create_boundary()
        self._create_resume_launch_and_log()
        self._create_segments_and_final_artifacts(segment_starts)

    def _base_event(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "variant": self.job.variant,
            "train_loss": 1.0 / epoch,
            "learning_rate": 0.001,
            "epoch_seconds": 0.01,
        }

    def _create_boundary(self) -> None:
        boundary_dir = completion._boundary_dir(self.candidate, self.job)
        boundary_dir.mkdir(parents=True)
        sources = {
            "metrics.jsonl": self.run_dir / "metrics.jsonl",
            "last.pth.tar": self.run_dir / "last.pth.tar",
            "best.pth.tar": self.run_dir / "best.pth.tar",
            "best_miou.pth.tar": self.run_dir / "best_miou.pth.tar",
            "protocol.json": self.run_dir / "protocol.json",
            "split.json": self.run_dir / "split.json",
            "original_launch_manifest.json": self.original_launch_path,
            "original_worker.log": self.original_log_path,
        }
        artifacts: dict[str, dict[str, Any]] = {}
        for name, source in sources.items():
            snapshot = boundary_dir / name
            shutil.copyfile(source, snapshot)
            digest = _sha256(snapshot)
            artifacts[name] = {
                "source": str(source),
                "source_sha256": digest,
                "snapshot_sha256": digest,
                "size_bytes": snapshot.stat().st_size,
            }
        self.boundary_path = boundary_dir / "boundary.json"
        _write_json(
            self.boundary_path,
            {
                "schema": completion.BOUNDARY_SCHEMA,
                "variant": self.job.variant,
                "seed": self.job.seed,
                "boundary_epoch": self.job.boundary_epoch,
                "run_directory": str(self.run_dir),
                "immutable_no_overwrite": True,
                "artifacts": artifacts,
            },
        )

    def _create_resume_launch_and_log(self) -> None:
        self.resume_manifest_path = (
            self.resume_root
            / "manifests"
            / f"{self.job.variant}_seed{self.job.seed}.json"
        )
        self.resume_log_path = (
            self.resume_root
            / "logs"
            / f"{self.job.variant}_seed{self.job.seed}.log"
        )
        _write_json(
            self.resume_manifest_path,
            {
                "schema": completion.RESUME_LAUNCH_SCHEMA,
                "variant": self.job.variant,
                "seed": self.job.seed,
                "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
                "run_directory": str(self.run_dir),
                "run_tag": completion.RUN_TAG,
                "boundary_epoch": self.job.boundary_epoch,
                "target_epoch": completion.TARGET_EPOCH,
                "boundary_directory": str(self.boundary_path.parent),
                "boundary_manifest_sha256": _sha256(self.boundary_path),
                "original_launch_manifest": str(self.original_launch_path),
                "original_launch_manifest_sha256": _sha256(
                    self.original_launch_path
                ),
                "original_gpu_uuid": self.job.original_gpu_uuid,
                "resume_gpu_uuid": self.job.resume_gpu_uuid,
                "resume_gpu_index": self.job.resume_gpu_index,
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "training_data_sha256": "1" * 64,
                "source_lock": str(self.resume_lock),
                "source_lock_sha256": _sha256(self.resume_lock),
                "original_source_lock": str(self.training_lock),
                "original_source_lock_sha256": _sha256(self.training_lock),
                "resource_snapshot": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "policy": {
                    "in_place_resume": True,
                    "fresh_run": False,
                    "original_results_preserved_by_boundary": True,
                    "immutable_resume_boundary": True,
                    "paired_variants": True,
                    "pre_registered_seeds": [42, 3407],
                    "allowed_gpu_indices": [2, 3],
                    "concurrent_jobs_per_gpu": 2,
                    "counterbalanced_mapping": True,
                    "efficiency_comparison_allowed": False,
                    "official_test_accessed": False,
                    "amp": False,
                    "cpu_replay_thread_cap": 1,
                },
            },
        )
        self.resume_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.resume_log_path.write_text(
            "TPDCLEANV3_RESUME_2X_START "
            f"variant={self.job.variant} seed={self.job.seed} "
            f"gpu_uuid={self.job.resume_gpu_uuid} "
            f"boundary_epoch={self.job.boundary_epoch} target_epoch=800 "
            "cpu_threads=1 "
            f"run_dir={self.run_dir}\n"
            "TPDCLEANV3_RESUME_2X_COMPLETE "
            f"variant={self.job.variant} seed={self.job.seed} "
            f"gpu_uuid={self.job.resume_gpu_uuid} "
            f"boundary_epoch={self.job.boundary_epoch} epochs=800\n",
            encoding="utf-8",
        )

    def _segment(self, index: int, resume_epoch: int) -> dict[str, Any]:
        source_sha = (
            _sha256(self.boundary_path.parent / "last.pth.tar")
            if index == 1
            else f"{index + 1:x}" * 64
        )
        return {
            "schema": completion.SEGMENT_SCHEMA,
            "segment_index": index,
            "created_at_utc": f"2026-07-26T00:00:0{index}+00:00",
            "resume_from_epoch": resume_epoch,
            "first_training_epoch": resume_epoch + 1,
            "target_epoch": completion.TARGET_EPOCH,
            "expected_resume_epoch": resume_epoch,
            "source_checkpoint": "last.pth.tar",
            "source_checkpoint_sha256": source_sha,
            "device": "cuda:0",
            "resume_gpu_uuid": self.job.resume_gpu_uuid,
            "process_restarted": True,
            "model_state_restored_strict": True,
            "adam_state_restored": True,
            "scaler_state_restored": True,
            "data_stream_replay": {
                "workers": 0,
                "seed": self.job.seed,
                "replayed_epochs": resume_epoch,
                "replayed_batches": resume_epoch * completion.TRAIN_BATCHES,
                "replayed_samples": resume_epoch * completion.TRAIN_COUNT,
                "elapsed_seconds": 0.1,
                "shuffle_generator_replayed": True,
                "crop_flip_python_random_replayed": True,
                "optimization_performed": False,
            },
            "continuity_claims": {
                "model_optimizer_scaler_restored": True,
                "shuffle_crop_flip_stream_replayed": True,
                "same_process_continuity": False,
                "cuda_bitwise_continuity": False,
            },
        }

    def _create_segments_and_final_artifacts(
        self, segment_starts: tuple[int, ...]
    ) -> None:
        self.segments = [
            self._segment(index, resume_epoch)
            for index, resume_epoch in enumerate(segment_starts, start=1)
        ]
        self.segments_path = self.run_dir / "resume_segments.jsonl"
        _write_jsonl(self.segments_path, self.segments)
        disclosure = {
            "process_restarted": True,
            "model_optimizer_scaler_restored": True,
            "data_shuffle_crop_flip_stream_replayed": True,
            "cuda_bitwise_continuity_claimed": False,
            "legacy_checkpoint_had_full_rng_state": False,
        }
        latest_sha = hashlib.sha256(
            completion._engine_json_bytes(self.segments[-1])
        ).hexdigest()
        provenance = {
            "schema": completion.PROVENANCE_SCHEMA,
            "engine_schema": completion.ENGINE_SCHEMA,
            "created_at_utc": "2026-07-26T00:00:00+00:00",
            "updated_at_utc": "2026-07-26T00:00:09+00:00",
            "run_directory": str(self.run_dir),
            "variant": self.job.variant,
            "dataset": completion.DATASET,
            "seed": self.job.seed,
            "protocol_sha256": _sha256(self.run_dir / "protocol.json"),
            "split_sha256": _sha256(self.run_dir / "split.json"),
            "engine_relative_path": "experiments/resume_tpd_clean_v3.py",
            "engine_sha256": _sha256(self.engine),
            "segments_file": "resume_segments.jsonl",
            "segments_sha256": _sha256(self.segments_path),
            "segment_count": len(self.segments),
            "latest_segment_index": len(self.segments),
            "latest_segment_sha256": latest_sha,
            "disclosure": disclosure,
        }
        self.provenance_path = self.run_dir / "resume_provenance.json"
        _write_json(self.provenance_path, provenance)
        provenance_sha = _sha256(self.provenance_path)

        final_rows = list(self.boundary_rows)
        for index, segment in enumerate(self.segments, start=1):
            start = int(segment["first_training_epoch"])
            end = (
                int(self.segments[index]["resume_from_epoch"])
                if index < len(self.segments)
                else completion.TARGET_EPOCH
            )
            segment_provenance_sha = (
                provenance_sha if index == len(self.segments) else f"{index:x}" * 64
            )
            for epoch in range(start, end + 1):
                final_rows.append(
                    {
                        **self._base_event(epoch),
                        "resumed": True,
                        "resume_segment_index": index,
                        "resume_provenance_sha256": segment_provenance_sha,
                    }
                )
        _write_jsonl(self.run_dir / "metrics.jsonl", final_rows)
        binding = {
            "resume_engine_schema": completion.ENGINE_SCHEMA,
            "resume_provenance_file": "resume_provenance.json",
            "resume_provenance_sha256": provenance_sha,
            "resume_segments_file": "resume_segments.jsonl",
            "resume_segments_sha256": _sha256(self.segments_path),
            "resume_segment_index": len(self.segments),
            "resume_segment_sha256": latest_sha,
        }
        _write_json(
            self.run_dir / "summary.json",
            {
                "status": "complete",
                "variant": self.job.variant,
                "seed": self.job.seed,
                "official_test_accessed": False,
                **binding,
                "resume_disclosure": disclosure,
            },
        )
        for name in ("last.pth.tar", "best.pth.tar", "best_miou.pth.tar"):
            torch.save(
                {**binding, "resume_disclosure": disclosure},
                self.run_dir / name,
            )

    def validate(self) -> dict[str, Any]:
        return completion.validate_resume_job(
            repo=self.repo,
            candidate_root=self.candidate,
            job=self.job,
            training_lock=self.training_lock,
            resume_lock=self.resume_lock,
        )


class ResumeCompletionValidatorTests(unittest.TestCase):
    def test_fixed_production_scope_and_input_count(self) -> None:
        observed = [
            (job.variant, job.seed, job.boundary_epoch, job.resume_gpu_index)
            for job in completion.JOBS
        ]
        self.assertEqual(
            observed,
            [
                ("tpd_clean_v3_full", 42, 279, 3),
                ("tpd_clean_v3_sal_capacity", 42, 331, 2),
                ("tpd_clean_v3_full", 3407, 323, 2),
                ("tpd_clean_v3_sal_capacity", 3407, 372, 3),
            ],
        )
        self.assertEqual(completion.EXPECTED_INPUT_COUNTS["total_files"], 116)
        self.assertEqual(completion.RUN_TAG, "screen800_pd_fp32_shared4x5090_v1")

    def test_accepts_single_segment_with_engine_compact_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeCompletionFixture(Path(temporary), segment_starts=(2,))
            result = fixture.validate()
        self.assertEqual(result["resume_segment_count"], 1)
        self.assertTrue(result["data_stream_replayed"])
        self.assertFalse(result["same_process_continuity"])
        self.assertFalse(result["cuda_bitwise_continuity"])

    def test_accepts_multiple_strictly_contiguous_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeCompletionFixture(
                Path(temporary), segment_starts=(2, 400)
            )
            result = fixture.validate()
        self.assertEqual(result["resume_segment_count"], 2)
        self.assertEqual(result["latest_resume_segment_index"], 2)

    def test_accepts_zero_progress_segment_before_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeCompletionFixture(
                Path(temporary), segment_starts=(2, 2)
            )
            result = fixture.validate()
        self.assertEqual(result["resume_segment_count"], 2)
        self.assertEqual(result["latest_resume_segment_index"], 2)

    def test_rejects_resume_gpu_mapping_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeCompletionFixture(Path(temporary), segment_starts=(2,))
            payload = json.loads(
                fixture.resume_manifest_path.read_text(encoding="utf-8")
            )
            payload["resume_gpu_index"] = 2
            _write_json(fixture.resume_manifest_path, payload)
            with self.assertRaises(completion.ResumeCompletionValidationError):
                fixture.validate()

    def test_rejects_boundary_snapshot_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeCompletionFixture(Path(temporary), segment_starts=(2,))
            snapshot = fixture.boundary_path.parent / "protocol.json"
            snapshot.write_text("drift\n", encoding="utf-8")
            with self.assertRaises(completion.ResumeCompletionValidationError):
                fixture.validate()

    def test_rejects_noncontiguous_final_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeCompletionFixture(Path(temporary), segment_starts=(2,))
            metrics_path = fixture.run_dir / "metrics.jsonl"
            rows = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[499]["epoch"] = 501
            _write_jsonl(metrics_path, rows)
            with self.assertRaises(completion.ResumeCompletionValidationError):
                fixture.validate()

    def test_rejects_false_data_stream_replay_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ResumeCompletionFixture(Path(temporary), segment_starts=(2,))
            payload = json.loads(
                fixture.provenance_path.read_text(encoding="utf-8")
            )
            payload["disclosure"][
                "data_shuffle_crop_flip_stream_replayed"
            ] = False
            _write_json(fixture.provenance_path, payload)
            with self.assertRaises(completion.ResumeCompletionValidationError):
                fixture.validate()


if __name__ == "__main__":
    unittest.main()
