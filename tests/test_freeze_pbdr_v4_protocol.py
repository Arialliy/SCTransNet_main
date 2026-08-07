from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from experiments import freeze_pbdr_v4_protocol as subject
from experiments import pbdr_v4_candidate_pool as pool_io
from experiments import pbdr_v4_split_authority as split_authority
from experiments.pbdr_v4_state_contract import state_semantic_sha256
from experiments.pbdr_v4_zero_margin_selector import FROZEN_TIE_ORDER


DATASET = "NUAA-SIRST"
ROLE = "best_miou"
KINDS = {
    "Original": "original_checkpoint",
    "Current": "current_checkpoint",
    "V3-calibrated": "v3_residual_calibration",
    "V4-Stage1": "v4_stage1_checkpoint",
    "V4-Stage2": "v4_stage2_checkpoint",
}


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": pool_io.file_sha256(path.resolve()),
    }


def _projection() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "synthetic",
        "datasets": {},
        "official_test_accessed": False,
    }
    payload["projection_sha256"] = split_authority.canonical_sha256(payload)
    return payload


def _pool_evidence(root: Path) -> subject.PoolEvidence:
    families = []
    for index, family in enumerate(FROZEN_TIE_ORDER):
        artifact = (root / f"artifact-{index}.bin").resolve()
        artifact.write_bytes(family.encode("utf-8"))
        families.append(
            subject.FamilyEvidence(
                family=family,
                name=f"{DATASET}/{ROLE}/{family}",
                kind=KINDS[family],
                artifact_path=artifact,
                state_sha256=f"{index + 1:x}" * 64,
                configuration={
                    "architecture": f"arch-{index}",
                    "family_parameter": index,
                },
            )
        )
    return subject.PoolEvidence(
        source_lock_sha256="a" * 64,
        split_projection_sha256="b" * 64,
        families=tuple(families),
    )


class FreezeSourceTests(unittest.TestCase):
    def test_json_projection_normalizes_historical_integer_keys(self) -> None:
        self.assertEqual(
            subject._json_ready(
                {"tail_z_thresholds": {4: 1.5, 3: 2.0, 2: 2.5}},
                label="historical metadata",
            ),
            {"tail_z_thresholds": {"2": 2.5, "3": 2.0, "4": 1.5}},
        )
        with self.assertRaisesRegex(
            subject.PBDRV4ProtocolFreezeError,
            "colliding JSON mapping keys",
        ):
            subject._json_ready({2: "integer", "2": "string"}, label="collision")

    def test_runtime_sources_merge_default_protocol_and_external_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory).resolve()
            internal = root / "runtime.py"
            external = Path(outside).resolve() / "external.py"
            internal.write_text("internal\n", encoding="utf-8")
            external.write_text("external\n", encoding="utf-8")
            paths, external_files = subject._merge_runtime_source_records(
                {
                    "registry": {
                        "inside": _file_record(internal),
                        "outside": _file_record(external),
                    }
                },
                repo_root=root,
            )
            self.assertEqual(tuple(paths), tuple(sorted(paths)))
            self.assertIn("runtime.py", paths)
            self.assertTrue(set(subject.PROTOCOL_SOURCE_RELATIVE_PATHS) <= set(paths))
            self.assertTrue(set(subject.source_lock_io.DEFAULT_SOURCE_RELATIVE_PATHS) <= set(paths))
            self.assertEqual(
                external_files["runtime_source::registry::outside"],
                external,
            )

    def test_runtime_source_record_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "runtime.py"
            path.write_text("old\n", encoding="utf-8")
            record = _file_record(path)
            path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(subject.PBDRV4ProtocolFreezeError, "byte count|SHA"):
                subject._merge_runtime_source_records(
                    {"registry": {"runtime.py": record}},
                    repo_root=root,
                )

    def test_persisted_projection_must_equal_live_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.json"
            projection = _projection()
            path.write_text(json.dumps(projection), encoding="utf-8")
            with mock.patch.object(subject.split_authority, "build_projection", return_value=projection):
                observed, observed_path = subject._validate_persisted_projection(path)
            self.assertEqual(observed, projection)
            self.assertEqual(observed_path, path.resolve())

            changed = dict(projection)
            changed["official_test_accessed"] = True
            path.write_text(json.dumps(changed), encoding="utf-8")
            with mock.patch.object(subject.split_authority, "build_projection", return_value=projection):
                with self.assertRaisesRegex(subject.PBDRV4ProtocolFreezeError, "differs"):
                    subject._validate_persisted_projection(path)

    def test_freeze_source_configures_determinism_first_and_locks_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection_path = (root / "projection.json").resolve()
            projection_path.write_text("{}\n", encoding="utf-8")
            authority = (root / "authority.json").resolve()
            authority.write_text("{}\n", encoding="utf-8")
            destination = root / "source-lock.json"
            events: list[str] = []

            def deterministic() -> None:
                events.append("determinism")

            def projection(_: Path):
                events.append("projection")
                return {"projection_sha256": "a" * 64}, projection_path

            def build(**kwargs):
                events.append("build")
                self.assertEqual(kwargs["external_files"]["split_projection"], projection_path)
                self.assertEqual(kwargs["external_files"]["authority"], authority)
                return {"synthetic": True}

            def write(path: Path, payload):
                events.append("write")
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            with (
                mock.patch.object(subject.training_core, "configure_determinism", side_effect=deterministic),
                mock.patch.object(subject, "_validate_persisted_projection", side_effect=projection),
                mock.patch.object(subject, "_merged_runtime_sources", return_value=(("source.py",), {})),
                mock.patch.object(subject, "_collect_authority_external_files", return_value={"authority": authority}),
                mock.patch.object(subject.source_lock_io, "build_source_lock", side_effect=build),
                mock.patch.object(subject.source_lock_io, "write_source_lock_exclusive", side_effect=write),
            ):
                self.assertEqual(
                    subject.freeze_source(
                        split_projection_path=projection_path,
                        output_path=destination,
                    ),
                    destination.resolve(),
                )
            self.assertEqual(events, ["determinism", "projection", "build", "write"])


class CalibratedArtifactTests(unittest.TestCase):
    def _inputs(self, root: Path):
        source = (root / "v3.pth.tar").resolve()
        source.write_bytes(b"v3-source")
        state = {
            "weight": torch.tensor([1.0, -2.0], dtype=torch.float32),
            "counter": torch.tensor(3, dtype=torch.int64),
        }
        run = SimpleNamespace(
            candidate_state=state,
            candidate_path=source,
            candidate_sha256=pool_io.file_sha256(source),
        )
        v3 = subject.V3Evidence(
            run=run,
            state_sha256="c" * 64,
            architecture_binding={"strict_load": True},
        )
        sweep_path = (root / "sweep.json").resolve()
        sweep_path.write_text("{}\n", encoding="utf-8")
        config = subject.residual_calibration.calibration_grid()[0]
        sweep = {
            "result_sha256": "d" * 64,
            "selected": {
                "grid_index": 0,
                "name": config.name,
                "config": config.as_dict(),
            },
            "cache_binding": {
                "commit_sha256": "e" * 64,
                "manifest_sha256": "f" * 64,
            },
        }
        cache = SimpleNamespace(manifest={"identity": {"identity_sha256": "1" * 64}})
        return v3, sweep, sweep_path, cache

    def _configuration_sha(self, v3, sweep) -> str:
        return subject._configuration_sha256(
            family="V3-calibrated",
            dataset=DATASET,
            role=ROLE,
            configuration={
                "state_sha256": state_semantic_sha256(v3.run.candidate_state),
                "calibration": sweep["selected"]["config"],
                "selected_on": "internal_validation",
            },
        )

    def test_v3_calibrated_artifact_is_semantic_and_strictly_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3, sweep, sweep_path, cache = self._inputs(root)
            configuration_sha = self._configuration_sha(v3, sweep)
            output = root / "calibrated.pth.tar"
            first, semantic_sha = subject._build_v3_calibrated_artifact(
                path=output,
                dataset=DATASET,
                role=ROLE,
                v3=v3,
                sweep=sweep,
                sweep_path=sweep_path,
                cache=cache,
                configuration_sha256=configuration_sha,
                source_lock_sha256="3" * 64,
                split_projection_sha256="4" * 64,
            )
            payload = subject.run_artifacts.load_torch_artifact(first)
            self.assertEqual(payload["schema"], subject.V3_CALIBRATED_SCHEMA)
            self.assertEqual(payload["selected_on"], "internal_validation")
            self.assertEqual(payload["calibration"], sweep["selected"]["config"])
            self.assertEqual(semantic_sha, state_semantic_sha256(payload["state_dict"]))
            self.assertNotEqual(semantic_sha, v3.state_sha256)

            second, second_sha = subject._build_v3_calibrated_artifact(
                path=output,
                dataset=DATASET,
                role=ROLE,
                v3=v3,
                sweep=sweep,
                sweep_path=sweep_path,
                cache=cache,
                configuration_sha256=configuration_sha,
                source_lock_sha256="3" * 64,
                split_projection_sha256="4" * 64,
            )
            self.assertEqual(second, first)
            self.assertEqual(second_sha, semantic_sha)

    def test_existing_v3_artifact_rejects_changed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3, sweep, sweep_path, cache = self._inputs(root)
            configuration_sha = self._configuration_sha(v3, sweep)
            output = root / "calibrated.pth.tar"
            subject._build_v3_calibrated_artifact(
                path=output,
                dataset=DATASET,
                role=ROLE,
                v3=v3,
                sweep=sweep,
                sweep_path=sweep_path,
                cache=cache,
                configuration_sha256=configuration_sha,
                source_lock_sha256="3" * 64,
                split_projection_sha256="4" * 64,
            )
            with self.assertRaisesRegex(subject.PBDRV4ProtocolFreezeError, "strict replay"):
                subject._build_v3_calibrated_artifact(
                    path=output,
                    dataset=DATASET,
                    role=ROLE,
                    v3=v3,
                    sweep=sweep,
                    sweep_path=sweep_path,
                    cache=cache,
                    configuration_sha256=configuration_sha,
                    source_lock_sha256="5" * 64,
                    split_projection_sha256="4" * 64,
                )

    def test_tampered_v3_artifact_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3, sweep, sweep_path, cache = self._inputs(root)
            configuration_sha = self._configuration_sha(v3, sweep)
            output = root / "calibrated.pth.tar"
            subject._build_v3_calibrated_artifact(
                path=output,
                dataset=DATASET,
                role=ROLE,
                v3=v3,
                sweep=sweep,
                sweep_path=sweep_path,
                cache=cache,
                configuration_sha256=configuration_sha,
                source_lock_sha256="3" * 64,
                split_projection_sha256="4" * 64,
            )
            payload = subject.run_artifacts.load_torch_artifact(output)
            payload["state_dict"]["weight"][0] += 1.0
            torch.save(payload, output)
            with self.assertRaisesRegex(subject.PBDRV4ProtocolFreezeError, "semantic state SHA"):
                subject._validate_v3_calibrated_artifact(
                    subject.run_artifacts.load_torch_artifact(output)
                )


class FreezePoolTests(unittest.TestCase):
    def test_family_evidence_rejects_wrong_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = _pool_evidence(Path(directory))
            with self.assertRaisesRegex(subject.PBDRV4ProtocolFreezeError, "order"):
                subject.PoolEvidence(
                    source_lock_sha256=evidence.source_lock_sha256,
                    split_projection_sha256=evidence.split_projection_sha256,
                    families=tuple(reversed(evidence.families)),
                )

    def test_freeze_pool_writes_exact_order_and_rejects_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _pool_evidence(root)
            output = root / "pool.json"
            arguments = dict(
                dataset=DATASET,
                role=ROLE,
                source_lock_path=root / "source.json",
                split_projection_path=root / "split.json",
                internal_cache_path=root / "cache",
                v3_sweep_path=root / "sweep.json",
                v3_calibrated_artifact_path=root / "calibrated.pth.tar",
                stage1_checkpoint_path=root / "stage1.pth.tar",
                stage2_checkpoint_path=root / "stage2.pth.tar",
                output_path=output,
            )
            with mock.patch.object(subject, "_collect_pool_evidence", return_value=evidence):
                destination = subject.freeze_pool(**arguments)
                payload = pool_io.load_candidate_pool(destination)
                self.assertEqual(
                    [candidate["family"] for candidate in payload["candidates"]],
                    list(FROZEN_TIE_ORDER),
                )
                self.assertEqual(
                    [candidate["state_sha256"] for candidate in payload["candidates"]],
                    [family.state_sha256 for family in evidence.families],
                )
                with self.assertRaisesRegex(pool_io.PBDRV4CandidatePoolError, "exists"):
                    subject.freeze_pool(**arguments)

    def test_candidate_byte_tampering_is_rejected_before_pool_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = _pool_evidence(root)
            candidates = tuple(
                item.candidate(dataset=DATASET, role=ROLE) for item in evidence.families
            )
            Path(candidates[2].artifact_path).write_bytes(b"tampered")
            with self.assertRaisesRegex(pool_io.PBDRV4CandidatePoolError, "SHA differs"):
                pool_io.build_candidate_pool(
                    dataset=DATASET,
                    role=ROLE,
                    source_lock_sha256=evidence.source_lock_sha256,
                    split_projection_sha256=evidence.split_projection_sha256,
                    candidates=candidates,
                )

    def test_cache_checkpoint_binding_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = (Path(directory) / "checkpoint.pth.tar").resolve()
            path.write_bytes(b"checkpoint")
            record = {
                "path": str(path),
                "state_sha256": "a" * 64,
            }
            binding = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "file_sha256": "b" * 64,
                "state_sha256": "a" * 64,
            }
            with self.assertRaisesRegex(subject.PBDRV4ProtocolFreezeError, "binding differs"):
                subject._assert_checkpoint_binding(record, binding, label="Current")

    def test_configuration_sha_changes_with_replayable_architecture(self) -> None:
        from experiments import evaluate_three_dataset_pbdr_v4_v1 as evaluator

        details = {"state_sha256": "a" * 64, "inference": "raw_final_logits"}
        self.assertEqual(
            subject._configuration_sha256(
                family="Current",
                dataset=DATASET,
                role=ROLE,
                configuration=details,
            ),
            evaluator.candidate_configuration_sha256(
                family="Current",
                dataset=DATASET,
                role=ROLE,
                details=details,
            ),
        )
        first = subject._configuration_sha256(
            family="Current",
            dataset=DATASET,
            role=ROLE,
            configuration={"architecture": {"depth": 4}},
        )
        second = subject._configuration_sha256(
            family="Current",
            dataset=DATASET,
            role=ROLE,
            configuration={"architecture": {"depth": 5}},
        )
        self.assertNotEqual(first, second)

    def test_frozen_pool_loads_through_default_candidate_factory(self) -> None:
        from experiments import evaluate_three_dataset_pbdr_v4_v1 as evaluator

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.mode = "test"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = (root / "original.pth.tar").resolve()
            current = (root / "current.pth.tar").resolve()
            original.write_bytes(b"original")
            current.write_bytes(b"current")
            original_state_sha = "1" * 64
            current_state_sha = "2" * 64

            v3_source = (root / "v3-source.pth.tar").resolve()
            v3_source.write_bytes(b"v3")
            v3_state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
            v3 = subject.V3Evidence(
                run=SimpleNamespace(
                    candidate_state=v3_state,
                    candidate_path=v3_source,
                    candidate_sha256=pool_io.file_sha256(v3_source),
                ),
                state_sha256="3" * 64,
                architecture_binding={"strict_load": True},
            )
            sweep_path = (root / "sweep.json").resolve()
            sweep_path.write_text("{}\n", encoding="utf-8")
            calibration = subject.residual_calibration.calibration_grid()[0]
            sweep = {
                "result_sha256": "4" * 64,
                "selected": {
                    "grid_index": 0,
                    "name": calibration.name,
                    "config": calibration.as_dict(),
                },
                "cache_binding": {
                    "commit_sha256": "5" * 64,
                    "manifest_sha256": "6" * 64,
                },
            }
            cache = SimpleNamespace(
                manifest={"identity": {"identity_sha256": "7" * 64}}
            )
            v3_details = {
                "state_sha256": state_semantic_sha256(v3_state),
                "calibration": calibration.as_dict(),
                "selected_on": "internal_validation",
            }
            calibrated, v3_semantic_sha = subject._build_v3_calibrated_artifact(
                path=root / "calibrated.pth.tar",
                dataset=DATASET,
                role=ROLE,
                v3=v3,
                sweep=sweep,
                sweep_path=sweep_path,
                cache=cache,
                configuration_sha256=subject._configuration_sha256(
                    family="V3-calibrated",
                    dataset=DATASET,
                    role=ROLE,
                    configuration=v3_details,
                ),
                source_lock_sha256="a" * 64,
                split_projection_sha256="b" * 64,
            )

            v4_records = []
            for family, stage, state_character in (
                ("V4-Stage1", "stage1", "8"),
                ("V4-Stage2", "stage2", "9"),
            ):
                path = (root / f"{stage}.pth.tar").resolve()
                payload = {
                    "dataset": DATASET,
                    "role": ROLE,
                    "parent_role": ROLE,
                    "stage": stage,
                    "state_sha256": state_character * 64,
                    "source_sha256": "a" * 64,
                    "split_sha256": "b" * 64,
                    "atlas_sha256": "c" * 64,
                    "initialization_sha256": "d" * 64,
                    "official_test_accessed": False,
                    "official_test_data_accessed": False,
                    "performance_acceptance_margin": None,
                    "smoke": False,
                }
                payload["candidate_manifest_sha256"] = subject._candidate_manifest_sha(
                    payload
                )
                torch.save(payload, path)
                details = {
                    "stage": stage,
                    "source_sha256": "a" * 64,
                    "split_sha256": "b" * 64,
                    "atlas_sha256": "c" * 64,
                    "initialization_sha256": "d" * 64,
                    "state_sha256": state_character * 64,
                }
                v4_records.append((family, stage, path, state_character * 64, details))

            evidence = subject.PoolEvidence(
                source_lock_sha256="a" * 64,
                split_projection_sha256="b" * 64,
                families=(
                    subject.FamilyEvidence(
                        "Original",
                        "Original",
                        "original_checkpoint",
                        original,
                        original_state_sha,
                        {"state_sha256": original_state_sha, "inference": "raw_final_logits"},
                    ),
                    subject.FamilyEvidence(
                        "Current",
                        "Current",
                        "current_checkpoint",
                        current,
                        current_state_sha,
                        {"state_sha256": current_state_sha, "inference": "raw_final_logits"},
                    ),
                    subject.FamilyEvidence(
                        "V3-calibrated",
                        "V3-calibrated",
                        "v3_residual_calibration",
                        calibrated,
                        v3_semantic_sha,
                        v3_details,
                    ),
                    subject.FamilyEvidence(
                        v4_records[0][0],
                        v4_records[0][0],
                        "v4_stage1_checkpoint",
                        v4_records[0][2],
                        v4_records[0][3],
                        v4_records[0][4],
                    ),
                    subject.FamilyEvidence(
                        v4_records[1][0],
                        v4_records[1][0],
                        "v4_stage2_checkpoint",
                        v4_records[1][2],
                        v4_records[1][3],
                        v4_records[1][4],
                    ),
                ),
            )
            pool_path = root / "pool.json"
            with mock.patch.object(subject, "_collect_pool_evidence", return_value=evidence):
                subject.freeze_pool(
                    dataset=DATASET,
                    role=ROLE,
                    source_lock_path=root / "source.json",
                    split_projection_path=root / "split.json",
                    internal_cache_path=root / "cache",
                    v3_sweep_path=sweep_path,
                    v3_calibrated_artifact_path=calibrated,
                    stage1_checkpoint_path=v4_records[0][2],
                    stage2_checkpoint_path=v4_records[1][2],
                    output_path=pool_path,
                )
            pool = pool_io.load_candidate_pool(pool_path)

            original_binding = {
                "path": str(original),
                "sha256": pool_io.file_sha256(original),
                "state_sha256": original_state_sha,
            }
            current_binding = {
                "path": str(current),
                "sha256": pool_io.file_sha256(current),
                "state_sha256": current_state_sha,
            }
            with (
                mock.patch.object(
                    evaluator.original_models,
                    "build_original_inference_model",
                    return_value=(TinyModel(), {"original_checkpoint": original_binding}),
                ),
                mock.patch.object(
                    evaluator.v4_models,
                    "build_frozen_current_reference_model",
                    return_value=(TinyModel(), {"parent_checkpoint": current_binding}),
                ),
                mock.patch.object(
                    evaluator.nuaa_v3_models,
                    "build_inference_model_from_candidate_state",
                    return_value=(TinyModel(), {"strict_load": True}),
                ),
                mock.patch.object(
                    evaluator.v4_models,
                    "build_candidate_inference_model",
                    return_value=(TinyModel(), {"strict_complete_payload": True}),
                ),
            ):
                runtimes = evaluator.default_candidate_factory(
                    candidate_pool=pool,
                    dataset=DATASET,
                    role=ROLE,
                    device=torch.device("cpu"),
                )
            self.assertEqual(tuple(runtimes), tuple(FROZEN_TIE_ORDER))


class CLITests(unittest.TestCase):
    def test_both_explicit_subcommands_parse(self) -> None:
        source = subject.parse_args(
            ["freeze-source", "--split-projection", "split.json", "--output", "lock.json"]
        )
        self.assertEqual(source.command, "freeze-source")
        pool = subject.parse_args(
            [
                "freeze-pool",
                "--dataset",
                DATASET,
                "--role",
                ROLE,
                "--source-lock",
                "source.json",
                "--split-projection",
                "split.json",
                "--internal-cache",
                "cache",
                "--v3-sweep",
                "sweep.json",
                "--v3-calibrated-artifact",
                "calibrated.pth.tar",
                "--stage1-checkpoint",
                "stage1.pth.tar",
                "--stage2-checkpoint",
                "stage2.pth.tar",
                "--output",
                "pool.json",
            ]
        )
        self.assertEqual(pool.command, "freeze-pool")
        self.assertEqual(pool.v3_calibrated_artifact, Path("calibrated.pth.tar"))


if __name__ == "__main__":
    unittest.main()
