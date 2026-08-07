from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from experiments.pbdr_v4_candidate_pool import (
    CandidateArtifact,
    PBDRV4CandidatePoolError,
    build_candidate_pool,
    canonical_sha256,
    file_sha256,
    load_candidate_pool,
    validate_candidate_pool,
    write_candidate_pool_exclusive,
)
from experiments.pbdr_v4_zero_margin_selector import FROZEN_TIE_ORDER


_KINDS = {
    "Original": "original_checkpoint",
    "Current": "current_checkpoint",
    "V3-calibrated": "v3_residual_calibration",
    "V4-Stage1": "v4_stage1_checkpoint",
    "V4-Stage2": "v4_stage2_checkpoint",
}


def _artifacts(root: Path) -> tuple[CandidateArtifact, ...]:
    result = []
    for index, family in enumerate(FROZEN_TIE_ORDER):
        path = (root / f"candidate-{index}.bin").resolve()
        path.write_bytes(family.encode("ascii"))
        result.append(
            CandidateArtifact(
                family=family,
                name=f"candidate-{index}",
                kind=_KINDS[family],
                artifact_path=str(path),
                artifact_sha256=file_sha256(path),
                state_sha256=f"{index + 1:x}" * 64,
                configuration_sha256=f"{index + 6:x}" * 64,
            )
        )
    return tuple(result)


class PBDRV4CandidatePoolTests(unittest.TestCase):
    def test_exact_five_family_pool_freezes_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_candidate_pool(
                dataset="NUDT-SIRST",
                role="best_pd",
                source_lock_sha256="a" * 64,
                split_projection_sha256="b" * 64,
                candidates=_artifacts(root),
            )
            self.assertEqual(payload["performance_acceptance_margin"], None)
            path = root / "pool.json"
            write_candidate_pool_exclusive(path, payload)
            self.assertEqual(load_candidate_pool(path), payload)
            with self.assertRaisesRegex(PBDRV4CandidatePoolError, "exists"):
                write_candidate_pool_exclusive(path, payload)

    def test_missing_extra_or_reordered_family_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidates = _artifacts(Path(directory))
            for changed in (candidates[:-1], (*candidates, candidates[-1]), tuple(reversed(candidates))):
                with self.assertRaisesRegex(PBDRV4CandidatePoolError, "families/order"):
                    build_candidate_pool(
                        dataset="NUDT-SIRST",
                        role="best_pd",
                        source_lock_sha256="a" * 64,
                        split_projection_sha256="b" * 64,
                        candidates=changed,
                    )

    def test_changed_candidate_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = _artifacts(root)
            payload = build_candidate_pool(
                dataset="NUDT-SIRST",
                role="best_pd",
                source_lock_sha256="a" * 64,
                split_projection_sha256="b" * 64,
                candidates=candidates,
            )
            Path(candidates[2].artifact_path).write_bytes(b"changed")
            with self.assertRaisesRegex(PBDRV4CandidatePoolError, "SHA differs"):
                validate_candidate_pool(payload)

    def test_family_kind_and_manifest_hash_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = list(_artifacts(root))
            candidates[0] = replace(candidates[0], kind="current_checkpoint")
            with self.assertRaisesRegex(PBDRV4CandidatePoolError, "family/kind"):
                build_candidate_pool(
                    dataset="NUDT-SIRST",
                    role="best_pd",
                    source_lock_sha256="a" * 64,
                    split_projection_sha256="b" * 64,
                    candidates=candidates,
                )

    def test_rehashed_bad_pool_semantics_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = build_candidate_pool(
                dataset="NUDT-SIRST",
                role="best_pd",
                source_lock_sha256="a" * 64,
                split_projection_sha256="b" * 64,
                candidates=_artifacts(root),
            )
            cases = (
                ("dataset", "other"),
                ("role", "other"),
                ("fixed_probability_rule", "greater_than_or_equal_0.5"),
                ("performance_acceptance_margin", 0.001),
                ("source_lock_sha256", "not-a-sha"),
                ("split_projection_sha256", "not-a-sha"),
            )
            for field, value in cases:
                tampered = dict(payload)
                tampered[field] = value
                unsigned = dict(tampered)
                unsigned.pop("candidate_pool_sha256", None)
                tampered["candidate_pool_sha256"] = canonical_sha256(unsigned)
                with self.subTest(field=field), self.assertRaises(
                    PBDRV4CandidatePoolError
                ):
                    validate_candidate_pool(tampered)


if __name__ == "__main__":
    unittest.main()
