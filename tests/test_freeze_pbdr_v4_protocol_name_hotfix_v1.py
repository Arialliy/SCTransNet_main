from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from experiments import freeze_pbdr_v4_protocol_name_hotfix_v1 as subject
from experiments import pbdr_v4_candidate_pool as pool_io
from experiments.pbdr_v4_zero_margin_selector import FROZEN_TIE_ORDER


KINDS = {
    "Original": "original_checkpoint",
    "Current": "current_checkpoint",
    "V3-calibrated": "v3_residual_calibration",
    "V4-Stage1": "v4_stage1_checkpoint",
    "V4-Stage2": "v4_stage2_checkpoint",
}


class FreezeNameHotfixTests(unittest.TestCase):
    def test_patch_is_exactly_one_literal_and_has_frozen_sha(self) -> None:
        locked = subject.LOCKED_FREEZER_PATH.read_text(encoding="utf-8")
        self.assertEqual(locked.count(subject.OLD_LITERAL), 1)
        patched = subject._derive_patched_freezer_source()
        self.assertNotIn(subject.OLD_LITERAL, patched)
        self.assertEqual(patched.count(subject.NEW_LITERAL), 1)
        self.assertEqual(
            hashlib.sha256(patched.encode("utf-8")).hexdigest(),
            subject.PATCHED_FREEZER_SHA256,
        )
        compile(patched, str(subject.LOCKED_FREEZER_PATH), "exec")

    def test_formal_amendment_is_deterministic_without_environment_check(self) -> None:
        source_lock = (
            subject.REPO_ROOT / "results/pbdr_v4_v1/protocol/source_lock.json"
        )
        first = subject.build_amendment_manifest(
            source_lock_path=source_lock,
            check_environment=False,
        )
        second = subject.build_amendment_manifest(
            source_lock_path=source_lock,
            check_environment=False,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["scope"], "candidate_pool_freeze_only")
        self.assertIsNone(first["invariants"]["performance_acceptance_margin"])
        self.assertFalse(first["official_test_accessed"])
        unsigned = dict(first)
        declared = unsigned.pop("amendment_sha256")
        self.assertEqual(declared, subject._canonical_sha256(unsigned))

    def test_candidate_pool_self_hash_covers_amendment_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            candidates = []
            for index, family in enumerate(FROZEN_TIE_ORDER):
                artifact = root / f"candidate-{index}.bin"
                artifact.write_bytes(family.encode("utf-8"))
                candidates.append(
                    pool_io.CandidateArtifact(
                        family=family,
                        name=f"NUAA-SIRST/best_miou/{family}",
                        kind=KINDS[family],
                        artifact_path=str(artifact),
                        artifact_sha256=pool_io.file_sha256(artifact),
                        state_sha256=f"{index + 1:x}" * 64,
                        configuration_sha256=f"{index + 6:x}" * 64,
                    )
                )
            binding = {
                "schema": subject.AMENDMENT_SCHEMA,
                "amendment_id": subject.AMENDMENT_ID,
                "scope": "candidate_pool_freeze_only",
                "amendment_sha256": "a" * 64,
                "official_test_accessed": False,
            }
            proxy = subject._PoolModuleProxy(pool_io, binding)
            payload = proxy.build_candidate_pool(
                dataset="NUAA-SIRST",
                role="best_miou",
                source_lock_sha256="b" * 64,
                split_projection_sha256="c" * 64,
                candidates=tuple(candidates),
            )
            self.assertEqual(payload["protocol_amendment_binding"], binding)
            self.assertIsNone(payload["performance_acceptance_margin"])
            declared = payload["candidate_pool_sha256"]
            unsigned = dict(payload)
            unsigned.pop("candidate_pool_sha256")
            self.assertEqual(declared, pool_io.canonical_sha256(unsigned))
            self.assertEqual(pool_io.validate_candidate_pool(payload), payload)


if __name__ == "__main__":
    unittest.main()
