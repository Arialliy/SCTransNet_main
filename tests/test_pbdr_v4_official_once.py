from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from experiments.pbdr_v4_candidate_pool import build_candidate_pool
from tests.test_pbdr_v4_candidate_pool import _artifacts
from experiments.pbdr_v4_official_once import (
    PBDRV4OfficialOnceError,
    build_joint_candidate_pool,
    execute_official_joint_once,
    execute_official_once,
)


class _CountingLoader:
    def __init__(self, batches: list[int]) -> None:
        self.batches = batches
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.batches)


class PBDRV4OfficialOnceTests(unittest.TestCase):
    def _pool(self, root: Path) -> dict[str, object]:
        return build_candidate_pool(
            dataset="NUDT-SIRST",
            role="best_pd",
            source_lock_sha256="a" * 64,
            split_projection_sha256="b" * 64,
            candidates=_artifacts(root),
        )

    def test_one_loader_iteration_and_one_forward_per_candidate_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self._pool(root)
            loader = _CountingLoader([2, 3])
            factory_calls = 0

            def factory():
                nonlocal factory_calls
                factory_calls += 1
                return loader

            bundle = execute_official_once(
                run_dir=root / "official",
                candidate_pool=pool,
                preflight=lambda: {"official_test_accessed": False},
                loader_factory=factory,
                consume_batch=lambda count: {
                    "sample_count": count,
                    "forward_counts": {name: count for name in pool["family_order"]},
                },
                finalize_metrics=lambda: {"ready": True},
            )
            self.assertEqual(factory_calls, 1)
            self.assertEqual(loader.iterations, 1)
            self.assertEqual(bundle["sample_count"], 5)
            self.assertTrue(all(value == 5 for value in bundle["forward_counts"].values()))

    def test_committed_bundle_recovery_never_constructs_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self._pool(root)
            run = root / "official"
            execute_official_once(
                run_dir=run,
                candidate_pool=pool,
                preflight=lambda: {"official_test_accessed": False},
                loader_factory=lambda: [1],
                consume_batch=lambda count: {
                    "sample_count": count,
                    "forward_counts": {name: count for name in pool["family_order"]},
                },
                finalize_metrics=lambda: {},
            )
            factory_calls = 0
            view_calls = 0

            def forbidden_factory():
                nonlocal factory_calls
                factory_calls += 1
                raise AssertionError("loader must not be constructed")

            def views(_bundle):
                nonlocal view_calls
                view_calls += 1

            execute_official_once(
                run_dir=run,
                candidate_pool=pool,
                preflight=lambda: (_ for _ in ()).throw(AssertionError("no preflight")),
                loader_factory=forbidden_factory,
                consume_batch=lambda _batch: {},
                finalize_metrics=lambda: {},
                materialize_views=views,
            )
            self.assertEqual(factory_calls, 0)
            self.assertEqual(view_calls, 1)

    def test_claim_then_exception_is_terminal_and_second_run_has_no_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self._pool(root)
            run = root / "official"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                execute_official_once(
                    run_dir=run,
                    candidate_pool=pool,
                    preflight=lambda: {"official_test_accessed": False},
                    loader_factory=lambda: [1],
                    consume_batch=lambda _batch: (_ for _ in ()).throw(RuntimeError("boom")),
                    finalize_metrics=lambda: {},
                )
            calls = 0

            def forbidden():
                nonlocal calls
                calls += 1
                return [1]

            with self.assertRaisesRegex(PBDRV4OfficialOnceError, "second pass"):
                execute_official_once(
                    run_dir=run,
                    candidate_pool=pool,
                    preflight=lambda: {"official_test_accessed": False},
                    loader_factory=forbidden,
                    consume_batch=lambda _batch: {},
                    finalize_metrics=lambda: {},
                )
            self.assertEqual(calls, 0)

    def test_forward_count_mismatch_consumes_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self._pool(root)
            with self.assertRaisesRegex(PBDRV4OfficialOnceError, "not forwarded"):
                execute_official_once(
                    run_dir=root / "official",
                    candidate_pool=pool,
                    preflight=lambda: {"official_test_accessed": False},
                    loader_factory=lambda: [2],
                    consume_batch=lambda count: {
                        "sample_count": count,
                        "forward_counts": {
                            name: (count - 1 if index == 4 else count)
                            for index, name in enumerate(pool["family_order"])
                        },
                    },
                    finalize_metrics=lambda: {},
                )

    def test_joint_preflight_must_bind_every_frozen_identity_before_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pools: dict[str, dict[str, object]] = {}
            for role in ("best_miou", "best_pd"):
                artifact_root = root / role
                artifact_root.mkdir()
                pools[role] = build_candidate_pool(
                    dataset="NUDT-SIRST",
                    role=role,
                    source_lock_sha256="a" * 64,
                    split_projection_sha256="b" * 64,
                    candidates=_artifacts(artifact_root),
                )
            joint = build_joint_candidate_pool(pools)
            good = {
                "dataset": joint["dataset"],
                "source_lock_sha256": joint["source_lock_sha256"],
                "split_projection_sha256": joint["split_projection_sha256"],
                "joint_candidate_pool_sha256": joint[
                    "joint_candidate_pool_sha256"
                ],
                "candidate_pool_sha256_by_role": joint[
                    "candidate_pool_sha256_by_role"
                ],
                "official_test_accessed": False,
            }
            cases = (
                ("dataset", "IRSTD-1K"),
                ("source_lock_sha256", "c" * 64),
                ("split_projection_sha256", "d" * 64),
                ("joint_candidate_pool_sha256", "e" * 64),
                (
                    "candidate_pool_sha256_by_role",
                    {"best_miou": "f" * 64, "best_pd": "0" * 64},
                ),
            )
            for index, (field, value) in enumerate(cases):
                preflight = dict(good)
                preflight[field] = value
                loader_calls = 0

                def loader():
                    nonlocal loader_calls
                    loader_calls += 1
                    return [1]

                run = root / f"joint-{index}"
                with self.subTest(field=field), self.assertRaisesRegex(
                    PBDRV4OfficialOnceError,
                    "preflight.*binding",
                ):
                    execute_official_joint_once(
                        run_dir=run,
                        joint_candidate_pool=joint,
                        preflight=lambda value=preflight: value,
                        loader_factory=loader,
                        consume_batch=lambda count: {
                            "sample_count": count,
                            "forward_counts": {
                                key: count for key in joint["execution_keys"]
                            },
                        },
                        finalize_metrics=lambda: {},
                    )
                self.assertEqual(loader_calls, 0)
                self.assertFalse((run / "official_claim.json").exists())


if __name__ == "__main__":
    unittest.main()
